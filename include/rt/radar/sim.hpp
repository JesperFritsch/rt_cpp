#pragma once
// the hardware side: the field simulation, sources, receivers and the phased array
#include <rt/fdtd_backend.hpp>
#include <rt/radar/types.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// per-cell feature bits, only used for drawing
constexpr uint8_t FEATURE_METAL = 0x1;
constexpr uint8_t FEATURE_RX    = 0x2;
constexpr uint8_t FEATURE_TX    = 0x4;

// source signal as a function of simulation time in seconds
using SourceFn = std::function<float(double)>;

struct Source {
  std::vector<Cell> cells;
  SourceFn fn;
};

// records ez at one cell every step; sample rate is 1 / dt
struct Receiver {
  Cell cell;
  std::vector<float> samples;
};

// all cells in the rectangle (x0, y0) - (x1, y1), inclusive; a line if x0 == x1 or y0 == y1
std::vector<Cell> line(std::size_t x0, std::size_t y0, std::size_t x1, std::size_t y1);

// all cells in a width x height grid where pred(x, y) is true
template <class Pred>
std::vector<Cell> cells_where(std::size_t width, std::size_t height, Pred pred) {
  std::vector<Cell> out;
  for (std::size_t x = 0; x < width; ++x)
    for (std::size_t y = 0; y < height; ++y)
      if (pred(x, y)) out.push_back({x, y});
  return out;
}

// filled disk of `radius` cells centered at (center_x, center_y)
std::vector<Cell> disk(std::size_t center_x, std::size_t center_y, std::size_t radius);

SourceFn sine(double freq_hz, double amplitude = 1.0);

// gaussian centered at t_center seconds, `width` seconds wide
SourceFn gaussian_pulse(double t_center, double width, double amplitude = 1.0);

// `cycles` periods of a sine, then silence
SourceFn burst(double freq_hz, int cycles, double amplitude = 1.0);

// linear array along x that fires steered pulses on demand
// `pulse_shape` is the signal with t = 0 at pulse start, only evaluated for t in [0, pulse_duration]
// theta is in radians from broadside; positive steers toward +x
class PhasedArray {
public:
  // copies share state, so fire on the copy returned to the caller reaches the copy kept by RadarSim
  PhasedArray(std::vector<Cell> cells, double cell_size, double length_cells, double theta);

  double theta() const { return state_->theta; }

  // element cells, in the order RadarSim::array_samples returns their samples
  std::vector<Cell> const& cells() const { return state_->cells; }

  // applies to pulses fired from now on; pulses already in flight keep their angle
  void set_theta(double theta) { state_->theta = theta; }

  // fire a pulse starting at sim time fire_time, steered at the current theta
  void fire(PulseRequest p_req);

  // drop pulses that have finished on every element
  void prune_pulses(double sim_time);

  // signal for the element `offset_cells` from the array's first element
  // the fns share state_, so fire and set_theta reach them after they are handed out
  SourceFn element_fn(double offset_cells) const;

private:
  struct Pulse {
    double fire_time;
    double sin_theta;  // latched when fired
    Waveform wave;
  };
  struct State {
    double cell_size;
    double length_cells;  // distance from the first to the last element
    double theta;
    std::vector<Pulse> pulses;  // fired pulses that may still be playing on some element
    std::vector<Cell> cells;
  };
  std::shared_ptr<State> state_;
};


class RadarSim {

private:
  std::unique_ptr<FdtdBackend> backend_;
  double cell_size_;  // meters per cell
  double dt_;         // seconds per step
  std::size_t step_ = 0;
  std::vector<Source> sources;
  std::vector<Receiver> receivers;
  std::vector<PhasedArray> arrays;
  std::size_t width_, height_;
  std::vector<uint8_t> cell_features;  // FEATURE_* bits per cell

  // the backend's source and probe cells are rebuilt before the next batch when this is set
  bool layout_dirty_ = true;
  std::vector<std::size_t> array_probe_offsets_;  // first probe of each array; arrays come first
  std::size_t receiver_probe_offset_ = 0;         // first probe of the receivers, after the arrays
  std::vector<float> source_values_;              // batch x source cells
  std::vector<float> probe_rows_;                 // batch x probe cells, from the latest batch
  std::size_t batch_size_ = 0;                    // steps in the latest batch

  void rebuild_layout();

public:

  // dt follows from cell_size and courant: dt = courant * cell_size / c
  // courant must be the one the backend was built with; the grid size is taken from the backend
  RadarSim(std::unique_ptr<FdtdBackend> backend, double cell_size = 0.01, float courant = 0.5);

  double cell_size() const { return cell_size_; }
  double dt() const { return dt_; }
  double time() const { return static_cast<double>(step_) * dt_; }

  void create_source(std::vector<Cell> cells, SourceFn fn);

  // `n_elements` single-cell elements along x, `spacing` cells apart, starting at `first`
  // keep spacing <= λ/2 to avoid grating lobes
  // the array is silent until PhasedArray::fire is called
  PhasedArray create_phased_array(Cell first, std::size_t n_elements, std::size_t spacing, bool x_plane = false, double theta = 0.0);

  // returns an id to read the samples back with
  std::size_t create_receiver(Cell cell);

  std::vector<float> const& samples(std::size_t receiver_id) const {
    return receivers[receiver_id].samples;
  }

  // mark cells as perfect electric conductor; waves reflect off them
  void add_metal(std::vector<Cell> const& cells);
  void remove_metal(std::vector<Cell> const& cells);

  std::vector<uint8_t> const& get_cell_features() const {
    return cell_features;
  }

  FieldView field() const { return backend_->ez(); }

  // the step the latest batch started at, and how many steps it ran
  std::size_t batch_start() const { return step_ - batch_size_; }
  std::size_t batch_size() const { return batch_size_; }

  // ez at each element of `array` after step `step_in_batch` of the latest batch,
  // in the order of that array's cells()
  float const* array_samples(std::size_t array, std::size_t step_in_batch) const {
    std::size_t row = step_in_batch * backend_->probe_count();
    return probe_rows_.data() + row + array_probe_offsets_[array];
  }

  // runs n steps; every source is evaluated at each step's time before the backend sees the batch,
  // so pulses fired between batches only take effect from the next batch on
  void step(std::size_t n);

};
