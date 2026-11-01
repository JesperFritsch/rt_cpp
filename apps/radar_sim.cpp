#include <rt/FDTDGrid.hpp>
#include <rt/ring_buffer_spsc.hpp>
#include <array>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <raylib.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <algorithm>
#include <string>
#include <thread>

// per-cell feature bits, only used for drawing
constexpr uint8_t FEATURE_METAL = 0x1;
constexpr uint8_t FEATURE_RX    = 0x2;
constexpr uint8_t FEATURE_TX    = 0x4;

constexpr double visual_gain = 5.0;
constexpr double pi = 3.14159265358979323846;
constexpr double speed_of_light = 299'792'458.0;  // m/s

// colors the ez texture: red for positive, blue for negative, gray for metal
constexpr const char* field_shader = R"(
#version 330
in vec2 fragTexCoord;
out vec4 finalColor;
uniform sampler2D texture0;     // ez, single float channel
uniform sampler2D features_tex; // FEATURE_* bits: 0x1 = metal, 0x2 = receiver, 0x4 = transmitter
uniform float gain;
void main() {
  int features = int(texture(features_tex, fragTexCoord).r * 255 + 0.5);
  if ((features & 0x1) != 0) {
    finalColor = vec4(0.5, 0.5, 0.5, 1.0);
    return;
  }
  if ((features & 0x2) != 0) {
    finalColor = vec4(0.9, 0.95, 0.0, 1.0);
    return;
  }
  if ((features & 0x4) != 0) {
    finalColor = vec4(0.02, 0.87, 0.98, 1.0);
    return;
  }
  float field = clamp(texture(texture0, fragTexCoord).r * gain, -1.0, 1.0);
  finalColor = vec4(max(field, 0.0), 0.0, max(-field, 0.0), 1.0);
}
)";

// source signal as a function of simulation time in seconds
using SourceFn = std::function<float(double)>;

struct Cell { std::size_t x, y; };

struct Waveform {
  double f_start;
  double f_end;
  double duration;
  double amplitude;
  double taper;  // share of the duration at each end that ramps in or out, 0 is a rectangular pulse
};

enum class PulseMode : uint8_t {
  Search,  // part of the scan over all angles
  Track,   // aimed at where a track is expected to be
};

struct PulseRequest {
  std::size_t step;  // step of the sample batch this request answers
  Waveform wave;
  double theta;
  double fire_time;
  PulseMode mode;
  std::size_t track_id;  // the track a Track pulse is aimed at, unused for Search
};

struct Source {
  std::vector<Cell> cells;
  SourceFn fn;
};

// records ez at one cell every step; sample rate is 1 / dt
struct Receiver {
  Cell cell;
  std::vector<float> samples;
};

// ez at every array element for one simulation step
template<std::size_t N_ELEMENTS>
struct SampleBatch {
  std::size_t step;  // simulation step the samples were taken at
  std::array<float, N_ELEMENTS> samples;
};

// an echo found in one pulse's window
struct Detection {
  double theta;       // radians, direction the echo came from
  double range;       // m from the array center
  float strength;     // envelope peak over the window's mean envelope
  double time;        // s, fire time of the pulse
  double beam_theta;  // radians, direction the pulse was steered
};

// the detections from one scan that belong to the same echo, merged
struct Plot {
  double theta;            // radians, strength-weighted mean of the merged detections
  double range;            // m from the array center, same weighting
  float strength;          // of the strongest merged detection
  double time;             // s, fire time of the latest merged detection
  std::size_t detections;  // how many were merged
};

// a target followed over several scans; positions are relative to the array center,
// `out` along broadside and `along` toward the last element
struct Track {
  std::size_t id;
  std::size_t report;      // set when sent to the sim; all tracks sent together share it
  double out, along;       // m
  double v_out, v_along;   // m per s of sim time
  double time;             // s, time of the plot the track was last corrected with
  std::size_t hits;        // plots the track has taken
  std::size_t misses;      // scans in a row without a plot
  bool confirmed;          // has taken enough plots to be trusted
};

struct Position { double x, y; };  // meters

// what the processor knows about the sim, all in meters and seconds
struct Geometry {
  double cell_size;                // m per cell
  double dt;                       // s per step
  double domain_width;             // m
  double domain_height;            // m
  double array_length;             // m, first to last element
  std::vector<Position> elements;  // same order as SampleBatch::samples
};

struct RadarConfig {
  // chosen
  // search and track pulses chirp over separate bands, so when they share a window the matched
  // filter for one barely responds to the other's echoes
  double search_f_start = 1.2e9;
  double search_f_end = 1.45e9;
  double track_f_start = 1.55e9;
  double track_f_end = 1.8e9;
  double pulse_duration = 10e-9;
  double amplitude = 2.0;
  double pulse_taper = 0.5;                   // see Waveform::taper; a rectangular pulse spreads into the
                                               // other band and shows up there as ghosts
  double max_scan_angle_r = 35.0 * pi / 180.0;  // radians from broadside
  double pri_margin = 1.0;                     // < 1 trims the conservative listen time
  double detection_factor = 8.0;               // detect where the envelope exceeds this times its mean
  double angle_step_r = 0.5 * pi / 180.0;      // trial angle step when estimating a detection's direction
  double merge_distance = 0.5;                 // m, detections in one scan closer than this are one plot
  double track_gate = 1.0;                     // m, a track only takes plots this close to its prediction
  double track_alpha = 0.5;                    // share of the position error a track corrects per plot
  double track_beta = 0.3;                     // share of the error, per time since the last plot, added to velocity
  std::size_t confirm_hits = 3;                // plots before a track is confirmed
  std::size_t max_misses = 3;                  // scans without a plot before a track is dropped
  std::size_t track_interval = 2;              // with confirmed tracks, every track_interval-th pulse is a
                                               // Track pulse; 2 alternates, 0 turns track pulses off
  bool simultaneous_tracks = true;             // fire a Track pulse together with every Search pulse instead,
                                               // so tracking does not slow the scan; track_interval is unused

  // derived
  double blind_range() const        { return speed_of_light * pulse_duration / 2.0; }
  double pulse_length_m() const     { return speed_of_light * pulse_duration; }
  Waveform search_waveform() const  { return {search_f_start, search_f_end, pulse_duration, amplitude, pulse_taper}; }
  Waveform track_waveform() const   { return {track_f_start, track_f_end, pulse_duration, amplitude, pulse_taper}; }

  // the coarser of the two bands'
  double range_resolution() const {
    double bandwidth = std::min(std::abs(search_f_end - search_f_start), std::abs(track_f_end - track_f_start));
    return speed_of_light / (2.0 * bandwidth);
  }

  // upper bound on the time from fire_time until the last echo has arrived:
  // out and back are each at most one domain diagonal, steering delays the
  // last element by up to array_length * sin(max_scan_angle_r), plus the pulse itself
  double listen_time(Geometry const& geometry) const {
    double diagonal = std::hypot(geometry.domain_width, geometry.domain_height);
    double steer_extra = geometry.array_length * std::sin(max_scan_angle_r);
    return (2.0 * diagonal + steer_extra) / speed_of_light + pulse_duration;
  }

  std::size_t pri_steps(Geometry const& geometry) const {
    return static_cast<std::size_t>(std::ceil(listen_time(geometry) * pri_margin / geometry.dt));
  }
};

// all cells in the rectangle (x0, y0) - (x1, y1), inclusive; a line if x0 == x1 or y0 == y1
std::vector<Cell> line(std::size_t x0, std::size_t y0, std::size_t x1, std::size_t y1) {
  std::vector<Cell> out;
  for (std::size_t x = x0; x <= x1; ++x)
    for (std::size_t y = y0; y <= y1; ++y)
      out.push_back({x, y});
  return out;
}

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
std::vector<Cell> disk(std::size_t center_x, std::size_t center_y, std::size_t radius) {
  std::vector<Cell> out;
  auto radius_squared = static_cast<double>(radius * radius);
  for (std::size_t x = center_x - radius; x <= center_x + radius; ++x)
    for (std::size_t y = center_y - radius; y <= center_y + radius; ++y) {
      double dx = static_cast<double>(x) - static_cast<double>(center_x);
      double dy = static_cast<double>(y) - static_cast<double>(center_y);
      if (dx * dx + dy * dy <= radius_squared) out.push_back({x, y});
    }
  return out;
}


SourceFn sine(double freq_hz, double amplitude = 1.0) {
  return [=](double t) { return amplitude * std::sin(2.0 * pi * freq_hz * t); };
}

// gaussian centered at t_center seconds, `width` seconds wide
SourceFn gaussian_pulse(double t_center, double width, double amplitude = 1.0) {
  return [=](double t) {
    double widths_from_center = (t - t_center) / width;
    return amplitude * std::exp(-widths_from_center * widths_from_center);
  };
}

// `cycles` periods of a sine, then silence
SourceFn burst(double freq_hz, int cycles, double amplitude = 1.0) {
  double duration = static_cast<double>(cycles) / freq_hz;
  return [=](double t) { return t < duration ? amplitude * std::sin(2.0 * pi * freq_hz * t) : 0.0; };
}

// linear chirp from f_start to f_end; the phase is the accumulated frequency,
// so it grows as f_start * t + 0.5 * chirp_rate * t^2 rather than f(t) * t
// `phase` shifts the carrier, pi / 2 gives the quadrature copy the matched filter uses
float wave_contrib(double time_in_pulse, Waveform const& wave, double phase = 0.0) {
  double chirp_rate = (wave.f_end - wave.f_start) / wave.duration;  // Hz per second
  double cycles = wave.f_start * time_in_pulse + 0.5 * chirp_rate * time_in_pulse * time_in_pulse;
  // a half cosine ramp over `taper` of the duration at each end (a Tukey window) instead of switching
  // on and off at once, which keeps the spectrum much closer to the chirp's own band
  double envelope = 1.0;
  double ramp = wave.taper * wave.duration;
  double from_edge = std::min(time_in_pulse, wave.duration - time_in_pulse);
  if (ramp > 0.0 && from_edge < ramp) envelope = 0.5 * (1.0 - std::cos(pi * from_edge / ramp));
  auto contrib = time_in_pulse < wave.duration ? envelope * wave.amplitude * std::sin(2.0 * pi * cycles + phase) : 0.0;
  return static_cast<float>(contrib);
}

// linear array along x that fires steered pulses on demand
// `pulse_shape` is the signal with t = 0 at pulse start, only evaluated for t in [0, pulse_duration]
// theta is in radians from broadside; positive steers toward +x
class PhasedArray {
public:
  // copies share state, so a copy kept by RadarSim records into the same samples
  PhasedArray(std::vector<Cell> cells, double cell_size, double length_cells, double theta)
    : state_(std::make_shared<State>()) {
    auto& state = *state_;
    state.cell_size = cell_size;
    state.length_cells = length_cells;
    state.theta = theta;
    state.samples.resize(cells.size());
    state.cells = std::move(cells);
  }

  double theta() const { return state_->theta; }

  // element cells, in the same order as samples()
  std::vector<Cell> const& cells() const { return state_->cells; }

  // ez at each element from the latest step; overwritten every step
  std::vector<float> const& samples() const { return state_->samples; }
  std::size_t sample_step() const { return state_->sample_step; }

  // copies the latest samples into out; N_ELEMENTS must equal the number of elements
  template <std::size_t N_ELEMENTS>
  void fill(SampleBatch<N_ELEMENTS>& out) const {
    assert(state_->samples.size() == N_ELEMENTS);
    out.step = state_->sample_step;
    std::copy_n(state_->samples.begin(), N_ELEMENTS, out.samples.begin());
  }

  // called by RadarSim::step for each element
  void record(std::size_t step, std::size_t element, float value) {
    state_->sample_step = step;
    state_->samples[element] = value;
  }

  // applies to pulses fired from now on; pulses already in flight keep their angle
  void set_theta(double theta) { state_->theta = theta; }

  // fire a pulse starting at sim time fire_time, steered at the current theta
  void fire(PulseRequest p_req) {
    auto& state = *state_;
    // drop pulses that have finished on every element
    state.pulses.push_back({p_req.fire_time, std::sin(p_req.theta), p_req.wave});
  }
  
  void prune_pulses(double sim_time) {
    double max_delay = state_->length_cells * state_->cell_size / speed_of_light;
    state_->pulses.erase(
      std::remove_if(state_->pulses.begin(), state_->pulses.end(),
                     [&](Pulse const& pulse) {
                       return sim_time > pulse.fire_time + pulse.wave.duration + max_delay;
                     }),
      state_->pulses.end());
  }

  

  // signal for the element `offset_cells` from the array's first element
  // the fns share state_, so fire and set_theta reach them after they are handed out
  SourceFn element_fn(double offset_cells) const {
    return [state = state_, offset_cells](double t) {
      float sum = 0.0f;
      for (auto const& pulse : state->pulses) {
        // distance in cells from the element that fires first, so every delay is >= 0:
        // the left end for positive theta, the right end for negative theta
        double cells_from_first_to_fire = pulse.sin_theta >= 0.0 ? offset_cells : offset_cells - state->length_cells;
        double delay = cells_from_first_to_fire * state->cell_size * pulse.sin_theta / speed_of_light;
        // how far this element is into its delayed copy of the pulse
        double time_into_pulse = t - pulse.fire_time - delay;
        if (time_into_pulse >= 0.0 && time_into_pulse <= pulse.wave.duration)
          sum += wave_contrib(time_into_pulse, pulse.wave);
      }
      return sum;
    };
  }

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
    std::size_t sample_step = 0;  // step the current samples were taken at
    std::vector<float> samples;
  };
  std::shared_ptr<State> state_;
};


class RadarSim {

private:
  FDTDGrid grid_;
  double cell_size_;  // meters per cell
  double dt_;         // seconds per step
  std::size_t step_ = 0;
  std::vector<Source> sources;
  std::vector<Receiver> receivers;
  std::vector<PhasedArray> arrays;
  std::size_t width_, height_;
  std::vector<uint8_t> cell_features;  // FEATURE_* bits per cell

public:

  // dt follows from cell_size and courant: dt = courant * cell_size / c
  RadarSim(
    std::size_t width,
    std::size_t height,
    std::size_t border,
    double cell_size = 0.01,
    float courant = 0.5,
    float damp = 1.0
  ) :
    grid_(width, height, border, courant, damp),
    cell_size_(cell_size),
    dt_(courant * cell_size / speed_of_light),
    width_(width),
    height_(height),
    cell_features(width_ * height_) // use the members to not get warning about unused height_
  {}

  double cell_size() const { return cell_size_; }
  double dt() const { return dt_; }
  double time() const { return static_cast<double>(step_) * dt_; }

  void create_source(std::vector<Cell> cells, SourceFn fn) {
    for (auto const& cell : cells) cell_features[cell.y * width_ + cell.x] |= FEATURE_TX;
    sources.push_back({std::move(cells), std::move(fn)});
  }

  // `n_elements` single-cell elements along x, `spacing` cells apart, starting at `first`
  // keep spacing <= λ/2 to avoid grating lobes
  // the array is silent until PhasedArray::fire is called
  PhasedArray create_phased_array(Cell first, std::size_t n_elements, std::size_t spacing, bool x_plane = false, double theta = 0.0) {
    auto length_cells = static_cast<double>((n_elements - 1) * spacing);
    std::vector<Cell> cells;
    for (std::size_t i = 0; i < n_elements; ++i) {
      if (x_plane) {
        cells.push_back({first.x + i * spacing, first.y});
      } else {
        cells.push_back({first.x, first.y + i * spacing});
      }
    }
    PhasedArray array(cells, cell_size_, length_cells, theta);
    for (auto const& cell : cells) {
        auto cell_offset = x_plane ? cell.x - first.x : cell.y - first.y;
        create_source({cell}, array.element_fn(static_cast<double>(cell_offset)));
    }
    arrays.push_back(array);
    return array;
  }

  // returns an id to read the samples back with
  std::size_t create_receiver(Cell cell) {
    receivers.push_back({cell, {}});
    cell_features[cell.y * width_ + cell.x] |= FEATURE_RX;
    return receivers.size() - 1;
  }

  std::vector<float> const& samples(std::size_t receiver_id) const {
    return receivers[receiver_id].samples;
  }

  // mark cells as perfect electric conductor; waves reflect off them
  void add_metal(std::vector<Cell> const& cells) {
    for (auto const& cell : cells) {
      cell_features[cell.y * width_ + cell.x] |= FEATURE_METAL;
      grid_.set_pec(cell.x, cell.y, true);
    }
  }

  void remove_metal(std::vector<Cell> const& cells) {
    for (auto const& cell : cells) {
      cell_features[cell.y * width_ + cell.x] &= static_cast<uint8_t>(~FEATURE_METAL);
      grid_.set_pec(cell.x, cell.y, false);
    }
  }

  std::vector<uint8_t> const& get_cell_features() const {
    return cell_features;
  }

  std::vector<float> const& grid() const {
    return grid_.get_ez();
  }

  void step() {
    double now = time();
    for (auto& source : sources) {
      auto value = source.fn(now);
      for (auto& cell : source.cells) {
        grid_.add_source(cell.x, cell.y, value);
      }
    }
    grid_.update();
    for (auto& receiver : receivers) {
      receiver.samples.push_back(grid_.get_ez(receiver.cell.x, receiver.cell.y));
    }
    for (auto& array : arrays) {
      auto const& cells = array.cells();
      for (std::size_t i = 0; i < cells.size(); ++i)
        array.record(step_, i, grid_.get_ez(cells[i].x, cells[i].y));
      array.prune_pulses(time());
    }
    ++step_;
  }

};

class PulseScheduler {
public:
  PulseScheduler(std::size_t pri_steps, double dt, Waveform search_wave, Waveform track_wave) :
    pri_steps_(pri_steps),
    dt_(dt),
    search_wave_(search_wave),
    track_wave_(track_wave)
  {

  }

  // appends the requests for `step` to out: none between pulses, otherwise a Search pulse, a Track
  // pulse, or both fired at the same step; a Search pulse comes first so its scan bookkeeping does too
  void on_step(std::size_t step, std::vector<PulseRequest>& out) {
    if (step < next_pulse_step_) return;
    next_pulse_step_ = step + pri_steps_;
    double fire_time = static_cast<double>(next_pulse_step_) * dt_;
    ++slot_;
    bool interleaved_track = track_interval_ > 0 && slot_ % track_interval_ == 0;
    bool with_track = !tracks_.empty() && (simultaneous_ || interleaved_track);
    bool with_search = simultaneous_ || !with_track;
    if (with_search) out.push_back(search_request(step, fire_time));
    if (with_track) out.push_back(track_request(step, fire_time));
  }

  void set_simultaneous_tracks(bool simultaneous) { simultaneous_ = simultaneous; }

  // the tracks to aim Track pulses at; only confirmed ones are kept
  void set_tracks(std::vector<Track> const& tracks) {
    tracks_.clear();
    for (auto const& track : tracks)
      if (track.confirmed) tracks_.push_back(track);
  }

  // 1 would make every pulse a Track pulse whenever there are tracks, and the scan would stop
  void set_track_interval(std::size_t interval) {
    assert(interval != 1);
    track_interval_ = interval;
  }

  void set_angles(double abs_angle_r, double ang_stride_r) {
    abs_theta_ = abs_angle_r;
    theta_stride_ = ang_stride_r;
  }

  // angles from -abs_theta_ up to, but not including, +abs_theta_, plus the Track pulses
  // between them when there are tracks to aim at
  std::size_t pulses_per_scan() const {
    auto search = static_cast<std::size_t>(std::lround(2.0 * abs_theta_ / theta_stride_));
    if (simultaneous_ || track_interval_ == 0) return search;
    return search * track_interval_ / (track_interval_ - 1);
  }

private:
  PulseRequest search_request(std::size_t step, double fire_time) {
    next_theta_ += theta_stride_;
    if (next_theta_ >= abs_theta_) {
      next_theta_ = -abs_theta_;
    }
    return {step, search_wave_, next_theta_, fire_time, PulseMode::Search, 0};
  }

  // the confirmed tracks take turns; aim where the track expects the target at fire_time
  PulseRequest track_request(std::size_t step, double fire_time) {
    auto const& track = tracks_[next_track_++ % tracks_.size()];
    double elapsed = fire_time - track.time;
    double out = track.out + track.v_out * elapsed;
    double along = track.along + track.v_along * elapsed;
    double theta = std::clamp(std::atan2(along, out), -abs_theta_, abs_theta_);
    return {step, track_wave_, theta, fire_time, PulseMode::Track, track.id};
  }

  double theta_stride_;
  double abs_theta_;
  std::size_t pri_steps_;
  double dt_;
  Waveform search_wave_;
  Waveform track_wave_;
  std::size_t next_pulse_step_ = 0;
  double next_theta_ = 0.0;
  std::size_t slot_ = 0;            // pulses scheduled so far
  std::size_t track_interval_ = 0;
  bool simultaneous_ = false;
  std::vector<Track> tracks_;       // confirmed tracks, as of the latest set_tracks
  std::size_t next_track_ = 0;      // round robin over tracks_
};

// alpha-beta tracker, updated once per scan with that scan's plots: each track is predicted to the
// plot's time, takes the closest plot within the gate and moves part of the way toward it;
// plots that no track takes start new tracks
class Tracker {
public:
  explicit Tracker(RadarConfig const& config) : config_(config) {
    tracks_.reserve(32);
    pairs_.reserve(256);
    plot_taken_.reserve(32);
    track_corrected_.reserve(32);
  }

  std::vector<Track> const& tracks() const { return tracks_; }

  void update(std::vector<Plot> const& plots) {
    // every track and plot pair within the gate, closest first; a track or plot is used at most once
    pairs_.clear();
    for (std::size_t t = 0; t < tracks_.size(); ++t) {
      for (std::size_t p = 0; p < plots.size(); ++p) {
        auto [out, along] = predict(tracks_[t], plots[p].time);
        double distance = std::hypot(out_of(plots[p]) - out, along_of(plots[p]) - along);
        if (distance <= config_.track_gate) pairs_.push_back({distance, t, p});
      }
    }
    std::sort(pairs_.begin(), pairs_.end(), [](Pair const& a, Pair const& b) { return a.distance < b.distance; });

    plot_taken_.assign(plots.size(), false);
    track_corrected_.assign(tracks_.size(), false);
    for (auto const& pair : pairs_) {
      if (plot_taken_[pair.plot] || track_corrected_[pair.track]) continue;
      correct(tracks_[pair.track], plots[pair.plot]);
      plot_taken_[pair.plot] = true;
      track_corrected_[pair.track] = true;
    }
    for (std::size_t t = 0; t < tracks_.size(); ++t)
      if (!track_corrected_[t]) ++tracks_[t].misses;
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](Track const& track) { return track.misses > config_.max_misses; }),
                  tracks_.end());

    for (std::size_t p = 0; p < plots.size(); ++p) {
      if (plot_taken_[p]) continue;
      tracks_.push_back({next_id_++, 0, out_of(plots[p]), along_of(plots[p]), 0.0, 0.0, plots[p].time,
                         1, 0, config_.confirm_hits <= 1});
    }
  }

  // corrects track `id` with the detection from a pulse aimed at it that is closest to its prediction,
  // if any is within the gate; returns whether the track was corrected
  // a pulse that finds nothing is no miss, misses are counted per scan by update
  bool update_track(std::size_t id, std::vector<Detection> const& detections) {
    auto track = std::find_if(tracks_.begin(), tracks_.end(), [&](Track const& t) { return t.id == id; });
    if (track == tracks_.end()) return false;  // dropped since the pulse was scheduled
    std::optional<Plot> closest;
    double closest_distance = config_.track_gate;
    for (auto const& detection : detections) {
      Plot plot{detection.theta, detection.range, detection.strength, detection.time, 1};
      auto [out, along] = predict(*track, plot.time);
      double distance = std::hypot(out_of(plot) - out, along_of(plot) - along);
      if (distance <= closest_distance) {
        closest_distance = distance;
        closest = plot;
      }
    }
    if (!closest) return false;
    correct(*track, *closest);
    return true;
  }

private:
  struct Pair {
    double distance;  // m, from the track's prediction to the plot
    std::size_t track;
    std::size_t plot;
  };

  static double out_of(Plot const& plot) { return plot.range * std::cos(plot.theta); }
  static double along_of(Plot const& plot) { return plot.range * std::sin(plot.theta); }

  static std::pair<double, double> predict(Track const& track, double time) {
    double elapsed = time - track.time;
    return {track.out + track.v_out * elapsed, track.along + track.v_along * elapsed};
  }

  void correct(Track& track, Plot const& plot) {
    double elapsed = plot.time - track.time;
    auto [out, along] = predict(track, plot.time);
    double error_out = out_of(plot) - out;
    double error_along = along_of(plot) - along;
    track.out = out + config_.track_alpha * error_out;
    track.along = along + config_.track_alpha * error_along;
    if (elapsed > 0.0) {
      track.v_out += config_.track_beta * error_out / elapsed;
      track.v_along += config_.track_beta * error_along / elapsed;
    }
    track.time = plot.time;
    ++track.hits;
    track.misses = 0;
    track.confirmed = track.hits >= config_.confirm_hits;
  }

  RadarConfig config_;
  std::vector<Track> tracks_;
  std::vector<Pair> pairs_;
  std::vector<bool> plot_taken_;
  std::vector<bool> track_corrected_;
  std::size_t next_id_ = 1;
};

// runs on the worker thread: takes sample batches from the sim, answers with pulse requests
// and reports what it finds as detections per pulse and plots per scan
template<std::size_t SAMPLE_RING_SIZE, std::size_t REQUEST_RING_SIZE, std::size_t N_ELEMENTS,
         std::size_t DETECTION_RING_SIZE, std::size_t PLOT_RING_SIZE, std::size_t TRACK_RING_SIZE>
class RadarProcessor {
public:
  RadarProcessor(rt::SPSCRing<PulseRequest, REQUEST_RING_SIZE>& pulse_requests,
                 rt::SPSCRing<SampleBatch<N_ELEMENTS>, SAMPLE_RING_SIZE>& sample_batches,
                 rt::SPSCRing<Detection, DETECTION_RING_SIZE>& detections_out,
                 rt::SPSCRing<Plot, PLOT_RING_SIZE>& plots_out,
                 rt::SPSCRing<Track, TRACK_RING_SIZE>& tracks_out,
                 RadarConfig config,
                 Geometry geometry)
    : pulse_requests_(pulse_requests),
      sample_batches_(sample_batches),
      detections_out_(detections_out),
      plots_out_(plots_out),
      tracks_out_(tracks_out),
      tracker_(config),
      config_(config),
      geometry_(std::move(geometry)),
      dt_(geometry_.dt),
      pri_steps_(config_.pri_steps(geometry_)),
      scheduler_(pri_steps_, dt_, config_.search_waveform(), config_.track_waveform())
  {
    assert(geometry_.elements.size() == N_ELEMENTS);
    // pulses are requested one PRI ahead, so at most two fire steps are in flight
    in_flight_.reserve(8);
    requests_.reserve(4);
    window_.resize(pri_steps_ * N_ELEMENTS);
    beam_.resize(pri_steps_);
    auto const& first = geometry_.elements.front();
    for (auto const& element : geometry_.elements)
      element_offsets_.push_back(std::hypot(element.x - first.x, element.y - first.y));
    auto reference_len = static_cast<std::size_t>(std::ceil(config_.pulse_duration / dt_));
    reference_.reserve(reference_len);
    reference_quadrature_.reserve(reference_len);
    envelope_.resize(pri_steps_);
    detections_.reserve(16);
    peak_lags_.reserve(16);
    angle_response_.reserve(128);
    scan_detections_.reserve(64);
    merged_.reserve(64);
    plots_.reserve(16);
    scheduler_.set_angles(config_.max_scan_angle_r, config_.max_scan_angle_r / 10.0);
    scheduler_.set_track_interval(config_.track_interval);
    scheduler_.set_simultaneous_tracks(config_.simultaneous_tracks);
  }

  std::size_t pri_steps() const { return pri_steps_; }

  // steps for one sweep over all scan angles, the longest it takes when Track pulses are interleaved
  std::size_t scan_steps() const { return pri_steps_ * scheduler_.pulses_per_scan(); }

  void run() {
    while ( auto batch = sample_batches_.pop_blocking() ) handle(*batch);
  }

  void handle(SampleBatch<N_ELEMENTS> const& batch) {
    requests_.clear();
    scheduler_.on_step(batch.step, requests_);
    for (auto const& request : requests_) {
      while (!pulse_requests_.push(request)) {}
      in_flight_.push_back(request);
    }
    collect(batch);
  }

private:
  std::size_t fire_step(PulseRequest const& req) const {
    return static_cast<std::size_t>(std::llround(req.fire_time / dt_));
  }

  // copies the batch into the window of the oldest in-flight pulses;
  // a window is the pri_steps steps from its fire step, so windows lie back to back,
  // and every pulse fired at that step shares it
  void collect(SampleBatch<N_ELEMENTS> const& batch) {
    if (in_flight_.empty()) return;
    std::size_t first_step = fire_step(in_flight_.front());
    if (batch.step < first_step) return;  // before the oldest pulses fired

    std::size_t row = batch.step - first_step;
    std::copy(batch.samples.begin(), batch.samples.end(), window_.begin() + static_cast<std::ptrdiff_t>(row * N_ELEMENTS));
    if (row + 1 < pri_steps_) return;

    // in_flight_ is in fire step order, so the pulses sharing this window come first
    auto shared_end = std::find_if(in_flight_.begin(), in_flight_.end(),
                                   [&](PulseRequest const& pulse) { return fire_step(pulse) != first_step; });
    // each pulse's own transmission is in the window, so every one is blind as long as the most steered
    double window_steer_time = 0.0;
    for (auto pulse = in_flight_.begin(); pulse != shared_end; ++pulse)
      window_steer_time = std::max(window_steer_time, steer_time(pulse->theta));
    for (auto pulse = in_flight_.begin(); pulse != shared_end; ++pulse)
      on_window(*pulse, window_steer_time);
    in_flight_.erase(in_flight_.begin(), shared_end);
  }

  // samples `wave` at the sim rate in phase and in quadrature, what the matched filter looks for;
  // rebuilt for every pulse, which costs little next to the matched filter itself
  void build_reference(Waveform const& wave) {
    reference_wave_ = wave;
    reference_.clear();
    reference_quadrature_.clear();
    auto reference_len = static_cast<std::size_t>(std::ceil(wave.duration / dt_));
    for (std::size_t i = 0; i < reference_len; ++i) {
      reference_.push_back(wave_contrib(static_cast<double>(i) * dt_, wave));
      reference_quadrature_.push_back(wave_contrib(static_cast<double>(i) * dt_, wave, pi / 2.0));
    }
  }

  // time from the first to the last element firing a pulse steered at theta
  double steer_time(double theta) const {
    return geometry_.array_length * std::abs(std::sin(theta)) / speed_of_light;
  }

  // delay and sum toward sin_theta: beam_[row] is what the array center receives at row,
  // so echoes from that direction add up in phase and others partly cancel
  // only rows in [first_row, end_row) are formed, the rest of beam_ is left as it was
  // returns the largest |shift| applied to an element, in steps
  std::size_t beamform(double sin_theta, std::size_t first_row = 0, std::size_t end_row = SIZE_MAX) {
    end_row = std::min(end_row, pri_steps_);
    std::fill(beam_.begin() + static_cast<std::ptrdiff_t>(first_row), beam_.begin() + static_cast<std::ptrdiff_t>(end_row), 0.0f);
    double center = geometry_.array_length / 2.0;
    auto rows = static_cast<std::ptrdiff_t>(pri_steps_);
    std::size_t max_shift = 0;
    for (std::size_t element = 0; element < N_ELEMENTS; ++element) {
      // an echo from theta reaches elements further along the array earlier than the center
      double lead = (element_offsets_[element] - center) * sin_theta / speed_of_light;
      auto shift = static_cast<std::ptrdiff_t>(std::llround(-lead / dt_));
      max_shift = std::max(max_shift, static_cast<std::size_t>(std::abs(shift)));
      for (auto row = static_cast<std::ptrdiff_t>(first_row); row < static_cast<std::ptrdiff_t>(end_row); ++row) {
        std::ptrdiff_t source_row = row + shift;
        if (source_row < 0 || source_row >= rows) continue;
        beam_[static_cast<std::size_t>(row)] += window_[static_cast<std::size_t>(source_row) * N_ELEMENTS + element];
      }
    }
    return max_shift;
  }

  // correlates the beam with the reference pulse and its quadrature copy; the magnitude of the two
  // is the envelope, large when an echo reaches the array center `lag` steps after the fire step,
  // without the carrier swing a single correlation has
  float matched_filter(std::size_t lag) const {
    float in_phase = 0.0f;
    float quadrature = 0.0f;
    for (std::size_t k = 0; k < reference_.size(); ++k) {
      in_phase += beam_[lag + k] * reference_[k];
      quadrature += beam_[lag + k] * reference_quadrature_[k];
    }
    return std::hypot(in_phase, quadrature);
  }

  // receive-beam scan: beamforms toward trial angles around beam_theta and returns the one where
  // the echo at `lag` is strongest; the transmit beam is the same for every trial angle, so it
  // scales the response but does not move its peak
  // overwrites the rows of beam_ the matched filter reads at `lag`
  double estimate_theta(double beam_theta, std::size_t lag) {
    // scan about one beam width each way; the beam widens as 1 / cos when steered off broadside
    double center_freq = (reference_wave_.f_start + reference_wave_.f_end) / 2.0;
    double beam_width = speed_of_light / center_freq / (geometry_.array_length * std::cos(beam_theta));
    auto half_steps = static_cast<std::ptrdiff_t>(std::ceil(beam_width / config_.angle_step_r));

    angle_response_.clear();
    std::size_t best = 0;
    for (std::ptrdiff_t i = -half_steps; i <= half_steps; ++i) {
      double trial = beam_theta + static_cast<double>(i) * config_.angle_step_r;
      float response = 0.0f;
      if (std::abs(trial) < pi / 2.0) {
        beamform(std::sin(trial), lag, lag + reference_.size());
        response = matched_filter(lag);
      }
      angle_response_.push_back(response);
      if (response > angle_response_[best]) best = angle_response_.size() - 1;
    }

    // a parabola through the best trial angle and its neighbours puts the peak between steps
    double offset = 0.0;
    if (best > 0 && best + 1 < angle_response_.size()) {
      double before = angle_response_[best - 1];
      double at = angle_response_[best];
      double after = angle_response_[best + 1];
      double curvature = before - 2.0 * at + after;
      if (curvature < 0.0) offset = 0.5 * (before - after) / curvature;
    }
    double best_step = static_cast<double>(best) - static_cast<double>(half_steps) + offset;
    return beam_theta + best_step * config_.angle_step_r;
  }

  // window_ holds pri_steps rows of N_ELEMENTS samples, row 0 is the fire step
  // fills detections_ with every echo past the blind range that stands out from the rest of the envelope
  // window_steer_time is the largest steer_time of the pulses sharing the window
  void on_window(PulseRequest const& pulse, double window_steer_time) {
    // the search pulses wrap from the largest angle back to the smallest, which ends a scan;
    // Track pulses are aimed anywhere, so only search pulses count
    bool search = pulse.mode == PulseMode::Search;
    if (search && pulse.theta < last_beam_theta_) finish_scan();
    if (search) last_beam_theta_ = pulse.theta;

    // look for this pulse's own waveform, so echoes of other pulses in the window barely register
    build_reference(pulse.wave);
    double sin_theta = std::sin(pulse.theta);
    std::size_t max_shift = beamform(sin_theta);

    // the steering delays make the array center transmit this many steps after the fire step
    double tx_offset_rows = steer_time(pulse.theta) / 2.0 / dt_;
    // our own pulses are on some element until the last one has fired them in full,
    // and the receive shifts can move that up to max_shift rows later in the beam
    // it also takes up to array_length / c to reach the far elements through the array
    double crossing_time = geometry_.array_length / speed_of_light;
    auto blind_rows = static_cast<std::size_t>(std::ceil((config_.pulse_duration + window_steer_time + crossing_time) / dt_)) + max_shift;

    std::size_t last_lag = pri_steps_ - reference_.size();
    if (blind_rows > last_lag) return;
    double envelope_sum = 0.0;
    float envelope_max = 0.0f;
    for (std::size_t lag = blind_rows; lag <= last_lag; ++lag) {
      envelope_[lag] = matched_filter(lag);
      envelope_sum += static_cast<double>(envelope_[lag]);
      envelope_max = std::max(envelope_max, envelope_[lag]);
    }
    double mean = envelope_sum / static_cast<double>(last_lag - blind_rows + 1);
    auto threshold = static_cast<float>(config_.detection_factor * mean);

    // each run of lags above the threshold is one echo, detected at its strongest lag
    detections_.clear();
    peak_lags_.clear();
    std::size_t peak_lag = 0;
    float peak = 0.0f;
    for (std::size_t lag = blind_rows; lag <= last_lag + 1; ++lag) {
      bool above = lag <= last_lag && envelope_[lag] > threshold;
      if (above && envelope_[lag] > peak) {
        peak = envelope_[lag];
        peak_lag = lag;
      }
      if (!above && peak > 0.0f) {
        // range from the array center: time from its transmit to the echo at it, there and back
        double range = speed_of_light * (static_cast<double>(peak_lag) - tx_offset_rows) * dt_ / 2.0;
        // theta is the beam direction until estimate_theta below refines it
        detections_.push_back({pulse.theta, range, static_cast<float>(static_cast<double>(peak) / mean),
                               pulse.fire_time, pulse.theta});
        peak_lags_.push_back(peak_lag);
        peak = 0.0f;
      }
    }
    // after the search above, since estimate_theta overwrites parts of beam_
    for (std::size_t i = 0; i < detections_.size(); ++i)
      detections_[i].theta = estimate_theta(pulse.theta, peak_lags_[i]);

    if (search)
      std::printf("pulse at step %zu, theta %+6.1f deg:", fire_step(pulse), pulse.theta * 180.0 / pi);
    else
      std::printf("pulse at step %zu, theta %+6.1f deg, track #%zu:", fire_step(pulse), pulse.theta * 180.0 / pi,
                  pulse.track_id);
    // with nothing detected, the strongest response shows how far below the threshold it is
    if (detections_.empty()) std::printf(" -  (max x%.1f)", static_cast<double>(envelope_max) / mean);
    for (auto const& detection : detections_)
      std::printf("  %.2f m at %+.1f deg (x%.1f)", detection.range, detection.theta * 180.0 / pi,
                  static_cast<double>(detection.strength));
    std::printf("\n");

    // detections are only drawn, so drop them rather than stall processing if the sim falls behind
    for (auto const& detection : detections_) (void)detections_out_.push(detection);

    // search detections wait for the scan to end; a Track pulse corrects its track right away,
    // which is what makes a track update more often than once per scan
    if (search) {
      scan_detections_.insert(scan_detections_.end(), detections_.begin(), detections_.end());
    } else if (tracker_.update_track(pulse.track_id, detections_)) {
      scheduler_.set_tracks(tracker_.tracks());
      send_tracks();
    }
  }

  // sends every track to the sim under a new report number, so it replaces the previous set
  void send_tracks() {
    ++report_;
    for (auto track : tracker_.tracks()) {
      track.report = report_;
      (void)tracks_out_.push(track);
    }
  }

  // merges the scan's detections into plots: the strongest detection not yet merged starts a plot
  // and takes every unmerged detection within merge_distance of it, so a target that several
  // pulses saw becomes one plot
  void finish_scan() {
    std::sort(scan_detections_.begin(), scan_detections_.end(),
              [](Detection const& a, Detection const& b) { return a.strength > b.strength; });
    merged_.assign(scan_detections_.size(), false);
    plots_.clear();

    // positions relative to the array center: out along broadside, along the array
    auto out_of = [](Detection const& detection) { return detection.range * std::cos(detection.theta); };
    auto along_of = [](Detection const& detection) { return detection.range * std::sin(detection.theta); };

    for (std::size_t seed = 0; seed < scan_detections_.size(); ++seed) {
      if (merged_[seed]) continue;
      double seed_out = out_of(scan_detections_[seed]);
      double seed_along = along_of(scan_detections_[seed]);
      double weight_sum = 0.0;
      double out_sum = 0.0;
      double along_sum = 0.0;
      double latest = 0.0;
      std::size_t count = 0;
      for (std::size_t i = seed; i < scan_detections_.size(); ++i) {
        if (merged_[i]) continue;
        auto const& detection = scan_detections_[i];
        double out = out_of(detection);
        double along = along_of(detection);
        if (std::hypot(out - seed_out, along - seed_along) > config_.merge_distance) continue;
        merged_[i] = true;
        auto weight = static_cast<double>(detection.strength);
        weight_sum += weight;
        out_sum += weight * out;
        along_sum += weight * along;
        latest = std::max(latest, detection.time);
        ++count;
      }
      double out = out_sum / weight_sum;
      double along = along_sum / weight_sum;
      plots_.push_back({std::atan2(along, out), std::hypot(out, along), scan_detections_[seed].strength, latest, count});
    }

    std::printf("scan done:");
    if (plots_.empty()) std::printf(" -");
    for (auto const& plot : plots_)
      std::printf("  %.2f m at %+.1f deg (%zu detections)", plot.range, plot.theta * 180.0 / pi, plot.detections);
    std::printf("\n");

    for (auto const& plot : plots_) (void)plots_out_.push(plot);
    scan_detections_.clear();

    tracker_.update(plots_);
    scheduler_.set_tracks(tracker_.tracks());
    std::printf("tracks:");
    if (tracker_.tracks().empty()) std::printf(" -");
    for (auto const& track : tracker_.tracks())
      std::printf("  #%zu %.2f m %+.1f deg%s", track.id, std::hypot(track.out, track.along),
                  std::atan2(track.along, track.out) * 180.0 / pi, track.confirmed ? "" : " (tentative)");
    std::printf("\n");
    send_tracks();
  }

  rt::SPSCRing<PulseRequest, REQUEST_RING_SIZE>& pulse_requests_;           // to the sim
  rt::SPSCRing<SampleBatch<N_ELEMENTS>, SAMPLE_RING_SIZE>& sample_batches_; // from the sim
  rt::SPSCRing<Detection, DETECTION_RING_SIZE>& detections_out_;            // to the sim, for drawing
  rt::SPSCRing<Plot, PLOT_RING_SIZE>& plots_out_;                           // to the sim, for drawing
  rt::SPSCRing<Track, TRACK_RING_SIZE>& tracks_out_;                        // to the sim, for drawing
  Tracker tracker_;
  std::size_t report_ = 0;                  // sets of tracks sent to the sim so far
  std::vector<Detection> scan_detections_;  // every detection since the scan started
  std::vector<bool> merged_;                // finish_scan's bookkeeping, same order as scan_detections_
  std::vector<Plot> plots_;                 // from the latest scan
  double last_beam_theta_ = -pi;            // lower than any scan angle, so the first pulse starts no scan
  std::vector<PulseRequest> in_flight_;  // requested pulses whose window is not complete, oldest first
  std::vector<PulseRequest> requests_;   // handle's scratch for the scheduler's requests
  std::vector<float> window_;            // pri_steps x N_ELEMENTS, time-major
  Waveform reference_wave_{};            // the waveform reference_ was built from
  std::vector<float> reference_;         // transmitted pulse, one sample per step
  std::vector<float> reference_quadrature_;  // reference_ with the carrier shifted 90 degrees
  std::vector<float> envelope_;          // matched filter magnitude per lag, pri_steps long
  std::vector<Detection> detections_;    // from the latest window
  std::vector<std::size_t> peak_lags_;   // envelope lag of each detection, same order
  std::vector<float> angle_response_;    // estimate_theta's matched filter output per trial angle
  std::vector<float> beam_;              // window_ beamformed toward the pulse's theta, pri_steps long
  std::vector<double> element_offsets_;  // m from the first element along the array
  RadarConfig config_;
  Geometry geometry_;
  double dt_;
  std::size_t pri_steps_;
  PulseScheduler scheduler_;
};

// prints a bar from a center line: left for negative, right for positive
// value is scaled so |value| == max_value fills half_width characters
void print_bar(float value, float max_value, int half_width) {
  float fraction = std::clamp(value / max_value, -1.0f, 1.0f);
  auto bar_length = static_cast<std::size_t>(std::lround(std::abs(fraction) * static_cast<float>(half_width)));
  auto side_width = static_cast<std::size_t>(half_width);

  std::string left(side_width, ' ');
  std::string right(side_width, ' ');
  if (fraction < 0)
    std::fill(left.end() - static_cast<std::ptrdiff_t>(bar_length), left.end(), '#');
  else
    std::fill(right.begin(), right.begin() + static_cast<std::ptrdiff_t>(bar_length), '#');

  std::printf("\r\033[2K%8.4f %s|%s", static_cast<double>(value), left.c_str(), right.c_str());
  std::fflush(stdout);
}


int main() {
  // λ = 20 cells at 1.5 GHz with 1 cm cells, so the grid is 50 x 35 wavelengths
  constexpr std::size_t grid_width = 1000;
  constexpr std::size_t grid_height = 700;
  constexpr std::size_t border = 60;  // sponge, 3λ thick
  // window has the grid's aspect ratio so cells stay square
  constexpr int window_width = 1400;
  constexpr int window_height = static_cast<int>(window_width * grid_height / grid_width);
  InitWindow(window_width, window_height, "radar sim");
  SetTargetFPS(512);
  constexpr int steps_per_frame = 15;
  constexpr double cell_size = 0.01;   // 1 cm per cell
  constexpr std::size_t sample_ring_size = 128;
  constexpr std::size_t request_ring_size = 128;
  constexpr std::size_t detection_ring_size = 128;
  constexpr std::size_t plot_ring_size = 32;
  constexpr std::size_t track_ring_size = 64;

  auto radar_sim = RadarSim(grid_width, grid_height, border, cell_size);
  auto const& ez_grid = radar_sim.grid();
  // raylib only reads from image data, so the const_cast is safe
  Image ez_img = {
    const_cast<float*>(ez_grid.data()),
    grid_width,
    grid_height,
    1,
    PIXELFORMAT_UNCOMPRESSED_R32
  };
  Texture2D ez_tex = LoadTextureFromImage(ez_img);

  // vertical array of 16 elements at λ/2 spacing, 2λ in from the sponge, facing +x
  constexpr std::size_t n_elements = 16;
  constexpr std::size_t spacing = 10;
  constexpr std::size_t array_start_x = border + 40;
  constexpr std::size_t array_start_y = grid_height / 2 - (n_elements - 1) * spacing / 2;
  // ground plane λ/4 behind the elements, extending 1λ past each end of the array
  constexpr std::size_t plane_distance = 5;
  constexpr std::size_t plane_margin = 20;
  radar_sim.add_metal(line(array_start_x - plane_distance - 1, array_start_y - plane_margin,
                           array_start_x - plane_distance, array_start_y + (n_elements - 1) * spacing + plane_margin));

  // targets, drag them with the left mouse button
  struct Target { std::size_t x, y, radius; };  // cells
  std::vector<Target> targets{
    {grid_width * 3 / 4, grid_height * 3 / 4, 30},
    {grid_width * 6 / 7, grid_height * 1 / 4, 30},
  };
  for (auto const& target : targets) radar_sim.add_metal(disk(target.x, target.y, target.radius));
  std::optional<std::size_t> dragged;  // index into targets

  auto array = radar_sim.create_phased_array(
    {array_start_x, array_start_y },
    n_elements, 
    spacing,
    false
  );

  rt::SPSCRing<PulseRequest, request_ring_size> pulse_requests;
  rt::SPSCRing<SampleBatch<n_elements>, sample_ring_size> sample_batches;
  rt::SPSCRing<Detection, detection_ring_size> detections;
  rt::SPSCRing<Plot, plot_ring_size> plots;
  rt::SPSCRing<Track, track_ring_size> tracks;


  // features are uploaded here and again whenever a target is moved
  auto const& features = radar_sim.get_cell_features();
  std::vector<unsigned char> feature_bytes(features.begin(), features.end());
  Image features_img = {
    feature_bytes.data(),
    grid_width,
    grid_height,
    1,
    PIXELFORMAT_UNCOMPRESSED_GRAYSCALE
  };
  Texture2D features_tex = LoadTextureFromImage(features_img);

  Shader shader = LoadShaderFromMemory(nullptr, field_shader);
  int features_loc = GetShaderLocation(shader, "features_tex");
  int gain_loc = GetShaderLocation(shader, "gain");
  float gain = static_cast<float>(visual_gain);
  SetShaderValue(shader, gain_loc, &gain, SHADER_UNIFORM_FLOAT);

  Geometry geometry{
    cell_size,
    radar_sim.dt(),
    static_cast<double>(grid_width) * cell_size,
    static_cast<double>(grid_height) * cell_size,
    static_cast<double>((n_elements - 1) * spacing) * cell_size,
    {}
  };
  for (auto const& cell : array.cells())
    geometry.elements.push_back({static_cast<double>(cell.x) * cell_size, static_cast<double>(cell.y) * cell_size});

  RadarConfig config;
  RadarProcessor processor(pulse_requests, sample_batches, detections, plots, tracks, config, geometry);
  std::printf("PRI %zu steps (%.1f ns), blind range %.2f m, range resolution %.2f m\n",
              processor.pri_steps(), static_cast<double>(processor.pri_steps()) * radar_sim.dt() * 1e9,
              config.blind_range(), config.range_resolution());
  auto worker_thread = std::thread([&]() { processor.run(); });

  // detections are kept for one scan and fade out over it
  double detection_lifetime = static_cast<double>(processor.scan_steps()) * radar_sim.dt();
  std::vector<Detection> shown_detections;
  // plots come once per scan, so the same lifetime shows each until the next one replaces it
  std::vector<Plot> shown_plots;
  auto drop_old = [&](auto& echoes, double now) {
    echoes.erase(std::remove_if(echoes.begin(), echoes.end(),
                                [&](auto const& echo) { return now - echo.time > detection_lifetime; }),
                 echoes.end());
  };

  // a detection or plot is `range` from the array center, theta from broadside toward the last element
  auto const& first_element = geometry.elements.front();
  auto const& last_element = geometry.elements.back();
  double axis_x = (last_element.x - first_element.x) / geometry.array_length;
  double axis_y = (last_element.y - first_element.y) / geometry.array_length;
  double broadside_x = axis_y;   // the array faces away from its ground plane,
  double broadside_y = -axis_x;  // which is on the left of the first-to-last direction
  double center_x = (first_element.x + last_element.x) / 2.0;
  double center_y = (first_element.y + last_element.y) / 2.0;
  auto to_screen = [&](double out, double along) {
    double pixels_per_meter = static_cast<double>(GetScreenWidth()) / geometry.domain_width;
    return Vector2{
      static_cast<float>((center_x + out * broadside_x + along * axis_x) * pixels_per_meter),
      static_cast<float>((center_y + out * broadside_y + along * axis_y) * pixels_per_meter),
    };
  };
  auto on_screen = [&](auto const& echo) {
    return to_screen(std::cos(echo.theta) * echo.range, std::sin(echo.theta) * echo.range);
  };

  // the latest set of tracks; a set with a new report number replaces it
  std::vector<Track> shown_tracks;
  std::size_t shown_report = 0;

  while (!WindowShouldClose()) {
    // to the sim a dragged target jumps, a frame is only steps_per_frame steps
    double cells_per_pixel = static_cast<double>(grid_width) / GetScreenWidth();
    Vector2 mouse = GetMousePosition();
    auto mouse_x = static_cast<std::ptrdiff_t>(mouse.x * cells_per_pixel);
    auto mouse_y = static_cast<std::ptrdiff_t>(mouse.y * cells_per_pixel);
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
      for (std::size_t i = 0; i < targets.size(); ++i) {
        auto dx = mouse_x - static_cast<std::ptrdiff_t>(targets[i].x);
        auto dy = mouse_y - static_cast<std::ptrdiff_t>(targets[i].y);
        auto radius = static_cast<std::ptrdiff_t>(targets[i].radius);
        if (dx * dx + dy * dy <= radius * radius) dragged = i;
      }
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) dragged.reset();
    if (dragged) {
      auto& target = targets[*dragged];
      // keep the disk out of the sponge and 1λ in front of the array
      auto radius = static_cast<std::ptrdiff_t>(target.radius);
      auto x = static_cast<std::size_t>(std::clamp(mouse_x,
        static_cast<std::ptrdiff_t>(array_start_x + 20) + radius,
        static_cast<std::ptrdiff_t>(grid_width - border) - radius));
      auto y = static_cast<std::size_t>(std::clamp(mouse_y,
        static_cast<std::ptrdiff_t>(border) + radius,
        static_cast<std::ptrdiff_t>(grid_height - border) - radius));
      if (x != target.x || y != target.y) {
        radar_sim.remove_metal(disk(target.x, target.y, target.radius));
        target.x = x;
        target.y = y;
        // re-add every target, so dragging one across another leaves no hole in it
        for (auto const& other : targets) radar_sim.add_metal(disk(other.x, other.y, other.radius));
        std::copy(features.begin(), features.end(), feature_bytes.begin());
        UpdateTexture(features_tex, feature_bytes.data());
      }
    }

    double start_time = GetTime();
    for (int s = 0; s < steps_per_frame; ++s) {
      radar_sim.step();
      SampleBatch<n_elements> batch;
      array.fill(batch);
      // make sure that no batches are dropped and lost
      while(!sample_batches.push(batch)) {};
      while (auto request = pulse_requests.pop()) {
        array.fire(*request);
      }
    }
    double work_time = GetTime() - start_time;
    UpdateTexture(ez_tex, ez_grid.data());
    BeginDrawing();
    BeginShaderMode(shader);
    SetShaderValueTexture(shader, features_loc, features_tex);  // must be inside BeginShaderMode
    DrawTexturePro(ez_tex,
      {0, 0, static_cast<float>(grid_width), static_cast<float>(grid_height)},
      {0, 0, static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight())},
      {0, 0}, 0.0f, WHITE);
    EndShaderMode();

    while (auto detection = detections.pop()) shown_detections.push_back(*detection);
    while (auto plot = plots.pop()) shown_plots.push_back(*plot);
    double now = radar_sim.time();
    drop_old(shown_detections, now);
    drop_old(shown_plots, now);
    for (auto const& detection : shown_detections) {
      auto age = static_cast<float>((now - detection.time) / detection_lifetime);
      DrawCircleV(on_screen(detection), 6.0f, Fade(YELLOW, 1.0f - age));
    }
    for (auto const& plot : shown_plots) {
      Vector2 center = on_screen(plot);
      DrawCircleLines(static_cast<int>(center.x), static_cast<int>(center.y), 14.0f, GREEN);
      DrawText(TextFormat("%zu", plot.detections), static_cast<int>(center.x) + 16, static_cast<int>(center.y) - 8, 16, GREEN);
    }

    while (auto track = tracks.pop()) {
      if (track->report != shown_report) {
        shown_tracks.clear();
        shown_report = track->report;
      }
      shown_tracks.push_back(*track);
    }
    for (auto const& track : shown_tracks) {
      Color color = track.confirmed ? SKYBLUE : GRAY;
      Vector2 center = to_screen(track.out, track.along);
      // the line points to where the track expects the target one scan from now
      Vector2 ahead = to_screen(track.out + track.v_out * detection_lifetime, track.along + track.v_along * detection_lifetime);
      DrawCircleLines(static_cast<int>(center.x), static_cast<int>(center.y), 22.0f, color);
      DrawLineV(center, ahead, color);
      DrawText(TextFormat("#%zu", track.id), static_cast<int>(center.x) - 10, static_cast<int>(center.y) - 42, 16, color);
    }

    DrawText(TextFormat("work %4.2f ms  idle %4.2f ms  fps %d", work_time * 1000.0, (GetFrameTime() - work_time) * 1000.0, GetFPS()), 10, 10, 20, WHITE);
    EndDrawing();
  }
  sample_batches.close();
  worker_thread.join();
  UnloadShader(shader);
  UnloadTexture(features_tex);
  UnloadTexture(ez_tex);
  CloseWindow();
  std::printf("closed, samples recorded at dt = %g s\n", radar_sim.dt());
}
