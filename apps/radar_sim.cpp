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
};

struct PulseRequest {
  std::size_t step;  // step of the sample batch this request answers
  Waveform wave;
  double theta;
  double fire_time;
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
  double f_start = 1.2e9;
  double f_end = 1.8e9;
  double pulse_duration = 10e-9;
  double amplitude = 2.0;
  double max_scan_angle_r = 60.0 * pi / 180.0;  // radians from broadside
  double pri_margin = 1.0;                     // < 1 trims the conservative listen time

  // derived
  double bandwidth() const          { return f_end - f_start; }
  double blind_range() const        { return speed_of_light * pulse_duration / 2.0; }
  double range_resolution() const   { return speed_of_light / (2.0 * bandwidth()); }
  double pulse_length_m() const     { return speed_of_light * pulse_duration; }
  Waveform waveform() const         { return {f_start, f_end, pulse_duration, amplitude}; }

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
float wave_contrib(double time_in_pulse, Waveform const& wave) {
  double chirp_rate = (wave.f_end - wave.f_start) / wave.duration;  // Hz per second
  double cycles = wave.f_start * time_in_pulse + 0.5 * chirp_rate * time_in_pulse * time_in_pulse;
  auto contrib = time_in_pulse < wave.duration ? wave.amplitude * std::sin(2.0 * pi * cycles) : 0.0;
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
    for (auto const& cell : cells) grid_.set_pec(cell.x, cell.y, false);
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
  PulseScheduler(std::size_t pri_steps, double dt, Waveform waveform) :
    pri_steps_(pri_steps),
    dt_(dt),
    wave_(waveform)
  {

  }

  std::optional<PulseRequest> on_step(std::size_t step) {
    if( step >= next_pulse_step_ ) {
      next_pulse_step_ = step + pri_steps_;
      next_theta_ += theta_stride_;
      if (next_theta_ >= abs_theta_) {
        next_theta_ = -abs_theta_;
      }
      return PulseRequest{
        step,
        wave_,
        next_theta_,
        next_pulse_step_ * dt_,
      };
    }
    return {};
  }

  void set_angles(double abs_angle_r, double ang_stride_r) {
    abs_theta_ = abs_angle_r;
    theta_stride_ = ang_stride_r;
  } 

private:
  double theta_stride_;
  double abs_theta_;
  std::size_t pri_steps_;
  double dt_;
  Waveform wave_;
  std::size_t next_pulse_step_ = 0;
  double next_theta_ = 0.0;
};

// runs on the worker thread: takes sample batches from the sim and answers with pulse requests
template<std::size_t SAMPLE_RING_SIZE, std::size_t REQUEST_RING_SIZE, std::size_t N_ELEMENTS>
class RadarProcessor {
public:
  RadarProcessor(rt::SPSCRing<PulseRequest, REQUEST_RING_SIZE>& pulse_requests,
                 rt::SPSCRing<SampleBatch<N_ELEMENTS>, SAMPLE_RING_SIZE>& sample_batches,
                 RadarConfig config,
                 Geometry geometry)
    : pulse_requests_(pulse_requests),
      sample_batches_(sample_batches),
      config_(config),
      geometry_(std::move(geometry)),
      dt_(geometry_.dt),
      pri_steps_(config_.pri_steps(geometry_)),
      scheduler_(pri_steps_, dt_, config_.waveform())
  {
    assert(geometry_.elements.size() == N_ELEMENTS);
    prev_requests_.reserve(max_requests_);
    scheduler_.set_angles(config_.max_scan_angle_r, config_.max_scan_angle_r / 10.0);
  }

  std::size_t pri_steps() const { return pri_steps_; }

  void run() {
    while ( auto batch = sample_batches_.pop_blocking() ) handle(*batch);
  }

  void handle(SampleBatch<N_ELEMENTS> const& batch) {
    if (auto req = scheduler_.on_step(batch.step)) {
      while (!pulse_requests_.push(*req)) {}
    }
    prev_requests_.clear();
  }

private:
  rt::SPSCRing<PulseRequest, REQUEST_RING_SIZE>& pulse_requests_;           // to the sim
  rt::SPSCRing<SampleBatch<N_ELEMENTS>, SAMPLE_RING_SIZE>& sample_batches_; // from the sim
  std::vector<PulseRequest> prev_requests_;
  std::size_t max_requests_ = 10;
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
  constexpr int window_width = 1600;
  constexpr int window_height = static_cast<int>(window_width * grid_height / grid_width);
  InitWindow(window_width, window_height, "radar sim");
  SetTargetFPS(512);
  constexpr int steps_per_frame = 10;
  constexpr double cell_size = 0.01;   // 1 cm per cell
  constexpr std::size_t sample_ring_size = 128;
  constexpr std::size_t request_ring_size = 128;

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

  // target
  radar_sim.add_metal(disk(grid_width * 3 / 4, grid_height * 3 / 4, 30));

  auto array = radar_sim.create_phased_array(
    {array_start_x, array_start_y },
    n_elements, 
    spacing,
    false
  );

  rt::SPSCRing<PulseRequest, request_ring_size> pulse_requests;
  rt::SPSCRing<SampleBatch<n_elements>, sample_ring_size> sample_batches;


  // features are uploaded once; re-upload feature_bytes with UpdateTexture if they change
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
  RadarProcessor processor(pulse_requests, sample_batches, config, geometry);
  std::printf("PRI %zu steps (%.1f ns), blind range %.2f m, range resolution %.2f m\n",
              processor.pri_steps(), static_cast<double>(processor.pri_steps()) * radar_sim.dt() * 1e9,
              config.blind_range(), config.range_resolution());
  auto worker_thread = std::thread([&]() { processor.run(); });

  while (!WindowShouldClose()) {
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
