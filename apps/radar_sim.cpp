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

constexpr uint8_t CF_METAL  = 0x1;
constexpr uint8_t CF_RX     = 0x2;
constexpr uint8_t CF_TX     = 0x4;

constexpr double visual_gain = 5.0;
constexpr double pi = 3.14159265358979323846;
constexpr double c0 = 299'792'458.0;  // speed of light, m/s

// colors the ez texture: red for positive, blue for negative, gray for metal
constexpr const char* field_shader = R"(
#version 330
in vec2 fragTexCoord;
out vec4 finalColor;
uniform sampler2D texture0;  // ez, single float channel
uniform sampler2D cf_tex;   // bit 1 = metal, bit 2 = receiver, bit 3 = transmitter
uniform float gain;
void main() {
  int f = int(texture(cf_tex, fragTexCoord).r * 255 + 0.5);
  if ((f & 0x1) != 0) {
    finalColor = vec4(0.5, 0.5, 0.5, 1.0);
    return;
  }
  if ((f & 0x2) != 0) {
    finalColor = vec4(0.9, 0.95, 0.0, 1.0);
    return;
  }
  if ((f & 0x4) != 0) {
    finalColor = vec4(0.02, 0.87, 0.98, 1.0);
    return;
  }
  float v = clamp(texture(texture0, fragTexCoord).r * gain, -1.0, 1.0);
  finalColor = vec4(max(v, 0.0), 0.0, max(-v, 0.0), 1.0);
}
)";

struct Cell { std::size_t x, y; };

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

// filled disk of radius r (cells) centered at (cx, cy)
std::vector<Cell> disk(std::size_t cx, std::size_t cy, std::size_t r) {
  std::vector<Cell> out;
  auto r2 = static_cast<double>(r * r);
  for (std::size_t x = cx - r; x <= cx + r; ++x)
    for (std::size_t y = cy - r; y <= cy + r; ++y) {
      double dx = static_cast<double>(x) - static_cast<double>(cx);
      double dy = static_cast<double>(y) - static_cast<double>(cy);
      if (dx * dx + dy * dy <= r2) out.push_back({x, y});
    }
  return out;
}

// source signal as a function of simulation time in seconds
using SourceFn = std::function<float(double)>;

SourceFn sine(double freq_hz, double amp = 1.0) {
  return [=](double t) { return amp * std::sin(2.0 * pi * freq_hz * t); };
}

// gaussian centered at t0 seconds, `width` seconds wide
SourceFn gaussian_pulse(double t0, double width, double amp = 1.0) {
  return [=](double t) {
    double u = (t - t0) / width;
    return amp * std::exp(-u * u);
  };
}

// `cycles` periods of a sine, then silence
SourceFn burst(double freq_hz, int cycles, double amp = 1.0) {
  double len = static_cast<double>(cycles) / freq_hz;
  return [=](double t) { return t < len ? amp * std::sin(2.0 * pi * freq_hz * t) : 0.0; };
}

struct Source {
  std::vector<Cell> cells;
  SourceFn fn;
};

// records ez at one cell every step; sample rate is 1 / dt
struct Receiver {
  Cell cell;
  std::vector<float> samples;
};

template<std::size_t N>
struct SampleBatch {
  std::size_t seq;
  std::array<float, N> samples;
};

// linear array along x that fires steered pulses on demand
// `pulse` is the pulse shape with t = 0 at pulse start, only evaluated for t in [0, pulse_len]
// theta is in radians from broadside; positive steers toward +x
class PhasedArray {
public:
  // copies share state, so a copy kept by RadarSim records into the same samples
  PhasedArray(std::vector<Cell> cells, SourceFn pulse, double pulse_len, double cell_size,
              double length_cells, double theta)
    : state_(std::make_shared<State>()) {
    auto& s = *state_;
    s.pulse = std::move(pulse);
    s.pulse_len = pulse_len;
    s.cell_size = cell_size;
    s.length_cells = length_cells;
    s.theta = theta;
    s.samples.resize(cells.size());
    s.cells = std::move(cells);
  }

  double theta() const { return state_->theta; }

  // element cells, in the same order as samples()
  std::vector<Cell> const& cells() const { return state_->cells; }

  // ez at each element from the latest step; overwritten every step
  std::vector<float> const& samples() const { return state_->samples; }
  std::size_t sample_seq() const { return state_->seq; }

  // copies the latest samples into out; N must equal the number of elements
  template <std::size_t N>
  void fill(SampleBatch<N>& out) const {
    assert(state_->samples.size() == N);
    out.seq = state_->seq;
    std::copy_n(state_->samples.begin(), N, out.samples.begin());
  }

  // called by RadarSim::step for each element
  void record(std::size_t seq, std::size_t element, float value) {
    state_->seq = seq;
    state_->samples[element] = value;
  }

  // applies to pulses fired from now on; pulses already in flight keep their angle
  void set_theta(double theta) { state_->theta = theta; }

  // fire a pulse starting at sim time t0, steered at the current theta
  void fire(double t0) {
    auto& s = *state_;
    // drop pulses that have finished on every element
    double max_delay = s.length_cells * s.cell_size / c0;
    s.pulses.erase(
      std::remove_if(s.pulses.begin(), s.pulses.end(),
                     [&](Pulse const& p) { return t0 > p.t0 + s.pulse_len + max_delay; }),
      s.pulses.end());
    s.pulses.push_back({t0, std::sin(s.theta)});
  }

  // signal for the element `offset_cells` from the array's first element
  // the fns share state_, so fire and set_theta reach them after they are handed out
  SourceFn element_fn(double offset_cells) const {
    return [s = state_, offset_cells](double t) {
      float sum = 0.0f;
      for (auto const& p : s->pulses) {
        // measure from the end that fires first so every delay is >= 0
        double d = p.sin_theta >= 0.0 ? offset_cells : offset_cells - s->length_cells;
        double tt = t - p.t0 - d * s->cell_size * p.sin_theta / c0;
        if (tt >= 0.0 && tt <= s->pulse_len) sum += s->pulse(tt);
      }
      return sum;
    };
  }

private:
  struct Pulse {
    double t0;
    double sin_theta;  // latched when fired
  };
  struct State {
    SourceFn pulse;
    double pulse_len;
    double cell_size;
    double length_cells;
    double theta;
    std::vector<Pulse> pulses;
    std::vector<Cell> cells;
    std::size_t seq = 0;
    std::vector<float> samples;
  };
  std::shared_ptr<State> state_;
};

struct Waveform {
  double f_start;
  double f_end;
  double duration;
  double amplitude;
};

struct PulseRequest {
  std::size_t seq;
  Waveform wave;
  double t_time;
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
  std::size_t width, height;
  std::vector<uint8_t> cell_features;

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
    dt_(courant * cell_size / c0),
    width(width),
    height(height),
    cell_features(width * height)
  {}

  double cell_size() const { return cell_size_; }
  double dt() const { return dt_; }
  double time() const { return static_cast<double>(step_) * dt_; }

  void create_source(std::vector<Cell> cells, SourceFn fn) {
    for (auto const& cell : cells) cell_features[cell.y * width + cell.x] |= CF_TX;
    sources.push_back({std::move(cells), std::move(fn)});
  }

  // `n` single-cell elements along x, `spacing` cells apart, starting at `first`
  // keep spacing <= λ/2 to avoid grating lobes
  // the array is silent until PhasedArray::fire is called
  PhasedArray create_phased_array(Cell first, std::size_t n, std::size_t spacing,
                                  SourceFn pulse, double pulse_len, double theta = 0.0) {
    auto length = static_cast<double>((n - 1) * spacing);
    std::vector<Cell> cells;
    for (std::size_t i = 0; i < n; ++i) cells.push_back({first.x + i * spacing, first.y});

    PhasedArray array(cells, std::move(pulse), pulse_len, cell_size_, length, theta);
    for (auto const& cell : cells)
      create_source({cell}, array.element_fn(static_cast<double>(cell.x - first.x)));
    arrays.push_back(array);
    return array;
  }

  // returns an id to read the samples back with
  std::size_t create_receiver(Cell cell) {
    receivers.push_back({cell, {}});
    cell_features[cell.y * width + cell.x] |= CF_RX;
    return receivers.size() - 1;
  }

  std::vector<float> const& samples(std::size_t receiver_id) const {
    return receivers[receiver_id].samples;
  }

  // mark cells as perfect electric conductor; waves reflect off them
  void add_metal(std::vector<Cell> const& cells) {
    for (auto const& cell : cells) {
      cell_features[cell.y * width + cell.x] |= CF_METAL;
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
    double t = time();
    for (auto& src : sources) {
      auto val = src.fn(t);
      for (auto& cell : src.cells) {
        grid_.add_source(cell.x, cell.y, val);
      }
    }
    grid_.update();
    for (auto& rx : receivers) {
      rx.samples.push_back(grid_.get_ez(rx.cell.x, rx.cell.y));
    }
    for (auto& array : arrays) {
      auto const& cells = array.cells();
      for (std::size_t i = 0; i < cells.size(); ++i)
        array.record(step_, i, grid_.get_ez(cells[i].x, cells[i].y));
    }
    ++step_;
  }

};

// prints a bar from a center line: left for negative, right for positive
// value is scaled so |value| == max_value fills half_width characters
void print_bar(float value, float max_value, int half_width) {
  float frac = std::clamp(value / max_value, -1.0f, 1.0f);
  auto len = static_cast<std::size_t>(std::lround(std::abs(frac) * static_cast<float>(half_width)));
  auto half = static_cast<std::size_t>(half_width);

  std::string left(half, ' ');
  std::string right(half, ' ');
  if (frac < 0)
    std::fill(left.end() - static_cast<std::ptrdiff_t>(len), left.end(), '#');
  else
    std::fill(right.begin(), right.begin() + static_cast<std::ptrdiff_t>(len), '#');

  std::printf("\r\033[2K%8.4f %s|%s", static_cast<double>(value), left.c_str(), right.c_str());
  std::fflush(stdout);
}

template<std::size_t RX, std::size_t TX, std::size_t SPL>
auto processing_worker(
  std::shared_ptr<rt::SPSCRing<PulseRequest, TX>> tx_buff,
  std::shared_ptr<rt::SPSCRing<SampleBatch<SPL>, RX>> rx_buff
) {
  return [=]() {
    while ( auto batch = rx_buff->pop_blocking() ) {
      printf("got some data for seq: %ld", batch.value().seq);
      (void)tx_buff->push({batch.value().seq, {0.0, 0.0, 0.0, 0.0}, 0.0});
    }
  };
}

int main() {
  InitWindow(1000, 1000, "radar sim");
  SetTargetFPS(512);
  constexpr std::size_t grid_size = 1200;
  constexpr std::size_t border = grid_size / 10;
  constexpr int steps_per_frame = 1;
  constexpr double cell_size = 0.01;   // 1 cm per cell
  constexpr double freq = 1.5e9;       // λ = c / f = 20 cm = 20 cells
  constexpr std::size_t rx_buff_size = 128;
  constexpr std::size_t tx_buff_size = 128;

  auto radar_sim = RadarSim(grid_size, grid_size, border, cell_size);
  auto const& ez_grid = radar_sim.grid();
  // raylib only reads from image data, so the const_cast is safe
  Image ez_img = {
    const_cast<float*>(ez_grid.data()),
    grid_size,
    grid_size,
    1,
    PIXELFORMAT_UNCOMPRESSED_R32
  };
  Texture2D ez_tex = LoadTextureFromImage(ez_img);

  // antenna just inside the sponge border
  // 21 elements at λ/4 spacing, pulse is 3 cycles; SPACE fires, LEFT/RIGHT steer
  constexpr std::size_t n_elements = 21;
  constexpr std::size_t spacing = 5;
  constexpr int pulse_cycles = 3;
  auto array = radar_sim.create_phased_array(
    {grid_size / 2 - (n_elements - 1) * spacing / 2, grid_size - 200},
    n_elements, spacing, burst(freq, pulse_cycles, 2.0), pulse_cycles / freq
  );
  constexpr double theta_step = 5.0 * pi / 180.0;

  auto tx_buff = std::make_shared<rt::SPSCRing<PulseRequest, tx_buff_size>>();
  auto rx_buff = std::make_shared<rt::SPSCRing<SampleBatch<n_elements>, rx_buff_size>>();

  // targets: a round object and a flat plate
  radar_sim.add_metal(disk(grid_size / 4, grid_size / 4, 30));
  // radar_sim.add_metal(line(3 * grid_size / 4, grid_size / 4, 3 * grid_size / 4 + 4, grid_size / 2));
  // radar_sim.add_metal(line(grid_size / 2 - 50, 199, grid_size / 2 + 50, 200));

  // metal is uploaded once; re-upload pec_bytes with UpdateTexture if it changes
  auto const& features = radar_sim.get_cell_features();
  std::vector<unsigned char> cf_bytes(features.begin(), features.end());
  Image cf_img = {
    cf_bytes.data(),
    grid_size,
    grid_size,
    1,
    PIXELFORMAT_UNCOMPRESSED_GRAYSCALE
  };
  Texture2D cf_tex = LoadTextureFromImage(cf_img);

  Shader shader = LoadShaderFromMemory(nullptr, field_shader);
  int cf_loc = GetShaderLocation(shader, "cf_tex");
  int gain_loc = GetShaderLocation(shader, "gain");
  float gain = static_cast<float>(visual_gain);
  SetShaderValue(shader, gain_loc, &gain, SHADER_UNIFORM_FLOAT);

  auto worker_t = std::thread(processing_worker(tx_buff, rx_buff));

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_LEFT)) array.set_theta(array.theta() - theta_step);
    if (IsKeyPressed(KEY_RIGHT)) array.set_theta(array.theta() + theta_step);
    if (IsKeyPressed(KEY_SPACE)) array.fire(radar_sim.time());
    double start_time = GetTime();
    for (int s = 0; s < steps_per_frame; ++s) radar_sim.step();
    SampleBatch<n_elements> batch;
    array.fill(batch);
    (void)rx_buff->emplace(std::move(batch));
    double work_time = GetTime() - start_time;
    UpdateTexture(ez_tex, ez_grid.data());
    BeginDrawing();
    BeginShaderMode(shader);
    SetShaderValueTexture(shader, cf_loc, cf_tex);  // must be inside BeginShaderMode
    DrawTexturePro(ez_tex,
      {0, 0, static_cast<float>(grid_size), static_cast<float>(grid_size)},
      {0, 0, static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight())},
      {0, 0}, 0.0f, WHITE);
    EndShaderMode();
    DrawText(TextFormat("work %4.2f ms  idle %4.2f ms  fps %d", work_time * 1000.0, (GetFrameTime() - work_time) * 1000.0, GetFPS()), 10, 10, 20, WHITE);
    EndDrawing();
  }
  rx_buff->close();
  worker_t.join();
  UnloadShader(shader);
  UnloadTexture(cf_tex);
  UnloadTexture(ez_tex);
  CloseWindow();
  std::printf("closed, samples recorded at dt = %g s\n", radar_sim.dt());
}
