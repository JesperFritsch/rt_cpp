#include <rt/FDTDGrid.hpp>
#include <cstdio>
#include <cmath>
#include <raylib.h>
#include <cstdint>
#include <functional>
#include <algorithm>
#include <string>

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

class PhasedArray {
public:
  PhasedArray(std::vector<Source> tx, std::vector<Receiver> rx) : transmitter_(tx), receiver_(rx) {}
private:
  std::vector<Source> transmitter_;
  std::vector<Receiver> receiver_; 
};

class RadarSim {

private:
  FDTDGrid grid_;
  double cell_size_;  // meters per cell
  double dt_;         // seconds per step
  std::size_t step_ = 0;
  std::vector<Source> sources;
  std::vector<Receiver> receivers;
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

int main() {
  InitWindow(1200, 1200, "radar sim");
  SetTargetFPS(512);
  constexpr std::size_t grid_size = 1200;
  constexpr std::size_t border = grid_size / 10;
  constexpr int steps_per_frame = 1;
  constexpr double cell_size = 0.01;   // 1 cm per cell
  constexpr double freq = 1.5e9;       // λ = c / f = 20 cm = 20 cells

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
  radar_sim.create_source(
    line(grid_size / 2 - 50, grid_size - 200, grid_size / 2 + 50, grid_size - 200),
    burst(freq, 3, 2.0)
  );

  auto rx = radar_sim.create_receiver({grid_size / 2, border + 10});

  // targets: a round object and a flat plate
  radar_sim.add_metal(disk(grid_size / 4, grid_size / 4, 30));
  radar_sim.add_metal(line(3 * grid_size / 4, grid_size / 4, 3 * grid_size / 4 + 4, grid_size / 2));
  radar_sim.add_metal(line(grid_size / 2 - 50, 200, grid_size / 2 + 50, 200));

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

  while (!WindowShouldClose()) {
    double start_time = GetTime();
    for (int s = 0; s < steps_per_frame; ++s) radar_sim.step();
    print_bar(radar_sim.samples(rx).back(), 1.0, 150);
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
  UnloadShader(shader);
  UnloadTexture(cf_tex);
  UnloadTexture(ez_tex);
  CloseWindow();
  std::printf("closed, %zu samples recorded at dt = %g s\n",
              radar_sim.samples(rx).size(), radar_sim.dt());
}
