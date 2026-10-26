#include <rt/FDTDGrid.hpp>
#include <cstdio>
#include <cmath>
#include <raylib.h>
#include <algorithm>
#include <functional>

constexpr double visual_gain = 5.0;
constexpr double pi = 3.14159265358979323846;
constexpr double c0 = 299'792'458.0;  // speed of light, m/s

Color to_red(double v) {
  v = std::clamp(v, 0.0, 1.0);
  return Color{ static_cast<unsigned char>(std::lround(v * 255)), 0, 0, 255 };
}

Color to_blue(double v) {
  v = std::clamp(v, 0.0, 1.0);
  return Color{ 0, 0, static_cast<unsigned char>(std::lround(v * 255)), 255 };
}

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
using SourceFn = std::function<double(double)>;

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
  std::vector<double> samples;
};

class RadarSim {

private:
  FDTDGrid grid_;
  double cell_size_;  // meters per cell
  double dt_;         // seconds per step
  std::size_t step_ = 0;
  std::vector<Source> sources;
  std::vector<Receiver> receivers;

public:

  // dt follows from cell_size and courant: dt = courant * cell_size / c
  RadarSim(
    std::size_t width,
    std::size_t height,
    std::size_t border,
    double cell_size = 0.01,
    double courant = 0.5,
    double damp = 1.0
  ) :
    grid_(width, height, border, courant, damp),
    cell_size_(cell_size),
    dt_(courant * cell_size / c0)
  {}

  double cell_size() const { return cell_size_; }
  double dt() const { return dt_; }
  double time() const { return static_cast<double>(step_) * dt_; }

  void create_source(std::vector<Cell> cells, SourceFn fn) {
    sources.push_back({std::move(cells), std::move(fn)});
  }

  // returns an id to read the samples back with
  std::size_t create_receiver(Cell cell) {
    receivers.push_back({cell, {}});
    return receivers.size() - 1;
  }

  std::vector<double> const& samples(std::size_t receiver_id) const {
    return receivers[receiver_id].samples;
  }

  // mark cells as perfect electric conductor; waves reflect off them
  void add_metal(std::vector<Cell> const& cells) {
    for (auto const& cell : cells) grid_.set_pec(cell.x, cell.y, true);
  }

  void remove_metal(std::vector<Cell> const& cells) {
    for (auto const& cell : cells) grid_.set_pec(cell.x, cell.y, false);
  }

  std::vector<double> const& grid() const {
    return grid_.get_ez();
  }

  std::vector<char> const& metal() const {
    return grid_.get_pec();
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

int main() {
  InitWindow(1200, 1200, "radar sim");
  SetTargetFPS(60);
  constexpr std::size_t grid_size = 1200;
  constexpr std::size_t border = 40;
  constexpr int steps_per_frame = 2;
  constexpr double cell_size = 0.01;   // 1 cm per cell
  constexpr double freq = 1.5e9;       // λ = c / f = 20 cm = 20 cells

  auto radar_sim = RadarSim(grid_size, grid_size, border, cell_size);
  auto const& ez_grid = radar_sim.grid();
  std::vector<Color> pixels(grid_size * grid_size);
  Image img = {
    pixels.data(),
    grid_size,
    grid_size,
    1,
    PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
  };
  Texture2D tex = LoadTextureFromImage(img);

  // antenna just inside the sponge border
  radar_sim.create_source(
    line(grid_size / 2 - 5, grid_size / 2, grid_size / 2 + 5, grid_size / 2),
    burst(freq, 3, 5.0)
  );

  radar_sim.create_source(
    line(grid_size / 2, grid_size / 3, grid_size / 2, grid_size / 3),
    gaussian_pulse(4.5e-9, 0.1e-9, 3.0)
  );
  auto rx = radar_sim.create_receiver({grid_size / 2, border + 10});

  // targets: a round object and a flat plate
  radar_sim.add_metal(disk(grid_size / 4, grid_size / 4, 30));
  radar_sim.add_metal(line(3 * grid_size / 4, grid_size / 4, 3 * grid_size / 4 + 4, grid_size / 2));

  auto const& metal = radar_sim.metal();

  while (!WindowShouldClose()) {
    for (int s = 0; s < steps_per_frame; ++s) radar_sim.step();
    std::transform(ez_grid.begin(), ez_grid.end(), metal.begin(), pixels.begin(),
      [](double v, char is_metal) {
        if (is_metal) return GRAY;
        auto gain_scaled = v * visual_gain;
        return v >= 0 ? to_red(gain_scaled) : to_blue(-gain_scaled);
      });
    UpdateTexture(tex, pixels.data());
    BeginDrawing();
    DrawTexturePro(tex,
      {0, 0, static_cast<float>(grid_size), static_cast<float>(grid_size)},
      {0, 0, static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight())},
      {0, 0}, 0.0f, WHITE);
    EndDrawing();
  }
  UnloadTexture(tex);
  CloseWindow();
  std::printf("closed, %zu samples recorded at dt = %g s\n",
              radar_sim.samples(rx).size(), radar_sim.dt());
}
