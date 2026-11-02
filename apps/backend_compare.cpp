// runs the CPU and GPU backends on the same scene with the same sources and compares
// every probe sample, then times both on the radar sim's grid size
//
// the two will not match bit for bit, the GPU sums in its own order and may fuse multiplies
// and adds, but the largest difference should be tiny next to the largest value
#include <rt/cpu_backend.hpp>
#include <rt/gpu_backend.hpp>
#include <raylib.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr double pi = 3.14159265358979323846;

// a few cycles of a sine with a gaussian envelope, period 20 steps (λ = 40 cells at courant 0.5)
float pulse(std::size_t step) {
  double t = static_cast<double>(step);
  double envelope = std::exp(-std::pow((t - 60.0) / 20.0, 2.0));
  return static_cast<float>(envelope * std::sin(2.0 * pi * t / 20.0));
}

void setup(FdtdBackend& backend, std::vector<Cell> const& sources, std::vector<Cell> const& probes,
           std::vector<Cell> const& metal) {
  backend.set_pec(metal, true);
  backend.set_source_cells(sources);
  backend.set_probe_cells(probes);
}

// runs `steps` steps in batches of `batch`, every source cell driven by pulse(), all probe rows appended to out
void run(FdtdBackend& backend, std::size_t steps, std::size_t batch, std::vector<float>& out) {
  std::size_t n_sources = backend.source_count();
  std::size_t n_probes = backend.probe_count();
  std::vector<float> values;
  std::vector<float> rows;
  for (std::size_t done = 0; done < steps;) {
    std::size_t n = std::min(batch, steps - done);
    values.resize(n * n_sources);
    for (std::size_t i = 0; i < n; ++i)
      std::fill_n(values.begin() + static_cast<std::ptrdiff_t>(i * n_sources), n_sources, pulse(done + i));
    rows.resize(n * n_probes);
    backend.step(n, values.data(), rows.data());
    out.insert(out.end(), rows.begin(), rows.end());
    done += n;
  }
}

double steps_per_second(FdtdBackend& backend, std::size_t steps, std::size_t batch) {
  std::vector<float> out;
  run(backend, batch, batch, out);  // warm up: first dispatches, page faults
  auto start = std::chrono::steady_clock::now();
  run(backend, steps, batch, out);
  double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return static_cast<double>(steps) / seconds;
}

}  // namespace

int main() {
  SetTraceLogLevel(LOG_WARNING);
  SetConfigFlags(FLAG_WINDOW_HIDDEN);
  InitWindow(64, 64, "backend compare");  // only for the OpenGL context

  // correctness: a small grid with a metal disk, a source line and probes spread over the grid
  constexpr std::size_t width = 300;
  constexpr std::size_t height = 200;
  constexpr std::size_t border = 20;
  constexpr float courant = 0.5f;

  std::vector<Cell> sources{{40, 90}, {40, 100}, {40, 110}};
  std::vector<Cell> probes = sources;
  for (std::size_t y = 10; y < height; y += 20)
    for (std::size_t x = 10; x < width; x += 20) probes.push_back({x, y});
  std::vector<Cell> metal;
  for (std::size_t y = 85; y <= 115; ++y)
    for (std::size_t x = 185; x <= 215; ++x)
      if ((x - 200) * (x - 200) + (y - 100) * (y - 100) <= 225) metal.push_back({x, y});

  CpuBackend cpu(width, height, border, courant);
  GpuBackend gpu(width, height, border, courant);
  setup(cpu, sources, probes, metal);
  setup(gpu, sources, probes, metal);

  // an odd batch size, so the last batch is a partial one
  constexpr std::size_t steps = 3000;
  constexpr std::size_t batch = 128;
  std::vector<float> cpu_out;
  std::vector<float> gpu_out;
  run(cpu, steps, batch, cpu_out);
  run(gpu, steps, batch, gpu_out);

  double max_value = 0.0;
  double max_diff = 0.0;
  std::size_t worst = 0;
  for (std::size_t i = 0; i < cpu_out.size(); ++i) {
    max_value = std::max(max_value, std::abs(static_cast<double>(cpu_out[i])));
    double diff = std::abs(static_cast<double>(cpu_out[i]) - static_cast<double>(gpu_out[i]));
    if (diff > max_diff) {
      max_diff = diff;
      worst = i;
    }
  }
  double relative = max_diff / max_value;
  std::printf("%zu steps, %zu probes: largest |ez| %.4g, largest difference %.3g (%.2g relative) at step %zu, probe %zu\n",
              steps, probes.size(), max_value, max_diff, relative, worst / probes.size(), worst % probes.size());
  bool match = relative < 1e-3;
  std::printf("%s\n", match ? "MATCH" : "MISMATCH");

  // speed on the radar sim's grid, 16 probes like the array
  constexpr std::size_t sim_width = 1000;
  constexpr std::size_t sim_height = 700;
  std::vector<Cell> array;
  for (std::size_t i = 0; i < 16; ++i) array.push_back({100, 275 + 10 * i});
  CpuBackend cpu_big(sim_width, sim_height, 60, courant);
  GpuBackend gpu_big(sim_width, sim_height, 60, courant);
  setup(cpu_big, array, array, {});
  setup(gpu_big, array, array, {});
  for (std::size_t sim_batch : {std::size_t{15}, std::size_t{128}, std::size_t{1024}}) {
    double cpu_rate = steps_per_second(cpu_big, 2048, sim_batch);
    double gpu_rate = steps_per_second(gpu_big, 2048, sim_batch);
    std::printf("%zux%zu, batch %4zu: CPU %7.0f steps/s, GPU %7.0f steps/s (%.1fx)\n",
                sim_width, sim_height, sim_batch, cpu_rate, gpu_rate, gpu_rate / cpu_rate);
  }

  CloseWindow();
  return match ? 0 : 1;
}
