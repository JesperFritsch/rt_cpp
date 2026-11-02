#pragma once
#include <rt/fdtd_backend.hpp>
#include <vector>

// the FDTD updates on the CPU, parallelized over rows with OpenMP
class CpuBackend final : public FdtdBackend {
public:
  CpuBackend(
    std::size_t width,
    std::size_t height,
    std::size_t border,
    float courant,
    float damp = 1.0f
  );

  std::size_t width() const override { return width_; }
  std::size_t height() const override { return height_; }

  // updateE only stops updating ez in metal, so it must be zero there already;
  // a nonzero ez frozen in metal would drive the h fields around it up forever
  void set_pec(std::vector<Cell> const& cells, bool metal) override;

  void set_source_cells(std::vector<Cell> cells) override;
  void set_probe_cells(std::vector<Cell> cells) override;
  std::size_t source_count() const override { return sources_.size(); }
  std::size_t probe_count() const override { return probes_.size(); }

  void step(std::size_t n, float const* source_values, float* probe_out) override;

  std::size_t current_step() const override { return step_; }
  FieldView ez() const override { return {ez_.data(), 0}; }

private:
  // ez: electric field along z (out of the screen); hx, hy: magnetic field along x and y
  // damp: per-cell multiplier, < 1 in the sponge border
  std::vector<float> ez_, hx_, hy_, damp_;
  std::vector<char> pec_;  // perfect electric conductor, 1 = metal cell
  std::size_t width_, height_;
  float courant_;  // cells the wave moves per step, must be < 1/sqrt(2) in 2D
  std::vector<std::size_t> sources_;  // flat indices, in set_source_cells order
  std::vector<std::size_t> probes_;   // flat indices, in set_probe_cells order
  std::size_t step_ = 0;

  void updateH();
  void updateE();
  // index into the flat arrays for cell (x, y)
  std::size_t idx(std::size_t x, std::size_t y) const { return y * width_ + x; }
};
