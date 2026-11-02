#pragma once
#include <rt/fdtd_backend.hpp>
#include <cstdint>
#include <vector>

// the FDTD updates as OpenGL 4.3 compute shaders; the fields stay in GPU buffers the whole run
// and the renderer reads ez straight from them
//
// needs a current OpenGL 4.3 context, so construct it after raylib's InitWindow, and call it
// from the thread that owns the context
//
// per step the batch runs four dispatches, each finishing before the next starts:
// add sources -> update H -> update E -> record probes
// the probe rows of the whole batch are read back once at the end
class GpuBackend final : public FdtdBackend {
public:
  GpuBackend(
    std::size_t width,
    std::size_t height,
    std::size_t border,
    float courant,
    float damp = 1.0f
  );
  ~GpuBackend() override;

  // owns GPU buffers and programs
  GpuBackend(GpuBackend const&) = delete;
  GpuBackend& operator=(GpuBackend const&) = delete;

  std::size_t width() const override { return width_; }
  std::size_t height() const override { return height_; }

  // uploads the whole metal buffer, then zeroes ez in every metal cell on the GPU
  void set_pec(std::vector<Cell> const& cells, bool metal) override;

  void set_source_cells(std::vector<Cell> cells) override;
  void set_probe_cells(std::vector<Cell> cells) override;
  std::size_t source_count() const override { return source_count_; }
  std::size_t probe_count() const override { return probe_count_; }

  void step(std::size_t n, float const* source_values, float* probe_out) override;

  std::size_t current_step() const override { return step_; }
  FieldView ez() const override { return {nullptr, ez_}; }

private:
  std::size_t width_, height_;
  float courant_;
  std::size_t source_count_ = 0;
  std::size_t probe_count_ = 0;
  std::size_t step_ = 0;
  std::vector<std::uint32_t> pec_;  // CPU copy of the metal buffer, 1 = metal cell

  // buffers, at the bindings the shaders declare them at
  unsigned ez_ = 0;             // 0
  unsigned hx_ = 0;             // 1
  unsigned hy_ = 0;             // 2
  unsigned damp_ = 0;           // 3
  unsigned pec_buffer_ = 0;     // 4
  unsigned source_cells_ = 0;   // 5, flat cell indices
  unsigned source_values_ = 0;  // 6, batch x source cells
  unsigned probe_cells_ = 0;    // 7, flat cell indices
  unsigned probe_out_ = 0;      // 8, batch x probe cells
  std::size_t source_values_capacity_ = 0;  // floats
  std::size_t probe_out_capacity_ = 0;      // floats

  unsigned add_sources_ = 0;  // programs
  unsigned update_h_ = 0;
  unsigned update_e_ = 0;
  unsigned record_probes_ = 0;
  unsigned zero_metal_ = 0;
  int add_sources_row_ = -1;  // uniform locations that change per step
  int record_probes_row_ = -1;

  void bind_buffers() const;
  unsigned upload_indices(unsigned buffer, std::vector<Cell> const& cells);
};
