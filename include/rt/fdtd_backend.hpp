#pragma once
#include <cstddef>
#include <vector>

struct Cell { std::size_t x, y; };

// per-cell multiplier applied to the fields every step: `damp` inside, falling off
// toward the edges over the `border` outermost cells so waves die out instead of reflecting
// shared by every backend so they all absorb the same way
std::vector<float> sponge_damping(std::size_t width, std::size_t height, std::size_t border, float damp);

// where the ez field can be read for drawing; exactly one of the two is set
struct FieldView {
  float const* cpu = nullptr;  // width * height floats, row-major
  unsigned gl_buffer = 0;      // SSBO with the same layout
};

// runs the field updates; the caller decides what to inject and what to record
//
// timing convention, shared by every backend: during step k the source values for k are
// added to ez at the source cells, then H and E are updated, then ez is read at the probe cells
class FdtdBackend {
public:
  virtual ~FdtdBackend() = default;

  virtual std::size_t width() const = 0;
  virtual std::size_t height() const = 0;

  // metal cells hold ez at zero, so waves reflect off them
  virtual void set_pec(std::vector<Cell> const& cells, bool metal) = 0;

  // the order of the cells is the order of the values in step's arrays
  virtual void set_source_cells(std::vector<Cell> cells) = 0;
  virtual void set_probe_cells(std::vector<Cell> cells) = 0;
  virtual std::size_t source_count() const = 0;
  virtual std::size_t probe_count() const = 0;

  // advances n steps
  // source_values[i * source_count() + s] is added to source cell s during step i of the batch
  // probe_out[i * probe_count() + p] receives ez at probe cell p after step i of the batch
  virtual void step(std::size_t n, float const* source_values, float* probe_out) = 0;

  virtual std::size_t current_step() const = 0;
  virtual FieldView ez() const = 0;
};
