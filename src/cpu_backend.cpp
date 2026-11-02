#include <rt/cpu_backend.hpp>
#include <algorithm>


void CpuBackend::updateH() {
  #pragma omp parallel for
  for (std::size_t y = 0; y < height_ - 1; ++y) {
    for (std::size_t x = 0; x < width_; ++x) {
      std::size_t cell = idx(x, y);
      hx_[cell] = (hx_[cell] - courant_ * (ez_[idx(x, y + 1)] - ez_[cell])) * damp_[cell];
    }
  }
  #pragma omp parallel for
  for (std::size_t y = 0; y < height_; ++y) {
    for (std::size_t x = 0; x < width_ - 1; ++x) {
      std::size_t cell = idx(x, y);
      hy_[cell] = (hy_[cell] + courant_ * (ez_[idx(x + 1, y)] - ez_[cell])) * damp_[cell];
    }
  }
}


void CpuBackend::updateE() {
  #pragma omp parallel for
  for (std::size_t y = 1; y < height_ - 1; ++y) {
    for (std::size_t x = 1; x < width_ - 1; ++x) {
      std::size_t cell = idx(x, y);
      ez_[cell] = ez_[cell] + ((courant_ * ((hy_[cell] - hy_[idx(x - 1, y)]) - (hx_[cell] - hx_[idx(x, y - 1)])))) * damp_[cell] * !pec_[cell];
    }
  }
}

CpuBackend::CpuBackend(
  std::size_t width,
  std::size_t height,
  std::size_t border,
  float courant,
  float damp
)
      : ez_(width * height),
        hx_(width * height),
        hy_(width * height),
        damp_(sponge_damping(width, height, border, damp)),
        pec_(width * height, 0),
        width_(width),
        height_(height),
        courant_(courant) {}

std::vector<float> sponge_damping(std::size_t width, std::size_t height, std::size_t border, float damp) {
  std::vector<float> out(width * height, damp);
  // sponge: fields are slightly attenuated each step near the edges
  for (std::size_t x = 0; x < width; ++x)
    for (std::size_t y = 0; y < height; ++y) {
      std::size_t dist_to_edge = std::min({x, y, width - 1 - x, height - 1 - y});
      if (dist_to_edge < border) {
        // 0 at the inner edge of the sponge, 1 at the grid edge
        float depth = static_cast<float>(border - dist_to_edge) / static_cast<float>(border);
        out[y * width + x] = 1.0f - 0.1f * depth * depth * depth;
      }
    }
  return out;
}

void CpuBackend::set_pec(std::vector<Cell> const& cells, bool metal) {
  for (auto const& cell : cells) {
    std::size_t i = idx(cell.x, cell.y);
    pec_[i] = metal ? 1 : 0;
    if (metal) ez_[i] = 0.0f;
  }
}

void CpuBackend::set_source_cells(std::vector<Cell> cells) {
  sources_.clear();
  for (auto const& cell : cells) sources_.push_back(idx(cell.x, cell.y));
}

void CpuBackend::set_probe_cells(std::vector<Cell> cells) {
  probes_.clear();
  for (auto const& cell : cells) probes_.push_back(idx(cell.x, cell.y));
}

void CpuBackend::step(std::size_t n, float const* source_values, float* probe_out) {
  for (std::size_t i = 0; i < n; ++i) {
    float const* values = source_values + i * sources_.size();
    for (std::size_t s = 0; s < sources_.size(); ++s) ez_[sources_[s]] += values[s];
    updateH();
    updateE();
    float* out = probe_out + i * probes_.size();
    for (std::size_t p = 0; p < probes_.size(); ++p) out[p] = ez_[probes_[p]];
    ++step_;
  }
}
