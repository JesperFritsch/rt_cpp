#include <rt/FDTDGrid.hpp>
#include <cstdio>
#include <algorithm>


void FDTDGrid::updateH() {
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


void FDTDGrid::updateE() {
  #pragma omp parallel for
  for (std::size_t y = 1; y < height_ - 1; ++y) {
    for (std::size_t x = 1; x < width_ - 1; ++x) {
      std::size_t cell = idx(x, y);
      ez_[cell] = ez_[cell] + ((courant_ * ((hy_[cell] - hy_[idx(x - 1, y)]) - (hx_[cell] - hx_[idx(x, y - 1)])))) * damp_[cell] * !pec_[cell];
    }
  }
}

FDTDGrid::FDTDGrid(
  std::size_t width,
  std::size_t height,
  std::size_t border,
  float courant,
  float dampening = 1.0
)
      : ez_(width * height),
        hx_(width * height),
        hy_(width * height),
        damp_(width * height, dampening),
        pec_(width * height, 0),
        width_(width),
        height_(height),
        border_(border),
        courant_(courant){
// sponge: fields are slightly attenuated each step near the edges
for (std::size_t x = 0; x < width; ++x)
  for (std::size_t y = 0; y < height; ++y) {
    std::size_t dist_to_edge = std::min({x, y, width - 1 - x, height - 1 - y});
    if (dist_to_edge < border) {
      // 0 at the inner edge of the sponge, 1 at the grid edge
      float depth = static_cast<float>(border - dist_to_edge) / static_cast<float>(border);
      damp_[idx(x, y)] = 1.0f - 0.1f * depth * depth * depth;
    }
  }
}


void FDTDGrid::update() {
  updateH();
  updateE();
}
