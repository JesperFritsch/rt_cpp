#include <rt/FDTDGrid.hpp>
#include <cstdio>
#include <algorithm>

  
void FDTDGrid::updateH() {
  #pragma omp parallel for
  for (std::size_t y = 0; y < height - 1; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      std::size_t b_idx = idx(x, y);
      hx[b_idx] = (hx[b_idx] - courant * (ez[idx(x, y + 1)] - ez[b_idx])) * damp[b_idx];
    }
  }
  #pragma omp parallel for
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width - 1; ++x) {
      std::size_t b_idx = idx(x, y);
      hy[b_idx] = (hy[b_idx] + courant * (ez[idx(x + 1, y)] - ez[b_idx])) * damp[b_idx];
    }
  }
}


void FDTDGrid::updateE() {
  #pragma omp parallel for
  for (std::size_t y = 1; y < height - 1; ++y) {
    for (std::size_t x = 1; x < width - 1; ++x) {
      std::size_t k = idx(x, y);
      ez[k] = ez[k] + ((courant * ((hy[k] - hy[idx(x - 1, y)]) - (hx[k] - hx[idx(x, y - 1)])))) * damp[k] * !pec[k];
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
      : ez(width * height),
        hx(width * height),
        hy(width * height),
        damp(width * height, dampening),
        pec(width * height, 0),
        width(width),
        height(height),
        border(border),
        courant(courant){
// sponge: fields are slightly attenuated each step near the edges
for (std::size_t i = 0; i < width; ++i)
  for (std::size_t j = 0; j < height; ++j) {
    std::size_t d = std::min({i, j, width - 1 - i, height - 1 - j});
    if (d < border) {
      double x = static_cast<double>(border - d) / static_cast<double>(border);
      damp[idx(i, j)] = 1.0 - 0.1 * x * x * x;
    }
  }
} 


void FDTDGrid::update() {
  updateH();
  updateE();
}
