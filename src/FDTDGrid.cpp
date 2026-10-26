#include <rt/FDTDGrid.hpp>
#include <cstdio>
#include <algorithm>

  
void FDTDGrid::updateH() {
  for (std::size_t i = 0; i < width; ++i)
    for (std::size_t j = 0; j < height - 1; ++j)
      hx[idx(i, j)] -= courant * (ez[idx(i, j + 1)] - ez[idx(i, j)]);
  for (std::size_t i = 0; i < width - 1; ++i)
    for (std::size_t j = 0; j < height; ++j)
      hy[idx(i, j)] += courant * (ez[idx(i + 1, j)] - ez[idx(i, j)]);
}

void FDTDGrid::applyDamping() {
  for (std::size_t k = 0; k < width * height; ++k) {
    ez[k] *= damp[k];
    hx[k] *= damp[k];
    hy[k] *= damp[k];
  }
}

void FDTDGrid::updateE() {
  for (std::size_t i = 1; i < width - 1; ++i)
    for (std::size_t j = 1; j < height - 1; ++j) {
      std::size_t k = idx(i, j);
      if (pec[k]) {
        ez[k] = 0.0;
        continue;
      }
      ez[k] += courant * ((hy[k] - hy[idx(i - 1, j)]) -
                            (hx[k] - hx[idx(i, j - 1)]));
    }
}
  

FDTDGrid::FDTDGrid(
  std::size_t width,
  std::size_t height,
  std::size_t border,
  double courant,
  double dampening = 1.0
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
      damp[idx(i, j)] = 1.0 - 0.5 * x * x * x;
    }
  }
} 


void FDTDGrid::update() {
  updateH();
  updateE();
  applyDamping();
}
