#pragma once
#include <vector>

class FDTDGrid {
private:
  // ez: electric field along z (out of the screen); hx, hy: magnetic field along x and y
  // damp: per-cell multiplier, < 1 in the sponge border
  std::vector<float> ez_, hx_, hy_, damp_;
  std::vector<char> pec_;  // perfect electric conductor, 1 = metal cell
  std::size_t width_, height_, border_;
  float courant_;  // cells the wave moves per step, must be < 1/sqrt(2) in 2D

  void updateH();
  void applyDamping();
  void updateE();
  // index into the flat arrays for cell (x, y)
  inline std::size_t idx(std::size_t x, std::size_t y) { return y * width_ + x; }
  

public:
  FDTDGrid(
    std::size_t width, 
    std::size_t height,
    std::size_t border,
    float courant,
    float damp
  );

  void add_source(std::size_t x, std::size_t y, float value) {
    ez_[y * width_ + x] += value;
  }

  std::vector<float> const& get_ez() const { return ez_; };

  float get_ez(std::size_t x, std::size_t y) const { return ez_[y * width_ + x]; }

  void set_pec(std::size_t x, std::size_t y, bool metal) {
    pec_[y * width_ + x] = metal ? 1 : 0;
  }
 
  void update(); 
};


