#pragma once
#include <vector>

class FDTDGrid {
private:
  std::vector<double> ez, hx, hy, damp;
  std::vector<char> pec;  // 1 = metal cell
  std::size_t height, width, border;
  double courant;

  void updateH(); 
  void applyDamping();
  void updateE(); 
  inline std::size_t idx(std::size_t x, std::size_t y) { return y * width + x; }
  

public:
  FDTDGrid(
    std::size_t width, 
    std::size_t height,
    std::size_t border,
    double courant,
    double damp
  );
  void add_source(std::size_t x, std::size_t y, double value) {
    ez[y * width + x] += value;
  }
  std::vector<double> const& get_ez() const { return ez; };
  double get_ez(std::size_t x, std::size_t y) const { return ez[y * width + x]; }
  void set_pec(std::size_t x, std::size_t y, bool metal) {
    pec[y * width + x] = metal ? 1 : 0;
  }
  std::vector<char> const& get_pec() const { return pec; }
  void update(); 
};


