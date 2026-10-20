// Placeholder so the scaffold compiles and the test suite is non-empty.
// Delete this file (and tests/test_scaffold.cpp, apps/demo.cpp) once the
// first real component lands.
#pragma once

#include <cstddef>

namespace rt {

[[nodiscard]] constexpr std::size_t round_up_pow2(std::size_t n) noexcept {
  std::size_t pwr = 1;
  while (pwr < n) {
    pwr <<= 1U;
  }
  return pwr;
}

}  // namespace rt
