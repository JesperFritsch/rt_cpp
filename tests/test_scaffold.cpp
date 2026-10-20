// Placeholder test. Delete once the first real component has its own tests.
#include <rt/scaffold.hpp>

#include <gtest/gtest.h>

namespace {

TEST(Scaffold, RoundUpPow2) {
  EXPECT_EQ(rt::round_up_pow2(1), 1U);
  EXPECT_EQ(rt::round_up_pow2(2), 2U);
  EXPECT_EQ(rt::round_up_pow2(3), 4U);
  EXPECT_EQ(rt::round_up_pow2(100), 128U);
  static_assert(rt::round_up_pow2(1000) == 1024U);
}

}  // namespace
