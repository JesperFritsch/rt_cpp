
#include <rt/ring_buffer_spsc.hpp>
#include <rt/scaffold.hpp>
#include <thread>
#include <numeric>
#include <atomic>

#include <gtest/gtest.h>

namespace {

TEST(SPSCRing, test_order) {
  constexpr int size = 100;
  rt::SPSCRing<int, rt::round_up_pow2(size)> ring;
  for(int i = 0; i < size; ++i) {
    ASSERT_TRUE(ring.push(i)) << "push failed at " << i;
  }
  std::vector<int> out;
  out.reserve(size);
  while (auto item = ring.pop()) {
    out.push_back(*item);
  }
  std::vector<int> expected(size);
  std::iota(expected.begin(), expected.end(), 0); 
  EXPECT_EQ(out, expected);
}  // namespace

TEST(SPSCRing, size_correct) {
  constexpr int size = 100;
  rt::SPSCRing<int, rt::round_up_pow2(size)> ring;
  for (int i = 0; i < size; ++i) {
    ASSERT_TRUE(ring.push(i)) << "push failed at " << i;
  }
  EXPECT_EQ(ring.items(), size);
}

TEST(SPSCRing, ordered_wrap) {
  constexpr int size = 100;
  constexpr int buff_size = rt::round_up_pow2(size);
  rt::SPSCRing<int, buff_size> ring;
  // First write enough elements to put head and tail in the middle of the buffer, so order can
  // be checked at wrap around.
  for(int i = 0; i < size + buff_size; ++i) {
    ASSERT_TRUE(ring.push(i)) << "push failed at " << i;
    ASSERT_EQ(ring.pop(), i) << "popped is not the same as inserted at" << i;
  }
  for(int i = 0; i < size; ++i) {
    ASSERT_TRUE(ring.push(i)) << "push failed at " << i;
  }
  std::vector<int> out;
  out.reserve(size);
  while (auto item = ring.pop()) {
    out.push_back(*item);
  }
  std::vector<int> expected(size);
  std::iota(expected.begin(), expected.end(), 0); 
  EXPECT_EQ(out, expected);
}  // namespace

TEST(SPSCRing, correct_full) {
  constexpr int size = 128;
  rt::SPSCRing<int, size> ring;
  for (int i = 0; i < size; ++i) {
    ASSERT_TRUE(ring.push(i)) << "push failed at " << i;
  }
  EXPECT_EQ(ring.items(), size);
  EXPECT_TRUE(!ring.push(1));
}

TEST(SPSCRingPerf, 500K_items) {
  constexpr int size = 4096;
  std::atomic<bool> stop{false};
  rt::SPSCRing<int, size> ring;
  std::uint64_t full_count = 0, received = 0, checksum = 0;
  bool order_ok = true;
  std::thread consumer([&] {
    while (true) {
      bool stopped = stop.load(std::memory_order_acquire);
      while (auto item = ring.pop()) {
        if (received != item) { order_ok = false; }
        ++received;
        checksum += static_cast<std::uint64_t>(*item);
      }
      if (stopped) { break; }
    }
  });
  constexpr int n_items = 500000U;
  for (int i = 0; i < n_items; ++i) {
    while(!ring.push(i)) { ++full_count; }
  }
  stop.store(true, std::memory_order_release);
  consumer.join();
  ASSERT_EQ(ring.items(), 0) << "ring was not drained";
  ASSERT_EQ(n_items, received);
  ASSERT_TRUE(order_ok) << "order was not preserved";
  ASSERT_GT(full_count, 0) << "buffer was never full, preassure never tested";
}

struct Counter {
  static inline std::size_t live = 0;
  Counter() {++live;}
  Counter(Counter&&) {++live;}
  ~Counter() {--live;}
};

TEST(SPSCRing, destruct_items) {

  constexpr int size = 8;
  {
    rt::SPSCRing<Counter, size> ring;
    for (int i = 0; i < size - 2; ++i) {
      ASSERT_TRUE(ring.push(Counter{})) << "push failed at " << i;
    }
    if (ring.pop()) {};
    if (ring.pop()) {};
    EXPECT_EQ(ring.items(), 4);
  }
  EXPECT_EQ(Counter::live, 0);
}


}
