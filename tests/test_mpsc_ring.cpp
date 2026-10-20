

#include <rt/ring_buffer_mpsc.hpp>
#include <rt/scaffold.hpp>
#include <thread>
#include <numeric>
#include <atomic>
#include <numeric>
#include <gtest/gtest.h>

namespace {

TEST(MPSCRing, test_order) {
  constexpr int size = 100;
  rt::MPSCRing<int, rt::round_up_pow2(size)> ring;
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

TEST(MPSCRing, size_correct) {
  constexpr int size = 100;
  rt::MPSCRing<int, rt::round_up_pow2(size)> ring;
  for (int i = 0; i < size; ++i) {
    ASSERT_TRUE(ring.push(i)) << "push failed at " << i;
  }
  EXPECT_EQ(ring.items(), size);
}

TEST(MPSCRing, ordered_wrap) {
  constexpr int size = 100;
  constexpr int buff_size = rt::round_up_pow2(size);
  rt::MPSCRing<int, buff_size> ring;
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

TEST(MPSCRing, correct_full) {
  constexpr int size = 128;
  rt::MPSCRing<int, size> ring;
  for (int i = 0; i < size; ++i) {
    ASSERT_TRUE(ring.push(i)) << "push failed at " << i;
  }
  EXPECT_EQ(ring.items(), size);
  EXPECT_TRUE(!ring.push(1));
}

TEST(MPSCRingPerf, 500K_items) {
  constexpr int size = 4096;
  std::atomic<bool> stop{false};
  rt::MPSCRing<int, size> ring;
  std::uint64_t full_count = 0, received = 0, checksum = 0;
  std::atomic<bool> was_full{false};
  bool order_ok = true;
  std::thread consumer([&] {
    while (!was_full.load(std::memory_order_relaxed)) {std::this_thread::yield();};
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
    while(!ring.push(i)) { ++full_count; was_full.store(true, std::memory_order_relaxed);}
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
  Counter() noexcept {++live;}
  Counter(Counter&&) noexcept {++live;}
  ~Counter() {--live;}
};

TEST(MPSCRing, destruct_items) {

  constexpr int size = 8;
  {
    rt::MPSCRing<Counter, size> ring;
    for (int i = 0; i < size - 2; ++i) {
      ASSERT_TRUE(ring.push(Counter{})) << "push failed at " << i;
    }
    if (ring.pop()) {};
    if (ring.pop()) {};
    EXPECT_EQ(ring.items(), 4);
  }
  EXPECT_EQ(Counter::live, 0);
}

TEST(MPSCRing, many_producers) {
  struct Product {
    std::size_t producer_id;
    std::size_t counter;
  };
  rt::MPSCRing<Product, 8> ring;
  constexpr std::size_t n_prods = 4;
  constexpr std::size_t items_per_prod = 10000;
  std::vector<std::thread> producers;
  producers.reserve(n_prods);
  std::atomic<std::size_t> cont_count{0};
  std::atomic<std::size_t> prods_done{0};
  for (std::size_t id = 0; id < n_prods; ++id) {
    producers.emplace_back([&, id] {
      for (std::size_t i = 0; i < items_per_prod; ++i) {
        while (!ring.push({id, i})) {cont_count.fetch_add(1, std::memory_order_relaxed);}
      }
      prods_done.fetch_add(1);
    });
  }
  std::vector<std::size_t> expected_counts(n_prods, 0);
  while (true) {
    auto is_done = prods_done.load(std::memory_order_acquire) == n_prods;
    auto item = ring.pop();
    if (item.has_value()) {
      auto& count = expected_counts[item->producer_id];
      EXPECT_EQ(item->counter, count) << "out of order item:" << count << " != " << item->counter << " for prod: " << item->producer_id;
      ++count;
    } else if (is_done){
      break;
    }
  }
  for (auto& prod : producers) {
    prod.join();
  }
  for (auto cnt : expected_counts) {
    EXPECT_EQ(items_per_prod, cnt);
  }
  EXPECT_TRUE(cont_count.load() != 0) << "no contention test was not accurate";
}

TEST(MPSCRing, consumer_exits) {
  std::thread thr;
  rt::MPSCRing<int, 8> ring;
  thr = std::thread([&] {
    (void)ring.pop_blocking();
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ring.close();
  thr.join();
}

}
