#include <rt/rt_logger.hpp>
#include <gtest/gtest.h>

namespace {

TEST(RTLogger, record_order) {
  int const iters = 5; 
  rt::RTLogger lgr{}; 
  
  auto itr = lgr.line_drain();
  for (int i = 0; i < iters; i++) {
    lgr.log("%d", i);
  }
  int i = 0;
  for (auto line : itr) {
    EXPECT_EQ(line, std::to_string(i));
    ++i;
  }
  EXPECT_EQ(i, iters);
}

TEST(RTLogger, consumer_stops) {
  {
    // Test that the consumer stops when there has been no loggings
    rt::RTLogger lgr{};
    (void)lgr.start_consumer([](auto lines) {
      for (auto line : lines) {
        (void)line;
      }
    });
  }
    std::atomic<std::size_t> count{0};
  {
    // test that the consumer stops when there has been loggings
    rt::RTLogger lgr{};
    (void)lgr.start_consumer([&](auto lines) { for (auto line : lines) { EXPECT_EQ("hejsan", line); ++count;} });
    for (int i = 0; i < 100; ++i) { lgr.log("hejsan"); }
  }
  EXPECT_EQ(count, 100);
}

TEST(RTLogger, drops_counted) {
  rt::RTLogger<8> lgr{};
  for (int i = 0; i < 10; ++i) {
    lgr.log("hej");
  }
  EXPECT_EQ(lgr.dropped_count(), 2);
}


}
