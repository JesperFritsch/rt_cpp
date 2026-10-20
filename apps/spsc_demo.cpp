#include <rt/ring_buffer_spsc.hpp>
#include <cstdio>
#include <cstddef>
#include <thread>
#include <chrono>
#include <atomic>
#include <random>
#include <csignal>
#include <optional>

using namespace std::chrono;

constexpr std::size_t ring_size = 1024;
std::atomic<bool> stop_flag{false};
std::atomic<bool> prod_done{false};

struct Status {
  char character;
  steady_clock::time_point time_stamp;
  int format(char* buf, std::size_t size, steady_clock::time_point since) const {
    const auto time = duration_cast<milliseconds>(time_stamp - since).count();
    return std::snprintf(buf, size, "%c: +%lldms", character, static_cast<long long>(time));
  }
};


int main() {
  std::signal(SIGINT, [](int){ stop_flag.store(true, std::memory_order_relaxed); });
  rt::SPSCRing<Status, ring_size> ring;
  std::thread consumer([&] {
    const std::size_t buf_size = 50;
    char buf[buf_size];
    steady_clock::time_point last_ts = steady_clock::now();
    while (true) {
      bool p_done = prod_done.load(std::memory_order_acquire);
      while ( auto status = ring.pop() ) {
        status->format(buf, buf_size, last_ts);
        last_ts = status->time_stamp;
        printf("\r%s\033[K", buf);
        std::fflush(stdout);
      }
      if (p_done) { break; }
    } 
    printf("\nConsumer finished!\n");
  });
  
  std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<int> s_dist(10, 200);
  std::uniform_int_distribution<int> c_dist(0, 25);
  while (!stop_flag.load(std::memory_order_relaxed)) {
    char character = static_cast<char>('A' + c_dist(rng));
    if (!ring.push({character, steady_clock::now()})) {
      printf("\r\033[KRingbuffer full!\n");
      std::fflush(stdout);
    }
    std::this_thread::sleep_for(milliseconds(s_dist(rng)));
  }
  prod_done.store(true, std::memory_order_release);
  consumer.join();
  printf("\n");

}
