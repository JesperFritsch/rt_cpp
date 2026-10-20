
#include <rt/ring_buffer_spsc.hpp>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <atomic>
#include <immintrin.h>

using namespace std::chrono;

bool pin_to_cpu(std::size_t cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  const int ret = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (ret != 0) {
    std::fprintf(stderr, "Failed to pin to cpu %ld (%s)", cpu, std::strerror(ret));
    return false;
  }
  return true;
}

int main() {
  pin_to_cpu(0);
  constexpr int size = 4096;
  std::atomic<bool> stop{false};
  rt::SPSCRing<int, size> ring;
  std::uint64_t received = 0;
  std::uint64_t checksum = 0;
  bool order_ok = true;
  std::thread consumer([&] {
    pin_to_cpu(2);
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
  auto start_time = steady_clock::now(); 
  constexpr int n_items = 50000000U;
  for (int i = 0; i < n_items; ++i) {
    while(!ring.push(i)) { _mm_pause(); }
  }
  stop.store(true, std::memory_order_release);
  consumer.join();
  auto duration = steady_clock::now() - start_time;
  auto [empts, fulls] = ring.refreshes();
  printf("ms: %ld, f_r: %ld, e_r: %ld, recv: %ld\n", duration_cast<milliseconds>(duration).count(), fulls, empts, received);
}



