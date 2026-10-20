

#include <rt/ring_buffer_spsc.hpp>
#include <cstdio>
#include <chrono>
#include <pthread.h>
#include <sched.h>
#include <cstring>

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
  rt::SPSCRing<int, size> ring;
  auto start_time = steady_clock::now(); 
  constexpr int n_items = 5000000U;
  // fill the buffer
  while (ring.push(1)) {}
  for (int i = 0; i < n_items; ++i) {
    if (!ring.pop().has_value()) {printf("fucked up");}
    if (!ring.push(i)) {printf("fucked up");}
  }
  auto duration = steady_clock::now() - start_time;
  auto [empts, fulls] = ring.refreshes();
  printf("ms: %ld, f_r: %ld, e_r: %ld\n", duration_cast<milliseconds>(duration).count(), fulls, empts);
}



