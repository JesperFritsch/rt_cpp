#include <rt/rt_logger.hpp>
#include <pthread.h>

#include <chrono>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <array>
#include <x86intrin.h>
#include <immintrin.h>


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


inline std::uint64_t tsc_start() {
  _mm_lfence();
  std::uint64_t tsc = __rdtsc();
  _mm_lfence();
  std::atomic_signal_fence(std::memory_order_seq_cst);
  return tsc;
}

inline std::uint64_t tsc_stop() {
  std::atomic_signal_fence(std::memory_order_seq_cst);
  _mm_lfence();
  return __rdtsc();
}

template<typename F>
void measure(std::vector<std::uint64_t>& samples, F&& body, std::size_t warmup, std::uint64_t wait_ticks) {
  for (std::size_t i = 0; i < warmup; ++i) { body(); }
  std::uint64_t next = __rdtsc();
  for (auto& spl : samples) {
    auto tsc0 = tsc_start();
    body();
    auto tsc1 = tsc_stop();
    spl = tsc1 - tsc0;
    while (__rdtsc() < next) { _mm_pause(); }
    next += wait_ticks;
  }
}

double ticks_per_ns() {
  auto start = std::chrono::steady_clock::now();
  auto tsc0 = __rdtsc();
  while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(200)) {}
  auto end = std::chrono::steady_clock::now();
  auto nanos = std::chrono::duration<double, std::nano>(end - start).count();
  auto tsc1 = __rdtsc();
  return double(tsc1 - tsc0) / nanos;
}

double ticks_to_ns(std::uint64_t ticks, double t_ns) {
  return static_cast<double>(ticks) / t_ns;
}

std::pair<double, double> bucket_range(std::size_t bucket, double t_ns) {
  return {ticks_to_ns(1ull << bucket, t_ns), ticks_to_ns((1ull << (bucket + 1)) - 1, t_ns)};
}

std::uint64_t ptile(std::vector<std::uint64_t>& samples, double ptile) {
  auto idx = std::min(samples.size(), static_cast<std::size_t>(samples.size() * ptile));
  return samples[idx];
}

void print_histogram(std::vector<std::uint64_t> samples) {
  std::array<std::size_t, 64> buckets;
  for (auto& bucket : buckets) {
    bucket = 0;
  }
  for (auto& sample : samples) {
    std::size_t bucket = sample > 0 ? 63 - static_cast<std::size_t>(__builtin_clzll(sample)) : 0;
    ++buckets[bucket];
  }
  std::size_t idx = 0;
  auto t_ns = ticks_per_ns();
  printf("min: %.2f, max: %.2f\n", ticks_to_ns(*samples.begin(), t_ns), ticks_to_ns(*--samples.end(), t_ns));
  char label[16];
  for (auto ptl : {0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999}) {
    std::snprintf(label, sizeof(label), "p%g:", ptl * 100);
    auto spl = ptile(samples, ptl);
    printf("%-8s %8.2f\n", label, ticks_to_ns(spl, t_ns));
  }
  for (auto bucket : buckets) {
    if (bucket > 0) {
      auto ns_r = bucket_range(idx, t_ns);
      printf("%8.2f-%8.2fns: %ld\n", ns_r.first, ns_r.second, bucket);
    }
    ++idx;
  }
}

int main() {
  pin_to_cpu(2);
  constexpr std::size_t nr_spl = 1000000;
  constexpr std::uint64_t wait = 1;
  std::vector<std::uint64_t> samples(nr_spl);
  
  auto run = [&](const char* name, auto&& body, std::uint64_t wait_ticks) {
    measure(samples, body, 4096, wait_ticks);
    std::sort(samples.begin(), samples.end());
    printf("== %s == \n", name);
    print_histogram(samples);
  };

  run("baseline", []{}, 0);

  {
    rt::RTLogger lgr;
    run("no consumer", [&] { lgr.log("log"); }, 0);
    printf("dropped: %ld\n", lgr.dropped_count());
  }
  {
    rt::RTLogger<4096> lgr;
    (void)lgr.start_consumer([&](auto lines){ 
      pin_to_cpu(2);
      for(auto line : lines){ (void)line; } 
    });
    pin_to_cpu(0);
    run("Consumer running", [&]{ lgr.log("log"); }, wait);  
    printf("dropped: %ld\n", lgr.dropped_count());
  }
}

