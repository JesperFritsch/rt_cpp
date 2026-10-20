#pragma once
#include <cstddef>
#include <atomic>
#include <optional>
#include <utility>
#include <type_traits>
#include <thread>
#include <cassert>
#include <unistd.h>

#include <rt/slot_storage.hpp>

struct Refreshes {
  std::size_t empty;
  std::size_t full;
};

namespace rt {

template<typename T, std::size_t N>
class SPSCRing {
  static_assert(N > 0, "Size must be larger than zero");
  static_assert((N & (N - 1)) == 0, "Size must be a power of two");
public:
  SPSCRing() = default;
  
  SPSCRing(const SPSCRing&) = delete;
  SPSCRing& operator=(const SPSCRing&) = delete;
  SPSCRing(SPSCRing&&) = delete;
  SPSCRing& operator=(SPSCRing&&) = delete;

  template<typename... Args>
  [[nodiscard]]
  bool emplace(Args&&... args) 
  noexcept(std::is_nothrow_constructible_v<T, Args...>) 
  {
    assert(!closed_.load(std::memory_order_acquire));
    std::size_t curr_write = write_.load(std::memory_order_relaxed);
    if (is_full(curr_write)) { return false; } 
    storage_.construct(curr_write & mask_, std::forward<Args>(args)...);
    write_.store(curr_write + 1, std::memory_order_release);
    return true;
  }


  [[nodiscard]]
  bool push(const T& item) 
  noexcept(std::is_nothrow_copy_constructible_v<T>) 
  {
    return emplace(item);
  }


  [[nodiscard]]
  bool push(T&& item) 
  noexcept(std::is_nothrow_move_constructible_v<T>) 
  {
    return emplace(std::move(item));
  }

  [[nodiscard]]
  std::optional<T> pop()
  noexcept(std::is_nothrow_move_constructible_v<T>) 
  {
    std::size_t curr_read = read_.load(std::memory_order_relaxed);
    if (is_empty(curr_read)) { return {}; }
    std::size_t idx = curr_read & mask_;
    T item = std::move(*storage_.get(idx));
    storage_.destroy(idx);
    read_.store(curr_read + 1, std::memory_order_release);
    return item;
  }

  [[nodiscard]]
  std::optional<T> pop_blocking()
  noexcept(std::is_nothrow_move_constructible_v<T>) 
  {
    wait_not_empty();
    return pop();
  }

  [[nodiscard]]
  std::size_t items() const noexcept {
    return write_.load(std::memory_order_relaxed) - read_.load(std::memory_order_relaxed);
  }

  [[nodiscard]]
  Refreshes refreshes() const noexcept {
    return {empt_refresh, full_refresh};
  }

  void wait_not_empty() noexcept {
    std::size_t curr_read = read_.load(std::memory_order_relaxed);
    while(is_empty(curr_read) && !closed_.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  void close() noexcept {
    closed_.store(true, std::memory_order_release); 
  }

  ~SPSCRing() {
    const std::size_t read = read_.load(std::memory_order_relaxed);
    const std::size_t write = write_.load(std::memory_order_relaxed);
    for (std::size_t i = read; i < write; ++i) {
      storage_.destroy(i & mask_);
    }
  }

private:

  bool is_full(std::size_t curr_write) {
    if (curr_write - cached_read_ >= N) {
      full_refresh++;
      cached_read_ = read_.load(std::memory_order_acquire);
      return curr_write - cached_read_ >= N;
    }
    return false;
  };

  bool is_empty(std::size_t curr_read) {
    if (cached_write_ - curr_read == 0) {
      empt_refresh++;
      cached_write_ = write_.load(std::memory_order_acquire);
      return cached_write_ - curr_read == 0;
    }
    return false;
  };

  static constexpr std::size_t mask_ = N - 1;
  static constexpr std::size_t cacheline_ = 64;
  alignas(cacheline_) std::size_t cached_write_ = 0;
  alignas(cacheline_) std::size_t cached_read_ = 0;
  alignas(cacheline_) std::size_t full_refresh = 0;
  alignas(cacheline_) std::size_t empt_refresh = 0;
  alignas(cacheline_) std::atomic<std::size_t> write_{0}; //the producer end
  alignas(cacheline_) std::atomic<std::size_t> read_{0}; //the consumer end
  alignas(cacheline_) std::atomic<bool> closed_{false};
  alignas(cacheline_) SlotStorage<T, N> storage_;
};
}
