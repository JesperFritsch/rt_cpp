
#pragma once
#include <cstddef>
#include <atomic>
#include <optional>
#include <utility>
#include <type_traits>
#include <thread>
#include <cassert>
#include <array>
#include <cstdint>
#include <rt/slot_storage.hpp>



namespace rt {

template<typename T, std::size_t N>
class MPSCRing {
  static_assert(N > 0, "Size must be larger than zero");
  static_assert((N & (N - 1)) == 0, "Size must be a power of two");
public:
  MPSCRing() noexcept {
    for (std::size_t i = 0; i < N; ++i) {
      slot_seqs[i].store(i, std::memory_order_relaxed);
    }
  }
  
  MPSCRing(const MPSCRing&) = delete;
  MPSCRing& operator=(const MPSCRing&) = delete;
  MPSCRing(MPSCRing&&) = delete;
  MPSCRing& operator=(MPSCRing&&) = delete;

  template<typename... Args>
  [[nodiscard]]
  bool emplace(Args&&... args) noexcept 
  {
    static_assert(std::is_nothrow_constructible_v<T, Args...>, "Constructor must not throw");
    assert(!closed_.load(std::memory_order_acquire));

    std::size_t curr_write = write_.load(std::memory_order_relaxed);
    if (is_full(curr_write)) { return false; } 
    while (!claim_slot(write_, curr_write)) {
      if (is_full(curr_write)) { return false; }
    }
    storage_.construct(curr_write & mask_, std::forward<Args>(args)...);
    slot_seqs[curr_write & mask_].store(curr_write + 1, std::memory_order_release);
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
    slot_seqs[curr_read & mask_].store(curr_read + N, std::memory_order_release);
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

  void wait_not_empty() noexcept {
    std::size_t curr_read = read_.load(std::memory_order_relaxed);
    while(is_empty(curr_read) && !closed_.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  void close() noexcept {
    closed_.store(true, std::memory_order_release); 
  }

  ~MPSCRing() {
    const std::size_t read = read_.load(std::memory_order_relaxed);
    const std::size_t write = write_.load(std::memory_order_relaxed);
    for (std::size_t i = read; i < write; ++i) {
      storage_.destroy(i & mask_);
    }
  }

private:

  bool is_full(std::size_t curr_write) {
    return static_cast<std::intptr_t>(slot_seqs[curr_write & mask_].load(std::memory_order_acquire) - curr_write) < 0;
  }

  bool is_empty(std::size_t curr_read) {
    return static_cast<std::intptr_t>(slot_seqs[curr_read & mask_].load(std::memory_order_acquire) - curr_read) <= 0;
  }

  bool claim_slot(std::atomic<std::size_t>& pos_counter, std::size_t& curr_val) {
    return pos_counter.compare_exchange_weak(
      curr_val,
      curr_val + 1,
      std::memory_order_relaxed
    );
  }

  static constexpr std::size_t mask_ = N - 1;
  static constexpr std::size_t cacheline_ = 64;
  alignas(cacheline_) std::atomic<std::size_t> write_{0}; //the producer end
  alignas(cacheline_) std::atomic<std::size_t> read_{0}; //the consumer end
  alignas(cacheline_) std::array<decltype(write_), N> slot_seqs;
  alignas(cacheline_) std::atomic<bool> closed_{false};
  alignas(cacheline_) SlotStorage<T, N> storage_;
};
}
