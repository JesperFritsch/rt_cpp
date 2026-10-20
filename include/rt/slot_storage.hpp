#pragma once
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace rt {

// Indexed storage for type T where only occupied slots actually contains objects. the rest is std::byte.
template<typename T, std::size_t N>
class SlotStorage {
  static_assert(N > 0, "Size must be larger than 0");
public:
  SlotStorage() = default;
  ~SlotStorage() = default;
  
  SlotStorage(const SlotStorage&) = delete;
  SlotStorage& operator=(const SlotStorage&) = delete;

  [[nodiscard]] static constexpr std::size_t slots() noexcept { return N; }

  template<typename... Args>
  T* construct(std::size_t idx, Args&&... args)
    noexcept(std::is_nothrow_constructible_v<T, Args...>) {
    return ::new (static_cast<void*>(raw(idx))) T(std::forward<Args>(args)...);
  }

  void destroy(std::size_t idx) noexcept { get(idx)->~T(); }

  [[nodiscard]] T* get(std::size_t idx) noexcept {
    return std::launder(reinterpret_cast<T*>(raw(idx)));
  }
  [[nodiscard]] const T* get(std::size_t idx) const noexcept {
    return std::launder(reinterpret_cast<const T*>(raw(idx)));
  }

  
private:
  [[nodiscard]] std::byte* raw (std::size_t idx) noexcept {
    return bytes_ + (idx * sizeof(T));
  }

  [[nodiscard]] const std::byte* raw(std::size_t idx) const noexcept {
    return bytes_ + (idx * sizeof(T));
  }
  
  alignas(T) std::byte bytes_[N * sizeof(T)];

};
}
