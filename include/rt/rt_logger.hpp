#pragma once
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <cstring>
#include <cstdio>
#include <tuple>
#include <algorithm>
#include <string_view>
#include <x86intrin.h>
#include <rt/ring_buffer_spsc.hpp>

namespace rt {

constexpr std::size_t record_p_size = 64; 
constexpr std::size_t record_fields = 24;

struct Sentinel {};


struct 
[[nodiscard]] 
Record {
  uint64_t tsc;
  const char* fmt;
  int (*decode)(const Record&, char*, std::size_t);
  std::byte payload[record_p_size - record_fields];
};

template<typename T>
T read(const std::byte* buf, std::size_t& off) {
  T val; 
  std::memcpy(&val, buf + off, sizeof(T));
  off += sizeof(T);
  return val;
}

template<typename ...Args> 
int decode_impl(const Record& rec, char* out, std::size_t n) {
  [[maybe_unused]] std::size_t off = 0;
  std::tuple<Args...> vals{ read<Args>(rec.payload, off)... };
  int len;
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wformat-nonliteral"
  #pragma GCC diagnostic ignored "-Wformat-security"
  std::apply([&] (auto ...args) { len = std::snprintf(out, n, rec.fmt, args...); }, vals);
  #pragma GCC diagnostic pop
  return len;
}

template<typename ...Args>
Record encode(const char* fmt, Args... args) {
    Record rec;
    rec.tsc = __rdtsc();
    rec.fmt = fmt;
    rec.decode = &decode_impl<Args...>;

    [[maybe_unused]] std::size_t off = 0;
    ((std::memcpy(rec.payload + off, &args, sizeof(Args)), off += sizeof(args)), ...);
    return rec;
}

template<std::size_t RING_SIZE = 512>
class RTLogger {
public:
  explicit RTLogger() = default;
  RTLogger(const RTLogger&) = delete;
  RTLogger& operator=(const RTLogger&) = delete;
  RTLogger(RTLogger&&) = delete;
  RTLogger& operator=(RTLogger&&) = delete;
  ~RTLogger() {
    ring_.close();
    if (consumer_.joinable()) {
      consumer_.join();
    }
  }
 
  template<bool Blocking>
  class BasicLineIterator;

  template<bool Blocking>
  class LineRange{
  public:
    explicit LineRange(RTLogger* lgr) : lg_(lgr) {}
    BasicLineIterator<Blocking>  begin() const;
    Sentinel      end()   const { return {}; }
  private:
    RTLogger* lg_;
  }; 

  using Lines = LineRange<true>;
  using DrainLines = LineRange<false>;

  Lines line_iter() {
    return Lines{this};
  }

  DrainLines line_drain() {
    return DrainLines{this};
  }

  template<typename F>
  [[nodiscard]] bool start_consumer(F&& call) {
    if (consumer_.joinable()) { return false; }
    static_assert(std::is_invocable_v<std::decay_t<F>&, Lines>, "consumer must be callable with Lines");
    consumer_ = std::thread([this, call = std::forward<F>(call)]() mutable {
      call(line_iter());
    });
    return true;
  }

  template<typename ...Args>
  void log(const char* fmt, Args... args){
    static_assert((std::is_trivially_copyable_v<Args> && ...), "Not all args are trivially copyable");
    static_assert((0 + ... + sizeof(args)) <= sizeof(Record::payload), "Record payload to big");
    auto rec = encode<Args...>(fmt, args...);
    if (!ring_.push(rec)) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  std::size_t dropped_count() const noexcept {
    return dropped_.load(std::memory_order_relaxed);
  }


private:
  std::atomic<std::size_t> dropped_{0};
  rt::SPSCRing<Record, RING_SIZE> ring_;
  std::thread consumer_;

};

template<std::size_t RING_SIZE>
template<bool Blocking>
class RTLogger<RING_SIZE>::BasicLineIterator {
public: 
  using iterator_category = std::input_iterator_tag;
  using value_type        = std::string_view;
  using difference_type   = std::ptrdiff_t;
  using pointer           = std::string_view*;
  using reference         = std::string_view&;
  static const std::size_t buff_size = 512;

  explicit BasicLineIterator(RTLogger* lg_) : lgr_(lg_) { advance(); }
  
  std::string_view operator*() const { return std::string_view(string_buff, str_len); }
  BasicLineIterator& operator++() { advance(); return *this; }
  void operator++(int) { advance(); }
  bool operator==(Sentinel) const { return done_; }
  bool operator!=(Sentinel) const { return !done_; }

private:
  
  void advance() {
    std::optional<Record> rec;
    if constexpr (Blocking) {
      rec = lgr_->ring_.pop_blocking();
    } else {
      rec = lgr_->ring_.pop();
    }
    if(rec) {
      int len = rec->decode(*rec, string_buff, buff_size);
      str_len = len < 0 ? 0 : std::min(static_cast<std::size_t>(len), buff_size - 1);
    } else {
      done_ = true; return;
    }
  }
  RTLogger* lgr_;
  char string_buff[buff_size];
  std::size_t str_len=0;
  bool done_ = false;
};

template<std::size_t RING_SIZE>
inline rt::RTLogger<RING_SIZE>& get_logger() {
  static rt::RTLogger<RING_SIZE> logger;
  return logger;
}

}

template<std::size_t RING_SIZE>
template<bool Blocking>
inline rt::RTLogger<RING_SIZE>::BasicLineIterator<Blocking> rt::RTLogger<RING_SIZE>::LineRange<Blocking>::begin() const { return BasicLineIterator<Blocking>{lg_}; }


