# RT C++ Practice Roadmap

Preparing for a C++ assignment with real-time / high-performance requirements.
Goal is solid base knowledge, not exotic C++.

## Where the project stands

The SPSC ring in `include/rt/ring_buffer_spsc.hpp` is finished as a learning
exercise. It is not production code and will not be extended further; the next
step is the logger below. What it covers:

- Power-of-two capacity with mask indexing, enforced by `static_assert`
- Cached opposite-end indices with lazy refresh, so the hot path rarely touches
  the other core's cache line
- Correct acquire/release pairing on the published indices
- Padding to a hard-coded 64-byte cache line
- Raw `std::byte` slot storage with placement `new` and explicit destruction
  (`include/rt/slot_storage.hpp`)
- `emplace(Args&&...)` with perfect forwarding, `push(const T&)` / `push(T&&)`
  with conditional `noexcept`
- Destructor that destroys remaining elements, explicitly deleted copy and move
- CPU affinity pinning and `_mm_pause` backoff in `apps/bench_spsc.cpp`
- CMake presets: debug, reldeb, release, asan, tsan
- Tests covering order, wrap-around, fullness, element destruction and a
  500K-item two-thread run that asserts the buffer actually filled

## Ring fixes (done)

- [X] `T buffer[N]` replaced by `alignas(T) std::byte` storage, placement `new`
      on push and an explicit `~T()` on pop.
- [X] `emplace(Args&&...)` with perfect forwarding, plus `const T&` and `T&&`
      push overloads.
- [X] `items()` uses relaxed loads.
- [X] Destructor and deleted copy and move.

Left open on purpose, since the ring is done: `capacity()` / `empty()` /
`full()`, compiling out the refresh counters, a direct `emplace` test, and
checking with `lscpu -e` that the bench's CPU 0 and CPU 2 are not hyperthread
siblings.

## C++ concepts to practice

Base knowledge, the kind an interviewer actually probes. Nothing exotic.
Checked items have been gone through; the notes record the key points.

### Lifetime and memory

- [X] Placement `new` and explicit destructor calls
- [X] `std::aligned_storage` versus raw `std::byte` plus `alignas`, and why
      `aligned_storage` is deprecated (C++23: easy to misuse by forgetting
      `::type`, vague wording; `alignas` is simpler)
- [X] `std::launder`: know it exists and roughly why
- [X] RAII, rule of 0/3/5, deleted copy and move. A user-declared destructor or
      copy suppresses the implicit move, which silently falls back to copy.
      Aim for rule of 0; only pure resource-managing classes need rule of 5.
- [X] Copy-and-swap: by-value `operator=` covers copy and move, strong
      exception guarantee, but always pays for a full copy
- [X] `std::unique_ptr` with a custom deleter (stateless deleter is zero size;
      a deleter can return to a pool instead of `free`), and why `shared_ptr`
      is usually wrong in real-time code (control-block allocation, atomic
      refcount, last owner pays for destruction on an unknown thread)
- [X] Ownership: one owner whose lifetime covers all users; non-owning views as
      `T*` or indices (no `vector<T&>`). `vector<unique_ptr<T>>` gives stable
      addresses; `vector<T>` plus indices is more cache friendly.
- [X] Custom allocators and `std::pmr`: `monotonic_buffer_resource` (bump
      pointer, free all at once, use `null_memory_resource()` upstream to
      forbid `malloc`) versus `*_pool_resource` (per-size free lists, for
      objects with independent lifetimes)

### Templates and compile time

- [X] `constexpr` functions
- [X] `if constexpr`: discarded branches are not instantiated, only inside a
      template (two-phase lookup)
- [X] SFINAE basics with `std::enable_if_t`
- [X] `static_assert` against type traits such as `is_trivially_copyable_v` and
      `is_nothrow_move_constructible_v`
- [X] Variadic templates and perfect forwarding: parameter packs and
      expansion, forwarding references, reference collapsing, `std::forward`
      as a conditional cast driven by the deduced type
- [ ] Fold expressions (needed for the logger)
- [ ] CRTP for static polymorphism, which replaces virtual calls in hot paths
- [ ] Tag dispatch

### Value semantics and correctness

- [X] Move semantics precisely: value categories belong to expressions, not
      objects; a named rvalue reference is an lvalue; `std::move` versus
      `std::forward`; references bind, they never own
- [ ] What a moved-from object is
- [ ] `noexcept`: what it actually guarantees, and where it changes codegen
      (`vector` growth, `move_if_noexcept`)
- [ ] `const` correctness, `mutable`, `[[nodiscard]]`
- [ ] `std::optional`, `std::variant`, `std::string_view`: the C++17 workhorses

### Concurrency

- [ ] The six memory orders, each with a concrete example of it being the wrong
      choice
- [X] Mutex ordering: `lock()` is acquire, `unlock()` is release; accesses can
      move into a critical section but not out
- [X] Atomic versus lock: atomics win for single values; not for large
      (non-lock-free) types, multi-variable invariants or many updates in a
      row. In real time the real win is that atomics never block.
- [ ] `compare_exchange_weak` versus `_strong`, the retry-loop idiom, the ABA
      problem
- [ ] `std::atomic<T>::is_lock_free` and `is_always_lock_free`
- [ ] `std::thread`, `jthread` (C++20, just know it), `mutex` / `lock_guard` /
      `unique_lock` / `condition_variable`, and when each is forbidden in a
      real-time thread
- [X] Function-local statics: one per function (per template instantiation),
      thread-safe initialization only; Meyers singleton
- [ ] `thread_local`
- [ ] Why `volatile` is not an atomic

### Tooling

- [ ] TSan, ASan, UBSan, and what each one catches
- [ ] `perf stat` and `perf record`; reading cache-miss and branch-miss counters
- [ ] Reading the `-O2` assembly of your own code on godbolt
- [ ] CMake beyond what you have: `FetchContent`, install and export rules

## Real-time concepts to practice

- [ ] Determinism over throughput: measure p99, p99.9 and max latency, not mean.
      The bench currently reports total ms, so switch it to a latency histogram
      (HdrHistogram-style, or a simple bucketed array).
- [ ] No unbounded operations in the real-time path: no `malloc` or `free`, no
      locks, no I/O, no exceptions, no `std::function` (heap), no `std::string`,
      no logging.
- [ ] Priority inversion, and priority-inheritance mutexes
      (`PTHREAD_PRIO_INHERIT`).
- [ ] `SCHED_FIFO` and `SCHED_RR` via `sched_setscheduler`, the `rlimit` setup
      they need, and the runaway-thread risk they carry.
- [ ] Page faults: `mlockall(MCL_CURRENT | MCL_FUTURE)`, pre-faulting the stack
      and heap at startup.
- [ ] CPU isolation: `isolcpus`, `nohz_full`, IRQ affinity, disabling turbo and
      C-states for measurement stability.
- [ ] Timing: `clock_gettime(CLOCK_MONOTONIC)` versus `rdtsc` versus
      `std::chrono::steady_clock`; TSC invariance.
- [ ] Jitter sources: cache misses, TLB misses, hugepages, NUMA locality, false
      sharing (already done), hyperthread siblings.
- [ ] Lock-free versus wait-free: this ring's push and pop are wait-free, while
      a CAS loop is only lock-free. Be able to state the difference out loud.
- [ ] Backpressure policies: block, drop-newest, drop-oldest, overwrite. Each is
      a different product decision, not just an implementation detail.
- [ ] Run `cyclictest` on this machine to see what the baseline jitter actually
      is.

## Next mini project: a real-time-safe logger

The most common real-time C++ exercise, and it reuses the ring directly.

**Design.** The real-time thread formats nothing and allocates nothing. It
writes a fixed-size POD record into the SPSC ring: a timestamp from `rdtsc`, a
`const char*` format string, and a small fixed payload of arguments. A
background thread pops records, formats them, and writes to a file.
Drop-on-full, with a dropped-count counter.

**Steps.**

- [ ] Fixed-size record and a variadic `LOG(fmt, args...)` that `memcpy`s each
      argument into the payload. Learn: fold expressions,
      `static_assert(std::is_trivially_copyable_v<Args>...)`.
- [ ] Decoding without type information: store a function pointer, instantiated
      from the argument types, that decodes and formats the payload. Learn:
      type erasure without `std::function` or the heap.
- [ ] Background thread: pop, format, write; clean shutdown that drains the
      ring.
- [ ] Measure: ns per log call, p50 / p99 / max, behaviour when the consumer
      falls behind.
- [ ] Optional: variable-size records in a byte ring.
- [ ] Optional: a `monotonic_buffer_resource` for the background thread's
      formatting, released per batch.

**Decisions to make first.**

- String arguments: `const char*` is only safe for string literals. Forbid
  other strings, or copy the bytes into the record.
- Format style: `printf`-style (simplest) or `{}`-style.
- Fixed versus variable records: start fixed.

**Lessons from reviewing `graiola/rt_logger`.** Keep: expensive setup outside
the real-time path, `trylock` and drop instead of blocking. Avoid: storing raw
pointers to the user's variables (dangling risk), iterating a `std::map` in the
hot path, copying whole messages with `std::string` / `std::vector` members
every cycle, unsynchronized add/remove versus publish, duplicated
implementations per platform.

## After that: MPSC, then MPMC

MPSC next: a single CAS loop on the write index. That teaches
`compare_exchange_weak`, the ABA problem, and why the "claim a slot, then
publish a per-slot sequence number" pattern (Vyukov, LMAX Disruptor) exists.

Full MPMC is a bigger jump, so do MPSC first and let MPMC follow from it.

## Alternative track

If a change of domain is preferable to extending the ring: a fixed-capacity,
allocation-free `static_vector<T, N>`, plus a bounded object pool with a
free-list.

Less exciting, but it hammers lifetime management, which is where C++ interviews
actually probe.
