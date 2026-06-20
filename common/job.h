#ifndef JOB_H
#define JOB_H

#include <coroutine>
#include <cstdint>
#include <exception>
#include <utility>

namespace vpipe {

// Opaque profiling tag carried by every pool coroutine. The pipeline
// layer sets it (Job::set_perf_tag) to the recording stage's
// graph-vertex id; the ThreadPool reads it (perf_tag_of) to attribute
// each worker resume-slice to a stage without the pool needing to know
// what a stage is. kNoPerfTag = untagged (e.g. the log-delegate
// consumer coroutine) -- the pool skips perf events for those.
inline constexpr std::uint32_t kNoPerfTag = 0xFFFFFFFFu;

// Coroutine return type for Stage::process() and any other async
// work that runs on the Session ThreadPool. Move-only RAII over a
// coroutine_handle. Implements the standard async-coroutine pattern:
//
//   * A Job is created suspended at initial_suspend.
//   * Either it is started directly by scheduling its handle onto a
//     worker (the per-stage driver path), or it is co_await'd by
//     another coroutine, which records itself as the continuation
//     and symmetric-transfers control to the inner coroutine.
//   * On completion, FinalAwaiter symmetric-transfers control back to
//     the recorded continuation (or to noop_coroutine for a detached
//     Job).
//
// Lifetime contract: ~Job() calls .destroy() on the handle. The
// caller must ensure the coroutine is suspended at initial_suspend or
// final_suspend (i.e. not mid-flight on a buffer wait list or running
// on a worker) before the Job is destroyed. The PipelineRuntime
// enforces this via wait_idle() before driver teardown.
//
// Thread safety: a Job object is not internally synchronized; it is
// owned by one thread at a time. The coroutine state itself may move
// between worker threads as the chain of symmetric transfers and
// re-schedules unfolds; that synchronization lives inside the
// individual awaiters (EdgeBuffer, etc.).
class Job {
public:
  struct promise_type;
  using handle_t = std::coroutine_handle<promise_type>;

  struct promise_type {
    std::coroutine_handle<> _continuation{};
    std::exception_ptr      _eptr{};
    std::uint32_t           _perf_tag = kNoPerfTag;

    Job
    get_return_object() noexcept
    {
      return Job{handle_t::from_promise(*this)};
    }

    std::suspend_always initial_suspend() noexcept { return {}; }

    // Symmetric-transfer back to the continuation on completion.
    // For a detached Job (no continuation set) we land at
    // noop_coroutine; the Job's destructor will reclaim the frame.
    struct FinalAwaiter {
      bool await_ready() const noexcept { return false; }

      std::coroutine_handle<>
      await_suspend(handle_t h) noexcept
      {
        auto& p = h.promise();
        if (p._continuation) {
          return p._continuation;
        }
        return std::noop_coroutine();
      }

      void await_resume() const noexcept {}
    };

    FinalAwaiter final_suspend() noexcept { return {}; }

    void return_void() noexcept {}

    void
    unhandled_exception() noexcept
    {
      _eptr = std::current_exception();
    }
  };

  Job() noexcept = default;

  Job(const Job&)            = delete;
  Job& operator=(const Job&) = delete;

  Job(Job&& o) noexcept
    : _h(std::exchange(o._h, {}))
  {}

  Job&
  operator=(Job&& o) noexcept
  {
    if (this != &o) {
      if (_h) {
        _h.destroy();
      }
      _h = std::exchange(o._h, {});
    }
    return *this;
  }

  ~Job()
  {
    if (_h) {
      _h.destroy();
    }
  }

  // Read-only handle accessor for ThreadPool scheduling. The caller
  // does NOT take ownership; they only .resume() the handle to let
  // the coroutine run on a worker thread.
  std::coroutine_handle<> handle() const noexcept { return _h; }

  // Tag this (suspended) coroutine with an opaque profiling id. The
  // pipeline driver sets it to the stage's gvid so the ThreadPool can
  // attribute resume-slices to a stage. No-op on an empty Job.
  void
  set_perf_tag(std::uint32_t tag) noexcept
  {
    if (_h) { _h.promise()._perf_tag = tag; }
  }

  bool done()  const noexcept { return !_h || _h.done(); }
  bool valid() const noexcept { return static_cast<bool>(_h); }

  // Awaiter for `co_await job`. Takes ownership of the handle out of
  // the (rvalue) Job; its destructor destroys the inner coroutine
  // when control returns past the co_await.
  struct Awaiter {
    handle_t _h{};

    Awaiter() noexcept = default;
    explicit Awaiter(handle_t h) noexcept : _h(h) {}

    Awaiter(const Awaiter&)            = delete;
    Awaiter& operator=(const Awaiter&) = delete;

    Awaiter(Awaiter&& o) noexcept
      : _h(std::exchange(o._h, {}))
    {}

    ~Awaiter()
    {
      if (_h) {
        _h.destroy();
      }
    }

    bool await_ready() const noexcept { return !_h || _h.done(); }

    // Record the caller as the continuation, then symmetric-transfer
    // to the inner coroutine to start it.
    std::coroutine_handle<>
    await_suspend(std::coroutine_handle<> caller) noexcept
    {
      _h.promise()._continuation = caller;
      return _h;
    }

    void
    await_resume()
    {
      if (auto& p = _h.promise(); p._eptr) {
        std::rethrow_exception(p._eptr);
      }
    }
  };

  Awaiter
  operator co_await() && noexcept
  {
    return Awaiter{std::exchange(_h, {})};
  }

private:
  explicit Job(handle_t h) noexcept : _h(h) {}

  handle_t _h{};
};

// Read the profiling tag of a pool-scheduled handle. Every coroutine
// that runs on the Session ThreadPool is a Job (see the Job class
// comment), so reinterpreting the handle's promise as Job::promise_type
// is valid. Returns kNoPerfTag for a null handle or an untagged Job.
inline std::uint32_t
perf_tag_of(std::coroutine_handle<> h) noexcept
{
  if (!h) { return kNoPerfTag; }
  return std::coroutine_handle<Job::promise_type>::from_address(
             h.address()).promise()._perf_tag;
}

}

#endif
