#include "generative-models/shared/tp-overlap.h"

#include "apple-silicon/metal-compute/metal-compute.h"

#include <pthread.h>
#include <pthread/qos.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <utility>

namespace vpipe::genai {

namespace {

// VPIPE_TP_FAKE=1 (a debugging aid): no channel at all -- every sum
// comes back zero at once -- to time a rank's own work alone. The run's
// output is wrong.
bool
fake_link_()
{
  static const bool f = std::getenv("VPIPE_TP_FAKE") != nullptr;
  return f;
}

// Past anything a run posts: the value done() is set to when the waits
// are released. A GPU wait compares >=.
constexpr std::uint64_t kRelease = std::uint64_t(1) << 62;

using Clock = std::chrono::steady_clock;

double
ms_since(Clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t0)
      .count();
}

}

TpOverlap::TpOverlap(metal_compute::MetalCompute* mc,
                     std::shared_ptr<TpChannel> ch, int lanes,
                     TpDType dtype)
  : _ch(std::move(ch)), _dtype(dtype)
{
  if (mc == nullptr || !_ch || lanes < 1) { return; }
  for (int i = 0; i < lanes; ++i) {
    _lanes.emplace_back();
    _lanes.back().ready = mc->make_event();
    if (!_lanes.back().ready.valid()) { return; }
  }
  _done = mc->make_event();
  if (!_done.valid()) { return; }
  _ok = true;
  _th = std::thread([this]() { run_(); });
}

TpOverlap::~TpOverlap()
{
  if (pending()) { abandon("the model went away"); }
  {
    std::lock_guard<std::mutex> lk(_mu);
    _stop = true;
  }
  _cv.notify_all();
  if (_th.joinable()) { _th.join(); }
}

TpOverlap::Ticket
TpOverlap::post(int lane, const metal_compute::SharedBuffer& partial,
                const metal_compute::SharedBuffer& others, std::size_t bytes,
                std::uint64_t seq)
{
  Job j;
  j.partial = partial.contents();
  j.others  = others.contents();
  j.bytes   = bytes;
  j.seq     = seq;
  j.lane    = lane;
  Ticket t;
  {
    std::lock_guard<std::mutex> lk(_mu);
    Lane& ln = _lanes[(std::size_t)lane];
    j.ready_at = ++ln.uses;
    j.v = _posted.load(std::memory_order_relaxed) + 1;
    _posted.store(j.v, std::memory_order_release);
    t.ready    = &ln.ready;
    t.ready_at = j.ready_at;
    t.done_at  = j.v;
    if (_abandoned.load(std::memory_order_acquire)) { return t; }
    _q.push_back(j);
  }
  _cv.notify_one();
  return t;
}

bool
TpOverlap::exchange_now(const void* partial, void* others, std::size_t bytes,
                        std::uint64_t seq, std::string* err)
{
  {
    std::unique_lock<std::mutex> lk(_mu);
    _idle.wait(lk, [&]() { return _q.empty(); });
  }
  if (failed()) {
    if (err != nullptr) { *err = error(); }
    return false;
  }
  const auto t0 = Clock::now();
  bool ok = true;
  if (fake_link_()) {
    std::memset(others, 0, bytes);
  } else {
    ok = call_exchange_(partial, others, bytes, seq, err);
  }
  std::lock_guard<std::mutex> lk(_mu);
  ++_stats.exchanges;
  _stats.bytes += bytes;
  _stats.link_ms += ms_since(t0);
  return ok;
}

bool
TpOverlap::call_exchange_(const void* partial, void* others,
                          std::size_t bytes, std::uint64_t seq,
                          std::string* err)
{
  std::lock_guard<std::mutex> lk(_ch_mu);
  try {
    return _ch->exchange(partial, others, bytes, _dtype, seq, err);
  } catch (const std::exception& e) {
    if (err != nullptr) {
      *err = std::string("the channel threw: ") + e.what();
    }
  } catch (...) {
    if (err != nullptr) { *err = "the channel threw"; }
  }
  return false;
}

void
TpOverlap::forget(const metal_compute::SharedBuffer& b)
{
  if (b.empty() || !_ch) { return; }
  // What was posted answered first: a buffer is forgotten only once no
  // exchange of it is queued or in flight -- the worker holds _ch_mu
  // through an exchange, so taking it waits out the one it is in.
  {
    std::unique_lock<std::mutex> lk(_mu);
    _idle.wait(lk, [&]() { return _q.empty(); });
  }
  std::lock_guard<std::mutex> lk(_ch_mu);
  try {
    _ch->forget(b.contents(), b.byte_size());
  } catch (...) {
    // Nothing to fail: the buffer goes either way.
  }
}

std::string
TpOverlap::error() const
{
  std::lock_guard<std::mutex> lk(_mu);
  return _err;
}

void
TpOverlap::fail_(const std::string& why)
{
  {
    std::lock_guard<std::mutex> lk(_mu);
    if (_err.empty()) { _err = why; }
  }
  _failed.store(true, std::memory_order_release);
  _done.set_signaled_value(kRelease);
}

void
TpOverlap::abandon(const std::string& why)
{
  _abandoned.store(true, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lk(_mu);
    _q.clear();
    _finished.store(_posted.load(std::memory_order_acquire),
                    std::memory_order_release);
  }
  _idle.notify_all();
  _cv.notify_all();
  fail_(why);
}

TpOverlap::Stats
TpOverlap::take_stats()
{
  std::lock_guard<std::mutex> lk(_mu);
  return std::exchange(_stats, Stats{});
}

bool
TpOverlap::wait_ready_(metal_compute::Event& ev, std::uint64_t v)
{
  // A SHORT SPIN, THEN THE KERNEL'S WAIT. The GPU is usually close: a
  // partial is posted as it is encoded, and the worker reaches it once
  // the one before has crossed. Spinning past that would burn a core for
  // the length of a forward the encoder ran far ahead of.
  const auto t0 = Clock::now();
  while (ev.signaled_value() < v) {
    if (_abandoned.load(std::memory_order_acquire)) { return false; }
    if (Clock::now() - t0 < std::chrono::microseconds(200)) { continue; }
    (void)ev.wait(v, std::chrono::milliseconds(2));
  }
  return true;
}

void
TpOverlap::run_()
{
  // The link's latency is the critical path whenever an exchange is not
  // hidden: on a performance core.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  for (;;) {
    Job j;
    {
      std::unique_lock<std::mutex> lk(_mu);
      _cv.wait(lk, [&]() { return _stop || !_q.empty(); });
      if (_q.empty()) { return; }
      j = _q.front();
    }
    const auto t0 = Clock::now();
    const bool ready =
        !failed() && wait_ready_(_lanes[(std::size_t)j.lane].ready,
                                 j.ready_at);
    const double idle = ms_since(t0);
    const auto t1 = Clock::now();
    if (ready) {
      std::string err;
      bool ok = true;
      if (fake_link_()) {
        std::memset(j.others, 0, j.bytes);
      } else {
        ok = call_exchange_(j.partial, j.others, j.bytes, j.seq, &err);
      }
      if (!ok) {
        fail_(err.empty() ? std::string("the exchange failed") : err);
      }
      static const bool trace = std::getenv("VPIPE_TP_TRACE") != nullptr;
      if (trace) {
        auto fnv = [](const void* p, std::size_t n) {
          std::uint64_t h = 1469598103934665603ull;
          const auto* b = static_cast<const unsigned char*>(p);
          for (std::size_t i = 0; i < n; ++i) {
            h = (h ^ b[i]) * 1099511628211ull;
          }
          return h;
        };
        std::fprintf(stderr, "[tp-trace] seq %llu lane %d bytes %zu sent "
                     "%016llx got %016llx\n", (unsigned long long)j.seq,
                     j.lane, j.bytes,
                     (unsigned long long)fnv(j.partial, j.bytes),
                     (unsigned long long)fnv(j.others, j.bytes));
      }
    }
    const double link = ms_since(t1);
    std::lock_guard<std::mutex> lk(_mu);
    if (_q.empty() || _q.front().v != j.v) { continue; }   // abandoned
    // ANSWERED BEFORE IT LEAVES THE QUEUE: pending() reads _finished, and
    // a forward that has seen its GPU finish must find nothing pending.
    _finished.store(j.v, std::memory_order_release);
    if (!failed()) { _done.set_signaled_value(j.v); }
    _q.pop_front();
    if (ready) {
      ++_stats.exchanges;
      _stats.bytes += j.bytes;
      _stats.link_ms += link;
      _stats.idle_ms += idle;
    }
    if (_q.empty()) { _idle.notify_all(); }
  }
}

}
