#include "minitest.h"
#include "apple-silicon/metal-compute/buffer-view.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/event.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

MetalCompute*
get_mc_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    return nullptr;
  }
  return mc;
}

}  // namespace

TEST(metal_compute_stream, make_stream_is_valid) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  CommandStream s = mc->make_command_stream();
  EXPECT_TRUE(s.valid());
  EXPECT_TRUE(s.mtl_queue() != nullptr);
}

TEST(metal_compute_stream, make_event_is_valid_with_zero_counter) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  Event ev = mc->make_event();
  EXPECT_TRUE(ev.valid());
  EXPECT_TRUE(ev.signaled_value() == 0u);
}

TEST(metal_compute_stream, event_set_and_read_round_trip) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  Event ev = mc->make_event();
  ev.set_signaled_value(7);
  EXPECT_TRUE(ev.signaled_value() == 7u);
  ev.set_signaled_value(42);
  EXPECT_TRUE(ev.signaled_value() == 42u);
}

TEST(metal_compute_stream, event_wait_returns_immediately_when_signaled) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  Event ev = mc->make_event();
  ev.set_signaled_value(5);
  EXPECT_TRUE(ev.wait(5, std::chrono::seconds(1)));
  EXPECT_TRUE(ev.wait(3, std::chrono::seconds(1)));  // 3 <= current
}

TEST(metal_compute_stream, event_wait_times_out_below_threshold) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  Event ev = mc->make_event();
  // Counter starts at 0; ask for 1 with a tiny timeout.
  EXPECT_FALSE(ev.wait(1, std::chrono::milliseconds(50)));
}

TEST(metal_compute_stream, commit_without_work_yields_invalid_fence) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  CommandStream s = mc->make_command_stream();
  CommandStream::Fence f = s.commit();
  EXPECT_FALSE(f.valid());
}

TEST(metal_compute_stream, empty_compute_dispatch_completes) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  CommandStream s = mc->make_command_stream();
  {
    ComputeEncoder e = s.begin_compute();
    EXPECT_TRUE(e.valid());
    // No dispatch -- just an open/end roundtrip on the encoder.
  }
  CommandStream::Fence f = s.commit();
  EXPECT_TRUE(f.valid());
  f.wait();
  EXPECT_TRUE(f.completed());
}

TEST(metal_compute_stream, on_completion_fires_after_wait) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  CommandStream s = mc->make_command_stream();
  { ComputeEncoder e = s.begin_compute(); (void)e; }

  std::mutex              mu;
  std::condition_variable cv;
  bool                    fired = false;

  // Attach the completion handler BEFORE commit; Metal asserts
  // if it's added after.
  s.on_completion([&]() {
    std::lock_guard<std::mutex> g(mu);
    fired = true;
    cv.notify_one();
  });
  CommandStream::Fence f = s.commit();
  f.wait();

  std::unique_lock<std::mutex> lk(mu);
  EXPECT_TRUE(cv.wait_for(lk, std::chrono::seconds(2),
                          [&] { return fired; }));
}

TEST(metal_compute_stream, cross_stream_signal_wait_propagates) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  CommandStream sA = mc->make_command_stream();
  CommandStream sB = mc->make_command_stream();
  Event ev = mc->make_event();
  if (!sA.valid() || !sB.valid() || !ev.valid()) {
    return;
  }

  std::atomic<int> handlers_fired{0};

  // Stream B waits for ev=1 before completing. We commit it
  // FIRST so it is queued and parked on the wait; if the sync
  // were broken, B would deadlock here (the wait would never
  // be satisfied).
  sB.encode_wait(ev, 1);
  sB.on_completion([&] { handlers_fired.fetch_add(1); });
  CommandStream::Fence fence_b = sB.commit();

  // Stream A signals ev=1. This must propagate to B's queue and
  // unblock its wait, even though A's CB was committed AFTER B's
  // and the two streams have independent ordering otherwise.
  sA.encode_signal(ev, 1);
  sA.on_completion([&] { handlers_fired.fetch_add(1); });
  CommandStream::Fence fence_a = sA.commit();

  // Both fences must complete without deadlock. fence_b.wait()
  // is the load-bearing assertion -- if the signal->wait edge
  // isn't honored, B's CB sits in the queue forever and we hang
  // until process timeout. CI hangs are easier to spot than
  // silent races, but we still want a structured failure if it
  // does happen, so cap with the event's CPU-side wait first.
  EXPECT_TRUE(ev.wait(1, std::chrono::seconds(5)));
  fence_a.wait();
  fence_b.wait();
  EXPECT_TRUE(fence_a.completed());
  EXPECT_TRUE(fence_b.completed());
  EXPECT_TRUE(ev.signaled_value() >= 1u);

  // Note: handler invocation order is NOT specified by Metal --
  // for trivial CBs that complete near-simultaneously, the two
  // completion-thread dispatches can race in either direction.
  // We only assert that both handlers fire eventually.
  for (int i = 0; i < 200 && handlers_fired.load() < 2; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(handlers_fired.load() == 2);
}

TEST(metal_compute_stream, fence_move_transfers_ownership) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  CommandStream s = mc->make_command_stream();
  { ComputeEncoder e = s.begin_compute(); (void)e; }
  CommandStream::Fence f1 = s.commit();
  EXPECT_TRUE(f1.valid());
  CommandStream::Fence f2(std::move(f1));
  EXPECT_FALSE(f1.valid());
  EXPECT_TRUE(f2.valid());
  f2.wait();
  EXPECT_TRUE(f2.completed());
}

TEST(metal_compute_stream, stream_move_transfers_open_buffer) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  CommandStream sA = mc->make_command_stream();
  { ComputeEncoder e = sA.begin_compute(); (void)e; }
  CommandStream sB(std::move(sA));
  EXPECT_FALSE(sA.valid());
  EXPECT_TRUE(sB.valid());
  CommandStream::Fence f = sB.commit();
  EXPECT_TRUE(f.valid());
  f.wait();
  EXPECT_TRUE(f.completed());
}

// The auto command-buffer split (armed by CommandStream::begin_compute at 50
// dispatches) ENDS the MTL encoder and opens a fresh one, and encoder state
// does NOT carry across that boundary. So the split may only fire at a CLEAN op
// boundary -- with nothing bound since the last dispatch. It fires from
// set_function() because a well-formed op names its function first; call sites
// that bind buffers BEFORE naming the function (the k-quant qmv helpers do)
// would otherwise have those bindings stranded on the retired encoder. Metal
// does not fault on an unbound slot, so the dispatch just reads garbage and the
// wrong answer propagates silently -- it cost a GGUF decode divergence that
// looked like a broken speculative decoder.
// This runs well past the split threshold with BOTH binding orders present: a
// clean op every 7th so the split keeps firing, dirty ops everywhere else so a
// split lands on one. (Strict alternation would NOT reproduce it -- the split
// threshold is even, so every split would land on the same parity.)
TEST(metal_compute_stream, auto_split_preserves_pending_encoder_state) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("saxpy");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("saxpy");
  if (!fn.valid()) {
    return;
  }

  constexpr std::size_t kElems = 64;
  constexpr int         kOps   = 240;   // >> the 50-dispatch split threshold
  constexpr float       kA     = 3.0f;

  BufferView view{};
  view.dtype      = DType::F32;
  view.rank       = 1;
  view.shape[0]   = static_cast<std::int64_t>(kElems);
  view.strides[0] = 1;
  view.offset     = 0;

  std::vector<SharedBuffer> xs, ys;
  for (int i = 0; i < kOps; ++i) {
    SharedBuffer x = mc->make_shared_buffer(kElems * sizeof(float));
    SharedBuffer y = mc->make_shared_buffer(kElems * sizeof(float));
    if (x.empty() || y.empty()) {
      return;
    }
    auto* xp = static_cast<float*>(x.contents());
    auto* yp = static_cast<float*>(y.contents());
    for (std::size_t e = 0; e < kElems; ++e) {
      xp[e] = static_cast<float>(i + 1);
      yp[e] = static_cast<float>(e);
    }
    x.set_view(view);
    y.set_view(view);
    xs.push_back(std::move(x));
    ys.push_back(std::move(y));
  }

  struct SaxpyParams { float a; };
  SaxpyParams params{ kA };
  const unsigned tew = fn.thread_execution_width();
  const unsigned tg  = tew > 0 ? tew : 32;

  CommandStream stream = mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    EXPECT_TRUE(enc.valid());
    for (int i = 0; i < kOps; ++i) {
      if (i % 7 == 0) {
        // Clean order -- the split is allowed to fire here.
        enc.set_function(fn);
        enc.set_buffer_view(/*buf*/ 0, xs[(std::size_t)i], /*meta*/ 1);
        enc.set_buffer_view(/*buf*/ 2, ys[(std::size_t)i], /*meta*/ 3);
        enc.set_constant(/*index*/ 4, params);
      } else {
        // Dirty order -- bindings first, function last.
        enc.set_buffer_view(/*buf*/ 0, xs[(std::size_t)i], /*meta*/ 1);
        enc.set_buffer_view(/*buf*/ 2, ys[(std::size_t)i], /*meta*/ 3);
        enc.set_constant(/*index*/ 4, params);
        enc.set_function(fn);
      }
      enc.dispatch({kElems, 1, 1}, {tg, 1, 1});
    }
  }
  stream.commit().wait();

  int bad_clean = 0, bad_dirty = 0;
  for (int i = 0; i < kOps; ++i) {
    const auto* yp =
        static_cast<const float*>(ys[(std::size_t)i].contents());
    for (std::size_t e = 0; e < kElems; ++e) {
      const float want = kA * static_cast<float>(i + 1)
                       + static_cast<float>(e);
      if (std::fabs(yp[e] - want) > 1e-4f) {
        if (i % 7 == 0) { ++bad_clean; } else { ++bad_dirty; }
        break;
      }
    }
  }
  if (bad_clean != 0 || bad_dirty != 0) {
    std::printf("[auto-split] wrong ops: %d clean-order, %d dirty-order "
                "(of %d total)\n", bad_clean, bad_dirty, kOps);
  }
  EXPECT_TRUE(bad_clean == 0);
  EXPECT_TRUE(bad_dirty == 0);
}

// Same stranding SHAPE as the auto split, but at a concurrent_scope boundary:
// entering and leaving a scope swaps the MTL encoder (reencode_). A scope
// boundary cannot be deferred the way the split can -- the exit has to restore
// Serial ordering before the next dependent dispatch -- so reencode_ replays
// the pending op's bindings instead. Two shapes: (A) bound before the scope
// opens, dispatched inside it; (B) bound inside the scope, dispatched after it
// closes.
//
// NOTE, so nobody mistakes this for a reproducer: it passes with OR without
// that replay. reencode_ reopens an encoder on the SAME command buffer, and
// this driver carries the argument table across that boundary -- which is why
// concurrent_scope has always worked. The auto-split case genuinely corrupts
// because it opens a NEW COMMAND BUFFER. Metal guarantees nothing about state
// across encoders, so the replay removes the dependence on that detail; this
// test pins the behaviour for a driver that stops being so forgiving.
TEST(metal_compute_stream, concurrent_scope_preserves_pending_encoder_state) {
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) {
    return;
  }
  ComputeLibrary lib = mc->load_library("saxpy");
  if (!lib.valid()) {
    return;
  }
  ComputeFunction fn = lib.function("saxpy");
  if (!fn.valid()) {
    return;
  }

  constexpr std::size_t kElems = 64;
  constexpr float       kA     = 4.0f;

  BufferView view{};
  view.dtype      = DType::F32;
  view.rank       = 1;
  view.shape[0]   = static_cast<std::int64_t>(kElems);
  view.strides[0] = 1;
  view.offset     = 0;

  SharedBuffer xs[2], ys[2];
  for (int i = 0; i < 2; ++i) {
    xs[i] = mc->make_shared_buffer(kElems * sizeof(float));
    ys[i] = mc->make_shared_buffer(kElems * sizeof(float));
    if (xs[i].empty() || ys[i].empty()) {
      return;
    }
    auto* xp = static_cast<float*>(xs[i].contents());
    auto* yp = static_cast<float*>(ys[i].contents());
    for (std::size_t e = 0; e < kElems; ++e) {
      xp[e] = static_cast<float>(i + 1);
      yp[e] = static_cast<float>(e);
    }
    xs[i].set_view(view);
    ys[i].set_view(view);
  }

  struct SaxpyParams { float a; };
  SaxpyParams params{ kA };
  const unsigned tew = fn.thread_execution_width();
  const unsigned tg  = tew > 0 ? tew : 32;

  CommandStream stream = mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    EXPECT_TRUE(enc.valid());
    // (A) bound BEFORE the scope opens -- the entry reencode must keep them.
    enc.set_function(fn);
    enc.set_buffer_view(/*buf*/ 0, xs[0], /*meta*/ 1);
    enc.set_buffer_view(/*buf*/ 2, ys[0], /*meta*/ 3);
    enc.set_constant(/*index*/ 4, params);
    {
      ComputeEncoder::ConcurrentScope scope = enc.concurrent_scope(true);
      enc.dispatch({kElems, 1, 1}, {tg, 1, 1});
      // (B) bound INSIDE the scope, dispatched after it closes -- the exit
      //     reencode must keep them too.
      enc.set_function(fn);
      enc.set_buffer_view(/*buf*/ 0, xs[1], /*meta*/ 1);
      enc.set_buffer_view(/*buf*/ 2, ys[1], /*meta*/ 3);
      enc.set_constant(/*index*/ 4, params);
    }
    enc.dispatch({kElems, 1, 1}, {tg, 1, 1});
  }
  stream.commit().wait();

  for (int i = 0; i < 2; ++i) {
    int bad = 0;
    const auto* yp = static_cast<const float*>(ys[i].contents());
    for (std::size_t e = 0; e < kElems; ++e) {
      const float want = kA * static_cast<float>(i + 1)
                       + static_cast<float>(e);
      if (std::fabs(yp[e] - want) > 1e-4f) { ++bad; }
    }
    if (bad != 0) {
      std::printf("[concurrent-scope] arm %c: %d/%zu elems wrong "
                  "(y[0]=%g want %g)\n", i == 0 ? 'A' : 'B', bad, kElems,
                  yp[0], kA * static_cast<float>(i + 1));
    }
    EXPECT_TRUE(bad == 0);
  }
}
