#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/event.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/session.h"

#include <atomic>
#include <chrono>
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
