#include "minitest.h"
#include "common/beat-payload-intf.h"
#include "common/oport-policy.h"
#include "common/session.h"
#include "pipeline/edge-reader.h"
#include "pipeline/oport-buffer.h"
#include "tests/unit-tests/payload-types.h"

#include <stdexcept>
#include <utility>

using namespace vpipe;

namespace {

// Default-policy buffer: capacity == 0, so soft thresholds at 1024
// (warn) and 2048 (throw) are armed.
OportPolicy
default_policy()
{
  return OportPolicy{};
}

}

TEST(oport_buffer_thresholds, default_mode_throws_at_soft_error)
{
  Session s;
  OportBuffer buf(&s, default_policy());
  EdgeReader  r(&buf, buf.attach_cursor());

  // Push without draining. The first soft_warn-1 writes succeed
  // silently; the soft_warn'th triggers a WARN; the soft_error'th
  // throws.
  bool threw = false;
  unsigned written = 0;
  try {
    for (unsigned i = 0; i < 3000; ++i) {
      OportBuffer::WriteAwaiter w{
          &buf, make_payload<test::IntPayload>(static_cast<int>(i))};
      if (!w.await_ready()) {
        throw std::runtime_error("unexpected suspend in default mode");
      }
      w.await_resume();
      ++written;
    }
  } catch (const std::runtime_error&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
  // Buffer accepts up to soft_error - 1 entries before the next
  // push throws (default soft_error == 2048).
  EXPECT_TRUE(written == 2048);
}

TEST(oport_buffer_thresholds, user_specified_capacity_disables_softs)
{
  Session s;
  OportPolicy p;
  p.capacity = 4;
  p.mode     = OverrunPolicy::Backpressure;
  OportBuffer buf(&s, p);
  EdgeReader  r(&buf, buf.attach_cursor());

  // capacity 4: backpressure kicks in at 4. No soft threshold throw.
  for (int i = 0; i < 4; ++i) {
    OportBuffer::WriteAwaiter w{
        &buf, make_payload<test::IntPayload>(i)};
    EXPECT_TRUE(w.await_ready());
    w.await_resume();
  }
  OportBuffer::WriteAwaiter w{
      &buf, make_payload<test::IntPayload>(4)};
  // Fifth write would suspend; we never reach a throw path.
  EXPECT_FALSE(w.await_ready());
}

TEST(oport_buffer_thresholds, user_specified_clamp_at_kmax)
{
  Session s;
  OportPolicy p;
  p.capacity = OportPolicy::kMaxCapacity + 100;
  OportBuffer buf(&s, p);
  EXPECT_TRUE(buf.capacity() == OportPolicy::kMaxCapacity);
}
