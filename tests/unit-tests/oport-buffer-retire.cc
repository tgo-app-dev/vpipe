#include "minitest.h"
#include "common/beat-payload-intf.h"
#include "common/oport-policy.h"
#include "common/session.h"
#include "pipeline/edge-reader.h"
#include "pipeline/oport-buffer.h"
#include "tests/unit-tests/payload-types.h"

#include <stdexcept>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

// Synchronous helpers driving the awaiters directly. Works for the
// non-suspending path: DropOldest writes never suspend, reads that
// can pop without waiting return true from await_ready().

bool
sync_write(OportBuffer& buf, int v)
{
  OportBuffer::WriteAwaiter w{
      &buf, make_payload<test::IntPayload>(v)};
  bool ready = w.await_ready();
  if (!ready) {
    return false;
  }
  if (w._saw_closed || w._saw_threshold) {
    throw std::runtime_error("write to closed or saturated buffer");
  }
  w.await_resume();
  return true;
}

std::optional<int>
sync_read(EdgeReader& r)
{
  EdgeReader::ReadAwaiter aw{&r, {}, false};
  if (!aw.await_ready()) {
    throw std::runtime_error("read would suspend");
  }
  auto p = aw.await_resume();
  if (!p) {
    return std::nullopt;
  }
  return static_cast<const test::IntPayload&>(*p).value;
}

OportPolicy
drop_oldest_policy(unsigned cap)
{
  OportPolicy p;
  p.capacity = cap;
  p.mode     = OverrunPolicy::DropOldest;
  return p;
}

OportPolicy
backpressure_policy(unsigned cap)
{
  OportPolicy p;
  p.capacity = cap;
  p.mode     = OverrunPolicy::Backpressure;
  return p;
}

}

TEST(oport_buffer_retire, drop_oldest_evicts_front_when_full)
{
  Session s;
  OportBuffer buf(&s, drop_oldest_policy(3));
  EdgeReader  r(&buf, buf.attach_cursor());

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(sync_write(buf, i));
  }
  EXPECT_TRUE(buf.size() == 3);
  EXPECT_TRUE(buf.dropped() == 1);
  EXPECT_TRUE(r.dropped() == 1);

  EXPECT_TRUE(sync_read(r) == 1);
  EXPECT_TRUE(sync_read(r) == 2);
  EXPECT_TRUE(sync_read(r) == 3);
  EXPECT_TRUE(buf.size() == 0);
}

TEST(oport_buffer_retire, drop_oldest_never_suspends)
{
  Session s;
  OportBuffer buf(&s, drop_oldest_policy(1));
  EdgeReader  r(&buf, buf.attach_cursor());

  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(sync_write(buf, i));
  }
  EXPECT_TRUE(buf.size() == 1);
  EXPECT_TRUE(buf.dropped() == 999);
  EXPECT_TRUE(sync_read(r) == 999);
}

TEST(oport_buffer_retire, dropped_counter_with_no_cursor_is_zero)
{
  Session s;
  OportBuffer buf(&s, drop_oldest_policy(2));

  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(sync_write(buf, i));
  }
  EXPECT_TRUE(buf.dropped() == 0);
  EXPECT_TRUE(buf.size() == 0);
}

TEST(oport_buffer_retire, default_mode_is_backpressure)
{
  Session s;
  OportBuffer buf(&s, backpressure_policy(2));
  EdgeReader  r(&buf, buf.attach_cursor());
  EXPECT_TRUE(buf.mode() == OverrunPolicy::Backpressure);

  EXPECT_TRUE(sync_write(buf, 100));
  EXPECT_TRUE(sync_write(buf, 101));
  OportBuffer::WriteAwaiter w{
      &buf, make_payload<test::IntPayload>(102)};
  bool ready = w.await_ready();
  EXPECT_FALSE(ready);
  EXPECT_TRUE(buf.dropped() == 0);
}

TEST(oport_buffer_retire, drop_oldest_respects_close)
{
  Session s;
  OportBuffer buf(&s, drop_oldest_policy(2));
  EdgeReader  r(&buf, buf.attach_cursor());

  EXPECT_TRUE(sync_write(buf, 1));
  buf.close();

  OportBuffer::WriteAwaiter w{
      &buf, make_payload<test::IntPayload>(2)};
  EXPECT_TRUE(w.await_ready());
  EXPECT_TRUE(w._saw_closed);
  bool threw = false;
  try {
    w.await_resume();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  EXPECT_TRUE(threw);

  EXPECT_TRUE(sync_read(r) == 1);
  EXPECT_FALSE(sync_read(r).has_value());
}

TEST(oport_buffer_retire, drop_oldest_rejects_second_cursor)
{
  // DropOldest mode is restricted to a single cursor by the new
  // policy. Asserting that here documents the contract.
  Session s;
  OportBuffer buf(&s, drop_oldest_policy(2));
  EdgeReader  r(&buf, buf.attach_cursor());

  bool threw = false;
  try {
    (void)buf.attach_cursor();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(oport_buffer_retire, slowest_cursor_acquires_others_clone)
{
  // Multi-cursor Backpressure: two cursors read the same beat. The
  // slowest one's read returns the originally-pushed payload by
  // move; the faster one's read returns a clone.
  Session s;
  OportBuffer buf(&s, backpressure_policy(4));
  EdgeReader  a(&buf, buf.attach_cursor());
  EdgeReader  b(&buf, buf.attach_cursor());

  EXPECT_TRUE(sync_write(buf, 42));

  // Read on a first: a is now ahead, b is the slowest. a's acquire
  // is the not-slowest path and goes through clone; b's path will
  // be the slowest acquire (move).
  EdgeReader::ReadAwaiter ra{&a, {}, false};
  EXPECT_TRUE(ra.await_ready());
  auto pa = ra.await_resume();
  EXPECT_TRUE(pa != nullptr);
  EXPECT_TRUE(static_cast<const test::IntPayload&>(*pa).value == 42);

  EdgeReader::ReadAwaiter rb{&b, {}, false};
  EXPECT_TRUE(rb.await_ready());
  auto pb = rb.await_resume();
  EXPECT_TRUE(pb != nullptr);
  EXPECT_TRUE(static_cast<const test::IntPayload&>(*pb).value == 42);

  // Distinct unique_ptr instances (no shared_ptr sharing in the new
  // model).
  EXPECT_TRUE(pa.get() != pb.get());

  // Buffer is empty now -- both cursors advanced past the slot.
  EXPECT_TRUE(buf.size() == 0);
}
