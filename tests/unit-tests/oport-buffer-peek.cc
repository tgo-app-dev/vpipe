#include "minitest.h"
#include "common/beat-payload-intf.h"
#include "common/oport-policy.h"
#include "common/session.h"
#include "pipeline/edge-reader.h"
#include "pipeline/oport-buffer.h"
#include "tests/unit-tests/payload-types.h"

#include <utility>

using namespace vpipe;

namespace {

OportPolicy
backpressure_policy(unsigned cap)
{
  OportPolicy p;
  p.capacity = cap;
  p.mode     = OverrunPolicy::Backpressure;
  return p;
}

bool
sync_write(OportBuffer& buf, int v)
{
  OportBuffer::WriteAwaiter w{
      &buf, make_payload<test::IntPayload>(v)};
  bool ready = w.await_ready();
  if (!ready) { return false; }
  if (w._saw_closed || w._saw_threshold) {
    throw std::runtime_error("write to closed/saturated buffer");
  }
  w.await_resume();
  return true;
}

const int*
sync_peek(EdgeReader& r, uint32_t offset)
{
  EdgeReader::PeekAwaiter pk{&r, offset, nullptr, false};
  if (!pk.await_ready()) { return nullptr; }
  auto p = pk.await_resume();
  if (!p) { return nullptr; }
  return &static_cast<const test::IntPayload&>(*p).value;
}

}

TEST(oport_buffer_peek, peek_at_offset_zero_returns_oldest)
{
  Session s;
  OportBuffer buf(&s, backpressure_policy(4));
  EdgeReader  r(&buf, buf.attach_cursor());

  EXPECT_TRUE(sync_write(buf, 10));
  EXPECT_TRUE(sync_write(buf, 11));
  EXPECT_TRUE(sync_write(buf, 12));

  const int* p0 = sync_peek(r, 0);
  EXPECT_TRUE(p0 != nullptr);
  EXPECT_TRUE(*p0 == 10);
  // Peek does NOT advance the cursor.
  EXPECT_TRUE(r.backlog() == 3);
}

TEST(oport_buffer_peek, peek_window_returns_consecutive_seqs)
{
  Session s;
  OportBuffer buf(&s, backpressure_policy(4));
  EdgeReader  r(&buf, buf.attach_cursor());

  EXPECT_TRUE(sync_write(buf, 100));
  EXPECT_TRUE(sync_write(buf, 101));
  EXPECT_TRUE(sync_write(buf, 102));

  EXPECT_TRUE(*sync_peek(r, 0) == 100);
  EXPECT_TRUE(*sync_peek(r, 1) == 101);
  EXPECT_TRUE(*sync_peek(r, 2) == 102);
  EXPECT_TRUE(r.backlog() == 3);
}

TEST(oport_buffer_peek, peek_beyond_backlog_does_not_ready)
{
  Session s;
  OportBuffer buf(&s, backpressure_policy(4));
  EdgeReader  r(&buf, buf.attach_cursor());

  EXPECT_TRUE(sync_write(buf, 1));
  // Only one beat available; peek at offset 1 should not be ready.
  EdgeReader::PeekAwaiter pk{&r, 1, nullptr, false};
  EXPECT_FALSE(pk.await_ready());
}

TEST(oport_buffer_peek, release_read_advances_cursor)
{
  Session s;
  OportBuffer buf(&s, backpressure_policy(4));
  EdgeReader  r(&buf, buf.attach_cursor());

  EXPECT_TRUE(sync_write(buf, 1));
  EXPECT_TRUE(sync_write(buf, 2));
  EXPECT_TRUE(sync_write(buf, 3));
  EXPECT_TRUE(r.backlog() == 3);

  r.release_read(2);
  EXPECT_TRUE(r.backlog() == 1);

  const int* p = sync_peek(r, 0);
  EXPECT_TRUE(p != nullptr);
  EXPECT_TRUE(*p == 3);
}

TEST(oport_buffer_peek, peek_across_eos_returns_null_after_drain)
{
  Session s;
  OportBuffer buf(&s, backpressure_policy(4));
  EdgeReader  r(&buf, buf.attach_cursor());

  EXPECT_TRUE(sync_write(buf, 7));
  buf.close();

  EXPECT_TRUE(*sync_peek(r, 0) == 7);
  r.release_read(1);
  // After the only beat is consumed, peek-at-0 hits EOS.
  EdgeReader::PeekAwaiter pk{&r, 0, nullptr, false};
  EXPECT_TRUE(pk.await_ready());
  EXPECT_TRUE(pk.await_resume() == nullptr);
}
