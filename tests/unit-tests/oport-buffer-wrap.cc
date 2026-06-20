#include "minitest.h"
#include "common/beat-payload-intf.h"
#include "common/oport-policy.h"
#include "common/session.h"
#include "pipeline/edge-reader.h"
#include "pipeline/oport-buffer.h"
#include "tests/unit-tests/payload-types.h"

#include <utility>

using namespace vpipe;

// Push/pop more than 2^20 beats through a small-ring buffer; verify
// the cursor observes consecutive seqs across the uint32 wrap-around
// arithmetic.

TEST(oport_buffer_wrap, cursor_sees_consecutive_seqs_through_wrap)
{
  Session s;
  OportPolicy p;
  p.capacity = 4;
  p.mode     = OverrunPolicy::Backpressure;
  OportBuffer buf(&s, p);
  EdgeReader  r(&buf, buf.attach_cursor());

  // Run ~5 * (1<<20) iterations to make sure the uint32 wp + cursor
  // seqs wrap and still match. We push, then immediately read, so
  // the buffer never fills and the writer never suspends.
  const unsigned N = 5u * (1u << 20);
  for (unsigned i = 0; i < N; ++i) {
    OportBuffer::WriteAwaiter w{
        &buf, make_payload<test::IntPayload>(static_cast<int>(i))};
    EXPECT_TRUE(w.await_ready());
    w.await_resume();

    EdgeReader::ReadAwaiter ar{&r, {}, false};
    EXPECT_TRUE(ar.await_ready());
    auto pl = ar.await_resume();
    EXPECT_TRUE(pl != nullptr);
    EXPECT_TRUE(static_cast<const test::IntPayload&>(*pl).value
                == static_cast<int>(i));
  }
}
