#include "minitest.h"
#include "common/session.h"
#include "pipeline/oport-buffer.h"
#include "pipeline/runtime-context.h"

#include <atomic>
#include <vector>

using namespace std;
using namespace vpipe;

TEST(runtime_context_has_consumers, unwired_oport_reports_false)
{
  // Two oports, both have an OportBuffer but no attached cursors.
  Session     s;
  OportBuffer buf0(&s);
  OportBuffer buf1(&s);
  atomic<bool> stop{false};

  vector<OportBuffer*> out_bufs{&buf0, &buf1};
  RuntimeContext ctx({}, std::move(out_bufs), &stop);

  EXPECT_FALSE(ctx.has_consumers(0));
  EXPECT_FALSE(ctx.has_consumers(1));
}

TEST(runtime_context_has_consumers, wired_oport_reports_true)
{
  Session     s;
  OportBuffer buf0(&s);
  OportBuffer buf1(&s);
  (void)buf0.attach_cursor();      // oport 0 wired to one consumer
  // oport 1 has no cursor attached.
  atomic<bool> stop{false};

  vector<OportBuffer*> out_bufs{&buf0, &buf1};
  RuntimeContext ctx({}, std::move(out_bufs), &stop);

  EXPECT_TRUE(ctx.has_consumers(0));
  EXPECT_FALSE(ctx.has_consumers(1));
}

TEST(runtime_context_has_consumers, out_of_range_is_false_not_throw)
{
  Session     s;
  OportBuffer buf0(&s);
  atomic<bool> stop{false};

  vector<OportBuffer*> out_bufs{&buf0};
  RuntimeContext ctx({}, std::move(out_bufs), &stop);

  // No throw, just false. Producers can query freely without
  // bounds-checking themselves.
  EXPECT_FALSE(ctx.has_consumers(999));
}
