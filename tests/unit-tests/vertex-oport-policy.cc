#include "minitest.h"
#include "common/oport-policy.h"
#include "common/session.h"
#include "common/vertex.h"

#include <stdexcept>

using namespace vpipe;

namespace {

// Vertex is the substrate Stage builds on; constructing one directly
// is enough to exercise the per-oport policy plumbing.
struct TestVertex : public Vertex {
  TestVertex(const SessionContextIntf* s)
    : Vertex(s, "test-vertex", {})
  {}
};

}

TEST(vertex_oport_policy, default_policy_is_backpressure_zero_capacity)
{
  Session s;
  TestVertex v(&s);
  v.allocate_oports(3);

  for (unsigned p = 0; p < 3; ++p) {
    const auto& pol = v.oport_policy(p);
    EXPECT_TRUE(pol.capacity == 0);
    EXPECT_TRUE(pol.mode == OverrunPolicy::Backpressure);
  }
}

TEST(vertex_oport_policy, allocate_oports_resizes_policy_vector)
{
  Session s;
  TestVertex v(&s);
  EXPECT_TRUE(v.num_oports() == 0);

  v.allocate_oports(2);
  EXPECT_TRUE(v.num_oports() == 2);
  // Both slots must be readable without throwing -- the parallel
  // vector grew alongside _oports.
  (void)v.oport_policy(0);
  (void)v.oport_policy(1);
}

TEST(vertex_oport_policy, set_then_get_round_trip)
{
  Session s;
  TestVertex v(&s);
  v.allocate_oports(2);

  v.set_oport_policy(0, {42, OverrunPolicy::DropOldest});
  v.set_oport_policy(1, {7, OverrunPolicy::Backpressure});

  EXPECT_TRUE(v.oport_policy(0).capacity == 42);
  EXPECT_TRUE(v.oport_policy(0).mode == OverrunPolicy::DropOldest);
  EXPECT_TRUE(v.oport_policy(1).capacity == 7);
  EXPECT_TRUE(v.oport_policy(1).mode == OverrunPolicy::Backpressure);
}

TEST(vertex_oport_policy, out_of_range_set_throws)
{
  Session s;
  TestVertex v(&s);
  v.allocate_oports(1);

  bool threw = false;
  try {
    v.set_oport_policy(5, {16, OverrunPolicy::DropOldest});
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(vertex_oport_policy, out_of_range_get_throws)
{
  Session s;
  TestVertex v(&s);
  v.allocate_oports(1);

  bool threw = false;
  try {
    (void)v.oport_policy(5);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}
