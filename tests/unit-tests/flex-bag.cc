// flex-bag.cc -- the growth seam every value struct at the plugin
// boundary ends in (common/flex-bag.h).
//
// What is under test is the contract a reader relies on: an ABSENT key
// is the default -- which is what every binary older than the key sends
// -- a null bag reads as empty, and the first write makes one. The last
// is the easy one to get wrong: the toolkit's accel helpers do nothing
// on a null bag, and a struct's `extra` starts null, so a writer built on
// them would drop every key it set without a word.
//
//   vpipe_test --filter 'flex_bag*'

#include "minitest.h"

#include "common/flex-bag.h"
#include "generative-models/gen-input.h"
#include "pipeline/memory-plan.h"
#include "pipeline/resource-plan.h"

#include <string>

using namespace vpipe;

TEST(flex_bag, an_absent_key_is_the_default) {
  const FlexData none;
  EXPECT_TRUE(none.is_null());
  EXPECT_FALSE(bag::has(none, "k"));
  EXPECT_TRUE(bag::flag(none, "k", true));
  EXPECT_TRUE(bag::integer(none, "k", 7) == 7);
  EXPECT_TRUE(bag::real(none, "k", 2.5) == 2.5);
  EXPECT_TRUE(bag::text(none, "k", "d") == "d");
  EXPECT_TRUE(bag::value(none, "k").is_null());

  // A null POINTER is a host that sent no bag at all.
  const FlexData* nil = nullptr;
  EXPECT_FALSE(bag::has(nil, "k"));
  EXPECT_TRUE(bag::integer(nil, "k", 3) == 3);
  EXPECT_TRUE(bag::text(nil, "k", "d") == "d");
}

TEST(flex_bag, the_first_write_makes_the_bag) {
  FlexData b;
  EXPECT_TRUE(bag::set_integer(b, "n", 42));
  EXPECT_TRUE(b.is_object());
  EXPECT_TRUE(bag::integer(b, "n") == 42);
  EXPECT_TRUE(bag::set_flag(b, "on", true));
  EXPECT_TRUE(bag::set_real(b, "x", 0.25));
  EXPECT_TRUE(bag::set_text(b, "s", "hi"));
  EXPECT_TRUE(bag::flag(b, "on"));
  EXPECT_TRUE(bag::real(b, "x") == 0.25);
  EXPECT_TRUE(bag::text(b, "s") == "hi");
  // Overwrites, rather than refusing or duplicating.
  EXPECT_TRUE(bag::set_integer(b, "n", 43));
  EXPECT_TRUE(bag::integer(b, "n") == 43);

  // A bag that holds something other than an object is not written
  // over. No host-defined bag ever does; the answer is still defined.
  FlexData arr = FlexData::make_array();
  EXPECT_FALSE(bag::set_integer(arr, "n", 1));
  EXPECT_TRUE(arr.is_array());
}

// The bag travels WITH the struct, by value, and a copy is its own: a
// holding the plan copies and annotates must not annotate the stage's.
TEST(flex_bag, a_struct_copy_owns_its_bag) {
  StageHolding h;
  h.source = "/m";
  h.preload = 10;
  bag::set_text(h.extra, "note", "a");
  StageHolding c = h;
  bag::set_text(c.extra, "note", "b");
  EXPECT_TRUE(bag::text(h.extra, "note") == "a");
  EXPECT_TRUE(bag::text(c.extra, "note") == "b");

  // And one that was never written costs nothing to carry.
  ResourceClaim r;
  EXPECT_TRUE(r.extra.is_null());
  ResourceClaim r2 = r;
  EXPECT_TRUE(r2.extra.is_null());
}

// A NAMED TENSOR'S FORMAT rides in its bag. Absent is f32 -- the only
// thing a 4-byte tensor could be before the key existed -- and a 2-byte
// tensor that does not say which 2-byte format it is has NO dtype,
// because bf16 read as f16 is the right magnitude and the wrong picture.
TEST(named_tensor, the_format_is_a_key_and_absent_is_f32) {
  namespace nt = genai::named_tensor;
  genai::NamedTensor t;
  EXPECT_TRUE(t.extra.is_null());
  EXPECT_TRUE(t.dtype() == nt::kF32);
  EXPECT_TRUE(t.is(nt::kF32));

  t.elem_size = 2;
  EXPECT_TRUE(t.dtype().empty());

  EXPECT_TRUE(t.set_dtype(nt::kBf16));
  EXPECT_TRUE(t.elem_size == 2);
  EXPECT_TRUE(t.is(nt::kBf16));
  EXPECT_FALSE(t.is(nt::kF16));
  EXPECT_TRUE(bag::text(t.extra, nt::kDtype) == "bf16");

  EXPECT_TRUE(t.set_dtype(nt::kFp8E4m3));
  EXPECT_TRUE(t.elem_size == 1);

  // A name this header does not know the width of changes nothing.
  EXPECT_FALSE(t.set_dtype("f4_packed"));
  EXPECT_TRUE(t.is(nt::kFp8E4m3));
  EXPECT_TRUE(t.elem_size == 1);

  // ...but a producer that knows it can still say so, and a reader that
  // does not recognise it gets the name back and the bytes to size.
  bag::set_text(t.extra, nt::kDtype, "f4_packed");
  EXPECT_TRUE(t.dtype() == "f4_packed");
  EXPECT_TRUE(nt::dtype_bytes(t.dtype()) == 0);
}
