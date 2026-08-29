// What an F32 RGB beat's numbers MEAN, and the mapping that follows.
//
// The dtype does not say. This tree's only f32 RGB producer is
// `video-to-rgb`, whose `normalize` key defaults ON and divides by 255 --
// its own doc calls the result [0,1] -- while `preview-stage`, the only
// other f32 consumer, offers exactly [0,1] or [0,255] and nothing else.
// vae-encode nevertheless read f32 as [-1,1] until it was measured
// against a round trip.
//
// THE COST OF THAT was not a failure. Encoding `p` where `2p - 1` was
// meant halves the contrast and lifts the blacks to mid grey, and the
// picture that comes back is recognisable -- so it decoded, it saved, it
// looked like the scene. MEASURED on the LTX video VAE at its native
// 960x544: 8.09 dB before, 43.09 dB after, with the pixel span restored
// from [0.459, 0.902] to [0.000, 0.804].
//
// So the mapping is pinned here, in the terms that were wrong: what a
// black pixel and a white pixel become.

#include "minitest.h"

#include "stages/vae-encode-stage.h"

#include <cmath>
#include <string>

namespace {

// `value * scale + offset` at the two ends of a range, which is the
// whole of what the mapping has to get right.
struct Ends {
  bool ok = false;
  float black = 0.0f;
  float white = 0.0f;
};

Ends
ends(const std::string& range, float lo, float hi)
{
  Ends e;
  float s = 0.0f, o = 0.0f;
  e.ok = vpipe::vae_input_range_scale(range, &s, &o);
  e.black = lo * s + o;
  e.white = hi * s + o;
  return e;
}

bool
near(float a, float b)
{
  return std::fabs(a - b) < 1e-6f;
}

}  // namespace

TEST(vae_encode_range, unit_is_the_default_and_maps_zero_one_to_minus_one_one)
{
  // The DEFAULT, and it is the default because [0,1] is what
  // video-to-rgb emits. An empty string is the unset config key.
  for (const char* name : {"", "unit"}) {
    const Ends e = ends(name, 0.0f, 1.0f);
    EXPECT_TRUE(e.ok);
    EXPECT_TRUE(near(e.black, -1.0f));
    EXPECT_TRUE(near(e.white, 1.0f));
  }
}

TEST(vae_encode_range, byte_maps_zero_two_fifty_five_to_minus_one_one)
{
  const Ends e = ends("byte", 0.0f, 255.0f);
  EXPECT_TRUE(e.ok);
  EXPECT_TRUE(near(e.black, -1.0f));
  EXPECT_TRUE(near(e.white, 1.0f));
}

TEST(vae_encode_range, signed_is_the_identity)
{
  const Ends e = ends("signed", -1.0f, 1.0f);
  EXPECT_TRUE(e.ok);
  EXPECT_TRUE(near(e.black, -1.0f));
  EXPECT_TRUE(near(e.white, 1.0f));
}

TEST(vae_encode_range, the_ranges_are_not_interchangeable)
{
  // THE BUG, stated as a test: reading a [0,1] beat as though it were
  // already [-1,1] leaves black at 0 -- mid grey -- and white at 1. That
  // is the half-contrast, lifted-black picture, and it is why `unit` and
  // `signed` must never be silently swapped.
  const Ends wrong = ends("signed", 0.0f, 1.0f);
  EXPECT_TRUE(wrong.ok);
  EXPECT_TRUE(near(wrong.black, 0.0f));
  EXPECT_FALSE(near(wrong.black, -1.0f));
}

TEST(vae_encode_range, an_unknown_name_is_refused)
{
  // Refused, not defaulted: a graph that names a range this does not
  // know has said something specific, and guessing would be the same
  // silent shift again. The stage turns false into a config failure.
  float s = 0.0f, o = 0.0f;
  EXPECT_FALSE(vpipe::vae_input_range_scale("normalized", &s, &o));
  EXPECT_FALSE(vpipe::vae_input_range_scale("[0,1]", &s, &o));
  EXPECT_FALSE(vpipe::vae_input_range_scale("UNIT", &s, &o));
}
