// The acceleration settings as a BAG, which is a statement about the
// plugin ABI more than about the settings.
//
// The thing being pinned is that a family compiled against an OLDER host
// keeps working. It cannot be tested by compiling against an older host,
// so it is tested as the two properties that make it true:
//
//   an unknown key is ignored      -- what a new host sends an old
//                                     reader
//   a missing key is the default   -- what an old host sends a new
//                                     reader
//
// Both have to hold for the same value in both directions, or "add a key
// and nothing breaks" is not a rule, it is a hope. The third property --
// that VideoGenRequest's layout does not move when a key is added -- is
// not observable from inside one build; what makes it true is that the
// struct holds one pointer, and what would break it is someone putting a
// typed field back.

#include "generative-models/shared/accel-settings.h"
#include "generative-models/shared/motion-cache.h"
#include "generative-models/shared/sage-attention.h"
#include "generative-models/shared/sol-attention.h"
#include "generative-models/video-model-registry.h"
#include "stages/generate-image-stage.h"
#include "stages/generate-video-stage.h"
#include "common/session.h"
#include "minitest.h"

#include <string>
#include <vector>

using namespace vpipe;
namespace accel = vpipe::genai::accel;
namespace sol = vpipe::genai::sol;
namespace sage = vpipe::genai::sage;

namespace {

FlexData
bag_of(std::initializer_list<std::pair<const char*, FlexData>> kv)
{
  FlexData b = FlexData::make_object();
  auto o = b.as_object();
  for (const auto& p : kv) { o.insert_or_assign(p.first, p.second); }
  return b;
}

}  // namespace

// A KEY THIS READER HAS NEVER HEARD OF CHANGES NOTHING.
//
// This is the new-host-old-plugin direction, and it is the one the whole
// design exists for: the host adds `sol_something_new`, an already-built
// family reads the bag with the readers it was compiled with, and gets
// exactly what it got before.
TEST(accel_settings, an_unknown_key_is_ignored)
{
  const FlexData plain = bag_of({
      {"sol_attn", FlexData::make_bool(true)},
      {"sol_tau", FlexData::make_real(1.5)},
      {"sol_key_block", FlexData::make_int(32)},
  });
  FlexData extended = bag_of({
      {"sol_attn", FlexData::make_bool(true)},
      {"sol_tau", FlexData::make_real(1.5)},
      {"sol_key_block", FlexData::make_int(32)},
      // Three shapes of thing a future host might add: a new tier, a new
      // knob on an existing one, and a nested object.
      {"quantum_attn", FlexData::make_bool(true)},
      {"sol_second_thoughts", FlexData::make_real(0.25)},
      {"sol_profile", FlexData::make_object()},
  });

  const sol::Config a = sol::config_from_flex(&plain);
  const sol::Config b = sol::config_from_flex(&extended);
  EXPECT_TRUE(a.enabled == b.enabled);
  EXPECT_TRUE(a.tau == b.tau);
  EXPECT_TRUE(a.key_block == b.key_block);
  EXPECT_TRUE(a.dense_layers == b.dense_layers);
  EXPECT_TRUE(a.local_radius == b.local_radius);
  // ...and the tier list is the tiers THIS reader knows, not every flag
  // in the bag. A family cannot implement a tier it has never heard of
  // and cannot report on one either.
  const auto on = accel::tiers_on(&extended);
  EXPECT_TRUE(on.size() == 1);
  EXPECT_TRUE(!on.empty() && on[0] == accel::kSolAttn);
}

// A MISSING KEY IS THE SHIPPED DEFAULT.
//
// The old-host-new-plugin direction, and the reason the readers carry
// defaults at all: a family must behave identically against a host that
// predates a key and one that has it turned off. A null bag is the
// extreme of the same case -- a host too old to send one.
TEST(accel_settings, a_missing_key_is_the_shipped_default)
{
  const sol::Config dflt;
  const sage::Config sdflt;

  const sol::Config from_null = sol::config_from_flex(nullptr);
  EXPECT_TRUE(from_null.enabled == dflt.enabled);
  EXPECT_TRUE(from_null.tau == dflt.tau);
  EXPECT_TRUE(from_null.key_block == dflt.key_block);
  EXPECT_TRUE(from_null.dense_layers == dflt.dense_layers);
  EXPECT_TRUE(from_null.local_radius == dflt.local_radius);

  const FlexData empty = FlexData::make_object();
  const sol::Config from_empty = sol::config_from_flex(&empty);
  EXPECT_TRUE(from_empty.tau == dflt.tau);
  EXPECT_TRUE(from_empty.dense_layers == dflt.dense_layers);
  EXPECT_TRUE(sage::config_from_flex(nullptr).enabled == sdflt.enabled);
  EXPECT_TRUE(sage::config_from_flex(&empty).dense_layers ==
              sdflt.dense_layers);

  // A PARTIAL bag -- one key of five, which is what an old host that
  // knew only the enable flag would send -- leaves the rest at their
  // defaults rather than at zero.
  const FlexData partial = bag_of({{"sol_attn", FlexData::make_bool(true)}});
  const sol::Config p = sol::config_from_flex(&partial);
  EXPECT_TRUE(p.enabled);
  EXPECT_TRUE(p.tau == dflt.tau);
  EXPECT_TRUE(p.dense_layers == dflt.dense_layers);
  EXPECT_TRUE(p.local_radius == dflt.local_radius);
}

// EVERY VALUE SURVIVES THE ROUND TRIP, which is the boring half and the
// one that catches a key written under one name and read under another.
TEST(accel_settings, the_settings_survive_the_bag)
{
  FlexData b = FlexData::make_object();
  accel::set_flag(&b, accel::kSolAttn, true);
  accel::set_real(&b, accel::kSolTau, 2.25);
  accel::set_integer(&b, accel::kSolKeyBlock, 32);
  accel::set_integer(&b, accel::kSolDenseLayers, 4);
  accel::set_integer(&b, accel::kSolLocalRadius, 3);
  accel::set_flag(&b, accel::kSageAttn, true);
  accel::set_integer(&b, accel::kSageDenseLayers, 2);
  accel::set_flag(&b, accel::kI8Gemm, true);

  const sol::Config c = sol::config_from_flex(&b);
  EXPECT_TRUE(c.enabled);
  EXPECT_TRUE(c.tau == 2.25f);
  EXPECT_TRUE(c.key_block == 32);
  EXPECT_TRUE(c.dense_layers == 4);
  EXPECT_TRUE(c.local_radius == 3);
  const sage::Config s = sage::config_from_flex(&b);
  EXPECT_TRUE(s.enabled);
  EXPECT_TRUE(s.dense_layers == 2);
  EXPECT_TRUE(accel::flag(&b, accel::kI8Gemm));
  EXPECT_TRUE(accel::tiers_on(&b).size() == 3);
}

// THE STAGE'S OWN VIEW AND THE ONE IT SENDS ARE THE SAME DECISION.
//
// generate-video keeps typed members for the in-tree models -- they are
// compiled with it and have no ABI to protect -- and sends the bag to
// everyone else. Those are two readings of one config, and the failure
// mode if they drift is that a plugin runs at settings the log line did
// not describe. So the members are read BACK out of the bag, and this is
// what says so.
TEST(accel_settings, the_stage_sends_what_it_says_it_sends)
{
  Session sess;
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("sol_attn", FlexData::make_bool(true));
    o.insert_or_assign("sol_tau", FlexData::make_real(1.75));
    o.insert_or_assign("sol_dense_layers", FlexData::make_int(3));
    o.insert_or_assign("sage_attn", FlexData::make_bool(true));
    o.insert_or_assign("i8_gemm", FlexData::make_bool(true));
    // Refused and corrected, which is the interesting one: what reaches
    // a family has to be what the stage SETTLED on, not what the graph
    // asked for, or every family repeats the validation and some of them
    // get it wrong.
    o.insert_or_assign("sol_key_block", FlexData::make_int(128));
  }
  GenerateVideoStage st(&sess, "gv", std::vector<InEdge>{}, cfg);
  EXPECT_TRUE(st.config_error().empty());

  const FlexData& bag = st.accel_settings();
  const sol::Config from_bag = sol::config_from_flex(&bag);
  EXPECT_TRUE(from_bag.enabled == st.sol_config().enabled);
  EXPECT_TRUE(from_bag.tau == st.sol_config().tau);
  EXPECT_TRUE(from_bag.dense_layers == st.sol_config().dense_layers);
  EXPECT_TRUE(from_bag.local_radius == st.sol_config().local_radius);
  // The corrected value, in both views.
  EXPECT_TRUE(st.sol_config().key_block == 0);
  EXPECT_TRUE(from_bag.key_block == 0);
  EXPECT_TRUE(accel::tiers_on(&bag).size() == 3);
}

// The SAME three tiers off generate-image, and the same validation.
//
// A separate test rather than a loop over the two stages, because what
// is being pinned is that the two stages agree -- and a helper that ran
// the same code against both would pass just as happily if they shared
// one wrong implementation. The values here are written out again on
// purpose.
TEST(accel_settings, the_image_stage_sends_the_same_three_tiers)
{
  Session sess;
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert_or_assign("sol_attn", FlexData::make_bool(true));
    o.insert_or_assign("sol_tau", FlexData::make_real(1.75));
    o.insert_or_assign("sol_dense_layers", FlexData::make_int(3));
    o.insert_or_assign("sol_local_radius", FlexData::make_int(2));
    o.insert_or_assign("sage_attn", FlexData::make_bool(true));
    o.insert_or_assign("i8_gemm", FlexData::make_bool(true));
    o.insert_or_assign("sol_key_block", FlexData::make_int(128));
  }
  GenerateImageStage st(&sess, "gi", std::vector<InEdge>{}, cfg);
  EXPECT_TRUE(st.config_error().empty());

  const FlexData& bag = st.accel_settings();
  const sol::Config from_bag = sol::config_from_flex(&bag);
  EXPECT_TRUE(from_bag.enabled == st.sol_config().enabled);
  EXPECT_TRUE(from_bag.tau == st.sol_config().tau);
  EXPECT_TRUE(from_bag.dense_layers == st.sol_config().dense_layers);
  EXPECT_TRUE(from_bag.local_radius == st.sol_config().local_radius);
  EXPECT_TRUE(from_bag.tau == 1.75f);
  EXPECT_TRUE(from_bag.dense_layers == 3);
  EXPECT_TRUE(from_bag.local_radius == 2);
  // Refused and corrected, in both views. What reaches a family has to
  // be what the stage SETTLED on, not what the graph asked for, or
  // every family repeats the validation and some of them get it wrong.
  EXPECT_TRUE(st.sol_config().key_block == 0);
  EXPECT_TRUE(from_bag.key_block == 0);
  EXPECT_TRUE(accel::tiers_on(&bag).size() == 3);
}

// A graph that asked for NOTHING still gets a bag, and it reads as
// every default.
//
// The case that matters for an out-of-tree family: `tiers_on` empty is
// how it knows to report nothing, and a bag that was merely absent
// would be indistinguishable from one where every tier was off. Both
// must produce the same answer, which is what the two halves check.
TEST(accel_settings, an_image_graph_that_asked_for_nothing_gets_defaults)
{
  Session sess;
  GenerateImageStage st(&sess, "gi", std::vector<InEdge>{},
                        FlexData::make_object());
  EXPECT_TRUE(st.config_error().empty());
  const FlexData& bag = st.accel_settings();
  EXPECT_TRUE(accel::tiers_on(&bag).empty());
  EXPECT_TRUE(accel::tiers_on(nullptr).empty());

  const sol::Config from_bag = sol::config_from_flex(&bag);
  const sol::Config from_null = sol::config_from_flex(nullptr);
  EXPECT_TRUE(!from_bag.enabled);
  EXPECT_TRUE(from_bag.enabled == from_null.enabled);
  EXPECT_TRUE(from_bag.dense_layers == from_null.dense_layers);
  EXPECT_TRUE(from_bag.local_radius == from_null.local_radius);
  EXPECT_TRUE(from_bag.key_block == from_null.key_block);
  // TAU IS THE ONE FIELD THE TWO READERS DISAGREE ON, on purpose.
  //
  // The absent-bag reader falls back to sol::Config's own default, which
  // is the value generate-video and every out-of-tree family still take.
  // generate-image SETTLES a lower one -- 0, the most accurate point
  // that still routes -- because 0.7 was reported to show artifacts on
  // real image graphs (the numbers are in the test below; the key's own
  // doc deliberately carries none, being config documentation a perf
  // figure would go stale in). So "asked for
  // nothing" is not the same as "no bag at all" for this field, and a
  // stage-settled value reaching the family is the whole point of
  // writing the bag rather than leaving it empty. THE ZERO IS ALSO WHY
  // the bag is written rather than left sparse: accel::real() falls
  // back on a key's ABSENCE, so a settled 0 that went unwritten would
  // reach the family as 1.0 -- the opposite end of the axis. Do not
  // restore the equality: it would only pass again by making one of the
  // two defaults wrong.
  EXPECT_TRUE(from_null.tau == 1.0f);
  EXPECT_TRUE(from_bag.tau == 0.0f);
  // The stage's own view agrees with the bag it wrote.
  EXPECT_TRUE(st.sol_config().dense_layers == from_bag.dense_layers);
  EXPECT_TRUE(st.sol_config().tau == from_bag.tau);
}

// THE TWO STAGES' DEFAULT TAU, spelled out, because nothing pinned
// either of them until a change to one surfaced as an unrelated
// equality failure in the test above.
//
// They differ deliberately: generate-image ships 0 and generate-video
// ships 1.0. The image default was 0.7 -- mid-plateau below a measured
// error step (rel-L2 0.0529 at tau <= 0.815 against 0.0680 at tau >=
// 0.82 on Krea-2 at 1024^2) -- and moved to 0 because 0.7 was reported
// to show artifacts in practice. The two readings agree: that same
// measurement put tau 0.7 at 18.9 dB PSNR against dense where two dense
// runs differ by 66.1 dB, so it was established as STABLE, never as
// faithful. Video sequences are ~5x longer, the published profile was
// measured there at 1.0/1.25/1.5, and neither the step nor the artifact
// report has been re-checked at that length, so video keeps 1.0.
//
// AND THE ZERO COSTS LITTLE WHERE THE TIER PAYS, which is what makes it
// affordable as a default. MEASURED on Krea-2 at 2048^2 on an M5 Pro,
// per DiT step: 21+ s dense, 18 s at tau 0, 17 s at tau 0.7 -- so tau 0
// takes three quarters of the available saving. At 1024^2 the whole
// tier was ~8% (143 s dense against 132 s at 0.7), i.e. inside a
// fanless box's own drift. Attention is O(seq^2) where the GEMMs are
// linear, so Sol is a HIGH-RESOLUTION tier and a share measured at
// 1024^2 says nothing about 2048^2. These numbers live here and not in
// the key's doc: that doc is config documentation, where a perf figure
// goes stale without anyone noticing.
//
// If one of these moves, this test should fail rather than the
// divergence being discovered somewhere else.
TEST(accel_settings, the_two_stages_ship_different_default_tau)
{
  Session sess;
  GenerateImageStage img(&sess, "gi", std::vector<InEdge>{},
                         FlexData::make_object());
  GenerateVideoStage vid(&sess, "gv", std::vector<InEdge>{},
                         FlexData::make_object());
  EXPECT_TRUE(img.config_error().empty());
  EXPECT_TRUE(vid.config_error().empty());
  EXPECT_TRUE(img.sol_config().tau == 0.0f);
  EXPECT_TRUE(vid.sol_config().tau == 1.0f);
  // Whatever each stage settled is what its bag carries, so a family
  // reading the bag cannot see a different number from the log line.
  EXPECT_TRUE(sol::config_from_flex(&img.accel_settings()).tau ==
              0.0f);
  EXPECT_TRUE(sol::config_from_flex(&vid.accel_settings()).tau == 1.0f);
}

// MotionCache off generate-video: what the graph said, SETTLED, in the
// bag -- and a value that would make the estimate meaningless falls back
// to the shipped one rather than reaching a family as said.
TEST(accel_settings, the_video_stage_settles_motion_cache)
{
  namespace mc = vpipe::genai::motion_cache;
  Session sess;
  {
    FlexData cfg = FlexData::make_object();
    auto o = cfg.as_object();
    o.insert_or_assign("motion_cache", FlexData::make_bool(true));
    o.insert_or_assign("motion_cache_threshold", FlexData::make_real(0.2));
    o.insert_or_assign("motion_cache_strength", FlexData::make_real(0.5));
    o.insert_or_assign("motion_cache_warmup", FlexData::make_int(3));
    o.insert_or_assign("motion_cache_max_skips", FlexData::make_int(1));
    o.insert_or_assign("motion_cache_start", FlexData::make_real(0.2));
    o.insert_or_assign("motion_cache_end", FlexData::make_real(0.9));
    o.insert_or_assign("motion_cache_subsample", FlexData::make_int(4));
    GenerateVideoStage st(&sess, "gv", std::vector<InEdge>{}, cfg);
    EXPECT_TRUE(st.config_error().empty());
    const FlexData& bag = st.accel_settings();
    const mc::Config c = mc::config_from_flex(&bag);
    EXPECT_TRUE(c.enabled && c.threshold == 0.2 &&
                c.motion_strength == 0.5 && c.warmup_steps == 3 &&
                c.max_skips == 1 && c.start == 0.2 && c.end == 0.9 &&
                c.subsample == 4);
    const auto on = accel::tiers_on(&bag);
    EXPECT_TRUE(on.size() == 1 && on[0] == accel::kMotionCache);
  }
  {
    FlexData cfg = FlexData::make_object();
    auto o = cfg.as_object();
    o.insert_or_assign("motion_cache", FlexData::make_bool(true));
    o.insert_or_assign("motion_cache_threshold", FlexData::make_real(-1));
    o.insert_or_assign("motion_cache_strength", FlexData::make_real(-2));
    o.insert_or_assign("motion_cache_warmup", FlexData::make_int(-1));
    o.insert_or_assign("motion_cache_max_skips", FlexData::make_int(0));
    // An inverted window is empty, not merely narrow.
    o.insert_or_assign("motion_cache_start", FlexData::make_real(0.9));
    o.insert_or_assign("motion_cache_end", FlexData::make_real(0.2));
    o.insert_or_assign("motion_cache_subsample", FlexData::make_int(0));
    GenerateVideoStage st(&sess, "gv", std::vector<InEdge>{}, cfg);
    EXPECT_TRUE(st.config_error().empty());
    const FlexData& bag = st.accel_settings();
    const mc::Config c = mc::config_from_flex(&bag);
    const mc::Config d;
    EXPECT_TRUE(c.enabled);
    EXPECT_TRUE(c.threshold == d.threshold);
    EXPECT_TRUE(c.motion_strength == d.motion_strength);
    EXPECT_TRUE(c.warmup_steps == d.warmup_steps);
    EXPECT_TRUE(c.max_skips == 1);
    EXPECT_TRUE(c.start == d.start && c.end == d.end);
    EXPECT_TRUE(c.subsample == d.subsample);
  }
  {
    // Nothing asked: off, at the shipped values.
    GenerateVideoStage st(&sess, "gv", std::vector<InEdge>{},
                          FlexData::make_object());
    const FlexData& bag = st.accel_settings();
    const mc::Config c = mc::config_from_flex(&bag);
    const mc::Config d;
    EXPECT_TRUE(!c.enabled && c.threshold == d.threshold &&
                c.warmup_steps == d.warmup_steps && c.end == d.end);
    EXPECT_TRUE(accel::tiers_on(&bag).empty());
  }
}
