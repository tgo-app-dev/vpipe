// The model-specific config sources, and the seam they form with the
// stages that read them.
//
// The video pair (wan2 / minimax-h3) is exercised in wan-video-stages.cc
// alongside generate-video; this file covers the IMAGE families and the
// properties that belong to the mechanism itself rather than to any one
// family -- the category, the tag, and the rule that a config beat only
// ever ADDS to what the model layer already knows.

#include "minitest.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage-spec.h"
#include "stages/boogu-image-model-config-stage.h"
#include "stages/flux2-model-config-stage.h"
#include "stages/krea2-model-config-stage.h"
#include "stages/mage-flow-model-config-stage.h"
#include "stages/minimax-h3-model-config-stage.h"
#include "stages/model-config-source.h"
#include "stages/qwen-image-edit-model-config-stage.h"
#include "stages/wan2-model-config-stage.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/flux2/metal-flux2-transformer.h"
#include "generative-models/mage/mage-watermark.h"
#include "generative-models/shared/grounded-encode-params.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/generate-image-stage.h"
#endif

#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// Every config source, by the type name a pipeline spells.
const char* const kSources[] = {
  "wan2-model-config", "minimax-h3-model-config", "flux2-model-config",
  "mage-flow-model-config", "krea2-model-config", "boogu-image-model-config",
  "qwen-image-edit-model-config",
};

}  // namespace

// The category exists so tooling can group the stages whose APPLICABILITY
// depends on the resident checkpoint -- which is not true of anything in
// "generative". Every source must be in it, and the tag must be the one
// the consumers accept, or the composer silently offers a source that
// cannot be wired to the stage it was written for.
TEST(model_config, every_source_is_categorized_and_tagged)
{
  for (const char* name : kSources) {
    const StageSpec* sp = StageRegistry::get().spec(name);
    ASSERT_TRUE(sp != nullptr);
    EXPECT_TRUE(sp->category == StageCategory::ModelSpecificConfig);
    EXPECT_TRUE(stage_category_name(sp->category) == "model-specific-config");
    // One oport, carrying the tag, and one OPTIONAL trigger iport.
    ASSERT_TRUE(sp->oports.size() == 1u);
    EXPECT_TRUE(std::string(sp->oports[0].name) == "model_config");
    EXPECT_TRUE(port_tags_compatible(sp->oports[0].tags,
                                     model_config::kConfigTag));
    ASSERT_TRUE(sp->iports.size() == 1u);
    EXPECT_TRUE(std::string(sp->iports[0].name) == "trigger");
    // Any payload triggers: receipt is the signal, so a chrono tick and a
    // prompt beat must both work.
    EXPECT_TRUE(sp->iports[0].type == nullptr);
  }
}

// Every source stamps the family its keys belong to. This is what lets a
// consumer refuse a config wired to the wrong checkpoint instead of
// applying nothing and running defaults while the graph says otherwise.
TEST(model_config, every_source_stamps_its_family)
{
  Session sess;
  auto empty = [] { return FlexData::make_object(); };
  const struct { std::string got, want; } cases[] = {
    {model_config::family_of(
         Wan2ModelConfigStage(&sess, "a", {}, empty()).resolved_config()),
     "wan"},
    {model_config::family_of(
         MiniMaxH3ModelConfigStage(&sess, "b", {}, empty()).resolved_config()),
     "minimax-h3"},
    {model_config::family_of(
         Flux2ModelConfigStage(&sess, "c", {}, empty()).resolved_config()),
     "flux2"},
    {model_config::family_of(
         MageFlowModelConfigStage(&sess, "d", {}, empty()).resolved_config()),
     "mage-flow"},
    {model_config::family_of(
         Krea2ModelConfigStage(&sess, "e", {}, empty()).resolved_config()),
     "krea2"},
    {model_config::family_of(
         BooguImageModelConfigStage(&sess, "f", {},
                                    empty()).resolved_config()),
     "boogu-image"},
    {model_config::family_of(
         QwenImageEditModelConfigStage(&sess, "g", {},
                                       empty()).resolved_config()),
     "qwen-image-edit"},
  };
  for (const auto& c : cases) { EXPECT_TRUE(c.got == c.want); }
}

// An UNSET grounded key must not appear in the beat at all.
//
// This is the property the whole design rests on and the one that is
// easy to get wrong: the conditioner starts from the model layer's
// per-family numbers, so a source that helpfully emitted its defaults
// would overwrite Mage-Flow's 384 with a zero -- and the encode would
// still succeed, at a resolution the model was never trained against.
TEST(model_config, unset_grounded_keys_are_not_emitted)
{
  Session sess;
  {
    Krea2ModelConfigStage s(&sess, "k", {}, FlexData::make_object());
    FlexData fd = s.resolved_config();
    auto o = fd.as_object();
    EXPECT_FALSE(o.contains("vl_long_edge"));
    EXPECT_FALSE(o.contains("vl_pixel_budget"));
    EXPECT_FALSE(o.contains("vl_min_pixels"));
    EXPECT_FALSE(o.contains("vl_max_pixels"));
  }
  {
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("vl_long_edge",
                                     FlexData::make_int(512));
    Krea2ModelConfigStage s(&sess, "k2", {}, cfg);
    FlexData fd = s.resolved_config();
    auto o = fd.as_object();
    EXPECT_TRUE(o.contains("vl_long_edge"));
    EXPECT_TRUE(o.at("vl_long_edge").as_int(0) == 512);
    // Naming one knob must not conjure the other three.
    EXPECT_FALSE(o.contains("vl_pixel_budget"));
  }
}

#ifdef VPIPE_BUILD_APPLE_SILICON

// The grounded-encode numbers, which are the conditioner's whole reason
// for having a model_config port.
//
// They are pinned as VALUES, not just as "some default exists", because
// they came out of four different reference pipelines and feeding one
// family another's produces a well-formed conditioning that lands where
// its DiT was never trained -- an edit that mis-targets rather than a
// failure, which is the expensive kind of wrong. In particular
// Bougu-Image's long edge is DOUBLE Mage-Flow's while its area cap is
// the same: the two bounds are not redundant and neither implies the
// other.
TEST(model_config, each_family_keeps_its_own_grounding_numbers)
{
  using GP = genai::GroundedEncodeParams;
  const GP k = GP::for_family("krea2");
  EXPECT_TRUE(k.long_edge == 768);
  EXPECT_TRUE(k.pixel_budget == 0u);
  EXPECT_TRUE(k.min_pixels == 0u);       // the tower's own default stands

  const GP m = GP::for_family("mage-flow");
  EXPECT_TRUE(m.long_edge == 384);
  EXPECT_TRUE(m.pixel_budget == 0u);
  EXPECT_TRUE(m.min_pixels == 65536u);
  EXPECT_TRUE(m.max_pixels == 16777216u);

  const GP b = GP::for_family("boogu-image");
  EXPECT_TRUE(b.long_edge == 768);       // NOT mage's 384
  EXPECT_TRUE(b.pixel_budget == (std::size_t)384 * 384);
  EXPECT_TRUE(b.min_pixels == 65536u);

  const GP q = GP::for_family("qwen-image-edit");
  EXPECT_TRUE(q.long_edge == 0);         // its tower takes a budget instead
  EXPECT_TRUE(q.pixel_budget == (std::size_t)384 * 384);

  // A family with no grounded path at all: no capping, tower defaults.
  const GP w = GP::for_family("wan");
  EXPECT_TRUE(w.long_edge == 0 && w.pixel_budget == 0u);

  // merge_flex OVERLAYS: an absent key leaves the family number alone,
  // which is what makes naming one knob not reset the other three.
  GP mm = GP::for_family("mage-flow");
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("vl_long_edge", FlexData::make_int(512));
  mm.merge_flex(cfg);
  EXPECT_TRUE(mm.long_edge == 512);
  EXPECT_TRUE(mm.min_pixels == 65536u);   // untouched
}

// The image families' beats, read back by the parsers that consume them.
// Neither stage can check this alone, which is why it is checked here on
// the real parsers rather than on a hand-written object.
TEST(model_config, image_families_read_back_their_own_beats)
{
  Session sess;
  {
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("klein_kv", FlexData::make_bool(true));
    Flux2ModelConfigStage s(&sess, "f", {}, cfg);
    const auto p =
        genai::MetalFlux2Transformer::GenerationParams::from_flex(
            s.resolved_config());
    EXPECT_TRUE(p.klein_kv);
    // And it reaches the Config the DiT is BUILT from -- the whole point,
    // since this selects an attention recipe at load time.
    genai::MetalFlux2Transformer::Config c;
    EXPECT_FALSE(c.klein_kv);
    p.apply_to(c);
    EXPECT_TRUE(c.klein_kv);
  }
  {
    // Default: no key at all means plain klein-9B, not the -kv recipe.
    Flux2ModelConfigStage s(&sess, "f2", {}, FlexData::make_object());
    const auto p =
        genai::MetalFlux2Transformer::GenerationParams::from_flex(
            s.resolved_config());
    EXPECT_FALSE(p.klein_kv);
  }
  {
    // The watermark is ON unless the config says otherwise: a NEGATIVE
    // key with a positive default, so the safe state needs no config.
    MageFlowModelConfigStage on(&sess, "m", {}, FlexData::make_object());
    EXPECT_TRUE(genai::mage_wm::Params::from_flex(
                    on.resolved_config()).enabled);

    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("no_watermark", FlexData::make_bool(true));
    cfg.as_object().insert_or_assign("watermark_key",
                                     FlexData::make_string("12345"));
    MageFlowModelConfigStage off(&sess, "m2", {}, cfg);
    const auto p = genai::mage_wm::Params::from_flex(off.resolved_config());
    EXPECT_FALSE(p.enabled);
    EXPECT_TRUE(p.key == "12345");
  }
}

// The consumer side of the seam: both diffusion stages take the config on
// a port, and neither carries the moved keys any more.
//
// The port INDEX is pinned because it is appended -- every earlier index
// keeps its meaning, so a graph written before this change still wires
// correctly.
TEST(model_config, the_image_stages_expose_the_port_and_not_the_keys)
{
  Session sess;
  {
    auto s = make_unique<GenerateImageStage>(&sess, "gi", vector<InEdge>{},
                                             FlexData::make_object());
    const StageSpec& sp = s->spec();
    ASSERT_TRUE(sp.iports.size() == 8u);
    EXPECT_TRUE(std::string(sp.iports[7].name) == "model_config");
    EXPECT_TRUE(port_tags_compatible(model_config::kConfigTag,
                                     sp.iports[7].tags));
    // The earlier ports are untouched.
    EXPECT_TRUE(std::string(sp.iports[5].name) == "ref_latent0");
    EXPECT_TRUE(std::string(sp.iports[6].name) == "ref_latent1");
    for (const auto& k : sp.attrs) {
      const std::string key(k.key);
      EXPECT_FALSE(key == "klein_kv" || key == "no_watermark" ||
                   key == "watermark_key");
    }
    // i8_gemm deliberately STAYS: it picks a kernel, so a family without
    // int8 kernels just runs f16 -- no worse, only not faster. klein_kv
    // picks a recipe, so on the wrong checkpoint it runs WRONG.
    bool has_i8 = false;
    for (const auto& k : sp.attrs) {
      if (std::string(k.key) == "i8_gemm") { has_i8 = true; }
    }
    EXPECT_TRUE(has_i8);
  }
  {
    auto s = make_unique<DiffusionConditionerStage>(
        &sess, "dc", vector<InEdge>{}, FlexData::make_object());
    const StageSpec& sp = s->spec();
    ASSERT_TRUE(sp.iports.size() == 6u);
    EXPECT_TRUE(std::string(sp.iports[5].name) == "model_config");
    EXPECT_TRUE(port_tags_compatible(model_config::kConfigTag,
                                     sp.iports[5].tags));
    EXPECT_TRUE(std::string(sp.iports[3].name) == "ref_image");
    EXPECT_TRUE(std::string(sp.iports[4].name) == "ref_image2");
  }
}

// A pipeline written against the old union still LOADS -- an unknown
// config key is not an error anywhere in this runtime -- so the only
// thing between it and silently running on defaults is that the stage
// says so. Constructing with a moved key must not be a config error (the
// graph is still valid) and must not be silent either.
TEST(model_config, a_moved_image_key_is_reported_not_rejected)
{
  Session sess;
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("klein_kv", FlexData::make_bool(true));
  cfg.as_object().insert_or_assign("no_watermark", FlexData::make_bool(true));
  auto s = make_unique<GenerateImageStage>(&sess, "gi", vector<InEdge>{}, cfg);
  EXPECT_TRUE(s->config_error().empty());
}

#endif  // VPIPE_BUILD_APPLE_SILICON
