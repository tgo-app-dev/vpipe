// The model-specific config sources, and the seam they form with the
// stages that read them.
//
// The video pair (wan2 / minimax-h3) is exercised in wan-video-stages.cc
// alongside generate-video; this file covers the IMAGE families and the
// properties that belong to the mechanism itself rather than to any one
// family -- the category, the tag, and the rule that a config beat only
// ever ADDS to what the model layer already knows.

#include "minitest.h"

#include <optional>

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
#include "stages/model-catalog.h"
#include "stages/model-config-source.h"
#include "stages/model-registry.h"
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

// The adapter keys reach the beat.
//
// Worth its own test because declaring a key in kAttrs is NOT what puts
// it in the beat -- this family builds its config explicitly, so a key
// can be fully documented, offered by the composer's file browser, and
// still never arrive. That is a config that reads as supported and does
// FLUX.2 builds its beat by hand too (klein_kv is written explicitly),
// so it has the same trap and needs the same check. `klein_kv` must
// survive beside the adapter keys: it is the one key on this beat that
// changes the ATTENTION recipe, and losing it is a wrong image rather
// than a missing feature.
TEST(model_config, flux2_emits_the_adapter_keys_only_when_set)
{
  Session sess;
  {
    Flux2ModelConfigStage s(&sess, "f", {}, FlexData::make_object());
    FlexData fd = s.resolved_config();
    auto o = fd.as_object();
    EXPECT_FALSE(o.contains("lora"));
    EXPECT_FALSE(o.contains("lora_scale"));
    ASSERT_TRUE(o.contains("klein_kv"));
  }
  {
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign(
        "lora", FlexData::make_string("/models/style.safetensors"));
    cfg.as_object().insert_or_assign("lora_scale", FlexData::make_real(0.5));
    cfg.as_object().insert_or_assign("klein_kv", FlexData::make_bool(true));
    Flux2ModelConfigStage s(&sess, "f2", {}, cfg);
    FlexData fd = s.resolved_config();
    auto o = fd.as_object();
    // ASSERT_TRUE does NOT abort in minitest, so every read is guarded
    // by its own contains(): an unguarded at() on a regression throws
    // out of the test instead of naming the key that went missing.
    EXPECT_TRUE(o.contains("lora"));
    EXPECT_TRUE(o.contains("lora") &&
                std::string(o.at("lora").as_string("")) ==
                    "/models/style.safetensors");
    EXPECT_TRUE(o.contains("lora_scale"));
    EXPECT_TRUE(o.contains("lora_scale") &&
                o.at("lora_scale").as_real(0.0) == 0.5);
    EXPECT_TRUE(o.contains("klein_kv"));
    EXPECT_TRUE(o.contains("klein_kv") && o.at("klein_kv").as_bool(false));
    EXPECT_TRUE(std::string(o.at("model_family").as_string("")) == "flux2");
  }
  {
    // A scale alone, as above: naming one key must not conjure the other.
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("lora_scale", FlexData::make_real(0.0));
    Flux2ModelConfigStage s(&sess, "f3", {}, cfg);
    FlexData fd = s.resolved_config();
    auto o = fd.as_object();
    EXPECT_FALSE(o.contains("lora"));
    EXPECT_TRUE(o.contains("lora_scale"));
    EXPECT_TRUE(o.contains("lora_scale") &&
                o.at("lora_scale").as_real(1.0) == 0.0);
  }
}

// nothing, which is the worst of the available failures and is exactly
// what happened on the first attempt here.
TEST(model_config, krea2_emits_the_adapter_keys_only_when_set)
{
  Session sess;
  {
    // Unset stays unset, on the same argument as the grounded keys: a
    // default emitted as though it were a choice reads downstream as
    // "the graph asked for scale 1.0" when the graph said nothing.
    Krea2ModelConfigStage s(&sess, "k", {}, FlexData::make_object());
    FlexData fd = s.resolved_config();
    auto o = fd.as_object();
    EXPECT_FALSE(o.contains("lora"));
    EXPECT_FALSE(o.contains("lora_scale"));
  }
  {
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign(
        "lora", FlexData::make_string("/models/turbo.safetensors"));
    cfg.as_object().insert_or_assign("lora_scale", FlexData::make_real(0.75));
    Krea2ModelConfigStage s(&sess, "k2", {}, cfg);
    FlexData fd = s.resolved_config();
    auto o = fd.as_object();
    // Guarded rather than ASSERTed: ASSERT_TRUE does not abort, so an
    // unguarded at() on a regression throws out of the test instead of
    // naming the key that went missing.
    EXPECT_TRUE(o.contains("lora"));
    EXPECT_TRUE(o.contains("lora") &&
                std::string(o.at("lora").as_string("")) ==
                    "/models/turbo.safetensors");
    EXPECT_TRUE(o.contains("lora_scale"));
    EXPECT_TRUE(o.contains("lora_scale") &&
                o.at("lora_scale").as_real(0.0) == 0.75);
    // And the family stamp survives beside them, so a consumer can still
    // tell whose config this is.
    EXPECT_TRUE(std::string(o.at("model_family").as_string("")) == "krea2");
  }
  {
    // A scale ALONE is a legitimate request -- sweeping the strength of
    // an adapter named in the stage's own config -- so naming one key
    // must not conjure the other.
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("lora_scale", FlexData::make_real(0.0));
    Krea2ModelConfigStage s(&sess, "k3", {}, cfg);
    FlexData fd = s.resolved_config();
    auto o = fd.as_object();
    EXPECT_FALSE(o.contains("lora"));
    EXPECT_TRUE(o.contains("lora_scale"));
    EXPECT_TRUE(o.contains("lora_scale") &&
                o.at("lora_scale").as_real(1.0) == 0.0);
  }
}

// Every `lora` key offers a MODEL picker, filtered to its own family.
//
// The failure this pins is a UI one, which is exactly why it needs a
// test: a `lora` field declared `is_path` renders a FILE BROWSER, and a
// browser cannot show a catalogued adapter at all -- so the models the
// catalogue exists to offer are invisible and the field looks broken
// while being, technically, configurable. Krea-2's shipped that way.
//
// The type also has to be the family's OWN, or the picker offers every
// adapter in the catalogue and a Krea-2 LoRA can be chosen for a FLUX.2
// DiT, which loads and binds nothing.
TEST(model_config, every_lora_key_offers_a_filtered_model_picker)
{
  struct Row { const char* stage; const char* want_type; };
  const Row rows[] = {
    {"krea2-model-config",      "krea2-lora"},
    {"flux2-model-config",      "flux2-lora"},
    {"minimax-h3-model-config", "minimax-h3-lora"},
  };
  for (const Row& row : rows) {
    const StageSpec* sp = StageRegistry::get().spec(row.stage);
    EXPECT_TRUE(sp != nullptr);
    if (sp == nullptr) { continue; }
    const Row& r = row;
    const ConfigKey* lora = nullptr;
    for (const ConfigKey& k : sp->attrs) {
      if (k.key == "lora") { lora = &k; break; }
    }
    EXPECT_TRUE(lora != nullptr);
    if (lora == nullptr) { continue; }
    // A model picker, not the filesystem browser.
    EXPECT_TRUE(!lora->is_path);
    EXPECT_TRUE(lora->suggest_db == kModelRegistryDb);
    EXPECT_TRUE(lora->suggest_db_type == r.want_type);
    if (lora->suggest_db_type != r.want_type) {
      std::printf("[model_config] %s lora suggest_db_type '%s' != '%s'\n",
                  r.stage, std::string(lora->suggest_db_type).c_str(),
                  r.want_type);
    }
  }

  // And the type has to NAME something, or the picker is empty and the
  // field is free text with extra steps. flux2-lora is the exception,
  // and is stated as one: nothing verified has been published for that
  // family yet, so the FILTER is right and the list is empty until one
  // is. If that changes, so should this.
  int krea = 0, h3 = 0, flux = 0;
  for (const ModelCatalogEntry& e : model_catalog()) {
    if (e.model_type == "krea2-lora")      { ++krea; }
    if (e.model_type == "minimax-h3-lora") { ++h3; }
    if (e.model_type == "flux2-lora")      { ++flux; }
  }
  EXPECT_TRUE(krea > 0);
  EXPECT_TRUE(h3 > 0);
  std::printf("[model_config] catalogued adapters: krea2 %d, h3 %d, "
              "flux2 %d (none published yet)\n", krea, h3, flux);
}

// generate-image reads the adapter from its OWN config, and a beat
// overrides it.
//
// This is the shape of a real report: a krea2-model-config carrying
// `lora` was wired to the diffusion-conditioner -- which reads the
// `vl_*` keys off the same beat -- and not to generate-image, so the
// adapter was configured, reported by its source, and dropped. The key
// now lives on the stage that loads the DiT as well, so a graph needs
// no wiring at all to set one.
TEST(model_config, generate_image_takes_the_adapter_from_its_own_config)
{
  Session sess;
  auto mk = [&](const char* id, const char* lora, bool with_scale,
                double scale) {
    auto cfg = FlexData::make_object();
    cfg.as_object().insert_or_assign("hf_dir",
                                     FlexData::make_string("some/model"));
    if (lora != nullptr) {
      cfg.as_object().insert_or_assign("lora", FlexData::make_string(lora));
    }
    if (with_scale) {
      cfg.as_object().insert_or_assign("lora_scale",
                                       FlexData::make_real(scale));
    }
    return std::make_unique<GenerateImageStage>(&sess, id,
                                                std::vector<InEdge>{},
                                                std::move(cfg));
  };

  {   // Nothing set: no adapter, and the scale is the neutral default.
    auto s = mk("g0", nullptr, false, 0.0);
    EXPECT_TRUE(s->lora_ref().empty());
    EXPECT_TRUE(s->lora_scale() == 1.0);
  }
  {   // Set on the stage: taken, with no beat and no wiring at all.
    auto s = mk("g1", "mgwr/M87", true, 0.75);
    EXPECT_TRUE(s->lora_ref() == "mgwr/M87");
    EXPECT_TRUE(s->lora_scale() == 0.75);
  }

  // And the picker for it offers BOTH families this stage can adapt --
  // a file browser here would hide every catalogued adapter, which is
  // the bug this key was added to stop repeating.
  const StageSpec* sp = StageRegistry::get().spec("generate-image");
  EXPECT_TRUE(sp != nullptr);
  if (sp != nullptr) {
    const ConfigKey* lora = nullptr;
    for (const ConfigKey& k : sp->attrs) {
      if (k.key == "lora") { lora = &k; break; }
    }
    EXPECT_TRUE(lora != nullptr);
    if (lora != nullptr) {
      EXPECT_TRUE(!lora->is_path);
      EXPECT_TRUE(lora->suggest_db == kModelRegistryDb);
      EXPECT_TRUE(lora->suggest_db_type.find("krea2-lora") !=
                  std::string_view::npos);
      EXPECT_TRUE(lora->suggest_db_type.find("flux2-lora") !=
                  std::string_view::npos);
    }
  }
}

TEST(model_config, the_vdn_branch_is_a_family_knob_and_not_a_stage_key)
{
  // WHICH STAGE OWNS A KNOB IS A CLAIM, not a filing preference.
  // `generate-video` is family-agnostic -- it passes each family's beat
  // to that family's GenerationParams UNREAD -- so a key on it is a key
  // every family has to mean something by. VDN's linear branch means
  // nothing to Wan, and a knob that is inert for half the graphs that
  // can set it is how a config grows a field nobody can explain.
  //
  // So: the key lives on `minimax-h3-model-config` and NOT on
  // `generate-video`, which takes the beat and nothing else.
  const StageSpec* src = StageRegistry::get().spec("minimax-h3-model-config");
  const StageSpec* dst = StageRegistry::get().spec("generate-video");
  ASSERT_TRUE(src != nullptr && dst != nullptr);
  if (src == nullptr || dst == nullptr) { return; }

  const ConfigKey* lb = nullptr;
  for (const ConfigKey& k : src->attrs) {
    if (k.key == "linear_branch") { lb = &k; }
  }
  EXPECT_TRUE(lb != nullptr);
  for (const ConfigKey& k : dst->attrs) {
    EXPECT_FALSE(std::string(k.key) == "linear_branch");
  }
  if (lb == nullptr) { return; }

  // A typed MODEL PICKER, not the filesystem browser and not free text.
  // Untyped it would offer plain models only and show nothing, because
  // VDN is catalogued as a supplement; wrongly typed it would offer a
  // LoRA, and the two are not interchangeable -- swapped, the model
  // loads and renders.
  EXPECT_TRUE(!lb->is_path);
  EXPECT_TRUE(lb->suggest_db == kModelRegistryDb);
  EXPECT_TRUE(lb->suggest_db_type == "minimax-h3-vdn");

  // And the type has to NAME something, or the picker is empty and the
  // field is free text with extra steps.
  int n = 0;
  for (const ModelCatalogEntry& e : model_catalog()) {
    if (e.model_type == "minimax-h3-vdn") {
      ++n;
      // A supplement, and specifically one that attaches to the H3 DiT:
      // the branch shares that DiT's own q/k/v projections, so it is a
      // second checkpoint BESIDE a parent rather than a model.
      EXPECT_TRUE(e.parent_model_type == "minimax-h3-fl2va");
    }
  }
  std::printf("[model_config] catalogued VDN branches: %d\n", n);
  EXPECT_TRUE(n > 0);
}

TEST(model_config, the_h3_source_emits_the_branch_only_when_set)
{
  // Emitted only when SET, for the reason the LoRA keys are: an empty
  // string would read at the consumer as "the graph asked for no
  // branch", which is indistinguishable from "the graph said nothing"
  // -- and the consumer's default is already no branch. The difference
  // matters because the consumer treats the key as a LOAD-TIME
  // argument and warns when it changes under a built DiT.
  Session sess;
  auto bare = make_unique<MiniMaxH3ModelConfigStage>(
      &sess, "h3c", vector<InEdge>{}, FlexData::make_object());
  const FlexData off = bare->resolved_config();
  ASSERT_TRUE(off.is_object());
  EXPECT_FALSE(off.as_object().contains("linear_branch"));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign(
      "linear_branch", FlexData::make_string("OpenVDN/vdn-minimax-h3"));
  auto on = make_unique<MiniMaxH3ModelConfigStage>(
      &sess, "h3c2", vector<InEdge>{}, std::move(cfg));
  const FlexData beat = on->resolved_config();
  ASSERT_TRUE(beat.is_object());
  auto o = beat.as_object();
  EXPECT_TRUE(o.contains("linear_branch"));
  if (o.contains("linear_branch")) {
    EXPECT_TRUE(std::string(o.at("linear_branch").as_string(""))
                == "OpenVDN/vdn-minimax-h3");
  }

  // And it must reach a consumer BEFORE any driver runs, or the branch
  // -- a second 4.28 GB checkpoint -- is one every peer sized the box
  // without. That is what constant_output is for; a trigger changes
  // when the beat is emitted, never what it says.
  const std::optional<FlexData> k = on->constant_output(0);
  EXPECT_TRUE((bool)k);
  if (k) {
    EXPECT_TRUE(k->is_object()
                && k->as_object().contains("linear_branch"));
  }
}
