// The image family registry: the seam that lets an out-of-tree image
// model join `generate-image` instead of bringing a stage of its own.
//
// It is the counterpart to the video one, and what it must get right is
// the same short list -- so this file is deliberately the same tests.
//
// THE ONE THAT MATTERS MOST IS FIRST-WINS. Two plugins shipping a family
// for one tag is a deployment mistake, and the failure if the second
// silently replaced the first is that a model runs under someone else's
// code with the right name in every log line. Refusing the second is
// what makes that visible.
//
// AND `claims` MUST NOT BE ABLE TO TAKE THE HOST DOWN. It runs for every
// registered family on every model resolve, it reads the filesystem, and
// it is a plugin's code -- so a throwing probe is a bug in that family
// and must cost only that family.

#include "generative-models/image-model-registry.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "stages/generate-image-stage.h"
#include "generative-models/conditioner-profile.h"
#include "generative-models/detect-profile.h"
#include "plugin/plugin-context.h"
#include "stages/diffusion-conditioner-stage.h"
#include "minitest.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;

namespace {

// A family that claims whatever `want` says, and counts its probes.
class Fake final : public ImageModelFamily {
public:
  Fake(std::string tag, std::string want, bool throws = false)
      : _tag(std::move(tag)), _want(std::move(want)), _throws(throws) {}

  std::string_view tag() const noexcept override { return _tag; }

  bool claims(const std::string& root, const std::string&) const override
  {
    ++probes;
    if (_throws) { throw std::runtime_error("probe went wrong"); }
    return root == _want;
  }

  std::unique_ptr<ImageGenerator> load(const ImageModelCreateArgs&) override
  {
    return nullptr;      // never loaded here; the seam is what is tested
  }

  mutable int probes = 0;

private:
  std::string _tag, _want;
  bool        _throws;
};

// Unique per run, because the registry is process-wide and every other
// test in this binary shares it.
std::string
tag_(const char* base)
{
  static int n = 0;
  return std::string("test-image-") + base + "-" + std::to_string(++n);
}

}  // namespace

TEST(image_model_registry, a_registered_family_claims_its_own_checkpoint)
{
  const std::string t = tag_("claims");
  auto f = std::make_unique<Fake>(t, "/models/acme");
  const Fake* raw = f.get();
  EXPECT_TRUE(ImageModelRegistry::get().add(std::move(f)));
  EXPECT_TRUE(ImageModelRegistry::get().find(t) == raw);

  ImageModelFamily* got =
      ImageModelRegistry::get().claim_for(nullptr, "/models/acme", "");
  EXPECT_TRUE(got == raw);
  // ...and does NOT claim someone else's, which is the half that matters:
  // claiming a checkpoint that is not yours loads at full cost and
  // computes nonsense.
  EXPECT_TRUE(ImageModelRegistry::get().claim_for(nullptr, "/models/other",
                                                  "") != got);
}

// FIRST-WINS, and the second registration is refused rather than
// ignored: a plugin that shipped a colliding tag should be able to see
// that it did.
TEST(image_model_registry, a_second_family_cannot_take_a_taken_tag)
{
  const std::string t = tag_("collide");
  auto a = std::make_unique<Fake>(t, "/models/a");
  const Fake* first = a.get();
  EXPECT_TRUE(ImageModelRegistry::get().add(std::move(a)));

  auto b = std::make_unique<Fake>(t, "/models/b");
  EXPECT_TRUE(!ImageModelRegistry::get().add(std::move(b)));
  // The family already present keeps the tag AND keeps its behaviour --
  // "refused" has to mean the first one still answers, not merely that
  // add() returned false.
  EXPECT_TRUE(ImageModelRegistry::get().find(t) == first);
  EXPECT_TRUE(ImageModelRegistry::get().claim_for(nullptr, "/models/a", "") ==
              first);
  EXPECT_TRUE(ImageModelRegistry::get().claim_for(nullptr, "/models/b", "") !=
              first);
}

// A NAMELESS FAMILY IS REFUSED. `find` and first-wins are both keyed on
// the tag, so an empty one would be a family nothing can name and every
// later empty tag would collide with.
TEST(image_model_registry, a_family_that_cannot_name_itself_is_refused)
{
  EXPECT_TRUE(!ImageModelRegistry::get().add(nullptr));
  auto nameless = std::make_unique<Fake>("", "/models/x");
  EXPECT_TRUE(!ImageModelRegistry::get().add(std::move(nameless)));
}

// A THROWING PROBE COSTS ONLY ITS OWN FAMILY. It runs for every family
// on every resolve and it is plugin code, so the registry has to keep
// asking the others -- and must never let it out.
TEST(image_model_registry, a_throwing_probe_is_skipped_not_fatal)
{
  const std::string bad = tag_("throws");
  const std::string good = tag_("after");
  EXPECT_TRUE(ImageModelRegistry::get().add(
      std::make_unique<Fake>(bad, "/models/boom", /*throws=*/true)));
  auto g = std::make_unique<Fake>(good, "/models/boom");
  const Fake* ok = g.get();
  EXPECT_TRUE(ImageModelRegistry::get().add(std::move(g)));

  // Registration order is probe order, so the thrower is asked first.
  ImageModelFamily* got =
      ImageModelRegistry::get().claim_for(nullptr, "/models/boom", "");
  EXPECT_TRUE(got == ok);
}

// The defaults are the ones a family that implements nothing gets, and
// each one has to be the answer that makes the STAGE fall back rather
// than believe a number.
TEST(image_model_registry, the_defaults_decline_rather_than_guess)
{
  Fake f("unused", "/nowhere");
  int gh = 0, gw = 0;
  f.size_grid("/nowhere", &gh, &gw);
  // 16 because the stage's own contract is a multiple of 16, so a family
  // with no opinion rounds to something every VAE here can tile.
  EXPECT_TRUE(gh == 16 && gw == 16);
  // 0 means "cannot say", and the stage then falls back to its default
  // size instead of inferring one from a ratio it guessed.
  EXPECT_TRUE(f.latent_scale("/nowhere") == 0);
  // Empty is a HOLE in the plan, not a zero cost -- but it is the honest
  // answer for a family that declined, and the stage says so.
  EXPECT_TRUE(f.declare_resources("/nowhere").empty());
  EXPECT_TRUE(f.declare_holdings("/nowhere").empty());
  EXPECT_TRUE(f.latent_bytes("/nowhere", 1024, 1024) == 0);
}

// A REGISTERED FAMILY IS OFFERED BY THE STAGE THAT RUNS IT.
//
// generate-image's own `hf_dir` picker is a channel CONSUMER: it lists
// the types it was compiled to run, so a plugin family it can now run
// was offered only through model-select (the channel source). The
// registry records the family against the stage it registers into, and
// the stage's config form appends it -- once, by whole entry.
TEST(image_model_registry, a_registered_family_is_offered_by_generate_image)
{
  const std::string t = tag_("picker");
  EXPECT_TRUE(ImageModelRegistry::get().add(
      std::make_unique<Fake>(t, "/models/picker")));
  Session sess;
  GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                           FlexData::make_object());
  std::string types;
  for (const auto& p : stage.config_params()) {
    if (p.key == "hf_dir") { types = p.suggest_db_type; }
  }
  const std::string padded = "," + types + ",";
  EXPECT_TRUE(padded.find("," + t + ",") != std::string::npos);
  // Once, however many times the form is built.
  std::size_t n = 0, at = 0;
  while ((at = padded.find("," + t + ",", at)) != std::string::npos) {
    ++n;
    ++at;
  }
  EXPECT_TRUE(n == 1);
  // The built-in list is still there, in front.
  EXPECT_TRUE(types.find("krea2") != std::string::npos);
}

// A family is registered under ONE tag while its checkpoints are filed
// under one catalogue type per instantiation -- a t2i and an edit repo of
// one architecture -- and the pickers filter on the type. The detect
// profile a plugin registers names those types, and a conditioning
// profile says diffusion-conditioner runs the family; between them, every
// stage that runs it offers BOTH types, with no host list naming them.
TEST(image_model_registry, a_profile_puts_every_catalogue_type_on_the_pickers)
{
  const std::string t = tag_("aliases");
  const std::string edit = t + "-edit";
  EXPECT_TRUE(ImageModelRegistry::get().add(
      std::make_unique<Fake>(t, "/models/aliases")));
  Session sess;
  VpipePluginContext ctx(&sess, "ut-aliases");
  FlexData det = FlexData::make_object();
  det.as_object().insert_or_assign(genai::detect::kModelType,
                                   FlexData::make_string(t));
  det.as_object().insert_or_assign(genai::detect::kModelTypeEdit,
                                   FlexData::make_string(edit));
  EXPECT_TRUE(ctx.register_family_profile(
      std::string(genai::detect::kDomain), t, det));
  EXPECT_TRUE(ctx.register_family_profile(
      std::string(genai::cond::kDomain), t, FlexData::make_object()));

  auto offered = [&](const Stage& st) {
    for (const auto& p : st.config_params()) {
      if (p.key == "hf_dir") { return "," + p.suggest_db_type + ","; }
    }
    return std::string();
  };
  GenerateImageStage gen(&sess, "t2i", std::vector<InEdge>{},
                         FlexData::make_object());
  DiffusionConditionerStage cond(&sess, "cond", std::vector<InEdge>{},
                                 FlexData::make_object());
  for (const std::string& types : {offered(gen), offered(cond)}) {
    EXPECT_TRUE(types.find("," + t + ",") != std::string::npos);
    EXPECT_TRUE(types.find("," + edit + ",") != std::string::npos);
  }
}
