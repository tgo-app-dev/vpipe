// What a VDN-H3 checkpoint says about itself, read from the REAL files.
//
// The values name ALGORITHMS, and each has a plausible sibling that runs
// to completion and produces a normal-looking video -- `sana_scaled` for
// `vdn_solve`, `none` for the alpha bridge, a different anchor mode. So
// the property under test is as much what the parser REFUSES as what it
// accepts: an unrecognised value must name itself, never fall back.

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/minimax-h3/vdn-config.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace vpipe;
using namespace vpipe::genai::minimax_h3;

namespace {

std::string
stage_(const char* name)
{
  if (const char* e = std::getenv("VPIPE_VDN_MODEL_PATH")) {
    return std::string(e) + "/" + name;
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) { return ""; }
  return std::string(home) + "/dock/dump/vpipe-test/models/OpenVDN/"
         + "vdn-minimax-h3/" + name;
}

vdn::Config
from_json_(const std::string& text, bool* ok, std::string* err)
{
  vdn::Config c;
  std::istringstream in(text);
  FlexData d = FlexData::from_json(in);
  *ok = vdn::parse_config(d, &c, err);
  return c;
}

const char* kGood =
    R"({"type":"hybrid_attention","version":2,"config":{
        "anchor_frames":"both","enable_softmax_gate":true,
        "linear_attention":{"a_fp32":true,"bridge":"alpha",
          "delta_rule":"vdn_solve","enable_text_state":true,
          "linear_head_dim":128,"short_conv":{"targets":["k","v"]}},
        "softmax_attention":{"chunk":5,"radius":1}}})";

}  // namespace

TEST(vdn_config, reads_the_released_stages)
{
  const char* stages[] = {"stage-b-step-2000", "stage-dmd-step-250"};
  int seen = 0;
  for (const char* name : stages) {
    const std::string dir = stage_(name);
    if (dir.empty()) { continue; }
    vdn::Config c;
    std::string err;
    // The stage directory and the weights beside it are two spellings of
    // the same model; both must resolve.
    if (!vdn::load_config(dir, &c, &err)) { continue; }   // not downloaded
    ++seen;
    vdn::Config c2;
    EXPECT_TRUE(vdn::load_config(dir + "/linear_branch", &c2, &err));

    EXPECT_TRUE(c.anchors == vdn::AnchorFrames::kBoth);
    EXPECT_TRUE(c.delta_rule == "vdn_solve");
    EXPECT_TRUE(c.bridge_alpha);
    EXPECT_TRUE(c.enable_text_state);
    EXPECT_TRUE(c.enable_softmax_gate);
    EXPECT_TRUE(c.a_fp32);
    EXPECT_TRUE(c.linear_head_dim == 128);
    EXPECT_TRUE(c.chunk == 5 && c.radius == 1);
    // K and V are convolved and Q is NOT: a conv on Q is a plausible
    // blur that no shape check would catch.
    EXPECT_TRUE(c.conv_k && c.conv_v && !c.conv_q);
    EXPECT_TRUE(c.supported(&err));
    EXPECT_TRUE(c2.anchors == c.anchors && c2.chunk == c.chunk
                && c2.delta_rule == c.delta_rule);

    // The adapters. stage-b ships `default`; stage-dmd adds `turbo`,
    // which varies rank and alpha PER MODULE.
    const std::vector<std::string> names = vdn::list_adapters(dir);
    EXPECT_TRUE(!names.empty());
    bool have_default = false, have_turbo = false;
    for (const std::string& an : names) {
      vdn::Adapter a;
      ASSERT_TRUE(vdn::load_adapter(dir + "/adapters/" + an, &a, &err));
      EXPECT_TRUE(a.rank > 0 && a.alpha > 0.0);
      EXPECT_TRUE(!a.targets.empty());
      if (an == "default") {
        have_default = true;
        // The DiT blocks name attn.orig.* -- HybridAttention keeps the
        // original module as `.orig` -- while the token refiner, which
        // is NOT wrapped, keeps the plain spelling. Both appear.
        bool orig = false, plain = false;
        for (const std::string& t : a.targets) {
          if (t.find("attn.orig.to_q") != std::string::npos) { orig = true; }
          if (t.find("token_refiner") != std::string::npos
              && t.find("attn.to_q") != std::string::npos) { plain = true; }
        }
        EXPECT_TRUE(orig);
        EXPECT_TRUE(plain);
      }
      if (an == "turbo") {
        have_turbo = true;
        EXPECT_TRUE(a.exact_targets);
        EXPECT_TRUE(!a.rank_pattern.empty() || !a.alpha_pattern.empty());
        // A module in the pattern takes its own value; one outside it
        // takes the top level. Getting that backwards scales a whole
        // adapter wrong and still renders.
        const std::string patterned = "transformer_blocks.0.adaln_proj.linear";
        if (a.alpha_pattern.count(patterned)) {
          EXPECT_TRUE(a.alpha_for(patterned) != a.alpha);
        }
        EXPECT_TRUE(a.alpha_for("nothing.named.this") == a.alpha);
        EXPECT_TRUE(a.rank_for("nothing.named.this") == a.rank);
        EXPECT_TRUE(a.scale_for("nothing.named.this")
                    == a.alpha / (double)a.rank);
      }
    }
    if (std::string(name) == "stage-b-step-2000") {
      EXPECT_TRUE(have_default && !have_turbo);
    } else {
      EXPECT_TRUE(have_default && have_turbo);
    }
  }
  // Not a failure when the 9.6 GB is not on this box, but say nothing
  // false either: the assertions above only ran if `seen` moved.
  EXPECT_TRUE(seen == 0 || seen == 2);
}

TEST(vdn_config, refuses_rather_than_defaults)
{
  // Each of these is a value the reference supports and this port does
  // not, or a shape it has never seen. Every one of them would run.
  struct Case {
    const char* what;
    const char* json;
  };
  const Case bad[] = {
      // NOT `sana_scaled`: that is a rule the FORMAT defines and this
      // port does not implement, which is supported()'s question, not
      // the parser's -- see the vdn_scaled case at the end. A name the
      // format has never had is the parser's.
      {"unknown delta rule",
       R"({"type":"hybrid_attention","version":2,"config":{
          "anchor_frames":"both","linear_attention":{"bridge":"alpha",
          "delta_rule":"gated_delta","linear_head_dim":128},
          "softmax_attention":{"chunk":5,"radius":1}}})"},
      {"unknown bridge",
       R"({"type":"hybrid_attention","version":2,"config":{
          "anchor_frames":"both","linear_attention":{"bridge":"decay",
          "delta_rule":"vdn_solve","linear_head_dim":128},
          "softmax_attention":{"chunk":5,"radius":1}}})"},
      {"unknown anchor mode",
       R"({"type":"hybrid_attention","version":2,"config":{
          "anchor_frames":"corners","linear_attention":{"bridge":"alpha",
          "delta_rule":"vdn_solve","linear_head_dim":128},
          "softmax_attention":{"chunk":5,"radius":1}}})"},
      {"unknown short_conv target",
       R"({"type":"hybrid_attention","version":2,"config":{
          "anchor_frames":"both","linear_attention":{"bridge":"alpha",
          "delta_rule":"vdn_solve","linear_head_dim":128,
          "short_conv":{"targets":["k","w"]}},
          "softmax_attention":{"chunk":5,"radius":1}}})"},
      {"a future transform version",
       R"({"type":"hybrid_attention","version":3,"config":{
          "anchor_frames":"both","linear_attention":{"bridge":"alpha",
          "delta_rule":"vdn_solve","linear_head_dim":128},
          "softmax_attention":{"chunk":5,"radius":1}}})"},
      {"a different transform",
       R"({"type":"quantization","version":2,"config":{}})"},
  };
  for (const Case& c : bad) {
    bool ok = true;
    std::string err;
    (void)from_json_(c.json, &ok, &err);
    EXPECT_FALSE(ok);
    // The message has to NAME the thing, or a user is left diffing JSON.
    EXPECT_TRUE(!err.empty());
  }

  bool ok = false;
  std::string err;
  const vdn::Config good = from_json_(kGood, &ok, &err);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(good.delta_rule == "vdn_solve" && good.bridge_alpha);
  EXPECT_TRUE(good.supported(&err));

  // Parsed cleanly, but the rule is one this port does not implement --
  // a separate question from whether the file was understood, and the
  // reason `supported()` is not folded into the parse.
  std::istringstream in(
      R"({"type":"hybrid_attention","version":2,"config":{
         "anchor_frames":"none","linear_attention":{"bridge":"none",
         "delta_rule":"vdn_scaled","linear_head_dim":128},
         "softmax_attention":{"chunk":0,"radius":2}}})");
  FlexData d = FlexData::from_json(in);
  vdn::Config other;
  EXPECT_TRUE(vdn::parse_config(d, &other, &err));
  EXPECT_TRUE(other.anchors == vdn::AnchorFrames::kNone);
  EXPECT_FALSE(other.bridge_alpha);
  EXPECT_FALSE(other.supported(&err));
  EXPECT_TRUE(err.find("vdn_scaled") != std::string::npos);
}

TEST(vdn_config, a_repo_root_says_which_stage_is_one_level_down)
{
  // THE ERROR THE REPO ROOT PRODUCES, which is the one a user actually
  // meets: OpenVDN publishes both stages from one repo, so a checkout
  // has the stages one level down and the root carries no config. Saying
  // only "nothing here" leaves the next step to guess; naming the stage
  // directories that ARE here is the answer.
  namespace fs = std::filesystem;
  auto base = fs::temp_directory_path() / "vpipe-vdn-root-XXXXXX";
  std::string tmpl = base.string();
  if (::mkdtemp(tmpl.data()) == nullptr) { return; }
  const fs::path root(tmpl);
  std::error_code ec;
  fs::create_directories(root / "stage-dmd-step-250" / "linear_branch", ec);
  { std::ofstream o((root / "stage-dmd-step-250" / "linear_branch"
                     / "config.json").string()); o << "{}"; }

  vdn::Config c;
  std::string err;
  EXPECT_FALSE(vdn::load_config(root.string(), &c, &err));
  std::printf("[vdn_config] %s\n", err.c_str());
  EXPECT_TRUE(err.find("stage-dmd-step-250") != std::string::npos);
  EXPECT_TRUE(err.find("one level down") != std::string::npos);

  // And the stage directory itself must NOT produce that error -- it is
  // the thing being pointed at.
  std::string err2;
  vdn::Config c2;
  (void)vdn::load_config((root / "stage-dmd-step-250").string(), &c2, &err2);
  EXPECT_TRUE(err2.find("one level down") == std::string::npos);
  fs::remove_all(root, ec);
}
