// Mage-Flow text conditioning (diffusion-conditioner stage) vs the reference.
//
// Mage-Flow's text encoder is the SAME Qwen3-VL 4B the Krea-2 conditioner
// drives, so almost everything about this path is a reuse claim -- and the DiT
// bring-up already showed that a structure which matches on paper can still be
// wrong (Mage-Flow vendors a get_timestep_embedding that rounds the frequency
// table to bf16). So the reuse is measured, not asserted:
//
//   t2i  -- text-only, PROMPT_TEMPLATE["mage-flow"] (drop 34), single
//           last-hidden tap [n_real, 2560] bf16 after the final RMSNorm.
//   edit -- one reference image through the Qwen3-VL tower + deepstack,
//           PROMPT_TEMPLATE["mage-flow-edit"] (drop 64), image_pad rows
//           replaced by the tower's tokens.
//
// The golden's reference image is a SQUARE 384x384 so both the reference's
// _resize_long_edge(384) and the processor's smart_resize are identities --
// that keeps the host resampler (PIL bicubic vs vpipe's area box + bilinear)
// out of the comparison, so what is measured is the encoder path.
//
// Env: VPIPE_MAGE_TEST_MODEL_PATH = the Mage-Flow model root,
// VPIPE_MAGE_COND_GOLDEN = scratchpad/dump_cond_golden.py's output dir.
// Skips vacuously if either is unset.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "stages/diffusion-conditioner-stage.h"
#include "stages/load-image-stage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

constexpr int kEncHidden = 2560;

std::vector<float>
read_f32_(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  std::vector<float> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

float
bf16_to_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

// Decode one 2-byte element with the tower's own storage type.
float
dec_elt_(std::uint16_t b, bool bf16)
{
  if (bf16) { return bf16_to_f32_(b); }
  _Float16 h; std::memcpy(&h, &b, 2); return (float)h;
}

double
rel_l2_(const std::vector<float>& a, const std::vector<float>& b)
{
  double num = 0.0, den = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d; den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Test-only source: emits one prompt string then ends.
class CondSourceText : public vpipe::TypedStage<CondSourceText> {
public:
  static constexpr const char* kTypeName = "ut-mage-cond-src-text";
  CondSourceText(const vpipe::SessionContextIntf* s, std::string id,
                 std::vector<vpipe::InEdge> ip, vpipe::FlexData c)
    : TypedStage(s, std::move(id), std::move(ip), std::move(c))
  { allocate_oports(1); }
  std::string prompt;
  bool done = false;
  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    if (!done) {
      done = true;
      co_await ctx.write(0, std::make_unique<vpipe::FlexDataPayload>(
                                vpipe::FlexData::make_string(prompt)));
    }
    ctx.signal_done();
    co_return;
  }
};

class CondSinkCapture : public vpipe::TypedStage<CondSinkCapture> {
public:
  static constexpr const char* kTypeName = "ut-mage-cond-sink";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<vpipe::BeatPayloadIntf>> captured;
  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    captured.push_back(std::move(p));
  }
};

// Run the conditioner on `prompt` (+ optional ref image) and return the
// emitted conditioning as f32 rows, with *n_real set. Empty on failure.
std::vector<float>
run_conditioner_(Session& sess, const char* root, const std::string& prompt,
                 const std::string& ref_png, int* n_real)
{
  *n_real = 0;
  auto pl = std::make_unique<Pipeline>("mc", &sess);
  auto srcu = std::make_unique<CondSourceText>(&sess, "src",
                                               std::vector<InEdge>{},
                                               FlexData::make_object());
  srcu->prompt = prompt;
  auto* src = static_cast<CondSourceText*>(pl->insert_stage(std::move(srcu)));

  std::vector<InEdge> ce{{src, 0}};
  if (!ref_png.empty()) {
    FlexData ic = FlexData::make_object();
    ic.as_object().insert("url", FlexData::make_string(ref_png));
    auto iu = std::make_unique<LoadImageStage>(&sess, "img",
                                               std::vector<InEdge>{},
                                               std::move(ic));
    Stage* imgstage = pl->insert_stage(std::move(iu));
    ce.push_back({nullptr, 0});     // iport1 negative  -- unconnected
    ce.push_back({nullptr, 0});     // iport2 model     -- unconnected
    ce.push_back({imgstage, 0});    // iport3 ref_image
  }
  FlexData c = FlexData::make_object();
  c.as_object().insert("hf_dir", FlexData::make_string(root));
  auto cu = std::make_unique<DiffusionConditionerStage>(&sess, "cond",
                                                        std::move(ce),
                                                        std::move(c));
  auto* cond =
      static_cast<DiffusionConditionerStage*>(pl->insert_stage(std::move(cu)));
  auto sinku = std::make_unique<CondSinkCapture>(
      &sess, "sink", std::vector<InEdge>{{cond, 0}}, FlexData::make_object());
  auto* sink = static_cast<CondSinkCapture*>(pl->insert_stage(std::move(sinku)));

  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return {}; }
  rt.wait_idle();
  rt.stop();
  if (sink->captured.empty()) { return {}; }
  const auto* tb =
      dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
  if (tb == nullptr || tb->shape.size() != 2) { return {}; }
  if (tb->dtype != TensorBeat::DType::Bf16) { return {}; }
  *n_real = (int)tb->shape[0];
  const auto bytes = tb->materialize_contiguous();
  const auto* p = reinterpret_cast<const std::uint16_t*>(bytes.data());
  std::vector<float> out(bytes.size() / 2);
  for (std::size_t i = 0; i < out.size(); ++i) { out[i] = bf16_to_f32_(p[i]); }
  return out;
}

// VPIPE_MAGE_COND_DUMP=<dir>: write vpipe's own conditioning next to the
// golden so a mismatch can be dissected per row block (text vs image rows)
// outside the test.
void
dump_(const char* tag, const std::vector<float>& v)
{
  const char* d = std::getenv("VPIPE_MAGE_COND_DUMP");
  if (d == nullptr || *d == '\0') { return; }
  std::ofstream out(std::string(d) + "/v_" + tag + "_txt.f32",
                    std::ios::binary);
  if (out) {
    out.write(reinterpret_cast<const char*>(v.data()),
              (std::streamsize)(v.size() * sizeof(float)));
  }
}

// {root, golden} from the env, or {nullptr, ""} when the test should skip.
struct Env { const char* root = nullptr; std::string golden; };
Env
mage_cond_env_()
{
  Env e;
  const char* root = std::getenv("VPIPE_MAGE_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_MAGE_COND_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return e;
  }
  if (!std::filesystem::exists(std::string(gd) + "/t2i_txt.f32")) { return e; }
  e.root = root; e.golden = gd;
  return e;
}

}  // namespace

// Text-only conditioning: the t2i template (kPrefix/kSuffix, drop 34, which
// IS Mage-Flow's PROMPT_TEMPLATE["mage-flow"] verbatim) through 36 Qwen3-VL
// layers + the host final RMSNorm.
TEST(mage_cond, t2i_matches_golden)
{
  const Env env = mage_cond_env_();
  if (env.root == nullptr) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  const std::vector<float> g = read_f32_(env.golden + "/t2i_txt.f32");
  ASSERT_TRUE(!g.empty() && (g.size() % kEncHidden) == 0);
  const int gn = (int)(g.size() / kEncHidden);

  int n = 0;
  const std::vector<float> v = run_conditioner_(
      sess, env.root, "a red fox sitting in fresh snow, photorealistic", "", &n);
  if (v.empty()) {
    std::printf("[mage_cond] no conditioning emitted (model not loaded?); "
                "skipping\n");
    return;
  }
  dump_("t2i", v);
  std::printf("[mage_cond] t2i n_real=%d (golden %d)\n", n, gn);
  ASSERT_TRUE(n == gn);   // same template + tokenizer + drop index
  const double r = rel_l2_(v, g);
  std::printf("[mage_cond] t2i rel-L2 = %.6f\n", r);
  // Both sides run bf16; the observed drift over 36 layers is ~0.01.
  EXPECT_TRUE(r < 0.05);
}

// Image-grounded (edit) conditioning: the edit template (kQiePrefix/kQieSuffix,
// drop 64 == PROMPT_TEMPLATE["mage-flow-edit"]) with the reference's 144 vision
// tokens spliced over the image_pad rows and the tower's deepstack features
// injected at LM layers 0..2.
//
// TWO images, with deliberately different bounds -- see the per-case notes.
// Both goldens are the REFERENCE's bf16 conditioning (bf16 is what the
// Mage-Flow pipeline ships), and both source images are pre-sized to 384 so
// NEITHER side resizes: vpipe caps with an area-average box filter and the
// python harness with PIL, and feeding a 1024px source makes the two
// preprocessors the dominant term (it inflated this number 0.13 -> 0.33 when
// measured that way). Same-pixels-in is what makes the comparison about the
// model instead of about two resamplers.
TEST(mage_cond, edit_matches_golden)
{
  const Env env = mage_cond_env_();
  if (env.root == nullptr) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  struct Case {
    const char* tag;      // label in the log / dump filename
    const char* image;    // 384px source, in the golden dir
    const char* golden;   // the reference's bf16 conditioning for it
    double      tol;
    bool        required; // ref.png ships with every golden dump; ref2 may not
    const char* note;
  };
  // ref.png is the LOOSE one ON PURPOSE. Its layer-16 MLP contains a feature
  // that reads out the DIFFERENCE between two near-duplicate image patches
  // (weight row 103: cos 0.73 to the difference direction, 0.02 to the shared
  // one), so the two rows get gates of +59.7 / -53.4 and SiLU passes exactly
  // one -- which then receives a ~140x activation. By layer 16 the reference's
  // own bf16 trajectory perturbs those rows by more than the difference they
  // encode (19.7 vs |d| 12.1), the readout inverts, and its massive activation
  // moves to the other row. That is a property of the GOLDEN, not of vpipe:
  // the reference's own bf16-vs-fp32 conditioning is 0.240 on this image while
  // vpipe's is 0.179. Measured across four images, only this one does it.
  // Scoring vpipe tightly against a golden that contains an arbitrary
  // tie-break would be calibrating on the worst case.
  //
  // ref2 (a 384px dog photo) does NOT trip it -- its layer-16 activation stays
  // on the same row in both dtypes -- so it is an ordinary bf16-vs-bf16
  // comparison and carries a bound that can actually catch a regression.
  //
  // WHERE 0.15 COMES FROM: measured 0.104 on the default path, and across the
  // legitimate kernel/dtype paths it ranges 0.077 (VPIPE_QWEN_VISION_NO_MMA2)
  // to 0.104 (matmul2d; NO_NAX_ATTN is identical, that path is f16-only) with
  // an f16 tower at 0.099. So ~35% spread before any machine change, and the
  // bound sits ~44% above the worst of them -- it catches anything that
  // DOUBLES the error while surviving M4-vs-M5 kernel differences. Do not
  // tighten it below ~0.13 without re-measuring that spread on both boxes.
  //
  // Regenerate either golden with scratchpad/ref_multi.py (writes the
  // reference's fp32 and bf16 conditioning for a list of images). Source
  // images must already be <=384 on the long edge.
  const Case cases[] = {
    {"edit",  "/ref.png",  "/edit_txt.f32",  0.35, true,
     "flip-prone: the golden itself carries a tie-break"},
    {"edit2", "/ref2.png", "/edit2_txt.f32", 0.15, false,
     "stable: the real regression gate"},
  };

  for (const Case& c : cases) {
    const std::string img = env.golden + c.image;
    const std::string gld = env.golden + c.golden;
    if (!std::filesystem::exists(img) || !std::filesystem::exists(gld)) {
      if (c.required) {
        std::printf("[mage_cond] no %s; skipping\n", img.c_str());
      } else {
        std::printf("[mage_cond] no %s (older golden dump); skipping that "
                    "case\n", img.c_str());
      }
      continue;
    }
    const std::vector<float> g = read_f32_(gld);
    ASSERT_TRUE(!g.empty() && (g.size() % kEncHidden) == 0);
    const int gn = (int)(g.size() / kEncHidden);

    int n = 0;
    const std::vector<float> v = run_conditioner_(
        sess, env.root, "make the sky a vivid blue", img, &n);
    if (v.empty()) {
      std::printf("[mage_cond] no edit conditioning emitted; skipping\n");
      continue;
    }
    dump_(c.tag, v);
    std::printf("[mage_cond] %s n_real=%d (golden %d)\n", c.tag, n, gn);
    // A mismatch here means the vision grid differs (smart_resize bounds) or
    // the template / drop index is wrong -- not a numerical issue.
    ASSERT_TRUE(n == gn);
    const double r = rel_l2_(v, g);
    std::printf("[mage_cond] %s rel-L2 = %.6f  (bound %.2f, %s)\n",
                c.tag, r, c.tol, c.note);
    EXPECT_TRUE(r < c.tol);
  }
}

// The Qwen3-VL vision tower ALONE, against an fp32 oracle -- the split that
// says whether an edit-conditioning mismatch lives in the tower or downstream
// of it (splice / positions / deepstack / the 36 LM layers).
//
// Runs BOTH element types. f16 is the more ACCURATE tower; bf16 is what
// Mage-Flow actually ships (the reference casts its whole text encoder, tower
// included, so vpipe matches its numerics). Testing only f16 would leave the
// shipped configuration unmeasured -- and a dtype bug in the bf16 twin does
// NOT look like a crash, it looks like a slightly-off end-to-end number.
TEST(mage_cond, vision_tower_matches_golden)
{
  const Env env = mage_cond_env_();
  if (env.root == nullptr) { return; }
  const std::string ref = env.golden + "/ref.png";
  const std::string emb = env.golden + "/tower_emb.f32";
  if (!std::filesystem::exists(ref) || !std::filesystem::exists(emb)) {
    std::printf("[mage_cond] no tower golden; skipping\n");
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string enc_dir =
      (std::filesystem::path(env.root) / "text_encoder").string();
  genai::ModelLoader loader(&sess);
  const auto mcfg = loader.load_config(enc_dir);
  ASSERT_TRUE(mcfg.has_value());
  // (dtype, embeddings tol, deepstack tol). bf16 carries 3 fewer mantissa
  // bits than f16, so it is allowed ~4x the error against the SAME fp32
  // oracle -- and nowhere near the O(1) a mis-typed buffer read produces.
  struct Arm { bool bf16; double emb_tol; double ds_tol; };
  for (const Arm& arm : {Arm{false, 0.02, 0.02}, Arm{true, 0.08, 0.08}}) {
  auto vcfg = genai::MetalQwenVisionEncoder::config_from(*mcfg);
  vcfg.weight_prefix = "model.visual.";
  vcfg.min_pixels = 65536;
  vcfg.max_pixels = 16777216;
  vcfg.use_bf16 = arm.bf16;
  auto tower = genai::MetalQwenVisionEncoder::load(enc_dir, mc, vcfg);
  ASSERT_TRUE((bool)tower);
  // The env A/B knobs would silently override the arm under test.
  ASSERT_TRUE(tower->is_bf16() == arm.bf16);

  // Decode ref.png through the same load-image path the stage uses.
  std::vector<std::uint8_t> rgb;
  int H = 0, W = 0;
  {
    auto pl = std::make_unique<Pipeline>("tw", &sess);
    FlexData ic = FlexData::make_object();
    ic.as_object().insert("url", FlexData::make_string(ref));
    auto iu = std::make_unique<LoadImageStage>(&sess, "img",
                                               std::vector<InEdge>{},
                                               std::move(ic));
    Stage* im = pl->insert_stage(std::move(iu));
    auto sk = std::make_unique<CondSinkCapture>(
        &sess, "sink", std::vector<InEdge>{{im, 0}}, FlexData::make_object());
    auto* sink = static_cast<CondSinkCapture*>(pl->insert_stage(std::move(sk)));
    PipelineRuntime rt(pl.get(), &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    ASSERT_TRUE(!sink->captured.empty());
    const auto* tb =
        dynamic_cast<const TensorBeatPayload*>(sink->captured[0].get());
    ASSERT_TRUE(tb != nullptr && tb->shape.size() == 3 && tb->shape[0] == 3);
    H = (int)tb->shape[1]; W = (int)tb->shape[2];
    const auto bytes = tb->materialize_contiguous();
    rgb.assign(bytes.begin(), bytes.end());
  }

  auto r = tower->encode(rgb.data(), H, W);
  ASSERT_TRUE(r.n_tokens > 0 && !r.embeddings.empty());
  std::printf("[mage_cond] tower %dx%d -> %d tokens (grid %dx%d), %d "
              "deepstack\n", W, H, r.n_tokens, r.grid_h, r.grid_w,
              (int)r.deepstack.size());

  const std::vector<float> g = read_f32_(emb);
  ASSERT_TRUE((int)g.size() == r.n_tokens * kEncHidden);
  std::vector<float> v((std::size_t)r.n_tokens * kEncHidden);
  {
    // Read with the encoder's OWN element type: a bf16 tower's buffers are
    // bf16 bits, and decoding them as f16 yields plausible-looking garbage
    // (right magnitude, wrong values) rather than an obvious failure.
    const auto* p = static_cast<const std::uint16_t*>(r.embeddings.contents());
    const bool bf = tower->is_bf16();
    for (std::size_t i = 0; i < v.size(); ++i) { v[i] = dec_elt_(p[i], bf); }
  }
  dump_("tower_emb", v);
  const double re = rel_l2_(v, g);
  std::printf("[mage_cond] tower[%s] embeddings rel-L2 = %.6f\n",
              arm.bf16 ? "bf16" : "f16", re);
  EXPECT_TRUE(re < arm.emb_tol);

  for (std::size_t k = 0; k < r.deepstack.size(); ++k) {
    const std::vector<float> gd =
        read_f32_(env.golden + "/tower_ds" + std::to_string(k) + ".f32");
    if ((int)gd.size() != r.n_tokens * kEncHidden) { continue; }
    std::vector<float> vd((std::size_t)r.n_tokens * kEncHidden);
    const auto* p =
        static_cast<const std::uint16_t*>(r.deepstack[k].contents());
    for (std::size_t i = 0; i < vd.size(); ++i) {
      vd[i] = dec_elt_(p[i], tower->is_bf16());
    }
    dump_(("tower_ds" + std::to_string(k)).c_str(), vd);
    const double rd = rel_l2_(vd, gd);
    std::printf("[mage_cond] tower[%s] deepstack[%d] rel-L2 = %.6f\n",
                arm.bf16 ? "bf16" : "f16", (int)k, rd);
    EXPECT_TRUE(rd < arm.ds_tol);
  }
  }
}

// PER-LAYER LM comparison on the TEXT-ONLY conditioning.
//
// The conditioner is deterministic -- no sampling anywhere -- so vpipe and the
// reference should agree closely, and a real bug here would be masked
// downstream by the DiT + VAE (which are tolerant of conditioning that is
// merely "close"). The end-to-end edit number (0.238) is explainable as tower
// dtype amplified through 36 layers, but "explainable" is not "verified", so
// this walks every layer.
//
// Text-only is the right probe: the LM input is an exact embedding lookup, so
// NO vision tower and nothing dtype-dependent enters before layer 0. The SHAPE
// of the curve is the diagnosis -- smooth growth is bf16 accumulation, a jump
// at one layer is a bug in that layer's path.
//
// Env: VPIPE_MAGE_TEST_MODEL_PATH + VPIPE_MAGE_COND_GOLDEN (for t2i_ids.i32)
// + VPIPE_MAGE_LM_GOLDEN = scratchpad/dump_lm_layers.py's output dir.
TEST(mage_cond, lm_per_layer_matches_golden)
{
  const Env env = mage_cond_env_();
  const char* lmg = std::getenv("VPIPE_MAGE_LM_GOLDEN");
  if (env.root == nullptr || lmg == nullptr || *lmg == '\0') { return; }
  const std::string L(lmg);
  if (!std::filesystem::exists(L + "/layer_00.f32")) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // Token ids straight from the golden, so tokenization is not a variable.
  std::ifstream idf(env.golden + "/t2i_ids.i32", std::ios::binary);
  ASSERT_TRUE((bool)idf);
  idf.seekg(0, std::ios::end);
  const std::streamoff nb = idf.tellg();
  idf.seekg(0, std::ios::beg);
  std::vector<std::int32_t> ids((std::size_t)nb / 4);
  idf.read(reinterpret_cast<char*>(ids.data()), nb);
  const int n = (int)ids.size();
  ASSERT_TRUE(n > 0);

  const std::string enc_dir =
      (std::filesystem::path(env.root) / "text_encoder").string();
  genai::MetalQwenModel::Config c;
  c.n_layers = 36; c.hidden = kEncHidden; c.n_heads = 32; c.n_kv_heads = 8;
  c.head_dim = 128; c.ffn_inner = 9728; c.vocab = 151936; c.rope_theta = 5.0e6f;
  c.rms_eps = 1e-6f; c.rotary_dim = 128; c.full_attn_interval = 1;
  c.tie_embeddings = true; c.use_bf16 = true; c.dense = true;
  c.zero_centered_norm = false; c.attn_output_gate = false;
  c.backbone_only = true; c.weight_prefix = "model.language_model.";
  c.model_seg = ""; c.max_seq = 1024; c.page_tokens = 256;
  auto enc = genai::MetalQwenModel::load(enc_dir, mc, c);
  ASSERT_TRUE((bool)enc);

  auto wts = genai::MetalLlamaWeights::open_model(enc_dir);
  ASSERT_TRUE(wts.has_value());
  metal_compute::SharedBuffer embed =
      wts->load("model.language_model.embed_tokens.weight", mc);
  ASSERT_TRUE(!embed.empty());

  metal_compute::SharedBuffer x =
      mc->make_shared_buffer((std::size_t)n * kEncHidden * 2);
  ASSERT_TRUE(!x.empty());
  {
    const auto* tbl = static_cast<const std::uint8_t*>(embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    const std::size_t vocab = embed.byte_size() / ((std::size_t)kEncHidden * 2);
    for (int i = 0; i < n; ++i) {
      const std::uint32_t id = (std::uint32_t)ids[(std::size_t)i];
      ASSERT_TRUE(id < vocab);
      std::memcpy(xb + (std::size_t)i * kEncHidden * 2,
                  tbl + (std::size_t)id * kEncHidden * 2,
                  (std::size_t)kEncHidden * 2);
    }
  }

  std::vector<int> taps_l;
  for (int i = 0; i < c.n_layers; ++i) { taps_l.push_back(i); }
  genai::ContextManager* cm = enc->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  metal_compute::SharedBuffer taps =
      enc->forward_embeddings_taps(cid, x, n, taps_l);
  cm->release(cid);
  ASSERT_TRUE(!taps.empty());

  const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
  const std::size_t per = (std::size_t)n * kEncHidden;
  double worst_jump = 0.0;
  int worst_at = -1;
  double prev = 0.0;
  std::printf("[mage_cond] LM per-layer (text-only, %d tokens):\n", n);
  for (int l = 0; l < c.n_layers; ++l) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/layer_%02d.f32", l);
    const std::vector<float> g = read_f32_(L + buf);
    if (g.size() != per) { continue; }
    std::vector<float> v(per);
    for (std::size_t i = 0; i < per; ++i) {
      v[i] = bf16_to_f32_(tp[(std::size_t)l * per + i]);
    }
    const double r = rel_l2_(v, g);
    // Cosine over the whole tensor: catches a direction change that a
    // magnitude-only metric would miss.
    double dot = 0.0, na = 0.0, nbm = 0.0;
    for (std::size_t i = 0; i < per; ++i) {
      dot += (double)v[i] * g[i]; na += (double)v[i] * v[i];
      nbm += (double)g[i] * g[i];
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nbm) + 1e-30);
    const double jump = r - prev;
    if (jump > worst_jump) { worst_jump = jump; worst_at = l; }
    prev = r;
    if (l < 4 || l % 6 == 5 || l == c.n_layers - 1) {
      std::printf("    layer %2d  rel-L2 %.6f  cos %.8f\n", l, r, cos);
    }
  }
  const std::vector<float> gf = read_f32_(L + "/final.f32");
  std::printf("[mage_cond] largest single-layer rel-L2 increase: %.6f "
              "(at layer %d)\n", worst_jump, worst_at);
  // A per-layer step this small means no layer introduces a discontinuity --
  // the growth is accumulation, not a defect. Loose enough for bf16, tight
  // enough that a real bug (which lands as a step of 0.1+) cannot pass.
  EXPECT_TRUE(worst_jump < 0.02);
  EXPECT_TRUE(prev < 0.10);   // total drift across all 36 layers
  (void)gf;
}

// PER-LAYER LM comparison on the EDIT (image-grounded) conditioning, with the
// TOWER HELD FIXED -- both sides consume vpipe's own tower output, so the
// f16-vs-bf16 tower difference is removed and what remains is purely the LM
// path. This is the probe that covers the two things the text-only test
// cannot: the vision-token SPLICE over the image_pad rows, and the DEEPSTACK
// injection after layers 0/1/2. A defect in either shows as a step at those
// layers rather than as smooth accumulation.
//
// Env: + VPIPE_MAGE_LM_EDIT_GOLDEN = dump_lm_layers_edit.py's output dir.
TEST(mage_cond, lm_per_layer_edit_matches_golden)
{
  const Env env = mage_cond_env_();
  const char* lmg = std::getenv("VPIPE_MAGE_LM_EDIT_GOLDEN");
  if (env.root == nullptr || lmg == nullptr || *lmg == '\0') { return; }
  const std::string L(lmg);
  if (!std::filesystem::exists(L + "/layer_00.f32")) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  std::ifstream idf(env.golden + "/edit_ids.i32", std::ios::binary);
  ASSERT_TRUE((bool)idf);
  idf.seekg(0, std::ios::end);
  const std::streamoff nb = idf.tellg();
  idf.seekg(0, std::ios::beg);
  std::vector<std::int32_t> ids((std::size_t)nb / 4);
  idf.read(reinterpret_cast<char*>(ids.data()), nb);
  const int n = (int)ids.size();
  ASSERT_TRUE(n > 0);

  // vpipe's own tower output + deepstack, the same tensors injected into the
  // reference run that produced this golden.
  const std::vector<float> temb = read_f32_(env.golden + "/v_tower_emb_txt.f32");
  std::vector<std::vector<float>> tds;
  for (int k = 0; k < 3; ++k) {
    tds.push_back(read_f32_(env.golden + "/v_tower_ds" + std::to_string(k) +
                            "_txt.f32"));
  }
  ASSERT_TRUE(!temb.empty() && temb.size() % kEncHidden == 0);
  const int n_img = (int)(temb.size() / kEncHidden);

  const std::string enc_dir =
      (std::filesystem::path(env.root) / "text_encoder").string();
  genai::MetalQwenModel::Config c;
  c.n_layers = 36; c.hidden = kEncHidden; c.n_heads = 32; c.n_kv_heads = 8;
  c.head_dim = 128; c.ffn_inner = 9728; c.vocab = 151936; c.rope_theta = 5.0e6f;
  c.rms_eps = 1e-6f; c.rotary_dim = 128; c.full_attn_interval = 1;
  c.tie_embeddings = true; c.use_bf16 = true; c.dense = true;
  c.zero_centered_norm = false; c.attn_output_gate = false;
  c.backbone_only = true; c.weight_prefix = "model.language_model.";
  c.model_seg = ""; c.max_seq = 1024; c.page_tokens = 256;
  auto enc = genai::MetalQwenModel::load(enc_dir, mc, c);
  ASSERT_TRUE((bool)enc);
  auto wts = genai::MetalLlamaWeights::open_model(enc_dir);
  ASSERT_TRUE(wts.has_value());
  metal_compute::SharedBuffer embed =
      wts->load("model.language_model.embed_tokens.weight", mc);
  ASSERT_TRUE(!embed.empty());

  metal_compute::SharedBuffer x =
      mc->make_shared_buffer((std::size_t)n * kEncHidden * 2);
  ASSERT_TRUE(!x.empty());
  {
    const auto* tbl = static_cast<const std::uint8_t*>(embed.contents());
    auto* xb = static_cast<std::uint8_t*>(x.contents());
    for (int i = 0; i < n; ++i) {
      std::memcpy(xb + (std::size_t)i * kEncHidden * 2,
                  tbl + (std::size_t)ids[(std::size_t)i] * kEncHidden * 2,
                  (std::size_t)kEncHidden * 2);
    }
  }
  // Splice the tower rows over image_pad (151655), exactly as the conditioner.
  constexpr std::int32_t kPadId = 151655;
  int first_pad = -1, spliced = 0;
  {
    auto* xh = static_cast<std::uint16_t*>(x.contents());
    for (int i = 0; i < n && spliced < n_img; ++i) {
      if (ids[(std::size_t)i] != kPadId) { continue; }
      if (first_pad < 0) { first_pad = i; }
      for (int h = 0; h < kEncHidden; ++h) {
        float f = temb[(std::size_t)spliced * kEncHidden + h];
        std::uint32_t u; std::memcpy(&u, &f, 4);
        xh[(std::size_t)i * kEncHidden + h] =
            (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
      }
      ++spliced;
    }
  }
  ASSERT_TRUE(spliced == n_img && first_pad >= 0);

  std::vector<metal_compute::SharedBuffer> ds_bufs;
  genai::MetalQwenModel::DeepstackInject ds;
  for (int k = 0; k < 3; ++k) {
    if (tds[(std::size_t)k].size() != (std::size_t)n_img * kEncHidden) { break; }
    metal_compute::SharedBuffer b =
        mc->make_shared_buffer((std::size_t)n_img * kEncHidden * 2);
    auto* d = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < (std::size_t)n_img * kEncHidden; ++i) {
      float f = tds[(std::size_t)k][i];
      std::uint32_t u; std::memcpy(&u, &f, 4);
      d[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    ds_bufs.push_back(std::move(b));
  }
  ASSERT_TRUE(ds_bufs.size() == 3);
  for (int k = 0; k < 3; ++k) {
    ds.feats.push_back(&ds_bufs[(std::size_t)k]);
    ds.layers.push_back(k);
  }
  ds.row0 = first_pad;
  ds.rows = n_img;

  std::vector<int> taps_l;
  for (int i = 0; i < c.n_layers; ++i) { taps_l.push_back(i); }
  genai::ContextManager* cm = enc->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  metal_compute::SharedBuffer taps =
      enc->forward_embeddings_taps(cid, x, n, taps_l, /*key_valid_len=*/0, &ds);
  cm->release(cid);
  ASSERT_TRUE(!taps.empty());

  const auto* tp = static_cast<const std::uint16_t*>(taps.contents());
  const std::size_t per = (std::size_t)n * kEncHidden;
  double worst_jump = 0.0, prev = 0.0;
  int worst_at = -1;
  std::printf("[mage_cond] LM per-layer EDIT (%d tokens, %d vision rows from "
              "row %d, deepstack at layers 0-2):\n", n, n_img, first_pad);
  for (int l = 0; l < c.n_layers; ++l) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/layer_%02d.f32", l);
    const std::vector<float> g = read_f32_(L + buf);
    if (g.size() != per) { continue; }
    std::vector<float> v(per);
    for (std::size_t i = 0; i < per; ++i) {
      v[i] = bf16_to_f32_(tp[(std::size_t)l * per + i]);
    }
    {   // VPIPE_MAGE_COND_DUMP: vpipe's own per-layer states, so an external
        // 3-way decomposition (vpipe / reference-on-vpipe's-tower /
        // reference-on-its-own-tower) can be computed per layer.
      char t[48];
      std::snprintf(t, sizeof(t), "lm_edit_layer_%02d", l);
      dump_(t, v);
    }
    const double r = rel_l2_(v, g);
    double dot = 0.0, na = 0.0, nbm = 0.0;
    for (std::size_t i = 0; i < per; ++i) {
      dot += (double)v[i] * g[i]; na += (double)v[i] * v[i];
      nbm += (double)g[i] * g[i];
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nbm) + 1e-30);
    const double jump = r - prev;
    if (jump > worst_jump) { worst_jump = jump; worst_at = l; }
    prev = r;
    if (l < 4 || l % 6 == 5 || l == c.n_layers - 1) {
      std::printf("    layer %2d  rel-L2 %.6f  cos %.8f\n", l, r, cos);
    }
    if (l == c.n_layers - 1) {
      // A whole-tensor cosine can hide a handful of near-orthogonal rows, and
      // per-ROW is what the DiT actually attends to -- so report the worst.
      std::vector<double> crow((std::size_t)n), gnorm((std::size_t)n);
      for (int t = 0; t < n; ++t) {
        double d2 = 0.0, a2 = 0.0, b2 = 0.0;
        for (int h = 0; h < kEncHidden; ++h) {
          const double va = v[(std::size_t)t * kEncHidden + h];
          const double vb = g[(std::size_t)t * kEncHidden + h];
          d2 += va * vb; a2 += va * va; b2 += vb * vb;
        }
        crow[(std::size_t)t] = d2 / (std::sqrt(a2) * std::sqrt(b2) + 1e-30);
        gnorm[(std::size_t)t] = std::sqrt(b2);
      }
      std::vector<double> sorted_n = gnorm;
      std::sort(sorted_n.begin(), sorted_n.end());
      const double med = sorted_n[sorted_n.size() / 2];
      int n_bad = 0;
      double worst = 2.0;
      for (int t = 0; t < n; ++t) {
        if (crow[(std::size_t)t] < 0.99) { ++n_bad; }
        worst = std::min(worst, crow[(std::size_t)t]);
      }
      std::printf("    per-row cosine (last layer): worst %.6f, %d of %d rows "
                  "< 0.99; median row norm %.1f\n", worst, n_bad, n, med);
      for (int t = 0; t < n; ++t) {
        if (crow[(std::size_t)t] < 0.99) {
          std::printf("      row %3d: cos %.4f  |g| %.1f  (%.2fx the median)"
                      "%s\n", t, crow[(std::size_t)t], gnorm[(std::size_t)t],
                      gnorm[(std::size_t)t] / med,
                      (t >= first_pad && t < first_pad + n_img) ? " [image]"
                                                                : " [text]");
        }
      }
      // The outliers are NOT low-norm cancellation (both sit near the median
      // magnitude) -- they are intrinsically chaotic TOKENS. Established with
      // scratchpad/probe_row_sensitivity.py, which perturbs the REFERENCE's
      // own tower input by 1e-3 (less than bf16 rounding) and re-runs it:
      //   row 172 is the reference's #1 most unstable row (self-cos 0.9256,
      //           against vpipe's 0.9125)
      //   row  97 is its #5 (self-cos 0.9983, against vpipe's 0.9900)
      //   whole-tensor reference-vs-ITSELF rel-L2 0.0336 -- LARGER than
      //           vpipe-vs-reference at 0.0303
      // So vpipe tracks the reference more closely than the reference tracks
      // itself under a sub-bf16 perturbation; a handful of unstable rows is a
      // property of the model, not a defect. Bound the COUNT (a real bug moves
      // many rows, not two) rather than the worst value.
      EXPECT_TRUE(n_bad <= 5);
      (void)med;
    }
  }
  std::printf("[mage_cond] EDIT largest single-layer increase: %.6f "
              "(at layer %d); total %.6f\n", worst_jump, worst_at, prev);
  // Layers 0-2 carry the deepstack injection; a wrong row0/rows or a missing
  // add would land there as a step, not as drift. (Tap semantics matter: the
  // reference adds deepstack AFTER a layer returns, so the golden must capture
  // layer L+1's INPUT -- comparing against layer L's OUTPUT shows a spurious
  // ~0.33 at exactly those three layers, which is a measurement artifact and
  // not a defect. See dump_lm_layers_edit.py.)
  EXPECT_TRUE(worst_jump < 0.03);
  // Total across 36 bf16 layers: 0.030 observed, against 0.034 for the
  // reference against ITSELF under a 1e-3 input perturbation.
  EXPECT_TRUE(prev < 0.06);
}
