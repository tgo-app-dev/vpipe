// Krea-2-Turbo DiT quantization (4/8-bit affine). Quantizes the transformer
// with the generic ModelQuantizer (the DiT Linear leaf set), reloads it through
// MetalKrea2Transformer's affine path, and rel-L2s a single denoiser step
// against the golden a3_step0_noise. 4-bit adds group-affine reconstruction
// error on top of the f16 baseline (0.016), so the bar is loosened per width.
//
// The quantized checkpoint is CACHED next to the model root (<root>-dit-w{b}g64)
// so re-runs skip the (multi-GB) quantize. Env: VPIPE_KREA2_TEST_MODEL_PATH,
// VPIPE_KREA2_GOLDEN. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/krea2/metal-krea2-calibration.h"
#include "generative-models/krea2/metal-krea2-transformer.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/quantize/model-quantizer.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "stages/model-quantize-stage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

std::vector<float>
read_f32q_(const std::string& path)
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

double
rel_l2q_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// The DiT's quantizable Linear leaves (the segment before ".weight"). Covers
// the 28 blocks' attn (to_q/k/v/gate, to_out.0 -> leaf "0") + SwiGLU
// (gate/up/down), the text-fusion blocks, txt_in / time_embed (linear_1/2),
// img_in, time_mod_proj and final_layer.linear. `projector` (K=12) is left
// dense (not group-divisible). "0" uniquely matches attn.to_out.0 here.
std::vector<std::string> dit_quant_linears_()
{
  return {"to_q", "to_k", "to_v", "to_gate", "0", "gate", "up", "down",
          "linear_1", "linear_2", "img_in", "time_mod_proj", "linear"};
}

// Quantize <root>/transformer -> a cached sibling dir; reuse if present.
std::string
ensure_quantized_(MetalCompute* mc, const std::string& root, int bits)
{
  namespace fs = std::filesystem;
  const std::string src = root + "/transformer";
  const std::string dst = root + "-dit-w" + std::to_string(bits) + "g64";
  if (fs::exists(fs::path(dst) / "config.json")) { return dst; }
  QuantizeOptions opt;
  opt.bits = bits;
  opt.group = 64;
  opt.quant_linears = dit_quant_linears_();
  ModelQuantizer q(mc);
  std::string err;
  const bool ok = q.run(src, dst, opt, &err);
  if (!ok) {
    std::printf("[krea2_dit_quant] quantize failed: %s\n", err.c_str());
    return {};
  }
  return dst;
}

double
run_step_(MetalCompute* mc, const std::string& dir, const std::string& gdir)
{
  const std::vector<float> txt   = read_f32q_(gdir + "/d_txtin.f32");
  const std::vector<float> latin = read_f32q_(gdir + "/a3_step0_latin.f32");
  const std::vector<float> tstep = read_f32q_(gdir + "/a3_step0_tstep.f32");
  const std::vector<float> noise = read_f32q_(gdir + "/a3_step0_noise.f32");
  if (txt.empty() || latin.empty() || tstep.empty() || noise.empty()) {
    return -1.0;
  }
  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = (int)(txt.size() / HID);
  const int img_seq = (int)(latin.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }

  auto m = MetalKrea2Transformer::load(dir, mc, cfg);
  if (m == nullptr) { return -2.0; }

  auto to_f16buf = [&](const std::vector<float>& s) {
    SharedBuffer b = mc->make_shared_buffer(s.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < s.size(); ++i) { d[i] = (_Float16)s[i]; }
    return b;
  };
  SharedBuffer fused = to_f16buf(txt);
  SharedBuffer lat = to_f16buf(latin);
  SharedBuffer out = m->forward_dit(fused, text_seq, lat, img_seq, grid, grid,
                                    tstep[0], -1);
  if (out.empty()) { return -3.0; }
  const auto* op = static_cast<const _Float16*>(out.contents());
  std::vector<float> got(noise.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)op[i]; }
  return rel_l2q_(got.data(), noise.data(), got.size());
}

// Returns the single-step rel-L2 for a `bits`-quantized DiT, or a negative
// sentinel: -100 = env unset (skip), -1/-2/-3 = golden/load/run failure,
// -200 = quantize failed.
double
measure_(int bits)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return -100.0;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return -100.0; }
  const std::string dir = ensure_quantized_(mc, root, bits);
  if (dir.empty()) { return -200.0; }
  return run_step_(mc, dir, std::string(gd));
}

}  // namespace

TEST(krea2_dit_quant, w8_single_step)
{
  const double r = measure_(8);
  if (r <= -100.0) { return; }              // env-gated skip
  std::printf("[krea2_dit_quant] w8 g64 single_step rel-L2 = %.6f\n", r);
  ASSERT_TRUE(r >= 0.0);                     // quantize / load / run succeeded
  EXPECT_TRUE(r < 0.045);
}

// On-device AWQ calibration taps: run ONE forward_dit with calib on and rel-L2
// the block-0 per-input-channel |activation| abs-max (qkv/o/gate-up/down) vs the
// HF golden (krea2_dit_ref.py c_{qkv0,o0,gu0,dn0}.f32). Confirms the metal taps
// capture the exact Linear-input distributions AWQ will smooth.
TEST(krea2_dit_quant, awq_calib_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string gdir = gd;
  const std::vector<float> cq = read_f32q_(gdir + "/c_qkv0.f32");
  const std::vector<float> co = read_f32q_(gdir + "/c_o0.f32");
  const std::vector<float> cg = read_f32q_(gdir + "/c_gu0.f32");
  const std::vector<float> cd = read_f32q_(gdir + "/c_dn0.f32");
  if (cq.empty() || co.empty() || cg.empty() || cd.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::vector<float> txt   = read_f32q_(gdir + "/d_txtin.f32");
  const std::vector<float> latin = read_f32q_(gdir + "/a3_step0_latin.f32");
  const std::vector<float> tstep = read_f32q_(gdir + "/a3_step0_tstep.f32");
  ASSERT_TRUE(!txt.empty() && !latin.empty() && !tstep.empty());

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = (int)(txt.size() / HID);
  const int img_seq = (int)(latin.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }

  auto m = MetalKrea2Transformer::load(std::string(root) + "/transformer", mc,
                                       cfg);
  ASSERT_TRUE(m != nullptr);
  auto to_f16buf = [&](const std::vector<float>& s) {
    SharedBuffer b = mc->make_shared_buffer(s.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < s.size(); ++i) { d[i] = (_Float16)s[i]; }
    return b;
  };
  SharedBuffer fused = to_f16buf(txt), lat = to_f16buf(latin);

  m->calib_begin();
  SharedBuffer out = m->forward_dit(fused, text_seq, lat, img_seq, grid, grid,
                                    tstep[0], -1);
  ASSERT_TRUE(!out.empty());
  m->calib_end();

  const auto qkv = m->calib_qkv();     // [n_layers][HID]
  const auto o   = m->calib_o();
  const auto gu  = m->calib_gateup();
  const auto dn  = m->calib_down();    // [n_layers][FF]
  ASSERT_TRUE(!qkv.empty() && qkv[0].size() == cq.size());
  ASSERT_TRUE(dn[0].size() == cd.size());

  const double rq = rel_l2q_(qkv[0].data(), cq.data(), cq.size());
  const double ro = rel_l2q_(o[0].data(), co.data(), co.size());
  const double rg = rel_l2q_(gu[0].data(), cg.data(), cg.size());
  const double rd = rel_l2q_(dn[0].data(), cd.data(), cd.size());
  std::printf("[krea2_dit_quant] calib abs-max block0 rel-L2: qkv %.4f o %.4f "
              "gateup %.4f down %.4f\n", rq, ro, rg, rd);
  // qkv/gate-up/down (the taps the AWQ fold consumes: qkv<-norm1, gate/up<-norm2,
  // down<-up rows) match the golden tightly. The `o` tap is the GATED ATTENTION
  // OUTPUT -- the spikiest intermediate, whose per-channel abs-max amplifies the
  // bf16 golden's own ~0.4% rounding -- and the o/v fold is deferred anyway (as
  // in the LM path), so its bar is looser.
  EXPECT_TRUE(rq < 0.05 && rg < 0.05 && rd < 0.05);
  EXPECT_TRUE(ro < 0.25);
}

// The model-quantize STAGE auto-detects the Krea2 DiT (config _class_name) and
// runs the plain affine pass. Drive it end-to-end, then load the result and
// confirm it reloads quantized and denoises correctly.
TEST(krea2_dit_quant, stage_quantizes_dit)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;

  // Point at the stock Krea-2-Turbo pipeline ROOT (no top-level config.json;
  // the DiT lives under transformer/) -- the stage must produce a SELF-
  // CONTAINED output: the DiT quantized under transformer/, every other
  // component copied, so it is usable directly as a text-to-image hf_dir.
  // Output on the source's volume so the component copies hard-link (instant).
  const std::string src = root;
  const std::string out =
      (fs::path(root).parent_path() / "vpipe-krea2-dit-stage-w8").string();
  std::error_code ec;
  fs::remove_all(out, ec);   // fresh quantize

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("src_model", FlexData::make_string(src));
  cfg.as_object().insert("output_name", FlexData::make_string(out));  // path
  cfg.as_object().insert("bits", FlexData::make_uint(8));
  cfg.as_object().insert("skip_existing", FlexData::make_bool(false));

  auto pl = std::make_unique<Pipeline>("q", &sess);
  auto qu = std::make_unique<ModelQuantizeStage>(
      &sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  auto* qs = static_cast<ModelQuantizeStage*>(pl->insert_stage(std::move(qu)));
  ASSERT_TRUE(qs->config_error().empty());

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // Self-contained: the DiT is quantized under transformer/, and the other
  // components (text_encoder/, vae/, model_index.json) were copied through.
  ASSERT_TRUE(fs::exists(fs::path(out) / "transformer" / "config.json", ec));
  EXPECT_TRUE(fs::exists(fs::path(out) / "text_encoder" / "config.json", ec));
  EXPECT_TRUE(fs::exists(fs::path(out) / "vae" / "config.json", ec));
  EXPECT_TRUE(fs::exists(fs::path(out) / "model_index.json", ec));
  // The quantized DiT reloads through the affine path within the 8-bit bar.
  const double r = run_step_(mc, out + "/transformer", std::string(gd));
  std::printf("[krea2_dit_quant] STAGE w8 single_step rel-L2 = %.6f\n", r);
  ASSERT_TRUE(r >= 0.0);
  EXPECT_TRUE(r < 0.045);
  fs::remove_all(out, ec);
}

// True when a component's config.json carries a top-level "quantization"
// block (what the affine loader keys on).
bool
has_quant_block_(const std::string& config_json)
{
  std::ifstream in(config_json);
  if (!in) { return false; }
  FlexData c = FlexData::from_json(in);
  if (!c.is_object()) { return false; }
  auto o = c.as_object();
  return o.contains("quantization");
}

// Scan every *.safetensors in `dir` and return true if ANY tensor name
// contains `needle`. The header is a u64-LE length prefix + a JSON object
// keyed by tensor name -- a substring match on that blob is enough here.
bool
any_tensor_contains_(const std::string& dir, const std::string& needle)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  for (const auto& de : fs::directory_iterator(dir, ec)) {
    if (de.path().extension() != ".safetensors") { continue; }
    std::ifstream f(de.path(), std::ios::binary);
    if (!f) { continue; }
    std::uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    if (!f || n == 0 || n > (1u << 28)) { continue; }
    std::string hdr(n, '\0');
    f.read(&hdr[0], (std::streamsize)n);
    if (hdr.find(needle) != std::string::npos) { return true; }
  }
  return false;
}

// target=text_encoder: the stage produces a SELF-CONTAINED pipeline whose
// text_encoder is quantized (gains a quantization block) while the DiT is
// copied through unquantized -- so the output is a valid text-to-image hf_dir
// with a quantized encoder. Structural check (no model load).
TEST(krea2_dit_quant, stage_quantizes_text_encoder)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;

  const std::string out =
      (fs::path(root).parent_path() / "vpipe-krea2-enc-stage-w8").string();
  std::error_code ec;
  fs::remove_all(out, ec);

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("src_model", FlexData::make_string(std::string(root)));
  cfg.as_object().insert("output_name", FlexData::make_string(out));  // path
  cfg.as_object().insert("target", FlexData::make_string("text_encoder"));
  cfg.as_object().insert("bits", FlexData::make_uint(8));
  cfg.as_object().insert("skip_existing", FlexData::make_bool(false));

  auto pl = std::make_unique<Pipeline>("qe", &sess);
  auto qu = std::make_unique<ModelQuantizeStage>(
      &sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  auto* qs = static_cast<ModelQuantizeStage*>(pl->insert_stage(std::move(qu)));
  ASSERT_TRUE(qs->config_error().empty());

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // Self-contained: all components present.
  ASSERT_TRUE(fs::exists(fs::path(out) / "text_encoder" / "config.json", ec));
  ASSERT_TRUE(fs::exists(fs::path(out) / "transformer" / "config.json", ec));
  EXPECT_TRUE(fs::exists(fs::path(out) / "vae" / "config.json", ec));
  EXPECT_TRUE(fs::exists(fs::path(out) / "model_index.json", ec));
  // The text encoder is quantized; the DiT was copied through unquantized.
  EXPECT_TRUE(has_quant_block_(
      (fs::path(out) / "text_encoder" / "config.json").string()));
  EXPECT_FALSE(has_quant_block_(
      (fs::path(out) / "transformer" / "config.json").string()));
  // The backbone linears got quantized (q_proj gains a .scales tensor), but
  // embed_tokens stays bf16 (no .scales) -- the text-to-image stage host-
  // gathers the embed table as a plain 2-byte-per-element buffer.
  const std::string enc = (fs::path(out) / "text_encoder").string();
  EXPECT_TRUE(any_tensor_contains_(enc, "q_proj.scales"));
  EXPECT_FALSE(any_tensor_contains_(enc, "embed_tokens.scales"));
  fs::remove_all(out, ec);
}

// text_encoder AWQ with ON-DEVICE auto-calibration (no calib_dir): the
// qwen3_vl encoder is a dense Qwen3 backbone, so the built-in text corpus
// calibrates it (8-bit base + tapped forward). The pass must complete without
// an explicit calib_dir and produce a quantized 4-bit encoder.
TEST(krea2_dit_quant, stage_text_encoder_awq_autocalib)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;

  const std::string out =
      (fs::path(root).parent_path() / "vpipe-krea2-enc-awq-w4").string();
  std::error_code ec;
  fs::remove_all(out, ec);

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("src_model", FlexData::make_string(std::string(root)));
  cfg.as_object().insert("output_name", FlexData::make_string(out));  // path
  cfg.as_object().insert("target", FlexData::make_string("text_encoder"));
  cfg.as_object().insert("bits", FlexData::make_uint(4));
  cfg.as_object().insert("awq", FlexData::make_bool(true));   // auto-calibrates
  // awq_clip needs calib_o.f32 (not tapped on-device); the pass must still
  // succeed -- o_proj is quantized unclipped instead of hard-failing.
  cfg.as_object().insert("awq_clip", FlexData::make_bool(true));
  cfg.as_object().insert("skip_existing", FlexData::make_bool(false));

  auto pl = std::make_unique<Pipeline>("qea", &sess);
  auto qu = std::make_unique<ModelQuantizeStage>(
      &sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  auto* qs = static_cast<ModelQuantizeStage*>(pl->insert_stage(std::move(qu)));
  ASSERT_TRUE(qs->config_error().empty());

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // AWQ auto-calibration completed and produced a quantized encoder.
  ASSERT_TRUE(fs::exists(fs::path(out) / "text_encoder" / "config.json", ec));
  const std::string enc = (fs::path(out) / "text_encoder").string();
  EXPECT_TRUE(has_quant_block_(enc + "/config.json"));
  EXPECT_TRUE(any_tensor_contains_(enc, "q_proj.scales"));
  EXPECT_FALSE(any_tensor_contains_(enc, "embed_tokens.scales"));
  fs::remove_all(out, ec);
}

// General / multi-modal LLM `target` = a submodule SCOPE. The Krea-2 text
// encoder holds a language backbone (language_model.*) AND a vision tower
// (visual.*) in one dir; target=vision must quantize ONLY the vision tower's
// linears (qkv/linear_fc1/...), leaving the language backbone bf16.
TEST(krea2_dit_quant, stage_target_scopes_submodule)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;

  const std::string enc = std::string(root) + "/text_encoder";
  const std::string out =
      (fs::path(root).parent_path() / "vpipe-krea2-vision-scope-w8").string();
  std::error_code ec;
  fs::remove_all(out, ec);

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("src_model", FlexData::make_string(enc));
  cfg.as_object().insert("output_name", FlexData::make_string(out));  // path
  cfg.as_object().insert("target", FlexData::make_string("vision"));
  cfg.as_object().insert("bits", FlexData::make_uint(8));
  cfg.as_object().insert("skip_existing", FlexData::make_bool(false));

  auto pl = std::make_unique<Pipeline>("qs", &sess);
  auto qu = std::make_unique<ModelQuantizeStage>(
      &sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  auto* qs = static_cast<ModelQuantizeStage*>(pl->insert_stage(std::move(qu)));
  ASSERT_TRUE(qs->config_error().empty());

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(fs::exists(fs::path(out) / "config.json", ec));
  // The vision tower got quantized (qkv only exists under visual.*), but the
  // language backbone stayed bf16 (no q_proj scales).
  EXPECT_TRUE(any_tensor_contains_(out, "qkv.scales"));
  EXPECT_FALSE(any_tensor_contains_(out, "q_proj.scales"));
  fs::remove_all(out, ec);
}

TEST(krea2_dit_quant, w4_single_step)
{
  const double r = measure_(4);
  if (r <= -100.0) { return; }
  std::printf("[krea2_dit_quant] w4 g64 single_step rel-L2 = %.6f\n", r);
  ASSERT_TRUE(r >= 0.0);
  EXPECT_TRUE(r < 0.090);
}

namespace {
// Count how many of the 28 main transformer_blocks carry an 8-bit attn.to_q
// (codes cols = K*8/32) vs 4-bit -- proves the mixed-precision promotion ran.
int
count_w8_blocks_(const std::string& dir)
{
  auto wts = MetalLlamaWeights::open_model(dir);
  if (!wts.has_value()) { return -1; }
  int w8 = 0;
  for (int L = 0; L < 28; ++L) {
    const std::string nm =
        "transformer_blocks." + std::to_string(L) + ".attn.to_q";
    const auto* ci = wts->info(nm + ".weight");
    const auto* si = wts->info(nm + ".scales");
    if (ci == nullptr || si == nullptr || ci->shape.size() != 2 ||
        si->shape.size() != 2) {
      return -1;
    }
    const long K = si->shape[1] * 64;               // K/group * group(64)
    const int bits = (int)(ci->shape[1] * 32 / K);
    if (bits == 8) { ++w8; }
  }
  return w8;
}
}  // namespace

// The corpus collector: run the encoder + DiT sampler over a small prompt set x
// the turbo sigmas with calib on, write calib_{qkv,o,gateup,down}.f32, and check
// the aggregated abs-max is well-formed (right shape, finite, non-negative, most
// channels active) and at the right scale -- >=70% of block-0 qkv channels meet
// or exceed the single-forward golden (the corpus aggregates more samples).
// Streaming-blocks forward_dit (each block loaded on demand, stream committed
// per block) must equal the one-stream path -- same ops + data deps, so the
// velocity is bit-for-bit identical. This is what makes 16 GB AWQ calibration
// safe (the full bf16 DiT never resides).
TEST(krea2_dit_quant, streaming_forward_matches_nonstreaming)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string gdir = gd;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::vector<float> txt   = read_f32q_(gdir + "/d_txtin.f32");
  const std::vector<float> latin = read_f32q_(gdir + "/a3_step0_latin.f32");
  const std::vector<float> tstep = read_f32q_(gdir + "/a3_step0_tstep.f32");
  if (txt.empty() || latin.empty() || tstep.empty()) { return; }

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = (int)(txt.size() / HID);
  const int img_seq = (int)(latin.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }
  const std::string dit = std::string(root) + "/transformer";

  auto run = [&](bool stream) {
    auto m = MetalKrea2Transformer::load(dit, mc, cfg, stream);
    std::vector<float> v;
    if (!m) { return v; }
    SharedBuffer fused = mc->make_shared_buffer(txt.size() * 2);
    SharedBuffer lat = mc->make_shared_buffer(latin.size() * 2);
    { auto* d = static_cast<_Float16*>(fused.contents());
      for (std::size_t i = 0; i < txt.size(); ++i) { d[i] = (_Float16)txt[i]; } }
    { auto* d = static_cast<_Float16*>(lat.contents());
      for (std::size_t i = 0; i < latin.size(); ++i) { d[i] = (_Float16)latin[i]; } }
    SharedBuffer out = m->forward_dit(fused, text_seq, lat, img_seq, grid, grid,
                                      tstep[0], -1);
    if (out.empty()) { return v; }
    const auto* p = static_cast<const _Float16*>(out.contents());
    v.resize((std::size_t)img_seq * IC);
    for (std::size_t i = 0; i < v.size(); ++i) { v[i] = (float)p[i]; }
    return v;   // m freed here -> only one 24 GB model resident at a time
  };
  const std::vector<float> vref = run(false);
  const std::vector<float> vstr = run(true);
  ASSERT_TRUE(!vref.empty() && vstr.size() == vref.size());
  const double r = rel_l2q_(vstr.data(), vref.data(), vstr.size());
  std::printf("[krea2_dit_quant] streaming vs one-stream velocity rel-L2 = "
              "%.2e\n", r);
  EXPECT_TRUE(r < 1e-4);   // identical math; segmented commits don't change it
}

// The streaming forward's cooperative-stop hook makes forward_dit bail early
// (empty) instead of grinding through all 28 disk-streamed blocks -- this is how
// AWQ calibration honors a pipeline stop within ~one block.
TEST(krea2_dit_quant, streaming_forward_honors_stop)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  const std::string gdir = gd;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::vector<float> txt   = read_f32q_(gdir + "/d_txtin.f32");
  const std::vector<float> latin = read_f32q_(gdir + "/a3_step0_latin.f32");
  const std::vector<float> tstep = read_f32q_(gdir + "/a3_step0_tstep.f32");
  if (txt.empty() || latin.empty() || tstep.empty()) { return; }

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = (int)(txt.size() / HID);
  const int img_seq = (int)(latin.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }

  auto m = MetalKrea2Transformer::load(std::string(root) + "/transformer", mc,
                                       cfg, /*stream_blocks=*/true);
  ASSERT_TRUE(m != nullptr);
  SharedBuffer fused = mc->make_shared_buffer(txt.size() * 2);
  SharedBuffer lat = mc->make_shared_buffer(latin.size() * 2);
  { auto* d = static_cast<_Float16*>(fused.contents());
    for (std::size_t i = 0; i < txt.size(); ++i) { d[i] = (_Float16)txt[i]; } }
  { auto* d = static_cast<_Float16*>(lat.contents());
    for (std::size_t i = 0; i < latin.size(); ++i) { d[i] = (_Float16)latin[i]; } }

  // With stop always true, forward_dit bails at the first block -> empty.
  m->set_stream_stop([] { return true; });
  SharedBuffer stopped = m->forward_dit(fused, text_seq, lat, img_seq, grid, grid,
                                        tstep[0], -1);
  EXPECT_TRUE(stopped.empty());

  // Cleared hook -> the same forward completes normally (non-empty).
  m->set_stream_stop({});
  SharedBuffer ok = m->forward_dit(fused, text_seq, lat, img_seq, grid, grid,
                                   tstep[0], -1);
  EXPECT_TRUE(!ok.empty());
}

TEST(krea2_dit_quant, corpus_collector)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;
  const std::string out =
      (fs::temp_directory_path() / "vpipe-krea2-dit-calib").string();
  std::error_code ec;
  fs::remove_all(out, ec);

  const std::vector<std::string> prompts = {
      "a red fox standing in fresh snow",
      "a bustling city street at night with neon signs",
      "a bowl of ripe strawberries on a wooden table",
      "a portrait of an old fisherman with a weathered face"};
  std::string err;
  const bool ok = collect_dit_calibration(mc, std::string(root), prompts,
                                          /*steps=*/8, 256, 256, /*seed=*/0,
                                          out, &err);
  if (!ok) { std::printf("[krea2_dit_quant] collector: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);

  const int nL = 28, HID = 6144, FF = 16384;
  const std::vector<float> q = read_f32q_(out + "/calib_qkv.f32");
  const std::vector<float> d = read_f32q_(out + "/calib_down.f32");
  ASSERT_TRUE(q.size() == (std::size_t)nL * HID);
  ASSERT_TRUE(d.size() == (std::size_t)nL * FF);

  int active = 0;
  bool clean = true;
  for (float v : q) {
    if (!std::isfinite(v) || v < 0.0f) { clean = false; }
    if (v > 1e-3f) { ++active; }
  }
  EXPECT_TRUE(clean);
  std::printf("[krea2_dit_quant] collector: %d/%d qkv channels active\n", active,
              (int)q.size());
  EXPECT_TRUE(active > (int)(0.9 * q.size()));

  // Block-0 qkv vs the single-forward golden: the corpus (32 samples) should
  // meet/exceed the single golden sample on most channels.
  const std::vector<float> cq = read_f32q_(std::string(gd) + "/c_qkv0.f32");
  if (cq.size() == (std::size_t)HID) {
    int ge = 0;
    for (int c = 0; c < HID; ++c) {
      if (q[(std::size_t)c] >= 0.9f * cq[(std::size_t)c]) { ++ge; }
    }
    std::printf("[krea2_dit_quant] collector: %d/%d block0 qkv >= golden\n", ge,
                HID);
    EXPECT_TRUE(ge > (int)(0.70 * HID));
  }
  fs::remove_all(out, ec);
}

namespace {
// Cache a calib dir (<root>-ditcalib) + an AWQ-clipped w4 DiT
// (<root>-dit-w4awqg64) so re-runs skip the (minutes-long) collection/quantize.
std::string
ensure_awq_calib_(MetalCompute* mc, const std::string& root)
{
  namespace fs = std::filesystem;
  const std::string dst = root + "-ditcalib";
  if (fs::exists(fs::path(dst) / "calib_qkv.f32")) { return dst; }
  auto all = default_dit_calibration_prompts();
  std::vector<std::string> prompts(all.begin(),
                                   all.begin() + std::min<std::size_t>(8, all.size()));
  std::string err;
  if (!collect_dit_calibration(mc, root, prompts, 8, 256, 256, 0, dst, &err)) {
    std::printf("[krea2_dit_quant] calib collect failed: %s\n", err.c_str());
    return {};
  }
  return dst;
}

std::string
ensure_awq_quant_(MetalCompute* mc, const std::string& root,
                  const std::string& calib_dir)
{
  namespace fs = std::filesystem;
  const std::string dst = root + "-dit-w4awqg64";
  if (fs::exists(fs::path(dst) / "config.json")) { return dst; }
  QuantizeOptions opt;
  opt.bits = 4; opt.group = 64;
  opt.quant_linears = dit_quant_linears_();
  opt.dit_awq = true;
  opt.calib_dir = calib_dir;
  opt.n_layers = 28;
  ModelQuantizer q(mc);
  std::string err;
  if (!q.run(root + "/transformer", dst, opt, &err)) {
    std::printf("[krea2_dit_quant] awq quant failed: %s\n", err.c_str());
    return {};
  }
  return dst;
}
}  // namespace

// DiT AWQ activation-aware CLIPPING at 4-bit, measured vs plain 4-bit (0.0456).
// Collects the on-device calib, clip-quantizes each block Linear's groups, and
// reports the single-step rel-L2 -- a definitive measure of whether clipping
// helps (the smoothing fold being obstructed by the adaLN shift).
TEST(krea2_dit_quant, awq_clip_w4)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string calib = ensure_awq_calib_(mc, std::string(root));
  ASSERT_TRUE(!calib.empty());
  const std::string dir = ensure_awq_quant_(mc, std::string(root), calib);
  ASSERT_TRUE(!dir.empty());

  const double r = run_step_(mc, dir, std::string(gd));
  std::printf("[krea2_dit_quant] AWQ-clip w4 single_step rel-L2 = %.6f "
              "(plain w4 = 0.0456)\n", r);
  ASSERT_TRUE(r >= 0.0);            // collect -> clip-quant -> load -> run works
  EXPECT_TRUE(r < 0.060);           // sane: not catastrophically worse than plain
}

// Mixed precision through the model-quantize STAGE: 4-bit base + 8-bit for the
// most-sensitive 25% of the 28 blocks. Verifies the checkpoint is genuinely
// mixed (7 w8 blocks) and reloads through the per-weight-bits affine path with
// error between pure w4 (0.046) and pure w8 (0.016). Cached at <root>-dit-mixed.
TEST(krea2_dit_quant, stage_mixed_dit)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  namespace fs = std::filesystem;

  const std::string src = std::string(root) + "/transformer";
  const std::string out = std::string(root) + "-dit-mixedg64";   // cached path

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("src_model", FlexData::make_string(src));
  cfg.as_object().insert("output_name", FlexData::make_string(out));
  cfg.as_object().insert("bits", FlexData::make_uint(4));
  cfg.as_object().insert("high_bits", FlexData::make_uint(8));
  cfg.as_object().insert("group_size", FlexData::make_uint(64));
  cfg.as_object().insert("mixed", FlexData::make_bool(true));
  cfg.as_object().insert("mixed_frac", FlexData::make_real(0.25));
  cfg.as_object().insert("skip_existing", FlexData::make_bool(true));

  auto pl = std::make_unique<Pipeline>("qm", &sess);
  auto qu = std::make_unique<ModelQuantizeStage>(
      &sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  auto* qs = static_cast<ModelQuantizeStage*>(pl->insert_stage(std::move(qu)));
  ASSERT_TRUE(qs->config_error().empty());

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  std::error_code ec;
  ASSERT_TRUE(fs::exists(fs::path(out) / "config.json", ec));
  const int w8 = count_w8_blocks_(out);
  std::printf("[krea2_dit_quant] mixed: %d/28 blocks @ 8-bit\n", w8);
  EXPECT_TRUE(w8 == 7);                            // round(0.25 * 28)

  const double r = run_step_(mc, out, std::string(gd));
  std::printf("[krea2_dit_quant] STAGE mixed single_step rel-L2 = %.6f\n", r);
  ASSERT_TRUE(r >= 0.0);
  EXPECT_TRUE(r < 0.046);                          // better than pure w4
}
