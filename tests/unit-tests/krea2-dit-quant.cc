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
#include "residency-reserve.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "generative-models/krea2/metal-krea2-calibration.h"
#include "generative-models/krea2/metal-krea2-transformer.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/quantize/model-quantizer.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "stages/model-quantize-stage.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
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
  // component copied, so it is usable directly as a generate-image hf_dir.
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
// copied through unquantized -- so the output is a valid generate-image hf_dir
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
  // embed_tokens stays bf16 (no .scales) -- the generate-image stage host-
  // gathers the embed table as a plain 2-byte-per-element buffer.
  const std::string enc = (fs::path(out) / "text_encoder").string();
  EXPECT_TRUE(any_tensor_contains_(enc, "q_proj.scales"));
  EXPECT_FALSE(any_tensor_contains_(enc, "embed_tokens.scales"));
  // A quantized encoder that cannot tokenize is not a usable output, and
  // the failure surfaces far away -- at load, as "no tokenizer.json",
  // which reads like a bad checkpoint rather than a missing copy. The
  // directory source gets this from the quantizer's sidecar sweep; a
  // Comfy-Org SINGLE-FILE source has no sidecars to sweep and takes it
  // from the repo root instead, so assert the end state both must reach.
  EXPECT_TRUE(fs::exists(fs::path(enc) / "tokenizer.json", ec));
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

// THE SIGNED INT8 LAYOUT on a real quantized pack (VPIPE_KREA2_DIT_DIR at a
// model-quantize'd DiT; i8_gemm on; M5 only), as flux2_smoke.signed_
// layout_matches_native: converted in the entries a preloaded block keeps,
// in a streamed read and in a refilled slot, copied by promotion -- so
// every arm is BYTE-IDENTICAL to the preloaded one -- and within a
// gross-error bar of the unsigned native route in distance from the
// dense (i8 off) forward.
// 1280 image rows; a 64-row text stream (the signed dequant into
// matmul2d, under the int8 floor) and an 8-row one (the steel tiles
// through the signed dequant).
TEST(krea2_dit_quant, signed_layout_matches_native)
{
  const char* ddir = std::getenv("VPIPE_KREA2_DIT_DIR");
  if (ddir == nullptr || *ddir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  // The split tuner acts between forwards and would make the promoted
  // arm's second forward a different reassociation: splits off.
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  const int gh = 32, gw = 40, img_seq = gh * gw;
  for (int TS : {64, 8}) {
    bool engaged = true;
    auto run = [&](bool sgn, bool i8, bool stream, bool grow, bool slots) {
      std::unique_ptr<Env> e1, e2;
      if (!sgn) { e1 = std::make_unique<Env>("VPIPE_I8_SIGNED", "0"); }
      if (!slots) { e2 = std::make_unique<Env>("VPIPE_KREA2_NO_SLOTS", "1"); }
      MetalKrea2Transformer::Config cfg;
      cfg.i8_gemm = i8;
      std::vector<float> v;
      auto m = MetalKrea2Transformer::load(ddir, mc, cfg, stream);
      if (!m) { return v; }
      if (m->signed_layout() != (sgn && i8)) { engaged = false; }
      m->set_residency_reserve(grow ? test_residency_reserve(4ull << 30) : 0);
      if (grow) { m->set_residency_schedule(4); }
      const int HID = cfg.hidden, IC = cfg.in_channels;
      SharedBuffer fused = mc->make_shared_buffer((std::size_t)TS * HID * 2);
      SharedBuffer lat = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
      std::mt19937 rng(20261016);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      { auto* d = static_cast<_Float16*>(fused.contents());
        for (std::size_t i = 0; i < (std::size_t)TS * HID; ++i) {
          d[i] = (_Float16)nd(rng);
        } }
      { auto* d = static_cast<_Float16*>(lat.contents());
        for (std::size_t i = 0; i < (std::size_t)img_seq * IC; ++i) {
          d[i] = (_Float16)nd(rng);
        } }
      SharedBuffer out;
      for (int it = 0; it < (grow ? 2 : 1); ++it) {
        out = m->forward_dit(fused, TS, lat, img_seq, gh, gw, 0.5f, -1);
      }
      if (grow && m->resident_block_count() == 0) { engaged = false; }
      if (out.empty()) { return v; }
      const auto* p = static_cast<const _Float16*>(out.contents());
      v.resize((std::size_t)img_seq * IC);
      for (std::size_t i = 0; i < v.size(); ++i) { v[i] = (float)p[i]; }
      return v;   // m freed here: one model resident at a time
    };
    const std::vector<float> dense = run(false, false, false, false, true);
    const std::vector<float> nat = run(false, true, false, false, true);
    const std::vector<float> pre = run(true, true, false, false, true);
    const std::vector<float> str = run(true, true, true, false, true);
    const std::vector<float> flat = run(true, true, true, false, false);
    const std::vector<float> grown = run(true, true, true, true, true);
    EXPECT_TRUE(engaged);
    ASSERT_TRUE(!pre.empty() && nat.size() == pre.size() &&
                dense.size() == pre.size());
    if (pre.empty() || nat.size() != pre.size()) { continue; }
    auto same = [&](const std::vector<float>& a) {
      return a.size() == pre.size() &&
             std::memcmp(a.data(), pre.data(), a.size() * 4) == 0;
    };
    const double rs = rel_l2q_(pre.data(), dense.data(), pre.size());
    const double rn = rel_l2q_(nat.data(), dense.data(), pre.size());
    std::printf("[krea2_dit_quant] signed layout, txt %d: vs the dense "
                "forward signed %.3e, native %.3e; streamed %s, no slots "
                "%s, promoted %s\n", TS, rs, rn,
                same(str) ? "IDENTICAL" : "DIFFERENT",
                same(flat) ? "IDENTICAL" : "DIFFERENT",
                same(grown) ? "IDENTICAL" : "DIFFERENT");
    EXPECT_TRUE(same(str));
    EXPECT_TRUE(same(flat));
    EXPECT_TRUE(same(grown));
    // A GROSS-ERROR guard, not a ranking: on random inputs this forward
    // is chaotic -- QI21's dense forward moved 4.8e-2 for a per-weight
    // change of 5e-5 (gemm_i8.signed_dequant_on_a_real_pack: both
    // dequants 7.3e-3 from the bf16 weights, equal to four digits) -- so
    // two equally accurate routes land a few percent apart either way.
    // gemm_i8.signed_layout_matches_oracle is the accuracy statement; a
    // misread layout is O(1) here.
    EXPECT_TRUE(rs <= 1.25 * rn);
  }
}

// A MIXED PACK (VPIPE_KREA2_MIXED_DIT = the transformer/ of a
// model-quantize bits 4 + mixed run): whole main blocks at w8, the rest
// at w4, so a slot built for one width is asked to hold the other --
// including the gate|up weave it derives after the read
// (BlockSlots::Ops::same_layout). The pack must really mix widths, and
// its four holds -- preloaded, streamed, no slots, promoted -- must be
// BYTE-IDENTICAL with i8 off, and on a matrix-core box under the signed
// layout and (when block 0, which gates the fast tier, is w8) the fast
// and token tiers. Before the slots rebuilt such a block, preloaded and
// no-slots agreed and the two slot holds did not.
TEST(krea2_dit_quant, a_mixed_pack_holds_agree)
{
  const char* ddir = std::getenv("VPIPE_KREA2_MIXED_DIT");
  if (ddir == nullptr || *ddir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  int n8 = 0, n4 = 0;
  {
    auto ws = vpipe::genai::WeightSet::open(ddir, nullptr);
    ASSERT_TRUE(ws != nullptr);
    if (ws == nullptr) { return; }
    for (int L = 0; L < 64; ++L) {
      const int b = vpipe::genai::stored_affine_bits(
          *ws, "transformer_blocks." + std::to_string(L) + ".attn.to_q", 64);
      n8 += b == 8;
      n4 += b == 4;
    }
  }
  std::printf("[krea2_dit_quant] mixed pack: %d main blocks w8, %d w4\n",
              n8, n4);
  ASSERT_TRUE(n8 > 0 && n4 > 0);
  if (n8 == 0 || n4 == 0) { return; }
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  const int gh = 32, gw = 40, img_seq = gh * gw, TS = 64;
  bool engaged = true;
  bool fast_on = false;
  int f8 = 0, f4 = 0;
  int promoted = 0, relaid = 0;
  int stack = 0;
  std::size_t grown_bytes = 0;
  auto run = [&](int tier, bool stream, bool slots, bool grow,
                 bool hold = false) {
    std::unique_ptr<Env> ns;
    if (!slots) { ns = std::make_unique<Env>("VPIPE_KREA2_NO_SLOTS", "1"); }
    MetalKrea2Transformer::Config cfg;
    cfg.i8_gemm = tier >= 0;
    cfg.i8_tier = tier < 0 ? 0 : tier;
    cfg.i8_hold_w4 = hold;
    std::vector<float> v;
    auto m = MetalKrea2Transformer::load(ddir, mc, cfg, stream);
    if (!m) { return v; }
    if (tier == 0 && !m->signed_layout()) { engaged = false; }
    if (tier > 0) {
      fast_on = m->fast_tier();
      if (!stream) {
        f8 = m->fast_weights(8);
        f4 = m->fast_weights(4);
      }
    }
    m->set_residency_reserve(grow ? test_residency_reserve(16ull << 30) : 0);
    if (grow) { m->set_residency_schedule(4); }
    const int HID = cfg.hidden, IC = cfg.in_channels;
    SharedBuffer fused = mc->make_shared_buffer((std::size_t)TS * HID * 2);
    SharedBuffer lat = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
    std::mt19937 rng(20261018);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    { auto* d = static_cast<_Float16*>(fused.contents());
      for (std::size_t i = 0; i < (std::size_t)TS * HID; ++i) {
        d[i] = (_Float16)nd(rng);
      } }
    { auto* d = static_cast<_Float16*>(lat.contents());
      for (std::size_t i = 0; i < (std::size_t)img_seq * IC; ++i) {
        d[i] = (_Float16)nd(rng);
      } }
    SharedBuffer out;
    for (int it = 0; it < (grow ? 2 : 1); ++it) {
      out = m->forward_dit(fused, TS, lat, img_seq, gh, gw, 0.5f, -1);
    }
    if (grow) {
      promoted = m->resident_block_count();
      stack = m->config().n_layers;
      grown_bytes = m->held_block_bytes();
    }
    if (stream && slots && !grow) { relaid = m->slot_relayouts(); }
    if (out.empty()) { return v; }
    const auto* p = static_cast<const _Float16*>(out.contents());
    v.resize((std::size_t)img_seq * IC);
    for (std::size_t i = 0; i < v.size(); ++i) { v[i] = (float)p[i]; }
    return v;   // m freed here: one model resident at a time
  };
  std::vector<int> tiers = {-1};
  if (mc->supports_matrix_cores()) { tiers.insert(tiers.end(), {0, 1, 2}); }
  const char* tname[3] = {"accurate (signed)", "fast", "token"};
  for (int tier : tiers) {
    engaged = true;
    const std::vector<float> pre = run(tier, false, true, false);
    ASSERT_TRUE(!pre.empty());
    if (pre.empty()) { return; }
    if (tier > 0) {
      // BOTH WIDTHS take the tier: a w8 block's codes converted once,
      // a w4 one's into the scratch before each GEMM. Uniform w8 packs:
      // fast_tier_matches_accurate.
      std::printf("[krea2_dit_quant] mixed pack, %s: %d w8 and %d w4 "
                  "projections in its layout\n", tname[tier], f8, f4);
      EXPECT_TRUE(fast_on && f8 > 0 && f4 > 0);
      if (!fast_on) { continue; }
    }
    const std::vector<float> str = run(tier, true, true, false);
    const std::vector<float> flat = run(tier, true, false, false);
    const std::vector<float> grown = run(tier, true, true, true);
    const std::size_t grown_pc = grown_bytes;
    // A HELD layout reached by PROMOTION: residency converts each w4
    // block it keeps (clone_block_), which then holds more than the
    // pack -- and must equal the held preload.
    const std::vector<float> grown_held =
        tier > 0 ? run(tier, true, true, true, true) : grown;
    const std::size_t grown_h = grown_bytes;
    const int grown_n = promoted;
    // ...and with NO SLOTS, where promote_into MOVES each per-block read
    // into the resident set: the keep pass (keep_block_) must leave it
    // holding exactly what the slots' clone does.
    const std::vector<float> flat_held =
        tier > 0 ? run(tier, true, false, true, true) : grown;
    const std::size_t flat_h = tier > 0 ? grown_bytes : grown_h;
    const int flat_n = tier > 0 ? promoted : grown_n;
    // The bytes compare only where BOTH kept the whole streamed stack:
    // a slotted run short of it still holds its slot pair, and a box
    // with little room (a 16 GB M5) keeps what its budget allows.
    const bool whole = grown_n == stack && flat_n == stack;
    // The engine's other answer for a w4 pack: its int8 layout
    // HELD (Config::i8_hold_w4) where it is converted per GEMM
    // above -- the same values, so the same bytes.
    const auto held =
        tier > 0 ? run(tier, false, true, false, true) : pre;
    auto same = [&](const std::vector<float>& a) {
      return a.size() == pre.size() &&
             std::memcmp(a.data(), pre.data(), a.size() * 4) == 0;
    };
    std::printf("[krea2_dit_quant] mixed pack, %s: streamed %s, no slots "
                "%s, promoted (%d blocks) %s%s\n",
                tier < 0 ? "no i8" : tname[tier],
                same(str) ? "IDENTICAL" : "DIFFERENT",
                same(flat) ? "IDENTICAL" : "DIFFERENT", promoted,
                same(grown) ? "IDENTICAL" : "DIFFERENT",
                engaged ? "" : " -- NOT ENGAGED");
    EXPECT_TRUE(engaged);
    EXPECT_TRUE(same(str));
    EXPECT_TRUE(same(flat));
    EXPECT_TRUE(promoted > 0 && same(grown));
    if (tier > 0) {
      std::printf("[krea2_dit_quant] ... w4 layout held: %s; held by "
                  "promotion %s (%zu MB kept against %zu per call), with "
                  "no slots %s (%zu MB)\n",
                  same(held) ? "IDENTICAL" : "DIFFERENT",
                  same(grown_held) ? "IDENTICAL" : "DIFFERENT",
                  grown_h >> 20, grown_pc >> 20,
                  same(flat_held) ? "IDENTICAL" : "DIFFERENT",
                  flat_h >> 20);
      EXPECT_TRUE(same(grown_held) && grown_h > grown_pc);
      // The moved blocks must hold the bytes the cloned ones do.
      EXPECT_TRUE(same(flat_held) && (!whole || flat_h == grown_h));
    }
    EXPECT_TRUE(same(held));
    // The slots crossed widths and REBUILT. A slot that refused such a
    // block instead turns itself off, and its holds still agree.
    std::printf("[krea2_dit_quant] ... %d streamed blocks built into their "
                "slot\n", relaid);
    EXPECT_TRUE(relaid > 0);
  }
}

// THE FAST AND TOKEN TIERS (i8_gemm_tier fast / token / token_plain) on
// the main blocks, as qwen_image_21_dit.fast_tier_matches_accurate: a w8
// g64 pack (VPIPE_KREA2_DIT_DIR) converted in the entries a preloaded
// block keeps, in a streamed read, in a refilled slot and copied by
// promotion -- every arm BYTE-IDENTICAL to the preloaded one -- and a
// bf16 checkpoint (VPIPE_KREA2_TEST_MODEL_PATH's transformer/) quantized
// once at load, or after each refill when streamed, its every hold
// byte-identical too. Each arm must ENGAGE (the tier at the group it names)
// before anything is compared, and sit within a gross-error bar of the
// accurate tier in distance from the dense (i8 off) forward. 1280 image
// rows, above the int8 floor of 1024; a 64-row text stream, whose fusion
// tower stays on the accurate tier. M5 only.
TEST(krea2_dit_quant, fast_tier_matches_accurate)
{
  const char* qd = std::getenv("VPIPE_KREA2_DIT_DIR");
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const std::string pack = qd != nullptr ? qd : "";
  const std::string bf16 =
      root != nullptr && *root != '\0' ? std::string(root) + "/transformer"
                                        : std::string();
  if (pack.empty() && bf16.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  struct Env {
    const char* k;
    Env(const char* key, const char* v) : k(key) { ::setenv(key, v, 1); }
    ~Env() { ::unsetenv(k); }
  };
  Env nosplit("VPIPE_I8_SPLITK_MAX_S", "0");
  const int gh = 32, gw = 40, img_seq = gh * gw, TS = 64;
  bool engaged = true;
  auto run = [&](const std::string& dir, bool i8, int tier, bool stream,
                 bool grow, bool slots) {
    std::unique_ptr<Env> e2;
    if (!slots) { e2 = std::make_unique<Env>("VPIPE_KREA2_NO_SLOTS", "1"); }
    MetalKrea2Transformer::Config cfg;
    cfg.i8_gemm = i8;
    cfg.i8_tier = tier;
    std::vector<float> v;
    auto m = MetalKrea2Transformer::load(dir, mc, cfg, stream);
    if (!m) { return v; }
    const int fg = m->fast_group();
    if (m->fast_tier() != (i8 && tier != 0) ||
        (i8 && tier >= 2 && fg != I8GemmContext::kTokenGroup) ||
        (i8 && tier == 1 && (fg <= 0 || fg == I8GemmContext::kTokenGroup))) {
      engaged = false;
    }
    m->set_residency_reserve(grow ? test_residency_reserve(4ull << 30) : 0);
    if (grow) { m->set_residency_schedule(4); }
    const int HID = cfg.hidden, IC = cfg.in_channels;
    SharedBuffer fused = mc->make_shared_buffer((std::size_t)TS * HID * 2);
    SharedBuffer lat = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
    std::mt19937 rng(20261018);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    { auto* d = static_cast<_Float16*>(fused.contents());
      for (std::size_t i = 0; i < (std::size_t)TS * HID; ++i) {
        d[i] = (_Float16)nd(rng);
      } }
    { auto* d = static_cast<_Float16*>(lat.contents());
      for (std::size_t i = 0; i < (std::size_t)img_seq * IC; ++i) {
        d[i] = (_Float16)nd(rng);
      } }
    SharedBuffer out;
    for (int it = 0; it < (grow ? 2 : 1); ++it) {
      out = m->forward_dit(fused, TS, lat, img_seq, gh, gw, 0.5f, -1);
    }
    if (grow && m->resident_block_count() == 0) { engaged = false; }
    if (out.empty()) { return v; }
    const auto* p = static_cast<const _Float16*>(out.contents());
    v.resize((std::size_t)img_seq * IC);
    for (std::size_t i = 0; i < v.size(); ++i) { v[i] = (float)p[i]; }
    return v;   // m freed here: one model resident at a time
  };
  auto same = [](const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && !a.empty() &&
           std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
  };
  // A GROSS-ERROR guard, not a ranking (see signed_layout_matches_native:
  // random-input forwards are chaotic): the fast tier's GEMMs are
  // 1.3-1.45x the accurate tier's error on outlier-free operands, the
  // plain token tier's ~1.7x on real weights and the rotated one's about
  // the fast tier's, so a misread layout is what this catches.
  const double bar[4] = {0.0, 2.5, 3.0, 4.0};
  const char* tname[4] = {"accurate", "fast", "token", "token_plain"};
  for (const std::string& dir : {pack, bf16}) {
    if (dir.empty()) { continue; }
    const bool is_pack = dir == pack;
    const std::vector<float> dense = run(dir, false, 0, false, false, true);
    const std::vector<float> acc = run(dir, true, 0, false, false, true);
    ASSERT_TRUE(!dense.empty() && acc.size() == dense.size());
    if (dense.empty() || acc.size() != dense.size()) { continue; }
    // The bars were set against the accurate tier with its groups
    // unrotated; the rotation (on by default) lowers that yardstick where
    // a channel stands out, so the bars hold against the larger of the
    // two, and the rotated forward is held to a gross bar of the plain.
    const double ra1 = rel_l2q_(acc.data(), dense.data(), acc.size());
    double ra0 = ra1;
    {
      Env plain("VPIPE_I8_ACC_ROT", "0");
      const std::vector<float> acc0 = run(dir, true, 0, false, false, true);
      if (acc0.size() == dense.size()) {
        ra0 = rel_l2q_(acc0.data(), dense.data(), acc0.size());
      }
    }
    std::printf("[krea2_dit_quant] %s, accurate tier vs the dense forward: "
                "%.3e rotated, %.3e plain\n", is_pack ? "w8 pack" : "bf16",
                ra1, ra0);
    EXPECT_TRUE(ra1 <= 1.25 * ra0);
    const double ra = std::max(ra1, ra0);
    for (int tier : {1, 2, 3}) {
      engaged = true;
      const std::vector<float> pre = run(dir, true, tier, false, false, true);
      // EVERY HOLD of either checkpoint runs the tier the preload runs: a
      // pack's codes converted in the slot, a bf16 block's source
      // quantized after each refill (QWeight::ds) and a promotion keeping
      // the layout alone.
      const std::vector<float> str = run(dir, true, tier, true, false, true);
      const std::vector<float> flat =
          run(dir, true, tier, true, false, false);
      const std::vector<float> grown =
          run(dir, true, tier, true, true, true);
      EXPECT_TRUE(engaged);
      ASSERT_TRUE(pre.size() == dense.size());
      if (pre.size() != dense.size()) { continue; }
      const double rf = rel_l2q_(pre.data(), dense.data(), pre.size());
      std::printf("[krea2_dit_quant] %s tier, %s, %d+%d rows: vs the dense "
                  "forward %.3e, accurate %.3e; streamed %s, no slots %s, "
                  "promoted %s\n", tname[tier],
                  is_pack ? "w8 pack" : "bf16", img_seq, TS, rf, ra,
                  same(str, pre) ? "IDENTICAL" : "DIFFERENT",
                  same(flat, pre) ? "IDENTICAL" : "DIFFERENT",
                  same(grown, pre) ? "IDENTICAL" : "DIFFERENT");
      EXPECT_TRUE(same(str, pre));
      EXPECT_TRUE(same(flat, pre));
      EXPECT_TRUE(same(grown, pre));
      EXPECT_TRUE(rf <= bar[tier] * ra);
    }
  }
}

// THE SIGNED LAYOUT'S WORTH in the Krea-2 forward, i8_gemm on, at 1024^2
// (4096 image rows + 512 text). Arms: the quantized pack
// (VPIPE_KREA2_DIT_DIR) signed and native (VPIPE_I8_SIGNED=0), both
// preloaded; and, where a box cannot hold the bf16 DiT, both formats
// STREAMED (VPIPE_KREA2_TEST_MODEL_PATH's transformer/ and the pack, no
// residency) so they meet on equal terms. Each arm warms two forwards and
// keeps the median of VPIPE_KREA2_SIGNED_BENCH_ITERS; the arms rotate
// through VPIPE_KREA2_SIGNED_BENCH_ROUNDS rounds, reversed every other
// one. VPIPE_KREA2_SIGNED_BENCH=1 runs it.
TEST(krea2_dit_quant, signed_layout_forward)
{
  const char* ddir = std::getenv("VPIPE_KREA2_DIT_DIR");
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (ddir == nullptr || *ddir == '\0' ||
      std::getenv("VPIPE_KREA2_SIGNED_BENCH") == nullptr) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  auto env_int = [](const char* k, int d) {
    const char* e = std::getenv(k);
    return e != nullptr ? std::max(1, std::atoi(e)) : d;
  };
  const int iters = env_int("VPIPE_KREA2_SIGNED_BENCH_ITERS", 3);
  const int rounds = env_int("VPIPE_KREA2_SIGNED_BENCH_ROUNDS", 2);
  const int gh = 64, gw = 64, img_seq = gh * gw, TS = 512;
  struct Arm { std::string name, dir; bool sgn, stream; };
  std::vector<Arm> arms = {
      {"w8 signed", ddir, true, false},
      {"w8 native", ddir, false, false},
      {"w8 signed, streamed", ddir, true, true},
  };
  if (root != nullptr && *root != '\0') {
    arms.push_back({"bf16, streamed", std::string(root) + "/transformer",
                    true, true});
  }
  const int A = (int)arms.size();
  std::vector<std::vector<double>> med((std::size_t)A);
  for (int r = 0; r < rounds; ++r) {
    for (int k = 0; k < A; ++k) {
      const int a = (r % 2 == 0) ? k : A - 1 - k;
      const Arm& arm = arms[(std::size_t)a];
      if (!arm.sgn) { ::setenv("VPIPE_I8_SIGNED", "0", 1); }
      MetalKrea2Transformer::Config cfg;
      cfg.i8_gemm = true;
      auto m = MetalKrea2Transformer::load(arm.dir, mc, cfg, arm.stream);
      ::unsetenv("VPIPE_I8_SIGNED");
      ASSERT_TRUE(m != nullptr);
      if (!m) { return; }
      m->set_residency_reserve(0);
      const int HID = cfg.hidden, IC = cfg.in_channels;
      SharedBuffer fused = mc->make_shared_buffer((std::size_t)TS * HID * 2);
      SharedBuffer lat = mc->make_shared_buffer((std::size_t)img_seq * IC * 2);
      std::mt19937 rng(0);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      { auto* d = static_cast<_Float16*>(fused.contents());
        for (std::size_t i = 0; i < (std::size_t)TS * HID; ++i) {
          d[i] = (_Float16)nd(rng);
        } }
      { auto* d = static_cast<_Float16*>(lat.contents());
        for (std::size_t i = 0; i < (std::size_t)img_seq * IC; ++i) {
          d[i] = (_Float16)nd(rng);
        } }
      for (int w = 0; w < 2; ++w) {
        ASSERT_TRUE(!m->forward_dit(fused, TS, lat, img_seq, gh, gw, 0.5f,
                                    -1).empty());
      }
      std::vector<double> t;
      for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        SharedBuffer out =
            m->forward_dit(fused, TS, lat, img_seq, gh, gw, 0.5f, -1);
        t.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count());
        ASSERT_TRUE(!out.empty());
      }
      std::sort(t.begin(), t.end());
      med[(std::size_t)a].push_back(t[t.size() / 2]);
      std::printf("[krea2_dit_quant] round %d %-20s %8.1f ms/forward "
                  "(signed layout %s)\n", r, arm.name.c_str(),
                  t[t.size() / 2], m->signed_layout() ? "on" : "off");
    }
  }
  std::printf("[krea2_dit_quant] 1024^2, %d+%d rows, medians over %d "
              "rounds:\n", img_seq, TS, rounds);
  for (int a = 0; a < A; ++a) {
    std::vector<double> v = med[(std::size_t)a];
    std::sort(v.begin(), v.end());
    std::printf("[krea2_dit_quant]   %-20s %8.1f ms\n",
                arms[(std::size_t)a].name.c_str(), v[v.size() / 2]);
  }
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

// THE ANE TIERS AND THEIR GPU FALLBACK, on the real checkpoint at 1024^2
// with random inputs. Four arms, each its own model:
//
//   GPU-only                      the reference
//   ANE ff + q|k|v|gate           a split: fp16 rows, so close, not equal
//   lost at the first split       VPIPE_ANE_FAIL=0:0 -- every ANE row of
//                                 block 0 goes back to the GPU and both
//                                 tiers stay there, so the forward is the
//                                 GPU's to the bit; any later split would
//                                 put fp16 rows back in
//   lost at a second chunk        a pinned share losing chunk 1: the
//                                 first chunk's ANE rows are kept and the
//                                 rest recomputed; the wrong range lands
//                                 far from the GPU arm. In 1024-row
//                                 chunks, because the q|k|v|gate tier
//                                 keeps its OWN auto share, which opens at
//                                 half the rows -- one 2048-row chunk here,
//                                 with no second chunk to lose
//
// Env: VPIPE_KREA2_TEST_MODEL_PATH.
TEST(krea2_ane, lost_rows_fall_back_to_the_gpu)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const MetalKrea2Transformer::Config cfg0;
  const int HID = cfg0.hidden, IC = cfg0.in_channels;
  // 64 x 64 = 4096 image rows plus the text: the tiers chunk at 2048, so
  // this is the smallest square that gives a pinned share two chunks.
  const int grid = 64, img_seq = grid * grid, text_seq = 64;
  const std::string dit = std::string(root) + "/transformer";
  auto rnd = [&](std::size_t n, std::uint32_t seed) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<_Float16*>(b.contents());
    std::uint32_t st = seed;
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      d[i] = (_Float16)(((float)(st >> 8) / 8388608.0f - 1.0f) * 0.5f);
    }
    return b;
  };
  const SharedBuffer fused = rnd((std::size_t)text_seq * HID, 71u);
  const SharedBuffer lat = rnd((std::size_t)img_seq * IC, 72u);
  ASSERT_TRUE(!fused.empty() && !lat.empty());
  if (fused.empty() || lat.empty()) { return; }

  enum Arm { kGpu, kAne, kLost0, kLost1, kArms };
  struct Out {
    std::vector<float> v;
    bool armed = false, latched = false;
  };
  auto run = [&](int arm) {
    Out o;
    MetalKrea2Transformer::Config cfg;
    if (arm != kGpu) {
      cfg.session  = &sess;
      cfg.ane_qkv  = true;
      cfg.ane_rows = arm == kLost1 ? 1.0f : 0.0f;
    }
    auto m = MetalKrea2Transformer::load(dit, mc, cfg);
    if (!m) { return o; }
    const char* spec = arm == kLost0 ? "0:0" : arm == kLost1 ? "0:1"
                                                             : nullptr;
    if (spec != nullptr) { setenv("VPIPE_ANE_FAIL", spec, 1); }
    if (arm == kLost1) { setenv("VPIPE_KREA2_ANE_CHUNK", "1024", 1); }
    SharedBuffer out = m->forward_dit(fused, text_seq, lat, img_seq, grid,
                                      grid, 0.5f, -1);
    if (spec != nullptr) { unsetenv("VPIPE_ANE_FAIL"); }
    if (arm == kLost1) { unsetenv("VPIPE_KREA2_ANE_CHUNK"); }
    if (out.byte_size() < (std::size_t)img_seq * IC * 2) { return o; }
    const auto* p = static_cast<const _Float16*>(out.contents());
    o.v.resize((std::size_t)img_seq * IC);
    for (std::size_t i = 0; i < o.v.size(); ++i) { o.v[i] = (float)p[i]; }
    o.armed   = m->ane_armed() && m->ane_qkv_armed();
    o.latched = m->ane_gpu_only() && m->ane_qkv_gpu_only();
    return o;      // the model goes here: one 24 GB DiT resident at a time
  };
  Out out[kArms];
  for (int a = 0; a < kArms; ++a) {
    out[a] = run(a);
    ASSERT_TRUE(out[a].v.size() == (std::size_t)img_seq * IC);
    if (out[a].v.size() != (std::size_t)img_seq * IC) { return; }
  }
  if (!out[kAne].armed) {
    std::printf("[krea2_ane] ANE did not arm -- SKIPPED\n");
    return;
  }
  auto finite = [](const std::vector<float>& v) {
    for (float f : v) {
      if (!std::isfinite(f)) { return false; }
    }
    return true;
  };
  const std::size_t n = out[kGpu].v.size();
  const double r_ane = rel_l2q_(out[kAne].v.data(), out[kGpu].v.data(), n);
  const double r_f0 = rel_l2q_(out[kLost0].v.data(), out[kGpu].v.data(), n);
  const double r_f1 = rel_l2q_(out[kLost1].v.data(), out[kGpu].v.data(), n);
  std::printf("[krea2_ane] against GPU-only at 1024^2: ANE split rel-L2 "
              "%.4e; rows lost -> GPU %.4e (first split, every row), %.4e "
              "(second chunk of a pinned share)\n", r_ane, r_f0, r_f1);
  EXPECT_TRUE(r_ane > 0.0 && r_ane < 0.05);   // rows reached the ANE
  EXPECT_TRUE(out[kLost0].latched);           // engaged: both fell back
  EXPECT_TRUE(out[kLost1].latched);
  EXPECT_TRUE(!out[kAne].latched);
  EXPECT_TRUE(finite(out[kLost0].v) && finite(out[kLost1].v));
  EXPECT_TRUE(r_f0 < 1e-6);                   // every row the GPU's
  EXPECT_TRUE(r_f0 < 0.01 * r_ane);
  EXPECT_TRUE(r_f1 < 0.05);                   // the right rows recomputed
  EXPECT_TRUE(r_f1 > r_f0);                   // the first chunk kept
}
