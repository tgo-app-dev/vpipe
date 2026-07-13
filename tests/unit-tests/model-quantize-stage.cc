// ModelQuantizeStage: always-on config-validation tests + an env-gated
// real-model quantization run (VPIPE_QUANTIZE_SRC_MODEL = a bf16/f16
// safetensors source dir). The kernel/writer math is covered separately by
// model_quantize_roundtrip; here we exercise the stage + the ModelQuantizer
// driver (enumerate/classify/passthrough/config-rewrite/reload).

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/chat-template.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/model-loader.h"
#include "generative-models/quantize/calibration.h"
#include "generative-models/tokenizer.h"
#include "stages/model-quantize-stage.h"
#include "stages/model-registry.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <streambuf>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace vpipe;

namespace {

class CerrSilencer {
public:
  CerrSilencer() : _saved(std::cerr.rdbuf()) { std::cerr.rdbuf(&_null); }
  ~CerrSilencer() { std::cerr.rdbuf(_saved); }
private:
  struct NullBuf : std::streambuf {
    int overflow(int c) override { return c; }
  };
  std::streambuf* _saved;
  NullBuf         _null;
};

FlexData
basic_cfg_()
{
  FlexData cfg = FlexData::make_object();
  auto o = cfg.as_object();
  o.insert("src_model", FlexData::make_string("/tmp/src-model"));
  o.insert("output_name", FlexData::make_string("/tmp/quantized"));
  return cfg;
}

}  // namespace

TEST(model_quantize_stage, type_is_registered)
{
  EXPECT_TRUE(std::string_view(ModelQuantizeStage::kTypeName) ==
              "model-quantize");
}

// model_dir_available (the post-trigger source-availability gate): an
// existing path is available; a bogus / empty ref is not. This is what
// lets a cascaded fetch->quantize wait until the model actually exists.
TEST(model_quantize_stage, source_availability_check)
{
  namespace fs = std::filesystem;
  Session sess;
  CerrSilencer hush;
  std::error_code ec;
  const auto tmp = fs::temp_directory_path() / "vpipe_avail_probe";
  fs::create_directories(tmp, ec);
  EXPECT_TRUE(model_dir_available(&sess, "models", tmp.string()));
  EXPECT_FALSE(model_dir_available(&sess, "models",
                                   "/no/such/vpipe/model/xyz"));
  EXPECT_FALSE(model_dir_available(&sess, "models", ""));
  fs::remove_all(tmp, ec);
}

// The stage exposes one trigger iport (any beat type) + one FlexData
// summary oport so it can cascade into a preparation recipe / save-text.
TEST(model_quantize_stage, trigger_and_summary_ports)
{
  Session sess;
  CerrSilencer hush;
  ModelQuantizeStage s(&sess, "mq", std::vector<InEdge>{}, basic_cfg_());
  const StageSpec& sp = s.spec();
  ASSERT_TRUE(sp.iports.size() == 1u);
  ASSERT_TRUE(sp.oports.size() == 1u);
  EXPECT_TRUE(std::string_view(sp.iports[0].name) == "trigger");
  EXPECT_TRUE(sp.iports[0].type == nullptr);          // any beat type
  EXPECT_TRUE(std::string_view(sp.oports[0].name) == "summary");
  // By mangled name, not typeid pointer (stage in libvpipe vs test image).
  ASSERT_TRUE(sp.oports[0].type != nullptr);
  EXPECT_TRUE(std::string_view(sp.oports[0].type->name())
              == typeid(FlexDataPayload).name());
}

TEST(model_quantize_stage, config_defaults)
{
  Session sess;
  CerrSilencer hush;
  ModelQuantizeStage s(&sess, "mq", std::vector<InEdge>{}, basic_cfg_());
  EXPECT_TRUE(s.src_model() == "/tmp/src-model");
  EXPECT_TRUE(s.output_name() == "/tmp/quantized");
  EXPECT_TRUE(s.bits() == 8);
  EXPECT_TRUE(s.group_size() == 64);
  EXPECT_TRUE(s.skip_existing());
  EXPECT_TRUE(s.config_error().empty());
}

TEST(model_quantize_stage, invalid_bits_deferred)
{
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"src_model":"/a","output_name":"/b","bits":16})");
  ModelQuantizeStage s(&sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(model_quantize_stage, missing_src_model_deferred)
{
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(R"({"output_name":"/b"})");
  ModelQuantizeStage s(&sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

// End-to-end driver run on a real source model. Skips unless the env points
// at a bf16/f16 safetensors source directory.
TEST(model_quantize_stage, real_model_quantize_reloads)
{
  const char* src = std::getenv("VPIPE_QUANTIZE_SRC_MODEL");
  if (src == nullptr || *src == '\0') { return; }

  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  namespace fs = std::filesystem;
  const fs::path out = fs::temp_directory_path() /
                       ("vpipe-mq-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(out, ec);

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("src_model", FlexData::make_string(src));
    o.insert("output_name", FlexData::make_string(out.string()));
    o.insert("bits", FlexData::make_int(8));
    o.insert("skip_existing", FlexData::make_bool(false));
  }
  ModelQuantizeStage s(&sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  ASSERT_TRUE(s.config_error().empty());
  ASSERT_TRUE(s.quantize_once());

  // Reload the quantized output via the real reader.
  auto wts = vpipe::genai::MetalLlamaWeights::open_model(out.string());
  ASSERT_TRUE(wts.has_value());

  // At least one quantized triple (.scales) must exist, and a U32 .weight.
  bool has_scales = false, has_u32_weight = false, has_bf16_passthrough = false;
  for (const auto& n : wts->tensor_names()) {
    const auto* ti = wts->info(n);
    if (ti == nullptr) { continue; }
    if (n.size() > 7 && n.compare(n.size() - 7, 7, ".scales") == 0) {
      has_scales = true;
    }
    if (ti->dtype == "U32") { has_u32_weight = true; }
    if (ti->dtype == "BF16") { has_bf16_passthrough = true; }
  }
  EXPECT_TRUE(has_scales);
  EXPECT_TRUE(has_u32_weight);
  EXPECT_TRUE(has_bf16_passthrough);   // embeddings/heads passed through

  // config.json carries the quantization block at the outermost level.
  std::ifstream cfg_in((out / "config.json").string());
  ASSERT_TRUE((bool)cfg_in);
  std::string cfg_txt((std::istreambuf_iterator<char>(cfg_in)),
                      std::istreambuf_iterator<char>());
  EXPECT_TRUE(cfg_txt.find("quantization") != std::string::npos);

  fs::remove_all(out, ec);
}

// The awq + mixed-precision config flows through the stage into ModelQuantizer:
// drive a real quantize with awq=true + mixed=true and confirm the output is a
// genuine MIXED checkpoint (both 4-bit and 8-bit affine weights present).
// Env: VPIPE_QUANTIZE_SRC_MODEL (bf16 src) + VPIPE_MOSS15_GOLDEN (calib_*.f32).
TEST(model_quantize_stage, awq_mixed_config_drives_quantizer)
{
  const char* src  = std::getenv("VPIPE_QUANTIZE_SRC_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS15_GOLDEN");
  if (src == nullptr || *src == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  namespace fs = std::filesystem;
  if (!fs::exists(fs::path(gold) / "calib_qkv.f32")) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  const fs::path out = fs::temp_directory_path() /
                       ("vpipe-mq-am-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(out, ec);

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("src_model", FlexData::make_string(src));
    o.insert("output_name", FlexData::make_string(out.string()));
    o.insert("bits", FlexData::make_int(4));
    o.insert("group_size", FlexData::make_int(64));
    o.insert("skip_existing", FlexData::make_bool(false));
    o.insert("awq", FlexData::make_bool(true));
    o.insert("mixed", FlexData::make_bool(true));
    o.insert("high_bits", FlexData::make_int(8));
    o.insert("mixed_frac", FlexData::make_real(0.5));
    o.insert("calib_dir", FlexData::make_string(gold));
    o.insert("layer_prefix", FlexData::make_string("transformer.layers."));
    o.insert("n_layers", FlexData::make_int(36));
  }
  ModelQuantizeStage s(&sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  ASSERT_TRUE(s.config_error().empty());
  ASSERT_TRUE(s.awq() && s.mixed() && s.n_layers() == 36);
  ASSERT_TRUE(s.quantize_once());

  // Output must mix 4-bit (q_proj.weight cols = K/8) and 8-bit (cols = K/4)
  // affine weights -- proof the mixed config reached ModelQuantizer.
  auto wts = vpipe::genai::MetalLlamaWeights::open_model(out.string());
  ASSERT_TRUE(wts.has_value());
  const int H = 2560;
  bool saw4 = false, saw8 = false;
  for (const auto& n : wts->tensor_names()) {
    const auto* ti = wts->info(n);
    if (ti == nullptr || ti->dtype != "U32" || ti->shape.size() != 2) {
      continue;
    }
    if (n.find(".self_attn.q_proj.weight") == std::string::npos) { continue; }
    const int bits = (int)((ti->shape[1] * 32) / H);
    if (bits == 4) { saw4 = true; }
    else if (bits == 8) { saw8 = true; }
  }
  std::fprintf(stderr, "[mq-stage] mixed output: saw4=%d saw8=%d\n",
               (int)saw4, (int)saw8);
  EXPECT_TRUE(saw4 && saw8);
  fs::remove_all(out, ec);
}

// awq=true with NO calib_dir => the stage auto-calibrates on-device (8-bit
// base + tapped forward over a built-in corpus) then AWQ-quantizes. Verifies
// the auto-calib path runs end-to-end and yields a valid quantized checkpoint.
// Env: VPIPE_QUANTIZE_SRC_MODEL (bf16 v1.5 src, with tokenizer.json).
TEST(model_quantize_stage, awq_auto_calibrates)
{
  const char* src = std::getenv("VPIPE_QUANTIZE_SRC_MODEL");
  if (src == nullptr || *src == '\0') { return; }
  namespace fs = std::filesystem;
  if (!fs::exists(fs::path(src) / "tokenizer.json")) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  const fs::path out = fs::temp_directory_path() /
                       ("vpipe-mq-ac-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(out, ec);

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("src_model", FlexData::make_string(src));
    o.insert("output_name", FlexData::make_string(out.string()));
    o.insert("bits", FlexData::make_int(4));
    o.insert("group_size", FlexData::make_int(64));
    o.insert("skip_existing", FlexData::make_bool(false));
    o.insert("awq", FlexData::make_bool(true));
    // NO calib_dir => on-device auto-calibration (arch defaults moss-tts-local).
    o.insert("n_layers", FlexData::make_int(36));
  }
  ModelQuantizeStage s(&sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  ASSERT_TRUE(s.config_error().empty());
  ASSERT_TRUE(s.awq() && s.calib_dir().empty());
  ASSERT_TRUE(s.quantize_once());

  // Output is a valid 4-bit affine checkpoint (auto-calib AWQ ran).
  auto wts = vpipe::genai::MetalLlamaWeights::open_model(out.string());
  ASSERT_TRUE(wts.has_value());
  bool has_scales = false, has_u32 = false;
  for (const auto& n : wts->tensor_names()) {
    const auto* ti = wts->info(n);
    if (ti == nullptr) { continue; }
    if (n.size() > 7 && n.compare(n.size() - 7, 7, ".scales") == 0) {
      has_scales = true;
    }
    if (ti->dtype == "U32") { has_u32 = true; }
  }
  EXPECT_TRUE(has_scales && has_u32);
  fs::remove_all(out, ec);
}

// Streaming MoE calibration MEMORY-SAFETY probe: stream the 35B MoE bf16
// source one layer at a time and confirm the loader is bounded (never a full
// load). Env VPIPE_CALIB_STREAM_SRC = the bf16 source dir. Asserts the probe
// completes (per-layer load/free + free-RAM guard); the watchdog of memory
// safety is the external vm_stat monitor cross-checked against the
// "[calib-stream] ... PEAK system-wired" line this emits.
TEST(model_quantize_stage, streaming_calib_memory_bounded)
{
  const char* src = std::getenv("VPIPE_CALIB_STREAM_SRC");
  if (src == nullptr || *src == '\0') { return; }
  namespace fs = std::filesystem;
  if (!fs::exists(fs::path(src) / "config.json") ||
      !fs::exists(fs::path(src) / "tokenizer.json")) {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  auto tok = vpipe::genai::Tokenizer::from_huggingface_json(
      std::string(src) + "/tokenizer.json", &sess);
  ASSERT_TRUE((bool)tok);
  // A small corpus is enough to prove the loader bound (the residual stream
  // size is what matters, not the layer-weight bytes).
  auto corpus = vpipe::genai::build_builtin_calibration_corpus(
      *tok, /*max_seqs=*/8, /*seq_len=*/256, /*apply_chat_template=*/true);
  ASSERT_TRUE(!corpus.empty());

  // Derive the full backbone shape (GDN/MoE dims, interval, rope) from the
  // checkpoint's config.json -- the same path arch-detect uses -- so the
  // per-layer forward is correctly configured (a hand-built shape with
  // cfg.dense=true would mis-classify the GDN layers as full-attention).
  vpipe::genai::ModelLoader loader(&sess);
  auto mcfg = loader.load_config(std::string(src));
  ASSERT_TRUE(mcfg.has_value());
  vpipe::genai::MetalQwenModel::Config cfg =
      vpipe::genai::MetalQwenModel::config_from(*mcfg);
  cfg.use_bf16 = true;
  // The 35B nests the LM under the REVERSED "model.language_model." root; pin
  // it so the embed/per-layer tensor names resolve (config_from guesses the
  // 4B layout "language_model." + "model.").
  cfg.weight_prefix = "model.language_model.";
  cfg.model_seg = "";

  // PROBE (default): per-layer build/free memory-bound scan only. Set
  // VPIPE_CALIB_STREAM_PROBE=0 to run the full per-layer forward + taps.
  if (std::getenv("VPIPE_CALIB_STREAM_PROBE") == nullptr) {
    ::setenv("VPIPE_CALIB_STREAM_PROBE", "1", 1);
  }
  const fs::path cal = fs::temp_directory_path() /
                       ("vpipe-calstream-" + std::to_string(::getpid()));
  std::string e;
  const bool ok = vpipe::genai::collect_backbone_calibration_streaming(
      sess.metal_compute(), src, cfg, corpus, cal.string(), &e);
  if (!ok) { std::fprintf(stderr, "[calib-stream] FAILED: %s\n", e.c_str()); }
  ASSERT_TRUE(ok);

  // The free-RAM helper must return a sane non-zero value on this box.
  EXPECT_TRUE(vpipe::genai::host_free_ram_bytes() > 0);
  std::error_code ec2;
  fs::remove_all(cal, ec2);
}

// End-to-end Gemma-4 quantization: take a raw-HF bf16 gemma-4 checkpoint,
// quantize it to N-bit MLX-affine via the stage, reload the AFFINE result and
// GENERATE. This exercises (a) the arch-agnostic quantizer on gemma (default
// linear set, no +1 norm fold, names verbatim) and (b) the affine gemma
// forward on the quantized weights -- including E2B's double-wide MLP
// (per-layer ffn). A deterministic prompt must yield "Paris"; a broken quant
// or affine-ffn path gives word-salad. Cross-check for the bf16 dense path.
// Env: VPIPE_GEMMA4_DENSE_TEST_MODEL_PATH (raw bf16 gemma-4 src) + optional
// VPIPE_GEMMA4_QUANT_BITS (8 default; 4 also valid).
TEST(model_quantize_stage, gemma_quantize_generates)
{
  const char* src = std::getenv("VPIPE_GEMMA4_DENSE_TEST_MODEL_PATH");
  if (src == nullptr || *src == '\0') { return; }
  namespace fs = std::filesystem;
  if (!fs::exists(fs::path(src) / "config.json")) { return; }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }

  int bits = 8;
  if (const char* b = std::getenv("VPIPE_GEMMA4_QUANT_BITS")) {
    const int v = std::atoi(b);
    if (v == 4 || v == 8) { bits = v; }
  }
  const fs::path out = fs::temp_directory_path() /
      ("vpipe-gemma-q" + std::to_string(bits) + "-" +
       std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(out, ec);

  {
    FlexData cfg = FlexData::make_object();
    auto o = cfg.as_object();
    o.insert("src_model", FlexData::make_string(src));
    o.insert("output_name", FlexData::make_string(out.string()));
    o.insert("bits", FlexData::make_int(bits));
    o.insert("skip_existing", FlexData::make_bool(false));
    ModelQuantizeStage q(&sess, "mq", std::vector<InEdge>{}, std::move(cfg));
    ASSERT_TRUE(q.config_error().empty());
    ASSERT_TRUE(q.quantize_once());
  }

  // Reload the quantized affine checkpoint + greedy-generate.
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  auto* mgr = sess.generative_model_manager();
  vpipe::genai::LoadSpec spec;
  spec.hf_dir = out.string();
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr ? mgr->load(spec) : nullptr;
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm && lm->valid());
  auto& tok = lm->tokenizer();
  auto tpl = vpipe::genai::make_chat_template(lm->config().architecture, tok);
  ASSERT_TRUE((bool)tpl);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with only the city name.",
      /*is_first_turn=*/true, &ids);
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::int32_t t = lm->prefill(ctx, ids);
  std::vector<std::int32_t> gen;
  for (int i = 0; i < 24 && t >= 0; ++i) {
    if (tpl->is_stop_token(t)) { break; }
    gen.push_back(t);
    t = lm->next_token(ctx);
  }
  const std::string ans =
      tok.decode(std::span<const std::int32_t>(gen.data(), gen.size()));
  // Larger gemmas (e.g. 31B) reason in a <|channel>thought block before
  // answering, so the raw decode is multi-line; flatten newlines for a
  // single legible log line.
  std::string flat = ans;
  for (char& c : flat) { if (c == '\n' || c == '\r') { c = ' '; } }
  std::printf("[gemma_q%d] OUT: %s\n", bits, flat.c_str());
  fs::remove_all(out, ec);
  EXPECT_TRUE(!gen.empty());
  EXPECT_TRUE(ans.find("Paris") != std::string::npos);
}
