// Env-driven driver + smoke for MOSS-TTS-Realtime (moss_tts_realtime).
// All tests skip vacuously without their env vars.
//
//   Quantize:  VPIPE_MOSS_RT_SRC=<raw bf16 dir> VPIPE_MOSS_RT_MODEL=<out 8-bit>
//              -> 8-bit affine quantize (arch moss-tts-local: embeds/heads bf16)
//   Forward:   VPIPE_MOSS_RT_MODEL=<8-bit dir> VPIPE_MOSS_RT_GOLDEN=<rt_golden.bin>
//              -> backbone hidden rel-L2 vs the python golden + local frame-0
//   Synth:     VPIPE_MOSS_RT_MODEL + VPIPE_MOSS_CODEC_MODEL (+ optional
//              VPIPE_MOSS_RT_TEXT) -> end-to-end generate + 24 kHz decode, RMS

#include "minitest.h"

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/session.h"
#include "generative-models/context-manager.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/moss/metal-moss-rt-model.h"
#include "generative-models/moss/moss-rt-processor.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/tokenizer.h"
#include "stages/model-quantize-stage.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace vpipe;

namespace {
float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}
const char* env_(const char* k) { const char* v = std::getenv(k); return v; }

// Fill the realtime model config (28-layer backbone + 4-layer local). Same for
// an unquantized bf16 dir OR a model-quantize'd 8-bit dir -- MetalQwenModel
// auto-detects raw vs affine from the checkpoint. zero_centered_norm=false:
// plain Qwen3 std RMSNorm (the dense path folds +1 otherwise).
void fill_cfg_(genai::MetalMossRtModel::Config& cfg)
{
  auto set = [&](genai::MetalQwenModel::Config& c, const std::string& pfx,
                 int nl, int ms, int pt) {
    c.n_layers = nl; c.hidden = 2048; c.n_heads = 16; c.n_kv_heads = 8;
    c.head_dim = 128; c.ffn_inner = 6144; c.vocab = 151936;
    c.rope_theta = 1.0e6f; c.rms_eps = 1e-6f; c.rotary_dim = 128;
    c.full_attn_interval = 1; c.tie_embeddings = false; c.use_bf16 = true;
    c.quant_bits = 8; c.dense = true; c.attn_output_gate = false;
    c.zero_centered_norm = false;
    c.backbone_only = true; c.weight_prefix = pfx; c.model_seg = "model.";
    c.max_seq = ms; c.page_tokens = pt;
  };
  set(cfg.backbone, "language_model.", 28, 2048, 256);
  set(cfg.local, "local_transformer.", 4, 32, 32);
}
}  // namespace

// (a) Quantize the raw bf16 checkpoint to 8-bit (persistent). arch
// "moss-tts-local" keeps embeddings/heads/norms bf16 (host-side gather).
TEST(moss_rt_driver, quantize)
{
  const char* src = env_("VPIPE_MOSS_RT_SRC");
  const char* out = env_("VPIPE_MOSS_RT_MODEL");
  if (src == nullptr || *src == '\0' || out == nullptr || *out == '\0') {
    return;
  }
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("src_model", FlexData::make_string(src));
    o.insert("output_name", FlexData::make_string(out));  // absolute -> verbatim
    o.insert("bits", FlexData::make_int(8));
    o.insert("group_size", FlexData::make_int(64));
    // No explicit arch: auto-detect must map config.json model_type
    // "moss_tts_realtime" -> "moss-tts-realtime" (embeds/heads stay bf16).
    o.insert("skip_existing", FlexData::make_bool(true));
  }
  ModelQuantizeStage q(&sess, "mq", std::vector<InEdge>{}, std::move(cfg));
  ASSERT_TRUE(q.config_error().empty());
  EXPECT_TRUE(q.quantize_once());
}

// (b) Forward correctness vs the python golden (teacher-forced grid): the
// backbone last-position hidden rel-L2 (loose -- 8-bit quant on top of bf16),
// and the leading local codebooks (which matched HF in fp32).
TEST(moss_rt_driver, forward)
{
  const char* dir = env_("VPIPE_MOSS_RT_MODEL");
  const char* gp  = env_("VPIPE_MOSS_RT_GOLDEN");
  if (dir == nullptr || *dir == '\0' || gp == nullptr || *gp == '\0') {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // Read the golden binary (little-endian, see dump_rt_golden.py).
  std::FILE* f = std::fopen(gp, "rb");
  if (f == nullptr) { return; }
  auto rd_i = [&]() { std::int32_t v = 0; std::fread(&v, 4, 1, f); return v; };
  const int seq = rd_i(), nvq1 = rd_i();
  std::vector<std::vector<std::int32_t>> grid((std::size_t)seq);
  for (int r = 0; r < seq; ++r) {
    grid[(std::size_t)r].resize((std::size_t)nvq1);
    std::fread(grid[(std::size_t)r].data(), 4, (std::size_t)nvq1, f);
  }
  const int H = rd_i();
  std::vector<float> gold_h((std::size_t)H);
  std::fread(gold_h.data(), 4, (std::size_t)H, f);
  std::vector<std::int32_t> gold_args(16);
  std::fread(gold_args.data(), 4, 16, f);
  std::fclose(f);

  genai::MetalMossRtModel::Config cfg;
  // Match the stage's rt_qwen_cfg_ (28-layer backbone, 4-layer local, 8-bit).
  auto set = [&](genai::MetalQwenModel::Config& c, const std::string& pfx,
                 int nl, int ms, int pt) {
    c.n_layers = nl; c.hidden = 2048; c.n_heads = 16; c.n_kv_heads = 8;
    c.head_dim = 128; c.ffn_inner = 6144; c.vocab = 151936;
    c.rope_theta = 1.0e6f; c.rms_eps = 1e-6f; c.rotary_dim = 128;
    c.full_attn_interval = 1; c.tie_embeddings = false; c.use_bf16 = true;
    c.quant_bits = 8; c.dense = true; c.attn_output_gate = false;
    c.zero_centered_norm = false;   // plain Qwen3 std RMSNorm (dense path)
    c.backbone_only = true; c.weight_prefix = pfx; c.model_seg = "model.";
    c.max_seq = ms; c.page_tokens = pt;
  };
  set(cfg.backbone, "language_model.", 28, 2048, 256);
  set(cfg.local, "local_transformer.", 4, 32, 32);

  auto m = genai::MetalMossRtModel::load(dir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  auto* bb = m->backbone();
  auto* cm = bb->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  metal_compute::SharedBuffer emb = m->assemble_embeds(grid, 0, seq);
  metal_compute::SharedBuffer h = bb->forward_embeddings_hidden(cid, emb, seq);
  ASSERT_TRUE(!h.empty());

  const auto* hb = static_cast<const std::uint16_t*>(h.contents());
  double num = 0.0, den = 0.0;
  for (int i = 0; i < H; ++i) {
    const float mv = bf16_to_f32_(hb[i]);
    const float gv = gold_h[(std::size_t)i];
    num += (double)(mv - gv) * (mv - gv);
    den += (double)gv * gv;
  }
  const double rel = std::sqrt(num / (den + 1e-12));
  std::fprintf(stderr, "[moss-rt] backbone hidden rel-L2 = %.4f\n", rel);
  EXPECT_TRUE(rel < 0.12);   // 8-bit affine on top of bf16 -> a few %

  genai::MossSampling greedy;   // temperature 0 => argmax
  std::uint64_t rng = 1;
  std::vector<int> codes = m->local_frame(h, greedy, rng);
  cm->release(cid);
  ASSERT_TRUE((int)codes.size() == 16);
  std::fprintf(stderr, "[moss-rt] local frame0 codes:");
  for (int c : codes) { std::fprintf(stderr, " %d", c); }
  std::fprintf(stderr, "\n  (golden:");
  for (int c : gold_args) { std::fprintf(stderr, " %d", c); }
  std::fprintf(stderr, ")\n");
  for (int c : codes) { EXPECT_TRUE(c >= 0 && c <= 1026); }
  // Codebook 0 is single-position (no cross-codebook attn) -> most robust to
  // quant; expect it to match the golden or be a near-tie neighbour.
  EXPECT_TRUE(codes[0] != 1026);   // not an immediate EOS (would be silence)
}

// (c) End-to-end synthesis: generate (sampled) + 24 kHz decode, non-silent RMS.
TEST(moss_rt_driver, synth)
{
  const char* dir = env_("VPIPE_MOSS_RT_MODEL");
  const char* cdir = env_("VPIPE_MOSS_CODEC_MODEL");
  if (dir == nullptr || *dir == '\0' || cdir == nullptr || *cdir == '\0') {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  genai::MetalMossRtModel::Config cfg;
  auto set = [&](genai::MetalQwenModel::Config& c, const std::string& pfx,
                 int nl, int ms, int pt) {
    c.n_layers = nl; c.hidden = 2048; c.n_heads = 16; c.n_kv_heads = 8;
    c.head_dim = 128; c.ffn_inner = 6144; c.vocab = 151936;
    c.rope_theta = 1.0e6f; c.rms_eps = 1e-6f; c.rotary_dim = 128;
    c.full_attn_interval = 1; c.tie_embeddings = false; c.use_bf16 = true;
    c.quant_bits = 8; c.dense = true; c.attn_output_gate = false;
    c.zero_centered_norm = false;   // plain Qwen3 std RMSNorm (dense path)
    c.backbone_only = true; c.weight_prefix = pfx; c.model_seg = "model.";
    c.max_seq = ms; c.page_tokens = pt;
  };
  set(cfg.backbone, "language_model.", 28, 2048, 256);
  set(cfg.local, "local_transformer.", 4, 32, 32);

  auto m = genai::MetalMossRtModel::load(dir, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  auto codec = genai::MetalMossCodec::load(cdir, mc, false, false);
  ASSERT_TRUE(codec != nullptr && codec->valid());
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(dir) + "/tokenizer.json", nullptr);
  ASSERT_TRUE(tok != nullptr);

  const char* te = env_("VPIPE_MOSS_RT_TEXT");
  const std::string text = (te && *te)
      ? std::string(te) : std::string("Hello, this is a test.");
  genai::MossRtPromptIds pids;
  auto prompt = genai::moss_rt_build_prompt_grid(*tok, pids);
  std::vector<std::int32_t> text_ids = tok->encode(text);

  genai::MossSampling sp;
  sp.temperature = 0.8f; sp.top_k = 30; sp.top_p = 0.6f;
  sp.repetition_penalty = 1.1f;
  const auto g0 = std::chrono::steady_clock::now();
  auto frames = m->generate(prompt, text_ids, 400, sp, /*seed=*/12345);
  const double gms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - g0).count();
  std::fprintf(stderr, "[moss-rt] synth: %d text ids -> %d frames | "
               "generate %.0f ms = %.1f ms/frame\n",
               (int)text_ids.size(), (int)frames.size(), gms,
               gms / (double)(frames.empty() ? 1 : frames.size()));
  ASSERT_TRUE(!frames.empty());

  std::vector<std::vector<std::int32_t>> codes;
  codes.reserve(frames.size());
  for (auto& fr : frames) { codes.emplace_back(fr.begin(), fr.end()); }
  std::vector<float> wav = codec->decode(codes, nullptr, m->config().n_vq);
  ASSERT_TRUE(!wav.empty());
  double e = 0.0;
  for (float s : wav) { e += (double)s * s; }
  const double rms = std::sqrt(e / (double)wav.size());
  std::fprintf(stderr, "[moss-rt] synth: %d frames -> %d samples (%.2fs) "
               "rms=%.5f\n", (int)frames.size(), (int)wav.size(),
               wav.size() / 24000.0, rms);
  EXPECT_TRUE(rms > 1e-4);   // non-silent

  // Optional: write a 16-bit PCM WAV (VPIPE_MOSS_RT_WAV) to audition.
  const char* wp = env_("VPIPE_MOSS_RT_WAV");
  if (wp != nullptr && *wp != '\0') {
    std::FILE* wf = std::fopen(wp, "wb");
    if (wf != nullptr) {
      const std::uint32_t sr = 24000, n = (std::uint32_t)wav.size();
      const std::uint32_t bytes = n * 2, chunk = 36 + bytes;
      auto w32 = [&](std::uint32_t v) { std::fwrite(&v, 4, 1, wf); };
      auto w16 = [&](std::uint16_t v) { std::fwrite(&v, 2, 1, wf); };
      std::fwrite("RIFF", 1, 4, wf); w32(chunk); std::fwrite("WAVE", 1, 4, wf);
      std::fwrite("fmt ", 1, 4, wf); w32(16); w16(1); w16(1);
      w32(sr); w32(sr * 2); w16(2); w16(16);
      std::fwrite("data", 1, 4, wf); w32(bytes);
      for (float s : wav) {
        float c = s < -1.0f ? -1.0f : (s > 1.0f ? 1.0f : s);
        w16((std::uint16_t)(std::int16_t)std::lround(c * 32767.0f));
      }
      std::fclose(wf);
      std::fprintf(stderr, "[moss-rt] wrote WAV %s\n", wp);
    }
  }
}

// (d) Voice cloning: synth a reference clip, codec-encode it (first 16
// codebooks), build the clone prompt, and generate a second utterance
// conditioned on that timbre. Checks the clone grid spliced the reference
// frames and the cloned synthesis is non-silent.
TEST(moss_rt_driver, voice_clone)
{
  const char* dir = env_("VPIPE_MOSS_RT_MODEL");
  const char* cdir = env_("VPIPE_MOSS_CODEC_MODEL");
  if (dir == nullptr || *dir == '\0' || cdir == nullptr || *cdir == '\0') {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  genai::MetalMossRtModel::Config cfg;
  auto set = [&](genai::MetalQwenModel::Config& c, const std::string& pfx,
                 int nl, int ms, int pt) {
    c.n_layers = nl; c.hidden = 2048; c.n_heads = 16; c.n_kv_heads = 8;
    c.head_dim = 128; c.ffn_inner = 6144; c.vocab = 151936;
    c.rope_theta = 1.0e6f; c.rms_eps = 1e-6f; c.rotary_dim = 128;
    c.full_attn_interval = 1; c.tie_embeddings = false; c.use_bf16 = true;
    c.quant_bits = 8; c.dense = true; c.attn_output_gate = false;
    c.zero_centered_norm = false;   // plain Qwen3 std RMSNorm (dense path)
    c.backbone_only = true; c.weight_prefix = pfx; c.model_seg = "model.";
    c.max_seq = ms; c.page_tokens = pt;
  };
  set(cfg.backbone, "language_model.", 28, 2048, 256);
  set(cfg.local, "local_transformer.", 4, 32, 32);

  auto m = genai::MetalMossRtModel::load(dir, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  auto codec = genai::MetalMossCodec::load(cdir, mc, false, /*with_encoder=*/true);
  ASSERT_TRUE(codec != nullptr && codec->valid() && codec->has_encoder());
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(dir) + "/tokenizer.json", nullptr);
  ASSERT_TRUE(tok != nullptr);

  genai::MossRtPromptIds pids;
  const int nvq = m->config().n_vq;
  genai::MossSampling sp;
  sp.temperature = 0.8f; sp.top_k = 30; sp.top_p = 0.6f;
  sp.repetition_penalty = 1.1f;

  // 1. Reference clip: an external 24 kHz mono f32 raw file
  // (VPIPE_MOSS_RT_REF_F32) when set, else self-generate one.
  std::vector<float> ref_wav;
  const char* rf = env_("VPIPE_MOSS_RT_REF_F32");
  if (rf != nullptr && *rf != '\0') {
    std::FILE* rff = std::fopen(rf, "rb");
    ASSERT_TRUE(rff != nullptr);
    std::fseek(rff, 0, SEEK_END);
    const long nb = std::ftell(rff);
    std::fseek(rff, 0, SEEK_SET);
    ref_wav.resize((std::size_t)(nb / 4));
    std::fread(ref_wav.data(), 4, ref_wav.size(), rff);
    std::fclose(rff);
    std::fprintf(stderr, "[moss-rt] clone: external ref %s (%.2fs)\n", rf,
                 ref_wav.size() / 24000.0);
  } else {
    auto ref_prompt = genai::moss_rt_build_prompt_grid(*tok, pids);
    auto ref_ids = tok->encode("This is my reference speaking voice.");
    auto ref_frames = m->generate(ref_prompt, ref_ids, 200, sp, /*seed=*/7);
    ASSERT_TRUE(!ref_frames.empty());
    std::vector<std::vector<std::int32_t>> ref_codes_full;
    for (auto& fr : ref_frames) {
      ref_codes_full.emplace_back(fr.begin(), fr.end());
    }
    ref_wav = codec->decode(ref_codes_full, nullptr, nvq);
  }
  ASSERT_TRUE(!ref_wav.empty());

  // 2. Encode the reference wav -> codes, keep the first nvq codebooks.
  auto enc = codec->encode(ref_wav);              // [Tref][32]
  ASSERT_TRUE(!enc.empty());
  std::vector<std::vector<std::int32_t>> ref_codes;
  for (auto& fr : enc) {
    const int mm = std::min<int>(nvq, (int)fr.size());
    ref_codes.emplace_back(fr.begin(), fr.begin() + mm);
  }

  // 3. Clone grid: must splice one ref_audio_pad(151654) row per ref frame.
  auto clone_grid = genai::moss_rt_build_clone_grid(*tok, ref_codes, pids);
  int ref_rows = 0;
  for (const auto& r : clone_grid) {
    if (r[0] == pids.ref_audio_pad) { ++ref_rows; }
  }
  std::fprintf(stderr, "[moss-rt] clone: %d ref frames -> %d spliced rows "
               "(grid %d)\n", (int)ref_codes.size(), ref_rows,
               (int)clone_grid.size());
  EXPECT_TRUE(ref_rows == (int)ref_codes.size());
  EXPECT_TRUE(ref_rows > 0);

  // 4. Cloned synthesis.
  auto text_ids = tok->encode("Hello, cloned voice test.");
  auto frames = m->generate(clone_grid, text_ids, 300, sp, /*seed=*/99);
  ASSERT_TRUE(!frames.empty());
  std::vector<std::vector<std::int32_t>> codes;
  for (auto& fr : frames) { codes.emplace_back(fr.begin(), fr.end()); }
  std::vector<float> wav = codec->decode(codes, nullptr, nvq);
  ASSERT_TRUE(!wav.empty());
  double ee = 0.0;
  for (float s : wav) { ee += (double)s * s; }
  const double rms = std::sqrt(ee / (double)wav.size());
  std::fprintf(stderr, "[moss-rt] clone synth: %d frames -> %.2fs rms=%.5f\n",
               (int)frames.size(), wav.size() / 24000.0, rms);
  EXPECT_TRUE(rms > 1e-4);

  const char* wp = env_("VPIPE_MOSS_RT_WAV");
  if (wp != nullptr && *wp != '\0') {
    std::FILE* wf = std::fopen(wp, "wb");
    if (wf != nullptr) {
      const std::uint32_t srr = 24000, nn = (std::uint32_t)wav.size();
      const std::uint32_t bytes = nn * 2, chunk = 36 + bytes;
      auto w32 = [&](std::uint32_t v) { std::fwrite(&v, 4, 1, wf); };
      auto w16 = [&](std::uint16_t v) { std::fwrite(&v, 2, 1, wf); };
      std::fwrite("RIFF", 1, 4, wf); w32(chunk); std::fwrite("WAVE", 1, 4, wf);
      std::fwrite("fmt ", 1, 4, wf); w32(16); w16(1); w16(1);
      w32(srr); w32(srr * 2); w16(2); w16(16);
      std::fwrite("data", 1, 4, wf); w32(bytes);
      for (float s : wav) {
        float c = s < -1.0f ? -1.0f : (s > 1.0f ? 1.0f : s);
        w16((std::uint16_t)(std::int16_t)std::lround(c * 32767.0f));
      }
      std::fclose(wf);
      std::fprintf(stderr, "[moss-rt] wrote clone WAV %s\n", wp);
    }
  }
}

// (e) Unquantized bf16 inference: load the ORIGINAL bf16 checkpoint (no
// model-quantize) -- MetalQwenModel auto-detects the dense raw path. Verifies
// the backbone hidden vs the golden (should BEAT the 8-bit rel-L2, no quant
// error) + a short end-to-end synth. Env VPIPE_MOSS_RT_RAW = the bf16 dir.
TEST(moss_rt_driver, unquantized)
{
  const char* dir = env_("VPIPE_MOSS_RT_RAW");
  const char* gp  = env_("VPIPE_MOSS_RT_GOLDEN");
  const char* cdir = env_("VPIPE_MOSS_CODEC_MODEL");
  if (dir == nullptr || *dir == '\0' || gp == nullptr || *gp == '\0') {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  std::FILE* f = std::fopen(gp, "rb");
  if (f == nullptr) { return; }
  auto rd_i = [&]() { std::int32_t v = 0; std::fread(&v, 4, 1, f); return v; };
  const int seq = rd_i(), nvq1 = rd_i();
  std::vector<std::vector<std::int32_t>> grid((std::size_t)seq);
  for (int r = 0; r < seq; ++r) {
    grid[(std::size_t)r].resize((std::size_t)nvq1);
    std::fread(grid[(std::size_t)r].data(), 4, (std::size_t)nvq1, f);
  }
  const int H = rd_i();
  std::vector<float> gold_h((std::size_t)H);
  std::fread(gold_h.data(), 4, (std::size_t)H, f);
  std::vector<std::int32_t> gold_args(16);
  std::fread(gold_args.data(), 4, 16, f);
  std::fclose(f);

  genai::MetalMossRtModel::Config cfg;
  fill_cfg_(cfg);
  // Load the UNQUANTIZED bf16 dir: MetalQwenModel sees a `.weight` with no
  // `.scales` and takes the dense raw path (quant_bits is ignored).
  auto m = genai::MetalMossRtModel::load(dir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  auto* bb = m->backbone();
  auto* cm = bb->context_manager();
  const genai::ContextId cid = cm->acquire_root();
  metal_compute::SharedBuffer emb = m->assemble_embeds(grid, 0, seq);
  metal_compute::SharedBuffer h = bb->forward_embeddings_hidden(cid, emb, seq);
  ASSERT_TRUE(!h.empty());
  const auto* hb = static_cast<const std::uint16_t*>(h.contents());
  double num = 0.0, den = 0.0;
  for (int i = 0; i < H; ++i) {
    const float mv = bf16_to_f32_(hb[i]);
    num += (double)(mv - gold_h[(std::size_t)i]) * (mv - gold_h[(std::size_t)i]);
    den += (double)gold_h[(std::size_t)i] * gold_h[(std::size_t)i];
  }
  const double rel = std::sqrt(num / (den + 1e-12));
  std::fprintf(stderr, "[moss-rt] UNQUANTIZED backbone hidden rel-L2 = %.4f\n",
               rel);
  EXPECT_TRUE(rel < 0.03);   // bf16 vs the fp32 golden (no quant error)

  genai::MossSampling greedy;
  std::uint64_t rng = 1;
  std::vector<int> codes = m->local_frame(h, greedy, rng);
  cm->release(cid);
  ASSERT_TRUE((int)codes.size() == 16);
  int lead = 0;
  for (int i = 0; i < 16 && codes[(std::size_t)i] == gold_args[(std::size_t)i];
       ++i) { lead = i + 1; }
  std::fprintf(stderr, "[moss-rt] UNQUANTIZED local frame0 leading-match=%d/16\n",
               lead);
  EXPECT_TRUE(lead >= 3);   // f16 local reproduces the fp32 golden's lead codes

  // Short end-to-end synth (proves the raw path drives generation).
  if (cdir != nullptr && *cdir != '\0') {
    auto codec = genai::MetalMossCodec::load(cdir, mc, false, false);
    ASSERT_TRUE(codec != nullptr && codec->valid());
    auto tok = genai::Tokenizer::from_huggingface_json(
        std::string(dir) + "/tokenizer.json", nullptr);
    ASSERT_TRUE(tok != nullptr);
    genai::MossRtPromptIds pids;
    auto prompt = genai::moss_rt_build_prompt_grid(*tok, pids);
    auto ids = tok->encode("Hello, unquantized voice.");
    genai::MossSampling sp;
    sp.temperature = 0.8f; sp.top_k = 30; sp.top_p = 0.6f;
    sp.repetition_penalty = 1.1f;
    auto frames = m->generate(prompt, ids, 300, sp, /*seed=*/5);
    std::fprintf(stderr, "[moss-rt] UNQUANTIZED synth: %d frames\n",
                 (int)frames.size());
    ASSERT_TRUE(!frames.empty());
    std::vector<std::vector<std::int32_t>> codes2;
    for (auto& fr : frames) { codes2.emplace_back(fr.begin(), fr.end()); }
    std::vector<float> wav = codec->decode(codes2, nullptr, m->config().n_vq);
    ASSERT_TRUE(!wav.empty());
  }
}

// (f) Profiler events: with profiling enabled, generate() emits ONE decode
// event PER FRAME (not one spanning the whole loop) + a prefill event, and the
// codec decode emits its audio-codec event. All land on the LLM aux lane, so
// its event count scales with the frame count -- the pre-fix one-per-loop bug
// gave a constant ~6 regardless of utterance length. Env VPIPE_MOSS_RT_MODEL +
// VPIPE_MOSS_CODEC_MODEL.
TEST(moss_rt_driver, perf_events)
{
  const char* dir = env_("VPIPE_MOSS_RT_MODEL");
  const char* cdir = env_("VPIPE_MOSS_CODEC_MODEL");
  if (dir == nullptr || *dir == '\0' || cdir == nullptr || *cdir == '\0') {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  ASSERT_TRUE(sess.enable_profiling(8192).code == 0);

  genai::MetalMossRtModel::Config cfg;
  fill_cfg_(cfg);
  auto m = genai::MetalMossRtModel::load(dir, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  auto codec = genai::MetalMossCodec::load(cdir, mc, false, false);
  ASSERT_TRUE(codec != nullptr && codec->valid());
  codec->set_session(&sess);   // route the codec's audio-codec events here
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(dir) + "/tokenizer.json", nullptr);
  ASSERT_TRUE(tok != nullptr);

  genai::MossRtPromptIds pids;
  auto prompt = genai::moss_rt_build_prompt_grid(*tok, pids);
  auto ids = tok->encode("Profiler event coverage test sentence here.");
  genai::MossSampling sp;
  sp.temperature = 0.8f; sp.top_k = 30; sp.top_p = 0.6f;
  sp.repetition_penalty = 1.1f;
  auto frames = m->generate(prompt, ids, 300, sp, /*seed=*/321);
  const int F = (int)frames.size();
  ASSERT_TRUE(F > 4);
  std::vector<std::vector<std::int32_t>> codes;
  for (auto& fr : frames) { codes.emplace_back(fr.begin(), fr.end()); }
  auto wav = codec->decode(codes, nullptr, m->config().n_vq);
  ASSERT_TRUE(!wav.empty());

  namespace fs = std::filesystem;
  const std::string path =
      (fs::temp_directory_path() / "vpipe-moss-rt-prof.bin").string();
  ASSERT_TRUE(sess.dump_profiling(path).code == 0);
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss; ss << in.rdbuf();
  std::error_code ec; fs::remove(path, ec);
  FlexData root = FlexData::from_binary(ss.str());
  auto ro = root.as_object();
  std::uint64_t llm_events = 0;
  FlexData threads_v = ro.at("threads");
  auto threads = threads_v.as_array();
  for (std::size_t i = 0; i < threads.size(); ++i) {
    FlexData e_v = threads.at(i);
    auto e = e_v.as_object();
    if (e.contains("label") &&
        std::string(e.at("label").as_string("")) == "LLM") {
      llm_events = e.at("events_count").get_uint();
    }
  }
  std::fprintf(stderr, "[moss-rt] perf: %d frames -> %llu LLM-lane events\n",
               F, (unsigned long long)llm_events);
  // prefill(2) + decode(2/frame) + codec(2) = 2F+4. >= 2F proves per-frame
  // decode (the one-per-loop bug would give a constant ~6).
  EXPECT_TRUE(llm_events >= (std::uint64_t)(2 * F));
}
