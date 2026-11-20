// Fish Audio Drama 3 (fishaudio/drama-3) bring-up tests: the prompt's
// tokens, the speech LM teacher-forced and through its decode path, each
// against the reference's own forward (tools/dump_drama3_golden.py: the
// backbone is transformers' Qwen3.5-MoE run a layer at a time with the
// block-FP8 experts dequantized, the tower restated from vllm-omni /
// sglang-omni, both in fp32 and bf16).
//
// Env:
//   VPIPE_DRAMA3_TEST_MODEL_PATH  the fishaudio/drama-3 directory.
//   VPIPE_DRAMA3_GOLDEN           the golden directory.
//   VPIPE_QWEN3_ASR_TEST_MODEL_PATH  Qwen3-ASR, for stage_speech_transcribes.
//   VPIPE_DRAMA3_DUMP             optional: a dir the takes are written to.
// Every test needs the first two and skips without them. The model holds
// ~36 GB: run these on a 64 GB box.
//
// GEOMETRY. The golden's sequence is a 169-row voice-clone prompt (an 85-
// frame reference) and 140 teacher-forced frames + end of speech: 310 rows,
// so every backbone GEMM clears the 64-row matmul2d floor, the tower's
// attention runs the flash kernel (32-row tiles) on the prefill and the
// multi-simdgroup scan on the decode rows, and neither length is a
// multiple of any tile.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/ffmpeg-libraries.h"
#include "common/media-decode.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-transcribe-stage.h"
#include "stages/audio-video/audio-temporal-resample-stage.h"
#include "stages/text-to-speech-stage.h"
#include "generative-models/drama3/drama3-prompt.h"
#include "generative-models/drama3/metal-drama3-model.h"
#include "generative-models/drama3/metal-fish-codec.h"
#include "generative-models/moe-expert-cache.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
namespace fs = std::filesystem;

namespace {

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return (v != nullptr && *v != '\0') ? std::string(v) : std::string();
}

// Two loads of the model compared byte for byte must run the same kernels,
// and two tuners pick theirs by timing at every load: the decode attention
// (VPIPE_QWEN_GQA_AUTOTUNE; MEASURED to flip between sessions) and, at head
// dim 256, the prefill attention (VPIPE_QWEN_PREFILL_AUTOTUNE; MEASURED to
// vote qtile in 5 of 24 loads on the M4 Pro, 4 of them a split vote that
// went to the first candidate, each time moving the hidden rows 4% --
// the bf16 stack's chaos over another rounding). Pinned for the scope.
struct PinnedTuners {
  static constexpr const char* kVars[2] = {"VPIPE_QWEN_GQA_AUTOTUNE",
                                           "VPIPE_QWEN_PREFILL_AUTOTUNE"};
  std::string was[2];
  PinnedTuners()
  {
    for (int i = 0; i < 2; ++i) {
      was[i] = env_(kVars[i]);
      ::setenv(kVars[i], "0", 1);
    }
  }
  ~PinnedTuners()
  {
    for (int i = 0; i < 2; ++i) {
      if (was[i].empty()) {
        ::unsetenv(kVars[i]);
      } else {
        ::setenv(kVars[i], was[i].c_str(), 1);
      }
    }
  }
};

template <typename T>
std::vector<T>
read_raw_(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary);
  std::vector<T> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / sizeof(T));
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

FlexData
read_json_(const fs::path& p)
{
  std::ifstream in(p);
  if (!in) { return FlexData(); }
  return FlexData::from_json(in);
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Of the rows whose fp32 top-1 leads its runner-up by more than `margin`,
// how many `got` picks the same way.
void
decisive_(const float* got, const float* ref, int rows, int cols,
          float margin, int* agree, int* total)
{
  *agree = 0;
  *total = 0;
  for (int r = 0; r < rows; ++r) {
    const float* g = got + (std::size_t)r * cols;
    const float* e = ref + (std::size_t)r * cols;
    int b0 = 0;
    for (int i = 1; i < cols; ++i) { if (e[i] > e[b0]) { b0 = i; } }
    float second = -1e30f;
    for (int i = 0; i < cols; ++i) {
      if (i != b0) { second = std::max(second, e[i]); }
    }
    if (e[b0] - second <= margin) { continue; }
    ++*total;
    int g0 = 0;
    for (int i = 1; i < cols; ++i) { if (g[i] > g[g0]) { g0 = i; } }
    if (g0 == b0) { ++*agree; }
  }
}

struct Golden {
  int n = 0, n_prompt = 0;
  std::vector<std::int32_t> frames;
  std::vector<float> hid32, hid16, main32, main16, head32, head16;
  bool ok = false;
};

Golden
load_golden_(const fs::path& g)
{
  Golden d;
  const FlexData meta = read_json_(g / "meta.json");
  if (!meta.is_object()) { return d; }
  auto mo = meta.as_object();
  d.n = (int)mo.at("n_rows").as_int();
  d.n_prompt = (int)mo.at("n_prompt").as_int();
  d.frames = read_raw_<std::int32_t>(g / "frames.i32");
  d.hid32 = read_raw_<float>(g / "hidden_f32.f32");
  d.hid16 = read_raw_<float>(g / "hidden_bf16.f32");
  d.main32 = read_raw_<float>(g / "main_logits_f32.f32");
  d.main16 = read_raw_<float>(g / "main_logits_bf16.f32");
  d.head32 = read_raw_<float>(g / "head_logits_f32.f32");
  d.head16 = read_raw_<float>(g / "head_logits_bf16.f32");
  d.ok = d.n > 0 &&
         d.frames.size() == (std::size_t)d.n * MetalDrama3Model::kFrame &&
         !d.hid32.empty() && !d.main32.empty() && !d.head32.empty();
  return d;
}

// The golden's head logits are [n, 9, V] (row-major per row); the model's
// [9, n, V]. One head's [rows, V] block, rows [r0, r0 + rows), from each.
std::vector<float>
golden_head_(const std::vector<float>& g, int n, int head, int r0, int rows,
             int V)
{
  std::vector<float> out((std::size_t)rows * V);
  for (int r = 0; r < rows; ++r) {
    std::copy_n(g.data() + (((std::size_t)(r0 + r) * 9) + head) * V, V,
                out.data() + (std::size_t)r * V);
  }
  (void)n;
  return out;
}

}  // namespace

TEST(drama3, prompt_tokens_match_reference)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  auto tok = Tokenizer::from_huggingface_json(
      (fs::path(mp) / "tokenizer.json").string(), nullptr);
  ASSERT_TRUE(tok != nullptr);
  if (tok == nullptr) { return; }
  const FlexData meta = read_json_(fs::path(gp) / "meta.json");
  ASSERT_TRUE(meta.is_object());
  if (!meta.is_object()) { return; }
  const FlexData segs = meta.as_object().at("prompt_segments");
  int ok = 0, n = 0;
  for (const FlexData& s : segs.as_array()) {
    const FlexData fs_ = s;
    auto o = fs_.as_object();
    const std::string text(o.at("text").as_string());
    std::vector<std::int32_t> want;
    const FlexData ids = o.at("ids");
    for (const FlexData& v : ids.as_array()) {
      want.push_back((std::int32_t)v.as_int());
    }
    const std::vector<std::int32_t> got = drama3::encode_spliced(*tok, text);
    ++n;
    if (got == want) {
      ++ok;
    } else {
      std::printf("[drama3] segment %d differs (%zu vs %zu ids)\n", n,
                  got.size(), want.size());
    }
  }
  std::printf("[drama3] prompt segments: %d/%d token-exact\n", ok, n);
  EXPECT_TRUE(n > 0 && ok == n);
  EXPECT_TRUE(tok->special_token_id("<|semantic:0|>") == 248099);
  EXPECT_TRUE(tok->special_token_id("<|speaker:0|>") == 248083);
}

// Long text: split_text / split_by_length / parse_speaker_segments against
// fish-speech's own functions over the cases tools/dump_drama3_chunks.py
// writes -- tags, cues, paired tags (open, nested, dashed), decimals, CJK
// and Korean marks, emoji, combining marks, many speakers -- each at three
// limits.
TEST(drama3, chunking_matches_reference)
{
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  if (gp.empty()) { return; }
  const FlexData cases = read_json_(fs::path(gp) / "chunks.json");
  ASSERT_TRUE(cases.is_array());
  if (!cases.is_array()) { return; }
  auto strings = [](const FlexData& a) {
    std::vector<std::string> v;
    for (const FlexData& x : a.as_array()) {
      v.emplace_back(x.as_string());
    }
    return v;
  };
  int n = 0, ok = 0, multi = 0;
  for (const FlexData& c : cases.as_array()) {
    const FlexData cc = c;
    auto o = cc.as_object();
    const std::string text(o.at("text").as_string());
    const int mb = (int)o.at("max_bytes").as_int();
    const int ms = (int)o.at("max_speakers").as_int();
    const auto want = strings(o.at("chunks"));
    const auto want_len = strings(o.at("by_length"));
    std::vector<std::pair<std::string, std::string>> want_seg;
    const FlexData segs = o.at("segments");
    for (const FlexData& sg : segs.as_array()) {
      const FlexData sgg = sg;
      auto a = sgg.as_array();
      want_seg.push_back({std::string(a[0].as_string()),
                          std::string(a[1].as_string())});
    }
    const auto got = drama3::split_text(text, mb, ms);
    const auto got_len = drama3::split_by_length(text, mb);
    const auto got_seg = drama3::parse_speaker_segments(text);
    ++n;
    if (want.size() > 1) { ++multi; }
    if (got == want && got_len == want_len && got_seg == want_seg) {
      ++ok;
      continue;
    }
    std::printf("[drama3] case %d (%d bytes, %d speakers) differs: chunks "
                "%zu/%zu%s, by_length %zu/%zu%s, segments %zu/%zu%s\n", n,
                mb, ms, got.size(), want.size(), got == want ? "" : " !",
                got_len.size(), want_len.size(),
                got_len == want_len ? "" : " !", got_seg.size(),
                want_seg.size(), got_seg == want_seg ? "" : " !");
    for (std::size_t k = 0; k < std::max(got.size(), want.size()); ++k) {
      const std::string a = k < got.size() ? got[k] : "<none>";
      const std::string b = k < want.size() ? want[k] : "<none>";
      if (a != b) {
        std::printf("  chunk %zu\n    got  [%s]\n    want [%s]\n", k,
                    a.c_str(), b.c_str());
        break;
      }
    }
  }
  std::printf("[drama3] chunking: %d/%d cases exact (%d split in more "
              "than one chunk)\n", ok, n, multi);
  EXPECT_TRUE(n > 0 && ok == n && multi > n / 2);
}

// The prompt with two references (speakers 0 and 1) and two earlier takes
// as finished turns, row for row against fish-speech's build_drama3_prompt.
TEST(drama3, history_prompt_matches_reference)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  auto tok = Tokenizer::from_huggingface_json(
      (fs::path(mp) / "tokenizer.json").string(), nullptr);
  ASSERT_TRUE(tok != nullptr);
  const FlexData h = read_json_(fs::path(gp) / "history.json");
  ASSERT_TRUE(h.is_object());
  if (tok == nullptr || !h.is_object()) { return; }
  auto o = h.as_object();
  auto takes = [](const FlexData& a) {
    std::vector<drama3::Take> v;
    for (const FlexData& x : a.as_array()) {
      const FlexData xx = x;
      auto t = xx.as_object();
      drama3::Take take;
      take.transcript = std::string(t.at("text").as_string());
      const FlexData cs = t.at("codes");
      for (const FlexData& c : cs.as_array()) {
        take.codes.push_back((std::int32_t)c.as_int());
      }
      v.push_back(std::move(take));
    }
    return v;
  };
  const std::vector<drama3::Reference> refs = takes(o.at("references"));
  const std::vector<drama3::Take> hist = takes(o.at("history"));
  std::vector<const drama3::Take*> hp;
  for (const auto& t : hist) { hp.push_back(&t); }
  std::vector<std::int32_t> want;
  const FlexData rows = o.at("rows");
  for (const FlexData& r : rows.as_array()) {
    const FlexData rr = r;
    for (const FlexData& v : rr.as_array()) {
      want.push_back((std::int32_t)v.as_int());
    }
  }
  std::string err;
  const auto got = drama3::build_prompt(
      *tok, std::string(o.at("text").as_string()), refs, hp, 248099, 4096,
      &err);
  if (!err.empty()) { std::printf("[drama3] prompt: %s\n", err.c_str()); }
  std::size_t first = 0;
  while (first < std::min(got.size(), want.size()) &&
         got[first] == want[first]) {
    ++first;
  }
  std::printf("[drama3] history prompt: %zu rows (reference %zu), first "
              "difference at row %zu\n", got.size() / 11, want.size() / 11,
              first / 11);
  EXPECT_TRUE(!want.empty() && got == want);
  // Past the row bound the prompt is refused AS too long.
  drama3::Take big;
  big.transcript = "A long take.";
  big.codes.assign((std::size_t)drama3::kMaxPromptRows * 10, 1);
  bool too_long = false;
  const auto none = drama3::build_prompt(*tok, "Short.", {}, {&big}, 248099,
                                         4096, &err, &too_long);
  EXPECT_TRUE(none.empty() && too_long);
}

// The history policy (Drama3History): the first take pinned, the latest
// kept, candidates longest first and ending with none.
TEST(drama3, history_pins_the_first_take)
{
  drama3::History h;
  EXPECT_TRUE(h.candidates().size() == 1 && h.candidates()[0].empty());
  for (int i = 0; i < 3; ++i) {
    drama3::Take t;
    t.transcript = "chunk " + std::to_string(i);
    t.codes.assign(10, i);
    h.push(std::move(t));
  }
  const auto c = h.candidates();
  ASSERT_TRUE(c.size() == 3);
  if (c.size() != 3) { return; }
  EXPECT_TRUE(c[0].size() == 2 && c[0][0]->transcript == "chunk 0" &&
              c[0][1]->transcript == "chunk 2");
  EXPECT_TRUE(c[1].size() == 1 && c[1][0]->transcript == "chunk 0");
  EXPECT_TRUE(c[2].empty());
}

// Teacher-forced over the golden's whole sequence, then the DECODE path
// over its frames, each held to the reference's own bf16 error against its
// fp32 run (x1.5), with every fp32-decisive pick agreeing.
namespace {

// The body of lm_matches_reference, for a checkpoint and the golden the
// reference wrote from IT (the FP8 release, or the int4 one).
// The checks that failed come back (minitest's macros live on the test).
#define D3_CHECK_(c)                                                        \
  do {                                                                      \
    if (!(c)) { fails.push_back(std::string(#c) + " (line " +             \
                                std::to_string(__LINE__) + ")"); }        \
  } while (0)
std::vector<std::string>
lm_matches_(const std::string& mp, const std::string& gp)
{
  std::vector<std::string> fails;
  // The heads read the backbone's hidden rows, and the bf16 backbone is
  // CHAOTIC at this depth: the prefill's two expert routes, each within
  // 2e-5 of f64 per GEMM (fp8_grouped_gemm_matches_gather), put the
  // hidden 0.0425 and 0.0533 from fp32 -- row by row now one ahead, now
  // the other, 0.046 apart. So the heads, which carry that realization
  // into all nine, are held at 2x the reference's own bf16 error (a broken
  // head errs O(1)); the main logits and hidden at 1.5x.
  constexpr double kHeadBar = 2.0;
  const Golden g = load_golden_(gp);
  D3_CHECK_(g.ok);
  if (!g.ok) { return fails; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return fails; }
  Drama3Config cfg;
  std::string err;
  D3_CHECK_(Drama3Config::from_dir(mp, &cfg, &err));
  if (!err.empty()) { std::printf("[drama3] config: %s\n", err.c_str()); }
  MetalDrama3Model::LoadOptions lo;
  lo.max_rows = 4096;
  auto m = MetalDrama3Model::load(WeightSet::open(mp, nullptr), mc, cfg, lo,
                                  &err);
  if (m == nullptr) { std::printf("[drama3] load: %s\n", err.c_str()); }
  D3_CHECK_(m != nullptr);
  if (m == nullptr) { return fails; }
  std::printf("[drama3] loaded: %zu MB held\n", m->resident_bytes() >> 20);
  const int N = g.n, P = g.n_prompt, H = cfg.backbone.hidden;
  const int NC = cfg.main_candidates(), V = cfg.tower.codebook_size;

  std::vector<float> hid, ml, hl;
  D3_CHECK_(m->teacher_forced(g.frames, N, &hid, &ml, &hl, &err));
  if (!err.empty()) { std::printf("[drama3] %s\n", err.c_str()); }
  if (hid.size() != (std::size_t)N * H) { return fails; }
  // The audio rows from the prompt's last on are what generation reads.
  const std::size_t r0 = (std::size_t)P - 1, rows = (std::size_t)N - r0;
  {
    const double ours = rel_l2_(hid.data(), g.hid32.data(), hid.size());
    const double ref = rel_l2_(g.hid16.data(), g.hid32.data(), hid.size());
    std::printf("[drama3] backbone hidden: vs fp32 %.4g (the reference's "
                "bf16 %.4g)\n", ours, ref);
    D3_CHECK_(ours < 1.5 * ref);
  }
  {
    const float* o = ml.data() + r0 * NC;
    const float* e = g.main32.data() + r0 * NC;
    const float* b = g.main16.data() + r0 * NC;
    const double ours = rel_l2_(o, e, rows * NC);
    const double ref = rel_l2_(b, e, rows * NC);
    int agree = 0, total = 0;
    decisive_(o, e, (int)rows, NC, 0.5f, &agree, &total);
    std::printf("[drama3] main logits: vs fp32 %.4g (the reference's bf16 "
                "%.4g), decisive picks %d/%d\n", ours, ref, agree, total);
    D3_CHECK_(ours < 1.5 * ref);
    D3_CHECK_(total > 0 && agree == total);
  }
  for (int j = 0; j < 9; ++j) {
    const std::vector<float> e =
        golden_head_(g.head32, N, j, (int)r0, (int)rows, V);
    const std::vector<float> b =
        golden_head_(g.head16, N, j, (int)r0, (int)rows, V);
    const float* o = hl.data() + ((std::size_t)j * N + r0) * V;
    const double ours = rel_l2_(o, e.data(), rows * V);
    const double ref = rel_l2_(b.data(), e.data(), rows * V);
    int agree = 0, total = 0;
    decisive_(o, e.data(), (int)rows, V, 0.5f, &agree, &total);
    std::printf("[drama3] head %d (code %d): vs fp32 %.4g (bf16 %.4g), "
                "decisive %d/%d\n", j, j + 1, ours, ref, agree, total);
    D3_CHECK_(ours < kHeadBar * ref);
    D3_CHECK_(agree == total);
  }

  // The DECODE path: the prompt prefilled, then every frame one row at a
  // time -- the backbone's decode step (GEMV routes, the FP8 gathers at one
  // row) and the one-row tower (the scan attention over the whole K/V).
  std::vector<float> dml, dhl, dhid;
  D3_CHECK_(m->teacher_decode(g.frames, N, P, &dml, &dhl, &err, &dhid));
  if (!err.empty()) { std::printf("[drama3] %s\n", err.c_str()); }
  if (dhid.size() == rows * H) {
    const double ours = rel_l2_(dhid.data(), g.hid32.data() + r0 * H,
                                rows * H);
    const double ref = rel_l2_(g.hid16.data() + r0 * H,
                               g.hid32.data() + r0 * H, rows * H);
    std::printf("[drama3] decode hidden: vs fp32 %.4g (bf16 %.4g)\n", ours,
                ref);
    D3_CHECK_(ours < 1.5 * ref);
  }
  if (dml.size() != rows * NC || dhl.size() != (std::size_t)9 * rows * V) {
    return fails;
  }
  {
    const float* e = g.main32.data() + r0 * NC;
    const float* b = g.main16.data() + r0 * NC;
    const double ours = rel_l2_(dml.data(), e, rows * NC);
    const double ref = rel_l2_(b, e, rows * NC);
    int agree = 0, total = 0;
    decisive_(dml.data(), e, (int)rows, NC, 0.5f, &agree, &total);
    std::printf("[drama3] decode main logits: vs fp32 %.4g (bf16 %.4g), "
                "decisive %d/%d\n", ours, ref, agree, total);
    D3_CHECK_(ours < 1.5 * ref);
    D3_CHECK_(total > 0 && agree == total);
  }
  for (int j = 0; j < 9; ++j) {
    const std::vector<float> e =
        golden_head_(g.head32, N, j, (int)r0, (int)rows, V);
    const std::vector<float> b =
        golden_head_(g.head16, N, j, (int)r0, (int)rows, V);
    const float* o = dhl.data() + (std::size_t)j * rows * V;
    const double ours = rel_l2_(o, e.data(), rows * V);
    const double ref = rel_l2_(b.data(), e.data(), rows * V);
    int agree = 0, total = 0;
    decisive_(o, e.data(), (int)rows, V, 0.5f, &agree, &total);
    std::printf("[drama3] decode head %d: vs fp32 %.4g (bf16 %.4g), "
                "decisive %d/%d\n", j, ours, ref, agree, total);
    D3_CHECK_(ours < kHeadBar * ref);
    D3_CHECK_(agree == total);
  }
  return fails;
}
#undef D3_CHECK_

}  // namespace

TEST(drama3, lm_matches_reference)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  const std::vector<std::string> f = lm_matches_(mp, gp);
  for (const std::string& x : f) {
    std::printf("[drama3] FAILED: %s\n", x.c_str());
  }
  EXPECT_TRUE(f.empty());
}

// fishaudio/drama-3-int4: its GPTQ experts converted to the affine w4 g64
// slabs (exact but for the scales' rounding to bf16) and its GPTQ dense
// projections dequantized to bf16, against the reference's fp32 forward of
// the SAME int4 weights (tools/dump_drama3_golden.py --model <int4 dir>),
// held to that reference's own bf16 error as the FP8 release is.
TEST(drama3, int4_matches_reference)
{
  const std::string mp = env_("VPIPE_DRAMA3_INT4_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_INT4_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  const std::vector<std::string> f = lm_matches_(mp, gp);
  for (const std::string& x : f) {
    std::printf("[drama3] FAILED: %s\n", x.c_str());
  }
  EXPECT_TRUE(f.empty());
}

// A sampled run from the golden's voice-clone prompt: it must end in end
// of speech, inside the length the text implies. The frames are written
// to VPIPE_DRAMA3_DUMP (when set) for a listen through the reference
// codec. Speed is reported, not held.
TEST(drama3, generates_a_take)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  const Golden g = load_golden_(gp);
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Drama3Config cfg;
  std::string err;
  ASSERT_TRUE(Drama3Config::from_dir(mp, &cfg, &err));
  MetalDrama3Model::LoadOptions lo;
  lo.max_rows = 4096;
  auto m = MetalDrama3Model::load(WeightSet::open(mp, nullptr), mc, cfg, lo,
                                  &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  std::vector<std::int32_t> prompt(g.frames.begin(),
                                   g.frames.begin() +
                                       (std::size_t)g.n_prompt * 11);
  MetalDrama3Model::Sampling sp;
  sp.seed = 1234;
  sp.max_frames = 600;
  std::vector<MetalDrama3Model::Frame> frames;
  const auto t0 = std::chrono::steady_clock::now();
  double t_first = 0.0;
  ASSERT_TRUE(m->generate(prompt, g.n_prompt, sp,
                          [&](const MetalDrama3Model::Frame&, int i) {
                            if (i == 0) {
                              t_first = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
                            }
                            return true;
                          },
                          &frames, &err));
  const double secs = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  const double audio = frames.size() * 2048.0 / 44100.0;
  std::printf("[drama3] seed %u: %zu frames (%.2f s of audio) in %.2f s, "
              "first frame %.0f ms, %.1f frames/s (RTF %.2f)\n", sp.seed,
              frames.size(), audio, secs, t_first * 1e3,
              (frames.size() - 1) / std::max(secs - t_first, 1e-6),
              secs / std::max(audio, 1e-6));
  // The text is ~6 s read aloud: an end of speech well short of the cap.
  EXPECT_TRUE(frames.size() > 40 && frames.size() < 600);
  bool codes_ok = true;
  for (const auto& f : frames) {
    for (int k = 0; k < 10; ++k) {
      codes_ok = codes_ok && f[(std::size_t)k] >= 0 &&
                 f[(std::size_t)k] < (k == 0 ? 4096 : 4096);
    }
  }
  EXPECT_TRUE(codes_ok);
  const std::string dump = env_("VPIPE_DRAMA3_DUMP");
  if (!dump.empty()) {
    std::ofstream out(dump, std::ios::binary);
    for (const auto& f : frames) {
      out.write(reinterpret_cast<const char*>(f.data()), 40);
    }
  }
}

// ---- the codec -----------------------------------------------------------

namespace {

// The golden's channel-major [C, T] -> channel-last [T, C].
std::vector<float>
to_tc_(const std::vector<float>& ct, int C)
{
  const std::size_t T = ct.size() / (std::size_t)C;
  std::vector<float> out(ct.size());
  for (std::size_t c = 0; c < (std::size_t)C; ++c) {
    for (std::size_t t = 0; t < T; ++t) { out[t * C + c] = ct[c * T + t]; }
  }
  return out;
}

double
snr_db_(const float* got, const float* ref, std::size_t n)
{
  double s = 0.0, e = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    s += (double)ref[i] * ref[i];
    const double d = (double)got[i] - ref[i];
    e += d * d;
  }
  return 10.0 * std::log10(s / std::max(e, 1e-30));
}

}  // namespace

// The codec's decode of the reference clip's codes, stage by stage, and its
// encode of the clip, against fish-speech's modded DAC in fp32. Decode is
// held per stage to an f16 bar.
//
// THE ENCODE IS AS SENSITIVE AS A RESIDUAL VQ GETS, so it is held on its
// semantic codes only. MEASURED (the golden's clip): the reference itself,
// fed the clip rounded to f16 and nothing else, keeps 84 / 83 / 82 / 80 ..
// 67 of 85 codes per codebook and decodes 16.0 dB from its fp32 codes;
// 1e-5 of input noise (-100 dB) keeps 76 / 70 / 60 .. 31. This encoder,
// f16 throughout, keeps 84 / 83 / 82 / 79 .. 66 and decodes 16.1 dB from
// them -- the reference's own figure at f16 input. (An f16-rounded run of
// the reference reproduces this encoder's per-block error to two digits:
// it is the precision, not a defect.)
TEST(drama3, codec_matches_reference)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  const fs::path g(gp);
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto codec = MetalFishCodec::load(
      WeightSet::open((fs::path(mp) / "codec.pth").string(), nullptr), mc,
      /*with_encoder=*/true, &err);
  if (codec == nullptr) { std::printf("[drama3] codec: %s\n", err.c_str()); }
  ASSERT_TRUE(codec != nullptr);
  if (codec == nullptr) { return; }
  std::printf("[drama3] codec: %zu MB held\n", codec->resident_bytes() >> 20);
  // [10, T] reference codes -> frame-major.
  const std::vector<std::int32_t> rc = read_raw_<std::int32_t>(
      g / "ref_codes.i32");
  ASSERT_TRUE(!rc.empty() && rc.size() % 10 == 0);
  if (rc.empty()) { return; }
  const int T = (int)(rc.size() / 10);
  std::vector<std::int32_t> codes((std::size_t)T * 10);
  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < 10; ++k) {
      codes[(std::size_t)t * 10 + k] = rc[(std::size_t)k * T + t];
    }
  }
  std::vector<float> vq, post, up;
  ASSERT_TRUE(codec->decode_stages(codes.data(), T, &vq, &post, &up, &err));
  struct S { const char* name; std::vector<float>* got; double bar; };
  for (const S& st : {S{"dec_vq", &vq, 1e-3}, S{"dec_post", &post, 2e-2},
                      S{"dec_up", &up, 2e-2}}) {
    const std::vector<float> want =
        to_tc_(read_raw_<float>(g / (std::string(st.name) + ".f32")), 1024);
    ASSERT_TRUE(want.size() == st.got->size());
    if (want.size() != st.got->size()) { return; }
    const double r = rel_l2_(st.got->data(), want.data(), want.size());
    std::printf("[drama3] codec %s: rel-L2 %.4g\n", st.name, r);
    EXPECT_TRUE(r < st.bar);
  }
  std::vector<float> pcm;
  ASSERT_TRUE(codec->decode(codes.data(), T, &pcm, &err));
  const std::vector<float> wav = read_raw_<float>(g / "dec_audio.f32");
  ASSERT_TRUE(pcm.size() == wav.size());
  if (pcm.size() == wav.size()) {
    const double snr = snr_db_(pcm.data(), wav.data(), wav.size());
    std::printf("[drama3] codec audio: %zu samples, SNR %.2f dB vs fp32\n",
                pcm.size(), snr);
    EXPECT_TRUE(snr > 30.0);
  }
  // Encode the 44.1 kHz clip the golden encoded.
  const std::vector<float> clip = read_raw_<float>(g / "ref_pcm44k.f32");
  std::vector<std::int32_t> got;
  std::vector<float> pre, lat, down;
  ASSERT_TRUE(codec->encode(clip.data(), (int)clip.size(), &got, &err,
                            &pre, &lat, &down));
  for (const auto& [name, v] :
       {std::pair<const char*, const std::vector<float>*>{"enc_latent", &lat},
        {"enc_down", &down}}) {
    const std::vector<float> want =
        to_tc_(read_raw_<float>(g / (std::string(name) + ".f32")), 1024);
    if (want.size() == v->size()) {
      std::printf("[drama3] codec %s: rel-L2 %.4g\n", name,
                  rel_l2_(v->data(), want.data(), want.size()));
    }
  }
  {
    // The quantizer's input: what the codes are chosen from.
    const std::vector<float> want =
        to_tc_(read_raw_<float>(g / "enc_pre.f32"), 1024);
    if (want.size() == pre.size()) {
      const double r = rel_l2_(pre.data(), want.data(), want.size());
      std::printf("[drama3] codec encode, the quantizer's input: rel-L2 "
                  "%.4g\n", r);
      EXPECT_TRUE(r < 2e-2);
    }
  }
  if (!err.empty()) { std::printf("[drama3] encode: %s\n", err.c_str()); }
  ASSERT_TRUE(got.size() == codes.size());
  if (got.size() != codes.size()) { return; }
  int same[10] = {0};
  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < 10; ++k) {
      const std::size_t i = (std::size_t)t * 10 + k;
      same[k] += got[i] == codes[i];
    }
  }
  std::printf("[drama3] codec encode, codes agreeing per codebook:");
  for (int k = 0; k < 10; ++k) { std::printf(" %d/%d", same[k], T); }
  std::printf("\n");
  EXPECT_TRUE(same[0] >= (int)(0.95 * T));
  // What the disagreements are worth: our codes decoded against the
  // reference codes decoded. A code that flips between two near-equal
  // entries changes little of the sound.
  std::vector<float> ours;
  ASSERT_TRUE(codec->decode(got.data(), T, &ours, &err));
  if (ours.size() == wav.size()) {
    const double snr = snr_db_(ours.data(), wav.data(), wav.size());
    std::printf("[drama3] codec round trip of our codes vs the reference's "
                "codes: SNR %.2f dB\n", snr);
  }
}

// A streamed decode is the one-shot decode: the golden's two clips' codes
// back to back (225 frames, so the transformer's 128-frame window slides
// and the key history is shifted) in chunks of 1, 7 and 16 (ragged
// ends), the histories carried between them.
TEST(drama3, codec_stream_is_the_decode)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto codec = MetalFishCodec::load(
      WeightSet::open((fs::path(mp) / "codec.pth").string(), nullptr), mc,
      /*with_encoder=*/false, &err);
  ASSERT_TRUE(codec != nullptr);
  if (codec == nullptr) { return; }
  std::vector<std::int32_t> codes;
  for (const char* clip : {"ref_codes.i32", "target_codes.i32"}) {
    const std::vector<std::int32_t> rc =
        read_raw_<std::int32_t>(fs::path(gp) / clip);
    const int n = (int)(rc.size() / 10);
    for (int t = 0; t < n; ++t) {
      for (int k = 0; k < 10; ++k) {
        codes.push_back(rc[(std::size_t)k * n + t]);
      }
    }
  }
  const int T = (int)(codes.size() / 10);
  ASSERT_TRUE(T > 200);
  std::vector<float> whole;
  ASSERT_TRUE(codec->decode(codes.data(), T, &whole, &err));
  for (int chunk : {1, 7, 16}) {
    auto st = codec->decode_stream_begin(chunk);
    ASSERT_TRUE(st != nullptr);
    if (st == nullptr) { return; }
    std::vector<float> got, part;
    for (int f = 0; f < T; f += chunk) {
      const int n = std::min(chunk, T - f);
      ASSERT_TRUE(codec->decode_stream_chunk(
          *st, codes.data() + (std::size_t)f * 10, n, &part, &err));
      got.insert(got.end(), part.begin(), part.end());
    }
    ASSERT_TRUE(got.size() == whole.size());
    if (got.size() != whole.size()) { return; }
    const bool same = std::equal(got.begin(), got.end(), whole.begin());
    const double snr = snr_db_(got.data(), whole.data(), whole.size());
    std::printf("[drama3] stream decode, chunks of %d: %s, %.1f dB from the "
                "one-shot decode\n", chunk,
                same ? "BIT-IDENTICAL" : "differs", snr);
    EXPECT_TRUE(same || snr > 80.0);
  }
}

// ---- the stage: what it says, heard back ------------------------------------

namespace {

class D3TextBeats : public TypedStage<D3TextBeats> {
public:
  static constexpr const char* kTypeName = "ut-drama3-text-beats";
  using TypedStage::TypedStage;
  std::vector<std::string> texts;
  std::vector<FlexData> beats;          // object beats, after the texts
  Job process(RuntimeContext& ctx) override
  {
    if (_i < texts.size()) {
      co_await ctx.write(0, make_payload<FlexDataPayload>(
          FlexData::make_string(texts[_i++])));
      co_return;
    }
    if (_i - texts.size() < beats.size()) {
      co_await ctx.write(0, make_payload<FlexDataPayload>(
          beats[_i++ - texts.size()]));
      co_return;
    }
    ctx.signal_done();
  }
private:
  std::size_t _i = 0;
};

// One mono f32 PCM beat (a reference voice), then done.
class D3PcmBeat : public TypedStage<D3PcmBeat> {
public:
  static constexpr const char* kTypeName = "ut-drama3-pcm-beat";
  using TypedStage::TypedStage;
  std::vector<float> pcm;
  int rate = 44100;
  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = {(std::int64_t)pcm.size()};
    tb.resize_contiguous(pcm.size());
    std::memcpy(tb.as_f32(), pcm.data(), pcm.size() * 4);
    tb.sideband = FlexData::make_object();
    tb.sideband.as_object().insert("sample_rate", FlexData::make_int(rate));
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
  }
private:
  bool _sent = false;
};

class D3Transcripts : public TypedStage<D3Transcripts> {
public:
  static constexpr const char* kTypeName = "ut-drama3-transcripts";
  using TypedStage::TypedStage;
  std::vector<std::string> got;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    if (const auto* f = dynamic_cast<const FlexDataPayload*>(p.get())) {
      if (f->data.is_object() && f->data.as_object().contains("text")) {
        got.emplace_back(f->data.as_object().at("text").as_string(""));
      }
    }
  }
};

std::vector<std::string>
words_(std::string s)
{
  const std::size_t tag = s.find("<asr_text>");
  if (tag != std::string::npos) { s = s.substr(tag + 10); }
  std::vector<std::string> out;
  std::string w;
  for (unsigned char ch : s) {
    if (std::isalnum(ch) || ch >= 0x80 || ch == '\'') {
      w.push_back((char)std::tolower(ch));
    } else if (!w.empty()) {
      out.push_back(w);
      w.clear();
    }
  }
  if (!w.empty()) { out.push_back(w); }
  return out;
}

double
wer_(const std::vector<std::string>& ref, const std::vector<std::string>& hyp)
{
  std::vector<std::size_t> prev(hyp.size() + 1), cur(hyp.size() + 1);
  for (std::size_t j = 0; j <= hyp.size(); ++j) { prev[j] = j; }
  for (std::size_t i = 1; i <= ref.size(); ++i) {
    cur[0] = i;
    for (std::size_t j = 1; j <= hyp.size(); ++j) {
      cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ref[i - 1] == hyp[j - 1] ? 0 : 1)});
    }
    std::swap(prev, cur);
  }
  return ref.empty() ? 0.0 : (double)prev[hyp.size()] / (double)ref.size();
}

}  // namespace

// Drama 3 through the text-to-speech stage, straight into Qwen3-ASR: plain
// (the model's own voice) and cloning the golden's reference clip from its
// audio and transcript. Word error against the text.
TEST(drama3, stage_speech_transcribes)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  const std::string asr = env_("VPIPE_QWEN3_ASR_TEST_MODEL_PATH");
  if (mp.empty() || gp.empty() || asr.empty()) { return; }
  const std::vector<float> ref_pcm =
      read_raw_<float>(fs::path(gp) / "ref_pcm44k.f32");
  const FlexData meta = read_json_(fs::path(gp) / "meta.json");
  ASSERT_TRUE(!ref_pcm.empty() && meta.is_object());
  if (ref_pcm.empty() || !meta.is_object()) { return; }
  const std::string ref_text(meta.as_object().at("ref_text").as_string(""));
  const std::string text =
      "Please remember to bring your umbrella tomorrow, it will rain.";
  // Each checkpoint configured: the FP8 release, and the int4 one when
  // VPIPE_DRAMA3_INT4_TEST_MODEL_PATH names it.
  std::vector<std::string> models{mp};
  if (const std::string q = env_("VPIPE_DRAMA3_INT4_TEST_MODEL_PATH");
      !q.empty()) {
    models.push_back(q);
  }
  for (std::size_t mi = 0; mi < models.size() * 2; ++mi) {
    const std::string& model = models[mi / 2];
    const bool clone = (mi % 2) == 1;
    Session sess;
    // The last iteration's 36 GB comes back asynchronously; load into a
    // box that has it back.
    if (auto* mc = sess.metal_compute()) {
      (void)mc->memory_budget_settled(15000);
    }
    Pipeline pl("drama3-asr", &sess);
    auto src = std::make_unique<D3TextBeats>(&sess, "text",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
    src->texts = {text};
    src->allocate_oports(1);
    auto* s0 = pl.insert_stage(std::move(src));
    std::vector<InEdge> tts_in{{s0, 0}};
    if (clone) {
      auto ref = std::make_unique<D3PcmBeat>(&sess, "ref",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
      ref->pcm = ref_pcm;
      ref->allocate_oports(1);
      tts_in.push_back({pl.insert_stage(std::move(ref)), 0});
    }
    FlexData tcfg = FlexData::make_object();
    {
      auto o = tcfg.as_object();
      o.insert("hf_dir", FlexData::make_string(model));
      o.insert("max_frames", FlexData::make_int(600));
      o.insert("stream_chunk_frames", FlexData::make_int(0));
      if (clone) {
        o.insert("reference_text", FlexData::make_string(ref_text));
        o.insert("wait_for_reference", FlexData::make_bool(true));
      }
    }
    auto* tts = pl.insert_stage(std::make_unique<TextToSpeechStage>(
        &sess, "tts", std::move(tts_in), std::move(tcfg)));
    auto* tst = static_cast<TextToSpeechStage*>(tts);
    FlexData rcfg = FlexData::make_object();
    {
      auto o = rcfg.as_object();
      o.insert("output_sample_rate", FlexData::make_int(16000));
      o.insert("stacked", FlexData::make_bool(true));
    }
    auto* rs = pl.insert_stage(std::make_unique<AudioTemporalResampleStage>(
        &sess, "resample", std::vector<InEdge>{{tts, 0}}, std::move(rcfg)));
    FlexData acfg = FlexData::make_object();
    {
      auto o = acfg.as_object();
      o.insert("hf_dir", FlexData::make_string(asr));
      o.insert("max_new_tokens", FlexData::make_int(256));
    }
    auto* a = pl.insert_stage(std::make_unique<AudioTranscribeStage>(
        &sess, "asr", std::vector<InEdge>{{rs, 0}}, std::move(acfg)));
    auto sink = std::make_unique<D3Transcripts>(
        &sess, "sink", std::vector<InEdge>{{a, 0}}, FlexData::make_object());
    auto* sk = static_cast<D3Transcripts*>(pl.insert_stage(std::move(sink)));
    PipelineRuntime rt(&pl, &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    EXPECT_TRUE(tst->drama3() != nullptr);
    if (tst->drama3() != nullptr) {
      // The take ENDED: its own end of speech, not the frame budget.
      EXPECT_TRUE(tst->drama3()->last_run().ended);
    }
    rt.stop();
    ASSERT_TRUE(sk->got.size() == 1);
    if (sk->got.size() != 1) { continue; }
    const double w = wer_(words_(text), words_(sk->got[0]));
    std::printf("[drama3] %s, %s: WER %.3f\n  said  '%s'\n  heard '%s'\n",
                fs::path(model).filename().c_str(),
                clone ? "cloned voice" : "own voice", w, text.c_str(),
                sk->got[0].c_str());
    EXPECT_TRUE(w <= 0.15);
  }
}


// The SAMPLED path is the decode path: a seeded run's frames, fed back
// through teacher_decode, give the main-head logits the run drew from,
// row for row -- the frame buffers, the code columns each tower layer
// reads, the next frame's embedding, the input frame's codes 2..9. Also
// the draws themselves: every main token lies inside the kept set of the
// distribution it was drawn from.
TEST(drama3, sampled_run_is_the_decode_path)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  const Golden g = load_golden_(gp);
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  Drama3Config cfg;
  std::string err;
  ASSERT_TRUE(Drama3Config::from_dir(mp, &cfg, &err));
  MetalDrama3Model::LoadOptions lo;
  lo.max_rows = 4096;
  auto m = MetalDrama3Model::load(WeightSet::open(mp, nullptr), mc, cfg, lo,
                                  &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const int P = g.n_prompt, NC = cfg.main_candidates();
  std::vector<std::int32_t> prompt(g.frames.begin(),
                                   g.frames.begin() + (std::size_t)P * 11);
  MetalDrama3Model::Sampling sp;
  sp.seed = 7;
  sp.max_frames = 120;
  std::vector<MetalDrama3Model::Frame> fr;
  std::vector<float> drawn_from;
  ASSERT_TRUE(m->generate(prompt, P, sp, {}, &fr, &err, &drawn_from));
  const int F = (int)fr.size();
  ASSERT_TRUE(F > 8);
  if (F <= 8) { return; }
  // The run's rows: prompt, then each frame [sem0 + c0, c0..c9].
  std::vector<std::int32_t> seq = prompt;
  for (const auto& f : fr) {
    seq.push_back(cfg.semantic_begin + f[0]);
    seq.insert(seq.end(), f.begin(), f.end());
  }
  std::vector<float> tml, thl;
  ASSERT_TRUE(m->teacher_decode(seq, P + F, P, &tml, &thl, &err));
  // drawn_from holds F (+1 if it ended) rows; teacher_decode F + 1.
  const std::size_t rows = std::min(drawn_from.size(), tml.size()) /
                           (std::size_t)NC;
  ASSERT_TRUE(rows >= (std::size_t)F);
  double worst = 0.0;
  int same_bits = 0;
  for (std::size_t r = 0; r < rows; ++r) {
    const float* a = drawn_from.data() + r * NC;
    const float* b = tml.data() + r * NC;
    worst = std::max(worst, rel_l2_(a, b, (std::size_t)NC));
    same_bits += std::equal(a, a + NC, b) ? 1 : 0;
  }
  std::printf("[drama3] sampled run: %d frames; main logits drawn from vs "
              "the decode path: %d/%zu rows bit-identical, worst rel-L2 "
              "%.3g\n", F, same_bits, rows, worst);
  EXPECT_TRUE(same_bits == (int)rows);
}


namespace {
class D3PcmSink : public TypedStage<D3PcmSink> {
public:
  static constexpr const char* kTypeName = "ut-drama3-pcm-sink";
  using TypedStage::TypedStage;
  std::vector<float> pcm;
  int beats = 0;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    if (const auto* t = dynamic_cast<const TensorBeatPayload*>(p.get())) {
      const float* f = t->as_f32();
      pcm.insert(pcm.end(), f, f + t->element_count());
      ++beats;
    }
  }
};
}  // namespace

// The stage's streamed output (stream_chunk_frames 16: one PCM beat per 16
// frames as they are drawn) is its one-shot output, sample for sample:
// two runs of the same text at the same seed -- a text of two chunks, so
// the stream restarted at the second chunk is that chunk decoded on its
// own and joined, as the one-shot path does. Two SESSIONS, so the
// backbone's attention tuners are pinned to their defaults (PinnedTuners):
// left to vote they may pick other kernels in the second, and a near-tie
// draw then goes the other way (MEASURED: frames identical to 94 of 99,
// then not).
TEST(drama3, stage_stream_is_the_one_shot)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  if (mp.empty()) { return; }
  static constexpr const char* kStreamText =
      "Good morning! The weather today is sunny with a light breeze, and "
      "the afternoon should stay warm and bright across the whole region. "
      "Tomorrow a band of clouds arrives from the west, bringing a few "
      "showers by the evening. The weekend looks calm again, with clear "
      "skies on Saturday and a cool, quiet morning on Sunday.";
  ASSERT_TRUE(genai::drama3::split_text(kStreamText).size() == 2);
  const PinnedTuners pinned;
  std::vector<float> got[2];
  int beats[2] = {0, 0};
  for (int run = 0; run < 2; ++run) {
    Session sess;
    if (auto* mc = sess.metal_compute()) {
      (void)mc->memory_budget_settled(15000);
    }
    Pipeline pl("drama3-stream", &sess);
    auto src = std::make_unique<D3TextBeats>(&sess, "text",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
    src->texts = {kStreamText};
    src->allocate_oports(1);
    auto* s0 = pl.insert_stage(std::move(src));
    FlexData tcfg = FlexData::make_object();
    {
      auto o = tcfg.as_object();
      o.insert("hf_dir", FlexData::make_string(mp));
      o.insert("max_frames", FlexData::make_int(600));
      o.insert("stream_chunk_frames", FlexData::make_int(run == 0 ? 0 : 16));
    }
    auto* tts = pl.insert_stage(std::make_unique<TextToSpeechStage>(
        &sess, "tts", std::vector<InEdge>{{s0, 0}}, std::move(tcfg)));
    auto* sk = static_cast<D3PcmSink*>(pl.insert_stage(
        std::make_unique<D3PcmSink>(&sess, "sink",
                                    std::vector<InEdge>{{tts, 0}},
                                    FlexData::make_object())));
    PipelineRuntime rt(&pl, &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    got[run] = sk->pcm;
    beats[run] = sk->beats;
  }
  std::printf("[drama3] stage: one-shot %zu samples in %d beat, streamed "
              "%zu in %d beats\n", got[0].size(), beats[0], got[1].size(),
              beats[1]);
  EXPECT_TRUE(beats[0] == 1 && beats[1] > 1);
  if (got[0].size() == got[1].size() && !got[0].empty()) {
    std::size_t first = got[0].size();
    for (std::size_t i = 0; i < got[0].size(); ++i) {
      if (got[0][i] != got[1][i]) { first = i; break; }
    }
    std::printf("[drama3] first differing sample %zu (frame %zu); SNR %.1f "
                "dB\n", first, first / 2048,
                snr_db_(got[1].data(), got[0].data(), got[0].size()));
  }
  EXPECT_TRUE(!got[0].empty() && got[0] == got[1]);
}

// The stage's long-text path, its context and its voices, in one session
// (one load of the 36 GB model) and through Qwen3-ASR:
//   1. a ~620-byte paragraph, cut in chunks the reference's way, every
//      chunk after the first spoken with the first and the latest in its
//      prompt, the chunks' audio joined into ONE beat;
//   2. a short line, then
//   3. its edit, naming 2 in "previous_takes" (its take in the context);
//   4. a two-speaker line, each speaker cloned from its own file through
//      `references` (<|speaker:1|> on the second transcript).
// Beats 1-3 are held to the words asked for. Beat 4 only to both voices
// in its prompt: on that line the MODEL now and then stops after the first
// speaker or speaks nothing and runs to max_frames (MEASURED over 12 seeds
// a session: 0-1 early stops and 3-6 silent runs, in the reference's own
// per-speaker formatting; vllm-omni's recipe notes the same), so its words
// depend on the draw. The attention tuners are pinned so a run is the same
// one each time on a box.
TEST(drama3, stage_long_text_voices_and_takes)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  const std::string asr = env_("VPIPE_QWEN3_ASR_TEST_MODEL_PATH");
  if (mp.empty() || gp.empty() || asr.empty()) { return; }
  const fs::path clips = fs::path(gp).parent_path();
  const fs::path a0 = clips / "common_voice_en_10119832.wav";
  const fs::path a1 = clips / "common_voice_en_103675.wav";
  if (!fs::exists(a0) || !fs::exists(a1)) {
    std::printf("[drama3] no reference clips beside the golden; skipped\n");
    return;
  }
  const std::string para =
      "The museum opens at nine in the morning, and the first tour leaves "
      "from the main hall at half past nine. Visitors are asked to leave "
      "large bags at the cloakroom near the entrance. The east wing holds "
      "the collection of maps, some of them more than four hundred years "
      "old, and the guides explain how sailors used them to cross the "
      "ocean. After lunch, the west wing opens its exhibition of clocks, "
      "where every hour the whole room chimes at once. Photography is "
      "allowed everywhere except in the map room, because the old paper "
      "fades under bright light.";
  const std::string wed =
      "The quarterly report is due on Wednesday, so please finish it by "
      "noon.";
  const std::string thu =
      "The quarterly report is due on Thursday, so please finish it by "
      "noon.";
  const std::string duo =
      "<|speaker:0|> Are you coming with us tonight? <|speaker:1|> Yes, I "
      "will meet you at the station at eight.";
  std::optional<PinnedTuners> pinned(std::in_place);
  Session sess;
  if (auto* mc = sess.metal_compute()) {
    (void)mc->memory_budget_settled(15000);
  }
  Pipeline pl("drama3-long", &sess);
  auto src = std::make_unique<D3TextBeats>(&sess, "text",
                                           std::vector<InEdge>{},
                                           FlexData::make_object());
  src->texts = {para, wed};
  {
    FlexData b = FlexData::make_object();
    auto o = b.as_object();
    o.insert("text", FlexData::make_string(thu));
    FlexData pt = FlexData::make_array();
    pt.as_array().push_back(FlexData::make_string(wed));
    o.insert("previous_takes", pt);
    src->beats.push_back(b);
    src->beats.push_back(FlexData::make_string(duo));
  }
  src->allocate_oports(1);
  auto* s0 = pl.insert_stage(std::move(src));
  FlexData tcfg = FlexData::make_object();
  {
    auto o = tcfg.as_object();
    o.insert("hf_dir", FlexData::make_string(mp));
    o.insert("max_frames", FlexData::make_int(600));
    o.insert("stream_chunk_frames", FlexData::make_int(0));
    o.insert("interrupt_on_new_text", FlexData::make_bool(false));
    FlexData refs = FlexData::make_array();
    for (const auto& [path, words] :
         {std::pair<std::string, std::string>{
              a0.string(), "We asked over twenty different people, and "
                           "they all said it was his."},
          {a1.string(), "<|speaker:1|> I'm never more aware of a room's "
                        "acoustics than when I'm trying to enjoy a snack "
                        "I have no intention of sharing."}}) {
      FlexData r = FlexData::make_object();
      r.as_object().insert("audio", FlexData::make_string(path));
      r.as_object().insert("text", FlexData::make_string(words));
      refs.as_array().push_back(r);
    }
    o.insert("references", refs);
  }
  auto* tts = pl.insert_stage(std::make_unique<TextToSpeechStage>(
      &sess, "tts", std::vector<InEdge>{{s0, 0}}, std::move(tcfg)));
  auto* tst = static_cast<TextToSpeechStage*>(tts);
  FlexData rcfg = FlexData::make_object();
  {
    auto o = rcfg.as_object();
    o.insert("output_sample_rate", FlexData::make_int(16000));
    o.insert("stacked", FlexData::make_bool(true));
  }
  auto* rs = pl.insert_stage(std::make_unique<AudioTemporalResampleStage>(
      &sess, "resample", std::vector<InEdge>{{tts, 0}}, std::move(rcfg)));
  FlexData acfg = FlexData::make_object();
  {
    auto o = acfg.as_object();
    o.insert("hf_dir", FlexData::make_string(asr));
    o.insert("max_new_tokens", FlexData::make_int(400));
  }
  auto* a = pl.insert_stage(std::make_unique<AudioTranscribeStage>(
      &sess, "asr", std::vector<InEdge>{{rs, 0}}, std::move(acfg)));
  auto* sk = static_cast<D3Transcripts*>(pl.insert_stage(
      std::make_unique<D3Transcripts>(&sess, "sink",
                                      std::vector<InEdge>{{a, 0}},
                                      FlexData::make_object())));
  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  pinned.reset();
  EXPECT_TRUE(tst->drama3_references().size() == 2);
  const auto& bs = tst->drama3_beats();
  ASSERT_TRUE(bs.size() == 4 && sk->got.size() == 4);
  if (bs.size() != 4 || sk->got.size() != 4) { return; }
  const std::string said[4] = {
      para, wed, thu,
      "Are you coming with us tonight? Yes, I will meet you at the station "
      "at eight."};
  for (int i = 0; i < 4; ++i) {
    const double w = wer_(words_(said[i]), words_(sk->got[(std::size_t)i]));
    std::printf("[drama3] beat %d: %d chunk(s), %d spoken, %d with "
                "earlier takes, %d previous, %d voices, prompt <= %d rows, "
                "%d frames; WER %.3f\n  heard '%s'\n", i + 1, bs[i].chunks,
                bs[i].spoken, bs[i].with_history, bs[i].previous,
                bs[i].voices, bs[i].max_prompt, bs[i].frames, w,
                sk->got[(std::size_t)i].c_str());
    EXPECT_TRUE(i == 3 || w <= 0.15);
    EXPECT_TRUE(bs[i].voices == 2);
  }
  // 1: chunked; every chunk after the first spoken after earlier takes.
  EXPECT_TRUE(bs[0].chunks >= 2 && bs[0].spoken == bs[0].chunks);
  EXPECT_TRUE(bs[0].with_history == bs[0].chunks - 1);
  // 2 / 3: one chunk each; 3 carries 2's take in its prompt.
  EXPECT_TRUE(bs[1].chunks == 1 && bs[1].previous == 0);
  EXPECT_TRUE(bs[2].chunks == 1 && bs[2].previous == 1 &&
              bs[2].with_history == 1);
  EXPECT_TRUE(bs[2].max_prompt > bs[1].max_prompt + bs[1].frames);
}

// ---- the block-FP8 expert kernels, on their own -------------------------

namespace {

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (std::uint16_t)(u >> 16);
}

float
from_bf16_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

double
fp8_e4m3_(std::uint8_t b)
{
  const int e = (b >> 3) & 0xF, m = b & 7;
  const double v = e == 0 ? m / 8.0 * std::ldexp(1.0, -6)
                          : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
  return (b & 0x80) ? -v : v;
}

}  // namespace

// The routed experts' prefill GEMMs (fp8_qmm_grouped_{swiglu,down}_b128:
// pairs sorted per expert, each expert's weight read once a 32-row tile)
// and the per-pair gather they replace (dense_moe_gather_*_fp8), both
// against an f64 reference over random E4M3 codes and 128x128 block
// scales -- bf16 activations, as Drama 3 runs. Several experts with a
// ragged pair count each, so pad slots and pad tiles are exercised.
TEST(drama3, fp8_grouped_gemm_matches_gather)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using metal_compute::CommandStream;
  using metal_compute::ComputeEncoder;
  using metal_compute::SharedBuffer;
  constexpr int E = 6, H = 1024, I = 512, M = 75, TOPK = 4;
  constexpr int NP = M * TOPK, IB = I / 128, HB = H / 128;
  std::uint32_t rng = 12345;
  auto next = [&]() {
    rng = rng * 1664525u + 1013904223u;
    return rng;
  };
  auto unif = [&]() { return (next() >> 8) * (1.0 / 16777216.0); };
  auto code = [&]() {
    std::uint8_t b = (std::uint8_t)(next() >> 24) & 0x7F;
    if (b == 0x7F) { b = 0x7E; }
    return (std::uint8_t)(b | ((next() >> 31) << 7));
  };
  std::vector<std::uint8_t> gu((std::size_t)E * 2 * I * H), dn(
      (std::size_t)E * H * I);
  for (auto& b : gu) { b = code(); }
  for (auto& b : dn) { b = code(); }
  // Scales: real values, and the kernels' (x256).
  std::vector<float> gus((std::size_t)E * 2 * IB * HB), dns((std::size_t)E *
                                                             HB * IB);
  for (auto& s : gus) { s = (float)(2e-3 + 6e-3 * unif()); }
  for (auto& s : dns) { s = (float)(2e-3 + 6e-3 * unif()); }
  std::vector<std::uint16_t> x((std::size_t)M * H);
  for (auto& v : x) { v = to_bf16_((float)(2.0 * unif() - 1.0)); }
  std::vector<std::int32_t> eid(NP);
  for (int p = 0; p < NP; ++p) {
    // Skewed routing: expert 5 rare, so its block is mostly padding.
    const int r = (int)(next() % 23u);
    eid[p] = r < 20 ? r % 5 : 5;
  }
  // The f64 reference: act = silu(x Wg^T) * (x Wu^T), part = act Wd^T.
  auto wgu = [&](int e, int r, int k) {     // interleaved row r
    const double s = gus[((std::size_t)e * 2 + (r & 1)) * IB * HB +
                         (std::size_t)((r >> 1) / 128) * HB + k / 128];
    return fp8_e4m3_(gu[((std::size_t)e * 2 * I + r) * H + k]) * s;
  };
  auto wd = [&](int e, int n, int k) {
    const double s =
        dns[(std::size_t)e * HB * IB + (std::size_t)(n / 128) * IB + k / 128];
    return fp8_e4m3_(dn[((std::size_t)e * H + n) * I + k]) * s;
  };
  std::vector<double> act_ref((std::size_t)NP * I), part_ref((std::size_t)NP *
                                                              H);
  for (int p = 0; p < NP; ++p) {
    const int e = eid[p], t = p / TOPK;
    for (int g = 0; g < I; ++g) {
      double a = 0.0, b = 0.0;
      for (int k = 0; k < H; ++k) {
        const double xv = from_bf16_(x[(std::size_t)t * H + k]);
        a += xv * wgu(e, 2 * g, k);
        b += xv * wgu(e, 2 * g + 1, k);
      }
      // The kernels store act in bf16: the reference rounds it too, so
      // the down GEMM is held to its own arithmetic.
      act_ref[(std::size_t)p * I + g] =
          from_bf16_(to_bf16_((float)(a / (1.0 + std::exp(-a)) * b)));
    }
    for (int n = 0; n < H; ++n) {
      double acc = 0.0;
      for (int k = 0; k < I; ++k) {
        acc += act_ref[(std::size_t)p * I + k] * wd(e, n, k);
      }
      part_ref[(std::size_t)p * H + n] = acc;
    }
  }
  // Sorted slots for the grouped GEMMs: each expert's pairs in pair
  // order, padded to whole 32-row tiles.
  std::vector<std::int32_t> srow, sdst, t2e;
  for (int e = 0; e < E; ++e) {
    std::vector<int> ps;
    for (int p = 0; p < NP; ++p) {
      if (eid[p] == e) { ps.push_back(p); }
    }
    const int tiles = ((int)ps.size() + 31) / 32;
    for (int t = 0; t < tiles; ++t) { t2e.push_back(e); }
    for (int k = 0; k < tiles * 32; ++k) {
      srow.push_back(k < (int)ps.size() ? ps[k] / TOPK : -1);
      sdst.push_back(k < (int)ps.size() ? ps[k] : -1);
    }
  }
  t2e.push_back(-1);                 // a pad tile, skipped
  for (int k = 0; k < 32; ++k) { srow.push_back(-1); sdst.push_back(-1); }
  const int npad = (int)srow.size();
  auto upload = [&](const void* p, std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n);
    std::memcpy(b.contents(), p, n);
    return b;
  };
  std::vector<float> gus_k(gus), dns_k(dns);
  for (auto& s : gus_k) { s *= 256.0f; }
  for (auto& s : dns_k) { s *= 256.0f; }
  SharedBuffer bgu = upload(gu.data(), gu.size()),
               bdn = upload(dn.data(), dn.size()),
               bgus = upload(gus_k.data(), gus_k.size() * 4),
               bdns = upload(dns_k.data(), dns_k.size() * 4),
               bx = upload(x.data(), x.size() * 2),
               beid = upload(eid.data(), eid.size() * 4),
               bsrow = upload(srow.data(), srow.size() * 4),
               bsdst = upload(sdst.data(), sdst.size() * 4),
               bt2e = upload(t2e.data(), t2e.size() * 4);
  SharedBuffer act = mc->make_shared_buffer((std::size_t)NP * I * 2),
               part = mc->make_shared_buffer((std::size_t)NP * H * 2),
               gact = mc->make_shared_buffer((std::size_t)npad * I * 2),
               gpart = mc->make_shared_buffer((std::size_t)NP * H * 2);
  auto lib_d = mc->load_library("dense_gemm_bf16");
  auto lib_q = mc->load_library("affine_qmm_steel_bf16");
  auto f_gs = lib_d.function("dense_moe_gather_swiglu_fp8");
  auto f_gd = lib_d.function("dense_moe_gather_down_fp8");
  auto f_qs = lib_q.function("fp8_qmm_grouped_swiglu_b128");
  auto f_qd = lib_q.function("fp8_qmm_grouped_down_b128");
  ASSERT_TRUE(f_gs.valid() && f_gd.valid() && f_qs.valid() && f_qd.valid());
  if (!f_gs.valid() || !f_gd.valid() || !f_qs.valid() || !f_qd.valid()) {
    return;
  }
  CommandStream st = mc->make_command_stream();
  {
    ComputeEncoder enc = st.begin_compute();
    enc.set_function(f_gs);
    enc.set_buffer(0, bgu); enc.set_buffer(1, bgus);
    enc.set_buffer(2, bx); enc.set_buffer(3, act);
    enc.set_constant(4, H); enc.set_constant(5, 2 * I);
    enc.set_buffer(6, beid); enc.set_constant(7, TOPK);
    enc.dispatch({32, (unsigned)(I / 2), (unsigned)NP}, {32, 2, 1});
    enc.set_function(f_gd);
    enc.set_buffer(0, bdn); enc.set_buffer(1, bdns);
    enc.set_buffer(2, act); enc.set_buffer(3, part);
    enc.set_constant(4, I); enc.set_constant(5, H); enc.set_buffer(6, beid);
    enc.dispatch({32, (unsigned)(H / 4), (unsigned)NP}, {32, 2, 1});
    enc.set_function(f_qs);
    enc.set_buffer(0, bgu); enc.set_buffer(1, bgus);
    enc.set_buffer(2, bx); enc.set_buffer(3, gact);
    enc.set_constant(4, H); enc.set_constant(5, 2 * I);
    enc.set_constant(6, npad); enc.set_buffer(7, bt2e);
    enc.set_buffer(8, bsrow);
    enc.dispatch({(unsigned)(((2 * I + 31) / 32) * 32),
                  (unsigned)(((npad + 31) / 32) * 2), 2}, {32, 2, 2});
    enc.set_function(f_qd);
    enc.set_buffer(0, bdn); enc.set_buffer(1, bdns);
    enc.set_buffer(2, gact); enc.set_buffer(3, gpart);
    enc.set_constant(4, I); enc.set_constant(5, H);
    enc.set_constant(6, npad); enc.set_buffer(7, bt2e);
    enc.set_buffer(8, bsdst);
    enc.dispatch({(unsigned)(((H + 31) / 32) * 32),
                  (unsigned)(((npad + 31) / 32) * 2), 2}, {32, 2, 2});
  }
  ASSERT_TRUE(st.commit().wait_ok());
  auto rel = [](const std::vector<double>& ref, const std::uint16_t* got,
                std::size_t n, const std::vector<std::size_t>* map) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double g = from_bf16_(got[map != nullptr ? (*map)[i] : i]);
      num += (g - ref[i]) * (g - ref[i]);
      den += ref[i] * ref[i];
    }
    return std::sqrt(num / den);
  };
  // gact holds pair p's row at its sorted slot.
  std::vector<std::size_t> slot_of((std::size_t)NP * I);
  for (int k = 0; k < npad; ++k) {
    if (sdst[k] < 0) { continue; }
    for (int g = 0; g < I; ++g) {
      slot_of[(std::size_t)sdst[k] * I + g] = (std::size_t)k * I + g;
    }
  }
  const auto* a16 = static_cast<const std::uint16_t*>(act.contents());
  const auto* p16 = static_cast<const std::uint16_t*>(part.contents());
  const auto* ga16 = static_cast<const std::uint16_t*>(gact.contents());
  const auto* gp16 = static_cast<const std::uint16_t*>(gpart.contents());
  const double ea = rel(act_ref, a16, act_ref.size(), nullptr);
  const double ep = rel(part_ref, p16, part_ref.size(), nullptr);
  const double ga = rel(act_ref, ga16, act_ref.size(), &slot_of);
  const double gp = rel(part_ref, gp16, part_ref.size(), nullptr);
  std::printf("[drama3] fp8 experts vs f64: gather act %.2e part %.2e, "
              "grouped act %.2e part %.2e (%d pairs, %d slots)\n", ea, ep,
              ga, gp, NP, npad);
  // bf16 storage of act and part is ~2e-3; anything wrong is O(1).
  EXPECT_TRUE(ea < 5e-3 && ep < 5e-3 && ga < 5e-3 && gp < 5e-3);
}



// THE ROUTED EXPERTS IN A CACHE (MetalQwenModel::Config::
// expert_cache_bytes) are the resident model's, byte for byte: for each
// checkpoint configured -- the release's block-FP8 experts and, when
// VPIPE_DRAMA3_INT4_TEST_MODEL_PATH is set, the int4 one's GPTQ experts --
// the golden's teacher-forced pass (a 310-row prefill: the grouped GEMMs
// over transient slots), its step-by-step decode path (the per-pair gather
// over cached slots) and a seeded sampled take, resident and with ~3 GB of
// cache (under 10% of the experts resident) and with 16 GB; the cached
// arms read ahead on every decode step. The attention tuners are pinned
// (PinnedTuners): before they were, one run in four had the RESIDENT load
// differ from both cached ones in every output -- its prefill tuner had
// voted qtile (resident_loads_agree found it: same weights, other kernel).
// A mismatch prints its rel-L2 and first differing index.
TEST(drama3, expert_cache_is_resident)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  const Golden g = load_golden_(gp);
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  std::optional<PinnedTuners> pinned(std::in_place);
  std::vector<std::string> models{mp};
  if (const std::string q = env_("VPIPE_DRAMA3_INT4_TEST_MODEL_PATH");
      !q.empty()) {
    models.push_back(q);
  }
  struct Out {
    std::vector<float> hid, ml, hl, dml, dhl, dhid;
    std::vector<MetalDrama3Model::Frame> frames;
    std::size_t held = 0;
    int slots = 0, redos = 0;
    bool ok = false, hs = false;
  };
  // drop: every drop-th of the cache's handshake turns abandoned (the GPU
  // runs on, the step is redone with commits)
  auto run = [&](const std::string& model, std::size_t cache, int drop = 0) {
    Out o;
    Session sess;
    MetalCompute* mc = sess.metal_compute();
    if (mc == nullptr) { return o; }
    (void)mc->memory_budget_settled(15000);
    Drama3Config cfg;
    std::string err;
    if (!Drama3Config::from_dir(model, &cfg, &err)) { return o; }
    MetalDrama3Model::LoadOptions lo;
    lo.max_rows = 4096;
    lo.expert_cache_bytes = cache;
    auto m = MetalDrama3Model::load(WeightSet::open(model, nullptr), mc, cfg,
                                    lo, &err);
    if (m == nullptr) {
      std::printf("[drama3] load: %s\n", err.c_str());
      return o;
    }
    o.held = m->resident_bytes();
    o.hs = m->backbone()->expert_handshake();
    m->backbone()->xhs_drop_every(drop);
    const int P = g.n_prompt;
    std::vector<std::int32_t> prompt(g.frames.begin(),
                                     g.frames.begin() + (std::size_t)P * 11);
    MetalDrama3Model::Sampling sp;
    sp.seed = 7;
    sp.max_frames = 120;
    o.ok = m->teacher_forced(g.frames, g.n, &o.hid, &o.ml, &o.hl, &err) &&
           m->teacher_decode(g.frames, g.n, P, &o.dml, &o.dhl, &err,
                             &o.dhid) &&
           m->generate(prompt, P, sp, {}, &o.frames, &err);
    if (!err.empty()) { std::printf("[drama3] %s\n", err.c_str()); }
    o.redos = m->backbone()->xhs_redos();
    if (const MoeExpertCache* c = m->backbone()->expert_cache()) {
      o.slots = c->slots();
      const auto& st = c->stats();
      std::printf("[drama3] %zu MB expert cache: %llu picks, hit rate "
                  "%.1f%%, %llu admitted, %llu transient, %llu evicted, %.0f "
                  "MB read in %.0f ms\n", cache >> 20,
                  (unsigned long long)st.picks, 100.0 * st.hit_rate(),
                  (unsigned long long)st.admitted,
                  (unsigned long long)st.transient,
                  (unsigned long long)st.evicted, st.bytes_read / 1048576.0,
                  st.read_ms);
    }
    return o;
  };
  auto same = [](const std::vector<float>& a, const std::vector<float>& b) {
    const bool eq = a.size() == b.size() && !a.empty() &&
                    std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    if (!eq && a.size() == b.size() && !a.empty()) {
      double d = 0, n = 0;
      std::size_t first = a.size();
      for (std::size_t i = 0; i < a.size(); ++i) {
        d += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
        n += (double)a[i] * a[i];
        if (first == a.size() && a[i] != b[i]) { first = i; }
      }
      std::printf("[drama3]   differs: rel-L2 %.3g, first at %zu of %zu\n",
                  std::sqrt(d / std::max(n, 1e-30)), first, a.size());
    }
    return eq;
  };
  for (const std::string& model : models) {
    const Out a = run(model, 0);
    const Out b = run(model, (std::size_t)3072 << 20);
    ASSERT_TRUE(a.ok && b.ok);
    if (!a.ok || !b.ok) { continue; }
    const bool hid = same(a.hid, b.hid), ml = same(a.ml, b.ml),
               hl = same(a.hl, b.hl), dml = same(a.dml, b.dml),
               dhl = same(a.dhl, b.dhl), dh = same(a.dhid, b.dhid),
               fr = a.frames == b.frames;
    std::printf("[drama3] %s: held %zu MB resident, %zu MB cached; "
                "identical: hidden %d main %d heads %d | decode main %d "
                "heads %d hidden %d | %zu sampled frames %d\n",
                fs::path(model).filename().c_str(), a.held >> 20,
                b.held >> 20, hid, ml, hl, dml, dhl, dh, a.frames.size(), fr);
    EXPECT_TRUE(hid && ml && hl && dml && dhl && dh && fr);
    EXPECT_TRUE(b.held < a.held / 2);
    // Where this GPU takes the expert cache's handshake (gpu-host-
    // handshake.h) the cached arms take it -- and redo a step whose turn
    // was abandoned from the state it found, every output the same.
    std::printf("[drama3] %s: the cached arm's turns %s\n",
                fs::path(model).filename().c_str(),
                b.hs ? "inside the running GPU work (the handshake)"
                     : "at a commit and wait each");
    if (b.hs) {
      const Out d = run(model, (std::size_t)3072 << 20, 97);
      ASSERT_TRUE(d.ok);
      if (!d.ok) { continue; }
      const bool all = same(a.hid, d.hid) && same(a.ml, d.ml) &&
                       same(a.hl, d.hl) && same(a.dml, d.dml) &&
                       same(a.dhl, d.dhl) && same(a.dhid, d.dhid) &&
                       a.frames == d.frames;
      std::printf("[drama3] %s, every 97th turn abandoned: %d steps "
                  "redone, identical %d\n",
                  fs::path(model).filename().c_str(), d.redos, all);
      EXPECT_TRUE(b.redos == 0);
      EXPECT_TRUE(d.redos > 0);
      EXPECT_TRUE(all);
    }
    // The FP8 kernels offset experts in 64 bits, so their cache is not
    // held under the affine kernels' 2^33-byte pool: a 16 GB cache puts
    // slots past 4096, the gate|up pool (2 MB a slot) past 8 GB.
    if (model == mp) {
      const Out c = run(model, (std::size_t)16384 << 20);
      ASSERT_TRUE(c.ok);
      if (!c.ok) { continue; }
      const bool big = same(a.hid, c.hid) && same(a.ml, c.ml) &&
                       same(a.hl, c.hl) && same(a.dml, c.dml) &&
                       same(a.dhl, c.dhl) && same(a.dhid, c.dhid) &&
                       a.frames == c.frames;
      std::printf("[drama3] %s, 16 GB cache: %d slots, identical %d\n",
                  fs::path(model).filename().c_str(), c.slots, big);
      EXPECT_TRUE(c.slots > 4096);
      EXPECT_TRUE(big);
    }
  }
  pinned.reset();
}

// THE SAME CHECKPOINT LOADED RESIDENT N TIMES (VPIPE_DRAMA3_REPEAT=N; off
// otherwise): every load must hold the same bytes and prefill the golden's
// prompt to the same hidden rows. Each load prints a digest of its weights
// and of its outputs; one that differs from the first names the buffers
// that did. Only the decode-attention tuner is pinned, so the prefill
// tuner votes at every load (VPIPE_QWEN_AUTOTUNE_LOG=1 prints the votes).
TEST(drama3, resident_loads_agree)
{
  const std::string mp = env_("VPIPE_DRAMA3_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_DRAMA3_GOLDEN");
  const int reps = std::atoi(env_("VPIPE_DRAMA3_REPEAT").c_str());
  if (mp.empty() || gp.empty() || reps <= 0) { return; }
  const Golden g = load_golden_(gp);
  ASSERT_TRUE(g.ok);
  if (!g.ok) { return; }
  const std::string was = env_("VPIPE_QWEN_GQA_AUTOTUNE");
  ::setenv("VPIPE_QWEN_GQA_AUTOTUNE", "0", 1);
  auto mix = [](const std::vector<float>& v) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (float f : v) {
      std::uint32_t u;
      std::memcpy(&u, &f, 4);
      h = (h ^ u) * 0x100000001b3ull;
    }
    return h;
  };
  std::vector<std::uint64_t> d0;
  std::vector<float> h0;
  int bad_w = 0, bad_o = 0;
  for (int r = 0; r < reps; ++r) {
    Session sess;
    MetalCompute* mc = sess.metal_compute();
    ASSERT_TRUE(mc != nullptr);
    if (mc == nullptr) { break; }
    (void)mc->memory_budget_settled(15000);
    Drama3Config cfg;
    std::string err;
    ASSERT_TRUE(Drama3Config::from_dir(mp, &cfg, &err));
    MetalDrama3Model::LoadOptions lo;
    lo.max_rows = 4096;
    auto m = MetalDrama3Model::load(WeightSet::open(mp, nullptr), mc, cfg,
                                    lo, &err);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { break; }
    const std::vector<std::uint64_t> d = m->backbone()->weight_digests();
    std::vector<float> hid, ml, hl;
    ASSERT_TRUE(m->teacher_forced(g.frames, g.n, &hid, &ml, &hl, &err));
    std::uint64_t dw = 0;
    for (std::uint64_t x : d) { dw = (dw ^ x) * 0x100000001b3ull; }
    std::printf("[drama3] load %d: weights %016llx, hidden %016llx\n", r,
                (unsigned long long)dw, (unsigned long long)mix(hid));
    if (r == 0) {
      d0 = d;
      h0 = hid;
      continue;
    }
    if (d != d0) {
      ++bad_w;
      std::string which;
      for (std::size_t i = 0; i < d.size() && i < d0.size(); ++i) {
        if (d[i] != d0[i]) { which += " " + std::to_string(i); }
      }
      std::printf("[drama3]   weights differ from load 0 in buffers%s "
                  "(of %zu)\n", which.c_str(), d.size());
    }
    if (hid != h0) {
      ++bad_o;
      double num = 0, den = 0;
      std::size_t first = hid.size();
      for (std::size_t i = 0; i < hid.size() && i < h0.size(); ++i) {
        const double e = (double)hid[i] - h0[i];
        num += e * e;
        den += (double)h0[i] * h0[i];
        if (first == hid.size() && hid[i] != h0[i]) { first = i; }
      }
      std::printf("[drama3]   hidden differs from load 0: rel-L2 %.3g, "
                  "first at %zu of %zu\n", std::sqrt(num / std::max(den,
                  1e-30)), first, hid.size());
    }
  }
  if (was.empty()) {
    ::unsetenv("VPIPE_QWEN_GQA_AUTOTUNE");
  } else {
    ::setenv("VPIPE_QWEN_GQA_AUTOTUNE", was.c_str(), 1);
  }
  std::printf("[drama3] %d loads: %d held other weights, %d other hidden "
              "rows\n", reps, bad_w, bad_o);
  EXPECT_TRUE(bad_w == 0 && bad_o == 0);
}
