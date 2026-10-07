// YuE2-3B bring-up tests: the protocol, the tiktoken text BPE, the AR
// decode and the flow-matching pass, each against the reference package's
// own output (tools/dump_yue2_golden.py).
//
// Env:
//   VPIPE_YUE2_TEST_MODEL_PATH  the m-a-p/YuE2-3B directory.
//   VPIPE_YUE2_GOLDEN           the golden directory.
// Every test that needs either skips without it; protocol_logic needs
// neither.
//
// GEOMETRY. The flow-matching golden runs 200 latent rows (202 with the
// borders) against an 793-token AR prefix: every GEMM clears the 64-row
// matmul2d floor and the 128-row wide steel tile, and neither the query
// nor the key length is a multiple of any attention tile, so the ragged
// edges are exercised too.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/stage-registry.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-transcribe-stage.h"
#include "stages/audio-vae-decode-stage.h"
#include "stages/audio-video/audio-temporal-resample-stage.h"
#include "stages/generate-audio-stage.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "generative-models/yue2/metal-oobleck-decoder.h"
#include "generative-models/yue2/metal-yue2-model.h"
#include "generative-models/yue2/yue2-protocol.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
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

FlexData
read_json_(const fs::path& p)
{
  std::ifstream in(p);
  if (!in) { return FlexData(); }
  return FlexData::from_json(in);
}

// The golden's demo request: tonight-awake from the model's examples,
// with the golden's own ABC for the score-given cases.
struct Demo {
  std::string style, lyrics, abc;
  std::uint64_t seed = 0;
};

bool
load_demo_(const fs::path& model, const fs::path& golden, Demo* d)
{
  const FlexData ex = read_json_(model / "examples" / "tonight-awake.json");
  const FlexData meta = read_json_(golden / "meta.json");
  if (!ex.is_object() || !meta.is_object()) { return false; }
  auto eo = ex.as_object();
  auto mo = meta.as_object();
  d->style = std::string(eo.at("style").as_string());
  d->lyrics = std::string(eo.at("lyrics").as_string());
  d->seed = (std::uint64_t)mo.at("seed").as_int();
  d->abc = std::string(mo.at("abc_text").as_string());
  return true;
}

std::unique_ptr<Tokenizer>
tokenizer_(const fs::path& model)
{
  return Tokenizer::from_tiktoken((model / "qwen.tiktoken").string(),
                                  yue2::tiktoken_specials(), nullptr);
}

void
report_ids_(const char* what, const std::vector<std::int32_t>& got,
            const std::vector<std::int32_t>& want)
{
  std::size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i]) { ++i; }
  std::printf("[yue2] %s: %zu/%zu match", what, i, want.size());
  if (i < want.size()) {
    std::printf(" (first diff @%zu: got %d want %d)", i,
                i < got.size() ? got[i] : -1, want[i]);
  }
  std::printf("\n");
}

}  // namespace

// ---- no model ------------------------------------------------------------

TEST(yue2, protocol_logic)
{
  using namespace yue2;
  // The two windows cover their phase's ids and nothing else they need.
  const Window a = phase_window(Phase::kAbc);
  EXPECT_TRUE(a.row0 == 0 && a.rows == kAbcEnd + 1 && a.end == kAbcEnd);
  const Window m = phase_window(Phase::kSemantic);
  EXPECT_TRUE(m.row0 == kMusicEnd && m.end == kMusicEnd &&
              m.row0 + m.rows == kCodecOffset + kCodecSize);

  // Greedy music: the end id wins on its logit but is barred before
  // min_tokens; afterwards it is picked.
  Sampling s = Sampling::semantic_default();
  s.temperature = 0.0f;
  s.min_tokens = 2;
  s.repetition_penalty = 1.0f;
  std::vector<float> lg((std::size_t)m.rows, 0.0f);
  lg[0] = 9.0f;                         // MUSIC_END
  lg[1 + 5] = 3.0f;                     // code 5
  TokenPicker p(Phase::kSemantic, s, 1, false);
  EXPECT_TRUE(p.pick(lg.data(), 0) == kCodecOffset + 5);
  EXPECT_TRUE(p.pick(lg.data(), 2) == kMusicEnd);

  // The windowed penalty: code 5 repeated twice in the window drops below
  // code 6 (3.0 / 1.2^2 = 2.08 < 2.5); outside the window it does not.
  s.min_tokens = 0;
  s.repetition_penalty = 1.2f;
  s.penalty_window = 2;
  lg[0] = -9.0f;
  lg[1 + 6] = 2.5f;
  TokenPicker q(Phase::kSemantic, s, 1, false);
  q.accept(kCodecOffset + 5);
  q.accept(kCodecOffset + 5);
  EXPECT_TRUE(q.pick(lg.data(), 0) == kCodecOffset + 6);
  q.accept(kCodecOffset + 7);
  q.accept(kCodecOffset + 8);           // code 5 has left the window
  EXPECT_TRUE(q.pick(lg.data(), 0) == kCodecOffset + 5);

  // The score phase never picks a special between <|endoftext|> and
  // </abc>, however large its logit.
  Sampling sa = Sampling::abc_default();
  sa.temperature = 0.0f;
  sa.min_tokens = 0;
  std::vector<float> la((std::size_t)a.rows, 0.0f);
  la[kEod] = 50.0f;
  la[kAbcStart] = 40.0f;
  la[100] = 1.0f;
  TokenPicker r(Phase::kAbc, sa, 1, false);
  EXPECT_TRUE(r.pick(la.data(), 0) == 100);

  // A sampled pick stays inside top-k, and is reproducible for a seed.
  Sampling ss = Sampling::semantic_default();
  ss.min_tokens = 0;
  ss.top_k = 3;
  std::vector<float> ls((std::size_t)m.rows, -5.0f);
  ls[1 + 10] = 4.0f;
  ls[1 + 11] = 3.9f;
  ls[1 + 12] = 3.8f;
  ls[1 + 13] = 3.7f;
  TokenPicker u1(Phase::kSemantic, ss, 42, false);
  TokenPicker u2(Phase::kSemantic, ss, 42, false);
  bool in_k = true, same = true;
  for (int i = 0; i < 64; ++i) {
    const std::int32_t x = u1.pick(ls.data(), i);
    const std::int32_t y = u2.pick(ls.data(), i);
    in_k = in_k && x >= kCodecOffset + 10 && x <= kCodecOffset + 12;
    same = same && x == y;
  }
  EXPECT_TRUE(in_k);
  EXPECT_TRUE(same);

  // CFG rounds after every operation, as the reference's bf16 tensors do.
  const float c = 1.234f, uu = 0.5f;
  float o = 0.0f;
  cfg_combine(&c, &uu, 1.01f, &o, 1);
  const float d = bf16_round(c - uu);
  EXPECT_TRUE(o == bf16_round(uu + bf16_round(1.01f * d)));

  // Chunks: one when the song fits, the reference's cut when it does not.
  auto r1 = chunk_ranges(5000, 2000);
  EXPECT_TRUE(r1.size() == 1 && r1[0].first == 0 && r1[0].second == 5000);
  auto r2 = chunk_ranges(9000, 8000);
  EXPECT_TRUE(r2.size() == 2 && r2[0].second == (24576 - 8003) / 2);

  // The noise is a standard normal.
  const std::vector<float> n = torch_cpu_randn(7, 64 * 1000);
  double mean = 0.0, var = 0.0;
  for (float x : n) { mean += x; }
  mean /= (double)n.size();
  for (float x : n) { var += ((double)x - mean) * ((double)x - mean); }
  var /= (double)n.size();
  EXPECT_TRUE(std::fabs(mean) < 0.02 && std::fabs(var - 1.0) < 0.03);
}

// ---- the text side, against the reference ------------------------------

TEST(yue2, tokenizer_and_prefixes_match_reference)
{
  const std::string mp = env_("VPIPE_YUE2_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_YUE2_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  const fs::path model(mp), golden(gp);
  auto tok = tokenizer_(model);
  ASSERT_TRUE(tok != nullptr);
  if (tok == nullptr) { return; }
  EXPECT_TRUE(tok->vocab_size() == 151643 + 208);
  EXPECT_TRUE(tok->special_token_id("<|endoftext|>") == yue2::kEod);
  EXPECT_TRUE(tok->special_token_id("<abc>") == yue2::kAbcStart);
  EXPECT_TRUE(tok->special_token_id("</abc>") == yue2::kAbcEnd);

  // Token for token on every case, the full request text included.
  const FlexData cases = read_json_(golden / "tok_cases.json");
  ASSERT_TRUE(cases.is_array());
  if (!cases.is_array()) { return; }
  int ok = 0, n = 0;
  for (const FlexData& c : cases.as_array()) {
    const FlexData fc = c;
    auto o = fc.as_object();
    const std::string text(o.at("text").as_string());
    std::vector<std::int32_t> want;
    const FlexData ids = o.at("ids");
    for (const FlexData& v : ids.as_array()) {
      want.push_back((std::int32_t)v.as_int());
    }
    const std::vector<std::int32_t> got = tok->encode(text);
    ++n;
    if (got == want) {
      ++ok;
    } else {
      report_ids_(("tok case " + std::to_string(n)).c_str(), got, want);
    }
    // And back: ordinary ids decode to the text (NFC aside).
    EXPECT_TRUE(tok->decode(got) == yue2::nfc(text));
  }
  std::printf("[yue2] tokenizer: %d/%d cases token-exact\n", ok, n);
  EXPECT_TRUE(ok == n);

  // The three prefixes the golden decoded from, and the CFG branch.
  Demo d;
  ASSERT_TRUE(load_demo_(model, golden, &d));
  yue2::Request r;
  r.style = d.style;
  r.lyrics = d.lyrics;
  r.seed = d.seed;
  r.cot = yue2::Cot::kFull;
  EXPECT_TRUE(yue2::token_prefix(r, *tok, nullptr) ==
              read_raw_<std::int32_t>(golden / "prefix_abc.i32"));
  r.abc = d.abc;
  EXPECT_TRUE(yue2::token_prefix(r, *tok, nullptr) ==
              read_raw_<std::int32_t>(golden / "prefix_sem.i32"));
  yue2::Request off = r;
  off.abc.clear();
  off.cot = yue2::Cot::kOff;
  EXPECT_TRUE(yue2::token_prefix(off, *tok, nullptr) ==
              read_raw_<std::int32_t>(golden / "prefix_off.i32"));
  EXPECT_TRUE(yue2::negative_prefix(off, *tok, nullptr) ==
              read_raw_<std::int32_t>(golden / "negative_off.i32"));
  EXPECT_TRUE(off.guidance() == 1.01f && r.guidance() == 1.0f);

  // The flow matching's noise is the reference's, bit for bit.
  const std::vector<float> want = read_raw_<float>(golden / "noise.f32");
  const std::vector<float> got = yue2::torch_cpu_randn(d.seed, want.size());
  ASSERT_TRUE(!want.empty());
  EXPECT_TRUE(std::memcmp(got.data(), want.data(), want.size() * 4) == 0);
}

// ---- the AR decode ---------------------------------------------------------

// Two holdings of the AR stack, against the same reference:
//
//   bf16  as published: logits within 1.5x the reference's own bf16
//         error, greedy token-exact through every mask and penalty
//   w8    its projections built affine w8 group-64 in memory
//         (Config::ar_quant_bits): logits held to a quantization bar, and
//         every fp32-DECISIVE pick still agrees. Free greedy runs are
//         reported, not held: a near-tie may land either side.
TEST(yue2, ar_matches_reference)
{
  const std::string mp = env_("VPIPE_YUE2_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_YUE2_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  const fs::path golden(gp);
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  for (int qb : {0, 8}) {
    const bool w8 = qb == 8;
    const char* arm = w8 ? "w8" : "bf16";
    std::string err;
    MetalYue2Model::Config mcfg;
    ASSERT_TRUE(MetalYue2Model::config_from_dir(mp, &mcfg, &err));
    mcfg.ar_quant_bits = qb;
    auto m = MetalYue2Model::load(WeightSet::open(mp, nullptr), mc, mcfg, &err);
    if (m == nullptr) { std::printf("[yue2] load: %s\n", err.c_str()); }
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    EXPECT_TRUE(m->ar_quantized() == w8);
    // The figure the stage books before loading is what the model holds
    // (the host-widened time embedder is the only difference, ~9 MB).
    const std::size_t plan = MetalYue2Model::weight_bytes(mp, qb);
    std::printf("[yue2] AR %s: %zu MB of weights held, %zu MB planned\n",
                arm, m->resident_bytes() >> 20, plan >> 20);
    EXPECT_TRUE(plan >= m->resident_bytes() &&
                plan - m->resident_bytes() < ((std::size_t)16 << 20));

    // Logits at the end of each prefix, over each phase's window.
    struct L { const char* prefix; const char* logits; yue2::Phase ph; };
    for (const L& c : {L{"prefix_abc.i32", "logits_abc.f32", yue2::Phase::kAbc},
                       L{"prefix_sem.i32", "logits_sem.f32",
                         yue2::Phase::kSemantic}}) {
      const auto pre = read_raw_<std::int32_t>(golden / c.prefix);
      const auto want = read_raw_<float>(golden / c.logits);
      std::string f32 = c.logits;
      f32.replace(f32.find(".f32"), 4, "_fp32.f32");
      const auto exact = read_raw_<float>(golden / f32);
      std::vector<float> got;
      ASSERT_TRUE(m->prefill_logits(pre, &got, &err));
      ASSERT_TRUE(got.size() == want.size() && exact.size() == want.size());
      if (got.size() != want.size() || exact.size() != want.size()) { return; }
      // The yardstick is the reference's OWN bf16 error against its fp32
      // run: the music window's logits are small, so both sit near 2%.
      const yue2::Window w = yue2::phase_window(c.ph);
      const std::size_t o = (std::size_t)w.row0, n = (std::size_t)w.rows;
      const double ref = rel_l2_(want.data() + o, exact.data() + o, n);
      const double ours = rel_l2_(got.data() + o, exact.data() + o, n);
      const double vs_ref = rel_l2_(got.data() + o, want.data() + o, n);
      std::printf("[yue2] %s %s: vs fp32 %.4g (the reference's bf16: %.4g), "
                  "vs the bf16 reference %.4g\n", arm, c.logits, ours, ref,
                  vs_ref);
      EXPECT_TRUE(ours < (w8 ? 2.0 : 1.5) * ref + 1e-3);
    }

    // Greedy, token for token, through the masks and the windowed penalty.
    auto greedy = [&](const char* prefix, yue2::Phase ph, int n,
                      const std::vector<std::int32_t>* neg, float cfg,
                      bool legacy) {
      yue2::Sampling s = ph == yue2::Phase::kAbc
                             ? yue2::Sampling::abc_default()
                             : yue2::Sampling::semantic_default();
      s.temperature = 0.0f;
      s.min_tokens = n;
      s.max_tokens = n;
      MetalYue2Model::DecodeResult res;
      const auto pre = read_raw_<std::int32_t>(golden / prefix);
      EXPECT_TRUE(m->decode(pre, ph, s, 1, neg, cfg, legacy, {}, &res, &err));
      return res;
    };
    {
      const auto want = read_raw_<std::int32_t>(golden / "abc_greedy.i32");
      const auto res = greedy("prefix_abc.i32", yue2::Phase::kAbc,
                              (int)want.size(), nullptr, 1.0f, false);
      report_ids_(w8 ? "abc greedy (w8)" : "abc greedy", res.ids, want);
      if (!w8) { EXPECT_TRUE(res.ids == want); }
      std::printf("[yue2] abc %s: %zu tokens in %.2f s (%.1f tok/s, prefill "
                  "%.2f s)\n", arm, res.ids.size(), res.seconds,
                  res.seconds > 0 ? (double)res.ids.size() / res.seconds : 0.0,
                  res.prefill_seconds);
    }
    {
      // The music phase is full of near-ties (top-2 margins of a few
      // hundredths are common, and the reference's own bf16 and fp32 runs
      // disagree inside 48 tokens), so a free run is reported, not held.
      // What IS held: teacher-forced along the reference's path, every pick
      // whose fp32 margin is decisive matches.
      const auto want = read_raw_<std::int32_t>(golden / "sem_greedy.i32");
      const auto res = greedy("prefix_sem.i32", yue2::Phase::kSemantic,
                              (int)want.size(), nullptr, 1.0f, false);
      report_ids_("semantic greedy (free run)", res.ids, want);
      const auto margin = read_raw_<float>(golden / "sem_tf_margin_fp32.f32");
      const auto top = read_raw_<std::int32_t>(golden / "sem_tf_top_fp32.i32");
      ASSERT_TRUE(margin.size() == want.size() && top.size() == want.size());
      yue2::Sampling s = yue2::Sampling::semantic_default();
      s.temperature = 0.0f;
      s.min_tokens = (int)want.size();
      s.max_tokens = (int)want.size();
      std::vector<std::int32_t> picks;
      const auto pre = read_raw_<std::int32_t>(golden / "prefix_sem.i32");
      ASSERT_TRUE(m->decode_forced(pre, yue2::Phase::kSemantic, s, want,
                                   &picks, &err));
      int decided = 0, agree = 0, close_agree = 0, close = 0;
      for (std::size_t k = 0; k < picks.size() && k < want.size(); ++k) {
        if (margin[k] >= 0.25f && top[k] == want[k]) {
          ++decided;
          if (picks[k] == want[k]) { ++agree; }
          else {
            std::printf("[yue2] step %zu: got %d want %d (margin %.3f)\n", k,
                        picks[k], want[k], margin[k]);
          }
        } else {
          ++close;
          if (picks[k] == want[k]) { ++close_agree; }
        }
      }
      std::printf("[yue2] semantic teacher-forced %s: %d/%d decided picks "
                  "agree; %d/%d near-ties agree\n", arm, agree, decided,
                  close_agree, close);
      EXPECT_TRUE(decided >= 24 && agree == decided);
    }
    {
      const auto want = read_raw_<std::int32_t>(golden / "off_cfg_greedy.i32");
      const auto neg = read_raw_<std::int32_t>(golden / "negative_off.i32");
      const auto res = greedy("prefix_off.i32", yue2::Phase::kSemantic,
                              (int)want.size(), &neg, 1.01f, true);
      report_ids_(w8 ? "cot=off CFG greedy (w8)" : "cot=off CFG greedy",
                  res.ids, want);
      if (!w8) { EXPECT_TRUE(res.ids == want); }
    }
  }
}

// ---- the flow matching -----------------------------------------------------

// The flow matching attends the AR stack's K/V, so it is held for both
// AR holdings (see ar_matches_reference): bf16 within 1.5x the
// reference's own bf16 error, w8 within a quantization bar.
TEST(yue2, nar_matches_reference)
{
  const std::string mp = env_("VPIPE_YUE2_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_YUE2_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  const fs::path golden(gp);
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  for (int qb : {0, 8}) {
    const bool w8 = qb == 8;
    const char* arm = w8 ? "w8 AR" : "bf16 AR";
    std::string err;
    MetalYue2Model::Config mcfg;
    ASSERT_TRUE(MetalYue2Model::config_from_dir(mp, &mcfg, &err));
    mcfg.ar_quant_bits = qb;
    auto m = MetalYue2Model::load(WeightSet::open(mp, nullptr), mc, mcfg,
                                  &err);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    std::printf("[yue2] NAR routes: %s GEMMs, %s attention\n",
                m->uses_mma() ? "matmul2d" : "steel",
                m->uses_attn_nax() ? "NAX" : "steel");
    // ENGAGEMENT: both matrix-core routes wherever there are matrix cores,
    // unless the A/B switches name them off.
    EXPECT_TRUE(m->uses_mma() == (mc->supports_matrix_cores() &&
                                  std::getenv("VPIPE_YUE2_NO_MMA2") ==
                                      nullptr));
    EXPECT_TRUE(m->uses_attn_nax() ==
                (mc->supports_matrix_cores() &&
                 std::getenv("VPIPE_YUE2_NO_NAX_ATTN") == nullptr));

    const auto ar = read_raw_<std::int32_t>(golden / "ar_tokens.i32");
    const auto noise = read_raw_<float>(golden / "noise.f32");
    ASSERT_TRUE(!ar.empty() && !noise.empty());
    struct V { double raw; const char* file; };
    for (const V& c : {V{20.0, "nar_v_t1.f32"}, V{0.0, "nar_v_t05.f32"}}) {
      std::vector<float> v;
      ASSERT_TRUE(m->velocity(ar, noise, c.raw, &v, &err));
      const auto want = read_raw_<float>(golden / c.file);
      std::string f32 = c.file;
      f32.replace(f32.find(".f32"), 4, "_fp32.f32");
      const auto exact = read_raw_<float>(golden / f32);
      ASSERT_TRUE(v.size() == want.size() && exact.size() == want.size());
      if (v.size() != want.size() || exact.size() != want.size()) { return; }
      const double ref = rel_l2_(want.data(), exact.data(), v.size());
      const double ours = rel_l2_(v.data(), exact.data(), v.size());
      std::printf("[yue2] %s velocity raw_t=%g: vs fp32 %.4g (the "
                  "reference's bf16: %.4g), vs the bf16 reference %.4g\n",
                  arm, c.raw, ours, ref,
                  rel_l2_(v.data(), want.data(), v.size()));
      EXPECT_TRUE(ours < (w8 ? 2.0 : 1.5) * ref + 1e-3);
    }

    // The whole ODE from the song's own noise (drawn here, not read): the
    // prefix and codes are the golden's sampled run.
    const auto prefix = read_raw_<std::int32_t>(golden / "prefix_sem.i32");
    const auto codec = read_raw_<std::int32_t>(golden / "codec.i32");
    const FlexData meta = read_json_(golden / "meta.json");
    ASSERT_TRUE(meta.is_object());
    if (!meta.is_object()) { return; }
    auto mo = meta.as_object();
    const std::uint64_t seed = (std::uint64_t)mo.at("seed").as_int();
    struct S { int steps; const char* file; double tol; };
    for (const S& c : {S{4, "nar_latents_s4.f32", 0.03},
                       S{32, "nar_latents.f32", 0.05}}) {
      std::vector<float> lat;
      int calls = 0;
      ASSERT_TRUE(m->synthesize(prefix, codec, seed, c.steps, &lat,
                                [&](int, int) { ++calls; return true; },
                                &err));
      EXPECT_TRUE(calls == 2 * c.steps);
      const auto want = read_raw_<float>(golden / c.file);
      ASSERT_TRUE(lat.size() == want.size());
      if (lat.size() != want.size()) { return; }
      const double rl = rel_l2_(lat.data(), want.data(), lat.size());
      std::printf("[yue2] %s latents (%d steps) rel-L2 %.4g\n", arm,
                  c.steps, rl);
      EXPECT_TRUE(rl < (w8 ? 2.0 : 1.0) * c.tol);
    }
  }
}

// ---- the VAE ----------------------------------------------------------------

TEST(yue2, vae_decode_matches_reference)
{
  const std::string vp = env_("VPIPE_YUE2_VAE_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_YUE2_GOLDEN");
  if (vp.empty() || gp.empty()) { return; }
  const fs::path golden(gp);
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto vae = MetalOobleckDecoder::load(vp, mc, &err);
  if (vae == nullptr) { std::printf("[yue2] vae: %s\n", err.c_str()); }
  ASSERT_TRUE(vae != nullptr);
  if (vae == nullptr) { return; }
  EXPECT_TRUE(vae->config().hop() == 1920 &&
              vae->config().sample_rate == 48000);
  // ENGAGEMENT: the matrix-core route wherever there are matrix cores.
  EXPECT_TRUE(vae->uses_mma() == (mc->supports_matrix_cores() &&
                                  std::getenv("VPIPE_YUE2_VAE_NO_MMA2") ==
                                      nullptr));
  std::printf("[yue2] VAE GEMMs on %s\n",
              vae->uses_mma() ? "matmul2d" : "steel");
  EXPECT_TRUE(vae->natural_length(200) == 1920 * 200 - 64);

  // The golden's latents are [T, 64]; the decoder takes [64, T].
  const auto lat = read_raw_<float>(golden / "nar_latents.f32");
  const int Z = 64, T = (int)(lat.size() / Z);
  ASSERT_TRUE(T == 200);
  std::vector<float> z((std::size_t)Z * T);
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < Z; ++c) {
      z[(std::size_t)c * T + t] = lat[(std::size_t)t * Z + c];
    }
  }
  const auto want = read_raw_<float>(golden / "vae_audio.f32");
  std::vector<float> whole;
  // One tile, then 64-frame tiles: four seams the halo must make exact, as
  // it does in the reference (whose full and tiled decodes are identical).
  for (int core : {256, 64}) {
    vae->set_core_frames(core);
    std::vector<float> pcm;
    int tiles = 0;
    ASSERT_TRUE(vae->decode(z.data(), T, &pcm, /*clamp=*/false,
                            [&](int, int) { ++tiles; return true; }, &err));
    ASSERT_TRUE(pcm.size() == want.size());
    if (pcm.size() != want.size()) { return; }
    const double rl = rel_l2_(pcm.data(), want.data(), pcm.size());
    const std::size_t n = want.size() / 2;
    const double rl_l = rel_l2_(pcm.data(), want.data(), n);
    const double rl_r = rel_l2_(pcm.data() + n, want.data() + n, n);
    std::printf("[yue2] VAE decode rel-L2 %.4g (L %.4g, R %.4g) over %d "
                "tile(s)\n", rl, rl_l, rl_r, tiles);
    EXPECT_TRUE(rl < 5e-3);
    if (whole.empty()) {
      whole = pcm;
    } else {
      EXPECT_TRUE(tiles == 4);
      // Exact at the seams; not bit-identical across tile SIZES on an M5,
      // where a small tile's GEMMs fall under matmul2d's row gate and run
      // steel -- the two kernels round differently.
      const double seam = rel_l2_(pcm.data(), whole.data(), pcm.size());
      std::printf("[yue2] tiled vs whole %.3g\n", seam);
      EXPECT_TRUE(seam < (vae->uses_mma() ? 1e-3 : 1e-6));
    }
  }
}

// ---- the stages ----------------------------------------------------------

namespace {

// Keeps the first PCM beat's geometry and peak.
class Yue2PcmSink : public TypedStage<Yue2PcmSink> {
public:
  static constexpr const char* kTypeName = "ut-yue2-pcm-sink";
  using TypedStage::TypedStage;
  int beats = 0, channels = 0, rate = 0;
  std::int64_t samples = 0;
  double peak = 0.0, rms = 0.0;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    const auto* t = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (t == nullptr || t->dtype != TensorBeat::DType::F32 ||
        t->shape.size() != 2) {
      co_return;
    }
    ++beats;
    channels = (int)t->shape[0];
    samples = t->shape[1];
    const float* d = t->as_f32();
    double sq = 0.0;
    for (std::int64_t i = 0; i < t->element_count(); ++i) {
      peak = std::max(peak, (double)std::fabs(d[i]));
      sq += (double)d[i] * d[i];
    }
    rms = std::sqrt(sq / (double)std::max<std::int64_t>(1,
                                                       t->element_count()));
    if (t->sideband.is_object()) {
      auto sb = t->sideband.as_object();
      if (sb.contains("sample_rate")) {
        rate = (int)sb.at("sample_rate").as_int(0);
      }
    }
    co_return;
  }
};

class Yue2TextSink : public TypedStage<Yue2TextSink> {
public:
  static constexpr const char* kTypeName = "ut-yue2-text-sink";
  using TypedStage::TypedStage;
  std::string text;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    if (const auto* f = dynamic_cast<const FlexDataPayload*>(p.get())) {
      if (f->data.is_string()) { text = std::string(f->data.as_string()); }
    }
    co_return;
  }
};

}  // namespace

TEST(yue2, generate_audio_stage_surface)
{
  Session sess;
  EXPECT_TRUE(StageRegistry::get().spec("generate-audio") != nullptr);
  GenerateAudioStage st(&sess, "ga", {}, FlexData::make_object());
  // hf_dir is required, but only at launch.
  EXPECT_TRUE(st.spec().iports.size() == 1 && st.spec().oports.size() == 2);
  FlexData bad = FlexData::make_object();
  bad.as_object().insert("hf_dir", FlexData::make_string("x"));
  bad.as_object().insert("cot", FlexData::make_string("sideways"));
  GenerateAudioStage st2(&sess, "ga2", {}, std::move(bad));
  EXPECT_TRUE(st2.songs_emitted() == 0);
  // lm_quant: w8 (the default) or bf16, nothing else.
  for (const char* q : {"w8", "bf16", "w4"}) {
    FlexData c = FlexData::make_object();
    c.as_object().insert("hf_dir", FlexData::make_string("x"));
    c.as_object().insert("lm_quant", FlexData::make_string(q));
    GenerateAudioStage st3(&sess, "ga3", {}, std::move(c));
    EXPECT_TRUE(st3.config_error().empty() == (std::string(q) != "w4"));
  }
}

// WHAT A SONG HOLDS, AS THE PLAN BOOKS IT -- no model needed: the figures
// come from the config (here the defaults, as for a checkpoint with no
// readable config) and the claims are what declare_resources() returns.
//
//   held          the AR pool (pages of 256 tokens, capacity grown by
//                 DOUBLING and kept) + the NAR row scratch, UNPHASED --
//                 neither is ever given back, so both are still there
//                 while the VAE decodes
//   ode           the flow matching's chunk K/V, denoise only
//   audio-decode  the downstream Oobleck arena, decode-audio
//   pcm           the PCM it hands on, decode-audio
//
// At the protocol's 9000-token cap behind a planned score the song phase
// holds 56 + 1 pages: capacity 64, 1879 MB. cot=off plans no score but
// asks for guidance (1.01), so it holds two shorter contexts, 82 pages:
// capacity 128, exactly 64 pages (1879 MB) more. The stage had booked a
// single context at a flat 112 KB a token, ~1.3 GB, for the denoise only.
// MEASURED against a real song (120 s, cot=off, M4 Pro): 1060 MB held,
// the figure this formula gives for that prompt's length.
TEST(yue2, generate_audio_declares_what_a_song_holds)
{
  Session sess;
  auto make = [&](const char* id, const char* cot, double max_s) {
    FlexData cfg = FlexData::make_object();
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string("/nonexistent/yue2-plan"));
    o.insert("cot", FlexData::make_string(cot));
    o.insert("max_seconds", FlexData::make_real(max_s));
    return std::make_unique<GenerateAudioStage>(
        &sess, id, std::vector<InEdge>{}, std::move(cfg));
  };
  const std::size_t page = 256ull * 28 * 2 * 8 * 128 * 2;   // 29.4 MB
  auto full = make("ga-full", "full", 0.0);
  auto off = make("ga-off", "off", 0.0);
  auto shortsong = make("ga-short", "full", 60.0);
  const auto f = full->planned_song_bytes();
  const auto g = off->planned_song_bytes();
  const auto sh = shortsong->planned_song_bytes();
  std::printf("[yue2] plan: held %zu MB (guided %zu, 60 s %zu) | ode %zu MB "
              "| audio decode %zu MB, pcm %zu MB\n", f.held >> 20,
              g.held >> 20, sh.held >> 20, f.transient >> 20,
              f.decode >> 20, f.pcm >> 20);
  // 9000 frames + the 5120-token plan prefix: 57 pages -> capacity 64.
  const std::size_t rows = (std::size_t)(9000 + 2) *
                           ((2 * 64 + 3 * 2048 + 4 * 2048 + 2 * 1024 +
                             2 * 6144) * 2);
  EXPECT_TRUE(f.held == 64 * page + rows);
  EXPECT_TRUE(g.held - f.held == 64 * page);   // the second context
  // 60 s = 1500 frames: 26 pages -> capacity 32, and far fewer rows.
  EXPECT_TRUE(sh.held < f.held && sh.held >= 32 * page);
  // The chunk: prefix + codes + END + START + latents + END, every layer.
  EXPECT_TRUE(f.transient == (std::size_t)(5120 + 9000 + 1 + 9002) *
                                 (page / 256));
  // YuE2-Vae at its 256-frame tile; 48 kHz stereo f32 for 9000 frames.
  EXPECT_TRUE(f.decode == 5ull * 288 * 1920 * 64 * 2 + (32ull << 20));
  EXPECT_TRUE(f.pcm == 9000ull * 1920 * 2 * 4);

  // The claims: one label per INSTANCE, each lifetime as what it is.
  const auto claims = full->declare_resources();
  int seen = 0;
  for (const ResourceClaim& c : claims) {
    if (c.kind != "activation-scratch") { continue; }
    const std::string label = c.key.substr(0, c.key.find('|'));
    if (label == "generate-audio:ga-full/held") {
      EXPECT_TRUE(c.phase.empty());
      ++seen;
    } else if (label == "generate-audio:ga-full/ode") {
      EXPECT_TRUE(c.phase == "denoise");
      ++seen;
    } else if (label == "generate-audio:ga-full/audio-decode") {
      EXPECT_TRUE(c.phase == "decode-audio");
      ++seen;
    } else if (label == "generate-audio:ga-full/pcm") {
      EXPECT_TRUE(c.phase == "decode-audio" &&
                  c.last_phase == "decode-audio");
      ++seen;
    } else {
      std::printf("[yue2] unexpected scratch claim '%s'\n", c.key.c_str());
      EXPECT_TRUE(false);
    }
  }
  EXPECT_TRUE(seen == 4);
}

// One short song from the stage's config, through audio-vae-decode: the
// whole graph a user wires, with the score handed in so the run is
// seconds rather than minutes.
TEST(yue2, generate_audio_end_to_end)
{
  const std::string mp = env_("VPIPE_YUE2_TEST_MODEL_PATH");
  const std::string vp = env_("VPIPE_YUE2_VAE_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_YUE2_GOLDEN");
  if (mp.empty() || vp.empty() || gp.empty()) { return; }
  Demo d;
  ASSERT_TRUE(load_demo_(mp, gp, &d));
  Session sess;
  Pipeline pl("yue2-e2e", &sess);
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(mp));
    o.insert("style", FlexData::make_string(d.style));
    o.insert("lyrics", FlexData::make_string(
                           "[Verse]\nCity lights are calling me\n"
                           "[Chorus]\nTonight we never sleep\n"));
    o.insert("abc", FlexData::make_string(d.abc));
    o.insert("max_seconds", FlexData::make_real(10.0));
    o.insert("ode_steps", FlexData::make_int(8));
    o.insert("seed", FlexData::make_int(7));
  }
  auto* gen = static_cast<GenerateAudioStage*>(pl.insert_stage(
      std::make_unique<GenerateAudioStage>(&sess, "gen", std::vector<InEdge>{},
                                           std::move(cfg))));
  FlexData vcfg = FlexData::make_object();
  vcfg.as_object().insert("hf_dir", FlexData::make_string(vp));
  auto* dec = pl.insert_stage(std::make_unique<AudioVaeDecodeStage>(
      &sess, "dec", std::vector<InEdge>{{gen, 0}}, std::move(vcfg)));
  auto* pcm = static_cast<Yue2PcmSink*>(pl.insert_stage(
      std::make_unique<Yue2PcmSink>(&sess, "pcm", std::vector<InEdge>{{dec, 0}},
                                    FlexData::make_object())));
  auto* score = static_cast<Yue2TextSink*>(pl.insert_stage(
      std::make_unique<Yue2TextSink>(&sess, "score",
                                     std::vector<InEdge>{{gen, 1}},
                                     FlexData::make_object())));
  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(gen->songs_emitted() == 1);
  EXPECT_TRUE(score->text == d.abc);
  std::printf("[yue2] e2e: %d beat(s), %d ch @ %d Hz, %lld samples "
              "(%.2f s), peak %.3f rms %.4f\n", pcm->beats, pcm->channels,
              pcm->rate, (long long)pcm->samples,
              pcm->rate > 0 ? (double)pcm->samples / pcm->rate : 0.0,
              pcm->peak, pcm->rms);
  EXPECT_TRUE(pcm->beats == 1 && pcm->channels == 2 && pcm->rate == 48000);
  // The model ends the song itself, but never before the 200-token floor
  // and never past the 10 s cap: 8 s .. 10 s of 48 kHz.
  EXPECT_TRUE(pcm->samples >= 200 * 1920 - 64 &&
              pcm->samples <= 250 * 1920 - 64);
  EXPECT_TRUE(pcm->peak > 0.01 && pcm->peak <= 1.0 && pcm->rms > 1e-3);
  // What the song left held beside the weights -- the AR pool at its
  // high-water capacity and the NAR row scratch -- is inside what the
  // plan booked for a song at this cap.
  const std::size_t held = gen->held_scratch_bytes();
  const std::size_t booked = gen->planned_song_bytes().held;
  std::printf("[yue2] e2e: held beside the weights %zu MB, booked %zu MB\n",
              held >> 20, booked >> 20);
  EXPECT_TRUE(held > 0 && held <= booked);
}

// SUNG LYRICS, HEARD BACK: the listening check as a test. One short song
// per AR holding (lm_quant bf16, then w8), same seed, through the stage,
// the VAE, a downmix, a 16 kHz resample and Qwen3-ASR; held is the share
// of the lyrics' words heard back. The songs are SAMPLED, so the two runs
// are two performances, not one compared bit for bit -- the forced tests
// above are where the holdings are compared. Opt-in (VPIPE_YUE2_ASR_CHECK
// =1, plus VPIPE_QWEN3_ASR_TEST_MODEL_PATH): two songs are minutes.
namespace {

class Yue2Mono : public TypedStage<Yue2Mono> {
public:
  static constexpr const char* kTypeName = "ut-yue2-mono";
  using TypedStage::TypedStage;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    const auto* t = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (t == nullptr || t->dtype != TensorBeat::DType::F32 ||
        t->shape.size() != 2) {
      co_return;
    }
    const std::int64_t C = t->shape[0], N = t->shape[1];
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {1, N};
    out->resize_contiguous((std::size_t)N);
    const float* src = t->as_f32();
    float* dst = out->as_f32();
    for (std::int64_t i = 0; i < N; ++i) {
      float acc = 0.0f;
      for (std::int64_t c = 0; c < C; ++c) { acc += src[c * N + i]; }
      dst[i] = acc / (float)C;
    }
    out->sideband = t->sideband;
    co_await ctx.write(0, std::move(out));
  }
};

class Yue2Transcript : public TypedStage<Yue2Transcript> {
public:
  static constexpr const char* kTypeName = "ut-yue2-transcript";
  using TypedStage::TypedStage;
  std::string text;
  Job process(RuntimeContext& ctx) override
  {
    auto p = co_await ctx.read(0);
    if (!p) { ctx.signal_done(); co_return; }
    if (const auto* f = dynamic_cast<const FlexDataPayload*>(p.get())) {
      if (f->data.is_object() && f->data.as_object().contains("text")) {
        text = std::string(f->data.as_object().at("text").as_string(""));
      }
    }
  }
};

// Lower-case words; section tags ("[Verse]") and Qwen3-ASR's leading
// "language X<asr_text>" dropped.
std::vector<std::string>
lyric_words_(std::string s)
{
  const std::size_t tag = s.find("<asr_text>");
  if (tag != std::string::npos) { s = s.substr(tag + 10); }
  std::vector<std::string> out;
  std::string w;
  bool in_tag = false;
  for (unsigned char ch : s) {
    if (ch == '[') { in_tag = true; continue; }
    if (ch == ']') { in_tag = false; continue; }
    if (in_tag) { continue; }
    if (std::isalnum(ch) || ch == '\'') {
      w.push_back((char)std::tolower(ch));
    } else if (!w.empty()) {
      out.push_back(w);
      w.clear();
    }
  }
  if (!w.empty()) { out.push_back(w); }
  return out;
}

}  // namespace

TEST(yue2, sung_lyrics_transcribe)
{
  const std::string mp = env_("VPIPE_YUE2_TEST_MODEL_PATH");
  const std::string vp = env_("VPIPE_YUE2_VAE_TEST_MODEL_PATH");
  const std::string ap = env_("VPIPE_QWEN3_ASR_TEST_MODEL_PATH");
  if (mp.empty() || vp.empty() || ap.empty() ||
      env_("VPIPE_YUE2_ASR_CHECK").empty()) {
    return;
  }
  const std::string lyrics =
      "[Verse]\nCity lights are calling me\nI can hear the river sing\n"
      "[Chorus]\nTonight we never sleep\nTonight the stars are ours to keep\n";
  const std::vector<std::string> said = lyric_words_(lyrics);
  for (const char* q : {"bf16", "w8"}) {
    Session sess;
    Pipeline pl(std::string("yue2-asr-") + q, &sess);
    FlexData cfg = FlexData::make_object();
    {
      auto o = cfg.as_object();
      o.insert("hf_dir", FlexData::make_string(mp));
      o.insert("style", FlexData::make_string(
                            "Pop, clear female vocal, piano, soft drums"));
      o.insert("lyrics", FlexData::make_string(lyrics));
      o.insert("cot", FlexData::make_string("melody"));
      o.insert("max_seconds", FlexData::make_real(30.0));
      o.insert("seed", FlexData::make_int(20261004));
      o.insert("lm_quant", FlexData::make_string(q));
    }
    auto* gen = static_cast<GenerateAudioStage*>(pl.insert_stage(
        std::make_unique<GenerateAudioStage>(
            &sess, "gen", std::vector<InEdge>{}, std::move(cfg))));
    FlexData vcfg = FlexData::make_object();
    vcfg.as_object().insert("hf_dir", FlexData::make_string(vp));
    auto* dec = pl.insert_stage(std::make_unique<AudioVaeDecodeStage>(
        &sess, "dec", std::vector<InEdge>{{gen, 0}}, std::move(vcfg)));
    // A test-local stage has no spec, so no oports until it is given one.
    auto mono_st = std::make_unique<Yue2Mono>(
        &sess, "mono", std::vector<InEdge>{{dec, 0}}, FlexData::make_object());
    mono_st->allocate_oports(1);
    auto* mono = pl.insert_stage(std::move(mono_st));
    FlexData rcfg = FlexData::make_object();
    rcfg.as_object().insert("output_sample_rate", FlexData::make_int(16000));
    auto* rs = pl.insert_stage(std::make_unique<AudioTemporalResampleStage>(
        &sess, "resample", std::vector<InEdge>{{mono, 0}}, std::move(rcfg)));
    FlexData acfg = FlexData::make_object();
    {
      auto o = acfg.as_object();
      o.insert("hf_dir", FlexData::make_string(ap));
      o.insert("max_new_tokens", FlexData::make_int(256));
      o.insert("pcm_buffer_s", FlexData::make_real(60.0));
    }
    auto* asr = pl.insert_stage(std::make_unique<AudioTranscribeStage>(
        &sess, "asr", std::vector<InEdge>{{rs, 0}}, std::move(acfg)));
    auto* sink = static_cast<Yue2Transcript*>(pl.insert_stage(
        std::make_unique<Yue2Transcript>(&sess, "sink",
                                         std::vector<InEdge>{{asr, 0}},
                                         FlexData::make_object())));
    pl.insert_stage(std::make_unique<Yue2TextSink>(
        &sess, "score", std::vector<InEdge>{{gen, 1}},
        FlexData::make_object()));
    PipelineRuntime rt(&pl, &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    EXPECT_TRUE(gen->songs_emitted() == 1);
    // Recall: the lyrics' words heard, each heard word used once.
    std::vector<std::string> heard = lyric_words_(sink->text);
    int hit = 0;
    for (const std::string& w : said) {
      auto it = std::find(heard.begin(), heard.end(), w);
      if (it != heard.end()) { ++hit; heard.erase(it); }
    }
    const double recall = said.empty() ? 0.0 : (double)hit / said.size();
    std::printf("[yue2] lm_quant %s: %d/%zu lyric words heard (%.2f)\n"
                "  heard '%s'\n", q, hit, said.size(), recall,
                sink->text.c_str());
    EXPECT_TRUE(recall >= 0.5);
  }
}

// ---- the flow-matching tiers -------------------------------------------------
//
// ENGAGEMENT BEFORE ACCURACY: a tier that never ran passes every numeric
// bar. The golden's 202 latent rows are under both floors -- the ANE takes
// whole 2048-row chunks (512 here, from the environment, ON PURPOSE) and
// Sol needs 16 key blocks -- so this runs 1200 latent rows against the
// golden's prefix: 1202 queries over 1995 keys.
// ---- the matrix-core routes and the tiers, WITHOUT a golden -----------------
//
// ENGAGEMENT IS A PROPERTY OF THE WIRING, NOT OF THE REFERENCE: whether a
// route was selected, whether a tier dispatched, over how many layers, and
// whether an exact one is exact. All of that can be asked of the model
// alone -- and it has to be, because the golden the accuracy tests below
// want is written by a Python reference that one machine here has. Behind
// VPIPE_YUE2_GOLDEN these checks skip, and a suite that skips them reads
// green on exactly the boxes where the matrix-core path is the point.
//
// So: synthetic conditioning, each arm measured against its OWN dense run,
// and no fp32 yardstick. What needs the reference stays in
// nar_matches_reference and nar_tiers_engage_and_track_dense.
//
// THE FLOORS, because every tier has one and a geometry under it declines
// while the test still passes:
//   matmul2d  M >= kMmaMinRows (64) and N >= 16. Here M = 1202 rows and
//             the narrowest projection is the latent's 64.
//   i8_gemm   K >= 1024. The model's own depths are 2048 and 6144; the
//             64-deep latent projections decline, which is why the GEMM
//             count is asserted > 0 rather than == every GEMM.
//   Sol       keys in blocks of sol::kBlock (64). L_tot = 1714 keys is 27
//             of them.
//   ANE       one chunk of rows, 2048 by default -- which this geometry is
//             under, so the chunk is lowered rather than making the song
//             long enough to reach it.
TEST(yue2, matrix_core_tiers_engage)
{
  const std::string mp = env_("VPIPE_YUE2_TEST_MODEL_PATH");
  if (mp.empty()) { return; }
  ::setenv("VPIPE_YUE2_ANE_CHUNK", "512", 1);
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto m = MetalYue2Model::load(mp, mc, &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  // THE TWO ROUTES, before any arm: one answer each, and each honours its
  // own A/B switch -- so a bisect with the switch set cannot quietly keep
  // the fast path, and a box without matrix cores cannot quietly claim it.
  const bool cores = mc->supports_matrix_cores();
  EXPECT_TRUE(m->uses_mma() ==
              (cores && std::getenv("VPIPE_YUE2_NO_MMA2") == nullptr));
  EXPECT_TRUE(m->uses_attn_nax() ==
              (cores && std::getenv("VPIPE_YUE2_NO_NAX_ATTN") == nullptr));
  std::printf("[yue2-engage] NAR on %s GEMMs, %s attention\n",
              m->uses_mma() ? "matmul2d" : "steel",
              m->uses_attn_nax() ? "NAX" : "steel");

  // Synthetic conditioning: a text prefix, then codec tokens. That is the
  // shape the sink is read from -- the first codec id ends the prefix --
  // and the CONTENT does not matter to engagement, only the extents.
  constexpr int kText = 512, kCodec = 512, T = 1200;
  const int Z = m->config().latent_dim;
  std::vector<std::int32_t> ar;
  ar.reserve((std::size_t)kText + kCodec);
  for (int i = 0; i < kText; ++i) { ar.push_back(1 + i); }
  for (int i = 0; i < kCodec; ++i) {
    ar.push_back(yue2::kCodecOffset + (i % yue2::kCodecSize));
  }
  const std::vector<float> state =
      yue2::torch_cpu_randn(5, (std::size_t)T * (std::size_t)Z);

  struct Arm {
    const char* name;
    MetalYue2Model::Accel a;
  };
  std::vector<Arm> arms;
  arms.push_back({"dense", {}});
  {
    MetalYue2Model::Accel a;
    a.sol.enabled = true;
    a.sol.tau = -1e9f;                      // keep everything
    arms.push_back({"sol all", a});
  }
  {
    MetalYue2Model::Accel a;
    a.sol.enabled = true;
    a.sol.tau = 0.0f;
    arms.push_back({"sol tau 0", a});
  }
  {
    MetalYue2Model::Accel a;
    a.sage.enabled = true;
    arms.push_back({"sage_attn", a});
  }
  {
    MetalYue2Model::Accel a;
    a.i8_gemm = true;
    arms.push_back({"i8_gemm", a});
  }
  {
    MetalYue2Model::Accel a;
    a.ane_ffn = true;
    a.ane_rows = 0.5f;
    arms.push_back({"ane_ffn", a});
  }

  std::vector<float> dense;
  for (const Arm& arm : arms) {
    ASSERT_TRUE(m->set_accel(arm.a, &err));
    // Twice: the ANE's module compiles at the first and its balancer has
    // a measurement only from the second on.
    std::vector<float> v;
    m->reset_nar_stats();
    ASSERT_TRUE(m->velocity(ar, state, 0.0, &v, &err));
    m->reset_nar_stats();
    ASSERT_TRUE(m->velocity(ar, state, 0.0, &v, &err));
    if (v.size() != (std::size_t)T * (std::size_t)Z) {
      EXPECT_TRUE(false);
      return;
    }
    const auto& st = m->nar_stats();
    if (dense.empty()) { dense = v; }
    const double rl = rel_l2_(v.data(), dense.data(), v.size());
    std::printf("[yue2-engage] %-10s %7.1f ms  vs dense %.4g  ane %lld/%lld "
                "rows  sol %d layers kept %.1f%%  sage %d layers  i8 %lld "
                "gemms\n", arm.name, st.ms, rl, st.ane_rows, st.ffn_rows,
                st.sol_layers, 100.0 * st.sol_kept(), st.sage_layers,
                st.i8_gemms);

    if (arm.a.sol.enabled) {
      EXPECT_TRUE(m->sol_armed());
      // Layer 0 is dense by default; the other 27 route.
      EXPECT_TRUE(st.sol_layers == 27 && st.sol_total > 0);
      EXPECT_TRUE(st.sol_kept() > 0.0 && st.sol_kept() <= 1.0);
      if (arm.a.sol.tau < -1.0f) {
        // Everything kept: the routed path IS the dense attention, to the
        // bit. This is the assertion that separates an integration bug
        // from an approximation, and it needs no reference at all.
        EXPECT_TRUE(st.sol_kept() == 1.0 && rl == 0.0);
      } else {
        EXPECT_TRUE(st.sol_kept() < 1.0 && rl > 0.0);
      }
    }
    if (arm.a.sage.enabled) {
      // A matrix-core instruction: armed where there are cores, and
      // declined -- not failed -- everywhere else.
      EXPECT_TRUE(m->sage_armed() == cores);
      if (m->sage_armed()) {
        EXPECT_TRUE(st.sage_layers == 28 && rl > 0.0 && rl < 0.05);
      } else {
        EXPECT_TRUE(st.sage_layers == 0 && rl == 0.0);
      }
    }
    if (arm.a.i8_gemm) {
      EXPECT_TRUE(m->i8_armed() == cores);
      if (m->i8_armed()) {
        EXPECT_TRUE(st.i8_gemms > 0 && rl > 0.0 && rl < 0.1);
      } else {
        EXPECT_TRUE(st.i8_gemms == 0 && rl == 0.0);
      }
    }
    if (arm.a.ane_ffn) {
      // Runs wherever it is asked, matrix cores or not -- and the rows it
      // took are whole chunks of the 512 set above, out of T + 2.
      EXPECT_TRUE(m->ane_armed() && st.ane_rows > 0);
      EXPECT_TRUE(st.ane_rows % 512 == 0 && st.ane_rows < st.ffn_rows);
      // Exact up to the ANE's fp16 -- which is not the same as small.
      // MEASURED 0.016 here, with 512 of 1202 rows on the ANE, so the
      // error is concentrated in the 43% of rows that went there and the
      // per-row figure is higher. The bound is loose enough for a
      // different split to land under it and tight enough to catch what
      // actually goes wrong: a mis-staged weight or a wrong permutation
      // is O(1), not O(1e-2).
      EXPECT_TRUE(rl < 0.05);
    }
  }
}

// The flow matching shows up on the perf lane -- which is the one claim
// about it a run cannot make for itself. A PerfAuxScope is a no-op until
// profiling is on, so "it compiles" says nothing: this enables profiling,
// runs ONE evaluation, and looks for the dit-denoise begin event on the
// LLM lane. Before the scope existed the lane carried the AR phases and
// the KV gather and then nothing, for the phase that is most of a song.
TEST(yue2, flow_matching_shows_on_the_perf_lane)
{
  const std::string mp = env_("VPIPE_YUE2_TEST_MODEL_PATH");
  if (mp.empty()) { return; }
  Session sess;
  ASSERT_TRUE(sess.enable_profiling(4096).code == 0);
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto m = MetalYue2Model::load(mp, mc, &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const int Z = m->config().latent_dim, T = 300;
  std::vector<std::int32_t> ar;
  for (int i = 0; i < 128; ++i) { ar.push_back(1 + i); }
  for (int i = 0; i < 128; ++i) {
    ar.push_back(yue2::kCodecOffset + i);
  }
  std::vector<float> v;
  ASSERT_TRUE(m->velocity(
      ar, yue2::torch_cpu_randn(5, (std::size_t)T * (std::size_t)Z), 0.0,
      &v, &err));

  const fs::path dump = fs::temp_directory_path() / "yue2_perf_dump.bin";
  ASSERT_TRUE(sess.dump_profiling(dump.string()).code == 0);
  std::ifstream in(dump, std::ios::binary);
  std::ostringstream buf;
  buf << in.rdbuf();
  in.close();
  std::error_code ec;
  fs::remove(dump, ec);
  // Bind the owning FlexData to locals: as_array()/as_object() are VIEWS.
  FlexData root = FlexData::from_binary(buf.str());
  ASSERT_TRUE(root.is_object());
  if (!root.is_object()) { return; }
  FlexData threads_v = root.as_object().at("threads");
  auto threads = threads_v.as_array();
  int dit_begin = 0, dit_end = 0;
  for (std::size_t i = 0; i < threads.size(); ++i) {
    FlexData entry_v = threads.at(i);
    if (!entry_v.is_object()) { continue; }
    auto entry = entry_v.as_object();
    if (!entry.contains("events")) { continue; }
    FlexData events_v = entry.at("events");
    if (!events_v.is_object()) { continue; }
    auto events = events_v.as_object();
    if (!events.contains("type")) { continue; }
    FlexData types_v = events.at("type");
    auto types = types_v.as_array();
    for (std::size_t k = 0; k < types.size(); ++k) {
      const std::uint64_t ty = types.at(k).get_uint();
      if (ty == kPerfLlmDitBegin) { ++dit_begin; }
      if (ty == kPerfLlmDitBegin + 1u) { ++dit_end; }
    }
  }
  std::printf("[yue2-engage] perf lane: %d dit-denoise begin / %d end\n",
              dit_begin, dit_end);
  EXPECT_TRUE(dit_begin == 1 && dit_end == 1);
}

// The decoder's matrix-core route, also without a golden: that it was
// selected, and that it still tiles exactly -- a whole decode against a
// four-tile one, which is the property the halo exists for and the one the
// mma gate relaxed (small tiles fall below it and take steel, so the bound
// is 1e-3 under mma where it is 1e-6 under steel).
TEST(yue2, vae_matrix_core_engages)
{
  const std::string vp = env_("VPIPE_YUE2_VAE_TEST_MODEL_PATH");
  if (vp.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto vae = MetalOobleckDecoder::load(vp, mc, &err);
  if (vae == nullptr) { std::printf("[yue2] vae: %s\n", err.c_str()); }
  ASSERT_TRUE(vae != nullptr);
  if (vae == nullptr) { return; }
  EXPECT_TRUE(vae->uses_mma() ==
              (mc->supports_matrix_cores() &&
               std::getenv("VPIPE_YUE2_VAE_NO_MMA2") == nullptr));
  std::printf("[yue2-engage] VAE GEMMs on %s\n",
              vae->uses_mma() ? "matmul2d" : "steel");

  const int Z = vae->config().latent_dim, T = 200;
  const std::vector<float> lat =
      yue2::torch_cpu_randn(7, (std::size_t)Z * (std::size_t)T);
  std::vector<float> whole;
  for (int core : {256, 64}) {
    vae->set_core_frames(core);
    std::vector<float> pcm;
    int tiles = 0;
    ASSERT_TRUE(vae->decode(lat.data(), T, &pcm, /*clamp=*/false,
                            [&](int, int) { ++tiles; return true; }, &err));
    if (pcm.empty()) { EXPECT_TRUE(false); return; }
    if (whole.empty()) {
      whole = pcm;
      EXPECT_TRUE(tiles == 1);
      continue;
    }
    if (pcm.size() != whole.size()) { EXPECT_TRUE(false); return; }
    const double seam = rel_l2_(pcm.data(), whole.data(), pcm.size());
    std::printf("[yue2-engage] VAE seam %.4g over %d tile(s)\n", seam,
                tiles);
    EXPECT_TRUE(tiles > 1);
    EXPECT_TRUE(seam < (vae->uses_mma() ? 1e-3 : 1e-6));
  }
}

// The SAME arms against the reference's fp32 velocity. Its engagement
// assertions are repeated in matrix_core_tiers_engage above, which needs
// no golden -- this one adds the fp32 distance, which does.
TEST(yue2, nar_tiers_engage_and_track_dense)
{
  const std::string mp = env_("VPIPE_YUE2_TEST_MODEL_PATH");
  const std::string gp = env_("VPIPE_YUE2_GOLDEN");
  if (mp.empty() || gp.empty()) { return; }
  ::setenv("VPIPE_YUE2_ANE_CHUNK", "512", 1);
  const fs::path golden(gp);
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto m = MetalYue2Model::load(mp, mc, &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const auto ar = read_raw_<std::int32_t>(golden / "ar_tokens.i32");
  const int T = 1200;
  const std::vector<float> state = yue2::torch_cpu_randn(5, (std::size_t)T * 64);

  struct Arm {
    const char* name;
    MetalYue2Model::Accel a;
  };
  std::vector<Arm> arms;
  arms.push_back({"dense", {}});
  {
    MetalYue2Model::Accel a;
    a.ane_ffn = true;
    a.ane_rows = 0.5f;
    arms.push_back({"ane_ffn", a});
  }
  for (float tau : {-1e9f, 0.0f, 1.0f}) {
    MetalYue2Model::Accel a;
    a.sol.enabled = true;
    a.sol.tau = tau;
    arms.push_back({tau < -1.0f ? "sol all" : tau == 0.0f ? "sol tau 0"
                                                         : "sol tau 1", a});
  }
  {
    MetalYue2Model::Accel a;
    a.sage.enabled = true;
    arms.push_back({"sage_attn", a});
  }
  {
    MetalYue2Model::Accel a;
    a.i8_gemm = true;
    arms.push_back({"i8_gemm", a});
  }
  // The yardstick: the reference's FP32 velocity at this geometry. Dense
  // bf16 sits some distance from it; an exact tier must not add much, a
  // routed one is reported against the same bar.
  const auto exact = read_raw_<float>(golden / "tier_v_fp32.f32");
  ASSERT_TRUE(exact.size() == (std::size_t)T * 64);
  if (exact.size() != (std::size_t)T * 64) { return; }
  double dense_err = 0.0;
  std::vector<float> dense;
  for (const Arm& arm : arms) {
    ASSERT_TRUE(m->set_accel(arm.a, &err));
    m->reset_nar_stats();
    std::vector<float> v;
    // Twice: the ANE's module compiles at the first, and its balancer
    // only has a measurement from the second on.
    ASSERT_TRUE(m->velocity(ar, state, 0.0, &v, &err));
    m->reset_nar_stats();
    ASSERT_TRUE(m->velocity(ar, state, 0.0, &v, &err));
    const auto& st = m->nar_stats();
    if (dense.empty()) {
      dense = v;
      dense_err = rel_l2_(v.data(), exact.data(), v.size());
    }
    const double rl = rel_l2_(v.data(), dense.data(), v.size());
    const double fe = rel_l2_(v.data(), exact.data(), v.size());
    std::printf("[yue2] %-10s %7.1f ms  vs fp32 %.4g  vs dense %.4g  ane "
                "rows %lld/%lld  sol %d layers kept %.1f%%  sage %d layers\n",
                arm.name, st.ms, fe, rl, st.ane_rows, st.ffn_rows,
                st.sol_layers, 100.0 * st.sol_kept(), st.sage_layers);
    if (arm.a.ane_ffn) {
      EXPECT_TRUE(m->ane_armed() && st.ane_rows > 0);
      // Exact up to the ANE's fp16: no further from fp32 than dense bf16
      // is, give or take its own rounding.
      EXPECT_TRUE(fe < 1.5 * dense_err);
    }
    if (arm.a.sol.enabled) {
      EXPECT_TRUE(m->sol_armed());
      // Layer 0 is dense by default; every other layer routes.
      EXPECT_TRUE(st.sol_layers == 27 && st.sol_total > 0);
      EXPECT_TRUE(st.sol_kept() > 0.0 && st.sol_kept() <= 1.0);
      if (arm.a.sol.tau < -1.0f) {
        // Everything kept: the routed path IS the dense attention, to
        // the bit -- which is what separates an integration bug from an
        // approximation.
        EXPECT_TRUE(st.sol_kept() == 1.0 && rl == 0.0);
      }
    }
    if (arm.a.i8_gemm) {
      // An M5 path too: engaged there, declined (and dense) elsewhere.
      EXPECT_TRUE(m->i8_armed() == mc->supports_matrix_cores());
      if (m->i8_armed()) {
        EXPECT_TRUE(st.i8_gemms > 0 && fe < 0.1);
      } else {
        EXPECT_TRUE(st.i8_gemms == 0 && rl == 0.0);
      }
      std::printf("[yue2] i8_gemm: %lld GEMMs on int8\n", st.i8_gemms);
    }
    if (arm.a.sage.enabled) {
      // The int8 QK is a matrix-core instruction: armed on an M5, and on
      // anything else declined (not failed), running dense.
      EXPECT_TRUE(m->sage_armed() == mc->supports_matrix_cores());
      if (m->sage_armed()) {
        EXPECT_TRUE(st.sage_layers == 28 && rl < 0.05);
      } else {
        EXPECT_TRUE(st.sage_layers == 0 && rl == 0.0);
      }
    }
  }
}

// ---- the tiers on a real song (bench) ----------------------------------------
//
// Gated by VPIPE_YUE2_BENCH_DIR. A whole song's flow matching per arm,
// from the SAME tokens: the first run writes them (the model's tonight-
// awake demo, seed 12300) and later runs reuse them. Reports the per-
// evaluation time and the latents' distance from the dense arm, and, with
// VPIPE_YUE2_VAE_TEST_MODEL_PATH, writes each arm's song as a WAV -- an
// approximation is judged by listening, not by a norm.
//   VPIPE_YUE2_BENCH_STEPS  midpoint steps (default 32, the protocol's)
//   VPIPE_YUE2_BENCH_ARMS   comma list from dense,ane,sol0,sol05,sol1,
//                           ane+sol0, and the M5 arms sage, i8, i8old (i8
//                           with every shape treated as past the deep-K
//                           cliff: untuned S=2, every width tuned),
//                           sage+sol0, i8+sage (default: the first six).
//                           Run in the order listed; a name may repeat, to
//                           interleave an A/B.
namespace {

void
write_wav16_(const fs::path& p, const std::vector<float>& planar, int ch,
             int rate)
{
  const std::size_t n = planar.size() / (std::size_t)ch;
  std::ofstream o(p, std::ios::binary);
  auto u32 = [&](std::uint32_t v) { o.write((const char*)&v, 4); };
  auto u16 = [&](std::uint16_t v) { o.write((const char*)&v, 2); };
  o.write("RIFF", 4); u32((std::uint32_t)(36 + n * ch * 2));
  o.write("WAVEfmt ", 8); u32(16); u16(1); u16((std::uint16_t)ch);
  u32((std::uint32_t)rate); u32((std::uint32_t)(rate * ch * 2));
  u16((std::uint16_t)(ch * 2)); u16(16);
  o.write("data", 4); u32((std::uint32_t)(n * ch * 2));
  for (std::size_t i = 0; i < n; ++i) {
    for (int c = 0; c < ch; ++c) {
      const float v = std::clamp(planar[(std::size_t)c * n + i], -1.0f, 1.0f);
      const std::int16_t s = (std::int16_t)std::lround(v * 32767.0f);
      o.write((const char*)&s, 2);
    }
  }
}

}  // namespace

TEST(yue2, nar_tiers_bench)
{
  const std::string bd = env_("VPIPE_YUE2_BENCH_DIR");
  const std::string mp = env_("VPIPE_YUE2_TEST_MODEL_PATH");
  if (bd.empty() || mp.empty()) { return; }
  const fs::path dir(bd);
  fs::create_directories(dir);
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  std::string err;
  auto m = MetalYue2Model::load(mp, mc, &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  std::vector<std::int32_t> prefix =
      read_raw_<std::int32_t>(dir / "song_prefix.i32");
  std::vector<std::int32_t> codec =
      read_raw_<std::int32_t>(dir / "song_codec.i32");
  if (prefix.empty() || codec.empty()) {
    auto tok = tokenizer_(mp);
    ASSERT_TRUE(tok != nullptr);
    const FlexData ex = read_json_(fs::path(mp) / "examples" /
                                   "tonight-awake.json");
    auto eo = ex.as_object();
    yue2::Request r;
    r.style = std::string(eo.at("style").as_string());
    r.lyrics = std::string(eo.at("lyrics").as_string());
    r.seed = 12300;
    MetalYue2Model::DecodeResult abc, song;
    ASSERT_TRUE(m->decode(yue2::token_prefix(r, *tok, nullptr),
                          yue2::Phase::kAbc, yue2::Sampling::abc_default(),
                          r.seed, nullptr, 1.0f, false, {}, &abc, &err));
    prefix = yue2::token_prefix(r, *tok, &abc.ids);
    ASSERT_TRUE(m->decode(prefix, yue2::Phase::kSemantic,
                          yue2::Sampling::semantic_default(), r.seed, nullptr,
                          1.0f, false, {}, &song, &err));
    codec.clear();
    for (std::int32_t id : song.ids) {
      codec.push_back(id - yue2::kCodecOffset);
    }
    std::ofstream(dir / "song_prefix.i32", std::ios::binary)
        .write((const char*)prefix.data(), (std::streamsize)prefix.size() * 4);
    std::ofstream(dir / "song_codec.i32", std::ios::binary)
        .write((const char*)codec.data(), (std::streamsize)codec.size() * 4);
  }
  std::printf("[yue2-bench] prefix %zu tokens, song %zu frames (%.1f s): "
              "%zu latent rows over %zu keys\n", prefix.size(), codec.size(),
              (double)codec.size() / 25.0, codec.size() + 2,
              prefix.size() + 2 * codec.size() + 3);

  const std::string se = env_("VPIPE_YUE2_BENCH_STEPS");
  const int steps = se.empty() ? 32 : std::max(1, std::atoi(se.c_str()));
  std::string want = env_("VPIPE_YUE2_BENCH_ARMS");
  if (want.empty()) { want = "dense,ane,sol0,sol05,sol1,ane+sol0"; }
  // Arms run in the order LISTED and a name may repeat, so an A/B can be
  // interleaved within one thermal state: dense,i8,i8old,i8old,i8,dense.
  struct Arm {
    std::string name;
    MetalYue2Model::Accel a;
    bool i8_old = false;        // i8 with every K past the cliff
  };
  auto sol = [](float tau) {
    MetalYue2Model::Accel a;
    a.sol.enabled = true;
    a.sol.tau = tau;
    return a;
  };
  auto make_arm = [&](const std::string& n, Arm* arm) {
    arm->name = n;
    arm->a = {};
    MetalYue2Model::Accel& a = arm->a;
    if (n == "dense") { return true; }
    if (n == "ane") { a.ane_ffn = true; return true; }
    if (n == "sol0") { a = sol(0.0f); return true; }
    if (n == "sol05") { a = sol(0.5f); return true; }
    if (n == "sol1") { a = sol(1.0f); return true; }
    if (n == "ane+sol0") { a = sol(0.0f); a.ane_ffn = true; return true; }
    if (n == "sage") { a.sage.enabled = true; return true; }
    if (n == "i8") { a.i8_gemm = true; return true; }
    if (n == "i8old") {
      a.i8_gemm = true;
      arm->i8_old = true;
      return true;
    }
    if (n == "sage+sol0") {
      a = sol(0.0f);
      a.sage.enabled = true;
      return true;
    }
    if (n == "i8+sage") {
      a.i8_gemm = true;
      a.sage.enabled = true;
      return true;
    }
    return false;
  };
  std::vector<Arm> arms;
  for (std::size_t b = 0; b <= want.size();) {
    std::size_t e = want.find(',', b);
    if (e == std::string::npos) { e = want.size(); }
    const std::string n = want.substr(b, e - b);
    Arm arm;
    if (!n.empty() && make_arm(n, &arm)) {
      arms.push_back(arm);
    } else if (!n.empty()) {
      std::printf("[yue2-bench] unknown arm '%s' -- skipped\n", n.c_str());
    }
    b = e + 1;
  }
  std::unique_ptr<MetalOobleckDecoder> vae;
  const std::string vp = env_("VPIPE_YUE2_VAE_TEST_MODEL_PATH");
  if (!vp.empty()) { vae = MetalOobleckDecoder::load(vp, mc, &err); }

  std::vector<float> ref = read_raw_<float>(dir / "lat_dense.f32");
  for (const Arm& arm : arms) {
    // The int8 context reads its split rule at construction, which
    // set_accel does: K >= 0 puts every shape past the cliff, so each
    // untuned shape takes S=2 and the tuner weighs every width. Restore
    // the caller's setting straight after.
    const char* had = std::getenv("VPIPE_I8_SPLIT_MIN_K");
    const std::string saved = had != nullptr ? had : "";
    if (arm.i8_old) { ::setenv("VPIPE_I8_SPLIT_MIN_K", "0", 1); }
    ASSERT_TRUE(m->set_accel(arm.a, &err));
    if (arm.i8_old) {
      if (had != nullptr) {
        ::setenv("VPIPE_I8_SPLIT_MIN_K", saved.c_str(), 1);
      } else {
        ::unsetenv("VPIPE_I8_SPLIT_MIN_K");
      }
    }
    std::vector<float> lat;
    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_TRUE(m->synthesize(prefix, codec, 12300, steps, &lat, {}, &err));
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const auto& st = m->nar_stats();
    if (arm.name == "dense") {
      ref = lat;
      std::ofstream(dir / "lat_dense.f32", std::ios::binary)
          .write((const char*)lat.data(), (std::streamsize)lat.size() * 4);
    }
    const double rl = (!ref.empty() && ref.size() == lat.size())
                          ? rel_l2_(lat.data(), ref.data(), lat.size())
                          : -1.0;
    std::printf("[yue2-bench] %-9s %6.1f s  %7.1f ms/eval  vs dense %.4g  "
                "ane %.1f%% of rows  sol kept %.1f%%  i8 gemms %d\n",
                arm.name.c_str(), secs,
                st.evals > 0 ? st.ms / st.evals : 0.0, rl,
                st.ffn_rows > 0 ? 100.0 * st.ane_rows / st.ffn_rows : 0.0,
                100.0 * st.sol_kept(),
                st.evals > 0 ? (int)(st.i8_gemms / st.evals) : 0);
    if (vae) {
      const int T = (int)(lat.size() / 64);
      std::vector<float> z((std::size_t)64 * T);
      for (int t = 0; t < T; ++t) {
        for (int c = 0; c < 64; ++c) {
          z[(std::size_t)c * T + t] = lat[(std::size_t)t * 64 + c];
        }
      }
      std::vector<float> pcm;
      if (vae->decode(z.data(), T, &pcm, true, {}, &err)) {
        write_wav16_(dir / ("song_" + arm.name + ".wav"), pcm, 2, 48000);
      }
    }
  }
}

// ---- the flow matching's GEMMs one shape at a time (bench) ------------------
//
// Gated by VPIPE_YUE2_BENCH_GEMM (the row count; 1 = the bench song's 5347)
// and by matrix cores. Each of the NAR layer's GEMM shapes, timed alone:
// the dense matmul2d route the model takes, against the int8 context with
// split-K OFF and at each width up to 4 it offers (2 is its untuned
// default). Rounds alternate direction and the median is reported, and the
// per-evaluation total each route implies (28 layers) is printed beside
// the end-to-end bench's ms/eval to compare against.
// VPIPE_YUE2_BENCH_GEMM_SHAPES="N:K,..." times those shapes instead (a K
// sweep for where the split starts to pay); they count toward no total.
// Each shape is then TUNED the way a caller's first forwards tune it --
// recorded by a gemm(), replayed by tune_pending() -- five times with a
// fresh context, printing what it cost and what it chose.
TEST(yue2, gemm_shape_rate) {
  const std::string e = env_("VPIPE_YUE2_BENCH_GEMM");
  if (e.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) {
    std::printf("[yue2-gemm] needs matrix cores -- skip\n");
    return;
  }
  const int M = std::atoi(e.c_str()) > 1 ? std::atoi(e.c_str()) : 5347;
  constexpr int kLayers = 28;
  struct Shape { std::string name; int N, K, per_layer; };
  std::vector<Shape> shapes = {{"q,o", 2048, 2048, 2},
                               {"k,v", 1024, 2048, 2},
                               {"gate,up", 6144, 2048, 2},
                               {"down", 2048, 6144, 1}};
  const std::string want = env_("VPIPE_YUE2_BENCH_GEMM_SHAPES");
  if (!want.empty()) {
    shapes.clear();
    for (std::size_t b = 0; b < want.size();) {
      std::size_t e = want.find(',', b);
      if (e == std::string::npos) { e = want.size(); }
      const std::string one = want.substr(b, e - b);
      const std::size_t c = one.find(':');
      if (c != std::string::npos) {
        shapes.push_back({one, std::atoi(one.c_str()),
                          std::atoi(one.c_str() + c + 1), 0});
      }
      b = e + 1;
    }
  }
  int Kmax = 0, Nmax = 0;
  for (const Shape& sh : shapes) {
    Kmax = std::max(Kmax, sh.K);
    Nmax = std::max(Nmax, sh.N);
  }
  metal_compute::ComputeLibrary lib = mc->load_library("dense_gemm_mma_bf16");
  metal_compute::ComputeFunction fn_n128 =
      lib.function("dense_gemm_mma_t_n128_f16");
  metal_compute::ComputeFunction fn_wide =
      lib.function("dense_gemm_mma_t_n128x256_f16");
  ASSERT_TRUE(fn_n128.valid() && fn_wide.valid());
  I8GemmContext ctx(mc, /*want=*/true, /*bf16=*/true);
  ASSERT_TRUE(ctx.enabled());
  if (!fn_n128.valid() || !fn_wide.valid() || !ctx.enabled()) { return; }

  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto fill_bf16 = [&](metal_compute::SharedBuffer& b, std::size_t n,
                       float scale) {
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const float v = scale * nd(rng);
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      p[i] = (std::uint16_t)(u >> 16);
    }
  };
  metal_compute::SharedBuffer x =
      mc->make_shared_buffer((std::size_t)M * Kmax * 2);
  metal_compute::SharedBuffer y =
      mc->make_shared_buffer((std::size_t)M * Nmax * 2);
  ASSERT_TRUE(!x.empty() && !y.empty());
  if (x.empty() || y.empty()) { return; }
  fill_bf16(x, (std::size_t)M * Kmax, 1.0f);

  // Per-evaluation totals, in ms: dense, then i8 at S = 0, 2.
  double tot_dense = 0.0, tot_s0 = 0.0, tot_s2 = 0.0;
  for (const Shape& sh : shapes) {
    const int N = sh.N, K = sh.K;
    metal_compute::SharedBuffer w =
        mc->make_shared_buffer((std::size_t)N * K * 2);
    ASSERT_TRUE(!w.empty());
    if (w.empty()) { return; }
    fill_bf16(w, (std::size_t)N * K, 0.02f);
    ASSERT_TRUE(M <= mma_row_band(N, K));

    // Arm -1 is dense; the rest are split widths (0 = no split).
    std::vector<int> arms{-1, 0};
    for (int S : ctx.split_candidates(M, N, K)) {
      if (S <= 4) { arms.push_back(S); }
    }
    auto encode = [&](metal_compute::ComputeEncoder& enc, int arm) {
      if (arm < 0) {
        const bool wide = mma_use_wide_tile(N, K);
        const int RN = wide ? 256 : 128;
        enc.set_function(wide ? fn_wide : fn_n128);
        enc.set_buffer(0, x);
        enc.set_buffer(1, w);
        enc.set_buffer(2, w);
        enc.set_buffer(3, y);
        enc.set_constant(4, K);
        enc.set_constant(5, N);
        enc.set_constant(6, M);
        enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                      (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        return true;
      }
      ctx.bypass_split = (arm == 0);
      ctx.force_splits = arm;
      return ctx.gemm(enc, x, 0, w, y, 0, M, N, K);
    };
    constexpr int kIters = 4, kRounds = 5;
    std::vector<std::vector<double>> ms(arms.size());
    for (int r = 0; r < kRounds + 1; ++r) {
      for (std::size_t j = 0; j < arms.size(); ++j) {
        const std::size_t a = (r % 2 == 0) ? j : arms.size() - 1 - j;
        bool ok = true;
        const auto t0 = std::chrono::steady_clock::now();
        {
          metal_compute::CommandStream st = mc->make_command_stream();
          {
            metal_compute::ComputeEncoder enc = st.begin_compute();
            for (int i = 0; i < kIters; ++i) {
              ok = encode(enc, arms[a]) && ok;
            }
          }
          st.commit().wait();
        }
        const double t = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0).count() /
                         kIters;
        EXPECT_TRUE(ok);
        if (r > 0) { ms[a].push_back(t); }   // round 0 warms up
      }
    }
    ctx.bypass_split = false;
    ctx.force_splits = 0;
    // The deferred tuner, as a caller meets it: fresh context, one gemm()
    // records the shape, tune_pending() replays it.
    std::string tuned;
    double tune_ms_max = 0.0;
    int n_cands = 0;
    for (int trial = 0; trial < 5; ++trial) {
      I8GemmContext tc(mc, /*want=*/true, /*bf16=*/true);
      n_cands = (int)tc.tune_candidates(std::min(M, 2048), N, K).size();
      {
        metal_compute::CommandStream st = mc->make_command_stream();
        {
          metal_compute::ComputeEncoder enc = st.begin_compute();
          EXPECT_TRUE(tc.gemm(enc, x, 0, w, y, 0, M, N, K));
        }
        st.commit().wait();
      }
      const auto t0 = std::chrono::steady_clock::now();
      tc.tune_pending(mc);
      const double tms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      tune_ms_max = std::max(tune_ms_max, tms);
      tuned += (trial > 0 ? "," : "") +
               std::to_string(tc.plan_for_test(M, N, K));
    }
    const double gflop = 2.0 * M * (double)N * K * 1e-9;
    std::printf("[yue2-gemm] %-7s M=%d N=%d K=%d (G=%d)  %.1f GFLOP\n",
                sh.name.c_str(), M, N, K, K / std::max(1, ctx.group()),
                gflop);
    double dense_ms = 0.0;
    for (std::size_t a = 0; a < arms.size(); ++a) {
      std::vector<double> v = ms[a];
      std::sort(v.begin(), v.end());
      const double med = v[v.size() / 2];
      if (arms[a] < 0) { dense_ms = med; }
      const std::string label =
          arms[a] < 0 ? std::string("bf16 mma")
                      : "i8 S=" + std::to_string(arms[a]);
      std::printf("[yue2-gemm]   %-9s %7.2f ms  %5.1f TFLOP/s  %.2fx dense\n",
                  label.c_str(), med, gflop / med,
                  med > 0 ? dense_ms / med : 0.0);
      const double per_eval = med * sh.per_layer * kLayers;
      if (arms[a] < 0) { tot_dense += per_eval; }
      if (arms[a] == 0) { tot_s0 += per_eval; }
      if (arms[a] == 2) { tot_s2 += per_eval; }
    }
    std::printf("[yue2-gemm]   tuned S = %s  (%d candidates, tune_pending "
                "<= %.0f ms)\n", tuned.c_str(), n_cands, tune_ms_max);
  }
  std::printf("[yue2-gemm] per evaluation (%d layers): bf16 %.0f ms, "
              "i8 no split %.0f ms, i8 S=2 %.0f ms\n", kLayers, tot_dense,
              tot_s0, tot_s2);
}
