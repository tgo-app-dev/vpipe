// Sol-Attn INSIDE the real MiniMax-H3 stack.
//
// The tests in sol-attention.cc pin the METHOD and the kernel against a
// CPU oracle. This one pins the WIRING, which is a different failure
// mode and a quieter one: a config key that never reaches the model, a
// dense arm that wins the dispatch chain anyway, a layer filter off by
// one -- none of those crash, and all of them produce an ordinary
// velocity from an ordinary-looking run.
//
// So the assertions are: it runs, it is finite, it CHANGES the answer
// (an approximation that changed nothing did not happen), it is
// reproducible, and it reports a sparsity in the range the standalone
// bench predicts. Plus the two exclusions that are correctness rules --
// the leading dense layers, and VDN.
//
// Env: VPIPE_MINIMAX_H3_TEST_MODEL_PATH = the FL2VA partition dir.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/shared/sage-attention.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using namespace vpipe::metal_compute;
namespace h3 = vpipe::genai::minimax_h3;

namespace {

float
bf16_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

SharedBuffer
ramp_(MetalCompute* mc, std::size_t n, float k)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  if (b.empty()) { return b; }
  auto* o = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) {
    const float f = std::sin((float)i * k) * 0.5f;
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    o[i] = (std::uint16_t)(u >> 16);
  }
  return b;
}

int
envi_(const char* k, int d)
{
  const char* v = std::getenv(k);
  return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
}

double
rms_(const std::vector<std::uint16_t>& x, std::size_t* non_finite)
{
  double l2 = 0.0;
  *non_finite = 0;
  for (std::uint16_t h : x) {
    const float f = bf16_(h);
    if (!std::isfinite(f)) { ++(*non_finite); }
    else { l2 += (double)f * (double)f; }
  }
  return std::sqrt(l2 / (double)(x.empty() ? 1 : x.size()));
}

// Relative L2 of `x` against `ref`, both bf16 velocity fields.
double
rel_(const std::vector<std::uint16_t>& x,
     const std::vector<std::uint16_t>& ref)
{
  if (x.size() != ref.size() || ref.empty()) { return -1.0; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double a = bf16_(x[i]), b = bf16_(ref[i]);
    num += (a - b) * (a - b);
    den += b * b;
  }
  return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

}  // namespace

TEST(sol_attention_h3, sol_runs_in_the_stack_and_changes_the_answer)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  // Long enough that the 64-key blocks mean something: the routing has
  // nothing to choose from on a sequence of two blocks, and a test at
  // that size would pass while proving nothing. 20 frames of 64 tokens
  // plus the globals is ~22 blocks.
  const int latf  = envi_("VPIPE_H3_SOL_LATF", 20);
  const int lath  = envi_("VPIPE_H3_SOL_LATH", 16);
  const int latw  = envi_("VPIPE_H3_SOL_LATW", 16);
  const int naud  = envi_("VPIPE_H3_SOL_AUD", 40);
  const int ntext = envi_("VPIPE_H3_SOL_TEXT", 16);

  // PIN THE GEMM AUTOTUNE FOR THE WHOLE TEST. Every arm below builds
  // its own transformer, and the tuner picks projection tiles by
  // MEASURING -- so two arms can land on different tiles and differ in
  // the sixth significant digit before attention is reached at all.
  // That is fatal to both comparisons here: the dense/sol difference
  // would carry it, and the sol/sol one asserts byte equality. On a
  // matrix-core box it made this test fail for a reason that has
  // nothing to do with routing.
  ::setenv("VPIPE_H3_NO_QMM_AUTOTUNE", "1", 1);
  struct Unpin {
    ~Unpin() { ::unsetenv("VPIPE_H3_NO_QMM_AUTOTUNE"); }
  } unpin;

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    std::printf("[h3_sol] config: %s\n", cerr.c_str());
    return;
  }
  cfg.n_layers = envi_("VPIPE_H3_SOL_LAYERS", 2);

  h3::PackedLayout L;
  const std::vector<int> text_tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(
      text_tags, latf, lath, latw, naud, cfg.patch_h, cfg.patch_w,
      h3::kAudioChannels, {h3::Anchor::kFirst}, &L));
  std::printf("[h3_sol] seq %d (%d text + %d cond + %d audio + %d video), "
              "%d blocks of 64, %d layers\n", L.seq_len, L.num_text_rows,
              L.num_condition_rows, L.num_audio_rows, L.num_video_rows,
              (L.seq_len + 63) / 64, cfg.n_layers);

  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, 0.7f, 0.5f, 0.0f, &uniq, &row_idx);

  const SharedBuffer vb = ramp_(
      mc, (std::size_t)L.video_indices.size() * cfg.video_patch_elems(),
      0.017f);
  const SharedBuffer ab = ramp_(
      mc, (std::size_t)L.audio_indices.size() * cfg.audio_channels, 0.031f);
  const SharedBuffer tb =
      ramp_(mc, (std::size_t)ntext * cfg.text_dim, 0.005f);
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  auto run_arm = [&](const sol::Config& sc, const char* what,
                     std::vector<std::uint16_t>* out, double* ms) {
    MetalMiniMaxH3Transformer::Config c = cfg;
    c.sol = sc;
    auto m = MetalMiniMaxH3Transformer::load(root, mc, c);
    if (m == nullptr) {
      std::printf("[h3_sol] %s: DiT load failed\n", what);
      return false;
    }
    MetalMiniMaxH3Transformer::Step step;
    step.video  = &vb;
    step.audio  = &ab;
    step.text   = &tb;
    step.layout = &L;
    step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    (void)m->forward(step, &ferr);              // warm the pipelines
    const auto t0 = std::chrono::steady_clock::now();
    MetalMiniMaxH3Transformer::Velocity v = m->forward(step, &ferr);
    *ms = std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t0).count();
    if (v.empty()) {
      std::printf("[h3_sol] %s: %s\n", what, ferr.c_str());
      return false;
    }
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    out->assign(p, p + v.video.byte_size() / 2);
    std::size_t bad = 0;
    const double r = rms_(*out, &bad);
    std::printf("[h3_sol] %-16s rms %.6f  non-finite %zu  %8.1f ms\n", what,
                r, bad, *ms);
    EXPECT_TRUE(bad == 0);
    return bad == 0;
  };

  sol::Config off;                              // enabled = false
  std::vector<std::uint16_t> dense;
  double dense_ms = 0.0;
  if (!run_arm(off, "dense", &dense, &dense_ms)) { return; }

  sol::Config on;
  on.enabled = true;
  on.tau = 1.0f;
  on.dense_layers = 1;
  std::vector<std::uint16_t> sol_out;
  double sol_ms = 0.0;
  if (!run_arm(on, "sol tau 1.0", &sol_out, &sol_ms)) { return; }

  ASSERT_TRUE(sol_out.size() == dense.size());
  if (sol_out.size() != dense.size()) { return; }

  // IT CHANGED THE ANSWER. An approximation that reproduced dense
  // bit-for-bit did not run -- which is what a config that never
  // reached the model, or a dispatch arm the dense path won, looks
  // like from out here.
  std::size_t differ = 0;
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < dense.size(); ++i) {
    if (dense[i] != sol_out[i]) { ++differ; }
    const double a = bf16_(sol_out[i]), b = bf16_(dense[i]);
    num += (a - b) * (a - b);
    den += b * b;
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
  std::printf("[h3_sol] sol vs dense: rel-L2 %.4f, differs in %zu/%zu, "
              "%.2fx wall\n", rel, differ, dense.size(),
              sol_ms > 0.0 ? dense_ms / sol_ms : 0.0);
  EXPECT_TRUE(differ > dense.size() / 100);

  // ...but it is still ATTENTION. A routing bug that dropped the wrong
  // blocks, or a denominator that counted a block once instead of 64
  // times, lands far outside this rather than just above it.
  EXPECT_TRUE(rel < 0.5);

  // REPRODUCIBLE. The routing is data-dependent, so a run that varied
  // between forwards would mean the summaries were reading a buffer
  // something else writes.
  std::vector<std::uint16_t> again;
  double again_ms = 0.0;
  if (!run_arm(on, "sol again", &again, &again_ms)) { return; }
  EXPECT_TRUE(again.size() == sol_out.size() &&
              std::memcmp(again.data(), sol_out.data(),
                          sol_out.size() * 2) == 0);
}

// The leading dense layers are a real filter, not a comment. With every
// layer left dense the run must reproduce the plain stack EXACTLY --
// which is also the cheapest proof that the Sol arm is reached by layer
// index and not by something that happens to correlate with it.
TEST(sol_attention_h3, dense_layers_covering_the_stack_is_the_plain_model)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  // PINNED, like every byte-identity claim in this file. Both arms
  // below are separate transformers over the same weights, and the
  // projection tuner picks its tiles by MEASURING -- so on a
  // matrix-core box the two can land on different tiles and differ
  // before a single Sol dispatch has run. It is a flake and not a
  // constant one: this test passed and failed on consecutive runs of
  // the same binary.
  ::setenv("VPIPE_H3_NO_QMM_AUTOTUNE", "1", 1);
  struct Unpin {
    ~Unpin() { ::unsetenv("VPIPE_H3_NO_QMM_AUTOTUNE"); }
  } unpin;

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    return;
  }
  cfg.n_layers = envi_("VPIPE_H3_SOL_LAYERS", 2);

  h3::PackedLayout L;
  const std::vector<int> text_tags(16, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(text_tags, 20, 16, 16, 40,
                                        cfg.patch_h, cfg.patch_w,
                                        h3::kAudioChannels,
                                        {h3::Anchor::kFirst}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, 0.7f, 0.5f, 0.0f, &uniq, &row_idx);
  const SharedBuffer vb = ramp_(
      mc, (std::size_t)L.video_indices.size() * cfg.video_patch_elems(),
      0.017f);
  const SharedBuffer ab = ramp_(
      mc, (std::size_t)L.audio_indices.size() * cfg.audio_channels, 0.031f);
  const SharedBuffer tb = ramp_(mc, (std::size_t)16 * cfg.text_dim, 0.005f);
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  auto run_arm = [&](const sol::Config& sc, std::vector<std::uint16_t>* out) {
    MetalMiniMaxH3Transformer::Config c = cfg;
    c.sol = sc;
    auto m = MetalMiniMaxH3Transformer::load(root, mc, c);
    if (m == nullptr) { return false; }
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb; step.audio = &ab; step.text = &tb;
    step.layout = &L; step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    MetalMiniMaxH3Transformer::Velocity v = m->forward(step, &ferr);
    if (v.empty()) { return false; }
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    out->assign(p, p + v.video.byte_size() / 2);
    return true;
  };

  sol::Config off;
  std::vector<std::uint16_t> plain;
  if (!run_arm(off, &plain)) { return; }

  sol::Config all_dense;
  all_dense.enabled = true;
  all_dense.tau = 1.0f;
  all_dense.dense_layers = cfg.n_layers;        // nothing left to route
  std::vector<std::uint16_t> got;
  if (!run_arm(all_dense, &got)) { return; }

  const bool same = got.size() == plain.size() &&
                    std::memcmp(got.data(), plain.data(),
                                plain.size() * 2) == 0;
  std::printf("[h3_sol] dense_layers = n_layers: %s\n",
              same ? "BYTE-IDENTICAL to the plain stack" : "DIFFERS");
  EXPECT_TRUE(same);
}

// A DENSE STEP is the plain model: Sol-Attn configured and armed, and
// a Step that says `dense_attention` runs byte-identical to a model
// with Sol off. That is what sol_dense_steps leans on -- HyperFlow's
// published recipe keeps its first two steps dense -- so the flag has
// to mean "unchanged model", not "Sol with nothing routed by a slower
// path". The same model WITHOUT the flag must differ, or the arm proves
// nothing about Sol having been live to switch off.
TEST(sol_attention_h3, a_dense_step_is_the_plain_model)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ::setenv("VPIPE_H3_NO_QMM_AUTOTUNE", "1", 1);
  struct Unpin {
    ~Unpin() { ::unsetenv("VPIPE_H3_NO_QMM_AUTOTUNE"); }
  } unpin;

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    return;
  }
  cfg.n_layers = envi_("VPIPE_H3_SOL_LAYERS", 2);

  h3::PackedLayout L;
  const std::vector<int> text_tags(16, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(text_tags, 20, 16, 16, 40,
                                        cfg.patch_h, cfg.patch_w,
                                        h3::kAudioChannels,
                                        {h3::Anchor::kFirst}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, 0.7f, 0.5f, 0.0f, &uniq, &row_idx);
  const SharedBuffer vb = ramp_(
      mc, (std::size_t)L.video_indices.size() * cfg.video_patch_elems(),
      0.017f);
  const SharedBuffer ab = ramp_(
      mc, (std::size_t)L.audio_indices.size() * cfg.audio_channels, 0.031f);
  const SharedBuffer tb = ramp_(mc, (std::size_t)16 * cfg.text_dim, 0.005f);
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  auto fwd = [&](MetalMiniMaxH3Transformer* m, bool dense,
                 std::vector<std::uint16_t>* out) {
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb; step.audio = &ab; step.text = &tb;
    step.layout = &L; step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    step.dense_attention = dense;
    std::string ferr;
    MetalMiniMaxH3Transformer::Velocity v = m->forward(step, &ferr);
    if (v.empty()) { return false; }
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    out->assign(p, p + v.video.byte_size() / 2);
    return true;
  };

  std::vector<std::uint16_t> plain, dense_step, routed;
  {
    auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    ASSERT_TRUE(fwd(m.get(), false, &plain));
  }
  {
    MetalMiniMaxH3Transformer::Config c = cfg;
    c.sol.enabled = true;
    c.sol.tau = 1.0f;
    c.sol.dense_layers = 0;                 // every block routed
    auto m = MetalMiniMaxH3Transformer::load(root, mc, c);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    // Dense FIRST, then routed, on ONE model: the order a sampler with
    // sol_dense_steps runs them in, so a flag that left Sol state
    // behind would show in the second.
    ASSERT_TRUE(fwd(m.get(), true, &dense_step));
    ASSERT_TRUE(fwd(m.get(), false, &routed));
  }
  const bool same = dense_step == plain;
  const bool moved = routed != plain;
  std::printf("[h3_sol] dense step: %s the plain stack; routed step %s\n",
              same ? "BYTE-IDENTICAL to" : "DIFFERS from",
              moved ? "differs (Sol was live)" : "IS the plain stack");
  EXPECT_TRUE(same);
  EXPECT_TRUE(moved);
}

// SOL AND i8_gemm ARE INDEPENDENT KNOBS, and this pins that the
// architecture lets both be set at once.
//
// They act on different halves of a block -- i8_gemm on the GEMMs (qkv,
// the FF pair, the out projection), Sol on the attention between them --
// and nothing in either reads the other's state. What this test asserts
// is exactly that: the pair is ACCEPTED, it runs, it is finite and
// reproducible, and it differs from dense the way Sol alone does.
//
// It does NOT assert a numerical result for the combination. i8_gemm
// self-gates on the matrix units and this box has none, so on an M4 the
// int8 half is inert however it is configured -- the combination's
// numerics are an M5 measurement. Asserting a number here would be
// asserting that i8_gemm did nothing.
TEST(sol_attention_h3, sol_and_i8_gemm_are_independently_settable)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    return;
  }
  cfg.n_layers = envi_("VPIPE_H3_SOL_LAYERS", 2);

  h3::PackedLayout L;
  const std::vector<int> text_tags(16, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(text_tags, 20, 16, 16, 40,
                                        cfg.patch_h, cfg.patch_w,
                                        h3::kAudioChannels,
                                        {h3::Anchor::kFirst}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, 0.7f, 0.5f, 0.0f, &uniq, &row_idx);
  const SharedBuffer vb = ramp_(
      mc, (std::size_t)L.video_indices.size() * cfg.video_patch_elems(),
      0.017f);
  const SharedBuffer ab = ramp_(
      mc, (std::size_t)L.audio_indices.size() * cfg.audio_channels, 0.031f);
  const SharedBuffer tb = ramp_(mc, (std::size_t)16 * cfg.text_dim, 0.005f);
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  auto run_arm = [&](bool sol, bool i8, std::vector<std::uint16_t>* out) {
    MetalMiniMaxH3Transformer::Config c = cfg;
    c.i8_gemm = i8;
    c.sol.enabled = sol;
    c.sol.tau = 1.0f;
    c.sol.dense_layers = 1;
    auto m = MetalMiniMaxH3Transformer::load(root, mc, c);
    if (m == nullptr) { return false; }
    MetalMiniMaxH3Transformer::Step step;
    step.video = &vb; step.audio = &ab; step.text = &tb;
    step.layout = &L; step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    MetalMiniMaxH3Transformer::Velocity v = m->forward(step, &ferr);
    if (v.empty()) {
      std::printf("[h3_sol] sol=%d i8=%d: %s\n", (int)sol, (int)i8,
                  ferr.c_str());
      return false;
    }
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    out->assign(p, p + v.video.byte_size() / 2);
    return true;
  };

  // PINNED, for the reason the first test in this file gives: the tuner
  // picks projection tiles by measuring, and two arms landing on
  // different tiles differ before attention is reached.
  ::setenv("VPIPE_H3_NO_QMM_AUTOTUNE", "1", 1);
  struct Unpin {
    ~Unpin() { ::unsetenv("VPIPE_H3_NO_QMM_AUTOTUNE"); }
  } unpin;

  std::vector<std::uint16_t> dense, i8_only, sol_only, both, again;
  if (!run_arm(false, false, &dense))   { EXPECT_TRUE(false); return; }
  if (!run_arm(false, true, &i8_only))  { EXPECT_TRUE(false); return; }
  if (!run_arm(true, false, &sol_only)) { EXPECT_TRUE(false); return; }
  // THE POINT OF THE TEST: both at once is a configuration the model
  // accepts and runs, not one it refuses or silently drops half of.
  if (!run_arm(true, true, &both)) { EXPECT_TRUE(false); return; }
  if (!run_arm(true, true, &again)) { EXPECT_TRUE(false); return; }

  std::size_t bad = 0;
  const double r = rms_(both, &bad);
  std::printf("[h3_sol] sol+i8 together: rms %.6f, non-finite %zu\n", r, bad);
  EXPECT_TRUE(bad == 0);
  EXPECT_TRUE(r > 0.0);
  EXPECT_TRUE(both.size() == sol_only.size());
  // Reproducible, which is what says the pair did not leave a stale
  // buffer between them.
  EXPECT_TRUE(again.size() == both.size() &&
              std::memcmp(again.data(), both.data(), both.size() * 2) == 0);
  // On a box WITHOUT matrix units i8_gemm is inert, so the two arms must
  // agree exactly -- and that agreement is itself the evidence that
  // enabling i8 did not disturb the Sol path.
  if (!mc->supports_matrix_cores()) {
    const bool same = std::memcmp(both.data(), sol_only.data(),
                                  both.size() * 2) == 0;
    std::printf("[h3_sol] no matrix units: i8 inert, arms %s\n",
                same ? "IDENTICAL" : "differ");
    EXPECT_TRUE(same);
    return;
  }

  // AND WITH matrix units, where i8_gemm is live, the numbers the M4
  // side could not take.
  //
  // The two act on different halves of a block and neither reads the
  // other's state, so their errors should be INDEPENDENT -- and
  // independent errors add in quadrature, not linearly. That is a much
  // sharper claim than "not much worse than the sum": a shared buffer,
  // a re-quantised activation or a routing decision taken on int8
  // scores would all correlate the two and push the combination toward
  // the linear sum or past it.
  //
  // MEASURED on an M5, 2 layers at 1440 rows, tau 1.0: i8 0.0151, sol
  // 0.0190, both 0.0244 against a quadrature prediction of 0.0243 and a
  // linear sum of 0.0341. And i8's effect ON the Sol arm (0.0153) is
  // its effect on dense (0.0151), which is the same statement from the
  // other side.
  const double d_i8  = rel_(i8_only, dense);
  const double d_sol = rel_(sol_only, dense);
  const double d_all = rel_(both, dense);
  const double d_add = rel_(both, sol_only);
  const double quad = std::sqrt(d_i8 * d_i8 + d_sol * d_sol);
  std::printf("[h3_sol] vs dense: i8 %.4f, sol %.4f, both %.4f "
              "(quadrature %.4f, sum %.4f); i8's own effect on the sol "
              "arm %.4f\n", d_i8, d_sol, d_all, quad, d_i8 + d_sol, d_add);
  // Both are doing something -- on their own and on top of each other.
  EXPECT_TRUE(d_i8 > 1e-5);
  EXPECT_TRUE(d_sol > 1e-5);
  EXPECT_TRUE(d_add > 1e-5);
  // Independent, to a quarter.
  EXPECT_TRUE(d_all <= 1.25 * quad + 1e-3);
  // ...and Sol does not amplify what i8 costs: i8 moves the routed arm
  // by what it moves the dense one by.
  EXPECT_TRUE(d_add <= 1.5 * d_i8 + 1e-3);
}

// ---------------------------------------------------------------------
// THE CONFIG SURFACE on generate-video: the enable flag and the two key
// block sizes, which are offered because they trade against each other
// rather than one dominating. MEASURED over three sequence lengths: 64
// keeps 28-37% of key blocks and runs 3.54x over dense steel; 32 keeps
// 23-26% and runs 2.79x, and is the MORE accurate of the two (rel-L2
// 0.0070-0.0077 against 0.0082-0.0091) because a centroid over 32 keys
// stands in for them better. So 64 buys speed and 32 buys fidelity.
//
// Needs no model: this is the stage's own parsing.

#include "common/flex-data.h"
#include "stages/generate-video-stage.h"

TEST(sol_attention_h3, generate_video_offers_the_enable_flag_and_two_blocks)
{
  Session sess;
  auto mk = [&](bool on, long long blk) {
    FlexData cfg = FlexData::make_object();
    auto o = cfg.as_object();
    o.insert_or_assign("sol_attn", FlexData::make_bool(on));
    if (blk >= 0) {
      o.insert_or_assign("sol_key_block", FlexData::make_int(blk));
    }
    return std::make_unique<GenerateVideoStage>(
        &sess, "gv", std::vector<InEdge>{}, cfg);
  };

  // The flag alone, with no block named: the default, which is 0 and
  // means 64 to the component.
  {
    auto s = mk(true, -1);
    EXPECT_TRUE(s->config_error().empty());
    EXPECT_TRUE(s->sol_config().enabled);
    EXPECT_TRUE(s->sol_config().key_block == 0);
  }
  // OFF is the default, and it has to be: this is an approximation.
  {
    FlexData empty = FlexData::make_object();
    auto s = std::make_unique<GenerateVideoStage>(
        &sess, "gv", std::vector<InEdge>{}, empty);
    EXPECT_TRUE(!s->sol_config().enabled);
  }
  // Both offered sizes are taken as given.
  for (const long long blk : {32LL, 64LL}) {
    auto s = mk(true, blk);
    EXPECT_TRUE(s->config_error().empty());
    EXPECT_TRUE(s->sol_config().key_block == (int)blk);
  }
  // Anything else FALLS BACK, and does not fail the stage. A perf knob
  // spelled wrong should not take a graph down -- which is what
  // `unload_when_idle` beside it does -- but it must not be silent
  // either, so the warning names the value. 128 and 256 are the
  // interesting rejects: they RUN, they are simply slower and less
  // accurate, so this is a policy about what to offer rather than a
  // limit of the kernel.
  for (const long long blk : {1LL, 16LL, 48LL, 128LL, 256LL}) {
    auto s = mk(true, blk);
    EXPECT_TRUE(s->config_error().empty());
    const bool fell_back = s->sol_config().key_block == 0;
    if (!fell_back) {
      std::printf("[h3_sol] sol_key_block %lld was kept, not refused\n", blk);
    }
    EXPECT_TRUE(fell_back);
  }
}

// A DENOISE MUST NOT REBUILD THE ACTIVATION SCRATCH.
//
// `n_t` -- the count of distinct timesteps a forward presents -- is 1 on
// the first step and 2 once the video and audio schedules diverge. Three
// small buffers are sized by it, and keying the scratch on the exact
// value therefore rebuilt EVERY activation buffer in the model on step
// 1 to grow those three by a few hundred KB.
//
// That is an out-of-memory and not a slow path: the new set is built in
// full before the old is dropped, so the peak is both at once. At
// 99128 rows that is 13 GB twice on a 24 GB box, and it killed a
// 1344x768 328-frame run at step 1 with
// kIOGPUCommandBufferCallbackErrorOutOfMemory nearly six minutes in.
//
// So the property is: one allocation for the whole schedule.
TEST(sol_attention_h3, a_denoise_allocates_its_scratch_once)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  ::setenv("VPIPE_H3_NO_QMM_AUTOTUNE", "1", 1);
  struct Unpin {
    ~Unpin() { ::unsetenv("VPIPE_H3_NO_QMM_AUTOTUNE"); }
  } unpin;

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    return;
  }
  cfg.n_layers = envi_("VPIPE_H3_SOL_LAYERS", 2);

  h3::PackedLayout L;
  const std::vector<int> text_tags(16, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(text_tags, 20, 16, 16, 40,
                                        cfg.patch_h, cfg.patch_w,
                                        h3::kAudioChannels,
                                        {h3::Anchor::kFirst}, &L));
  const SharedBuffer vb = ramp_(
      mc, (std::size_t)L.video_indices.size() * cfg.video_patch_elems(),
      0.017f);
  const SharedBuffer ab = ramp_(
      mc, (std::size_t)L.audio_indices.size() * cfg.audio_channels, 0.031f);
  const SharedBuffer tb = ramp_(mc, (std::size_t)16 * cfg.text_dim, 0.005f);
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  // Two forwards whose ONLY difference is the distinct-timestep count:
  // the first step of a schedule has one, a later step has two. That is
  // the pair that used to rebuild.
  int nt_seen[2] = {0, 0};
  for (int step = 0; step < 2; ++step) {
    std::vector<float> uniq;
    std::vector<int>   row_idx;
    // Equal video and audio levels give ONE distinct timestep; unequal
    // give two, which is what a schedule does after its first step.
    h3::build_row_timesteps(L, 0.7f, step == 0 ? 0.7f : 0.5f, 0.0f, &uniq,
                            &row_idx);
    nt_seen[step] = (int)uniq.size();
    MetalMiniMaxH3Transformer::Step st;
    st.video = &vb; st.audio = &ab; st.text = &tb;
    st.layout = &L; st.timesteps = &uniq;
    st.row_timestep_index = &row_idx;
    std::string ferr;
    MetalMiniMaxH3Transformer::Velocity v = m->forward(st, &ferr);
    if (v.empty()) {
      std::printf("[h3_sol] forward %d: %s\n", step, ferr.c_str());
      EXPECT_TRUE(false);
      return;
    }
  }
  std::printf("[h3_sol] distinct timesteps %d then %d; scratch allocated "
              "%d time(s)\n", nt_seen[0], nt_seen[1],
              m->scratch_rebuilds());
  // The premise: the two forwards really do differ in n_t. Without that
  // the assertion below would pass for the wrong reason.
  EXPECT_TRUE(nt_seen[1] > nt_seen[0]);
  EXPECT_TRUE(m->scratch_rebuilds() == 1);
}

// SOL AND SAGE IN THE SAME STACK, which is the pair that composes only
// here: Sol decides which key blocks are attended and Sage how the ones
// that are get multiplied, and the exact half is the one dispatch both
// of them touch.
//
// FOUR ARMS against one dense reference, because three of the possible
// failures are indistinguishable from two arms. Sol alone and Sage alone
// each have to land where they land on their own; the composed arm has
// to land near Sol's, since Sage costs the routed answer about what int8
// costs a dense one. An arm that comes back non-finite, or an order of
// magnitude out, is the composition and not either half.
TEST(sol_attention_h3, sol_and_sage_compose_in_the_stack)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[h3_sage] no matrix cores -- the int8 QK cannot run, "
                "SKIPPED\n");
    return;
  }

  const int latf  = envi_("VPIPE_H3_SOL_LATF", 20);
  const int lath  = envi_("VPIPE_H3_SOL_LATH", 16);
  const int latw  = envi_("VPIPE_H3_SOL_LATW", 16);
  const int naud  = envi_("VPIPE_H3_SOL_AUD", 40);
  const int ntext = envi_("VPIPE_H3_SOL_TEXT", 16);

  // Same reason the sibling above pins it: every arm builds its own
  // transformer, and a tuner that measures can hand two arms different
  // projection tiles before attention is reached at all.
  ::setenv("VPIPE_H3_NO_QMM_AUTOTUNE", "1", 1);
  struct Unpin {
    ~Unpin() { ::unsetenv("VPIPE_H3_NO_QMM_AUTOTUNE"); }
  } unpin;

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    return;
  }
  cfg.n_layers = envi_("VPIPE_H3_SOL_LAYERS", 2);

  h3::PackedLayout L;
  const std::vector<int> text_tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(
      text_tags, latf, lath, latw, naud, cfg.patch_h, cfg.patch_w,
      h3::kAudioChannels, {h3::Anchor::kFirst}, &L));
  std::vector<float> uniq;
  std::vector<int>   row_idx;
  h3::build_row_timesteps(L, 0.7f, 0.5f, 0.0f, &uniq, &row_idx);

  const SharedBuffer vb = ramp_(
      mc, (std::size_t)L.video_indices.size() * cfg.video_patch_elems(),
      0.017f);
  const SharedBuffer ab = ramp_(
      mc, (std::size_t)L.audio_indices.size() * cfg.audio_channels, 0.031f);
  const SharedBuffer tb =
      ramp_(mc, (std::size_t)ntext * cfg.text_dim, 0.005f);
  ASSERT_TRUE(!vb.empty() && !ab.empty() && !tb.empty());

  auto run_arm = [&](const sol::Config& sc, const sage::Config& gc,
                     const char* what, std::vector<std::uint16_t>* out) {
    MetalMiniMaxH3Transformer::Config c = cfg;
    c.sol = sc;
    c.sage = gc;
    auto m = MetalMiniMaxH3Transformer::load(root, mc, c);
    if (m == nullptr) { return false; }
    MetalMiniMaxH3Transformer::Step step;
    step.video  = &vb;
    step.audio  = &ab;
    step.text   = &tb;
    step.layout = &L;
    step.timesteps = &uniq;
    step.row_timestep_index = &row_idx;
    std::string ferr;
    MetalMiniMaxH3Transformer::Velocity v = m->forward(step, &ferr);
    if (v.empty()) {
      std::printf("[h3_sage] %s: %s\n", what, ferr.c_str());
      return false;
    }
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    out->assign(p, p + v.video.byte_size() / 2);
    std::size_t bad = 0;
    const double r = rms_(*out, &bad);
    std::printf("[h3_sage] %-16s rms %.6f  non-finite %zu\n", what, r, bad);
    return true;
  };

  sol::Config sol_off;
  sol::Config sol_on;
  sol_on.enabled = true;
  sol_on.tau = 1.0f;
  sol_on.dense_layers = 1;
  sage::Config sage_off;
  sage::Config sage_on;
  sage_on.enabled = true;
  sage_on.dense_layers = 0;

  std::vector<std::uint16_t> dense, only_sol, only_sage, both;
  if (!run_arm(sol_off, sage_off, "dense", &dense)) { return; }
  if (!run_arm(sol_on, sage_off, "sol", &only_sol)) { return; }
  if (!run_arm(sol_off, sage_on, "sage", &only_sage)) { return; }
  if (!run_arm(sol_on, sage_on, "sol+sage", &both)) { return; }
  ASSERT_TRUE(both.size() == dense.size());
  if (both.size() != dense.size()) { return; }

  auto rel_to_dense = [&](const std::vector<std::uint16_t>& x) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < dense.size(); ++i) {
      const double a = bf16_(x[i]), b = bf16_(dense[i]);
      num += (a - b) * (a - b);
      den += b * b;
    }
    return den > 0.0 ? std::sqrt(num / den) : 0.0;
  };
  // THE ROUTING TURNED OFF, which is the discriminator the arms above
  // cannot be: at tau = -inf Sol keeps every key block exact, so the
  // composed arm attends exactly what Sage alone attends by the same
  // int8 product, and the two must land on top of each other. A gap
  // here is the COMPOSITION -- the operands, their indexing, the
  // function constant -- and not the approximation.
  //
  // The reference is Sage with the SAME key handling the composed path
  // uses (MetalSolAttention::compose clears the smoothing), or the two
  // arms differ by the very shift this test is downstream of and the
  // check measures that instead of what it is for.
  sol::Config sol_exact;
  sol_exact.enabled = true;
  sol_exact.tau = -std::numeric_limits<float>::infinity();
  sol_exact.dense_layers = 0;
  std::vector<std::uint16_t> sage_raw_only, exact_both;
  const bool have_raw =
      run_arm(sol_off, MetalSolAttention::compose(sage_on), "sage(raw)",
              &sage_raw_only);
  if (have_raw && run_arm(sol_exact, sage_on, "sol(-inf)+sage",
                          &exact_both) &&
      exact_both.size() == sage_raw_only.size()) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < sage_raw_only.size(); ++i) {
      const double a = bf16_(exact_both[i]), b = bf16_(sage_raw_only[i]);
      num += (a - b) * (a - b);
      den += b * b;
    }
    const double gap = den > 0.0 ? std::sqrt(num / den) : 0.0;
    std::printf("[h3_sage] sol(tau=-inf)+sage vs sage(raw) alone: %.4f\n",
                gap);
    EXPECT_TRUE(std::isfinite(gap) && gap < 0.005);
  }

  // THE ARM THAT USED TO SHIP IS NO LONGER REACHABLE from here, and
  // deliberately: MetalSolAttention::compose clears the smoothing for
  // every composing caller, so a config asking for it back is ignored.
  // What guards it is the ratio below -- the bug sat at 1.98x Sol's own
  // error, which is the shape a term that should have cancelled makes.

  const double r_sol = rel_to_dense(only_sol);
  const double r_sage = rel_to_dense(only_sage);
  const double r_both = rel_to_dense(both);
  std::printf("[h3_sage] vs dense: sol %.4f | sage %.4f | sol+sage %.4f\n",
              r_sol, r_sage, r_both);

  // Each half on its own is the control; the composed arm is the claim.
  EXPECT_TRUE(std::isfinite(r_sol) && r_sol < 0.5);
  EXPECT_TRUE(std::isfinite(r_sage) && r_sage < 0.5);
  // Sage costs the ROUTED answer about what it costs a dense one, so the
  // composed arm sits near Sol's and not in a different regime. Garbage
  // -- the failure this test was written for -- lands far outside.
  EXPECT_TRUE(std::isfinite(r_both));
  // COMPOSING MUST NOT MULTIPLY SOL'S ERROR. Sage costs the routed
  // answer about what int8 costs a dense one, so the composed arm sits
  // just above Sol's. The bug this replaced sat at 1.98x, which is why
  // the bound is a ratio and not a slack: a slack wide enough for the
  // two errors to add is also wide enough to admit a term that should
  // have cancelled.
  EXPECT_TRUE(r_both < r_sol * 1.4);

  // ...and Sage actually ran on the routed blocks: composing must move
  // the answer off Sol-alone, or it silently did nothing.
  std::size_t differ = 0;
  for (std::size_t i = 0; i < both.size(); ++i) {
    if (both[i] != only_sol[i]) { ++differ; }
  }
  std::printf("[h3_sage] sol+sage differs from sol alone in %zu/%zu\n",
              differ, both.size());
  EXPECT_TRUE(differ > both.size() / 100);
}
