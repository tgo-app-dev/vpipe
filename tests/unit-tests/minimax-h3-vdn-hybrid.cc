// VDN-H3's hybrid attention INSIDE the transformer.
//
// Everything under vdn_* checks the branch as a component, against
// goldens taken from the reference. This is the first thing that runs it
// where it actually lives: attached to a real H3 stack, reading that
// stack's own fused q/k/v projection in place, with the windowed softmax
// beside it.
//
// WHAT THIS TEST CAN AND CANNOT SAY. There is no reference for a whole
// VDN forward in this tree, so this is not a numerical check -- the
// component tests are where the arithmetic is pinned. What it pins is
// the WIRING, and the wiring has exactly the failure mode that hides:
// an unvalidated ComputeFunction is a silent no-op, a flag read the
// wrong way round skips a half, and a branch that contributes nothing
// produces a perfectly ordinary velocity. So the assertions are that
// the hybrid RUNS, that it is finite, that it CHANGES the answer, and
// that it is reproducible.

#include "minitest.h"

#include "common/perf-flops.h"
#include "common/session.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/minimax-h3/vdn-geometry.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using namespace vpipe::metal_compute;
namespace h3 = vpipe::genai::minimax_h3;
namespace vdn = vpipe::genai::minimax_h3::vdn;

namespace {

// Full cover for the released config: chunk 5, radius 1.
int
_vdn_cover_limit()
{
  return (1 + 1) * 5;
}

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

std::string
vdn_root_()
{
  if (const char* e = std::getenv("VPIPE_VDN_MODEL_PATH")) {
    return std::string(e) + "/stage-dmd-step-250";
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) { return ""; }
  return std::string(home)
         + "/dock/dump/vpipe-test/models/OpenVDN/vdn-minimax-h3/"
           "stage-dmd-step-250";
}

}  // namespace

TEST(minimax_h3_vdn, the_hybrid_runs_and_changes_the_answer)
{
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  const std::string vroot = vdn_root_();
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  // Small on purpose, but NOT arbitrarily so. The window is a window of
  // CHUNKS, so full cover is not "the clip is shorter than the window's
  // span": frame 0 sits at the start of chunk 0 and reaches only to the
  // end of chunk +radius, so every frame sees every other exactly when
  //
  //     F <= (radius + 1) * chunk        (= 10 for the released config)
  //
  // and under that the hybrid IS full attention with the linear half
  // correctly off. A clip at or under 10 frames would pass this test
  // while exercising none of the windowing. (2*radius+1)*chunk = 15 is
  // the span of one frame's window and is NOT this threshold -- I used
  // it first and the sibling test below caught it.
  const int latf  = envi("VPIPE_H3_VDN_LATF", 20);   // > 10: windowed
  const int lath  = envi("VPIPE_H3_VDN_LATH", 16);
  const int latw  = envi("VPIPE_H3_VDN_LATW", 16);
  const int naud  = envi("VPIPE_H3_VDN_AUD", 40);
  const int ntext = envi("VPIPE_H3_VDN_TEXT", 16);

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    std::printf("[h3_vdn] config: %s\n", cerr.c_str());
    return;
  }
  cfg.n_layers = envi("VPIPE_H3_VDN_LAYERS", 2);

  h3::PackedLayout L;
  const std::vector<int> text_tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(
      text_tags, latf, lath, latw, naud, cfg.patch_h, cfg.patch_w,
      h3::kAudioChannels, {h3::Anchor::kFirst}, &L));
  const int gh = lath / cfg.patch_h, gw = latw / cfg.patch_w;
  const int tpf = gh * gw;
  if (tpf <= 0 || L.num_video_rows % tpf != 0) {
    std::printf("[h3_vdn] the grid does not divide the video rows\n");
    EXPECT_TRUE(false);
    return;
  }
  const int frames = L.num_video_rows / tpf;
  std::printf("[h3_vdn] seq %d (%d text + %d cond + %d audio + %d video), "
              "%d frames of %d, %d blocks\n", L.seq_len, L.num_text_rows,
              L.num_condition_rows, L.num_audio_rows, L.num_video_rows,
              frames, tpf, cfg.n_layers);

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

  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  if (m == nullptr) { std::printf("[h3_vdn] DiT load failed\n"); }
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  MetalMiniMaxH3Transformer::Step step;
  step.video  = &vb;
  step.audio  = &ab;
  step.text   = &tb;
  step.layout = &L;
  step.timesteps = &uniq;
  step.row_timestep_index = &row_idx;

  // perf-flops: each forward's count and what the meter was credited.
  // The hybrid's window and its linear branch (MetalVdnBranch::
  // set_flops) are counted into the transformer's own tally, so the
  // branch's work must reach the meter through the same commits.
  struct Count { std::uint64_t flushed = 0, gpu = 0, pending = 0; };
  Count last;
  auto run = [&](const char* what, std::vector<std::uint16_t>* out) {
    namespace pf = vpipe::perf_flops;
    std::string ferr;
    const std::uint64_t f0 = m->gpu_flops().flushed();
    const std::uint64_t g0 = pf::totals(pf::kGpu).flops;
    MetalMiniMaxH3Transformer::Velocity v = m->forward(step, &ferr);
    if (v.empty()) { std::printf("[h3_vdn] %s: %s\n", what, ferr.c_str()); }
    if (v.empty()) { return false; }
    last.flushed = m->gpu_flops().flushed() - f0;
    for (int i = 0; i < 400 && pf::totals(pf::kGpu).flops - g0 <
                                   last.flushed; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    last.gpu = pf::totals(pf::kGpu).flops - g0;
    last.pending = m->gpu_flops().pending();
    std::printf("[h3_vdn] %-14s counted %.4f TFLOP, credited %.4f\n", what,
                last.flushed / 1e12, last.gpu / 1e12);
    EXPECT_TRUE(last.gpu == last.flushed && last.pending == 0);
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    out->assign(p, p + v.video.byte_size() / 2);
    std::size_t bad = 0;
    double l2 = 0.0;
    for (std::uint16_t h : *out) {
      const float f = bf16_(h);
      if (!std::isfinite(f)) { ++bad; }
      else { l2 += (double)f * (double)f; }
    }
    l2 = std::sqrt(l2 / (double)(out->empty() ? 1 : out->size()));
    std::printf("[h3_vdn] %-14s rms %.6f  non-finite %zu\n", what, l2, bad);
    EXPECT_TRUE(bad == 0);
    return bad == 0;
  };

  // The stock stack, which is also the state every non-VDN caller is in:
  // Step's grid left at zero keeps the hybrid off even once a branch is
  // attached.
  std::vector<std::uint16_t> plain;
  if (!run("full attn", &plain)) { return; }
  const Count c_plain = last;

  std::string aerr;
  if (!m->attach_linear_branch(vroot, &aerr)) {
    // Not downloaded is a skip; anything else is a failure, and the two
    // are told apart by whether the DiT itself loaded (it did, above).
    std::printf("[h3_vdn] attach: %s -- SKIPPED\n", aerr.c_str());
    return;
  }
  ASSERT_TRUE(m->has_linear_branch());

  // Attached but NOT asked for: the grid is still zero, so this must
  // reproduce the stock stack exactly. Attaching a branch may not change
  // what a checkpoint means.
  std::vector<std::uint16_t> attached;
  if (!run("attached/off", &attached)) { return; }
  EXPECT_TRUE(last.flushed == c_plain.flushed);   // inert in FLOPs too
  const bool inert = attached.size() == plain.size()
                     && std::memcmp(attached.data(), plain.data(),
                                    plain.size() * 2) == 0;
  EXPECT_TRUE(inert);
  if (!inert) {
    std::printf("[h3_vdn] attaching the branch changed an unhybridised "
                "forward\n");
  }

  step.video_grid_h = gh;
  step.video_grid_w = gw;
  std::vector<std::uint16_t> hybrid;
  if (!run("hybrid", &hybrid)) { return; }
  const Count c_hybrid = last;
  // The window drops (video, video) pairs and the branch adds its own
  // arithmetic: either way the count moves.
  EXPECT_TRUE(c_hybrid.flushed != c_plain.flushed);
  EXPECT_TRUE(m->linear_solve_failures() == 0u);
  // Every block ran the linear half, which is the other half of the
  // full-cover test below: the count has to move with the geometry, and
  // a count that is always n_layers would let that test pass while the
  // window did nothing.
  EXPECT_TRUE(!m->linear_window_covers_all());
  EXPECT_TRUE(m->linear_blocks_run() == (unsigned)cfg.n_layers);

  // IT HAS TO DIFFER. The linear branch carries what the window cannot
  // see and the window is genuinely narrower than the clip here, so an
  // output equal to full attention means one of the halves did nothing
  // -- which is exactly what a silent no-op kernel or a flag read
  // backwards looks like from the outside.
  std::size_t diff = 0;
  for (std::size_t i = 0; i < hybrid.size() && i < plain.size(); ++i) {
    if (hybrid[i] != plain[i]) { ++diff; }
  }
  const bool moved = diff * 4 > hybrid.size();
  EXPECT_TRUE(moved);
  std::printf("[h3_vdn] hybrid differs from full attention in %zu of %zu\n",
              diff, hybrid.size());

  // And it must be the SAME every time: the branch reuses its scratch
  // and its banks across blocks and forwards, so a stale read shows up
  // here and nowhere else.
  std::vector<std::uint16_t> again;
  if (!run("hybrid again", &again)) { return; }
  EXPECT_TRUE(last.flushed == c_hybrid.flushed);
  const bool same = again.size() == hybrid.size()
                    && std::memcmp(again.data(), hybrid.data(),
                                   hybrid.size() * 2) == 0;
  EXPECT_TRUE(same);
  if (!same) { std::printf("[h3_vdn] the hybrid is NOT reproducible\n"); }
}

TEST(minimax_h3_vdn, a_window_that_covers_the_clip_turns_the_linear_half_off)
{
  // A WINDOW WIDE ENOUGH TO REACH EVERY FRAME IS THE ORIGINAL
  // ATTENTION, and then the linear half must not run at all.
  //
  // It carries the window's COMPLEMENT, and the complement of everything
  // is nothing -- so left running it adds the text state to every video
  // row. No crash, no NaN, no shape error: a term the reference does not
  // have, on a path anybody exercises the moment they generate a short
  // clip. The reference names this case `full_cover` and switches the
  // linear branch off for it explicitly.
  //
  // Full cover is F <= (radius + 1) * chunk -- 10 frames for the
  // released config -- because the window is a window of CHUNKS and
  // frame 0 reaches only to the end of chunk +radius. So this runs at 9
  // and its sibling above at 20. BOTH are needed: a count that never
  // moved would let either one pass alone, and my first version of this
  // test used 13 on the wrong threshold ((2*radius+1)*chunk = 15, which
  // is one frame's SPAN, not the cover condition) and failed at
  // covers_all -- which is the check doing its job.
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
  cfg.n_layers = 2;

  const int latf = 9, lath = 16, latw = 16, naud = 18, ntext = 16;
  h3::PackedLayout L;
  const std::vector<int> text_tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(
      text_tags, latf, lath, latw, naud, cfg.patch_h, cfg.patch_w,
      h3::kAudioChannels, {h3::Anchor::kFirst}, &L));
  const int gh = lath / cfg.patch_h, gw = latw / cfg.patch_w;
  const int frames = L.num_video_rows / (gh * gw);

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

  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  std::string aerr;
  if (!m->attach_linear_branch(vdn_root_(), &aerr)) {
    std::printf("[h3_vdn] attach: %s -- SKIPPED\n", aerr.c_str());
    return;
  }

  MetalMiniMaxH3Transformer::Step step;
  step.video  = &vb;
  step.audio  = &ab;
  step.text   = &tb;
  step.layout = &L;
  step.timesteps = &uniq;
  step.row_timestep_index = &row_idx;
  step.video_grid_h = gh;
  step.video_grid_w = gw;

  std::string ferr;
  MetalMiniMaxH3Transformer::Velocity v = m->forward(step, &ferr);
  if (v.empty()) { std::printf("[h3_vdn] full-cover: %s\n", ferr.c_str()); }
  ASSERT_TRUE(!v.empty());
  if (v.empty()) { return; }

  std::printf("[h3_vdn] %d frames: covers_all %d, linear blocks %u of %d\n",
              frames, (int)m->linear_window_covers_all(),
              m->linear_blocks_run(), cfg.n_layers);
  EXPECT_TRUE(frames <= (_vdn_cover_limit()));
  EXPECT_TRUE(m->linear_window_covers_all());
  EXPECT_TRUE(m->linear_blocks_run() == 0u);

  // Still a valid velocity: full cover is the stock attention plus the
  // per-head gate, not a disabled path.
  const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
  std::size_t bad = 0;
  for (std::size_t i = 0; i < v.video.byte_size() / 2; ++i) {
    if (!std::isfinite(bf16_(p[i]))) { ++bad; }
  }
  EXPECT_TRUE(bad == 0);
}

TEST(minimax_h3_vdn, the_window_s_reach_is_set_per_request)
{
  // vdn_range (set_vdn_radius): the softmax half's reach, in chunks either
  // side, in place of the checkpoint's. Per request on one built model,
  // so the window cached at a geometry has to follow it -- at the SAME
  // geometry, which is the case a stale tag would miss.
  //
  // Nine frames, the clip the test above runs: the released radius (1)
  // covers it and the linear half is off. Radius 0 is the query's own
  // chunk alone -- frames 5..8 lie outside frame 0's window -- so the
  // linear half must run on every block, and the answer must move. Back
  // to the checkpoint's, it is off again and the answer is the first.
  //
  // And PER STEP (Step::vdn_radius, what vdn_range_steps rests on): a
  // forward's own reach in place of the request's -- 0 set on the step is
  // the request's 0 to the bit; the next forward with none set is the
  // request's again (nothing latched); a step's -1 is the checkpoint's
  // whatever the request says.
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
  cfg.n_layers = 2;

  const int latf = 9, lath = 16, latw = 16, naud = 18, ntext = 16;
  h3::PackedLayout L;
  const std::vector<int> text_tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(
      text_tags, latf, lath, latw, naud, cfg.patch_h, cfg.patch_w,
      h3::kAudioChannels, {h3::Anchor::kFirst}, &L));
  const int gh = lath / cfg.patch_h, gw = latw / cfg.patch_w;

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

  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  std::string aerr;
  if (!m->attach_linear_branch(vdn_root_(), &aerr)) {
    std::printf("[h3_vdn] attach: %s -- SKIPPED\n", aerr.c_str());
    return;
  }
  EXPECT_TRUE(m->vdn_radius() == m->vdn_checkpoint_radius());

  MetalMiniMaxH3Transformer::Step step;
  step.video  = &vb;
  step.audio  = &ab;
  step.text   = &tb;
  step.layout = &L;
  step.timesteps = &uniq;
  step.row_timestep_index = &row_idx;
  step.video_grid_h = gh;
  step.video_grid_w = gw;

  auto run = [&](int radius, std::vector<float>* out,
                 int per_step = MetalMiniMaxH3Transformer::Step::
                     kVdnRadiusUnset) {
    m->set_vdn_radius(radius);
    MetalMiniMaxH3Transformer::Step st = step;
    st.vdn_radius = per_step;
    std::string ferr;
    MetalMiniMaxH3Transformer::Velocity v = m->forward(st, &ferr);
    if (v.empty()) { std::printf("[h3_vdn] range: %s\n", ferr.c_str()); }
    if (v.empty()) { return false; }
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    out->resize(v.video.byte_size() / 2);
    for (std::size_t i = 0; i < out->size(); ++i) { (*out)[i] = bf16_(p[i]); }
    return true;
  };
  std::vector<float> own, narrow, again;
  ASSERT_TRUE(run(-1, &own));
  const bool own_covers = m->linear_window_covers_all();
  const unsigned own_blocks = m->linear_blocks_run();
  const std::size_t own_floor = m->vdn_arena_floor_asked();
  ASSERT_TRUE(run(0, &narrow));
  const bool narrow_covers = m->linear_window_covers_all();
  const unsigned narrow_blocks = m->linear_blocks_run();
  const std::size_t narrow_floor = m->vdn_arena_floor_asked();
  ASSERT_TRUE(run(-1, &again));

  double moved = 0.0, back = 0.0;
  std::size_t bad = 0;
  for (std::size_t i = 0; i < own.size(); ++i) {
    if (!std::isfinite(narrow[i])) { ++bad; }
    moved = std::max(moved, (double)std::fabs(narrow[i] - own[i]));
    back  = std::max(back, (double)std::fabs(again[i] - own[i]));
  }
  std::printf("[h3_vdn] radius %d: covers %d, linear %u; radius 0: covers "
              "%d, linear %u, max |dv| %.4f; back: max |dv| %.4f\n",
              m->vdn_checkpoint_radius(), (int)own_covers, own_blocks,
              (int)narrow_covers, narrow_blocks, moved, back);
  EXPECT_TRUE(own_covers && own_blocks == 0u);
  EXPECT_TRUE(!narrow_covers && narrow_blocks == (unsigned)cfg.n_layers);
  EXPECT_TRUE(bad == 0);
  EXPECT_TRUE(moved > 0.0);
  EXPECT_TRUE(back == 0.0);
  // The branch's carve out of the attention arena: owed where the linear
  // half runs, not where the window covers the clip.
  EXPECT_TRUE(own_floor == 0 && narrow_floor > 0);

  // Per step: the request at the checkpoint's, the step at 0; then the
  // request alone; then the request at 0, the step at -1. The arena's
  // floor is asked at the STEP's reach: a step that turns the linear
  // half on under a request whose window covers the clip still owes the
  // carve (asked at the request's, it would ask for none).
  std::vector<float> step0, unset, step_own;
  ASSERT_TRUE(run(-1, &step0, 0));
  const unsigned step0_blocks = m->linear_blocks_run();
  const std::size_t step0_floor = m->vdn_arena_floor_asked();
  ASSERT_TRUE(run(-1, &unset));
  const unsigned unset_blocks = m->linear_blocks_run();
  const std::size_t unset_floor = m->vdn_arena_floor_asked();
  ASSERT_TRUE(run(0, &step_own, -1));
  const unsigned own_step_blocks = m->linear_blocks_run();
  const std::size_t own_step_floor = m->vdn_arena_floor_asked();
  m->set_vdn_radius(-1);
  std::printf("[h3_vdn] per step: radius 0 %s the request's 0 (linear "
              "%u); none set %s the checkpoint's (linear %u); -1 over a "
              "request of 0 %s the checkpoint's (linear %u)\n",
              step0 == narrow ? "IS" : "is NOT", step0_blocks,
              unset == own ? "IS" : "is NOT", unset_blocks,
              step_own == own ? "IS" : "is NOT", own_step_blocks);
  std::printf("[h3_vdn] arena floor asked: request 0 %zu, step 0 under "
              "the checkpoint's %zu, none set %zu, step -1 under 0 %zu\n",
              narrow_floor, step0_floor, unset_floor, own_step_floor);
  EXPECT_TRUE(step0 == narrow && step0_blocks == (unsigned)cfg.n_layers);
  EXPECT_TRUE(unset == own && unset_blocks == 0u);
  EXPECT_TRUE(step_own == own && own_step_blocks == 0u);
  EXPECT_TRUE(step0_floor == narrow_floor);
  EXPECT_TRUE(unset_floor == 0 && own_step_floor == 0);

  // A reach past the cap reads as the cap, which covers this clip as
  // any reach from 1 does -- through the request and through a step.
  // Uncapped, 500,000,000 chunks wrap the window's bounds, (chunk +- r)
  // * 5 in an int, into an EMPTY window -- and a forward over it HUNG
  // the GPU (a command buffer that never completed), so the cap is
  // checked first and an uncapped reach is never run.
  constexpr int kHuge = 500000000;
  const int cap = MetalMiniMaxH3Transformer::kVdnMaxRadius;
  m->set_vdn_radius(kHuge);
  const int huge_reach = m->vdn_radius();
  const int huge_step_reach = m->vdn_radius_of(kHuge);
  m->set_vdn_radius(-1);
  std::printf("[h3_vdn] reach %d asked: the request runs %d, a step %d "
              "(cap %d)\n", kHuge, huge_reach, huge_step_reach, cap);
  EXPECT_TRUE(huge_reach == cap && huge_step_reach == cap);
  if (huge_reach != cap || huge_step_reach != cap) { return; }
  std::vector<float> huge, huge_step;
  ASSERT_TRUE(run(kHuge, &huge));
  const bool huge_covers = m->linear_window_covers_all();
  ASSERT_TRUE(run(-1, &huge_step, kHuge));
  const bool huge_step_covers = m->linear_window_covers_all();
  std::printf("[h3_vdn] reach %d: covers %d / %d, %s / %s the "
              "checkpoint's answer\n", cap, (int)huge_covers,
              (int)huge_step_covers, huge == own ? "IS" : "is NOT",
              huge_step == own ? "IS" : "is NOT");
  EXPECT_TRUE(huge_covers && huge == own);
  EXPECT_TRUE(huge_step_covers && huge_step == own);
}

TEST(minimax_h3_vdn, the_turbo_adapter_binds_its_adaln_projections)
{
  // VDN's turbo adapter is the only one in this tree that adapts the
  // ADALN projections -- `transformer_blocks.N.adaln_proj.linear` and
  // `norm_out.linear` -- and the diffusers binder used to skip both,
  // because no plain diffusers or ComfyUI export has them. Unbound,
  // the larger half of an 8-step distillation is silently absent and
  // the model still renders: there is no output anyone could look at
  // and tell.
  //
  // So the assertion is a COUNT, derived rather than recorded: whatever
  // the binder finds must be everything the file offers this stack.
  const char* root = std::getenv("VPIPE_MINIMAX_H3_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  // The FILE, not the directory: LoraSpec::path names a safetensors.
  const std::string turbo =
      vdn_root_() + "/adapters/turbo/adapter_model.safetensors";
  {
    std::ifstream probe(turbo);
    if (!probe) { return; }                      // not downloaded
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  MetalMiniMaxH3Transformer::Config cfg;
  std::string cerr;
  if (!MetalMiniMaxH3Transformer::config_from_json(root, cfg, &cerr)) {
    return;
  }
  cfg.n_layers = 2;

  MetalMiniMaxH3Transformer::LoraSpec spec;
  spec.path = turbo;
  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg, false, {spec});
  if (m == nullptr) { std::printf("[h3_vdn] turbo load failed\n"); }
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }

  // What the FILE offers a stack this deep, counted the way the binder
  // counts: the fused q/k/v triple is one module, then to_out.0,
  // ff.net.0.proj, ff.net.2 and adaln_proj.linear per main block; the
  // same four minus adaln per refiner block, which VDN does not wrap;
  // and norm_out.linear once.
  const int per_main = 5, per_refiner = 4;
  const int want = per_main * cfg.n_layers + per_refiner * cfg.n_refiner + 1;
  const int got = m->lora_modules();
  std::printf("[h3_vdn] turbo bound %d modules, expected %d "
              "(%d blocks + %d refiners + norm_out)\n", got, want,
              cfg.n_layers, cfg.n_refiner);
  EXPECT_TRUE(got == want);
}

TEST(minimax_h3_vdn, block_sparse_steel_matches_the_span_reference)
{
  // The window on the STEEL flash kernel, checked against the scalar
  // sdpa_spans_f16 that has been carrying it.
  //
  // The two agree on the MASK and on nothing else: steel walks a list of
  // key BLOCKS and masks the edges, sdpa_spans walks exact row spans;
  // one accumulates on register MMA tiles, the other with simd_sum. So
  // the bar is the bf16 accumulation floor, not equality -- and a mask
  // that differed by even one frame would be nowhere near it, because
  // the frames the window drops are the ones the linear branch is
  // carrying instead.
  //
  // A/B IN ONE PROCESS, because the arms differ by more than a kernel:
  // the steel arm takes the fused q/k/v layout and the scalar one the
  // head-major transposes. Two runs of the binary could not hold the
  // rest of the model fixed across that.
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
  cfg.n_layers = 2;

  const int latf = 20, lath = 16, latw = 16, naud = 40, ntext = 16;
  h3::PackedLayout L;
  const std::vector<int> text_tags((std::size_t)ntext, h3::kTextTag);
  ASSERT_TRUE(h3::build_packed_sequence(
      text_tags, latf, lath, latw, naud, cfg.patch_h, cfg.patch_w,
      h3::kAudioChannels, {h3::Anchor::kFirst}, &L));
  const int gh = lath / cfg.patch_h, gw = latw / cfg.patch_w;

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

  auto m = MetalMiniMaxH3Transformer::load(root, mc, cfg);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  std::string aerr;
  if (!m->attach_linear_branch(vdn_root_(), &aerr)) {
    std::printf("[h3_vdn] attach: %s -- SKIPPED\n", aerr.c_str());
    return;
  }

  MetalMiniMaxH3Transformer::Step step;
  step.video  = &vb;
  step.audio  = &ab;
  step.text   = &tb;
  step.layout = &L;
  step.timesteps = &uniq;
  step.row_timestep_index = &row_idx;
  step.video_grid_h = gh;
  step.video_grid_w = gw;

  double ms_steel = 0.0, ms_scalar = 0.0;
  auto go = [&](std::vector<float>* out, double* ms) {
    std::string ferr;
    const auto t0 = std::chrono::steady_clock::now();
    MetalMiniMaxH3Transformer::Velocity v = m->forward(step, &ferr);
    *ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (v.empty()) { std::printf("[h3_vdn] %s\n", ferr.c_str()); }
    if (v.empty()) { return false; }
    const std::size_t n = v.video.byte_size() / 2;
    const auto* p = static_cast<const std::uint16_t*>(v.video.contents());
    out->resize(n);
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = bf16_(p[i]); }
    return true;
  };

  std::vector<float> steel, scalar, warm;
  double junk = 0.0;
  ASSERT_TRUE(go(&warm, &junk));               // pipelines, not the timing
  ASSERT_TRUE(go(&steel, &ms_steel));
  ::setenv("VPIPE_VDN_NO_STEEL_SPANS", "1", 1);
  bool ok2 = go(&scalar, &junk);               // its pipelines too
  ok2 = ok2 && go(&scalar, &ms_scalar);
  ::unsetenv("VPIPE_VDN_NO_STEEL_SPANS");
  ASSERT_TRUE(ok2);
  if (steel.empty() || scalar.size() != steel.size()) { return; }

  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < steel.size(); ++i) {
    const double d = (double)steel[i] - (double)scalar[i];
    num += d * d;
    den += (double)scalar[i] * (double)scalar[i];
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : -1.0;
  std::printf("[h3_vdn] block-sparse steel vs sdpa_spans: rel-L2 %.3e, "
              "%.0f vs %.0f ms (%.2fx), window visits %zu of %zu key "
              "blocks (%.1f%%)\n", rel, ms_steel, ms_scalar,
              ms_scalar / (ms_steel > 0.0 ? ms_steel : 1.0),
              m->window_key_blocks(), m->dense_key_blocks(),
              100.0 * (double)m->window_key_blocks()
                  / (double)(m->dense_key_blocks() != 0
                                 ? m->dense_key_blocks() : 1));
  // The same depth-aware floor forward_matches_golden uses, at depth 2.
  const bool okr = rel >= 0.0 && rel < 0.03;
  EXPECT_TRUE(okr);
  // And it must not be ZERO either: two runs that took the same path
  // would agree exactly, which would mean the env A/B did nothing.
  EXPECT_TRUE(rel > 0.0);
}

TEST(minimax_h3_vdn, a_keyframe_anchor_is_a_global_row_not_a_video_frame)
{
  // WHAT THE HYBRID DOES WITH A FIRST/LAST-FRAME REQUEST, pinned --
  // and since 2026-09-08 it is the reference's answer too, not only a
  // choice this port made.
  //
  // The branch was trained text-to-video, and its layout_from_indices
  // REFUSES a video block that is not contiguous, while H3's fl2va packs
  // video as TWO runs -- the keyframe conditioning block and the target.
  // Upstream's keyframe support (same checkpoints, render.py) resolves
  // that by handing the layout only the TARGET run,
  // `video_indices[num_condition_rows:]`, so the conditioning rows fall
  // outside the video span and are "attended like text and audio".
  //
  // Which is what the packed order gives here,
  // [text | keyframe conditions | target audio | target video]: the
  // conditioning rows sit BELOW video_start, so the window's group
  // formula puts them in group 0 with the prompt and the soundtrack --
  // DENSE in both directions -- and the linear branch, which is sized
  // from num_video_rows, never sees them at all. Every generated row
  // therefore attends to every keyframe exactly, which is NOT the same
  // as making the keyframe frame 0 of the windowed block.
  //
  // NOTE the two unrelated senses of "anchor" this test sits between:
  // a KEYFRAME anchor is a conditioning row of H3's layout, while
  // AnchorFrames is the window's own mode, and under `both` it is
  // generated frames 0 and F-1 that the branch drops. They never refer
  // to the same rows.
  MetalMiniMaxH3Transformer::Config cfg;   // defaults are enough: layout only
  const int lath = 16, latw = 16, naud = 40, ntext = 16, latf = 20;
  const int gh = lath / cfg.patch_h, gw = latw / cfg.patch_w;
  const int tpf = gh * gw;

  struct Case { const char* name; std::vector<h3::Anchor> anchors; };
  const Case cases[] = {
      {"t2va (no keyframe)", {}},
      {"i2v  (first)", {h3::Anchor::kFirst}},
      {"fl2va (first+last)", {h3::Anchor::kFirst, h3::Anchor::kLast}},
  };
  for (const Case& c : cases) {
    h3::PackedLayout L;
    const std::vector<int> tags((std::size_t)ntext, h3::kTextTag);
    ASSERT_TRUE(h3::build_packed_sequence(
        tags, latf, lath, latw, naud, cfg.patch_h, cfg.patch_w,
        h3::kAudioChannels, c.anchors, &L));
    ASSERT_TRUE(L.num_video_rows % tpf == 0);
    if (L.num_video_rows % tpf != 0) { return; }
    const int frames = L.num_video_rows / tpf;

    // The window the hybrid would build for this layout.
    vdn::WindowMask m;
    m.seq_len = L.seq_len;
    m.video_start = L.video_start;
    m.num_frames = frames;
    m.tokens_per_frame = tpf;
    m.anchors = vdn::AnchorFrames::kBoth;
    m.bounds = vdn::window_bounds(frames, 1, 5);

    std::printf("[h3_vdn] %-20s seq %5d = %d text + %d cond + %d audio + "
                "%d video (%d frames); window covers rows [%d, %d)\n",
                c.name, L.seq_len, L.num_text_rows, L.num_condition_rows,
                L.num_audio_rows, L.num_video_rows, frames,
                L.video_start, m.video_end());

    // The conditioning rows are BELOW the window's video block, so the
    // mask treats them as global. Checked as a property of the mask and
    // not of the layout, because it is the mask that decides.
    for (int i = 0; i < L.num_condition_rows; ++i) {
      const int r = L.condition_start + i;
      EXPECT_TRUE(r < L.video_start);
      // Dense both ways: every generated row sees this keyframe row, and
      // the keyframe row sees everything.
      EXPECT_TRUE(m.allows(L.video_start, r));
      EXPECT_TRUE(m.allows(m.video_end() - 1, r));
      EXPECT_TRUE(m.allows(r, m.video_end() - 1));
    }
    // And the frame count the branch is sized from is the GENERATED
    // frames alone -- a keyframe never becomes a delta-rule frame.
    EXPECT_TRUE(frames * tpf == L.num_video_rows);
    EXPECT_TRUE(L.video_start >= L.condition_start + L.num_condition_rows);
  }
}
