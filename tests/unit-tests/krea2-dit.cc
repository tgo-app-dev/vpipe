// Krea-2-Turbo DiT bring-up (M3), staged against the diffusers golden taps.
//
// M3a (text-fusion tower + txt_in): feed the golden 12-layer encoder stack
// a1_prompt_embeds (text_seq, 12, 2560) to MetalKrea2Transformer::forward_text
// and rel-L2 the (text_seq, 6144) fused-text output against d_txtin.f32 (the
// HF Krea2Transformer2DModel's txt_in output, dumped by krea2_dit_ref.py).
//
// Env: VPIPE_KREA2_TEST_MODEL_PATH = the Krea-2-Turbo model root (uses
// <root>/transformer), VPIPE_KREA2_GOLDEN = the golden dir. Skips if unset.

#include "ane-int8-expected.h"
#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "common/flex-data.h"
#include "common/perf-flops.h"
#include "generative-models/krea2/metal-krea2-transformer.h"

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
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

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

}  // namespace

TEST(krea2_dit, text_fusion_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string tdir = std::string(root) + "/transformer";
  const std::string gdir = gd;

  // Golden inputs/outputs.
  const std::vector<float> pe = read_f32_(gdir + "/a1_prompt_embeds.f32");
  const std::vector<float> golden = read_f32_(gdir + "/d_txtin.f32");
  ASSERT_TRUE(!pe.empty());
  ASSERT_TRUE(!golden.empty());

  MetalKrea2Transformer::Config cfg;             // Krea2 DiT defaults
  const int NL = cfg.n_text_layers, TH = cfg.text_hidden, HID = cfg.hidden;
  ASSERT_TRUE(pe.size() % ((std::size_t)NL * TH) == 0);
  const int text_seq = (int)(pe.size() / ((std::size_t)NL * TH));
  ASSERT_TRUE(golden.size() == (std::size_t)text_seq * HID);

  auto m = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  // Build the f16 encoder-tap stack [text_seq, 12, 2560] from the golden f32.
  SharedBuffer ehs = mc->make_shared_buffer(pe.size() * 2);
  ASSERT_TRUE(!ehs.empty());
  auto* e = static_cast<_Float16*>(ehs.contents());
  for (std::size_t i = 0; i < pe.size(); ++i) { e[i] = (_Float16)pe[i]; }

  SharedBuffer out = m->forward_text(ehs, text_seq);
  ASSERT_TRUE(!out.empty());
  ASSERT_TRUE(out.byte_size() >= (std::size_t)text_seq * HID * 2);

  const auto* op = static_cast<const _Float16*>(out.contents());
  std::vector<float> got((std::size_t)text_seq * HID);
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)op[i]; }

  const double r = rel_l2_(got.data(), golden.data(), got.size());
  std::printf("[krea2_dit] text_fusion+txt_in rel-L2 = %.6f (text_seq=%d)\n",
              r, text_seq);
  EXPECT_TRUE(r < 0.05);
}

// A/B oracle (needs only the model, no golden): the M5 matrix-core matmul2d
// projection path must match the steel dense_gemm_t path within f32-accumulation
// rounding. Runs the text-fusion tower + txt_in (forward_text) on a deterministic
// random encoder-tap stack twice -- once default (matmul2d on M5) and once with
// VPIPE_KREA2_NO_MMA2 forcing steel -- and rel-L2s the two fused-text outputs.
// This exercises the block attention/SwiGLU projections + txt_in through the
// same gemm_mma_ routing the main DiT blocks use. stream_blocks=true keeps the
// 28 main transformer blocks OFF the heap (the 12B DiT would blow a 16GB box);
// forward_text only touches the small text tower. On a non-matrix-core GPU both
// loads take the steel path -> a trivial (still valid) exact match.
TEST(krea2_dit, text_fusion_mma_matches_steel)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  const int NL = cfg.n_text_layers, TH = cfg.text_hidden, HID = cfg.hidden;
  const int text_seq = 128;
  // Deterministic pseudo-random encoder-tap stack [text_seq, 12, 2560] in
  // ~[-0.5, 0.5].
  std::vector<float> pe((std::size_t)text_seq * NL * TH);
  std::uint32_t s = 0x01234567u;
  for (auto& v : pe) {
    s = s * 1664525u + 1013904223u;
    v = ((float)(s >> 9) / 4194304.0f - 1.0f) * 0.5f;
  }
  auto make_ehs = [&]() {
    SharedBuffer e = mc->make_shared_buffer(pe.size() * 2);
    auto* d = static_cast<_Float16*>(e.contents());
    for (std::size_t i = 0; i < pe.size(); ++i) { d[i] = (_Float16)pe[i]; }
    return e;
  };
  const std::size_t n = (std::size_t)text_seq * HID;

  std::printf("[krea2_dit] text_fusion mma_matches_steel: matrix_cores=%d\n",
              (int)mc->supports_matrix_cores());

  // matmul2d path (default on M5).
  ::unsetenv("VPIPE_KREA2_NO_MMA2");
  auto m_mma =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ASSERT_TRUE(m_mma != nullptr);
  SharedBuffer o_mma = m_mma->forward_text(make_ehs(), text_seq);
  ASSERT_TRUE(!o_mma.empty() && o_mma.byte_size() >= n * 2);

  // steel path (force off at load time).
  ::setenv("VPIPE_KREA2_NO_MMA2", "1", 1);
  auto m_steel =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ::unsetenv("VPIPE_KREA2_NO_MMA2");
  ASSERT_TRUE(m_steel != nullptr);
  SharedBuffer o_steel = m_steel->forward_text(make_ehs(), text_seq);
  ASSERT_TRUE(!o_steel.empty() && o_steel.byte_size() >= n * 2);

  const auto* a = static_cast<const _Float16*>(o_mma.contents());
  const auto* b = static_cast<const _Float16*>(o_steel.contents());
  std::vector<float> fa(n), fb(n);
  for (std::size_t i = 0; i < n; ++i) { fa[i] = (float)a[i]; fb[i] = (float)b[i]; }
  const double r = rel_l2_(fa.data(), fb.data(), n);
  std::printf(
      "[krea2_dit] text_fusion mma-vs-steel rel-L2 = %.6g (text_seq=%d)\n", r,
      text_seq);
  EXPECT_TRUE(r < 3e-2);
}

// A/B oracle for the DiT main-block path (needs only the model, no golden):
// runs forward_dit through 2 streamed real transformer blocks (tap mode) on
// deterministic random fused-text + latents, once with matmul2d and once with
// VPIPE_KREA2_NO_MMA2 forcing steel, and rel-L2s the joint [text+img, hidden]
// states. Unlike forward_text this exercises the img_in projection (which the
// mma path writes into `joint` at an ELEMENT OFFSET) plus the block q/k/v/gate/
// o + SwiGLU projections through gemm_mma_. stream_blocks=true keeps the full
// 28-block DiT off the heap; only blocks 0..1 are read from the source mmap.
// STREAMED must equal PRELOADED, byte for byte.
//
// The streaming path reads each block into a reusable slot with pread
// and issues the next block's read under the current block's GPU work
// (shared/block-slots.h). That moves where the bytes come from and when
// they arrive; it must not move a single bit of the result.
//
// This model is the case with the sharpest trap in it: its RMSNorm
// weights are stored ZERO-CENTRED and folded to (1 + w) at load, so
// the bytes in the destination are not the bytes in the checkpoint. A
// raw refill would place the checkpoint's and be silently wrong -- the
// buffer is the right size and full of plausible numbers. Those tensors
// are marked kDerived; this test is what says so.
//
// Runs blocks 0..1 in tap mode: enough to exercise the slot pair, the
// prefetch and the refill without the 28-block residual compounding
// that random inputs amplify.
TEST(krea2_dit, streamed_matches_preloaded)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = 128, grid = 8;
  const int img_seq = grid * grid;
  // Two blocks by default -- enough to exercise the slot pair, the
  // prefetch and the refill without a 28-block read on every run.
  // VPIPE_KREA2_STREAM_BLOCKS=-1 runs the whole stack, which is what a
  // TIMING comparison needs: with two blocks the one-off allocation of
  // the second slot is most of what is being measured.
  int stop = 1;
  if (const char* sb = std::getenv("VPIPE_KREA2_STREAM_BLOCKS")) {
    stop = std::atoi(sb);
  }

  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t s = 0x0badf00du;
  auto fill = [&](std::vector<float>& v, float amp) {
    for (auto& e : v) {
      s = s * 1664525u + 1013904223u;
      e = ((float)(s >> 9) / 4194304.0f - 1.0f) * amp;
    }
  };
  fill(txt, 1.0f);
  fill(lat, 1.0f);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = stop >= 0
      ? (std::size_t)(text_seq + img_seq) * HID
      : (std::size_t)img_seq * cfg.in_channels;

  double fwd_s = 0.0;
  auto run = [&](bool stream, std::vector<float>& out) -> bool {
    auto m = MetalKrea2Transformer::load(tdir, mc, cfg, stream);
    if (m == nullptr) { return false; }
    const auto t0 = std::chrono::steady_clock::now();
    SharedBuffer o = m->forward_dit(to_f16buf(txt), text_seq, to_f16buf(lat),
                                    img_seq, grid, grid, 0.5f, stop);
    fwd_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    if (o.empty() || o.byte_size() < n * 2) { return false; }
    const auto* p = static_cast<const _Float16*>(o.contents());
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)p[i]; }
    return true;
  };

  std::vector<float> pre, str;
  ASSERT_TRUE(run(false, pre));
  const double pre_s = fwd_s;
  ASSERT_TRUE(run(true, str));
  std::printf("[krea2_dit] forward: preloaded %.1f s, streamed %.1f s "
              "(slots %s)\n", pre_s, fwd_s,
              std::getenv("VPIPE_KREA2_NO_SLOTS") != nullptr ? "OFF" : "on");

  std::size_t diff = 0;
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (pre[i] != str[i]) {
      ++diff;
      worst = std::max(worst, (double)std::fabs(pre[i] - str[i]));
    }
  }
  std::printf("[krea2_dit] streamed vs preloaded: %zu of %zu differ "
              "(worst %.3e)\n", diff, n, worst);
  EXPECT_TRUE(diff == 0);

  if (const char* out = std::getenv("VPIPE_KREA2_STREAM_DUMP")) {
    std::ofstream f(out, std::ios::binary);
    f.write(reinterpret_cast<const char*>(str.data()),
            (std::streamsize)(str.size() * sizeof(float)));
  }
}

TEST(krea2_dit, forward_dit_mma_matches_steel)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = 128, grid = 8;      // img_seq = 64 -> seq = 192
  const int img_seq = grid * grid;
  const int stop = 1;                       // run blocks 0..1, return joint

  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t s = 0x0badf00du;
  auto fill = [&](std::vector<float>& v, float amp) {
    for (auto& e : v) {
      s = s * 1664525u + 1013904223u;
      e = ((float)(s >> 9) / 4194304.0f - 1.0f) * amp;
    }
  };
  fill(txt, 1.0f);
  fill(lat, 1.0f);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = (std::size_t)(text_seq + img_seq) * HID;

  std::printf("[krea2_dit] forward_dit mma_matches_steel: matrix_cores=%d\n",
              (int)mc->supports_matrix_cores());

  ::unsetenv("VPIPE_KREA2_NO_MMA2");
  auto m_mma =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ASSERT_TRUE(m_mma != nullptr);
  SharedBuffer o_mma = m_mma->forward_dit(to_f16buf(txt), text_seq,
                                          to_f16buf(lat), img_seq, grid, grid,
                                          0.5f, stop);
  ASSERT_TRUE(!o_mma.empty() && o_mma.byte_size() >= n * 2);

  ::setenv("VPIPE_KREA2_NO_MMA2", "1", 1);
  auto m_steel =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ::unsetenv("VPIPE_KREA2_NO_MMA2");
  ASSERT_TRUE(m_steel != nullptr);
  SharedBuffer o_steel = m_steel->forward_dit(to_f16buf(txt), text_seq,
                                              to_f16buf(lat), img_seq, grid,
                                              grid, 0.5f, stop);
  ASSERT_TRUE(!o_steel.empty() && o_steel.byte_size() >= n * 2);

  const auto* a = static_cast<const _Float16*>(o_mma.contents());
  const auto* b = static_cast<const _Float16*>(o_steel.contents());
  std::vector<float> fa(n), fb(n);
  for (std::size_t i = 0; i < n; ++i) { fa[i] = (float)a[i]; fb[i] = (float)b[i]; }
  const double r = rel_l2_(fa.data(), fb.data(), n);
  std::printf("[krea2_dit] forward_dit mma-vs-steel rel-L2 = %.6g (seq=%d)\n", r,
              text_seq + img_seq);
  EXPECT_TRUE(r < 3e-2);
}

// M5 matrix-core int8-accel A/B: the same 2-block forward with the dynamic-int8
// GEMMs (VPIPE_I8_GEMM=1) vs the default bf16 matmul2d path. LOSSY by design
// (int8, rel-L2 ~1e-2 per GEMM), so the bar is the drift budget, not exactness.
// The DiT runs bf16, so the i8 path must load the _bf16 i8 kernels; through the
// f16 kernels it reads the bf16 activation/weight/output buffers as f16 ->
// garbage (rel-L2 ~1). Grid is large enough (joint seq >= 1024) to pass the i8
// M-gate; stream_blocks keeps the DiT off the heap. Off matrix-core GPUs the i8
// path is capability-gated off and this skips vacuously.
TEST(krea2_dit, forward_dit_i8_matches_f16)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = 64, grid = 32;      // img_seq 1024 -> joint 1088 (>=1024)
  const int img_seq = grid * grid;
  const int stop = 1;                       // run blocks 0..1, return joint

  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t s = 0x1278c0deu;
  auto fill = [&](std::vector<float>& v) {
    for (auto& e : v) {
      s = s * 1664525u + 1013904223u;
      e = ((float)(s >> 9) / 4194304.0f - 1.0f);
    }
  };
  fill(txt);
  fill(lat);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = (std::size_t)(text_seq + img_seq) * HID;

  auto run = [&](bool i8) {
    ::setenv("VPIPE_I8_GEMM", i8 ? "1" : "0", 1);
    auto m = MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
    ::unsetenv("VPIPE_I8_GEMM");
    std::vector<float> out;
    if (m == nullptr) { return out; }
    SharedBuffer o = m->forward_dit(to_f16buf(txt), text_seq, to_f16buf(lat),
                                    img_seq, grid, grid, 0.5f, stop);
    if (o.empty() || o.byte_size() < n * 2) { return out; }
    out.resize(n);
    const auto* p = static_cast<const _Float16*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)p[i]; }
    return out;
  };

  const std::vector<float> v_i8 = run(true);
  const std::vector<float> v_f16 = run(false);
  ASSERT_TRUE(!v_i8.empty());
  ASSERT_TRUE(v_i8.size() == v_f16.size());
  const double r = rel_l2_(v_i8.data(), v_f16.data(), v_i8.size());
  std::printf("[krea2_dit] forward_dit i8-vs-f16 rel-L2 = %.6g (seq=%d)\n", r,
              text_seq + img_seq);
  EXPECT_TRUE(std::isfinite(r) && r < 0.10);
}

// SOL-ATTN OVER THE JOINT SEQUENCE, and the two things that have to be
// true of any routing tier before its speed is worth discussing.
//
// 1. TURNING IT ON MUST NOT MOVE THE LAYERS IT DOES NOT ROUTE. With
//    `sol_dense_layers` covering every block the answer has to be
//    BYTE-IDENTICAL to the tier being off -- same kernel, same rounding.
//    That is the ablation which says the filter works, and it is also
//    the bug H3 hit: forcing the routed path off globally sent the dense
//    layers to the scalar kernel, 19x slower, looking exactly like the
//    new kernels being slow.
//
// 2. WHEN IT DOES ROUTE, THE ANSWER MOVES AND STAYS FINITE. An
//    approximation that changed nothing would mean the routing kept
//    everything (or ran nothing at all), and a NaN would mean the merge
//    lost a partial.
//
// The geometry is the i8 test's: joint 1088 tokens is 17 routing blocks
// at the default key block, which clears the 16-block floor forward_dit
// applies. Below that the local band, the sink and the always-exact tail
// cover nearly the whole sequence and the tier declines -- so a smaller
// grid here would assert on a path that never engaged.
TEST(krea2_dit, sol_attn_routes_the_joint_attention)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config base;
  const int HID = base.hidden, IC = base.in_channels;
  // THE GRID IS SWEEPABLE TOO, and it matters more than it looks. At
  // grid 32 the joint sequence is 1088 tokens = 17 routing blocks, and
  // the local band, the sink and the always-exact tail already force
  // most of them exact -- so the THRESHOLD decides very few blocks and
  // tau reads as nearly inert (0.033-0.037 rel-L2 from tau 0.5 to 1.5).
  // That is a property of the fixture, not of tau. VPIPE_KREA2_SOL_GRID
  // =64 gives joint 4160 = 65 routing blocks, the regime a real 1024^2
  // generation is in.
  const char* genv = std::getenv("VPIPE_KREA2_SOL_GRID");
  const int text_seq = 64;
  const int grid = (genv != nullptr && *genv != '\0') ? std::atoi(genv) : 32;
  const int img_seq = grid * grid;
  // HOW MANY BLOCKS ACTUALLY ROUTE, which is the quietest limit on what
  // this test can say. `stop = 1` runs blocks 0..1 and the default
  // `dense_layers = 1` leaves block 0 dense -- so ONE routed block
  // carries the whole measurement, and a tau curve drawn from it can be
  // the behaviour of a single block's proxy score rather than of tau.
  // VPIPE_KREA2_SOL_STOP raises it; error compounds with depth, so the
  // numbers are comparable only at equal depth.
  const char* senv = std::getenv("VPIPE_KREA2_SOL_STOP");
  const int stop = (senv != nullptr && *senv != '\0') ? std::atoi(senv) : 1;

  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t s = 0x50173a7bu;
  auto fill = [&](std::vector<float>& v) {
    for (auto& e : v) {
      s = s * 1664525u + 1013904223u;
      e = ((float)(s >> 9) / 4194304.0f - 1.0f);
    }
  };
  fill(txt);
  fill(lat);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = (std::size_t)(text_seq + img_seq) * HID;

  // TAU IS SWEEPABLE FROM THE ENVIRONMENT, because this is the cheap
  // instrument for the question "what does tau do": ~10 s and a small
  // footprint at seq 1088, against ~7 minutes and a 15 GB peak for one
  // 1024^2 image. Default 1.0, which is the shipped value and what the
  // assertions below are written against.
  const char* tenv = std::getenv("VPIPE_KREA2_SOL_TAU");
  const float tau = (tenv != nullptr && *tenv != '\0')
                        ? (float)std::atof(tenv) : 1.0f;
  auto run = [&](bool on, int dense_layers) {
    MetalKrea2Transformer::Config cfg;
    cfg.sol.enabled      = on;
    cfg.sol.tau          = tau;
    cfg.sol.dense_layers = dense_layers;
    cfg.sol.key_block    = 64;
    cfg.sol.local_radius = 1;
    auto m = MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
    std::vector<float> out;
    if (m == nullptr) { return out; }
    SharedBuffer o = m->forward_dit(to_f16buf(txt), text_seq, to_f16buf(lat),
                                    img_seq, grid, grid, 0.5f, stop);
    if (o.empty() || o.byte_size() < n * 2) { return out; }
    out.resize(n);
    const auto* p = static_cast<const _Float16*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)p[i]; }
    return out;
  };

  // COULD THE BASELINE EVEN RUN? minitest's ASSERT_TRUE records a
  // failure and keeps going, so asserting on the arms below against an
  // empty baseline would report a Sol bug where the truth is that no
  // forward happened at all -- a checkpoint this box cannot load, or a
  // geometry it cannot fit. Say which, and stop.
  const std::vector<float> off = run(false, 0);
  if (off.empty()) {
    std::printf("[krea2_dit] sol: the dense baseline produced no forward "
                "(load or geometry); nothing to compare\n");
    ASSERT_TRUE(!off.empty());
    return;
  }

  // ALL DENSE: the tier is on and routes nothing, so every block runs
  // the attention it would have run anyway. Exact equality, not a
  // tolerance -- a difference here is the tier disturbing layers it was
  // told to leave alone.
  const std::vector<float> all_dense = run(true, base.n_layers);
  ASSERT_TRUE(all_dense.size() == off.size());
  std::size_t diff = 0;
  for (std::size_t i = 0; i < off.size(); ++i) {
    if (off[i] != all_dense[i]) { ++diff; }
  }
  std::printf("[krea2_dit] sol all-dense vs off: %zu of %zu elements differ\n",
              diff, off.size());
  EXPECT_TRUE(diff == 0);

  // ROUTED: both blocks this forward runs are routed.
  const std::vector<float> routed = run(true, 0);
  ASSERT_TRUE(routed.size() == off.size());
  const double r = rel_l2_(routed.data(), off.data(), routed.size());
  // REALIZED SPARSITY BESIDE THE ERROR, because rel-L2 alone cannot say
  // whether a tau did nothing because the threshold was loose or because
  // the local band and the sink had already kept everything. Sol counts
  // steel key blocks, so `kept` is in those units and the fraction is
  // what a speed claim would have to be built on.
  // The routed arm above runs with dense_layers = 0, so every block this
  // forward executes is routed. Say that from the arm's own setting
  // rather than from `base`, whose sol.dense_layers is the shipped
  // default (1) and would misreport it for every arm.
  std::printf("[krea2_dit] sol routed vs dense rel-L2 = %.6g (seq=%d, "
              "grid=%d, tau=%.2f, %d routing blocks, %d routed layer(s))\n",
              r, text_seq + img_seq, grid, (double)tau,
              (text_seq + img_seq + 63) / 64, stop + 1);
  EXPECT_TRUE(std::isfinite(r));
  // It approximated something...
  EXPECT_TRUE(r > 1e-5);
  // ...but it is still attention over the same sequence, not noise. The
  // bar is loose on purpose: realized sparsity is a property of the
  // activations, and two blocks of a 28-block stack is not a quality
  // claim about a generated image.
  EXPECT_TRUE(r < 0.5);
}

// Reference-image conditioning (Qwen-Image-Edit multi-reference): forward_dit
// with reference latents (a) still emits ONLY the generated-token velocity
// [img_seq, in_channels] (refs dropped from the output), (b) stays finite with
// one and two references, and (c) actually CHANGES the velocity vs the no-ref
// forward -- proving the reference tokens are embedded via img_in, frame-offset
// on the RoPE and folded into the joint attention. A dead ref path would give a
// byte-identical result. Note: this checks the MECHANISM is wired, not that a
// text-to-image checkpoint semantically uses references. Same random inputs.
// The ANE feed-forward, against the same forward with the tier off.
//
// Rows are independent in a feed-forward, so giving the ANE a band of
// them is EXACT arithmetic -- the only difference is fp16 on the ANE
// against bf16 on the GPU, and that is safe here because the FFN is
// self-contained: only its output re-enters the residual stream, and
// that addition stays on the GPU. So this is a PRECISION comparison,
// not a correctness one, and a broken split (wrong rows, wrong offset,
// stale `o`) lands nowhere near the bar.
//
// Runs on the shared runtime-weight module the tier uses -- compiled once
// per shape (cached after the first), weights staged per block into
// IOSurface slots -- so it needs no templates and bakes nothing. Capped at
// the two blocks `stop` runs.
TEST(krea2_dit, forward_dit_ane_matches_gpu)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config base;
  const int HID = base.hidden, IC = base.in_channels;
  // Default grid 64 == 1024x1024, seq 4160: the ANE path is CHUNKED at
  // 2048 rows, so a smaller grid gives it fewer than one whole chunk
  // and the tier correctly switches itself off -- which this test then
  // reads as "the tier did nothing", because it asserts the two runs
  // DIFFER.
  const char* gs = std::getenv("VPIPE_KREA2_ANE_GRID");
  const int text_seq = 64;
  const int grid = (gs != nullptr) ? std::atoi(gs) : 64;
  const int img_seq = grid * grid;
  const int stop = 1;                        // blocks 0..1
  const int seq = text_seq + img_seq;

  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t sd = 0x51ced00du;
  auto fill = [&](std::vector<float>& v) {
    for (auto& e : v) {
      sd = sd * 1664525u + 1013904223u;
      e = ((float)(sd >> 9) / 4194304.0f - 1.0f);
    }
  };
  fill(txt);
  fill(lat);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = (std::size_t)seq * HID;

  const char* lora_path = std::getenv("VPIPE_KREA2_ANE_LORA");
  const bool qkv = std::getenv("VPIPE_KREA2_ANE_QKV") != nullptr;
  auto run = [&](bool ane, bool with_lora, bool int8 = false,
                 int* n8 = nullptr) {
    MetalKrea2Transformer::Config cfg = base;
    if (ane) {
      cfg.session       = &sess;
      cfg.ane_rows      = 0.5f;     // 544 of 1088
      cfg.ane_layers    = stop + 1;
      cfg.ane_qkv = qkv;
      cfg.ane_int8 = int8;
    }
    std::vector<float> out;
    // VPIPE_KREA2_ANE_LORA: a runtime adapter on BOTH arms. The GPU applies
    // it as side GEMMs; the ANE's rows must get it merged into their staged
    // weights, or this comparison measures the adapter rather than fp16.
    std::vector<MetalKrea2Transformer::LoraSpec> loras;
    if (with_lora && lora_path != nullptr) {
      MetalKrea2Transformer::LoraSpec sp;
      sp.path = lora_path;
      loras.push_back(sp);
    }
    auto m = MetalKrea2Transformer::load(tdir, mc, cfg,
                                         /*stream_blocks=*/true, loras);
    if (m == nullptr) { return out; }
    SharedBuffer o = m->forward_dit(to_f16buf(txt), text_seq, to_f16buf(lat),
                                    img_seq, grid, grid, 0.5f, stop);
    if (n8 != nullptr) { *n8 = m->ane_int8_modules(); }
    if (o.empty() || o.byte_size() < n * 2) { return out; }
    out.resize(n);
    const auto* p = static_cast<const _Float16*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)p[i]; }
    return out;
  };

  const std::vector<float> v_gpu = run(false, true);
  const std::vector<float> v_ane = run(true, true);
  ASSERT_TRUE(!v_gpu.empty());
  ASSERT_TRUE(v_ane.size() == v_gpu.size());
  if (v_ane.size() != v_gpu.size()) { return; }
  const double r = rel_l2_(v_ane.data(), v_gpu.data(), v_ane.size());
  std::printf("[krea2_dit] forward_dit ANE-vs-GPU rel-L2 = %.6g "
              "(%dx%d, seq=%d)\n", r, grid * 16, grid * 16, seq);
  // The two runs must DIFFER -- an identical result means the tier
  // silently did nothing, which is the failure mode a tolerance alone
  // would pass.
  EXPECT_TRUE(r > 0.0);
  EXPECT_TRUE(std::isfinite(r) && r < 0.10);
  // ROTATED INT8 (ane_int8): the same tiers in the GPU token tier's
  // arithmetic, ENGAGED (each module that armed) before measured.
  {
    int n8 = -1;
    const std::vector<float> v8 = run(true, true, true, &n8);
    ASSERT_TRUE(v8.size() == v_gpu.size());
    if (v8.size() != v_gpu.size()) { return; }
    const double r8 = rel_l2_(v8.data(), v_gpu.data(), v8.size());
    std::printf("[krea2_dit] ... in rotated int8 (%d modules int8): "
                "rel-L2 = %.6g\n", n8, r8);
    if (ane_int8_expected()) {
      EXPECT_TRUE(n8 == (qkv ? 2 : 1));
      EXPECT_TRUE(r8 > 0.0 && std::isfinite(r8) && r8 < 0.10);
    }
  }
  // With an adapter, the CONTROLS: how far the adapter itself moves the
  // GPU's output, and the ANE's fp16 floor on the same run WITHOUT it. An
  // ANE whose rows dropped the adapter adds roughly that effect (over its
  // share of the rows) on top of the floor; one that merged it sits AT the
  // floor. The floor is measured, not assumed: with q/k/v/gate on the ANE
  // too, a 4-bit checkpoint's rises to ~0.011 (the q/k fp16 error goes
  // through the q/k norms and the attention), above half the ~0.016 effect
  // a fixed fraction would have demanded.
  if (lora_path != nullptr) {
    const std::vector<float> v_base = run(false, false);
    const std::vector<float> v_ane_base = run(true, false);
    ASSERT_TRUE(v_base.size() == v_gpu.size() &&
                v_ane_base.size() == v_gpu.size());
    if (v_base.size() != v_gpu.size() ||
        v_ane_base.size() != v_gpu.size()) {
      return;
    }
    const double eff = rel_l2_(v_base.data(), v_gpu.data(), v_gpu.size());
    const double floor = rel_l2_(v_ane_base.data(), v_base.data(),
                                 v_base.size());
    std::printf("[krea2_dit] adapter effect on the GPU output rel-L2 = %.6g, "
                "ANE fp16 floor without it %.6g (ANE-vs-GPU with it %.6g)\n",
                eff, floor, r);
    EXPECT_TRUE(eff > 0.0);
    EXPECT_TRUE(r < 1.25 * floor + 1e-4);
  }
}

// Does the ANE split actually make the block faster?
//
// Times the SECOND forward of each arm, so the one-time module compile
// and first staging are excluded and what is left is the steady
// state a real run would see. Opt-in, since
// each arm loads the DiT again.
TEST(krea2_dit, ane_split_is_faster)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || std::getenv("VPIPE_KREA2_ANE_BENCH") == nullptr) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config base;
  const int HID = base.hidden, IC = base.in_channels;
  // Krea-2 patches 2x2 over an 8x VAE downsample, so one token is 16
  // pixels: grid 32 == 512x512, 64 == 1024x1024, 128 == 2048x2048.
  // The default is the SMALL one because it is quick; 1024x1024 is the
  // case that matters and is where the GPU's feed-forward should stop
  // being insensitive to row count.
  const char* gs = std::getenv("VPIPE_KREA2_ANE_GRID");
  const int text_seq = 64;
  const int grid = (gs != nullptr) ? std::atoi(gs) : 32;
  const int img_seq = grid * grid, stop = 1;
  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t sd = 0x1a2b3c4du;
  auto fill = [&](std::vector<float>& v) {
    for (auto& e : v) {
      sd = sd * 1664525u + 1013904223u;
      e = ((float)(sd >> 9) / 4194304.0f - 1.0f);
    }
  };
  fill(txt);
  fill(lat);
  auto buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };

  auto timed = [&](bool ane) {
    MetalKrea2Transformer::Config cfg = base;
    if (ane) {
      cfg.session = &sess;
      // Sweepable. The CHUNKED path rounds this down to whole 2048-row
      // chunks, so one template serves every ratio.
      // UNSET means 0, which the family reads as "auto-balance".
      const char* rr = std::getenv("VPIPE_KREA2_ANE_ROWS");
      cfg.ane_rows = (rr != nullptr) ? (float)std::atof(rr) : 0.0f;
      cfg.ane_layers = stop + 1;
      cfg.ane_qkv = std::getenv("VPIPE_KREA2_ANE_QKV") != nullptr;
    }
    // PRELOADED, not streamed: a streamed run re-reads ~850 MB per
    // block per forward, which dwarfs the ~100 ms of feed-forward and
    // hides the thing being measured. The first attempt timed 1868 ms
    // per block that way and read 0.93x -- disk, not compute.
    auto m = MetalKrea2Transformer::load(tdir, mc, cfg, false);
    if (m == nullptr) { return 0.0; }
    m->forward_dit(buf(txt), text_seq, buf(lat), img_seq, grid, grid, 0.5f,
                   stop);                      // warm + bake
    const auto t0 = std::chrono::steady_clock::now();
    m->forward_dit(buf(txt), text_seq, buf(lat), img_seq, grid, grid, 0.5f,
                   stop);
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
  };

  const double gpu = timed(false);
  const double ane = timed(true);
  const char* rs = std::getenv("VPIPE_KREA2_ANE_ROWS");
  const double share = (rs != nullptr) ? std::atof(rs) : 0.5;
  std::printf("[krea2_dit] %dx%d, %d blocks, seq %d: GPU-only %.1f ms | "
              "ANE split (%d/%d) %.1f ms | %.3fx\n", grid * 16, grid * 16,
              stop + 1, text_seq + img_seq, gpu,
              (int)((text_seq + img_seq) * share), text_seq + img_seq, ane,
              gpu > 0 ? gpu / ane : 0.0);
  EXPECT_TRUE(gpu > 0.0 && ane > 0.0);
}

TEST(krea2_dit, forward_dit_reference_images_change_output)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = 96, grid = 8, img_seq = grid * grid;

  std::uint32_t s = 0x51eed00du;
  auto fill = [&](_Float16* d, std::size_t cnt, float amp) {
    for (std::size_t i = 0; i < cnt; ++i) {
      s = s * 1664525u + 1013904223u;
      d[i] = (_Float16)(((float)(s >> 9) / 4194304.0f - 1.0f) * amp);
    }
  };
  auto rand_buf = [&](std::size_t cnt, float amp) {
    SharedBuffer b = mc->make_shared_buffer(cnt * 2);
    fill(static_cast<_Float16*>(b.contents()), cnt, amp);
    return b;
  };
  // A random packed reference latent [rseq, in_channels] on a rgh x rgw grid,
  // mirroring the 2x2-patchified vae-encode latent the stage feeds.
  auto make_ref = [&](int rgh, int rgw) -> MetalKrea2Transformer::RefImage {
    MetalKrea2Transformer::RefImage r;
    r.seq = rgh * rgw; r.grid_h = rgh; r.grid_w = rgw;
    r.latents = rand_buf((std::size_t)r.seq * IC, 1.0f);
    return r;
  };

  auto dit = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(dit != nullptr);
  SharedBuffer txt = rand_buf((std::size_t)text_seq * HID, 1.0f);
  SharedBuffer lat = rand_buf((std::size_t)img_seq * IC, 1.0f);

  const std::size_t n = (std::size_t)img_seq * IC;
  auto to_vec = [&](const SharedBuffer& b) {
    std::vector<float> v(n);
    const auto* p = static_cast<const _Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)p[i]; }
    return v;
  };

  SharedBuffer v0 = dit->forward_dit(txt, text_seq, lat, img_seq, grid, grid,
                                     0.5f);
  ASSERT_TRUE(!v0.empty() && v0.byte_size() >= n * 2);
  // RefImage is move-only (holds a SharedBuffer), so move into the vectors.
  std::vector<MetalKrea2Transformer::RefImage> one;
  one.push_back(make_ref(4, 6));
  SharedBuffer v1 = dit->forward_dit(txt, text_seq, lat, img_seq, grid, grid,
                                     0.5f, -1, one);
  ASSERT_TRUE(!v1.empty());
  ASSERT_TRUE(v1.byte_size() >= n * 2);   // still generated-token count only
  std::vector<MetalKrea2Transformer::RefImage> two;
  two.push_back(make_ref(4, 6));
  two.push_back(make_ref(6, 4));
  SharedBuffer v2 = dit->forward_dit(txt, text_seq, lat, img_seq, grid, grid,
                                     0.5f, -1, two);
  ASSERT_TRUE(!v2.empty() && v2.byte_size() >= n * 2);

  const std::vector<float> a = to_vec(v0), b1 = to_vec(v1), b2 = to_vec(v2);
  bool finite = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(b1[i]) || !std::isfinite(b2[i])) { finite = false; break; }
  }
  EXPECT_TRUE(finite);
  const double r1 = rel_l2_(b1.data(), a.data(), n);
  const double r2 = rel_l2_(b2.data(), a.data(), n);
  std::printf("[krea2_dit] ref-image velocity delta: 1 ref rel-L2 %.4f, 2 refs "
              "rel-L2 %.4f (vs no-ref)\n", r1, r2);
  EXPECT_TRUE(r1 > 1e-3);          // a reference materially changes the output
  EXPECT_TRUE(r2 > 1e-3);
}

// Full-DiT velocity golden: the bf16 DiT (forward_text -> forward_dit) against a
// diffusers Krea2Transformer2DModel reference (dump_dit.py, bf16 -- the model's
// native dtype). Feeds the REAL 18-token fox conditioning (a1_prompt_embeds,
// txt_in absmax ~82 -- the large-magnitude activations that stress the residual
// stream and would overflow an f16 DiT, exactly the FLUX.2 class of bug this
// bf16 switch fixes) plus the golden's deterministic packed latents, at t=0.5 on
// a 16x16 grid. Checks (a) the txt_in text-fusion vs d_txtin and (b) the
// velocity vs d_velocity, both to a bf16-vs-bf16 rounding floor. Env:
// VPIPE_KREA2_TEST_MODEL_PATH + VPIPE_KREA2_GOLDEN (needs d_latents/d_velocity).
TEST(krea2_dit, forward_dit_velocity_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";
  const std::string gdir = gd;

  const std::vector<float> pe   = read_f32_(gdir + "/a1_prompt_embeds.f32");
  const std::vector<float> gtx  = read_f32_(gdir + "/d_txtin.f32");
  const std::vector<float> glat = read_f32_(gdir + "/d_latents.f32");
  const std::vector<float> gvel = read_f32_(gdir + "/d_velocity.f32");
  if (pe.empty() || glat.empty() || gvel.empty()) { return; }   // no DiT golden

  MetalKrea2Transformer::Config cfg;
  const int NL = cfg.n_text_layers, TH = cfg.text_hidden, HID = cfg.hidden;
  const int IC = cfg.in_channels;
  ASSERT_TRUE(pe.size() % ((std::size_t)NL * TH) == 0);
  const int text_seq = (int)(pe.size() / ((std::size_t)NL * TH));
  const int img_seq  = (int)(glat.size() / (std::size_t)IC);
  const int grid = (int)(std::lround(std::sqrt((double)img_seq)));
  ASSERT_TRUE(grid * grid == img_seq);          // square grid (dump_dit default)
  ASSERT_TRUE(gvel.size() == (std::size_t)img_seq * IC);

  auto m = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  // f16 conditioning taps [text_seq, 12, 2560] and packed latents [img_seq, IC].
  auto to_buf = [&](const std::vector<float>& v) {
    SharedBuffer b = mc->make_shared_buffer(v.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { d[i] = (_Float16)v[i]; }
    return b;
  };
  SharedBuffer ehs = to_buf(pe);
  SharedBuffer lat = to_buf(glat);

  SharedBuffer fused = m->forward_text(ehs, text_seq);
  ASSERT_TRUE(!fused.empty());
  if (!gtx.empty() && gtx.size() == (std::size_t)text_seq * HID) {
    const auto* fp = static_cast<const _Float16*>(fused.contents());
    std::vector<float> ftx((std::size_t)text_seq * HID);
    for (std::size_t i = 0; i < ftx.size(); ++i) { ftx[i] = (float)fp[i]; }
    const double rt = rel_l2_(ftx.data(), gtx.data(), ftx.size());
    std::printf("[krea2_dit] velocity-golden txt_in rel-L2 = %.6f\n", rt);
    EXPECT_TRUE(rt < 0.06);
  }

  SharedBuffer vel = m->forward_dit(fused, text_seq, lat, img_seq, grid, grid,
                                    0.5f);
  ASSERT_TRUE(!vel.empty());
  ASSERT_TRUE(vel.byte_size() >= (std::size_t)img_seq * IC * 2);
  const auto* vp = static_cast<const _Float16*>(vel.contents());
  std::vector<float> got((std::size_t)img_seq * IC);
  bool finite = true;
  for (std::size_t i = 0; i < got.size(); ++i) {
    got[i] = (float)vp[i];
    if (!std::isfinite(got[i])) { finite = false; }
  }
  EXPECT_TRUE(finite);                           // the whole point: no f16 NaN
  const double r = rel_l2_(got.data(), gvel.data(), got.size());
  std::printf("[krea2_dit] forward_dit velocity rel-L2 = %.6f "
              "(seq=%d img=%d grid=%d) [bf16-vs-bf16 floor]\n",
              r, text_seq + img_seq, img_seq, grid);
  EXPECT_TRUE(r < 0.10);                          // bf16-vs-bf16 rounding floor
}

// A/B oracle for the BM=128 quantized qmm tile (_qmm_tile=2) vs BM=64
// (_qmm_tile=1): the tall-M steel qmm + fused-SwiGLU GEMMs pick a 128-row
// output tile (WM=4, 256 threads) when seq >= 1024, quartering the fused
// gate|up code re-reads (that GEMM had stayed at BM=32). Needs a QUANTIZED
// transformer dir (VPIPE_KREA2_QMM_DIR, e.g. Krea-2-Turbo-dit-w4g64); runs
// forward_dit through 2 streamed blocks (tap) at grid=32 (img_seq=1024,
// seq=1152 -> the BM=128 gate engages) on deterministic random inputs and
// rel-L2s the joint. Different tiling changes only the f32 accumulation
// order, so the two paths are numerically equivalent (bar 3e-2, expect
// ~1e-3). Skips vacuously without the dir. Only meaningful on the steel qmm
// path (M4 / no matrix cores); on M5 the quant GEMMs route through mma2 and
// both tiles run identical, so rel-L2 ~0 (no regression).
TEST(krea2_dit, forward_dit_bm128_matches_bm64)
{
  const char* qdir = std::getenv("VPIPE_KREA2_QMM_DIR");
  if (qdir == nullptr || *qdir == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = 128, grid = 32;     // img_seq = 1024 -> seq = 1152
  const int img_seq = grid * grid;
  const int stop = 1;                       // run blocks 0..1, return joint

  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t s = 0x0badf00du;
  auto fill = [&](std::vector<float>& v, float amp) {
    for (auto& e : v) {
      s = s * 1664525u + 1013904223u;
      e = ((float)(s >> 9) / 4194304.0f - 1.0f) * amp;
    }
  };
  fill(txt, 1.0f);
  fill(lat, 1.0f);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = (std::size_t)(text_seq + img_seq) * HID;

  ::setenv("VPIPE_KREA2_QMM_TILE", "2", 1);
  auto m128 =
      MetalKrea2Transformer::load(qdir, mc, cfg, /*stream_blocks=*/true);
  ASSERT_TRUE(m128 != nullptr);
  SharedBuffer o128 = m128->forward_dit(to_f16buf(txt), text_seq,
                                        to_f16buf(lat), img_seq, grid, grid,
                                        0.5f, stop);
  ASSERT_TRUE(!o128.empty() && o128.byte_size() >= n * 2);

  ::setenv("VPIPE_KREA2_QMM_TILE", "1", 1);
  auto m64 =
      MetalKrea2Transformer::load(qdir, mc, cfg, /*stream_blocks=*/true);
  ::unsetenv("VPIPE_KREA2_QMM_TILE");
  ASSERT_TRUE(m64 != nullptr);
  SharedBuffer o64 = m64->forward_dit(to_f16buf(txt), text_seq,
                                      to_f16buf(lat), img_seq, grid, grid,
                                      0.5f, stop);
  ASSERT_TRUE(!o64.empty() && o64.byte_size() >= n * 2);

  const auto* a = static_cast<const _Float16*>(o128.contents());
  const auto* b = static_cast<const _Float16*>(o64.contents());
  std::vector<float> fa(n), fb(n);
  for (std::size_t i = 0; i < n; ++i) { fa[i] = (float)a[i]; fb[i] = (float)b[i]; }
  const double r = rel_l2_(fa.data(), fb.data(), n);
  std::printf("[krea2_dit] forward_dit bm128-vs-bm64 rel-L2 = %.6g (seq=%d)\n", r,
              text_seq + img_seq);
  EXPECT_TRUE(r < 3e-2);
}

// A/B oracle for the DiT's M5 matrix-core NAX attention (attn_steel_nax_h_bd128)
// vs the ALU steel (attn_steel_h_bd128): runs forward_dit through 2 streamed
// blocks (tap mode) on deterministic random inputs, once with NAX (default on
// M5) and once with VPIPE_KREA2_NO_ATTN_NAX forcing the ALU steel, and rel-L2s
// the joint state. On M4 (no matrix cores) BOTH paths are the ALU steel, so the
// rel-L2 is ~0 (the test passes trivially, validating no wiring regression);
// the REAL NAX-vs-steel comparison happens when run on M5 (matrix_cores=1).
TEST(krea2_dit, forward_dit_nax_matches_steel)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = 10, grid = 16;      // img_seq = 256 -> seq = 266
  const int img_seq = grid * grid;
  const int stop = 1;                       // run blocks 0..1, return joint

  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t s = 0x51ee7a11u;
  auto fill = [&](std::vector<float>& v) {
    for (auto& e : v) {
      s = s * 1664525u + 1013904223u;
      e = ((float)(s >> 9) / 4194304.0f - 1.0f);
    }
  };
  fill(txt);
  fill(lat);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = (std::size_t)(text_seq + img_seq) * HID;

  std::printf("[krea2_dit] forward_dit nax_matches_steel: matrix_cores=%d "
              "(NAX active only when 1)\n", (int)mc->supports_matrix_cores());

  ::unsetenv("VPIPE_KREA2_NO_ATTN_NAX");
  auto m_nax =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ASSERT_TRUE(m_nax != nullptr);
  SharedBuffer o_nax = m_nax->forward_dit(to_f16buf(txt), text_seq,
                                          to_f16buf(lat), img_seq, grid, grid,
                                          0.5f, stop);
  ASSERT_TRUE(!o_nax.empty() && o_nax.byte_size() >= n * 2);

  ::setenv("VPIPE_KREA2_NO_ATTN_NAX", "1", 1);
  auto m_steel =
      MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
  ::unsetenv("VPIPE_KREA2_NO_ATTN_NAX");
  ASSERT_TRUE(m_steel != nullptr);
  SharedBuffer o_steel = m_steel->forward_dit(to_f16buf(txt), text_seq,
                                              to_f16buf(lat), img_seq, grid,
                                              grid, 0.5f, stop);
  ASSERT_TRUE(!o_steel.empty() && o_steel.byte_size() >= n * 2);

  const auto* a = static_cast<const _Float16*>(o_nax.contents());
  const auto* b = static_cast<const _Float16*>(o_steel.contents());
  std::vector<float> fa(n), fb(n);
  for (std::size_t i = 0; i < n; ++i) { fa[i] = (float)a[i]; fb[i] = (float)b[i]; }
  const double r = rel_l2_(fa.data(), fb.data(), n);
  std::printf("[krea2_dit] forward_dit nax-vs-steel rel-L2 = %.6g (seq=%d)\n", r,
              text_seq + img_seq);
  // Both fp32 online softmax -> within the flash tolerance (0 on M4, small on
  // M5). Bar matches the mma-vs-steel oracle.
  EXPECT_TRUE(r < 3e-2);
}

// M3b: img_in + time_embed + ONE main DiT block (adaLN + Flux 3-axis rope +
// GQA 48/12 + sigmoid gate). Feed the golden fused text (d_txtin), packed
// latents (a3_step0_latin) and timestep (a3_step0_tstep); rel-L2 the joint
// [text+img, hidden] state after block 0 against d_blk0.f32.
TEST(krea2_dit, block0_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";
  const std::string gdir = gd;

  const std::vector<float> txt   = read_f32_(gdir + "/d_txtin.f32");
  const std::vector<float> latin = read_f32_(gdir + "/a3_step0_latin.f32");
  const std::vector<float> tstep = read_f32_(gdir + "/a3_step0_tstep.f32");
  const std::vector<float> blk0  = read_f32_(gdir + "/d_blk0.f32");
  ASSERT_TRUE(!txt.empty() && !latin.empty() && !tstep.empty() && !blk0.empty());

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = (int)(txt.size() / HID);
  const int img_seq = (int)(latin.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }          // square latent grid
  ASSERT_TRUE(grid * grid == img_seq);
  ASSERT_TRUE(blk0.size() == (std::size_t)(text_seq + img_seq) * HID);

  auto m = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  SharedBuffer fused = to_f16buf(txt);
  SharedBuffer lat = to_f16buf(latin);

  SharedBuffer out = m->forward_dit(fused, text_seq, lat, img_seq, grid, grid,
                                    tstep[0], /*stop_after_block=*/0);
  ASSERT_TRUE(!out.empty());
  ASSERT_TRUE(out.byte_size() >= blk0.size() * 2);

  const auto* op = static_cast<const _Float16*>(out.contents());
  std::vector<float> got(blk0.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)op[i]; }

  const double r = rel_l2_(got.data(), blk0.data(), got.size());
  std::printf("[krea2_dit] block0 rel-L2 = %.6f (text=%d img=%d grid=%d)\n",
              r, text_seq, img_seq, grid);
  EXPECT_TRUE(r < 0.05);
}

// M3c: the FULL single denoiser step -- all 28 blocks + final_layer -> the
// predicted velocity. rel-L2 the (img_seq, in_channels) output against
// a3_step0_noise.f32 (the HF transformer's step-0 velocity).
TEST(krea2_dit, single_step_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";
  const std::string gdir = gd;

  const std::vector<float> txt   = read_f32_(gdir + "/d_txtin.f32");
  const std::vector<float> latin = read_f32_(gdir + "/a3_step0_latin.f32");
  const std::vector<float> tstep = read_f32_(gdir + "/a3_step0_tstep.f32");
  const std::vector<float> noise = read_f32_(gdir + "/a3_step0_noise.f32");
  ASSERT_TRUE(!txt.empty() && !latin.empty() && !tstep.empty() && !noise.empty());

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = (int)(txt.size() / HID);
  const int img_seq = (int)(latin.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }
  ASSERT_TRUE(grid * grid == img_seq);
  ASSERT_TRUE(noise.size() == (std::size_t)img_seq * IC);

  auto m = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  SharedBuffer fused = to_f16buf(txt);
  SharedBuffer lat = to_f16buf(latin);

  SharedBuffer out = m->forward_dit(fused, text_seq, lat, img_seq, grid, grid,
                                    tstep[0], /*stop_after_block=*/-1);
  ASSERT_TRUE(!out.empty());
  ASSERT_TRUE(out.byte_size() >= noise.size() * 2);

  const auto* op = static_cast<const _Float16*>(out.contents());
  std::vector<float> got(noise.size());
  for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)op[i]; }

  const double r = rel_l2_(got.data(), noise.data(), got.size());
  std::printf("[krea2_dit] single_step rel-L2 = %.6f\n", r);
  EXPECT_TRUE(r < 0.05);
}

// M3d: the full 8-step FlowMatchEuler turbo sampler (free-running) driving the
// metal DiT. Start from a3_step0_latin, step latents += (sigma[i+1]-sigma[i]) *
// velocity with the golden sigma schedule, and rel-L2 the final packed latents
// against a3_final_latents.f32. (Fused text is the golden d_txtin, isolating the
// DiT+sampler from the text tower.)
TEST(krea2_dit, sampler_matches_golden)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* gd   = std::getenv("VPIPE_KREA2_GOLDEN");
  if (root == nullptr || *root == '\0' || gd == nullptr || *gd == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";
  const std::string gdir = gd;

  const std::vector<float> txt   = read_f32_(gdir + "/d_txtin.f32");
  const std::vector<float> latin = read_f32_(gdir + "/a3_step0_latin.f32");
  const std::vector<float> fin   = read_f32_(gdir + "/a3_final_latents.f32");
  ASSERT_TRUE(!txt.empty() && !latin.empty() && !fin.empty());

  // The turbo (mu=1.15 dynamic-shift) sigma schedule from the golden meta.json
  // (linspace(1, 1/8, 8) shifted); 9 values incl. the terminal 0.
  const double sig[9] = {1.0, 0.956723690032959, 0.9045307636260986,
                         0.8403487801551819, 0.7595109343528748,
                         0.654566764831543, 0.5128441452980042,
                         0.31090107560157776, 0.0};

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = (int)(txt.size() / HID);
  const int img_seq = (int)(latin.size() / IC);
  int grid = 1;
  while (grid * grid < img_seq) { ++grid; }
  ASSERT_TRUE(grid * grid == img_seq);

  auto m = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  SharedBuffer fused = mc->make_shared_buffer(txt.size() * 2);
  { auto* d = static_cast<_Float16*>(fused.contents());
    for (std::size_t i = 0; i < txt.size(); ++i) { d[i] = (_Float16)txt[i]; } }

  SharedBuffer latbuf = mc->make_shared_buffer(latin.size() * 2);
  double worst_tf = 0.0;
  for (int s = 0; s < 8; ++s) {
    // TEACHER-FORCED per-step check: feed the golden latin[s], apply one Euler
    // step with our velocity, compare to the golden latin[s+1] (or, for the
    // last step, a3_final_latents). Isolates per-step correctness from f16
    // trajectory drift.
    const std::string cur = "/a3_step" + std::to_string(s) + "_latin.f32";
    const std::vector<float> gl = read_f32_(gdir + cur);
    ASSERT_TRUE(!gl.empty());
    std::vector<float> gnext = (s < 7)
        ? read_f32_(gdir + "/a3_step" + std::to_string(s + 1) + "_latin.f32")
        : fin;
    ASSERT_TRUE(!gnext.empty());
    auto* lb = static_cast<_Float16*>(latbuf.contents());
    for (std::size_t i = 0; i < gl.size(); ++i) { lb[i] = (_Float16)gl[i]; }
    SharedBuffer np = m->forward_dit(fused, text_seq, latbuf, img_seq, grid,
                                     grid, (float)sig[s], -1);
    ASSERT_TRUE(!np.empty());
    const auto* p = static_cast<const _Float16*>(np.contents());
    const double dt = sig[s + 1] - sig[s];
    std::vector<float> stepped(gl.size());
    for (std::size_t i = 0; i < gl.size(); ++i) {
      stepped[i] = gl[i] + (float)(dt * (double)(float)p[i]);
    }
    const double rt = rel_l2_(stepped.data(), gnext.data(), stepped.size());
    std::printf("[krea2_dit]   teacher-forced step %d rel-L2 = %.6f\n", s, rt);
    if (rt > worst_tf) { worst_tf = rt; }
  }
  std::printf("[krea2_dit] sampler teacher-forced worst per-step rel-L2 = %.6f\n",
              worst_tf);
  EXPECT_TRUE(worst_tf < 0.03);
}

// Perf bench (opt-in, no golden): time forward_dit at a target latent grid so
// the per-section breakdown (VPIPE_KREA2_DIT_PROFILE) can be read at real
// resolutions. Gated on VPIPE_KREA2_DIT_BENCH so it never runs in the default
// suite. Env:
//   VPIPE_KREA2_DIT_BENCH       -- enable.
//   VPIPE_KREA2_TEST_MODEL_PATH -- model root (uses <root>/transformer unless
//                                  overridden).
//   VPIPE_KREA2_DIT_BENCH_DIR   -- explicit transformer dir (e.g. a quantized
//                                  variant) overriding <root>/transformer.
//   VPIPE_KREA2_DIT_BENCH_GRID  -- latent grid side (default 64 => 1024px; a
//                                  256px image is grid 16, 512px is 32).
//   VPIPE_KREA2_DIT_BENCH_ITERS -- timed forward_dit calls (default 3).
// Forces VPIPE_KREA2_DIT_PROFILE on so each call logs its section breakdown.
TEST(krea2_dit, forward_dit_bench)
{
  if (std::getenv("VPIPE_KREA2_DIT_BENCH") == nullptr) { return; }
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  const char* dir_override = std::getenv("VPIPE_KREA2_DIT_BENCH_DIR");
  if ((root == nullptr || *root == '\0')
      && (dir_override == nullptr || *dir_override == '\0')) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = (dir_override != nullptr && *dir_override != '\0')
      ? std::string(dir_override)
      : std::string(root) + "/transformer";

  int grid = 64;
  if (const char* g = std::getenv("VPIPE_KREA2_DIT_BENCH_GRID")) {
    const int gv = std::atoi(g);
    if (gv > 0) { grid = gv; }
  }
  int iters = 3;
  if (const char* it = std::getenv("VPIPE_KREA2_DIT_BENCH_ITERS")) {
    const int iv = std::atoi(it);
    if (iv > 0) { iters = iv; }
  }
  const int text_seq = 10;                   // typical compact prompt length
  const int img_seq = grid * grid;

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  std::printf("[krea2_dit] bench: dir=%s grid=%d img_seq=%d seq=%d iters=%d "
              "matrix_cores=%d\n", tdir.c_str(), grid, img_seq,
              text_seq + img_seq, iters, (int)mc->supports_matrix_cores());

  auto m = MetalKrea2Transformer::load(tdir, mc, cfg);
  ASSERT_TRUE(m != nullptr);

  std::uint32_t s = 0x1234567u;
  auto rndbuf = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      s = s * 1664525u + 1013904223u;
      d[i] = (_Float16)((float)(s >> 9) / 4194304.0f - 1.0f);
    }
    return b;
  };
  SharedBuffer fused = rndbuf((std::size_t)text_seq * HID);
  SharedBuffer latbuf = rndbuf((std::size_t)img_seq * IC);

  // Profile the FIRST iter (section breakdown, barriers inflate its wall time);
  // run the rest barrier-free so best/mean reflect the true step latency.
  double best = 1e30, sum = 0.0;
  int timed = 0;
  for (int i = 0; i < iters; ++i) {
    if (i == 0) { ::setenv("VPIPE_KREA2_DIT_PROFILE", "1", 1); }
    else { ::unsetenv("VPIPE_KREA2_DIT_PROFILE"); }
    const auto t0 = std::chrono::steady_clock::now();
    SharedBuffer v = m->forward_dit(fused, text_seq, latbuf, img_seq, grid,
                                    grid, 0.5f, -1);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(!v.empty());
    std::printf("[krea2_dit] bench iter %d: %.1f ms/step%s\n", i, ms,
                i == 0 ? " (profiled -- barriers inflate)" : "");
    if (i > 0) { if (ms < best) { best = ms; } sum += ms; ++timed; }
  }
  ::unsetenv("VPIPE_KREA2_DIT_PROFILE");
  if (timed == 0) { best = sum = 0.0; timed = 1; }
  std::printf("[krea2_dit] bench: best %.1f ms/step, mean %.1f ms/step "
              "(barrier-free iters; iter 0 is the profiled section split)\n",
              best, sum / timed);
  EXPECT_TRUE(best >= 0.0);
}

// vec4 adaLN / gate twins: same arithmetic per element, so the block output
// must be BIT-IDENTICAL. The guard is on the plumbing -- the (N, total) ->
// (N/4, total/N) rewrite and the 4-alignment of every bound element offset (an
// unaligned vec4 access is undefined, so a wrong offset gives garbage, not a
// rounding difference), not on the arithmetic.
TEST(krea2_dit, vec4_elementwise_matches_scalar)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  const int HID = cfg.hidden, IC = cfg.in_channels;
  const int text_seq = 128, grid = 8, img_seq = grid * grid, stop = 1;
  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t s = 0x5eed1234u;
  auto fill = [&](std::vector<float>& v) {
    for (auto& e : v) {
      s = s * 1664525u + 1013904223u;
      e = (float)(s >> 9) / 4194304.0f - 1.0f;
    }
  };
  fill(txt); fill(lat);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = (std::size_t)(text_seq + img_seq) * HID;
  auto run = [&](bool v4) -> std::vector<std::uint16_t> {
    if (v4) { ::unsetenv("VPIPE_NO_ELT_V4"); }
    else    { ::setenv("VPIPE_NO_ELT_V4", "1", 1); }
    auto m = MetalKrea2Transformer::load(tdir, mc, cfg, /*stream_blocks=*/true);
    if (!m) { return {}; }
    SharedBuffer o = m->forward_dit(to_f16buf(txt), text_seq, to_f16buf(lat),
                                    img_seq, grid, grid, 0.5f, stop);
    if (o.empty() || o.byte_size() < n * 2) { return {}; }
    std::vector<std::uint16_t> out(n);
    std::memcpy(out.data(), o.contents(), n * 2);
    return out;
  };
  const std::vector<std::uint16_t> a = run(true);
  const std::vector<std::uint16_t> b = run(false);
  ::unsetenv("VPIPE_NO_ELT_V4");
  ASSERT_TRUE(!a.empty() && a.size() == b.size());
  std::size_t diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i) { diff += (a[i] != b[i]) ? 1 : 0; }
  EXPECT_TRUE(diff == 0);
  std::printf("[krea2_dit] vec4 vs scalar elementwise: %zu/%zu words differ\n",
              diff, a.size());
}

// SAGE IN THE MODEL, which the driver tests next door cannot show.
//
// metal-sage-attention.cc verifies the class against a reference
// attention. What it cannot verify is that this family HANDS IT THE
// RIGHT THING: the operands are the head-major transposes, the head
// counts are HED query and KVH key -- Krea-2 is genuinely grouped -- and
// the block index gates on `dense_layers`. Every one of those is a
// number this file passes and that file trusts.
//
// So: the same forward twice, differing only in cfg.sage.enabled, and
// the two required CLOSE but NOT IDENTICAL. Identical would mean the
// function constant never applied and both arms ran dense, which is the
// failure a tolerance-only check reads as a pass.
TEST(krea2_dit, forward_dit_sage_matches_f16)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config base;
  const int HID = base.hidden, IC = base.in_channels;
  const int text_seq = 64, grid = 32;
  const int img_seq = grid * grid;
  const int stop = 2;                       // two blocks, both routed

  std::vector<float> txt((std::size_t)text_seq * HID);
  std::vector<float> lat((std::size_t)img_seq * IC);
  std::uint32_t s = 0x5a6e12c0u;
  auto fill = [&](std::vector<float>& v) {
    for (auto& e : v) {
      s = s * 1664525u + 1013904223u;
      e = ((float)(s >> 9) / 4194304.0f - 1.0f);
    }
  };
  fill(txt);
  fill(lat);
  auto to_f16buf = [&](const std::vector<float>& src) {
    SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
    return b;
  };
  const std::size_t n = (std::size_t)(text_seq + img_seq) * HID;

  auto run = [&](bool sage_on) {
    MetalKrea2Transformer::Config cfg = base;
    cfg.sage.enabled = sage_on;
    cfg.sage.dense_layers = 0;
    auto m = MetalKrea2Transformer::load(tdir, mc, cfg,
                                         /*stream_blocks=*/true);
    std::vector<float> out;
    if (m == nullptr) { return out; }
    SharedBuffer o = m->forward_dit(to_f16buf(txt), text_seq, to_f16buf(lat),
                                    img_seq, grid, grid, 0.5f, stop);
    if (o.empty() || o.byte_size() < n * 2) { return out; }
    out.resize(n);
    const auto* p = static_cast<const _Float16*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)p[i]; }
    return out;
  };

  const std::vector<float> v_sage = run(true);
  const std::vector<float> v_f16 = run(false);
  ASSERT_TRUE(!v_sage.empty());
  ASSERT_TRUE(v_sage.size() == v_f16.size());
  const double r = rel_l2_(v_sage.data(), v_f16.data(), v_sage.size());
  std::size_t differ = 0;
  for (std::size_t i = 0; i < v_sage.size(); ++i) {
    differ += (v_sage[i] != v_f16[i]) ? 1 : 0;
  }
  std::printf("[krea2_dit] forward_dit sage-vs-f16 rel-L2 = %.6g, %zu of "
              "%zu differ (seq=%d)\n", r, differ, v_sage.size(),
              text_seq + img_seq);
  EXPECT_TRUE(std::isfinite(r) && r < 0.10);
  // The int8 path RAN. Without this the test passes on two dense arms.
  EXPECT_TRUE(differ > v_sage.size() / 4);
}

// ---- perf-flops ----------------------------------------------------
//
// Both forwards' arithmetic, as the FLOP/s meter is credited with it
// (common/perf-flops.h, shared/dit-flops.h):
//
//   * the text-fusion tower and a two-block DiT, each EXACT against a
//     count derived here from the structure, and each credited whole
//     once its buffers retire;
//   * with Sol routing every block, the first forward counts the GEMMs
//     at once and credits the attention Sol kept when it finishes, and
//     the second counts its routed attention at that ratio as it
//     encodes -- so the two forwards' attention agree.
//
// Grid 32 with 64 prompt tokens is 1088 joint rows, 17 routing blocks:
// past Sol's floor of 16. Env: VPIPE_KREA2_TEST_MODEL_PATH.
TEST(krea2_dit, forward_credits_its_flops)
{
  namespace pf = vpipe::perf_flops;
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const std::string tdir = std::string(root) + "/transformer";

  MetalKrea2Transformer::Config cfg;
  cfg.n_layers = 2;
  const int TS = 64, grid = 32, IS = grid * grid, seq = TS + IS;
  const long long HID = cfg.hidden, IC = cfg.in_channels;
  const long long TH = cfg.text_hidden, NL = cfg.n_text_layers;
  const long long TFF = cfg.text_ffn, THD = cfg.text_head_dim();
  const long long HD = cfg.head_dim, FF = cfg.ffn;
  const long long qd = (long long)cfg.n_heads * HD;
  const long long kd = (long long)cfg.n_kv_heads * HD;

  std::uint32_t st = 0x7f10b5e1u;
  auto f16buf = [&](std::size_t n, float amp) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    if (b.empty()) { return b; }
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      st = st * 1664525u + 1013904223u;
      d[i] = (_Float16)(((float)(st >> 9) / 4194304.0f - 1.0f) * amp);
    }
    return b;
  };
  const SharedBuffer ehs = f16buf((std::size_t)TS * NL * TH, 1.0f);
  const SharedBuffer txt = f16buf((std::size_t)TS * HID, 1.0f);
  const SharedBuffer lat = f16buf((std::size_t)IS * IC, 1.0f);
  ASSERT_TRUE(!ehs.empty() && !txt.empty() && !lat.empty());
  if (ehs.empty() || txt.empty() || lat.empty()) { return; }

  // The counts, from the structure. A text-fusion block runs `batch`
  // independent length-T sequences: the layerwise ones over the 12
  // tapped layers of each token, the refiners over the prompt.
  auto text_block = [&](long long batch, long long T) {
    const long long BT = batch * T;
    return 5 * gemm_flops(BT, TH, TH) + 2 * gemm_flops(BT, TFF, TH) +
           gemm_flops(BT, TH, TFF) +
           attention_flops(batch * cfg.text_heads, T * T, THD);
  };
  const std::uint64_t want_text =
      (std::uint64_t)cfg.n_layerwise * text_block(TS, NL) +
      gemm_flops(TS * TH, 1, NL) +
      (std::uint64_t)cfg.n_refiner * text_block(1, TS) +
      gemm_flops(TS, HID, TH) + gemm_flops(TS, HID, HID);
  const std::uint64_t attn =
      attention_flops(cfg.n_heads, (long long)seq * seq, HD);
  const std::uint64_t gemms_per_block =
      gemm_flops(seq, qd, HID) + 2 * gemm_flops(seq, kd, HID) +
      2 * gemm_flops(seq, HID, HID) + gemm_flops(seq, 2 * FF, HID) +
      gemm_flops(seq, HID, FF);
  const std::uint64_t gemms =
      gemm_flops(1, HID, cfg.timestep_dim) + gemm_flops(1, HID, HID) +
      gemm_flops(1, 6 * HID, HID) + gemm_flops(IS, HID, IC) +
      (std::uint64_t)cfg.n_layers * gemms_per_block +
      gemm_flops(IS, IC, HID);
  const std::uint64_t want_dit = gemms + (std::uint64_t)cfg.n_layers * attn;

  // One call; what the tally flushed and the meter was credited.
  struct Got { std::uint64_t flushed = 0, gpu = 0; bool ok = false; };
  auto measure = [&](MetalKrea2Transformer& m, auto&& call) {
    Got g;
    const std::uint64_t f0 = m.gpu_flops().flushed();
    const pf::Totals g0 = pf::totals(pf::kGpu);
    g.ok = call();
    g.flushed = m.gpu_flops().flushed() - f0;
    // The last buffer's handler may run after the wait that saw it --
    // and Sol's finish() credits synchronously, so reaching `flushed` is
    // not proof that every handler has run. Let them settle.
    for (int i = 0; i < 400 && pf::totals(pf::kGpu).flops - g0.flops <
                                   g.flushed; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    g.gpu = pf::totals(pf::kGpu).flops - g0.flops;
    return g;
  };
  auto dit = [&](MetalKrea2Transformer& m) {
    return !m.forward_dit(txt, TS, lat, IS, grid, grid, 0.5f).empty();
  };

  {
    auto m = MetalKrea2Transformer::load(tdir, mc, cfg, /*stream=*/true);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    const Got t = measure(*m, [&]() {
      return !m->forward_text(ehs, TS).empty();
    });
    const Got d = measure(*m, [&]() { return dit(*m); });
    std::printf("[krea2_dit] flops: text tower %.4f TFLOP (derived %.4f, "
                "credited %.4f); DiT %lld rows x %d blocks %.4f TFLOP "
                "(derived %.4f, credited %.4f)\n", t.flushed / 1e12,
                want_text / 1e12, t.gpu / 1e12, (long long)seq,
                cfg.n_layers, d.flushed / 1e12, want_dit / 1e12,
                d.gpu / 1e12);
    ASSERT_TRUE(t.ok && d.ok);
    EXPECT_TRUE(t.flushed == want_text);
    EXPECT_TRUE(t.gpu == t.flushed);
    EXPECT_TRUE(d.flushed == want_dit);
    EXPECT_TRUE(d.gpu == d.flushed);
  }

  // Sol on every block.
  MetalKrea2Transformer::Config sc = cfg;
  sc.sol.enabled      = true;
  sc.sol.tau          = 1.0f;
  sc.sol.dense_layers = 0;
  sc.sol.key_block    = 64;
  sc.sol.local_radius = 1;
  auto m = MetalKrea2Transformer::load(tdir, mc, sc, /*stream=*/true);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const Got a = measure(*m, [&]() { return dit(*m); });
  ASSERT_TRUE(a.ok);
  if (!a.ok) { return; }
  // A dense count means nothing routed: no Sol kernels on this box, or
  // a geometry it declined.
  if (a.flushed == want_dit) {
    std::printf("[krea2_dit] flops: Sol did not route here; its half is "
                "skipped\n");
    return;
  }
  const Got b = measure(*m, [&]() { return dit(*m); });
  ASSERT_TRUE(b.ok);
  // The first forward: GEMMs as encoded, the kept attention at finish.
  const std::uint64_t kept1 = a.gpu - a.flushed;
  // The second: the routed attention at that ratio, as encoded -- one
  // floor per routed call, so within a FLOP a block of the first.
  const std::uint64_t kept2 = b.flushed - gemms;
  const std::uint64_t all = (std::uint64_t)cfg.n_layers * attn;
  std::printf("[krea2_dit] flops with Sol: forward 1 counted %.4f + kept "
              "%.4f at finish, forward 2 counted %.4f (kept %.4f); dense "
              "attention %.4f TFLOP\n", a.flushed / 1e12, kept1 / 1e12,
              b.flushed / 1e12, kept2 / 1e12, all / 1e12);
  EXPECT_TRUE(a.flushed == gemms);
  EXPECT_TRUE(a.gpu >= a.flushed);
  EXPECT_TRUE(kept1 > 0 && kept1 <= all);
  EXPECT_TRUE(b.gpu == b.flushed);
  const std::uint64_t gap = kept1 > kept2 ? kept1 - kept2 : kept2 - kept1;
  EXPECT_TRUE(gap <= (std::uint64_t)cfg.n_layers);
}
