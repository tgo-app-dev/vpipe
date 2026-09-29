// AneModule: stamping real weights into a compiled ANE graph.
//
// The graph is produced offline (tools/make_ane_module.py, which also
// records ANE residency in a .census sidecar) with placeholder weights;
// AneModule::build copies it, stamps the caller's bytes into the weight
// blob, and loads it. This checks the whole chain: that the blob parses
// as the documented index, that stamped bytes are what the module
// actually computes with, and that the result comes back through the
// zero-copy Metal path.
//
// Gated on a template produced by that tool:
//
//     VPIPE_ANE_TEMPLATE=/path/to/tmpl_4096x2048x2048.mlmodelc
//
// Unset -> passes trivially, so default suites stay green.

#include "minitest.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include <mach/mach.h>
#include <CoreVideo/CoreVideo.h>
#endif

#include "generative-models/shared/ane-module.h"
#include "generative-models/shared/ane-ffn.h"
#include "apple-silicon/coreml/ane-emitter.h"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "interfaces/session-services-intf.h"
#include "common/session.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#endif

#include <chrono>
#include <filesystem>
#include <functional>
#include <fstream>
#include <cstring>
#include <iterator>
#include <sstream>
#include <memory>
#include <algorithm>
#include <array>
#include <set>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

const char*
env_or_null_(const char* name)
{
  const char* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? v : nullptr;
}

}  // namespace

TEST(ane_module, stamped_weights_are_what_it_computes)
{
  const char* tmpl = env_or_null_("VPIPE_ANE_TEMPLATE");
  if (tmpl == nullptr) { return; }

#ifndef VPIPE_BUILD_APPLE_SILICON
  (void)tmpl;
#else
  using namespace vpipe;
  using namespace vpipe::genai;
  using namespace vpipe::metal_compute;

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  // The blob index must parse before anything else is worth trying --
  // this is the check that catches the undocumented layout moving under
  // a macOS update.
  std::vector<AneWeightSlot> slots;
  std::string                err;
  const std::string blob = std::string(tmpl) + "/weights/weight.bin";
  const bool parsed = parse_weight_blob(blob, &slots, &err);
  ASSERT_TRUE(parsed);
  if (!parsed) {
    std::printf("[ane_module] blob did not parse: %s\n", err.c_str());
    return;
  }
  ASSERT_TRUE(slots.size() == 1u);
  if (slots.size() != 1u) { return; }
  std::printf("[ane_module] template has %zu weight slot(s), first is "
              "%zu bytes at offset %zu\n",
              slots.size(), slots[0].bytes, slots[0].offset);

  // Derive M, K, N: in = M*K, out = M*N, slot = K*N*2, so
  // M^2 = in*out / (slot/2).
  const std::size_t wk_elems = slots[0].bytes / 2;

  // A deterministic weight whose product is easy to check, and whose
  // magnitudes stay inside fp16 once summed over K.
  // A proper RNG, not an LCG: W is indexed W[k*N + n], so along the
  // reduction axis k the stride is N. A cheap LCG strided by N can
  // collapse to a short cycle, leaving W near-constant in k -- the sums
  // then cancel, and a cancelling sum inflates RELATIVE error without
  // anything being wrong. That cost a confusing 5e-2 before.
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<_Float16> W(wk_elems);
  for (std::size_t i = 0; i < wk_elems; ++i) {
    W[i] = (_Float16)(dist(rng) * 0.05);
  }
  AneWeight w{W.data(), slots[0].bytes};
  const AneWeight ws[1] = {w};

  const std::string work = std::string(tmpl) + ".work";
  auto t_build0 = std::chrono::steady_clock::now();
  auto mod = AneModule::build(&sess, tmpl, work, ws);
  const double build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_build0).count();
  ASSERT_TRUE(mod != nullptr);
  if (mod == nullptr) { return; }
  ASSERT_TRUE(mod->valid());
  if (!mod->valid()) { return; }

  const std::size_t in_elems  = mod->input_elems();
  const std::size_t out_elems = mod->output_elems();
  const double m2 = (double)in_elems * (double)out_elems / (double)wk_elems;
  const std::size_t M = (std::size_t)std::llround(std::sqrt(m2));
  ASSERT_TRUE(M > 0 && in_elems % M == 0 && out_elems % M == 0);
  if (M == 0 || in_elems % M != 0 || out_elems % M != 0) { return; }
  const std::size_t K = in_elems / M;
  const std::size_t N = out_elems / M;
  ASSERT_TRUE(K * N == wk_elems);
  if (K * N != wk_elems) { return; }

  std::printf("[ane_module] M=%zu K=%zu N=%zu | weights %.1f MB | "
              "build (copy+stamp+ANE compile) %.0f ms = %.1f ms/MB | "
              "ane_resident=%d\n",
              M, K, N, slots[0].bytes / 1e6, build_ms,
              build_ms / (slots[0].bytes / 1e6),
              mod->ane_resident() ? 1 : 0);
  EXPECT_TRUE(mod->ane_resident());

  SharedBuffer x = mc->make_shared_buffer(in_elems * 2);
  SharedBuffer y = mc->make_shared_buffer(out_elems * 2);
  ASSERT_TRUE(!x.empty() && !y.empty());
  if (x.empty() || y.empty()) { return; }

  auto* xp = static_cast<_Float16*>(x.contents());
  for (std::size_t i = 0; i < in_elems; ++i) {
    xp[i] = (_Float16)(dist(rng) * 0.2);
  }
  std::memset(y.contents(), 0, out_elems * 2);

  ASSERT_TRUE(mod->run(x, y));

  // Spot-check a few rows against a CPU reference. Checking every row
  // would be 34 GFLOP on the host; a handful is enough to catch a
  // mis-stamped or permuted weight, which is the failure this guards.
  const auto* yp = static_cast<const _Float16*>(y.contents());
  double num = 0.0, den = 0.0;
  const std::size_t kRows = 4;
  for (std::size_t r = 0; r < kRows && r < M; ++r) {
    for (std::size_t n = 0; n < N; ++n) {
      double acc = 0.0;
      for (std::size_t k = 0; k < K; ++k) {
        acc += (double)xp[r * K + k] * (double)W[k * N + n];
      }
      const double got = (double)yp[r * N + n];
      num += (got - acc) * (got - acc);
      den += acc * acc;
    }
  }
  const double rel = (den > 0.0) ? std::sqrt(num / den) : 1.0;
  // Report the reference RMS too: a tiny one means the sums cancel and
  // the relative figure is inflated by the DATA, not by the module.
  const double ref_rms = std::sqrt(den / (double)(kRows * N));
  std::printf("[ane_module] rel-L2 vs f64 reference over %zu rows: "
              "%.3e  (reference RMS %.4f)\n", kRows, rel, ref_rms);
  // fp16 accumulation over K=2048; ~1e-2 is the honest bar here, and a
  // wrong stamp lands at O(1), not near the bar.
  EXPECT_TRUE(rel < 3e-2);

  const int kIters = 8;
  for (int i = 0; i < 3; ++i) { mod->run(x, y); }
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) { mod->run(x, y); }
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count() / kIters;
  std::printf("[ane_module] %.3f ms/dispatch | %.2f TOPS delivered "
              "(includes the ~1-2 ms per-predict floor)\n",
              ms, 2.0 * M * K * N / (ms * 1e-3) / 1e12);
  EXPECT_TRUE(ms > 0.0);
#endif
}

// The whole point of build_tiled(): a tiled template carries one blob
// entry PER TILE, and every ordering of those tiles produces correctly
// SHAPED slabs. So nothing but numerics can prove the gather agrees
// with the order tools/make_ane_module.py emitted -- a transposed or
// rotated order loads fine and computes nonsense.
//
//     VPIPE_ANE_FFN_TEMPLATE=/path/to/ffn_2048x4096.mlmodelc
//     VPIPE_ANE_FFN_DF=2048,4096        (D,F the template was built at)
TEST(ane_module, tiled_ffn_matches_reference)
{
  const char* tmpl = env_or_null_("VPIPE_ANE_FFN_TEMPLATE");
  const char* df   = env_or_null_("VPIPE_ANE_FFN_DF");
  if (tmpl == nullptr || df == nullptr) { return; }
#ifdef VPIPE_BUILD_APPLE_SILICON
  using namespace vpipe;
  using namespace vpipe::genai;
  using namespace vpipe::metal_compute;

  std::size_t D = 0, F = 0;
  if (std::sscanf(df, "%zu,%zu", &D, &F) != 2 || D == 0 || F == 0) {
    std::printf("[ane_module] VPIPE_ANE_FFN_DF must be \"D,F\"\n");
    return;
  }
  const std::size_t Bk = 2048, Bn = 2048;   // the tool's defaults

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  std::mt19937 rng(777);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<_Float16> W1(D * F), W2(F * D);
  for (auto& v : W1) { v = (_Float16)(dist(rng) * 0.04); }
  for (auto& v : W2) { v = (_Float16)(dist(rng) * 0.02); }

  const AneTiledWeight tw[2] = {
      {W1.data(), D, F, Bk, std::min(Bn, F)},
      {W2.data(), F, D, Bk, std::min(Bn, D)}};
  auto mod = AneModule::build_tiled(&sess, tmpl,
                                    std::string(tmpl) + ".work", tw);
  ASSERT_TRUE(mod != nullptr);
  if (mod == nullptr) { return; }

  const std::size_t M = mod->input_elems() / D;
  ASSERT_TRUE(M > 0 && mod->output_elems() == M * D);
  if (M == 0 || mod->output_elems() != M * D) { return; }

  SharedBuffer x = mc->make_shared_buffer(mod->input_elems() * 2);
  SharedBuffer y = mc->make_shared_buffer(mod->output_elems() * 2);
  ASSERT_TRUE(!x.empty() && !y.empty());
  if (x.empty() || y.empty()) { return; }
  auto* xp = static_cast<_Float16*>(x.contents());
  for (std::size_t i = 0; i < mod->input_elems(); ++i) {
    xp[i] = (_Float16)(dist(rng) * 0.3);
  }
  std::memset(y.contents(), 0, mod->output_elems() * 2);
  ASSERT_TRUE(mod->run(x, y));

  // Reference on two rows: gelu is mb.gelu's default EXACT mode,
  // 0.5*v*(1+erf(v/sqrt2)) -- a tanh approximation here would show up
  // as a small bias and muddy the ordering signal this test carries.
  const auto* yp = static_cast<const _Float16*>(y.contents());
  double num = 0.0, den = 0.0;
  const std::size_t kRows = 2;
  std::vector<double> h(F);
  for (std::size_t r = 0; r < kRows && r < M; ++r) {
    for (std::size_t f = 0; f < F; ++f) {
      double acc = 0.0;
      for (std::size_t d = 0; d < D; ++d) {
        acc += (double)xp[r * D + d] * (double)W1[d * F + f];
      }
      h[f] = 0.5 * acc * (1.0 + std::erf(acc / std::sqrt(2.0)));
    }
    for (std::size_t d = 0; d < D; ++d) {
      double acc = 0.0;
      for (std::size_t f = 0; f < F; ++f) {
        acc += h[f] * (double)W2[f * D + d];
      }
      const double got = (double)yp[r * D + d];
      num += (got - acc) * (got - acc);
      den += acc * acc;
    }
  }
  const double rel = (den > 0.0) ? std::sqrt(num / den) : 1.0;
  std::printf("[ane_module] tiled FFN D=%zu F=%zu M=%zu: rel-L2 %.3e "
              "(reference RMS %.4f)\n", D, F, M, rel,
              std::sqrt(den / (double)(kRows * D)));
  // A COARSE check: it catches a module computing the wrong thing at
  // all (that lands at O(1)), not small numeric drift. fp16
  // accumulation through TWO chained matmuls plus a nonlinearity is
  // genuinely expensive -- 1.6e-2 already at M=512/D=512/F=1024, and
  // it grows with depth. MEASURED that the GELU mode is NOT the cause:
  // an exact-erf reference and a tanh-approximation one agree to
  // within 1.6e-2 of the model and of each other, so there is no
  // mode to get wrong here. Precision is carried by
  // tilings_agree_on_the_same_weights, where every modelling
  // assumption cancels between the two arms.
  EXPECT_TRUE(rel < 2e-1);

  const int kIters = 6;
  for (int i = 0; i < 2; ++i) { mod->run(x, y); }
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) { mod->run(x, y); }
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count() / kIters;
  std::printf("[ane_module] %.2f ms/FFN | %.2f TOPS delivered\n", ms,
              2.0 * M * D * F * 2 / (ms * 1e-3) / 1e12);
#endif
}

// The assumption-free ordering check. Two templates of the SAME FFN
// built at DIFFERENT tilings are stamped from the SAME logical W1/W2,
// so build_tiled() has to produce two different slab sets from one
// matrix. If both gathers are right the outputs agree to fp16
// reassociation; if the order is wrong, one of them computes a
// permuted weight and they diverge. Nothing here depends on knowing
// CoreML's GELU mode or its accumulation precision -- every modelling
// assumption cancels between the two arms, which a hand-written
// reference cannot do.
//
//     VPIPE_ANE_FFN_TEMPLATE  = tiling A
//     VPIPE_ANE_FFN_TEMPLATE2 = tiling B, same D/F/M
//     VPIPE_ANE_FFN_DF        = D,F
//     VPIPE_ANE_FFN_TILE2     = Bk,Bn of tiling B (A uses 2048,2048)
TEST(ane_module, tilings_agree_on_the_same_weights)
{
  const char* ta = env_or_null_("VPIPE_ANE_FFN_TEMPLATE");
  const char* tb = env_or_null_("VPIPE_ANE_FFN_TEMPLATE2");
  const char* df = env_or_null_("VPIPE_ANE_FFN_DF");
  const char* t2 = env_or_null_("VPIPE_ANE_FFN_TILE2");
  if (ta == nullptr || tb == nullptr || df == nullptr || t2 == nullptr) {
    return;
  }
#ifdef VPIPE_BUILD_APPLE_SILICON
  using namespace vpipe;
  using namespace vpipe::genai;
  using namespace vpipe::metal_compute;

  std::size_t D = 0, F = 0, Bk2 = 0, Bn2 = 0;
  if (std::sscanf(df, "%zu,%zu", &D, &F) != 2 ||
      std::sscanf(t2, "%zu,%zu", &Bk2, &Bn2) != 2) {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  std::mt19937 rng(4242);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<_Float16> W1(D * F), W2(F * D);
  for (auto& v : W1) { v = (_Float16)(dist(rng) * 0.04); }
  for (auto& v : W2) { v = (_Float16)(dist(rng) * 0.02); }

  const AneTiledWeight twa[2] = {{W1.data(), D, F, 2048, std::min<std::size_t>(2048, F)},
                                 {W2.data(), F, D, 2048, std::min<std::size_t>(2048, D)}};
  const AneTiledWeight twb[2] = {{W1.data(), D, F, Bk2, std::min(Bn2, F)},
                                 {W2.data(), F, D, Bk2, std::min(Bn2, D)}};
  auto ma = AneModule::build_tiled(&sess, ta, std::string(ta) + ".xa", twa);
  auto mb = AneModule::build_tiled(&sess, tb, std::string(tb) + ".xb", twb);
  ASSERT_TRUE(ma != nullptr && mb != nullptr);
  if (ma == nullptr || mb == nullptr) { return; }
  ASSERT_TRUE(ma->input_elems() == mb->input_elems() &&
              ma->output_elems() == mb->output_elems());
  if (ma->input_elems() != mb->input_elems()) { return; }

  SharedBuffer x  = mc->make_shared_buffer(ma->input_elems() * 2);
  SharedBuffer ya = mc->make_shared_buffer(ma->output_elems() * 2);
  SharedBuffer yb = mc->make_shared_buffer(mb->output_elems() * 2);
  ASSERT_TRUE(!x.empty() && !ya.empty() && !yb.empty());
  if (x.empty() || ya.empty() || yb.empty()) { return; }
  auto* xp = static_cast<_Float16*>(x.contents());
  for (std::size_t i = 0; i < ma->input_elems(); ++i) {
    xp[i] = (_Float16)(dist(rng) * 0.3);
  }
  ASSERT_TRUE(ma->run(x, ya));
  ASSERT_TRUE(mb->run(x, yb));

  const auto* pa = static_cast<const _Float16*>(ya.contents());
  const auto* pb = static_cast<const _Float16*>(yb.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < ma->output_elems(); ++i) {
    const double a = (double)pa[i], b = (double)pb[i];
    num += (a - b) * (a - b);
    den += b * b;
  }
  const double rel = (den > 0.0) ? std::sqrt(num / den) : 1.0;
  std::printf("[ane_module] tiling A vs B on identical weights: "
              "rel-L2 %.3e\n", rel);
  // Only K-split reassociation separates them, so this is tight. A
  // wrong gather order lands at O(1).
  EXPECT_TRUE(rel < 1e-2);
#endif
}

TEST(ane_module, wrong_weight_count_is_refused)
{
  const char* tmpl = env_or_null_("VPIPE_ANE_TEMPLATE");
  if (tmpl == nullptr) { return; }
#ifdef VPIPE_BUILD_APPLE_SILICON
  using namespace vpipe;
  using namespace vpipe::genai;
  Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  // Two weights into a one-weight template must fail cleanly rather
  // than stamp a prefix and load something that quietly computes the
  // wrong thing.
  std::vector<_Float16> junk(16);
  const AneWeight ws[2] = {{junk.data(), junk.size() * 2},
                           {junk.data(), junk.size() * 2}};
  auto mod = AneModule::build(&sess, tmpl,
                              std::string(tmpl) + ".badwork", ws);
  EXPECT_TRUE(mod == nullptr);
#endif
}

#ifdef VPIPE_BUILD_APPLE_SILICON
// WHAT DOES THE BAKE ACTUALLY COST, and is any of it avoidable?
//
// The tier's memory is per BLOCK and held for the run, which puts it out
// of reach of a 16 GB box -- the box with the most to gain, since a base
// M4 carries the same 16-core ANE against roughly half the GPU. The way
// out would be to bake a block JUST IN TIME and free it again, which is
// only possible if the compile is cheap enough to hide under the GPU
// work of the same block.
//
// So: split the build into its phases and time each, at the real Krea-2
// feed-forward shape and at a small one, and build the SAME graph twice
// with DIFFERENT weights -- which separates a cost that belongs to the
// graph (paid once per shape) from one that belongs to the bytes (paid
// per block, every time).
//
// VPIPE_ANE_COMPILE_BENCH=1 to run; it is tens of seconds.
TEST(ane_module, compile_cost_breakdown)
{
  if (env_or_null_("VPIPE_ANE_COMPILE_BENCH") == nullptr) { return; }
  vpipe::Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  if (!vpipe::genai::AneModule::emitter_usable(&sess)) { return; }

  struct Shape { const char* name; int M, K, N, bk, bn; };
  const Shape shapes[] = {
      {"small",  512,  1024,  2048, 1024, 1024},
      {"krea2", 2048,  6144, 16384, 2048, 2048},
  };
  for (const Shape& s : shapes) {
    const std::size_t K = (std::size_t)s.K, N = (std::size_t)s.N;
    // gate [K,N], up [K,N], down [N,K] -- the SwiGLU feed-forward.
    std::vector<std::uint16_t> g(K * N), u(K * N), d(N * K);
    auto fill = [&](std::vector<std::uint16_t>& v, std::uint16_t bias,
                    std::uint32_t seed) {
      std::mt19937 rng(seed);
      for (auto& e : v) { e = (std::uint16_t)((rng() & 0x03ffu) | bias); }
    };
    const std::size_t bytes = (g.size() + u.size() + d.size()) * 2;

    vpipe::AneGraphSpec spec;
    spec.kind = vpipe::AneGraphSpec::Kind::SwiGluFfn;
    spec.M = s.M; spec.K = s.K; spec.N = s.N;
    spec.bk = s.bk; spec.bn = s.bn;

    auto build_once = [&](std::uint16_t bias, std::uint32_t seed, int tag) {
      fill(g, bias, seed); fill(u, bias, seed + 1); fill(d, bias, seed + 2);
      const vpipe::genai::AneTiledWeight tw[3] = {
          {g.data(), K, N, (std::size_t)s.bk, (std::size_t)s.bn},
          {u.data(), K, N, (std::size_t)s.bk, (std::size_t)s.bn},
          {d.data(), N, K, (std::size_t)s.bk, (std::size_t)s.bn}};
      char work[320];
      std::snprintf(work, sizeof(work), "%s/vpipe-ane-bench-%s-%d.mlmodelc",
                    std::getenv("TMPDIR") != nullptr ? std::getenv("TMPDIR")
                                                     : "/tmp",
                    s.name, tag);
      const auto t0 = std::chrono::steady_clock::now();
      auto m = vpipe::genai::AneModule::build_emitted(&sess, spec, tw, work);
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      return std::make_pair(std::move(m), ms);
    };

    auto a = build_once(0x3800, 11u, 0);
    auto b = build_once(0x3400, 22u, 1);      // same GRAPH, different BYTES
    // ...and the SAME BYTES as #1. The compiled ANE program is cached on
    // disk under ~/Library/Caches/com.apple.e5rt.e5bundlecache/<macOS
    // build>/<content hash>, so this is the question of whether a
    // compile can be done ONCE rather than on every load -- the
    // difference between a 9.5 min first run and a 9.5 min every run.
    auto c = build_once(0x3800, 11u, 2);
    EXPECT_TRUE(a.first != nullptr && b.first != nullptr &&
                c.first != nullptr);
    if (a.first == nullptr || b.first == nullptr || c.first == nullptr) {
      continue;
    }

    // And what one call costs once it is built, so the compile can be
    // read against the work it would have to hide behind.
    vpipe::metal_compute::SharedBuffer x =
        sess.metal_compute()->make_shared_buffer(
            a.first->input_elems() * 2);
    vpipe::metal_compute::SharedBuffer y =
        sess.metal_compute()->make_shared_buffer(
            a.first->output_elems() * 2);
    auto* xp = static_cast<std::uint16_t*>(x.contents());
    for (std::size_t i = 0; i < a.first->input_elems(); ++i) {
      xp[i] = 0x3400;
    }
    const auto r0 = std::chrono::steady_clock::now();
    a.first->run(x, y);
    const double first_run = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - r0).count();
    double best = 1e18;
    for (int i = 0; i < 5; ++i) {
      const auto t = std::chrono::steady_clock::now();
      a.first->run(x, y);
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t).count();
      if (ms < best) { best = ms; }
    }
    std::printf("[ane_module] %s %dx%dx%d weights %zu MB | #1 new %.0f ms "
                "(%.1f ms/MB) | #2 same graph NEW bytes %.0f ms | #3 same "
                "graph SAME bytes %.0f ms | first predict %.1f ms | steady "
                "%.1f ms | ane=%d\n",
                s.name, s.M, s.K, s.N, bytes >> 20, a.second,
                a.second / (double)(bytes >> 20), b.second, c.second,
                first_run, best, a.first->ane_resident() ? 1 : 0);
  }
}
#endif

#ifdef VPIPE_BUILD_APPLE_SILICON
// CAN THE COMPILE BE DONE ONCE?
//
// An .mlmodelc is already "compiled" in CoreML's sense, but the ANE
// PROGRAM inside it is built by ANECompilerService at load, and that is
// where the ~40 ms/MB goes. macOS caches those programs under
// ~/Library/Caches/com.apple.e5rt.e5bundlecache/<macOS build>/<hash>,
// which raises the obvious question: is the cost paid once per file, or
// once per LOAD?
//
// The difference decides whether the tier can be built offline (or on
// first run) and merely LOADED thereafter. So: emit one module, keep
// the directory, and load it three times -- dropping the shared_ptr
// between loads, because the manager's cache holds weak_ptr and would
// otherwise answer from memory and prove nothing.
//
// VPIPE_ANE_COMPILE_BENCH=1 to run. VPIPE_ANE_KEEP_DIR=1 leaves the
// emitted directory behind so a SECOND process can be pointed at it --
// the strongest form of the question, since nothing is warm.
TEST(ane_module, precompiled_file_reload_cost)
{
  if (env_or_null_("VPIPE_ANE_COMPILE_BENCH") == nullptr) { return; }
  vpipe::Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return; }
  if (!vpipe::genai::AneModule::emitter_usable(&sess)) { return; }

  vpipe::AneGraphSpec spec;
  spec.kind = vpipe::AneGraphSpec::Kind::SwiGluFfn;
  spec.M = 2048; spec.K = 6144; spec.N = 16384;
  spec.bk = 2048; spec.bn = 2048;

  const char* tmp = std::getenv("TMPDIR");
  const std::string dir =
      std::string(tmp != nullptr ? tmp : "/tmp") + "/vpipe-ane-reload.mlmodelc";

  double emit_ms = 0.0;
  if (!std::filesystem::exists(dir)) {
    const std::vector<std::size_t> sizes =
        vpipe::AneEmitter::weight_slabs(spec);
    std::vector<std::vector<std::uint16_t>> store;
    store.reserve(sizes.size());
    std::vector<vpipe::AneBlobSlab> slabs;
    slabs.reserve(sizes.size());
    std::size_t total = 0;
    // VPIPE_ANE_SEED varies the BYTES, which varies the content hash the
    // compiled-program cache is keyed on -- the only way to force a COLD
    // compile once a shape has been seen.
    const char* sd = env_or_null_("VPIPE_ANE_SEED");
    std::mt19937 rng(sd != nullptr ? (std::uint32_t)std::atoi(sd)
                                   : 0x5eed1234u);
    for (std::size_t n : sizes) {
      // RANDOM, not a constant fill: a graph whose weights are all one
      // value is something a compiler may fold, and the number here has
      // to be the one a real block pays.
      store.emplace_back(n / 2);
      for (auto& e : store.back()) {
        e = (std::uint16_t)((rng() & 0x03ffu) | 0x3400u);
      }
      slabs.push_back(vpipe::AneBlobSlab{store.back().data(), n});
      total += n;
    }
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = vpipe::AneEmitter::emit(spec, slabs, dir, &err);
    emit_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(ok);
    if (!ok) { return; }
    std::printf("[ane_module] emitted %zu MB to disk in %.0f ms\n",
                total >> 20, emit_ms);
  } else {
    std::printf("[ane_module] reusing the module left by a previous run\n");
  }

  for (int i = 0; i < 3; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    auto m = mgr->load(dir, 3);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[ane_module] load #%d of the SAME .mlmodelc: %.0f ms%s\n",
                i + 1, ms, m != nullptr ? "" : " (FAILED)");
    EXPECT_TRUE(m != nullptr);
    if (m == nullptr) { break; }
    // A FAST LOAD PROVES NOTHING BY ITSELF: a module that quietly fell
    // off the ANE onto BNNS also loads fast, and then computes the
    // right answer slowly. 825 GFLOP is ~88 ms on the ANE and seconds
    // on the CPU, so the predict time is the residency check.
    std::vector<std::uint16_t> xin((std::size_t)spec.M * spec.K, 0x3400u);
    std::vector<std::uint16_t> yout((std::size_t)spec.M * spec.K, 0u);
    vpipe::CoreMLPredictInput in;
    in.data  = xin.data();
    in.dtype = vpipe::CoreMLDType::F16;
    in.shape = {(std::int64_t)spec.M, (std::int64_t)spec.K};
    vpipe::CoreMLPredictOutput out;
    out.want          = vpipe::CoreMLDType::F16;
    out.backing       = yout.data();
    out.backing_elems = yout.size();
    m->predict({&in, 1}, {&out, 1});             // warm
    double best = 1e18;
    for (int k = 0; k < 3; ++k) {
      const auto p0 = std::chrono::steady_clock::now();
      const bool ok = m->predict({&in, 1}, {&out, 1});
      const double pms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - p0).count();
      if (ok && pms < best) { best = pms; }
    }
    std::printf("[ane_module]   ...and predicts in %.1f ms\n", best);
    // Dropped here, so the next load is a real one: the manager's cache
    // is weak, and a live reference would answer from memory.
  }
  // THE A/B THAT MATTERS. build_emitted() deletes the .mlmodelc as soon
  // as the load returns ("CoreML has the weights by now"), and the
  // question is what that costs: a module loaded from a file on disk
  // can have its weights MAPPED, and mapped pages are clean and
  // evictable. Hold the module either way and let a sampler outside
  // read the resident set.
  {
    auto m = mgr->load(dir, 3);
    EXPECT_TRUE(m != nullptr);
    const bool del = env_or_null_("VPIPE_ANE_DELETE_AFTER_LOAD") != nullptr;
    if (del) {
      std::error_code ec;
      std::filesystem::remove_all(dir, ec);
    }
    std::vector<std::uint16_t> xin((std::size_t)spec.M * spec.K, 0x3400u);
    std::vector<std::uint16_t> yout((std::size_t)spec.M * spec.K, 0u);
    vpipe::CoreMLPredictInput in;
    in.data  = xin.data();
    in.dtype = vpipe::CoreMLDType::F16;
    in.shape = {(std::int64_t)spec.M, (std::int64_t)spec.K};
    vpipe::CoreMLPredictOutput out;
    out.want          = vpipe::CoreMLDType::F16;
    out.backing       = yout.data();
    out.backing_elems = yout.size();
    if (m != nullptr) {
      for (int k = 0; k < 40; ++k) { m->predict({&in, 1}, {&out, 1}); }
    }
    std::printf("[ane_module] held a loaded module for 40 predicts "
                "(.mlmodelc %s)\n", del ? "DELETED after load" : "kept");
  }
  if (env_or_null_("VPIPE_ANE_KEEP_DIR") == nullptr) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
}
#endif

#ifdef VPIPE_BUILD_APPLE_SILICON
// WHERE DO A CACHED PROGRAM'S WEIGHTS COME FROM?
//
// The compiled-program cache is ~1 MB a module against 576 MB of weights,
// so a warm load must be getting the bytes from somewhere else -- most
// plausibly the .mlmodelc's own weight.bin, mapped. If so, a program
// compiled for ONE block could be re-pointed at another block's bytes
// with no compile at all. If not, the cache key includes the bytes and
// every block pays its own compile.
//
// Compile A and B (same graph, different weights), then overwrite A's
// weight.bin with B's bytes in place and reload A. Three arms separate
// what the cache key can be sensitive to: the content with the mtime
// restored, the content with a fresh mtime, and the mtime alone.
// Numerics say which weights the reloaded module actually used.
//
// VPIPE_ANE_COMPILE_BENCH=1 to run.
TEST(ane_module, where_cached_weights_come_from)
{
  if (env_or_null_("VPIPE_ANE_COMPILE_BENCH") == nullptr) { return; }
  vpipe::Session sess;
  if (sess.metal_compute() == nullptr) { return; }
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return; }
  if (!vpipe::genai::AneModule::emitter_usable(&sess)) { return; }
  namespace fs = std::filesystem;
  using clk = std::chrono::steady_clock;

  vpipe::AneGraphSpec spec;
  spec.kind = vpipe::AneGraphSpec::Kind::SwiGluFfn;
  spec.M = 512; spec.K = 1024; spec.N = 2048;
  spec.bk = 1024; spec.bn = 1024;
  // VPIPE_ANE_SWAP_SHAPE=krea2: the real 576 MB module. The small shape
  // is 12 MB, which a loader could hash in the time a warm load takes;
  // 576 MB it could not, so the answer may differ.
  if (const char* sh = env_or_null_("VPIPE_ANE_SWAP_SHAPE")) {
    if (std::string(sh) == "krea2") {
      spec.M = 2048; spec.K = 6144; spec.N = 16384;
      spec.bk = 2048; spec.bn = 2048;
    }
  }
  const std::size_t in_n = (std::size_t)spec.M * spec.K;

  const char* tmp = std::getenv("TMPDIR");
  const std::string root = std::string(tmp != nullptr ? tmp : "/tmp");
  // A seed per run, so an earlier run's cache entry cannot answer.
  const std::uint32_t run = (std::uint32_t)std::chrono::duration_cast<
      std::chrono::microseconds>(clk::now().time_since_epoch()).count();

  auto emit = [&](const std::string& dir, std::uint32_t seed) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    std::vector<std::size_t> sizes = vpipe::AneEmitter::weight_slabs(spec);
    std::vector<std::vector<std::uint16_t>> store;
    store.reserve(sizes.size());
    std::vector<vpipe::AneBlobSlab> slabs;
    std::mt19937 rng(seed);
    for (std::size_t n : sizes) {
      store.emplace_back(n / 2);
      // ~0.008 magnitude with random sign: keeps SiLU and the products
      // finite in fp16, so the outputs are comparable bit for bit.
      for (auto& e : store.back()) {
        e = (std::uint16_t)((rng() & 0x83ffu) | 0x2000u);
      }
      slabs.push_back(vpipe::AneBlobSlab{store.back().data(), n});
    }
    std::string err;
    return vpipe::AneEmitter::emit(spec, slabs, dir, &err);
  };
  std::vector<std::uint16_t> x(in_n);
  {
    std::mt19937 rng(0x1234u);
    for (auto& e : x) { e = (std::uint16_t)((rng() & 0x83ffu) | 0x3000u); }
  }
  // Load, time it, predict once, drop -- the manager's cache is weak.
  auto load_run = [&](const std::string& dir, double* ms) {
    std::vector<std::uint16_t> y(in_n, 0u);
    const auto t0 = clk::now();
    auto m = mgr->load(dir, 3);
    *ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    if (m == nullptr) { return std::vector<std::uint16_t>{}; }
    vpipe::CoreMLPredictInput in;
    in.data = x.data(); in.dtype = vpipe::CoreMLDType::F16;
    in.shape = {(std::int64_t)spec.M, (std::int64_t)spec.K};
    vpipe::CoreMLPredictOutput out;
    out.want = vpipe::CoreMLDType::F16;
    out.backing = y.data(); out.backing_elems = y.size();
    if (!m->predict({&in, 1}, {&out, 1})) {
      return std::vector<std::uint16_t>{};
    }
    return y;
  };
  auto same = [](const std::vector<std::uint16_t>& a,
                 const std::vector<std::uint16_t>& b) {
    return !a.empty() && a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
  };
  auto overwrite_in_place = [](const std::string& dst,
                               const std::string& src) {
    std::ifstream i(src, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(i)),
                      std::istreambuf_iterator<char>());
    std::fstream o(dst, std::ios::binary | std::ios::in | std::ios::out);
    o.seekp(0);
    o.write(bytes.data(), (std::streamsize)bytes.size());
    return (bool)o;
  };

  const char* arms[] = {"content, mtime RESTORED", "content, NEW mtime",
                        "mtime only, same content",
                        "RE-EMITTED, identical bytes",
                        "COPIED to a new path"};
  // VPIPE_ANE_SWAP_ARMS="0,3,4" picks arms; default all.
  const char* sel = env_or_null_("VPIPE_ANE_SWAP_ARMS");
  for (int arm = 0; arm < 5; ++arm) {
    if (sel != nullptr &&
        std::string(sel).find(std::to_string(arm)) == std::string::npos) {
      continue;
    }
    const std::string a = root + "/vpipe-ane-swapA.mlmodelc";
    const std::string b = root + "/vpipe-ane-swapB.mlmodelc";
    const std::uint32_t sa = run * 7u + (std::uint32_t)arm * 101u + 1u;
    const std::uint32_t sb = sa + 50000u;
    ASSERT_TRUE(emit(a, sa) && emit(b, sb));
    double ms_a = 0, ms_b = 0, ms_a2 = 0, ms_sw = 0;
    const auto ya = load_run(a, &ms_a);        // cold compile A
    const auto ya2 = load_run(a, &ms_a2);      // warm A
    const auto yb = load_run(b, &ms_b);        // cold compile B
    ASSERT_TRUE(!ya.empty() && !yb.empty() && same(ya, ya2) &&
                !same(ya, yb));
    const std::string wa = a + "/weights/weight.bin";
    const std::string wb = b + "/weights/weight.bin";
    const auto mtime = fs::last_write_time(wa);
    std::string reload = a;
    if (arm == 0 || arm == 1) {
      ASSERT_TRUE(overwrite_in_place(wa, wb));
    }
    if (arm == 0) {
      fs::last_write_time(wa, mtime);
    } else if (arm == 1 || arm == 2) {
      fs::last_write_time(wa, fs::file_time_type::clock::now() +
                                  std::chrono::seconds(5));
    } else if (arm == 3) {
      // What ensure_ane_module_ does on every run: remove and re-emit
      // the same directory with the same bytes.
      ASSERT_TRUE(emit(a, sa));
    } else if (arm == 4) {
      reload = root + "/vpipe-ane-swapC.mlmodelc";
      std::error_code ec;
      fs::remove_all(reload, ec);
      fs::copy(a, reload, fs::copy_options::recursive, ec);
      ASSERT_TRUE(!ec);
    }
    const auto ys = load_run(reload, &ms_sw);
    std::printf("[ane_module] arm %d (%s): cold A %.0f ms, warm A %.0f ms, "
                "cold B %.0f ms | reload A after change %.0f ms -> output "
                "is %s\n", arm, arms[arm], ms_a, ms_a2, ms_b, ms_sw,
                ys.empty() ? "FAILED" : same(ys, yb) ? "B's weights"
                : same(ys, ya) ? "A's weights" : "NEITHER");
    std::error_code ec;
    fs::remove_all(a, ec);
    fs::remove_all(b, ec);
    fs::remove_all(root + "/vpipe-ane-swapC.mlmodelc", ec);
  }
}
#endif


#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

// System-wide wired memory, which is where the ANE keeps weights -- no
// process RSS counts it.
double
wired_mb_()
{
  vm_statistics64_data_t v;
  mach_msg_type_number_t c = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        (host_info64_t)&v, &c) != KERN_SUCCESS) {
    return -1.0;
  }
  return (double)v.wire_count * (double)vm_page_size / 1048576.0;
}

// fp16 bytes at a chosen ADDRESS alignment; `odd` adds one byte so the
// address is aligned to nothing.
struct AlignedFp16 {
  void*          base = nullptr;
  std::uint16_t* data = nullptr;
  std::size_t    elems = 0;

  AlignedFp16(std::size_t n, std::size_t align, bool odd) : elems(n)
  {
    const std::size_t a = std::max(align, sizeof(void*));
    const std::size_t bytes = (n * 2 + 1 + a - 1) / a * a;
    if (posix_memalign(&base, a, bytes) != 0) { base = nullptr; return; }
    data = static_cast<std::uint16_t*>(
        static_cast<void*>(static_cast<char*>(base) + (odd ? 1 : 0)));
  }
  ~AlignedFp16() { std::free(base); }
  AlignedFp16(const AlignedFp16&)            = delete;
  AlignedFp16& operator=(const AlignedFp16&) = delete;
};

struct RtForm {
  const char* name;
  bool        dyn;   // weight is a model INPUT rather than a constant
  bool        nt;    // weight declared [out,in], read with transpose_y
};

// Layer i of the chain alternates K->N and N->K.
void
wdims_(bool nt, int i, int K, int N, int* r, int* c)
{
  const int in  = (i % 2 == 0) ? K : N;
  const int out = (i % 2 == 0) ? N : K;
  *r = nt ? out : in;
  *c = nt ? in : out;
}

std::string
chain_mil_(const RtForm& f, int M, int K, int N, int L,
           const std::vector<std::size_t>& offs)
{
  std::ostringstream o;
  o << vpipe::AneEmitter::mil_preamble()
    << "    func main<ios18>(tensor<fp16, [" << M << ", " << K << "]> x";
  if (f.dyn) {
    for (int i = 0; i < L; ++i) {
      int r = 0, c = 0;
      wdims_(f.nt, i, K, N, &r, &c);
      o << ", tensor<fp16, [" << r << ", " << c << "]> w" << i;
    }
  }
  o << ") {\n";
  std::string prev = "x";
  for (int i = 0; i < L; ++i) {
    int r = 0, c = 0;
    wdims_(f.nt, i, K, N, &r, &c);
    const int out = (i % 2 == 0) ? N : K;
    const std::string w = "w" + std::to_string(i);
    if (!f.dyn) {
      o << "            tensor<fp16, [" << r << ", " << c << "]> " << w
        << " = const()[name = string(\"" << w << "\"), val = tensor<fp16, ["
        << r << ", " << c << "]>(BLOBFILE(path = string(\"@model_path/"
        << "weights/weight.bin\"), offset = uint64(" << offs[(std::size_t)i]
        << ")))];\n";
    }
    const std::string t = "t" + std::to_string(i);
    const std::string h = (i == L - 1) ? "out" : ("h" + std::to_string(i));
    o << "            bool " << t << "x = const()[name = string(\"" << t
      << "x\"), val = bool(false)];\n"
      << "            bool " << t << "y = const()[name = string(\"" << t
      << "y\"), val = bool(" << (f.nt ? "true" : "false") << ")];\n"
      << "            tensor<fp16, [" << M << ", " << out << "]> " << h
      << " = matmul(transpose_x = " << t << "x, transpose_y = " << t
      << "y, x = " << prev << ", y = " << w << ")[name = string(\"" << h
      << "\")];\n";
    prev = h;
  }
  o << "        } -> (out);\n}";
  return o.str();
}

double
rel_l2_f16_(const std::vector<std::uint16_t>& a,
            const std::vector<std::uint16_t>& b)
{
  if (a.size() != b.size() || a.empty()) { return -1.0; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    _Float16 x, y;
    std::memcpy(&x, &a[i], 2);
    std::memcpy(&y, &b[i], 2);
    const double d = (double)x - (double)y;
    num += d * d;
    den += (double)y * (double)y;
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace

// RUNTIME WEIGHTS, and whether the hardware-native layout helps them.
//
// A baked module holds its block's weights in wired memory for the run
// (~0.84 GB a Krea-2 feed-forward), which is what keeps the tier off a
// 16 GB box. A graph whose weights are INPUTS holds nothing between
// predicts -- if it still lands on the ANE, and at a usable rate.
//
// Constant weights compiled ~3.7x faster in the [out,in] + transpose_y
// form, i.e. the engine's native layout. A runtime weight is handed
// over as-is, with no compiler in between to fix the layout, so if
// anything the layout should matter MORE here.
//
// Four forms of the same chain -- const/input x NN/NT -- at three
// shapes, measured for compile, predict rate, wired memory and numeric
// agreement against const NN. Then, for the input-NT form, the INPUT
// buffer's address alignment, since a constant's data offset is always
// 64-aligned and an input's address is whatever the caller allocated.
//
// VPIPE_ANE_RUNTIME_BENCH=1 to run; VPIPE_ANE_RT_SHAPES="s1,s3" picks.
TEST(ane_module, runtime_weight_forms)
{
  if (env_or_null_("VPIPE_ANE_RUNTIME_BENCH") == nullptr) { return; }
  vpipe::Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return; }
  if (!vpipe::genai::AneModule::emitter_usable(&sess)) { return; }
  namespace fs = std::filesystem;
  using clk = std::chrono::steady_clock;
  const char* tmp = std::getenv("TMPDIR");
  const std::string root = tmp != nullptr ? tmp : "/tmp";
  // A fresh path per run: the program cache keys on file identity, so
  // this is what makes every compile below a COLD one.
  const long long run = std::chrono::duration_cast<std::chrono::microseconds>(
      clk::now().time_since_epoch()).count();
  const std::size_t page = (std::size_t)vm_page_size;

  struct Shape { const char* name; int M, K, N, L; };
  const Shape shapes[] = {
      {"s1", 2048, 2048,  2048, 4},   //   8 MB a weight: under the cliff
      {"s2", 2048, 4096,  4096, 4},   //  32 MB
      {"s3", 2048, 6144, 16384, 2},   // 193 MB: Krea-2's W1, untiled
  };
  const RtForm forms[] = {{"const NN", false, false},
                          {"const NT", false, true},
                          {"input NN", true, false},
                          {"input NT", true, true}};
  const char* only = env_or_null_("VPIPE_ANE_RT_SHAPES");
  // VPIPE_ANE_RT_M overrides the row count: holding the weight bytes
  // fixed while the FLOPs scale is what separates a fixed per-call
  // weight-staging cost (the gap in ms stays put) from a proportional
  // one (the gap in TOPS stays put).
  const char* m_env = env_or_null_("VPIPE_ANE_RT_M");

  for (const Shape& sh : shapes) {
    if (only != nullptr && std::string(only).find(sh.name) ==
                               std::string::npos) {
      continue;
    }
    const int M = (m_env != nullptr) ? std::atoi(m_env) : sh.M;
    const int K = sh.K, N = sh.N, L = sh.L;
    const double flops = 2.0 * M * K * N * L;

    // Logical weights Y_i [in,out], variance 1/in so the chain neither
    // blows up nor vanishes in fp16.
    std::vector<std::vector<float>> Y((std::size_t)L);
    {
      std::mt19937 rng(0xabcdu);
      for (int i = 0; i < L; ++i) {
        const int in = (i % 2 == 0) ? K : N, out = (i % 2 == 0) ? N : K;
        Y[(std::size_t)i].resize((std::size_t)in * out);
        const float a = std::sqrt(3.0f / (float)in);
        std::uniform_real_distribution<float> d(-a, a);
        for (auto& e : Y[(std::size_t)i]) { e = d(rng); }
      }
    }
    auto layout = [&](int i, bool nt) {
      const int in = (i % 2 == 0) ? K : N, out = (i % 2 == 0) ? N : K;
      std::vector<std::uint16_t> b((std::size_t)in * out);
      const std::vector<float>& y = Y[(std::size_t)i];
      for (int r = 0; r < in; ++r) {
        for (int c = 0; c < out; ++c) {
          const _Float16 v = (_Float16)y[(std::size_t)r * out + c];
          const std::size_t idx = nt ? (std::size_t)c * in + r
                                     : (std::size_t)r * out + c;
          std::memcpy(&b[idx], &v, 2);
        }
      }
      return b;
    };
    AlignedFp16 xin((std::size_t)M * K, page, false);
    {
      std::mt19937 rng(0x77u);
      std::uniform_real_distribution<float> d(-1.0f, 1.0f);
      for (std::size_t i = 0; i < xin.elems; ++i) {
        const _Float16 v = (_Float16)d(rng);
        std::memcpy(&xin.data[i], &v, 2);
      }
    }

    std::vector<std::uint16_t> ref;
    for (std::size_t fi = 0; fi < 4; ++fi) {
      const RtForm& f = forms[fi];
      std::vector<std::vector<std::uint16_t>> wb((std::size_t)L);
      for (int i = 0; i < L; ++i) { wb[(std::size_t)i] = layout(i, f.nt); }

      std::vector<vpipe::AneBlobSlab> slabs;
      std::vector<std::size_t> sizes;
      std::vector<vpipe::AneEmitter::Feature> ins, outs;
      ins.push_back({"x", {M, K}});
      for (int i = 0; i < L; ++i) {
        int r = 0, c = 0;
        wdims_(f.nt, i, K, N, &r, &c);
        if (f.dyn) {
          ins.push_back({"w" + std::to_string(i), {r, c}});
        } else {
          slabs.push_back({wb[(std::size_t)i].data(),
                           wb[(std::size_t)i].size() * 2});
          sizes.push_back(wb[(std::size_t)i].size() * 2);
        }
      }
      outs.push_back({"out", {M, K}});
      const std::vector<std::size_t> offs =
          vpipe::AneEmitter::blob_offsets(sizes);
      const std::string dir = root + "/vpipe-ane-rt-" + std::to_string(run) +
                              "-" + sh.name + "-" + std::to_string(fi) +
                              ".mlmodelc";
      std::string err;
      const bool wrote = vpipe::AneEmitter::write_bundle(
          dir, chain_mil_(f, M, K, N, L, offs), ins, outs, slabs, &err);
      ASSERT_TRUE(wrote);
      if (!wrote) {
        std::printf("[ane_rt] %s %s: write failed: %s\n", sh.name, f.name,
                    err.c_str());
        continue;
      }
      if (!f.dyn) { wb.clear(); }   // the bundle has them now

      const double w0 = wired_mb_();
      auto t0 = clk::now();
      auto m = mgr->load(dir, 3);
      const double compile_ms =
          std::chrono::duration<double, std::milli>(clk::now() - t0).count();
      if (m == nullptr) {
        std::printf("[ane_rt] %s %dx%dx%d L=%d %s: LOAD FAILED (%.0f ms)\n",
                    sh.name, M, K, N, L, f.name, compile_ms);
        std::error_code ec;
        fs::remove_all(dir, ec);
        continue;
      }
      const double w1 = wired_mb_();

      std::vector<std::unique_ptr<AlignedFp16>> wbuf;
      std::vector<vpipe::CoreMLPredictInput> inputs(1);
      inputs[0].name  = "x";
      inputs[0].data  = xin.data;
      inputs[0].dtype = vpipe::CoreMLDType::F16;
      inputs[0].shape = {M, K};
      auto bind_weights = [&](std::size_t align, bool odd, bool metal,
                              std::vector<vpipe::metal_compute::SharedBuffer>*
                                  keep) {
        wbuf.clear();
        if (keep != nullptr) { keep->clear(); }
        inputs.resize(1);
        for (int i = 0; i < L; ++i) {
          const auto& src = wb[(std::size_t)i];
          const void* p = nullptr;
          if (metal) {
            keep->push_back(mc->make_shared_buffer(src.size() * 2));
            std::memcpy(keep->back().contents(), src.data(), src.size() * 2);
            p = keep->back().contents();
          } else {
            wbuf.push_back(std::make_unique<AlignedFp16>(src.size(), align,
                                                         odd));
            std::memcpy(wbuf.back()->data, src.data(), src.size() * 2);
            p = wbuf.back()->data;
          }
          int r = 0, c = 0;
          wdims_(f.nt, i, K, N, &r, &c);
          vpipe::CoreMLPredictInput in;
          in.name  = "w" + std::to_string(i);
          in.data  = p;
          in.dtype = vpipe::CoreMLDType::F16;
          in.shape = {r, c};
          inputs.push_back(std::move(in));
        }
      };
      if (f.dyn) { bind_weights(page, false, false, nullptr); }

      std::vector<std::uint16_t> y((std::size_t)M * K, 0u);
      vpipe::CoreMLPredictOutput out;
      out.name          = "out";
      out.want          = vpipe::CoreMLDType::F16;
      out.backing       = y.data();
      out.backing_elems = y.size();
      auto best_of = [&](int n, bool cpu) {
        double best = 1e18;
        for (int k = 0; k < n; ++k) {
          const auto p0 = clk::now();
          const bool ok = m->predict(inputs, {&out, 1}, cpu);
          const double ms = std::chrono::duration<double, std::milli>(
              clk::now() - p0).count();
          if (ok && ms < best) { best = ms; }
        }
        return best;
      };
      best_of(2, false);                       // warm
      const double best = best_of(5, false);
      const double w2 = wired_mb_();
      if (ref.empty()) { ref = y; }
      std::printf("[ane_rt] %s %dx%dx%d L=%d %-8s compile %6.0f ms | "
                  "predict %7.2f ms = %5.2f TOPS | wired +%4.0f MB load, "
                  "%+5.0f MB run | rel-L2 vs const NN %.2e\n",
                  sh.name, M, K, N, L, f.name, compile_ms, best,
                  flops / (best / 1000.0) / 1e12, w1 - w0, w2 - w1,
                  rel_l2_f16_(y, ref));

      if (f.dyn && f.nt) {
        // What the ADDRESS of a runtime weight costs. A constant's data
        // offset is always 64-aligned; an input's is whatever the caller
        // allocated, and a copy CoreML makes to fix that would show up
        // here as time.
        struct Al { const char* name; std::size_t a; bool odd; bool metal; };
        const Al arms[] = {{"odd address", 64, true, false},
                           {"16-aligned (malloc)", 16, false, false},
                           {"64-aligned", 64, false, false},
                           {"page-aligned", page, false, false},
                           {"Metal SharedBuffer", page, false, true}};
        std::vector<vpipe::metal_compute::SharedBuffer> keep;
        for (const Al& al : arms) {
          bind_weights(al.a, al.odd, al.metal, &keep);
          best_of(1, false);
          const double b = best_of(4, false);
          std::printf("[ane_rt]   %s input NT, weights %-20s predict %7.2f "
                      "ms (%5.2f TOPS), rel-L2 %.2e\n", sh.name, al.name, b,
                      flops / (b / 1000.0) / 1e12, rel_l2_f16_(y, ref));
        }
        keep.clear();
        // No CPU arm: CoreMLLoadedModel::predict's cpu_only is
        // MLPredictionOptions.usesCPUOnly, which an ML program ignores,
        // so it measured the ANE a second time and read as parity.
      }
      m.reset();
      std::error_code ec;
      fs::remove_all(dir, ec);
    }
  }
}
#endif

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

// Emits the TILED matmul of `src` [M, in] by a logical [in, out] weight,
// tiles bk x bn, n-tile major and k-tile minor -- the emitter's order.
// Where each tile's bytes come from:
//   mode 0  a constant (BLOBFILE, offsets consumed in order)
//   mode 1  its own model input
//   mode 2  a slice of ONE whole-matrix input
struct RtTiler {
  std::ostringstream*                       body;
  std::vector<vpipe::AneEmitter::Feature>*  ins;
  const std::vector<std::size_t>*           offs;
  std::size_t                               next_off = 0;
  int                                       M = 0;
  bool                                      nt = false;
  int                                       mode = 0;
  int                                       uid = 0;

  std::string
  op(const std::string& src, int in, int out, int bk, int bn,
     const std::string& pfx, const std::string& final_name)
  {
    std::ostringstream& o = *body;
    const int nk = in / bk, nn = out / bn;
    std::string whole;
    if (mode == 2) {
      whole = pfx + "W";
      ins->push_back({whole, nt ? std::vector<std::int64_t>{out, in}
                                : std::vector<std::int64_t>{in, out}});
    }
    auto i2 = [&](const std::string& nm, int a, int b) {
      o << "            tensor<int32, [2]> " << nm << " = const()[name = "
        << "string(\"" << nm << "\"), val = tensor<int32, [2]>([" << a
        << ", " << b << "])];\n";
    };
    std::vector<std::string> cols;
    for (int j = 0; j < nn; ++j) {
      const int n0 = j * bn;
      std::string acc;
      for (int t = 0; t < nk; ++t) {
        const int k0 = t * bk;
        const std::string id = pfx + std::to_string(uid++);
        std::string xin = src;
        if (nk > 1) {
          i2(id + "_xb", 0, k0);
          i2(id + "_xe", M, k0 + bk);
          o << "            tensor<fp16, [" << M << ", " << bk << "]> " << id
            << "_x = slice_by_index(begin = " << id << "_xb, end = " << id
            << "_xe, x = " << src << ")[name = string(\"" << id
            << "_x\")];\n";
          xin = id + "_x";
        }
        const int wr = nt ? bn : bk, wc = nt ? bk : bn;
        std::string w = id + "_w";
        if (mode == 0) {
          o << "            tensor<fp16, [" << wr << ", " << wc << "]> " << w
            << " = const()[name = string(\"" << w << "\"), val = tensor<"
            << "fp16, [" << wr << ", " << wc << "]>(BLOBFILE(path = string("
            << "\"@model_path/weights/weight.bin\"), offset = uint64("
            << (*offs)[next_off++] << ")))];\n";
        } else if (mode == 1) {
          ins->push_back({w, {wr, wc}});
        } else {
          const int r0 = nt ? n0 : k0, c0 = nt ? k0 : n0;
          i2(id + "_wb", r0, c0);
          i2(id + "_we", r0 + wr, c0 + wc);
          o << "            tensor<fp16, [" << wr << ", " << wc << "]> " << w
            << " = slice_by_index(begin = " << id << "_wb, end = " << id
            << "_we, x = " << whole << ")[name = string(\"" << w
            << "\")];\n";
        }
        o << "            bool " << id << "_tx = const()[name = string(\""
          << id << "_tx\"), val = bool(false)];\n"
          << "            bool " << id << "_ty = const()[name = string(\""
          << id << "_ty\"), val = bool(" << (nt ? "true" : "false")
          << ")];\n"
          << "            tensor<fp16, [" << M << ", " << bn << "]> " << id
          << "_m = matmul(transpose_x = " << id << "_tx, transpose_y = "
          << id << "_ty, x = " << xin << ", y = " << w << ")[name = string(\""
          << id << "_m\")];\n";
        if (acc.empty()) {
          acc = id + "_m";
        } else {
          o << "            tensor<fp16, [" << M << ", " << bn << "]> " << id
            << "_a = add(x = " << acc << ", y = " << id << "_m)[name = "
            << "string(\"" << id << "_a\")];\n";
          acc = id + "_a";
        }
      }
      cols.push_back(acc);
    }
    o << "            int32 " << final_name << "_ax = const()[name = string(\""
      << final_name << "_ax\"), val = int32(1)];\n"
      << "            bool " << final_name << "_il = const()[name = string(\""
      << final_name << "_il\"), val = bool(false)];\n"
      << "            tensor<fp16, [" << M << ", " << out << "]> " << final_name
      << " = concat(axis = " << final_name << "_ax, interleave = "
      << final_name << "_il, values = (";
    for (std::size_t i = 0; i < cols.size(); ++i) {
      o << (i ? ", " : "") << cols[i];
    }
    o << "))[name = string(\"" << final_name << "\")];\n";
    return final_name;
  }
};

}  // namespace

// THE DESIGN NUMBER: a whole Krea-2 SwiGLU feed-forward (6144 -> 16384
// -> 6144, 2048 rows) with RUNTIME weights, against the same graph with
// constants. The const graph tiles 2048x2048 because the ANE's weight
// rate collapses above ~8-10 MB an op; a runtime-weight graph has to
// tile too, and the open question is WHERE the cliff sits for an input:
// on the op (so one whole-matrix input sliced in-graph is as good as
// tile inputs) or on the input (so it must be handed over pre-tiled).
//
// VPIPE_ANE_RT_FFN=1 to run.
TEST(ane_module, runtime_weight_ffn_tiled)
{
  if (env_or_null_("VPIPE_ANE_RT_FFN") == nullptr) { return; }
  vpipe::Session sess;
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (sess.metal_compute() == nullptr || mgr == nullptr) { return; }
  if (!vpipe::genai::AneModule::emitter_usable(&sess)) { return; }
  namespace fs = std::filesystem;
  using clk = std::chrono::steady_clock;
  const char* tmp = std::getenv("TMPDIR");
  const std::string root = tmp != nullptr ? tmp : "/tmp";
  const long long run = std::chrono::duration_cast<std::chrono::microseconds>(
      clk::now().time_since_epoch()).count();

  // VPIPE_ANE_RT_FFN_BLOCK: the tile. VPIPE_ANE_RT_FFN_ARMS="0,4" picks
  // arms, one per process when wired memory has to be attributed.
  const char* blk = env_or_null_("VPIPE_ANE_RT_FFN_BLOCK");
  const int M = 2048, K = 6144, N = 16384;
  const int B = (blk != nullptr) ? std::atoi(blk) : 2048;
  const char* arm_sel = env_or_null_("VPIPE_ANE_RT_FFN_ARMS");
  const double flops = 6.0 * M * K * N;

  // Logical weights: gate/up [K,N], down [N,K].
  std::vector<float> Yg((std::size_t)K * N), Yu((std::size_t)K * N),
                     Yd((std::size_t)N * K);
  {
    std::mt19937 rng(0x5151u);
    auto fill = [&](std::vector<float>& v, int in) {
      const float a = std::sqrt(3.0f / (float)in);
      std::uniform_real_distribution<float> d(-a, a);
      for (auto& e : v) { e = d(rng); }
    };
    fill(Yg, K); fill(Yu, K); fill(Yd, N);
  }
  auto f16 = [](float v) {
    const _Float16 h = (_Float16)v;
    std::uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
  };
  // Tiles of a logical [in,out] matrix in the emitter's order.
  auto tiles = [&](const std::vector<float>& Y, int in, int out, bool nt) {
    std::vector<std::vector<std::uint16_t>> ts;
    for (int j = 0; j < out / B; ++j) {
      for (int t = 0; t < in / B; ++t) {
        std::vector<std::uint16_t> b((std::size_t)B * B);
        for (int a = 0; a < B; ++a) {
          for (int c = 0; c < B; ++c) {
            const std::uint16_t v = f16(
                Y[(std::size_t)(t * B + a) * out + (std::size_t)(j * B + c)]);
            b[nt ? (std::size_t)c * B + a : (std::size_t)a * B + c] = v;
          }
        }
        ts.push_back(std::move(b));
      }
    }
    return ts;
  };
  auto whole = [&](const std::vector<float>& Y, int in, int out, bool nt) {
    std::vector<std::uint16_t> b((std::size_t)in * out);
    for (int r = 0; r < in; ++r) {
      for (int c = 0; c < out; ++c) {
        b[nt ? (std::size_t)c * in + r : (std::size_t)r * out + c] =
            f16(Y[(std::size_t)r * out + c]);
      }
    }
    return b;
  };
  std::vector<std::uint16_t> xin((std::size_t)M * K);
  {
    std::mt19937 rng(0x99u);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& e : xin) { e = f16(d(rng)); }
  }

  struct Arm { const char* name; int mode; bool nt; };
  const Arm arms[] = {{"const NT, tiled",               0, true},
                      {"input NN, 72 tile inputs",       1, false},
                      {"input NT, 72 tile inputs",       1, true},
                      {"input NN, 3 whole, sliced",      2, false},
                      {"input NT, 3 whole, sliced",      2, true}};
  std::vector<std::uint16_t> ref;
  for (std::size_t ai = 0; ai < 5; ++ai) {
    const Arm& arm = arms[ai];
    if (arm_sel != nullptr &&
        std::string(arm_sel).find(std::to_string(ai)) == std::string::npos) {
      continue;
    }
    // Bytes, in exactly the order the graph consumes them.
    std::vector<std::vector<std::uint16_t>> bytes;
    if (arm.mode == 2) {
      bytes.push_back(whole(Yg, K, N, arm.nt));
      bytes.push_back(whole(Yu, K, N, arm.nt));
      bytes.push_back(whole(Yd, N, K, arm.nt));
    } else {
      for (auto* spec : {&Yg, &Yu}) {
        for (auto& t : tiles(*spec, K, N, arm.nt)) {
          bytes.push_back(std::move(t));
        }
      }
      for (auto& t : tiles(Yd, N, K, arm.nt)) { bytes.push_back(std::move(t)); }
    }
    std::vector<std::size_t> sizes;
    std::vector<vpipe::AneBlobSlab> slabs;
    if (arm.mode == 0) {
      for (auto& b : bytes) {
        sizes.push_back(b.size() * 2);
        slabs.push_back({b.data(), b.size() * 2});
      }
    }
    const std::vector<std::size_t> offs =
        vpipe::AneEmitter::blob_offsets(sizes);

    std::ostringstream body;
    std::vector<vpipe::AneEmitter::Feature> dyn;
    RtTiler tl{&body, &dyn, &offs};
    tl.M = M; tl.nt = arm.nt; tl.mode = arm.mode;
    const std::string g = tl.op("x", K, N, B, B, "g", "gout");
    const std::string u = tl.op("x", K, N, B, B, "u", "uout");
    body << "            tensor<fp16, [" << M << ", " << N << "]> sg = silu(x"
         << " = " << g << ")[name = string(\"sg\")];\n"
         << "            tensor<fp16, [" << M << ", " << N << "]> hh = mul(x"
         << " = sg, y = " << u << ")[name = string(\"hh\")];\n";
    tl.op("hh", N, K, B, B, "d", "out");
    std::ostringstream mil;
    mil << vpipe::AneEmitter::mil_preamble() << "    func main<ios18>(tensor<"
        << "fp16, [" << M << ", " << K << "]> x";
    for (const auto& f : dyn) {
      mil << ", tensor<fp16, [" << f.shape[0] << ", " << f.shape[1] << "]> "
          << f.name;
    }
    mil << ") {\n" << body.str() << "        } -> (out);\n}";

    std::vector<vpipe::AneEmitter::Feature> ins{{"x", {M, K}}};
    for (const auto& f : dyn) { ins.push_back(f); }
    const std::vector<vpipe::AneEmitter::Feature> outs{{"out", {M, K}}};
    const std::string dir = root + "/vpipe-ane-rtffn-" + std::to_string(run) +
                            "-" + std::to_string(ai) + ".mlmodelc";
    std::string err;
    const bool wrote = vpipe::AneEmitter::write_bundle(dir, mil.str(), ins,
                                                       outs, slabs, &err);
    ASSERT_TRUE(wrote);
    if (!wrote) {
      std::printf("[ane_rtffn] write failed: %s\n", err.c_str());
      continue;
    }
    if (arm.mode == 0) { bytes.clear(); }

    const double w0 = wired_mb_();
    auto t0 = clk::now();
    auto m = mgr->load(dir, 3);
    const double compile_ms =
        std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    if (m == nullptr) {
      std::printf("[ane_rtffn] %-26s LOAD FAILED after %.0f ms\n", arm.name,
                  compile_ms);
      std::error_code ec;
      fs::remove_all(dir, ec);
      continue;
    }
    const double w1 = wired_mb_();
    std::vector<vpipe::CoreMLPredictInput> inputs(1);
    inputs[0].name = "x"; inputs[0].data = xin.data();
    inputs[0].dtype = vpipe::CoreMLDType::F16; inputs[0].shape = {M, K};
    for (std::size_t i = 0; i < dyn.size(); ++i) {
      vpipe::CoreMLPredictInput in;
      in.name = dyn[i].name; in.data = bytes[i].data();
      in.dtype = vpipe::CoreMLDType::F16; in.shape = dyn[i].shape;
      inputs.push_back(std::move(in));
    }
    std::vector<std::uint16_t> y((std::size_t)M * K, 0u);
    vpipe::CoreMLPredictOutput out;
    out.name = "out"; out.want = vpipe::CoreMLDType::F16;
    out.backing = y.data(); out.backing_elems = y.size();
    double first = -1.0, best = 1e18;
    for (int k = 0; k < 5; ++k) {
      const auto p0 = clk::now();
      const bool ok = m->predict(inputs, {&out, 1});
      const double ms = std::chrono::duration<double, std::milli>(
          clk::now() - p0).count();
      if (!ok) { break; }
      if (k == 0) { first = ms; } else if (ms < best) { best = ms; }
    }
    const double w2 = wired_mb_();
    if (ref.empty()) { ref = y; }
    std::printf("[ane_rtffn] tile %d, %-26s compile %6.0f ms | first "
                "predict %7.1f ms"
                " | predict %7.1f ms = %5.2f TOPS | wired +%4.0f MB load, "
                "%+5.0f MB run | rel-L2 vs const %.2e\n",
                B, arm.name, compile_ms, first, best,
                flops / (best / 1000.0) / 1e12, w1 - w0, w2 - w1,
                rel_l2_f16_(y, ref));
    m.reset();
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
}
#endif

#ifdef VPIPE_BUILD_APPLE_SILICON
// ONE runtime-weight module, MANY blocks: the pattern a shared module
// would actually run, and the two things that decide whether it is
// viable.
//
//   wired  A runtime module stages its weight inputs into wired memory
//          (~840 MB for a Krea-2 feed-forward) and keeps them after the
//          call. If that staging is per MODEL, one module costs it once
//          for all 28 blocks. If CoreML keeps one per distinct input
//          BUFFER, rotating blocks rebuilds the per-block bill.
//   truth  If staging is cached by pointer, refilling ONE buffer set
//          in place with the next block's bytes -- the natural design,
//          it is what BlockSlots does -- could silently reuse the
//          previous block's weights. Every output is checked against
//          its own set's reference, so that fails loudly rather than
//          quietly.
//
// The fastest form measured (NN, three whole inputs sliced in-graph),
// rotated over 4 weight sets: distinct buffers per set, then one buffer
// set refilled before each call.
//
// VPIPE_ANE_RT_REBIND=1 to run.
TEST(ane_module, runtime_weight_rebinding)
{
  if (env_or_null_("VPIPE_ANE_RT_REBIND") == nullptr) { return; }
  vpipe::Session sess;
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (sess.metal_compute() == nullptr || mgr == nullptr) { return; }
  if (!vpipe::genai::AneModule::emitter_usable(&sess)) { return; }
  namespace fs = std::filesystem;
  using clk = std::chrono::steady_clock;
  const char* tmp = std::getenv("TMPDIR");
  const std::string root = tmp != nullptr ? tmp : "/tmp";
  const long long run = std::chrono::duration_cast<std::chrono::microseconds>(
      clk::now().time_since_epoch()).count();
  const int M = 2048, K = 6144, N = 16384, B = 2048;
  const int kSets = 4;

  std::ostringstream body;
  std::vector<vpipe::AneEmitter::Feature> dyn;
  const std::vector<std::size_t> no_offs;
  RtTiler tl{&body, &dyn, &no_offs};
  tl.M = M; tl.nt = false; tl.mode = 2;
  const std::string g = tl.op("x", K, N, B, B, "g", "gout");
  const std::string u = tl.op("x", K, N, B, B, "u", "uout");
  body << "            tensor<fp16, [" << M << ", " << N << "]> sg = silu(x = "
       << g << ")[name = string(\"sg\")];\n"
       << "            tensor<fp16, [" << M << ", " << N << "]> hh = mul(x = "
       << "sg, y = " << u << ")[name = string(\"hh\")];\n";
  tl.op("hh", N, K, B, B, "d", "out");
  std::ostringstream mil;
  mil << vpipe::AneEmitter::mil_preamble() << "    func main<ios18>(tensor<"
      << "fp16, [" << M << ", " << K << "]> x";
  for (const auto& f : dyn) {
    mil << ", tensor<fp16, [" << f.shape[0] << ", " << f.shape[1] << "]> "
        << f.name;
  }
  mil << ") {\n" << body.str() << "        } -> (out);\n}";
  std::vector<vpipe::AneEmitter::Feature> ins{{"x", {M, K}}};
  for (const auto& f : dyn) { ins.push_back(f); }
  const std::vector<vpipe::AneEmitter::Feature> outs{{"out", {M, K}}};
  const std::string dir = root + "/vpipe-ane-rebind-" + std::to_string(run) +
                          ".mlmodelc";
  std::string err;
  ASSERT_TRUE(vpipe::AneEmitter::write_bundle(dir, mil.str(), ins, outs, {},
                                              &err));
  ASSERT_TRUE(dyn.size() == 3u);
  if (dyn.size() != 3u) { return; }

  // Weight sets, generated straight to fp16 (whole NN matrices, in the
  // order the graph declares them: g, u, d).
  auto f16 = [](float v) {
    const _Float16 h = (_Float16)v;
    std::uint16_t w;
    std::memcpy(&w, &h, 2);
    return w;
  };
  std::vector<std::array<std::vector<std::uint16_t>, 3>> sets(kSets);
  for (int s = 0; s < kSets; ++s) {
    std::mt19937 rng(0x1000u + (std::uint32_t)s);
    for (int i = 0; i < 3; ++i) {
      const int in = (i < 2) ? K : N;
      const float a = std::sqrt(3.0f / (float)in);
      std::uniform_real_distribution<float> d(-a, a);
      auto& v = sets[(std::size_t)s][(std::size_t)i];
      v.resize((std::size_t)K * N);
      for (auto& e : v) { e = f16(d(rng)); }
    }
  }
  std::vector<std::uint16_t> xin((std::size_t)M * K);
  {
    std::mt19937 rng(0x99u);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& e : xin) { e = f16(d(rng)); }
  }

  const double w0 = wired_mb_();
  auto m = mgr->load(dir, 3);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const double w_load = wired_mb_();

  std::vector<std::uint16_t> y((std::size_t)M * K, 0u);
  vpipe::CoreMLPredictOutput out;
  out.name = "out"; out.want = vpipe::CoreMLDType::F16;
  out.backing = y.data(); out.backing_elems = y.size();
  auto predict = [&](const std::array<const void*, 3>& w, double* ms) {
    std::vector<vpipe::CoreMLPredictInput> inputs(4);
    inputs[0].name = "x"; inputs[0].data = xin.data();
    inputs[0].dtype = vpipe::CoreMLDType::F16; inputs[0].shape = {M, K};
    for (int i = 0; i < 3; ++i) {
      inputs[(std::size_t)i + 1].name  = dyn[(std::size_t)i].name;
      inputs[(std::size_t)i + 1].data  = w[(std::size_t)i];
      inputs[(std::size_t)i + 1].dtype = vpipe::CoreMLDType::F16;
      inputs[(std::size_t)i + 1].shape = dyn[(std::size_t)i].shape;
    }
    const auto p0 = clk::now();
    const bool ok = m->predict(inputs, {&out, 1});
    *ms = std::chrono::duration<double, std::milli>(clk::now() - p0).count();
    return ok;
  };

  // Arm A: every set in its own buffers. Round 0 also records each
  // set's reference output.
  std::vector<std::vector<std::uint16_t>> refs(kSets);
  std::vector<double> times_a;
  std::printf("[ane_rebind] model loaded: wired %+.0f MB\n", w_load - w0);
  for (int round = 0; round < 3; ++round) {
    for (int s = 0; s < kSets; ++s) {
      const auto& st = sets[(std::size_t)s];
      double ms = 0.0;
      ASSERT_TRUE(predict({st[0].data(), st[1].data(), st[2].data()}, &ms));
      if (round == 0) {
        refs[(std::size_t)s] = y;
      } else {
        times_a.push_back(ms);
        const double r = rel_l2_f16_(y, refs[(std::size_t)s]);
        if (r != 0.0) {
          std::printf("[ane_rebind]   A round %d set %d MISMATCH rel-L2 %.2e\n",
                      round, s, r);
        }
        EXPECT_TRUE(r == 0.0);
      }
      if (round == 0) {
        std::printf("[ane_rebind]   A first visit of set %d: %.1f ms, wired "
                    "%+.0f MB since load\n", s, ms, wired_mb_() - w_load);
      }
    }
    std::printf("[ane_rebind]   A after round %d: wired %+.0f MB since load\n",
                round, wired_mb_() - w_load);
  }
  // The sets must actually differ, or the check above proves nothing.
  EXPECT_TRUE(rel_l2_f16_(refs[0], refs[1]) > 1e-3);

  // Arm B: ONE buffer set, refilled in place before each call.
  const double w_b0 = wired_mb_();
  std::array<std::vector<std::uint16_t>, 3> slot;
  for (int i = 0; i < 3; ++i) {
    slot[(std::size_t)i].resize((std::size_t)K * N);
  }
  std::vector<double> times_b, fills_b;
  int wrong = 0;
  for (int round = 0; round < 3; ++round) {
    for (int s = 0; s < kSets; ++s) {
      const auto f0 = clk::now();
      for (int i = 0; i < 3; ++i) {
        std::memcpy(slot[(std::size_t)i].data(),
                    sets[(std::size_t)s][(std::size_t)i].data(),
                    slot[(std::size_t)i].size() * 2);
      }
      fills_b.push_back(std::chrono::duration<double, std::milli>(
          clk::now() - f0).count());
      double ms = 0.0;
      ASSERT_TRUE(predict({slot[0].data(), slot[1].data(), slot[2].data()},
                          &ms));
      times_b.push_back(ms);
      const double r = rel_l2_f16_(y, refs[(std::size_t)s]);
      if (r != 0.0) {
        ++wrong;
        std::printf("[ane_rebind]   B round %d set %d WRONG WEIGHTS: rel-L2 "
                    "%.2e vs its own reference\n", round, s, r);
      }
    }
    std::printf("[ane_rebind]   B after round %d: wired %+.0f MB since "
                "arm B began\n", round, wired_mb_() - w_b0);
  }
  EXPECT_TRUE(wrong == 0);

  auto median = [](std::vector<double> v) {
    if (v.empty()) { return 0.0; }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const double flops = 6.0 * M * K * N;
  std::printf("[ane_rebind] A distinct buffers, rotating: median %.1f ms "
              "(%.2f TOPS), min %.1f, max %.1f\n", median(times_a),
              flops / (median(times_a) / 1000.0) / 1e12,
              *std::min_element(times_a.begin(), times_a.end()),
              *std::max_element(times_a.begin(), times_a.end()));
  std::printf("[ane_rebind] B one buffer set, refilled: median %.1f ms "
              "(%.2f TOPS), min %.1f, max %.1f | refill median %.1f ms | "
              "%d wrong outputs\n", median(times_b),
              flops / (median(times_b) / 1000.0) / 1e12,
              *std::min_element(times_b.begin(), times_b.end()),
              *std::max_element(times_b.begin(), times_b.end()),
              median(fills_b), wrong);
  // Arm C: the same rotation, with the weights in IOSurface-backed
  // half-float PIXEL BUFFERS refilled in place. The input penalty scales
  // with rows (the weight is re-staged per internal chunk); an array
  // that already IS ANE-mappable memory is the one binding that could
  // let the engine read it where it lives instead.
  {
    auto make_pb = [](int rows, int cols) -> CVPixelBufferRef {
      CFDictionaryRef empty = CFDictionaryCreate(
          kCFAllocatorDefault, nullptr, nullptr, 0,
          &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
      const void* keys[] = {kCVPixelBufferIOSurfacePropertiesKey};
      const void* vals[] = {empty};
      CFDictionaryRef attrs = CFDictionaryCreate(
          kCFAllocatorDefault, keys, vals, 1, &kCFTypeDictionaryKeyCallBacks,
          &kCFTypeDictionaryValueCallBacks);
      CVPixelBufferRef pb = nullptr;
      const CVReturn rc = CVPixelBufferCreate(
          kCFAllocatorDefault, (size_t)cols, (size_t)rows,
          kCVPixelFormatType_OneComponent16Half, attrs, &pb);
      CFRelease(attrs);
      CFRelease(empty);
      return rc == kCVReturnSuccess ? pb : nullptr;
    };
    std::array<CVPixelBufferRef, 3> pbs{};
    bool pb_ok = true;
    for (int i = 0; i < 3; ++i) {
      const auto& shp = dyn[(std::size_t)i].shape;
      pbs[(std::size_t)i] = make_pb((int)shp[0], (int)shp[1]);
      if (pbs[(std::size_t)i] == nullptr) { pb_ok = false; }
    }
    ASSERT_TRUE(pb_ok);
    if (pb_ok) {
      for (int i = 0; i < 3; ++i) {
        CVPixelBufferRef pb = pbs[(std::size_t)i];
        CVPixelBufferLockBaseAddress(pb, 0);
        const auto base = (std::uintptr_t)CVPixelBufferGetBaseAddress(pb);
        std::printf("[ane_rebind]   C buffer %d: %zux%zu, bytes/row %zu "
                    "(tight %zu), IOSurface %s, base %% 16384 = %zu\n", i,
                    CVPixelBufferGetHeight(pb), CVPixelBufferGetWidth(pb),
                    CVPixelBufferGetBytesPerRow(pb),
                    CVPixelBufferGetWidth(pb) * 2,
                    CVPixelBufferGetIOSurface(pb) != nullptr ? "yes" : "NO",
                    (std::size_t)(base % 16384));
        CVPixelBufferUnlockBaseAddress(pb, 0);
      }
      auto predict_pb = [&](double* ms) {
        std::vector<vpipe::CoreMLPredictInput> inputs(4);
        inputs[0].name = "x"; inputs[0].data = xin.data();
        inputs[0].dtype = vpipe::CoreMLDType::F16; inputs[0].shape = {M, K};
        for (int i = 0; i < 3; ++i) {
          auto& in = inputs[(std::size_t)i + 1];
          in.name         = dyn[(std::size_t)i].name;
          in.pixel_buffer = pbs[(std::size_t)i];
          in.dtype        = vpipe::CoreMLDType::F16;
          in.shape        = dyn[(std::size_t)i].shape;
        }
        const auto p0 = clk::now();
        const bool ok = m->predict(inputs, {&out, 1});
        *ms = std::chrono::duration<double, std::milli>(clk::now() - p0)
                  .count();
        return ok;
      };
      const double w_c0 = wired_mb_();
      std::vector<double> times_c, fills_c;
      int wrong_c = 0;
      for (int round = 0; round < 3; ++round) {
        for (int s = 0; s < kSets; ++s) {
          const auto f0 = clk::now();
          for (int i = 0; i < 3; ++i) {
            CVPixelBufferRef pb = pbs[(std::size_t)i];
            const int rows = (int)dyn[(std::size_t)i].shape[0];
            const int cols = (int)dyn[(std::size_t)i].shape[1];
            const auto& src = sets[(std::size_t)s][(std::size_t)i];
            CVPixelBufferLockBaseAddress(pb, 0);
            auto* dst = static_cast<std::uint8_t*>(
                CVPixelBufferGetBaseAddress(pb));
            const std::size_t bpr = CVPixelBufferGetBytesPerRow(pb);
            for (int r = 0; r < rows; ++r) {
              std::memcpy(dst + (std::size_t)r * bpr,
                          src.data() + (std::size_t)r * (std::size_t)cols,
                          (std::size_t)cols * 2);
            }
            CVPixelBufferUnlockBaseAddress(pb, 0);
          }
          fills_c.push_back(std::chrono::duration<double, std::milli>(
              clk::now() - f0).count());
          double ms = 0.0;
          const bool ok = predict_pb(&ms);
          ASSERT_TRUE(ok);
          if (!ok) { break; }
          if (round > 0) { times_c.push_back(ms); }
          const double r = rel_l2_f16_(y, refs[(std::size_t)s]);
          if (r != 0.0) {
            ++wrong_c;
            std::printf("[ane_rebind]   C round %d set %d: rel-L2 %.2e vs its "
                        "own reference\n", round, s, r);
          }
          if (round == 0) {
            std::printf("[ane_rebind]   C first pass, set %d: %.1f ms\n", s,
                        ms);
          }
        }
        std::printf("[ane_rebind]   C after round %d: wired %+.0f MB since "
                    "arm C began\n", round, wired_mb_() - w_c0);
      }
      EXPECT_TRUE(wrong_c == 0);
      if (!times_c.empty()) {
        std::printf("[ane_rebind] C IOSurface pixel buffers, refilled: median "
                    "%.1f ms (%.2f TOPS), min %.1f, max %.1f | refill median "
                    "%.1f ms | %d wrong outputs\n", median(times_c),
                    flops / (median(times_c) / 1000.0) / 1e12,
                    *std::min_element(times_c.begin(), times_c.end()),
                    *std::max_element(times_c.begin(), times_c.end()),
                    median(fills_c), wrong_c);
      }
    }
    for (CVPixelBufferRef pb : pbs) {
      if (pb != nullptr) { CVPixelBufferRelease(pb); }
    }
  }
  m.reset();
  std::error_code ec;
  fs::remove_all(dir, ec);
}
#endif

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

// The fastest runtime-weight FFN form (NN, three whole inputs sliced
// in-graph). Fills `dyn` with the weight inputs in declaration order.
std::string
rt_ffn_mil_(int M, int K, int N, int B,
            std::vector<vpipe::AneEmitter::Feature>* dyn)
{
  std::ostringstream body;
  const std::vector<std::size_t> no_offs;
  RtTiler tl{&body, dyn, &no_offs};
  tl.M = M; tl.mode = 2;
  // VPIPE_ANE_RT_ISO_NT: declare the weights [out,in] and read them with
  // transpose_y -- the checkpoint's own layout, which lets staging copy
  // rows sequentially instead of transposing 576 MB per block.
  tl.nt = env_or_null_("VPIPE_ANE_RT_ISO_NT") != nullptr;
  const std::string g = tl.op("x", K, N, B, B, "g", "gout");
  const std::string u = tl.op("x", K, N, B, B, "u", "uout");
  body << "            tensor<fp16, [" << M << ", " << N << "]> sg = silu(x = "
       << g << ")[name = string(\"sg\")];\n"
       << "            tensor<fp16, [" << M << ", " << N << "]> hh = mul(x = "
       << "sg, y = " << u << ")[name = string(\"hh\")];\n";
  tl.op("hh", N, K, B, B, "d", "out");
  std::ostringstream mil;
  mil << vpipe::AneEmitter::mil_preamble() << "    func main<ios18>(tensor<"
      << "fp16, [" << M << ", " << K << "]> x";
  for (const auto& f : *dyn) {
    mil << ", tensor<fp16, [" << f.shape[0] << ", " << f.shape[1] << "]> "
        << f.name;
  }
  mil << ") {\n" << body.str() << "        } -> (out);\n}";
  return mil.str();
}

CVPixelBufferRef
make_f16_pixel_buffer_(int rows, int cols)
{
  CFDictionaryRef empty = CFDictionaryCreate(
      kCFAllocatorDefault, nullptr, nullptr, 0,
      &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  const void* keys[] = {kCVPixelBufferIOSurfacePropertiesKey};
  const void* vals[] = {empty};
  CFDictionaryRef attrs = CFDictionaryCreate(
      kCFAllocatorDefault, keys, vals, 1, &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  CVPixelBufferRef pb = nullptr;
  const CVReturn rc = CVPixelBufferCreate(
      kCFAllocatorDefault, (size_t)cols, (size_t)rows,
      kCVPixelFormatType_OneComponent16Half, attrs, &pb);
  CFRelease(attrs);
  CFRelease(empty);
  return rc == kCVReturnSuccess ? pb : nullptr;
}

void
fill_pixel_buffer_(CVPixelBufferRef pb, const std::uint16_t* src, int rows,
                   int cols)
{
  CVPixelBufferLockBaseAddress(pb, 0);
  auto* dst = static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(pb));
  const std::size_t bpr = CVPixelBufferGetBytesPerRow(pb);
  for (int r = 0; r < rows; ++r) {
    std::memcpy(dst + (std::size_t)r * bpr,
                src + (std::size_t)r * (std::size_t)cols,
                (std::size_t)cols * 2);
  }
  CVPixelBufferUnlockBaseAddress(pb, 0);
}

std::uint64_t
fnv1a_(const std::vector<std::uint16_t>& v)
{
  std::uint64_t h = 1469598103934665603ull;
  const auto* p = static_cast<const unsigned char*>(
      static_cast<const void*>(v.data()));
  for (std::size_t i = 0; i < v.size() * 2; ++i) {
    h = (h ^ p[i]) * 1099511628211ull;
  }
  return h;
}

}  // namespace

// ONE BINDING PER PROCESS, so wired memory can be attributed. In a shared
// process the model's weight staging and the caller's IOSurfaces land in
// the same system-wide counter and cannot be told apart -- which is the
// whole question: does an IOSurface-backed input REPLACE the ~625 MB the
// model stages, or add to it?
//
//   VPIPE_ANE_RT_ISO=B   plain host buffers, refilled
//   VPIPE_ANE_RT_ISO=C   weights in IOSurface pixel buffers, refilled
//   VPIPE_ANE_RT_ISO=D   weights AND the activation in pixel buffers
//
// Each set's output hash is printed, so correctness compares ACROSS
// processes: all three arms must print the same two hashes.
TEST(ane_module, runtime_weight_binding_isolated)
{
  const char* arm_env = env_or_null_("VPIPE_ANE_RT_ISO");
  if (arm_env == nullptr) { return; }
  const char arm = arm_env[0];
  vpipe::Session sess;
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (sess.metal_compute() == nullptr || mgr == nullptr) { return; }
  if (!vpipe::genai::AneModule::emitter_usable(&sess)) { return; }
  namespace fs = std::filesystem;
  using clk = std::chrono::steady_clock;
  const char* tmp = std::getenv("TMPDIR");
  const std::string root = tmp != nullptr ? tmp : "/tmp";
  const long long run = std::chrono::duration_cast<std::chrono::microseconds>(
      clk::now().time_since_epoch()).count();
  // VPIPE_ANE_RT_ISO_BLOCK: the tile. 1024x1024 runtime operands measured
  // at the constant path's peak rate, where 2048x2048 ones paid ~1.45x.
  const char* blk = env_or_null_("VPIPE_ANE_RT_ISO_BLOCK");
  const int B = (blk != nullptr) ? std::atoi(blk) : 2048;
  const int M = 2048, K = 6144, N = 16384, kSets = 2;
  const double flops = 6.0 * M * K * N;

  std::vector<vpipe::AneEmitter::Feature> dyn;
  const std::string mil = rt_ffn_mil_(M, K, N, B, &dyn);
  std::vector<vpipe::AneEmitter::Feature> ins{{"x", {M, K}}};
  for (const auto& f : dyn) { ins.push_back(f); }
  const std::vector<vpipe::AneEmitter::Feature> outs{{"out", {M, K}}};
  const std::string dir = root + "/vpipe-ane-iso-" + std::to_string(run) +
                          ".mlmodelc";
  std::string err;
  ASSERT_TRUE(vpipe::AneEmitter::write_bundle(dir, mil, ins, outs, {}, &err));
  ASSERT_TRUE(dyn.size() == 3u);
  if (dyn.size() != 3u) { return; }

  auto f16 = [](float v) {
    const _Float16 h = (_Float16)v;
    std::uint16_t w;
    std::memcpy(&w, &h, 2);
    return w;
  };
  // The same seeds as runtime_weight_rebinding, so hashes are comparable.
  std::vector<std::array<std::vector<std::uint16_t>, 3>> sets(kSets);
  for (int s = 0; s < kSets; ++s) {
    std::mt19937 rng(0x1000u + (std::uint32_t)s);
    for (int i = 0; i < 3; ++i) {
      const int in = (i < 2) ? K : N;
      const float a = std::sqrt(3.0f / (float)in);
      std::uniform_real_distribution<float> d(-a, a);
      auto& v = sets[(std::size_t)s][(std::size_t)i];
      v.resize((std::size_t)K * N);
      for (auto& e : v) { e = f16(d(rng)); }
    }
  }
  std::vector<std::uint16_t> xin((std::size_t)M * K);
  {
    std::mt19937 rng(0x99u);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& e : xin) { e = f16(d(rng)); }
  }

  const double w_start = wired_mb_();
  // Caller-owned buffers FIRST, so their own cost is measured before the
  // model exists to muddy it.
  std::array<std::vector<std::uint16_t>, 3> host;
  std::array<CVPixelBufferRef, 3> pbw{};
  CVPixelBufferRef pbx = nullptr;
  if (arm == 'B') {
    for (int i = 0; i < 3; ++i) {
      host[(std::size_t)i].resize((std::size_t)K * N);
    }
  } else {
    for (int i = 0; i < 3; ++i) {
      pbw[(std::size_t)i] = make_f16_pixel_buffer_(
          (int)dyn[(std::size_t)i].shape[0], (int)dyn[(std::size_t)i].shape[1]);
      ASSERT_TRUE(pbw[(std::size_t)i] != nullptr);
    }
    if (arm == 'D') {
      pbx = make_f16_pixel_buffer_(M, K);
      ASSERT_TRUE(pbx != nullptr);
      fill_pixel_buffer_(pbx, xin.data(), M, K);
    }
    // Touch every page, so an IOSurface that wires lazily shows its cost.
    for (int i = 0; i < 3; ++i) {
      fill_pixel_buffer_(pbw[(std::size_t)i], sets[0][(std::size_t)i].data(),
                         (int)dyn[(std::size_t)i].shape[0],
                         (int)dyn[(std::size_t)i].shape[1]);
    }
  }
  const double w_buffers = wired_mb_();

  const auto c0 = clk::now();
  auto m = mgr->load(dir, 3);
  const double compile_ms =
      std::chrono::duration<double, std::milli>(clk::now() - c0).count();
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const double w_load = wired_mb_();

  std::vector<std::uint16_t> y((std::size_t)M * K, 0u);
  vpipe::CoreMLPredictOutput out;
  out.name = "out"; out.want = vpipe::CoreMLDType::F16;
  out.backing = y.data(); out.backing_elems = y.size();
  auto refill = [&](int s) {
    for (int i = 0; i < 3; ++i) {
      const auto& src = sets[(std::size_t)s][(std::size_t)i];
      if (arm == 'B') {
        std::memcpy(host[(std::size_t)i].data(), src.data(), src.size() * 2);
      } else {
        fill_pixel_buffer_(pbw[(std::size_t)i], src.data(),
                           (int)dyn[(std::size_t)i].shape[0],
                           (int)dyn[(std::size_t)i].shape[1]);
      }
    }
  };
  auto predict = [&](double* ms) {
    std::vector<vpipe::CoreMLPredictInput> inputs(4);
    inputs[0].name = "x"; inputs[0].dtype = vpipe::CoreMLDType::F16;
    inputs[0].shape = {M, K};
    if (arm == 'D') { inputs[0].pixel_buffer = pbx; }
    else            { inputs[0].data = xin.data(); }
    for (int i = 0; i < 3; ++i) {
      auto& in = inputs[(std::size_t)i + 1];
      in.name  = dyn[(std::size_t)i].name;
      in.dtype = vpipe::CoreMLDType::F16;
      in.shape = dyn[(std::size_t)i].shape;
      if (arm == 'B') { in.data = host[(std::size_t)i].data(); }
      else            { in.pixel_buffer = pbw[(std::size_t)i]; }
    }
    const auto p0 = clk::now();
    const bool ok = m->predict(inputs, {&out, 1});
    *ms = std::chrono::duration<double, std::milli>(clk::now() - p0).count();
    return ok;
  };

  refill(0);
  double ms = 0.0;
  ASSERT_TRUE(predict(&ms));
  const double w_first = wired_mb_();
  std::vector<double> times;
  std::array<std::uint64_t, 2> hashes{};
  for (int round = 0; round < 3; ++round) {
    for (int s = 0; s < kSets; ++s) {
      refill(s);
      ASSERT_TRUE(predict(&ms));
      if (round > 0) { times.push_back(ms); }
      hashes[(std::size_t)s] = fnv1a_(y);
    }
  }
  const double w_end = wired_mb_();
  std::sort(times.begin(), times.end());
  const double med = times.empty() ? 0.0 : times[times.size() / 2];
  std::printf("[ane_iso] arm %c, tile %d, compile %.0f ms | wired: "
              "buffers %+.0f MB, model load "
              "%+.0f, first predict %+.0f, rotation %+.0f => TOTAL %+.0f MB "
              "| predict median %.1f ms (%.2f TOPS) | out hash set0 %016llx "
              "set1 %016llx\n",
              arm, B, compile_ms, w_buffers - w_start, w_load - w_buffers,
              w_first - w_load,
              w_end - w_first, w_end - w_start, med,
              flops / (med / 1000.0) / 1e12, (unsigned long long)hashes[0],
              (unsigned long long)hashes[1]);
  m.reset();
  for (CVPixelBufferRef pb : pbw) {
    if (pb != nullptr) { CVPixelBufferRelease(pb); }
  }
  if (pbx != nullptr) { CVPixelBufferRelease(pbx); }
  std::error_code ec;
  fs::remove_all(dir, ec);
}
#endif


#ifdef VPIPE_BUILD_APPLE_SILICON
// WHICH SIDE OF A RUNTIME MATMUL WANTS TO BE TRANSPOSED?
//
// With a constant weight, the [out,in] + transpose_y form compiled 3.7-11x
// faster at the same rate; with a runtime weight it ran 3-19% SLOWER.
// Both of those had a transpose on one side only, on a rectangular
// weight, so layout and shape moved together. Here both operands are
// runtime inputs and SQUARE (1024 x 1024), so every transpose combination
// has identical shapes and identical FLOPs, and whatever differs is the
// transpose itself:
//
//   A @ B,  AT @ B,  A @ BT,  AT @ BT
//
// One 1024^3 matmul is ~1 GFLOP -- well under the ~1-2 ms fixed cost of a
// predict -- so each form is a CHAIN (h = op(h) @ op(b), same flags every
// layer), timed at L = 1 and L = 16, and the per-matmul figure is the
// marginal. Tiny graphs can leave the ANE entirely, and predict's
// cpu_only is ignored by ML programs, so the ENGINE is read from the
// compile-cache entries each load creates rather than inferred from time.
//
// VPIPE_ANE_RT_TRANSPOSE=1 to run; VPIPE_ANE_RT_T_PB=1 binds both inputs
// as IOSurface-backed pixel buffers instead of host memory.
TEST(ane_module, runtime_transpose_square)
{
  if (env_or_null_("VPIPE_ANE_RT_TRANSPOSE") == nullptr) { return; }
  const bool use_pb = env_or_null_("VPIPE_ANE_RT_T_PB") != nullptr;
  // VPIPE_ANE_RT_T_CONST=1: B becomes a CONSTANT, the baseline that says
  // whether a runtime operand costs anything at this size.
  const bool use_const = env_or_null_("VPIPE_ANE_RT_T_CONST") != nullptr;
  vpipe::Session sess;
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (sess.metal_compute() == nullptr || mgr == nullptr) { return; }
  if (!vpipe::genai::AneModule::emitter_usable(&sess)) { return; }
  namespace fs = std::filesystem;
  using clk = std::chrono::steady_clock;
  const char* tmp = std::getenv("TMPDIR");
  const std::string root = tmp != nullptr ? tmp : "/tmp";
  const long long run = std::chrono::duration_cast<std::chrono::microseconds>(
      clk::now().time_since_epoch()).count();
  const int D = 1024;
  const double flops_one = 2.0 * D * D * D;

  // A uniform +-1; B scaled so a chain of 16 neither overflows nor
  // vanishes in fp16.
  std::vector<float> Af((std::size_t)D * D), Bf((std::size_t)D * D);
  {
    std::mt19937 rng(0x7a5eu);
    std::uniform_real_distribution<float> da(-1.0f, 1.0f);
    const float sb = std::sqrt(3.0f / (float)D);
    std::uniform_real_distribution<float> db(-sb, sb);
    for (auto& e : Af) { e = da(rng); }
    for (auto& e : Bf) { e = db(rng); }
  }
  auto to_f16 = [](const std::vector<float>& v) {
    std::vector<std::uint16_t> o(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      const _Float16 h = (_Float16)v[i];
      std::memcpy(&o[i], &h, 2);
    }
    return o;
  };
  const std::vector<std::uint16_t> A16 = to_f16(Af), B16 = to_f16(Bf);
  CVPixelBufferRef pa = nullptr, pbb = nullptr;
  if (use_pb) {
    pa = make_f16_pixel_buffer_(D, D);
    pbb = make_f16_pixel_buffer_(D, D);
    ASSERT_TRUE(pa != nullptr && pbb != nullptr);
    if (pa == nullptr || pbb == nullptr) { return; }
    fill_pixel_buffer_(pa, A16.data(), D, D);
    fill_pixel_buffer_(pbb, B16.data(), D, D);
  }

  const char* home = std::getenv("HOME");
  const fs::path cache = fs::path(home != nullptr ? home : "") /
                         "Library/Caches/vpipe_test/"
                         "com.apple.e5rt.e5bundlecache";
  auto cache_entries = [&]() {
    std::set<std::string> out;
    std::error_code ec;
    if (!fs::exists(cache, ec)) { return out; }
    for (const auto& b : fs::directory_iterator(cache, ec)) {
      if (!b.is_directory()) { continue; }
      for (const auto& e : fs::directory_iterator(b.path(), ec)) {
        out.insert(e.path().string());
      }
    }
    return out;
  };
  auto engines_of = [&](const std::set<std::string>& before,
                        const std::set<std::string>& after) {
    std::set<std::string> found;
    for (const std::string& e : after) {
      if (before.count(e) != 0u) { continue; }
      std::error_code ec;
      for (auto it = fs::recursive_directory_iterator(e, ec);
           it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) { break; }
        const std::string n = it->path().filename().string();
        for (const char* k : {"main_ane", "main_bnns", "main_cpu",
                              "main_gpu"}) {
          if (n == k) { found.insert(k + 5); }
        }
      }
    }
    std::string s;
    for (const auto& f : found) { s += (s.empty() ? "" : "+") + f; }
    return s.empty() ? std::string("no new cache entry") : s;
  };

  struct Form { const char* name; bool tx, ty; };
  const Form forms[] = {{"A @ B",   false, false},
                        {"AT @ B",  true,  false},
                        {"A @ BT",  false, true},
                        {"AT @ BT", true,  true}};
  const int lens[] = {1, 16};
  std::printf("[ane_tsq] runtime inputs, %dx%d, bound as %s\n", D, D,
              use_const ? "host memory, B a CONSTANT"
              : use_pb ? "IOSurface pixel buffers" : "host memory");
  for (const Form& f : forms) {
    // CPU reference for one matmul with this form's transposes.
    std::vector<double> ref((std::size_t)D * D, 0.0);
    for (int i = 0; i < D; ++i) {
      for (int k = 0; k < D; ++k) {
        const double a = f.tx ? Af[(std::size_t)k * D + i]
                              : Af[(std::size_t)i * D + k];
        if (a == 0.0) { continue; }
        for (int j = 0; j < D; ++j) {
          const double b = f.ty ? Bf[(std::size_t)j * D + k]
                                : Bf[(std::size_t)k * D + j];
          ref[(std::size_t)i * D + j] += a * b;
        }
      }
    }
    double t_at[2] = {0.0, 0.0};
    for (int li = 0; li < 2; ++li) {
      const int L = lens[li];
      std::ostringstream o;
      o << vpipe::AneEmitter::mil_preamble() << "    func main<ios18>(tensor<"
        << "fp16, [" << D << ", " << D << "]> a";
      if (!use_const) {
        o << ", tensor<fp16, [" << D << ", " << D << "]> b";
      }
      o << ") {\n";
      if (use_const) {
        o << "            tensor<fp16, [" << D << ", " << D << "]> b = const()"
          << "[name = string(\"b\"), val = tensor<fp16, [" << D << ", " << D
          << "]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"),"
          << " offset = uint64(64)))];\n";
      }
      o
        << "            bool tx = const()[name = string(\"tx\"), val = bool("
        << (f.tx ? "true" : "false") << ")];\n"
        << "            bool ty = const()[name = string(\"ty\"), val = bool("
        << (f.ty ? "true" : "false") << ")];\n";
      std::string prev = "a";
      for (int i = 0; i < L; ++i) {
        const std::string h = (i == L - 1) ? "out" : "h" + std::to_string(i);
        o << "            tensor<fp16, [" << D << ", " << D << "]> " << h
          << " = matmul(transpose_x = tx, transpose_y = ty, x = " << prev
          << ", y = b)[name = string(\"" << h << "\")];\n";
        prev = h;
      }
      o << "        } -> (out);\n}";
      std::vector<vpipe::AneEmitter::Feature> ins{{"a", {D, D}}};
      if (!use_const) { ins.push_back({"b", {D, D}}); }
      std::vector<vpipe::AneBlobSlab> slabs;
      if (use_const) { slabs.push_back({B16.data(), B16.size() * 2}); }
      const std::vector<vpipe::AneEmitter::Feature> outs{{"out", {D, D}}};
      const std::string dir = root + "/vpipe-ane-tsq-" + std::to_string(run) +
                              "-" + std::to_string((int)(f.tx) * 2 + f.ty) +
                              "-L" + std::to_string(L) + ".mlmodelc";
      std::string err;
      ASSERT_TRUE(vpipe::AneEmitter::write_bundle(dir, o.str(), ins, outs,
                                                  slabs, &err));
      const auto before = cache_entries();
      const auto t0 = clk::now();
      auto m = mgr->load(dir, 3);
      const double compile_ms =
          std::chrono::duration<double, std::milli>(clk::now() - t0).count();
      const std::string engine = engines_of(before, cache_entries());
      ASSERT_TRUE(m != nullptr);
      if (m == nullptr) { continue; }

      std::vector<vpipe::CoreMLPredictInput> inputs(use_const ? 1 : 2);
      inputs[0].name = "a";
      if (!use_const) { inputs[1].name = "b"; }
      for (auto& in : inputs) {
        in.dtype = vpipe::CoreMLDType::F16;
        in.shape = {D, D};
      }
      if (use_pb) {
        inputs[0].pixel_buffer = pa;
        if (!use_const) { inputs[1].pixel_buffer = pbb; }
      } else {
        inputs[0].data = A16.data();
        if (!use_const) { inputs[1].data = B16.data(); }
      }
      std::vector<std::uint16_t> y((std::size_t)D * D, 0u);
      vpipe::CoreMLPredictOutput out;
      out.name = "out"; out.want = vpipe::CoreMLDType::F16;
      out.backing = y.data(); out.backing_elems = y.size();
      for (int k = 0; k < 3; ++k) { m->predict(inputs, {&out, 1}); }
      double best = 1e18;
      for (int k = 0; k < 20; ++k) {
        const auto p0 = clk::now();
        const bool ok = m->predict(inputs, {&out, 1});
        const double ms = std::chrono::duration<double, std::milli>(
            clk::now() - p0).count();
        if (ok && ms < best) { best = ms; }
      }
      t_at[li] = best;
      double rl = -1.0;
      if (L == 1) {
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
          _Float16 h;
          std::memcpy(&h, &y[i], 2);
          const double d = (double)h - ref[i];
          num += d * d;
          den += ref[i] * ref[i];
        }
        rl = std::sqrt(num / den);
        EXPECT_TRUE(rl < 1e-2);
      }
      std::printf("[ane_tsq]   %-8s L=%-2d compile %5.0f ms | predict best "
                  "%6.3f ms | engine %s%s\n", f.name, L, compile_ms, best,
                  engine.c_str(),
                  rl >= 0.0 ? (" | rel-L2 vs CPU reference " +
                               std::to_string(rl)).c_str() : "");
      m.reset();
      std::error_code ec;
      fs::remove_all(dir, ec);
    }
    const double marginal = (t_at[1] - t_at[0]) / 15.0;
    std::printf("[ane_tsq] %-8s marginal %.3f ms per matmul = %.2f TOPS "
                "(L=1 %.3f ms, L=16 %.3f ms)\n", f.name, marginal,
                marginal > 0.0 ? flops_one / (marginal / 1000.0) / 1e12 : 0.0,
                t_at[0], t_at[1]);
  }
  if (pa != nullptr) { CVPixelBufferRelease(pa); }
  if (pbb != nullptr) { CVPixelBufferRelease(pbb); }
}
#endif

#ifdef VPIPE_BUILD_APPLE_SILICON
// The runtime-weight module against the constant one, on the same
// weights -- the check that emitting the weights as INPUTS (sliced into
// the same tiles inside the graph) computes what baking them did. Small
// enough to need no model files, so it runs unconditionally. Also checks
// that slots of the wrong shape are refused rather than bound, since a
// mis-shaped input would load and compute garbage.
TEST(ane_module, runtime_module_matches_const)
{
  vpipe::Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!vpipe::genai::AneModule::emitter_usable(&sess)) { return; }
  using vpipe::genai::AneModule;
  using vpipe::genai::AneWeightSlots;
  using vpipe::genai::AneTiledWeight;

  vpipe::AneGraphSpec cs;
  cs.kind = vpipe::AneGraphSpec::Kind::SwiGluFfn;
  cs.M = 64; cs.K = 128; cs.N = 256; cs.bk = 64; cs.bn = 64;
  vpipe::AneGraphSpec rs = cs;
  rs.runtime_weights = true;
  ASSERT_TRUE(cs.valid() && rs.valid());

  std::mt19937 rng(0x2f2fu);
  std::uniform_real_distribution<float> d(-0.15f, 0.15f);
  auto mat = [&](int R, int C) {
    std::vector<std::uint16_t> v((std::size_t)R * C);
    for (auto& e : v) {
      const _Float16 h = (_Float16)d(rng);
      std::memcpy(&e, &h, 2);
    }
    return v;
  };
  const std::vector<std::uint16_t> g = mat(cs.K, cs.N);
  const std::vector<std::uint16_t> u = mat(cs.K, cs.N);
  const std::vector<std::uint16_t> dn = mat(cs.N, cs.K);

  const std::string tmp = std::filesystem::temp_directory_path().string();
  const AneTiledWeight tw[3] = {
      {g.data(), (std::size_t)cs.K, (std::size_t)cs.N, 64, 64},
      {u.data(), (std::size_t)cs.K, (std::size_t)cs.N, 64, 64},
      {dn.data(), (std::size_t)cs.N, (std::size_t)cs.K, 64, 64}};
  auto mconst = AneModule::build_emitted(&sess, cs, tw,
                                         tmp + "/ane-rt-vs-const-c.mlmodelc");
  auto mrt = AneModule::build_runtime(&sess, rs,
                                      tmp + "/ane-rt-vs-const-r.mlmodelc");
  auto slots = AneWeightSlots::create(rs);
  ASSERT_TRUE(mconst != nullptr && mrt != nullptr && slots != nullptr);
  if (mconst == nullptr || mrt == nullptr || slots == nullptr) { return; }
  ASSERT_TRUE(slots->count() == 3u && mrt->weight_inputs() == 3u);

  const std::vector<std::uint16_t>* src[3] = {&g, &u, &dn};
  for (std::size_t i = 0; i < 3; ++i) {
    const int rows = slots->rows(i), cols = slots->cols(i);
    ASSERT_TRUE(slots->fill(i, [&](void* base, std::size_t bpr) {
      auto* dst = static_cast<std::uint8_t*>(base);
      for (int r = 0; r < rows; ++r) {
        std::memcpy(dst + (std::size_t)r * bpr,
                    src[i]->data() + (std::size_t)r * (std::size_t)cols,
                    (std::size_t)cols * 2);
      }
    }));
  }

  const std::size_t n = (std::size_t)cs.M * cs.K;
  auto x = mc->make_shared_buffer(n * 2);
  auto yc = mc->make_shared_buffer(n * 2);
  auto yr = mc->make_shared_buffer(n * 2);
  {
    auto* xp = static_cast<std::uint16_t*>(x.contents());
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) {
      const _Float16 h = (_Float16)dx(rng);
      std::memcpy(&xp[i], &h, 2);
    }
  }
  ASSERT_TRUE(mconst->run(x, 0, yc, 0));
  ASSERT_TRUE(mrt->run(x, 0, yr, 0, *slots));
  double num = 0.0, den = 0.0;
  const auto* pc = static_cast<const std::uint16_t*>(yc.contents());
  const auto* pr = static_cast<const std::uint16_t*>(yr.contents());
  for (std::size_t i = 0; i < n; ++i) {
    _Float16 a, b;
    std::memcpy(&a, &pr[i], 2);
    std::memcpy(&b, &pc[i], 2);
    num += ((double)a - (double)b) * ((double)a - (double)b);
    den += (double)b * (double)b;
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : 1.0;
  std::printf("[ane_module] runtime-weight module vs constant: rel-L2 %.3e\n",
              rel);
  EXPECT_TRUE(den > 0.0 && rel < 1e-3);

  // ...and the [out,in] form, filled row for row from each matrix's
  // transpose: the layout a checkpoint stores, which the graph reads with
  // transpose_y.
  {
    vpipe::AneGraphSpec ro = rs;
    ro.weights_out_in = true;
    ASSERT_TRUE(ro.valid());
    vpipe::AneGraphSpec bad = cs;
    bad.weights_out_in = true;              // constants have no [out,in]
    EXPECT_FALSE(bad.valid());
    auto mro = AneModule::build_runtime(&sess, ro,
                                        tmp + "/ane-rt-vs-const-o.mlmodelc");
    auto so = AneWeightSlots::create(ro);
    ASSERT_TRUE(mro != nullptr && so != nullptr);
    if (mro != nullptr && so != nullptr) {
      for (std::size_t i = 0; i < 3; ++i) {
        const int rows = so->rows(i), cols = so->cols(i);
        ASSERT_TRUE(so->fill(i, [&](void* base, std::size_t bpr) {
          auto* dst = static_cast<std::uint8_t*>(base);
          for (int r = 0; r < rows; ++r) {
            for (int c2 = 0; c2 < cols; ++c2) {
              std::memcpy(dst + (std::size_t)r * bpr + (std::size_t)c2 * 2,
                          src[i]->data() + (std::size_t)c2 * rows + r, 2);
            }
          }
        }));
      }
      auto yo = mc->make_shared_buffer(n * 2);
      ASSERT_TRUE(mro->run(x, 0, yo, 0, *so));
      double num2 = 0.0, den2 = 0.0;
      const auto* po = static_cast<const std::uint16_t*>(yo.contents());
      for (std::size_t i = 0; i < n; ++i) {
        _Float16 a, b;
        std::memcpy(&a, &po[i], 2);
        std::memcpy(&b, &pc[i], 2);
        num2 += ((double)a - (double)b) * ((double)a - (double)b);
        den2 += (double)b * (double)b;
      }
      const double rel2 = den2 > 0.0 ? std::sqrt(num2 / den2) : 1.0;
      std::printf("[ane_module] runtime-weight [out,in] module vs constant: "
                  "rel-L2 %.3e\n", rel2);
      EXPECT_TRUE(den2 > 0.0 && rel2 < 1e-3);
    }
  }

  // Slots shaped for a different graph must be refused, not bound.
  vpipe::AneGraphSpec other = rs;
  other.N = 128;
  auto wrong = AneWeightSlots::create(other);
  ASSERT_TRUE(wrong != nullptr);
  if (wrong != nullptr) {
    EXPECT_FALSE(mrt->run(x, 0, yr, 0, *wrong));
  }
}
#endif


#ifdef VPIPE_BUILD_APPLE_SILICON
// HOW FAST CAN A BLOCK'S WEIGHTS BE STAGED?
//
// A runtime-weight feed-forward converts each block's gate/up/down --
// 576 MB for Krea-2 -- from the checkpoint's bf16 [out,in] into the
// module's fp16 inputs at the start of every block. The first version
// took ~500 ms a block, longer than the attention it runs under, which
// turned a 1.40x split into a 0.78x one.
//
// Two independent levers:
//   * CONVERSION: bf16 is the top half of an f32, so bf16 -> fp16 is a
//     pure function of 16 bits -- a 65536-entry table (131 KB, in L2)
//     instead of a float round-trip per element.
//   * ACCESS ORDER: an [in,out] input transposes the checkpoint, which
//     reads or writes against the grain; an [out,in] input (read with
//     transpose_y in the graph) copies row for row.
//
// Arms over one 193 MB matrix through the session's convert pool, best of
// 3, every output checked against the current implementation.
//
// VPIPE_ANE_STAGE_BENCH=1 to run.
TEST(ane_module, stage_conversion_speed)
{
  if (env_or_null_("VPIPE_ANE_STAGE_BENCH") == nullptr) { return; }
  vpipe::Session sess;
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return; }
  vpipe::AneWorker& w = mgr->ane_worker();
  using clk = std::chrono::steady_clock;
  const std::size_t OUT = 16384, IN = 6144;      // gate: [FF, HID]
  const std::size_t n = OUT * IN;

  // A bf16 source of plausible weights.
  std::vector<std::uint16_t> src(n);
  {
    std::mt19937 rng(0xbf16u);
    std::normal_distribution<float> d(0.0f, 0.02f);
    for (auto& e : src) {
      const float f = d(rng);
      std::uint32_t bits = 0;
      std::memcpy(&bits, &f, 4);
      e = (std::uint16_t)(bits >> 16);
    }
  }
  std::vector<std::uint16_t> lut(65536);
  for (std::uint32_t v = 0; v < 65536u; ++v) {
    const std::uint32_t bits = v << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, 4);
    const _Float16 h = (_Float16)f;
    std::memcpy(&lut[v], &h, 2);
  }
  const std::uint16_t* L = lut.data();
  const std::uint16_t* S = src.data();

  // Row-granular parallelism over an element count: each row belongs to
  // the slice holding its first element, so rows are processed once, and
  // the pool's inline threshold (in elements) is not tripped by passing a
  // small row count.
  auto for_rows = [&](std::size_t rows, std::size_t cols,
                      const std::function<void(std::size_t, std::size_t)>&
                          body) {
    w.parallel_for(rows * cols, [&](std::size_t lo, std::size_t hi) {
      const std::size_t r0 = (lo + cols - 1) / cols;
      const std::size_t r1 = std::min(rows, (hi + cols - 1) / cols);
      if (r0 < r1) { body(r0, r1); }
    });
  };

  std::vector<std::uint16_t> ref(n), out(n);
  struct Arm {
    const char* name;
    bool transposed;   // dst is [IN, OUT]
    std::function<void(std::uint16_t*)> run;
  };
  const Arm arms[] = {
      {"float, transpose, element split (current)", true,
       [&](std::uint16_t* D) {
         w.parallel_for(n, [&](std::size_t lo, std::size_t hi) {
           std::size_t r = lo / OUT, c = lo - r * OUT;
           for (std::size_t e = lo; e < hi; ++e) {
             const std::uint32_t bits = (std::uint32_t)S[c * IN + r] << 16;
             float f = 0.0f;
             std::memcpy(&f, &bits, 4);
             const _Float16 h = (_Float16)f;
             std::memcpy(&D[r * OUT + c], &h, 2);
             if (++c == OUT) { c = 0; ++r; }
           }
         });
       }},
      {"LUT, transpose, element split", true,
       [&](std::uint16_t* D) {
         w.parallel_for(n, [&](std::size_t lo, std::size_t hi) {
           std::size_t r = lo / OUT, c = lo - r * OUT;
           for (std::size_t e = lo; e < hi; ++e) {
             D[r * OUT + c] = L[S[c * IN + r]];
             if (++c == OUT) { c = 0; ++r; }
           }
         });
       }},
      {"LUT, transpose, 256-blocked", true,
       [&](std::uint16_t* D) {
         const std::size_t bs = 256;
         // Parallel over dst row BLOCKS; within one, sweep source-row
         // blocks so both sides stay in cache.
         for_rows((IN + bs - 1) / bs, 1, [&](std::size_t b0, std::size_t b1) {
           for (std::size_t rb = b0; rb < b1; ++rb) {
             const std::size_t r_end = std::min(IN, (rb + 1) * bs);
             for (std::size_t cb = 0; cb < OUT; cb += bs) {
               const std::size_t c_end = std::min(OUT, cb + bs);
               for (std::size_t r = rb * bs; r < r_end; ++r) {
                 for (std::size_t c = cb; c < c_end; ++c) {
                   D[r * OUT + c] = L[S[c * IN + r]];
                 }
               }
             }
           }
         });
       }},
      {"LUT, sequential ([out,in] input)", false,
       [&](std::uint16_t* D) {
         for_rows(OUT, IN, [&](std::size_t r0, std::size_t r1) {
           for (std::size_t r = r0; r < r1; ++r) {
             const std::uint16_t* s = S + r * IN;
             std::uint16_t* d = D + r * IN;
             for (std::size_t c = 0; c < IN; ++c) { d[c] = L[s[c]]; }
           }
         });
       }},
      {"float, sequential ([out,in] input)", false,
       [&](std::uint16_t* D) {
         for_rows(OUT, IN, [&](std::size_t r0, std::size_t r1) {
           for (std::size_t r = r0; r < r1; ++r) {
             for (std::size_t c = 0; c < IN; ++c) {
               const std::uint32_t bits = (std::uint32_t)S[r * IN + c] << 16;
               float f = 0.0f;
               std::memcpy(&f, &bits, 4);
               const _Float16 h = (_Float16)f;
               std::memcpy(&D[r * IN + c], &h, 2);
             }
           }
         });
       }},
  };
  arms[0].run(ref.data());   // the reference, [IN, OUT]
  for (const Arm& a : arms) {
    a.run(out.data());         // warm the pages
    double best = 1e18;
    for (int k = 0; k < 3; ++k) {
      const auto t0 = clk::now();
      a.run(out.data());
      const double ms =
          std::chrono::duration<double, std::milli>(clk::now() - t0).count();
      if (ms < best) { best = ms; }
    }
    // Check against the reference, transposing where the layout differs.
    std::size_t bad = 0;
    for (std::size_t r = 0; r < IN && bad == 0; ++r) {
      for (std::size_t c = 0; c < OUT; ++c) {
        const std::uint16_t got = a.transposed ? out[r * OUT + c]
                                               : out[c * IN + r];
        if (got != ref[r * OUT + c]) { ++bad; break; }
      }
    }
    std::printf("[ane_stage] %-44s %7.1f ms  %6.2f GB/s  %s\n", a.name, best,
                (double)(n * 2) / 1e9 / (best / 1000.0),
                bad == 0 ? "matches" : "MISMATCH");
    EXPECT_TRUE(bad == 0);
  }
}
#endif

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

double
footprint_mb_()
{
  task_vm_info_data_t vmi;
  mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vmi, &cnt) !=
      KERN_SUCCESS) {
    return -1.0;
  }
  return (double)vmi.phys_footprint / 1048576.0;
}

double
system_free_mb_()
{
  vm_statistics64_data_t v;
  mach_msg_type_number_t c = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        (host_info64_t)&v, &c) != KERN_SUCCESS) {
    return -1.0;
  }
  return (double)(v.free_count + v.speculative_count) * (double)vm_page_size /
         1048576.0;
}

}  // namespace

// WHAT THE RUNTIME-WEIGHT TIER ACTUALLY COSTS, term by term, as the
// production Krea-2 path builds it ([out,in] inputs, 1024 tiles). The
// plan books ane_runtime_bytes() = 2.45x the weights, from two terms
// measured separately: ~1.45x wired staging and 1x IOSurface slots. This
// checks both, in every accounting that could see them -- system wired,
// this process's physical footprint, and system free pages -- and whether
// the staging moves with the chunk's ROW count, which the formula ignores
// but VPIPE_KREA2_ANE_CHUNK now changes. It also checks that dropping the
// module and the slots gives the memory back.
//
// One process per row count, for attribution: VPIPE_ANE_RT_MEM=<rows>.
TEST(ane_module, runtime_memory_components)
{
  const char* m_env = env_or_null_("VPIPE_ANE_RT_MEM");
  if (m_env == nullptr) { return; }
  const int M = std::atoi(m_env);
  vpipe::Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || M <= 0) { return; }
  if (!vpipe::genai::AneModule::emitter_usable(&sess)) { return; }
  using vpipe::genai::AneModule;
  using vpipe::genai::AneWeightSlots;
  const int K = 6144, N = 16384;
  vpipe::AneGraphSpec spec;
  spec.kind = vpipe::AneGraphSpec::Kind::SwiGluFfn;
  spec.M = M; spec.K = K; spec.N = N; spec.bk = 1024; spec.bn = 1024;
  spec.runtime_weights = true;
  spec.weights_out_in = true;
  const double weights_mb = 3.0 * K * N * 2.0 / 1048576.0;

  struct Snap { double wired, fp, free; };
  auto snap = [] {
    return Snap{wired_mb_(), footprint_mb_(), system_free_mb_()};
  };
  auto line = [&](const char* what, const Snap& a, const Snap& b) {
    std::printf("[ane_mem] M=%d %-34s wired %+6.0f | footprint %+6.0f | "
                "system free %+6.0f MB\n", M, what, b.wired - a.wired,
                b.fp - a.fp, b.free - a.free);
  };

  const Snap s0 = snap();
  auto slots = AneWeightSlots::create(spec);
  ASSERT_TRUE(slots != nullptr);
  if (slots == nullptr) { return; }
  const Snap s1 = snap();
  line("slots created", s0, s1);
  // Touch every page, as staging does.
  for (std::size_t i = 0; i < slots->count(); ++i) {
    const int rows = slots->rows(i), cols = slots->cols(i);
    slots->fill(i, [&](void* base, std::size_t bpr) {
      auto* d = static_cast<std::uint8_t*>(base);
      for (int r = 0; r < rows; ++r) {
        std::memset(d + (std::size_t)r * bpr, 0x11, (std::size_t)cols * 2);
      }
    });
  }
  const Snap s2 = snap();
  line("slots filled (576 MB of weights)", s1, s2);

  const auto t0 = std::chrono::steady_clock::now();
  auto m = AneModule::build_runtime(
      &sess, spec,
      std::filesystem::temp_directory_path().string() +
          "/vpipe-ane-mem-" + std::to_string(M) + ".mlmodelc");
  const double build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  const Snap s3 = snap();
  line("module built", s2, s3);

  const std::size_t n = (std::size_t)M * K;
  auto x = mc->make_shared_buffer(n * 2);
  auto y = mc->make_shared_buffer(n * 2);
  std::memset(x.contents(), 0, n * 2);
  const Snap s4 = snap();
  ASSERT_TRUE(m->run(x, 0, y, 0, *slots));
  const Snap s5 = snap();
  line("first predict", s4, s5);
  for (int k = 0; k < 3; ++k) { m->run(x, 0, y, 0, *slots); }
  const Snap s6 = snap();
  line("three more predicts", s5, s6);
  std::printf("[ane_mem] M=%d build %.0f ms | TOTAL held: wired %+.0f, "
              "footprint %+.0f, system free %+.0f MB | weights %.0f MB, "
              "plan books %.0f MB\n", M, build_ms, s6.wired - s0.wired,
              s6.fp - s0.fp, s6.free - s0.free, weights_mb,
              weights_mb * 2.45);

  m.reset();
  const Snap s7 = snap();
  line("module dropped", s6, s7);
  slots.reset();
  const Snap s8 = snap();
  line("slots dropped", s7, s8);
  std::printf("[ane_mem] M=%d left behind after both dropped: wired %+.0f, "
              "footprint %+.0f MB\n", M, s8.wired - s0.wired, s8.fp - s0.fp);
}
#endif

#ifdef VPIPE_BUILD_APPLE_SILICON
// HOW FAST CAN A QUANTIZED BLOCK'S FEED-FORWARD BE STAGED?
//
// The ANE module is fp16 whatever the checkpoint, so a quantized block has
// to be dequantized into the IOSurface slots on the host at the start of
// the block -- the codes and group scales cannot be handed to the engine
// (the CoreML inputs have no 8-bit type, and MIL has no bitwise ops to
// unpack a 4-bit nibble in the graph). A dense block stages in ~44 ms; this
// asks what dequantizing costs, since it has to hide under the block's
// attention the same way.
//
// Layout as the GPU affine_dequant kernel reads it: u32 codes row-major,
// 4-bit = 8 nibbles a word ((w >> 4i) & 0xf), 8-bit = 4 bytes a word; one
// bf16 scale and bias per group; value = scale * q + bias in float,
// narrowed to fp16. Arms, through the session's convert pool, best of 3,
// bit-checked against a naive reference:
//   direct   scale * q + bias per value;
//   table    a per-group table of every value q can take (16 for 4-bit,
//            256 for 8-bit), then a lookup per value.
//
// VPIPE_ANE_STAGE_BENCH=1 to run.
TEST(ane_module, stage_dequant_speed)
{
  if (env_or_null_("VPIPE_ANE_STAGE_BENCH") == nullptr) { return; }
  vpipe::Session sess;
  auto* svc = sess.services();
  auto* mgr = (svc != nullptr) ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return; }
  vpipe::AneWorker& w = mgr->ane_worker();
  using clk = std::chrono::steady_clock;
  const std::size_t OUT = 16384, IN = 6144, G = 64;     // gate: [FF, HID]
  const std::size_t groups = IN / G;

  auto bf16_of = [](float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, 4);
    return (std::uint16_t)(bits >> 16);
  };
  auto float_of_bf16 = [](std::uint16_t v) {
    const std::uint32_t bits = (std::uint32_t)v << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, 4);
    return f;
  };
  auto for_rows = [&](std::size_t rows, std::size_t cols,
                      const std::function<void(std::size_t, std::size_t)>&
                          body) {
    w.parallel_for(rows * cols, [&](std::size_t lo, std::size_t hi) {
      const std::size_t r0 = (lo + cols - 1) / cols;
      const std::size_t r1 = std::min(rows, (hi + cols - 1) / cols);
      if (r0 < r1) { body(r0, r1); }
    });
  };

  for (int bits : {4, 8}) {
    const std::size_t per = 32 / (std::size_t)bits;       // values per word
    const std::size_t words = IN / per;
    const unsigned mask = (1u << bits) - 1u;
    std::vector<std::uint32_t> codes(OUT * words);
    std::vector<std::uint16_t> sc(OUT * groups), bi(OUT * groups);
    {
      std::mt19937 rng(0xdeaull + (unsigned)bits);
      for (auto& c : codes) { c = rng(); }
      std::uniform_real_distribution<float> ds(0.0005f, 0.004f);
      std::uniform_real_distribution<float> db(-0.03f, 0.0f);
      for (std::size_t i = 0; i < sc.size(); ++i) {
        sc[i] = bf16_of(ds(rng));
        bi[i] = bf16_of(db(rng));
      }
    }
    std::vector<std::uint16_t> ref(OUT * IN), out(OUT * IN);
    for (std::size_t r = 0; r < OUT; ++r) {
      for (std::size_t k = 0; k < IN; ++k) {
        const std::uint32_t word = codes[r * words + k / per];
        const unsigned q = (word >> (bits * (k % per))) & mask;
        const float s = float_of_bf16(sc[r * groups + k / G]);
        const float b = float_of_bf16(bi[r * groups + k / G]);
        const _Float16 h = (_Float16)(s * (float)q + b);
        std::memcpy(&ref[r * IN + k], &h, 2);
      }
    }
    const std::uint32_t* C = codes.data();
    const std::uint16_t* S = sc.data();
    const std::uint16_t* B = bi.data();
    struct Arm { const char* name; std::function<void(std::uint16_t*)> run; };
    const Arm arms[] = {
        {"direct", [&](std::uint16_t* D) {
           for_rows(OUT, IN, [&](std::size_t r0, std::size_t r1) {
             for (std::size_t r = r0; r < r1; ++r) {
               const std::uint32_t* cw = C + r * words;
               std::uint16_t* d = D + r * IN;
               for (std::size_t g = 0; g < groups; ++g) {
                 const float s = float_of_bf16(S[r * groups + g]);
                 const float b = float_of_bf16(B[r * groups + g]);
                 for (std::size_t k = g * G; k < (g + 1) * G; ++k) {
                   const unsigned q =
                       (cw[k / per] >> (bits * (k % per))) & mask;
                   const _Float16 h = (_Float16)(s * (float)q + b);
                   std::memcpy(&d[k], &h, 2);
                 }
               }
             }
           });
         }},
        {"per-group table", [&](std::uint16_t* D) {
           for_rows(OUT, IN, [&](std::size_t r0, std::size_t r1) {
             std::uint16_t tbl[256];
             const std::size_t nq = (std::size_t)mask + 1;
             for (std::size_t r = r0; r < r1; ++r) {
               const std::uint32_t* cw = C + r * words;
               std::uint16_t* d = D + r * IN;
               for (std::size_t g = 0; g < groups; ++g) {
                 const float s = float_of_bf16(S[r * groups + g]);
                 const float b = float_of_bf16(B[r * groups + g]);
                 for (std::size_t q = 0; q < nq; ++q) {
                   const _Float16 h = (_Float16)(s * (float)q + b);
                   std::memcpy(&tbl[q], &h, 2);
                 }
                 for (std::size_t wd = g * G / per; wd < (g + 1) * G / per;
                      ++wd) {
                   std::uint32_t word = cw[wd];
                   for (std::size_t i = 0; i < per; ++i) {
                     d[wd * per + i] = tbl[word & mask];
                     word >>= bits;
                   }
                 }
               }
             }
           });
         }},
    };
    for (const Arm& a : arms) {
      a.run(out.data());
      double best = 1e18;
      for (int k = 0; k < 3; ++k) {
        const auto t0 = clk::now();
        a.run(out.data());
        const double ms =
            std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (ms < best) { best = ms; }
      }
      const bool same =
          std::memcmp(out.data(), ref.data(), ref.size() * 2) == 0;
      std::printf("[ane_stage] w%d %-16s one 193 MB matrix %6.1f ms (%5.2f "
                  "GB/s out) -> a block's three ~%5.0f ms  %s\n", bits,
                  a.name, best, (double)(OUT * IN * 2) / 1e9 / (best / 1000.0),
                  best * 3.0, same ? "matches" : "MISMATCH");
      EXPECT_TRUE(same);
    }
  }
}
#endif

// The ANE feed-forward tier at an arbitrary shape, end to end through
// AneFeedForward: runtime weights staged from a bf16 [gate; up] fc1, and
// bf16 in and out -- what a family pays, which a baked-weight bench does not
// show. The ANE takes every whole chunk and the GPU side is left idle.
//
// Env: VPIPE_ANE_FFN_BENCH=1, and _HID (2048) _FFN (8192) _SEQ (1797)
// _CHUNK (1024) _TILE (1024) _ITERS (5). The defaults are the MiniMax-H3
// video-VAE decoder's feed-forward over one tile.
TEST(ane_module, ffn_tier_shape_bench)
{
  if (env_or_null_("VPIPE_ANE_FFN_BENCH") == nullptr) { return; }
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int H     = envi("VPIPE_ANE_FFN_BENCH_HID", 2048);
  const int F     = envi("VPIPE_ANE_FFN_BENCH_FFN", 8192);
  const int seq   = envi("VPIPE_ANE_FFN_BENCH_SEQ", 1797);
  const int chunk = envi("VPIPE_ANE_FFN_BENCH_CHUNK", 1024);
  const int tile  = envi("VPIPE_ANE_FFN_BENCH_TILE", 1024);
  const int iters = std::max(1, envi("VPIPE_ANE_FFN_BENCH_ITERS", 5));
  // _GELU=1: LTX-2.5's shape of feed-forward, two biased projections
  // around a tanh-approximated GELU, checked against the host.
  const bool gelu = envi("VPIPE_ANE_FFN_BENCH_GELU", 0) != 0;
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  using Clk = std::chrono::steady_clock;
  auto ms_since = [](Clk::time_point t) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t).count();
  };

  vpipe::genai::AneFeedForward::Options o;
  o.gelu    = gelu;
  o.biases  = gelu;
  o.session = &sess;
  o.mc      = mc;
  o.tag     = "ffn-bench";
  o.hidden  = H;
  o.ffn     = F;
  o.seq     = seq;
  o.rows    = 1.0f;
  o.chunk   = chunk;
  o.tile    = tile;
  const auto tb = Clk::now();
  auto ff = vpipe::genai::AneFeedForward::create(o);
  const double build_ms = ms_since(tb);
  ASSERT_TRUE(ff != nullptr);
  if (ff == nullptr) { return; }

  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto bf16 = [&](std::size_t n, float sc) {
    vpipe::metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const float f = nd(rng) * sc;
      std::uint32_t bits = 0;
      std::memcpy(&bits, &f, sizeof(bits));
      p[i] = (std::uint16_t)(bits >> 16);
    }
    return b;
  };
  const vpipe::metal_compute::SharedBuffer fc1 =
      bf16((std::size_t)2 * F * H, 1.0f / std::sqrt((float)H));
  const vpipe::metal_compute::SharedBuffer fc2 =
      bf16((std::size_t)H * F, 1.0f / std::sqrt((float)F));
  const vpipe::metal_compute::SharedBuffer bu = bf16((std::size_t)F, 0.1f);
  const vpipe::metal_compute::SharedBuffer bd = bf16((std::size_t)H, 0.1f);
  vpipe::genai::AneFfnSource g, u, d;
  g.w = &fc1;
  u.w = &fc1;
  u.offset = (std::size_t)F;
  d.w = &fc2;
  if (gelu) {
    u.offset = 0;
    u.bias = &bu;
    d.bias = &bd;
  }
  double stage_best = 0.0;
  for (int i = 0; i < 3; ++i) {
    const auto ts = Clk::now();
    if (gelu) {
      ff->stage(0, std::vector<vpipe::genai::AneFfnSource>{u, d}, 64);
    } else {
      ff->stage(0, g, u, d, 64);
    }
    const bool ok = ff->join_stage(0);
    const double ms = ms_since(ts);
    ASSERT_TRUE(ok);
    if (stage_best == 0.0 || ms < stage_best) { stage_best = ms; }
  }

  const vpipe::metal_compute::SharedBuffer x =
      bf16((std::size_t)seq * H, 0.5f);
  vpipe::metal_compute::SharedBuffer y =
      mc->make_shared_buffer((std::size_t)seq * H * 2);
  double best = 0.0;
  int a = 0;
  for (int i = 0; i <= iters; ++i) {
    const auto tr = Clk::now();
    a = ff->begin(x, y, seq);
    const bool ok = ff->finish(0, seq, 0.0, 0.0);
    const double ms = ms_since(tr);
    ASSERT_TRUE(ok && a > 0);
    if (i > 0 && (best == 0.0 || ms < best)) { best = ms; }
  }
  double worst = 0.0;
  if (gelu) {
    // Two ANE rows through the host: the first and the last.
    auto f32of = [](const vpipe::metal_compute::SharedBuffer& b,
                    std::size_t i) {
      const std::uint16_t v = static_cast<const std::uint16_t*>(
          b.contents())[i];
      const std::uint32_t bits = (std::uint32_t)v << 16;
      float f = 0.0f;
      std::memcpy(&f, &bits, sizeof(f));
      return f;
    };
    const double kc = std::sqrt(2.0 / M_PI);
    // The cubic coefficient. torch's `approximate="tanh"` (and the
    // tree's gelu_tanh kernels) is 0.044715; _GELU_CUBIC tries another,
    // to name which one a graph's gelu op implements.
    const char* cub = std::getenv("VPIPE_ANE_FFN_BENCH_GELU_CUBIC");
    const double c3 = cub != nullptr ? std::atof(cub) : 0.044715;
    std::vector<double> hh((std::size_t)F);
    for (const int r : {seq - a, seq - 1}) {
      const std::size_t xo = (std::size_t)r * H;
      for (int j = 0; j < F; ++j) {
        double acc = f32of(bu, (std::size_t)j);
        for (int i = 0; i < H; ++i) {
          acc += (double)f32of(fc1, (std::size_t)j * H + i) *
                 f32of(x, xo + i);
        }
        // A negative coefficient checks the EXACT (erf) form instead.
        hh[(std::size_t)j] =
            c3 < 0.0
                ? 0.5 * acc * (1.0 + std::erf(acc / std::sqrt(2.0)))
                : 0.5 * acc *
                      (1.0 + std::tanh(kc * (acc + c3 * acc * acc * acc)));
      }
      double num = 0.0, den = 0.0;
      for (int i = 0; i < H; ++i) {
        double acc = f32of(bd, (std::size_t)i);
        for (int j = 0; j < F; ++j) {
          acc += (double)f32of(fc2, (std::size_t)i * F + j) *
                 hh[(std::size_t)j];
        }
        const double e = (double)f32of(y, xo + i) - acc;
        num += e * e;
        den += acc * acc;
      }
      worst = std::max(worst, std::sqrt(num / std::max(den, 1e-30)));
    }
    // fp16 through two 16384-wide projections floors near 7e-3.
    EXPECT_TRUE(worst < 1e-2);
  }
  const double flop =
      2.0 * (double)a * (gelu ? 2.0 : 3.0) * (double)H * (double)F;
  std::printf("[ane_module] %s tier %dx%d, seq %d, chunk %d, tile %d: rel "
              "%.2e, build "
              "%.0f ms, stage %.1f ms, %d ANE rows in %.1f ms (%.2f ms/1k "
              "rows) = %.2f TOPS end to end\n", gelu ? "gelu" : "ffn",
              H, F, seq, chunk, tile, worst,
              build_ms, stage_best, a, best, best * 1000.0 / (double)a,
              best > 0.0 ? flop / best / 1e9 : 0.0);
}

// WHICH GELU the ANE's gelu(mode = TANH_APPROXIMATION) computes. Identity
// projections and zero biases make every output element gelu() of a known
// input, swept over [-6, 6], so the max error against each candidate form
// names the one it implements -- a random-weight feed-forward cannot, its
// fp16 floor (~7e-3 relative at 4096x16384) swamps the difference.
TEST(ane_module, gelu_graph_names_its_approximation)
{
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const int H = 1024, seq = 768, chunk = 256;
  vpipe::genai::AneFeedForward::Options o;
  o.session = &sess;
  o.mc      = mc;
  o.tag     = "gelu-probe";
  o.hidden  = H;
  o.ffn     = H;
  o.seq     = seq;
  o.rows    = 1.0f;
  o.chunk   = chunk;
  o.tile    = 1024;
  o.gelu    = true;
  o.biases  = true;
  auto ff = vpipe::genai::AneFeedForward::create(o);
  if (ff == nullptr) { return; }     // no ANE on this box
  auto put = [](vpipe::metal_compute::SharedBuffer& b, std::size_t i,
                float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    static_cast<std::uint16_t*>(b.contents())[i] =
        (std::uint16_t)(bits >> 16);
  };
  auto get = [](const vpipe::metal_compute::SharedBuffer& b, std::size_t i) {
    const std::uint32_t bits =
        (std::uint32_t)static_cast<const std::uint16_t*>(b.contents())[i]
        << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  };
  vpipe::metal_compute::SharedBuffer eye =
      mc->make_shared_buffer((std::size_t)H * H * 2);
  vpipe::metal_compute::SharedBuffer zero =
      mc->make_shared_buffer((std::size_t)H * 2);
  std::memset(eye.contents(), 0, eye.byte_size());
  std::memset(zero.contents(), 0, zero.byte_size());
  for (int i = 0; i < H; ++i) { put(eye, (std::size_t)i * H + i, 1.0f); }
  vpipe::genai::AneFfnSource u, d;
  u.w = &eye;
  u.bias = &zero;
  d.w = &eye;
  d.bias = &zero;
  ff->stage(0, std::vector<vpipe::genai::AneFfnSource>{u, d}, 64);
  ASSERT_TRUE(ff->join_stage(0));
  vpipe::metal_compute::SharedBuffer x =
      mc->make_shared_buffer((std::size_t)seq * H * 2);
  vpipe::metal_compute::SharedBuffer y =
      mc->make_shared_buffer((std::size_t)seq * H * 2);
  const std::size_t n = (std::size_t)seq * H;
  for (std::size_t i = 0; i < n; ++i) {
    // bf16 then fp16 on the way in: keep every value exact in both.
    const float v = std::round((-6.0f + 12.0f * (float)i / (float)n) * 256) /
                    256.0f;
    put(x, i, v);
  }
  const int a = ff->begin(x, y, seq);
  ASSERT_TRUE(ff->finish(0, seq, 0.0, 0.0) && a > 0);
  const double kc = std::sqrt(2.0 / M_PI);
  double e_erf = 0.0, e_t44 = 0.0, e_t58 = 0.0, v_erf = 0.0, v_t44 = 0.0;
  // And within |v| < 3, where nearly every activation lives.
  double c_erf = 0.0, c_t44 = 0.0;
  for (std::size_t i = (std::size_t)(seq - a) * H; i < n; ++i) {
    const double v = get(x, i), got = get(y, i);
    const double erf = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
    auto tanh_c = [&](double c) {
      return 0.5 * v * (1.0 + std::tanh(kc * (v + c * v * v * v)));
    };
    const double de = std::fabs(got - erf);
    const double dt = std::fabs(got - tanh_c(0.044715));
    if (de > e_erf) { e_erf = de; v_erf = v; }
    if (dt > e_t44) { e_t44 = dt; v_t44 = v; }
    if (std::fabs(v) < 3.0) {
      c_erf = std::max(c_erf, de);
      c_t44 = std::max(c_t44, dt);
    }
    e_t58 = std::max(e_t58, std::fabs(got - tanh_c(0.058512)));
  }
  std::printf("[ane_module] gelu op over [-6,6], %d rows: max |err| erf "
              "%.2e (at %.3f; %.2e within 3), tanh(0.044715) %.2e (at %.3f; "
              "%.2e within 3), tanh(0.058512) %.2e\n", a, e_erf, v_erf,
              c_erf, e_t44, v_t44, c_t44, e_t58);
  // MEASURED (M4 Pro and M5 Pro alike): ~1.5e-2 from erf AND from torch's
  // approximate="tanh" (0.044715), worst near |v| ~ 2.5, and the same when
  // the approximation is spelled out as elementwise ops -- so it is the
  // ANE's elementwise precision, not the mode. What is pinned is that
  // bound, and that the op is not the 0.058512 variant, which is worse.
  EXPECT_TRUE(e_t44 < 2e-2);
  EXPECT_TRUE(e_t44 < e_t58);
}

// ---- ane::Tier: the plugin-facing surface --------------------------------

#include "generative-models/shared/accel-settings.h"
#include "generative-models/shared/ane-tier.h"

#include <thread>

namespace {

vpipe::FlexData
tier_spec_(const std::string& kind, const std::string& act, int hidden,
           int inner, int seq, int chunk)
{
  namespace ane = vpipe::genai::ane;
  namespace acc = vpipe::genai::accel;
  vpipe::FlexData s = vpipe::FlexData::make_object();
  acc::set_text(&s, ane::kKind, kind);
  if (!act.empty()) { acc::set_text(&s, ane::kActivation, act); }
  acc::set_integer(&s, ane::kHidden, hidden);
  acc::set_integer(&s, ane::kInner, inner);
  acc::set_integer(&s, ane::kSeq, seq);
  acc::set_integer(&s, ane::kChunk, chunk);
  return s;
}

}  // namespace

// A spec the host cannot build is refused BY NAME, and claims nothing --
// the promise an old host makes a plugin that asks for a newer kind.
TEST(ane_tier, refuses_what_it_cannot_build)
{
  namespace ane = vpipe::genai::ane;
  namespace acc = vpipe::genai::accel;
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  std::string why;

  const auto attn = tier_spec_("attention", "", 64, 256, 768, 256);
  EXPECT_TRUE(ane::create(&sess, mc, attn, &why) == nullptr);
  EXPECT_TRUE(why.find("unknown kind 'attention'") != std::string::npos);
  EXPECT_TRUE(ane::runtime_bytes(attn) == 0u);

  why.clear();
  const auto relu = tier_spec_("ffn", "relu", 64, 256, 768, 256);
  EXPECT_TRUE(ane::create(&sess, mc, relu, &why) == nullptr);
  EXPECT_TRUE(why.find("unknown ffn activation 'relu'") != std::string::npos);

  why.clear();
  auto noseq = tier_spec_("ffn", "gelu_tanh", 64, 256, 768, 256);
  acc::set_integer(&noseq, ane::kSeq, 0);
  EXPECT_TRUE(ane::create(&sess, mc, noseq, &why) == nullptr);
  EXPECT_TRUE(!why.empty());

  // The byte counts are the machinery's own, per kind.
  EXPECT_TRUE(ane::runtime_bytes(
                  tier_spec_("ffn", "gelu_tanh", 64, 256, 768, 256)) ==
              vpipe::genai::AneFeedForward::gelu_runtime_bytes(64, 256, 256));
  EXPECT_TRUE(ane::runtime_bytes(tier_spec_("ffn", "", 64, 256, 768, 256)) ==
              vpipe::genai::AneFeedForward::runtime_bytes(64, 256, 768, 256));
}

// The surface is a translation, not a second implementation: the same
// weights through ane::Tier and through AneFeedForward give the SAME BYTES.
// And a binding the kind does not read is refused before anything runs.
TEST(ane_tier, gelu_ffn_matches_the_feed_forward_it_wraps)
{
  namespace ane = vpipe::genai::ane;
  namespace acc = vpipe::genai::accel;
  using vpipe::genai::AneFeedForward;
  using vpipe::genai::AneFfnSource;
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  constexpr int H = 256, F = 1024, seq = 768, chunk = 256;

  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto bf16 = [&](std::size_t n, float sc) {
    vpipe::metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const float f = nd(rng) * sc;
      std::uint32_t bits = 0;
      std::memcpy(&bits, &f, sizeof(bits));
      p[i] = (std::uint16_t)(bits >> 16);
    }
    return b;
  };
  const auto wu = bf16((std::size_t)F * H, 1.0f / std::sqrt((float)H));
  const auto wd = bf16((std::size_t)H * F, 1.0f / std::sqrt((float)F));
  const auto bu = bf16((std::size_t)F, 0.1f);
  const auto bd = bf16((std::size_t)H, 0.1f);
  const auto x  = bf16((std::size_t)seq * H, 0.5f);
  auto y1 = mc->make_shared_buffer((std::size_t)seq * H * 2);
  auto y2 = mc->make_shared_buffer((std::size_t)seq * H * 2);
  std::memset(y1.contents(), 0, y1.byte_size());
  std::memset(y2.contents(), 0, y2.byte_size());

  // The machinery, directly.
  AneFeedForward::Options o;
  o.session = &sess;
  o.mc      = mc;
  o.tag     = "tier-ref";
  o.hidden  = H;
  o.ffn     = F;
  o.seq     = seq;
  o.rows    = 1.0f;
  o.chunk   = chunk;
  o.gelu    = true;
  o.biases  = true;
  auto ff = AneFeedForward::create(o);
  if (ff == nullptr) { return; }                 // no ANE on this box
  AneFfnSource su, sd;
  su.w = &wu;
  su.bias = &bu;
  sd.w = &wd;
  sd.bias = &bd;
  ff->stage(0, std::vector<AneFfnSource>{su, sd}, 0);
  ASSERT_TRUE(ff->join_stage(0));
  const int a1 = ff->begin(x, y1, seq);
  ASSERT_TRUE(ff->finish(0, seq, 0.0, 0.0) && a1 > 0);

  // The same through the described surface.
  auto spec = tier_spec_("ffn", "gelu_tanh", H, F, seq, chunk);
  acc::set_flag(&spec, ane::kBiases, true);
  acc::set_real(&spec, ane::kRows, 1.0);
  std::string why;
  auto tier = ane::create(&sess, mc, spec, &why);
  ASSERT_TRUE(tier != nullptr);
  if (tier == nullptr) {
    std::printf("[ane_tier] create refused: %s\n", why.c_str());
    return;
  }
  const vpipe::FlexData params = vpipe::FlexData::make_object();

  // A misspelt name: refused, with the name in the reason.
  const ane::Binding typo[] = {{"up.weight", &wu}};
  EXPECT_FALSE(tier->stage(0, params, typo));
  const vpipe::FlexData inf = tier->info();
  EXPECT_TRUE(acc::text(&inf, ane::kInfoError).find("up.weight") !=
              std::string::npos);

  const std::string nuw = ane::key(ane::kProjUp, "w");
  const std::string nub = ane::key(ane::kProjUp, "bias");
  const std::string ndw = ane::key(ane::kProjDown, "w");
  const std::string ndb = ane::key(ane::kProjDown, "bias");
  const ane::Binding wb[] = {{nuw, &wu}, {nub, &bu}, {ndw, &wd}, {ndb, &bd}};
  ASSERT_TRUE(tier->stage(0, params, wb));
  ASSERT_TRUE(tier->join_stage(0));
  vpipe::FlexData args = vpipe::FlexData::make_object();
  acc::set_integer(&args, ane::kSeq, seq);
  const ane::Binding io[] = {{ane::kIoIn, &x}, {ane::kIoOut, &y2}};
  const int a2 = tier->begin(args, io);
  ASSERT_TRUE(tier->finish(0, vpipe::FlexData::make_object()) && a2 > 0);

  EXPECT_TRUE(a1 == a2);
  const std::size_t r0 = (std::size_t)(seq - a1) * H * 2;
  EXPECT_TRUE(std::memcmp(static_cast<const char*>(y1.contents()) + r0,
                          static_cast<const char*>(y2.contents()) + r0,
                          (std::size_t)a1 * H * 2) == 0);
}

// WHAT THE ANE TIER HOLDS, AND WHERE IT IS CHARGED. Weight slots are
// IOSurfaces, host rows are unwired SharedBuffers, and CoreML's compiled
// program is the driver's -- none of it goes through the wired pool. This
// snapshots the process footprint, its graphics ledger, and SYSTEM-WIDE
// wired memory across create / stage / predict / destroy at a real shape,
// so the unaccounted bytes are measured rather than assumed.
//
// Env: VPIPE_ANE_MEM_PROBE=1, and _HID (4096) _FFN (16384) _SEQ (8160)
// _CHUNK (2048) _GELU (1).
TEST(ane_tier, memory_it_holds_and_where)
{
  if (env_or_null_("VPIPE_ANE_MEM_PROBE") == nullptr) { return; }
  namespace ane = vpipe::genai::ane;
  namespace acc = vpipe::genai::accel;
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int H = envi("VPIPE_ANE_MEM_PROBE_HID", 4096);
  const int F = envi("VPIPE_ANE_MEM_PROBE_FFN", 16384);
  const int seq = envi("VPIPE_ANE_MEM_PROBE_SEQ", 8160);
  const int chunk = envi("VPIPE_ANE_MEM_PROBE_CHUNK", 2048);
  const bool gelu = envi("VPIPE_ANE_MEM_PROBE_GELU", 1) != 0;
  // _MATMUL: one projection _HID -> _FFN instead (e.g. FLUX.2-klein-9B's
  // single-block to_qkv_mlp_proj, 4096 -> 36864).
  const bool matmul = std::getenv("VPIPE_ANE_MEM_PROBE_MATMUL") != nullptr;
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  // Every input the caller owns is allocated BEFORE the first snapshot, so
  // the deltas are the tier's alone.
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto bf16 = [&](std::size_t n, float sc) {
    vpipe::metal_compute::SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const float f = nd(rng) * sc;
      std::uint32_t bits = 0;
      std::memcpy(&bits, &f, sizeof(bits));
      p[i] = (std::uint16_t)(bits >> 16);
    }
    return b;
  };
  const auto wg = bf16(gelu ? 1 : (std::size_t)F * H, 0.01f);
  const auto wu = bf16((std::size_t)F * H, 1.0f / std::sqrt((float)H));
  const auto wd = bf16((std::size_t)H * F, 1.0f / std::sqrt((float)F));
  const auto bu = bf16((std::size_t)F, 0.1f);
  const auto bd = bf16((std::size_t)H, 0.1f);
  const auto x  = bf16((std::size_t)seq * H, 0.5f);
  const auto ww = bf16(matmul ? (std::size_t)F * H : 1,
                       1.0f / std::sqrt((float)H));
  auto y = mc->make_shared_buffer((std::size_t)seq * (matmul ? F : H) * 2);

  struct Snap {
    const char* what;
    vpipe::metal_compute::MetalCompute::MemoryBudget b;
  };
  std::vector<Snap> snaps;
  auto snap = [&](const char* what) {
    snaps.push_back({what, mc->memory_budget()});
  };
  snap("before");

  auto spec = tier_spec_("ffn", gelu ? "gelu_tanh" : "swiglu", H, F, seq,
                         chunk);
  acc::set_flag(&spec, ane::kBiases, gelu && !matmul);
  acc::set_real(&spec, ane::kRows, 0.75);
  if (matmul) {
    acc::set_text(&spec, ane::kKind, std::string(ane::kKindMatmul));
    acc::set_integer(&spec, ane::kInWidth, H);
    acc::set_integer(&spec, ane::kOutWidth, F);
  }
  std::string why;
  auto tier = ane::create(&sess, mc, spec, &why);
  if (tier == nullptr) {
    std::printf("[ane_tier] mem probe: create refused: %s\n", why.c_str());
    return;
  }
  snap("created");

  const vpipe::FlexData params = vpipe::FlexData::make_object();
  const std::string ng = ane::key(ane::kProjGate, "w");
  const std::string nu = ane::key(ane::kProjUp, "w");
  const std::string nd_ = ane::key(ane::kProjDown, "w");
  const std::string nub = ane::key(ane::kProjUp, "bias");
  const std::string ndb = ane::key(ane::kProjDown, "bias");
  const std::string nw = ane::key(ane::kProjW, "w");
  std::vector<ane::Binding> wb = {{nu, &wu}, {nd_, &wd}};
  if (matmul) {
    wb = {{nw, &ww}};
  } else if (gelu) {
    wb.push_back({nub, &bu});
    wb.push_back({ndb, &bd});
  } else {
    wb.push_back({ng, &wg});
  }
  ASSERT_TRUE(tier->stage(0, params, wb) && tier->join_stage(0));
  snap("staged");

  vpipe::FlexData args = vpipe::FlexData::make_object();
  acc::set_integer(&args, ane::kSeq, seq);
  const ane::Binding io[] = {{ane::kIoIn, &x}, {ane::kIoOut, &y}};
  const int a = tier->begin(args, io);
  ASSERT_TRUE(tier->finish(0, vpipe::FlexData::make_object()) && a > 0);
  snap("1 predict");
  for (int i = 1; i < 8; ++i) {
    ASSERT_TRUE(tier->stage(i, params, wb) && tier->join_stage(i));
    (void)tier->begin(args, io);
    ASSERT_TRUE(tier->finish(i, vpipe::FlexData::make_object()));
  }
  snap("8 blocks");
  const vpipe::FlexData inf = tier->info();
  tier.reset();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  snap("destroyed");

  auto mb = [](std::size_t v) { return (long long)(v >> 20); };
  std::printf("[ane_tier] mem probe %s %dx%d, seq %d, chunk %d, %d ANE rows; "
              "tier estimate %lld MB (host rows %lld MB)\n",
              matmul ? "matmul" : gelu ? "gelu" : "swiglu", H, F, seq, chunk,
              a,
              mb(ane::runtime_bytes(spec)),
              (long long)(acc::integer(&inf, ane::kInfoHostBytes, 0) >> 20));
  std::printf("[ane_tier]   %-10s %10s %10s %10s %10s %10s\n", "step",
              "footprint", "graphics", "sys wired", "avail", "free");
  const auto& b0 = snaps.front().b;
  for (const Snap& s : snaps) {
    std::printf("[ane_tier]   %-10s %+10lld %+10lld %+10lld %+10lld %+10lld\n",
                s.what,
                mb(s.b.self_footprint) - mb(b0.self_footprint),
                mb(s.b.self_graphics) - mb(b0.self_graphics),
                mb(s.b.wired) - mb(b0.wired),
                mb(s.b.available_physical) - mb(b0.available_physical),
                mb(s.b.free_physical) - mb(b0.free_physical));
  }
}

// ONE runtime-weight matmul on the ANE at an arbitrary shape: y = x W^T
// with W [N,K] bound [out,in] through an IOSurface slot, fp16 in and out,
// chunked at M rows -- the cost a family would pay to hand a projection's
// rows to the ANE (e.g. MiniMax-H3's fused qkv, 5376 -> 21504). Reports
// predict alone and predict plus the bf16<->fp16 host conversions.
//
// Env: VPIPE_ANE_MM_BENCH=1, and _K (5376) _N (21504) _M (2048) _TILE
// (1024) _ITERS (5).
TEST(ane_module, matmul_tier_shape_bench)
{
  if (env_or_null_("VPIPE_ANE_MM_BENCH") == nullptr) { return; }
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int K     = envi("VPIPE_ANE_MM_BENCH_K", 5376);
  const int N     = envi("VPIPE_ANE_MM_BENCH_N", 21504);
  const int M     = envi("VPIPE_ANE_MM_BENCH_M", 2048);
  const int tile  = envi("VPIPE_ANE_MM_BENCH_TILE", 1024);
  const int iters = std::max(1, envi("VPIPE_ANE_MM_BENCH_ITERS", 5));
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  using Clk = std::chrono::steady_clock;
  auto ms_since = [](Clk::time_point t) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t).count();
  };

  vpipe::AneGraphSpec spec;
  spec.kind = vpipe::AneGraphSpec::Kind::Matmul;
  spec.M = M;
  spec.K = K;
  spec.N = N;
  spec.bk = tile;
  spec.bn = tile;
  spec.runtime_weights = true;
  spec.weights_out_in  = true;
  ASSERT_TRUE(spec.valid());
  if (!spec.valid()) { return; }
  char work[256];
  std::snprintf(work, sizeof(work), "%s/vpipe-ane-mm-bench-%dx%dx%d-t%d",
                std::filesystem::temp_directory_path().c_str(), M, K, N,
                tile);
  const auto tb = Clk::now();
  auto mod = vpipe::genai::AneModule::build_runtime(&sess, spec, work);
  auto slots = vpipe::genai::AneWeightSlots::create(spec);
  const double build_ms = ms_since(tb);
  ASSERT_TRUE(mod != nullptr && slots != nullptr);
  if (mod == nullptr || slots == nullptr) { return; }

  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const float ws = 1.0f / std::sqrt((float)K);
  ASSERT_TRUE(slots->fill(0, [&](void* base, std::size_t bpr) {
    auto* d = static_cast<std::uint8_t*>(base);
    for (int r = 0; r < slots->rows(0); ++r) {
      for (int c = 0; c < slots->cols(0); ++c) {
        const _Float16 hv = (_Float16)(nd(rng) * ws);
        std::memcpy(d + (std::size_t)r * bpr + (std::size_t)c * 2, &hv, 2);
      }
    }
  }));
  // bf16 host input and output, as a DiT holds them; fp16 on the module.
  const std::size_t in_n = (std::size_t)M * K, out_n = (std::size_t)M * N;
  std::vector<std::uint16_t> xb(in_n), yb(out_n);
  for (auto& v : xb) {
    const float f = nd(rng) * 0.5f;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, 4);
    v = (std::uint16_t)(bits >> 16);
  }
  vpipe::metal_compute::SharedBuffer x = mc->make_shared_buffer(in_n * 2);
  vpipe::metal_compute::SharedBuffer y = mc->make_shared_buffer(out_n * 2);
  ASSERT_TRUE(!x.empty() && !y.empty());
  if (x.empty() || y.empty()) { return; }

  double best_pred = 0.0, best_all = 0.0;
  for (int i = 0; i <= iters; ++i) {
    const auto t0 = Clk::now();
    {
      auto* d = static_cast<_Float16*>(x.contents());
      for (std::size_t k = 0; k < in_n; ++k) {
        const std::uint32_t bits = (std::uint32_t)xb[k] << 16;
        float f = 0.0f;
        std::memcpy(&f, &bits, 4);
        d[k] = (_Float16)f;
      }
    }
    const auto tp = Clk::now();
    const bool ok = mod->run(x, 0, y, 0, *slots);
    const double pred = ms_since(tp);
    ASSERT_TRUE(ok);
    {
      const auto* s = static_cast<const _Float16*>(y.contents());
      for (std::size_t k = 0; k < out_n; ++k) {
        const float f = (float)s[k];
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, 4);
        yb[k] = (std::uint16_t)(bits >> 16);
      }
    }
    const double all = ms_since(t0);
    if (i > 0) {
      if (best_pred == 0.0 || pred < best_pred) { best_pred = pred; }
      if (best_all == 0.0 || all < best_all) { best_all = all; }
    }
  }
  const double flop = 2.0 * (double)M * (double)K * (double)N;
  std::printf("[ane_module] matmul tier %d->%d, M %d, tile %d: build %.0f ms, "
              "predict %.1f ms (%.2f TOPS), with host conversion %.1f ms "
              "(%.2f TOPS, serial) = %.2f ms/1k rows\n", K, N, M, tile,
              build_ms, best_pred, flop / best_pred / 1e9, best_all,
              flop / best_all / 1e9, best_pred * 1000.0 / (double)M);
}

// WHERE fp16 OVERFLOW CAN ARISE IN THE ANE's FEED-FORWARD, op by op. Each
// graph isolates one op with identity projections, so the output is that
// op as the ANE computes it, over fp16 inputs swept log-spaced from 1e-2
// to 65504 on both signs:
//   silu     SwiGLU with Wg = I, Wu = 0 + bu = 1, Wd = I  -> silu(x)
//   silu*x   SwiGLU with Wg = Wu = Wd = I                 -> silu(x) * x
//   gelu     GELU  with Wu = Wd = I                        -> gelu_tanh(x)
// A non-finite output where the true value is finite and inside fp16 is an
// overflow INSIDE the op's implementation; one where the true value is
// itself past 65504 is the op's result not fitting. Env-gated
// (VPIPE_ANE_RANGE_PROBE=1): a characterisation, not a contract.
TEST(ane_module, activation_range_probe)
{
  if (std::getenv("VPIPE_ANE_RANGE_PROBE") == nullptr) { return; }
  vpipe::Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!vpipe::genai::AneModule::emitter_usable(&sess)) { return; }
  using vpipe::genai::AneModule;
  using vpipe::genai::AneWeightSlots;
  // ONE VALUE PER ROW. The down projection sums a whole row, so a single
  // inf in `hh` turns every column of its row into inf * 0 = NaN -- a
  // sweep that mixed magnitudes or signs within a row would charge that
  // spread to the op under test.
  // M, the row count the graph is compiled for, CHANGES the lowering of
  // the activations -- VPIPE_ANE_RANGE_M to compare (the tiers use 2048).
  const int M = [] {
    const char* e = std::getenv("VPIPE_ANE_RANGE_M");
    return e != nullptr ? std::max(16, std::atoi(e)) : 2048;
  }();
  const int D = [] {
    const char* e = std::getenv("VPIPE_ANE_RANGE_D");
    return e != nullptr ? std::max(64, std::atoi(e)) : 256;
  }();
  // VPIPE_ANE_RANGE_ONLY=<label>: run just that probe.
  const char* only = std::getenv("VPIPE_ANE_RANGE_ONLY");
  const std::size_t n = (std::size_t)M * D;
  std::vector<_Float16> xs(n);
  for (int r = 0; r < M; ++r) {
    const double t = (double)(r / 2) / (double)(M / 2 - 1);
    const double m = std::exp(std::log(1e-2) +
                              t * (std::log(65504.0) - std::log(1e-2)));
    for (int c = 0; c < D; ++c) {
      xs[(std::size_t)r * D + c] = (_Float16)((r % 2) ? -m : m);
    }
  }
  const double kc = std::sqrt(2.0 / M_PI);
  auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };
  auto gelu = [&](double v) {
    return 0.5 * v * (1.0 + std::tanh(kc * (v + 0.044715 * v * v * v)));
  };
  int tile_k = 0;   // K tiling of the graphs below (0 = none)
  // A slot's contents: `eye` * v on the diagonal, or v everywhere.
  struct Fill { bool eye; float v; };
  const Fill I{true, 1.0f}, Z{false, 0.0f}, O{false, 1.0f};
  auto probe = [&](const char* label, vpipe::AneGraphSpec::Kind kind,
                   std::vector<Fill> fills,
                   const std::function<double(double)>& ref) {
    if (only != nullptr && std::string(only) != label) { return; }
    vpipe::AneGraphSpec s;
    s.kind = kind;
    s.M = M; s.K = D; s.N = D;
    s.bk = tile_k;
    s.runtime_weights = true;
    s.biases = kind != vpipe::AneGraphSpec::Kind::Matmul;
    ASSERT_TRUE(s.valid());
    auto mod = AneModule::build_runtime(
        &sess, s, std::filesystem::temp_directory_path().string() +
                      "/ane-range-" + label + ".mlmodelc");
    auto slots = AneWeightSlots::create(s);
    ASSERT_TRUE(mod != nullptr && slots != nullptr &&
                slots->count() == fills.size());
    if (mod == nullptr || slots == nullptr) { return; }
    for (std::size_t i = 0; i < fills.size(); ++i) {
      const int rows = slots->rows(i), cols = slots->cols(i);
      ASSERT_TRUE(slots->fill(i, [&](void* base, std::size_t bpr) {
        for (int r = 0; r < rows; ++r) {
          auto* d = reinterpret_cast<_Float16*>(
              static_cast<std::uint8_t*>(base) + (std::size_t)r * bpr);
          for (int c = 0; c < cols; ++c) {
            d[c] = (!fills[i].eye || r == c) ? (_Float16)fills[i].v
                                             : (_Float16)0.0f;
          }
        }
      }));
    }
    auto x = mc->make_shared_buffer(n * 2);
    auto y = mc->make_shared_buffer(n * 2);
    std::memcpy(x.contents(), xs.data(), n * 2);
    ASSERT_TRUE(mod->run(x, 0, y, 0, *slots));
    const auto* yp = static_cast<const _Float16*>(y.contents());
    // What the op returns at a few magnitudes, both signs -- the tail a
    // relative error hides, where the true value is ~0.
    {
      std::string line;
      for (double want_x : {8.0, 12.0, 16.0, 24.0, 40.0, 100.0, 300.0,
                            1000.0}) {
        for (double sg : {1.0, -1.0}) {
          std::size_t best = 0;
          for (std::size_t i = 0; i < n; ++i) {
            if (std::fabs((double)xs[i] - sg * want_x) <
                std::fabs((double)xs[best] - sg * want_x)) {
              best = i;
            }
          }
          char buf[64];
          std::snprintf(buf, sizeof(buf), " f(%.4g)=%.4g", (double)xs[best],
                        (double)yp[best]);
          line += buf;
        }
      }
      std::printf("[ane_range] %-7s samples:%s\n", label, line.c_str());
    }
    // Per sign: the smallest |x| whose output is non-finite although the
    // true value fits fp16 (an INTERNAL overflow), the smallest whose true
    // value does not fit (the result overflowing), and the worst relative
    // error among finite outputs of |true| > 1e-2.
    for (int sign : {+1, -1}) {
      double first_internal = 0.0, first_result = 0.0, worst = 0.0,
             worst_at = 0.0, abs8 = 0.0;
      int internal = 0, result = 0;
      for (std::size_t i = 0; i < n; ++i) {
        const double v = (double)xs[i];
        if ((v > 0) != (sign > 0)) { continue; }
        const double want = ref(v);
        const double got = (double)yp[i];
        const bool fits = std::fabs(want) <= 65504.0;
        if (!std::isfinite(got)) {
          if (fits) {
            ++internal;
            if (first_internal == 0.0) { first_internal = std::fabs(v); }
          } else {
            ++result;
            if (first_result == 0.0) { first_result = std::fabs(v); }
          }
          continue;
        }
        if (fits && std::fabs(want) > 1e-2) {
          const double e = std::fabs(got - want) / std::fabs(want);
          if (e > worst) { worst = e; worst_at = v; }
        }
        if (std::fabs(v) <= 8.0) {
          abs8 = std::max(abs8, std::fabs(got - want));
        }
      }
      std::printf("[ane_range] %-7s x%c0: %5d non-finite with a result that "
                  "FITS (first at |x| = %.4g), %5d where the result does not "
                  "(first at |x| = %.4g); worst finite rel err %.2e at %.4g; "
                  "max abs err over |x| <= 8 %.2e\n",
                  label, sign > 0 ? '>' : '<', internal, first_internal,
                  result, first_result, worst, worst_at, abs8);
    }
  };
  using K = vpipe::AneGraphSpec::Kind;
  probe("silu", K::SwiGluFfn, {I, Z, I, Z, O, Z}, silu);
  probe("silu*x", K::SwiGluFfn, {I, I, I, Z, Z, Z},
        [&](double v) { return silu(v) * v; });
  probe("gelu", K::GeluFfn, {I, I, Z, Z}, gelu);
  // Which op overflows: the product (a constant 256 through the up bias),
  // the down matmul (silu, then Wd = 256 I), or a bare matmul.
  const Fill C{false, 256.0f}, E{true, 256.0f};
  probe("silu*256", K::SwiGluFfn, {I, Z, I, Z, C, Z},
        [&](double v) { return silu(v) * 256.0; });
  probe("silu@256I", K::SwiGluFfn, {I, Z, E, Z, O, Z},
        [&](double v) { return silu(v) * 256.0; });
  probe("gelu@256I", K::GeluFfn, {I, E, Z, Z},
        [&](double v) { return gelu(v) * 256.0; });
  probe("x@256I", K::Matmul, {E}, [](double v) { return v * 256.0; });
  // The same matmul split into two K tiles joined by an add: identical
  // arithmetic (the second tile contributes 0 to every column the first
  // covers and vice versa), but each tile's result is now CONSUMED by
  // another op instead of being the output.
  tile_k = 128;
  probe("x@256I/2t", K::Matmul, {E}, [](double v) { return v * 256.0; });
  tile_k = 0;
  // The fused silu(g) * u: sweep the constant u to see what its overflow
  // threshold on |g| depends on.
  for (float u : {2.0f, 8.0f, 32.0f, 64.0f, 128.0f, 512.0f, 2048.0f}) {
    const std::string l = "silu*" + std::to_string((int)u);
    probe(l.c_str(), K::SwiGluFfn, {I, Z, I, Z, Fill{false, u}, Z},
          [&](double v) { return silu(v) * (double)u; });
  }
}

// THE ANE MATMUL's OUTPUT CEILING. W has 256 on its (i, i) diagonal and
// the x rows are constant, so out = 256 * x on the first min(K, N)
// columns, swept across 16384..81920. MEASURED on an M4 Pro at M = 2048:
// K = N = 256 overflows where fp16 does (65536), but K >= 512 OR N >= 512
// overflows at 32768 -- half fp16's range. Every real projection is past
// 256, which is what the feed-forward's fp16 headroom retry is covering
// (a MiniMax-H3 down projection of 40142 came back inf). Env-gated
// (VPIPE_ANE_CEILING_PROBE=1): a characterisation, not a contract.
TEST(ane_module, matmul_ceiling_k_vs_n)
{
  if (std::getenv("VPIPE_ANE_CEILING_PROBE") == nullptr) { return; }
  vpipe::Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  using vpipe::genai::AneModule;
  using vpipe::genai::AneWeightSlots;
  const int M = 2048;
  for (const auto& [K, N] : std::vector<std::pair<int, int>>{
           {256, 256}, {512, 256}, {1024, 256}, {4096, 256},
           {256, 512}, {256, 1024}, {256, 4096}}) {
    vpipe::AneGraphSpec s;
    s.kind = vpipe::AneGraphSpec::Kind::Matmul;
    s.M = M; s.K = K; s.N = N;
    s.runtime_weights = true;
    auto mod = AneModule::build_runtime(
        &sess, s, std::filesystem::temp_directory_path().string() +
                      "/ane-ceil.mlmodelc");
    auto slots = AneWeightSlots::create(s);
    ASSERT_TRUE(mod != nullptr && slots != nullptr);
    if (mod == nullptr || slots == nullptr) { continue; }
    slots->fill(0, [&](void* base, std::size_t bpr) {
      for (int r = 0; r < slots->rows(0); ++r) {
        auto* d = reinterpret_cast<_Float16*>(
            static_cast<std::uint8_t*>(base) + (std::size_t)r * bpr);
        for (int c = 0; c < slots->cols(0); ++c) {
          d[c] = (r == c) ? (_Float16)256.0f : (_Float16)0.0f;
        }
      }
    });
    auto x = mc->make_shared_buffer((std::size_t)M * K * 2);
    auto y = mc->make_shared_buffer((std::size_t)M * N * 2);
    auto* xp = static_cast<_Float16*>(x.contents());
    for (int r = 0; r < M; ++r) {
      const double v = 64.0 + 256.0 * (double)r / (double)M;   // 64..320
      for (int c = 0; c < K; ++c) { xp[(std::size_t)r * K + c] = (_Float16)v; }
    }
    mod->run(x, 0, y, 0, *slots);
    const auto* yp = static_cast<const _Float16*>(y.contents());
    double first_bad = 0.0;
    for (int r = 0; r < M && first_bad == 0.0; ++r) {
      if (!std::isfinite((float)yp[(std::size_t)r * N])) {
        first_bad = 256.0 * (double)xp[(std::size_t)r * K];
      }
    }
    std::printf("[ane_ceiling] K=%4d N=%4d: first non-finite at a true "
                "result of %.0f\n", K, N, first_bad);
  }
}
