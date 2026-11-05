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
#include "generative-models/shared/fp8-layout.h"
#include "generative-models/shared/fp8.h"
#include "generative-models/shared/i8-gemm.h"
#include "apple-silicon/coreml/ane-emitter.h"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "interfaces/session-services-intf.h"
#include "common/flex-data.h"
#include "common/perf-flops.h"
#include "common/session.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#endif

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>
#include <functional>
#include <fstream>
#include <cstring>
#include <iterator>
#include <map>
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

// HOW MUCH OF THE NEURAL ENGINE does one runtime-weight module reach? The
// tier drives ONE module from ONE thread (AneWorker), and an M5 Ultra is
// two dies, each with its own engine -- so whether one predict runs on
// both, or a second module on a second thread adds the other, is a
// property of the machine. This runs N instances of the tier's own graph
// (MiniMax-H3's feed-forward by default), each its own CoreML model, from
// N threads at once, alone and beside a GPU GEMM load, and prints the
// rate each engine achieved. Every instance holds the same weights and
// input, so their outputs must agree bit for bit.
//
//   VPIPE_ANE_PROBE=1               run it
//   VPIPE_ANE_PROBE_HID / _FFN      the graph's widths (5376 / 14336)
//   VPIPE_ANE_PROBE_CHUNK / _TILE   rows a predict, in-graph tile
//   VPIPE_ANE_PROBE_MATMUL=1        one projection, hidden -> ffn
//   VPIPE_ANE_PROBE_MAX             the most instances (4)
//   VPIPE_ANE_PROBE_SECS            seconds an arm (3)
//   VPIPE_ANE_PROBE_ROUNDS          rounds, every other one reversed (2)
//   VPIPE_ANE_PROBE_NO_GPU=1        skip the arms with a GPU load
//   VPIPE_ANE_PROBE_GPU             the load: bf16 (a dense GEMM), or
//                                   token / token_plain -- one H3
//                                   block's four projections on the
//                                   per-token int8 tier, rotated or not
TEST(ane_module, concurrent_engines_probe)
{
  if (env_or_null_("VPIPE_ANE_PROBE") == nullptr) { return; }
  namespace mcx = vpipe::metal_compute;
  using vpipe::genai::AneModule;
  using vpipe::genai::AneWeightSlots;
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int  H      = envi("VPIPE_ANE_PROBE_HID", 5376);
  const int  F      = envi("VPIPE_ANE_PROBE_FFN", 14336);
  const int  chunk  = envi("VPIPE_ANE_PROBE_CHUNK", 2048);
  const int  tile   = envi("VPIPE_ANE_PROBE_TILE", 1024);
  const bool mm     = envi("VPIPE_ANE_PROBE_MATMUL", 0) != 0;
  const int  nmax   = std::max(1, envi("VPIPE_ANE_PROBE_MAX", 4));
  const double secs = std::max(1, envi("VPIPE_ANE_PROBE_SECS", 3));
  const int  rounds = std::max(1, envi("VPIPE_ANE_PROBE_ROUNDS", 2));
  const bool gpu_arms = envi("VPIPE_ANE_PROBE_NO_GPU", 0) == 0;
  vpipe::Session sess;
  mcx::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!AneModule::emitter_usable(&sess)) { return; }
  using Clk = std::chrono::steady_clock;
  auto sec_since = [](Clk::time_point t) {
    return std::chrono::duration<double>(Clk::now() - t).count();
  };

  vpipe::AneGraphSpec spec;
  spec.kind = mm ? vpipe::AneGraphSpec::Kind::Matmul
                 : vpipe::AneGraphSpec::Kind::SwiGluFfn;
  spec.M  = chunk;
  spec.K  = H;
  spec.N  = F;
  spec.bk = tile;
  spec.bn = tile;
  spec.runtime_weights = true;
  spec.weights_out_in  = true;      // as AneFeedForward builds it
  ASSERT_TRUE(spec.valid());
  if (!spec.valid()) { return; }
  const double flop = 2.0 * chunk * (double)H * (double)F * (mm ? 1.0 : 3.0);

  // A cheap generator: the bytes only have to be the same everywhere.
  std::uint32_t lcg = 0x9e3779b9u;
  auto uni = [&]() {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)(lcg >> 8) * (2.0f / 16777216.0f) - 1.0f;
  };
  auto f16 = [](float v) {
    const _Float16 h = (_Float16)v;
    std::uint16_t w = 0;
    std::memcpy(&w, &h, 2);
    return w;
  };

  struct Inst {
    std::unique_ptr<AneModule>      m;
    std::unique_ptr<AneWeightSlots> w;
    mcx::SharedBuffer x, y;
    double build_ms = 0.0, first_ms = 0.0;
  };
  std::vector<Inst> inst((std::size_t)nmax);
  std::vector<std::vector<std::uint16_t>> wsrc;   // slot k's rows
  std::vector<std::uint16_t> xsrc;
  const std::string tmp = std::filesystem::temp_directory_path().string();
  const long long run =
      std::chrono::duration_cast<std::chrono::microseconds>(
          Clk::now().time_since_epoch())
          .count();
  for (int i = 0; i < nmax; ++i) {
    Inst& s = inst[(std::size_t)i];
    // A path of its own, so the manager loads ANOTHER model rather than
    // handing back the first (it caches by path); the compiled program
    // is cached on the graph's content, so only the first compiles.
    const std::string dir = tmp + "/vpipe-ane-probe-" +
                            std::to_string(run) + "-" + std::to_string(i) +
                            ".mlmodelc";
    const auto t0 = Clk::now();
    s.m = AneModule::build_runtime(&sess, spec, dir);
    s.build_ms = sec_since(t0) * 1e3;
    s.w = AneWeightSlots::create(spec);
    ASSERT_TRUE(s.m != nullptr && s.w != nullptr);
    if (s.m == nullptr || s.w == nullptr) { return; }
    if (wsrc.empty()) {
      for (std::size_t k = 0; k < s.w->count(); ++k) {
        const std::size_t rows = (std::size_t)s.w->rows(k);
        const std::size_t cols = (std::size_t)s.w->cols(k);
        const float a = 1.0f / std::sqrt((float)cols);   // [out, in]
        std::vector<std::uint16_t> v(rows * cols);
        for (auto& e : v) { e = f16(uni() * a); }
        wsrc.push_back(std::move(v));
      }
      xsrc.resize(s.m->input_elems());
      for (auto& e : xsrc) { e = f16(uni()); }
    }
    for (std::size_t k = 0; k < s.w->count(); ++k) {
      const std::size_t rows = (std::size_t)s.w->rows(k);
      const std::size_t cols = (std::size_t)s.w->cols(k);
      const std::uint16_t* src = wsrc[k].data();
      const bool ok = s.w->fill(k, [&](void* base, std::size_t bpr) {
        for (std::size_t r = 0; r < rows; ++r) {
          std::memcpy(static_cast<std::uint8_t*>(base) + r * bpr,
                      src + r * cols, cols * 2);
        }
      });
      ASSERT_TRUE(ok);
    }
    s.x = mc->make_shared_buffer(s.m->input_elems() * 2);
    s.y = mc->make_shared_buffer(s.m->output_elems() * 2);
    std::memcpy(s.x.contents(), xsrc.data(), xsrc.size() * 2);
    const auto t1 = Clk::now();
    const bool ok = s.m->run(s.x, 0, s.y, 0, *s.w);
    s.first_ms = sec_since(t1) * 1e3;
    ASSERT_TRUE(ok);
    if (!ok) { return; }
    std::printf("[ane_probe] instance %d: built in %.0f ms, first predict "
                "%.1f ms\n", i, s.build_ms, s.first_ms);
  }
  std::printf("[ane_probe] %s %dx%d, %d rows a predict, tile %d: %.1f "
              "GFLOP a predict, %zu MB of weights an instance\n",
              mm ? "matmul" : "swiglu ffn", H, F, chunk, tile, flop / 1e9,
              (std::size_t)inst[0].w->bytes() >> 20);

  // The GPU load: a bf16 GEMM, on matrix cores where there are some.
  const bool mx = mc->supports_matrix_cores();
  const int GM = 8192, GN = 14336, GK = 5376;
  const char* gmode_env = env_or_null_("VPIPE_ANE_PROBE_GPU");
  const std::string gmode = gmode_env != nullptr ? gmode_env : "bf16";
  const bool tok = gmode == "token" || gmode == "token_plain";
  // ...or the per-token int8 tier: q|k|v, out, fc1 and fc2 of a
  // MiniMax-H3 block at GM rows, each weight held in the token layout and
  // each GEMM quantizing its own activation (gemm_fast), as the DiT's
  // route runs them.
  using I8 = vpipe::genai::I8GemmContext;
  I8 i8(mc, tok, /*bf16=*/true);
  struct Proj {
    int N = 0, K = 0, gc = 0;
    mcx::SharedBuffer codes, scales, x, y;
  };
  std::vector<Proj> projs;
  double unit_flop = 2.0 * 2.0 * GM * (double)GN * GK;   // two GEMMs
  if (tok) {
    ASSERT_TRUE(i8.enabled());
    if (!i8.enabled()) { return; }
    i8.set_tier(I8::Tier::kToken);
    i8.set_token_rotation(gmode == "token");
    const int shapes[4][2] = {
        {21504, 5376}, {5376, 7168}, {28672, 5376}, {5376, 14336}};
    unit_flop = 0.0;
    mcx::CommandStream st = mc->make_command_stream();
    std::vector<mcx::SharedBuffer> dense;
    {
      mcx::ComputeEncoder enc = st.begin_compute();
      for (const auto& s : shapes) {
        Proj p;
        p.N  = s[0];
        p.K  = s[1];
        p.gc = i8.fast_group(p.K);
        ASSERT_TRUE(p.gc == I8::kTokenGroup &&
                    i8.fast_accepts(GM, p.N, p.K));
        const std::size_t nw = (std::size_t)p.N * p.K;
        mcx::SharedBuffer w = mc->make_shared_buffer(nw * 2);
        auto* wp = static_cast<std::uint16_t*>(w.contents());
        const float a = 1.0f / std::sqrt((float)p.K);
        for (std::size_t i = 0; i < nw; ++i) {
          const float f = uni() * a;
          std::uint32_t bits = 0;
          std::memcpy(&bits, &f, 4);
          wp[i] = (std::uint16_t)(bits >> 16);
        }
        p.codes  = mc->make_shared_buffer(nw);
        p.scales =
            mc->make_shared_buffer(I8::fast_scale_bytes(p.N, p.K, p.gc));
        ASSERT_TRUE(i8.encode_quant_fast(enc, w, p.codes, p.scales, p.gc,
                                         p.N, p.K));
        dense.push_back(std::move(w));
        const std::size_t nx = (std::size_t)GM * p.K;
        p.x = mc->make_shared_buffer(nx * 2);
        auto* xp = static_cast<std::uint16_t*>(p.x.contents());
        for (std::size_t i = 0; i < nx; ++i) {
          const float f = uni();
          std::uint32_t bits = 0;
          std::memcpy(&bits, &f, 4);
          xp[i] = (std::uint16_t)(bits >> 16);
        }
        p.y = mc->make_shared_buffer((std::size_t)GM * p.N * 2);
        unit_flop += 2.0 * GM * (double)p.N * p.K;
        projs.push_back(std::move(p));
      }
    }
    ASSERT_TRUE(st.commit().wait_ok());
    std::printf("[ane_probe] GPU load: per-token int8 (%s, %s), one H3 "
                "block of 4 projections at %d rows = %.2f TFLOP a "
                "unit\n", i8.token_rotated() ? "rotated" : "plain",
                i8.token_kernel().c_str(), GM, unit_flop / 1e12);
  } else {
    std::printf("[ane_probe] GPU load: bf16 %s GEMM %dx%dx%d, two a "
                "unit\n", mx ? "matrix-core" : "steel", GM, GN, GK);
  }
  mcx::ComputeLibrary glib =
      mc->load_library(mx ? "dense_gemm_mma_bf16" : "dense_gemm_bf16");
  mcx::ComputeFunction gfn = glib.function(
      mx ? "dense_gemm_mma_t_n128x256_f16" : "dense_gemm_t_bm64bn64_f16");
  ASSERT_TRUE(!gpu_arms || gfn.valid());
  mcx::SharedBuffer gx = mc->make_shared_buffer((std::size_t)GM * GK * 2);
  mcx::SharedBuffer gw = mc->make_shared_buffer((std::size_t)GN * GK * 2);
  mcx::SharedBuffer gy = mc->make_shared_buffer((std::size_t)GM * GN * 2);
  {
    auto fill = [](mcx::SharedBuffer& b, std::size_t n) {
      auto* p = static_cast<std::uint16_t*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) {
        p[i] = (std::uint16_t)(0x3c80u + (i % 7));   // ~1/64, bf16
      }
    };
    fill(gx, (std::size_t)GM * GK);
    fill(gw, (std::size_t)GN * GK);
  }
  auto gemm = [&](mcx::ComputeEncoder& enc) {
    enc.set_function(gfn);
    enc.set_buffer(0, gx);
    enc.set_buffer(1, gw);
    enc.set_buffer(2, gw);
    enc.set_buffer(3, gy);
    enc.set_constant(4, GK);
    enc.set_constant(5, GN);
    enc.set_constant(6, GM);
    enc.set_constant(7, 0);
    if (mx) {
      enc.dispatch({(unsigned)(((GN + 255) / 256) * 256),
                    (unsigned)((GM + 127) / 128), 1}, {256, 1, 1});
    } else {
      enc.dispatch({(unsigned)(((GN + 63) / 64) * 32),
                    (unsigned)(((GM + 63) / 64) * 2), 2}, {32, 2, 2});
    }
  };
  std::atomic<bool> gstop{false};
  std::atomic<long> gdone{0};
  std::atomic<bool> gbad{false};
  std::thread gthr;
  const mcx::SharedBuffer no_bias{};
  auto gpu_unit = [&](mcx::ComputeEncoder& enc) {
    if (!tok) {
      gemm(enc);
      gemm(enc);
      return true;
    }
    bool ok = true;
    for (const Proj& p : projs) {
      ok = i8.gemm_fast(enc, p.x, 0, p.codes, p.scales, p.gc, no_bias, p.y,
                        0, GM, p.N, p.K) && ok;
    }
    return ok;
  };
  auto gpu_start = [&]() {
    gstop = false;
    gthr = std::thread([&]() {
      mcx::CommandStream st = mc->make_command_stream();
      while (!gstop.load()) {
        bool ok = false;
        {
          mcx::ComputeEncoder enc = st.begin_compute();
          ok = gpu_unit(enc);
        }
        if (!st.commit().wait_ok() || !ok) { gbad = true; }
        ++gdone;
      }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  };
  auto gpu_stop = [&]() {
    gstop = true;
    if (gthr.joinable()) { gthr.join(); }
  };

  // n threads predicting until the deadline, on n instances or (shared)
  // all on the first. Returns predicts a second.
  auto ane = [&](int n, bool shared, double* ms_each) {
    std::vector<long> cnt((std::size_t)n, 0);
    std::atomic<int>  ready{0};
    std::atomic<bool> go{false};
    std::atomic<bool> bad{false};
    Clk::time_point end;
    std::vector<std::thread> th;
    for (int t = 0; t < n; ++t) {
      th.emplace_back([&, t]() {
        Inst& s = inst[shared ? 0 : (std::size_t)t];
        ++ready;
        while (!go.load()) { std::this_thread::yield(); }
        while (Clk::now() < end) {
          if (!s.m->run(s.x, 0, s.y, 0, *s.w)) { bad = true; }
          ++cnt[(std::size_t)t];
        }
      });
    }
    while (ready.load() < n) { std::this_thread::yield(); }
    const auto t0 = Clk::now();
    end = t0 + std::chrono::milliseconds((long)(secs * 1000.0));
    go = true;
    for (auto& x : th) { x.join(); }
    const double wall = sec_since(t0);
    ASSERT_TRUE(!bad.load());
    long total = 0;
    for (long c : cnt) { total += c; }
    *ms_each = total > 0 ? wall * 1e3 * n / (double)total : 0.0;
    return (double)total / wall;
  };

  struct Arm { int n; bool shared; bool gpu; };
  std::vector<Arm> arms;
  for (int n = 1; n <= nmax; n = (n < 2 ? 2 : n * 2)) {
    arms.push_back({n, false, false});
  }
  if (nmax >= 2) { arms.push_back({2, true, false}); }
  if (gpu_arms && gfn.valid()) {
    arms.push_back({0, false, true});
    for (int n = 1; n <= nmax; n = (n < 2 ? 2 : n * 2)) {
      arms.push_back({n, false, true});
    }
  }
  for (int r = 0; r < rounds; ++r) {
    std::vector<Arm> order = arms;
    if (r % 2 == 1) { std::reverse(order.begin(), order.end()); }
    for (const Arm& a : order) {
      if (a.gpu) { gpu_start(); }
      const long g0 = gdone.load();
      const auto tg = Clk::now();
      double per_s = 0.0, ms_each = 0.0;
      if (a.n > 0) {
        per_s = ane(a.n, a.shared, &ms_each);
      } else {
        std::this_thread::sleep_for(
            std::chrono::milliseconds((long)(secs * 1000.0)));
      }
      const double gs = sec_since(tg);
      const long g1 = gdone.load();
      if (a.gpu) { gpu_stop(); }
      char what[64];
      std::snprintf(what, sizeof(what), "%d %s x %d thr%s", a.n,
                    a.shared ? "shared inst" : "inst", a.n,
                    a.gpu ? " + GPU" : "");
      if (a.n == 0) { std::snprintf(what, sizeof(what), "GPU alone"); }
      std::printf("[ane_probe] round %d  %-26s ANE %6.2f TFLOP/s "
                  "(%5.1f predicts/s, %6.1f ms a predict a thread)  GPU "
                  "%6.1f TFLOP/s\n", r, what, per_s * flop / 1e12, per_s,
                  ms_each,
                  a.gpu ? (double)(g1 - g0) * unit_flop / gs / 1e12 : 0.0);
      std::fflush(stdout);
    }
  }
  ASSERT_TRUE(!gbad.load());

  // Every instance computed the same thing.
  int differ = 0;
  for (int i = 1; i < nmax; ++i) {
    if (std::memcmp(inst[(std::size_t)i].y.contents(), inst[0].y.contents(),
                    inst[0].y.byte_size()) != 0) {
      ++differ;
    }
  }
  std::size_t nonfinite = 0;
  {
    const auto* p = static_cast<const _Float16*>(inst[0].y.contents());
    for (std::size_t i = 0; i < inst[0].m->output_elems(); ++i) {
      if (!std::isfinite((float)p[i])) { ++nonfinite; }
    }
  }
  std::printf("[ane_probe] outputs: %d of %d instances differ from the "
              "first, %zu non-finite\n", differ, nmax - 1, nonfinite);
  EXPECT_TRUE(differ == 0);
  EXPECT_TRUE(nonfinite == 0);
}

// WHAT THE NEURAL ENGINE RUNS WELL, OP BY OP. One projection stated three
// ways -- a matmul over [rows, K], a 1x1 convolution over a 64x64 grid in
// NCHW, and the same convolution fed NHWC, transposed in and out inside
// the graph (the layout a DiT's token-major rows already have) -- and a
// 3x3 convolution in both orders. Each with fp16 weights; with int8
// weights dequantized by the graph (constexpr_blockwise_shift_scale per
// output channel, the form coremltools' linear_quantize_weights writes:
// W8A16); and with those plus every op's input quantized at a CONSTANT
// per-tensor scale (quantize -> dequantize: W8A8 -- a scale taken per
// token at run time is not expressible). One op and a chain of L, so the
// marginal op's rate is read apart from the predict's I/O. Every graph's
// output is checked on sampled pixels against a double-precision
// restatement of the same quantized arithmetic, and the bundles are kept
// so CoreML's compute plan can say where each op ran.
//
//   VPIPE_ANE_OPS_PROBE=1     run it
//   VPIPE_ANE_OPS_DIR         where the bundles go (kept)
//   VPIPE_ANE_OPS_ITERS       timed predicts an arm (12)
//   VPIPE_ANE_OPS_ONLY        only the arms whose name contains this
TEST(ane_module, ops_probe)
{
  if (env_or_null_("VPIPE_ANE_OPS_PROBE") == nullptr) { return; }
  vpipe::Session sess;
  auto* svc = sess.services();
  auto* mgr = svc != nullptr ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return; }
  namespace fs = std::filesystem;
  using Clk = std::chrono::steady_clock;
  auto ms_since = [](Clk::time_point t) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t).count();
  };
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const char* od = env_or_null_("VPIPE_ANE_OPS_DIR");
  const std::string root =
      od != nullptr ? std::string(od)
                    : (fs::temp_directory_path() / "vpipe-ane-ops").string();
  std::error_code ec;
  fs::create_directories(root, ec);
  const int iters = std::max(3, envi("VPIPE_ANE_OPS_ITERS", 12));
  const char* only = env_or_null_("VPIPE_ANE_OPS_ONLY");

  struct Arm {
    std::string op;      // mm | c1 | c3
    bool        nhwc;
    std::string q;       // f16 | w8 | w8a8
    int K, N, H, W, L;
  };
  std::vector<Arm> arms;
  for (int C : {512, 1024, 2048, 4096}) {
    for (int L : {1, 8}) {
      for (const char* q : {"f16", "w8", "w8a8"}) {
        arms.push_back({"mm", false, q, C, C, 64, 64, L});
        arms.push_back({"c1", false, q, C, C, 64, 64, L});
        arms.push_back({"c1", true, q, C, C, 64, 64, L});
      }
    }
  }
  for (int C : {256, 512}) {
    for (int L : {1, 4}) {
      for (const char* q : {"f16", "w8", "w8a8"}) {
        arms.push_back({"c3", false, q, C, C, 64, 64, L});
        arms.push_back({"c3", true, q, C, C, 64, 64, L});
      }
    }
  }
  // Half of a MiniMax-H3 fc1, untiled: far past the weight buffer.
  for (const char* q : {"f16", "w8"}) {
    arms.push_back({"mm", false, q, 5376, 14336, 64, 64, 1});
    arms.push_back({"c1", false, q, 5376, 14336, 64, 64, 1});
    arms.push_back({"c1", true, q, 5376, 14336, 64, 64, 1});
  }

  std::uint32_t lcg = 0x2545f491u;
  auto uni = [&]() {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)(lcg >> 8) * (2.0f / 16777216.0f) - 1.0f;
  };
  auto h16 = [](float v) { return (float)(_Float16)v; };
  auto bits16 = [](float v) {
    const _Float16 h = (_Float16)v;
    std::uint16_t w = 0;
    std::memcpy(&w, &h, 2);
    return w;
  };
  auto dims = [](std::initializer_list<long long> d) {
    std::string s = "[";
    bool first = true;
    for (long long v : d) {
      s += (first ? "" : ", ") + std::to_string(v);
      first = false;
    }
    return s + "]";
  };
  auto blob = [](const std::string& ty, const std::string& d,
                 std::size_t off) {
    return "tensor<" + ty + ", " + d + ">(BLOBFILE(path = string(\"" +
           "@model_path/weights/weight.bin\"), offset = uint64(" +
           std::to_string(off) + ")))";
  };
  // W8A8's activation step, as stored: fp16.
  const float a_step = h16(3.0f / 127.0f);
  auto qd = [&](double v) {
    const double q = std::clamp(std::nearbyint(v / a_step), -128.0, 127.0);
    return q * a_step;
  };

  std::map<std::string, double> one_op_ms;    // L = 1, by arm key
  for (const Arm& a : arms) {
    const bool conv = a.op != "mm";
    const int kk = a.op == "c3" ? 3 : 1;
    const int H = a.H, W = a.W, M = H * W, K = a.K, N = a.N, L = a.L;
    const bool w8 = a.q != "f16", a8 = a.q == "w8a8";
    const std::string key = a.op +
                            (conv ? (a.nhwc ? "-nhwc" : "-nchw") : "") +
                            "-" + a.q + "-" + std::to_string(K) + "x" +
                            std::to_string(N);
    const std::string name = key + "-L" + std::to_string(L);
    if (only != nullptr && name.find(only) == std::string::npos) {
      continue;
    }
    if (L > 1 && K != N) { continue; }

    // The weights as the reference reads them: fp16-rounded, or the int8
    // codes times their fp16 per-channel step.
    const std::size_t per = (std::size_t)K * N * kk * kk;
    std::vector<std::vector<float>> wv((std::size_t)L);
    std::vector<std::vector<unsigned char>> slab;
    std::vector<std::uint32_t> code;   // the blob's dtype: 1 fp16, 4 int8
    for (int l = 0; l < L; ++l) {
      std::vector<float> w(per);
      const float amp = std::sqrt(3.0f / (float)(K * kk * kk));
      for (auto& e : w) { e = uni() * amp; }
      // mm's weight is [K, N] (column n), a conv's [N, K, kk, kk].
      auto chan = [&](std::size_t i) {
        return conv ? (int)(i / ((std::size_t)K * kk * kk))
                    : (int)(i % (std::size_t)N);
      };
      if (!w8) {
        std::vector<unsigned char> b(per * 2);
        for (std::size_t i = 0; i < per; ++i) {
          const std::uint16_t v = bits16(w[i]);
          std::memcpy(b.data() + i * 2, &v, 2);
          w[i] = h16(w[i]);
        }
        slab.push_back(std::move(b));
        code.push_back(1);
      } else {
        std::vector<float> s((std::size_t)N, 0.0f);
        for (std::size_t i = 0; i < per; ++i) {
          float& m = s[(std::size_t)chan(i)];
          m = std::max(m, std::fabs(w[i]));
        }
        for (auto& e : s) { e = h16(std::max(e, 1e-6f) / 127.0f); }
        std::vector<unsigned char> qb(per), sb((std::size_t)N * 2);
        for (std::size_t i = 0; i < per; ++i) {
          const float st = s[(std::size_t)chan(i)];
          const int q = (int)std::clamp(std::nearbyint(w[i] / st), -127.0f,
                                        127.0f);
          qb[i] = (unsigned char)(std::int8_t)q;
          w[i] = (float)q * st;
        }
        for (int n = 0; n < N; ++n) {
          const std::uint16_t v = bits16(s[(std::size_t)n]);
          std::memcpy(sb.data() + (std::size_t)n * 2, &v, 2);
        }
        slab.push_back(std::move(qb));
        code.push_back(4);
        slab.push_back(std::move(sb));
        code.push_back(1);
      }
      wv[(std::size_t)l] = std::move(w);
    }
    std::vector<std::size_t> sizes;
    for (const auto& b : slab) { sizes.push_back(b.size()); }
    const std::vector<std::size_t> off =
        vpipe::AneEmitter::blob_offsets(sizes);

    // The graph.
    const std::string in_d = !conv  ? dims({M, K})
                           : a.nhwc ? dims({1, H, W, K})
                                    : dims({1, K, H, W});
    const std::string out_d = !conv  ? dims({M, N})
                            : a.nhwc ? dims({1, H, W, N})
                                     : dims({1, N, H, W});
    std::ostringstream o;
    o << vpipe::AneEmitter::mil_preamble() << "    func main<ios18>("
      << "tensor<fp16, " << in_d << "> x) {\n";
    auto line = [&](const std::string& s) {
      o << "            " << s << ";\n";
    };
    auto cst = [&](const std::string& ty, const std::string& nm,
                   const std::string& val) {
      line(ty + " " + nm + " = const()[name = string(\"" + nm +
           "\"), val = " + val + "]");
    };
    if (!conv) {
      cst("bool", "ftx", "bool(false)");
      cst("bool", "fty", "bool(false)");
    } else {
      const std::string p = std::to_string(kk / 2);
      cst("tensor<int32, [2]>", "cst", "tensor<int32, [2]>([1, 1])");
      cst("tensor<int32, [2]>", "cdl", "tensor<int32, [2]>([1, 1])");
      cst("int32", "cgr", "int32(1)");
      cst("string", "cpt", "string(\"custom\")");
      cst("tensor<int32, [4]>", "cpd",
          "tensor<int32, [4]>([" + p + ", " + p + ", " + p + ", " + p +
              "])");
    }
    if (a8) {
      char sv[32];
      std::snprintf(sv, sizeof(sv), "fp16(%.9g)", (double)a_step);
      cst("fp16", "asc", sv);
      cst("string", "adt", "string(\"int8\")");
    }
    std::string cur = "x";
    if (conv && a.nhwc) {
      cst("tensor<int32, [4]>", "pin", "tensor<int32, [4]>([0, 3, 1, 2])");
      line("tensor<fp16, " + dims({1, K, H, W}) +
           "> xt = transpose(perm = pin, x = x)[name = string(\"xt\")]");
      cur = "xt";
    }
    const std::string act_d = !conv ? dims({M, K}) : dims({1, K, H, W});
    const std::string y_d = !conv ? dims({M, N}) : dims({1, N, H, W});
    const std::string w_d = !conv ? dims({K, N}) : dims({N, K, kk, kk});
    const std::string s_d = !conv ? dims({1, N}) : dims({N, 1, 1, 1});
    std::size_t si = 0;
    for (int l = 0; l < L; ++l) {
      const std::string ls = std::to_string(l);
      const std::string wn = "w" + ls;
      if (!w8) {
        cst("tensor<fp16, " + w_d + ">", wn, blob("fp16", w_d, off[si]));
        si += 1;
      } else {
        line("tensor<fp16, " + w_d + "> " + wn +
             " = constexpr_blockwise_shift_scale(data = " +
             blob("int8", w_d, off[si]) + ", scale = " +
             blob("fp16", s_d, off[si + 1]) + ")[name = string(\"" + wn +
             "\")]");
        si += 2;
      }
      std::string in = cur;
      if (a8) {
        line("tensor<int8, " + act_d + "> q" + ls + " = quantize(input = " +
             in + ", output_dtype = adt, scale = asc)[name = string(\"q" +
             ls + "\")]");
        line("tensor<fp16, " + act_d + "> d" + ls + " = dequantize(input = q" +
             ls + ", scale = asc)[name = string(\"d" + ls + "\")]");
        in = "d" + ls;
      }
      const std::string y = "y" + ls;
      if (!conv) {
        line("tensor<fp16, " + y_d + "> " + y +
             " = matmul(transpose_x = ftx, transpose_y = fty, x = " + in +
             ", y = " + wn + ")[name = string(\"" + y + "\")]");
      } else {
        line("tensor<fp16, " + y_d + "> " + y +
             " = conv(dilations = cdl, groups = cgr, pad = cpd, pad_type = "
             "cpt, strides = cst, weight = " + wn + ", x = " + in +
             ")[name = string(\"" + y + "\")]");
      }
      cur = y;
    }
    if (conv && a.nhwc) {
      cst("tensor<int32, [4]>", "pout",
          "tensor<int32, [4]>([0, 2, 3, 1])");
      line("tensor<fp16, " + out_d + "> yo = transpose(perm = pout, x = " +
           cur + ")[name = string(\"yo\")]");
      cur = "yo";
    }
    o << "        } -> (" << cur << ");\n}";

    auto shape_of = [&](bool in) {
      std::vector<std::int64_t> s;
      const int C = in ? K : N;
      if (!conv) {
        s = {M, C};
      } else if (a.nhwc) {
        s = {1, H, W, C};
      } else {
        s = {1, C, H, W};
      }
      return s;
    };
    const vpipe::AneEmitter::Feature fin{"x", shape_of(true)};
    const vpipe::AneEmitter::Feature fout{cur, shape_of(false)};
    std::vector<vpipe::AneBlobSlab> bs;
    for (const auto& b : slab) { bs.push_back({b.data(), b.size()}); }
    const std::string dir = root + "/" + name + ".mlmodelc";
    std::string err;
    if (!vpipe::AneEmitter::write_bundle(dir, o.str(), {&fin, 1}, {&fout, 1},
                                         bs, &err)) {
      std::printf("[ane_ops] %-30s NOT WRITTEN: %s\n", name.c_str(),
                  err.c_str());
      continue;
    }
    // write_bundle labels every slab fp16; the int8 ones say so.
    {
      std::fstream f(dir + "/weights/weight.bin",
                     std::ios::in | std::ios::out | std::ios::binary);
      for (std::size_t i = 0; i < code.size(); ++i) {
        if (code[i] == 1) { continue; }
        f.seekp((std::streamoff)(off[i] + 4));
        f.write(reinterpret_cast<const char*>(&code[i]), 4);
      }
    }
    slab.clear();
    const auto t0 = Clk::now();
    auto m = mgr->load(dir, 3);
    const double compile_ms = ms_since(t0);
    if (m == nullptr || !m->valid()) {
      std::printf("[ane_ops] %-30s DID NOT LOAD\n", name.c_str());
      continue;
    }

    // Input: token-major values, laid out as the graph declares them.
    std::vector<float> xt((std::size_t)M * K);
    for (auto& e : xt) { e = h16(uni()); }
    std::vector<std::uint16_t> xb(xt.size()), yb((std::size_t)M * N);
    for (int p = 0; p < M; ++p) {
      for (int c = 0; c < K; ++c) {
        const std::size_t d = conv && !a.nhwc
                                  ? (std::size_t)c * M + p
                                  : (std::size_t)p * K + c;
        xb[d] = bits16(xt[(std::size_t)p * K + c]);
      }
    }
    vpipe::CoreMLPredictInput in;
    in.name  = "x";
    in.data  = xb.data();
    in.dtype = vpipe::CoreMLDType::F16;
    in.shape = shape_of(true);
    vpipe::CoreMLPredictOutput out;
    out.name          = cur;
    out.want          = vpipe::CoreMLDType::F16;
    out.backing       = yb.data();
    out.backing_elems = yb.size();
    const auto t1 = Clk::now();
    bool ok = m->predict({&in, 1}, {&out, 1});
    const double first_ms = ms_since(t1);
    std::vector<double> t;
    for (int i = 0; i < iters && ok; ++i) {
      const auto ti = Clk::now();
      ok = m->predict({&in, 1}, {&out, 1});
      t.push_back(ms_since(ti));
    }
    if (!ok || t.empty()) {
      std::printf("[ane_ops] %-30s PREDICT FAILED\n", name.c_str());
      ASSERT_TRUE(false);
      continue;
    }
    std::sort(t.begin(), t.end());
    const double med = t[t.size() / 2];

    // The reference at four pixels: corners, an edge, the middle.
    double num = 0.0, den = 0.0;
    const int samples[4][2] = {{0, 0}, {0, W - 1}, {H / 2, W / 2 + 7},
                               {H - 1, W - 1}};
    for (const auto& sp : samples) {
      const int h0 = sp[0], w0 = sp[1];
      // Values of the current layer's input over the window it needs.
      std::map<int, std::vector<double>> curv;
      const int R = kk == 3 ? L : 0;
      for (int dh = -R; dh <= R; ++dh) {
        for (int dw = -R; dw <= R; ++dw) {
          const int h = h0 + dh, w = w0 + dw;
          if (h < 0 || h >= H || w < 0 || w >= W) { continue; }
          std::vector<double> v((std::size_t)K);
          for (int c = 0; c < K; ++c) {
            v[(std::size_t)c] = xt[(std::size_t)(h * W + w) * K + c];
          }
          curv[h * W + w] = std::move(v);
        }
      }
      for (int l = 0; l < L; ++l) {
        if (a8) {
          for (auto& kv : curv) {
            for (double& e : kv.second) { e = qd(e); }
          }
        }
        const std::vector<float>& wl = wv[(std::size_t)l];
        const int r = kk == 3 ? L - 1 - l : 0;
        std::map<int, std::vector<double>> nxt;
        for (int dh = -r; dh <= r; ++dh) {
          for (int dw = -r; dw <= r; ++dw) {
            const int h = h0 + dh, w = w0 + dw;
            if (h < 0 || h >= H || w < 0 || w >= W) { continue; }
            std::vector<double> y((std::size_t)N, 0.0);
            for (int ky = 0; ky < kk; ++ky) {
              for (int kx = 0; kx < kk; ++kx) {
                const int hh = h + ky - kk / 2, ww = w + kx - kk / 2;
                if (hh < 0 || hh >= H || ww < 0 || ww >= W) { continue; }
                const std::vector<double>& v = curv.at(hh * W + ww);
                // Each layout's weights read in order: mm's [K, N] by
                // row, a conv's [N, K, kk, kk] by filter.
                if (!conv) {
                  for (int c = 0; c < K; ++c) {
                    const float* wr = wl.data() + (std::size_t)c * N;
                    const double vc = v[(std::size_t)c];
                    for (int n = 0; n < N; ++n) {
                      y[(std::size_t)n] += (double)wr[n] * vc;
                    }
                  }
                  continue;
                }
                for (int n = 0; n < N; ++n) {
                  double acc = 0.0;
                  for (int c = 0; c < K; ++c) {
                    const std::size_t wi =
                        (((std::size_t)n * K + c) * kk + ky) * kk + kx;
                    acc += (double)wl[wi] * v[(std::size_t)c];
                  }
                  y[(std::size_t)n] += acc;
                }
              }
            }
            nxt[h * W + w] = std::move(y);
          }
        }
        curv = std::move(nxt);
      }
      const std::vector<double>& ref = curv.at(h0 * W + w0);
      const int p = h0 * W + w0;
      for (int n = 0; n < N; ++n) {
        const std::size_t d = conv && !a.nhwc
                                  ? (std::size_t)n * M + p
                                  : (std::size_t)p * N + n;
        _Float16 hv;
        std::memcpy(&hv, &yb[d], 2);
        const double e = (double)hv - ref[(std::size_t)n];
        num += e * e;
        den += ref[(std::size_t)n] * ref[(std::size_t)n];
      }
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const double op_flop = 2.0 * M * (double)K * N * kk * kk;
    std::printf("[ane_ops] %-30s compile %7.0f ms, first %7.1f ms, "
                "median %8.2f ms = %6.2f TOP/s, rel %.1e", name.c_str(),
                compile_ms, first_ms, med, L * op_flop / med / 1e9, rel);
    if (L == 1) {
      one_op_ms[key] = med;
    } else if (one_op_ms.count(key) != 0) {
      const double marg = (med - one_op_ms[key]) / (L - 1);
      std::printf(", marginal op %.2f ms = %.2f TOP/s", marg,
                  marg > 0.0 ? op_flop / marg / 1e9 : 0.0);
    }
    std::printf("\n");
    std::fflush(stdout);
    EXPECT_TRUE(rel < (a8 ? 5e-2 : 2e-2));
  }
}

// INT8 WITHOUT BAKING. macOS 26 gave CoreML int8 arrays
// (MLMultiArrayDataTypeInt8, which an IOSurface-backed OneComponent8
// pixel buffer can carry), so a projection's int8 codes can be a model
// INPUT -- one compiled program for every block, as the runtime-weight
// tier's fp16 slots are -- dequantized inside the graph. What a graph
// cannot take at run time is a quantizer's SCALE (quantize / dequantize
// want constants), so every scale that is not known when the graph is
// compiled multiplies the product's OUTPUT instead: a weight's per
// output channel, and an activation's per token. Constant power-of-two
// scales (1/128) keep the int8 products inside fp16. A chain of L
// projections y = x W^T, W [N, K] as a checkpoint stores it, each layer
// its own weight input, in eight forms:
//
//   f16-rt          fp16 weights as inputs (the tier's form)
//   w8-cast-rt      int8 weight inputs, cast to fp16, x s_w at the output
//   w8-dq-rt        the same through dequantize (scale 1/128)
//   w8a8-static-rt  + every input quantized at a constant per-tensor step
//   w8a8-dyn-rt     + every input quantized PER TOKEN, the scale computed
//                   in the graph (max |x| a row / 127), x s_a at the
//                   output -- the GPU's token tier's arithmetic
//   w8a8-in-rt      the first layer's input given as int8 codes and per-
//                   token scales (what the GPU's token quantizer writes),
//                   the later layers as w8a8-dyn-rt
//   w8a8-dyn1-rt    w8a8-dyn-rt with each activation dequantized at the
//   w8a8-in1-rt     step it was quantized with (1; the 1/128 on the
//                   weight side) -- the M5 compiler takes only such pairs
//   w8a8-dynr-rt    w8a8-dyn1-rt dividing by a row's scale as a multiply
//                   by its reciprocal
//   f16-baked       fp16 weights as constants
//   w8a8-baked      int8 constants, constant per-tensor activation step
//
// Each graph's output is checked on sampled rows against a double
// restatement of the same arithmetic; the bundles stay under
// VPIPE_ANE_RTQ_DIR for CoreML's compute plan.
//
//   VPIPE_ANE_RTQ_PROBE=1   run it
//   VPIPE_ANE_RTQ_DIR       where the bundles go (kept)
//   VPIPE_ANE_RTQ_ITERS     timed predicts an arm (10)
//   VPIPE_ANE_RTQ_ONLY      only the arms whose name contains this
TEST(ane_module, runtime_int8_probe)
{
  if (env_or_null_("VPIPE_ANE_RTQ_PROBE") == nullptr) { return; }
  vpipe::Session sess;
  auto* svc = sess.services();
  auto* mgr = svc != nullptr ? svc->coreml_model_manager() : nullptr;
  if (mgr == nullptr) { return; }
  namespace fs = std::filesystem;
  using Clk = std::chrono::steady_clock;
  auto ms_since = [](Clk::time_point t) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t).count();
  };
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const char* od = env_or_null_("VPIPE_ANE_RTQ_DIR");
  const std::string root =
      od != nullptr ? std::string(od)
                    : (fs::temp_directory_path() / "vpipe-ane-rtq").string();
  std::error_code ec;
  fs::create_directories(root, ec);
  const int iters = std::max(3, envi("VPIPE_ANE_RTQ_ITERS", 10));
  const char* only = env_or_null_("VPIPE_ANE_RTQ_ONLY");

  std::uint32_t lcg = 0x6c8e9cf5u;
  auto uni = [&]() {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)(lcg >> 8) * (2.0f / 16777216.0f) - 1.0f;
  };
  auto h16 = [](double v) { return (double)(_Float16)v; };
  auto bits16 = [](double v) {
    const _Float16 h = (_Float16)v;
    std::uint16_t w = 0;
    std::memcpy(&w, &h, 2);
    return w;
  };
  auto dims = [](long long a, long long b) {
    return "[" + std::to_string(a) + ", " + std::to_string(b) + "]";
  };
  auto blob = [](const std::string& ty, const std::string& d,
                 std::size_t off) {
    return "tensor<" + ty + ", " + d + ">(BLOBFILE(path = string(\"" +
           "@model_path/weights/weight.bin\"), offset = uint64(" +
           std::to_string(off) + ")))";
  };
  // An IOSurface-backed pixel buffer of fp16 or int8 elements, rows x
  // cols, filled row by row (a pixel buffer may pad its rows).
  auto make_pb = [](int rows, int cols, bool i8, const void* src) {
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
        i8 ? kCVPixelFormatType_OneComponent8
           : kCVPixelFormatType_OneComponent16Half,
        attrs, &pb);
    CFRelease(attrs);
    CFRelease(empty);
    if (rc != kCVReturnSuccess || pb == nullptr) { return pb; }
    const std::size_t es = i8 ? 1 : 2;
    CVPixelBufferLockBaseAddress(pb, 0);
    auto* dst = static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(pb));
    const std::size_t bpr = CVPixelBufferGetBytesPerRow(pb);
    for (int r = 0; r < rows; ++r) {
      std::memcpy(dst + (std::size_t)r * bpr,
                  static_cast<const std::uint8_t*>(src) +
                      (std::size_t)r * cols * es,
                  (std::size_t)cols * es);
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return pb;
  };

  struct Cfg { int M, K, L; };          // N == K, so the layers chain
  const Cfg cfgs[] = {{2048, 2048, 4}, {2048, 4096, 2}, {2048, 4096, 1}};
  const char* modes[] = {"f16-rt",         "w8-cast-rt",  "w8-dq-rt",
                         "w8a8-static-rt", "w8a8-dyn-rt", "w8a8-in-rt",
                         "w8a8-dyn1-rt",   "w8a8-dynr-rt", "w8a8-in1-rt",
                         "f16-baked",      "w8a8-baked"};
  const double a_step = h16(3.0 / 127.0);    // the static activation step
  const double i127   = h16(1.0 / 127.0);
  for (const Cfg& cf : cfgs) {
    const int M = cf.M, K = cf.K, N = cf.K, L = cf.L;
    // The weights, the inputs and every arm's reference share these.
    std::vector<std::vector<float>> wf((std::size_t)L);
    std::vector<std::vector<std::uint16_t>> w16((std::size_t)L);
    std::vector<std::vector<std::int8_t>> w8((std::size_t)L);
    std::vector<std::vector<double>> sw((std::size_t)L);   // fp16 steps
    for (int l = 0; l < L; ++l) {
      const std::size_t n = (std::size_t)N * K;
      const float amp = std::sqrt(3.0f / (float)K);
      wf[(std::size_t)l].resize(n);
      w16[(std::size_t)l].resize(n);
      w8[(std::size_t)l].resize(n);
      sw[(std::size_t)l].assign((std::size_t)N, 0.0);
      for (std::size_t i = 0; i < n; ++i) {
        wf[(std::size_t)l][i] = uni() * amp;
        w16[(std::size_t)l][i] = bits16(wf[(std::size_t)l][i]);
      }
      for (int r = 0; r < N; ++r) {
        double m = 0.0;
        for (int c = 0; c < K; ++c) {
          m = std::max(m, (double)std::fabs(
                              wf[(std::size_t)l][(std::size_t)r * K + c]));
        }
        const double s = h16(std::max(m, 1e-6) / 127.0);
        sw[(std::size_t)l][(std::size_t)r] = s;
        for (int c = 0; c < K; ++c) {
          const std::size_t i = (std::size_t)r * K + c;
          w8[(std::size_t)l][i] = (std::int8_t)std::clamp(
              std::nearbyint(wf[(std::size_t)l][i] / s), -127.0, 127.0);
        }
      }
    }
    std::vector<double> xf((std::size_t)M * K);
    std::vector<std::uint16_t> x16(xf.size());
    for (std::size_t i = 0; i < xf.size(); ++i) {
      xf[i] = h16(uni());
      x16[i] = bits16(xf[i]);
    }
    // The first input as the GPU's token quantizer would hand it over.
    std::vector<std::int8_t> x8(xf.size());
    std::vector<double> xs((std::size_t)M);
    for (int m = 0; m < M; ++m) {
      double a = 0.0;
      for (int c = 0; c < K; ++c) {
        a = std::max(a, std::fabs(xf[(std::size_t)m * K + c]));
      }
      xs[(std::size_t)m] = h16(std::max(a, 1e-6) / 127.0);
      for (int c = 0; c < K; ++c) {
        const std::size_t i = (std::size_t)m * K + c;
        x8[i] = (std::int8_t)std::clamp(
            std::nearbyint(xf[i] / xs[(std::size_t)m]), -127.0, 127.0);
      }
    }

    for (const char* mode_c : modes) {
      const std::string mode = mode_c;
      const std::string name = mode + "-" + std::to_string(K) + "x" +
                               std::to_string(N) + "-L" + std::to_string(L);
      if (only != nullptr && name.find(only) == std::string::npos) {
        continue;
      }
      const bool rt     = mode.size() > 3 &&
                          mode.compare(mode.size() - 3, 3, "-rt") == 0;
      const bool w_i8   = rt && mode != "f16-rt";
      const bool baked8 = mode == "w8a8-baked";
      const bool a_stat = mode == "w8a8-static-rt" || baked8;
      const bool pair1  = mode == "w8a8-dyn1-rt" || mode == "w8a8-dynr-rt" ||
                          mode == "w8a8-in1-rt";
      const bool recip  = mode == "w8a8-dynr-rt";
      const bool a_dyn  = mode == "w8a8-dyn-rt" || mode == "w8a8-in-rt" ||
                          pair1;
      const bool x_in8  = mode == "w8a8-in-rt" || mode == "w8a8-in1-rt";
      const bool cast   = mode == "w8-cast-rt";

      // Baked weight slabs (blob dtype 1 fp16, 4 int8).
      std::vector<std::vector<unsigned char>> slab;
      std::vector<std::uint32_t> code;
      if (!rt) {
        for (int l = 0; l < L; ++l) {
          const std::size_t n = (std::size_t)N * K;
          if (!baked8) {
            std::vector<unsigned char> b(n * 2);
            std::memcpy(b.data(), w16[(std::size_t)l].data(), n * 2);
            slab.push_back(std::move(b));
            code.push_back(1);
          } else {
            std::vector<unsigned char> q(n), s((std::size_t)N * 2);
            std::memcpy(q.data(), w8[(std::size_t)l].data(), n);
            for (int r = 0; r < N; ++r) {
              const std::uint16_t v = bits16(sw[(std::size_t)l][(size_t)r]);
              std::memcpy(s.data() + (std::size_t)r * 2, &v, 2);
            }
            slab.push_back(std::move(q));
            code.push_back(4);
            slab.push_back(std::move(s));
            code.push_back(1);
          }
        }
      }
      std::vector<std::size_t> sizes;
      for (const auto& b : slab) { sizes.push_back(b.size()); }
      const std::vector<std::size_t> off =
          vpipe::AneEmitter::blob_offsets(sizes);

      // The interface, in declaration order; which features are int8.
      std::vector<vpipe::AneEmitter::Feature> ins;
      std::vector<bool> ins_i8;
      if (x_in8) {
        ins.push_back({"x8", {M, K}});
        ins_i8.push_back(true);
        ins.push_back({"sa", {M, 1}});
        ins_i8.push_back(false);
      } else {
        ins.push_back({"x", {M, K}});
        ins_i8.push_back(false);
      }
      if (rt) {
        for (int l = 0; l < L; ++l) {
          ins.push_back({"w" + std::to_string(l), {N, K}});
          ins_i8.push_back(w_i8);
          if (w_i8) {
            ins.push_back({"s" + std::to_string(l), {1, N}});
            ins_i8.push_back(false);
          }
        }
      }
      const std::string MK = dims(M, K), MN = dims(M, N), NK = dims(N, K);
      const std::string M1 = dims(M, 1);

      auto mil_for = [&](const std::string& opset) {
        std::ostringstream o;
        o << vpipe::AneEmitter::mil_preamble() << "    func main<" << opset
          << ">(";
        for (std::size_t i = 0; i < ins.size(); ++i) {
          o << (i ? ", " : "") << "tensor<" << (ins_i8[i] ? "int8" : "fp16")
            << ", " << dims(ins[i].shape[0], ins[i].shape[1]) << "> "
            << ins[i].name;
        }
        o << ") {\n";
        auto line = [&](const std::string& s) {
          o << "            " << s << ";\n";
        };
        auto cst = [&](const std::string& ty, const std::string& nm,
                       const std::string& val) {
          line(ty + " " + nm + " = const()[name = string(\"" + nm +
               "\"), val = " + val + "]");
        };
        auto fp = [](double v) {
          char b[48];
          std::snprintf(b, sizeof(b), "fp16(%.9g)", v);
          return std::string(b);
        };
        cst("bool", "ftx", "bool(false)");
        cst("bool", "tty", "bool(true)");
        cst("fp16", "c128", fp(1.0 / 128.0));
        cst("fp16", "f128", fp(128.0));
        cst("fp16", "one", fp(1.0));
        cst("fp16", "asc", fp(a_step));
        cst("fp16", "i127", fp(i127));
        cst("fp16", "eps", fp(1e-4));
        cst("string", "adt", "string(\"int8\")");
        cst("string", "dt16", "string(\"fp16\")");
        cst("tensor<int32, [1]>", "rax", "tensor<int32, [1]>([1])");
        cst("bool", "kd", "bool(true)");
        std::string cur = "x";
        std::size_t si = 0;
        for (int l = 0; l < L; ++l) {
          const std::string ls = std::to_string(l);
          // The weight as the matmul reads it.
          std::string wl = "w" + ls;
          if (cast) {
            line("tensor<fp16, " + NK + "> wh" + ls + " = cast(dtype = dt16, "
                 "x = w" + ls + ")[name = string(\"wh" + ls + "\")]");
            wl = "wh" + ls;
          } else if (w_i8) {
            line("tensor<fp16, " + NK + "> wh" + ls + " = dequantize(input = "
                 "w" + ls + ", scale = c128)[name = string(\"wh" + ls +
                 "\")]");
            wl = "wh" + ls;
          } else if (!rt && !baked8) {
            cst("tensor<fp16, " + NK + ">", wl, blob("fp16", NK, off[si]));
            si += 1;
          } else if (baked8) {
            line("tensor<fp16, " + NK + "> " + wl +
                 " = constexpr_blockwise_shift_scale(data = " +
                 blob("int8", NK, off[si]) + ", scale = " +
                 blob("fp16", dims(N, 1), off[si + 1]) +
                 ")[name = string(\"" + wl + "\")]");
            si += 2;
          }
          // The activation as the matmul reads it, and the per-token
          // multiplier its output then takes (empty: none).
          std::string a = cur, tok;
          // pair1: dequantize at the step it quantized with (1), the
          // 1/128 left on the weight side; the multiplier then s_a alone.
          const std::string dqs = pair1 ? "one" : "c128";
          if (x_in8 && l == 0) {
            line("tensor<fp16, " + MK + "> d0 = dequantize(input = x8, "
                 "scale = " + dqs + ")[name = string(\"d0\")]");
            a   = "d0";
            tok = "sa";
          } else if (a_dyn) {
            line("tensor<fp16, " + MK + "> ab" + ls + " = abs(x = " + cur +
                 ")[name = string(\"ab" + ls + "\")]");
            line("tensor<fp16, " + M1 + "> am" + ls + " = reduce_max(axes = "
                 "rax, keep_dims = kd, x = ab" + ls + ")[name = string(\"am" +
                 ls + "\")]");
            line("tensor<fp16, " + M1 + "> sc" + ls + " = mul(x = am" + ls +
                 ", y = i127)[name = string(\"sc" + ls + "\")]");
            line("tensor<fp16, " + M1 + "> sa" + ls + " = maximum(x = sc" +
                 ls + ", y = eps)[name = string(\"sa" + ls + "\")]");
            if (recip) {
              line("tensor<fp16, " + M1 + "> rs" + ls + " = real_div(x = one"
                   ", y = sa" + ls + ")[name = string(\"rs" + ls + "\")]");
              line("tensor<fp16, " + MK + "> xs" + ls + " = mul(x = " + cur +
                   ", y = rs" + ls + ")[name = string(\"xs" + ls + "\")]");
            } else {
              line("tensor<fp16, " + MK + "> xs" + ls + " = real_div(x = " +
                   cur + ", y = sa" + ls + ")[name = string(\"xs" + ls +
                   "\")]");
            }
            line("tensor<int8, " + MK + "> q" + ls + " = quantize(input = xs" +
                 ls + ", output_dtype = adt, scale = one)[name = string(\"q" +
                 ls + "\")]");
            line("tensor<fp16, " + MK + "> d" + ls + " = dequantize(input = q" +
                 ls + ", scale = " + dqs + ")[name = string(\"d" + ls +
                 "\")]");
            a = "d" + ls;
            if (pair1) {
              tok = "sa" + ls;
            } else {
              line("tensor<fp16, " + M1 + "> sb" + ls + " = mul(x = sa" + ls +
                   ", y = f128)[name = string(\"sb" + ls + "\")]");
              tok = "sb" + ls;
            }
          } else if (a_stat) {
            line("tensor<int8, " + MK + "> q" + ls + " = quantize(input = " +
                 cur + ", output_dtype = adt, scale = asc)[name = string(\"q" +
                 ls + "\")]");
            line("tensor<fp16, " + MK + "> d" + ls + " = dequantize(input = q" +
                 ls + ", scale = asc)[name = string(\"d" + ls + "\")]");
            a = "d" + ls;
          }
          std::string y = "y" + ls;
          line("tensor<fp16, " + MN + "> " + y + " = matmul(transpose_x = "
               "ftx, transpose_y = tty, x = " + a + ", y = " + wl +
               ")[name = string(\"" + y + "\")]");
          if (!tok.empty()) {
            line("tensor<fp16, " + MN + "> yt" + ls + " = mul(x = " + y +
                 ", y = " + tok + ")[name = string(\"yt" + ls + "\")]");
            y = "yt" + ls;
          }
          if (w_i8) {
            line("tensor<fp16, " + MN + "> ys" + ls + " = mul(x = " + y +
                 ", y = s" + ls + ")[name = string(\"ys" + ls + "\")]");
            y = "ys" + ls;
          }
          cur = y;
        }
        o << "        } -> (" << cur << ");\n}";
        return std::make_pair(o.str(), cur);
      };

      // Write, relabel and load, trying the int8-capable target first.
      // The iOS 26 generation's opset is named by its internal number:
      // `ios19` parses, `ios26` is "not in blocks". The header's 9 stays:
      // written as 10, the model loads and every int8 INPUT is refused at
      // predict ("Invalid multi-array data type: 00020008").
      std::shared_ptr<vpipe::CoreMLLoadedModel> m;
      std::string out_name, how, dir;
      double compile_ms = 0.0;
      const char* opsets[] = {"ios19", "ios18"};
      const std::uint32_t specs[] = {9, 10};
      for (const char* os : opsets) {
        for (std::uint32_t sv : specs) {
          if (m != nullptr) { break; }
          auto mo = mil_for(os);
          out_name = mo.second;
          dir = root + "/" + name + "-" + os + "-v" + std::to_string(sv) +
                ".mlmodelc";
          const vpipe::AneEmitter::Feature fout{out_name, {M, N}};
          std::vector<vpipe::AneBlobSlab> bs;
          for (const auto& b : slab) { bs.push_back({b.data(), b.size()}); }
          std::string err;
          if (!vpipe::AneEmitter::write_bundle(dir, mo.first, ins,
                                               {&fout, 1}, bs, &err)) {
            std::printf("[ane_rtq] %s: not written (%s)\n", name.c_str(),
                        err.c_str());
            continue;
          }
          for (std::size_t i = 0; i < code.size(); ++i) {
            if (code[i] == 1) { continue; }
            std::fstream f(dir + "/weights/weight.bin",
                           std::ios::in | std::ios::out | std::ios::binary);
            f.seekp((std::streamoff)(off[i] + 4));
            f.write(reinterpret_cast<const char*>(&code[i]), 4);
          }
          // coremldata.bin: the spec version, and each int8 feature's
          // array type (FLOAT16 65552 -> INT8 131080, both three-byte
          // varints after the field's tag 0x10, in feature order).
          {
            std::fstream f(dir + "/coremldata.bin",
                           std::ios::in | std::ios::out | std::ios::binary);
            std::string b((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
            std::memcpy(&b[0x04], &sv, 4);
            std::memcpy(&b[0x2b], &sv, 4);
            const char pat[4] = {'\x10', '\x90', '\x80', '\x04'};
            std::size_t pos = 0, k = 0;
            while ((pos = b.find(std::string(pat, 4), pos)) !=
                   std::string::npos) {
              if (k < ins_i8.size() && ins_i8[k]) {
                b[pos + 1] = '\x88';
                b[pos + 2] = '\x80';
                b[pos + 3] = '\x08';
              }
              ++k;
              pos += 4;
            }
            ASSERT_TRUE(k == ins.size() + 1);
            f.clear();
            f.seekp(0);
            f.write(b.data(), (std::streamsize)b.size());
          }
          const auto t0 = Clk::now();
          m = mgr->load(dir, 3);
          compile_ms = ms_since(t0);
          if (m != nullptr && !m->valid()) { m = nullptr; }
          how = std::string(os) + "/spec " + std::to_string(sv);
          if (m == nullptr) {
            std::printf("[ane_rtq] %s: %s did not load\n", name.c_str(),
                        how.c_str());
          }
        }
      }
      slab.clear();
      if (m == nullptr) {
        std::printf("[ane_rtq] %-24s LOADED UNDER NO TARGET\n",
                    name.c_str());
        continue;
      }

      // Bind: x on the host, every weight an IOSurface of its type.
      std::vector<CVPixelBufferRef> pbs;
      std::vector<vpipe::CoreMLPredictInput> pin;
      std::vector<std::vector<std::uint16_t>> host;   // scales
      host.reserve(2 * (std::size_t)L + 1);
      for (std::size_t i = 0; i < ins.size(); ++i) {
        vpipe::CoreMLPredictInput in;
        in.name  = ins[i].name;
        in.dtype = vpipe::CoreMLDType::F16;
        in.shape = ins[i].shape;
        const std::string& nm = ins[i].name;
        if (nm == "x") {
          in.data = x16.data();
        } else if (nm == "x8") {
          pbs.push_back(make_pb(M, K, true, x8.data()));
          in.pixel_buffer = pbs.back();
        } else if (nm == "sa") {
          std::vector<std::uint16_t> v((std::size_t)M);
          for (int r = 0; r < M; ++r) {
            v[(std::size_t)r] =
                bits16(xs[(std::size_t)r] * (pair1 ? 1.0 : 128.0));
          }
          host.push_back(std::move(v));
          in.data = host.back().data();
        } else if (nm[0] == 'w') {
          const int l = std::atoi(nm.c_str() + 1);
          const void* src = w_i8
                                ? (const void*)w8[(std::size_t)l].data()
                                : (const void*)w16[(std::size_t)l].data();
          pbs.push_back(make_pb(N, K, w_i8, src));
          in.pixel_buffer = pbs.back();
        } else {                                   // s<l>
          const int l = std::atoi(nm.c_str() + 1);
          const double k = cast ? 1.0 : 128.0;
          std::vector<std::uint16_t> v((std::size_t)N);
          for (int r = 0; r < N; ++r) {
            v[(std::size_t)r] = bits16(sw[(std::size_t)l][(size_t)r] * k);
          }
          host.push_back(std::move(v));
          in.data = host.back().data();
        }
        pin.push_back(std::move(in));
      }
      std::vector<std::uint16_t> yb((std::size_t)M * N);
      vpipe::CoreMLPredictOutput out;
      out.name          = out_name;
      out.want          = vpipe::CoreMLDType::F16;
      out.backing       = yb.data();
      out.backing_elems = yb.size();
      bool ok = m->predict(pin, {&out, 1});
      std::vector<double> t;
      for (int i = 0; i < iters && ok; ++i) {
        const auto ti = Clk::now();
        ok = m->predict(pin, {&out, 1});
        t.push_back(ms_since(ti));
      }
      for (CVPixelBufferRef pb : pbs) {
        if (pb != nullptr) { CVPixelBufferRelease(pb); }
      }
      if (!ok || t.empty()) {
        std::printf("[ane_rtq] %-24s PREDICT FAILED (%s)\n", name.c_str(),
                    how.c_str());
        ASSERT_TRUE(false);
        continue;
      }
      std::sort(t.begin(), t.end());
      const double med = t[t.size() / 2];

      // The reference: the same arithmetic in double on four rows -- and,
      // for what a model would see, the UNQUANTIZED product (the float
      // weights, the input as given), whose distance is the mode's real
      // cost.
      double num = 0.0, den = 0.0, enum_ = 0.0, eden = 0.0;
      for (const int r : {0, 1, M / 2, M - 1}) {
        std::vector<double> v((std::size_t)K), ve((std::size_t)K);
        for (int c = 0; c < K; ++c) {
          v[(std::size_t)c] =
              x_in8 ? (double)x8[(std::size_t)r * K + c] *
                          xs[(std::size_t)r]
                    : xf[(std::size_t)r * K + c];
          ve[(std::size_t)c] = xf[(std::size_t)r * K + c];
        }
        for (int l = 0; l < L; ++l) {
          std::vector<double> yex((std::size_t)N, 0.0);
          for (int n = 0; n < N; ++n) {
            const float* wr = wf[(std::size_t)l].data() + (std::size_t)n * K;
            double acc = 0.0;
            for (int c = 0; c < K; ++c) {
              acc += (double)wr[c] * ve[(std::size_t)c];
            }
            yex[(std::size_t)n] = acc;
          }
          ve = std::move(yex);
          if (a_stat) {
            for (double& e : v) {
              e = std::clamp(std::nearbyint(e / a_step), -128.0, 127.0) *
                  a_step;
            }
          } else if (a_dyn && !(x_in8 && l == 0)) {
            double am = 0.0;
            for (double e : v) { am = std::max(am, std::fabs(h16(e))); }
            const double s = std::max(h16(am * i127), 1e-4);
            for (double& e : v) {
              e = std::clamp(std::nearbyint(e / s), -128.0, 127.0) * s;
            }
          }
          std::vector<double> y((std::size_t)N, 0.0);
          for (int n = 0; n < N; ++n) {
            const std::size_t o = (std::size_t)n * K;
            double acc = 0.0;
            if (w_i8 || baked8) {
              for (int c = 0; c < K; ++c) {
                acc += (double)w8[(std::size_t)l][o + c] * v[(std::size_t)c];
              }
              acc *= sw[(std::size_t)l][(std::size_t)n];
            } else {
              for (int c = 0; c < K; ++c) {
                _Float16 hw;
                std::memcpy(&hw, &w16[(std::size_t)l][o + c], 2);
                acc += (double)hw * v[(std::size_t)c];
              }
            }
            y[(std::size_t)n] = acc;
          }
          v = std::move(y);
        }
        for (int n = 0; n < N; ++n) {
          _Float16 hv;
          std::memcpy(&hv, &yb[(std::size_t)r * N + n], 2);
          const double e = (double)hv - v[(std::size_t)n];
          num += e * e;
          den += v[(std::size_t)n] * v[(std::size_t)n];
          const double ee = (double)hv - ve[(std::size_t)n];
          enum_ += ee * ee;
          eden += ve[(std::size_t)n] * ve[(std::size_t)n];
        }
      }
      const double rel = std::sqrt(num / std::max(den, 1e-30));
      const double erel = std::sqrt(enum_ / std::max(eden, 1e-30));
      const double flop = 2.0 * L * (double)M * K * N;
      std::printf("[ane_rtq] %-24s %-13s compile %6.0f ms, median %7.2f ms "
                  "= %6.2f TOP/s, rel %.1e (from unquantized %.1e)\n",
                  name.c_str(), how.c_str(), compile_ms, med,
                  flop / med / 1e9, rel, erel);
      std::fflush(stdout);
      EXPECT_TRUE(rel < 5e-2);
    }
  }
}

// THE TIER'S FEED-FORWARD WITH INT8 INPUTS (see runtime_int8_probe for the
// mechanism). One SwiGLU chunk at MiniMax-H3's widths, tiled as
// AneEmitter tiles the tier's own graph (x sliced along K, each weight
// tile sliced out of the whole [out, in] input, matmul with transpose_y,
// summed over K, concatenated over N), the int8 variants slicing the int8
// tensors and dequantizing each tile right before its matmul:
//
//   tier       the tier's module itself (AneModule::build_runtime)
//   f16        this generator, fp16 -- must equal the tier
//   w8a16      int8 weight inputs for all three projections
//   a8gu-in    W8A8 gate|up from int8 codes + per-token scales given as
//              inputs (what the GPU's token quantizer writes), W8A16 down
//   a8gu-dyn   the same with x quantized per token inside the graph
//   a8all-in   gate|up as a8gu-in, the down projection's input quantized
//              per token inside the graph (one in-graph quantization)
//   a8all-dyn  both quantized inside the graph
//
// and each again ROTATED (the GPU's `token` tier's arithmetic): every
// weight row Walsh-Hadamard transformed per 256-block of K before its
// int8 (or fp16) step, and every activation by the same transform inside
// the graph -- [M, K] reshaped to [M, K/256, 256] and multiplied by the
// orthonormal Hadamard, a model input -- before it is quantized (the
// gate|up input given as the GPU's rotated quantizer writes it):
//   f16-rot, w8a16-rot, a8gu-rot-in, a8all-rot-in, a8all-rot-dyn
//
// beside two CPU emulations of the GPU's token tiers over the same
// weights and rows -- token_plain (per-token activations, per-row
// weights, all three projections) and token (the same after a
// Walsh-Hadamard rotation of every 256-block of K) -- for scale.
//
// On SYNTHETIC data by default (random weights and inputs: no outlier
// channels), or on REAL data: VPIPE_ANE_RTQF_REAL names a directory of
// feed-forward inputs a forward saved (VPIPE_H3_FF_CAPTURE: ff_in_<L>.bf16,
// [rows, hidden]) and VPIPE_ANE_RTQF_CKPT the checkpoint that forward ran
// (a single safetensors file, blocks.<L>.mlp.fc1 / fc2), the rows
// repeated to fill the chunk. Each form is scored on sampled rows against
// a double restatement of its own arithmetic and of the unquantized
// feed-forward.
//
//   VPIPE_ANE_RTQF_PROBE=1         run it
//   VPIPE_ANE_RTQF_REAL / _CKPT    real data (above)
//   VPIPE_ANE_RTQF_BLOCKS          real blocks ("0,12,24,36,49")
//   VPIPE_ANE_RTQF_UPSHIFT         up projection / 2^k (fp16 headroom)
//   VPIPE_ANE_RTQF_HID / _FFN      synthetic widths (5376 / 14336)
//   VPIPE_ANE_RTQF_ROWS / _TILE    rows a chunk (2048), tile (1024)
//   VPIPE_ANE_RTQF_ITERS           timed predicts an arm (8)
//   VPIPE_ANE_RTQF_ONLY / _DIR     arms whose name contains this; bundles
TEST(ane_module, runtime_int8_ffn_probe)
{
  if (env_or_null_("VPIPE_ANE_RTQF_PROBE") == nullptr) { return; }
  namespace mcx = vpipe::metal_compute;
  vpipe::Session sess;
  auto* svc = sess.services();
  auto* mgr = svc != nullptr ? svc->coreml_model_manager() : nullptr;
  mcx::MetalCompute* mc = sess.metal_compute();
  if (mgr == nullptr || mc == nullptr || !mc->valid()) { return; }
  namespace fs = std::filesystem;
  using Clk = std::chrono::steady_clock;
  auto ms_since = [](Clk::time_point t) {
    return std::chrono::duration<double, std::milli>(Clk::now() - t).count();
  };
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const char* real = env_or_null_("VPIPE_ANE_RTQF_REAL");
  const char* ckpt = env_or_null_("VPIPE_ANE_RTQF_CKPT");
  const bool is_real = real != nullptr && ckpt != nullptr;
  int H = envi("VPIPE_ANE_RTQF_HID", 5376);
  int F = envi("VPIPE_ANE_RTQF_FFN", 14336);
  const int M = envi("VPIPE_ANE_RTQF_ROWS", 2048);
  const int T = envi("VPIPE_ANE_RTQF_TILE", 1024);
  const int iters = std::max(3, envi("VPIPE_ANE_RTQF_ITERS", 8));
  const char* only = env_or_null_("VPIPE_ANE_RTQF_ONLY");
  const char* od = env_or_null_("VPIPE_ANE_RTQF_DIR");
  const std::string root =
      od != nullptr ? std::string(od)
                    : (fs::temp_directory_path() / "vpipe-ane-rtqf").string();
  std::error_code ec;
  fs::create_directories(root, ec);

  std::uint32_t lcg = 0x1b873593u;
  auto uni = [&]() {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)(lcg >> 8) * (2.0f / 16777216.0f) - 1.0f;
  };
  auto h16 = [](double v) { return (double)(_Float16)v; };
  auto bits16 = [](double v) {
    const _Float16 h = (_Float16)v;
    std::uint16_t w = 0;
    std::memcpy(&w, &h, 2);
    return w;
  };
  auto bf16f = [](std::uint16_t b) {
    const std::uint32_t u = (std::uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  auto make_pb = [](int rows, int cols, bool i8, const void* src) {
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
        i8 ? kCVPixelFormatType_OneComponent8
           : kCVPixelFormatType_OneComponent16Half,
        attrs, &pb);
    CFRelease(attrs);
    CFRelease(empty);
    if (rc != kCVReturnSuccess || pb == nullptr) { return pb; }
    const std::size_t es = i8 ? 1 : 2;
    CVPixelBufferLockBaseAddress(pb, 0);
    auto* dst = static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(pb));
    const std::size_t bpr = CVPixelBufferGetBytesPerRow(pb);
    for (int r = 0; r < rows; ++r) {
      std::memcpy(dst + (std::size_t)r * bpr,
                  static_cast<const std::uint8_t*>(src) +
                      (std::size_t)r * cols * es,
                  (std::size_t)cols * es);
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return pb;
  };
  // Orthonormal Walsh-Hadamard over every 256-block of v (the GPU token
  // tier's rotation; self-inverse).
  auto fwht256 = [](double* v, std::size_t n) {
    for (std::size_t b0 = 0; b0 + 256 <= n; b0 += 256) {
      double* p = v + b0;
      for (int h = 1; h < 256; h <<= 1) {
        for (int i = 0; i < 256; i += 2 * h) {
          for (int j = i; j < i + h; ++j) {
            const double a = p[j], c = p[j + h];
            p[j] = a + c;
            p[j + h] = a - c;
          }
        }
      }
      for (int i = 0; i < 256; ++i) { p[i] *= 1.0 / 16.0; }
    }
  };

  // A weight, [out, in] as a checkpoint stores it: float, fp16, int8 per
  // output row with its fp16 step, and (for the rotated emulation) int8
  // per row of the rotated rows.
  struct Wt {
    int rows = 0, cols = 0;
    std::vector<float> f;
    std::vector<std::uint16_t> h, hr;     // hr: the rotated rows, fp16
    std::vector<std::int8_t> q, qr;
    std::vector<double> s, sr;
  };
  auto quantize_w = [&](Wt& w) {
    const std::size_t n = (std::size_t)w.rows * w.cols;
    w.h.resize(n);
    w.q.resize(n);
    w.s.resize((std::size_t)w.rows);
    w.qr.resize(n);
    w.hr.resize(n);
    w.sr.resize((std::size_t)w.rows);
    std::vector<double> row((std::size_t)w.cols);
    for (int r = 0; r < w.rows; ++r) {
      const std::size_t o = (std::size_t)r * w.cols;
      double m = 0.0;
      for (int c = 0; c < w.cols; ++c) {
        w.h[o + c] = bits16(w.f[o + c]);
        m = std::max(m, (double)std::fabs(w.f[o + c]));
        row[(std::size_t)c] = w.f[o + c];
      }
      const double s = h16(std::max(m, 1e-8) / 127.0);
      w.s[(std::size_t)r] = s;
      for (int c = 0; c < w.cols; ++c) {
        w.q[o + c] = (std::int8_t)std::clamp(std::nearbyint(w.f[o + c] / s),
                                             -127.0, 127.0);
      }
      fwht256(row.data(), row.size());
      double mr = 0.0;
      for (double e : row) { mr = std::max(mr, std::fabs(e)); }
      const double sr = h16(std::max(mr, 1e-12) / 127.0);
      w.sr[(std::size_t)r] = sr;
      for (int c = 0; c < w.cols; ++c) {
        w.hr[o + c] = bits16(row[(std::size_t)c]);
        w.qr[o + c] = (std::int8_t)std::clamp(
            std::nearbyint(row[(std::size_t)c] / sr), -127.0, 127.0);
      }
    }
  };

  // The cases: synthetic, or each real block asked for.
  struct Case {
    std::string name;
    Wt wg, wu, wd;
    int R = 0;                          // distinct input rows
    std::vector<double> xr;             // [R, H], bf16-exact
  };
  std::vector<int> blocks;
  if (is_real) {
    const char* bl = env_or_null_("VPIPE_ANE_RTQF_BLOCKS");
    std::stringstream ss(bl != nullptr ? bl : "0,12,24,36,49");
    std::string t;
    while (std::getline(ss, t, ',')) { blocks.push_back(std::atoi(t.c_str())); }
  } else {
    blocks.push_back(-1);
  }
  // The checkpoint's header, once.
  vpipe::FlexData hdr;
  std::uint64_t data0 = 0;
  if (is_real) {
    std::ifstream f(ckpt, std::ios::binary);
    std::uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    std::string js(n, '\0');
    f.read(js.data(), (std::streamsize)n);
    ASSERT_TRUE(f.good());
    if (!f.good()) { return; }
    hdr = vpipe::FlexData::from_json(js);
    data0 = 8 + n;
  }
  auto read_bf16 = [&](const std::string& name, int* rows, int* cols,
                       std::vector<float>* out) {
    auto o = hdr.as_object();
    if (!o.contains(name)) { return false; }
    const vpipe::FlexData t = o.at(name);
    auto to = t.as_object();
    const vpipe::FlexData shape = to.at("shape");
    const vpipe::FlexData offs = to.at("data_offsets");
    auto sa = shape.as_array();
    auto oa = offs.as_array();
    *rows = (int)sa.at(0).as_int();
    *cols = (int)sa.at(1).as_int();
    const std::uint64_t b = (std::uint64_t)oa.at(0).as_int();
    const std::size_t n = (std::size_t)(*rows) * (std::size_t)(*cols);
    std::vector<std::uint16_t> raw(n);
    std::ifstream f(ckpt, std::ios::binary);
    f.seekg((std::streamoff)(data0 + b));
    f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)(n * 2));
    if (!f.good()) { return false; }
    out->resize(n);
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = bf16f(raw[i]); }
    return true;
  };

  const double i127 = h16(1.0 / 127.0);
  // Per-mode flags; the emulations are CPU only.
  struct Mode {
    const char* name;
    bool w8;
    int  a_gu;     // 0 fp16, 1 codes given, 2 quantized in the graph
    int  a_d;      // 0 fp16, 2 quantized in the graph
    bool rot;      // the rotated emulation
  };
  const Mode modes[] = {{"f16", false, 0, 0, false},
                        {"w8a16", true, 0, 0, false},
                        {"a8gu-in", true, 1, 0, false},
                        {"a8gu-dyn", true, 2, 0, false},
                        {"a8all-in", true, 1, 2, false},
                        {"a8all-dyn", true, 2, 2, false},
                        {"f16-rot", false, 0, 0, true},
                        {"w8a16-rot", true, 0, 0, true},
                        {"a8gu-rot-in", true, 1, 0, true},
                        {"a8all-rot-in", true, 1, 2, true},
                        {"a8all-rot-dyn", true, 2, 2, true}};
  const Mode emu_plain{"emu-token-plain", true, 2, 2, false};
  const Mode emu_rot{"emu-token-rot", true, 2, 2, true};

  // The compiled forms, built at the first case and reused.
  struct Built {
    std::shared_ptr<vpipe::CoreMLLoadedModel> m;
    std::vector<vpipe::AneEmitter::Feature> ins;
    std::string out;
    double compile_ms = 0.0;
  };
  std::map<std::string, Built> built;
  // The orthonormal Hadamard, fp16 (every entry +-1/16, exact). A MODEL
  // INPUT, not a constant: a graph with no weight file is cached on its
  // content, one with a weight file on the file -- re-emitted at every
  // load, it would recompile cold (11-14 s for these graphs) each process.
  std::vector<std::uint16_t> hmat(256 * 256);
  for (std::size_t i = 0; i < hmat.size(); ++i) {
    const int r = (int)(i / 256), c = (int)(i % 256);
    hmat[i] = bits16((__builtin_popcount(r & c) & 1) ? -1.0 / 16.0
                                                     : 1.0 / 16.0);
  }
  std::unique_ptr<vpipe::genai::AneModule> tier;
  std::unique_ptr<vpipe::genai::AneWeightSlots> tier_slots;
  double tier_ms = 0.0;

  for (const int blk : blocks) {
    Case cs;
    if (blk < 0) {
      cs.name = "synthetic";
      auto fill = [&](Wt& w, int r, int c) {
        w.rows = r;
        w.cols = c;
        w.f.resize((std::size_t)r * c);
        const float amp = std::sqrt(3.0f / (float)c);
        for (auto& e : w.f) { e = uni() * amp; }
      };
      fill(cs.wg, F, H);
      fill(cs.wu, F, H);
      fill(cs.wd, H, F);
      cs.R = M;
      cs.xr.resize((std::size_t)M * H);
      for (auto& e : cs.xr) { e = (double)(float)(_Float16)uni(); }
    } else {
      cs.name = "block " + std::to_string(blk);
      const std::string p = "blocks." + std::to_string(blk) + ".mlp.";
      std::vector<float> fc1;
      int r1 = 0, c1 = 0, r2 = 0, c2 = 0;
      ASSERT_TRUE(read_bf16(p + "fc1.weight", &r1, &c1, &fc1));
      ASSERT_TRUE(read_bf16(p + "fc2.weight", &r2, &c2, &cs.wd.f));
      if (fc1.empty() || cs.wd.f.empty()) { continue; }
      H = c1;
      F = r1 / 2;
      // fc1 is [gate; up], gate first.
      cs.wg.rows = cs.wu.rows = F;
      cs.wg.cols = cs.wu.cols = H;
      cs.wg.f.assign(fc1.begin(), fc1.begin() + (std::ptrdiff_t)F * H);
      cs.wu.f.assign(fc1.begin() + (std::ptrdiff_t)F * H, fc1.end());
      cs.wd.rows = r2;
      cs.wd.cols = c2;
      const std::string xp =
          std::string(real) + "/ff_in_" + std::to_string(blk) + ".bf16";
      std::ifstream xf(xp, std::ios::binary | std::ios::ate);
      ASSERT_TRUE(xf.good());
      if (!xf.good()) { continue; }
      const std::size_t bytes = (std::size_t)xf.tellg();
      cs.R = (int)(bytes / ((std::size_t)H * 2));
      std::vector<std::uint16_t> raw((std::size_t)cs.R * H);
      xf.seekg(0);
      xf.read(reinterpret_cast<char*>(raw.data()),
              (std::streamsize)(raw.size() * 2));
      cs.xr.resize(raw.size());
      for (std::size_t i = 0; i < raw.size(); ++i) {
        cs.xr[i] = bf16f(raw[i]);
      }
    }
    // fp16 HEADROOM, as the tier takes it: the up projection divided by
    // 2^k (VPIPE_ANE_RTQF_UPSHIFT), so silu(gate) * up stays inside fp16
    // on a block whose activations reach past it (H3's last). Every score
    // is relative and every reference reads the same weights, so the
    // output's scale cancels.
    if (const int k = envi("VPIPE_ANE_RTQF_UPSHIFT", 0); k > 0) {
      for (float& e : cs.wu.f) { e = std::ldexp(e, -k); }
    }
    quantize_w(cs.wg);
    quantize_w(cs.wu);
    quantize_w(cs.wd);

    // The chunk's rows: the case's rows repeated. Its inputs as each form
    // reads them -- fp16 values (bf16 is exact in fp16 only in range, so
    // the narrowing is part of every ANE form), and int8 codes per token.
    std::vector<double> xf((std::size_t)M * H);
    for (int m = 0; m < M; ++m) {
      for (int c = 0; c < H; ++c) {
        xf[(std::size_t)m * H + c] =
            h16(cs.xr[(std::size_t)(m % cs.R) * H + c]);
      }
    }
    std::vector<std::uint16_t> x16(xf.size());
    for (std::size_t i = 0; i < xf.size(); ++i) { x16[i] = bits16(xf[i]); }
    std::vector<std::int8_t> x8(xf.size());
    std::vector<double> xs((std::size_t)M);
    for (int m = 0; m < M; ++m) {
      double a = 0.0;
      for (int c = 0; c < H; ++c) {
        a = std::max(a, std::fabs(xf[(std::size_t)m * H + c]));
      }
      xs[(std::size_t)m] = h16(std::max(a, 1e-6) / 127.0);
      for (int c = 0; c < H; ++c) {
        const std::size_t i = (std::size_t)m * H + c;
        x8[i] = (std::int8_t)std::clamp(
            std::nearbyint(xf[i] / xs[(std::size_t)m]), -127.0, 127.0);
      }
    }
    // ...and as the GPU's ROTATED token quantizer writes them: each row
    // Walsh-Hadamard rotated per 256-block, then quantized per token.
    std::vector<std::int8_t> x8r(xf.size());
    std::vector<double> xsr((std::size_t)M);
    {
      std::vector<double> row((std::size_t)H);
      for (int m = 0; m < M; ++m) {
        for (int c = 0; c < H; ++c) {
          row[(std::size_t)c] = xf[(std::size_t)m * H + c];
        }
        fwht256(row.data(), row.size());
        double a = 0.0;
        for (double e : row) { a = std::max(a, std::fabs(e)); }
        xsr[(std::size_t)m] = h16(std::max(a, 1e-6) / 127.0);
        for (int c = 0; c < H; ++c) {
          x8r[(std::size_t)m * H + c] = (std::int8_t)std::clamp(
              std::nearbyint(row[(std::size_t)c] / xsr[(std::size_t)m]),
              -127.0, 127.0);
        }
      }
    }
    // How heavy the input's tails are: per-channel amax over the rows,
    // max against median, and the largest |x|.
    {
      std::vector<double> ch((std::size_t)H, 0.0);
      double mx = 0.0;
      for (int r = 0; r < cs.R; ++r) {
        for (int c = 0; c < H; ++c) {
          const double a = std::fabs(cs.xr[(std::size_t)r * H + c]);
          ch[(std::size_t)c] = std::max(ch[(std::size_t)c], a);
          mx = std::max(mx, a);
        }
      }
      std::vector<double> sorted = ch;
      std::sort(sorted.begin(), sorted.end());
      std::printf("[ane_rtqf] %s: %d rows x %d -> %d, max |x| %.3g, "
                  "channel amax max / median %.1f\n", cs.name.c_str(),
                  cs.R, H, F, mx,
                  sorted.back() / std::max(sorted[sorted.size() / 2], 1e-30));
    }

    // Sampled rows: spread over the case's distinct rows.
    std::vector<int> rows;
    const int ns = blk < 0 ? 4 : 12;
    for (int i = 0; i < ns; ++i) {
      rows.push_back((int)std::lround((double)i * (cs.R - 1) / (ns - 1)));
    }
    auto ref_row = [&](int r, const Mode* md, std::vector<double>* out) {
      const bool exact = md == nullptr;
      std::vector<double> v((std::size_t)H);
      for (int c = 0; c < H; ++c) {
        const std::size_t i = (std::size_t)r * H + c;
        v[(std::size_t)c] =
            exact             ? cs.xr[i]
            : md->a_gu != 1   ? xf[i]
            : md->rot         ? (double)x8r[i] * xsr[(std::size_t)r]
                              : (double)x8[i] * xs[(std::size_t)r];
      }
      // An emulation of the GPU's tiers takes f32 scales; the ANE's graph
      // computes its in fp16.
      const bool gpu = !exact && md->name[0] == 'e';
      auto dynq = [&](std::vector<double>& a) {
        double am = 0.0;
        for (double e : a) { am = std::max(am, std::fabs(gpu ? e : h16(e))); }
        const double s = gpu ? std::max(am, 1e-12) / 127.0
                             : std::max(h16(am * i127), 1e-4);
        for (double& e : a) {
          e = std::clamp(std::nearbyint(e / s), -128.0, 127.0) * s;
        }
      };
      if (!exact && md->rot && md->a_gu != 1) {
        fwht256(v.data(), v.size());
      }
      if (!exact && md->a_gu == 2) { dynq(v); }
      auto proj = [&](const std::vector<double>& a, const Wt& w) {
        std::vector<double> y((std::size_t)w.rows, 0.0);
        for (int n = 0; n < w.rows; ++n) {
          const std::size_t o = (std::size_t)n * w.cols;
          double acc = 0.0;
          if (exact) {
            for (int c = 0; c < w.cols; ++c) {
              acc += (double)w.f[o + c] * a[(std::size_t)c];
            }
          } else if (md->rot && md->w8) {
            for (int c = 0; c < w.cols; ++c) {
              acc += (double)w.qr[o + c] * a[(std::size_t)c];
            }
            acc *= w.sr[(std::size_t)n];
          } else if (md->rot) {
            for (int c = 0; c < w.cols; ++c) {
              _Float16 hw;
              std::memcpy(&hw, &w.hr[o + c], 2);
              acc += (double)hw * a[(std::size_t)c];
            }
          } else if (md->w8) {
            for (int c = 0; c < w.cols; ++c) {
              acc += (double)w.q[o + c] * a[(std::size_t)c];
            }
            acc *= w.s[(std::size_t)n];
          } else {
            for (int c = 0; c < w.cols; ++c) {
              _Float16 hw;
              std::memcpy(&hw, &w.h[o + c], 2);
              acc += (double)hw * a[(std::size_t)c];
            }
          }
          y[(std::size_t)n] = acc;
        }
        return y;
      };
      const std::vector<double> g = proj(v, cs.wg), u = proj(v, cs.wu);
      std::vector<double> hh((std::size_t)F);
      for (int n = 0; n < F; ++n) {
        const double gg = g[(std::size_t)n];
        hh[(std::size_t)n] = gg / (1.0 + std::exp(-gg)) * u[(std::size_t)n];
      }
      if (!exact && md->rot) { fwht256(hh.data(), hh.size()); }
      if (!exact && md->a_d == 2) { dynq(hh); }
      *out = proj(hh, cs.wd);
    };
    std::vector<std::vector<double>> exact(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
      ref_row(rows[i], nullptr, &exact[i]);
    }
    // ANE output (or an emulation's rows) against both references.
    auto score = [&](const std::vector<std::uint16_t>* y, const Mode& md,
                     double* rel, double* erel) {
      double n1 = 0, d1 = 0, n2 = 0, d2 = 0;
      for (std::size_t i = 0; i < rows.size(); ++i) {
        std::vector<double> ref;
        ref_row(rows[i], &md, &ref);
        for (int n = 0; n < H; ++n) {
          double a = ref[(std::size_t)n];
          if (y != nullptr) {
            _Float16 hv;
            std::memcpy(&hv, &(*y)[(std::size_t)rows[i] * H + n], 2);
            a = (double)hv;
          }
          const double r = ref[(std::size_t)n];
          n1 += (a - r) * (a - r);
          d1 += r * r;
          const double e = exact[i][(std::size_t)n];
          n2 += (a - e) * (a - e);
          d2 += e * e;
        }
      }
      *rel = std::sqrt(n1 / std::max(d1, 1e-30));
      *erel = std::sqrt(n2 / std::max(d2, 1e-30));
    };
    const double flop = 6.0 * (double)M * H * F;
    auto report = [&](const std::string& name, double compile_ms,
                      double med, double rel, double erel) {
      if (med > 0.0) {
        std::printf("[ane_rtqf] %-9s %-15s compile %6.0f ms, median %7.2f ms "
                    "= %6.2f TOP/s, rel %.1e, from unquantized %.2e\n",
                    cs.name.c_str(), name.c_str(), compile_ms, med,
                    flop / med / 1e9, rel, erel);
      } else {
        std::printf("[ane_rtqf] %-9s %-15s (CPU emulation)%49s from "
                    "unquantized %.2e\n", cs.name.c_str(), name.c_str(), "",
                    erel);
      }
      std::fflush(stdout);
    };

    // ---- the tier's own module -------------------------------------------
    if (only == nullptr ||
        std::string("tier").find(only) != std::string::npos) {
      if (tier == nullptr) {
        vpipe::AneGraphSpec spec;
        spec.kind = vpipe::AneGraphSpec::Kind::SwiGluFfn;
        spec.M  = M;
        spec.K  = H;
        spec.N  = F;
        spec.bk = T;
        spec.bn = T;
        spec.runtime_weights = true;
        spec.weights_out_in  = true;
        const auto t0 = Clk::now();
        tier = vpipe::genai::AneModule::build_runtime(
            &sess, spec, root + "/tier.mlmodelc");
        tier_ms = ms_since(t0);
        tier_slots = vpipe::genai::AneWeightSlots::create(spec);
        ASSERT_TRUE(tier != nullptr && tier_slots != nullptr);
      }
      if (tier != nullptr && tier_slots != nullptr) {
        const Wt* ws[3] = {&cs.wg, &cs.wu, &cs.wd};
        for (std::size_t k = 0; k < 3; ++k) {
          const Wt& w = *ws[k];
          tier_slots->fill(k, [&](void* base, std::size_t bpr) {
            for (int r = 0; r < w.rows; ++r) {
              std::memcpy(static_cast<std::uint8_t*>(base) + r * bpr,
                          w.h.data() + (std::size_t)r * w.cols,
                          (std::size_t)w.cols * 2);
            }
          });
        }
        mcx::SharedBuffer xb = mc->make_shared_buffer(x16.size() * 2);
        mcx::SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * H * 2);
        std::memcpy(xb.contents(), x16.data(), x16.size() * 2);
        bool ok = tier->run(xb, 0, yb, 0, *tier_slots);
        std::vector<double> t;
        for (int i = 0; i < iters && ok; ++i) {
          const auto ti = Clk::now();
          ok = tier->run(xb, 0, yb, 0, *tier_slots);
          t.push_back(ms_since(ti));
        }
        ASSERT_TRUE(ok);
        if (ok) {
          std::sort(t.begin(), t.end());
          std::vector<std::uint16_t> y((std::size_t)M * H);
          std::memcpy(y.data(), yb.contents(), y.size() * 2);
          double rel = 0, erel = 0;
          score(&y, modes[0], &rel, &erel);
          report("tier", tier_ms, t[t.size() / 2], rel, erel);
          tier_ms = 0.0;
        }
      }
    }

    // ---- the generated forms ---------------------------------------------
    for (const Mode& md : modes) {
      const std::string name = md.name;
      if (only != nullptr && name.find(only) == std::string::npos) {
        continue;
      }
      if (built.count(name) == 0) {
        Built bt;
        std::vector<bool> ins_i8;
        auto add_in = [&](const std::string& nm, long long r, long long c,
                          bool i8) {
          bt.ins.push_back({nm, {r, c}});
          ins_i8.push_back(i8);
        };
        if (md.a_gu == 1) {
          add_in("x8", M, H, true);
          add_in("sa", M, 1, false);
        } else {
          add_in("x", M, H, false);
        }
        add_in("wg", F, H, md.w8);
        add_in("wu", F, H, md.w8);
        add_in("wd", H, F, md.w8);
        if (md.w8) {
          add_in("sgw", 1, F, false);
          add_in("suw", 1, F, false);
          add_in("sdw", 1, H, false);
        }
        // VPIPE_ANE_RTQF_HINLINE=1: the Hadamard as a constant written
        // INLINE in model.mil (no weight file) rather than an input.
        const bool hinline = envi("VPIPE_ANE_RTQF_HINLINE", 0) != 0;
        if (md.rot && !hinline) { add_in("hmat", 256, 256, false); }
        std::ostringstream o;
        o << vpipe::AneEmitter::mil_preamble() << "    func main<ios19>(";
        for (std::size_t i = 0; i < bt.ins.size(); ++i) {
          o << (i ? ", " : "") << "tensor<" << (ins_i8[i] ? "int8" : "fp16")
            << ", [" << bt.ins[i].shape[0] << ", " << bt.ins[i].shape[1]
            << "]> " << bt.ins[i].name;
        }
        o << ") {\n";
        auto line = [&](const std::string& s) {
          o << "            " << s << ";\n";
        };
        auto cst = [&](const std::string& ty, const std::string& nm,
                       const std::string& val) {
          line(ty + " " + nm + " = const()[name = string(\"" + nm +
               "\"), val = " + val + "]");
        };
        auto fp = [](double v) {
          char b[48];
          std::snprintf(b, sizeof(b), "fp16(%.9g)", v);
          return std::string(b);
        };
        auto ty = [](bool i8, long long r, long long c) {
          return std::string("tensor<") + (i8 ? "int8" : "fp16") + ", [" +
                 std::to_string(r) + ", " + std::to_string(c) + "]>";
        };
        auto i2 = [](long long a, long long b) {
          return "tensor<int32, [2]>([" + std::to_string(a) + ", " +
                 std::to_string(b) + "])";
        };
        cst("bool", "ftx", "bool(false)");
        cst("bool", "tty", "bool(true)");
        cst("fp16", "c128", fp(1.0 / 128.0));
        cst("fp16", "one", fp(1.0));
        cst("fp16", "i127", fp(i127));
        cst("fp16", "eps", fp(1e-4));
        cst("string", "adt", "string(\"int8\")");
        cst("tensor<int32, [1]>", "rax", "tensor<int32, [1]>([1])");
        cst("bool", "kd", "bool(true)");
        cst("int32", "cax", "int32(1)");
        cst("bool", "cil", "bool(false)");
        int uid = 0;
        auto nid = [&](const char* p) { return p + std::to_string(uid++); };
        auto slice = [&](const std::string& src, bool i8, long long r0,
                         long long c0, long long r1, long long c1) {
          const std::string b = nid("sb"), e = nid("se"), s = nid("sl");
          cst("tensor<int32, [2]>", b, i2(r0, c0));
          cst("tensor<int32, [2]>", e, i2(r1, c1));
          line(ty(i8, r1 - r0, c1 - c0) + " " + s +
               " = slice_by_index(begin = " + b + ", end = " + e + ", x = " +
               src + ")[name = string(\"" + s + "\")]");
          return s;
        };
        auto proj = [&](const std::string& src, bool src8, int K, int N,
                        const std::string& w, bool w8) {
          const int nk = (K + T - 1) / T, nn = (N + T - 1) / T;
          std::vector<std::string> cols;
          for (int j = 0; j < nn; ++j) {
            const int n0 = j * T, n = std::min(T, N - n0);
            std::string acc;
            for (int t = 0; t < nk; ++t) {
              const int k0 = t * T, k = std::min(T, K - k0);
              std::string xin = nk > 1 ? slice(src, src8, 0, k0, M, k0 + k)
                                       : src;
              if (src8) {
                const std::string d = nid("xd");
                line(ty(false, M, k) + " " + d + " = dequantize(input = " +
                     xin + ", scale = one)[name = string(\"" + d + "\")]");
                xin = d;
              }
              std::string wt = slice(w, w8, n0, k0, n0 + n, k0 + k);
              if (w8) {
                const std::string d = nid("wq");
                line(ty(false, n, k) + " " + d + " = dequantize(input = " +
                     wt + ", scale = c128)[name = string(\"" + d + "\")]");
                wt = d;
              }
              const std::string mm = nid("mm");
              line(ty(false, M, n) + " " + mm + " = matmul(transpose_x = "
                   "ftx, transpose_y = tty, x = " + xin + ", y = " + wt +
                   ")[name = string(\"" + mm + "\")]");
              if (acc.empty()) {
                acc = mm;
              } else {
                const std::string ad = nid("ad");
                line(ty(false, M, n) + " " + ad + " = add(x = " + acc +
                     ", y = " + mm + ")[name = string(\"" + ad + "\")]");
                acc = ad;
              }
            }
            cols.push_back(acc);
          }
          if (cols.size() == 1u) { return cols[0]; }
          const std::string cc = nid("cc");
          std::string vals;
          for (std::size_t i = 0; i < cols.size(); ++i) {
            vals += (i ? ", " : "") + cols[i];
          }
          line(ty(false, M, N) + " " + cc + " = concat(axis = cax, "
               "interleave = cil, values = (" + vals + "))[name = string(\"" +
               cc + "\")]");
          return cc;
        };
        auto dynq = [&](const std::string& src, int K, std::string* sa) {
          const std::string ab = nid("ab"), am = nid("am"), sc = nid("sc");
          const std::string s = nid("sa"), xs_ = nid("xs"), q = nid("q");
          line(ty(false, M, K) + " " + ab + " = abs(x = " + src +
               ")[name = string(\"" + ab + "\")]");
          line(ty(false, M, 1) + " " + am + " = reduce_max(axes = rax, "
               "keep_dims = kd, x = " + ab + ")[name = string(\"" + am +
               "\")]");
          line(ty(false, M, 1) + " " + sc + " = mul(x = " + am +
               ", y = i127)[name = string(\"" + sc + "\")]");
          line(ty(false, M, 1) + " " + s + " = maximum(x = " + sc +
               ", y = eps)[name = string(\"" + s + "\")]");
          line(ty(false, M, K) + " " + xs_ + " = real_div(x = " + src +
               ", y = " + s + ")[name = string(\"" + xs_ + "\")]");
          line(ty(true, M, K) + " " + q + " = quantize(input = " + xs_ +
               ", output_dtype = adt, scale = one)[name = string(\"" + q +
               "\")]");
          *sa = s;
          return q;
        };
        // The rotation: [M, K] -> [M, K/256, 256], times the orthonormal
        // Hadamard (an input, see hmat), back to [M, K].
        if (md.rot && hinline) {
          std::string lit = "tensor<fp16, [256, 256]>([";
          for (int r = 0; r < 256; ++r) {
            lit += (r ? ", [" : "[");
            for (int c = 0; c < 256; ++c) {
              lit += (c ? ", " : "");
              lit += (__builtin_popcount(r & c) & 1) ? "-0x1p-4" : "0x1p-4";
            }
            lit += "]";
          }
          lit += "])";
          cst("tensor<fp16, [256, 256]>", "hmat", lit);
        }
        auto rot = [&](const std::string& src, int K) {
          const std::string s3 = nid("r3"), a = nid("ra"), r = nid("rb");
          const std::string s2 = nid("r2"), c = nid("rc");
          const std::string d3 = "[" + std::to_string(M) + ", " +
                                 std::to_string(K / 256) + ", 256]";
          cst("tensor<int32, [3]>", s3, "tensor<int32, [3]>(" + d3 + ")");
          line("tensor<fp16, " + d3 + "> " + a + " = reshape(shape = " + s3 +
               ", x = " + src + ")[name = string(\"" + a + "\")]");
          line("tensor<fp16, " + d3 + "> " + r + " = matmul(transpose_x = "
               "ftx, transpose_y = ftx, x = " + a + ", y = hmat)[name = "
               "string(\"" + r + "\")]");
          cst("tensor<int32, [2]>", s2, i2(M, K));
          line(ty(false, M, K) + " " + c + " = reshape(shape = " + s2 +
               ", x = " + r + ")[name = string(\"" + c + "\")]");
          return c;
        };
        auto scale_by = [&](const std::string& src, int N,
                            const std::string& by) {
          const std::string y = nid("ys");
          line(ty(false, M, N) + " " + y + " = mul(x = " + src + ", y = " +
               by + ")[name = string(\"" + y + "\")]");
          return y;
        };
        const std::string x0 = md.rot && md.a_gu != 1 ? rot("x", H) : "x";
        std::string src = x0, tok;
        bool src8 = false;
        if (md.a_gu == 1) {
          src = "x8";
          src8 = true;
          tok = "sa";
        } else if (md.a_gu == 2) {
          src = dynq(x0, H, &tok);
          src8 = true;
        }
        std::string g = proj(src, src8, H, F, "wg", md.w8);
        std::string u = proj(src, src8, H, F, "wu", md.w8);
        if (!tok.empty()) {
          g = scale_by(g, F, tok);
          u = scale_by(u, F, tok);
        }
        if (md.w8) {
          g = scale_by(g, F, "sgw");
          u = scale_by(u, F, "suw");
        }
        // SiLU as the tier's graph states it (the stable two-path form).
        {
          cst("fp16", "s_zero", fp(0.0));
          cst("fp16", "s_onef", fp(1.0));
          cst("fp16", "s_neg", fp(-1.0));
          const std::string TT = ty(false, M, F) + " ";
          line(TT + "s_min = minimum(x = " + g + ", y = s_zero)[name = "
               "string(\"s_min\")]");
          line(TT + "s_num = exp(x = s_min)[name = string(\"s_num\")]");
          line(TT + "s_abs = abs(x = " + g + ")[name = string(\"s_abs\")]");
          line(TT + "s_nab = mul(x = s_abs, y = s_neg)[name = "
               "string(\"s_nab\")]");
          line(TT + "s_e = exp(x = s_nab)[name = string(\"s_e\")]");
          line(TT + "s_den = add(x = s_e, y = s_onef)[name = "
               "string(\"s_den\")]");
          line(TT + "sgm = real_div(x = s_num, y = s_den)[name = "
               "string(\"sgm\")]");
          line(TT + "sg = mul(x = " + g + ", y = sgm)[name = string(\"sg\")]");
          line(TT + "hh = mul(x = sg, y = " + u + ")[name = string(\"hh\")]");
        }
        const std::string h0 = md.rot ? rot("hh", F) : "hh";
        std::string dsrc = h0, dtok;
        bool d8 = false;
        if (md.a_d == 2) {
          dsrc = dynq(h0, F, &dtok);
          d8 = true;
        }
        bt.out = proj(dsrc, d8, F, H, "wd", md.w8);
        if (!dtok.empty()) { bt.out = scale_by(bt.out, H, dtok); }
        if (md.w8) { bt.out = scale_by(bt.out, H, "sdw"); }
        o << "        } -> (" << bt.out << ");\n}";

        const vpipe::AneEmitter::Feature fout{bt.out, {M, H}};
        const std::string dir = root + "/" + name + ".mlmodelc";
        std::string err;
        if (!vpipe::AneEmitter::write_bundle(dir, o.str(), bt.ins,
                                             {&fout, 1}, {}, &err)) {
          std::printf("[ane_rtqf] %s: not written (%s)\n", name.c_str(),
                      err.c_str());
          continue;
        }
        // Each int8 feature's array type, in feature order (FLOAT16 65552
        // -> INT8 131080, three-byte varints after the field's tag 0x10).
        {
          std::fstream f(dir + "/coremldata.bin",
                         std::ios::in | std::ios::out | std::ios::binary);
          std::string b((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
          const std::string pat("\x10\x90\x80\x04", 4);
          std::size_t pos = 0, k = 0;
          while ((pos = b.find(pat, pos)) != std::string::npos) {
            if (k < ins_i8.size() && ins_i8[k]) {
              b[pos + 1] = '\x88';
              b[pos + 2] = '\x80';
              b[pos + 3] = '\x08';
            }
            ++k;
            pos += 4;
          }
          ASSERT_TRUE(k == bt.ins.size() + 1);
          f.clear();
          f.seekp(0);
          f.write(b.data(), (std::streamsize)b.size());
        }
        const auto t0 = Clk::now();
        bt.m = mgr->load(dir, 3);
        bt.compile_ms = ms_since(t0);
        if (bt.m == nullptr || !bt.m->valid()) {
          std::printf("[ane_rtqf] %-10s DID NOT LOAD\n", name.c_str());
          continue;
        }
        built[name] = std::move(bt);
      }
      Built& bt = built[name];

      std::vector<CVPixelBufferRef> pbs;
      std::vector<vpipe::CoreMLPredictInput> pin;
      std::vector<std::vector<std::uint16_t>> host;
      host.reserve(8);
      auto steps = [&](const std::vector<double>& s, double k) {
        std::vector<std::uint16_t> v(s.size());
        for (std::size_t i = 0; i < s.size(); ++i) {
          v[i] = bits16(s[i] * k);
        }
        host.push_back(std::move(v));
        return (const void*)host.back().data();
      };
      for (const auto& f : bt.ins) {
        vpipe::CoreMLPredictInput in;
        in.name  = f.name;
        in.dtype = vpipe::CoreMLDType::F16;
        in.shape = f.shape;
        const Wt* w = f.name == "wg"   ? &cs.wg
                      : f.name == "wu" ? &cs.wu
                      : f.name == "wd" ? &cs.wd
                                       : nullptr;
        if (f.name == "x") {
          in.data = x16.data();
        } else if (f.name == "hmat") {
          in.data = hmat.data();
        } else if (f.name == "x8") {
          pbs.push_back(make_pb(M, H, true, md.rot ? x8r.data() : x8.data()));
          in.pixel_buffer = pbs.back();
        } else if (f.name == "sa") {
          in.data = steps(md.rot ? xsr : xs, 1.0);
        } else if (w != nullptr) {
          const void* src = md.w8 ? (md.rot ? (const void*)w->qr.data()
                                             : (const void*)w->q.data())
                                  : (md.rot ? (const void*)w->hr.data()
                                            : (const void*)w->h.data());
          pbs.push_back(make_pb(w->rows, w->cols, md.w8, src));
          in.pixel_buffer = pbs.back();
        } else {                    // sgw / suw / sdw: the step x 128
          const Wt& sw = f.name == "sgw"   ? cs.wg
                         : f.name == "suw" ? cs.wu
                                           : cs.wd;
          in.data = steps(md.rot ? sw.sr : sw.s, 128.0);
        }
        pin.push_back(std::move(in));
      }
      std::vector<std::uint16_t> yb((std::size_t)M * H);
      vpipe::CoreMLPredictOutput outp;
      outp.name          = bt.out;
      outp.want          = vpipe::CoreMLDType::F16;
      outp.backing       = yb.data();
      outp.backing_elems = yb.size();
      bool ok = bt.m->predict(pin, {&outp, 1});
      std::vector<double> t;
      for (int i = 0; i < iters && ok; ++i) {
        const auto ti = Clk::now();
        ok = bt.m->predict(pin, {&outp, 1});
        t.push_back(ms_since(ti));
      }
      for (CVPixelBufferRef pb : pbs) {
        if (pb != nullptr) { CVPixelBufferRelease(pb); }
      }
      if (!ok || t.empty()) {
        std::printf("[ane_rtqf] %-9s %-15s PREDICT FAILED\n",
                    cs.name.c_str(), name.c_str());
        continue;
      }
      std::sort(t.begin(), t.end());
      double rel = 0, erel = 0;
      score(&yb, md, &rel, &erel);
      report(name, bt.compile_ms, t[t.size() / 2], rel, erel);
      bt.compile_ms = 0.0;
      EXPECT_TRUE(rel < 5e-2);
    }

    // ---- the GPU's token tiers, emulated over the same rows --------------
    for (const Mode* em : {&emu_plain, &emu_rot}) {
      if (only != nullptr && std::string(em->name).find(only) ==
                                 std::string::npos) {
        continue;
      }
      double rel = 0, erel = 0;
      score(nullptr, *em, &rel, &erel);
      report(em->name, 0.0, 0.0, rel, erel);
    }
  }
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

  // An int8 tier this host does not offer is refused by name, never
  // built as another (the grouped module is the machinery's, not the
  // spec's).
  if (mc != nullptr && mc->valid()) {
    why.clear();
    auto fast = tier_spec_("ffn", "", 256, 512, 768, 256);
    acc::set_flag(&fast, ane::kInt8, true);
    acc::set_text(&fast, ane::kInt8Tier, "fast");
    EXPECT_TRUE(ane::create(&sess, mc, fast, &why) == nullptr);
    EXPECT_TRUE(why.find("unknown int8_tier 'fast'") != std::string::npos);
  }

  // The byte counts are the machinery's own, per kind.
  EXPECT_TRUE(ane::runtime_bytes(
                  tier_spec_("ffn", "gelu_tanh", 64, 256, 768, 256)) ==
              vpipe::genai::AneFeedForward::gelu_runtime_bytes(64, 256, 256));
  EXPECT_TRUE(ane::runtime_bytes(tier_spec_("ffn", "", 64, 256, 768, 256)) ==
              vpipe::genai::AneFeedForward::runtime_bytes(64, 256, 768, 256));
}

// The spec's int8 tier reaches the module and info() names it back:
// "token" (the default) rotated, "token_plain" unrotated -- where the OS
// takes int8 arrays (macOS 26); elsewhere both build fp16 and info() says
// int8 false.
TEST(ane_tier, the_spec_names_its_int8_tier)
{
  namespace ane = vpipe::genai::ane;
  namespace acc = vpipe::genai::accel;
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  for (const char* t : {"", "token", "token_plain"}) {
    auto spec = tier_spec_("ffn", "", 1024, 2048, 1040, 512);
    acc::set_flag(&spec, ane::kInt8, true);
    if (*t != '\0') { acc::set_text(&spec, ane::kInt8Tier, std::string(t)); }
    std::string why;
    auto tier = ane::create(&sess, mc, spec, &why);
    if (tier == nullptr) { return; }                // no ANE on this box
    const vpipe::FlexData inf = tier->info();
    if (!acc::flag(&inf, ane::kInt8)) {
      std::printf("[ane_tier] no int8 arrays here: int8 tiers skipped\n");
      EXPECT_TRUE(!acc::has(&inf, ane::kInt8Tier));
      return;
    }
    const std::string want =
        std::string(t) == "token_plain" ? "token_plain" : "token";
    EXPECT_TRUE(acc::text(&inf, ane::kInt8Tier) == want);
  }
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

  // ane::kMaxAdapters slots a projection: the LAST one's names are known
  // -- refused here only for the missing .rank, which is checked after
  // every name -- and one past it is refused by name.
  EXPECT_TRUE(ane::kMaxAdapters == 8);
  const std::string l7a = ane::key(ane::kProjUp, "lora7.a");
  const std::string l7b = ane::key(ane::kProjUp, "lora7.b");
  const std::string l8a = ane::key(ane::kProjUp, "lora8.a");
  const ane::Binding last[] = {{l7a, &wu}, {l7b, &wu}};
  EXPECT_FALSE(tier->stage(0, params, last));
  const vpipe::FlexData inf7 = tier->info();
  const std::string e7 = acc::text(&inf7, ane::kInfoError);
  EXPECT_TRUE(e7.find("unknown binding") == std::string::npos &&
              e7.find("up.lora7") != std::string::npos);
  const ane::Binding past[] = {{l8a, &wu}};
  EXPECT_FALSE(tier->stage(0, params, past));
  const vpipe::FlexData inf8 = tier->info();
  EXPECT_TRUE(acc::text(&inf8, ane::kInfoError).find("up.lora8.a") !=
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

// perf-flops: every predict the tier runs is credited to the ANE, at the
// FLOPs the same rows would cost the GPU -- three projections a row for a
// SwiGLU feed-forward, two for a GELU one, one for a matmul tier -- so a
// family moving rows between the two moves no FLOPs.
TEST(ane_tier, credits_its_predicts_to_the_flops_meter)
{
  namespace pf = vpipe::perf_flops;
  using vpipe::genai::AneFeedForward;
  using vpipe::genai::AneFfnSource;
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  constexpr int H = 256, F = 512, seq = 768, chunk = 256;
  std::mt19937 rng(5);
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
  const float su = 1.0f / std::sqrt((float)H);
  const float sd = 1.0f / std::sqrt((float)F);
  const auto wg = bf16((std::size_t)F * H, su);
  const auto wu = bf16((std::size_t)F * H, su);
  const auto wd = bf16((std::size_t)H * F, sd);
  const auto x  = bf16((std::size_t)seq * H, 0.5f);
  auto y = mc->make_shared_buffer((std::size_t)seq * F * 2);

  enum Kind { kSwiglu, kGelu, kMatmul };
  for (Kind k : {kSwiglu, kGelu, kMatmul}) {
    AneFeedForward::Options o;
    o.session = &sess;
    o.mc      = mc;
    o.tag     = "flops";
    o.hidden  = H;
    o.ffn     = F;
    o.seq     = seq;
    o.rows    = 1.0f;
    o.chunk   = chunk;
    o.gelu    = k == kGelu;
    o.matmul  = k == kMatmul;
    auto ff = AneFeedForward::create(o);
    if (ff == nullptr) { return; }               // no ANE on this box
    AneFfnSource g, u, d;
    g.w = &wg;
    u.w = &wu;
    d.w = &wd;
    if (k == kSwiglu) {
      ff->stage(0, g, u, d, 0);
    } else if (k == kGelu) {
      ff->stage(0, std::vector<AneFfnSource>{u, d}, 0);
    } else {
      ff->stage(0, u, 0);
    }
    ASSERT_TRUE(ff->join_stage(0));
    const pf::Totals t0 = pf::totals(pf::kAne);
    const int a = ff->begin(x, y, seq);
    ASSERT_TRUE(ff->finish(0, seq, 0.0, 0.0) && a > 0);
    const std::uint64_t per_row =
        2ull * H * F * (k == kMatmul ? 1 : (k == kGelu ? 2 : 3));
    const pf::Totals t1 = pf::totals(pf::kAne);
    std::printf("  %s: %d rows on the ANE, %llu FLOPs credited in %llu "
                "predicts\n",
                k == kSwiglu ? "swiglu" : (k == kGelu ? "gelu" : "matmul"),
                a, (unsigned long long)(t1.flops - t0.flops),
                (unsigned long long)(t1.records - t0.records));
    EXPECT_TRUE(t1.flops - t0.flops == (std::uint64_t)a * per_row);
    EXPECT_TRUE(t1.records - t0.records == (std::uint64_t)(a / chunk));
  }
}

// INTEGER SOURCES STAGE AS THEIR WIDENING. ComfyUI's int8 (per-tensor or
// per-row scale, ConvRot or not) and W4A8 (with a codebook and without)
// staged straight from their codes must put in the slots EXACTLY what a
// dense bf16 source holding the GPU's widening of the same codes puts
// there (fp8-layout.h's decoders are that widening to the bit) -- so the
// ANE's rows come out bit-identical. The f32 path too: an adapter folded
// into both arms. And the control that keeps it from passing vacuously:
// the same codes staged as if unrotated come out different.
TEST(ane_tier, integer_sources_stage_as_their_widening)
{
  namespace fp8 = vpipe::genai::fp8;
  using vpipe::genai::AneFeedForward;
  using vpipe::genai::AneFfnSource;
  using vpipe::metal_compute::SharedBuffer;
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  // K 1024: every ConvRot group up to 1024 divides it.
  constexpr int K = 1024, N = 512, seq = 768, chunk = 256;
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto bf16_of = [](float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    return (std::uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  };
  auto bf16 = [&](std::size_t n, float sc) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { p[i] = bf16_of(nd(rng) * sc); }
    return b;
  };
  auto bytes = [&](std::size_t n, int lo, int hi) {
    SharedBuffer b = mc->make_shared_buffer(n);
    std::uniform_int_distribution<int> d(lo, hi);
    auto* p = static_cast<std::uint8_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { p[i] = (std::uint8_t)d(rng); }
    return b;
  };
  auto floats = [&](const std::vector<float>& v) {
    SharedBuffer b = mc->make_shared_buffer(v.size() * sizeof(float));
    std::memcpy(b.contents(), v.data(), v.size() * sizeof(float));
    return b;
  };
  const SharedBuffer x = bf16((std::size_t)seq * K, 0.5f);
  const SharedBuffer la = bf16((std::size_t)8 * K, 0.05f);   // A [8, K]
  const SharedBuffer lb = bf16((std::size_t)N * 8, 0.05f);   // B [N, 8]

  AneFeedForward::Options o;
  o.session = &sess;
  o.mc      = mc;
  o.tag     = "int-src";
  o.hidden  = K;
  o.ffn     = N;
  o.seq     = seq;
  o.rows    = 1.0f;
  o.chunk   = chunk;
  o.matmul  = true;
  auto mm = AneFeedForward::create(o);
  if (mm == nullptr) { return; }                 // no ANE on this box
  int layer = 0;
  // The ANE's rows of a forward with `s` staged.
  auto run = [&](AneFfnSource s, bool adapted) {
    if (adapted) {
      s.deltas = 1;
      s.delta[0].a = &la;
      s.delta[0].b = &lb;
      s.delta[0].rank = 8;
      s.delta[0].scale = 0.75f;
    }
    SharedBuffer y = mc->make_shared_buffer((std::size_t)seq * N * 2);
    std::memset(y.contents(), 0, y.byte_size());
    mm->stage(++layer, s, 0);
    std::vector<std::uint16_t> out;
    if (!mm->join_stage(layer)) { return out; }
    const int a = mm->begin(x, y, seq);
    if (!mm->finish(layer, seq, 0.0, 0.0) || a <= 0) { return out; }
    const auto* p = static_cast<const std::uint16_t*>(y.contents());
    out.assign(p + (std::size_t)(seq - a) * N, p + (std::size_t)seq * N);
    return out;
  };
  struct Case {
    const char* name;
    bool w4a8;
    int  rot;
    bool per_row;
    int  qgroup;
    bool codebook;
  };
  const Case cases[] = {
      {"int8 per-tensor, unrotated", false, 0, false, 0, false},
      {"int8 per-row, ConvRot 64", false, 64, true, 0, false},
      {"int8 per-row, ConvRot 256", false, 256, true, 0, false},
      {"int8 per-tensor, ConvRot 1024", false, 1024, false, 0, false},
      {"W4A8 codebook, group 16, ConvRot 256", true, 256, true, 16, true},
      {"W4A8 code - 8, group 32, ConvRot 64", true, 64, true, 32, false},
      {"W4A8 codebook, group 16, unrotated", true, 0, true, 16, true},
  };
  for (const Case& c : cases) {
    std::uniform_real_distribution<float> us(0.0005f, 0.004f);
    std::vector<float> sc(c.per_row ? (std::size_t)N : 1);
    for (float& v : sc) { v = us(rng); }
    const SharedBuffer scb = floats(sc);
    std::vector<std::uint16_t> wide((std::size_t)N * K);
    SharedBuffer codes, srel, cbb;
    std::vector<float> cbv;
    if (!c.w4a8) {
      codes = bytes((std::size_t)N * K, 0, 255);
      // -128 is not a code ComfyUI writes; +-127 is the range.
      auto* cp = static_cast<std::uint8_t*>(codes.contents());
      for (std::size_t i = 0; i < (std::size_t)N * K; ++i) {
        if (cp[i] == 0x80) { cp[i] = 0x81; }
      }
      fp8::decode_bf16(cp, wide.data(), N, K, fp8::Format::kI8, sc, c.rot);
    } else {
      codes = bytes((std::size_t)N * K / 2, 0, 255);
      // E4M3 relative scales from 2^-3 to ~2^4: codes 0x20..0x58.
      srel = bytes((std::size_t)N * (K / c.qgroup), 0x20, 0x58);
      if (c.codebook) {
        std::uniform_real_distribution<float> uc(-8.0f, 8.0f);
        cbv.resize(16);
        for (float& v : cbv) { v = uc(rng); }
        cbb = floats(cbv);
      }
      fp8::decode_w4a8_bf16(
          static_cast<const std::uint8_t*>(codes.contents()),
          static_cast<const std::uint8_t*>(srel.contents()),
          c.codebook ? cbv.data() : nullptr, sc, N, K, c.qgroup, c.rot,
          wide.data());
    }
    SharedBuffer dense = mc->make_shared_buffer(wide.size() * 2);
    std::memcpy(dense.contents(), wide.data(), wide.size() * 2);
    AneFfnSource ref;
    ref.w = &dense;
    AneFfnSource src;
    src.w        = &codes;
    src.bits     = c.w4a8 ? AneFfnSource::kW4A8 : AneFfnSource::kInt8;
    src.scales   = &scb;
    src.rot      = c.rot;
    src.srel     = c.w4a8 ? &srel : nullptr;
    src.codebook = c.codebook ? &cbb : nullptr;
    src.qgroup   = c.qgroup;
    ASSERT_TRUE(src.stageable());
    for (bool adapted : {false, true}) {
      const std::vector<std::uint16_t> want = run(ref, adapted);
      const std::vector<std::uint16_t> got  = run(src, adapted);
      ASSERT_TRUE(!want.empty() && want.size() == got.size());
      const bool same = !want.empty() && want == got;
      std::printf("  %-38s%s: %s\n", c.name, adapted ? " +adapter" : "",
                  same ? "identical" : "DIFFERENT");
      EXPECT_TRUE(same);
    }
    if (c.rot > 0) {
      AneFfnSource unrot = src;
      unrot.rot = 0;
      const std::vector<std::uint16_t> want = run(ref, false);
      const std::vector<std::uint16_t> got  = run(unrot, false);
      EXPECT_TRUE(!got.empty() && want != got);
    }
  }
  // Refused, not misread: a rotation that is not a power of four, a W4A8
  // source without its relative scales, an int8 one without its scale.
  AneFfnSource bad;
  SharedBuffer c8 = bytes((std::size_t)N * K, 0, 255);
  SharedBuffer s1 = floats({0.01f});
  bad.w = &c8;
  bad.bits = AneFfnSource::kInt8;
  bad.scales = &s1;
  bad.rot = 128;
  EXPECT_FALSE(bad.stageable());
  bad.rot = 0;
  EXPECT_TRUE(bad.stageable());
  bad.scales = nullptr;
  EXPECT_FALSE(bad.stageable());
  bad.scales = &s1;
  bad.bits = AneFfnSource::kW4A8;
  bad.qgroup = 16;
  EXPECT_FALSE(bad.stageable());
}

// THE ROTATED INT8 TIER (Options::int8). A SwiGLU and a single projection
// with OUTLIER channels planted in their inputs and weights -- what makes
// the rotation matter -- through the tier's own stage / begin / finish,
// against double restatements of the graph's arithmetic (rows rotated per
// 256-block and quantized per row, inputs rotated and quantized per token
// in fp16), of the same WITHOUT the rotation, and of the unquantized
// result. Then an input large enough to overflow fp16 inside the graph:
// the headroom retry rescales the up projection's steps, not its codes,
// and the rows still match. Skips where CoreML has no int8 arrays (before
// macOS 26), as create() falls back to fp16 there.
TEST(ane_tier, rotated_int8_runs_the_token_tiers_arithmetic)
{
  using vpipe::genai::AneFeedForward;
  using vpipe::genai::AneFfnSource;
  using vpipe::metal_compute::SharedBuffer;
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  constexpr int K = 1024, chunk = 512, seq = 2 * chunk + 16;
  std::mt19937 rng(23);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto bf16_of = [](float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    return (std::uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  };
  auto f32_of = [](std::uint16_t b) {
    const std::uint32_t u = (std::uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  auto h16 = [](double v) { return (double)(_Float16)v; };
  // [rows, cols] bf16, N(0, sc), the first four columns `boost` times
  // larger: the outlier channels.
  auto mat = [&](int rows, int cols, float sc, float boost) {
    SharedBuffer b = mc->make_shared_buffer((std::size_t)rows * cols * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        p[(std::size_t)r * cols + c] =
            bf16_of(nd(rng) * sc * (c % 251 < 4 ? boost : 1.0f));
      }
    }
    return b;
  };
  auto fwht = [](double* v, std::size_t n) {
    for (std::size_t b0 = 0; b0 + 256 <= n; b0 += 256) {
      double* p = v + b0;
      for (std::size_t h = 1; h < 256; h <<= 1) {
        for (std::size_t i = 0; i < 256; i += 2 * h) {
          for (std::size_t j = i; j < i + h; ++j) {
            const double a = p[j], c = p[j + h];
            p[j] = a + c;
            p[j + h] = a - c;
          }
        }
      }
      for (std::size_t i = 0; i < 256; ++i) { p[i] /= 16.0; }
    }
  };
  // A weight as the graph reads it: rows (rotated or not) quantized per
  // row, each code times its fp16 step (128 x the step, over 128).
  auto wq = [&](const SharedBuffer& w, int rows, int cols, bool rot) {
    const auto* p = static_cast<const std::uint16_t*>(w.contents());
    std::vector<double> out((std::size_t)rows * cols), row((std::size_t)cols);
    for (int r = 0; r < rows; ++r) {
      double am = 0.0;
      for (int c = 0; c < cols; ++c) {
        row[(std::size_t)c] = f32_of(p[(std::size_t)r * cols + c]);
      }
      if (rot) { fwht(row.data(), row.size()); }
      for (double v : row) { am = std::max(am, std::fabs(v)); }
      const double st = (double)(float)((float)am / 127.0f);
      const double st128 = h16(st * 128.0) / 128.0;
      for (int c = 0; c < cols; ++c) {
        const double q =
            std::clamp(std::nearbyint(row[(std::size_t)c] / st), -127.0,
                       127.0);
        out[(std::size_t)r * cols + c] = q * st128;
      }
    }
    return out;
  };
  // An activation as the graph quantizes it: per token, its step in fp16.
  // (Past fp16's range the graph has rescaled the up projection by then,
  // and the quantizer does not care about scale: the max stays exact.)
  auto aq = [&](std::vector<double>& v) {
    double am = 0.0;
    for (double e : v) {
      am = std::max(am, std::fabs(e) < 6.0e4 ? std::fabs(h16(e))
                                             : std::fabs(e));
    }
    const double sc = am / 127.0 < 6.0e4 ? h16(am * h16(1.0 / 127.0))
                                         : am / 127.0;
    const double st = std::max(sc, h16(1e-4));
    for (double& e : v) {
      e = std::clamp(std::nearbyint(e / st), -128.0, 127.0) * st;
    }
  };
  auto proj = [](const std::vector<double>& v, const std::vector<double>& w,
                 int rows, int cols) {
    std::vector<double> y((std::size_t)rows, 0.0);
    for (int n = 0; n < rows; ++n) {
      double acc = 0.0;
      for (int c = 0; c < cols; ++c) {
        acc += w[(std::size_t)n * cols + c] * v[(std::size_t)c];
      }
      y[(std::size_t)n] = acc;
    }
    return y;
  };
  auto rel = [](const std::vector<std::vector<double>>& a,
                const std::vector<std::vector<double>>& b) {
    double n = 0.0, d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      for (std::size_t j = 0; j < a[i].size(); ++j) {
        const double e = a[i][j] - b[i][j];
        n += e * e;
        d += b[i][j] * b[i][j];
      }
    }
    return std::sqrt(n / std::max(d, 1e-300));
  };

  int layer = 0;
  for (const bool mm : {false, true}) {
    const int F = mm ? 1536 : 2048;          // the output / inner width
    AneFeedForward::Options o;
    o.session = &sess;
    o.mc      = mc;
    o.tag     = mm ? "i8-mm" : "i8-ffn";
    o.hidden  = K;
    o.ffn     = F;
    o.seq     = seq;
    o.rows    = 1.0f;
    o.chunk   = chunk;
    o.matmul  = mm;
    o.int8    = true;
    auto ff = AneFeedForward::create(o);
    if (ff == nullptr) { return; }                 // no ANE on this box
    if (!ff->int8()) {
      std::printf("[ane_tier] no int8 arrays here: rotated int8 skipped\n");
      return;
    }
    const float a = 1.0f / std::sqrt((float)K);
    const SharedBuffer wg = mat(F, K, a, 8.0f), wu = mat(F, K, a, 8.0f);
    const SharedBuffer wd = mat(K, F, 1.0f / std::sqrt((float)F), 8.0f);
    AneFfnSource sg, su, sd;
    sg.w = &wg;
    su.w = &wu;
    sd.w = &wd;
    const int Nout = mm ? F : K;
    // The references' weights, rotated and not.
    const std::vector<double> rg = wq(wg, F, K, true), ru = wq(wu, F, K, true);
    const std::vector<double> ng = wq(wg, F, K, false);
    const std::vector<double> nu = wq(wu, F, K, false);
    const std::vector<double> rd = mm ? std::vector<double>{}
                                      : wq(wd, K, F, true);
    const std::vector<double> nd_ = mm ? std::vector<double>{}
                                       : wq(wd, K, F, false);
    // 6: the hidden state reaches ~3e5, past fp16 until the up
    // projection is rescaled (by 16).
    for (const float xs : {0.5f, 6.0f}) {
      const SharedBuffer x = mat(seq, K, xs, 24.0f);
      SharedBuffer y = mc->make_shared_buffer((std::size_t)seq * Nout * 2);
      std::memset(y.contents(), 0, y.byte_size());
      if (mm) {
        ff->stage(++layer, sg, 0);
      } else {
        ff->stage(++layer, sg, su, sd, 0);
      }
      ASSERT_TRUE(ff->join_stage(layer));
      const int an = ff->begin(x, y, seq);
      const bool ok = ff->finish(layer, seq, 0.0, 0.0);
      ASSERT_TRUE(ok && an > 0);
      if (!ok || an <= 0) { return; }
      const auto* xp = static_cast<const std::uint16_t*>(x.contents());
      const auto* yp = static_cast<const std::uint16_t*>(y.contents());
      std::vector<std::vector<double>> got, rot, plain, exact;
      for (int i = 0; i < 6; ++i) {
        const int r = seq - an + i * (an - 1) / 5;
        std::vector<double> xe((std::size_t)K);
        for (int c = 0; c < K; ++c) {
          xe[(std::size_t)c] = f32_of(xp[(std::size_t)r * K + c]);
        }
        std::vector<double> g8;
        for (int c = 0; c < Nout; ++c) {
          g8.push_back(f32_of(yp[(std::size_t)r * Nout + c]));
        }
        got.push_back(g8);
        // The three arithmetics over this row.
        auto run = [&](int mode) {             // 0 exact, 1 rotated, 2 plain
          std::vector<double> v((std::size_t)K);
          for (int c = 0; c < K; ++c) {
            v[(std::size_t)c] = mode == 0 ? xe[(std::size_t)c]
                                          : h16(xe[(std::size_t)c]);
          }
          std::vector<double> wgx, wux, wdx;
          if (mode == 0) {
            auto dense = [&](const SharedBuffer& w, int rr, int cc) {
              const auto* p = static_cast<const std::uint16_t*>(w.contents());
              std::vector<double> o2((std::size_t)rr * cc);
              for (std::size_t k = 0; k < o2.size(); ++k) {
                o2[k] = f32_of(p[k]);
              }
              return o2;
            };
            wgx = dense(wg, F, K);
            if (!mm) {
              wux = dense(wu, F, K);
              wdx = dense(wd, K, F);
            }
          }
          const std::vector<double>& G = mode == 0 ? wgx : mode == 1 ? rg : ng;
          const std::vector<double>& U = mode == 0 ? wux : mode == 1 ? ru : nu;
          const std::vector<double>& D = mode == 0 ? wdx : mode == 1 ? rd : nd_;
          if (mode == 1) { fwht(v.data(), v.size()); }
          if (mode != 0) { aq(v); }
          std::vector<double> g = proj(v, G, F, K);
          if (mm) { return g; }
          const std::vector<double> u = proj(v, U, F, K);
          for (int n = 0; n < F; ++n) {
            const double gg = g[(std::size_t)n];
            g[(std::size_t)n] = gg / (1.0 + std::exp(-gg)) * u[(std::size_t)n];
          }
          if (mode == 1) { fwht(g.data(), g.size()); }
          if (mode != 0) { aq(g); }
          return proj(g, D, K, F);
        };
        exact.push_back(run(0));
        rot.push_back(run(1));
        plain.push_back(run(2));
      }
      const double own = rel(got, rot), e_ane = rel(got, exact);
      const double e_rot = rel(rot, exact), e_plain = rel(plain, exact);
      std::printf("[ane_tier] rotated int8 %s, x * %.1f: against its "
                  "arithmetic %.2e; from unquantized ANE %.2e, rotated "
                  "%.2e, unrotated %.2e\n", mm ? "matmul" : "SwiGLU", xs,
                  own, e_ane, e_rot, e_plain);
      // The graph computes what was restated (fp16 inside, so not to the
      // bit); the rotation is worth something on outlier channels; and
      // the ANE adds little to the rotated arithmetic's own error.
      EXPECT_TRUE(own < 2.5e-2);
      EXPECT_TRUE(e_rot < 0.7 * e_plain);
      EXPECT_TRUE(e_ane < 1.3 * e_rot + 3e-3);
      if (mm) { break; }          // a single matmul has no headroom retry
    }
  }
}

// THE INT8 TIER AGAINST THE GPU'S INT8 TIERS, on real activations. The
// tier's rotated int8 module, its unrotated twin (Options::int8_plain), its
// grouped one (Options::int8_group, the `fast` tier's arithmetic) and its
// fp16 module run real feed-forward inputs (a forward's capture,
// VPIPE_H3_FF_CAPTURE) through a real block's weights, beside the GPU's
// `token` and `token_plain` tiers restated as their kernels compute
// (quant_i8_tok / quant_i8_tok_r and the weight converters: f32 values,
// the 256-block Walsh-Hadamard in f32, a row's step am / 127 in f32, codes
// rint and clamped to +-127, one integer sum a dot product times the two
// f32 steps, every projection's output and the SwiGLU's hidden state
// rounded to bf16 -- the restatement gemm_i8.per_token pins to the
// kernels) and the unquantized feed-forward in double. "Close" is the
// ANE's distance from its GPU tier against that tier's own distance from
// exact: the same arithmetic, rounded in fp16 where the GPU keeps f32.
// The `fast` tier restated as its kernels compute too (quant_i8_row_gc*_r,
// gemm_i8c_*): each group of K rotated whole, a step per (row, group)
// stored in the element type, a group's integer sum times both steps
// folded in f32. Each module's second predict is timed.
//
// Env: VPIPE_ANE_I8_VS_GPU=1; VPIPE_ANE_RTQF_REAL (the capture's
// directory: ff_in_<L>.bf16, [rows, hidden]) and VPIPE_ANE_RTQF_CKPT (the
// checkpoint that forward ran, one safetensors file with
// blocks.<L>.mlp.fc1 = [gate; up] and fc2); _BLOCKS ("0,12,24,36,49"),
// _ROWS (rows compared a block, 48).
TEST(ane_tier, int8_rows_against_the_gpu_token_tiers)
{
  using vpipe::genai::AneFeedForward;
  using vpipe::genai::AneFfnSource;
  using vpipe::metal_compute::SharedBuffer;
  if (env_or_null_("VPIPE_ANE_I8_VS_GPU") == nullptr) { return; }
  const char* real = env_or_null_("VPIPE_ANE_RTQF_REAL");
  const char* ckpt = env_or_null_("VPIPE_ANE_RTQF_CKPT");
  if (real == nullptr || ckpt == nullptr) {
    std::printf("[ane_i8_vs_gpu] needs VPIPE_ANE_RTQF_REAL and _CKPT\n");
    return;
  }
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const char* bl = env_or_null_("VPIPE_ANE_RTQF_BLOCKS");
  std::vector<int> blocks;
  {
    std::stringstream ss(bl != nullptr ? bl : "0,12,24,36,49");
    std::string t;
    while (std::getline(ss, t, ',')) {
      if (!t.empty()) { blocks.push_back(std::atoi(t.c_str())); }
    }
  }
  const char* rs = env_or_null_("VPIPE_ANE_RTQF_ROWS");
  const int want_rows = rs != nullptr ? std::max(1, std::atoi(rs)) : 48;

  vpipe::FlexData hdr;
  std::uint64_t data0 = 0;
  {
    std::ifstream f(ckpt, std::ios::binary);
    std::uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    std::string js(n, '\0');
    f.read(js.data(), (std::streamsize)n);
    ASSERT_TRUE(f.good());
    if (!f.good()) { return; }
    hdr = vpipe::FlexData::from_json(js);
    data0 = 8 + n;
  }
  // A bf16 tensor's bytes as stored, [rows, cols].
  auto raw16 = [&](const std::string& name, int* rows, int* cols) {
    std::vector<std::uint16_t> raw;
    auto o = hdr.as_object();
    if (!o.contains(name)) { return raw; }
    const vpipe::FlexData t = o.at(name);
    auto to = t.as_object();
    const vpipe::FlexData shape = to.at("shape");
    const vpipe::FlexData offs = to.at("data_offsets");
    auto sa = shape.as_array();
    auto oa = offs.as_array();
    *rows = (int)sa.at(0).as_int();
    *cols = (int)sa.at(1).as_int();
    raw.resize((std::size_t)(*rows) * (std::size_t)(*cols));
    std::ifstream f(ckpt, std::ios::binary);
    f.seekg((std::streamoff)(data0 + (std::uint64_t)oa.at(0).as_int()));
    f.read(reinterpret_cast<char*>(raw.data()),
           (std::streamsize)(raw.size() * 2));
    if (!f.good()) { raw.clear(); }
    return raw;
  };
  auto bf = [](std::uint16_t b) {
    const std::uint32_t u = (std::uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  auto rbf = [&](float f) {      // f rounded to bf16, nearest-even
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
  };
  // ...or to f16: the token tier on an f16 model rounds there instead.
  auto rh = [](float f) { return (float)(_Float16)f; };
  // The orthonormal Walsh-Hadamard over every block of b (256 the token
  // tier's), in f32.
  auto fwht = [](float* v, std::size_t n, int b = 256) {
    const float norm = 1.0f / std::sqrt((float)b);
    for (std::size_t b0 = 0; b0 + (std::size_t)b <= n; b0 += b) {
      float* p = v + b0;
      for (int h = 1; h < b; h <<= 1) {
        for (int i = 0; i < b; i += 2 * h) {
          for (int j = i; j < i + h; ++j) {
            const float a = p[j], c = p[j + h];
            p[j] = a + c;
            p[j + h] = a - c;
          }
        }
      }
      for (int i = 0; i < b; ++i) { p[i] *= norm; }
    }
  };
  // A row to its codes and per-group steps (am / 127, unrounded: the
  // element type rounds them at use), as quant_i8_row_gc*_r writes it.
  auto qrow_g = [&](std::vector<float> v, int gc, std::int8_t* q,
                    float* st) {
    fwht(v.data(), v.size(), gc);
    for (std::size_t g0 = 0, gi = 0; g0 < v.size(); g0 += gc, ++gi) {
      float am = 0.0f;
      for (std::size_t k = g0; k < g0 + gc; ++k) {
        am = std::max(am, std::fabs(v[k]));
      }
      const float inv = am > 0.0f ? 127.0f / am : 0.0f;
      for (std::size_t k = g0; k < g0 + gc; ++k) {
        q[k] = (std::int8_t)std::clamp((int)std::rint(v[k] * inv), -127,
                                       127);
      }
      st[gi] = am * (1.0f / 127.0f);
    }
  };
  // A row to its codes and step, as the token quantizers write it.
  auto qrow = [&](std::vector<float> v, bool rot, std::int8_t* q) {
    if (rot) { fwht(v.data(), v.size()); }
    float am = 0.0f;
    for (float e : v) { am = std::max(am, std::fabs(e)); }
    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
    for (std::size_t k = 0; k < v.size(); ++k) {
      q[k] = (std::int8_t)std::clamp((int)std::rint(v[k] * inv), -127, 127);
    }
    return am * (1.0f / 127.0f);
  };
  auto par = [](int n, const std::function<void(int)>& fn) {
    const int T = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    std::atomic<int> next{0};
    for (int t = 0; t < T; ++t) {
      th.emplace_back([&]() {
        for (int i = next++; i < n; i = next++) { fn(i); }
      });
    }
    for (auto& t : th) { t.join(); }
  };
  // A weight's codes and steps, rotated or not (per row), or per group
  // of gc (s then [rows, cols / gc]).
  struct Q {
    std::vector<std::int8_t> c;
    std::vector<float> s;
  };
  auto qmat_g = [&](const std::uint16_t* w, int rows, int cols, int gc) {
    Q m;
    m.c.resize((std::size_t)rows * cols);
    m.s.resize((std::size_t)rows * (cols / gc));
    par(rows, [&](int r) {
      std::vector<float> v((std::size_t)cols);
      for (int k = 0; k < cols; ++k) {
        v[(std::size_t)k] = bf(w[(std::size_t)r * cols + k]);
      }
      qrow_g(std::move(v), gc, m.c.data() + (std::size_t)r * cols,
             m.s.data() + (std::size_t)r * (cols / gc));
    });
    return m;
  };
  auto qmat = [&](const std::uint16_t* w, int rows, int cols, bool rot) {
    Q m;
    m.c.resize((std::size_t)rows * cols);
    m.s.resize((std::size_t)rows);
    par(rows, [&](int r) {
      std::vector<float> v((std::size_t)cols);
      for (int k = 0; k < cols; ++k) {
        v[(std::size_t)k] = bf(w[(std::size_t)r * cols + k]);
      }
      m.s[(std::size_t)r] = qrow(std::move(v), rot,
                                 m.c.data() + (std::size_t)r * cols);
    });
    return m;
  };
  auto rel = [](const std::vector<std::vector<double>>& a,
                const std::vector<std::vector<double>>& b) {
    double n = 0.0, d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      for (std::size_t j = 0; j < a[i].size(); ++j) {
        const double e = a[i][j] - b[i][j];
        n += e * e;
        d += b[i][j] * b[i][j];
      }
    }
    return std::sqrt(n / std::max(d, 1e-300));
  };

  int layer = 0;
  for (const int blk : blocks) {
    const std::string p = "blocks." + std::to_string(blk) + ".mlp.";
    int r1 = 0, c1 = 0, r2 = 0, c2 = 0;
    const std::vector<std::uint16_t> fc1 = raw16(p + "fc1.weight", &r1, &c1);
    const std::vector<std::uint16_t> fc2 = raw16(p + "fc2.weight", &r2, &c2);
    ASSERT_TRUE(!fc1.empty() && !fc2.empty());
    if (fc1.empty() || fc2.empty()) { return; }
    const int H = c1, F = r1 / 2;
    ASSERT_TRUE(r2 == H && c2 == F);
    std::vector<std::uint16_t> xr;
    {
      const std::string xp =
          std::string(real) + "/ff_in_" + std::to_string(blk) + ".bf16";
      std::ifstream xf(xp, std::ios::binary | std::ios::ate);
      ASSERT_TRUE(xf.good());
      if (!xf.good()) { return; }
      xr.resize((std::size_t)xf.tellg() / 2);
      xf.seekg(0);
      xf.read(reinterpret_cast<char*>(xr.data()),
              (std::streamsize)(xr.size() * 2));
    }
    const int R = (int)(xr.size() / (std::size_t)H);
    // fc1 is [gate; up], gate first.
    auto buf = [&](const std::uint16_t* s, std::size_t n) {
      SharedBuffer b = mc->make_shared_buffer(n * 2);
      std::memcpy(b.contents(), s, n * 2);
      return b;
    };
    const std::size_t FH = (std::size_t)F * H;
    const SharedBuffer wg = buf(fc1.data(), FH), wu = buf(fc1.data() + FH, FH);
    const SharedBuffer wd = buf(fc2.data(), FH);
    constexpr int kChunk = 2048;
    const int seq = kChunk + 64;
    SharedBuffer x = mc->make_shared_buffer((std::size_t)seq * H * 2);
    for (int r = 0; r < seq; ++r) {
      std::memcpy(static_cast<std::uint8_t*>(x.contents()) +
                      (std::size_t)r * H * 2,
                  xr.data() + (std::size_t)(r % R) * H, (std::size_t)H * 2);
    }
    // The modules: rotated int8, unrotated int8, grouped int8, fp16.
    enum { kRot, kPlain, kFast, kF16, kMods };
    const char* const kName[kMods] = {"rotated int8", "unrotated int8",
                                      "grouped int8", "fp16"};
    constexpr int kTarget = 512;     // the fast tier's default group
    double ms[kMods] = {};
    std::vector<std::vector<double>> got[kMods];
    std::vector<int> sample;
    bool ran[kMods] = {};
    for (int m = 0; m < kMods; ++m) {
      AneFeedForward::Options o;
      o.session    = &sess;
      o.mc         = mc;
      o.tag        = std::string("i8-vs-gpu-") + std::to_string(m);
      o.hidden     = H;
      o.ffn        = F;
      o.seq        = seq;
      o.rows       = 1.0f;
      o.chunk      = kChunk;
      o.int8       = m != kF16;
      o.int8_plain = m == kPlain;
      o.int8_group = m == kFast ? kTarget : 0;
      auto ff = AneFeedForward::create(o);
      if (ff == nullptr) { return; }            // no ANE on this box
      if (m != kF16 && (!ff->int8() || ff->int8_plain() != (m == kPlain) ||
                        ff->int8_group() != o.int8_group)) {
        std::printf("[ane_i8_vs_gpu] %s did not build here; skipped\n",
                    kName[m]);
        continue;
      }
      AneFfnSource sg, su, sd;
      sg.w = &wg;
      su.w = &wu;
      sd.w = &wd;
      SharedBuffer y = mc->make_shared_buffer((std::size_t)seq * H * 2);
      int an = 0;
      for (int rep = 0; rep < 2; ++rep) {      // the second is timed
        ff->stage(++layer, sg, su, sd, 0);
        ASSERT_TRUE(ff->join_stage(layer));
        const auto t0 = std::chrono::steady_clock::now();
        an = ff->begin(x, y, seq);
        const bool ok = ff->finish(layer, seq, 0.0, 0.0);
        ms[m] = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
        ASSERT_TRUE(ok && an > 0);
        if (!ok || an <= 0) { return; }
      }
      if (sample.empty()) {
        // Distinct captured rows inside the ANE's span.
        const int span = std::min(R, an), S = std::min(want_rows, span);
        for (int i = 0; i < S; ++i) {
          sample.push_back(seq - an + i * (span / S));
        }
      }
      const auto* yp = static_cast<const std::uint16_t*>(y.contents());
      for (const int r : sample) {
        std::vector<double> v((std::size_t)H);
        for (int c = 0; c < H; ++c) {
          v[(std::size_t)c] = bf(yp[(std::size_t)r * H + c]);
        }
        got[m].push_back(std::move(v));
      }
      ran[m] = true;
    }
    ASSERT_TRUE(!sample.empty());
    if (sample.empty()) { return; }

    // The GPU tiers' weights, and the references over the sampled rows.
    const Q gr = qmat(fc1.data(), F, H, true);
    const Q ur = qmat(fc1.data() + FH, F, H, true);
    const Q dr = qmat(fc2.data(), H, F, true);
    const Q gp = qmat(fc1.data(), F, H, false);
    const Q up = qmat(fc1.data() + FH, F, H, false);
    const Q dp = qmat(fc2.data(), H, F, false);
    const int g1 = vpipe::AneGraphSpec::group_for(kTarget, H);
    const int g2 = vpipe::AneGraphSpec::group_for(kTarget, F);
    const Q gf = qmat_g(fc1.data(), F, H, g1);
    const Q uf = qmat_g(fc1.data() + FH, F, H, g1);
    const Q df = qmat_g(fc2.data(), H, F, g2);
    const int S = (int)sample.size();
    // tok / plain: the tiers on a bf16 model, as H3 runs them; tok16 /
    // plain16: on an f16 model -- the same arithmetic rounded where the
    // ANE rounds, which says how much of the distance is that alone.
    std::vector<std::vector<double>> exact((std::size_t)S), tok((std::size_t)S),
        plain((std::size_t)S), tok16((std::size_t)S), plain16((std::size_t)S),
        fast((std::size_t)S), fast16((std::size_t)S);
    par(S, [&](int i) {
      const std::uint16_t* xs = xr.data() + (std::size_t)(sample[i] % R) * H;
      std::vector<float> xv((std::size_t)H);
      for (int c = 0; c < H; ++c) { xv[(std::size_t)c] = bf(xs[c]); }
      // Unquantized, in double.
      std::vector<double> h((std::size_t)F), out((std::size_t)H);
      for (int n = 0; n < F; ++n) {
        double g = 0.0, u = 0.0;
        const std::uint16_t* gw = fc1.data() + (std::size_t)n * H;
        const std::uint16_t* uw = gw + FH;
        for (int c = 0; c < H; ++c) {
          g += (double)bf(gw[c]) * xv[(std::size_t)c];
          u += (double)bf(uw[c]) * xv[(std::size_t)c];
        }
        h[(std::size_t)n] = g / (1.0 + std::exp(-g)) * u;
      }
      for (int k = 0; k < H; ++k) {
        double a = 0.0;
        const std::uint16_t* dw = fc2.data() + (std::size_t)k * F;
        for (int n = 0; n < F; ++n) {
          a += (double)bf(dw[n]) * h[(std::size_t)n];
        }
        out[(std::size_t)k] = a;
      }
      exact[(std::size_t)i] = out;
      // The GPU's token tiers.
      for (int arm = 0; arm < 4; ++arm) {
        const bool rot = arm % 2 == 0, f16 = arm >= 2;
        auto rd = [&](float f) { return f16 ? rh(f) : rbf(f); };
        const Q& G = rot ? gr : gp;
        const Q& U = rot ? ur : up;
        const Q& D = rot ? dr : dp;
        std::vector<std::int8_t> qx((std::size_t)H), qh((std::size_t)F);
        const float sx = qrow(xv, rot, qx.data());
        std::vector<float> hv((std::size_t)F);
        for (int n = 0; n < F; ++n) {
          std::int32_t ag = 0, au = 0;
          const std::int8_t* gc = G.c.data() + (std::size_t)n * H;
          const std::int8_t* uc = U.c.data() + (std::size_t)n * H;
          for (int c = 0; c < H; ++c) {
            ag += (std::int32_t)gc[c] * qx[(std::size_t)c];
            au += (std::int32_t)uc[c] * qx[(std::size_t)c];
          }
          const float g = rd((float)ag * (sx * G.s[(std::size_t)n]));
          const float u = rd((float)au * (sx * U.s[(std::size_t)n]));
          hv[(std::size_t)n] = rd(g / (1.0f + std::exp(-g)) * u);
        }
        const float sh = qrow(hv, rot, qh.data());
        std::vector<double> o2((std::size_t)H);
        for (int k = 0; k < H; ++k) {
          std::int32_t a = 0;
          const std::int8_t* dc = D.c.data() + (std::size_t)k * F;
          for (int n = 0; n < F; ++n) {
            a += (std::int32_t)dc[n] * qh[(std::size_t)n];
          }
          o2[(std::size_t)k] = rd((float)a * (sh * D.s[(std::size_t)k]));
        }
        auto& dst = f16 ? (rot ? tok16 : plain16) : (rot ? tok : plain);
        dst[(std::size_t)i] = std::move(o2);
      }
      // The GPU's fast tier: steps in the element type, the groups folded
      // in f32 as gemm_i8c_impl folds them.
      for (const bool f16 : {false, true}) {
        auto rd = [&](float f) { return f16 ? rh(f) : rbf(f); };
        auto gemm = [&](const std::int8_t* qa, const float* sa, const Q& W,
                        int k, int gc, int n) {
          float acc = 0.0f;
          const std::int8_t* wc = W.c.data() + (std::size_t)n * k;
          const float* ws = W.s.data() + (std::size_t)n * (k / gc);
          for (int g = 0; g < k / gc; ++g) {
            std::int32_t dot = 0;
            for (int c = g * gc; c < (g + 1) * gc; ++c) {
              dot += (std::int32_t)wc[c] * qa[c];
            }
            acc = std::fma((float)dot * rd(ws[g]), rd(sa[g]), acc);
          }
          return rd(acc);
        };
        std::vector<std::int8_t> qx((std::size_t)H), qh((std::size_t)F);
        std::vector<float> sx((std::size_t)(H / g1)), sh((std::size_t)(F / g2));
        qrow_g(xv, g1, qx.data(), sx.data());
        std::vector<float> hv((std::size_t)F);
        for (int n = 0; n < F; ++n) {
          const float g = gemm(qx.data(), sx.data(), gf, H, g1, n);
          const float u = gemm(qx.data(), sx.data(), uf, H, g1, n);
          hv[(std::size_t)n] = rd(g / (1.0f + std::exp(-g)) * u);
        }
        qrow_g(hv, g2, qh.data(), sh.data());
        std::vector<double> o2((std::size_t)H);
        for (int k = 0; k < H; ++k) {
          o2[(std::size_t)k] = gemm(qh.data(), sh.data(), df, F, g2, k);
        }
        (f16 ? fast16 : fast)[(std::size_t)i] = std::move(o2);
      }
    });
    const double e_tok = rel(tok, exact), e_plain = rel(plain, exact);
    const double e_fast = rel(fast, exact);
    std::printf("[ane_i8_vs_gpu] block %d (%d x %d, %d rows of %d): from "
                "exact -- GPU token %.3e, GPU token_plain %.3e; the tiers "
                "on bf16 vs on f16: token %.3e, token_plain %.3e\n", blk, H,
                F, S, R, e_tok, e_plain, rel(tok, tok16),
                rel(plain, plain16));
    std::printf("[ane_i8_vs_gpu]   GPU fast (groups %d / %d) from exact "
                "%.3e; on bf16 vs on f16 %.3e\n", g1, g2, e_fast,
                rel(fast, fast16));
    for (int m = 0; m < kMods; ++m) {
      if (!ran[m]) { continue; }
      const double ex = rel(got[m], exact);
      const double vt = rel(got[m], tok), vp = rel(got[m], plain);
      const double vf = rel(got[m], fast);
      std::printf("[ane_i8_vs_gpu]   ANE %-15s %6.1f ms a chunk, from exact "
                  "%.3e | from GPU token %.3e (%.2fx its error; on f16 "
                  "%.3e), token_plain %.3e (%.2fx; on f16 %.3e), fast %.3e "
                  "(%.2fx; on f16 %.3e)\n", kName[m], ms[m], ex, vt,
                  vt / e_tok, rel(got[m], tok16), vp, vp / e_plain,
                  rel(got[m], plain16), vf, vf / e_fast,
                  rel(got[m], fast16));
    }
    // THE SAME ERROR, ROUNDED ELSEWHERE. Each per-token module lands as far
    // from exact as its GPU tier (within 1% on H3's blocks 0/12/24/36/49),
    // and nearer that tier than the other one -- the rotation is what
    // separates them -- but not ON it: 0.57-0.73x the token tier's error
    // from it (unrotated 0.32-0.47x from token_plain), where the GPU tier
    // itself moves 0.4-0.5x between a bf16 model and an f16 one. The ANE
    // rotates, scales and sums its tiles in fp16. MEASURED, M4 Pro.
    if (ran[kRot]) {
      const double d = rel(got[kRot], tok);
      EXPECT_TRUE(rel(got[kRot], exact) < 1.05 * e_tok);
      EXPECT_TRUE(d < 0.8 * e_tok);
      EXPECT_TRUE(d < 0.3 * rel(got[kRot], plain));
    }
    if (ran[kPlain]) {
      const double d = rel(got[kPlain], plain);
      EXPECT_TRUE(rel(got[kPlain], exact) < 1.05 * e_plain);
      EXPECT_TRUE(d < 0.6 * e_plain);
      EXPECT_TRUE(d < 0.5 * rel(got[kPlain], tok));
    }
    // Grouped, against the fast tier: 3-5% nearer exact than it (fp16
    // steps where the GPU stores bf16), 0.64-0.83x its error from it --
    // about the tier's own spread between a bf16 and an f16 model -- and
    // well nearer it than the token tier. MEASURED, M4 Pro.
    if (ran[kFast]) {
      const double d = rel(got[kFast], fast);
      EXPECT_TRUE(rel(got[kFast], exact) < 1.05 * e_fast);
      EXPECT_TRUE(d < 0.9 * e_fast);
      EXPECT_TRUE(d < 0.6 * rel(got[kFast], tok));
    }
  }
}

// The rotated int8 graph written out for inspection (VPIPE_ANE_EMIT_DIR):
// a SwiGLU at VPIPE_ANE_EMIT_HID x _FFN (1024 x 2048), _ROWS (512) rows,
// 1024 tiles, as AneFeedForward emits it.
TEST(ane_module, emit_int8_graph_for_inspection)
{
  const char* dir = env_or_null_("VPIPE_ANE_EMIT_DIR");
  if (dir == nullptr) { return; }
  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  vpipe::AneGraphSpec spec;
  spec.kind = vpipe::AneGraphSpec::Kind::SwiGluFfn;
  spec.M  = envi("VPIPE_ANE_EMIT_ROWS", 512);
  spec.K  = envi("VPIPE_ANE_EMIT_HID", 1024);
  spec.N  = envi("VPIPE_ANE_EMIT_FFN", 2048);
  spec.bk = 1024;
  spec.bn = 1024;
  spec.runtime_weights = true;
  spec.weights_out_in  = true;
  spec.int8 = true;
  std::string err;
  const bool ok = vpipe::AneEmitter::emit(spec, {}, std::string(dir) +
                                          "/int8-ffn.mlmodelc", &err);
  if (!ok) { std::printf("[ane_module] emit: %s\n", err.c_str()); }
  ASSERT_TRUE(ok);
}

// STAGING HAS ITS OWN QUEUE. A module stages while another module's
// predict runs on the ANE worker, and a block staged ahead is the block
// the split reads: stage() returns without waiting for the other
// module's predict, queued() names the block, and a block staged
// before a predict of its own module is waited for, not raced.
TEST(ane_tier, staging_runs_beside_the_other_module)
{
  using vpipe::genai::AneFeedForward;
  using vpipe::genai::AneFfnSource;
  using vpipe::metal_compute::SharedBuffer;
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  constexpr int K = 512, N = 1536, seq = 2048, chunk = 512;
  std::mt19937 rng(13);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto bf16 = [&](std::size_t n, float sc) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const float f = nd(rng) * sc;
      std::uint32_t bits = 0;
      std::memcpy(&bits, &f, sizeof(bits));
      p[i] = (std::uint16_t)(bits >> 16);
    }
    return b;
  };
  const SharedBuffer x = bf16((std::size_t)seq * K, 0.5f);
  const SharedBuffer w1 = bf16((std::size_t)N * K, 0.05f);
  const SharedBuffer w2 = bf16((std::size_t)N * K, 0.05f);
  AneFeedForward::Options o;
  o.session = &sess;
  o.mc      = mc;
  o.hidden  = K;
  o.ffn     = N;
  o.seq     = seq;
  o.rows    = 0.75f;
  o.chunk   = chunk;
  o.matmul  = true;
  o.tag     = "stage-a";
  auto a = AneFeedForward::create(o);
  o.tag     = "stage-b";
  auto b = AneFeedForward::create(o);
  if (a == nullptr || b == nullptr) { return; }
  auto out = [&]() {
    SharedBuffer y = mc->make_shared_buffer((std::size_t)seq * N * 2);
    std::memset(y.contents(), 0, y.byte_size());
    return y;
  };
  AneFfnSource s1, s2;
  s1.w = &w1;
  s2.w = &w2;
  // Reference: each module alone, staged and run in turn.
  a->stage(1, s1, 0);
  ASSERT_TRUE(a->join_stage(1));
  SharedBuffer ya = out();
  const int ra = a->begin(x, ya, seq);
  ASSERT_TRUE(a->finish(1, seq, 0.0, 0.0) && ra > 0);
  b->stage(1, s2, 0);
  ASSERT_TRUE(b->join_stage(1));
  SharedBuffer yb = out();
  const int rb = b->begin(x, yb, seq);
  ASSERT_TRUE(b->finish(1, seq, 0.0, 0.0) && rb > 0);
  // Overlapped: b's next block staged while a's predict runs, then a
  // staged AHEAD of its turn while b predicts, each joined at its split.
  a->stage(2, s1, 0);
  ASSERT_TRUE(a->join_stage(2));
  SharedBuffer ya2 = out(), yb2 = out();
  ASSERT_TRUE(a->begin(x, ya2, seq) == ra);
  b->stage(2, s2, 0);                    // beside a's predict
  EXPECT_TRUE(b->queued(2) && !b->queued(1));
  ASSERT_TRUE(a->finish(2, seq, 0.0, 0.0));
  ASSERT_TRUE(b->join_stage(2));
  ASSERT_TRUE(b->begin(x, yb2, seq) == rb);
  a->stage(3, s2, 0);                    // ahead, beside b's predict
  ASSERT_TRUE(b->finish(2, seq, 0.0, 0.0));
  // a's slots now hold w2: its next split computes b's projection.
  ASSERT_TRUE(a->join_stage(3) && a->queued(3));
  SharedBuffer ya3 = out();
  ASSERT_TRUE(a->begin(x, ya3, seq) == ra);
  ASSERT_TRUE(a->finish(3, seq, 0.0, 0.0));
  const std::size_t bytes_n = (std::size_t)seq * N * 2;
  EXPECT_TRUE(std::memcmp(ya.contents(), ya2.contents(), bytes_n) == 0);
  EXPECT_TRUE(std::memcmp(yb.contents(), yb2.contents(), bytes_n) == 0);
  EXPECT_TRUE(std::memcmp(yb.contents(), ya3.contents(), bytes_n) == 0);
  // Staged into a module whose predict is in flight: waited for, so the
  // predict reads the weights it began with (w2), not the new ones.
  SharedBuffer ya4 = out();
  ASSERT_TRUE(a->begin(x, ya4, seq) == ra);
  a->stage(4, s1, 0);
  ASSERT_TRUE(a->finish(3, seq, 0.0, 0.0));
  EXPECT_TRUE(std::memcmp(yb.contents(), ya4.contents(), bytes_n) == 0);
  ASSERT_TRUE(a->join_stage(4));
  a->forget_staged();
  EXPECT_FALSE(a->queued(4));
}

// THE GPU FALLBACK'S CONTRACT, through the described surface a plugin
// uses. A split whose second chunk is lost (VPIPE_ANE_FAIL, read when the
// tier is built) must:
//   * fail finish(), and name its first lost row in info()["lost_row0"];
//   * have written the chunk BEFORE it, exactly as a clean run does;
//   * leave every row from lost_row0 on as it was -- run IN PLACE here,
//     which is what every in-place family's fallback leans on: those rows
//     are still the input, and the GPU recomputes them from it;
//   * answer kGpu to every later plan_block(), with no barrier to take.
// And a tier that lost nothing says so: lost_row0 -1, gpu_only false.
TEST(ane_tier, a_lost_chunk_is_named_and_the_tier_stays_off)
{
  namespace ane = vpipe::genai::ane;
  namespace acc = vpipe::genai::accel;
  vpipe::Session sess;
  vpipe::metal_compute::MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  // Two 256-row chunks: a pinned share takes rows [256, 768).
  constexpr int H = 256, F = 1024, seq = 768, chunk = 256;
  std::mt19937 rng(5);
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
  const auto x0 = bf16((std::size_t)seq * H, 0.5f);
  const std::size_t bytes = (std::size_t)seq * H * 2;
  auto copy_of = [&](const vpipe::metal_compute::SharedBuffer& src) {
    auto b = mc->make_shared_buffer(bytes);
    std::memcpy(b.contents(), src.contents(), bytes);
    return b;
  };
  auto spec = tier_spec_("ffn", "gelu_tanh", H, F, seq, chunk);
  acc::set_real(&spec, ane::kRows, 1.0);
  const std::string nuw = ane::key(ane::kProjUp, "w");
  const std::string ndw = ane::key(ane::kProjDown, "w");
  const ane::Binding wb[] = {{nuw, &wu}, {ndw, &wd}};
  vpipe::FlexData args = vpipe::FlexData::make_object();
  acc::set_integer(&args, ane::kSeq, seq);
  // One split, in place over `plane`: true when finish() succeeded.
  auto split = [&](ane::Tier& t, vpipe::metal_compute::SharedBuffer& plane,
                   int* rows) {
    const vpipe::FlexData params = vpipe::FlexData::make_object();
    if (!t.stage(0, params, wb) || !t.join_stage(0)) { return false; }
    const ane::Binding io[] = {{ane::kIoIn, &plane}, {ane::kIoOut, &plane}};
    *rows = t.begin(args, io);
    return t.finish(0, vpipe::FlexData::make_object());
  };

  std::string why;
  auto clean = ane::create(&sess, mc, spec, &why);
  if (clean == nullptr) { return; }              // no ANE on this box
  auto good = copy_of(x0);
  int a_good = 0;
  ASSERT_TRUE(split(*clean, good, &a_good));
  ASSERT_TRUE(a_good == 2 * chunk);
  {
    const vpipe::FlexData inf = clean->info();
    EXPECT_TRUE(acc::integer(&inf, ane::kInfoLostRow0, 0) == -1);
    EXPECT_FALSE(acc::flag(&inf, ane::kInfoGpuOnly));
  }

  setenv("VPIPE_ANE_FAIL", "0:1", 1);
  auto lossy = ane::create(&sess, mc, spec, &why);
  unsetenv("VPIPE_ANE_FAIL");
  ASSERT_TRUE(lossy != nullptr);
  if (lossy == nullptr) { return; }
  auto plane = copy_of(x0);
  int a_lost = 0;
  EXPECT_FALSE(split(*lossy, plane, &a_lost));
  EXPECT_TRUE(a_lost == a_good);
  const vpipe::FlexData inf = lossy->info();
  const int lost = (int)acc::integer(&inf, ane::kInfoLostRow0, -1);
  const int row0 = seq - a_lost;
  EXPECT_TRUE(lost == row0 + chunk);             // the second chunk
  EXPECT_TRUE(acc::flag(&inf, ane::kInfoGpuOnly));
  if (lost != row0 + chunk) { return; }
  const auto* got = static_cast<const char*>(plane.contents());
  const std::size_t rb = (std::size_t)H * 2;
  // The first chunk: written, and exactly as the clean run wrote it.
  EXPECT_TRUE(std::memcmp(got + (std::size_t)row0 * rb,
                          static_cast<const char*>(good.contents()) +
                              (std::size_t)row0 * rb,
                          (std::size_t)chunk * rb) == 0);
  // The lost rows, and the GPU's head: untouched, i.e. still the input.
  EXPECT_TRUE(std::memcmp(got + (std::size_t)lost * rb,
                          static_cast<const char*>(x0.contents()) +
                              (std::size_t)lost * rb,
                          (std::size_t)(seq - lost) * rb) == 0);
  EXPECT_TRUE(std::memcmp(got, x0.contents(), (std::size_t)row0 * rb) == 0);
  // And it stays off.
  for (int b = 0; b < 3; ++b) {
    EXPECT_TRUE(lossy->plan_block() == ane::Tier::Plan::kGpu);
    EXPECT_FALSE(lossy->needs_barrier());
  }
  EXPECT_TRUE(clean->plan_block() != ane::Tier::Plan::kGpu);
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
  // _INT8: the module in rotated int8 (ane::kInt8; a swiglu or a matmul,
  // so set _GELU=0 for a feed-forward).
  const bool int8 = std::getenv("VPIPE_ANE_MEM_PROBE_INT8") != nullptr;
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
  acc::set_flag(&spec, ane::kInt8, int8);
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
  std::printf("[ane_tier] mem probe %s%s %dx%d, seq %d, chunk %d, %d ANE "
              "rows; tier estimate %lld MB (host rows %lld MB)\n",
              matmul ? "matmul" : gelu ? "gelu" : "swiglu",
              acc::flag(&inf, ane::kInt8) ? " int8" : "", H, F, seq, chunk,
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
  // MOSS-Image's SITU, each half alone: the gate's factor with u pinned
  // by its bias (Wu = 0, bu = 10), and the hidden one's with g pinned.
  auto situ_a = [](double g) {
    return 4.0 * std::tanh(g / 4.0) / (1.0 + std::exp(-g));
  };
  auto situ_h = [](double h) { return 25.0 * std::tanh(h / 25.0); };
  const Fill T{false, 10.0f};
  probe("situ_g", K::SituFfn, {I, Z, I, Z, T, Z},
        [&](double v) { return situ_a(v) * situ_h(10.0); });
  probe("situ_u", K::SituFfn, {Z, I, I, T, Z, Z},
        [&](double v) { return situ_a(10.0) * situ_h(v); });
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
