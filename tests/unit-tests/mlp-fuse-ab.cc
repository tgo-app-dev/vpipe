// mlp-fuse-ab.cc -- gate|up GEMM + SiLU + multiply measured as ONE unit.
//
// Two ways to spend a SwiGLU MLP's first half on a matrix-core Mac:
//
//   FUSED    affine_qmm_swiglu_*: one steel GEMM over the interleaved
//            gate|up weight whose epilogue is register-local (column 2g is
//            the gate, 2g+1 the up, and the frag holds both), so it writes
//            [M, ffn] directly -- no [M, 2*ffn] intermediate and no second
//            pass. Costs the matrix units: the epilogue is simdgroup_matrix.
//   UNFUSED  dequant the weight once, run the dense matmul2d over it into
//            [M, 2*ffn], then fold with swiglu_interleaved. Three
//            dispatches and 3x the activation traffic, on the matrix cores.
//
// Every LM and DiT here picks one or the other, and they do not agree:
// llama-3 fuses; qwen3 fuses below _mma_min_m (64 rows) and takes the
// unfused matmul2d above it; FLUX.2 and Krea-2 build the fused path and
// then default it OFF under _use_mma2; MiniMax-H3 has no fused arm at all.
// This measures the trade at the shapes those models actually run, so the
// choice is a number rather than a inherited assumption.
//
// The arms are INTERLEAVED round by round and reduced with a median: this
// box has a 4-5% power-budget spread between back-to-back runs, which is
// wider than some of the gaps below.
//
// By default only the arms-agree cross-check runs (a couple of seconds) --
// that is the part with a pass/fail, and it catches a layout regression in
// either the fused epilogue or the interleaved fold. VPIPE_MLP_FUSE_AB (any
// value) adds the timing ladder: 64..9382 rows over three shapes, ~3 min.

#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/mma-splitk.h"
#include "stages/gpu-telemetry.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

bool full_()
{
  return std::getenv("VPIPE_MLP_FUSE_AB") != nullptr;
}

double secs_(std::chrono::steady_clock::time_point a,
             std::chrono::steady_clock::time_point b)
{
  return std::chrono::duration<double>(b - a).count();
}

double median_(std::vector<double> v)
{
  if (v.empty()) { return 0.0; }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// A packed affine gate|up weight (group 64, 4-bit) + scales/biases, filled
// with deterministic garbage. Both arms read the same bytes and do the same
// work whatever the values are; the cross-check below compares arms against
// each other, not against an oracle.
struct QW {
  SharedBuffer w, s, b;
};

QW make_qw_(MetalCompute* mc, int N, int K, std::uint32_t seed)
{
  const std::size_t codes = (std::size_t)N * K * 4 / 8;
  const std::size_t ng = (std::size_t)N * (K / 64);
  QW q{mc->make_shared_buffer(codes), mc->make_shared_buffer(ng * 2),
       mc->make_shared_buffer(ng * 2)};
  auto* wp = static_cast<std::uint8_t*>(q.w.contents());
  std::uint32_t r = seed | 1u;
  for (std::size_t i = 0; i < codes; ++i) {
    r = r * 1664525u + 1013904223u;
    wp[i] = (std::uint8_t)(r >> 24);
  }
  auto* sp = static_cast<_Float16*>(q.s.contents());
  auto* bp = static_cast<_Float16*>(q.b.contents());
  for (std::size_t i = 0; i < ng; ++i) {
    sp[i] = (_Float16)0.01f;
    bp[i] = (_Float16)(-0.08f);
  }
  return q;
}

SharedBuffer make_act_(MetalCompute* mc, std::size_t n, std::uint32_t seed)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  auto* p = static_cast<_Float16*>(b.contents());
  std::uint32_t r = seed | 1u;
  for (std::size_t i = 0; i < n; ++i) {
    r = r * 1664525u + 1013904223u;
    p[i] = (_Float16)(((float)(r >> 16) / 32768.0f - 1.0f) * 0.1f);
  }
  return b;
}

enum class Arm {
  Fused32,     // affine_qmm_swiglu_w4g64
  Fused64,     // ... _bm64
  Fused128,    // ... _bm128
  Steel32Act,  // affine_qmm_steel_w4g64 + swiglu_interleaved
  Dq128Act,    // dequant + dense_gemm_mma_t_n128 + swiglu_interleaved
  Dq256Act,    // dequant + dense_gemm_mma_t_n128x256 + swiglu_interleaved
  MmaAct,      // affine_qmm_mma_w4g64 + swiglu_interleaved
};

const char* arm_tag_(Arm a)
{
  switch (a) {
    case Arm::Fused32:    return "fused32";
    case Arm::Fused64:    return "fused64";
    case Arm::Fused128:   return "fused128";
    case Arm::Steel32Act: return "steel32+act";
    case Arm::Dq128Act:   return "dq+n128+act";
    case Arm::Dq256Act:   return "dq+n256+act";
    case Arm::MmaAct:     return "qmm_mma+act";
  }
  return "?";
}

bool arm_is_fused_(Arm a)
{
  return a == Arm::Fused32 || a == Arm::Fused64 || a == Arm::Fused128;
}

bool arm_is_mma_(Arm a)
{
  return a == Arm::Dq128Act || a == Arm::Dq256Act || a == Arm::MmaAct;
}

struct Kernels {
  ComputeLibrary lib_steel, lib_dq, lib_dense, lib_mma, lib_elt;
  ComputeFunction fused32, fused64, fused128;
  ComputeFunction steel32, dq, dense128, dense256, qmm_mma, swiglu_inter;
  bool have_mma = false;

  void load(MetalCompute* mc)
  {
    lib_steel = mc->load_library("affine_qmm_steel");
    lib_dq    = mc->load_library("affine_dequant");
    lib_dense = mc->load_library("dense_gemm_mma");
    lib_mma   = mc->load_library("affine_qmm_mma");
    lib_elt   = mc->load_library("llm_elementwise");
    fused32   = lib_steel.function("affine_qmm_swiglu_w4g64");
    fused64   = lib_steel.function("affine_qmm_swiglu_w4g64_bm64");
    fused128  = lib_steel.function("affine_qmm_swiglu_w4g64_bm128");
    steel32   = lib_steel.function("affine_qmm_steel_w4g64");
    dq        = lib_dq.function("affine_dequant_w4g64");
    dense128  = lib_dense.function("dense_gemm_mma_t_n128_f16");
    dense256  = lib_dense.function("dense_gemm_mma_t_n128x256_f16");
    qmm_mma   = lib_mma.function("affine_qmm_mma_w4g64");
    swiglu_inter = lib_elt.function("swiglu_interleaved_f16");
    have_mma = mc->supports_matrix_cores() && dense128.valid()
               && swiglu_inter.valid();
  }

  bool available(Arm a) const
  {
    switch (a) {
      case Arm::Fused32:    return fused32.valid();
      case Arm::Fused64:    return fused64.valid();
      case Arm::Fused128:   return fused128.valid();
      case Arm::Steel32Act: return steel32.valid() && swiglu_inter.valid();
      case Arm::Dq128Act:   return have_mma && dq.valid();
      case Arm::Dq256Act:   return have_mma && dq.valid()
                                   && dense256.valid();
      case Arm::MmaAct:     return have_mma && qmm_mma.valid();
    }
    return false;
  }
};

// Scratch shared by every arm, sized once for the widest shape / tallest M.
struct Bufs {
  SharedBuffer x;     // [M, K]
  SharedBuffer y;     // [M, ffn]   the unit's output
  SharedBuffer gu;    // [M, 2*ffn] the unfused intermediate
  SharedBuffer wdq;   // [2*ffn, K] dequantized weight
};

// One whole gate|up + act + mul through one arm. N is the FUSED width
// (2*ffn); the unit writes [M, ffn] into b.y whichever way it gets there.
void encode_(ComputeEncoder& enc, const Kernels& kn, Arm a, const QW& q,
             const Bufs& b, int M, int N, int K)
{
  const int ffn = N / 2;
  auto fold = [&]() {
    enc.set_function(kn.swiglu_inter);
    enc.set_buffer(0, b.gu);
    enc.set_buffer(1, b.y);
    enc.set_constant(2, M);
    enc.set_constant(3, ffn);
    enc.dispatch({(unsigned)((std::size_t)M * ffn), 1, 1}, {256, 1, 1});
  };
  if (arm_is_fused_(a)) {
    const int bm = (a == Arm::Fused128) ? 128 : (a == Arm::Fused64) ? 64 : 32;
    const unsigned tgz = (a == Arm::Fused128) ? 4u : 2u;
    enc.set_function(a == Arm::Fused128 ? kn.fused128
                     : a == Arm::Fused64 ? kn.fused64 : kn.fused32);
    enc.set_buffer(0, q.w);
    enc.set_buffer(1, q.s);
    enc.set_buffer(2, q.b);
    enc.set_buffer(3, b.x);
    enc.set_buffer(4, b.y);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.set_constant(7, M);
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                  (unsigned)(((M + bm - 1) / bm) * 2), tgz}, {32, 2, tgz});
    return;
  }
  if (a == Arm::Steel32Act || a == Arm::MmaAct) {
    enc.set_function(a == Arm::MmaAct ? kn.qmm_mma : kn.steel32);
    enc.set_buffer(0, q.w);
    enc.set_buffer(1, q.s);
    enc.set_buffer(2, q.b);
    enc.set_buffer(3, b.x);
    enc.set_buffer(4, b.gu);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.set_constant(7, M);
    if (a == Arm::MmaAct) {
      enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                    (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
    } else {
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
    }
    fold();
    return;
  }
  // Dequant once, then the dense matmul2d, then the fold. The dequant is
  // timed WITH the GEMM because that is what a forward pays: every block
  // carries its own weights, so it is per GEMM and not once per model.
  enc.set_function(kn.dq);
  enc.set_buffer(0, q.w);
  enc.set_buffer(1, q.s);
  enc.set_buffer(2, q.b);
  enc.set_buffer(3, b.wdq);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.dispatch({(unsigned)(K / 8), (unsigned)N, 1}, {64, 1, 1});
  const int BN = (a == Arm::Dq256Act) ? 256 : 128;
  enc.set_function(a == Arm::Dq256Act ? kn.dense256 : kn.dense128);
  enc.set_buffer(0, b.x);
  enc.set_buffer(1, b.wdq);
  enc.set_buffer(2, b.wdq);          // bias slot unused (has_bias = 0)
  enc.set_buffer(3, b.gu);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  fold();
}

// Milliseconds for one call of the whole unit, from a timed run of `iters`
// back-to-back calls in one command buffer.
double time_ms_(MetalCompute* mc, const Kernels& kn, Arm a, const QW& q,
                const Bufs& b, int M, int N, int K, int iters)
{
  const auto t0 = std::chrono::steady_clock::now();
  CommandStream st = mc->make_command_stream();
  {
    ComputeEncoder enc = st.begin_compute();
    for (int i = 0; i < iters; ++i) {
      encode_(enc, kn, a, q, b, M, N, K);
    }
  }
  st.commit().wait();
  const auto t1 = std::chrono::steady_clock::now();
  return secs_(t0, t1) * 1e3 / iters;
}

double rel_l2_(const SharedBuffer& a, const SharedBuffer& b, std::size_t n)
{
  const auto* pa = static_cast<const _Float16*>(a.contents());
  const auto* pb = static_cast<const _Float16*>(b.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double x = (double)(float)pa[i], y = (double)(float)pb[i];
    num += (x - y) * (x - y);
    den += x * x;
  }
  return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

struct Shape {
  const char* tag;
  int K, ffn;
};

// ---- down_proj -------------------------------------------------------
//
// The other half of the MLP, and the half where matmul2d does NOT get its
// rate for free: down contracts over ffn, so K is 2-4x deeper than gate|up's
// and N is the narrow hidden. Only N/BN threadgroups exist, each walking one
// long serial reduction, and the matrix units stall with nothing else in
// flight. Split-K gives each K chunk its own plane (grid.z) and folds them.
//
// `kc2` / `kc4` are the 2-way and 4-way chunk widths; a kernel exists per
// chunk width, so a depth no KC divides has no split arm at all.
struct Down {
  const char* tag;
  int N, K, kc2, kc4, kc8;
  bool shipped;   // does a model dispatch a split at this depth today?
};

// dequant + one dense matmul2d, no epilogue -- the shape both directions are
// compared in, so the parity number is not carrying a fold on one side.
void encode_gemm_(ComputeEncoder& enc, const Kernels& kn, const QW& q,
                  const SharedBuffer& x, const SharedBuffer& wdq,
                  const SharedBuffer& y, int M, int N, int K, int BN)
{
  enc.set_function(kn.dq);
  enc.set_buffer(0, q.w);
  enc.set_buffer(1, q.s);
  enc.set_buffer(2, q.b);
  enc.set_buffer(3, wdq);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.dispatch({(unsigned)(K / 8), (unsigned)N, 1}, {64, 1, 1});
  enc.set_function(BN == 256 ? kn.dense256 : kn.dense128);
  enc.set_buffer(0, x);
  enc.set_buffer(1, wdq);
  enc.set_buffer(2, wdq);            // bias slot unused
  enc.set_buffer(3, y);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
}

// Median GFLOP/s of `arm` over 3 rounds, with the GPU clock sampled across
// the timed region. Callers size `iters` for ~0.3 s of GPU work per round:
// the clock sampler polls every 85 ms, so a 30 ms region reports whatever it
// happened to catch -- which is the idle clock (~340 MHz), not the running
// one. `mhz` (optional) receives the average clock -- on a
// fanless box a slow arm and a throttled arm look identical without it.
double rate_(MetalCompute* mc, double f1, int iters, double* mhz,
             double* mhz_max,
             const std::function<void(ComputeEncoder&)>& arm)
{
  auto once = [&](int n) {
    const auto t0 = std::chrono::steady_clock::now();
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      for (int i = 0; i < n; ++i) { arm(e); }
    }
    st.commit().wait();
    const auto t1 = std::chrono::steady_clock::now();
    return f1 * n / 1e9 / secs_(t0, t1);
  };
  once(1);
  GpuTelemetrySampler tel;
  tel.start();
  std::vector<double> v;
  for (int r = 0; r < 3; ++r) { v.push_back(once(iters)); }
  const GpuTelemetry t = tel.stop();
  if (mhz != nullptr) { *mhz = t.freq_mhz.ok ? t.freq_mhz.avg : 0.0; }
  if (mhz_max != nullptr) {
    *mhz_max = t.freq_mhz.ok ? t.freq_mhz.max : 0.0;
  }
  return median_(v);
}

// The parity reference: the SAME shape's gate|up GEMM at the same M, best of
// the two dense tiles. Measured here rather than quoted from the sweep above
// so the ratio is within one process and one power budget.
double ref_rate_(MetalCompute* mc, const Kernels& kn, int M, int K, int ffn)
{
  const int N = 2 * ffn;
  QW q = make_qw_(mc, N, K, 3u + (unsigned)N);
  SharedBuffer x = make_act_(mc, (std::size_t)M * K, 9u);
  SharedBuffer y = mc->make_shared_buffer((std::size_t)M * N * 2);
  SharedBuffer wdq = mc->make_shared_buffer((std::size_t)N * K * 2);
  const double f1 = 2.0 * M * N * K;
  const int iters = std::max(2, std::min(200, (int)(3.0e12 / f1)));
  double best = 0.0;
  for (int BN : {128, 256}) {
    double mhz = 0.0;
    best = std::max(best, rate_(mc, f1, iters, &mhz, nullptr,
                                [&](ComputeEncoder& e) {
      encode_gemm_(e, kn, q, x, wdq, y, M, N, K, BN);
    }));
  }
  return best;
}

}  // namespace

// The unit is gate|up + SiLU + multiply. down_proj is common to both shapes
// and deliberately outside: it is the same GEMM either way.
TEST(mlp_fuse, gate_up_act)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.available(Arm::Fused32)) {
    std::printf("[mlp_fuse] no fused SwiGLU GEMM -- skip\n");
    return;
  }
  const bool full = full_();
  std::vector<Shape> shapes = {
      {"h3-dit fc1", 5376, 14336},     // MiniMax-H3 DiT, hidden 5376
      {"qwen3.5-9B mlp", 4096, 12288},
  };
  if (full) { shapes.push_back({"gemma-31B mlp", 5376, 21504}); }
  const std::vector<int> ms =
      full ? std::vector<int>{64, 256, 1024, 4096, 9382} : std::vector<int>{};
  if (!full) {
    std::printf("[mlp_fuse] cross-check only; "
                "VPIPE_MLP_FUSE_AB=1 for the timing ladder\n");
  }
  const Arm arms[] = {Arm::Fused32,  Arm::Fused64,  Arm::Fused128,
                      Arm::Steel32Act, Arm::Dq128Act, Arm::Dq256Act,
                      Arm::MmaAct};
  if (!kn.have_mma) {
    std::printf("[mlp_fuse] no matrix cores -- steel arms only\n");
  }

  for (const Shape& sh : shapes) {
    const int N = 2 * sh.ffn, K = sh.K;
    const int max_m = ms.empty() ? 256 : ms.back();   // 256 = cross-check M
    QW q = make_qw_(mc, N, K, 11u + (unsigned)N);
    Bufs b;
    b.x = make_act_(mc, (std::size_t)max_m * K, 5u + (unsigned)K);
    b.y = mc->make_shared_buffer((std::size_t)max_m * sh.ffn * 2);
    b.gu = mc->make_shared_buffer((std::size_t)max_m * N * 2);
    if (kn.have_mma) {
      b.wdq = mc->make_shared_buffer((std::size_t)N * K * 2);
    }

    // Do the arms agree? A fused epilogue and a matmul2d + fold reduce K in
    // different orders, so this is a numerics cross-check, not a bit-exact
    // one -- but a wrong LAYOUT (gate/up swapped, halves vs interleave)
    // shows up here as a large number rather than as a plausible model.
    {
      const int m = 256;
      SharedBuffer ref = mc->make_shared_buffer((std::size_t)m * sh.ffn * 2);
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        encode_(enc, kn, Arm::Fused32, q, b, m, N, K);
      }
      st.commit().wait();
      std::memcpy(ref.contents(), b.y.contents(),
                  (std::size_t)m * sh.ffn * 2);
      for (Arm a : arms) {
        if (a == Arm::Fused32 || !kn.available(a)) { continue; }
        CommandStream s2 = mc->make_command_stream();
        {
          ComputeEncoder enc = s2.begin_compute();
          encode_(enc, kn, a, q, b, m, N, K);
        }
        s2.commit().wait();
        const double r = rel_l2_(ref, b.y, (std::size_t)m * sh.ffn);
        std::printf("[mlp_fuse] %-15s %-12s vs fused32 rel-L2 %.3e\n",
                    sh.tag, arm_tag_(a), r);
        EXPECT_TRUE(r < 5e-2);
      }
    }

    for (int M : ms) {
      const double f1 = 2.0 * M * N * K;
      const int iters = std::max(2, std::min(30, (int)(4.0e11 / f1)));
      // Warm every arm once, then interleave the timed rounds -- an arm
      // measured entirely before another is measured on a different power
      // budget, and the spread between runs is wider than some gaps here.
      for (Arm a : arms) {
        if (kn.available(a)) { time_ms_(mc, kn, a, q, b, M, N, K, 1); }
      }
      std::vector<std::vector<double>> t(std::size(arms));
      for (int r = 0; r < 3; ++r) {
        for (std::size_t i = 0; i < std::size(arms); ++i) {
          if (!kn.available(arms[i])) { continue; }
          t[i].push_back(
              time_ms_(mc, kn, arms[i], q, b, M, N, K, iters));
        }
      }
      double best_fused = 1e30, best_mma = 1e30;
      const char* best_fused_tag = "-";
      const char* best_mma_tag = "-";
      std::printf("[mlp_fuse] %-15s M=%4d N=%5d K=%5d |", sh.tag, M, N, K);
      for (std::size_t i = 0; i < std::size(arms); ++i) {
        if (t[i].empty()) { continue; }
        const double ms_ = median_(t[i]);
        std::printf(" %s %6.2f ms (%5.0f GF/s) |", arm_tag_(arms[i]), ms_,
                    f1 / 1e9 / (ms_ * 1e-3));
        if (arm_is_fused_(arms[i]) && ms_ < best_fused) {
          best_fused = ms_;
          best_fused_tag = arm_tag_(arms[i]);
        }
        if (arm_is_mma_(arms[i]) && ms_ < best_mma) {
          best_mma = ms_;
          best_mma_tag = arm_tag_(arms[i]);
        }
      }
      std::printf("\n");
      if (best_fused < 1e29 && best_mma < 1e29) {
        std::printf("[mlp_fuse]   -> best fused %s %.2f ms vs best mma %s "
                    "%.2f ms = %.2fx %s\n",
                    best_fused_tag, best_fused, best_mma_tag, best_mma,
                    best_fused / best_mma,
                    best_mma < best_fused ? "(unfused matmul2d wins)"
                                          : "(fused steel wins)");
      }
      EXPECT_TRUE(best_fused < 1e29);
    }
  }
}

// ---- down_proj: is it getting the matrix cores' RATE, or only their
// kernel? ----------------------------------------------------------------
//
// Down contracts over ffn, so it is the deep-K direction: only (N/BN)x(M/BM)
// threadgroups exist and each walks one long serial reduction, leaving the
// matrix units with nothing else in flight. Split-K gives each K chunk its
// own plane (grid.z) and folds them, which trades weight re-reads for
// occupancy.
//
// Each shape is measured against ITS OWN gate|up GEMM at the same M -- same
// weights, same kernel, same box, shallow direction -- so parity is a ratio
// and not a comparison against a number remembered from another run.
//
// The split arm is SCANNED, not fixed: every split count whose chunk width
// is exact AND has a compiled kernel is measured. That is the data the
// chunk-width chooser in mma-splitk.h is set from, so it is collected the
// same way the chooser searches.
//
// GPU CLOCK is sampled per arm. This box is fanless and clocks from ~1500
// MHz down to ~700 under sustained load, which is wider than most of the
// gaps here -- a row measured at 800 MHz is not comparable to one measured
// at 1450, and without the number a thermal dip reads as a slow kernel.
// (M5 only. On M4 -- no matrix cores -- the steel path is not sensitive to
// contraction depth and none of this applies.)
TEST(mlp_fuse, down_proj)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.have_mma) {
    std::printf("[mlp_down] no matrix cores -- skip\n");
    return;
  }
  if (!full_()) {
    std::printf("[mlp_down] VPIPE_MLP_FUSE_AB=1 to run\n");
    return;
  }
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise");
  ComputeFunction f_fold = lib_elt.function("splitk_fold_f32_f16");
  struct Down {
    const char* tag;
    int N, K;
    bool shipped;   // does a model dispatch a split at this depth today?
  };
  const Down downs[] = {
      // H3's VAEs, whose GEMMs reach the same regime from the other side:
      // a 3x3x3 conv is an im2col GEMM with K = cin*27, so the encoder's
      // deep levels contract over 13824 and 27648 with N = cout. The ViT
      // decoder's ff-down (8192) and the audio VAE's k=11 conv at 1024
      // channels (11264) sit just BELOW the shipped floor, which is the
      // question those two rows are here to answer.
      {"h3-vvae enc c512",  512, 13824, false},
      {"h3-vvae enc c1024",1024, 27648, false},
      {"h3-vvae vit ffdn", 2048,  8192, false},
      {"h3-avae res k11",  1024, 11264, false},
      {"h3-dit fc2",      5376, 14336, false},
      {"qwen3.5-9B down", 4096, 12288, false},
      {"qwen-27B down",   5120, 17408, true},
      {"gemma-31B down",  5376, 21504, true},
  };
  for (const Down& d : downs) {
    // gate|up for this shape is [2*ffn, hidden]: N and K swapped, ffn side
    // doubled. Same weight bytes per output element either way.
    QW q = make_qw_(mc, d.N, d.K, 41u + (unsigned)d.K);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)d.N * d.K * 2);
    for (int M : {1024, 4096, 9382}) {
      SharedBuffer x = make_act_(mc, (std::size_t)M * d.K, 43u);
      SharedBuffer y = mc->make_shared_buffer((std::size_t)M * d.N * 2);
      const std::size_t plane = (std::size_t)M * d.N;
      const double f1 = 2.0 * M * d.N * d.K;
      const int iters = std::max(2, std::min(200, (int)(3.0e12 / f1)));
      double mhz = 0.0, mhz_max = 0.0;
      const double g_ref = ref_rate_(mc, kn, M, d.N, d.K);
      const double g128 = rate_(mc, f1, iters, &mhz, &mhz_max, [&](ComputeEncoder& e) {
        encode_gemm_(e, kn, q, x, wdq, y, M, d.N, d.K, 128);
      });
      const double mhz128 = mhz, mhz128p = mhz_max;
      const double g256 = rate_(mc, f1, iters, &mhz, &mhz_max, [&](ComputeEncoder& e) {
        encode_gemm_(e, kn, q, x, wdq, y, M, d.N, d.K, 256);
      });
      std::printf("[mlp_down] %-16s M=%4d N=%5d K=%5d%s\n"
                  "[mlp_down]   single n128 %5.0f @%4.0f/%4.0fMHz  n256 "
                  "%5.0f @%4.0f/%4.0fMHz | gate|up ref %5.0f\n",
                  d.tag, M, d.N, d.K, d.shipped ? "  [ships a split]" : "",
                  g128, mhz128, mhz128p, g256, mhz, mhz_max, g_ref);
      // Scan every exact split whose chunk kernel exists.
      double best = std::max(g128, g256);
      int best_s = 1, best_kc = d.K;
      std::printf("[mlp_down]   split:");
      for (int S = 2; S <= 16; ++S) {
        if (d.K % S != 0) { continue; }
        const int kc = d.K / S;
        ComputeFunction f_sk = kn.lib_dense.function(
            std::string("dense_gemm_mma_splitk32_n128x256_k")
            + std::to_string(kc) + "_f16");
        if (!f_sk.valid() || !f_fold.valid()) { continue; }
        SharedBuffer sk =
            mc->make_shared_buffer(plane * (std::size_t)S * 4);
        if (sk.empty()) { continue; }
        const double g = rate_(mc, f1, iters, &mhz, &mhz_max, [&](ComputeEncoder& e) {
          // The dequant is inside every timed arm: a forward pays it per
          // GEMM, because each block carries its own weights.
          e.set_function(kn.dq);
          e.set_buffer(0, q.w); e.set_buffer(1, q.s); e.set_buffer(2, q.b);
          e.set_buffer(3, wdq);
          e.set_constant(4, d.K); e.set_constant(5, d.N);
          e.dispatch({(unsigned)(d.K / 8), (unsigned)d.N, 1}, {64, 1, 1});
          e.set_function(f_sk);
          e.set_buffer(0, x); e.set_buffer(1, wdq); e.set_buffer(2, sk);
          e.set_constant(3, d.K); e.set_constant(4, d.N);
          e.set_constant(5, M);
          e.dispatch({(unsigned)(((d.N + 255) / 256) * 256),
                      (unsigned)((M + 127) / 128), (unsigned)S},
                     {256, 1, 1});
          e.set_function(f_fold);
          e.set_buffer(0, sk); e.set_buffer(1, y);
          e.set_constant(2, (int)plane); e.set_constant(3, S);
          e.dispatch({(unsigned)plane, 1, 1}, {256, 1, 1});
        });
        std::printf(" S=%-2d(kc%4d) %5.0f@%4.0f/%4.0f |", S, kc, g, mhz,
                    mhz_max);
        if (g > best) { best = g; best_s = S; best_kc = kc; }
      }
      std::printf("\n[mlp_down]   BEST S=%d kc=%d %5.0f = %.2f of gate|up\n",
                  best_s, best_kc, best, g_ref > 0.0 ? best / g_ref : 0.0);
      EXPECT_TRUE(best > 0.0);
    }
  }
}

// Does the deep-K stall reach the int8 accel path too? The i8 GEMM walks K
// in GI8_KC chunks INSIDE one threadgroup (accumulating in threadgroup
// memory) over a 64x64 tile, so it has 8x the threadgroups of the 128x256
// f16 tile at the same shape -- the occupancy argument for split-K may
// simply not apply. Measured the same way: the down direction against the
// same shape's gate|up, both through I8GemmContext.
TEST(mlp_fuse, i8_deep_k)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[mlp_i8] no matrix cores -- skip\n");
    return;
  }
  if (!full_()) {
    std::printf("[mlp_i8] VPIPE_MLP_FUSE_AB=1 to run\n");
    return;
  }
  genai::I8GemmContext i8(mc, /*want=*/true, /*bf16=*/false);
  if (!i8.enabled()) {
    std::printf("[mlp_i8] i8 gemm unavailable -- skip\n");
    return;
  }
  struct Case {
    const char* tag;
    int hidden, ffn;
  };
  const Case cases[] = {
      {"h3-dit ff",      5376, 14336},
      {"qwen3.5-9B mlp", 4096, 12288},
      {"qwen-27B mlp",   5120, 17408},
      {"gemma-31B mlp",  5376, 21504},
  };
  for (const Case& c : cases) {
    for (int M : {1024, 4096}) {
      // down: [M, ffn] x [hidden, ffn]^T ; gate|up: [M, hidden] x
      // [2*ffn, hidden]^T. Dense f16 operands -- the i8 path requantizes
      // both sides itself, which is exactly what it costs in a forward.
      auto run = [&](int N, int K, double* mhz, double* mhz_max) {
        SharedBuffer x = make_act_(mc, (std::size_t)M * K, 51u);
        SharedBuffer w = make_act_(mc, (std::size_t)N * K, 53u);
        SharedBuffer y = mc->make_shared_buffer((std::size_t)M * N * 2);
        if (!i8.accepts(M, N, K)) { return 0.0; }
        const double f1 = 2.0 * M * N * K;
        const int iters = std::max(2, std::min(200, (int)(3.0e12 / f1)));
        return rate_(mc, f1, iters, mhz, mhz_max, [&](ComputeEncoder& e) {
          i8.gemm(e, x, 0, w, y, 0, M, N, K);
        });
      };
      double m1 = 0.0, m2 = 0.0, p1 = 0.0, p2 = 0.0;
      const double g_down = run(c.hidden, c.ffn, &m1, &p1);
      // gate|up contracts over `hidden`, and the i8 path only accepts K a
      // multiple of 512 -- hidden 5376 is not, so two of these shapes have
      // no i8 gate|up at all. That is a fact about the gate, not a failure.
      const double g_gu = run(2 * c.ffn, c.hidden, &m2, &p2);
      if (g_gu <= 0.0) {
        std::printf("[mlp_i8] %-16s M=%4d | down (N=%5d K=%5d) %5.0f "
                    "@%4.0f/%4.0fMHz | gate|up K=%d rejected (K %% 512 = "
                    "%d)\n",
                    c.tag, M, c.hidden, c.ffn, g_down, m1, p1, c.hidden,
                    c.hidden % 512);
        EXPECT_TRUE(g_down > 0.0);
        continue;
      }
      std::printf("[mlp_i8] %-16s M=%4d | down (N=%5d K=%5d) %5.0f "
                  "@%4.0f/%4.0fMHz | gate|up (N=%5d K=%5d) %5.0f "
                  "@%4.0f/%4.0fMHz | %.2f of gate|up\n",
                  c.tag, M, c.hidden, c.ffn, g_down, m1, p1, 2 * c.ffn,
                  c.hidden, g_gu, m2, p2, g_down / g_gu);
      EXPECT_TRUE(g_down > 0.0 && g_gu > 0.0);
    }
  }
}

// The shipped path, end to end: MmaSplitK::plan picking a chunk width, and
// MmaSplitK::encode running it in row blocks that fit the plane budget.
//
// Two things are checked that the kernel scan above cannot see. RATE: the
// chooser has to land near the best arm of the scan, and the row blocking
// (which the scan does not do) must not cost that back. ANSWER: the split
// reassociates the contraction, so the output moves -- and the shipped
// numbers for that were measured at S=4, while the chooser now reaches
// S=8..14. A decoder held to greedy token-exact cares about the tail of
// that distribution, so it is reported per depth rather than assumed to
// have stayed where it was.
TEST(mlp_fuse, splitk_plan)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.have_mma) {
    std::printf("[mlp_plan] no matrix cores -- skip\n");
    return;
  }
  ComputeLibrary lib_dense = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise");
  genai::MmaSplitK sk;
  sk.load(mc, lib_dense, lib_elt);
  if (!sk.enabled) {
    std::printf("[mlp_plan] split-K unavailable -- skip\n");
    return;
  }
  struct Case { const char* tag; int N, K; };
  const Case cases[] = {
      {"h3-dit fc2",      5376, 14336},
      {"qwen3.5-9B down", 4096, 12288},
      {"qwen-27B down",   5120, 17408},
      {"gemma-31B down",  5376, 21504},
  };
  // Down to 64 rows on purpose: the caller's matrix-core threshold is 64, so
  // that is the shortest prompt this path can see, and a rule that only ever
  // gets checked at prefill-sized M is how a split that LOSES at 91 rows
  // ships (measured: a 91-token turn ran 0.99 s of prefill split vs 0.38 s
  // unsplit before this test had a small-M rung).
  const std::vector<int> ms =
      full_() ? std::vector<int>{64, 128, 256, 512, 1024, 4096}
              : std::vector<int>{128, 1024};
  for (const Case& c : cases) {
    QW q = make_qw_(mc, c.N, c.K, 61u + (unsigned)c.K);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)c.N * c.K * 2);
    for (int M : ms) {
      SharedBuffer x = make_act_(mc, (std::size_t)M * c.K, 67u);
      SharedBuffer y1 = mc->make_shared_buffer((std::size_t)M * c.N * 2);
      SharedBuffer y2 = mc->make_shared_buffer((std::size_t)M * c.N * 2);
      const genai::MmaSplitK::Plan p = sk.plan(c.K, c.N, M);
      const int block = p.splits > 0
          ? std::min(M, genai::MmaSplitK::rows_per_block(p.splits, c.N)) : 0;
      // Dequant once; both arms below read the same dense weight.
      {
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          e.set_function(kn.dq);
          e.set_buffer(0, q.w); e.set_buffer(1, q.s); e.set_buffer(2, q.b);
          e.set_buffer(3, wdq);
          e.set_constant(4, c.K); e.set_constant(5, c.N);
          e.dispatch({(unsigned)(c.K / 8), (unsigned)c.N, 1}, {64, 1, 1});
        }
        st.commit().wait();
      }
      const double f1 = 2.0 * M * c.N * c.K;
      const int iters = std::max(2, std::min(200, (int)(3.0e12 / f1)));
      double m1 = 0.0, m2 = 0.0, p1 = 0.0, p2 = 0.0;
      const double g_single = rate_(mc, f1, iters, &m1, &p1,
                                    [&](ComputeEncoder& e) {
        e.set_function(kn.dense256);
        e.set_buffer(0, x); e.set_buffer(1, wdq); e.set_buffer(2, wdq);
        e.set_buffer(3, y1);
        e.set_constant(4, c.K); e.set_constant(5, c.N);
        e.set_constant(6, M); e.set_constant(7, 0);
        e.dispatch({(unsigned)(((c.N + 255) / 256) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      });
      // encode() can decline where plan() would not (a row block that does
      // not fit); measure only what it actually encodes, or an empty command
      // buffer reads as an infinitely fast GEMM.
      bool encoded = false;
      {
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          encoded = sk.encode(mc, e, x, wdq, y2, c.K, c.N, M);
        }
        st.commit().wait();
      }
      double g_plan = 0.0;
      if (p.fn != nullptr && encoded) {
        g_plan = rate_(mc, f1, iters, &m2, &p2, [&](ComputeEncoder& e) {
          sk.encode(mc, e, x, wdq, y2, c.K, c.N, M);
        });
      }
      std::printf("[mlp_plan] %-16s M=%4d N=%5d K=%5d | plan S=%d kc=%d "
                  "rows/block=%d | single %5.0f @%4.0fMHz | planned %5.0f "
                  "@%4.0fMHz = %.2fx",
                  c.tag, M, c.N, c.K, p.splits, p.splits ? c.K / p.splits : 0,
                  block, g_single, m1, g_plan, m2,
                  g_single > 0.0 ? g_plan / g_single : 0.0);
      if (p.fn != nullptr && encoded) {
        const auto* a = static_cast<const _Float16*>(y1.contents());
        const auto* b = static_cast<const _Float16*>(y2.contents());
        const std::size_t n = (std::size_t)M * c.N;
        std::size_t ndiff = 0;
        double maxabs = 0.0, maxrel = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          const double va = (double)a[i], vb = (double)b[i];
          if (va != vb) { ++ndiff; }
          const double d = std::fabs(va - vb);
          if (d > maxabs) { maxabs = d; }
          if (std::fabs(va) > 1e-3 && d / std::fabs(va) > maxrel) {
            maxrel = d / std::fabs(va);
          }
        }
        std::printf(" | differ %.2f%% max|d| %.5f max rel %.2e",
                    100.0 * (double)ndiff / (double)n, maxabs, maxrel);
      }
      std::printf("\n");
      EXPECT_TRUE(g_single > 0.0);
    }
  }
}

// The i8 path's K padding, checked against the thing it stands in for.
//
// A contraction that is not a multiple of 512 has no whole chunk to sit in,
// so I8GemmContext zero-fills up to the next one. That is only sound if the
// padded run is INDISTINGUISHABLE from one where the caller had padded the
// buffers itself -- zeros add nothing to a dot product, and cannot move a
// group's absmax scale. So the two are compared bit for bit rather than
// within a tolerance: any difference means the tail group quantized
// differently, which is the one way this can be subtly wrong.
//
// K = 5376 is MiniMax-H3's hidden, the width that kept qkv and fc1 off the
// int8 path before the padding existed.
TEST(mlp_fuse, i8_kpad)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid() || !mc->supports_matrix_cores()) {
    return;
  }
  genai::I8GemmContext i8(mc, /*want=*/true, /*bf16=*/false);
  if (!i8.enabled()) {
    std::printf("[mlp_i8pad] i8 gemm unavailable -- skip\n");
    return;
  }
  // The padding-cost gate: exact K is always in, a small pad on a deep K is
  // in, the same pad on a shallow K is out. 511 is the largest pad there
  // is, so everything at or above 10*512 = 5120 passes regardless.
  struct Gate { int K; bool want; const char* why; };
  const Gate gates[] = {
      {5120, true,  "exact, no pad"},
      {5376, true,  "pad 256 of 5376 = 4.8%"},
      {2816, true,  "pad 256 of 2816 = 9.1%"},
      {1600, false, "pad 448 of 1600 = 28%"},
      {1088, false, "pad 448 of 1088 = 41%"},
  };
  for (const Gate& g : gates) {
    const bool got = i8.accepts(4096, 512, g.K);
    std::printf("[mlp_i8pad] accepts(K=%4d) = %-5s (want %-5s) -- %s\n",
                g.K, got ? "true" : "false", g.want ? "true" : "false",
                g.why);
    EXPECT_TRUE(got == g.want);
  }
  const int M = 1024, N = 512, K = 5376, KP = 5632;
  if (!i8.accepts(M, N, K)) {
    std::printf("[mlp_i8pad] K=%d still rejected -- padding not wired\n", K);
    EXPECT_TRUE(false);
    return;
  }
  SharedBuffer x = make_act_(mc, (std::size_t)M * K, 71u);
  SharedBuffer w = make_act_(mc, (std::size_t)N * K, 73u);
  SharedBuffer y_pad = mc->make_shared_buffer((std::size_t)M * N * 2);
  SharedBuffer y_ref = mc->make_shared_buffer((std::size_t)M * N * 2);
  // The control: the same rows already widened to KP with explicit zeros.
  SharedBuffer xw = mc->make_shared_buffer((std::size_t)M * KP * 2);
  SharedBuffer ww = mc->make_shared_buffer((std::size_t)N * KP * 2);
  std::memset(xw.contents(), 0, (std::size_t)M * KP * 2);
  std::memset(ww.contents(), 0, (std::size_t)N * KP * 2);
  for (int r = 0; r < M; ++r) {
    std::memcpy(static_cast<char*>(xw.contents()) + (std::size_t)r * KP * 2,
                static_cast<const char*>(x.contents())
                    + (std::size_t)r * K * 2, (std::size_t)K * 2);
  }
  for (int r = 0; r < N; ++r) {
    std::memcpy(static_cast<char*>(ww.contents()) + (std::size_t)r * KP * 2,
                static_cast<const char*>(w.contents())
                    + (std::size_t)r * K * 2, (std::size_t)K * 2);
  }
  bool a = false, b = false;
  {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      a = i8.gemm(e, x, 0, w, y_pad, 0, M, N, K);
    }
    st.commit().wait();
  }
  {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      b = i8.gemm(e, xw, 0, ww, y_ref, 0, M, N, KP);
    }
    st.commit().wait();
  }
  EXPECT_TRUE(a && b);
  if (!a || !b) { return; }
  const auto* pa = static_cast<const _Float16*>(y_pad.contents());
  const auto* pb = static_cast<const _Float16*>(y_ref.contents());
  std::size_t ndiff = 0;
  double maxabs = 0.0;
  for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
    if (pa[i] != pb[i]) { ++ndiff; }
    maxabs = std::max(maxabs, std::fabs((double)pa[i] - (double)pb[i]));
  }
  std::printf("[mlp_i8pad] K=%d vs explicit KP=%d: %zu/%d differ, max|d| "
              "%.3e\n", K, KP, ndiff, M * N, maxabs);
  EXPECT_TRUE(ndiff == 0);
}

// The band/split interaction for the H3 video VAE's encoder convs.
//
// A 3x3x3 conv runs as an im2col GEMM of [band, 27*cin] x [cout, 27*cin]^T,
// and the band is col_cap/(27*cin) so the gather stays a fixed size. That
// couples two decisions that are usually independent: how many rows the GEMM
// gets, and whether splitting its contraction pays. Today's 16 MB cap gives
// the DEEPEST level -- the only one whose weight is big enough to want a
// split -- the FEWEST rows (303), which is why the split declines there.
//
// So this sweeps the band as if col_cap had been raised, and reports both
// arms at each. Dense weights, no dequant: the VAE's are not quantized, and
// the band decision must not be read off a harness that pays a cost the
// tower does not.
TEST(mlp_fuse, vae_band)
{
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.have_mma) { return; }
  if (!full_()) {
    std::printf("[mlp_band] VPIPE_MLP_FUSE_AB=1 to run\n");
    return;
  }
  ComputeFunction f_fold =
      mc->load_library("llm_elementwise").function("splitk_fold_f32_f16");
  struct Lv { const char* tag; int N, K, S; };
  const Lv lvs[] = {
      {"enc cin=1024", 1024, 27648, 9},
      {"enc cin=512",   512, 13824, 6},
  };
  // A ladder that is not powers of two: the first sweep read as
  // non-monotone, and a doubling ladder cannot tell a real optimum from an
  // interaction with the 128-row tile quantization.
  const int ms[] = {303, 404, 606, 808, 1213, 1617, 2427, 3234, 4096};
  const int kArms = 3;   // n128, n256, split(S)
  const char* arm_tag[kArms] = {"n128", "n256", "split"};
  for (const Lv& lv : lvs) {
    SharedBuffer w = make_act_(mc, (std::size_t)lv.N * lv.K, 81u);
    ComputeFunction f_sk = kn.lib_dense.function(
        std::string("dense_gemm_mma_splitk32_n128x256_k")
        + std::to_string(lv.K / lv.S) + "_f16");
    const std::size_t nm = sizeof(ms) / sizeof(ms[0]);
    std::vector<std::vector<double>> got(nm * kArms);
    // THREE passes over the whole ladder rather than three rounds at each
    // rung: a rung measured to convergence before the next one starts wears
    // whatever the clock was doing during it, and that is exactly what makes
    // a thermal drift look like a shape effect.
    for (int pass = 0; pass < 3; ++pass) {
      for (std::size_t mi = 0; mi < nm; ++mi) {
        const int M = ms[mi];
        SharedBuffer x = make_act_(mc, (std::size_t)M * lv.K, 83u);
        SharedBuffer y = mc->make_shared_buffer((std::size_t)M * lv.N * 2);
        const std::size_t plane = (std::size_t)M * lv.N;
        const double f1 = 2.0 * M * lv.N * lv.K;
        const int iters = std::max(2, std::min(200, (int)(2.0e12 / f1)));
        SharedBuffer sk = f_sk.valid()
            ? mc->make_shared_buffer(plane * (std::size_t)lv.S * 4)
            : SharedBuffer{};
        for (int a = 0; a < kArms; ++a) {
          if (a == 2 && (!f_sk.valid() || sk.empty())) { continue; }
          double mhz = 0.0, mx = 0.0;
          const double g = rate_(mc, f1, iters, &mhz, &mx,
                                 [&](ComputeEncoder& e) {
            if (a == 2) {
              e.set_function(f_sk);
              e.set_buffer(0, x); e.set_buffer(1, w); e.set_buffer(2, sk);
              e.set_constant(3, lv.K); e.set_constant(4, lv.N);
              e.set_constant(5, M);
              e.dispatch({(unsigned)(((lv.N + 255) / 256) * 256),
                          (unsigned)((M + 127) / 128), (unsigned)lv.S},
                         {256, 1, 1});
              e.set_function(f_fold);
              e.set_buffer(0, sk); e.set_buffer(1, y);
              e.set_constant(2, (int)plane); e.set_constant(3, lv.S);
              e.dispatch({(unsigned)plane, 1, 1}, {256, 1, 1});
              return;
            }
            const int BN = (a == 0) ? 128 : 256;
            e.set_function(a == 0 ? kn.dense128 : kn.dense256);
            e.set_buffer(0, x); e.set_buffer(1, w); e.set_buffer(2, w);
            e.set_buffer(3, y);
            e.set_constant(4, lv.K); e.set_constant(5, lv.N);
            e.set_constant(6, M); e.set_constant(7, 0);
            e.dispatch({(unsigned)(((lv.N + BN - 1) / BN) * 256),
                        (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
          });
          got[mi * kArms + (std::size_t)a].push_back(g);
        }
      }
    }
    for (std::size_t mi = 0; mi < nm; ++mi) {
      const int M = ms[mi];
      const double cap_mb =
          (double)((std::size_t)M * lv.K * 2) / (1024.0 * 1024.0);
      // x + weight, the two operands competing for cache at this band.
      const double live_mb = cap_mb +
          (double)((std::size_t)lv.N * lv.K * 2) / (1024.0 * 1024.0);
      std::printf("[mlp_band] %-13s band=%4d cap=%4.0fMB live=%4.0fMB |",
                  lv.tag, M, cap_mb, live_mb);
      double best = 0.0;
      const char* bt = "-";
      for (int a = 0; a < kArms; ++a) {
        const double g = median_(got[mi * kArms + (std::size_t)a]);
        std::printf(" %s %5.0f |", arm_tag[a], g);
        if (g > best) { best = g; bt = arm_tag[a]; }
      }
      std::printf(" best %s %5.0f\n", bt, best);
      EXPECT_TRUE(best > 0.0);
    }
  }
}

// A telemetry WATCHER, not a benchmark: samples the GPU while something
// else runs and prints a line per window.
//
// It exists because the interesting runs on this machine are whole
// generations driven by `vpipe`, and a fanless box's numbers are
// uninterpretable without the clock beside them -- a slow step and a
// throttled step look identical in a wall clock. This does no GPU work
// of its own (IOReport/SMC reads only), so it can sit alongside a real
// run without competing for the device, which matters here: two heavy
// GPU tasks on this box wedge it rather than merely sharing it.
//
// VPIPE_GPU_WATCH=<seconds> turns it on; VPIPE_GPU_WATCH_EVERY=<seconds>
// sets the window (default 10).
TEST(mlp_fuse, gpu_watch)
{
  const char* on = std::getenv("VPIPE_GPU_WATCH");
  if (on == nullptr || *on == '\0') { return; }
  const double total = std::atof(on);
  const char* ev = std::getenv("VPIPE_GPU_WATCH_EVERY");
  const double win = (ev != nullptr && *ev != '\0') ? std::atof(ev) : 10.0;
  if (total <= 0.0 || win <= 0.0) { return; }

  std::printf("[gpu] %s, %.0f cores | watching %.0fs in %.0fs windows\n",
              GpuTelemetrySampler::gpu_model().c_str(),
              (double)GpuTelemetrySampler::gpu_core_count(), total, win);
  std::fflush(stdout);
  const auto t0 = std::chrono::steady_clock::now();
  int n = 0;
  while (std::chrono::duration<double>(
             std::chrono::steady_clock::now() - t0).count() < total) {
    GpuTelemetrySampler tel;
    tel.start();
    const auto w0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(
               std::chrono::steady_clock::now() - w0).count() < win) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    const GpuTelemetry t = tel.stop();
    const double el = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[gpu] t=%6.0fs  clock %4.0f/%4.0f/%4.0f MHz  util %3.0f%%  "
                "power %5.1f W  temp %4.1f C  rss %6.0f MB  thermal %s\n",
                el,
                t.freq_mhz.ok ? t.freq_mhz.min : 0.0,
                t.freq_mhz.ok ? t.freq_mhz.avg : 0.0,
                t.freq_mhz.ok ? t.freq_mhz.max : 0.0,
                t.util_pct.ok ? t.util_pct.avg : 0.0,
                t.power_w.ok ? t.power_w.avg : 0.0,
                t.temp_c.ok ? t.temp_c.avg : 0.0,
                t.footprint_mb.ok ? t.footprint_mb.avg : 0.0,
                t.thermal_state.c_str());
    std::fflush(stdout);
    ++n;
  }
  std::printf("[gpu] %d windows\n", n);
  EXPECT_TRUE(n > 0);
}
