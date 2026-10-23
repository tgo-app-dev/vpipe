// metal-compute-gemm-mma.cc -- correctness + throughput for the M5
// matrix-core dense GEMM (dense_gemm_mma, MetalPerformancePrimitives
// matmul2d over cooperative_tensor) vs the steel simdgroup_matrix dense
// GEMM (dense_gemm_t). Both compute y[M,N] = x[M,K] @ W[N,K]^T in bf16
// with f32 accumulation, so the matrix-core output must match the steel
// output (and a CPU f32 oracle) within bf16 rounding.
//
// Runs in BOTH builds (no MLX dependency -- the matrix-core path is the
// no-MLX shipping target). On a GPU without matrix cores the
// dense_gemm_mma function fails to validate; the test then exercises
// only the steel path and notes the skip (never a false pass).
//
// The throughput bench (always prints) is the linchpin number for the
// M5 tensor-core bring-up: GFLOP/s steel vs matrix-core at the
// Qwen3.5-4B projection shapes.

#include <algorithm>
#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"
#include "generative-models/shared/mma-tile.h"

#include <Metal/Metal.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <tuple>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

// bf16 <-> f32. bf16 is the high 16 bits of f32; round-to-nearest-even
// on the way down so test inputs match what the GPU sees.
inline std::uint16_t f32_to_bf16(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, 4);
  const std::uint32_t rounding = 0x7fff + ((x >> 16) & 1);
  return (std::uint16_t)((x + rounding) >> 16);
}
inline float bf16_to_f32(std::uint16_t h) {
  std::uint32_t x = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

MetalCompute* get_mc_(Session& s) {
  MetalCompute* mc = s.metal_compute();
  return (mc != nullptr && mc->valid()) ? mc : nullptr;
}

double secs_(std::chrono::steady_clock::time_point a,
             std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// Fill with small deterministic pseudo-random bf16 values.
void fill_bf16(std::vector<std::uint16_t>& v, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  for (auto& e : v) { e = f32_to_bf16(d(rng) * 0.1f); }
}

struct Shape { int M, N, K; const char* tag; };

}  // namespace

// Correctness: matrix-core dense GEMM == steel dense GEMM == CPU oracle,
// within bf16 rounding. Skips the matrix-core half (with a printed note)
// if the GPU has no matrix cores.
TEST(gemm_mma, correctness) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }

  ComputeLibrary lib_steel = mc->load_library("dense_gemm_bf16");
  ComputeLibrary lib_mma   = mc->load_library("dense_gemm_mma_bf16");
  ComputeFunction f_steel = lib_steel.function("dense_gemm_t_f16");
  ComputeFunction f_mma   = lib_mma.function("dense_gemm_mma_t_f16");
  ASSERT_TRUE(f_steel.valid());
  const bool have_mma = f_mma.valid();
  if (!have_mma) {
    std::printf("[gemm_mma] matrix-core fn unavailable -- no tensor "
                "cores on this GPU; steel-only.\n");
  }

  const Shape shapes[] = {
      {64, 64, 64, "tiny"},
      {128, 256, 128, "small"},
      {320, 512, 256, "mid-unaligned"},   // M,N not multiples of 64? 320/64=5,512/64=8 ok
      {200, 320, 192, "edge"},            // none multiple of 64 -> exercises tails
  };

  for (const auto& sh : shapes) {
    const int M = sh.M, N = sh.N, K = sh.K;
    std::vector<std::uint16_t> x((std::size_t)M * K), W((std::size_t)N * K);
    fill_bf16(x, 1000 + M + N + K);
    fill_bf16(W, 7000 + M + N + K);

    // CPU oracle in f32 from the bf16-rounded inputs.
    std::vector<float> ref((std::size_t)M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
          acc += bf16_to_f32(x[(std::size_t)m * K + k]) *
                 bf16_to_f32(W[(std::size_t)n * K + k]);
        }
        ref[(std::size_t)m * N + n] = acc;
      }
    }

    SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * K * 2);
    SharedBuffer wb = mc->make_shared_buffer((std::size_t)N * K * 2);
    SharedBuffer bb = mc->make_shared_buffer((std::size_t)N * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    std::memcpy(xb.contents(), x.data(), x.size() * 2);
    std::memcpy(wb.contents(), W.data(), W.size() * 2);
    std::memset(bb.contents(), 0, (std::size_t)N * 2);

    auto run = [&](ComputeFunction& fn, bool mma) {
      std::memset(yb.contents(), 0, (std::size_t)M * N * 2);
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, xb);
        enc.set_buffer(1, wb);
        enc.set_buffer(2, bb);
        enc.set_buffer(3, yb);
        enc.set_constant(4, K);
        enc.set_constant(5, N);
        enc.set_constant(6, M);
        enc.set_constant(7, 0);
        if (mma) {
          enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                        (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
        } else {
          enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                        (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
        }
      }
      st.commit().wait();
      std::vector<float> out((std::size_t)M * N);
      const auto* yp = static_cast<const std::uint16_t*>(yb.contents());
      for (std::size_t i = 0; i < out.size(); ++i) { out[i] = bf16_to_f32(yp[i]); }
      return out;
    };

    auto rel_l2 = [&](const std::vector<float>& got) {
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - (double)ref[i];
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
      }
      return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    };

    const auto ys = run(f_steel, false);
    const double e_steel = rel_l2(ys);
    std::printf("[gemm_mma] %-14s M=%d N=%d K=%d  steel rel-L2=%.4e",
                sh.tag, M, N, K, e_steel);
    EXPECT_TRUE(e_steel < 3e-2);

    if (have_mma) {
      const auto ym = run(f_mma, true);
      const double e_mma = rel_l2(ym);
      std::printf("  mma rel-L2=%.4e", e_mma);
      EXPECT_TRUE(e_mma < 3e-2);
    }
    std::printf("\n");
  }
}

// THE PRODUCTION TILES AT A DiT's REAL DEPTHS, against f64. `correctness`
// above holds only the 64x64 reference tile, at K <= 256, to 3e-2 --
// while every image DiT's block projections run the 128-wide tiles at K
// 3072..12288 and thousands of rows. Here each production tile at
// Qwen-Image-2.1's shapes (hidden 4096, ffn 12288; 4160 joint rows at
// 1024^2, a ragged last row tile): n128 at K 4096 (q / k / v, out,
// ff-up), the deep n128x256 at K 12288 (ff-down), and tn2 at K 8192
// (other families' middle depth), each dispatched as the model does --
// beside the steel tile and the int8 accurate GEMM on the SAME operands.
// The oracle is f64 from the bf16-rounded inputs over sampled rows: every
// row tile's first and last row, the tail's, and random ones -- so a
// tile, a row band or a column region that is wrong shows in the
// per-row worst as well as the total. Activations carry outlier
// channels, as a DiT's hidden states do.
//
// A bf16 GEMM is held to its output rounding (~1e-3 rel-L2); the int8
// tier is ~7e-3, so a dense tile at or past the int8 GEMM's distance is
// a defect, not a precision trade.
TEST(gemm_mma, production_tiles_match_f64_at_dit_depths) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[gemm_mma] no matrix cores -- the production tiles do "
                "not run here; skipped\n");
    return;
  }
  ComputeLibrary lib = mc->load_library("dense_gemm_mma_bf16");
  ComputeLibrary lib_steel = mc->load_library("dense_gemm_bf16");
  ComputeFunction f_steel = lib_steel.function("dense_gemm_t_bm64bn64_f16");
  vpipe::genai::I8GemmContext i8(mc, /*want=*/true, /*bf16=*/true);
  i8.set_tier(vpipe::genai::I8GemmContext::Tier::kAccurate);
  struct Arm { const char* fn; int rn; int M, N, K; const char* what; };
  const Arm arms[] = {
      {"dense_gemm_mma_t_n128_f16", 128, 4160, 12288, 4096, "qkv"},
      {"dense_gemm_mma_t_n128_f16", 128, 4160, 4096, 4096, "out"},
      {"dense_gemm_mma_t_n128x256_f16", 256, 4160, 4096, 12288, "ff-down"},
      {"dense_gemm_mma_t_n128x256_tn2_f16", 512, 4160, 4096, 8192,
       "mid depth"},
  };
  for (const Arm& a : arms) {
    ComputeFunction fn = lib.function(a.fn);
    ASSERT_TRUE(fn.valid());
    if (!fn.valid()) { continue; }
    const int M = a.M, N = a.N, K = a.K;
    std::vector<std::uint16_t> x((std::size_t)M * K), W((std::size_t)N * K);
    {
      std::mt19937 rng(91 + N + K);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      // Eight outlier channels at 30x: a DiT's hidden states.
      std::vector<float> cs((std::size_t)K, 1.0f);
      for (int i = 0; i < 8; ++i) { cs[(std::size_t)(rng() % K)] = 30.0f; }
      for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = f32_to_bf16(nd(rng) * cs[i % (std::size_t)K]);
      }
      const float ws = 1.0f / std::sqrt((float)K);
      for (auto& e : W) { e = f32_to_bf16(nd(rng) * ws); }
    }
    // Sampled rows: each 128-row tile's first and last, and 32 random.
    std::vector<int> rows;
    for (int m0 = 0; m0 < M; m0 += 128) {
      rows.push_back(m0);
      rows.push_back(std::min(M, m0 + 128) - 1);
    }
    {
      std::mt19937 rng(7 + M);
      for (int i = 0; i < 32; ++i) { rows.push_back((int)(rng() % M)); }
    }
    std::vector<double> ref(rows.size() * (std::size_t)N);
    {
      std::vector<double> xr((std::size_t)K), wr((std::size_t)N * K);
      for (std::size_t i = 0; i < wr.size(); ++i) { wr[i] = bf16_to_f32(W[i]); }
      for (std::size_t r = 0; r < rows.size(); ++r) {
        const std::size_t m = (std::size_t)rows[r];
        for (int k = 0; k < K; ++k) {
          xr[(std::size_t)k] = bf16_to_f32(x[m * K + k]);
        }
        for (int n = 0; n < N; ++n) {
          const double* wn = &wr[(std::size_t)n * K];
          double acc = 0.0;
          for (int k = 0; k < K; ++k) { acc += xr[(std::size_t)k] * wn[k]; }
          ref[r * N + n] = acc;
        }
      }
    }
    SharedBuffer xb = mc->make_shared_buffer(x.size() * 2);
    SharedBuffer wb = mc->make_shared_buffer(W.size() * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    ASSERT_TRUE(!xb.empty() && !wb.empty() && !yb.empty());
    if (xb.empty() || wb.empty() || yb.empty()) { continue; }
    std::memcpy(xb.contents(), x.data(), x.size() * 2);
    std::memcpy(wb.contents(), W.data(), W.size() * 2);
    // rel-L2 over the sampled rows, and the worst single row's.
    auto score = [&](double* worst) {
      const auto* yp = static_cast<const std::uint16_t*>(yb.contents());
      double num = 0.0, den = 0.0;
      *worst = 0.0;
      for (std::size_t r = 0; r < rows.size(); ++r) {
        double rn = 0.0, rd = 0.0;
        for (int n = 0; n < N; ++n) {
          const double g = bf16_to_f32(yp[(std::size_t)rows[r] * N + n]);
          const double e = g - ref[r * N + n];
          rn += e * e;
          rd += ref[r * N + n] * ref[r * N + n];
        }
        num += rn; den += rd;
        *worst = std::max(*worst, rd > 0 ? std::sqrt(rn / rd) : 0.0);
      }
      return den > 0 ? std::sqrt(num / den) : 0.0;
    };
    auto run = [&](const std::function<bool(ComputeEncoder&)>& enc_fn,
                   double* worst) {
      std::memset(yb.contents(), 0xff, (std::size_t)M * N * 2);   // NaN
      CommandStream st = mc->make_command_stream();
      bool ok = false;
      {
        ComputeEncoder enc = st.begin_compute();
        ok = enc_fn(enc);
      }
      st.commit().wait();
      return ok ? score(worst) : -1.0;
    };
    double w_mma = 0.0, w_steel = 0.0, w_i8 = 0.0;
    const double e_mma = run([&](ComputeEncoder& enc) {
      enc.set_function(fn);
      enc.set_buffer(0, xb); enc.set_buffer(1, wb); enc.set_buffer(2, wb);
      enc.set_buffer(3, yb);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + a.rn - 1) / a.rn) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      return true;
    }, &w_mma);
    const double e_steel = !f_steel.valid() ? -1.0 : run(
        [&](ComputeEncoder& enc) {
          enc.set_function(f_steel);
          enc.set_buffer(0, xb); enc.set_buffer(1, wb);
          enc.set_buffer(2, wb); enc.set_buffer(3, yb);
          enc.set_constant(4, K); enc.set_constant(5, N);
          enc.set_constant(6, M); enc.set_constant(7, 0);
          enc.dispatch({(unsigned)(((N + 63) / 64) * 32),
                        (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
          return true;
        }, &w_steel);
    const double e_i8 = !i8.enabled() ? -1.0 : run(
        [&](ComputeEncoder& enc) {
          return i8.gemm(enc, xb, 0, wb, yb, 0, M, N, K);
        }, &w_i8);
    std::printf("[gemm_mma] %-9s M=%d N=%d K=%d %-33s rel-L2 %.3e (worst "
                "row %.3e); steel %.3e (%.3e); int8 accurate %.3e (%.3e)\n",
                a.what, M, N, K, a.fn, e_mma, w_mma, e_steel, w_steel, e_i8,
                w_i8);
    EXPECT_TRUE(e_mma >= 0.0 && e_mma < 2e-3 && w_mma < 4e-3);
    if (e_steel >= 0.0) { EXPECT_TRUE(e_mma <= 1.5 * e_steel + 1e-4); }
    if (e_i8 >= 0.0) { EXPECT_TRUE(e_mma < e_i8); }
  }
}

// Split-K deep-reduction dense GEMM must match the single-op deep kernel
// (dense_gemm_mma_t_n128x256_f16) within f16 double-rounding. Validates that
// slice(k0,..) + a compile-time KC contraction extent reads the intended K
// sub-range with the correct per-row stride. K = 16384 = 2 * 8192 (the ff-down
// shape); M has a tail (not a multiple of 128).
TEST(gemm_mma, splitk_matches_single_op) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("dense_gemm_mma_bf16");
  ComputeFunction f_ref = lib.function("dense_gemm_mma_t_n128x256_f16");
  ComputeFunction f_spl = lib.function("dense_gemm_mma_splitk_n128x256_k8192_f16");
  if (!f_ref.valid() || !f_spl.valid()) {
    std::printf("[splitk] matrix-core fn unavailable -- skip\n");
    return;
  }
  const int M = 260, N = 512, K = 16384;   // 2 N-tiles, 3 M-tiles (tail), 2 splits
  std::vector<std::uint16_t> x((std::size_t)M * K), W((std::size_t)N * K);
  fill_bf16(x, 101);
  fill_bf16(W, 202);
  SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * K * 2);
  SharedBuffer wb = mc->make_shared_buffer((std::size_t)N * K * 2);
  SharedBuffer bb = mc->make_shared_buffer((std::size_t)N * 2);
  SharedBuffer yref = mc->make_shared_buffer((std::size_t)M * N * 2);
  SharedBuffer yp = mc->make_shared_buffer((std::size_t)2 * M * N * 2);
  std::memcpy(xb.contents(), x.data(), x.size() * 2);
  std::memcpy(wb.contents(), W.data(), W.size() * 2);
  std::memset(bb.contents(), 0, (std::size_t)N * 2);

  {
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder enc = st.begin_compute();
      enc.set_function(f_ref);
      enc.set_buffer(0, xb); enc.set_buffer(1, wb); enc.set_buffer(2, wb);
      enc.set_buffer(3, yref);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + 255) / 256) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      // split-K into 2 planes (BM=128, BN=256).
      enc.set_function(f_spl);
      enc.set_buffer(0, xb); enc.set_buffer(1, wb); enc.set_buffer(2, yp);
      enc.set_constant(3, K); enc.set_constant(4, N); enc.set_constant(5, M);
      enc.dispatch({(unsigned)(((N + 255) / 256) * 256),
                    (unsigned)((M + 127) / 128), 2}, {256, 1, 1});
    }
    st.commit().wait();
  }

  const auto* rp = static_cast<const std::uint16_t*>(yref.contents());
  const auto* pp = static_cast<const std::uint16_t*>(yp.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
    const float ref = bf16_to_f32(rp[i]);
    const float got = bf16_to_f32(pp[i]) + bf16_to_f32(pp[(std::size_t)M * N + i]);
    num += (double)(got - ref) * (got - ref);
    den += (double)ref * ref;
  }
  const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  std::printf("[splitk] M=%d N=%d K=%d  split-K vs single-op rel-L2=%.3e\n",
              M, N, K, rel);
  EXPECT_TRUE(rel < 5e-3);
}

namespace {

// Build a 4-bit affine-quantized weight [N,K] (group 64): random nibbles
// q in [0,15], per-group bf16 scale + bias. Returns packed weight bytes
// (row-major, 2 nibbles/byte: even k = low nibble), scales, biases, and
// the f32 dequantized matrix for the oracle.
struct QWeight {
  std::vector<std::uint8_t> packed;     // N * (K/2) bytes
  std::vector<std::uint16_t> scales;    // N * (K/64) bf16
  std::vector<std::uint16_t> biases;    // N * (K/64) bf16
  std::vector<float> deq;               // N * K  f32
};
QWeight make_qweight(int N, int K, std::uint32_t seed, int group = 64) {
  const int G = K / group;
  QWeight q;
  q.packed.assign((std::size_t)N * (K / 2), 0);   // 4-bit: 2/byte
  q.scales.resize((std::size_t)N * G);
  q.biases.resize((std::size_t)N * G);
  q.deq.resize((std::size_t)N * K);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> nib(0, 15);
  std::uniform_real_distribution<float> sd(0.005f, 0.02f);
  std::uniform_real_distribution<float> bd(-0.1f, 0.1f);
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < G; ++g) {
      const float s = sd(rng), b = bd(rng);
      q.scales[(std::size_t)n * G + g] = f32_to_bf16(s);
      q.biases[(std::size_t)n * G + g] = f32_to_bf16(b);
    }
    for (int k = 0; k < K; ++k) {
      const int v = nib(rng);
      const std::size_t bi = (std::size_t)n * (K / 2) + (k >> 1);
      if (k & 1) { q.packed[bi] |= (std::uint8_t)(v << 4); }
      else       { q.packed[bi] |= (std::uint8_t)v; }
      const int g = k / group;
      const float s = bf16_to_f32(q.scales[(std::size_t)n * G + g]);
      const float b = bf16_to_f32(q.biases[(std::size_t)n * G + g]);
      q.deq[(std::size_t)n * K + k] = s * (float)v + b;
    }
  }
  return q;
}

}  // namespace

// Ragged N (N % BN != 0) through the steel qmm. The kernel used to be
// instantiated with a compile-time aligned_N promise that let the WEIGHT loader
// run load_unsafe() on the partial last N-tile -- reading a full BN=32 rows and
// so up to 31 rows past the end of the codes / scales / biases. The store was
// already bound-checked, so results were right and the over-read was a silent
// fault risk; Boogu-Image's GQA k/v projections (N = 7*120 = 840) are the first
// real caller with a ragged N. These widths cover the awkward cases: one row
// past a tile boundary, one row short of one, and Boogu's own 840.
//
// HONEST SCOPE OF THIS TEST: the output is correct EITHER WAY -- the over-read
// only ever polluted accumulator columns >= N, which store_result_safe drops --
// and it was CHECKED that pre-fix and post-fix outputs are identical to the last
// digit (2.4483e-03 for boogu-kv both ways). Metal shader validation
// (MTL_SHADER_VALIDATION=1, incl. _GLOBAL_MEMORY) does NOT flag the pre-fix
// read either, so no tool guards this. What this test does cover is that ragged
// N is arithmetically right at all -- there was NO ragged-N coverage before, the
// nearest case being N=320 -- and the over-read is now impossible by
// construction (the loader cannot address past num_outs). If someone
// "optimizes" the safe loader back out, this test will still pass; the guard is
// the note on qmm_t_impl.
TEST(qmm_mma, ragged_n_matches_reference) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("affine_qmm_steel_bf16");
  ComputeFunction f4 = lib.function("affine_qmm_steel_w4g64");
  ComputeFunction f4g32 = lib.function("affine_qmm_steel_w4g32");
  ASSERT_TRUE(f4.valid() && f4g32.valid());

  // group must divide K, so Boogu's K=3360 (52.5 groups of 64) rides the g32
  // entry point -- the same reason its checkpoints are quantized at g32.
  struct RShape { int M, N, K, group; const char* tag; };
  const RShape shapes[] = {
      {64, 840, 3360, 32, "boogu-kv"},   // 26.25 tiles of 32
      {33, 33, 128, 64, "n33"},          // 1 row into the last tile
      {64, 95, 192, 64, "n95"},          // 1 row short of a full tile
      {96, 161, 256, 64, "n161"},
  };
  for (const auto& sh : shapes) {
    const int M = sh.M, N = sh.N, K = sh.K;
    ComputeFunction& fn = (sh.group == 32) ? f4g32 : f4;
    std::vector<std::uint16_t> x((std::size_t)M * K);
    fill_bf16(x, 1234 + N);
    QWeight q = make_qweight(N, K, 4321 + N, sh.group);

    std::vector<float> ref((std::size_t)M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
          acc += bf16_to_f32(x[(std::size_t)m * K + k]) *
                 q.deq[(std::size_t)n * K + k];
        }
        ref[(std::size_t)m * N + n] = acc;
      }
    }
    // Buffers sized EXACTLY to the tensors: no slack for an over-read to land
    // in, so shader validation has something to catch.
    SharedBuffer wb = mc->make_shared_buffer(q.packed.size());
    SharedBuffer sb = mc->make_shared_buffer(q.scales.size() * 2);
    SharedBuffer bb = mc->make_shared_buffer(q.biases.size() * 2);
    SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * K * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    ASSERT_TRUE(!wb.empty() && !sb.empty() && !bb.empty() && !yb.empty());
    std::memcpy(wb.contents(), q.packed.data(), q.packed.size());
    std::memcpy(sb.contents(), q.scales.data(), q.scales.size() * 2);
    std::memcpy(bb.contents(), q.biases.data(), q.biases.size() * 2);
    std::memcpy(xb.contents(), x.data(), x.size() * 2);
    std::memset(yb.contents(), 0, (std::size_t)M * N * 2);
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, wb); enc.set_buffer(1, sb); enc.set_buffer(2, bb);
        enc.set_buffer(3, xb); enc.set_buffer(4, yb);
        enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
        enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                      (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
      }
      st.commit().wait();
    }
    const auto* yp = static_cast<const std::uint16_t*>(yb.contents());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
      const double d = (double)bf16_to_f32(yp[i]) - (double)ref[i];
      num += d * d; den += (double)ref[i] * (double)ref[i];
    }
    const double e = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    std::printf("[qmm_mma] ragged %-9s M=%d N=%d K=%d  rel-L2=%.4e\n",
                sh.tag, M, N, K, e);
    EXPECT_TRUE(e < 3e-2);
  }
}

// Diagnostic: print the GPU's family / capability signals so we can pick
// a deterministic gate for the matrix-core path.
TEST(gpu_caps, report) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  MTL::Device* dev = mc->device();
  ASSERT_TRUE(dev != nullptr);
  auto sf = [&](MTL::GPUFamily f) { return dev->supportsFamily(f) ? 1 : 0; };
  std::printf("[gpu_caps] name=%s\n", dev->name()->utf8String());
  std::printf("[gpu_caps] Apple7=%d Apple8=%d Apple9=%d Apple10=%d "
              "Metal3=%d Metal4=%d\n",
              sf(MTL::GPUFamilyApple7), sf(MTL::GPUFamilyApple8),
              sf(MTL::GPUFamilyApple9), sf(MTL::GPUFamilyApple10),
              sf(MTL::GPUFamilyMetal3), sf(MTL::GPUFamilyMetal4));
  std::printf("[gpu_caps] maxThreadgroupMemory=%lu\n",
              (unsigned long)dev->maxThreadgroupMemoryLength());
}

// 4-bit quantized matrix-core GEMM == steel quantized GEMM == oracle.
TEST(qmm_mma, correctness) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib_steel = mc->load_library("affine_qmm_steel_bf16");
  ComputeLibrary lib_mma   = mc->load_library("affine_qmm_mma_bf16");
  ComputeFunction f_steel = lib_steel.function("affine_qmm_steel_w4g64");
  ComputeFunction f_mma   = lib_mma.function("affine_qmm_mma_w4g64");
  ASSERT_TRUE(f_steel.valid());
  const bool have_mma = f_mma.valid();
  if (!have_mma) {
    std::printf("[qmm_mma] matrix-core fn unavailable -- steel-only.\n");
  }

  const Shape shapes[] = {
      {64, 64, 64, "tiny"},
      {96, 128, 128, "small"},
      {130, 320, 256, "edge"},     // M,N tails
      {512, 256, 2560, "deepK"},
  };
  for (const auto& sh : shapes) {
    const int M = sh.M, N = sh.N, K = sh.K;
    std::vector<std::uint16_t> x((std::size_t)M * K);
    fill_bf16(x, 555 + M + N + K);
    QWeight q = make_qweight(N, K, 999 + N + K);

    std::vector<float> ref((std::size_t)M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
          acc += bf16_to_f32(x[(std::size_t)m * K + k]) *
                 q.deq[(std::size_t)n * K + k];
        }
        ref[(std::size_t)m * N + n] = acc;
      }
    }

    SharedBuffer wb = mc->make_shared_buffer(q.packed.size());
    SharedBuffer sb = mc->make_shared_buffer(q.scales.size() * 2);
    SharedBuffer bb = mc->make_shared_buffer(q.biases.size() * 2);
    SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * K * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    std::memcpy(wb.contents(), q.packed.data(), q.packed.size());
    std::memcpy(sb.contents(), q.scales.data(), q.scales.size() * 2);
    std::memcpy(bb.contents(), q.biases.data(), q.biases.size() * 2);
    std::memcpy(xb.contents(), x.data(), x.size() * 2);

    auto run = [&](ComputeFunction& fn, bool mma) {
      std::memset(yb.contents(), 0, (std::size_t)M * N * 2);
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, wb); enc.set_buffer(1, sb); enc.set_buffer(2, bb);
        enc.set_buffer(3, xb); enc.set_buffer(4, yb);
        enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
        if (mma) {
          enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                        (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
        } else {
          enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                        (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
        }
      }
      st.commit().wait();
      std::vector<float> out((std::size_t)M * N);
      const auto* yp = static_cast<const std::uint16_t*>(yb.contents());
      for (std::size_t i = 0; i < out.size(); ++i) { out[i] = bf16_to_f32(yp[i]); }
      return out;
    };
    auto rel_l2 = [&](const std::vector<float>& got) {
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - (double)ref[i];
        num += d * d; den += (double)ref[i] * (double)ref[i];
      }
      return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    };

    const double e_steel = rel_l2(run(f_steel, false));
    std::printf("[qmm_mma] %-8s M=%d N=%d K=%d  steel rel-L2=%.4e",
                sh.tag, M, N, K, e_steel);
    EXPECT_TRUE(e_steel < 3e-2);
    if (have_mma) {
      const double e_mma = rel_l2(run(f_mma, true));
      std::printf("  mma rel-L2=%.4e", e_mma);
      EXPECT_TRUE(e_mma < 3e-2);
    }
    std::printf("\n");
  }
}

// Matrix-core fused 3x3 conv2d (im2col staged in threadgroup memory -> matmul2d,
// no DRAM column matrix) == zero-padded CPU conv oracle. Pins the semantics the
// Krea-2 VAE relies on: the 3x3 halo (read from neighbouring tiles), border
// reads supplied as 0, the K=9*Cin tail (K need not be a multiple of BK), and
// Cout/pixel tail masking. Stride 1 (resnet/upsample, pad 1) and stride 2
// (encoder downsample, pad (0,1,0,1) -- im2col_hwc_3x3_s2's convention; this
// oracle read stride 2 as pad 1 and so pinned the kernel's wrong one) at
// VAE-shaped and deliberately non-tile-multiple sizes, down to the 4x4
// output the hardware op declines. f16 = exactly what the VAE binds.
TEST(conv2d_mma, matches_im2col) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[conv2d_mma] no matrix cores -- skip\n");
    return;
  }
  ComputeLibrary lib = mc->load_library("conv2d_mma");
  ComputeFunction f_s1 = lib.function("conv2d_mma_3x3_s1_f16");
  ComputeFunction f_s2 = lib.function("conv2d_mma_3x3_s2_f16");
  ASSERT_TRUE(f_s1.valid() && f_s2.valid());

  auto test = [&](int H, int W, int Cin, int Cout, int stride,
                  const char* tag) {
    const int OH = (stride == 2) ? H / 2 : H;
    const int OW = (stride == 2) ? W / 2 : W;
    std::mt19937 rng(77u + (unsigned)(H + Cin * 7 + Cout * 13 + stride));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<_Float16> in((std::size_t)H * W * Cin);
    std::vector<_Float16> wt((std::size_t)3 * 3 * Cin * Cout);
    for (auto& v : in) { v = (_Float16)(d(rng) * 0.5f); }
    for (auto& v : wt) { v = (_Float16)(d(rng) * 0.3f); }
    // CPU oracle: NHWC in, HWCO weights, pad 1 (zero), given stride.
    std::vector<float> ref((std::size_t)OH * OW * Cout, 0.0f);
    for (int oy = 0; oy < OH; ++oy) {
      for (int ox = 0; ox < OW; ++ox) {
        for (int oc = 0; oc < Cout; ++oc) {
          float acc = 0.0f;
          for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
              const int iy = stride == 2 ? 2 * oy + ky : oy + ky - 1;
              const int ix = stride == 2 ? 2 * ox + kx : ox + kx - 1;
              if (iy < 0 || iy >= H || ix < 0 || ix >= W) { continue; }
              for (int ci = 0; ci < Cin; ++ci) {
                const float a = (float)in[((std::size_t)iy * W + ix) * Cin + ci];
                // weights are [Cout, 9*Cin] with (ky,kx,ci)-flattened columns
                // (the same layout the VAE im2col path uses).
                const float w = (float)wt[(std::size_t)oc * (9 * Cin) +
                                          ((ky * 3 + kx) * Cin + ci)];
                acc += a * w;
              }
            }
          }
          ref[((std::size_t)oy * OW + ox) * Cout + oc] = acc;
        }
      }
    }
    SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
    SharedBuffer wb = mc->make_shared_buffer(wt.size() * 2);
    SharedBuffer ob = mc->make_shared_buffer((std::size_t)OH * OW * Cout * 2);
    std::memcpy(ib.contents(), in.data(), in.size() * 2);
    std::memcpy(wb.contents(), wt.data(), wt.size() * 2);
    std::memset(ob.contents(), 0, (std::size_t)OH * OW * Cout * 2);
    const int BM = 64, BN = 64, SG = 4, Mpix = OH * OW;
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(stride == 2 ? f_s2 : f_s1);
        enc.set_buffer(0, ib); enc.set_buffer(1, wb); enc.set_buffer(2, ob);
        enc.set_constant(3, H); enc.set_constant(4, W); enc.set_constant(5, Cin);
        enc.set_constant(6, Cout); enc.set_constant(7, OH); enc.set_constant(8, OW);
        enc.dispatch({(unsigned)(((Cout + BN - 1) / BN) * SG * 32),
                      (unsigned)((Mpix + BM - 1) / BM), 1},
                     {(unsigned)(SG * 32), 1, 1});
      }
      st.commit().wait();
    }
    std::vector<float> got((std::size_t)OH * OW * Cout);
    const auto* op = static_cast<const _Float16*>(ob.contents());
    for (std::size_t i = 0; i < got.size(); ++i) { got[i] = (float)op[i]; }
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      const double e = (double)got[i] - (double)ref[i];
      num += e * e; den += (double)ref[i] * (double)ref[i];
    }
    const double r = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    std::printf("[conv2d_mma] %-10s H=%d W=%d Cin=%d Cout=%d s%d  rel-L2=%.4e\n",
                tag, H, W, Cin, Cout, stride, r);
    EXPECT_TRUE(r < 3e-2);
  };
  test(16, 16, 32, 48, 1, "s1-basic");
  test(32, 32, 96, 96, 1, "s1-vae");
  test(16, 16, 64, 64, 2, "s2-down");
  test(8, 8, 768, 768, 2, "s2-4x4");       // a 64^2 picture's last downsample
  test(20, 12, 96, 96, 2, "s2-odd");       // outputs not multiples of 8
  test(12, 20, 48, 80, 1, "s1-oddtail");   // non-tile-multiple spatial+channels
  test(16, 16, 3, 96, 1, "s1-cin3");       // K=27 < BK (VAE conv_in / conv_out)
  test(16, 16, 16, 64, 1, "s1-cin16");     // K=144 (VAE z-channel convs)
}

// MPP convolution2d HARDWARE-OP semantics probe (see conv2d_hw_probe_f16 in
// conv2d_mma.metal). The op reads the FULL device NHWC activation itself --
// the gather-free mode -- but its descriptor has no padding field and the
// set_offsets coordinate space is undocumented (the original bring-up only
// verified single-tile). This probe runs a fixed 16x16xCIN -> 16x16xCOUT
// 3x3/s1/pad1 conv as FOUR 8x8 destination tiles and sweeps the offset
// interpretation at runtime; per-mode + per-tile rel-L2 against a CPU oracle
// shows which (if any) interpretation the driver implements, and whether the
// border halo is zero-filled by the hardware. Informational: prints a verdict
// per mode; only fails if the kernel/dispatch itself is broken.
TEST(conv2d_mma, hw_op_offset_probe) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[conv2d_hw] no matrix cores -- skip\n");
    return;
  }
  ComputeLibrary lib = mc->load_library("conv2d_mma");
  ComputeFunction fn = lib.function("conv2d_hw_probe_f16");
  if (!fn.valid()) {
    std::printf("[conv2d_hw] probe kernel invalid (driver rejected the "
                "convolution2d op) -- that IS the probe result.\n");
    return;
  }

  // Geometry must match the kernel's compile-time constants.
  const int HW = 16, TH = 8, TW = 8, CIN = 32, COUT = 64, SG = 4;
  std::mt19937 rng(4321);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<_Float16> in((std::size_t)HW * HW * CIN);
  std::vector<_Float16> wt((std::size_t)3 * 3 * CIN * COUT);   // HWIO
  for (auto& v : in) { v = (_Float16)(d(rng) * 0.5f); }
  for (auto& v : wt) { v = (_Float16)(d(rng) * 0.3f); }
  // CPU oracle: NHWC activation, HWIO weights (oc fastest), pad-1 zero.
  std::vector<float> ref((std::size_t)HW * HW * COUT, 0.0f);
  for (int oy = 0; oy < HW; ++oy) {
    for (int ox = 0; ox < HW; ++ox) {
      for (int oc = 0; oc < COUT; ++oc) {
        float acc = 0.0f;
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            const int iy = oy + ky - 1, ix = ox + kx - 1;
            if (iy < 0 || iy >= HW || ix < 0 || ix >= HW) { continue; }
            for (int ci = 0; ci < CIN; ++ci) {
              acc += (float)in[((std::size_t)iy * HW + ix) * CIN + ci] *
                     (float)wt[(((std::size_t)ky * 3 + kx) * CIN + ci) * COUT +
                               oc];
            }
          }
        }
        ref[((std::size_t)oy * HW + ox) * COUT + oc] = acc;
      }
    }
  }
  SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
  SharedBuffer wb = mc->make_shared_buffer(wt.size() * 2);
  SharedBuffer ob = mc->make_shared_buffer(ref.size() * 2);
  std::memcpy(ib.contents(), in.data(), in.size() * 2);
  std::memcpy(wb.contents(), wt.data(), wt.size() * 2);

  bool any_ok = false;
  for (int mode = 0; mode < 4; ++mode) {
    std::memset(ob.contents(), 0, ref.size() * 2);
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, ib); enc.set_buffer(1, wb); enc.set_buffer(2, ob);
        enc.set_constant(3, mode);
        enc.dispatch({(unsigned)((HW / TW) * SG * 32),
                      (unsigned)(HW / TH), 1}, {(unsigned)(SG * 32), 1, 1});
      }
      st.commit().wait();
    }
    const auto* op = static_cast<const _Float16*>(ob.contents());
    // Overall + per-tile rel-L2 (tile pattern diagnoses offset/halo).
    double tn[2][2] = {{0, 0}, {0, 0}}, td[2][2] = {{0, 0}, {0, 0}};
    for (int oy = 0; oy < HW; ++oy) {
      for (int ox = 0; ox < HW; ++ox) {
        for (int oc = 0; oc < COUT; ++oc) {
          const std::size_t i = ((std::size_t)oy * HW + ox) * COUT + oc;
          const double e = (double)op[i] - (double)ref[i];
          tn[oy / TH][ox / TW] += e * e;
          td[oy / TH][ox / TW] += (double)ref[i] * (double)ref[i];
        }
      }
    }
    double num = 0.0, den = 0.0;
    for (int ty = 0; ty < 2; ++ty) {
      for (int tx = 0; tx < 2; ++tx) { num += tn[ty][tx]; den += td[ty][tx]; }
    }
    const double r = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    const bool ok = r < 3e-2;
    any_ok = any_ok || ok;
    std::printf("[conv2d_hw] off_mode %d: rel-L2 %.4e %s | tiles "
                "%.1e %.1e / %.1e %.1e\n",
                mode, r, ok ? "VERIFIED" : "no",
                std::sqrt(tn[0][0] / (td[0][0] > 0 ? td[0][0] : 1)),
                std::sqrt(tn[0][1] / (td[0][1] > 0 ? td[0][1] : 1)),
                std::sqrt(tn[1][0] / (td[1][0] > 0 ? td[1][0] : 1)),
                std::sqrt(tn[1][1] / (td[1][1] > 0 ? td[1][1] : 1)));
  }
  std::printf("[conv2d_hw] verdict: %s\n",
              any_ok ? "an offset interpretation VERIFIES -- the hw op "
                       "auto-handles the halo; worth a perf pass"
                     : "no interpretation verified -- halo semantics still "
                       "not usable multi-tile");
}

// Probe 2 family, one variable per entry point (see conv2d_hw_probe2_impl):
//   2a: STALE 16x16 descriptor on a 32x32 image  -> do runtime tensor
//       extents govern the bounds/halo (one kernel serves every size)?
//   2b: matching descriptor, COUT=128 tiled as two 64-channel slices  ->
//       does weight/dest-slice channel tiling work (offset is spatial-only)?
//   2c: matching 32x32 descriptor, single channel tile -> big-image sanity.
TEST(conv2d_mma, hw_op_extents_channel_tiling_probe) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma");
  const int TH = 8, TW = 8, CIN = 32, TC = 64, SG = 4;

  auto run_probe = [&](const char* fname, int HW, int COUT,
                       const char* what) {
    ComputeFunction fn = lib.function(fname);
    if (!fn.valid()) {
      std::printf("[conv2d_hw2] %s invalid -- skip\n", fname);
      return;
    }
    std::mt19937 rng(8765u + (unsigned)(HW + COUT));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<_Float16> in((std::size_t)HW * HW * CIN);
    std::vector<_Float16> wt((std::size_t)3 * 3 * CIN * COUT);   // HWIO
    for (auto& v : in) { v = (_Float16)(d(rng) * 0.5f); }
    for (auto& v : wt) { v = (_Float16)(d(rng) * 0.3f); }
    std::vector<float> ref((std::size_t)HW * HW * COUT, 0.0f);
    for (int oy = 0; oy < HW; ++oy) {
      for (int ox = 0; ox < HW; ++ox) {
        for (int oc = 0; oc < COUT; ++oc) {
          float acc = 0.0f;
          for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
              const int iy = oy + ky - 1, ix = ox + kx - 1;
              if (iy < 0 || iy >= HW || ix < 0 || ix >= HW) { continue; }
              for (int ci = 0; ci < CIN; ++ci) {
                acc += (float)in[((std::size_t)iy * HW + ix) * CIN + ci] *
                       (float)wt[(((std::size_t)ky * 3 + kx) * CIN + ci) *
                                 COUT + oc];
              }
            }
          }
          ref[((std::size_t)oy * HW + ox) * COUT + oc] = acc;
        }
      }
    }
    SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
    SharedBuffer wb = mc->make_shared_buffer(wt.size() * 2);
    SharedBuffer ob = mc->make_shared_buffer(ref.size() * 2);
    std::memcpy(ib.contents(), in.data(), in.size() * 2);
    std::memcpy(wb.contents(), wt.data(), wt.size() * 2);
    std::memset(ob.contents(), 0, ref.size() * 2);
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, ib); enc.set_buffer(1, wb); enc.set_buffer(2, ob);
        enc.dispatch({(unsigned)((HW / TW) * SG * 32), (unsigned)(HW / TH),
                      (unsigned)(COUT / TC)}, {(unsigned)(SG * 32), 1, 1});
      }
      st.commit().wait();
    }
    const auto* op = static_cast<const _Float16*>(ob.contents());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const double e = (double)op[i] - (double)ref[i];
      num += e * e; den += (double)ref[i] * (double)ref[i];
    }
    const double r = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    std::printf("[conv2d_hw2] %-3s (%dx%d, %d ch): rel-L2 %.4e %s\n", what,
                HW, HW, COUT, r, r < 3e-2 ? "VERIFIED" : "no");
  };
  run_probe("conv2d_hw_probe2a_f16", 32, 64, "2a");   // stale extents
  run_probe("conv2d_hw_probe2b_f16", 16, 128, "2b");  // channel slices
  run_probe("conv2d_hw_probe2c_f16", 32, 64, "2c");   // matching 32x32

  // 2d: 16x16-templated descriptor RUNTIME-PATCHED to the real 32x32 (and
  // 48x32 non-square) source dims via the detail __run entry. Verifying
  // means one compiled kernel serves every image size.
  auto run_probe2d = [&](int Wd, int Hd) {
    ComputeFunction fn = lib.function("conv2d_hw_probe2d_f16");
    if (!fn.valid()) {
      std::printf("[conv2d_hw2] probe2d invalid -- skip\n");
      return;
    }
    const int COUT = 64;
    std::mt19937 rng(999u + (unsigned)(Wd * 3 + Hd));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<_Float16> in((std::size_t)Hd * Wd * CIN);
    std::vector<_Float16> wt((std::size_t)3 * 3 * CIN * COUT);
    for (auto& v : in) { v = (_Float16)(d(rng) * 0.5f); }
    for (auto& v : wt) { v = (_Float16)(d(rng) * 0.3f); }
    std::vector<float> ref((std::size_t)Hd * Wd * COUT, 0.0f);
    for (int oy = 0; oy < Hd; ++oy) {
      for (int ox = 0; ox < Wd; ++ox) {
        for (int oc = 0; oc < COUT; ++oc) {
          float acc = 0.0f;
          for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
              const int iy = oy + ky - 1, ix = ox + kx - 1;
              if (iy < 0 || iy >= Hd || ix < 0 || ix >= Wd) { continue; }
              for (int ci = 0; ci < CIN; ++ci) {
                acc += (float)in[((std::size_t)iy * Wd + ix) * CIN + ci] *
                       (float)wt[(((std::size_t)ky * 3 + kx) * CIN + ci) *
                                 COUT + oc];
              }
            }
          }
          ref[((std::size_t)oy * Wd + ox) * COUT + oc] = acc;
        }
      }
    }
    SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
    SharedBuffer wb = mc->make_shared_buffer(wt.size() * 2);
    SharedBuffer ob = mc->make_shared_buffer(ref.size() * 2);
    std::memcpy(ib.contents(), in.data(), in.size() * 2);
    std::memcpy(wb.contents(), wt.data(), wt.size() * 2);
    std::memset(ob.contents(), 0, ref.size() * 2);
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, ib); enc.set_buffer(1, wb); enc.set_buffer(2, ob);
        enc.set_constant(3, Wd); enc.set_constant(4, Hd);
        enc.dispatch({(unsigned)((Wd / TW) * SG * 32), (unsigned)(Hd / TH),
                      1}, {(unsigned)(SG * 32), 1, 1});
      }
      st.commit().wait();
    }
    const auto* op = static_cast<const _Float16*>(ob.contents());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const double e = (double)op[i] - (double)ref[i];
      num += e * e; den += (double)ref[i] * (double)ref[i];
    }
    const double r = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    std::printf("[conv2d_hw2] 2d  (%dx%d via runtime-patched desc): rel-L2 "
                "%.4e %s\n", Wd, Hd, r,
                r < 3e-2 ? "VERIFIED (one kernel serves all sizes)" : "no");
  };
  run_probe2d(32, 32);
  run_probe2d(48, 32);
}

// Stride-2 hw conv offset/padding probe (conv2d_hw_3x3_s2_f16): the VAE
// encoder's downsample convention is iy = oy*2 + ky - 1 (symmetric pad-1,
// OH = H/2, matching im2col_hwc_3x3_s2). Sweep the offset interpretations
// against a CPU oracle; production passes the winning off_mode.
TEST(conv2d_mma, hw_op_s2_probe) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma");
  ComputeFunction fn = lib.function("conv2d_hw_3x3_s2_f16");
  if (!fn.valid()) {
    std::printf("[conv2d_s2] kernel invalid -- skip\n");
    return;
  }
  const int HW = 32, OHW = 16, TH = 8, TW = 8, CIN = 32, COUT = 64, SG = 4;
  std::mt19937 rng(654);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<_Float16> in((std::size_t)HW * HW * CIN);
  std::vector<_Float16> wt((std::size_t)3 * 3 * CIN * COUT);   // HWIO
  for (auto& v : in) { v = (_Float16)(d(rng) * 0.5f); }
  for (auto& v : wt) { v = (_Float16)(d(rng) * 0.3f); }
  // Two padding conventions: SYMMETRIC pad 1 (iy = 2*oy + ky - 1) and the
  // ASYMMETRIC (0,1,0,1) diffusers-Downsample2D convention the VAE
  // encoders use (iy = 2*oy + ky; im2col_hwc_3x3_s2).
  auto oracle = [&](bool asym) {
    std::vector<float> ref((std::size_t)OHW * OHW * COUT, 0.0f);
    for (int oy = 0; oy < OHW; ++oy) {
      for (int ox = 0; ox < OHW; ++ox) {
        for (int oc = 0; oc < COUT; ++oc) {
          float acc = 0.0f;
          for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
              const int iy = oy * 2 + ky - (asym ? 0 : 1);
              const int ix = ox * 2 + kx - (asym ? 0 : 1);
              if (iy < 0 || iy >= HW || ix < 0 || ix >= HW) { continue; }
              for (int ci = 0; ci < CIN; ++ci) {
                acc += (float)in[((std::size_t)iy * HW + ix) * CIN + ci] *
                       (float)wt[(((std::size_t)ky * 3 + kx) * CIN + ci) *
                                 COUT + oc];
              }
            }
          }
          ref[((std::size_t)oy * OHW + ox) * COUT + oc] = acc;
        }
      }
    }
    return ref;
  };
  const std::vector<float> ref_sym = oracle(false);
  const std::vector<float> ref_asym = oracle(true);
  SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
  SharedBuffer wb = mc->make_shared_buffer(wt.size() * 2);
  SharedBuffer ob = mc->make_shared_buffer(ref_sym.size() * 2);
  std::memcpy(ib.contents(), in.data(), in.size() * 2);
  std::memcpy(wb.contents(), wt.data(), wt.size() * 2);
  bool sym_ok = false, asym_ok = false;
  for (int mode = 0; mode < 4; ++mode) {
    std::memset(ob.contents(), 0, ref_sym.size() * 2);
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, ib); enc.set_buffer(1, wb); enc.set_buffer(2, ob);
        enc.set_constant(3, HW); enc.set_constant(4, HW);
        enc.set_constant(5, CIN); enc.set_constant(6, COUT);
        enc.set_constant(7, mode);
        enc.dispatch({(unsigned)((OHW / TW) * SG * 32),
                      (unsigned)(OHW / TH), (unsigned)(COUT / 64)},
                     {(unsigned)(SG * 32), 1, 1});
      }
      st.commit().wait();
    }
    const auto* op = static_cast<const _Float16*>(ob.contents());
    auto rel = [&](const std::vector<float>& ref) {
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        const double e = (double)op[i] - (double)ref[i];
        num += e * e; den += (double)ref[i] * (double)ref[i];
      }
      return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    };
    const double rs = rel(ref_sym), ra = rel(ref_asym);
    sym_ok = sym_ok || (rs < 3e-2);
    asym_ok = asym_ok || (ra < 3e-2);
    std::printf("[conv2d_s2] off_mode %d: sym rel-L2 %.4e %s | asym "
                "rel-L2 %.4e %s\n", mode, rs,
                rs < 3e-2 ? "VERIFIED" : "no", ra,
                ra < 3e-2 ? "VERIFIED" : "no");
  }
  EXPECT_TRUE(sym_ok && asym_ok);
}

// General MPP hw conv (conv2d_hw_3x3_s1_f16): oracle-verify at a small
// Cin=128 shape, then perf-A/B against the EXACT im2col + dense matmul2d
// sequence the VAEs ship (im2col_hwc_3x3_f16 -> dense_gemm_mma_t_n128, K =
// 9*Cin < 6144) at real decoder shapes. GFLOP/s + ratio printed; the hw op
// skips the [H*W, 9*Cin] im2col DRAM round-trip entirely.
TEST(conv2d_mma, hw_op_depthwise_probe) {
  // CAN THE HARDWARE CONV OP TAKE A DEPTHWISE CONV, AND DOES IT PAY?
  //
  // MPP's convolution2d static_asserts groups == 1, so a depthwise conv --
  // one KxK kernel per channel, no cross-channel term -- has exactly one
  // mapping onto it: dense over a block of B channels with a weight that is
  // ZERO off the diagonal. That multiplies the arithmetic by B and lands it
  // on the matrix units, so it is a trade, and the point of this probe is to
  // price it rather than argue it.
  //
  // Driven by VDN-H3's short conv, which is 5x5 depthwise over 7168 channels
  // per latent frame and the largest remaining fp32 stage of its linear
  // branch (36.8% of it on an M5). Its activation is a strided window inside
  // a fused per-head projection, so the probe also settles the semantics
  // that would let it run WITHOUT a repack: whether the descriptor's Cin may
  // be smaller than the activation tensor's innermost extent, the way Cout
  // already may (hw_op_extents_channel_tiling_probe).
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma_bf16");
  if (!lib.valid()) { return; }

  auto bf = [](float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  };
  auto un_bf = [](std::uint16_t b) {
    const std::uint32_t u = (std::uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };

  // The real grid: 960x544 / 120 frames is 17x30 patches. Deliberately NOT a
  // multiple of the 8x8 dest tile -- the ragged tiles are where a halo or an
  // offset convention goes wrong, and the shipped VAE shapes never show it.
  const int gw = 30, gh = 17, KS = 5, pad = KS / 2;
  const int frames = 3;

  // `ok` is whether the block size is EXPECTED to be right, and B = 8 is
  // in this list precisely because it is not.
  //
  // THE MATRIX UNIT HAS A FLOOR AT 16 CHANNELS AND DOES NOT SAY SO. An
  // 8-wide block does not fail to build, does not fault and does not
  // return zeros -- it returns a plausible field that is 64% wrong. It
  // was found here only because the perf sweep wanted a smaller block
  // (the waste is exactly B, so 8 would halve it) and the correctness
  // list was widened to match the perf list. Anyone reaching for a
  // narrower block to cut the waste will reach for exactly this, so it
  // stays in the list, asserted to be wrong.
  struct Case { const char* name; int B; int TW; int TH; bool ok; };
  const Case cases[] = {
      {"conv2d_hw_dw5_b32h8_f16", 32, 32, 8, true},
      {"conv2d_hw_dw5_b16h4_f16", 16, 32, 4, true},
      {"conv2d_hw_dw5_b16h6_f16", 16, 32, 6, true},
      {"conv2d_hw_dw5_b16h8_f16", 16, 32, 8, true},
      {"conv2d_hw_dw5_b16h9_f16", 16, 32, 9, true},
      {"conv2d_hw_dw5_b8h8_f16",   8, 32, 8, false},
  };

  bool any = false;
  for (const Case& cs : cases) {
    ComputeFunction fn = lib.function(cs.name);
    if (!fn.valid()) {
      std::printf("[conv2d_dw] %s did not validate -- skip\n", cs.name);
      continue;
    }
    const int C = cs.B * 4;                 // four blocks, so nblk > 1
    const int nblk = C / cs.B;
    const std::size_t npix = (std::size_t)frames * gh * gw;
    std::mt19937 rng(4242u + (unsigned)cs.B);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    std::vector<std::uint16_t> in(npix * (std::size_t)C);
    std::vector<float> w((std::size_t)C * KS * KS);
    for (auto& v : in) { v = bf(d(rng) * 0.5f); }
    for (auto& v : w) { v = (float)bf(d(rng) * 0.4f) * 0.0f; }
    for (auto& v : w) { v = un_bf(bf(d(rng) * 0.4f)); }

    // The block-diagonal weight, hwio: [nblk][ky][kx][i][o], o innermost.
    std::vector<std::uint16_t> wb(
        (std::size_t)nblk * KS * KS * cs.B * cs.B, 0);
    for (int cb = 0; cb < nblk; ++cb) {
      for (int ky = 0; ky < KS; ++ky) {
        for (int kx = 0; kx < KS; ++kx) {
          for (int i = 0; i < cs.B; ++i) {
            const std::size_t o =
                ((((std::size_t)cb * KS + ky) * KS + kx) * cs.B + i) * cs.B
                + i;
            wb[o] = bf(w[((std::size_t)(cb * cs.B + i) * KS + ky) * KS + kx]);
          }
        }
      }
    }

    // The depthwise reference, in the layout the kernel writes.
    std::vector<float> ref(npix * (std::size_t)C, 0.0f);
    for (int f = 0; f < frames; ++f) {
      for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
          for (int c = 0; c < C; ++c) {
            float acc = 0.0f;
            for (int ky = 0; ky < KS; ++ky) {
              const int sy = y + ky - pad;
              if (sy < 0 || sy >= gh) { continue; }
              for (int kx = 0; kx < KS; ++kx) {
                const int sx = x + kx - pad;
                if (sx < 0 || sx >= gw) { continue; }
                acc += un_bf(in[((((std::size_t)f * gh + sy) * gw) + sx)
                                * (std::size_t)C + c])
                       * w[((std::size_t)c * KS + ky) * KS + kx];
              }
            }
            ref[((((std::size_t)f * gh + y) * gw) + x) * (std::size_t)C + c] =
                acc;
          }
        }
      }
    }

    SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
    SharedBuffer wbb = mc->make_shared_buffer(wb.size() * 2);
    SharedBuffer ob = mc->make_shared_buffer(ref.size() * 2);
    if (ib.empty() || wbb.empty() || ob.empty()) { continue; }
    std::memcpy(ib.contents(), in.data(), in.size() * 2);
    std::memcpy(wbb.contents(), wb.data(), wb.size() * 2);
    std::memset(ob.contents(), 0, ob.byte_size());

    const int SG = 4;
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, ib); enc.set_buffer(1, wbb); enc.set_buffer(2, ob);
        enc.set_constant(3, gw);
        enc.set_constant(4, gh);
        enc.set_constant(5, C);            // a_pitch
        enc.set_constant(6, C);            // d_pitch
        enc.set_constant(7, nblk);
        enc.set_constant(8, frames);
        enc.set_constant(9, C);            // hd  (tight: one group)
        enc.set_constant(10, C);           // hst
        enc.dispatch({(unsigned)(((gw + cs.TW - 1) / cs.TW) * SG * 32),
                      (unsigned)((gh + cs.TH - 1) / cs.TH),
                      (unsigned)(frames * nblk)},
                     {(unsigned)(SG * 32), 1, 1});
      }
      std::string e;
      if (!st.commit().wait_ok(&e)) {
        std::printf("[conv2d_dw] %s: %s\n", cs.name, e.c_str());
        continue;
      }
    }
    const auto* o = static_cast<const std::uint16_t*>(ob.contents());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const double e = (double)un_bf(o[i]) - (double)ref[i];
      num += e * e;
      den += (double)ref[i] * (double)ref[i];
    }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    const bool good = rel >= 0.0 && rel < 3e-2;
    std::printf("[conv2d_dw] %-24s B=%2d tile %dx%d: rel-L2 %.4e %s%s\n",
                cs.name, cs.B, cs.TW, cs.TH, rel,
                good ? "VERIFIED" : "WRONG",
                cs.ok ? "" : "  (expected -- below the 16-channel floor)");
    EXPECT_TRUE(good == cs.ok);
    any = true;
  }
  EXPECT_TRUE(any);
}

TEST(conv2d_mma, hw_op_depthwise_perf) {
  // THE PRICE, at the shape that asked the question: VDN-H3's short conv,
  // 5x5 depthwise over 7168 channels on a 17x30 patch grid, 37 latent
  // frames. Two arms on the same buffers, INTERLEAVED:
  //
  //   vdn_conv_spatial_f32   the shipping kernel -- one thread per channel
  //                          per run of 8 x, 25 taps, no matrix units
  //   conv2d_hw_dw5_*        the same conv as a block-diagonal DENSE conv
  //                          on the MPP hardware op, B channels at a time
  //
  // The hardware arm does B times the arithmetic for the same answer, so
  // this is the whole of the trade: useful FLOPs are the depthwise count in
  // both columns, and the hardware column's ISSUED FLOPs are B times that.
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma_bf16");
  ComputeLibrary vdn = mc->load_library("vdn_branch");
  ComputeFunction base = vdn.function("vdn_conv_spatial_f32");
  if (!lib.valid() || !base.valid()) { return; }

  auto envi = [](const char* k, int d) {
    const char* v = std::getenv(k);
    return (v != nullptr && *v != '\0') ? std::atoi(v) : d;
  };
  const int gw = envi("VPIPE_DW_GW", 30), gh = envi("VPIPE_DW_GH", 17);
  const int KS = 5, C = 7168, SG = 4;
  // How many frames ONE dispatch carries. The branch tiles, so its
  // dispatches are a handful of frames rather than the whole clip --
  // and whether that costs anything per frame is the question this
  // knob asks.
  const int frames = envi("VPIPE_DW_FRAMES", 37);
  const int iters = envi("VPIPE_DW_ITERS", 1);
  // The REAL source: VDN's short conv reads the transformer's fused
  // [rows, 3*inner] projection where it lies, grouped per head. So the row
  // stride is 3*C and a head is 128 channels 384 apart -- which is the
  // read pattern that matters, and the one a tight-buffer probe flatters.
  // VPIPE_DW_TIGHT measures the flattering case for comparison.
  const bool tight = std::getenv("VPIPE_DW_TIGHT") != nullptr;
  const int rst = tight ? C : 3 * C;
  const int hd = 128, hst = tight ? hd : 3 * hd;
  const std::size_t npix = (std::size_t)frames * gh * gw;
  SharedBuffer ib = mc->make_shared_buffer(npix * (std::size_t)rst * 2);
  SharedBuffer ob = mc->make_shared_buffer(npix * (std::size_t)C * 2);
  SharedBuffer wf = mc->make_shared_buffer((std::size_t)C * KS * KS * 4);
  if (ib.empty() || ob.empty() || wf.empty()) {
    std::printf("[conv2d_dw] cannot allocate %.2f GB\n",
                (double)(2 * npix * C * 2) / 1073741824.0);
    return;
  }
  std::memset(ib.contents(), 0x3c, ib.byte_size());
  std::memset(ob.contents(), 0, ob.byte_size());
  {
    auto* w = static_cast<float*>(wf.contents());
    for (std::size_t i = 0; i < (std::size_t)C * KS * KS; ++i) {
      w[i] = 0.01f;
    }
  }
  const double useful_gflop =
      2.0 * (double)npix * (double)C * (double)(KS * KS) / 1e9;

  // Arm A: the shipping kernel.
  const int run = 8, nxg = (gw + run - 1) / run;
  auto run_base = [&]() {
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder enc = st.begin_compute();
      enc.set_function(base);
      enc.set_buffer(0, ib); enc.set_buffer(1, wf); enc.set_buffer(2, ob);
      enc.set_constant(3, frames);
      enc.set_constant(4, gh);
      enc.set_constant(5, gw);
      enc.set_constant(6, C);
      enc.set_constant(7, KS);
      enc.set_buffer(8, ib);
      enc.set_constant(9, 1);            // bf16 in
      enc.set_constant(10, rst);         // row stride
      enc.set_constant(11, hst);         // head stride
      enc.set_constant(12, hd);          // head_dim
      enc.set_constant(13, run);
      enc.set_buffer(14, ob);
      enc.set_constant(15, 1);           // bf16 out
      enc.dispatch({(unsigned)((std::size_t)frames * gh * nxg * C), 1, 1},
                   {256, 1, 1});
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::string e;
    if (!st.commit().wait_ok(&e)) { return -1.0; }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
  };

  struct Case { const char* name; int B; int TW; int TH; };
  const Case cases[] = {
      {"conv2d_hw_dw5_b32h8_f16", 32, 32, 8},
      {"conv2d_hw_dw5_b16h4_f16", 16, 32, 4},
      {"conv2d_hw_dw5_b16h6_f16", 16, 32, 6},
      {"conv2d_hw_dw5_b16h8_f16", 16, 32, 8},
      {"conv2d_hw_dw5_b16h9_f16", 16, 32, 9},
  };
  // The block-diagonal weight, built once per B -- in production it is
  // rebuilt per layer into scratch, which is 5.7 MB of writes and free.
  auto make_wb = [&](int B) {
    const int nblk = C / B;
    SharedBuffer b = mc->make_shared_buffer(
        (std::size_t)nblk * KS * KS * B * B * 2);
    if (!b.empty()) { std::memset(b.contents(), 0, b.byte_size()); }
    return b;
  };

  double best_base = -1.0;
  std::vector<double> best(sizeof(cases) / sizeof(cases[0]), -1.0);
  std::vector<SharedBuffer> wbs;
  for (const Case& cs : cases) { wbs.push_back(make_wb(cs.B)); }

  for (int r = 0; r < 4; ++r) {
    const double a = run_base();
    if (r > 0 && a >= 0.0 && (best_base < 0.0 || a < best_base)) {
      best_base = a;
    }
    for (std::size_t i = 0; i < wbs.size(); ++i) {
      const Case& cs = cases[i];
      ComputeFunction fn = lib.function(cs.name);
      if (!fn.valid() || wbs[i].empty()) { continue; }
      const int nblk = C / cs.B;
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        // ITERS dispatches in ONE command buffer, so a small one is not
        // measured against the submit round trip -- which is the whole
        // question when the caller tiles and its dispatches are a
        // handful of frames each. The reported ms is per iteration.
        for (int it = 0; it < iters; ++it) {
        enc.set_function(fn);
        enc.set_buffer(0, ib); enc.set_buffer(1, wbs[i]);
        enc.set_buffer(2, ob);
        enc.set_constant(3, gw);
        enc.set_constant(4, gh);
        enc.set_constant(5, rst);
        enc.set_constant(6, C);
        enc.set_constant(7, nblk);
        enc.set_constant(8, frames);
        enc.set_constant(9, hd);
        enc.set_constant(10, hst);
        enc.dispatch({(unsigned)(((gw + cs.TW - 1) / cs.TW) * SG * 32),
                      (unsigned)((gh + cs.TH - 1) / cs.TH),
                      (unsigned)(frames * nblk)},
                     {(unsigned)(SG * 32), 1, 1});
        }
      }
      const auto t0 = std::chrono::steady_clock::now();
      std::string e;
      if (!st.commit().wait_ok(&e)) { continue; }
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count() / (double)iters;
      if (r > 0 && (best[i] < 0.0 || ms < best[i])) { best[i] = ms; }
    }
  }

  std::printf("[conv2d_dw] %d frames of %dx%d, %d channels, 5x5 depthwise "
              "(%.2f useful GFLOP), source %s\n", frames, gh, gw, C,
              useful_gflop, tight ? "TIGHT" : "fused (row 3C, head 3*128)");
  std::printf("[conv2d_dw]   %-24s %8.2f ms  %6.3f TFLOP/s useful\n",
              "vdn_conv_spatial_f32", best_base,
              best_base > 0.0 ? useful_gflop / (best_base / 1000.0) / 1000.0
                              : 0.0);
  for (std::size_t i = 0; i < wbs.size(); ++i) {
    if (best[i] < 0.0) { continue; }
    std::printf("[conv2d_dw]   %-24s %8.2f ms  %6.3f useful / %6.2f issued "
                "TFLOP/s   %.2fx\n", cases[i].name, best[i],
                useful_gflop / (best[i] / 1000.0) / 1000.0,
                (double)cases[i].B * useful_gflop / (best[i] / 1000.0)
                    / 1000.0,
                best_base > 0.0 ? best_base / best[i] : 0.0);
  }
  EXPECT_TRUE(best_base > 0.0);
}

TEST(conv2d_mma, hw_op_vs_im2col_perf) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma");
  ComputeLibrary lib_mma = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise");
  ComputeFunction f_hw = lib.function("conv2d_hw_3x3_s1_f16");
  ComputeFunction f_mma = lib_mma.function("dense_gemm_mma_t_n128_f16");
  ComputeFunction f_i2c = lib_elt.function("im2col_hwc_3x3_f16");
  if (!f_hw.valid() || !f_mma.valid() || !f_i2c.valid()) {
    std::printf("[conv2d_hwp] kernels unavailable -- skip\n");
    return;
  }
  const int TH = 8, TW = 8, TC = 64, SG = 4;

  auto run_shape = [&](int HW, int C, bool check, int iters) {
    const int Cin = C, Cout = C;
    std::mt19937 rng(31u + (unsigned)(HW + C));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<_Float16> in((std::size_t)HW * HW * Cin);
    std::vector<_Float16> w_hwio((std::size_t)3 * 3 * Cin * Cout);
    for (auto& v : in) { v = (_Float16)(d(rng) * 0.5f); }
    for (auto& v : w_hwio) { v = (_Float16)(d(rng) * 0.3f); }
    // The im2col path's weight layout: [Cout, 9*Cin], (ky,kx,ci) columns.
    std::vector<_Float16> w_o9i((std::size_t)Cout * 9 * Cin);
    for (int ky = 0; ky < 3; ++ky) {
      for (int kx = 0; kx < 3; ++kx) {
        for (int ci = 0; ci < Cin; ++ci) {
          for (int oc = 0; oc < Cout; ++oc) {
            w_o9i[(std::size_t)oc * (9 * Cin) + ((ky * 3 + kx) * Cin + ci)] =
                w_hwio[(((std::size_t)ky * 3 + kx) * Cin + ci) * Cout + oc];
          }
        }
      }
    }
    SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
    SharedBuffer wb = mc->make_shared_buffer(w_hwio.size() * 2);
    SharedBuffer wb2 = mc->make_shared_buffer(w_o9i.size() * 2);
    SharedBuffer ob = mc->make_shared_buffer((std::size_t)HW * HW * Cout * 2);
    SharedBuffer ob2 = mc->make_shared_buffer((std::size_t)HW * HW * Cout * 2);
    SharedBuffer col =
        mc->make_shared_buffer((std::size_t)HW * HW * 9 * Cin * 2);
    if (col.empty() || ib.empty()) {
      std::printf("[conv2d_hwp] alloc failed at %d/%d -- skip\n", HW, C);
      return;
    }
    std::memcpy(ib.contents(), in.data(), in.size() * 2);
    std::memcpy(wb.contents(), w_hwio.data(), w_hwio.size() * 2);
    std::memcpy(wb2.contents(), w_o9i.data(), w_o9i.size() * 2);

    auto launch_hw = [&](int n) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          enc.set_function(f_hw);
          enc.set_buffer(0, ib); enc.set_buffer(1, wb); enc.set_buffer(2, ob);
          enc.set_constant(3, HW); enc.set_constant(4, HW);
          enc.set_constant(5, Cin); enc.set_constant(6, Cout);
          enc.dispatch({(unsigned)((HW / TW) * SG * 32),
                        (unsigned)(HW / TH), (unsigned)(Cout / TC)},
                       {(unsigned)(SG * 32), 1, 1});
        }
      }
      st.commit().wait();
    };
    auto launch_i2c = [&](int n) {
      const int M = HW * HW, K = 9 * Cin;
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          enc.set_function(f_i2c);
          enc.set_buffer(0, ib); enc.set_buffer(1, col);
          enc.set_constant(2, HW); enc.set_constant(3, HW);
          enc.set_constant(4, Cin);
          enc.dispatch({(unsigned)((std::size_t)M * K), 1, 1}, {256, 1, 1});
          enc.set_function(f_mma);
          enc.set_buffer(0, col); enc.set_buffer(1, wb2);
          enc.set_buffer(2, wb2); enc.set_buffer(3, ob2);
          enc.set_constant(4, K); enc.set_constant(5, Cout);
          enc.set_constant(6, M); enc.set_constant(7, 0);
          enc.dispatch({(unsigned)(((Cout + 127) / 128) * 256),
                        (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        }
      }
      st.commit().wait();
    };

    launch_hw(1); launch_i2c(1);              // warm + outputs for check
    if (check) {
      const auto* a = static_cast<const _Float16*>(ob.contents());
      const auto* b = static_cast<const _Float16*>(ob2.contents());
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < (std::size_t)HW * HW * Cout; ++i) {
        const double e = (double)a[i] - (double)b[i];
        num += e * e; den += (double)b[i] * (double)b[i];
      }
      const double r = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
      std::printf("[conv2d_hwp] hw-vs-im2col rel-L2 %.4e (%dx%dx%d) %s\n",
                  r, HW, HW, C, r < 3e-2 ? "MATCH" : "(BAD)");
      EXPECT_TRUE(r < 3e-2);
    }
    const double gflop =
        2.0 * HW * HW * Cout * 9.0 * Cin * iters / 1e9;
    auto t0 = std::chrono::steady_clock::now();
    launch_hw(iters);
    auto t1 = std::chrono::steady_clock::now();
    launch_i2c(iters);
    auto t2 = std::chrono::steady_clock::now();
    const double s_hw = secs_(t0, t1), s_i2c = secs_(t1, t2);
    std::printf("[conv2d_hwp] %4dx%-4d C=%3d: hw %.0f GF/s (%.2f ms)  "
                "im2col+mma %.0f GF/s (%.2f ms)  ratio %.2fx\n",
                HW, HW, C, gflop / s_hw, s_hw * 1e3 / iters,
                gflop / s_i2c, s_i2c * 1e3 / iters, s_i2c / s_hw);
  };
  run_shape(64, 128, /*check=*/true, 20);     // correctness + small
  run_shape(256, 128, /*check=*/false, 10);   // VAE mid decoder
  run_shape(512, 128, /*check=*/false, 5);    // VAE late decoder
  run_shape(256, 256, /*check=*/false, 10);   // deeper-channel mid
}

// Bias-fused hw conv (mode::multiply_accumulate onto a bias-preloaded
// cooperative destination): out = bias + conv with NO separate bias pass.
// Oracle-verifies the fold at 64x64x128, then perf-A/Bs (hw conv +
// bias_add_rows) vs the fused kernel at the 512x512x128 late-decoder shape
// -- the win is bias_add_rows' full-image y read+write, plus one dispatch.
TEST(conv2d_mma, hw_op_bias_fused) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma");
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise");
  ComputeFunction f_hw = lib.function("conv2d_hw_3x3_s1_f16");
  ComputeFunction f_hwb = lib.function("conv2d_hw_3x3_s1_bias_f16");
  ComputeFunction f_bias = lib_elt.function("bias_add_rows_f16");
  if (!f_hw.valid() || !f_hwb.valid() || !f_bias.valid()) {
    std::printf("[conv2d_hwb] kernels unavailable -- skip\n");
    return;
  }
  const int TH = 8, TW = 8, TC = 64, SG = 4;

  auto run_shape = [&](int HW, int C, bool check, int iters) {
    const int Cin = C, Cout = C;
    std::mt19937 rng(55u + (unsigned)(HW + C));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<_Float16> in((std::size_t)HW * HW * Cin);
    std::vector<_Float16> w_hwio((std::size_t)3 * 3 * Cin * Cout);
    std::vector<_Float16> bs((std::size_t)Cout);
    for (auto& v : in) { v = (_Float16)(d(rng) * 0.5f); }
    for (auto& v : w_hwio) { v = (_Float16)(d(rng) * 0.3f); }
    for (auto& v : bs) { v = (_Float16)d(rng); }
    SharedBuffer ib = mc->make_shared_buffer(in.size() * 2);
    SharedBuffer wb = mc->make_shared_buffer(w_hwio.size() * 2);
    SharedBuffer bb = mc->make_shared_buffer(bs.size() * 2);
    SharedBuffer ob = mc->make_shared_buffer((std::size_t)HW * HW * Cout * 2);
    SharedBuffer ob2 = mc->make_shared_buffer((std::size_t)HW * HW * Cout * 2);
    std::memcpy(ib.contents(), in.data(), in.size() * 2);
    std::memcpy(wb.contents(), w_hwio.data(), w_hwio.size() * 2);
    std::memcpy(bb.contents(), bs.data(), bs.size() * 2);

    auto conv_common = [&](ComputeEncoder& enc, ComputeFunction& fn,
                           SharedBuffer& dst, bool with_bias) {
      enc.set_function(fn);
      enc.set_buffer(0, ib); enc.set_buffer(1, wb); enc.set_buffer(2, dst);
      enc.set_constant(3, HW); enc.set_constant(4, HW);
      enc.set_constant(5, Cin); enc.set_constant(6, Cout);
      if (with_bias) { enc.set_buffer(7, bb); }
      enc.dispatch({(unsigned)((HW / TW) * SG * 32), (unsigned)(HW / TH),
                    (unsigned)(Cout / TC)}, {(unsigned)(SG * 32), 1, 1});
    };
    auto launch_unfused = [&](int n) {      // conv + separate bias pass
      const int M = HW * HW;
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          conv_common(enc, f_hw, ob2, false);
          enc.set_function(f_bias);
          enc.set_buffer(0, ob2); enc.set_buffer(1, bb);
          enc.set_constant(2, Cout); enc.set_constant(3, M * Cout);
          enc.dispatch({(unsigned)((std::size_t)M * Cout), 1, 1},
                       {256, 1, 1});
        }
      }
      st.commit().wait();
    };
    auto launch_fused = [&](int n) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) { conv_common(enc, f_hwb, ob, true); }
      }
      st.commit().wait();
    };

    launch_fused(1); launch_unfused(1);
    if (check) {
      const auto* a = static_cast<const _Float16*>(ob.contents());
      const auto* b = static_cast<const _Float16*>(ob2.contents());
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < (std::size_t)HW * HW * Cout; ++i) {
        const double e = (double)a[i] - (double)b[i];
        num += e * e; den += (double)b[i] * (double)b[i];
      }
      const double r = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
      std::printf("[conv2d_hwb] fused-vs-(conv+bias_add) rel-L2 %.4e "
                  "(%dx%dx%d) %s\n", r, HW, HW, C,
                  r < 3e-2 ? "MATCH" : "(BAD)");
      EXPECT_TRUE(r < 3e-2);
    }
    auto t0 = std::chrono::steady_clock::now();
    launch_fused(iters);
    auto t1 = std::chrono::steady_clock::now();
    launch_unfused(iters);
    auto t2 = std::chrono::steady_clock::now();
    const double ms_f = secs_(t0, t1) * 1e3 / iters;
    const double ms_u = secs_(t1, t2) * 1e3 / iters;
    std::printf("[conv2d_hwb] %4dx%-4d C=%3d: fused %.2f ms  conv+bias "
                "%.2f ms  ratio %.2fx\n", HW, HW, C, ms_f, ms_u, ms_u / ms_f);
  };
  run_shape(64, 128, /*check=*/true, 20);
  run_shape(512, 128, /*check=*/false, 5);
}

// int8 x int8 -> f16 hw conv (conv2d_hw_3x3_s1_i8f16) vs the f16 hw conv:
// does the M5 run integer conv above the f16 rate? (matmul2d measured i8 AT
// the f16 rate -- bandwidth halves but the matrix pipeline doesn't speed
// up.) Oracle uses small int values ([-2,1]) so the raw integer sums stay
// exactly representable in the f16 destination.
TEST(conv2d_mma, hw_op_i8_vs_f16_perf) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma");
  ComputeFunction f_hw = lib.function("conv2d_hw_3x3_s1_f16");
  ComputeFunction f_i8 = lib.function("conv2d_hw_3x3_s1_i8f16");
  if (!f_hw.valid() || !f_i8.valid()) {
    std::printf("[conv2d_i8] kernels unavailable -- skip\n");
    return;
  }
  const int TH = 8, TW = 8, TC = 64, SG = 4;

  auto run_shape = [&](int HW, int C, bool check, int iters) {
    const int Cin = C, Cout = C;
    std::mt19937 rng(91u + (unsigned)(HW + C));
    std::uniform_int_distribution<int> di(-2, 1);
    std::vector<std::int8_t> in8((std::size_t)HW * HW * Cin);
    std::vector<std::int8_t> w8((std::size_t)3 * 3 * Cin * Cout);   // HWIO
    for (auto& v : in8) { v = (std::int8_t)di(rng); }
    for (auto& v : w8) { v = (std::int8_t)di(rng); }
    // f16 mirrors of the same values (identical math, so the f16 kernel is
    // both the perf baseline AND a second oracle).
    std::vector<_Float16> inh(in8.size()), wh(w8.size());
    for (std::size_t i = 0; i < in8.size(); ++i) {
      inh[i] = (_Float16)in8[i];
    }
    for (std::size_t i = 0; i < w8.size(); ++i) { wh[i] = (_Float16)w8[i]; }
    SharedBuffer i8b = mc->make_shared_buffer(in8.size());
    SharedBuffer w8b = mc->make_shared_buffer(w8.size());
    SharedBuffer ihb = mc->make_shared_buffer(inh.size() * 2);
    SharedBuffer whb = mc->make_shared_buffer(wh.size() * 2);
    SharedBuffer ob = mc->make_shared_buffer((std::size_t)HW * HW * Cout * 2);
    SharedBuffer ob2 = mc->make_shared_buffer((std::size_t)HW * HW * Cout * 2);
    std::memcpy(i8b.contents(), in8.data(), in8.size());
    std::memcpy(w8b.contents(), w8.data(), w8.size());
    std::memcpy(ihb.contents(), inh.data(), inh.size() * 2);
    std::memcpy(whb.contents(), wh.data(), wh.size() * 2);

    auto launch = [&](ComputeFunction& fn, SharedBuffer& xin,
                      SharedBuffer& win, SharedBuffer& dst, int n) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          enc.set_function(fn);
          enc.set_buffer(0, xin); enc.set_buffer(1, win);
          enc.set_buffer(2, dst);
          enc.set_constant(3, HW); enc.set_constant(4, HW);
          enc.set_constant(5, Cin); enc.set_constant(6, Cout);
          enc.dispatch({(unsigned)((HW / TW) * SG * 32),
                        (unsigned)(HW / TH), (unsigned)(Cout / TC)},
                       {(unsigned)(SG * 32), 1, 1});
        }
      }
      st.commit().wait();
    };

    launch(f_i8, i8b, w8b, ob, 1);
    launch(f_hw, ihb, whb, ob2, 1);
    if (check) {
      const auto* a = static_cast<const _Float16*>(ob.contents());
      const auto* b = static_cast<const _Float16*>(ob2.contents());
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < (std::size_t)HW * HW * Cout; ++i) {
        const double e = (double)a[i] - (double)b[i];
        num += e * e; den += (double)b[i] * (double)b[i];
      }
      const double r = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
      std::printf("[conv2d_i8] i8-vs-f16 rel-L2 %.4e (%dx%dx%d) %s\n",
                  r, HW, HW, C, r < 1e-2 ? "MATCH" : "(BAD)");
      EXPECT_TRUE(r < 1e-2);
    }
    const double gflop = 2.0 * HW * HW * Cout * 9.0 * Cin * iters / 1e9;
    auto t0 = std::chrono::steady_clock::now();
    launch(f_i8, i8b, w8b, ob, iters);
    auto t1 = std::chrono::steady_clock::now();
    launch(f_hw, ihb, whb, ob2, iters);
    auto t2 = std::chrono::steady_clock::now();
    const double s_i8 = secs_(t0, t1), s_f16 = secs_(t1, t2);
    std::printf("[conv2d_i8] %4dx%-4d C=%3d: i8 %.0f GF/s (%.2f ms)  f16 "
                "%.0f GF/s (%.2f ms)  i8/f16 %.2fx\n",
                HW, HW, C, gflop / s_i8, s_i8 * 1e3 / iters,
                gflop / s_f16, s_f16 * 1e3 / iters, s_f16 / s_i8);
  };
  run_shape(64, 128, /*check=*/true, 20);
  run_shape(256, 128, /*check=*/false, 10);
  run_shape(512, 128, /*check=*/false, 5);
  run_shape(256, 256, /*check=*/false, 10);
}

// 1x1 hw conv, i8 vs f16 (conv2d_hw_1x1_s1_{i8f16,f16}) -- the pure-GEMM
// regime (M = H*W, N = Cout, K = Cin, small K): operand-bandwidth-leaning,
// so the i8 halving matters even if the matrix rate doesn't change. The
// equivalent dense_gemm_mma_t_n128_f16 GEMM (activation viewed as
// [H*W, Cin]) is timed alongside as the reference the VAE's 1x1 path
// dispatches today.
TEST(conv2d_mma, hw_op_1x1_i8_vs_f16_perf) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma");
  ComputeLibrary lib_mma = mc->load_library("dense_gemm_mma");
  ComputeFunction f_f16 = lib.function("conv2d_hw_1x1_s1_f16");
  ComputeFunction f_i8 = lib.function("conv2d_hw_1x1_s1_i8f16");
  ComputeFunction f_mma = lib_mma.function("dense_gemm_mma_t_n128_f16");
  if (!f_f16.valid() || !f_i8.valid() || !f_mma.valid()) {
    std::printf("[conv2d_1x1] kernels unavailable -- skip\n");
    return;
  }
  const int TH = 8, TW = 8, TC = 64, SG = 4;

  auto run_shape = [&](int HW, int C, bool check, int iters) {
    const int Cin = C, Cout = C;
    std::mt19937 rng(17u + (unsigned)(HW + C));
    std::uniform_int_distribution<int> di(-2, 1);
    std::vector<std::int8_t> in8((std::size_t)HW * HW * Cin);
    std::vector<std::int8_t> w8((std::size_t)Cin * Cout);   // HWIO, 1x1
    for (auto& v : in8) { v = (std::int8_t)di(rng); }
    for (auto& v : w8) { v = (std::int8_t)di(rng); }
    std::vector<_Float16> inh(in8.size()), wh(w8.size());
    for (std::size_t i = 0; i < in8.size(); ++i) {
      inh[i] = (_Float16)in8[i];
    }
    for (std::size_t i = 0; i < w8.size(); ++i) { wh[i] = (_Float16)w8[i]; }
    // The GEMM reference wants W[N,K] row-major = [Cout, Cin]; the HWIO
    // 1x1 weight is [Cin, Cout] (O fastest) -> transpose.
    std::vector<_Float16> wt_nk((std::size_t)Cout * Cin);
    for (int ci = 0; ci < Cin; ++ci) {
      for (int oc = 0; oc < Cout; ++oc) {
        wt_nk[(std::size_t)oc * Cin + ci] = wh[(std::size_t)ci * Cout + oc];
      }
    }
    SharedBuffer i8b = mc->make_shared_buffer(in8.size());
    SharedBuffer w8b = mc->make_shared_buffer(w8.size());
    SharedBuffer ihb = mc->make_shared_buffer(inh.size() * 2);
    SharedBuffer whb = mc->make_shared_buffer(wh.size() * 2);
    SharedBuffer wnk = mc->make_shared_buffer(wt_nk.size() * 2);
    SharedBuffer ob = mc->make_shared_buffer((std::size_t)HW * HW * Cout * 2);
    SharedBuffer ob2 = mc->make_shared_buffer((std::size_t)HW * HW * Cout * 2);
    std::memcpy(i8b.contents(), in8.data(), in8.size());
    std::memcpy(w8b.contents(), w8.data(), w8.size());
    std::memcpy(ihb.contents(), inh.data(), inh.size() * 2);
    std::memcpy(whb.contents(), wh.data(), wh.size() * 2);
    std::memcpy(wnk.contents(), wt_nk.data(), wt_nk.size() * 2);

    auto launch_conv = [&](ComputeFunction& fn, SharedBuffer& xin,
                           SharedBuffer& win, SharedBuffer& dst, int n) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          enc.set_function(fn);
          enc.set_buffer(0, xin); enc.set_buffer(1, win);
          enc.set_buffer(2, dst);
          enc.set_constant(3, HW); enc.set_constant(4, HW);
          enc.set_constant(5, Cin); enc.set_constant(6, Cout);
          enc.dispatch({(unsigned)((HW / TW) * SG * 32),
                        (unsigned)(HW / TH), (unsigned)(Cout / TC)},
                       {(unsigned)(SG * 32), 1, 1});
        }
      }
      st.commit().wait();
    };
    auto launch_gemm = [&](int n) {
      const int M = HW * HW;
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          enc.set_function(f_mma);
          enc.set_buffer(0, ihb); enc.set_buffer(1, wnk);
          enc.set_buffer(2, wnk); enc.set_buffer(3, ob2);
          enc.set_constant(4, Cin); enc.set_constant(5, Cout);
          enc.set_constant(6, M); enc.set_constant(7, 0);
          enc.dispatch({(unsigned)(((Cout + 127) / 128) * 256),
                        (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        }
      }
      st.commit().wait();
    };

    launch_conv(f_i8, i8b, w8b, ob, 1);
    launch_gemm(1);
    if (check) {
      const auto* a = static_cast<const _Float16*>(ob.contents());
      const auto* b = static_cast<const _Float16*>(ob2.contents());
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < (std::size_t)HW * HW * Cout; ++i) {
        const double e = (double)a[i] - (double)b[i];
        num += e * e; den += (double)b[i] * (double)b[i];
      }
      const double r = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
      std::printf("[conv2d_1x1] i8-conv-vs-f16-gemm rel-L2 %.4e (%dx%dx%d) "
                  "%s\n", r, HW, HW, C, r < 1e-2 ? "MATCH" : "(BAD)");
      EXPECT_TRUE(r < 1e-2);
    }
    const double gflop = 2.0 * HW * HW * Cout * (double)Cin * iters / 1e9;
    auto t0 = std::chrono::steady_clock::now();
    launch_conv(f_i8, i8b, w8b, ob, iters);
    auto t1 = std::chrono::steady_clock::now();
    launch_conv(f_f16, ihb, whb, ob, iters);
    auto t2 = std::chrono::steady_clock::now();
    launch_gemm(iters);
    auto t3 = std::chrono::steady_clock::now();
    const double s_i8 = secs_(t0, t1), s_f16 = secs_(t1, t2),
                 s_gemm = secs_(t2, t3);
    std::printf("[conv2d_1x1] %4dx%-4d C=%3d: i8 %.0f GF/s (%.2f ms)  f16 "
                "%.0f GF/s (%.2f ms)  gemm-f16 %.0f GF/s (%.2f ms)  i8/f16 "
                "%.2fx  i8/gemm %.2fx\n",
                HW, HW, C, gflop / s_i8, s_i8 * 1e3 / iters,
                gflop / s_f16, s_f16 * 1e3 / iters,
                gflop / s_gemm, s_gemm * 1e3 / iters,
                s_f16 / s_i8, s_gemm / s_i8);
  };
  run_shape(64, 128, /*check=*/true, 20);
  run_shape(256, 128, /*check=*/false, 20);
  run_shape(512, 128, /*check=*/false, 10);
  run_shape(256, 256, /*check=*/false, 20);
  run_shape(128, 512, /*check=*/false, 20);
}

// f16 -> i8 group-64 quantization kernels (quant_f16_i8_g64_{bfp,amax} in
// affine_dequant.metal): the activation-quant pass an int8 conv/GEMM
// pipeline needs. bfp = block-floating-point (block max EXPONENT +
// mantissa shift, power-of-2 scale, pure integer); amax = the float
// baseline (127/amax scale). Reports GB/s (traffic ~3 B/elem: 2 read + 1
// write + scales) and the dequantized rel-L2 of each scheme.
TEST(quant_kernels, f16_i8_g64_bench) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("affine_dequant");
  ComputeFunction f_bfp = lib.function("quant_f16_i8_g64_bfp");
  ComputeFunction f_amax = lib.function("quant_f16_i8_g64_amax");
  ASSERT_TRUE(f_bfp.valid() && f_amax.valid());

  const std::size_t N = 32u * 1024 * 1024;    // 64 MB f16 in, 32 MB i8 out
  std::mt19937 rng(2718);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<_Float16> x(N);
  for (auto& v : x) { v = (_Float16)nd(rng); }
  SharedBuffer xb = mc->make_shared_buffer(N * 2);
  SharedBuffer qb = mc->make_shared_buffer(N);
  SharedBuffer sb = mc->make_shared_buffer((N / 64) * 2);
  ASSERT_TRUE(!xb.empty() && !qb.empty() && !sb.empty());
  std::memcpy(xb.contents(), x.data(), N * 2);

  auto launch = [&](ComputeFunction& fn, int iters) {
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder enc = st.begin_compute();
      for (int i = 0; i < iters; ++i) {
        enc.set_function(fn);
        enc.set_buffer(0, xb); enc.set_buffer(1, qb); enc.set_buffer(2, sb);
        enc.set_constant(3, (int)N);
        enc.dispatch({(unsigned)(N / 2), 1, 1}, {128, 1, 1});
      }
    }
    st.commit().wait();
  };
  auto accuracy = [&]() {
    const auto* qp = static_cast<const char*>(qb.contents());
    const auto* sp = static_cast<const _Float16*>(sb.contents());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
      const double dq = (double)qp[i] * (double)sp[i / 64];
      const double e = dq - (double)x[i];
      num += e * e; den += (double)x[i] * (double)x[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  };

  const double bytes = (double)N * (2 + 1) + (double)(N / 64) * 2;
  const int ITERS = 25;
  ComputeFunction f_bfp2 = lib.function("quant_f16_i8_g64_bfp2");
  ASSERT_TRUE(f_bfp2.valid());
  struct { const char* tag; ComputeFunction* fn; } vs[] = {
      {"bfp (int, pow2 scale)", &f_bfp},
      {"bfp2 (float, pow2 scale)", &f_bfp2},
      {"amax (float scale)", &f_amax}};
  for (auto& v : vs) {
    launch(*v.fn, 1);                          // warm + output for accuracy
    const double r = accuracy();
    const auto t0 = std::chrono::steady_clock::now();
    launch(*v.fn, ITERS);
    const double s = secs_(t0, std::chrono::steady_clock::now());
    std::printf("[quant_i8] %-22s: %.0f GB/s (%.0f Gelem/s, %.3f ms / 32Mi)"
                "  dequant rel-L2 %.4e\n",
                v.tag, bytes * ITERS / s / 1e9, (double)N * ITERS / s / 1e9,
                s * 1e3 / ITERS, r);
    EXPECT_TRUE(r < 2e-2);                     // both must be sane int8
  }
}

// Full int8 GEMM pipeline prototype (the int8-FFN path): per-token
// activation quant (quant_f16_i8_row) -> i8 x i8 matmul2d into a tgmem
// i32 tile with the dequant-scale epilogue fused before the f16 store
// (gemm_i8i8_sc_f16_n64) -- weights quantized per-out-channel offline.
// Oracle: exact int8 simulation on the GPU's own quantized operands;
// quality: rel-L2 vs the f32 reference of the ORIGINAL f16 GEMM. Perf:
// (act-quant + i8 GEMM) vs the production f16 matmul2d routing at the
// flux2 FFN shapes.
TEST(gemm_i8, ffn_prototype) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib_mma = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_dq = mc->load_library("affine_dequant");
  ComputeFunction f_i8 = lib_mma.function("gemm_i8i8_sc_f16_n64");
  ComputeFunction f_qr = lib_dq.function("quant_f16_i8_row");
  ComputeFunction f_n128 = lib_mma.function("dense_gemm_mma_t_n128_f16");
  ComputeFunction f_deep = lib_mma.function("dense_gemm_mma_t_n128x256_f16");
  // THE MATCHED TILE. The i8 kernel is 64x64 over 4 simdgroups and
  // cannot be wider -- its destination is a THREADGROUP i32 tile, 16 KB
  // at 64x64 and 64 KB at 128x128, past the budget -- so comparing it
  // against the 128x128 f16 entry measures the tile and the dtype at
  // once. This is the same 64x64/4 shape in f16, which separates them.
  ComputeFunction f_64 = lib_mma.function("dense_gemm_mma_t_f16");
  if (!f_i8.valid() || !f_qr.valid() || !f_n128.valid() ||
      !f_deep.valid() || !f_64.valid()) {
    std::printf("[gemm_i8] kernels unavailable -- skip\n");
    return;
  }

  auto run_shape = [&](int M, int N, int K, bool check, int iters) {
    std::mt19937 rng(11u + (unsigned)(M + N + K));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<_Float16> x((std::size_t)M * K), w((std::size_t)N * K);
    for (auto& v : x) { v = (_Float16)(nd(rng) * 0.5f); }
    for (auto& v : w) { v = (_Float16)(nd(rng) * 0.05f); }
    // Offline per-out-channel weight quant (CPU): wq[n,k], ws[n].
    std::vector<std::int8_t> wq((std::size_t)N * K);
    std::vector<_Float16> ws(N);
    for (int n = 0; n < N; ++n) {
      float am = 0.0f;
      for (int k = 0; k < K; ++k) {
        am = std::max(am, std::fabs((float)w[(std::size_t)n * K + k]));
      }
      const float inv = am > 0 ? 127.0f / am : 0.0f;
      ws[n] = (_Float16)(am / 127.0f);
      for (int k = 0; k < K; ++k) {
        const float qv = std::rint((float)w[(std::size_t)n * K + k] * inv);
        wq[(std::size_t)n * K + k] =
            (std::int8_t)std::max(-127.0f, std::min(127.0f, qv));
      }
    }
    SharedBuffer xb = mc->make_shared_buffer(x.size() * 2);
    SharedBuffer xqb = mc->make_shared_buffer((std::size_t)M * K);
    SharedBuffer asb = mc->make_shared_buffer((std::size_t)M * 2);
    SharedBuffer wqb = mc->make_shared_buffer(wq.size());
    SharedBuffer wsb = mc->make_shared_buffer((std::size_t)N * 2);
    SharedBuffer wb = mc->make_shared_buffer(w.size() * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    SharedBuffer yb2 = mc->make_shared_buffer((std::size_t)M * N * 2);
    if (yb.empty() || yb2.empty() || wb.empty()) {
      std::printf("[gemm_i8] alloc failed %dx%dx%d -- skip\n", M, N, K);
      return;
    }
    std::memcpy(xb.contents(), x.data(), x.size() * 2);
    std::memcpy(wqb.contents(), wq.data(), wq.size());
    std::memcpy(wsb.contents(), ws.data(), (std::size_t)N * 2);
    std::memcpy(wb.contents(), w.data(), w.size() * 2);

    auto launch_i8 = [&](int n) {              // act quant + i8 GEMM
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          enc.set_function(f_qr);
          enc.set_buffer(0, xb); enc.set_buffer(1, xqb);
          enc.set_buffer(2, asb);
          enc.set_constant(3, K);
          enc.dispatch({256, (unsigned)M, 1}, {256, 1, 1});
          enc.set_function(f_i8);
          enc.set_buffer(0, xqb); enc.set_buffer(1, wqb);
          enc.set_buffer(2, asb); enc.set_buffer(3, wsb);
          enc.set_buffer(4, yb);
          enc.set_constant(5, K); enc.set_constant(6, N);
          enc.set_constant(7, M);
          enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                        (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
        }
      }
      st.commit().wait();
    };
    auto launch_f16 = [&](int n) {             // production f16 routing
      ComputeFunction& fn = (K < 6144) ? f_n128 : f_deep;
      const int RN = (K < 6144) ? 128 : 256;
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          enc.set_function(fn);
          enc.set_buffer(0, xb); enc.set_buffer(1, wb);
          enc.set_buffer(2, wb); enc.set_buffer(3, yb2);
          enc.set_constant(4, K); enc.set_constant(5, N);
          enc.set_constant(6, M); enc.set_constant(7, 0);
          enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                        (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        }
      }
      st.commit().wait();
    };

    auto launch_f16_64 = [&](int n) {           // the i8 kernel's own tile
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          enc.set_function(f_64);
          enc.set_buffer(0, xb); enc.set_buffer(1, wb);
          enc.set_buffer(2, wb); enc.set_buffer(3, yb2);
          enc.set_constant(4, K); enc.set_constant(5, N);
          enc.set_constant(6, M); enc.set_constant(7, 0);
          enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                        (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
        }
      }
      st.commit().wait();
    };

    launch_i8(1); launch_f16(1); launch_f16_64(1);
    if (check) {
      // Oracle: integer simulation with the GPU's own xq/as (read back)
      // + the CPU wq/ws -- the GPU result must match to f16 store
      // rounding. Quality: the i8 result vs the f64 reference of the
      // original f16 GEMM.
      const auto* xqp = static_cast<const std::int8_t*>(xqb.contents());
      const auto* asp = static_cast<const _Float16*>(asb.contents());
      const auto* yp = static_cast<const _Float16*>(yb.contents());
      double n_or = 0.0, d_or = 0.0, n_q = 0.0, d_q = 0.0;
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          std::int64_t isum = 0;
          double fsum = 0.0;
          for (int k = 0; k < K; ++k) {
            isum += (std::int64_t)xqp[(std::size_t)m * K + k] *
                    (std::int64_t)wq[(std::size_t)n * K + k];
            fsum += (double)x[(std::size_t)m * K + k] *
                    (double)w[(std::size_t)n * K + k];
          }
          const double ref =
              (double)isum * (double)asp[m] * (double)ws[n];
          const double got = (double)yp[(std::size_t)m * N + n];
          n_or += (got - ref) * (got - ref); d_or += ref * ref;
          n_q += (got - fsum) * (got - fsum); d_q += fsum * fsum;
        }
      }
      const double r_or = d_or > 0 ? std::sqrt(n_or / d_or) : 0.0;
      const double r_q = d_q > 0 ? std::sqrt(n_q / d_q) : 0.0;
      std::printf("[gemm_i8] oracle rel-L2 %.4e %s | int8-vs-f32 quality "
                  "rel-L2 %.4e (%dx%dx%d)\n", r_or,
                  r_or < 1e-2 ? "MATCH" : "(BAD)", r_q, M, N, K);
      EXPECT_TRUE(r_or < 1e-2);
    }
    const double gflop = 2.0 * M * N * (double)K * iters / 1e9;
    auto t0 = std::chrono::steady_clock::now();
    launch_i8(iters);
    auto t1 = std::chrono::steady_clock::now();
    launch_f16(iters);
    auto t2 = std::chrono::steady_clock::now();
    launch_f16_64(iters);
    auto t3 = std::chrono::steady_clock::now();
    const double s_i8 = secs_(t0, t1), s_f16 = secs_(t1, t2),
                 s_64 = secs_(t2, t3);
    std::printf("[gemm_i8] M=%4d N=%5d K=%5d: i8 %.0f GF/s  f16-best %.0f"
                "  f16-same-tile %.0f  |  i8/best %.2fx  i8/same-tile "
                "%.2fx\n", M, N, K, gflop / s_i8, gflop / s_f16,
                gflop / s_64, s_f16 / s_i8, s_64 / s_i8);
  };
  run_shape(256, 512, 1024, /*check=*/true, 10);      // oracle + quality
  run_shape(4096, 12288, 4096, /*check=*/false, 5);   // flux2 block proj
  run_shape(4096, 4096, 12288, /*check=*/false, 5);   // flux2 ff-down
  run_shape(4096, 36864, 4096, /*check=*/false, 3);   // flux2 qkv_mlp

  // AND THE CONTRACTION-DEPTH SWEEP, at one output size so that only K
  // moves. This is the question an int8 ATTENTION turns on:
  // SageAttention's speedup comes from running QK^T in int8, and that
  // product contracts over head_dim -- 128 in every image and video
  // model in this tree, 64 in the vision and audio encoders, and not a
  // number a kernel can choose. So whether the method can pay here is
  // exactly whether int8 is faster than f16 at K = 128, which the flux2
  // rows above (K >= 4096) cannot answer.
  //
  // READ THE CHUNK-DEPTH SWEEP IN gemm_i8.k512_chunked BEFORE CONCLUDING
  // ANYTHING FROM THIS ONE. The rows below are standalone GEMMs, which
  // pay a threadgroup's setup, epilogue and DRAM store for every K of
  // contraction -- so at K=128 they are measuring that store, not the
  // matrix pipe, and they understate int8 by nearly 2x for any caller
  // whose loop amortises the store. A flash attention is exactly such a
  // caller.
  //
  // MEASURED on an M5 (M=N=4096), TFLOP/s, against f16 at the SAME 64x64
  // tile so the dtype is the only variable:
  //
  //     K      64   128   256   512  1024  2048
  //   int8    5.3   9.2  12.6  17.1  22.4  23.4
  //   f16     4.7   7.9  12.3  13.5  13.6  11.2
  //   ratio  1.14  1.16  1.02  1.26  1.65  2.09
  //
  // THE 2x IS REAL AND IT ARRIVES WITH DEPTH -- 17 TFLOP/s at K=512 and
  // 22+ past 1024, which is where every number anyone quotes for int8 on
  // this part comes from. Attention has no depth to give it: the QK
  // product contracts over head_dim, so K is 128, where the ratio is
  // 1.16.
  //
  // AND AT K=128 A STANDALONE GEMM IS BADLY SERVED IN BOTH DTYPES -- 7.9
  // and 9.2 against their own 13.6 and 23.4 bests. That is a property of
  // the standalone shape and NOT of the pipe: the same 128-deep int8
  // product issued back to back into a live accumulator holds 20.8
  // TFLOP/s, which is 0.92x of what it does at 512. The fixed cost being
  // amortised is the THREADGROUP's, over how much work that threadgroup
  // does -- not the matmul2d call's.
  //
  // Sage's OTHER lever is not available either: it runs P*V with an f16
  // accumulator, and matmul2d's relaxed_precision was measured on this
  // part as rate-neutral (the f32 accumulator stays). See
  // sage-attention.cc for the method's accuracy, which is not the
  // problem -- 0.99991 cosine at the kernel's own 64x32 block.
  for (const int k : {64, 128, 256, 512, 1024, 2048}) {
    run_shape(4096, 4096, k, /*check=*/false, 5);
  }

  // AND THE SAME FLOPS AT EVERY DEPTH, which is what separates the two
  // explanations for the shallow-K shortfall.
  //
  // Every row below is 4.29 GFLOP. What changes is how it is divided:
  // 4096 output tiles each reducing over 128, down to 64 tiles each
  // reducing over 8192. If the rate is flat, the cost is proportional to
  // the arithmetic and depth is irrelevant; if it climbs with K, the
  // per-tile fixed cost -- accumulator setup, the epilogue, the store,
  // and the matrix pipe's own fill and drain -- is what shallow tiles
  // cannot amortise. Note the shallow rows move MORE bytes, not fewer
  // (35.6 MB against 10.5), so a bandwidth answer would point the other
  // way.
  for (const auto& [m, k] : std::vector<std::pair<int, int>>{
           {4096, 128}, {2048, 512}, {1024, 2048}, {512, 8192}}) {
    run_shape(m, m, k, /*check=*/false, 5);
  }
}

// K=512-chunked int8 GEMM variants (see dense_gemm_mma.metal):
//   kacc -- mode::multiply_accumulate accumulates raw i32 chunk partials
//           (same scales as the single op; must match it EXACTLY -- i32
//           addition is associative -- so it isolates the chunking cost);
//   g512 -- per-chunk multiply + PER-GROUP scales (as[m,g] * ws[n,g])
//           folded into a register float accumulator: the accuracy win
//           (512-deep quant groups along K on BOTH operands).
// Quality + oracle at a small shape, then perf vs the single-op i8 kernel
// at the flux2 FFN shapes.
TEST(gemm_i8, the_vdn_readout_projection_survives_its_zero_rows) {
  // THE VDN BRANCH'S OUTPUT PROJECTION under accelerated mode.
  //
  // The branch hands its readout to the TRANSFORMER's GEMM -- that is
  // where the quantisation lives -- so with `i8_gemm` on, MiniMax-H3's
  // to_out_linear takes the int8 path. Its shape qualifies exactly:
  // K = inner = 7168 is 14 whole 512-groups, so there is not even a pad,
  // and M is the video row count, far past the 1024-row gate.
  //
  // WHAT IS UNUSUAL ABOUT IT is the input. Under `anchors: both` the
  // branch never writes frames 0 and F-1 -- the softmax half covers them
  // in both directions, so their readout is exactly zero by the
  // partition -- and the buffer is zeroed once and reused across all 50
  // blocks. So ~1020 of 18870 rows are ALL ZERO on every block, which is
  // a shape a dynamic per-row-group absmax quantiser has to have an
  // answer for: the scale is the absmax, and the absmax of nothing is 0.
  //
  // The answer is a guard in quant_f16_i8_row_g512 (`am > 0 ? 127/am :
  // 0`), and this is what says it holds end to end rather than by
  // reading it -- an unguarded divide would put inf in the scale and NaN
  // in the residual stream of two frames, which renders as two bad
  // frames in an otherwise fine clip.
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  // bf16, as H3 constructs it: its residual stream is bf16 throughout.
  vpipe::genai::I8GemmContext i8(mc, /*want=*/true, /*bf16=*/true);
  if (!i8.enabled()) {
    std::printf("[gemm_i8vdn] accelerated mode unavailable -- skip\n");
    return;
  }
  const int M = 2040, N = 5376, K = 7168;   // 4 frames of 510, hidden, inner
  EXPECT_TRUE(i8.accepts(M, N, K));
  ComputeLibrary lib = mc->load_library("dense_gemm_mma_bf16");
  ComputeFunction ref = lib.function("dense_gemm_mma_t_n128_f16");
  if (!ref.valid()) { return; }

  auto bf = [](float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  };
  auto unbf = [](std::uint16_t b) {
    const std::uint32_t u = (std::uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * K * 2);
  SharedBuffer wb = mc->make_shared_buffer((std::size_t)N * K * 2);
  SharedBuffer y8 = mc->make_shared_buffer((std::size_t)M * N * 2);
  SharedBuffer yr = mc->make_shared_buffer((std::size_t)M * N * 2);
  if (xb.empty() || wb.empty() || y8.empty() || yr.empty()) { return; }
  // The FIRST and LAST frame's rows are the anchors: exactly zero.
  const int tpf = 510;
  std::mt19937 rng(99u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  {
    auto* x = static_cast<std::uint16_t*>(xb.contents());
    for (int r = 0; r < M; ++r) {
      const bool anchor = (r < tpf) || (r >= M - tpf);
      for (int k = 0; k < K; ++k) {
        x[(std::size_t)r * K + k] = anchor ? 0 : bf(nd(rng) * 0.5f);
      }
    }
    auto* w = static_cast<std::uint16_t*>(wb.contents());
    for (std::size_t i = 0; i < (std::size_t)N * K; ++i) {
      w[i] = bf(nd(rng) * 0.05f);
    }
    std::memset(y8.contents(), 0xff, y8.byte_size());   // not pre-zeroed
    std::memset(yr.contents(), 0xff, yr.byte_size());
  }

  { CommandStream st = mc->make_command_stream();
    { ComputeEncoder enc = st.begin_compute();
      const bool took = i8.gemm(enc, xb, 0, wb, y8, 0, M, N, K);
      EXPECT_TRUE(took);
      if (!took) { return; }
      // The bf16 arm the mode replaces, for the same operands.
      enc.set_function(ref);
      enc.set_buffer(0, xb); enc.set_buffer(1, wb); enc.set_buffer(2, wb);
      enc.set_buffer(3, yr);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + 127) / 128) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      enc.end();
    }
    std::string e;
    ASSERT_TRUE(st.commit().wait_ok(&e));
  }

  const auto* a = static_cast<const std::uint16_t*>(y8.contents());
  const auto* b = static_cast<const std::uint16_t*>(yr.contents());
  double num = 0.0, den = 0.0;
  std::size_t nonfinite = 0, anchor_nonzero = 0;
  for (int r = 0; r < M; ++r) {
    const bool anchor = (r < tpf) || (r >= M - tpf);
    for (int n = 0; n < N; ++n) {
      const float g = unbf(a[(std::size_t)r * N + n]);
      const float w = unbf(b[(std::size_t)r * N + n]);
      if (!std::isfinite(g)) { ++nonfinite; }
      if (anchor && g != 0.0f) { ++anchor_nonzero; }
      const double d = (double)g - (double)w;
      num += d * d;
      den += (double)w * (double)w;
    }
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : -1.0;
  std::printf("[gemm_i8vdn] %dx%dx%d, %d anchor rows: rel-L2 %.3e, "
              "%zu non-finite, %zu anchor outputs non-zero\n", M, N, K,
              2 * tpf, rel, nonfinite, anchor_nonzero);
  // The zero rows are the point: exactly zero out, not NaN and not noise.
  EXPECT_TRUE(nonfinite == 0);
  EXPECT_TRUE(anchor_nonzero == 0);
  // And the rest is int8's usual ~1e-2, which is what the mode costs
  // everywhere and is why it is opt-in.
  EXPECT_TRUE(rel >= 0.0 && rel < 0.05);
}

TEST(gemm_i8, k512_chunked) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib_mma = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_dq = mc->load_library("affine_dequant");
  ComputeFunction f_one = lib_mma.function("gemm_i8i8_sc_f16_n64");
  ComputeFunction f_ka = lib_mma.function("gemm_i8i8_sc_f16_n64_kacc");
  // THE CHUNK DEPTH SWEPT DOWN TO ATTENTION'S, AND THIS IS THE ROW THAT
  // MATTERS FOR AN INT8 ATTENTION.
  //
  // A flash kernel issues one 128-deep QK product per key block, back to
  // back into a live accumulator, with ONE store for the whole loop --
  // which is this kernel's shape and not a standalone K=128 GEMM's.
  //
  // MEASURED on an M5, TFLOP/s at 128-deep chunks: int8 20.8, f16 11.0,
  // a ratio of 1.89x. The f16 figure is itself the check that this
  // harness is faithful: the real flash kernel measures 10.5 TFLOP/s at
  // video geometry, within 5% of the 11.0 here at the same depth and
  // structure. So the 1.16x a standalone K=128 GEMM shows was measuring
  // that GEMM's per-tile store, and an int8 QK is worth ~1.9x on half a
  // flash kernel's matrix work -- about 1.3x on the kernel.
  ComputeFunction f_k256 =
      lib_mma.function("gemm_i8i8_sc_f16_n64_kacc_k256");
  ComputeFunction f_k128 =
      lib_mma.function("gemm_i8i8_sc_f16_n64_kacc_k128");
  // ...and the f16 twin, so the ratio is the dtype rather than the shape.
  ComputeFunction f_f128 = lib_mma.function("dense_gemm_mma_kacc_k128_f16");
  ComputeFunction f_f512 = lib_mma.function("dense_gemm_mma_kacc_k512_f16");
  ComputeFunction f_g5 = lib_mma.function("gemm_i8i8_sc_f16_n64_g512");
  ComputeFunction f_qr = lib_dq.function("quant_f16_i8_row");
  ComputeFunction f_qg = lib_dq.function("quant_f16_i8_row_g512");
  if (!f_one.valid() || !f_ka.valid() || !f_g5.valid() || !f_qr.valid() ||
      !f_qg.valid()) {
    std::printf("[gemm_i8k] kernels unavailable -- skip\n");
    return;
  }

  auto run_shape = [&](int M, int N, int K, bool check, int iters) {
    const int G = K / 512;
    std::mt19937 rng(23u + (unsigned)(M + N + K));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<_Float16> x((std::size_t)M * K), w((std::size_t)N * K);
    for (auto& v : x) { v = (_Float16)(nd(rng) * 0.5f); }
    for (auto& v : w) { v = (_Float16)(nd(rng) * 0.05f); }
    // Offline weight quant, BOTH granularities: per-channel (single/kacc)
    // and per-(channel, 512-group) (g512).
    std::vector<std::int8_t> wq((std::size_t)N * K), wqg((std::size_t)N * K);
    std::vector<_Float16> ws(N), wsg((std::size_t)N * G);
    for (int n = 0; n < N; ++n) {
      float am = 0.0f;
      for (int k = 0; k < K; ++k) {
        am = std::max(am, std::fabs((float)w[(std::size_t)n * K + k]));
      }
      float inv = am > 0 ? 127.0f / am : 0.0f;
      ws[n] = (_Float16)(am / 127.0f);
      for (int k = 0; k < K; ++k) {
        const float qv = std::rint((float)w[(std::size_t)n * K + k] * inv);
        wq[(std::size_t)n * K + k] =
            (std::int8_t)std::max(-127.0f, std::min(127.0f, qv));
      }
      for (int g = 0; g < G; ++g) {
        float ag = 0.0f;
        for (int k = g * 512; k < (g + 1) * 512; ++k) {
          ag = std::max(ag, std::fabs((float)w[(std::size_t)n * K + k]));
        }
        const float ig = ag > 0 ? 127.0f / ag : 0.0f;
        wsg[(std::size_t)n * G + g] = (_Float16)(ag / 127.0f);
        for (int k = g * 512; k < (g + 1) * 512; ++k) {
          const float qv = std::rint((float)w[(std::size_t)n * K + k] * ig);
          wqg[(std::size_t)n * K + k] =
              (std::int8_t)std::max(-127.0f, std::min(127.0f, qv));
        }
      }
    }
    SharedBuffer xb = mc->make_shared_buffer(x.size() * 2);
    SharedBuffer xqb = mc->make_shared_buffer((std::size_t)M * K);
    SharedBuffer asb = mc->make_shared_buffer((std::size_t)M * 2);
    SharedBuffer asgb = mc->make_shared_buffer((std::size_t)M * G * 2);
    SharedBuffer wqb = mc->make_shared_buffer(wq.size());
    SharedBuffer wqgb = mc->make_shared_buffer(wqg.size());
    SharedBuffer wsb = mc->make_shared_buffer((std::size_t)N * 2);
    SharedBuffer wsgb = mc->make_shared_buffer((std::size_t)N * G * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    SharedBuffer yb2 = mc->make_shared_buffer((std::size_t)M * N * 2);
    // The UNQUANTIZED weight, for the f16 arm of the chunk-depth sweep.
    SharedBuffer wfb = mc->make_shared_buffer(w.size() * 2);
    if (!wfb.empty()) {
      std::memcpy(wfb.contents(), w.data(), w.size() * 2);
    }
    if (yb.empty() || yb2.empty() || wqgb.empty()) {
      std::printf("[gemm_i8k] alloc failed %dx%dx%d -- skip\n", M, N, K);
      return;
    }
    std::memcpy(xb.contents(), x.data(), x.size() * 2);
    std::memcpy(wqb.contents(), wq.data(), wq.size());
    std::memcpy(wqgb.contents(), wqg.data(), wqg.size());
    std::memcpy(wsb.contents(), ws.data(), (std::size_t)N * 2);
    std::memcpy(wsgb.contents(), wsg.data(), (std::size_t)N * G * 2);

    // One generic launcher: quant kernel + gemm kernel + operand set.
    auto launch = [&](ComputeFunction& fq, ComputeFunction& fg,
                      SharedBuffer& scl, SharedBuffer& wqx,
                      SharedBuffer& wsx, SharedBuffer& dst, int n) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          enc.set_function(fq);
          enc.set_buffer(0, xb); enc.set_buffer(1, xqb);
          enc.set_buffer(2, scl);
          enc.set_constant(3, K);
          enc.dispatch({256, (unsigned)M, 1}, {256, 1, 1});
          enc.set_function(fg);
          enc.set_buffer(0, xqb); enc.set_buffer(1, wqx);
          enc.set_buffer(2, scl); enc.set_buffer(3, wsx);
          enc.set_buffer(4, dst);
          enc.set_constant(5, K); enc.set_constant(6, N);
          enc.set_constant(7, M);
          enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                        (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
        }
      }
      st.commit().wait();
    };

    if (check) {
      // kacc must match the single op EXACTLY (i32 associativity).
      launch(f_qr, f_one, asb, wqb, wsb, yb, 1);
      launch(f_qr, f_ka, asb, wqb, wsb, yb2, 1);
      const auto* a = static_cast<const _Float16*>(yb.contents());
      const auto* b = static_cast<const _Float16*>(yb2.contents());
      std::size_t diff = 0;
      for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
        if ((float)a[i] != (float)b[i]) { ++diff; }
      }
      std::printf("[gemm_i8k] kacc vs single-op: %zu/%d mismatches %s\n",
                  diff, M * N, diff == 0 ? "EXACT" : "(BAD)");
      EXPECT_TRUE(diff == 0);

      // g512: oracle (GPU xq/as + CPU wqg/wsg int sim) + quality vs f32.
      launch(f_qg, f_g5, asgb, wqgb, wsgb, yb2, 1);
      const auto* xqp = static_cast<const std::int8_t*>(xqb.contents());
      const auto* asp = static_cast<const _Float16*>(asgb.contents());
      const auto* yp = static_cast<const _Float16*>(yb2.contents());
      double n_or = 0.0, d_or = 0.0, n_q = 0.0, d_q = 0.0;
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          double acc = 0.0, fsum = 0.0;
          for (int g = 0; g < G; ++g) {
            std::int64_t isum = 0;
            for (int k = g * 512; k < (g + 1) * 512; ++k) {
              isum += (std::int64_t)xqp[(std::size_t)m * K + k] *
                      (std::int64_t)wqg[(std::size_t)n * K + k];
            }
            acc += (double)isum * (double)asp[(std::size_t)m * G + g] *
                   (double)wsg[(std::size_t)n * G + g];
          }
          for (int k = 0; k < K; ++k) {
            fsum += (double)x[(std::size_t)m * K + k] *
                    (double)w[(std::size_t)n * K + k];
          }
          const double got = (double)yp[(std::size_t)m * N + n];
          n_or += (got - acc) * (got - acc); d_or += acc * acc;
          n_q += (got - fsum) * (got - fsum); d_q += fsum * fsum;
        }
      }
      const double r_or = d_or > 0 ? std::sqrt(n_or / d_or) : 0.0;
      const double r_q = d_q > 0 ? std::sqrt(n_q / d_q) : 0.0;
      std::printf("[gemm_i8k] g512 oracle rel-L2 %.4e %s | g512-vs-f32 "
                  "quality rel-L2 %.4e (%dx%dx%d)\n", r_or,
                  r_or < 1e-2 ? "MATCH" : "(BAD)", r_q, M, N, K);
      EXPECT_TRUE(r_or < 1e-2);
      return;
    }

    const double gflop = 2.0 * M * N * (double)K * iters / 1e9;
    launch(f_qr, f_one, asb, wqb, wsb, yb, 1);      // warm
    auto t0 = std::chrono::steady_clock::now();
    launch(f_qr, f_one, asb, wqb, wsb, yb, iters);
    auto t1 = std::chrono::steady_clock::now();
    launch(f_qr, f_ka, asb, wqb, wsb, yb, iters);
    auto t2 = std::chrono::steady_clock::now();
    launch(f_qg, f_g5, asgb, wqgb, wsgb, yb, iters);
    auto t3 = std::chrono::steady_clock::now();
    launch(f_qr, f_k256, asb, wqb, wsb, yb, iters);
    auto t4 = std::chrono::steady_clock::now();
    launch(f_qr, f_k128, asb, wqb, wsb, yb, iters);
    auto t5 = std::chrono::steady_clock::now();
    auto launch_f16k = [&](ComputeFunction& fn, int n) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          enc.set_function(fn);
          enc.set_buffer(0, xb); enc.set_buffer(1, wfb);
          enc.set_buffer(2, wfb); enc.set_buffer(3, yb);
          enc.set_constant(4, K); enc.set_constant(5, N);
          enc.set_constant(6, M); enc.set_constant(7, 0);
          enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                        (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
        }
      }
      st.commit().wait();
    };
    launch_f16k(f_f512, 1);
    auto t6 = std::chrono::steady_clock::now();
    launch_f16k(f_f512, iters);
    auto t7 = std::chrono::steady_clock::now();
    launch_f16k(f_f128, iters);
    auto t8 = std::chrono::steady_clock::now();
    const double sf5 = secs_(t6, t7), sf1 = secs_(t7, t8);
    const double s1 = secs_(t0, t1), s2 = secs_(t1, t2), s3 = secs_(t2, t3),
                 s4 = secs_(t3, t4), s5 = secs_(t4, t5);
    std::printf("[gemm_i8k] M=%4d N=%5d K=%5d: single %.0f GF/s (%.2f ms)"
                "  kacc %.0f GF/s (%.2f ms, %.2fx)  g512 %.0f GF/s "
                "(%.2f ms, %.2fx)\n",
                M, N, K, gflop / s1, s1 * 1e3 / iters,
                gflop / s2, s2 * 1e3 / iters, s1 / s2,
                gflop / s3, s3 * 1e3 / iters, s1 / s3);
    std::printf("[gemm_i8k]   chunk depth, same accumulator: int8 512 %.0f"
                "  256 %.0f  128 %.0f GF/s | f16 512 %.0f  128 %.0f | "
                "i8/f16 at 128 %.2fx\n",
                gflop / s2, gflop / s4, gflop / s5, gflop / sf5,
                gflop / sf1, sf1 / s5);
  };
  run_shape(256, 512, 1024, /*check=*/true, 1);       // oracles + quality
  run_shape(4096, 12288, 4096, /*check=*/false, 5);   // flux2 block proj
  run_shape(4096, 4096, 12288, /*check=*/false, 5);   // flux2 ff-down
  run_shape(4096, 36864, 4096, /*check=*/false, 3);   // flux2 qkv_mlp
}

// Shift-aligned i32 accumulation (gemm_i8i8_sc_f16_n64_g512i): per-512-
// group POW2 quant scales carried as int8 exponents; the K-chunk loop
// keeps a pure i32 accumulator + tracked exponent per element, aligning
// by rounded right-shifts (block-floating-point accumulate -- no float
// scale multiplies; one ldexp at the end). Oracle = an exact CPU replica
// of the algorithm on the GPU's own quantized operands; quality vs f32;
// perf vs the single-op and float-accumulate g512 kernels.
TEST(gemm_i8, k512_shift_acc) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib_mma = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_dq = mc->load_library("affine_dequant");
  ComputeFunction f_one = lib_mma.function("gemm_i8i8_sc_f16_n64");
  ComputeFunction f_g5 = lib_mma.function("gemm_i8i8_sc_f16_n64_g512");
  ComputeFunction f_gi = lib_mma.function("gemm_i8i8_sc_f16_n64_g512i");
  ComputeFunction f_qr = lib_dq.function("quant_f16_i8_row");
  ComputeFunction f_qg = lib_dq.function("quant_f16_i8_row_g512");
  ComputeFunction f_qb = lib_dq.function("quant_f16_i8_row_g512_bfp");
  if (!f_one.valid() || !f_g5.valid() || !f_gi.valid() || !f_qr.valid() ||
      !f_qg.valid() || !f_qb.valid()) {
    std::printf("[gemm_i8s] kernels unavailable -- skip\n");
    return;
  }

  auto run_shape = [&](int M, int N, int K, bool check, int iters) {
    const int G = K / 512;
    std::mt19937 rng(37u + (unsigned)(M + N + K));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<_Float16> x((std::size_t)M * K), w((std::size_t)N * K);
    for (auto& v : x) { v = (_Float16)(nd(rng) * 0.5f); }
    for (auto& v : w) { v = (_Float16)(nd(rng) * 0.05f); }
    // Offline pow2 weight quant per (channel, 512-group), mirroring the
    // GPU activation scheme: scale = 2^(Emax16 - 21).
    std::vector<std::int8_t> wq((std::size_t)N * K), ewv((std::size_t)N * G);
    // amax f16 scales too (for the single-op / float-g512 baselines).
    std::vector<std::int8_t> wqa((std::size_t)N * K);
    std::vector<_Float16> ws(N), wsg((std::size_t)N * G);
    for (int n = 0; n < N; ++n) {
      float am = 0.0f;
      for (int k = 0; k < K; ++k) {
        am = std::max(am, std::fabs((float)w[(std::size_t)n * K + k]));
      }
      const float inv = am > 0 ? 127.0f / am : 0.0f;
      ws[n] = (_Float16)(am / 127.0f);
      for (int k = 0; k < K; ++k) {
        const float qv = std::rint((float)w[(std::size_t)n * K + k] * inv);
        wqa[(std::size_t)n * K + k] =
            (std::int8_t)std::max(-127.0f, std::min(127.0f, qv));
      }
      for (int g = 0; g < G; ++g) {
        int em = 0;
        for (int k = g * 512; k < (g + 1) * 512; ++k) {
          std::uint16_t u;
          std::memcpy(&u, &w[(std::size_t)n * K + k], 2);
          em = std::max(em, (int)((u >> 10) & 0x1F));
        }
        ewv[(std::size_t)n * G + g] = (std::int8_t)(em - 21);
        const float ig = std::ldexp(1.0f, 21 - em);
        for (int k = g * 512; k < (g + 1) * 512; ++k) {
          const float qv = std::rint((float)w[(std::size_t)n * K + k] * ig);
          wq[(std::size_t)n * K + k] =
              (std::int8_t)std::max(-127.0f, std::min(127.0f, qv));
        }
      }
    }
    SharedBuffer xb = mc->make_shared_buffer(x.size() * 2);
    SharedBuffer xqb = mc->make_shared_buffer((std::size_t)M * K);
    SharedBuffer asb = mc->make_shared_buffer((std::size_t)M * 2);
    SharedBuffer eab = mc->make_shared_buffer((std::size_t)M * G);
    SharedBuffer wqb = mc->make_shared_buffer(wq.size());
    SharedBuffer wqab = mc->make_shared_buffer(wqa.size());
    SharedBuffer ewb = mc->make_shared_buffer(ewv.size());
    SharedBuffer wsb = mc->make_shared_buffer((std::size_t)N * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    if (yb.empty() || wqb.empty() || wqab.empty()) {
      std::printf("[gemm_i8s] alloc failed %dx%dx%d -- skip\n", M, N, K);
      return;
    }
    std::memcpy(xb.contents(), x.data(), x.size() * 2);
    std::memcpy(wqb.contents(), wq.data(), wq.size());
    std::memcpy(wqab.contents(), wqa.data(), wqa.size());
    std::memcpy(ewb.contents(), ewv.data(), ewv.size());
    std::memcpy(wsb.contents(), ws.data(), (std::size_t)N * 2);

    auto launch = [&](ComputeFunction& fq, ComputeFunction& fg,
                      SharedBuffer& scl, SharedBuffer& wqx,
                      SharedBuffer& wsx, int n) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          enc.set_function(fq);
          enc.set_buffer(0, xb); enc.set_buffer(1, xqb);
          enc.set_buffer(2, scl);
          enc.set_constant(3, K);
          enc.dispatch({256, (unsigned)M, 1}, {256, 1, 1});
          enc.set_function(fg);
          enc.set_buffer(0, xqb); enc.set_buffer(1, wqx);
          enc.set_buffer(2, scl); enc.set_buffer(3, wsx);
          enc.set_buffer(4, yb);
          enc.set_constant(5, K); enc.set_constant(6, N);
          enc.set_constant(7, M);
          enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                        (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
        }
      }
      st.commit().wait();
    };

    if (check) {
      launch(f_qb, f_gi, eab, wqb, ewb, 1);
      // Exact CPU replica of the shift-align accumulate on the GPU's own
      // xq/ea + the CPU wq/ew (same group order, same rounded shifts).
      const auto* xqp = static_cast<const std::int8_t*>(xqb.contents());
      const auto* eap = static_cast<const std::int8_t*>(eab.contents());
      const auto* yp = static_cast<const _Float16*>(yb.contents());
      double n_or = 0.0, d_or = 0.0, n_q = 0.0, d_q = 0.0;
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          std::int64_t acc = 0;
          int ae = -1000;
          double fsum = 0.0;
          for (int g = 0; g < G; ++g) {
            std::int64_t p = 0;
            for (int k = g * 512; k < (g + 1) * 512; ++k) {
              p += (std::int64_t)xqp[(std::size_t)m * K + k] *
                   (std::int64_t)wq[(std::size_t)n * K + k];
            }
            const int eg = (int)eap[(std::size_t)m * G + g] +
                           (int)ewv[(std::size_t)n * G + g];
            const int d = eg - ae;
            if (d > 0) {
              acc = (d >= 31) ? 0 : ((acc + (1ll << (d - 1))) >> d);
              acc += p;
              ae = eg;
            } else if (d < 0) {
              const int dd = -d;
              p = (dd >= 31) ? 0 : ((p + (1ll << (dd - 1))) >> dd);
              acc += p;
            } else {
              acc += p;
            }
          }
          for (int k = 0; k < K; ++k) {
            fsum += (double)x[(std::size_t)m * K + k] *
                    (double)w[(std::size_t)n * K + k];
          }
          const double ref =
              (double)(_Float16)std::ldexp((float)acc, ae);
          const double got = (double)yp[(std::size_t)m * N + n];
          n_or += (got - ref) * (got - ref); d_or += ref * ref;
          n_q += (got - fsum) * (got - fsum); d_q += fsum * fsum;
        }
      }
      const double r_or = d_or > 0 ? std::sqrt(n_or / d_or) : 0.0;
      const double r_q = d_q > 0 ? std::sqrt(n_q / d_q) : 0.0;
      std::printf("[gemm_i8s] g512i oracle rel-L2 %.4e %s | g512i-vs-f32 "
                  "quality rel-L2 %.4e (%dx%dx%d)\n", r_or,
                  r_or < 1e-3 ? "MATCH" : "(BAD)", r_q, M, N, K);
      EXPECT_TRUE(r_or < 1e-3);
      return;
    }

    // Perf: single-op (amax weights) vs shift-align g512i (pow2 weights).
    const double gflop = 2.0 * M * N * (double)K * iters / 1e9;
    launch(f_qr, f_one, asb, wqab, wsb, 1);           // warm
    auto t0 = std::chrono::steady_clock::now();
    launch(f_qr, f_one, asb, wqab, wsb, iters);
    auto t1 = std::chrono::steady_clock::now();
    launch(f_qb, f_gi, eab, wqb, ewb, iters);
    auto t2 = std::chrono::steady_clock::now();
    const double s1 = secs_(t0, t1), s2 = secs_(t1, t2);
    std::printf("[gemm_i8s] M=%4d N=%5d K=%5d: single %.0f GF/s (%.2f ms)"
                "  g512i-shift %.0f GF/s (%.2f ms, %.2fx)\n",
                M, N, K, gflop / s1, s1 * 1e3 / iters,
                gflop / s2, s2 * 1e3 / iters, s1 / s2);
  };
  run_shape(256, 512, 1024, /*check=*/true, 1);       // oracle + quality
  run_shape(4096, 12288, 4096, /*check=*/false, 5);   // flux2 block proj
  run_shape(4096, 4096, 12288, /*check=*/false, 5);   // flux2 ff-down
  run_shape(4096, 36864, 4096, /*check=*/false, 3);   // flux2 qkv_mlp
}

// The Krea-2 DiT quant path (metal-krea2-transformer gemm_mma_) expands an
// affine weight with affine_dequant then runs the DENSE matmul2d -- not the
// fused affine_qmm_mma. Verify that exact f16 sequence (affine_dequant_w{4,8}g64
// -> dense_gemm_mma_t_n128_f16) matches an f32 oracle AND the steel qmm the DiT
// falls back to, for both bit widths (group 64 = the Krea-2 quantization
// default; mixed precision picks w4/w8 per weight). This is the kernel-level
// oracle for the quant path the DiT ships with (no pre-quantized checkpoint is
// needed on the box). f16 libs = exactly what the DiT loads.
TEST(qmm_mma, dequant_dense_matches_steel) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[dequant_dense] no matrix cores -- skip\n");
    return;
  }
  ComputeLibrary lib_dq    = mc->load_library("affine_dequant");
  ComputeLibrary lib_dense = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_steel = mc->load_library("affine_qmm_steel");
  ComputeFunction f_dq4   = lib_dq.function("affine_dequant_w4g64");
  ComputeFunction f_dq8   = lib_dq.function("affine_dequant_w8g64");
  ComputeFunction f_dense = lib_dense.function("dense_gemm_mma_t_n128_f16");
  ComputeFunction f_st4   = lib_steel.function("affine_qmm_steel_w4g64");
  ComputeFunction f_st8   = lib_steel.function("affine_qmm_steel_w8g64");
  ASSERT_TRUE(f_dq4.valid() && f_dq8.valid() && f_dense.valid());
  ASSERT_TRUE(f_st4.valid() && f_st8.valid());

  const int M = 192, N = 512, K = 256;   // DiT-shaped (seq x proj x hidden slice)
  std::mt19937 rng(4242);
  std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
  std::vector<_Float16> x((std::size_t)M * K);
  for (auto& v : x) { v = (_Float16)(xd(rng) * 0.1f); }
  SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * K * 2);
  std::memcpy(xb.contents(), x.data(), x.size() * 2);

  auto run_bits = [&](int bits, ComputeFunction& f_dq, ComputeFunction& f_st) {
    const int G = K / 64;
    const int qmax = (bits == 8) ? 255 : 15;
    // w4 packs 2 nibbles/byte (K/2 bytes/row); w8 one byte/value (K bytes/row).
    const std::size_t row_bytes =
        (bits == 8) ? (std::size_t)K : (std::size_t)(K / 2);
    std::uniform_int_distribution<int> qd(0, qmax);
    std::uniform_real_distribution<float> sd(0.003f, 0.02f);
    std::uniform_real_distribution<float> bd(-0.1f, 0.1f);
    std::vector<std::uint8_t> packed((std::size_t)N * row_bytes, 0);
    std::vector<_Float16> scales((std::size_t)N * G), biases((std::size_t)N * G);
    std::vector<float> deq((std::size_t)N * K);
    for (int n = 0; n < N; ++n) {
      for (int g = 0; g < G; ++g) {
        scales[(std::size_t)n * G + g] = (_Float16)sd(rng);
        biases[(std::size_t)n * G + g] = (_Float16)bd(rng);
      }
      for (int k = 0; k < K; ++k) {
        const int v = qd(rng);
        if (bits == 8) {
          packed[(std::size_t)n * row_bytes + k] = (std::uint8_t)v;
        } else {
          const std::size_t bi = (std::size_t)n * row_bytes + (k >> 1);
          if (k & 1) { packed[bi] |= (std::uint8_t)(v << 4); }
          else       { packed[bi] |= (std::uint8_t)v; }
        }
        const int g = k / 64;
        deq[(std::size_t)n * K + k] =
            (float)scales[(std::size_t)n * G + g] * (float)v +
            (float)biases[(std::size_t)n * G + g];
      }
    }
    // f32 oracle: ref = x @ deq^T.
    std::vector<float> ref((std::size_t)M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
          acc += (float)x[(std::size_t)m * K + k] * deq[(std::size_t)n * K + k];
        }
        ref[(std::size_t)m * N + n] = acc;
      }
    }
    SharedBuffer wb  = mc->make_shared_buffer(packed.size());
    SharedBuffer sb  = mc->make_shared_buffer(scales.size() * 2);
    SharedBuffer bb  = mc->make_shared_buffer(biases.size() * 2);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)N * K * 2);
    SharedBuffer yb  = mc->make_shared_buffer((std::size_t)M * N * 2);
    std::memcpy(wb.contents(), packed.data(), packed.size());
    std::memcpy(sb.contents(), scales.data(), scales.size() * 2);
    std::memcpy(bb.contents(), biases.data(), biases.size() * 2);
    auto read = [&]() {
      std::vector<float> o((std::size_t)M * N);
      const auto* p = static_cast<const _Float16*>(yb.contents());
      for (std::size_t i = 0; i < o.size(); ++i) { o[i] = (float)p[i]; }
      return o;
    };
    auto rl2 = [&](const std::vector<float>& g) {
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < g.size(); ++i) {
        const double d = (double)g[i] - (double)ref[i];
        num += d * d; den += (double)ref[i] * (double)ref[i];
      }
      return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    };

    // (a) dequant -> dense matmul2d: the DiT M5 quant path, verbatim.
    std::memset(yb.contents(), 0, (std::size_t)M * N * 2);
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(f_dq);
        enc.set_buffer(0, wb); enc.set_buffer(1, sb); enc.set_buffer(2, bb);
        enc.set_buffer(3, wdq);
        enc.set_constant(4, K); enc.set_constant(5, N);
        const unsigned words = (bits == 8) ? (unsigned)(K / 4) : (unsigned)(K / 8);
        enc.dispatch({words, (unsigned)N, 1}, {64, 1, 1});
        enc.set_function(f_dense);
        enc.set_buffer(0, xb); enc.set_buffer(1, wdq); enc.set_buffer(2, wdq);
        enc.set_buffer(3, yb);
        enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
        enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((N + 127) / 128) * 256),
                      (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      }
      st.commit().wait();
    }
    const double e_mma = rl2(read());

    // (b) steel qmm: the DiT non-mma fallback (must agree with (a)).
    std::memset(yb.contents(), 0, (std::size_t)M * N * 2);
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(f_st);
        enc.set_buffer(0, wb); enc.set_buffer(1, sb); enc.set_buffer(2, bb);
        enc.set_buffer(3, xb); enc.set_buffer(4, yb);
        enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
        enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                      (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
      }
      st.commit().wait();
    }
    const double e_steel = rl2(read());
    std::printf("[dequant_dense] w%d g64 M=%d N=%d K=%d  mma rel-L2=%.4e  "
                "steel rel-L2=%.4e\n", bits, M, N, K, e_mma, e_steel);
    EXPECT_TRUE(e_mma < 3e-2);
    EXPECT_TRUE(e_steel < 3e-2);
  };
  run_bits(4, f_dq4, f_st4);
  run_bits(8, f_dq8, f_st8);
}

// 4-bit quantized throughput: steel vs matrix-core at projection shapes.
TEST(qmm_mma, throughput) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib_steel = mc->load_library("affine_qmm_steel_bf16");
  ComputeLibrary lib_mma   = mc->load_library("affine_qmm_mma_bf16");
  ComputeLibrary lib_dq    = mc->load_library("affine_dequant_bf16");
  ComputeLibrary lib_dense = mc->load_library("dense_gemm_mma_bf16");
  ComputeFunction f_steel = lib_steel.function("affine_qmm_steel_w4g64");
  ComputeFunction f_mma   = lib_mma.function("affine_qmm_mma_w4g64");
  ComputeFunction f_dq    = lib_dq.function("affine_dequant_w4g64");
  ComputeFunction f_dense = lib_dense.function("dense_gemm_mma_t_f16");
  ASSERT_TRUE(f_steel.valid());
  const bool have_mma = f_mma.valid();
  const bool have_dd  = f_dq.valid() && f_dense.valid();

  const Shape shapes[] = {
      {512,  4096, 2560, "qkv@512"},
      {1024, 4096, 2560, "qkv@1024"},
      {1024, 2560, 4096, "o@1024"},
      {512,  9728, 2560, "gate_up@512"},
  };
  for (const auto& sh : shapes) {
    const int M = sh.M, N = sh.N, K = sh.K;
    std::vector<std::uint16_t> x((std::size_t)M * K);
    fill_bf16(x, 31 + M);
    QWeight q = make_qweight(N, K, 42 + N);
    SharedBuffer wb = mc->make_shared_buffer(q.packed.size());
    SharedBuffer sb = mc->make_shared_buffer(q.scales.size() * 2);
    SharedBuffer bb = mc->make_shared_buffer(q.biases.size() * 2);
    SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * K * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)N * K * 2);
    std::memcpy(wb.contents(), q.packed.data(), q.packed.size());
    std::memcpy(sb.contents(), q.scales.data(), q.scales.size() * 2);
    std::memcpy(bb.contents(), q.biases.data(), q.biases.size() * 2);
    std::memcpy(xb.contents(), x.data(), x.size() * 2);

    const int ITERS = 40;
    const double gflop = 2.0 * M * N * K * ITERS / 1e9;
    auto bench = [&](ComputeFunction& fn, bool mma) {
      for (int w = 0; w < 3; ++w) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder enc = st.begin_compute();
          enc.set_function(fn);
          enc.set_buffer(0, wb); enc.set_buffer(1, sb); enc.set_buffer(2, bb);
          enc.set_buffer(3, xb); enc.set_buffer(4, yb);
          enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
          if (mma) {
            enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                          (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
          } else {
            enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                          (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
          }
        }
        st.commit().wait();
      }
      const auto t0 = std::chrono::steady_clock::now();
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < ITERS; ++i) {
          enc.set_function(fn);
          enc.set_buffer(0, wb); enc.set_buffer(1, sb); enc.set_buffer(2, bb);
          enc.set_buffer(3, xb); enc.set_buffer(4, yb);
          enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
          if (mma) {
            enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                          (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
          } else {
            enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                          (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
          }
        }
      }
      st.commit().wait();
      const auto t1 = std::chrono::steady_clock::now();
      return gflop / secs_(t0, t1);
    };
    // dequant-once + dense matmul2d: each iter dequants W -> wdq then runs
    // the dense matrix-core GEMM (the realistic per-projection cost when
    // the weight is re-expanded each call).
    auto bench_dd = [&]() {
      auto enc_dq = [&](ComputeEncoder& enc) {
        enc.set_function(f_dq);
        enc.set_buffer(0, wb); enc.set_buffer(1, sb); enc.set_buffer(2, bb);
        enc.set_buffer(3, wdq);
        enc.set_constant(4, K); enc.set_constant(5, N);
        enc.dispatch({(unsigned)(K / 8), (unsigned)N, 1}, {64, 1, 1});
      };
      auto enc_mm = [&](ComputeEncoder& enc) {
        enc.set_function(f_dense);
        enc.set_buffer(0, xb); enc.set_buffer(1, wdq); enc.set_buffer(2, bb);
        enc.set_buffer(3, yb);
        enc.set_constant(4, K); enc.set_constant(5, N);
        enc.set_constant(6, M); enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                      (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
      };
      for (int w = 0; w < 3; ++w) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder enc = st.begin_compute(); enc_dq(enc); enc_mm(enc); }
        st.commit().wait();
      }
      const auto t0 = std::chrono::steady_clock::now();
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < ITERS; ++i) { enc_dq(enc); enc_mm(enc); } }
      st.commit().wait();
      const auto t1 = std::chrono::steady_clock::now();
      return gflop / secs_(t0, t1);
    };

    const double g_steel = bench(f_steel, false);
    if (have_mma) {
      const double g_mma = bench(f_mma, true);
      const double g_dd = have_dd ? bench_dd() : 0.0;
      std::printf("[qmm_mma BENCH] %-12s M=%4d N=%4d K=%4d | steel %7.1f | "
                  "fused-mma %7.1f (%.2fx) | dq+dense %7.1f (%.2fx) GFLOP/s\n",
                  sh.tag, M, N, K, g_steel, g_mma, g_mma / g_steel,
                  g_dd, g_dd / g_steel);
    } else {
      std::printf("[qmm_mma BENCH] %-12s steel %7.1f GFLOP/s (no mma)\n",
                  sh.tag, g_steel);
    }
    EXPECT_TRUE(g_steel > 0.0);
  }
}

// Dequant kernel bandwidth in isolation. The dq+dense path's overhead vs
// pure dense GEMM is the dequant pass; this measures its achieved GB/s
// (writes 2*N*K bytes of bf16 weight) so we know how much headroom the
// dequant has against M5 DRAM bandwidth.
TEST(qmm_mma, dequant_bandwidth) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib_dq = mc->load_library("affine_dequant_bf16");
  ComputeFunction f_dq  = lib_dq.function("affine_dequant_w4g64");
  if (!f_dq.valid()) { std::printf("[dequant_bw] unavailable\n"); return; }

  const Shape shapes[] = {
      {0, 4096, 2560, "qkv"},
      {0, 2560, 4096, "o"},
      {0, 9728, 2560, "gate_up"},
      {0, 2560, 9728, "down"},
  };
  for (const auto& sh : shapes) {
    const int N = sh.N, K = sh.K;
    QWeight q = make_qweight(N, K, 42 + N);
    SharedBuffer wb = mc->make_shared_buffer(q.packed.size());
    SharedBuffer sb = mc->make_shared_buffer(q.scales.size() * 2);
    SharedBuffer bb = mc->make_shared_buffer(q.biases.size() * 2);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)N * K * 2);
    std::memcpy(wb.contents(), q.packed.data(), q.packed.size());
    std::memcpy(sb.contents(), q.scales.data(), q.scales.size() * 2);
    std::memcpy(bb.contents(), q.biases.data(), q.biases.size() * 2);

    const int ITERS = 60;
    const double gb = (double)N * K * 2 * ITERS / 1e9;  // bytes written
    auto enc_dq = [&](ComputeEncoder& enc) {
      enc.set_function(f_dq);
      enc.set_buffer(0, wb); enc.set_buffer(1, sb); enc.set_buffer(2, bb);
      enc.set_buffer(3, wdq);
      enc.set_constant(4, K); enc.set_constant(5, N);
      enc.dispatch({(unsigned)(K / 8), (unsigned)N, 1}, {64, 1, 1});
    };
    for (int w = 0; w < 3; ++w) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute(); enc_dq(enc); }
      st.commit().wait();
    }
    const auto t0 = std::chrono::steady_clock::now();
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder enc = st.begin_compute();
      for (int i = 0; i < ITERS; ++i) { enc_dq(enc); } }
    st.commit().wait();
    const auto t1 = std::chrono::steady_clock::now();
    const double bw = gb / secs_(t0, t1);
    std::printf("[dequant_bw] %-8s N=%4d K=%4d  write %.1f MB  %.1f GB/s\n",
                sh.tag, N, K, (double)N * K * 2 / 1e6, bw);
    EXPECT_TRUE(bw > 0.0);
  }
}

// Tile-size tuning for the matrix-core dense GEMM. Benches several
// (region, simdgroup) configs of dense_gemm_mma at the Qwen3.5-4B
// projection shapes + checks each against a CPU f32 oracle. The 64x64
// baseline is memory-bound on weight streaming at large K; bigger output
// regions raise arithmetic intensity. Picks the winner for the model path.
TEST(gemm_mma, tune) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("dense_gemm_mma_bf16");
  struct Variant {
    const char* fn; int region_m; int region_n; int tg; const char* tag;
    // relaxed_precision probe: accumulates at operand precision (bf16
    // here), so the oracle check only reports its drift instead of
    // failing on it -- the drift IS part of the measurement.
    bool relaxed = false;
  };
  const Variant vs[] = {
      {"dense_gemm_mma_t_n128_f16",    128, 128, 256, "128x128/8"},
      {"dense_gemm_mma_t_n128x256_f16",128, 256, 256, "128x256/8"},
      {"dense_gemm_mma_t_n128x256_tn2_f16",128, 512, 256, "128x256tn2"},
      {"dense_gemm_mma_t_n128_rp_f16",    128, 128, 256, "128rp", true},
      {"dense_gemm_mma_t_n128x256_rp_f16",128, 256, 256, "256rp", true},
  };
  constexpr int NV = 5;
  ComputeFunction fns[NV];
  for (int i = 0; i < NV; ++i) { fns[i] = lib.function(vs[i].fn); }
  if (!fns[0].valid()) {
    std::printf("[gemm_tune] no matrix cores -- skipped.\n");
    return;
  }

  const Shape shapes[] = {
      {1711, 4096, 2560, "qkv@1711"},
      {1711, 2560, 9728, "down@1711"},
      // Krea2 DiT block GEMMs at 1024px (joint seq 4106): K=6144 projections +
      // the ff-down split chunk (K=8192) and the full ff-down (K=16384).
      {4106, 6144, 6144, "k2-o@4106"},
      {4106, 16384, 6144, "k2-ffup@4106"},
      {4106, 6144, 16384, "k2-ffdn@4106"},
      {4106, 6144, 8192,  "k2-ffdn-k8192"},
      // Unaligned M and N (N not a multiple of the tn2 512-region) -- exercises
      // the matmul2d tensor-extent tail clamp for every swept tile.
      {1700, 2600, 4100, "tail-a"},
  };
  for (const auto& sh : shapes) {
    const int M = sh.M, N = sh.N, K = sh.K;
    std::vector<std::uint16_t> x((std::size_t)M * K), W((std::size_t)N * K);
    fill_bf16(x, 11 + M);
    fill_bf16(W, 22 + N);
    SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * K * 2);
    SharedBuffer wb = mc->make_shared_buffer((std::size_t)N * K * 2);
    SharedBuffer bb = mc->make_shared_buffer((std::size_t)N * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    std::memcpy(xb.contents(), x.data(), x.size() * 2);
    std::memcpy(wb.contents(), W.data(), W.size() * 2);
    std::memset(bb.contents(), 0, (std::size_t)N * 2);

    // CPU oracle on a sparse sample of (m,n) for the correctness check.
    auto oracle = [&](int m, int n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += bf16_to_f32(x[(std::size_t)m * K + k]) *
               bf16_to_f32(W[(std::size_t)n * K + k]);
      }
      return acc;
    };

    const int ITERS = 40;
    const double gflop = 2.0 * M * N * K * ITERS / 1e9;
    std::printf("[gemm_tune] %-12s M=%4d N=%4d K=%4d |", sh.tag, M, N, K);
    for (int i = 0; i < NV; ++i) {
      if (!fns[i].valid()) { std::printf(" %s n/a |", vs[i].tag); continue; }
      const unsigned gx =
          (unsigned)(((N + vs[i].region_n - 1) / vs[i].region_n) * vs[i].tg);
      const unsigned gy =
          (unsigned)((M + vs[i].region_m - 1) / vs[i].region_m);
      auto launch_n = [&](int count) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder enc = st.begin_compute();
          for (int c = 0; c < count; ++c) {
            enc.set_function(fns[i]);
            enc.set_buffer(0, xb); enc.set_buffer(1, wb);
            enc.set_buffer(2, bb); enc.set_buffer(3, yb);
            enc.set_constant(4, K); enc.set_constant(5, N);
            enc.set_constant(6, M); enc.set_constant(7, 0);
            enc.dispatch({gx, gy, 1}, {(unsigned)vs[i].tg, 1, 1});
          }
        }
        st.commit().wait();
      };
      std::memset(yb.contents(), 0, (std::size_t)M * N * 2);
      launch_n(1);
      // Correctness sample.
      const auto* yp = static_cast<const std::uint16_t*>(yb.contents());
      double num = 0.0, den = 0.0;
      const int ms[] = {0, M / 2, M - 1};
      const int ns[] = {0, N / 2, N - 1};
      for (int mm : ms) {
        for (int nn : ns) {
          const float ref = oracle(mm, nn);
          const float got = bf16_to_f32(yp[(std::size_t)mm * N + nn]);
          num += (got - ref) * (got - ref); den += ref * ref;
        }
      }
      const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
      launch_n(3);
      const auto t0 = std::chrono::steady_clock::now();
      launch_n(ITERS);
      const auto t1 = std::chrono::steady_clock::now();
      const double g = gflop / secs_(t0, t1);
      if (vs[i].relaxed) {
        // Report the relaxed-accumulate drift; don't gate on it.
        std::printf(" %s %.0f(r%.1g) |", vs[i].tag, g, rel);
      } else {
        std::printf(" %s %.0f%s |", vs[i].tag, g, rel < 3e-2 ? "" : "(BAD)");
        EXPECT_TRUE(rel < 3e-2);
      }
    }
    std::printf("\n");
  }
}

// Throughput: steel vs matrix-core GFLOP/s at Qwen3.5-4B projection
// shapes. Always prints; the EXPECT only checks the kernels ran.
TEST(gemm_mma, throughput) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }

  ComputeLibrary lib_steel = mc->load_library("dense_gemm_bf16");
  ComputeLibrary lib_mma   = mc->load_library("dense_gemm_mma_bf16");
  ComputeFunction f_steel = lib_steel.function("dense_gemm_t_f16");
  ComputeFunction f_mma   = lib_mma.function("dense_gemm_mma_t_f16");
  ASSERT_TRUE(f_steel.valid());
  const bool have_mma = f_mma.valid();

  // (M, N, K): prefill rows x projection shapes for Qwen3.5-4B
  // (hidden=2560, qkv fused widths, o_proj, mlp).
  const Shape shapes[] = {
      {512,  4096, 2560, "qkv@512"},
      {1024, 4096, 2560, "qkv@1024"},
      {1024, 2560, 4096, "o@1024"},
      {1024, 2560, 2560, "sq@1024"},
      {512,  9728, 2560, "gate_up@512"},   // ffn fused (example)
  };

  for (const auto& sh : shapes) {
    const int M = sh.M, N = sh.N, K = sh.K;
    std::vector<std::uint16_t> x((std::size_t)M * K), W((std::size_t)N * K);
    fill_bf16(x, 11 + M);
    fill_bf16(W, 22 + N);
    SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * K * 2);
    SharedBuffer wb = mc->make_shared_buffer((std::size_t)N * K * 2);
    SharedBuffer bb = mc->make_shared_buffer((std::size_t)N * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    std::memcpy(xb.contents(), x.data(), x.size() * 2);
    std::memcpy(wb.contents(), W.data(), W.size() * 2);
    std::memset(bb.contents(), 0, (std::size_t)N * 2);

    const int ITERS = 40;
    const double gflop = 2.0 * M * N * K * ITERS / 1e9;

    auto bench = [&](ComputeFunction& fn, bool mma) {
      // Warmup.
      for (int w = 0; w < 3; ++w) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder enc = st.begin_compute();
          enc.set_function(fn);
          enc.set_buffer(0, xb); enc.set_buffer(1, wb);
          enc.set_buffer(2, bb); enc.set_buffer(3, yb);
          enc.set_constant(4, K); enc.set_constant(5, N);
          enc.set_constant(6, M); enc.set_constant(7, 0);
          if (mma) {
            enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                          (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
          } else {
            enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                          (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
          }
        }
        st.commit().wait();
      }
      const auto t0 = std::chrono::steady_clock::now();
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < ITERS; ++i) {
          enc.set_function(fn);
          enc.set_buffer(0, xb); enc.set_buffer(1, wb);
          enc.set_buffer(2, bb); enc.set_buffer(3, yb);
          enc.set_constant(4, K); enc.set_constant(5, N);
          enc.set_constant(6, M); enc.set_constant(7, 0);
          if (mma) {
            enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                          (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
          } else {
            enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                          (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
          }
        }
      }
      st.commit().wait();
      const auto t1 = std::chrono::steady_clock::now();
      return gflop / secs_(t0, t1);
    };

    const double g_steel = bench(f_steel, false);
    if (have_mma) {
      const double g_mma = bench(f_mma, true);
      std::printf("[gemm_mma BENCH] %-12s M=%4d N=%4d K=%4d | steel %7.1f "
                  "GFLOP/s | mma %7.1f GFLOP/s | %.2fx\n",
                  sh.tag, M, N, K, g_steel, g_mma, g_mma / g_steel);
    } else {
      std::printf("[gemm_mma BENCH] %-12s M=%4d N=%4d K=%4d | steel %7.1f "
                  "GFLOP/s | mma  (no matrix cores)\n",
                  sh.tag, M, N, K, g_steel);
    }
    EXPECT_TRUE(g_steel > 0.0);
  }
}

// De-risk for the materialized-decode steel lever (§57-58): the decode QK/PV are
// GEMMs with a TINY N=G=4 (the GQA group). steel BlockMMA tiles N in BN=32, so
// N=4 wastes 28/32 of the N tile. This measures dense_gemm_t_f16's achieved DRAM
// bandwidth (x = the big K/V operand, read once) at the decode shapes vs an
// N=32 aligned baseline -- if N=4 drops far below peak, steel won't beat the
// hand-rolled GEMV on M4. QK: scores^T[T,G]=K[T,D]@Q[G,D]^T (M=T,N=G,K=D). PV:
// out^T[D,G]=V^T[D,T]@w[G,T]^T (M=D,N=G,K=T). Gated VPIPE_GEMM_DECODE_BENCH.
TEST(gemm_mma, decode_shapes) {
  if (std::getenv("VPIPE_GEMM_DECODE_BENCH") == nullptr) { return; }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeFunction f = mc->load_library("dense_gemm_bf16")
                          .function("dense_gemm_t_f16");
  ASSERT_TRUE(f.valid());
  struct DS { int M, N, K; const char* tag; };
  const DS shapes[] = {
      {16384, 4,  512,   "QK T16k N=G4"},
      {16384, 32, 512,   "QK T16k N=32 "},   // aligned baseline
      {512,   4,  16384, "PV D512 N=G4"},
      {512,   32, 16384, "PV D512 N=32 "},   // aligned baseline
  };
  const double peak_gbs = 273.0;             // M4 Pro LPDDR5X
  for (const auto& sh : shapes) {
    const int M = sh.M, N = sh.N, K = sh.K;
    std::vector<std::uint16_t> x((std::size_t)M * K), W((std::size_t)N * K);
    fill_bf16(x, 11 + M + K); fill_bf16(W, 22 + N + K);
    SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * K * 2);
    SharedBuffer wb = mc->make_shared_buffer((std::size_t)N * K * 2);
    SharedBuffer bb = mc->make_shared_buffer((std::size_t)N * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    std::memcpy(xb.contents(), x.data(), x.size() * 2);
    std::memcpy(wb.contents(), W.data(), W.size() * 2);
    std::memset(bb.contents(), 0, (std::size_t)N * 2);
    const int ITERS = 60;
    auto run = [&](int count) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int c = 0; c < count; ++c) {
          enc.set_function(f);
          enc.set_buffer(0, xb); enc.set_buffer(1, wb);
          enc.set_buffer(2, bb); enc.set_buffer(3, yb);
          enc.set_constant(4, K); enc.set_constant(5, N);
          enc.set_constant(6, M); enc.set_constant(7, 0);
          enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                        (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
        }
      }
      st.commit().wait();
    };
    run(3);
    const auto t0 = std::chrono::steady_clock::now();
    run(ITERS);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = secs_(t0, t1) * 1e3 / ITERS;
    const double gbs = (double)M * K * 2 * ITERS / secs_(t0, t1) / 1e9;
    std::printf("[gemm_decode] %-14s M=%5d N=%2d K=%5d | %.3f ms/call | "
                "%6.1f GB/s (%.0f%% of %.0f peak)\n",
                sh.tag, M, N, K, ms, gbs, 100.0 * gbs / peak_gbs, peak_gbs);
    EXPECT_TRUE(gbs > 0.0);
  }
}

// Raw ALU-rate probes (opt-in, VPIPE_MMA_RATE_BENCH): register-resident
// simdgroup-MMA and scalar-FMA loops, f32 fragments vs f16 fragments, no
// memory traffic in the timed loop. Answers "does this GPU have a
// double-rate FP16 pipe?" -- the premise behind the _acc16 half-accumulate
// GEMM variants. Prints TFLOP/s for all four probes.
TEST(gemm_mma, alu_rate_f16_vs_f32)
{
  if (std::getenv("VPIPE_MMA_RATE_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("dense_gemm");
  ASSERT_TRUE(lib.valid());

  const int SG = 4;                      // simdgroups per threadgroup
  const unsigned tgsz = 32 * SG;
  const unsigned tgs = 4096;             // threadgroups
  const int mma_iters = 50000;           // ~0.5s per rep at expected rates
  const int fma_iters = 200000;

  auto run = [&](const char* name, int iters, double total_flops) {
    ComputeFunction fn = lib.function(name);
    if (!fn.valid()) {
      std::printf("[alu_rate] %-16s MISSING\n", name);
      return;
    }
    SharedBuffer out = mc->make_shared_buffer(64);
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {   // rep 0 warms up
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, out);
        enc.set_constant(1, iters);
        enc.dispatch({tgsz * tgs, 1, 1}, {tgsz, 1, 1});
      }
      const auto t0 = std::chrono::steady_clock::now();
      st.commit().wait();
      const auto t1 = std::chrono::steady_clock::now();
      const double s = secs_(t0, t1);
      if (rep > 0 && s < best) { best = s; }
    }
    std::printf("[alu_rate] %-16s %7.2f TFLOP/s\n", name,
                total_flops / best / 1e12);
  };

  // simdgroup MMA: per simdgroup per iter = 4 accum * 2*8*8*8 flops.
  const double mma_flops =
      (double)tgs * SG * mma_iters * 4.0 * 1024.0;
  // scalar FMA: per thread per iter = 4 chains * 4 lanes * 2 flops.
  const double fma_flops =
      (double)tgs * tgsz * fma_iters * 4.0 * 4.0 * 2.0;
  run("dg_mma_rate_acc32", mma_iters, mma_flops);
  run("dg_mma_rate_acc16", mma_iters, mma_flops);
  run("dg_mma_rate_accbf", mma_iters, mma_flops);
  run("dg_mma_rate_mixed", mma_iters, mma_flops);   // 2 f32 + 2 f16 / iter
  run("dg_fma_rate_f32", fma_iters, fma_flops);
  run("dg_fma_rate_f16", fma_iters, fma_flops);
  // v2 high-ILP scalar probes: 8 vec4 chains, 4x unrolled -> 256 flops per
  // thread-iter (the v1 4-chain probes were ILP-bound ~30% below peak).
  const int fma2_iters = 50000;
  const double fma2_flops =
      (double)tgs * tgsz * fma2_iters * 8.0 * 4.0 * 2.0 * 4.0;
  run("dg_fma_rate2_f32", fma2_iters, fma2_flops);
  run("dg_fma_rate2_f16", fma2_iters, fma2_flops);
  run("dg_fma_rate2_bf16", fma2_iters, fma2_flops);
  run("dg_fma_rate2_mixed", fma2_iters, fma2_flops);
  // 8+8 mixed: 16 chains * 4 lanes * 2 flops * 2 unroll per thread-iter.
  const double fma88_flops =
      (double)tgs * tgsz * fma2_iters * 16.0 * 4.0 * 2.0 * 2.0;
  run("dg_fma_rate2_mixed88", fma2_iters, fma88_flops);
  // MMA + scalar-f16 co-issue: per SIMDGROUP per iter = 4*1024 (mma) +
  // 32 threads * 4 chains * 4 lanes * 2 (scalar f16).
  const double mmamix_flops =
      (double)tgs * SG * mma_iters * (4.0 * 1024.0 + 32.0 * 4.0 * 4.0 * 2.0);
  run("dg_mma_scalar_mix", mma_iters, mmamix_flops);
  // Integer mad probes (int8/int16/int32 + int8|f16 co-issue): same op
  // accounting as fma2 (mul-add = 2 ops); the printed TFLOP/s is TOPS here.
  run("dg_imad_rate_i8", fma2_iters, fma2_flops);
  run("dg_imad_rate_i16", fma2_iters, fma2_flops);
  run("dg_imad_rate_i32", fma2_iters, fma2_flops);
  run("dg_imad_rate_mix_i8f16", fma2_iters, fma2_flops);
  EXPECT_TRUE(true);
}

// THE 2^31-BYTE OPERAND BAND IS DEFENSIVE, AND MUST STAY THAT WAY.
//
// mma_row_band caps how many rows one matmul2d dispatch may cover so that
// no operand's byte span passes 2^31, where the tiles silently stop
// storing. Two families were already wrong here -- MiniMax-H3's fc2 (band
// taken from N, source unguarded) and the Wan VAE (split by a row cap, its
// 27-tap im2col source unguarded) -- so the rule now guards every caller.
//
// Guarding every caller creates a NEW failure mode, and a quiet one. There
// are two kinds of caller and the band means different things to each:
//
//   BANDING callers (MiniMax-H3, both VAEs) loop over row bands. A band
//   that fires costs an extra dispatch and nothing else -- H3's qkv has
//   banded at 49920 since long before any of this, correctly.
//
//   DECLINING callers (the four image DiTs) return false and let the
//   caller keep its int64-safe steel path. A band that fires there drops
//   the model off matmul2d entirely, at 2-3x. Every existing test still
//   passes, because steel computes the right answer.
//
// So the property is asymmetric, and this pins the half that can go quiet:
// at the shapes the image DiTs actually run, the guard must NOT fire.
TEST(gemm_mma, row_band_does_not_fire_at_shipped_shapes)
{
  using vpipe::genai::mma_row_band;

  // The DECLINING callers, at the largest geometry they are run at today.
  // One token is 16x16 px, so 1024px is 4096 tokens plus text.
  struct Shape { const char* tag; int M, N, K; };
  const Shape decliners[] = {
      {"krea2 qkv",         4106,  9216,  3072},
      {"krea2 ff-up",       4106, 24576,  3072},
      {"krea2 ff-down",     4106,  3072, 16384},
      {"flux2 ff-down",     4106,  3072, 16384},
      {"qie ff-down",       4106,  3072, 16384},
      {"boogu ff-down",     4106,  3072,  6784},
      {"krea2 ff-down 2k", 16384,  3072, 16384},
      {"krea2 ff-up 2k",   16384, 24576,  3072},
  };
  bool clean = true;
  for (const Shape& s : decliners) {
    const int b = mma_row_band(s.N, s.K);
    if (s.M > b) {
      std::printf("[gemm_mma] %-18s M=%7d N=%6d K=%6d  band %9d  FIRES -- "
                  "this shape just lost matmul2d\n",
                  s.tag, s.M, s.N, s.K, b);
      clean = false;
    }
    EXPECT_TRUE(s.M <= b);
  }

  // The BANDING callers: firing is legal, but the 2D VAE convs should not
  // start splitting where the 2^19 row cap alone already served them --
  // that would be a silent extra dispatch per conv at every resolution.
  const Shape vae[] = {
      {"vae top 3x3",  262144,   128,  1152},
      {"vae mid 3x3",  262144,   256,  2304},
      {"vae deep 3x3",  16384,   512,  4608},
      {"wan 480p 96ch",262144,    96,  2592},
  };
  for (const Shape& s : vae) {
    const int cap = (1 << 19);
    const int was = s.M < cap ? s.M : cap;
    const int now = was < mma_row_band(s.N, s.K)
                        ? was : mma_row_band(s.N, s.K);
    if (was != now) {
      std::printf("[gemm_mma] %-18s chunk %d -> %d\n", s.tag, was, now);
      clean = false;
    }
    EXPECT_TRUE(was == now);
  }

  // POSITIVE CONTROLS. The two shapes that did come back wrong must still
  // be caught, or the rule has been weakened into a no-op.
  //   H3 fc2 at 1376x768x243: source 75136 x 14336 bf16 = 2.15 GB.
  //   Wan VAE 720p, 192-channel level: 230400 x 5184 = 2.22 GiB.
  EXPECT_TRUE(75136 > mma_row_band(5376, 14336));
  EXPECT_TRUE(230400 > mma_row_band(192, 5184));

  // And whatever a band is, it must keep the widest operand under the line
  // -- the property the whole rule exists to provide.
  const Shape all[] = {
      {"h3 fc2",  75136,  5376, 14336}, {"h3 qkv", 75136, 21504, 5376},
      {"wan 720p",230400,  192,  5184}, {"krea2 ff-down", 4106, 3072, 16384},
  };
  for (const Shape& s : all) {
    const long long b = mma_row_band(s.N, s.K);
    const long long w = s.N > s.K ? s.N : s.K;
    EXPECT_TRUE(b * w * 2 < (1LL << 31));
  }
  std::printf("[gemm_mma] row band: %s\n",
              clean ? "inactive at every shipped shape, controls still caught"
                    : "FIRES WHERE IT SHOULD NOT");
}

// THE INT8 FRAGMENT MMA, at attention's own shape.
//
// attn_steel_nax does not issue a 128-deep matmul2d per key block: it
// builds that product out of EIGHT 16-deep fragment MMAs
// (NAXFrag_t::mma, a matmul2d<16,32,16>) chained into the same register
// fragments. gemm_i8.k512_chunked answers the question at 128-deep
// chunks (int8 20.8 TFLOP/s against f16 11.0); this answers it at the
// depth the kernel actually runs, which is the one that decides whether
// an int8 QK is worth wiring in.
//
// Operands stay in registers for the whole loop and there is one store
// at the end, so what is timed is the matrix pipe and not the loads --
// the same thing the flash kernel's inner loop is bounded by.
TEST(attn_qk_i8, frag_rate)
{
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[qk_i8] no matrix cores -- SKIPPED\n");
    return;
  }
  ComputeLibrary lib = mc->load_library("attn_qk_i8_probe");
  ComputeFunction f_f16 = lib.function("qk_frag_rate_f16");
  ComputeFunction f_i8 = lib.function("qk_frag_rate_i8");
  if (!f_f16.valid() || !f_i8.valid()) {
    std::printf("[qk_i8] probe kernels unavailable -- skip\n");
    return;
  }
  constexpr int kSG = 4;          // the kernel's wm
  constexpr int kTG = 4096;       // threadgroups, to fill the machine
  constexpr int kIters = 512;     // key blocks walked per threadgroup
  SharedBuffer ob = mc->make_shared_buffer((std::size_t)kTG * 4);
  ASSERT_TRUE(!ob.empty());

  auto run = [&](ComputeFunction& fn, int reps) {
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder enc = st.begin_compute();
      for (int i = 0; i < reps; ++i) {
        enc.set_function(fn);
        enc.set_buffer(0, ob);
        enc.set_constant(1, kIters);
        enc.dispatch({(unsigned)(32 * kSG * kTG), 1, 1},
                     {(unsigned)(32 * kSG), 1, 1});
      }
    }
    st.commit().wait();
  };

  // 8 fragment MMAs of 16x32x16 per iteration per simdgroup.
  const double flop_per_rep = 2.0 * 8.0 * 16.0 * 32.0 * 16.0 *
                              (double)kIters * kSG * kTG;
  const int reps = 20;
  run(f_f16, 2); run(f_i8, 2);                       // warm
  auto t0 = std::chrono::steady_clock::now();
  run(f_f16, reps);
  auto t1 = std::chrono::steady_clock::now();
  run(f_i8, reps);
  auto t2 = std::chrono::steady_clock::now();
  const double s_f16 = secs_(t0, t1), s_i8 = secs_(t1, t2);
  const double g_f16 = flop_per_rep * reps / s_f16 / 1e9;
  const double g_i8 = flop_per_rep * reps / s_i8 / 1e9;
  std::printf("[qk_i8] 16-deep fragment chain, TD=8, %d tg x %d sg: f16 "
              "%.0f GF/s  int8 %.0f GF/s  i8/f16 %.2fx\n", kTG, kSG,
              g_f16, g_i8, s_f16 / s_i8);
  // The claim under test. A ratio near 1 would say the fragment MMA does
  // not carry int8's advantage down to 16-deep, and an int8 QK would be
  // pointless however accurate it is.
  EXPECT_TRUE(g_f16 > 0.0 && g_i8 > 0.0);
}

// INT8 AND SPLIT-K AT THE SAME TIME, which is the combination neither
// shipped path offers and the one H3's fc2 wants.
//
// The two accelerations answer different deficits. int8 doubles the
// matrix pipe's rate; split-K fixes an OCCUPANCY problem -- a single-op
// reduction over a deep K leaves only (M/64)*(N/64) threadgroups each
// walking one long serial contraction, and the units stall with nothing
// else in flight. Neither cures the other, so a deep-K int8 GEMM has
// both problems and today can only be given one treatment: H3's helper
// tries i8 first and split-K second, so on fc2 (K = ffn = 14336, which
// is 28 whole 512-groups) the int8 arm wins the race and the split never
// runs. The prototype record already flagged this -- ff-down at K=12288
// measured 18.5 TOPS against 22.5-23.6 at K=4096, the same cliff the f16
// path has.
//
// gemm_i8i8_sc_f16_n64_g512_sk is _g512 cut across grid.z planes, left in
// f32 for the existing splitk_fold_f32_f16 to sum. The split is by GROUP
// INDEX, so every plane boundary lands on a 512 multiple for free.
TEST(gemm_i8, splitk) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib_mma = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_dq = mc->load_library("affine_dequant");
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise");
  ComputeFunction f_g5 = lib_mma.function("gemm_i8i8_sc_f16_n64_g512");
  ComputeFunction f_sk = lib_mma.function("gemm_i8i8_sc_f16_n64_g512_sk");
  ComputeFunction f_qg = lib_dq.function("quant_f16_i8_row_g512");
  ComputeFunction f_fold = lib_elt.function("splitk_fold_f32_f16");
  if (!f_g5.valid() || !f_sk.valid() || !f_qg.valid() || !f_fold.valid()) {
    std::printf("[gemm_i8sk] kernels unavailable -- skip\n");
    return;
  }

  auto run = [&](int M, int N, int K, const std::vector<int>& splits,
                 bool check, int iters) {
    const int G = K / 512;
    std::mt19937 rng(77u + (unsigned)(M + N + K));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<_Float16> x((std::size_t)M * K), w((std::size_t)N * K);
    for (auto& v : x) { v = (_Float16)(nd(rng) * 0.5f); }
    for (auto& v : w) { v = (_Float16)(nd(rng) * 0.05f); }
    std::vector<std::int8_t> wqg((std::size_t)N * K);
    std::vector<_Float16> wsg((std::size_t)N * G);
    for (int n = 0; n < N; ++n) {
      for (int g = 0; g < G; ++g) {
        float ag = 0.0f;
        for (int k = g * 512; k < (g + 1) * 512; ++k) {
          ag = std::max(ag, std::fabs((float)w[(std::size_t)n * K + k]));
        }
        const float ig = ag > 0 ? 127.0f / ag : 0.0f;
        wsg[(std::size_t)n * G + g] = (_Float16)(ag / 127.0f);
        for (int k = g * 512; k < (g + 1) * 512; ++k) {
          const float qv = std::rint((float)w[(std::size_t)n * K + k] * ig);
          wqg[(std::size_t)n * K + k] =
              (std::int8_t)std::max(-127.0f, std::min(127.0f, qv));
        }
      }
    }
    SharedBuffer xb = mc->make_shared_buffer(x.size() * 2);
    SharedBuffer xqb = mc->make_shared_buffer((std::size_t)M * K);
    SharedBuffer asgb = mc->make_shared_buffer((std::size_t)M * G * 2);
    SharedBuffer wqgb = mc->make_shared_buffer(wqg.size());
    SharedBuffer wsgb = mc->make_shared_buffer((std::size_t)N * G * 2);
    SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
    SharedBuffer yb2 = mc->make_shared_buffer((std::size_t)M * N * 2);
    int smax = 1;
    for (int s : splits) { smax = std::max(smax, s); }
    SharedBuffer pl =
        mc->make_shared_buffer((std::size_t)smax * M * N * 4);
    if (yb.empty() || yb2.empty() || pl.empty() || wqgb.empty()) {
      std::printf("[gemm_i8sk] alloc failed %dx%dx%d -- skip\n", M, N, K);
      return;
    }
    std::memcpy(xb.contents(), x.data(), x.size() * 2);
    std::memcpy(wqgb.contents(), wqg.data(), wqg.size());
    std::memcpy(wsgb.contents(), wsg.data(), (std::size_t)N * G * 2);

    auto quant = [&](ComputeEncoder& enc) {
      enc.set_function(f_qg);
      enc.set_buffer(0, xb); enc.set_buffer(1, xqb); enc.set_buffer(2, asgb);
      enc.set_constant(3, K);
      enc.dispatch({256, (unsigned)M, 1}, {256, 1, 1});
    };
    // S == 1 is the single-op kernel, so the sweep includes the baseline
    // in the SAME interleaved loop rather than timing it separately --
    // this box throttles enough that two loops can invert a ratio.
    auto one = [&](ComputeEncoder& enc, int S) {
      quant(enc);
      if (S <= 1) {
        enc.set_function(f_g5);
        enc.set_buffer(0, xqb); enc.set_buffer(1, wqgb);
        enc.set_buffer(2, asgb); enc.set_buffer(3, wsgb);
        enc.set_buffer(4, yb);
        enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
        enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                      (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
        return;
      }
      const int gpp = (G + S - 1) / S;
      enc.set_function(f_sk);
      enc.set_buffer(0, xqb); enc.set_buffer(1, wqgb);
      enc.set_buffer(2, asgb); enc.set_buffer(3, wsgb);
      enc.set_buffer(4, pl);
      enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
      enc.set_constant(8, gpp);
      enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                    (unsigned)((M + 63) / 64), (unsigned)S}, {128, 1, 1});
      enc.set_function(f_fold);
      enc.set_buffer(0, pl); enc.set_buffer(1, yb2);
      const int n_el = M * N;
      enc.set_constant(2, n_el); enc.set_constant(3, S);
      enc.dispatch({(unsigned)n_el, 1, 1}, {256, 1, 1});
    };

    if (check) {
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute(); one(e, 1); }
        st.commit().wait(); }
      for (int S : splits) {
        if (S <= 1) { continue; }
        { CommandStream st = mc->make_command_stream();
          { ComputeEncoder e = st.begin_compute(); one(e, S); }
          st.commit().wait(); }
        const auto* a = static_cast<const _Float16*>(yb.contents());
        const auto* b = static_cast<const _Float16*>(yb2.contents());
        std::size_t diff = 0;
        double worst = 0.0;
        for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
          const double d = std::fabs((double)a[i] - (double)b[i]);
          if (d != 0.0) { ++diff; }
          const double r = std::fabs((double)a[i]) > 0
                               ? d / std::fabs((double)a[i]) : d;
          worst = std::max(worst, r);
        }
        std::printf("[gemm_i8sk] S=%-2d vs single-op: %5.2f%% differ, worst "
                    "rel %.2e\n", S,
                    100.0 * (double)diff / (double)(M * N), worst);
        // A REASSOCIATION, not an approximation: the per-group summands
        // are bit-identical whichever plane owns them, so the only
        // difference is the order of G float adds and one f16 rounding.
        EXPECT_TRUE(worst < 5e-3);
      }
      return;
    }

    const double gflop = 2.0 * (double)M * N * K * 1e-9;
    std::printf("[gemm_i8sk] M=%d N=%d K=%d (G=%d)\n", M, N, K, G);
    double base = 0.0;
    for (int S : splits) {
      // Warm, then time -- and the arms run in one loop so a clock that
      // drifts hits them all.
      for (int w2 = 0; w2 < 2; ++w2) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute(); one(e, S); }
        st.commit().wait();
      }
      const auto t0 = std::chrono::steady_clock::now();
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          for (int i = 0; i < iters; ++i) { one(e, S); } }
        st.commit().wait(); }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      const double tops = gflop * iters / ms;
      if (S <= 1) { base = tops; }
      std::printf("[gemm_i8sk]   S=%-2d %7.2f ms  %6.2f TOP/s  %.2fx\n", S,
                  ms / iters, tops, base > 0 ? tops / base : 1.0);
    }
  };

  // Correctness small, where the CPU-free GPU-to-GPU comparison is the
  // right one: split-K is a reassociation of the SAME kernel.
  run(128, 256, 4096, {1, 2, 4, 8}, /*check=*/true, 1);
  // H3's fc2: K = ffn = 14336, N = inner = 7168. M is one row block of
  // what the split-K budget admits at video geometry (~2670 rows), so
  // this is one dispatch round of the real thing.
  //
  // TWO ROW COUNTS, because the deficit split-K fixes is an OCCUPANCY
  // one and M is half of what sets it: the base grid is (N/64)*(M/64)
  // threadgroups, so a wide M already fills the machine and has less for
  // a split to recover. 2560 is about the row block the 512 MB plane
  // budget admits at video geometry; 256 is a short prefill.
  run(256, 7168, 14336, {1, 2, 4, 7}, /*check=*/false, 4);
  run(1024, 7168, 14336, {1, 2, 4, 7, 14}, /*check=*/false, 4);
  run(2560, 7168, 14336, {1, 2, 4}, /*check=*/false, 3);
}

// THE SPLIT THROUGH I8GemmContext, which is what a model actually calls.
//
// gemm_i8.splitk above drives the kernels by hand; this drives the class,
// including the parts the kernels know nothing about -- the K padding, the
// per-call weight requant, the row blocking against the plane budget, and
// the plan. A split that is right in the kernel and wrong in the row bands
// produces a correct first block and garbage after it, which is exactly
// what a single-shape test would miss, so the row budget is squeezed here
// until the band count is forced above one.
TEST(gemm_i8, context_split_matches_single_op) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }

  const int M = 512, N = 1024, K = 4096;
  std::mt19937 rng(4242u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<_Float16> x((std::size_t)M * K), w((std::size_t)N * K);
  for (auto& v : x) { v = (_Float16)(nd(rng) * 0.5f); }
  for (auto& v : w) { v = (_Float16)(nd(rng) * 0.05f); }
  SharedBuffer xb = mc->make_shared_buffer(x.size() * 2);
  SharedBuffer wb = mc->make_shared_buffer(w.size() * 2);
  SharedBuffer y0 = mc->make_shared_buffer((std::size_t)M * N * 2);
  SharedBuffer y1 = mc->make_shared_buffer((std::size_t)M * N * 2);
  if (y1.empty()) { return; }
  std::memcpy(xb.contents(), x.data(), x.size() * 2);
  std::memcpy(wb.contents(), w.data(), w.size() * 2);

  // min_m below M so the shape qualifies at this size.
  ::setenv("VPIPE_I8_GEMM_MIN_M", "64", 1);
  struct Unset {
    ~Unset() {
      ::unsetenv("VPIPE_I8_GEMM_MIN_M");
      ::unsetenv("VPIPE_I8_SPLITK_MAX_MB");
      ::unsetenv("VPIPE_I8_GROUP");
    }
  } unset;

  // BOTH GROUPS: 64 is what ships, 512 is the drained pair kept for the
  // A/B, and each has its own split-K twin to get right.
  for (int grp : {32, 64, 128, 512}) {
  ::setenv("VPIPE_I8_GROUP", std::to_string(grp).c_str(), 1);
  ::unsetenv("VPIPE_I8_SPLITK_MAX_MB");
  std::printf("[gemm_i8ctx] ---- group %d ----\n", grp);

  auto run = [&](vpipe::genai::I8GemmContext& ctx, SharedBuffer& dst) {
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder e = st.begin_compute();
      EXPECT_TRUE(ctx.gemm(e, xb, 0, wb, dst, 0, M, N, K)); }
    st.commit().wait();
  };

  vpipe::genai::I8GemmContext base(mc, /*want=*/true, /*bf16=*/false);
  if (!base.enabled()) { return; }
  EXPECT_TRUE(base.group() == grp);
  base.bypass_split = true;                 // the single-op reference
  run(base, y0);

  // Every balanced split this shape admits (G = 8 at 512, 64 at 64),
  // each forced, and each at a plane budget that makes the row blocking
  // do real work: 8 MB of planes at S=8 is 256 rows, so M=512 becomes
  // two bands.
  ::setenv("VPIPE_I8_SPLITK_MAX_MB", "8", 1);
  vpipe::genai::I8GemmContext ctx(mc, /*want=*/true, /*bf16=*/false);
  // A vacuous pass is what hid a deleted split kernel once: the split
  // twins are part of the contract now, not optional here.
  EXPECT_TRUE(ctx.split_available());
  if (!ctx.split_available()) {
    std::printf("[gemm_i8ctx] split kernels unavailable\n");
    return;
  }
  const std::vector<int> cands = ctx.split_candidates(M, N, K);
  EXPECT_TRUE(!cands.empty());
  const auto* a = static_cast<const _Float16*>(y0.contents());
  for (int S : cands) {
    ctx.force_splits = S;
    run(ctx, y1);
    ctx.force_splits = 0;
    const auto* b = static_cast<const _Float16*>(y1.contents());
    // ULP DISTANCE, not relative error. A reassociation moves an output
    // by a last-place bit or two; measured RELATIVELY that is unbounded
    // wherever the output happens to land near zero, and with half a
    // million outputs one always does. The dense split-K reports its
    // drift the same way (mma-splitk.h: "differ by at most one f16 ulp").
    //
    // ...GATED BY MAGNITUDE, because an f16 ulp near zero is tiny in
    // absolute terms and the f32 reassociation noise is not: with 64
    // float adds per output (group 64) instead of 8, a cancelling output
    // of ~1e-3 can sit 9 of ITS ulps from the single op while the two
    // f32 sums differ by 1e-5. So the bound is asked of outputs at or
    // above 1/16, where one f16 ulp (6e-5) exceeds that noise and only
    // rounding-boundary flips remain; the worst ulp overall is reported
    // with the magnitude it occurred at.
    std::size_t diff = 0, worst_ulp = 0, worst_ulp_big = 0;
    double worst_abs = 0.0, num = 0.0, den = 0.0, worst_at = 0.0;
    for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
      const double va = (double)a[i], vb = (double)b[i];
      const double d = std::fabs(va - vb);
      if (d != 0.0) { ++diff; }
      worst_abs = std::max(worst_abs, d);
      num += (va - vb) * (va - vb);
      den += va * va;
      std::uint16_t ba, bb;
      std::memcpy(&ba, &a[i], 2);
      std::memcpy(&bb, &b[i], 2);
      // Same-sign f16 bit patterns are monotone in magnitude, so the
      // pattern difference IS the ulp distance; opposite signs only
      // happen at zero crossings, where the absolute figure is the one
      // that means anything.
      if ((ba & 0x8000u) == (bb & 0x8000u)) {
        const std::size_t u = (std::size_t)(ba > bb ? ba - bb : bb - ba);
        if (u > worst_ulp) { worst_ulp = u; worst_at = std::fabs(va); }
        if (std::fabs(va) >= 0.0625) {
          worst_ulp_big = std::max(worst_ulp_big, u);
        }
      }
    }
    std::printf("[gemm_i8ctx] S=%-2d bands %d: %5.2f%% differ, worst %zu ulp "
                "(at |y| %.1e; %zu ulp at |y| >= 1/16), max |d| %.2e, "
                "rel-L2 %.2e\n", S, (M + 255) / 256,
                100.0 * (double)diff / (double)(M * N), worst_ulp, worst_at,
                worst_ulp_big, worst_abs,
                den > 0.0 ? std::sqrt(num / den) : 0.0);
    EXPECT_TRUE(worst_ulp_big <= 2);
    EXPECT_TRUE(den > 0.0 && std::sqrt(num / den) < 1e-4);
  }
  // THE TUNER ITSELF, which is the templated code every caller
  // instantiates and the only part of this that runs its own command
  // streams. What it must do: return a candidate (or 0), and return the
  // SAME one next time -- a tuner free to answer differently per call
  // makes the output irreproducible, since a split moves ~1 ulp.
  {
    vpipe::genai::I8GemmContext t(mc, /*want=*/true, /*bf16=*/false);
    if (t.split_available()) {
      auto encode_one = [&](ComputeEncoder& e) {
        (void)t.gemm(e, xb, 0, wb, y1, 0, M, N, K);
      };
      const int pick = t.tune(mc, K, N, M, encode_one);
      const std::vector<int> tc = t.split_candidates(M, N, K);
      const bool legal =
          pick == 0 || std::find(tc.begin(), tc.end(), pick) != tc.end();
      const int again = t.tune(mc, K, N, M, encode_one);
      std::printf("[gemm_i8ctx] tuner picked S=%d (legal %d, cached %d)\n",
                  pick, legal ? 1 : 0, again == pick ? 1 : 0);
      EXPECT_TRUE(legal);
      EXPECT_TRUE(again == pick);
      // ...and the flags it toggles are left clean, or the next GEMM
      // through this context runs whatever the last candidate was.
      EXPECT_TRUE(!t.bypass_split && t.force_splits == 0);
    }
  }

  // THE BIAS, which VOSR needs and nothing else here has: every one of
  // that model's projections carries one, so without a bias term the
  // int8 path could not serve a single GEMM in it. Checked on BOTH arms,
  // because the split adds it in the fold (once, over the summed planes)
  // where the single op adds it in its own epilogue -- two different
  // places to get the same answer, and multiplying it by S is exactly
  // what a fold that added it per plane would do.
  {
    std::vector<_Float16> bias((std::size_t)N);
    std::mt19937 br(99u);
    std::normal_distribution<float> bn(0.0f, 1.0f);
    for (auto& v : bias) { v = (_Float16)(bn(br) * 0.25f); }
    SharedBuffer bb = mc->make_shared_buffer(bias.size() * 2);
    if (!bb.empty()) {
      std::memcpy(bb.contents(), bias.data(), bias.size() * 2);
      for (int S : {0, 2, 4}) {
        vpipe::genai::I8GemmContext bctx(mc, /*want=*/true, /*bf16=*/false);
        if (!bctx.enabled()) { break; }
        bctx.bypass_split = (S == 0);
        bctx.force_splits = S;
        { CommandStream st = mc->make_command_stream();
          { ComputeEncoder e = st.begin_compute();
            EXPECT_TRUE(bctx.gemm(e, xb, 0, wb, bb, y1, 0, M, N, K)); }
          st.commit().wait(); }
        // The no-bias reference is y0, so the difference must be the
        // bias EXACTLY -- a fold that added it S times lands S-1 biases
        // away and a missing one lands one bias away.
        const auto* g = static_cast<const _Float16*>(y1.contents());
        double worst = 0.0, ymax = 0.0;
        for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
          const double want =
              (double)a[i] + (double)(float)bias[(std::size_t)(i % N)];
          worst = std::max(worst, std::fabs(want - (double)g[i]));
          ymax = std::max(ymax, std::fabs(want));
        }
        std::printf("[gemm_i8ctx] bias S=%d: worst |got - (y0+b)| %.2e at "
                    "max|y| %.2f\n", S, worst, ymax);
        // Two f16 ulps of the largest output (the reference rounds y0
        // before adding b, the kernel after), plus the reassociation the
        // split is entitled to.
        EXPECT_TRUE(worst <= 2.0 * ymax * 0.001 + 1e-3);
      }
    }
  }

  // THE DEFERRED PATH, which is what flux2, krea2 and qwen use: gemm()
  // records the shape, and a later call with no encoder open measures it
  // off the scratches that call left behind. The three of them have no
  // closure to hand -- their weights may still be streaming when the
  // shape is first known, and a quantized checkpoint's dense operand is a
  // dequant scratch that exists only inside an encoder.
  {
    vpipe::genai::I8GemmContext d(mc, /*want=*/true, /*bf16=*/false);
    if (d.split_available()) {
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          EXPECT_TRUE(d.gemm(e, xb, 0, wb, y1, 0, M, N, K)); }
        st.commit().wait(); }
      // Nothing is tuned yet, so the run above used the default...
      const int before = d.plan_for_test(M, N, K);
      d.tune_pending(mc);
      const int after = d.plan_for_test(M, N, K);
      // ...and now the answer is the tuner's, whatever it is -- which
      // may well be the same width the default picked, so the value is
      // not the check. What IS checked: it is a legal candidate, it is
      // stable across a second call, and the GEMM that follows is still
      // right. `tune()` above is what proves a candidate gets chosen by
      // measurement rather than by falling through.
      const std::vector<int> dc = d.split_candidates(M, N, K);
      const bool legal =
          after == 0 || std::find(dc.begin(), dc.end(), after) != dc.end();
      std::printf("[gemm_i8ctx] deferred: default S=%d -> tuned S=%d "
                  "(legal %d)\n", before, after, legal ? 1 : 0);
      EXPECT_TRUE(legal);
      // Idempotent and cheap the second time: nothing pending, same answer.
      d.tune_pending(mc);
      EXPECT_TRUE(d.plan_for_test(M, N, K) == after);
      // ...and the result is USED. Re-running the same GEMM must now go
      // through whatever the tuner chose, which is only observable as the
      // answer still being correct.
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          EXPECT_TRUE(d.gemm(e, xb, 0, wb, y1, 0, M, N, K)); }
        st.commit().wait(); }
      const auto* c2 = static_cast<const _Float16*>(y1.contents());
      double n2 = 0.0, d2 = 0.0;
      for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
        n2 += ((double)a[i] - (double)c2[i]) * ((double)a[i] - (double)c2[i]);
        d2 += (double)a[i] * (double)a[i];
      }
      EXPECT_TRUE(d2 > 0.0 && std::sqrt(n2 / d2) < 1e-4);
    }
  }

  // ...and the untuned default is a split, not the single op: a caller
  // that never tunes should still get the part of this that is free.
  vpipe::genai::I8GemmContext deflt(mc, /*want=*/true, /*bf16=*/false);
  run(deflt, y1);
  const auto* b = static_cast<const _Float16*>(y1.contents());
  std::size_t ddiff = 0;
  for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
    if ((double)a[i] != (double)b[i]) { ++ddiff; }
  }
  std::printf("[gemm_i8ctx] untuned default vs single-op: %.2f%% differ\n",
              100.0 * (double)ddiff / (double)(M * N));
  }  // for grp
}

// THE SPLIT'S WORTH UNDER EACH GROUP, through the class. gemm_i8.splitk
// prices the drained g512 kernels by hand; this prices what a caller gets
// -- quantize passes included -- at H3's fc2 shape, for the register g64
// pair and the drained g512 one: every split each admits against its own
// single op. What it decides is plan_()'s untuned default, which was
// measured on the drained kernel and need not carry over: the register
// kernel has much less of the deep-K cliff the split exists to fix.
TEST(gemm_i8, context_split_rate) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ::setenv("VPIPE_I8_GEMM_MIN_M", "64", 1);
  struct Unset {
    ~Unset() {
      ::unsetenv("VPIPE_I8_GEMM_MIN_M");
      ::unsetenv("VPIPE_I8_GROUP");
    }
  } unset;
  const int N = 7168, K = 14336;
  SharedBuffer w = mc->make_shared_buffer((std::size_t)N * K * 2);
  if (w.empty()) { return; }
  {
    auto* pw = static_cast<_Float16*>(w.contents());
    for (std::size_t i = 0; i < (std::size_t)N * K; ++i) {
      pw[i] = (_Float16)(0.05f * (float)((int)(i % 7) - 3));
    }
  }
  for (int grp : {32, 64, 128, 512}) {
    ::setenv("VPIPE_I8_GROUP", std::to_string(grp).c_str(), 1);
    vpipe::genai::I8GemmContext ctx(mc, /*want=*/true, /*bf16=*/false);
    if (!ctx.enabled() || !ctx.split_available()) {
      std::printf("[gemm_i8rate] group %d unavailable -- skip\n", grp);
      continue;
    }
    for (int M : {256, 1024}) {
      SharedBuffer x = mc->make_shared_buffer((std::size_t)M * K * 2);
      SharedBuffer y = mc->make_shared_buffer((std::size_t)M * N * 2);
      if (x.empty() || y.empty()) { continue; }
      auto* px = static_cast<_Float16*>(x.contents());
      for (std::size_t i = 0; i < (std::size_t)M * K; ++i) {
        px[i] = (_Float16)(0.5f * (float)((int)(i % 5) - 2));
      }
      std::vector<int> cands{0};
      for (int S : ctx.split_candidates(M, N, K)) { cands.push_back(S); }
      std::printf("[gemm_i8rate] group %d  M=%d N=%d K=%d (G=%d)\n", grp, M,
                  N, K, ctx.group() > 0 ? K / ctx.group() : 0);
      const double gflop = 2.0 * M * (double)N * K * 1e-9;
      const int iters = 3;
      double base = 0.0;
      for (int S : cands) {
        ctx.bypass_split = (S == 0);
        ctx.force_splits = S;
        auto one = [&](ComputeEncoder& e) {
          EXPECT_TRUE(ctx.gemm(e, x, 0, w, y, 0, M, N, K));
        };
        for (int wu = 0; wu < 2; ++wu) {
          CommandStream st = mc->make_command_stream();
          { ComputeEncoder e = st.begin_compute(); one(e); }
          st.commit().wait();
        }
        const auto t0 = std::chrono::steady_clock::now();
        { CommandStream st = mc->make_command_stream();
          { ComputeEncoder e = st.begin_compute();
            for (int i = 0; i < iters; ++i) { one(e); } }
          st.commit().wait(); }
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count() /
                          iters;
        const double tops = gflop / ms;
        if (S == 0) { base = tops; }
        std::printf("[gemm_i8rate]   S=%-2d %7.2f ms  %5.2f TOP/s  %.2fx\n",
                    S, ms, tops, base > 0 ? tops / base : 0.0);
      }
      ctx.bypass_split = false;
      ctx.force_splits = 0;
    }
  }
}

// NATIVE AFFINE CONSUMPTION (I8GemmContext::gemm_affine): the int8
// activation against the checkpoint's own w4 / w8 codes, no dequant
// scratch and no requant. Three things are checked per bit width: the
// GPU matches a CPU replica of the exact integer algorithm (which also
// settles the 4-bit nibble order -- a wrong one is not subtle); its f32
// quality against the dequant + requant path on the SAME weight, which
// it should beat, having quantized the weight zero extra times; and the
// split-K and bias forms against the single op.
static void pack_affine_(int bits, int N, int K, std::mt19937& rng,
                         std::vector<std::uint8_t>* packed,
                         std::vector<_Float16>* scales,
                         std::vector<_Float16>* biases,
                         std::vector<float>* deq, std::vector<int>* q)
{
  const int G = K / 64;
  const int qmax = (bits == 8) ? 255 : 15;
  const std::size_t row_bytes =
      (bits == 8) ? (std::size_t)K : (std::size_t)(K / 2);
  std::uniform_int_distribution<int> qd(0, qmax);
  std::uniform_real_distribution<float> sd(0.003f, 0.02f);
  std::uniform_real_distribution<float> bd(-0.1f, 0.1f);
  packed->assign((std::size_t)N * row_bytes, 0);
  scales->resize((std::size_t)N * G);
  biases->resize((std::size_t)N * G);
  deq->resize((std::size_t)N * K);
  q->resize((std::size_t)N * K);
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < G; ++g) {
      (*scales)[(std::size_t)n * G + g] = (_Float16)sd(rng);
      (*biases)[(std::size_t)n * G + g] = (_Float16)bd(rng);
    }
    for (int k = 0; k < K; ++k) {
      const int v = qd(rng);
      (*q)[(std::size_t)n * K + k] = v;
      if (bits == 8) {
        (*packed)[(std::size_t)n * row_bytes + k] = (std::uint8_t)v;
      } else {
        const std::size_t bi = (std::size_t)n * row_bytes + (k >> 1);
        if (k & 1) { (*packed)[bi] |= (std::uint8_t)(v << 4); }
        else       { (*packed)[bi] |= (std::uint8_t)v; }
      }
      const int g = k / 64;
      (*deq)[(std::size_t)n * K + k] =
          (float)(*scales)[(std::size_t)n * G + g] * (float)v +
          (float)(*biases)[(std::size_t)n * G + g];
    }
  }
}

TEST(gemm_i8, native_affine_matches_oracle) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ::setenv("VPIPE_I8_GEMM_MIN_M", "64", 1);
  struct Unset { ~Unset() { ::unsetenv("VPIPE_I8_GEMM_MIN_M"); } } unset;
  ComputeLibrary lib_dq = mc->load_library("affine_dequant");
  ComputeFunction f_dq4 = lib_dq.function("affine_dequant_w4g64");
  ComputeFunction f_dq8 = lib_dq.function("affine_dequant_w8g64");
  vpipe::genai::I8GemmContext ctx(mc, /*want=*/true, /*bf16=*/false);
  if (!ctx.enabled()) { return; }
  ctx.set_native_policy(12);   // both widths, whatever the default
  // The UNSIGNED kernel: gemm_affine converts to the signed layout per
  // call where it can, and the replica below is the unsigned algorithm.
  ctx.set_signed_policy(0);
  EXPECT_TRUE(ctx.native_available(4) && ctx.native_available(8));
  if (!ctx.native_available(4) || !ctx.native_available(8)) {
    std::printf("[gemm_i8nat] native kernels unavailable\n");
    return;
  }

  const int M = 256, N = 512, K = 1024, G = K / 64;
  std::mt19937 rng(515u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<_Float16> x((std::size_t)M * K);
  for (auto& v : x) { v = (_Float16)(nd(rng) * 0.5f); }
  SharedBuffer xb = mc->make_shared_buffer(x.size() * 2);
  std::memcpy(xb.contents(), x.data(), x.size() * 2);
  // The CPU's copy of the quantizer: the same f32 ops, so the codes and
  // scales are what the GPU made.
  std::vector<int> xq((std::size_t)M * K);
  std::vector<float> xa((std::size_t)M * G);
  std::vector<int> xsum((std::size_t)M * G, 0);
  for (int m = 0; m < M; ++m) {
    for (int g = 0; g < G; ++g) {
      float am = 0.0f;
      for (int k = g * 64; k < (g + 1) * 64; ++k) {
        am = std::max(am, std::fabs((float)x[(std::size_t)m * K + k]));
      }
      const float inv = am > 0.0f ? 127.0f / am : 0.0f;
      xa[(std::size_t)m * G + g] = (float)(_Float16)(am * (1.0f / 127.0f));
      for (int k = g * 64; k < (g + 1) * 64; ++k) {
        int v = (int)std::rint((float)x[(std::size_t)m * K + k] * inv);
        v = std::max(-127, std::min(127, v));
        xq[(std::size_t)m * K + k] = v;
        xsum[(std::size_t)m * G + g] += v;
      }
    }
  }

  for (int cmode : {0, 1}) {
  ctx.set_native_cmode(cmode);
  std::printf("[gemm_i8nat] ---- cmode %d ----\n", cmode);
  for (int bits : {8, 4}) {
    std::vector<std::uint8_t> packed;
    std::vector<_Float16> scales, biases;
    std::vector<float> deq;
    std::vector<int> q;
    pack_affine_(bits, N, K, rng, &packed, &scales, &biases, &deq, &q);
    SharedBuffer wb = mc->make_shared_buffer(packed.size());
    SharedBuffer sb = mc->make_shared_buffer(scales.size() * 2);
    SharedBuffer bb = mc->make_shared_buffer(biases.size() * 2);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)N * K * 2);
    const std::size_t ybytes = (std::size_t)M * N * 2;
    SharedBuffer y_nat = mc->make_shared_buffer(ybytes);
    SharedBuffer y_sk = mc->make_shared_buffer(ybytes);
    SharedBuffer y_rq = mc->make_shared_buffer(ybytes);
    SharedBuffer y_b = mc->make_shared_buffer(ybytes);
    if (y_b.empty() || wdq.empty()) { return; }
    std::memcpy(wb.contents(), packed.data(), packed.size());
    std::memcpy(sb.contents(), scales.data(), scales.size() * 2);
    std::memcpy(bb.contents(), biases.data(), biases.size() * 2);

    // Native, single op.
    ctx.bypass_split = true;
    ctx.force_splits = 0;
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder e = st.begin_compute();
        EXPECT_TRUE(ctx.gemm_affine(e, xb, 0, wb, sb, bb, bits, y_nat, 0,
                                    M, N, K)); }
      st.commit().wait(); }
    // The requant path on the same weight: dequant, then gemm().
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder e = st.begin_compute();
        e.set_function(bits == 8 ? f_dq8 : f_dq4);
        e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, wdq);
        e.set_constant(4, K); e.set_constant(5, N);
        e.dispatch({(unsigned)(bits == 8 ? K / 4 : K / 8), (unsigned)N, 1},
                   {64, 1, 1});
        EXPECT_TRUE(ctx.gemm(e, xb, 0, wdq, y_rq, 0, M, N, K)); }
      st.commit().wait(); }

    // The replica, and f32 truth.
    const auto* pn = static_cast<const _Float16*>(y_nat.contents());
    const auto* pr = static_cast<const _Float16*>(y_rq.contents());
    double n_or = 0, d_or = 0, n_nat = 0, n_rq = 0, d_f = 0;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float facc = 0.0f, ub = 0.0f;
        double truth = 0.0;
        for (int g = 0; g < G; ++g) {
          long long P = 0, Q = 0;
          for (int k = g * 64; k < (g + 1) * 64; ++k) {
            P += (long long)xq[(std::size_t)m * K + k] *
                 (long long)q[(std::size_t)n * K + k];
            Q += q[(std::size_t)n * K + k];
          }
          (void)Q;   // the GPU adds 128*sum(q) and takes it back exactly
          const float a = xa[(std::size_t)m * G + g];
          // u is stored in the element type by the quantizer.
          const float u = (float)(_Float16)(a * (float)xsum[(std::size_t)m * G + g]);
          const float sc = (float)scales[(std::size_t)n * G + g];
          const float bi = (float)biases[(std::size_t)n * G + g];
          facc += a * sc * (float)P;
          ub += u * bi;
        }
        facc += ub;
        for (int k = 0; k < K; ++k) {
          truth += (double)x[(std::size_t)m * K + k] *
                   (double)deq[(std::size_t)n * K + k];
        }
        const std::size_t o = (std::size_t)m * N + n;
        const double ref = (double)(_Float16)facc;
        const double vn = (double)pn[o], vr = (double)pr[o];
        n_or += (vn - ref) * (vn - ref); d_or += ref * ref;
        n_nat += (vn - truth) * (vn - truth);
        n_rq += (vr - truth) * (vr - truth);
        d_f += truth * truth;
      }
    }
    const double r_or = std::sqrt(n_or / d_or);
    const double q_nat = std::sqrt(n_nat / d_f), q_rq = std::sqrt(n_rq / d_f);
    std::printf("[gemm_i8nat] w%d: replica rel-L2 %.3e %s | vs f32: native "
                "%.4e, dequant+requant %.4e (%dx%dx%d)\n", bits, r_or,
                r_or < 1e-4 ? "MATCH" : "(BAD)", q_nat, q_rq, M, N, K);
    EXPECT_TRUE(r_or < (cmode == 1 ? 1e-3 : 1e-4));
    EXPECT_TRUE(q_nat < q_rq);

    // Split-K forms against the single op, and the bias form.
    for (int S : {2, 4}) {
      ctx.bypass_split = false;
      ctx.force_splits = S;
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          EXPECT_TRUE(ctx.gemm_affine(e, xb, 0, wb, sb, bb, bits, y_sk, 0,
                                      M, N, K)); }
        st.commit().wait(); }
      const auto* ps = static_cast<const _Float16*>(y_sk.contents());
      double n2 = 0, d2 = 0;
      for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
        n2 += ((double)ps[i] - (double)pn[i]) * ((double)ps[i] - (double)pn[i]);
        d2 += (double)pn[i] * (double)pn[i];
      }
      std::printf("[gemm_i8nat] w%d: S=%d vs single rel-L2 %.2e\n", bits, S,
                  std::sqrt(n2 / d2));
      EXPECT_TRUE(std::sqrt(n2 / d2) < 1e-4);
    }
    ctx.bypass_split = false;
    ctx.force_splits = 0;
    {
      std::vector<_Float16> cb((std::size_t)N);
      for (auto& v : cb) { v = (_Float16)(nd(rng) * 0.25f); }
      SharedBuffer cbb = mc->make_shared_buffer(cb.size() * 2);
      std::memcpy(cbb.contents(), cb.data(), cb.size() * 2);
      ctx.bypass_split = true;
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          EXPECT_TRUE(ctx.gemm_affine(e, xb, 0, wb, sb, bb, bits, cbb, y_b,
                                      0, M, N, K)); }
        st.commit().wait(); }
      ctx.bypass_split = false;
      const auto* pb = static_cast<const _Float16*>(y_b.contents());
      double worst = 0.0;
      for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
        const double want = (double)pn[i] + (double)(float)cb[i % N];
        worst = std::max(worst, std::fabs(want - (double)pb[i]));
      }
      double ymax = 0.0;
      for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
        ymax = std::max(ymax, std::fabs((double)pn[i]));
      }
      std::printf("[gemm_i8nat] w%d: bias worst |got - (y+b)| %.2e at "
                  "max|y| %.1f\n", bits, worst, ymax);
      // Two f16 ulps of the largest output: the reference rounds y before
      // adding b, the kernel after.
      EXPECT_TRUE(worst <= 2.0 * ymax * 0.001);
    }
  }
  }  // cmode
  ctx.set_native_cmode(0);
}

// THE SIGNED-CODE LAYOUT (I8GemmContext::gemm_signed): codes stored q ^
// 0x80 / q ^ 0x8 per nibble, the zero point moved to b' = b + 128 s / 8 s.
// Per width: the GPU's conversion is the CPU's to the byte (into a new
// buffer and in place); the GEMM matches a CPU replica of the exact
// integer algorithm; gemm_affine's per-call conversion is the stored
// layout's result BYTE FOR BYTE; its f32 quality is at least the requant
// route's and close to the unsigned kernel's (b' is rounded once, where
// the unsigned kernel keeps b); the signed dequant is the CPU's; and the
// split and bias forms agree with the single op.
TEST(gemm_i8, signed_layout_matches_oracle) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ::setenv("VPIPE_I8_GEMM_MIN_M", "64", 1);
  struct Unset { ~Unset() { ::unsetenv("VPIPE_I8_GEMM_MIN_M"); } } unset;
  ComputeLibrary lib_dq = mc->load_library("affine_dequant");
  ComputeFunction f_dq4 = lib_dq.function("affine_dequant_w4g64");
  ComputeFunction f_dq8 = lib_dq.function("affine_dequant_w8g64");
  vpipe::genai::I8GemmContext ctx(mc, /*want=*/true, /*bf16=*/false);
  if (!ctx.enabled()) { return; }
  ctx.set_native_policy(12);
  ctx.set_signed_policy(12);
  EXPECT_TRUE(ctx.signed_available(4) && ctx.signed_available(8));
  if (!ctx.signed_available(4) || !ctx.signed_available(8)) {
    std::printf("[gemm_i8sig] signed kernels unavailable\n");
    return;
  }
  const int M = 256, N = 512, K = 1024, G = K / 64;
  std::mt19937 rng(616u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<_Float16> x((std::size_t)M * K);
  for (auto& v : x) { v = (_Float16)(nd(rng) * 0.5f); }
  SharedBuffer xb = mc->make_shared_buffer(x.size() * 2);
  std::memcpy(xb.contents(), x.data(), x.size() * 2);
  // The CPU's copy of the quantizer: the same f32 ops as the GPU's.
  std::vector<int> xq((std::size_t)M * K);
  std::vector<float> xa((std::size_t)M * G), xu((std::size_t)M * G);
  for (int m = 0; m < M; ++m) {
    for (int g = 0; g < G; ++g) {
      float am = 0.0f;
      for (int k = g * 64; k < (g + 1) * 64; ++k) {
        am = std::max(am, std::fabs((float)x[(std::size_t)m * K + k]));
      }
      const float inv = am > 0.0f ? 127.0f / am : 0.0f;
      const float a = (float)(_Float16)(am * (1.0f / 127.0f));
      int sum = 0;
      for (int k = g * 64; k < (g + 1) * 64; ++k) {
        int v = (int)std::rint((float)x[(std::size_t)m * K + k] * inv);
        v = std::max(-127, std::min(127, v));
        xq[(std::size_t)m * K + k] = v;
        sum += v;
      }
      xa[(std::size_t)m * G + g] = a;
      xu[(std::size_t)m * G + g] = (float)(_Float16)(a * (float)sum);
    }
  }
  auto rel = [](const _Float16* a, const _Float16* b, std::size_t n) {
    double nn = 0, dd = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const double d = (double)a[i] - (double)b[i];
      nn += d * d;
      dd += (double)b[i] * (double)b[i];
    }
    return std::sqrt(nn / dd);
  };
  for (int bits : {8, 4}) {
    std::vector<std::uint8_t> packed;
    std::vector<_Float16> scales, biases;
    std::vector<float> deq;
    std::vector<int> q;
    pack_affine_(bits, N, K, rng, &packed, &scales, &biases, &deq, &q);
    const int off = bits == 8 ? 128 : 8;
    const std::uint8_t flip = bits == 8 ? 0x80 : 0x88;
    std::vector<std::uint8_t> spk(packed.size());
    for (std::size_t i = 0; i < packed.size(); ++i) {
      spk[i] = (std::uint8_t)(packed[i] ^ flip);
    }
    std::vector<_Float16> zb(biases.size());
    for (std::size_t i = 0; i < zb.size(); ++i) {
      zb[i] = (_Float16)((float)biases[i] + (float)off * (float)scales[i]);
    }
    auto up = [&](const void* p, std::size_t n) {
      SharedBuffer b = mc->make_shared_buffer(n);
      std::memcpy(b.contents(), p, n);
      return b;
    };
    SharedBuffer wb = up(packed.data(), packed.size());
    SharedBuffer sb = up(scales.data(), scales.size() * 2);
    SharedBuffer bb = up(biases.data(), biases.size() * 2);
    SharedBuffer gcs = mc->make_shared_buffer(packed.size());
    SharedBuffer gzb = mc->make_shared_buffer(zb.size() * 2);
    SharedBuffer ipc = up(packed.data(), packed.size());     // in place
    SharedBuffer ipb = up(biases.data(), biases.size() * 2);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)N * K * 2);
    SharedBuffer sdq = mc->make_shared_buffer((std::size_t)N * K * 2);
    const std::size_t ybytes = (std::size_t)M * N * 2;
    SharedBuffer y_sig = mc->make_shared_buffer(ybytes);
    SharedBuffer y_aff = mc->make_shared_buffer(ybytes);
    SharedBuffer y_nat = mc->make_shared_buffer(ybytes);
    SharedBuffer y_rq = mc->make_shared_buffer(ybytes);
    SharedBuffer y_sk = mc->make_shared_buffer(ybytes);
    SharedBuffer y_b = mc->make_shared_buffer(ybytes);
    if (y_b.empty() || sdq.empty()) { return; }
    ctx.bypass_split = true;
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder e = st.begin_compute();
        EXPECT_TRUE(ctx.encode_to_signed(e, wb, sb, bb, gcs, gzb, bits, N,
                                         K));
        EXPECT_TRUE(ctx.encode_to_signed(e, ipc, sb, ipb, ipc, ipb, bits, N,
                                         K));
        EXPECT_TRUE(ctx.gemm_signed(e, xb, 0, gcs, sb, gzb, bits, y_sig, 0,
                                    M, N, K));
        EXPECT_TRUE(ctx.gemm_affine(e, xb, 0, wb, sb, bb, bits, y_aff, 0, M,
                                    N, K));
        EXPECT_TRUE(ctx.encode_dequant_signed(e, gcs, sb, gzb, bits, sdq, N,
                                              K));
        e.set_function(bits == 8 ? f_dq8 : f_dq4);
        e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, wdq);
        e.set_constant(4, K); e.set_constant(5, N);
        e.dispatch({(unsigned)(bits == 8 ? K / 4 : K / 8), (unsigned)N, 1},
                   {64, 1, 1});
        EXPECT_TRUE(ctx.gemm(e, xb, 0, wdq, y_rq, 0, M, N, K)); }
      st.commit().wait(); }
    ctx.set_signed_policy(0);
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder e = st.begin_compute();
        EXPECT_TRUE(ctx.gemm_affine(e, xb, 0, wb, sb, bb, bits, y_nat, 0, M,
                                    N, K)); }
      st.commit().wait(); }
    ctx.set_signed_policy(12);
    ctx.bypass_split = false;

    // The conversion, to the byte, both ways.
    EXPECT_TRUE(std::memcmp(gcs.contents(), spk.data(), spk.size()) == 0);
    EXPECT_TRUE(std::memcmp(ipc.contents(), spk.data(), spk.size()) == 0);
    EXPECT_TRUE(std::memcmp(gzb.contents(), zb.data(), zb.size() * 2) == 0);
    EXPECT_TRUE(std::memcmp(ipb.contents(), zb.data(), zb.size() * 2) == 0);
    // Per call IS the stored layout's result.
    EXPECT_TRUE(std::memcmp(y_aff.contents(), y_sig.contents(), ybytes) ==
                0);

    // The replica, and f32 truth.
    const auto* ps = static_cast<const _Float16*>(y_sig.contents());
    const auto* pn = static_cast<const _Float16*>(y_nat.contents());
    const auto* pr = static_cast<const _Float16*>(y_rq.contents());
    double n_or = 0, d_or = 0, n_s = 0, n_n = 0, n_r = 0, d_f = 0;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float facc = 0.0f, ub = 0.0f;
        double truth = 0.0;
        for (int g = 0; g < G; ++g) {
          long long P = 0;
          for (int k = g * 64; k < (g + 1) * 64; ++k) {
            P += (long long)xq[(std::size_t)m * K + k] *
                 (long long)(q[(std::size_t)n * K + k] - off);
          }
          const std::size_t wg = (std::size_t)n * G + g;
          facc += xa[(std::size_t)m * G + g] * (float)scales[wg] * (float)P;
          ub += xu[(std::size_t)m * G + g] * (float)zb[wg];
        }
        facc += ub;
        for (int k = 0; k < K; ++k) {
          truth += (double)x[(std::size_t)m * K + k] *
                   (double)deq[(std::size_t)n * K + k];
        }
        const std::size_t o = (std::size_t)m * N + n;
        const double ref = (double)(_Float16)facc;
        n_or += ((double)ps[o] - ref) * ((double)ps[o] - ref);
        d_or += ref * ref;
        n_s += ((double)ps[o] - truth) * ((double)ps[o] - truth);
        n_n += ((double)pn[o] - truth) * ((double)pn[o] - truth);
        n_r += ((double)pr[o] - truth) * ((double)pr[o] - truth);
        d_f += truth * truth;
      }
    }
    const double r_or = std::sqrt(n_or / d_or);
    const double q_s = std::sqrt(n_s / d_f), q_n = std::sqrt(n_n / d_f);
    const double q_r = std::sqrt(n_r / d_f);
    std::printf("[gemm_i8sig] w%d: replica rel-L2 %.3e | vs f32: signed "
                "%.4e, unsigned %.4e, dequant+requant %.4e\n", bits, r_or,
                q_s, q_n, q_r);
    EXPECT_TRUE(r_or < 1e-3);
    EXPECT_TRUE(q_s < q_r);
    EXPECT_TRUE(q_s < 1.15 * q_n);

    // The signed dequant: the CPU's s * code + b', and within the zero
    // point's rounding of the unsigned dequant.
    {
      std::vector<_Float16> want((std::size_t)N * K);
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
          const std::size_t wg = (std::size_t)n * G + k / 64;
          want[(std::size_t)n * K + k] = (_Float16)(
              (float)scales[wg] * (float)(q[(std::size_t)n * K + k] - off) +
              (float)zb[wg]);
        }
      }
      const double rd = rel(static_cast<const _Float16*>(sdq.contents()),
                            want.data(), want.size());
      const double ru = rel(static_cast<const _Float16*>(sdq.contents()),
                            static_cast<const _Float16*>(wdq.contents()),
                            want.size());
      std::printf("[gemm_i8sig] w%d: signed dequant vs CPU %.2e, vs the "
                  "unsigned dequant %.2e\n", bits, rd, ru);
      EXPECT_TRUE(rd < 1e-4);
      EXPECT_TRUE(ru < 2e-3);
    }

    // Split-K against the single op.
    for (int S : {2, 4}) {
      ctx.force_splits = S;
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          EXPECT_TRUE(ctx.gemm_signed(e, xb, 0, gcs, sb, gzb, bits, y_sk, 0,
                                      M, N, K)); }
        st.commit().wait(); }
      const double r = rel(static_cast<const _Float16*>(y_sk.contents()),
                           ps, (std::size_t)M * N);
      std::printf("[gemm_i8sig] w%d: S=%d vs single rel-L2 %.2e\n", bits, S,
                  r);
      EXPECT_TRUE(r < 1e-4);
    }
    ctx.force_splits = 0;
    // THE ROUTES RECORD THEIR SHAPES. The native route never did, so a
    // quantized checkpoint's GEMMs kept the untuned split at every shape;
    // both now queue for tune_pending, which settles them over the
    // recorded weight, and the GEMM after is still right.
    for (int route = 0; route < 2; ++route) {
      vpipe::genai::I8GemmContext d(mc, /*want=*/true, /*bf16=*/false);
      if (!d.split_available()) { break; }
      d.set_signed_policy(route == 0 ? 0 : 12);
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          EXPECT_TRUE(route == 0
              ? d.gemm_affine(e, xb, 0, wb, sb, bb, bits, y_sk, 0, M, N, K)
              : d.gemm_signed(e, xb, 0, gcs, sb, gzb, bits, y_sk, 0, M, N,
                              K)); }
        st.commit().wait(); }
      EXPECT_TRUE(d.pending_for_test() == 1);
      EXPECT_TRUE(!d.tuned_for_test(M, N, K));
      d.tune_pending(mc);
      EXPECT_TRUE(d.pending_for_test() == 0);
      EXPECT_TRUE(d.tuned_for_test(M, N, K));
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          EXPECT_TRUE(route == 0
              ? d.gemm_affine(e, xb, 0, wb, sb, bb, bits, y_sk, 0, M, N, K)
              : d.gemm_signed(e, xb, 0, gcs, sb, gzb, bits, y_sk, 0, M, N,
                              K)); }
        st.commit().wait(); }
      EXPECT_TRUE(d.pending_for_test() == 0);
      const double r = rel(static_cast<const _Float16*>(y_sk.contents()),
                           route == 0 ? pn : ps, (std::size_t)M * N);
      std::printf("[gemm_i8sig] w%d: %s route tuned to S=%d, rel-L2 %.2e "
                  "from the single op\n", bits,
                  route == 0 ? "unsigned" : "signed",
                  d.plan_for_test(M, N, K), r);
      EXPECT_TRUE(r < 1e-4);
    }
    // The bias form.
    {
      std::vector<_Float16> cb((std::size_t)N);
      for (auto& v : cb) { v = (_Float16)(nd(rng) * 0.25f); }
      SharedBuffer cbb = up(cb.data(), cb.size() * 2);
      ctx.bypass_split = true;
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          EXPECT_TRUE(ctx.gemm_signed(e, xb, 0, gcs, sb, gzb, bits, cbb, y_b,
                                      0, M, N, K)); }
        st.commit().wait(); }
      ctx.bypass_split = false;
      const auto* pb = static_cast<const _Float16*>(y_b.contents());
      double worst = 0.0, ymax = 0.0;
      for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
        const double want = (double)ps[i] + (double)(float)cb[i % N];
        worst = std::max(worst, std::fabs(want - (double)pb[i]));
        ymax = std::max(ymax, std::fabs((double)ps[i]));
      }
      std::printf("[gemm_i8sig] w%d: bias worst |got - (y+b)| %.2e at "
                  "max|y| %.1f\n", bits, worst, ymax);
      EXPECT_TRUE(worst <= 2.0 * ymax * 0.001);
    }
  }
}

// THE SIGNED LAYOUT ON A REAL PACK's weights: the signed dequant against
// the unsigned one, weight by weight, so the one rounding the layout adds
// (b' to the element type) is measured where it lives rather than through
// a model. VPIPE_I8_SIGNED_PACK=<model-quantize'd dir>, optionally
// VPIPE_I8_SIGNED_PACK_MATCH=<substring> to pick the weights (default:
// the first 12 quantized ones). bf16, as the image DiTs run.
TEST(gemm_i8, signed_dequant_on_a_real_pack) {
  const char* dir = std::getenv("VPIPE_I8_SIGNED_PACK");
  if (dir == nullptr || *dir == '\0') { return; }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  vpipe::genai::I8GemmContext ctx(mc, /*want=*/true, /*bf16=*/true);
  if (!ctx.signed_available(8)) { return; }
  ComputeLibrary lib = mc->load_library("affine_dequant_bf16");
  ComputeFunction f8 = lib.function("affine_dequant_w8g64");
  ComputeFunction f4 = lib.function("affine_dequant_w4g64");
  auto ws = vpipe::genai::WeightSet::open(dir, nullptr);
  ASSERT_TRUE(ws != nullptr);
  if (!ws) { return; }
  const auto& src = ws->src();
  const char* mt = std::getenv("VPIPE_I8_SIGNED_PACK_MATCH");
  std::shared_ptr<vpipe::genai::WeightSet> orig;
  if (const char* od = std::getenv("VPIPE_I8_SIGNED_PACK_ORIG")) {
    orig = vpipe::genai::WeightSet::open(od, nullptr);
  }
  auto f16_to_bf16 = [](const SharedBuffer& b) {
    const std::size_t n = b.byte_size() / 2;
    std::vector<std::uint16_t> out(n);
    const auto* h = static_cast<const _Float16*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const float f = (float)h[i];
      std::uint32_t u = 0;
      std::memcpy(&u, &f, 4);
      u += 0x7fffu + ((u >> 16) & 1u);
      out[i] = (std::uint16_t)(u >> 16);
    }
    return out;
  };
  auto bf = [](std::uint16_t h) {
    const std::uint32_t u = (std::uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return (double)f;
  };
  int done = 0;
  for (const std::string& nm : src.tensor_names()) {
    if (done >= 12) { break; }
    const std::string sfx = ".scales";
    if (nm.size() < sfx.size() ||
        nm.compare(nm.size() - sfx.size(), sfx.size(), sfx) != 0) {
      continue;
    }
    const std::string base = nm.substr(0, nm.size() - sfx.size());
    if (mt != nullptr && base.find(mt) == std::string::npos) { continue; }
    const auto* ci = src.info(base + ".weight");
    const auto* si = src.info(nm);
    if (ci == nullptr || si == nullptr || ci->shape.size() != 2) { continue; }
    const int N = (int)ci->shape[0];
    const int K = (int)si->shape[1] * 64;
    const int bits = (int)(ci->shape[1] * 32 / K);
    if (bits != 8 && bits != 4) { continue; }
    SharedBuffer codes = ws->read(base + ".weight", mc,
                                  vpipe::genai::WeightSet::Residency::Copied);
    SharedBuffer sc16 = ws->read(nm, mc,
                                 vpipe::genai::WeightSet::Residency::Copied);
    SharedBuffer bi16 = ws->read(base + ".biases", mc,
                                 vpipe::genai::WeightSet::Residency::Copied);
    if (codes.empty() || sc16.empty() || bi16.empty()) { continue; }
    // F16 in a model-quantize'd pack; anything already bf16 is copied.
    auto to_bf = [&](const SharedBuffer& b, const std::string& dt) {
      if (dt == "F16") { return f16_to_bf16(b); }
      std::vector<std::uint16_t> v(b.byte_size() / 2);
      std::memcpy(v.data(), b.contents(), v.size() * 2);
      return v;
    };
    const std::vector<std::uint16_t> sh = to_bf(sc16, si->dtype);
    const std::vector<std::uint16_t> bh =
        to_bf(bi16, src.info(base + ".biases")->dtype);
    SharedBuffer sb = mc->make_shared_buffer(sh.size() * 2);
    SharedBuffer bb = mc->make_shared_buffer(bh.size() * 2);
    std::memcpy(sb.contents(), sh.data(), sh.size() * 2);
    std::memcpy(bb.contents(), bh.data(), bh.size() * 2);
    SharedBuffer cs = mc->make_shared_buffer(codes.byte_size());
    SharedBuffer zb = mc->make_shared_buffer(bh.size() * 2);
    SharedBuffer du = mc->make_shared_buffer((std::size_t)N * K * 2);
    SharedBuffer dsg = mc->make_shared_buffer((std::size_t)N * K * 2);
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder e = st.begin_compute();
        e.set_function(bits == 8 ? f8 : f4);
        e.set_buffer(0, codes); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, du);
        e.set_constant(4, K); e.set_constant(5, N);
        e.dispatch({(unsigned)(bits == 8 ? K / 4 : K / 8), (unsigned)N, 1},
                   {64, 1, 1});
        EXPECT_TRUE(ctx.encode_to_signed(e, codes, sb, bb, cs, zb, bits, N,
                                         K));
        EXPECT_TRUE(ctx.encode_dequant_signed(e, cs, sb, zb, bits, dsg, N,
                                              K)); }
      st.commit().wait(); }
    const auto* pu = static_cast<const std::uint16_t*>(du.contents());
    const auto* pg = static_cast<const std::uint16_t*>(dsg.contents());
    double n2 = 0, d2 = 0, worst = 0;
    for (std::size_t i = 0; i < (std::size_t)N * K; ++i) {
      const double d = bf(pg[i]) - bf(pu[i]);
      n2 += d * d;
      d2 += bf(pu[i]) * bf(pu[i]);
      worst = std::max(worst, std::fabs(d));
    }
    // How large b' runs against a quantization step, over the groups.
    double zr = 0, zmax = 0;
    const auto* pz = static_cast<const std::uint16_t*>(zb.contents());
    for (std::size_t g = 0; g < sh.size(); ++g) {
      const double r = std::fabs(bf(pz[g])) / std::max(1e-30, bf(sh[g]));
      zr += r / (double)sh.size();
      zmax = std::max(zmax, r);
    }
    std::printf("[gemm_i8pack] %-48s w%d %5dx%-5d signed vs unsigned "
                "dequant rel-L2 %.2e worst %.2e | |b'|/s mean %.1f max "
                "%.0f\n", base.c_str(), bits, N, K, std::sqrt(n2 / d2),
                worst, zr, zmax);
    // Against the weight the pack quantized, where its checkpoint is
    // given (VPIPE_I8_SIGNED_PACK_ORIG): each dequant's own error.
    if (orig) {
      SharedBuffer ow = orig->read(base + ".weight", mc,
                                   vpipe::genai::WeightSet::Residency::Copied);
      const auto* oi = orig->src().info(base + ".weight");
      if (!ow.empty() && oi != nullptr &&
          ow.byte_size() == (std::size_t)N * K * (oi->dtype == "F32" ? 4 : 2)) {
        auto ov = [&](std::size_t i) {
          if (oi->dtype == "F32") {
            return (double)static_cast<const float*>(ow.contents())[i];
          }
          if (oi->dtype == "F16") {
            return (double)static_cast<const _Float16*>(ow.contents())[i];
          }
          return bf(static_cast<const std::uint16_t*>(ow.contents())[i]);
        };
        double eu = 0, es = 0, dd = 0;
        for (std::size_t i = 0; i < (std::size_t)N * K; ++i) {
          const double o = ov(i);
          eu += (bf(pu[i]) - o) * (bf(pu[i]) - o);
          es += (bf(pg[i]) - o) * (bf(pg[i]) - o);
          dd += o * o;
        }
        std::printf("[gemm_i8pack]   vs the %s original: unsigned "
                    "dequant %.3e, signed %.3e\n", oi->dtype.c_str(),
                    std::sqrt(eu / dd), std::sqrt(es / dd));
      }
    }
    ++done;
  }
  EXPECT_TRUE(done > 0);
}

// THE NATIVE PATH'S WORTH: per call, what a DiT pays for a quantized
// projection on each route -- dequant + act quant + weight requant + GEMM
// against act quant + code sums + GEMM -- at FLUX.2's and H3's shapes,
// w4 and w8, single op and the default split. Rates count the GEMM's
// FLOPs only, so the passes each route adds are in the time and not in
// the numerator.
TEST(gemm_i8, native_affine_rate) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ::setenv("VPIPE_I8_GEMM_MIN_M", "64", 1);
  struct Unset { ~Unset() { ::unsetenv("VPIPE_I8_GEMM_MIN_M"); } } unset;
  ComputeLibrary lib_dq = mc->load_library("affine_dequant");
  ComputeFunction f_dq4 = lib_dq.function("affine_dequant_w4g64");
  ComputeFunction f_dq8 = lib_dq.function("affine_dequant_w8g64");
  vpipe::genai::I8GemmContext ctx(mc, /*want=*/true, /*bf16=*/false);
  if (ctx.enabled()) { ctx.set_native_policy(12); }
  if (!ctx.enabled() || !ctx.native_available(4) ||
      !ctx.native_available(8)) {
    std::printf("[gemm_i8natr] unavailable -- skip\n");
    return;
  }
  struct Shape { int M, N, K, bits; std::string what; };
  std::vector<Shape> shapes = {
      {4096, 12288, 4096, 4, "flux2 block proj, w4"},
      {4096, 4096, 12288, 4, "deep K, w4"},
      {1024, 7168, 14336, 8, "H3 fc2, w8"},
      {4096, 4096, 4096, 8, "square, w8"},
  };
  // VPIPE_I8_NATR_SHAPES="M:N:K:bits,..." times those instead (a family's
  // own projections, to see where its untuned split lands).
  if (const char* e = std::getenv("VPIPE_I8_NATR_SHAPES")) {
    shapes.clear();
    const std::string want = e;
    for (std::size_t b = 0; b < want.size();) {
      std::size_t end = want.find(',', b);
      if (end == std::string::npos) { end = want.size(); }
      const std::string one = want.substr(b, end - b);
      int m = 0, n = 0, k = 0, bits = 0;
      if (std::sscanf(one.c_str(), "%d:%d:%d:%d", &m, &n, &k, &bits) == 4) {
        shapes.push_back({m, n, k, bits, "w" + std::to_string(bits)});
      }
      b = end + 1;
    }
  }
  for (const Shape& sh : shapes) {
    const int M = sh.M, N = sh.N, K = sh.K, bits = sh.bits, G = K / 64;
    const std::size_t row_bytes = bits == 8 ? (std::size_t)K : (std::size_t)K / 2;
    SharedBuffer xb = mc->make_shared_buffer((std::size_t)M * K * 2);
    SharedBuffer wb = mc->make_shared_buffer((std::size_t)N * row_bytes);
    SharedBuffer sb = mc->make_shared_buffer((std::size_t)N * G * 2);
    SharedBuffer bb = mc->make_shared_buffer((std::size_t)N * G * 2);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)N * K * 2);
    SharedBuffer y = mc->make_shared_buffer((std::size_t)M * N * 2);
    if (y.empty() || wdq.empty() || wb.empty()) {
      std::printf("[gemm_i8natr] alloc failed at %s -- skip\n",
                  sh.what.c_str());
      continue;
    }
    // Patterned operands: the rates are data-independent, the bytes only
    // have to be finite.
    auto* px = static_cast<_Float16*>(xb.contents());
    for (std::size_t i = 0; i < (std::size_t)M * K; ++i) {
      px[i] = (_Float16)(0.25f * (float)((int)(i % 7) - 3));
    }
    std::memset(wb.contents(), 0x5a, wb.byte_size());
    auto* ps = static_cast<_Float16*>(sb.contents());
    auto* pbias = static_cast<_Float16*>(bb.contents());
    for (std::size_t i = 0; i < (std::size_t)N * G; ++i) {
      ps[i] = (_Float16)0.01f;
      pbias[i] = (_Float16)-0.05f;
    }
    const double gflop = 2.0 * M * (double)N * K * 1e-9;
    const int iters = 3;
    std::printf("[gemm_i8natr] %s  M=%d N=%d K=%d\n", sh.what.c_str(), M,
                N, K);
    struct ArmC { const char* name; bool native; int S; int cmode; };
    const ArmC arms[] = {
        {"dequant+requant", false, 0, 0},
        {"dequant+requant S=2", false, 2, 0},
        {"native cm0", true, 0, 0},
        {"native cm0 S=2", true, 2, 0},
        {"native cm1", true, 0, 1},
        {"native cm1 S=2", true, 2, 1},
    };
    double base = 0.0;
    for (const ArmC& a : arms) {
      ctx.set_native_cmode(a.cmode);
      ctx.bypass_split = (a.S == 0);
      ctx.force_splits = a.S;
      auto one = [&](ComputeEncoder& e) {
        if (a.native) {
          EXPECT_TRUE(ctx.gemm_affine(e, xb, 0, wb, sb, bb, bits, y, 0, M,
                                      N, K));
          return;
        }
        e.set_function(bits == 8 ? f_dq8 : f_dq4);
        e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, wdq);
        e.set_constant(4, K); e.set_constant(5, N);
        e.dispatch({(unsigned)(bits == 8 ? K / 4 : K / 8), (unsigned)N, 1},
                   {64, 1, 1});
        EXPECT_TRUE(ctx.gemm(e, xb, 0, wdq, y, 0, M, N, K));
      };
      for (int wu = 0; wu < 2; ++wu) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute(); one(e); }
        st.commit().wait();
      }
      const auto t0 = std::chrono::steady_clock::now();
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          for (int i = 0; i < iters; ++i) { one(e); } }
        st.commit().wait(); }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count() /
                        iters;
      if (base == 0.0) { base = ms; }
      std::printf("[gemm_i8natr]   %-20s %7.2f ms  %5.2f TOP/s  %.2fx\n",
                  a.name, ms, gflop / ms, base / ms);
    }
    ctx.bypass_split = false;
    ctx.force_splits = 0;
  }
}

// WHERE THE NATIVE KERNEL'S TIME GOES: kernel-level arms at the FLUX.2
// block shape, direct dispatch, no quant passes. The unsigned MMA
// ceilings (no epilogue) bound what uint8 x uint4 and uint8 x uint8 can
// do at all; the strip variants price the staging footprint; g64rs is
// the requant kernel the native one has to beat.
TEST(gemm_i8, native_affine_kernel_arms) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("dense_gemm_mma");
  struct Arm { const char* name; const char* fn; int kind; int bits;
               ComputeFunction f; };
  // kind: 0 = kaccr ceiling (0:xu 1:codes 2:y 3:K 4:N 5:M), 1 = native
  // (the 13-buffer contract), 2 = g64rs (the i8 contract).
  std::vector<Arm> arms;
  auto add = [&](const char* name, const char* fn, int kind, int bits) {
    Arm a{name, fn, kind, bits, lib.function(fn)};
    if (!a.f.valid()) { std::printf("[gemm_i8arm] %s missing\n", fn); }
    arms.push_back(std::move(a));
  };
  add("kaccr i8xi8",   "gemm_i8i8_sc_f16_n64_kaccr_k64", 0, 8);
  add("kaccr u8xu8",   "gemm_u8u8_kaccr_k64", 0, 8);
  add("kaccr u8xu4",   "gemm_u8u4_kaccr_k64", 0, 4);
  add("kaccr i8xi4",   "gemm_i8i4_kaccr_k64", 0, 4);
  add("g64rs (requant)", "gemm_i8i8_sc_f16_n64_g64rs", 2, 8);
  add("native w8 cm0", "gemm_u8q_w8g64", 1, 8);
  add("native w8 cm1", "gemm_u8q_w8g64_cm", 1, 8);
  add("native w8 v1",  "gemm_u8q_w8g64_v1", 1, 8);
  add("signed w8",     "gemm_i8q_w8g64s", 3, 8);
  add("native w4 cm0", "gemm_u8q_w4g64", 1, 4);
  add("native w4 cm1", "gemm_u8q_w4g64_cm", 1, 4);
  add("native w4 v1",  "gemm_u8q_w4g64_v1", 1, 4);
  add("signed w4",     "gemm_i8q_w4g64s", 3, 4);
  for (const Arm& a : arms) { if (!a.f.valid()) { return; } }

  auto sweep = [&](int M, int N, int K) {
    const int G = K / 64;
    SharedBuffer xu = mc->make_shared_buffer((std::size_t)M * K);
    SharedBuffer w8 = mc->make_shared_buffer((std::size_t)N * K);
    SharedBuffer as = mc->make_shared_buffer((std::size_t)M * G * 2);
    SharedBuffer xs = mc->make_shared_buffer((std::size_t)M * G * 2);
    SharedBuffer ws = mc->make_shared_buffer((std::size_t)N * G * 2);
    SharedBuffer wb = mc->make_shared_buffer((std::size_t)N * G * 2);
    SharedBuffer wsum = mc->make_shared_buffer((std::size_t)N * G * 2);
    SharedBuffer y = mc->make_shared_buffer((std::size_t)M * N * 2);
    if (y.empty() || w8.empty()) { return; }
    std::memset(xu.contents(), 0x81, xu.byte_size());
    std::memset(w8.contents(), 0x5a, w8.byte_size());
    std::memset(as.contents(), 0, as.byte_size());
    std::memset(xs.contents(), 0, xs.byte_size());
    std::memset(ws.contents(), 0, ws.byte_size());
    std::memset(wb.contents(), 0, wb.byte_size());
    std::memset(wsum.contents(), 0, wsum.byte_size());
    const double gflop = 2.0 * M * (double)N * K * 1e-9;
    const unsigned nx = (unsigned)(((N + 63) / 64) * 128);
    std::printf("[gemm_i8arm] M=%d N=%d K=%d\n", M, N, K);
    for (const Arm& a : arms) {
      auto one = [&](ComputeEncoder& e) {
        e.set_function(a.f);
        if (a.kind == 0) {
          e.set_buffer(0, xu); e.set_buffer(1, w8); e.set_buffer(2, y);
          e.set_constant(3, K); e.set_constant(4, N); e.set_constant(5, M);
        } else if (a.kind == 1) {
          e.set_buffer(0, xu); e.set_buffer(1, w8); e.set_buffer(2, as);
          e.set_buffer(3, xs); e.set_buffer(4, ws); e.set_buffer(5, wb);
          e.set_buffer(6, wsum); e.set_buffer(7, y);
          e.set_constant(8, K); e.set_constant(9, N); e.set_constant(10, M);
          e.set_buffer(11, ws); e.set_constant(12, 0);
          e.set_buffer(13, xs); e.set_buffer(14, ws); e.set_buffer(15, wb);
        } else if (a.kind == 3) {
          e.set_buffer(0, xu); e.set_buffer(1, w8); e.set_buffer(2, as);
          e.set_buffer(3, ws); e.set_buffer(4, wb); e.set_buffer(5, xs);
          e.set_buffer(6, y);
          e.set_constant(7, K); e.set_constant(8, N); e.set_constant(9, M);
        } else {
          e.set_buffer(0, xu); e.set_buffer(1, w8); e.set_buffer(2, as);
          e.set_buffer(3, ws); e.set_buffer(4, y);
          e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, M);
          e.set_buffer(8, ws); e.set_constant(9, 0);
        }
        e.dispatch({nx, (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
      };
      for (int wu = 0; wu < 2; ++wu) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute(); one(e); }
        st.commit().wait();
      }
      const int iters = 4;
      const auto t0 = std::chrono::steady_clock::now();
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          for (int i = 0; i < iters; ++i) { one(e); } }
        st.commit().wait(); }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count() /
                        iters;
      std::printf("[gemm_i8arm]   %-16s %7.2f ms  %5.2f TOP/s\n", a.name,
                  ms, gflop / ms);
    }
  };
  sweep(4096, 12288, 4096);
  sweep(4096, 4096, 4096);
}

// WHERE THE UNTUNED SPLIT SHOULD START, by measurement: the single op
// against every balanced split, per route (a bf16 weight under requant,
// a w8 weight in the signed layout), per row count, at the deep-K
// projections of the families in tree. Two weights are cycled so no call
// finds its weight warm; candidates are interleaved, the order reversed
// every round, and the median kept. VPIPE_I8_SPLIT_SWEEP=1 runs it;
// VPIPE_I8_SPLIT_SWEEP_SHAPES="N:K,..." and _ROWS="M,..." narrow it.
TEST(gemm_i8, split_rule_sweep) {
  if (std::getenv("VPIPE_I8_SPLIT_SWEEP") == nullptr) {
    std::printf("[gemm_i8sw] VPIPE_I8_SPLIT_SWEEP unset -- skip\n");
    return;
  }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ::setenv("VPIPE_I8_GEMM_MIN_M", "64", 1);
  vpipe::genai::I8GemmContext ctx(mc, /*want=*/true, /*bf16=*/true);
  ::unsetenv("VPIPE_I8_GEMM_MIN_M");
  if (!ctx.enabled() || !ctx.split_available() ||
      !ctx.signed_available(8)) {
    std::printf("[gemm_i8sw] unavailable -- skip\n");
    return;
  }
  auto parse = [](const char* e, int arity) {
    std::vector<std::vector<int>> out;
    const std::string s = e;
    for (std::size_t b = 0; b < s.size();) {
      std::size_t end = s.find(',', b);
      if (end == std::string::npos) { end = s.size(); }
      std::vector<int> v;
      std::size_t p = b;
      while (p < end) {
        std::size_t q = s.find(':', p);
        if (q == std::string::npos || q > end) { q = end; }
        v.push_back(std::atoi(s.substr(p, q - p).c_str()));
        p = q + 1;
      }
      if ((int)v.size() == arity) { out.push_back(v); }
      b = end + 1;
    }
    return out;
  };
  std::vector<std::vector<int>> shapes = {
      {3072, 9216}, {3072, 12288}, {4096, 12288}, {3840, 10240},
      {4096, 16384}, {7168, 14336}, {3072, 3072}, {12288, 4096}};
  std::vector<int> rows = {512, 1024, 2048, 4096, 8192};
  if (const char* e = std::getenv("VPIPE_I8_SPLIT_SWEEP_SHAPES")) {
    shapes = parse(e, 2);
  }
  if (const char* e = std::getenv("VPIPE_I8_SPLIT_SWEEP_ROWS")) {
    rows.clear();
    for (const auto& v : parse(e, 1)) { rows.push_back(v[0]); }
  }
  constexpr int kWeights = 2, kIters = 4, kRounds = 3;
  for (const auto& sh : shapes) {
    const int N = sh[0], K = sh[1], G = K / 64;
    std::vector<SharedBuffer> wd, wc, ws, wz;
    for (int i = 0; i < kWeights; ++i) {
      wd.push_back(mc->make_shared_buffer((std::size_t)N * K * 2));
      wc.push_back(mc->make_shared_buffer((std::size_t)N * K));
      ws.push_back(mc->make_shared_buffer((std::size_t)N * G * 2));
      wz.push_back(mc->make_shared_buffer((std::size_t)N * G * 2));
      if (wd.back().empty() || wc.back().empty()) { break; }
      std::memset(wd.back().contents(), 0x3c, wd.back().byte_size());
      std::memset(wc.back().contents(), 0x1a + i, wc.back().byte_size());
      std::memset(ws.back().contents(), 0x10, ws.back().byte_size());
      std::memset(wz.back().contents(), 0x08, wz.back().byte_size());
    }
    if ((int)wd.size() != kWeights || wd.back().empty() ||
        wc.back().empty()) {
      continue;
    }
    for (int M : rows) {
      SharedBuffer x = mc->make_shared_buffer((std::size_t)M * K * 2);
      SharedBuffer y = mc->make_shared_buffer((std::size_t)M * N * 2);
      if (x.empty() || y.empty()) { continue; }
      std::memset(x.contents(), 0x3c, x.byte_size());
      std::vector<int> cands{0};
      for (int S : ctx.split_candidates(M, N, K)) {
        if (S <= 4) { cands.push_back(S); }
      }
      if (cands.size() < 2) { continue; }
      for (int route = 0; route < 2; ++route) {
        auto sample = [&](int S) {
          ctx.bypass_split = (S == 0);
          ctx.force_splits = S;
          CommandStream st = mc->make_command_stream();
          const auto t0 = std::chrono::steady_clock::now();
          { ComputeEncoder e = st.begin_compute();
            for (int i = 0; i < kIters; ++i) {
              const int w = i % kWeights;
              if (route == 0) {
                EXPECT_TRUE(ctx.gemm(e, x, 0, wd[w], y, 0, M, N, K));
              } else {
                EXPECT_TRUE(ctx.gemm_signed(e, x, 0, wc[w], ws[w], wz[w],
                                            8, y, 0, M, N, K));
              }
            } }
          st.commit().wait();
          ctx.bypass_split = false;
          ctx.force_splits = 0;
          return std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0).count() / kIters;
        };
        for (int S : cands) { (void)sample(S); }
        std::vector<std::vector<double>> t(cands.size());
        for (int r = 0; r < kRounds; ++r) {
          for (std::size_t k = 0; k < cands.size(); ++k) {
            const std::size_t c = (r % 2 == 0) ? k : cands.size() - 1 - k;
            t[c].push_back(sample(cands[c]));
          }
        }
        std::string line;
        double t0 = 0.0, best = 0.0;
        int best_s = 0;
        for (std::size_t c = 0; c < cands.size(); ++c) {
          std::sort(t[c].begin(), t[c].end());
          const double m = t[c][t[c].size() / 2];
          if (c == 0) { t0 = m; best = m; }
          if (m < best) { best = m; best_s = cands[c]; }
          char buf[64];
          std::snprintf(buf, sizeof buf, " S%d %.2fx", cands[c], t0 / m);
          line += buf;
        }
        std::printf("[gemm_i8sw] %-7s M=%-5d N=%-5d K=%-5d S0 %7.2f ms "
                    "|%s | best S=%d\n", route == 0 ? "requant" : "signed",
                    M, N, K, t0, line.c_str(), best_s);
      }
    }
  }
}

// W8 AGAINST BF16, EVERY ROUTE A DiT CAN TAKE, as a bf16 family pays it
// per projection: the dense matmul2d tile it would pick by K, the same
// weight under i8 (act quant + weight requant + int8 GEMM), and a w8g64
// pack four ways -- dequant then matmul2d (i8 off), dequant then the
// requant route (i8 on, a family with no native call), the native route
// (act quant + code sums + u8 GEMM), and the steel qmm (no matrix
// cores). Several weights are cycled so no call finds its weight in the
// cache from the call before, as no projection does in a model. Arms are
// interleaved, the order reversed every round, and the median kept: this
// box throttles. VPIPE_I8_W8_SHAPES="M:N:K,..." times other shapes.
TEST(gemm_i8, w8_against_bf16_routes) {
  if (std::getenv("VPIPE_I8_W8_BENCH") == nullptr) {
    std::printf("[gemm_i8w8] VPIPE_I8_W8_BENCH unset -- skip\n");
    return;
  }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib_mma = mc->load_library("dense_gemm_mma_bf16");
  ComputeLibrary lib_dq = mc->load_library("affine_dequant_bf16");
  ComputeLibrary lib_qmm = mc->load_library("affine_qmm_steel_bf16");
  ComputeFunction f_n128 = lib_mma.function("dense_gemm_mma_t_n128_f16");
  ComputeFunction f_deep = lib_mma.function("dense_gemm_mma_t_n128x256_f16");
  ComputeFunction f_tn2 =
      lib_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");
  ComputeFunction f_dq8 = lib_dq.function("affine_dequant_w8g64");
  ComputeFunction f_qmm8 = lib_qmm.function("affine_qmm_steel_w8g64");
  vpipe::genai::I8GemmContext ctx(mc, /*want=*/true, /*bf16=*/true);
  if (!f_n128.valid() || !f_deep.valid() || !f_tn2.valid() ||
      !f_dq8.valid() || !f_qmm8.valid() || !ctx.enabled() ||
      !ctx.native_available(8) || !ctx.signed_available(8)) {
    std::printf("[gemm_i8w8] a kernel is missing -- skip\n");
    return;
  }
  auto to_bf16 = [](float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (std::uint16_t)(u >> 16);
  };
  auto from_bf16 = [](std::uint16_t h) {
    const std::uint32_t u = (std::uint32_t)h << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  // ACCURACY FIRST, on random operands: each w8 route and bf16 + i8 (on
  // the dequantized weight, which is what a bf16 checkpoint of these
  // values would hold) against f64 truth.
  {
    const int M = 256, N = 512, K = 1024, G = K / 64;
    std::mt19937 rng(77u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> qd(0, 255);
    std::uniform_real_distribution<float> sd(0.003f, 0.02f);
    std::uniform_real_distribution<float> bd(-0.1f, 0.1f);
    std::vector<std::uint16_t> xh((std::size_t)M * K), sh((std::size_t)N * G),
        bh((std::size_t)N * G), b2h((std::size_t)N * G),
        dh((std::size_t)N * K);
    std::vector<std::uint8_t> q((std::size_t)N * K), qs((std::size_t)N * K);
    for (auto& v : xh) { v = to_bf16(nd(rng) * 0.5f); }
    for (std::size_t i = 0; i < sh.size(); ++i) {
      sh[i] = to_bf16(sd(rng));
      bh[i] = to_bf16(bd(rng));
      b2h[i] = to_bf16(from_bf16(bh[i]) + 128.0f * from_bf16(sh[i]));
    }
    for (std::size_t i = 0; i < q.size(); ++i) {
      q[i] = (std::uint8_t)qd(rng);
      qs[i] = (std::uint8_t)(q[i] ^ 0x80u);
      const std::size_t gi = (i / K) * G + (i % K) / 64;
      dh[i] = to_bf16(from_bf16(sh[gi]) * (float)q[i] + from_bf16(bh[gi]));
    }
    auto up = [&](const void* p, std::size_t n) {
      SharedBuffer b = mc->make_shared_buffer(n);
      std::memcpy(b.contents(), p, n);
      return b;
    };
    SharedBuffer xb = up(xh.data(), xh.size() * 2);
    SharedBuffer cb = up(q.data(), q.size());
    SharedBuffer csb = up(qs.data(), qs.size());
    SharedBuffer sb = up(sh.data(), sh.size() * 2);
    SharedBuffer bb = up(bh.data(), bh.size() * 2);
    SharedBuffer b2b = up(b2h.data(), b2h.size() * 2);
    SharedBuffer db = up(dh.data(), dh.size() * 2);
    SharedBuffer wdq0 = mc->make_shared_buffer((std::size_t)N * K * 2);
    // The GPU's conversion, checked against the CPU's below.
    SharedBuffer gcs = mc->make_shared_buffer(q.size());
    SharedBuffer gzb = mc->make_shared_buffer(b2h.size() * 2);
    const std::size_t yb = (std::size_t)M * N * 2;
    SharedBuffer y_nat = mc->make_shared_buffer(yb);
    SharedBuffer y_sig = mc->make_shared_buffer(yb);
    SharedBuffer y_rq = mc->make_shared_buffer(yb);
    SharedBuffer y_bf = mc->make_shared_buffer(yb);
    ::setenv("VPIPE_I8_GEMM_MIN_M", "64", 1);
    vpipe::genai::I8GemmContext small(mc, /*want=*/true, /*bf16=*/true);
    ::unsetenv("VPIPE_I8_GEMM_MIN_M");
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder e = st.begin_compute();
        EXPECT_TRUE(small.gemm_affine(e, xb, 0, cb, sb, bb, 8, y_nat, 0, M,
                                      N, K));
        EXPECT_TRUE(small.encode_to_signed(e, cb, sb, bb, gcs, gzb, 8, N,
                                           K));
        EXPECT_TRUE(small.gemm_signed(e, xb, 0, csb, sb, b2b, 8, y_sig, 0,
                                      M, N, K));
        e.set_function(f_dq8);
        e.set_buffer(0, cb); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, wdq0);
        e.set_constant(4, K); e.set_constant(5, N);
        e.dispatch({(unsigned)(K / 4), (unsigned)N, 1}, {64, 1, 1});
        EXPECT_TRUE(small.gemm(e, xb, 0, wdq0, y_rq, 0, M, N, K));
        EXPECT_TRUE(small.gemm(e, xb, 0, db, y_bf, 0, M, N, K)); }
      st.commit().wait(); }
    double nt = 0, e_nat = 0, e_sig = 0, e_rq = 0, e_bf = 0;
    const auto* pn = static_cast<const std::uint16_t*>(y_nat.contents());
    const auto* ps = static_cast<const std::uint16_t*>(y_sig.contents());
    const auto* pr = static_cast<const std::uint16_t*>(y_rq.contents());
    const auto* pb = static_cast<const std::uint16_t*>(y_bf.contents());
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        double t = 0.0;
        for (int k = 0; k < K; ++k) {
          const std::size_t gi = (std::size_t)n * G + k / 64;
          t += (double)from_bf16(xh[(std::size_t)m * K + k]) *
               ((double)from_bf16(sh[gi]) * q[(std::size_t)n * K + k] +
                (double)from_bf16(bh[gi]));
        }
        const std::size_t o = (std::size_t)m * N + n;
        nt += t * t;
        e_nat += std::pow(from_bf16(pn[o]) - t, 2);
        e_sig += std::pow(from_bf16(ps[o]) - t, 2);
        e_rq += std::pow(from_bf16(pr[o]) - t, 2);
        e_bf += std::pow(from_bf16(pb[o]) - t, 2);
      }
    }
    EXPECT_TRUE(std::memcmp(gcs.contents(), qs.data(), qs.size()) == 0);
    EXPECT_TRUE(std::memcmp(gzb.contents(), b2h.data(), b2h.size() * 2) ==
                0);
    const double r_nat = std::sqrt(e_nat / nt), r_sig = std::sqrt(e_sig / nt);
    const double r_rq = std::sqrt(e_rq / nt), r_bf = std::sqrt(e_bf / nt);
    std::printf("[gemm_i8w8] rel-L2 vs f64 at %dx%dx%d: native %.3e  "
                "signed %.3e  dequant+requant %.3e  bf16+i8 %.3e\n", M, N, K,
                r_nat, r_sig, r_rq, r_bf);
    // The signed route holds the same codes exactly; only b' rounds.
    EXPECT_TRUE(r_sig < 1.25 * r_nat);
  }
  struct Shape { int M, N, K; };
  std::vector<Shape> shapes = {
      {4096, 3072, 3072},  {4096, 18432, 3072}, {4096, 3072, 9216},
      {4096, 3840, 3840},  {4096, 10240, 3840}, {4096, 3840, 10240},
      {4096, 12288, 4096}, {4096, 4096, 12288},
  };
  if (const char* e = std::getenv("VPIPE_I8_W8_SHAPES")) {
    shapes.clear();
    const std::string want = e;
    for (std::size_t b = 0; b < want.size();) {
      std::size_t end = want.find(',', b);
      if (end == std::string::npos) { end = want.size(); }
      int m = 0, n = 0, k = 0;
      if (std::sscanf(want.substr(b, end - b).c_str(), "%d:%d:%d", &m, &n,
                      &k) == 3) {
        shapes.push_back({m, n, k});
      }
      b = end + 1;
    }
  }
  auto bf16 = [](float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return (std::uint16_t)(u >> 16);
  };
  constexpr int kWeights = 4;   // cycled, so each call reads cold
  constexpr int kIters = 8;     // calls a sample
  constexpr int kRounds = 6;
  for (const Shape& sh : shapes) {
    const int M = sh.M, N = sh.N, K = sh.K, G = K / 64;
    SharedBuffer x = mc->make_shared_buffer((std::size_t)M * K * 2);
    SharedBuffer y = mc->make_shared_buffer((std::size_t)M * N * 2);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)N * K * 2);
    // The per-call arm's conversion targets.
    SharedBuffer pcs = mc->make_shared_buffer((std::size_t)N * K);
    SharedBuffer pzb = mc->make_shared_buffer((std::size_t)N * G * 2);
    std::vector<SharedBuffer> wd, wc, ws, wb, wcs, wb2;
    for (int i = 0; i < kWeights; ++i) {
      wd.push_back(mc->make_shared_buffer((std::size_t)N * K * 2));
      wc.push_back(mc->make_shared_buffer((std::size_t)N * K));
      ws.push_back(mc->make_shared_buffer((std::size_t)N * G * 2));
      wb.push_back(mc->make_shared_buffer((std::size_t)N * G * 2));
      wcs.push_back(mc->make_shared_buffer((std::size_t)N * K));
      wb2.push_back(mc->make_shared_buffer((std::size_t)N * G * 2));
      if (wd.back().empty() || wc.back().empty() || wcs.back().empty()) {
        break;
      }
      auto* pd = static_cast<std::uint16_t*>(wd.back().contents());
      for (std::size_t j = 0; j < (std::size_t)N * K; ++j) {
        pd[j] = bf16(0.01f * (float)((int)((j + i) % 11) - 5));
      }
      std::memset(wc.back().contents(), 0x5a + i, wc.back().byte_size());
      std::memset(wcs.back().contents(), (0x5a + i) ^ 0x80,
                  wcs.back().byte_size());
      auto* ps = static_cast<std::uint16_t*>(ws.back().contents());
      auto* pb = static_cast<std::uint16_t*>(wb.back().contents());
      auto* pb2 = static_cast<std::uint16_t*>(wb2.back().contents());
      for (std::size_t j = 0; j < (std::size_t)N * G; ++j) {
        ps[j] = bf16(0.0005f);
        pb[j] = bf16(-0.05f);
        pb2[j] = bf16(-0.05f + 128.0f * 0.0005f);
      }
    }
    if (x.empty() || y.empty() || wdq.empty() ||
        (int)wd.size() != kWeights || wd.back().empty() ||
        wc.back().empty()) {
      std::printf("[gemm_i8w8] alloc failed at %dx%dx%d -- skip\n", M, N,
                  K);
      continue;
    }
    auto* px = static_cast<std::uint16_t*>(x.contents());
    for (std::size_t j = 0; j < (std::size_t)M * K; ++j) {
      px[j] = bf16(0.25f * (float)((int)(j % 7) - 3));
    }
    // The family's own tile choice by reduction depth (FLUX.3's).
    const ComputeFunction* fmma = &f_deep;
    int RN = 256;
    if (K < 6144) { fmma = &f_n128; RN = 128; }
    else if (K < 12288) { fmma = &f_tn2; RN = 512; }
    auto mma = [&](ComputeEncoder& e, const SharedBuffer& w) {
      e.set_function(*fmma);
      e.set_buffer(0, x); e.set_buffer(1, w); e.set_buffer(2, w);
      e.set_buffer(3, y);
      e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
      e.set_constant(7, 0);
      e.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    };
    auto dequant = [&](ComputeEncoder& e, int i) {
      e.set_function(f_dq8);
      e.set_buffer(0, wc[i]); e.set_buffer(1, ws[i]); e.set_buffer(2, wb[i]);
      e.set_buffer(3, wdq);
      e.set_constant(4, K); e.set_constant(5, N);
      e.dispatch({(unsigned)(K / 4), (unsigned)N, 1}, {64, 1, 1});
    };
    struct Arm { const char* name; std::function<void(ComputeEncoder&, int)>
                 run; };
    std::vector<Arm> arms = {
        {"bf16 matmul2d", [&](ComputeEncoder& e, int i) { mma(e, wd[i]); }},
        {"bf16 + i8",
         [&](ComputeEncoder& e, int i) {
           EXPECT_TRUE(ctx.gemm(e, x, 0, wd[i], y, 0, M, N, K));
         }},
        {"w8 dequant+matmul2d",
         [&](ComputeEncoder& e, int i) { dequant(e, i); mma(e, wdq); }},
        {"w8 dequant+requant",
         [&](ComputeEncoder& e, int i) {
           dequant(e, i);
           EXPECT_TRUE(ctx.gemm(e, x, 0, wdq, y, 0, M, N, K));
         }},
        {"w8 native i8",
         [&](ComputeEncoder& e, int i) {
           EXPECT_TRUE(ctx.gemm_affine(e, x, 0, wc[i], ws[i], wb[i], 8, y,
                                       0, M, N, K));
         }},
        {"w8 signed i8",
         [&](ComputeEncoder& e, int i) {
           EXPECT_TRUE(ctx.gemm_signed(e, x, 0, wcs[i], ws[i], wb2[i], 8, y,
                                       0, M, N, K));
         }},
        {"w8 signed per call",
         [&](ComputeEncoder& e, int i) {
           EXPECT_TRUE(ctx.encode_to_signed(e, wc[i], ws[i], wb[i], pcs, pzb,
                                            8, N, K));
           EXPECT_TRUE(ctx.gemm_signed(e, x, 0, pcs, ws[i], pzb, 8, y, 0, M,
                                       N, K));
         }},
        {"w8 qmm steel",
         [&](ComputeEncoder& e, int i) {
           e.set_function(f_qmm8);
           e.set_buffer(0, wc[i]); e.set_buffer(1, ws[i]);
           e.set_buffer(2, wb[i]); e.set_buffer(3, x); e.set_buffer(4, y);
           e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, M);
           e.dispatch({(unsigned)(((N + 31) / 32) * 32),
                       (unsigned)(((M + 31) / 32) * 2), 2u}, {32, 2, 2});
         }},
    };
    auto sample = [&](const Arm& a) {
      CommandStream st = mc->make_command_stream();
      const auto t0 = std::chrono::steady_clock::now();
      { ComputeEncoder e = st.begin_compute();
        for (int i = 0; i < kIters; ++i) { a.run(e, i % kWeights); } }
      st.commit().wait();
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0).count() / kIters;
    };
    for (const Arm& a : arms) { (void)sample(a); }   // warm, build scratch
    std::vector<std::vector<double>> t(arms.size());
    for (int r = 0; r < kRounds; ++r) {
      for (std::size_t k = 0; k < arms.size(); ++k) {
        const std::size_t a = (r % 2 == 0) ? k : arms.size() - 1 - k;
        t[a].push_back(sample(arms[a]));
      }
    }
    std::vector<double> med(arms.size());
    for (std::size_t a = 0; a < arms.size(); ++a) {
      std::sort(t[a].begin(), t[a].end());
      med[a] = t[a][t[a].size() / 2];
    }
    const double gflop = 2.0 * M * (double)N * K * 1e-9;
    std::printf("[gemm_i8w8] M=%d N=%d K=%d (%.1f GFLOP)\n", M, N, K, gflop);
    for (std::size_t a = 0; a < arms.size(); ++a) {
      std::printf("[gemm_i8w8]   %-20s %7.2f ms  %5.2f TOP/s  "
                  "%.2fx bf16  %.2fx bf16+i8\n", arms[a].name, med[a],
                  gflop / med[a], med[0] / med[a], med[1] / med[a]);
    }
  }
}

// WHAT A FINER GROUP BUYS, and on which data. Both operands quantized
// per group at 32, 64 and 512 on the CPU (amax), the register kernels
// run on the GPU, rel-L2 against f64 truth -- on gaussian data, where
// per-group scales barely matter, and on data with a few OUTLIER
// channels (x scaled 32x on 1 channel in 64), which is what real
// activations look like and where a group's width decides how many
// neighbours an outlier's scale crushes.
TEST(gemm_i8, group_quality) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("dense_gemm_mma");
  struct Arm { const char* name; int kc; ComputeFunction f; };
  std::vector<Arm> arms;
  auto add = [&](const char* name, int kc, const char* fn) {
    Arm a{name, kc, lib.function(fn)};
    arms.push_back(std::move(a));
  };
  add("g32rs", 32, "gemm_i8i8_sc_f16_n64_g32rs");
  add("g64rs", 64, "gemm_i8i8_sc_f16_n64_g64rs");
  add("g128rs", 128, "gemm_i8i8_sc_f16_n64_g128rs");
  add("g512rs", 512, "gemm_i8i8_sc_f16_n64_g512rs");
  for (const Arm& a : arms) { if (!a.f.valid()) { return; } }

  const int M = 256, N = 512, K = 2048;
  for (int outliers : {0, 1}) {
    std::mt19937 rng(9u + (unsigned)outliers);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<_Float16> x((std::size_t)M * K), w((std::size_t)N * K);
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        float v = nd(rng) * 0.5f;
        if (outliers && (k % 64) == 17) { v *= 32.0f; }
        x[(std::size_t)m * K + k] = (_Float16)v;
      }
    }
    for (auto& v : w) { v = (_Float16)(nd(rng) * 0.05f); }
    std::vector<double> truth((std::size_t)M * N, 0.0);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          acc += (double)x[(std::size_t)m * K + k] *
                 (double)w[(std::size_t)n * K + k];
        }
        truth[(std::size_t)m * N + n] = acc;
      }
    }
    std::printf("[gemm_i8q] %s (%dx%dx%d)\n",
                outliers ? "outlier channels (1 in 64, x32)" : "gaussian",
                M, N, K);
    for (const Arm& a : arms) {
      const int kc = a.kc, G = K / kc;
      std::vector<std::int8_t> xq((std::size_t)M * K), wq((std::size_t)N * K);
      std::vector<_Float16> xs((std::size_t)M * G), ws((std::size_t)N * G);
      auto quant = [&](const std::vector<_Float16>& src, int rows,
                       std::vector<std::int8_t>* q, std::vector<_Float16>* sc) {
        for (int r = 0; r < rows; ++r) {
          for (int g = 0; g < G; ++g) {
            float am = 0.0f;
            for (int k = g * kc; k < (g + 1) * kc; ++k) {
              am = std::max(am, std::fabs((float)src[(std::size_t)r * K + k]));
            }
            const float inv = am > 0 ? 127.0f / am : 0.0f;
            (*sc)[(std::size_t)r * G + g] = (_Float16)(am / 127.0f);
            for (int k = g * kc; k < (g + 1) * kc; ++k) {
              const float v = std::rint((float)src[(std::size_t)r * K + k] * inv);
              (*q)[(std::size_t)r * K + k] =
                  (std::int8_t)std::max(-127.0f, std::min(127.0f, v));
            }
          }
        }
      };
      quant(x, M, &xq, &xs);
      quant(w, N, &wq, &ws);
      SharedBuffer xqb = mc->make_shared_buffer(xq.size());
      SharedBuffer wqb = mc->make_shared_buffer(wq.size());
      SharedBuffer xsb = mc->make_shared_buffer(xs.size() * 2);
      SharedBuffer wsb = mc->make_shared_buffer(ws.size() * 2);
      SharedBuffer yb = mc->make_shared_buffer((std::size_t)M * N * 2);
      if (yb.empty()) { return; }
      std::memcpy(xqb.contents(), xq.data(), xq.size());
      std::memcpy(wqb.contents(), wq.data(), wq.size());
      std::memcpy(xsb.contents(), xs.data(), xs.size() * 2);
      std::memcpy(wsb.contents(), ws.data(), ws.size() * 2);
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          e.set_function(a.f);
          e.set_buffer(0, xqb); e.set_buffer(1, wqb);
          e.set_buffer(2, xsb); e.set_buffer(3, wsb);
          e.set_buffer(4, yb);
          e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, M);
          e.set_buffer(8, wsb); e.set_constant(9, 0);
          e.dispatch({(unsigned)((N / 64) * 128), (unsigned)(M / 64), 1},
                     {128, 1, 1}); }
        st.commit().wait(); }
      const auto* py = static_cast<const _Float16*>(yb.contents());
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
        const double d = (double)py[i] - truth[i];
        num += d * d;
        den += truth[i] * truth[i];
      }
      const double r = std::sqrt(num / den);
      std::printf("[gemm_i8q]   %-7s rel-L2 %.4e\n", a.name, r);
      EXPECT_TRUE(std::isfinite(r) && r < 0.1);
    }
  }
}

// GROUP SIZE AGAINST RATE, and the two costs it hides.
//
// In the per-group kernel the group width IS the chunk depth: one
// matmul2d per group, drained and scaled before the next. So a finer
// group makes the calls shallower AND multiplies the drains, and those
// are different costs that a single sweep reports as one number.
//
// This separates them. The `kacc` arms chunk K at the same widths with
// RAW i32 accumulation -- no scales, no drain -- so they price the DEPTH
// alone. The `g` arms are the same widths with per-group scales, so the
// gap between the two rows at one width is the GROUP TAX at that width.
//
// The question this answers: is 64 a usable group for an int8 GEMM, or
// does the depth fall off before it.
TEST(gemm_i8, group_size) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib_mma = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_dq = mc->load_library("affine_dequant");
  ComputeFunction f_q = lib_dq.function("quant_f16_i8_row_g512");
  // grouped: the per-group twins. shift: the register twins that read
  // int8 EXPONENTS at 2/3 instead of f16 scales -- and whose shift path
  // is data-dependent, so they get exponents that actually vary.
  struct Arm { const char* name = nullptr; int kc = 0;
               bool grouped = false; bool shift = false; bool f16 = false;
               ComputeFunction fn; };
  // ComputeFunction is move-only, so the arms are pushed rather than
  // brace-initialised.
  std::vector<Arm> arms;
  auto add = [&](const char* name, int kc, bool grouped, bool shift,
                 const char* fn) {
    Arm a;
    a.name = name; a.kc = kc; a.grouped = grouped; a.shift = shift;
    a.fn = lib_mma.function(fn);
    arms.push_back(std::move(a));
  };
  add("kacc-512", 512, false, false, "gemm_i8i8_sc_f16_n64_kacc");
  add("kacc-256", 256, false, false, "gemm_i8i8_sc_f16_n64_kacc_k256");
  add("kacc-128", 128, false, false, "gemm_i8i8_sc_f16_n64_kacc_k128");
  add("kacc-64",   64, false, false, "gemm_i8i8_sc_f16_n64_kacc_k64");
  add("kacc-32",   32, false, false, "gemm_i8i8_sc_f16_n64_kacc_k32");
  add("g512",     512, true,  false, "gemm_i8i8_sc_f16_n64_g512");
  add("g128",     128, true,  false, "gemm_i8i8_sc_f16_n64_g128");
  add("g64",       64, true,  false, "gemm_i8i8_sc_f16_n64_g64");
  add("g32",       32, true,  false, "gemm_i8i8_sc_f16_n64_g32");
  // Register-resident twins: float accumulate (r) and shift-aligned
  // integer accumulate (ri), no drain between groups.
  add("g512r",    512, false, false, "gemm_i8i8_sc_f16_n64_g512r");
  add("g512ri",   512, false, true,  "gemm_i8i8_sc_f16_n64_g512ri");
  add("g128r",    128, false, false, "gemm_i8i8_sc_f16_n64_g128r");
  add("g128ri",   128, false, true,  "gemm_i8i8_sc_f16_n64_g128ri");
  add("g64r",      64, false, false, "gemm_i8i8_sc_f16_n64_g64r");
  add("g64ri",     64, false, true,  "gemm_i8i8_sc_f16_n64_g64ri");
  // The cooperative-destination ceiling (no per-group work at all).
  add("kaccr-512", 512, false, false, "gemm_i8i8_sc_f16_n64_kaccr_k512");
  add("kaccr-64",   64, false, false, "gemm_i8i8_sc_f16_n64_kaccr_k64");
  add("kaccr-32",   32, false, false, "gemm_i8i8_sc_f16_n64_kaccr_k32");
  add("kaccr-128", 128, false, false, "gemm_i8i8_sc_f16_n64_kaccr_k128");
  // Staged scales (one tgmem strip per 8 groups), and the per-row-
  // activation twins that stage only the weight side.
  add("g512rs",   512, false, false, "gemm_i8i8_sc_f16_n64_g512rs");
  add("g512rsi",  512, false, true,  "gemm_i8i8_sc_f16_n64_g512rsi");
  add("g64rs",     64, false, false, "gemm_i8i8_sc_f16_n64_g64rs");
  add("g64rsi",    64, false, true,  "gemm_i8i8_sc_f16_n64_g64rsi");
  add("g32rs",     32, false, false, "gemm_i8i8_sc_f16_n64_g32rs");
  add("g128rs",   128, false, false, "gemm_i8i8_sc_f16_n64_g128rs");
  add("w64rs",     64, false, false, "gemm_i8i8_sc_f16_n64_w64rs");
  add("w64rsi",    64, false, true,  "gemm_i8i8_sc_f16_n64_w64rsi");
  add("w64rsf",    64, false, true,  "gemm_i8i8_sc_f16_n64_w64rsf");
  // The f16 twin of the kacc loop -- same tile, same chunking, same
  // accumulate mode -- so the int8 rates have a like-for-like dtype
  // baseline in the same run.
  add("f16-k512", 512, false, false, "dense_gemm_mma_kacc_k512_f16");
  arms.back().f16 = true;
  if (!f_q.valid()) { return; }
  for (const Arm& a : arms) {
    if (!a.fn.valid()) {
      std::printf("[gemm_i8g] %s unavailable -- skip\n", a.name);
      return;
    }
  }

  auto sweep = [&](int M, int N, int K, int iters) {
    std::mt19937 rng(31u + (unsigned)(M + N + K));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<_Float16> x((std::size_t)M * K);
    for (auto& v : x) { v = (_Float16)(nd(rng) * 0.5f); }
    SharedBuffer xb = mc->make_shared_buffer(x.size() * 2);
    SharedBuffer xq = mc->make_shared_buffer((std::size_t)M * K);
    SharedBuffer wq = mc->make_shared_buffer((std::size_t)N * K);
    SharedBuffer y  = mc->make_shared_buffer((std::size_t)M * N * 2);
    // Scales at the FINEST width, so one buffer serves every arm: an arm
    // at a coarser group reads a prefix of it. Only the TRAFFIC has to be
    // right here -- an int8 GEMM's time does not depend on its data, and
    // correctness at each width is gemm_i8.k512_chunked's job.
    const std::size_t gmax = (std::size_t)K / 32;
    SharedBuffer as = mc->make_shared_buffer((std::size_t)M * gmax * 2);
    SharedBuffer ws = mc->make_shared_buffer((std::size_t)N * gmax * 2);
    SharedBuffer ea = mc->make_shared_buffer((std::size_t)M * gmax);
    SharedBuffer ew = mc->make_shared_buffer((std::size_t)N * gmax);
    SharedBuffer wh = mc->make_shared_buffer((std::size_t)N * K * 2);
    if (y.empty() || ws.empty() || wq.empty() || ew.empty() || wh.empty()) {
      std::printf("[gemm_i8g] alloc failed -- skip\n");
      return;
    }
    std::memcpy(xb.contents(), x.data(), x.size() * 2);
    std::memset(wq.contents(), 1, wq.byte_size());
    std::memset(wh.contents(), 0, wh.byte_size());
    {
      // Exponents that move by a few binades group to group, which is
      // what real activations do and what makes the shift path run its
      // align branches rather than the cheap equal-exponent one.
      std::uniform_int_distribution<int> ud(-24, -18);
      auto* pa = static_cast<std::int8_t*>(ea.contents());
      for (std::size_t i = 0; i < ea.byte_size(); ++i) {
        pa[i] = (std::int8_t)ud(rng);
      }
      auto* pw = static_cast<std::int8_t*>(ew.contents());
      for (std::size_t i = 0; i < ew.byte_size(); ++i) {
        pw[i] = (std::int8_t)ud(rng);
      }
    }
    for (std::size_t i = 0; i < as.byte_size() / 2; ++i) {
      static_cast<_Float16*>(as.contents())[i] = (_Float16)0.01f;
    }
    for (std::size_t i = 0; i < ws.byte_size() / 2; ++i) {
      static_cast<_Float16*>(ws.contents())[i] = (_Float16)0.01f;
    }
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder e = st.begin_compute();
        e.set_function(f_q);
        e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, as);
        e.set_constant(3, K);
        e.dispatch({256, (unsigned)M, 1}, {256, 1, 1}); }
      st.commit().wait(); }

    const double gflop = 2.0 * (double)M * N * K * 1e-9;
    std::printf("[gemm_i8g] M=%d N=%d K=%d\n", M, N, K);
    double base = 0.0;
    for (const Arm& a : arms) {
      if (K % a.kc != 0) { continue; }
      auto one = [&](ComputeEncoder& e) {
        e.set_function(a.fn);
        if (a.f16) {
          e.set_buffer(0, xb); e.set_buffer(1, wh); e.set_buffer(2, ws);
          e.set_buffer(3, y);
          e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
          e.set_constant(7, 0);
          e.dispatch({(unsigned)(((N + 63) / 64) * 128),
                      (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
          return;
        }
        e.set_buffer(0, xq); e.set_buffer(1, wq);
        if (a.shift) {
          e.set_buffer(2, ea); e.set_buffer(3, ew);
        } else {
          e.set_buffer(2, as); e.set_buffer(3, ws);
        }
        e.set_buffer(4, y);
        e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, M);
        // Bias slots, bound for every arm: the drained and staged twins
        // read has_bias, the rest declare no buffer there and ignore it.
        e.set_buffer(8, ws); e.set_constant(9, 0);
        e.dispatch({(unsigned)(((N + 63) / 64) * 128),
                    (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
      };
      for (int w = 0; w < 2; ++w) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute(); one(e); }
        st.commit().wait();
      }
      const auto t0 = std::chrono::steady_clock::now();
      { CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          for (int i = 0; i < iters; ++i) { one(e); } }
        st.commit().wait(); }
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
      const double tops = gflop * iters / ms;
      if (base == 0.0) { base = tops; }
      std::printf("[gemm_i8g]   %-9s %6.1f TOP/s  %.2fx of kacc-512\n",
                  a.name, tops, tops / base);
    }
  };
  // The block-projection shape the int8 path was built for, and a deep-K
  // one where the drains compete with a longer weight stream.
  sweep(4096, 12288, 4096, 4);
  sweep(4096, 4096, 12288, 4);
}

// Register-resident per-group accumulation (gemm_i8i8_sc_f16_n64_g64r /
// _g64ri): the matmul2d partial stays in a cooperative tensor and is
// folded into per-thread registers, with no threadgroup drain between
// groups. (1) probe the destination layout the kernel's CAP assumes;
// (2) oracle the shift-aligned twin against an exact CPU replica of the
// accumulate on the GPU's own quantized operands; (3) quality of both
// register twins against f32, and the float twin against the tgmem
// kernel it re-implements. The perf curve is gemm_i8.group_size.
TEST(gemm_i8, g64_register_acc) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib_mma = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_dq = mc->load_library("affine_dequant");
  ComputeFunction f_probe = lib_mma.function("gemm_i8i8_coop_probe");
  ComputeFunction f_ri = lib_mma.function("gemm_i8i8_sc_f16_n64_g64ri");
  ComputeFunction f_r = lib_mma.function("gemm_i8i8_sc_f16_n64_g64r");
  ComputeFunction f_t = lib_mma.function("gemm_i8i8_sc_f16_n64_g64");
  ComputeFunction f_qe = lib_dq.function("quant_f16_i8_g64_bfpe");
  ComputeFunction f_qa = lib_dq.function("quant_f16_i8_g64_amax");
  if (!f_probe.valid() || !f_ri.valid() || !f_r.valid() || !f_t.valid() ||
      !f_qe.valid() || !f_qa.valid()) {
    std::printf("[gemm_i8r] kernels unavailable -- skip\n");
    return;
  }

  // (1) The layout probe: 32 valid elements per thread, and the 128
  // threads' coordinates cover the 64x64 tile exactly once. Region 2 is
  // the f16 x f16 -> f32 destination of the same tile, compared element
  // by element with the int8 one.
  {
    const std::size_t n_out = 2 * (1 + 128 * 64 * 3);
    SharedBuffer ob = mc->make_shared_buffer(n_out * sizeof(int));
    std::memset(ob.contents(), 0, ob.byte_size());
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder e = st.begin_compute();
        e.set_function(f_probe);
        e.set_buffer(0, ob);
        e.dispatch({128, 1, 1}, {128, 1, 1}); }
      st.commit().wait(); }
    const int* o = static_cast<const int*>(ob.contents());
    const int cap = o[0];
    std::vector<int> cover(64 * 64, 0);
    int valid = 0, bad = 0;
    for (int t = 0; t < 128; ++t) {
      for (int i = 0; i < 64 && i < cap; ++i) {
        const int* e = o + 1 + (t * 64 + i) * 3;
        if (e[0] != 1) { continue; }
        ++valid;
        if (e[1] < 0 || e[1] >= 64 || e[2] < 0 || e[2] >= 64) {
          ++bad;
          continue;
        }
        cover[e[2] * 64 + e[1]]++;
      }
    }
    int once = 0;
    for (int c : cover) { once += (c == 1) ? 1 : 0; }
    // Thread 0's first few elements, so a layout change is legible.
    std::printf("[gemm_i8r] coop layout: capacity %d, %d valid, %d/4096 "
                "cells covered once%s; t0: (%d,%d) (%d,%d) (%d,%d) (%d,%d)\n",
                cap, valid, once, bad ? " OUT-OF-TILE" : "",
                o[1 + 0 * 3 + 1], o[1 + 0 * 3 + 2], o[1 + 1 * 3 + 1],
                o[1 + 1 * 3 + 2], o[1 + 2 * 3 + 1], o[1 + 2 * 3 + 2],
                o[1 + 3 * 3 + 1], o[1 + 3 * 3 + 2]);
    EXPECT_TRUE(cap == 32);
    EXPECT_TRUE(valid == 128 * 32);
    EXPECT_TRUE(once == 64 * 64 && bad == 0);
    if (cap != 32 || once != 64 * 64) { return; }
    const int* o2 = o + 1 + 128 * 64 * 3;
    int same = 0, total = 0;
    for (int t = 0; t < 128; ++t) {
      for (int i = 0; i < 64 && i < cap && i < o2[0]; ++i) {
        const int* e = o + 1 + (t * 64 + i) * 3;
        const int* f = o2 + 1 + (t * 64 + i) * 3;
        ++total;
        same += (e[0] == f[0] && e[1] == f[1] && e[2] == f[2]) ? 1 : 0;
      }
    }
    std::printf("[gemm_i8r] f16->f32 destination layout: capacity %d, "
                "%d/%d elements at the int8 layout's coordinates\n",
                o2[0], same, total);
    // gemm_u8q_impl adds its post-loop matmul in registers on the
    // strength of this; it silently misplaces every term if it changes.
    EXPECT_TRUE(o2[0] == 32 && same == total && total == 128 * 32);
  }

  ComputeFunction f_rs = lib_mma.function("gemm_i8i8_sc_f16_n64_g64rs");
  ComputeFunction f_rsi = lib_mma.function("gemm_i8i8_sc_f16_n64_g64rsi");
  ComputeFunction f_wrs = lib_mma.function("gemm_i8i8_sc_f16_n64_w64rs");
  ComputeFunction f_wrsi = lib_mma.function("gemm_i8i8_sc_f16_n64_w64rsi");
  ComputeFunction f_wrsf = lib_mma.function("gemm_i8i8_sc_f16_n64_w64rsf");
  if (!f_rs.valid() || !f_rsi.valid() || !f_wrs.valid() || !f_wrsi.valid() ||
      !f_wrsf.valid()) {
    std::printf("[gemm_i8r] staged kernels unavailable -- skip\n");
    return;
  }

  const int M = 256, N = 512, K = 1024, G = K / 64;
  std::mt19937 rng(41u);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<_Float16> x((std::size_t)M * K), w((std::size_t)N * K);
  for (auto& v : x) { v = (_Float16)(nd(rng) * 0.5f); }
  for (auto& v : w) { v = (_Float16)(nd(rng) * 0.05f); }
  // Offline weight quant per (channel, 64-group): pow2 (exponent
  // Emax16 - 21, the activation scheme's mirror) for the shift twins,
  // amax f16 scales for the float twins.
  std::vector<std::int8_t> wqp((std::size_t)N * K), wqa((std::size_t)N * K);
  std::vector<std::int8_t> ewv((std::size_t)N * G);
  std::vector<_Float16> wsg((std::size_t)N * G);
  auto exp_field = [](_Float16 v) {
    std::uint16_t u;
    std::memcpy(&u, &v, 2);
    return (int)((u >> 10) & 0x1F);
  };
  auto q8 = [](float v) {
    return (std::int8_t)std::max(-127.0f, std::min(127.0f, std::rint(v)));
  };
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < G; ++g) {
      int em = 0;
      float am = 0.0f;
      for (int k = g * 64; k < (g + 1) * 64; ++k) {
        em = std::max(em, exp_field(w[(std::size_t)n * K + k]));
        am = std::max(am, std::fabs((float)w[(std::size_t)n * K + k]));
      }
      ewv[(std::size_t)n * G + g] = (std::int8_t)(em - 21);
      wsg[(std::size_t)n * G + g] = (_Float16)(am / 127.0f);
      const float ig = std::ldexp(1.0f, 21 - em);
      const float ia = am > 0 ? 127.0f / am : 0.0f;
      for (int k = g * 64; k < (g + 1) * 64; ++k) {
        const float wv = (float)w[(std::size_t)n * K + k];
        wqp[(std::size_t)n * K + k] = q8(wv * ig);
        wqa[(std::size_t)n * K + k] = q8(wv * ia);
      }
    }
  }
  // Per-ROW activation quant for the w64 twins, on the CPU: amax f16
  // scale and pow2 exponent, one each per token.
  std::vector<std::int8_t> xqr((std::size_t)M * K), xqre((std::size_t)M * K);
  std::vector<std::int8_t> ear(M);
  std::vector<_Float16> asr(M);
  for (int m = 0; m < M; ++m) {
    int em = 0;
    float am = 0.0f;
    for (int k = 0; k < K; ++k) {
      em = std::max(em, exp_field(x[(std::size_t)m * K + k]));
      am = std::max(am, std::fabs((float)x[(std::size_t)m * K + k]));
    }
    ear[m] = (std::int8_t)(em - 21);
    asr[m] = (_Float16)(am / 127.0f);
    const float ig = std::ldexp(1.0f, 21 - em);
    const float ia = am > 0 ? 127.0f / am : 0.0f;
    for (int k = 0; k < K; ++k) {
      const float xv = (float)x[(std::size_t)m * K + k];
      xqre[(std::size_t)m * K + k] = q8(xv * ig);
      xqr[(std::size_t)m * K + k] = q8(xv * ia);
    }
  }

  auto buf = [&](std::size_t bytes) { return mc->make_shared_buffer(bytes); };
  auto upload = [&](SharedBuffer& b, const void* src, std::size_t bytes) {
    std::memcpy(b.contents(), src, bytes);
  };
  SharedBuffer xb = buf(x.size() * 2);
  SharedBuffer xqe = buf((std::size_t)M * K), xqa = buf((std::size_t)M * K);
  SharedBuffer xqrb = buf(xqr.size()), xqreb = buf(xqre.size());
  SharedBuffer eab = buf((std::size_t)M * G), asb = buf((std::size_t)M * G * 2);
  SharedBuffer earb = buf(ear.size()), asrb = buf(asr.size() * 2);
  SharedBuffer wqpb = buf(wqp.size()), wqab = buf(wqa.size());
  SharedBuffer ewb = buf(ewv.size()), wsb = buf(wsg.size() * 2);
  const std::size_t ybytes = (std::size_t)M * N * 2;
  SharedBuffer yri = buf(ybytes), yr = buf(ybytes), yt = buf(ybytes);
  SharedBuffer yrs = buf(ybytes), yrsi = buf(ybytes);
  SharedBuffer ywrs = buf(ybytes), ywrsi = buf(ybytes), ywrsf = buf(ybytes);
  if (ywrsi.empty() || ywrsf.empty() || wqpb.empty() || wqab.empty() || xqreb.empty()) {
    std::printf("[gemm_i8r] alloc failed -- skip\n");
    return;
  }
  upload(xb, x.data(), x.size() * 2);
  upload(xqrb, xqr.data(), xqr.size());
  upload(xqreb, xqre.data(), xqre.size());
  upload(earb, ear.data(), ear.size());
  upload(asrb, asr.data(), asr.size() * 2);
  upload(wqpb, wqp.data(), wqp.size());
  upload(wqab, wqa.data(), wqa.size());
  upload(ewb, ewv.data(), ewv.size());
  upload(wsb, wsg.data(), wsg.size() * 2);

  auto quant = [&](ComputeFunction& fq, SharedBuffer& xqdst,
                   SharedBuffer& scl) {
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder enc = st.begin_compute();
      enc.set_function(fq);
      enc.set_buffer(0, xb); enc.set_buffer(1, xqdst); enc.set_buffer(2, scl);
      enc.set_constant(3, M * K);
      enc.dispatch({(unsigned)(M * K / 2), 1, 1}, {128, 1, 1}); }
    st.commit().wait();
  };
  auto gemm = [&](ComputeFunction& fg, SharedBuffer& xqsrc,
                  SharedBuffer& scl, SharedBuffer& wqx, SharedBuffer& wsx,
                  SharedBuffer& yb, bool bias_slots) {
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder enc = st.begin_compute();
      enc.set_function(fg);
      enc.set_buffer(0, xqsrc); enc.set_buffer(1, wqx);
      enc.set_buffer(2, scl); enc.set_buffer(3, wsx);
      enc.set_buffer(4, yb);
      enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
      // No bias here, but the slots must be bound: the drained and the
      // staged twins both read has_bias.
      (void)bias_slots;
      enc.set_buffer(8, wsx); enc.set_constant(9, 0);
      enc.dispatch({(unsigned)((N / 64) * 128), (unsigned)(M / 64), 1},
                   {128, 1, 1}); }
    st.commit().wait();
  };
  // Each activation quantization gets its own buffer: the replica reads
  // the pow2 one back, so nothing may overwrite it.
  quant(f_qe, xqe, eab);
  quant(f_qa, xqa, asb);
  gemm(f_ri, xqe, eab, wqpb, ewb, yri, false);
  gemm(f_rsi, xqe, eab, wqpb, ewb, yrsi, false);
  gemm(f_r, xqa, asb, wqab, wsb, yr, false);
  gemm(f_rs, xqa, asb, wqab, wsb, yrs, false);
  gemm(f_t, xqa, asb, wqab, wsb, yt, true);
  gemm(f_wrs, xqrb, asrb, wqab, wsb, ywrs, false);
  gemm(f_wrsi, xqreb, earb, wqpb, ewb, ywrsi, false);
  gemm(f_wrsf, xqreb, earb, wqpb, ewb, ywrsf, false);

  // (2) The oracles: the same accumulate on the CPU, on the GPU's own
  // xq/ea (or the CPU's row quant) and the CPU's wq/ew, in the same
  // group order with the same rounded shifts. (3) f32 quality of every
  // arm.
  auto shift_acc = [](std::int64_t& acc, int& ae, std::int64_t p, int eg) {
    const int d = eg - ae;
    if (d > 0) {
      acc = (d >= 31) ? 0 : ((acc + (1ll << (d - 1))) >> d);
      acc += p;
      ae = eg;
    } else if (d < 0) {
      const int dd = -d;
      p = (dd >= 31) ? 0 : ((p + (1ll << (dd - 1))) >> dd);
      acc += p;
    } else {
      acc += p;
    }
  };
  const auto* xqep = static_cast<const std::int8_t*>(xqe.contents());
  const auto* eap = static_cast<const std::int8_t*>(eab.contents());
  struct Out { const char* name; const _Float16* p; double n = 0; };
  Out outs[] = {
      {"g64ri", static_cast<const _Float16*>(yri.contents())},
      {"g64rsi", static_cast<const _Float16*>(yrsi.contents())},
      {"g64r", static_cast<const _Float16*>(yr.contents())},
      {"g64rs", static_cast<const _Float16*>(yrs.contents())},
      {"g64(tgmem)", static_cast<const _Float16*>(yt.contents())},
      {"w64rs", static_cast<const _Float16*>(ywrs.contents())},
      {"w64rsi", static_cast<const _Float16*>(ywrsi.contents())},
      {"w64rsf", static_cast<const _Float16*>(ywrsf.contents())},
  };
  double d_f = 0, n_or_ri = 0, n_or_rsi = 0, n_or_w = 0, n_or_f = 0,
         d_or_g = 0, d_or_w = 0, d_or_f = 0, n_rt = 0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      std::int64_t accg = 0, accw = 0, accf = 0;
      int aeg = -1000, aew = -1000, efix = -1000;
      for (int g = 0; g < G; ++g) {
        efix = std::max(efix, (int)ewv[(std::size_t)n * G + g]);
      }
      for (int g = 0; g < G; ++g) {
        std::int64_t pg = 0, pw = 0;
        for (int k = g * 64; k < (g + 1) * 64; ++k) {
          pg += (std::int64_t)xqep[(std::size_t)m * K + k] *
                (std::int64_t)wqp[(std::size_t)n * K + k];
          pw += (std::int64_t)xqre[(std::size_t)m * K + k] *
                (std::int64_t)wqp[(std::size_t)n * K + k];
        }
        const int ew = (int)ewv[(std::size_t)n * G + g];
        shift_acc(accg, aeg, pg, (int)eap[(std::size_t)m * G + g] + ew);
        shift_acc(accw, aew, pw, ew);
        const int sf = std::min(std::max(efix - ew, 0), 31);
        accf += (pw + ((1ll << sf) >> 1)) >> sf;
      }
      double fsum = 0.0;
      for (int k = 0; k < K; ++k) {
        fsum += (double)x[(std::size_t)m * K + k] *
                (double)w[(std::size_t)n * K + k];
      }
      const double refg = (double)(_Float16)std::ldexp((float)accg, aeg);
      const double refw =
          (double)(_Float16)std::ldexp((float)accw, aew + (int)ear[m]);
      const double reff =
          (double)(_Float16)std::ldexp((float)accf, efix + (int)ear[m]);
      const std::size_t o = (std::size_t)m * N + n;
      for (Out& ot : outs) {
        const double v = (double)ot.p[o];
        ot.n += (v - fsum) * (v - fsum);
      }
      d_f += fsum * fsum;
      const double vri = (double)outs[0].p[o], vrsi = (double)outs[1].p[o];
      const double vw = (double)outs[6].p[o];
      n_or_ri += (vri - refg) * (vri - refg);
      n_or_rsi += (vrsi - refg) * (vrsi - refg);
      d_or_g += refg * refg;
      n_or_w += (vw - refw) * (vw - refw);
      d_or_w += refw * refw;
      const double vf = (double)outs[7].p[o];
      n_or_f += (vf - reff) * (vf - reff);
      d_or_f += reff * reff;
      const double vr = (double)outs[2].p[o], vt = (double)outs[4].p[o];
      n_rt += (vr - vt) * (vr - vt);
    }
  }
  const double or_ri = std::sqrt(n_or_ri / d_or_g);
  const double or_rsi = std::sqrt(n_or_rsi / d_or_g);
  const double or_w = std::sqrt(n_or_w / d_or_w);
  const double or_f = std::sqrt(n_or_f / d_or_f);
  const double r_rt = std::sqrt(n_rt / d_f);
  std::printf("[gemm_i8r] oracle rel-L2: g64ri %.4e %s  g64rsi %.4e %s  "
              "w64rsi %.4e %s  w64rsf %.4e %s | g64r-vs-g64 %.4e "
              "(%dx%dx%d)\n",
              or_ri, or_ri < 1e-3 ? "MATCH" : "(BAD)",
              or_rsi, or_rsi < 1e-3 ? "MATCH" : "(BAD)",
              or_w, or_w < 1e-3 ? "MATCH" : "(BAD)",
              or_f, or_f < 1e-3 ? "MATCH" : "(BAD)", r_rt, M, N, K);
  std::printf("[gemm_i8r] vs f32:");
  for (Out& ot : outs) {
    std::printf("  %s %.3e", ot.name, std::sqrt(ot.n / d_f));
    EXPECT_TRUE(std::sqrt(ot.n / d_f) < 3e-2);
  }
  std::printf("\n");
  EXPECT_TRUE(or_ri < 1e-3 && or_rsi < 1e-3 && or_w < 1e-3 && or_f < 1e-3);
  EXPECT_TRUE(r_rt < 1e-3);
}

// THE WAN VAE'S CAUSAL 3x3x3 CONV ON THE HARDWARE CONV.
//
// The video VAE ran every 3x3x3 conv as im2col3d + matmul2d: a 27*C-wide
// gather per output pixel, 11.5 GB of it per full-resolution frame at
// 1920x1152, written and then read back -- and a 1080p decode measured
// ~3.2 TFLOP/s end to end, bandwidth-bound. A causal conv over frames
// (t-2, t-1, t) is three 2D 3x3 convs, one per tap, summed -- which the
// gather-free MPP convolution2d runs with no gather and no concat: the
// first live tap writes the destination, every later one accumulates
// onto it (conv2d_hw_3x3_s1{,_c32}_acc_f16).
//
// Each tap reads its own [3,3,C,Cout] block of the weight, im2col column
// ((kt*3+ky)*3+kx)*C + c becoming block kt, tap (ky, kx), channel c. A
// masked tap (a frame before the sequence) is skipped, so a masked FIRST
// tap makes the second the one that writes. Checked at small shapes on the
// 64-channel tile, the 32-channel one (the 96-channel full-resolution
// convs), and the deep-K GEMM (C=384). VPIPE_HWCONV_WAN_BENCH times both
// routes at the three real level shapes.
// THE NARROW HARDWARE-CONV TILES AT SMALL PLANES: the 32- and 16-channel
// destination tiles (conv2d_hw_3x3_s{1,2}_c{32,16}_f16) against a CPU
// conv, at planes from 8 to 128 a side -- a Qwen-Image VAE round trip of
// a 64x64 picture read 17.4 dB on an M5 with the 32 tile and 35.0 with
// it off. Stride 2 in offset mode 3 (pad (0,1,0,1)).
TEST(conv2d_mma, narrow_hw_tiles_agree_at_small_planes) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma");
  struct F { const char* fn; int tile, stride; };
  const F fs[] = {{"conv2d_hw_3x3_s1_c32_f16", 32, 1},
                  {"conv2d_hw_3x3_s1_c16_f16", 16, 1},
                  {"conv2d_hw_3x3_s2_c32_f16", 32, 2},
                  {"conv2d_hw_3x3_s2_c16_f16", 16, 2}};
  int bad = 0;
  for (const F& f : fs) {
    ComputeFunction fn = lib.function(f.fn);
    ASSERT_TRUE(fn.valid());
    if (!fn.valid()) { return; }
    for (int side : {16, 32, 64, 128}) {
      for (int cin : {96, 192}) {
        const int Cout = 96, H = side, W = side;
        const int OH = H / f.stride, OW = W / f.stride;
        if (OH % 8 != 0 || OW % 8 != 0) { continue; }
        std::mt19937 rng(5u + (unsigned)(side * 3 + cin));
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        std::vector<_Float16> x((std::size_t)H * W * cin);
        std::vector<_Float16> w((std::size_t)9 * cin * Cout);
        for (auto& v : x) { v = (_Float16)d(rng); }
        const float ws = 1.0f / std::sqrt(9.0f * cin);
        for (auto& v : w) { v = (_Float16)(d(rng) * ws); }
        SharedBuffer bx = mc->make_shared_buffer(x.size() * 2);
        SharedBuffer bw = mc->make_shared_buffer(w.size() * 2);
        SharedBuffer by =
            mc->make_shared_buffer((std::size_t)OH * OW * Cout * 2);
        std::memcpy(bx.contents(), x.data(), x.size() * 2);
        std::memcpy(bw.contents(), w.data(), w.size() * 2);
        std::memset(by.contents(), 0, by.byte_size());
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          e.set_function(fn);
          e.set_buffer(0, bx); e.set_buffer(1, bw); e.set_buffer(2, by);
          e.set_constant(3, W); e.set_constant(4, H);
          e.set_constant(5, cin); e.set_constant(6, Cout);
          if (f.stride == 2) { e.set_constant(7, 3); }
          e.dispatch({(unsigned)((OW / 8) * 128), (unsigned)(OH / 8),
                      (unsigned)(Cout / f.tile)}, {128, 1, 1});
        }
        st.commit().wait();
        const auto* yp = static_cast<const _Float16*>(by.contents());
        double num = 0.0, den = 0.0;
        for (int oy = 0; oy < OH; ++oy) {
          for (int ox = 0; ox < OW; ++ox) {
            for (int co = 0; co < Cout; ++co) {
              double a = 0.0;
              for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                  const int iy = f.stride == 1 ? oy + ky - 1 : 2 * oy + ky;
                  const int ix = f.stride == 1 ? ox + kx - 1 : 2 * ox + kx;
                  if (iy < 0 || ix < 0 || iy >= H || ix >= W) { continue; }
                  const _Float16* xp = &x[((std::size_t)iy * W + ix) * cin];
                  const _Float16* wp =
                      &w[((std::size_t)(ky * 3 + kx) * cin) * Cout + co];
                  for (int ci = 0; ci < cin; ++ci) {
                    a += (double)xp[ci] * (double)wp[(std::size_t)ci * Cout];
                  }
                }
              }
              const double g =
                  (double)yp[((std::size_t)oy * OW + ox) * Cout + co];
              num += (g - a) * (g - a);
              den += a * a;
            }
          }
        }
        const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
        const bool ok = rel < 2e-3;
        if (!ok) { ++bad; }
        std::printf("[conv2d_mma narrow] %s %3dx%-3d cin %3d: rel-L2 %.3g%s\n",
                    f.fn, side, side, cin, rel, ok ? "" : "  WRONG");
        EXPECT_TRUE(ok);
      }
    }
  }
  std::printf("[conv2d_mma narrow] %d wrong\n", bad);
}

TEST(conv2d_mma, hw_op_wan_causal_conv3d) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma");
  ComputeLibrary lib_mma = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise");
  ComputeFunction f_hw64 = lib.function("conv2d_hw_3x3_s1_f16");
  ComputeFunction f_hw32 = lib.function("conv2d_hw_3x3_s1_c32_f16");
  ComputeFunction f_acc64 = lib.function("conv2d_hw_3x3_s1_acc_f16");
  ComputeFunction f_acc32 = lib.function("conv2d_hw_3x3_s1_c32_acc_f16");
  ComputeFunction f_mma = lib_mma.function("dense_gemm_mma_t_n128_f16");
  ComputeFunction f_mma_deep =
      lib_mma.function("dense_gemm_mma_t_n128x256_f16");
  ComputeFunction f_i2c3 = lib_elt.function("im2col_hwc_3x3x3_tiled_f16");
  ASSERT_TRUE(f_hw64.valid() && f_hw32.valid() && f_acc64.valid() &&
              f_acc32.valid() && f_mma.valid() && f_mma_deep.valid() &&
              f_i2c3.valid());
  if (!f_acc32.valid() || !f_acc64.valid() || !f_i2c3.valid()) { return; }
  const int SG = 4;

  auto run = [&](int W, int H, int C, int Cout, int mask, bool check,
                 int iters) {
    const std::size_t hw = (std::size_t)W * H;
    const int K = 27 * C;
    std::mt19937 rng(71u + (unsigned)(W * 7 + C * 3 + Cout));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    const float ws = 1.0f / std::sqrt((float)K);
    // Weights in the VAE loader's im2col layout, and their per-tap HWIO
    // blocks: block kt, then (ky, kx), then channel, out-channel fastest.
    std::vector<_Float16> w_col((std::size_t)Cout * K);
    for (auto& v : w_col) { v = (_Float16)(d(rng) * ws); }
    const std::size_t blk = (std::size_t)9 * C * Cout;
    std::vector<_Float16> w_hwio(3 * blk);
    for (int o = 0; o < Cout; ++o) {
      for (int kt = 0; kt < 3; ++kt) {
        for (int s = 0; s < 9; ++s) {
          for (int c = 0; c < C; ++c) {
            w_hwio[(((std::size_t)kt * 9 + s) * C + c) * Cout + o] =
                w_col[(std::size_t)o * K + ((kt * 9 + s) * C + c)];
          }
        }
      }
    }
    SharedBuffer taps[3];
    for (auto& t : taps) {
      t = mc->make_shared_buffer(hw * C * 2);
      auto* p = static_cast<_Float16*>(t.contents());
      for (std::size_t i = 0; i < hw * C; ++i) {
        p[i] = (_Float16)(d(rng) * 0.5f);
      }
    }
    SharedBuffer wc = mc->make_shared_buffer(w_col.size() * 2);
    SharedBuffer wh = mc->make_shared_buffer(w_hwio.size() * 2);
    std::memcpy(wc.contents(), w_col.data(), w_col.size() * 2);
    std::memcpy(wh.contents(), w_hwio.data(), w_hwio.size() * 2);
    SharedBuffer o_hw = mc->make_shared_buffer(hw * Cout * 2);
    SharedBuffer o_i2c = mc->make_shared_buffer(hw * Cout * 2);
    // The decoder's band: the matmul2d row cap, and under 2^31 bytes.
    std::size_t rows = std::min<std::size_t>(hw, 262144);
    rows = std::min<std::size_t>(rows, (std::size_t)0x7fffffff / (2 * K));
    SharedBuffer col = mc->make_shared_buffer(rows * K * 2);
    if (col.empty() || o_hw.empty() || o_i2c.empty()) {
      std::printf("[wan_hwconv] alloc failed at %dx%d C=%d -- skip\n", W, H,
                  C);
      return;
    }
    const bool c32 = (Cout % 64) != 0;
    const int TC = c32 ? 32 : 64;

    auto launch_hw = [&](int n) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          bool first = true;
          for (int kt = 0; kt < 3; ++kt) {
            if (((mask >> kt) & 1) == 0) { continue; }
            enc.set_function(first ? (c32 ? f_hw32 : f_hw64)
                                   : (c32 ? f_acc32 : f_acc64));
            enc.set_buffer(0, taps[kt]);
            enc.set_buffer(1, wh, (std::size_t)kt * blk * 2);
            enc.set_buffer(2, o_hw);
            enc.set_constant(3, W); enc.set_constant(4, H);
            enc.set_constant(5, C); enc.set_constant(6, Cout);
            enc.dispatch({(unsigned)((W / 8) * SG * 32), (unsigned)(H / 8),
                          (unsigned)(Cout / TC)}, {(unsigned)(SG * 32), 1, 1});
            first = false;
          }
        }
      }
      st.commit().wait();
    };
    auto launch_i2c = [&](int n) {
      const bool deep = K >= 6144;
      const int BN = deep ? 256 : 128;
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int i = 0; i < n; ++i) {
          for (std::size_t r0 = 0; r0 < hw; r0 += rows) {
            const int m = (int)std::min(rows, hw - r0);
            enc.set_function(f_i2c3);
            for (int t = 0; t < 3; ++t) {
              enc.set_buffer((unsigned)t, taps[t]);
            }
            enc.set_buffer(3, col);
            enc.set_constant(4, H); enc.set_constant(5, W);
            enc.set_constant(6, C); enc.set_constant(7, (int)r0);
            enc.set_constant(8, m); enc.set_constant(9, mask);
            enc.dispatch({(unsigned)K, (unsigned)m, 1}, {64, 1, 1});
            enc.set_function(deep ? f_mma_deep : f_mma);
            enc.set_buffer(0, col); enc.set_buffer(1, wc);
            enc.set_buffer(2, wc); enc.set_buffer(3, o_i2c, r0 * Cout * 2);
            enc.set_constant(4, K); enc.set_constant(5, Cout);
            enc.set_constant(6, m); enc.set_constant(7, 0);
            enc.dispatch({(unsigned)(((Cout + BN - 1) / BN) * 256),
                          (unsigned)((m + 127) / 128), 1}, {256, 1, 1});
          }
        }
      }
      st.commit().wait();
    };

    launch_hw(1);
    launch_i2c(1);
    if (check) {
      const auto* a = static_cast<const _Float16*>(o_hw.contents());
      const auto* b = static_cast<const _Float16*>(o_i2c.contents());
      double num = 0.0, den = 0.0;
      for (std::size_t i = 0; i < hw * Cout; ++i) {
        const double e = (double)a[i] - (double)b[i];
        num += e * e;
        den += (double)b[i] * (double)b[i];
      }
      const double r = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
      std::printf("[wan_hwconv] %dx%d 3x%d->%d mask %d (%s tile): rel-L2 "
                  "%.3e\n", W, H, C, Cout, mask, c32 ? "32" : "64", r);
      EXPECT_TRUE(r < 1e-2);
    }
    if (iters <= 0) { return; }
    const auto t0 = std::chrono::steady_clock::now();
    launch_hw(iters);
    const auto t1 = std::chrono::steady_clock::now();
    launch_i2c(iters);
    const auto t2 = std::chrono::steady_clock::now();
    const double s_hw = secs_(t0, t1) / iters, s_i2c = secs_(t1, t2) / iters;
    const double tf = 2.0 * (double)hw * Cout * K / 1e12;
    std::printf("[wan_hwconv] %4dx%-4d 3x%-3d->%-3d: per-tap hw %.0f ms "
                "(%.1f TF/s)  im2col3d+mma %.0f ms (%.1f TF/s)  %.2fx\n", W,
                H, C, Cout, s_hw * 1e3, tf / s_hw, s_i2c * 1e3, tf / s_i2c,
                s_i2c / s_hw);
  };

  run(48, 32, 96, 96, 7, /*check=*/true, 0);      // 32-channel tile
  run(48, 32, 96, 96, 6, /*check=*/true, 0);      // first tap masked
  run(48, 32, 64, 192, 7, /*check=*/true, 0);     // 64-channel tile
  run(48, 32, 384, 384, 3, /*check=*/true, 0);    // deep K, last masked
  if (std::getenv("VPIPE_HWCONV_WAN_BENCH") != nullptr) {
    run(1920, 1152, 96, 96, 7, /*check=*/true, 3);
    run(960, 576, 192, 192, 7, /*check=*/true, 3);
    run(480, 288, 384, 384, 7, /*check=*/true, 3);
  }
}

// Where the MPP convolution2d op stops being right: an f16 3x3 over planes
// either side of 2^31 BYTES of activation and destination, sampled rows on
// both sides of the row where the plane crosses the line, against a CPU
// dot product. MEASURED on the M5: every row below it at ~2e-4, every row
// past it unwritten with the stream reporting success -- which is why the
// VAE callers judge an operand by tensor_op_operand_fits(), its bytes,
// where they had judged its elements (these planes are under 2^31 of
// those). The rows below must stay right and the helper must refuse each
// plane that crosses. ~5 GB of buffers; VPIPE_HWCONV_2GB=1.
TEST(conv2d_mma, hw_op_past_the_2gb_line) {
  if (std::getenv("VPIPE_HWCONV_2GB") == nullptr) { return; }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma");
  struct Case { int W, H, cin, cout; const char* fn; int tile; };
  const Case cases[] = {
      {1024, 1024, 288, 288, "conv2d_hw_3x3_s1_c32_f16", 32},
      {2048, 2048, 288, 288, "conv2d_hw_3x3_s1_c32_f16", 32},
      {2048, 2112, 256, 256, "conv2d_hw_3x3_s1_f16", 64},
  };
  auto hv = [](std::size_t i, std::uint32_t salt) {
    std::uint32_t x = (std::uint32_t)(i * 2654435761u) ^ salt;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    return (float)(x & 0xffff) / 65535.0f - 0.5f;
  };
  for (const Case& c : cases) {
    ComputeFunction fn = lib.function(c.fn);
    ASSERT_TRUE(fn.valid());
    if (!fn.valid()) { return; }
    const std::size_t ni = (std::size_t)c.W * c.H * c.cin;
    const std::size_t no = (std::size_t)c.W * c.H * c.cout;
    const std::size_t nw = (std::size_t)9 * c.cin * c.cout;
    const bool fits = genai::tensor_op_operand_fits(ni, 2) &&
                      genai::tensor_op_operand_fits(no, 2);
    SharedBuffer ib = mc->make_shared_buffer(ni * 2);
    SharedBuffer wb = mc->make_shared_buffer(nw * 2);
    SharedBuffer ob = mc->make_shared_buffer(no * 2);
    ASSERT_TRUE(!ib.empty() && !wb.empty() && !ob.empty());
    if (ib.empty() || wb.empty() || ob.empty()) { return; }
    auto* in = static_cast<_Float16*>(ib.contents());
    auto* w = static_cast<_Float16*>(wb.contents());
    for (std::size_t i = 0; i < ni; ++i) { in[i] = (_Float16)hv(i, 7u); }
    const float ws = 2.0f / std::sqrt(9.0f * c.cin);
    for (std::size_t i = 0; i < nw; ++i) {
      w[i] = (_Float16)(hv(i, 99u) * ws);
    }
    std::memset(ob.contents(), 0, no * 2);
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder enc = st.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, ib); enc.set_buffer(1, wb); enc.set_buffer(2, ob);
      enc.set_constant(3, c.W); enc.set_constant(4, c.H);
      enc.set_constant(5, c.cin); enc.set_constant(6, c.cout);
      enc.dispatch({(unsigned)(c.W / 8) * 128u, (unsigned)(c.H / 8),
                    (unsigned)(c.cout / c.tile)},
                   {128, 1, 1});
    }
    const bool ok = st.commit().wait_ok();
    const auto* out = static_cast<const _Float16*>(ob.contents());
    // The row at which (row * W) * max(cin, cout) * 2 crosses 2^31.
    const double line =
        2147483648.0 / (2.0 * c.W * std::max(c.cin, c.cout));
    std::printf("[hw_2gb] %dx%d cin %d cout %d: %.2f GB in, stream ok %d, "
                "helper says %s, line at row %.1f\n", c.W, c.H, c.cin,
                c.cout, ni * 2 / 1e9, ok ? 1 : 0, fits ? "fits" : "refuse",
                line);
    EXPECT_TRUE(fits == (line >= c.H));
    const int rows[] = {0, c.H / 2, (int)line - 40, (int)line - 9,
                        (int)line + 9, (int)line + 40, c.H - 1};
    const int cols[] = {0, 517, c.W / 2 + 3, c.W - 1};
    const int chans[] = {0, 31, c.cout / 2 + 1, c.cout - 1};
    for (const int y : rows) {
      if (y < 0 || y >= c.H) { continue; }
      double num = 0.0, den = 0.0;
      for (const int x : cols) {
        for (const int oc : chans) {
          double ref = 0.0;
          for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
              const int sy = y + ky - 1, sx = x + kx - 1;
              if (sy < 0 || sy >= c.H || sx < 0 || sx >= c.W) { continue; }
              const _Float16* ip =
                  in + ((std::size_t)sy * c.W + sx) * c.cin;
              const _Float16* wp =
                  w + (std::size_t)(ky * 3 + kx) * c.cin * c.cout + oc;
              for (int ic = 0; ic < c.cin; ++ic) {
                ref += (double)ip[ic] * (double)wp[(std::size_t)ic * c.cout];
              }
            }
          }
          const double got =
              (double)out[((std::size_t)y * c.W + x) * c.cout + oc];
          num += (got - ref) * (got - ref); den += ref * ref;
        }
      }
      const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
      const bool below = y + 8 <= line;     // its whole 8-row tile below
      std::printf("[hw_2gb]   row %5d (%s): rel-L2 %.3e\n", y,
                  below ? "below the line" : "past the line ", rel);
      if (below) { EXPECT_TRUE(ok && rel < 2e-3); }
    }
  }
}

// The MiniMax-H3 encoder's causal 3x3x3 conv on the hardware op
// (conv3d_hw_causal_s{1,2}_*): every entry against a CPU oracle -- the
// three tuned tiles and the partial-tile one, at stride 1 and 2, a
// temporal stride of 2, and the fused residual. The input is handed in
// already padded, so the oracle reads it as given: dest (t, y, x) is
// bias + res + the sum over live taps kt (input frame t*stride_t+kt-2)
// of in[si][S*y+ky][S*x+kx] . w[kt][ky][kx]. bf16, as the encoder runs.
namespace {

struct C3Case {
  int W, H, C, Cout, Ti, stride_t, stride_s;
};

// Runs entry `name` on one case; returns rel-L2 against the oracle.
double
c3_run_case_(MetalCompute* mc, ComputeLibrary& lib, const std::string& name,
             int tw, int th, int tc, int sg, const C3Case& k)
{
  const int pad = (k.stride_s == 2) ? 1 : 2;
  const int Wp = k.W + pad, Hp = k.H + pad;
  const int Wo = (k.stride_s == 2) ? (k.W - 2) / 2 + 1 : k.W;
  const int Ho = (k.stride_s == 2) ? (k.H - 2) / 2 + 1 : k.H;
  const int To = (k.Ti - 1) / k.stride_t + 1;
  std::mt19937 rng(91u + (unsigned)(k.W * 31 + k.C + k.stride_s));
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  const std::size_t nin = (std::size_t)k.Ti * Hp * Wp * k.C;
  const std::size_t nw = (std::size_t)27 * k.C * k.Cout;
  const std::size_t nout = (std::size_t)To * Ho * Wo * k.Cout;
  std::vector<float> in(nin), w(nw), b((std::size_t)k.Cout), r(nout);
  const float ws = 1.0f / std::sqrt(27.0f * (float)k.C);
  // Values rounded to bf16 first, so the oracle sees what the GPU does.
  auto q = [](float v) { return bf16_to_f32(f32_to_bf16(v)); };
  for (auto& v : in) { v = q(d(rng)); }
  for (auto& v : w) { v = q(d(rng) * ws); }
  for (auto& v : b) { v = q(0.1f * d(rng)); }
  for (auto& v : r) { v = q(d(rng)); }
  ComputeFunction fn = lib.function(name);
  if (!fn.valid()) { return 1.0; }
  auto upload = [&](const std::vector<float>& v) {
    SharedBuffer sb = mc->make_shared_buffer(v.size() * 2);
    auto* p = static_cast<std::uint16_t*>(sb.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { p[i] = f32_to_bf16(v[i]); }
    return sb;
  };
  SharedBuffer bin = upload(in), bw = upload(w), bb = upload(b),
               br = upload(r);
  SharedBuffer bo = mc->make_shared_buffer(nout * 2);
  std::memset(bo.contents(), 0xff, nout * 2);       // NaN: unwritten shows
  {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder enc = st.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, bin); enc.set_buffer(1, bw); enc.set_buffer(2, bo);
      enc.set_buffer(3, bb); enc.set_buffer(4, br);
      enc.set_constant(5, Wp); enc.set_constant(6, Hp);
      enc.set_constant(7, k.C); enc.set_constant(8, k.Cout);
      enc.set_constant(9, Wo); enc.set_constant(10, Ho);
      enc.set_constant(11, k.Ti); enc.set_constant(12, k.stride_t);
      enc.set_constant(13, 1);
      enc.set_constant(14, 0);
      enc.dispatch({(unsigned)(((Wo + tw - 1) / tw) * sg * 32),
                    (unsigned)((Ho + th - 1) / th),
                    (unsigned)(To * (k.Cout / tc))},
                   {(unsigned)(sg * 32), 1, 1});
    }
    if (!st.commit().wait_ok()) { return 1.0; }
  }
  const auto* go = static_cast<const std::uint16_t*>(bo.contents());
  double num = 0.0, den = 0.0;
  const int S = k.stride_s;
  for (int t = 0; t < To; ++t) {
    for (int y = 0; y < Ho; ++y) {
      for (int x = 0; x < Wo; ++x) {
        for (int o = 0; o < k.Cout; ++o) {
          const std::size_t oi =
              (((std::size_t)t * Ho + y) * Wo + x) * k.Cout + o;
          double acc = b[(std::size_t)o] + r[oi];
          for (int kt = 0; kt < 3; ++kt) {
            const int si = t * k.stride_t + kt - 2;
            if (si < 0 || si >= k.Ti) { continue; }
            for (int ky = 0; ky < 3; ++ky) {
              for (int kx = 0; kx < 3; ++kx) {
                const float* ip =
                    &in[(((std::size_t)si * Hp + S * y + ky) * Wp + S * x +
                         kx) * k.C];
                const float* wp =
                    &w[(((std::size_t)kt * 3 + ky) * 3 + kx) * k.C * k.Cout +
                       o];
                for (int c = 0; c < k.C; ++c) {
                  acc += (double)ip[c] * (double)wp[(std::size_t)c * k.Cout];
                }
              }
            }
          }
          const double e = (double)bf16_to_f32(go[oi]) - acc;
          num += e * e;
          den += acc * acc;
        }
      }
    }
  }
  const double rel = den > 0.0 ? std::sqrt(num / den) : 1.0;
  return std::isfinite(rel) ? rel : 1.0;
}

}  // namespace

TEST(conv2d_mma, hw_op_causal3d_matches_oracle) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma_bf16");
  struct Entry { const char* sfx; int tw, th, tc, sg; bool tail; };
  const Entry entries[] = {{"c128", 8, 8, 128, 4, false},
                           {"c128d", 8, 8, 128, 4, false},
                           {"c128w16d", 16, 8, 128, 8, false},
                           {"tail", 8, 8, 128, 4, true}};
  // Whole tiles for every entry; then grids only the tail entry serves.
  const C3Case whole[] = {{32, 16, 64, 128, 4, 1, 1},
                          {32, 32, 64, 256, 5, 2, 2},
                          {32, 16, 3, 128, 4, 1, 1},
                          {32, 16, 8, 128, 4, 1, 1}};
  const C3Case ragged[] = {{20, 12, 64, 128, 3, 1, 1},
                           {26, 22, 64, 128, 5, 2, 2}};
  for (const Entry& e : entries) {
    for (int st = 1; st <= 2; ++st) {
      const std::string name = std::string("conv3d_hw_causal_s") +
                               (st == 1 ? "1_" : "2_") + e.sfx + "_f16";
      for (const C3Case& k : whole) {
        if (k.stride_s != st) { continue; }
        const double rel = c3_run_case_(mc, lib, name, e.tw, e.th, e.tc,
                                        e.sg, k);
        std::printf("[c3d] %-32s %dx%d %d->%d Ti=%d st=%d/%d rel %.2e\n",
                    name.c_str(), k.W, k.H, k.C, k.Cout, k.Ti, k.stride_t,
                    k.stride_s, rel);
        EXPECT_TRUE(rel < 1e-2);
      }
      if (!e.tail) { continue; }
      for (const C3Case& k : ragged) {
        if (k.stride_s != st) { continue; }
        const double rel = c3_run_case_(mc, lib, name, e.tw, e.th, e.tc,
                                        e.sg, k);
        std::printf("[c3d] %-32s %dx%d %d->%d Ti=%d st=%d/%d rel %.2e\n",
                    name.c_str(), k.W, k.H, k.C, k.Cout, k.Ti, k.stride_t,
                    k.stride_s, rel);
        EXPECT_TRUE(rel < 1e-2);
      }
    }
  }
  // The 16-channel entry (stride 1, guarded): the encoder's conv_out,
  // 1024 -> 48, on a whole grid and a ragged one.
  const C3Case c16[] = {{16, 16, 128, 48, 5, 1, 1}, {20, 12, 64, 48, 3, 1, 1}};
  for (const C3Case& k : c16) {
    const double rel = c3_run_case_(mc, lib, "conv3d_hw_causal_s1_c16_f16",
                                    8, 8, 16, 4, k);
    std::printf("[c3d] %-32s %dx%d %d->%d Ti=%d rel %.2e\n",
                "conv3d_hw_causal_s1_c16_f16", k.W, k.H, k.C, k.Cout, k.Ti,
                rel);
    EXPECT_TRUE(rel < 1e-2);
  }
}

// The same entries TIMED at the encoder's real shapes -- its 256x256
// tile's levels -- for the tile table in conv2d_mma.metal. A sweep, not a
// gate: it prints. Env: VPIPE_HWCONV_C3D_BENCH=1.
TEST(conv2d_mma, hw_op_causal3d_tile_sweep) {
  const char* on = std::getenv("VPIPE_HWCONV_C3D_BENCH");
  if (on == nullptr || *on == '\0' || *on == '0') { return; }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib = mc->load_library("conv2d_mma_bf16");
  struct Var { const char* name; int tw, th, sg; };
  const Var vars[] = {{"c128", 8, 8, 4}, {"c128d", 8, 8, 4},
                      {"c128w16d", 16, 8, 8}};
  struct Shape { int W, H, C, Cout, T; };
  // VPIPE_HWCONV_C3D_T: frames (5 by default; a reference clip is 17
  // at the first two levels).
  const char* te = std::getenv("VPIPE_HWCONV_C3D_T");
  const int T = (te != nullptr && std::atoi(te) > 0) ? std::atoi(te) : 5;
  const char* ce = std::getenv("VPIPE_HWCONV_C3D_CHUNK");
  const int chunk = (ce != nullptr && std::atoi(ce) > 0) ? std::atoi(ce) : T;
  const Shape shapes[] = {{256, 256, 128, 128, T}, {128, 128, 128, 256, T},
                          {128, 128, 256, 256, T}, {64, 64, 256, 256, T},
                          {32, 32, 512, 512, T}, {16, 16, 1024, 1024, T}};
  for (const Shape& sh : shapes) {
    const int Wp = sh.W + 2, Hp = sh.H + 2;
    const std::size_t nin = (std::size_t)sh.T * Hp * Wp * sh.C;
    const std::size_t nw = (std::size_t)27 * sh.C * sh.Cout;
    const std::size_t nout = (std::size_t)sh.T * sh.H * sh.W * sh.Cout;
    SharedBuffer in = mc->make_shared_buffer(nin * 2);
    SharedBuffer wt = mc->make_shared_buffer(nw * 2);
    SharedBuffer bs = mc->make_shared_buffer((std::size_t)sh.Cout * 2);
    SharedBuffer out = mc->make_shared_buffer(nout * 2);
    ASSERT_TRUE(!in.empty() && !wt.empty() && !bs.empty() && !out.empty());
    {
      std::mt19937 rng(5u + (unsigned)sh.C);
      std::uniform_real_distribution<float> d(-1.0f, 1.0f);
      auto* pi = static_cast<std::uint16_t*>(in.contents());
      for (std::size_t i = 0; i < nin; ++i) { pi[i] = f32_to_bf16(d(rng)); }
      auto* pw = static_cast<std::uint16_t*>(wt.contents());
      const float ws = 1.0f / std::sqrt(27.0f * (float)sh.C);
      for (std::size_t i = 0; i < nw; ++i) {
        pw[i] = f32_to_bf16(d(rng) * ws);
      }
      std::memset(bs.contents(), 0, (std::size_t)sh.Cout * 2);
    }
    double flops = 0.0;
    for (int t = 0; t < sh.T; ++t) {
      const int live = std::min(t, 2) + 1;
      flops += 2.0 * sh.H * sh.W * sh.Cout * 9.0 * live * sh.C;
    }
    std::printf("[c3d] %dx%d %d->%d T=%d (%.1f GFLOP)\n", sh.W, sh.H, sh.C,
                sh.Cout, sh.T, flops * 1e-9);
    for (const Var& v : vars) {
      ComputeFunction fn = lib.function(std::string("conv3d_hw_causal_s1_") +
                                        v.name + "_f16");
      ASSERT_TRUE(fn.valid());
      double best = 1e30;
      for (int r = 0; r < 4; ++r) {
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder enc = st.begin_compute();
          // `chunk` output frames a dispatch (VPIPE_HWCONV_C3D_CHUNK).
          for (int f0 = 0; f0 < sh.T; f0 += chunk) {
            const int nf = std::min(chunk, sh.T - f0);
            enc.set_function(fn);
            enc.set_buffer(0, in); enc.set_buffer(1, wt);
            enc.set_buffer(2, out); enc.set_buffer(3, bs);
            enc.set_buffer(4, out);
            enc.set_constant(5, Wp); enc.set_constant(6, Hp);
            enc.set_constant(7, sh.C); enc.set_constant(8, sh.Cout);
            enc.set_constant(9, sh.W); enc.set_constant(10, sh.H);
            enc.set_constant(11, sh.T); enc.set_constant(12, 1);
            enc.set_constant(13, 0); enc.set_constant(14, f0);
            enc.dispatch({(unsigned)((sh.W / v.tw) * v.sg * 32),
                          (unsigned)(sh.H / v.th),
                          (unsigned)(nf * (sh.Cout / 128))},
                         {(unsigned)(v.sg * 32), 1, 1});
          }
        }
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = st.commit().wait_ok();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (ok && r > 0) { best = std::min(best, ms); }
      }
      std::printf("[c3d]   %-10s %8.2f ms  %6.2f TFLOP/s\n", v.name, best,
                  flops * 1e-9 / best);
    }
  }
}

// The H3 encoder's level-0 conv against a large bf16 matmul2d GEMM,
// ALTERNATED for several seconds: on a power-limited GPU (the fanless M5)
// a burst runs ~2x what the box sustains, so the only figure that says
// how well the conv uses the matrix hardware is its rate RELATIVE to a
// GEMM's in the same thermal state. Env: VPIPE_HWCONV_C3D_SUSTAIN=<rounds>.
TEST(conv2d_mma, hw_op_causal3d_vs_gemm_sustained) {
  const char* on = std::getenv("VPIPE_HWCONV_C3D_SUSTAIN");
  if (on == nullptr || std::atoi(on) <= 0) { return; }
  const int rounds = std::atoi(on);
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lc = mc->load_library("conv2d_mma_bf16");
  ComputeLibrary lg = mc->load_library("dense_gemm_mma_bf16");
  ComputeFunction fc = lc.function("conv3d_hw_causal_s1_c128_f16");
  ComputeFunction fg = lg.function("dense_gemm_mma_t_n128x256_f16");
  ASSERT_TRUE(fc.valid() && fg.valid());
  // Conv: a reference clip's level 0, 17 frames of 256^2, 128 -> 128.
  const int T = 17, W = 256, H = 256, C = 128, Co = 128;
  const int Wp = W + 2, Hp = H + 2;
  double cflops = 0.0;
  for (int t = 0; t < T; ++t) {
    cflops += 2.0 * H * W * Co * 9.0 * (std::min(t, 2) + 1) * C;
  }
  // GEMM: 4096^3.
  const int M = 4096, N = 4096, K = 4096;
  const double gflops = 2.0 * M * N * K;
  auto rnd = [](SharedBuffer& b, std::size_t n, float s, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { p[i] = f32_to_bf16(d(rng) * s); }
  };
  const std::size_t nin = (std::size_t)T * Hp * Wp * C;
  const std::size_t nw = (std::size_t)27 * C * Co;
  const std::size_t nout = (std::size_t)T * H * W * Co;
  SharedBuffer cin = mc->make_shared_buffer(nin * 2);
  SharedBuffer cw = mc->make_shared_buffer(nw * 2);
  SharedBuffer cb = mc->make_shared_buffer((std::size_t)Co * 2);
  SharedBuffer co = mc->make_shared_buffer(nout * 2);
  SharedBuffer gx = mc->make_shared_buffer((std::size_t)M * K * 2);
  SharedBuffer gw = mc->make_shared_buffer((std::size_t)N * K * 2);
  SharedBuffer gy = mc->make_shared_buffer((std::size_t)M * N * 2);
  ASSERT_TRUE(!cin.empty() && !cw.empty() && !co.empty() && !gx.empty() &&
              !gw.empty() && !gy.empty());
  rnd(cin, nin, 1.0f, 1);
  rnd(cw, nw, 0.03f, 2);
  rnd(cb, (std::size_t)Co, 0.1f, 3);
  rnd(gx, (std::size_t)M * K, 1.0f, 4);
  rnd(gw, (std::size_t)N * K, 0.02f, 5);
  auto timed = [&](auto&& enc_fn) {
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder enc = st.begin_compute(); enc_fn(enc); }
    const auto t0 = std::chrono::steady_clock::now();
    (void)st.commit().wait_ok();
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
  };
  double tc = 0.0, tg = 0.0;
  for (int r = 0; r < rounds; ++r) {
    const double a = timed([&](ComputeEncoder& enc) {
      enc.set_function(fc);
      enc.set_buffer(0, cin); enc.set_buffer(1, cw); enc.set_buffer(2, co);
      enc.set_buffer(3, cb); enc.set_buffer(4, co);
      enc.set_constant(5, Wp); enc.set_constant(6, Hp);
      enc.set_constant(7, C); enc.set_constant(8, Co);
      enc.set_constant(9, W); enc.set_constant(10, H);
      enc.set_constant(11, T); enc.set_constant(12, 1);
      enc.set_constant(13, 0); enc.set_constant(14, 0);
      enc.dispatch({(unsigned)((W / 8) * 128), (unsigned)(H / 8),
                    (unsigned)(T * (Co / 128))}, {128, 1, 1});
    });
    const double b = timed([&](ComputeEncoder& enc) {
      enc.set_function(fg);
      enc.set_buffer(0, gx); enc.set_buffer(1, gw); enc.set_buffer(2, gw);
      enc.set_buffer(3, gy);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + 255) / 256) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    });
    tc += a;
    tg += b;
    if (r == 0 || r + 1 == rounds || (r + 1) % 10 == 0) {
      std::printf("[c3d-sus] round %3d: conv %.2f TFLOP/s, gemm %.2f "
                  "TFLOP/s\n", r + 1, cflops * 1e-9 / a, gflops * 1e-9 / b);
    }
  }
  const double rc = cflops * rounds * 1e-9 / tc;
  const double rg = gflops * rounds * 1e-9 / tg;
  std::printf("[c3d-sus] %d rounds, %.1f s: conv %.2f TFLOP/s, gemm %.2f "
              "TFLOP/s -- conv at %.0f%% of the GEMM's rate\n", rounds,
              (tc + tg) * 1e-3, rc, rg, 100.0 * rc / rg);
}

// THE COARSE TIER (gemm_i8c_*, quant_i8_row_gc*, affine_to_i8c_w8_gc*,
// i8c_dequant_gc*): symmetric int8 at a group of GC for both operands.
// (1) Always: every kernel against a CPU replica of its own arithmetic,
// at a ragged M and N -- the GEMM to a bf16 ulp from the integer sums
// folded in the kernel's order, the quantizers to one code.
// (2) VPIPE_I8_COARSE_STUDY=1: the ROUTE's accuracy against an f64
// product with the original weight -- g64 (a w8 g64 pack in the signed
// layout), coarse requantized from that pack, coarse quantized straight
// from the bf16 weight, and the bf16 matmul2d floor -- then its speed at
// Qwen-Image-2.1's projection shapes (activation quantizer + GEMM, two
// weights cycled so each call reads cold).
TEST(gemm_i8, coarse_tier) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lg = mc->load_library("dense_gemm_mma_bf16");
  ComputeLibrary lq = mc->load_library("affine_dequant_bf16");
  const int kGroups[] = {256, 512, 1024};
  const char* kTiles[] = {"64x64", "64x128", "128x64", "128x128"};
  const int kTileBM[] = {64, 64, 128, 128}, kTileBN[] = {64, 128, 64, 128};
  const int kTileSG[] = {4, 8, 8, 8};
  auto fn = [&](ComputeLibrary& l, const std::string& n) {
    ComputeFunction f = l.function(n.c_str());
    if (!f.valid()) { std::printf("[gemm_i8c] %s missing\n", n.c_str()); }
    return f;
  };
  auto up = [&](const void* p, std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n);
    if (!b.empty() && p != nullptr) { std::memcpy(b.contents(), p, n); }
    return b;
  };
  auto run = [&](const std::function<void(ComputeEncoder&)>& f) {
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder e = st.begin_compute(); f(e); }
    st.commit().wait();
  };
  // gc 0: the g64 quantizers' grid ({256, rows}); otherwise the coarse
  // tier's -- one simdgroup per (row, group), four groups a threadgroup.
  auto rows_pass = [&](ComputeEncoder& e, const ComputeFunction& f,
                       int rows, int K = 0, int gc = 0) {
    e.set_function(f);
    if (gc == 0) {
      e.dispatch({256u, (unsigned)rows, 1}, {256, 1, 1});
    } else {
      const unsigned tgs = (unsigned)((K / gc + 3) / 4);
      e.dispatch({tgs * 128u, (unsigned)rows, 1}, {128, 1, 1});
    }
  };
  auto gemm = [&](ComputeEncoder& e, const ComputeFunction& f, int t,
                  const SharedBuffer& xq, const SharedBuffer& wq,
                  const SharedBuffer& as, const SharedBuffer& ws,
                  const SharedBuffer& y, int M, int N, int K) {
    e.set_function(f);
    e.set_buffer(0, xq); e.set_buffer(1, wq); e.set_buffer(2, as);
    e.set_buffer(3, ws); e.set_buffer(4, y);
    e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, M);
    e.set_buffer(8, ws); e.set_constant(9, 0);
    const unsigned tg = (unsigned)(32 * kTileSG[t]);
    e.dispatch({(unsigned)((N + kTileBN[t] - 1) / kTileBN[t]) * tg,
                (unsigned)((M + kTileBM[t] - 1) / kTileBM[t]), 1},
               {tg, 1, 1});
  };

  // ---- (1) the kernels against their CPU replicas ----------------------
  {
    const int M = 200, N = 136, K = 2048;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::uint16_t> x((std::size_t)M * K), w((std::size_t)N * K);
    for (auto& v : x) { v = f32_to_bf16(nd(rng)); }
    for (auto& v : w) { v = f32_to_bf16(nd(rng) * 0.05f); }
    SharedBuffer xb = up(x.data(), x.size() * 2);
    SharedBuffer wb = up(w.data(), w.size() * 2);
    for (int GC : kGroups) {
      const int G = K / GC;
      const std::string sfx = std::to_string(GC);
      ComputeFunction fq = fn(lq, "quant_i8_row_gc" + sfx);
      ComputeFunction fdq = fn(lq, "i8c_dequant_gc" + sfx);
      if (!fq.valid() || !fdq.valid()) { ASSERT_TRUE(false); return; }
      SharedBuffer xq = up(nullptr, (std::size_t)M * K);
      SharedBuffer as = up(nullptr, (std::size_t)M * G * 2);
      SharedBuffer wq = up(nullptr, (std::size_t)N * K);
      SharedBuffer ws = up(nullptr, (std::size_t)N * G * 2);
      SharedBuffer wd = up(nullptr, (std::size_t)N * K * 2);
      run([&](ComputeEncoder& e) {
        e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, as);
        e.set_constant(3, K);
        rows_pass(e, fq, M, K, GC);
        e.set_buffer(0, wb); e.set_buffer(1, wq); e.set_buffer(2, ws);
        e.set_constant(3, K);
        rows_pass(e, fq, N, K, GC);
        e.set_buffer(0, wq); e.set_buffer(1, ws); e.set_buffer(2, wd);
        e.set_constant(3, K);
        rows_pass(e, fdq, N, K, GC);
      });
      // The quantizer, against its replica: one code at most, rarely.
      const auto* q8 = static_cast<const std::int8_t*>(xq.contents());
      const auto* asp = static_cast<const std::uint16_t*>(as.contents());
      std::size_t off1 = 0, offmore = 0, sbad = 0;
      for (int m = 0; m < M; ++m) {
        for (int g = 0; g < G; ++g) {
          float am = 0.0f;
          for (int k = g * GC; k < (g + 1) * GC; ++k) {
            const float v = bf16_to_f32(x[(std::size_t)m * K + k]);
            am = std::max(am, std::fabs(v));
          }
          if (asp[(std::size_t)m * G + g] != f32_to_bf16(am / 127.0f)) {
            ++sbad;
          }
          const float inv = am > 0.0f ? 127.0f / am : 0.0f;
          for (int k = g * GC; k < (g + 1) * GC; ++k) {
            const std::size_t i = (std::size_t)m * K + k;
            int r = (int)std::nearbyint(bf16_to_f32(x[i]) * inv);
            r = std::clamp(r, -127, 127);
            const int d = std::abs(r - (int)q8[i]);
            off1 += d == 1 ? 1 : 0;
            offmore += d > 1 ? 1 : 0;
          }
        }
      }
      // The GEMM, against the integer sums folded in its order.
      const auto* w8 = static_cast<const std::int8_t*>(wq.contents());
      const auto* wsp = static_cast<const std::uint16_t*>(ws.contents());
      std::vector<float> ref((std::size_t)M * N);
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          float acc = 0.0f;
          for (int g = 0; g < G; ++g) {
            std::int32_t p = 0;
            for (int k = g * GC; k < (g + 1) * GC; ++k) {
              p += (std::int32_t)q8[(std::size_t)m * K + k] *
                   (std::int32_t)w8[(std::size_t)n * K + k];
            }
            const float a = bf16_to_f32(asp[(std::size_t)m * G + g]);
            const float s = bf16_to_f32(wsp[(std::size_t)n * G + g]);
            acc = std::fma((float)p * s, a, acc);
          }
          ref[(std::size_t)m * N + n] = acc;
        }
      }
      // The dequant, against codes * scale.
      std::size_t dbad = 0;
      const auto* wdp = static_cast<const std::uint16_t*>(wd.contents());
      for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
          const float v = (float)w8[(std::size_t)n * K + k] *
                          bf16_to_f32(wsp[(std::size_t)n * G + k / GC]);
          if (wdp[(std::size_t)n * K + k] != f32_to_bf16(v)) { ++dbad; }
        }
      }
      std::printf("[gemm_i8c] gc%-4d quant: %zu/%zu codes 1 off, %zu more, "
                  "%zu scales off; dequant %zu off\n", GC, off1, x.size(),
                  offmore, sbad, dbad);
      EXPECT_TRUE(offmore == 0 && off1 * 1000 < x.size() && sbad == 0 &&
                  dbad == 0);
      for (int t = 0; t < 4; ++t) {
        const std::string nm =
            "gemm_i8c_g" + sfx + "_t" + std::string(kTiles[t]);
        ComputeFunction fg = fn(lg, nm);
        if (!fg.valid()) { ASSERT_TRUE(false); continue; }
        SharedBuffer y = up(nullptr, (std::size_t)M * N * 2);
        std::memset(y.contents(), 0xff, y.byte_size());
        run([&](ComputeEncoder& e) {
          gemm(e, fg, t, xq, wq, as, ws, y, M, N, K);
        });
        const auto* yp = static_cast<const std::uint16_t*>(y.contents());
        std::size_t bad = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
          const float r = ref[i], u = bf16_to_f32(yp[i]);
          if (!(std::fabs(u - r) <= std::fabs(r) * (1.0f / 128.0f) + 1e-6f)) {
            ++bad;
          }
        }
        std::printf("[gemm_i8c]   %-24s %zu of %zu past a bf16 ulp\n",
                    nm.c_str(), bad, ref.size());
        EXPECT_TRUE(bad == 0);
      }
    }
    // The converter from a g64 affine w8 pack, in place, against its
    // replica (dequantize, requantize at GC): one code at most.
    for (int GC : kGroups) {
      const int G = K / GC, G64 = K / 64;
      ComputeFunction fc = fn(lq, "affine_to_i8c_w8_gc" + std::to_string(GC));
      if (!fc.valid()) { ASSERT_TRUE(false); continue; }
      std::vector<std::uint8_t> codes((std::size_t)N * K);
      std::vector<std::uint16_t> sc((std::size_t)N * G64), bi(sc.size());
      std::uniform_int_distribution<int> cd(0, 255);
      std::uniform_real_distribution<float> sd(0.0005f, 0.002f);
      std::uniform_real_distribution<float> bd(-0.15f, -0.05f);
      for (auto& c : codes) { c = (std::uint8_t)cd(rng); }
      for (std::size_t i = 0; i < sc.size(); ++i) {
        sc[i] = f32_to_bf16(sd(rng));
        bi[i] = f32_to_bf16(bd(rng));
      }
      SharedBuffer cb = up(codes.data(), codes.size());
      SharedBuffer sb = up(sc.data(), sc.size() * 2);
      SharedBuffer bb = up(bi.data(), bi.size() * 2);
      SharedBuffer cs = up(nullptr, (std::size_t)N * G * 2);
      run([&](ComputeEncoder& e) {
        e.set_buffer(0, cb); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, cb); e.set_buffer(4, cs); e.set_constant(5, K);
        rows_pass(e, fc, N, K, GC);
      });
      const auto* q = static_cast<const std::int8_t*>(cb.contents());
      std::size_t off1 = 0, offmore = 0;
      for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
          std::vector<float> v(GC);
          float am = 0.0f;
          for (int i = 0; i < GC; ++i) {
            const int k = g * GC + i;
            const std::size_t s = (std::size_t)n * G64 + k / 64;
            v[i] = (float)codes[(std::size_t)n * K + k] * bf16_to_f32(sc[s]) +
                   bf16_to_f32(bi[s]);
            am = std::max(am, std::fabs(v[i]));
          }
          const float inv = am > 0.0f ? 127.0f / am : 0.0f;
          for (int i = 0; i < GC; ++i) {
            const int r = std::clamp((int)std::nearbyint(v[i] * inv), -127,
                                     127);
            const int d =
                std::abs(r - (int)q[(std::size_t)n * K + g * GC + i]);
            off1 += d == 1 ? 1 : 0;
            offmore += d > 1 ? 1 : 0;
          }
        }
      }
      std::printf("[gemm_i8c] affine_to_i8c_w8_gc%-4d in place: %zu codes "
                  "1 off, %zu more\n", GC, off1, offmore);
      EXPECT_TRUE(offmore == 0 && off1 * 1000 < codes.size());
    }
  }

  if (std::getenv("VPIPE_I8_COARSE_STUDY") == nullptr) { return; }

  // ---- (2) accuracy of the routes, against f64 ---------------------------
  ComputeFunction f_qs = fn(lq, "quant_f16_i8s_row_g64");
  ComputeFunction f_gs = fn(lg, "gemm_i8s_w8g64");
  ComputeFunction f_bf = fn(lg, "dense_gemm_mma_t_n128_f16");
  if (!f_qs.valid() || !f_gs.valid() || !f_bf.valid()) { return; }
  // A w8 g64 affine pack of a bf16 weight, as model-quantize writes one:
  // per (channel, 64) min/max, codes 0..255 -- then the signed layout.
  auto pack_g64 = [](const std::vector<std::uint16_t>& w, int N, int K,
                     std::vector<std::uint8_t>* codes,
                     std::vector<std::uint16_t>* sc,
                     std::vector<std::uint16_t>* bi) {
    const int G = K / 64;
    codes->resize((std::size_t)N * K);
    sc->resize((std::size_t)N * G);
    bi->resize((std::size_t)N * G);
    for (int n = 0; n < N; ++n) {
      for (int g = 0; g < G; ++g) {
        float lo = 1e30f, hi = -1e30f;
        for (int k = g * 64; k < g * 64 + 64; ++k) {
          const float v = bf16_to_f32(w[(std::size_t)n * K + k]);
          lo = std::min(lo, v);
          hi = std::max(hi, v);
        }
        const float s = std::max((hi - lo) / 255.0f, 1e-12f);
        const std::uint16_t sb = f32_to_bf16(s), bb = f32_to_bf16(lo);
        (*sc)[(std::size_t)n * G + g] = sb;
        (*bi)[(std::size_t)n * G + g] = bb;
        const float sf = bf16_to_f32(sb), bf = bf16_to_f32(bb);
        for (int k = g * 64; k < g * 64 + 64; ++k) {
          const float v = bf16_to_f32(w[(std::size_t)n * K + k]);
          (*codes)[(std::size_t)n * K + k] = (std::uint8_t)std::clamp(
              (int)std::nearbyint((v - bf) / sf), 0, 255);
        }
      }
    }
  };
  {
    const int M = 256, N = 512, K = 4096;
    struct Case { const char* name; int every; float boost; };
    for (const Case& cs : {Case{"no outliers", 0, 1.0f},
                           Case{"1/64 channels x24", 64, 24.0f},
                           Case{"1/512 channels x64", 512, 64.0f}}) {
      std::mt19937 rng(11);
      std::normal_distribution<float> nd(0.0f, 1.0f);
      std::vector<std::uint16_t> x((std::size_t)M * K), w((std::size_t)N * K);
      for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
          float v = nd(rng);
          if (cs.every > 0 && (k % cs.every) == 37) { v *= cs.boost; }
          x[(std::size_t)m * K + k] = f32_to_bf16(v);
        }
      }
      for (auto& v : w) { v = f32_to_bf16(nd(rng) * 0.02f); }
      std::vector<double> ref((std::size_t)M * N, 0.0);
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          double a = 0.0;
          for (int k = 0; k < K; ++k) {
            a += (double)bf16_to_f32(x[(std::size_t)m * K + k]) *
                 (double)bf16_to_f32(w[(std::size_t)n * K + k]);
          }
          ref[(std::size_t)m * N + n] = a;
        }
      }
      auto rel = [&](const SharedBuffer& y) {
        const auto* yp = static_cast<const std::uint16_t*>(y.contents());
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
          const double d = (double)bf16_to_f32(yp[i]) - ref[i];
          num += d * d;
          den += ref[i] * ref[i];
        }
        return std::sqrt(num / den);
      };
      SharedBuffer xb = up(x.data(), x.size() * 2);
      SharedBuffer wb = up(w.data(), w.size() * 2);
      SharedBuffer y = up(nullptr, (std::size_t)M * N * 2);
      // bf16 floor.
      run([&](ComputeEncoder& e) {
        e.set_function(f_bf);
        e.set_buffer(0, xb); e.set_buffer(1, wb); e.set_buffer(2, wb);
        e.set_buffer(3, y);
        e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
        e.set_constant(7, 0);
        e.dispatch({(unsigned)(((N + 127) / 128) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      });
      const double r_bf = rel(y);
      // g64: the pack, signed.
      std::vector<std::uint8_t> codes;
      std::vector<std::uint16_t> sc, bi;
      pack_g64(w, N, K, &codes, &sc, &bi);
      std::vector<std::uint8_t> scodes(codes.size());
      std::vector<std::uint16_t> zb(sc.size());
      for (std::size_t i = 0; i < codes.size(); ++i) {
        scodes[i] = (std::uint8_t)(codes[i] ^ 0x80u);
      }
      for (std::size_t i = 0; i < sc.size(); ++i) {
        zb[i] = f32_to_bf16(bf16_to_f32(bi[i]) +
                            128.0f * bf16_to_f32(sc[i]));
      }
      const int G64 = K / 64;
      SharedBuffer scb = up(scodes.data(), scodes.size());
      SharedBuffer ssb = up(sc.data(), sc.size() * 2);
      SharedBuffer szb = up(zb.data(), zb.size() * 2);
      SharedBuffer xq = up(nullptr, (std::size_t)M * K);
      SharedBuffer as = up(nullptr, (std::size_t)M * G64 * 2);
      SharedBuffer uf = up(nullptr, (std::size_t)M * G64 * 2);
      run([&](ComputeEncoder& e) {
        e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, as);
        e.set_buffer(3, uf); e.set_constant(4, K);
        rows_pass(e, f_qs, M);
        e.set_function(f_gs);
        e.set_buffer(0, xq); e.set_buffer(1, scb); e.set_buffer(2, as);
        e.set_buffer(3, ssb); e.set_buffer(4, szb); e.set_buffer(5, uf);
        e.set_buffer(6, y);
        e.set_constant(7, K); e.set_constant(8, N); e.set_constant(9, M);
        e.set_buffer(10, ssb); e.set_constant(11, 0);
        e.dispatch({(unsigned)(((N + 63) / 64) * 128),
                    (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
      });
      const double r_g64 = rel(y);
      std::printf("[gemm_i8c] accuracy, %s (%dx%dx%d), rel-L2 vs f64: bf16 "
                  "%.3e, g64 route %.3e\n", cs.name, M, N, K, r_bf, r_g64);
      for (int GC : kGroups) {
        const int G = K / GC;
        const std::string sfx = std::to_string(GC);
        ComputeFunction fq = fn(lq, "quant_i8_row_gc" + sfx);
        ComputeFunction fc = fn(lq, "affine_to_i8c_w8_gc" + sfx);
        ComputeFunction fg = fn(lg, "gemm_i8c_g" + sfx + "_t64x64");
        SharedBuffer cxq = up(nullptr, (std::size_t)M * K);
        SharedBuffer cas = up(nullptr, (std::size_t)M * G * 2);
        SharedBuffer pwq = up(codes.data(), codes.size());
        SharedBuffer pws = up(nullptr, (std::size_t)N * G * 2);
        SharedBuffer bwq = up(nullptr, (std::size_t)N * K);
        SharedBuffer bws = up(nullptr, (std::size_t)N * G * 2);
        SharedBuffer ub = up(sc.data(), sc.size() * 2);
        SharedBuffer bb = up(bi.data(), bi.size() * 2);
        SharedBuffer y2 = up(nullptr, (std::size_t)M * N * 2);
        run([&](ComputeEncoder& e) {
          e.set_buffer(0, xb); e.set_buffer(1, cxq); e.set_buffer(2, cas);
          e.set_constant(3, K);
          rows_pass(e, fq, M, K, GC);
          e.set_buffer(0, wb); e.set_buffer(1, bwq); e.set_buffer(2, bws);
          e.set_constant(3, K);
          rows_pass(e, fq, N, K, GC);
          e.set_buffer(0, pwq); e.set_buffer(1, ub); e.set_buffer(2, bb);
          e.set_buffer(3, pwq); e.set_buffer(4, pws); e.set_constant(5, K);
          rows_pass(e, fc, N, K, GC);
          gemm(e, fg, 0, cxq, pwq, cas, pws, y, M, N, K);
          gemm(e, fg, 0, cxq, bwq, cas, bws, y2, M, N, K);
        });
        std::printf("[gemm_i8c]   gc%-4d from the w8 pack %.3e (%.2fx g64), "
                    "from bf16 %.3e (%.2fx)\n", GC, rel(y), rel(y) / r_g64,
                    rel(y2), rel(y2) / r_g64);
        // ROTATED: the same, with the group Walsh-Hadamard on both sides.
        ComputeFunction fqr = fn(lq, "quant_i8_row_gc" + sfx + "_r");
        ComputeFunction fcr = fn(lq, "affine_to_i8c_w8_gc" + sfx + "_r");
        ComputeFunction fdr = fn(lq, "i8c_dequant_gc" + sfx + "_r");
        if (!fqr.valid() || !fcr.valid() || !fdr.valid()) { continue; }
        SharedBuffer pwq2 = up(codes.data(), codes.size());
        SharedBuffer wrt = up(nullptr, (std::size_t)N * K * 2);
        run([&](ComputeEncoder& e) {
          e.set_buffer(0, xb); e.set_buffer(1, cxq); e.set_buffer(2, cas);
          e.set_constant(3, K);
          rows_pass(e, fqr, M, K, GC);
          e.set_buffer(0, wb); e.set_buffer(1, bwq); e.set_buffer(2, bws);
          e.set_constant(3, K);
          rows_pass(e, fqr, N, K, GC);
          e.set_buffer(0, pwq2); e.set_buffer(1, ub); e.set_buffer(2, bb);
          e.set_buffer(3, pwq2); e.set_buffer(4, pws); e.set_constant(5, K);
          rows_pass(e, fcr, N, K, GC);
          gemm(e, fg, 0, cxq, pwq2, cas, pws, y, M, N, K);
          gemm(e, fg, 0, cxq, bwq, cas, bws, y2, M, N, K);
          // The rotated dequant must give the weight back.
          e.set_buffer(0, bwq); e.set_buffer(1, bws); e.set_buffer(2, wrt);
          e.set_constant(3, K);
          rows_pass(e, fdr, N, K, GC);
        });
        double wn = 0.0, wd = 0.0;
        const auto* wrp = static_cast<const std::uint16_t*>(wrt.contents());
        for (std::size_t i = 0; i < w.size(); ++i) {
          const double d = (double)bf16_to_f32(wrp[i]) - bf16_to_f32(w[i]);
          wn += d * d;
          wd += (double)bf16_to_f32(w[i]) * bf16_to_f32(w[i]);
        }
        std::printf("[gemm_i8c]   gc%-4d ROTATED: from the w8 pack %.3e "
                    "(%.2fx g64), from bf16 %.3e (%.2fx); weight round trip "
                    "%.3e\n", GC, rel(y), rel(y) / r_g64, rel(y2),
                    rel(y2) / r_g64, std::sqrt(wn / wd));
        EXPECT_TRUE(std::sqrt(wn / wd) < 2e-2);
        // THE ROTATION BLOCK at g512: both operands rotated in blocks of
        // RB (the same entry quantizes the bf16 weight).
        if (GC != 512) { continue; }
        for (int RB : {16, 64, 128, 256}) {
          ComputeFunction fb =
              fn(lq, "quant_i8_row_gc512_rb" + std::to_string(RB));
          if (!fb.valid()) { continue; }
          run([&](ComputeEncoder& e) {
            e.set_buffer(0, xb); e.set_buffer(1, cxq); e.set_buffer(2, cas);
            e.set_constant(3, K);
            rows_pass(e, fb, M, K, GC);
            e.set_buffer(0, wb); e.set_buffer(1, bwq); e.set_buffer(2, bws);
            e.set_constant(3, K);
            rows_pass(e, fb, N, K, GC);
            gemm(e, fg, 0, cxq, bwq, cas, bws, y2, M, N, K);
          });
          std::printf("[gemm_i8c]   gc512 rotated in blocks of %-3d: from "
                      "bf16 %.3e (%.2fx g64)\n", RB, rel(y2),
                      rel(y2) / r_g64);
        }
      }
    }
  }

  // ---- (2b) speed at Qwen-Image-2.1's shapes -----------------------------
  struct Sh { int M, N, K; const char* tag; };
  for (const Sh& sh : {Sh{4096, 4096, 4096, "q/k/v/out"},
                       Sh{4096, 12288, 4096, "ff gate/up"},
                       Sh{4096, 4096, 12288, "ff down"}}) {
    const int M = sh.M, N = sh.N, K = sh.K, G64 = K / 64;
    std::vector<std::uint16_t> hx((std::size_t)M * K);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : hx) { v = f32_to_bf16(nd(rng)); }
    SharedBuffer xb = up(hx.data(), hx.size() * 2);
    SharedBuffer y = up(nullptr, (std::size_t)M * N * 2);
    std::uniform_int_distribution<int> cd(-127, 127);
    auto rnd8 = [&](std::size_t n) {
      SharedBuffer b = up(nullptr, n);
      auto* p = static_cast<std::int8_t*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { p[i] = (std::int8_t)cd(rng); }
      return b;
    };
    auto rndbf = [&](std::size_t n, float s) {
      std::vector<std::uint16_t> v(n);
      for (auto& e : v) { e = f32_to_bf16(s * (1.0f + 0.1f * nd(rng))); }
      return up(v.data(), n * 2);
    };
    SharedBuffer w8[2] = {rnd8((std::size_t)N * K), rnd8((std::size_t)N * K)};
    SharedBuffer xq = up(nullptr, (std::size_t)M * K);
    SharedBuffer as = up(nullptr, (std::size_t)M * G64 * 2);
    SharedBuffer uf = up(nullptr, (std::size_t)M * G64 * 2);
    SharedBuffer s64[2] = {rndbf((std::size_t)N * G64, 1e-3f),
                           rndbf((std::size_t)N * G64, 1e-3f)};
    struct Arm { std::string name; std::function<void(ComputeEncoder&, int)> f;
                 std::vector<double> ms; };
    std::vector<Arm> arms;
    arms.push_back({"g64 signed (today)", [&](ComputeEncoder& e, int wi) {
      e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, as);
      e.set_buffer(3, uf); e.set_constant(4, K);
      rows_pass(e, f_qs, M);
      e.set_function(f_gs);
      e.set_buffer(0, xq); e.set_buffer(1, w8[wi]); e.set_buffer(2, as);
      e.set_buffer(3, s64[wi]); e.set_buffer(4, s64[wi]);
      e.set_buffer(5, uf); e.set_buffer(6, y);
      e.set_constant(7, K); e.set_constant(8, N); e.set_constant(9, M);
      e.set_buffer(10, s64[wi]); e.set_constant(11, 0);
      e.dispatch({(unsigned)(((N + 63) / 64) * 128),
                  (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
    }, {}});
    // Kept alive for the arms' lifetime: the closures hold references.
    static std::vector<SharedBuffer> keep;
    static std::vector<ComputeFunction> fkeep;
    keep.clear();
    fkeep.clear();
    keep.reserve(64);
    fkeep.reserve(64);
    for (int GC : kGroups) {
      if (K % GC != 0) { continue; }
      const int G = K / GC;
      const std::string sfx = std::to_string(GC);
      fkeep.push_back(fn(lq, "quant_i8_row_gc" + sfx));
      const ComputeFunction* fq = &fkeep.back();
      keep.push_back(up(nullptr, (std::size_t)M * G * 2));
      const SharedBuffer* cas = &keep.back();
      keep.push_back(rndbf((std::size_t)N * G, 1e-3f));
      const SharedBuffer* cws = &keep.back();
      fkeep.push_back(fn(lq, "quant_i8_row_gc" + sfx + "_r"));
      const ComputeFunction* fqr = &fkeep.back();
      for (int t = 0; t < 4; ++t) {
        fkeep.push_back(fn(lg, "gemm_i8c_g" + sfx + "_t" + kTiles[t]));
        const ComputeFunction* fg = &fkeep.back();
        for (int rot = 0; rot < 2; ++rot) {
          // The rotated quantizer at the two tiles that matter only.
          if (rot == 1 && t != 1 && t != 2) { continue; }
          const ComputeFunction* qf = rot == 1 ? fqr : fq;
          arms.push_back({"gc" + sfx + " " + kTiles[t] + (rot ? " rot" : ""),
                          [&, qf, fg, cas, cws, t, GC](ComputeEncoder& e,
                                                       int wi) {
            e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, *cas);
            e.set_constant(3, K);
            rows_pass(e, *qf, M, K, GC);
            gemm(e, *fg, t, xq, w8[wi], *cas, *cws, y, M, N, K);
          }, {}});
        }
      }
    }
    // The quantizers alone, plain and rotated, at this shape's input.
    for (int GC : kGroups) {
      if (K % GC != 0) { continue; }
      const std::string sfx = std::to_string(GC);
      keep.push_back(up(nullptr, (std::size_t)M * (K / GC) * 2));
      const SharedBuffer* qs = &keep.back();
      for (int rot = 0; rot < 2 + (GC == 512 ? 4 : 0); ++rot) {
        static const int kRb[4] = {16, 64, 128, 256};
        const std::string fname =
            rot < 2 ? "quant_i8_row_gc" + sfx + (rot ? "_r" : "")
                    : "quant_i8_row_gc512_rb" + std::to_string(kRb[rot - 2]);
        const std::string aname =
            rot < 2 ? "quant gc" + sfx + (rot ? " rot" : "")
                    : "quant gc512 rb" + std::to_string(kRb[rot - 2]);
        fkeep.push_back(fn(lq, fname));
        const ComputeFunction* qf = &fkeep.back();
        arms.push_back({aname, [&, qf, qs, GC](ComputeEncoder& e, int) {
          e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, *qs);
          e.set_constant(3, K);
          rows_pass(e, *qf, M, K, GC);
        }, {}});
      }
    }
    // VPIPE_I8_COARSE_ARMS="gc512 128x64 rot,gc256 64x128 rot": only the
    // named arms beside today's, for a box that must not run long.
    if (const char* only = std::getenv("VPIPE_I8_COARSE_ARMS")) {
      std::vector<Arm> kept;
      for (std::size_t i = 0; i < arms.size(); ++i) {
        if (i == 0 || std::strstr(only, arms[i].name.c_str()) != nullptr) {
          kept.push_back(std::move(arms[i]));
        }
      }
      arms = std::move(kept);
    }
    const double gop = 2.0 * M * (double)N * K * 1e-9;
    for (int r = 0; r < 4; ++r) {
      for (std::size_t k = 0; k < arms.size(); ++k) {
        Arm& a = arms[(r % 2 == 0) ? k : arms.size() - 1 - k];
        run([&](ComputeEncoder& e) { a.f(e, 0); });
        const auto t0 = std::chrono::steady_clock::now();
        run([&](ComputeEncoder& e) {
          for (int i = 0; i < 6; ++i) { a.f(e, i & 1); }
        });
        a.ms.push_back(secs_(t0, std::chrono::steady_clock::now()) * 1e3 /
                       6.0);
      }
    }
    std::printf("[gemm_i8c] speed %s M=%d N=%d K=%d (quantizer + GEMM)\n",
                sh.tag, M, N, K);
    double base = 0.0;
    for (Arm& a : arms) {
      std::sort(a.ms.begin(), a.ms.end());
      const double ms = a.ms[a.ms.size() / 2];
      if (base == 0.0) { base = ms; }
      std::printf("[gemm_i8c]   %-20s %7.3f ms %6.1f TOP/s %.3fx\n",
                  a.name.c_str(), ms, gop / ms, base / ms);
    }
  }
}

// THE PER-TOKEN TIER (gemm_i8pt_*, the i8x layout): int8 weights per
// output channel and activations per token, both over all of K, so the
// GEMM is one int32 accumulation with the scales applied at the store.
// (1) every quantizer and GEMM entry against its CPU replica -- always.
// (2) VPIPE_I8_PT_SPEED=1: quantizer + GEMM at Qwen-Image-2.1's shapes
// against the rotated g512 tier, interleaved, medians of 4 rounds;
// VPIPE_I8_PT_ARMS="a,b" keeps only the named arms beside g512.
TEST(gemm_i8, per_token) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lg = mc->load_library("dense_gemm_mma_bf16");
  ComputeLibrary lq = mc->load_library("affine_dequant_bf16");
  auto up = [&](const void* p, std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n);
    if (!b.empty() && p != nullptr) { std::memcpy(b.contents(), p, n); }
    return b;
  };
  auto run = [&](const std::function<void(ComputeEncoder&)>& f) {
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder e = st.begin_compute(); f(e); }
    st.commit().wait();
  };
  // Every GEMM entry: name, tile, register layout, K step.
  // reg: a register-fragment kernel, wm x wn simdgroups; flat: dispatched
  // as one 1-D threadgroup of 32 * wm * wn threads instead of
  // {32, wn, wm} (the kernel reads its simdgroup index either way).
  struct G { std::string name; int bm, bn, sg, bk; bool reg;
             int wm = 0, wn = 0; bool flat = false; };
  std::vector<G> gs;
  for (const char* t : {"64x64", "64x128", "128x64", "128x128"}) {
    const int bm = t[0] == '6' ? 64 : 128;
    const int bn = std::strcmp(t + (bm == 64 ? 3 : 4), "64") == 0 ? 64 : 128;
    const int sg = bm * bn == 64 * 64 ? 4 : 8;
    gs.push_back({std::string("gemm_i8pt_t") + t + "_k128", bm, bn, sg, 128,
                  false});
  }
  for (int bk : {64, 256}) {
    gs.push_back({"gemm_i8pt_t128x64_k" + std::to_string(bk), 128, 64, 8,
                  bk, false});
    gs.push_back({"gemm_i8pt_t128x128_k" + std::to_string(bk), 128, 128,
                  8, bk, false});
  }
  // The permuted-K register kernels (32x32 a simdgroup): MLX's blocked
  // loop (the tier's), then the double-buffered A/B, each 3-D and flat.
  struct P { const char* n; int bm, bn; };
  for (const P& p : {P{"m128x64", 128, 64}, P{"m64x128", 64, 128},
                     P{"d128x64", 128, 64}, P{"d64x128", 64, 128}}) {
    const int wm = p.bm / 32, wn = p.bn / 32;
    for (int flat = 0; flat < 2; ++flat) {
      gs.push_back({std::string("gemm_i8pt_") + p.n, p.bm, p.bn, wm * wn,
                    64, true, wm, wn, flat != 0});
    }
  }
  auto gfn = [&](const G& g, int) { return lg.function(g.name.c_str()); };
  auto gemm = [&](ComputeEncoder& e, const ComputeFunction& f, const G& g,
                  const SharedBuffer& xq, const SharedBuffer& wq,
                  const SharedBuffer& as, const SharedBuffer& ws,
                  const SharedBuffer& y, const SharedBuffer& bias,
                  bool has_bias, int M, int N, int K) {
    e.set_function(f);
    e.set_buffer(0, xq); e.set_buffer(1, wq); e.set_buffer(2, as);
    e.set_buffer(3, ws); e.set_buffer(4, y);
    e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, M);
    e.set_buffer(8, bias); e.set_constant(9, has_bias ? 1 : 0);
    const unsigned nt = (unsigned)((N + g.bn - 1) / g.bn);
    const unsigned mt = (unsigned)((M + g.bm - 1) / g.bm);
    if (g.reg && g.flat) {
      const unsigned tg = 32u * (unsigned)(g.wm * g.wn);
      e.dispatch({nt * tg, mt, 1}, {tg, 1, 1});
    } else if (g.reg) {
      const unsigned wn = (unsigned)g.wn, wm = (unsigned)g.wm;
      e.dispatch({nt * 32u, mt * wn, wm}, {32u, wn, wm});
    } else {
      const unsigned tg = (unsigned)(32 * g.sg);
      e.dispatch({nt * tg, mt, 1}, {tg, 1, 1});
    }
  };
  // K > 0: the quantizers' grid, a threadgroup of K/16 threads a row;
  // K < 0: the rotated ones', a simdgroup per 256-block of -K.
  auto rows_pass = [&](ComputeEncoder& e, const ComputeFunction& f,
                       int rows, int K = 0) {
    e.set_function(f);
    const unsigned T =
        K < 0   ? 32u * (unsigned)std::min(-K / 256, 32)
        : K > 0 ? (unsigned)std::max(32, (K / 16 + 31) / 32 * 32)
                : 256u;
    e.dispatch({T, (unsigned)rows, 1}, {T, 1, 1});
  };
  ComputeFunction fq = lq.function("quant_i8_tok");
  ComputeFunction fsq = lq.function("swiglu_quant_i8_tok");
  ComputeFunction faq = lq.function("affine_to_i8pt_w8");
  ComputeFunction fdq = lq.function("i8pt_dequant");
  ComputeFunction fqr = lq.function("quant_i8_tok_r");
  ComputeFunction fsqr = lq.function("swiglu_quant_i8_tok_r");
  ComputeFunction faqr = lq.function("affine_to_i8pt_w8_r");
  ComputeFunction fdqr = lq.function("i8pt_dequant_r");
  if (!fq.valid() || !fsq.valid() || !faq.valid() || !fdq.valid() ||
      !fqr.valid() || !fsqr.valid() || !faqr.valid() || !fdqr.valid()) {
    ASSERT_TRUE(false);
    return;
  }

  // ---- (1) replicas -------------------------------------------------------
  {
    // Edges on both sides of every tile, K a multiple of every step.
    const int M = 203, N = 196, K = 1024;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::uint16_t> x((std::size_t)M * K), x2(x.size());
    for (auto& v : x) { v = f32_to_bf16(nd(rng)); }
    for (auto& v : x2) { v = f32_to_bf16(nd(rng)); }
    SharedBuffer xb = up(x.data(), x.size() * 2);
    SharedBuffer x2b = up(x2.data(), x2.size() * 2);
    SharedBuffer xq = up(nullptr, (std::size_t)M * K);
    SharedBuffer as = up(nullptr, (std::size_t)M * 4);
    SharedBuffer sq = up(nullptr, (std::size_t)M * K);
    SharedBuffer ss = up(nullptr, (std::size_t)M * 4);
    // A g64 w8 pack for the weight side.
    const int G64 = K / 64;
    std::vector<std::uint8_t> codes((std::size_t)N * K);
    std::vector<std::uint16_t> psc((std::size_t)N * G64), pbi(psc.size());
    std::uniform_int_distribution<int> cd(0, 255);
    for (auto& c : codes) { c = (std::uint8_t)cd(rng); }
    for (std::size_t i = 0; i < psc.size(); ++i) {
      const float s = 1e-3f * (1.0f + 0.5f * std::fabs(nd(rng)));
      psc[i] = f32_to_bf16(s);
      pbi[i] = f32_to_bf16(-128.0f * s + 0.02f * nd(rng));
    }
    SharedBuffer cb = up(codes.data(), codes.size());
    SharedBuffer scb = up(psc.data(), psc.size() * 2);
    SharedBuffer bib = up(pbi.data(), pbi.size() * 2);
    SharedBuffer wq = up(nullptr, (std::size_t)N * K);
    SharedBuffer ws = up(nullptr, (std::size_t)N * 4);
    SharedBuffer wd = up(nullptr, (std::size_t)N * K * 2);
    // In place too: the conversion may overwrite its own codes.
    SharedBuffer cip = up(codes.data(), codes.size());
    SharedBuffer wsip = up(nullptr, (std::size_t)N * 4);
    run([&](ComputeEncoder& e) {
      e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, as);
      e.set_constant(3, K);
      rows_pass(e, fq, M, K);
      e.set_buffer(0, xb); e.set_buffer(1, x2b); e.set_buffer(2, sq);
      e.set_buffer(3, ss); e.set_constant(4, K);
      rows_pass(e, fsq, M, K);
      e.set_buffer(0, cb); e.set_buffer(1, scb); e.set_buffer(2, bib);
      e.set_buffer(3, wq); e.set_buffer(4, ws); e.set_constant(5, K);
      rows_pass(e, faq, N, K);
      e.set_buffer(0, cip); e.set_buffer(1, scb); e.set_buffer(2, bib);
      e.set_buffer(3, cip); e.set_buffer(4, wsip); e.set_constant(5, K);
      rows_pass(e, faq, N, K);
      e.set_buffer(0, wq); e.set_buffer(1, ws); e.set_buffer(2, wd);
      e.set_constant(3, K);
      rows_pass(e, fdq, N);
    });
    // Rows -> (codes, scale) on the CPU, as the kernels do it.
    auto quant_rows = [&](const std::vector<float>& v, int rows,
                          const std::int8_t* q, const float* sc,
                          const char* what) {
      std::size_t off1 = 0, offmore = 0, sbad = 0;
      for (int r = 0; r < rows; ++r) {
        float am = 0.0f;
        for (int k = 0; k < K; ++k) {
          am = std::max(am, std::fabs(v[(std::size_t)r * K + k]));
        }
        if (sc[r] != am * (1.0f / 127.0f)) { ++sbad; }
        const float inv = am > 0.0f ? 127.0f / am : 0.0f;
        for (int k = 0; k < K; ++k) {
          const std::size_t i = (std::size_t)r * K + k;
          const int c = std::clamp((int)std::nearbyint(v[i] * inv), -127,
                                   127);
          const int d = std::abs(c - (int)q[i]);
          off1 += d == 1 ? 1 : 0;
          offmore += d > 1 ? 1 : 0;
        }
      }
      std::printf("[gemm_i8pt] %-22s scales off %zu/%d, codes off by 1 "
                  "%zu, by more %zu of %zu\n", what, sbad, rows, off1,
                  offmore, (std::size_t)rows * K);
      EXPECT_TRUE(sbad == 0);
      EXPECT_TRUE(offmore == 0);
      EXPECT_TRUE(off1 * 10000 <= (std::size_t)rows * K);
    };
    std::vector<float> xf(x.size()), hf(x.size()), wf((std::size_t)N * K);
    for (std::size_t i = 0; i < x.size(); ++i) {
      xf[i] = bf16_to_f32(x[i]);
      const float g = bf16_to_f32(x[i]), u = bf16_to_f32(x2[i]);
      hf[i] = bf16_to_f32(f32_to_bf16(g / (1.0f + std::exp(-g)) * u));
    }
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        const std::size_t i = (std::size_t)n * K + k;
        const std::size_t gi = (std::size_t)n * G64 + k / 64;
        wf[i] = bf16_to_f32(f32_to_bf16(
            (float)codes[i] * bf16_to_f32(psc[gi]) + bf16_to_f32(pbi[gi])));
      }
    }
    const auto* q8 = static_cast<const std::int8_t*>(xq.contents());
    const auto* asp = static_cast<const float*>(as.contents());
    const auto* w8 = static_cast<const std::int8_t*>(wq.contents());
    const auto* wsp = static_cast<const float*>(ws.contents());
    quant_rows(xf, M, q8, asp, "quant_i8_tok");
    quant_rows(hf, M, static_cast<const std::int8_t*>(sq.contents()),
               static_cast<const float*>(ss.contents()),
               "swiglu_quant_i8_tok");
    quant_rows(wf, N, w8, wsp, "affine_to_i8pt_w8");
    EXPECT_TRUE(std::memcmp(cip.contents(), wq.contents(),
                            (std::size_t)N * K) == 0);
    EXPECT_TRUE(std::memcmp(wsip.contents(), ws.contents(),
                            (std::size_t)N * 4) == 0);
    std::size_t dbad = 0;
    const auto* wdp = static_cast<const std::uint16_t*>(wd.contents());
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        const std::size_t i = (std::size_t)n * K + k;
        if (wdp[i] != f32_to_bf16((float)w8[i] * wsp[n])) { ++dbad; }
      }
    }
    EXPECT_TRUE(dbad == 0);
    // The GEMMs: integer sums, scaled as the store does.
    std::vector<std::uint16_t> bias(N);
    for (auto& b : bias) { b = f32_to_bf16(0.1f * nd(rng)); }
    SharedBuffer bb = up(bias.data(), bias.size() * 2);
    std::vector<std::int32_t> acc((std::size_t)M * N);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        std::int32_t p = 0;
        for (int k = 0; k < K; ++k) {
          p += (std::int32_t)q8[(std::size_t)m * K + k] *
               (std::int32_t)w8[(std::size_t)n * K + k];
        }
        acc[(std::size_t)m * N + n] = p;
      }
    }
    SharedBuffer y = up(nullptr, (std::size_t)M * N * 2);
    for (const G& g : gs) {
      ComputeFunction f = gfn(g, K);
      if (!f.valid()) {
        std::printf("[gemm_i8pt] %s missing\n", g.name.c_str());
        ASSERT_TRUE(false);
        continue;
      }
      for (int hb = 0; hb < 2; ++hb) {
        std::memset(y.contents(), 0xff, (std::size_t)M * N * 2);
        run([&](ComputeEncoder& e) {
          gemm(e, f, g, xq, wq, as, ws, y, bb, hb != 0, M, N, K);
        });
        const auto* yp = static_cast<const std::uint16_t*>(y.contents());
        std::size_t bad = 0;
        double worst = 0.0;
        for (int m = 0; m < M; ++m) {
          for (int n = 0; n < N; ++n) {
            const std::size_t i = (std::size_t)m * N + n;
            const float b = hb != 0 ? bf16_to_f32(bias[n]) : 0.0f;
            const float r = std::fma((float)acc[i] * asp[m], wsp[n], b);
            const float gv = bf16_to_f32(yp[i]);
            // One bf16 ulp: the edge columns' scalar store rounds in a
            // different order.
            const double d = std::fabs((double)gv - r) /
                             std::max(std::fabs((double)r), 1e-6);
            worst = std::max(worst, d);
            if (yp[i] != f32_to_bf16(r) && d > 1.0 / 128.0) { ++bad; }
          }
        }
        std::printf("[gemm_i8pt] %-28s%s bias %d: %zu bad, worst rel "
                    "%.2e\n", g.name.c_str(), g.flat ? " flat" : "", hb, bad,
                    worst);
        EXPECT_TRUE(bad == 0);
      }
    }
    // THE ROTATED QUANTIZERS: each 256-block Walsh-Hadamard transformed
    // (natural order, orthonormal, the GPU's butterfly order) before the
    // row's scale is taken; the dequant transforms back.
    auto fwht = [](float* v) {
      for (int h = 1; h < 256; h <<= 1) {
        for (int i = 0; i < 256; i += 2 * h) {
          for (int j = i; j < i + h; ++j) {
            const float a = v[j], b = v[j + h];
            v[j] = a + b;
            v[j + h] = a - b;
          }
        }
      }
      for (int i = 0; i < 256; ++i) { v[i] *= 0.0625f; }
    };
    auto rotated = [&](const std::vector<float>& v) {
      std::vector<float> r = v;
      for (std::size_t o = 0; o < r.size(); o += 256) { fwht(&r[o]); }
      return r;
    };
    SharedBuffer rq = up(nullptr, (std::size_t)M * K);
    SharedBuffer rs = up(nullptr, (std::size_t)M * 4);
    SharedBuffer rsq = up(nullptr, (std::size_t)M * K);
    SharedBuffer rss = up(nullptr, (std::size_t)M * 4);
    SharedBuffer rwq = up(nullptr, (std::size_t)N * K);
    SharedBuffer rws = up(nullptr, (std::size_t)N * 4);
    SharedBuffer rwd = up(nullptr, (std::size_t)N * K * 2);
    run([&](ComputeEncoder& e) {
      e.set_buffer(0, xb); e.set_buffer(1, rq); e.set_buffer(2, rs);
      e.set_constant(3, K);
      rows_pass(e, fqr, M, -K);
      e.set_buffer(0, xb); e.set_buffer(1, x2b); e.set_buffer(2, rsq);
      e.set_buffer(3, rss); e.set_constant(4, K);
      rows_pass(e, fsqr, M, -K);
      e.set_buffer(0, cb); e.set_buffer(1, scb); e.set_buffer(2, bib);
      e.set_buffer(3, rwq); e.set_buffer(4, rws); e.set_constant(5, K);
      rows_pass(e, faqr, N, -K);
      e.set_buffer(0, rwq); e.set_buffer(1, rws); e.set_buffer(2, rwd);
      e.set_constant(3, K);
      rows_pass(e, fdqr, N, -K);
    });
    const auto* rw8 = static_cast<const std::int8_t*>(rwq.contents());
    const auto* rwsp = static_cast<const float*>(rws.contents());
    quant_rows(rotated(xf), M, static_cast<const std::int8_t*>(rq.contents()),
               static_cast<const float*>(rs.contents()), "quant_i8_tok_r");
    quant_rows(rotated(hf), M,
               static_cast<const std::int8_t*>(rsq.contents()),
               static_cast<const float*>(rss.contents()),
               "swiglu_quant_i8_tok_r");
    quant_rows(rotated(wf), N, rw8, rwsp, "affine_to_i8pt_w8_r");
    {
      std::size_t rbad = 0;
      const auto* p = static_cast<const std::uint16_t*>(rwd.contents());
      std::vector<float> row(256);
      for (int n = 0; n < N; ++n) {
        for (int b = 0; b < K; b += 256) {
          for (int k = 0; k < 256; ++k) {
            row[k] = (float)rw8[(std::size_t)n * K + b + k] * rwsp[n];
          }
          fwht(row.data());
          for (int k = 0; k < 256; ++k) {
            if (p[(std::size_t)n * K + b + k] != f32_to_bf16(row[k])) {
              ++rbad;
            }
          }
        }
      }
      std::printf("[gemm_i8pt] i8pt_dequant_r         %zu of %zu off\n",
                  rbad, (std::size_t)N * K);
      EXPECT_TRUE(rbad == 0);
    }
  }

  // ---- (1b) what the rotation buys -----------------------------------------
  // x with OUTLIER channels (1 in 64 scaled x24, as gemm_i8.coarse_tier),
  // an outlier-free weight, against the f64 product of the bf16 operands:
  // plain per token, rotated per token.
  {
    const int M = 256, N = 512, K = 4096;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::uint16_t> x((std::size_t)M * K), w((std::size_t)N * K);
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        const float s = (k % 64) == 7 ? 24.0f : 1.0f;
        x[(std::size_t)m * K + k] = f32_to_bf16(s * nd(rng));
      }
    }
    for (auto& v : w) { v = f32_to_bf16(0.02f * nd(rng)); }
    SharedBuffer xb = up(x.data(), x.size() * 2);
    SharedBuffer wb = up(w.data(), w.size() * 2);
    std::vector<double> ref((std::size_t)M * N);
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        double a = 0.0;
        for (int k = 0; k < K; ++k) {
          a += (double)bf16_to_f32(x[(std::size_t)m * K + k]) *
               bf16_to_f32(w[(std::size_t)n * K + k]);
        }
        ref[(std::size_t)m * N + n] = a;
      }
    }
    const G& g = *std::find_if(gs.begin(), gs.end(), [](const G& c) {
      return c.name == "gemm_i8pt_m128x64";
    });
    ComputeFunction fg = gfn(g, K);
    SharedBuffer xq = up(nullptr, (std::size_t)M * K);
    SharedBuffer as = up(nullptr, (std::size_t)M * 4);
    SharedBuffer wq = up(nullptr, (std::size_t)N * K);
    SharedBuffer ws = up(nullptr, (std::size_t)N * 4);
    SharedBuffer y = up(nullptr, (std::size_t)M * N * 2);
    double err[2] = {0.0, 0.0};
    for (int rot = 0; rot < 2; ++rot) {
      const ComputeFunction& q = rot ? fqr : fq;
      run([&](ComputeEncoder& e) {
        e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, as);
        e.set_constant(3, K);
        rows_pass(e, q, M, rot ? -K : K);
        e.set_buffer(0, wb); e.set_buffer(1, wq); e.set_buffer(2, ws);
        e.set_constant(3, K);
        rows_pass(e, q, N, rot ? -K : K);
        gemm(e, fg, g, xq, wq, as, ws, y, y, false, M, N, K);
      });
      const auto* yp = static_cast<const std::uint16_t*>(y.contents());
      double n2 = 0.0, d2 = 0.0;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = bf16_to_f32(yp[i]) - ref[i];
        n2 += d * d;
        d2 += ref[i] * ref[i];
      }
      err[rot] = std::sqrt(n2 / d2);
    }
    std::printf("[gemm_i8pt] outlier channels (1/64 x24), %s: plain "
                "%.3e, rotated %.3e (%.2fx)\n", g.name.c_str(), err[0],
                err[1], err[0] / err[1]);
    EXPECT_TRUE(err[1] < err[0]);
  }

  // ---- (2) speed ----------------------------------------------------------
  const char* sp = std::getenv("VPIPE_I8_PT_SPEED");
  if (sp == nullptr || std::atoi(sp) == 0) { return; }
  ComputeFunction fc_q = lq.function("quant_i8_row_gc512_r");
  ComputeFunction fc_g = lg.function("gemm_i8c_g512_t128x64");
  if (!fc_q.valid() || !fc_g.valid()) { ASSERT_TRUE(false); return; }
  struct Sh { int M, N, K; const char* tag; };
  const int rows = [] {
    const char* e = std::getenv("VPIPE_I8_PT_ROWS");
    return e != nullptr ? std::atoi(e) : 4096;
  }();
  for (const Sh& sh : {Sh{rows, 4096, 4096, "q/k/v/out"},
                       Sh{rows, 12288, 4096, "ff gate/up"},
                       Sh{rows, 4096, 12288, "ff down"}}) {
    const int M = sh.M, N = sh.N, K = sh.K;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::uint16_t> hx((std::size_t)M * K);
    for (auto& v : hx) { v = f32_to_bf16(nd(rng)); }
    SharedBuffer xb = up(hx.data(), hx.size() * 2);
    SharedBuffer y = up(nullptr, (std::size_t)M * N * 2);
    std::uniform_int_distribution<int> cd(-127, 127);
    auto rnd8 = [&](std::size_t n) {
      SharedBuffer b = up(nullptr, n);
      auto* p = static_cast<std::int8_t*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { p[i] = (std::int8_t)cd(rng); }
      return b;
    };
    auto rndf = [&](std::size_t n, bool bf) {
      SharedBuffer b = up(nullptr, n * (bf ? 2 : 4));
      for (std::size_t i = 0; i < n; ++i) {
        const float v = 1e-3f * (1.0f + 0.1f * nd(rng));
        if (bf) {
          static_cast<std::uint16_t*>(b.contents())[i] = f32_to_bf16(v);
        } else {
          static_cast<float*>(b.contents())[i] = v;
        }
      }
      return b;
    };
    // Two weights alternated, so no arm runs out of a warm cache.
    SharedBuffer w8[2] = {rnd8((std::size_t)N * K), rnd8((std::size_t)N * K)};
    SharedBuffer xq = up(nullptr, (std::size_t)M * K);
    SharedBuffer as = up(nullptr, (std::size_t)M * (K / 512) * 2 + M * 4);
    SharedBuffer cws[2] = {rndf((std::size_t)N * (K / 512), true),
                           rndf((std::size_t)N * (K / 512), true)};
    SharedBuffer pws[2] = {rndf(N, false), rndf(N, false)};
    struct Arm { std::string name; std::function<void(ComputeEncoder&, int)> f;
                 std::vector<double> ms; };
    std::vector<Arm> arms;
    arms.push_back({"g512 rot 128x64", [&](ComputeEncoder& e, int wi) {
      e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, as);
      e.set_constant(3, K);
      e.set_function(fc_q);
      e.dispatch({(unsigned)((K / 512 + 3) / 4) * 128u, (unsigned)M, 1},
                 {128, 1, 1});
      e.set_function(fc_g);
      e.set_buffer(0, xq); e.set_buffer(1, w8[wi]); e.set_buffer(2, as);
      e.set_buffer(3, cws[wi]); e.set_buffer(4, y);
      e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, M);
      e.set_buffer(8, cws[wi]); e.set_constant(9, 0);
      e.dispatch({(unsigned)((N + 63) / 64) * 256u, (unsigned)((M + 127) /
                  128), 1}, {256, 1, 1});
    }, {}});
    arms.push_back({"g512 gemm only", [&](ComputeEncoder& e, int wi) {
      e.set_function(fc_g);
      e.set_buffer(0, xq); e.set_buffer(1, w8[wi]); e.set_buffer(2, as);
      e.set_buffer(3, cws[wi]); e.set_buffer(4, y);
      e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, M);
      e.set_buffer(8, cws[wi]); e.set_constant(9, 0);
      e.dispatch({(unsigned)((N + 63) / 64) * 256u, (unsigned)((M + 127) /
                  128), 1}, {256, 1, 1});
    }, {}});
    arms.push_back({"quant tok", [&](ComputeEncoder& e, int) {
      e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, as);
      e.set_constant(3, K);
      rows_pass(e, fq, M, K);
    }, {}});
    arms.push_back({"quant tok rot", [&](ComputeEncoder& e, int) {
      e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, as);
      e.set_constant(3, K);
      rows_pass(e, fqr, M, -K);
    }, {}});
    static std::vector<ComputeFunction> fkeep;
    fkeep.clear();
    fkeep.reserve(gs.size());
    for (const G& g : gs) {
      fkeep.push_back(gfn(g, K));
      const ComputeFunction* f = &fkeep.back();
      const std::string nm = g.name.substr(std::strlen("gemm_i8pt_")) +
                             (g.flat ? " flat" : "");
      arms.push_back({nm, [&, f, g](ComputeEncoder& e, int wi) {
        e.set_buffer(0, xb); e.set_buffer(1, xq); e.set_buffer(2, as);
        e.set_constant(3, K);
        rows_pass(e, fq, M, K);
        gemm(e, *f, g, xq, w8[wi], as, pws[wi], y, y, false, M, N, K);
      }, {}});
    }
    if (const char* only = std::getenv("VPIPE_I8_PT_ARMS")) {
      std::vector<Arm> kept;
      for (std::size_t i = 0; i < arms.size(); ++i) {
        if (i == 0 || std::strstr(only, arms[i].name.c_str()) != nullptr) {
          kept.push_back(std::move(arms[i]));
        }
      }
      arms = std::move(kept);
    }
    // VPIPE_I8_PT_ITERS: dispatches a timed batch (6); a small shape
    // wants more, or the commit's own latency is the measurement.
    const int iters = [] {
      const char* e = std::getenv("VPIPE_I8_PT_ITERS");
      return e != nullptr && std::atoi(e) > 0 ? std::atoi(e) : 6;
    }();
    // VPIPE_I8_PT_ROUNDS: interleaved rounds (4), more on a box that
    // will not sit still.
    const int rounds = [] {
      const char* e = std::getenv("VPIPE_I8_PT_ROUNDS");
      return e != nullptr && std::atoi(e) > 0 ? std::atoi(e) : 4;
    }();
    const double gop = 2.0 * M * (double)N * K * 1e-9;
    for (int r = 0; r < rounds; ++r) {
      for (std::size_t k = 0; k < arms.size(); ++k) {
        Arm& a = arms[(r % 2 == 0) ? k : arms.size() - 1 - k];
        run([&](ComputeEncoder& e) { a.f(e, 0); });
        const auto t0 = std::chrono::steady_clock::now();
        run([&](ComputeEncoder& e) {
          for (int i = 0; i < iters; ++i) { a.f(e, i & 1); }
        });
        a.ms.push_back(secs_(t0, std::chrono::steady_clock::now()) * 1e3 /
                       iters);
      }
    }
    std::printf("[gemm_i8pt] speed %s M=%d N=%d K=%d (quantizer + GEMM)\n",
                sh.tag, M, N, K);
    double base = 0.0;
    for (Arm& a : arms) {
      std::sort(a.ms.begin(), a.ms.end());
      const double ms = a.ms[a.ms.size() / 2];
      if (base == 0.0) { base = ms; }
      std::printf("[gemm_i8pt]   %-20s %7.3f ms %6.1f TOP/s %.3fx\n",
                  a.name.c_str(), ms, gop / ms, base / ms);
    }
  }
}

// A W4 PACK INTO THE FAST AND TOKEN LAYOUTS (I8GemmContext::
// encode_to_fast at bits 4, out of place: the int8 codes are twice the
// nibbles' size). The w4 converters read a code as a nibble where the w8
// ones read a byte, and that is all they differ by -- so the same codes
// widened one a byte must convert BYTE-IDENTICAL through the w8 form, at
// every group and both token readings, which also settles the nibble
// order. And the layout dequantized back must be the pack's own dequant
// to within int8's step.
TEST(gemm_i8, a_w4_pack_converts_into_the_fast_layouts) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  using vpipe::genai::I8GemmContext;
  // The conversions are plain compute (no matrix cores); a box whose
  // context loaded no fast tier at all has nothing to convert into.
  I8GemmContext ctx(mc, /*want=*/true, /*bf16=*/false);
  if (!ctx.fast_converts(8, 512)) {
    std::printf("[gemm_i8 w4->fast] no fast tier on this box, skipped\n");
    return;
  }
  const int N = 80, K = 3072;
  std::mt19937 rng(11);
  std::vector<std::uint8_t> p4;
  std::vector<_Float16> sc, bi;
  std::vector<float> deq;
  std::vector<int> q;
  pack_affine_(4, N, K, rng, &p4, &sc, &bi, &deq, &q);
  // Zero-centred, as a weight is: a rotation gathers a row's mean into
  // one coefficient per block, and the skewed minima pack_affine_ draws
  // would make that coefficient -- not int8's step -- set the scale.
  {
    std::uniform_real_distribution<float> jd(-0.002f, 0.002f);
    const int G = K / 64;
    for (std::size_t i = 0; i < bi.size(); ++i) {
      bi[i] = (_Float16)(-7.5f * (float)sc[i] + jd(rng));
    }
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        const std::size_t gi = (std::size_t)n * G + k / 64;
        deq[(std::size_t)n * K + k] =
            (float)sc[gi] * (float)q[(std::size_t)n * K + k] +
            (float)bi[gi];
      }
    }
  }
  std::vector<std::uint8_t> p8(q.begin(), q.end());
  auto up = [&](const void* p, std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n);
    if (!b.empty() && p != nullptr) { std::memcpy(b.contents(), p, n); }
    return b;
  };
  auto run = [&](const std::function<bool(ComputeEncoder&)>& f) {
    CommandStream st = mc->make_command_stream();
    bool ok = false;
    { ComputeEncoder e = st.begin_compute(); ok = f(e); }
    return ok && st.commit().wait_ok(nullptr);
  };
  SharedBuffer c4 = up(p4.data(), p4.size());
  SharedBuffer c8 = up(p8.data(), p8.size());
  SharedBuffer s = up(sc.data(), sc.size() * 2);
  SharedBuffer b = up(bi.data(), bi.size() * 2);
  struct Arm { const char* tag; int gc; bool rot; };
  const int tok = I8GemmContext::kTokenGroup;
  for (const Arm& a : {Arm{"g256", 256, false}, Arm{"g512", 512, false},
                       Arm{"g1024", 1024, false}, Arm{"token", tok, false},
                       Arm{"token_r", tok, true}}) {
    ctx.set_token_rotation(a.rot);
    if (a.gc == tok && ctx.token_rotated() != a.rot) {
      std::printf("[gemm_i8 w4->fast] %s: VPIPE_I8_TOKEN_ROT pins the "
                  "other reading, skipped\n", a.tag);
      continue;
    }
    ASSERT_TRUE(ctx.fast_converts(4, a.gc));
    ASSERT_TRUE(ctx.fast_converts(8, a.gc));
    const std::size_t nk = (std::size_t)N * K;
    const std::size_t sb = I8GemmContext::fast_scale_bytes(N, K, a.gc);
    SharedBuffer q4 = mc->make_shared_buffer(nk);
    SharedBuffer q8 = mc->make_shared_buffer(nk);
    SharedBuffer s4 = mc->make_shared_buffer(sb);
    SharedBuffer s8 = mc->make_shared_buffer(sb);
    SharedBuffer back = mc->make_shared_buffer(nk * 2);
    const bool ok4 = run([&](ComputeEncoder& e) {
      return ctx.encode_to_fast(e, c4, s, b, q4, s4, a.gc, N, K, 4);
    });
    const bool ok8 = run([&](ComputeEncoder& e) {
      return ctx.encode_to_fast(e, c8, s, b, q8, s8, a.gc, N, K, 8);
    });
    const bool okd = run([&](ComputeEncoder& e) {
      return ctx.encode_dequant_fast(e, q4, s4, a.gc, back, N, K);
    });
    // In place is refused at w4 (each row's writes would land on rows
    // other threadgroups are still reading), with nothing encoded.
    SharedBuffer wide = mc->make_shared_buffer(nk);
    std::memcpy(wide.contents(), p4.data(), p4.size());
    bool encoded_in_place = true;
    run([&](ComputeEncoder& e) {
      encoded_in_place =
          ctx.encode_to_fast(e, wide, s, b, wide, s4, a.gc, N, K, 4);
      return true;
    });
    EXPECT_TRUE(!encoded_in_place);
    ASSERT_TRUE(ok4 && ok8 && okd);
    if (!(ok4 && ok8 && okd)) { continue; }
    const bool same_codes =
        std::memcmp(q4.contents(), q8.contents(), nk) == 0;
    const bool same_scales =
        std::memcmp(s4.contents(), s8.contents(), sb) == 0;
    const auto* h = static_cast<const _Float16*>(back.contents());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < nk; ++i) {
      const double d = (double)h[i] - (double)deq[i];
      num += d * d;
      den += (double)deq[i] * (double)deq[i];
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    std::printf("[gemm_i8 w4->fast] %-7s codes %s, scales %s against the "
                "w8 form; dequantized back %.3e from the pack's dequant\n",
                a.tag, same_codes ? "IDENTICAL" : "DIFFERENT",
                same_scales ? "IDENTICAL" : "DIFFERENT", rel);
    EXPECT_TRUE(same_codes);
    EXPECT_TRUE(same_scales);
    EXPECT_TRUE(rel < 2e-2);
  }
}

// WHAT UNPACKING A 4-BIT WEIGHT INSIDE THE TOKEN GEMM COSTS (the
// gemm_i8pt_wx_* study kernels, m128x64): against the int8 layout held,
//   i8 reg / tg   the int8 weight read as the tier reads it, and staged
//                 through threadgroup memory (what the staging alone costs)
//   w4 reg / tg   an int4-per-row weight (offset binary) widened to int8
//                 in registers, or once a block into threadgroup memory
//   aff reg / tg  the w4 g64 affine PACK as stored, requantized to the
//                 token_plain code per element inside the GEMM -- the held
//                 conversion (affine_to_i8pt_w4) moved into the kernel
//   cvt + gemm    the pack converted into an int8 scratch before every
//                 GEMM (affine_to_i8pt_w4, and its rotated twin for the
//                 `token` tier), then the tier's GEMM
// Correctness (any matrix-core box): every arm BYTE-IDENTICAL to the
// tier's own kernel on the same values. Timing: VPIPE_I8_W4_UNPACK=1, at
// Qwen-Image-2.1's block shapes, 4096 rows (VPIPE_I8_PT_ROWS), the arms
// interleaved with the order reversed every round, median kept.
TEST(gemm_i8, w4_unpack_cost) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lg = mc->load_library("dense_gemm_mma_bf16");
  ComputeLibrary lq = mc->load_library("affine_dequant_bf16");
  ComputeFunction prod = lg.function("gemm_i8pt_m128x64");
  ComputeFunction cvt = lq.function("affine_to_i8pt_w4");
  ComputeFunction cvtr = lq.function("affine_to_i8pt_w4_r");
  ComputeFunction cvt512 = lq.function("affine_to_i8c_w4_gc512_r");
  struct Wx { const char* name; int wf; ComputeFunction f; };
  std::vector<Wx> wx;
  for (const auto& [name, wf, fn] :
       {std::tuple<const char*, int, const char*>{"i8 reg", 0,
                                                  "gemm_i8pt_wx_i8_reg"},
        {"i8 tg", 0, "gemm_i8pt_wx_i8_tg"},
        {"w4 reg", 1, "gemm_i8pt_wx_w4_reg"},
        {"w4 tg", 1, "gemm_i8pt_wx_w4_tg"},
        {"aff reg", 2, "gemm_i8pt_wx_aff_reg"},
        {"aff tg", 2, "gemm_i8pt_wx_aff_tg"}}) {
    wx.push_back(Wx{name, wf, lg.function(fn)});
  }
  ASSERT_TRUE(prod.valid() && cvt.valid() && cvtr.valid() &&
              cvt512.valid());
  for (const Wx& w : wx) { ASSERT_TRUE(w.f.valid()); }
  if (!prod.valid() || !cvt.valid() || !cvtr.valid()) { return; }
  auto up = [&](const void* p, std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n);
    if (!b.empty() && p != nullptr) { std::memcpy(b.contents(), p, n); }
    return b;
  };
  auto gemm_prod = [&](ComputeEncoder& e, const SharedBuffer& xq,
                       const SharedBuffer& wq, const SharedBuffer& as,
                       const SharedBuffer& ws, const SharedBuffer& y, int M,
                       int N, int K) {
    e.set_function(prod);
    e.set_buffer(0, xq); e.set_buffer(1, wq); e.set_buffer(2, as);
    e.set_buffer(3, ws); e.set_buffer(4, y);
    e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, M);
    e.set_buffer(8, ws); e.set_constant(9, 0);
    e.dispatch({(unsigned)(N / 64) * 32u, (unsigned)(M / 128) * 2u, 4u},
               {32u, 2u, 4u});
  };
  auto gemm_wx = [&](ComputeEncoder& e, const Wx& a, const SharedBuffer& xq,
                     const SharedBuffer& w, const SharedBuffer& as,
                     const SharedBuffer& ws, const SharedBuffer& y,
                     const SharedBuffer& sc, const SharedBuffer& bi,
                     const SharedBuffer& inv, int M, int N, int K) {
    e.set_function(a.f);
    e.set_buffer(0, xq); e.set_buffer(1, w); e.set_buffer(2, as);
    e.set_buffer(3, ws); e.set_buffer(4, y);
    e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, M);
    e.set_buffer(8, sc); e.set_buffer(9, bi); e.set_buffer(10, inv);
    e.dispatch({(unsigned)(N / 64) * 32u, (unsigned)(M / 128) * 2u, 4u},
               {32u, 2u, 4u});
  };
  // The conversions' grid: a row a threadgroup of K/16 threads (plain), a
  // simdgroup per 256-block (rotated).
  auto convert = [&](ComputeEncoder& e, bool rot, const SharedBuffer& c,
                     const SharedBuffer& sc, const SharedBuffer& bi,
                     const SharedBuffer& q, const SharedBuffer& qs, int N,
                     int K) {
    e.set_function(rot ? cvtr : cvt);
    e.set_buffer(0, c); e.set_buffer(1, sc); e.set_buffer(2, bi);
    e.set_buffer(3, q); e.set_buffer(4, qs); e.set_constant(5, K);
    const unsigned T =
        rot ? 32u * (unsigned)std::min(K / 256, 32)
            : (unsigned)std::max(32, (K / 16 + 31) / 32 * 32);
    e.dispatch({T, (unsigned)N, 1}, {T, 1, 1});
  };
  auto run = [&](const std::function<void(ComputeEncoder&)>& f) {
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder e = st.begin_compute(); f(e); }
    st.commit().wait();
  };
  // Everything one shape needs: activations, the three weight forms of
  // one random w4 pack and of one random int4-per-row weight.
  struct Op {
    SharedBuffer xq, as, w8, w4, ws, pc, psc, pbi, pinv, h8, hs, y;
  };
  auto make = [&](int M, int N, int K, unsigned seed) {
    Op o;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> c8(-127, 127), c4(-8, 7), cu(0, 15);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::int8_t> xq((std::size_t)M * K);
    for (auto& v : xq) { v = (std::int8_t)c8(rng); }
    std::vector<float> as(M), ws(N);
    for (auto& v : as) { v = 1e-2f * (1.0f + 0.1f * nd(rng)); }
    for (auto& v : ws) { v = 1e-3f * (1.0f + 0.1f * nd(rng)); }
    // int4 per row: the widened int8 and the packed nibbles, byte i of a
    // word = e_i | e_{i+4} << 4 in offset binary.
    std::vector<std::int8_t> w8((std::size_t)N * K);
    std::vector<std::uint8_t> w4((std::size_t)N * K / 2, 0);
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        const int q = c4(rng);
        w8[(std::size_t)n * K + k] = (std::int8_t)q;
        const int word = k / 8, j = k % 8;
        const std::size_t byte = (std::size_t)n * K / 2 + word * 4 + (j % 4);
        w4[byte] |= (std::uint8_t)((q + 8) << (j < 4 ? 0 : 4));
      }
    }
    // A w4 g64 affine pack (MLX nibble order) and the row's 127 / amax
    // over its element-type dequant -- what the held conversion uses.
    const int G = K / 64;
    std::vector<std::uint8_t> pc((std::size_t)N * K / 2, 0);
    std::vector<std::uint16_t> psc((std::size_t)N * G), pbi(psc.size());
    for (std::size_t i = 0; i < psc.size(); ++i) {
      const float s = 2e-3f * (1.0f + 0.5f * std::fabs(nd(rng)));
      psc[i] = f32_to_bf16(s);
      pbi[i] = f32_to_bf16(-7.5f * s + 2e-3f * nd(rng));
    }
    std::vector<float> pinv(N);
    for (int n = 0; n < N; ++n) {
      float am = 0.0f;
      for (int k = 0; k < K; ++k) {
        const int c = cu(rng);
        pc[((std::size_t)n * K + k) / 2] |=
            (std::uint8_t)(c << ((k & 1) ? 4 : 0));
        const std::size_t g = (std::size_t)n * G + k / 64;
        const float v = bf16_to_f32(f32_to_bf16(
            (float)c * bf16_to_f32(psc[g]) + bf16_to_f32(pbi[g])));
        am = std::max(am, std::fabs(v));
      }
      pinv[n] = am > 0.0f ? 127.0f / am : 0.0f;
    }
    o.xq = up(xq.data(), xq.size());
    o.as = up(as.data(), as.size() * 4);
    o.ws = up(ws.data(), ws.size() * 4);
    o.w8 = up(w8.data(), w8.size());
    o.w4 = up(w4.data(), w4.size());
    o.pc = up(pc.data(), pc.size());
    o.psc = up(psc.data(), psc.size() * 2);
    o.pbi = up(pbi.data(), pbi.size() * 2);
    o.pinv = up(pinv.data(), pinv.size() * 4);
    o.h8 = up(nullptr, (std::size_t)N * K);
    o.hs = up(nullptr, (std::size_t)N * 4);
    o.y = up(nullptr, (std::size_t)M * N * 2);
    // The held token_plain layout of the pack, as the tier converts it.
    run([&](ComputeEncoder& e) {
      convert(e, false, o.pc, o.psc, o.pbi, o.h8, o.hs, N, K);
    });
    return o;
  };

  // ---- correctness ------------------------------------------------------
  {
    const int M = 256, N = 192, K = 1024;
    Op o = make(M, N, K, 3);
    const std::size_t yb = (std::size_t)M * N * 2;
    std::vector<std::uint8_t> ref_i4(yb), ref_aff(yb);
    run([&](ComputeEncoder& e) {
      gemm_prod(e, o.xq, o.w8, o.as, o.ws, o.y, M, N, K);
    });
    std::memcpy(ref_i4.data(), o.y.contents(), yb);
    run([&](ComputeEncoder& e) {
      gemm_prod(e, o.xq, o.h8, o.as, o.hs, o.y, M, N, K);
    });
    std::memcpy(ref_aff.data(), o.y.contents(), yb);
    for (const Wx& a : wx) {
      std::memset(o.y.contents(), 0xff, yb);
      const SharedBuffer& w = a.wf == 0 ? o.w8 : a.wf == 1 ? o.w4 : o.pc;
      const SharedBuffer& ws = a.wf == 2 ? o.hs : o.ws;
      run([&](ComputeEncoder& e) {
        gemm_wx(e, a, o.xq, w, o.as, ws, o.y, o.psc, o.pbi, o.pinv, M, N,
                K);
      });
      const auto& ref = a.wf == 2 ? ref_aff : ref_i4;
      const bool same = std::memcmp(o.y.contents(), ref.data(), yb) == 0;
      std::size_t ndiff = 0;
      double worst = 0.0;
      {
        const auto* g = static_cast<const std::uint16_t*>(o.y.contents());
        const auto* r = reinterpret_cast<const std::uint16_t*>(ref.data());
        for (std::size_t i = 0; i < yb / 2; ++i) {
          if (g[i] == r[i]) { continue; }
          ++ndiff;
          const double rv = bf16_to_f32(r[i]);
          worst = std::max(worst, std::fabs(bf16_to_f32(g[i]) - rv) /
                                      std::max(std::fabs(rv), 1e-6));
        }
      }
      std::printf("[gemm_i8 w4 unpack] %-8s %s the tier's kernel on %s"
                  " (%zu of %zu outputs differ, worst rel %.2e)\n",
                  a.name, same ? "IDENTICAL to" : "DIFFERENT from",
                  a.wf == 2 ? "the held conversion of the pack"
                            : "the widened weight", ndiff, yb / 2, worst);
      EXPECT_TRUE(same);
    }
  }

  // ---- timing -----------------------------------------------------------
  if (std::getenv("VPIPE_I8_W4_UNPACK") == nullptr) {
    std::printf("[gemm_i8 w4 unpack] VPIPE_I8_W4_UNPACK unset -- timing "
                "skipped\n");
    return;
  }
  const int rows = [] {
    const char* e = std::getenv("VPIPE_I8_PT_ROWS");
    return e != nullptr ? std::atoi(e) : 4096;
  }();
  const int rounds = [] {
    const char* e = std::getenv("VPIPE_I8_PT_ROUNDS");
    return e != nullptr && std::atoi(e) > 0 ? std::atoi(e) : 5;
  }();
  struct Sh { int N, K; const char* tag; };
  for (const Sh& sh : {Sh{4096, 4096, "q/k/v/out"},
                       Sh{12288, 4096, "ff gate/up"},
                       Sh{4096, 12288, "ff down"}}) {
    const int M = rows, N = sh.N, K = sh.K;
    // Two weights cycled so no call finds its weight warm in cache.
    Op o[2] = {make(M, N, K, 7), make(M, N, K, 8)};
    SharedBuffer scratch = up(nullptr, (std::size_t)N * K);
    SharedBuffer sscale = up(nullptr, std::max((std::size_t)N * 4,
                                               (std::size_t)N * (K / 512) * 2));
    struct Arm {
      std::string name;
      std::function<void(ComputeEncoder&, Op&)> f;
      std::vector<double> ms;
    };
    std::vector<Arm> arms;
    arms.push_back({"tier (int8 held)", [&](ComputeEncoder& e, Op& p) {
      gemm_prod(e, p.xq, p.w8, p.as, p.ws, p.y, M, N, K);
    }, {}});
    for (const Wx& a : wx) {
      const Wx* ap = &a;
      arms.push_back({a.name, [&, ap](ComputeEncoder& e, Op& p) {
        const SharedBuffer& w =
            ap->wf == 0 ? p.w8 : ap->wf == 1 ? p.w4 : p.pc;
        gemm_wx(e, *ap, p.xq, w, p.as, ap->wf == 2 ? p.hs : p.ws, p.y,
                p.psc, p.pbi, p.pinv, M, N, K);
      }, {}});
    }
    for (int rot = 0; rot < 2; ++rot) {
      arms.push_back({rot ? "cvt_r + gemm" : "cvt + gemm",
                      [&, rot](ComputeEncoder& e, Op& p) {
        convert(e, rot != 0, p.pc, p.psc, p.pbi, scratch, sscale, N, K);
        gemm_prod(e, p.xq, scratch, p.as, sscale, p.y, M, N, K);
      }, {}});
    }
    // The conversions ALONE, against the GEMM they feed: the token
    // tier's plain and rotated, and the fast tier's (group 512, rotated).
    for (int rot = 0; rot < 2; ++rot) {
      arms.push_back({rot ? "cvt_r only" : "cvt only",
                      [&, rot](ComputeEncoder& e, Op& p) {
        convert(e, rot != 0, p.pc, p.psc, p.pbi, scratch, sscale, N, K);
      }, {}});
    }
    arms.push_back({"cvt_g512 only", [&](ComputeEncoder& e, Op& p) {
      e.set_function(cvt512);
      e.set_buffer(0, p.pc); e.set_buffer(1, p.psc); e.set_buffer(2, p.pbi);
      e.set_buffer(3, scratch); e.set_buffer(4, sscale);
      e.set_constant(5, K);
      e.dispatch({(unsigned)((K / 512 + 3) / 4) * 128u, (unsigned)N, 1},
                 {128u, 1u, 1u});
    }, {}});
    const int iters = 6;
    for (int r = 0; r < rounds; ++r) {
      for (std::size_t k = 0; k < arms.size(); ++k) {
        Arm& a = arms[(r % 2 == 0) ? k : arms.size() - 1 - k];
        run([&](ComputeEncoder& e) { a.f(e, o[0]); });
        const auto t0 = std::chrono::steady_clock::now();
        run([&](ComputeEncoder& e) {
          for (int i = 0; i < iters; ++i) { a.f(e, o[i & 1]); }
        });
        a.ms.push_back(secs_(t0, std::chrono::steady_clock::now()) * 1e3 /
                       iters);
      }
    }
    std::printf("[gemm_i8 w4 unpack] %s M=%d N=%d K=%d\n", sh.tag, M, N, K);
    double base = 0.0;
    const double gop = 2.0 * M * (double)N * K * 1e-9;
    for (Arm& a : arms) {
      std::sort(a.ms.begin(), a.ms.end());
      const double ms = a.ms[a.ms.size() / 2];
      if (base == 0.0) { base = ms; }
      std::printf("[gemm_i8 w4 unpack]   %-17s %7.3f ms %6.1f TOP/s %.3fx\n",
                  a.name.c_str(), ms, gop / ms, base / ms);
    }
  }
}

// WHAT A W4 PACK'S CONVERSION INTO THE INT8 LAYOUTS COSTS, kernel by
// kernel (VPIPE_I8_CVT_RATE=1): each converter 32 times in one command
// buffer over a [N, K] weight, against the activation quantizers that do
// the same rotation over a bf16 [N, K] -- the bound a converter that
// rotates should come near. Median of interleaved rounds; GB/s counts
// the bytes each must move (codes in, int8 out; bf16 in for the
// quantizers).
TEST(gemm_i8, fast_conversion_rate) {
  if (std::getenv("VPIPE_I8_CVT_RATE") == nullptr) { return; }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  ComputeLibrary lq = mc->load_library("affine_dequant_bf16");
  // sdiv: a rotated token kernel's simdgroups a row, ceil(G / sdiv)
  // (each takes sdiv 256-blocks; the kernel holds up to two). The
  // context dispatches two.
  struct K { const char* name; const char* fn; int kind; int sdiv = 2; };
  // kind 0: token plain (K/16 threads a row), 1: token rotated (32 S a
  // row), 2: g512 (128 a threadgroup, 4 groups); input w4 pack (codes,
  // scales, biases) or, `+ 3`, a bf16 activation; 6 / 7: g256 / g1024
  // from a pack, 10: g256 from bf16.
  const std::vector<K> ks = {
      {"w4 -> token plain", "affine_to_i8pt_w4", 0},
      {"w4 -> token rotated", "affine_to_i8pt_w4_r", 1},
      {"w4 -> token rotated, a block each", "affine_to_i8pt_w4_r", 1, 1},
      {"w4 -> g512 plain", "affine_to_i8c_w4_gc512", 2},
      {"w4 -> g256 rotated", "affine_to_i8c_w4_gc256_r", 6},
      {"w4 -> g512 rotated", "affine_to_i8c_w4_gc512_r", 2},
      {"w4 -> g1024 rotated", "affine_to_i8c_w4_gc1024_r", 7},
      {"bf16 -> token plain", "quant_i8_tok", 3},
      {"bf16 -> token rotated", "quant_i8_tok_r", 4},
      {"bf16 -> token rotated, a block each", "quant_i8_tok_r", 4, 1},
      {"bf16 -> g512 plain", "quant_i8_row_gc512", 5},
      {"bf16 -> g256 rotated", "quant_i8_row_gc256_r", 10},
      {"bf16 -> g512 rotated", "quant_i8_row_gc512_r", 5}};
  for (int N : {12288, 4096}) {
    const int Kd = N == 12288 ? 4096 : 12288;
    const std::size_t nk = (std::size_t)N * Kd;
    SharedBuffer codes = mc->make_shared_buffer(nk / 2);
    SharedBuffer sc = mc->make_shared_buffer(nk / 64 * 2);
    SharedBuffer bi = mc->make_shared_buffer(nk / 64 * 2);
    SharedBuffer x = mc->make_shared_buffer(nk * 2);
    SharedBuffer q = mc->make_shared_buffer(nk);
    // Scales of every shape benched: up to f32 per 256-block.
    SharedBuffer qs = mc->make_shared_buffer(nk / 64);
    std::memset(codes.contents(), 0x5a, nk / 2);
    for (std::size_t i = 0; i < nk / 64; ++i) {
      static_cast<std::uint16_t*>(sc.contents())[i] = f32_to_bf16(1e-3f);
      static_cast<std::uint16_t*>(bi.contents())[i] = f32_to_bf16(-7e-3f);
    }
    for (std::size_t i = 0; i < nk; ++i) {
      static_cast<std::uint16_t*>(x.contents())[i] =
          f32_to_bf16((float)((i * 2654435761u) % 1000) * 1e-3f - 0.5f);
    }
    std::vector<std::vector<double>> ms(ks.size());
    auto one = [&](ComputeEncoder& e, const K& k, const ComputeFunction& f) {
      e.set_function(f);
      const bool pack = (k.kind < 3 || k.kind >= 6) && k.kind != 10;
      if (pack) {
        e.set_buffer(0, codes); e.set_buffer(1, sc); e.set_buffer(2, bi);
        e.set_buffer(3, q); e.set_buffer(4, qs); e.set_constant(5, Kd);
      } else {
        e.set_buffer(0, x); e.set_buffer(1, q); e.set_buffer(2, qs);
        e.set_constant(3, Kd);
      }
      const int c = k.kind >= 6 ? 2 : k.kind % 3;
      const int gc = k.kind == 6 || k.kind == 10 ? 256
                     : k.kind == 7                ? 1024
                                                  : 512;
      if (c == 2) {
        e.dispatch({(unsigned)((Kd / gc + 3) / 4) * 128u, (unsigned)N, 1},
                   {128u, 1u, 1u});
      } else {
        const int G = Kd / 256;
        const unsigned T =
            c == 1 ? 32u * (unsigned)std::min((G + k.sdiv - 1) / k.sdiv, 32)
                   : (unsigned)std::max(32, (Kd / 16 + 31) / 32 * 32);
        e.dispatch({T, (unsigned)N, 1}, {T, 1, 1});
      }
    };
    std::vector<ComputeFunction> fs;
    for (const K& k : ks) { fs.push_back(lq.function(k.fn)); }
    for (std::size_t i = 0; i < ks.size(); ++i) {
      ASSERT_TRUE(fs[i].valid());
      if (!fs[i].valid()) { return; }
    }
    const int reps = 32;
    // One stream: a fresh one is a new queue whose first commit costs
    // ~1-2 ms (a_fresh_stream_per_commit_costs), 0.05 ms a rep here.
    CommandStream st = mc->make_command_stream();
    for (int r = 0; r < 5; ++r) {
      for (std::size_t j = 0; j < ks.size(); ++j) {
        const std::size_t i = (r % 2 == 0) ? j : ks.size() - 1 - j;
        {
          { ComputeEncoder e = st.begin_compute(); one(e, ks[i], fs[i]); }
          st.commit().wait();
        }
        const auto t0 = std::chrono::steady_clock::now();
        {
          ComputeEncoder e = st.begin_compute();
          for (int k = 0; k < reps; ++k) { one(e, ks[i], fs[i]); }
        }
        st.commit().wait();
        ms[i].push_back(secs_(t0, std::chrono::steady_clock::now()) * 1e3 /
                        reps);
      }
    }
    std::printf("[gemm_i8 cvt] [%d, %d]\n", N, Kd);
    for (std::size_t i = 0; i < ks.size(); ++i) {
      std::sort(ms[i].begin(), ms[i].end());
      const double m = ms[i][ms[i].size() / 2];
      const bool pack = (ks[i].kind < 3 || ks[i].kind >= 6) &&
                        ks[i].kind != 10;
      const double bytes = pack ? nk * 1.5 : nk * 3.0;
      std::printf("[gemm_i8 cvt]   %-34s %7.3f ms %6.0f GB/s\n", ks[i].name,
                  m, bytes / (m * 1e6));
    }
  }
}

// THE ROTATED TOKEN KERNELS ARE THE SAME BYTES AT EITHER DISPATCH: a
// simdgroup a 256-block (up to 32 of them) or two (what the context
// dispatches) --
// activation, SwiGLU and a w4 / w8 pack's conversion -- BYTE-IDENTICAL
// codes and row scales at K values with an even and an odd block count,
// and the dequant back. Outlier channels in the activations, so the
// scales are not all alike.
TEST(gemm_i8, rotated_token_kernels_agree_at_either_dispatch) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lq = mc->load_library("affine_dequant_bf16");
  std::mt19937 rng(23);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::uniform_int_distribution<int> cu(0, 255);
  int checked = 0;
  for (int K : {1024, 3072, 3840, 4096, 12288}) {
    const int M = 40, G = K / 256;
    const std::size_t mk = (std::size_t)M * K;
    std::vector<std::uint16_t> x(mk), u(mk);
    for (std::size_t i = 0; i < mk; ++i) {
      float v = nd(rng);
      if ((i % (std::size_t)K) % 97 == 5) { v *= 30.0f; }
      x[i] = f32_to_bf16(v);
      u[i] = f32_to_bf16(nd(rng));
    }
    std::vector<std::uint8_t> codes(mk);
    for (auto& c : codes) { c = (std::uint8_t)cu(rng); }
    std::vector<std::uint16_t> sc(mk / 64), bi(mk / 64);
    for (std::size_t i = 0; i < sc.size(); ++i) {
      const float s = 2e-3f * (1.0f + std::fabs(nd(rng)));
      sc[i] = f32_to_bf16(s);
      bi[i] = f32_to_bf16(-7.5f * s + 1e-3f * nd(rng));
    }
    auto up = [&](const void* p, std::size_t n) {
      SharedBuffer b = mc->make_shared_buffer(n);
      if (p != nullptr) { std::memcpy(b.contents(), p, n); }
      return b;
    };
    SharedBuffer bx = up(x.data(), mk * 2), bu = up(u.data(), mk * 2);
    SharedBuffer bs = up(sc.data(), sc.size() * 2);
    SharedBuffer bb = up(bi.data(), bi.size() * 2);
    const std::size_t sb = (std::size_t)M * 4;
    struct Case { const char* what; const char* fn; int mode; };
    for (const Case& c : {Case{"activation", "quant_i8_tok_r", 0},
                          Case{"swiglu", "swiglu_quant_i8_tok_r", 1},
                          Case{"w4 pack", "affine_to_i8pt_w4_r", 2},
                          Case{"w8 pack", "affine_to_i8pt_w8_r", 3}}) {
      ComputeFunction f = lq.function(c.fn);
      ComputeFunction fd = lq.function("i8pt_dequant_r");
      ASSERT_TRUE(f.valid() && fd.valid());
      if (!f.valid() || !fd.valid()) { return; }
      SharedBuffer q[2] = {up(nullptr, mk), up(nullptr, mk)};
      SharedBuffer s2[2] = {up(nullptr, sb), up(nullptr, sb)};
      SharedBuffer w[2] = {up(nullptr, mk * 2), up(nullptr, mk * 2)};
      for (int a = 0; a < 2; ++a) {
        // A w8 pack converts in place.
        SharedBuffer bc = up(codes.data(), c.mode == 2 ? mk / 2 : mk);
        std::memset(q[a].contents(), 0x55, mk);
        std::memset(s2[a].contents(), 0x55, sb);
        // The old dispatch (a block each up to 32 simdgroups) and the
        // context's (two each).
        const unsigned T =
            32u * (unsigned)(a == 0 ? std::min(G, 32) : (G + 1) / 2);
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          e.set_function(f);
          int slot = 0;
          if (c.mode == 0) {
            e.set_buffer(0, bx); slot = 1;
          } else if (c.mode == 1) {
            e.set_buffer(0, bx); e.set_buffer(1, bu); slot = 2;
          } else {
            e.set_buffer(0, bc); e.set_buffer(1, bs); e.set_buffer(2, bb);
            slot = 3;
          }
          e.set_buffer(slot, c.mode == 3 ? bc : q[a]);
          e.set_buffer(slot + 1, s2[a]);
          e.set_constant(slot + 2, K);
          e.dispatch({T, (unsigned)M, 1}, {T, 1, 1});
        }
        st.commit().wait();
        if (c.mode == 3) { std::memcpy(q[a].contents(), bc.contents(), mk); }
        {
          ComputeEncoder e = st.begin_compute();
          e.set_function(fd);
          e.set_buffer(0, q[a]); e.set_buffer(1, s2[a]);
          e.set_buffer(2, w[a]); e.set_constant(3, K);
          e.dispatch({T, (unsigned)M, 1}, {T, 1, 1});
        }
        st.commit().wait();
      }
      const bool same =
          std::memcmp(q[0].contents(), q[1].contents(), mk) == 0 &&
          std::memcmp(s2[0].contents(), s2[1].contents(), sb) == 0 &&
          std::memcmp(w[0].contents(), w[1].contents(), mk * 2) == 0;
      if (!same) {
        std::printf("[gemm_i8 tok_r] K=%d %s: DIFFERENT\n", K, c.what);
      }
      EXPECT_TRUE(same);
      ++checked;
    }
  }
  std::printf("[gemm_i8 tok_r] %d (K, kernel) cases checked\n", checked);
}

// WHAT ENCODING A PER-CALL CONVERSION COSTS THE CPU (VPIPE_I8_CVT_RATE=1):
// a held stack that drains every few blocks (a stop check installed)
// leaves the GPU idle while the next blocks are encoded, so the encode
// is on the clock there. encode_to_fast_scratch against a plain
// dispatch of the same kernel, timed over many encodes without a commit.
TEST(gemm_i8, conversion_encode_cost) {
  if (std::getenv("VPIPE_I8_CVT_RATE") == nullptr) { return; }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  vpipe::genai::I8GemmContext ctx(mc, /*want=*/true, /*bf16=*/true);
  ctx.set_tier(vpipe::genai::I8GemmContext::Tier::kToken);
  ctx.set_token_rotation(true);
  const int N = 3072, K = 3072, gc = vpipe::genai::I8GemmContext::kTokenGroup;
  SharedBuffer c = mc->make_shared_buffer((std::size_t)N * K / 2);
  SharedBuffer s2 = mc->make_shared_buffer((std::size_t)N * K / 64 * 2);
  SharedBuffer b = mc->make_shared_buffer((std::size_t)N * K / 64 * 2);
  ComputeLibrary lq = mc->load_library("affine_dequant_bf16");
  ComputeFunction fr = lq.function("affine_to_i8pt_w4_r");
  ASSERT_TRUE(fr.valid());
  if (!fr.valid()) { return; }
  const int reps = 2000;
  for (int arm = 0; arm < 2; ++arm) {
    CommandStream st = mc->make_command_stream();
    double us = 0.0;
    {
      ComputeEncoder e = st.begin_compute();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i) {
        if (arm == 0) {
          ctx.encode_to_fast_scratch(e, c, s2, b, gc, N, K);
        } else {
          e.set_function(fr);
          e.set_buffer(0, c); e.set_buffer(1, s2); e.set_buffer(2, b);
          e.set_buffer(3, c); e.set_buffer(4, s2); e.set_constant(5, K);
          e.dispatch({256u, (unsigned)N, 1}, {256u, 1, 1});
        }
      }
      us = secs_(t0, std::chrono::steady_clock::now()) * 1e6 / reps;
    }
    std::printf("[gemm_i8 enc] %s: %.2f us of CPU an encode\n",
                arm == 0 ? "encode_to_fast_scratch" : "bare dispatch", us);
    // Not committed: the buffers are only bound, nothing need run.
  }
}

// CAN A W4 CONVERSION HIDE UNDER THE GEMM BEFORE IT? (VPIPE_I8_CVT_RATE=1)
// The rotated converters are shuffle-bound and the int8 GEMM matrix-unit
// bound, so the two might share the GPU. A block's six K = 3072
// projections at 4096 rows (q, k, v, out 3072 wide, gate and up 12288),
// one activation quantized for all: held layouts; converted per call
// into one scratch, in order; and converted one GEMM AHEAD into the
// other of two scratches, each conversion in a concurrent scope with the
// GEMM before it.
TEST(gemm_i8, a_conversion_overlaps_the_gemm) {
  if (std::getenv("VPIPE_I8_CVT_RATE") == nullptr) { return; }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  using Ctx = vpipe::genai::I8GemmContext;
  const int M = 4096, K = 3072;
  const std::vector<int> Ns = {3072, 3072, 3072, 3072, 12288, 12288};
  const int n = (int)Ns.size();
  SharedBuffer x = mc->make_shared_buffer((std::size_t)M * K * 2);
  SharedBuffer y = mc->make_shared_buffer((std::size_t)M * 12288 * 2);
  for (std::size_t i = 0; i < (std::size_t)M * K; ++i) {
    static_cast<std::uint16_t*>(x.contents())[i] =
        f32_to_bf16((float)((i * 2654435761u) % 1000) * 1e-3f - 0.5f);
  }
  std::vector<SharedBuffer> c(n), sc(n), bi(n);
  for (int i = 0; i < n; ++i) {
    const std::size_t nk = (std::size_t)Ns[i] * K;
    c[i] = mc->make_shared_buffer(nk / 2);
    sc[i] = mc->make_shared_buffer(nk / 64 * 2);
    bi[i] = mc->make_shared_buffer(nk / 64 * 2);
    auto* cp = static_cast<std::uint8_t*>(c[i].contents());
    for (std::size_t j = 0; j < nk / 2; ++j) {
      cp[j] = (std::uint8_t)((j * 2654435761u + i) >> 7);
    }
    for (std::size_t j = 0; j < nk / 64; ++j) {
      static_cast<std::uint16_t*>(sc[i].contents())[j] = f32_to_bf16(1e-2f);
      static_cast<std::uint16_t*>(bi[i].contents())[j] =
          f32_to_bf16(-7.5e-2f);
    }
  }
  for (int tier = 0; tier < 2; ++tier) {
    Ctx ctx(mc, /*want=*/true, /*bf16=*/true);
    ctx.set_tier(tier == 0 ? Ctx::Tier::kToken : Ctx::Tier::kFast);
    ctx.set_token_rotation(true);
    const int gc = ctx.fast_group(K);
    if (gc <= 0 || !ctx.fast_converts(4, gc)) { continue; }
    std::vector<SharedBuffer> hq(n), hs(n);
    SharedBuffer sq[2], ss[2];
    for (int k = 0; k < 2; ++k) {
      sq[k] = mc->make_shared_buffer((std::size_t)12288 * K);
      ss[k] = mc->make_shared_buffer(Ctx::fast_scale_bytes(12288, K, gc));
    }
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      for (int i = 0; i < n; ++i) {
        hq[i] = mc->make_shared_buffer((std::size_t)Ns[i] * K);
        hs[i] = mc->make_shared_buffer(Ctx::fast_scale_bytes(Ns[i], K, gc));
        ASSERT_TRUE(ctx.encode_to_fast(e, c[i], sc[i], bi[i], hq[i], hs[i],
                                       gc, Ns[i], K, 4));
      }
    }
    st.commit().wait();
    const SharedBuffer none;
    auto gemm = [&](ComputeEncoder& e, const SharedBuffer& q,
                    const SharedBuffer& s, int i) {
      return ctx.gemm_fast_quantized(e, q, s, gc, none, y, 0, M, Ns[i], K);
    };
    auto cvt = [&](ComputeEncoder& e, int i, int slot) {
      return ctx.encode_to_fast(e, c[i], sc[i], bi[i], sq[slot], ss[slot],
                                gc, Ns[i], K, 4);
    };
    const char* names[3] = {"held", "per call", "one ahead, concurrent"};
    auto run = [&](ComputeEncoder& e, int arm) {
      bool ok = ctx.quantize_fast(e, x, 0, M, K, gc);
      if (arm == 0) {
        for (int i = 0; i < n; ++i) { ok = gemm(e, hq[i], hs[i], i) && ok; }
      } else if (arm == 1) {
        for (int i = 0; i < n; ++i) {
          ok = cvt(e, i, 0) && ok;
          ok = gemm(e, sq[0], ss[0], i) && ok;
        }
      } else {
        ok = cvt(e, 0, 0) && ok;
        for (int i = 0; i < n; ++i) {
          auto scope = e.concurrent_scope();
          ok = gemm(e, sq[i % 2], ss[i % 2], i) && ok;
          if (i + 1 < n) { ok = cvt(e, i + 1, (i + 1) % 2) && ok; }
        }
      }
      return ok;
    };
    const int reps = 8;
    std::vector<double> ms[3];
    for (int r = 0; r < 7; ++r) {
      for (int j = 0; j < 3; ++j) {
        const int arm = r % 2 == 0 ? j : 2 - j;
        {
          ComputeEncoder e = st.begin_compute();
          ASSERT_TRUE(run(e, arm));
        }
        st.commit().wait();
        const auto t0 = std::chrono::steady_clock::now();
        {
          ComputeEncoder e = st.begin_compute();
          for (int k = 0; k < reps; ++k) { run(e, arm); }
        }
        st.commit().wait();
        ms[arm].push_back(secs_(t0, std::chrono::steady_clock::now()) *
                          1e3 / reps);
      }
    }
    double base = 0.0;
    for (int arm = 0; arm < 3; ++arm) {
      std::sort(ms[arm].begin(), ms[arm].end());
      const double m = ms[arm][ms[arm].size() / 2];
      if (arm == 0) { base = m; }
      std::printf("[gemm_i8 overlap] %s, %-22s %7.3f ms  %.3fx held\n",
                  tier == 0 ? "token" : "fast ", names[arm], m, base / m);
    }
  }
}

// WHAT A FRESH COMMAND STREAM COSTS (VPIPE_I8_CVT_RATE=1): every
// make_command_stream() is a new MTLCommandQueue, and the DiTs' stop
// drains and streamed blocks make one per chunk / per block. A tiny
// dispatch committed and waited on fresh streams against one stream
// reused for every commit.
TEST(gemm_i8, a_fresh_stream_per_commit_costs) {
  if (std::getenv("VPIPE_I8_CVT_RATE") == nullptr) { return; }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lq = mc->load_library("affine_dequant_bf16");
  ComputeFunction f = lq.function("quant_i8_tok");
  ASSERT_TRUE(f.valid());
  if (!f.valid()) { return; }
  SharedBuffer x = mc->make_shared_buffer(4096 * 2);
  SharedBuffer q = mc->make_shared_buffer(4096);
  SharedBuffer s = mc->make_shared_buffer(64);
  auto tiny = [&](ComputeEncoder& e) {
    e.set_function(f);
    e.set_buffer(0, x); e.set_buffer(1, q); e.set_buffer(2, s);
    e.set_constant(3, 4096);
    e.dispatch({256u, 1u, 1u}, {256u, 1u, 1u});
  };
  const int reps = 200;
  for (int round = 0; round < 2; ++round) {
    {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute(); tiny(e); }
        st.commit().wait();
      }
      std::printf("[gemm_i8 stream] fresh stream a commit: %.1f us\n",
                  secs_(t0, std::chrono::steady_clock::now()) * 1e6 / reps);
    }
    {
      CommandStream st = mc->make_command_stream();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i) {
        { ComputeEncoder e = st.begin_compute(); tiny(e); }
        st.commit().wait();
      }
      std::printf("[gemm_i8 stream] one stream reused: %.1f us\n",
                  secs_(t0, std::chrono::steady_clock::now()) * 1e6 / reps);
    }
    {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < reps; ++i) {
        CommandStream st = mc->make_command_stream();
        (void)st;
      }
      std::printf("[gemm_i8 stream] make_command_stream alone: %.1f us\n",
                  secs_(t0, std::chrono::steady_clock::now()) * 1e6 / reps);
    }
  }
}
