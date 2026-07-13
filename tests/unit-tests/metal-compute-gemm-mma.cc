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

#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <Metal/Metal.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
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
QWeight make_qweight(int N, int K, std::uint32_t seed) {
  const int G = K / 64;
  QWeight q;
  q.packed.assign((std::size_t)N * (K / 2), 0);
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
      const int g = k / 64;
      const float s = bf16_to_f32(q.scales[(std::size_t)n * G + g]);
      const float b = bf16_to_f32(q.biases[(std::size_t)n * G + g]);
      q.deq[(std::size_t)n * K + k] = s * (float)v + b;
    }
  }
  return q;
}

}  // namespace

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
// reads supplied as 0 (pad-1 zero padding), the K=9*Cin tail (K need not be a
// multiple of BK), and Cout/pixel tail masking. Both stride 1 (resnet/upsample)
// and stride 2 (encoder downsample) at VAE-shaped and deliberately
// non-tile-multiple sizes. f16 = exactly what the VAE binds.
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
              const int iy = oy * stride + ky - 1;
              const int ix = ox * stride + kx - 1;
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
  if (!f_i8.valid() || !f_qr.valid() || !f_n128.valid() ||
      !f_deep.valid()) {
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

    launch_i8(1); launch_f16(1);
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
    const double s_i8 = secs_(t0, t1), s_f16 = secs_(t1, t2);
    std::printf("[gemm_i8] M=%4d N=%5d K=%5d: i8+quant %.0f GF/s (%.2f ms)"
                "  f16 %.0f GF/s (%.2f ms)  speedup %.2fx\n",
                M, N, K, gflop / s_i8, s_i8 * 1e3 / iters,
                gflop / s_f16, s_f16 * 1e3 / iters, s_f16 / s_i8);
  };
  run_shape(256, 512, 1024, /*check=*/true, 10);      // oracle + quality
  run_shape(4096, 12288, 4096, /*check=*/false, 5);   // flux2 block proj
  run_shape(4096, 4096, 12288, /*check=*/false, 5);   // flux2 ff-down
  run_shape(4096, 36864, 4096, /*check=*/false, 3);   // flux2 qkv_mlp
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
TEST(gemm_i8, k512_chunked) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) { return; }
  ComputeLibrary lib_mma = mc->load_library("dense_gemm_mma");
  ComputeLibrary lib_dq = mc->load_library("affine_dequant");
  ComputeFunction f_one = lib_mma.function("gemm_i8i8_sc_f16_n64");
  ComputeFunction f_ka = lib_mma.function("gemm_i8i8_sc_f16_n64_kacc");
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
    const double s1 = secs_(t0, t1), s2 = secs_(t1, t2), s3 = secs_(t2, t3);
    std::printf("[gemm_i8k] M=%4d N=%5d K=%5d: single %.0f GF/s (%.2f ms)"
                "  kacc %.0f GF/s (%.2f ms, %.2fx)  g512 %.0f GF/s "
                "(%.2f ms, %.2fx)\n",
                M, N, K, gflop / s1, s1 * 1e3 / iters,
                gflop / s2, s2 * 1e3 / iters, s1 / s2,
                gflop / s3, s3 * 1e3 / iters, s1 / s3);
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
