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
  };
  const Variant vs[] = {
      {"dense_gemm_mma_t_f16",          64,  64, 128, "64x64/4"},
      {"dense_gemm_mma_t_n128_f16",    128, 128, 256, "128x128/8"},
      {"dense_gemm_mma_t_n128x256_f16",128, 256, 256, "128x256/8"},
  };
  constexpr int NV = 3;
  ComputeFunction fns[NV];
  for (int i = 0; i < NV; ++i) { fns[i] = lib.function(vs[i].fn); }
  if (!fns[0].valid()) {
    std::printf("[gemm_tune] no matrix cores -- skipped.\n");
    return;
  }

  const Shape shapes[] = {
      {1711, 4096, 2560, "qkv@1711"},
      {1711, 2560, 4096, "o@1711"},
      {1711, 9728, 2560, "gate_up@1711"},
      {1711, 2560, 9728, "down@1711"},
      // Unaligned M and N (not multiples of 128/256) -- exercises the
      // matmul2d tensor-extent tail clamp for the larger production tiles.
      {1700, 2600, 4100, "tail-a"},
      {777,  1990, 9700, "tail-b"},
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
      std::printf(" %s %.0f%s |", vs[i].tag, g, rel < 3e-2 ? "" : "(BAD)");
      EXPECT_TRUE(rel < 3e-2);
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
