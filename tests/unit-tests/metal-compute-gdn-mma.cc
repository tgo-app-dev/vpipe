// metal-compute-gdn-mma.cc -- Qwen3.5 gated-DeltaNet (GDN) prefill step:
// the shipping recurrent-ndv speedup + the (parked) chunkwise de-risk.
//
// recurrent_v2_sweep: validates the ndv2/4/8 recurrent step (N dv per
//   simdgroup, k/q read once/step) is token-EXACT vs the per-dv v1 kernel
//   and (under VPIPE_GDN_BENCH) sweeps the speedup -- ndv4 is ~1.33x at
//   prefill T and is the shipped prefill GDN kernel.
// bmm_correctness_and_throughput / recurrent_baseline: the chunkwise
//   matrix-core de-risk -- batched matmul2d primitives + fast triangular
//   solve vs the recurrent baseline. Conclusion: chunkwise ~= recurrent on
//   M5 (solve-bound), so it was NOT built. Bench bodies gated behind
//   VPIPE_GDN_BENCH; the correctness checks always run.
//
// Runs in BOTH builds; pure metal-compute, no model load.

#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

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

inline std::uint16_t f32_to_h(float f) {
  std::uint32_t x; std::memcpy(&x, &f, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  std::int32_t exp = (std::int32_t)((x >> 23) & 0xff) - 127 + 15;
  std::uint32_t man = x & 0x7fffffu;
  if (exp <= 0) { return (std::uint16_t)sign; }
  if (exp >= 0x1f) { return (std::uint16_t)(sign | 0x7c00u); }
  return (std::uint16_t)(sign | (exp << 10) | (man >> 13));
}

MetalCompute* get_mc_(Session& s) {
  MetalCompute* mc = s.metal_compute();
  return (mc != nullptr && mc->valid()) ? mc : nullptr;
}
double secs_(std::chrono::steady_clock::time_point a,
             std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

inline float h_to_f32(std::uint16_t h) {
  const std::uint32_t sign = (std::uint32_t)(h & 0x8000) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1f;
  const std::uint32_t man = h & 0x3ff;
  std::uint32_t out;
  if (exp == 0) { out = sign; }
  else if (exp == 0x1f) { out = sign | 0x7f800000u | (man << 13); }
  else { out = sign | ((exp - 15 + 127) << 23) | (man << 13); }
  float f; std::memcpy(&f, &out, 4); return f;
}

}  // namespace

// Correctness + throughput for the batched matmul2d GDN primitives at the
// real chunkwise shapes (Hv heads x chunks). De-risks the chunkwise path:
// if matrix cores are poor at these tiny tiles (BT<=64, K in {64,128}) the
// approach can't beat the recurrent step. Compares the summed matmul cost
// (x24 layers) against the recurrent baseline (~158 ms @ T=1711).
TEST(gdn_mma, bmm_correctness_and_throughput) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("gdn_mma");
  ComputeFunction nt = lib.function("gdn_bmm_nt_f16");
  ComputeFunction nn = lib.function("gdn_bmm_nn_f16");
  ComputeFunction tn = lib.function("gdn_bmm_tn_f16");
  if (!nt.valid()) { std::printf("[gdn_bmm] no matrix cores -- skip\n"); return; }

  // ---- correctness: NT (kkT=K@K^T), NN (P@d), TN (d^T@k) small batch ----
  auto rel_check = [&](ComputeFunction& fn, int M, int N, int K, int Batch,
                       int tL, int tR, const char* tag) {
    // A stored: tL? [K,M]:[M,K]; B stored: tR? [N,K]:[K,N]; C [M,N].
    const int aR = tL ? K : M, aC = tL ? M : K;
    const int bR = tR ? N : K, bC = tR ? K : N;
    std::mt19937 rng(1 + M + N + K);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<std::uint16_t> A((size_t)Batch * aR * aC),
        B((size_t)Batch * bR * bC), C((size_t)Batch * M * N, 0);
    for (auto& e : A) { e = f32_to_h(d(rng) * 0.3f); }
    for (auto& e : B) { e = f32_to_h(d(rng) * 0.3f); }
    auto Ax = [&](int b, int m, int k) {
      return h_to_f32(A[((size_t)b * aR + (tL ? k : m)) * aC + (tL ? m : k)]);
    };
    auto Bx = [&](int b, int k, int n) {
      return h_to_f32(B[((size_t)b * bR + (tR ? n : k)) * bC + (tR ? k : n)]);
    };
    SharedBuffer ab = mc->make_shared_buffer(A.size() * 2);
    SharedBuffer bb = mc->make_shared_buffer(B.size() * 2);
    SharedBuffer cb = mc->make_shared_buffer(C.size() * 2);
    std::memcpy(ab.contents(), A.data(), A.size() * 2);
    std::memcpy(bb.contents(), B.data(), B.size() * 2);
    CommandStream st = mc->make_command_stream();
    { ComputeEncoder enc = st.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, ab); enc.set_buffer(1, bb); enc.set_buffer(2, cb);
      enc.set_constant(3, M); enc.set_constant(4, N); enc.set_constant(5, K);
      enc.set_constant(6, aR * aC); enc.set_constant(7, bR * bC);
      enc.set_constant(8, M * N);
      enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                    (unsigned)((M + 63) / 64), (unsigned)Batch}, {128, 1, 1});
    }
    st.commit().wait();
    const auto* cp = static_cast<const std::uint16_t*>(cb.contents());
    double num = 0, den = 0;
    for (int b = 0; b < Batch; ++b) {
      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          float ref = 0;
          for (int k = 0; k < K; ++k) { ref += Ax(b, m, k) * Bx(b, k, n); }
          const float got = h_to_f32(cp[((size_t)b * M + m) * N + n]);
          num += (got - ref) * (got - ref); den += ref * ref;
        }
      }
    }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    std::printf("[gdn_bmm] %-4s M=%d N=%d K=%d B=%d  rel-L2=%.3e%s\n",
                tag, M, N, K, Batch, rel, rel < 3e-2 ? "" : " BAD");
    EXPECT_TRUE(rel < 3e-2);
  };
  rel_check(nt, 64, 64, 128, 4, 0, 1, "nt");
  rel_check(nt, 64, 128, 128, 4, 0, 1, "nt2");
  rel_check(nn, 64, 128, 64, 4, 0, 0, "nn");
  rel_check(tn, 128, 128, 64, 4, 1, 0, "tn");

  // The throughput/de-risk timing below is heavy; gate it (the rel_check
  // correctness above always runs). VPIPE_GDN_BENCH=1 to enable.
  if (!std::getenv("VPIPE_GDN_BENCH")) { return; }

  // ---- throughput at GDN chunkwise shapes (T=1711, BT=64) ----
  const int BT = 64, Dk = 128, Dv = 128, Hv = 32;
  const int n_chunks = (1711 + BT - 1) / BT;       // 27
  const int Bb = Hv * n_chunks;                     // 864 batched matmuls
  struct MM { ComputeFunction* fn; int M, N, K; const char* tag; };
  MM mms[] = {
      {&nt, BT, BT, Dk, "kkT"}, {&nt, BT, BT, Dk, "qkT"},
      {&nt, BT, Dv, Dk, "wS"},  {&nt, BT, Dv, Dk, "qS"},
      {&nn, BT, Dv, BT, "Pd"},  {&tn, Dv, Dk, BT, "dTk"},
  };
  SharedBuffer big = mc->make_shared_buffer((size_t)Bb * 256 * 256 * 2);
  double total_ms = 0;

  // Doubling-inverse probe: the blocked solve computes M=(I+L)^-1 via ~12
  // batched 64x64x64 NN matmuls (P<-P^2, N<-N+N*P over log2(64)=6 steps)
  // then X=M*B. Time one 64^3 NN matmul (batch=Hv*chunks) to estimate the
  // whole inverse-based solve vs the 108 ms register substitution.
  {
    auto launch = [&](int reps) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int r = 0; r < reps; ++r) {
          enc.set_function(nn);
          enc.set_buffer(0, big); enc.set_buffer(1, big); enc.set_buffer(2, big);
          enc.set_constant(3, 64); enc.set_constant(4, 64); enc.set_constant(5, 64);
          enc.set_constant(6, 64 * 64); enc.set_constant(7, 64 * 64);
          enc.set_constant(8, 64 * 64);
          enc.dispatch({128u, 1u, (unsigned)Bb}, {128, 1, 1});
        }
      }
      st.commit().wait();
    };
    launch(3);
    const int it = 20;
    const auto t0 = std::chrono::steady_clock::now();
    launch(it);
    const auto t1 = std::chrono::steady_clock::now();
    const double m64 = secs_(t0, t1) / it * 1e3;
    std::printf("[gdn_bmm] nn64^3 B=%d  %.4f ms  -> doubling-inverse est "
                "~%.2f ms/layer (12x) | x24 ~%.1f ms\n",
                Bb, m64, m64 * 12, m64 * 12 * 24);
  }
  for (auto& m : mms) {
    auto launch = [&](int reps) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int r = 0; r < reps; ++r) {
          enc.set_function(*m.fn);
          enc.set_buffer(0, big); enc.set_buffer(1, big); enc.set_buffer(2, big);
          enc.set_constant(3, m.M); enc.set_constant(4, m.N);
          enc.set_constant(5, m.K);
          enc.set_constant(6, m.M * m.K); enc.set_constant(7, m.N * m.K);
          enc.set_constant(8, m.M * m.N);
          enc.dispatch({(unsigned)(((m.N + 63) / 64) * 128),
                        (unsigned)((m.M + 63) / 64), (unsigned)Bb}, {128, 1, 1});
        }
      }
      st.commit().wait();
    };
    launch(3);
    const int it = 20;
    const auto t0 = std::chrono::steady_clock::now();
    launch(it);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = secs_(t0, t1) / it * 1e3;
    total_ms += ms;
    std::printf("[gdn_bmm] %-4s B=%d M=%d N=%d K=%d  %.4f ms\n",
                m.tag, Bb, m.M, m.N, m.K, ms);
  }
  std::printf("[gdn_bmm] all-6-matmuls/layer %.4f ms | x24 layers %.2f ms "
              "(recurrent baseline ~158 ms @T=1711)\n",
              total_ms, total_ms * 24);

  // The chunkwise path's one inherently-SEQUENTIAL kernel: the per-(head,
  // chunk) triangular solve (I+L)[u|w]=[B_v|B_w]. Bench the parked
  // build_L_solve at GDN shapes (batch=Hv*chunks, bt=64, rhs_w=Dv+Dk=256)
  // -- if it dominates, the chunkwise approach can't beat recurrent.
  ComputeLibrary wy = mc->load_library("qwen3_5_gdn_wy");
  ComputeFunction solve = wy.function("qwen3_5_gdn_chunk_build_L_solve_f32");
  if (solve.valid()) {
    const int rhs_w = Dv + Dk;   // 256
    SharedBuffer kkTb = mc->make_shared_buffer((size_t)Bb * BT * BT * 4);
    SharedBuffer gcb = mc->make_shared_buffer((size_t)Bb * BT * 4);
    SharedBuffer betab = mc->make_shared_buffer((size_t)Bb * BT * 4);
    SharedBuffer Bcat = mc->make_shared_buffer((size_t)Bb * BT * rhs_w * 4);
    SharedBuffer Xout = mc->make_shared_buffer((size_t)Bb * BT * rhs_w * 4);
    // G_cum must be nonzero (kernel divides by it); fill 1.0.
    auto* gp = static_cast<float*>(gcb.contents());
    for (size_t i = 0; i < (size_t)Bb * BT; ++i) { gp[i] = 1.0f; }
    auto launch = [&](int reps) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int r = 0; r < reps; ++r) {
          enc.set_function(solve);
          enc.set_buffer(0, kkTb); enc.set_buffer(1, gcb); enc.set_buffer(2, betab);
          enc.set_buffer(3, Bcat); enc.set_buffer(4, Xout);
          enc.set_constant(5, BT); enc.set_constant(6, rhs_w);
          enc.dispatch({32u, (unsigned)((rhs_w + 31) / 32), (unsigned)Bb},
                       {32, 1, 1});
        }
      }
      st.commit().wait();
    };
    launch(3);
    const int it = 20;
    const auto t0 = std::chrono::steady_clock::now();
    launch(it);
    const auto t1 = std::chrono::steady_clock::now();
    const double sm = secs_(t0, t1) / it * 1e3;
    std::printf("[gdn_bmm] solve(parked) B=%d bt=%d rhs=%d  %.4f ms/layer | "
                "x24 %.2f ms\n", Bb, BT, rhs_w, sm, sm * 24);

    // Fast solve (build L once + register substitution).
    ComputeFunction fsolve = lib.function("gdn_solve_f16");
    if (fsolve.valid()) {
      // Correctness vs CPU forward-sub on a small batch with random
      // kkT(half)/G/beta/B.
      {
        const int bt = 16, rw = 24, B2 = 3;
        std::mt19937 rng(5);
        std::uniform_real_distribution<float> d(-0.5f, 0.5f);
        std::vector<std::uint16_t> kk((size_t)B2 * bt * bt);
        std::vector<float> G((size_t)B2 * bt), be((size_t)B2 * bt),
            Bm((size_t)B2 * bt * rw), Xc((size_t)B2 * bt * rw, 0);
        for (auto& e : kk) { e = f32_to_h(d(rng)); }
        for (auto& e : G) { e = 0.6f + 0.4f * std::fabs(d(rng)); }
        for (auto& e : be) { e = 0.5f + 0.3f * d(rng); }
        for (auto& e : Bm) { e = d(rng); }
        SharedBuffer kb = mc->make_shared_buffer(kk.size() * 2);
        SharedBuffer Gb = mc->make_shared_buffer(G.size() * 4);
        SharedBuffer beb = mc->make_shared_buffer(be.size() * 4);
        SharedBuffer Bb2 = mc->make_shared_buffer(Bm.size() * 4);
        SharedBuffer Xb = mc->make_shared_buffer(Xc.size() * 4);
        std::memcpy(kb.contents(), kk.data(), kk.size() * 2);
        std::memcpy(Gb.contents(), G.data(), G.size() * 4);
        std::memcpy(beb.contents(), be.data(), be.size() * 4);
        std::memcpy(Bb2.contents(), Bm.data(), Bm.size() * 4);
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder enc = st.begin_compute();
          enc.set_function(fsolve);
          enc.set_buffer(0, kb); enc.set_buffer(1, Gb); enc.set_buffer(2, beb);
          enc.set_buffer(3, Bb2); enc.set_buffer(4, Xb);
          enc.set_constant(5, bt); enc.set_constant(6, rw);
          enc.dispatch({(unsigned)rw, 1, (unsigned)B2}, {(unsigned)rw, 1, 1});
        }
        st.commit().wait();
        const auto* xg = static_cast<const float*>(Xb.contents());
        double num = 0, den = 0;
        for (int b = 0; b < B2; ++b) {
          for (int c = 0; c < rw; ++c) {
            std::vector<float> xr(bt);
            for (int i = 0; i < bt; ++i) {
              float acc = Bm[((size_t)b * bt + i) * rw + c];
              for (int j = 0; j < i; ++j) {
                const float L = be[(size_t)b * bt + i] *
                    (G[(size_t)b * bt + i] / G[(size_t)b * bt + j]) *
                    h_to_f32(kk[((size_t)b * bt + i) * bt + j]);
                acc -= L * xr[j];
              }
              xr[i] = acc;
              const float got = xg[((size_t)b * bt + i) * rw + c];
              num += (got - xr[i]) * (got - xr[i]); den += xr[i] * xr[i];
            }
          }
        }
        const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
        std::printf("[gdn_bmm] fast-solve rel-L2=%.3e%s\n", rel,
                    rel < 1e-3 ? "" : " BAD");
        EXPECT_TRUE(rel < 1e-3);
      }
      auto flaunch = [&](int reps) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder enc = st.begin_compute();
          for (int r = 0; r < reps; ++r) {
            enc.set_function(fsolve);
            enc.set_buffer(0, big); enc.set_buffer(1, gcb); enc.set_buffer(2, betab);
            enc.set_buffer(3, Bcat); enc.set_buffer(4, Xout);
            enc.set_constant(5, BT); enc.set_constant(6, rhs_w);
            enc.dispatch({(unsigned)rhs_w, 1, (unsigned)Bb},
                         {(unsigned)rhs_w, 1, 1});
          }
        }
        st.commit().wait();
      };
      flaunch(3);
      const auto f0 = std::chrono::steady_clock::now();
      flaunch(it);
      const auto f1 = std::chrono::steady_clock::now();
      const double fm = secs_(f0, f1) / it * 1e3;
      std::printf("[gdn_bmm] solve(fast)   B=%d bt=%d rhs=%d  %.4f ms/layer | "
                  "x24 %.2f ms\n", Bb, BT, rhs_w, fm, fm * 24);
      std::printf("[gdn_bmm] CHUNKWISE est (matmuls+fast-solve)/layer %.3f ms "
                  "| x24 %.1f ms vs recurrent ~158 ms\n",
                  total_ms + fm, (total_ms + fm) * 24);
    }
  }
  EXPECT_TRUE(total_ms >= 0.0);
}

// Baseline: recurrent GDN step at prefill shapes. Reports ms/call (one
// GDN layer's recurrence over T tokens); multiply by the GDN layer count
// (24 for Qwen3.5-4B) to estimate the prefill GDN budget.
TEST(gdn_mma, recurrent_baseline) {
  if (!std::getenv("VPIPE_GDN_BENCH")) { return; }   // pure bench
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("qwen3_5_gated_delta");
  ComputeFunction fn = lib.function("qwen3_5_gated_delta_step_f16");
  if (!fn.valid()) {
    std::printf("[gdn_mma] recurrent kernel unavailable -- skipped.\n");
    return;
  }

  const int Hk = 16, Hv = 32, Dk = 128, Dv = 128;
  const int gdn_layers = 24;     // Qwen3.5-4B: ~24 GDN layers
  struct Cfg { int T; const char* tag; };
  const Cfg cfgs[] = {{64, "T64"}, {256, "T256"}, {512, "T512"},
                      {1024, "T1024"}, {1711, "T1711"}};

  for (const auto& cf : cfgs) {
    const int T = cf.T;
    std::mt19937 rng(7 + T);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    std::vector<std::uint16_t> q((size_t)T * Hk * Dk), k((size_t)T * Hk * Dk),
        v((size_t)T * Hv * Dv);
    for (auto& e : q) { e = f32_to_h(d(rng) * 0.2f); }
    for (auto& e : k) { e = f32_to_h(d(rng) * 0.2f); }
    for (auto& e : v) { e = f32_to_h(d(rng) * 0.2f); }
    std::vector<float> g((size_t)T * Hv), beta((size_t)T * Hv);
    for (auto& e : g) { e = 0.95f + 0.05f * d(rng); }       // decay ~ (.9,1)
    for (auto& e : beta) { e = 0.5f + 0.4f * d(rng); }
    std::vector<float> state((size_t)Hv * Dv * Dk, 0.0f);

    SharedBuffer qb = mc->make_shared_buffer(q.size() * 2);
    SharedBuffer kb = mc->make_shared_buffer(k.size() * 2);
    SharedBuffer vb = mc->make_shared_buffer(v.size() * 2);
    SharedBuffer gb = mc->make_shared_buffer(g.size() * 4);
    SharedBuffer bb = mc->make_shared_buffer(beta.size() * 4);
    SharedBuffer sib = mc->make_shared_buffer(state.size() * 4);
    SharedBuffer sob = mc->make_shared_buffer(state.size() * 4);
    SharedBuffer yb = mc->make_shared_buffer((size_t)T * Hv * Dv * 2);
    std::memcpy(qb.contents(), q.data(), q.size() * 2);
    std::memcpy(kb.contents(), k.data(), k.size() * 2);
    std::memcpy(vb.contents(), v.data(), v.size() * 2);
    std::memcpy(gb.contents(), g.data(), g.size() * 4);
    std::memcpy(bb.contents(), beta.data(), beta.size() * 4);
    std::memcpy(sib.contents(), state.data(), state.size() * 4);

    // Dispatch all GDN layers into ONE command stream (as real prefill
    // does) so the measurement is GPU compute, not per-submit latency.
    // The dispatches chain on sib/sob/yb hazards -> serialized, matching
    // the sequential layer dependency.
    auto run = [&]() {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        for (int l = 0; l < gdn_layers; ++l) {
          enc.set_function(fn);
          enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
          enc.set_buffer(3, gb); enc.set_buffer(4, bb);
          enc.set_buffer(5, sib); enc.set_buffer(6, yb); enc.set_buffer(7, sob);
          enc.set_constant(8, T);
          enc.set_constant(9, Hk); enc.set_constant(10, Hv);
          enc.dispatch({32, (unsigned)Dv, (unsigned)Hv}, {32, 4, 1});
        }
      }
      st.commit().wait();
    };

    auto bench = [&](auto launch) {
      for (int w = 0; w < 3; ++w) { launch(); }
      const int it = 20;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < it; ++i) { launch(); }
      const auto t1 = std::chrono::steady_clock::now();
      return secs_(t0, t1) / it * 1e3;
    };
    const double ms = bench(run);     // total for all gdn_layers dispatches
    std::printf("[gdn_mma] recurrent %-6s T=%4d  %.4f ms/layer  | "
                "x%d layers = %.3f ms\n",
                cf.tag, T, ms / gdn_layers, gdn_layers, ms);
    EXPECT_TRUE(ms >= 0.0);
  }
}

// Recurrent v2 (NDV dv per simdgroup, k/q loaded once/step) vs v1. Verifies
// v2 is token-exact (bit-identical y + state) and sweeps NDV to find the
// fastest amortization factor. The dv-per-simdgroup amortizes the per-step
// k/q reads that v1 redundantly issues from all 128 dv-simdgroups of a head.
TEST(gdn_mma, recurrent_v2_sweep) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("qwen3_5_gated_delta");
  ComputeFunction v1 = lib.function("qwen3_5_gated_delta_step_f16");
  ComputeFunction v2[3] = {lib.function("qwen3_5_gated_delta_step_ndv2_f16"),
                           lib.function("qwen3_5_gated_delta_step_ndv4_f16"),
                           lib.function("qwen3_5_gated_delta_step_ndv8_f16")};
  if (!v1.valid() || !v2[0].valid() || !v2[1].valid() || !v2[2].valid()) {
    std::printf("[gdn_v2] kernels unavailable -- skipped.\n"); return;
  }
  const int Hk = 16, Hv = 32, Dk = 128, Dv = 128, GL = 24;
  const int Ts[] = {512, 1024, 1711};
  const int ndvs[] = {2, 4, 8};
  // Always validate ndv token-exactness (one T); the timing sweep across T
  // is heavy -> gate it on VPIPE_GDN_BENCH.
  const bool BENCH = std::getenv("VPIPE_GDN_BENCH") != nullptr;
  for (int T : Ts) {
    if (!BENCH && T != 512) { continue; }
    std::mt19937 rng(7 + T);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<std::uint16_t> q((size_t)T * Hk * Dk), k((size_t)T * Hk * Dk),
        v((size_t)T * Hv * Dv);
    for (auto& e : q) { e = f32_to_h(d(rng) * 0.2f); }
    for (auto& e : k) { e = f32_to_h(d(rng) * 0.2f); }
    for (auto& e : v) { e = f32_to_h(d(rng) * 0.2f); }
    std::vector<float> g((size_t)T * Hv), beta((size_t)T * Hv);
    for (auto& e : g) { e = 0.95f + 0.05f * d(rng); }
    for (auto& e : beta) { e = 0.5f + 0.4f * d(rng); }
    std::vector<float> st0((size_t)Hv * Dv * Dk, 0.0f);
    SharedBuffer qb = mc->make_shared_buffer(q.size() * 2);
    SharedBuffer kb = mc->make_shared_buffer(k.size() * 2);
    SharedBuffer vb = mc->make_shared_buffer(v.size() * 2);
    SharedBuffer gb = mc->make_shared_buffer(g.size() * 4);
    SharedBuffer bb = mc->make_shared_buffer(beta.size() * 4);
    SharedBuffer sib = mc->make_shared_buffer(st0.size() * 4);
    SharedBuffer sob = mc->make_shared_buffer(st0.size() * 4);
    SharedBuffer yb = mc->make_shared_buffer((size_t)T * Hv * Dv * 2);
    std::memcpy(qb.contents(), q.data(), q.size() * 2);
    std::memcpy(kb.contents(), k.data(), k.size() * 2);
    std::memcpy(vb.contents(), v.data(), v.size() * 2);
    std::memcpy(gb.contents(), g.data(), g.size() * 4);
    std::memcpy(bb.contents(), beta.data(), beta.size() * 4);
    std::memcpy(sib.contents(), st0.data(), st0.size() * 4);

    auto bench = [&](auto launch) {
      for (int w = 0; w < 3; ++w) { launch(); }
      const int it = 20;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < it; ++i) { launch(); }
      const auto t1 = std::chrono::steady_clock::now();
      return secs_(t0, t1) / it * 1e3;
    };
    // v1 reference (single dispatch -> reference y/state).
    {
      CommandStream s = mc->make_command_stream();
      { ComputeEncoder e = s.begin_compute();
        e.set_function(v1);
        e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
        e.set_buffer(3, gb); e.set_buffer(4, bb); e.set_buffer(5, sib);
        e.set_buffer(6, yb); e.set_buffer(7, sob);
        e.set_constant(8, T); e.set_constant(9, Hk); e.set_constant(10, Hv);
        e.dispatch({32, (unsigned)Dv, (unsigned)Hv}, {32, 4, 1});
      }
      s.commit().wait();
    }
    std::vector<std::uint16_t> y_ref((size_t)T * Hv * Dv);
    std::vector<float> st_ref((size_t)Hv * Dv * Dk);
    std::memcpy(y_ref.data(), yb.contents(), y_ref.size() * 2);
    std::memcpy(st_ref.data(), sob.contents(), st_ref.size() * 4);
    auto v1run = [&]() {
      CommandStream s = mc->make_command_stream();
      { ComputeEncoder e = s.begin_compute();
        for (int l = 0; l < GL; ++l) {
          e.set_function(v1);
          e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
          e.set_buffer(3, gb); e.set_buffer(4, bb); e.set_buffer(5, sib);
          e.set_buffer(6, yb); e.set_buffer(7, sob);
          e.set_constant(8, T); e.set_constant(9, Hk); e.set_constant(10, Hv);
          e.dispatch({32, (unsigned)Dv, (unsigned)Hv}, {32, 4, 1});
        }
      }
      s.commit().wait();
    };
    const double t_v1 = BENCH ? bench(v1run) : 0.0;
    if (BENCH) { std::printf("[gdn_v2] T=%4d  v1 %.3f ms/24L | ", T, t_v1); }
    for (int vi = 0; vi < 3; ++vi) {
      const int ndv = ndvs[vi];
      ComputeFunction& vf = v2[vi];
      auto v2run = [&]() {
        CommandStream s = mc->make_command_stream();
        { ComputeEncoder e = s.begin_compute();
          for (int l = 0; l < GL; ++l) {
            e.set_function(vf);
            e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
            e.set_buffer(3, gb); e.set_buffer(4, bb); e.set_buffer(5, sib);
            e.set_buffer(6, yb); e.set_buffer(7, sob);
            e.set_constant(8, T); e.set_constant(9, Hk); e.set_constant(10, Hv);
            e.dispatch({32, (unsigned)(Dv / ndv), (unsigned)Hv}, {32, 4, 1});
          }
        }
        s.commit().wait();
      };
      // correctness: single v2 dispatch vs v1 reference (bit-identical).
      std::memset(yb.contents(), 0, (size_t)T * Hv * Dv * 2);
      { CommandStream s = mc->make_command_stream();
        { ComputeEncoder e = s.begin_compute();
          e.set_function(vf);
          e.set_buffer(0, qb); e.set_buffer(1, kb); e.set_buffer(2, vb);
          e.set_buffer(3, gb); e.set_buffer(4, bb); e.set_buffer(5, sib);
          e.set_buffer(6, yb); e.set_buffer(7, sob);
          e.set_constant(8, T); e.set_constant(9, Hk); e.set_constant(10, Hv);
          e.dispatch({32, (unsigned)(Dv / ndv), (unsigned)Hv}, {32, 4, 1});
        }
        s.commit().wait();
      }
      const auto* yp = static_cast<const std::uint16_t*>(yb.contents());
      const auto* sp = static_cast<const float*>(sob.contents());
      int ymis = 0; double sdiff = 0;
      for (size_t i = 0; i < y_ref.size(); ++i) { if (yp[i] != y_ref[i]) ++ymis; }
      for (size_t i = 0; i < st_ref.size(); ++i) {
        sdiff = std::max(sdiff, (double)std::fabs(sp[i] - st_ref[i]));
      }
      if (BENCH) {
        const double tm = bench(v2run);
        std::printf("ndv%d %.3f(%.2fx,%s) ", ndv, tm, t_v1 / tm,
                    (ymis == 0 && sdiff == 0.0) ? "exact" : "DIFF");
      } else {
        std::printf("[gdn_v2] T=%d ndv%d %s\n", T, ndv,
                    (ymis == 0 && sdiff == 0.0) ? "token-exact" : "DIFF");
      }
      EXPECT_TRUE(ymis == 0 && sdiff == 0.0);
    }
    if (BENCH) { std::printf("\n"); }
  }
}
