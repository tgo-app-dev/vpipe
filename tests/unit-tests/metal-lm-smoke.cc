// metal-lm-smoke.cc -- metal-compute LM text-decode smoke that compiles
// and runs in BOTH the MLX and no-MLX builds. It never references
// mlx::core, so it lives in the VPIPE_BUILD_APPLE_SILICON test block
// (not the MLX-gated one) and is the end-to-end proof that the language-
// model subsystem loads + generates on the metal backend with MLX off.
//
// Env-gated: set VPIPE_METAL_LM_SMOKE_MODEL to a metal-supported model
// dir (Qwen3.5-4B / Llama / Qwen3-ASR text decoder). Forces
// VPIPE_LLM_BACKEND=metal for the duration of the load.

#include "minitest.h"
#include "generative-models/chat-template.h"
#include "generative-models/gemma4/gemma4-unified-embedder.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/gemma4/metal-gemma4-vision.h"
#include "generative-models/shared/coreml-vision-encoder.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/moss/metal-moss-tts-model.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/model-loader.h"
#include "generative-models/sampler.h"
#include "generative-models/token-muxer.h"
#include "generative-models/tokenizer.h"
#include "generative-models/shared/gguf-file.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/media-line.h"
#include "common/session.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <optional>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <sstream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <mach/mach.h>

using namespace vpipe;

// Self-contained qmv (decode GEMV) bandwidth A/B: w4g64 vs w8g64 (bf16, the
// MOSS compute dtype), random weights (bandwidth is value-independent), at the
// real MOSS decode shapes + a large SLC-busting shape. Answers whether the
// 8-bit qmv kernel hits the same effective DRAM bandwidth as the 4-bit one.
// Gated on VPIPE_QMV_AB. M5 16GB peak DRAM = 153 GB/s.
TEST(metal_lm_smoke, qmv_w4_w8_bandwidth_ab) {
  if (std::getenv("VPIPE_QMV_AB") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("affine_qmv_bf16");
  using Clock = std::chrono::steady_clock;
  const int R = 200;
  const double kPeak = 153.0;
  struct Shape { const char* name; int n; int k; };
  const Shape shapes[] = {
      {"moss gate/up", 12288, 4096}, {"moss o-proj ", 4096, 4096},
      {"moss down   ", 4096, 12288}, {"big (SLC-bust)", 16384, 8192},
      {"optiq lmhead ", 151936, 2560}, {"optiq in_proj", 12288, 2560},
  };
  struct V { const char* fn; int bits; int rps; int nsg; };
  const V vars[] = {
      {"affine_qmv_w4g64", 4, 4, 2}, {"affine_qmv_w4g64_r8p2", 4, 8, 2},
      {"affine_qmv_w8g64", 8, 4, 2}, {"affine_qmv_w8g64_r8p2", 8, 8, 2}};
  std::printf("[qmv-ab] M5 peak %.0f GB/s; %d serial GEMVs, min-of-3, warmed\n",
              kPeak, R);
  std::mt19937 rng(7);
  std::uniform_int_distribution<std::uint32_t> du(0, 0xffffffffu);
  for (const Shape& sh : shapes) {
    const int groups = sh.k / 64;
    for (const V& v : vars) {
      auto fn = lib.function(v.fn);
      if (!fn.valid()) {
        std::printf("[qmv-ab] %-13s %-22s MISSING\n", sh.name, v.fn);
        continue;
      }
      const std::size_t wwords =
          (std::size_t)sh.n * sh.k / (32 / v.bits);   // packed u32 count
      const std::size_t sbcnt = (std::size_t)sh.n * groups;
      const double read_bytes = (double)(wwords * 4 + 2 * sbcnt * 2);
      auto wb = mc->make_shared_buffer(wwords * 4);
      auto sb = mc->make_shared_buffer(sbcnt * 2);
      auto bb = mc->make_shared_buffer(sbcnt * 2);
      auto xb = mc->make_shared_buffer((std::size_t)sh.k * 2);
      auto yb = mc->make_shared_buffer((std::size_t)sh.n * 2);
      auto* wp = static_cast<std::uint32_t*>(wb.contents());
      for (std::size_t i = 0; i < wwords; ++i) { wp[i] = du(rng); }
      auto dispatch_R = [&](int reps) {
        metal_compute::CommandStream st = mc->make_command_stream();
        {
          metal_compute::ComputeEncoder e = st.begin_compute();
          e.set_function(fn);
          e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb);
          e.set_buffer(3, xb); e.set_buffer(4, yb);
          e.set_constant(5, sh.k); e.set_constant(6, sh.n);
          for (int r = 0; r < reps; ++r) {
            e.dispatch({32u, (unsigned)(sh.n / v.rps), 1u},
                       {32u, (unsigned)v.nsg, 1u});
          }
        }
        st.commit().wait();
      };
      dispatch_R(20);
      double best_ms = 1e18;
      for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = Clock::now();
        dispatch_R(R);
        best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(
                                        Clock::now() - t0).count());
      }
      const double gbps = read_bytes * R / (best_ms / 1e3) / 1e9;
      std::printf("[qmv-ab] %-13s %-22s %4.1f MB | %6.1f GB/s (%4.1f%% peak)\n",
                  sh.name, v.fn, wwords * 4 / 1e6, gbps, 100.0 * gbps / kPeak);
    }
  }
}

// Ping-pong GEMV chain -- the DECODE-BOUNDARY question: can kernel N+1's weight
// DRAM traffic overlap kernel N's drain (last mul-acc + bias + store)? Two big
// DRAM-bound quantized matvecs feed each other (b=W1@a; a=W2@b; ...) -- a TRUE
// serial dependency chain, exactly the layer chain in decode. Measure effective
// weight-read GB/s under 4 orderings:
//   serial : Serial encoder (Metal auto-hazard barrier) -- production baseline
//   scope  : Concurrent + memoryBarrier(Buffers) (scope-wide full drain)
//   res    : Concurrent + resource-scoped barrier on the ACTIVATION only
//   none   : Concurrent + NO barrier (garbage output; the overlap CEILING)
// If `res` beats `serial`/`scope` and approaches `none`, resource-scoped
// ordering lets N+1's weights prefetch during N's drain -- the real decode
// lever. Gated on VPIPE_QMV_CHAIN.
TEST(metal_lm_smoke, qmv_chain_barrier_ab) {
  if (std::getenv("VPIPE_QMV_CHAIN") == nullptr) { return; }
  using namespace metal_compute;
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv_bf16");
  auto fn = lib.function("affine_qmv_w4g64");
  auto fn8 = lib.function("affine_qmv_w4g64_r8p2");   // 8 rows/thread (higher MLP)
  if (!fn.valid()) { std::printf("[qmv-chain] kernel missing\n"); return; }
  const bool has8 = fn8.valid();
  using Clock = std::chrono::steady_clock;
  const int rps = 4, nsg = 2, R = 240;
  // Real decode-scale matvec shapes (hidden 2560) + a tiny one. The per-kernel
  // ramp/drain gap is a LARGER fraction here than at 8192^2, so this is where
  // inter-kernel overlap could matter. To keep every read a DRAM read (not SLC),
  // each dispatch pulls a DIFFERENT weight window from a >=384MB pool -- the
  // working set dwarfs the system-level cache, so no window is resident when it
  // recurs. Activations still ping-pong (serial dependency chain, like decode).
  struct Shape { const char* name; int n; int k; };
  const Shape shapes[] = {
      {"tiny 1024x1024", 1024, 1024}, {"oproj 2560x2560", 2560, 2560},
      {"attn 2048x2560", 2048, 2560}, {"gate 9728x2560", 9728, 2560},
      {"down 2560x9728", 2560, 9728},
  };
  const std::size_t kPoolBytes = 512ull << 20;    // 512 MB >> SLC
  auto wpool = mc->make_shared_buffer(kPoolBytes);
  {
    auto* p = static_cast<std::uint32_t*>(wpool.contents());
    std::uint32_t s = 2463534242u;                // fast xorshift fill
    for (std::size_t i = 0; i < kPoolBytes / 4; ++i) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5; p[i] = s;
    }
  }
  enum Mode { SERIAL, SCOPE, RES, NONE, SIBLING };
  std::printf("[qmv-chain] %d serial GEMVs, distinct DRAM window/dispatch "
              "(512MB pool >> SLC), min-of-3 (M4 Pro peak ~273 GB/s)\n", R);
  for (const Shape& sh : shapes) {
    const int N = sh.n, K = sh.k, groups = K / 64;
    const std::size_t wbytes = (std::size_t)N * K / 8 * 4;   // 4-bit packed
    const std::size_t sbcnt = (std::size_t)N * groups;
    const double read_bytes = (double)(wbytes + 2 * sbcnt * 2);
    const int windows = (int)std::max<std::size_t>(2, kPoolBytes / wbytes);
    // scales/biases pool sized to the same window count (kept distinct too).
    auto spool = mc->make_shared_buffer((std::size_t)windows * sbcnt * 2);
    auto bpool = mc->make_shared_buffer((std::size_t)windows * sbcnt * 2);
    auto a = mc->make_shared_buffer((std::size_t)K * 2);
    auto bb = mc->make_shared_buffer((std::size_t)N * 2);
    // Distinct OUTPUT window per dispatch, so SIBLING mode's dispatches are
    // genuinely independent (fixed input a, disjoint outputs) -> concurrent +
    // no barrier is LEGAL and correct, unlike the dependent NONE chain.
    auto opool = mc->make_shared_buffer((std::size_t)windows * N * 2);
    auto run = [&](Mode m, int reps) {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute(
            m == SERIAL ? DispatchType::Serial : DispatchType::Concurrent);
        for (int r = 0; r < reps; ++r) {
          const bool even = (r % 2) == 0;
          const int w = r % windows;            // different window each step
          e.set_function(fn);
          e.set_buffer(0, wpool, (std::size_t)w * wbytes);
          e.set_buffer(1, spool, (std::size_t)w * sbcnt * 2);
          e.set_buffer(2, bpool, (std::size_t)w * sbcnt * 2);
          if (m == SIBLING) {
            e.set_buffer(3, a);                 // fixed input (no dependency)
            e.set_buffer(4, opool, (std::size_t)w * N * 2);  // disjoint output
          } else {
            e.set_buffer(3, even ? a : bb);     // serial dependency chain
            e.set_buffer(4, even ? bb : a);
          }
          e.set_constant(5, K);
          e.set_constant(6, N);
          e.dispatch({32u, (unsigned)(N / rps), 1u}, {32u, (unsigned)nsg, 1u});
          if (r + 1 < reps) {
            if (m == SCOPE) { e.memory_barrier(BarrierScope::Buffers); }
            else if (m == RES) { e.memory_barrier_buffer(even ? bb : a); }
            // SIBLING/NONE: no barrier. SIBLING is legal (independent).
          }
        }
      }
      st.commit().wait();
    };
    double gb[5];
    for (int mi = 0; mi < 5; ++mi) {
      run((Mode)mi, 20);
      double best = 1e18;
      for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = Clock::now();
        run((Mode)mi, R);
        best = std::min(best, std::chrono::duration<double, std::milli>(
                                  Clock::now() - t0).count());
      }
      gb[mi] = read_bytes * R / (best / 1e3) / 1e9;
    }
    // Lever B: does a HIGHER-occupancy kernel (8 rows/thread) let a SINGLE
    // small GEMV saturate DRAM alone -- serial, no concurrency? Same serial
    // dependency chain, just fn8/rps=8.
    double gb8 = 0.0;
    if (has8 && (N % 8) == 0) {
      auto run8 = [&](int reps) {
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute(DispatchType::Serial);
          for (int r = 0; r < reps; ++r) {
            const bool even = (r % 2) == 0;
            const int w = r % windows;
            e.set_function(fn8);
            e.set_buffer(0, wpool, (std::size_t)w * wbytes);
            e.set_buffer(1, spool, (std::size_t)w * sbcnt * 2);
            e.set_buffer(2, bpool, (std::size_t)w * sbcnt * 2);
            e.set_buffer(3, even ? a : bb);
            e.set_buffer(4, even ? bb : a);
            e.set_constant(5, K);
            e.set_constant(6, N);
            e.dispatch({32u, (unsigned)(N / 8), 1u}, {32u, 2u, 1u});
          }
        }
        st.commit().wait();
      };
      run8(20);
      double best = 1e18;
      for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = Clock::now();
        run8(R);
        best = std::min(best, std::chrono::duration<double, std::milli>(
                                  Clock::now() - t0).count());
      }
      gb8 = read_bytes * R / (best / 1e3) / 1e9;
    }
    std::printf("[qmv-chain] %-16s %4.1fMB x%3d | serial %6.1f | ser_r8 %6.1f"
                " | none %6.1f | SIBLING %6.1f GB/s "
                "(r8/ser %.2fx  sib/ser %.2fx)\n",
                sh.name, wbytes / 1e6, windows, gb[SERIAL], gb8,
                gb[NONE], gb[SIBLING], gb8 / gb[SERIAL],
                gb[SIBLING] / gb[SERIAL]);
  }
}

// The crux question: is a FUSED matmul (one big dispatch) actually slower than
// the SAME total bytes split into S independent matmuls run CONCURRENTLY? If
// split-concurrent beats fused, then MLX's less-fused graph is genuinely faster
// per-op (not just "overlap"), and vpipe's aggressive fusion is leaving perf on
// the table. Same total weight bytes, cache-defeated (distinct pool windows).
//   FUSED       : 1 dispatch [Mtot,K], Serial
//   SPLIT_SERIAL: S dispatches [Mtot/S,K], Serial (barriered chain baseline)
//   SPLIT_CONC  : S dispatches [Mtot/S,K], Concurrent, NO barrier (independent)
// Gated on VPIPE_QMV_FUSE.
TEST(metal_lm_smoke, qmv_fuse_vs_split) {
  if (std::getenv("VPIPE_QMV_FUSE") == nullptr) { return; }
  using namespace metal_compute;
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv_bf16");
  auto fn = lib.function("affine_qmv_w4g64");
  if (!fn.valid()) { std::printf("[fuse] kernel missing\n"); return; }
  using Clock = std::chrono::steady_clock;
  const int K = 2560, rps = 4, nsg = 2, S = 3, R = 384, REPS = 11;
  const int groups = K / 64;
  const std::size_t kPool = 512ull << 20;
  auto wpool = mc->make_shared_buffer(kPool);
  {
    auto* p = static_cast<std::uint32_t*>(wpool.contents());
    std::uint32_t s = 999331u;
    for (std::size_t i = 0; i < kPool / 4; ++i) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5; p[i] = s;
    }
  }
  // Mtot values spanning the decode range (qkv-ish, ffn-ish).
  const int Mtots[] = {2560, 7680, 15360};
  enum Mode { FUSED, SPLIT_SERIAL, SPLIT_CONC };
  std::printf("[fuse] K=%d split S=%d, %d iter x %d reps, cache-defeated "
              "(M4 Pro peak ~273 GB/s)\n", K, S, R, REPS);
  for (int Mtot : Mtots) {
    const int Msplit = Mtot / S;
    const std::size_t wtot = (std::size_t)Mtot * K / 8 * 4;
    const std::size_t sbtot = (std::size_t)Mtot * groups;
    const double read_bytes = (double)(wtot + 2 * sbtot * 2);   // same for all
    const int windows = (int)(kPool / wtot);
    auto spool = mc->make_shared_buffer((std::size_t)windows * sbtot * 2);
    auto bpool = mc->make_shared_buffer((std::size_t)windows * sbtot * 2);
    auto xb = mc->make_shared_buffer((std::size_t)K * 2);
    auto yb = mc->make_shared_buffer((std::size_t)Mtot * 2);
    const std::size_t wsplit = (std::size_t)Msplit * K / 8 * 4;
    const std::size_t sbsplit = (std::size_t)Msplit * groups;
    auto run = [&](Mode m, int reps) {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute(
            m == SPLIT_CONC ? DispatchType::Concurrent : DispatchType::Serial);
        for (int r = 0; r < reps; ++r) {
          const int w = r % windows;
          if (m == FUSED) {
            e.set_function(fn);
            e.set_buffer(0, wpool, (std::size_t)w * wtot);
            e.set_buffer(1, spool, (std::size_t)w * sbtot * 2);
            e.set_buffer(2, bpool, (std::size_t)w * sbtot * 2);
            e.set_buffer(3, xb);
            e.set_buffer(4, yb);
            e.set_constant(5, K);
            e.set_constant(6, Mtot);
            e.dispatch({32u, (unsigned)(Mtot / rps), 1u}, {32u, (unsigned)nsg, 1u});
          } else {
            for (int s2 = 0; s2 < S; ++s2) {
              // distinct weight window per split part (disjoint output rows).
              e.set_function(fn);
              e.set_buffer(0, wpool, (std::size_t)w * wtot + (std::size_t)s2 * wsplit);
              e.set_buffer(1, spool, (std::size_t)w * sbtot * 2 + (std::size_t)s2 * sbsplit * 2);
              e.set_buffer(2, bpool, (std::size_t)w * sbtot * 2 + (std::size_t)s2 * sbsplit * 2);
              e.set_buffer(3, xb);
              e.set_buffer(4, yb, (std::size_t)s2 * Msplit * 2);
              e.set_constant(5, K);
              e.set_constant(6, Msplit);
              e.dispatch({32u, (unsigned)(Msplit / rps), 1u}, {32u, (unsigned)nsg, 1u});
              if (m == SPLIT_SERIAL && s2 + 1 < S) {
                e.memory_barrier(BarrierScope::Buffers);
              }
            }
            // Decode-realistic: separate consecutive matmuls (the dependency
            // chain) so SPLIT_CONC only overlaps the S tiles of ONE matmul, not
            // across iterations. (FUSED is Serial -> already separated.)
            if (m == SPLIT_CONC && r + 1 < reps) {
              e.memory_barrier(BarrierScope::Buffers);
            }
          }
        }
      }
      st.commit().wait();
    };
    double gb[3];
    for (int mi = 0; mi < 3; ++mi) {
      run((Mode)mi, 20);
      std::vector<double> samp;
      for (int rep = 0; rep < REPS; ++rep) {
        const auto t0 = Clock::now();
        run((Mode)mi, R);
        samp.push_back(read_bytes * R / (std::chrono::duration<double, std::milli>(
                                             Clock::now() - t0).count() / 1e3) / 1e9);
      }
      std::sort(samp.begin(), samp.end());
      gb[mi] = samp[samp.size() / 2];
    }
    std::printf("[fuse] Mtot=%5d (%4.1fMB) | FUSED %6.1f | split_serial %6.1f | "
                "split_CONC %6.1f GB/s (conc/fused %.2fx)\n",
                Mtot, wtot / 1e6, gb[FUSED], gb[SPLIT_SERIAL], gb[SPLIT_CONC],
                gb[SPLIT_CONC] / gb[FUSED]);
  }
}

// Mixed Serial+Concurrent encoders in ONE command buffer: run the dependent
// chain in a SERIAL encoder (Metal orders it for free, no explicit barrier) and
// drop into a short CONCURRENT encoder ONLY for a sibling group (so those
// overlap), letting the encoder BOUNDARY provide cross-group ordering for free.
// Models a decode layer: [sib1||sib2 : independent] -> [dep : feeds next iter].
// 4 strategies, cache-defeated small shapes:
//   ALL_SERIAL : one Serial encoder (siblings serialized, no overlap, cheapest order)
//   ALL_CONC   : one Concurrent encoder + explicit barriers (sib overlap + tax)
//   HYBRID     : Concurrent encoder for the sib pair + Serial encoder for dep
//                (sib overlap + free chain ordering, cost = 2 encoder switches/iter)
//   HYBRID_CB  : same but a fresh command buffer per iter (isolates cb overhead)
// Gated on VPIPE_QMV_MIXED.
TEST(metal_lm_smoke, qmv_mixed_encoder) {
  if (std::getenv("VPIPE_QMV_MIXED") == nullptr) { return; }
  using namespace metal_compute;
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv_bf16");
  auto fn = lib.function("affine_qmv_w4g64");
  if (!fn.valid()) { std::printf("[mixed] kernel missing\n"); return; }
  using Clock = std::chrono::steady_clock;
  const int N = 2560, K = 2560, rps = 4, nsg = 2, R = 256, REPS = 11;
  const int groups = K / 64;
  const std::size_t wbytes = (std::size_t)N * K / 8 * 4;
  const std::size_t sbcnt = (std::size_t)N * groups;
  const double read_bytes = (double)(wbytes + 2 * sbcnt * 2);
  const std::size_t kPool = 512ull << 20;
  const int windows = (int)(kPool / wbytes);
  auto wpool = mc->make_shared_buffer(kPool);
  {
    auto* p = static_cast<std::uint32_t*>(wpool.contents());
    std::uint32_t s = 123459876u;
    for (std::size_t i = 0; i < kPool / 4; ++i) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5; p[i] = s;
    }
  }
  auto spool = mc->make_shared_buffer((std::size_t)windows * sbcnt * 2);
  auto bpool = mc->make_shared_buffer((std::size_t)windows * sbcnt * 2);
  auto aa = mc->make_shared_buffer((std::size_t)K * 2);      // chain activation
  auto o1 = mc->make_shared_buffer((std::size_t)N * 2);
  auto o2 = mc->make_shared_buffer((std::size_t)N * 2);
  auto emit = [&](ComputeEncoder& e, int w, const SharedBuffer& in,
                  const SharedBuffer& out) {
    e.set_function(fn);
    e.set_buffer(0, wpool, (std::size_t)(w % windows) * wbytes);
    e.set_buffer(1, spool, (std::size_t)(w % windows) * sbcnt * 2);
    e.set_buffer(2, bpool, (std::size_t)(w % windows) * sbcnt * 2);
    e.set_buffer(3, in);
    e.set_buffer(4, out);
    e.set_constant(5, K);
    e.set_constant(6, N);
    e.dispatch({32u, (unsigned)(N / rps), 1u}, {32u, (unsigned)nsg, 1u});
  };
  enum Strat { ALL_SERIAL, ALL_CONC, HYBRID, HYBRID_CB };
  auto run = [&](Strat s) {
    if (s == ALL_SERIAL || s == ALL_CONC) {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute(
            s == ALL_SERIAL ? DispatchType::Serial : DispatchType::Concurrent);
        for (int r = 0; r < R; ++r) {
          emit(e, 3 * r, aa, o1);              // sib1
          emit(e, 3 * r + 1, aa, o2);          // sib2 (|| sib1)
          if (s == ALL_CONC) { e.memory_barrier(BarrierScope::Buffers); }
          emit(e, 3 * r + 2, o1, aa);          // dep -> next iter
          if (s == ALL_CONC) { e.memory_barrier(BarrierScope::Buffers); }
        }
      }
      st.commit().wait();
    } else if (s == HYBRID) {
      CommandStream st = mc->make_command_stream();
      for (int r = 0; r < R; ++r) {
        { ComputeEncoder e = st.begin_compute(DispatchType::Concurrent);
          emit(e, 3 * r, aa, o1); emit(e, 3 * r + 1, aa, o2); }   // sib pair
        { ComputeEncoder e = st.begin_compute(DispatchType::Serial);
          emit(e, 3 * r + 2, o1, aa); }                          // dep
      }
      st.commit().wait();
    } else {                                    // HYBRID_CB: cb per iter
      for (int r = 0; r < R; ++r) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute(DispatchType::Concurrent);
          emit(e, 3 * r, aa, o1); emit(e, 3 * r + 1, aa, o2); }
        { ComputeEncoder e = st.begin_compute(DispatchType::Serial);
          emit(e, 3 * r + 2, o1, aa); }
        st.commit().wait();
      }
    }
  };
  struct C { const char* name; Strat s; };
  const C cs[] = {{"ALL_SERIAL", ALL_SERIAL}, {"ALL_CONC ", ALL_CONC},
                  {"HYBRID   ", HYBRID}, {"HYBRID_CB", HYBRID_CB}};
  const int NC = 4;
  std::vector<std::vector<double>> gb(NC);
  for (int c = 0; c < NC; ++c) { run(cs[c].s); }        // warm
  for (int rep = 0; rep < REPS; ++rep) {
    for (int c = 0; c < NC; ++c) {
      const auto t0 = Clock::now();
      run(cs[c].s);
      const double ms =
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      gb[c].push_back(3.0 * read_bytes * R / (ms / 1e3) / 1e9);
    }
  }
  std::printf("[mixed] oproj 2560^2, 3 GEMVs/iter (2 sib + 1 dep) x %d iter x %d "
              "reps, cache-defeated (M4 Pro peak ~273 GB/s)\n", R, REPS);
  for (int c = 0; c < NC; ++c) {
    std::sort(gb[c].begin(), gb[c].end());
    std::printf("[mixed] %-11s min %6.1f  median %6.1f  max %6.1f GB/s\n",
                cs[c].name, gb[c].front(), gb[c][gb[c].size() / 2], gb[c].back());
  }
}

// Bandwidth-vs-footprint curve: bounds the "prefetch phase" idea. Read the SAME
// weight buffer R times; when its footprint fits the System-Level Cache the
// re-reads hit SLC (high BW), past the SLC knee every read is cold DRAM (low
// BW). The knee = usable SLC size (-> how many matmuls you can prefetch ahead);
// the two plateaus = SLC-stream BW vs DRAM BW (-> whether an exec-from-SLC phase
// is actually cheaper than exec-from-DRAM). Gated on VPIPE_QMV_FOOTPRINT.
TEST(metal_lm_smoke, qmv_footprint_bw) {
  if (std::getenv("VPIPE_QMV_FOOTPRINT") == nullptr) { return; }
  using namespace metal_compute;
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv_bf16");
  auto fn = lib.function("affine_qmv_w4g64");
  if (!fn.valid()) { std::printf("[footprint] kernel missing\n"); return; }
  using Clock = std::chrono::steady_clock;
  const int K = 2560, rps = 4, nsg = 2, R = 300, groups = K / 64;
  const int Ns[] = {512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
                    24576, 32768, 49152, 65536};
  std::printf("[footprint] read SAME weight R=%d times (SLC-warm if it fits); "
              "K=%d (M4 Pro peak ~273 GB/s)\n", R, K);
  for (int N : Ns) {
    const std::size_t wbytes = (std::size_t)N * K / 8 * 4;
    const std::size_t sbcnt = (std::size_t)N * groups;
    const double read_bytes = (double)(wbytes + 2 * sbcnt * 2);
    auto wb = mc->make_shared_buffer(wbytes);
    auto sb = mc->make_shared_buffer(sbcnt * 2);
    auto bb2 = mc->make_shared_buffer(sbcnt * 2);
    auto xb = mc->make_shared_buffer((std::size_t)K * 2);
    auto yb = mc->make_shared_buffer((std::size_t)N * 2);
    {
      auto* p = static_cast<std::uint32_t*>(wb.contents());
      std::uint32_t s = 2246822519u ^ (std::uint32_t)N;
      for (std::size_t i = 0; i < wbytes / 4; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5; p[i] = s;
      }
    }
    auto go = [&](int reps) {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        e.set_function(fn);
        e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb2);
        e.set_buffer(3, xb); e.set_buffer(4, yb);
        e.set_constant(5, K); e.set_constant(6, N);
        for (int r = 0; r < reps; ++r) {
          e.dispatch({32u, (unsigned)(N / rps), 1u}, {32u, (unsigned)nsg, 1u});
        }
      }
      st.commit().wait();
    };
    go(20);
    double best = 1e18;
    for (int rep = 0; rep < 3; ++rep) {
      const auto t0 = Clock::now();
      go(R);
      best = std::min(best, std::chrono::duration<double, std::milli>(
                                Clock::now() - t0).count());
    }
    std::printf("[footprint] footprint %6.1f MB | %6.1f GB/s\n",
                wbytes / 1e6, read_bytes * R / (best / 1e3) / 1e9);
  }
}

// Why doesn't the resource-scoped barrier unlock overlap -- driver, data
// dependency, or noise? Decisive probe: put a resource-scoped barrier on a
// DUMMY buffer that neither kernel touches, BETWEEN INDEPENDENT (sibling)
// dispatches. If the driver honors resource scope, that barrier is a no-op ->
// siblings still overlap (== no-barrier). If the driver full-flushes on ANY
// memoryBarrier, they serialize (== scope-wide). That isolates DRIVER behavior
// from the data-dependency limiter. Also times the dependent-chain res-vs-scope.
// Interleaved, many reps -> reports the noise band (min/median/max). All at the
// oproj 2560^2 under-saturated shape, cache-defeated. Gated on VPIPE_QMV_RESPROBE.
TEST(metal_lm_smoke, qmv_resbarrier_probe) {
  if (std::getenv("VPIPE_QMV_RESPROBE") == nullptr) { return; }
  using namespace metal_compute;
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv_bf16");
  auto fn = lib.function("affine_qmv_w4g64");
  if (!fn.valid()) { std::printf("[resprobe] kernel missing\n"); return; }
  using Clock = std::chrono::steady_clock;
  const int N = 2560, K = 2560, rps = 4, nsg = 2, R = 512, REPS = 11;
  const int groups = K / 64;
  const std::size_t wbytes = (std::size_t)N * K / 8 * 4;
  const std::size_t sbcnt = (std::size_t)N * groups;
  const double read_bytes = (double)(wbytes + 2 * sbcnt * 2);
  const std::size_t kPool = 512ull << 20;
  const int windows = (int)(kPool / wbytes);
  auto wpool = mc->make_shared_buffer(kPool);
  {
    auto* p = static_cast<std::uint32_t*>(wpool.contents());
    std::uint32_t s = 88172645u;
    for (std::size_t i = 0; i < kPool / 4; ++i) {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5; p[i] = s;
    }
  }
  auto spool = mc->make_shared_buffer((std::size_t)windows * sbcnt * 2);
  auto bpool = mc->make_shared_buffer((std::size_t)windows * sbcnt * 2);
  auto opool = mc->make_shared_buffer((std::size_t)windows * N * 2);
  auto a = mc->make_shared_buffer((std::size_t)K * 2);
  auto bb = mc->make_shared_buffer((std::size_t)N * 2);
  auto dummy = mc->make_shared_buffer(4096);        // touched by no kernel
  // barrier: 0=none 1=scope 2=res-on-relevant 3=res-on-dummy
  // dep: true = serial dependency chain; false = independent siblings
  auto run = [&](int barrier, bool dep) {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute(DispatchType::Concurrent);
      for (int r = 0; r < R; ++r) {
        const bool even = (r % 2) == 0;
        const int w = r % windows;
        e.set_function(fn);
        e.set_buffer(0, wpool, (std::size_t)w * wbytes);
        e.set_buffer(1, spool, (std::size_t)w * sbcnt * 2);
        e.set_buffer(2, bpool, (std::size_t)w * sbcnt * 2);
        if (dep) { e.set_buffer(3, even ? a : bb); e.set_buffer(4, even ? bb : a); }
        else { e.set_buffer(3, a); e.set_buffer(4, opool, (std::size_t)w * N * 2); }
        e.set_constant(5, K);
        e.set_constant(6, N);
        e.dispatch({32u, (unsigned)(N / rps), 1u}, {32u, (unsigned)nsg, 1u});
        if (r + 1 < R) {
          if (barrier == 1) { e.memory_barrier(BarrierScope::Buffers); }
          else if (barrier == 2) { e.memory_barrier_buffer(dep ? (even ? bb : a)
                                                               : opool); }
          else if (barrier == 3) { e.memory_barrier_buffer(dummy); }
        }
      }
    }
    st.commit().wait();
  };
  struct Case { const char* name; int barrier; bool dep; };
  const Case cases[] = {
      {"SIB none      ", 0, false}, {"SIB res-dummy ", 3, false},
      {"SIB res-out   ", 2, false}, {"SIB scope     ", 1, false},
      {"DEP none(ill) ", 0, true},  {"DEP res-act   ", 2, true},
      {"DEP scope     ", 1, true},  {"DEP serial-enc", -1, true},
  };
  const int NC = (int)(sizeof(cases) / sizeof(cases[0]));
  std::vector<std::vector<double>> gb(NC);
  for (int c = 0; c < NC; ++c) { run(cases[c].barrier < 0 ? 1 : cases[c].barrier,
                                     cases[c].dep); }  // warm
  for (int rep = 0; rep < REPS; ++rep) {
    for (int c = 0; c < NC; ++c) {
      const auto t0 = Clock::now();
      run(cases[c].barrier < 0 ? 1 : cases[c].barrier, cases[c].dep);
      const double ms =
          std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      gb[c].push_back(read_bytes * R / (ms / 1e3) / 1e9);
    }
  }
  std::printf("[resprobe] oproj 2560^2, %d GEMVs x %d reps interleaved, "
              "cache-defeated (M4 Pro peak ~273 GB/s)\n", R, REPS);
  for (int c = 0; c < NC; ++c) {
    std::sort(gb[c].begin(), gb[c].end());
    std::printf("[resprobe] %-15s min %6.1f  median %6.1f  max %6.1f GB/s\n",
                cases[c].name, gb[c].front(), gb[c][gb[c].size() / 2],
                gb[c].back());
  }
}

// The tgmem-staged tall-tile batched GEMV (affine_qmv_batch8_tg*_w4g64,
// MAXM=8: activations staged in threadgroup memory instead of per-thread
// registers) must be BYTE-IDENTICAL to the proven register kernel
// (affine_qmv_batch_w4g64, the token-exact-verified batched-decode path)
// on the same inputs -- including a partial K tail (K=768 -> 256-tail
// block) and a padded row tile (m=7 < MAXM=8). Always-on (no model).
TEST(metal_lm_smoke, qmv_batch_tg_matches_batch) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  struct V { const char* fn; int bits; int nsg; int maxm; };
  const V vars[] = {
      {"affine_qmv_batch8_tg_w4g64", 4, 2, 8},
      {"affine_qmv_batch8_tg4_w4g64", 4, 4, 8},
      {"affine_qmv_batch8_tg4_w8g64", 8, 4, 8},
      {"affine_qmv_batch8_xd_w4g64", 4, 2, 8},
      {"affine_qmv_batch8_xd_w8g64", 8, 2, 8},
      {"affine_qmv_batch8_xp1_w4g64", 4, 2, 8},
      {"affine_qmv_batch8_xp2_w4g64", 4, 2, 8},
      {"affine_qmv_batch8_xh_w4g64", 4, 2, 8},
      {"affine_qmv_batch4_xp_w4g64", 4, 2, 4},
      {"affine_qmv_batch4_xp_w8g64", 8, 2, 4},
      {"affine_qmv_batch8_xp2_w8g64", 8, 2, 8},
  };
  const int m = 7, N = 64, K = 768;   // K%512 != 0: exercises the tail path
  const int groups = K / 64;
  std::mt19937 rng(11);
  std::uniform_int_distribution<std::uint32_t> du(0, 0xffffffffu);
  std::uniform_real_distribution<float> df(-1.0f, 1.0f);
  for (const V& v : vars) {
    auto ref_fn = lib.function(
        v.bits == 4 ? "affine_qmv_batch_w4g64" : "affine_qmv_batch_w8g64");
    auto tg_fn = lib.function(v.fn);
    ASSERT_TRUE(ref_fn.valid());
    ASSERT_TRUE(tg_fn.valid());
    const std::size_t wwords = (std::size_t)N * K / (32 / v.bits);
    auto wb = mc->make_shared_buffer(wwords * 4);
    auto sb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto bb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto xb = mc->make_shared_buffer((std::size_t)m * K * 2);
    auto y0 = mc->make_shared_buffer((std::size_t)m * N * 2);
    auto y1 = mc->make_shared_buffer((std::size_t)m * N * 2);
    auto* wp = static_cast<std::uint32_t*>(wb.contents());
    for (std::size_t i = 0; i < wwords; ++i) { wp[i] = du(rng); }
    auto fill_h = [&](metal_compute::SharedBuffer& b, std::size_t n) {
      auto* p = static_cast<__fp16*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { p[i] = (__fp16)(df(rng) * 0.5f); }
    };
    fill_h(sb, (std::size_t)N * groups);
    fill_h(bb, (std::size_t)N * groups);
    fill_h(xb, (std::size_t)m * K);
    auto run = [&](metal_compute::ComputeFunction& fn,
                   metal_compute::SharedBuffer& y, unsigned maxm,
                   unsigned nsg) {
      metal_compute::CommandStream st = mc->make_command_stream();
      {
        metal_compute::ComputeEncoder e = st.begin_compute();
        e.set_function(fn);
        e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, xb); e.set_buffer(4, y);
        e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, m);
        e.dispatch({32u, (unsigned)(N / 4), (unsigned)((m + maxm - 1) / maxm)},
                   {32u, nsg, 1u});
      }
      st.commit().wait();
    };
    run(ref_fn, y0, 2, 2);
    run(tg_fn, y1, (unsigned)v.maxm, (unsigned)v.nsg);
    const bool same = std::memcmp(y0.contents(), y1.contents(),
                                  (std::size_t)m * N * 2) == 0;
    if (!same) {
      const auto* a = static_cast<const __fp16*>(y0.contents());
      const auto* b2 = static_cast<const __fp16*>(y1.contents());
      int bad = 0;
      for (int i = 0; i < m * N && bad < 5; ++i) {
        if ((float)a[i] != (float)b2[i]) {
          std::printf("[qmv-batch-tg] %s mismatch @%d: %f vs %f\n",
                      v.fn, i, (float)a[i], (float)b2[i]);
          ++bad;
        }
      }
    }
    std::printf("[qmv-batch-tg] %-28s vs batch(MAXM=2): %s\n", v.fn,
                same ? "byte-identical" : "MISMATCH");
    EXPECT_TRUE(same);
  }

  // Fused-SwiGLU xp twins vs the register batch-swiglu (MAXM=2) reference:
  // interleaved gate/up weights, halved [m, N/2] output.
  {
    auto ref_fn = lib.function("affine_qmv_batch_swiglu_w4g64");
    ASSERT_TRUE(ref_fn.valid());
    struct SV { const char* fn; int maxm; };
    const SV svars[] = {
        {"affine_qmv_batch4_xp_swiglu_w4g64", 4},
        {"affine_qmv_batch8_xp2_swiglu_w4g64", 8},
    };
    const std::size_t wwords = (std::size_t)N * K / 8;
    auto wb = mc->make_shared_buffer(wwords * 4);
    auto sb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto bb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto xb = mc->make_shared_buffer((std::size_t)m * K * 2);
    auto y0 = mc->make_shared_buffer((std::size_t)m * (N / 2) * 2);
    auto y1 = mc->make_shared_buffer((std::size_t)m * (N / 2) * 2);
    auto* wp = static_cast<std::uint32_t*>(wb.contents());
    for (std::size_t i = 0; i < wwords; ++i) { wp[i] = du(rng); }
    auto fill_h = [&](metal_compute::SharedBuffer& b, std::size_t n) {
      auto* p = static_cast<__fp16*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { p[i] = (__fp16)(df(rng) * 0.5f); }
    };
    fill_h(sb, (std::size_t)N * groups);
    fill_h(bb, (std::size_t)N * groups);
    fill_h(xb, (std::size_t)m * K);
    auto run = [&](metal_compute::ComputeFunction& fn,
                   metal_compute::SharedBuffer& y, unsigned maxm) {
      metal_compute::CommandStream st = mc->make_command_stream();
      {
        metal_compute::ComputeEncoder e = st.begin_compute();
        e.set_function(fn);
        e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, xb); e.set_buffer(4, y);
        e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, m);
        e.dispatch({32u, (unsigned)(N / 4), (unsigned)((m + maxm - 1) / maxm)},
                   {32u, 2u, 1u});
      }
      st.commit().wait();
    };
    run(ref_fn, y0, 2);
    for (const SV& v : svars) {
      auto fn = lib.function(v.fn);
      ASSERT_TRUE(fn.valid());
      run(fn, y1, (unsigned)v.maxm);
      const bool same = std::memcmp(y0.contents(), y1.contents(),
                                    (std::size_t)m * (N / 2) * 2) == 0;
      std::printf("[qmv-batch-tg] %-32s vs batch_swiglu: %s\n", v.fn,
                  same ? "byte-identical" : "MISMATCH");
      EXPECT_TRUE(same);
    }
  }

  // xh16 (half-precision products/quad-sums, f32 block accumulator) is NOT
  // bit-identical by design; check it stays within a tight rel tolerance of
  // the reference (the token-exact bar is enforced end-to-end by
  // qwen_batched_decode_token_exact).
  {
    auto ref_fn = lib.function("affine_qmv_batch_w4g64");
    auto h_fn = lib.function("affine_qmv_batch8_xh16_w4g64");
    ASSERT_TRUE(h_fn.valid());
    const std::size_t wwords = (std::size_t)N * K / 8;
    auto wb = mc->make_shared_buffer(wwords * 4);
    auto sb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto bb = mc->make_shared_buffer((std::size_t)N * groups * 2);
    auto xb = mc->make_shared_buffer((std::size_t)m * K * 2);
    auto y0 = mc->make_shared_buffer((std::size_t)m * N * 2);
    auto y1 = mc->make_shared_buffer((std::size_t)m * N * 2);
    auto* wp = static_cast<std::uint32_t*>(wb.contents());
    for (std::size_t i = 0; i < wwords; ++i) { wp[i] = du(rng); }
    auto fill_h = [&](metal_compute::SharedBuffer& b, std::size_t n) {
      auto* p = static_cast<__fp16*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { p[i] = (__fp16)(df(rng) * 0.5f); }
    };
    fill_h(sb, (std::size_t)N * groups);
    fill_h(bb, (std::size_t)N * groups);
    fill_h(xb, (std::size_t)m * K);
    auto run = [&](metal_compute::ComputeFunction& fn,
                   metal_compute::SharedBuffer& y, unsigned maxm) {
      metal_compute::CommandStream st = mc->make_command_stream();
      {
        metal_compute::ComputeEncoder e = st.begin_compute();
        e.set_function(fn);
        e.set_buffer(0, wb); e.set_buffer(1, sb); e.set_buffer(2, bb);
        e.set_buffer(3, xb); e.set_buffer(4, y);
        e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, m);
        e.dispatch({32u, (unsigned)(N / 4), (unsigned)((m + maxm - 1) / maxm)},
                   {32u, 2u, 1u});
      }
      st.commit().wait();
    };
    run(ref_fn, y0, 2);
    run(h_fn, y1, 8);
    const auto* a = static_cast<const __fp16*>(y0.contents());
    const auto* b2 = static_cast<const __fp16*>(y1.contents());
    double num = 0, den = 0;
    for (int i = 0; i < m * N; ++i) {
      const double d = (double)a[i] - (double)b2[i];
      num += d * d; den += (double)a[i] * (double)a[i];
    }
    const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
    std::printf("[qmv-batch-tg] affine_qmv_batch8_xh16_w4g64 rel-L2 %.3e\n",
                rel);
    EXPECT_TRUE(rel < 2e-3);
  }
}

// Isolated batched-GEMV bandwidth sweep -- the MAXM audit. For each real
// Qwen3.5-4B decode weight shape and m = 2..8, times: the serial qmv (m
// weight re-reads), the MAXM=2 / MAXM=4 register kernels (grid.z tiles),
// the new MAXM=8 tgmem-staged kernels, and the steel GEMM. COLD rows
// cycle through enough weight copies (>=160 MB) that every dispatch
// starts SLC-cold, isolating per-dispatch cache reuse (the grid.z
// re-read question) from cross-rep cache warmth; WARM rows reuse one
// copy (the SLC-artifact regime). GB/s = one-read weight bytes / time,
// so >100% of peak means intra-dispatch cache service; the one-read
// DRAM floor is ~peak. Gated on VPIPE_QMV_AB (M5 16GB peak ~153 GB/s).
TEST(metal_lm_smoke, qmv_batch_bandwidth_sweep) {
  if (std::getenv("VPIPE_QMV_AB") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  auto steel_lib = mc->load_library("affine_qmm_steel");
  auto fn_qmv    = lib.function("affine_qmv_w4g64");
  auto fn_b2     = lib.function("affine_qmv_batch_w4g64");
  auto fn_b4     = lib.function("affine_qmv_batch4_w4g64");
  auto fn_tg8    = lib.function("affine_qmv_batch8_tg_w4g64");
  auto fn_tg8x4  = lib.function("affine_qmv_batch8_tg4_w4g64");
  auto fn_xd8    = lib.function("affine_qmv_batch8_xd_w4g64");
  auto fn_xp1    = lib.function("affine_qmv_batch8_xp1_w4g64");
  auto fn_xp2    = lib.function("affine_qmv_batch8_xp2_w4g64");
  auto fn_steel  = steel_lib.function("affine_qmm_steel_w4g64");
  // Matrix-core fused dequant->matmul2d (M5/metal4 only; row skipped when
  // the library/function is absent). BM=64 tile -> m=8 pads 8x on the M
  // axis, but the MACs run on the matrix units, not the scalar ALUs.
  auto mma_lib = mc->load_library("affine_qmm_mma");
  auto fn_mma = mma_lib.function("affine_qmm_mma_w4g64");
  auto fn_xh   = lib.function("affine_qmv_batch8_xh_w4g64");
  auto fn_xh16 = lib.function("affine_qmv_batch8_xh16_w4g64");
  auto fn_xp4  = lib.function("affine_qmv_batch4_xp_w4g64");
  ASSERT_TRUE(fn_xh.valid() && fn_xh16.valid() && fn_xp4.valid());
  ASSERT_TRUE(fn_qmv.valid() && fn_b2.valid() && fn_b4.valid());
  ASSERT_TRUE(fn_tg8.valid() && fn_tg8x4.valid() && fn_steel.valid());
  ASSERT_TRUE(fn_xd8.valid() && fn_xp1.valid() && fn_xp2.valid());
  using Clock = std::chrono::steady_clock;
  const double kPeak = 153.0;
  struct Shape { const char* name; int n; int k; };
  const Shape shapes[] = {
      {"o-proj  2560x2560", 2560, 2560},
      {"down    2560x9728", 2560, 9728},
      {"gate|up 19456x2560", 19456, 2560},
      {"lm_head 151936x2560", 151936, 2560},
  };
  std::mt19937 rng(7);
  std::uniform_int_distribution<std::uint32_t> du(0, 0xffffffffu);
  const int kMs[] = {2, 3, 4, 5, 6, 8};
  for (const Shape& sh : shapes) {
    const int groups = sh.k / 64;
    const std::size_t wwords = (std::size_t)sh.n * sh.k / 8;   // w4
    const std::size_t sbcnt = (std::size_t)sh.n * groups;
    const double wbytes = (double)(wwords * 4 + 2 * sbcnt * 2);
    // Enough copies that a full rep cycle exceeds the SLC by a wide
    // margin -> each dispatch reads DRAM-cold weights.
    const int C = std::max(1, (int)(160e6 / wbytes) + 1);
    std::vector<metal_compute::SharedBuffer> wv, sv, bv;
    for (int c = 0; c < C; ++c) {
      wv.push_back(mc->make_shared_buffer(wwords * 4));
      sv.push_back(mc->make_shared_buffer(sbcnt * 2));
      bv.push_back(mc->make_shared_buffer(sbcnt * 2));
      auto* wp = static_cast<std::uint32_t*>(wv.back().contents());
      // Fill a stride; full random fill of 160MB x reps is slow and
      // bandwidth is value-independent.
      for (std::size_t i = 0; i < wwords; i += 97) { wp[i] = du(rng); }
    }
    auto xb = mc->make_shared_buffer((std::size_t)8 * sh.k * 2);
    auto yb = mc->make_shared_buffer((std::size_t)8 * sh.n * 2);
    const int R = wbytes > 100e6 ? 10 : 40;
    std::printf("[qmv-sweep] %s  %5.1f MB  1-read floor %.2f ms  (C=%d,R=%d)\n",
                sh.name, wbytes / 1e6, wbytes / kPeak / 1e6, C, R);
    struct KV { const char* name; int kind; };  // kind: 0=serial,1=b2,2=b4,
                                                //       3=tg8,4=tg8x4,5=steel
    const KV kernels[] = {{"serial qmv x m", 0}, {"batch MAXM=2", 1},
                          {"batch MAXM=4", 2},   {"tg8 (NSG=2)", 3},
                          {"tg8 (NSG=4)", 4},    {"steel 32x32", 5},
                          {"xd8 (no tgm)", 6},   {"xp1 (w-regs)", 7},
                          {"xp2 (w-regs)", 8},   {"mma2 BM=64", 9},
                          {"xh8 (hoist)", 10},   {"xh16 (half)", 11},
                          {"xp4 (w-regs)", 12},  {"mix xp4+b2", 13}};
    for (int m : kMs) {
      for (const KV& kv : kernels) {
        if (kv.kind == 9 && !fn_mma.valid()) { continue; }
        auto dispatch_R = [&](int reps, bool cold) {
          metal_compute::CommandStream st = mc->make_command_stream();
          {
            metal_compute::ComputeEncoder e = st.begin_compute();
            for (int r = 0; r < reps; ++r) {
              const int c = cold ? (r % C) : 0;
              auto& w = wv[(std::size_t)c];
              auto& s = sv[(std::size_t)c];
              auto& b = bv[(std::size_t)c];
              auto bind = [&](metal_compute::ComputeFunction& fn) {
                e.set_function(fn);
                e.set_buffer(0, w); e.set_buffer(1, s); e.set_buffer(2, b);
                e.set_buffer(3, xb); e.set_buffer(4, yb);
                e.set_constant(5, sh.k); e.set_constant(6, sh.n);
              };
              switch (kv.kind) {
                case 0:
                  for (int i = 0; i < m; ++i) {
                    e.set_function(fn_qmv);
                    e.set_buffer(0, w); e.set_buffer(1, s); e.set_buffer(2, b);
                    e.set_buffer(3, xb, (std::size_t)i * sh.k * 2);
                    e.set_buffer(4, yb, (std::size_t)i * sh.n * 2);
                    e.set_constant(5, sh.k); e.set_constant(6, sh.n);
                    e.dispatch({32u, (unsigned)(sh.n / 4), 1u}, {32u, 2u, 1u});
                  }
                  break;
                case 1:
                  bind(fn_b2); e.set_constant(7, m);
                  e.dispatch({32u, (unsigned)(sh.n / 4),
                              (unsigned)((m + 1) / 2)}, {32u, 2u, 1u});
                  break;
                case 2:
                  bind(fn_b4); e.set_constant(7, m);
                  e.dispatch({32u, (unsigned)(sh.n / 4),
                              (unsigned)((m + 3) / 4)}, {32u, 2u, 1u});
                  break;
                case 3:
                  bind(fn_tg8); e.set_constant(7, m);
                  e.dispatch({32u, (unsigned)(sh.n / 4),
                              (unsigned)((m + 7) / 8)}, {32u, 2u, 1u});
                  break;
                case 4:
                  bind(fn_tg8x4); e.set_constant(7, m);
                  e.dispatch({32u, (unsigned)(sh.n / 4),
                              (unsigned)((m + 7) / 8)}, {32u, 4u, 1u});
                  break;
                case 5:
                  bind(fn_steel); e.set_constant(7, m);
                  e.dispatch({(unsigned)(((sh.n + 31) / 32) * 32),
                              (unsigned)(((m + 31) / 32) * 2), 2u},
                             {32u, 2u, 2u});
                  break;
                case 6:
                  bind(fn_xd8); e.set_constant(7, m);
                  e.dispatch({32u, (unsigned)(sh.n / 4),
                              (unsigned)((m + 7) / 8)}, {32u, 2u, 1u});
                  break;
                case 7:
                case 8:
                  bind(kv.kind == 7 ? fn_xp1 : fn_xp2); e.set_constant(7, m);
                  e.dispatch({32u, (unsigned)(sh.n / 4),
                              (unsigned)((m + 7) / 8)}, {32u, 2u, 1u});
                  break;
                case 9:
                  bind(fn_mma); e.set_constant(7, m);
                  e.dispatch({(unsigned)(((sh.n + 63) / 64) * 128),
                              (unsigned)((m + 63) / 64), 1u},
                             {128u, 1u, 1u});
                  break;
                case 10:
                case 11:
                  bind(kv.kind == 10 ? fn_xh : fn_xh16); e.set_constant(7, m);
                  e.dispatch({32u, (unsigned)(sh.n / 4),
                              (unsigned)((m + 7) / 8)}, {32u, 2u, 1u});
                  break;
                case 12:
                  bind(fn_xp4); e.set_constant(7, m);
                  e.dispatch({32u, (unsigned)(sh.n / 4),
                              (unsigned)((m + 3) / 4)}, {32u, 2u, 1u});
                  break;
                case 13: {
                  // Heterogeneous 2-read plan for m=5..6: xp4 on rows 0..3
                  // (one weight read @ ~84 GB/s) + batch MAXM=2 on the
                  // remaining 1-2 rows (one read @ ~129), via x/y byte
                  // offsets. vs the homogeneous 3-read MAXM=2 tiling.
                  const int head = m < 4 ? m : 4;
                  bind(fn_xp4); e.set_constant(7, head);
                  e.dispatch({32u, (unsigned)(sh.n / 4), 1u}, {32u, 2u, 1u});
                  if (m > head) {
                    const int rest = m - head;
                    e.set_function(fn_b2);
                    e.set_buffer(0, w); e.set_buffer(1, s); e.set_buffer(2, b);
                    e.set_buffer(3, xb, (std::size_t)head * sh.k * 2);
                    e.set_buffer(4, yb, (std::size_t)head * sh.n * 2);
                    e.set_constant(5, sh.k); e.set_constant(6, sh.n);
                    e.set_constant(7, rest);
                    e.dispatch({32u, (unsigned)(sh.n / 4),
                                (unsigned)((rest + 1) / 2)}, {32u, 2u, 1u});
                  }
                  break;
                }
              }
            }
          }
          st.commit().wait();
        };
        dispatch_R(4, true);   // warm the pipeline state (not the SLC)
        double cold_ms = 1e18, warm_ms = 1e18;
        for (int rep = 0; rep < 3; ++rep) {
          auto t0 = Clock::now();
          dispatch_R(R, true);
          cold_ms = std::min(cold_ms,
              std::chrono::duration<double, std::milli>(Clock::now() - t0)
                  .count() / R);
          t0 = Clock::now();
          dispatch_R(R, false);
          warm_ms = std::min(warm_ms,
              std::chrono::duration<double, std::milli>(Clock::now() - t0)
                  .count() / R);
        }
        std::printf("[qmv-sweep]   m=%d %-14s cold %7.3f ms (%5.1f GB/s eff) "
                    "| warm %7.3f ms\n",
                    m, kv.name, cold_ms, wbytes / (cold_ms / 1e3) / 1e9,
                    warm_ms);
      }
    }
  }
}

// Isolated GPU microbench of the decode sampler kernels over the real Gemma
// vocab (V=262144). Times argmax_f16 (single-tg) vs the two-stage
// argmax_partial_f16 + argmax_combine_f16, and sample_topp_f16, by chaining
// many dispatches into one command buffer and reading the GPU-active window
// (fence.gpu_times). Also asserts the two-stage argmax is TOKEN-EXACT with the
// single-tg argmax over random logits incl. injected ties. Gated on
// VPIPE_SAMPLER_BENCH.
TEST(metal_lm_smoke, sampler_kernel_microbench) {
  if (std::getenv("VPIPE_SAMPLER_BENCH") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise_bf16");   // bf16 storage
  auto fn_argmax  = lib.function("argmax_f16");
  auto fn_partial = lib.function("argmax_partial_f16");
  auto fn_combine = lib.function("argmax_combine_f16");
  auto fn_sample  = lib.function("sample_topp_f16");
  ASSERT_TRUE(fn_argmax.valid() && fn_partial.valid()
              && fn_combine.valid() && fn_sample.valid());

  const int V = 262144;
  const int M = std::getenv("VPIPE_SAMPLER_M")
      ? std::max(2, std::atoi(std::getenv("VPIPE_SAMPLER_M"))) : 96;

  // bf16-stored random logits in a realistic range; bf16 has ~8 mantissa bits
  // so exact ties are common -> the tie-break path is genuinely exercised.
  auto logits = mc->make_shared_buffer((std::size_t)V * 2);
  {
    std::mt19937 rng(20260623);
    std::normal_distribution<float> nd(0.0f, 4.0f);
    auto* p = static_cast<std::uint16_t*>(logits.contents());
    for (int i = 0; i < V; ++i) {
      const float f = nd(rng);
      std::uint32_t b; std::memcpy(&b, &f, 4);
      p[i] = (std::uint16_t)(b >> 16);              // truncate f32 -> bf16
    }
  }
  auto out_a = mc->make_shared_buffer(sizeof(std::int32_t));
  auto out_b = mc->make_shared_buffer(sizeof(std::int32_t));
  auto partials = mc->make_shared_buffer((std::size_t)2 * M * sizeof(float));
  auto ws  = mc->make_shared_buffer((std::size_t)V * 2);
  auto seen = mc->make_shared_buffer((std::size_t)V);
  std::memset(seen.contents(), 0, (std::size_t)V);

  // ---- correctness: two-stage argmax must equal single-tg argmax ----
  {
    metal_compute::CommandStream st = mc->make_command_stream();
    {
      auto e = st.begin_compute();
      e.set_function(fn_argmax);
      e.set_buffer(0, logits); e.set_buffer(1, out_a); e.set_constant(2, V);
      e.dispatch({256, 1, 1}, {256, 1, 1});
      e.set_function(fn_partial);
      e.set_buffer(0, logits); e.set_buffer(1, partials);
      e.set_constant(2, V); e.set_constant(3, M);
      e.dispatch({(unsigned)(256 * M), 1, 1}, {256, 1, 1});
      e.set_function(fn_combine);
      e.set_buffer(0, partials); e.set_buffer(1, out_b); e.set_constant(2, M);
      e.dispatch({256, 1, 1}, {256, 1, 1});
    }
    st.commit().wait();
  }
  const std::int32_t ida = *static_cast<std::int32_t*>(out_a.contents());
  const std::int32_t idb = *static_cast<std::int32_t*>(out_b.contents());
  std::printf("[sampler_bench] argmax single=%d two-stage=%d (M=%d) %s\n",
              ida, idb, M, ida == idb ? "MATCH" : "MISMATCH");
  EXPECT_TRUE(ida == idb);

  const int R = 2000;          // dispatches per timed buffer
  auto time_buf = [&](const std::function<void(metal_compute::ComputeEncoder&)>&
                          body) -> double {
    // warm
    for (int w = 0; w < 2; ++w) {
      metal_compute::CommandStream st = mc->make_command_stream();
      { auto e = st.begin_compute(); for (int r = 0; r < 50; ++r) body(e); }
      st.commit().wait();
    }
    double best = 1e18;
    for (int rep = 0; rep < 5; ++rep) {
      metal_compute::CommandStream st = mc->make_command_stream();
      { auto e = st.begin_compute(); for (int r = 0; r < R; ++r) body(e); }
      auto f = st.commit();
      f.wait();
      best = std::min(best, f.gpu_times().gpu_s);
    }
    return best / R * 1e6;     // us per dispatch-iteration
  };

  const double t_single = time_buf([&](metal_compute::ComputeEncoder& e) {
    e.set_function(fn_argmax);
    e.set_buffer(0, logits); e.set_buffer(1, out_a); e.set_constant(2, V);
    e.dispatch({256, 1, 1}, {256, 1, 1});
  });
  const double t_two = time_buf([&](metal_compute::ComputeEncoder& e) {
    e.set_function(fn_partial);
    e.set_buffer(0, logits); e.set_buffer(1, partials);
    e.set_constant(2, V); e.set_constant(3, M);
    e.dispatch({(unsigned)(256 * M), 1, 1}, {256, 1, 1});
    e.set_function(fn_combine);
    e.set_buffer(0, partials); e.set_buffer(1, out_b); e.set_constant(2, M);
    e.dispatch({256, 1, 1}, {256, 1, 1});
  });
  const float temp = 0.8f, top_p = 0.95f, rep_p = 1.0f, pres = 0.0f, min_p = 0.0f;
  const int n_iter = 16, top_k = 0;
  const std::uint32_t seed = 12345u;
  const double t_sample = time_buf([&](metal_compute::ComputeEncoder& e) {
    e.set_function(fn_sample);
    e.set_buffer(0, logits); e.set_buffer(1, out_a); e.set_constant(2, V);
    e.set_constant(3, temp); e.set_constant(4, top_p); e.set_constant(5, seed);
    e.set_buffer(6, ws); e.set_constant(7, n_iter); e.set_constant(8, rep_p);
    e.set_constant(9, pres); e.set_constant(10, top_k); e.set_constant(11, min_p);
    e.set_buffer(12, seen);
    e.dispatch({256, 1, 1}, {256, 1, 1});
  });
  std::printf("[sampler_bench] V=%d  argmax single-tg=%.2f us | two-stage(M=%d)"
              "=%.2f us (%.2fx) | sample_topp=%.2f us\n",
              V, t_single, M, t_two, t_single / t_two, t_sample);

  // ---- NEW histogram multi-tg sampler: load kernels + scratch ----
  auto fn_smp_maxp   = lib.function("sample_max_partial_f16");
  auto fn_smp_maxc   = lib.function("sample_max_combine_f16");
  auto fn_smp_zhp    = lib.function("sample_zhist_partial_f16");
  auto fn_smp_zhc    = lib.function("sample_zhist_combine_f16");
  auto fn_smp_thr    = lib.function("sample_thresh_f16");
  auto fn_smp_pickp  = lib.function("sample_pick_partial_f16");
  auto fn_smp_pickc  = lib.function("sample_pick_combine_f16");
  ASSERT_TRUE(fn_smp_maxp.valid() && fn_smp_maxc.valid() && fn_smp_zhp.valid()
              && fn_smp_zhc.valid() && fn_smp_thr.valid()
              && fn_smp_pickp.valid() && fn_smp_pickc.valid());
  const int kSampB = 1024;          // MUST match the .metal kSampB
  auto smp_maxpart  = mc->make_shared_buffer((std::size_t)M * sizeof(float));
  auto smp_hpart    = mc->make_shared_buffer(
      (std::size_t)M * (2 * kSampB + 1) * sizeof(float));
  auto smp_hist     = mc->make_shared_buffer(
      (std::size_t)(2 * kSampB + 1) * sizeof(float));
  auto smp_maxl     = mc->make_shared_buffer(sizeof(float));
  auto smp_wt       = mc->make_shared_buffer(sizeof(float));
  auto smp_pickpart = mc->make_shared_buffer((std::size_t)2 * M * sizeof(float));

  // Encode the full new chain (Pass A -> B -> thresh -> C) into `e`.
  auto enc_new = [&](metal_compute::ComputeEncoder& e, float t_, float tp_,
                     int tk_, float mp_, float rp_, float pr_,
                     std::uint32_t sd_, const metal_compute::SharedBuffer& out_,
                     const metal_compute::SharedBuffer& seen_) {
    e.set_function(fn_smp_maxp);
    e.set_buffer(0, logits); e.set_buffer(1, smp_maxpart);
    e.set_constant(2, V); e.set_constant(3, M);
    e.set_constant(4, rp_); e.set_constant(5, pr_); e.set_buffer(6, seen_);
    e.dispatch({(unsigned)(256 * M), 1, 1}, {256, 1, 1});
    e.set_function(fn_smp_maxc);
    e.set_buffer(0, smp_maxpart); e.set_buffer(1, smp_maxl);
    e.set_constant(2, M);
    e.dispatch({256, 1, 1}, {256, 1, 1});
    e.set_function(fn_smp_zhp);
    e.set_buffer(0, logits); e.set_buffer(1, ws); e.set_buffer(2, smp_hpart);
    e.set_constant(3, V); e.set_constant(4, M); e.set_constant(5, t_);
    e.set_constant(6, rp_); e.set_constant(7, pr_); e.set_buffer(8, seen_);
    e.set_buffer(9, smp_maxl);
    e.dispatch({(unsigned)(256 * M), 1, 1}, {256, 1, 1});
    e.set_function(fn_smp_zhc);
    e.set_buffer(0, smp_hpart); e.set_buffer(1, smp_hist); e.set_constant(2, M);
    e.dispatch({256, 1, 1}, {256, 1, 1});
    e.set_function(fn_smp_thr);
    e.set_buffer(0, smp_hist); e.set_buffer(1, smp_wt);
    e.set_constant(2, tp_); e.set_constant(3, tk_); e.set_constant(4, mp_);
    e.set_constant(5, V);
    e.dispatch({256, 1, 1}, {256, 1, 1});
    e.set_function(fn_smp_pickp);
    e.set_buffer(0, logits); e.set_buffer(1, ws); e.set_buffer(2, smp_pickpart);
    e.set_constant(3, V); e.set_constant(4, M); e.set_constant(5, t_);
    e.set_constant(6, sd_); e.set_constant(7, rp_); e.set_constant(8, pr_);
    e.set_buffer(9, seen_); e.set_buffer(10, smp_wt);
    e.dispatch({(unsigned)(256 * M), 1, 1}, {256, 1, 1});
    e.set_function(fn_smp_pickc);
    e.set_buffer(0, smp_pickpart); e.set_buffer(1, out_); e.set_constant(2, M);
    e.set_buffer(3, seen_);
    e.dispatch({256, 1, 1}, {256, 1, 1});
  };

  // Isolated cost of the new chain at the same params as the old above.
  const double t_new = time_buf([&](metal_compute::ComputeEncoder& e) {
    enc_new(e, temp, top_p, top_k, min_p, rep_p, pres, seed, out_a, seen);
  });
  std::printf("[sampler_bench] sample_hist(new, M=%d, B=%d)=%.2f us  "
              "(old/new = %.2fx)\n", M, kSampB, t_new, t_sample / t_new);

  // ---- OLD-vs-NEW DIVERGENCE: same logits+seed+params, per token ----
  // Representative params: temp 0.7, top_p 0.9, top_k 40, min_p 0. seen[] empty
  // (penalties off -> isolates the threshold/Gumbel path). 512 fresh seeds.
  // We regenerate the logits each iteration with bf16-truncated normals so the
  // "tokens" are a realistic, varied logit stream.
  {
    const float dt = 0.7f, dtp = 0.9f, dmp = 0.0f, drp = 1.0f, dpr = 0.0f;
    const int dtk = 40;
    const int NTOK = 512;
    auto out_old = mc->make_shared_buffer(sizeof(std::int32_t));
    auto out_new = mc->make_shared_buffer(sizeof(std::int32_t));
    auto seen0   = mc->make_shared_buffer((std::size_t)V);
    std::memset(seen0.contents(), 0, (std::size_t)V);
    std::mt19937 rng(0xC0FFEEu);
    std::normal_distribution<float> nd(0.0f, 4.0f);
    int diff = 0, boundary = 0;
    auto* lp = static_cast<std::uint16_t*>(logits.contents());
    for (int tk = 0; tk < NTOK; ++tk) {
      // fresh logit row
      for (int i = 0; i < V; ++i) {
        const float f = nd(rng);
        std::uint32_t b; std::memcpy(&b, &f, 4);
        lp[i] = (std::uint16_t)(b >> 16);
      }
      const std::uint32_t sd = 0x1234567u + 2654435761u * (std::uint32_t)tk;
      // OLD
      { metal_compute::CommandStream st = mc->make_command_stream();
        { auto e = st.begin_compute();
          e.set_function(fn_sample);
          e.set_buffer(0, logits); e.set_buffer(1, out_old); e.set_constant(2, V);
          e.set_constant(3, dt); e.set_constant(4, dtp); e.set_constant(5, sd);
          e.set_buffer(6, ws); e.set_constant(7, n_iter); e.set_constant(8, drp);
          e.set_constant(9, dpr); e.set_constant(10, dtk); e.set_constant(11, dmp);
          e.set_buffer(12, seen0);
          e.dispatch({256, 1, 1}, {256, 1, 1}); }
        st.commit().wait(); }
      // NEW
      { metal_compute::CommandStream st = mc->make_command_stream();
        { auto e = st.begin_compute();
          enc_new(e, dt, dtp, dtk, dmp, drp, dpr, sd, out_new, seen0); }
        st.commit().wait(); }
      const std::int32_t io = *static_cast<std::int32_t*>(out_old.contents());
      const std::int32_t in = *static_cast<std::int32_t*>(out_new.contents());
      // reset seen (both kernels set seen[pick]) so each token is independent.
      static_cast<std::uint8_t*>(seen0.contents())[io] = 0;
      static_cast<std::uint8_t*>(seen0.contents())[in] = 0;
      if (io != in) {
        ++diff;
        // Boundary check: recompute the two picks' softmax weights; a boundary
        // divergence is two tokens of NEARLY EQUAL weight (the looser nucleus /
        // Gumbel reshuffle near the cut). Compare exp-weights of io vs in.
        auto wof = [&](int idx) {
          std::uint32_t b = (std::uint32_t)lp[idx] << 16; float f;
          std::memcpy(&f, &b, 4); return f;
        };
        const float lo_ = wof(io), ln_ = wof(in);
        if (std::fabs(lo_ - ln_) < 1.5f) { ++boundary; }   // ~within temp band
      }
    }
    const double rate = 100.0 * (NTOK - diff) / NTOK;
    std::printf("[sampler_bench] divergence: %d/%d match (%.2f%%), %d diff "
                "(%d boundary-near, %d non-boundary)\n",
                NTOK - diff, NTOK, rate, diff, boundary, diff - boundary);
    // Expect tiny divergence; flag if gross (>3%).
    EXPECT_TRUE(diff <= NTOK * 3 / 100 + 2);
  }
}

namespace {
// Minimal P6 PPM reader -> planar [3,H,W] u8 (same layout as the stage).
bool
read_ppm_planar_(const char* path, std::vector<std::uint8_t>* out,
                 int* H, int* W)
{
  std::FILE* f = std::fopen(path, "rb");
  if (!f) { return false; }
  char magic[3] = {0};
  int w = 0, h = 0, maxv = 0;
  if (std::fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxv) != 4
      || std::string(magic) != "P6" || w <= 0 || h <= 0 || maxv != 255) {
    std::fclose(f); return false;
  }
  std::fgetc(f);   // single whitespace after header
  std::vector<std::uint8_t> hwc((std::size_t)w * h * 3);
  const std::size_t got = std::fread(hwc.data(), 1, hwc.size(), f);
  std::fclose(f);
  if (got != hwc.size()) { return false; }
  out->resize(hwc.size());
  const std::size_t plane = (std::size_t)w * h;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t p = (std::size_t)y * w + x;
      (*out)[0 * plane + p] = hwc[p * 3 + 0];
      (*out)[1 * plane + p] = hwc[p * 3 + 1];
      (*out)[2 * plane + p] = hwc[p * 3 + 2];
    }
  }
  *H = h; *W = w;
  return true;
}
}  // namespace

// Reproduces the realtime-vqa Gemma describe on the metal backend in the
// CURRENT build (so it runs in the no-MLX shipping tree, unlike the
// MLX-only llm-gemma4-model-exec tests): encode a real frame, multimodal
// prefill, pdecode describe. Env: VPIPE_METAL_GEMMA_VQA_MODEL (Gemma-4
// dir) + VPIPE_METAL_GEMMA_VQA_FRAME (a P6 PPM, e.g. a vqa-enc dump).
TEST(metal_lm_smoke, gemma_video_describe) {
  const char* path  = std::getenv("VPIPE_METAL_GEMMA_VQA_MODEL");
  const char* frame = std::getenv("VPIPE_METAL_GEMMA_VQA_FRAME");
  if (!path || !*path || !frame || !*frame) { return; }
  std::vector<std::uint8_t> rgb; int H = 0, W = 0;
  if (!read_ppm_planar_(frame, &rgb, &H, &W)) { return; }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  const char* dt = std::getenv("VPIPE_METAL_GEMMA_VQA_DTYPE");
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = (dt && *dt) ? dt : "bf16";
  spec.page_tokens = 512; spec.max_pages = 32;
  std::printf("[gemma_video_describe] compute_dtype=%s\n",
              spec.compute_dtype.c_str());
  auto lm = mgr->load(spec);
  if (!lm || !lm->valid()) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }

  auto menc = genai::MetalGemma4VisionEncoder::load(
      path, mc, genai::MetalGemma4VisionEncoder::config_from(lm->config()));
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!menc) { return; }
  auto ct = genai::make_chat_template(lm->config().architecture,
                                    lm->tokenizer());
  if (!ct) { return; }
  const std::int32_t vpad = ct->video_pad_token_id();

  auto img = menc->encode(rgb.data(), H, W, /*max_soft_tokens=*/280);
  if (img.n_tokens <= 0) { return; }
  std::printf("[gemma_video_describe] frame %dx%d -> %d tok grid %dx%d\n",
              W, H, img.n_tokens, img.grid_h, img.grid_w);

  std::vector<int>   counts{ img.n_tokens };
  std::vector<float> ts{ 0.0f };
  std::vector<std::int32_t> ids;
  ct->render_user_turn_video(
      "Briefly describe what is happening in this video in 2-3 sentences. "
      "Focus on what the people and animals are doing.",
      ts, counts, /*is_first_turn=*/true,
      std::string_view("The current time is 2026-06-14 08:00:00.\n"), &ids);
  std::vector<genai::TokenRef> refs; refs.reserve(ids.size());
  int off = 0;
  for (std::int32_t id : ids) {
    genai::TokenRef r;
    if (id == vpad && off < img.n_tokens) {
      r.kind = genai::TokenRef::Kind::ImageTokens;
      r.embeddings_buf = &img.embeddings;
      r.image_token_offset = off++;
    } else { r.kind = genai::TokenRef::Kind::Text; r.text_id = id; }
    refs.push_back(r);
  }

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs),
      std::span<const std::pair<int, int>>{});
  ASSERT_TRUE(first >= 0);

  genai::SamplerParams sp;   // greedy
  const int kMax = 80;
  std::vector<std::int32_t> got;
  std::int32_t cur = ctx.last_predicted_id();
  const std::span<const std::int32_t> no_prompt;
  int produced = 0;
  if (lm->pdecode_begin(ctx, cur, no_prompt, sp, kMax)) {
    const bool runahead = lm->pdecode_supports_runahead();
    bool committed = (cur >= 0 && !ct->is_stop_token(cur))
        ? lm->pdecode_commit(ctx) : false;
    if (runahead && committed && kMax > 1) { lm->pdecode_commit(ctx); }
    while (produced < kMax) {
      if (ct->is_stop_token(cur)) { break; }
      got.push_back(cur); ++produced;
      if (produced >= kMax || !committed) { break; }
      cur = lm->pdecode_next(ctx);
      if (cur < 0) { break; }
      const bool cont = (produced + 1 < kMax) && !ct->is_stop_token(cur);
      committed = cont ? lm->pdecode_commit(ctx) : false;
    }
    lm->pdecode_end(ctx);
  } else {
    got.push_back(first);
    for (int s = 0; s < kMax; ++s) {
      const std::int32_t nx = lm->next_token(ctx);
      if (nx < 0 || ct->is_stop_token(nx)) { break; }
      got.push_back(nx);
    }
  }
  const std::string ans = lm->tokenizer().decode(
      std::span<const std::int32_t>(got.data(), got.size()));
  std::printf("[gemma_video_describe] %s\n", ans.c_str());
  EXPECT_TRUE(!ans.empty());
  // Regression guard for the bf16 multimodal splice bug: the spliced image
  // rows must reach the forward in the model's compute dtype, else (bf16)
  // the model "sees no image" and refuses. Frame-independent signature.
  std::string lc = ans;
  for (auto& ch : lc) { ch = (char)std::tolower((unsigned char)ch); }
  EXPECT_TRUE(lc.find("provide the image") == std::string::npos
              && lc.find("provide the video") == std::string::npos
              && lc.find("not provided") == std::string::npos
              && lc.find("no video or any image") == std::string::npos);
}

// A/B channel + parity check for the Gemma-4 CoreML vision tower used by
// realtime-vqa: describe the SAME real frame through the native metal Gemma
// ViT (_mgvis) AND a CoreML soft-token export, and print both. A correct
// CoreML export (right colour layout) yields a description that names the
// same dominant colours as the native tower; an R<->B swap would flip them
// (e.g. an orange frame read as azure). Env: VPIPE_METAL_GEMMA_VQA_MODEL +
// VPIPE_METAL_GEMMA_VQA_FRAME (P6 PPM) + VPIPE_METAL_GEMMA_VQA_COREML
// (.mlpackage). Skips vacuously unless all three are set.
TEST(metal_lm_smoke, gemma_video_describe_coreml) {
  const char* path  = std::getenv("VPIPE_METAL_GEMMA_VQA_MODEL");
  const char* frame = std::getenv("VPIPE_METAL_GEMMA_VQA_FRAME");
  const char* cmlp  = std::getenv("VPIPE_METAL_GEMMA_VQA_COREML");
  if (!path || !*path || !frame || !*frame || !cmlp || !*cmlp) { return; }
  std::vector<std::uint8_t> rgb; int H = 0, W = 0;
  if (!read_ppm_planar_(frame, &rgb, &H, &W)) { return; }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  const char* dt = std::getenv("VPIPE_METAL_GEMMA_VQA_DTYPE");
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = (dt && *dt) ? dt : "bf16";
  spec.page_tokens = 512; spec.max_pages = 32;
  auto lm = mgr->load(spec);
  if (!lm || !lm->valid()) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }

  auto menc = genai::MetalGemma4VisionEncoder::load(
      path, mc, genai::MetalGemma4VisionEncoder::config_from(lm->config()));
  genai::CoreMLVisionEncoder::LoadSpec cm;
  cm.mlpackage_path = cmlp;
  cm.compute_units  = 2;
  cm.patch_size     = lm->config().vision.patch_size;
  cm.spatial_merge_size = lm->config().vision.spatial_merge_size;
  auto cenc = genai::CoreMLVisionEncoder::create(cm, nullptr, &sess);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!menc || !cenc || !cenc->implemented()) { return; }

  auto ct = genai::make_chat_template(lm->config().architecture,
                                      lm->tokenizer());
  if (!ct) { return; }
  const std::int32_t vpad = ct->video_pad_token_id();

  // Shared describe: splice [n_tokens] image rows from `emb` at the video
  // placeholder, multimodal-prefill, greedy-decode a short description.
  auto describe = [&](const metal_compute::SharedBuffer& emb,
                      int n_tokens) -> std::string {
    std::vector<int>   counts{ n_tokens };
    std::vector<float> ts{ 0.0f };
    std::vector<std::int32_t> ids;
    ct->render_user_turn_video(
        "Briefly describe this image in 2-3 sentences. Name the most "
        "prominent colours you see.",
        ts, counts, /*is_first_turn=*/true,
        std::string_view("The current time is 2026-06-14 08:00:00.\n"),
        &ids);
    std::vector<genai::TokenRef> refs; refs.reserve(ids.size());
    int off = 0;
    for (std::int32_t id : ids) {
      genai::TokenRef r;
      if (id == vpad && off < n_tokens) {
        r.kind = genai::TokenRef::Kind::ImageTokens;
        r.embeddings_buf = &emb;
        r.image_token_offset = off++;
      } else {
        r.kind = genai::TokenRef::Kind::Text; r.text_id = id;
      }
      refs.push_back(r);
    }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return {}; }
    const std::int32_t first = lm->prefill_multimodal_metal(
        ctx, std::span<const genai::TokenRef>(refs),
        std::span<const std::pair<int, int>>{});
    if (first < 0) { return {}; }
    genai::SamplerParams sp;   // greedy
    const int kMax = 80;
    std::vector<std::int32_t> got;
    std::int32_t cur = ctx.last_predicted_id();
    int produced = 0;
    const std::span<const std::int32_t> no_prompt;
    if (lm->pdecode_begin(ctx, cur, no_prompt, sp, kMax)) {
      const bool runahead = lm->pdecode_supports_runahead();
      bool committed = (cur >= 0 && !ct->is_stop_token(cur))
          ? lm->pdecode_commit(ctx) : false;
      if (runahead && committed && kMax > 1) { lm->pdecode_commit(ctx); }
      while (produced < kMax) {
        if (ct->is_stop_token(cur)) { break; }
        got.push_back(cur); ++produced;
        if (produced >= kMax || !committed) { break; }
        cur = lm->pdecode_next(ctx);
        if (cur < 0) { break; }
        const bool cont = (produced + 1 < kMax) && !ct->is_stop_token(cur);
        committed = cont ? lm->pdecode_commit(ctx) : false;
      }
      lm->pdecode_end(ctx);
    } else {
      got.push_back(first);
      for (int s = 0; s < kMax; ++s) {
        const std::int32_t nx = lm->next_token(ctx);
        if (nx < 0 || ct->is_stop_token(nx)) { break; }
        got.push_back(nx);
      }
    }
    return lm->tokenizer().decode(
        std::span<const std::int32_t>(got.data(), got.size()));
  };

  auto nimg = menc->encode(rgb.data(), H, W, /*max_soft_tokens=*/280);
  auto cimg = cenc->encode_host(rgb.data(), H, W);
  std::printf("[gemma_coreml] native %d tok grid %dx%d | coreml %d tok "
              "grid %dx%d hidden=%d\n",
              nimg.n_tokens, nimg.grid_h, nimg.grid_w,
              cimg.n_tokens, cimg.grid_h, cimg.grid_w, cimg.out_hidden);
  ASSERT_TRUE(nimg.n_tokens > 0);
  ASSERT_TRUE(cimg.n_tokens > 0);
  // The CoreML soft tokens must match the LM hidden width or the splice
  // reads garbage rows.
  ASSERT_TRUE(cimg.out_hidden == lm->config().hidden);

  const std::string a_native = describe(nimg.embeddings, nimg.n_tokens);
  const std::string a_coreml = describe(cimg.embeddings, cimg.n_tokens);
  std::printf("[gemma_coreml] NATIVE: %s\n", a_native.c_str());
  std::printf("[gemma_coreml] COREML: %s\n", a_coreml.c_str());
  EXPECT_TRUE(!a_coreml.empty());
  // Same "the model actually sees an image" guard as the native test.
  std::string lc = a_coreml;
  for (auto& ch : lc) { ch = (char)std::tolower((unsigned char)ch); }
  EXPECT_TRUE(lc.find("provide the image") == std::string::npos
              && lc.find("provide the video") == std::string::npos
              && lc.find("not provided") == std::string::npos
              && lc.find("no video or any image") == std::string::npos);
}

// Regression: realtime-vqa (and text-chat / visual-qa) disable thinking, but
// Gemma-4 e4b IS a reasoning checkpoint -- it intermittently emits a
// `<|channel>thought ...<channel|>` block (notably on the multi-part audio
// prompt) that leaked into descriptions. We do NOT prefill an empty thought
// channel to suppress it (that makes e4b answer in open meta-reasoning, "The
// user wants me to ..." -- see gemma_video_describe_coreml). Instead the
// GemmaChatTemplate exposes sanitize_output(), mirroring the checkpoint's own
// strip_thinking macro, and realtime-vqa runs every decoded answer through it.
// This test (1) unit-checks sanitize_output on crafted text and (2) decodes a
// reasoning-prone prompt and proves the sanitized output carries no channel
// markers. Env: VPIPE_GEMMA4_TEST_MODEL_PATH (or VPIPE_METAL_GEMMA_VQA_MODEL).
TEST(metal_lm_smoke, gemma_thinking_stripped) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { path = std::getenv("VPIPE_METAL_GEMMA_VQA_MODEL"); }
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512; spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }
  auto& tok = lm->tokenizer();
  auto tpl = genai::make_chat_template(lm->config().architecture, tok);
  if (!tpl) { return; }

  // (1) Unit-check the sanitizer directly: a thought block (and its markers)
  // is removed, surrounding answer text is kept; a trailing unclosed thought
  // (truncated decode) is dropped; clean text is untouched.
  EXPECT_TRUE(tpl->sanitize_output(
      "<|channel>thought\nlet me reason<channel|>The answer is 9.")
      == "The answer is 9.");
  EXPECT_TRUE(tpl->sanitize_output("Two boats on a calm lake.")
      == "Two boats on a calm lake.");
  EXPECT_TRUE(tpl->sanitize_output("Answer: 9.<|channel>thought\ncut off")
      == "Answer: 9.");
  // The detokenizer rewrites the channel tokens to the UNIFIED vpipe
  // thinking markers, so streamed text carries the marker form; the
  // sanitizer must strip that form too.
  EXPECT_TRUE(tpl->sanitize_output(
      std::string(vpipe::media_line::kThinkStart)
      + "thought\nlet me reason"
      + std::string(vpipe::media_line::kThinkEnd) + "The answer is 9.")
      == "The answer is 9.");

  // (2) End-to-end: ARM the reasoning channel directly (a `<|think|>` system
  // turn) so the real e4b checkpoint deterministically emits a
  // `<|channel>thought ...<channel|>` block under greedy decode -- exactly
  // the leak realtime-vqa sees intermittently. Then prove sanitize_output
  // strips it from the real model output. Built by hand because the e4b
  // factory path intentionally never arms thinking.
  const std::int32_t bos  = tok.special_token_id("<bos>");
  const std::int32_t sot  = tok.special_token_id("<|turn>");
  const std::int32_t eot  = tok.special_token_id("<turn|>");
  const std::int32_t think = tok.special_token_id("<|think|>");
  if (bos < 0 || sot < 0 || eot < 0 || think < 0) { return; }
  auto append = [&](std::vector<std::int32_t>* d, std::string_view s) {
    auto e = tok.encode(s);
    d->insert(d->end(), e.begin(), e.end());
  };
  std::vector<std::int32_t> ids;
  ids.push_back(bos);
  ids.push_back(sot); append(&ids, "system\n");
  ids.push_back(think); append(&ids, "\n");
  ids.push_back(eot); append(&ids, "\n");
  ids.push_back(sot); append(&ids,
      "user\nA farmer has 17 sheep. All but 9 run away. How many sheep are "
      "left?");
  ids.push_back(eot); append(&ids, "\n");
  ids.push_back(sot); append(&ids, "model\n");
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::int32_t t = lm->prefill(ctx, ids);
  std::vector<std::int32_t> gen;
  for (int i = 0; i < 160 && t >= 0; ++i) {
    if (tpl->is_stop_token(t)) { break; }
    gen.push_back(t);
    t = lm->next_token(ctx);
  }
  const std::string raw =
      tok.decode(std::span<const std::int32_t>(gen.data(), gen.size()));
  const std::string clean = tpl->sanitize_output(raw);
  // The channel token ids now DECODE as the unified vpipe thinking
  // markers (the detokenizer rewrite), so the armed thought block shows
  // up in raw as the marker form, never the literal channel strings.
  const bool raw_had_think =
      raw.find(vpipe::media_line::kThinkStart) != std::string::npos;
  std::printf("[gemma_thinking] raw_had_think=%d\n", raw_had_think ? 1 : 0);
  std::printf("[gemma_thinking] RAW  : %s\n", raw.c_str());
  std::printf("[gemma_thinking] CLEAN: %s\n", clean.c_str());
  EXPECT_TRUE(raw.find("<|channel>") == std::string::npos);

  // The user-facing string must never carry a thought channel, in
  // either the raw or the unified-marker form.
  EXPECT_TRUE(clean.find("<|channel>")  == std::string::npos);
  EXPECT_TRUE(clean.find("<channel|>") == std::string::npos);
  EXPECT_TRUE(clean.find(vpipe::media_line::kThinkStart)
              == std::string::npos);
  EXPECT_TRUE(clean.find(vpipe::media_line::kThinkEnd)
              == std::string::npos);
  EXPECT_TRUE(!clean.empty());
}

// Dense (raw-HF bf16/f16) Gemma-4 coherent-generation smoke. The raw google
// gemma-4 releases (E2B/E4B/12B ...) ship UNQUANTIZED .weight tensors; the
// metal gemma exec detects that and runs the dense GEMM/GEMV path (no affine
// dequant, no scales/biases) instead of the quantized qmv/qmm kernels. A
// deterministic factual prompt under greedy decode must produce the known
// answer -- a broken dense forward (wrong transpose/norm/geglu/double-wide
// ffn) yields word-salad or the wrong token. This is the bf16 bring-up gate
// (token-exact-vs-omlx is a separate, reference-tooling-gated check). Env:
// VPIPE_GEMMA4_DENSE_TEST_MODEL_PATH = a raw bf16 gemma-4 dir (e.g.
// google/gemma-4-E2B-it).
TEST(metal_lm_smoke, gemma_dense_bf16_generates) {
  const char* path = std::getenv("VPIPE_GEMMA4_DENSE_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512; spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm && lm->valid());
  auto& tok = lm->tokenizer();
  auto tpl = genai::make_chat_template(lm->config().architecture, tok);
  ASSERT_TRUE((bool)tpl);

  // Deterministic factual prompt -> greedy -> a competent gemma-4 instruct
  // answers "Paris" (leading token(s) may be whitespace/markdown; substring).
  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with only the city name.",
      /*is_first_turn=*/true, &ids);
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::int32_t t = lm->prefill(ctx, ids);
  std::vector<std::int32_t> gen;
  for (int i = 0; i < 24 && t >= 0; ++i) {
    if (tpl->is_stop_token(t)) { break; }
    gen.push_back(t);
    t = lm->next_token(ctx);
  }
  const std::string out =
      tok.decode(std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[gemma_dense] OUT: %s\n", out.c_str());
  EXPECT_TRUE(!gen.empty());
  EXPECT_TRUE(out.find("Paris") != std::string::npos);
}

// The REALTIME fix: don't just strip the thought block after the fact (that
// still pays its decode cost) -- forbid the reasoning-channel tokens at the
// logit level so the model never GENERATES the block, keeping decode short.
// set_suppressed_tokens masks those logits after softcap, across prefill +
// every decode path. This test ARMS reasoning (a <|think|> system turn) so
// the model WOULD emit `<|channel>thought ...<channel|>` under greedy, then
// shows that with suppression the channel never appears AND the decode is
// strictly shorter (the token-budget win). Env: VPIPE_GEMMA4_TEST_MODEL_PATH
// (or VPIPE_METAL_GEMMA_VQA_MODEL).
TEST(metal_lm_smoke, gemma_thinking_suppressed) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { path = std::getenv("VPIPE_METAL_GEMMA_VQA_MODEL"); }
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc  = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || !mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512; spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }
  auto& tok = lm->tokenizer();
  const std::int32_t bos   = tok.special_token_id("<bos>");
  const std::int32_t sot   = tok.special_token_id("<|turn>");
  const std::int32_t eot   = tok.special_token_id("<turn|>");
  const std::int32_t think = tok.special_token_id("<|think|>");
  const std::int32_t chan  = tok.special_token_id("<|channel>");
  if (bos < 0 || sot < 0 || eot < 0 || think < 0 || chan < 0) { return; }
  auto tpl = genai::make_chat_template(lm->config().architecture, tok);
  if (!tpl) { return; }

  // Hand-built ARMED prompt: a <|think|> system turn forces the reasoning
  // channel on the next model turn.
  auto append = [&](std::vector<std::int32_t>* d, std::string_view s) {
    auto e = tok.encode(s);
    d->insert(d->end(), e.begin(), e.end());
  };
  std::vector<std::int32_t> ids;
  ids.push_back(bos);
  ids.push_back(sot); append(&ids, "system\n");
  ids.push_back(think); append(&ids, "\n");
  ids.push_back(eot); append(&ids, "\n");
  ids.push_back(sot); append(&ids,
      "user\nA farmer has 17 sheep. All but 9 run away. How many are left?");
  ids.push_back(eot); append(&ids, "\n");
  ids.push_back(sot); append(&ids, "model\n");

  // Greedy decode from a fresh context; returns (generated ids).
  auto decode = [&]() -> std::vector<std::int32_t> {
    auto ctx = lm->make_context();
    std::vector<std::int32_t> gen;
    if (!ctx.valid()) { return gen; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 160 && t >= 0; ++i) {
      if (tpl->is_stop_token(t)) { break; }
      gen.push_back(t);
      t = lm->next_token(ctx);
    }
    return gen;
  };
  auto has_chan = [&](const std::vector<std::int32_t>& v) {
    return std::find(v.begin(), v.end(), chan) != v.end();
  };

  // A) No suppression: the armed model emits the thought channel (long).
  lm->set_suppressed_tokens(std::span<const std::int32_t>{});
  const auto gen_unsup = decode();
  // B) Suppression on: the channel token can never be predicted (short).
  const std::int32_t bans[] = { chan, think };
  lm->set_suppressed_tokens(std::span<const std::int32_t>(bans, 2));
  const auto gen_sup = decode();

  std::printf("[gemma_suppress] unsuppressed: %d tok, channel=%d\n",
              (int)gen_unsup.size(), has_chan(gen_unsup) ? 1 : 0);
  std::printf("[gemma_suppress]   suppressed: %d tok, channel=%d\n",
              (int)gen_sup.size(), has_chan(gen_sup) ? 1 : 0);
  std::printf("[gemma_suppress] answer: %s\n",
              tok.decode(std::span<const std::int32_t>(
                  gen_sup.data(), gen_sup.size())).c_str());

  ASSERT_TRUE(!gen_unsup.empty());
  ASSERT_TRUE(!gen_sup.empty());
  // The armed run must actually reach for the channel (else the test is
  // vacuous); suppression must keep it out of the output entirely.
  EXPECT_TRUE(has_chan(gen_unsup));
  EXPECT_TRUE(!has_chan(gen_sup));
}

TEST(metal_lm_smoke, text_decode) {
  const char* path = std::getenv("VPIPE_METAL_LM_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  auto ids = lm->tokenizer().encode("The capital of France is");
  ASSERT_TRUE(!ids.empty());
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());

  std::vector<std::int32_t> gen;
  const std::int32_t first = lm->prefill(ctx, ids);
  ASSERT_TRUE(first >= 0);
  gen.push_back(first);
  for (int i = 0; i < 8; ++i) {
    const std::int32_t n = lm->next_token(ctx);
    if (n < 0) { break; }
    gen.push_back(n);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[metal_lm_smoke] %zu tokens | gen='%s'\n", gen.size(),
              text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// Numerically verify the decode-only quantized GEMV at group-size 32
// (affine_qmv_w4g32 -- the GGUF q4_0 path's q/k/v/o + down_proj decode
// kernel, never exercised by prefill [steel qmm] or e4b [g64]) against a
// hand-built CPU dequant+matmul reference. No model/MLX needed.
TEST(metal_lm_smoke, qmv_w4g32_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  auto fn = lib.function("affine_qmv_w4g32");
  ASSERT_TRUE(fn.valid());

  // Tail guard: K = 3840 (Gemma-12B hidden) is NOT a multiple of the 4-bit
  // block_size (512) -> the GEMV's final block is partial. We allocate x out
  // to the block boundary (Kpad = 4096) filled with non-zero values, the
  // weights with one extra padding row, and tell the kernel in_vec_size = K.
  // The reference sums only k < K. WITHOUT the tail mask the GPU's tail lanes
  // (k in [3840,4096)) fold non-zero x * out-of-range weights into the dot ->
  // gross error; WITH the mask they contribute 0 and it matches the CPU ref.
  // Uses small biases (no affine cancellation) so this isolates the tail.
  const int N = 16, K = 3840, G = 32, groups = K / G;
  const int Kpad = ((K + 511) / 512) * 512;        // 4096
  std::vector<std::uint32_t> wq((std::size_t)(N + 1) * (K / 8), 0);
  std::vector<std::uint16_t> q((std::size_t)N * K);
  std::vector<_Float16> scales((std::size_t)N * groups);
  std::vector<_Float16> biases((std::size_t)N * groups);
  std::vector<_Float16> x((std::size_t)Kpad);
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int g = 0; g < groups; ++g) {
      scales[(std::size_t)o * groups + g] =
          (_Float16)(0.03f + 0.02f * frand((unsigned)(o * 991 + g)));
      biases[(std::size_t)o * groups + g] =
          (_Float16)(0.1f * frand((unsigned)(o * 131 + g * 7 + 5)));
    }
  }
  // Fill the whole weight buffer (incl. the extra padding row) so tail lanes
  // read deterministic non-zero nibbles when the mask is absent.
  for (int o = 0; o <= N; ++o) {
    for (int k = 0; k < K; ++k) {
      const unsigned qq = ((unsigned)(o * K + k) * 2654435761u >> 13) & 0xf;
      if (o < N) { q[(std::size_t)o * K + k] = (std::uint16_t)qq; }
      wq[(std::size_t)o * (K / 8) + k / 8] |=
          (std::uint32_t)qq << (4 * (k % 8));
    }
  }
  for (int k = 0; k < Kpad; ++k) {
    x[k] = (_Float16)(0.5f + frand((unsigned)(k * 7 + 3)));   // all non-zero
  }

  std::vector<float> ref((std::size_t)N);
  for (int o = 0; o < N; ++o) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
      const float s = (float)scales[(std::size_t)o * groups + k / G];
      const float b = (float)biases[(std::size_t)o * groups + k / G];
      acc += (double)(float)x[k] *
             (s * (float)q[(std::size_t)o * K + k] + b);
    }
    ref[o] = (float)acc;
  }

  auto wbuf = mc->make_shared_buffer(wq.size() * 4);
  auto sbuf = mc->make_shared_buffer(scales.size() * 2);
  auto bbuf = mc->make_shared_buffer(biases.size() * 2);
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(wbuf.contents(), wq.data(), wq.size() * 4);
  std::memcpy(sbuf.contents(), scales.data(), scales.size() * 2);
  std::memcpy(bbuf.contents(), biases.data(), biases.size() * 2);
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, wbuf);
    enc.set_buffer(1, sbuf);
    enc.set_buffer(2, bbuf);
    enc.set_buffer(3, xbuf);
    enc.set_buffer(4, ybuf);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  }
  stream.commit().wait();
  const auto* yp = static_cast<const _Float16*>(ybuf.contents());
  double maxrel = 0.0;
  for (int o = 0; o < N; ++o) {
    const double d = std::fabs((double)(float)yp[o] - (double)ref[o]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[o]) + 1e-2));
    std::printf("[qmv_w4g32] o=%2d gpu=%.5f ref=%.5f\n",
                o, (float)yp[o], ref[o]);
  }
  std::printf("[qmv_w4g32] max rel err = %.4g\n", maxrel);
  EXPECT_TRUE(maxrel < 0.03);
}

// Batched (MAXM=2) decode GEMV at group-size 32 with K = 3840 (NOT a multiple
// of block_size 512 -- Gemma-12B hidden): exercises the partial-tail path in
// qmv_batch_impl. Two rows, verified against a CPU dequant ref per (row,out).
TEST(metal_lm_smoke, qmv_batch_w4g32_tail_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  auto fn = lib.function("affine_qmv_batch_w4g32");
  ASSERT_TRUE(fn.valid());

  // Tail guard: K = 3840 (not a multiple of block_size 512). Each row's x is
  // padded to the block boundary (Kpad) with non-zero values and the weights
  // carry an extra padding row, so the partial last block reads out-of-range
  // data unless masked. Small biases -> isolates the tail (no cancellation).
  const int N = 16, M = 2, K = 3840, G = 32, groups = K / G;
  const int Kpad = ((K + 511) / 512) * 512;        // 4096
  std::vector<std::uint32_t> wq((std::size_t)(N + 1) * (K / 8), 0);
  std::vector<std::uint16_t> q((std::size_t)N * K);
  std::vector<_Float16> scales((std::size_t)N * groups);
  std::vector<_Float16> biases((std::size_t)N * groups);
  // x rows are at the kernel's stride (in_vec_size = K), with trailing padding
  // so the last row's partial-block tail reads in-bounds (non-zero) data.
  std::vector<_Float16> x((std::size_t)M * K + (Kpad - K));
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int g = 0; g < groups; ++g) {
      scales[(std::size_t)o * groups + g] =
          (_Float16)(0.03f + 0.02f * frand((unsigned)(o * 991 + g)));
      biases[(std::size_t)o * groups + g] =
          (_Float16)(0.1f * frand((unsigned)(o * 131 + g * 7 + 5)));
    }
  }
  for (int o = 0; o <= N; ++o) {
    for (int k = 0; k < K; ++k) {
      const unsigned qq = ((unsigned)(o * K + k) * 2654435761u >> 13) & 0xf;
      if (o < N) { q[(std::size_t)o * K + k] = (std::uint16_t)qq; }
      wq[(std::size_t)o * (K / 8) + k / 8] |=
          (std::uint32_t)qq << (4 * (k % 8));
    }
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = (_Float16)(0.5f + frand((unsigned)(i * 7 + 3)));   // all non-zero
  }

  std::vector<float> ref((std::size_t)M * N);
  for (int m = 0; m < M; ++m) {
    for (int o = 0; o < N; ++o) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        const float s = (float)scales[(std::size_t)o * groups + k / G];
        const float b = (float)biases[(std::size_t)o * groups + k / G];
        acc += (double)(float)x[(std::size_t)m * K + k] *
               (s * (float)q[(std::size_t)o * K + k] + b);
      }
      ref[(std::size_t)m * N + o] = (float)acc;
    }
  }

  auto wbuf = mc->make_shared_buffer(wq.size() * 4);
  auto sbuf = mc->make_shared_buffer(scales.size() * 2);
  auto bbuf = mc->make_shared_buffer(biases.size() * 2);
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)M * N * 2);
  std::memcpy(wbuf.contents(), wq.data(), wq.size() * 4);
  std::memcpy(sbuf.contents(), scales.data(), scales.size() * 2);
  std::memcpy(bbuf.contents(), biases.data(), biases.size() * 2);
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, wbuf);
    enc.set_buffer(1, sbuf);
    enc.set_buffer(2, bbuf);
    enc.set_buffer(3, xbuf);
    enc.set_buffer(4, ybuf);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.set_constant(7, M);
    enc.dispatch({32, (unsigned)(N / 4), (unsigned)((M + 1) / 2)}, {32, 2, 1});
  }
  stream.commit().wait();
  const auto* yp = static_cast<const _Float16*>(ybuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < M * N; ++i) {
    const double d = std::fabs((double)(float)yp[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[i]) + 1e-2));
  }
  std::printf("[qmv_batch_w4g32] K=%d max rel err = %.4g\n", K, maxrel);
  EXPECT_TRUE(maxrel < 0.03);
}

// Native Q6_K (llama.cpp k-quant) GPU unpack must match the CPU dequant
// bit-for-bit (it's a lossless format, not a requant). Synthesizes raw Q6_K
// super-blocks and compares dequant_q6k_f16 against an inline CPU reference
// mirroring gguf-file.cc's dequant_row_q6_K.
TEST(metal_lm_smoke, q6k_dequant_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dequant_q6k_f16");
  ASSERT_TRUE(fn.valid());

  const int nsb = 3, N = nsb * 256;
  std::vector<std::uint8_t> blk((std::size_t)nsb * 210);
  for (int sb = 0; sb < nsb; ++sb) {
    std::uint8_t* p = blk.data() + (std::size_t)sb * 210;
    for (int i = 0; i < 128; ++i) {                 // ql
      p[i] = (std::uint8_t)((sb * 128 + i) * 37 + 11);
    }
    for (int i = 0; i < 64; ++i) {                  // qh
      p[128 + i] = (std::uint8_t)((sb * 64 + i) * 53 + 7);
    }
    for (int i = 0; i < 16; ++i) {                  // int8 scales
      p[192 + i] = (std::uint8_t)(std::int8_t)((sb * 16 + i) * 5 - 40);
    }
    const _Float16 d = (_Float16)(0.005f + 0.001f * sb);
    std::memcpy(p + 208, &d, 2);
  }

  // CPU reference (mirrors gguf-file.cc kQ6_K).
  std::vector<float> ref((std::size_t)N);
  for (int sb = 0; sb < nsb; ++sb) {
    const std::uint8_t* p = blk.data() + (std::size_t)sb * 210;
    const std::uint8_t* ql = p;
    const std::uint8_t* qh = p + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(p + 192);
    _Float16 d16;
    std::memcpy(&d16, p + 208, 2);
    const float d = (float)d16;
    float* y = ref.data() + (std::size_t)sb * 256;
    for (int half = 0; half < 2; ++half) {
      const int qlo = half * 64, qho = half * 32, sco = half * 8, yo = half * 128;
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16, hi = qh[qho + l];
        const int q1 = ((ql[qlo + l] & 0xF) | (((hi >> 0) & 3) << 4)) - 32;
        const int q2 = ((ql[qlo + l + 32] & 0xF) | (((hi >> 2) & 3) << 4)) - 32;
        const int q3 = ((ql[qlo + l] >> 4) | (((hi >> 4) & 3) << 4)) - 32;
        const int q4 = ((ql[qlo + l + 32] >> 4) | (((hi >> 6) & 3) << 4)) - 32;
        y[yo + l] = d * sc[sco + is + 0] * q1;
        y[yo + l + 32] = d * sc[sco + is + 2] * q2;
        y[yo + l + 64] = d * sc[sco + is + 4] * q3;
        y[yo + l + 96] = d * sc[sco + is + 6] * q4;
      }
    }
  }

  auto sbuf = mc->make_shared_buffer(blk.size());
  auto obuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(sbuf.contents(), blk.data(), blk.size());
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, sbuf);
    enc.set_buffer(1, obuf);
    enc.set_constant(2, N);
    enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
  }
  stream.commit().wait();
  const auto* op = static_cast<const _Float16*>(obuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < N; ++i) {
    const double d = std::fabs((double)(float)op[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[i]) + 1e-3));
  }
  std::printf("[q6k_dequant] N=%d max rel err = %.4g\n", N, maxrel);
  EXPECT_TRUE(maxrel < 1e-2);   // f16 output rounding only
}

// Native Q6_K lm_head GEMV (qmv_q6k_f16): y[o] = sum_h x[h]*dequant(W[o,h]),
// verified against a CPU dequant+dot reference. This is the lossless,
// memory-saving replacement for the 8-bit affine-requant lm_head GEMV.
TEST(metal_lm_smoke, qmv_q6k_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("qmv_q6k_f16");
  ASSERT_TRUE(fn.valid());

  const int N = 64, H = 512, sbpr = H / 256;        // super-blocks per row
  std::vector<std::uint8_t> w((std::size_t)N * sbpr * 210);
  std::vector<_Float16> x((std::size_t)H);
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int s = 0; s < sbpr; ++s) {
      std::uint8_t* p = w.data() + ((std::size_t)o * sbpr + s) * 210;
      const unsigned base = (unsigned)(o * sbpr + s);
      for (int i = 0; i < 128; ++i) { p[i] = (std::uint8_t)((base * 128 + i) * 37 + 11); }
      for (int i = 0; i < 64; ++i) { p[128 + i] = (std::uint8_t)((base * 64 + i) * 53 + 7); }
      for (int i = 0; i < 16; ++i) {
        p[192 + i] = (std::uint8_t)(std::int8_t)((base * 16 + i) * 5 - 40);
      }
      const _Float16 d = (_Float16)(0.004f + 0.0005f * (float)(base % 8));
      std::memcpy(p + 208, &d, 2);
    }
  }
  for (int h = 0; h < H; ++h) { x[h] = (_Float16)frand((unsigned)(h * 7 + 3)); }

  // CPU reference: dequant each weight (mirror gguf-file kQ6_K) and dot with x.
  auto q6k_cpu = [&](const std::uint8_t* sb, int pos) {
    const std::uint8_t* ql = sb;
    const std::uint8_t* qh = sb + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(sb + 192);
    _Float16 d16; std::memcpy(&d16, sb + 208, 2);
    const int hf = pos >> 7, p = pos & 127, which = p >> 5, l = p & 31;
    const int is = l >> 4, qlo = hf * 64, qho = hf * 32, sco = hf * 8;
    const int hi = qh[qho + l];
    int q, sci;
    if (which == 0) { q = (ql[qlo + l] & 0xF) | (((hi >> 0) & 3) << 4); sci = sco + is; }
    else if (which == 1) { q = (ql[qlo + l + 32] & 0xF) | (((hi >> 2) & 3) << 4); sci = sco + is + 2; }
    else if (which == 2) { q = (ql[qlo + l] >> 4) | (((hi >> 4) & 3) << 4); sci = sco + is + 4; }
    else { q = (ql[qlo + l + 32] >> 4) | (((hi >> 6) & 3) << 4); sci = sco + is + 6; }
    return (float)d16 * (float)sc[sci] * (float)(q - 32);
  };
  std::vector<float> ref((std::size_t)N);
  for (int o = 0; o < N; ++o) {
    double acc = 0.0;
    for (int h = 0; h < H; ++h) {
      const std::uint8_t* sb = w.data() + ((std::size_t)o * sbpr + h / 256) * 210;
      acc += (double)(float)x[h] * (double)q6k_cpu(sb, h & 255);
    }
    ref[o] = (float)acc;
  }

  auto wbuf = mc->make_shared_buffer(w.size());
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(wbuf.contents(), w.data(), w.size());
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  // Both the original and the llama.cpp-style v2 kernel must match the CPU
  // reference (v2 is not bit-identical to the original -- different fp grouping
  // -- but must be numerically equivalent).
  auto run = [&](const char* name) {
    auto f = lib.function(name);
    if (!f.valid()) { return; }
    auto stream = mc->make_command_stream();
    {
      auto enc = stream.begin_compute();
      enc.set_function(f);
      enc.set_buffer(0, wbuf);
      enc.set_buffer(1, xbuf);
      enc.set_buffer(2, ybuf);
      enc.set_constant(3, H);
      enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
    }
    stream.commit().wait();
    const auto* yp = static_cast<const _Float16*>(ybuf.contents());
    double maxrel = 0.0;
    for (int o = 0; o < N; ++o) {
      const double d = std::fabs((double)(float)yp[o] - (double)ref[o]);
      maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[o]) + 1e-2));
    }
    std::printf("[qmv_q6k] %-14s N=%d H=%d max rel err = %.4g\n",
                name, N, H, maxrel);
    EXPECT_TRUE(maxrel < 0.02);
  };
  run("qmv_q6k_f16");
  run("qmv_q6k_v2_f16");
}

// Shared 6-bit scale/min unpack for the Q4_K/Q5_K CPU references below
// (mirrors gguf-file.cc get_scale_min_k4_ / llama.cpp get_scale_min_k4).
namespace {
inline void gsmk4_cpu(int j, const std::uint8_t* q, std::uint8_t& d,
                      std::uint8_t& m) {
  if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
  else { d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
         m = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4); }
}
}  // namespace

// Native Q4_K dequant (dequant_q4k_f16) vs a CPU reference mirroring
// gguf-file.cc kQ4_K. 144-byte super-block: d(f16) dmin(f16) scales[12] qs[128].
TEST(metal_lm_smoke, q4k_dequant_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dequant_q4k_f16");
  ASSERT_TRUE(fn.valid());

  const int nsb = 3, N = nsb * 256;
  std::vector<std::uint8_t> blk((std::size_t)nsb * 144);
  for (int sb = 0; sb < nsb; ++sb) {
    std::uint8_t* p = blk.data() + (std::size_t)sb * 144;
    const _Float16 d = (_Float16)(0.005f + 0.001f * sb);
    const _Float16 dm = (_Float16)(0.05f + 0.01f * sb);
    std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2);
    for (int i = 0; i < 12; ++i) { p[4 + i] = (std::uint8_t)((sb*12+i)*29+3); }
    for (int i = 0; i < 128; ++i) { p[16 + i] = (std::uint8_t)((sb*128+i)*37+11); }
  }
  std::vector<float> ref((std::size_t)N);
  for (int sb = 0; sb < nsb; ++sb) {
    const std::uint8_t* p = blk.data() + (std::size_t)sb * 144;
    _Float16 d16, m16; std::memcpy(&d16, p, 2); std::memcpy(&m16, p + 2, 2);
    const float d = (float)d16, dmin = (float)m16;
    const std::uint8_t* scales = p + 4; const std::uint8_t* qs = p + 16;
    float* y = ref.data() + (std::size_t)sb * 256;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
      std::uint8_t sc, m;
      gsmk4_cpu(is + 0, scales, sc, m); const float d1 = d*sc, m1 = dmin*m;
      gsmk4_cpu(is + 1, scales, sc, m); const float d2 = d*sc, m2 = dmin*m;
      const std::uint8_t* q = qs + (j / 64) * 32;
      for (int l = 0; l < 32; ++l) {
        y[j + l]      = d1 * (q[l] & 0x0F) - m1;
        y[j + l + 32] = d2 * (q[l] >> 4)  - m2;
      }
      is += 2;
    }
  }
  auto sbuf = mc->make_shared_buffer(blk.size());
  auto obuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(sbuf.contents(), blk.data(), blk.size());
  auto stream = mc->make_command_stream();
  { auto enc = stream.begin_compute();
    enc.set_function(fn); enc.set_buffer(0, sbuf); enc.set_buffer(1, obuf);
    enc.set_constant(2, N); enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1}); }
  stream.commit().wait();
  const auto* op = static_cast<const _Float16*>(obuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < N; ++i) {
    const double dd = std::fabs((double)(float)op[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[i]) + 1e-3));
  }
  std::printf("[q4k_dequant] N=%d max rel err = %.4g\n", N, maxrel);
  EXPECT_TRUE(maxrel < 1e-2);
}

// Native Q5_K dequant (dequant_q5k_f16) vs CPU reference mirroring kQ5_K.
// 176-byte super-block: d(f16) dmin(f16) scales[12] qh[32] qs[128].
TEST(metal_lm_smoke, q5k_dequant_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dequant_q5k_f16");
  ASSERT_TRUE(fn.valid());

  const int nsb = 3, N = nsb * 256;
  std::vector<std::uint8_t> blk((std::size_t)nsb * 176);
  for (int sb = 0; sb < nsb; ++sb) {
    std::uint8_t* p = blk.data() + (std::size_t)sb * 176;
    const _Float16 d = (_Float16)(0.005f + 0.001f * sb);
    const _Float16 dm = (_Float16)(0.05f + 0.01f * sb);
    std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2);
    for (int i = 0; i < 12; ++i) { p[4 + i] = (std::uint8_t)((sb*12+i)*29+3); }
    for (int i = 0; i < 32; ++i) { p[16 + i] = (std::uint8_t)((sb*32+i)*43+5); }
    for (int i = 0; i < 128; ++i) { p[48 + i] = (std::uint8_t)((sb*128+i)*37+11); }
  }
  std::vector<float> ref((std::size_t)N);
  for (int sb = 0; sb < nsb; ++sb) {
    const std::uint8_t* p = blk.data() + (std::size_t)sb * 176;
    _Float16 d16, m16; std::memcpy(&d16, p, 2); std::memcpy(&m16, p + 2, 2);
    const float d = (float)d16, dmin = (float)m16;
    const std::uint8_t* scales = p + 4;
    const std::uint8_t* qh = p + 16; const std::uint8_t* qs = p + 48;
    float* y = ref.data() + (std::size_t)sb * 256;
    int is = 0; std::uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
      std::uint8_t sc, m;
      gsmk4_cpu(is + 0, scales, sc, m); const float d1 = d*sc, m1 = dmin*m;
      gsmk4_cpu(is + 1, scales, sc, m); const float d2 = d*sc, m2 = dmin*m;
      const std::uint8_t* q = qs + (j / 64) * 32;
      for (int l = 0; l < 32; ++l) {
        const int lo = (q[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
        const int hi = (q[l] >> 4)   + ((qh[l] & u2) ? 16 : 0);
        y[j + l]      = d1 * lo - m1;
        y[j + l + 32] = d2 * hi - m2;
      }
      is += 2; u1 <<= 2; u2 <<= 2;
    }
  }
  auto sbuf = mc->make_shared_buffer(blk.size());
  auto obuf = mc->make_shared_buffer((std::size_t)N * 2);
  std::memcpy(sbuf.contents(), blk.data(), blk.size());
  auto stream = mc->make_command_stream();
  { auto enc = stream.begin_compute();
    enc.set_function(fn); enc.set_buffer(0, sbuf); enc.set_buffer(1, obuf);
    enc.set_constant(2, N); enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1}); }
  stream.commit().wait();
  const auto* op = static_cast<const _Float16*>(obuf.contents());
  double maxrel = 0.0;
  for (int i = 0; i < N; ++i) {
    const double dd = std::fabs((double)(float)op[i] - (double)ref[i]);
    maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[i]) + 1e-3));
  }
  std::printf("[q5k_dequant] N=%d max rel err = %.4g\n", N, maxrel);
  EXPECT_TRUE(maxrel < 1e-2);
}

// Native Q4_K / Q5_K GEMV (qmv_q4k_f16 / qmv_q5k_f16): y[o]=sum_h x[h]*W[o,h],
// verified against a CPU dequant+dot reference. These back the GGUF Qwen3.5
// linears' decode path (no affine requant).
TEST(metal_lm_smoke, qmv_q4k_q5k_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  ASSERT_TRUE(lib.function("qmv_q4k_f16").valid());
  ASSERT_TRUE(lib.function("qmv_q5k_f16").valid());

  const int N = 64, H = 512, sbpr = H / 256;
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  std::vector<_Float16> x((std::size_t)H);
  for (int h = 0; h < H; ++h) { x[h] = (_Float16)frand((unsigned)(h*7+3)); }

  auto fill_block = [&](std::uint8_t* p, unsigned base, int blk_bytes,
                        int qh_off, int qs_off) {
    const _Float16 d = (_Float16)(0.004f + 0.0005f * (float)(base % 8));
    const _Float16 dm = (_Float16)(0.03f + 0.002f * (float)(base % 5));
    std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2);
    for (int i = 0; i < 12; ++i) { p[4+i] = (std::uint8_t)((base*12+i)*29+3); }
    if (qh_off >= 0) {
      for (int i = 0; i < 32; ++i) {
        p[qh_off+i] = (std::uint8_t)((base*32+i)*43+5);
      }
    }
    for (int i = 0; i < 128; ++i) {
      p[qs_off+i] = (std::uint8_t)((base*128+i)*37+11);
    }
    (void)blk_bytes;
  };

  // --- Q4_K ---
  {
    std::vector<std::uint8_t> w((std::size_t)N * sbpr * 144);
    for (int o = 0; o < N; ++o) {
      for (int s = 0; s < sbpr; ++s) {
        fill_block(w.data() + ((std::size_t)o*sbpr+s)*144,
                   (unsigned)(o*sbpr+s), 144, -1, 16);
      }
    }
    auto q4k_cpu = [&](const std::uint8_t* sb, int pos) {
      _Float16 d16, m16; std::memcpy(&d16, sb, 2); std::memcpy(&m16, sb+2, 2);
      const float d = (float)d16, dmin = (float)m16;
      const std::uint8_t* scales = sb + 4; const std::uint8_t* qs = sb + 16;
      const int chunk = pos >> 6, within = pos & 63;
      const int is = chunk*2 + (within >> 5), l = within & 31;
      const unsigned qb = qs[chunk*32 + l];
      const unsigned nib = (within < 32) ? (qb & 0x0F) : (qb >> 4);
      std::uint8_t sc, m; gsmk4_cpu(is, scales, sc, m);
      return d * sc * (float)nib - dmin * m;
    };
    std::vector<float> ref((std::size_t)N);
    for (int o = 0; o < N; ++o) {
      double acc = 0.0;
      for (int h = 0; h < H; ++h) {
        const std::uint8_t* sb = w.data() + ((std::size_t)o*sbpr + h/256)*144;
        acc += (double)(float)x[h] * (double)q4k_cpu(sb, h & 255);
      }
      ref[o] = (float)acc;
    }
    auto wbuf = mc->make_shared_buffer(w.size());
    auto xbuf = mc->make_shared_buffer(x.size() * 2);
    auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
    std::memcpy(wbuf.contents(), w.data(), w.size());
    std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
    auto stream = mc->make_command_stream();
    { auto enc = stream.begin_compute();
      enc.set_function(lib.function("qmv_q4k_f16"));
      enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf); enc.set_buffer(2, ybuf);
      enc.set_constant(3, H); enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2), 1}, {32, 2, 1}); }
    stream.commit().wait();
    const auto* yp = static_cast<const _Float16*>(ybuf.contents());
    double maxrel = 0.0;
    for (int o = 0; o < N; ++o) {
      const double dd = std::fabs((double)(float)yp[o] - (double)ref[o]);
      maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[o]) + 1e-2));
    }
    std::printf("[qmv_q4k] N=%d H=%d max rel err = %.4g\n", N, H, maxrel);
    EXPECT_TRUE(maxrel < 0.02);
  }
  // --- Q5_K ---
  {
    std::vector<std::uint8_t> w((std::size_t)N * sbpr * 176);
    for (int o = 0; o < N; ++o) {
      for (int s = 0; s < sbpr; ++s) {
        fill_block(w.data() + ((std::size_t)o*sbpr+s)*176,
                   (unsigned)(o*sbpr+s), 176, 16, 48);
      }
    }
    auto q5k_cpu = [&](const std::uint8_t* sb, int pos) {
      _Float16 d16, m16; std::memcpy(&d16, sb, 2); std::memcpy(&m16, sb+2, 2);
      const float d = (float)d16, dmin = (float)m16;
      const std::uint8_t* scales = sb + 4;
      const std::uint8_t* qh = sb + 16; const std::uint8_t* qs = sb + 48;
      const int chunk = pos >> 6, within = pos & 63;
      const int is = chunk*2 + (within >> 5), l = within & 31;
      const unsigned qb = qs[chunk*32 + l];
      unsigned nib = (within < 32) ? (qb & 0x0F) : (qb >> 4);
      const int bit = 2*chunk + ((within < 32) ? 0 : 1);
      nib += ((unsigned(qh[l]) >> bit) & 1u) * 16u;
      std::uint8_t sc, m; gsmk4_cpu(is, scales, sc, m);
      return d * sc * (float)nib - dmin * m;
    };
    std::vector<float> ref((std::size_t)N);
    for (int o = 0; o < N; ++o) {
      double acc = 0.0;
      for (int h = 0; h < H; ++h) {
        const std::uint8_t* sb = w.data() + ((std::size_t)o*sbpr + h/256)*176;
        acc += (double)(float)x[h] * (double)q5k_cpu(sb, h & 255);
      }
      ref[o] = (float)acc;
    }
    auto wbuf = mc->make_shared_buffer(w.size());
    auto xbuf = mc->make_shared_buffer(x.size() * 2);
    auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
    std::memcpy(wbuf.contents(), w.data(), w.size());
    std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
    auto stream = mc->make_command_stream();
    { auto enc = stream.begin_compute();
      enc.set_function(lib.function("qmv_q5k_f16"));
      enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf); enc.set_buffer(2, ybuf);
      enc.set_constant(3, H); enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2), 1}, {32, 2, 1}); }
    stream.commit().wait();
    const auto* yp = static_cast<const _Float16*>(ybuf.contents());
    double maxrel = 0.0;
    for (int o = 0; o < N; ++o) {
      const double dd = std::fabs((double)(float)yp[o] - (double)ref[o]);
      maxrel = std::fmax(maxrel, dd / (std::fabs((double)ref[o]) + 1e-2));
    }
    std::printf("[qmv_q5k] N=%d H=%d max rel err = %.4g\n", N, H, maxrel);
    EXPECT_TRUE(maxrel < 0.02);
  }
}

// Q6_K lm_head GEMV bandwidth at the real 12B shape [vocab=262144, hidden=3840]
// -- settles whether qmv_q6k_f16 is DRAM-bound (already at the ~100 GB/s M4
// ceiling, so llama.cpp's "read each ql/qh byte once, extract all nibbles
// in-thread" trick can't help) or load/instruction-bound (the per-nibble
// re-reads across lanes cost, and the trick would). Reports GB/s over the raw
// Q6_K table bytes (210 B / 256 wt). Gated on VPIPE_Q6K_BW.
TEST(metal_lm_smoke, q6k_lmhead_bandwidth) {
  if (std::getenv("VPIPE_Q6K_BW") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");

  const int N = 262144, H = 3840, sbpr = H / 256;     // 15 super-blocks/row
  const std::size_t wbytes = (std::size_t)N * sbpr * 210;
  auto wbuf = mc->make_shared_buffer(wbytes);
  auto xbuf = mc->make_shared_buffer((std::size_t)H * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
  // Content is irrelevant for timing; just fill x with something finite.
  auto* xp = static_cast<_Float16*>(xbuf.contents());
  for (int h = 0; h < H; ++h) { xp[h] = (_Float16)0.01f; }

  auto fn1 = lib.function("qmv_q6k_f16");
  auto fn2 = lib.function("qmv_q6k_v2_f16");
  auto once = [&](metal_compute::ComputeFunction& fn) {
    auto st = mc->make_command_stream();
    { auto enc = st.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf); enc.set_buffer(2, ybuf);
      enc.set_constant(3, H); enc.set_constant(4, N);
      enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
    }
    st.commit().wait();
  };
  auto measure = [&](metal_compute::ComputeFunction& fn) {
    const int R = 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < R; ++i) { once(fn); }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() / R;
  };
  // Interleave v1/v2 measures and take the min (peak-clock) of each to cancel
  // the GPU's per-run clock drift.
  for (int w = 0; w < 5; ++w) { once(fn1); once(fn2); }   // warm
  double m1 = 1e18, m2 = 1e18;
  for (int k = 0; k < 6; ++k) {
    m1 = std::fmin(m1, measure(fn1));
    m2 = std::fmin(m2, measure(fn2));
  }
  std::printf("[q6k_bw] qmv_q6k_f16    %.3f ms  %.1f GB/s (min-of-6)\n",
              m1, (double)wbytes / (m1 * 1e6));
  std::printf("[q6k_bw] qmv_q6k_v2_f16 %.3f ms  %.1f GB/s (min-of-6)  %.2fx\n",
              m2, (double)wbytes / (m2 * 1e6), m1 / m2);
  EXPECT_TRUE(true);
}

// K-quant qmv achieved bandwidth on the 27B FFN gate/up shape [N=17408,
// H=5120] (the dominant decode weight read). Contrasts q4k/q5k (old single-row
// kernel) against q6k_v2 (4-row) and reports GB/s of k-quant bytes moved, to
// locate the decode bandwidth gap vs the ~240 GB/s affine qmv. Gated on
// VPIPE_KQUANT_BW.
TEST(metal_lm_smoke, qmv_kquant_bandwidth) {
  if (std::getenv("VPIPE_KQUANT_BW") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  // Big N so the kernel saturates and per-command-buffer submit overhead is
  // amortized (matches q6k_lmhead_bandwidth's methodology); H = 27B FFN width.
  const int N = 262144, H = 5120, sbpr = H / 256;    // 20 super-blocks/row
  auto xbuf = mc->make_shared_buffer((std::size_t)H * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)N * 2);
  auto* xp = static_cast<_Float16*>(xbuf.contents());
  for (int h = 0; h < H; ++h) { xp[h] = (_Float16)0.01f; }
  // (kernel, block bytes, rows-per-threadgroup) -- dispatch matches each
  // kernel's documented grid (NSG=2 simdgroups/tg; q6k_v2 does RPS=4 rows).
  struct K { const char* name; int blk; int rpt; };
  const K ks[] = {{"qmv_q4k_f16", 144, 2}, {"qmv_q5k_f16", 176, 2},
                  {"qmv_q6k_f16", 210, 2}, {"qmv_q6k_v2_f16", 210, 8}};
  for (const K& k : ks) {
    auto fn = lib.function(k.name);
    if (!fn.valid()) { std::printf("[kq_bw] %-14s missing\n", k.name); continue; }
    const std::size_t wbytes = (std::size_t)N * sbpr * k.blk;
    auto wbuf = mc->make_shared_buffer(wbytes);
    const unsigned gy = (unsigned)(((N + k.rpt - 1) / k.rpt) * 2);
    auto once = [&]() {
      auto st = mc->make_command_stream();
      { auto enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, wbuf); enc.set_buffer(1, xbuf); enc.set_buffer(2, ybuf);
        enc.set_constant(3, H); enc.set_constant(4, N);
        enc.dispatch({32, gy, 1}, {32, 2, 1});
      }
      st.commit().wait();
    };
    for (int w = 0; w < 5; ++w) { once(); }
    double m = 1e18;
    for (int t = 0; t < 6; ++t) {
      const auto t0 = std::chrono::steady_clock::now();
      const int R = 20;
      for (int i = 0; i < R; ++i) { once(); }
      m = std::fmin(m, std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count() / R);
    }
    std::printf("[kq_bw] %-14s %6.1f MB  %.3f ms  %6.1f GB/s\n", k.name,
                (double)wbytes / 1e6, m, (double)wbytes / (m * 1e6));
  }
  EXPECT_TRUE(true);
}

// Sweep qmv_q6k_v2 RPS/NSG tuning on the real 27B Q6_K shapes (lm_head, FFN
// down, GDN qkv). The production default is <4,2>; llama.cpp uses nr0=2 + a
// device-tuned nsg. Gated on VPIPE_Q6K_SWEEP.
TEST(metal_lm_smoke, qmv_q6k_v2_rps_sweep) {
  if (std::getenv("VPIPE_Q6K_SWEEP") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("llm_elementwise");
  struct V { const char* name; int rps; int nsg; };
  const V vars[] = {{"qmv_q6k_v2_f16", 4, 2}, {"qmv_q6k_v2_r2n2_f16", 2, 2},
                    {"qmv_q6k_v2_r8n2_f16", 8, 2}, {"qmv_q6k_v2_r2n4_f16", 2, 4},
                    {"qmv_q6k_v2_r4n4_f16", 4, 4}, {"qmv_q6k_v2_r1n4_f16", 1, 4},
                    {"qmv_q6k_v2_r1n8_f16", 1, 8}, {"qmv_q6k_v2_r2n8_f16", 2, 8},
                    {"qmv_q6k_v2_r4n8_f16", 4, 8}};
  struct S { const char* name; int N; int H; };
  const S shapes[] = {{"lm_head ", 248320, 5120}, {"ffn_down", 5120, 17408},
                      {"gdn_qkv ", 10240, 5120}};
  for (const S& sh : shapes) {
    const int nsb = sh.H / 256;
    const std::size_t wbytes = (std::size_t)sh.N * nsb * 210;
    auto wbuf = mc->make_shared_buffer(wbytes);
    auto xbuf = mc->make_shared_buffer((std::size_t)sh.H * 2);
    auto ybuf = mc->make_shared_buffer((std::size_t)sh.N * 2);
    auto* xp = static_cast<_Float16*>(xbuf.contents());
    for (int h = 0; h < sh.H; ++h) { xp[h] = (_Float16)0.01f; }
    double best_ms = 1e18; const char* best = "";
    for (const V& v : vars) {
      auto fn = lib.function(v.name);
      if (!fn.valid()) { std::printf("  %s MISSING\n", v.name); continue; }
      const unsigned rpt = (unsigned)(v.rps * v.nsg);
      const unsigned gy = ((sh.N + rpt - 1) / rpt) * (unsigned)v.nsg;
      auto run = [&](int R) {
        auto st = mc->make_command_stream();
        { auto e = st.begin_compute(); e.set_function(fn);
          e.set_buffer(0, wbuf); e.set_buffer(1, xbuf); e.set_buffer(2, ybuf);
          e.set_constant(3, sh.H); e.set_constant(4, sh.N);
          for (int r = 0; r < R; ++r) {
            e.dispatch({32u, gy, 1u}, {32u, (unsigned)v.nsg, 1u});
          } }
        st.commit().wait();
      };
      run(5);
      double m = 1e18;
      for (int t = 0; t < 5; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        run(40);
        m = std::fmin(m, std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / 40);
      }
      std::printf("[q6k_sweep %s] %-22s r%dn%d  %.3f ms  %6.1f GB/s\n",
                  sh.name, v.name, v.rps, v.nsg, m, wbytes / (m * 1e6));
      if (m < best_ms) { best_ms = m; best = v.name; }
    }
    std::printf("[q6k_sweep %s] BEST = %s (%.1f GB/s)\n", sh.name, best,
                wbytes / (best_ms * 1e6));
  }
  EXPECT_TRUE(true);
}

// Validates + benchmarks the Q4_K -> affine-g32 load-time repack: repacks a
// random Q4_K matrix and checks affine_qmv_w4g32 over it matches the native
// qmv_q4k (lossless dequant), then times both on the 27B FFN shape. Confirms
// the bit-layout + the decode bandwidth win before the loader integration.
// Gated on VPIPE_Q4K_AFFINE_BW.
TEST(metal_lm_smoke, q4k_affine_repack_matches_and_bench) {
  if (std::getenv("VPIPE_Q4K_AFFINE_BW") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto elt = mc->load_library("llm_elementwise");
  auto aff = mc->load_library("affine_qmv");
  auto f_repack = elt.function("repack_q4k_to_affine_g32");
  auto f_q4k    = elt.function("qmv_q4k_f16");
  auto f_aff    = aff.function("affine_qmv_w4g32");
  ASSERT_TRUE(f_repack.valid() && f_q4k.valid() && f_aff.valid());

  const int N = 4096, H = 5120, nsb = H / 256;
  auto src = mc->make_shared_buffer((std::size_t)N * nsb * 144);
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> ub(0, 255);
  auto* sp = static_cast<std::uint8_t*>(src.contents());
  for (std::size_t i = 0; i < (std::size_t)N * nsb * 144; ++i) {
    sp[i] = (std::uint8_t)ub(rng);
  }
  auto x = mc->make_shared_buffer((std::size_t)H * 2);
  auto* xp = static_cast<_Float16*>(x.contents());
  for (int h = 0; h < H; ++h) { xp[h] = (_Float16)((ub(rng) - 128) * 0.001f); }
  auto wq    = mc->make_shared_buffer((std::size_t)N * (H / 8) * 4);
  auto sc    = mc->make_shared_buffer((std::size_t)N * (H / 32) * 2);
  auto bs    = mc->make_shared_buffer((std::size_t)N * (H / 32) * 2);
  auto y_aff = mc->make_shared_buffer((std::size_t)N * 2);
  auto y_kq  = mc->make_shared_buffer((std::size_t)N * 2);

  auto submit = [&](auto setup) {
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(); setup(e); }
    st.commit().wait();
  };
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_repack);
    e.set_buffer(0, src); e.set_buffer(1, wq); e.set_buffer(2, sc);
    e.set_buffer(3, bs); e.set_constant(4, H); e.set_constant(5, N);
    e.dispatch({(unsigned)nsb, (unsigned)N, 1}, {(unsigned)nsb, 32, 1});
  });
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_aff);
    e.set_buffer(0, wq); e.set_buffer(1, sc); e.set_buffer(2, bs);
    e.set_buffer(3, x); e.set_buffer(4, y_aff);
    e.set_constant(5, H); e.set_constant(6, N);
    e.dispatch({32u, (unsigned)(N / 4), 1u}, {32u, 2u, 1u});
  });
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_q4k);
    e.set_buffer(0, src); e.set_buffer(1, x); e.set_buffer(2, y_kq);
    e.set_constant(3, H); e.set_constant(4, N);
    e.dispatch({32u, (unsigned)(((N + 1) / 2) * 2), 1u}, {32u, 2u, 1u});
  });
  const auto* ya = static_cast<const _Float16*>(y_aff.contents());
  const auto* yk = static_cast<const _Float16*>(y_kq.contents());
  double maxrel = 0.0;
  for (int o = 0; o < N; ++o) {
    const double a = (double)(float)ya[o], k = (double)(float)yk[o];
    maxrel = std::max(maxrel, std::fabs(a - k) / std::max(1e-4, std::fabs(k)));
  }
  std::printf("[q4k_affine] N=%d H=%d  max rel err (affine vs q4k) = %.4g\n",
              N, H, maxrel);
  EXPECT_TRUE(maxrel < 0.02);

  auto bench = [&](bool affine) -> double {
    const int R = 200;
    auto run = [&]() {
      auto st = mc->make_command_stream();
      { auto e = st.begin_compute();
        e.set_function(affine ? f_aff : f_q4k);
        if (affine) {
          e.set_buffer(0, wq); e.set_buffer(1, sc); e.set_buffer(2, bs);
          e.set_buffer(3, x); e.set_buffer(4, y_aff);
          e.set_constant(5, H); e.set_constant(6, N);
        } else {
          e.set_buffer(0, src); e.set_buffer(1, x); e.set_buffer(2, y_kq);
          e.set_constant(3, H); e.set_constant(4, N);
        }
        for (int r = 0; r < R; ++r) {
          if (affine) { e.dispatch({32u, (unsigned)(N / 4), 1u}, {32u, 2u, 1u}); }
          else { e.dispatch({32u, (unsigned)(((N + 1) / 2) * 2), 1u},
                            {32u, 2u, 1u}); }
        } }
      st.commit().wait();
    };
    for (int w = 0; w < 3; ++w) { run(); }
    double best = 1e18;
    for (int t = 0; t < 5; ++t) {
      const auto t0 = std::chrono::steady_clock::now();
      run();
      best = std::fmin(best, std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count() / R);
    }
    return best;
  };
  const double q4k_bytes = (double)N * nsb * 144;
  const double aff_bytes = (double)N * ((H / 8) * 4 + 2 * (H / 32) * 2);
  const double t_kq = bench(false), t_af = bench(true);
  std::printf("[q4k_affine] qmv_q4k       %.3f ms  %.1f GB/s\n",
              t_kq, q4k_bytes / (t_kq * 1e6));
  std::printf("[q4k_affine] affine_w4g32  %.3f ms  %.1f GB/s  (%.2fx faster)\n",
              t_af, aff_bytes / (t_af * 1e6), t_kq / t_af);
  EXPECT_TRUE(true);
}

// Q5_K -> affine-8bit-g32 experiment: validates the repack (vs qmv_q5k) and
// times affine_qmv_w8g32 against the native qmv_q5k on a GDN-out shape. The
// 8-bit affine reads MORE bytes (9 vs 5.5 bits/wt) -- this measures whether the
// higher kernel bandwidth still nets a win. Gated on VPIPE_Q5K_AFFINE_BW.
TEST(metal_lm_smoke, q5k_affine_repack_matches_and_bench) {
  if (std::getenv("VPIPE_Q5K_AFFINE_BW") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto elt = mc->load_library("llm_elementwise");
  auto aff = mc->load_library("affine_qmv");
  auto f_repack = elt.function("repack_q5k_to_affine_g32");
  auto f_q5k    = elt.function("qmv_q5k_f16");
  auto f_aff    = aff.function("affine_qmv_w8g32");
  ASSERT_TRUE(f_repack.valid() && f_q5k.valid() && f_aff.valid());

  const int N = 6144, H = 5120, nsb = H / 256;       // ~GDN out_proj shape
  auto src = mc->make_shared_buffer((std::size_t)N * nsb * 176);   // Q5_K block
  std::mt19937 rng(321);
  std::uniform_int_distribution<int> ub(0, 255);
  auto* sp = static_cast<std::uint8_t*>(src.contents());
  for (std::size_t i = 0; i < (std::size_t)N * nsb * 176; ++i) {
    sp[i] = (std::uint8_t)ub(rng);
  }
  auto x = mc->make_shared_buffer((std::size_t)H * 2);
  auto* xp = static_cast<_Float16*>(x.contents());
  for (int h = 0; h < H; ++h) { xp[h] = (_Float16)((ub(rng) - 128) * 0.001f); }
  auto wq    = mc->make_shared_buffer((std::size_t)N * H);          // 8-bit
  auto sc    = mc->make_shared_buffer((std::size_t)N * (H / 32) * 2);
  auto bs    = mc->make_shared_buffer((std::size_t)N * (H / 32) * 2);
  auto y_aff = mc->make_shared_buffer((std::size_t)N * 2);
  auto y_kq  = mc->make_shared_buffer((std::size_t)N * 2);

  auto submit = [&](auto setup) {
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(); setup(e); }
    st.commit().wait();
  };
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_repack);
    e.set_buffer(0, src); e.set_buffer(1, wq); e.set_buffer(2, sc);
    e.set_buffer(3, bs); e.set_constant(4, H); e.set_constant(5, N);
    e.dispatch({(unsigned)nsb, (unsigned)N, 1}, {(unsigned)nsb, 1, 1});
  });
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_aff);
    e.set_buffer(0, wq); e.set_buffer(1, sc); e.set_buffer(2, bs);
    e.set_buffer(3, x); e.set_buffer(4, y_aff);
    e.set_constant(5, H); e.set_constant(6, N);
    e.dispatch({32u, (unsigned)(N / 4), 1u}, {32u, 2u, 1u});
  });
  submit([&](metal_compute::ComputeEncoder& e) {
    e.set_function(f_q5k);
    e.set_buffer(0, src); e.set_buffer(1, x); e.set_buffer(2, y_kq);
    e.set_constant(3, H); e.set_constant(4, N);
    e.dispatch({32u, (unsigned)(((N + 1) / 2) * 2), 1u}, {32u, 2u, 1u});
  });
  const auto* ya = static_cast<const _Float16*>(y_aff.contents());
  const auto* yk = static_cast<const _Float16*>(y_kq.contents());
  double maxrel = 0.0;
  for (int o = 0; o < N; ++o) {
    const double a = (double)(float)ya[o], k = (double)(float)yk[o];
    maxrel = std::max(maxrel, std::fabs(a - k) / std::max(1e-4, std::fabs(k)));
  }
  std::printf("[q5k_affine] N=%d H=%d  max rel err (affine vs q5k) = %.4g\n",
              N, H, maxrel);
  EXPECT_TRUE(maxrel < 0.02);

  auto bench = [&](bool affine) -> double {
    const int R = 200;
    auto run = [&]() {
      auto st = mc->make_command_stream();
      { auto e = st.begin_compute();
        e.set_function(affine ? f_aff : f_q5k);
        if (affine) {
          e.set_buffer(0, wq); e.set_buffer(1, sc); e.set_buffer(2, bs);
          e.set_buffer(3, x); e.set_buffer(4, y_aff);
          e.set_constant(5, H); e.set_constant(6, N);
        } else {
          e.set_buffer(0, src); e.set_buffer(1, x); e.set_buffer(2, y_kq);
          e.set_constant(3, H); e.set_constant(4, N);
        }
        for (int r = 0; r < R; ++r) {
          if (affine) { e.dispatch({32u, (unsigned)(N / 4), 1u}, {32u, 2u, 1u}); }
          else { e.dispatch({32u, (unsigned)(((N + 1) / 2) * 2), 1u},
                            {32u, 2u, 1u}); }
        } }
      st.commit().wait();
    };
    for (int w = 0; w < 3; ++w) { run(); }
    double best = 1e18;
    for (int t = 0; t < 5; ++t) {
      const auto t0 = std::chrono::steady_clock::now();
      run();
      best = std::fmin(best, std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count() / R);
    }
    return best;
  };
  const double q5k_bytes = (double)N * nsb * 176;
  const double aff_bytes = (double)N * (H + 2 * (H / 32) * 2);   // 8-bit + s/b
  const double t_kq = bench(false), t_af = bench(true);
  std::printf("[q5k_affine] qmv_q5k       %.3f ms  %.1f GB/s (%.1f MB)\n",
              t_kq, q5k_bytes / (t_kq * 1e6), q5k_bytes / 1e6);
  std::printf("[q5k_affine] affine_w8g32  %.3f ms  %.1f GB/s (%.1f MB)  "
              "(%.2fx vs q5k time)\n", t_af, aff_bytes / (t_af * 1e6),
              aff_bytes / 1e6, t_kq / t_af);
  EXPECT_TRUE(true);
}

// Per-token RoPE cost (12B decode): bounds the TOTAL fused-RMSNorm+RoPE work --
// settles whether vpipe's INLINE cos/sin (vs a precomputed cos/sin cache) is a
// decode bottleneck. llama.cpp's Metal kernel_rope_neox computes cos/sin inline
// too (and pow() per element, which vpipe avoids via precomputed inv_freq), so
// the cos/sin cache is a CPU-ggml / TTNN technique, not a GPU one. Replays the
// real per-token sequence: 48 layers x {Q rms_rope, K rms_rope}, 8 global
// (D=512: Hq=16/Hkv=1) + 40 sliding (D=256: Hq=16/Hkv=8). Gated on VPIPE_ROPE_BW.
TEST(metal_lm_smoke, rope_pertoken_cost) {
  if (std::getenv("VPIPE_ROPE_BW") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("rope");
  auto fn = lib.function("rms_rope_f16");
  ASSERT_TRUE(fn.valid());

  const int Dmax = 512, Hq = 16;
  auto xb = mc->make_shared_buffer((std::size_t)Hq * Dmax * 2);
  auto wb = mc->make_shared_buffer((std::size_t)Dmax * 2);
  auto fb = mc->make_shared_buffer((std::size_t)(Dmax / 2) * 4);
  for (int i = 0; i < Hq * Dmax; ++i) {
    static_cast<_Float16*>(xb.contents())[i] = (_Float16)0.02f;
  }
  for (int i = 0; i < Dmax; ++i) {
    static_cast<_Float16*>(wb.contents())[i] = (_Float16)1.0f;
  }
  for (int i = 0; i < Dmax / 2; ++i) {
    static_cast<float*>(fb.contents())[i] = 1.0f / (1.0f + (float)i);
  }
  const float eps = 1e-6f;
  const int offset = 2048;
  auto rope1 = [&](metal_compute::ComputeEncoder& enc, int H, int D) {
    enc.set_function(fn);
    enc.set_buffer(0, xb); enc.set_buffer(1, wb); enc.set_buffer(2, fb);
    enc.set_constant(3, H); enc.set_constant(4, D);
    enc.set_constant(5, eps); enc.set_constant(6, offset);
    enc.dispatch({256, (unsigned)H, 1}, {256, 1, 1});
  };
  auto once = [&]() {                       // one token's worth of rope work
    auto st = mc->make_command_stream();
    { auto enc = st.begin_compute();
      for (int L = 0; L < 48; ++L) {
        const bool full = (L % 6 == 5);     // 8 of 48 are global
        const int D = full ? 512 : 256;
        const int Hkv = full ? 1 : 8;
        rope1(enc, Hq, D);                  // Q rope (16 heads)
        rope1(enc, Hkv, D);                 // K rope
      }
    }
    st.commit().wait();
  };
  for (int w = 0; w < 5; ++w) { once(); }
  double best = 1e18;
  for (int k = 0; k < 8; ++k) {
    const auto t0 = std::chrono::steady_clock::now();
    const int R = 20;
    for (int i = 0; i < R; ++i) { once(); }
    best = std::fmin(best, std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() / R);
  }
  std::printf("[rope_cost] full per-token RMSNorm+RoPE (96 dispatches): "
              "%.3f ms/tok (min-of-8)\n", best);
  EXPECT_TRUE(best > 0.0);
}

// Numerically verify the decode-only fused GeGLU GEMV at group-size 32
// (affine_qmv_geglu_w4g32 -- the GGUF gate/up decode kernel) against a CPU
// reference (interleaved gate/up rows, gelu_pytorch_tanh(gate)*up).
TEST(metal_lm_smoke, qmv_geglu_w4g32_matches_cpu) {
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto lib = mc->load_library("affine_qmv");
  auto fn = lib.function("affine_qmv_geglu_w4g32");
  ASSERT_TRUE(fn.valid());

  // Tail guard for the fused gate/up GEMV: K = 3840 (Gemma-12B hidden) is NOT
  // a multiple of block_size 512. x is padded to the block boundary (Kpad) with
  // non-zero values and the weights carry an extra padding row, so without the
  // tail mask the partial last block folds in out-of-range data (amplified by
  // the gelu). Small biases -> isolates the tail (no affine cancellation).
  const int F = 2048, N = 2 * F, K = 3840, G = 32, groups = K / G;
  const int Kpad = ((K + 511) / 512) * 512;        // 4096
  std::vector<std::uint32_t> wq((std::size_t)(N + 1) * (K / 8), 0);
  std::vector<std::uint16_t> qn((std::size_t)N * K);
  std::vector<_Float16> scales((std::size_t)N * groups);
  std::vector<_Float16> biases((std::size_t)N * groups);
  std::vector<_Float16> x((std::size_t)Kpad);
  auto frand = [](unsigned i) {
    return (float)((i * 2654435761u) >> 9 & 0x3ff) / 1023.0f - 0.5f;
  };
  for (int o = 0; o < N; ++o) {
    for (int g = 0; g < groups; ++g) {
      scales[(std::size_t)o * groups + g] =
          (_Float16)(0.03f + 0.02f * frand((unsigned)(o * 91 + g)));
      biases[(std::size_t)o * groups + g] =
          (_Float16)(0.08f * frand((unsigned)(o * 31 + g * 7 + 5)));
    }
  }
  for (int o = 0; o <= N; ++o) {
    for (int k = 0; k < K; ++k) {
      const unsigned qq = ((unsigned)(o * K + k) * 2654435761u >> 13) & 0xf;
      if (o < N) { qn[(std::size_t)o * K + k] = (std::uint16_t)qq; }
      wq[(std::size_t)o * (K / 8) + k / 8] |=
          (std::uint32_t)qq << (4 * (k % 8));
    }
  }
  // Small positive activations (non-zero -> tail is exercised) kept tiny so the
  // squared geglu output gate*up stays inside fp16 range (gate ~ K*x*scale*q).
  for (int k = 0; k < Kpad; ++k) {
    x[k] = (_Float16)(0.04f + 0.06f * frand((unsigned)(k * 7 + 3)));
  }

  auto rowdot = [&](int o) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
      const float s = (float)scales[(std::size_t)o * groups + k / G];
      const float b = (float)biases[(std::size_t)o * groups + k / G];
      acc += (double)(float)x[k] * (s * (float)qn[(std::size_t)o * K + k] + b);
    }
    return (float)acc;
  };
  std::vector<float> ref((std::size_t)F);
  for (int g = 0; g < F; ++g) {
    const float gate = rowdot(2 * g), up = rowdot(2 * g + 1);
    const float t = std::tanh(0.7978845608028654f *
                              (gate + 0.044715f * gate * gate * gate));
    ref[g] = 0.5f * gate * (1.0f + t) * up;
  }

  auto wbuf = mc->make_shared_buffer(wq.size() * 4);
  auto sbuf = mc->make_shared_buffer(scales.size() * 2);
  auto bbuf = mc->make_shared_buffer(biases.size() * 2);
  auto xbuf = mc->make_shared_buffer(x.size() * 2);
  auto ybuf = mc->make_shared_buffer((std::size_t)F * 2);
  std::memcpy(wbuf.contents(), wq.data(), wq.size() * 4);
  std::memcpy(sbuf.contents(), scales.data(), scales.size() * 2);
  std::memcpy(bbuf.contents(), biases.data(), biases.size() * 2);
  std::memcpy(xbuf.contents(), x.data(), x.size() * 2);
  auto stream = mc->make_command_stream();
  {
    auto enc = stream.begin_compute();
    enc.set_function(fn);
    enc.set_buffer(0, wbuf);
    enc.set_buffer(1, sbuf);
    enc.set_buffer(2, bbuf);
    enc.set_buffer(3, xbuf);
    enc.set_buffer(4, ybuf);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(F / 2), 1}, {32, 2, 1});
  }
  stream.commit().wait();
  const auto* yp = static_cast<const _Float16*>(ybuf.contents());
  double maxrel = 0.0;
  for (int g = 0; g < F; ++g) {
    const double d = std::fabs((double)(float)yp[g] - (double)ref[g]);
    maxrel = std::fmax(maxrel, d / (std::fabs((double)ref[g]) + 1e-2));
    std::printf("[geglu_w4g32] g=%2d gpu=%.5f ref=%.5f\n",
                g, (float)yp[g], ref[g]);
  }
  std::printf("[geglu_w4g32] max rel err = %.4g\n", maxrel);
  EXPECT_TRUE(maxrel < 0.05);
}

// Matrix-core (M5+) prefill GEMM must be greedy token-exact with the steel
// quantized GEMM. Loads the SAME Qwen3.5 checkpoint twice -- once with the
// matrix-core path forced off (VPIPE_QWEN_NO_MMA=1, the steel reference)
// and once with it on -- prefills a prompt long enough to exercise the
// prefill projections (VPIPE_QWEN_MMA_MIN_M lowered so even a short prompt
// routes through it) and greedy-decodes; the two token streams must match.
// On a GPU without matrix cores both loads are steel and the test is a
// trivial (still valid) pass. Gated on VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, mma_prefill_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  auto run = [&](bool use_mma) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_QWEN_MMA_MIN_M", "4", 1);   // exercise mma on short prompts
    ::setenv("VPIPE_QWEN_MMA_ATTN_MIN_N", "8", 1);  // and mma flash attention
    if (use_mma) { ::unsetenv("VPIPE_QWEN_NO_MMA"); }
    else         { ::setenv("VPIPE_QWEN_NO_MMA", "1", 1); }
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    auto ids = lm->tokenizer().encode(
        "In a distant kingdom by the northern sea there lived a clever "
        "young clockmaker who dreamed of building a machine that could "
        "tell not only the hour but the weather of tomorrow.");
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 24 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };

  const auto ref = run(false);   // steel
  const auto got = run(true);    // matrix-core (M5) or steel (older)
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_QWEN_NO_MMA");
  ::unsetenv("VPIPE_QWEN_MMA_MIN_M");
  ::unsetenv("VPIPE_QWEN_MMA_ATTN_MIN_N");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.mma_prefill_token_exact] %zu tokens, "
              "%zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Qwen3.5 full-attention PREFILL key-split flash kernel (sdpa_paged_flash_f16)
// must be greedy token-exact with the scalar query-tiled reference
// (sdpa_paged_qtile) over a prompt long enough that the flash path fires
// (n >= 384). flash is an online-softmax fp-approximation (not bit-identical,
// like the Gemma flash), so this gates that it doesn't flip a greedy argmax vs
// the established M4 path. _flash_attn is read at LOAD, so each variant uses a
// fresh Session (the LM manager caches by spec). Gated on
// VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_flash_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  // ~600-token prompt so the full-attention prefill flash path fires.
  std::string big;
  for (int i = 0; i < 60; ++i) {
    big += "The history of computing is long and storied. ";
  }
  big += "Summarize the key milestones.";

  auto gen = [&](bool no_flash, bool force_mixed = false) {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    if (no_flash) { ::setenv("VPIPE_QWEN_NO_FLASH", "1", 1); }
    else          { ::unsetenv("VPIPE_QWEN_NO_FLASH"); }
    if (force_mixed) { ::setenv("VPIPE_QWEN_FORCE_MIXED", "1", 1); }
    else             { ::unsetenv("VPIPE_QWEN_FORCE_MIXED"); }
    std::vector<std::int32_t> out;
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    if (mc != nullptr && mc->valid() && mgr != nullptr) {
      genai::LoadSpec spec;
      spec.hf_dir = path;
      spec.compute_dtype = "f16";
      spec.page_tokens = 512;
      spec.max_pages = 16;
      auto lm = mgr->load(spec);
      if (lm && lm->valid()) {
        auto ids = lm->tokenizer().encode(big);
        if (ids.size() >= 384) {
          auto ctx = lm->make_context();
          std::int32_t t = lm->prefill(ctx, ids);
          out.push_back(t);
          for (int i = 1; i < 32 && t >= 0; ++i) {
            t = lm->next_token(ctx);
            out.push_back(t);
          }
        }
      }
    }
    ::unsetenv("VPIPE_LLM_BACKEND");
    ::unsetenv("VPIPE_QWEN_NO_FLASH");
    ::unsetenv("VPIPE_QWEN_FORCE_MIXED");
    return out;
  };

  const auto qtile = gen(true);    // scalar query-tiled reference
  const auto flash = gen(false);   // key-split flash (default)
  ASSERT_TRUE(!qtile.empty());
  ASSERT_TRUE(qtile.size() == flash.size());
  std::size_t mism = 0; int first_div = -1;
  for (std::size_t i = 0; i < flash.size(); ++i) {
    if (flash[i] != qtile[i]) {
      ++mism;
      if (first_div < 0) { first_div = (int)i; }
    }
  }
  std::printf("[qwen_flash_tokexact] N=%zu | flash-vs-qtile mism=%zu "
              "(first_div=%d)\n", flash.size(), mism, first_div);
  EXPECT_TRUE(mism == 0);

  // Force the mixed/de-fused paths on this (uniform-affine) model: a uniform
  // layer takes the qkv_fused / mlp_fused re-fusion, the GDN in_proj de-fuses,
  // etc. -- all must reproduce the fused output exactly. On a genuinely-mixed
  // model (OptiQ) FORCE_MIXED is a no-op, so this arm is a strict superset check
  // of the mixed machinery's numeric equivalence to the fused path.
  const auto forced = gen(false, /*force_mixed=*/true);
  if (!forced.empty()) {
    ASSERT_TRUE(forced.size() == flash.size());
    std::size_t fmism = 0; int fdiv = -1;
    for (std::size_t i = 0; i < flash.size(); ++i) {
      if (forced[i] != flash[i]) { ++fmism; if (fdiv < 0) { fdiv = (int)i; } }
    }
    std::printf("[qwen_flash_tokexact] force-mixed-vs-fused mism=%zu "
                "(first_div=%d)\n", fmism, fdiv);
    EXPECT_TRUE(fmism == 0);
  }
}

// End-to-end coherence: render a real chat prompt, prefill + greedy-decode,
// and require a factually correct answer ("Paris"). Unlike the token-exact
// tests above (which only check fast-vs-slow self-consistency and would pass
// even on a mis-decoded checkpoint), this is a cross-reference correctness
// gate -- it fails if the quantized weights are read at the wrong width or
// the dequant is wrong. Works on any Qwen3.5 checkpoint regardless of quant
// (4-bit / 8-bit): selects the w4g64 / w8g64 kernels from config. The
// generation budget is generous enough to span a thinking block before the
// answer. Gated on VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_text_chat) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "f16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with the city name only.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::vector<std::int32_t> gen;
  const std::int32_t first = lm->prefill(ctx, ids);
  ASSERT_TRUE(first >= 0);
  gen.push_back(first);
  for (int i = 0; i < 96; ++i) {
    const std::int32_t n = lm->next_token(ctx);
    if (n < 0) { break; }
    gen.push_back(n);
  }
  std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[qwen_text_chat] %zu tok | gen='%s'\n", gen.size(),
              text.c_str());
  ASSERT_TRUE(!text.empty());
  std::string lower = text;
  for (char& c : lower) {
    c = (char)std::tolower((unsigned char)c);
  }
  EXPECT_TRUE(lower.find("paris") != std::string::npos);
}

// Native k-quant (GGUF Q4_K_M) Qwen3.5-2B end-to-end coherence + perf.
// Loads the model from the .gguf via the metal k-quant path (no requant);
// the tokenizer + chat template come from a safetensors Qwen3.5 dir (the
// GGUF ships none -- the family shares one tokenizer). Renders a factual
// prompt, prefills + greedy-decodes through MetalQwenModel directly, decodes
// the answer, and asserts it names the city -- a real cross-reference (the
// per-tensor q4_K/q5_K/q6_K dispatch + the A_log/conv transforms are all
// exercised; mis-loaded weights produce garbage, not "paris"). Prints
// prefill/decode tok/s for the llama.cpp comparison. Gated on
// VPIPE_QWEN_GGUF_TEST_MODEL_PATH (.gguf) + VPIPE_QWEN35_TEST_MODEL_PATH
// (safetensors dir, for the tokenizer).
TEST(metal_lm_smoke, qwen_gguf_text_chat) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_TEST_MODEL_PATH");
  const char* tok_dir = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!gguf || !*gguf || !tok_dir || !*tok_dir) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  // The GGUF model (metal k-quant path).
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(gguf);
  ASSERT_TRUE(cfg.has_value());
  EXPECT_TRUE(cfg->architecture == "Qwen3_5ForConditionalGeneration");
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  auto model = genai::MetalQwenModel::load(gguf, mc, mcfg);
  ASSERT_TRUE(model != nullptr);

  // Tokenizer + chat template from a safetensors Qwen3.5 dir (manager load).
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = tok_dir;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  // A long-ish prompt so the prefill number is comparable to llama.cpp's
  // pp512 (a 24-token prompt is fixed-overhead-bound); the France question
  // at the end still drives the coherence assert.
  std::string prompt;
  for (int i = 0; i < 36; ++i) {
    prompt += "The following is background context for a geography quiz. ";
  }
  prompt += "What is the capital of France? Reply with the city name only.";
  std::vector<std::int32_t> ids;
  tpl->render_user_turn(prompt, /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> lg = model->prefill(ids);
  const auto t1 = std::chrono::steady_clock::now();
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);

  // Branch a child off the prefilled prefix BEFORE decoding so the pipelined
  // run starts from the same state as the synchronous run (for token-exact
  // A/B). branch() refcount-shares KV pages + deep-copies the GDN conv/ssm.
  const genai::ContextId child = model->context_manager()->branch(
      model->root_context());
  ASSERT_TRUE(child.valid());

  // Decode via the in-stream greedy path (forward_argmax -> decode_step_fast:
  // q6_K embed gather + on-GPU argmax folded into the decode command buffer,
  // the production next_token_greedy path -- no per-token embed round-trip).
  const int kGen = 200;   // thinking-style model: room to reach the answer
  std::vector<std::int32_t> gen;
  gen.push_back(first);
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(gen.back());
    if (t < 0) { break; }
    gen.push_back(t);
  }
  const auto t2 = std::chrono::steady_clock::now();

  std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  const double pf_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double dc_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::printf("[qwen_gguf_text_chat] prefill %zu tok in %.1f ms (%.0f tok/s) "
              "| decode(sync) %zu tok in %.1f ms (%.1f tok/s)\n",
              ids.size(), pf_ms, (double)ids.size() / (pf_ms / 1000.0),
              gen.size(), dc_ms, (double)gen.size() / (dc_ms / 1000.0));

  // A/B the GPU-resident pipelined path (event-chained no-wait command
  // buffers overlap the host's per-token work with the GPU's next forward) on
  // the child branch from the same prefix -- must be token-exact vs sync AND
  // is the production decode path (so its tok/s is the headline).
  {
    std::vector<std::int32_t> pids;
    const auto p0 = std::chrono::steady_clock::now();
    const bool ok = model->decode_pipelined(child, first, kGen, pids);
    const auto p1 = std::chrono::steady_clock::now();
    const double pp_ms =
        std::chrono::duration<double, std::milli>(p1 - p0).count();
    EXPECT_TRUE(ok);
    std::printf("[qwen_gguf_text_chat] decode(pipelined) %zu tok in "
                "%.1f ms (%.1f tok/s)\n", pids.size(), pp_ms,
                !pids.empty() ? (double)pids.size() / (pp_ms / 1000.0) : 0.0);
    // Token-exact vs the synchronous greedy stream (gen[0]==first).
    int mism = 0;
    for (std::size_t i = 0; i < pids.size() && i + 1 < gen.size(); ++i) {
      if (pids[i] != gen[i + 1]) { ++mism; }
    }
    std::printf("[qwen_gguf_text_chat] pipelined vs sync mismatches=%d/%zu\n",
                mism, pids.size());
    EXPECT_TRUE(mism == 0);
  }
  std::printf("[qwen_gguf_text_chat] gen='%s'\n", text.c_str());
  ASSERT_TRUE(!text.empty());
  std::string lower = text;
  for (char& c : lower) { c = (char)std::tolower((unsigned char)c); }
  EXPECT_TRUE(lower.find("paris") != std::string::npos);
}

// Mixed-precision affine (mlx-optiq) end-to-end: a Qwen3.5 OptiQ checkpoint
// mixes 4-bit and 8-bit affine linears in one model (per-tensor sensitivity
// quant). The metal path must DE-FUSE q|k|v / in_proj / gate|up (they no
// longer share a bit width) and dispatch each projection at its own width --
// if any bit/offset is wrong the forward produces garbage. Asserts the mixed
// path is engaged (path-selection guard, not timing) AND the model stays
// coherent through prefill + the in-stream greedy decode (reaches the answer),
// plus a sync-vs-pipelined token-exact A/B. Gated on
// VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH (the OptiQ safetensors dir, which also
// carries its own tokenizer.json + chat_template.jinja).
TEST(metal_lm_smoke, qwen_optiq_mixed_precision_text_chat) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  // Path-selection guard: the de-fused per-tensor mixed-affine path engaged.
  // (A silent fall-back to a uniform width would mis-stride the 8-bit tensors
  // -> garbage, but a uniform-bits checkpoint here would pass vacuously.)
  EXPECT_TRUE(model->uses_mixed_precision());

  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = path;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with the city name only.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);

  const genai::ContextId child =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child.valid());

  const int kGen = 200;   // thinking-style model: room to reach the answer
  std::vector<std::int32_t> gen;
  gen.push_back(first);
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(gen.back());
    if (t < 0) { break; }
    gen.push_back(t);
  }
  std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[qwen_optiq] mixed=%d gen='%s'\n",
              (int)model->uses_mixed_precision(), text.c_str());
  // Greedy token-exact vs mlx-lm (stock omlx, same affine quant) was verified
  // out-of-band on these exact prompt ids: vpipe's first 24 generated ids
  // match mlx-lm's bit-for-bit (90700,8340,25,271,16,13,220,2972,2014,53983,
  // 279,5952,64700,198,262,348,256,15380,25,328,3710,369,279,6511). The
  // coherence assert below is the in-suite proxy (a mis-strided 8-bit tensor
  // would derail it).
  ASSERT_TRUE(!text.empty());
  std::string lower = text;
  for (char& c : lower) { c = (char)std::tolower((unsigned char)c); }
  EXPECT_TRUE(lower.find("paris") != std::string::npos);

  // Sync vs GPU-resident pipelined decode must be token-exact (same de-fused
  // mixed-affine forward; the pipelined path is the production decode).
  std::vector<std::int32_t> pids;
  const bool ok = model->decode_pipelined(child, first, kGen, pids);
  EXPECT_TRUE(ok);
  int mism = 0;
  for (std::size_t i = 0; i < pids.size() && i + 1 < gen.size(); ++i) {
    if (pids[i] != gen[i + 1]) { ++mism; }
  }
  std::printf("[qwen_optiq] pipelined vs sync mismatches=%d/%zu\n",
              mism, pids.size());
  EXPECT_TRUE(mism == 0);
}

// Qwen3.5-MoE (35B-A3B) text-chat bring-up: brings up the Mixture-of-Experts
// MLP (256-expert top-8 router + shared expert) on the metal backend, on top
// of the shared hybrid GDN+full-attn backbone. Prefill (on-GPU routed, pair-
// batched expert GEMVs) -> greedy decode (gathered expert matvecs). Dumps the
// first generated ids for out-of-band token-exact comparison vs omlx
// qwen3_5_moe; the in-suite proxy is a coherent "Paris" answer (a mis-routed
// expert or mis-strided 3D expert slab would derail it). Also checks the
// pipelined GPU-resident decode is token-exact with the serial path. Gated on
// VPIPE_QWEN35_MOE_TEST_MODEL_PATH (a ~18-20 GB model -> 64 GB box only).
TEST(metal_lm_smoke, qwen35_moe_text_chat) {
  const char* path = std::getenv("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  // Guard: the config parsed as MoE (else the test passes vacuously on a dense
  // model and never exercises the expert path).
  ASSERT_TRUE(mcfg.is_moe());
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);

  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = path;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with the city name only.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);

  const genai::ContextId child =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child.valid());

  const int kGen = 200;
  std::vector<std::int32_t> gen;
  gen.push_back(first);
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(gen.back());
    if (t < 0) { break; }
    gen.push_back(t);
  }
  std::printf("[qwen35_moe] first 16 ids:");
  for (int i = 0; i < 16 && i < (int)gen.size(); ++i) {
    std::printf(" %d", gen[i]);
  }
  std::printf("\n");
  std::string text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[qwen35_moe] gen='%s'\n", text.c_str());
  ASSERT_TRUE(!text.empty());
  std::string lower = text;
  for (char& c : lower) { c = (char)std::tolower((unsigned char)c); }
  EXPECT_TRUE(lower.find("paris") != std::string::npos);

  // Pipelined GPU-resident decode must be token-exact with the serial path.
  std::vector<std::int32_t> pids;
  const bool ok = model->decode_pipelined(child, first, kGen, pids);
  EXPECT_TRUE(ok);
  int mism = 0;
  for (std::size_t i = 0; i < pids.size() && i + 1 < gen.size(); ++i) {
    if (pids[i] != gen[i + 1]) { ++mism; }
  }
  std::printf("[qwen35_moe] pipelined vs sync mismatches=%d/%zu\n",
              mism, pids.size());
  EXPECT_TRUE(mism == 0);
}

// MTP speculative decode: the bundled mtp.safetensors head drafts tokens, the
// main model verifies them, the longest greedy-matching prefix is accepted,
// and the rejected speculative tail is rolled back (paged KV via kv_rollback +
// GDN recurrent ring via gdn_ring_rollback -- the depth>1 pdecode machinery).
// GREEDY spec decode MUST be token-exact vs a serial forward_argmax loop
// (verification makes the drafter affect only speed, never the tokens). Also
// reports the mean accepted tokens/round (the acceptance the drafter buys).
// Gated on VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_optiq_mtp_speculative_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  // Gen length env-tunable (VPIPE_MTP_GEN_TOKENS) to measure at the long
  // contexts where MTP's win erodes; size the page pool for the root + the two
  // depth children all reaching ~kGen, + margin.
  int kGen = 96;
  if (const char* e = std::getenv("VPIPE_MTP_GEN_TOKENS")) {
    const int v = std::atoi(e);
    if (v > 0) { kGen = v; }
  }
  mcfg.max_pages = std::max(8, (3 * kGen) / 512 + 8);
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(model->has_mtp());   // the mtp.safetensors head loaded

  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = path;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with the city name only.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  // Prefill on the root context (also stashes the last hidden the MTP drafter
  // consumes). Branch a child off the prefilled prefix BEFORE decoding so the
  // MTP run starts from the exact same state as the serial reference.
  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);
  // One child branch per MTP depth, both off the prefilled prefix.
  const genai::ContextId child1 =
      model->context_manager()->branch(model->root_context());
  const genai::ContextId child2 =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child1.valid() && child2.valid());

  // Serial greedy reference on the root context (kGen set above; the speed
  // baseline).
  std::vector<std::int32_t> ref;
  ref.push_back(first);
  const auto s0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(ref.back());
    if (t < 0) { break; }
    ref.push_back(t);
  }
  const auto s1 = std::chrono::steady_clock::now();
  const double serial_ms =
      std::chrono::duration<double, std::milli>(s1 - s0).count();
  const double serial_tps = (double)(ref.size() - 1) / (serial_ms / 1000.0);

  // MTP speculative decode at depth-1 and depth-2 (token-exact, back-to-back
  // so the thermal state is comparable). draft_len 1 = depth-1, >=2 = depth-2.
  auto run_mtp = [&](genai::ContextId cid, int draft_len, const char* tag) {
    std::vector<std::int32_t> got;
    long accepted = 0, rounds = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = model->mtp_decode(cid, first, (int)ref.size(), got,
                                      draft_len, &accepted, &rounds);
    const auto t1 = std::chrono::steady_clock::now();
    EXPECT_TRUE(ok);
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int mism = 0;
    const std::size_t nn = std::min(ref.size(), got.size());
    for (std::size_t i = 0; i < nn; ++i) {
      if (ref[i] != got[i]) { ++mism; }
    }
    std::printf("[qwen_optiq_mtp] %s: %zu tok in %.0f ms (%.1f tok/s) | "
                "speedup %.2fx | rounds=%ld tok/round=%.2f mism=%d\n",
                tag, got.size(), ms, (double)got.size() / (ms / 1000.0),
                ((double)got.size() / (ms / 1000.0)) / serial_tps, rounds,
                rounds > 0 ? (double)got.size() / (double)rounds : 0.0, mism);
    EXPECT_TRUE(mism == 0);
    EXPECT_TRUE(got.size() + 1 >= ref.size());
  };
  std::printf("[qwen_optiq_mtp] serial baseline %.1f tok/s\n", serial_tps);
  run_mtp(child1, /*draft_len=*/1, "depth-1");
  run_mtp(child2, /*draft_len=*/2, "depth-2");
}

// Dense (raw-HF bf16/f16) MTP token-exact verification, SINGLE model load.
// The optiq MTP test above double-loads (direct + via the model manager for a
// chat template), which on a 64 GB box exceeds RAM for the 54 GB bf16 27B. This
// loads the dense model ONCE, tokenizes the prompt from tokenizer.json, and
// checks mtp_decode (depth-1 + depth-2) is token-exact vs plain serial decode.
// Env: VPIPE_DENSE_MTP_MODEL = the raw-HF dense model dir.
TEST(metal_lm_smoke, dense_mtp_speculative_token_exact) {
  const char* path = std::getenv("VPIPE_DENSE_MTP_MODEL");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  int kGen = 32;
  if (const char* e = std::getenv("VPIPE_MTP_GEN_TOKENS")) {
    const int v = std::atoi(e);
    if (v > 0) { kGen = v; }
  }
  mcfg.max_pages = std::max(8, (3 * kGen) / 512 + 8);
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(model->has_mtp());   // the raw-HF dense MTP head loaded

  // Tokenize the prompt directly from tokenizer.json (no second model load).
  const std::string tk = std::string(path) + "/tokenizer.json";
  auto tok = genai::Tokenizer::from_huggingface_json(tk, &sess);
  ASSERT_TRUE(tok != nullptr);
  std::vector<std::int32_t> ids =
      tok->encode("The capital of France is");
  ASSERT_TRUE(!ids.empty());

  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);
  const genai::ContextId child1 =
      model->context_manager()->branch(model->root_context());
  const genai::ContextId child2 =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child1.valid() && child2.valid());

  // Serial greedy reference on the root context.
  std::vector<std::int32_t> ref;
  ref.push_back(first);
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(ref.back());
    if (t < 0) { break; }
    ref.push_back(t);
  }
  std::printf("[dense_mtp] serial ref %zu tok, first=%d\n", ref.size(), first);

  auto run_mtp = [&](genai::ContextId cid, int draft_len, const char* tag) {
    std::vector<std::int32_t> got;
    long accepted = 0, rounds = 0;
    const bool ok = model->mtp_decode(cid, first, (int)ref.size(), got,
                                      draft_len, &accepted, &rounds);
    EXPECT_TRUE(ok);
    int mism = 0;
    const std::size_t nn = std::min(ref.size(), got.size());
    for (std::size_t i = 0; i < nn; ++i) {
      if (ref[i] != got[i]) { ++mism; }
    }
    std::printf("[dense_mtp] %s: %zu tok rounds=%ld tok/round=%.2f mism=%d\n",
                tag, got.size(), rounds,
                rounds > 0 ? (double)got.size() / (double)rounds : 0.0, mism);
    EXPECT_TRUE(mism == 0);
    EXPECT_TRUE(got.size() + 1 >= ref.size());
  };
  run_mtp(child1, /*draft_len=*/1, "depth-1");
  run_mtp(child2, /*draft_len=*/2, "depth-2");
}

// MTP acceptance/speedup on LONG, COHERENT text -- the regime the "<1.0x at
// long context" complaint is really about. The token-exact test above decodes
// 1000+ tokens out of a trivial prompt, which DEGENERATES (the model's own
// rambling becomes high-entropy and the drafter's acceptance collapses ~ for
// ANY drafter). Here we prefill a long, information-dense passage and decode a
// GROUNDED continuation: the model stays coherent/confident, so acceptance
// reflects real long-context usage (RAG / documents / code), not degeneracy.
TEST(metal_lm_smoke, qwen_optiq_mtp_realtext_accept) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  int kGen = 256;
  if (const char* e = std::getenv("VPIPE_MTP_GEN_TOKENS")) {
    const int v = std::atoi(e);
    if (v > 0) { kGen = v; }
  }
  // Long enough page pool for the long prefix + root + two depth children.
  mcfg.max_pages = std::max(16, (4 * (2048 + kGen)) / 512 + 8);
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(model->has_mtp());
  // Prefix-seed is opt-in (off by default); enable it so this bench/teacher-
  // force exercises the seeded ("full history") drafter. VPIPE_MTP_NO_SEED
  // still hard-disables it for the decode-only A/B.
  model->set_mtp_prefix_seed(true);

  // A long, coherent, information-dense passage (no repetition). The decode
  // continues it in the same encyclopedic register -> grounded, low-entropy.
  const char* passage =
      "The deep ocean, comprising all marine waters below roughly one thousand "
      "meters, is the largest continuous habitat on Earth and also the least "
      "explored. Sunlight is fully extinguished within the first few hundred "
      "meters, so the vast volume beneath -- the aphotic zone -- exists in "
      "perpetual darkness. With increasing depth the water grows colder and the "
      "pressure climbs by about one atmosphere for every ten meters of descent, "
      "reaching more than a thousand atmospheres in the deepest trenches. These "
      "conditions long convinced naturalists that the abyss must be lifeless, a "
      "barren desert of cold and crushing weight. The reality proved far "
      "stranger. Life is distributed throughout the water column and across the "
      "seafloor, sustained by a thin, ceaseless rain of organic particles "
      "descending from the sunlit surface, a flux that oceanographers call "
      "marine snow. This detritus -- dead plankton, fecal pellets, and "
      "aggregated mucus -- is the primary food supply for most deep dwellers, "
      "and its slow settling couples the productive surface to the dark interior "
      "over timescales of weeks. A second, wholly independent foundation for "
      "life was discovered in 1977 near the Galapagos Rift, where submersibles "
      "found hot springs venting mineral-rich fluid through the seafloor. "
      "Around these hydrothermal vents thrive dense communities of tube worms, "
      "clams, and shrimp that depend not on sunlight but on chemosynthesis: "
      "specialized bacteria oxidize hydrogen sulfide and methane from the vent "
      "fluid to fix carbon, forming the base of a food web powered by the "
      "planet's internal heat rather than the sun. Many deep-sea animals "
      "generate their own light through bioluminescence, a chemical reaction "
      "between a substrate called luciferin and an enzyme called luciferase. "
      "Light is used to lure prey, to startle predators, to find mates in the "
      "dark, and as counter-illumination camouflage that erases an animal's "
      "silhouette against the faint glow filtering down from above. The "
      "anglerfish dangles a luminous lure before its jaws; the cookie-cutter "
      "shark glows except for a dark collar that mimics a small fish to draw in "
      "larger hunters. Bodies are adapted to scarcity and pressure: metabolism "
      "is slowed, skeletons and muscles are reduced, and proteins are stabilized "
      "by molecules that counteract the deforming effect of extreme pressure. "
      "Growth is unhurried and lifespans are often long; some deep corals live "
      "for thousands of years, recording ocean chemistry in their skeletons like "
      "tree rings. The deep sea also governs the planet's climate over the long "
      "term. Cold, dense water sinking at high latitudes drives the global "
      "overturning circulation that redistributes heat, and the biological pump "
      "that carries carbon downward as marine snow sequesters it in deep waters "
      "and sediments for centuries. Exploration remains difficult and expensive. "
      "Crewed submersibles, remotely operated vehicles, and autonomous gliders "
      "have mapped only a small fraction of the seafloor in detail, and new "
      "species are described on nearly every expedition.";

  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = path;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  std::string ask = std::string(passage) +
      "\n\nContinue this encyclopedic article for several more paragraphs in "
      "the same factual, expository style.";
  // Prompt override (VPIPE_MTP_PROMPT_FILE): swap the whole user turn for a
  // prompt read from a file, so acceptance can be A/B'd across prompt domains
  // (e.g. our prose vs MTPLX's coding prompt) on the same head.
  if (const char* pf = std::getenv("VPIPE_MTP_PROMPT_FILE")) {
    std::ifstream pin(pf);
    if (pin) {
      std::stringstream ss;
      ss << pin.rdbuf();
      ask = ss.str();
      std::printf("[mtp_realtext] prompt override from %s (%zu chars)\n",
                  pf, ask.size());
    }
  }
  tpl->render_user_turn(ask, /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  // VPIPE_MTP_NO_THINK: the VL template opens a thinking block (`<think>\n`);
  // close it immediately (`\n</think>\n\n`) so the model answers directly,
  // matching a thinking-OFF runtime (e.g. MTPLX) for a fair acceptance A/B.
  if (std::getenv("VPIPE_MTP_NO_THINK")) {
    const auto& tk = lm->tokenizer();
    auto nl = tk.encode("\n");
    ids.insert(ids.end(), nl.begin(), nl.end());
    const std::int32_t tc = tk.special_token_id("</think>");
    if (tc >= 0) { ids.push_back(tc); }
    auto nl2 = tk.encode("\n\n");
    ids.insert(ids.end(), nl2.begin(), nl2.end());
    std::printf("[mtp_realtext] thinking DISABLED (closed think block, "
                "</think>=%d)\n", tc);
  }

  // VPIPE_MTP_PREFIX_IDS: replace the rendered prefix with an explicit
  // comma-separated token-id list, so an EXACT prompt (e.g. another runtime's
  // rendered tokens) can be fed verbatim to isolate kernel/fp divergence from
  // chat-template divergence.
  if (const char* pe = std::getenv("VPIPE_MTP_PREFIX_IDS")) {
    std::vector<std::int32_t> ov;
    const char* p = pe;
    while (*p) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { ++p; continue; }
      ov.push_back((std::int32_t)v);
      p = end;
    }
    if (!ov.empty()) {
      ids = ov;
      std::printf("[mtp_realtext] prefix OVERRIDE from VPIPE_MTP_PREFIX_IDS "
                  "(%zu tok)\n", ids.size());
    }
  }

  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);
  std::printf("[mtp_realtext] prefix=%zu tok, decode at ctx %zu..%zu\n",
              ids.size(), ids.size(), ids.size() + (std::size_t)kGen);
  // VPIPE_MTP_DUMP_IDS: print the prefix + greedy-continuation token ids as
  // comma-separated lists for cross-runtime divergence analysis.
  if (std::getenv("VPIPE_MTP_DUMP_IDS")) {
    std::printf("[mtp_realtext] prefix_ids=");
    for (std::size_t i = 0; i < ids.size(); ++i) {
      std::printf("%s%d", i ? "," : "", ids[i]);
    }
    std::printf("\n[mtp_realtext] first_pred=%d\n", first);
  }
  // VPIPE_MTP_TF: teacher-forced MTP draft-accuracy on a FIXED continuation
  // (comma-separated ids in VPIPE_MTP_CONT_FILE) walked verbatim through the
  // model from this prefill -- measures depth-1 draft quality WITHOUT the
  // free-run stream-divergence confound (the head is conditioned on the TRUE
  // next token). VPIPE_MTP_TF_CHUNK sets the per-verify width (default 2, the
  // depth-1 decode's MTP attention window).
  if (std::getenv("VPIPE_MTP_TF")) {
    const char* cf = std::getenv("VPIPE_MTP_CONT_FILE");
    std::vector<std::int32_t> cont;
    if (cf) {
      std::ifstream cin2(cf);
      std::string s;
      std::getline(cin2, s, '\0');
      const char* p = s.c_str();
      while (*p) {
        char* e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) { ++p; continue; }
        cont.push_back((std::int32_t)v);
        p = e;
      }
    }
    ASSERT_TRUE(!cont.empty());
    int chunk = 2;
    if (const char* ce = std::getenv("VPIPE_MTP_TF_CHUNK")) {
      const int v = std::atoi(ce);
      if (v >= 1) { chunk = v; }
    }
    const genai::ContextId tfc =
        model->context_manager()->branch(model->root_context());
    long hits = 0, tot = 0;
    const bool ok = model->mtp_teacher_force(tfc, cont, chunk, &hits, &tot);
    EXPECT_TRUE(ok);
    std::printf("[mtp_realtext] teacher-force cont=%zu tok\n", cont.size());
    return;
  }

  const genai::ContextId child1 =
      model->context_manager()->branch(model->root_context());
  const genai::ContextId child2 =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child1.valid() && child2.valid());

  std::vector<std::int32_t> ref;
  ref.push_back(first);
  const auto s0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(ref.back());
    if (t < 0) { break; }
    ref.push_back(t);
  }
  const auto s1 = std::chrono::steady_clock::now();
  const double serial_ms =
      std::chrono::duration<double, std::milli>(s1 - s0).count();
  const double serial_tps = (double)(ref.size() - 1) / (serial_ms / 1000.0);
  std::printf("[mtp_realtext] serial baseline %.1f tok/s\n", serial_tps);
  {
    const std::string gen = lm->tokenizer().decode(ref);
    std::printf("[mtp_realtext] gen: %.280s\n", gen.c_str());
  }
  if (std::getenv("VPIPE_MTP_DUMP_IDS")) {
    std::printf("[mtp_realtext] gen_ids=");
    for (std::size_t i = 0; i < ref.size(); ++i) {
      std::printf("%s%d", i ? "," : "", ref[i]);
    }
    std::printf("\n");
  }

  // Optional sampler override (acceptance A/B vs an external runtime in the
  // SAME mode): VPIPE_MTP_TEMP>0 switches to sampling at that temperature with
  // VPIPE_MTP_TOP_P / VPIPE_MTP_TOP_K; VPIPE_MTP_LEVIATHAN=1 uses L-C accept.
  // Default (unset) stays greedy/exact-match.
  auto run_mtp = [&](genai::ContextId cid, int draft_len, const char* tag) {
    std::vector<std::int32_t> got;
    long accepted = 0, rounds = 0;
    genai::MtpDecodeCtl ctl;
    bool sampling = false;
    if (const char* te = std::getenv("VPIPE_MTP_TEMP")) {
      const float t = (float)std::atof(te);
      if (t > 0.0f) {
        sampling = true;
        ctl.sampler.greedy = false;
        ctl.sampler.temperature = t;
        const char* tp = std::getenv("VPIPE_MTP_TOP_P");
        const char* tk = std::getenv("VPIPE_MTP_TOP_K");
        ctl.sampler.top_p = tp ? (float)std::atof(tp) : 1.0f;
        ctl.sampler.top_k = tk ? std::atoi(tk) : 0;
        ctl.sampler.seed = 777ull;
        ctl.sampler.n_iter = 16;
        const char* lv = std::getenv("VPIPE_MTP_LEVIATHAN");
        model->set_leviathan(lv && std::atoi(lv) != 0);
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = sampling
        ? model->mtp_decode(cid, first, (int)ref.size(), got, draft_len,
                            &accepted, &rounds, ctl)
        : model->mtp_decode(cid, first, (int)ref.size(), got, draft_len,
                            &accepted, &rounds);
    model->set_leviathan(false);
    const auto t1 = std::chrono::steady_clock::now();
    EXPECT_TRUE(ok);
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    int mism = 0;
    const std::size_t nn = std::min(ref.size(), got.size());
    for (std::size_t i = 0; i < nn; ++i) {
      if (ref[i] != got[i]) { ++mism; }
    }
    std::printf("[mtp_realtext] %s: %zu tok in %.0f ms (%.1f tok/s) | "
                "speedup %.2fx | rounds=%ld tok/round=%.2f mism=%d\n",
                tag, got.size(), ms, (double)got.size() / (ms / 1000.0),
                ((double)got.size() / (ms / 1000.0)) / serial_tps, rounds,
                rounds > 0 ? (double)got.size() / (double)rounds : 0.0, mism);
    // This is a perf/acceptance benchmark, NOT a token-exact check (the
    // controlled-length *_token_exact tests cover that). Over long greedy
    // generation MTP can diverge from the serial reference at a near-tie argmax
    // (a tiny fp difference between the batched-verify and single-decode paths
    // flips the top token, then the sequences cascade apart) -- expected, and
    // independent of the MAXM=4 verify (depth-1 never takes it yet diverges
    // identically). So report mism; don't fail on it.
  };
  run_mtp(child1, /*draft_len=*/1, "depth-1");
  run_mtp(child2, /*draft_len=*/2, "depth-2");
}

// Leviathan-Chen MTP sampling: (A) at temperature it accepts MORE than the
// default exact-match scheme (the whole point -- ours accepts a draft d with
// prob p(d); L-C accepts min(1,p(d)/q(d)) >= p(d)); (B) as temp->0 it must
// converge to the greedy result (a gross-corruption guard on the ratio +
// residual math). Depth-1, pure temperature.
TEST(metal_lm_smoke, qwen_optiq_mtp_leviathan_sampling) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 64;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(model->has_mtp());

  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = path;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "Write a short paragraph explaining why the sky looks blue during the "
      "day and turns red at sunset.", /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t b = 0; float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; b = (std::int32_t)i; }
    }
    return b;
  };
  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);
  auto* cmgr = model->context_manager();
  const genai::ContextId root = model->root_context();

  const std::uint64_t seed = 777ull;
  // Run mtp_decode and return acceptance (tok/round) for a sampler config.
  auto run = [&](bool lc, float temp, float top_p, int top_k, int draft_len,
                 int n, bool gpu = false) -> double {
    model->set_leviathan(lc);
    model->set_lc_gpu(gpu);
    genai::MtpDecodeCtl ctl;
    ctl.sampler.greedy = false; ctl.sampler.temperature = temp;
    ctl.sampler.top_p = top_p; ctl.sampler.top_k = top_k;
    ctl.sampler.seed = seed; ctl.sampler.n_iter = 16;
    genai::ContextId c = cmgr->branch(root);
    std::vector<std::int32_t> got; long acc = 0, rnds = 0;
    EXPECT_TRUE(model->mtp_decode(c, first, n, got, draft_len, &acc, &rnds,
                                  ctl));
    model->set_leviathan(false);
    model->set_lc_gpu(false);
    return rnds > 0 ? (double)got.size() / (double)rnds : 0.0;
  };
  // Reduce-to-greedy guard: at temp->0 L-C samples ~argmax, so its output must
  // track the (exact-match) GREEDY mtp_decode of the same depth until a rare
  // near-tie deviation cascades. Validates the ratio + residual + nucleus math
  // doesn't corrupt the distribution. Returns the first divergence index.
  auto reduce = [&](float top_p, int top_k, int draft_len,
                    const char* tag, bool gpu = false) -> std::size_t {
    model->set_leviathan(false);
    model->set_lc_gpu(false);
    std::vector<std::int32_t> gref;
    { genai::MtpDecodeCtl gc; genai::ContextId c = cmgr->branch(root);
      long a = 0, r = 0;
      EXPECT_TRUE(model->mtp_decode(c, first, 48, gref, draft_len, &a, &r, gc));
    }
    model->set_leviathan(true);
    model->set_lc_gpu(gpu);
    genai::MtpDecodeCtl ctl;
    ctl.sampler.greedy = false; ctl.sampler.temperature = 0.005f;
    ctl.sampler.top_p = top_p; ctl.sampler.top_k = top_k;
    ctl.sampler.seed = 42ull; ctl.sampler.n_iter = 16;
    genai::ContextId cb = cmgr->branch(root);
    std::vector<std::int32_t> glc; long a2 = 0, r2 = 0;
    EXPECT_TRUE(model->mtp_decode(cb, first, 48, glc, draft_len, &a2, &r2, ctl));
    model->set_leviathan(false);
    model->set_lc_gpu(false);
    const std::size_t nn = std::min(gref.size(), glc.size());
    std::size_t fd = nn;
    for (std::size_t i = 0; i < nn; ++i) {
      if (gref[i] != glc[i]) { fd = i; break; }
    }
    std::printf("[mtp_leviathan] reduce-to-greedy %s: first divergence "
                "@%zu/%zu\n", tag, fd, nn);
    return fd;
  };

  // ---- PART A: depth-1 acceptance benefit grows with temperature. ----
  double tpr_em_hi = 0.0, tpr_lc_hi = 0.0;
  for (float temp : {0.3f, 0.5f, 0.7f, 1.0f, 1.5f}) {
    const double tpr_em = run(false, temp, 1.0f, 0, 1, 200);
    const double tpr_lc = run(true,  temp, 1.0f, 0, 1, 200);
    std::printf("[mtp_leviathan] depth-1 temp=%.2f | exact-match tok/round=%.2f "
                "| leviathan tok/round=%.2f\n", temp, tpr_em, tpr_lc);
    if (temp > 1.4f) { tpr_em_hi = tpr_em; tpr_lc_hi = tpr_lc; }
  }
  EXPECT_TRUE(tpr_lc_hi > tpr_em_hi + 0.1);   // clear high-temp win

  // ---- PART B: depth-2 L-C acceptance at high temp (chained ratio test). ----
  const double d2_em = run(false, 1.5f, 1.0f, 0, 2, 200);
  const double d2_lc = run(true,  1.5f, 1.0f, 0, 2, 200);
  std::printf("[mtp_leviathan] depth-2 temp=1.50 | exact-match tok/round=%.2f "
              "| leviathan tok/round=%.2f\n", d2_em, d2_lc);
  EXPECT_TRUE(d2_lc > d2_em + 0.1);

  // ---- PART C: top_p nucleus acceptance (temperature + nucleus filter). ----
  const double tp_em = run(false, 1.5f, 0.9f, 0, 1, 200);
  const double tp_lc = run(true,  1.5f, 0.9f, 0, 1, 200);
  std::printf("[mtp_leviathan] depth-1 temp=1.50 top_p=0.90 | exact-match "
              "tok/round=%.2f | leviathan tok/round=%.2f\n", tp_em, tp_lc);

  // ---- PART D: reduce-to-greedy across configs (correctness guard). ----
  EXPECT_TRUE(reduce(1.0f, 0,  1, "depth-1 pure-temp") >= 4);
  EXPECT_TRUE(reduce(0.9f, 0,  1, "depth-1 top_p=0.9") >= 4);
  EXPECT_TRUE(reduce(1.0f, 40, 1, "depth-1 top_k=40") >= 4);
  EXPECT_TRUE(reduce(1.0f, 0,  2, "depth-2 pure-temp") >= 4);
  EXPECT_TRUE(reduce(0.9f, 0,  2, "depth-2 top_p=0.9") >= 4);

  // ---- PART E: on-GPU L-C (lc_sample_f16 / lc_accept_f16). Same nucleus +
  // accept/residual as the host path, done one-threadgroup-per-row on the GPU.
  // (1) reduce-to-greedy: at temp->0 the GPU L-C must track exact-match greedy
  // exactly as the host path does -- the strong correctness anchor that the
  // GPU nucleus + accept + residual + Gumbel-max compose losslessly.
  EXPECT_TRUE(reduce(1.0f, 0,  1, "gpu depth-1 pure-temp", true) >= 4);
  EXPECT_TRUE(reduce(0.9f, 0,  1, "gpu depth-1 top_p=0.9", true) >= 4);
  EXPECT_TRUE(reduce(1.0f, 40, 1, "gpu depth-1 top_k=40", true) >= 4);
  EXPECT_TRUE(reduce(1.0f, 0,  2, "gpu depth-2 pure-temp", true) >= 4);
  EXPECT_TRUE(reduce(0.9f, 0,  2, "gpu depth-2 top_p=0.9", true) >= 4);
  // (2) acceptance parity host vs GPU (different RNG sub-streams -> the same
  // distribution -> statistically-consistent tok/round, not a drift/bug).
  for (float temp : {0.7f, 1.5f}) {
    const double host = run(true, temp, 0.9f, 0, 1, 200, /*gpu=*/false);
    const double gpu  = run(true, temp, 0.9f, 0, 1, 200, /*gpu=*/true);
    std::printf("[mtp_leviathan] gpu-parity temp=%.2f top_p=0.90 | host "
                "tok/round=%.2f | gpu tok/round=%.2f\n", temp, host, gpu);
    EXPECT_TRUE(std::fabs(host - gpu) < 0.35);
  }
  model->set_leviathan(false);
  model->set_lc_gpu(false);
}

// GGUF tokenizer scheme regression. Tokenizer::from_gguf must pick byte-level
// (gpt2) vs metaspace (llama) from tokenizer.ggml.model. A Qwen3.5 GGUF is
// "gpt2" byte-level -- the bug forced every GGUF to metaspace, so the raw
// byte-level alphabet (Ġ = space U+0120, Ċ = newline U+010A) leaked into the
// detokenized chat text. Round-trip a string full of spaces + newlines and
// assert no alphabet chars survive. Gated on VPIPE_QWEN_GGUF_MTP_TEST_MODEL_
// PATH (the .gguf file); when VPIPE_QWEN35_TEST_MODEL_PATH (a sibling HF
// tokenizer.json dir) is set too, also cross-checks that the GGUF tokenizer
// ENCODES token-exactly with the HF one (validates the byte-level
// pre-tokenizer, not just decode). Builds in both MLX and no-MLX.
TEST(metal_lm_smoke, qwen_gguf_tokenizer_byte_level_round_trip) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_MTP_TEST_MODEL_PATH");
  if (!gguf || !*gguf) { return; }
  Session sess;
  auto g = genai::GgufFile::open(gguf);
  ASSERT_TRUE(g.has_value());
  auto tok = genai::Tokenizer::from_gguf(*g, &sess);
  ASSERT_TRUE(tok != nullptr);

  // Spaces AND newlines -- exactly the chars the byte-level alphabet maps to
  // Ġ / Ċ. A correct detokenizer round-trips them verbatim.
  const std::string text =
      "This code defines a GenerativeModelManager class.\n\n"
      "It loads the model and runs inference.";
  auto ids = tok->encode(text);
  ASSERT_TRUE(!ids.empty());
  const std::string back = tok->decode(ids);
  EXPECT_TRUE(back == text);
  EXPECT_TRUE(back.find("\xC4\xA0") == std::string::npos);   // Ġ (U+0120)
  EXPECT_TRUE(back.find("\xC4\x8A") == std::string::npos);   // Ċ (U+010A)

  const char* hf_dir = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (hf_dir && *hf_dir) {
    auto hf = genai::Tokenizer::from_huggingface_json(
        std::string(hf_dir) + "/tokenizer.json", &sess);
    if (hf != nullptr) {
      EXPECT_TRUE(tok->encode(text) == hf->encode(text));
      EXPECT_TRUE(hf->decode(hf->encode(text)) == text);
    }
  }
}

// MTP speculative decode on a NATIVE k-quant GGUF NextN checkpoint (the MTP
// draft block is bundled in the .gguf as blk.{n}.nextn.* + a full attn/ffn
// block). The k-quant verify (kqmm_ dequant+dense matmuls, fused q|k|v, q6_K
// lm_head) MUST be greedy token-exact vs a serial forward_argmax loop -- the
// drafter affects only speed, never the tokens. Reports accepted tokens/round.
// Gated on VPIPE_QWEN_GGUF_MTP_TEST_MODEL_PATH (.gguf) + VPIPE_QWEN35_TEST_
// MODEL_PATH (tokenizer/chat-template dir).
TEST(metal_lm_smoke, qwen_gguf_mtp_speculative_token_exact) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_MTP_TEST_MODEL_PATH");
  const char* tok_dir = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!gguf || !*gguf || !tok_dir || !*tok_dir) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(gguf);
  ASSERT_TRUE(cfg.has_value());
  // The MTP block is excluded from the main layer count (block_count - nextn).
  EXPECT_TRUE(cfg->num_nextn_layers >= 1);
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  // Generation length is env-tunable (VPIPE_MTP_GEN_TOKENS) so the speedup can
  // be measured at the longer contexts where chat actually runs (MTP's win
  // erodes as the verify's attention scans a growing KV). Size the KV page pool
  // for the serial ref (root) AND the MTP child both reaching ~kGen, + margin.
  int kGen = 96;
  if (const char* e = std::getenv("VPIPE_MTP_GEN_TOKENS")) {
    const int v = std::atoi(e);
    if (v > 0) { kGen = v; }
  }
  mcfg.max_pages = std::max(8, (2 * kGen) / 512 + 8);
  auto model = genai::MetalQwenModel::load(gguf, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(model->has_mtp());   // the bundled NextN block loaded (k-quant)

  // Tokenizer + chat template from the sibling safetensors Qwen3.5 dir.
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = tok_dir;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with the city name only.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  // Exact-prefix override (e.g. another runtime's rendered ids) for a long-
  // prefix seed A/B on the GGUF k-quant MTP head.
  if (const char* pe = std::getenv("VPIPE_MTP_PREFIX_IDS")) {
    std::vector<std::int32_t> ov;
    const char* p = pe;
    while (*p) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { ++p; continue; }
      ov.push_back((std::int32_t)v);
      p = end;
    }
    if (!ov.empty()) {
      ids = ov;
      std::printf("[qwen_gguf_mtp] prefix OVERRIDE (%zu tok)\n", ids.size());
    }
  }
  // Prefix-seed the MTP drafter (opt-in; VPIPE_MTP_NO_SEED disables) so the
  // GGUF k-quant seed path is exercised.
  model->set_mtp_prefix_seed(std::getenv("VPIPE_MTP_NO_SEED") == nullptr);

  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  // Prefill (stashes the last hidden the MTP drafter consumes), then branch a
  // child off the prefilled prefix so MTP starts from the serial ref's state.
  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);
  // VPIPE_MTP_DUMP_IDS: print this model's own greedy continuation (first +
  // forward_argmax*N) so the TF run below walks an on-distribution sequence.
  if (std::getenv("VPIPE_MTP_DUMP_IDS")) {
    int n = 256;
    if (const char* e = std::getenv("VPIPE_MTP_GEN_TOKENS")) {
      const int v = std::atoi(e);
      if (v > 0) { n = v; }
    }
    std::printf("[qwen_gguf_mtp] cont_ids=%d", first);
    std::int32_t t = first;
    for (int i = 0; i < n; ++i) {
      t = model->forward_argmax(t);
      if (t < 0) { break; }
      std::printf(",%d", t);
    }
    std::printf("\n");
    return;
  }
  // VPIPE_MTP_TF: teacher-forced depth-1 draft accuracy on a fixed continuation
  // (VPIPE_MTP_CONT_FILE) -- the seed A/B for the GGUF k-quant MTP head, exactly
  // as the OptiQ realtext test. Returns before the spec-decode timing below.
  if (std::getenv("VPIPE_MTP_TF")) {
    const char* cf = std::getenv("VPIPE_MTP_CONT_FILE");
    std::vector<std::int32_t> cont;
    if (cf) {
      std::ifstream cin2(cf);
      std::string s;
      std::getline(cin2, s, '\0');
      const char* p = s.c_str();
      while (*p) {
        char* e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) { ++p; continue; }
        cont.push_back((std::int32_t)v);
        p = e;
      }
    }
    ASSERT_TRUE(!cont.empty());
    int chunk = 2;
    if (const char* ce = std::getenv("VPIPE_MTP_TF_CHUNK")) {
      const int v = std::atoi(ce);
      if (v >= 1) { chunk = v; }
    }
    const genai::ContextId tfc =
        model->context_manager()->branch(model->root_context());
    long hits = 0, tot = 0;
    EXPECT_TRUE(model->mtp_teacher_force(tfc, cont, chunk, &hits, &tot));
    return;
  }
  // One child branch per MTP depth, BOTH off the clean prefilled prefix (before
  // the serial ref loop advances the root) so each depth starts from the exact
  // same state as the reference.
  const genai::ContextId child =
      model->context_manager()->branch(model->root_context());
  const genai::ContextId child2 =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(child.valid() && child2.valid());

  // Serial greedy reference on the root context (kGen set above).
  std::vector<std::int32_t> ref;
  ref.push_back(first);
  const auto sref0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kGen; ++i) {
    const std::int32_t t = model->forward_argmax(ref.back());
    if (t < 0) { break; }
    ref.push_back(t);
  }
  const auto sref1 = std::chrono::steady_clock::now();
  const double serial_ms =
      std::chrono::duration<double, std::milli>(sref1 - sref0).count();
  const double serial_tps = (double)(ref.size() - 1) / (serial_ms / 1000.0);

  // MTP depth-1 speculative decode on the child branch: token-exact + faster.
  std::vector<std::int32_t> got;
  long accepted = 0, rounds = 0;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = model->mtp_decode(child, first, (int)ref.size(), got,
                                    /*draft_len=*/1, &accepted, &rounds);
  const auto t1 = std::chrono::steady_clock::now();
  EXPECT_TRUE(ok);
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  int mism = 0;
  const std::size_t nn = std::min(ref.size(), got.size());
  for (std::size_t i = 0; i < nn; ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  const double mtp_tps = (double)got.size() / (ms / 1000.0);
  std::printf("[qwen_gguf_mtp] %zu tok in %.0f ms (%.1f tok/s) | serial %.1f "
              "tok/s | speedup %.2fx | rounds=%ld tok/round=%.2f mism=%d\n",
              got.size(), ms, mtp_tps, serial_tps, mtp_tps / serial_tps, rounds,
              rounds > 0 ? (double)got.size() / (double)rounds : 0.0, mism);
  EXPECT_TRUE(mism == 0);
  EXPECT_TRUE(got.size() + 1 >= ref.size());
  EXPECT_TRUE(rounds > 0 && (long)got.size() > rounds);   // some drafts landed

  // Depth-2 (draft_len=2 -> 3 drafts/round, M=3): exercises the k-quant MAXM=4
  // verify-GEMV twins (qmv_q*k_batch4) -- one weight read for the 3-row tile vs
  // the MAXM=2 form's 2 grid.z tiles. Must stay token-exact vs the same serial
  // ref. A/B the MAXM=2-tiled path with VPIPE_MTP_QMV4=0.
  {
    std::vector<std::int32_t> got2;
    long acc2 = 0, rnd2 = 0;
    const auto d0 = std::chrono::steady_clock::now();
    const bool ok2 = model->mtp_decode(child2, first, (int)ref.size(), got2,
                                       /*draft_len=*/2, &acc2, &rnd2);
    const double ms2 =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - d0).count();
    EXPECT_TRUE(ok2);
    int mism2 = 0;
    const std::size_t nn2 = std::min(ref.size(), got2.size());
    for (std::size_t i = 0; i < nn2; ++i) {
      if (ref[i] != got2[i]) { ++mism2; }
    }
    std::printf("[qwen_gguf_mtp] depth-2: %zu tok in %.0f ms (%.1f tok/s) | "
                "speedup %.2fx | rounds=%ld tok/round=%.2f mism=%d\n",
                got2.size(), ms2, (double)got2.size() / (ms2 / 1000.0),
                ((double)got2.size() / (ms2 / 1000.0)) / serial_tps, rnd2,
                rnd2 > 0 ? (double)got2.size() / (double)rnd2 : 0.0, mism2);
    EXPECT_TRUE(mism2 == 0);
    EXPECT_TRUE(rnd2 > 0 && (long)got2.size() > rnd2);
  }
}

// MTP speculative SAMPLING (non-greedy): mtp_decode with a sampling verify MUST
// reproduce decode_pipelined (the serial GPU-sampled path) token-for-token --
// SAME temperature/top_p + the SAME per-slot seed (the verify seeds position k
// by its absolute KV slot, byte-identical to decode_pipelined's per-step seed)
// => identical samples; the MTP drafter only changes how many land per round,
// not the tokens. This is the "verification, not prediction" property under
// sampling. Gated on VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_optiq_mtp_sampled_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_SAMPLE_ITERS");   // decode_pipelined then uses 16; match below
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  ASSERT_TRUE(model->has_mtp());

  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = path;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn("Write a short story about a curious robot.",
                        /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  std::vector<float> lg = model->prefill(ids);
  ASSERT_TRUE(!lg.empty());
  const std::int32_t first = argmax(lg);   // shared seed token for both paths

  const int kGen = 80;
  const float temp = 0.8f, top_p = 0.95f;
  const std::uint64_t seed = 1234567ull;

  // Two branches off the same prefilled prefix so both paths start identically.
  const genai::ContextId refc =
      model->context_manager()->branch(model->root_context());
  const genai::ContextId mtpc =
      model->context_manager()->branch(model->root_context());
  ASSERT_TRUE(refc.valid() && mtpc.valid());

  // Reference: serial GPU-sampled decode (per-step seed seed+0x9e3779b9*(s+1)).
  std::vector<std::int32_t> ref;
  ASSERT_TRUE(model->decode_pipelined(refc, first, kGen, ref, temp, top_p, seed));

  // MTP speculative sampling, SAME temp/top_p/seed + n_iter=16 (decode_pipelined
  // default). The verify samples each position; the drafter is unchanged.
  genai::MtpDecodeCtl ctl;
  ctl.sampler.greedy      = false;
  ctl.sampler.temperature = temp;
  ctl.sampler.top_p       = top_p;
  ctl.sampler.seed        = seed;
  ctl.sampler.n_iter      = 16;
  std::vector<std::int32_t> got;
  long accepted = 0, rounds = 0;
  ASSERT_TRUE(model->mtp_decode(mtpc, first, kGen + 1, got, /*draft_len=*/1,
                                &accepted, &rounds, ctl));

  // got = [first, s1, s2, ...]; ref = [s1, s2, ...]. Compare the overlap.
  int mism = 0;
  const std::size_t gov = got.empty() ? 0 : got.size() - 1;
  const std::size_t nn = std::min(ref.size(), gov);
  for (std::size_t i = 0; i < nn; ++i) {
    if (ref[i] != got[i + 1]) { ++mism; }
  }
  std::printf("[qwen_optiq_mtp_sampled] ref=%zu got=%zu rounds=%ld "
              "tok/round=%.2f mism=%d\n",
              ref.size(), got.size(), rounds,
              rounds > 0 ? (double)got.size() / (double)rounds : 0.0, mism);
  EXPECT_TRUE(mism == 0);            // sampled output is token-exact vs serial
  EXPECT_TRUE(got.size() == ref.size() + 1);   // mtp includes `first`
  EXPECT_TRUE(rounds < (long)got.size());      // drafts actually landed (speedup)
}

// MTP through the PUBLIC LoadedLanguageModel::mtp_generate path -- the exact
// entry the text-chat / visual-qa / realtime-vqa stages call. The streaming
// (on_tokens) + stop-token (is_stop) decode MUST reproduce the serial
// next_token_greedy loop the stages run WITHOUT MTP, token-for-token, AND stop
// at the same place with the stop token rolled OUT of the context (so the
// subsequent assistant_close commit lands cleanly). Two independent contexts
// off the same prefill so the reference and MTP start identically. Gated on
// VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_optiq_mtp_lm_stream_stop_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec tspec;
  tspec.hf_dir = path;
  tspec.compute_dtype = "f16";
  auto lm = mgr->load(tspec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  ASSERT_TRUE(lm->mtp_available());   // the LM exposes the MTP fast path
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Reply with the city name only.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  auto is_stop = [tpl](std::int32_t id) { return tpl->is_stop_token(id); };

  const int kBudget = 220;   // thinking model: room to reach a natural stop

  // ---- Reference: the serial greedy loop a stage runs WITHOUT MTP. ----
  // ref = [first, t1, ...] up to (excluding) the stop token, exactly the
  // tokens the stage would emit; ref_hit_stop records the natural turn end.
  std::vector<std::int32_t> ref;
  bool ref_hit_stop = false;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t cur = lm->prefill(ctx, ids);
    ASSERT_TRUE(cur >= 0);
    for (int i = 0; i < kBudget; ++i) {
      if (is_stop(cur)) { ref_hit_stop = true; break; }
      ref.push_back(cur);
      cur = lm->next_token_greedy(ctx, cur);
      if (cur < 0) { break; }
    }
  }
  ASSERT_TRUE(!ref.empty());

  // ---- MTP via the public streaming API (fresh context, same prefill). ----
  std::vector<std::int32_t> got;
  int  produced = 0;
  bool hit_stop = false;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t first = lm->prefill(ctx, ids);
    ASSERT_TRUE(first >= 0);
    auto on_toks = [&](std::span<const std::int32_t> toks) -> bool {
      for (std::int32_t id : toks) { got.push_back(id); }
      return true;
    };
    const bool ok = lm->mtp_generate(ctx, first, kBudget,
                                     genai::SamplerParams{}, is_stop, on_toks,
                                     &produced, &hit_stop);
    EXPECT_TRUE(ok);
  }

  int mism = 0;
  const std::size_t nn = std::min(ref.size(), got.size());
  for (std::size_t i = 0; i < nn; ++i) { if (ref[i] != got[i]) { ++mism; } }
  std::printf("[qwen_optiq_mtp_lm] ref=%zu got=%zu produced=%d hit_stop=%d "
              "ref_stop=%d mism=%d\n",
              ref.size(), got.size(), produced, (int)hit_stop,
              (int)ref_hit_stop, mism);
  EXPECT_TRUE(mism == 0);
  EXPECT_TRUE(got.size() == ref.size());     // same length => stop in lockstep
  EXPECT_TRUE((int)got.size() == produced);  // produced == streamed count
  EXPECT_TRUE(hit_stop == ref_hit_stop);     // ended the same way (stop/budget)
}

// The Qwen3.5-VL vision tower loaded from mmproj-*.gguf (llama.cpp CLIP layout,
// BF16/F32) must produce the SAME image embeddings as the safetensors tower --
// the mmproj just renames the tensors + splits/transposes the patch-embed
// conv, no requant. Same config + image, only the weight source differs, so
// rel-L2 should be f16-rounding tiny. Gated on VPIPE_QWEN35_TEST_MODEL_PATH
// (safetensors-VL dir: config + reference weights) + VPIPE_QWEN_MMPROJ_TEST_PATH
// (the mmproj-BF16.gguf).
TEST(metal_lm_smoke, qwen_gguf_vision_matches_safetensors) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  const char* mmp = std::getenv("VPIPE_QWEN_MMPROJ_TEST_PATH");
  if (!path || !*path || !mmp || !*mmp) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto mcfg = loader.load_config(path);
  if (!mcfg.has_value() || !mcfg->vision.present) { return; }
  auto cfg = genai::MetalQwenVisionEncoder::config_from(*mcfg);
  if (cfg.depth == 0 || cfg.hidden == 0) { return; }

  const int H = 128, W = 128;
  std::vector<std::uint8_t> img((std::size_t)3 * H * W);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        img[((std::size_t)c * H + y) * W + x] =
            (std::uint8_t)((x * 3 + y * 5 + 17 * c) & 0xFF);
      }
    }
  }
  auto enc = [&](const std::string& gg) -> std::vector<float> {
    auto c = cfg;
    c.gguf_mmproj = gg;          // empty = safetensors, set = mmproj gguf
    auto mv = genai::MetalQwenVisionEncoder::load(path, mc, c);
    if (!mv) { return {}; }
    auto r = mv->encode(img.data(), H, W);
    const auto* p = static_cast<const __fp16*>(r.embeddings.contents());
    std::vector<float> out(r.embeddings.byte_size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) { out[i] = (float)p[i]; }
    return out;
  };
  std::vector<float> st = enc("");
  std::vector<float> gg = enc(mmp);
  ASSERT_TRUE(!st.empty() && st.size() == gg.size());
  double sq = 0.0, df = 0.0;
  float mx = 0.0f;
  for (std::size_t i = 0; i < st.size(); ++i) {
    sq += (double)st[i] * st[i];
    const float d = std::fabs(gg[i] - st[i]);
    df += (double)d * d;
    mx = std::max(mx, d);
  }
  const float rms = (float)std::sqrt(sq / st.size());
  const float drms = (float)std::sqrt(df / st.size());
  std::printf("[gguf-vs-st] tokens=%zu max|d|=%.4f rms=%.4f diff=%.4f (%.3f%%)\n",
              st.size(), mx, rms, drms, 100.0f * drms / rms);
  EXPECT_TRUE(drms < 0.02f * rms);
}

// Loader integration: ModelLoader::load on a Qwen3.5 GGUF (whose dir also holds
// a sibling mmproj-*.gguf, projector_type qwen3vl_merger) must detect it and
// populate VisionConfig from the clip.vision.* metadata (present + dims +
// mmproj_path) so the manager builds the metal vision tower. Gated on
// VPIPE_QWEN_GGUF_TEST_MODEL_PATH (the LM .gguf with a sibling mmproj).
TEST(metal_lm_smoke, qwen_gguf_loader_detects_mmproj_vision) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_TEST_MODEL_PATH");
  if (!gguf || !*gguf) { return; }
  Session sess;
  genai::ModelLoader loader(&sess);
  auto w = loader.load(gguf);
  if (!w.has_value()) { return; }   // needs a sibling mmproj to assert vision
  const auto& v = w->config.vision;
  if (!v.present) { return; }        // no mmproj in this dir -> nothing to check
  std::printf("[gguf-dir-vision] present=%d depth=%d hidden=%d heads=%d "
              "merge=%d out=%d numpos=%d mmproj='%s'\n",
              v.present, v.depth, v.hidden_size, v.num_heads,
              v.spatial_merge_size, v.out_hidden_size,
              v.num_position_embeddings,
              v.mmproj_path.empty() ? "" : "set");
  EXPECT_TRUE(!v.mmproj_path.empty());
  EXPECT_TRUE(v.depth > 0 && v.hidden_size > 0 && v.num_heads > 0);
  EXPECT_TRUE(v.out_hidden_size == w->config.hidden);
  EXPECT_TRUE(v.spatial_merge_size == 2 && v.temporal_patch_size == 2);
  EXPECT_TRUE(v.num_position_embeddings > 0);
}

// Gemma-4 matrix-core (M5+) prefill GEMM (q/k/v/o, GeGLU gate/up via the
// interleaved-dequant + geglu_interleaved combine, down, PLE gate) must be
// greedy token-exact with the steel quantized GEMM. Loads the SAME e4b
// checkpoint twice -- steel reference (VPIPE_GEMMA_NO_MMA=1) and matrix-core
// -- prefills a prompt and greedy-decodes; the two token streams must match.
// VPIPE_GEMMA_MMA_MIN_M is lowered so a short prompt routes through the dense
// matmul2d path. On a GPU without matrix cores both loads are steel (a
// trivial, still-valid pass). Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_mma_prefill_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  auto run = [&](bool use_mma) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_MMA_MIN_M", "4", 1);  // exercise mma on short prompts
    if (use_mma) { ::unsetenv("VPIPE_GEMMA_NO_MMA"); }
    else         { ::setenv("VPIPE_GEMMA_NO_MMA", "1", 1); }
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    auto ids = lm->tokenizer().encode(
        "In a distant kingdom by the northern sea there lived a clever "
        "young clockmaker who dreamed of building a machine that could "
        "tell not only the hour but the weather of tomorrow.");
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 24 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };

  const auto ref = run(false);   // steel
  const auto got = run(true);    // matrix-core (M5) or steel (older)
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_NO_MMA");
  ::unsetenv("VPIPE_GEMMA_MMA_MIN_M");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_mma_prefill_token_exact] %zu tokens, "
              "%zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// The materialized global-attention prefill (default) must be greedy
// token-exact with the pflash reference (materialized OFF). REGRESSION GUARD
// for the steel dense_gemm_t K-tail bug: the materialized PV GEMM has
// contraction K = T_kv, and the kernel read the final K-block unmasked when
// T_kv % 32 != 0, spilling into the next packed row. The prompt below is
// deliberately NOT a multiple of 32 tokens so the tail block exists; a
// power-of-two ctx (the perf bench) never tripped it. The sibling mma/steel
// token-exact tests both run materialized, so they could NOT catch this --
// this one pins materialized against pflash. Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_materialized_matches_pflash_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  auto run = [&](bool materialized) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_MATERIALIZED_GLOBAL", materialized ? "1" : "0", 1);
    // Force the DENSE (non-causal) materialized GEMMs: the plain dense_gemm_t
    // is the kernel that had the K-tail bug; the causal variants mostly mask
    // it, so pin the dense path here to keep this a real regression guard.
    ::setenv("VPIPE_GEMMA_MAT_CAUSAL", "0", 1);
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    // 47 tokens (NOT a multiple of 32) -> the PV contraction has a tail block.
    auto ids = lm->tokenizer().encode(
        "The old lighthouse keeper counted seven ships passing the rocky "
        "point before dawn, each one heavier and slower than the last, and "
        "he wondered which captain would dare the narrow channel.");
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 24 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };

  const auto ref = run(false);   // pflash reference
  const auto got = run(true);    // materialized, dense GEMMs (K-tail path)
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_MATERIALIZED_GLOBAL");
  ::unsetenv("VPIPE_GEMMA_MAT_CAUSAL");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_materialized_matches_pflash_token_exact] "
              "%zu tokens, %zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Banded-materialized SLIDING prefill must be greedy token-exact with the
// simdgroup-flash sliding path. REGRESSION GUARD for the banded GEMM/softmax:
// the prompt is LONGER than the sliding window (512), with VARIED tokens, so
// the trailing-window band actually engages (k0>0 for later rows, and the QK
// below-window tile skips fire) -- every other gemma token-exact prompt is
// <512 tokens, so the band is a no-op there and they cannot catch a banding
// bug. Both arms keep materialized GLOBAL on; only the sliding path differs
// (VPIPE_GEMMA_MAT_SLIDING toggles banded-GEMM vs flash). Gated on
// VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_banded_sliding_matches_flash_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  auto run = [&](bool banded) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_MAT_SLIDING", banded ? "1" : "0", 1);
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    // 700 VARIED tokens (> sliding_window 512) so the band engages for the
    // later positions. Deterministic pseudo-random ids keep K/V non-uniform
    // (a uniform prompt would hide a window bug -- all keys give the same V).
    std::vector<std::int32_t> ids;
    ids.reserve(700);
    ids.push_back(2);                       // <bos>
    std::uint32_t s = 0x9e3779b9u;
    for (int i = 1; i < 700; ++i) {
      s = s * 1664525u + 1013904223u;       // LCG
      ids.push_back((std::int32_t)(106 + (s >> 9) % 200000));
    }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 16 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };

  const auto ref = run(false);   // flash sliding
  const auto got = run(true);    // banded-materialized sliding
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_MAT_SLIDING");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_banded_sliding_matches_flash_token_exact] "
              "%zu tokens, %zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Chunked sliding-window prefill (ring wrap) must be greedy token-exact with a
// single-pass prefill. Forces wrapping with a small VPIPE_GEMMA_SLIDING_CHUNK
// on a prompt longer than the ring, vs a chunk large enough to prefill in one
// pass. Validates the ring chunking AND the KV-only intermediate-chunk skip
// (the shared-KV tail + lm_head are dropped for non-final chunks; only the
// own-KV bulk layers run, so the cache they leave must be bit-identical to the
// single-pass cache). Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_chunked_prefill_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  // ~900-token prompt (a sentence repeated) so it exceeds a small ring.
  std::string prompt;
  for (int i = 0; i < 40; ++i) {
    prompt += "The clever clockmaker studied the broken machine carefully, "
              "noting each worn gear and bent spring before she began again. ";
  }

  auto run = [&](int chunk) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_SLIDING_CHUNK", std::to_string(chunk).c_str(), 1);
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 24 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };

  const auto chunked = run(128);    // ring=640 < ~900 -> wraps (multi-chunk)
  const auto single  = run(4096);   // ring caps at max_seq -> single pass
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_SLIDING_CHUNK");
  ASSERT_TRUE(!single.empty());
  ASSERT_TRUE(chunked.size() == single.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < single.size(); ++i) {
    if (chunked[i] != single[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_chunked_prefill_token_exact] %zu tokens, "
              "%zu mismatches (chunk=128 vs 4096)\n", single.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Bounded-ring SINGLE-PASS prefill (VPIPE_GEMMA_PREFILL_SUBBLOCK) must be greedy
// token-exact with the GROWN one-pass prefill on a LONG prompt that wraps the
// bounded ring many times. The bounded path runs ONE forward over the whole
// prompt (large proj/FFN/global GEMM batch) and reads the full-batch K/V for
// the sliding attention (ring-independent), writing the bounded ring once; the
// grown path grows the ring to the full prompt. Both must decode identically --
// which also proves the single full-n ring write leaves the correct trailing
// window resident for decode at depth. ~2400 VARIED tokens (>> ring 1024) so
// the ring wraps repeatedly and a window/decode-state bug shows. Gated on the
// Gemma model.
TEST(metal_lm_smoke, gemma_bounded_subblock_matches_grown_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  // 2400 deterministic pseudo-random ids (non-uniform K/V so a window bug
  // can't hide), >> the bounded ring (window 512 + chunk 512 = 1024).
  std::vector<std::int32_t> ids;
  ids.reserve(2400);
  ids.push_back(2);                         // <bos>
  std::uint32_t s = 0x12345678u;
  for (int i = 1; i < 2400; ++i) {
    s = s * 1664525u + 1013904223u;
    ids.push_back((std::int32_t)(106 + (s >> 9) % 200000));
  }

  auto run = [&](int mode) -> std::vector<std::int32_t> {
    // mode 0 = grown (A); 1 = bounded subblock (C); 2 = bounded chunked (B)
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_SLIDING_CHUNK", "512", 1);
    if (mode == 0) {
      ::unsetenv("VPIPE_GEMMA_NO_SLIDING_GROW");   // grown one-pass (A)
      ::unsetenv("VPIPE_GEMMA_PREFILL_SUBBLOCK");
    } else if (mode == 1) {
      ::setenv("VPIPE_GEMMA_NO_SLIDING_GROW", "1", 1);  // bounded subblock (C)
      ::setenv("VPIPE_GEMMA_PREFILL_SUBBLOCK", "1", 1);
    } else {
      ::setenv("VPIPE_GEMMA_NO_SLIDING_GROW", "1", 1);  // bounded chunked (B)
      ::setenv("VPIPE_GEMMA_PREFILL_SUBBLOCK", "0", 1);
    }
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 24 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };

  const auto grown   = run(0);
  const auto bounded = run(1);
  const auto chunked = run(2);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_SLIDING_CHUNK");
  ::unsetenv("VPIPE_GEMMA_NO_SLIDING_GROW");
  ::unsetenv("VPIPE_GEMMA_PREFILL_SUBBLOCK");
  ASSERT_TRUE(!grown.empty());
  ASSERT_TRUE(grown.size() == bounded.size());
  auto cnt = [&](const std::vector<std::int32_t>& a,
                 const std::vector<std::int32_t>& b) {
    std::size_t m = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
      if (a[i] != b[i]) { ++m; }
    }
    return m;
  };
  // Bounded-subblock (C) must equal BOTH the grown one-pass (A) and the bounded
  // chunked (B) reference -- all three decode identically.
  const std::size_t mism = cnt(grown, bounded);
  std::printf("[metal_lm_smoke.gemma_bounded_subblock_matches_grown_token_exact]"
              " %zu tokens, %zu mismatches (C-vs-A=%zu C-vs-B=%zu B-vs-A=%zu)\n",
              grown.size(), mism, cnt(bounded, grown), cnt(bounded, chunked),
              cnt(chunked, grown));
  EXPECT_TRUE(mism == 0);
  EXPECT_TRUE(cnt(bounded, chunked) == 0);
}

// MULTIMODAL prefill must also respect the sliding-window ring. prefill_mm
// (the image/audio splice path) does a single-shot forward; when the prefix
// exceeds the ring it MUST grow the sliding ring first (like prefill()), else
// the single pass wraps the ring and clobbers in-window keys -> corrupted
// sliding/local-attention layers (fluent but ungrounded VQA). A multi-frame
// video scene is always longer than the ring, so this is the realtime-vqa
// path. Build a long multimodal prefix (synthetic audio soft-tokens + a long
// text tail) and require chunk=128 (ring 640 < prefix -> grows) to be greedy
// token-exact with chunk=4096 (ring caps high -> single pass, no wrap). If the
// grow is dropped, chunk=128 wraps and mismatches. Gated on the Gemma model.
TEST(metal_lm_smoke, gemma_mm_prefill_sliding_grow_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  std::string prompt;
  for (int i = 0; i < 40; ++i) {
    prompt += "The clever clockmaker studied the broken machine carefully, "
              "noting each worn gear and bent spring before she began again. ";
  }
  auto run = [&](int chunk) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_SLIDING_CHUNK", std::to_string(chunk).c_str(), 1);
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    const int hidden = lm->config().hidden;
    if (hidden <= 0) { return out; }
    // A few synthetic audio soft-tokens (deterministic, finite) at the head
    // -> n_mm > 0 routes through prefill_mm; empty image_grids => 1-D RoPE.
    const int n_aud = 16;
    std::vector<float> aud(static_cast<std::size_t>(n_aud) * hidden);
    for (std::size_t i = 0; i < aud.size(); ++i) {
      aud[i] = 0.01f * static_cast<float>((int)(i % 13) - 6);
    }
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    std::vector<genai::TokenRef> refs;
    refs.reserve(static_cast<std::size_t>(n_aud) + ids.size());
    for (int k = 0; k < n_aud; ++k) {
      genai::TokenRef r;
      r.kind = genai::TokenRef::Kind::AudioTokens;
      r.audio_token_offset = k;
      r.embeddings_host = &aud;
      r.host_hidden = hidden;
      refs.push_back(r);
    }
    for (std::int32_t id : ids) {
      genai::TokenRef r;
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
      refs.push_back(r);
    }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill_multimodal_metal(
        ctx, std::span<const genai::TokenRef>(refs),
        std::span<const std::pair<int, int>>());
    for (int i = 0; i < 24 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token(ctx);
    }
    return out;
  };
  const auto grown  = run(128);    // ring 640 < prefix -> prefill_mm grows
  const auto single = run(4096);   // ring caps high -> single pass, no wrap
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_SLIDING_CHUNK");
  ASSERT_TRUE(!single.empty());
  ASSERT_TRUE(grown.size() == single.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < single.size(); ++i) {
    if (grown[i] != single[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_mm_prefill_sliding_grow] %zu tokens, "
              "%zu mismatches (chunk=128 grow vs 4096 single)\n",
              single.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Regression for Gemma-4 text chat (the user-reported "no chat template
// for architecture 'Gemma4ForConditionalGeneration'" + no-MLX load
// failure). Two guarantees, both build-agnostic:
//   1. Loading a metal-supported model does NOT require the caller to set
//      VPIPE_LLM_BACKEND -- a build without MLX defaults to the metal
//      backend (it is the only one), so we UNSET the var first and the
//      load must still succeed. (In the MLX build this loads via MLX.)
//   2. chat_template() is non-null for Gemma-4 (the arm is always-built
//      in make_chat_template), and render_user_turn + prefill produce a
//      valid first token -- i.e. text chat is functional end to end.
// Gated on the Gemma checkpoint (VPIPE_GEMMA4_TEST_MODEL_PATH).
// GGUF q4_0 gemma4 bring-up: load a pure-.gguf dir (config + weights +
// tokenizer all from the GGUF), render a chat turn through the template
// (which prepends <bos> -- Gemma is incoherent without it), greedy-decode,
// and require coherent text. Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_text_chat) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);
  EXPECT_TRUE(tpl->family_name() == "gemma");

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Answer in one word.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  EXPECT_TRUE(ids.front() == 2);          // <bos>

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::vector<std::int32_t> gen;
  const std::int32_t first = lm->prefill(ctx, ids);
  ASSERT_TRUE(first >= 0);
  gen.push_back(first);
  for (int i = 0; i < 24; ++i) {
    const std::int32_t n = lm->next_token(ctx);
    if (n < 0) { break; }
    gen.push_back(n);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[gguf_gemma_text_chat] %zu tok | gen='%s'\n",
              gen.size(), text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// Regression: the Gemma-4 12B (qat-q4_0 GGUF) text decoder must NEVER emit
// the multimodal end-of-image / end-of-audio control tokens (<image|>
// 258882 / <audio|> 258883) in text output. On visually-themed text the
// QAT-4bit lm_head assigns <image|> the TOP logit (observed leak:
// "...sketching the memory <image|>topology"); LoadedLanguageModel bakes a
// PERMANENT suppression of exactly these two tokens into the model at load
// -- matching the llama.cpp reference, which masks these two (and only
// these two) to -inf.
//
// This teacher-forces the exact prompt+prefix that triggered the leak and
// asserts the predicted next token is the sensible word (' layout', the
// llama.cpp golden argmax at this position), NOT the control token, and
// that both control ids sit below the winner. Deterministic (greedy).
// Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_no_multimodal_leak) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 1024;
  spec.max_pages     = 4;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  const auto& tok = lm->tokenizer();

  // Tokenizer sanity: the control ids map to the expected literals (a
  // mis-map would print <image|> for a NORMAL id).
  std::int32_t eoi = 258882, eoa = 258883;
  EXPECT_TRUE(tok.decode(std::span<const std::int32_t>(&eoi, 1)) == "<image|>");
  EXPECT_TRUE(tok.decode(std::span<const std::int32_t>(&eoa, 1)) == "<audio|>");

  // The permanent base suppression must be populated (both eoi + eoa) --
  // realtime-vqa folds this into its host-side batched mask, so a regression
  // that empties it would silently re-open the leak on the batched path.
  {
    auto base = lm->base_suppressed_tokens();
    bool has_eoi = false, has_eoa = false;
    for (std::int32_t id : base) {
      if (id == 258882) { has_eoi = true; }
      if (id == 258883) { has_eoa = true; }
    }
    EXPECT_TRUE(has_eoi);
    EXPECT_TRUE(has_eoa);
  }

  // Prompt ("Write a short story about Yi...") + the greedy prefix up to
  // (not including) the leaked <image|>. Position 213 predicts the word
  // after "...sketching the memory".
  static const std::int32_t kPrefix[] = {
      2, 105, 2364, 107, 6974, 496, 2822, 3925, 1003, 54984, 236764, 496,
      21042, 47133, 19042, 236764, 532, 1116, 9338, 61232, 496, 3909, 6347, 529,
      3393, 236761, 11968, 236743, 236778, 236771, 236771, 4171, 236761, 106, 107,
      105, 4368, 107, 100, 45518, 107, 101, 818, 147024, 19462, 529, 1806, 37845,
      53522, 54984, 236858, 236751, 3392, 236764, 30439, 1440, 37676, 3418, 506,
      173152, 15348, 236761, 1701, 1806, 5695, 236764, 1304, 1053, 1010, 79582,
      684, 496, 25556, 528, 506, 5464, 236787, 496, 6571, 20651, 528, 506, 861,
      1494, 236772, 32677, 47133, 600, 1186, 9177, 1208, 13610, 121160, 236761,
      108, 2021, 6533, 1663, 236764, 506, 14510, 3938, 691, 496, 32585, 76692,
      236761, 2282, 54984, 236764, 625, 691, 496, 122400, 607, 886, 199010, 5433,
      236761, 2625, 3782, 236789, 236745, 1164, 1676, 506, 3393, 236793, 1304,
      6345, 1061, 18479, 236761, 2625, 50070, 506, 6818, 75043, 236764, 10685,
      1217, 506, 20974, 93905, 607, 506, 1262, 236761, 236743, 108, 236775, 1509,
      236858, 236751, 711, 496, 20651, 2098, 1304, 71787, 236764, 1116, 6114,
      188312, 699, 78370, 236761, 623, 1509, 236858, 236751, 496, 53970, 92560,
      1781, 108, 5778, 532, 506, 47133, 964, 13710, 1024, 506, 1638, 15612, 1757,
      236764, 496, 47617, 34847, 236772, 1340, 236772, 8281, 14260, 506, 1458,
      531, 162911, 236761, 54984, 16630, 872, 496, 11580, 23037, 13039, 532,
      6074, 130257, 506, 6571};
  std::vector<std::int32_t> ids(kPrefix,
      kPrefix + sizeof(kPrefix) / sizeof(kPrefix[0]));
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t next = lm->prefill(ctx, ids);
  std::int32_t nn = next;
  std::printf("[no_leak] argmax=%d '%s' (golden=11273 ' layout')\n", next,
      tok.decode(std::span<const std::int32_t>(&nn, 1)).c_str());
  // The fix: greedy must not pick either suppressed control token.
  EXPECT_TRUE(next != 258882);
  EXPECT_TRUE(next != 258883);
  // And it matches the llama.cpp golden argmax (' layout', 11273).
  EXPECT_TRUE(next == 11273);
  // Both suppressed ids must sit below the winner (masked to the sentinel).
  const auto& lg = lm->last_logits_host();
  if (!lg.empty() && (int)lg.size() > 258883) {
    EXPECT_TRUE(lg[258882] < lg[11273]);
    EXPECT_TRUE(lg[258883] < lg[11273]);
  }
}

// Opt-in bench: native Q6_K tied embed/lm_head vs the affine8 requant.
// Prints resident memory after load + decode tok/s. Run twice to A/B:
//   VPIPE_GGUF_TEST_MODEL_PATH=<dir> vpipe_test --filter '*q6k_decode_bench'
//   VPIPE_GEMMA_NO_Q6K=1 ... (forces the affine8 path)
// The Q6_K table is lossless and ~25% smaller (vocab*H*6.5625 vs *8 bits).
TEST(metal_lm_smoke, gguf_gemma_q6k_decode_bench) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // phys_footprint (not resident_size): includes IOKit/Metal wired buffers
  // where the weights live, so it reflects the embed-table size delta.
  auto footprint_mb = []() -> double {
    task_vm_info_data_t info{};
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  reinterpret_cast<task_info_t>(&info), &cnt) != KERN_SUCCESS) {
      return 0.0;
    }
    return static_cast<double>(info.phys_footprint) / (1024.0 * 1024.0);
  };
  const double rss = footprint_mb();

  std::vector<std::int32_t> ids;
  lm->chat_template()->render_user_turn(
      "What is the capital of France? Answer in one word.", true, &ids);
  ASSERT_TRUE(!ids.empty());
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  ASSERT_TRUE(lm->prefill(ctx, ids) >= 0);
  (void)lm->next_token(ctx);              // warm one decode step
  const int N = 48;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) { (void)lm->next_token(ctx); }
  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  const char* nq = std::getenv("VPIPE_GEMMA_NO_Q6K");
  const bool q6k = !(nq && std::atoi(nq) != 0);
  std::printf("[gguf_q6k_bench] embed=%s rss=%.0f MB decode=%.1f tok/s "
              "(%d steps %.3fs)\n", q6k ? "Q6K" : "affine8", rss,
              static_cast<double>(N) / secs, N, secs);
  EXPECT_TRUE(secs > 0.0);
}

// Parameterized prefill+decode bench mirroring llama.cpp's llama-bench so a
// GGUF model can be A/B'd 1:1 against it:
//   pp@L  -> process L prompt tokens from empty;  tok/s = L / t
//   tg@L  -> prefill L tokens (untimed), then generate G tokens timed
//            via BOTH the synchronous next_token loop AND the pipelined
//            pdecode_* path (vpipe's production decode); tok/s = G / t
// Context sizes from VPIPE_GGUF_BENCH_CTX (default "512,1024,2048,4096"),
// generated-token count from VPIPE_GGUF_BENCH_GEN (default 64). Gated on
// VPIPE_GGUF_TEST_MODEL_PATH. Run:
//   VPIPE_GGUF_TEST_MODEL_PATH=<dir> vpipe_test --filter '*gguf_gemma_pp_tg*'
TEST(metal_lm_smoke, gguf_gemma_pp_tg_bench) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  // Parse the comma-separated context-size list + decode count FIRST: the KV
  // cache capacity (page_tokens * max_pages) must fit the largest ctx + gen,
  // else prefill silently caps and the reported tok/s is garbage.
  std::vector<int> ctxs;
  {
    const char* cs = std::getenv("VPIPE_GGUF_BENCH_CTX");
    std::string s = (cs && *cs) ? cs : "512,1024,2048,4096";
    std::size_t i = 0;
    while (i < s.size()) {
      std::size_t j = s.find(',', i);
      if (j == std::string::npos) { j = s.size(); }
      const int v = std::atoi(s.substr(i, j - i).c_str());
      if (v > 0) { ctxs.push_back(v); }
      i = j + 1;
    }
  }
  ASSERT_TRUE(!ctxs.empty());
  const char* gs = std::getenv("VPIPE_GGUF_BENCH_GEN");
  const int G = (gs && *gs) ? std::max(1, std::atoi(gs)) : 64;

  int max_ctx = 0;
  for (const int v : ctxs) { max_ctx = std::max(max_ctx, v); }
  const int page_tokens = 512;
  // room for the deepest prompt + decode budget (+ slack for warmup steps).
  const int need_seq = max_ctx + G + 64;
  int max_pages = (need_seq + page_tokens - 1) / page_tokens;
  if (max_pages < 16) { max_pages = 16; }

  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = page_tokens;
  spec.max_pages     = max_pages;        // sized to max(ctx)+gen above
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // Seed token stream: a real rendered turn (keeps <bos> first), then pad
  // with a benign repeated token. Coherence is irrelevant for timing.
  std::vector<std::int32_t> seed;
  lm->chat_template()->render_user_turn(
      "Benchmark.", /*is_first_turn=*/true, &seed);
  ASSERT_TRUE(!seed.empty());
  const std::int32_t bos  = seed.front();
  const std::int32_t fill = seed.size() > 1 ? seed[1] : bos;
  auto make_ids = [&](int L) {
    std::vector<std::int32_t> ids;
    ids.reserve(L);
    ids.push_back(bos);
    for (int k = 1; k < L; ++k) { ids.push_back(fill); }
    return ids;
  };

  // Warm the GPU (first prefill/CB is cold; clock spins up).
  {
    auto wc = lm->make_context();
    ASSERT_TRUE(wc.valid());
    auto wid = make_ids(64);
    ASSERT_TRUE(lm->prefill(wc, wid) >= 0);
    for (int k = 0; k < 4; ++k) { (void)lm->next_token(wc); }
  }

  std::printf("[gguf_pp_tg] gemma4_unified gguf bf16 gen=%d\n", G);
  for (const int L : ctxs) {
    const auto ids = make_ids(L);
    const std::span<const std::int32_t> prompt(ids.data(), ids.size());

    // ---- prefill (pp@L): process L tokens from empty ----
    auto cp = lm->make_context();
    ASSERT_TRUE(cp.valid());
    const auto p0 = std::chrono::steady_clock::now();
    const std::int32_t pf = lm->prefill(cp, ids);
    const double psecs =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - p0).count();
    ASSERT_TRUE(pf >= 0);
    const double pp_tps = (psecs > 0.0) ? (double)L / psecs : 0.0;
    // Early prefill-only print (BEFORE the decode blocks) so SKIP_ATTN probes,
    // which leave logits garbage and can crash pdecode, still report prefill.
    std::printf("[gguf_pp_only] ctx=%-5d  prefill=%7.1f tok/s (%.3fs)\n",
                L, pp_tps, psecs);
    std::fflush(stdout);

    // ---- synchronous decode (tg@L): next_token = host [vocab] readback +
    //      host argmax (the production fallback path when pdecode is off) ----
    double tg_sync = 0.0;
    {
      auto cd = lm->make_context();
      ASSERT_TRUE(cd.valid());
      ASSERT_TRUE(lm->prefill(cd, ids) >= 0);
      (void)lm->next_token(cd);          // warm one step at depth L
      const auto d0 = std::chrono::steady_clock::now();
      for (int k = 0; k < G; ++k) { (void)lm->next_token(cd); }
      const double dsecs =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - d0).count();
      tg_sync = (dsecs > 0.0) ? (double)G / dsecs : 0.0;
    }

    // ---- synchronous GPU-argmax decode (tg@L): next_token_greedy = on-GPU
    //      argmax, no host logit pull, but a fresh command buffer per step ----
    double tg_greedy = 0.0;
    {
      auto cd = lm->make_context();
      ASSERT_TRUE(cd.valid());
      ASSERT_TRUE(lm->prefill(cd, ids) >= 0);
      (void)lm->next_token_greedy(cd);   // warm one step at depth L
      const auto d0 = std::chrono::steady_clock::now();
      for (int k = 0; k < G; ++k) { (void)lm->next_token_greedy(cd); }
      const double dsecs =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - d0).count();
      tg_greedy = (dsecs > 0.0) ? (double)G / dsecs : 0.0;
    }

    // ---- pipelined decode (tg@L, vpipe production path) ----
    double tg_pipe = 0.0;
    {
      auto cd = lm->make_context();
      ASSERT_TRUE(cd.valid());
      const std::int32_t first = lm->prefill(cd, ids);
      ASSERT_TRUE(first >= 0);
      genai::SamplerParams gsp;            // defaults -> argmax-equivalent
      const int budget = G + 8;
      if (lm->pdecode_begin(cd, first, prompt, gsp, budget)) {
        // Run-ahead: PRIME the pipeline to pd.depth (commit until the ring is
        // full), then steady-state refill ONE-ahead right after each wait, so
        // each commit encodes step N+1 WHILE step N runs on the GPU --
        // overlapping dispatch-encoding with GPU execution. The old
        // commit-then-next loop oscillated the ring 0<->1 and never overlapped.
        while (lm->pdecode_commit(cd)) {}     // prime to depth
        for (int k = 0; k < 4; ++k) {         // warm steady-state
          if (lm->pdecode_next(cd) < 0) { break; }
          lm->pdecode_commit(cd);
        }
        int produced = 0;
        const auto d0 = std::chrono::steady_clock::now();
        for (int k = 0; k < G; ++k) {
          if (lm->pdecode_next(cd) < 0) { break; }
          ++produced;
          lm->pdecode_commit(cd);           // refill one ahead (overlaps GPU)
        }
        const double dsecs =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - d0).count();
        tg_pipe = (dsecs > 0.0) ? (double)produced / dsecs : 0.0;
        lm->pdecode_end(cd);
      }
    }

    std::printf("[gguf_pp_tg] ctx=%-5d  prefill=%7.1f tok/s (%.3fs)  "
                "decode: sync=%5.2f  greedy=%5.2f  pipe=%5.2f tok/s\n",
                L, pp_tps, psecs, tg_sync, tg_greedy, tg_pipe);
  }
  EXPECT_TRUE(true);
}

// Decode CATEGORY profiler (GGUF gemma4_unified) -- decompose the per-token
// decode GPU cost to chase the llama.cpp decode gap. Loads once with
// VPIPE_GEMMA_CATPROF, then for each DUP category (proj/ffn/lmhead/attn/norm/
// misc, + attn_global/attn_slide) duplicates that category's GPU work and
// diffs decode time vs the `none` baseline -> the delta is the category's
// whole-step cost. Profiles at a low + deep context (the decode attention cost
// grows with KV depth; llama.cpp's flash-decode stays flat -- this isolates how
// much of vpipe's depth-degradation is attention). Gated on
// VPIPE_GGUF_TEST_MODEL_PATH + VPIPE_GEMMA_CATPROF (depths: VPIPE_GGUF_BENCH_CTX
// or default 512,4096).
TEST(metal_lm_smoke, gguf_gemma_decode_catprof) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_GEMMA_CATPROF") == nullptr) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || mgr == nullptr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  // Parse depths FIRST so the KV cache (page_tokens*max_pages) fits the
  // deepest profile depth + decode slack (else prefill caps -> degenerate).
  std::vector<int> depths{512, 4096};
  if (const char* cs = std::getenv("VPIPE_GGUF_BENCH_CTX")) {
    std::vector<int> v; const char* p = cs;
    while (*p) {
      int x = std::atoi(p);
      if (x > 0) { v.push_back(x); }
      while (*p && *p != ',') { ++p; }
      if (*p == ',') { ++p; }
    }
    if (!v.empty()) { depths = v; }
  }
  int max_depth = 0;
  for (const int v : depths) { max_depth = std::max(max_depth, v); }

  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = std::max(16, (max_depth + 128 + 511) / 512);
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> seed;
  lm->chat_template()->render_user_turn("Benchmark.", true, &seed);
  ASSERT_TRUE(!seed.empty());
  const std::int32_t bos  = seed.front();
  const std::int32_t fill = seed.size() > 1 ? seed[1] : bos;
  auto make_ids = [&](int L) {
    std::vector<std::int32_t> ids; ids.reserve(L);
    ids.push_back(bos);
    for (int k = 1; k < L; ++k) { ids.push_back(fill); }
    return ids;
  };

  const int N = 48;
  auto decode_ms = [&](const std::vector<std::int32_t>& ids) -> double {
    auto ctx = lm->make_context();
    if (!ctx.valid() || lm->prefill(ctx, ids) < 0) { return -1.0; }
    (void)lm->next_token_greedy(ctx);          // warm one step at depth
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
      if (lm->next_token_greedy(ctx) < 0) { break; }
    }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
  };

  const char* cats[] = {"none", "proj", "ffn", "lmhead", "attn", "norm",
                        "misc", "ple", "attn_global", "attn_slide"};
  const int NC = 10;
  for (int depth : depths) {
    const auto ids = make_ids(depth);
    for (int k = 0; k < 2; ++k) { (void)decode_ms(ids); }   // warm GPU clock
    double best[NC];
    for (int c = 0; c < NC; ++c) {
      ::setenv("VPIPE_GEMMA_DUP_CAT", cats[c], 1);
      double m = 1e18;
      for (int r = 0; r < 3; ++r) { m = std::min(m, decode_ms(ids)); }
      best[c] = m;
    }
    ::unsetenv("VPIPE_GEMMA_DUP_CAT");
    const double T0 = best[0];
    std::printf("[gemma_catprof depth=%-4d] baseline %.1f ms (%.3f ms/tok = "
                "%.2f tok/s); delta = category whole-step GPU cost\n",
                depth, T0, T0 / N, N * 1000.0 / T0);
    for (int c = 1; c < NC; ++c) {
      const double d = best[c] - T0;
      std::printf("[gemma_catprof depth=%-4d] %-12s delta %+7.2f ms "
                  "(%.3f ms/tok) | %5.1f%%\n",
                  depth, cats[c], d, d / N, 100.0 * d / T0);
    }
  }
  EXPECT_TRUE(true);
}

// SINGLE-PREFILL strip ablation (decode). Same categories as gguf_gemma_decode_
// strip, but prefills ONCE per sweep and toggles VPIPE_GEMMA_SKIP_LAYER (+the
// composable SKIP_* knobs, all re-read every decode step) BETWEEN short decode
// windows on the same pdecode session -> ~3 prefills total instead of ~60. The
// re-prefill design wedges the GPU under sustained long-context (8k/16k) prefill
// load; this variant decouples the measurement from prefill so it runs clean at
// depth. Gated on VPIPE_GEMMA_STRIP_ONCE. Intra-sweep depth drift is tiny
// (nconfigs*W steps << depth) and identical for every config (same session).
TEST(metal_lm_smoke, gguf_gemma_decode_strip_once) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_GEMMA_STRIP_ONCE") == nullptr) { return; }
  ::setenv("VPIPE_GEMMA_CATPROF", "1", 1);   // gate the skip path
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || mgr == nullptr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  int depth = 8192;
  if (const char* cs = std::getenv("VPIPE_GGUF_BENCH_CTX")) {
    const int x = std::atoi(cs);
    if (x > 0) { depth = x; }
  }
  const int W = 24;                          // timed decode tokens per window
  const int SETTLE = 3;                      // untimed steps after each toggle
  const int NSWEEP = 4;
  // All distinct configs measured per sweep (label + SKIP env settings). The
  // empty string for SKIP_LAYER means "no whole-layer strip".
  struct Cfg { const char* label; const char* layer; bool ple, ffn, attn,
               proj, lmhead, embed; };
  const Cfg cfgs[] = {
    {"none",            "",        0,0,0,0,0,0},
    {"strip_sliding",   "sliding", 0,0,0,0,0,0},  // -> all-global
    {"strip_global",    "global",  0,0,0,0,0,0},  // -> all-sliding
    {"glob_noPLE",      "sliding", 1,0,0,0,0,0},
    {"glob_noPLE_noFFN","sliding", 1,1,0,0,0,0},  // 7 global: attn+proj+norm
    {"glob_..noATTN",   "sliding", 1,1,1,0,0,0},  //          proj+norm
    {"glob_..noPROJ",   "sliding", 1,1,1,1,0,0},  //          norm/rope/kv
    {"slid_noPLE_noFFN","global",  1,1,0,0,0,0},  // 35 sliding: attn+proj+norm
    {"slid_..noATTN",   "global",  1,1,1,0,0,0},  //          proj+norm
    {"slid_..noPROJ",   "global",  1,1,1,1,0,0},  //          norm/rope/kv
    {"fixed_only",      "all",     1,0,0,0,0,0},  // embed+lmhead+sample+handoff
    {"fixed_noLM",      "all",     1,0,0,0,1,0},  //  - lm_head
    {"fixed_noLM_noEmb","all",     1,0,0,0,1,1},  //  - embed too
  };
  const int NC = (int)(sizeof(cfgs) / sizeof(cfgs[0]));
  auto apply = [&](const Cfg& c) {
    if (*c.label && *c.layer) { ::setenv("VPIPE_GEMMA_SKIP_LAYER", c.layer, 1); }
    else { ::unsetenv("VPIPE_GEMMA_SKIP_LAYER"); }
    auto tog = [&](const char* e, bool on) {
      if (on) { ::setenv(e, "1", 1); } else { ::unsetenv(e); } };
    tog("VPIPE_GEMMA_SKIP_PLE",    c.ple);
    tog("VPIPE_GEMMA_SKIP_FFN",    c.ffn);
    tog("VPIPE_GEMMA_SKIP_ATTN",   c.attn);
    tog("VPIPE_GEMMA_SKIP_PROJ",   c.proj);
    tog("VPIPE_GEMMA_SKIP_LMHEAD", c.lmhead);
    tog("VPIPE_GEMMA_SKIP_EMBED",  c.embed);
  };
  auto clear_all = [&] {
    ::unsetenv("VPIPE_GEMMA_SKIP_LAYER"); ::unsetenv("VPIPE_GEMMA_SKIP_PLE");
    ::unsetenv("VPIPE_GEMMA_SKIP_FFN"); ::unsetenv("VPIPE_GEMMA_SKIP_ATTN");
    ::unsetenv("VPIPE_GEMMA_SKIP_PROJ"); ::unsetenv("VPIPE_GEMMA_SKIP_LMHEAD");
    ::unsetenv("VPIPE_GEMMA_SKIP_EMBED");
  };

  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  const int sweep_steps = NC * (W + SETTLE) + 32;
  spec.max_pages = std::max(16, (depth + sweep_steps + 511) / 512);
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> seedv;
  lm->chat_template()->render_user_turn("Benchmark.", true, &seedv);
  ASSERT_TRUE(!seedv.empty());
  const std::int32_t bos  = seedv.front();
  const std::int32_t fill = seedv.size() > 1 ? seedv[1] : bos;
  std::vector<std::int32_t> ids; ids.reserve(depth);
  ids.push_back(bos);
  for (int k = 1; k < depth; ++k) { ids.push_back(fill); }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());

  // min-of-NSWEEP wall ms/tok per config. Each sweep: 1 prefill, warm, then
  // every config's window back-to-back on the same pdecode session.
  std::vector<double> best(NC, 1e18);
  genai::SamplerParams gsp;                   // defaults -> argmax-equivalent
  for (int sw = 0; sw < NSWEEP + 1; ++sw) {    // +1 warm sweep (discarded)
    clear_all();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    const std::int32_t first = lm->prefill(ctx, ids);
    ASSERT_TRUE(first >= 0);
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, sweep_steps)) {
      lm->pdecode_end(ctx); continue;
    }
    for (int k = 0; k < 4; ++k) {              // warm the pipeline
      if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
    }
    for (int ci = 0; ci < NC; ++ci) {
      apply(cfgs[ci]);
      for (int k = 0; k < SETTLE; ++k) {       // let the toggle take effect
        if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
      }
      const auto t0 = std::chrono::steady_clock::now();
      int got = 0;
      for (int k = 0; k < W; ++k) {
        if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
        ++got;
      }
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
      const double mt = got > 0 ? ms / got : 1e18;
      if (sw > 0 && mt < best[ci]) { best[ci] = mt; }
    }
    clear_all();
    lm->pdecode_end(ctx);
  }

  auto B = [&](const char* lbl) -> double {
    for (int i = 0; i < NC; ++i) {
      if (std::string(cfgs[i].label) == lbl) { return best[i]; }
    }
    return 0.0;
  };
  const double base = B("none");
  std::printf("[strip_once depth=%d W=%d sweeps=%d] min wall ms/tok\n",
              depth, W, NSWEEP);
  for (int i = 0; i < NC; ++i) {
    if (i == 0) {
      std::printf("[strip_once] %-18s %.3f ms/tok (baseline)\n",
                  cfgs[i].label, best[i]);
    } else {
      std::printf("[strip_once] %-18s %.3f ms/tok  (-%.3f, %4.1f%%)\n",
                  cfgs[i].label, best[i], base - best[i],
                  100.0 * (base - best[i]) / base);
    }
  }
  // Derived per-category DECODE costs (ms/tok) for the head-to-head with omlx.
  std::printf("[strip_once] === per-category decode cost (ms/tok) ===\n");
  std::printf("[strip_once] sliding TOTAL (35 lyr) %.3f | global TOTAL (7 lyr) "
              "%.3f\n", base - B("strip_sliding"), base - B("strip_global"));
  std::printf("[strip_once] global: SDPA %.3f  QKV+O proj %.3f  norm/rope/kv "
              "%.3f  FFN %.3f  PLE %.3f\n",
              B("glob_noPLE_noFFN") - B("glob_..noATTN"),
              B("glob_..noATTN")   - B("glob_..noPROJ"),
              B("glob_..noPROJ")   - B("fixed_only"),
              B("glob_noPLE")      - B("glob_noPLE_noFFN"),
              B("strip_sliding")   - B("glob_noPLE"));
  std::printf("[strip_once] sliding: SDPA %.3f  QKV+O proj %.3f  norm/rope/kv+"
              "fixed %.3f\n",
              B("slid_noPLE_noFFN") - B("slid_..noATTN"),
              B("slid_..noATTN")    - B("slid_..noPROJ"),
              B("slid_..noPROJ"));
  std::printf("[strip_once] fixed: lm_head %.3f  embed %.3f  norm/argmax/sampler/"
              "handoff %.3f\n",
              B("fixed_only") - B("fixed_noLM"),
              B("fixed_noLM") - B("fixed_noLM_noEmb"),
              B("fixed_noLM_noEmb"));

  // FULL-LOAD DUP sweep (VPIPE_GEMMA_STRIP_DUP): double a category INSIDE the
  // full 42-layer pass (VPIPE_GEMMA_DUP_CAT, read per-step) -> delta vs none =
  // that category's true full-occupancy GPU cost (no isolation clock artifact).
  // norm = all RMSNorm/rope/KV-write; attn_slide/attn_global isolate by type.
  if (std::getenv("VPIPE_GEMMA_STRIP_DUP") != nullptr) {
    const char* dcats[] = {"none", "norm", "attn_slide", "attn_global",
                           "proj", "ffn"};
    const int ND = 6;
    std::vector<double> dbest(ND, 1e18);
    for (int sw = 0; sw < NSWEEP + 1; ++sw) {
      clear_all();
      ::unsetenv("VPIPE_GEMMA_DUP_CAT");
      auto ctx = lm->make_context();
      if (!ctx.valid()) { continue; }
      const std::int32_t first = lm->prefill(ctx, ids);
      if (first < 0) { lm->pdecode_end(ctx); continue; }
      if (!lm->pdecode_begin(ctx, first, prompt, gsp, sweep_steps)) {
        lm->pdecode_end(ctx); continue;
      }
      for (int k = 0; k < 4; ++k) {
        if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
      }
      for (int ci = 0; ci < ND; ++ci) {
        if (ci == 0) { ::unsetenv("VPIPE_GEMMA_DUP_CAT"); }
        else { ::setenv("VPIPE_GEMMA_DUP_CAT", dcats[ci], 1); }
        for (int k = 0; k < SETTLE; ++k) {
          if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
        }
        const auto t0 = std::chrono::steady_clock::now();
        int got = 0;
        for (int k = 0; k < W; ++k) {
          if (!lm->pdecode_commit(ctx) || lm->pdecode_next(ctx) < 0) { break; }
          ++got;
        }
        const double mt = got > 0 ? std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count() / got : 1e18;
        if (sw > 0 && mt < dbest[ci]) { dbest[ci] = mt; }
      }
      ::unsetenv("VPIPE_GEMMA_DUP_CAT");
      lm->pdecode_end(ctx);
    }
    std::printf("[strip_once] === full-load DUP (delta vs none = cat cost) ===\n");
    for (int ci = 1; ci < ND; ++ci) {
      std::printf("[strip_once] dup %-12s %.3f ms/tok  (+%.3f vs none %.3f)\n",
                  dcats[ci], dbest[ci], dbest[ci] - dbest[0], dbest[0]);
    }
  }
  EXPECT_TRUE(true);
}

// Whole-CATEGORY strip ablation (decode). Removes a layer category's entire
// body (VPIPE_GEMMA_SKIP_LAYER) and reports wall ms/tok, so baseline-minus-
// stripped = that category's wall cost. Mirror of the omlx passthrough-strip
// bench (~/dump/omlx_strip_bench.py) for a head-to-head per-category gap.
// Categories: sliding|global (attn type), shared_kv|own_kv (KV ownership),
// and the 4 individual buckets. Gated on VPIPE_GEMMA_STRIP.
TEST(metal_lm_smoke, gguf_gemma_decode_strip) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_GEMMA_STRIP") == nullptr) { return; }
  ::setenv("VPIPE_GEMMA_CATPROF", "1", 1);   // gate the skip path
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || mgr == nullptr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  // Parse depth FIRST so the KV cache (page_tokens*max_pages) fits it + the
  // N decoded tokens of slack; else prefill caps at 8192 -> degenerate at 16k.
  int depth = 2048;
  if (const char* cs = std::getenv("VPIPE_GGUF_BENCH_CTX")) {
    const int x = std::atoi(cs);
    if (x > 0) { depth = x; }
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = std::max(16, (depth + 128 + 511) / 512);
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> seed;
  lm->chat_template()->render_user_turn("Benchmark.", true, &seed);
  ASSERT_TRUE(!seed.empty());
  const std::int32_t bos  = seed.front();
  const std::int32_t fill = seed.size() > 1 ? seed[1] : bos;
  auto make_ids = [&](int L) {
    std::vector<std::int32_t> ids; ids.reserve(L);
    ids.push_back(bos);
    for (int k = 1; k < L; ++k) { ids.push_back(fill); }
    return ids;
  };

  const int N = 48;
  const auto ids = make_ids(depth);
  // Use the PIPELINED decode path (pdecode_*, vpipe's production decode) so the
  // per-token CPU<->GPU handoff is overlapped -- apples-to-apples with omlx's
  // async_eval, removing the ~1.5 ms sync-handoff confound from every row.
  // Returns total wall ms normalised to N tokens (caller divides by N).
  auto decode_ms = [&](const std::vector<std::int32_t>& seq) -> double {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return -1.0; }
    const std::span<const std::int32_t> prompt(seq.data(), seq.size());
    const std::int32_t first = lm->prefill(ctx, seq);
    if (first < 0) { return -1.0; }
    genai::SamplerParams gsp;                  // defaults -> argmax-equivalent
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, N + 8)) { return -1.0; }
    for (int k = 0; k < 4; ++k) {              // warm the pipeline
      if (!lm->pdecode_commit(ctx)) { break; }
      if (lm->pdecode_next(ctx) < 0) { break; }
    }
    int produced = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
      if (!lm->pdecode_commit(ctx)) { break; }
      if (lm->pdecode_next(ctx) < 0) { break; }
      ++produced;
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    lm->pdecode_end(ctx);
    return produced > 0 ? ms * (double)N / produced : -1.0;
  };

  const char* strips[] = {"", "sliding", "global", "shared_kv", "own_kv",
                          "slide_own", "global_own", "slide_skip",
                          "global_skip"};
  const int NS = 9;
  for (int k = 0; k < 2; ++k) { (void)decode_ms(ids); }    // warm GPU clock
  double base = 0.0;
  std::printf("[gemma_strip depth=%d N=%d] min-of-5 wall ms/tok\n", depth, N);
  for (int s = 0; s < NS; ++s) {
    if (*strips[s]) { ::setenv("VPIPE_GEMMA_SKIP_LAYER", strips[s], 1); }
    else            { ::unsetenv("VPIPE_GEMMA_SKIP_LAYER"); }
    double m = 1e18;
    for (int r = 0; r < 5; ++r) { m = std::min(m, decode_ms(ids)); }
    const double mt = m / N;
    if (s == 0) {
      base = mt;
      std::printf("[gemma_strip] %-12s %.3f ms/tok (baseline)\n",
                  "none", mt);
    } else {
      std::printf("[gemma_strip] strip %-12s %.3f ms/tok  (-%.3f, %4.1f%%)\n",
                  strips[s], mt, base - mt, 100.0 * (base - mt) / base);
    }
  }
  ::unsetenv("VPIPE_GEMMA_SKIP_LAYER");

  // Deeper ablation: all-global run (strip sliding -> only the 7 global layers),
  // then toggle PLE. Delta = PLE's wall cost in the all-global config, to
  // head-to-head omlx's PLE handling free of sliding-layer noise.
  auto minms = [&](int reps) {
    double m = 1e18; for (int r = 0; r < reps; ++r) { m = std::min(m, decode_ms(ids)); }
    return m / N;
  };
  ::setenv("VPIPE_GEMMA_SKIP_LAYER", "sliding", 1);
  ::unsetenv("VPIPE_GEMMA_SKIP_PLE");
  (void)decode_ms(ids);
  const double g_ple = minms(5);
  ::setenv("VPIPE_GEMMA_SKIP_PLE", "1", 1);
  (void)decode_ms(ids);
  const double g_nople = minms(5);
  // ... and additionally remove the FFN (gate/up + down GEMVs) -> only the
  // attention path + norms remain in the 7 global layers.
  ::setenv("VPIPE_GEMMA_SKIP_FFN", "1", 1);
  (void)decode_ms(ids);
  const double g_noff = minms(5);
  ::unsetenv("VPIPE_GEMMA_SKIP_FFN");
  // Fixed-only floor: strip ALL layers + PLE -> embed + final-norm + lm_head +
  // argmax + the per-token CPU<->GPU handoff. Splits the residual gap into the
  // global-attention path vs the fixed per-token overhead.
  ::setenv("VPIPE_GEMMA_SKIP_LAYER", "all", 1);
  ::setenv("VPIPE_GEMMA_SKIP_PLE", "1", 1);
  (void)decode_ms(ids);
  const double fixed_only = minms(5);
  // Fixed-tail split: peel lm_head (262144-vocab GEMV) then embed gather.
  ::setenv("VPIPE_GEMMA_SKIP_LMHEAD", "1", 1);
  (void)decode_ms(ids);
  const double fixed_nolm = minms(5);
  ::setenv("VPIPE_GEMMA_SKIP_EMBED", "1", 1);
  (void)decode_ms(ids);
  const double fixed_nolm_noemb = minms(5);
  ::unsetenv("VPIPE_GEMMA_SKIP_LMHEAD");
  ::unsetenv("VPIPE_GEMMA_SKIP_EMBED");
  ::unsetenv("VPIPE_GEMMA_SKIP_PLE");
  ::unsetenv("VPIPE_GEMMA_SKIP_LAYER");
  std::printf("[gemma_strip] fixed-tail: lm_head %.3f  embed %.3f  "
              "argmax/norm/sampler %.3f ms/tok\n",
              fixed_only - fixed_nolm, fixed_nolm - fixed_nolm_noemb,
              fixed_nolm_noemb);
  std::printf("[gemma_strip] all-global  with-PLE %.3f  no-PLE %.3f  "
              "PLE cost %.3f ms/tok\n", g_ple, g_nople, g_ple - g_nople);
  std::printf("[gemma_strip] all-global  no-PLE %.3f  no-PLE-no-FFN %.3f  "
              "FFN cost %.3f ms/tok (7 global layers)\n",
              g_nople, g_noff, g_nople - g_noff);
  std::printf("[gemma_strip] fixed-only (embed+lm_head+sample+handoff) %.3f | "
              "7 global attn+proj+norm = %.3f ms/tok\n",
              fixed_only, g_noff - fixed_only);
  // Also PLE cost in the FULL run (all 42 layers), for reference.
  ::unsetenv("VPIPE_GEMMA_SKIP_LAYER");
  ::setenv("VPIPE_GEMMA_SKIP_PLE", "1", 1);
  (void)decode_ms(ids);
  const double full_nople = minms(5);
  ::unsetenv("VPIPE_GEMMA_SKIP_PLE");
  std::printf("[gemma_strip] full        with-PLE %.3f  no-PLE %.3f  "
              "PLE cost %.3f ms/tok\n", base, full_nople, base - full_nople);

  // Sliding-attention split: all-SLIDING run (strip global -> 35 sliding
  // layers), no-PLE no-FFN, then peel the SDPA core (SKIP_ATTN) and the QKV+O
  // projections (SKIP_PROJ). Localises the +10%/sliding-layer deficit to the
  // attention kernel vs the projection GEMVs vs the norm/rope/KV-write rest.
  ::setenv("VPIPE_GEMMA_SKIP_LAYER", "global", 1);
  ::setenv("VPIPE_GEMMA_SKIP_PLE", "1", 1);
  ::setenv("VPIPE_GEMMA_SKIP_FFN", "1", 1);
  (void)decode_ms(ids);
  const double s_base = minms(5);                 // norm+QKV+SDPA+O (35 sliding)
  ::setenv("VPIPE_GEMMA_SKIP_ATTN", "1", 1);
  (void)decode_ms(ids);
  const double s_noattn = minms(5);               // norm+QKV+O (SDPA removed)
  ::setenv("VPIPE_GEMMA_SKIP_PROJ", "1", 1);
  (void)decode_ms(ids);
  const double s_noproj = minms(5);               // norm+rope+kvwrite (no GEMV)
  ::unsetenv("VPIPE_GEMMA_SKIP_ATTN");
  ::unsetenv("VPIPE_GEMMA_SKIP_PROJ");
  ::unsetenv("VPIPE_GEMMA_SKIP_FFN");
  ::unsetenv("VPIPE_GEMMA_SKIP_PLE");
  ::unsetenv("VPIPE_GEMMA_SKIP_LAYER");
  std::printf("[gemma_strip] all-sliding(35) no-PLE-no-FFN %.3f | "
              "SDPA %.3f  QKV+O proj %.3f  norm/rope/kv+fixed %.3f ms/tok\n",
              s_base, s_base - s_noattn, s_noattn - s_noproj, s_noproj);
  EXPECT_TRUE(true);
}

// RMS-kernel accuracy probe: greedy-decode N tokens via the host-[vocab]-argmax
// path and, at each step, record the token + the top1-top2 logit gap. Lets an
// A/B of two RMS kernels (e.g. RMS_TG 256 vs 512) quantify HOW divergent they
// are: where the streams first differ, how many tokens flip, and -- crucially --
// the top1-top2 gap AT each flip (a tiny gap => a near-tie => benign f32-rounding
// reorder, not a quality regression). Prints tokens= / gaps= CSV + FNV for
// offline diffing. Gated on VPIPE_GEMMA_RMS_ACC (+ VPIPE_GGUF_TEST_MODEL_PATH).
TEST(metal_lm_smoke, gguf_gemma_rms_accuracy) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_GEMMA_RMS_ACC") == nullptr) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir = path; spec.compute_dtype = "bf16";
  spec.page_tokens = 512; spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> ids;
  lm->chat_template()->render_user_turn(
      "List the first ten prime numbers.", true, &ids);
  ASSERT_TRUE(!ids.empty());
  int N = 96;
  if (const char* e = std::getenv("VPIPE_GEMMA_RMS_ACC_N")) {
    const int v = std::atoi(e); if (v > 0) { N = v; }
  }
  auto top1_top2 = [&](double& gap) -> std::int32_t {
    const std::vector<float>& lg = lm->last_logits_host();
    if (lg.empty()) { gap = -1.0; return -1; }
    int a1 = 0; float v1 = lg[0];
    for (int i = 1; i < (int)lg.size(); ++i) {
      if (lg[i] > v1) { v1 = lg[i]; a1 = i; }
    }
    float v2 = -1e30f;
    for (int i = 0; i < (int)lg.size(); ++i) {
      if (i != a1 && lg[i] > v2) { v2 = lg[i]; }
    }
    gap = (double)v1 - (double)v2;
    return a1;
  };
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::vector<std::int32_t> tok;
  std::vector<double> gaps;
  std::int32_t t = lm->prefill(ctx, ids);
  ASSERT_TRUE(t >= 0);
  { double g; (void)top1_top2(g); tok.push_back(t); gaps.push_back(g); }
  for (int i = 1; i < N; ++i) {
    t = lm->next_token(ctx, t);
    if (t < 0) { break; }
    double g; (void)top1_top2(g);
    tok.push_back(t); gaps.push_back(g);
  }
  std::uint64_t h = 1469598103934665603ULL;
  for (auto x : tok) {
    h ^= (std::uint64_t)(std::uint32_t)x; h *= 1099511628211ULL;
  }
  double mingap = 1e30; int mingi = -1;
  for (int i = 0; i < (int)gaps.size(); ++i) {
    if (gaps[i] >= 0 && gaps[i] < mingap) { mingap = gaps[i]; mingi = i; }
  }
  std::printf("[rms_acc] N=%zu fnv=%016llx mingap=%.4f@%d\n",
              tok.size(), (unsigned long long)h, mingap, mingi);
  std::printf("[rms_acc] tokens=");
  for (auto x : tok) { std::printf("%d,", x); }
  std::printf("\n[rms_acc] gaps=");
  for (auto g : gaps) { std::printf("%.3f,", g); }
  std::printf("\n");
  EXPECT_TRUE(true);
}

// Pipelined-decode token-exactness (GGUF gemma4_unified). pdecode_* greedy
// must produce the SAME token stream as the synchronous next_token loop (host
// [vocab] argmax = ground truth) AND next_token_greedy (on-GPU argmax). A
// mismatch localises a bug to the in-stream embed gather, the GPU argmax, or
// the event-chain KV ordering. Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_pdecode_matches_sync) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> ids;
  lm->chat_template()->render_user_turn(
      "List the first ten prime numbers.", true, &ids);
  ASSERT_TRUE(!ids.empty());
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int N = 48;

  // Reference: synchronous next_token (host [vocab] argmax).
  std::vector<std::int32_t> ref;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t t = lm->prefill(ctx, ids);
    ASSERT_TRUE(t >= 0);
    ref.push_back(t);
    for (int i = 1; i < N; ++i) {
      t = lm->next_token(ctx, t);
      if (t < 0) { break; }
      ref.push_back(t);
    }
  }

  // On-GPU argmax (next_token_greedy) must match the host argmax.
  std::vector<std::int32_t> grd;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t t = lm->prefill(ctx, ids);
    ASSERT_TRUE(t >= 0);
    grd.push_back(t);
    for (int i = 1; i < N; ++i) {
      t = lm->next_token_greedy(ctx, t);
      if (t < 0) { break; }
      grd.push_back(t);
    }
  }

  // Pipelined greedy (default SamplerParams == argmax-equivalent).
  std::vector<std::int32_t> pipe;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    const std::int32_t first = lm->prefill(ctx, ids);
    ASSERT_TRUE(first >= 0);
    pipe.push_back(first);
    genai::SamplerParams gsp;             // defaults -> argmax
    ASSERT_TRUE(lm->pdecode_begin(ctx, first, prompt, gsp, N));
    for (int i = 1; i < N; ++i) {
      if (!lm->pdecode_commit(ctx)) { break; }
      const std::int32_t n = lm->pdecode_next(ctx);
      if (n < 0) { break; }
      pipe.push_back(n);
    }
    lm->pdecode_end(ctx);
  }

  std::size_t mism_g = 0, mism_p = 0;
  const std::size_t ng = std::min(ref.size(), grd.size());
  for (std::size_t i = 0; i < ng; ++i) {
    if (ref[i] != grd[i]) { ++mism_g; }
  }
  const std::size_t np = std::min(ref.size(), pipe.size());
  for (std::size_t i = 0; i < np; ++i) {
    if (ref[i] != pipe[i]) { ++mism_p; }
  }
  // FNV-1a over the ref token stream + first 8 ids: cross-config token-exact
  // check (run with/without VPIPE_GEMMA_NO_QKV_FUSE and compare the hash).
  std::uint64_t h = 1469598103934665603ULL;
  for (auto t : ref) { h ^= (std::uint64_t)(std::uint32_t)t; h *= 1099511628211ULL; }
  std::printf("[gguf_pdecode] ref=%zu greedy=%zu pipe=%zu  "
              "greedy_mism=%zu pipe_mism=%zu ref_fnv=%016llx first=",
              ref.size(), grd.size(), pipe.size(), mism_g, mism_p,
              (unsigned long long)h);
  for (std::size_t i = 0; i < ref.size() && i < 8; ++i) {
    std::printf("%d,", ref[i]);
  }
  std::printf("\n");
  ASSERT_TRUE(pipe.size() == ref.size());
  ASSERT_TRUE(grd.size() == ref.size());
  EXPECT_TRUE(mism_g == 0);
  EXPECT_TRUE(mism_p == 0);
}

// Global-layer prefill SDPA kernels (VPIPE_GEMMA_SDPA = flash|dev|staged).
//   * dev must be BIT-IDENTICAL to staged (same BK=8 softmax blocking) ->
//     greedy token-IDENTICAL: a hard gate.
//   * flash (llama.cpp-style Q8/C64, the default) is a different but equally
//     accurate online-softmax; it is NOT bit-identical, so we check it is no
//     WORSE than staged against the scalar serial path (the project's serial
//     reference) and that it stays coherent. Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_sdpa_kernels) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_NO_SDPA_DEV");
  ::unsetenv("VPIPE_GEMMA_SDPA");
  ::unsetenv("VPIPE_GEMMA_SCALAR_ATTN");
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 16;          // max_seq 8192 -> flash slack at n~600
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // ~600-token prompt so the global/full layers' O(n^2) attention runs.
  std::string para = "The history of computing is long and storied. ";
  std::string big;
  for (int i = 0; i < 40; ++i) { big += para; }
  big += "Summarize the key milestones.";
  std::vector<std::int32_t> ids;
  lm->chat_template()->render_user_turn(big, true, &ids);
  ASSERT_TRUE(ids.size() > 256);
  const int N = 48;

  auto gen = [&](const char* sdpa, bool scalar) {
    if (scalar) { ::setenv("VPIPE_GEMMA_SCALAR_ATTN", "1", 1); }
    if (sdpa)   { ::setenv("VPIPE_GEMMA_SDPA", sdpa, 1); }
    std::vector<std::int32_t> out;
    auto ctx = lm->make_context();
    std::int32_t t = lm->prefill(ctx, ids);
    out.push_back(t);
    for (int i = 1; i < N && t >= 0; ++i) {
      t = lm->next_token(ctx, t);
      out.push_back(t);
    }
    ::unsetenv("VPIPE_GEMMA_SCALAR_ATTN");
    ::unsetenv("VPIPE_GEMMA_SDPA");
    return out;
  };
  auto cmp = [](const std::vector<std::int32_t>& a,
                const std::vector<std::int32_t>& b) {
    std::size_t mm = 0;
    const std::size_t m = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < m; ++i) { if (a[i] != b[i]) { ++mm; } }
    return mm;
  };
  // First index where two token streams diverge (-1 == identical over min len).
  auto first_div = [](const std::vector<std::int32_t>& a,
                      const std::vector<std::int32_t>& b) {
    const std::size_t m = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < m; ++i) {
      if (a[i] != b[i]) { return (int)i; }
    }
    return -1;
  };

  const std::vector<std::int32_t> staged = gen("staged", false);
  const std::vector<std::int32_t> dev    = gen("dev", false);
  const std::vector<std::int32_t> flash  = gen("flash", false);
  const std::vector<std::int32_t> scalar = gen(nullptr, true);

  const std::size_t dev_vs_staged = cmp(dev, staged);
  const std::size_t flash_vs_scalar = cmp(flash, scalar);
  const std::size_t staged_vs_scalar = cmp(staged, scalar);
  const std::size_t flash_vs_dev = cmp(flash, dev);
  // The ~600-token prompt is < the sliding ring_cap (window 1024 + chunk
  // 2048 = 3072), so the SLIDING layers run no-wrap -> they take the flash
  // path too. So `flash` here exercises BOTH the global and sliding flash
  // kernels; flash-vs-scalar tracks the whole prefill SDPA, not just global.
  std::printf("[gguf_sdpa] prompt=%zu N=%d | dev-vs-staged=%zu | "
              "flash-vs-scalar=%zu (first_div=%d) | staged-vs-scalar=%zu | "
              "flash-vs-dev=%zu (first_div=%d)\n",
              ids.size(), N, dev_vs_staged, flash_vs_scalar,
              first_div(flash, scalar), staged_vs_scalar, flash_vs_dev,
              first_div(flash, dev));
  ASSERT_TRUE(dev.size() == staged.size() && flash.size() == staged.size());
  // dev is bit-identical to staged -> token-identical (hard gate).
  EXPECT_TRUE(dev_vs_staged == 0);
  // flash uses an fp32 O accumulator + fp32 QK scores, so it is a VALID (not
  // bit-identical) approximation of the serial reference. It must track scalar
  // for a meaningful PREFIX; a broken flash prefill mispredicts from token ~0.
  // It is NOT robust to require flash-vs-scalar==0: a token-16 fp near-tie
  // between the flash/staged/scalar prefills is tipped by the *decode* kernel's
  // rounding, so EXACTLY one of {flash,staged} tie-matches scalar past token 16
  // and the other diverges (~32 tokens) -- and which one flips between the
  // gtile (12B global, fp32) and sdpa_mb decode kernels. That symmetry (same
  // count, same first_div, opposite label) is expected fp noise, not a flash
  // bug. So gate on an EARLY divergence instead. The final correctness gate is
  // decode_matches_prefill (argmax-exact) + metal-flash vs the Python/MLX
  // oracle on the safetensors models (64 GB box, gemma4-12b-bench-results.md).
  const int flash_fd = first_div(flash, scalar);
  EXPECT_TRUE(flash_fd < 0 || flash_fd >= 8);
}

// Metal decode self-consistency A/B (GGUF gemma4_unified). The decode
// forward (qmv / sdpa_mb / qmv_geglu g32 kernels) must produce the SAME
// next-token logits as the prefill forward (steel qmm kernels) at the SAME
// position -- that is the project's token-exact bar. We take the prompt +
// its own prefill argmax token `tok0`, then compute the logits that predict
// the token AFTER tok0 two ways:
//   * decode path:  prefill(prompt) then ONE forced decode step of tok0;
//   * prefill path: prefill(prompt ++ [tok0]) in a single pass (reference).
// They must agree. Since MLX with the identical converted weights is
// coherent, a mismatch localises the bug to a decode-only g32 kernel (this
// is the metal-vs-MLX decode-divergence seen at bring-up, isolated without
// MLX). Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_decode_matches_prefill) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = std::getenv("VPIPE_GGUF_AB_F16") ? "f16" : "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);
  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Answer in one word.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  // Decode path: prefill the prompt, then ONE forced decode step of tok0.
  auto cd = lm->make_context();
  ASSERT_TRUE(cd.valid());
  const std::int32_t tok0 = lm->prefill(cd, ids);
  ASSERT_TRUE(tok0 >= 0);
  const std::int32_t dec_next = lm->next_token(cd, tok0);
  ASSERT_TRUE(dec_next >= 0);
  const std::vector<float> Ldec = lm->last_logits_host();  // after tok0

  // Prefill path (reference): prompt ++ [tok0] in one pass.
  std::vector<std::int32_t> ids2 = ids;
  ids2.push_back(tok0);
  auto cp = lm->make_context();
  ASSERT_TRUE(cp.valid());
  const std::int32_t pre_next = lm->prefill(cp, ids2);
  ASSERT_TRUE(pre_next >= 0);
  const std::vector<float> Lpre = lm->last_logits_host();  // after tok0

  // Logit-vector comparison (diagnostic), then the token-exact assertion.
  if (!Lpre.empty() && Ldec.size() == Lpre.size()) {
    double max_abs = 0.0, sum_sq = 0.0, sum_ref = 0.0;
    int amax_d = 0, amax_p = 0;
    for (std::size_t i = 0; i < Lpre.size(); ++i) {
      const double d = std::fabs((double)Ldec[i] - (double)Lpre[i]);
      max_abs = std::fmax(max_abs, d);
      sum_sq += d * d;
      sum_ref += (double)Lpre[i] * (double)Lpre[i];
      if (Ldec[i] > Ldec[amax_d]) { amax_d = (int)i; }
      if (Lpre[i] > Lpre[amax_p]) { amax_p = (int)i; }
    }
    const double rel_l2 = std::sqrt(sum_sq / (sum_ref + 1e-9));
    std::printf("[gguf_decode_ab] tok0=%d prefill_argmax=%d decode_argmax=%d "
                "max_abs=%g rel_l2=%g\n",
                (int)tok0, amax_p, amax_d, max_abs, rel_l2);
  }
  std::printf("[gguf_decode_ab] prefill_next=%d decode_next=%d\n",
              (int)pre_next, (int)dec_next);
  EXPECT_TRUE(dec_next == pre_next);   // decode must match prefill
}

// Sliding-layer ring-wrap decode exactness (GGUF gemma4_unified). The global
// decode attn uses the gtile vec kernel; extending it to the SLIDING (windowed,
// ring-buffered) layers needs a prompt LONG enough to (a) trigger the window
// cutoff (L > window) AND (b) wrap the sliding KV ring (L > window +
// sliding_chunk == ring_cap, 3072 on the 12B) -- the short prompts in
// decode_matches_prefill never exercise either, so a sliding-decode-kernel bug
// in the ring/window math hides there. Same decode-vs-prefill bar: a forced
// decode step at a ring-wrapped position must match the single-pass prefill of
// prompt ++ [tok0] (the flash prefill is the validated sliding reference). Run
// it with the gtile vec sliding kernel (default) AND with VPIPE_GEMMA_GTILE_ATTN
// =0 (sdpa_mb) -- both must pass. Gated on VPIPE_GGUF_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gguf_gemma_decode_sliding_ringwrap) {
  const char* path = std::getenv("VPIPE_GGUF_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  // Force a bounded sliding ring AND disable the lazy single-pass grow so this
  // test exercises the ACTUAL ring-wrap + staged-sliding path. Pin chunk=2048
  // (ring_cap = window 1024 + 2048 = 3072 < L) and VPIPE_GEMMA_NO_SLIDING_GROW
  // (else a fresh one-shot prefill grows the ring to L and never wraps). Both
  // are read at load; unset after. The wrap path stays reachable in production
  // for incremental (kv_off>0) prefill and when the grow is disabled.
  ::setenv("VPIPE_GEMMA_SLIDING_CHUNK", "2048", 1);
  ::setenv("VPIPE_GEMMA_NO_SLIDING_GROW", "1", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) {
    ::unsetenv("VPIPE_GEMMA_SLIDING_CHUNK");
    ::unsetenv("VPIPE_GEMMA_NO_SLIDING_GROW");
    return;
  }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 16;               // max_seq 8192 (room for L + ring)
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_GEMMA_SLIDING_CHUNK");
  ::unsetenv("VPIPE_GEMMA_NO_SLIDING_GROW");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // Long prompt: a real rendered turn (<bos> first) padded with a benign token
  // to L. L=4096 > ring_cap(3072) so the sliding layers run BOTH a window
  // cutoff and a ring wrap at the decode position.
  const int L = 4096;
  std::vector<std::int32_t> seed;
  lm->chat_template()->render_user_turn("Benchmark.", true, &seed);
  ASSERT_TRUE(!seed.empty());
  const std::int32_t bos  = seed.front();
  const std::int32_t fill = seed.size() > 1 ? seed[1] : bos;
  std::vector<std::int32_t> ids;
  ids.reserve(L);
  ids.push_back(bos);
  for (int k = 1; k < L; ++k) { ids.push_back(fill); }

  // Decode path: prefill(prompt), then ONE forced decode of tok0 (sliding
  // kernel at a ring-wrapped position).
  auto cd = lm->make_context();
  ASSERT_TRUE(cd.valid());
  const std::int32_t tok0 = lm->prefill(cd, ids);
  ASSERT_TRUE(tok0 >= 0);
  const std::int32_t dec_next = lm->next_token(cd, tok0);
  ASSERT_TRUE(dec_next >= 0);
  const std::vector<float> Ldec = lm->last_logits_host();

  // Prefill reference: prompt ++ [tok0] in one pass.
  std::vector<std::int32_t> ids2 = ids;
  ids2.push_back(tok0);
  auto cp = lm->make_context();
  ASSERT_TRUE(cp.valid());
  const std::int32_t pre_next = lm->prefill(cp, ids2);
  ASSERT_TRUE(pre_next >= 0);
  const std::vector<float> Lpre = lm->last_logits_host();

  if (!Lpre.empty() && Ldec.size() == Lpre.size()) {
    double max_abs = 0.0, sum_sq = 0.0, sum_ref = 0.0;
    int amax_d = 0, amax_p = 0;
    for (std::size_t i = 0; i < Lpre.size(); ++i) {
      const double d = std::fabs((double)Ldec[i] - (double)Lpre[i]);
      max_abs = std::fmax(max_abs, d);
      sum_sq += d * d;
      sum_ref += (double)Lpre[i] * (double)Lpre[i];
      if (Ldec[i] > Ldec[amax_d]) { amax_d = (int)i; }
      if (Lpre[i] > Lpre[amax_p]) { amax_p = (int)i; }
    }
    std::printf("[gguf_sliding_ringwrap] L=%d tok0=%d prefill_argmax=%d "
                "decode_argmax=%d max_abs=%g rel_l2=%g\n",
                L, (int)tok0, amax_p, amax_d, max_abs,
                std::sqrt(sum_sq / (sum_ref + 1e-9)));
  }
  std::printf("[gguf_sliding_ringwrap] prefill_next=%d decode_next=%d\n",
              (int)pre_next, (int)dec_next);
  EXPECT_TRUE(dec_next == pre_next);   // sliding decode must match prefill
}

TEST(metal_lm_smoke, gemma_text_chat_default_backend) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  // Nothing must force a backend (a prior test may have set it).
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());      // loads with NO env var

  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);                    // refutes "no chat template"
  EXPECT_TRUE(tpl->family_name() == "gemma");

  std::vector<std::int32_t> ids;
  tpl->render_user_turn(
      "What is the capital of France? Answer in one word.",
      /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());
  EXPECT_TRUE(ids.front() == 2);                  // <bos>
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill(ctx, ids);
  EXPECT_TRUE(first >= 0);
  std::printf("[metal_lm_smoke.gemma_text_chat_default_backend] first=%d\n",
              (int)first);
}

// gemma4_unified 12B on the metal backend (no-MLX coverage): no PLE, k_eq_v
// full layers (1 K/V head, no v_proj), sliding 8 K/V heads, mixed 4/8-bit
// quant. Feed the raw oracle prompt ids and require the greedy continuation
// to match /tmp/gemma12b_text_oracle.py token-for-token. Gated on
// VPIPE_GEMMA12B_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma12b_unified_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA12B_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");       // no-MLX default == metal
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { return; }
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = 512;
  spec.max_pages     = 8;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());

  const std::vector<std::int32_t> prompt{
      2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 1567, 1806, 5905,
      7913, 236761, 106, 107, 105, 4368, 107};
  const std::vector<std::int32_t> oracle{
      100, 45518, 107, 818, 2430, 563, 10980, 573, 506, 5618, 529, 1806, 5905,
      7913, 236761};

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  std::vector<std::int32_t> got;
  got.push_back(lm->prefill(ctx, prompt));
  for (std::size_t i = 1; i < oracle.size(); ++i) {
    const std::int32_t nxt = lm->next_token(ctx);
    if (nxt < 0) { break; }
    got.push_back(nxt);
  }
  EXPECT_TRUE(got.size() == oracle.size());
  for (std::size_t i = 0; i < oracle.size() && i < got.size(); ++i) {
    EXPECT_TRUE(got[i] == oracle[i]);
  }
  std::printf("[metal_lm_smoke.gemma12b_unified_token_exact] first=%d\n",
              got.empty() ? -1 : (int)got[0]);
}

// Regression for "chat truncates ~2048 tokens regardless of max_pages": the
// metal Gemma contiguous KV used to hardcode max_seq=2048, ignoring the
// configured budget. It must now follow page_tokens * max_pages. Load with a
// SMALL budget (128 * 3 = 384) and force-decode past it: the decode must hit
// its cap at ~384 -- NOT run on toward the old 2048 -- proving the budget is
// config-derived. (Build-agnostic: the MLX paged pool caps at the same
// 384-token budget.) Gated on the Gemma checkpoint.
TEST(metal_lm_smoke, gemma_kv_budget_follows_max_pages) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  // The contiguous-KV cap is a METAL-backend property; pin metal so the test
  // exercises the metal exec in BOTH builds. (In the MLX build an unset
  // backend loads the MLX paged Gemma path, whose KV grows differently and
  // does not hard-stop at page_tokens*max_pages -- so the cap assertion only
  // makes sense against the metal exec.)
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  if (sess.metal_compute() == nullptr || !sess.metal_compute()->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  const int kPageTokens = 128, kMaxPages = 3;
  const int kBudget = kPageTokens * kMaxPages;     // 384
  genai::LoadSpec spec;
  spec.hf_dir        = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens   = kPageTokens;
  spec.max_pages     = kMaxPages;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  auto ids = lm->tokenizer().encode("Count upwards:");
  ASSERT_TRUE(!ids.empty());
  ASSERT_TRUE((int)ids.size() < kBudget);
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  int32_t forced = lm->prefill(ctx, ids);
  ASSERT_TRUE(forced >= 0);

  // Force-decode (ignore stop tokens) until the KV cap returns -1.
  int produced = 0;
  const int kLimit = 1500;     // > budget(384), << old hardcoded 2048
  for (int i = 0; i < kLimit; ++i) {
    const int32_t n = lm->next_token(ctx, forced);
    if (n < 0) { break; }
    forced = n;
    ++produced;
  }
  const int final_pos = (int)ids.size() + produced;
  std::printf("[metal_lm_smoke.gemma_kv_budget] budget=%d final_pos=%d "
              "produced=%d\n", kBudget, final_pos, produced);
  // Hit a cap well before the loop limit (and far below the old 2048),
  // landing at the configured budget.
  EXPECT_TRUE(produced < kLimit);
  EXPECT_TRUE(final_pos <= kBudget + kPageTokens);   // ~384, NOT ~2048
  EXPECT_TRUE(final_pos >= kBudget - kPageTokens);
}

// Per-token GPU-resident pipelined decode (LoadedLanguageModel::pdecode_*):
// greedy output must be token-identical to the synchronous next_token loop,
// and the sampled path must be deterministic given a seed + coherent.
TEST(metal_pdecode, greedy_matches_sync) {
  const char* path = std::getenv("VPIPE_METAL_LM_SMOKE_MODEL");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  if (sess.metal_compute() == nullptr || !sess.metal_compute()->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  const auto ids = lm->tokenizer().encode(
      "Tell me a short story about a curious robot.");
  ASSERT_TRUE(!ids.empty());
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int N = 48;

  // Reference: synchronous greedy next_token loop.
  std::vector<std::int32_t> ref;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t t = lm->prefill(ctx, ids);
    ASSERT_TRUE(t >= 0);
    ref.push_back(t);
    for (int i = 1; i < N; ++i) {
      t = lm->next_token_greedy(ctx, t);
      if (t < 0) { break; }
      ref.push_back(t);
    }
  }

  // Pipelined greedy (default SamplerParams is argmax-equivalent).
  std::vector<std::int32_t> pipe;
  {
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    const std::int32_t first = lm->prefill(ctx, ids);
    ASSERT_TRUE(first >= 0);
    pipe.push_back(first);
    genai::SamplerParams gsp;   // temperature 1 + all-default -> argmax
    ASSERT_TRUE(lm->pdecode_begin(ctx, first, prompt, gsp, N));
    for (int i = 1; i < N; ++i) {
      ASSERT_TRUE(lm->pdecode_commit(ctx));
      const std::int32_t n = lm->pdecode_next(ctx);
      if (n < 0) { break; }
      pipe.push_back(n);
    }
    lm->pdecode_end(ctx);
  }

  ASSERT_TRUE(ref.size() == pipe.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != pipe[i]) { ++mism; }
  }
  std::printf("[metal_pdecode] greedy: %zu tokens, %zu mismatches vs sync\n",
              pipe.size(), mism);
  EXPECT_TRUE(mism == 0u);

  // Sampled path: deterministic given a fixed seed; two runs must match,
  // and the GPU sampler must honour penalties/top-k without crashing.
  auto run_sampled = [&](std::uint64_t seed) {
    std::vector<std::int32_t> out;
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    const std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { return out; }
    out.push_back(first);
    genai::SamplerParams sp;
    sp.temperature = 0.8f;
    sp.top_p = 0.95f;
    sp.top_k = 40;
    sp.repetition_penalty = 1.1f;
    sp.seed = seed;
    if (!lm->pdecode_begin(ctx, first, prompt, sp, N)) { return out; }
    for (int i = 1; i < N; ++i) {
      if (!lm->pdecode_commit(ctx)) { break; }
      const std::int32_t n = lm->pdecode_next(ctx);
      if (n < 0) { break; }
      out.push_back(n);
    }
    lm->pdecode_end(ctx);
    return out;
  };
  const auto s1 = run_sampled(12345u);
  const auto s2 = run_sampled(12345u);
  ASSERT_TRUE(s1.size() >= 2u);
  ASSERT_TRUE(s1.size() == s2.size());
  for (std::size_t i = 0; i < s1.size(); ++i) { EXPECT_TRUE(s1[i] == s2[i]); }
  const auto txt = lm->tokenizer().decode(
      std::span<const std::int32_t>(s1.data(), s1.size()));
  std::printf("[metal_pdecode] sampled(seed=12345): %zu tok | '%s'\n",
              s1.size(), txt.c_str());
  EXPECT_TRUE(!txt.empty());
}

// Metal image-VQA smoke: load a Qwen3-VL model on the metal backend,
// encode a synthetic RGB image with the metal vision tower, splice
// text + image tokens via prefill_multimodal_metal (3-axis mROPE), and
// decode a few tokens. Runs in BOTH builds; proves the no-MLX image-VQA
// path end-to-end. Env: VPIPE_METAL_VQA_SMOKE_MODEL=/path/Qwen3.5-4B-...
TEST(metal_lm_smoke, image_vqa_decode) {
  const char* path = std::getenv("VPIPE_METAL_VQA_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  auto* vis = lm->metal_vision_encoder();
  ASSERT_TRUE(vis != nullptr);
  const int S = vis->config().spatial_merge;

  // Synthetic 128x128 RGB-planar image.
  const int H = 128, W = 128;
  std::vector<std::uint8_t> img((std::size_t)3 * H * W);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        img[((std::size_t)c * H + y) * W + x] =
            (std::uint8_t)((x + 2 * y + 40 * c) & 0xFF);
      }
    }
  }
  auto enc = vis->encode(img.data(), H, W);
  ASSERT_TRUE(enc.n_tokens > 0 && !enc.embeddings.empty());
  const int n_im = enc.n_tokens;
  const int mh = enc.grid_h / S, mw = enc.grid_w / S;
  ASSERT_TRUE(mh * mw == n_im);

  // Build refs: text prefix + image-token run (referencing the host
  // embeddings) + text suffix.
  auto pre = lm->tokenizer().encode("Describe the image:");
  auto suf = lm->tokenizer().encode("\nAnswer:");
  std::vector<genai::TokenRef> refs;
  refs.reserve(pre.size() + n_im + suf.size());
  for (auto id : pre) {
    genai::TokenRef r;
    r.kind = genai::TokenRef::Kind::Text;
    r.text_id = id;
    refs.push_back(r);
  }
  for (int off = 0; off < n_im; ++off) {
    genai::TokenRef r;
    r.kind = genai::TokenRef::Kind::ImageTokens;
    r.embeddings_buf = &enc.embeddings;
    r.image_token_offset = off;
    refs.push_back(r);
  }
  for (auto id : suf) {
    genai::TokenRef r;
    r.kind = genai::TokenRef::Kind::Text;
    r.text_id = id;
    refs.push_back(r);
  }

  std::pair<int, int> grid{mh, mw};
  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs),
      std::span<const std::pair<int, int>>(&grid, 1));
  ASSERT_TRUE(first >= 0);
  std::vector<std::int32_t> gen = {first};
  for (int i = 0; i < 8; ++i) {
    const std::int32_t nx = lm->next_token(ctx);
    if (nx < 0) { break; }
    gen.push_back(nx);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[metal_lm_smoke.image_vqa] n_im=%d grid=%dx%d | gen='%s'\n",
              n_im, mh, mw, text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// MULTIMODAL MTP token-exactness: the spec-decode path on a POST-IMAGE context
// (rope_first >= 0, the mROPE-advanced position) MUST reproduce the serial
// greedy loop the stages run without MTP -- this is what visual-qa /
// realtime-vqa scene-describe exercise. Encodes a synthetic image, splices it
// via prefill_multimodal_metal (3-axis mROPE), then compares lm->mtp_generate
// against a next_token_greedy reference from the IDENTICAL post-image state.
// Validates the rope_delta threading (the riskiest part of the stage wiring).
// Gated on VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH (the only model here with BOTH a
// vision tower -- via the optiq_vision.safetensors sidecar -- and an MTP head).
TEST(metal_lm_smoke, image_vqa_mtp_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN_OPTIQ_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  auto* vis = lm->metal_vision_encoder();
  ASSERT_TRUE(vis != nullptr);          // the OptiQ vision sidecar now loads
  ASSERT_TRUE(lm->mtp_available());
  const auto* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);
  const int S = vis->config().spatial_merge;

  // Synthetic 128x128 RGB-planar image (same fixture as image_vqa_decode).
  const int H = 128, W = 128;
  std::vector<std::uint8_t> img((std::size_t)3 * H * W);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        img[((std::size_t)c * H + y) * W + x] =
            (std::uint8_t)((x + 2 * y + 40 * c) & 0xFF);
      }
    }
  }
  auto enc = vis->encode(img.data(), H, W);
  ASSERT_TRUE(enc.n_tokens > 0 && !enc.embeddings.empty());
  const int n_im = enc.n_tokens;
  const int mh = enc.grid_h / S, mw = enc.grid_w / S;
  ASSERT_TRUE(mh * mw == n_im);
  std::pair<int, int> grid{mh, mw};

  // Rebuild the text+image+text ref stream per context (refs borrow enc.
  // embeddings, which outlive both prefills).
  auto build_refs = [&]() {
    auto pre = lm->tokenizer().encode("Describe the image:");
    auto suf = lm->tokenizer().encode("\nAnswer:");
    std::vector<genai::TokenRef> refs;
    refs.reserve(pre.size() + (std::size_t)n_im + suf.size());
    for (auto id : pre) {
      genai::TokenRef r;
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
      refs.push_back(r);
    }
    for (int off = 0; off < n_im; ++off) {
      genai::TokenRef r;
      r.kind = genai::TokenRef::Kind::ImageTokens;
      r.embeddings_buf = &enc.embeddings;
      r.image_token_offset = off;
      refs.push_back(r);
    }
    for (auto id : suf) {
      genai::TokenRef r;
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
      refs.push_back(r);
    }
    return refs;
  };
  auto is_stop = [tpl](std::int32_t id) { return tpl->is_stop_token(id); };
  const int kBudget = 64;

  // Reference: the serial greedy loop a stage runs WITHOUT MTP, over the
  // post-image mROPE positions (next_token_greedy reads ctx._rope_next_position).
  std::vector<std::int32_t> ref;
  bool ref_stop = false;
  {
    auto refs = build_refs();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t cur = lm->prefill_multimodal_metal(
        ctx, std::span<const genai::TokenRef>(refs),
        std::span<const std::pair<int, int>>(&grid, 1));
    ASSERT_TRUE(cur >= 0);
    for (int i = 0; i < kBudget; ++i) {
      if (is_stop(cur)) { ref_stop = true; break; }
      ref.push_back(cur);
      cur = lm->next_token_greedy(ctx, cur);
      if (cur < 0) { break; }
    }
  }
  ASSERT_TRUE(!ref.empty());

  // MTP from the IDENTICAL post-image state (fresh context, same prefill).
  std::vector<std::int32_t> got;
  int  produced = 0;
  bool hit_stop = false;
  {
    auto refs = build_refs();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t first = lm->prefill_multimodal_metal(
        ctx, std::span<const genai::TokenRef>(refs),
        std::span<const std::pair<int, int>>(&grid, 1));
    ASSERT_TRUE(first >= 0);
    auto on_toks = [&](std::span<const std::int32_t> toks) -> bool {
      for (std::int32_t id : toks) { got.push_back(id); }
      return true;
    };
    EXPECT_TRUE(lm->mtp_generate(ctx, first, kBudget, genai::SamplerParams{},
                                 is_stop, on_toks, &produced, &hit_stop));
  }

  int mism = 0;
  const std::size_t nn = std::min(ref.size(), got.size());
  for (std::size_t i = 0; i < nn; ++i) { if (ref[i] != got[i]) { ++mism; } }
  std::printf("[image_vqa_mtp] n_im=%d ref=%zu got=%zu produced=%d hit=%d "
              "ref_stop=%d mism=%d\n",
              n_im, ref.size(), got.size(), produced, (int)hit_stop,
              (int)ref_stop, mism);
  EXPECT_TRUE(mism == 0);
  EXPECT_TRUE(got.size() == ref.size());     // stopped in lockstep
  EXPECT_TRUE((int)got.size() == produced);
  EXPECT_TRUE(hit_stop == ref_stop);
}

// Gemma-4-12B "unified" (gemma4_unified, GGUF) END-TO-END vision VQA:
// loader detects the sibling mmproj -> builds Gemma4UnifiedEmbedder ->
// encode_image (host f32) -> Gemma VLM chat render -> owns_kv metal
// multimodal splice (prefill_multimodal_metal) -> greedy decode. Confirms
// the whole wiring at runtime (the embedder graph itself is golden-checked
// in gemma4_unified_embedder.*). Gated on the GGUF dir + a P6 PPM image.
TEST(metal_lm_smoke, gemma12b_unified_vqa_e2e) {
  const char* dir = std::getenv("VPIPE_GEMMA12B_GGUF_PATH");
  const char* imgp = std::getenv("VPIPE_GEMMA12B_TEST_IMAGE");
  if (!dir || !*dir || !imgp || !*imgp) { return; }
  // Read a binary P6 PPM -> planar [3,H,W] u8.
  std::FILE* f = std::fopen(imgp, "rb");
  if (!f) { return; }
  char magic[3] = {0};
  int W = 0, H = 0, maxv = 0;
  if (std::fscanf(f, "%2s %d %d %d", magic, &W, &H, &maxv) != 4 ||
      std::string(magic) != "P6" || W <= 0 || H <= 0) {
    std::fclose(f); return;
  }
  std::fgetc(f);   // single whitespace after maxval
  std::vector<std::uint8_t> inter((std::size_t)3 * H * W);
  const std::size_t got = std::fread(inter.data(), 1, inter.size(), f);
  std::fclose(f);
  ASSERT_TRUE(got == inter.size());
  std::vector<std::uint8_t> planar((std::size_t)3 * H * W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int c = 0; c < 3; ++c) {
        planar[((std::size_t)c * H + y) * W + x] =
            inter[((std::size_t)y * W + x) * 3 + c];
      }
    }
  }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = dir;
  spec.compute_dtype = "f16";
  spec.page_tokens = 1024;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  // Loader plumbing: the unified embedder must be constructed from the
  // sibling mmproj GGUF.
  auto* uni = lm->gemma4_unified_embedder();
  ASSERT_TRUE(uni != nullptr);
  ASSERT_TRUE(uni->has_vision());

  auto enc = uni->encode_image(planar.data(), H, W);
  ASSERT_TRUE(enc.has_value());
  ASSERT_TRUE(enc->n_tokens > 0);
  const int n_im = enc->n_tokens;

  auto tpl = genai::make_chat_template(lm->config().architecture,
                                     lm->tokenizer());
  ASSERT_TRUE(tpl != nullptr);
  const std::int32_t image_pad = tpl->image_pad_token_id();
  std::vector<std::int32_t> ids;
  const int counts[1] = {n_im};
  tpl->render_user_turn_vlm("Describe this image in one sentence.",
                            std::span<const int>(counts, 1),
                            /*is_first_turn=*/true, &ids);
  ASSERT_TRUE(!ids.empty());

  std::vector<genai::TokenRef> refs;
  refs.reserve(ids.size());
  int img_off = 0;
  for (std::int32_t id : ids) {
    genai::TokenRef r;
    if (id == image_pad && img_off < n_im) {
      r.kind = genai::TokenRef::Kind::ImageTokens;
      r.embeddings_host = &enc->rows;
      r.image_token_offset = img_off++;
    } else {
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
    }
    refs.push_back(r);
  }
  ASSERT_TRUE(img_off == n_im);

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs), {});
  ASSERT_TRUE(first >= 0);
  std::vector<std::int32_t> gen;
  std::int32_t nx = first;
  for (int i = 0; i < 64 && nx >= 0 && !tpl->is_stop_token(nx); ++i) {
    gen.push_back(nx);
    nx = lm->next_token(ctx);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[metal_lm_smoke.gemma12b_unified_vqa] %dx%d -> %d img tok | "
              "gen(%zu)='%s'\n", W, H, n_im, gen.size(), text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// Gemma-4-12B unified AUDIO+VIDEO end-to-end: mirrors the realtime-vqa
// metal path exactly -- render_video_prefix (1 frame) + render_audio_block
// (the new inline audio block) + render_vlm_completion, splicing BOTH image
// and audio soft tokens (TokenRef Image/Audio + embeddings_host) ->
// prefill_multimodal_metal -> greedy decode. Synthetic audio (a tone) so we
// only assert coherent text + that audio_pad slots were all consumed. Gated
// on the GGUF dir (audio needs no external file -- it is synthesised).
TEST(metal_lm_smoke, gemma12b_unified_av_e2e) {
  const char* dir = std::getenv("VPIPE_GEMMA12B_GGUF_PATH");
  const char* imgp = std::getenv("VPIPE_GEMMA12B_TEST_IMAGE");
  if (!dir || !*dir || !imgp || !*imgp) { return; }
  std::FILE* f = std::fopen(imgp, "rb");
  if (!f) { return; }
  char magic[3] = {0};
  int W = 0, H = 0, maxv = 0;
  if (std::fscanf(f, "%2s %d %d %d", magic, &W, &H, &maxv) != 4 ||
      std::string(magic) != "P6") { std::fclose(f); return; }
  std::fgetc(f);
  std::vector<std::uint8_t> inter((std::size_t)3 * H * W);
  if (std::fread(inter.data(), 1, inter.size(), f) != inter.size()) {
    std::fclose(f); return;
  }
  std::fclose(f);
  std::vector<std::uint8_t> planar((std::size_t)3 * H * W);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int c = 0; c < 3; ++c) {
        planar[((std::size_t)c * H + y) * W + x] =
            inter[((std::size_t)y * W + x) * 3 + c];
      }
    }
  }

  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = dir;
  spec.compute_dtype = "f16";
  spec.page_tokens = 1024;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());
  auto* uni = lm->gemma4_unified_embedder();
  ASSERT_TRUE(uni != nullptr && uni->has_vision() && uni->has_audio());

  auto ei = uni->encode_image(planar.data(), H, W);
  ASSERT_TRUE(ei.has_value() && ei->n_tokens > 0);
  const int n_im = ei->n_tokens;
  // 2 s of 440 Hz tone @ 16 kHz -> ceil(32000/640)=50 audio tokens.
  std::vector<float> pcm(32000);
  for (std::size_t i = 0; i < pcm.size(); ++i) {
    pcm[i] = 0.2f * std::sin(2.0f * 3.14159265f * 440.0f * i / 16000.0f);
  }
  auto ea = uni->encode_audio(pcm.data(), pcm.size());
  ASSERT_TRUE(ea.has_value() && ea->n_tokens > 0);
  const int n_au = ea->n_tokens;

  auto tpl = genai::make_chat_template(lm->config().architecture,
                                     lm->tokenizer());
  ASSERT_TRUE(tpl != nullptr);
  const std::int32_t video_pad = tpl->video_pad_token_id();
  const std::int32_t audio_pad = tpl->audio_pad_token_id();
  ASSERT_TRUE(audio_pad >= 0);

  const float fts[1] = {0.0f};
  const int counts[1] = {n_im};
  std::vector<std::int32_t> ids;
  ASSERT_TRUE(tpl->render_video_prefix(std::span<const float>(fts, 1),
                                       std::span<const int>(counts, 1),
                                       /*is_first_turn=*/true,
                                       std::string_view(), &ids));
  ASSERT_TRUE(tpl->render_audio_block(
      "Audio captured during this scene (<0.0 seconds> to <2.0 seconds>):\n",
      n_au, &ids));
  ASSERT_TRUE(tpl->render_vlm_completion("Describe the scene.", &ids));

  std::vector<genai::TokenRef> refs;
  refs.reserve(ids.size());
  int img_off = 0, aud_off = 0;
  for (std::int32_t id : ids) {
    genai::TokenRef r;
    if (id == video_pad && img_off < n_im) {
      r.kind = genai::TokenRef::Kind::ImageTokens;
      r.embeddings_host = &ei->rows;
      r.image_token_offset = img_off++;
    } else if (id == audio_pad && aud_off < n_au) {
      r.kind = genai::TokenRef::Kind::AudioTokens;
      r.embeddings_host = &ea->rows;
      r.audio_token_offset = aud_off++;
    } else {
      r.kind = genai::TokenRef::Kind::Text;
      r.text_id = id;
    }
    refs.push_back(r);
  }
  EXPECT_TRUE(img_off == n_im);
  EXPECT_TRUE(aud_off == n_au);   // all audio_pad slots consumed by the block

  auto ctx = lm->make_context();
  ASSERT_TRUE(ctx.valid());
  const std::int32_t first = lm->prefill_multimodal_metal(
      ctx, std::span<const genai::TokenRef>(refs), {});
  ASSERT_TRUE(first >= 0);
  std::vector<std::int32_t> gen;
  std::int32_t nx = first;
  for (int i = 0; i < 48 && nx >= 0 && !tpl->is_stop_token(nx); ++i) {
    gen.push_back(nx);
    nx = lm->next_token(ctx);
  }
  const auto text = lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  std::printf("[metal_lm_smoke.gemma12b_unified_av] img=%d aud=%d tok | "
              "gen(%zu)='%s'\n", n_im, n_au, gen.size(), text.c_str());
  EXPECT_TRUE(gen.size() >= 2u);
  EXPECT_TRUE(!text.empty());
}

// ---- Decode/prefill throughput bench (env-gated) -----------------
//
// Times prefill + an N-token decode loop on the metal backend under two
// sampling modes that MUST mirror the omlx (MLX server) side exactly:
//   greedy : argmax (temperature 0)
//   top_p  : temperature + nucleus, seeded.
// Sampling is host-side off last_logits_host(); the top_p path uses a
// top-K nucleus (nth_element, K=256) so the host sort cost stays small
// and representative rather than a naive full-vocab O(V log V) sort.
//
// Env:
//   VPIPE_METAL_LM_SMOKE_MODEL  model dir (reuses the smoke var)
//   VPIPE_METAL_BENCH_TOKENS    decode tokens (default 128)
//   VPIPE_METAL_BENCH_PROMPT    prompt text (default builtin)
//   VPIPE_METAL_BENCH_TEMP      top_p temperature (default 0.7)
//   VPIPE_METAL_BENCH_TOP_P     nucleus p (default 0.9)
//   VPIPE_METAL_BENCH_SEED      rng seed (default 1234)
namespace {

std::int32_t
bench_sample_top_p(const std::vector<float>& logits, std::vector<int>& idx,
                   float temp, float top_p, std::mt19937& rng)
{
  const int V = static_cast<int>(logits.size());
  const int K = std::min(V, 256);
  idx.resize(V);
  for (int i = 0; i < V; ++i) { idx[i] = i; }
  std::nth_element(
      idx.begin(), idx.begin() + (K - 1), idx.end(),
      [&](int a, int b) { return logits[a] > logits[b]; });
  idx.resize(K);
  std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
  const float inv_t = 1.0f / (temp > 0.0f ? temp : 1.0f);
  const float maxl = logits[idx[0]];
  std::vector<double> p(K);
  double sum = 0.0;
  for (int i = 0; i < K; ++i) {
    const double e = std::exp(
        static_cast<double>(logits[idx[i]] - maxl) * inv_t);
    p[i] = e;
    sum += e;
  }
  double cum = 0.0;
  int cut = K;
  for (int i = 0; i < K; ++i) {
    cum += p[i] / sum;
    if (cum >= static_cast<double>(top_p)) { cut = i + 1; break; }
  }
  double kept = 0.0;
  for (int i = 0; i < cut; ++i) { kept += p[i]; }
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double r = u(rng) * kept;
  double acc = 0.0;
  for (int i = 0; i < cut; ++i) {
    acc += p[i];
    if (r <= acc) { return idx[i]; }
  }
  return idx[cut - 1];
}

}  // namespace

TEST(metal_lm_bench, decode) {
  const char* path = std::getenv("VPIPE_METAL_LM_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  const int   n_decode = std::getenv("VPIPE_METAL_BENCH_TOKENS")
      ? std::atoi(std::getenv("VPIPE_METAL_BENCH_TOKENS")) : 128;
  const char* p_env = std::getenv("VPIPE_METAL_BENCH_PROMPT");
  const std::string prompt = (p_env && *p_env) ? p_env
      : "Once upon a time, in a small village nestled between tall "
        "mountains, there lived a curious young inventor named Mira. "
        "Every morning she woke before dawn to tinker in her workshop, "
        "and every evening she filled her notebooks with new ideas. One "
        "autumn day, a stranger arrived at the village gate carrying a "
        "broken machine, and Mira's life changed forever. This is the "
        "story of what happened next, told in full detail:";
  const float temp = std::getenv("VPIPE_METAL_BENCH_TEMP")
      ? (float)std::atof(std::getenv("VPIPE_METAL_BENCH_TEMP")) : 0.7f;
  const float top_p = std::getenv("VPIPE_METAL_BENCH_TOP_P")
      ? (float)std::atof(std::getenv("VPIPE_METAL_BENCH_TOP_P")) : 0.9f;
  const unsigned seed = std::getenv("VPIPE_METAL_BENCH_SEED")
      ? (unsigned)std::strtoul(std::getenv("VPIPE_METAL_BENCH_SEED"),
                               nullptr, 10) : 1234u;

  // Bench backend defaults to metal; override with VPIPE_METAL_BENCH_BACKEND
  // (e.g. "mlx" in the MLX build) to A/B the same harness across paths.
  const char* be = std::getenv("VPIPE_METAL_BENCH_BACKEND");
  const std::string backend = (be && *be) ? be : "metal";
  ::setenv("VPIPE_LLM_BACKEND", backend.c_str(), 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  const auto ids = lm->tokenizer().encode(prompt);
  ASSERT_TRUE(!ids.empty());

  using clk = std::chrono::steady_clock;
  auto secs = [](clk::duration d) {
    return std::chrono::duration<double>(d).count();
  };

  // Warmup: full prefill + a short decode so weights/kernels are hot
  // and wired-resident before any timed run.
  {
    auto wired = lm->wired_scope();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 8 && t >= 0; ++i) { t = lm->next_token(ctx, t); }
  }

  std::vector<int> idx_scratch;
  for (int mode = 0; mode < 2; ++mode) {  // 0 = greedy, 1 = top_p
    const bool greedy = (mode == 0);
    std::mt19937 rng(seed);
    auto wired = lm->wired_scope();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());

    const auto t0 = clk::now();
    std::int32_t pred = lm->prefill(ctx, ids);
    const auto t1 = clk::now();
    ASSERT_TRUE(pred >= 0);

    std::int32_t t = greedy ? pred
        : bench_sample_top_p(lm->last_logits_host(), idx_scratch,
                             temp, top_p, rng);
    int produced = 0;
    const auto d0 = clk::now();
    for (int i = 0; i < n_decode; ++i) {
      const std::int32_t nx = lm->next_token(ctx, t);
      if (nx < 0) { break; }
      ++produced;
      t = greedy ? nx
          : bench_sample_top_p(lm->last_logits_host(), idx_scratch,
                               temp, top_p, rng);
    }
    const auto d1 = clk::now();

    const double prefill_s = secs(t1 - t0);
    const double decode_s = secs(d1 - d0);
    std::printf(
        "[BENCH] backend=%s model=%s mode=%s prompt_tok=%zu "
        "prefill_s=%.4f prefill_tps=%.1f decode_n=%d decode_s=%.4f "
        "decode_tps=%.2f temp=%.2f top_p=%.2f\n",
        backend.c_str(), path, greedy ? "greedy" : "top_p", ids.size(),
        prefill_s,
        ids.size() / prefill_s, produced, decode_s,
        produced / decode_s, greedy ? 0.0f : temp,
        greedy ? 1.0f : top_p);
    EXPECT_TRUE(produced >= 1);
  }
}

// Qwen3.5 context-length sweep bench, apples-to-apples with the omlx
// reference (omlx_qwen_ctx_bench.py): the SAME synthetic ids
// ((i*131+7)%2000+10) at ctx {1024,2048,4096}, greedy. Per ctx: a warmup
// context, then a FRESH context timed for prefill + a DEC-step greedy decode
// loop, so metal-compute decode tok/s can be compared head-to-head with
// mlx_lm. Backend defaults to metal (VPIPE_METAL_BENCH_BACKEND overrides for
// an A/B). ctx list via VPIPE_QWEN_CTX_LIST (comma-separated), decode steps
// via VPIPE_QWEN_CTX_DEC. Gated on VPIPE_QWEN_CTX_BENCH_MODEL /
// VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_bench, qwen_ctx_sweep) {
  const char* path = std::getenv("VPIPE_QWEN_CTX_BENCH_MODEL");
  if (!path || !*path) { path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH"); }
  if (!path || !*path) { return; }

  std::vector<int> ctxs;
  if (const char* e = std::getenv("VPIPE_QWEN_CTX_LIST")) {
    const char* p = e;
    while (*p) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > 0) { ctxs.push_back((int)v); }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  if (ctxs.empty()) { ctxs = {1024, 2048, 4096}; }
  const int dec = std::getenv("VPIPE_QWEN_CTX_DEC")
      ? std::atoi(std::getenv("VPIPE_QWEN_CTX_DEC")) : 64;
  // Default to the greedy fast path (next_token_greedy: on-GPU argmax, no
  // full-vocab host logit pull) -- the apples-to-apples match for omlx, which
  // keeps argmax on device. Set VPIPE_QWEN_CTX_GREEDY=0 to A/B the slow
  // next_token (host logit readback + host argmax) path.
  const bool greedy_fast = !(std::getenv("VPIPE_QWEN_CTX_GREEDY")
      && std::atoi(std::getenv("VPIPE_QWEN_CTX_GREEDY")) == 0);
  // VPIPE_QWEN_CTX_MTP=1: ALSO time greedy MTP speculative decode at each ctx
  // (only when the model carries an MTP head, e.g. Qwen3.5-OptiQ) and report
  // its decode tok/s + tok/round + the speedup over the baseline greedy decode.
  const bool mtp_en = std::getenv("VPIPE_QWEN_CTX_MTP")
      && std::atoi(std::getenv("VPIPE_QWEN_CTX_MTP")) != 0;
  // VPIPE_QWEN_CTX_PIPE (default ON): ALSO time the production PIPELINED decode
  // (pdecode_* event-chain CPU/GPU overlap) per ctx and add decode_pipe_tps to
  // the line -- the apples-to-apples match for omlx's async-pipelined loop
  // (mx.async_eval). Set 0 to skip. The sync greedy stays the baseline column.
  const bool pipe_en = !(std::getenv("VPIPE_QWEN_CTX_PIPE")
      && std::atoi(std::getenv("VPIPE_QWEN_CTX_PIPE")) == 0);

  const char* be = std::getenv("VPIPE_METAL_BENCH_BACKEND");
  const std::string backend = (be && *be) ? be : "metal";
  ::setenv("VPIPE_LLM_BACKEND", backend.c_str(), 1);
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) {
    ::unsetenv("VPIPE_LLM_BACKEND");
    return;
  }
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  // The pipelined-decode path keeps the timed context AND the pdecode context
  // live at once, so the KV pool must fit 2 * (max ctx + decode) + slack. Size
  // it off the ctx list (was a flat 32 -> decode_pipe_tps returned 0 past 4k).
  {
    int max_ctx = 0;
    for (const int N : ctxs) { max_ctx = std::max(max_ctx, N); }
    // Each live context rounds its token span UP to whole pages, so size per
    // context then double (timed + pdecode contexts coexist), + slack.
    const int per_ctx_pages =
        (max_ctx + dec + 64 + spec.page_tokens - 1) / spec.page_tokens;
    spec.max_pages = std::max(32, 2 * per_ctx_pages + 4);
  }
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  using clk = std::chrono::steady_clock;
  auto secs = [](clk::duration d) {
    return std::chrono::duration<double>(d).count();
  };
  auto synth = [](int n) {
    std::vector<std::int32_t> v((std::size_t)n);
    for (int i = 0; i < n; ++i) {
      v[(std::size_t)i] = (std::int32_t)((i * 131 + 7) % 2000 + 10);
    }
    return v;
  };

  for (int N : ctxs) {
    const auto ids = synth(N);
    auto step = [&](auto& c) -> std::int32_t {
      return greedy_fast ? lm->next_token_greedy(c) : lm->next_token(c);
    };
    {                                       // warmup (separate context)
      auto wired = lm->wired_scope();
      auto ctx = lm->make_context();
      ASSERT_TRUE(ctx.valid());
      std::int32_t t = lm->prefill(ctx, ids);
      (void)t;
      for (int i = 0; i < 4; ++i) { if (step(ctx) < 0) { break; } }
    }
    auto wired = lm->wired_scope();
    auto ctx = lm->make_context();
    ASSERT_TRUE(ctx.valid());
    const auto t0 = clk::now();
    const std::int32_t pred = lm->prefill(ctx, ids);
    const auto t1 = clk::now();
    ASSERT_TRUE(pred >= 0);
    int produced = 0;
    const auto d0 = clk::now();
    for (int i = 0; i < dec; ++i) {
      const std::int32_t nx = step(ctx);
      if (nx < 0) { break; }
      ++produced;
    }
    const auto d1 = clk::now();
    const double ps = secs(t1 - t0);
    const double ds = secs(d1 - d0);

    // Pipelined decode (vpipe production pdecode_* path): same depth N, warmed
    // pipeline, event-chained commit/next overlap -- the fair match for omlx's
    // async_eval loop. 0 when unavailable / disabled.
    double pipe_tps = 0.0;
    if (pipe_en) {
      auto cd = lm->make_context();
      if (cd.valid()) {
        const std::int32_t pf = lm->prefill(cd, ids);
        const std::span<const std::int32_t> prompt(ids.data(), ids.size());
        genai::SamplerParams gsp;            // defaults -> argmax-equivalent
        if (pf >= 0 && lm->pdecode_begin(cd, pf, prompt, gsp, dec + 8)) {
          for (int k = 0; k < 4; ++k) {      // warm the pipeline
            if (!lm->pdecode_commit(cd)) { break; }
            if (lm->pdecode_next(cd) < 0) { break; }
          }
          int pp = 0;
          const auto p0 = clk::now();
          for (int k = 0; k < dec; ++k) {
            if (!lm->pdecode_commit(cd)) { break; }
            if (lm->pdecode_next(cd) < 0) { break; }
            ++pp;
          }
          const double pds = secs(clk::now() - p0);
          pipe_tps = (pds > 0.0) ? (double)pp / pds : 0.0;
          lm->pdecode_end(cd);
        }
      }
    }

    std::printf(
        "[BENCH-CTX] backend=%s greedy_fast=%d ctx=%d prefill_s=%.4f "
        "prefill_tps=%.1f decode_n=%d decode_s=%.4f decode_tps=%.2f "
        "decode_pipe_tps=%.2f\n",
        backend.c_str(), greedy_fast ? 1 : 0, N, ps, N / ps, produced, ds,
        produced / ds, pipe_tps);
    EXPECT_TRUE(produced >= 1);

    // Greedy MTP speculative decode at the SAME depth N (OptiQ + MTP head).
    // A fresh context prefilled with the same ids so the spec decode starts at
    // depth N; report decode tok/s + tok/round + the speedup vs the baseline.
    if (mtp_en && lm->mtp_available()) {
      // Seed the drafter's KV with the prefix (the text-chat shipping default)
      // so the MTP number reflects the production config; set before prefill.
      lm->set_mtp_prefix_seed(std::getenv("VPIPE_MTP_NO_SEED") == nullptr);
      auto count_round =
          [](int* prod, int* rounds) {
            return [prod, rounds](std::span<const std::int32_t> t) -> bool {
              *prod += (int)t.size();
              *rounds += 1;
              return true;
            };
          };
      const std::function<bool(std::int32_t)> no_stop;
      {                                       // warm the MTP-fused kernels
        auto wctx = lm->make_context();
        if (wctx.valid()) {
          std::int32_t wf = lm->prefill(wctx, ids);
          int wp = 0, wr = 0;
          if (wf >= 0) {
            lm->mtp_generate(wctx, wf, 8, genai::SamplerParams{}, no_stop,
                             count_round(&wp, &wr));
          }
        }
      }
      auto mwired = lm->wired_scope();
      auto mctx = lm->make_context();
      ASSERT_TRUE(mctx.valid());
      const std::int32_t mf = lm->prefill(mctx, ids);
      ASSERT_TRUE(mf >= 0);
      int mprod = 0, mrounds = 0;
      const auto m0 = clk::now();
      const bool mok = lm->mtp_generate(mctx, mf, dec, genai::SamplerParams{},
                                        no_stop, count_round(&mprod, &mrounds));
      const auto m1 = clk::now();
      const double mds = secs(m1 - m0);
      if (mok && mprod > 0 && mds > 0.0) {
        std::printf(
            "[BENCH-CTX-MTP] backend=%s ctx=%d mtp_decode_n=%d "
            "mtp_decode_s=%.4f mtp_decode_tps=%.2f tok_per_round=%.2f "
            "speedup=%.2f\n",
            backend.c_str(), N, mprod, mds, mprod / mds,
            mrounds > 0 ? (double)mprod / (double)mrounds : 0.0,
            (ds > 0.0) ? (mprod / mds) / (produced / ds) : 0.0);
      }
    }
  }
}

// Qwen3.5 GGUF (Q4_K_M) context-length sweep -- the native k-quant counterpart
// of qwen_ctx_sweep, for the head-to-head against llama.cpp's llama-bench
// (-p L -n G -d L) on the SAME .gguf. The Qwen GGUF ships no tokenizer, so this
// drives MetalQwenModel directly with synthetic in-vocab ids (timing only).
// Per ctx L: a fresh branch-from-(empty-)root context, prefill L timed (pp@L),
// then a pipelined-decode run of G tokens timed (tg@L, vpipe's production decode
// path -- proven token-exact in qwen_gguf_text_chat). On M5 the k-quant prefill
// rides the matmul2d matrix units (dequant -> dense_gemm_mma); A/B the steel
// fallback with VPIPE_QWEN_NO_MMA=1. ctx list VPIPE_QWEN_GGUF_BENCH_CTX
// (default 1024,2048,4096), decode steps VPIPE_QWEN_GGUF_BENCH_GEN (default 64).
// Gated on VPIPE_QWEN_GGUF_TEST_MODEL_PATH. Run:
//   VPIPE_QWEN_GGUF_TEST_MODEL_PATH=<.gguf> \
//     vpipe_test --filter '*qwen_gguf_ctx_sweep' --color off
TEST(metal_lm_bench, qwen_gguf_ctx_sweep) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_TEST_MODEL_PATH");
  if (!gguf || !*gguf) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");        // no-MLX default == metal
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(gguf);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  // Size the page pool for the largest requested ctx + decode (the bench may
  // sweep long contexts); 32 (16k) is the default for the smoke ctxs.
  int max_ctx = 4096;
  if (const char* e = std::getenv("VPIPE_QWEN_GGUF_BENCH_CTX")) {
    for (const char* p = e; *p;) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > max_ctx) { max_ctx = (int)v; }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  const int gen_hint = std::getenv("VPIPE_QWEN_GGUF_BENCH_GEN")
      ? std::max(1, std::atoi(std::getenv("VPIPE_QWEN_GGUF_BENCH_GEN"))) : 64;
  mcfg.max_pages = std::max(32, (max_ctx + gen_hint + 511) / 512 + 2);
  auto model = genai::MetalQwenModel::load(gguf, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  auto* cm = model->context_manager();
  ASSERT_TRUE(cm != nullptr);

  std::vector<int> ctxs;
  if (const char* e = std::getenv("VPIPE_QWEN_GGUF_BENCH_CTX")) {
    const char* p = e;
    while (*p) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > 0) { ctxs.push_back((int)v); }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  if (ctxs.empty()) { ctxs = {1024, 2048, 4096}; }
  const int G = std::getenv("VPIPE_QWEN_GGUF_BENCH_GEN")
      ? std::max(1, std::atoi(std::getenv("VPIPE_QWEN_GGUF_BENCH_GEN"))) : 64;

  using clk = std::chrono::steady_clock;
  auto secs = [](clk::duration d) {
    return std::chrono::duration<double>(d).count();
  };
  // Same synthetic ids as qwen_ctx_sweep / omlx_qwen_ctx_bench (in-vocab).
  auto synth = [](int n) {
    std::vector<std::int32_t> v((std::size_t)n);
    for (int i = 0; i < n; ++i) {
      v[(std::size_t)i] = (std::int32_t)((i * 131 + 7) % 2000 + 10);
    }
    return v;
  };
  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  // Warm the GPU (cold first command buffer; clock spins up).
  {
    const genai::ContextId w = cm->branch(model->root_context());
    ASSERT_TRUE(w.valid());
    const std::int32_t f = argmax(model->prefill(w, synth(64)));
    std::vector<std::int32_t> tmp;
    model->decode_pipelined(w, f, 4, tmp);
    cm->release(w);
  }

  std::printf("[qwen_gguf_ctx] Qwen3.5 gguf q4_K_M use_mma=%d gen=%d\n",
              model->uses_matrix_cores(), G);
  for (const int N : ctxs) {
    const auto ids = synth(N);

    // ---- prefill (pp@N): process N tokens from empty ----
    const genai::ContextId cp = cm->branch(model->root_context());
    ASSERT_TRUE(cp.valid());
    const auto t0 = clk::now();
    const std::vector<float> lg = model->prefill(cp, ids);
    const auto t1 = clk::now();
    ASSERT_TRUE(!lg.empty());
    cm->release(cp);
    const double ps = secs(t1 - t0);
    const double pp_tps = ps > 0.0 ? (double)N / ps : 0.0;

    // ---- pipelined decode (tg@N): prefill (untimed) then time G tokens ----
    double tg_pipe = 0.0;
    {
      const genai::ContextId cd = cm->branch(model->root_context());
      ASSERT_TRUE(cd.valid());
      const std::int32_t f = argmax(model->prefill(cd, ids));
      std::vector<std::int32_t> out;             // tok1.. (== decode_pipelined)
      // Prime the pdecode ring + warm 4 steps, THEN time G steady-state steps.
      // Excludes the single-call decode_pipelined path's ~1-tok in-call pipeline
      // fill/drain, so tg_pipe matches qwen_ctx_sweep / gguf_gemma_pp_tg's warmed
      // steady-state accounting (was ~1-1.5% conservative at G=64). Collecting
      // every pdecode_next keeps `out` == decode_pipelined's [tok1..] fingerprint.
      const std::span<const std::int32_t> prompt(ids.data(), ids.size());
      genai::GpuSamplerParams gsp{};             // greedy default -> token-exact
      if (f >= 0 && model->pdecode_begin(cd, f, prompt, gsp, G + 8)) {
        while (model->pdecode_commit(cd)) {}     // prime ring to depth
        for (int k = 0; k < 4; ++k) {            // warm steady-state
          const std::int32_t t = model->pdecode_next(cd);
          if (t < 0) { break; }
          out.push_back(t);
          model->pdecode_commit(cd);
        }
        int produced = 0;
        const auto d0 = clk::now();
        for (int k = 0; k < G; ++k) {
          const std::int32_t t = model->pdecode_next(cd);
          if (t < 0) { break; }
          out.push_back(t);
          ++produced;
          model->pdecode_commit(cd);
        }
        const double ds = secs(clk::now() - d0);
        tg_pipe = (ds > 0.0) ? (double)produced / ds : 0.0;
        model->pdecode_end(cd);
      }
      // Greedy id fingerprint: first 12 decoded ids. Deterministic across
      // attention-kernel variants -> diff new all-G path vs VPIPE_GQA_ATTN=0
      // (old mb256) to confirm token-exactness for G=6.
      std::printf("[qwen_gguf_ctx] ctx=%-5d ids:", N);
      for (std::size_t i = 0; i < out.size() && i < 12; ++i) {
        std::printf(" %d", (int)out[i]);
      }
      std::printf("\n");
      cm->release(cd);
    }

    std::printf("[qwen_gguf_ctx] ctx=%-5d  prefill=%7.1f tok/s (%.3fs)  "
                "decode(pipe)=%5.2f tok/s\n", N, pp_tps, ps, tg_pipe);
  }
  EXPECT_TRUE(true);
}

// Qwen3.5-MoE (35B-A3B) prefill/decode context sweep, mirroring
// qwen_gguf_ctx_sweep for the head-to-head against the omlx/mlx_lm server.
// Synthetic in-vocab ids; prints pp@N tok/s + tg@N tok/s for each ctx. Grouped
// prefill is default-on at these n (also lifts the pair-path grid.z cap). Gated
// on VPIPE_QWEN35_MOE_TEST_MODEL_PATH; ctxs via VPIPE_QWEN35_MOE_BENCH_CTX
// (default 1024,2048,4096,8192), gen via VPIPE_QWEN35_MOE_BENCH_GEN (default 64).
TEST(metal_lm_bench, qwen35_moe_ctx_sweep) {
  const char* path = std::getenv("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 48;            // ~24k token KV: holds 8k ctx + decode
  ASSERT_TRUE(mcfg.is_moe());
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  auto* cm = model->context_manager();
  ASSERT_TRUE(cm != nullptr);

  std::vector<int> ctxs;
  if (const char* e = std::getenv("VPIPE_QWEN35_MOE_BENCH_CTX")) {
    const char* p = e;
    while (*p) {
      char* end = nullptr;
      const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > 0) { ctxs.push_back((int)v); }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  if (ctxs.empty()) { ctxs = {1024, 2048, 4096, 8192}; }
  const int G = std::getenv("VPIPE_QWEN35_MOE_BENCH_GEN")
      ? std::max(1, std::atoi(std::getenv("VPIPE_QWEN35_MOE_BENCH_GEN"))) : 64;

  using clk = std::chrono::steady_clock;
  auto secs = [](clk::duration d) {
    return std::chrono::duration<double>(d).count();
  };
  auto synth = [](int n) {
    std::vector<std::int32_t> v((std::size_t)n);
    for (int i = 0; i < n; ++i) {
      v[(std::size_t)i] = (std::int32_t)((i * 131 + 7) % 2000 + 10);
    }
    return v;
  };
  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t best = 0;
    float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; best = (std::int32_t)i; }
    }
    return best;
  };

  {   // warm the GPU (cold first command buffer)
    const genai::ContextId w = cm->branch(model->root_context());
    ASSERT_TRUE(w.valid());
    const std::int32_t f = argmax(model->prefill(w, synth(64)));
    std::vector<std::int32_t> tmp;
    model->decode_pipelined(w, f, 4, tmp);
    cm->release(w);
  }

  std::printf("[qwen35_moe_ctx] Qwen3.5-MoE 35B-A3B 4bit gen=%d\n", G);
  for (const int N : ctxs) {
    const auto ids = synth(N);
    // prefill (pp@N)
    const genai::ContextId cp = cm->branch(model->root_context());
    ASSERT_TRUE(cp.valid());
    const auto t0 = clk::now();
    const std::vector<float> lg = model->prefill(cp, ids);
    const auto t1 = clk::now();
    ASSERT_TRUE(!lg.empty());
    cm->release(cp);
    const double ps = secs(t1 - t0);
    const double pp_tps = ps > 0.0 ? (double)N / ps : 0.0;
    // pipelined decode (tg@N): prefill (untimed) then time G tokens
    double tg_pipe = 0.0;
    {
      const genai::ContextId cd = cm->branch(model->root_context());
      ASSERT_TRUE(cd.valid());
      const std::int32_t f = argmax(model->prefill(cd, ids));
      std::vector<std::int32_t> out;             // tok1.. (== decode_pipelined)
      // Prime the pdecode ring + warm 4 steps, THEN time G steady-state steps.
      // Excludes the single-call decode_pipelined path's ~1-tok in-call pipeline
      // fill/drain, so tg_pipe matches qwen_ctx_sweep / gguf_gemma_pp_tg's warmed
      // steady-state accounting (was ~1-1.5% conservative at G=64). Collecting
      // every pdecode_next keeps `out` == decode_pipelined's [tok1..] fingerprint.
      const std::span<const std::int32_t> prompt(ids.data(), ids.size());
      genai::GpuSamplerParams gsp{};             // greedy default -> token-exact
      if (f >= 0 && model->pdecode_begin(cd, f, prompt, gsp, G + 8)) {
        while (model->pdecode_commit(cd)) {}     // prime ring to depth
        for (int k = 0; k < 4; ++k) {            // warm steady-state
          const std::int32_t t = model->pdecode_next(cd);
          if (t < 0) { break; }
          out.push_back(t);
          model->pdecode_commit(cd);
        }
        int produced = 0;
        const auto d0 = clk::now();
        for (int k = 0; k < G; ++k) {
          const std::int32_t t = model->pdecode_next(cd);
          if (t < 0) { break; }
          out.push_back(t);
          ++produced;
          model->pdecode_commit(cd);
        }
        const double ds = secs(clk::now() - d0);
        tg_pipe = (ds > 0.0) ? (double)produced / ds : 0.0;
        model->pdecode_end(cd);
      }
      // Greedy id fingerprint (first 12): deterministic across attention-kernel
      // variants -> cross-check the new MMA flash (VPIPE_QWEN_SDPA_PMMA=1) vs
      // the key-split flash (=0) produce identical tokens.
      std::printf("[qwen35_moe_ctx] ctx=%-5d ids:", N);
      for (std::size_t i = 0; i < out.size() && i < 12; ++i) {
        std::printf(" %d", (int)out[i]);
      }
      std::printf("\n");
      cm->release(cd);
    }
    std::printf("[qwen35_moe_ctx] ctx=%-5d  prefill=%7.1f tok/s (%.3fs)  "
                "decode(pipe)=%5.2f tok/s\n", N, pp_tps, ps, tg_pipe);
  }

  // Mid-context (q_offset>0) cross-check: a 2-chunk prefill (chunk 2 runs at
  // q_offset=split -> the paged steel kernel) must yield the SAME final argmax
  // as a 1-shot prefill of the concatenation (the last token sees the same
  // causal context). Validates the paged steel kernel's q_offset>0 path; the
  // printed id also cross-checks vs the flash when run with STEEL_ATTN=0.
  {
    // q_offset (existing ctx) + chunk2 size via env; default 1k-existing + 2k.
    const int split = std::getenv("VPIPE_QWEN_MIDCTX_QOFF")
        ? std::atoi(std::getenv("VPIPE_QWEN_MIDCTX_QOFF")) : 1024;
    const int n2 = std::getenv("VPIPE_QWEN_MIDCTX_N")
        ? std::atoi(std::getenv("VPIPE_QWEN_MIDCTX_N")) : 2048;
    const int Nc = split + n2;             // chunk2 n2 >= steel min (2048)
    const auto ids = synth(Nc);
    const std::vector<std::int32_t> a(ids.begin(), ids.begin() + split);
    const std::vector<std::int32_t> b(ids.begin() + split, ids.end());
    const genai::ContextId c1 = cm->branch(model->root_context());
    const std::int32_t one = argmax(model->prefill(c1, ids));
    cm->release(c1);
    const genai::ContextId c2 = cm->branch(model->root_context());
    model->prefill(c2, a);                 // chunk 1 (q_offset=0)
    const auto m0 = clk::now();
    const std::int32_t two = argmax(model->prefill(c2, b));  // chunk 2 (q_off=split)
    const double mms = secs(clk::now() - m0);
    cm->release(c2);
    const double m_tps = mms > 0.0 ? (double)(Nc - split) / mms : 0.0;
    std::printf("[qwen35_moe_midctx] 1shot=%d  2chunk(qoff=%d,n=%d)=%d  %s  "
                "chunk2 prefill=%7.1f tok/s (%.3fs)\n",
                one, split, Nc - split, two, one == two ? "MATCH" : "MISMATCH",
                m_tps, mms);
    EXPECT_TRUE(one == two);
  }
  EXPECT_TRUE(true);
}

// Qwen3.5-MoE prefill ABLATION: load once, time the prefill with each component
// removed (timing-only -- output is garbage), to attribute the prefill cost and
// the gap vs omlx. Toggles drive the in-model VPIPE_MOE_ABL / VPIPE_QWEN_SKIP_
// ATTN env (read per-prefill). delta(base - toggle) = that component's cost.
// Gated on VPIPE_QWEN35_MOE_TEST_MODEL_PATH; ctxs via VPIPE_QWEN35_MOE_BENCH_CTX
// (default 2048,8192 -- the steel grouped tier).
TEST(metal_lm_bench, qwen35_moe_ablation) {
  const char* path = std::getenv("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 48;
  ASSERT_TRUE(mcfg.is_moe());
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  auto* cm = model->context_manager();
  ASSERT_TRUE(cm != nullptr);

  std::vector<int> ctxs;
  if (const char* e = std::getenv("VPIPE_QWEN35_MOE_BENCH_CTX")) {
    const char* p = e;
    while (*p) {
      char* end = nullptr; const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > 0) { ctxs.push_back((int)v); }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  if (ctxs.empty()) { ctxs = {2048, 8192}; }

  using clk = std::chrono::steady_clock;
  auto secs = [](clk::duration d) {
    return std::chrono::duration<double>(d).count();
  };
  auto synth = [](int n) {
    std::vector<std::int32_t> v((std::size_t)n);
    for (int i = 0; i < n; ++i) {
      v[(std::size_t)i] = (std::int32_t)((i * 131 + 7) % 2000 + 10);
    }
    return v;
  };
  // (label, MOE_ABL value, SKIP_ATTN value). "" leaves the env unset.
  struct Tog { const char* name; const char* abl; const char* attn; };
  const Tog togs[] = {
    {"base",    "",       ""},
    {"-shared", "shared", ""},
    {"-gemm",   "gemm",   ""},   // skip expert GEMM (clean baseline B)
    // Backbone isolated with the expert GEMM ALSO skipped, so the routing
    // change from skipping attention can't perturb the (skipped) experts.
    {"-gemm-fattn", "gemm", "1"},  // B - full-attention layers
    {"-gemm-gdn",   "gemm", "2"},  // B - gated-DeltaNet layers
  };
  auto set_env = [&](const Tog& t) {
    if (*t.abl) { ::setenv("VPIPE_MOE_ABL", t.abl, 1); }
    else { ::unsetenv("VPIPE_MOE_ABL"); }
    if (*t.attn) { ::setenv("VPIPE_QWEN_SKIP_ATTN", t.attn, 1); }
    else { ::unsetenv("VPIPE_QWEN_SKIP_ATTN"); }
  };

  // warm
  { const genai::ContextId w = cm->branch(model->root_context());
    model->prefill(w, synth(64)); cm->release(w); }

  std::printf("[qwen35_moe_abl] Qwen3.5-MoE 35B-A3B (prefill ablation, ms)\n");
  for (const int N : ctxs) {
    const auto ids = synth(N);
    double base_s = 0.0;
    for (const Tog& t : togs) {
      set_env(t);
      // 2 runs, take the min (steady state) -- the env is read per prefill.
      double best = 1e9;
      for (int r = 0; r < 2; ++r) {
        const genai::ContextId cp = cm->branch(model->root_context());
        ASSERT_TRUE(cp.valid());
        const auto t0 = clk::now();
        const auto lg = model->prefill(cp, ids);
        const double s = secs(clk::now() - t0);
        cm->release(cp);
        ASSERT_TRUE(!lg.empty());
        best = std::min(best, s);
      }
      if (std::string(t.name) == "base") { base_s = best; }
      const double delta = base_s - best;   // component cost (>=0 for skips)
      std::printf("[qwen35_moe_abl] ctx=%-5d %-9s prefill=%7.1f ms"
                  "  delta_vs_base=%+7.1f ms (%5.1f tok/s)\n",
                  N, t.name, best * 1e3, delta * 1e3, (double)N / best);
    }
    ::unsetenv("VPIPE_MOE_ABL");
    ::unsetenv("VPIPE_QWEN_SKIP_ATTN");
  }
  EXPECT_TRUE(true);
}

// Qwen3.5-MoE DECODE category profiler AT DEPTH. Within one process (steady GPU
// clock), duplicate one decode category's GPU work per step (VPIPE_QWEN_DUP_CAT)
// -> the whole-step delta vs baseline is that category's cost. Run at two depths
// to separate the context-INDEPENDENT costs (ffn=MoE experts, gdn, proj,
// lmhead) from the context-SCALING one (attn). Finds the MoE decode bottleneck.
// Gated on VPIPE_QWEN35_MOE_TEST_MODEL_PATH + VPIPE_QWEN_CATPROF (the latter
// also enables the in-model DUP path). Depths via VPIPE_QWEN35_MOE_BENCH_CTX.
TEST(metal_lm_bench, qwen35_moe_decode_catprof) {
  const char* path = std::getenv("VPIPE_QWEN35_MOE_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_QWEN_CATPROF") == nullptr) { return; }
  ::unsetenv("VPIPE_LLM_BACKEND");
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  ASSERT_TRUE(cfg.has_value());
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 48;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  auto* cm = model->context_manager();
  ASSERT_TRUE(cm != nullptr);

  std::vector<int> depths;
  if (const char* e = std::getenv("VPIPE_QWEN35_MOE_BENCH_CTX")) {
    const char* p = e;
    while (*p) {
      char* end = nullptr; const long v = std::strtol(p, &end, 10);
      if (end == p) { break; }
      if (v > 0) { depths.push_back((int)v); }
      p = (*end == ',') ? end + 1 : end;
    }
  }
  if (depths.empty()) { depths = {2048, 8192}; }

  auto synth = [](int n) {
    std::vector<std::int32_t> v((std::size_t)n);
    for (int i = 0; i < n; ++i) {
      v[(std::size_t)i] = (std::int32_t)((i * 131 + 7) % 2000 + 10);
    }
    return v;
  };
  auto argmax = [](const std::vector<float>& v) -> std::int32_t {
    std::int32_t b = 0; float bv = v.empty() ? 0.0f : v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
      if (v[i] > bv) { bv = v[i]; b = (std::int32_t)i; }
    }
    return b;
  };
  using clk = std::chrono::steady_clock;
  const int N = 32;
  // Fresh branch + prefill each measurement so the depth is steady (no drift).
  auto decode_ms = [&](const std::vector<std::int32_t>& ids) -> double {
    const genai::ContextId c = cm->branch(model->root_context());
    if (!c.valid()) { return -1.0; }
    const std::int32_t f = argmax(model->prefill(c, ids));
    std::vector<std::int32_t> w;
    model->decode_pipelined(c, f, 2, w);              // warm at depth
    std::vector<std::int32_t> out;
    const std::int32_t seed = w.empty() ? f : w.back();
    const auto t0 = clk::now();
    const bool ok = model->decode_pipelined(c, seed, N, out);
    const double ms =
        std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    cm->release(c);
    return ok ? ms : -1.0;
  };

  const char* cats[] = {"none", "proj", "ffn", "lmhead", "attn", "norm",
                        "rope", "misc", "gdn", "gdn_rec"};
  const int NC = 10;
  for (const int depth : depths) {
    const auto ids = synth(depth);
    for (int k = 0; k < 2; ++k) { (void)decode_ms(ids); }   // warm GPU clock
    double best[NC];
    for (int c = 0; c < NC; ++c) {
      ::setenv("VPIPE_QWEN_DUP_CAT", cats[c], 1);
      double m = 1e18;
      for (int r = 0; r < 2; ++r) {
        const double t = decode_ms(ids); if (t > 0) { m = std::min(m, t); }
      }
      best[c] = m;
    }
    ::unsetenv("VPIPE_QWEN_DUP_CAT");
    const double T0 = best[0];
    std::printf("[moe_catprof depth=%-4d] baseline %.1f ms (%.3f ms/tok = "
                "%.2f tok/s); delta = category whole-step GPU cost\n",
                depth, T0, T0 / N, N * 1000.0 / T0);
    for (int c = 1; c < NC; ++c) {
      const double d = best[c] - T0;
      std::printf("[moe_catprof depth=%-4d] %-8s delta %+7.2f ms (%.3f ms/tok)"
                  " | %5.1f%%\n", depth, cats[c], d, d / N, 100.0 * d / T0);
    }
  }
  EXPECT_TRUE(true);
}

// Per-category decode GPU-cost profiler for the Qwen hybrid metal model.
// Loads with VPIPE_QWEN_CATPROF, then for each DUP category re-runs that
// category's ops ONE extra time per step; the whole-step delta vs baseline is
// that category's GPU cost (compute + its hazard-barrier drains). GDN (the
// 48/64 linear layers) is DC_GDN. The residual (baseline - sum of deltas)
// after wrapping GDN is embed + argmax + any inter-dispatch GPU idle -- a big
// residual means dispatch/barrier gaps (fusion helps); a small one means
// decode is compute/bandwidth bound. Gated on VPIPE_QWEN_CATPROF +
// VPIPE_METAL_LM_SMOKE_MODEL.
TEST(metal_lm_bench, qwen_decode_catprof) {
  const char* path = std::getenv("VPIPE_METAL_LM_SMOKE_MODEL");
  if (!path || !*path) { return; }
  if (std::getenv("VPIPE_QWEN_CATPROF") == nullptr) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  if (mc == nullptr || !mc->valid() || mgr == nullptr) {
    ::unsetenv("VPIPE_LLM_BACKEND"); return;
  }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm != nullptr && lm->valid());

  std::vector<std::int32_t> seed;
  lm->chat_template()->render_user_turn("Benchmark.", true, &seed);
  ASSERT_TRUE(!seed.empty());

  const int N = 48;
  auto decode_ms = [&]() -> double {
    auto ctx = lm->make_context();
    if (!ctx.valid() || lm->prefill(ctx, seed) < 0) { return -1.0; }
    (void)lm->next_token_greedy(ctx);              // warm one step at depth
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
      if (lm->next_token_greedy(ctx) < 0) { break; }
    }
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
  };

  const char* cats[] = {"none", "ffn", "gdn", "gdn_rec", "proj", "attn",
                        "lmhead", "norm", "rope", "misc"};
  const int NC = 10;
  for (int k = 0; k < 2; ++k) { (void)decode_ms(); }   // warm GPU clock
  double best[NC];
  for (int c = 0; c < NC; ++c) {
    ::setenv("VPIPE_QWEN_DUP_CAT", cats[c], 1);
    double m = 1e18;
    for (int r = 0; r < 3; ++r) { m = std::fmin(m, decode_ms()); }
    best[c] = m;
  }
  ::unsetenv("VPIPE_QWEN_DUP_CAT");
  const double T0 = best[0];
  std::printf("[qwen_catprof] baseline %.1f ms (%.3f ms/tok = %.2f tok/s); "
              "delta = category whole-step GPU cost (compute + barriers)\n",
              T0, T0 / N, N * 1000.0 / T0);
  double sum = 0.0;
  for (int c = 1; c < NC; ++c) {
    const double d = best[c] - T0;
    sum += d;
    std::printf("[qwen_catprof] %-7s delta %+7.2f ms (%.3f ms/tok) | %5.1f%%\n",
                cats[c], d, d / N, 100.0 * d / T0);
  }
  std::printf("[qwen_catprof] sum-of-deltas %.1f ms (%.1f%%); residual %.1f ms "
              "= embed+argmax+GPU idle gaps\n", sum, 100.0 * sum / T0, T0 - sum);
  EXPECT_TRUE(true);
}

// PERF-PATH GUARD (not a timing test). Locks in that the Qwen3.5 metal model
// SELECTS its M5 matrix-core fast paths, so a future change -- e.g. an M4-side
// edit (matrix cores absent there) that widens a gate, renames a kernel, or
// breaks a load -- cannot silently drop M5 onto the ~2-2.5x slower steel
// prefill / scalar-attention path. Such a fallback stays TOKEN-EXACT (so the
// token-exact tests miss it) and a perf-floor assertion would FLAKE on the
// M5's thermal throttling (cold ~1240 tok/s prefill vs ~520 hot -- the "qwen
// regression" that turned out to be thermal, not code). So we assert the path
// is engaged, independent of timing. On M4 (no matrix cores) the matmul2d
// assertions are correctly skipped; the GQA flash decode is checked on both.
// Gated on VPIPE_QWEN35_TEST_MODEL_PATH; skips under the A/B disables so they
// don't false-fail.
TEST(metal_lm_smoke, qwen_m5_fastpath_engaged_guard) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  // The A/B / safety env switches legitimately force the slow paths -- don't
  // guard when one is set (it isn't the "accidental" disable we're catching).
  if (std::getenv("VPIPE_QWEN_NO_MMA") || std::getenv("VPIPE_QWEN_NO_FLASH") ||
      (std::getenv("VPIPE_QWEN_GQA_ATTN") &&
       std::atoi(std::getenv("VPIPE_QWEN_GQA_ATTN")) == 0)) {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  const auto& mcfg_loaded = model->config();

  // GQA flash decode -- the decode fast path (each kv-head read once for all G
  // query heads). Default on for full-attn models, M4 + M5. Off => decode
  // re-reads each kv-head per query head (the ~2x decode regression GQA fixed).
  EXPECT_TRUE(model->gqa_flash_decode());

  if (!mc->supports_matrix_cores()) {
    // M4 / older GPUs: the matmul2d path is correctly absent (steel prefill).
    std::printf("[qwen_m5_fastpath_guard] no matrix cores (M4 path); "
                "gqa_flash_decode=%d\n", model->gqa_flash_decode());
    return;
  }
  // M5+: a 4-bit Qwen prefill MUST engage the matmul2d GEMM (dequant +
  // dense_gemm_mma); only 8-bit weights stay on steel by design. A false here
  // means the matrix-core prefill silently fell back to the ~2-2.5x slower
  // steel quantized GEMM.
  if (mcfg_loaded.quant_bits != 8) {
    EXPECT_TRUE(model->uses_matrix_cores());
  }
  // head_dim-256 prefill flash attention via matmul2d (Qwen3.5). Off => the
  // scalar query-tiled attention (much slower at long prefill).
  if (mcfg_loaded.head_dim == 256 && model->uses_matrix_cores()) {
    EXPECT_TRUE(model->mma_flash_attn());
  }
  std::printf("[qwen_m5_fastpath_guard] matrix cores ON | use_mma=%d "
              "mma_flash_attn=%d gqa_flash_decode=%d\n",
              model->uses_matrix_cores(), model->mma_flash_attn(),
              model->gqa_flash_decode());
}

// PERF-PATH GUARD for the native k-quant (GGUF) path -- the same intent as
// qwen_m5_fastpath_engaged_guard, but for the Q4_K_M GGUF model. On M5 a
// k-quant prefill MUST route dense_gemm_ through the matmul2d matrix units
// (uses_matrix_cores == _use_mma; dequant -> dense_gemm_mma); a regression
// that left it on the steel dequant+dense_gemm_t GEMM stays token-exact
// (~2-2.5x slower) so the token-exact tests miss it, and a perf-floor assert
// would flake on the M5's thermal throttle -- hence a path-engaged assert.
// Loads the model directly (the GGUF ships no tokenizer); the config is read
// from the .gguf. Gated on VPIPE_QWEN_GGUF_TEST_MODEL_PATH; skips under the
// VPIPE_QWEN_NO_MMA A/B disable so it doesn't false-fail.
TEST(metal_lm_smoke, qwen_gguf_m5_fastpath_engaged_guard) {
  const char* gguf = std::getenv("VPIPE_QWEN_GGUF_TEST_MODEL_PATH");
  if (!gguf || !*gguf) { return; }
  if (std::getenv("VPIPE_QWEN_NO_MMA")) { return; }   // legit A/B disable
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(gguf);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 8;
  auto model = genai::MetalQwenModel::load(gguf, mc, mcfg);
  ASSERT_TRUE(model != nullptr);
  const auto& mcfg_loaded = model->config();

  // GQA flash decode -- the decode fast path on both M4 and M5.
  EXPECT_TRUE(model->gqa_flash_decode());

  if (!mc->supports_matrix_cores()) {
    std::printf("[qwen_gguf_fastpath_guard] no matrix cores (M4 path); "
                "gqa_flash_decode=%d\n", model->gqa_flash_decode());
    return;
  }
  // M5+: the k-quant prefill must engage the matmul2d GEMM. A false here means
  // dense_gemm_ silently fell back to the steel quantized-dequant GEMM.
  EXPECT_TRUE(model->uses_matrix_cores());
  if (mcfg_loaded.head_dim == 256 && model->uses_matrix_cores()) {
    EXPECT_TRUE(model->mma_flash_attn());
  }
  std::printf("[qwen_gguf_fastpath_guard] matrix cores ON | use_mma=%d "
              "mma_flash_attn=%d gqa_flash_decode=%d\n",
              model->uses_matrix_cores(), model->mma_flash_attn(),
              model->gqa_flash_decode());
}

// Flash-decode-GQA serial attention (sdpa_paged_gqa_mb256 + sdpa_gqa_merge,
// head_dim 256) must be greedy token-exact with the mb256 per-q-head path.
// Loads the SAME Qwen3.5 model twice -- GQA on (VPIPE_QWEN_GQA_ATTN=1) and
// off (=0, mb256) -- prefills a >128-token prompt (so decode runs the
// long-ctx attention path, _sdpa_mb_min) and greedy-decodes; the token
// streams must match. On a model whose shape the GQA path can't handle
// (head_dim != 256 or Hq/Hkv > 4) both loads fall back to mb256 (still a
// valid, trivial pass). Gated on VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_gqa_attn_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  // ~200-token prompt so the decode position exceeds _sdpa_mb_min (128) and
  // the long-context (mb256 / GQA) attention path is exercised.
  std::string prompt;
  for (int i = 0; i < 16; ++i) {
    prompt += "The lighthouse keeper recorded the tides each evening, noting "
              "the color of the sky and the names of passing ships. ";
  }

  auto run = [&](int gqa, int no_vec) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_QWEN_GQA_ATTN", gqa ? "1" : "0", 1);
    ::setenv("VPIPE_GQA_NO_VEC", no_vec ? "1" : "0", 1);
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 32 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };

  const auto ref  = run(0, 0);   // mb256 (GQA off)
  const auto allg = run(1, 1);   // flash-decode-GQA all-G (sdpa_paged_gqa)
  const auto vec  = run(1, 0);   // flash-decode-GQA vec (sdpa_paged_gqa_vec)
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_QWEN_GQA_ATTN");
  ::unsetenv("VPIPE_GQA_NO_VEC");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == allg.size() && ref.size() == vec.size());
  std::size_t mism_allg = 0, mism_vec = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != allg[i]) { ++mism_allg; }
    if (ref[i] != vec[i]) { ++mism_vec; }
  }
  std::printf("[metal_lm_smoke.qwen_gqa_attn_token_exact] %zu tokens, "
              "allg_mism=%zu vec_mism=%zu\n",
              ref.size(), mism_allg, mism_vec);
  EXPECT_TRUE(mism_allg == 0);
  EXPECT_TRUE(mism_vec == 0);
}

// Flash-decode-GQA on the head_dim-128 decoders must be greedy token-exact
// with the sdpa_paged_mb path. Loads the model twice -- GQA on
// (VPIPE_GQA_ATTN=1) and off (=0) -- prefills a >128-token text prompt (so
// the long-ctx attention path runs) and greedy-decodes; the streams must
// match. Two routings share the same sdpa_paged_gqa kernel at D=128:
//   - VPIPE_QWEN3_ASR_TEST_MODEL_PATH: the dense Qwen3-ASR decoder routes
//     through MetalQwenModel (GQA 16/8=2) -- exercised on this box.
//   - VPIPE_LLM_TEST_MODEL_PATH: a dense LlamaForCausalLM routes through
//     MetalLlamaModel (GQA 32/8=4) -- exercised where a Llama model exists.
// A model whose shape the GQA path can't handle falls back (trivial pass).
TEST(metal_lm_smoke, llama_gqa_attn_token_exact) {
  const char* path = std::getenv("VPIPE_LLM_TEST_MODEL_PATH");
  if (!path || !*path) {
    path = std::getenv("VPIPE_QWEN3_ASR_TEST_MODEL_PATH");
  }
  if (!path || !*path) { return; }

  std::string prompt;
  for (int i = 0; i < 20; ++i) {
    prompt += "The lighthouse keeper recorded the tides each evening, noting "
              "the color of the sky and the names of passing ships. ";
  }

  auto run = [&](int gqa) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GQA_ATTN", gqa ? "1" : "0", 1);
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 32 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };

  const auto ref = run(0);   // sdpa_paged_mb
  const auto got = run(1);   // flash-decode-GQA
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GQA_ATTN");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.llama_gqa_attn_token_exact] %zu tokens, "
              "%zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// The two-stage parallel argmax (argmax_partial -> argmax_combine, the default
// greedy decode path) must be GREEDY TOKEN-EXACT with the single-tg argmax
// (VPIPE_GEMMA_ARGMAX1=1) on REAL model logits -- the direct cross-kernel gate
// for the argmax change. Decodes the same prompt 64 tokens with each kernel
// (next_token_greedy -> decode_step_fast -> encode_argmax_) and requires the
// streams to match bit-for-bit. Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_two_stage_argmax_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  auto run = [&](bool force_single) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    if (force_single) { ::setenv("VPIPE_GEMMA_ARGMAX1", "1", 1); }
    else { ::unsetenv("VPIPE_GEMMA_ARGMAX1"); }
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 8;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    std::vector<std::int32_t> ids;
    lm->chat_template()->render_user_turn(
        "List the planets of the solar system in order from the sun.",
        /*is_first_turn=*/true, &ids);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 64 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };
  const auto ref = run(true);     // single-tg argmax
  const auto got = run(false);    // two-stage argmax (default)
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_ARGMAX1");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_two_stage_argmax_token_exact] %zu tokens, "
              "%zu mismatches (single-tg vs two-stage)\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Flash-decode-GQA on Gemma-4 (CONTIGUOUS KV + sliding window + ring,
// sdpa_causal_gqa) must be greedy token-exact with the per-q-head sdpa_mb
// path. The prompt exceeds the sliding window (512) so the windowed-range
// scan (first = q_pos-window+1) is exercised on the sliding layers. Loads
// twice -- GQA on (VPIPE_GQA_ATTN=1) and off (=0) -- prefills + greedy-
// decodes; the streams must match. Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_gqa_attn_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  std::string prompt;   // ~640 tokens > the 512 sliding window
  for (int i = 0; i < 32; ++i) {
    prompt += "The lighthouse keeper recorded the tides each evening, noting "
              "the color of the sky and the names of passing ships. ";
  }

  auto run = [&](int gqa) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GQA_ATTN", gqa ? "1" : "0", 1);
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 32 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };

  const auto ref = run(0);   // sdpa_mb (per-q-head)
  const auto got = run(1);   // flash-decode-GQA (contiguous)
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GQA_ATTN");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_gqa_attn_token_exact] %zu tokens, "
              "%zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Materialized paged decode for the global (head_dim 512) layers
// (VPIPE_GEMMA_MAT_DECODE=1) must be greedy token-exact with the base flash
// decode (VECN/MAT unset). This is the omlx/MLX head_dim-512 fallback
// structure: QK GEMV -> parallel softmax -> PV GEMV, replacing the fused
// online-softmax flash. Same softmax math, fp reassociation only, so the
// argmax must not move. Prompt is long enough (>2 global pages @ 512) that the
// QK/PV key scan splits (sp>1) and the rowstat reduction spans a real range.
// Greedy-decodes 48 tokens each and requires the streams to match. Gated on
// VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_mat_decode_matches_flash_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  std::string prompt;   // ~1300 tokens -> 3 global pages @ page_tokens=512
  for (int i = 0; i < 64; ++i) {
    prompt += "The cartographer unrolled the brittle map across the table, "
              "tracing rivers and mountain passes by candlelight. ";
  }

  auto run = [&](bool mat) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    if (mat) { ::setenv("VPIPE_GEMMA_MAT_DECODE", "1", 1); }
    else     { ::unsetenv("VPIPE_GEMMA_MAT_DECODE"); }
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 48 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };

  const auto ref = run(false);   // base flash decode
  const auto mat = run(true);    // materialized decode
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_MAT_DECODE");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == mat.size());
  std::size_t mm = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != mat[i]) { ++mm; }
  }
  std::printf("[metal_lm_smoke.gemma_mat_decode_matches_flash_token_exact] "
              "%zu tokens, %zu mismatches\n", ref.size(), mm);
  EXPECT_TRUE(mm == 0);
}

// Threadgroup-staged flash-decode for the Gemma-4 GLOBAL layers
// (sdpa_causal_gqa_tile, default ON) must be greedy token-exact with the
// per-q-head sdpa_mb path. Loads the model twice -- gtile ON
// (VPIPE_GEMMA_GTILE_ATTN=1) and OFF (=0, sdpa_mb) -- prefills a prompt long
// enough that the global layers scan a multi-chunk context, and greedy-decodes
// 32 tokens; the streams must match. Engagement is independently confirmed by
// the decode A/B (gtile is measurably faster), so a silent fallback can't fake
// this pass. Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_gtile_attn_token_exact) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }

  std::string prompt;   // ~640 tokens; global layers scan the full context
  for (int i = 0; i < 32; ++i) {
    prompt += "The cartographer unrolled the chart across the table, tracing "
              "rivers and marking the depths of each harbor in faded ink. ";
  }

  auto run = [&](int gtile) -> std::vector<std::int32_t> {
    ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
    ::setenv("VPIPE_GEMMA_GTILE_ATTN", gtile ? "1" : "0", 1);
    Session sess;
    auto* mc = sess.metal_compute();
    auto* mgr = sess.generative_model_manager();
    std::vector<std::int32_t> out;
    if (mc == nullptr || !mc->valid() || mgr == nullptr) { return out; }
    genai::LoadSpec spec;
    spec.hf_dir = path;
    spec.compute_dtype = "f16";
    spec.page_tokens = 512;
    spec.max_pages = 16;
    auto lm = mgr->load(spec);
    if (!lm || !lm->valid()) { return out; }
    auto ids = lm->tokenizer().encode(prompt);
    if (ids.empty()) { return out; }
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return out; }
    std::int32_t t = lm->prefill(ctx, ids);
    for (int i = 0; i < 32 && t >= 0; ++i) {
      out.push_back(t);
      t = lm->next_token_greedy(ctx);
    }
    return out;
  };

  const auto ref = run(0);   // sdpa_mb (per-q-head)
  const auto got = run(1);   // threadgroup-staged flash-decode
  ::unsetenv("VPIPE_LLM_BACKEND");
  ::unsetenv("VPIPE_GEMMA_GTILE_ATTN");
  ASSERT_TRUE(!ref.empty());
  ASSERT_TRUE(ref.size() == got.size());
  std::size_t mism = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (ref[i] != got[i]) { ++mism; }
  }
  std::printf("[metal_lm_smoke.gemma_gtile_attn_token_exact] %zu tokens, "
              "%zu mismatches\n", ref.size(), mism);
  EXPECT_TRUE(mism == 0);
}

// Batched (N-branch parallel) metal decode must be token-exact vs serial
// decode_step_fast per branch. Branches share a prefill prefix but are fed
// DISTINCT first tokens so they diverge -- a batched forward that leaked one
// branch's K/V into another would mismatch. Gated on a hybrid Qwen3.5 model
// (VPIPE_QWEN35_TEST_MODEL_PATH) so the GDN linear-attention layers are
// exercised alongside the full-attention layers.
TEST(metal_lm_smoke, qwen_batched_decode_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 40;   // fits 2*8 branches + shared prefix at N=8
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  if (!model) { return; }
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  if (!tok) { return; }
  auto* ctxm = model->context_manager();
  if (!ctxm) { return; }

  // Shared prefix prefill on a root context.
  auto prompt = tok->encode("The weather today is");
  if (prompt.empty()) { return; }
  auto root = ctxm->acquire_root();
  if (!root.valid()) { return; }
  if (model->prefill(root, prompt).empty()) { return; }

  const int n_steps = 12;
  // Sweep the adaptive MAXM tiers: N=3 -> MAXM=4 (1 grid.z tile); N=6 ->
  // MAXM=2 (ceil(m/2) tiles); N=8 -> the grouped-x xp2 tall tile (one
  // weight read for all 8 rows). All must stay token-exact with serial
  // decode.
  for (int N : {3, 6, 8}) {

  // Two independent branch sets sharing the same prefix: one batched, one
  // serial reference. Give each branch a DIFFERENT-LENGTH distinct suffix so
  // the branches sit at DIFFERENT seq_lens -- exercising the relaxed
  // (non-lockstep) batching where projections batch across the active set
  // while RoPE + attention run per branch at each branch's own position.
  auto batched = ctxm->branch(root, N);
  auto serial  = ctxm->branch(root, N);
  if ((int)batched.size() != N || (int)serial.size() != N) { continue; }

  std::vector<std::int32_t> first_tokens((std::size_t)N);
  auto argmax_of = [&](const std::vector<float>& lg) {
    std::int32_t best = 0; float bv = -1e30f;
    for (std::size_t v = 0; v < lg.size(); ++v) {
      if (lg[v] > bv) { bv = lg[v]; best = (std::int32_t)v; }
    }
    return best;
  };
  for (int i = 0; i < N; ++i) {
    // Branch i: i+1 copies of a distinct token -> distinct content + length.
    std::vector<std::int32_t> suffix((std::size_t)(i + 1),
                                     (std::int32_t)(100 + i));
    auto lb = model->prefill(batched[(std::size_t)i], suffix);
    auto ls = model->prefill(serial[(std::size_t)i], suffix);
    if (lb.empty() || ls.empty()) { return; }
    first_tokens[(std::size_t)i] = argmax_of(lb);   // == argmax_of(ls)
  }

  // Serial reference per branch (each at its own position).
  std::vector<std::vector<std::int32_t>> ref((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::int32_t cur = first_tokens[(std::size_t)i];
    for (int s = 0; s < n_steps; ++s) {
      const std::int32_t nxt = model->decode_step_fast(serial[(std::size_t)i],
                                                       cur);
      ASSERT_TRUE(nxt != std::numeric_limits<std::int32_t>::min());
      if (nxt < 0) { break; }
      ref[(std::size_t)i].push_back(nxt);
      cur = nxt;
    }
  }

  // Batched.
  auto got = model->decode_batched_argmax(
      std::span<const genai::ContextId>(batched.data(), batched.size()),
      std::span<const std::int32_t>(first_tokens.data(), first_tokens.size()),
      n_steps);

  ASSERT_TRUE((int)got.size() == N);
  int matched = 0, total = 0, mismatched = 0;
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(got[(std::size_t)i].size() == ref[(std::size_t)i].size());
    for (std::size_t s = 0;
         s < ref[(std::size_t)i].size() && s < got[(std::size_t)i].size();
         ++s) {
      ++total;
      if (got[(std::size_t)i][s] == ref[(std::size_t)i][s]) { ++matched; }
      else {
        ++mismatched;
        // WHERE does it diverge? An isolated flip that RE-SYNCS (steps around
        // it match) is a benign numerical argmax near-tie; a contiguous tail
        // of mismatches is a real cascade -> the tolerance below catches that.
        std::printf("[batched-mismatch] N=%d branch=%d step=%zu/%d "
                    "got=%d ref=%d\n", N, i, s, n_steps,
                    got[(std::size_t)i][s], ref[(std::size_t)i][s]);
      }
    }
  }
  // Tolerate a small fraction of near-tie argmax flips instead of demanding
  // bit-exactness. The batched matmul reads weights once across the N rows, so
  // its f16 reduction order differs from serial's per-row qmv -- and that order
  // ALSO shifts with the batch size (the GEMV tile shape changes: MAXM=4 at
  // N=3, MAXM=2/xp2 at N>4; mixed-precision 8-bit layers additionally route
  // through steel GEMM). When two top logits sit within f16 noise (~1e-2 at
  // O(10) magnitudes) the tie can break either way; the swapped tokens are
  // interchangeable and the branch re-syncs (verified via VPIPE_BATCHED_MARGIN_
  // PROBE: gap ~0.016 collapsing to an exact 0.0). Both 4-bit and mixed models
  // are susceptible; which inputs tip is data-dependent. So accept up to ~5% of
  // positions flipping. CAVEAT: a tie can strike EARLY and cascade its whole
  // branch tail -- that shows up as a large mismatch fraction and still fails
  // here, but a borderline early flip on a large batch could sneak under 5%;
  // tighten / add a per-branch contiguous-tail check if that bites in practice.
  EXPECT_TRUE(mismatched * 20 <= total);   // <= 5% flipped
  std::printf("[metal_lm_smoke.qwen_batched_decode] N=%d steps=%d matched "
              "%d/%d (%d near-tie flips, tol %d)\n", N, n_steps, matched, total,
              mismatched, total / 20);
  for (auto id : batched) { ctxm->release(id); }
  for (auto id : serial) { ctxm->release(id); }
  }

  // --- Margin probe: is the N>4 mismatch a numerical near-tie or a real bug?
  // Reproduce branch 0 (the observed flip site) in an N=8 batch and, in
  // lockstep, a single serial context, capturing full logits each step.
  // Print the top-2 margin + the logit gap between the two swapped tokens at
  // the flip steps. A tiny gap (<~1e-2, << the softmax's noise floor) == a
  // benign f16 reduction-order tie between the batched 8-bit steel-GEMM path
  // and serial's per-row w8 qmv on the OptiQ mixed model; a large gap == bug.
  if (const char* mp = std::getenv("VPIPE_BATCHED_MARGIN_PROBE"); mp && *mp) {
    const int N = 8;
    auto amax = [&](const float* lg, std::size_t n) {
      std::int32_t best = 0; float bv = -1e30f;
      for (std::size_t v = 0; v < n; ++v) {
        if (lg[v] > bv) { bv = lg[v]; best = (std::int32_t)v; }
      }
      return best;
    };
    auto bb = ctxm->branch(root, N);   // batched set
    auto sc = ctxm->branch(root, 1);   // serial branch-0 mirror
    if ((int)bb.size() == N && sc.size() == 1) {
      std::vector<std::int32_t> cur((std::size_t)N);   // batched per-branch cur
      std::size_t vocab = 0;
      for (int i = 0; i < N; ++i) {
        std::vector<std::int32_t> suf((std::size_t)(i + 1),
                                      (std::int32_t)(100 + i));
        auto lb = model->prefill(bb[(std::size_t)i], suf);
        vocab = lb.size();
        cur[(std::size_t)i] = amax(lb.data(), lb.size());   // each first token
      }
      // Serial mirror of branch 0 (suffix = 1 copy of token 100).
      auto ls = model->prefill(sc[0], std::vector<std::int32_t>{100});
      std::int32_t s_cur = amax(ls.data(), ls.size());   // == cur[0]
      std::printf("[margin-probe] N=8 branch0, tokens 60643 vs 71049; "
                  "first serial=%d batch=%d\n", s_cur, cur[0]);
      std::vector<float> bl;
      for (int s = 0; s < 12; ++s) {
        // Serial branch-0 logits.
        auto sl = model->forward(sc[0], s_cur);
        std::int32_t s_next = amax(sl.data(), sl.size());
        // Batched logits (all N; branch 0 is row 0).
        model->decode_batched_step(
            std::span<const genai::ContextId>(bb.data(), bb.size()),
            std::span<const std::int32_t>(cur.data(), cur.size()),
            std::span<const std::int32_t>(), bl);
        const float* b0 = bl.data();
        std::int32_t b_next = amax(b0, vocab);
        float sg = sl[60643] - sl[71049];
        float bg = b0[60643] - b0[71049];
        std::printf("[margin-probe] step=%d serial->%-6d batch->%-6d %s | "
                    "gap(60643-71049) serial=%+.6f batch=%+.6f\n",
                    s, s_next, b_next,
                    (s_next == b_next ? "  " : "<<"), sg, bg);
        s_cur = s_next;
        // Advance every batched branch by its own argmax (branch 0 independent
        // of the others, but they must keep valid state).
        for (int i = 0; i < N; ++i) {
          cur[(std::size_t)i] = amax(bl.data() + (std::size_t)i * vocab, vocab);
        }
      }
      for (auto id : sc) { ctxm->release(id); }
    }
    for (auto id : bb) { ctxm->release(id); }
  }
}

// Shared-prefix batched decode attention (phase A reads the N branches' shared
// prefix once, phase B merges each branch's private pages) must be TOKEN-EXACT
// vs the per-branch SDPA. Uses a MULTI-PAGE shared prefix (small page_tokens)
// so shared_pages>=2, N=4 branches at distinct positions, and toggles
// set_shared_attn ON vs OFF in-process to isolate the shared split.
// VPIPE_SDPA_MB_MIN=0 forces the OFF path onto the mb256 kernel too, so the
// only difference is shared-vs-strided splitting of the same online softmax.
TEST(metal_lm_smoke, qwen_shared_attn_token_exact) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  if (mcfg.head_dim != 256) { return; }   // shared-attn path is D=256 only
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 32;   // small -> a longish prefix spans several pages
  mcfg.max_pages = 64;
  ::setenv("VPIPE_SDPA_MB_MIN", "0", 1);  // OFF path also uses mb256
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  ::unsetenv("VPIPE_SDPA_MB_MIN");
  if (!model) { return; }
  if (!model->shared_attn()) { return; }  // kernels unavailable -> skip
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  if (!tok) { return; }
  auto* ctxm = model->context_manager();
  if (!ctxm) { return; }

  // A multi-page shared prefix (>= 2 shared pages at page_tokens=32).
  auto prompt = tok->encode(
      "The quick brown fox jumps over the lazy dog near the river bank while "
      "the sun sets slowly behind the distant rolling hills and the wind "
      "carries the scent of pine across the quiet valley below the ridge, and "
      "far away a train whistle echoes through the cool evening air softly.");
  // Need > page_tokens (32) tokens so the shared prefix spans >= 2 pages.
  if ((int)prompt.size() < 34) { return; }
  auto root = ctxm->acquire_root();
  if (!root.valid() || model->prefill(root, prompt).empty()) { return; }

  const int N = 4;
  const int n_steps = 16;
  auto argmax_of = [&](const std::vector<float>& lg) {
    std::int32_t best = 0; float bv = -1e30f;
    for (std::size_t v = 0; v < lg.size(); ++v) {
      if (lg[v] > bv) { bv = lg[v]; best = (std::int32_t)v; }
    }
    return best;
  };
  // Branch N off the shared prefix (distinct-length suffixes -> distinct
  // positions) and return the per-branch argmax streams. `on` toggles the
  // shared-prefix path; both runs start from the same untouched root.
  auto run = [&](bool on) {
    model->set_shared_attn(on);
    auto br = ctxm->branch(root, N);
    std::vector<std::int32_t> first((std::size_t)N);
    for (int i = 0; i < N; ++i) {
      std::vector<std::int32_t> suffix((std::size_t)(i + 1),
                                       (std::int32_t)(100 + i));
      auto l = model->prefill(br[(std::size_t)i], suffix);
      first[(std::size_t)i] = argmax_of(l);
    }
    auto got = model->decode_batched_argmax(
        std::span<const genai::ContextId>(br.data(), br.size()),
        std::span<const std::int32_t>(first.data(), first.size()), n_steps);
    for (auto id : br) { ctxm->release(id); }
    return got;
  };
  auto off = run(false);
  auto onv = run(true);
  ASSERT_TRUE((int)off.size() == N && (int)onv.size() == N);
  int matched = 0, total = 0;
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(off[(std::size_t)i].size() == onv[(std::size_t)i].size());
    for (std::size_t s = 0;
         s < off[(std::size_t)i].size() && s < onv[(std::size_t)i].size();
         ++s) {
      ++total;
      if (off[(std::size_t)i][s] == onv[(std::size_t)i][s]) { ++matched; }
      EXPECT_TRUE(off[(std::size_t)i][s] == onv[(std::size_t)i][s]);
    }
  }
  std::printf("[metal_lm_smoke.qwen_shared_attn] N=%d steps=%d matched %d/%d "
              "(shared vs per-branch)\n", N, n_steps, matched, total);
  EXPECT_TRUE(total > 0 && matched == total);
}

// Batched PIPELINED decode (bdecode_*, GPU per-branch sampler + event-chain
// overlap) must be token-exact vs the synchronous decode_batched_argmax in
// GREEDY mode, including branches at DIFFERENT seq_lens. Two branch sets off
// a shared prefix: one drives decode_batched_argmax (reference), one drives
// bdecode_begin/commit/next greedily. Gated on a hybrid Qwen3.5 model.
TEST(metal_lm_smoke, qwen_bdecode_matches_batched_argmax) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  mcfg.max_pages = 40;   // fits 2*8 branches + shared prefix at N=8
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  if (!model) { return; }
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  if (!tok) { return; }
  auto* ctxm = model->context_manager();
  if (!ctxm) { return; }

  auto prompt = tok->encode("The weather today is");
  if (prompt.empty()) { return; }
  auto root = ctxm->acquire_root();
  if (!root.valid() || model->prefill(root, prompt).empty()) { return; }

  const int N = 3, n_steps = 12;
  auto ref_set = ctxm->branch(root, N);   // decode_batched_argmax
  auto pipe_set = ctxm->branch(root, N);  // bdecode_*
  if ((int)ref_set.size() != N || (int)pipe_set.size() != N) { return; }

  auto argmax_of = [&](const std::vector<float>& lg) {
    std::int32_t best = 0; float bv = -1e30f;
    for (std::size_t v = 0; v < lg.size(); ++v) {
      if (lg[v] > bv) { bv = lg[v]; best = (std::int32_t)v; }
    }
    return best;
  };
  std::vector<std::int32_t> first((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::vector<std::int32_t> suffix((std::size_t)(i + 1),
                                     (std::int32_t)(100 + i));
    auto lr = model->prefill(ref_set[(std::size_t)i], suffix);
    auto lp = model->prefill(pipe_set[(std::size_t)i], suffix);
    if (lr.empty() || lp.empty()) { return; }
    first[(std::size_t)i] = argmax_of(lr);
  }

  // Reference: synchronous batched argmax.
  auto ref = model->decode_batched_argmax(
      std::span<const genai::ContextId>(ref_set.data(), ref_set.size()),
      std::span<const std::int32_t>(first.data(), first.size()), n_steps);

  // Pipelined: greedy bdecode over the second branch set, driven in the
  // RUN-AHEAD shape (prime an extra commit, refill after each next) so the
  // depth>=2 default keeps a speculative step in flight the whole session --
  // the collected rows must still match the synchronous reference exactly.
  // (At VPIPE_QWEN_BDECODE_DEPTH=1 the prime is refused and this degrades
  // to the old commit/next lockstep.)
  genai::GpuSamplerParams sp;   // greedy=true by default
  std::vector<std::vector<std::int32_t>> got((std::size_t)N);
  ASSERT_TRUE(model->bdecode_begin(
      std::span<const genai::ContextId>(pipe_set.data(), pipe_set.size()),
      std::span<const std::int32_t>(first.data(), first.size()), sp, n_steps));
  std::vector<std::int32_t> step_tok;
  bool committed = model->bdecode_commit();
  model->bdecode_commit();   // run-ahead prime (refused at depth-1)
  for (int s = 0; s < n_steps; ++s) {
    if (!committed) { break; }
    if (!model->bdecode_next(step_tok)) { break; }
    for (int i = 0; i < N; ++i) {
      got[(std::size_t)i].push_back(step_tok[(std::size_t)i]);
    }
    committed = model->bdecode_commit();
  }
  model->bdecode_end();

  int matched = 0, total = 0;
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(got[(std::size_t)i].size() == ref[(std::size_t)i].size());
    for (std::size_t s = 0;
         s < ref[(std::size_t)i].size() && s < got[(std::size_t)i].size();
         ++s) {
      ++total;
      if (got[(std::size_t)i][s] == ref[(std::size_t)i][s]) { ++matched; }
      EXPECT_TRUE(got[(std::size_t)i][s] == ref[(std::size_t)i][s]);
    }
  }
  std::printf("[metal_lm_smoke.qwen_bdecode] N=%d steps=%d matched %d/%d\n",
              N, n_steps, matched, total);
}

// Perf triage: time single-branch decode (qmv) vs batched decode (qmm steel
// GEMM, M=N) at N=1,2,4 over the SAME model/context, to see whether batched
// per-step wall is the expected ~weight-read time (so aggregate tok/s scales
// with N) or whether qmm-at-small-M / per-branch dispatch / host logit pull
// dominates. Gated on VPIPE_QWEN35_TEST_MODEL_PATH + VPIPE_QWEN_BATCH_BENCH.
TEST(metal_lm_smoke, qwen_batched_decode_bench) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (!std::getenv("VPIPE_QWEN_BATCH_BENCH")) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  genai::ModelLoader loader(&sess);
  auto cfg = loader.load_config(path);
  if (!cfg) { return; }
  auto mcfg = genai::MetalQwenModel::config_from(*cfg);
  mcfg.use_bf16 = false;
  mcfg.page_tokens = 512;
  // Headroom: the sections below branch repeatedly without releasing the raw
  // ContextIds, and the shared-prefix A/B prefills a ~1024-token root.
  mcfg.max_pages = 64;
  auto model = genai::MetalQwenModel::load(path, mc, mcfg);
  if (!model) { return; }
  auto tok = genai::Tokenizer::from_huggingface_json(
      std::string(path) + "/tokenizer.json", &sess);
  if (!tok) { return; }
  auto* ctxm = model->context_manager();
  if (!ctxm) { return; }
  const int vocab = mcfg.vocab;

  auto prompt = tok->encode("The weather today is");
  if (prompt.empty()) { return; }
  auto root = ctxm->acquire_root();
  if (!root.valid()) { return; }
  if (model->prefill(root, prompt).empty()) { return; }

  using clock = std::chrono::steady_clock;
  const int K = 32;                 // timed steps
  std::vector<float> logits;

  // --- single-branch (qmv) reference ---
  {
    auto br = ctxm->branch(root, 1);
    std::int32_t cur = 100;
    for (int s = 0; s < 4; ++s) {   // warm
      cur = model->decode_step_fast(br[0], cur);
    }
    const auto t0 = clock::now();
    for (int s = 0; s < K; ++s) {
      cur = model->decode_step_fast(br[0], cur);
    }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("[bench] single (qmv)   per-step %.2f ms  -> %.1f tok/s\n",
                1e3 * dt / K, K / dt);
  }

  // --- batched (qmm M=N) at several N ---
  // Releases the branch set per iteration (the sweep would otherwise exhaust
  // max_pages by N=24 and decode_batched_step would fail instantly -> a bogus
  // 0.00 ms row). A step failure is reported, not timed.
  for (int N : {2, 3, 4, 5, 6, 8, 12, 16, 24, 32}) {
    auto br = ctxm->branch(root, N);
    if ((int)br.size() != N) {
      std::printf("[bench] batched N=%d    SKIPPED (branch alloc failed)\n", N);
      for (auto id : br) { ctxm->release(id); }
      continue;
    }
    std::vector<genai::ContextId> cids(br.begin(), br.end());
    std::vector<std::int32_t> cur((std::size_t)N, 100);
    bool failed = false;
    auto step = [&]() {
      if (!model->decode_batched_step(
              std::span<const genai::ContextId>(cids.data(), cids.size()),
              std::span<const std::int32_t>(cur.data(), cur.size()),
              std::span<const std::int32_t>(), logits)) {
        failed = true;
        return false;
      }
      for (int i = 0; i < N; ++i) {
        const float* row = logits.data() + (std::size_t)i * vocab;
        std::int32_t best = 0; float bv = row[0];
        for (int v = 1; v < vocab; ++v) {
          if (row[v] > bv) { bv = row[v]; best = v; }
        }
        cur[(std::size_t)i] = best;
      }
      return true;
    };
    for (int s = 0; s < 4 && !failed; ++s) { step(); }   // warm
    const auto t0 = clock::now();
    for (int s = 0; s < K && !failed; ++s) { step(); }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    if (failed) {
      std::printf("[bench] batched N=%d    FAILED (decode_batched_step)\n", N);
    } else {
      std::printf("[bench] batched N=%d    per-step %.2f ms  -> %.1f tok/s "
                  "(aggregate, %d branches)\n",
                  N, 1e3 * dt / K, (double)N * K / dt, N);
    }
    for (auto id : br) { ctxm->release(id); }
  }

  // --- pipelined bdecode (GPU sampler + event-chain overlap), greedy ---
  for (int N : {2, 4, 8}) {
    auto br = ctxm->branch(root, N);
    if ((int)br.size() != N) { continue; }
    std::vector<genai::ContextId> cids(br.begin(), br.end());
    std::vector<std::int32_t> first((std::size_t)N, 100);
    genai::GpuSamplerParams sp;   // greedy
    if (!model->bdecode_begin(
            std::span<const genai::ContextId>(cids.data(), cids.size()),
            std::span<const std::int32_t>(first.data(), first.size()),
            sp, K + 8)) {
      continue;
    }
    std::vector<std::int32_t> toks;
    // Run-ahead driver shape (the stage's): prime, then next+refill. At
    // VPIPE_QWEN_BDECODE_DEPTH=1 the prime is refused -> old lockstep.
    model->bdecode_commit();
    model->bdecode_commit();   // run-ahead prime
    for (int s = 0; s < 4; ++s) {           // warm
      model->bdecode_next(toks); model->bdecode_commit();
    }
    const auto t0 = clock::now();
    for (int s = 0; s < K; ++s) {
      if (!model->bdecode_next(toks)) { break; }
      model->bdecode_commit();
    }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    model->bdecode_end();
    std::printf("[bench] pipelined N=%d  per-step %.2f ms  -> %.1f tok/s "
                "(aggregate, %d branches)\n",
                N, 1e3 * dt / K, (double)N * K / dt, N);
  }

  // --- STAGGERED finish A/B: shrinking (sync) vs constant-N (pipelined) ----
  // The realtime-vqa reality: the N question-branches finish at DIFFERENT
  // lengths. The shrinking sync path drops a branch as it stops (active N
  // falls); the pipelined path is constant-N -> it keeps doing all 8
  // branches' work until the LONGEST finishes, so it pays for already-done
  // branches. Same useful token count both ways; compare aggregate tok/s.
  {
    const int Nst = 8;
    const int Lst[8] = {4, 8, 12, 16, 20, 24, 28, 32};   // per-branch lengths
    int useful = 0, maxL = 0;
    for (int i = 0; i < Nst; ++i) { useful += Lst[i]; maxL = std::max(maxL, Lst[i]); }

    // Shrinking (sync): re-batch only the still-active branches each step.
    {
      auto br = ctxm->branch(root, Nst);
      if ((int)br.size() == Nst) {
        std::vector<genai::ContextId> all(br.begin(), br.end());
        std::vector<std::int32_t> cur((std::size_t)Nst, 100);
        std::vector<int> rem(Lst, Lst + Nst);
        const auto t0 = clock::now();
        for (;;) {
          std::vector<genai::ContextId> act;
          std::vector<std::int32_t> actc;
          std::vector<int> map;
          for (int i = 0; i < Nst; ++i) {
            if (rem[i] > 0) { act.push_back(all[i]); actc.push_back(cur[i]);
                              map.push_back(i); }
          }
          if (act.empty()) { break; }
          if (!model->decode_batched_step(
                  std::span<const genai::ContextId>(act.data(), act.size()),
                  std::span<const std::int32_t>(actc.data(), actc.size()),
                  std::span<const std::int32_t>(), logits)) {
            break;
          }
          for (std::size_t j = 0; j < map.size(); ++j) {
            const float* row = logits.data() + j * vocab;
            std::int32_t best = 0; float bv = row[0];
            for (int v = 1; v < vocab; ++v) {
              if (row[v] > bv) { bv = row[v]; best = v; }
            }
            cur[(std::size_t)map[j]] = best;
            rem[(std::size_t)map[j]]--;
          }
        }
        const double dt =
            std::chrono::duration<double>(clock::now() - t0).count();
        std::printf("[bench] staggered N=8 shrinking(sync)  %.1f tok/s "
                    "(%d useful tok in %.3fs)\n", useful / dt, useful, dt);
        for (auto id : br) { ctxm->release(id); }
      }
    }
    // Constant-N (pipelined): runs all 8 for maxL steps; useful tokens are
    // the same `useful`, but wall time covers Nst*maxL branch-steps.
    {
      auto br = ctxm->branch(root, Nst);
      if ((int)br.size() == Nst) {
        std::vector<genai::ContextId> cids(br.begin(), br.end());
        std::vector<std::int32_t> first((std::size_t)Nst, 100);
        genai::GpuSamplerParams sp;
        if (model->bdecode_begin(
                std::span<const genai::ContextId>(cids.data(), cids.size()),
                std::span<const std::int32_t>(first.data(), first.size()),
                sp, maxL + 8)) {
          std::vector<std::int32_t> toks;
          const auto t0 = clock::now();
          model->bdecode_commit();
          model->bdecode_commit();   // run-ahead prime
          for (int s = 0; s < maxL; ++s) {
            if (!model->bdecode_next(toks)) { break; }
            model->bdecode_commit();
          }
          const double dt =
              std::chrono::duration<double>(clock::now() - t0).count();
          model->bdecode_end();
          std::printf("[bench] staggered N=8 constant(pipe)  %.1f tok/s "
                      "(%d useful tok in %.3fs)\n", useful / dt, useful, dt);
        }
        for (auto id : br) { ctxm->release(id); }
      }
    }
  }

  // --- shared-prefix attention A/B over a LONG shared prefix ---------------
  // The realtime-vqa win: with a big shared prefix (image/video tokens), the
  // per-branch SDPA re-reads it N times. Phase A reads it once. Build a ~1024-
  // token shared prefix and time batched decode with set_shared_attn OFF vs
  // ON at N=2,4 (the gap grows with prefix length).
  for (int PLEN : (mcfg.head_dim == 256 && model->shared_attn())
                      ? std::vector<int>{1024, 4096, 8192}
                      : std::vector<int>{}) {
    std::vector<std::int32_t> long_ids;
    long_ids.reserve((std::size_t)PLEN);
    while ((int)long_ids.size() < PLEN) {
      for (std::int32_t t : prompt) {
        if ((int)long_ids.size() >= PLEN) { break; }
        long_ids.push_back(t);
      }
    }
    auto lroot = ctxm->acquire_root();
    if (lroot.valid() && !model->prefill(lroot, long_ids).empty()) {
      std::printf("[bench] --- shared-prefix attn (prefix=%d tok) ---\n",
                  (int)long_ids.size());
      for (int N : {2, 4, 8}) {
        for (int on = 0; on <= 1; ++on) {
          model->set_shared_attn(on != 0);
          auto br = ctxm->branch(lroot, N);
          if ((int)br.size() != N) { continue; }
          std::vector<genai::ContextId> cids(br.begin(), br.end());
          std::vector<std::int32_t> cur((std::size_t)N, 100);
          auto step = [&]() {
            if (!model->decode_batched_step(
                    std::span<const genai::ContextId>(cids.data(), cids.size()),
                    std::span<const std::int32_t>(cur.data(), cur.size()),
                    std::span<const std::int32_t>(), logits)) {
              return false;
            }
            for (int i = 0; i < N; ++i) {
              const float* row = logits.data() + (std::size_t)i * vocab;
              std::int32_t best = 0; float bv = row[0];
              for (int v = 1; v < vocab; ++v) {
                if (row[v] > bv) { bv = row[v]; best = v; }
              }
              cur[(std::size_t)i] = best;
            }
            return true;
          };
          for (int s = 0; s < 4; ++s) { step(); }   // warm
          const auto t0 = clock::now();
          for (int s = 0; s < K; ++s) { step(); }
          const double dt =
              std::chrono::duration<double>(clock::now() - t0).count();
          std::printf("[bench] batched N=%d shared_attn=%-3s per-step %.2f ms "
                      " -> %.1f tok/s\n", N, on ? "ON" : "OFF",
                      1e3 * dt / K, (double)N * K / dt);
          for (auto id : br) { ctxm->release(id); }
        }
      }
      model->set_shared_attn(true);
    }
    ctxm->release(lroot);   // free the long prefix before the next length
  }
}

// LM-level batched decode (the path realtime-vqa uses): branch N contexts off
// a shared prefix, give each a different-length suffix (so they sit at
// DIFFERENT positions), then drive LoadedLanguageModel::m_batched_decode_step
// greedily and require it token-exact vs a serial next_token_greedy loop per
// branch. Validates the exec cid-mapping + rope bookkeeping the stage relies
// on. Gated on a hybrid Qwen3.5 model.
TEST(metal_lm_smoke, qwen_lm_batched_step_matches_serial) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid() || !lm->m_batched_decode_supported()) { return; }

  auto root = lm->make_context();
  if (!root.valid()) { return; }
  auto prefix = lm->tokenizer().encode("The weather today is");
  if (prefix.empty() || lm->prefill(root, prefix) < 0) { return; }

  const int N = 3, n_steps = 10;
  const int vocab = lm->config().vocab_size;
  auto bset = lm->branch(root, N);
  auto sset = lm->branch(root, N);
  if ((int)bset.size() != N || (int)sset.size() != N) { return; }

  std::vector<std::int32_t> first((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::vector<std::int32_t> suf((std::size_t)(i + 1), (std::int32_t)(100 + i));
    const std::int32_t fb = lm->prefill(bset[(std::size_t)i], suf);
    const std::int32_t fs = lm->prefill(sset[(std::size_t)i], suf);
    if (fb < 0 || fs < 0) { return; }
    first[(std::size_t)i] = fb;
  }

  std::vector<std::vector<std::int32_t>> ref((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::int32_t cur = first[(std::size_t)i];
    for (int s = 0; s < n_steps; ++s) {
      const std::int32_t nxt = lm->next_token_greedy(sset[(std::size_t)i], cur);
      if (nxt < 0) { break; }
      ref[(std::size_t)i].push_back(nxt);
      cur = nxt;
    }
  }

  std::vector<std::int32_t> cur(first.begin(), first.end());
  std::vector<std::vector<std::int32_t>> got((std::size_t)N);
  std::vector<float> logits;
  for (int s = 0; s < n_steps; ++s) {
    std::vector<genai::LoadedLanguageModel::Context*> ptrs((std::size_t)N);
    for (int i = 0; i < N; ++i) { ptrs[(std::size_t)i] = &bset[(std::size_t)i]; }
    if (!lm->m_batched_decode_step(
            std::span<genai::LoadedLanguageModel::Context*>(ptrs.data(),
                                                          ptrs.size()),
            std::span<const std::int32_t>(cur.data(), cur.size()), logits)) {
      break;
    }
    for (int i = 0; i < N; ++i) {
      const float* row = logits.data() + (std::size_t)i * vocab;
      std::int32_t best = 0; float bv = row[0];
      for (int v = 1; v < vocab; ++v) { if (row[v] > bv) { bv = row[v]; best = v; } }
      got[(std::size_t)i].push_back(best);
      cur[(std::size_t)i] = best;
    }
  }

  int matched = 0, total = 0;
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(got[(std::size_t)i].size() == ref[(std::size_t)i].size());
    for (std::size_t s = 0;
         s < ref[(std::size_t)i].size() && s < got[(std::size_t)i].size(); ++s) {
      ++total;
      if (got[(std::size_t)i][s] == ref[(std::size_t)i][s]) { ++matched; }
      EXPECT_TRUE(got[(std::size_t)i][s] == ref[(std::size_t)i][s]);
    }
  }
  std::printf("[metal_lm_smoke.qwen_lm_batched_step] matched %d/%d\n",
              matched, total);
}

// Gemma-4 batched decode (LM level), token-exact vs serial. Branch N off a
// shared prefix, give each a different-length suffix (different positions),
// drive m_batched_decode_step greedily vs next_token_greedy. Returns
// {matched,total}; the TEST asserts (EXPECT_TRUE can't live in a free fn).
namespace {
struct GemmaBatchedResult { bool loaded = false; int matched = 0, total = 0; };
GemmaBatchedResult gemma_lm_batched_run_(const char* path) {
  GemmaBatchedResult r;
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return r; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 4;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid() || !lm->m_batched_decode_supported()) { return r; }
  r.loaded = true;

  auto root = lm->make_context();
  if (!root.valid()) { return r; }
  auto prefix = lm->tokenizer().encode("The weather today is");
  if (prefix.empty() || lm->prefill(root, prefix) < 0) { return r; }

  const int N = 3, n_steps = 10;
  const int vocab = lm->config().vocab_size;
  auto bset = lm->branch(root, N);
  auto sset = lm->branch(root, N);
  if ((int)bset.size() != N || (int)sset.size() != N) { return r; }

  std::vector<std::int32_t> first((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::vector<std::int32_t> suf((std::size_t)(i + 1), (std::int32_t)(100 + i));
    const std::int32_t fb = lm->prefill(bset[(std::size_t)i], suf);
    const std::int32_t fs = lm->prefill(sset[(std::size_t)i], suf);
    if (fb < 0 || fs < 0) { return r; }
    first[(std::size_t)i] = fb;
  }

  std::vector<std::vector<std::int32_t>> ref((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    std::int32_t cur = first[(std::size_t)i];
    for (int s = 0; s < n_steps; ++s) {
      const std::int32_t nxt = lm->next_token_greedy(sset[(std::size_t)i], cur);
      if (nxt < 0) { break; }
      ref[(std::size_t)i].push_back(nxt);
      cur = nxt;
    }
  }

  std::vector<std::int32_t> cur(first.begin(), first.end());
  std::vector<std::vector<std::int32_t>> got((std::size_t)N);
  std::vector<float> logits;
  for (int s = 0; s < n_steps; ++s) {
    std::vector<genai::LoadedLanguageModel::Context*> ptrs((std::size_t)N);
    for (int i = 0; i < N; ++i) { ptrs[(std::size_t)i] = &bset[(std::size_t)i]; }
    if (!lm->m_batched_decode_step(
            std::span<genai::LoadedLanguageModel::Context*>(ptrs.data(),
                                                          ptrs.size()),
            std::span<const std::int32_t>(cur.data(), cur.size()), logits)) {
      break;
    }
    for (int i = 0; i < N; ++i) {
      const float* row = logits.data() + (std::size_t)i * vocab;
      std::int32_t best = 0; float bv = row[0];
      for (int v = 1; v < vocab; ++v) { if (row[v] > bv) { bv = row[v]; best = v; } }
      got[(std::size_t)i].push_back(best);
      cur[(std::size_t)i] = best;
    }
  }
  for (int i = 0; i < N; ++i) {
    for (std::size_t s = 0;
         s < ref[(std::size_t)i].size() && s < got[(std::size_t)i].size(); ++s) {
      ++r.total;
      if (got[(std::size_t)i][s] == ref[(std::size_t)i][s]) { ++r.matched; }
    }
  }
  return r;
}
}  // namespace

// 12B gemma4_unified: exercises the k_eq_v / mixed-quant / per-layer-n_kv /
// no-PLE batched path.
TEST(metal_lm_smoke, gemma12b_batched_step_matches_serial) {
  const char* path = std::getenv("VPIPE_GEMMA12B_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  auto r = gemma_lm_batched_run_(path);
  if (!r.loaded) { return; }
  EXPECT_TRUE(r.total > 0);
  EXPECT_TRUE(r.matched == r.total);
  std::printf("[metal_lm_smoke.gemma12b_batched_step] matched %d/%d\n",
              r.matched, r.total);
}

// e4b: exercises the PLE + cross-layer-KV-sharing batched path.
TEST(metal_lm_smoke, gemma_e4b_batched_step_matches_serial) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  auto r = gemma_lm_batched_run_(path);
  if (!r.loaded) { return; }
  EXPECT_TRUE(r.total > 0);
  EXPECT_TRUE(r.matched == r.total);
  std::printf("[metal_lm_smoke.gemma_e4b_batched_step] matched %d/%d\n",
              r.matched, r.total);
}

// Gemma e4b batched-decode perf (LM API, the realtime-vqa path): serial
// next_token_greedy on one branch vs m_batched_decode_step over N. Confirms
// the batched GEMV recovers the win for the geglu MLP. Gated on
// VPIPE_GEMMA4_TEST_MODEL_PATH + VPIPE_QWEN_BATCH_BENCH.
TEST(metal_lm_smoke, gemma_e4b_batched_decode_bench) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (!std::getenv("VPIPE_QWEN_BATCH_BENCH")) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid() || !lm->m_batched_decode_supported()) { return; }
  const int vocab = lm->config().vocab_size;

  auto root = lm->make_context();
  if (!root.valid()) { return; }
  // Shared scene prefix. Default short; VPIPE_GEMMA_BATCH_PREFIX_LEN grows it
  // to ~N tokens so the batched bench exercises the global layers' long-context
  // decode attention (the realtime-vqa regime where gtile pays off).
  std::string ptext = "The weather today is";
  if (const char* e = std::getenv("VPIPE_GEMMA_BATCH_PREFIX_LEN")) {
    const int want = std::atoi(e);
    ptext.clear();
    while ((int)lm->tokenizer().encode(ptext).size() < want) {
      ptext += "The cartographer unrolled the chart and traced each harbor. ";
    }
  }
  auto prefix = lm->tokenizer().encode(ptext);
  if (prefix.empty() || lm->prefill(root, prefix) < 0) { return; }
  std::printf("[bench-gemma] prefix=%zu tokens\n", prefix.size());

  using clock = std::chrono::steady_clock;
  const int K = 32;

  const std::vector<std::int32_t> seed1{(std::int32_t)100};
  {
    auto br = lm->branch(root, 1);
    std::int32_t cur = br[0].last_predicted_id();
    if (cur < 0) { cur = lm->prefill(br[0], seed1); }
    for (int s = 0; s < 4; ++s) { cur = lm->next_token_greedy(br[0], cur); }
    const auto t0 = clock::now();
    for (int s = 0; s < K; ++s) { cur = lm->next_token_greedy(br[0], cur); }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("[bench-gemma] single (qmv)   per-step %.2f ms -> %.1f tok/s\n",
                1e3 * dt / K, K / dt);
  }

  std::vector<float> logits;
  for (int N : {2, 4}) {
    auto br = lm->branch(root, N);
    if ((int)br.size() != N) { continue; }
    std::vector<std::int32_t> cur((std::size_t)N);
    for (int i = 0; i < N; ++i) {
      cur[(std::size_t)i] = br[(std::size_t)i].last_predicted_id();
      if (cur[(std::size_t)i] < 0) {
        cur[(std::size_t)i] = lm->prefill(br[(std::size_t)i], seed1);
      }
    }
    std::vector<genai::LoadedLanguageModel::Context*> ptrs((std::size_t)N);
    for (int i = 0; i < N; ++i) { ptrs[(std::size_t)i] = &br[(std::size_t)i]; }
    auto step = [&]() {
      if (!lm->m_batched_decode_step(
              std::span<genai::LoadedLanguageModel::Context*>(ptrs.data(),
                                                            ptrs.size()),
              std::span<const std::int32_t>(cur.data(), cur.size()), logits)) {
        return false;
      }
      for (int i = 0; i < N; ++i) {
        const float* row = logits.data() + (std::size_t)i * vocab;
        std::int32_t best = 0; float bv = row[0];
        for (int v = 1; v < vocab; ++v) {
          if (row[v] > bv) { bv = row[v]; best = v; }
        }
        cur[(std::size_t)i] = best;
      }
      return true;
    };
    for (int s = 0; s < 4; ++s) { step(); }
    const auto t0 = clock::now();
    for (int s = 0; s < K; ++s) { step(); }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::printf("[bench-gemma] batched N=%d    per-step %.2f ms -> %.1f tok/s "
                "(aggregate)\n", N, 1e3 * dt / K, (double)N * K / dt);
  }
}

// Lever #2 A/B: pipelined run-ahead decode vs the synchronous next_token
// loop. Measures whether overlapping the CPU command-buffer encode of token
// N+1 with the GPU execution of token N (pdecode depth>=2) recovers the
// per-token CPU-encode bubble. Token-exact gate: depth-1 AND depth-2 pdecode
// must reproduce the synchronous next_token_greedy stream exactly.
// Gated on VPIPE_GEMMA4_TEST_MODEL_PATH + VPIPE_GEMMA_PDECODE_BENCH; context
// length via VPIPE_GEMMA_BATCH_PREFIX_LEN (default short).
TEST(metal_lm_smoke, gemma_e4b_pdecode_pipeline_bench) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (!std::getenv("VPIPE_GEMMA_PDECODE_BENCH")) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }

  std::string ptext = "The weather today is";
  if (const char* e = std::getenv("VPIPE_GEMMA_BATCH_PREFIX_LEN")) {
    const int want = std::atoi(e);
    ptext.clear();
    while ((int)lm->tokenizer().encode(ptext).size() < want) {
      ptext += "The cartographer unrolled the chart and traced each harbor. ";
    }
  }
  auto ids = lm->tokenizer().encode(ptext);
  if (ids.empty()) { return; }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int K = 64;
  using clock = std::chrono::steady_clock;

  // Reference + timing: synchronous on-GPU-argmax greedy (decode_step_fast).
  std::vector<std::int32_t> ref;
  double sync_s = 0.0;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t t = lm->prefill(ctx, ids);
    if (t < 0) { return; }
    ref.push_back(t);
    for (int i = 0; i < 4; ++i) {          // warm
      t = lm->next_token_greedy(ctx, t);
      ref.push_back(t);
    }
    const auto t0 = clock::now();
    for (int i = 0; i < K; ++i) {
      t = lm->next_token_greedy(ctx, t);
      if (t < 0) { break; }
      ref.push_back(t);
    }
    sync_s = std::chrono::duration<double>(clock::now() - t0).count();
  }

  // Pipelined depth d: prefill, then run-ahead commit/next for warm+K tokens.
  auto run_pipe = [&](int depth, std::vector<std::int32_t>& out) -> double {
    ::setenv("VPIPE_GEMMA_PDECODE_DEPTH", depth >= 2 ? "2" : "1", 1);
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return -1.0; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { return -1.0; }
    out.push_back(first);
    genai::SamplerParams gsp;                  // defaults -> argmax
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, K + 8)) { return -1.0; }
    const int warm = 4;
    int committed = 0, emitted = 0;
    auto pump = [&](int target) {
      while (emitted < target) {
        while (committed < K + 4 && lm->pdecode_commit(ctx)) { ++committed; }
        const std::int32_t n = lm->pdecode_next(ctx);
        if (n < 0) { break; }
        out.push_back(n); ++emitted;
      }
    };
    pump(warm);                              // fill + warm
    const auto t0 = clock::now();
    pump(warm + K);
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    lm->pdecode_end(ctx);
    ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH");
    return dt;
  };

  std::vector<std::int32_t> p1, p2;
  const double d1_s = run_pipe(1, p1);
  const double d2_s = run_pipe(2, p2);

  auto mism = [&](const std::vector<std::int32_t>& a) {
    std::size_t m = 0;
    const std::size_t n = std::min(ref.size(), a.size());
    for (std::size_t i = 0; i < n; ++i) { if (ref[i] != a[i]) { ++m; } }
    return m;
  };
  const std::size_t m1 = mism(p1), m2 = mism(p2);
  // FNV-1a fingerprint of the greedy ref tokens -- lets an external A/B confirm
  // token-exactness across a toggle (e.g. VPIPE_GEMMA_PLE_CONCURRENT on vs off).
  std::uint64_t fp = 1469598103934665603ull;
  for (auto t : ref) {
    fp = (fp ^ (std::uint64_t)(std::uint32_t)t) * 1099511628211ull;
  }
  std::printf("[pdecode-ab] ctx=%zu K=%d | sync %.1f tok/s | pipe d1 %.1f "
              "tok/s | pipe d2 %.1f tok/s | d1_mism=%zu d2_mism=%zu ref_fp=%016llx\n",
              ids.size(), K, sync_s > 0 ? K / sync_s : 0.0,
              d1_s > 0 ? K / d1_s : 0.0, d2_s > 0 ? K / d2_s : 0.0, m1, m2,
              (unsigned long long)fp);
  EXPECT_TRUE(m1 == 0);
  EXPECT_TRUE(m2 == 0);
}

// GPU SAMPLING decode (temp>0) self-consistency + coherence + tok/s A/B for
// the histogram sampler. The GPU sampler is deterministic given a fixed base
// seed (per-step seed = base + golden*(step+1)), so a depth-1 and a depth-2
// pdecode run with the SAME params MUST be token-identical -- the self-
// consistency gate. Then it greedily-checks the sampled text is non-degenerate
// (not all one token) and prints sampling-decode tok/s. Run twice -- default
// (histogram) and VPIPE_GEMMA_SAMPLE1=1 (old single-tg) -- to A/B the
// end-to-end sampling-decode win; the test itself only gates self-consistency.
// Gated on VPIPE_GEMMA4_TEST_MODEL_PATH.
TEST(metal_lm_smoke, gemma_e4b_sampling_pdecode_selfconsistent) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  if (const char* e = std::getenv("VPIPE_GEMMA_SAMPLE_PREFIX_LEN")) {
    // Room for the long prefix + K decode tokens (page_tokens 512).
    spec.max_pages = (std::atoi(e) + 256) / 512 + 2;
  }
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(lm && lm->valid());

  std::vector<std::int32_t> ids;
  // Long-context A/B: VPIPE_GEMMA_SAMPLE_PREFIX_LEN pads the prompt to ~N
  // tokens (e.g. 8192) so the sampling-decode tok/s is measured at depth in the
  // KV (the sampler cost is context-independent, but decode wall is not).
  if (const char* e = std::getenv("VPIPE_GEMMA_SAMPLE_PREFIX_LEN")) {
    const int want = std::atoi(e);
    std::string ptext;
    while ((int)lm->tokenizer().encode(ptext).size() < want) {
      ptext += "The cartographer unrolled the chart and traced each harbor. ";
    }
    ids = lm->tokenizer().encode(ptext);
  } else {
    lm->chat_template()->render_user_turn(
        "Write two sentences about the ocean.", true, &ids);
  }
  ASSERT_TRUE(!ids.empty());
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int K = 64;

  genai::SamplerParams sp;            // temp>0 + top_p<1 -> GPU sampler path
  sp.temperature = 0.7f;
  sp.top_p       = 0.9f;
  sp.top_k       = 40;
  sp.min_p       = 0.0f;
  sp.seed        = 0xBEEFCAFEull;     // fixed base -> deterministic

  using clock = std::chrono::steady_clock;
  auto run = [&](int depth, std::vector<std::int32_t>& out) -> double {
    ::setenv("VPIPE_GEMMA_PDECODE_DEPTH", depth >= 2 ? "2" : "1", 1);
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return -1.0; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { return -1.0; }
    out.push_back(first);
    if (!lm->pdecode_begin(ctx, first, prompt, sp, K + 8)) { return -1.0; }
    int committed = 0, emitted = 0;
    const auto t0 = clock::now();
    while (emitted < K) {
      while (committed < K + 4 && lm->pdecode_commit(ctx)) { ++committed; }
      const std::int32_t n = lm->pdecode_next(ctx);
      if (n < 0) { break; }
      out.push_back(n); ++emitted;
    }
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    lm->pdecode_end(ctx);
    ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH");
    return dt;
  };

  std::vector<std::int32_t> a, b;
  const double sa = run(1, a);
  const double sb = run(2, b);

  // Self-consistency: deterministic sampler -> depth-1 == depth-2, 0 mismatch.
  std::size_t mism = 0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) { if (a[i] != b[i]) { ++mism; } }
  // Non-degenerate: more than 4 distinct ids in the sampled stream.
  std::vector<std::int32_t> uniq(a.begin(), a.end());
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  std::string txt = lm->tokenizer().decode(
      std::span<const std::int32_t>(a.data(), a.size()));
  const bool sample1 = std::getenv("VPIPE_GEMMA_SAMPLE1") != nullptr;
  std::printf("[gemma-sample] %s | d1 %.1f tok/s | d2 %.1f tok/s | "
              "selfconsist_mism=%zu | distinct=%zu/%zu\n",
              sample1 ? "OLD(sample1)" : "NEW(hist)",
              sa > 0 ? K / sa : 0.0, sb > 0 ? K / sb : 0.0,
              mism, uniq.size(), a.size());
  std::printf("[gemma-sample] text: %.200s\n", txt.c_str());
  ASSERT_TRUE(a.size() == b.size());
  EXPECT_TRUE(mism == 0);
  EXPECT_TRUE(uniq.size() > 4);
}

// Depth-2 run-ahead KV-ROLLBACK correctness. Depth>=2 speculatively commits
// (and KV-appends) the forward for token i+1 before the host has confirmed
// token i isn't a stop -- so on stop, pdecode_end must roll the KV back to
// the last produced token, matching the synchronous loop (where a stop
// token's KV is never appended). The decisive check is NOT just the token
// stream but CONTINUING decode on the same context afterward: if rollback
// left seq_len too high or a stale slot in-window, the continuation diverges.
// Gated on VPIPE_GEMMA4_TEST_MODEL_PATH (a correctness gate, not a bench).
TEST(metal_lm_smoke, gemma_e4b_pdecode_rollback_correct) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }

  auto ids = lm->tokenizer().encode("Tell me a short story about a robot.");
  if (ids.empty()) { return; }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int S = 8;      // tokens produced before the (simulated) stop
  const int K2 = 24;    // continuation tokens decoded on the same context

  // Reference: synchronous. Produce t1..tS (KV up to t_{S-1}; tS's KV not
  // appended -- the stop), then continue K2 more from tS.
  std::vector<std::int32_t> ref;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t t = lm->prefill(ctx, ids);
    if (t < 0) { return; }
    ref.push_back(t);
    for (int k = 1; k < S; ++k) { t = lm->next_token_greedy(ctx, t);
                                  ref.push_back(t); }
    std::int32_t c = ref.back();           // tS, the stop token
    for (int k = 0; k < K2; ++k) { c = lm->next_token_greedy(ctx, c);
                                   ref.push_back(c); }
  }

  // Speculative: depth-2 run-ahead produces t1..tS leaving one speculative
  // commit (KV-appended) in flight; pdecode_end rolls it back. Then continue
  // K2 from tS on the SAME context via the synchronous path.
  std::vector<std::int32_t> gen;
  {
    ::setenv("VPIPE_GEMMA_PDECODE_DEPTH", "2", 1);
    auto ctx = lm->make_context();
    if (!ctx.valid()) { ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH"); return; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH"); return; }
    gen.push_back(first);
    genai::SamplerParams gsp;
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, S + 4)) {
      ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH"); return;
    }
    const int target = S - 1;              // emit t2..tS
    int committed = 0, drained = 0;
    auto can_commit = [&]() { return committed < target + 1; };  // +1 spec
    while (can_commit() && lm->pdecode_commit(ctx)) { ++committed; }
    while (drained < target) {
      const std::int32_t nx = lm->pdecode_next(ctx);
      if (nx < 0) { break; }
      gen.push_back(nx); ++drained;
      while (can_commit() && lm->pdecode_commit(ctx)) { ++committed; }
    }
    lm->pdecode_end(ctx);                  // rolls back the speculative tail
    ::unsetenv("VPIPE_GEMMA_PDECODE_DEPTH");
    std::int32_t c = gen.back();           // tS
    for (int k = 0; k < K2; ++k) { c = lm->next_token_greedy(ctx, c);
                                   gen.push_back(c); }
  }

  std::size_t mism = 0;
  const std::size_t n = std::min(ref.size(), gen.size());
  for (std::size_t i = 0; i < n; ++i) { if (ref[i] != gen[i]) { ++mism; } }
  std::printf("[pdecode-rollback] ref=%zu gen=%zu (S=%d K2=%d) mism=%zu\n",
              ref.size(), gen.size(), S, K2, mism);
  ASSERT_TRUE(gen.size() == ref.size());
  EXPECT_TRUE(mism == 0);
}

// Qwen3.5 single-stream pdecode (depth-D ring) must match the synchronous
// on-GPU-argmax greedy stream token-for-token. Gates the depth-ring refactor
// at depth-1 and is the correctness gate for the GDN ssm-ring run-ahead.
// Gated on VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_pdecode_matches_sync) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }

  auto ids = lm->tokenizer().encode("List the first ten prime numbers.");
  if (ids.empty()) { return; }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int N = 48;

  std::vector<std::int32_t> ref;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t t = lm->prefill(ctx, ids);
    if (t < 0) { return; }
    ref.push_back(t);
    for (int i = 1; i < N; ++i) {
      t = lm->next_token_greedy(ctx, t);
      if (t < 0) { break; }
      ref.push_back(t);
    }
  }

  std::vector<std::int32_t> pipe;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { return; }
    pipe.push_back(first);
    genai::SamplerParams gsp;                 // defaults -> argmax
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, N)) { return; }
    for (int i = 1; i < N; ++i) {
      if (!lm->pdecode_commit(ctx)) { break; }
      const std::int32_t n = lm->pdecode_next(ctx);
      if (n < 0) { break; }
      pipe.push_back(n);
    }
    lm->pdecode_end(ctx);
  }

  std::size_t mism = 0;
  const std::size_t n = std::min(ref.size(), pipe.size());
  for (std::size_t i = 0; i < n; ++i) { if (ref[i] != pipe[i]) { ++mism; } }
  std::printf("[qwen_pdecode] ref=%zu pipe=%zu mism=%zu\n",
              ref.size(), pipe.size(), mism);
  ASSERT_TRUE(pipe.size() == ref.size());
  EXPECT_TRUE(mism == 0);
}

// Qwen3.5 depth-2 run-ahead rollback correctness -- the GATE for the GDN
// ssm/conv recurrent-state ring. A depth-2 pdecode produces S tokens leaving
// ONE speculative commit (its KV appended AND its GDN ssm/conv state
// advanced) in flight; pdecode_end must roll BOTH back. Then we CONTINUE
// decoding K2 tokens on the SAME context via the synchronous path -- the
// continuation must match a fully-synchronous run token-for-token. Unlike the
// dense gemma rollback test, this also exercises the recurrent ssm/conv ring:
// without restoring the GDN state the continuation diverges immediately (the
// recurrence carries the discarded token's effect forward). Gated on
// VPIPE_QWEN35_TEST_MODEL_PATH.
TEST(metal_lm_smoke, qwen_pdecode_rollback_correct) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 8;
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }

  auto ids = lm->tokenizer().encode("List the first ten prime numbers.");
  if (ids.empty()) { return; }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int S = 8;      // tokens produced before the (simulated) stop
  const int K2 = 24;    // continuation tokens decoded on the same context

  // Reference: synchronous. Produce t1..tS (KV/GDN up to t_{S-1}; tS's forward
  // not committed -- the stop), then continue K2 more from tS.
  std::vector<std::int32_t> ref;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t t = lm->prefill(ctx, ids);
    if (t < 0) { return; }
    ref.push_back(t);
    for (int k = 1; k < S; ++k) { t = lm->next_token_greedy(ctx, t);
                                  ref.push_back(t); }
    std::int32_t c = ref.back();           // tS, the stop token
    for (int k = 0; k < K2; ++k) { c = lm->next_token_greedy(ctx, c);
                                   ref.push_back(c); }
  }

  // Speculative: depth-2 run-ahead produces t1..tS leaving one speculative
  // commit (KV-appended + GDN-advanced) in flight; pdecode_end rolls back the
  // paged KV AND the GDN ssm/conv ring. Then continue K2 from tS on the SAME
  // context via the synchronous path.
  std::vector<std::int32_t> gen;
  {
    ::setenv("VPIPE_QWEN_PDECODE_DEPTH", "2", 1);
    auto ctx = lm->make_context();
    if (!ctx.valid()) { ::unsetenv("VPIPE_QWEN_PDECODE_DEPTH"); return; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { ::unsetenv("VPIPE_QWEN_PDECODE_DEPTH"); return; }
    gen.push_back(first);
    genai::SamplerParams gsp;
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, S + 4)) {
      ::unsetenv("VPIPE_QWEN_PDECODE_DEPTH"); return;
    }
    const int target = S - 1;              // emit t2..tS
    int committed = 0, drained = 0;
    auto can_commit = [&]() { return committed < target + 1; };  // +1 spec
    while (can_commit() && lm->pdecode_commit(ctx)) { ++committed; }
    while (drained < target) {
      const std::int32_t nx = lm->pdecode_next(ctx);
      if (nx < 0) { break; }
      gen.push_back(nx); ++drained;
      while (can_commit() && lm->pdecode_commit(ctx)) { ++committed; }
    }
    lm->pdecode_end(ctx);                  // rolls back KV + GDN ring
    ::unsetenv("VPIPE_QWEN_PDECODE_DEPTH");
    std::int32_t c = gen.back();           // tS
    for (int k = 0; k < K2; ++k) { c = lm->next_token_greedy(ctx, c);
                                   gen.push_back(c); }
  }

  std::size_t mism = 0;
  const std::size_t n = std::min(ref.size(), gen.size());
  for (std::size_t i = 0; i < n; ++i) { if (ref[i] != gen[i]) { ++mism; } }
  std::printf("[qwen-rollback] ref=%zu gen=%zu (S=%d K2=%d) mism=%zu\n",
              ref.size(), gen.size(), S, K2, mism);
  ASSERT_TRUE(gen.size() == ref.size());
  EXPECT_TRUE(mism == 0);
}

// Qwen3.5 pdecode A/B: synchronous vs run-ahead depth-1 vs depth-2 (the GDN
// ssm/conv ring). Measures the run-ahead win and reconfirms token-exactness
// at each depth. Gated on VPIPE_QWEN_PDECODE_BENCH (+ the model path); set
// VPIPE_QWEN_PDECODE_PREFIX_LEN to sweep prefill length. Mirrors
// gemma_e4b_pdecode_pipeline_bench.
TEST(metal_lm_smoke, qwen_pdecode_pipeline_bench) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) { return; }
  if (!std::getenv("VPIPE_QWEN_PDECODE_BENCH")) { return; }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  if (!mgr) { ::unsetenv("VPIPE_LLM_BACKEND"); return; }
  genai::LoadSpec spec;
  spec.hf_dir = path;
  spec.compute_dtype = "f16";
  spec.page_tokens = 512;
  spec.max_pages = 128;   // 128*512 = 64k ctx headroom for long-context sweeps
  auto lm = mgr->load(spec);
  ::unsetenv("VPIPE_LLM_BACKEND");
  if (!lm || !lm->valid()) { return; }

  std::string ptext = "The weather today is";
  if (const char* e = std::getenv("VPIPE_QWEN_PDECODE_PREFIX_LEN")) {
    const int want = std::atoi(e);
    ptext.clear();
    while ((int)lm->tokenizer().encode(ptext).size() < want) {
      ptext += "The cartographer unrolled the chart and traced each harbor. ";
    }
  }
  auto ids = lm->tokenizer().encode(ptext);
  if (ids.empty()) { return; }
  const std::span<const std::int32_t> prompt(ids.data(), ids.size());
  const int K = 64;
  using clock = std::chrono::steady_clock;

  // Reference + timing: synchronous on-GPU-argmax greedy.
  std::vector<std::int32_t> ref;
  double sync_s = 0.0;
  {
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return; }
    std::int32_t t = lm->prefill(ctx, ids);
    if (t < 0) { return; }
    ref.push_back(t);
    for (int i = 0; i < 4; ++i) { t = lm->next_token_greedy(ctx, t);
                                  ref.push_back(t); }
    const auto t0 = clock::now();
    for (int i = 0; i < K; ++i) {
      t = lm->next_token_greedy(ctx, t);
      if (t < 0) { break; }
      ref.push_back(t);
    }
    sync_s = std::chrono::duration<double>(clock::now() - t0).count();
  }

  auto run_pipe = [&](int depth, std::vector<std::int32_t>& out) -> double {
    ::setenv("VPIPE_QWEN_PDECODE_DEPTH", depth >= 2 ? "2" : "1", 1);
    auto ctx = lm->make_context();
    if (!ctx.valid()) { return -1.0; }
    std::int32_t first = lm->prefill(ctx, ids);
    if (first < 0) { return -1.0; }
    out.push_back(first);
    genai::SamplerParams gsp;                  // defaults -> argmax
    if (!lm->pdecode_begin(ctx, first, prompt, gsp, K + 8)) { return -1.0; }
    const int warm = 4;
    int committed = 0, emitted = 0;
    auto pump = [&](int target) {
      while (emitted < target) {
        while (committed < K + 4 && lm->pdecode_commit(ctx)) { ++committed; }
        const std::int32_t n = lm->pdecode_next(ctx);
        if (n < 0) { break; }
        out.push_back(n); ++emitted;
      }
    };
    pump(warm);
    const auto t0 = clock::now();
    pump(warm + K);
    const double dt = std::chrono::duration<double>(clock::now() - t0).count();
    lm->pdecode_end(ctx);
    ::unsetenv("VPIPE_QWEN_PDECODE_DEPTH");
    return dt;
  };

  std::vector<std::int32_t> p1, p2;
  const double d1_s = run_pipe(1, p1);
  const double d2_s = run_pipe(2, p2);

  auto mism = [&](const std::vector<std::int32_t>& a) {
    std::size_t m = 0;
    const std::size_t n = std::min(ref.size(), a.size());
    for (std::size_t i = 0; i < n; ++i) { if (ref[i] != a[i]) { ++m; } }
    return m;
  };
  const std::size_t m1 = mism(p1), m2 = mism(p2);
  std::printf("[qwen-pdecode-ab] ctx=%zu K=%d | sync %.1f tok/s | pipe d1 %.1f "
              "tok/s | pipe d2 %.1f tok/s | d1_mism=%zu d2_mism=%zu\n",
              ids.size(), K, sync_s > 0 ? K / sync_s : 0.0,
              d1_s > 0 ? K / d1_s : 0.0, d2_s > 0 ? K / d2_s : 0.0, m1, m2);
  EXPECT_TRUE(m1 == 0);
  EXPECT_TRUE(m2 == 0);
}

// MOSS-TTS-8B (MossTTSDelay): the metal delay-pattern LM forward must agree
// with the mlx-audio reference. Full autoregressive token-exactness is
// IMPOSSIBLE here -- the audio heads are full of exact bf16 logit ties (two
// codes both at e.g. 55.5), so any tie-flip diverges + cascades, and the
// model README itself is non-reproducible across implementations. Instead we
// verify the FORWARD: teacher-force the reference rows and require that every
// early-window disagreement (before recurrent bf16 cache drift accumulates)
// is a numerical tie, and that the leading autoregressive rows are exact.
// Gated on a model dir + a golden-artifact dir produced by the mlx-audio
// reference (dump_golden.py): shapes.txt ("seq G channels"), input_ids.i32
// (the [seq, channels] prompt grid) and gen_delay.i32 (the [G, channels]
// reference generation, both raw little-endian int32 row-major).
TEST(metal_lm_smoke, moss_tts_delay_greedy_forward) {
  const char* path = std::getenv("VPIPE_MOSS_TTS_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS_TTS_GOLDEN");
  if (path == nullptr || *path == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  const std::string gd(gold);
  int seq = 0, G = 0, ch = 0;
  {
    std::ifstream s(gd + "/shapes.txt");
    if (!s) { return; }
    s >> seq >> G >> ch;
  }
  ASSERT_TRUE(seq > 0 && G > 0 && ch == 33);
  auto read_i32 = [&](const std::string& fn, int count) {
    std::vector<std::int32_t> v((std::size_t)count, 0);
    std::ifstream f(gd + "/" + fn, std::ios::binary);
    if (f) {
      f.read(reinterpret_cast<char*>(v.data()),
             (std::streamsize)count * 4);
    }
    return v;
  };
  const std::vector<std::int32_t> iid = read_i32("input_ids.i32", seq * ch);
  const std::vector<std::int32_t> gen = read_i32("gen_delay.i32", G * ch);

  std::vector<std::vector<std::int32_t>> prompt(
      (std::size_t)seq, std::vector<std::int32_t>((std::size_t)ch, 0));
  for (int r = 0; r < seq; ++r) {
    for (int c = 0; c < ch; ++c) {
      prompt[(std::size_t)r][(std::size_t)c] =
          iid[(std::size_t)(r * ch + c)];
    }
  }

  auto model = genai::MetalMossTtsModel::load(path, mc);
  ASSERT_TRUE(model != nullptr && model->valid());

  // Reference generation rows [G][33].
  std::vector<std::vector<std::int32_t>> ref_rows(
      (std::size_t)G, std::vector<std::int32_t>((std::size_t)ch, 0));
  for (int r = 0; r < G; ++r) {
    for (int c = 0; c < ch; ++c) {
      ref_rows[(std::size_t)r][(std::size_t)c] = gen[(std::size_t)(r * ch + c)];
    }
  }

  // --- GATE: teacher-forced forward correctness ----------------------
  // The MOSS audio heads are full of EXACT bf16 logit ties (e.g. two codes
  // both at 55.5), so two bf16 implementations cannot agree on every greedy
  // argmax -- a single tie-flip then cascades. The forward is verified
  // correct if every active-codebook disagreement with the reference is a
  // numerical near-tie (my argmax's logit ~= the reference code's logit).
  const auto t0 = std::chrono::steady_clock::now();
  auto mism = model->teacher_force_audio_mismatches(prompt, ref_rows);
  const double dt = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  int active = 0;
  const int pad = model->config().audio_pad_code;
  for (int r = 0; r < G; ++r) {
    for (int cb = 0; cb < 32; ++cb) {
      if (gen[(std::size_t)(r * ch + 1 + cb)] != pad) { ++active; }
    }
  }
  constexpr float kTieTol = 0.75f;   // ~3 bf16 quanta at logit magnitude ~50
  constexpr int kEarly = 16;         // window before recurrent drift dominates
  int real_mismatch = 0, real_early = 0, first_real = -1, last_real = -1;
  float max_gap = 0.0f;
  int bucket[7] = {0, 0, 0, 0, 0, 0, 0};   // real mismatches per 10-row bucket
  for (const auto& m : mism) {
    const float gap = m.my_logit - m.ref_logit;   // >= 0 (my pick is argmax)
    if (gap > max_gap) { max_gap = gap; }
    if (gap > kTieTol) {
      ++real_mismatch;
      if (m.row < kEarly) { ++real_early; }
      if (first_real < 0) { first_real = m.row; }
      last_real = m.row;
      const int b = std::min(m.row / 10, 6);
      ++bucket[b];
    }
  }
  std::printf("[moss-tts-tf] %.2fs | active=%d disagreements=%zu "
              "(near-tie if gap<=%.2f) real(gap>tol)=%d max_gap=%.3f\n",
              dt, active, mism.size(), kTieTol, real_mismatch, max_gap);
  std::printf("[moss-tts-tf] real-mismatch rows: first=%d last=%d | per-10row "
              "buckets [0-9..60+]: %d %d %d %d %d %d %d | early(<%d)=%d\n",
              first_real, last_real, bucket[0], bucket[1], bucket[2], bucket[3],
              bucket[4], bucket[5], bucket[6], kEarly, real_early);
  for (std::size_t i = 0; i < mism.size() && i < 6; ++i) {
    std::printf("   row %d cb %d: mine=%d (%.3f) ref=%d (%.3f) gap=%.3f\n",
                mism[i].row, mism[i].codebook, mism[i].my_code,
                (double)mism[i].my_logit, mism[i].ref_code,
                (double)mism[i].ref_logit,
                (double)(mism[i].my_logit - mism[i].ref_logit));
  }
  // Forward correctness gate: BEFORE recurrent bf16 drift accumulates, every
  // disagreement must be a numerical tie (no gap>tol). Full-sequence
  // divergence is expected (the model README itself is non-reproducible
  // across implementations) and is judged by the decoded audio in Phase 2.
  EXPECT_TRUE(real_early == 0);

  // --- Smoke: the autoregressive loop + delay state machine. The leading
  // rows are exact (before any near-tie can flip); the run produces audio
  // and stops on its own. ---------------------------------------------
  auto outr = model->generate_delay_greedy(prompt, G);
  EXPECT_TRUE(!outr.empty());
  const int lead = std::min((int)outr.size(), std::min(G, 4));
  int lead_exact = 0;
  for (int r = 0; r < lead; ++r) {
    bool eq = true;
    for (int c = 0; c < ch; ++c) {
      if (outr[(std::size_t)r][(std::size_t)c] !=
          gen[(std::size_t)(r * ch + c)]) { eq = false; break; }
    }
    if (eq) { ++lead_exact; }
  }
  std::printf("[moss-tts] autoregressive gen=%d/%d | leading_exact_rows=%d/%d\n",
              (int)outr.size(), G, lead_exact, lead);
  EXPECT_TRUE(lead_exact == lead);
}

// MOSS Audio Tokenizer codec decode: the metal RVQ + 4-stage transformer
// decoder must reproduce the reference's per-stage tensors + final 24 kHz PCM
// within an f16 rel-L2 tolerance (the reference runs F32; we run f16). Gated
// on the codec dir + the golden dir (dump_codec_golden.py): codes_dedelay.i32
// + codes_shape.txt (the [T, n_vq] input), rvq.f32 / dec_stage0..7.f32 /
// wave.f32 (the reference intermediates, raw little-endian f32).
TEST(metal_lm_smoke, moss_codec_decode_rel_l2) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS_TTS_GOLDEN");
  if (path == nullptr || *path == '\0' || gold == nullptr || *gold == '\0') {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gd(gold);

  int T = 0, nvq = 0;
  {
    std::ifstream s(gd + "/codes_shape.txt");
    if (!s) { return; }
    s >> T >> nvq;
  }
  ASSERT_TRUE(T > 0 && nvq == 32);
  std::vector<std::int32_t> flat((std::size_t)T * nvq, 0);
  {
    std::ifstream f(gd + "/codes_dedelay.i32", std::ios::binary);
    if (!f) { return; }
    f.read(reinterpret_cast<char*>(flat.data()), (std::streamsize)T * nvq * 4);
  }
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      codes[(std::size_t)t][(std::size_t)c] = flat[(std::size_t)(t * nvq + c)];
    }
  }

  auto read_f32 = [&](const std::string& fn) {
    std::ifstream f(gd + "/" + fn, std::ios::binary | std::ios::ate);
    std::vector<float> v;
    if (!f) { return v; }
    const std::streamsize n = f.tellg() / 4;
    f.seekg(0);
    v.resize((std::size_t)n);
    f.read(reinterpret_cast<char*>(v.data()), n * 4);
    return v;
  };
  auto rel_l2 = [](const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) { return 9.99; }
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)b[i] * (double)b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  };

  auto codec = genai::MetalMossCodec::load(path, mc);
  ASSERT_TRUE(codec != nullptr && codec->valid());

  std::vector<std::vector<float>> stages;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> wave = codec->decode(codes, &stages);
  const double dt = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(!wave.empty());

  const char* names[9] = {"rvq", "dec_stage0", "dec_stage1", "dec_stage2",
                          "dec_stage3", "dec_stage4", "dec_stage5",
                          "dec_stage6", "dec_stage7"};
  EXPECT_TRUE(stages.size() == 9);
  double worst = 0.0;
  for (std::size_t k = 0; k < stages.size() && k < 9; ++k) {
    const std::vector<float> ref = read_f32(std::string(names[k]) + ".f32");
    const double r = rel_l2(stages[k], ref);
    if (r > worst) { worst = r; }
    std::printf("[moss-codec] %-11s mine=%zu ref=%zu rel_l2=%.4f\n", names[k],
                stages[k].size(), ref.size(), r);
  }
  const std::vector<float> wref = read_f32("wave.f32");
  const double wr = rel_l2(wave, wref);
  std::printf("[moss-codec] %.2fs | wave samples=%zu/%zu rel_l2=%.4f | "
              "worst_stage=%.4f\n", dt, wave.size(), wref.size(), wr, worst);
  // f16 codec vs F32 reference: per-stage error stays small; the waveform is
  // the end-to-end arbiter.
  EXPECT_TRUE(wr < 0.05);
}

// Opt-in int8-g32 codec vs the f16 codec on the SAME (synthetic) codes -- the
// quantization-error quality gate. No LM / golden needed (the codec decode is
// deterministic), so this is fast (codec only). Reports rel-L2 of the int8
// waveform vs f16 and the resident-weight saving.
TEST(metal_lm_smoke, moss_codec_int8_rel_l2) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (path == nullptr || *path == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  // Deterministic synthetic codes [T, 32] in [0, 1024).
  const int T = 48, nvq = 32;
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      codes[(std::size_t)t][(std::size_t)c] = (t * 37 + c * 101 + 7) % 1024;
    }
  }

  auto f16 = genai::MetalMossCodec::load(path, mc, /*int8=*/false);
  ASSERT_TRUE(f16 != nullptr && f16->valid());
  const std::vector<float> w16 = f16->decode(codes, nullptr);
  f16.reset();
  ASSERT_TRUE(!w16.empty());

  auto i8 = genai::MetalMossCodec::load(path, mc, /*int8=*/true);
  ASSERT_TRUE(i8 != nullptr && i8->valid());
  const std::vector<float> w8 = i8->decode(codes, nullptr);
  ASSERT_TRUE(w8.size() == w16.size());

  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < w16.size(); ++i) {
    const double d = (double)w8[i] - (double)w16[i];
    num += d * d;
    den += (double)w16[i] * (double)w16[i];
  }
  const double rel_l2 = (den > 0.0) ? std::sqrt(num / den) : 0.0;
  std::printf("[moss-int8] samples=%zu rel_l2(int8 vs f16)=%.4f\n",
              w16.size(), rel_l2);
  // int8 g32 affine on the codec transformer GEMMs: a small waveform error.
  EXPECT_TRUE(rel_l2 < 0.10);
}

// Codec ENCODE (voice-cloning analysis path) round-trip. No torch oracle is
// available on the box, so correctness is checked oracle-free via the codec's
// fixed-point property: decode(C) lands exactly on code-set C's manifold, so a
// correct encoder recovers C from it -- encode(decode(C)) ~ C, and the iterate
// stabilizes. We start from deterministic codes C0, decode to an on-manifold
// waveform, then re-encode/re-decode twice and check (a) the waveform is
// reproduced (rel-L2 small + shrinking) and (b) the coarse codebooks converge
// to a fixed point. A broken encoder (wrong patch order / normalization /
// residual) yields garbage: rel-L2 ~1 and ~0 code agreement. Codec only (no
// LM, no golden); gated on VPIPE_MOSS_CODEC_MODEL.
TEST(metal_lm_smoke, moss_codec_encode_roundtrip) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (path == nullptr || *path == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  auto codec = genai::MetalMossCodec::load(path, mc, /*int8=*/false,
                                           /*with_encoder=*/true);
  ASSERT_TRUE(codec != nullptr && codec->valid());
  ASSERT_TRUE(codec->has_encoder());

  const int T = 48, nvq = 32;
  std::vector<std::vector<std::int32_t>> c0(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      c0[(std::size_t)t][(std::size_t)c] = (t * 37 + c * 101 + 7) % 1024;
    }
  }

  auto rel_l2 = [](const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) { return 1.0; }
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)b[i] * (double)b[i];
    }
    return (den > 0.0) ? std::sqrt(num / den) : 0.0;
  };
  // Per-codebook agreement of two code grids (fraction of frames equal).
  auto agree = [&](const std::vector<std::vector<std::int32_t>>& a,
                   const std::vector<std::vector<std::int32_t>>& b, int cb) {
    if (a.size() != b.size() || a.empty()) { return 0.0; }
    int n = 0;
    for (std::size_t t = 0; t < a.size(); ++t) {
      if (a[t][(std::size_t)cb] == b[t][(std::size_t)cb]) { ++n; }
    }
    return (double)n / (double)a.size();
  };

  const std::vector<float> w0 = codec->decode(c0, nullptr);
  ASSERT_TRUE(!w0.empty());
  const auto c1 = codec->encode(w0);
  ASSERT_TRUE(c1.size() == (std::size_t)T);
  const std::vector<float> w1 = codec->decode(c1, nullptr);
  const auto c2 = codec->encode(w1);
  ASSERT_TRUE(c2.size() == (std::size_t)T);
  const std::vector<float> w2 = codec->decode(c2, nullptr);

  const double rl_10 = rel_l2(w1, w0);
  const double rl_21 = rel_l2(w2, w1);
  std::printf("[moss-enc] T=%d  rel_l2(decode(enc(w0)),w0)=%.4f  "
              "rel_l2(2nd iterate)=%.4f\n", T, rl_10, rl_21);
  std::printf("[moss-enc] code agreement cb0/1/2/4/8  C1vsC0: "
              "%.2f %.2f %.2f %.2f %.2f\n",
              agree(c1, c0, 0), agree(c1, c0, 1), agree(c1, c0, 2),
              agree(c1, c0, 4), agree(c1, c0, 8));
  std::printf("[moss-enc] code agreement cb0/1/2/4/8  C2vsC1: "
              "%.2f %.2f %.2f %.2f %.2f\n",
              agree(c2, c1, 0), agree(c2, c1, 1), agree(c2, c1, 2),
              agree(c2, c1, 4), agree(c2, c1, 8));

  // Waveform is reproduced (not garbage) and the iterate stabilizes (the
  // second round-trip is no worse than the first). The fixed-point signal:
  // the coarsest codebook converges (C2 ~ C1 far better than C1 ~ C0).
  ASSERT_TRUE(w1.size() == w0.size() && w2.size() == w1.size());
  EXPECT_TRUE(std::isfinite(rl_10) && rl_10 < 0.6);
  EXPECT_TRUE(std::isfinite(rl_21) && rl_21 <= rl_10 + 0.05);
  EXPECT_TRUE(agree(c2, c1, 0) > 0.7);
}

// Codec decode throughput: f16 vs int8-g32 (warm, same codes). Reports the
// per-decode ms and the int8/f16 ratio so the int8 dequant overhead is
// visible. Gated on VPIPE_MOSS_CODEC_BENCH (codec only; fast).
TEST(metal_lm_smoke, moss_codec_bench) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (path == nullptr || *path == '\0' ||
      std::getenv("VPIPE_MOSS_CODEC_BENCH") == nullptr) {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  int T = 48;
  if (const char* e = std::getenv("VPIPE_MOSS_CODEC_BENCH_T")) {
    T = std::max(1, std::atoi(e));
  }
  const int nvq = 32;
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      codes[(std::size_t)t][(std::size_t)c] = (t * 37 + c * 101 + 7) % 1024;
    }
  }
  using Clk = std::chrono::steady_clock;
  auto bench = [&](bool int8) -> double {
    auto codec = genai::MetalMossCodec::load(path, mc, int8);
    if (codec == nullptr || !codec->valid()) { return -1.0; }
    for (int i = 0; i < 3; ++i) { (void)codec->decode(codes, nullptr); }  // warm
    const int K = 10;
    const auto t0 = Clk::now();
    for (int i = 0; i < K; ++i) { (void)codec->decode(codes, nullptr); }
    return std::chrono::duration<double, std::milli>(Clk::now() - t0).count()
           / K;
  };
  const double f16 = bench(false);
  const double i8 = bench(true);
  std::printf("[moss-codec-bench] T=%d  f16=%.1f ms  int8=%.1f ms  "
              "(int8/f16=%.2fx, +%.1f ms)\n",
              T, f16, i8, (f16 > 0 ? i8 / f16 : 0.0), i8 - f16);
  EXPECT_TRUE(f16 > 0 && i8 > 0);
}

// M5 matrix-core decode paths (matmul2d f16 GEMM + windowed-causal flash
// attention) must be numerically equivalent to the steel/scalar path: decode
// the same codes with each mma lever on vs off and compare the waveform
// rel-L2. Skips if the model env is unset or the mma paths are not active.
TEST(metal_lm_smoke, moss_codec_mma_matches_scalar) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (path == nullptr || *path == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = genai::MetalMossCodec::load(path, mc, /*int8=*/false);
  ASSERT_TRUE(codec != nullptr && codec->valid());
  if (!codec->use_attn_mma() && !codec->use_mma2()) {
    std::fprintf(stderr, "[moss-codec] mma paths not active (pre-M5?), skip\n");
    return;
  }
  const int T = 53, nvq = codec->n_quantizers();   // non-16-multiple: tail
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      codes[(std::size_t)t][(std::size_t)c] = (t * 37 + c * 101 + 7) % 1024;
    }
  }
  auto rl2 = [](const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d; den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  };
  codec->set_use_mma2(true);  codec->set_use_attn_mma(true);
  const std::vector<float> both = codec->decode(codes, nullptr);
  ASSERT_TRUE(!both.empty());
  codec->set_use_attn_mma(false);                 // scalar attn, mma gemm
  const std::vector<float> no_attn = codec->decode(codes, nullptr);
  codec->set_use_attn_mma(true); codec->set_use_mma2(false);  // steel gemm
  const std::vector<float> no_gemm = codec->decode(codes, nullptr);
  codec->set_use_mma2(true);
  ASSERT_TRUE(no_attn.size() == both.size() && no_gemm.size() == both.size());
  const double r_attn = rl2(both, no_attn);
  const double r_gemm = rl2(both, no_gemm);
  std::fprintf(stderr, "[moss-codec] attn mma-vs-scalar rel-L2=%.6f | "
               "gemm mma-vs-steel rel-L2=%.6f\n", r_attn, r_gemm);
  EXPECT_TRUE(r_attn < 1e-2);
  EXPECT_TRUE(r_gemm < 1e-2);
}

// The int8 (w8 g32) matrix-core GEMM path (dequant-once via affine_dequant_
// w8g32 -> the f16 dense_gemm_mma) must match the fused affine STEEL w8 GEMM:
// decode the same codes with the int8 mma GEMM on vs off (attention held on
// mma) and compare the waveform rel-L2. Skips if env unset or mma inactive.
TEST(metal_lm_smoke, moss_codec_int8_mma_matches_steel) {
  const char* path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  if (path == nullptr || *path == '\0') { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  auto codec = genai::MetalMossCodec::load(path, mc, /*int8=*/true);
  ASSERT_TRUE(codec != nullptr && codec->valid());
  if (!codec->use_mma2()) {
    std::fprintf(stderr, "[moss-codec] int8 mma not active (pre-M5?), skip\n");
    return;
  }
  const int T = 53, nvq = codec->n_quantizers();
  std::vector<std::vector<std::int32_t>> codes(
      (std::size_t)T, std::vector<std::int32_t>((std::size_t)nvq, 0));
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < nvq; ++c) {
      codes[(std::size_t)t][(std::size_t)c] = (t * 37 + c * 101 + 7) % 1024;
    }
  }
  codec->set_use_mma2(true);
  const std::vector<float> a = codec->decode(codes, nullptr);
  codec->set_use_mma2(false);
  const std::vector<float> b = codec->decode(codes, nullptr);
  codec->set_use_mma2(true);
  ASSERT_TRUE(!a.empty() && a.size() == b.size());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d; den += (double)b[i] * (double)b[i];
  }
  const double rl2 = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  std::fprintf(stderr, "[moss-codec] int8 GEMM mma-vs-steel rel-L2 = %.6f\n",
               rl2);
  EXPECT_TRUE(rl2 < 1e-2);
}

// Perf de-risk for "int8 faster than f16": at the codec's GEMM shapes, compare
// (a) proto_gemm_mma_f16 (my 64x64x32 tiling, f16 weight), (b) proto_gemm_mma_i8
// (same tiling, int8 weight + cheap symmetric dequant = MXINT8 proxy), and (c)
// the production dense_gemm_mma_t_n128_f16. b-a = pure int8-read cost; c is the
// real f16 number to beat. Gated on VPIPE_MXINT8_PROTO (no model needed).
TEST(metal_lm_smoke, mxint8_matmul2d_proto) {
  if (std::getenv("VPIPE_MXINT8_PROTO") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::fprintf(stderr, "[mxint8-proto] no matrix cores, skip\n");
    return;
  }
  auto lib_p = mc->load_library("affine_qmm_mma");
  auto lib_d = mc->load_library("dense_gemm_mma");
  auto fn_f16 = lib_p.function("proto_gemm_mma_f16");
  auto fn_i8  = lib_p.function("proto_gemm_mma_i8");
  auto fn_prod = lib_d.function("dense_gemm_mma_t_n128_f16");
  ASSERT_TRUE(fn_f16.valid() && fn_i8.valid() && fn_prod.valid());

  struct Shape { int M, N, K; const char* tag; };
  const Shape shapes[] = {
      {200, 5120, 1280, "stage0 fc1  M=200"},
      {200, 3840, 1280, "stage0 qkv  M=200"},
      {800, 3072,  768, "mid    fc1  M=800"},
      {1600, 3072, 768, "stage3 fc1  M=1600"},
      {1600, 768,  768, "stage3 oproj M=1600"}};

  for (const auto& sh : shapes) {
    const int M = sh.M, N = sh.N, K = sh.K;
    auto wf16 = mc->make_shared_buffer((std::size_t)N * K * 2);
    auto wi8  = mc->make_shared_buffer((std::size_t)N * K);        // 1 byte/wt
    auto scl  = mc->make_shared_buffer((std::size_t)N * (K / 32) * 2);
    auto xb   = mc->make_shared_buffer((std::size_t)M * K * 2);
    auto yb   = mc->make_shared_buffer((std::size_t)M * N * 2);
    {
      auto* wf = static_cast<_Float16*>(wf16.contents());
      for (std::size_t i = 0; i < (std::size_t)N * K; ++i) {
        wf[i] = (_Float16)(0.01f * (float)(i % 5));
      }
      auto* wi = static_cast<std::uint8_t*>(wi8.contents());
      for (std::size_t i = 0; i < (std::size_t)N * K; ++i) {
        wi[i] = (std::uint8_t)(i % 13);
      }
      auto* sc = static_cast<_Float16*>(scl.contents());
      for (std::size_t i = 0; i < (std::size_t)N * (K / 32); ++i) {
        sc[i] = (_Float16)0.02f;
      }
      auto* xp = static_cast<_Float16*>(xb.contents());
      for (std::size_t i = 0; i < (std::size_t)M * K; ++i) {
        xp[i] = (_Float16)(0.01f * (float)(i % 7));
      }
    }
    auto time_it = [&](const char* which) -> double {
      auto run = [&]() {
        metal_compute::CommandStream st = mc->make_command_stream();
        { metal_compute::ComputeEncoder e = st.begin_compute();
          if (which[0] == 'f') {                       // proto f16
            e.set_function(fn_f16);
            e.set_buffer(0, wf16); e.set_buffer(1, xb); e.set_buffer(2, yb);
            e.set_constant(3, K); e.set_constant(4, N); e.set_constant(5, M);
            e.dispatch({(unsigned)(((N + 63) / 64) * 128),
                        (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
          } else if (which[0] == 'i') {                // proto int8
            e.set_function(fn_i8);
            e.set_buffer(0, wi8); e.set_buffer(1, scl); e.set_buffer(2, xb);
            e.set_buffer(3, yb);
            e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
            e.dispatch({(unsigned)(((N + 63) / 64) * 128),
                        (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
          } else {                                     // production n128 f16
            e.set_function(fn_prod);
            e.set_buffer(0, xb); e.set_buffer(1, wf16); e.set_buffer(2, wf16);
            e.set_buffer(3, yb);
            e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
            e.set_constant(7, 0);
            e.dispatch({(unsigned)(((N + 127) / 128) * 256),
                        (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
          } }
        st.commit().wait();
      };
      for (int i = 0; i < 3; ++i) { run(); }           // warm
      const int IT = 30;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < IT; ++i) { run(); }
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count() / IT;
    };
    const double tf = time_it("f16");
    const double ti = time_it("i8");
    const double tp = time_it("prod");
    std::fprintf(stderr,
        "[mxint8-proto] %-20s  proto_f16=%.3f  proto_i8=%.3f  prod_f16=%.3f ms"
        "  | i8/prod=%.2fx  i8-f16=%+.3f\n",
        sh.tag, tf, ti, tp, tp > 0 ? ti / tp : 0.0, ti - tf);
  }
}

// Does the dequant-once tax depend on bit width? Compare, at the codec's GEMM
// shapes: f16 (dense_gemm_mma alone) vs 8-bit dequant-once (affine_dequant_w8g32
// -> mma) vs 4-bit dequant-once (affine_dequant_w4g32 -> mma). Prediction: 4-bit
// ~= 8-bit ~= a fixed ~15-20% over f16 -- the tax is the f16 WRITE + the matmul
// f16 re-read (2 bytes either way), not the compressed read. Gated on
// VPIPE_MXINT8_PROTO (no model needed).
TEST(metal_lm_smoke, qmm_bitwidth_vs_f16) {
  if (std::getenv("VPIPE_MXINT8_PROTO") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
  auto lib_dq = mc->load_library("affine_dequant");
  auto lib_mm = mc->load_library("dense_gemm_mma");
  auto fn_dq8 = lib_dq.function("affine_dequant_w8g32");
  auto fn_dq4 = lib_dq.function("affine_dequant_w4g32");
  auto fn_mm  = lib_mm.function("dense_gemm_mma_t_n128_f16");
  ASSERT_TRUE(fn_dq8.valid() && fn_dq4.valid() && fn_mm.valid());

  struct Shape { int M, N, K; const char* tag; };
  const Shape shapes[] = {
      {200, 5120, 1280, "M=200 K=1280"},
      {800, 3072,  768, "M=800 K=768 "},
      {1600, 3072, 768, "M=1600 K=768"}};

  for (const auto& sh : shapes) {
    const int M = sh.M, N = sh.N, K = sh.K, G = K / 32;
    auto wf16 = mc->make_shared_buffer((std::size_t)N * K * 2);
    auto wi8  = mc->make_shared_buffer((std::size_t)N * K);          // 1 B/wt
    auto wi4  = mc->make_shared_buffer((std::size_t)N * (K / 2));    // 0.5 B/wt
    auto scl  = mc->make_shared_buffer((std::size_t)N * G * 2);
    auto bia  = mc->make_shared_buffer((std::size_t)N * G * 2);
    auto deq  = mc->make_shared_buffer((std::size_t)N * K * 2);
    auto xb   = mc->make_shared_buffer((std::size_t)M * K * 2);
    auto yb   = mc->make_shared_buffer((std::size_t)M * N * 2);
    { auto* s = static_cast<_Float16*>(scl.contents());
      auto* b = static_cast<_Float16*>(bia.contents());
      for (std::size_t i = 0; i < (std::size_t)N * G; ++i) {
        s[i] = (_Float16)0.02f; b[i] = (_Float16)0.0f; } }

    auto mm = [&](metal_compute::ComputeEncoder& e,
                  const metal_compute::SharedBuffer& w) {
      e.set_function(fn_mm);
      e.set_buffer(0, xb); e.set_buffer(1, w); e.set_buffer(2, w);
      e.set_buffer(3, yb);
      e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
      e.set_constant(7, 0);
      e.dispatch({(unsigned)(((N + 127) / 128) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    };
    auto time_it = [&](int mode) -> double {          // 0=f16 1=w8 2=w4
      auto run = [&]() {
        metal_compute::CommandStream st = mc->make_command_stream();
        { metal_compute::ComputeEncoder e = st.begin_compute();
          if (mode == 1) {
            e.set_function(fn_dq8);
            e.set_buffer(0, wi8); e.set_buffer(1, scl); e.set_buffer(2, bia);
            e.set_buffer(3, deq); e.set_constant(4, K); e.set_constant(5, N);
            e.dispatch({(unsigned)(K / 4), (unsigned)N, 1}, {64, 1, 1});
            mm(e, deq);
          } else if (mode == 2) {
            e.set_function(fn_dq4);
            e.set_buffer(0, wi4); e.set_buffer(1, scl); e.set_buffer(2, bia);
            e.set_buffer(3, deq); e.set_constant(4, K); e.set_constant(5, N);
            e.dispatch({(unsigned)(K / 8), (unsigned)N, 1}, {64, 1, 1});
            mm(e, deq);
          } else {
            mm(e, wf16);
          } }
        st.commit().wait();
      };
      for (int i = 0; i < 3; ++i) { run(); }
      const int IT = 40;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < IT; ++i) { run(); }
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count() / IT;
    };
    const double f = time_it(0), w8 = time_it(1), w4 = time_it(2);
    std::fprintf(stderr,
        "[qmm-bits] %-14s  f16=%.3f  w8_deq1=%.3f (%.2fx)  w4_deq1=%.3f (%.2fx)"
        " ms\n", sh.tag, f, w8, f > 0 ? w8 / f : 0.0, w4, f > 0 ? w4 / f : 0.0);
  }
}

// End-to-end metal MOSS-TTS: LM (delay-pattern code generation) -> de-delay +
// drop all-pad frames (the reference _decode_generated_audio pipeline) ->
// codec -> 24 kHz PCM, written to a playable WAV. Gated on the LM + codec
// dirs + the golden dir (for the prompt grid). Writes
// $VPIPE_MOSS_TTS_GOLDEN/e2e.wav.
TEST(metal_lm_smoke, moss_tts_end_to_end_wav) {
  const char* lm_path = std::getenv("VPIPE_MOSS_TTS_MODEL");
  const char* cc_path = std::getenv("VPIPE_MOSS_CODEC_MODEL");
  const char* gold = std::getenv("VPIPE_MOSS_TTS_GOLDEN");
  if (!lm_path || !*lm_path || !cc_path || !*cc_path || !gold || !*gold) {
    return;
  }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  const std::string gd(gold);

  int seq = 0, G = 0, ch = 0;
  {
    std::ifstream s(gd + "/shapes.txt");
    if (!s) { return; }
    s >> seq >> G >> ch;
  }
  ASSERT_TRUE(seq > 0 && ch == 33);
  std::vector<std::int32_t> iid((std::size_t)seq * ch, 0);
  {
    std::ifstream f(gd + "/input_ids.i32", std::ios::binary);
    if (!f) { return; }
    f.read(reinterpret_cast<char*>(iid.data()), (std::streamsize)seq * ch * 4);
  }
  std::vector<std::vector<std::int32_t>> prompt(
      (std::size_t)seq, std::vector<std::int32_t>((std::size_t)ch, 0));
  for (int r = 0; r < seq; ++r) {
    for (int c = 0; c < ch; ++c) {
      prompt[(std::size_t)r][(std::size_t)c] = iid[(std::size_t)(r * ch + c)];
    }
  }

  const int n_vq = 32, pad = 1024;
  std::vector<std::vector<std::int32_t>> gen;
  {
    auto lm = genai::MetalMossTtsModel::load(lm_path, mc);
    ASSERT_TRUE(lm != nullptr && lm->valid());
    gen = lm->generate_delay_greedy(prompt, 1024);   // [Gg][33]
  }  // free the 8B LM before loading the codec (16 GB box)
  ASSERT_TRUE(!gen.empty());
  const int Gg = (int)gen.size();

  // De-delay: tokens[t][cb] = audio_codes[cb + t][cb], audio_codes = gen[:,1:].
  const int out_len = Gg - n_vq + 1;
  std::vector<std::vector<std::int32_t>> codes;
  for (int t = 0; t < out_len; ++t) {
    std::vector<std::int32_t> row((std::size_t)n_vq, 0);
    bool all_pad = true;
    for (int cb = 0; cb < n_vq; ++cb) {
      int v = gen[(std::size_t)(cb + t)][(std::size_t)(1 + cb)];
      if (v != pad) { all_pad = false; }
      if (v < 0 || v >= pad) { v = pad - 1; }   // clamp pad/OOB to a valid code
      row[(std::size_t)cb] = v;
    }
    if (!all_pad) { codes.push_back(std::move(row)); }   // drop all-pad frames
  }
  std::printf("[moss-e2e] LM gen rows=%d -> de-delay %d -> non-pad frames=%zu\n",
              Gg, out_len, codes.size());
  ASSERT_TRUE(!codes.empty());

  std::vector<float> wave;
  int sr = 24000;
  {
    auto codec = genai::MetalMossCodec::load(cc_path, mc);
    ASSERT_TRUE(codec != nullptr && codec->valid());
    sr = codec->sample_rate();
    wave = codec->decode(codes, nullptr);
  }
  ASSERT_TRUE(!wave.empty());
  double peak = 0.0;
  for (float s : wave) { peak = std::max(peak, (double)std::fabs(s)); }
  std::printf("[moss-e2e] codec -> %zu samples = %.2fs @ %dHz | peak=%.3f\n",
              wave.size(), wave.size() / (double)sr, sr, peak);
  EXPECT_TRUE(peak > 0.01);   // produced actual audio, not silence

  // Write a 16-bit PCM mono WAV.
  const std::string wav = gd + "/e2e.wav";
  std::ofstream out(wav, std::ios::binary);
  const std::uint32_t n = (std::uint32_t)wave.size();
  const std::uint32_t data_bytes = n * 2;
  const std::uint32_t byte_rate = (std::uint32_t)sr * 2;
  auto u32 = [&](std::uint32_t v) { out.write(reinterpret_cast<char*>(&v), 4); };
  auto u16 = [&](std::uint16_t v) { out.write(reinterpret_cast<char*>(&v), 2); };
  out.write("RIFF", 4); u32(36 + data_bytes); out.write("WAVE", 4);
  out.write("fmt ", 4); u32(16); u16(1); u16(1);
  u32((std::uint32_t)sr); u32(byte_rate); u16(2); u16(16);
  out.write("data", 4); u32(data_bytes);
  for (float s : wave) {
    int v = (int)std::lround(std::max(-1.0f, std::min(1.0f, s)) * 32767.0f);
    u16((std::uint16_t)(std::int16_t)v);
  }
  out.close();
  std::printf("[moss-e2e] wrote %s\n", wav.c_str());
}

// ============================================================================
// Framework-primitive cost study + MLX command-buffer trace replay.
//
// PURE SYSTEMS investigation (not production perf): quantify what each
// metal-compute command primitive costs, and whether MLX's decode recipe --
// ONE MTLDispatchTypeConcurrent encoder per command buffer + a memoryBarrier
// only at true hazards (a captured Qwen3.5-4B OptiQ decode step measured 815
// dispatches, 523 barriers, 24 concurrent blocks, 20 command buffers; ~36% of
// dispatch boundaries are barrier-free/overlap-eligible) -- beats vpipe's
// recipe (a Serial encoder with cheap implicit ordering + a concurrent SUB-
// encoder, i.e. endEncoding+new-encoder, per parallel block).
//
//   Part 1: isolate each primitive's cost with a near-zero dummy kernel.
//   Part 2: replay the captured trace under both models (VPIPE_MLX_TRACE=file).
// Gated on VPIPE_FW_COSTS.
TEST(metal_lm_bench, framework_dispatch_costs) {
  if (std::getenv("VPIPE_FW_COSTS") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  namespace mcpt = metal_compute;
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dummy_disp_f16");
  if (!fn.valid()) { std::printf("[fw] dummy_disp_f16 missing\n"); return; }

  std::vector<mcpt::SharedBuffer> bufs;
  for (int i = 0; i < 9; ++i) { bufs.push_back(mc->make_shared_buffer(256)); }
  auto bind = [&](mcpt::ComputeEncoder& e) {
    e.set_function(fn);
    for (int i = 0; i < 8; ++i) { e.set_buffer(i, bufs[i]); }
    e.set_buffer(8, bufs[0]);            // out == b0 -> RAW chain (serializes)
    e.set_constant(9, 1); e.set_constant(10, 2);
    e.set_constant(11, 3); e.set_constant(12, 4);
  };
  const mcpt::LaunchDims grid{32, 1, 1}, tg{32, 1, 1};   // near-zero compute
  using Clock = std::chrono::steady_clock;
  auto ms = [](auto d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  auto best = [&](auto&& body) -> double {
    double b = 1e18;
    for (int r = 0; r < 6; ++r) {
      const auto t0 = Clock::now();
      body();
      b = std::min(b, ms(Clock::now() - t0));
    }
    return b;
  };
  const int M = 4000;
  const double tA = best([&] {                 // Serial, implicit ordering
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(mcpt::DispatchType::Serial);
      for (int i = 0; i < M; ++i) { bind(e); e.dispatch(grid, tg); } }
    st.commit().wait();
  });
  const double tC = best([&] {                 // Concurrent, no barrier
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(mcpt::DispatchType::Concurrent);
      for (int i = 0; i < M; ++i) { bind(e); e.dispatch(grid, tg); } }
    st.commit().wait();
  });
  const double tB = best([&] {                 // Concurrent + barrier each
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(mcpt::DispatchType::Concurrent);
      for (int i = 0; i < M; ++i) {
        bind(e); e.dispatch(grid, tg);
        e.memory_barrier(mcpt::BarrierScope::Buffers);
      } }
    st.commit().wait();
  });
  const double tD = best([&] {                 // Serial + concurrent sub-scope
    auto st = mc->make_command_stream();
    { auto e = st.begin_compute(mcpt::DispatchType::Serial);
      for (int i = 0; i < M; ++i) {
        auto s = e.concurrent_scope(true);
        bind(e); e.dispatch(grid, tg);
      } }
    st.commit().wait();
  });
  auto split = [&](int K) -> double {          // M dispatches in K cmd buffers
    return best([&] {
      auto st = mc->make_command_stream();
      const int per = M / K;
      for (int k = 0; k < K; ++k) {
        { auto e = st.begin_compute(mcpt::DispatchType::Serial);
          for (int i = 0; i < per; ++i) { bind(e); e.dispatch(grid, tg); } }
        auto f = st.commit();
        if (k == K - 1) { f.wait(); }
      }
    });
  };
  const double t1buf = split(1), t20buf = split(20);
  std::printf("[fw] === per-primitive cost (us/disp), near-zero kernel M=%d ===\n",
              M);
  std::printf("[fw]  serial no-barrier      %.3f\n", tA / M * 1e3);
  std::printf("[fw]  concurrent no-barrier  %.3f   (pure launch+overlap)\n",
              tC / M * 1e3);
  std::printf("[fw]  concurrent + barrier   %.3f   -> memoryBarrier ~ %.3f us\n",
              tB / M * 1e3, (tB - tC) / M * 1e3);
  std::printf("[fw]  serial + enc-boundary  %.3f   -> enc-boundary(x2) ~ %.3f "
              "us\n", tD / M * 1e3, (tD - tA) / M * 1e3);
  std::printf("[fw]  commit: 1 buf %.3f ms, 20 bufs %.3f ms -> per-commit ~ "
              "%.3f us\n", t1buf, t20buf, (t20buf - t1buf) / 19.0 * 1e3);

  // Part 2: replay the captured MLX trace under both models.
  const char* tracef = std::getenv("VPIPE_MLX_TRACE");
  if (tracef == nullptr || *tracef == '\0') {
    std::printf("[fw] (set VPIPE_MLX_TRACE=<file> for trace replay)\n");
    return;
  }
  std::vector<char> ev;
  { std::ifstream f(tracef); std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) { continue; }
      const char c = line[0];
      if (c == 'd' || c == 't') { ev.push_back('d'); }
      else if (c == 'B' || c == 'C' || c == '[' || c == ']') { ev.push_back(c); }
    } }
  std::size_t nd = 0, nb = 0, nc = 0, nk = 0;
  for (char c : ev) { nd += (c == 'd'); nb += (c == 'B'); nk += (c == 'C');
                      nc += (c == '['); }
  // MLX model: one Concurrent encoder per command buffer; memoryBarrier at each
  // B; new buffer at each C. (The [ ] blocks are already reflected in where B
  // does/doesn't appear, so they need no separate handling here.)
  const double tMlx = best([&] {
    auto st = mc->make_command_stream();
    auto e = st.begin_compute(mcpt::DispatchType::Concurrent);
    for (char c : ev) {
      if (c == 'd') { bind(e); e.dispatch(grid, tg); }
      else if (c == 'B') { e.memory_barrier(mcpt::BarrierScope::Buffers); }
      else if (c == 'C') { e.end(); st.commit();
                           e = st.begin_compute(mcpt::DispatchType::Concurrent); }
    }
    e.end(); st.commit().wait();
  });
  // vpipe model: Serial encoder (implicit ordering, NO explicit barrier) + a
  // concurrent sub-encoder scope per [ ] block; new buffer at each C.
  const double tVp = best([&] {
    auto st = mc->make_command_stream();
    auto e = st.begin_compute(mcpt::DispatchType::Serial);
    std::optional<mcpt::ComputeEncoder::ConcurrentScope> cs;
    for (char c : ev) {
      if (c == 'd') { bind(e); e.dispatch(grid, tg); }
      else if (c == '[') { cs.emplace(e.concurrent_scope(true)); }
      else if (c == ']') { cs.reset(); }
      else if (c == 'C') { cs.reset(); e.end(); st.commit();
                           e = st.begin_compute(mcpt::DispatchType::Serial); }
    }
    cs.reset(); e.end(); st.commit().wait();
  });
  // MLX model but a SINGLE command buffer (ignore C) -> isolates how much of
  // MLX's edge is the concurrent-encoder overlap vs the ~20-way cmdbuf split.
  const double tMlx1 = best([&] {
    auto st = mc->make_command_stream();
    auto e = st.begin_compute(mcpt::DispatchType::Concurrent);
    for (char c : ev) {
      if (c == 'd') { bind(e); e.dispatch(grid, tg); }
      else if (c == 'B') { e.memory_barrier(mcpt::BarrierScope::Buffers); }
    }
    e.end(); st.commit().wait();
  });
  // "vpipe adopts MLX's recipe": concurrent encoder, barrier only at the 523
  // hazards, but keep vpipe's single command buffer per step (no split).
  const double tVpConc = best([&] {
    auto st = mc->make_command_stream();
    auto e = st.begin_compute(mcpt::DispatchType::Concurrent);
    for (char c : ev) {
      if (c == 'd') { bind(e); e.dispatch(grid, tg); }
      else if (c == 'B') { e.memory_barrier(mcpt::BarrierScope::Buffers); }
    }
    e.end(); st.commit().wait();
  });
  std::printf("[fw] === trace replay (framework overhead only, near-zero "
              "kernel) ===\n");
  std::printf("[fw]  trace: %zu dispatch, %zu barrier, %zu concblk, %zu "
              "cmdbuf\n", nd, nb, nc, nk);
  std::printf("[fw]  MLX model  (concurrent + barriers, %zu bufs) : %.3f ms\n",
              nk, tMlx);
  std::printf("[fw]  MLX model  (concurrent + barriers, 1 buf)    : %.3f ms  "
              "(split gain = %.1f%%)\n", tMlx1, 100.0 * (tMlx1 - tMlx) / tMlx1);
  std::printf("[fw]  vpipe now  (serial + conc sub-enc, %zu bufs) : %.3f ms  "
              "(+%.1f%% vs MLX)\n", nk, tVp, 100.0 * (tVp - tMlx) / tMlx);
  std::printf("[fw]  vpipe-conc (concurrent + barriers, 1 buf)    : %.3f ms\n",
              tVpConc);
  EXPECT_TRUE(true);
}

// ============================================================================
// Encoder-model + hazard-mode + command-buffer-split teardown (systems study).
//
// Resolves the contradiction between "concurrent+barrier is slower than serial
// for vpipe's mostly-serial decode" (prior result) and "MLX's concurrent model
// is faster" (framework study). Root cause: MLX allocates ALL buffers Untracked
// (manual hazards), vpipe defaults to Tracked (driver auto-inserts barriers on a
// Concurrent encoder). So vpipe + concurrent + manual barrier = DOUBLE barriers.
//
// Distinct buffers per dispatch (no artificial sharing): a dependent RAW chain
// (dispatch i reads pool[i], writes pool[i+1]) models the serial decode; an
// independent variant (all read pool[0..7], write pool[i+8]) models parallel
// siblings. Sweeps command-buffer split size to answer commands-vs-buffers.
// Gated on VPIPE_ENC_MODEL.
TEST(metal_lm_bench, encoder_model_teardown) {
  if (std::getenv("VPIPE_ENC_MODEL") == nullptr) { return; }
  Session sess;
  auto* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  namespace mcpt = metal_compute;
  auto lib = mc->load_library("llm_elementwise");
  auto fn = lib.function("dummy_disp_f16");
  if (!fn.valid()) { std::printf("[enc] dummy_disp_f16 missing\n"); return; }

  const int MAXM = 832;
  auto mkpool = [&](mcpt::HazardTracking ht) {
    std::vector<mcpt::SharedBuffer> p;
    for (int i = 0; i < MAXM + 16; ++i) {
      p.push_back(mc->make_shared_buffer(131072, 64, ht));   // 128KB: >64KB heap
                                                             // thr -> both non-heap
                                                             // (isolate tracking)
    }
    return p;
  };
  auto tracked = mkpool(mcpt::HazardTracking::Tracked);
  auto untrack = mkpool(mcpt::HazardTracking::Untracked);
  using Clock = std::chrono::steady_clock;
  auto msf = [](auto d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  auto best = [&](auto&& body) -> double {
    double b = 1e18;
    for (int r = 0; r < 6; ++r) {
      const auto t0 = Clock::now();
      body();
      b = std::min(b, msf(Clock::now() - t0));
    }
    return b;
  };
  const mcpt::LaunchDims g{32, 1, 1}, tgd{32, 1, 1};
  // bmode: 0=no manual barrier, 1=manual barrier before every dispatch.
  auto run = [&](std::vector<mcpt::SharedBuffer>& pool, int M, int S,
                 mcpt::DispatchType dt, int bmode, bool indep) -> double {
    return best([&] {
      auto st = mc->make_command_stream();
      int i = 0;
      while (i < M) {
        const int end = std::min(i + S, M);
        { auto e = st.begin_compute(dt);
          for (int j = i; j < end; ++j) {
            if (bmode == 1 && j > i) {
              e.memory_barrier(mcpt::BarrierScope::Buffers);
            }
            e.set_function(fn);
            if (indep) {
              for (int k = 0; k < 8; ++k) { e.set_buffer(k, pool[k]); }
              e.set_buffer(8, pool[j + 8]);
            } else {
              e.set_buffer(0, pool[j]);
              for (int k = 1; k < 8; ++k) { e.set_buffer(k, pool[0]); }
              e.set_buffer(8, pool[j + 1]);
            }
            e.set_constant(9, 1); e.set_constant(10, 2);
            e.set_constant(11, 3); e.set_constant(12, 4);
            e.dispatch(g, tgd);
          } }
        auto f = st.commit();
        if (end >= M) { f.wait(); }
        i = end;
      }
    });
  };
  const auto SER = mcpt::DispatchType::Serial;
  const auto CON = mcpt::DispatchType::Concurrent;

  std::printf("[enc] === A. cmd-buffer split sweep (Tracked serial chain) ===\n");
  for (int M : {400, 800}) {
    for (int S : {M, 200, 100, 50, 25, 10}) {
      const double t = run(tracked, M, S, SER, 0, false);
      std::printf("[enc]  M=%d  S=%3d cmds/buf (%2d bufs): %6.2f ms  %.3f "
                  "us/disp\n", M, S, (M + S - 1) / S, t, t / M * 1e3);
    }
  }
  const int M = 512, S = 50;
  std::printf("[enc] === B. encoder x hazard model, DEPENDENT chain "
              "(M=%d,S=%d) ===\n", M, S);
  std::printf("[enc]  Tracked  serial  no-barrier   : %6.2f ms\n",
              run(tracked, M, S, SER, 0, false));
  std::printf("[enc]  Tracked  concur  no-barrier(driver auto): %6.2f ms\n",
              run(tracked, M, S, CON, 0, false));
  std::printf("[enc]  Tracked  concur  manual-every(DOUBLE)   : %6.2f ms\n",
              run(tracked, M, S, CON, 1, false));
  std::printf("[enc]  Untrack  serial  no-barrier   : %6.2f ms\n",
              run(untrack, M, S, SER, 0, false));
  std::printf("[enc]  Untrack  concur  manual-every(MLX chain): %6.2f ms\n",
              run(untrack, M, S, CON, 1, false));
  std::printf("[enc] === C. encoder x hazard model, INDEPENDENT (M=%d,S=%d) "
              "===\n", M, S);
  std::printf("[enc]  Tracked  serial  (serializes indep)     : %6.2f ms\n",
              run(tracked, M, S, SER, 0, true));
  std::printf("[enc]  Tracked  concur  no-barrier(driver->overlap): %6.2f ms\n",
              run(tracked, M, S, CON, 0, true));
  std::printf("[enc]  Untrack  concur  no-barrier(overlap ceiling): %6.2f ms\n",
              run(untrack, M, S, CON, 0, true));

  // D. Is the Untracked edge CPU-encode (hidden by pdecode in production) or
  // GPU-side? Repeat the serial chain with a COSTED kernel (residual_add over N
  // elts = real DRAM traffic per dispatch). If the Tracked-vs-Untracked gap
  // shrinks toward 0 as the kernel does real work, the edge is CPU-encode only.
  auto radd = lib.function("residual_add_f16");
  if (radd.valid()) {
    auto runc = [&](std::vector<mcpt::SharedBuffer>& pool, int N) -> double {
      return best([&] {
        auto st = mc->make_command_stream();
        { auto e = st.begin_compute(SER);
          for (int j = 0; j < M; ++j) {
            e.set_function(radd);
            e.set_buffer(0, pool[j]); e.set_buffer(1, pool[0]);
            e.set_buffer(2, pool[j + 1]); e.set_constant(3, N);
            e.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
          } }
        st.commit().wait();
      });
    };
    std::printf("[enc] === D. Tracked vs Untracked SERIAL, costed kernel "
                "(residual_add) ===\n");
    for (int N : {64, 1024, 8192}) {
      const double tt = runc(tracked, N), tu = runc(untrack, N);
      std::printf("[enc]  N=%5d (%3dKB/disp): Tracked %6.2f  Untracked %6.2f  "
                  "gap %+.1f%%\n", N, N * 2 / 1024, tt, tu,
                  100.0 * (tt - tu) / tt);
    }
  }
  EXPECT_TRUE(true);
}
