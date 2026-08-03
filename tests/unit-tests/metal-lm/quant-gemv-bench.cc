// quant-gemv-bench.cc -- qmv (decode-GEMV) bandwidth and A/B probes: w4-vs-w8
// effective DRAM bandwidth, chain/residency barriers, fuse-vs-split, mixed-
// width encoders, the batch sweep, the sampler kernel microbench, per-token
// rope cost and the bit-width-vs-f16 / mxint8 matmul prototypes. All env-gated.

#include "tests/unit-tests/metal-lm/metal-lm-test-common.h"

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
