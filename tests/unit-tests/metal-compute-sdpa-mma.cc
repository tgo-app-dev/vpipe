// metal-compute-sdpa-mma.cc -- correctness + throughput for the M5
// matrix-core flash attention (sdpa_mma, head_dim 256) vs a CPU oracle
// and the scalar query-tiled kernel (sdpa_paged_qtile). Contiguous K/V,
// causal, GQA. Runs in BOTH builds; skips the matrix-core half on GPUs
// without matrix cores (steel-only is still a valid pass).

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

inline std::uint16_t f32_to_h(float f) {  // f32 -> IEEE half (round-to-nearest)
  std::uint32_t x; std::memcpy(&x, &f, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  std::int32_t exp = (std::int32_t)((x >> 23) & 0xff) - 127 + 15;
  std::uint32_t man = x & 0x7fffffu;
  if (exp <= 0) { return (std::uint16_t)sign; }
  if (exp >= 0x1f) { return (std::uint16_t)(sign | 0x7c00u); }
  return (std::uint16_t)(sign | (exp << 10) | (man >> 13));
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

MetalCompute* get_mc_(Session& s) {
  MetalCompute* mc = s.metal_compute();
  return (mc != nullptr && mc->valid()) ? mc : nullptr;
}
double secs_(std::chrono::steady_clock::time_point a,
             std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

}  // namespace

TEST(sdpa_mma, correctness_and_speed) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("sdpa_mma");
  ComputeFunction fn = lib.function("sdpa_mma_f16");
  if (!fn.valid()) {
    std::printf("[sdpa_mma] no matrix cores -- skipped.\n");
    return;
  }
  ComputeLibrary lib_q = mc->load_library("sdpa");
  ComputeFunction fq = lib_q.function("sdpa_paged_qtile_f16");

  const int D = 256, Hq = 16, Hkv = 4;
  struct Cfg { int n; const char* tag; };
  const Cfg cfgs[] = {{64, "n64"}, {512, "n512"}, {1024, "n1024"}};

  for (const auto& cf : cfgs) {
    const int n = cf.n;
    std::mt19937 rng(123 + n);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<std::uint16_t> q((size_t)Hq * n * D), k((size_t)Hkv * n * D),
        vv((size_t)Hkv * n * D);
    for (auto& e : q) e = f32_to_h(d(rng) * 0.2f);
    for (auto& e : k) e = f32_to_h(d(rng) * 0.2f);
    for (auto& e : vv) e = f32_to_h(d(rng) * 0.2f);
    const float scale = 1.0f / std::sqrt((float)D);

    SharedBuffer qb = mc->make_shared_buffer(q.size() * 2);
    SharedBuffer kb = mc->make_shared_buffer(k.size() * 2);
    SharedBuffer vb = mc->make_shared_buffer(vv.size() * 2);
    SharedBuffer ob = mc->make_shared_buffer((size_t)Hq * n * D * 2);
    std::memcpy(qb.contents(), q.data(), q.size() * 2);
    std::memcpy(kb.contents(), k.data(), k.size() * 2);
    std::memcpy(vb.contents(), vv.data(), vv.size() * 2);

    // Single-page paged setup: pool [1, Hkv, page_tokens=n, D] == our
    // contiguous [Hkv, n, D]; page_table {pid=0, nvalid=n, gstart=0}.
    std::vector<std::int32_t> pt = {0, n, 0};
    SharedBuffer ptb = mc->make_shared_buffer(pt.size() * 4);
    std::memcpy(ptb.contents(), pt.data(), pt.size() * 4);
    auto run_mma = [&]() {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
        enc.set_buffer(3, ob);
        enc.set_constant(4, scale); enc.set_constant(5, D);
        enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
        enc.set_constant(8, n); enc.set_constant(9, 0);
        enc.set_constant(10, n); enc.set_constant(11, 1);
        enc.set_buffer(12, ptb);
        const unsigned tiles = (unsigned)((n + 15) / 16);   // SA_BQ=16
        enc.dispatch({128, (unsigned)Hq, tiles}, {128, 1, 1});
      }
      st.commit().wait();
    };

    // Correctness vs CPU oracle on a few (h, qi) samples (full O is large).
    run_mma();
    const auto* op = static_cast<const std::uint16_t*>(ob.contents());
    double num = 0.0, den = 0.0;
    const int hs[] = {0, 7, 15};
    for (int h : hs) {
      const int kvh = h / (Hq / Hkv);
      for (int qi = 0; qi < n; qi += (n / 8 > 0 ? n / 8 : 1)) {
        std::vector<float> sc(qi + 1);
        float mx = -1e30f;
        for (int ki = 0; ki <= qi; ++ki) {
          float s = 0.0f;
          for (int dd = 0; dd < D; ++dd) {
            s += h_to_f32(q[((size_t)h * n + qi) * D + dd]) *
                 h_to_f32(k[((size_t)kvh * n + ki) * D + dd]);
          }
          sc[ki] = s * scale; mx = std::max(mx, sc[ki]);
        }
        float l = 0.0f; for (float& s : sc) { s = std::exp(s - mx); l += s; }
        for (int dd = 0; dd < D; ++dd) {
          float o = 0.0f;
          for (int ki = 0; ki <= qi; ++ki) {
            o += sc[ki] * h_to_f32(vv[((size_t)kvh * n + ki) * D + dd]);
          }
          o /= l;
          const float got = h_to_f32(op[((size_t)h * n + qi) * D + dd]);
          num += (got - o) * (got - o); den += o * o;
        }
      }
    }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);

    // Speed: matrix-core vs scalar qtile (single-page paged setup).
    auto bench = [&](auto launch) {
      for (int w = 0; w < 3; ++w) { launch(); }
      const auto t0 = std::chrono::steady_clock::now();
      const int it = 20;
      for (int i = 0; i < it; ++i) { launch(); }
      const auto t1 = std::chrono::steady_clock::now();
      return secs_(t0, t1) / it * 1e3;   // ms/call
    };
    const double ms_mma = bench(run_mma);

    double ms_q = 0.0;
    if (fq.valid()) {
      auto run_q = [&]() {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder enc = st.begin_compute();
          enc.set_function(fq);
          enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
          enc.set_buffer(3, ob);
          enc.set_constant(4, scale); enc.set_constant(5, D);
          enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
          enc.set_constant(8, n); enc.set_constant(9, 0);
          enc.set_constant(10, n); enc.set_constant(11, 1);
          enc.set_buffer(12, ptb);
          const unsigned tiles = (unsigned)((n + 15) / 16);
          enc.dispatch({512, (unsigned)Hq, tiles}, {512, 1, 1});
        }
        st.commit().wait();
      };
      ms_q = bench(run_q);
    }
    std::printf("[sdpa_mma] %-6s n=%4d  rel-L2=%.3e | mma %.3f ms | "
                "scalar-qtile %.3f ms | %.2fx\n",
                cf.tag, n, rel, ms_mma, ms_q, ms_q > 0 ? ms_q / ms_mma : 0.0);
    EXPECT_TRUE(rel < 5e-2);
  }
}

// ENCODER full (bidirectional) attention: sdpa_full_mma_f16 vs the scalar
// sdpa_full_f16, contiguous [H, n, D] Q/K/V, head_dim 64 (the ViT/Conformer
// shape). Correctness (rel-L2 vs scalar) + speed at the vision-tower seq
// length. This is the fix for the metal encoder perf gap (scalar 1-simdgroup-
// per-query was ~93% of the vision encode wall).
TEST(sdpa_mma, full_encoder_d64) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("sdpa");
  ComputeFunction f_sc = lib.function("sdpa_full_f16");
  ComputeFunction f_mma = lib.function("sdpa_full_mma_f16");
  if (!f_sc.valid() || !f_mma.valid()) {
    std::printf("[full_enc] missing kernels -- skipped\n"); return;
  }
  const int D = 64, H = 12;
  const float scale = 1.0f / std::sqrt((float)D);
  for (int n : {512, 2394}) {
    std::mt19937 rng(11 + n);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<std::uint16_t> q((size_t)H * n * D), k((size_t)H * n * D),
        vv((size_t)H * n * D);
    for (auto& e : q) e = f32_to_h(d(rng) * 0.3f);
    for (auto& e : k) e = f32_to_h(d(rng) * 0.3f);
    for (auto& e : vv) e = f32_to_h(d(rng) * 0.3f);
    SharedBuffer qb = mc->make_shared_buffer(q.size() * 2);
    SharedBuffer kb = mc->make_shared_buffer(k.size() * 2);
    SharedBuffer vb = mc->make_shared_buffer(vv.size() * 2);
    SharedBuffer o_sc = mc->make_shared_buffer((size_t)H * n * D * 2);
    SharedBuffer o_mma = mc->make_shared_buffer((size_t)H * n * D * 2);
    std::memcpy(qb.contents(), q.data(), q.size() * 2);
    std::memcpy(kb.contents(), k.data(), k.size() * 2);
    std::memcpy(vb.contents(), vv.data(), vv.size() * 2);

    auto run_sc = [&]() {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(f_sc);
        enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
        enc.set_buffer(3, o_sc);
        enc.set_constant(4, scale); enc.set_constant(5, n);
        enc.set_constant(6, D); enc.set_constant(7, H);
        enc.set_constant(8, H); enc.set_constant(9, n);
        enc.set_constant(10, n);
        enc.dispatch({32, (unsigned)H, (unsigned)n}, {32, 1, 1}); }
      st.commit().wait();
    };
    auto run_mma = [&]() {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(f_mma);
        enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
        enc.set_buffer(3, o_mma);
        enc.set_constant(4, scale); enc.set_constant(5, n);
        enc.set_constant(6, D); enc.set_constant(7, H);
        enc.set_constant(8, H); enc.set_constant(9, n);
        enc.set_constant(10, n);
        const unsigned tiles = (unsigned)((n + 31) / 32);  // MMAF_BQ=32
        enc.dispatch({128, (unsigned)H, tiles}, {128, 1, 1}); }  // WM*WD*32
      st.commit().wait();
    };
    run_sc(); run_mma();
    const auto* a = static_cast<const std::uint16_t*>(o_sc.contents());
    const auto* b = static_cast<const std::uint16_t*>(o_mma.contents());
    double num = 0, den = 0;
    for (size_t i = 0; i < (size_t)H * n * D; ++i) {
      const float x = h_to_f32(a[i]), y = h_to_f32(b[i]);
      num += (x - y) * (x - y); den += x * x;
    }
    const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
    auto bench = [&](auto fn) {
      for (int w = 0; w < 3; ++w) { fn(); }
      const auto t0 = std::chrono::steady_clock::now();
      const int it = 10;
      for (int i = 0; i < it; ++i) { fn(); }
      return secs_(t0, std::chrono::steady_clock::now()) / it * 1e3;
    };
    const double ms_sc = bench(run_sc), ms_mma = bench(run_mma);
    std::printf("[full_enc] n=%4d D=%d H=%d | rel-L2=%.3e | scalar %.3f ms | "
                "mma %.3f ms | %.2fx\n", n, D, H, rel, ms_sc, ms_mma,
                ms_mma > 0 ? ms_sc / ms_mma : 0.0);
    EXPECT_TRUE(rel < 2e-2);
  }
}

// Gemma-4 GLOBAL-layer prefill attention: sdpa_causal_mma_f16 over a
// CONTIGUOUS [Hkv, T_kv, D] K/V, head_dim 512, full causal, Hkv=1 (unified
// K/V). This is the O(n^2) path that dominates long-context prefill. Bench
// ms/call at n in {1024,2048,4096} + correctness (rel-L2 vs CPU oracle on
// sampled rows). The iteration harness for optimizing this kernel.
TEST(sdpa_mma, gemma_global_d512) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("sdpa");
  ComputeFunction fn = lib.function("sdpa_causal_mma_f16");
  if (!fn.valid()) { std::printf("[sdpa_global] no sdpa_causal_mma\n"); return; }

  ComputeFunction fn_dev = lib.function("sdpa_causal_mma_dev_f16");
  ComputeFunction fn_flash = lib.function("sdpa_causal_flash_f16");
  // Matrix-core (matmul2d) candidate: QK^T/PV on the M5 matrix units vs the
  // simdgroup_matrix (ALU) flash kernel above.
  ComputeLibrary lib_mma = mc->load_library("sdpa_mma");
  ComputeFunction fn_mma2 = lib_mma.function("sdpa_causal_mma2_d512_f16");
  const int D = 512, Hq = 16, Hkv = 1;
  struct Cfg { int n, it; const char* tag; };
  const Cfg cfgs[] = {{1024, 20, "n1024"}, {2048, 10, "n2048"},
                      {4096, 5, "n4096"}};

  for (const auto& cf : cfgs) {
    const int n = cf.n;
    std::mt19937 rng(7 + n);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    // K/V padded by 32 rows (zeroed): the device-direct kernel over-reads the
    // final key block past T_kv; the slack keeps it inside the allocation.
    std::vector<std::uint16_t> q((size_t)Hq * n * D),
        k((size_t)(n + 32) * D, 0), vv((size_t)(n + 32) * D, 0);
    for (auto& e : q) { e = f32_to_h(d(rng) * 0.2f); }
    for (size_t i = 0; i < (size_t)n * D; ++i) { k[i] = f32_to_h(d(rng) * 0.2f); }
    for (size_t i = 0; i < (size_t)n * D; ++i) { vv[i] = f32_to_h(d(rng) * 0.2f); }
    const float scale = 1.0f / std::sqrt((float)D);

    SharedBuffer qb = mc->make_shared_buffer(q.size() * 2);
    SharedBuffer kb = mc->make_shared_buffer(k.size() * 2);
    SharedBuffer vb = mc->make_shared_buffer(vv.size() * 2);
    // out padded by MMAF_BQ(32) rows: the kernel over-reads the last Q tile.
    SharedBuffer obA = mc->make_shared_buffer((size_t)Hq * (n + 32) * D * 2);
    SharedBuffer obB = mc->make_shared_buffer((size_t)Hq * (n + 32) * D * 2);
    SharedBuffer obC = mc->make_shared_buffer((size_t)Hq * (n + 32) * D * 2);
    SharedBuffer obD = mc->make_shared_buffer((size_t)Hq * (n + 32) * D * 2);
    std::memcpy(qb.contents(), q.data(), q.size() * 2);
    std::memcpy(kb.contents(), k.data(), k.size() * 2);
    std::memcpy(vb.contents(), vv.data(), vv.size() * 2);

    auto run = [&](ComputeFunction& f, SharedBuffer& ob) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(f);
        enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
        enc.set_buffer(3, ob);
        enc.set_constant(4, scale); enc.set_constant(5, n);     // T_kv
        enc.set_constant(6, D); enc.set_constant(7, Hq);
        enc.set_constant(8, Hkv); enc.set_constant(9, n);       // n_q
        enc.set_constant(10, 0); enc.set_constant(11, n);       // q_off,kv_str
        enc.set_constant(12, 0); enc.set_constant(13, 0);       // window,ring
        const unsigned tg = (unsigned)(4 * (D / 64) * 32);      // WM*WD*32
        enc.dispatch({tg, (unsigned)Hq, (unsigned)((n + 31) / 32)},
                     {tg, 1, 1});
      }
      st.commit().wait();
    };

    // rel-L2 of a kernel's output vs a CPU oracle on sampled (h,qi,col).
    auto rel_vs_oracle = [&](SharedBuffer& ob) {
      const auto* op = static_cast<const std::uint16_t*>(ob.contents());
      double num = 0.0, den = 0.0;
      const int hs[] = {0, 15};
      for (int h : hs) {
        for (int qi = 0; qi < n; qi += (n / 6 > 0 ? n / 6 : 1)) {
          std::vector<float> sc(qi + 1);
          float mx = -1e30f;
          for (int ki = 0; ki <= qi; ++ki) {
            float s = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
              s += h_to_f32(q[((size_t)h * n + qi) * D + dd]) *
                   h_to_f32(k[(size_t)ki * D + dd]);   // Hkv=1 -> kvh=0
            }
            sc[ki] = s * scale; mx = std::max(mx, sc[ki]);
          }
          float l = 0.0f; for (float& s : sc) { s = std::exp(s - mx); l += s; }
          for (int dd = 0; dd < D; dd += 37) {     // sample cols
            float o = 0.0f;
            for (int ki = 0; ki <= qi; ++ki) {
              o += sc[ki] * h_to_f32(vv[(size_t)ki * D + dd]);
            }
            o /= l;
            const float got = h_to_f32(op[((size_t)h * n + qi) * D + dd]);
            num += (got - o) * (got - o); den += o * o;
          }
        }
      }
      return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    };

    auto bench = [&](ComputeFunction& f, SharedBuffer& ob) {
      for (int w = 0; w < 3; ++w) { run(f, ob); }
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < cf.it; ++i) { run(f, ob); }
      const auto t1 = std::chrono::steady_clock::now();
      return secs_(t0, t1) / cf.it * 1e3;
    };
    // Flash kernel: Q=8 rows/tg, NSG=8 simdgroups (256 threads), grid n/8.
    auto run_flash = [&](SharedBuffer& ob) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn_flash);
        enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
        enc.set_buffer(3, ob);
        enc.set_constant(4, scale); enc.set_constant(5, n);
        enc.set_constant(6, D); enc.set_constant(7, Hq);
        enc.set_constant(8, Hkv); enc.set_constant(9, n);
        enc.set_constant(10, 0); enc.set_constant(11, n);
        enc.set_constant(12, 0); enc.set_constant(13, 0);
        enc.dispatch({256, (unsigned)Hq, (unsigned)((n + 7) / 8)},
                     {256, 1, 1});
      }
      st.commit().wait();
    };
    auto bench_flash = [&](SharedBuffer& ob) {
      for (int w = 0; w < 3; ++w) { run_flash(ob); }
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < cf.it; ++i) { run_flash(ob); }
      const auto t1 = std::chrono::steady_clock::now();
      return secs_(t0, t1) / cf.it * 1e3;
    };
    // matmul2d candidate: BQ=8 rows/tg, SA2_SG=4 simdgroups (128 threads),
    // grid ceil(n/8). Same contiguous-KV buffer contract as run().
    auto run_mma2 = [&](SharedBuffer& ob) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn_mma2);
        enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
        enc.set_buffer(3, ob);
        enc.set_constant(4, scale); enc.set_constant(5, n);
        enc.set_constant(6, D); enc.set_constant(7, Hq);
        enc.set_constant(8, Hkv); enc.set_constant(9, n);
        enc.set_constant(10, 0); enc.set_constant(11, n);
        enc.set_constant(12, 0); enc.set_constant(13, 0);
        enc.dispatch({128, (unsigned)Hq, (unsigned)((n + 7) / 8)},
                     {128, 1, 1});
      }
      st.commit().wait();
    };
    auto bench_mma2 = [&](SharedBuffer& ob) {
      for (int w = 0; w < 3; ++w) { run_mma2(ob); }
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < cf.it; ++i) { run_mma2(ob); }
      const auto t1 = std::chrono::steady_clock::now();
      return secs_(t0, t1) / cf.it * 1e3;
    };

    run(fn, obA);
    const double relA = rel_vs_oracle(obA);
    const double msA = bench(fn, obA);

    double relC = 0.0, msC = 0.0;
    if (fn_flash.valid()) {
      run_flash(obC);
      relC = rel_vs_oracle(obC);
      msC = bench_flash(obC);
    }

    double relB = 0.0, msB = 0.0, relBA = 0.0;
    if (fn_dev.valid()) {
      run(fn_dev, obB);
      relB = rel_vs_oracle(obB);
      // B vs A (should be ~identical; same math, device-direct vs staged).
      const auto* pa = static_cast<const std::uint16_t*>(obA.contents());
      const auto* pb = static_cast<const std::uint16_t*>(obB.contents());
      double num = 0.0, den = 0.0;
      for (size_t i = 0; i < (size_t)Hq * n * D; i += 101) {
        const float a = h_to_f32(pa[i]), b = h_to_f32(pb[i]);
        num += (a - b) * (a - b); den += a * a;
      }
      relBA = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
      msB = bench(fn_dev, obB);
    }
    double relD = 0.0, msD = 0.0;
    if (fn_mma2.valid()) {
      run_mma2(obD);
      relD = rel_vs_oracle(obD);
      msD = bench_mma2(obD);
    }
    std::printf("[sdpa_global] %-6s n=%4d | A(staged) rel=%.2e %.3f ms | "
                "B(dev) rel=%.2e %.3f ms | C(flash) rel=%.2e %.3f ms | "
                "D(mma2) rel=%.2e %.3f ms | C-speedup %.2fx | "
                "D-vs-C %.2fx\n",
                cf.tag, n, relA, msA, relB, msB, relC, msC, relD, msD,
                msC > 0 ? msA / msC : 0.0, msD > 0 ? msC / msD : 0.0);
    EXPECT_TRUE(relA < 5e-2);
    if (fn_dev.valid()) {
      EXPECT_TRUE(relB < 5e-2);
      EXPECT_TRUE(relBA < 5e-2);
    }
    if (fn_flash.valid()) { EXPECT_TRUE(relC < 5e-2); }
    if (fn_mma2.valid()) { EXPECT_TRUE(relD < 5e-2); }
  }
}

// Gemma-4 SLIDING-layer prefill attention in the NO-WRAP regime:
// sdpa_causal_flash_f16 over a CONTIGUOUS [Hkv, cap, D] K/V at head_dim 256
// with a trailing window (1024) -- the path the dispatch now routes sliding
// layers to before the ring wraps. Validates the window mask + the window-base
// scan skip against a CPU oracle (and the staged kernel), exercises the
// FL_C(64) over-read into the zeroed cache tail (n not a multiple of 64), and
// benches flash vs staged. window mask: key ki contributes to query qi only
// when max(0, qi-window+1) <= ki <= qi.
TEST(sdpa_mma, gemma_sliding_d256_flash) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("sdpa");
  ComputeFunction fn = lib.function("sdpa_causal_mma_f16");
  ComputeFunction fn_flash = lib.function("sdpa_causal_flash_f16");
  if (!fn.valid() || !fn_flash.valid()) {
    std::printf("[sdpa_sliding] kernels unavailable -- skipped\n");
    return;
  }
  ComputeLibrary lib_mma = mc->load_library("sdpa_mma");
  ComputeFunction fn_mma2 = lib_mma.function("sdpa_causal_mma2_d256_f16");
  const int D = 256, Hq = 16, Hkv = 1, window = 1024;
  struct Cfg { int n, it; const char* tag; };
  const Cfg cfgs[] = {{1536, 12, "n1536"}, {2040, 8, "n2040"},
                      {3000, 6, "n3000"}};

  for (const auto& cf : cfgs) {
    const int n = cf.n;
    std::mt19937 rng(31 + n);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    // No-wrap single forward: cap >= n + 64 (the over-read slack), tail zeroed.
    const int cap = n + 64;
    std::vector<std::uint16_t> q((size_t)Hq * n * D),
        k((size_t)cap * D, 0), vv((size_t)cap * D, 0);
    for (auto& e : q) { e = f32_to_h(d(rng) * 0.2f); }
    for (size_t i = 0; i < (size_t)n * D; ++i) { k[i] = f32_to_h(d(rng) * 0.2f); }
    for (size_t i = 0; i < (size_t)n * D; ++i) { vv[i] = f32_to_h(d(rng) * 0.2f); }
    const float scale = 1.0f / std::sqrt((float)D);

    SharedBuffer qb = mc->make_shared_buffer(q.size() * 2);
    SharedBuffer kb = mc->make_shared_buffer(k.size() * 2);
    SharedBuffer vb = mc->make_shared_buffer(vv.size() * 2);
    SharedBuffer obA = mc->make_shared_buffer((size_t)Hq * (n + 32) * D * 2);
    SharedBuffer obC = mc->make_shared_buffer((size_t)Hq * (n + 32) * D * 2);
    SharedBuffer obD = mc->make_shared_buffer((size_t)Hq * (n + 32) * D * 2);
    std::memcpy(qb.contents(), q.data(), q.size() * 2);
    std::memcpy(kb.contents(), k.data(), k.size() * 2);
    std::memcpy(vb.contents(), vv.data(), vv.size() * 2);

    auto run_staged = [&](SharedBuffer& ob) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn);
        enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
        enc.set_buffer(3, ob);
        enc.set_constant(4, scale); enc.set_constant(5, n);
        enc.set_constant(6, D); enc.set_constant(7, Hq);
        enc.set_constant(8, Hkv); enc.set_constant(9, n);
        enc.set_constant(10, 0); enc.set_constant(11, cap);    // kv_stride
        enc.set_constant(12, window); enc.set_constant(13, 0); // no wrap
        const unsigned tg = (unsigned)(4 * (D / 64) * 32);
        enc.dispatch({tg, (unsigned)Hq, (unsigned)((n + 31) / 32)},
                     {tg, 1, 1});
      }
      st.commit().wait();
    };
    auto run_flash = [&](SharedBuffer& ob) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn_flash);
        enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
        enc.set_buffer(3, ob);
        enc.set_constant(4, scale); enc.set_constant(5, n);
        enc.set_constant(6, D); enc.set_constant(7, Hq);
        enc.set_constant(8, Hkv); enc.set_constant(9, n);
        enc.set_constant(10, 0); enc.set_constant(11, cap);
        enc.set_constant(12, window); enc.set_constant(13, 0);
        enc.dispatch({256, (unsigned)Hq, (unsigned)((n + 7) / 8)},
                     {256, 1, 1});
      }
      st.commit().wait();
    };

    auto rel_vs_oracle = [&](SharedBuffer& ob) {
      const auto* op = static_cast<const std::uint16_t*>(ob.contents());
      double num = 0.0, den = 0.0;
      const int hs[] = {0, 15};
      for (int h : hs) {
        for (int qi = 0; qi < n; qi += (n / 6 > 0 ? n / 6 : 1)) {
          const int first = std::max(0, qi - window + 1);
          std::vector<float> sc((size_t)qi + 1, 0.0f);
          float mx = -1e30f;
          for (int ki = first; ki <= qi; ++ki) {
            float s = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
              s += h_to_f32(q[((size_t)h * n + qi) * D + dd]) *
                   h_to_f32(k[(size_t)ki * D + dd]);   // Hkv=1 -> kvh=0
            }
            sc[ki] = s * scale; mx = std::max(mx, sc[ki]);
          }
          float l = 0.0f;
          for (int ki = first; ki <= qi; ++ki) {
            sc[ki] = std::exp(sc[ki] - mx); l += sc[ki];
          }
          for (int dd = 0; dd < D; dd += 37) {
            float o = 0.0f;
            for (int ki = first; ki <= qi; ++ki) {
              o += sc[ki] * h_to_f32(vv[(size_t)ki * D + dd]);
            }
            o /= l;
            const float got = h_to_f32(op[((size_t)h * n + qi) * D + dd]);
            num += (got - o) * (got - o); den += o * o;
          }
        }
      }
      return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    };

    auto bench = [&](auto launch, SharedBuffer& ob) {
      for (int w = 0; w < 3; ++w) { launch(ob); }
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < cf.it; ++i) { launch(ob); }
      const auto t1 = std::chrono::steady_clock::now();
      return secs_(t0, t1) / cf.it * 1e3;
    };

    auto run_mma2 = [&](SharedBuffer& ob) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(fn_mma2);
        enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
        enc.set_buffer(3, ob);
        enc.set_constant(4, scale); enc.set_constant(5, n);
        enc.set_constant(6, D); enc.set_constant(7, Hq);
        enc.set_constant(8, Hkv); enc.set_constant(9, n);
        enc.set_constant(10, 0); enc.set_constant(11, cap);
        enc.set_constant(12, window); enc.set_constant(13, 0);
        enc.dispatch({128, (unsigned)Hq, (unsigned)((n + 15) / 16)},
                     {128, 1, 1});
      }
      st.commit().wait();
    };

    run_staged(obA);
    const double relA = rel_vs_oracle(obA);
    const double msA = bench(run_staged, obA);
    run_flash(obC);
    const double relC = rel_vs_oracle(obC);
    const double msC = bench(run_flash, obC);
    double relD = 0.0, msD = 0.0;
    if (fn_mma2.valid()) {
      run_mma2(obD);
      relD = rel_vs_oracle(obD);
      msD = bench(run_mma2, obD);
    }
    std::printf("[sdpa_sliding] %-6s n=%4d w=%d | staged rel=%.2e %.3f ms | "
                "flash rel=%.2e %.3f ms | mma2 rel=%.2e %.3f ms (%.2fx C) | "
                "%.2fx\n",
                cf.tag, n, window, relA, msA, relC, msC, relD, msD,
                msD > 0 ? msC / msD : 0.0,
                msC > 0 ? msA / msC : 0.0);
    EXPECT_TRUE(relA < 5e-2);
    EXPECT_TRUE(relC < 5e-2);
    if (fn_mma2.valid()) { EXPECT_TRUE(relD < 5e-2); }
  }
}

// Qwen3.5 full-attention PREFILL: sdpa_paged_flash_f16 over a PAGED K/V pool
// [n_pages, Hkv, page_tokens, D] at head_dim 256, GQA, causal. Multi-page +
// a partial last page exercise the page loop AND the C-block over-read into
// the zeroed page tail (masked by key<bk). Validates rel-L2 vs a CPU oracle
// and benches flash vs the scalar query-tiled kernel (sdpa_paged_qtile).
TEST(sdpa_mma, qwen_paged_flash_d256) {
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("sdpa");
  ComputeFunction fn = lib.function("sdpa_paged_flash_f16");
  ComputeFunction fq = lib.function("sdpa_paged_qtile_f16");
  if (!fn.valid()) { std::printf("[qwen_flash] no kernel -- skipped\n"); return; }

  const int D = 256, Hq = 16, Hkv = 4, page_tokens = 192;
  struct Cfg { int n; const char* tag; };
  // n=300 -> 2 pages (192 + 108); 108 spans a full 64-block + a 44 partial.
  // n=700 -> 4 pages (192*3 + 124). Both have a partial final page.
  const Cfg cfgs[] = {{300, "n300"}, {700, "n700"}};

  for (const auto& cf : cfgs) {
    const int n = cf.n;
    const int n_pages = (n + page_tokens - 1) / page_tokens;
    std::mt19937 rng(53 + n);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    // Flat K/V by GLOBAL position [n, Hkv, D] for the oracle; the pool is the
    // same data scattered into [n_pages, Hkv, page_tokens, D] (zeroed tail).
    std::vector<std::uint16_t> q((size_t)Hq * n * D);
    std::vector<std::uint16_t> kfull((size_t)n * Hkv * D),
        vfull((size_t)n * Hkv * D);
    for (auto& e : q) { e = f32_to_h(d(rng) * 0.2f); }
    for (auto& e : kfull) { e = f32_to_h(d(rng) * 0.2f); }
    for (auto& e : vfull) { e = f32_to_h(d(rng) * 0.2f); }
    const float scale = 1.0f / std::sqrt((float)D);

    const size_t pool_elts = (size_t)n_pages * Hkv * page_tokens * D;
    std::vector<std::uint16_t> kpool(pool_elts, 0), vpool(pool_elts, 0);
    std::vector<std::int32_t> pt;            // {pid, nvalid, gstart} per page
    for (int p = 0; p < n_pages; ++p) {
      const int gstart = p * page_tokens;
      const int nvalid = std::min(page_tokens, n - gstart);
      pt.push_back(p); pt.push_back(nvalid); pt.push_back(gstart);
      for (int s = 0; s < nvalid; ++s) {
        const int g = gstart + s;
        for (int kv = 0; kv < Hkv; ++kv) {
          const size_t dst = ((size_t)(p * Hkv + kv) * page_tokens + s) * D;
          const size_t src = ((size_t)g * Hkv + kv) * D;
          for (int dd = 0; dd < D; ++dd) {
            kpool[dst + dd] = kfull[src + dd];
            vpool[dst + dd] = vfull[src + dd];
          }
        }
      }
    }

    SharedBuffer qb = mc->make_shared_buffer(q.size() * 2);
    SharedBuffer kb = mc->make_shared_buffer(kpool.size() * 2);
    SharedBuffer vb = mc->make_shared_buffer(vpool.size() * 2);
    SharedBuffer ob = mc->make_shared_buffer((size_t)Hq * n * D * 2);
    SharedBuffer ptb = mc->make_shared_buffer(pt.size() * 4);
    std::memcpy(qb.contents(), q.data(), q.size() * 2);
    std::memcpy(kb.contents(), kpool.data(), kpool.size() * 2);
    std::memcpy(vb.contents(), vpool.data(), vpool.size() * 2);
    std::memcpy(ptb.contents(), pt.data(), pt.size() * 4);

    auto run = [&](ComputeFunction& f, bool flash) {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder enc = st.begin_compute();
        enc.set_function(f);
        enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
        enc.set_buffer(3, ob);
        enc.set_constant(4, scale); enc.set_constant(5, D);
        enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
        enc.set_constant(8, n); enc.set_constant(9, 0);       // q_offset
        enc.set_constant(10, page_tokens); enc.set_constant(11, n_pages);
        enc.set_buffer(12, ptb);
        if (flash) {
          enc.dispatch({256, (unsigned)Hq, (unsigned)((n + 7) / 8)},
                       {256, 1, 1});
        } else {
          enc.dispatch({512, (unsigned)Hq, (unsigned)((n + 15) / 16)},
                       {512, 1, 1});
        }
      }
      st.commit().wait();
    };

    run(fn, true);
    const auto* op = static_cast<const std::uint16_t*>(ob.contents());
    double num = 0.0, den = 0.0;
    const int hs[] = {0, 7, 15};
    for (int h : hs) {
      const int kvh = h / (Hq / Hkv);
      for (int qi = 0; qi < n; qi += (n / 8 > 0 ? n / 8 : 1)) {
        std::vector<float> sc((size_t)qi + 1);
        float mx = -1e30f;
        for (int ki = 0; ki <= qi; ++ki) {
          float s = 0.0f;
          for (int dd = 0; dd < D; ++dd) {
            s += h_to_f32(q[((size_t)h * n + qi) * D + dd]) *
                 h_to_f32(kfull[((size_t)ki * Hkv + kvh) * D + dd]);
          }
          sc[ki] = s * scale; mx = std::max(mx, sc[ki]);
        }
        float l = 0.0f; for (float& s : sc) { s = std::exp(s - mx); l += s; }
        for (int dd = 0; dd < D; dd += 31) {
          float o = 0.0f;
          for (int ki = 0; ki <= qi; ++ki) {
            o += sc[ki] * h_to_f32(vfull[((size_t)ki * Hkv + kvh) * D + dd]);
          }
          o /= l;
          const float got = h_to_f32(op[((size_t)h * n + qi) * D + dd]);
          num += (got - o) * (got - o); den += o * o;
        }
      }
    }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);

    auto bench = [&](ComputeFunction& f, bool flash) {
      for (int w = 0; w < 3; ++w) { run(f, flash); }
      const auto t0 = std::chrono::steady_clock::now();
      const int it = 20;
      for (int i = 0; i < it; ++i) { run(f, flash); }
      const auto t1 = std::chrono::steady_clock::now();
      return secs_(t0, t1) / it * 1e3;
    };
    const double ms_f = bench(fn, true);
    const double ms_q = fq.valid() ? bench(fq, false) : 0.0;
    std::printf("[qwen_flash] %-5s n=%4d pages=%d | rel-L2=%.3e | flash %.3f ms"
                " | qtile %.3f ms | %.2fx\n",
                cf.tag, n, n_pages, rel, ms_f, ms_q,
                ms_q > 0 ? ms_q / ms_f : 0.0);
    EXPECT_TRUE(rel < 5e-2);
  }
}

// Per-dispatch overhead probe: the Gemma-4 decode gap vs omlx is a context-
// INDEPENDENT fixed offset (~4ms), concentrated in the small non-GEMV ops
// (norm/ple/sliding) whose per-token cost is launch + inter-dispatch drain,
// not compute. This times a tiny rms_norm (H=2560, the decode shape, 1
// threadgroup) dispatched R times in ONE command buffer: DEPENDENT (all write
// the same out -> WAW hazard serialises them, exactly like the decode op
// chain) vs INDEPENDENT (round-robin over many out buffers -> Metal can
// overlap -> isolates pure launch cost). The dependent number is the real
// per-op floor; (dependent - independent) is the drain. Gated on
// VPIPE_DISP_OVH.
TEST(sdpa_mma, dispatch_overhead) {
  if (std::getenv("VPIPE_DISP_OVH") == nullptr) { return; }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("rms_norm");
  ComputeFunction rms = lib.function("rms_norm_f16");
  if (!rms.valid()) { return; }
  const int H = 2560;
  const int NB = 128;          // independent output buffers
  SharedBuffer xb = mc->make_shared_buffer((size_t)H * 2);
  SharedBuffer wb = mc->make_shared_buffer((size_t)H * 2);
  std::memset(xb.contents(), 0, (size_t)H * 2);
  std::memset(wb.contents(), 0, (size_t)H * 2);
  std::vector<SharedBuffer> outs;
  for (int i = 0; i < NB; ++i) {
    outs.push_back(mc->make_shared_buffer((size_t)H * 2));
  }
  const float eps = 1e-6f;
  const int R = 1000;
  auto bench = [&](bool dependent) {
    auto run = [&]() {
      CommandStream st = mc->make_command_stream();
      { ComputeEncoder e = st.begin_compute();
        for (int r = 0; r < R; ++r) {
          e.set_function(rms);
          e.set_buffer(0, xb); e.set_buffer(1, wb);
          e.set_buffer(2, dependent ? outs[0] : outs[r % NB]);
          e.set_constant(3, H); e.set_constant(4, eps);
          e.dispatch({256, 1, 1}, {256, 1, 1});
        }
      }
      st.commit().wait();
    };
    run(); run();
    double best = 1e18;
    for (int rep = 0; rep < 3; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      run();
      best = std::min(best, secs_(t0, std::chrono::steady_clock::now()));
    }
    return best / R * 1e6;   // us/dispatch
  };
  const double dep = bench(true);
  const double indep = bench(false);
  std::printf("[disp-ovh] rms H=%d: dependent %.2f us/dispatch | "
              "independent %.2f us/dispatch | drain %.2f us\n",
              H, dep, indep, dep - indep);
}

// Microscopic decode-attention probe for the Gemma-4 GLOBAL layers (head_dim
// 512, full-context, the decode bottleneck). Synthetic single-query decode at
// growing context -- NO model load, so kernel variants A/B in seconds. Times
// kernel+merge with reps batched in one command buffer (hazard-serialised,
// like the real per-token decode) and checks rel-L2 vs a CPU oracle. Compares
// the staged gtile against the direct-read (no staging, fast::exp) kernel and
// sweeps the split count, for both the global (D512) and sliding (D256/win512)
// layer regimes.
// Gated on VPIPE_SDPA_MICRO (a profiling probe, off in normal suites).
TEST(sdpa_mma, gqa_decode_micro) {
  if (std::getenv("VPIPE_SDPA_MICRO") == nullptr) { return; }
  Session sess;
  auto* mc = get_mc_(sess);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("sdpa");
  ComputeFunction f_mb    = lib.function("sdpa_causal_mb_f16");
  ComputeFunction f_tile  = lib.function("sdpa_causal_gqa_tile_f16");
  ComputeFunction f_dir   = lib.function("sdpa_causal_gqa_direct_f16");
  ComputeFunction f_vec   = lib.function("sdpa_causal_gqa_vec_f16");
  ComputeFunction f_allg  = lib.function("sdpa_causal_gqa_f16");
  ComputeFunction f_pgqa  = lib.function("sdpa_paged_gqa_mb256_f16");
  ComputeFunction f_pvec  = lib.function("sdpa_paged_gqa_vec_f16");
  ComputeFunction f_mma   = lib.function("sdpa_causal_mma_qhead_f16");
  ComputeFunction f_merge = lib.function("sdpa_gqa_merge_f16");
  if (!f_mb.valid() || !f_tile.valid() || !f_merge.valid()) {
    std::printf("[gqa-micro] sdpa lib missing functions -- skipped.\n");
    return;
  }

  const int Smax = 256;
  const int reps = 40;

  // Three regimes: e4b GLOBAL (head_dim 512, full causal, Hkv=2/G=4), e4b
  // SLIDING (head_dim 256, window 512), and the 12B gemma4_unified GLOBAL
  // (head_dim 512, Hkv=1/G=16 -- the lone-KV-head over-read case).
  struct Cfg { int D; int win; int Hq; int Hkv; const char* rt; };
  const Cfg cfgs[] = {{512, 0, 8, 2, "global   "},
                      {256, 512, 8, 2, "slide    "},
                      {256, 0, 16, 4, "qwen(G4) "},
                      {512, 0, 16, 1, "g12b(G16)"}};
  for (const auto& cfg : cfgs) {
  const int D = cfg.D;
  const int win = cfg.win;
  const int Hq = cfg.Hq, Hkv = cfg.Hkv, G = Hq / Hkv;
  const float scale = 1.0f / std::sqrt((float)D);
  for (int T : {1024, 2048, 4096}) {
    std::mt19937 rng(7 + T);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<std::uint16_t> q((size_t)Hq * D), k((size_t)Hkv * T * D),
        vv((size_t)Hkv * T * D);
    for (auto& e : q) e = f32_to_h(d(rng) * 0.2f);
    for (auto& e : k) e = f32_to_h(d(rng) * 0.2f);
    for (auto& e : vv) e = f32_to_h(d(rng) * 0.2f);

    SharedBuffer qb = mc->make_shared_buffer(q.size() * 2);
    SharedBuffer kb = mc->make_shared_buffer(k.size() * 2);
    SharedBuffer vb = mc->make_shared_buffer(vv.size() * 2);
    SharedBuffer ob = mc->make_shared_buffer((size_t)Hq * D * 2);
    SharedBuffer oacc = mc->make_shared_buffer((size_t)Hq * Smax * D * 4);
    SharedBuffer mbuf = mc->make_shared_buffer((size_t)Hq * Smax * 4);
    SharedBuffer lbuf = mc->make_shared_buffer((size_t)Hq * Smax * 4);
    std::memcpy(qb.contents(), q.data(), q.size() * 2);
    std::memcpy(kb.contents(), k.data(), k.size() * 2);
    std::memcpy(vb.contents(), vv.data(), vv.size() * 2);

    // Single-page paged view: pool [1, Hkv, page_tokens=T, D] == contiguous
    // [Hkv, T, D] == kb/vb byte-for-byte; page_table {pid=0, nvalid=T, gs=0}.
    std::vector<std::int32_t> pt = {0, T, 0};
    SharedBuffer ptb = mc->make_shared_buffer(pt.size() * 4);
    std::memcpy(ptb.contents(), pt.data(), pt.size() * 4);

    // CPU oracle for two heads: decode q at T-1 attends [first, T-1], where
    // first = T-win for a sliding layer (win>0) else 0 (full causal).
    auto oracle = [&](int h, std::vector<float>& out) {
      const int kvh = h / G;
      const int q_pos = T - 1;
      const int first = (win > 0) ? std::max(0, q_pos - win + 1) : 0;
      std::vector<float> sc(T, 0.0f);
      float mx = -1e30f;
      for (int ki = first; ki <= q_pos; ++ki) {
        float s = 0.0f;
        for (int dd = 0; dd < D; ++dd) {
          s += h_to_f32(q[(size_t)h * D + dd]) *
               h_to_f32(k[((size_t)kvh * T + ki) * D + dd]);
        }
        sc[ki] = s * scale; mx = std::max(mx, sc[ki]);
      }
      float l = 0.0f;
      for (int ki = first; ki <= q_pos; ++ki) {
        sc[ki] = std::exp(sc[ki] - mx); l += sc[ki];
      }
      out.assign(D, 0.0f);
      for (int dd = 0; dd < D; ++dd) {
        float o = 0.0f;
        for (int ki = first; ki <= q_pos; ++ki) {
          o += sc[ki] * h_to_f32(vv[((size_t)kvh * T + ki) * D + dd]);
        }
        out[dd] = o / l;
      }
    };
    std::vector<float> ora0, ora7; oracle(0, ora0); oracle(Hq - 1, ora7);
    auto rel_of = [&]() {
      const auto* op = static_cast<const std::uint16_t*>(ob.contents());
      double num = 0, den = 0;
      for (int dd = 0; dd < D; ++dd) {
        const float g0 = h_to_f32(op[dd]) - ora0[dd];
        const float g7 = h_to_f32(op[(size_t)(Hq - 1) * D + dd]) - ora7[dd];
        num += g0 * g0 + g7 * g7;
        den += ora0[dd] * ora0[dd] + ora7[dd] * ora7[dd];
      }
      return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    };

    auto enc_gqa = [&](ComputeEncoder& enc, ComputeFunction& fn, int S) {
      enc.set_function(fn);
      enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
      enc.set_buffer(3, oacc); enc.set_buffer(4, mbuf); enc.set_buffer(5, lbuf);
      enc.set_constant(6, scale); enc.set_constant(7, T);
      enc.set_constant(8, D); enc.set_constant(9, Hq);
      enc.set_constant(10, Hkv);
      enc.set_constant(11, T - 1); enc.set_constant(12, T);
      enc.set_constant(13, win); enc.set_constant(14, 0);
      enc.set_constant(15, S);
      enc.dispatch({32, (unsigned)Hq, (unsigned)S},
                   {32, (unsigned)G, 1});
      enc.set_function(f_merge);
      enc.set_buffer(0, oacc); enc.set_buffer(1, mbuf); enc.set_buffer(2, lbuf);
      enc.set_buffer(3, ob);
      enc.set_constant(4, D); enc.set_constant(5, S); enc.set_constant(6, Hq);
      enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
    };
    // All-G single-simdgroup flash-decode (sdpa_causal_gqa_f16 -- the
    // contiguous analogue of Qwen's PRODUCTION sdpa_paged_gqa_mb256_f16):
    // each (kv, split) simdgroup reads its KV slice ONCE, applies every key
    // to all G heads. grid (32, Hkv, S), tg (32,1,1).
    auto enc_allg = [&](ComputeEncoder& enc, int S) {
      enc.set_function(f_allg);
      enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
      enc.set_buffer(3, oacc); enc.set_buffer(4, mbuf); enc.set_buffer(5, lbuf);
      enc.set_constant(6, scale); enc.set_constant(7, T);
      enc.set_constant(8, D); enc.set_constant(9, Hq);
      enc.set_constant(10, Hkv);
      enc.set_constant(11, T - 1); enc.set_constant(12, T);
      enc.set_constant(13, win); enc.set_constant(14, 0);
      enc.set_constant(15, S);
      enc.dispatch({32, (unsigned)Hkv, (unsigned)S}, {32, 1, 1});
      enc.set_function(f_merge);
      enc.set_buffer(0, oacc); enc.set_buffer(1, mbuf); enc.set_buffer(2, lbuf);
      enc.set_buffer(3, ob);
      enc.set_constant(4, D); enc.set_constant(5, S); enc.set_constant(6, Hq);
      enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
    };
    // PAGED kernels (single-page view): the production Qwen decode path.
    // Buffer contract: 0:q 1:kpool 2:vpool 3:o_acc 4:m 5:l 6:scale 7:D 8:Hq
    // 9:Hkv 10:q_offset 11:page_tokens 12:n_pages 13:page_table 14:split.
    auto enc_paged = [&](ComputeEncoder& enc, ComputeFunction& fn, int S,
                         bool per_head) {
      enc.set_function(fn);
      enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
      enc.set_buffer(3, oacc); enc.set_buffer(4, mbuf); enc.set_buffer(5, lbuf);
      enc.set_constant(6, scale); enc.set_constant(7, D);
      enc.set_constant(8, Hq); enc.set_constant(9, Hkv);
      enc.set_constant(10, T - 1); enc.set_constant(11, T);
      const int np = 1; enc.set_constant(12, np);
      enc.set_buffer(13, ptb); enc.set_constant(14, S);
      if (per_head) {
        enc.dispatch({32, (unsigned)Hq, (unsigned)S}, {32, (unsigned)G, 1});
      } else {
        enc.dispatch({32, (unsigned)Hkv, (unsigned)S}, {32, 1, 1});
      }
      enc.set_function(f_merge);
      enc.set_buffer(0, oacc); enc.set_buffer(1, mbuf); enc.set_buffer(2, lbuf);
      enc.set_buffer(3, ob);
      enc.set_constant(4, D); enc.set_constant(5, S); enc.set_constant(6, Hq);
      enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
    };
    // MMA flash-decode: 8 q-heads per tile (needs 8|G), grid (256, Hq/8, S).
    auto enc_mma = [&](ComputeEncoder& enc, int S) {
      enc.set_function(f_mma);
      enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
      enc.set_buffer(3, oacc); enc.set_buffer(4, mbuf); enc.set_buffer(5, lbuf);
      enc.set_constant(6, scale); enc.set_constant(7, T);
      enc.set_constant(8, D); enc.set_constant(9, Hq);
      enc.set_constant(10, Hkv);
      enc.set_constant(11, T - 1); enc.set_constant(12, T);
      enc.set_constant(13, win); enc.set_constant(14, 0);
      enc.set_constant(15, S);
      enc.dispatch({256, (unsigned)(Hq / 8), (unsigned)S}, {256, 1, 1});
      enc.set_function(f_merge);
      enc.set_buffer(0, oacc); enc.set_buffer(1, mbuf); enc.set_buffer(2, lbuf);
      enc.set_buffer(3, ob);
      enc.set_constant(4, D); enc.set_constant(5, S); enc.set_constant(6, Hq);
      enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
    };
    auto enc_mb = [&](ComputeEncoder& enc) {
      enc.set_function(f_mb);
      enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
      enc.set_buffer(3, ob);
      enc.set_constant(4, scale); enc.set_constant(5, T);
      enc.set_constant(6, D); enc.set_constant(7, Hq);
      enc.set_constant(8, Hkv); enc.set_constant(9, 1);
      enc.set_constant(10, T - 1); enc.set_constant(11, T);
      enc.set_constant(12, win); enc.set_constant(13, 0);
      const unsigned BN = 4096 / D;
      enc.dispatch({BN * 32, (unsigned)Hq, 1}, {BN * 32, 1, 1});
    };
    auto bench = [&](auto enc_one) {
      auto once = [&]() {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder enc = st.begin_compute();
          for (int r = 0; r < reps; ++r) { enc_one(enc); } }
        st.commit().wait();
      };
      for (int w = 0; w < 2; ++w) { once(); }
      const auto t0 = std::chrono::steady_clock::now();
      const int it = 5;
      for (int i = 0; i < it; ++i) { once(); }
      const auto t1 = std::chrono::steady_clock::now();
      return secs_(t0, t1) / it / reps * 1e3;   // ms/dispatch
    };

    // mb baseline
    { auto once = [&]() {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder enc = st.begin_compute(); enc_mb(enc); }
        st.commit().wait();
      }; once(); }
    const double ms_mb = bench([&](ComputeEncoder& e) { enc_mb(e); });
    double rel_mb = rel_of();
    std::printf("[gqa-micro] %s T=%4d  mb       rel=%.2e  %.4f ms\n",
                cfg.rt, T, rel_mb, ms_mb);

    struct Var { ComputeFunction* fn; const char* tag; };
    Var vars[] = {{&f_tile, "gtile "}, {&f_dir, "direct"}, {&f_vec, "vec   "}};
    for (auto& var : vars) {
      if (!var.fn->valid()) { continue; }
      for (int S : {32, 64, 96, 128, 256}) {
        { CommandStream st = mc->make_command_stream();
          { ComputeEncoder enc = st.begin_compute(); enc_gqa(enc, *var.fn, S); }
          st.commit().wait(); }
        const double rel = rel_of();
        const double ms = bench([&](ComputeEncoder& e) {
          enc_gqa(e, *var.fn, S);
        });
        std::printf("[gqa-micro] %s T=%4d  %s S=%2d  rel=%.2e  %.4f ms  "
                    "(%.2fx mb)\n", cfg.rt, T, var.tag, S, rel, ms, ms_mb / ms);
        EXPECT_TRUE(rel < 5e-2);
      }
    }
    // All-G single-simdgroup (Qwen's production design): read KV once for all
    // G heads. head_dim <= 256 only (per <= 8). S=32 is the production split.
    if (f_allg.valid() && D <= 256) {
      for (int S : {16, 32, 64, 128}) {
        { CommandStream st = mc->make_command_stream();
          { ComputeEncoder enc = st.begin_compute(); enc_allg(enc, S); }
          st.commit().wait(); }
        const double rel = rel_of();
        const double ms = bench([&](ComputeEncoder& e) { enc_allg(e, S); });
        std::printf("[gqa-micro] %s T=%4d  allg   S=%2d  rel=%.2e  %.4f ms  "
                    "(%.2fx mb)\n", cfg.rt, T, S, rel, ms, ms_mb / ms);
        EXPECT_TRUE(rel < 5e-2);
      }
    }
    // PAGED decode (Qwen's production path): pgqa = all-G single-simdgroup
    // (current sdpa_paged_gqa_mb256_f16), pvec = per-head UK+vec4 (candidate).
    // Full-causal only (paged kernels have no window). D % 128 == 0 for pvec.
    if (win == 0 && D <= 256 && f_pgqa.valid()) {
      for (int S : {32, 64}) {
        { CommandStream st = mc->make_command_stream();
          { ComputeEncoder enc = st.begin_compute();
            enc_paged(enc, f_pgqa, S, false); }
          st.commit().wait(); }
        const double rel = rel_of();
        const double ms = bench([&](ComputeEncoder& e) {
          enc_paged(e, f_pgqa, S, false); });
        std::printf("[gqa-micro] %s T=%4d  pgqa   S=%2d  rel=%.2e  %.4f ms  "
                    "(%.2fx mb)\n", cfg.rt, T, S, rel, ms, ms_mb / ms);
        EXPECT_TRUE(rel < 5e-2);
      }
    }
    if (win == 0 && D <= 256 && (D % 128 == 0) && f_pvec.valid()) {
      for (int S : {32, 64}) {
        { CommandStream st = mc->make_command_stream();
          { ComputeEncoder enc = st.begin_compute();
            enc_paged(enc, f_pvec, S, true); }
          st.commit().wait(); }
        const double rel = rel_of();
        const double ms = bench([&](ComputeEncoder& e) {
          enc_paged(e, f_pvec, S, true); });
        std::printf("[gqa-micro] %s T=%4d  pvec   S=%2d  rel=%.2e  %.4f ms  "
                    "(%.2fx mb)\n", cfg.rt, T, S, rel, ms, ms_mb / ms);
        EXPECT_TRUE(rel < 5e-2);
      }
    }
    // MMA q-head flash-decode (8 heads/tile, needs 8|G): the depth-flat
    // candidate. split = ceil(T/FL_C(64)) keeps each split ~1 key block.
    if (f_mma.valid() && (G % 8 == 0) && (D % 8 == 0)) {
      for (int S : {(T + 63) / 64, 64, 128}) {
        if (S < 1 || S > Smax) { continue; }
        { CommandStream st = mc->make_command_stream();
          { ComputeEncoder enc = st.begin_compute(); enc_mma(enc, S); }
          st.commit().wait(); }
        const double rel = rel_of();
        const double ms = bench([&](ComputeEncoder& e) { enc_mma(e, S); });
        std::printf("[gqa-micro] %s T=%4d  mma    S=%2d  rel=%.2e  %.4f ms  "
                    "(%.2fx mb)\n", cfg.rt, T, S, rel, ms, ms_mb / ms);
        EXPECT_TRUE(rel < 5e-2);
      }
    }
  }
  }
}
