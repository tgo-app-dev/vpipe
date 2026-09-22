// Sol-Attn on the MATRIX UNITS: the exact half on the tree's own steel
// flash kernel, the approximate half on a small MMA flash loop over the
// summaries, and the two merged.
//
// The first port ran the whole method in one plain simdgroup kernel and
// measured 281 GF/s against steel's 6656 -- so the routing's 3.8x saving
// was buried under a 24x kernel deficit. This is the same method with
// the exact blocks handed to steel, which is what the published CuTe
// kernels do and what this tree's VDN window already proved possible
// (has_spans + loader jump, measured at 1.85x and converting sparsity to
// time nearly 1:1).
//
// THE BASELINE HERE IS DENSE STEEL, not the earlier kernel. Comparing a
// new kernel against the one it replaces flatters it; the question is
// whether Sol beats the attention the model would otherwise run.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/shared/metal-sol-attention.h"
#include "generative-models/shared/sol-attention.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using namespace vpipe::metal_compute;

namespace {

// Steel's tiles for head_dim 128, from attn_steel.metal's header.
constexpr int kSteelBQ = 32;
constexpr int kSteelBK = 16;
// The matrix-core flash kernel's own tiles. Coarser on both axes, so
// the routing decides for 64 rows at a time instead of 32 -- a real
// difference in what gets kept, not only a faster way to keep it.
constexpr int kNaxBQ = 64;
constexpr int kNaxBK = 32;
constexpr int kD = 128;

struct AttnP {
  int B, H, D, qL, kL, gqa;
  float scale;
  int NQ, NK, NQa, NKa, qL_rem, kL_rem, qL_off;
  std::int64_t Qs[3], Ks[3], Vs[3], Os[3];
};

std::uint32_t rnd_(std::uint32_t& s)
{
  s = s * 1664525u + 1013904223u;
  return s;
}
float uni_(std::uint32_t& s)
{
  return (float)(rnd_(s) >> 8) / 8388608.0f - 1.0f;
}

// Clustered q/k, as in sol-attention.cc: uniform noise has no routing
// decision to make and would report a sparsity nobody sees.
void
clustered_(std::vector<float>& q, std::vector<float>& k,
           std::vector<float>& v, int H, int T, int centres, float spread,
           std::uint32_t seed)
{
  const std::size_t n = (std::size_t)H * T * kD;
  q.resize(n); k.resize(n); v.resize(n);
  std::uint32_t s = seed;
  std::vector<float> dir((std::size_t)centres * kD);
  for (auto& x : dir) { x = uni_(s); }
  auto fill = [&](std::vector<float>& dst) {
    for (int h = 0; h < H; ++h) {
      for (int t = 0; t < T; ++t) {
        const int c = (t / 37) % centres;
        float* row = dst.data() + ((std::size_t)h * T + t) * kD;
        float nrm = 0.0f;
        for (int i = 0; i < kD; ++i) {
          row[i] = dir[(std::size_t)c * kD + i] * (1.0f - spread) +
                   uni_(s) * spread;
          nrm += row[i] * row[i];
        }
        nrm = std::sqrt(nrm > 0.0f ? nrm : 1.0f);
        for (int i = 0; i < kD; ++i) { row[i] /= nrm; }
      }
    }
  };
  fill(q); fill(k);
  for (auto& x : v) { x = uni_(s); }
}

SharedBuffer up_(MetalCompute* mc, const std::vector<float>& x, bool bf16)
{
  SharedBuffer b = mc->make_shared_buffer(x.size() * 2);
  if (b.empty()) { return b; }
  if (!bf16) {
    auto* d = static_cast<_Float16*>(b.contents());
    for (std::size_t i = 0; i < x.size(); ++i) { d[i] = (_Float16)x[i]; }
    return b;
  }
  auto* d = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < x.size(); ++i) {
    std::uint32_t u;
    std::memcpy(&u, &x[i], 4);
    d[i] = (std::uint16_t)(u >> 16);
  }
  return b;
}

double rel_(const std::vector<float>& a, const std::vector<float>& b)
{
  double n = 0.0, d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double e = (double)a[i] - (double)b[i];
    n += e * e;
    d += (double)b[i] * (double)b[i];
  }
  return d > 0.0 ? std::sqrt(n / d) : std::sqrt(n);
}

// Everything one geometry needs, so the two tests share the setup.
struct Rig {
  MetalCompute* mc = nullptr;
  ComputeLibrary lib_sol, lib_steel;
  ComputeFunction f_sum, f_stats, f_route, f_scan, f_emit, f_approx, f_merge;
  ComputeFunction f_steel_dense, f_steel_spans;
  bool bf16 = false;
  // WHICH FLASH KERNEL TAKES THE EXACT HALF, and therefore what the CSR
  // and the routing are quantised to: the ALU steel entry is bq 32 /
  // bk 16, the matrix-core one bq 64 / bk 32. The approximate half
  // tiles 32 rows either way, so NQA is its own count.
  bool nax = false;
  int BQ = 0, BK = 0;
  int H = 0, T = 0, TPAD = 0, NQ = 0, NQA = 0, NK = 0, BLK = 0;
  float scale = 0.0f;
  SharedBuffer q, k, v, o_dense, o_exact, o_out;
  SharedBuffer qc, kc, vc, mean, var, flags, kept, qb_off, qb_blk;
  SharedBuffer o_a, m_a, l_a, m_e, l_e, counts, params, sp_params, sp_bounds;
  SharedBuffer arena;

  bool load(MetalCompute* m, int heads, int tokens, int blk,
            bool want_nax = false)
  {
    mc = m; H = heads; T = tokens; BLK = blk;
    nax = want_nax && m->supports_matrix_cores();
    BQ = nax ? kNaxBQ : kSteelBQ;
    BK = nax ? kNaxBK : kSteelBK;
    if (BLK % BK != 0) { return false; }
    NQ = (T + BQ - 1) / BQ;
    NQA = (T + kSteelBQ - 1) / kSteelBQ;
    NK = (T + BLK - 1) / BLK;
    TPAD = NQA * kSteelBQ;
    scale = 1.0f / std::sqrt((float)kD);
    // The dtype is an env switch so the two can be compared in one run:
    // H3 is bf16 and the tests default to f16, and a kernel that is fast
    // in one and not the other is a fact about simdgroup_matrix support,
    // not about the method.
    bf16 = std::getenv("VPIPE_SOL_BF16") != nullptr;
    lib_sol = mc->load_library(bf16 ? "sol_attn_mma_bf16" : "sol_attn_mma");
    lib_steel = mc->load_library(nax ? "attn_steel_nax" : "attn_steel");
    f_sum = lib_sol.function("sol_summaries_mma");
    f_stats = lib_sol.function("sol_kc_stats_mma");
    f_route = lib_sol.function("sol_route_mma");
    f_scan = lib_sol.function("sol_scan_mma");
    f_emit = lib_sol.function("sol_emit_mma");
    f_approx = lib_sol.function("sol_approx_mma");
    f_merge = lib_sol.function("sol_merge_mma");
    FunctionConstants fd;
    fd.set_bool(200, (T % 64) == 0).set_bool(201, (T % 32) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(303, false).set_bool(304, false);
    const char* dense_name =
        nax ? (bf16 ? "attn_steel_nax_h_bd128_bf16" : "attn_steel_nax_h_bd128")
            : (bf16 ? "attn_steel_h_bd128_bf16" : "attn_steel_h_bd128");
    f_steel_dense = lib_steel.function(dense_name, fd);
    FunctionConstants fs;
    fs.set_bool(200, (T % 64) == 0).set_bool(201, (T % 32) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(303, true).set_bool(304, true);
    f_steel_spans = lib_steel.function(dense_name, fs);
    return f_sum.valid() && f_stats.valid() && f_route.valid() &&
           f_scan.valid() && f_emit.valid() && f_approx.valid() &&
           f_merge.valid() && f_steel_dense.valid() && f_steel_spans.valid();
  }

  bool alloc()
  {
    const std::size_t n = (std::size_t)H * T * kD;
    o_dense = mc->make_shared_buffer(n * 2);
    o_exact = mc->make_shared_buffer(n * 2);
    o_out   = mc->make_shared_buffer(n * 2);
    qc = mc->make_shared_buffer((std::size_t)H * NQ * kD * 2);
    kc = mc->make_shared_buffer((std::size_t)H * NK * kD * 2);
    vc = mc->make_shared_buffer((std::size_t)H * NK * kD * 2);
    mean = mc->make_shared_buffer((std::size_t)H * kD * 4);
    var  = mc->make_shared_buffer((std::size_t)H * kD * 4);
    flags = mc->make_shared_buffer((std::size_t)H * NQ * NK);
    kept  = mc->make_shared_buffer((std::size_t)H * NQ * 4);
    qb_off = mc->make_shared_buffer((std::size_t)H * (NQ + 1) * 4);
    // Worst case: every query block keeps every key block.
    qb_blk = mc->make_shared_buffer((std::size_t)H * NQ *
                                    (std::size_t)(T / BK + 1) * 4);
    o_a = mc->make_shared_buffer((std::size_t)H * TPAD * kD * 4);
    m_a = mc->make_shared_buffer((std::size_t)H * T * 4);
    l_a = mc->make_shared_buffer((std::size_t)H * T * 4);
    m_e = mc->make_shared_buffer((std::size_t)H * T * 4);
    l_e = mc->make_shared_buffer((std::size_t)H * T * 4);
    counts = mc->make_shared_buffer(8);
    params = mc->make_shared_buffer(sizeof(AttnP));
    sp_params = mc->make_shared_buffer(5 * 4);
    sp_bounds = mc->make_shared_buffer(8);
    if (params.empty() || qb_blk.empty() || o_a.empty()) { return false; }
    auto* p = static_cast<AttnP*>(params.contents());
    p->B = 1; p->H = H; p->D = kD; p->qL = T; p->kL = T; p->gqa = 1;
    p->scale = scale;
    p->NQ = NQ; p->NK = (T + BK - 1) / BK;
    p->NQa = T / BQ; p->NKa = T / BK;
    p->qL_rem = T - p->NQa * BQ;
    p->kL_rem = T - p->NKa * BK;
    p->qL_off = 0;
    const std::int64_t hm[3] = {(std::int64_t)H * T * kD,
                                (std::int64_t)T * kD, kD};
    for (int i = 0; i < 3; ++i) {
      p->Qs[i] = hm[i]; p->Ks[i] = hm[i]; p->Vs[i] = hm[i];
      p->Os[i] = hm[i];
    }
    int* sp = static_cast<int*>(sp_params.contents());
    // tokens_per_frame 0 switches steel's span-edge predicate off: the
    // Sol list is exact at BK granularity, so nothing needs masking.
    sp[0] = 0; sp[1] = 0; sp[2] = 0; sp[3] = 0;
    sp[4] = NQ + 1;              // qb_off is [H][NQ + 1]
    std::memset(sp_bounds.contents(), 0, sp_bounds.byte_size());
    return true;
  }

  void encode_dense(ComputeEncoder& e)
  {
    e.set_function(f_steel_dense);
    e.set_buffer(0, q); e.set_buffer(1, k); e.set_buffer(2, v);
    e.set_buffer(3, o_dense); e.set_buffer(4, params);
    e.dispatch({32u * (unsigned)NQ, 4u * (unsigned)H, 1}, {32, 4, 1});
  }

  // A bitmask of passes to LEAVE OUT, so the bench can attribute its own
  // cost by difference: 1 summaries, 2 stats, 4 route, 8 scan, 16 emit,
  // 32 approx, 64 exact, 128 merge. The output is nonsense with
  // anything set -- this is a stopwatch, not a mode. Mirrors
  // VPIPE_SOL_SKIP in MetalSolAttention.
  int skip = 0;

  void encode_sol(ComputeEncoder& e, float tau, int radius, int sink_lo,
                  int sink_hi)
  {
    const int per = BLK / BK;
    const int nks = (T + BK - 1) / BK;
    // Summaries: twice, because q is summarised at the QUERY block size
    // and k/v at the KEY block size, and they are independent here.
    if ((skip & 1) == 0) {
    e.set_function(f_sum);
    e.set_buffer(0, q); e.set_buffer(1, k); e.set_buffer(2, v);
    e.set_buffer(3, qc); e.set_buffer(4, kc); e.set_buffer(5, vc);
    e.set_constant(6, T); e.set_constant(7, kD); e.set_constant(8, NK);
    e.set_constant(9, BLK); e.set_constant(10, 6);
    e.set_constant(11, T);   // head stride == tokens: not a band
    e.dispatch({kD, (unsigned)H, (unsigned)NK}, {kD, 1, 1});
    e.set_function(f_sum);
    e.set_buffer(0, q); e.set_buffer(1, q); e.set_buffer(2, q);
    e.set_buffer(3, qc); e.set_buffer(4, qc); e.set_buffer(5, qc);
    e.set_constant(6, T); e.set_constant(7, kD); e.set_constant(8, NQ);
    e.set_constant(9, BQ); e.set_constant(10, 1);
    e.set_constant(11, T);   // head stride == tokens: not a band
    e.dispatch({kD, (unsigned)H, (unsigned)NQ}, {kD, 1, 1});
    }

    if ((skip & 2) == 0) {
    e.set_function(f_stats);
    e.set_buffer(0, kc); e.set_buffer(1, mean); e.set_buffer(2, var);
    e.set_constant(3, kD); e.set_constant(4, NK);
    e.dispatch({kD, (unsigned)H, 1}, {kD, 1, 1});
    }

    if ((skip & 4) == 0) {
    e.set_function(f_route);
    e.set_buffer(0, qc); e.set_buffer(1, kc); e.set_buffer(2, mean);
    e.set_buffer(3, var); e.set_buffer(4, flags); e.set_buffer(5, kept);
    e.set_buffer(6, counts);
    e.set_constant(7, scale); e.set_constant(8, tau);
    e.set_constant(9, kD); e.set_constant(10, NK); e.set_constant(11, NQ);
    e.set_constant(12, BLK); e.set_constant(13, BQ);
    e.set_constant(14, radius);
    e.set_constant(15, sink_lo); e.set_constant(16, sink_hi);
    e.set_constant(17, per);
    e.set_constant(18, nks);
    e.set_constant(20, 0);   // query offset: a whole-sequence call
    e.dispatch({32, (unsigned)H, (unsigned)NQ}, {32, 1, 1});
    }

    if ((skip & 8) == 0) {
    e.set_function(f_scan);
    e.set_buffer(0, kept); e.set_buffer(1, qb_off);
    e.set_constant(2, NQ); e.set_constant(3, H); e.set_constant(4, per);
    e.dispatch({256, 1, 1}, {256, 1, 1});
    }

    if ((skip & 16) == 0) {
    e.set_function(f_emit);
    e.set_buffer(0, flags); e.set_buffer(1, qb_off); e.set_buffer(2, qb_blk);
    e.set_constant(3, NQ); e.set_constant(4, NK); e.set_constant(5, per);
    e.set_constant(6, nks);
    e.dispatch({32, (unsigned)H, (unsigned)NQ}, {32, 1, 1});
    }

    if ((skip & 32) == 0) {
    e.set_function(f_approx);
    e.set_buffer(0, q); e.set_buffer(1, kc); e.set_buffer(2, vc);
    e.set_buffer(3, flags); e.set_buffer(4, o_a); e.set_buffer(5, m_a);
    e.set_buffer(6, l_a);
    e.set_constant(7, scale); e.set_constant(8, T); e.set_constant(9, kD);
    e.set_constant(10, NK); e.set_constant(11, NQ);
    e.set_constant(12, BLK); e.set_constant(13, TPAD);
    e.set_constant(14, BQ);       // the ROUTING block; this tiles 32
    // The grid is in THREADS and the threadgroup is 2-D, so y must be
    // 4 * H for tid.y to be the head -- exactly as the steel dispatch
    // below spells it. {128, H, NQA} instead gives H/4 threadgroups in y
    // and four redundant ones in x, which writes a quarter of the heads
    // and leaves the rest at whatever the buffer held.
    e.dispatch({32, 4u * (unsigned)H, (unsigned)NQA}, {32, 4, 1});
    }

    if ((skip & 64) == 0) {
    e.set_function(f_steel_spans);
    e.set_buffer(0, q); e.set_buffer(1, k); e.set_buffer(2, v);
    e.set_buffer(3, o_exact); e.set_buffer(4, params);
    e.set_buffer(8, qb_off); e.set_buffer(9, qb_blk);
    e.set_buffer(10, sp_params); e.set_buffer(11, sp_bounds);
    e.set_buffer(12, m_e); e.set_buffer(13, l_e);
    e.dispatch({32u * (unsigned)NQ, 4u * (unsigned)H, 1}, {32, 4, 1});
    }

    if ((skip & 128) == 0) {
    e.set_function(f_merge);
    e.set_buffer(0, o_exact); e.set_buffer(1, m_e); e.set_buffer(2, l_e);
    e.set_buffer(3, o_a); e.set_buffer(4, m_a); e.set_buffer(5, l_a);
    e.set_buffer(6, o_out);
    e.set_constant(7, T); e.set_constant(8, kD); e.set_constant(9, TPAD);
    e.set_constant(10, T);   // destination stride == tokens: not a band
    e.dispatch({kD, (unsigned)H, (unsigned)T}, {kD, 1, 1});
    }
  }

  // How many times one timed command buffer encodes the pipeline. 1 is
  // the bench; the per-pass profile raises it, because attributing an
  // 8 ms pass by subtracting two 480 ms runs is a difference of the same
  // order as the noise between them. Every pass here is idempotent given
  // its inputs, so a repeat measures the same work again.
  int repeat = 1;

  bool run(bool sol, float tau, int radius, int sink_lo, int sink_hi,
           double* ms)
  {
    std::memset(counts.contents(), 0, counts.byte_size());
    const auto t0 = std::chrono::steady_clock::now();
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      for (int i = 0; i < repeat; ++i) {
        if (sol) { encode_sol(e, tau, radius, sink_lo, sink_hi); }
        else     { encode_dense(e); }
      }
    }
    std::string err;
    if (!st.commit().wait_ok(&err)) {
      std::printf("[sol-mma] dispatch failed: %s\n", err.c_str());
      return false;
    }
    *ms = std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t0).count();
    return true;
  }

  std::vector<float> read(const SharedBuffer& b) const
  {
    const std::size_t n = (std::size_t)H * T * kD;
    std::vector<float> o(n);
    if (!bf16) {
      const auto* p = static_cast<const _Float16*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = (float)p[i]; }
      return o;
    }
    const auto* p = static_cast<const std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t u = (std::uint32_t)p[i] << 16;
      std::memcpy(&o[i], &u, 4);
    }
    return o;
  }

  double kept_fraction() const
  {
    // The counter is in STEEL key blocks, so the denominator is too.
    const auto* c = static_cast<const std::uint32_t*>(counts.contents());
    const long long nks = (T + BK - 1) / BK;
    return (double)c[0] / (double)((long long)H * NQ * nks);
  }
};

}  // namespace

// The composed path against DENSE STEEL, at a size where the routing has
// something to choose from. Both arms are matrix-core kernels over the
// same inputs, so what separates them is the approximation and nothing
// else -- and tau = -inf, which keeps every block, must reproduce dense
// to the accumulation floor with the whole five-pass machinery still in
// the way.
TEST(sol_attention_mma, matches_dense_steel)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  int H = 4, T = 2048, BLK = 64;
  if (const char* e = std::getenv("VPIPE_SOL_T")) { T = std::atoi(e); }
  if (const char* e = std::getenv("VPIPE_SOL_H")) { H = std::atoi(e); }
  if (const char* e = std::getenv("VPIPE_SOL_BLK")) { BLK = std::atoi(e); }
  // BOTH FLASH KERNELS, and the matrix-core one is not a second copy of
  // the same statement: it routes at a 64-row query block against the
  // ALU kernel's 32, walks a CSR in 32-key units rather than 16, and
  // exports its softmax statistics from a cooperative-tensor fragment
  // rather than a simdgroup one. tau = -inf is what pins all three at
  // once -- keep every block and the whole machinery has to reproduce
  // that kernel's own dense output.
  for (int arm = 0; arm < 2; ++arm) {
  const bool want_nax = arm == 1;
  if (want_nax && !mc->supports_matrix_cores()) {
    std::printf("[sol-mma] no matrix cores -- NAX arm SKIPPED\n");
    break;
  }
  Rig r;
  if (!r.load(mc, H, T, BLK, want_nax)) {
    std::printf("[sol-mma] kernels did not validate\n");
    return;
  }
  ASSERT_TRUE(r.alloc());
  std::vector<float> q, k, v;
  clustered_(q, k, v, r.H, r.T, 8, 0.35f, 0xa11ce001u);
  r.q = up_(mc, q, r.bf16); r.k = up_(mc, k, r.bf16);
  r.v = up_(mc, v, r.bf16);
  ASSERT_TRUE(!r.q.empty() && !r.k.empty() && !r.v.empty());

  double ms = 0.0;
  ASSERT_TRUE(r.run(false, 0.0f, 1, 0, 0, &ms));
  const std::vector<float> dense = r.read(r.o_dense);

  // EVERY BLOCK EXACT. The five passes still run, the merge still
  // combines two partials, and the answer must come back dense -- which
  // pins the merge, the m/l export and the CSR against the one case
  // with a known answer.
  ASSERT_TRUE(r.run(true, -1.0e30f, 1, 0, 0, &ms));
  const std::vector<float> all_exact = r.read(r.o_out);
  const double e0 = rel_(all_exact, dense);
  std::printf("[sol-mma] %-3s blk %d: tau=-inf vs dense: rel-L2 %.3e "
              "(%.1f%% kept)\n", r.nax ? "nax" : "alu", BLK, e0,
              r.kept_fraction() * 100.0);
  EXPECT_TRUE(e0 < 5e-3);

  ASSERT_TRUE(r.run(true, 1.0f, 1, 0, 0, &ms));
  const std::vector<float> sol = r.read(r.o_out);
  const double e1 = rel_(sol, dense);
  std::printf("[sol-mma] %-3s blk %d: tau=1.0  vs dense: rel-L2 %.4f "
              "(%.1f%% kept)\n", r.nax ? "nax" : "alu", BLK, e1,
              r.kept_fraction() * 100.0);
  EXPECT_TRUE(e1 < 0.1);
  EXPECT_TRUE(e1 > 1e-4);          // it really did approximate something
  }
}

// THE BENCHMARK (VPIPE_SOL_BENCH=1), at H3's production attention
// geometry, against the dense steel kernel the model actually runs.
TEST(sol_attention_mma, bench)
{
  if (std::getenv("VPIPE_SOL_BENCH") == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  int heads = 8, tokens = 20036;
  if (const char* e = std::getenv("VPIPE_SOL_BENCH_HEADS")) {
    heads = std::atoi(e);
  }
  if (const char* e = std::getenv("VPIPE_SOL_BENCH_ROWS")) {
    tokens = std::atoi(e);
  }
  std::vector<float> q, k, v;
  clustered_(q, k, v, heads, tokens, 24, 0.35f, 0x50150150u);

  std::printf("[sol-mma] %d heads x %d rows x 128\n", heads, tokens);
  std::printf("[sol-mma] %4s %6s %9s %9s %9s %8s %8s\n", "arm", "BLK",
              "dense", "rig", "class", "speedup", "kept");
  // THE KEY BLOCK SIZE IS THE SWEEP. Bigger blocks make the routing
  // cheaper (fewer proxy scores, shorter CSR) and the summary coarser
  // (a centroid over more keys represents each of them worse), so there
  // is a knee somewhere and nothing says 64 is it on this hardware.
  // BOTH ARMS, and the dense column moves with them: comparing Sol on
  // the matrix-core kernel against dense on the ALU one would credit
  // the approximation with the port. What each row says is what routing
  // buys ON THAT KERNEL; what the two blocks of rows say to each other
  // is what the port buys.
  for (int arm = 0; arm < 2; ++arm) {
  const bool want_nax = arm == 1;
  if (want_nax && !mc->supports_matrix_cores()) { break; }
  for (const int blk : {32, 64, 128, 256}) {
    Rig r;
    if (!r.load(mc, heads, tokens, blk, want_nax)) { continue; }
    if (!r.alloc()) { continue; }
    r.q = up_(mc, q, r.bf16); r.k = up_(mc, k, r.bf16);
  r.v = up_(mc, v, r.bf16);
    if (r.q.empty()) { continue; }
    double warm = 0.0, dense_ms = 0.0, sol_ms = 0.0;
    // The sink H3 sets: everything below video_start is prompt and
    // soundtrack and must be read exactly. Mirrored here so the bench
    // measures the configuration the model actually runs.
    int sink_tok = 0;
    if (const char* e = std::getenv("VPIPE_SOL_SINK")) {
      sink_tok = std::atoi(e);
    }
    const int slo = 0;
    const int shi = sink_tok > 0 ? (sink_tok + blk - 1) / blk : 0;
    if (!r.run(true, 1.0f, 1, slo, shi, &warm)) { continue; }
    if (!r.run(false, 0.0f, 1, 0, 0, &dense_ms)) { continue; }
    if (!r.run(false, 0.0f, 1, 0, 0, &dense_ms)) { continue; }
    if (!r.run(true, 1.0f, 1, slo, shi, &sol_ms)) { continue; }
    // AND THE CLASS THE MODEL CALLS, which is not the rig: on a
    // matrix-core box its approximate half is the flash kernel with the
    // routing's flags as a mask, where the rig's is still the
    // simdgroup one. That difference is the point of the column.
    double cls_ms = 0.0;
    {
      if (want_nax) { ::unsetenv("VPIPE_SOL_NO_NAX"); }
      else          { ::setenv("VPIPE_SOL_NO_NAX", "1", 1); }
      std::string err;
      std::unique_ptr<MetalSolAttention> sol =
          MetalSolAttention::load(mc, r.bf16, &err);
      SharedBuffer out =
          mc->make_shared_buffer((std::size_t)heads * tokens * kD * 2);
      sol::Config cfg;
      cfg.enabled = true;
      cfg.tau = 1.0f;
      cfg.local_radius = 1;
      cfg.key_block = blk;
      cfg.sink_start = 0;
      cfg.sink_tokens = sink_tok;
      auto once = [&](double* ms) {
        const auto t0 = std::chrono::steady_clock::now();
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          if (!sol->encode(e, r.q, r.k, r.v, out, heads, tokens, kD,
                           r.scale, cfg, &err)) {
            return false;
          }
        }
        if (!st.commit().wait_ok(&err)) { return false; }
        *ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
        return true;
      };
      if (!sol || !once(&cls_ms) || !once(&cls_ms)) { cls_ms = 0.0; }
      ::unsetenv("VPIPE_SOL_NO_NAX");
    }
    std::printf("[sol-mma] %4s %6d %8.1fms %8.1fms %8.1fms %7.2fx %7.1f%%\n",
                r.nax ? "nax" : "alu", blk, dense_ms, sol_ms, cls_ms,
                cls_ms > 0.0 ? dense_ms / cls_ms : 0.0,
                r.kept_fraction() * 100.0);
    // WHERE THE REMAINING TIME IS, by difference. The exact half is the
    // one pass whose cost the routing controls; everything else is
    // Sol's own, and once the exact half moves to the matrix cores that
    // own cost is most of what is left. Attributed by leaving one pass
    // out rather than by timing it alone, because a per-pass commit
    // pays to make ~1 GB of scratch resident again and would report
    // that instead.
    if (std::getenv("VPIPE_SOL_BENCH_PROFILE") != nullptr) {
      struct { int bit; const char* name; } part[] = {
          {1, "summaries"}, {2, "stats"}, {4, "route"}, {8, "scan"},
          {16, "emit"}, {32, "approx"}, {64, "exact"}, {128, "merge"}};
      for (const auto& pt : part) {
        double without = 0.0;
        r.skip = pt.bit;
        if (!r.run(true, 1.0f, 1, slo, shi, &without)) { continue; }
        if (!r.run(true, 1.0f, 1, slo, shi, &without)) { continue; }
        std::printf("[sol-mma]        %-10s %7.1fms (%.0f%%)\n", pt.name,
                    sol_ms - without,
                    sol_ms > 0.0 ? 100.0 * (sol_ms - without) / sol_ms : 0.0);
      }
      r.skip = 0;
    }
  }
  }
  EXPECT_TRUE(true);
}

// SOL'S OWN PASSES, one at a time and repeated, because the bench
// attributes them by subtracting two runs of the whole thing -- and at
// the production geometry the two are ~480 ms apart while the pass
// between them is 8. Here each pass is the ONLY thing in the command
// buffer, encoded VPIPE_SOL_BENCH_PASSES times (default 16), so what is
// timed is the pass and the noise divides.
//
// The EXACT half is left out: it is not Sol's scaffolding, it is the
// attention, and it would dominate the run it is supposed to be
// measured beside. The approximate half is in, because on this arm it
// IS one of Sol's own kernels.
TEST(sol_attention_mma, bench_passes)
{
  if (std::getenv("VPIPE_SOL_BENCH_PASSES") == nullptr) { return; }
  int R = std::atoi(std::getenv("VPIPE_SOL_BENCH_PASSES"));
  if (R <= 0) { R = 16; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }

  int heads = 56, tokens = 20036, blk = 64;
  if (const char* e = std::getenv("VPIPE_SOL_BENCH_HEADS")) {
    heads = std::atoi(e);
  }
  if (const char* e = std::getenv("VPIPE_SOL_BENCH_ROWS")) {
    tokens = std::atoi(e);
  }
  if (const char* e = std::getenv("VPIPE_SOL_BLK")) { blk = std::atoi(e); }
  std::vector<float> q, k, v;
  clustered_(q, k, v, heads, tokens, 24, 0.35f, 0x50150150u);

  for (int arm = 0; arm < 2; ++arm) {
    const bool want_nax = arm == 1;
    if (want_nax && !mc->supports_matrix_cores()) { break; }
    Rig r;
    if (!r.load(mc, heads, tokens, blk, want_nax)) { continue; }
    if (!r.alloc()) { continue; }
    r.q = up_(mc, q, r.bf16); r.k = up_(mc, k, r.bf16);
    r.v = up_(mc, v, r.bf16);
    if (r.q.empty()) { continue; }
    double warm = 0.0;
    r.run(true, 1.0f, 1, 0, 16, &warm);
    std::printf("[sol-pass] %s %d heads x %d rows, block %d, x%d\n",
                r.nax ? "nax" : "alu", heads, tokens, blk, R);
    struct { int bit; const char* name; } part[] = {
        {1, "summaries"}, {2, "stats"}, {4, "route"}, {8, "scan"},
        {16, "emit"}, {32, "approx"}, {128, "merge"}};
    r.repeat = R;
    for (const auto& pt : part) {
      r.skip = 0xff & ~pt.bit;
      double ms = 0.0;
      if (!r.run(true, 1.0f, 1, 0, 16, &ms)) { continue; }
      if (!r.run(true, 1.0f, 1, 0, 16, &ms)) { continue; }
      std::printf("[sol-pass]   %-10s %7.2f ms\n", pt.name, ms / R);
    }
    r.repeat = 1;
    r.skip = 0;
  }
  EXPECT_TRUE(true);
}

// THE SUMMARY PASS WRITES ONLY WHAT IT WAS ASKED FOR, which is the
// contract that keeps it in bounds.
//
// `N` is ONE count for all three destinations, and the caller runs the
// kernel twice with two different block sizes -- k/v at the routing
// block, q at the exact half's query block. So a k/v call carries
// N = nk, and while it also wrote q it was writing nk query centroids
// into a buffer holding nq. That is harmless wherever nq >= nk, which
// is every geometry the ALU kernel routes, and a 2x overrun on the
// matrix-core one at key block 32 -- where the query block is 64 and nk
// is therefore twice nq.
//
// The geometry here is that one, on purpose and on any GPU: nk = 2 * nq
// with the destinations sized honestly and filled with a pattern, so a
// write outside the mask's permission is a changed byte rather than a
// fault that may or may not happen.
TEST(sol_attention_mma, a_summary_pass_writes_only_what_it_was_asked_for)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const bool bf16 = std::getenv("VPIPE_SOL_BF16") != nullptr;
  ComputeLibrary lib =
      mc->load_library(bf16 ? "sol_attn_mma_bf16" : "sol_attn_mma");
  ComputeFunction f_sum = lib.function("sol_summaries_mma");
  ASSERT_TRUE(f_sum.valid());
  if (!f_sum.valid()) { return; }

  // The matrix-core arm's shape: query block 64, key block 32.
  const int H = 2, T = 512, BQ = 64, BLK = 32;
  const int NQ = T / BQ, NK = T / BLK;      // 8 and 16
  std::vector<float> q, k, v;
  clustered_(q, k, v, H, T, 4, 0.4f, 0xA5A5u);
  SharedBuffer bq_ = up_(mc, q, bf16), bk = up_(mc, k, bf16),
               bv = up_(mc, v, bf16);
  SharedBuffer qc = mc->make_shared_buffer((std::size_t)H * NQ * kD * 2);
  SharedBuffer kc = mc->make_shared_buffer((std::size_t)H * NK * kD * 2);
  SharedBuffer vc = mc->make_shared_buffer((std::size_t)H * NK * kD * 2);
  ASSERT_TRUE(!bq_.empty() && !qc.empty() && !kc.empty() && !vc.empty());
  if (qc.empty() || kc.empty() || vc.empty() || bq_.empty()) { return; }
  auto fill = [](SharedBuffer& b, unsigned char c) {
    std::memset(b.contents(), c, b.byte_size());
  };
  auto unchanged = [](const SharedBuffer& b, unsigned char c) {
    const auto* p = static_cast<const unsigned char*>(b.contents());
    for (std::size_t i = 0; i < b.byte_size(); ++i) {
      if (p[i] != c) { return false; }
    }
    return true;
  };
  auto changed = [](const SharedBuffer& b, unsigned char c) {
    const auto* p = static_cast<const unsigned char*>(b.contents());
    for (std::size_t i = 0; i < b.byte_size(); ++i) {
      if (p[i] != c) { return true; }
    }
    return false;
  };
  auto encode = [&](int N, int blk, int which) {
    std::string err;
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      e.set_function(f_sum);
      e.set_buffer(0, bq_); e.set_buffer(1, bk); e.set_buffer(2, bv);
      e.set_buffer(3, qc); e.set_buffer(4, kc); e.set_buffer(5, vc);
      e.set_constant(6, T); e.set_constant(7, (int)kD);
      e.set_constant(8, N); e.set_constant(9, blk);
      e.set_constant(10, which);
      e.set_constant(11, T);   // head stride == tokens: not a band
      e.dispatch({kD, (unsigned)H, (unsigned)N}, {kD, 1, 1});
    }
    return st.commit().wait_ok(&err);
  };

  // k and v, at the ROUTING block -- the call whose N exceeds nq.
  fill(qc, 0xAB); fill(kc, 0xCD); fill(vc, 0xCD);
  ASSERT_TRUE(encode(NK, BLK, 6));
  // Not one byte of the query centroids, all NQ * D of which sit inside
  // the NK * D this dispatch would have written.
  EXPECT_TRUE(unchanged(qc, 0xAB));
  EXPECT_TRUE(changed(kc, 0xCD));
  EXPECT_TRUE(changed(vc, 0xCD));

  // ...and q, at the QUERY block, leaves the other two alone.
  fill(qc, 0xAB); fill(kc, 0xCD); fill(vc, 0xCD);
  ASSERT_TRUE(encode(NQ, BQ, 1));
  EXPECT_TRUE(changed(qc, 0xAB));
  EXPECT_TRUE(unchanged(kc, 0xCD));
  EXPECT_TRUE(unchanged(vc, 0xCD));
}

// The APPROXIMATE half alone, against a CPU computation of exactly what
// it is supposed to produce. Isolated on purpose: composed with steel
// and the merge, an error here is a number between 0 and 2 with nothing
// to say which of five passes produced it.
TEST(sol_attention_mma, the_approximate_partial_is_what_it_claims)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  Rig r;
  const int H = 1, T = 512, BLK = 64;
  if (!r.load(mc, H, T, BLK)) { return; }
  ASSERT_TRUE(r.alloc());
  std::vector<float> q, k, v;
  clustered_(q, k, v, H, T, 4, 0.4f, 0x1234u);
  r.q = up_(mc, q, r.bf16); r.k = up_(mc, k, r.bf16);
  r.v = up_(mc, v, r.bf16);
  ASSERT_TRUE(!r.q.empty());

  // tau enormous and no local band: everything the routing can drop, it
  // drops -- so the approximate half carries almost the whole answer and
  // an error in it cannot hide.
  double ms = 0.0;
  ASSERT_TRUE(r.run(true, 1.0e30f, -1, 0, 0, &ms));

  const int NK = r.NK;
  const auto* fl = static_cast<const unsigned char*>(r.flags.contents());
  const auto* kcp = static_cast<const _Float16*>(r.kc.contents());
  const auto* vcp = static_cast<const _Float16*>(r.vc.contents());
  const auto* map = static_cast<const float*>(r.m_a.contents());
  const auto* lap = static_cast<const float*>(r.l_a.contents());
  const auto* oap = static_cast<const float*>(r.o_a.contents());

  const float ls = r.scale * 1.44269504088896340736f;
  const float bonus = std::log2((float)BLK);
  double worst_o = 0.0, worst_l = 0.0;
  int checked = 0;
  for (int t = 0; t < T; t += 37) {
    const int qb = t / kSteelBQ;
    // The CPU version of the same partial: one logit per approximate
    // block, the block MEAN as its value, +log2(BLK) on the score.
    std::vector<double> acc((std::size_t)kD, 0.0);
    double m = -INFINITY, l = 0.0;
    for (int n = 0; n < NK; ++n) {
      if (fl[(std::size_t)qb * NK + n] != 0) { continue; }
      double dot = 0.0;
      for (int i = 0; i < kD; ++i) {
        dot += (double)q[(std::size_t)t * kD + i] *
               (double)(float)kcp[(std::size_t)n * kD + i];
      }
      const double s = dot * ls + bonus;
      const double mn = std::max(m, s);
      const double corr = (m == -INFINITY) ? 0.0 : std::exp2(m - mn);
      const double p = std::exp2(s - mn);
      l = l * corr + p;
      for (int i = 0; i < kD; ++i) {
        acc[(std::size_t)i] = acc[(std::size_t)i] * corr +
                              p * (double)(float)vcp[(std::size_t)n * kD + i];
      }
      m = mn;
    }
    if (l <= 0.0) { continue; }
    ++checked;
    // Compare at the SAME max: the kernel and this may have taken
    // different running maxima, and only the ratio is meaningful.
    const double sh = std::exp2(m - (double)map[t]);
    double num = 0.0, den = 0.0;
    for (int i = 0; i < kD; ++i) {
      const double got = oap[((std::size_t)t) * kD + i];
      const double want = acc[(std::size_t)i] * sh;
      num += (got - want) * (got - want);
      den += want * want;
    }
    const double eo = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
    const double el = std::abs((double)lap[t] - l * sh) /
                      std::max(1e-12, l * sh);
    worst_o = std::max(worst_o, eo);
    worst_l = std::max(worst_l, el);
  }
  std::printf("[sol-mma] approx partial over %d rows: worst o %.4f, "
              "worst l %.4f\n", checked, worst_o, worst_l);
  EXPECT_TRUE(checked > 0);
  EXPECT_TRUE(worst_o < 0.02);
  EXPECT_TRUE(worst_l < 0.02);
}

// THE CLASS THE MODEL ACTUALLY CALLS, against dense.
//
// Everything above drives the kernels through a rig that fills
// AttnParams itself. MetalSolAttention fills its own, and a field it
// forgets is a zero rather than an error -- which is how the exact half
// shipped running every routed block at scale 0, a uniform average of
// the values it kept. Nothing else could see it: the rig supplies the
// field, and the model-level tests ask only that the answer be finite,
// reproducible and different from dense, all of which a uniform
// attention is.
//
// tau = -inf keeps every block, so the composed call must reproduce
// dense to the accumulation floor -- with the routing, both partials
// and the merge still in the way.
TEST(sol_attention_mma, the_class_reproduces_dense)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const int H = 4, T = 2048, BLK = 64;
  std::vector<float> q, k, v;
  clustered_(q, k, v, H, T, 8, 0.35f, 0xc1a55e51u);

  for (int arm = 0; arm < 2; ++arm) {
    const bool want_nax = arm == 1;
    if (want_nax && !mc->supports_matrix_cores()) { break; }
    // The class reads the environment once per process, so the ALU arm
    // has to be asked for before the first load.
    if (!want_nax) { ::setenv("VPIPE_SOL_NO_NAX", "1", 1); }
    else           { ::unsetenv("VPIPE_SOL_NO_NAX"); }
    Rig r;
    if (!r.load(mc, H, T, BLK, want_nax)) { continue; }
    ASSERT_TRUE(r.alloc());
    r.q = up_(mc, q, r.bf16); r.k = up_(mc, k, r.bf16);
    r.v = up_(mc, v, r.bf16);
    if (r.q.empty()) { return; }
    double ms = 0.0;
    ASSERT_TRUE(r.run(false, 0.0f, 1, 0, 0, &ms));
    const std::vector<float> dense = r.read(r.o_dense);

    std::string err;
    std::unique_ptr<MetalSolAttention> sol =
        MetalSolAttention::load(mc, r.bf16, &err);
    ASSERT_TRUE(sol != nullptr);
    if (!sol) { std::printf("[sol-mma] %s\n", err.c_str()); return; }
    // The class picks its own kernel; if it disagrees with the arm we
    // asked for, the comparison would be against the wrong dense.
    EXPECT_TRUE(sol->uses_matrix_cores() == want_nax);

    SharedBuffer out = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
    ASSERT_TRUE(!out.empty());
    sol::Config cfg;
    cfg.enabled = true;
    cfg.tau = -1.0e30f;
    cfg.local_radius = 1;
    cfg.key_block = BLK;
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ASSERT_TRUE(sol->encode(e, r.q, r.k, r.v, out, H, T, kD, r.scale,
                                cfg, &err));
      }
      ASSERT_TRUE(st.commit().wait_ok(&err));
    }
    const std::vector<float> got = r.read(out);
    const double e0 = rel_(got, dense);
    std::printf("[sol-mma] %-3s class tau=-inf vs dense: rel-L2 %.3e\n",
                want_nax ? "nax" : "alu", e0);
    EXPECT_TRUE(e0 >= 0.0 && e0 < 5e-3);

    // ...and a real tau still approximates rather than degenerating.
    cfg.tau = 1.0f;
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ASSERT_TRUE(sol->encode(e, r.q, r.k, r.v, out, H, T, kD, r.scale,
                                cfg, &err));
      }
      ASSERT_TRUE(st.commit().wait_ok(&err));
    }
    const double e1 = rel_(r.read(out), dense);
    std::printf("[sol-mma] %-3s class tau=1.0  vs dense: rel-L2 %.4f\n",
                want_nax ? "nax" : "alu", e1);
    EXPECT_TRUE(e1 > 1e-4 && e1 < 0.1);
  }
  ::unsetenv("VPIPE_SOL_NO_NAX");
}

// A RECTANGULAR BAND of a longer sequence.
//
// A block-causal layout does not have one attention -- it has one
// dispatch per segment, each a row band of a buffer holding the whole
// joint sequence, and each rectangular: a target image block attends
// itself plus everything before it, so qL < kL. Sol's method does not
// care (the routing scores query-block centroids against key-block
// centroids, and neither has to be the same sequence); what cared was
// the plumbing. Three kernels reused the token count as a head stride,
// and the local band assumed a query's row index equalled a key's.
//
// TWO CLAIMS, and the first is the one that catches a stride:
//
//  * the band read IN PLACE out of the long buffers must equal the same
//    rows COPIED OUT and attended contiguously -- bit for bit, because
//    it is the same arithmetic reached by different strides; and
//  * at tau = -inf it must reproduce dense attention over that
//    rectangle, which is what says the rectangle itself is right and
//    not merely self-consistent.
TEST(sol_attention_mma, rectangular_band_matches_contiguous_and_dense)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const int H = 4, T = 2048, BLK = 64;
  // A band that is rectangular AND offset: queries are the last 1024
  // rows, keys are everything up to and including them, which is the
  // shape a target image block has in a joint sequence.
  int qL = 1024, kL = 1536, off = 512;
  if (const char* e = std::getenv("VPIPE_SOL_QL")) { qL = std::atoi(e); }
  if (const char* e = std::getenv("VPIPE_SOL_KL")) { kL = std::atoi(e); }
  if (const char* e = std::getenv("VPIPE_SOL_OFF")) { off = std::atoi(e); }
  std::vector<float> q, k, v;
  clustered_(q, k, v, H, T, 8, 0.35f, 0xb0dd1e55u);

  for (int arm = 0; arm < 2; ++arm) {
    const bool want_nax = arm == 1;
    if (want_nax && !mc->supports_matrix_cores()) { break; }
    if (!want_nax) { ::setenv("VPIPE_SOL_NO_NAX", "1", 1); }
    else           { ::unsetenv("VPIPE_SOL_NO_NAX"); }
    Rig r;
    if (!r.load(mc, H, T, BLK, want_nax)) { continue; }
    ASSERT_TRUE(r.alloc());
    r.q = up_(mc, q, r.bf16); r.k = up_(mc, k, r.bf16);
    r.v = up_(mc, v, r.bf16);
    if (r.q.empty()) { return; }
    std::string err;
    std::unique_ptr<MetalSolAttention> sol =
        MetalSolAttention::load(mc, r.bf16, &err);
    ASSERT_TRUE(sol != nullptr);
    if (!sol) { return; }

    sol::Config cfg;
    cfg.enabled = true;
    cfg.tau = -1.0e30f;      // every block exact: the answer is dense
    cfg.local_radius = 1;
    cfg.key_block = BLK;

    // ARM 1: in place, out of the full-length buffers.
    SharedBuffer out_band = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
    ASSERT_TRUE(!out_band.empty());
    MetalSolAttention::Band band{qL, kL, T, T, off, off};
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ASSERT_TRUE(sol->encode(e, r.q, r.k, r.v, out_band, H, H, band, kD,
                                r.scale, cfg, &err));
      }
      ASSERT_TRUE(st.commit().wait_ok(&err));
    }

    // ARM 2: the same rows, copied out and attended contiguously. The
    // band's q_off stays `off` -- it is the query's SEQUENCE position,
    // which the local band is measured against, and copying the rows
    // does not move them in the sequence.
    std::vector<float> qc((std::size_t)H * qL * kD);
    std::vector<float> kc((std::size_t)H * kL * kD);
    std::vector<float> vc((std::size_t)H * kL * kD);
    for (int h = 0; h < H; ++h) {
      for (int t = 0; t < qL; ++t) {
        std::memcpy(&qc[((std::size_t)h * qL + t) * kD],
                    &q[((std::size_t)h * T + off + t) * kD],
                    (std::size_t)kD * sizeof(float));
      }
      for (int t = 0; t < kL; ++t) {
        std::memcpy(&kc[((std::size_t)h * kL + t) * kD],
                    &k[((std::size_t)h * T + t) * kD],
                    (std::size_t)kD * sizeof(float));
        std::memcpy(&vc[((std::size_t)h * kL + t) * kD],
                    &v[((std::size_t)h * T + t) * kD],
                    (std::size_t)kD * sizeof(float));
      }
    }
    SharedBuffer qcb = up_(mc, qc, r.bf16);
    SharedBuffer kcb = up_(mc, kc, r.bf16);
    SharedBuffer vcb = up_(mc, vc, r.bf16);
    SharedBuffer out_c =
        mc->make_shared_buffer((std::size_t)H * qL * kD * 2);
    ASSERT_TRUE(!qcb.empty() && !kcb.empty() && !vcb.empty() &&
                !out_c.empty());
    // Same SEQUENCE position, row 0 of its own buffer: the pair the
    // band form exists to tell apart.
    MetalSolAttention::Band tight{qL, kL, qL, kL, off, 0};
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ASSERT_TRUE(sol->encode(e, qcb, kcb, vcb, out_c, H, H, tight, kD,
                                r.scale, cfg, &err));
      }
      ASSERT_TRUE(st.commit().wait_ok(&err));
    }

    // Compare the band's rows against the contiguous answer. Read by
    // SIZE rather than through the rig, whose reader assumes the full
    // sequence -- the contiguous arm's buffer is shorter than that.
    auto read_n = [&](const SharedBuffer& buf, std::size_t n) {
      std::vector<float> o(n);
      if (!r.bf16) {
        const auto* p2 = static_cast<const _Float16*>(buf.contents());
        for (std::size_t i = 0; i < n; ++i) { o[i] = (float)p2[i]; }
        return o;
      }
      const auto* p2 = static_cast<const std::uint16_t*>(buf.contents());
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t u = (std::uint32_t)p2[i] << 16;
        std::memcpy(&o[i], &u, 4);
      }
      return o;
    };
    const std::vector<float> a = read_n(out_band, (std::size_t)H * T * kD);
    const std::vector<float> b = read_n(out_c, (std::size_t)H * qL * kD);
    ASSERT_TRUE(a.size() == (std::size_t)H * T * kD);
    ASSERT_TRUE(b.size() == (std::size_t)H * qL * kD);
    std::size_t differ = 0;
    double num = 0.0, den = 0.0;
    for (int h = 0; h < H; ++h) {
      for (int t = 0; t < qL; ++t) {
        for (int d = 0; d < kD; ++d) {
          const float x = a[((std::size_t)h * T + off + t) * kD + d];
          const float y = b[((std::size_t)h * qL + t) * kD + d];
          if (x != y) { ++differ; }
          num += (double)(x - y) * (x - y);
          den += (double)y * y;
        }
      }
    }
    // Only on a failure, and it is worth the lines: the first version
    // of this plumbing was wrong for exactly the first `off` rows, and
    // "which rows" is what said so.
    if (differ > 0) {
      int h0 = -1, t0 = -1, d0 = -1, tmin = 1 << 30, tmax = -1;
      std::vector<int> per_head(H, 0);
      for (int h = 0; h < H; ++h) {
        for (int t = 0; t < qL; ++t) {
          bool row_bad = false;
          for (int d = 0; d < kD; ++d) {
            if (a[((std::size_t)h * T + off + t) * kD + d] !=
                b[((std::size_t)h * qL + t) * kD + d]) {
              row_bad = true;
              if (h0 < 0) { h0 = h; t0 = t; d0 = d; }
            }
          }
          if (row_bad) {
            ++per_head[h];
            tmin = std::min(tmin, t); tmax = std::max(tmax, t);
          }
        }
      }
      std::printf("[sol-mma] first diff h=%d t=%d d=%d; bad rows t in "
                  "[%d, %d]; per head:", h0, t0, d0, tmin, tmax);
      for (int h = 0; h < H; ++h) { std::printf(" %d", per_head[h]); }
      std::printf(" (of %d)\n", qL);
    }
    std::printf("[sol-mma] %-3s band vs contiguous: %zu of %zu words "
                "differ\n", want_nax ? "nax" : "alu", differ,
                (std::size_t)H * qL * kD);
    EXPECT_TRUE(differ == 0);
    (void)num; (void)den;

    // AND THE RECTANGLE IS RIGHT, not just self-consistent: at
    // tau = -inf every key block is exact, so this is dense attention
    // over [off, off+qL) x [0, kL).
    std::vector<float> want((std::size_t)H * qL * kD, 0.0f);
    for (int h = 0; h < H; ++h) {
      for (int t = 0; t < qL; ++t) {
        const float* qr = &q[((std::size_t)h * T + off + t) * kD];
        std::vector<float> lse((std::size_t)kL);
        float m = -INFINITY;
        for (int j = 0; j < kL; ++j) {
          const float* kr = &k[((std::size_t)h * T + j) * kD];
          float dot = 0.0f;
          for (int d = 0; d < kD; ++d) { dot += qr[d] * kr[d]; }
          lse[(std::size_t)j] = dot * r.scale;
          m = std::max(m, lse[(std::size_t)j]);
        }
        float den2 = 0.0f;
        for (int j = 0; j < kL; ++j) {
          lse[(std::size_t)j] = std::exp(lse[(std::size_t)j] - m);
          den2 += lse[(std::size_t)j];
        }
        float* o = &want[((std::size_t)h * qL + t) * kD];
        for (int j = 0; j < kL; ++j) {
          const float w = lse[(std::size_t)j] / den2;
          if (w == 0.0f) { continue; }
          const float* vr = &v[((std::size_t)h * T + j) * kD];
          for (int d = 0; d < kD; ++d) { o[d] += w * vr[d]; }
        }
      }
    }
    const double rel = rel_(b, want);
    std::printf("[sol-mma] %-3s rectangle tau=-inf vs CPU dense: rel-L2 "
                "%.3e\n", want_nax ? "nax" : "alu", rel);
    EXPECT_TRUE(rel < 5e-3);
  }
  ::unsetenv("VPIPE_SOL_NO_NAX");
}

// GQA: K AND V CARRY FEWER HEADS THAN Q.
//
// Sol summarises K and V, so the centroids and their statistics are per
// KV head, while the routing decision, the flags, the CSR and both
// partial softmaxes stay per QUERY head. A query head therefore has to
// read the group it belongs to, and getting that index wrong is not a
// crash -- it is a clean-looking answer built from another group's keys.
//
// THE CONTROL IS THE SAME CALL WITH K/V EXPANDED. An MHA forward over
// K/V repeated `gqa` times is the same arithmetic over the same values:
// identical centroids, thresholds, routing decisions and exact blocks.
// So the two must agree to rounding and the bar can be TIGHT, rather
// than an approximation tolerance a wrong group could hide inside. It
// also needs no new reference -- the MHA arm is the path every caller
// took before GQA existed, and it is checked against dense above.
TEST(sol_attention_mma, gqa_matches_mha_with_expanded_kv)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const int H = 8, KVH = 2, T = 1024, BLK = 64;
  const int gqa = H / KVH;
  const bool bf16 = std::getenv("VPIPE_SOL_BF16") != nullptr;
  const float scale = 1.0f / std::sqrt((float)kD);

  std::string err;
  std::unique_ptr<MetalSolAttention> sol =
      MetalSolAttention::load(mc, bf16, &err);
  ASSERT_TRUE(sol != nullptr);
  if (!sol) { std::printf("[sol-mma] %s\n", err.c_str()); return; }

  // q from one draw and k/v from another, so the groups are genuinely
  // different tensors rather than one tensor read twice.
  std::vector<float> q, k_ignored, v_ignored;
  clustered_(q, k_ignored, v_ignored, H, T, 8, 0.35f, 0x50a1b2c3u);
  std::vector<float> q_ignored, k, v;
  clustered_(q_ignored, k, v, KVH, T, 8, 0.35f, 0x9e3779b9u);

  // What an MHA caller would have to build: every KV head repeated for
  // the query heads in its group.
  const std::size_t hs = (std::size_t)T * kD;
  std::vector<float> ke((std::size_t)H * hs), ve((std::size_t)H * hs);
  for (int h = 0; h < H; ++h) {
    std::memcpy(&ke[(std::size_t)h * hs], &k[(std::size_t)(h / gqa) * hs],
                hs * sizeof(float));
    std::memcpy(&ve[(std::size_t)h * hs], &v[(std::size_t)(h / gqa) * hs],
                hs * sizeof(float));
  }

  const SharedBuffer bq = up_(mc, q, bf16);
  const SharedBuffer bke = up_(mc, ke, bf16), bve = up_(mc, ve, bf16);
  const SharedBuffer bk = up_(mc, k, bf16), bv = up_(mc, v, bf16);
  if (bq.empty() || bk.empty()) { return; }
  SharedBuffer o_mha = mc->make_shared_buffer((std::size_t)H * hs * 2);
  SharedBuffer o_gqa = mc->make_shared_buffer((std::size_t)H * hs * 2);
  ASSERT_TRUE(!o_mha.empty() && !o_gqa.empty());

  sol::Config cfg;
  cfg.enabled = true;
  cfg.tau = 1.0f;
  cfg.local_radius = 1;
  cfg.key_block = BLK;

  // The MHA arm goes through the LEGACY overload -- the one the plugins
  // and H3 call -- so the delegation is covered here too.
  auto run_mha = [&](SharedBuffer& o) {
    CommandStream st = mc->make_command_stream();
    bool ok = false;
    {
      ComputeEncoder e = st.begin_compute();
      ok = sol->encode(e, bq, bke, bve, o, H, T, kD, scale, cfg, &err);
    }
    return ok && st.commit().wait_ok(&err);
  };
  auto run_gqa = [&](SharedBuffer& o, int kvh) {
    CommandStream st = mc->make_command_stream();
    bool ok = false;
    {
      ComputeEncoder e = st.begin_compute();
      ok = sol->encode(e, bq, bk, bv, o, H, kvh, T, kD, scale, cfg, &err);
    }
    return ok && st.commit().wait_ok(&err);
  };

  // The simdgroup approximate kernel indexes the summary sequence by the
  // QUERY head, so GQA is refused on that arm rather than answered from
  // the wrong group. It is a bench A/B, and a wrong answer there would
  // be worse than not having it.
  if (std::getenv("VPIPE_SOL_NO_MASKED_APPROX") != nullptr) {
    EXPECT_FALSE(run_gqa(o_gqa, KVH));
    std::printf("[sol-mma] gqa refused on the simdgroup approx: %s\n",
                err.c_str());
    return;
  }

  // ONE OBJECT FOR BOTH ARMS, which also exercises the rebuild: the key
  // summaries are sized by the KV head count, so changing only that has
  // to invalidate the scratch. Until `_kv_heads` joined the geometry
  // tag it did not, and the second arm would have run against buffers
  // shaped for the first.
  ASSERT_TRUE(run_mha(o_mha));
  ASSERT_TRUE(run_gqa(o_gqa, KVH));

  auto read = [&](const SharedBuffer& b) {
    std::vector<float> o((std::size_t)H * hs);
    if (!bf16) {
      const auto* p = static_cast<const _Float16*>(b.contents());
      for (std::size_t i = 0; i < o.size(); ++i) { o[i] = (float)p[i]; }
      return o;
    }
    const auto* p = static_cast<const std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < o.size(); ++i) {
      const std::uint32_t u = (std::uint32_t)p[i] << 16;
      std::memcpy(&o[i], &u, 4);
    }
    return o;
  };
  const double e = rel_(read(o_gqa), read(o_mha));
  std::printf("[sol-mma] gqa %d/%d vs mha-expanded: rel-L2 %.3e\n", H, KVH,
              e);
  EXPECT_TRUE(e >= 0.0 && e < 1e-6);

  // A GROUP FACTOR THAT DOES NOT DIVIDE IS REFUSED, not rounded: the
  // ragged last group would read key centroids that were never
  // summarised, which is the failure this whole index exists to avoid.
  err.clear();
  {
    CommandStream st = mc->make_command_stream();
    ComputeEncoder e2 = st.begin_compute();
    EXPECT_FALSE(sol->encode(e2, bq, bk, bv, o_gqa, H, 3, T, kD, scale, cfg,
                             &err));
  }
  EXPECT_TRUE(!err.empty());
}

// THE TWO ROUTING KERNELS MUST KEEP THE SAME BLOCKS.
//
// The proxy is the same dot product either way -- accumulated in f32
// and stored in f32 -- so moving it from a per-key simdgroup reduction
// into one batched GEMM changes the summation ORDER and nothing else.
// The bar is therefore the routing DECISION and not the output quality:
// a threshold crossing that moved would show up here as a different
// kept count, where in the output it would hide inside a tolerance that
// the approximation already occupies.
TEST(sol_attention_mma, the_gemm_proxy_routes_identically)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[sol-mma] no matrix cores -- proxy A/B SKIPPED\n");
    return;
  }
  const int H = 4, T = 4096, BLK = 64;
  std::vector<float> q, k, v;
  clustered_(q, k, v, H, T, 12, 0.35f, 0x9e3779b9u);

  Rig r;
  if (!r.load(mc, H, T, BLK, /*want_nax=*/true)) { return; }
  ASSERT_TRUE(r.alloc());
  r.q = up_(mc, q, r.bf16); r.k = up_(mc, k, r.bf16);
  r.v = up_(mc, v, r.bf16);
  if (r.q.empty()) { return; }

  std::vector<float> out[2];
  long long kept[2] = {0, 0};
  for (int arm = 0; arm < 2; ++arm) {
    if (arm == 0) { ::setenv("VPIPE_SOL_NO_PROXY_GEMM", "1", 1); }
    else          { ::unsetenv("VPIPE_SOL_NO_PROXY_GEMM"); }
    std::string err;
    std::unique_ptr<MetalSolAttention> sol =
        MetalSolAttention::load(mc, r.bf16, &err);
    ASSERT_TRUE(sol != nullptr);
    if (!sol) { return; }
    sol->reset_counts();
    SharedBuffer o = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
    ASSERT_TRUE(!o.empty());
    sol::Config cfg;
    cfg.enabled = true;
    cfg.tau = 1.0f;
    cfg.local_radius = 1;
    cfg.key_block = BLK;
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ASSERT_TRUE(sol->encode(e, r.q, r.k, r.v, o, H, T, kD, r.scale, cfg,
                                &err));
      }
      ASSERT_TRUE(st.commit().wait_ok(&err));
    }
    out[arm] = r.read(o);
    kept[arm] = sol->exact_blocks();
  }
  ::unsetenv("VPIPE_SOL_NO_PROXY_GEMM");
  const double rel = rel_(out[1], out[0]);
  std::printf("[sol-mma] proxy A/B: fused kept %lld, gemm kept %lld, "
              "outputs rel-L2 %.3e\n", kept[0], kept[1], rel);
  EXPECT_TRUE(kept[0] > 0 && kept[0] == kept[1]);
  // Same blocks and the same exact-half arithmetic, so the outputs are
  // the same bytes -- the proxy feeds a decision and nothing else.
  EXPECT_TRUE(rel == 0.0);
}

// scratch_bytes() must equal what ensure_scratch_ actually allocates.
//
// It is spent at the FIRST ENCODE -- inside the first forward, long
// after anything that plans memory has had its say -- so a planner asks
// this instead, and an estimate never compared with the allocation it
// predicts drifts the first time a buffer is added. It had already
// drifted twice before this test existed: it counted an fp32 [H, TPAD,
// D] approximate output the matrix-core arm does not allocate, and
// missed both buffers that arm does.
//
// Checked at both arms and two key blocks, because the two change
// different terms: the arm swaps a 4-byte padded partial for a 2-byte
// one and adds the proxy matrix, the block sets the summary length.
TEST(sol_attention_mma, the_scratch_estimate_matches_the_allocation)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const int H = 8, T = 8192;

  for (int arm = 0; arm < 2; ++arm) {
    const bool want_nax = arm == 1;
    if (want_nax && !mc->supports_matrix_cores()) { break; }
    if (!want_nax) { ::setenv("VPIPE_SOL_NO_NAX", "1", 1); }
    else           { ::unsetenv("VPIPE_SOL_NO_NAX"); }
    for (const int blk : {32, 64}) {
      std::string err;
      std::unique_ptr<MetalSolAttention> sol =
          MetalSolAttention::load(mc, /*bf16=*/true, &err);
      ASSERT_TRUE(sol != nullptr);
      if (!sol) { return; }
      EXPECT_TRUE(sol->uses_matrix_cores() == want_nax);
      SharedBuffer q = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
      SharedBuffer o = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
      ASSERT_TRUE(!q.empty() && !o.empty());
      sol::Config cfg;
      cfg.enabled = true;
      cfg.tau = 1.0f;
      cfg.key_block = blk;
      {
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          const bool ok =
              sol->encode(e, q, q, q, o, H, T, kD, 0.08f, cfg, &err);
          if (!ok) { std::printf("[sol_mem] encode: %s\n", err.c_str()); }
          ASSERT_TRUE(ok);
        }
        ASSERT_TRUE(st.commit().wait_ok(&err));
      }
      const std::size_t want =
          MetalSolAttention::scratch_bytes(H, T, kD, blk, want_nax);
      const std::size_t got = sol->resident_bytes();
      std::printf("[sol_mem] %-3s blk %3d: estimate %7.1f MB, allocated "
                  "%7.1f MB%s\n", want_nax ? "nax" : "alu", blk,
                  (double)want / 1048576.0, (double)got / 1048576.0,
                  want == got ? "" : "   MISMATCH");
      // EXACT, not a bound: any difference is a buffer one of them knows
      // about and the other does not, which is the whole failure this
      // catches.
      EXPECT_TRUE(want == got);
    }
  }
  ::unsetenv("VPIPE_SOL_NO_NAX");
}

// LENT SCRATCH IS THE SAME SCRATCH.
//
// Everything Sol holds bar a few hundred bytes lives for one call, so
// the caller can hand it memory it is not using -- and the DiT has 4
// attention windows of exactly that: the fused qkv projection, dead
// from the transpose to the feed-forward, and the arena's last window,
// dead until the transpose that follows the call.
//
// Two things have to hold and neither is obvious. The carve must be
// BYTE-IDENTICAL in its answer, because a buffer that overlapped a live
// one would show up as an approximation moving rather than as a crash.
// And private_bytes() must predict what is left, exactly -- a planner
// that under-counts is the failure this whole change is about.
TEST(sol_attention_mma, a_lent_scratch_is_the_same_scratch)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  const int H = 4, T = 4096, BLK = 64;
  std::vector<float> q, k, v;
  clustered_(q, k, v, H, T, 12, 0.35f, 0x5011euL);

  Rig r;
  if (!r.load(mc, H, T, BLK, mc->supports_matrix_cores())) { return; }
  ASSERT_TRUE(r.alloc());
  r.q = up_(mc, q, r.bf16); r.k = up_(mc, k, r.bf16);
  r.v = up_(mc, v, r.bf16);
  if (r.q.empty()) { return; }

  // What the DiT lends, in this rig's terms: the fused projection is
  // three attention windows and the spare arena window is one.
  const std::size_t lend_a = (std::size_t)T * 3 * (std::size_t)H * kD * 2;
  const std::size_t lend_b = (std::size_t)T * (std::size_t)H * kD * 2;
  SharedBuffer arena_a = mc->make_shared_buffer(lend_a);
  SharedBuffer arena_b = mc->make_shared_buffer(lend_b);
  ASSERT_TRUE(!arena_a.empty() && !arena_b.empty());
  // POISONED, so nothing Sol is supposed to write is accidentally right
  // because the allocator handed over zeros.
  std::memset(arena_a.contents(), 0x5a, lend_a);
  std::memset(arena_b.contents(), 0x5a, lend_b);

  std::vector<float> out[2];
  std::size_t owned[2] = {0, 0};
  for (int arm = 0; arm < 2; ++arm) {
    std::string err;
    std::unique_ptr<MetalSolAttention> sol =
        MetalSolAttention::load(mc, r.bf16, &err);
    ASSERT_TRUE(sol != nullptr);
    if (!sol) { return; }
    if (arm == 1) { sol->set_arena(arena_a, arena_b); }
    SharedBuffer o = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
    ASSERT_TRUE(!o.empty());
    sol::Config cfg;
    cfg.enabled = true;
    cfg.tau = 1.0f;
    cfg.local_radius = 1;
    cfg.key_block = BLK;
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ASSERT_TRUE(sol->encode(e, r.q, r.k, r.v, o, H, T, kD, r.scale, cfg,
                                &err));
      }
      ASSERT_TRUE(st.commit().wait_ok(&err));
    }
    out[arm] = r.read(o);
    owned[arm] = sol->resident_bytes();
  }
  const bool same = out[0].size() == out[1].size() &&
                    std::memcmp(out[0].data(), out[1].data(),
                                out[0].size() * 4) == 0;
  const std::size_t want = MetalSolAttention::private_bytes(
      H, T, kD, BLK, mc->supports_matrix_cores(), lend_a, lend_b);
  std::printf("[sol_mem] own %7.1f MB unlent -> %7.1f MB lent (predicted "
              "%7.1f), outputs %s\n", (double)owned[0] / 1048576.0,
              (double)owned[1] / 1048576.0,
              (double)(want - MetalSolAttention::pinned_bytes()) / 1048576.0,
              same ? "IDENTICAL" : "DIFFER");
  EXPECT_TRUE(same);
  // The lend has to actually take, and the prediction has to be the
  // carve -- resident_bytes() excludes the pinned four, which
  // private_bytes() includes.
  EXPECT_TRUE(owned[1] < owned[0]);
  EXPECT_TRUE(owned[1] + MetalSolAttention::pinned_bytes() == want);
}

// THE LENDER GETS ITS ARENA BACK. This is the bug the 1344x768 x 328
// frame H3 run died of, and it is a LIFETIME bug rather than a sizing
// one, which is why every memory figure in that run looked reasonable.
//
// Sol announces its scratch to the process-wide residency set, which
// RETAINS what it holds -- and most of that scratch is windows carved
// out of the arena the DiT lent. A subview shares its parent's
// MTL::Buffer, so the set was handed the DiT's whole multi-gigabyte
// forward arena. `unload_when_idle: destroy` then destroyed the DiT and
// freed nothing: the set was still holding it, and the set lives on
// MetalCompute, which outlives every model. The VAE decode that ran next
// found the box full and refused.
//
// Measured through the DEVICE's allocated size rather than Sol's own
// accounting, because Sol's accounting was never wrong -- it reported
// the few hundred bytes it really owns, which is exactly why nothing
// noticed.
TEST(sol_attention_mma, destroying_sol_gives_the_lent_arena_back)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->residency_set_supported()) { return; }
  const int H = 4, T = 4096, BLK = 64;
  std::vector<float> q, k, v;
  clustered_(q, k, v, H, T, 12, 0.35f, 0x5011euL);

  Rig r;
  if (!r.load(mc, H, T, BLK, mc->supports_matrix_cores())) { return; }
  ASSERT_TRUE(r.alloc());
  r.q = up_(mc, q, r.bf16); r.k = up_(mc, k, r.bf16);
  r.v = up_(mc, v, r.bf16);
  if (r.q.empty()) { return; }

  const std::size_t lend_a = (std::size_t)T * 3 * (std::size_t)H * kD * 2;
  const std::size_t lend_b = (std::size_t)T * (std::size_t)H * kD * 2;
  const std::size_t lent = lend_a + lend_b;

  const std::size_t base = mc->memory_budget().allocated;
  {
    SharedBuffer arena_a = mc->make_shared_buffer(lend_a);
    SharedBuffer arena_b = mc->make_shared_buffer(lend_b);
    ASSERT_TRUE(!arena_a.empty() && !arena_b.empty());
    std::string err;
    std::unique_ptr<MetalSolAttention> sol =
        MetalSolAttention::load(mc, r.bf16, &err);
    ASSERT_TRUE(sol != nullptr);
    if (!sol) { return; }
    sol->set_arena(arena_a, arena_b);
    SharedBuffer o = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
    ASSERT_TRUE(!o.empty());
    sol::Config cfg;
    cfg.enabled = true;
    cfg.tau = 1.0f;
    cfg.local_radius = 1;
    cfg.key_block = BLK;
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ASSERT_TRUE(sol->encode(e, r.q, r.k, r.v, o, H, T, kD, r.scale, cfg,
                                &err));
      }
      ASSERT_TRUE(st.commit().wait_ok(&err));
    }
    // Everything is alive here, so the arena is legitimately on the books.
    EXPECT_TRUE(mc->memory_budget().allocated >= base + lent);
  }
  // ...and gone here. Before the fix this stayed up by the whole lend,
  // for the life of the process.
  const std::size_t after = mc->memory_budget().allocated;
  EXPECT_TRUE(after < base + lent / 2);
}

// The same thing on the REBUILD path, which is how it compounded: the
// DiT takes its arena back (set_arena with nothing) before allocating a
// bigger one, so a run that grew its scratch left one dead arena in the
// set per rebuild.
TEST(sol_attention_mma, taking_the_arena_back_releases_it)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->residency_set_supported()) { return; }
  const int H = 4, T = 2048, BLK = 64;
  std::vector<float> q, k, v;
  clustered_(q, k, v, H, T, 8, 0.35f, 0x77e1uL);

  Rig r;
  if (!r.load(mc, H, T, BLK, mc->supports_matrix_cores())) { return; }
  ASSERT_TRUE(r.alloc());
  r.q = up_(mc, q, r.bf16); r.k = up_(mc, k, r.bf16);
  r.v = up_(mc, v, r.bf16);
  if (r.q.empty()) { return; }

  const std::size_t lend_a = (std::size_t)T * 3 * (std::size_t)H * kD * 2;
  const std::size_t lend_b = (std::size_t)T * (std::size_t)H * kD * 2;
  const std::size_t lent = lend_a + lend_b;

  std::string err;
  std::unique_ptr<MetalSolAttention> sol =
      MetalSolAttention::load(mc, r.bf16, &err);
  ASSERT_TRUE(sol != nullptr);
  if (!sol) { return; }

  // COUNTED, NOT WEIGHED. The sibling test above reads
  // currentAllocatedSize(), which is the whole DEVICE's figure and moves
  // with every other Metal object in the process -- so a bound of half
  // the lent bytes bounds unrelated activity as much as it bounds this,
  // and it failed here at EXACTLY the tolerance the moment a test that
  // touches Metal was added ahead of it. What the bug actually was is an
  // allocation the residency set never gave back, and the set can be
  // asked how many it is holding.
  const std::size_t held = mc->residency_stats().current;
  {
    SharedBuffer arena_a = mc->make_shared_buffer(lend_a);
    SharedBuffer arena_b = mc->make_shared_buffer(lend_b);
    ASSERT_TRUE(!arena_a.empty() && !arena_b.empty());
    sol->set_arena(arena_a, arena_b);
    SharedBuffer o = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
    ASSERT_TRUE(!o.empty());
    sol::Config cfg;
    cfg.enabled = true;
    cfg.tau = 1.0f;
    cfg.local_radius = 1;
    cfg.key_block = BLK;
    {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        ASSERT_TRUE(sol->encode(e, r.q, r.k, r.v, o, H, T, kD, r.scale, cfg,
                                &err));
      }
      ASSERT_TRUE(st.commit().wait_ok(&err));
    }
    // ...and the whole scratch came OUT of the arena, which is the other
    // half of the claim: a take-back that returns nothing is trivially
    // clean if nothing was ever lent.
    EXPECT_TRUE(sol->resident_bytes() == 0);
    // ...AND THE SET ACTUALLY TOOK THE ARENA, which is what makes the
    // count below an invariant rather than a tautology. `held` is read
    // outside the scope and is 0 on a fresh session, so `after == held`
    // is 0 == 0 -- true whether the adds happen or not. The sibling test
    // has this witness in its byte form (>= base + lent); converting
    // this one to counts dropped it.
    EXPECT_TRUE(mc->residency_stats().current > held);
    // The lender takes it back, exactly as the DiT does before it
    // reallocates. Sol outlives this scope; the arena must not.
    sol->set_arena(SharedBuffer{}, SharedBuffer{});
  }
  const std::size_t after = mc->residency_stats().current;
  std::printf("[sol_mem] take-back: the set held %llu allocations, now "
              "%llu\n", (unsigned long long)held,
              (unsigned long long)after);
  EXPECT_TRUE(after == held);
  (void)lent;
}

// SOL AND SAGE TOGETHER, which before this composed only in the config
// file: the branch that routes returns before the dispatch Sage lives
// on, so a graph naming both ran Sage on the one block Sol leaves dense
// and nothing said so.
//
// They are orthogonal by construction -- Sol decides WHICH key blocks
// are attended, Sage how the ones that are get multiplied -- so the bar
// is that turning Sage on moves the answer by about what int8 costs and
// not by what a routing change would cost. Three arms against ONE dense
// reference: Sol alone, Sage alone, and both.
//
// The composed arm is required to differ from Sol-alone BITWISE as well.
// Two arms agreeing perfectly is exactly what a function constant that
// failed to apply looks like, and it is the failure this test exists
// for -- the whole bug was a path that silently did nothing.
TEST(sol_attention_mma, sage_composes_with_the_exact_half)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[sol-sage] no matrix cores -- the int8 QK cannot run, "
                "SKIPPED\n");
    return;
  }
  const int H = 4, T = 4096, BLK = 64;
  std::vector<float> q, k, v;
  clustered_(q, k, v, H, T, 12, 0.35f, 0x5a6e77uL);

  Rig r;
  if (!r.load(mc, H, T, BLK, /*nax=*/true)) { return; }
  ASSERT_TRUE(r.alloc());
  r.q = up_(mc, q, r.bf16); r.k = up_(mc, k, r.bf16);
  r.v = up_(mc, v, r.bf16);
  if (r.q.empty()) { return; }

  std::string err;
  std::unique_ptr<MetalSageAttention> sage =
      MetalSageAttention::load(mc, r.bf16, &err);
  ASSERT_TRUE(sage != nullptr);
  if (!sage) { return; }

  // tau = -inf keeps every block exact, so the routing is the identity
  // and what is left between the arms is the int8 product alone. That is
  // the discriminator: under a routed tau the two effects would be mixed
  // and neither bound would mean anything.
  auto run = [&](bool sage_on, float tau, std::vector<float>* out,
                 bool* engaged) {
    std::unique_ptr<MetalSolAttention> sol =
        MetalSolAttention::load(mc, r.bf16, &err);
    if (!sol) { return false; }
    SharedBuffer o = mc->make_shared_buffer((std::size_t)H * T * kD * 2);
    if (o.empty()) { return false; }
    sol::Config cfg;
    cfg.enabled = true;
    cfg.tau = tau;
    cfg.local_radius = 1;
    cfg.key_block = BLK;
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      bool ok = true;
      if (sage_on) {
        const MetalSageAttention::Operand qo{&r.q, 0, kD, T * kD};
        const MetalSageAttention::Operand ko{&r.k, 0, kD, T * kD};
        // compose(), not a bare Config: the key smoothing is exact
        // under ONE softmax and Sol's row is split across two, so a
        // composing caller quantizes K raw. Passing the bare config
        // here would test a configuration no model can reach and would
        // hide the term that does not cancel -- see
        // sol_attention_h3.sol_and_sage_compose_in_the_stack, where the
        // difference is 0.0376 against 0.0191.
        sage::Config gc;
        gc.enabled = true;
        ok = sage->prepare(e, qo, ko, H, H, T, T, kD, sol->query_block(),
                           sol->key_block_unit(),
                           MetalSolAttention::compose(gc), &err);
      }
      if (!ok) { return false; }
      sol->set_sage(sage_on ? sage.get() : nullptr);
      if (!sol->encode(e, r.q, r.k, r.v, o, H, T, kD, r.scale, cfg, &err)) {
        return false;
      }
    }
    if (!st.commit().wait_ok(&err)) { return false; }
    *engaged = sol->sage_engaged();
    *out = r.read(o);
    return true;
  };

  std::vector<float> dense, sol_only, both;
  bool e0 = false, e1 = false, e2 = false;
  ASSERT_TRUE(run(false, -std::numeric_limits<float>::infinity(), &dense,
                  &e0));
  ASSERT_TRUE(run(false, 1.0f, &sol_only, &e1));
  ASSERT_TRUE(run(true, 1.0f, &both, &e2));
  if (dense.empty() || both.empty()) { return; }
  // The arms are the arms: without this the test can run Sol twice.
  EXPECT_TRUE(!e0 && !e1);
  EXPECT_TRUE(e2);

  const double r_sol = rel_(sol_only, dense);
  const double r_both = rel_(both, dense);
  std::size_t differ = 0;
  for (std::size_t i = 0; i < both.size(); ++i) {
    differ += (both[i] != sol_only[i]) ? 1 : 0;
  }
  std::printf("[sol-sage] vs dense: sol %.4f | sol+sage %.4f, %zu of %zu "
              "differ from sol alone\n", r_sol, r_both, differ, both.size());
  // Sage costs the routed answer about what int8 costs a dense one --
  // not what a changed routing would cost, which is the failure that
  // would show up here as the second bound blowing past the first.
  EXPECT_TRUE(std::isfinite(r_both));
  EXPECT_TRUE(r_both < r_sol + 0.02);
  // ...and it RAN. One percent, not the quarter this bound started at:
  // that figure was calibrated against a composition that was WRONG --
  // the key smoothing displaced the exact half's logits against the
  // approximate half's, which moved 90% of the output. Corrected, Sage
  // costs the routed answer about what int8 costs a dense one and moves
  // a fifth of it. A bound fitted to the broken number would have
  // rejected the fix.
  EXPECT_TRUE(differ > both.size() / 100);
}


namespace {

// Width-parameterised twins of the file's kD=128 helpers, for the
// head_dim-64 test below. Kept local rather than templating the Rig:
// the Rig carries a dozen kD-shaped buffers and only this one test
// needs another width.
float h2f_d_(std::uint16_t h)
{
  _Float16 v;
  std::memcpy(&v, &h, 2);
  return (float)v;
}

void clustered_d_(std::vector<float>& q, std::vector<float>& k,
                  std::vector<float>& v, int H, int T, int D, int nclust,
                  float spread, unsigned long seed)
{
  const std::size_t n = (std::size_t)H * T * D;
  q.resize(n); k.resize(n); v.resize(n);
  std::uint32_t s = (std::uint32_t)seed;
  auto rnd = [&]() {
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xffff) / 32768.0f - 1.0f;
  };
  // Clustered keys, so the routing has something to keep and something
  // to drop -- a uniform field makes every block look alike and tau
  // stops meaning anything.
  std::vector<float> c((std::size_t)nclust * D);
  for (auto& e : c) { e = rnd(); }
  for (int h = 0; h < H; ++h) {
    for (int t = 0; t < T; ++t) {
      const int cl = (t * nclust) / T;
      for (int d = 0; d < D; ++d) {
        const std::size_t i =
            ((std::size_t)h * T + (std::size_t)t) * D + (std::size_t)d;
        const float base = c[(std::size_t)cl * D + (std::size_t)d];
        q[i] = base + spread * rnd();
        k[i] = base + spread * rnd();
        v[i] = rnd();
      }
    }
  }
}

SharedBuffer up_d_(MetalCompute* mc, const std::vector<float>& x)
{
  SharedBuffer b = mc->make_shared_buffer(x.size() * 2);
  if (b.empty()) { return b; }
  auto* p = static_cast<std::uint16_t*>(b.contents());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const _Float16 h = (_Float16)x[i];
    std::memcpy(&p[i], &h, 2);
  }
  return b;
}

// The dense bd64 attention, through the same kernel arm Sol's exact half
// takes -- so the comparison isolates Sol and not the kernel.
bool dense_bd64_(MetalCompute* mc, const SharedBuffer& q,
                 const SharedBuffer& k, const SharedBuffer& v,
                 SharedBuffer& out, int H, int T, float scale, bool nax)
{
  struct P {
    int B, Hh, D, qL, kL, gqa;
    float sc;
    int NQ, NK, NQa, NKa, qL_rem, kL_rem, qL_off;
    std::int64_t Qs[3], Ks[3], Vs[3], Os[3];
  };
  const int D = 64;
  const int bq = nax ? 64 : 32, bk = nax ? 32 : 16;
  ComputeLibrary lib =
      mc->load_library(nax ? "attn_steel_nax" : "attn_steel");
  if (!lib.valid()) { return false; }
  FunctionConstants fc;
  fc.set_bool(200, (T % bq) == 0).set_bool(201, (T % bk) == 0)
      .set_bool(300, false).set_bool(301, false).set_bool(302, false)
      .set_bool(303, false).set_bool(304, false).set_bool(305, false)
      .set_bool(306, false);
  ComputeFunction fn = lib.function(
      nax ? "attn_steel_nax_h_bd64" : "attn_steel_h_bd64", fc);
  if (!fn.valid()) { return false; }
  SharedBuffer pb = mc->make_shared_buffer(sizeof(P));
  if (pb.empty()) { return false; }
  auto* p = static_cast<P*>(pb.contents());
  p->B = 1; p->Hh = H; p->D = D; p->qL = T; p->kL = T; p->gqa = 1;
  p->sc = scale;
  p->NQ = (T + bq - 1) / bq; p->NK = (T + bk - 1) / bk;
  p->NQa = T / bq; p->NKa = T / bk;
  p->qL_rem = T - p->NQa * bq; p->kL_rem = T - p->NKa * bk;
  p->qL_off = 0;
  const std::int64_t hm[3] = {(std::int64_t)H * T * D,
                              (std::int64_t)T * D, D};
  for (int i = 0; i < 3; ++i) {
    p->Qs[i] = hm[i]; p->Ks[i] = hm[i]; p->Vs[i] = hm[i]; p->Os[i] = hm[i];
  }
  std::string err;
  CommandStream st = mc->make_command_stream();
  {
    ComputeEncoder e = st.begin_compute();
    e.set_function(fn);
    e.set_buffer(0, q); e.set_buffer(1, k); e.set_buffer(2, v);
    e.set_buffer(3, out); e.set_buffer(4, pb);
    e.dispatch({32u * (unsigned)p->NQ, 4u * (unsigned)H, 1}, {32, 4, 1});
  }
  return st.commit().wait_ok(&err);
}

}  // namespace

// SOL AT HEAD_DIM 64, which is VOSR's width and which Sol asserted
// against from the day it shipped.
//
// Both flash kernels are instantiated at 64 and 128 with the SAME tiles
// (ALU 32/16, NAX 64/32), so nothing about the routing, the CSR or the
// block statistics moves with the width -- only the entry point's name
// and the summaries' row length, and both were already parameters. What
// really is 128-specialised is sol_approx_mma, whose register-tiled
// output is built around SOL_D; that kernel left the default path when
// the block-masked flash half replaced it, so the width is refused only
// under VPIPE_SOL_NO_MASKED_APPROX.
//
// tau = -inf is the discriminator: with every block kept exact, Sol is
// an identity over the dense attention, so a wrong entry point, a wrong
// tile or a wrong stride shows up as a mismatch rather than as a
// plausible approximation.
TEST(sol_attention_mma, head_dim_64_is_a_supported_width)
{
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr || !mc->valid()) { return; }
  constexpr int D64 = 64;
  const int H = 4, T = 2048, BLK = 64;

  // Both arms, because the entry point differs on each and a box with
  // matrix cores runs the one the ALU box never sees.
  for (int arm = 0; arm < 2; ++arm) {
    const bool want_nax = (arm == 1);
    if (want_nax && !mc->supports_matrix_cores()) { continue; }
    if (want_nax) { ::unsetenv("VPIPE_SOL_NO_NAX"); }
    else          { ::setenv("VPIPE_SOL_NO_NAX", "1", 1); }
    std::string err;
    std::unique_ptr<MetalSolAttention> sol =
        MetalSolAttention::load(mc, /*bf16=*/false, &err);
    ::unsetenv("VPIPE_SOL_NO_NAX");
    if (!sol) { continue; }
    if (sol->uses_matrix_cores() != want_nax) { continue; }

    std::vector<float> q, k, v;
    clustered_d_(q, k, v, H, T, D64, 8, 0.35f, 0x64d0uL + (unsigned)arm);
    const std::size_t n = (std::size_t)H * T * D64;
    SharedBuffer qb = up_d_(mc, q), kb = up_d_(mc, k), vb = up_d_(mc, v);
    SharedBuffer o_sol = mc->make_shared_buffer(n * 2);
    SharedBuffer o_den = mc->make_shared_buffer(n * 2);
    if (qb.empty() || o_den.empty()) { return; }
    const float scale = 1.0f / std::sqrt((float)D64);

    // Dense reference through the SAME kernel Sol's exact half uses.
    if (!dense_bd64_(mc, qb, kb, vb, o_den, H, T, scale, want_nax)) {
      std::printf("[sol-d64] arm %d: dense bd64 unavailable -- skip\n", arm);
      continue;
    }
    sol::Config cfg;
    cfg.enabled = true;
    cfg.tau = -std::numeric_limits<float>::infinity();
    cfg.local_radius = 1;
    cfg.key_block = BLK;
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder e = st.begin_compute();
        ASSERT_TRUE(sol->encode(e, qb, kb, vb, o_sol, H, T, D64, scale, cfg,
                                &err)); }
      ASSERT_TRUE(st.commit().wait_ok(&err)); }

    const auto* a = static_cast<const std::uint16_t*>(o_den.contents());
    const auto* b = static_cast<const std::uint16_t*>(o_sol.contents());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double x = h2f_d_(a[i]), y = h2f_d_(b[i]);
      num += (x - y) * (x - y);
      den += x * x;
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : 0.0;
    std::printf("[sol-d64] %s arm, tau=-inf vs dense bd64: rel-L2 %.2e\n",
                want_nax ? "nax" : "alu", rel);
    // Every block kept exact means Sol IS the dense attention, modulo
    // the merge's one extra rounding.
    EXPECT_TRUE(std::isfinite(rel) && rel < 1e-3);

    // ...and it still routes at this width.
    cfg.tau = 1.0f;
    { CommandStream st = mc->make_command_stream();
      { ComputeEncoder e = st.begin_compute();
        ASSERT_TRUE(sol->encode(e, qb, kb, vb, o_sol, H, T, D64, scale, cfg,
                                &err)); }
      ASSERT_TRUE(st.commit().wait_ok(&err)); }
    double n2 = 0.0, d2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double x = h2f_d_(a[i]), y = h2f_d_(b[i]);
      n2 += (x - y) * (x - y);
      d2 += x * x;
    }
    const double r2 = d2 > 0.0 ? std::sqrt(n2 / d2) : 0.0;
    std::printf("[sol-d64] %s arm, tau=1.0  vs dense bd64: rel-L2 %.4f\n",
                want_nax ? "nax" : "alu", r2);
    EXPECT_TRUE(std::isfinite(r2) && r2 > 0.0 && r2 < 0.5);
  }
}
