// minimax-h3-blocks.cc -- block-level kernel selection for the MiniMax-H3
// DiT on M5 matrix-core (matmul2d) hardware.
//
// H3 was brought up on a box with NO matrix cores, so every kernel it
// dispatches -- the steel quantized GEMM and the ALU steel flash attention
// -- was chosen where matmul2d does not exist. Its own profile says GEMM is
// 71% of a step and attention 23%, both already at 83-86% of that machine's
// ALU ceiling, which is precisely the regime where the matrix units are the
// only structural lever left.
//
// The 33B checkpoint is 66 GB at bf16 and ~16.5 GB at w4, so this box cannot
// load it and cannot A/B a route end to end. What it CAN do is run every
// GEMM and both attentions at the REAL shapes -- taken from the released
// config.json and recorded below -- against every kernel that could serve
// them. That is the optiq_blocks approach, and it answers the question a
// checkpoint-less box can actually answer: does the matrix-core path work at
// these shapes, and is it faster.
//
// What this file does NOT establish is that the model's forward is wired to
// the winning kernel correctly. Only a real checkpoint shows that, and the
// Boogu bring-up is the cautionary case -- its NAX wiring shipped unmeasured
// from a non-matrix-core box and its VAE was silently broken. So the perf
// arms here are REPORTED, and what is ASSERTED is agreement between the
// arms: a route that disagrees with steel is wired wrong, whatever its rate.
//
// Runs on synthetic weights, so it needs no model and no env var. The rate
// sweep is gated behind VPIPE_H3_BLOCKS (=full widens the row ladder), since
// timing is not something a default suite run should pay for; the
// correctness arms always run.

#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

// ---- the released MiniMaxAI/MiniMax-H3 FL2VA transformer config ---------
// hidden 5376, 56 heads x head_dim 128 = 7168 attention inner width (WIDER
// than the residual stream), SwiGLU ffn 14336 (fc1 is 2x), 50 blocks.
constexpr int kHidden  = 5376;
constexpr int kHeads   = 56;
constexpr int kHeadDim = 128;
constexpr int kInner   = kHeads * kHeadDim;   // 7168
constexpr int kFfn     = 14336;

// The four distinct projection shapes one block dispatches.
struct Proj { const char* tag; int N, K; };
const Proj kProjs[] = {
    {"qkv", 3 * kInner, kHidden},    // 21504 x 5376
    {"o",   kHidden,    kInner},     //  5376 x 7168
    {"fc1", 2 * kFfn,   kHidden},    // 28672 x 5376
    {"fc2", kHidden,    kFfn},       //  5376 x 14336
};

// Row counts. 602 and 9382 are from H3's own step_bench -- the latter is the
// production 960x544 layout -- and the packed sequence runs to ~19k.
const int kRowsDefault[] = {602, 9382};
const int kRowsFull[]    = {602, 2304, 4282, 9382, 19008};

MetalCompute* mc_(Session& s)
{
  MetalCompute* mc = s.metal_compute();
  return (mc != nullptr && mc->valid()) ? mc : nullptr;
}

double secs_(std::chrono::steady_clock::time_point a,
             std::chrono::steady_clock::time_point b)
{
  return std::chrono::duration<double>(b - a).count();
}

bool bench_on_() { return std::getenv("VPIPE_H3_BLOCKS") != nullptr; }

bool bench_full_()
{
  const char* e = std::getenv("VPIPE_H3_BLOCKS");
  return e != nullptr && std::string(e) == "full";
}

// H3 runs bf16 throughout, so every buffer here is bf16 and the helpers
// convert explicitly. Round-to-nearest-even on the way in, because a
// truncating cast would bias every value the same way and quietly widen
// every rel-L2 below.
std::uint16_t to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = (u >> 16) & 1u;
  u += 0x7fffu + r;
  return (std::uint16_t)(u >> 16);
}

float from_bf16_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

SharedBuffer make_act_(MetalCompute* mc, std::size_t n, std::uint32_t seed)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  auto* p = static_cast<std::uint16_t*>(b.contents());
  std::uint32_t r = seed | 1u;
  for (std::size_t i = 0; i < n; ++i) {
    r = r * 1664525u + 1013904223u;
    p[i] = to_bf16_(((float)(r >> 16) / 32768.0f - 1.0f) * 0.1f);
  }
  return b;
}

// A packed affine weight (group 64) + scales/biases, filled with
// deterministic garbage. The VALUES do not matter for rate, and for the
// agreement arms all that matters is that both kernels read the same bytes.
struct QW { SharedBuffer w, s, b; };

QW make_qw_(MetalCompute* mc, int N, int K, int bits, std::uint32_t seed)
{
  const std::size_t codes = (std::size_t)N * K * bits / 8;
  const std::size_t ng = (std::size_t)N * (K / 64);
  QW q{mc->make_shared_buffer(codes), mc->make_shared_buffer(ng * 2),
       mc->make_shared_buffer(ng * 2)};
  if (q.w.empty() || q.s.empty() || q.b.empty()) { return QW{}; }
  auto* wp = static_cast<std::uint8_t*>(q.w.contents());
  std::uint32_t r = seed | 1u;
  for (std::size_t i = 0; i < codes; ++i) {
    r = r * 1664525u + 1013904223u;
    wp[i] = (std::uint8_t)(r >> 24);
  }
  auto* sp = static_cast<std::uint16_t*>(q.s.contents());
  auto* bp = static_cast<std::uint16_t*>(q.b.contents());
  for (std::size_t i = 0; i < ng; ++i) {
    sp[i] = to_bf16_(0.01f);
    bp[i] = to_bf16_(-0.08f);
  }
  return q;
}

// The routes MetalMiniMaxH3Transformer::GemmRoute can pick. Kept as a
// parallel enum rather than including the model header: this file must run
// without a checkpoint and without constructing a transformer, and what it
// tests is the KERNEL sequence, which is what the model dispatches.
enum class Route { Steel32, Steel64, Steel128, Mma128, Mma256, MmaTn2 };

const char* route_tag_(Route r)
{
  switch (r) {
    case Route::Steel32:  return "bm32";
    case Route::Steel64:  return "bm64";
    case Route::Steel128: return "bm128";
    case Route::Mma128:   return "mma128";
    case Route::Mma256:   return "mma128x256";
    case Route::MmaTn2:   return "mma-tn2";
  }
  return "?";
}

bool is_mma_(Route r)
{
  return r == Route::Mma128 || r == Route::Mma256 || r == Route::MmaTn2;
}

struct Kernels {
  ComputeLibrary lib_steel, lib_dq, lib_dense, lib_attn, lib_attn_nax;
  ComputeFunction steel[2], steel64[2], steel128[2];   // [0]=w4 [1]=w8
  ComputeFunction dq[2];
  ComputeFunction dense128, dense256, dense_tn2;
  bool have_mma = false;

  void load(MetalCompute* mc)
  {
    // The _bf16 twins, because H3 is a bf16 model -- the f16 libraries are
    // the same sources built for the other element type.
    lib_steel = mc->load_library("affine_qmm_steel_bf16");
    lib_dq    = mc->load_library("affine_dequant_bf16");
    lib_dense = mc->load_library("dense_gemm_mma_bf16");
    lib_attn  = mc->load_library("attn_steel");
    lib_attn_nax = mc->load_library("attn_steel_nax");
    steel[0]    = lib_steel.function("affine_qmm_steel_w4g64");
    steel[1]    = lib_steel.function("affine_qmm_steel_w8g64");
    steel64[0]  = lib_steel.function("affine_qmm_steel_w4g64_bm64");
    steel64[1]  = lib_steel.function("affine_qmm_steel_w8g64_bm64");
    steel128[0] = lib_steel.function("affine_qmm_steel_w4g64_bm128");
    steel128[1] = lib_steel.function("affine_qmm_steel_w8g64_bm128");
    dq[0]      = lib_dq.function("affine_dequant_w4g64");
    dq[1]      = lib_dq.function("affine_dequant_w8g64");
    dense128   = lib_dense.function("dense_gemm_mma_t_n128_f16");
    dense256   = lib_dense.function("dense_gemm_mma_t_n128x256_f16");
    dense_tn2  = lib_dense.function("dense_gemm_mma_t_n128x256_tn2_f16");
    have_mma = mc->supports_matrix_cores() && dense128.valid()
               && dense256.valid() && dq[0].valid();
  }

  bool available(Route r, int bits) const
  {
    const int bi = (bits == 8) ? 1 : 0;
    switch (r) {
      case Route::Steel32:  return steel[bi].valid();
      case Route::Steel64:  return steel64[bi].valid();
      case Route::Steel128: return steel128[bi].valid();
      case Route::Mma128:   return have_mma && dq[bi].valid();
      case Route::Mma256:   return have_mma && dq[bi].valid();
      case Route::MmaTn2:   return have_mma && dq[bi].valid()
                                   && dense_tn2.valid();
    }
    return false;
  }
};

// One projection through one route -- the same kernel sequence
// MetalMiniMaxH3Transformer::gemm_route_dispatch_ encodes, with the bias
// pass omitted (it is identical across routes and would only add noise).
void encode_(ComputeEncoder& enc, const Kernels& kn, Route r, const QW& q,
             const SharedBuffer& x, const SharedBuffer& y,
             const SharedBuffer& wdq, int M, int N, int K, int bits)
{
  const int bi = (bits == 8) ? 1 : 0;
  if (is_mma_(r)) {
    enc.set_function(kn.dq[bi]);
    enc.set_buffer(0, q.w);
    enc.set_buffer(1, q.s);
    enc.set_buffer(2, q.b);
    enc.set_buffer(3, wdq);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.dispatch({(unsigned)(K * bits / 32), (unsigned)N, 1}, {64, 1, 1});
    int RN = 128;
    const ComputeFunction* fn = &kn.dense128;
    if (r == Route::Mma256) { fn = &kn.dense256; RN = 256; }
    else if (r == Route::MmaTn2) { fn = &kn.dense_tn2; RN = 512; }
    enc.set_function(*fn);
    enc.set_buffer(0, x);
    enc.set_buffer(1, wdq);
    enc.set_buffer(2, wdq);          // bias slot unused (has_bias = 0)
    enc.set_buffer(3, y);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    return;
  }
  const int bm = (r == Route::Steel128) ? 128 : (r == Route::Steel64) ? 64 : 32;
  enc.set_function(r == Route::Steel128 ? kn.steel128[bi]
                   : r == Route::Steel64 ? kn.steel64[bi]
                                         : kn.steel[bi]);
  enc.set_buffer(0, q.w);
  enc.set_buffer(1, q.s);
  enc.set_buffer(2, q.b);
  enc.set_buffer(3, x);
  enc.set_buffer(4, y);
  enc.set_constant(5, K);
  enc.set_constant(6, N);
  enc.set_constant(7, M);
  const unsigned tgz = (bm == 128) ? 4u : 2u;
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), tgz}, {32, 2, tgz});
}

double rel_l2_(const SharedBuffer& a, const SharedBuffer& b, std::size_t n)
{
  const auto* pa = static_cast<const std::uint16_t*>(a.contents());
  const auto* pb = static_cast<const std::uint16_t*>(b.contents());
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double x = from_bf16_(pa[i]), y = from_bf16_(pb[i]);
    num += (x - y) * (x - y);
    den += x * x;
  }
  return den > 0.0 ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
}

}  // namespace

// Every block projection, every route it could take, at H3's real shapes.
//
// The arms ALTERNATE and both are warmed before either is timed. That is not
// fussiness: this box has 4-5% of spread between processes and the effects
// here start at a few percent, and the OptiQ audit shipped two "wins" that
// were nothing but the first arm absorbing first-touch cost while the two
// sat in different SoC power states.
TEST(minimax_h3_blocks, gemm_route_sweep)
{
  if (!bench_on_()) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.steel[0].valid()) { return; }

  const int bits = 4;   // the width a 16 GB box can hold the DiT at
  std::vector<int> rows;
  if (bench_full_()) {
    rows.assign(std::begin(kRowsFull), std::end(kRowsFull));
  } else {
    rows.assign(std::begin(kRowsDefault), std::end(kRowsDefault));
  }
  const Route all[] = {Route::Steel32, Route::Steel64, Route::Steel128,
                       Route::Mma128,  Route::Mma256,  Route::MmaTn2};

  std::printf("[h3_blocks] matrix cores: %s\n", kn.have_mma ? "yes" : "no");
  // One scratch for the widest weight (fc1: 28672 x 5376 bf16 = 308 MB),
  // shared by every mma arm exactly as the model shares it across a block.
  std::size_t wmax = 0;
  for (const Proj& p : kProjs) {
    wmax = std::max(wmax, (std::size_t)p.N * (std::size_t)p.K * 2);
  }
  SharedBuffer wdq =
      kn.have_mma ? mc->make_shared_buffer(wmax) : SharedBuffer{};

  for (int M : rows) {
    for (const Proj& p : kProjs) {
      QW q = make_qw_(mc, p.N, p.K, bits, (std::uint32_t)(p.N * 31 + p.K));
      if (q.w.empty()) { continue; }
      SharedBuffer x = make_act_(mc, (std::size_t)M * p.K, 7u);
      SharedBuffer y = mc->make_shared_buffer((std::size_t)M * p.N * 2);
      if (x.empty() || y.empty()) { continue; }
      const double flops = 2.0 * M * p.N * p.K;

      // Collect the candidate arms first, then round-robin them, so no arm
      // ever runs all its rounds back to back.
      std::vector<Route> cands;
      for (Route r : all) {
        if (kn.available(r, bits) && !(is_mma_(r) && wdq.empty())) {
          cands.push_back(r);
        }
      }
      if (cands.empty()) { continue; }
      auto once = [&](Route r, int iters) {
        const auto t0 = std::chrono::steady_clock::now();
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          for (int i = 0; i < iters; ++i) {
            encode_(e, kn, r, q, x, y, wdq, M, p.N, p.K, bits);
          }
        }
        st.commit().wait();
        const auto t1 = std::chrono::steady_clock::now();
        return flops * iters / 1e9 / secs_(t0, t1);
      };
      const int iters = std::min(20, std::max(2, (int)(3.0e11 / flops)));
      for (Route r : cands) { once(r, 1); }        // warm ALL before timing ANY
      std::vector<std::vector<double>> g(cands.size());
      for (int round = 0; round < 5; ++round) {
        for (std::size_t i = 0; i < cands.size(); ++i) {
          g[i].push_back(once(cands[i], iters));
        }
      }
      std::printf("[h3_blocks] M=%5d %-4s N=%5d K=%5d |", M, p.tag, p.N, p.K);
      double base = 0.0;
      for (std::size_t i = 0; i < cands.size(); ++i) {
        std::sort(g[i].begin(), g[i].end());
        const double med = g[i][g[i].size() / 2];
        if (i == 0) { base = med; }
        std::printf(" %s %.0f (%.2fx)", route_tag_(cands[i]), med,
                    base > 0.0 ? med / base : 0.0);
      }
      std::printf("\n");
    }
  }
  EXPECT_TRUE(true);
}

// The matrix-core route must AGREE with steel at every shape it serves.
//
// This is the arm that catches a wiring mistake, and it is the reason the
// sweep above is allowed to report rather than assert: a wrong grid, a
// swapped buffer index or an N that the tile addresses past all produce a
// plausible number at a plausible rate. Boogu's TN=2 tail bug was exactly
// this -- rel-L2 0.19 from a store that landed on the following rows, with
// nothing else to show for it.
//
// The bar is loose on purpose. The two arms are genuinely different
// computations: steel dequantizes each weight tile in registers and
// accumulates in f32, while the mma arm materializes the whole weight in
// bf16 first, so it carries one extra rounding per weight element. What a
// wiring error looks like is 1e-1 and up, which is two orders clear of this.
TEST(minimax_h3_blocks, mma_matches_steel)
{
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.steel[0].valid()) { return; }
  if (!kn.have_mma) { return; }   // pre-M5: nothing to compare against

  // 128 rows: past the 128-row tile so every mma tile is live, and small
  // enough that four shapes' outputs stay well inside a 16 GB box.
  const int M = 128;
  SharedBuffer wdq;
  {
    std::size_t wmax = 0;
    for (const Proj& p : kProjs) {
      wmax = std::max(wmax, (std::size_t)p.N * (std::size_t)p.K * 2);
    }
    wdq = mc->make_shared_buffer(wmax);
  }
  ASSERT_TRUE(!wdq.empty());

  bool ran = false;
  for (int bits : {4, 8}) {
    if (!kn.dq[(bits == 8) ? 1 : 0].valid()) { continue; }
    for (const Proj& p : kProjs) {
      QW q = make_qw_(mc, p.N, p.K, bits, (std::uint32_t)(p.N + bits));
      if (q.w.empty()) { continue; }
      SharedBuffer x = make_act_(mc, (std::size_t)M * p.K, 11u);
      const std::size_t on = (std::size_t)M * p.N;
      SharedBuffer y_ref = mc->make_shared_buffer(on * 2);
      SharedBuffer y_mma = mc->make_shared_buffer(on * 2);
      if (x.empty() || y_ref.empty() || y_mma.empty()) { continue; }

      for (Route r : {Route::Steel32, Route::Mma128, Route::Mma256,
                      Route::MmaTn2}) {
        if (!kn.available(r, bits)) { continue; }
        const SharedBuffer& dst = (r == Route::Steel32) ? y_ref : y_mma;
        std::memset(dst.contents(), 0, on * 2);
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          encode_(e, kn, r, q, x, dst, wdq, M, p.N, p.K, bits);
        }
        st.commit().wait();
        if (r == Route::Steel32) { continue; }
        const double rl2 = rel_l2_(y_ref, y_mma, on);
        std::printf("[h3_blocks] w%d %-4s N=%5d K=%5d %-11s rel-L2 %.3e\n",
                    bits, p.tag, p.N, p.K, route_tag_(r), rl2);
        EXPECT_TRUE(rl2 < 3e-2);
        ran = true;
      }
    }
  }
  EXPECT_TRUE(ran);
}

// The NAX flash attention must agree with the ALU steel one at H3's real
// attention shape, and the two must be interchangeable at the call site.
//
// H3 is the first model here to take the bd128 bf16 NAX entry at 56 heads,
// and the tile change (bq 32->64, bk 16->32) moves BOTH the param block and
// the dispatch grid. Getting one and not the other is the failure this
// catches: it does not crash, it computes attention over the wrong block
// boundaries.
TEST(minimax_h3_blocks, attn_nax_matches_steel)
{
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.lib_attn.valid()) { return; }
  if (!mc->supports_matrix_cores() || !kn.lib_attn_nax.valid()) { return; }

  // C++ mirror of mlx::steel::AttnParams, as in the model.
  struct P {
    int B, H, D, qL, kL, gqa_factor;
    float scale;
    int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
    std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
  };
  // 602 is a real short packed sequence and is a multiple of NEITHER tile,
  // so both kernels run their ragged-tail path -- which is where a tile/param
  // mismatch shows up. 2048 is aligned for both, as the control.
  //
  // The head_dim-64 rows are the VIDEO VAE's ViT (32 heads x 64, bf16). That
  // combination had no NAX entry point until this model needed one -- NAX
  // shipped bd64 in f16 and bd128 in bf16 -- so this arm is also what proves
  // the new instantiation is real. An unvalidated ComputeFunction is a
  // silent no-op in this framework, which would leave `o_nax` at the zeros
  // it was memset to; a zero output cannot pass a rel-L2 against a nonzero
  // reference, so that failure mode is covered here rather than assumed.
  struct AttnShape { int seq, nh, hd; const char* who; };
  const AttnShape shapes[] = {
      { 602, kHeads,  kHeadDim, "dit"},
      {2048, kHeads,  kHeadDim, "dit"},
      { 602,     32,        64, "vvae"},
      {2048,     32,        64, "vvae"},
  };
  bool ran = false;
  for (const AttnShape& sh : shapes) {
    const int seq = sh.seq, NH = sh.nh, HD = sh.hd;
    const char* entry = (HD == 64) ? "attn_steel_nax_h_bd64_bf16"
                                   : "attn_steel_nax_h_bd128_bf16";
    const char* entry_alu = (HD == 64) ? "attn_steel_h_bd64_bf16"
                                       : "attn_steel_h_bd128_bf16";
    const std::size_t n = (std::size_t)NH * seq * HD;
    SharedBuffer q = make_act_(mc, n, 3u);
    SharedBuffer k = make_act_(mc, n, 5u);
    SharedBuffer v = make_act_(mc, n, 9u);
    SharedBuffer o_ref = mc->make_shared_buffer(n * 2);
    SharedBuffer o_nax = mc->make_shared_buffer(n * 2);
    SharedBuffer pb = mc->make_shared_buffer(sizeof(float) * 64);
    if (q.empty() || o_ref.empty() || o_nax.empty() || pb.empty()) { continue; }
    const float scale = 1.0f / std::sqrt((float)HD);

    for (int arm = 0; arm < 2; ++arm) {
      const bool nax = (arm == 1);
      const int BQ = nax ? 64 : 32, BK = nax ? 32 : 16;
      auto* p = static_cast<P*>(pb.contents());
      p->B = 1; p->H = NH; p->D = HD;
      p->qL = seq; p->kL = seq;
      p->gqa_factor = 1; p->scale = scale;
      p->NQ = (seq + BQ - 1) / BQ; p->NK = (seq + BK - 1) / BK;
      p->NQ_aligned = seq / BQ; p->NK_aligned = seq / BK;
      p->qL_rem = seq - p->NQ_aligned * BQ;
      p->kL_rem = seq - p->NK_aligned * BK;
      p->qL_off = 0;
      p->Q_strides[0] = (std::int64_t)NH * seq * HD;
      p->Q_strides[1] = (std::int64_t)seq * HD;
      p->Q_strides[2] = HD;
      for (int i = 0; i < 3; ++i) {
        p->K_strides[i] = p->Q_strides[i];
        p->V_strides[i] = p->Q_strides[i];
        p->O_strides[i] = p->Q_strides[i];
      }
      FunctionConstants fc;
      fc.set_bool(200, (seq % BQ) == 0).set_bool(201, (seq % BK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      ComputeFunction fn = nax ? kn.lib_attn_nax.function(entry, fc)
                               : kn.lib_attn.function(entry_alu, fc);
      // Not `continue`: a NAX entry that does not build is the thing this
      // test exists to catch, so it fails rather than skipping quietly.
      EXPECT_TRUE(fn.valid());
      if (!fn.valid()) { continue; }
      const SharedBuffer& dst = nax ? o_nax : o_ref;
      std::memset(dst.contents(), 0, n * 2);
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        e.set_function(fn);
        e.set_buffer(0, q); e.set_buffer(1, k);
        e.set_buffer(2, v); e.set_buffer(3, dst);
        e.set_buffer(4, pb);
        e.dispatch({32 * (unsigned)((seq + BQ - 1) / BQ), 4 * (unsigned)NH, 1},
                   {32, 4, 1});
      }
      st.commit().wait();
      ran = ran || nax;
    }
    const double rl2 = rel_l2_(o_ref, o_nax, n);
    std::printf("[h3_blocks] attn %-4s seq=%4d heads=%2d hd=%3d  nax-vs-steel "
                "rel-L2 %.3e\n", sh.who, seq, NH, HD, rl2);
    // Both accumulate the softmax in f32 and differ only in tiling, so this
    // is bf16 store rounding and nothing else. A tile/param mismatch scores
    // O(1) here -- it is attending over the wrong rows, not rounding badly.
    EXPECT_TRUE(rl2 < 5e-2);
  }
  EXPECT_TRUE(ran);
}

// Rate of the two attentions at the shapes H3 actually runs. Reported, not
// asserted: attention is 23% of a step and quadratic in the sequence, so
// what matters is the ratio at REAL lengths, and a bar on it would flake on
// this box's power-budget clock.
TEST(minimax_h3_blocks, attn_nax_rate)
{
  if (!bench_on_()) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.lib_attn.valid() || !mc->supports_matrix_cores()
      || !kn.lib_attn_nax.valid()) {
    return;
  }
  struct P {
    int B, H, D, qL, kL, gqa_factor;
    float scale;
    int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
    std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
  };
  const int NH = kHeads, HD = kHeadDim;
  std::vector<int> seqs = bench_full_()
      ? std::vector<int>{602, 2304, 4282, 9382, 19008}
      : std::vector<int>{602, 9382};
  for (int seq : seqs) {
    const std::size_t n = (std::size_t)NH * seq * HD;
    SharedBuffer q = make_act_(mc, n, 3u);
    SharedBuffer k = make_act_(mc, n, 5u);
    SharedBuffer v = make_act_(mc, n, 9u);
    SharedBuffer o = mc->make_shared_buffer(n * 2);
    SharedBuffer pb[2] = {mc->make_shared_buffer(sizeof(float) * 64),
                          mc->make_shared_buffer(sizeof(float) * 64)};
    if (q.empty() || o.empty() || pb[0].empty() || pb[1].empty()) { continue; }
    const float scale = 1.0f / std::sqrt((float)HD);
    ComputeFunction fn[2];
    for (int arm = 0; arm < 2; ++arm) {
      const bool nax = (arm == 1);
      const int BQ = nax ? 64 : 32, BK = nax ? 32 : 16;
      auto* p = static_cast<P*>(pb[arm].contents());
      p->B = 1; p->H = NH; p->D = HD;
      p->qL = seq; p->kL = seq;
      p->gqa_factor = 1; p->scale = scale;
      p->NQ = (seq + BQ - 1) / BQ; p->NK = (seq + BK - 1) / BK;
      p->NQ_aligned = seq / BQ; p->NK_aligned = seq / BK;
      p->qL_rem = seq - p->NQ_aligned * BQ;
      p->kL_rem = seq - p->NK_aligned * BK;
      p->qL_off = 0;
      p->Q_strides[0] = (std::int64_t)NH * seq * HD;
      p->Q_strides[1] = (std::int64_t)seq * HD;
      p->Q_strides[2] = HD;
      for (int i = 0; i < 3; ++i) {
        p->K_strides[i] = p->Q_strides[i];
        p->V_strides[i] = p->Q_strides[i];
        p->O_strides[i] = p->Q_strides[i];
      }
      FunctionConstants fc;
      fc.set_bool(200, (seq % BQ) == 0).set_bool(201, (seq % BK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      fn[arm] = nax
          ? kn.lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
          : kn.lib_attn.function("attn_steel_h_bd128_bf16", fc);
    }
    if (!fn[0].valid() || !fn[1].valid()) { continue; }
    // 4 * seq^2 * HD * NH: QK^T and P*V, two FLOPs each.
    const double flops = 4.0 * (double)seq * seq * HD * NH;
    const int iters = std::min(20, std::max(2, (int)(3.0e11 / flops)));
    auto once = [&](int arm, int it) {
      const int BQ = (arm == 1) ? 64 : 32;
      const auto t0 = std::chrono::steady_clock::now();
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        for (int i = 0; i < it; ++i) {
          e.set_function(fn[arm]);
          e.set_buffer(0, q); e.set_buffer(1, k);
          e.set_buffer(2, v); e.set_buffer(3, o);
          e.set_buffer(4, pb[arm]);
          e.dispatch({32 * (unsigned)((seq + BQ - 1) / BQ),
                      4 * (unsigned)NH, 1}, {32, 4, 1});
        }
      }
      st.commit().wait();
      const auto t1 = std::chrono::steady_clock::now();
      return flops * it / 1e9 / secs_(t0, t1);
    };
    once(0, 1); once(1, 1);                    // warm both before timing either
    std::vector<double> a, b;
    for (int r = 0; r < 5; ++r) {
      a.push_back(once(0, iters));
      b.push_back(once(1, iters));
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    const double ga = a[a.size() / 2], gb = b[b.size() / 2];
    std::printf("[h3_blocks] attn seq=%5d | steel %6.0f GF/s | nax %6.0f "
                "GF/s | %.2fx\n", seq, ga, gb, ga > 0.0 ? gb / ga : 0.0);
  }
  EXPECT_TRUE(true);
}
