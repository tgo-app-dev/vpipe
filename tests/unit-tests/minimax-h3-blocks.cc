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
#include "generative-models/shared/mma-tile.h"
#include "stages/gpu-telemetry.h"

#include <algorithm>
#include <chrono>
#include <functional>
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
  ComputeLibrary lib_gemm;
  ComputeFunction steel[2], steel64[2], steel128[2];   // [0]=w4 [1]=w8
  ComputeFunction dq[2];
  ComputeFunction dense128, dense256, dense_tn2;
  // The dense (unquantized) GEMMs the runtime LoRA rides: the plain steel
  // tile, its accumulating twin, and the 64-wide matmul2d entry -- the only
  // mma tile narrow enough to fit a rank-64 output without half of it
  // hanging past N.
  ComputeFunction gemm_bm64, gemm_bm64_acc, dense64;
  // The LoRA-shaped matmul2d tiles: a scaled store for the rank-wide A
  // half, an accumulating seed for the rank-deep B half.
  ComputeFunction dense64_sc, dense128_sc, dense128_acc, dense256_acc;
  // The base tile that also carries the adapter's second factor.
  ComputeFunction dense128_lora, dense256_lora;
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
    dense64    = lib_dense.function("dense_gemm_mma_t_f16");
    dense64_sc   = lib_dense.function("dense_gemm_mma_t_scaled_f16");
    dense128_sc  = lib_dense.function("dense_gemm_mma_t_n128_scaled_f16");
    dense128_acc = lib_dense.function("dense_gemm_mma_t_n128_acc_f16");
    dense256_acc = lib_dense.function("dense_gemm_mma_t_n128x256_acc_f16");
    dense128_lora = lib_dense.function("dense_gemm_mma_t_n128_lora_f16");
    dense256_lora =
        lib_dense.function("dense_gemm_mma_t_n128x256_lora_f16");
    lib_gemm      = mc->load_library("dense_gemm_bf16");
    gemm_bm64     = lib_gemm.function("dense_gemm_t_bm64_f16");
    gemm_bm64_acc = lib_gemm.function("dense_gemm_t_bm64_acc_f16");
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

    for (int arm = 0; arm < 3; ++arm) {
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
    for (int arm = 0; arm < 3; ++arm) {
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

// ---- the SwiGLU epilogue the H3 block does NOT fuse ---------------------
//
// H3's feed-forward runs ONE fused-width GEMM (N = 2*ffn = 28672) into a
// [rows, 2*ffn] buffer and then a SEPARATE elementwise pass that reads it
// back and writes silu(gate)*up at [rows, ffn]. Every other quantized DiT
// here -- Krea-2, FLUX.2 -- folds that epilogue into the GEMM's
// register-local store instead (`affine_qmm_swiglu`), so the double-width
// intermediate is never written and never re-read.
//
// The two arms below are exactly those, at H3's real fc1 shape. The
// difference is bandwidth, not arithmetic: per block the unfused path
// stores rows*2*ffn, reads it back, and writes rows*ffn, where the fused
// one stores rows*ffn once.
//
// What the fusion COSTS at the call site is a load-time INTERLEAVE. The
// fused epilogue takes its (gate, up) pair out of one accumulator
// fragment -- even column gate, odd column up -- so it needs the weight
// as [g0,u0,g1,u1,...] where H3's checkpoint stores [all gate | all up].
// The agreement arm builds both layouts from the same codes, which is
// what makes "a permutation, not a different computation" a measured
// claim rather than a reading of the kernel.

namespace {

// Interleave a fused [gate block | up block] quantized weight into the
// [g0,u0,g1,u1,...] pairing the fused epilogue reads.
//
// Rows are the OUTER dimension of all three arrays, so this is whole-row
// copies -- codes and the per-group scales/biases move together and no
// group is ever split. That is the whole reason the interleave is a load
// -time transform and not a kernel change.
QW interleave_gu_(MetalCompute* mc, const QW& src, int N, int K, int bits)
{
  const std::size_t row_codes = (std::size_t)K * bits / 8;
  const std::size_t row_grp   = (std::size_t)K / 64;
  QW out{mc->make_shared_buffer((std::size_t)N * row_codes),
         mc->make_shared_buffer((std::size_t)N * row_grp * 2),
         mc->make_shared_buffer((std::size_t)N * row_grp * 2)};
  if (out.w.empty() || out.s.empty() || out.b.empty()) { return QW{}; }
  const auto* sw = static_cast<const std::uint8_t*>(src.w.contents());
  const auto* ss = static_cast<const std::uint16_t*>(src.s.contents());
  const auto* sb = static_cast<const std::uint16_t*>(src.b.contents());
  auto* dw = static_cast<std::uint8_t*>(out.w.contents());
  auto* ds = static_cast<std::uint16_t*>(out.s.contents());
  auto* db = static_cast<std::uint16_t*>(out.b.contents());
  const int half = N / 2;
  for (int j = 0; j < half; ++j) {
    const int from[2] = {j, half + j};        // gate row, then up row
    for (int h = 0; h < 2; ++h) {
      const std::size_t d = (std::size_t)(2 * j + h);
      std::memcpy(dw + d * row_codes, sw + (std::size_t)from[h] * row_codes,
                  row_codes);
      std::memcpy(ds + d * row_grp, ss + (std::size_t)from[h] * row_grp,
                  row_grp * 2);
      std::memcpy(db + d * row_grp, sb + (std::size_t)from[h] * row_grp,
                  row_grp * 2);
    }
  }
  return out;
}

struct FfKernels {
  ComputeLibrary lib_qmm, lib_elt;
  ComputeFunction steel[3];     // [0]=bm32 [1]=bm64 [2]=bm128
  ComputeFunction fused[3];
  ComputeFunction split;        // the separate elementwise pass

  void load(MetalCompute* mc)
  {
    lib_qmm = mc->load_library("affine_qmm_steel_bf16");
    lib_elt = mc->load_library("llm_elementwise_bf16");
    steel[0] = lib_qmm.function("affine_qmm_steel_w4g64");
    steel[1] = lib_qmm.function("affine_qmm_steel_w4g64_bm64");
    steel[2] = lib_qmm.function("affine_qmm_steel_w4g64_bm128");
    fused[0] = lib_qmm.function("affine_qmm_swiglu_w4g64");
    fused[1] = lib_qmm.function("affine_qmm_swiglu_w4g64_bm64");
    fused[2] = lib_qmm.function("affine_qmm_swiglu_w4g64_bm128");
    split    = lib_elt.function("swiglu_split_gate_first_f16");
  }
  bool ok() const
  {
    return steel[0].valid() && fused[0].valid() && split.valid();
  }
};

const int kBm[3] = {32, 64, 128};

// One quantized GEMM at tile `ti`, fused or not. The grids are the ones
// the models encode: z is 4 at BM=128 (WM=4) and 2 below it.
void encode_qmm_(ComputeEncoder& e, const ComputeFunction& fn, const QW& q,
                 const SharedBuffer& x, const SharedBuffer& y, int M, int N,
                 int K, int ti)
{
  e.set_function(fn);
  e.set_buffer(0, q.w); e.set_buffer(1, q.s); e.set_buffer(2, q.b);
  e.set_buffer(3, x); e.set_buffer(4, y);
  e.set_constant(5, K); e.set_constant(6, N); e.set_constant(7, M);
  const unsigned tgz = (kBm[ti] == 128) ? 4u : 2u;
  e.dispatch({(unsigned)(((N + 31) / 32) * 32),
              (unsigned)(((M + kBm[ti] - 1) / kBm[ti]) * 2), tgz},
             {32, 2, tgz});
}

}  // namespace

// The fused epilogue must compute what the GEMM-then-elementwise pair
// computes, given the interleaved weight. Always runs: this is the arm
// that says the saving below is available, and it is cheap.
TEST(minimax_h3_blocks, fused_swiglu_matches_split)
{
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  FfKernels kn;
  kn.load(mc);
  if (!kn.ok()) { return; }

  const int N = 2 * kFfn, K = kHidden, M = 64;
  QW blk = make_qw_(mc, N, K, 4, 4242u);
  if (blk.w.empty()) { return; }
  QW inter = interleave_gu_(mc, blk, N, K, 4);
  if (inter.w.empty()) { return; }
  SharedBuffer x  = make_act_(mc, (std::size_t)M * K, 13u);
  SharedBuffer wide = mc->make_shared_buffer((std::size_t)M * N * 2);
  SharedBuffer ref  = mc->make_shared_buffer((std::size_t)M * kFfn * 2);
  SharedBuffer fus  = mc->make_shared_buffer((std::size_t)M * kFfn * 2);
  if (x.empty() || wide.empty() || ref.empty() || fus.empty()) { return; }

  CommandStream st = mc->make_command_stream();
  {
    ComputeEncoder e = st.begin_compute();
    encode_qmm_(e, kn.steel[0], blk, x, wide, M, N, K, 0);
    e.set_function(kn.split);
    e.set_buffer(0, wide); e.set_buffer(1, ref);
    e.set_constant(2, M); e.set_constant(3, kFfn);
    e.dispatch({(unsigned)(M * kFfn), 1, 1}, {256, 1, 1});
    encode_qmm_(e, kn.fused[0], inter, x, fus, M, N, K, 0);
  }
  st.commit().wait();

  const double rl2 = rel_l2_(ref, fus, (std::size_t)M * kFfn);
  std::printf("[h3_blocks] fused-swiglu vs gemm+split  rel-L2 %.3e\n", rl2);
  // The arms differ by exactly one rounding: the split path stores the
  // GEMM's gate and up as bf16 and reads them back, the fused one keeps
  // them in the accumulator. A gate/up mix-up -- the interleave written
  // the other way round -- scores O(1) here, not O(1e-3).
  EXPECT_TRUE(rl2 < 3e-2);
}

// What the fusion is worth at H3's real fc1 shape.
//
// Reported, not asserted, for the same reason the GEMM sweep is: this is
// a rate on a laptop with a power-budget clock. The arms alternate and
// both are warmed before either is timed.
TEST(minimax_h3_blocks, fused_swiglu_rate)
{
  if (!bench_on_()) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  FfKernels kn;
  kn.load(mc);
  if (!kn.ok()) { return; }

  const int N = 2 * kFfn, K = kHidden;
  QW blk = make_qw_(mc, N, K, 4, 4242u);
  if (blk.w.empty()) { return; }
  QW inter = interleave_gu_(mc, blk, N, K, 4);
  if (inter.w.empty()) { return; }

  std::vector<int> rows;
  if (bench_full_()) {
    rows.assign(std::begin(kRowsFull), std::end(kRowsFull));
  } else {
    rows.assign(std::begin(kRowsDefault), std::end(kRowsDefault));
  }

  for (int M : rows) {
    SharedBuffer x    = make_act_(mc, (std::size_t)M * K, 13u);
    SharedBuffer wide = mc->make_shared_buffer((std::size_t)M * N * 2);
    SharedBuffer out  = mc->make_shared_buffer((std::size_t)M * kFfn * 2);
    if (x.empty() || wide.empty() || out.empty()) { continue; }

    // The unfused arm is the GEMM plus the elementwise pass, because that
    // pair is what the fused kernel replaces -- timing the GEMM alone
    // would credit the fusion with a saving it does not make.
    auto unfused = [&](int ti, int iters) {
      const auto t0 = std::chrono::steady_clock::now();
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        for (int i = 0; i < iters; ++i) {
          encode_qmm_(e, kn.steel[ti], blk, x, wide, M, N, K, ti);
          e.set_function(kn.split);
          e.set_buffer(0, wide); e.set_buffer(1, out);
          e.set_constant(2, M); e.set_constant(3, kFfn);
          e.dispatch({(unsigned)(M * kFfn), 1, 1}, {256, 1, 1});
        }
      }
      st.commit().wait();
      const auto t1 = std::chrono::steady_clock::now();
      return secs_(t0, t1) * 1e3 / iters;
    };
    auto fused = [&](int ti, int iters) {
      const auto t0 = std::chrono::steady_clock::now();
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        for (int i = 0; i < iters; ++i) {
          encode_qmm_(e, kn.fused[ti], inter, x, out, M, N, K, ti);
        }
      }
      st.commit().wait();
      const auto t1 = std::chrono::steady_clock::now();
      return secs_(t0, t1) * 1e3 / iters;
    };

    const double flops = 2.0 * M * N * K;
    const int iters = std::min(20, std::max(2, (int)(3.0e11 / flops)));
    std::vector<int> tiles;
    for (int ti = 0; ti < 3; ++ti) {
      if (kn.steel[ti].valid() && kn.fused[ti].valid()) { tiles.push_back(ti); }
    }
    for (int ti : tiles) { unfused(ti, 1); fused(ti, 1); }   // warm both

    double best_u = 1e30, best_f = 1e30;
    int bu = 0, bf = 0;
    for (int ti : tiles) {
      std::vector<double> u, f;
      for (int r = 0; r < 5; ++r) {
        u.push_back(unfused(ti, iters));
        f.push_back(fused(ti, iters));
      }
      std::sort(u.begin(), u.end());
      std::sort(f.begin(), f.end());
      const double mu = u[u.size() / 2], mf = f[f.size() / 2];
      std::printf("[h3_blocks] ff M=%5d bm%-3d | split %7.2f ms | fused "
                  "%7.2f ms | %.3fx\n", M, kBm[ti], mu, mf,
                  mf > 0.0 ? mu / mf : 0.0);
      if (mu < best_u) { best_u = mu; bu = ti; }
      if (mf < best_f) { best_f = mf; bf = ti; }
    }
    // Best-of-arm against best-of-arm: the tile the model would pick for
    // each path, not the tile that flatters the comparison. Extrapolated
    // by the 50 main blocks, since the FF runs once per block per step.
    std::printf("[h3_blocks] ff M=%5d BEST | split bm%d %7.2f ms | fused "
                "bm%d %7.2f ms | %.3fx | %.0f ms/step over %d blocks\n",
                M, kBm[bu], best_u, kBm[bf], best_f,
                best_f > 0.0 ? best_u / best_f : 0.0,
                (best_u - best_f) * 50.0, 50);
  }
  EXPECT_TRUE(true);
}

// Does the fusion's 3% move with the shape?
//
// The two terms scale differently, so it should. What the fusion removes
// is the intermediate -- M*N stored, read back, and half of it written
// again -- and that grows with M*N. What it hides behind is the GEMM,
// which grows with M*N*K. So the RATIO should fall as 1/K and be flat in
// both M and N, and fc1's own K (hidden = 5376) is the only one the
// model gets: a checkpoint with a wider residual stream would see less
// of this, a narrower one more.
//
// Worth measuring rather than deriving. The prediction assumes the GEMM
// is compute-bound and its stores hide under the math, which is true at
// 87% of this box's ALU roofline and would stop being true on hardware
// where it is not -- and a shape rule that was only ever reasoned about
// is how the wrong tile ends up hard-coded.
TEST(minimax_h3_blocks, fused_swiglu_shape_sweep)
{
  if (!bench_on_()) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  FfKernels kn;
  kn.load(mc);
  if (!kn.ok()) { return; }

  // One row count throughout: M was swept in fused_swiglu_rate and came
  // back flat, so holding it fixed here isolates the other two axes.
  const int M = 4282;

  // bm64 alone -- it won at every shape in both of the other sweeps, and
  // three tiles per point would triple a run that is already minutes.
  auto measure = [&](int N, int K) {
    QW blk = make_qw_(mc, N, K, 4, (std::uint32_t)(N * 7 + K));
    if (blk.w.empty()) { return; }
    QW inter = interleave_gu_(mc, blk, N, K, 4);
    if (inter.w.empty()) { return; }
    SharedBuffer x    = make_act_(mc, (std::size_t)M * K, 13u);
    SharedBuffer wide = mc->make_shared_buffer((std::size_t)M * N * 2);
    SharedBuffer out  = mc->make_shared_buffer((std::size_t)M * (N / 2) * 2);
    if (x.empty() || wide.empty() || out.empty()) { return; }
    const int half = N / 2;
    auto split_arm = [&](int iters) {
      const auto t0 = std::chrono::steady_clock::now();
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        for (int i = 0; i < iters; ++i) {
          encode_qmm_(e, kn.steel[1], blk, x, wide, M, N, K, 1);
          e.set_function(kn.split);
          e.set_buffer(0, wide); e.set_buffer(1, out);
          e.set_constant(2, M); e.set_constant(3, half);
          e.dispatch({(unsigned)(M * half), 1, 1}, {256, 1, 1});
        }
      }
      st.commit().wait();
      return secs_(t0, std::chrono::steady_clock::now()) * 1e3 / iters;
    };
    auto fused_arm = [&](int iters) {
      const auto t0 = std::chrono::steady_clock::now();
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        for (int i = 0; i < iters; ++i) {
          encode_qmm_(e, kn.fused[1], inter, x, out, M, N, K, 1);
        }
      }
      st.commit().wait();
      return secs_(t0, std::chrono::steady_clock::now()) * 1e3 / iters;
    };
    const double flops = 2.0 * M * N * K;
    const int iters = std::min(20, std::max(2, (int)(3.0e11 / flops)));
    split_arm(1); fused_arm(1);
    std::vector<double> u, f;
    for (int r = 0; r < 5; ++r) {
      u.push_back(split_arm(iters));
      f.push_back(fused_arm(iters));
    }
    std::sort(u.begin(), u.end());
    std::sort(f.begin(), f.end());
    const double mu = u[u.size() / 2], mf = f[f.size() / 2];
    std::printf("[h3_blocks] shape M=%d N=%5d K=%5d | split %7.2f ms | "
                "fused %7.2f ms | %.3fx%s\n", M, N, K, mu, mf,
                mf > 0.0 ? mu / mf : 0.0,
                (N == 2 * kFfn && K == kHidden) ? "   <- H3's fc1" : "");
  };

  // K at H3's own fc1 width: an eighth of the residual stream up to four
  // times it.
  for (int K : {672, 1344, 2688, 5376, 10752, 21504}) {
    measure(2 * kFfn, K);
  }
  // N at H3's own K. Every value is a multiple of 64 so the fused
  // epilogue's N % BN == 0 requirement holds.
  for (int N : {3584, 7168, 14336, 28672, 57344}) {
    measure(N, kHidden);
  }
  EXPECT_TRUE(true);
}

// The DOWN projection across {M, K}, every tile.
//
// fc2 is the other half of the feed-forward and nothing about it is like
// fc1: N is the residual stream (5376, the NARROWEST output in the
// block) and K is the ffn (14336, the DEEPEST reduction). It carries no
// activation, so there is nothing to fuse -- what it has instead is the
// tile question, and the block's two FF GEMMs sit at opposite corners of
// the shape space, which is exactly the case where one hard-coded tile
// serves one of them badly.
//
// Reported, not asserted. This is a rate on a laptop, and what it is for
// is the autotuner's candidate list: a tile that never wins at any (M,K)
// here is one the tuner is paying to measure for nothing. That is how the
// BM=16 arm was retired -- built, swept, slowest everywhere, removed.
TEST(minimax_h3_blocks, down_proj_shape_sweep)
{
  if (!bench_on_()) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  FfKernels kn;
  kn.load(mc);
  if (!kn.ok()) { return; }

  // One measurement of one (M, N, K) on every tile that built.
  auto sweep = [&](int M, int N, int K, const char* mark) {
    QW q = make_qw_(mc, N, K, 4, (std::uint32_t)(N * 13 + K));
    if (q.w.empty()) { return; }
    SharedBuffer x = make_act_(mc, (std::size_t)M * K, 19u);
    SharedBuffer y = mc->make_shared_buffer((std::size_t)M * N * 2);
    if (x.empty() || y.empty()) { return; }
    const double flops = 2.0 * M * N * K;
    const int iters = std::min(20, std::max(2, (int)(3.0e11 / flops)));
    std::vector<int> tiles;
    for (int ti = 0; ti < 3; ++ti) {
      if (kn.steel[ti].valid()) { tiles.push_back(ti); }
    }
    if (tiles.empty()) { return; }
    auto once = [&](int ti, int it) {
      const auto t0 = std::chrono::steady_clock::now();
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        for (int i = 0; i < it; ++i) {
          encode_qmm_(e, kn.steel[ti], q, x, y, M, N, K, ti);
        }
      }
      st.commit().wait();
      return flops * it / 1e9 /
             secs_(t0, std::chrono::steady_clock::now());
    };
    for (int ti : tiles) { once(ti, 1); }       // warm ALL before timing ANY
    std::vector<std::vector<double>> g(tiles.size());
    for (int r = 0; r < 5; ++r) {
      for (std::size_t i = 0; i < tiles.size(); ++i) {
        g[i].push_back(once(tiles[i], r == 0 ? iters : iters));
      }
    }
    std::printf("[h3_blocks] down M=%5d N=%5d K=%5d |", M, N, K);
    double best = 0.0;
    int    bt = 0;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
      std::sort(g[i].begin(), g[i].end());
      const double med = g[i][g[i].size() / 2];
      std::printf(" bm%-3d %5.0f", kBm[tiles[i]], med);
      if (med > best) { best = med; bt = kBm[tiles[i]]; }
    }
    std::printf(" | best bm%d%s\n", bt, mark);
  };

  // M at the released fc2 shape: the row ladder the DiT actually runs.
  const int N = kHidden, K = kFfn;
  std::vector<int> rows;
  if (bench_full_()) {
    rows.assign(std::begin(kRowsFull), std::end(kRowsFull));
  } else {
    rows.assign(std::begin(kRowsDefault), std::end(kRowsDefault));
  }
  for (int M : rows) {
    sweep(M, N, K, (M == 9382) ? "   <- production layout" : "");
  }
  // K at a fixed, real M. fc2's K is the ffn, so this is the axis a
  // checkpoint with a different expansion ratio moves along -- and the
  // deep-K end is where a small tile's extra weight re-reads are paid
  // most often.
  for (int Ks : {1792, 3584, 7168, 14336, 28672}) {
    sweep(4282, N, Ks, (Ks == kFfn) ? "   <- H3's fc2" : "");
  }
}

// The runtime-LoRA composition: y = W x, then y += B (A x).
//
// Two things are new here and both are silent when wrong. The
// accumulating GEMM folds the EXISTING y into its accumulator instead of
// a bias row, so a mistake there overwrites the base projection rather
// than adding to it -- and an overwritten y is a plausible tensor. And
// the two factors are used in the checkpoint's own orientation, A as
// [rank, K] and B as [N, rank], so a transpose confusion between them
// still type-checks at every shape where rank divides evenly.
//
// Against a CPU reference at H3's real out-projection shape, and at the
// rank the Turbo adapter uses.
namespace {
// Defined with the LoRA route sweep below; declared here so the f64
// reference arm can drive the same dispatch the model does.
void lora_a_(ComputeEncoder& e, const Kernels& kn, bool mma_wide, bool mma,
             const SharedBuffer& x, const SharedBuffer& A,
             const SharedBuffer& t, int M, int r, int K, float scale);
}  // namespace

TEST(minimax_h3_blocks, lora_accumulate_matches_reference)
{
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  ComputeLibrary lib = mc->load_library("dense_gemm_bf16");
  ComputeFunction gemm = lib.function("dense_gemm_t_bm64_f16");
  ComputeFunction acc  = lib.function("dense_gemm_t_bm64_acc_f16");
  if (!gemm.valid() || !acc.valid()) { return; }

  const int M = 96, N = kHidden, K = kInner, R = 64;
  // A strength that is neither 1 nor 0: the two values a scale bug is
  // most likely to survive.
  const float kScale = 0.625f;
  SharedBuffer x = make_act_(mc, (std::size_t)M * K, 21u);
  SharedBuffer W = make_act_(mc, (std::size_t)N * K, 23u);
  SharedBuffer A = make_act_(mc, (std::size_t)R * K, 29u);
  SharedBuffer B = make_act_(mc, (std::size_t)N * R, 31u);
  SharedBuffer t = mc->make_shared_buffer((std::size_t)M * R * 2);
  SharedBuffer y = mc->make_shared_buffer((std::size_t)M * N * 2);
  if (x.empty() || W.empty() || A.empty() || B.empty() || t.empty() ||
      y.empty()) {
    return;
  }
  // Which route runs the adapter. Both are measured against the SAME f64
  // reference below, which is the only way to say which of the two is
  // right -- comparing them to each other says only that they differ.
  // "1" = the matrix-core adapter as two GEMMs, "fused" = its second
  // GEMM absorbed into the base tile. MEASURED against f64 at this
  // shape: steel 2.348e-3, mma pair 2.398e-3, fused 2.398e-3. The fold
  // changes how many times y is stored, so it was worth checking whether
  // it also changes the answer -- it does not, to three digits. What it
  // buys is traffic, not accuracy, and the end-to-end difference between
  // folded and separate (3.4e-3 of velocity) is this model amplifying a
  // last-bit difference, the same way it does for any kernel swap.
  const char* ma = std::getenv("VPIPE_H3_LORA_REF_MMA");
  const bool want_fused = ma != nullptr && std::string(ma) == "fused";
  const bool mma_arm = ma != nullptr && kn.have_mma && kn.dense64_sc.valid();
  const bool fused_arm = want_fused && mma_arm && kn.dense128_lora.valid();
  {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      auto dispatch = [&](const ComputeFunction& fn, const SharedBuffer& xin,
                          const SharedBuffer& Win, const SharedBuffer& yout,
                          int n, int k, float sc) {
        e.set_function(fn);
        e.set_buffer(0, xin); e.set_buffer(1, Win); e.set_buffer(2, Win);
        e.set_buffer(3, yout);
        e.set_constant(4, k); e.set_constant(5, n); e.set_constant(6, M);
        e.set_constant(7, 0);
        // The accumulating twin scales its own product before folding y
        // in; the overwrite twin has no such argument and ignores it.
        e.set_constant(8, sc);
        e.dispatch({(unsigned)(((n + 31) / 32) * 32),
                    (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
      };
      if (fused_arm) {
        // t first: the fused tile consumes it. Then ONE dispatch does
        // y = x W^T + t B^T, so y is stored once instead of twice.
        lora_a_(e, kn, /*mma_wide=*/false, /*mma=*/true, x, A, t, M, R, K,
                kScale);
        e.set_function(kn.dense128_lora);
        e.set_buffer(0, x); e.set_buffer(1, W); e.set_buffer(2, W);
        e.set_buffer(3, y);
        e.set_constant(4, K); e.set_constant(5, N); e.set_constant(6, M);
        e.set_constant(7, 0);
        e.set_buffer(8, t); e.set_buffer(9, B);
        e.set_constant(10, R);
        e.dispatch({(unsigned)(((N + 127) / 128) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      } else if (mma_arm) {
        dispatch(gemm, x, W, y, N, K, 1.0f);    // y  = x W^T
        lora_a_(e, kn, /*mma_wide=*/false, /*mma=*/true, x, A, t, M, R, K,
                kScale);
        dispatch(acc, t, B, y, N, R, 1.0f);     // strength already in t
      } else {
        dispatch(gemm, x, W, y, N, K, 1.0f);    // y  = x W^T
        dispatch(gemm, x, A, t, R, K, 1.0f);    // t  = x A^T
        dispatch(acc,  t, B, y, N, R, kScale);  // y += kScale * t B^T
      }
    }
    st.commit().wait();
  }

  // CPU: base + B @ (A @ x), all in f64 off the same bf16 inputs.
  const auto* xp = static_cast<const std::uint16_t*>(x.contents());
  const auto* Wp = static_cast<const std::uint16_t*>(W.contents());
  const auto* Ap = static_cast<const std::uint16_t*>(A.contents());
  const auto* Bp = static_cast<const std::uint16_t*>(B.contents());
  const auto* yp = static_cast<const std::uint16_t*>(y.contents());
  double num = 0.0, den = 0.0, dnum = 0.0;
  std::vector<double> tr((std::size_t)R);
  for (int i = 0; i < M; ++i) {
    for (int r = 0; r < R; ++r) {
      double v = 0.0;
      for (int k = 0; k < K; ++k) {
        v += (double)from_bf16_(xp[(std::size_t)i * K + k]) *
             (double)from_bf16_(Ap[(std::size_t)r * K + k]);
      }
      tr[(std::size_t)r] = v;
    }
    for (int n = 0; n < N; ++n) {
      double base = 0.0;
      for (int k = 0; k < K; ++k) {
        base += (double)from_bf16_(xp[(std::size_t)i * K + k]) *
                (double)from_bf16_(Wp[(std::size_t)n * K + k]);
      }
      double d = 0.0;
      for (int r = 0; r < R; ++r) {
        d += tr[(std::size_t)r] *
             (double)from_bf16_(Bp[(std::size_t)n * R + r]);
      }
      d *= (double)kScale;
      const double want = base + d;
      const double got = (double)from_bf16_(yp[(std::size_t)i * N + n]);
      num += (got - want) * (got - want);
      den += want * want;
      dnum += d * d;
    }
  }
  const double rl2 = std::sqrt(num / den);
  std::printf("[h3_blocks] lora(%-5s) y=Wx+%.3f*B(Ax) vs f64 reference: "
              "rel-L2 %.3e  (delta is %.1f%% of the output)\n",
              fused_arm ? "fused" : mma_arm ? "mma" : "steel",
              (double)kScale, rl2,
              100.0 * std::sqrt(dnum / den));
  // bf16 stores on t and y, and a f32 accumulator either side.
  EXPECT_TRUE(rl2 < 5e-3);
  // The delta must be a real part of the answer, or the bar above passes
  // on a kernel that ignored B entirely.
  EXPECT_TRUE(std::sqrt(dnum / den) > 0.05);
}

// ---- the runtime LoRA's two GEMMs, against every route that could run --
//
// An adapted projection computes y = W x + s * B (A x), encoded as
//
//   A-GEMM   t[M, r]  = x[M, K] A^T        rank-wide OUTPUT  (N = r)
//   B-GEMM   y[M, N] += s * t[M, r] B^T    rank-deep  INPUT  (K = r)
//
// Both are steel today, and that is not a decision -- the runtime adapter
// was written on a box with no matrix cores. The two halves are skinny in
// DIFFERENT dimensions, so they need separate answers:
//
//   * the A-GEMM reads the whole [M, K] activation to produce [M, 64].
//     2*M*K*r flops against M*K*2 bytes is r flops/byte, so at rank 64 and
//     this machine's ~150 GB/s the CEILING is ~9.6 TFLOP/s -- high enough
//     that the matrix units have room to matter. Steel does not get there:
//     its 32-wide output tile streams x once per tile, i.e. twice at r=64.
//   * the B-GEMM's traffic is dominated by neither operand but by y, which
//     it READS and WRITES in full. 2*M*r*N flops against 2*M*N*2 bytes is
//     r/2 flops/byte -- ~4.8 TFLOP/s at rank 64, and no choice of kernel
//     moves it, because the read-modify-write is the algorithm and not the
//     implementation. Measured anyway: an assumption about which side of a
//     roofline a shape sits on is exactly what this file exists to check.
//
// Both are reported against the BASE projection at the same shape and the
// same M, measured in the same process -- what matters is not the LoRA's
// own rate but what it adds to a step, and the base moved onto matmul2d
// this month, so a ratio quoted from the steel era would be wrong twice.
//
// The `mma` arm on the B-GEMM does NOT accumulate (no such entry exists),
// so it is an upper bound rather than a candidate: it prices the multiply
// without the y read, and a matmul2d arm that cannot beat steel even with
// the dominant traffic REMOVED is one there is no point building.
namespace {

// Rank 64 is the larryvrh Turbo adapter; 128 is what the lightx2v files
// carry on everything except qkv, where three are stacked into 384.
const int kRanks[] = {64, 128, 384};

void lora_a_(ComputeEncoder& e, const Kernels& kn, bool mma_wide, bool mma,
             const SharedBuffer& x, const SharedBuffer& A,
             const SharedBuffer& t, int M, int r, int K, float scale)
{
  e.set_function(mma ? (mma_wide ? kn.dense128_sc : kn.dense64_sc)
                     : kn.gemm_bm64);
  e.set_buffer(0, x);
  e.set_buffer(1, A);
  e.set_buffer(2, A);           // bias slot unused (has_bias = 0)
  e.set_buffer(3, t);
  e.set_constant(4, K);
  e.set_constant(5, r);
  e.set_constant(6, M);
  e.set_constant(7, 0);
  e.set_constant(8, scale);     // the live strength (mma arm only)
  if (mma && mma_wide) {
    e.dispatch({(unsigned)(((r + 127) / 128) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  } else if (mma) {
    e.dispatch({(unsigned)(((r + 63) / 64) * 128),
                (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
  } else {
    e.dispatch({(unsigned)(((r + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
  }
}

// arm: 0 = steel accumulating (what ships), 1 = steel overwriting (prices
// the read-modify-write on its own), 2 = mma 128, 3 = mma 128x256.
void lora_b_(ComputeEncoder& e, const Kernels& kn, int arm,
             const SharedBuffer& t, const SharedBuffer& B,
             const SharedBuffer& y, int M, int N, int r, float scale)
{
  const ComputeFunction* fn = &kn.gemm_bm64_acc;
  if (arm == 1) { fn = &kn.gemm_bm64; }
  else if (arm == 2) { fn = &kn.dense128_acc; }
  else if (arm == 3) { fn = &kn.dense256_acc; }
  e.set_function(*fn);
  e.set_buffer(0, t);
  e.set_buffer(1, B);
  e.set_buffer(2, B);
  e.set_buffer(3, y);
  e.set_constant(4, r);
  e.set_constant(5, N);
  e.set_constant(6, M);
  e.set_constant(7, 0);
  e.set_constant(8, scale);     // ignored by every arm but the first
  if (arm >= 2) {
    const int BN = (arm == 2) ? 128 : 256;
    e.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  } else {
    e.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
  }
}

// Median GFLOP/s over 3 rounds with the clock sampled across the timed
// region. `iters` is sized by the caller for ~0.3 s of GPU work: the
// sampler polls every 85 ms, so a shorter region reports the idle clock
// (~340 MHz) and a thermal dip reads as a slow kernel.
double lrate_(MetalCompute* mc, double f1, int iters, double* mhz,
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
    return f1 * n / 1e9 / secs_(t0, std::chrono::steady_clock::now());
  };
  once(1);
  GpuTelemetrySampler tel;
  tel.start();
  std::vector<double> v;
  for (int r = 0; r < 3; ++r) { v.push_back(once(iters)); }
  const GpuTelemetry t = tel.stop();
  if (mhz != nullptr) { *mhz = t.freq_mhz.ok ? t.freq_mhz.avg : 0.0; }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

TEST(minimax_h3_blocks, lora_route_sweep)
{
  if (!bench_on_()) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.gemm_bm64.valid() || !kn.gemm_bm64_acc.valid()) { return; }
  const bool mma = kn.have_mma && kn.dense64_sc.valid() &&
                   kn.dense128_sc.valid() && kn.dense128_acc.valid() &&
                   kn.dense256_acc.valid();

  std::vector<int> rows;
  if (bench_full_()) {
    rows.assign(std::begin(kRowsFull), std::end(kRowsFull));
  } else {
    rows.assign(std::begin(kRowsDefault), std::end(kRowsDefault));
  }
  // Small enough that 3 rounds x ~400 accumulating iterations stay in
  // range; the arithmetic is identical at any scale.
  const float kScale = 0.0625f;
  const int bits = 4;

  // One allocation per buffer at the widest shape, reused across the
  // sweep -- 19008 rows x fc1's 28672 outputs is already 1.09 GB of y, and
  // re-allocating per shape on a 16 GB box measures the allocator.
  int Mx = 0, Nx = 0, Kx = 0, Rx = 0;
  for (int M : rows) { Mx = std::max(Mx, M); }
  for (const Proj& p : kProjs) {
    Nx = std::max(Nx, p.N);
    Kx = std::max(Kx, p.K);
  }
  for (int r : kRanks) { Rx = std::max(Rx, r); }
  SharedBuffer x = make_act_(mc, (std::size_t)Mx * Kx, 7u);
  SharedBuffer A = make_act_(mc, (std::size_t)Rx * Kx, 29u);
  SharedBuffer B = make_act_(mc, (std::size_t)Nx * Rx, 31u);
  SharedBuffer t = mc->make_shared_buffer((std::size_t)Mx * Rx * 2);
  SharedBuffer y = mc->make_shared_buffer((std::size_t)Mx * Nx * 2);
  SharedBuffer wdq =
      mma ? mc->make_shared_buffer((std::size_t)Nx * Kx * 2) : SharedBuffer{};
  if (x.empty() || A.empty() || B.empty() || t.empty() || y.empty()) {
    std::printf("[h3_blocks] lora sweep: allocation failed, skipping\n");
    return;
  }
  std::printf("[h3_blocks] lora sweep: matrix cores %s\n",
              mma ? "yes" : "no");

  for (int M : rows) {
    for (const Proj& p : kProjs) {
      // The base projection at this shape, in this process and this power
      // budget -- the denominator for both halves below.
      QW q = make_qw_(mc, p.N, p.K, bits, (std::uint32_t)(p.N * 31 + p.K));
      double base_ms = 0.0;
      if (!q.w.empty()) {
        const Route br = mma ? (genai::mma_use_wide_tile(p.N, p.K) ? Route::Mma256
                                                    : Route::Mma128)
                             : Route::Steel64;
        const double bf = 2.0 * M * p.N * p.K;
        const int bi = std::min(20, std::max(2, (int)(3.0e11 / bf)));
        const double g = lrate_(mc, bf, bi, nullptr, [&](ComputeEncoder& e) {
          encode_(e, kn, br, q, x, y, wdq, M, p.N, p.K, bits);
        });
        base_ms = g > 0.0 ? bf / 1e9 / g * 1e3 : 0.0;
      }

      for (int r : kRanks) {
        // ---- A-GEMM: t = x A^T, N = rank -----------------------------
        const double fa = 2.0 * M * p.K * r;
        const int ia = std::min(400, std::max(2, (int)(3.0e12 / fa)));
        struct Arm { const char* tag; double g, mhz; };
        std::vector<Arm> av{{"steel", 0, 0}};
        if (mma && r <= 128) { av.push_back({"mma64", 0, 0}); }
        if (mma && r >= 128) { av.push_back({"mma128", 0, 0}); }
        for (Arm& a : av) {           // warm every arm before timing any
          const bool w = std::string(a.tag) == "mma128";
          const bool m = a.tag[0] == 'm';
          CommandStream st = mc->make_command_stream();
          { ComputeEncoder e = st.begin_compute();
            lora_a_(e, kn, w, m, x, A, t, M, r, p.K, 1.0f); }
          st.commit().wait();
        }
        for (Arm& a : av) {
          const bool w = std::string(a.tag) == "mma128";
          const bool m = a.tag[0] == 'm';
          a.g = lrate_(mc, fa, ia, &a.mhz, [&](ComputeEncoder& e) {
            lora_a_(e, kn, w, m, x, A, t, M, r, p.K, 1.0f);
          });
        }
        std::printf("[h3_blocks] M=%5d %-3s r=%3d A[K=%5d,N=%3d] |", M,
                    p.tag, r, p.K, r);
        for (const Arm& a : av) {
          std::printf(" %s %.0f (%.3f ms)", a.tag, a.g,
                      a.g > 0.0 ? fa / 1e9 / a.g * 1e3 : 0.0);
        }
        if (base_ms > 0.0 && av[0].g > 0.0) {
          std::printf("  base %.2f ms", base_ms);
        }
        std::printf("  %.0f MHz\n", av[0].mhz);

        // ---- B-GEMM: y += s * t B^T, K = rank ------------------------
        const double fb = 2.0 * M * p.N * r;
        const int ib = std::min(400, std::max(2, (int)(3.0e12 / fb)));
        std::vector<int> barms{0, 1};
        if (mma) { barms.push_back(2); barms.push_back(3); }
        const char* btag[] = {"steel-acc", "steel-ovr", "mma128-acc",
                        "mma256-acc"};
        std::vector<double> bg(barms.size(), 0.0);
        double bmhz = 0.0;
        for (std::size_t i = 0; i < barms.size(); ++i) {
          CommandStream st = mc->make_command_stream();
          { ComputeEncoder e = st.begin_compute();
            lora_b_(e, kn, barms[i], t, B, y, M, p.N, r, kScale); }
          st.commit().wait();
        }
        for (std::size_t i = 0; i < barms.size(); ++i) {
          bg[i] = lrate_(mc, fb, ib, i == 0 ? &bmhz : nullptr,
                         [&](ComputeEncoder& e) {
                           lora_b_(e, kn, barms[i], t, B, y, M, p.N, r,
                                   kScale);
                         });
        }
        std::printf("[h3_blocks] M=%5d %-3s r=%3d B[K=%3d,N=%5d] |", M,
                    p.tag, r, r, p.N);
        for (std::size_t i = 0; i < barms.size(); ++i) {
          std::printf(" %s %.0f (%.3f ms)", btag[barms[i]], bg[i],
                      bg[i] > 0.0 ? fb / 1e9 / bg[i] * 1e3 : 0.0);
        }
        std::printf("  %.0f MHz\n", bmhz);
      }
    }
  }
  EXPECT_TRUE(true);
}

// The matrix-core LoRA tiles must AGREE with the steel pair they replace.
//
// Same doctrine as mma_matches_steel: the sweep above is allowed to report
// because THIS asserts. Three things can go wrong in a way that still
// produces a plausible rate -- the scaled tile dropping its scale (which
// the steel path carries in a different place, so a route swap can lose
// it), the accumulating tile overwriting y instead of folding it in, and
// the 64-wide tile storing past a rank that is not a multiple of its
// tile. The first two are checked by construction below: the reference is
// the steel path at a scale that is neither 0 nor 1, and y is seeded with
// a base GEMM whose contribution must survive.
//
// The bar is tight because both arms compute the SAME product in f32 and
// differ only in which registers hold it -- unlike mma_matches_steel,
// there is no dequantization on one side to widen it.
TEST(minimax_h3_blocks, lora_mma_matches_steel)
{
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.gemm_bm64.valid() || !kn.gemm_bm64_acc.valid()) { return; }
  if (!kn.have_mma || !kn.dense64_sc.valid() || !kn.dense128_sc.valid() ||
      !kn.dense128_acc.valid() || !kn.dense256_acc.valid()) {
    std::printf("[h3_blocks] no matrix cores, lora mma arm skipped\n");
    return;
  }
  // A rank that is NOT a multiple of the 64-wide tile, and rows that are
  // not a multiple of 128: the ragged tails are where a tile addresses
  // past its extent, and rank 48 with 300 rows exercises both at once.
  // `bmma` says whether the SECOND GEMM takes a matrix-core tile too.
  // The model runs mixed at rank 64 -- the first half goes to matmul2d
  // and the second stays on steel, because the rank IS its contraction
  // depth and 64 is too shallow to beat a kernel that is just streaming
  // y. That mixture is where the strength changes hands, so it is the
  // combination most likely to apply it twice or not at all, and it has
  // to be checked as its own case rather than inferred from the two pure
  // ones.
  const struct { int M, N, K, r; bool bmma; } kCases[] = {
      {300, kHidden, kInner, 48, true},       // ragged M and ragged rank
      {256, 3 * kInner, kHidden, 64, true},   // the shipped Turbo rank
      {384, 2 * kFfn, kHidden, 128, true},    // the lightx2v rank
      {256, 3 * kInner, kHidden, 64, false},  // what rank 64 ACTUALLY runs
      {3864, kHidden, kInner, 64, false},     // ...at a production row count
  };
  const float kScale = 0.625f;

  for (const auto& c : kCases) {
    SharedBuffer x = make_act_(mc, (std::size_t)c.M * c.K, 21u);
    SharedBuffer W = make_act_(mc, (std::size_t)c.N * c.K, 23u);
    SharedBuffer A = make_act_(mc, (std::size_t)c.r * c.K, 29u);
    SharedBuffer B = make_act_(mc, (std::size_t)c.N * c.r, 31u);
    SharedBuffer t = mc->make_shared_buffer((std::size_t)c.M * c.r * 2);
    SharedBuffer y[3];       // [2] is base-only: the adapter's own yardstick
    for (int i = 0; i < 3; ++i) {
      y[i] = mc->make_shared_buffer((std::size_t)c.M * c.N * 2);
    }
    if (x.empty() || W.empty() || A.empty() || B.empty() || t.empty() ||
        y[0].empty() || y[1].empty() || y[2].empty()) {
      continue;
    }
    // arm 0 = steel pair, arm 1 = matmul2d pair. The BASE projection is
    // the same steel GEMM either way, so what the comparison isolates is
    // the adapter.
    for (int arm = 0; arm < 3; ++arm) {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        e.set_function(kn.gemm_bm64);          // y = x W^T (the base)
        e.set_buffer(0, x); e.set_buffer(1, W); e.set_buffer(2, W);
        e.set_buffer(3, y[arm]);
        e.set_constant(4, c.K); e.set_constant(5, c.N);
        e.set_constant(6, c.M); e.set_constant(7, 0);
        e.dispatch({(unsigned)(((c.N + 31) / 32) * 32),
                    (unsigned)(((c.M + 63) / 64) * 2), 2}, {32, 2, 2});
        if (arm == 0) {
          // steel: t = x A^T, then y += s * t B^T.
          e.set_function(kn.gemm_bm64);
          e.set_buffer(0, x); e.set_buffer(1, A); e.set_buffer(2, A);
          e.set_buffer(3, t);
          e.set_constant(4, c.K); e.set_constant(5, c.r);
          e.set_constant(6, c.M); e.set_constant(7, 0);
          e.dispatch({(unsigned)(((c.r + 31) / 32) * 32),
                      (unsigned)(((c.M + 63) / 64) * 2), 2}, {32, 2, 2});
          e.set_function(kn.gemm_bm64_acc);
          e.set_buffer(0, t); e.set_buffer(1, B); e.set_buffer(2, B);
          e.set_buffer(3, y[arm]);
          e.set_constant(4, c.r); e.set_constant(5, c.N);
          e.set_constant(6, c.M); e.set_constant(7, 0);
          e.set_constant(8, kScale);
          e.dispatch({(unsigned)(((c.N + 31) / 32) * 32),
                      (unsigned)(((c.M + 63) / 64) * 2), 2}, {32, 2, 2});
        } else if (arm == 1) {
          // matmul2d: the scale rides the FIRST tile here, so a swap that
          // kept the steel placement would silently apply it twice or not
          // at all -- which is what this arm is for.
          const bool wide = c.r > 64;
          lora_a_(e, kn, wide, true, x, A, t, c.M, c.r, c.K, kScale);
          // The strength is already in t, so whichever tile runs the
          // second GEMM must NOT apply it again -- steel's epilogue
          // takes 1.0 and the accumulating tile has no scale at all.
          lora_b_(e, kn, c.bmma ? 2 : 0, t, B, y[arm], c.M, c.N, c.r, 1.0f);
        }
        // arm 2 stops after the base GEMM.
      }
      st.commit().wait();
    }
    const double rl2 = rel_l2_(y[1], y[0], (std::size_t)c.M * c.N);
    const double delta = rel_l2_(y[0], y[2], (std::size_t)c.M * c.N);
    std::printf("[h3_blocks] lora mma vs steel M=%5d N=%5d K=%5d r=%3d "
                "B=%-5s: rel-L2 %.3e (delta is %.1f%% of y, so the routes "
                "differ by %.2f%% of the delta)\n",
                c.M, c.N, c.K, c.r, c.bmma ? "mma" : "steel", rl2,
                100.0 * delta, delta > 0.0 ? 100.0 * rl2 / delta : 0.0);
    EXPECT_TRUE(rl2 < 5e-3);
    // ...and the adapter has to be MOVING the output, or two kernels that
    // both ignore B agree perfectly and the bar above means nothing.
    EXPECT_TRUE(delta > 0.05);
  }
}

// What the fold is worth: the base projection carrying the adapter's
// second GEMM, against the two of them encoded separately.
//
// The A half (t = x A^T) is common to both and excluded, so what is timed
// is exactly the difference. The separate arm is the base tile plus the
// accumulating second GEMM; the fused arm is one tile that does both.
//
// The saving is traffic, and it is easy to predict where it will be big:
// the second GEMM reads and writes the whole [M, N] output for only
// 2*M*r*N flops, so it costs in proportion to N while contributing in
// proportion to r. The wide projections (fc1 at 28672, qkv at 21504) are
// therefore where a separate delta hurts most and the fold helps most.
TEST(minimax_h3_blocks, lora_fused_rate)
{
  if (!bench_on_()) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.have_mma || !kn.dense128_lora.valid() ||
      !kn.dense256_lora.valid() || !kn.gemm_bm64_acc.valid()) {
    std::printf("[h3_blocks] no fused lora tiles, skipping\n");
    return;
  }
  std::vector<int> rows;
  if (bench_full_()) {
    rows.assign(std::begin(kRowsFull), std::end(kRowsFull));
  } else {
    rows.assign(std::begin(kRowsDefault), std::end(kRowsDefault));
  }
  const int R = 64;      // the shipped Turbo rank

  int Mx = 0, Nx = 0, Kx = 0;
  for (int M : rows) { Mx = std::max(Mx, M); }
  for (const Proj& p : kProjs) {
    Nx = std::max(Nx, p.N);
    Kx = std::max(Kx, p.K);
  }
  SharedBuffer x = make_act_(mc, (std::size_t)Mx * Kx, 7u);
  SharedBuffer W = make_act_(mc, (std::size_t)Nx * Kx, 11u);
  SharedBuffer B = make_act_(mc, (std::size_t)Nx * R, 31u);
  SharedBuffer t = mc->make_shared_buffer((std::size_t)Mx * R * 2);
  SharedBuffer y = mc->make_shared_buffer((std::size_t)Mx * Nx * 2);
  if (x.empty() || W.empty() || B.empty() || t.empty() || y.empty()) {
    return;
  }

  for (int M : rows) {
    for (const Proj& p : kProjs) {
      const bool wide = genai::mma_use_wide_tile(p.N, p.K);
      const int BN = wide ? 256 : 128;
      const ComputeFunction& base  = wide ? kn.dense256 : kn.dense128;
      const ComputeFunction& fused = wide ? kn.dense256_lora
                                          : kn.dense128_lora;
      auto enc_base = [&](ComputeEncoder& e) {
        e.set_function(base);
        e.set_buffer(0, x); e.set_buffer(1, W); e.set_buffer(2, W);
        e.set_buffer(3, y);
        e.set_constant(4, p.K); e.set_constant(5, p.N); e.set_constant(6, M);
        e.set_constant(7, 0);
        e.dispatch({(unsigned)(((p.N + BN - 1) / BN) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      };
      auto enc_sep = [&](ComputeEncoder& e) {
        enc_base(e);
        lora_b_(e, kn, 0, t, B, y, M, p.N, R, 1.0f);   // steel accumulate
      };
      auto enc_fused = [&](ComputeEncoder& e) {
        e.set_function(fused);
        e.set_buffer(0, x); e.set_buffer(1, W); e.set_buffer(2, W);
        e.set_buffer(3, y);
        e.set_constant(4, p.K); e.set_constant(5, p.N); e.set_constant(6, M);
        e.set_constant(7, 0);
        e.set_buffer(8, t); e.set_buffer(9, B);
        e.set_constant(10, R);
        e.dispatch({(unsigned)(((p.N + BN - 1) / BN) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      };
      // Sized on the BASE projection's flops, which both arms do, so the
      // two rates are directly comparable and the ratio is the saving.
      const double f1 = 2.0 * M * p.N * p.K;
      const int it = std::min(40, std::max(2, (int)(3.0e12 / f1)));
      double mhz = 0.0;
      // Warm both before timing either.
      for (int i = 0; i < 2; ++i) {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute();
          if (i == 0) { enc_base(e); } else { enc_fused(e); } }
        st.commit().wait();
      }
      const double g_base  = lrate_(mc, f1, it, &mhz, enc_base);
      const double g_sep   = lrate_(mc, f1, it, nullptr, enc_sep);
      const double g_fused = lrate_(mc, f1, it, nullptr, enc_fused);
      const double ms = [&](double g) { return f1 / 1e9 / g * 1e3; }(1.0);
      (void)ms;
      const double ms_base  = f1 / 1e9 / g_base * 1e3;
      const double ms_sep   = f1 / 1e9 / g_sep * 1e3;
      const double ms_fused = f1 / 1e9 / g_fused * 1e3;
      std::printf("[h3_blocks] M=%5d %-3s N=%5d K=%5d r=%d | base %.2f ms | "
                  "separate %.2f (+%.2f) | fused %.2f (+%.2f) | adapter "
                  "%.2fx cheaper  %.0f MHz\n", M, p.tag, p.N, p.K, R,
                  ms_base, ms_sep, ms_sep - ms_base, ms_fused,
                  ms_fused - ms_base,
                  (ms_fused - ms_base) > 1e-6
                      ? (ms_sep - ms_base) / (ms_fused - ms_base) : 0.0,
                  mhz);
    }
  }
  EXPECT_TRUE(true);
}
