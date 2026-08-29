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
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/mma-splitk.h"
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
             const SharedBuffer& wdq, int M, int N, int K, int bits,
             std::size_t x_off = 0, std::size_t y_off = 0)
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
    enc.set_buffer(0, x, x_off * 2);
    enc.set_buffer(1, wdq);
    enc.set_buffer(2, wdq);          // bias slot unused (has_bias = 0)
    enc.set_buffer(3, y, y_off * 2);
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
  enc.set_buffer(3, x, x_off * 2);
  enc.set_buffer(4, y, y_off * 2);
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
// THE GEMM ROUTES, under the same row-invariance rule.
//
// Row r of x @ w^T does not depend on how many rows came with it, on any
// route: steel walks the quantized weight per output tile, matmul2d
// dequantizes it once and tiles, and the int8 mode quantizes the
// ACTIVATION per (row, group) and the weight per (channel, group) -- all
// three per-row or per-weight, none of them per-M. So the same check
// applies, and it reaches the one path a passing bf16 run never took.
//
// The shapes are the ones the forward actually dispatches at ~63k rows.
// qkv and fc1 are banded there (mma_row_band_ caps them at 49920 / 37376),
// so they are probed at their band; `o` and `fc2` are NOT banded -- N=5376
// puts their cap at 199680 -- so they see the full height, which makes
// fc2 the deepest unbanded contraction on the path and the largest i8
// activation scratch the model ever builds (~900 MB at K=14336).
//
// Env: VPIPE_H3_ROW_PROBE=1.
TEST(minimax_h3_blocks, gemm_row_invariance)
{
  if (std::getenv("VPIPE_H3_ROW_PROBE") == nullptr) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.steel[1].valid()) { return; }
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise_bf16");

  constexpr int kRef = 2048;
  // Per shape: the heights to compare, capped at what the forward would
  // dispatch for that shape after banding.
  struct Shape { const char* tag; int N, K, top; };
  const Shape shapes[] = {
      {"o",   kHidden, kInner, 76800},   // unbanded: sees the whole sequence
      {"fc2", kHidden, kFfn,   76800},   // unbanded, deepest K
      {"qkv", 3 * kInner, kHidden, 49920},   // its band
      {"fc1", 2 * kFfn,   kHidden, 37376},   // its band
  };

  std::uint16_t tab[256];
  for (int i = 0; i < 256; ++i) {
    tab[i] = to_bf16_(((float)i / 255.0f) * 0.4f - 0.2f);
  }

  int checked = 0;
  bool clean = true;
  for (const Shape& sh : shapes) {
    const int rows[] = {kRef, 55400 < sh.top ? 55400 : sh.top / 2, sh.top};
    const int M_MAX = sh.top;
    SharedBuffer x =
        mc->make_shared_buffer((std::size_t)M_MAX * sh.K * 2);
    SharedBuffer y =
        mc->make_shared_buffer((std::size_t)M_MAX * sh.N * 2);
    SharedBuffer wdq =
        mc->make_shared_buffer((std::size_t)sh.N * sh.K * 2);
    QW q = make_qw_(mc, sh.N, sh.K, 8, (std::uint32_t)(sh.N + 8));
    if (x.empty() || y.empty() || wdq.empty() || q.w.empty()) {
      std::printf("[h3_blocks] %s: allocation failed, skipped\n", sh.tag);
      continue;
    }
    {
      auto* p = static_cast<std::uint16_t*>(x.contents());
      for (std::size_t r = 0; r < (std::size_t)M_MAX; ++r) {
        std::uint16_t* row = p + r * (std::size_t)sh.K;
        for (int c = 0; c < sh.K; ++c) { row[c] = tab[(r + c) & 255]; }
      }
    }
    // The int8 mode is a MODEL-level switch, so the context is built with
    // want=true here rather than left to the env -- a probe that silently
    // measured the bf16 path would pass while testing nothing.
    vpipe::genai::I8GemmContext i8(mc, /*want=*/true,
                                  /*bf16=*/true);
    // The split-K fold, which the tuner picks for fc2 (K=14336) and which
    // no other arm here reaches: it replaces the single-op tile with S
    // partial f32 planes and a fold. Its own row blocking is chosen from
    // (S, N), so it is the one arm whose DISPATCH SHAPE changes with M.
    vpipe::genai::MmaSplitK splitk;
    splitk.load(mc, kn.lib_dense, lib_elt);

    // mode: 0 = a plain route, 1 = the int8 arm, 2 = the split-K fold.
    struct Arm { const char* tag; Route r; int mode; };
    const Arm arms[] = {
        {"steel-bm32",    Route::Steel32, 0},
        {"mma128",        Route::Mma128,  0},
        {"mma128x256tn2", Route::MmaTn2,  0},
        {"i8",            Route::Mma128,  1},
        {"splitk",        Route::Mma128,  2},
    };
    for (const Arm& a : arms) {
      if (a.mode == 0 && !kn.available(a.r, 8)) { continue; }
      if (a.mode == 1 && !i8.enabled()) {
        std::printf("[h3_blocks] %s i8: not built, skipped\n", sh.tag);
        continue;
      }
      if (a.mode == 2 && !splitk.enabled) { continue; }
      std::vector<std::uint16_t> ref;
      bool ran = true;
      for (int M : rows) {
        if (a.mode == 1 && !i8.accepts(M, sh.N, sh.K)) { ran = false; break; }
        std::memset(y.contents(), 0, (std::size_t)M * sh.N * 2);
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          if (a.mode != 0) {
            // Dequantize into wdq first, exactly as gemm_mma_ does: both
            // of these arms take the DENSE weight the matmul2d tile would
            // have read, so it has to be built the same way.
            e.set_function(kn.dq[1]);
            e.set_buffer(0, q.w); e.set_buffer(1, q.s); e.set_buffer(2, q.b);
            e.set_buffer(3, wdq);
            e.set_constant(4, sh.K);
            e.set_constant(5, sh.N);
            e.dispatch({(unsigned)(sh.K * 8 / 32), (unsigned)sh.N, 1},
                       {64, 1, 1});
            if (a.mode == 1) {
              if (!i8.gemm(e, x, 0, wdq, y, 0, M, sh.N, sh.K)) {
                ran = false;
              }
            } else if (!splitk.encode(mc, e, x, wdq, y, sh.K, sh.N, M)) {
              ran = false;   // declined this shape; reported below
            }
          } else {
            encode_(e, kn, a.r, q, x, y, wdq, M, sh.N, sh.K, 8);
          }
        }
        st.commit().wait();
        if (!ran) { break; }
        const auto* p = static_cast<const std::uint16_t*>(y.contents());
        const std::size_t n = (std::size_t)kRef * sh.N;
        if (M == kRef) { ref.assign(p, p + n); continue; }
        std::size_t bad = 0;
        long long first = -1;
        for (std::size_t i = 0; i < n; ++i) {
          if (p[i] != ref[i]) {
            ++bad;
            if (first < 0) { first = (long long)(i / sh.N); }
          }
        }
        std::printf("[h3_blocks] %-4s %-14s N=%5d K=%5d M=%6d  differing "
                    "%8zu (first row %lld)\n", sh.tag, a.tag, sh.N, sh.K, M,
                    bad, first);
        if (bad != 0) { clean = false; }
        EXPECT_TRUE(bad == 0);
        ++checked;
      }
      if (!ran) {
        std::printf("[h3_blocks] %s %s: shape refused, skipped\n", sh.tag,
                    a.tag);
      }
    }
  }
  std::printf("[h3_blocks] gemm row invariance: %d checks, %s\n", checked,
              clean ? "all routes row-invariant" : "SOME ROUTE IS NOT");
  EXPECT_TRUE(checked > 0);
}

// THE TAIL -- the window gemm_row_invariance above cannot see.
//
// That test compares the FIRST kRef rows across heights. The property is
// right and the window is wrong: an operand span that outgrows a 32-bit
// BYTE offset goes wrong at the END of the buffer and leaves the head
// untouched, so the check passes while the fault sits past where it
// looks.
//
// MEASURED in the model at 1376x768x243 (75136 packed rows): block 6's
// fc2 mints non-finite values in exactly 1032192 elements = 192 rows x
// 5376, i.e. rows [74944, 75136). And 74944 is ceil64(2^31 / 28672) --
// the first 64-row tile at or past the 2 GB line in a 14336-wide f16
// tensor, which is exactly the SwiGLU output fc2 contracts over. Three
// runs named the same row.
//
// So compare the LAST kRef rows, against a reference computed in its own
// SMALL buffers where no offset is large. Same property, other end: row
// r's output cannot depend on how many rows precede it.
TEST(minimax_h3_blocks, tail_rows_match_a_small_reference)
{
  if (std::getenv("VPIPE_H3_ROW_PROBE") == nullptr) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.steel[1].valid()) { return; }
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise_bf16");

  constexpr int kRef = 2048;
  struct Shape { const char* tag; int N, K, top; };
  const Shape shapes[] = {
      {"fc2", kHidden,    kFfn,   76800},   // unbanded, deepest K: 2.2 GB
      {"o",   kHidden,    kInner, 76800},   // unbanded, half the depth
      {"qkv", 3 * kInner, kHidden, 49920},  // its band
      {"fc1", 2 * kFfn,   kHidden, 37376},  // its band
  };

  std::uint16_t tab[256];
  for (int i = 0; i < 256; ++i) {
    tab[i] = to_bf16_(((float)i / 255.0f) * 0.4f - 0.2f);
  }

  int checked = 0;
  bool clean = true;
  std::size_t _n_bad = 0;
  for (const Shape& sh : shapes) {
    const int M = sh.top;
    SharedBuffer x  = mc->make_shared_buffer((std::size_t)M * sh.K * 2);
    SharedBuffer y  = mc->make_shared_buffer((std::size_t)M * sh.N * 2);
    SharedBuffer xs = mc->make_shared_buffer((std::size_t)kRef * sh.K * 2);
    SharedBuffer ys = mc->make_shared_buffer((std::size_t)kRef * sh.N * 2);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)sh.N * sh.K * 2);
    QW q = make_qw_(mc, sh.N, sh.K, 8, (std::uint32_t)(sh.N + 8));
    if (x.empty() || y.empty() || xs.empty() || ys.empty() || wdq.empty() ||
        q.w.empty()) {
      std::printf("[h3_blocks] %s: allocation failed, skipped\n", sh.tag);
      continue;
    }
    // Row r's content depends on r ALONE, so the small buffer can hold a
    // byte-identical copy of the tail rows at their own low offsets.
    {
      auto* p = static_cast<std::uint16_t*>(x.contents());
      for (std::size_t r = 0; r < (std::size_t)M; ++r) {
        std::uint16_t* row = p + r * (std::size_t)sh.K;
        for (int c = 0; c < sh.K; ++c) { row[c] = tab[(r + c) & 255]; }
      }
      auto* ps = static_cast<std::uint16_t*>(xs.contents());
      for (int j = 0; j < kRef; ++j) {
        const std::size_t r = (std::size_t)(M - kRef + j);
        std::uint16_t* row = ps + (std::size_t)j * sh.K;
        for (int c = 0; c < sh.K; ++c) { row[c] = tab[(r + c) & 255]; }
      }
    }
    vpipe::genai::I8GemmContext i8(mc, /*want=*/true, /*bf16=*/true);
    vpipe::genai::MmaSplitK splitk;
    splitk.load(mc, kn.lib_dense, lib_elt);

    struct Arm { const char* tag; Route r; int mode; };
    const Arm arms[] = {
        {"steel-bm32",    Route::Steel32, 0},
        {"mma128",        Route::Mma128,  0},
        {"mma128x256tn2", Route::MmaTn2,  0},
        {"i8",            Route::Mma128,  1},
        {"splitk",        Route::Mma128,  2},
    };
    for (const Arm& a : arms) {
      if (a.mode == 0 && !kn.available(a.r, 8)) { continue; }
      if (a.mode == 1 && !i8.enabled()) { continue; }
      if (a.mode == 2 && !splitk.enabled) { continue; }
      bool ran = true;
      // One encode, parameterised by which (x, y, height) it runs on, so
      // the two arms differ ONLY in the size of the offsets they compute.
      auto run = [&](const SharedBuffer& xb, const SharedBuffer& yb,
                     int rows, int band) {
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          if (a.mode != 0) {
            e.set_function(kn.dq[1]);
            e.set_buffer(0, q.w); e.set_buffer(1, q.s); e.set_buffer(2, q.b);
            e.set_buffer(3, wdq);
            e.set_constant(4, sh.K);
            e.set_constant(5, sh.N);
            e.dispatch({(unsigned)(sh.K * 8 / 32), (unsigned)sh.N, 1},
                       {64, 1, 1});
          }
          // gemm_'s loop, verbatim: each band rebases both operands, so
          // every offset the kernel computes restarts near zero.
          for (int m0 = 0; m0 < rows && ran; m0 += band) {
            const int n = (rows - m0) < band ? (rows - m0) : band;
            const std::size_t xo = (std::size_t)m0 * sh.K;
            const std::size_t yo = (std::size_t)m0 * sh.N;
            if (a.mode == 1) {
              if (!i8.gemm(e, xb, xo, wdq, yb, yo, n, sh.N, sh.K)) {
                ran = false;
              }
            } else if (a.mode == 2) {
              if (!splitk.encode(mc, e, xb, wdq, yb, sh.K, sh.N, n, xo, yo)) {
                ran = false;
              }
            } else {
              encode_(e, kn, a.r, q, xb, yb, wdq, n, sh.N, sh.K, 8, xo, yo);
            }
          }
        }
        // wait_ok, not wait: an out-of-memory command buffer leaves the
        // destination untouched, and read as a result that is a reported
        // fault in a kernel that never ran.
        std::string err;
        if (!st.commit().wait_ok(&err)) {
          std::printf("[h3_blocks] %-4s %-14s DISPATCH FAILED (%s)\n",
                      sh.tag, a.tag, err.empty() ? "GPU error" : err.c_str());
          ran = false;
        }
      };
      if (a.mode == 1 && !i8.accepts(M, sh.N, sh.K)) { continue; }
      // The reference: the same tail rows in their own small buffers,
      // where no offset is large. Always one band -- kRef is far below
      // any line.
      std::memset(ys.contents(), 0, (std::size_t)kRef * sh.N * 2);
      run(xs, ys, kRef, kRef);
      if (!ran) {
        std::printf("[h3_blocks] %s %s: refused, skipped\n", sh.tag, a.tag);
        continue;
      }
      const auto* sml = static_cast<const std::uint16_t*>(ys.contents());
      auto tail_diff = [&](int band) {
        std::memset(y.contents(), 0, (std::size_t)M * sh.N * 2);
        run(x, y, M, band);
        if (!ran) { return (long long)-2; }
        const auto* big = static_cast<const std::uint16_t*>(y.contents()) +
                          (std::size_t)(M - kRef) * sh.N;
        long long first = -1;
        std::size_t bad = 0;
        for (std::size_t i = 0; i < (std::size_t)kRef * sh.N; ++i) {
          if (big[i] != sml[i]) {
            ++bad;
            if (first < 0) { first = (long long)(M - kRef + (int)(i / sh.N)); }
          }
        }
        _n_bad = bad;
        return first;
      };
      const double gb = (double)((std::size_t)M * sh.K * 2) / 1073741824.0;
      // DIAGNOSTIC, never asserted: one pass over the whole height, which
      // is what the model encoded while the band was computed from N
      // alone. Printed so the cliff stays visible, and NOT a failure --
      // a machine where Apple moved or fixed it is not a regression.
      const long long raw = tail_diff(M);
      const std::size_t raw_n = _n_bad;
      // THE PROPERTY THE MODEL RESTS ON: at the shipped band the tail is
      // bit-identical to a reference computed without large offsets.
      const int band = vpipe::genai::MetalMiniMaxH3Transformer::mma_row_band(sh.N, sh.K);
      const long long got = tail_diff(band);
      std::printf("[h3_blocks] %-4s %-14s N=%5d K=%5d M=%6d src %5.2f GB  "
                  "band %6d  unbanded %8zu (row %lld)  banded %8zu "
                  "(row %lld)\n",
                  sh.tag, a.tag, sh.N, sh.K, M, gb, band, raw_n, raw,
                  _n_bad, got);
      if (got > -2) {
        if (_n_bad != 0) { clean = false; }
        EXPECT_TRUE(_n_bad == 0);
        ++checked;
      }
    }
  }
  std::printf("[h3_blocks] tail rows: %d checks, %s\n", checked,
              clean ? "every route matches its small reference"
                    : "SOME ROUTE DOES NOT");
  EXPECT_TRUE(checked > 0);
}

// EVERY KERNEL ON THE BLOCK PATH, AT THE ROW COUNTS THAT BREAK.
//
// A 1376x768x243 MiniMax-H3 generation (~63k packed rows) comes back as
// whole-latent noise in BOTH video and audio, while the same weights at
// ~32k rows are clean. Both modalities means the fault is in the DiT, not
// in either VAE; the size means something on the block path stops being
// correct between those two row counts. An end-to-end bisect costs 44
// minutes a point, so bisect the KERNELS instead.
//
// THE PROPERTY: every kernel here is a PER-ROW operation. Row r's output
// cannot depend on how many rows the dispatch covers. So run each kernel
// at a small reference height and again at the heights in question, and
// compare the first `kRef` rows BIT-EXACT. No reference implementation, no
// tolerance, and no checkpoint -- and a kernel that reads or writes past a
// 32-bit index, wraps a grid dimension, or drops a tile has to change one
// of those rows to do it.
//
// Attention is NOT here and cannot be: every query attends to every key,
// so its output legitimately changes with the row count.
// minimax_h3_blocks.attn_nax_matches_steel covers it by cross-checking two
// independent kernels instead.
//
// Inputs are filled from a small table indexed by (row + col), so a row's
// content depends on the ROW as well as the column: a kernel that mixed
// two rows would land on different values rather than on the same ones.
//
// Env: VPIPE_H3_ROW_PROBE=1 (it allocates several GB).
TEST(minimax_h3_blocks, kernel_row_invariance)
{
  if (std::getenv("VPIPE_H3_ROW_PROBE") == nullptr) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }

  ComputeLibrary lib_elt  = mc->load_library("llm_elementwise_bf16");
  ComputeLibrary lib_rms  = mc->load_library("rms_norm_bf16");
  ComputeLibrary lib_rope = mc->load_library("rope_bf16");
  ComputeFunction fn_rms    = lib_rms.function("rms_norm_fast_f16");
  ComputeFunction fn_heads  = lib_rope.function("rms_norm_heads_strided_f16");
  ComputeFunction fn_trope  =
      lib_rope.function("transpose_rope_half_part_ftab_f16");
  ComputeFunction fn_mod    = lib_elt.function("adaln_modulate_idx_f16");
  ComputeFunction fn_gated  = lib_elt.function("gated_residual_idx_f16");
  ComputeFunction fn_resid  = lib_elt.function("residual_add_f16");
  ComputeFunction fn_swiglu = lib_elt.function("swiglu_split_gate_first_f16");
  ComputeFunction fn_bias   = lib_elt.function("bias_add_rows_f16");

  // The reference height, and the heights under suspicion: the largest
  // clean generation, the qkv band, and the failing one.
  // The REAL packed heights of a 1376x768 clip, from
  // video_latent_num_frames(F,17,5) = (F-5)/17*5 + 2 latent frames at
  // 1032 tokens each: 124 frames -> 38184 rows (clean), 158 -> 48504
  // (clean), 243 -> 74304 (whole-latent noise). 49920 is the qkv band,
  // which is the one thing that turns on between the last clean height
  // and the broken one.
  // The REAL packed heights, which are NOT the video rows alone: the
  // sequence is [text | keyframe condition | audio | video], and at
  // 1376x768 the condition is one latent frame (1032 rows) and the audio
  // is 2 rows per latent at 40 latents/second. So 175 frames is ~55400
  // and 243 is ~76250 -- and 243 is what comes back as noise.
  //
  // 74898 is where THREE 32-bit limits land at once, all of them
  // rows * 57344 bytes or rows * 28672 as a signed int: the ff buffer's
  // size, the ob/proj subview offsets into the attention arena, and any
  // elementwise total over a [rows, 2*ffn] tensor. The clean 175 run is
  // far below it and the broken 243 run is just above, so the heights
  // straddle it deliberately rather than stopping short as an earlier
  // cut of this test did.
  constexpr int kRef = 2048;
  const int kRows[] = {kRef, 55400, 74752, 76800};

  // A 256-entry table of safe bf16 values. Filling multi-gigabyte inputs
  // from it is one lookup per element, and it keeps every value in a
  // range where rms / swiglu cannot produce a NaN -- a NaN would compare
  // equal to itself and quietly weaken every check below.
  std::uint16_t tab[256];
  for (int i = 0; i < 256; ++i) {
    tab[i] = to_bf16_(((float)i / 255.0f) * 1.5f - 0.75f);
  }
  auto fill = [&](const SharedBuffer& b, std::size_t rows, std::size_t cols) {
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t r = 0; r < rows; ++r) {
      std::uint16_t* row = p + r * cols;
      for (std::size_t c = 0; c < cols; ++c) { row[c] = tab[(r + c) & 255]; }
    }
  };
  // Bit-exact over the first kRef rows of a [*, cols] destination.
  auto same_rows = [&](const std::vector<std::uint16_t>& ref,
                       const SharedBuffer& got, std::size_t cols) {
    const auto* p = static_cast<const std::uint16_t*>(got.contents());
    std::size_t bad = 0;
    long long first = -1;
    for (std::size_t i = 0; i < (std::size_t)kRef * cols; ++i) {
      if (p[i] != ref[i]) {
        ++bad;
        if (first < 0) { first = (long long)(i / cols); }
      }
    }
    return std::pair<std::size_t, long long>{bad, first};
  };

  int checked = 0;
  bool clean = true;
  // One kernel: `encode` dispatches it at `M`, `cols` is the destination
  // row width, and the first kRef rows are compared against the kRef run.
  auto probe = [&](const char* tag, std::size_t cols,
                   const SharedBuffer& dst,
                   const std::function<void(ComputeEncoder&, int)>& encode) {
    std::vector<std::uint16_t> ref;
    for (int M : kRows) {
      std::memset(dst.contents(), 0, (std::size_t)M * cols * 2);
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        encode(e, M);
      }
      st.commit().wait();
      if (M == kRef) {
        const auto* p = static_cast<const std::uint16_t*>(dst.contents());
        ref.assign(p, p + (std::size_t)kRef * cols);
        continue;
      }
      const auto [bad, first] = same_rows(ref, dst, cols);
      std::printf("[h3_blocks] %-22s M=%6d  differing %8zu (first row %lld)\n",
                  tag, M, bad, first);
      if (bad != 0) { clean = false; }
      EXPECT_TRUE(bad == 0);
      ++checked;
    }
  };
  const int M_MAX = kRows[std::size(kRows) - 1];

  // ---- rms + the three per-row elementwise ops, all [rows, hidden] -----
  if (fn_rms.valid() && fn_mod.valid() && fn_gated.valid() &&
      fn_resid.valid()) {
    const std::size_t H = kHidden;
    SharedBuffer x = mc->make_shared_buffer((std::size_t)M_MAX * H * 2);
    SharedBuffer y = mc->make_shared_buffer((std::size_t)M_MAX * H * 2);
    SharedBuffer g = mc->make_shared_buffer(H * 2);
    // Four distinct timesteps, addressed per row -- the shape the real
    // AdaLN takes, so an index that wrapped would show here.
    const int n_t = 4;
    SharedBuffer mod = mc->make_shared_buffer((std::size_t)n_t * 6 * H * 2);
    SharedBuffer idx = mc->make_shared_buffer((std::size_t)M_MAX * sizeof(int));
    if (!x.empty() && !y.empty() && !g.empty() && !mod.empty() &&
        !idx.empty()) {
      fill(x, M_MAX, H);
      fill(g, 1, H);
      fill(mod, n_t, 6 * H);
      auto* ip = static_cast<int*>(idx.contents());
      for (int r = 0; r < M_MAX; ++r) { ip[r] = r % n_t; }

      probe("rms", H, y, [&](ComputeEncoder& e, int M) {
        e.set_function(fn_rms);
        e.set_buffer(0, x); e.set_buffer(1, g); e.set_buffer(2, y);
        e.set_constant(3, (int)H);
        e.set_constant(4, 1e-5f);
        e.dispatch({256, (unsigned)M, 1}, {256, 1, 1});
      });
      probe("adaln_modulate", H, y, [&](ComputeEncoder& e, int M) {
        e.set_function(fn_mod);
        e.set_buffer(0, x); e.set_buffer(1, mod); e.set_buffer(2, idx);
        e.set_buffer(3, y);
        e.set_constant(4, (int)H);
        e.set_constant(5, (int)(6 * H));
        e.set_constant(6, 0);
        e.set_constant(7, (int)H);
        e.set_constant(8, (int)(M * H));
        e.dispatch({(unsigned)(M * H), 1, 1}, {256, 1, 1});
      });
      probe("gated_residual", H, y, [&](ComputeEncoder& e, int M) {
        // Writes THROUGH buffer 0, so the destination is seeded per run.
        std::memcpy(y.contents(), x.contents(), (std::size_t)M * H * 2);
        e.set_function(fn_gated);
        e.set_buffer(0, y); e.set_buffer(1, mod); e.set_buffer(2, idx);
        e.set_buffer(3, x);
        e.set_constant(4, (int)H);
        e.set_constant(5, (int)(6 * H));
        e.set_constant(6, (int)(2 * H));
        e.set_constant(7, (int)(M * H));
        e.dispatch({(unsigned)(M * H), 1, 1}, {256, 1, 1});
      });
      probe("residual_add", H, y, [&](ComputeEncoder& e, int M) {
        e.set_function(fn_resid);
        e.set_buffer(0, x); e.set_buffer(1, x); e.set_buffer(2, y);
        e.set_constant(3, (int)(M * H));
        e.dispatch({(unsigned)(M * H), 1, 1}, {256, 1, 1});
      });
    }
  }

  // ---- qk_norm + rope, both STRIDED over the [rows, 3*inner] qkv ------
  // 2.7 GB at the failing height, and the widest row stride on the path.
  if (fn_heads.valid() && fn_trope.valid()) {
    const std::size_t W = 3 * (std::size_t)kInner;      // 21504
    SharedBuffer qkv = mc->make_shared_buffer((std::size_t)M_MAX * W * 2);
    SharedBuffer gam = mc->make_shared_buffer((std::size_t)kHeadDim * 2);
    SharedBuffer out = mc->make_shared_buffer((std::size_t)M_MAX * kInner * 2);
    const int rot = 96, half = rot / 2;
    SharedBuffer rc =
        mc->make_shared_buffer((std::size_t)M_MAX * half * sizeof(float));
    SharedBuffer rs =
        mc->make_shared_buffer((std::size_t)M_MAX * half * sizeof(float));
    if (!qkv.empty() && !gam.empty() && !out.empty() && !rc.empty() &&
        !rs.empty()) {
      fill(gam, 1, kHeadDim);
      auto* cp = static_cast<float*>(rc.contents());
      auto* sp = static_cast<float*>(rs.contents());
      for (std::size_t i = 0; i < (std::size_t)M_MAX * half; ++i) {
        cp[i] = std::cos((float)(i % 617) * 0.01f);
        sp[i] = std::sin((float)(i % 617) * 0.01f);
      }
      // qk_norm is IN PLACE, so the source is re-filled per run rather
      // than once -- otherwise every height past the first would norm an
      // already-normed buffer and agree for the wrong reason.
      probe("rms_norm_heads(qkv)", W, qkv, [&](ComputeEncoder& e, int M) {
        fill(qkv, (std::size_t)M, W);
        e.set_function(fn_heads);
        e.set_buffer(0, qkv); e.set_buffer(1, gam);
        e.set_constant(2, M);
        e.set_constant(3, kHeads);
        e.set_constant(4, kHeadDim);
        e.set_constant(5, (int)W);
        e.set_constant(6, 0);                      // Q at offset 0
        e.set_constant(7, 1e-5f);
        e.set_constant(8, kHeadDim);               // flat grouping
        e.dispatch({32, (unsigned)(M * kHeads), 1}, {32, 1, 1});
      });
      fill(qkv, M_MAX, W);
      // trope's DESTINATION is [heads, rows, head_dim], so row r sits at a
      // different offset for every M. Compared head by head rather than
      // through the flat helper.
      std::vector<std::uint16_t> ref;
      for (int M : kRows) {
        std::memset(out.contents(), 0, (std::size_t)M * kInner * 2);
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          e.set_function(fn_trope);
          e.set_buffer(0, qkv); e.set_buffer(1, out);
          e.set_buffer(2, rc); e.set_buffer(3, rs);
          e.set_constant(4, kHeads);
          e.set_constant(5, M);
          e.set_constant(6, kHeadDim);
          e.set_constant(7, rot);
          e.set_constant(8, (int)W);
          e.set_constant(9, 0);
          e.set_constant(10, kHeadDim);
          e.dispatch({(unsigned)kHeadDim, (unsigned)M, (unsigned)kHeads},
                     {(unsigned)kHeadDim, 1, 1});
        }
        st.commit().wait();
        const auto* p = static_cast<const std::uint16_t*>(out.contents());
        const std::size_t n = (std::size_t)kHeads * kRef * kHeadDim;
        if (M == kRef) { ref.assign(p, p + n); continue; }
        std::size_t bad = 0;
        long long first = -1;
        for (int h = 0; h < kHeads; ++h) {
          for (int t = 0; t < kRef; ++t) {
            const std::size_t src = ((std::size_t)h * M + t) * kHeadDim;
            const std::size_t dst = ((std::size_t)h * kRef + t) * kHeadDim;
            for (int d = 0; d < kHeadDim; ++d) {
              if (p[src + d] != ref[dst + d]) {
                ++bad;
                if (first < 0) { first = t; }
              }
            }
          }
        }
        std::printf("[h3_blocks] %-22s M=%6d  differing %8zu "
                    "(first row %lld)\n", "transpose_rope", M, bad, first);
        if (bad != 0) { clean = false; }
        EXPECT_TRUE(bad == 0);
        ++checked;
      }
    }
  }

  // ---- swiglu + bias_add, the two widest 1-D grids on the path --------
  // rows * 2*ffn is 1.8e9 threads in ONE grid dimension at the failing
  // height, which is the largest single dispatch this model makes.
  if (fn_swiglu.valid() && fn_bias.valid()) {
    const std::size_t FF = 2 * (std::size_t)kFfn;      // 28672
    SharedBuffer ff = mc->make_shared_buffer((std::size_t)M_MAX * FF * 2);
    SharedBuffer y  = mc->make_shared_buffer((std::size_t)M_MAX * kFfn * 2);
    SharedBuffer bias = mc->make_shared_buffer(FF * 2);
    if (!ff.empty() && !y.empty() && !bias.empty()) {
      fill(ff, M_MAX, FF);
      fill(bias, 1, FF);
      probe("swiglu", kFfn, y, [&](ComputeEncoder& e, int M) {
        e.set_function(fn_swiglu);
        e.set_buffer(0, ff); e.set_buffer(1, y);
        e.set_constant(2, M);
        e.set_constant(3, kFfn);
        e.dispatch({(unsigned)(M * kFfn), 1, 1}, {256, 1, 1});
      });
      // bias_add is IN PLACE over the widest destination there is.
      probe("bias_add_rows", FF, ff, [&](ComputeEncoder& e, int M) {
        fill(ff, (std::size_t)M, FF);
        e.set_function(fn_bias);
        e.set_buffer(0, ff); e.set_buffer(1, bias);
        e.set_constant(2, (int)FF);
        e.set_constant(3, (int)(M * FF));
        e.dispatch({(unsigned)(M * FF), 1, 1}, {256, 1, 1});
      });
    }
  }

  std::printf("[h3_blocks] row invariance: %d checks, %s\n", checked,
              clean ? "all kernels row-invariant" : "SOME KERNEL IS NOT");
  EXPECT_TRUE(checked > 0);
}

// DOES matmul2d ACTUALLY STORE A DESTINATION PAST 2 GB?
//
// mma_row_band_ splits a projection's rows so that no single dispatch
// writes more than 2^31 BYTES, because matmul2d was found to stop storing
// past that. The band is derived from that assumed limit and never
// measured, and the margin it leaves is not uniform: at H3's fc1
// (N = 28672) the banded span lands 4 MB under 2^31, but at qkv
// (N = 21504) it lands 512 KB under -- eight times tighter. If the real
// cliff is a hair below 2^31 (a padded size, a signed compare on
// something slightly larger), qkv is the shape that falls off first and
// fc1 does not, which is a symptom nobody would read as an off-by-a-hair
// in a band.
//
// So: measure it. Sweep M across the boundary at H3's qkv N, prefill the
// destination with a value the GEMM CANNOT produce, and see which rows
// come back still holding it. An unwritten row is the exact reported
// failure mode -- silently unwritten output, not a wrong answer -- and a
// sentinel finds it without needing a reference at all.
//
// K IS DELIBERATELY TINY. The cliff under test is a property of the
// destination, not of the contraction, so a 128-deep K keeps this at
// ~0.3 TFLOP instead of 11 while the destination stays the full 2.4 GB.
//
// The arithmetic is EXACT by construction: both operands are in
// {-1, 0, 1} and K is 128, so every output is an integer in [-128, 128],
// which bf16 represents exactly. A value check here needs no tolerance,
// and a wrong answer cannot hide inside a rounding budget.
//
// Env: VPIPE_H3_MMA_DEST_PROBE=1 (it allocates ~2.5 GB, which is not
// something the default suite should do on a 16 GB box).
TEST(minimax_h3_blocks, mma_dest_limit_probe)
{
  if (std::getenv("VPIPE_H3_MMA_DEST_PROBE") == nullptr) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.have_mma || !kn.dense128.valid()) {
    std::printf("[h3_blocks] no matrix cores; dest-limit probe skipped\n");
    return;
  }

  // BOTH shapes the model actually bands. qkv has the tighter margin
  // (512 KB under 2^31 against fc1's 4 MB), fc1 the bigger destination.
  const int kNs[] = {3 * kInner, 2 * kFfn};   // 21504 (qkv), 28672 (fc1)
  const int K = 128;
  int n_max = 0;
  for (int n : kNs) { n_max = std::max(n_max, n); }
  bool any_bad = false;
for (const int N : kNs) {
  // The row at which M * N * 2 reaches 2^31, and the band the model ships.
  const long long lim = ((1LL << 31) - 1) / ((long long)N * 2);
  const int band = (int)((lim / 128) * 128);
  const int rows[] = {band, (int)lim, (int)lim + 1, 51200, 55000};
  int m_max = 0;
  for (int m : rows) { m_max = std::max(m_max, m); }
  std::printf("[h3_blocks] N=%d K=%d: 2^31 bytes at M=%lld, shipped band "
              "%d\n", N, K, lim, band);

  // {-1, 0, 1} operands: exact products, and non-constant in BOTH indices
  // so a row or column mix-up cannot produce the right answer.
  SharedBuffer x = mc->make_shared_buffer((std::size_t)m_max * K * 2);
  SharedBuffer w = mc->make_shared_buffer((std::size_t)n_max * K * 2);
  SharedBuffer y =
      mc->make_shared_buffer((std::size_t)m_max * (std::size_t)n_max * 2);
  if (x.empty() || w.empty() || y.empty()) {
    std::printf("[h3_blocks] dest-limit probe needs ~%.1f GB; allocation "
                "failed\n",
                (double)((std::size_t)m_max * N * 2) / 1073741824.0);
    return;
  }
  auto xv = [&](int m, int k) { return (float)(((m + k) % 3) - 1); };
  auto wv = [&](int n, int k) { return (float)(((n * 7 + k) % 3) - 1); };
  {
    auto* xp = static_cast<std::uint16_t*>(x.contents());
    for (int m = 0; m < m_max; ++m) {
      for (int k = 0; k < K; ++k) {
        xp[(std::size_t)m * K + k] = to_bf16_(xv(m, k));
      }
    }
    auto* wp = static_cast<std::uint16_t*>(w.contents());
    for (int n = 0; n < n_max; ++n) {
      for (int k = 0; k < K; ++k) {
        wp[(std::size_t)n * K + k] = to_bf16_(wv(n, k));
      }
    }
  }
  // bf16 NaN. Every output is an integer in [-128, 128], so this is a
  // value the kernel cannot write and a survivor is provably unwritten.
  constexpr std::uint16_t kSentinel = 0x7FC0;
  auto* yp = static_cast<std::uint16_t*>(y.contents());

  auto ref = [&](int m, int n) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) { acc += xv(m, k) * wv(n, k); }
    return acc;
  };

  // ALL THREE mma tiles, because the band does not depend on which one
  // runs and the autotune does. On this M5 the tuner picks the tn2 tile
  // for qkv and fc1 -- the 512-wide one, the only tile that can address a
  // sub-tile past N -- so probing the 128-wide entry alone would measure
  // a kernel the model is not dispatching.
  struct Tile { const char* tag; const ComputeFunction* fn; int rn; };
  const Tile tiles[] = {
      {"mma128",        &kn.dense128,  128},
      {"mma128x256",    &kn.dense256,  256},
      {"mma128x256tn2", &kn.dense_tn2, 512},
  };
  auto run_at = [&](const Tile& t, int M) -> bool {
    const std::size_t on = (std::size_t)M * (std::size_t)N;
    for (std::size_t i = 0; i < on; ++i) { yp[i] = kSentinel; }
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      e.set_function(*t.fn);
      e.set_buffer(0, x);
      e.set_buffer(1, w);
      e.set_buffer(2, w);            // bias slot, unread (has_bias = 0)
      e.set_buffer(3, y);
      e.set_constant(4, K);
      e.set_constant(5, N);
      e.set_constant(6, M);
      e.set_constant(7, 0);
      e.dispatch({(unsigned)(((N + t.rn - 1) / t.rn) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    }
    // wait_ok, NOT wait. This probe reports unwritten elements, and a
    // command buffer that FAILED leaves every element unwritten -- so
    // with the status discarded it cannot tell "the kernel did not
    // store" from "the kernel never ran", which is the one distinction
    // it exists to make. MEASURED: run alongside the other probes on a
    // 24 GB box, this dispatch failed out of memory and the test read it
    // as matmul2d dropping a billion elements at the shipped band. Run
    // alone it passes. An unchecked commit produced a false FINDING,
    // which is the same defect this file is chasing in the model.
    std::string err;
    if (!st.commit().wait_ok(&err)) {
      std::printf("[h3_blocks]   %-14s M=%6d: DISPATCH FAILED (%s) -- not "
                  "a storage limit\n", t.tag, M,
                  err.empty() ? "GPU error" : err.c_str());
      return false;
    }
    return true;
  };

  for (const Tile& t : tiles) {
    if (t.fn == nullptr || !t.fn->valid()) {
      std::printf("[h3_blocks]   %-14s not built\n", t.tag);
      continue;
    }
  for (int M : rows) {
    const std::size_t on = (std::size_t)M * (std::size_t)N;
    if (!run_at(t, M)) { continue; }   // reported above; not a result
    // First row holding a survivor, and the total count.
    long long first_unwritten = -1, n_unwritten = 0;
    for (std::size_t i = 0; i < on; ++i) {
      if (yp[i] == kSentinel) {
        ++n_unwritten;
        if (first_unwritten < 0) { first_unwritten = (long long)(i / N); }
      }
    }
    // Value check on rows that WERE written, at both ends and the middle.
    int wrong = 0;
    for (int m : {0, M / 2, M - 1}) {
      for (int n : {0, N / 2, N - 1}) {
        const std::uint16_t got = yp[(std::size_t)m * N + n];
        if (got == kSentinel) { continue; }   // counted above, not here
        if (from_bf16_(got) != ref(m, n)) { ++wrong; }
      }
    }
    const double gb = (double)(on * 2) / 1073741824.0;
    std::printf("[h3_blocks]   %-14s M=%6d  dest %6.3f GB  unwritten %lld"
                " (first row %lld)  wrong %d\n",
                t.tag, M, gb, n_unwritten, first_unwritten, wrong);
    if (n_unwritten != 0 || wrong != 0) { any_bad = true; }
    // THE PROPERTY THE MODEL RESTS ON, stated directly: wherever the
    // cliff is, it must not be below the band mma_row_band_ hands out.
    // Asserting the cliff's exact LOCATION would fail on a machine where
    // Apple moved or fixed it, which is not a regression -- this is.
    EXPECT_TRUE(first_unwritten < 0 || first_unwritten >= band);
  }
  }
  // The claim the band rests on: at and below the shipped band, matmul2d
  // writes every element and writes it correctly. Above it is ALLOWED to
  // fail -- that is what the band exists for -- so this asserts only the
  // rows the model would actually dispatch.
  // EVERY tile, at the band the model would actually dispatch. Above the
  // band a tile is ALLOWED to fail -- that is what banding is for -- so
  // this asserts only what the forward can reach.
  for (const Tile& t : tiles) {
    if (t.fn == nullptr || !t.fn->valid()) { continue; }
    const std::size_t on = (std::size_t)band * (std::size_t)N;
    if (!run_at(t, band)) { continue; }
    long long bad = 0;
    for (std::size_t i = 0; i < on; ++i) { if (yp[i] == kSentinel) { ++bad; } }
    std::printf("[h3_blocks] %-14s N=%5d at the shipped band M=%d: %lld "
                "unwritten\n", t.tag, N, band, bad);
    EXPECT_TRUE(bad == 0);
  }
}   // for N
  if (!any_bad) {
    std::printf("[h3_blocks] no cliff found at either shape: the 2^31 "
                "assumption is right or conservative here\n");
  }
}

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
// DOES THE FIRE-AND-FORGET WATCH ACTUALLY RUN?
//
// ComputeEncoder splits its stream every 50 dispatches and commits those
// buffers without waiting. Nothing else can see one fail, so a completion
// handler latches the failure and Fence::wait_ok reports it. That was
// added, and then four corrupt runs reported no failure -- which was read
// as "no command buffer failed" when it equally supports "the handler
// never ran". Those are opposite conclusions from the same silence, and
// nothing distinguished them.
//
// This does. Encode past the split threshold, commit, and check the
// COMPLETED counter moved. A zero there means the watch is dead and every
// silent run has to be re-read; a nonzero means silence is evidence.
//
// It does not try to force a FAILURE -- there is no reliable way to make
// a command buffer fail on demand, and an unreliable one would make this
// test flaky rather than informative. Proving the handler runs is the
// half that was actually missing.
TEST(minimax_h3_blocks, fire_and_forget_buffers_are_watched)
{
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("llm_elementwise_bf16");
  ComputeFunction fn = lib.function("residual_add_f16");
  if (!fn.valid()) { return; }

  const int n = 4096;
  SharedBuffer a = make_act_(mc, (std::size_t)n, 5u);
  SharedBuffer y = mc->make_shared_buffer((std::size_t)n * 2);
  if (a.empty() || y.empty()) { return; }

  unsigned long long done0 = 0, bad0 = 0;
  CommandStream::fire_and_forget_stats(&done0, &bad0);

  // Well past the 50-dispatch split default, so several buffers are
  // committed and released before the one the Fence holds.
  {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      for (int i = 0; i < 160; ++i) {
        e.set_function(fn);
        e.set_buffer(0, a); e.set_buffer(1, a); e.set_buffer(2, y);
        e.set_constant(3, n);
        e.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
      }
    }
    std::string err;
    const bool ok = st.commit().wait_ok(&err);
    EXPECT_TRUE(ok);
    if (!ok) { std::printf("[h3_blocks] stream failed: %s\n", err.c_str()); }
  }

  unsigned long long done1 = 0, bad1 = 0;
  CommandStream::fire_and_forget_stats(&done1, &bad1);
  std::printf("[h3_blocks] fire-and-forget: %llu completed (+%llu), %llu in "
              "error (+%llu)\n", done1, done1 - done0, bad1, bad1 - bad0);
  // 160 dispatches at a 50-dispatch split is at least two intermediate
  // buffers; the last one goes to the Fence and is not counted here.
  EXPECT_TRUE(done1 > done0);
  EXPECT_TRUE(bad1 == bad0);
}

// CAN THE INT8 ARM TURN A FINITE ACTIVATION INTO NaN?
//
// A 1376x768x243 generation returns static, and the block-0 dissection
// caught the moment it happens: fc2 -- the down projection -- takes a
// healthy swiglu output (video rms 0.29) and emits NaN on the video rows
// while text and audio survive the same op. Disabling the int8 arm
// (VPIPE_I8_GEMM=0) makes the whole run clean, output quality confirmed
// by the encoder's average quantizer dropping 44825 -> 658.
//
// So the arm is the source. This asks WHY, and it does it locally in
// seconds instead of thirty minutes.
//
// THE SUSPECT IS THE GROUP SCALE. The quantizer takes a per-(row, 512)
// group maximum and forms `inv = am > 0 ? 127/am : 0`. The guard catches
// a group that is entirely zero -- but not one whose maximum is merely
// TINY. At am ~ 1e-40, 127/am overflows f32 to inf, and every zero in
// that group becomes 0 * inf = NaN; rint() of that, converted to int, is
// undefined in MSL.
//
// fc2 is where it would bite first: its K is 14336 = 28*512 exactly, so
// it takes the unpadded quantizer, and its input is the SwiGLU output,
// which is the smallest-magnitude activation on the block path (audio
// rms 0.018 in the failing run). More rows means more groups means more
// chances, which is the row-count sensitivity and the intermittency.
//
// The sweep walks a group maximum from ordinary down to denormal with
// the rest of the group ZERO -- the arrangement that makes 0 * inf --
// and asks only whether the result is finite. A finite wrong answer is
// not what is being hunted here; a NaN is.
TEST(minimax_h3_blocks, i8_group_scale_survives_tiny_maxima)
{
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  vpipe::genai::I8GemmContext i8(mc, /*want=*/true, /*bf16=*/true);
  if (!i8.enabled()) {
    std::printf("[h3_blocks] int8 arm not built; scale probe skipped\n");
    return;
  }
  // fc2's shape at the REAL row count. 1024 -- the smallest the arm
  // accepts -- was the first cut of this test, and it removed the one
  // variable the failure is known to depend on: the run breaks at 75136
  // packed rows and not at 70945, so a probe that cannot reach either
  // number cannot see whatever separates them. VPIPE_H3_I8_PROBE_M
  // overrides for a box with less memory (~4.2 GB at the default).
  const int N = kHidden, K = kFfn;               // 5376 x 14336
  int M = 75136;
  if (const char* e = std::getenv("VPIPE_H3_I8_PROBE_M")) {
    if (*e != '\0') { M = std::atoi(e); }
  }
  if (!i8.accepts(M, N, K)) {
    std::printf("[h3_blocks] int8 arm declines %dx%dx%d; skipped\n", M, N, K);
    return;
  }
  std::printf("[h3_blocks] int8 group-scale probe at M=%d (fc2 shape)\n", M);
  SharedBuffer x = mc->make_shared_buffer((std::size_t)M * K * 2);
  SharedBuffer w = make_act_(mc, (std::size_t)N * K, 77u);
  SharedBuffer y = mc->make_shared_buffer((std::size_t)M * N * 2);
  if (x.empty() || w.empty() || y.empty()) {
    std::printf("[h3_blocks] scale probe allocation failed; skipped\n");
    return;
  }
  auto* xp = static_cast<std::uint16_t*>(x.contents());
  auto* yp = static_cast<std::uint16_t*>(y.contents());

  // Ordinary, then progressively smaller, then flush-to-zero. 1.18e-38 is
  // the smallest NORMAL float; below it the reciprocal starts to leave
  // f32's range, and 127/1e-40 does not fit at all.
  const float mags[] = {1.0f,    1e-3f,  1e-10f, 1e-20f,
                        1e-30f,  1e-35f, 1e-38f, 1e-40f, 1e-44f};
  int bad_at = 0;
  for (float mag : mags) {
    // A group whose MAXIMUM is `mag`, with the rest of it spread over
    // many decades below and a share of exact zeros.
    //
    // The first cut put one element at `mag` and left the other 511 at
    // zero, which is a shape no activation has: a real post-SwiGLU group
    // has a wide dynamic range WITHIN it, and the interesting arithmetic
    // is between the largest and smallest members, not between the
    // largest and nothing. A test whose input cannot occur can only
    // exonerate the code against inputs that cannot occur.
    const int G = K / 512;
    std::uint32_t rs = 12345u;
    for (int r = 0; r < M; ++r) {
      for (int g = 0; g < G; ++g) {
        const std::size_t b0 = (std::size_t)r * K + (std::size_t)g * 512;
        for (int i = 0; i < 512; ++i) {
          rs = rs * 1664525u + 1013904223u;
          const int bucket = (int)((rs >> 8) % 24u);
          float v;
          if (bucket < 4) { v = 0.0f; }                 // exact zeros
          else {
            // Log-spread from the group max down ~18 decades.
            const float dec = (float)(bucket - 4) * 0.9f;
            v = mag * std::pow(10.0f, -dec);
            if ((rs >> 31) != 0) { v = -v; }
          }
          xp[b0 + (std::size_t)i] = to_bf16_(v);
        }
        // Pin the maximum to exactly `mag` so the sweep means what it says.
        xp[b0] = to_bf16_(mag);
      }
    }
    std::memset(yp, 0, (std::size_t)M * N * 2);
    {
      CommandStream st = mc->make_command_stream();
      bool encoded = false;
      {
        ComputeEncoder e = st.begin_compute();
        encoded = i8.gemm(e, x, 0, w, y, 0, M, N, K);
      }
      std::string err;
      const bool ok = st.commit().wait_ok(&err);
      if (!encoded) {
        std::printf("[h3_blocks] mag %.0e: arm declined the shape\n", mag);
        continue;
      }
      if (!ok) {
        std::printf("[h3_blocks] mag %.0e: DISPATCH FAILED (%s)\n", mag,
                    err.empty() ? "GPU error" : err.c_str());
        continue;
      }
    }
    std::size_t nan_n = 0, inf_n = 0;
    for (std::size_t i = 0; i < (std::size_t)M * N; ++i) {
      const float v = from_bf16_(yp[i]);
      if (std::isnan(v)) { ++nan_n; }
      else if (std::isinf(v)) { ++inf_n; }
    }
    // The bf16 value the group max actually round-trips to, since a
    // denormal may flush and then the case under test never happened.
    const float got = from_bf16_(to_bf16_(mag));
    std::printf("[h3_blocks] group max %.0e (stored %.3e): nan %zu inf %zu\n",
                mag, (double)got, nan_n, inf_n);
    if (nan_n != 0 || inf_n != 0) { ++bad_at; }
  }
  // A finite activation must not produce a non-finite product, at any
  // scale the input can legally hold. This is the whole claim.
  if (bad_at != 0) {
    std::printf("[h3_blocks] the int8 arm produced non-finite output at %d "
                "of %zu group magnitudes\n", bad_at, std::size(mags));
  }
  EXPECT_TRUE(bad_at == 0);
}

// THE REAL ARENA AT THE FIRST FAILING GEOMETRY.
//
// 226 frames is clean and 243 is not, at four denoise steps either way,
// so the step count is out and the boundary lies between ~70945 and
// ~76162 packed rows. 74898 is inside that window, and it is where
// rows * 57344 crosses 2^32 -- which is BOTH `ff`'s byte SIZE and the
// byte OFFSET of `ob` and `proj` into the attention arena.
//
// The offset alone was already checked. What was not is the whole
// arena at the real size with all seven windows laid out together:
// 5.47 GB in one allocation, `ff` a 4.07 GiB-long subview from zero,
// and `ob` beginning at exactly the byte where `ff` ends. Every earlier
// probe allocated a standalone buffer per tensor, so the adjacency that
// the model actually depends on -- ff's tail butting against ob's head
// across the 4 GiB line -- had never been exercised.
//
// Three claims, and the third is the one the layout rests on:
//   1. each window addresses its own slice, at its head AND its tail
//   2. writing one window leaves the others alone
//   3. ff's LAST bytes and ob's FIRST bytes are different memory, even
//      though the boundary between them sits past 2^32
//
// Slabs at both ends of every window rather than whole windows: the
// interesting addresses are the boundaries, and a truncated offset
// misses by gigabytes, not by bytes.
//
// Env: VPIPE_H3_ROW_PROBE=1 (~5.5 GB).
TEST(minimax_h3_blocks, arena_windows_at_the_failing_geometry)
{
  if (std::getenv("VPIPE_H3_ROW_PROBE") == nullptr) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise_bf16");
  ComputeFunction fn = lib_elt.function("bias_add_rows_f16");
  if (!fn.valid()) { return; }

  // ensure_scratch_'s arithmetic, at a 243-frame clip rounded to a tile.
  const std::size_t S = 76288;
  const std::size_t I = kInner, H = kHidden, FF = 2 * (std::size_t)kFfn;
  const std::size_t arena_el = std::max<std::size_t>(5 * I, FF + H);
  const std::size_t win = S * I * 2;              // one window, bytes
  const std::size_t ff_bytes = S * FF * 2;        // ff, bytes
  const std::size_t proj_bytes = S * H * 2;
  std::printf("[h3_blocks] arena %.2f GB; win %.3f GB; ff %.3f GB; "
              "ob/proj at %.3f GB (2^32 = 4.000 GB)\n",
              (double)(S * arena_el * 2) / 1073741824.0,
              (double)win / 1073741824.0,
              (double)ff_bytes / 1073741824.0,
              (double)(4 * win) / 1073741824.0);
  // The layout only makes sense if ff ends exactly where ob begins --
  // that is what lets proj be parked past ff without colliding.
  ASSERT_TRUE(4 * win == ff_bytes);
  ASSERT_TRUE(4 * win + proj_bytes <= S * arena_el * 2);

  SharedBuffer arena = mc->make_shared_buffer(S * arena_el * 2);
  if (arena.empty()) {
    std::printf("[h3_blocks] arena probe needs %.2f GB; skipped\n",
                (double)(S * arena_el * 2) / 1073741824.0);
    return;
  }
  auto* ap = static_cast<std::uint16_t*>(arena.contents());
  const int N = 4096;
  const std::size_t slab = (std::size_t)N * 64;    // elements per probe slab
  SharedBuffer bias = mc->make_shared_buffer((std::size_t)N * 2);
  auto* bp = static_cast<std::uint16_t*>(bias.contents());

  struct Win { const char* tag; std::size_t off, bytes; };
  const Win wins[] = {
      {"qh",   0 * win, win}, {"kh", 1 * win, win}, {"vh", 2 * win, win},
      {"oh",   3 * win, win}, {"ob", 4 * win, win},
      {"ff",   0,        ff_bytes},
      {"proj", ff_bytes, proj_bytes},
  };
  // Every slab this test inspects, so they can be zeroed without
  // touching 5.5 GB and so the "left alone" check knows where to look.
  std::vector<std::size_t> spots;
  for (const Win& w : wins) {
    spots.push_back(w.off / 2);
    spots.push_back(w.off / 2 + w.bytes / 2 - slab);
  }
  auto zero_spots = [&]() {
    for (std::size_t e : spots) { std::memset(ap + e, 0, slab * 2); }
  };

  auto write_at = [&](const Win& w, bool tail, int seed) {
    for (int i = 0; i < N; ++i) {
      bp[i] = to_bf16_(1.0f + (float)((i + seed) % 11));
    }
    const std::size_t byte_off =
        tail ? w.off + w.bytes - slab * 2 : w.off;
    SharedBuffer sub = arena.subview(byte_off, slab * 2);
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      e.set_function(fn);
      e.set_buffer(0, sub);
      e.set_buffer(1, bias);
      e.set_constant(2, N);
      e.set_constant(3, (unsigned)slab);
      e.dispatch({(unsigned)slab, 1, 1}, {256, 1, 1});
    }
    st.commit().wait();
    return byte_off;
  };

  int checked = 0;
  bool clean = true;
  int seed = 0;
  for (const Win& w : wins) {
    for (int tail = 0; tail < 2; ++tail) {
      zero_spots();
      const std::size_t at = write_at(w, tail != 0, ++seed);
      // 1. the slab holds its pattern, at the ABSOLUTE offset asked for
      std::size_t wrong = 0;
      for (std::size_t i = 0; i < slab; ++i) {
        if (ap[at / 2 + i] != bp[i % (std::size_t)N]) { ++wrong; }
      }
      // 2. every OTHER inspected slab is untouched
      std::size_t spill = 0;
      for (std::size_t e : spots) {
        if (e == at / 2) { continue; }
        for (std::size_t i = 0; i < slab; ++i) {
          if (ap[e + i] != 0) { ++spill; }
        }
      }
      if (wrong != 0 || spill != 0) {
        clean = false;
        std::printf("[h3_blocks] %-4s %-4s at %.3f GB: wrong %zu, spill "
                    "%zu\n", w.tag, tail ? "tail" : "head",
                    (double)at / 1073741824.0, wrong, spill);
      }
      EXPECT_TRUE(wrong == 0);
      EXPECT_TRUE(spill == 0);
      ++checked;
    }
  }
  std::printf("[h3_blocks] arena windows: %d slabs, %s\n", checked,
              clean ? "each addresses its own slice" : "ALIASED OR TRUNCATED");
  EXPECT_TRUE(checked > 0);
}

// A SUBVIEW BOUND PAST 4 GB HAS TO ADDRESS ITS OWN SLICE.
//
// Every other probe here allocates a buffer and binds it at offset 0.
// The forward does not. qh|kh|vh|oh|ob and ff|proj are WINDOWS into one
// attention arena, and the window offsets are rows * inner * 2 apart --
// so `ob`, which every block writes its attention output into, is bound
// at 4 * rows * inner * 2 = rows * 57344 bytes.
//
// That crosses 2^32 at 74898 rows. A 209-frame clip packs ~65800 rows
// (3.77 GB, under) and a 243-frame one ~76250 (4.37 GB, over) -- which is
// exactly the gap between the tallest generation that comes back clean
// and the one that comes back as whole-latent noise in both modalities.
// `ob` being misaddressed would do precisely that: every block's
// attention output written somewhere else, for the whole sequence, so
// video and audio rows alike.
//
// The offset path is std::size_t into NS::UInteger and both are 64-bit,
// so this SHOULD be fine. Every wrong answer in this investigation has
// been something that should have been fine, so measure it: put a
// subview past 4 GB, write a known pattern through it, and check that
// the bytes landed at the right absolute offset in the parent -- AND
// that the low-memory alias a truncated offset would have hit is still
// untouched. A 32-bit truncation would write at (off mod 2^32), which
// for this offset is ~77 MB into the parent, so the alias check is what
// names the failure rather than merely detecting it.
//
// Env: VPIPE_H3_ROW_PROBE=1 (allocates ~4.5 GB).
TEST(minimax_h3_blocks, subview_past_four_gigabytes_addresses_its_slice)
{
  if (std::getenv("VPIPE_H3_ROW_PROBE") == nullptr) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise_bf16");
  ComputeFunction fn = lib_elt.function("bias_add_rows_f16");
  if (!fn.valid()) { return; }

  // The real geometry: `ob` at a 243-frame clip. 76288 rows is the packed
  // height rounded to a tile, and 4 * 76288 * 7168 * 2 is where its window
  // begins.
  const std::size_t rows = 76288, inner = kInner;
  const std::size_t off = 4 * rows * inner * 2;      // 4.37 GB
  const std::size_t win = rows * inner * 2;
  ASSERT_TRUE(off > (std::size_t)1 << 32);
  const std::size_t alias = off & 0xFFFFFFFFull;     // where truncation lands
  std::printf("[h3_blocks] subview offset %.3f GB, 32-bit alias at "
              "%.1f MB\n", (double)off / 1073741824.0,
              (double)alias / 1048576.0);

  // Only the first slab of the window is written, which is all the check
  // needs and keeps the dispatch small.
  const int N = 4096, M = 256;
  SharedBuffer parent = mc->make_shared_buffer(off + win);
  SharedBuffer bias = mc->make_shared_buffer((std::size_t)N * 2);
  if (parent.empty() || bias.empty()) {
    std::printf("[h3_blocks] subview probe needs %.1f GB; skipped\n",
                (double)(off + win) / 1073741824.0);
    return;
  }
  auto* pp = static_cast<std::uint16_t*>(parent.contents());
  auto* bp = static_cast<std::uint16_t*>(bias.contents());
  for (int i = 0; i < N; ++i) { bp[i] = to_bf16_(1.0f + (float)(i % 7)); }
  // Zero only what is inspected: the target slab, the alias slab and the
  // head of the buffer. Zeroing 4.4 GB would dominate the test.
  const std::size_t n_el = (std::size_t)M * N;
  std::memset(pp + off / 2, 0, n_el * 2);
  std::memset(pp + alias / 2, 0, n_el * 2);
  std::memset(pp, 0, n_el * 2);

  SharedBuffer sub = parent.subview(off, win);
  ASSERT_TRUE(!sub.empty());
  ASSERT_TRUE(sub.byte_offset() == off);
  {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder e = st.begin_compute();
      e.set_function(fn);
      e.set_buffer(0, sub);
      e.set_buffer(1, bias);
      e.set_constant(2, N);
      e.set_constant(3, (unsigned)n_el);
      e.dispatch({(unsigned)n_el, 1, 1}, {256, 1, 1});
    }
    st.commit().wait();
  }
  // 1. The slice itself holds the pattern.
  std::size_t wrong = 0;
  for (std::size_t i = 0; i < n_el; ++i) {
    if (pp[off / 2 + i] != bp[i % (std::size_t)N]) { ++wrong; }
  }
  // 2. Nothing landed at the 32-bit alias, or at the buffer's head.
  std::size_t at_alias = 0, at_head = 0;
  for (std::size_t i = 0; i < n_el; ++i) {
    if (pp[alias / 2 + i] != 0) { ++at_alias; }
    if (pp[i] != 0) { ++at_head; }
  }
  std::printf("[h3_blocks] slice wrong %zu, written at alias %zu, at head "
              "%zu\n", wrong, at_alias, at_head);
  EXPECT_TRUE(wrong == 0);
  EXPECT_TRUE(at_alias == 0);
  EXPECT_TRUE(at_head == 0);

  // And the CPU view has to agree with the GPU's: contents() carries the
  // same offset, so a reader that took the subview rather than the parent
  // must see what the kernel wrote.
  const auto* sp = static_cast<const std::uint16_t*>(sub.contents());
  std::size_t cpu_wrong = 0;
  for (std::size_t i = 0; i < n_el; ++i) {
    if (sp[i] != bp[i % (std::size_t)N]) { ++cpu_wrong; }
  }
  std::printf("[h3_blocks] subview contents() disagreeing %zu\n", cpu_wrong);
  EXPECT_TRUE(cpu_wrong == 0);
}

// THE BAND LOOP ITSELF: many dispatches must equal one.
//
// gemm_row_invariance drives each route at a single M. The forward does
// not: above mma_row_band_(N) it splits the rows and issues SEVERAL
// dispatches, each with its own element offset into the source and the
// destination. That loop is untested, and it is the only code path that
// turns on between the tallest clean generation (48504 rows) and the one
// that returns noise (74304) -- so its offset arithmetic, and the shared
// scratches it re-uses across bands, are exactly what is left.
//
// THE BAR IS BIT-EXACT, and it is available because banding does not
// change what any row computes: band b's dispatch sees the same weight,
// the same rows and the same contraction order it would have seen alone.
// Anything else -- an offset scaled by the wrong width, a shared scratch
// read after the next band overwrote it, a band boundary off a tile -- is
// a different answer, not a rounder one.
//
// M is held UNDER the shipped band so the single-dispatch arm is the
// reference rather than a second suspect: the point here is the loop, not
// the cliff (mma_dest_limit_probe covers that). The band is then forced
// small by hand, which is what VPIPE_H3_MMA_ROW_BAND does in the model.
//
// The int8 arm matters most. Its activation scratch is written from row 0
// on every call, so two bands write DIFFERENT contents to one buffer
// inside one encoder -- the only shared scratch on the path whose bytes
// differ between bands (the bf16 route's dequant scratch holds the same
// weight either way, so re-writing it is harmless).
//
// Env: VPIPE_H3_ROW_PROBE=1.
TEST(minimax_h3_blocks, banded_dispatch_matches_one_shot)
{
  if (std::getenv("VPIPE_H3_ROW_PROBE") == nullptr) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.steel[1].valid()) { return; }
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise_bf16");

  // Under every shipped band (qkv 49920, fc1 37376), so the one-shot arm
  // is known good, and not a multiple of any band tried below.
  const int M = 36096;
  struct Shape { const char* tag; int N, K; };
  const Shape shapes[] = {
      {"qkv", 3 * kInner, kHidden},
      {"fc1", 2 * kFfn,   kHidden},
      {"fc2", kHidden,    kFfn},
  };
  const int bands[] = {8192, 12800, 16384};

  std::uint16_t tab[256];
  for (int i = 0; i < 256; ++i) {
    tab[i] = to_bf16_(((float)i / 255.0f) * 0.4f - 0.2f);
  }
  int checked = 0;
  bool clean = true;
  for (const Shape& sh : shapes) {
    SharedBuffer x = mc->make_shared_buffer((std::size_t)M * sh.K * 2);
    SharedBuffer y = mc->make_shared_buffer((std::size_t)M * sh.N * 2);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)sh.N * sh.K * 2);
    QW q = make_qw_(mc, sh.N, sh.K, 8, (std::uint32_t)(sh.N + 8));
    if (x.empty() || y.empty() || wdq.empty() || q.w.empty()) { continue; }
    {
      auto* p = static_cast<std::uint16_t*>(x.contents());
      for (std::size_t r = 0; r < (std::size_t)M; ++r) {
        std::uint16_t* row = p + r * (std::size_t)sh.K;
        for (int c = 0; c < sh.K; ++c) { row[c] = tab[(r + c) & 255]; }
      }
    }
    vpipe::genai::I8GemmContext i8(mc, /*want=*/true, /*bf16=*/true);
    vpipe::genai::MmaSplitK splitk;
    splitk.load(mc, kn.lib_dense, lib_elt);

    // Encode ONE band: rows [m0, m0+rows) with the offsets the forward
    // hands down. band == M is the one-shot arm.
    auto encode_band = [&](ComputeEncoder& e, int mode, int m0, int rows) {
      const std::size_t xo = (std::size_t)m0 * sh.K;
      const std::size_t yo = (std::size_t)m0 * sh.N;
      if (mode == 0) {
        // The steel qmm contract is qmm_dispatch_'s, verbatim: the packed
        // weight comes FIRST and the activation is buffer 3, and the grid
        // is (N tiles, M tiles x 2, tgz) over a {32, 2, tgz} threadgroup.
        // Hand-rolling it from the DENSE GEMM's grid instead -- which is
        // what the first cut of this test did -- writes a prefix of the
        // rows and then reports the untouched remainder as a banding
        // failure, at a row that is not a band boundary.
        constexpr int bm = 32;
        constexpr unsigned tgz = 2u;
        e.set_function(kn.steel[1]);
        e.set_buffer(0, q.w); e.set_buffer(1, q.s); e.set_buffer(2, q.b);
        e.set_buffer(3, x, xo * 2);
        e.set_buffer(4, y, yo * 2);
        e.set_constant(5, sh.K); e.set_constant(6, sh.N);
        e.set_constant(7, rows);
        e.dispatch({(unsigned)(((sh.N + 31) / 32) * 32),
                    (unsigned)(((rows + bm - 1) / bm) * 2), tgz},
                   {32, 2, tgz});
        return true;
      }
      e.set_function(kn.dq[1]);
      e.set_buffer(0, q.w); e.set_buffer(1, q.s); e.set_buffer(2, q.b);
      e.set_buffer(3, wdq);
      e.set_constant(4, sh.K); e.set_constant(5, sh.N);
      e.dispatch({(unsigned)(sh.K * 8 / 32), (unsigned)sh.N, 1}, {64, 1, 1});
      if (mode == 1) {
        e.set_function(kn.dense128);
        e.set_buffer(0, x, xo * 2); e.set_buffer(1, wdq);
        e.set_buffer(2, wdq); e.set_buffer(3, y, yo * 2);
        e.set_constant(4, sh.K); e.set_constant(5, sh.N);
        e.set_constant(6, rows); e.set_constant(7, 0);
        e.dispatch({(unsigned)(((sh.N + 127) / 128) * 256),
                    (unsigned)((rows + 127) / 128), 1}, {256, 1, 1});
        return true;
      }
      if (mode == 2) { return i8.gemm(e, x, xo, wdq, y, yo, rows, sh.N, sh.K); }
      return splitk.encode(mc, e, x, wdq, y, sh.K, sh.N, rows, xo, yo);
    };

    const char* mode_tag[] = {"steel", "mma128", "i8", "splitk"};
    for (int mode = 0; mode < 4; ++mode) {
      if (mode == 1 && !kn.have_mma) { continue; }
      if (mode == 2 && !i8.enabled()) { continue; }
      if (mode == 3 && !splitk.enabled) { continue; }
      // One shot, as the reference.
      std::vector<std::uint16_t> ref;
      {
        std::memset(y.contents(), 0, (std::size_t)M * sh.N * 2);
        bool ok = true;
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          ok = encode_band(e, mode, 0, M);
        }
        st.commit().wait();
        if (!ok) {
          std::printf("[h3_blocks] %s %s: shape refused, skipped\n", sh.tag,
                      mode_tag[mode]);
          continue;
        }
        const auto* p = static_cast<const std::uint16_t*>(y.contents());
        ref.assign(p, p + (std::size_t)M * sh.N);
      }
      for (int band : bands) {
        std::memset(y.contents(), 0, (std::size_t)M * sh.N * 2);
        bool ok = true;
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          for (int m0 = 0; m0 < M && ok; m0 += band) {
            ok = encode_band(e, mode, m0, std::min(band, M - m0));
          }
        }
        st.commit().wait();
        if (!ok) { continue; }
        const auto* p = static_cast<const std::uint16_t*>(y.contents());
        std::size_t bad = 0;
        long long first = -1;
        for (std::size_t i = 0; i < (std::size_t)M * sh.N; ++i) {
          if (p[i] != ref[i]) {
            ++bad;
            if (first < 0) { first = (long long)(i / sh.N); }
          }
        }
        std::printf("[h3_blocks] %-4s %-7s M=%d in %d bands of %5d  "
                    "differing %8zu (first row %lld)\n", sh.tag,
                    mode_tag[mode], M, (M + band - 1) / band, band, bad,
                    first);
        if (bad != 0) { clean = false; }
        EXPECT_TRUE(bad == 0);
        ++checked;
      }
    }
  }
  std::printf("[h3_blocks] band loop: %d checks, %s\n", checked,
              clean ? "banded == one-shot everywhere" : "BANDING CHANGES IT");
  EXPECT_TRUE(checked > 0);
}

// ATTENTION AT THE HEIGHTS THAT BREAK, AGAINST FLOAT64 GROUND TRUTH.
//
// Attention is the one kernel on the block path that row invariance
// cannot cover: every query attends to every key, so its output is
// SUPPOSED to change with the sequence length. It is also the kernel a
// sibling implementation of this model was just root-caused on --
// MPSGraph's scaledDotProductAttention silently returns garbage above
// ~15k query rows, which is the same symptom class this is chasing.
// vpipe does not call MPSGraph, and its flash kernel blocks the query
// dimension at 64 rows, so it should be structurally immune. "Should be"
// is what this test replaces.
//
// The method is the one that settled it there: spot rows against a
// double-precision CPU reference. That is ground truth rather than a
// cross-check, so a bug shared by both GPU kernels still fails it. Both
// arms are run anyway, because they are independent implementations and
// disagreement between them is informative on its own.
//
// A few rows suffice and are all that is affordable: one reference row is
// 2 * seq * head_dim double MACs, against the kernel's seq^2 -- so the
// check costs milliseconds where the dispatch it checks costs seconds.
// Rows are taken at both ends and the middle, on the first and last
// head, because a wrapped index is far likelier to strand the tail than
// the head.
//
// Env: VPIPE_H3_ROW_PROBE=1 (~4.3 GB and ~90 s of GPU).
TEST(minimax_h3_blocks, attn_matches_f64_at_video_heights)
{
  if (std::getenv("VPIPE_H3_ROW_PROBE") == nullptr) { return; }
  Session s;
  MetalCompute* mc = mc_(s);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.lib_attn.valid()) { return; }

  struct P {
    int B, H, D, qL, kL, gqa_factor;
    float scale;
    int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
    std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
  };
  const int NH = kHeads, HD = kHeadDim;
  const float scale = 1.0f / std::sqrt((float)HD);
  // 2048 as the control, then the tallest clean generation and the one
  // that comes back as noise. 1376x768 at 158 and 243 frames.
  // Attention is never banded, so it sees the WHOLE packed sequence --
  // video plus the keyframe condition, the audio rows and the text. At
  // 1376x768 that is ~55400 for a clean 175-frame clip and ~76250 for the
  // 243-frame one that returns noise.
  const int kSeqs[] = {2048, 55400, 76800};
  const int seq_max = kSeqs[std::size(kSeqs) - 1];

  const std::size_t n = (std::size_t)NH * seq_max * HD;
  SharedBuffer q = make_act_(mc, n, 3u);
  SharedBuffer k = make_act_(mc, n, 5u);
  SharedBuffer v = make_act_(mc, n, 9u);
  SharedBuffer o = mc->make_shared_buffer(n * 2);
  SharedBuffer pb = mc->make_shared_buffer(sizeof(float) * 64);
  if (q.empty() || k.empty() || v.empty() || o.empty() || pb.empty()) {
    std::printf("[h3_blocks] attention probe needs ~4.3 GB; skipped\n");
    return;
  }
  const auto* qp = static_cast<const std::uint16_t*>(q.contents());
  const auto* kp = static_cast<const std::uint16_t*>(k.contents());
  const auto* vp = static_cast<const std::uint16_t*>(v.contents());
  const auto* op = static_cast<const std::uint16_t*>(o.contents());

  // Exact attention for one (head, query row), in double. Softmax is
  // max-shifted for the same reason the kernel's is: without it the
  // exponentials overflow long before 74k terms.
  std::vector<double> ref(HD), pr;
  auto reference = [&](int seq, int h, int r) {
    const std::size_t hb = (std::size_t)h * seq * HD;
    pr.assign((std::size_t)seq, 0.0);
    double mx = -1e300;
    for (int j = 0; j < seq; ++j) {
      double acc = 0.0;
      for (int d = 0; d < HD; ++d) {
        acc += (double)from_bf16_(qp[hb + (std::size_t)r * HD + d]) *
               (double)from_bf16_(kp[hb + (std::size_t)j * HD + d]);
      }
      acc *= (double)scale;
      pr[(std::size_t)j] = acc;
      if (acc > mx) { mx = acc; }
    }
    double sum = 0.0;
    for (int j = 0; j < seq; ++j) {
      pr[(std::size_t)j] = std::exp(pr[(std::size_t)j] - mx);
      sum += pr[(std::size_t)j];
    }
    for (int d = 0; d < HD; ++d) { ref[(std::size_t)d] = 0.0; }
    for (int j = 0; j < seq; ++j) {
      const double w = pr[(std::size_t)j] / sum;
      for (int d = 0; d < HD; ++d) {
        ref[(std::size_t)d] +=
            w * (double)from_bf16_(vp[hb + (std::size_t)j * HD + d]);
      }
    }
  };

  int checked = 0;
  bool clean = true;
  for (int seq : kSeqs) {
    for (int arm = 0; arm < 2; ++arm) {
      const bool nax = (arm == 1);
      if (nax && (!mc->supports_matrix_cores() || !kn.lib_attn_nax.valid())) {
        continue;
      }
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
      ComputeFunction fn =
          nax ? kn.lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
              : kn.lib_attn.function("attn_steel_h_bd128_bf16", fc);
      if (!fn.valid()) { continue; }
      // Zeroed first: an unvalidated function is a silent no-op in this
      // framework, and zeros cannot pass the check below.
      std::memset(o.contents(), 0, (std::size_t)NH * seq * HD * 2);
      {
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          e.set_function(fn);
          e.set_buffer(0, q); e.set_buffer(1, k); e.set_buffer(2, v);
          e.set_buffer(3, o); e.set_buffer(4, pb);
          e.dispatch({32 * (unsigned)((seq + BQ - 1) / BQ),
                      4 * (unsigned)NH, 1}, {32, 4, 1});
        }
        st.commit().wait();
      }
      double worst = 0.0;
      int worst_h = -1, worst_r = -1;
      for (int h : {0, NH - 1}) {
        for (int r : {0, seq / 2, seq - 1}) {
          reference(seq, h, r);
          double num = 0.0, den = 0.0;
          const std::size_t base = ((std::size_t)h * seq + r) * HD;
          for (int d = 0; d < HD; ++d) {
            const double g = (double)from_bf16_(op[base + d]);
            const double e = ref[(std::size_t)d];
            num += (g - e) * (g - e);
            den += e * e;
          }
          const double rel = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
          if (rel > worst) { worst = rel; worst_h = h; worst_r = r; }
          ++checked;
        }
      }
      std::printf("[h3_blocks] attn %-5s seq=%6d  worst rel-L2 %.3e "
                  "(head %d row %d)\n", nax ? "nax" : "steel", seq, worst,
                  worst_h, worst_r);
      // bf16 inputs with f32 accumulation over up to 74k keys: a correct
      // answer sits near 1e-3. A dropped tile, a wrapped index or an
      // unwritten row is O(1), three orders clear of any tolerance
      // argument.
      if (!(worst < 5e-2)) { clean = false; }
      EXPECT_TRUE(worst < 5e-2);
    }
  }
  std::printf("[h3_blocks] attention: %d rows checked, %s\n", checked,
              clean ? "matches f64 at every height" : "DIVERGES");
  EXPECT_TRUE(checked > 0);
}

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
  struct Case { int M, N, K, r; bool bmma; };
  std::vector<Case> kCases = {
      {300, kHidden, kInner, 48, true},       // ragged M and ragged rank
      {256, 3 * kInner, kHidden, 64, true},   // the shipped Turbo rank
      {384, 2 * kFfn, kHidden, 128, true},    // the lightx2v rank
      {256, 3 * kInner, kHidden, 64, false},  // what rank 64 ACTUALLY runs
      {3864, kHidden, kInner, 64, false},     // ...at a production row count
  };
  // VIDEO row counts, which is where the adapter actually runs and where
  // nothing had checked it: 3864 rows is an image geometry, and a
  // 1376x768 clip packs ~55400 rows at 175 frames and ~76250 at 243.
  //
  // This is the adapter path the model REACHES on that box, which is not
  // the obvious one. gemm_mma_ tries the int8 arm and split-K before the
  // fused-LoRA tile, and returns as soon as one of them takes the base
  // projection -- so with i8_gemm on, the fold never happens and the
  // adapter goes through this separate pair of GEMMs on every projection.
  //
  // Gated, because 76288 rows is ~3.7 GB of buffers: the default suite
  // keeps the small cases, which are the ones about ragged tiles.
  if (std::getenv("VPIPE_H3_ROW_PROBE") != nullptr) {
    kCases.push_back({55400, kHidden, kInner, 64, false});
    kCases.push_back({76288, kHidden, kInner, 64, false});
  }
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

// ---- the >2 GB fc1 destination -----------------------------------------
//
// A user reported the tail of a 1344x768 / 124-frame clip coming back
// corrupt -- the last four pixel frames, bottom of the image. Video is
// packed LAST in this model's sequence, so "the tail of the picture" and
// "the tail of the packed sequence" are the same rows, which is what an
// index that stops counting would look like.
//
// At that geometry the sequence is 38222 rows and the fc1 intermediate is
// [38222, 2*14336] bf16 = 2,191,802,368 bytes: the ONE buffer this model
// allocates that crosses 2^31. Row 37449 is where a byte offset would
// wrap, leaving the last 773 rows -- the bottom ~77% of the final latent
// frame -- reading from somewhere else. At 960x544, the geometry that
// works, the same buffer is 1.13 GB and the question never arises.
//
// So: run the real fc1 shape into a real 2.19 GB destination and check
// every row. y[m][n] = f(m) * g(n) with both factors carried in K = 0,
// which makes a row that picked up another row's data, or none at all,
// a wrong number rather than a plausible one.
//
// Gated: it allocates ~2.9 GB, which is not a default suite's business.
TEST(minimax_h3_blocks, fc1_destination_past_2gb)
{
  if (std::getenv("VPIPE_H3_BIG_GEMM") == nullptr) {
    std::printf("[ SKIP     ] set VPIPE_H3_BIG_GEMM=1 (allocates ~2.9 GB)\n");
    return;
  }
  auto session = std::make_shared<Session>();
  MetalCompute* mc = session->metal_compute();
  if (mc == nullptr) { EXPECT_TRUE(false); return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.have_mma) {
    std::printf("[ SKIP     ] no matrix cores\n");
    return;
  }
  // 38222 = the packed sequence of 1344x768 at 124 frames; 37449 is
  // floor(2^31 / (2*14336*2)), the row a 32-bit BYTE offset stops at.
  const int M = 38222, N = 2 * kFfn, K = kHidden;
  const int kWrapRow = (int)((1LL << 31) / ((long long)N * 2));
  const std::size_t elems = (std::size_t)M * N;
  std::printf("  fc1 [%d, %d] = %zu elems, %.3f GB (2^31 = 2.147 GB), "
              "wrap row %d\n", M, N, elems, (double)(elems * 2) / 1e9,
              kWrapRow);

  SharedBuffer x = mc->make_shared_buffer((std::size_t)M * K * 2);
  SharedBuffer w = mc->make_shared_buffer((std::size_t)N * K * 2);
  SharedBuffer y = mc->make_shared_buffer(elems * 2);
  if (x.empty() || w.empty() || y.empty()) {
    std::printf("  allocation failed -- not enough memory for this test\n");
    EXPECT_TRUE(false);
    return;
  }
  auto fm = [](int m) { return 1.0f + (float)(m % 128) / 128.0f; };
  auto gn = [](int n) { return 1.0f + (float)(n % 128) / 128.0f; };
  {
    auto* p = static_cast<std::uint16_t*>(x.contents());
    std::memset(p, 0, (std::size_t)M * K * 2);
    for (int m = 0; m < M; ++m) { p[(std::size_t)m * K] = to_bf16_(fm(m)); }
    auto* q = static_cast<std::uint16_t*>(w.contents());
    std::memset(q, 0, (std::size_t)N * K * 2);
    for (int n = 0; n < N; ++n) { q[(std::size_t)n * K] = to_bf16_(gn(n)); }
  }
  // A distinctive poison, so a row the GEMM never reached is a loud
  // failure rather than a zero that some other bug could also explain.
  {
    auto* p = static_cast<std::uint16_t*>(y.contents());
    for (std::size_t i = 0; i < elems; ++i) { p[i] = 0xFF7F; }   // -3.4e38
  }

  for (int arm = 0; arm < 2; ++arm) {
    const ComputeFunction& fn = arm == 0 ? kn.dense128 : kn.dense256;
    const int RN = arm == 0 ? 128 : 256;
    const char* name = arm == 0 ? "mma128" : "mma128x256";
    if (!fn.valid()) { continue; }
    CommandStream stream = mc->make_command_stream();
    {
      ComputeEncoder enc = stream.begin_compute();
      enc.set_function(fn);
      enc.set_buffer(0, x);
      enc.set_buffer(1, w);
      enc.set_buffer(2, w);            // bias slot, unread
      enc.set_buffer(3, y);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    }
    stream.commit().wait();

    // Every row, at a spread of columns: a band that wrapped shows up
    // wherever it is, and 38222 x 8 is cheap to check on the host.
    const int cols[] = {0, 1, 127, 4096, 12345, N / 2, N - 129, N - 1};
    const auto* p = static_cast<const std::uint16_t*>(y.contents());
    int bad = 0, first_bad = -1, last_good = -1;
    double worst = 0.0;
    for (int m = 0; m < M; ++m) {
      bool row_ok = true;
      for (int c : cols) {
        const float got = from_bf16_(p[(std::size_t)m * N + c]);
        const float want = fm(m) * gn(c);
        const double e = std::fabs((double)got - (double)want);
        if (e > worst) { worst = e; }
        if (!(e <= 0.05)) { row_ok = false; }
      }
      if (!row_ok) {
        ++bad;
        if (first_bad < 0) { first_bad = m; }
      } else {
        last_good = m;
      }
    }
    std::printf("  %-11s bad rows %d/%d  first bad %d  last good %d  "
                "worst |err| %.4f\n", name, bad, M, first_bad, last_good,
                worst);
    if (bad > 0) {
      std::printf("  >>> first bad row %d vs the 2^31-byte wrap row %d "
                  "(%+d, the 128-row tile it sits in)\n", first_bad,
                  kWrapRow, first_bad - kWrapRow);
    }
  }
  // REPORTED, not asserted. What is asserted is that the banded encoding
  // the transformer emits gets it right -- see the test below. Failing
  // here on the day MPP starts addressing past 2^31 would be a platform
  // improvement reported as a regression.
  EXPECT_TRUE(true);
}

// The same shape through the BANDED encoding the transformer now emits:
// each dispatch rebased so no tile ever computes an offset past the line.
// Same kernel, same arithmetic, one extra dispatch -- so this passing
// while the test above fails is the whole of the fix.
TEST(minimax_h3_blocks, fc1_destination_past_2gb_banded)
{
  if (std::getenv("VPIPE_H3_BIG_GEMM") == nullptr) {
    std::printf("[ SKIP     ] set VPIPE_H3_BIG_GEMM=1 (allocates ~2.9 GB)\n");
    return;
  }
  auto session = std::make_shared<Session>();
  MetalCompute* mc = session->metal_compute();
  if (mc == nullptr) { EXPECT_TRUE(false); return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.have_mma) {
    std::printf("[ SKIP     ] no matrix cores\n");
    return;
  }
  const int M = 38222, N = 2 * kFfn, K = kHidden;
  const std::size_t elems = (std::size_t)M * N;
  // mma_row_band_(28672): floor((2^31 - 1) / 57344 / 128) * 128.
  const int band = (int)(((((long long)1 << 31) - 1) /
                          ((long long)N * 2)) / 128) * 128;
  std::printf("  band %d rows -> %d dispatches (base of the last: "
              "%.3f GB)\n", band, (M + band - 1) / band,
              (double)((std::size_t)(M / band * band) * N * 2) / 1e9);

  SharedBuffer x = mc->make_shared_buffer((std::size_t)M * K * 2);
  SharedBuffer w = mc->make_shared_buffer((std::size_t)N * K * 2);
  SharedBuffer y = mc->make_shared_buffer(elems * 2);
  if (x.empty() || w.empty() || y.empty()) {
    std::printf("  allocation failed -- not enough memory for this test\n");
    EXPECT_TRUE(false);
    return;
  }
  auto fm = [](int m) { return 1.0f + (float)(m % 128) / 128.0f; };
  auto gn = [](int n) { return 1.0f + (float)(n % 128) / 128.0f; };
  {
    auto* p = static_cast<std::uint16_t*>(x.contents());
    std::memset(p, 0, (std::size_t)M * K * 2);
    for (int m = 0; m < M; ++m) { p[(std::size_t)m * K] = to_bf16_(fm(m)); }
    auto* q = static_cast<std::uint16_t*>(w.contents());
    std::memset(q, 0, (std::size_t)N * K * 2);
    for (int n = 0; n < N; ++n) { q[(std::size_t)n * K] = to_bf16_(gn(n)); }
    auto* r = static_cast<std::uint16_t*>(y.contents());
    for (std::size_t i = 0; i < elems; ++i) { r[i] = 0xFF7F; }
  }

  const int RN = 128;
  CommandStream stream = mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    for (int m0 = 0; m0 < M; m0 += band) {
      const int rows = std::min(band, M - m0);
      enc.set_function(kn.dense128);
      enc.set_buffer(0, x, (std::size_t)m0 * K * 2);
      enc.set_buffer(1, w);
      enc.set_buffer(2, w);
      enc.set_buffer(3, y, (std::size_t)m0 * N * 2);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, rows);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                    (unsigned)((rows + 127) / 128), 1}, {256, 1, 1});
    }
  }
  stream.commit().wait();

  const int cols[] = {0, 1, 127, 4096, 12345, N / 2, N - 129, N - 1};
  const auto* p = static_cast<const std::uint16_t*>(y.contents());
  int bad = 0, first_bad = -1;
  double worst = 0.0;
  for (int m = 0; m < M; ++m) {
    for (int c : cols) {
      const float got = from_bf16_(p[(std::size_t)m * N + c]);
      const double e = std::fabs((double)got - (double)(fm(m) * gn(c)));
      if (e > worst) { worst = e; }
      if (!(e <= 0.05)) {
        ++bad;
        if (first_bad < 0) { first_bad = m; }
      }
    }
  }
  std::printf("  banded     bad %d  first bad row %d  worst |err| %.4f\n",
              bad, first_bad, worst);
  EXPECT_TRUE(bad == 0);
}
