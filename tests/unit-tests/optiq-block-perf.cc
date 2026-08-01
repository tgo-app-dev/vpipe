// optiq-block-perf.cc -- block-level kernel selection for the four large
// OptiQ checkpoints on M5 matrix-core (matmul2d) hardware:
//
//   mlx-community/Qwen3.6-27B-OptiQ-4bit          dense, hybrid GDN
//   mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit      MoE, 256 routed experts
//   mlx-community/gemma-4-31B-it-OptiQ-4bit       dense
//   mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit   MoE, 128 routed experts
//
// All four were brought up on a box with NO matrix cores, and none of them
// fits in 16 GB, so the M5 cannot run them end to end and cannot A/B a route
// the way the smaller checkpoints are A/B'd. What it CAN do is run every
// prefill GEMM they dispatch, at the real [N, K, bits] taken from each
// checkpoint's config.json (shapes + per-tensor OptiQ widths are recorded
// per block below), against every kernel that could serve it. That is enough
// to decide whether the shipped router picks the fastest kernel on M5, which
// is the question a 64 GB non-matrix-core box cannot answer.
//
// Correctness of each kernel is NOT re-tested here -- gemm_mma /qmm_mma
// already pin dense_gemm_mma, affine_qmm_mma and the dequant+dense sequence
// against a CPU oracle and against steel. This file measures rate only, and
// deliberately materializes no f32 oracle: the 31B's gate|up block is
// 43008x5376, whose dequantized copy alone would be ~900 MB.
//
// The default set is the FLOP-dominant blocks at two row counts.
// VPIPE_OPTIQ_BLOCKS (any value) also runs mlp_tile_probe; =full additionally
// widens both sweeps to every block and a 64..4096 row ladder.

#include "minitest.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::metal_compute;

namespace {

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

bool opted_in_()
{
  return std::getenv("VPIPE_OPTIQ_BLOCKS") != nullptr;
}

bool full_sweep_()
{
  const char* e = std::getenv("VPIPE_OPTIQ_BLOCKS");
  return e != nullptr && std::string(e) == "full";
}

struct Shape3 {
  int M, N, K;
  const char* tag;
};

// A packed affine weight (group 64) + its scales/biases, filled with
// deterministic garbage. The VALUES do not matter -- every kernel here reads
// the same bytes and does the same work regardless -- and skipping a CPU-side
// dequant keeps a 43008x5376 block inside a few hundred MB.
struct QW {
  SharedBuffer w, s, b;
};

QW make_qw_(MetalCompute* mc, int N, int K, int bits, std::uint32_t seed)
{
  const std::size_t codes = (std::size_t)N * K * bits / 8;
  const std::size_t ng = (std::size_t)N * (K / 64);
  QW q{mc->make_shared_buffer(codes), mc->make_shared_buffer(ng * 2),
       mc->make_shared_buffer(ng * 2)};
  auto* wp = static_cast<std::uint8_t*>(q.w.contents());
  std::uint32_t r = seed | 1u;
  for (std::size_t i = 0; i < codes; ++i) {
    r = r * 1664525u + 1013904223u;
    wp[i] = (std::uint8_t)(r >> 24);
  }
  auto* sp = static_cast<_Float16*>(q.s.contents());
  auto* bp = static_cast<_Float16*>(q.b.contents());
  for (std::size_t i = 0; i < ng; ++i) {
    sp[i] = (_Float16)0.01f;
    bp[i] = (_Float16)(-0.08f);
  }
  return q;
}

SharedBuffer make_act_(MetalCompute* mc, std::size_t n, std::uint32_t seed)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  auto* p = static_cast<_Float16*>(b.contents());
  std::uint32_t r = seed | 1u;
  for (std::size_t i = 0; i < n; ++i) {
    r = r * 1664525u + 1013904223u;
    p[i] = (_Float16)(((float)(r >> 16) / 32768.0f - 1.0f) * 0.1f);
  }
  return b;
}

// The routes a prefill projection can take on M5. Steel is the simdgroup_
// matrix quantized GEMM (the only route on a non-matrix-core box); DqDense is
// the shipped M5 route (expand the weight with affine_dequant, then the dense
// matmul2d); FusedMma is affine_qmm_mma, which dequantizes into threadgroup
// memory inside the matmul2d K loop and so never writes an f16 weight copy.
enum class Route {
  Steel32,
  Steel64,
  Steel128,
  DqDense128,
  DqDense256,
  FusedMma,
};

const char* route_tag_(Route r)
{
  switch (r) {
    case Route::Steel32:    return "steel32";
    case Route::Steel64:    return "steel64";
    case Route::Steel128:   return "steel128";
    case Route::DqDense128: return "dq+n128";
    case Route::DqDense256: return "dq+n256";
    case Route::FusedMma:   return "fused-mma";
  }
  return "?";
}

// What metal-{qwen,gemma}-model.cc actually dispatches for a prefill GEMM of
// `rows` rows: the matrix-core route past _mma_min_m (64 on both), with the
// deep 128x256 tile at K >= 6144, else the 32-row steel tile.
Route shipped_route_(int rows, int K, bool have_mma)
{
  if (!have_mma || rows < 64) { return Route::Steel32; }
  return K >= 6144 ? Route::DqDense256 : Route::DqDense128;
}

// Kernel handles, loaded once per test.
struct Kernels {
  ComputeLibrary lib_steel, lib_dq, lib_dense, lib_mma;
  ComputeFunction steel[2], steel64[2], steel128[2];   // [0]=w4 [1]=w8
  ComputeFunction dq[2];
  ComputeFunction dense128, dense256;
  ComputeFunction fused;                               // w4 only
  bool have_mma = false;

  void load(MetalCompute* mc)
  {
    lib_steel = mc->load_library("affine_qmm_steel");
    lib_dq    = mc->load_library("affine_dequant");
    lib_dense = mc->load_library("dense_gemm_mma");
    lib_mma   = mc->load_library("affine_qmm_mma");
    steel[0]    = lib_steel.function("affine_qmm_steel_w4g64");
    steel[1]    = lib_steel.function("affine_qmm_steel_w8g64");
    steel64[0]  = lib_steel.function("affine_qmm_steel_w4g64_bm64");
    steel64[1]  = lib_steel.function("affine_qmm_steel_w8g64_bm64");
    steel128[0] = lib_steel.function("affine_qmm_steel_w4g64_bm128");
    steel128[1] = lib_steel.function("affine_qmm_steel_w8g64_bm128");
    dq[0]    = lib_dq.function("affine_dequant_w4g64");
    dq[1]    = lib_dq.function("affine_dequant_w8g64");
    dense128 = lib_dense.function("dense_gemm_mma_t_n128_f16");
    dense256 = lib_dense.function("dense_gemm_mma_t_n128x256_f16");
    fused    = lib_mma.function("affine_qmm_mma_w4g64");
    have_mma = mc->supports_matrix_cores() && dense128.valid();
  }

  bool available(Route r, int bits) const
  {
    const int bi = (bits == 8) ? 1 : 0;
    switch (r) {
      case Route::Steel32:    return steel[bi].valid();
      case Route::Steel64:    return steel64[bi].valid();
      case Route::Steel128:   return steel128[bi].valid();
      case Route::DqDense128: return have_mma && dq[bi].valid();
      case Route::DqDense256: return have_mma && dense256.valid()
                                     && dq[bi].valid();
      // affine_qmm_mma has a 4-bit instantiation only. Every 8-bit OptiQ
      // tensor is therefore restricted to steel or dequant+dense.
      case Route::FusedMma:   return have_mma && bits == 4 && fused.valid();
    }
    return false;
  }
};

// One GEMM through one route. `wdq` is the shared dequant scratch (only the
// DqDense routes touch it); it is sized by the caller to the largest block.
void encode_(ComputeEncoder& enc, const Kernels& kn, Route r, const QW& q,
             const SharedBuffer& x, const SharedBuffer& y,
             const SharedBuffer& wdq, int M, int N, int K, int bits)
{
  const int bi = (bits == 8) ? 1 : 0;
  if (r == Route::DqDense128 || r == Route::DqDense256) {
    enc.set_function(kn.dq[bi]);
    enc.set_buffer(0, q.w);
    enc.set_buffer(1, q.s);
    enc.set_buffer(2, q.b);
    enc.set_buffer(3, wdq);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.dispatch({(unsigned)(K * bits / 32), (unsigned)N, 1}, {64, 1, 1});
    const int BN = (r == Route::DqDense256) ? 256 : 128;
    enc.set_function(r == Route::DqDense256 ? kn.dense256 : kn.dense128);
    enc.set_buffer(0, x);
    enc.set_buffer(1, wdq);
    enc.set_buffer(2, wdq);          // bias slot unused (has_bias = 0)
    enc.set_buffer(3, y);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    return;
  }
  // Quantized GEMMs share one binding; only the tile / grid differ.
  switch (r) {
    case Route::Steel32:  enc.set_function(kn.steel[bi]); break;
    case Route::Steel64:  enc.set_function(kn.steel64[bi]); break;
    case Route::Steel128: enc.set_function(kn.steel128[bi]); break;
    default:              enc.set_function(kn.fused); break;
  }
  enc.set_buffer(0, q.w);
  enc.set_buffer(1, q.s);
  enc.set_buffer(2, q.b);
  enc.set_buffer(3, x);
  enc.set_buffer(4, y);
  enc.set_constant(5, K);
  enc.set_constant(6, N);
  enc.set_constant(7, M);
  switch (r) {
    case Route::Steel32:
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
      break;
    case Route::Steel64:
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
      break;
    case Route::Steel128:
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + 127) / 128) * 2), 4}, {32, 2, 4});
      break;
    default:
      enc.dispatch({(unsigned)(((N + 63) / 64) * 128),
                    (unsigned)((M + 63) / 64), 1}, {128, 1, 1});
      break;
  }
}

// GFLOP/s for one (block, route, M). Warms up, then times a run whose length
// adapts to the block so every measurement covers a comparable slice of GPU
// work -- the 31B's gate|up is 40x the FLOPs of a 26B attention projection.
double bench_(MetalCompute* mc, const Kernels& kn, Route r, const QW& q,
              const SharedBuffer& x, const SharedBuffer& y,
              const SharedBuffer& wdq, int M, int N, int K, int bits)
{
  const double f1 = 2.0 * M * N * K;
  const int iters = std::min(40, std::max(3, (int)(3.0e11 / f1)));
  for (int w = 0; w < 2; ++w) {
    CommandStream st = mc->make_command_stream();
    {
      ComputeEncoder enc = st.begin_compute();
      encode_(enc, kn, r, q, x, y, wdq, M, N, K, bits);
    }
    st.commit().wait();
  }
  const auto t0 = std::chrono::steady_clock::now();
  CommandStream st = mc->make_command_stream();
  {
    ComputeEncoder enc = st.begin_compute();
    for (int i = 0; i < iters; ++i) {
      encode_(enc, kn, r, q, x, y, wdq, M, N, K, bits);
    }
  }
  st.commit().wait();
  const auto t1 = std::chrono::steady_clock::now();
  return f1 * iters / 1e9 / secs_(t0, t1);
}

// One weight block of one checkpoint. `layers` is how many decoder layers
// carry this block AT THIS WIDTH (OptiQ picks the width per tensor, so a
// projection usually appears twice in a model's list, once per width); it
// weights the per-model rollup. `hot` marks the blocks the default sweep
// runs -- together they are the bulk of each model's prefill GEMM FLOPs.
struct Block {
  const char* model;
  const char* tag;
  int N, K, bits, layers;
  bool hot;
};

// Shapes from each checkpoint's config.json; widths + layer counts from its
// `quantization` map. Qwen 27B: hidden 5120, ffn 17408, 16 full-attention
// layers (q 24x256, kv 4x256) + 48 GDN layers (in_proj_qkv = 16x128 q +
// 16x128 k + 48x128 v, out_proj K = 48x128). Qwen 35B-A3B: hidden 2048,
// shared expert 512. gemma 31B: hidden 5376, ffn 21504, q 32x256, kv 16x256.
// gemma 26B-A4B: hidden 2816, dense ffn 2112, q 16x256, kv 8x256.
const Block kBlocks[] = {
    // ---- Qwen3.6-27B (dense) ---------------------------------------
    {"qwen-27B", "mlp gate|up w4", 34816, 5120, 4, 44, true},
    {"qwen-27B", "mlp up w8",      17408, 5120, 8, 20, true},
    {"qwen-27B", "mlp down w4",     5120, 17408, 4, 62, true},
    {"qwen-27B", "gdn in_qkv w4",  10240, 5120, 4, 47, true},
    {"qwen-27B", "gdn out w8",      5120, 6144, 8, 48, false},
    {"qwen-27B", "gdn z w4",        6144, 5120, 4, 40, false},
    {"qwen-27B", "attn q w4",       6144, 5120, 4, 15, false},
    {"qwen-27B", "attn o w8",       5120, 6144, 8, 16, false},
    // ---- gemma-4-31B (dense) ---------------------------------------
    {"gemma-31B", "mlp gate|up w4", 43008, 5376, 4, 37, true},
    {"gemma-31B", "mlp up w8",      21504, 5376, 8, 23, true},
    {"gemma-31B", "mlp down w4",     5376, 21504, 4, 58, true},
    {"gemma-31B", "attn q w4",       8192, 5376, 4, 41, true},
    {"gemma-31B", "attn o w8",       5376, 8192, 8, 60, false},
    {"gemma-31B", "attn k w4",       4096, 5376, 4, 32, false},
    {"gemma-31B", "attn v w8",       4096, 5376, 8, 50, false},
    // ---- Qwen3.6-35B-A3B (MoE; dense part only) --------------------
    {"qwen-35B", "shared gate|up w8", 1024, 2048, 8, 40, true},
    {"qwen-35B", "shared down w8",    2048,  512, 8, 40, true},
    {"qwen-35B", "attn o w8",         2048, 4096, 8, 10, false},
    // ---- gemma-4-26B-A4B (MoE; dense part only) --------------------
    {"gemma-26B", "dense gate|up w8", 4224, 2816, 8, 30, true},
    {"gemma-26B", "dense down w8",    2816, 2112, 8, 30, true},
    {"gemma-26B", "attn q w8",        4096, 2816, 8, 23, false},
    {"gemma-26B", "attn o w8",        2816, 4096, 8, 30, false},
};

}  // namespace

// Every prefill GEMM the four checkpoints dispatch, at its real shape and
// OptiQ width, through every route that can serve it. Prints the shipped
// router's pick beside the measured best so a misroute is visible as a
// number rather than inferred from the source.
TEST(optiq_blocks, dense_route)
{
  Session sess;
  auto* mc = mc_(sess);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.steel[0].valid()) {
    std::printf("[optiq_blocks] steel qmm unavailable -- skip\n");
    return;
  }
  if (!kn.have_mma) {
    std::printf("[optiq_blocks] no matrix cores -- steel only, the M5 "
                "question cannot be answered on this GPU\n");
  }
  const bool full = full_sweep_();
  const std::vector<int> ms = full
      ? std::vector<int>{64, 128, 256, 512, 1024, 2048, 4096}
      : std::vector<int>{256, 1024};
  const Route routes[] = {Route::Steel32, Route::Steel64, Route::Steel128,
                          Route::DqDense128, Route::DqDense256,
                          Route::FusedMma};

  // Per-model rollup: seconds of GEMM per prefill row, summed over layers,
  // under the shipped route vs. the best route measured for each block. Only
  // the blocks actually swept contribute, so the ratio is over those.
  //
  // Every number in this test is SINGLE-SHOT, which on a box whose GPU clock
  // follows the SoC power budget is worth about +-10%. That is fine for the
  // 2-3x steel-vs-matrix-core gaps this test exists to find, and NOT fine for
  // deciding a close call like n128 vs n128x256 -- the rollup ratio will read
  // 1.00x on differences mlp_tile_probe resolves cleanly with medians.
  struct Roll { double shipped = 0.0, best = 0.0; };
  std::vector<std::pair<std::string, Roll>> roll;
  auto roll_at = [&](const char* model) -> Roll& {
    for (auto& e : roll) {
      if (e.first == model) { return e.second; }
    }
    roll.emplace_back(model, Roll{});
    return roll.back().second;
  };

  for (const auto& bk : kBlocks) {
    if (!full && !bk.hot) { continue; }
    QW q = make_qw_(mc, bk.N, bk.K, bk.bits, 7u + (unsigned)bk.N);
    SharedBuffer wdq =
        kn.have_mma ? mc->make_shared_buffer((std::size_t)bk.N * bk.K * 2)
                    : SharedBuffer{};
    for (int M : ms) {
      SharedBuffer x = make_act_(mc, (std::size_t)M * bk.K, 3u + (unsigned)M);
      SharedBuffer y = mc->make_shared_buffer((std::size_t)M * bk.N * 2);
      const Route want = shipped_route_(M, bk.K, kn.have_mma);
      double g_shipped = 0.0, g_best = 0.0;
      const char* best_tag = "-";
      std::printf("[optiq] %-10s %-16s M=%4d N=%5d K=%5d |", bk.model, bk.tag,
                  M, bk.N, bk.K);
      for (Route r : routes) {
        if (!kn.available(r, bk.bits)) { continue; }
        const double g =
            bench_(mc, kn, r, q, x, y, wdq, M, bk.N, bk.K, bk.bits);
        std::printf(" %s %5.0f%s |", route_tag_(r), g, r == want ? "*" : "");
        if (r == want) { g_shipped = g; }
        if (g > g_best) { g_best = g; best_tag = route_tag_(r); }
      }
      std::printf(" best=%s", best_tag);
      if (g_shipped > 0.0 && g_best > 0.0) {
        std::printf(" (%.2fx over shipped)", g_best / g_shipped);
      }
      std::printf("\n");
      EXPECT_TRUE(g_best > 0.0);
      // Roll the largest swept M into the model total -- that is the regime
      // a real prefill runs in, and the one the router is tuned for.
      if (M == ms.back() && g_shipped > 0.0) {
        Roll& rr = roll_at(bk.model);
        const double f = 2.0 * bk.N * bk.K * bk.layers / 1e9;
        rr.shipped += f / g_shipped;
        rr.best += f / g_best;
      }
    }
  }
  for (const auto& e : roll) {
    const double gain = e.second.shipped / std::max(e.second.best, 1e-12);
    std::printf("[optiq ROLLUP] %-10s swept blocks: shipped %.2f ms/row, "
                "best-per-block %.2f ms/row (%.2fx)\n",
                e.first.c_str(), e.second.shipped * 1e3,
                e.second.best * 1e3, gain);
  }
}

// Does the n128 / n128x256 choice change the ANSWER, or only the rate? Both
// pass the full K as a dynamic contraction extent and differ only in how many
// output columns a threadgroup owns, so the K reduction each accumulator walks
// should be the same one -- but "should" is not a verification bar, and the
// router is about to start choosing between them on N. If they are bit-
// identical the tile is a pure perf knob; if they are not, every tile change
// owes a token-exactness run.
//
// Checked at the shapes the N-keyed rule actually moves (large N, K < 6144),
// plus a ragged N and a ragged M so the tail clamp is included.
TEST(optiq_blocks, tile_choice_is_bit_identical)
{
  Session sess;
  auto* mc = mc_(sess);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.have_mma) {
    std::printf("[optiq_tile] no matrix cores -- skip\n");
    return;
  }
  const Shape3 cases[] = {
      {512, 34816, 5120, "qwen-27B gate|up"},
      {512, 43008, 5376, "gemma-31B gate|up"},
      {512, 21504, 5376, "gemma-31B up w8"},
      {300, 17400, 5120, "ragged M and N"},
  };
  for (const auto& c : cases) {
    QW q = make_qw_(mc, c.N, c.K, 4, 23u + (unsigned)c.N);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)c.N * c.K * 2);
    SharedBuffer x = make_act_(mc, (std::size_t)c.M * c.K, 29u);
    SharedBuffer y128 = mc->make_shared_buffer((std::size_t)c.M * c.N * 2);
    SharedBuffer y256 = mc->make_shared_buffer((std::size_t)c.M * c.N * 2);
    auto run = [&](Route r, const SharedBuffer& y) {
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        encode_(e, kn, r, q, x, y, wdq, c.M, c.N, c.K, 4);
      }
      st.commit().wait();
    };
    std::memset(y128.contents(), 0, y128.byte_size());
    std::memset(y256.contents(), 0, y256.byte_size());
    run(Route::DqDense128, y128);
    run(Route::DqDense256, y256);
    const std::size_t n = (std::size_t)c.M * c.N;
    const auto* a = static_cast<const std::uint16_t*>(y128.contents());
    const auto* b = static_cast<const std::uint16_t*>(y256.contents());
    std::size_t diff = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (a[i] != b[i]) { ++diff; }
    }
    std::printf("[optiq_tile] %-18s M=%4d N=%5d K=%5d  differing f16 "
                "outputs: %zu / %zu\n", c.tag, c.M, c.N, c.K, diff, n);
    EXPECT_TRUE(diff == 0);
  }
}

// Two things dense_route leaves open, both about the DENSE models' MLP.
//
// (1) down_proj (K = 17408 / 21504) tops out around 4.4-5.4 TFLOP/s while
//     gate|up on the same weights reaches 10-11. metal-krea2-transformer
//     hit the identical wall on its ff-down ("the single-op full reduction
//     runs ~0.7x the K<=9728 rate") and fixed it with a split-K kernel; the
//     LM path never got one. Probed here two ways: a transpose control at
//     identical FLOPs and weight bytes, which separates "deep K" from "narrow
//     N", and the split-K kernel itself.
//
//     Split-K needs K == n_splits * KC with KC a COMPILE-TIME constant, and
//     the instantiated KCs (8192 / 6784 / 3392) divide neither 17408 nor
//     21504. So the split-K arm runs at the nearest KC-compatible K (16960
//     and 20352, 2.6% and 5.4% short) -- close enough to say whether a new
//     KC instantiation is worth cutting, which is the decision this informs.
//
// (2) dense_route's n128-vs-n256 winner flips run to run at the big gate|up
//     shapes. The A/B here alternates the two within one command stream and
//     takes medians over several rounds, because a single-shot comparison on
//     this box measures the SoC power state as much as the kernel.
TEST(optiq_blocks, mlp_tile_probe)
{
  // Opt-in: this is the arbiter for a decision already made, not part of the
  // audit, and its medians cost ~35 s. VPIPE_OPTIQ_BLOCKS (any value) runs it.
  if (!opted_in_()) { return; }
  Session sess;
  auto* mc = mc_(sess);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.have_mma) {
    std::printf("[optiq_probe] no matrix cores -- skip\n");
    return;
  }
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise");
  ComputeFunction f_res = lib_elt.function("residual_add_f16");
  ComputeFunction f_sk3392 =
      kn.lib_dense.function("dense_gemm_mma_splitk_n128x256_k3392_f16");

  // Median-of-rounds A/B, TRULY alternating: both arms are warmed before
  // either is timed, and then each round times A and B back to back.
  //
  // Running all of A's rounds and then all of B's -- the obvious structure --
  // is biased, and the bias is big enough to invert a conclusion. The first
  // arm measured eats the first-touch cost of the freshly allocated x / y /
  // weight buffers, and the two arms sit in different SoC power states. An
  // earlier cut of this probe did exactly that and reported the 128x256 tile
  // winning 1.07-1.22x at the wide gate|up shapes; alternating says 128x128
  // wins there, and an end-to-end Qwen3.5-4B prefill A/B agrees with the
  // alternating answer (1160 vs 1074 tok/s). Do not un-interleave this.
  auto ab_median = [&](int rounds, double fa, double fb, auto&& arm_a,
                       auto&& arm_b, double* ga, double* gb) {
    const int ia = std::min(20, std::max(2, (int)(3.0e11 / fa)));
    const int ib = std::min(20, std::max(2, (int)(3.0e11 / fb)));
    auto once = [&](auto&& arm, int iters, double flops) {
      const auto t0 = std::chrono::steady_clock::now();
      CommandStream st = mc->make_command_stream();
      {
        ComputeEncoder e = st.begin_compute();
        for (int i = 0; i < iters; ++i) { arm(e); }
      }
      st.commit().wait();
      const auto t1 = std::chrono::steady_clock::now();
      return flops * iters / 1e9 / secs_(t0, t1);
    };
    once(arm_a, 1, fa);          // warm BOTH before timing EITHER
    once(arm_b, 1, fb);
    std::vector<double> va, vb;
    for (int r = 0; r < rounds; ++r) {
      va.push_back(once(arm_a, ia, fa));
      vb.push_back(once(arm_b, ib, fb));
    }
    std::sort(va.begin(), va.end());
    std::sort(vb.begin(), vb.end());
    *ga = va[va.size() / 2];
    *gb = vb[vb.size() / 2];
  };

  struct Probe {
    const char* tag;
    int N, K, ksplit;   // ksplit: KC-compatible stand-in for K (0 = none)
  };
  const Probe probes[] = {
      {"qwen-27B  down",   5120, 17408, 16960},
      {"qwen-27B  down^T", 17408, 5120, 0},
      {"gemma-31B down",   5376, 21504, 20352},
      {"gemma-31B down^T", 21504, 5376, 0},
      // The wide-N gate|up ladder. K matters as much as N here and the first
      // cut of the N-keyed rule missed it: Qwen3.5-4B (N=18432 K=2560) and
      // gemma-4-12B (N=30720 K=3840) are wide but SHALLOW, and routing them
      // to the 128x256 tile cost 7% of 4B prefill end to end. So the ladder
      // walks K at large N -- 2560, 3840, 5120, 5376 -- and the rule has to
      // separate the shallow pair from the deep pair, not just call them all
      // wide.
      {"qwen-4B   gate|up", 18432, 2560, 0},
      {"gemma-12B gate|up", 30720, 3840, 0},
      {"qwen-27B  gate|up", 34816, 5120, 0},
      {"gemma-31B gate|up", 43008, 5376, 0},
      // Same N as the 27B's gate|up at the 4B's K, and vice versa, to say
      // which of the two axes the flip actually follows.
      {"wideN shallowK",   34816, 2560, 0},
      {"narrowN deepK",    18432, 5376, 0},
  };
  const int ms[] = {1024, 2048, 4096};
  for (const auto& p : probes) {
    const int Kmax = std::max(p.K, p.ksplit);
    QW q = make_qw_(mc, p.N, Kmax, 4, 17u + (unsigned)p.N);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)p.N * Kmax * 2);
    for (int M : ms) {
      SharedBuffer x = make_act_(mc, (std::size_t)M * Kmax, 9u);
      SharedBuffer y = mc->make_shared_buffer((std::size_t)M * p.N * 2);
      const double f1 = 2.0 * M * p.N * p.K;
      // Weight expansion is common to all three arms; keep it out of the
      // comparison so this measures the GEMM, not the dequant.
      auto dq = [&](ComputeEncoder& e) {
        e.set_function(kn.dq[0]);
        e.set_buffer(0, q.w);
        e.set_buffer(1, q.s);
        e.set_buffer(2, q.b);
        e.set_buffer(3, wdq);
        e.set_constant(4, p.K);
        e.set_constant(5, p.N);
        e.dispatch({(unsigned)(p.K / 8), (unsigned)p.N, 1}, {64, 1, 1});
      };
      {
        CommandStream st = mc->make_command_stream();
        { ComputeEncoder e = st.begin_compute(); dq(e); }
        st.commit().wait();
      }
      auto dense = [&](int BN, int K) {
        return [&, BN, K](ComputeEncoder& e) {
          e.set_function(BN == 256 ? kn.dense256 : kn.dense128);
          e.set_buffer(0, x);
          e.set_buffer(1, wdq);
          e.set_buffer(2, wdq);
          e.set_buffer(3, y);
          e.set_constant(4, K);
          e.set_constant(5, p.N);
          e.set_constant(6, M);
          e.set_constant(7, 0);
          e.dispatch({(unsigned)(((p.N + BN - 1) / BN) * 256),
                      (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        };
      };
      double g128 = 0.0, g256 = 0.0;
      ab_median(5, f1, f1, dense(128, p.K), dense(256, p.K), &g128, &g256);
      std::printf("[optiq_probe] %-18s M=%4d N=%5d K=%5d | n128 %5.0f | "
                  "n256 %5.0f | n256/n128 %.2fx", p.tag, M, p.N, p.K, g128,
                  g256, g256 / g128);
      // Split-K arm: S planes + an S-1 fold, at the nearest KC-multiple K.
      if (p.ksplit > 0 && f_sk3392.valid() && f_res.valid()) {
        const int splits = p.ksplit / 3392;
        const std::size_t plane = (std::size_t)M * p.N;
        SharedBuffer sk =
            mc->make_shared_buffer(plane * (std::size_t)splits * 2);
        const double fk = 2.0 * M * p.N * p.ksplit;
        // Paired against the best single-op tile, alternating, so the split-K
        // ratio is not a comparison across two power states.
        double gk = 0.0, gref = 0.0;
        ab_median(5, fk, f1, [&](ComputeEncoder& e) {
          e.set_function(f_sk3392);
          e.set_buffer(0, x);
          e.set_buffer(1, wdq);
          e.set_buffer(2, sk);
          e.set_constant(3, p.ksplit);
          e.set_constant(4, p.N);
          e.set_constant(5, M);
          e.dispatch({(unsigned)(((p.N + 255) / 256) * 256),
                      (unsigned)((M + 127) / 128), (unsigned)splits},
                     {256, 1, 1});
          for (int s = 1; s < splits; ++s) {
            e.set_function(f_res);
            e.set_buffer(0, s == 1 ? sk : y);
            e.set_buffer(1, sk, plane * (std::size_t)s * 2);
            e.set_buffer(2, y);
            e.set_constant(3, (int)plane);
            e.dispatch({(unsigned)plane, 1, 1}, {256, 1, 1});
          }
        }, dense(g256 >= g128 ? 256 : 128, p.K), &gk, &gref);
        std::printf(" | splitk(K=%d,S=%d) %5.0f vs %5.0f (%.2fx)",
                    p.ksplit, splits, gk, gref, gk / gref);
      }
      std::printf("\n");
      EXPECT_TRUE(g128 > 0.0 && g256 > 0.0);
    }
  }
}

// Split-K at the TRUE down_proj depths, both halves of the decision:
//
//   rate   -- against the best single-op tile, arms interleaved.
//   answer -- how far the split output drifts from the single-op output.
//             Split-K writes S partial planes and folds them, so each output
//             takes S f16 roundings where the single-op path takes one. That
//             is fine for the rel-L2-verified DiT this kernel was written for
//             and is NOT obviously fine for an LM held to greedy token-exact,
//             so it gets measured rather than assumed. Reported as max abs
//             deviation and as the fraction of outputs that differ at all.
//
// This is the shape the wiring below keys on, so it runs by default.
TEST(optiq_blocks, splitk_down_proj)
{
  Session sess;
  auto* mc = mc_(sess);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  if (!kn.have_mma) {
    std::printf("[optiq_splitk] no matrix cores -- skip\n");
    return;
  }
  ComputeLibrary lib_elt = mc->load_library("llm_elementwise");
  ComputeFunction f_fold = lib_elt.function("splitk_fold_f32_f16");
  struct Case {
    const char* tag;
    int N, K, kc;
  };
  const Case cases[] = {
      {"qwen-27B down",  5120, 17408, 8704},    // S=2
      {"qwen-27B down",  5120, 17408, 4352},    // S=4
      {"gemma-31B down", 5376, 21504, 10752},   // S=2
      {"gemma-31B down", 5376, 21504, 5376},    // S=4
  };
  for (const auto& c : cases) {
    ComputeFunction f_sk = kn.lib_dense.function(
        std::string("dense_gemm_mma_splitk32_n128x256_k")
        + std::to_string(c.kc) + "_f16");
    if (!f_sk.valid() || !f_fold.valid()) {
      std::printf("[optiq_splitk] %s: kernel unavailable -- skip\n", c.tag);
      continue;
    }
    const int splits = c.K / c.kc;
    QW q = make_qw_(mc, c.N, c.K, 4, 31u + (unsigned)c.N);
    SharedBuffer wdq = mc->make_shared_buffer((std::size_t)c.N * c.K * 2);
    for (int M : {1024, 4096}) {
      SharedBuffer x = make_act_(mc, (std::size_t)M * c.K, 37u);
      SharedBuffer y1 = mc->make_shared_buffer((std::size_t)M * c.N * 2);
      SharedBuffer y2 = mc->make_shared_buffer((std::size_t)M * c.N * 2);
      const std::size_t plane = (std::size_t)M * c.N;
      SharedBuffer sk =
          mc->make_shared_buffer(plane * (std::size_t)splits * 4);   // f32
      {
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          e.set_function(kn.dq[0]);
          e.set_buffer(0, q.w);
          e.set_buffer(1, q.s);
          e.set_buffer(2, q.b);
          e.set_buffer(3, wdq);
          e.set_constant(4, c.K);
          e.set_constant(5, c.N);
          e.dispatch({(unsigned)(c.K / 8), (unsigned)c.N, 1}, {64, 1, 1});
        }
        st.commit().wait();
      }
      auto single = [&](ComputeEncoder& e) {
        e.set_function(kn.dense256);
        e.set_buffer(0, x);
        e.set_buffer(1, wdq);
        e.set_buffer(2, wdq);
        e.set_buffer(3, y1);
        e.set_constant(4, c.K);
        e.set_constant(5, c.N);
        e.set_constant(6, M);
        e.set_constant(7, 0);
        e.dispatch({(unsigned)(((c.N + 255) / 256) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      };
      auto split = [&](ComputeEncoder& e) {
        e.set_function(f_sk);
        e.set_buffer(0, x);
        e.set_buffer(1, wdq);
        e.set_buffer(2, sk);
        e.set_constant(3, c.K);
        e.set_constant(4, c.N);
        e.set_constant(5, M);
        e.dispatch({(unsigned)(((c.N + 255) / 256) * 256),
                    (unsigned)((M + 127) / 128), (unsigned)splits},
                   {256, 1, 1});
        e.set_function(f_fold);
        e.set_buffer(0, sk);
        e.set_buffer(1, y2);
        e.set_constant(2, (int)plane);
        e.set_constant(3, splits);
        e.dispatch({(unsigned)plane, 1, 1}, {256, 1, 1});
      };
      const double f1 = 2.0 * M * c.N * c.K;
      const int iters = std::min(20, std::max(2, (int)(3.0e11 / f1)));
      auto once = [&](auto&& arm, int n) {
        const auto t0 = std::chrono::steady_clock::now();
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          for (int i = 0; i < n; ++i) { arm(e); }
        }
        st.commit().wait();
        const auto t1 = std::chrono::steady_clock::now();
        return f1 * n / 1e9 / secs_(t0, t1);
      };
      once(single, 1);
      once(split, 1);
      std::vector<double> vs, vk;
      for (int r = 0; r < 5; ++r) {
        vs.push_back(once(single, iters));
        vk.push_back(once(split, iters));
      }
      std::sort(vs.begin(), vs.end());
      std::sort(vk.begin(), vk.end());
      const double g1 = vs[vs.size() / 2], gk = vk[vk.size() / 2];
      // Deviation of the folded split output from the single-op output.
      const auto* a = static_cast<const _Float16*>(y1.contents());
      const auto* b = static_cast<const _Float16*>(y2.contents());
      double maxabs = 0.0, maxrel = 0.0;
      std::size_t ndiff = 0;
      for (std::size_t i = 0; i < plane; ++i) {
        const double va = (double)a[i], vb = (double)b[i];
        if (va != vb) { ++ndiff; }
        const double d = std::fabs(va - vb);
        if (d > maxabs) { maxabs = d; }
        const double den = std::fabs(va);
        if (den > 1e-3 && d / den > maxrel) { maxrel = d / den; }
      }
      std::printf("[optiq_splitk] %-15s M=%4d N=%5d K=%5d S=%d | single "
                  "%5.0f | split %5.0f (%.2fx) | differ %.1f%% max|d| %.4f "
                  "max rel %.2e\n", c.tag, M, c.N, c.K, splits, g1, gk,
                  gk / g1, 100.0 * (double)ndiff / (double)plane, maxabs,
                  maxrel);
      EXPECT_TRUE(gk > 0.0 && g1 > 0.0);
    }
  }
}

// The routed-expert GEMM, which is where both MoE checkpoints spend their
// prefill and the ONLY block in any of the four with no matrix-core route at
// all: metal-{qwen,gemma}-model.cc dispatch affine_qmm_grouped_* (steel,
// BM=32) for it and there is no grouped matmul2d kernel to dispatch instead.
//
// Three measurements per shape, all on the same total work (E experts x
// `rows_per_expert` rows each):
//   grouped   -- the shipped kernel: ONE dispatch, tile->expert via t2e, the
//                row gather fused into the activation loader.
//   steel/e   -- the same steel GEMM run per expert, E dispatches. Isolates
//                what the grouped tiling + fused gather are worth.
//   mma/e     -- affine_qmm_mma per expert, E dispatches. Same launch count
//                as steel/e, so mma/e over steel/e is the pure kernel ratio,
//                and grouped x that ratio estimates a grouped-mma tier.
// The gate|up GEMM is benched as a plain N=2I GEMM: the grouped kernels fold
// SwiGLU/GeGLU into the store, which is a store-time detail, not the GEMM.
TEST(optiq_blocks, moe_expert_route)
{
  Session sess;
  auto* mc = mc_(sess);
  if (mc == nullptr) { return; }
  Kernels kn;
  kn.load(mc);
  ComputeLibrary lib_steel = mc->load_library("affine_qmm_steel");
  // The two grouped entry points the MoE models dispatch. Qwen's routed
  // experts are SwiGLU, gemma's GeGLU; `affine_qmm_grouped_w4g64` (no
  // activation) serves down on both.
  ComputeFunction g_swiglu =
      lib_steel.function("affine_qmm_grouped_swiglu_w4g64");
  ComputeFunction g_geglu =
      lib_steel.function("affine_qmm_grouped_geglu_w4g64");
  ComputeFunction g_down = lib_steel.function("affine_qmm_grouped_w4g64");
  if (!g_down.valid() || !kn.steel[0].valid()) {
    std::printf("[optiq_moe] grouped kernels unavailable -- skip\n");
    return;
  }

  struct MoeBlock {
    const char* model;
    const char* tag;
    int E, N, K;
    bool gated;      // gate|up (N = 2*I, store halves it) vs plain down
    bool geglu;      // gemma stores GeGLU, qwen SwiGLU
  };
  // Qwen 35B-A3B: 256 experts, moe_inner 512, hidden 2048.
  // gemma 26B-A4B: 128 experts, moe_inner 704, hidden 2816.
  const MoeBlock blocks[] = {
      {"qwen-35B",  "expert gate|up", 256, 1024, 2048, true,  false},
      {"qwen-35B",  "expert down",    256, 2048,  512, false, false},
      {"gemma-26B", "expert gate|up", 128, 1408, 2816, true,  true},
      {"gemma-26B", "expert down",    128, 2816,  704, false, true},
  };
  // Rows per expert = prompt_rows * top_k / E; top_k is 8 on both. So 32
  // rows/expert is a ~1k prompt on the 256-expert Qwen and ~512 on the
  // 128-expert gemma, and 256 rows/expert is 8k / 4k respectively.
  const std::vector<int> mes =
      full_sweep_() ? std::vector<int>{32, 64, 128, 256, 512}
                    : std::vector<int>{32, 128, 256};

  for (const auto& bk : blocks) {
    // One [E, N, K] expert slab, exactly as the model binds it.
    QW q = make_qw_(mc, bk.E * bk.N, bk.K, 4, 11u + (unsigned)bk.N);
    const std::size_t wstride = (std::size_t)bk.N * bk.K / 2;      // bytes
    const std::size_t gstride = (std::size_t)bk.N * (bk.K / 64);   // elems
    for (int me : mes) {
      const int npad = bk.E * me;
      SharedBuffer x = make_act_(mc, (std::size_t)npad * bk.K, 5u);
      // Sized for the per-expert arms, which write all N columns; the
      // grouped gate|up arm folds the activation and writes only N/2.
      SharedBuffer y = mc->make_shared_buffer((std::size_t)npad * bk.N * 2);
      // t2e: one expert per BM=32 row-tile. srow: identity gather (the
      // gather's COST is what matters here, not which row it lands on).
      SharedBuffer t2e = mc->make_shared_buffer((std::size_t)(npad / 32) * 4);
      SharedBuffer srow = mc->make_shared_buffer((std::size_t)npad * 4);
      auto* tp = static_cast<std::int32_t*>(t2e.contents());
      for (int t = 0; t < npad / 32; ++t) { tp[t] = t / (me / 32); }
      auto* rp = static_cast<std::int32_t*>(srow.contents());
      for (int i = 0; i < npad; ++i) { rp[i] = i; }

      const double f1 = 2.0 * npad * bk.N * bk.K;
      const int iters = std::min(30, std::max(3, (int)(4.0e11 / f1)));

      // Every arm is warmed before ANY arm is timed, and the timed rounds
      // interleave, for the reason spelled out on mlp_tile_probe's ab_median:
      // measuring arm A to completion and then arm B charges A for first-touch
      // and compares two different SoC power states. An earlier cut of this
      // test did that and its dq+dense numbers were inflated.
      auto once = [&](auto&& enc_one, int n) {
        const auto t0 = std::chrono::steady_clock::now();
        CommandStream st = mc->make_command_stream();
        {
          ComputeEncoder e = st.begin_compute();
          for (int i = 0; i < n; ++i) { enc_one(e); }
        }
        st.commit().wait();
        const auto t1 = std::chrono::steady_clock::now();
        return f1 * n / 1e9 / secs_(t0, t1);
      };
      auto medians = [&](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
      };

      ComputeFunction& gfn =
          bk.gated ? (bk.geglu ? g_geglu : g_swiglu) : g_down;
      auto arm_grouped = [&](ComputeEncoder& e) {
        e.set_function(gfn);
        e.set_buffer(0, q.w);
        e.set_buffer(1, q.s);
        e.set_buffer(2, q.b);
        e.set_buffer(3, x);
        e.set_buffer(4, y);
        e.set_constant(5, bk.K);
        e.set_constant(6, bk.N);
        e.set_constant(7, npad);
        e.set_buffer(8, t2e);
        e.set_buffer(9, srow);
        e.dispatch({(unsigned)(((bk.N + 31) / 32) * 32),
                    (unsigned)(((npad + 31) / 32) * 2), 2}, {32, 2, 2});
      };

      // Per-expert: same total work, E dispatches, each expert's slab bound
      // at its own offset. Output goes to y at that expert's row segment.
      // `dq` expands that expert's weight into a 1-expert scratch and runs
      // the dense matmul2d -- the route every DENSE block of these models
      // takes, applied per expert. Per-expert (not whole-slab) because the
      // slab is 1.1 GB on the 35B and would have to be re-expanded every
      // layer either way; the traffic is identical and the scratch is not.
      SharedBuffer wdq =
          kn.have_mma ? mc->make_shared_buffer((std::size_t)bk.N * bk.K * 2)
                      : SharedBuffer{};
      enum class Arm { Steel, Fused, DqDense };
      auto per_expert = [&](Arm arm) {
        return [&, arm](ComputeEncoder& e) {
          for (int ex = 0; ex < bk.E; ++ex) {
            const std::size_t woff = (std::size_t)ex * wstride;
            const std::size_t goff = (std::size_t)ex * gstride * 2;
            if (arm == Arm::DqDense) {
              e.set_function(kn.dq[0]);
              e.set_buffer(0, q.w, woff);
              e.set_buffer(1, q.s, goff);
              e.set_buffer(2, q.b, goff);
              e.set_buffer(3, wdq);
              e.set_constant(4, bk.K);
              e.set_constant(5, bk.N);
              e.dispatch({(unsigned)(bk.K / 8), (unsigned)bk.N, 1},
                         {64, 1, 1});
              e.set_function(kn.dense128);
              e.set_buffer(0, x, (std::size_t)ex * me * bk.K * 2);
              e.set_buffer(1, wdq);
              e.set_buffer(2, wdq);
              e.set_buffer(3, y, (std::size_t)ex * me * bk.N * 2);
              e.set_constant(4, bk.K);
              e.set_constant(5, bk.N);
              e.set_constant(6, me);
              e.set_constant(7, 0);
              e.dispatch({(unsigned)(((bk.N + 127) / 128) * 256),
                          (unsigned)((me + 127) / 128), 1}, {256, 1, 1});
              continue;
            }
            e.set_function(arm == Arm::Fused ? kn.fused : kn.steel[0]);
            e.set_buffer(0, q.w, woff);
            e.set_buffer(1, q.s, goff);
            e.set_buffer(2, q.b, goff);
            e.set_buffer(3, x, (std::size_t)ex * me * bk.K * 2);
            e.set_buffer(4, y, (std::size_t)ex * me * bk.N * 2);
            e.set_constant(5, bk.K);
            e.set_constant(6, bk.N);
            e.set_constant(7, me);
            if (arm == Arm::Fused) {
              e.dispatch({(unsigned)(((bk.N + 63) / 64) * 128),
                          (unsigned)((me + 63) / 64), 1}, {128, 1, 1});
            } else {
              e.dispatch({(unsigned)(((bk.N + 31) / 32) * 32),
                          (unsigned)(((me + 31) / 32) * 2), 2}, {32, 2, 2});
            }
          }
        };
      };
      const bool mma_ok = kn.have_mma && kn.fused.valid();
      auto a_steel = per_expert(Arm::Steel);
      auto a_fused = per_expert(Arm::Fused);
      auto a_dq = per_expert(Arm::DqDense);
      // Warm every arm, then interleave the timed rounds.
      if (gfn.valid()) { once(arm_grouped, 1); }
      once(a_steel, 1);
      if (mma_ok) { once(a_fused, 1); }
      if (kn.have_mma) { once(a_dq, 1); }
      std::vector<double> vg, vs, vf, vd;
      for (int r = 0; r < 3; ++r) {
        if (gfn.valid()) { vg.push_back(once(arm_grouped, iters)); }
        vs.push_back(once(a_steel, iters));
        if (mma_ok) { vf.push_back(once(a_fused, iters)); }
        if (kn.have_mma) { vd.push_back(once(a_dq, iters)); }
      }
      const double g_grouped = medians(vg);
      const double g_steel_e = medians(vs);
      const double g_mma_e = medians(vf);
      const double g_dq_e = medians(vd);

      std::printf("[optiq_moe] %-10s %-15s E=%3d rows/e=%3d N=%4d K=%4d | "
                  "grouped %5.0f | steel/e %5.0f | fused-mma/e %5.0f | "
                  "dq+dense/e %5.0f",
                  bk.model, bk.tag, bk.E, me, bk.N, bk.K, g_grouped,
                  g_steel_e, g_mma_e, g_dq_e);
      // Best matrix-core option per expert, scaled by what the grouped
      // tiling is worth on steel -- the rate a grouped matmul2d tier would
      // have to beat `grouped` with.
      const double best_mma = std::max(g_mma_e, g_dq_e);
      if (best_mma > 0.0 && g_steel_e > 0.0) {
        std::printf(" | best-mma/steel %.2fx -> grouped-mma ~%.0f",
                    best_mma / g_steel_e, g_grouped * best_mma / g_steel_e);
      }
      std::printf("\n");
      EXPECT_TRUE(g_steel_e > 0.0);
    }
  }
}
