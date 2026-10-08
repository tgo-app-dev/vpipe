#include "generative-models/flashvsr/metal-flashvsr-transformer.h"

#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/mma-splitk.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <algorithm>
#include <random>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <deque>
#include <utility>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

constexpr const char* kKey = "flashvsr-dit/bf16/";

std::uint16_t
to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = (u >> 16) & 1u;
  u += 0x7fffu + r;
  return (std::uint16_t)(u >> 16);
}

float
from_bf16(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Read any supported dtype as f32. The denoiser ships F32, but the
// projection and context beside it are BF16, and one reader for the
// directory is one place to be wrong.
std::vector<float>
read_f32(WeightSet& ws, MetalCompute* mc, const std::string& nm)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr) { return {}; }
  SharedBuffer b = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (b.empty()) { return {}; }
  std::size_t n = 1;
  for (const auto d : info->shape) { n *= (std::size_t)d; }
  std::vector<float> out(n, 0.0f);
  const void* p = b.contents();
  if (info->dtype == "F32") {
    std::memcpy(out.data(), p, n * 4);
  } else if (info->dtype == "BF16") {
    const auto* s = (const std::uint16_t*)p;
    for (std::size_t i = 0; i < n; ++i) { out[i] = from_bf16(s[i]); }
  } else {
    return {};
  }
  return out;
}

SharedBuffer
bf16_buf(MetalCompute* mc, const float* src, std::size_t n)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  if (b.empty()) { return b; }
  auto* d = (std::uint16_t*)b.contents();
  for (std::size_t i = 0; i < n; ++i) { d[i] = to_bf16(src[i]); }
  return b;
}

std::string
blk(int i, const char* rest)
{
  return "blocks." + std::to_string(i) + "." + rest;
}

inline float
silu(float x)
{
  return x / (1.0f + std::exp(-x));
}

inline float
gelu_tanh(float x)
{
  const float c = 0.7978845608028654f;   // sqrt(2/pi)
  const float t = c * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(t));
}

// y[M,N] = x[M,K] * w[N,K]^T + b[N], on the HOST.
//
// Used only for the load-time constants, which are small and run once;
// everything per-forward goes through the GPU. Keeping them here means
// the folded timestep and the cross-attention cache are computed in f32
// rather than bf16, which is the reference's own working precision for
// these -- it builds its timestep sinusoid in float64.
void
host_linear(const std::vector<float>& x, const std::vector<float>& w,
            const std::vector<float>& b, int M, int N, int K,
            std::vector<float>* y)
{
  y->assign((std::size_t)M * N, 0.0f);
  // THREADED, because the cross-attention fold is 72 GFLOP over the 30
  // blocks and a scalar loop spent 42 s of every load on it. Rows are
  // independent, so this is a split and not an algorithm.
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  const int nthreads = (int)std::min<unsigned>(hw, (unsigned)std::max(M, 1));
  auto run = [&](int m0, int m1) {
    for (int m = m0; m < m1; ++m) {
      const float* xr = x.data() + (std::size_t)m * K;
      for (int n = 0; n < N; ++n) {
        double acc = b.empty() ? 0.0 : (double)b[(std::size_t)n];
        const float* wr = w.data() + (std::size_t)n * K;
        for (int k = 0; k < K; ++k) { acc += (double)xr[k] * (double)wr[k]; }
        (*y)[(std::size_t)m * N + n] = (float)acc;
      }
    }
  };
  if (nthreads <= 1) { run(0, M); return; }
  std::vector<std::thread> pool;
  pool.reserve((std::size_t)nthreads);
  const int span = (M + nthreads - 1) / nthreads;
  for (int t = 0; t < nthreads; ++t) {
    const int m0 = t * span;
    const int m1 = std::min(M, m0 + span);
    if (m0 >= m1) { break; }
    pool.emplace_back(run, m0, m1);
  }
  for (auto& th : pool) { th.join(); }
}

}  // namespace

// One transformer block. Everything the checkpoint stores plus the two
// things this model lets us FOLD: the six modulation vectors (constant,
// because the timestep is) and the cross-attention key and value
// (constant, because the conditioning is).
struct MetalFlashVsrTransformer::Block {
  SharedBuffer q1, k1, v1, o1;          // self-attention projections
  SharedBuffer q1b, k1b, v1b, o1b;
  SharedBuffer qn1, kn1;                // q/k RMS gammas, across heads
  SharedBuffer q2, o2, q2b, o2b;        // cross-attention q and out only
  SharedBuffer qn2;
  SharedBuffer ck, cv;                  // the folded cross KV [tokens, dim]
  SharedBuffer n3w, n3b;                // norm3 is the one norm with affine
  SharedBuffer ff_in, ff_in_b, ff_out, ff_out_b;
  SharedBuffer mod;                     // [6, dim], modulation + t_mod
};

// A STREAMING load's two block destinations and the scratch an F32
// tensor is read into before it is narrowed into one.
struct MetalFlashVsrTransformer::Streaming {
  BlockSlots<Block>         slots;
  std::vector<std::uint8_t> scratch;
};

namespace {

// Every tensor of a block that is READ at run time, against the field it
// lands in -- everything but what the load folds (the cross-attention
// key and value, the modulation). The one list both the resident load
// and the slots walk, so a field cannot be loaded one way and streamed
// another.
template <class B, class Fn>
void
each_run_tensor(int i, B& b, Fn&& fn)
{
  const std::string p = "blocks." + std::to_string(i) + ".";
  fn(p + "self_attn.q.weight", b.q1);
  fn(p + "self_attn.k.weight", b.k1);
  fn(p + "self_attn.v.weight", b.v1);
  fn(p + "self_attn.o.weight", b.o1);
  fn(p + "self_attn.q.bias", b.q1b);
  fn(p + "self_attn.k.bias", b.k1b);
  fn(p + "self_attn.v.bias", b.v1b);
  fn(p + "self_attn.o.bias", b.o1b);
  fn(p + "self_attn.norm_q.weight", b.qn1);
  fn(p + "self_attn.norm_k.weight", b.kn1);
  fn(p + "cross_attn.q.weight", b.q2);
  fn(p + "cross_attn.o.weight", b.o2);
  fn(p + "cross_attn.q.bias", b.q2b);
  fn(p + "cross_attn.o.bias", b.o2b);
  fn(p + "cross_attn.norm_q.weight", b.qn2);
  fn(p + "norm3.weight", b.n3w);
  fn(p + "norm3.bias", b.n3b);
  fn(p + "ffn.0.weight", b.ff_in);
  fn(p + "ffn.0.bias", b.ff_in_b);
  fn(p + "ffn.2.weight", b.ff_out);
  fn(p + "ffn.2.bias", b.ff_out_b);
}

}  // namespace

MetalFlashVsrTransformer::~MetalFlashVsrTransformer()
{
  // A read ahead may be in flight into a slot; it must land before the
  // slots go.
  if (_streaming) { _streaming->slots.join(); }
}

// ----------------------------------------------------------- geometry

int
MetalFlashVsrTransformer::align_frames(int frames)
{
  // 8k+1: the denoise consumes eight source frames per chunk. Round UP,
  // because rounding down silently delivers less video than was asked
  // for -- the rule every align here follows.
  if (frames <= 1) { return 1; }
  const int r = (frames - 1) % 8;
  return r == 0 ? frames : frames + (8 - r);
}

int
MetalFlashVsrTransformer::latent_frames(int aligned_frames)
{
  // The OPENING chunk emits six latent frames and every later one emits
  // two, which is not the uniform rule it looks like: 2*chunks is four
  // frames short, and a caller sizing a latent by that allocates a clip
  // it cannot hold.
  const int chunks = (aligned_frames - 1) / 8 - 2;
  return chunks > 0 ? 2 * chunks + 4 : 0;
}

int
MetalFlashVsrTransformer::restored_frames(int aligned_frames)
{
  // Twenty frames go to the warmup that fills the caches and is never
  // emitted. What comes back is the VAE's own expansion of the latent
  // above, 4k+1, not eight frames per chunk.
  const int t = latent_frames(aligned_frames);
  return t > 0 ? 1 + 4 * (t - 1) : 0;
}

int
MetalFlashVsrTransformer::kv_windows(double kv_ratio)
{
  const int keep = kv_ratio > 0.0 ? (int)(kv_ratio + 0.5) : 0;
  return std::max(keep + 1, 3);
}

int
MetalFlashVsrTransformer::ring_windows(int frames, double kv_ratio)
{
  // The opening chunk writes three windows and attends them; only a
  // SECOND chunk needs the slot past them that the ratio provides for.
  const int chunks = (align_frames(frames) - 1) / 8 - 2;
  return chunks <= 1 ? 3 : kv_windows(kv_ratio);
}

std::uint64_t
MetalFlashVsrTransformer::kv_ring_bytes(const Config& cfg, int height,
                                        int width, int windows)
{
  // The token grid is the frame over 16 (VAE 8x, patch 2); one temporal
  // window is two latent frames of it.
  const std::uint64_t th = (std::uint64_t)(height / 16);
  const std::uint64_t tw = (std::uint64_t)(width / 16);
  const std::uint64_t window = 2 * th * tw;
  return window * (std::uint64_t)windows * (std::uint64_t)cfg.hidden *
         2u /* bf16 */ * 2u /* K and V */;
}

bool
MetalFlashVsrTransformer::blocks_outer(const Config& cfg, int height,
                                       int width, int frames,
                                       const Params& prm)
{
  if (const char* e = std::getenv("VPIPE_FVSR_CHUNK_OUTER")) {
    if (*e != '\0' && std::string(e) != "0") { return false; }
  }
  using U = std::uint64_t;
  const U tok = (U)(height / 16) * (U)(width / 16);
  const int T = latent_frames(align_frames(frames));
  // What each order holds that the other does not. Chunk-outer: a window
  // per block, so n_layers - 1 more of them. Block-outer: the residual
  // stream and the RoPE tables of every chunk, where chunk-outer holds
  // one chunk's -- the opening one, six latent frames, is the largest.
  const U ring = kv_ring_bytes(cfg, height, width,
                               ring_windows(frames, prm.kv_ratio));
  const U chunk_extra = (U)(cfg.n_layers - 1) * ring;
  const U per_frame = tok * ((U)cfg.hidden * 2 + (U)cfg.head_dim * 4 * 2);
  const U block_extra = (T > 6 ? (U)(T - 6) : 0) * per_frame;
  return block_extra <= chunk_extra;
}

std::uint64_t
MetalFlashVsrTransformer::kv_window_bytes(const Config& cfg, int height,
                                          int width, int frames,
                                          const Params& prm)
{
  // WHAT generate() ALLOCATES, which is the number a plan has to hold.
  // It once reported the `kv_ratio` retained windows alone while the
  // forward allocated a window more and a spare K and V per block on top
  // -- 9.6 GB reported against 25.5 GB allocated at 1920x1152.
  const std::uint64_t ring =
      kv_ring_bytes(cfg, height, width, ring_windows(frames, prm.kv_ratio));
  return blocks_outer(cfg, height, width, frames, prm)
             ? ring
             : ring * (std::uint64_t)cfg.n_layers;
}

std::uint64_t
MetalFlashVsrTransformer::streaming_floor_bytes(const Config& cfg)
{
  using U = std::uint64_t;
  const U D = (U)cfg.hidden, FF = (U)cfg.ffn, L = (U)cfg.n_layers;
  // Folded, per block: the cross key and value over the fixed context,
  // and the six modulation vectors.
  const U folds = L * (2 * (U)cfg.text_tokens * D + 6 * D) * 2;
  // One block's run-time weights, bf16: q/k/v/o, the cross q/o and the
  // feed-forward, with their biases and norms -- and two of them.
  const U block = (6 * D * D + 2 * D * FF + 13 * D + FF) * 2;
  // The largest tensor as the checkpoint stores it (F32), which is what
  // the narrowing scratch settles at.
  const U scratch = D * FF * 4;
  // The trunk: patch embedding and head, and the folded time vectors.
  const U out_w = (U)(cfg.out_channels * cfg.patch_t * cfg.patch_h *
                      cfg.patch_w);
  const U in_w = (U)(cfg.in_channels * cfg.patch_t * cfg.patch_h *
                     cfg.patch_w);
  const U trunk = (D * in_w + D + out_w * D + out_w + 9 * D) * 2;
  return folds + 2 * block + scratch + trunk;
}

std::uint64_t
MetalFlashVsrTransformer::denoise_scratch_bytes(const Config& cfg, int height,
                                                int width, int frames,
                                                const Params& prm)
{
  using U = std::uint64_t;
  const U th = (U)(height / 16), tw = (U)(width / 16);
  const U D = (U)cfg.hidden, NH = (U)cfg.n_heads, HD = (U)cfg.head_dim;
  const int T = latent_frames(align_frames(frames));
  // THE OPENING CHUNK is the largest -- three windows, six latent frames
  // -- and the pool keeps what it grew to for the rest of the clip.
  const U seq = 6 * th * tw;
  const U n = seq * D * 2;                            // one [seq, D] bf16
  const U S = (th / 8) * (tw / 8);                    // blocks per window
  const U cap_blocks = (U)ring_windows(frames, prm.kv_ratio) * S;
  const U out_w =
      (U)(cfg.out_channels * cfg.patch_t * cfg.patch_h * cfg.patch_w);
  U b = kv_window_bytes(cfg, height, width, frames, prm);
  // A block's five scratch tensors and its feed-forward.
  b += 5 * n + seq * (U)cfg.ffn * 2;
  if (blocks_outer(cfg, height, width, frames, prm)) {
    // Every chunk's residual stream and RoPE tables; the source rows of
    // one chunk, at block 0; the head's output.
    const U all = (T > 0 ? (U)T : 6) * th * tw;
    b += all * D * 2 + all * HD * 4 * 2;
    b += n + seq * out_w * 2;
  } else {
    // The residual, the chunk's tokens and its source rows; the head's
    // output; RoPE.
    b += 3 * n + seq * out_w * 2;
    b += 2 * seq * HD * 4;
  }
  // Routing: block means, scores, the keep map, and the span CSR sized
  // for the ALU entry's 32-query / 16-key tiles -- the finer, larger one.
  const U nq = 3 * S;
  const U NQ = seq / 32;
  b += NH * (seq / 128) * HD * 2 + NH * cap_blocks * HD * 2;
  b += NH * nq * cap_blocks * 4 + NH * nq * cap_blocks;
  b += NH * (NQ + 1) * 4 + NH * NQ * cap_blocks * (128 / 16) * 4;
  // The clip's noise and latent, f32 on the host.
  if (T > 0) {
    b += 2 * (U)cfg.out_channels * (U)T * (U)(height / 8) * (U)(width / 8) *
         4;
  }
  return b;
}

// ------------------------------------------------------------- config

bool
MetalFlashVsrTransformer::config_from_checkpoint(const std::string& dit_dir,
                                                 Config*            out,
                                                 std::string*       err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (out == nullptr) { return fail("no output config"); }
  auto w = MetalLlamaWeights::open_model(dit_dir);
  if (!w.has_value()) { return fail("cannot open " + dit_dir); }

  Config c;
  // FlashVSR ships no transformer config -- its config.json names the
  // family and nothing else -- so every field comes off a tensor shape.
  // That beats a table keyed on a checkpoint hash, which is what the
  // reference does and what silently mis-loads the next release.
  const auto* pe = w->info("patch_embedding.weight");
  if (pe == nullptr || pe->shape.size() != 5) {
    return fail("patch_embedding.weight missing or not 5-D");
  }
  c.hidden      = (int)pe->shape[0];
  c.in_channels = (int)pe->shape[1];
  c.patch_t     = (int)pe->shape[2];
  c.patch_h     = (int)pe->shape[3];
  c.patch_w     = (int)pe->shape[4];

  int n = 0;
  while (w->has(blk(n, "self_attn.q.weight"))) { ++n; }
  if (n == 0) { return fail("no blocks.* in checkpoint"); }
  c.n_layers = n;

  const auto* ff = w->info(blk(0, "ffn.0.weight"));
  if (ff == nullptr || ff->shape.size() != 2) {
    return fail("blocks.0.ffn.0.weight missing");
  }
  c.ffn = (int)ff->shape[0];

  const auto* te = w->info("text_embedding.0.weight");
  if (te == nullptr || te->shape.size() != 2) {
    return fail("text_embedding.0.weight missing");
  }
  c.text_dim = (int)te->shape[1];

  const auto* tm = w->info("time_embedding.0.weight");
  if (tm == nullptr || tm->shape.size() != 2) {
    return fail("time_embedding.0.weight missing");
  }
  c.freq_dim = (int)tm->shape[1];

  const auto* hd = w->info("head.head.weight");
  if (hd == nullptr || hd->shape.size() != 2) {
    return fail("head.head.weight missing");
  }
  const int out_patch = (int)hd->shape[0];
  const int patch_vol = c.patch_t * c.patch_h * c.patch_w;
  if (patch_vol <= 0 || out_patch % patch_vol != 0) {
    return fail("head width is not a multiple of the patch volume");
  }
  c.out_channels = out_patch / patch_vol;

  // The one field no shape carries. Wan's head_dim is 128 at every size
  // it ships -- 1536/12 and 5120/40 alike -- so the head count follows
  // from the width rather than from a table.
  if (c.hidden % 128 != 0) {
    return fail(fmt("hidden {} is not a multiple of the 128 head_dim every "
                    "Wan config uses", c.hidden)());
  }
  c.head_dim = 128;
  c.n_heads  = c.hidden / 128;

  const auto* ctx = w->info("posi_context");
  if (ctx != nullptr && ctx->shape.size() == 2) {
    c.text_tokens = (int)ctx->shape[0];
  }
  *out = c;
  return true;
}

// --------------------------------------------------------------- load

std::unique_ptr<MetalFlashVsrTransformer>
MetalFlashVsrTransformer::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                               const Config& cfg, std::string* err)
{
  std::shared_ptr<WeightSet> ctx = ws;
  return load(std::move(ws), std::move(ctx), "posi_context", mc, cfg, err);
}

std::unique_ptr<MetalFlashVsrTransformer>
MetalFlashVsrTransformer::load(std::shared_ptr<WeightSet> ws,
                               std::shared_ptr<WeightSet> ctx_ws,
                               const std::string& ctx_name, MetalCompute* mc,
                               const Config& cfg, std::string* err)
{
  auto fail = [&](std::string m) -> std::unique_ptr<MetalFlashVsrTransformer> {
    if (err != nullptr) { *err = std::move(m); }
    return nullptr;
  };
  if (ws == nullptr || ctx_ws == nullptr || mc == nullptr) {
    return fail("no weights or backend");
  }

  auto m = std::unique_ptr<MetalFlashVsrTransformer>(
      new MetalFlashVsrTransformer());
  m->_cfg = cfg;
  m->_mc = mc;
  m->_ws = std::move(ws);
  m->_ctx_ws = std::move(ctx_ws);
  m->_ctx_name = ctx_name;
  WeightSet& w = *m->_ws;

  // ---- MATRIX CORES, on by default where the GPU has them -----------
  //
  // The rule every sibling DiT follows. Two independent switches because
  // the two kernels fail independently: the matmul2d GEMM tiles (with
  // the bias pass none of them applies) and the NAX flash entry, which
  // carries the span path and the int8 QK this model's attention needs.
  m->_lib_elt = mc->load_library("llm_elementwise_bf16");
  m->_splitk = std::make_unique<MmaSplitK>();
  if (mc->supports_matrix_cores()) {
    if (std::getenv("VPIPE_FVSR_NO_MMA2") == nullptr) {
      m->_lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
      m->_fn_gemm_mma =
          m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
      m->_fn_gemm_mma_wide =
          m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
      m->_fn_bias_rows = m->_lib_elt.function("bias_add_rows_f16");
      m->_use_mma2 = m->_fn_gemm_mma.valid() &&
                     m->_fn_gemm_mma_wide.valid() &&
                     m->_fn_bias_rows.valid();
      // The ff-down's 8960 sits below the split's measured floor, so this
      // splits nothing by default; it is loaded so the split's own env
      // knobs can reach this model for the experiment.
      if (m->_use_mma2) {
        m->_splitk->load(mc, m->_lib_dense_mma, m->_lib_elt);
      }
    }
    if (std::getenv("VPIPE_FVSR_NO_ATTN_NAX") == nullptr) {
      m->_lib_attn_nax = mc->load_library("attn_steel_nax");
      m->_use_attn_nax = m->_lib_attn_nax.valid();
    }
  }

  // ---- the two opt-in tiers -----------------------------------------
  //
  // Independent of each other and of the switches above: i8_gemm decides
  // how a block's GEMMs are computed, sage_attn how the attended keys are
  // multiplied. The routing that decides WHICH keys are attended is the
  // checkpoint's own and neither tier touches it.
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm,
                                              /*bf16=*/true);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }
  {
    bool fatal = false;
    m->_sage = MetalSageAttention::load_for_model(
        mc, /*bf16=*/true, cfg.sage, "MetalFlashVsrTransformer", &fatal);
    if (fatal) { return fail("sage_attn requested but its kernels failed"); }
    // Sage lives inside the NAX flash entry. Said rather than implied: a
    // run that asked for it and quietly got the f16 product would be
    // reported as a Sage run.
    if (m->_sage && !m->_use_attn_nax) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt(
            "MetalFlashVsrTransformer: sage_attn is off -- the matrix-core "
            "flash entry is unavailable here"));
      }
      m->_sage.reset();
    }
    if (m->_sage) {
      std::string serr;
      m->_sage_x = MetalSageAttention::load(mc, /*bf16=*/true, &serr);
      if (!m->_sage_x) { return fail("sage_attn (cross): " + serr); }
      // ...and a second of each for the OPENING chunk, whose three
      // windows are a different shape from every later chunk's one.
      // Block-outer runs both shapes inside every block, and a single
      // object would rebuild its scratch twice per block.
      m->_sage_open = MetalSageAttention::load(mc, /*bf16=*/true, &serr);
      m->_sage_x_open = MetalSageAttention::load(mc, /*bf16=*/true, &serr);
      if (!m->_sage_open || !m->_sage_x_open) {
        return fail("sage_attn (opening chunk): " + serr);
      }
    }
  }

  if (!m->fold_constants_(w, err)) { return nullptr; }
  if (!m->load_blocks_(w, err)) { return nullptr; }
  if (cfg.stream_blocks) { m->configure_streaming_(); }
  return m;
}

// The slots' half of the contract: how a block of this checkpoint is
// enumerated, built, copied and sized. BlockSlots owns the policy -- the
// pair, the read ahead, the fallback.
void
MetalFlashVsrTransformer::configure_streaming_()
{
  _streaming = std::make_unique<Streaming>();
  BlockSlots<Block>::Ops ops;
  // kBf16: placed as-is when the checkpoint is bf16, and refused as
  // UNSERVABLE when it is F32 -- this one is -- which is what routes the
  // read through fill_unservable below rather than an allocation.
  ops.each = [](int i, Block& b, const BlockSlots<Block>::TensorFn& fn) {
    each_run_tensor(i, b, [&](const std::string& nm, SharedBuffer& d) {
      fn(nm, d, Placement::kBf16);
    });
  };
  ops.rebuild_one = [this](const std::string& nm, Placement) {
    return _ws->stream_derived([&]() -> SharedBuffer {
      const auto v = read_f32(*_ws, _mc, nm);
      if (v.empty()) { return {}; }
      return bf16_buf(_mc, v.data(), v.size());
    });
  };
  // THE CHECKPOINT IS F32, twice the width of the slot, so a raw read has
  // nowhere to land: pread into one scratch for the run and narrow out of
  // it. The scratch settles at the largest tensor (the feed-forward's,
  // 55 MB) on the first block. Only one read is ever in flight -- an
  // acquire joins an outstanding prefetch before reading itself -- so one
  // scratch serves both.
  ops.fill_unservable = [this](const std::string& nm,
                               const SharedBuffer& dst) {
    const auto* info = _ws->src().info(nm);
    if (info == nullptr || info->shape.empty()) { return false; }
    std::size_t n = 1;
    for (const auto d : info->shape) { n *= (std::size_t)d; }
    if (dst.byte_size() < n * 2) { return false; }
    auto& sc = _streaming->scratch;
    if (sc.size() < (std::size_t)info->nbytes) {
      sc.resize((std::size_t)info->nbytes);
    }
    if (!_ws->stream_into(nm, sc.data(), (std::size_t)info->nbytes)) {
      return false;
    }
    auto* o = static_cast<std::uint16_t*>(dst.contents());
    if (info->dtype == "F32") {
      const auto* f = reinterpret_cast<const float*>(sc.data());
      for (std::size_t i = 0; i < n; ++i) { o[i] = to_bf16(f[i]); }
      return true;
    }
    if (info->dtype == "BF16") {
      std::memcpy(o, sc.data(), n * 2);
      return true;
    }
    return false;
  };
  ops.build = [this](int i, Block& b) {
    bool ok = true;
    each_run_tensor(i, b, [&](const std::string& nm, SharedBuffer& d) {
      if (!ok) { return; }
      const auto v = read_f32(*_ws, _mc, nm);
      d = v.empty() ? SharedBuffer{} : bf16_buf(_mc, v.data(), v.size());
      if (d.empty()) { ok = false; }
    });
    return ok;
  };
  ops.clone = [this](const Block& src, Block& dst, bool copy) {
    bool ok = true;
    // Field by field over the same list, src and dst in step.
    std::vector<const SharedBuffer*> from;
    each_run_tensor(0, const_cast<Block&>(src),
                    [&](const std::string&, SharedBuffer& b) {
                      from.push_back(&b);
                    });
    std::size_t k = 0;
    each_run_tensor(0, dst, [&](const std::string&, SharedBuffer& d) {
      const SharedBuffer& s0 = *from[k++];
      if (!ok || s0.empty()) { return; }
      d = _mc->make_shared_buffer(s0.byte_size());
      if (d.empty()) { ok = false; return; }
      if (copy) { std::memcpy(d.contents(), s0.contents(), s0.byte_size()); }
    });
    return ok;
  };
  ops.bytes = [](const Block& b) {
    std::size_t n = 0;
    each_run_tensor(0, const_cast<Block&>(b),
                    [&](const std::string&, SharedBuffer& d) {
                      n += d.byte_size();
                    });
    return n;
  };
  ops.empty = [](const Block& b) { return b.q1.empty(); };
  _streaming->slots.set_weight_set(_ws.get());
  _streaming->slots.configure(_mc, std::move(ops), "MetalFlashVsrTransformer",
                              "VPIPE_FVSR_NO_SLOTS");
}

bool
MetalFlashVsrTransformer::acquire_block_(int bi, int next, const Block** out,
                                         std::string* err)
{
  *out = nullptr;
  if (!_streaming) { return true; }
  const Block* b = _streaming->slots.acquire(bi);
  if (b == nullptr) {
    _streaming->slots.join();
    if (err != nullptr) {
      *err = fmt("block {} could not be read from the checkpoint", bi)();
    }
    return false;
  }
  // The other slot holds the block before this one, whose work has
  // finished -- every block commits and waits -- so the read ahead can
  // start now and run under ALL of this block's chunks.
  _streaming->slots.prefetch(next);
  *out = b;
  return true;
}

bool
MetalFlashVsrTransformer::streams_blocks() const noexcept
{
  return _streaming != nullptr;
}

int
MetalFlashVsrTransformer::prefetch_hits() const noexcept
{
  return _streaming ? _streaming->slots.prefetch_hits() : 0;
}

// THE CONSTANTS THIS MODEL LETS US FOLD.
//
// FlashVSR runs at one fixed timestep over one fixed conditioning, both
// shipped with the checkpoint, with guidance distilled away. So the
// timestep embedding, its six modulation vectors, the head's two, and
// every block's cross-attention key and value are computed ONCE here and
// never again. That removes two projections per block from the forward
// and leaves cross-attention reading a constant.
//
// All of it in f32 on the host: it is a few hundred megaflops once, and
// the reference builds its own timestep sinusoid in float64, so the
// cheap thing and the faithful thing agree.
bool
MetalFlashVsrTransformer::fold_constants_(WeightSet& w, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int D = _cfg.hidden;
  const int F = _cfg.freq_dim;

  // sinusoidal_embedding_1d, in double as the reference computes it.
  std::vector<float> sinu((std::size_t)F, 0.0f);
  {
    const double pos = (double)_cfg.timestep;
    const int half = F / 2;
    for (int i = 0; i < half; ++i) {
      const double f = std::pow(10000.0, -(double)i / (double)half);
      sinu[(std::size_t)i] = (float)std::cos(pos * f);
      sinu[(std::size_t)(half + i)] = (float)std::sin(pos * f);
    }
  }

  const auto te0w = read_f32(w, _mc, "time_embedding.0.weight");
  const auto te0b = read_f32(w, _mc, "time_embedding.0.bias");
  const auto te2w = read_f32(w, _mc, "time_embedding.2.weight");
  const auto te2b = read_f32(w, _mc, "time_embedding.2.bias");
  const auto tpw  = read_f32(w, _mc, "time_projection.1.weight");
  const auto tpb  = read_f32(w, _mc, "time_projection.1.bias");
  if (te0w.empty() || te2w.empty() || tpw.empty()) {
    return fail("time embedding weights missing");
  }

  std::vector<float> h1, t_vec, t_mod;
  host_linear(sinu, te0w, te0b, 1, D, F, &h1);
  for (auto& v : h1) { v = silu(v); }
  host_linear(h1, te2w, te2b, 1, D, D, &t_vec);

  std::vector<float> t_act = t_vec;
  for (auto& v : t_act) { v = silu(v); }
  host_linear(t_act, tpw, tpb, 1, 6 * D, D, &t_mod);

  _t_vec = bf16_buf(_mc, t_vec.data(), t_vec.size());
  _t_mod = bf16_buf(_mc, t_mod.data(), t_mod.size());
  if (_t_vec.empty() || _t_mod.empty()) { return fail("timestep fold failed"); }
  _t_vec_f = t_vec;
  _t_mod_f = t_mod;

  // The head's two modulation vectors are `head.modulation + t`, where
  // the HEAD is handed the timestep vector itself and not the six-way
  // projection -- a broadcast of [dim] over [2, dim].
  const auto hm = read_f32(w, _mc, "head.modulation");
  if (hm.size() != (std::size_t)2 * D) { return fail("head.modulation missing"); }
  std::vector<float> head_mod((std::size_t)2 * D, 0.0f);
  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < D; ++i) {
      head_mod[(std::size_t)c * D + i] =
          hm[(std::size_t)c * D + i] + t_vec[(std::size_t)i];
    }
  }
  _head_mod = bf16_buf(_mc, head_mod.data(), head_mod.size());
  _head_mod_f = std::move(head_mod);

  // The conditioning, through text_embedding, once. Its token count comes
  // off the tensor, which is [tokens, width] converted and
  // [1, tokens, width] as published.
  const auto* ci = _ctx_ws->src().info(_ctx_name);
  const auto ctx = read_f32(*_ctx_ws, _mc, _ctx_name);
  if (ci == nullptr || ctx.empty()) {
    return fail(fmt("{} missing -- the fixed conditioning is its own file "
                    "(posi_prompt.pth), which the reference ships in its "
                    "GitHub repo rather than on the model page", _ctx_name)());
  }
  const bool ctx_shape_ok =
      !ci->shape.empty() && ci->shape.back() == _cfg.text_dim &&
      (ci->shape.size() == 2 || (ci->shape.size() == 3 && ci->shape[0] == 1));
  if (!ctx_shape_ok) {
    return fail(fmt("{} is not [tokens, {}]", _ctx_name, _cfg.text_dim)());
  }
  _cfg.text_tokens = (int)(ctx.size() / (std::size_t)_cfg.text_dim);
  const int T = _cfg.text_tokens;
  const auto x0w = read_f32(w, _mc, "text_embedding.0.weight");
  const auto x0b = read_f32(w, _mc, "text_embedding.0.bias");
  const auto x2w = read_f32(w, _mc, "text_embedding.2.weight");
  const auto x2b = read_f32(w, _mc, "text_embedding.2.bias");
  if (x0w.empty() || x2w.empty()) { return fail("text_embedding missing"); }
  std::vector<float> c1;
  host_linear(ctx, x0w, x0b, T, D, _cfg.text_dim, &c1);
  for (auto& v : c1) { v = gelu_tanh(v); }
  host_linear(c1, x2w, x2b, T, D, D, &_ctx_f);
  return true;
}

bool
MetalFlashVsrTransformer::load_blocks_(WeightSet& w, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int D = _cfg.hidden;
  const int T = _cfg.text_tokens;

  auto mat = [&](const std::string& nm) -> SharedBuffer {
    return w.derived(kKey + std::string("m|") + nm, [&]() -> SharedBuffer {
      const auto v = read_f32(w, _mc, nm);
      if (v.empty()) { return {}; }
      return bf16_buf(_mc, v.data(), v.size());
    });
  };

  // The patch embedding's 5-D weight [dim, C, 1, 2, 2] is ALREADY in
  // the (channel, row, column) order the pack emits, so it is read flat
  // rather than permuted.
  _patch_w = mat("patch_embedding.weight");
  _patch_b = mat("patch_embedding.bias");
  _head_w  = mat("head.head.weight");
  _head_b  = mat("head.head.bias");
  if (_patch_w.empty() || _head_w.empty()) {
    return fail("patch embedding or output head missing");
  }

  _blocks.resize((std::size_t)_cfg.n_layers);
  for (int i = 0; i < _cfg.n_layers; ++i) {
    Block& b = _blocks[(std::size_t)i];
    // A STREAMING load keeps none of these: the slots read them when the
    // block runs. It still asks the checkpoint for every one, so a block
    // that is incomplete is refused here and not mid-clip.
    bool complete = true;
    each_run_tensor(i, b, [&](const std::string& nm, SharedBuffer& dst) {
      if (_cfg.stream_blocks) {
        if (w.src().info(nm) == nullptr) { complete = false; }
        return;
      }
      dst = mat(nm);
      if (dst.empty()) { complete = false; }
    });
    if (!complete) { return fail(fmt("block {} is incomplete", i)()); }

    // modulation + t_mod, folded. Both are [6, dim] constants.
    const auto modw = read_f32(w, _mc, blk(i, "modulation"));
    if (modw.size() != (std::size_t)6 * D) {
      return fail(fmt("blocks.{}.modulation is not [1,6,dim]", i)());
    }
    std::vector<float> mod((std::size_t)6 * D, 0.0f);
    for (std::size_t j = 0; j < mod.size(); ++j) {
      mod[j] = modw[j] + _t_mod_f[j];
    }
    b.mod = bf16_buf(_mc, mod.data(), mod.size());

    // The cross-attention key and value, over the fixed conditioning.
    // norm_k is applied to the key here, so the forward reads a constant
    // and runs neither projection nor norm.
    const auto kw = read_f32(w, _mc, blk(i, "cross_attn.k.weight"));
    const auto kb = read_f32(w, _mc, blk(i, "cross_attn.k.bias"));
    const auto vw = read_f32(w, _mc, blk(i, "cross_attn.v.weight"));
    const auto vb = read_f32(w, _mc, blk(i, "cross_attn.v.bias"));
    const auto kn = read_f32(w, _mc, blk(i, "cross_attn.norm_k.weight"));
    if (kw.empty() || vw.empty() || kn.empty()) {
      return fail(fmt("block {} cross-attention kv weights missing", i)());
    }
    std::vector<float> ck, cv;
    host_linear(_ctx_f, kw, kb, T, D, D, &ck);
    host_linear(_ctx_f, vw, vb, T, D, D, &cv);
    // RMS over the WHOLE projection, not per head -- the reference's
    // RMSNorm reduces the last axis, which is the full width before the
    // head split. Normalising per head is a different function.
    for (int r = 0; r < T; ++r) {
      float* row = ck.data() + (std::size_t)r * D;
      double acc = 0.0;
      for (int j = 0; j < D; ++j) { acc += (double)row[j] * (double)row[j]; }
      const float inv = (float)(1.0 / std::sqrt(acc / (double)D +
                                                (double)_cfg.norm_eps));
      for (int j = 0; j < D; ++j) { row[j] = row[j] * inv * kn[(std::size_t)j]; }
    }
    // Stored HEAD-MAJOR [H, tokens, head_dim], which is what the
    // attention reads. Transposing once here beats transposing a
    // constant on every chunk of every clip.
    const int NH = _cfg.n_heads, HD = _cfg.head_dim;
    std::vector<float> ckh(ck.size()), cvh(cv.size());
    for (int t = 0; t < T; ++t) {
      for (int hh = 0; hh < NH; ++hh) {
        for (int d = 0; d < HD; ++d) {
          const std::size_t src = (std::size_t)t * D + (std::size_t)hh * HD + d;
          const std::size_t dst = ((std::size_t)hh * T + t) * HD + d;
          ckh[dst] = ck[src];
          cvh[dst] = cv[src];
        }
      }
    }
    b.ck = bf16_buf(_mc, ckh.data(), ckh.size());
    b.cv = bf16_buf(_mc, cvh.data(), cvh.size());
    if (b.ck.empty() || b.cv.empty()) {
      return fail(fmt("block {} cross-attention fold failed", i)());
    }
  }
  return true;
}


// ------------------------------------------------------- the forward

namespace {

// Scratch pool, same shape as the projection's.
struct FwdCtx {
  ComputeEncoder* enc = nullptr;
  struct Slot { SharedBuffer buf; std::size_t cap; bool used; };
  std::deque<Slot> pool;
  bool alloc_ok = true;

  SharedBuffer&
  alloc(MetalCompute* mc, std::size_t elems)
  {
    const std::size_t bytes = elems * 2;
    for (auto& s : pool) {
      if (!s.used && !s.buf.empty() && s.cap >= bytes) {
        s.used = true;
        return s.buf;
      }
    }
    pool.push_back(Slot{mc->make_shared_buffer(bytes), bytes, true});
    if (pool.back().buf.empty()) { alloc_ok = false; }
    return pool.back().buf;
  }
  void release_all() { for (auto& s : pool) { s.used = false; } }
};

}  // namespace

// What one generate() holds for the clip: the key/value windows, the
// routing and span buffers, the RoPE tables and the block scratch.
//
// THE WINDOWS. Head-major [H, cap_tokens, head_dim] each, a RING of whole
// temporal windows allocated for one chunk more than the cache retains,
// so an append never has to grow and a trim moves nothing. The steel
// parameters carry the ALLOCATION stride separately from the logical
// length, which is what lets one buffer serve a growing cache without a
// reallocation per chunk.
//
// HOW MANY is the ORDER's question. Run chunk-outer (every block of a
// chunk, then the next chunk), every block's window has to survive until
// that block's next chunk: thirty of them, 12.7 GB at 1920x1152. Run
// block-outer (every chunk of a block, then the next block), a block's
// window is finished with before the next block starts, so ONE serves
// the whole stack. See generate().
struct MetalFlashVsrTransformer::Stream {
  struct Cache {
    // [H, cap_blocks * 128, HD] each, bf16: a RING of whole temporal
    // windows, so a trim moves nothing and needs no spare. It used to
    // shift the retained tail into a second K and V per block, which
    // doubled the window -- the largest allocation this model makes.
    metal_compute::SharedBuffer k, v;
    int valid  = 0;                  // temporal windows held
    int oldest = 0;                  // the slot holding the oldest of them
  };
  std::vector<Cache> per_block;      // one per block, or one for all
  int cap_windows = 0;               // ring capacity, temporal windows
  int cap_blocks = 0;                // ...in window blocks
  int one_len = 0;                   // window blocks in a steady chunk
  int h = 0, w = 0;                  // the token grid
  int topk = 0;                      // the routing's top-k, per clip
  // RoPE tables, built in WINDOW-PERMUTED order so the permutation
  // happens once on q/k/v rather than once per table lookup. One slot
  // rebuilt per chunk chunk-outer; one per chunk, kept, block-outer --
  // a rebuild is host trigonometry over the whole chunk.
  struct Rope {
    int f = -1, t_off = -1;
    metal_compute::SharedBuffer cos, sin;
  };
  std::vector<Rope> rope;
  // The routing's own buffers, sized once per clip for the largest chunk
  // so nothing is allocated per block.
  metal_compute::SharedBuffer band, p, thr, flags;
  // The span path's own buffers: the block-level decision, the CSR
  // offsets and list, and the two parameter blocks attn_steel reads
  // under has_spans.
  metal_compute::SharedBuffer keep, qb_off, qb_blocks, sp_params, sp_bounds;
  // The block's scratch pool, the routing's block means and the
  // attention's parameters, which every block reuses: each block commits
  // and waits before the next is encoded.
  FwdCtx cx;
  metal_compute::SharedBuffer qm_buf, km_buf, params;
  // VPIPE_FVSR_DIT_PROFILE, accumulated per section until reported.
  bool prof = false;
  double t_qkv = 0, t_prep = 0, t_route = 0, t_attn = 0, t_oproj = 0,
         t_cross = 0, t_ffn = 0, t_elt = 0;
  int kL_last = 0, seq_last = 0;
};

namespace {

// Mirrors steel/attn/params.h. Restated because the vendored header is
// a Metal source, not a C++ one.
struct AttnSpanParams {
  int video_start;
  int tokens_per_frame;
  int num_frames;
  int anchors;
  int qb_stride;
};

struct SteelAttnParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK;
  int NQ_aligned, NK_aligned;
  int qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

// Natural (frame, row, column) index of window-permuted position p.
// The inverse of flashvsr_window_permute; used to build the RoPE table
// in permuted order so the permutation is paid once, on q/k/v, rather
// than once per table lookup.
inline void
unpermute_pos(int p, int H, int W, int* f, int* y, int* x)
{
  const int nh = H / 8, nw = W / 8;
  const int blkid = p / 128, idx = p % 128;
  const int ww = blkid % nw;
  const int wh = (blkid / nw) % nh;
  const int wf = blkid / (nw * nh);
  *x = ww * 8 + (idx % 8);
  *y = wh * 8 + ((idx / 8) % 8);
  *f = wf * 2 + (idx / 64);
}

}  // namespace

void
MetalFlashVsrTransformer::build_rope_(int f_len, int h, int w, int t_off,
                                      SharedBuffer& cos_out,
                                      SharedBuffer& sin_out) const
{
  const int D = _cfg.head_dim;
  const int seq = f_len * h * w;
  const int axes[3] = {_cfg.rope_t(), _cfg.rope_h(), _cfg.rope_w()};
  const double theta = _cfg.rope_theta;
  auto* cb = static_cast<float*>(cos_out.contents());
  auto* sb = static_cast<float*>(sin_out.contents());

  const int pairs = D / 2;
  std::vector<double> pair_freq((std::size_t)pairs);
  std::vector<int> pair_axis((std::size_t)pairs), pair_off((std::size_t)pairs);
  {
    int base = 0, pair = 0;
    for (int a = 0; a < 3; ++a) {
      const int Dax = axes[a];
      for (int i = 0; i < Dax / 2; ++i, ++pair) {
        pair_freq[(std::size_t)pair] =
            1.0 / std::pow(theta, (2.0 * i) / (double)Dax);
        pair_axis[(std::size_t)pair] = a;
        pair_off[(std::size_t)pair] = base + 2 * i;
      }
      base += Dax;
    }
  }
  for (int p = 0; p < seq; ++p) {
    int ff = 0, yy = 0, xx = 0;
    unpermute_pos(p, h, w, &ff, &yy, &xx);
    // THE TEMPORAL POSITION IS ABSOLUTE across the clip, not per chunk:
    // the reference indexes its frequency table at 4 + 2*chunk, so a
    // chunk's frames keep the positions they had in the whole video.
    // Resetting it per chunk is a plausible-looking port that makes
    // every chunk think it is the opening one.
    const double coord[3] = {(double)(ff + t_off), (double)yy, (double)xx};
    for (int pair = 0; pair < pairs; ++pair) {
      const double ang =
          coord[pair_axis[(std::size_t)pair]] * pair_freq[(std::size_t)pair];
      const float c = (float)std::cos(ang);
      const float sn = (float)std::sin(ang);
      const std::size_t o = (std::size_t)p * D + pair_off[(std::size_t)pair];
      cb[o] = c; cb[o + 1] = c;
      sb[o] = sn; sb[o + 1] = sn;
    }
  }
}

// FlashVSR's locality-constrained sparse attention, decided on the host.
//
// THE DECISION IS THE MODEL; the skipping is an optimisation. This
// builds the same block mask the reference builds, from the same block
// means, and hands it to the tree's own flash kernel through the
// per-query-block key flags it already accepts -- so no attention kernel
// is written here at all.
//
// Why the host: the scores are [heads, Nq, Nk] over WINDOW BLOCKS, which
// at any real geometry is thousands of numbers, not millions. The means
// are reduced on the GPU (flashvsr_block_mean) precisely so that this
// reads back kilobytes.
//
// `qm`/`km` are block means [H, Nq, HD] and [H, Nk, HD]; `sq` is the
// number of TEMPORAL windows in the query, which is what the reference
// groups its top-k by. Returns flags [H, NQ_steel, kL], nonzero EXCLUDES.
std::vector<std::uint8_t>
MetalFlashVsrTransformer::route_(const std::vector<float>& qm,
                                 const std::vector<float>& km, int Nq, int Nk,
                                 int sq, int nh_blk, int nw_blk, int topk,
                                 int local_range, int kL, int bq) const
{
  const int H = _cfg.n_heads, HD = _cfg.head_dim;
  const int S = nh_blk * nw_blk;             // spatial blocks per frame pair
  const float inv = 1.0f / std::sqrt((float)HD);

  // The always-exact spatial band, over the (nh, nw) block grid. The
  // reference's slide is UNCLAMPED -- a block near an edge simply has
  // fewer neighbours -- which is not the same as clamping the window
  // back inside the grid.
  std::vector<std::uint8_t> band((std::size_t)S * S, 0);
  const int half = local_range / 2;
  for (int i = 0; i < S; ++i) {
    const int ry = i / nw_blk, rx = i % nw_blk;
    for (int j = 0; j < S; ++j) {
      const int cy = j / nw_blk, cx = j % nw_blk;
      const bool in = (cy >= ry - half) && (cy <= ry - half + local_range - 1) &&
                      (cx >= rx - half) && (cx <= rx - half + local_range - 1);
      band[(std::size_t)i * S + j] = in ? 1u : 0u;
    }
  }

  const int NQ = (kL > 0) ? ((_seq_for_route + bq - 1) / bq) : 0;
  std::vector<std::uint8_t> flags((std::size_t)H * NQ * kL, 1u);
  const int s1 = (sq > 0) ? Nq / sq : Nq;    // query blocks per group

  std::vector<float> row((std::size_t)Nk, 0.0f);
  std::vector<float> group((std::size_t)s1 * Nk, 0.0f);
  std::vector<float> sorted;
  std::vector<std::uint8_t> keep((std::size_t)Nq * Nk, 0u);

  for (int h = 0; h < H; ++h) {
    for (int g = 0; g < sq; ++g) {
      for (int r = 0; r < s1; ++r) {
        const int i = g * s1 + r;
        const float* qv = qm.data() + ((std::size_t)h * Nq + i) * HD;
        float mx = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < Nk; ++j) {
          if (band[(std::size_t)(i % S) * S + (j % S)] == 0u) {
            row[(std::size_t)j] = -std::numeric_limits<float>::infinity();
            continue;
          }
          const float* kv = km.data() + ((std::size_t)h * Nk + j) * HD;
          double acc = 0.0;
          for (int d = 0; d < HD; ++d) { acc += (double)qv[d] * (double)kv[d]; }
          row[(std::size_t)j] = (float)acc * inv;
          if (row[(std::size_t)j] > mx) { mx = row[(std::size_t)j]; }
        }
        // Softmax, so a band-excluded block becomes exactly zero -- which
        // is what makes the top-k below drop it. The two mechanisms are
        // not independent: at a grid the band covers entirely, the top-k
        // drops only the single lowest block.
        double den = 0.0;
        for (int j = 0; j < Nk; ++j) {
          const float v = row[(std::size_t)j];
          const double e = (v == -std::numeric_limits<float>::infinity())
                               ? 0.0 : std::exp((double)(v - mx));
          group[(std::size_t)r * Nk + j] = (float)e;
          den += e;
        }
        if (den > 0.0) {
          for (int j = 0; j < Nk; ++j) {
            group[(std::size_t)r * Nk + j] /= (float)den;
          }
        }
      }
      // The threshold is the (k+1)-th largest of the WHOLE group, and
      // the comparison is strictly greater -- so every zeroed block goes
      // when the threshold lands at zero, which is the usual case.
      const std::size_t n = (std::size_t)s1 * Nk;
      const int apply = std::min((int)n - 1, topk);
      sorted.assign(group.begin(), group.begin() + (std::ptrdiff_t)n);
      std::nth_element(sorted.begin(),
                       sorted.begin() + (std::ptrdiff_t)(apply),
                       sorted.end(), std::greater<float>());
      const float thr = sorted[(std::size_t)apply];
      for (int r = 0; r < s1; ++r) {
        const int i = g * s1 + r;
        for (int j = 0; j < Nk; ++j) {
          keep[(std::size_t)i * Nk + j] =
              group[(std::size_t)r * Nk + j] > thr ? 1u : 0u;
        }
      }
    }
    // Expand window blocks to the flash kernel's own query tiles: one
    // routing block spans 128/bq of them, all sharing a mask row.
    for (int qb = 0; qb < NQ; ++qb) {
      const int i = std::min(Nq - 1, (qb * bq) / 128);
      std::uint8_t* f = flags.data() + ((std::size_t)h * NQ + qb) * kL;
      for (int k = 0; k < kL; ++k) {
        const int j = std::min(Nk - 1, k / 128);
        f[(std::size_t)k] = keep[(std::size_t)i * Nk + j] ? 0u : 1u;
      }
    }
  }
  return flags;
}

// TWO DENSE GEMMS, and the difference is the whole of this model's
// throughput.
//
// `dense_gemm_bias` is the straightforward one: a 16x16 threadgroup, one
// output element per thread, every operand read from device memory. It
// is what this port used while it was being made correct, and it is what
// left the denoise at 11% of roofline.
//
// `dense_gemm_t_bm64` tiles 64 rows by 32 columns per threadgroup, which
// is what the Wan denoiser next door runs and what every other DiT here
// reaches for. Same operands in the same slots -- but the CONSTANTS ARE
// IN A DIFFERENT ORDER (K, N, M against M, N, K) and the dispatch
// geometry is its own. Both mistakes compute a wrong answer of the right
// shape, which is why the swap is guarded by the goldens.
//
// VPIPE_FVSR_PLAIN_GEMM selects the old one, for an A/B inside ONE
// process: this box has a few percent of thermal spread between runs,
// so a cross-process comparison cannot resolve a change of that size.
void
MetalFlashVsrTransformer::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                                const SharedBuffer& w, const SharedBuffer& b,
                                const SharedBuffer& y, int M, int N, int K)
{
  // ACCELERATED MODE FIRST, bias and all: it takes the same operands the
  // tiles below do and declines with nothing encoded on a shape it does
  // not want (M under 1024 rows, K under 1024), so it sits ahead of the
  // tile choice rather than beside it. Then the matrix cores, then the
  // ALU tiles this port was verified on.
  if (_i8 && _i8->gemm(enc, x, 0, w, b, y, 0, M, N, K)) { return; }
  if (gemm_mma_(enc, x, w, b, y, M, N, K)) { return; }
  if (_gemm_tile == 2) {
    // BM 64 / BN 64: a wider output tile re-reads the activation half as
    // often, at the cost of registers and therefore occupancy. Which way
    // that lands is a per-shape question, so it is a knob and not a
    // rule -- the same one Boogu's DiT carries.
    enc.set_function(_fn_gemm_bn64);
    enc.set_buffer(0, x); enc.set_buffer(1, w);
    enc.set_buffer(2, b.empty() ? w : b);
    enc.set_buffer(3, y);
    enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
    enc.set_constant(7, b.empty() ? 0 : 1);
    enc.dispatch({(unsigned)(((N + 63) / 64) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
    return;
  }
  if (_gemm_tile == 0) {
    enc.set_function(_fn_gemm);
    enc.set_buffer(0, x); enc.set_buffer(1, w);
    enc.set_buffer(2, b.empty() ? w : b);
    enc.set_buffer(3, y);
    enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
    enc.set_constant(7, b.empty() ? 0 : 1);
    enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                  (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
    return;
  }
  enc.set_function(_fn_gemm_bm64);
  enc.set_buffer(0, x); enc.set_buffer(1, w);
  enc.set_buffer(2, b.empty() ? w : b);
  enc.set_buffer(3, y);
  enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
  enc.set_constant(7, b.empty() ? 0 : 1);
  // BM 64 / BN 32, threadgroup (32, 2, 2).
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

// THE MATRIX-CORE ROUTE, bias included.
//
// No matmul2d tile applies a bias -- each takes the slot for signature
// compatibility with the ALU kernel and discards it -- and every
// projection in this model carries one, so the bias is a second pass.
// Without it the answer is not slower but different, which is why load()
// declines the whole route when that kernel is missing.
//
// K is 1536 for every projection and 8960 for the ff-down, which takes
// the wider tile (shared/mma-tile.h). Rows are banded on the tiles'
// 32-bit addressing: at 25344 rows -- the opening chunk at 768x1408 --
// the ff-down's source is 454 MB, far under the line, so the band is one
// pass at every geometry this model is sized for. It is here because the
// line belongs to the operands, and a band is cheaper than remembering
// where the line is.
bool
MetalFlashVsrTransformer::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                                   const SharedBuffer& w, const SharedBuffer& b,
                                   const SharedBuffer& y, int M, int N, int K)
{
  if (!_use_mma2 || M < 64 || N < 16) { return false; }
  if (!_splitk->encode(_mc, enc, x, w, y, K, N, M)) {
    const bool wide = mma_use_wide_tile(N, K);
    const metal_compute::ComputeFunction& fn =
        wide ? _fn_gemm_mma_wide : _fn_gemm_mma;
    const int RN = wide ? 256 : 128;
    const int band = mma_row_band(N, K);
    for (int r0 = 0; r0 < M; r0 += band) {
      const int rows = std::min(band, M - r0);
      enc.set_function(fn);
      enc.set_buffer(0, x, (std::size_t)r0 * K * 2);
      enc.set_buffer(1, w);
      enc.set_buffer(2, w);              // unread; the slot must be bound
      enc.set_buffer(3, y, (std::size_t)r0 * N * 2);
      enc.set_constant(4, K); enc.set_constant(5, N);
      enc.set_constant(6, rows); enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                    (unsigned)((rows + 127) / 128), 1}, {256, 1, 1});
    }
  }
  if (!b.empty()) {
    enc.set_function(_fn_bias_rows);
    enc.set_buffer(0, y); enc.set_buffer(1, b);
    enc.set_constant(2, N);
    const unsigned total = (unsigned)((std::size_t)M * N);
    enc.set_constant(3, total);
    enc.dispatch({(unsigned)N, (unsigned)M, 1}, {32, 8, 1});
  }
  return true;
}

void
MetalFlashVsrTransformer::rms_(ComputeEncoder& enc, const SharedBuffer& x,
                               const SharedBuffer& g, const SharedBuffer& y,
                               int rows, int C)
{
  enc.set_function(_fn_rms);
  enc.set_buffer(0, x); enc.set_buffer(1, g); enc.set_buffer(2, y);
  enc.set_constant(3, C); enc.set_constant(4, _cfg.norm_eps);
  enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
}

void
MetalFlashVsrTransformer::permute_(ComputeEncoder& enc, const SharedBuffer& x,
                                   const SharedBuffer& y, int f, int h, int w,
                                   int D, bool inverse)
{
  enc.set_function(_fn_window);
  enc.set_buffer(0, x); enc.set_buffer(1, y);
  enc.set_constant(2, f); enc.set_constant(3, h); enc.set_constant(4, w);
  enc.set_constant(5, D); enc.set_constant(6, inverse ? 1 : 0);
  enc.dispatch({(unsigned)D, (unsigned)(f * h * w), 1}, {64, 1, 1});
}

void
MetalFlashVsrTransformer::trope_(ComputeEncoder& enc, const SharedBuffer& x,
                                 const SharedBuffer& y,
                                 const SharedBuffer& rcos,
                                 const SharedBuffer& rsin, int NH, int seq,
                                 int HD)
{
  enc.set_function(_fn_trope);
  enc.set_buffer(0, x); enc.set_buffer(1, y);
  enc.set_buffer(2, rcos); enc.set_buffer(3, rsin);
  enc.set_constant(4, NH); enc.set_constant(5, seq); enc.set_constant(6, HD);
  enc.dispatch({(unsigned)(HD / 2), (unsigned)seq, (unsigned)NH},
               {(unsigned)(HD / 2), 1, 1});
}

void
MetalFlashVsrTransformer::blockmean_(ComputeEncoder& enc,
                                     const SharedBuffer& x,
                                     const SharedBuffer& y, int HD, int seq,
                                     int blk, int NH, int nb)
{
  // `seq` is the SOURCE slab stride and `nb` both the blocks wanted and
  // the DESTINATION stride. They are independent: a cache reduces only
  // its live prefix out of a longer allocation, and deriving the
  // destination stride from the source one silently shifts every head
  // after the first.
  enc.set_function(_fn_blockmean);
  enc.set_buffer(0, x); enc.set_buffer(1, y);
  enc.set_constant(2, HD); enc.set_constant(3, seq); enc.set_constant(4, blk);
  enc.set_constant(5, nb);
  enc.dispatch({(unsigned)HD, (unsigned)nb, (unsigned)NH},
               {(unsigned)std::min(HD, 64), 1, 1});
}

// Block `bi`'s SELF-ATTENTION over one chunk, from tokens that already
// carry the source projection.
//
// A seam of its own because it is where every new mechanism in this
// model meets: the modulation folded at load, the three-axis RoPE at an
// absolute temporal offset, the 3-D window permutation, the routing, and
// the masked flash attention. Everything before it is a projection and
// everything after it is a feed-forward, so a failure here is unambiguous.
bool
MetalFlashVsrTransformer::block_self_attention(int bi, const float* x_tokens,
                                               int f, int h, int w, int t_off,
                                               const Params& prm,
                                               std::vector<float>* out,
                                               std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (x_tokens == nullptr || out == nullptr) { return fail("no tokens"); }
  if (bi < 0 || bi >= (int)_blocks.size()) { return fail("no such block"); }
  if (f <= 0 || f % 2 != 0 || h % 8 != 0 || w % 8 != 0) {
    return fail("chunk geometry must divide the (2,8,8) window");
  }
  if (!ensure_kernels_(err)) { return false; }

  const Block& b = _blocks[(std::size_t)bi];
  const int D = _cfg.hidden, NH = _cfg.n_heads, HD = _cfg.head_dim;
  const int seq = f * h * w;
  const int nb = seq / 128;                 // window blocks
  const int nh_blk = h / 8, nw_blk = w / 8;
  const int S = nh_blk * nw_blk;            // = window_size, blocks per pair
  const int sq = f / 2;                     // temporal windows in the query
  const int kL = seq;                       // no cache on this seam
  const int Nk = nb;

  // ---- RoPE, permuted and at an absolute offset.
  SharedBuffer rcos = _mc->make_shared_buffer((std::size_t)seq * HD * 4);
  SharedBuffer rsin = _mc->make_shared_buffer((std::size_t)seq * HD * 4);
  if (rcos.empty() || rsin.empty()) { return fail("rope table allocation"); }
  build_rope_(f, h, w, t_off, rcos, rsin);

  SharedBuffer x = _mc->make_shared_buffer((std::size_t)seq * D * 2);
  if (x.empty()) { return fail("token upload allocation"); }
  {
    auto* p = (std::uint16_t*)x.contents();
    for (std::size_t i = 0; i < (std::size_t)seq * D; ++i) {
      p[i] = to_bf16(x_tokens[i]);
    }
  }

  FwdCtx cx;
  SharedBuffer qm_buf = _mc->make_shared_buffer((std::size_t)NH * nb * HD * 2);
  SharedBuffer km_buf = _mc->make_shared_buffer((std::size_t)NH * Nk * HD * 2);
  SharedBuffer params = _mc->make_shared_buffer(sizeof(SteelAttnParams));
  if (qm_buf.empty() || km_buf.empty() || params.empty()) {
    return fail("routing allocation failed");
  }

  const int A_BQ = attn_bq_(), A_BK = attn_bk_();
  {
    auto* p = static_cast<SteelAttnParams*>(params.contents());
    p->B = 1; p->H = NH; p->D = HD;
    p->qL = seq; p->kL = kL;
    p->gqa_factor = 1;
    p->scale = 1.0f / std::sqrt((float)HD);
    p->NQ = (seq + A_BQ - 1) / A_BQ;
    p->NK = (kL + A_BK - 1) / A_BK;
    p->NQ_aligned = seq / A_BQ;
    p->NK_aligned = kL / A_BK;
    p->qL_rem = seq - p->NQ_aligned * A_BQ;
    p->kL_rem = kL - p->NK_aligned * A_BK;
    p->qL_off = 0;
    p->Q_strides[0] = (std::int64_t)NH * seq * HD;
    p->Q_strides[1] = (std::int64_t)seq * HD;
    p->Q_strides[2] = HD;
    p->K_strides[0] = (std::int64_t)NH * kL * HD;
    p->K_strides[1] = (std::int64_t)kL * HD;
    p->K_strides[2] = HD;
    p->V_strides[0] = p->K_strides[0];
    p->V_strides[1] = p->K_strides[1];
    p->V_strides[2] = HD;
    p->O_strides[0] = p->Q_strides[0];
    p->O_strides[1] = p->Q_strides[1];
    p->O_strides[2] = HD;
  }
  metal_compute::FunctionConstants fc;
  fc.set_bool(200, (seq % A_BQ) == 0).set_bool(201, (kL % A_BK) == 0)
      .set_bool(300, false).set_bool(301, false).set_bool(302, false)
      .set_bool(305, true);
  metal_compute::ComputeFunction fn_attn =
      attn_fn_(fc);
  if (!fn_attn.valid()) { return fail("masked steel attention unavailable"); }

  const std::size_t n_tok = (std::size_t)seq * D;
  SharedBuffer& y  = cx.alloc(_mc, n_tok);
  SharedBuffer& q  = cx.alloc(_mc, n_tok);
  SharedBuffer& k  = cx.alloc(_mc, n_tok);
  SharedBuffer& v  = cx.alloc(_mc, n_tok);
  SharedBuffer& qp = cx.alloc(_mc, n_tok);
  SharedBuffer& kp = cx.alloc(_mc, n_tok);
  SharedBuffer& vp = cx.alloc(_mc, n_tok);
  SharedBuffer& qh = cx.alloc(_mc, n_tok);
  SharedBuffer& kh = cx.alloc(_mc, n_tok);
  SharedBuffer& vh = cx.alloc(_mc, n_tok);
  if (!cx.alloc_ok) { return fail("self-attention scratch allocation failed"); }

  // ---- pass one: everything up to the block means.
  {
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    cx.enc = &enc;

    // modulate(norm1(x), shift_msa, scale_msa): rows 0 and 1 of the
    // six folded vectors.
    enc.set_function(_fn_ln_mod);
    enc.set_buffer(0, x);
    enc.set_buffer(1, b.mod, (std::size_t)1 * D * 2);   // scale
    enc.set_buffer(2, b.mod, 0);                        // shift
    enc.set_buffer(3, y);
    enc.set_constant(4, D);
    enc.set_constant(5, _cfg.norm_eps);
    enc.dispatch({256, (unsigned)seq, 1}, {256, 1, 1});

    gemm_(enc, y, b.q1, b.q1b, q, seq, D, D);
    gemm_(enc, y, b.k1, b.k1b, k, seq, D, D);
    gemm_(enc, y, b.v1, b.v1b, v, seq, D, D);

    // q/k RMS norm runs over the WHOLE projection, before the head
    // split -- normalising per head is a different function.
    rms_(enc, q, b.qn1, q, seq, D);
    rms_(enc, k, b.kn1, k, seq, D);

    permute_(enc, q, qp, f, h, w, D, false);
    permute_(enc, k, kp, f, h, w, D, false);
    permute_(enc, v, vp, f, h, w, D, false);

    trope_(enc, qp, qh, rcos, rsin, NH, seq, HD);
    trope_(enc, kp, kh, rcos, rsin, NH, seq, HD);
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, vp); enc.set_buffer(1, vh);
    enc.set_constant(2, seq); enc.set_constant(3, NH); enc.set_constant(4, HD);
    enc.dispatch({(unsigned)HD, (unsigned)NH, (unsigned)seq},
                 {(unsigned)HD, 1, 1});

    blockmean_(enc, qh, qm_buf, HD, seq, 128, NH, nb);
    blockmean_(enc, kh, km_buf, HD, kL, 128, NH, Nk);
    enc.end();
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) {
      return fail("self-attention prologue failed: " + ge);
    }
  }

  // ---- the routing, on the host from the means just reduced.
  std::vector<float> qm((std::size_t)NH * nb * HD);
  std::vector<float> km((std::size_t)NH * Nk * HD);
  {
    const auto* p = (const std::uint16_t*)qm_buf.contents();
    for (std::size_t i = 0; i < qm.size(); ++i) { qm[i] = from_bf16(p[i]); }
    const auto* r = (const std::uint16_t*)km_buf.contents();
    for (std::size_t i = 0; i < km.size(); ++i) { km[i] = from_bf16(r[i]); }
  }
  // The reference scales its own knob by the frame area, so one value
  // means one selectivity at every resolution.
  const double topk_ratio =
      prm.sparse_ratio * 768.0 * 1280.0 / ((double)(h * 16) * (w * 16));
  const int topk = (int)((double)(S * S) * topk_ratio) - 1;
  _seq_for_route = seq;
  const auto flags =
      route_(qm, km, nb, Nk, sq, nh_blk, nw_blk, topk, prm.local_range, kL,
             A_BQ);
  SharedBuffer flag_buf = _mc->make_shared_buffer(flags.size());
  if (flag_buf.empty()) { return fail("flag allocation failed"); }
  std::memcpy(flag_buf.contents(), flags.data(), flags.size());

  // ---- pass two: the masked attention and the output projection.
  SharedBuffer& oh = cx.alloc(_mc, n_tok);
  SharedBuffer& op = cx.alloc(_mc, n_tok);
  SharedBuffer& on = cx.alloc(_mc, n_tok);
  SharedBuffer& o  = cx.alloc(_mc, n_tok);
  if (!cx.alloc_ok) { return fail("attention output allocation failed"); }
  {
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    cx.enc = &enc;
    enc.set_function(fn_attn);
    enc.set_buffer(0, qh); enc.set_buffer(1, kh); enc.set_buffer(2, vh);
    enc.set_buffer(3, oh); enc.set_buffer(4, params);
    enc.set_buffer(14, flag_buf);
    enc.dispatch({32u * (unsigned)((seq + A_BQ - 1) / A_BQ),
                  4u * (unsigned)NH, 1}, {32, 4, 1});

    enc.set_function(_fn_transpose);
    enc.set_buffer(0, oh); enc.set_buffer(1, op);
    enc.set_constant(2, NH); enc.set_constant(3, seq); enc.set_constant(4, HD);
    enc.dispatch({(unsigned)HD, (unsigned)seq, (unsigned)NH},
                 {(unsigned)HD, 1, 1});

    permute_(enc, op, on, f, h, w, D, true);
    gemm_(enc, on, b.o1, b.o1b, o, seq, D, D);
    enc.end();
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) {
      return fail("masked attention failed: " + ge);
    }
  }

  out->assign(n_tok, 0.0f);
  const auto* p = (const std::uint16_t*)o.contents();
  for (std::size_t i = 0; i < n_tok; ++i) { (*out)[i] = from_bf16(p[i]); }
  return true;
}

// One chunk through the whole stack: 30 blocks, then the head.
//
// The key/value cache is what makes this a chunk rather than a clip. Per
// block it holds the window-permuted keys and values of the last
// `kv_ratio` chunks, head-major, in a buffer sized one chunk larger so an
// append never reallocates. The steel parameters carry the ALLOCATION
// stride separately from the logical length, which is what lets a growing
// cache live in one buffer.
// ----------------------------------------------- the clip's state

bool
MetalFlashVsrTransformer::begin_clip_(Stream& st, int h, int w, int f_max,
                                      int cap_windows, int n_caches,
                                      const Params& prm, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int NH = _cfg.n_heads, HD = _cfg.head_dim;
  const int A_BQ = attn_bq_(), A_BK = attn_bk_();
  const int nh_blk = h / 8, nw_blk = w / 8;
  const int S = nh_blk * nw_blk;
  // The LARGEST chunk sizes everything per-chunk here: the opening one,
  // three temporal windows, in every clip generate() runs.
  const int wq_max = std::max(1, f_max / 2);
  st.h = h;
  st.w = w;
  st.one_len = S;
  st.cap_windows = std::max(cap_windows, wq_max);
  st.cap_blocks = st.cap_windows * S;
  const std::size_t cap_tok = (std::size_t)st.cap_blocks * 128;
  st.per_block.clear();
  st.per_block.resize((std::size_t)n_caches);
  for (auto& c : st.per_block) {
    c.k = _mc->make_shared_buffer((std::size_t)NH * cap_tok * HD * 2);
    c.v = _mc->make_shared_buffer((std::size_t)NH * cap_tok * HD * 2);
    if (c.k.empty() || c.v.empty()) {
      return fail("key/value window allocation failed");
    }
  }
  const int nq_max = wq_max * S;
  st.band = _mc->make_shared_buffer((std::size_t)S * S);
  st.p = _mc->make_shared_buffer((std::size_t)NH * nq_max *
                                 st.cap_blocks * 4);
  st.thr = _mc->make_shared_buffer((std::size_t)NH * wq_max * 4);
  // The per-KEY flags belong to the MASK path; the span path reads the
  // block-level keep map and never these. They are hundreds of MB at
  // full HD, so they exist only when something will read them.
  if (!_use_spans) {
    st.flags = _mc->make_shared_buffer(
        (std::size_t)NH * ((nq_max * 128 + A_BQ - 1) / A_BQ) *
        ((std::size_t)st.cap_blocks * 128));
  }
  if (st.band.empty() || st.p.empty() || st.thr.empty() ||
      (!_use_spans && st.flags.empty())) {
    return fail("routing buffer allocation failed");
  }
  build_band_(nh_blk, nw_blk, prm.local_range, st.band);
  if (_use_spans) {
    // In the ATTENTION ENTRY'S tiles, like everything the kernel reads.
    const int NQ_max = (f_max * h * w + A_BQ - 1) / A_BQ;
    st.keep = _mc->make_shared_buffer((std::size_t)NH * nq_max *
                                      st.cap_blocks);
    st.qb_off = _mc->make_shared_buffer((std::size_t)NH * (NQ_max + 1) * 4);
    st.qb_blocks = _mc->make_shared_buffer(
        (std::size_t)NH * NQ_max * st.cap_blocks * (128 / A_BK) * 4);
    st.sp_params = _mc->make_shared_buffer(sizeof(AttnSpanParams));
    st.sp_bounds = _mc->make_shared_buffer(sizeof(int) * 2);
    if (st.keep.empty() || st.qb_off.empty() || st.qb_blocks.empty() ||
        st.sp_params.empty() || st.sp_bounds.empty()) {
      return fail("span buffer allocation failed");
    }
    // tokens_per_frame = 0 switches OFF the window mask attn_steel
    // applies under has_spans: its `q_video` test is false for every
    // row, so span_bounds is never dereferenced. The spans here are
    // the whole selection.
    auto* sp = static_cast<AttnSpanParams*>(st.sp_params.contents());
    sp->video_start = 0;
    sp->tokens_per_frame = 0;
    sp->num_frames = 0;
    sp->anchors = 0;
    sp->qb_stride = 0;          // filled per block, in block_
    std::memset(st.sp_bounds.contents(), 0, sizeof(int) * 2);
  }
  st.qm_buf = _mc->make_shared_buffer((std::size_t)NH * nq_max * HD * 2);
  st.km_buf = _mc->make_shared_buffer((std::size_t)NH * st.cap_blocks * HD *
                                      2);
  st.params = _mc->make_shared_buffer(sizeof(SteelAttnParams));
  if (st.qm_buf.empty() || st.km_buf.empty() || st.params.empty()) {
    return fail("routing allocation failed");
  }
  const double topk_ratio =
      prm.sparse_ratio * 768.0 * 1280.0 / ((double)(h * 16) * (w * 16));
  st.topk = (int)((double)(S * S) * topk_ratio) - 1;
  st.prof = std::getenv("VPIPE_FVSR_DIT_PROFILE") != nullptr;
  return true;
}

bool
MetalFlashVsrTransformer::rope_(Stream& st, int slot, int f, int t_off)
{
  if (slot < 0) { return false; }
  if ((int)st.rope.size() <= slot) { st.rope.resize((std::size_t)slot + 1); }
  Stream::Rope& rp = st.rope[(std::size_t)slot];
  if (rp.f != f || rp.t_off != t_off || rp.cos.empty()) {
    const std::size_t n = (std::size_t)f * st.h * st.w * _cfg.head_dim * 4;
    rp.cos = _mc->make_shared_buffer(n);
    rp.sin = _mc->make_shared_buffer(n);
    if (rp.cos.empty() || rp.sin.empty()) {
      rp = Stream::Rope{};
      return false;
    }
    build_rope_(f, st.h, st.w, t_off, rp.cos, rp.sin);
    rp.f = f;
    rp.t_off = t_off;
  }
  return true;
}

// ----------------------------------------------------- one block

// Block `bi` over one chunk's residual stream `x`, IN PLACE, against the
// key/value window `cache_idx`, which this appends the chunk to and then
// trims. Whichever ORDER drives it -- every block of a chunk before the
// next chunk, or every chunk of a block before the next block -- a block
// sees the same inputs: its own residual, and its own window holding the
// same earlier chunks of the same block.
bool
MetalFlashVsrTransformer::block_(Stream& st, int bi, const Block* wb,
                                 int cache_idx, int f, int t_off,
                                 int rope_slot, SharedBuffer& x,
                                 const SharedBuffer* lq, const Params& prm,
                                 std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int D = _cfg.hidden, NH = _cfg.n_heads, HD = _cfg.head_dim;
  const int FF = _cfg.ffn;
  const int h = st.h, w = st.w;
  const int seq = f * h * w;
  const int nb = seq / 128;
  const int S = (h / 8) * (w / 8);              // window blocks per window
  const int sq = f / 2;
  const int T = _cfg.text_tokens;
  const int A_BQ = attn_bq_(), A_BK = attn_bk_();
  const int topk = st.topk;
  const std::size_t cap_tok = (std::size_t)st.cap_blocks * 128;
  const std::size_t n_tok = (std::size_t)seq * D;
  // The Sage objects sized for THIS chunk's shape: the opening chunk's
  // three windows, or a later chunk's one.
  MetalSageAttention* sage_self =
      (f > 2 && _sage_open) ? _sage_open.get() : _sage.get();
  MetalSageAttention* sage_cross =
      (f > 2 && _sage_x_open) ? _sage_x_open.get() : _sage_x.get();
  if (cache_idx < 0 || cache_idx >= (int)st.per_block.size()) {
    return fail("no key/value window for this block");
  }
  if (!rope_(st, rope_slot, f, t_off)) { return fail("rope allocation"); }
  const Stream::Rope& rp = st.rope[(std::size_t)rope_slot];
  FwdCtx& cx = st.cx;
  SharedBuffer& qm_buf = st.qm_buf;
  SharedBuffer& km_buf = st.km_buf;
  SharedBuffer& params = st.params;
  const bool prof = st.prof;
  double& t_qkv = st.t_qkv;
  double& t_prep = st.t_prep;
  double& t_route = st.t_route;
  double& t_attn = st.t_attn;
  double& t_oproj = st.t_oproj;
  double& t_cross = st.t_cross;
  double& t_ffn = st.t_ffn;
  double& t_elt = st.t_elt;
  int& kL_last = st.kL_last;
  st.seq_last = seq;

  // ---- CROSS-ATTENTION ON THE FLASH KERNEL.
  //
  // MEASURED, and it is why this exists: with the scalar sdpa it was
  // 43-49% of a block, against 20-22% for the feed-forward that does
  // 35x its arithmetic. One threadgroup of 32 threads per query row,
  // each walking 512 keys, is a shape no dense kernel would choose.
  //
  // Its key length is the FIXED conditioning, so the shape is settled
  // once per chunk rather than per block, and there is no mask -- the
  // context is 512 tokens with nothing to exclude.
  SharedBuffer xparams = _mc->make_shared_buffer(sizeof(SteelAttnParams));
  if (xparams.empty()) { return fail("cross params allocation failed"); }
  {
    auto* pp = static_cast<SteelAttnParams*>(xparams.contents());
    pp->B = 1; pp->H = NH; pp->D = HD;
    pp->qL = seq; pp->kL = T;
    pp->gqa_factor = 1;
    pp->scale = 1.0f / std::sqrt((float)HD);
    pp->NQ = (seq + A_BQ - 1) / A_BQ;
    pp->NK = (T + A_BK - 1) / A_BK;
    pp->NQ_aligned = seq / A_BQ;
    pp->NK_aligned = T / A_BK;
    pp->qL_rem = seq - pp->NQ_aligned * A_BQ;
    pp->kL_rem = T - pp->NK_aligned * A_BK;
    pp->qL_off = 0;
    pp->Q_strides[0] = (std::int64_t)NH * seq * HD;
    pp->Q_strides[1] = (std::int64_t)seq * HD;
    pp->Q_strides[2] = HD;
    pp->K_strides[0] = (std::int64_t)NH * T * HD;
    pp->K_strides[1] = (std::int64_t)T * HD;
    pp->K_strides[2] = HD;
    pp->V_strides[0] = pp->K_strides[0];
    pp->V_strides[1] = pp->K_strides[1];
    pp->V_strides[2] = HD;
    pp->O_strides[0] = pp->Q_strides[0];
    pp->O_strides[1] = pp->Q_strides[1];
    pp->O_strides[2] = HD;
  }
  metal_compute::ComputeFunction fn_xattn;
  {
    metal_compute::FunctionConstants xfc;
    xfc.set_bool(200, (seq % A_BQ) == 0).set_bool(201, (T % A_BK) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false);
    fn_xattn = attn_fn_(xfc);
  }
  const bool use_steel_cross =
      fn_xattn.valid() && std::getenv("VPIPE_FVSR_SCALAR_CROSS") == nullptr;
  // ...and its int8-QK twin, for Sage. The context's key length is fixed,
  // so like the dense one it is settled once per chunk.
  metal_compute::ComputeFunction fn_xattn8;
  if (sage_cross != nullptr && use_steel_cross) {
    metal_compute::FunctionConstants xfc8;
    xfc8.set_bool(200, (seq % A_BQ) == 0).set_bool(201, (T % A_BK) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(sage::kQkInt8Constant, true);
    fn_xattn8 = attn_fn_(xfc8);
  }


  // The WEIGHTS from a slot when streaming; what the load FOLDED --
  // modulation, cross key and value -- always from the resident block.
  const Block& fb = _blocks[(std::size_t)bi];
  const Block& b = wb != nullptr ? *wb : fb;
  auto& cache = st.per_block[(std::size_t)cache_idx];
  // THE WINDOW IS A RING of whole temporal windows. This chunk's keys
  // go into the slot after the newest -- once the ring is full, the one
  // the last trim released -- so nothing is ever shifted, and every
  // attention still reads a VALID PREFIX: the ring is either filling
  // from slot 0 or entirely full. The ORDER of the keys is free to
  // change because nothing reads it: they are cached after RoPE at
  // their absolute temporal positions, the routing thresholds a group
  // rather than ranking positions, and the span list is ordered by
  // slot, which is all the kernel asks.
  const int nwin = nb / st.one_len;              // 3 opening, 1 steady
  const int slot = (cache.oldest + cache.valid) % st.cap_windows;
  if (nb % st.one_len != 0 || slot + nwin > st.cap_windows ||
      (cache.oldest != 0 && cache.valid + nwin != st.cap_windows)) {
    return fail("key/value window layout broken");
  }
  const int held = slot * st.one_len * 128;      // where this chunk goes
  const int kL = (cache.valid + nwin) * st.one_len * 128;

  // FIVE SCRATCH TENSORS A BLOCK. One encoder runs its dispatches in
  // the order they were encoded, so a tensor is free for reuse the
  // moment the dispatch that READS its value has been encoded; each
  // hand-off is named where it happens. There were fourteen, 159 MB
  // apiece in the opening chunk at 1920x1152.
  SharedBuffer& y = cx.alloc(_mc, n_tok);
  SharedBuffer& q = cx.alloc(_mc, n_tok);
  SharedBuffer& k = cx.alloc(_mc, n_tok);
  SharedBuffer& v = cx.alloc(_mc, n_tok);
  SharedBuffer& a = cx.alloc(_mc, n_tok);
  if (!cx.alloc_ok) { return fail("block scratch allocation failed"); }

  // ---- params and the attention function are settled on the HOST
  // first: kL is known before anything is dispatched, so nothing here
  // has to wait for the GPU.
  const int Nk = kL / 128;
  const int NQ = (seq + A_BQ - 1) / A_BQ;
  {
    auto* pp = static_cast<SteelAttnParams*>(params.contents());
    pp->B = 1; pp->H = NH; pp->D = HD;
    pp->qL = seq; pp->kL = kL;
    pp->gqa_factor = 1;
    pp->scale = 1.0f / std::sqrt((float)HD);
    pp->NQ = NQ;
    pp->NK = (kL + A_BK - 1) / A_BK;
    pp->NQ_aligned = seq / A_BQ;
    pp->NK_aligned = kL / A_BK;
    pp->qL_rem = seq - pp->NQ_aligned * A_BQ;
    pp->kL_rem = kL - pp->NK_aligned * A_BK;
    pp->qL_off = 0;
    pp->Q_strides[0] = (std::int64_t)NH * seq * HD;
    pp->Q_strides[1] = (std::int64_t)seq * HD;
    pp->Q_strides[2] = HD;
    // The ALLOCATION stride, not kL: the cache is one buffer that a
    // growing logical length lives inside.
    pp->K_strides[0] = (std::int64_t)NH * (std::int64_t)cap_tok * HD;
    pp->K_strides[1] = (std::int64_t)cap_tok * HD;
    pp->K_strides[2] = HD;
    pp->V_strides[0] = pp->K_strides[0];
    pp->V_strides[1] = pp->K_strides[1];
    pp->V_strides[2] = HD;
    pp->O_strides[0] = pp->Q_strides[0];
    pp->O_strides[1] = pp->Q_strides[1];
    pp->O_strides[2] = HD;
  }
  metal_compute::FunctionConstants fc;
  fc.set_bool(200, (seq % A_BQ) == 0).set_bool(201, (kL % A_BK) == 0)
      .set_bool(300, false).set_bool(301, false).set_bool(302, false);
  // SPANS SKIP; THE MASK ONLY MASKS. Both say the same thing, and only
  // one of them costs less than dense -- see flashvsr_route.metal.
  if (_use_spans) { fc.set_bool(303, true); }
  else            { fc.set_bool(305, true); }
  metal_compute::ComputeFunction fn_attn =
      attn_fn_(fc);
  if (!fn_attn.valid()) { return fail("sparse steel attention unavailable"); }
  // SAGE OVER THE ROUTED KEYS. The smoothing it applies to K stays exact
  // under the routing: subtracting the key mean shifts every score in a
  // row by the same <q, mean>, and a softmax over a SUBSET of a row's
  // keys is as blind to that shift as one over all of them.
  const bool sage_here =
      sage_self != nullptr && _cfg.sage.enabled &&
      bi >= _cfg.sage.dense_layers;
  metal_compute::ComputeFunction fn_attn8;
  if (sage_here) {
    metal_compute::FunctionConstants fc8;
    fc8.set_bool(200, (seq % A_BQ) == 0).set_bool(201, (kL % A_BK) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(_use_spans ? 303 : 305, true)
        .set_bool(sage::kQkInt8Constant, true);
    fn_attn8 = attn_fn_(fc8);
  }
  if (_use_spans) {
    static_cast<AttnSpanParams*>(st.sp_params.contents())->qb_stride =
        NQ + 1;
  }

  // ---- ONE command stream for the whole block. The routing used to
  // sit in the middle of this as a readback, which split it in two and
  // drained the queue thirty times a chunk for arithmetic measured in
  // kilobytes.
  {
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    cx.enc = &enc;
    std::chrono::steady_clock::time_point mark =
        std::chrono::steady_clock::now();
    auto psplit = [&](double& acc) {
      if (!prof) { return; }
      enc.end();
      stream.commit().wait();
      acc += std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - mark).count();
      stream = _mc->make_command_stream();
      enc = stream.begin_compute();
      mark = std::chrono::steady_clock::now();
    };
    // The source projection enters the residual stream here. v1.1
    // ships one output layer, so this fires at block 0 alone.
    // THE SOURCE ENTERS AT BLOCK 0. v1.1's projection ships one
    // output layer, so one row set reaches one block; a checkpoint
    // with more would emit more row sets and the conditioner would
    // say so on the beat.
    if (lq != nullptr && bi == 0) {
      enc.set_function(_fn_resadd);
      enc.set_buffer(0, x); enc.set_buffer(1, *lq); enc.set_buffer(2, x);
      enc.set_constant(3, (int)n_tok);
      enc.dispatch({(unsigned)n_tok, 1, 1}, {256, 1, 1});
    }
    enc.set_function(_fn_ln_mod);
    enc.set_buffer(0, x);
    enc.set_buffer(1, fb.mod, (std::size_t)1 * D * 2);
    enc.set_buffer(2, fb.mod, 0);
    enc.set_buffer(3, y);
    enc.set_constant(4, D); enc.set_constant(5, _cfg.norm_eps);
    enc.dispatch({256, (unsigned)seq, 1}, {256, 1, 1});

    gemm_(enc, y, b.q1, b.q1b, q, seq, D, D);
    gemm_(enc, y, b.k1, b.k1b, k, seq, D, D);
    gemm_(enc, y, b.v1, b.v1b, v, seq, D, D);
    rms_(enc, q, b.qn1, q, seq, D);
    rms_(enc, k, b.kn1, k, seq, D);
    psplit(t_qkv);
    // Each window-permuted through `a` and back: q becomes the RoPE'd
    // head-major query, k and v the chunk's head-major keys and values.
    permute_(enc, q, a, f, h, w, D, false);
    trope_(enc, a, q, rp.cos, rp.sin, NH, seq, HD);
    permute_(enc, k, a, f, h, w, D, false);
    trope_(enc, a, k, rp.cos, rp.sin, NH, seq, HD);
    permute_(enc, v, a, f, h, w, D, false);
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, a); enc.set_buffer(1, v);
    enc.set_constant(2, seq); enc.set_constant(3, NH);
    enc.set_constant(4, HD);
    enc.dispatch({(unsigned)HD, (unsigned)NH, (unsigned)seq},
                 {(unsigned)HD, 1, 1});
    append_cache_(enc, k, cache.k, NH, seq, HD, cap_tok, held);
    append_cache_(enc, v, cache.v, NH, seq, HD, cap_tok, held);

    psplit(t_prep);
    blockmean_(enc, q, qm_buf, HD, seq, 128, NH, nb);
    blockmean_(enc, cache.k, km_buf, HD, (int)cap_tok, 128, NH, Nk);
    route_gpu_(enc, qm_buf, km_buf, st.band, st.p, st.thr, st.flags, nb, Nk,
               sq, S, topk, kL, NQ, A_BQ);
    if (_use_spans) {
      build_spans_(enc, st, nb, Nk, sq, NQ, A_BQ);
    }
    // VPIPE_FVSR_DENSE_ATTN: keep every key block, to measure what the
    // sparsity actually BUYS in time. has_block_mask is a per-key
    // flag, not a span list -- so the question is whether the kernel
    // skips the excluded keys or merely zeroes them, and only the
    // clock can answer it.
    if (_dense_attn && !st.flags.empty()) {
      std::memset(st.flags.contents(), 0,
                  (std::size_t)NH * NQ * kL);
    }
    psplit(t_route);

    // ---- the attention and the rest of the block, same stream.
    SharedBuffer& ff = cx.alloc(_mc, (std::size_t)seq * FF);
    if (!cx.alloc_ok) { return fail("block scratch allocation failed"); }

    bool sage_ok = false;
    if (sage_here && fn_attn8.valid()) {
      // q is tight head-major; the cache is head-major at its ALLOCATION
      // stride, so the head stride is passed and never derived from kL.
      const MetalSageAttention::Operand qo{&q, 0, HD, seq * HD};
      const MetalSageAttention::Operand ko{&cache.k, 0, HD,
                                           (int)cap_tok * HD};
      std::string serr;
      sage_ok = sage_self->prepare(enc, qo, ko, NH, NH, seq, kL, HD, A_BQ,
                               A_BK, _cfg.sage, &serr);
    }
    enc.set_function(sage_ok ? fn_attn8 : fn_attn);
    enc.set_buffer(0, q); enc.set_buffer(1, cache.k);
    enc.set_buffer(2, cache.v);
    enc.set_buffer(3, a); enc.set_buffer(4, params);
    if (_use_spans) {
      enc.set_buffer(8, st.qb_off); enc.set_buffer(9, st.qb_blocks);
      enc.set_buffer(10, st.sp_params); enc.set_buffer(11, st.sp_bounds);
    } else {
      enc.set_buffer(14, st.flags);
    }
    if (sage_ok) { sage_self->bind(enc); }
    enc.dispatch({32u * (unsigned)((seq + A_BQ - 1) / A_BQ),
                  4u * (unsigned)NH, 1}, {32, 4, 1});

    psplit(t_attn);
    // Attention (a) -> token-major into v, whose values are in the ring
    // now -> un-permuted back into a -> the output projection into k,
    // whose keys are in the ring too.
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, a); enc.set_buffer(1, v);
    enc.set_constant(2, NH); enc.set_constant(3, seq);
    enc.set_constant(4, HD);
    enc.dispatch({(unsigned)HD, (unsigned)seq, (unsigned)NH},
                 {(unsigned)HD, 1, 1});
    permute_(enc, v, a, f, h, w, D, true);
    gemm_(enc, a, b.o1, b.o1b, k, seq, D, D);

    psplit(t_oproj);
    // x = x + gate_msa * attn
    enc.set_function(_fn_gated);
    enc.set_buffer(0, x); enc.set_buffer(1, fb.mod, (std::size_t)2 * D * 2);
    enc.set_buffer(2, k);
    enc.set_constant(3, D); enc.set_constant(4, (int)n_tok);
    enc.dispatch({(unsigned)n_tok, 1, 1}, {256, 1, 1});

    // Cross-attention into the FOLDED constant: a query projection, an
    // attention, an output projection. No key or value work at all.
    // norm3 (y) -> query (q, its self-attention reader encoded) ->
    // head-major (a) -> attention (v) -> token-major (q) -> o-proj (k).
    enc.set_function(_fn_ln_aff);
    enc.set_buffer(0, x); enc.set_buffer(1, b.n3w); enc.set_buffer(2, b.n3b);
    enc.set_buffer(3, y);
    enc.set_constant(4, D); enc.set_constant(5, _cfg.norm_eps);
    enc.dispatch({256, (unsigned)seq, 1}, {256, 1, 1});
    gemm_(enc, y, b.q2, b.q2b, q, seq, D, D);
    rms_(enc, q, b.qn2, q, seq, D);
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, q); enc.set_buffer(1, a);
    enc.set_constant(2, seq); enc.set_constant(3, NH);
    enc.set_constant(4, HD);
    enc.dispatch({(unsigned)HD, (unsigned)NH, (unsigned)seq},
                 {(unsigned)HD, 1, 1});
    if (use_steel_cross) {
      bool xsage = false;
      if (fn_xattn8.valid() && bi >= _cfg.sage.dense_layers) {
        const MetalSageAttention::Operand qo{&a, 0, HD, seq * HD};
        const MetalSageAttention::Operand ko{&fb.ck, 0, HD, T * HD};
        std::string serr;
        xsage = sage_cross->prepare(enc, qo, ko, NH, NH, seq, T, HD, A_BQ,
                                 A_BK, _cfg.sage, &serr);
      }
      enc.set_function(xsage ? fn_xattn8 : fn_xattn);
      enc.set_buffer(0, a); enc.set_buffer(1, fb.ck);
      enc.set_buffer(2, fb.cv);
      enc.set_buffer(3, v); enc.set_buffer(4, xparams);
      if (xsage) { sage_cross->bind(enc); }
      enc.dispatch({32u * (unsigned)((seq + A_BQ - 1) / A_BQ),
                    4u * (unsigned)NH, 1}, {32, 4, 1});
    } else {
      const float scale = 1.0f / std::sqrt((float)HD);
      enc.set_function(_fn_sdpa);
      enc.set_buffer(0, a); enc.set_buffer(1, fb.ck);
      enc.set_buffer(2, fb.cv);
      enc.set_buffer(3, v);
      enc.set_constant(4, scale); enc.set_constant(5, T);
      enc.set_constant(6, HD); enc.set_constant(7, NH);
      enc.set_constant(8, NH);
      enc.set_constant(9, seq); enc.set_constant(10, T);
      enc.dispatch({32, (unsigned)NH, (unsigned)seq}, {32, 1, 1});
    }
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, v); enc.set_buffer(1, q);
    enc.set_constant(2, NH); enc.set_constant(3, seq);
    enc.set_constant(4, HD);
    enc.dispatch({(unsigned)HD, (unsigned)seq, (unsigned)NH},
                 {(unsigned)HD, 1, 1});
    gemm_(enc, q, b.o2, b.o2b, k, seq, D, D);
    enc.set_function(_fn_resadd);
    enc.set_buffer(0, x); enc.set_buffer(1, k); enc.set_buffer(2, x);
    enc.set_constant(3, (int)n_tok);
    enc.dispatch({(unsigned)n_tok, 1, 1}, {256, 1, 1});

    psplit(t_cross);
    // The feed-forward is UNGATED: one gelu between two linears.
    enc.set_function(_fn_ln_mod);
    enc.set_buffer(0, x);
    enc.set_buffer(1, fb.mod, (std::size_t)4 * D * 2);
    enc.set_buffer(2, fb.mod, (std::size_t)3 * D * 2);
    enc.set_buffer(3, y);
    enc.set_constant(4, D); enc.set_constant(5, _cfg.norm_eps);
    enc.dispatch({256, (unsigned)seq, 1}, {256, 1, 1});
    gemm_(enc, y, b.ff_in, b.ff_in_b, ff, seq, FF, D);
    enc.set_function(_fn_gelu);
    enc.set_buffer(0, ff); enc.set_buffer(1, ff);
    enc.set_constant(2, (int)((std::size_t)seq * FF));
    enc.dispatch({(unsigned)((std::size_t)seq * FF), 1, 1}, {256, 1, 1});
    gemm_(enc, ff, b.ff_out, b.ff_out_b, k, seq, D, FF);
    psplit(t_ffn);
    enc.set_function(_fn_gated);
    enc.set_buffer(0, x); enc.set_buffer(1, fb.mod, (std::size_t)5 * D * 2);
    enc.set_buffer(2, k);
    enc.set_constant(3, D); enc.set_constant(4, (int)n_tok);
    enc.dispatch({(unsigned)n_tok, 1, 1}, {256, 1, 1});
    psplit(t_elt);
    enc.end();
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) {
      return fail(fmt("block {} failed: {}", bi, ge)());
    }
  }
  // QUERY TILES THE ROUTING LEFT WITH NO KEY BLOCK AT ALL. The span
  // kernel's softmax over nothing is 0/0, so each one is a NaN row that
  // every later block's attention spreads. Counted from the CSR the
  // block just used -- a few thousand ints, read after its stream has
  // finished -- so a clip that goes non-finite can say whether this is
  // why.
  if (_use_spans && !st.qb_off.empty()) {
    const int NQs = (seq + A_BQ - 1) / A_BQ;
    const auto* off = (const int*)st.qb_off.contents();
    for (int hh = 0; hh < NH; ++hh) {
      for (int qb = 0; qb < NQs; ++qb) {
        const int i0 = hh * (NQs + 1) + qb;
        if (off[i0 + 1] == off[i0]) { ++_empty_tiles; }
      }
    }
  }

  // ---- the ring now holds this chunk too; trim to kv_ratio windows.
  kL_last = kL;
  cache.valid += nwin;
  trim_cache_(st, cache_idx, prm);
  cx.release_all();
  return true;
}

// The head: the same modulate, then one projection to the patch.
bool
MetalFlashVsrTransformer::head_(Stream& st, const SharedBuffer& x, int seq,
                                SharedBuffer* noise_out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int D = _cfg.hidden;
  const int out_w = _cfg.out_channels * _cfg.patch_t * _cfg.patch_h *
                    _cfg.patch_w;
  const std::size_t n_tok = (std::size_t)seq * D;
  FwdCtx& cx = st.cx;
  SharedBuffer& yh = cx.alloc(_mc, n_tok);
  if (noise_out->empty()) {
    *noise_out = _mc->make_shared_buffer((std::size_t)seq * out_w * 2);
  }
  if (!cx.alloc_ok || noise_out->empty()) {
    return fail("head allocation failed");
  }
  CommandStream stream = _mc->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  cx.enc = &enc;
  enc.set_function(_fn_ln_mod);
  enc.set_buffer(0, x);
  enc.set_buffer(1, _head_mod, (std::size_t)1 * D * 2);
  enc.set_buffer(2, _head_mod, 0);
  enc.set_buffer(3, yh);
  enc.set_constant(4, D); enc.set_constant(5, _cfg.norm_eps);
  enc.dispatch({256, (unsigned)seq, 1}, {256, 1, 1});
  gemm_(enc, yh, _head_w, _head_b, *noise_out, seq, out_w, D);
  enc.end();
  std::string ge;
  const bool ok = stream.commit().wait_ok(&ge);
  cx.release_all();
  if (!ok) { return fail("head failed: " + ge); }
  return true;
}

// VPIPE_FVSR_DIT_PROFILE: the per-section split since the last report,
// then reset. Read the SHARE, not the total -- see block_.
void
MetalFlashVsrTransformer::report_prof_(Stream& st, const char* what,
                                       int n_blocks)
{
  if (!st.prof) { return; }
  if (_use_spans && !st.qb_off.empty() && st.seq_last > 0) {
    // What the spans actually SHORTENED. If the average list is close to
    // the dense block count, the routing is keeping everything and the
    // sparsity is notional.
    const int A_BQ = attn_bq_(), A_BK = attn_bk_();
    const int NQ = (st.seq_last + A_BQ - 1) / A_BQ;
    const int kb_lim = (st.kL_last + A_BK - 1) / A_BK;
    const auto* off = (const int*)st.qb_off.contents();
    double sum = 0.0;
    for (int hh = 0; hh < _cfg.n_heads; ++hh) {
      for (int qb = 0; qb < NQ; ++qb) {
        const int i0 = hh * (NQ + 1) + qb;
        sum += (double)(off[i0 + 1] - off[i0]);
      }
    }
    const double avg = sum / ((double)_cfg.n_heads * NQ);
    std::fprintf(stderr,
                 "[fvsr-span] avg %.1f of %d key blocks visited (%.0f%%)\n",
                 avg, kb_lim, 100.0 * avg / (double)kb_lim);
  }
  const double tot = st.t_qkv + st.t_prep + st.t_route + st.t_attn +
                     st.t_oproj + st.t_cross + st.t_ffn + st.t_elt;
  if (tot > 0.0) {
    std::fprintf(stderr,
                 "[fvsr-prof] %s %d blocks %.0f ms: qkv %.0f%% prep "
                 "%.0f%% route %.0f%% attn %.0f%% oproj %.0f%% cross %.0f%% "
                 "ffn %.0f%% elt %.0f%%\n", what, n_blocks, tot,
                 100 * st.t_qkv / tot, 100 * st.t_prep / tot,
                 100 * st.t_route / tot, 100 * st.t_attn / tot,
                 100 * st.t_oproj / tot, 100 * st.t_cross / tot,
                 100 * st.t_ffn / tot, 100 * st.t_elt / tot);
  }
  st.t_qkv = st.t_prep = st.t_route = st.t_attn = st.t_oproj = st.t_cross =
      st.t_ffn = st.t_elt = 0.0;
}

// One chunk through the whole stack, then the head: the CHUNK-OUTER
// order, which holds a key/value window per block for the whole clip.
// generate() takes it only where it would hold less than the block-outer
// one (see blocks_outer), and the test seams take it always.
bool
MetalFlashVsrTransformer::forward_chunk_(Stream& st, int f, int h, int w,
                                         int t_off, int chunk_idx,
                                         const SharedBuffer& x_in,
                                         const SharedBuffer* lq,
                                         const Params& prm,
                                         SharedBuffer* noise_out,
                                         const Request& req, int max_blocks,
                                         std::vector<float>* x_out,
                                         std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int D = _cfg.hidden;
  const int seq = f * h * w;
  const std::size_t n_tok = (std::size_t)seq * D;

  // The int8 split's width, for shapes an earlier chunk recorded. Here
  // because it runs its own command streams and so needs no encoder open;
  // the opening chunk's shapes are three times a later one's, so each is
  // tuned at the next chunk that follows it.
  if (_i8) { _i8->tune_pending(_mc); }

  // A seam's lone chunk: a window per block at the ratio's capacity.
  if (st.per_block.empty() &&
      !begin_clip_(st, h, w, f, kv_windows(prm.kv_ratio), _cfg.n_layers, prm,
                   err)) {
    return false;
  }
  if (st.h != h || st.w != w ||
      (int)st.per_block.size() != _cfg.n_layers) {
    return fail("the chunk-outer order needs a window per block");
  }

  SharedBuffer x = _mc->make_shared_buffer(n_tok * 2);
  if (x.empty()) { return fail("residual stream allocation failed"); }
  std::memcpy(x.contents(), x_in.contents(), n_tok * 2);

  const int n_run = (max_blocks > 0 && max_blocks < _cfg.n_layers)
                        ? max_blocks : _cfg.n_layers;
  for (int bi = 0; bi < n_run; ++bi) {
    const Block* wb = nullptr;
    if (!acquire_block_(bi, (bi + 1) % _cfg.n_layers, &wb, err)) {
      return false;
    }
    // The source projection enters the residual stream at block 0.
    if (!block_(st, bi, wb, bi, f, t_off, 0, x, bi == 0 ? lq : nullptr, prm,
                err)) {
      if (_streaming) { _streaming->slots.join(); }
      return false;
    }
    if (req.block_progress && !req.block_progress(bi + 1, _cfg.n_layers)) {
      return fail("aborted");
    }
  }

  // A caller asking for fewer blocks than the stack wants the residual
  // stream, not a noise prediction the head has no business making.
  if (x_out != nullptr) {
    x_out->assign(n_tok, 0.0f);
    const auto* p = (const std::uint16_t*)x.contents();
    for (std::size_t i = 0; i < n_tok; ++i) { (*x_out)[i] = from_bf16(p[i]); }
  }
  if (n_run < _cfg.n_layers) { return true; }
  const std::string what = fmt("chunk {}", chunk_idx)();
  report_prof_(st, what.c_str(), n_run);
  return head_(st, x, seq, noise_out, err);
}


// Drop the oldest window when the cache is over its limit. The reference
// drops exactly ONE window per chunk, never more, and the count it
// compares is in TEMPORAL WINDOWS rather than blocks or tokens.
void
MetalFlashVsrTransformer::trim_cache_(Stream& st, int bi, const Params& prm)
{
  Stream::Cache& cache = st.per_block[(std::size_t)bi];
  const int keep = prm.kv_ratio > 0.0 ? (int)(prm.kv_ratio + 0.5) : 0;
  if (st.cap_windows <= 0 || cache.valid <= keep) { return; }
  // The oldest window's slot is RELEASED, not emptied: the next chunk
  // writes there. Nothing moves, so there is no GPU work and no spare.
  cache.oldest = (cache.oldest + 1) % st.cap_windows;
  --cache.valid;
}

// Copy a contiguous [heads, seq, head_dim] block into a cache slab of a
// different stride, at token offset `held`.
void
MetalFlashVsrTransformer::append_cache_(ComputeEncoder& enc,
                                        const SharedBuffer& src,
                                        const SharedBuffer& dst, int NH,
                                        int seq, int HD, std::size_t cap_tok,
                                        int held)
{
  const std::size_t n = (std::size_t)seq * HD;
  for (int hh = 0; hh < NH; ++hh) {
    enc.set_function(_fn_copy);
    enc.set_buffer(0, src, (std::size_t)hh * n * 2);
    enc.set_buffer(1, dst,
                   ((std::size_t)hh * cap_tok * HD +
                    (std::size_t)held * HD) * 2);
    enc.set_constant(2, 0);
    enc.set_constant(3, (int)n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  }
}

bool
MetalFlashVsrTransformer::run_chunk_blocks(int n_blocks, const float* tokens,
                                           const float* lq_rows, int f, int h,
                                           int w, int t_off,
                                           const Params& prm,
                                           std::vector<float>* out,
                                           std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (tokens == nullptr || out == nullptr) { return fail("no tokens"); }
  if (!ensure_kernels_(err)) { return false; }
  const int D = _cfg.hidden;
  const std::size_t n_tok = (std::size_t)f * h * w * D;

  SharedBuffer x = _mc->make_shared_buffer(n_tok * 2);
  if (x.empty()) { return fail("token upload allocation"); }
  {
    auto* p = (std::uint16_t*)x.contents();
    for (std::size_t i = 0; i < n_tok; ++i) { p[i] = to_bf16(tokens[i]); }
  }
  SharedBuffer lq;
  if (lq_rows != nullptr) {
    lq = _mc->make_shared_buffer(n_tok * 2);
    if (lq.empty()) { return fail("source row allocation"); }
    auto* p = (std::uint16_t*)lq.contents();
    for (std::size_t i = 0; i < n_tok; ++i) { p[i] = to_bf16(lq_rows[i]); }
  }
  Stream st;
  SharedBuffer noise;
  Request req;
  return forward_chunk_(st, f, h, w, t_off, 0, x,
                        lq_rows != nullptr ? &lq : nullptr, prm, &noise, req,
                        n_blocks, out, err);
}

// The always-exact spatial band, over the (nh, nw) block grid.
//
// The reference's slide is UNCLAMPED on purpose: a block near an edge
// simply has fewer neighbours, which is not the same as sliding the
// window back inside the grid. Built on the host because it depends only
// on the geometry and the knob, so it is built once per clip.
void
MetalFlashVsrTransformer::build_band_(int nh_blk, int nw_blk, int local_range,
                                      SharedBuffer& out) const
{
  const int S = nh_blk * nw_blk;
  auto* b = (std::uint8_t*)out.contents();
  const int half = local_range / 2;
  for (int i = 0; i < S; ++i) {
    const int ry = i / nw_blk, rx = i % nw_blk;
    for (int j = 0; j < S; ++j) {
      const int cy = j / nw_blk, cx = j % nw_blk;
      const bool in = (cy >= ry - half) && (cy <= ry - half + local_range - 1) &&
                      (cx >= rx - half) && (cx <= rx - half + local_range - 1);
      b[(std::size_t)i * S + j] = in ? 1u : 0u;
    }
  }
}

// The routing, entirely on the GPU: scores, softmax, threshold, flags.
//
// WHAT THIS REPLACED. The same decision ran on the host off the block
// means, which is correct and was verified end to end -- but it meant
// reading those means back once per block, thirty times a chunk, and a
// round trip that computes almost nothing still drains the queue. The
// arithmetic was never the cost; the stall was.
//
// route_() is kept beside this as the reference implementation, and a
// test holds the two to byte equality -- which is reachable because the
// threshold is found exactly, by bisection on the float bits, rather
// than approximated.
void
MetalFlashVsrTransformer::route_gpu_(ComputeEncoder& enc,
                                     const SharedBuffer& qm,
                                     const SharedBuffer& km,
                                     const SharedBuffer& band,
                                     const SharedBuffer& p,
                                     const SharedBuffer& thr,
                                     const SharedBuffer& flags, int Nq,
                                     int Nk, int sq, int S, int topk, int kL,
                                     int NQ, int bq)
{
  const int H = _cfg.n_heads, HD = _cfg.head_dim;
  const int s1 = (sq > 0) ? Nq / sq : Nq;
  const float scale = 1.0f / std::sqrt((float)HD);
  const int apply = std::min(s1 * Nk - 1, topk);

  enc.set_function(_fn_rt_scores);
  enc.set_buffer(0, qm); enc.set_buffer(1, km); enc.set_buffer(2, band);
  enc.set_buffer(3, p);
  enc.set_constant(4, HD); enc.set_constant(5, Nq); enc.set_constant(6, Nk);
  enc.set_constant(7, S); enc.set_constant(8, scale);
  enc.dispatch({(unsigned)Nk, (unsigned)Nq, (unsigned)H}, {32, 1, 1});

  enc.set_function(_fn_rt_softmax);
  enc.set_buffer(0, p);
  enc.set_constant(1, Nk);
  enc.dispatch({256, (unsigned)(Nq * H), 1}, {256, 1, 1});

  enc.set_function(_fn_rt_thresh);
  enc.set_buffer(0, p); enc.set_buffer(1, thr);
  enc.set_constant(2, Nk); enc.set_constant(3, s1);
  enc.set_constant(4, apply);
  enc.dispatch({256, (unsigned)(sq * H), 1}, {256, 1, 1});

  // The per-key flags are the mask path's; the span path has none.
  if (flags.empty()) { return; }
  enc.set_function(_fn_rt_flags);
  enc.set_buffer(0, p); enc.set_buffer(1, thr); enc.set_buffer(2, flags);
  enc.set_constant(3, Nq); enc.set_constant(4, Nk); enc.set_constant(5, NQ);
  enc.set_constant(6, kL); enc.set_constant(7, bq); enc.set_constant(8, s1);
  enc.dispatch({(unsigned)kL, (unsigned)NQ, (unsigned)H}, {64, 1, 1});
}

// Test seam: the routing decision for one chunk, computed BOTH ways.
//
// The GPU path replaced a host one that was verified end to end, so the
// question is not "does it work" but "does it decide the same thing".
// It cannot be bit-identical -- the host summed its proxy in double and
// reduced its softmax serially, where the GPU does both in float across
// simdgroups -- so what matters is how many blocks that disagreement
// actually flips. This counts them.
//
// The GPU is the one that matches the REFERENCE's precision: PyTorch
// computed these scores in f32 too, so the host's double was the outlier.
bool
MetalFlashVsrTransformer::route_compare(const float* x_tokens, int f, int h,
                                        int w, int t_off, const Params& prm,
                                        std::size_t* n_flags,
                                        std::size_t* n_diff,
                                        std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (x_tokens == nullptr || !ensure_kernels_(err)) {
    return fail("no tokens or kernels");
  }
  const Block& b = _blocks[0];
  const int D = _cfg.hidden, NH = _cfg.n_heads, HD = _cfg.head_dim;
  const int seq = f * h * w, nb = seq / 128;
  const int nh_blk = h / 8, nw_blk = w / 8, S = nh_blk * nw_blk;
  const int sq = f / 2, kL = seq, Nk = nb, A_BQ = 32;
  const int NQ = (seq + A_BQ - 1) / A_BQ;
  const std::size_t n_tok = (std::size_t)seq * D;

  SharedBuffer rcos = _mc->make_shared_buffer((std::size_t)seq * HD * 4);
  SharedBuffer rsin = _mc->make_shared_buffer((std::size_t)seq * HD * 4);
  if (rcos.empty() || rsin.empty()) { return fail("rope allocation"); }
  build_rope_(f, h, w, t_off, rcos, rsin);

  SharedBuffer x = _mc->make_shared_buffer(n_tok * 2);
  if (x.empty()) { return fail("token allocation"); }
  {
    auto* pt = (std::uint16_t*)x.contents();
    for (std::size_t i = 0; i < n_tok; ++i) { pt[i] = to_bf16(x_tokens[i]); }
  }

  FwdCtx cx;
  SharedBuffer qm = _mc->make_shared_buffer((std::size_t)NH * nb * HD * 2);
  SharedBuffer km = _mc->make_shared_buffer((std::size_t)NH * Nk * HD * 2);
  SharedBuffer band = _mc->make_shared_buffer((std::size_t)S * S);
  SharedBuffer pp = _mc->make_shared_buffer((std::size_t)NH * nb * Nk * 4);
  SharedBuffer thr = _mc->make_shared_buffer((std::size_t)NH * sq * 4);
  SharedBuffer flags = _mc->make_shared_buffer((std::size_t)NH * NQ * kL);
  if (qm.empty() || km.empty() || band.empty() || pp.empty() || thr.empty() ||
      flags.empty()) {
    return fail("routing allocation failed");
  }
  build_band_(nh_blk, nw_blk, prm.local_range, band);

  const double topk_ratio =
      prm.sparse_ratio * 768.0 * 1280.0 / ((double)(h * 16) * (w * 16));
  const int topk = (int)((double)(S * S) * topk_ratio) - 1;

  {
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    cx.enc = &enc;
    SharedBuffer& y  = cx.alloc(_mc, n_tok);
    SharedBuffer& q  = cx.alloc(_mc, n_tok);
    SharedBuffer& k  = cx.alloc(_mc, n_tok);
    SharedBuffer& qp = cx.alloc(_mc, n_tok);
    SharedBuffer& kp = cx.alloc(_mc, n_tok);
    SharedBuffer& qh = cx.alloc(_mc, n_tok);
    SharedBuffer& kh = cx.alloc(_mc, n_tok);
    if (!cx.alloc_ok) { return fail("scratch allocation failed"); }
    enc.set_function(_fn_ln_mod);
    enc.set_buffer(0, x);
    enc.set_buffer(1, b.mod, (std::size_t)1 * D * 2);
    enc.set_buffer(2, b.mod, 0);
    enc.set_buffer(3, y);
    enc.set_constant(4, D); enc.set_constant(5, _cfg.norm_eps);
    enc.dispatch({256, (unsigned)seq, 1}, {256, 1, 1});
    gemm_(enc, y, b.q1, b.q1b, q, seq, D, D);
    gemm_(enc, y, b.k1, b.k1b, k, seq, D, D);
    rms_(enc, q, b.qn1, q, seq, D);
    rms_(enc, k, b.kn1, k, seq, D);
    permute_(enc, q, qp, f, h, w, D, false);
    permute_(enc, k, kp, f, h, w, D, false);
    trope_(enc, qp, qh, rcos, rsin, NH, seq, HD);
    trope_(enc, kp, kh, rcos, rsin, NH, seq, HD);
    blockmean_(enc, qh, qm, HD, seq, 128, NH, nb);
    blockmean_(enc, kh, km, HD, kL, 128, NH, Nk);
    route_gpu_(enc, qm, km, band, pp, thr, flags, nb, Nk, sq, S, topk, kL,
               NQ, A_BQ);
    enc.end();
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) { return fail("routing failed: " + ge); }
  }

  std::vector<float> qmf((std::size_t)NH * nb * HD);
  std::vector<float> kmf((std::size_t)NH * Nk * HD);
  {
    const auto* a = (const std::uint16_t*)qm.contents();
    for (std::size_t i = 0; i < qmf.size(); ++i) { qmf[i] = from_bf16(a[i]); }
    const auto* c = (const std::uint16_t*)km.contents();
    for (std::size_t i = 0; i < kmf.size(); ++i) { kmf[i] = from_bf16(c[i]); }
  }
  _seq_for_route = seq;
  std::vector<std::uint8_t> host;
  if (std::getenv("VPIPE_FVSR_NO_HOST_ROUTE") == nullptr) {
    host = route_(qmf, kmf, nb, Nk, sq, nh_blk, nw_blk, topk,
                  prm.local_range, kL, A_BQ);
  } else {
    host.assign((std::size_t)NH * NQ * kL, 0u);
    const auto* gp = (const std::uint8_t*)flags.contents();
    std::copy(gp, gp + host.size(), host.begin());
  }
  const auto* gpu = (const std::uint8_t*)flags.contents();
  std::size_t diff = 0;
  for (std::size_t i = 0; i < host.size(); ++i) {
    if ((gpu[i] != 0u) != (host[i] != 0u)) { ++diff; }
  }
  if (n_flags != nullptr) { *n_flags = host.size(); }
  if (n_diff != nullptr) { *n_diff = diff; }
  return true;
}

// The routing's decision, turned into the CSR list attn_steel walks.
//
// Three passes: the block-level keep bitmap, the offsets (serial in one
// thread -- the array is a couple of thousand entries and has to be
// globally monotonic across heads), then the list itself.
void
MetalFlashVsrTransformer::build_spans_(ComputeEncoder& enc, Stream& st, int Nq,
                                       int Nk, int sq, int NQ, int bq)
{
  const int H = _cfg.n_heads;
  const int s1 = (sq > 0) ? Nq / sq : Nq;
  // The CSR's unit is the READING kernel's key block: 16 keys on the ALU
  // entry, 32 on the NAX one. A list built in the other unit visits the
  // wrong keys and still produces a plausible frame.
  const int per_k = 128 / attn_bk_();  // routing block over the entry's BK
  const int per_q = 128 / bq;    // routing block over its BQ

  enc.set_function(_fn_rt_keep);
  enc.set_buffer(0, st.p); enc.set_buffer(1, st.thr); enc.set_buffer(2, st.keep);
  enc.set_constant(3, Nq); enc.set_constant(4, Nk); enc.set_constant(5, s1);
  enc.dispatch({(unsigned)Nk, (unsigned)Nq, (unsigned)H}, {32, 1, 1});

  enc.set_function(_fn_sp_off);
  enc.set_buffer(0, st.keep); enc.set_buffer(1, st.qb_off);
  enc.set_constant(2, Nq); enc.set_constant(3, Nk); enc.set_constant(4, NQ);
  enc.set_constant(5, H); enc.set_constant(6, per_k);
  enc.set_constant(7, per_q);
  enc.dispatch({1, 1, 1}, {1, 1, 1});

  enc.set_function(_fn_sp_emit);
  enc.set_buffer(0, st.keep); enc.set_buffer(1, st.qb_off);
  enc.set_buffer(2, st.qb_blocks);
  enc.set_constant(3, Nq); enc.set_constant(4, Nk); enc.set_constant(5, NQ);
  enc.set_constant(6, per_k); enc.set_constant(7, per_q);
  enc.dispatch({(unsigned)NQ, (unsigned)H, 1}, {64, 1, 1});
}

// ------------------------------------------------------ the clip loop

bool
MetalFlashVsrTransformer::generate(const Request& req,
                                   std::vector<float>* latent,
                                   std::vector<int>* shape, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (latent == nullptr || shape == nullptr) { return fail("no output"); }
  if (req.rows == nullptr || req.row_frames <= 0) {
    return fail("flashvsr conditions on the source rows and nothing else; "
                "none were supplied");
  }
  if (req.height % kSizeGrid != 0 || req.width % kSizeGrid != 0) {
    return fail(fmt("output {}x{} must be a multiple of {}", req.width,
                    req.height, kSizeGrid)());
  }
  if (align_frames(req.frames) != req.frames) {
    return fail("frame count must be 8k+1");
  }
  if (!ensure_kernels_(err)) { return false; }

  const int C = _cfg.out_channels, D = _cfg.hidden;
  const int h8 = req.height / 8, w8 = req.width / 8;
  const int th = h8 / 2, tw = w8 / 2;
  const int chunks = (req.frames - 1) / 8 - 2;
  if (chunks <= 0) {
    return fail(fmt("a clip of {} frames is all warmup; {} are needed before "
                    "anything is restored", req.frames, 8 * 3 + 1)());
  }
  const int T = latent_frames(req.frames);
  const std::size_t plane = (std::size_t)h8 * w8;

  // THE NOISE IS AN INPUT when the caller supplies one. Two runtimes
  // cannot agree on a pseudo-random field, so a comparison against the
  // reference has to hold it fixed; a real run draws its own.
  std::vector<float> noise((std::size_t)C * T * plane, 0.0f);
  if (req.init_noise != nullptr) {
    std::memcpy(noise.data(), req.init_noise, noise.size() * sizeof(float));
  } else {
    std::mt19937_64 rng(req.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : noise) { v = nd(rng); }
  }

  latent->assign((std::size_t)C * T * plane, 0.0f);
  *shape = {C, T, h8, w8};

  Stream st;
  _empty_tiles = 0;

  // The source rows arrive ALREADY PROJECTED, one row frame per four
  // source frames, as the conditioner emitted them. Slicing them is all
  // that is left here -- see the class comment for why the projection
  // moved out.
  const std::size_t tok_per_frame = (std::size_t)th * tw;
  const int want_rf = T;
  if (req.row_frames < want_rf) {
    return fail(fmt("source rows cover {} frames; this clip needs {}",
                    req.row_frames, want_rf)());
  }

  if (_streaming) { _streaming->slots.begin_forward(); }
  const bool outer = blocks_outer(_cfg, req.height, req.width, req.frames,
                                  req.params);
  if (!begin_clip_(st, th, tw, 6, ring_windows(req.frames,
                                               req.params.kv_ratio),
                   outer ? 1 : _cfg.n_layers, req.params, err)) {
    return false;
  }
  if (outer) { return generate_blocks_outer_(req, noise, st, latent, err); }

  for (int ci = 0; ci < chunks; ++ci) {
    const int f = (ci == 0) ? 6 : 2;
    const int t_off = (ci == 0) ? 0 : (4 + ci * 2);
    const int t_first = t_off;
    const int seq = f * th * tw;

    // The chunk's own slice of the rows. Row frame index IS the latent
    // frame index, which is what makes this a slice rather than a
    // schedule.
    SharedBuffer lq_all = _mc->make_shared_buffer((std::size_t)seq * D * 2);
    if (lq_all.empty()) { return fail("source row slice failed"); }
    std::memcpy(lq_all.contents(),
                (const std::uint8_t*)req.rows +
                    (std::size_t)t_first * tok_per_frame * D * 2,
                (std::size_t)seq * D * 2);

    // ---- the noise slice this chunk denoises, patchified.
    std::vector<float> slice((std::size_t)C * f * plane, 0.0f);
    for (int c = 0; c < C; ++c) {
      for (int t = 0; t < f; ++t) {
        std::memcpy(slice.data() + ((std::size_t)c * f + t) * plane,
                    noise.data() + ((std::size_t)c * T + (t_first + t)) * plane,
                    plane * sizeof(float));
      }
    }
    std::vector<float> tok;
    if (!patchify_tokens(slice.data(), f, h8, w8, &tok, err)) { return false; }
    SharedBuffer x = _mc->make_shared_buffer((std::size_t)seq * D * 2);
    if (x.empty()) { return fail("token allocation failed"); }
    {
      auto* p = (std::uint16_t*)x.contents();
      for (std::size_t i = 0; i < tok.size(); ++i) { p[i] = to_bf16(tok[i]); }
    }

    SharedBuffer pred;
    if (!forward_chunk_(st, f, th, tw, t_off, ci, x, &lq_all, req.params,
                        &pred, req, 0, nullptr, err)) {
      return false;
    }

    // ---- un-patchify and take the single step. One step at a fixed
    // timestep, so the update is a subtraction and not a schedule.
    std::vector<float> upd((std::size_t)C * f * plane, 0.0f);
    if (!unpatchify_(pred, f, h8, w8, &upd, err)) { return false; }
    // A NON-FINITE PREDICTION IS REFUSED, not decoded: the VAE turns it
    // into a clip of black frames that looks like a finished run.
    std::size_t bad = 0;
    for (const float u : upd) {
      if (!std::isfinite(u)) { ++bad; }
    }
    if (bad > 0) {
      return fail(fmt("chunk {} of {} predicted {} non-finite values of {}; "
                      "{} query tile(s) so far were routed to no key block "
                      "at all", ci + 1, chunks, bad, upd.size(),
                      _empty_tiles)());
    }
    for (int c = 0; c < C; ++c) {
      for (int t = 0; t < f; ++t) {
        const std::size_t src = ((std::size_t)c * f + t) * plane;
        const std::size_t dst = ((std::size_t)c * T + (t_first + t)) * plane;
        for (std::size_t i = 0; i < plane; ++i) {
          (*latent)[dst + i] = slice[src + i] - upd[src + i];
        }
      }
    }
    if (req.progress && !req.progress(ci + 1, chunks)) {
      return fail("aborted");
    }
  }
  return true;
}

// THE BLOCK-OUTER ORDER: every chunk through block 0, then every chunk
// through block 1, and so on, then the head over each.
//
// Exactly the arithmetic of the chunk-outer loop above, reordered. A
// block's work on a chunk reads two things -- that chunk's residual
// stream, and the same block's key/value window over the chunks before
// it -- and both are identical either way round. What changes is what
// has to stay alive: chunk-outer, each block's window must survive until
// that block's NEXT chunk, so all thirty are held at once (12.7 GB at
// 1920x1152); block-outer, a block's window is done with before the next
// block begins and ONE serves the stack. The price is the residual
// stream and the RoPE tables of every chunk at once instead of one --
// ~35 MB a latent frame at 1920x1152, against 425 MB per window.
bool
MetalFlashVsrTransformer::generate_blocks_outer_(
    const Request& req, const std::vector<float>& noise, Stream& st,
    std::vector<float>* latent, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int C = _cfg.out_channels, D = _cfg.hidden;
  const int h8 = req.height / 8, w8 = req.width / 8;
  const int th = h8 / 2, tw = w8 / 2;
  const int chunks = (req.frames - 1) / 8 - 2;
  const int T = latent_frames(req.frames);
  const std::size_t plane = (std::size_t)h8 * w8;
  const std::size_t tok_per_frame = (std::size_t)th * tw;
  auto chunk_f = [](int ci) { return ci == 0 ? 6 : 2; };
  auto chunk_t = [](int ci) { return ci == 0 ? 0 : 4 + ci * 2; };

  // Every chunk's noise, patchified into the residual stream it starts
  // from. They are all live from here to the head.
  std::vector<SharedBuffer> xs((std::size_t)chunks);
  for (int ci = 0; ci < chunks; ++ci) {
    const int f = chunk_f(ci), t_first = chunk_t(ci);
    std::vector<float> slice((std::size_t)C * f * plane, 0.0f);
    for (int c = 0; c < C; ++c) {
      for (int t = 0; t < f; ++t) {
        std::memcpy(slice.data() + ((std::size_t)c * f + t) * plane,
                    noise.data() + ((std::size_t)c * T + (t_first + t)) * plane,
                    plane * sizeof(float));
      }
    }
    std::vector<float> tok;
    if (!patchify_tokens(slice.data(), f, h8, w8, &tok, err)) { return false; }
    xs[(std::size_t)ci] = _mc->make_shared_buffer(tok.size() * 2);
    if (xs[(std::size_t)ci].empty()) { return fail("token allocation failed"); }
    auto* p = (std::uint16_t*)xs[(std::size_t)ci].contents();
    for (std::size_t i = 0; i < tok.size(); ++i) { p[i] = to_bf16(tok[i]); }
  }

  // The bar's unit is a BLOCK here -- a step of the stack over the whole
  // clip -- and its sub-unit a chunk.
  for (int bi = 0; bi < _cfg.n_layers; ++bi) {
    // A fresh window for this block: the previous block's is finished.
    Stream::Cache& cache = st.per_block[0];
    cache.valid = 0;
    cache.oldest = 0;
    // Streaming, the block is read ONCE for the whole clip, and the next
    // one -- block 0 again after the last, for the next clip -- under
    // all of this block's chunks.
    const Block* wb = nullptr;
    if (!acquire_block_(bi, (bi + 1) % _cfg.n_layers, &wb, err)) {
      return false;
    }
    for (int ci = 0; ci < chunks; ++ci) {
      const int f = chunk_f(ci), t_first = chunk_t(ci);
      const std::size_t seq = (std::size_t)f * tok_per_frame;
      // The int8 split's width for shapes recorded since; no encoder is
      // open here.
      if (_i8) { _i8->tune_pending(_mc); }
      // The source rows enter the residual stream at block 0, and only
      // there -- so the slice exists for that one pass.
      SharedBuffer lq;
      if (bi == 0) {
        lq = _mc->make_shared_buffer(seq * D * 2);
        if (lq.empty()) { return fail("source row slice failed"); }
        std::memcpy(lq.contents(),
                    (const std::uint8_t*)req.rows +
                        (std::size_t)t_first * tok_per_frame * D * 2,
                    seq * D * 2);
      }
      if (!block_(st, bi, wb, 0, f, t_first, ci, xs[(std::size_t)ci],
                  bi == 0 ? &lq : nullptr, req.params, err)) {
        if (_streaming) { _streaming->slots.join(); }
        return false;
      }
      if (req.block_progress && !req.block_progress(ci + 1, chunks)) {
        return fail("aborted");
      }
    }
    if (req.progress && !req.progress(bi + 1, _cfg.n_layers)) {
      return fail("aborted");
    }
  }
  report_prof_(st, "clip", _cfg.n_layers);
  // The windows and the RoPE tables are done with; the head needs the
  // residual streams and its own scratch.
  st.per_block.clear();
  st.rope.clear();

  for (int ci = 0; ci < chunks; ++ci) {
    const int f = chunk_f(ci), t_first = chunk_t(ci);
    const int seq = f * th * tw;
    SharedBuffer pred;
    if (!head_(st, xs[(std::size_t)ci], seq, &pred, err)) { return false; }
    xs[(std::size_t)ci] = SharedBuffer{};
    std::vector<float> upd((std::size_t)C * f * plane, 0.0f);
    if (!unpatchify_(pred, f, h8, w8, &upd, err)) { return false; }
    // A NON-FINITE PREDICTION IS REFUSED, not decoded: the VAE turns it
    // into a clip of black frames that looks like a finished run.
    std::size_t bad = 0;
    for (const float u : upd) {
      if (!std::isfinite(u)) { ++bad; }
    }
    if (bad > 0) {
      return fail(fmt("chunk {} of {} predicted {} non-finite values of {}; "
                      "{} query tile(s) were routed to no key block at all",
                      ci + 1, chunks, bad, upd.size(), _empty_tiles)());
    }
    for (int c = 0; c < C; ++c) {
      for (int t = 0; t < f; ++t) {
        const std::size_t src = ((std::size_t)c * f + t) * plane;
        const std::size_t dst = ((std::size_t)c * T + (t_first + t)) * plane;
        const float* nz = noise.data() + dst;
        for (std::size_t i = 0; i < plane; ++i) {
          (*latent)[dst + i] = nz[i] - upd[src + i];
        }
      }
    }
  }
  return true;
}

bool
MetalFlashVsrTransformer::unpatchify_(const SharedBuffer& tokens, int T,
                                      int h8, int w8, std::vector<float>* out,
                                      std::string* err)
{
  const int C = _cfg.out_channels;
  const int cols = C * _cfg.patch_t * _cfg.patch_h * _cfg.patch_w;
  const std::size_t rows = (std::size_t)T * (h8 / 2) * (w8 / 2);
  const std::size_t n = (std::size_t)C * T * h8 * w8;
  SharedBuffer dst = _mc->make_shared_buffer(n * 4);
  if (dst.empty()) {
    if (err != nullptr) { *err = "unpatchify allocation failed"; }
    return false;
  }
  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(_fn_unpack);
    enc.set_buffer(0, tokens); enc.set_buffer(1, dst);
    enc.set_constant(2, C); enc.set_constant(3, T);
    enc.set_constant(4, h8); enc.set_constant(5, w8);
    enc.dispatch({(unsigned)cols, (unsigned)rows, 1}, {64, 1, 1});
    enc.end();
  }
  std::string ge;
  if (!stream.commit().wait_ok(&ge)) {
    if (err != nullptr) { *err = "unpatchify failed: " + ge; }
    return false;
  }
  out->assign(n, 0.0f);
  std::memcpy(out->data(), dst.contents(), n * 4);
  return true;
}

// ----------------------------------------------------------- patchify

bool
MetalFlashVsrTransformer::ensure_kernels_(std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (_fn_pack.valid()) { return true; }
  // load() opened this one already; the matrix-core bias pass and the
  // split's fold are functions of it and must not see it replaced.
  if (!_lib_elt.valid()) {
    _lib_elt = _mc->load_library("llm_elementwise_bf16");
  }
  _lib_gemm = _mc->load_library("dense_gemm_bf16");
  _lib_rms  = _mc->load_library("rms_norm_bf16");
  _fn_pack   = _lib_elt.function("flashvsr_patch_pack_f16");
  _fn_unpack = _lib_elt.function("flashvsr_patch_unpack_f16");
  _fn_window = _lib_elt.function("flashvsr_window_permute_f16");
  _fn_rms    = _lib_rms.function("rms_norm_fast_f16");
  _fn_gemm   = _lib_gemm.function("dense_gemm_bias_f16");
  _fn_gemm_bm64 = _lib_gemm.function("dense_gemm_t_bm64_f16");
  _fn_gemm_bn64 = _lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  // 0 = the plain 16x16 kernel, 1 = BM64/BN32 (default), 2 = BM64/BN64.
  _gemm_tile = 1;
  if (const char* e = std::getenv("VPIPE_FVSR_GEMM_TILE")) {
    _gemm_tile = std::atoi(e);
  }
  if (std::getenv("VPIPE_FVSR_PLAIN_GEMM") != nullptr) { _gemm_tile = 0; }
  if (_gemm_tile == 2 && !_fn_gemm_bn64.valid()) { _gemm_tile = 1; }
  _dense_attn = std::getenv("VPIPE_FVSR_DENSE_ATTN") != nullptr;
  _fn_ln_mod = _lib_elt.function("layernorm_modulate_f16");
  _fn_ln_aff = _lib_elt.function("layer_norm_affine_f16");
  _fn_transpose = _lib_elt.function("transpose_abd_f16");
  _fn_blockmean = _lib_elt.function("flashvsr_block_mean_f16");
  _fn_gelu   = _lib_elt.function("gelu_tanh_ff_f16");
  _fn_copy   = _lib_elt.function("copy_f16");
  _lib_sdpa  = _mc->load_library("sdpa_bf16");
  _fn_sdpa   = _lib_sdpa.function("sdpa_full_f16");
  _lib_route = _mc->load_library("flashvsr_route_bf16");
  _fn_rt_scores  = _lib_route.function("flashvsr_route_scores");
  _fn_rt_softmax = _lib_route.function("flashvsr_route_softmax");
  _fn_rt_thresh  = _lib_route.function("flashvsr_route_threshold");
  _fn_rt_flags   = _lib_route.function("flashvsr_route_flags");
  _fn_rt_keep    = _lib_route.function("flashvsr_route_keep");
  _fn_sp_off     = _lib_route.function("flashvsr_span_offsets");
  _fn_sp_emit    = _lib_route.function("flashvsr_span_emit");
  // Spans by default: they SKIP where the mask only masks. The mask
  // path is kept for the A/B that established that.
  _use_spans = std::getenv("VPIPE_FVSR_MASK_ATTN") == nullptr &&
               _fn_rt_keep.valid() && _fn_sp_off.valid() &&
               _fn_sp_emit.valid();
  _fn_gated  = _lib_elt.function("gated_residual_f16");
  _fn_resadd = _lib_elt.function("residual_add_f16");
  _lib_rope  = _mc->load_library("rope_bf16");
  _fn_trope  = _lib_rope.function("transpose_rope_pair_ftab_f16");
  _lib_attn  = _mc->load_library("attn_steel");
  // An unvalidated ComputeFunction is a SILENT no-op, so each is checked
  // here rather than found later as a zero activation.
  if (!_fn_pack.valid() || !_fn_unpack.valid() || !_fn_window.valid() ||
      !_fn_rms.valid() || !_fn_gemm.valid() || !_fn_ln_mod.valid() ||
      !_fn_ln_aff.valid() || !_fn_transpose.valid() ||
      !_fn_blockmean.valid() || !_fn_gelu.valid() || !_fn_gated.valid() ||
      !_fn_resadd.valid() || !_fn_trope.valid() || !_lib_attn.valid() ||
      !_fn_copy.valid() || !_fn_sdpa.valid() ||
      !_fn_rt_scores.valid() || !_fn_rt_softmax.valid() ||
      !_fn_rt_thresh.valid() || !_fn_rt_flags.valid() ||
      !_fn_gemm_bm64.valid()) {
    return fail("flashvsr denoiser kernels failed to validate");
  }
  return true;
}

bool
MetalFlashVsrTransformer::patchify_tokens(const float* latent, int T, int h8,
                                          int w8, std::vector<float>* tokens,
                                          std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (latent == nullptr || tokens == nullptr) { return fail("no latent"); }
  if (T <= 0 || h8 <= 0 || w8 <= 0 || h8 % 2 != 0 || w8 % 2 != 0) {
    return fail("latent geometry must be even in both spatial axes");
  }
  if (!ensure_kernels_(err)) { return false; }
  if (_patch_w.empty()) { return fail("patch embedding not loaded"); }

  const int C = _cfg.in_channels;
  const int K = C * _cfg.patch_t * _cfg.patch_h * _cfg.patch_w;
  const int D = _cfg.hidden;
  const std::size_t rows =
      (std::size_t)T * (h8 / 2) * (w8 / 2);
  const std::size_t n_lat = (std::size_t)C * T * h8 * w8;

  SharedBuffer src = _mc->make_shared_buffer(n_lat * 4);
  if (src.empty()) { return fail("latent upload allocation failed"); }
  std::memcpy(src.contents(), latent, n_lat * 4);
  SharedBuffer cols = _mc->make_shared_buffer(rows * (std::size_t)K * 2);
  SharedBuffer out  = _mc->make_shared_buffer(rows * (std::size_t)D * 2);
  if (cols.empty() || out.empty()) { return fail("patchify allocation failed"); }

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(_fn_pack);
    enc.set_buffer(0, src);
    enc.set_buffer(1, cols);
    enc.set_constant(2, C);
    enc.set_constant(3, T);
    enc.set_constant(4, h8);
    enc.set_constant(5, w8);
    enc.dispatch({(unsigned)K, (unsigned)rows, 1}, {64, 1, 1});

    enc.set_function(_fn_gemm);
    enc.set_buffer(0, cols);
    enc.set_buffer(1, _patch_w);
    enc.set_buffer(2, _patch_b.empty() ? _patch_w : _patch_b);
    enc.set_buffer(3, out);
    enc.set_constant(4, (int)rows);
    enc.set_constant(5, D);
    enc.set_constant(6, K);
    enc.set_constant(7, _patch_b.empty() ? 0 : 1);
    enc.dispatch({(unsigned)(((D + 15) / 16) * 16),
                  (unsigned)(((int)rows + 15) / 16 * 16), 1}, {16, 16, 1});
    enc.end();
  }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail("patchify GPU work failed: " + gpu_err);
  }
  tokens->assign(rows * (std::size_t)D, 0.0f);
  const auto* p = (const std::uint16_t*)out.contents();
  for (std::size_t i = 0; i < tokens->size(); ++i) { (*tokens)[i] = from_bf16(p[i]); }
  return true;
}

bool
MetalFlashVsrTransformer::cross_kv(int block, std::vector<float>* k,
                                   std::vector<float>* v) const
{
  if (block < 0 || block >= (int)_blocks.size()) { return false; }
  const Block& b = _blocks[(std::size_t)block];
  if (b.ck.empty() || b.cv.empty()) { return false; }
  const int T = _cfg.text_tokens, NH = _cfg.n_heads, HD = _cfg.head_dim;
  const int D = _cfg.hidden;
  const std::size_t n = (std::size_t)T * D;
  // Held head-major; reported TOKEN-major, which is the layout the
  // reference's own tensor has. An accessor that reports the internal
  // layout would make the comparison against a golden meaningless.
  auto grab = [&](const SharedBuffer& src, std::vector<float>* dst) {
    dst->assign(n, 0.0f);
    const auto* p = (const std::uint16_t*)src.contents();
    for (int hh = 0; hh < NH; ++hh) {
      for (int t = 0; t < T; ++t) {
        for (int d = 0; d < HD; ++d) {
          (*dst)[(std::size_t)t * D + (std::size_t)hh * HD + d] =
              from_bf16(p[((std::size_t)hh * T + t) * HD + d]);
        }
      }
    }
  };
  if (k != nullptr) { grab(b.ck, k); }
  if (v != nullptr) { grab(b.cv, v); }
  return true;
}

std::uint64_t
MetalFlashVsrTransformer::resident_bytes() const noexcept
{
  std::uint64_t n = 0;
  auto add = [&n](const SharedBuffer& b) { n += b.byte_size(); };
  for (const auto& b : _blocks) {
    add(b.q1); add(b.k1); add(b.v1); add(b.o1);
    add(b.q1b); add(b.k1b); add(b.v1b); add(b.o1b);
    add(b.qn1); add(b.kn1);
    add(b.q2); add(b.o2); add(b.q2b); add(b.o2b); add(b.qn2);
    add(b.ck); add(b.cv);
    add(b.n3w); add(b.n3b);
    add(b.ff_in); add(b.ff_in_b); add(b.ff_out); add(b.ff_out_b);
    add(b.mod);
  }
  add(_t_vec); add(_t_mod); add(_head_mod);
  // Streaming, the blocks above hold only what the load folded; the
  // weights live in the slot pair, which a holding must count.
  if (_streaming && _streaming->slots.on()) {
    n += (std::uint64_t)_streaming->slots.last_bytes() *
         (_streaming->slots.paired() ? 2u : 1u);
    n += _streaming->scratch.size();
  }
  return n;
}

bool
MetalFlashVsrTransformer::uses_i8_gemm() const noexcept
{
  return _i8 != nullptr;
}

bool
MetalFlashVsrTransformer::uses_sage() const noexcept
{
  return _sage != nullptr;
}

std::string
MetalFlashVsrTransformer::accel_summary() const
{
  return fmt("GEMM {}, attention {}, i8_gemm {}, sage_attn {}",
             _use_mma2 ? "matmul2d" : "ALU",
             _use_attn_nax ? "attn_steel_nax" : "ALU steel",
             _i8 ? "on" : "off", _sage ? "on" : "off")();
}

metal_compute::ComputeFunction
MetalFlashVsrTransformer::attn_fn_(const metal_compute::FunctionConstants& fc)
{
  return _use_attn_nax
             ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
             : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
}

}  // namespace genai
}  // namespace vpipe
