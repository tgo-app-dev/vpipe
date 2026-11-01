#ifndef VPIPE_GENERATIVE_MODELS_LFM3_METAL_LFM3_MODEL_H
#define VPIPE_GENERATIVE_MODELS_LFM3_METAL_LFM3_MODEL_H

// MetalLfm3Model -- Liquid AI's LFM3 language model (the decoder of
// LFM3-VL) on metal-compute, no MLX in the forward.
//
// A hybrid of three layer kinds, each followed by a SwiGLU feed-forward,
// with "sandwich" norms (a sublayer's output RMS-normed before the
// residual add, the feed-forward's too):
//   * conv: the gated short convolution -- in_proj to [B | C | x], a
//     causal depthwise conv of width K over B*x, gated by C, out_proj.
//     A context carries the last K - 1 values of B*x per channel.
//   * sliding_attention: GQA attention over the last `sliding_window`
//     positions, full-head RoPE at its own theta.
//   * full_attention: GQA attention over the whole context, RoPE over
//     half the head -- the first quarter of the dims' pairs turn at the
//     full head's frequency progression, the rest stay put.
// Both attentions norm q and k per head and gate each head's output by a
// sigmoid of its own logit (gate_proj). The final norm is named
// embedding_norm; the head is the tied embedding table.
//
// K/V: every attention layer's in the paged pool of the model's own
// ContextManager (a branch shares its parent's pages), the sliding layers
// windowed at read time -- the paged decode kernel by shifting the page
// table and the query position by the window's first position, the paged
// prefill kernel by its WIN twin. The conv layers' state is the
// manager's linear-layer conv state, deep-copied on a branch.
//
// One forward serves every shape: a list of SEGMENTS, each a context and
// its next rows -- a prefill (one segment of n rows), a decode step (one
// of one row) or a batched step over branches (several of one row each).
// The projections and norms run over every row at once; the convs and
// attentions per segment. Weights dense F16 (converted once from the
// checkpoint's BF16); the activations F16 (the residual stays under ~30
// on real prompts).

#include "generative-models/context-manager.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vpipe { class SessionContextIntf; }
namespace vpipe::metal_compute { class MetalCompute; class ComputeEncoder; }

namespace vpipe::genai {

struct ModelConfig;
class WeightSet;
class I8GemmContext;

class MetalLfm3Model {
public:
  enum class Kind : std::uint8_t { Conv, Sliding, Full };

  struct Config {
    int   n_layers   = 24;
    int   hidden     = 2048;
    int   n_heads    = 16;
    int   n_kv_heads = 4;
    int   head_dim   = 128;
    int   ffn        = 7680;
    int   vocab      = 128000;
    int   conv_k     = 7;           // conv_L_cache
    int   window     = 256;         // sliding_window
    float eps        = 1e-5f;       // norm_eps
    std::vector<Kind> kinds;        // [n_layers]
    // Per attention kind, the rotary pairs' inverse frequencies
    // [head_dim / 2]: zero past the rotated pairs.
    std::vector<float> inv_freq_full, inv_freq_sliding;
    int max_seq     = 131072;       // per-context cap
    int page_tokens = 256;
    int max_pages   = 0;            // 0: the manager derives it

    int qd() const { return n_heads * head_dim; }
    int kd() const { return n_kv_heads * head_dim; }
    // The fused attention projection's width: q | k | v | gate.
    int qkvg() const { return qd() + 2 * kd() + n_heads; }

    // The text half of an LFM3(-VL) config.json: the root (LFM3) or its
    // text_config (LFM3-VL). False with `err` on anything this port does
    // not run (a rope type but `default`, a conv bias, no sandwich norm).
    static bool from_dir(const std::string& dir, Config* out,
                         std::string* err);
  };

  // The checkpoint the manager holds; the model KEEPS the set. Every
  // weight under `prefix` (model.language_model. in LFM3-VL).
  static std::unique_ptr<MetalLfm3Model> load(
      std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
      const Config& cfg, std::string* err = nullptr);
  ~MetalLfm3Model();

  // What the language model holds once loaded (dense F16, the fused
  // projections included, the tied head counted once), off the tensor
  // headers -- the plan's figure.
  static std::size_t weight_bytes(const std::string& model_dir);
  // Its working memory beside the weights: a prefill piece's buffers
  // (kPieceRows rows), the logits, and the paged K/V of `kv_tokens` at
  // the capacity the pool's doubling reaches, plus `contexts` contexts'
  // conv state -- and with `i8` (the int8 prefill engaged) its context's
  // grow-only scratches at a piece.
  static std::size_t run_bytes(const Config& c, std::int64_t kv_tokens,
                               int contexts, bool i8 = false);

  // Rows a forward takes at most; a longer prompt runs in pieces.
  static constexpr int kPieceRows = 2048;

  ContextManager* context_manager() { return _ctx.get(); }
  const Config& config() const { return _cfg; }

  // One forward. `rows` are the inputs: token ids (embedded on the GPU)
  // or, when `emb` is non-null, embeddings [total, hidden] f32 on the
  // host. Segment i takes rows [row0_i, row0_i + n_i), appended to its
  // context in order; a context appears once. Logits [k, vocab] f32 for
  // rows [logits_from, total) (k = total - logits_from) go to `logits`
  // when non-null. `capture` (tests): the residual after every layer,
  // [n_layers][total * hidden] f32. False on failure (append, GPU).
  struct Segment {
    ContextId cid;
    int       row0 = 0;
    int       n    = 0;
  };
  bool forward(std::span<const Segment> segs,
               std::span<const std::int32_t> ids,
               const std::vector<float>* emb, int logits_from,
               std::vector<float>* logits,
               std::vector<std::vector<float>>* capture = nullptr);

  // A prompt of any length into `cid`, in pieces of kPieceRows; the
  // last row's logits to `logits`. `emb` as forward().
  bool prefill(ContextId cid, std::span<const std::int32_t> ids,
               const std::vector<float>* emb, std::vector<float>* logits);

  // Embedding rows of `ids` as host f32 [n, hidden].
  std::vector<float> embed_rows(std::span<const std::int32_t> ids) const;

  // Bytes of K/V and conv state held now.
  std::size_t kv_bytes() const;

  // Matrix-core GEMMs in use (M5; VPIPE_LFM3_NO_MMA2 the A/B).
  bool uses_matrix_cores() const noexcept { return _use_mma2; }

  // Dynamic-int8 prefill GEMMs (LOSSY, opt-in: the stages' i8_prefill;
  // shared/i8-gemm.h). The context engages only on matrix cores and from
  // its row floor (1024), so decode, a verify and short prompts keep the
  // f16 routes; the head never takes it. VPIPE_I8_GEMM=0|1 overrides.
  void set_i8_gemm(bool on);
  bool i8_gemm_enabled() const noexcept { return _i8 != nullptr; }
  // The context itself (tests: asked again for the state it is in, the
  // model keeps it, and with it the shapes it settled).
  const I8GemmContext* i8_context() const noexcept { return _i8.get(); }
  // GEMMs that took the int8 route so far (tests).
  std::uint64_t i8_gemm_count() const noexcept { return _i8_count; }

private:
  MetalLfm3Model() = default;

  struct Layer {
    Kind kind = Kind::Conv;
    metal_compute::SharedBuffer op_norm, post_op_norm, ffn_norm,
        post_ffn_norm;
    metal_compute::SharedBuffer in_w, conv_w, out_w;   // conv
    metal_compute::SharedBuffer qkvg_w, q_norm, k_norm, o_w;   // attention
    metal_compute::SharedBuffer w13, w2;               // feed-forward
  };

  struct Scratch {
    int rows = 0, logits = 0;
    metal_compute::SharedBuffer x, hn, big, mid, act, o;
    metal_compute::SharedBuffer qt, kt, vt, at;
    metal_compute::SharedBuffer cos_f, sin_f, cos_s, sin_s;
    metal_compute::SharedBuffer ids, lrows, lg;
    metal_compute::SharedBuffer oacc, m, l;
    metal_compute::SharedBuffer pgt;
  };
  bool ensure_scratch_(int rows, int logit_rows, int segs);

  // y[M, N] = x[M, K] @ w[N, K]^T at byte offsets; the route by M.
  // `int8`: the int8 prefill may take it (not the head).
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t xoff,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer& y, std::size_t yoff,
             int M, int N, int K, bool int8 = true);

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;
  std::shared_ptr<WeightSet> _ws;
  std::unique_ptr<ContextManager> _ctx;
  std::vector<Layer> _layers;
  metal_compute::SharedBuffer _embed, _final_norm;

  metal_compute::ComputeLibrary _lib_dense, _lib_mma, _lib_rms, _lib_rope,
      _lib_elt, _lib_sdpa, _lib_attn, _lib_lfm3;
  metal_compute::ComputeFunction _fn_gemm, _fn_gemm32, _fn_gemv_batch,
      _fn_mma, _fn_mma_deep, _fn_rms, _fn_rms_add, _fn_rms_rope,
      _fn_transpose, _fn_embed, _fn_swiglu, _fn_kv_write, _fn_dec_attn,
      _fn_dec_merge, _fn_attn, _fn_attn_win, _fn_conv, _fn_conv_state,
      _fn_conv_step, _fn_gate;
  bool _use_mma2 = false;
  std::unique_ptr<I8GemmContext> _i8;   // null when off
  std::uint64_t _i8_count = 0;
  int _i8_want = -1;   // the last set_i8_gemm's `on` (-1: none yet)
  Scratch _s;
};

}  // namespace vpipe::genai

#endif
