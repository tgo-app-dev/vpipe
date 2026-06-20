#ifndef VPIPE_GENERATIVE_MODELS_METAL_LLAMA_MODEL_H
#define VPIPE_GENERATIVE_MODELS_METAL_LLAMA_MODEL_H

// MetalLlamaModel -- procedural Llama-3.1 forward pass on the
// metal-compute framework (no MLX), chaining the verified kernels
// (affine_qmv, rms_norm, rope, sdpa_causal, swiglu, residual,
// dequant_embed_gather, kv_write). Processes one token per forward
// (n=1) against a contiguous KV cache, so prefill = looping the
// prompt and generation = looping argmax. v1: correctness-first
// (GEMV everywhere, single command buffer per token, host argmax).

#include "generative-models/context-manager.h"   // ContextId, ContextManager
#include "generative-models/metal-token-muxer.h"
#include "generative-models/model-exec.h"         // GpuSamplerParams
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/event.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; class ComputeEncoder; }

namespace vpipe::genai {

class MetalLlamaModel {
public:
  struct Config {
    int   n_layers   = 32;
    int   hidden     = 4096;
    int   n_heads    = 32;
    int   n_kv_heads = 8;
    int   head_dim   = 128;
    int   ffn_inner  = 14336;
    int   vocab      = 128256;
    float rope_theta = 500000.0f;
    float rms_eps    = 1e-5f;
    // llama3 rope scaling
    float rope_factor   = 8.0f;
    float rope_low_freq  = 1.0f;
    float rope_high_freq = 4.0f;
    float rope_orig_ctx  = 8192.0f;
    int   max_seq     = 512;   // per-context logical length cap
    int   page_tokens = 256;   // paged KV: tokens per page
    int   max_pages   = 0;     // paged KV: pool cap (0 -> derived)
  };

  // Load + bind all weights from `model_dir`/model.safetensors.
  // Returns nullptr on failure.
  static std::unique_ptr<MetalLlamaModel> load(
      const std::string& model_dir,
      metal_compute::MetalCompute* mc,
      const Config& cfg);

  // Multi-context: operate on a specific KV context (the exec maps
  // each LoadedLanguageModel context to one of these). context_manager
  // mints/branches them. The token's absolute position comes from the
  // context manager's append() (paged KV), so callers no longer pass a
  // position -- the manager is the single source of truth.
  ContextManager* context_manager() { return _ctx.get(); }
  std::vector<float> forward(ContextId cid, std::int32_t token_id);
  std::vector<float> prefill(ContextId cid,
                             const std::vector<std::int32_t>& ids);

  // Single-context convenience (default context), used by the
  // standalone tests. Appends one token and returns [vocab] logits.
  std::vector<float> forward(std::int32_t token_id) {
    return forward(_cid, token_id);
  }
  // Batched prefill of the whole prompt; returns [vocab] logits at the
  // last position.
  std::vector<float> prefill(const std::vector<std::int32_t>& ids) {
    return prefill(_cid, ids);
  }

  // Convenience: argmax of forward().
  std::int32_t forward_argmax(std::int32_t token_id);

  // Pipelined decode (prototype). Generates `n_steps` tokens starting
  // from `first_token`, appending to `cid`'s paged KV. The whole
  // next-token->embedding feedback stays GPU-resident: a per-step kernel
  // writes the next id (argmax for greedy, or temperature+top-p sampling)
  // into a buffer the embed gather reads straight back, so the token
  // never round-trips to the host. Per-token command buffers are
  // committed without waiting and chained by a GPU event, so the host
  // encodes ahead while the GPU runs back-to-back -- none of the
  // per-token host work (logits pull, host argmax/sampling, muxer embed
  // fetch) the synchronous loop pays. Decode positions/page tables are
  // deterministic (independent of the sampled token) and computed ahead.
  // `temperature <= 0` selects greedy argmax (output then matches a
  // forward_argmax() loop); otherwise tokens are sampled on-GPU with the
  // given `top_p` and `seed` (varied per step). Fills `out_ids`
  // (size n_steps). Returns false on append failure.
  bool decode_pipelined(ContextId cid, std::int32_t first_token,
                        int n_steps, std::vector<std::int32_t>& out_ids,
                        float temperature = 0.0f, float top_p = 1.0f,
                        std::uint64_t seed = 0);

  // ---- Per-token streaming pipelined decode -------------------------
  // The streaming counterpart of decode_pipelined: exposes the GPU-
  // resident chain one token at a time so the caller can stream /
  // stop-check / abort while overlapping the host's per-token work with
  // the GPU's next forward. See ModelExec::pdecode_* for the contract.
  // Llama is text-only, so rope_first is ignored (decode is sequential).
  bool pdecode_begin(ContextId cid, std::int32_t first_token,
                     std::span<const std::int32_t> prompt,
                     const GpuSamplerParams& sp, int max_tokens,
                     int rope_first = -1);
  bool pdecode_commit(ContextId cid);
  std::int32_t pdecode_next(ContextId cid);
  void pdecode_end(ContextId cid);

  const Config& config() const { return _cfg; }

private:
  MetalLlamaModel() = default;

  // Encode one decode token's layer stack + final norm + lm_head into
  // `enc`, gathering the input token's embedding from in_ids[in_off]
  // straight into the residual stream and writing `_logits`. Shared by
  // decode_pipelined and the per-token pdecode_* path.
  void encode_decode_step_(
      metal_compute::ComputeEncoder& enc, ContextId cid, int pos,
      std::size_t page_off, int n_pages,
      const ContextManager::AppendSlot& slot,
      const metal_compute::SharedBuffer& in_ids, std::size_t in_off,
      const metal_compute::SharedBuffer& pgtab, std::size_t pgtab_off);

  // Flash-decode-GQA attention for one decode query (head_dim <= 256, GQA
  // G=Hq/Hkv <= 4): scan each KV head once for all G query heads + merge,
  // writing `out` [Hq*D]. Shared by forward() + encode_decode_step_.
  void encode_gqa_attn_(
      metal_compute::ComputeEncoder& enc,
      const metal_compute::SharedBuffer& q,
      const metal_compute::SharedBuffer& kp,
      const metal_compute::SharedBuffer& vp,
      const metal_compute::SharedBuffer& out,
      const metal_compute::SharedBuffer& pgtab, std::size_t pgtab_off,
      float scale, int D, int Hq, int Hkv, int pos, int page_tokens,
      int n_pages);

  // Encode the next-token pick (greedy argmax or full GPU sampler) into
  // `enc`: reads `logits`, writes the chosen id to out_id[out_off], and
  // (sampled path) updates `seen`. Shared by decode_pipelined / pdecode_*.
  void encode_sample_(metal_compute::ComputeEncoder& enc,
                      const metal_compute::SharedBuffer& logits,
                      const metal_compute::SharedBuffer& out_id,
                      std::size_t out_off, const GpuSamplerParams& sp,
                      std::uint32_t step_seed,
                      const metal_compute::SharedBuffer& sample_ws,
                      const metal_compute::SharedBuffer& seen);

  // Per-context streaming-pipeline decode state (pdecode_*).
  struct PDecode {
    metal_compute::SharedBuffer gen_ids;     // [max+1] int32 GPU chain
    metal_compute::SharedBuffer seen;        // [vocab] uint8 (sampled)
    metal_compute::SharedBuffer sample_ws;   // [vocab] f16 (sampled)
    metal_compute::SharedBuffer pgt;         // [max_pages*3] int32 slice
    metal_compute::CommandStream stream;
    metal_compute::Event ev;
    metal_compute::CommandStream::Fence inflight;
    GpuSamplerParams sp;
    ContextId cid;
    int           cap           = 0;
    int           produced      = 0;
    int           pending       = -1;
    std::uint64_t gpu_step      = 0;
    bool          have_inflight = false;
  };
  std::unordered_map<std::uint32_t, PDecode> _pdec;

  struct Layer {
    metal_compute::SharedBuffer in_ln, post_ln;
    metal_compute::SharedBuffer qw, qs, qb, kw, ks, kb, vw, vs, vb, ow, os, ob;
    // gate_proj and up_proj fused + interleaved by output feature into
    // one [2*ffn, hidden] quantized matrix (row 2g=gate g, row 2g+1=up
    // g) so the SwiGLU activation fuses into the matmul store; see
    // affine_qmm_swiglu / affine_qmv_swiglu.
    metal_compute::SharedBuffer guw, gus, gub, dw, ds, db;
    // KV now lives in the ContextManager (metal backend), not per Layer.
  };

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;

  // Kernel functions.
  metal_compute::ComputeLibrary _lib_qmv, _lib_qmm, _lib_rms, _lib_elt,
      _lib_rope, _lib_sdpa, _lib_attn;
  metal_compute::ComputeFunction _fn_qmv, _fn_qmv_add, _fn_qmm, _fn_rms,
      _fn_residual, _fn_rope, _fn_embed, _fn_transpose, _fn_argmax,
      _fn_sample;
  // Fused SwiGLU MLP (interleaved gate/up weight -> silu(gate)*up in one
  // pass): GEMV for decode, GEMM for prefill.
  metal_compute::ComputeFunction _fn_qmv_swiglu, _fn_qmm_swiglu;

  std::vector<Layer> _layers;
  metal_compute::SharedBuffer _embed_w, _embed_s, _embed_b;
  metal_compute::SharedBuffer _lm_w, _lm_s, _lm_b;
  metal_compute::SharedBuffer _final_ln;
  metal_compute::SharedBuffer _inv_freq;

  // Backend-native embedding + KV abstractions (shared with the rest
  // of the metal LLM core).
  std::unique_ptr<MetalTokenMuxer>     _muxer;
  std::unique_ptr<ContextManager>      _ctx;
  ContextId                            _cid;

  // Reused scratch (one decode token; prefill allocates its own).
  metal_compute::SharedBuffer _x, _hn, _q, _k, _v, _attn, _ao;
  metal_compute::SharedBuffer _sg, _mo, _logits;
  // Paged KV: page-table scratch [max_pages * 3] int32 (reused across
  // layers within a forward; the page list is per-context) + the paged
  // write/attention kernels.
  metal_compute::SharedBuffer _pgtab;
  metal_compute::ComputeFunction _fn_kv_write_paged, _fn_kv_gather_paged,
      _fn_sdpa_paged, _fn_sdpa_paged_mb, _fn_sdpa_paged_mma;
  // Flash-decode-GQA serial attention (head_dim <= 256, GQA G=Hq/Hkv <= 4):
  // read each KV head ONCE for all G query heads (the same sdpa.metal kernel
  // the Qwen3.5 path uses; covers dense Llama + the Qwen3-ASR text decoder).
  // Default ON when capable; VPIPE_GQA_ATTN=0/1 overrides, VPIPE_GQA_SPLIT
  // sets the position split. Partials are f32: O [Hq,split,D], m/l [Hq,split].
  metal_compute::ComputeFunction _fn_sdpa_gqa, _fn_sdpa_gqa_merge;
  metal_compute::SharedBuffer _d_gqa_oacc, _d_gqa_m, _d_gqa_l;
  bool _gqa_attn = false;
  int  _gqa_split = 32;
  // Steel (MLX-vendored) flash attention for fresh prefill (q_offset==0,
  // contiguous K/V): the PSO is function-constant-specialized per
  // (align_Q, align_K) so it is resolved per-prefill from _lib_attn, not
  // cached as a single ComputeFunction. _attn_params holds the AttnParams.
  metal_compute::SharedBuffer _attn_params;
  // Decode attention switches to the multi-simdgroup kernel once the
  // context reaches this length (single-simdgroup wins at short ctx;
  // multi-simdgroup recovers ~32x KV-scan parallelism at long ctx).
  // Override with VPIPE_SDPA_MB_MIN.
  int _sdpa_mb_min = 128;
};

}  // namespace vpipe::genai

#endif
