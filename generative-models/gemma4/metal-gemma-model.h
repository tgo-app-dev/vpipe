#ifndef VPIPE_GENERATIVE_MODELS_METAL_GEMMA_MODEL_H
#define VPIPE_GENERATIVE_MODELS_METAL_GEMMA_MODEL_H

// MetalGemmaModel -- procedural Gemma-4 text forward pass on the
// metal-compute framework (no MLX in the forward), the Gemma counterpart
// of MetalQwenModel. Gemma-4 (gemma3n-style) differs from Qwen3.5:
//   * Sandwich norms (input / post_attn / pre_ffn / post_ffn), all
//     RMSNorm applying the weight directly. The post_attn / post_ffn
//     norm is applied to the sublayer OUTPUT before the residual add.
//   * Per-layer-type head_dim: sliding layers 256, full layers 512
//     (global_head_dim). SDPA scale is 1.0 (q_norm/k_norm absorb 1/sqrt).
//     v is RMSNorm'd with NO learnable weight.
//   * "Proportional" RoPE on full layers (first 128 of 512 rotated, base
//     1e6) encoded as a zero-tail inv_freq into the full rope_f16 kernel;
//     sliding layers full rope over 256, base 1e4.
//   * Cross-layer KV sharing: the first (n_layers-num_kv_shared) layers
//     own their K/V; the rest reuse an earlier same-type layer's K/V.
//   * Per-Layer-Input (PLE) embeddings + gate, geglu MLP, embed scale
//     sqrt(hidden), tied quantized lm_head with final-logit softcapping.
//
// KV: unlike MetalQwenModel (paged ContextManager metal backend),
// Gemma's heterogeneous head_dim + KV-sharing don't fit the paged pool,
// so this model OWNS a simple per-context contiguous KV (per-layer
// growable f16 [Hkv, max_seq, head_dim]) and uses the NON-paged
// sdpa_causal_f16 / sdpa_causal_window_f16 + kv_write_f16 kernels.
//
// v1 (this file): correctness-first decode (n=1); prefill loops forward()
// over the prompt. Pipelined / batched paths are a later (Phase 4) add.

#include "generative-models/context-manager.h"   // ContextId
#include "generative-models/model-exec.h"         // GpuSamplerParams
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/event.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; class ComputeEncoder; }

namespace vpipe::genai {

class TuningReport;   // generative-models/shared/kernel-autotune.h

struct ModelConfig;

class MetalGemmaModel {
public:
  struct Config {
    int   n_layers        = 42;
    int   hidden          = 2560;
    int   n_heads         = 8;
    int   n_kv_heads      = 2;
    int   head_dim_sliding = 256;
    int   head_dim_full    = 512;
    int   ffn_inner       = 10240;
    int   vocab           = 262144;
    int   num_kv_shared   = 18;     // last N layers share KV
    int   hpli            = 256;    // hidden_size_per_layer_input
    int   sliding_window  = 512;
    // Ring chunk B: sliding layers store a bounded ring of
    // min(max_seq, sliding_window + sliding_chunk) instead of max_seq,
    // and prefill is chunked to <= sliding_chunk tokens per dispatch so a
    // wrap never clobbers an in-window key. 0 disables the ring (full
    // max_seq). Override with VPIPE_GEMMA_SLIDING_CHUNK.
    //
    // Default 2048 mirrors mlx_vlm's prefill_step_size: each prefill chunk
    // re-reads (and, on the matrix-core path, re-dequants) every projection
    // weight, so a small B re-pays that bandwidth per B tokens and tanks
    // long-context prefill (256 was ~2x slower at 2k). 2048 keeps prompts up
    // to ~2k single-pass while still bounding the sliding ring to window+2048
    // (~+68MB/ctx vs 256), independent of max_seq.
    int   sliding_chunk   = 2048;
    float final_softcap   = 30.0f;
    float rope_theta_sliding = 1.0e4f;
    float rope_theta_full    = 1.0e6f;
    float full_partial_rotary = 0.25f;
    float rms_eps         = 1e-6f;
    bool  use_bf16        = false;
    int   quant_bits      = 4;      // base bits (attn / down / lm_head)
    // gemma4_unified mixes precision: the MLP (gate/up/down) is 8-bit while
    // the rest is `quant_bits`. mlp_bits is derived at load from the weight
    // shapes; 0 == "same as quant_bits" (e4b).
    int   mlp_bits        = 0;
    // The embedding / tied lm_head can carry its own bit-width too: the
    // GGUF path keeps embed at 8-bit (requantised q6_K) over a 4-bit body.
    // Derived at load from the embed weight shape; 0 == "same as quant_bits".
    int   embed_bits      = 0;
    // Affine quantization group size. 64 for the mlx-converted checkpoints;
    // 32 for GGUF q4_0 (whose blocks are group-32). Selects the kernel
    // variant suffix (w<bits>g<group>).
    int   quant_group     = 64;
    // Symmetric quant (GGUF Q4_0): bias == -8*scale, so the 4-bit g32 decode
    // GEMVs read scale-only (skip the bias buffer). Bit-identical; ~10% fewer
    // weight-read bytes. False for asymmetric MLX-affine checkpoints.
    bool  quant_symmetric = false;
    // gemma4_unified full-attention layers reuse K as V (no v_proj) and use
    // fewer K/V heads than sliding layers. layer_n_kv_heads is empty for e4b
    // (uniform n_kv_heads).
    bool  attention_k_eq_v = false;
    std::vector<int> layer_n_kv_heads;
    // Mixture-of-Experts (Gemma-4 26B-A4B). enable_moe adds, ON EVERY LAYER, a
    // MoE block ALONGSIDE the existing dense MLP (a hybrid dense+MoE FFN): a
    // router (proj [E,H] + scale [H] + per_expert_scale [E]) selects top_k of
    // n_experts, whose gate|up (fused [E,2*moe_inner,H]) + down ([E,H,moe_inner])
    // run gelu-gated and combine; the dense MLP (ffn_inner) is unchanged. Five
    // FFN norms sandwich the two branches. n_experts==0 / !enable_moe => the
    // plain dense/PLE path. See the hybrid layer forward in encode_step_.
    int   n_experts   = 0;      // total experts (128)
    int   top_k       = 0;      // experts per token (8)
    int   moe_inner   = 0;      // per-expert intermediate (704)
    bool  enable_moe  = false;
    bool  is_moe() const { return enable_moe && n_experts > 0; }
    std::string weight_prefix = "language_model.model.";
    int   max_seq         = 4096;   // per-context KV preallocation
    int   page_tokens     = 256;    // bookkeeping ctx manager
    int   max_pages       = 0;
    // Per-layer attention kind (true = full_attention). Size == n_layers.
    std::vector<bool> is_full_layer;

    int n_kv(int L) const {
      return (L < (int)layer_n_kv_heads.size()) ? layer_n_kv_heads[(size_t)L]
                                                 : n_kv_heads;
    }
    bool k_eq_v(int L) const {
      return attention_k_eq_v && layer_is_full(L);
    }

    int first_shared() const { return n_layers - num_kv_shared; }
    bool layer_is_full(int L) const {
      return (L < (int)is_full_layer.size()) && is_full_layer[(size_t)L];
    }
    int head_dim(int L) const {
      return layer_is_full(L) ? head_dim_full : head_dim_sliding;
    }
  };

  static std::unique_ptr<MetalGemmaModel> load(
      const std::string& model_dir,
      metal_compute::MetalCompute* mc, const Config& cfg);

  // Derive the shape-affecting Config from a parsed ModelConfig.
  static Config config_from(const ModelConfig& c);

  // Decode one token: append to `cid`'s own KV and return [vocab] f32
  // logits at the new position. rope_pos<0 uses the KV slot position.
  std::vector<float> forward(ContextId cid, std::int32_t token_id,
                             int rope_pos = -1);
  // Greedy fast decode: one command buffer that gathers the input token's
  // embedding in-stream, runs the decode step, and computes the argmax
  // on-GPU -- returning a single token id WITHOUT pulling the full [vocab]
  // logits to host. Returns INT32_MIN if the argmax kernel is unavailable
  // (caller falls back to forward()); -1 on a genuine error.
  std::int32_t decode_step_fast(ContextId cid, std::int32_t token_id,
                                int rope_pos = -1);

  // ---- GPU-resident pipelined decode (pdecode_*) -------------------
  // The per-token streaming counterpart of decode_step_fast: the sampled
  // token stays GPU-resident (in-stream embed gather of the prior step's
  // token + on-GPU argmax/sample, no host logit pull), and per-token command
  // buffers chain via a GPU event so the host's stop-check overlaps the GPU's
  // next forward. Lifecycle mirrors the Qwen/Llama metal path: begin (once,
  // first_token already decided), commit (encode+commit ONE forward,
  // non-blocking), next (await the in-flight commit, return its id), end.
  // Greedy uses _fn_argmax; sampled uses _fn_sample. begin returns false if
  // the kernels are unavailable so callers fall back to the sync loop.
  // rope_first < 0 == sequential text (rope == KV slot); >= 0 is the
  // mROPE-advanced position after a multimodal prefill.
  bool pdecode_begin(ContextId cid, std::int32_t first_token,
                     std::span<const std::int32_t> prompt,
                     const GpuSamplerParams& sp, int max_tokens,
                     int rope_first);
  bool pdecode_commit(ContextId cid);
  std::int32_t pdecode_next(ContextId cid);
  void pdecode_end(ContextId cid);

  // Prefill: returns [vocab] logits at the last position. Empty on
  // failure. Batched (single forward over the n tokens) when n>1.
  std::vector<float> prefill(ContextId cid,
                             const std::vector<std::int32_t>& ids);

  // Multimodal prefill (owns_kv): `ids` is the mixed sequence (the
  // multimodal positions carry a placeholder id); `mm_rows` is [n_mm *
  // hidden] host f32 (vision/audio encoder rows, in prompt order, NOT
  // embed-scaled); `positions[k]` is the index in `ids` of mm row k.
  // Splices those rows into the main embedding stream + zeros the PLE id
  // at those positions (matching the MLX Gemma multimodal path). Returns
  // [vocab] logits at the last position. Empty on failure.
  std::vector<float> prefill_mm(ContextId cid,
                                const std::vector<std::int32_t>& ids,
                                const std::vector<float>& mm_rows, int n_mm,
                                const std::vector<int>& positions);

  // Own-KV lifecycle (the exec calls these on branch / release).
  bool branch_kv(ContextId parent, ContextId child);
  void release_kv(ContextId cid);

  const Config& config() const { return _cfg; }

  // Forbid a small set of token ids from EVER being predicted (argmax or
  // sampled), across prefill + every decode path. The logits at these ids
  // are masked to a far-negative sentinel right after the final lm_head /
  // softcap, so the host read-back, the GPU argmax, and the GPU sampler all
  // see them suppressed. Used by realtime stages to ban Gemma's reasoning-
  // channel tokens (<|channel>/<|think|>) so the model can't burn the decode
  // budget on a thought block. Pass an empty span to clear. Idempotent.
  void set_suppressed_tokens(std::span<const std::int32_t> ids);

private:
  MetalGemmaModel() = default;

  bool ensure_scratch_();

  // Per-machine decode-attention split autotune (sets _gtile_split + _gtile_kps
  // for this GPU; called once from ensure_scratch_). Greedy-token-exact safe.
  void tune_decode_attn_(TuningReport& rep);

  // Encode one decode token's full layer stack + final norm + lm_head +
  // softcap into `enc`, reading the residual stream `_d_x` (caller fills
  // it with the token embedding) and writing `_d_logits`. `kv` is the
  // per-context KV; `kv_off` the current seq_len (KV/rope position).
  // `tok_src` (optional) overrides the input-token source for the embedding
  // gather: when non-null the embed reads `tok_src` at byte offset `tok_off`
  // (a GPU-resident token, e.g. the pipelined-decode chain) instead of the
  // host-filled `_d_tok`. Null == read `_d_tok` (forward/decode_step_fast).
  // `pgtab_off` (bytes) selects which slot of the _d_pgtab ring the FULL-layer
  // page table is written to / read from -- so run-ahead pdecode (depth>1) gives
  // each in-flight step its own page table (the table changes per token). 0 for
  // the synchronous callers (they wait before reusing slot 0).
  void encode_step_(metal_compute::ComputeEncoder& enc, ContextId cid,
                    int kv_off,
                    const metal_compute::SharedBuffer* tok_src = nullptr,
                    std::size_t tok_off = 0,
                    std::size_t pgtab_off = 0);

  // Mask the suppressed (banned) token ids in a [vocab] logits buffer to a
  // far-negative sentinel (suppress_logits_f16). MUST be encoded AFTER the
  // final softcap. No-op when nothing is suppressed. Called from BOTH the
  // decode step (_d_logits) and the prefill chunk (its local logits) so the
  // banned tokens can't be predicted on the first token either.
  void encode_suppress_(metal_compute::ComputeEncoder& enc,
                        const metal_compute::SharedBuffer& logits, int vocab);

  // Encode a greedy argmax of `_d_logits` (the [vocab] row) into `out`/`out_off`
  // (an int32 slot). Dispatches the two-stage argmax_partial -> argmax_combine
  // when those kernels + the partials scratch are available (the 262k-vocab read
  // parallelised across kArgmaxM cores), else the single-tg _fn_argmax. Both are
  // token-exact (global lowest-index tie-break). Caller must hold _d_argmax_part
  // sized via ensure_scratch_().
  void encode_argmax_(metal_compute::ComputeEncoder& enc,
                      const metal_compute::SharedBuffer& out,
                      std::size_t out_off, int vocab);

  // Encode the GPU top-p/top-k/min-p sampler over `_d_logits` into `out`/
  // `out_off`. Dispatches the histogram multi-tg path (sample_*_f16) when those
  // kernels + scratch are available, else the single-tg _fn_sample fallback
  // (forced by VPIPE_GEMMA_SAMPLE1=1). `ws`/`seen` are the per-context sampler
  // scratch (sized [vocab]*sizeof(elt) / [vocab] uint8); `step_seed` is the
  // per-step derived seed. Caller holds the histogram scratch via
  // ensure_scratch_().
  void encode_sample_(metal_compute::ComputeEncoder& enc,
                      const metal_compute::SharedBuffer& out,
                      std::size_t out_off,
                      const metal_compute::SharedBuffer& ws,
                      const metal_compute::SharedBuffer& seen,
                      const GpuSamplerParams& sp, std::uint32_t step_seed,
                      int vocab);

  // Batched prefill of `n` tokens (n>1) in a SINGLE forward pass: one
  // command buffer of steel GEMMs (M=n) + batched attention, amortising
  // the weight reads across the n tokens (vs the per-token forward()
  // loop). Appends to `cid`'s own KV at slot kv_off; returns [vocab]
  // logits at the LAST token. Empty on failure.
  // `mm_emb` (optional) is an [n_mm, hidden] f16 buffer whose rows
  // overlay the main embeddings at `mm_pos` (the multimodal splice); the
  // PLE ids at those positions are zeroed. Null == pure text.
  // want_logits=false (intermediate chunks of a multi-chunk sliding-window
  // prefill): run ONLY the own-KV bulk layers to populate the K/V cache and
  // skip the shared-KV tail + final norm + lm_head (their residual/logits are
  // discarded for non-final chunks -- the tail writes no KV). Returns a
  // 1-element success sentinel. Halves the per-intermediate-chunk weight read.
  std::vector<float> forward_chunk_(ContextId cid,
                                    const std::vector<std::int32_t>& ids,
                                    const metal_compute::SharedBuffer* mm_emb
                                        = nullptr,
                                    const std::vector<int>* mm_pos = nullptr,
                                    bool want_logits = true);

public:
  // ---- Batched (N-branch parallel) decode --------------------------
  // VQA fanout: N branch contexts that share a prefix each decode one token
  // per step. The weight-bound matmuls (q/k/v/o, geglu MLP, lm_head, PLE
  // projections) run ONCE over the [N, hidden] stack (steel GEMM M=N); only
  // RoPE + the per-layer Contiguous-KV attention loop over N. Branches need
  // NOT share a seq_len -- each uses its own kv_off / rope position. Sync
  // (commit+wait) core; the caller samples each [vocab] row.
  bool decode_batched_step(std::span<const ContextId>      cids,
                           std::span<const std::int32_t>   in_tokens,
                           std::span<const std::int32_t>   rope_pos,
                           std::vector<float>&             out_logits);

  // Read-only KV length (seq_len / rope position, tokens) of a
  // LoadedLanguageModel context; 0 when the context has no KV yet.
  // Never materializes a context (unlike cm_for_'s lazy acquire).
  int context_seq_len(ContextId lm_cid) const;

private:
  struct BScratch {
    int n = 0;
    metal_compute::SharedBuffer x, hn, q3, kbuf, vbuf, attn, o, act, mlp,
        ple, pleproj, pli, plg, plp, logits;
    metal_compute::SharedBuffer tok_in, argmax_id;
    // Per-branch flash-decode partials (f32) for the global-layer gtile path:
    // [n, Hq, gtile_split, head_dim_full] + m/l [n, Hq, gtile_split]. Branch i
    // uses its own slice so the N branches don't serialize on shared scratch.
    metal_compute::SharedBuffer gqa_oacc, gqa_m, gqa_l;
    // Per-branch FULL-layer page tables [n, max_pages*3] (full_layers_paged):
    // branch i's paged attention reads slot i so the N branches don't share a
    // table. Refilled each step from each branch's page list.
    metal_compute::SharedBuffer pgtab;
  };
  bool ensure_bscratch_(BScratch& bs, int n);
  BScratch _bdec;

  // Encode one N-branch decode step into `enc`: reads `bs.x` [N, hidden],
  // appends per-branch K/V (Contiguous KV at each branch's kv_off), and
  // writes [N, vocab] into `bs.logits`. All metadata is per-branch (size N);
  // rope_pos_v[i] is branch i's RoPE position.
  void encode_batched_step_(metal_compute::ComputeEncoder& enc, BScratch& bs,
                            std::span<const ContextId> cids,
                            const std::vector<int>& kv_off_v,
                            const std::vector<int>& rope_pos_v);

  struct Layer {
    bool is_full   = false;
    int  head_dim  = 0;
    int  n_kv      = 0;    // K/V heads (gemma4_unified: full=1, sliding=8)
    bool k_eq_v    = false; // full layers reuse K as V (no v_proj weight)
    int  kv_source = -1;   // shared layers read this layer's K/V; -1 own
    metal_compute::SharedBuffer in_ln, post_attn_ln, pre_ffn_ln,
        post_ffn_ln, post_pli_ln;
    // ---- Mixture-of-Experts (Gemma-4 26B-A4B, iff _cfg.is_moe()) ---------
    // A MoE FFN block alongside the dense MLP (gate/up=guw, down=dw). Router:
    // proj (f16 dense [E,H]) + router_norm_w (scale[H]/sqrt(H), the folded
    // pre-norm weight) + per_expert_scale (f16 [E]). Experts: gate|up
    // interleaved [E,2I,H] affine (or f16 for _dense) in eguw/egus/egub; down
    // [E,H,I] affine (or f16) in edw2/eds2/edb2 (SEPARATE from the dense-MLP
    // dw/ds/db). Three extra norms (pre_ffn_ln/post_ffn_ln already above):
    // pre_ffn_ln_2 (MoE-branch input), post_ffn_ln_1 (dense-branch out),
    // post_ffn_ln_2 (MoE-branch out).
    metal_compute::SharedBuffer pre_ffn_ln_2, post_ffn_ln_1, post_ffn_ln_2;
    metal_compute::SharedBuffer router_proj, router_norm_w, per_expert_scale;
    metal_compute::SharedBuffer eguw, egus, egub;   // experts gate|up [E,2I,H]
    metal_compute::SharedBuffer edw2, eds2, edb2;   // experts down [E,H,I]
    int  eg_bits = 4;   // routed-expert quant width (4/8); selects w4/w8 gather
    metal_compute::SharedBuffer qw, qs, qb, kw, ks, kb, vw, vs, vb, ow, os, ob;
    // QKV-fused decode GEMV: q|k|v weights row-concatenated into one buffer so
    // a single full-occupancy GEMV replaces the 2-3 small per-proj GEMVs (the
    // 512-row k/v matvecs under-occupy the GPU). Outputs still split to the
    // separate _d_q/_d_k/_d_v. qkv_vrows==0 => q|k only (k_eq_v layers).
    metal_compute::SharedBuffer qkvw, qkvs, qkvb;
    int  qkv_qrows = 0, qkv_krows = 0, qkv_vrows = 0;
    bool qkv_fused = false;
    metal_compute::SharedBuffer q_norm, k_norm;   // [head_dim]
    // MLP: gate/up INTERLEAVED into one [2*ffn, hidden] weight (row 2g=gate
    // g, 2g+1=up g) for the fused GeGLU qmv/qmm (no separate gate/up).
    metal_compute::SharedBuffer guw, gus, gub, dw, ds, db;
    metal_compute::SharedBuffer plg_w, plg_s, plg_b;   // per_layer_input_gate
    metal_compute::SharedBuffer plp_w, plp_s, plp_b;   // per_layer_projection
    // Dense (raw-HF bf16/f16) path ONLY: plain [out,in] elt weights, no
    // scales/biases. q/k/v/o/down/plg/plp reuse the buffers above (qw/kw/vw/
    // ow/dw/plg_w/plp_w); gate/up stay SEPARATE here (the affine gate|up
    // interleave is a quant-packing trick, unused for dense). Set iff _dense.
    metal_compute::SharedBuffer dgate, dup;
    // Per-layer MLP intermediate size. Uniform (== Config::ffn_inner) for e4b /
    // gemma4_unified, but the raw-HF E2B DOUBLES it on the KV-shared layers
    // (use_double_wide_mlp: gate/up/down widened 2x on L >= first_shared). The
    // dense forward reads ly.ffn instead of the global ffn_inner.
    int ffn = 0;
    float layer_scalar = 1.0f;
  };

  // Per-context contiguous KV now lives in the shared ContextManager
  // (Spec::kv_layout == Contiguous): per-layer [Hkv, max_seq, head_dim(L)]
  // f16 with per-layer head_dim, cross-layer KV sharing (kv_source), and
  // sliding-window layers. cm_for_ maps a LoadedLanguageModel ContextId
  // to this model's ContextManager context (lazily acquire_root).
  ContextId cm_for_(ContextId lm_cid);

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;

  metal_compute::ComputeLibrary _lib_qmv, _lib_qmm, _lib_rms, _lib_elt,
      _lib_rope, _lib_sdpa, _lib_dense, _lib_dequant, _lib_dense_mma,
      _lib_sdpa_mma;
  // _fn_qmv / _fn_qmm run at the base quant_bits (attn / lm_head); the
  // _mlp / geglu variants run at mlp_bits (gemma4_unified MLP is 8-bit).
  metal_compute::ComputeFunction _fn_qmv, _fn_qmv_add, _fn_qmv_geglu,
      // e4b per-layer-input gate: gelu(qmv(plg_w,x))*pli[L] fused into the GEMV
      // write (folds the standalone geglu dispatch -- 1 fewer/layer at decode).
      _fn_qmv_gelu_mul,
      // QKV-fused decode GEMV (one full-occupancy matvec over the q|k|v
      // row-concat, base 4-bit only). Empty -> per-proj GEMV fallback.
      _fn_qmv_qkv,
      // input_layernorm RMSNorm fused into the QKV decode GEMV (one fewer
      // dispatch + no normed-x device round-trip). Empty -> separate rms+qkv.
      _fn_qmv_qkv_rms,
      // tied lm_head GEMV at embed_bits (8-bit on the GGUF path; aliases
      // _fn_qmv when embed_bits == base quant_bits).
      _fn_qmv_embed,
      _fn_qmv_mlp, _fn_qmm_mlp,
      // batched-decode GEMV (MAXM=2, N>2 tiles grid.z); _batch=base 4-bit
      // (attn/lm_head/PLE), _batch_geglu/_batch_mlp=mlp 4-bit (invalid for
      // the 12B 8-bit MLP -> steel GEMM).
      _fn_qmv_batch, _fn_qmv_batch_geglu, _fn_qmv_batch_mlp,
      _fn_qmm, _fn_qmm_geglu, _fn_transpose,
      _fn_rms, _fn_rms_add, _fn_rms_fast, _fn_rms_add_fast,
      _fn_rope, _fn_rms_rope, _fn_rms_rope2,
      _fn_rms_rope3, _fn_rms_rope3_kvwrite, _fn_geglu, _fn_softcap,
      // Mixture-of-Experts (Gemma-4 26B-A4B). Router softmax-topk-renorm
      // (_fn_moe_route, reused from Qwen), per-expert router-weight scale
      // (_fn_moe_expert_scale, gemma-specific), weighted partial combine
      // (_fn_moe_combine, reused). Expert gate|up gather (gelu-gated) + down
      // gather: affine (_fn_moe_gather_geglu / _fn_moe_gather_down, w4/w8 twins)
      // and the raw-HF dense f16 twins (_..._dense).
      _fn_moe_route, _fn_moe_expert_scale, _fn_moe_combine,
      _fn_moe_gather_geglu, _fn_moe_gather_down,
      _fn_moe_gather_geglu_dense, _fn_moe_gather_down_dense,
      // Bans a small set of token ids by masking their logits after softcap
      // (realtime thinking-channel suppression). See set_suppressed_tokens.
      _fn_suppress,
      _fn_scale, _fn_residual, _fn_copy, _fn_dummy_disp,
      // Native Q6_K tied embed/lm_head (GGUF path): reads the raw 6.5625-bit
      // table directly (lossless, ~25% smaller than an affine8 requant).
      // Bound + used only when an `embed_tokens.q6k` weight is present.
      _fn_embed_q6k, _fn_qmv_q6k,
      _fn_embed, _fn_kv_write, _fn_kv_write2, _fn_kv_write_sub,
      _fn_sdpa_causal, _fn_sdpa_window,
      _fn_sdpa_mma,
      // Device-direct contiguous (global/full-layer) prefill kernel: no tg K/V
      // staging + 3x larger key block, BIT-IDENTICAL to _fn_sdpa_mma (~1.3x).
      _fn_sdpa_mma_dev,
      // llama.cpp-style flash kernel (Q=8/C=64/key-split/parallel-softmax) for
      // the contiguous global layers: ~2.3x over _fn_sdpa_mma, as accurate but
      // NOT bit-identical (online-softmax reblock). Default global-prefill SDPA.
      _fn_sdpa_flash,
      _fn_sdpa_mb, _fn_dense_t,
      // Causal-tiled dense GEMM variants for the materialized global-attn
      // prefill: QK skips score tiles above the diagonal, PV caps the key
      // contraction at the row block's last query. ~halve each GEMM's work.
      _fn_dense_t_qkcausal, _fn_dense_t_pvcausal,
      // Dense f16 GEMV (M=1 decode) for the e4b PLE's two f16 projections
      // (per_layer_model_projection + per_layer_projection). dense_gemm_t is
      // a 32-row-tiled GEMM that runs at ~half DRAM bandwidth at M=1; this
      // reads W once at full bandwidth (qmv-style). VPIPE_GEMMA_NO_PLE_GEMV=1
      // reverts to dense_t for A/B.
      _fn_dense_gemv,
      _fn_argmax, _fn_row_scatter,
      // Two-stage parallel argmax (argmax_partial -> argmax_combine): M
      // threadgroups each reduce a contiguous vocab slab, a 1-tg combine picks
      // the global max. ~17x the single-tg _fn_argmax on the 262k vocab read in
      // isolation; token-exact (global lowest-index tie-break preserved). Used
      // by the greedy decode/pdecode paths; _fn_argmax is the small-V fallback.
      _fn_argmax_partial, _fn_argmax_combine,
      // GPU sampler for pdecode_* (top-k/top-p/min-p/penalties); greedy uses
      // _fn_argmax. From _lib_elt (dtype-correct), same kernel as Qwen.
      _fn_sample,
      // Histogram-based multi-tg sampler (the default; _fn_sample is the
      // VPIPE_GEMMA_SAMPLE1=1 single-tg fallback). Same semantics, ~3 multi-tg
      // vocab passes + a per-bin threshold scan instead of ~18 single-tg
      // rescans. See sample_*_f16 in llm_elementwise.metal.
      _fn_smp_max_partial, _fn_smp_max_combine,
      _fn_smp_zhist_partial, _fn_smp_zhist_combine,
      _fn_smp_thresh, _fn_smp_pick_partial, _fn_smp_pick_combine,
      // Flash-decode-GQA serial attention (contiguous KV, head_dim <= 256,
      // GQA G=Hq/n_kv(L) <= 4): read each KV head ONCE for all G query heads
      // + position-split merge, vs sdpa_mb's per-q-head re-scan. Reuses the
      // paged path's sdpa_gqa_merge_f16.
      _fn_sdpa_gqa, _fn_sdpa_gqa_merge,
      // Materialized paged decode for the global (head_dim 512) layers (the
      // omlx/MLX D=512 fallback: QK GEMV -> parallel softmax -> PV GEMV). M4-
      // neutral (scratch round-trip cancels it); kept default-off as the M5
      // matrix-core substrate. VPIPE_GEMMA_MAT_DECODE opts in.
      _fn_dec_qk, _fn_dec_rowstat, _fn_dec_pv, _fn_dec_merge,
      // Direct-read flash-decode for the GLOBAL (head_dim 512, full-context)
      // layers -- the decode bottleneck. Reads K/V straight from device memory
      // (no threadgroup staging/barriers, fast::exp), like MLX sdpa_vector_2pass:
      // the G q-head simdgroups share the kv-head's keys via L2 and per-key DRAM
      // latency is hidden by hardware ILP. Merged by sdpa_gqa_merge_f16.
      // _fn_sdpa_gqa_tile is the older threadgroup-staged variant (slower;
      // VPIPE_GEMMA_ATTN_STAGED=1 selects it for A/B via _gtile_staged).
      // _fn_sdpa_gqa_vec is the DEFAULT: same direct-read flash-decode but with
      // UK-key unrolling + vec4 K/V loads (bit-identical to _direct, ~2x faster
      // -- the global decode attn is latency-bound). Used when D % 128 == 0.
      _fn_sdpa_gqa_tile, _fn_sdpa_gqa_direct, _fn_sdpa_gqa_vec,
      // Linearized-ring sibling of _fn_sdpa_gqa_vec for BOUNDED sliding decode:
      // scans the trailing window as ONE contiguous physical span (no `% ring`
      // wrap), relying on the K/V mirror tail. Flag VPIPE_GEMMA_RING_LINEAR.
      _fn_sdpa_gqa_vec_lin,
      // MMA flash-DECODE for the global layers (8 q-heads/tile sharing one kv
      // head -> matrix-core QK/PV, no per-key simd_sum). ~1.7-2x the vec kernel
      // + flatter with depth. Used when 8|G and D%64==0; split scales with depth
      // (each split ~one 64-key block). Folded by _fn_sdpa_gqa_merge.
      _fn_sdpa_mma_qhead,
      // Matrix-core (M5+) matmul2d flash attention for the GLOBAL (head_dim
      // 512) PREFILL layers -- QK^T/PV on the hardware matrix units, ~1.7-1.9x
      // the simdgroup_matrix flash kernel (the prefill attention was the 12B
      // prefill gap vs llama.cpp). From the sdpa_mma metallib (metal4.0).
      _fn_sdpa_mma2, _fn_sdpa_mma2_d256,
      // PAGED K/V siblings for the FULL (global) layers when full_layers_paged
      // (the e4b VQA branch-sharing path): write K/V into the shared page pool
      // (kv_write_paged) and attend over it -- sdpa_paged_causal (scalar,
      // universal fallback), sdpa_paged_gqa_d512 + sdpa_gqa_merge (KV-split GQA
      // flash-decode, the fast full-layer decode), sdpa_paged_mma2_d512 (M5
      // matrix-core prefill).
      _fn_kv_write_paged, _fn_sdpa_paged_causal, _fn_sdpa_pgqa, _fn_sdpa_pmma2,
      // M4 / non-matrix-core paged GLOBAL-layer prefill (page-walked sibling of
      // sdpa_causal_flash_f16, D=512); fills the tier between _fn_sdpa_pmma2
      // (M5) and the scalar _fn_sdpa_paged_causal.
      _fn_sdpa_pflash,
      // Materialized GLOBAL (head_dim 512) prefill softmax: in-place causal-
      // masked softmax over a per-head materialized score matrix (the QK^T and
      // PV use the steel dense GEMM). Gated by VPIPE_GEMMA_MATERIALIZED_GLOBAL.
      _fn_causal_softmax,
      // Matrix-core (M5+) prefill GEMM: dequant a 4-bit weight to an f16
      // scratch, then dense matmul2d on the hardware matrix units.
      // _geglu_inter combines the interleaved gate|up dense output into
      // gelu(gate)*up so the matrix-core MLP stays token-exact with steel.
      _fn_dequant, _fn_dense_mma, _fn_dense_mma_deep, _fn_geglu_inter,
      _fn_dense_mma_qkcausal;

  // Matrix-core prefill GEMM state (M5+). _use_mma gates the dense
  // matmul2d path (4-bit checkpoint + supports_matrix_cores); _mma_min_m is
  // the row threshold below which the steel quantized GEMM wins (dequant
  // cost is M-independent). _w_deq is the reused dequantized-weight scratch.
  // _skip_dequant is a diagnostic A/B (VPIPE_GEMMA_MMA_NODQ). _mat_mma routes
  // the MATERIALIZED attention GEMMs (causal QK + full-K PV) onto the matrix
  // units (VPIPE_GEMMA_MAT_NO_MMA2 reverts to steel).
  bool _use_mma = false;
  bool _mat_mma = false;
  // Bounded-ring single-pass prefill (VPIPE_GEMMA_PREFILL_SUBBLOCK). When the
  // sliding ring is bounded (no grow), run the whole prompt in one forward
  // (large proj/FFN/global GEMM batch) and let the sliding attention read the
  // full-batch K/V scratch; only the ring KV-write stays page-bounded.
  bool _prefill_subblock = true;
  // Materialized banded sliding prefill enabled (VPIPE_GEMMA_MAT_SLIDING != 0).
  // The ring-independent (kt/vt-reading) sliding attention; the bounded
  // single-pass prefill above requires it, else a bounded ring would be read
  // wrapped. Cached at construction so prefill() and forward_chunk_ agree.
  bool _mat_sliding = true;
  int  _mma_min_m = 64;
  bool _skip_dequant = false;
  metal_compute::SharedBuffer _w_deq;

  // Unquantized dense (raw-HF bf16/f16) checkpoint: the linear weights ship as
  // plain [out,in] .weight tensors (no affine .scales/.biases). Detected in
  // load() by probing a representative gate_proj dtype; when set, every
  // projection matmul routes the dense f16 GEMM/GEMV (dense_gemm_t / dense_gemv
  // / embed_gather_f16) instead of the affine qmv/qmm kernels. Gated so the
  // quantized e4b / gemma4_unified paths are untouched.
  bool _dense = false;
  // Max per-layer MLP intermediate (>= ffn_inner; 2x on the E2B double-wide
  // shared layers). Sizes the shared MLP decode/prefill scratch.
  int  _ffn_max = 0;
  metal_compute::SharedBuffer _embed_dense;                     // dense tied embed
  metal_compute::ComputeFunction _fn_embed_dense;               // embed_gather_f16

  std::vector<Layer> _layers;
  metal_compute::SharedBuffer _embed_w, _embed_s, _embed_b;      // tied lm_head
  // Raw Q6_K tied embed/lm_head table (GGUF path); when set, the affine8
  // _embed_w/s/b are NOT loaded and the q6k kernels are used instead.
  metal_compute::SharedBuffer _embed_q6k;
  bool                        _embed_is_q6k = false;
  metal_compute::SharedBuffer _ple_w, _ple_s, _ple_b;            // PLE embed
  // per_layer_model_projection ships BF16 (not quantized) -> f16 dense, same
  // bytes as omlx (no quant opportunity, unlike per_layer_projection/plp).
  metal_compute::SharedBuffer _plm_proj_w;                       // f16 dense
  metal_compute::SharedBuffer _ple_proj_norm;                    // [hpli]
  metal_compute::SharedBuffer _final_ln;                         // [hidden]
  metal_compute::SharedBuffer _inv_freq_sliding, _inv_freq_full; // rope tables
  metal_compute::SharedBuffer _ones_vnorm;                       // [head_dim_full]

  std::unique_ptr<ContextManager>            _ctx;     // Contiguous KV
  std::unordered_map<std::uint32_t, ContextId> _ctxmap; // LM cid -> CM cid
  int                                        _sliding_chunk = 0;  // ring B
  bool                                       _sliding_grow = true; // lazy 1-pass
  bool                                       _ple_gemv = true;     // M=1 PLE GEMV
  bool                                       _ple_quant = true;    // 4-bit PLE
  bool                                       _qkv_fuse = true;     // fused QKV GEMV (~1% GPU win)
  bool                                       _qkv_rms_fuse = false; // input_norm+QKV fusion (opt-in; slower)
  bool                                       _rope_kv_fuse = true;  // rope3 + ring KV-write fused
  bool                                       _rms_fast = true;      // simd_sum norms (default; RMS_FAST=0 -> tree)

  // Reused decode scratch (one token's worth, sized for head_dim_full).
  bool _scratch_ready = false;
  metal_compute::SharedBuffer _d_x, _d_hn, _d_q, _d_k, _d_v, _d_attn, _d_o,
      _d_act, _d_mlp, _d_ple, _d_pleproj, _d_pli, _d_plg,
      _d_plp, _d_logits, _d_tok;
  // Dense-path decode scratch: the raw-HF gate output [ffn] (the affine path
  // fuses gate|up+geglu into _d_act, but dense runs two separate GEMVs). Only
  // allocated when _dense.
  metal_compute::SharedBuffer _d_gate;
  // MoE decode scratch (M=1, iff _cfg.is_moe()). logits[E], ids[top_k] int32,
  // w[top_k], act[top_k*moe_inner], part[top_k*H], out[H] (combined MoE), h1[H]
  // (dense-branch normed). The dense-branch MLP reuses _d_hn/_d_act/_d_mlp; the
  // MoE-branch pre-norm reuses _d_hn after the dense branch consumes it.
  metal_compute::SharedBuffer _d_moe_logits, _d_moe_ids, _d_moe_w, _d_moe_act,
      _d_moe_part, _d_moe_out, _d_moe_h1;
  metal_compute::SharedBuffer _d_argmax_id;   // [1] int32, GPU argmax out
  // Suppressed (banned) token ids: their _d_logits entries are masked to a
  // far-negative sentinel after softcap (suppress_logits_f16) so they are
  // never predicted. Tiny ([n] int32); set by set_suppressed_tokens.
  metal_compute::SharedBuffer _d_suppress_ids;
  int                         _n_suppress = 0;
  // [2*kArgmaxM] f32 partials (val,idx) for the two-stage argmax. Allocated
  // once in ensure_scratch_(); only used when _fn_argmax_partial is valid.
  metal_compute::SharedBuffer _d_argmax_part;
  static constexpr int kArgmaxM = 64;         // threadgroups in stage 1
  // Histogram-sampler scratch (multi-tg sample_*_f16). kSampM = stage-1
  // threadgroups; kSampB MUST match the kSampB in llm_elementwise.metal.
  // _d_smp_maxpart [M] f32 max partials; _d_smp_hpart [M*(2B+1)] f32 per-tg
  // (count,mass) histograms + partial Z; _d_smp_hist [2B+1] f32 combined;
  // _d_smp_maxl/_d_smp_wt [1] f32; _d_smp_pickpart [2M] f32 (score,idx). All
  // shared across pdecode steps (GPU-serialized by the per-context event).
  static constexpr int kSampM = 64;
  static constexpr int kSampB = 1024;
  metal_compute::SharedBuffer _d_smp_maxpart, _d_smp_hpart, _d_smp_hist,
      _d_smp_maxl, _d_smp_wt, _d_smp_pickpart;
  metal_compute::SharedBuffer _d_dummy;       // [256] f16 scratch for DUMMY_DISP
  // Materialized GLOBAL-prefill scratch (VPIPE_GEMMA_MATERIALIZED_GLOBAL):
  // _d_scores = f16 [CAP, CAP] per-head score matrix; _d_vT = f16 [Hkv,D,CAP]
  // V transpose. CAP = min(max_seq, 8192). Allocated only when the flag is on.
  metal_compute::SharedBuffer _d_scores, _d_vT;
  // Pipelined PLE production (Design A): split _d_pli into per-chunk buffers so
  // chunks 1..N-1 overlap the layer chain's bubbles (only chunk 0 gates layer
  // 0). _d_x_ple is a private snapshot of _d_x so the chunk GEMVs don't WAR with
  // the per-layer residual writes. _ple_chunk_k = layers/chunk (0 = disabled).
  std::vector<metal_compute::SharedBuffer> _d_pli_ch, _d_pleproj_ch;
  metal_compute::SharedBuffer _d_x_ple;
  int _ple_chunk_k = 0;
  metal_compute::SharedBuffer _d_probe_in, _d_probe_out;  // concurrency probe
  // Page table {page_id,n_valid,global_start} triplets for the FULL-layer paged
  // attention (full_layers_paged); refilled per step/chunk. kPgtabSlots tables
  // ring so run-ahead pdecode (depth<=4) doesn't clobber an in-flight table.
  static constexpr int kPgtabSlots = 5;       // max pdecode depth (4) + 1
  metal_compute::SharedBuffer _d_pgtab;
  // Flash-decode-GQA (contiguous KV) state + f32 partials O [Hq,split,D],
  // m/l [Hq,split]. Default ON when capable; VPIPE_GQA_ATTN=0/1 overrides,
  // VPIPE_GQA_SPLIT sets the split. Per-layer gated on head_dim<=256 + G<=4.
  bool _gqa_attn = false;
  int  _gqa_split = 32;
  metal_compute::SharedBuffer _d_gqa_oacc, _d_gqa_m, _d_gqa_l;
  // Materialized global decode (VPIPE_GEMMA_MAT_DECODE): [Hq, Tstride] f32
  // scores scratch (Tstride = max_pages*page_tokens); PV partials/m/l reuse
  // _d_gqa_oacc/_d_gqa_m/_d_gqa_l. _dec_tstride is the per-head row stride.
  bool _mat_decode = false;
  int  _dec_tstride = 0;
  metal_compute::SharedBuffer _d_dec_scores;

  // Per-context pipelined-decode state (pdecode_*). The token chain
  // (gen_ids) + sampler workspace stay GPU-resident; one command stream +
  // event chains the per-token forwards. Contiguous KV (no page table),
  // so simpler than the Qwen PDecode (no pgt / append-slot).
  //
  // Run-ahead pipeline (lever #2): up to `depth` command buffers in flight,
  // so the CPU encodes token N+1 while the GPU executes token N. The event
  // chain + Tracked-buffer hazard tracking serialize GPU execution, so the
  // shared decode scratch is reused in order (only CPU-encode overlaps).
  // depth=1 reproduces the original commit/next behavior exactly. depth>1
  // run-ahead drops the stop-token contract (it commits a token's KV before
  // the host has seen it) -- valid only for fixed-length / benchmark decode.
  struct PDecode {
    metal_compute::SharedBuffer  gen_ids;     // [cap] int32 GPU chain
    metal_compute::SharedBuffer  seen;        // [vocab] uint8 (sampled)
    metal_compute::SharedBuffer  sample_ws;   // [vocab] f16  (sampled)
    metal_compute::CommandStream stream;
    metal_compute::Event         ev;
    struct InFlight {
      metal_compute::CommandStream::Fence fence;
      int idx = -1;                            // gen_ids slot it writes
    };
    std::deque<InFlight> ring;                  // FIFO of in-flight steps
    GpuSamplerParams sp;
    ContextId     cm;                          // mapped ContextManager ctx
    int           cap        = 0;              // gen_ids slot capacity
    int           produced   = 0;             // gen_ids[0..produced-1] final
    int           committed  = 0;             // next out_idx to assign
    int           depth      = 1;             // run-ahead pipeline depth
    std::uint64_t gpu_step   = 0;             // event signal counter
  };
  std::unordered_map<std::uint32_t, PDecode> _pdec;

  // Direct-read flash-decode for the GLOBAL layers -- DEFAULT ON when present.
  // Those D=512 full-context layers are the whole decode gap.
  // VPIPE_GEMMA_GTILE_ATTN=0 reverts them to sdpa_mb; VPIPE_GEMMA_GTILE_SPLIT
  // sets the key-range split (more threadgroups; scratch sizes to it).
  bool _gtile_attn = false;
  bool _gtile_staged = false;   // VPIPE_GEMMA_ATTN_STAGED: old staged kernel
  bool _gtile_direct = false;   // VPIPE_GEMMA_ATTN_DIRECT: scalar direct (A/B)
  // VPIPE_GEMMA_RING_LINEAR: read the bounded sliding window as ONE contiguous
  // physical span (mirror-tail linearized ring) -- no `% ring_cap` in the
  // decode SDPA hot loop. Default OFF. Mirror alloc/writes are always-on.
  bool _ring_linear = false;
  int  _gtile_split = 128;       // CAP on the per-layer position-split
  // Adaptive per-layer KV split: sp = clamp(ceil(scan_keys / _gtile_kps), 1,
  // _gtile_split). The global (full-context, ~2k+ keys) and sliding (512-key
  // window) layers want very different splits -- a fixed split over-splits the
  // tiny sliding window (sp=64 over 512 keys = 8 keys/split + a 64-partial
  // merge that reads more than the scan). Keys-per-split ~64 => global sp~32,
  // sliding sp~16. VPIPE_GEMMA_GTILE_KPS overrides; =0 restores the fixed split.
  // 16 keeps the 512-key sliding window at sp=32. Global is CAP-limited: at
  // depth >= kps*_gtile_split keys, sp pins to the cap, so the cap sets the
  // global keys/split. The cap was 64 (set when only <=4k was tested); the
  // LONG-CONTEXT decode is LATENCY/OCCUPANCY-bound (NOT bandwidth -- the mb256
  // KV-read-once form is ~40% SLOWER at 16k), so MORE split-parallelism wins as
  // depth grows. M4 Pro re-sweep raising the cap to 128: pipe decode 4k 60.7->
  // 60.5 (neutral), 8k 56.7->57.4 (+1.2%), 16k 49.8->51.8 (+4.2%); 256 over-
  // splits (16k 51.2 < 128's 51.8). Sliding is unaffected (sp=32 < cap either
  // way). Token-exact (split count changes only the f32 merge order).
  int  _gtile_kps = 16;
  // MMA q-head flash-decode for global layers (default ON when available + 8|G).
  // VPIPE_GEMMA_NO_MMA_QHEAD=1 forces the vec kernel for A/B. _mma_qhead_cap is
  // the max KV-split (partials buffer sizing); split = ceil(scan/64) at runtime.
  bool _mma_qhead = true;
  int  _mma_qhead_cap = 256;
  // Matrix-core matmul2d attention for GLOBAL (head_dim 512) PREFILL layers
  // (default ON when the GPU has matrix cores + the kernel loaded).
  // VPIPE_GEMMA_NO_MMA2_ATTN=1 reverts to the simdgroup_matrix flash kernel.
  bool _mma2_attn = false;
  // FULL (global) layers routed through the shared PAGED pool (no per-branch
  // KV copy) -- mirrors ContextManager::full_layers_paged(). _pmma2_attn: the
  // matrix-core sdpa_paged_mma2_d512 loaded (M5; used for the paged prefill).
  bool _full_paged = false;
  bool _pmma2_attn = false;
  // M4 / non-matrix-core paged-prefill flash (sdpa_paged_flash_d512) loaded:
  // the simdgroup_matrix flash tier used for bulk global paged prefill when
  // _pmma2_attn is off (no matrix cores). VPIPE_GEMMA_NO_PFLASH=1 reverts to
  // the scalar sdpa_paged_causal.
  bool _pflash_attn = false;
  // MATERIALIZED GLOBAL (head_dim 512) prefill path (VPIPE_GEMMA_MATERIALIZED_
  // GLOBAL=1; default OFF). Mirrors MLX's head_dim-512 fallback: per-head
  // Q.K^T (steel GEMM) -> causal softmax (_fn_causal_softmax) -> P.V (steel
  // GEMM), replacing the inefficient fused sdpa_paged_flash_d512 for the
  // single-chunk (qpos==0) global prefill only. Capped at T_kv<=_scores_cap.
  bool _materialized_global = false;
  int  _scores_cap = 0;       // CAP = min(max_seq, 8192); 0 until scratch ready
  // KV-split GQA paged decode (sdpa_paged_gqa_d512 + merge) available -- the
  // fast full-layer decode path; falls back to scalar sdpa_paged_causal when
  // off / G>SDPA_GQA_MAXG / partials absent. VPIPE_GEMMA_NO_PGQA=1 reverts.
  bool _pgqa_attn = false;

  // Decode category profiler (VPIPE_GEMMA_CATPROF set at load): the decode
  // step reads VPIPE_GEMMA_DUP_CAT per step and doubles that category's GPU
  // work, so a within-process A/B isolates per-category cost at a steady
  // clock. Off in production (no per-step getenv). See encode_step_.
  bool _catprof = false;
};

}  // namespace vpipe::genai

#endif
