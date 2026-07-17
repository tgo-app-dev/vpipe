#ifndef VPIPE_GENERATIVE_MODELS_METAL_QWEN_MODEL_H
#define VPIPE_GENERATIVE_MODELS_METAL_QWEN_MODEL_H

// MetalQwenModel -- procedural Qwen3.5 forward pass on the metal-compute
// framework (no MLX in the forward), the Qwen counterpart of
// MetalLlamaModel. Qwen3.5 is a HYBRID transformer: layers with
// L % full_attn_interval == (full_attn_interval-1) are full attention
// (GQA, head_dim 256, QK-norm, partial RoPE, output gate); the rest are
// gated-DeltaNet (GDN) linear-attention layers (depthwise conv1d + SiLU,
// q/k L2-norm, delta-rule recurrence over a per-context ssm_state,
// gated-RMSNorm output). MLP is plain SwiGLU on every layer. Embeddings
// are 4-bit affine quantized and (4B) tied to lm_head.
//
// v1: correctness-first decode (n=1) -- prefill loops forward() over the
// prompt. Reuses MetalLlamaWeights (name-agnostic loader), MetalTokenMuxer
// (quantized embed gather), and ContextManager configured with its
// metal-compute backend (Spec::metal != nullptr -> paged KV + GDN
// conv/ssm state in SharedBuffers). Every kernel it calls is independently
// oracle-verified in tests/unit-tests/metal-compute-llm-ops.cc.

#include "generative-models/context-manager.h"   // ContextId, ContextManager
#include "generative-models/metal-token-muxer.h"
#include "generative-models/model-exec.h"         // GpuSamplerParams
#include "generative-models/shared/kernel-autotune.h"  // TuningReport
#include "generative-models/shared/kernel-sets/decode-gqa-attn-set.h"
#include "generative-models/shared/kernel-sets/prefill-gqa-attn-set.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/event.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <deque>
#include <unordered_map>
#include <vector>

namespace vpipe::metal_compute { class MetalCompute; class ComputeEncoder; }

namespace vpipe::genai {

struct ModelConfig;
class I8GemmContext;   // fwd (shared/i8-gemm.h)

// k-quant family of a NATIVE GGUF weight (no requant). kNone => the weight
// is on the affine (safetensors / gemma-style) path. The metal forward
// dispatches the matching qmv (decode) / dequant-to-f16 + dense GEMM
// (prefill) per tensor; q4_K and q5_K share get_scale_min_k4, q6_K is its
// own block format. See gguf-convert GgufQwen35Converter.
enum class KQ : std::uint8_t { kNone = 0, kQ4K, kQ5K, kQ6K };

// Streaming / stop / position control for MetalQwenModel::mtp_decode. All
// fields optional; the defaults reproduce the original generate-to-budget,
// sequential-rope, no-stop behavior used by the perf smoke test. (Defined at
// namespace scope, not nested, so it can serve as a `= {}` default argument.)
struct MtpDecodeCtl {
  // RoPE position of `first_token` (the first decoded token). -1 => the
  // sequential text path (rope == KV slot). >= 0 => the mROPE-advanced
  // position after a multimodal prefill: the verify offsets every token's rope
  // by the SAME (rope_first - seq_len) delta so MTP reproduces the serial
  // decode path's positions (and stays token-exact vs it).
  int rope_first = -1;
  // End-of-turn predicate. When an accepted token satisfies it, generation
  // stops and that token is NOT kept/streamed (its speculative KV/GDN is rolled
  // back), exactly matching the serial stop-token loop. Empty => the decode
  // never stops on a token (budget / abort only).
  std::function<bool(std::int32_t)> is_stop;
  // Per-round callback with this round's newly accepted (non-stop) token ids,
  // for streaming. Return false to abort after the round. Empty => run to
  // budget. The span is valid only for the duration of the call.
  std::function<bool(std::span<const std::int32_t>)> on_round;
  // Set true iff a stop token (is_stop) ended generation (vs budget/abort).
  bool* hit_stop = nullptr;
  // Sampler for the VERIFY (speculative sampling). greedy (default) accepts a
  // draft iff it == argmax(P); a non-greedy sampler accepts iff it == a token
  // SAMPLED from P at that position (per-position seed). Either way every
  // committed token IS the verifier's pick from the correctly-conditioned P, so
  // the output distribution is exactly autoregressive sampling from the main
  // model -- the drafter (MTP head, still argmax) only affects acceptance rate,
  // not the tokens. (Penalties/seen-set are NOT applied in the verify; the
  // caller restricts the sampled path to no-penalty configs.)
  GpuSamplerParams sampler{};
};

class MetalQwenModel {
public:
  ~MetalQwenModel();   // out-of-line (unique_ptr members of fwd types)

  struct Config {
    int   n_layers   = 32;
    int   hidden     = 2560;
    int   n_heads    = 16;
    int   n_kv_heads = 4;
    int   head_dim   = 256;
    int   ffn_inner  = 9216;
    int   vocab      = 248320;
    float rope_theta = 1.0e7f;
    float rms_eps    = 1e-6f;
    int   rotary_dim = 64;     // floor(head_dim * partial_rotary_factor)
    int   full_attn_interval = 4;   // L % interval == interval-1 -> full
    // Gated-DeltaNet dims (shared by 4B and 9B).
    int   gdn_conv_kernel = 4;
    int   gdn_conv_dim    = 8192;   // 2*key_dim + value_dim
    int   gdn_k_heads     = 16;
    int   gdn_v_heads     = 32;
    int   gdn_k_dim       = 128;
    int   gdn_v_dim       = 128;
    bool  tie_embeddings  = true;
    // Mixture-of-Experts (Qwen3.5-MoE). n_experts > 0 replaces every
    // layer's dense MLP with a SparseMoeBlock: softmax router (mlp.gate,
    // w8) over all n_experts -> top_k -> score-weighted sum of the
    // selected experts' MLPs (mlp.switch_mlp, batched 3D w4, inner=
    // moe_inner) + a sigmoid-gated always-on shared expert
    // (mlp.shared_expert, dense w4, inner=moe_shared_inner). The
    // GDN+full-attn backbone is unchanged. 0 for dense / non-MoE models.
    int   n_experts        = 0;       // total experts (256)
    int   top_k            = 0;       // num_experts_per_tok (8)
    int   moe_inner        = 0;       // per-expert intermediate (512)
    int   moe_shared_inner = 0;       // shared-expert intermediate (512)
    bool  moe_norm_topk    = true;    // renorm top-k weights
    bool  is_moe() const { return n_experts > 0; }
    // Element/compute dtype: false = f16 (default), true = bf16. bf16
    // loads the *_bf16 kernel metallibs and keeps the checkpoint's bf16
    // scales/norms/conv raw (no load-time conversion).
    bool  use_bf16    = false;
    // Quantization bit-width of the linear weights: 4 (Qwen3.5/Llama) or
    // 8 (Qwen3-ASR). Selects the affine_qmv/affine_qmm_steel w4g64 vs
    // w8g64 kernel entry points and the packed-weight row stride.
    int   quant_bits  = 4;
    // Dense mode: every layer is full-attention (no gated-DeltaNet
    // layers). Used by the Qwen3-ASR text decoder, a plain Qwen3 dense
    // transformer.
    bool  dense       = false;
    // Whether the RAW (unquantized, dense) checkpoint stores zero-centered
    // RMSNorm weights -- i.e. the model applies (1+weight) and vpipe must add
    // 1.0 to every loaded norm at bind. True for the raw-HF Qwen3.5 family
    // (the default). FALSE for plain-Qwen3 backbones that use standard
    // RMSNorm (weight*h directly, ones-init), e.g. MOSS-TTS-Local-v1.5's
    // dense Qwen3 backbone -- adding 1 there corrupts the logits. Only
    // consulted on the dense (raw bf16/f16) path; quantized loads fold the
    // offset at quantize time, so this has no effect there.
    bool  zero_centered_norm = true;
    // Whether q_proj carries a per-head output gate (Qwen3.5: q_proj
    // emits 2*qd, the second half gates the attention output).
    // Qwen3-ASR has no gate (q_proj emits qd).
    bool  attn_output_gate = true;
    // Backbone-only load: skip the token-embedding muxer + lm_head (and the
    // tied/affine embed table). For host-fed-embedding models that supply
    // their own embeddings and output heads (MOSS-TTS: 33 bf16 code/text
    // heads + 32 audio-code embeddings on top of a dense Qwen3 backbone).
    // forward_embeddings_hidden() is the only forward path in this mode.
    bool  backbone_only = false;
    // Streaming-calibration load (MoE only): skip the per-layer weight load in
    // load() entirely (leave _layers sized + is_full set, weights EMPTY) so the
    // 35B never resides whole. calib_build_layer/calib_run_layer/
    // calib_free_layer then stream ONE layer at a time over a held residual.
    // Implies backbone_only behaviour (no lm_head). No effect on any non-calib
    // forward (the per-layer weights are simply never populated).
    bool  calib_stream = false;
    // Weight-name root prepended before "model." / "lm_head." Qwen3.5-VL
    // nests the LM under "language_model."; Qwen3-ASR is at the root ("").
    std::string weight_prefix = "language_model.";
    // Segment between weight_prefix and "layers."/"embed_tokens"/"norm".
    // Default "model." gives the HF "<prefix>model.layers.N" convention.
    // MOSS-TTS-Local-v1.5 nests the backbone under "transformer." with NO
    // "model." segment, so it sets weight_prefix="transformer." + model_seg="".
    std::string model_seg = "model.";
    int   max_seq     = 512;
    int   page_tokens = 256;
    int   max_pages   = 0;
    // mROPE section split (T/H/W) over rotary_dim/2 pairs; default
    // Qwen3.5 [11,11,10]. Only used by the multimodal (image) prefill.
    std::vector<int> mrope_section = {11, 11, 10};

    int key_dim()   const { return gdn_k_heads * gdn_k_dim; }   // 2048
    int value_dim() const { return gdn_v_heads * gdn_v_dim; }   // 4096
    int qd()        const { return n_heads * head_dim; }        // 4096
    int kd()        const { return n_kv_heads * head_dim; }     // 1024
    bool is_full_attn(int L) const {
      return (L % full_attn_interval) == (full_attn_interval - 1);
    }
    // Dense models have no linear-attention layers -- every layer is
    // full attention.
    bool layer_is_full(int L) const { return dense || is_full_attn(L); }
  };

  static std::unique_ptr<MetalQwenModel> load(
      const std::string& model_dir,
      metal_compute::MetalCompute* mc,
      const Config& cfg);

  // Derive the shape-affecting Config from a parsed ModelConfig
  // (config.json). The single source of truth for mapping the HF
  // config onto this backend, shared by the production loader and the
  // tests so any model in the family (4B/9B/...) self-configures
  // instead of relying on the 4B-shaped struct defaults. Does NOT set
  // the runtime-policy fields (use_bf16, page_tokens, max_pages); the
  // caller layers those on top.
  static Config config_from(const ModelConfig& c);

  ContextManager* context_manager() { return _ctx.get(); }
  // The model's own root context (the one the cid-less forward()/prefill()/
  // decode_pipelined() overloads drive). Exposed so tests can branch() it.
  ContextId root_context() const { return _cid; }
  // Decode one token. rope_pos < 0 uses the KV slot position (text); a
  // non-negative value overrides the full-attn RoPE position (post-image
  // decode, where mROPE positions diverge from KV slots).
  std::vector<float> forward(ContextId cid, std::int32_t token_id,
                             int rope_pos = -1);
  std::vector<float> forward(std::int32_t token_id) {
    return forward(_cid, token_id);
  }
  // Multimodal prefill: pre-spliced embeddings (text + image) as
  // [n*hidden] row-major f32, and 3-axis mROPE position_ids as [3*n]
  // (row 0=T, 1=H, 2=W). Returns [vocab] logits at the last position.
  std::vector<float> prefill_multimodal(
      ContextId cid, const std::vector<float>& embeddings,
      const std::vector<std::int32_t>& position_ids);

  // Prefill a pre-assembled [n*hidden] row-major f32 embedding stream
  // (e.g. text rows spliced with audio-encoder rows for Qwen3-ASR) using
  // plain 1-D RoPE at sequential positions. Returns [vocab] logits at
  // the last position. Empty on failure.
  std::vector<float> prefill_embeddings(
      ContextId cid, const std::vector<float>& embeddings, int n);

  // Per-token text embeddings [n*hidden] f32 (quantized embed gather),
  // for assembling the multimodal spliced embedding stream (text rows
  // here, image rows from the vision encoder).
  std::vector<float> embed_text(const std::vector<std::int32_t>& ids);

  // ---- Native-f16 zero-copy splice ---------------------------------
  // Text embeddings for `ids` as a [k*hidden] SharedBuffer in the
  // compute dtype (f16) -- the embed-table gather WITHOUT the host-f32
  // readback embed_text() does. Empty on failure.
  metal_compute::SharedBuffer
  embed_text_buf(const std::vector<std::int32_t>& ids);

  // prefill_multimodal / prefill_embeddings, but taking a pre-assembled
  // [n*hidden] SharedBuffer already in the compute dtype (f16) -- skips
  // the host-f32 -> GPU upload + cast. Returns [vocab] logits at the
  // last position; empty on failure.
  std::vector<float> prefill_multimodal_buf(
      ContextId cid, metal_compute::SharedBuffer&& x,
      const std::vector<std::int32_t>& position_ids, int n);
  std::vector<float> prefill_embeddings_buf(
      ContextId cid, metal_compute::SharedBuffer&& x, int n);

  // Backbone forward over a pre-assembled [n*hidden] f16 embedding stream
  // that returns the LAST position's FINAL-NORMED hidden as a [hidden]-row
  // f16 SharedBuffer (GPU-resident) instead of lm_head logits -- the seam
  // for models that own their output heads (MOSS-TTS applies 33 code/text
  // heads to this hidden). Appends to cid's paged KV exactly like prefill,
  // so calling it with n=1 per step IS the decode loop. Empty on failure.
  metal_compute::SharedBuffer
  forward_embeddings_hidden(ContextId cid,
                            const metal_compute::SharedBuffer& x, int n);

  // Backbone forward over a pre-assembled [n*hidden] compute-dtype embedding
  // stream that captures the per-layer hidden states (the un-normed residual
  // after each requested 0-indexed layer) instead of logits -- the seam for
  // the Krea-2 text encoder, which conditions the DiT on a stack of tapped
  // Qwen3-VL layers. Returns a [tap_layers.size()*n*hidden] compute-dtype
  // SharedBuffer laid out [tap_slot][pos][hidden] (slot j = tap_layers[j]).
  // Plain 1-D RoPE at sequential positions; skips the lm_head. Empty on
  // failure. HF output_hidden_states[k] == the tap after layer k-1.
  metal_compute::SharedBuffer
  forward_embeddings_taps(ContextId cid, const metal_compute::SharedBuffer& x,
                          int n, const std::vector<int>& tap_layers);

  // ---- On-device AWQ calibration ------------------------------------------
  // Accumulate per-input-channel running |x| abs-max for the qkv / gate-up /
  // down Linear inputs across every prefill chunk run while enabled. The taps
  // sit in the verified forward (guarded; no effect when off) -- so this is
  // the model's REAL activation distribution, streamed layer-by-layer. While
  // enabled, forward_embeddings_hidden runs the full MLP on every layer (no
  // last-layer prune) so the deepest layer's stats cover all positions. Used
  // to replace the offline HF calib script; feeds ModelQuantizer's AWQ folds.
  // Dense full-attention models only (the v1.5 backbone); GDN/MoE not tapped.
  void calib_begin();
  void calib_end() { _calib_on = false; }
  bool calibrating() const { return _calib_on; }
  // [n_layers][channels] abs-max (hidden for qkv/gateup, ffn_inner for down).
  const std::vector<std::vector<float>>& calib_qkv()    const { return _calib_qkv; }
  const std::vector<std::vector<float>>& calib_gateup() const { return _calib_gu; }
  const std::vector<std::vector<float>>& calib_down()   const { return _calib_dn; }

  // ---- Streaming per-layer MoE calibration (calib_stream load) ------------
  // The memory-safe layer-by-layer calibration forward: the MoE counterpart of
  // the calib_begin()/forward_embeddings_hidden() full-model path. The model is
  // loaded with Config::calib_stream so _layers carry no weights; the streaming
  // orchestrator (collect_backbone_calibration_streaming) then, for each layer
  // L: calib_build_layer (load ONLY layer L's weights into _layers[L]),
  // calib_run_layer (run layer L's forward over the held residual stream,
  // advancing it in place + tapping the per-input-channel |activation| incl.
  // PER-EXPERT gate/up + down stats), calib_free_layer (drop _layers[L]). Peak
  // resident stays ~one layer's weights. calib_begin_streaming sizes the
  // accumulators; the per-expert stats are exposed below.
  void calib_begin_streaming();
  bool calib_build_layer(const class MetalLlamaWeights& wts, int L,
                         std::uint64_t* bytes_out, std::string* err);
  // resid[s] is sequence s's [seq_lens[s]*hidden] f16 residual stream, advanced
  // in place by layer L. Reuses _ctx for the (transient, per-seq) KV / GDN
  // conv+ssm state. Taps into the accumulators. Returns false + *err on failure.
  bool calib_run_layer(int L, std::vector<metal_compute::SharedBuffer>& resid,
                       const std::vector<int>& seq_lens, std::string* err);
  void calib_free_layer(int L);
  // [n_layers][n_experts*hidden] per-expert routed gate/up input abs-max and
  // [n_layers][n_experts*moe_inner] per-expert down input abs-max.
  const std::vector<std::vector<float>>& calib_expert_gateup() const {
    return _calib_eg;
  }
  const std::vector<std::vector<float>>& calib_expert_down() const {
    return _calib_ed;
  }

  // OPTIMIZED single-token decode from a pre-computed [hidden] embedding row
  // (the MOSS-TTS summed embedding), in the compute dtype. Reuses the
  // pre-allocated decode scratch + qmv kernels (the encode_decode_step_
  // machinery) instead of forward_embeddings_hidden's prefill GEMM at M=1.
  // Appends one position to cid's paged KV and leaves the final-normed hidden
  // in the returned [hidden] buffer (a pointer to internal scratch, valid
  // until the NEXT decode call on this model). nullptr on failure. rope_pos<0
  // uses the KV slot position.
  // `post_hidden` (optional): appended to the SAME command buffer after the
  // final norm leaves the hidden in the returned buffer, so a caller can fuse
  // its own dispatches (MOSS-TTS' output heads) into the decode -- one
  // commit+wait instead of two. It receives the encoder + the [hidden] buffer.
  const metal_compute::SharedBuffer*
  decode_embedding_hidden(
      ContextId cid, const metal_compute::SharedBuffer& emb, int rope_pos = -1,
      const std::function<void(metal_compute::ComputeEncoder&,
                               const metal_compute::SharedBuffer&)>*
          post_hidden = nullptr);

  // ---- Fused whole-frame decode support (MOSS-TTS-Realtime depth loop) ----
  // The decode-input residual buffer (compute dtype [hidden]). A caller fusing
  // many decode steps into ONE command buffer writes the next step's input
  // here -- e.g. an embed-gather kernel targeting this buffer -- instead of the
  // host copy decode_embedding_hidden does, then calls encode_decode_prewritten,
  // so the whole frame stays on-GPU in a single command buffer. Allocates the
  // decode scratch on first call; empty on failure.
  metal_compute::SharedBuffer& decode_input_buffer();
  // Encode ONE decode step into a caller-owned encoder WITHOUT committing,
  // reading decode_input_buffer() AS-IS (the caller already populated it this
  // step -- host memcpy or a prior GPU dispatch in the same buffer). Appends a
  // KV slot and returns the final-normed hidden (&_d_hn, valid until the next
  // step/commit). The caller owns the command buffer + commit, so multiple
  // steps + custom dispatches (heads, on-GPU sample, embed-gather) fuse into
  // one buffer. nullptr on failure. rope_pos<0 uses the KV slot position.
  const metal_compute::SharedBuffer*
  encode_decode_prewritten(metal_compute::ComputeEncoder& enc, ContextId cid,
                           int rope_pos = -1);
  // Prefill the prompt by looping forward() over the ids (v1 has no
  // batched GEMM path yet); returns the [vocab] logits at the last
  // position. Empty on failure.
  std::vector<float> prefill(ContextId cid,
                             const std::vector<std::int32_t>& ids);
  std::vector<float> prefill(const std::vector<std::int32_t>& ids) {
    return prefill(_cid, ids);
  }
  std::int32_t forward_argmax(std::int32_t token_id);

  // Greedy fast decode: one command buffer that gathers the input token's
  // embedding in-stream (no separate muxer command buffer + host memcpy),
  // runs the decode step, and computes the argmax on-GPU -- returning a
  // single token id WITHOUT pulling the full [vocab] logits to host or
  // scanning them. Same forward math as forward(); byte-identical argmax
  // (argmax_f16 casts to f32 + tie-breaks by lowest index). For greedy
  // callers that don't need last logits (sampling still uses forward()).
  // Returns INT32_MIN (before touching any state) if the embed/argmax
  // kernels are unavailable, so the caller can fall back; -1 on a genuine
  // (post-append) error.
  std::int32_t decode_step_fast(ContextId cid, std::int32_t token_id,
                                int rope_pos = -1);

  // Pipelined decode (the Qwen counterpart of MetalLlamaModel::
  // decode_pipelined). Generates `n_steps` tokens from `first_token`,
  // appending to `cid` (paged KV for full-attn layers, in-place conv/ssm
  // state for GDN layers). The next-token->embedding feedback stays
  // GPU-resident: a per-step kernel writes the next id (greedy argmax, or
  // temperature+top-p sampling) into a buffer the embed gather reads back,
  // so the token never round-trips to the host. Per-token command buffers
  // are committed without waiting and chained by a GPU event (the event
  // also orders the in-place GDN state updates + reused scratch across
  // buffers). `temperature <= 0` => greedy. Fills `out_ids` (size
  // n_steps). Text decode only (1-D rope at sequential positions);
  // returns false on append failure. Greedy output matches a
  // forward_argmax() loop.
  bool decode_pipelined(ContextId cid, std::int32_t first_token,
                        int n_steps, std::vector<std::int32_t>& out_ids,
                        float temperature = 0.0f, float top_p = 1.0f,
                        std::uint64_t seed = 0);
  bool decode_pipelined(std::int32_t first_token, int n_steps,
                        std::vector<std::int32_t>& out_ids,
                        float temperature = 0.0f, float top_p = 1.0f,
                        std::uint64_t seed = 0) {
    return decode_pipelined(_cid, first_token, n_steps, out_ids,
                            temperature, top_p, seed);
  }

  // ---- Per-token streaming pipelined decode -------------------------
  // The streaming counterpart of decode_pipelined: instead of generating
  // a fixed N tokens behind a single wait, it exposes the GPU-resident
  // chain one token at a time so the caller can stream / stop-check /
  // abort, while still overlapping the host's per-token work with the
  // GPU's next forward. See ModelExec::pdecode_* for the contract.
  // Keyed by ContextId so concurrent contexts on a shared model don't
  // collide. begin returns false if the embed/argmax/sample kernels are
  // unavailable.
  bool pdecode_begin(ContextId cid, std::int32_t first_token,
                     std::span<const std::int32_t> prompt,
                     const GpuSamplerParams& sp, int max_tokens,
                     int rope_first = -1);
  bool pdecode_commit(ContextId cid);
  std::int32_t pdecode_next(ContextId cid);
  void pdecode_end(ContextId cid);

  // One synchronous batched decode step over the N active branches `cids`:
  // append in_tokens[i] to branch i's K/V at its own position, run a SINGLE
  // forward (projection / MLP / lm_head matmuls batched across all N; RoPE +
  // attention + GDN state per branch), and fill `out_logits` with the N rows
  // of [vocab] f32 logits (row-major, size N*vocab). Branches need NOT share
  // a seq_len. `rope_pos` overrides the RoPE position per branch (post-image
  // mROPE); pass an EMPTY span for plain text decode (rope == KV slot). The
  // caller samples each row and feeds the chosen token back as the next
  // step's in_token, dropping a branch from `cids` once it stops. Returns
  // false if the batched kernels are unavailable or an append fails.
  bool decode_batched_step(std::span<const ContextId>      cids,
                           std::span<const std::int32_t>   in_tokens,
                           std::span<const std::int32_t>   rope_pos,
                           std::vector<float>&             out_logits);

  // Self-contained greedy batched decode (synchronous), built on
  // decode_batched_step: generate `n_steps` argmax tokens for each branch.
  // out[b] = the per-step argmax ids. Verified token-exact vs serial
  // decode_step_fast, including branches at DIFFERENT seq_lens.
  std::vector<std::vector<std::int32_t>> decode_batched_argmax(
      std::span<const ContextId> cids,
      std::span<const std::int32_t> first_tokens, int n_steps);

  // ---- Batched pipelined decode (bdecode_*) --------------------------
  // The N-branch counterpart of pdecode_*: the per-branch sampled tokens
  // stay GPU-resident -- a per-step kernel writes each branch's next id
  // into gen_ids, which the NEXT step's batched embed gathers, so logits
  // never round-trip to the host. Per-token command buffers are committed
  // without waiting and chained by a GPU event so the host's per-token
  // stop-check overlaps the GPU's next forward, AND the per-branch GPU
  // sampler replaces the [N, vocab] host pull. Constant-N: all branches
  // are carried until the caller ends the session; a branch that has
  // stopped on the host is simply not collected (its matmuls cost nothing
  // extra -- the weight read is amortised across N regardless). Greedy
  // (sp.greedy) is token-exact vs decode_batched_argmax. `cids` are this
  // model's contexts; `rope_pos[i] >= 0` overrides branch i's RoPE anchor
  // (post-image mROPE), empty span / <0 => sequential text decode.
  bool bdecode_begin(std::span<const ContextId> cids,
                     std::span<const std::int32_t> first_tokens,
                     const GpuSamplerParams& sp, int max_tokens,
                     std::span<const std::int32_t> rope_pos = {});
  bool bdecode_commit();
  bool bdecode_next(std::vector<std::int32_t>& out_tokens);
  void bdecode_end();

  const Config& config() const { return _cfg; }

  // ---- MTP speculative decode (Multi-Token Prediction) ----------------
  // Whether a bundled MTP head (mtp.safetensors: one full-attn decoder layer
  // + the eh-fusion `fc` and pre-norms) was loaded. Drives the spec-decode
  // fast path; false => callers use the plain serial decode.
  bool has_mtp() const noexcept { return _mtp.ok; }

  // Speculative greedy decode with the MTP head as the drafter and THIS model
  // as the verifier. Generates up to `n_steps` tokens from `first_token`
  // (whose prefill must have just run on `cid`, leaving the last hidden in
  // _mtp_h), appending accepted tokens to `cid`. Each round the MTP head
  // drafts `draft_len` tokens, the main model verifies all of them in a chain
  // of forwards, the longest greedy-matching prefix is accepted, and the
  // rejected speculative tail is rolled back (paged KV via kv_rollback + GDN
  // recurrent ring via gdn_ring_rollback -- the depth>1 pdecode machinery).
  // GREEDY output is token-exact vs a forward_argmax loop (verification makes
  // the drafter's quality affect only speed, never the tokens). Fills
  // `out_ids` (<= n_steps). `accepted_out` (optional) returns the total
  // drafted-token acceptances for an acceptance-rate report. Returns false if
  // the MTP head / decode scratch is unavailable.
  // draft_len: 1 = depth-1 (default, fastest here -- 1.14x; the MTP head runs
  // ONCE in the fused verify), >=2 = depth-2 (chains a 2nd MTP application; more
  // tokens/round but net SLOWER on this model -- a 2nd full-vocab lm_head/round
  // outweighs the gain).
  // `ctl` adds optional streaming (on_round), early stop (is_stop), and the
  // post-multimodal rope anchor (rope_first); the defaults preserve the
  // generate-to-budget, sequential-rope behavior. `out_ids` excludes any stop
  // token (its KV is rolled back so the context ends at the last kept token).
  bool mtp_decode(ContextId cid, std::int32_t first_token, int n_steps,
                  std::vector<std::int32_t>& out_ids, int draft_len = 1,
                  long* accepted_out = nullptr, long* rounds_out = nullptr,
                  const MtpDecodeCtl& ctl = {});

  // Teacher-forced MTP depth-1 draft accuracy on a FIXED token sequence (a
  // diagnostic to size MTP draft quality vs a reference WITHOUT the free-run
  // stream-divergence confound). `cont` is walked verbatim through the base
  // model on `cid` (already prefilled with the prefix); at each position i the
  // MTP head is conditioned on the TRUE next token cont[i+1] and its draft is
  // checked against cont[i+2]. `chunk` is the per-verify width (2 mirrors the
  // depth-1 decode's MTP attention window). Prints + returns the hit/total.
  bool mtp_teacher_force(ContextId cid, const std::vector<std::int32_t>& cont,
                         int chunk, long* d1_hits, long* d1_total);

  // Toggle the shared-prefix batched decode attention at runtime (default set
  // at load from VPIPE_QWEN_SHARED_ATTN + kernel availability). For A/B tests
  // and benchmarks that compare the shared-prefix path against per-branch.
  void set_shared_attn(bool on) noexcept { _shared_attn = on; }
  // Leviathan-Chen rejection sampling (with residual correction) for the MTP
  // verify, instead of the default deterministic-seed exact-match acceptance.
  // Opt-in: it raises SAMPLING (temp>0) acceptance but is NOT token-exact vs
  // the serial sampler. Depth-1 + pure-temperature only (else falls back).
  void set_leviathan(bool on) noexcept { _leviathan = on; }
  bool leviathan() const noexcept { return _leviathan; }
  // Run the L-C nucleus + accept/residual on the GPU (lc_sample / lc_accept)
  // instead of the host CPU grind. Opt-in (also VPIPE_MTP_GPU_LC); no-op unless
  // leviathan() is on.
  void set_lc_gpu(bool on) noexcept { _lc_gpu = on; }
  bool lc_gpu() const noexcept { return _lc_gpu; }
  bool shared_attn() const noexcept { return _shared_attn; }
  // MTP prefix seed (decode- vs prefill-throughput tradeoff): when on,
  // prefill() captures the per-position post-norm hiddens + ids so mtp_decode
  // can seed the drafter's KV with the prompt (higher draft acceptance, small
  // extra prefill cost). DEFAULT OFF -- a stage opts in (text-chat) or leaves
  // it off (realtime-vqa). VPIPE_MTP_NO_SEED hard-disables regardless.
  void set_mtp_prefix_seed(bool on) noexcept { _mtp_seed_enabled = on; }
  bool mtp_prefix_seed() const noexcept { return _mtp_seed_enabled; }

  // M5 matrix-core fast-path engagement -- for perf-GUARD tests, NOT timing.
  // A silent fallback off these paths (a broken kernel load, or a gate
  // condition an M4-side change accidentally widened) is ~2-2.5x slower at
  // prefill/decode but stays TOKEN-EXACT, so token-exact tests can't catch it,
  // and a perf-floor test would flake on the M5's thermal throttling (cold
  // ~1240 tok/s prefill drops to ~520 hot). These let a guard assert the fast
  // path is SELECTED, independent of timing:
  //   uses_matrix_cores -- matmul2d prefill GEMM + dequant (_use_mma; M5 only,
  //                        4-bit weights). Off => steel quantized GEMM.
  //   mma_flash_attn    -- head_dim-256 matmul2d flash prefill attention.
  //   gqa_flash_decode  -- GQA flash-decode kernel (each kv-head read once for
  //                        all G query heads; default on for full-attn models).
  bool uses_matrix_cores() const noexcept { return _use_mma; }
  bool mma_flash_attn() const noexcept { return _fn_sdpa_mma.valid(); }
  bool gqa_flash_decode() const noexcept { return _gqa_vec; }
  // Dynamic-int8 accelerated PREFILL GEMMs (see shared/i8-gemm.h): LOSSY
  // (int8 quantization, ~1e-2 per GEMM -- NOT token-exact), strictly
  // opt-in. Decode is untouched (M=1 never qualifies). VPIPE_I8_GEMM
  // overrides. Wired from the stage config via the exec.
  void set_i8_gemm(bool on);
  bool i8_gemm_enabled() const noexcept { return _i8 != nullptr; }
  // Mixed-precision affine (OptiQ) de-fused per-tensor path engaged (some
  // 4-bit, some 8-bit linears). For the token-exact OptiQ guard test.
  bool uses_mixed_precision() const noexcept { return _mixed; }
  // Native k-quant (GGUF) path engaged (per-tensor q4_K/q5_K/q6_K blocks).
  bool uses_kquant() const noexcept { return _kquant; }

private:
  MetalQwenModel() = default;

  // Allocate the reused decode scratch buffers (lazy, first decode).
  bool ensure_decode_scratch_();

  // Lazily allocate the two-stage-argmax + histogram-sampler scratch (mirrors
  // MetalGemmaModel::ensure_scratch_'s _d_argmax_part/_d_smp_*). Allocated ONCE
  // and reused across every decode path (single-stream, pdecode, batched,
  // MTP/verify): the serial command encoder serialises the WAR hazard on the
  // shared scratch, so no per-branch copy is needed. Independent of
  // ensure_decode_scratch_/ensure_bscratch_ so the MTP heads (which use their
  // own vlogits) get it too.
  bool ensure_sampler_scratch_();

  // Two-stage parallel argmax (argmax_partial -> argmax_combine) over
  // logits[logits_off .. +vocab], lowest-index tie-break, writing the id to
  // out[out_off]. Falls back to the single-tg _fn_argmax (VPIPE_QWEN_ARGMAX1=1
  // or missing kernels/scratch). Token/bit-exact vs the single-tg path.
  void encode_argmax_(metal_compute::ComputeEncoder& enc,
                      const metal_compute::SharedBuffer& logits,
                      std::size_t logits_off,
                      const metal_compute::SharedBuffer& out,
                      std::size_t out_off, int vocab);

  // Histogram multi-tg sampler core (sample_*_f16) over logits[logits_off ..],
  // writing the chosen id to out[out_off] and updating seen[seen_off..].
  // Deterministic (integer-atomic histogram). Falls back to the single-tg
  // _fn_sample (VPIPE_QWEN_SAMPLE1=1 or missing kernels/scratch).
  void encode_sample_core_(metal_compute::ComputeEncoder& enc,
                           const metal_compute::SharedBuffer& logits,
                           std::size_t logits_off,
                           const metal_compute::SharedBuffer& out,
                           std::size_t out_off,
                           const metal_compute::SharedBuffer& sample_ws,
                           std::size_t ws_off,
                           const metal_compute::SharedBuffer& seen,
                           std::size_t seen_off, const GpuSamplerParams& sp,
                           std::uint32_t step_seed, int vocab);

  // Encode one decode token's layer stack + final norm + lm_head into
  // `enc`, reading the residual stream `_d_x` (the caller fills it with
  // the token embedding -- host memcpy in forward(), in-stream gather in
  // decode_pipelined()) and writing `_d_logits`. `pgtab`@`pgtab_off` is
  // the page table for the full-attn layers (a per-step slice in the
  // pipelined path, _pgtab in forward()).
  // return_hidden (MOSS-TTS): stop after the final norm leaves the normed
  // hidden in _d_hn; skip the lm_head (backbone_only models have no lm_head).
  void encode_decode_step_(
      metal_compute::ComputeEncoder& enc, ContextId cid, int pos, int rpos,
      std::size_t page_off, int n_pages,
      const ContextManager::AppendSlot& slot,
      const metal_compute::SharedBuffer& pgtab, std::size_t pgtab_off,
      bool return_hidden = false);

  // Encode the next-token pick (greedy argmax or full GPU sampler) into
  // `enc`: reads `logits`, writes the chosen id to out_id[out_off], and
  // (sampled path) updates `seen`. Shared by decode_pipelined and the
  // per-token pdecode_* path.
  void encode_sample_(metal_compute::ComputeEncoder& enc,
                      const metal_compute::SharedBuffer& logits,
                      const metal_compute::SharedBuffer& out_id,
                      std::size_t out_off, const GpuSamplerParams& sp,
                      std::uint32_t step_seed,
                      const metal_compute::SharedBuffer& sample_ws,
                      const metal_compute::SharedBuffer& seen);

  // Per-context streaming-pipeline decode state (pdecode_*).
  // Run-ahead pipeline (mirrors MetalGemmaModel::PDecode): up to `depth`
  // command buffers in flight so the CPU encodes token N+1 while the GPU
  // runs token N. depth=1 (the DEFAULT) reproduces the original single-in-
  // flight behavior exactly. depth>1 run-ahead commits a token's KV (paged)
  // before the host has seen it -> pdecode_end rolls back the speculative tail
  // via ContextManager::kv_rollback PLUS the GDN recurrent (ssm/conv) ring
  // (gdn_ring_rollback + gdn_ring_end). depth>1 is correct + rollback-safe and
  // pdecode_supports_runahead() is true, but measured ~0 win for Qwen3.5-4B
  // (the encode bubble is tiny vs the GPU step), so it stays OPT-IN via
  // VPIPE_QWEN_PDECODE_DEPTH (kept for lower variance + the MTP foundation).
  struct PDecode {
    metal_compute::SharedBuffer gen_ids;     // [cap] int32 GPU chain
    metal_compute::SharedBuffer seen;        // [vocab] uint8 (sampled)
    metal_compute::SharedBuffer sample_ws;   // [vocab] f16 (sampled)
    metal_compute::SharedBuffer pgt;         // [max_pages*3] int32 slice
    metal_compute::CommandStream stream;
    metal_compute::Event ev;
    struct InFlight {
      metal_compute::CommandStream::Fence fence;
      int idx = -1;                          // gen_ids slot it writes
    };
    std::deque<InFlight> ring;               // FIFO of in-flight steps
    GpuSamplerParams sp;
    ContextId cid;
    int           rope_base  = -1;  // mROPE pos of first decode token
    int           kv_base    = 0;   // KV seq_len at begin (rope anchor)
    int           cap        = 0;   // gen_ids slot capacity
    int           produced   = 0;   // gen_ids[0..produced-1] finalised
    int           committed  = 0;   // next out_idx to assign
    int           depth      = 1;   // run-ahead pipeline depth
    std::uint64_t gpu_step   = 0;   // event signal counter
  };
  std::unordered_map<std::uint32_t, PDecode> _pdec;

  // Shared batched forward over a pre-filled [n, hidden] residual stream
  // x. When mrope_cos/mrope_sin are non-null the full-attn layers use
  // table-driven mROPE; otherwise scalar partial-RoPE at the KV offset.
  // verify_all (MTP batched verify): instead of the last-position logits,
  // compute the greedy argmax at EVERY position (the per-position next-token
  // predictions, into *preds_out, size n) and run the final-layer MLP for all
  // rows (no last-position prune). The pre-final-norm hidden of every position
  // is left in `x` (the residual stream) for the caller to read. The returned
  // vector is empty in verify mode (the per-position logits aren't pulled).
  // return_hidden (MOSS-TTS): skip lm_head; final-norm the last position
  // into *hidden_out (a moved-out [n*hidden] f16 SharedBuffer, last row at
  // [0:hidden]) and return empty. Mutually exclusive with verify_all.
  // allhidden_out (MTP prefix seed): in the normal prefill path (no verify_all,
  // no return_hidden), ALSO final-norm ALL n positions into a moved-out [n,H]
  // f16 SharedBuffer -- the per-position post-norm hiddens the MTP drafter
  // conditions on. The normal logit return is unaffected. Only the mixed /
  // k-quant paths (the MTP-capable ones) compute the full last-layer residual
  // for every position, so the capture is valid exactly where an MTP head can
  // consume it.
  // tap_layers / taps_out (Krea-2 text encoder): snapshot the un-normed
  // residual stream after each 0-indexed layer L listed in *tap_layers into
  // *taps_out at slot offset j*n*H (j = index into tap_layers) -- the
  // per-layer hidden states a downstream consumer conditions on (matches HF
  // output_hidden_states[L+1]). *taps_out must be a [tap_layers->size()*n*H]
  // compute-dtype buffer. No effect when either is null.
  std::vector<float> forward_chunk_(
      ContextId cid, const metal_compute::SharedBuffer& x, int n,
      const metal_compute::SharedBuffer* mrope_cos,
      const metal_compute::SharedBuffer* mrope_sin,
      bool verify_all = false, std::vector<std::int32_t>* preds_out = nullptr,
      bool return_hidden = false,
      metal_compute::SharedBuffer* hidden_out = nullptr,
      metal_compute::SharedBuffer* allhidden_out = nullptr,
      const std::vector<int>* tap_layers = nullptr,
      metal_compute::SharedBuffer* taps_out = nullptr);

  // ---- Batched (N-branch parallel) decode --------------------------
  // VQA fanout: N branched contexts that share a prefix each decode one
  // token per step IN LOCKSTEP (all at the same seq_len). The weight-bound
  // matmuls (q/k/v/o proj, MLP, lm_head) run ONCE over the [N, hidden]
  // stack (steel GEMM M=N), amortising the DRAM-bound weight reads across
  // all N branches; only the per-branch parts (KV write + paged SDPA on
  // full-attn layers, conv1d + ssm recurrence on GDN layers) loop over N.
  // Reused [N]-wide scratch + per-branch page tables, sized for the batch.
  struct BScratch {
    int n = 0;                                   // batch size it was sized for
    metal_compute::SharedBuffer x, hn, qfull, q3, gate3, kbuf, vbuf, at, ao,
        mixqkv, zbuf, abuf, bbuf, convout, gbuf, betabuf, ygdn, normout, sg,
        upb,   // mixed-precision MLP: de-fused up_proj (gate -> sg, up -> upb)
        logits;
    // MoE batched scratch (N branches; npair = N*top_k). Mirrors the prefill
    // MoE temps. Allocated only when _cfg.is_moe().
    metal_compute::SharedBuffer moe_logits, moe_eid, moe_w, moe_act, moe_part,
        moe_out, moe_ssg, moe_sout, moe_gate;
    metal_compute::SharedBuffer pgt;             // [N * max_pages * 3] int32
    metal_compute::SharedBuffer tok_in;          // [N] int32 (embed gather in)
    metal_compute::SharedBuffer argmax_id;       // [N] int32 (greedy out)
    // Shared-prefix attention partials (phase A -> phase B), f32:
    //   shacc [N, Hq, D] un-normalized O, shm/shl [N, Hq] online-softmax m/l.
    metal_compute::SharedBuffer shacc, shm, shl;
  };
  bool ensure_bscratch_(BScratch& bs, int n);
  BScratch _bdec;   // cached batched-decode scratch (reused across steps)

  struct Layer;     // defined below; fwd-declared for encode_moe_mlp_ (ref arg)

  // Size-based quantized-matmul entrance: y[M,Nout] = x[M,K] @ dequant(w)^T,
  // picking the kernel by the row count M (what MLX's quantized_matmul does
  // internally; the raw metal path needs it explicit). M==1 -> qmv (matvec);
  // 2 <= M <= kQmvBatchMaxRows -> batched GEMV (qmv bandwidth, weights read
  // once across the rows, MAXM=2 + grid.z tiling); larger M -> steel GEMM.
  // 8-bit weights have no batched-GEMV -> qmv (M==1) / steel. This is THE
  // place the batched decode adapts to the (shrinking) active branch count.
  static constexpr int kQmvBatchMaxRows = 8;
  // MoE grouped-GEMM GEMV->steel crossover (prefill row count above which the
  // matrix-tiled steel expert GEMM beats the bandwidth GEMV). Machine-dependent
  // (matrix cores shift it down); default 1024, VPIPE_QWEN_MOE_STEEL_MIN sets
  // it, and a future per-machine probe can resolve it at load.
  int _moe_steel_min = 1024;
  // Grouped-prefill scratch (counting sort by expert + segmented MAXM-batched
  // expert GEMV). When passed to encode_moe_mlp_ the expert compute reads each
  // expert's weight once per MAXM-row tile instead of once per routed token.
  struct MoeGrouped {
    const metal_compute::SharedBuffer *hist = nullptr, *boff = nullptr,
        *curs = nullptr, *t2e = nullptr, *ntile = nullptr, *srow = nullptr,
        *sdst = nullptr, *gact = nullptr, *gdout = nullptr;
    int npad = 0, maxtiles = 0, maxm = 4;
    // steel: matrix-tiled grouped GEMM (BM=maxm=32 tiles) for large prefill;
    // else the bandwidth GEMV grouping (maxm=2). The steel tier fuses the
    // row-gather into the gate|up load and the scatter into the down store.
    bool steel = false;
    // Timing-only ablation toggles (VPIPE_MOE_ABL=sort,gs,gemm,shared): skip a
    // component to measure its prefill cost. Output is garbage -- bench only.
    bool abl_sort = false, abl_gs = false, abl_gemm = false, abl_shared = false;
  };
  // Mixture-of-Experts MLP for M rows (Qwen3.5-MoE), shared by the batched
  // decode path (M=N branches). Router(w8) -> per-row top-k route -> expert
  // gate|up+SwiGLU -> down -> weighted combine + sigmoid-gated shared expert;
  // finalize adds both into the residual x (in-place). M==1 uses the qmv
  // router/shared, M>1 the steel GEMM. All buffers caller-owned. When `grp` is
  // non-null the expert compute uses the grouped (sorted) path.
  void encode_moe_mlp_(metal_compute::ComputeEncoder& enc, const Layer& ly,
                       int M, const metal_compute::SharedBuffer& x,
                       const metal_compute::SharedBuffer& hn,
                       const metal_compute::SharedBuffer& logits,
                       const metal_compute::SharedBuffer& eid,
                       const metal_compute::SharedBuffer& w,
                       const metal_compute::SharedBuffer& act,
                       const metal_compute::SharedBuffer& part,
                       const metal_compute::SharedBuffer& moe_out,
                       const metal_compute::SharedBuffer& ssg,
                       const metal_compute::SharedBuffer& sout,
                       const metal_compute::SharedBuffer& gate,
                       const MoeGrouped* grp = nullptr);
  void qmm_auto_(metal_compute::ComputeEncoder& enc, int m,
                 const metal_compute::SharedBuffer& w,
                 const metal_compute::SharedBuffer& s,
                 const metal_compute::SharedBuffer& b,
                 const metal_compute::SharedBuffer& xin,
                 const metal_compute::SharedBuffer& y, int Kk, int Nout);
  // Fused SwiGLU MLP variant (gate/up interleaved -> y[M, Nout/2]); same
  // size-based selection over _fn_qmv_swiglu / _fn_qmv_batch_swiglu /
  // _fn_qmm_swiglu. `Nout` is the fused width (2*ffn).
  void qmm_auto_swiglu_(metal_compute::ComputeEncoder& enc, int m,
                        const metal_compute::SharedBuffer& w,
                        const metal_compute::SharedBuffer& s,
                        const metal_compute::SharedBuffer& b,
                        const metal_compute::SharedBuffer& xin,
                        const metal_compute::SharedBuffer& y, int Kk, int Nout);

  // GPU per-branch sampler over the [N, vocab] `bs.logits`: loops the
  // single-row argmax / sample_topp kernels, writing each branch's chosen
  // id into gen_ids row `out_idx` (gen_ids is [cap][N] step-major). Reuses
  // _fn_argmax / _fn_sample with per-row buffer offsets; per-branch seed so
  // branches don't sample in lock-step.
  void encode_sample_batched_(metal_compute::ComputeEncoder& enc,
                              const metal_compute::SharedBuffer& gen_ids,
                              int out_idx, int n,
                              const GpuSamplerParams& sp,
                              std::uint32_t step_seed);

  // Per-session batched pipelined-decode state (bdecode_*). One session at
  // a time (the realtime-vqa stage decodes one scene's branches), so this
  // is a single member rather than a per-cid map like _pdec.
  // Run-ahead ring (mirrors PDecode): up to `depth` committed steps in
  // flight, so the CPU encode of step N+1 AND the host's emit/stop-check
  // both overlap the GPU's step N. Unlike the serial pdecode, depth>1 needs
  // NO rollback machinery (no GDN ring, no kv_rollback): constant-N bdecode
  // already over-advances stopped branches' KV/GDN by design and bdecode_end
  // never rolls back -- a speculative uncollected tail is the same kind of
  // discard. Collected rows stay byte-identical: the GPU event chain orders
  // the steps and the embed gather reads gen_ids on-device.
  // VPIPE_QWEN_BDECODE_DEPTH overrides (default 2, 1 = the old lockstep).
  struct BDecode {
    std::vector<ContextId>      cids;        // the N branch contexts
    std::vector<int>            rope_base;   // mROPE anchor / -1 sequential
    std::vector<int>            kv_base;     // KV seq_len at begin per branch
    metal_compute::SharedBuffer gen_ids;     // [cap][N] int32 GPU chain
    metal_compute::SharedBuffer seen;        // [N][vocab] uint8 (sampled)
    metal_compute::SharedBuffer sample_ws;   // [N][vocab] f16 (sampled)
    metal_compute::CommandStream stream;
    metal_compute::Event ev;
    struct InFlight {
      metal_compute::CommandStream::Fence fence;
      int idx = -1;                          // gen_ids step row it writes
    };
    std::deque<InFlight> ring;               // FIFO of in-flight steps
    GpuSamplerParams sp;
    int           n             = 0;
    int           cap           = 0;   // gen_ids step capacity
    int           produced      = 0;   // step rows drained via next()
    int           committed     = 0;   // next step row to assign
    int           depth         = 1;   // run-ahead pipeline depth
    std::uint64_t gpu_step      = 0;
    bool          active        = false;
  };
  BDecode _bdec_sess;

  // Encode one N-branch decode step into `enc`: reads the [N, hidden]
  // residual `bs.x` (the caller fills it via the embed gather), writes the
  // per-branch K/V + advances GDN state, and writes [N, vocab] into
  // `bs.logits`. All metadata is PER-BRANCH (size N) -- branches need NOT
  // be at the same seq_len: the projection / MLP / lm_head matmuls are
  // position-independent and batch across the whole active set, while RoPE
  // (rope_pos_v[i]) and the paged attention (slots[i].position + its own
  // page table) run per branch. `rope_pos_v` overrides the RoPE position
  // (post-image mROPE); pass slots[i].position for plain text decode.
  // `shared_pages` (>0) is the number of leading page-table entries the N
  // branches share (identical page ids); when set and N>1 and head_dim==256,
  // the full-attn layers read that shared prefix once (phase A) and merge each
  // branch's private pages (phase B). 0 => the per-branch SDPA path.
  void encode_batched_step_(
      metal_compute::ComputeEncoder& enc, BScratch& bs,
      std::span<const ContextId> cids,
      const std::vector<ContextManager::AppendSlot>& slots,
      const std::vector<std::size_t>& page_offs,
      const std::vector<int>& n_pages_v,
      const std::vector<int>& rope_pos_v,
      int shared_pages);

  struct Layer {
    bool is_full = false;
    metal_compute::SharedBuffer in_ln, post_ln;
    // Full-attention weights (is_full).
    metal_compute::SharedBuffer qw, qs, qb;   // q_proj (2*qd out, gated)
    metal_compute::SharedBuffer kw, ks, kb, vw, vs, vb, ow, os, ob;
    metal_compute::SharedBuffer q_norm, k_norm;     // [head_dim]
    // Gated-DeltaNet weights (!is_full).
    metal_compute::SharedBuffer iqw, iqs, iqb;   // in_proj_qkv -> conv_dim
    metal_compute::SharedBuffer izw, izs, izb;   // in_proj_z   -> value_dim
    metal_compute::SharedBuffer ibw, ibs, ibb;   // in_proj_b   -> v_heads
    metal_compute::SharedBuffer iaw, ias, iab;   // in_proj_a   -> v_heads
    metal_compute::SharedBuffer gow, gos, gob;   // out_proj (value_dim -> hidden)
    metal_compute::SharedBuffer conv_w;          // [conv_dim, conv_kernel] f16
    metal_compute::SharedBuffer A_log, dt_bias;  // [v_heads] f32
    metal_compute::SharedBuffer gdn_norm;        // [v_head_dim] gated-rms weight
    // MLP (every layer): fused interleaved gate/up + down.
    metal_compute::SharedBuffer guw, gus, gub, dw, ds, db;

    // ---- Mixture-of-Experts weights (Qwen3.5-MoE, iff _cfg.is_moe()) ------
    // Replaces the dense MLP above. Router (mlp.gate): w8 affine [E, H].
    // Experts (mlp.switch_mlp): batched 3D w4 affine -- gate|up fused
    // interleaved [E, 2*moe_inner, H], down [E, H, moe_inner]. Shared expert
    // (mlp.shared_expert): dense w4 gate|up interleaved [2*shared_inner, H] +
    // down [H, shared_inner], sigmoid-gated by mlp.shared_expert_gate
    // (w8 [1, H]). All affine group_size 64.
    metal_compute::SharedBuffer rgw, rgs, rgb;     // router gate (w8) [E,H]
    metal_compute::SharedBuffer eguw, egus, egub;  // experts gate|up (w4) [E,2I,H]
    metal_compute::SharedBuffer edw, eds, edb;     // experts down (w4) [E,H,I]
    metal_compute::SharedBuffer sguw, sgus, sgub;  // shared gate|up (w4) [2S,H]
    metal_compute::SharedBuffer sdw, sds, sdb;     // shared down (w4) [H,S]
    metal_compute::SharedBuffer segw, segs, segb;  // shared_expert_gate (w8) [1,H]

    // ---- Mixed-precision affine (OptiQ) -------------------------------
    // Heterogeneous per-tensor bit-width (4 or 8 affine) across one model
    // (mlx-optiq sensitivity quant). Used iff the model is _mixed; the
    // fused affine buffers (qw|k|v, iqw, guw-interleaved) are NOT built
    // then. Each projection keeps its own triple + bit width and dispatches
    // per-tensor, exactly like the heterogeneous k-quant path above:
    //   full-attn: q->qw, k->kw, v->vw, o->ow  (each its own triple)
    //   GDN:       qkv->iqw, z->izw, a->iaw, b->ibw, out->gow
    //   MLP:       gate->guw, up->uw (added here), down->dw
    metal_compute::SharedBuffer uw, us, ub;   // up_proj (mixed only)
    // Per-projection affine bit width (4 or 8). Default 4; set at load.
    int q_bits = 4, k_bits = 4, v_bits = 4, o_bits = 4;
    int qkv_bits = 4, z_bits = 4, a_bits = 4, b_bits = 4, gout_bits = 4;
    int gate_bits = 4, up_bits = 4, down_bits = 4;
    // Mixed-precision MLP fusion: when gate_bits==up_bits==4 the gate|up are
    // interleaved into guw (like the uniform path) and the fused swiglu qmv /
    // qmm kernels run, recovering the fusion the per-tensor de-fused path
    // would drop (28/32 OptiQ layers). down stays per-tensor (own bits). The
    // 4 genuinely mixed-bit gate/up layers keep the de-fused path (uw built).
    bool mlp_fused = false;
    // Mixed-precision QKV fusion: when a full-attn layer's q/k/v are uniform at
    // the base width, they are fused into one q|k|v GEMM (like the non-mixed
    // path) and the forward's fused branch (gated on !qkv_fused) dispatches it
    // with the base-width qmv/qmm. Genuinely mixed layers keep qkv_fused=false
    // (kw/vw built). o_proj is always per-tensor (own o_bits).
    bool qkv_fused = false;
    // Routed-expert (MoE switch_mlp) quant width: 4 or 8. Selects the w4/w8
    // gather + grouped expert kernels at dispatch (router/shared stay w8).
    int eg_bits = 4;

    // ---- Native k-quant (GGUF) weights (used iff the model is _kquant) --
    // Heterogeneous per-tensor quant (Q4_K_M mixes q4/q5/q6), so each
    // projection keeps its own raw block buffer + family tag. q+k fuse
    // (both q4_K); v / o / in_proj parts / mlp are separate. a+b are tiny
    // Q8_0 -> dequant'd to f16 at load (kqab) so they ride a plain f16 GEMV.
    metal_compute::SharedBuffer kqk;     // q|k fused raw  [kqk_n, H]
    metal_compute::SharedBuffer kqv;     // v_proj raw     [kd, H]
    metal_compute::SharedBuffer kqo;     // o_proj raw     [H, qd]
    metal_compute::SharedBuffer kqkv;    // in_proj_qkv    [Cd, H]
    metal_compute::SharedBuffer kqz;     // in_proj_z      [vald, H]
    metal_compute::SharedBuffer kqab;    // a|b f16        [2*Hv, H]
    metal_compute::SharedBuffer kqout;   // out_proj       [H, vald]
    metal_compute::SharedBuffer kqgate;  // mlp gate       [ffn, H]
    metal_compute::SharedBuffer kqup;    // mlp up         [ffn, H]
    metal_compute::SharedBuffer kqdown;  // mlp down       [H, ffn]
    KQ kqk_t = KQ::kNone, kqv_t = KQ::kNone, kqo_t = KQ::kNone,
       kqkv_t = KQ::kNone, kqz_t = KQ::kNone, kqout_t = KQ::kNone,
       kqgate_t = KQ::kNone, kqup_t = KQ::kNone, kqdown_t = KQ::kNone;
    int kqk_n = 0;   // rows in the fused q|k buffer (2*qd + kd)

    // ---- Q4_K -> affine-g32 decode repack (VPIPE_QWEN_Q4K_AFFINE) -------
    // The native qmv_q4k is ~1.9x slower than the affine qmv (its hierarchical
    // scale unpack is the bottleneck). When enabled, Q4_K MLP gate/up are
    // losslessly repacked at load to the affine-4bit-g32 form (scale=d*sub,
    // bias=-dmin*min) and decode dispatches affine_qmv_w4g32. The raw kq*
    // buffers are kept (prefill's dequant->GEMM still reads them).
    metal_compute::SharedBuffer gate_aw, gate_as, gate_ab;   // MLP gate affine
    metal_compute::SharedBuffer up_aw, up_as, up_ab;         // MLP up affine
    bool ffn_q4k_aff = false;
    metal_compute::SharedBuffer qk_aw, qk_as, qk_ab;   // fused q|k affine
    metal_compute::SharedBuffer o_aw, o_as, o_ab;      // attn o_proj affine
    bool qk_q4k_aff = false, o_q4k_aff = false;
  };

  Config _cfg;
  metal_compute::MetalCompute* _mc = nullptr;

  metal_compute::ComputeLibrary _lib_qmv, _lib_qmm, _lib_rms, _lib_elt,
      _lib_rope, _lib_sdpa, _lib_gdn,
      // Matrix-core (M5+) prefill GEMM: dequant 4-bit weight -> bf16/f16
      // scratch, then dense matmul2d on the hardware matrix units; plus the
      // matrix-core flash attention (head_dim 256).
      _lib_dequant, _lib_dense_mma, _lib_sdpa_mma,
      // Native k-quant (GGUF): plain dense f16 GEMM/GEMV (dense_gemm.metal),
      // M4-capable -- the prefill stage of kqmm_ after the typed dequant.
      // On M5 dense_gemm_ instead routes the dequant'd GEMM onto the
      // matmul2d matrix units (_fn_dense_mma*, via _use_mma); this steel
      // GEMM is then the M4 / small-M (< _mma_min_m) fallback.
      _lib_dense,
      // MLX steel register-resident flash attention (attn_steel, half-only):
      // the bd256 instantiation drives Qwen full-attn FRESH prefill (contiguous
      // K/V). Invalid for bf16 models -> the paged path stays the fallback.
      _lib_attn;
  metal_compute::ComputeFunction _fn_qmv, _fn_qmv_add, _fn_qmv_swiglu, _fn_qmm,
      _fn_qmm_swiglu,
      // Q4_K->affine-g32 decode repack: the affine 4-bit g32 qmv (+fused-add)
      // that consumes the repacked MLP gate/up, and the GPU repack kernel.
      _fn_qmv_w4g32, _fn_qmv_w4g32_add, _fn_repack_q4k,
      // Mixed-precision affine (OptiQ): the 8-bit counterparts of qmv/qmv_add/
      // qmm + the 8-bit dequant (affine_dequant_w8g64) + the 8-bit batched-GEMV
      // (affine_qmv_batch_w8g64, weight read once across MAXM=2 rows -- lets the
      // MTP verify batch its 8-bit tensors at qmv bandwidth instead of steel).
      // Loaded only when _mixed so a tensor at either bit width dispatches its
      // matching kernel (the 4-bit set above + these). _fn_dequant (w4) shared.
      _fn_qmv8, _fn_qmv8_add, _fn_qmm8, _fn_dequant8, _fn_qmv8_batch,
      _fn_requant_w8w4,
      _fn_transpose,
      _fn_rms, _fn_swiglu, _fn_residual, _fn_rope_partial, _fn_rms_rope,
      _fn_mul_sigmoid,
      _fn_head_slice, _fn_sdpa_paged, _fn_sdpa_paged_mb256, _fn_sdpa_paged_mb,
      _fn_sdpa_paged_qtile,
      // simdgroup_matrix key-split flash prefill (head_dim 256, drop-in for
      // qtile): runs on M4 (no matrix cores), unlike the matmul2d _fn_sdpa_mma.
      _fn_sdpa_paged_flash, _fn_sdpa_paged_mma, _fn_kv_write_paged,
      // MLX steel register-softmax flash with a paged K/V tg loader (mid-
      // context prefill, q_offset>0 -- no de-paged kfull scratch). No function
      // constants (bounds/causal handled in-kernel) -> a plain cached function.
      _fn_steel_paged,
      // Shared-prefix batched decode attention (head_dim 256): phase A reads
      // the N branches' shared prefix once, phase B merges per-branch private.
      _fn_sdpa_shared_mb256, _fn_sdpa_merge_mb256,
      // Flash-decode-GQA serial attention (head_dim <= 256): read each KV
      // head ONCE for all Hq/Hkv query heads (phase A) + position-split merge
      // (phase B), vs mb256/mb128's GQA-redundant per-q-head re-scan.
      _fn_sdpa_gqa, _fn_sdpa_gqa_vec, _fn_sdpa_gqa_kbl, _fn_sdpa_gqa_kbl128,
      _fn_sdpa_gqa_vec1, _fn_sdpa_gqa_vec2, _fn_sdpa_gqa_merge2,
      _fn_sdpa_gqa_merge,
      _fn_gdn_step, _fn_gdn_step_ndv4, _fn_gdn_conv1d, _fn_gdn_g_beta, _fn_mrope,
      _fn_gdn_qk_norm, _fn_gdn_gated_rms,
      // batched-decode GEMV (4-bit, MAXM=2; N>2 tiles along grid.z)
      _fn_qmv_batch, _fn_qmv_batch_swiglu,
      // MAXM=4 twins: the MTP verify at draft depth>=2 (n=3..4) reads each
      // weight ONCE instead of the MAXM=2 form's 2 grid.z tiles (the depth-2
      // cliff). Bit-identical per row; gated to the verify path (see _qmv4_*).
      _fn_qmv_batch4, _fn_qmv8_batch4, _fn_qmv_batch4_swiglu,
      // MAXM=8 grouped-x tall tile (xp2) for m=7..8 (see _qmv8_enabled),
      // its fused-swiglu twin, and the w8 twin (OptiQ mixed verify).
      _fn_qmv_batch8_xp, _fn_qmv_batch8_xp_swiglu, _fn_qmv8_batch8_xp,
      // Matrix-core prefill: 4-bit -> dense expand + dense matmul2d GEMM,
      // the interleaved-gate/up SwiGLU combine for the matrix-core MLP, and
      // the matrix-core flash attention (head_dim 256, drop-in for qtile).
      _fn_dequant, _fn_dense_mma, _fn_dense_mma_deep, _fn_swiglu_inter,
      _fn_sdpa_mma,        // head_dim 256 (Qwen3.5)
      _fn_sdpa_mma_d128,   // head_dim 128 (Llama-3 / Qwen3-VL text encoder)
      _fn_embed, _fn_argmax, _fn_sample,   // pipelined-decode kernels
      // Unquantized dense f16 in-stream embed gather (raw-HF path): row gather
      // from the [vocab, H] f16 table for decode_step_fast (the muxer covers
      // forward/prefill via its dense-table ctor).
      _fn_embed_dense,
      // Two-stage parallel argmax + histogram multi-tg sampler (the defaults;
      // _fn_argmax/_fn_sample are the VPIPE_QWEN_ARGMAX1/SAMPLE1 single-tg
      // fallbacks). Same kernels as Gemma's llm_elementwise.metal.
      _fn_argmax_partial, _fn_argmax_combine,
      _fn_smp_max_partial, _fn_smp_max_combine,
      _fn_smp_zhist_partial, _fn_smp_zhist_combine,
      _fn_smp_thresh, _fn_smp_pick_partial, _fn_smp_pick_combine,
      // On-GPU Leviathan-Chen MTP correction (lc_sample / lc_accept): the
      // nucleus + accept/residual that the host L-C path used to grind over the
      // full vocab on the CPU, done one-threadgroup-per-row on the GPU.
      _fn_lc_sample, _fn_lc_accept, _fn_lc_sample_batch,
      // Native k-quant (GGUF): per-family qmv (decode) + dequant-to-f16
      // (prefill) + plain dense f16 GEMV/GEMM + q6_K embed gather/lm_head.
      _fn_qmv_q4k, _fn_qmv_q5k, _fn_qmv_q6k, _fn_embed_q6k, _fn_embed_q4k,
      // Batched k-quant GEMV (compile-time MAXM=2, weight read once across the
      // 2-row tile, grid.z tiles larger M): the MTP verify's weight-bound matmul.
      _fn_qmv_q4k_batch, _fn_qmv_q5k_batch, _fn_qmv_q6k_batch,
      // MAXM=4 twins (depth-2 verify n=3..4 in one tile -> weight read ONCE vs
      // the MAXM=2 form's 2 grid.z tiles; the k-quant depth-2 cliff). _qmv4_*.
      _fn_qmv_q4k_batch4, _fn_qmv_q5k_batch4, _fn_qmv_q6k_batch4,
      _fn_dequant_q4k, _fn_dequant_q5k, _fn_dequant_q6k, _fn_copy,
      _fn_dequant_w4g32,   // affine g32 prefill dequant (Q4_K->affine repack)
      // Mixture-of-Experts (Qwen3.5-MoE): on-GPU router (softmax+top-k+renorm),
      // gathered expert gate|up+SwiGLU and down GEMVs (index a 3D expert slab),
      // the shared-expert sigmoid gate (w8 dot), and the combine/finalize.
      _fn_moe_route, _fn_moe_gather_swiglu, _fn_moe_gather_down,
      _fn_moe_gate, _fn_moe_combine, _fn_moe_finalize,
      _fn_moe_finalize_combined,
      // Grouped prefill (counting-sort + segmented MAXM-batched expert GEMV):
      // reads each expert's weight once per tile instead of once per token.
      _fn_moe_grouped_swiglu, _fn_moe_grouped_down, _fn_moe_ifill,
      _fn_moe_hist, _fn_moe_sort_setup, _fn_moe_scatter, _fn_moe_scatter_back,
      // Steel (matrix-tiled) grouped expert GEMM + the sorted-row gather.
      _fn_moe_qmm_grouped, _fn_moe_qmm_grouped_swiglu,
      // 8-bit twins of the routed-expert kernels above (selected at dispatch
      // when the experts are w8-quantized; router/shared gates stay w8).
      _fn_moe_gather_swiglu_w8, _fn_moe_gather_down_w8,
      _fn_moe_grouped_swiglu_w8, _fn_moe_grouped_down_w8,
      _fn_moe_qmm_grouped_w8, _fn_moe_qmm_grouped_swiglu_w8,
      // Dense f16 MoE gather GEMVs (raw-HF bf16 MoE): the dense twins of the
      // affine gather_swiglu / gather_down / moe_gate above.
      _fn_moe_gather_swiglu_dense, _fn_moe_gather_down_dense,
      _fn_moe_gate_dense,
      _fn_dense_gemv, _fn_dense_gemm;
  // Decode full-attn switches to the multi-simdgroup paged kernel once
  // the context reaches this length (scalar wins at short ctx).
  int _sdpa_mb_min = 128;
  // Shared-prefix batched decode attention (head_dim 256): when the N
  // branches of a batched step share a prefix, read it ONCE (phase A) then
  // merge each branch's private pages (phase B), instead of N per-branch
  // SDPAs each re-reading the shared K/V. Default on; VPIPE_QWEN_SHARED_ATTN
  // =0 forces the per-branch path (A/B + safety). Requires the shared/merge
  // kernels to have loaded.
  bool _shared_attn = true;
  // Flash-decode-GQA serial attention (head_dim 256, GQA G=Hq/Hkv<=4). When
  // on, serial decode full-attn reads each KV head ONCE for all G query
  // heads (sdpa_paged_gqa_mb256 + sdpa_gqa_merge) instead of mb256's
  // per-q-head re-scan (G x the KV bandwidth). _gqa_split position-slices the
  // scan across Hkv*split single-simdgroup threadgroups for occupancy.
  // Capability-gated at load (D==256, G<=4, kernels valid);
  // VPIPE_QWEN_GQA_ATTN=0/1 overrides, VPIPE_QWEN_GQA_SPLIT sets the split.
  bool _gqa_attn = false;
  // Latency-optimal decode form: per-head simdgroup with UK key-unroll + vec4
  // (sdpa_paged_gqa_vec) -- ~2x the all-G mb256 form (head_dim % 128 == 0).
  // VPIPE_GQA_NO_VEC=1 forces the all-G kernel (A/B).
  bool _gqa_vec = true;
  bool _gqa_blk = false;   // key-across-lanes block decode (head_dim 256;
                           // default ON when available, VPIPE_GQA_BLK=0 opts out)
  int  _gqa_split = 32;
  bool _gqa_split_fixed = false;   // true if VPIPE_GQA_SPLIT pinned it
  // Context-adaptive flash-decode split: the optimal #position-splits grows
  // with KV length (~pos/128 per the gqa_decode_micro sweep: 16@2k, 64@8k,
  // 128@16k, 256@32k). A fixed 32 was 15-20% off at 16k-32k. Pinned by env.
  int gqa_split_for(int pos) const {
    if (_gqa_split_fixed) { return _gqa_split; }
    const int s = (pos + 127) / 128;
    return s < 16 ? 16 : (s > 256 ? 256 : s);
  }
  // Adaptive batched GEMV: read each weight ONCE per MAXM rows vs the MAXM=2
  // form's ceil(m/2) grid.z tiles. qmm_auto_ et al. pick MAXM by row count m:
  // m in 3..4 -> MAXM=4 (one tile), else m>1 -> MAXM=2, else m==1 -> plain qmv.
  // MAXM=4 is the MTP verify's depth-2 cliff win (n=3..4, the >L2 lm_head).
  // m=5..6 (realtime-vqa batched decode) STAYS on MAXM=2. The REGISTER-
  // resident MAXM=8 form blows the register budget (occupancy collapse,
  // ~4x slower at m=5..8) and must not come back; m=7..8 instead uses the
  // GROUPED-x xp2 kernel (affine_qmv_batch8_xp2_w4g64: weight packs hoisted
  // to registers, x register-resident 2 rows at a time) -- one weight read
  // for 8 rows at a ~43 GB/s pass vs MAXM=2's ~129, a measured ~1.3x win on
  // the >=14MB matrices whose grid.z re-reads are NOT cache-served (the
  // qmv_batch_bandwidth_sweep audit; o-proj-size matrices ARE SLC-served and
  // neutral). Bit-identical per row. _qmv4_enabled gates the MAXM=4 tier
  // (VPIPE_MTP_QMV4); _qmv8_enabled gates the xp2 tier (VPIPE_QMV_XP8),
  // affine-4bit only.
  bool _qmv4_enabled = false;
  bool _qmv8_enabled = false;
  // m=5..6 heterogeneous 2-read plan: one MAXM=4 tile (rows 0..3) + one
  // MAXM=2 tile (rows 4..m-1) instead of the homogeneous 3-tile MAXM=2
  // plan's 3 weight reads -- ~1.17x on the >=14MB matrices, bit-identical
  // per row (each row runs a verified kernel). VPIPE_QMV_MIX56=0 reverts.
  bool _qmv_mix56 = false;
  // ---- Per-machine batched-GEMV LADDER (probed at first decode) -------
  // Every tile kernel is bit-identical per row, so the per-m plan is a
  // pure PERF choice, and plan costs are ADDITIVE in tile times (the
  // encoder serializes dispatches; measured within 1% on M5). Probing
  // just THREE single-tile times -- T2 (MAXM=2), T4 (xp4), T8 (xp2) --
  // on real gate|up weights CYCLED ACROSS LAYERS (so every dispatch is
  // DRAM-cold regardless of the machine's SLC size) therefore resolves
  // the whole ladder per machine: bigger-SLC / more-core chips (M4 Pro /
  // Max / Ultra) that re-serve tile re-reads or under-fill the tall
  // one-read kernels get their own crossovers instead of this M5's.
  // Static defaults (= the M5-measured ladder) apply when the probe is
  // skipped (VPIPE_QMV_AUTOTUNE=0, MoE/mixed/k-quant, missing kernels).
  enum class QmvPlan : std::uint8_t {
    kTile2,   // MAXM=2, grid.z = ceil(m/2)
    kXp4,     // xp4 tiles, grid.z = ceil(m/4)
    kXp8,     // one xp2/xh16 tall tile
    kMix4R,   // xp4 head (rows 0..3) + MAXM=2 tail (rows 4..m-1)
  };
  QmvPlan _qmv_plan[kQmvBatchMaxRows + 1] = {};
  bool _qmv_plan_probed = false;
  void qmv_ladder_defaults_();
  void autotune_qmv_ladder_();
  // Matrix-core prefill path (M5+). Set at load when the GPU has matrix
  // cores (MetalCompute::supports_matrix_cores()) and both the dequant +
  // dense matmul2d kernels validated. When on, prefill projections with
  // enough rows route through dequant-once + dense matmul2d (the matrix
  // units) instead of the steel quantized GEMM; decode + small M stay on
  // the existing paths. Off -> behaviour is byte-identical to before.
  bool _use_mma = false;
  // Dynamic-int8 accelerated prefill GEMMs (set_i8_gemm); null when off.
  std::unique_ptr<I8GemmContext> _i8;
  // Min rows (M) to prefer the matrix-core GEMM. Below this the steel
  // path's lower fixed cost + no dequant pass wins. Override with
  // VPIPE_QWEN_MMA_MIN_M.
  int _mma_min_m = 64;
  // Min prefill length (n) to prefer the matrix-core flash attention over
  // the scalar query-tiled kernel (attention is tiny below this; matmul2d
  // tile overhead isn't repaid). Override with VPIPE_QWEN_MMA_ATTN_MIN_N.
  int _mma_attn_min_n = 384;
  // Prefer the simdgroup_matrix key-split flash prefill kernel over the scalar
  // qtile on the non-matrix-core (M4) path. Default on when the kernel loads;
  // VPIPE_QWEN_NO_FLASH=1 forces qtile (A/B + safety).
  bool _flash_attn = true;
  // Reusable dequant scratch for the matrix-core GEMM weight (grown to the
  // largest projection N*K seen). One projection at a time, so a single
  // buffer suffices; measured: the dequant pass overlaps surrounding work
  // and is ~0% of prefill (VPIPE_QWEN_MMA_NODQ A/B), so no double-buffer.
  metal_compute::SharedBuffer _w_deq;
  // Diagnostic (VPIPE_QWEN_MMA_NODQ=1): skip the dequant dispatch in the
  // matrix-core GEMM path -> dense GEMM runs on STALE weights (garbage
  // output) but identical GPU work minus the dequant. A/B vs the normal
  // path isolates the dequant share of prefill. Never set in production.
  bool _skip_dequant = false;
  // Diagnostic (VPIPE_QWEN_GDN_V1=1): force the per-dv recurrent GDN step
  // even in prefill (instead of the ndv4 variant). A/B for the GDN speedup.
  bool _gdn_force_v1 = false;
  // Decode category profiler: enabled at load by VPIPE_QWEN_CATPROF. Only
  // then does encode_decode_step_ read VPIPE_QWEN_DUP_CAT per step (so a
  // within-process profiler can toggle the duplicated category between
  // passes); production never pays the per-step getenv.
  bool _catprof = false;

  // ---- Native k-quant (GGUF) decode/prefill dispatch ------------------
  // Decode (M=1): one qmv per typed weight into an output offset.
  void kqmv_(metal_compute::ComputeEncoder& enc, KQ type,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer& x, std::size_t xoff,
             const metal_compute::SharedBuffer& y, std::size_t yoff,
             int K, int N);
  // Batched multi-row decode (the MTP verify draft window): y[m, yoff + 0:N] =
  // dequant(w[N,K]) @ x[m,0:K] with the raw k-quant weight read ONCE across each
  // 2-row tile (compile-time MAXM=2; grid.z tiles ceil(M/2)). At depth-1 (M=2)
  // a 2-token verify costs ~1 decode step. Output rows are `ystride` apart
  // (ystride==N is contiguous [M,N]; a wider stride writes a column slice of a
  // fused buffer at `yoff`). Falls back to looped single-row qmv when the
  // batched kernel is unavailable or VPIPE_GGUF_MTP_LOOPED_GEMV is set; M==1 ->
  // plain qmv.
  void kqmv_batch_(metal_compute::ComputeEncoder& enc, KQ type,
                   const metal_compute::SharedBuffer& w,
                   const metal_compute::SharedBuffer& x,
                   const metal_compute::SharedBuffer& y, int K, int N, int M,
                   int ystride, int yoff);
  // Prefill (M rows): dequant the k-quant weight into the reusable f16
  // scratch _w_deq, then a dense f16 GEMM y[M,N] = x[M,K] @ wdeq[N,K]^T.
  void kqmm_(metal_compute::ComputeEncoder& enc, KQ type,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer& x,
             const metal_compute::SharedBuffer& y, int K, int N, int M);
  // Fused-prefill primitives: dequant a k-quant part (or copy an f16 part)
  // into the shared scratch _w_deq at an element offset, then one dense GEMM
  // covers the whole [Ntot, K] -- the heterogeneous types vanish post-dequant.
  void kdequant_(metal_compute::ComputeEncoder& enc, KQ type,
                 const metal_compute::SharedBuffer& w, std::size_t dst_off,
                 int count);
  void kcopy_(metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& src, std::size_t dst_off,
              int count);
  // f16 GEMV / GEMM (the dequant'd a/b path + the kqmm_ dense stage).
  void dense_gemv_(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& w,
                   const metal_compute::SharedBuffer& x, std::size_t xoff,
                   const metal_compute::SharedBuffer& y, std::size_t yoff,
                   int K, int N);
  void dense_gemm_(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& w,
                   const metal_compute::SharedBuffer& x,
                   const metal_compute::SharedBuffer& y, int K, int N, int M);
  // Raw k-quant embed gather: out[t,:] = dequant(_embed_q6k[ids[t], :]).
  // Picks the Q4_K / Q6_K gather kernel by _embed_kqt (the _embed_q6k buffer
  // holds the raw table for either family).
  void embed_q6k_(metal_compute::ComputeEncoder& enc,
                  const metal_compute::SharedBuffer& ids, std::size_t ioff,
                  const metal_compute::SharedBuffer& out, int n);

  // ---- Mixed-precision affine (OptiQ) dispatch ------------------------
  // Dequant ONE affine projection part (`bits` 4 or 8, group 64) into the
  // shared f16 scratch _w_deq at element offset `dst_off`, for a fused
  // prefill GEMM (heterogeneous-bit parts share one f16 scratch, then a
  // single dense_gemm_ -- the affine twin of kdequant_).
  void adequant_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& w,
                 const metal_compute::SharedBuffer& s,
                 const metal_compute::SharedBuffer& b, int bits,
                 std::size_t dst_off, int N, int K);
  // Prefill a single affine projection (M rows): adequant the whole weight
  // into _w_deq then a dense f16 GEMM (matrix-core on M5). The affine twin
  // of kqmm_; used for the standalone o/out/down/gate/up projections.
  void aqmm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer& s,
             const metal_compute::SharedBuffer& b, int bits,
             const metal_compute::SharedBuffer& x,
             const metal_compute::SharedBuffer& y, int K, int N, int M);
  // Affine-4bit-g32 prefill twins (the Q4_K->affine repack path): dequant the
  // repacked weight into _w_deq (at dst_off for the fused q|k|v path), and the
  // standalone projection prefill (adequant_g32 + dense GEMM).
  void adequant_g32_(metal_compute::ComputeEncoder& enc,
                     const metal_compute::SharedBuffer& w,
                     const metal_compute::SharedBuffer& s,
                     const metal_compute::SharedBuffer& b,
                     std::size_t dst_off, int N, int K);
  void aqmm_g32_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& w,
                 const metal_compute::SharedBuffer& s,
                 const metal_compute::SharedBuffer& b,
                 const metal_compute::SharedBuffer& x,
                 const metal_compute::SharedBuffer& y, int K, int N, int M);
  // Batched (multi-branch) decode over a repacked Q4_K projection: loops the
  // affine g32 qmv per branch row (mirrors kqmv_batch_'s looped fallback).
  void amv_g32_batch_(metal_compute::ComputeEncoder& enc,
                      const metal_compute::SharedBuffer& w,
                      const metal_compute::SharedBuffer& s,
                      const metal_compute::SharedBuffer& b,
                      const metal_compute::SharedBuffer& x,
                      const metal_compute::SharedBuffer& y, int K, int N,
                      int M, int ystride, int yoff);

  // ---- MTP head (Multi-Token Prediction drafter) ----------------------
  // A bundled mtp.safetensors: the eh-fusion `fc` ([H, 2H], dense f16) +
  // pre-fc norms for the embedding and the carried hidden, ONE full-attention
  // decoder layer (same shape as a main full-attn layer; 4-bit affine), and a
  // final norm before the SHARED lm_head. Drives mtp_decode's drafter:
  //   combined = fc( [ norm_e(embed(tok)) | norm_h(h_prev) ] )   ([2H] -> [H])
  //   h_mtp    = mtp_layer(combined)         (own KV; local per-round positions)
  //   draft    = argmax(lm_head(mtp_norm(h_mtp)))
  // The MTP layer attends in its OWN small paged context (_mtp_ctx, reset each
  // round), so it never perturbs the main model's KV. All linears here are
  // 4-bit affine (uniform) regardless of the main model's mixed precision.
  struct MtpHead {
    bool ok = false;
    Layer lyr;                                 // the full-attn decoder layer
    // The eh-fusion fc, [H, 2H], split at load into its embedding/hidden
    // halves (each [H, H], contiguous): combined = fc_e @ norm_e(emb) +
    // fc_h @ norm_h(hidden). Splitting avoids interleaving the two normed
    // inputs into a [n, 2H] buffer (rms can't write a strided sub-row).
    metal_compute::SharedBuffer fc_e, fc_h;
    metal_compute::SharedBuffer prenorm_e;     // pre_fc_norm_embedding [H]
    metal_compute::SharedBuffer prenorm_h;     // pre_fc_norm_hidden    [H]
    metal_compute::SharedBuffer final_norm;    // mtp.norm [H]
    std::unique_ptr<ContextManager> ctx;       // 1 full-attn layer
    ContextId cid;
    metal_compute::SharedBuffer pgt;           // MTP-ctx page table
    // Optional 4-bit affine DRAFT lm_head (group 64), requantized at load from
    // the 8-bit tied embed. Read by the MTP draft's vocab GEMV ONLY (the
    // verifier keeps the 8-bit head), halving its bandwidth -- the draft is
    // verify-corrected so the output stays exact. Empty => use the 8-bit head.
    metal_compute::SharedBuffer dlm_w, dlm_s, dlm_b;
    bool draft_head = false;
  };
  MtpHead _mtp;
  // Last-position pre-final-norm hidden of the most recent prefill / verify
  // forward [H] -- the main-model hidden the MTP drafter consumes.
  metal_compute::SharedBuffer _mtp_h;
  // MTP prefix seed (opt-in, _mtp_seed_enabled): the last text prefill's
  // per-position POST-final-norm main hiddens [s, H] (f16) + its token ids, so
  // mtp_decode can populate the MTP head's self-attention KV with the prompt
  // before drafting -- the drafter then attends over the whole prompt, not just
  // the decode tail (closes the draft-accept gap vs a full-history runtime).
  // Captured by prefill() when enabled; set _valid; mtp_seed_prefix_ consumes
  // them (without clearing, so repeated decodes from one prefill seed
  // identically). Empty / !_valid (multimodal prefill, no MTP head, seed
  // disabled, or VPIPE_MTP_NO_SEED) => no seed.
  metal_compute::SharedBuffer _mtp_prefix_h;
  std::vector<std::int32_t> _mtp_prefix_ids;
  int _mtp_prefix_len = 0;
  bool _mtp_prefix_valid = false;
  // Gate for capture+seed (set_mtp_prefix_seed). Default OFF: prefill pays
  // nothing unless a stage opts in; realtime-vqa keeps it off.
  bool _mtp_seed_enabled = false;

  // Per-(linear/GDN layer) cache of the verify's GDN recurrent-step INPUTS:
  // the conv input qkv [maxK, Cd] (f16) and g / beta [maxK, Hv] (f32). The
  // main batched verify writes these (free -- it just retargets the GDN scratch
  // to per-layer buffers), so a partial accept can re-advance ONLY the GDN
  // recurrent state (conv1d -> qk_norm -> gdn_step over the kept tokens, from
  // the round's S0 snapshot) instead of re-forwarding the whole main model.
  // Indexed in linear-layer order (same order as the snapshot's glayers).
  // Allocated once per mtp_decode and reused across rounds.
  struct GdnVerifyCache {
    std::vector<int>                         layers;
    std::vector<metal_compute::SharedBuffer> qkv, gbuf, betabuf;
  };

  // (The sibling mtp.safetensors is loaded inline in load(), reusing the same
  // weight reader + 4-bit qtri/fuse/interleave helpers as the main layers.)
  // Reset the MTP head's local KV context (a fresh drafting window per verify).
  void mtp_ctx_reset_();
  // Seed the MTP head's KV from the captured prefix (the last prefill's
  // positions): condition position t on (post-norm hidden_t, embed of the token
  // at t+1; the last seeded position uses `first_token`, the decode's first
  // token). KV-only (no attention/draft -- the written K/V is a pure projection
  // of each position's input, so it is bit-identical to running the full MTP
  // step), one command buffer. The caller resets the MTP ctx first. Tail-caps
  // to the MTP page (RoPE attention is shift-invariant, so a tail seeded at
  // slots 0.. keeps every relative distance). Returns the number of positions
  // seeded (0 if no prefix was captured). Called at mtp_decode start (persist).
  int mtp_seed_prefix_(std::int32_t first_token);

  // Direct quantized batched matmul y[m,N] = x[m,K] @ dequant_bits(w)^T,
  // reading the quantized weight ONCE across the m rows -- NO f16 dequant
  // scratch (unlike the prefill forward_chunk_, whose dequant-to-f16 moves ~9x
  // the bandwidth, ruinous at the tiny m of a speculative verify). 4-bit:
  // qmm_auto_ (qmv_batch MAXM=2 / qmv / steel by m). 8-bit: qmv (m==1) / steel
  // (_fn_qmm8; no 8-bit batched-GEMV). The MTP batched-verify chokepoint.
  void vqmm_(metal_compute::ComputeEncoder& enc, int m,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer& s,
             const metal_compute::SharedBuffer& b, int bits,
             const metal_compute::SharedBuffer& xin,
             const metal_compute::SharedBuffer& y, int K, int N);
  // Single-token flash-decode attention via the MLX sdpa_vector 2-pass kernels
  // (vec2 pass-1 + coalesced merge2), shared by the single-token decode, the
  // MTP per-draft verify loop, and the batched per-branch fallback. Writes the
  // [Hq,D] attention for query `q`+qoff (at global position `pos`) into
  // `out`+outoff via the shared _d_gqa_oacc/m/l scratch (serial per call).
  // Split = MLX block count (must be %32 for merge2).
  static int gqa_vec2_split(int pos) { return pos <= 8192 ? 128 : 256; }
  // env (VPIPE_QWEN_GQA_VEC2) override else the per-regime autotune for `pos`.
  bool gqa_vec2_on(int D, int pos) const;
  // Tuned #position-splits for `pos`'s regime (the GQA split coefficient is
  // autotuned jointly with the kernel); falls back to the heuristic on override.
  int  gqa_decode_split(int D, int pos) const;
  void encode_gqa_vec2_(metal_compute::ComputeEncoder& enc,
      const metal_compute::SharedBuffer& q, std::size_t qoff,
      const metal_compute::SharedBuffer& kp,
      const metal_compute::SharedBuffer& vp,
      const metal_compute::SharedBuffer& out, std::size_t outoff,
      int pos, float scale, int D, int Hq, int Hkv,
      int page_tokens, int n_pages,
      const metal_compute::SharedBuffer& pgtab, std::size_t pgoff);
  // The decode GQA attention KERNEL SET (vec2/kbl, paged, f16/bf16): owns its
  // member kernels + partial scratch, autotunes kernel+split per machine/regime
  // at load, and exposes the unified dispatch() the decode sites tap into. The
  // model no longer carries the kernel selection / autotune / dispatch itself.
  DecodeGqaAttnSet _decode_set;
  // The prefill GQA attention set (steel/flash/qtile/scalar/mma): picks the
  // member per query-count regime at load (discovers the steel/flash crossover
  // per GPU). The prefill site taps its dispatch().
  PrefillGqaAttnSet _prefill_set;
  // Debug-log (via the session delegate) the resolved decode-attn kernel per
  // regime at load. No-op without a session or a vec2/kbl-capable model.
  void log_decode_attn_choice_() const;
  // MTP batched verify (direct-quantized), with the MTP head FUSED in. One
  // forward over the n drafts: the main model's weight-bound matmuls run as
  // direct quantized batched-GEMV (vqmm_), causal multi-token attention, and
  // batched GDN (in-place; the caller snapshots/restores it for partial-accept
  // rollback). Fills `preds` (the main model's per-position greedy argmax) and
  // leaves each position's pre-final-norm hidden in `x`. When `mtp_preds` is
  // non-null the MTP head ALSO runs in the SAME command buffer -- it consumes
  // each position's main hidden + the embedding of that position's main pred
  // and emits the next-next-token draft (the "verify is a decode step that
  // also drafts" fusion, so no separate mtp_draft_ forward is needed). Returns
  // false on append failure. Mixed-precision only (the de-fused weights).
  // mtp_preds2 (optional, depth-2): chain a SECOND MTP application from the
  // depth-1 MTP hidden + draft, in the SAME command buffer (a second window
  // appended to the MTP ctx). Fills the depth-2 next-next-next-token drafts.
  // `rope_delta` is added to the main model's RoPE position (the verify token's
  // KV slot) so a post-multimodal decode uses the mROPE-advanced position while
  // attention masking still keys off the true KV slot; 0 for sequential text.
  // `sp` drives the per-position decision (greedy argmax or, for speculative
  // sampling, a sampled token); when sampling, position k is seeded by its
  // absolute KV slot via `seed_slot0` (the seq_len at decode start) so it is
  // byte-identical to decode_pipelined's per-step seed for that slot.
  bool mtp_verify_chunk_(ContextId cid, const metal_compute::SharedBuffer& x,
                         int n, std::vector<std::int32_t>* preds,
                         std::vector<std::int32_t>* mtp_preds = nullptr,
                         std::vector<std::int32_t>* mtp_preds2 = nullptr,
                         int rope_delta = 0, const GpuSamplerParams& sp = {},
                         int seed_slot0 = 0, GdnVerifyCache* gcache = nullptr,
                         bool lc_mode = false, bool gdn_ring = false,
                         const metal_compute::SharedBuffer* mtp_cond = nullptr,
                         const std::function<
                             void(metal_compute::ComputeEncoder&)>&
                             pre_commit = {});

  // Partial-accept GDN rollback. With every linear layer's conv/ssm recurrent
  // state restored to the round's S0 snapshot, replay conv1d -> qk_norm ->
  // gdn_step for the first `keep` tokens from the verify's cached GDN inputs
  // (gc), advancing each layer's state to S_keep. Reuses the exact GDN kernels
  // on the exact inputs, so the recurrent state is bit-identical to a fresh
  // keep-token forward (token-exact). GDN-only -- no attention / projections /
  // lm_head -- so it replaces the full main-model re-forward the partial accept
  // used to pay.
  void gdn_replay_(ContextId cid, int keep, const GdnVerifyCache& gc);

  std::vector<Layer> _layers;
  metal_compute::SharedBuffer _embed_w, _embed_s, _embed_b;   // tied lm_head
  metal_compute::SharedBuffer _lm_w, _lm_s, _lm_b;            // untied (9B)
  metal_compute::SharedBuffer _final_ln;
  metal_compute::SharedBuffer _inv_freq;                      // [rotary_dim/2]
  // Constant weight vectors [gdn_k_dim] for the GDN q/k RMS-no-weight via
  // rms_norm_f16: q uses inv_scale^2, k uses inv_scale (inv_scale=1/sqrt(Dk)).
  metal_compute::SharedBuffer _gdn_qscale, _gdn_kscale;
  bool _tied = true;
  // Native k-quant (GGUF Q4_K_M etc.): heterogeneous per-tensor quant
  // (q4_K/q5_K/q6_K), no requant. Set at load when the weights are a
  // qwen35 GGUF; selects the kqmv_/kqmm_ dispatch + the q6_K embed/lm_head.
  bool _kquant = false;
  // q6_K decode qmv tuning: the loaded qmv_q6k_v2 variant's NSG (simdgroups/
  // threadgroup) and rows-per-threadgroup (RPS*NSG). Tuned to r2n8 (nsg=8,
  // rpt=16); fallbacks set nsg=2/rpt=8. Used by kqmv_'s q6_K dispatch.
  int _q6k_nsg = 2;
  int _q6k_rpt = 8;
  // Mixed-precision affine (mlx-optiq): some linear weights are 4-bit, some
  // 8-bit, in one checkpoint (per-tensor sensitivity quant). Set at load by
  // probing the on-disk packed-weight shapes; selects the per-tensor
  // de-fused affine dispatch (Layer.*_bits + the separate triples) instead
  // of the fused single-bit-width path. Mutually exclusive with _kquant.
  bool _mixed = false;
  // Unquantized dense f16 (raw-HF bf16 checkpoint, narrowed to the compute
  // element at load): each linear is a plain [N,K] f16 weight in the same
  // fused/interleaved *w slot, with its *s/*b triples left EMPTY. The forward
  // dispatches the dense GEMM/GEMV when the scales buffer is empty. Mutually
  // exclusive with _kquant / _mixed / is_moe().
  bool _dense = false;
  // MOSS-TTS text artifact: an AFFINE backbone whose embed/tied-lm_head were
  // kept bf16 on disk (no .scales). The embed table + tied lm_head bind via
  // the dense f16 path (muxer gather + dense GEMV head) so the full-precision
  // head is not double-quantized, while the backbone stays affine. The
  // backbone linears still dispatch the affine kernels (this flag is NOT
  // _dense). Mutually exclusive with _dense / _kquant.
  bool _dense_embed = false;
  // On-device AWQ calibration accumulators (per-layer per-channel |x| abs-max).
  bool _calib_on = false;
  std::vector<std::vector<float>> _calib_qkv, _calib_gu, _calib_dn;
  // Streaming per-expert accumulators: [n_layers][n_experts*hidden] routed
  // gate/up input abs-max, [n_layers][n_experts*moe_inner] down input abs-max.
  std::vector<std::vector<float>> _calib_eg, _calib_ed;
  // Affine bit width of the (tied) embed table / lm_head. OptiQ keeps these
  // at 8-bit; the muxer gather + lm_head qmv pick the matching kernel.
  int  _embed_bits = 4;
  int  _lm_bits = 4;
  bool _embed_is_q6k = false;               // raw k-quant embed table present
  metal_compute::SharedBuffer _embed_q6k;   // raw k-quant embed table (Q6_K
                                            // tied, or Q4_K when untied)
  KQ _embed_kqt = KQ::kQ6K;                  // embed table k-quant family
  // Untied k-quant lm_head (GGUF output.weight; Q6_K on the 27B). When _tied
  // the lm_head matvec reuses _embed_q6k / _embed_kqt instead.
  metal_compute::SharedBuffer _lm_q6k;
  KQ _lm_kqt = KQ::kNone;
  // The lm_head raw table + family: the separate output.weight when untied,
  // else the (tied) embed table. Used by the k-quant lm_head matvec sites.
  const metal_compute::SharedBuffer& lm_head_kq_() const {
    return _tied ? _embed_q6k : _lm_q6k;
  }
  KQ lm_head_kqt_() const { return _tied ? _embed_kqt : _lm_kqt; }
  // Per-pair mROPE axis lookup [rotary_dim/2] (0=T,1=H,2=W); built from
  // mrope_section at load. Empty -> all-T (plain 1D).
  std::vector<std::uint8_t> _mrope_axis;

  std::unique_ptr<MetalTokenMuxer>     _muxer;
  std::unique_ptr<ContextManager> _ctx;
  ContextId                            _cid;

  metal_compute::SharedBuffer _pgtab;

  // Reused decode scratch (shared by forward() + decode_pipelined();
  // allocated lazily by ensure_decode_scratch_()). One token's worth.
  bool _dec_ready = false;
  metal_compute::SharedBuffer _d_x, _d_hn, _d_logits, _d_qfull, _d_kbuf,
      _d_vbuf, _d_q3, _d_gate3, _d_attn, _d_mixqkv, _d_zbuf, _d_abuf,
      _d_bbuf, _d_convout, _d_gbuf, _d_betabuf, _d_ygdn, _d_normout, _d_sg,
      // k-quant decode: [H] temp for qmv-then-residual-add (o/out/down),
      // and [ffn] up-proj temp (gate/up run as two qmv then swiglu).
      _d_radd, _d_up;
  // MoE decode scratch (allocated only when _cfg.is_moe()). logits [E];
  // routed ids [top_k] (int) + weights [top_k]; expert act [top_k*moe_inner];
  // down partials [top_k*H]; combined [H]; shared-expert swiglu [shared_inner]
  // + down [H]; sigmoid gate [1].
  metal_compute::SharedBuffer _d_moe_logits, _d_moe_ids, _d_moe_w,
      _d_moe_act, _d_moe_part, _d_moe_out, _d_moe_ssg, _d_moe_sout, _d_moe_gate;
  // Single-zero pair_eid for the dense-MoE shared-expert gather (the shared
  // expert is one slab -> "expert 0", top_k=1). Allocated lazily in load.
  metal_compute::SharedBuffer _moe_zero_eid;
  // 1-int scratch for decode_step_fast: input token id (in-stream embed)
  // and on-GPU argmax output.
  metal_compute::SharedBuffer _d_tok_in, _d_argmax_id;
  // Two-stage-argmax + histogram-sampler scratch (mirrors Gemma). Allocated
  // ONCE by ensure_sampler_scratch_() and reused across every decode path
  // (serial encoder serialises the WAR hazard). kArgmaxM/kSampM = stage-1 tgs,
  // kSampB = histogram bins (MUST match sample_*_f16 in llm_elementwise.metal).
  bool _smp_scratch_ready = false;
  static constexpr int kArgmaxM = 64;
  static constexpr int kSampM = 64;
  static constexpr int kSampB = 1024;
  metal_compute::SharedBuffer _d_argmax_part;
  metal_compute::SharedBuffer _d_smp_maxpart, _d_smp_hpart, _d_smp_hist,
      _d_smp_maxl, _d_smp_wt, _d_smp_pickpart;
  // Flash-decode-GQA partials (f32): un-normalized O [Hq,split,D], m/l
  // [Hq,split]. Allocated only when _gqa_attn (D==256). Reused across layers.
  metal_compute::SharedBuffer _d_gqa_oacc, _d_gqa_m, _d_gqa_l;
  // Leviathan-Chen MTP sampling: opt-in (set_leviathan / VPIPE_MTP_LEVIATHAN).
  // _lc_vlogits / _lc_mlogits hold the verify's full verifier / MTP-head logits
  // [n,vocab] (copied off the reused vlogits) so mtp_decode can do the ratio
  // test + residual/bonus sampling on the host (UMA). Default exact-match.
  bool _leviathan = false;
  // Persistent MTP self-attention KV: keep the 1-layer MTP head's K/V over the
  // committed decode positions (vs the per-round wipe in mtp_ctx_reset_) so each
  // draft attends over the decode history -- measured +0.25 depth-1 draft
  // accuracy (0.63->0.88) on a fixed sequence, the whole vpipe->reference gap.
  // Set per-decode from VPIPE_MTP_NO_PERSIST in mtp_decode (default ON).
  bool _mtp_persist = false;
  // _lc_mlogits2 holds the SECOND chained MTP-head application's logits (the
  // depth-2 draft q2); _lc_mlogits is the first (q1), _lc_vlogits the verifier.
  metal_compute::SharedBuffer _lc_vlogits, _lc_mlogits, _lc_mlogits2;
  // On-GPU L-C (set_lc_gpu / VPIPE_MTP_GPU_LC). _lc_ws_p / _lc_ws_q are the
  // per-row [vocab] weight scratch the nucleus stage caches into (reused across
  // the round's serial dispatches); _lc_qcarry / _lc_qcarry2 hold the carried
  // drafter logit rows for the pending d1 / d2 (copied out before the verify
  // overwrites _lc_mlogits next round); _lc_out is the small [<=16] token-id
  // readback. Lazily allocated on first GPU-L-C decode.
  bool _lc_gpu = false;
  metal_compute::SharedBuffer _lc_ws_p, _lc_ws_q, _lc_qcarry, _lc_qcarry2,
                              _lc_out;
  // Batched-sample (lc_sample_batch_f16) extras: _lc_ws_p is sized [K*vocab]
  // for the K concurrent rows; _lc_seed_in holds the per-row 32-bit seeds.
  metal_compute::SharedBuffer _lc_seed_in;
};

}  // namespace vpipe::genai

#endif
