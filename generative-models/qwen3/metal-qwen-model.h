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
    // Whether q_proj carries a per-head output gate (Qwen3.5: q_proj
    // emits 2*qd, the second half sigmoid-gates the attention output).
    // Qwen3-ASR has no gate (q_proj emits qd).
    bool  attn_output_gate = true;
    // Backbone-only load: skip the token-embedding muxer + lm_head (and the
    // tied/affine embed table). For host-fed-embedding models that supply
    // their own embeddings and output heads (MOSS-TTS: 33 bf16 code/text
    // heads + 32 audio-code embeddings on top of a dense Qwen3 backbone).
    // forward_embeddings_hidden() is the only forward path in this mode.
    bool  backbone_only = false;
    // Weight-name root prepended before "model." / "lm_head." Qwen3.5-VL
    // nests the LM under "language_model."; Qwen3-ASR is at the root ("").
    std::string weight_prefix = "language_model.";
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

  // Toggle the shared-prefix batched decode attention at runtime (default set
  // at load from VPIPE_QWEN_SHARED_ATTN + kernel availability). For A/B tests
  // and benchmarks that compare the shared-prefix path against per-branch.
  void set_shared_attn(bool on) noexcept { _shared_attn = on; }
  bool shared_attn() const noexcept { return _shared_attn; }

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
  // Mixed-precision affine (OptiQ) de-fused per-tensor path engaged (some
  // 4-bit, some 8-bit linears). For the token-exact OptiQ guard test.
  bool uses_mixed_precision() const noexcept { return _mixed; }

private:
  MetalQwenModel() = default;

  // Allocate the reused decode scratch buffers (lazy, first decode).
  bool ensure_decode_scratch_();

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
  std::vector<float> forward_chunk_(
      ContextId cid, const metal_compute::SharedBuffer& x, int n,
      const metal_compute::SharedBuffer* mrope_cos,
      const metal_compute::SharedBuffer* mrope_sin,
      bool verify_all = false, std::vector<std::int32_t>* preds_out = nullptr,
      bool return_hidden = false,
      metal_compute::SharedBuffer* hidden_out = nullptr);

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
        logits;
    metal_compute::SharedBuffer pgt;             // [N * max_pages * 3] int32
    metal_compute::SharedBuffer tok_in;          // [N] int32 (embed gather in)
    metal_compute::SharedBuffer argmax_id;       // [N] int32 (greedy out)
    // Shared-prefix attention partials (phase A -> phase B), f32:
    //   shacc [N, Hq, D] un-normalized O, shm/shl [N, Hq] online-softmax m/l.
    metal_compute::SharedBuffer shacc, shm, shl;
  };
  bool ensure_bscratch_(BScratch& bs, int n);
  BScratch _bdec;   // cached batched-decode scratch (reused across steps)

  // Size-based quantized-matmul entrance: y[M,Nout] = x[M,K] @ dequant(w)^T,
  // picking the kernel by the row count M (what MLX's quantized_matmul does
  // internally; the raw metal path needs it explicit). M==1 -> qmv (matvec);
  // 2 <= M <= kQmvBatchMaxRows -> batched GEMV (qmv bandwidth, weights read
  // once across the rows, MAXM=2 + grid.z tiling); larger M -> steel GEMM.
  // 8-bit weights have no batched-GEMV -> qmv (M==1) / steel. This is THE
  // place the batched decode adapts to the (shrinking) active branch count.
  static constexpr int kQmvBatchMaxRows = 8;
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
  struct BDecode {
    std::vector<ContextId>      cids;        // the N branch contexts
    std::vector<int>            rope_base;   // mROPE anchor / -1 sequential
    std::vector<int>            kv_base;     // KV seq_len at begin per branch
    metal_compute::SharedBuffer gen_ids;     // [cap][N] int32 GPU chain
    metal_compute::SharedBuffer seen;        // [N][vocab] uint8 (sampled)
    metal_compute::SharedBuffer sample_ws;   // [N][vocab] f16 (sampled)
    metal_compute::CommandStream stream;
    metal_compute::Event ev;
    metal_compute::CommandStream::Fence inflight;
    GpuSamplerParams sp;
    int           n             = 0;
    int           cap           = 0;   // gen_ids step capacity
    int           produced      = 0;   // step rows finalised in gen_ids
    int           pending       = -1;  // gen_ids row awaited by next()
    std::uint64_t gpu_step      = 0;
    bool          have_inflight = false;
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
      _lib_dense;
  metal_compute::ComputeFunction _fn_qmv, _fn_qmv_add, _fn_qmv_swiglu, _fn_qmm,
      _fn_qmm_swiglu,
      // Mixed-precision affine (OptiQ): the 8-bit counterparts of qmv/qmv_add/
      // qmm + the 8-bit dequant (affine_dequant_w8g64) + the 8-bit batched-GEMV
      // (affine_qmv_batch_w8g64, weight read once across MAXM=2 rows -- lets the
      // MTP verify batch its 8-bit tensors at qmv bandwidth instead of steel).
      // Loaded only when _mixed so a tensor at either bit width dispatches its
      // matching kernel (the 4-bit set above + these). _fn_dequant (w4) shared.
      _fn_qmv8, _fn_qmv8_add, _fn_qmm8, _fn_dequant8, _fn_qmv8_batch,
      _fn_transpose,
      _fn_rms, _fn_swiglu, _fn_residual, _fn_rope_partial, _fn_rms_rope,
      _fn_mul_sigmoid,
      _fn_head_slice, _fn_sdpa_paged, _fn_sdpa_paged_mb256, _fn_sdpa_paged_mb,
      _fn_sdpa_paged_qtile,
      // simdgroup_matrix key-split flash prefill (head_dim 256, drop-in for
      // qtile): runs on M4 (no matrix cores), unlike the matmul2d _fn_sdpa_mma.
      _fn_sdpa_paged_flash, _fn_kv_write_paged,
      // Shared-prefix batched decode attention (head_dim 256): phase A reads
      // the N branches' shared prefix once, phase B merges per-branch private.
      _fn_sdpa_shared_mb256, _fn_sdpa_merge_mb256,
      // Flash-decode-GQA serial attention (head_dim <= 256): read each KV
      // head ONCE for all Hq/Hkv query heads (phase A) + position-split merge
      // (phase B), vs mb256/mb128's GQA-redundant per-q-head re-scan.
      _fn_sdpa_gqa, _fn_sdpa_gqa_vec, _fn_sdpa_gqa_merge,
      _fn_gdn_step, _fn_gdn_step_ndv4, _fn_gdn_conv1d, _fn_gdn_g_beta, _fn_mrope,
      _fn_gdn_qk_norm, _fn_gdn_gated_rms,
      // batched-decode GEMV (4-bit, MAXM=2; N>2 tiles along grid.z)
      _fn_qmv_batch, _fn_qmv_batch_swiglu,
      // Matrix-core prefill: 4-bit -> dense expand + dense matmul2d GEMM,
      // the interleaved-gate/up SwiGLU combine for the matrix-core MLP, and
      // the matrix-core flash attention (head_dim 256, drop-in for qtile).
      _fn_dequant, _fn_dense_mma, _fn_dense_mma_deep, _fn_swiglu_inter,
      _fn_sdpa_mma,
      _fn_embed, _fn_argmax, _fn_sample,   // pipelined-decode kernels
      // Native k-quant (GGUF): per-family qmv (decode) + dequant-to-f16
      // (prefill) + plain dense f16 GEMV/GEMM + q6_K embed gather/lm_head.
      _fn_qmv_q4k, _fn_qmv_q5k, _fn_qmv_q6k, _fn_embed_q6k,
      // Batched k-quant GEMV (compile-time MAXM=2, weight read once across the
      // 2-row tile, grid.z tiles larger M): the MTP verify's weight-bound matmul.
      _fn_qmv_q4k_batch, _fn_qmv_q5k_batch, _fn_qmv_q6k_batch,
      _fn_dequant_q4k, _fn_dequant_q5k, _fn_dequant_q6k, _fn_copy,
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
  int  _gqa_split = 32;
  // Matrix-core prefill path (M5+). Set at load when the GPU has matrix
  // cores (MetalCompute::supports_matrix_cores()) and both the dequant +
  // dense matmul2d kernels validated. When on, prefill projections with
  // enough rows route through dequant-once + dense matmul2d (the matrix
  // units) instead of the steel quantized GEMM; decode + small M stay on
  // the existing paths. Off -> behaviour is byte-identical to before.
  bool _use_mma = false;
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
  // q6_K embed gather: out[t,:] = dequant(_embed_q6k[ids[t], :]).
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
  };
  MtpHead _mtp;
  // Last-position pre-final-norm hidden of the most recent prefill / verify
  // forward [H] -- the main-model hidden the MTP drafter consumes.
  metal_compute::SharedBuffer _mtp_h;

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
                         int seed_slot0 = 0, GdnVerifyCache* gcache = nullptr);

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
  // Mixed-precision affine (mlx-optiq): some linear weights are 4-bit, some
  // 8-bit, in one checkpoint (per-tensor sensitivity quant). Set at load by
  // probing the on-disk packed-weight shapes; selects the per-tensor
  // de-fused affine dispatch (Layer.*_bits + the separate triples) instead
  // of the fused single-bit-width path. Mutually exclusive with _kquant.
  bool _mixed = false;
  // Affine bit width of the (tied) embed table / lm_head. OptiQ keeps these
  // at 8-bit; the muxer gather + lm_head qmv pick the matching kernel.
  int  _embed_bits = 4;
  int  _lm_bits = 4;
  bool _embed_is_q6k = false;
  metal_compute::SharedBuffer _embed_q6k;   // raw Q6_K tied table
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
  // 1-int scratch for decode_step_fast: input token id (in-stream embed)
  // and on-GPU argmax output.
  metal_compute::SharedBuffer _d_tok_in, _d_argmax_id;
  // Flash-decode-GQA partials (f32): un-normalized O [Hq,split,D], m/l
  // [Hq,split]. Allocated only when _gqa_attn (D==256). Reused across layers.
  metal_compute::SharedBuffer _d_gqa_oacc, _d_gqa_m, _d_gqa_l;
};

}  // namespace vpipe::genai

#endif
