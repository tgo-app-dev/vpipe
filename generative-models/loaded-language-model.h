#ifndef VPIPE_GENERATIVE_MODELS_LOADED_LANGUAGE_MODEL_H
#define VPIPE_GENERATIVE_MODELS_LOADED_LANGUAGE_MODEL_H

#include "generative-models/context-manager.h"
#include "generative-models/model-loader.h"
#include "generative-models/token-muxer.h"
#include "generative-models/tokenizer.h"


#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace vpipe {
class MlxRuntime;
class SessionContextIntf;
}

namespace vpipe::genai {

struct SamplerParams;
class AudioEncoder;
class ChatTemplate;
class LlamaModelExec;
class MetalAudioEncoder;
class MetalGemma4AudioEncoder;
class MetalGemma4VisionEncoder;
class Gemma4UnifiedEmbedder;
class MetalQwenVisionEncoder;
class TokenMuxer;
class VisionEncoder;

// MLX-free compute-dtype selector for the LM forward. The MLX backend
// maps this to mlx::core::Dtype internally; the metal backend only needs
// the f16-vs-bf16 distinction. Keeps the public LoadedLanguageModel /
// GenerativeModelManager API free of mlx::core types so the subsystem
// builds without MLX.
enum class ComputeDtype { F16, BF16, F32 };

// Per-load handle returned by GenerativeModelManager::load(). Owns the
// weight map, tokenizer, context manager, token muxer, and the
// concrete ModelExec (LlamaModelExec for v1). The public API is the
// token-stream generator promised in the LLM design plan:
//   prefill(ctx, prompt_ids) -> first predicted id
//   next_token(ctx)          -> next predicted id (feeds the last
//                               predicted id internally)
//   branch(parent)           -> independent Context for fanout
//
// All forward passes go through tokenizer.encode (callers) ->
// TokenMuxer (embedding fetch) -> LlamaModelExec::forward_chunk
// (the per-layer pass). Text input takes the fast path; the muxer
// is wired for image-token mixing once a vision encoder lands.
//
// Thread safety: prefill / next_token / branch / make_context are
// internally serialised on a single mutex. MLX is single-stream
// per device, so this matches the underlying compute reality; the
// future batch-decode path will relax the serialisation.
class LoadedLanguageModel {
public:
  ~LoadedLanguageModel();

  LoadedLanguageModel(const LoadedLanguageModel&)            = delete;
  LoadedLanguageModel& operator=(const LoadedLanguageModel&) = delete;

  // Construction is the responsibility of GenerativeModelManager; the
  // constructor is public so make_unique works but the manager is
  // the only intended caller.
  //
  // `runtime` is the MLX executor. When non-null, every MLX op the
  // LM performs (the body of this constructor, every prefill /
  // next_token, the ctx_mgr's release path) is dispatched through
  // the runtime so all MLX state stays bound to its single worker
  // thread. When null (tests that bypass the manager and drive
  // ContextManager / LlamaModelExec directly on the test thread),
  // every op runs inline on the caller's thread.
  // `model_dir` is the on-disk model directory (for backends that
  // load their own weights, e.g. the metal-compute path selected by
  // the VPIPE_LLM_BACKEND=metal env flag). Empty for the default MLX
  // path, which binds the already-loaded `weights`.
  LoadedLanguageModel(LoadedWeights              weights,
                      std::unique_ptr<Tokenizer> tokenizer,
                      ComputeDtype               compute_dtype,
                      int                        page_tokens,
                      std::uint32_t              max_pages,
                      MlxRuntime*                runtime,
                      const SessionContextIntf*  session,
                      const std::string&         model_dir = {});

  // Built successfully? On any internal failure (missing weight,
  // model-spec mismatch with the ContextManager, etc.) the
  // constructor logs through `session` and sets valid()==false.
  // All accessors below are still safe to call but the generation
  // entry points will return -1.
  bool valid() const noexcept;

  const ModelConfig& config()    const noexcept;
  const Tokenizer&   tokenizer() const noexcept;

  // Per-model-family chat-template renderer. Built at load time by
  // dispatching on config().architecture. Returns nullptr when the
  // family has no registered template (raw prefill still works).
  // The pointer is stable for the lifetime of this LoadedLanguageModel.
  const ChatTemplate* chat_template() const noexcept;

  // Per-model-family vision tower. Built at load time when the
  // architecture is a known VLM (Qwen3.5-VL, ...). Returns nullptr
  // for text-only models. The returned encoder may be a stub whose
  // implemented() returns false -- callers should check that bit
  // and route around the encoder when the tower body isn't yet
  // implemented for the loaded family. The pointer is stable for
  // the lifetime of this LoadedLanguageModel.
  VisionEncoder* vision_encoder() const noexcept;

  // Metal-compute vision tower (Qwen3-VL on the metal backend, no MLX).
  // Non-null only when the metal backend is active and the model has a
  // vision config. encode() returns host-f32 image embeddings to splice
  // via prefill_multimodal_metal(). nullptr otherwise.
  MetalQwenVisionEncoder* metal_vision_encoder() const noexcept;

  // Metal-compute audio tower (Qwen3-ASR on the metal backend, no MLX).
  // Non-null only when the metal backend is active and the model has an
  // audio config. encode() returns host-f32 audio embeddings to splice
  // via prefill_multimodal_metal() (empty image_grids => 1-D RoPE).
  // nullptr otherwise.
  MetalAudioEncoder* metal_audio_encoder() const noexcept;

  // Metal-compute Gemma-4 vision / audio towers (no MLX). Non-null only
  // on the metal backend for the Gemma-4 family. The vision encode()
  // returns a native-f16 SharedBuffer; the audio encode() host f32; both
  // spliced via the owns_kv metal multimodal path
  // (prefill_multimodal_metal -> the MetalGemmaModelExec branch).
  MetalGemma4VisionEncoder* metal_gemma4_vision_encoder() const noexcept;
  MetalGemma4AudioEncoder*  metal_gemma4_audio_encoder() const noexcept;

  // Gemma-4-12B "unified" (gemma4_unified, GGUF): the encoder-less shallow
  // multimodal embedder (vision + audio). Non-null only on the metal
  // backend when a gemma4uv/gemma4ua mmproj was found. encode_image returns
  // host-f32 rows; spliced via the same owns_kv metal multimodal path.
  Gemma4UnifiedEmbedder* gemma4_unified_embedder() const noexcept;

  // Host-float logits ([vocab]) of the last position predicted by the
  // metal backend's prefill/next_token. The MLX-free sampling source
  // (the metal path's counterpart to last_logits()); empty on the MLX
  // path or before any forward. Valid until the next decode call.
  const std::vector<float>& last_logits_host() const noexcept;

  // Per-model-family audio tower (Qwen3-ASR et al). Same shape as
  // vision_encoder(): returns nullptr for non-audio architectures,
  // a stub encoder (implemented() == false) when the family has an
  // audio_config but vpipe doesn't yet ship a tower implementation
  // for it, or a real encoder otherwise. The pointer is stable for
  // the lifetime of this LoadedLanguageModel.
  AudioEncoder* audio_encoder() const noexcept;

  // RAII handle for a generation context. Holds the underlying
  // ContextId and the last predicted token id (so next_token() can
  // be called with no explicit argument). Move-only; releases the
  // underlying ContextId on destruction.
  class Context {
  public:
    Context() noexcept = default;
    ~Context();

    Context(Context&& o) noexcept;
    Context& operator=(Context&& o) noexcept;
    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;

    bool valid() const noexcept { return _lm != nullptr; }
    ContextId id() const noexcept { return _id; }

    // Length of the context's logical sequence, in tokens.
    int seq_len() const;

    // The most recent token id predicted by prefill / next_token.
    // -1 before the first call.
    std::int32_t last_predicted_id() const noexcept
    { return _last_predicted; }

    // Override the most-recently-predicted id. Used by samplers that
    // need to feed a sampled (rather than argmax) token as the input
    // to the next lazy decode step (next_token_lazy /
    // batched_next_token_lazy both read this field). Caller's
    // responsibility to pass an id within the model's vocab range.
    void set_last_predicted_id(std::int32_t id) noexcept
    { _last_predicted = id; }

  private:
    friend class LoadedLanguageModel;
    LoadedLanguageModel* _lm = nullptr;
    ContextId            _id;
    std::int32_t         _last_predicted = -1;
    // Next rotary position to use for the upcoming token. -1 means
    // "use ContextManager::seq_len_of(ctx)" (the default text-only
    // path). Set by prefill_multimodal when mrope position_ids
    // produce a non-trivial post-image-block offset; incremented by
    // each next_token after that.
    int                  _rope_next_position = -1;
  };

  // RAII handle that pins this LM's working set as wired-resident
  // GPU memory while alive. Mirrors mlx-vlm's `wired_limit`
  // context manager, but scoped at a coarser grain than a single
  // forward call so the residency-set commit cost (proportional
  // to the number of buffers in the model) amortises across the
  // whole generation. Hold one of these around a prefill +
  // decode-loop sequence:
  //
  //   auto wired = lm->wired_scope();
  //   lm->prefill(ctx, prompt);
  //   while (...) lm->next_token(ctx);
  //   // wired goes out of scope here -- one synchronize + restore
  //
  // Without an active WiredScope, generation calls execute with
  // wired_limit=0. Decode is then meaningfully slower for models
  // with heavy per-token state I/O (Qwen3.5 sees ~2x; Llama family
  // sees no measurable diff). Multiple overlapping WiredScopes
  // (e.g. one per concurrent context) refcount correctly: the
  // wired state is set on the first scope, restored when the last
  // exits, with a synchronize before restore to drain pending
  // GPU work.
  //
  // Costs: one residency-set resize on enter (single bulk commit
  // of every already-allocated buffer), one synchronize +
  // residency-set resize on exit. Negligible vs the cost of a
  // multi-token generation. Disabled when VPIPE_LLM_WIRED_LIMIT=0
  // is set (no-op).
  class WiredScope {
  public:
    WiredScope() noexcept = default;
    ~WiredScope();
    WiredScope(WiredScope&&) noexcept;
    WiredScope& operator=(WiredScope&&) noexcept;
    WiredScope(const WiredScope&)            = delete;
    WiredScope& operator=(const WiredScope&) = delete;

  private:
    friend class LoadedLanguageModel;
    LoadedLanguageModel* _lm = nullptr;
  };

  WiredScope wired_scope();

  // Mint a fresh root context.
  Context make_context();

  // Branch a parent context into an independent sibling. The
  // parent's partial tail page (if any) is frozen and shared by
  // refcount with the child; both sides allocate a fresh page on
  // their next prefill / next_token (no K/V copy needed). Common
  // use case: vision prefill once, branch per question.
  Context branch(const Context& parent);

  // Branch a parent context into N independent siblings in one call.
  // All N children share the parent's pages (including any partial
  // tail) by refcount; each child writes to its own freshly-allocated
  // page on first append. Returns an empty vector on failure.
  std::vector<Context> branch(const Context& parent, int n_branches);

  // Run prefill over `tokens`. Returns the argmax of the next-token
  // logits, or -1 on failure. Also stashes the predicted id on the
  // Context so a subsequent next_token() picks it up.
  std::int32_t prefill(Context&                          ctx,
                       std::span<const std::int32_t>     tokens);

  // Multimodal prefill: each TokenRef is either a text id (looked up
  // in the embedding table) or a borrow into a vision encoder's
  // image-token embedding buffer (TokenRef::Kind::ImageTokens). The
  // TokenMuxer fetches the right embedding row per ref; the result
  // is one [seq_len, hidden] tensor fed through forward_chunk in a
  // single pass. Returns the argmax of the next-token logits, or
  // -1 on failure.
  //
  // Callers retain ownership of any tensors referenced by
  // ImageTokens refs (typically VisionEncoder::EncodedImage members);
  // they must outlive the prefill_multimodal call.
  std::int32_t prefill_multimodal(Context&                  ctx,
                                  std::span<const TokenRef> refs);

  // Metal-backend multimodal prefill (MLX-free; works in both builds).
  // Each TokenRef is text (looked up in the metal embed table) or an
  // image/audio row referencing a host-f32 [rows, hidden] buffer via
  // TokenRef::embeddings_host + image_token_offset/audio_token_offset.
  // When `image_grids` is non-empty (one (grid_h, grid_w) per image-token
  // run, left-to-right), 3-axis interleaved mROPE positions are built and
  // ctx's post-image rotary position is set for subsequent next_token();
  // empty grids => plain 1-D RoPE (audio/text). Returns the next-token
  // argmax or -1. Requires the metal backend.
  std::int32_t prefill_multimodal_metal(
      Context&                                  ctx,
      std::span<const TokenRef>                 refs,
      std::span<const std::pair<int, int>>      image_grids);


  // Decode one step using the context's last predicted id.
  std::int32_t next_token(Context& ctx);

  // Decode one step using `forced` as the input token (overrides
  // the context's last-predicted bookkeeping). Useful for tests
  // and constrained-decoding loops.
  std::int32_t next_token(Context& ctx, std::int32_t forced);

  // Greedy-only decode: returns the argmax token id. On the metal backend
  // this folds the embed + argmax into the decode command buffer and skips
  // the per-token host logit read-back (last_logits_host() is NOT updated
  // -- callers that sample must use next_token() instead). Falls back to
  // next_token() otherwise. Same predicted id as next_token() for greedy.
  std::int32_t next_token_greedy(Context& ctx);
  std::int32_t next_token_greedy(Context& ctx, std::int32_t forced);

  // ---- GPU-resident pipelined decode (metal backend) ---------------
  // The production counterpart of the MLX step_pipelined path, for the
  // metal/no-MLX backend: a per-token streaming pipeline that keeps the
  // token chain on the GPU (in-stream embed + on-device argmax/sampling,
  // no host logit pull) and overlaps the host's per-token work with the
  // GPU's next forward. Per decode turn:
  //   pdecode_begin(ctx, first, prompt, params, max)  -- once
  //   pdecode_commit(ctx)   -- start the forward for the NEXT token
  //   pdecode_next(ctx)     -- await + return the in-flight token
  //   pdecode_end(ctx)      -- once (drains any in-flight commit)
  // The caller commits the step that consumes a token (and appends ITS
  // KV) only after confirming that token is not a stop token, so a stop
  // token's KV is never appended (matching next_token). `prompt` primes
  // the penalty seen-set; `params` is the same sampler config the host
  // Sampler would use (argmax configs run the GPU argmax kernel). Returns
  // false from begin when the backend can't pipeline (MLX execs, or a
  // metal model lacking the kernels) -- callers then fall back to the
  // synchronous next_token loop.
  bool         pdecode_begin(Context& ctx, std::int32_t first_token,
                             std::span<const std::int32_t> prompt,
                             const SamplerParams& params, int max_tokens);
  bool         pdecode_commit(Context& ctx);
  std::int32_t pdecode_next(Context& ctx);
  void         pdecode_end(Context& ctx);

  // True iff the backend's pdecode supports run-ahead (commit token N+1
  // before confirming token N isn't a stop; pdecode_end rolls back any
  // speculative KV). A driver uses it to choose the run-ahead loop vs the
  // synchronous "commit only after not-stop" loop.
  bool         pdecode_supports_runahead() const;

  // ---- Batched (N-branch parallel) decode -- metal backend ---------
  // One synchronous batched decode step over N active branch contexts that
  // share a prefix (VQA fanout): append in_tokens[i] to ctxs[i]'s K/V (at
  // its own position / mROPE-advanced rope), run ONE forward with the
  // weight-bound matmuls batched across all N branches, and fill
  // `out_logits` with the N rows of [vocab_size] f32 logits (row-major,
  // size N*vocab_size). The caller samples each row, feeds the chosen token
  // back as the next step's in_token, and drops a branch (shrinks `ctxs`)
  // once it stops. Branches need NOT share a seq_len. Returns false when the
  // backend has no batched-decode path (non-metal, or a model/exec not yet
  // wired) so the caller falls back to per-branch serial decode.
  bool         m_batched_decode_step(
                   std::span<Context*>           ctxs,
                   std::span<const std::int32_t> in_tokens,
                   std::vector<float>&           out_logits);
  // True when the metal backend exposes the batched-decode path above.
  bool         m_batched_decode_supported() const;

  // ---- Batched GPU-resident PIPELINED decode -- metal backend -------
  // The N-branch counterpart of the pdecode_* pipeline: the per-branch
  // sampled tokens stay GPU-resident (on-device sampler, no host logit
  // pull) and per-token command buffers chain via a GPU event so the
  // host's stop-check overlaps the next forward. Constant-N (all branches
  // carried; the caller simply stops collecting a branch's text once it
  // ends). Lifecycle: m_bdecode_begin once (first_tokens[i] = branch i's
  // already-decided first id), then m_bdecode_commit / m_bdecode_next per
  // step, m_bdecode_end once. next() fills the N sampled tokens for the
  // committed step. Returns false / unsupported -> fall back to
  // m_batched_decode_step. The Sampler params drive the GPU sampler.
  bool         m_bdecode_supported() const;
  bool         m_bdecode_begin(std::span<Context*>           ctxs,
                               std::span<const std::int32_t> first_tokens,
                               const SamplerParams& params, int max_tokens);
  bool         m_bdecode_commit();
  bool         m_bdecode_next(std::vector<std::int32_t>& out_tokens);
  void         m_bdecode_end();

  // ---- Reusable branch pool -- metal backend -----------------------
  // For stages that re-branch the same N children off a fresh parent every
  // scene (realtime-vqa: one image prefix, N question branches): pre-allocate
  // the branch contexts ONCE (their exec-private KV pages + SSM/conv buffers)
  // and reuse them, instead of branch()/release() each scene. Reserve N
  // pooled Contexts sized to ~max_tokens of private (question + answer)
  // tokens, then m_rebranch each off the new base before prefill. Returns an
  // empty vector / false when unsupported (the caller falls back to branch()).
  bool                 m_reserve_branches_supported() const;
  std::vector<Context> m_reserve_branches(int n, int max_tokens);
  // Reset a pooled `child` to a fresh branch of `parent`, reusing storage.
  bool                 m_rebranch(Context& child, const Context& parent);
  // Detach a pooled `child` from its current parent: release the pages it
  // shares with that parent (keeping the reserved slot + buffers) so the
  // next base context's prefill isn't starved by pages the pool would
  // otherwise pin across scenes. Returns false when unsupported.
  bool                 m_detach_branch(Context& child);

  // ---- MTP speculative decode (metal Qwen3.5-OptiQ) -----------------
  // True iff the loaded model carries a bundled MTP head (mtp.safetensors) and
  // the backend can drive greedy speculative decode. When false, callers use
  // the normal next_token / pdecode loop.
  bool mtp_available() const;

  // Speculative decode via the MTP head: token-exact vs the serial decode, but
  // the drafter lets the verifier accept multiple tokens per forward.
  // `first_token` is the prefill's already-decided first token on `ctx` (not
  // yet appended); MTP generates from it, appending up to `max_tokens` total
  // tokens (including first_token) to `ctx`. `params` selects the VERIFY: a
  // greedy (argmax) sampler reproduces serial greedy decode; a non-greedy
  // sampler reproduces serial sampling (speculative sampling -- the committed
  // tokens are exactly autoregressive samples from the main model; the drafter
  // only changes acceptance rate). The verify does NOT apply repetition /
  // presence penalties, so a caller that wants those must pass a penalty-free
  // sampler (else fall back to the normal loop). `is_stop` ends the decode at
  // the first accepted stop token WITHOUT keeping it (its speculative KV is
  // rolled back so the context ends cleanly). `on_tokens(span)` is invoked per
  // spec round with that round's newly accepted (non-stop) tokens, for
  // streaming; return false to abort. `*produced` (optional) gets the number of
  // tokens generated (excluding the stop token); `*hit_stop` (optional) is true
  // iff a stop token ended the decode. Returns false when MTP is unavailable
  // (the caller then falls back to the normal loop).
  bool mtp_generate(
      Context& ctx, std::int32_t first_token, int max_tokens,
      const SamplerParams&                                      params,
      const std::function<bool(std::int32_t)>&                  is_stop,
      const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
      int* produced = nullptr, bool* hit_stop = nullptr);


  // ---- Per-stage profile (for perf comparison vs mlx-lm) ------
  // Enables eval() fences around each forward-pass stage in the
  // underlying LlamaModelExec; per-stage wall time accumulates in
  // ProfileTotals. The fences add stall overhead, so totals are an
  // upper bound -- the useful signal is the *relative* split.
  struct ProfileTotals {
    double qkv_proj_s   = 0.0;
    double rope_kv_s    = 0.0;
    double sdpa_s       = 0.0;
    double o_proj_s     = 0.0;
    double mlp_proj_s   = 0.0;
    double mlp_act_s    = 0.0;
    double final_head_s = 0.0;
    int    n_forwards   = 0;
  };
  void set_profile(bool b);
  ProfileTotals profile_totals() const;
  void profile_reset();

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;

};

// Build mROPE 3-axis position_ids from a TokenRef stream + per-image
// merged-grid dims. This is the same algorithm
// LoadedLanguageModel::prefill_multimodal uses internally; exposed as
// a free function so unit tests can validate it against mlx-vlm's
// get_rope_index outputs without needing a real model load.
//
//   refs[i].kind == Text       -> position[*][i] = cur, cur++
//   refs[i].kind == ImageTokens
//     (one continuous run of N_k = mh_k*mw_k entries per image k)
//                              -> T = base, H = base + (off/mw),
//                                 W = base + (off%mw); base = cur at
//                                 start of image; after image,
//                                 cur = base + max(mh, mw)
//
// `image_grids[k]` is the post-merger (grid_h, grid_w) for the k-th
// image-token run encountered while walking `refs` left-to-right.
// `out_pos_t/h/w` are resized to `refs.size()` and filled in.
// `out_rope_next_position` (optional) is set to the value of `cur`
// after the last token, i.e. the rotary position to use for the FIRST
// decode token after the prefill. Returns false on grid/refs mismatch.
bool
build_mrope_position_ids(
    std::span<const TokenRef>                refs,
    std::span<const std::pair<int, int>>     image_grids,
    std::vector<std::int32_t>*               out_pos_t,
    std::vector<std::int32_t>*               out_pos_h,
    std::vector<std::int32_t>*               out_pos_w,
    int*                                     out_rope_next_position);

}

#endif
