#ifndef VPIPE_GENERATIVE_MODELS_MODEL_EXEC_H
#define VPIPE_GENERATIVE_MODELS_MODEL_EXEC_H

#include "generative-models/context-manager.h"
#include "apple-silicon/metal-compute/shared-buffer.h"


#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace vpipe::genai {

class TokenMuxer;

// GPU-resident sampler config for the pipelined decode path. Mirrors the
// host genai::SamplerParams; `greedy` is precomputed by the caller from
// Sampler::is_argmax() so the metal kernel picks the argmax kernel vs the
// full temperature/top-k/min-p/top-p/penalty sample kernel. `n_iter` is
// the threshold binary-search depth (shared by top-k and top-p).
struct GpuSamplerParams {
  bool          greedy             = true;
  float         temperature        = 1.0f;
  int           top_k              = 0;
  float         top_p              = 1.0f;
  float         min_p              = 0.0f;
  float         repetition_penalty = 1.0f;
  float         presence_penalty   = 0.0f;
  std::uint64_t seed               = 0;
  int           n_iter             = 24;
};

// Abstract LLM forward-pass driver. One concrete subclass per
// supported model architecture (LlamaModelExec for the dense
// Llama / Llama-style family; future MoE / VLM / linear-attention
// classes derive from this same interface).
//
// Lifetime: bound to a single LoadedLanguageModel; not safe to
// share across models because each exec hard-references its
// LoadedWeights and a ContextManager configured for THIS model's
// K/V geometry.
//
// Thread-safety: a single exec instance is owned by one thread at
// a time. LoadedLanguageModel serialises prefill / next_token
// across contexts of the same model with an internal mutex, so
// callers don't need to synchronise themselves.
class ModelExec {
public:
  virtual ~ModelExec() = default;

  // One-shot prefill. Runs the model over `tokens` against the
  // sequence already in `ctx`, appending their K/V slots to the
  // context manager and returning the argmax of the predicted
  // logits for the next position. Internally calls mlx::core::eval
  // so the K/V cache is materialised before returning.
  //
  // Returns -1 on failure (logged through session).
  virtual std::int32_t
  prefill(ContextId ctx, std::span<const std::int32_t> tokens) = 0;

  // Decode a single token: append `id`'s K/V to the context and
  // return the next-token argmax. Returns -1 on failure.
  virtual std::int32_t
  decode_one(ContextId ctx, std::int32_t id) = 0;


  // Host-float logits of the last predicted position [vocab_size].
  // The MLX-free path the metal backend + the sampler use (no
  // mlx::core::array). MLX execs may leave this empty (they expose
  // last_logits() instead); metal execs return their host buffer.
  virtual const std::vector<float>&
  last_logits_host() const
  {
    static const std::vector<float> kEmpty;
    return kEmpty;
  }

  // Did the constructor successfully bind every weight tensor the
  // forward pass needs? When false, prefill / decode_one / forward_
  // chunk all log and return -1.
  virtual bool valid() const noexcept = 0;

  // True when this exec owns its own per-context K/V cache (keyed by
  // ContextId) rather than threading K/V through the shared paged
  // ContextManager. When true, LoadedLanguageModel (a) drives prefill /
  // decode through the token-id path (prefill(tokens) / decode_one(id))
  // -- the metal execs embed + cache internally, and Gemma-4 needs the
  // raw ids for its per-layer-input embeddings -- and (b) forks / frees
  // this exec's KV on branch / release. The metal execs and
  // Gemma4ModelExec return true; the MLX dense / hybrid execs (which
  // keep K/V in the ContextManager) keep the default false.
  virtual bool owns_kv() const noexcept { return false; }

  // Fork exec-private per-context state from `parent` to `child` when
  // LoadedLanguageModel::branch forks the ContextManager. The default
  // is a no-op: MLX execs keep all per-context state in the
  // ContextManager (which branch()es itself), so they need nothing
  // here. The metal exec, which owns its own KV cache keyed by
  // ContextId, overrides this to copy the parent's KV to the child.
  // Returns false if the exec couldn't honor the branch.
  virtual bool branch_context(ContextId parent, ContextId child)
  {
    (void)parent;
    (void)child;
    return true;
  }

  // Release exec-private per-context state for `ctx`. Called by
  // LoadedLanguageModel::Context's destructor / move-assign when a
  // context is finished. The default is a no-op: MLX execs keep all
  // per-context K/V in the ContextManager, which LoadedLanguageModel
  // releases directly. The metal execs own their OWN KV-page pool keyed
  // by ContextId, so they override this to return the context's pages to
  // the pool (and drop the bookkeeping entry) -- without it every
  // make_context / branch on the metal backend leaked its pages, so a
  // streaming caller (realtime-vqa) exhausted the pool after a few
  // scenes and the next prefill failed with -1.
  virtual void release_context(ContextId ctx) { (void)ctx; }

  // ---- Reusable branch pool (own-KV execs) -------------------------
  // Lets a stage that re-branches the same N children off a fresh parent
  // every scene (realtime-vqa) pre-allocate the exec-private KV/state ONCE
  // and reuse it, instead of paying branch_context + release_context each
  // scene. Default: unsupported (the caller falls back to branch/release).
  virtual bool supports_branch_pool() const noexcept { return false; }

  // Pre-allocate a pooled, reusable KV/state slot for the (already-minted)
  // context id `child`, sized to hold at least `max_tokens` of appended
  // private tokens. The slot is kept until release_context(child). Returns
  // false if unsupported or allocation fails.
  virtual bool reserve_branch_context(ContextId child, int max_tokens)
  {
    (void)child;
    (void)max_tokens;
    return false;
  }

  // Reset the pooled `child` to a fresh branch of `parent`, REUSING the
  // child's KV/state storage (no allocation): the in-place counterpart of
  // branch_context. Returns false if unsupported or either id is unknown.
  virtual bool rebranch_context(ContextId parent, ContextId child)
  {
    (void)parent;
    (void)child;
    return false;
  }

  // Detach a pooled `child` (reserved via reserve_branch_context) from
  // its current parent: release the pages it shares with that parent so
  // a streaming caller doesn't pin them across scenes, KEEPING the
  // reserved slot + its KV/state buffers for the next rebranch. Default:
  // no-op (unsupported).
  virtual bool detach_branch_context(ContextId child)
  {
    (void)child;
    return false;
  }


  // Metal-backend multimodal prefill. `rows` is a [n*hidden] row-major
  // f32 embedding stream the caller has already spliced (text rows from
  // the metal embed table + pre-encoded audio/vision rows pulled to
  // host); the forward runs with plain 1-D RoPE and returns the
  // next-token argmax. Default -1: MLX execs use the forward_chunk*
  // family with mlx::core::array embeddings instead.
  virtual std::int32_t
  prefill_embeddings(ContextId ctx, const std::vector<float>& rows, int n)
  {
    (void)ctx;
    (void)rows;
    (void)n;
    return -1;
  }

  // Metal-backend text-embedding gather: dequantize the embed-table rows
  // for `ids` into a host f32 vector of length ids.size()*hidden. Used by
  // the metal multimodal splice to fill the text positions. Default
  // empty (MLX execs build embeddings through make_token_muxer()).
  virtual std::vector<float>
  embed_text_rows(std::span<const std::int32_t> ids)
  {
    (void)ids;
    return {};
  }

  // Metal-backend multimodal prefill with 3-axis mROPE. `rows` is the
  // [n*hidden] host-f32 spliced stream (text + image/vision rows);
  // `position_ids` is [3*n] (T/H/W axes) for interleaved mrope. Returns
  // the next-token argmax. Default -1.
  virtual std::int32_t
  prefill_embeddings_mrope(ContextId ctx, const std::vector<float>& rows,
                           const std::vector<std::int32_t>& position_ids,
                           int n)
  {
    (void)ctx;
    (void)rows;
    (void)position_ids;
    (void)n;
    return -1;
  }

  // ---- Native-f16 zero-copy splice (metal backend) -----------------
  // The caller pre-assembles the [n, hidden] spliced embeddings as a
  // SharedBuffer already in the model's compute dtype (f16), so the
  // forward skips the host-f32 -> GPU upload + f32->f16 cast that the
  // _mrope / _embeddings variants above incur. Image/vision rows are
  // copied straight from the encoder's f16 SharedBuffer (zero cast),
  // text rows from embed_text_buf(). Default -1 (MLX execs / bf16).
  virtual std::int32_t
  prefill_embeddings_buf(ContextId ctx, metal_compute::SharedBuffer&& rows,
                         int n)
  {
    (void)ctx;
    (void)rows;
    (void)n;
    return -1;
  }

  virtual std::int32_t
  prefill_embeddings_mrope_buf(ContextId ctx,
                               metal_compute::SharedBuffer&& rows,
                               const std::vector<std::int32_t>& position_ids,
                               int n)
  {
    (void)ctx;
    (void)rows;
    (void)position_ids;
    (void)n;
    return -1;
  }

  // Gather the text-embedding rows for `ids` into a [k, hidden]
  // SharedBuffer in the compute dtype (f16) -- the metal embed-table
  // gather without the host-f32 readback embed_text_rows() does. Empty
  // buffer on failure / non-metal exec.
  virtual metal_compute::SharedBuffer
  embed_text_buf(std::span<const std::int32_t> ids)
  {
    (void)ids;
    return {};
  }

  // Metal-backend decode with an explicit RoPE position override (used
  // after a multimodal prefill, where the next-token rotary position is
  // not the simple KV-slot position). Default ignores the override.
  virtual std::int32_t
  decode_one_at(ContextId ctx, std::int32_t id, int rope_pos)
  {
    (void)rope_pos;
    return decode_one(ctx, id);
  }

  // Greedy-only decode that may skip the host logit read-back (the caller
  // promises it only needs the argmax token, NOT last_logits()). Lets a
  // metal backend fold the embed + argmax into the decode command buffer
  // and return a single id. Default: the normal logit-producing path.
  virtual std::int32_t
  decode_one_greedy(ContextId ctx, std::int32_t id)
  {
    return decode_one(ctx, id);
  }
  virtual std::int32_t
  decode_one_greedy_at(ContextId ctx, std::int32_t id, int rope_pos)
  {
    return decode_one_at(ctx, id, rope_pos);
  }

  // ---- GPU-resident pipelined decode (metal backend) ----------------
  // A per-token streaming pipeline: the token chain stays on the GPU
  // (in-stream embed gather + on-device argmax/sample, no host logit
  // pull) and the host's per-token work (stop check + detokenize + emit)
  // overlaps the GPU's NEXT forward. Lifecycle per decode turn:
  //   pdecode_begin(ctx, first, prompt, sp, max)  -- once; first_token is
  //       the already-decided first generated id; prompt primes the
  //       penalty seen-set
  //   pdecode_commit(ctx)   -- encode + commit ONE forward (input = the
  //       last produced token, GPU-resident) WITHOUT waiting
  //   pdecode_next(ctx)     -- await the in-flight commit, return its id
  //   pdecode_end(ctx)      -- once; drains any in-flight commit
  // CONTRACT: pdecode_commit() (which starts the step that consumes a
  // token as input and appends ITS KV) must be called only after the
  // host has confirmed that token is not a stop token -- so a stop
  // token's KV is never appended, exactly matching the synchronous loop
  // (the context manager has no append rollback). Default: unsupported
  // (MLX execs); metal execs override. begin returns false if the
  // backend can't pipeline, so callers fall back to the sync loop.
  // `rope_first` is the rotary position of the first decode token: -1
  // for sequential text decode (rope == KV slot), or the mROPE-advanced
  // position after a multimodal prefill (where rope diverges from the KV
  // slot but still advances by 1 per token).
  virtual bool
  pdecode_begin(ContextId ctx, std::int32_t first_token,
                std::span<const std::int32_t> prompt,
                const GpuSamplerParams& sp, int max_tokens, int rope_first)
  {
    (void)ctx; (void)first_token; (void)prompt; (void)sp; (void)max_tokens;
    (void)rope_first;
    return false;
  }
  virtual bool pdecode_commit(ContextId ctx) { (void)ctx; return false; }
  virtual std::int32_t pdecode_next(ContextId ctx) { (void)ctx; return -1; }
  virtual void pdecode_end(ContextId ctx) { (void)ctx; }

  // True iff this backend's pdecode supports run-ahead: the driver may
  // commit token N+1's forward BEFORE confirming token N isn't a stop,
  // because pdecode_end rolls back any speculative KV past the last
  // produced token (hiding the per-token encode bubble -- the GPU runs the
  // next forward while the host detokenizes AND while the CPU is free).
  // Backends without KV append-rollback (paged) return false, so the driver
  // must keep the synchronous "commit only after not-stop" contract.
  virtual bool pdecode_supports_runahead() const { return false; }

  // ---- Batched (N-branch parallel) decode (metal execs) -------------
  // One synchronous batched decode step over N active branch contexts that
  // share a prefix (VQA fanout): append in_tokens[i] to ctx[i]'s K/V at
  // rope_pos[i] (< 0 => the KV slot position, i.e. sequential text), run ONE
  // forward with the weight-bound matmuls batched across all N branches, and
  // fill out_logits (size N*vocab, row-major) with each branch's [vocab] f32
  // logits. Branches need NOT be at the same seq_len. Returns false when the
  // exec has no batched-decode path (MLX execs, or a model not yet wired) so
  // the caller falls back to per-branch serial decode. Default: unsupported.
  virtual bool supports_batched_decode() const { return false; }
  virtual bool
  batched_decode_logits(std::span<const ContextId>      cids,
                        std::span<const std::int32_t>   in_tokens,
                        std::span<const std::int32_t>   rope_pos,
                        std::vector<float>&             out_logits)
  {
    (void)cids; (void)in_tokens; (void)rope_pos; (void)out_logits;
    return false;
  }

  // ---- Batched GPU-resident pipelined decode (metal execs) ----------
  // The N-branch counterpart of pdecode_*: the per-branch sampled tokens
  // stay GPU-resident (a per-step GPU sampler writes the next ids, the
  // next step's batched embed gathers them) so logits never round-trip,
  // and per-token command buffers chain via a GPU event so the host's
  // stop-check overlaps the next forward. Constant-N: all branches are
  // carried; the caller stops collecting a branch once it ends (the
  // weight read is amortised across N, so a finished branch costs only
  // its cheap per-branch attention). first_tokens[i] = branch i's already
  // -decided first id; rope_pos[i] >= 0 overrides branch i's mROPE anchor
  // (empty span / < 0 => sequential text). Lifecycle mirrors pdecode_*.
  // Default: unsupported; metal execs override. Returns false to fall
  // back to the synchronous batched_decode_logits / serial decode.
  virtual bool supports_batched_pipelined_decode() const { return false; }
  virtual bool
  bdecode_begin(std::span<const ContextId>    cids,
                std::span<const std::int32_t> first_tokens,
                const GpuSamplerParams& sp, int max_tokens,
                std::span<const std::int32_t> rope_pos)
  {
    (void)cids; (void)first_tokens; (void)sp; (void)max_tokens; (void)rope_pos;
    return false;
  }
  virtual bool bdecode_commit() { return false; }
  virtual bool bdecode_next(std::vector<std::int32_t>& out_tokens)
  { (void)out_tokens; return false; }
  virtual void bdecode_end() {}

  // ---- MTP speculative decode (metal Qwen3.5-OptiQ) ------------------
  // True iff this exec carries a bundled MTP head (mtp.safetensors) and can
  // drive greedy speculative decode. Default: unsupported.
  virtual bool supports_mtp() const { return false; }

  // Enable seeding the MTP drafter's KV with the prompt at decode start (a
  // prefill-throughput vs decode-throughput tradeoff; see MetalQwenModel::
  // set_mtp_prefix_seed). Default: no-op (only the metal Qwen exec honors it).
  virtual void set_mtp_prefix_seed(bool on) { (void)on; }

  // Speculative decode with the MTP head as the drafter and this model as the
  // verifier. Generates from `first_token` (the prefill's already-decided first
  // token, NOT yet appended), appending up to `max_tokens` total tokens to
  // `ctx`. `rope_first` is the rotary position of `first_token` (-1 =>
  // sequential text; >= 0 => the mROPE-advanced position after a multimodal
  // prefill). `sp` drives the VERIFY: greedy accepts a draft iff it ==
  // argmax(P), a sampler accepts iff it == a token sampled from P (per-position
  // seed) -- either way the committed tokens are exactly autoregressive
  // greedy/sampling from the main model (the drafter only affects speed).
  // `is_stop` (optional) ends the decode at the first accepted stop token
  // WITHOUT keeping it (its speculative KV is rolled back), matching the serial
  // loop. `on_tokens` is invoked per spec round with that round's newly accepted
  // (non-stop) tokens for streaming; return false to abort. `*produced`
  // (optional) gets the token count (excluding the stop token); `*hit_stop`
  // (optional) is true iff a stop token ended it. Token-exact vs the serial
  // decode (greedy: argmax loop; sampling: decode_pipelined with the same
  // per-slot seed). Default: unsupported.
  virtual bool
  mtp_generate(ContextId ctx, std::int32_t first_token, int max_tokens,
               int rope_first, const GpuSamplerParams& sp,
               const std::function<bool(std::int32_t)>&                 is_stop,
               const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
               int* produced, bool* hit_stop)
  {
    (void)ctx; (void)first_token; (void)max_tokens; (void)rope_first; (void)sp;
    (void)is_stop; (void)on_tokens; (void)produced; (void)hit_stop;
    return false;
  }


  // Toggle per-layer eval inside the forward loop. Used by
  // LoadedLanguageModel's warmup: the first forward triggers the
  // disk-bound quantized-weight reads and must serialise per layer
  // to stay under Metal's GPU watchdog; after warmup the exec turns
  // this off and lets MLX pack the whole forward into one (or a
  // few) command buffers for ~2x decode speedup.
  virtual void set_eval_per_layer(bool b) noexcept = 0;

};

}

#endif
