#ifndef VPIPE_GENERATIVE_MODELS_CONTEXT_MANAGER_H
#define VPIPE_GENERATIVE_MODELS_CONTEXT_MANAGER_H

#include "common/session-member.h"
#include "apple-silicon/metal-compute/shared-buffer.h"


#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace vpipe {
class SessionContextIntf;
}
namespace vpipe::metal_compute {
class MetalCompute;
}

namespace vpipe::genai {

// Strong typedefs so PageId and ContextId can't be accidentally
// swapped in callers (both are 32-bit ids but they index different
// tables).
struct PageId    { std::uint32_t v = kInvalid;
                   static constexpr std::uint32_t kInvalid =
                       0xFFFFFFFFu;
                   bool valid() const noexcept { return v != kInvalid; } };
struct ContextId { std::uint32_t v = kInvalid;
                   static constexpr std::uint32_t kInvalid =
                       0xFFFFFFFFu;
                   bool valid() const noexcept { return v != kInvalid; } };

// Paged K/V cache topology for one model load.
//
// Storage backend: the K/V page pool and the SSM (conv/ssm) recurrent
// state live in metal_compute::SharedBuffer (the kpool / vpool /
// fill_page_table / conv_state / ssm_state API). The exec encodes
// kv_write_paged / sdpa_paged into its own command buffer using the
// borrowed pool buffers, on the apple-silicon axis with no MLX.
// The page bookkeeping below (acquire/branch/release/append, refcount,
// per-page valid counts) is shared by every layout.
//
// Why "tree":
//   * A root context is acquired (acquire_root).
//   * branch() forks the parent's prefix into N independent children
//     that initially see the same logical sequence and then diverge
//     as the model decodes different continuations. Common use case
//     is asking a VLM several questions on one image -- prefill the
//     image once, branch N times, decode N answers.
//
// Branch semantics (no-CoW design):
//   * branch() FREEZES the parent's partial tail page (if any).
//     Children share every page of the parent by refcount, including
//     the now-frozen partial tail. The frozen page contributes its
//     recorded valid-token count (M = seq_len % page_tokens) to
//     gather_kv. NO data copy happens; the tail is shared
//     immutably.
//   * After branch, the parent's tail page is also marked frozen.
//     The next append on the parent OR any child allocates a fresh
//     private page; writes go to slot 0 of that fresh page (NOT
//     slot M of the shared frozen page). This wastes at most
//     (page_tokens - M) slots of pool space, which is acceptable
//     given page_tokens is small.
//   * Logical positions stay continuous: a child's first appended
//     token after the branch sits at position seq_len_parent, not
//     at the page boundary. gather_kv() respects per-page
//     n_valid so attention sees exactly the originally-written
//     tokens with no zero-padding artefacts.
//
// Per-page valid-token tracking: pages_valid_of() returns the
// per-page valid-token counts in the same order as pages_of(); the
// last entry corresponds to the page currently being written to (or
// the frozen tail if no fresh page was allocated yet).
//
// Thread safety: every public method is MT-safe (a single mutex
// guards all state). Append-heavy workloads serialize on that
// mutex; for v1 there's one context per pipeline stage in flight,
// so contention is negligible.
class ContextManager : public SessionMember {
public:
  // Per-layer attention kind. Full = classical paged/contiguous K/V;
  // Sliding = trailing-window attention (bounded KV in the contiguous
  // layout); Linear = linear-attention/SSM (no K/V, conv+ssm state).
  enum class LayerKind : std::uint8_t { Full = 0, Sliding = 1, Linear = 2 };

  // K/V storage layout for the metal backend. Paged = the per-layer
  // page pool (Llama/Qwen). Contiguous = per-(context,layer) contiguous
  // K/V buffers with per-layer head_dim, cross-layer KV sharing, and
  // bounded sliding windows (Gemma-4). The MLX backend is always paged.
  enum class KvLayout : std::uint8_t { Paged = 0, Contiguous = 1 };

  struct Spec {
    // Tokens per page. Tunable. Smaller pages waste less when
    // sequences are short; larger pages reduce indirection.
    int           page_tokens = 16;

    // Page-pool capacity. Total tokens across all live contexts in
    // the context tree is bounded by page_tokens * max_pages.
    // Allocation past this limit fails soft (append returns an
    // invalid slot, logged through session).
    //
    // Memory model: the per-layer K/V pool tensors are NOT sized to
    // max_pages at construction. Instead the pool starts at one page
    // (so a freshly-loaded model holds the smallest possible K/V
    // working set) and grows one page at a time, staying one page
    // ahead of the highest page-id ever claimed. Growth runs once
    // per claimed page (i.e. once per page_tokens decode steps) and
    // copies the existing pool data into the larger tensor via
    // slice_update; the amortised cost is negligible compared to a
    // forward pass. max_pages is the hard cap on the leading dim.
    std::uint32_t max_pages   = 4096;

    // K/V cache dimensions. When n_layers > 0 the manager
    // allocates per-layer K and V page pools (each shape
    // [max_pages, n_kv_heads, page_tokens, head_dim]) at
    // construction and supports write_kv / gather_kv. When
    // n_layers == 0 (the default for v1 bookkeeping-only tests)
    // no pool is allocated; write_kv/gather_kv are no-ops and
    // branch() skips the partial-tail data copy (only the
    // bookkeeping fires).
    int                n_layers   = 0;
    int                n_kv_heads = 0;
    int                head_dim   = 0;

    // ---- metal-compute backend -------------------------------------
    // When non-null, the manager allocates its K/V page pools and SSM
    // recurrent state as metal_compute::SharedBuffer (f16 KV / conv,
    // f32 ssm) instead of mlx::core::array, and the kpool/vpool/
    // fill_page_table/conv_state/ssm_state API is live (the MLX
    // write_kv/gather_kv API is inert). The two backends are mutually
    // exclusive per manager instance.
    metal_compute::MetalCompute* metal = nullptr;

    // Per-context logical length cap (metal backend only). Also used to
    // derive max_pages when max_pages == 0 (enough pages for several
    // full-length contexts: ceil(max_seq/page_tokens) * 16). For the
    // Contiguous layout this is the Full-layer K/V capacity.
    int                max_seq    = 0;

    // ---- Contiguous KV layout (metal backend, Gemma-4) -------------
    // When kv_layout == Contiguous, the manager allocates per-(context,
    // layer) contiguous K/V buffers [n_kv_heads, cap(L), head_dim(L)]
    // f16 instead of the shared paged pool, supporting per-layer
    // head_dim, cross-layer KV sharing, and bounded sliding windows.
    // Selected by the model (Gemma); Llama/Qwen keep Paged.
    KvLayout           kv_layout  = KvLayout::Paged;

    // Per-layer attention kind (size n_layers; empty => all Full).
    // Sliding layers attend only the last `sliding_window` tokens and
    // (Phase 2) store a bounded ring; Linear layers carry SSM state.
    // When non-empty this SUPERSEDES is_linear_layer (a Linear entry
    // here == is_linear_layer[L]); is_linear_layer stays as the
    // back-compat shorthand for hybrid models that only need Linear.
    std::vector<LayerKind> layer_kind;

    // Per-layer K/V head_dim (size n_layers; empty => uniform head_dim).
    // Gemma sliding layers 256, full layers 512.
    std::vector<int>   layer_head_dim;

    // Per-layer K/V head count (size n_layers; empty => uniform n_kv_heads).
    // gemma4_unified full-attention layers use a smaller K/V head count
    // (e.g. 1) than sliding layers (e.g. 8); the Contiguous K/V buffer is
    // sized per layer accordingly. e4b/Llama/Qwen leave this empty.
    std::vector<int>   layer_n_kv_heads;

    // Per-layer KV source (size n_layers; -1 = owns its K/V, else the
    // index of an earlier same-kind layer whose K/V this layer reuses).
    // Shared layers allocate/compute no K/V (Gemma cross-layer sharing).
    std::vector<int>   kv_source;

    // Trailing sliding-window size for Sliding layers (tokens).
    int                sliding_window = 0;

    // Ring-buffer chunk headroom B (tokens). When > 0 (and a layer is
    // Sliding with sliding_window > 0), that layer's K/V is BOUNDED to a
    // ring of capacity min(max_seq, sliding_window + B) instead of the
    // full max_seq -- so long contexts don't pay max_seq for every
    // sliding layer. B must be >= the largest token batch the exec writes
    // in one kv_write dispatch (the prefill chunk size), so a wrap never
    // clobbers a still-in-window key. 0 = no ring (full max_seq, Phase-1
    // window-mask behaviour).
    int                sliding_chunk = 0;

    // ---- Hybrid attention (Qwen3.5 Gated DeltaNet) -----------------
    // Per-layer kind. When empty, every layer is treated as a
    // classical full-attention layer (current Llama behaviour --
    // K/V page pool per layer). When non-empty, size must equal
    // n_layers; `is_linear_layer[L] == true` means layer L runs
    // the linear-attention (SSM) branch, in which case its K/V
    // pool entry is NOT allocated and the context-local
    // conv_state / ssm_state slots are used instead.
    std::vector<bool>  is_linear_layer;

    // SSM (Gated DeltaNet) sizing. Required when is_linear_layer
    // has any true entry; ignored otherwise. conv_dim = 2*key_dim
    // + value_dim, with key_dim = ssm_num_k_heads*ssm_k_head_dim
    // and value_dim = ssm_num_v_heads*ssm_v_head_dim. The
    // conv_state slot has shape
    //   [1, ssm_conv_kernel - 1, ssm_conv_dim]
    // and the ssm_state slot has shape
    //   [1, ssm_num_v_heads, ssm_v_head_dim, ssm_k_head_dim]
    // (mlx-vlm's recurrent-state layout). Both are allocated
    // lazily per context on acquire_root / branch.
    int                ssm_conv_kernel  = 0;
    int                ssm_conv_dim     = 0;
    int                ssm_num_k_heads  = 0;
    int                ssm_num_v_heads  = 0;
    int                ssm_k_head_dim   = 0;
    int                ssm_v_head_dim   = 0;
    // ssm_state dtype is always float32 to match the mlx-lm /
    // mlx-vlm reference recurrence; the conv_state inherits
    // kv_dtype (it carries the QKV-projected pre-conv activations,
    // which live at compute_dtype).
  };

  ContextManager(Spec spec, const SessionContextIntf* session);
  ~ContextManager() override;

  ContextManager(const ContextManager&)            = delete;
  ContextManager& operator=(const ContextManager&) = delete;

  // Mint a fresh root context with seq_len == 0 and no pages.
  ContextId acquire_root();

  // Fork a parent into one child. The child shares every page of
  // the parent by refcount, including any partial tail page (which
  // is marked frozen so it becomes immutable). Both the parent and
  // the child will allocate a fresh private page on their next
  // append; this avoids any K/V copy. The child inherits the
  // parent's seq_len and SSM recurrent state.
  //
  // Returns an invalid ContextId on failure (parent not found).
  ContextId branch(ContextId parent);

  // Fork a parent into N children in one call. Each child is fully
  // independent: writes to one do not affect any other. The parent's
  // partial tail page (if any) is frozen and shared; the parent and
  // every child allocate fresh private pages on their next append.
  // Useful for VQA-style "ask N questions on one image prefix"
  // patterns where we want to share the vision K/V cache across
  // questions.
  //
  // Returns an empty vector if `parent` is invalid or `n_branches`
  // < 1.
  std::vector<ContextId> branch(ContextId parent, int n_branches);

  // ---- Reusable branch pool (pre-allocate once, rebranch per scene) ---
  // A stage that re-branches the same N children off a fresh parent every
  // scene (e.g. realtime-vqa: one image prefix, N question branches) would
  // otherwise pay branch()+release() each scene: per-child SSM/conv buffer
  // allocation + a fresh private page on first append + the teardown. These
  // two calls let it allocate ONCE and reuse the slots forever.
  //
  // reserve_branches mints `n` pooled child contexts: each gets its own
  // SSM/conv + contiguous-KV buffers AND `pages_each` private pages, all
  // held permanently (the pages stay owned across rebranch, so the page
  // pool never churns as long as a branch's appended length stays within
  // `pages_each`). The children start empty (seq_len 0, no shared prefix);
  // call rebranch to point one at a parent. Returns fewer than `n` (or
  // empty) if the page pool can't satisfy the reservation. The caller owns
  // the returned ids and must release() each when done.
  std::vector<ContextId> reserve_branches(int n, int pages_each);

  // Reset a pooled `child` (from reserve_branches) to be a fresh branch of
  // `parent`, REUSING the child's buffers + retained pages (no allocation):
  // releases the child's current non-retained pages, re-shares the parent's
  // pages by refcount, copies the parent's seq_len + SSM/contig state, and
  // makes the retained private pages available again. Equivalent in effect
  // to release(child)+branch(parent) but without freeing/reallocating the
  // child's storage. Returns false if either id is invalid.
  bool rebranch(ContextId child, ContextId parent);

  // Detach a pooled `child` from whatever parent it was last rebranched
  // onto: release its current NON-retained pages (the shared parent
  // pages + any overflow pages) and reset it to the empty
  // reserved-idle state (seq_len 0, retained cursor rewound), KEEPING
  // its retained private pages + SSM/conv/contig buffers for the next
  // rebranch. A streaming caller calls this before building the next
  // scene's base context so the pool stops pinning the PREVIOUS scene's
  // base-context pages -- otherwise that doubles peak page usage and can
  // starve a large multi-frame prefill. No-op on an invalid id.
  void detach_branch(ContextId child);

  // Drop refcounts on all of `id`'s pages (including any retained private
  // pages from reserve_branches), returning any that hit zero to the free
  // list. Idempotent: calling release(id) twice on the same id, or release
  // on an invalid id, is a no-op.
  void release(ContextId id);

  // Bookkeeping handed back from append(). Tells the caller where
  // in the page pool the new tokens live and what their absolute
  // positions in the context's sequence are. The model writes K/V
  // into pool[page_id][slot_offset .. slot_offset + n_tokens, :, :]
  // for each layer.
  //
  // valid() == false when the append failed (capacity exhausted,
  // invalid context, or n_tokens out of range).
  struct AppendSlot {
    PageId page_id     = {};
    int    slot_offset = 0;
    int    position    = 0;   // absolute pos in the context sequence
    int    n_tokens    = 0;

    bool valid() const noexcept { return page_id.valid() && n_tokens > 0; }
  };

  // Append `n_tokens` to `id`. The call writes within ONE page; if
  // the current tail page can hold at most C more tokens (where
  // C = page_tokens when seq_len is page-aligned, else page_tokens
  // - seq_len % page_tokens), the caller must pass n_tokens <= C.
  // Crossing a page boundary is done by chaining two append() calls.
  AppendSlot append(ContextId id, int n_tokens);

  // Total length of `id`'s sequence in tokens. Returns 0 for an
  // invalid or released id.
  int seq_len_of(ContextId id) const;

  // Number of pages currently held by `id`. == ceil(seq_len /
  // page_tokens) for valid contexts, 0 otherwise.
  int page_count_of(ContextId id) const;

  // Copy of `id`'s page list. The engine walks this to gather K/V
  // views for attention.
  std::vector<PageId> pages_of(ContextId id) const;

  // Per-page valid-token counts, parallel to pages_of(). Required
  // because branch() may leave a PARTIAL page in the middle of the
  // list (a frozen tail page whose remaining slots will never be
  // filled), so gather_kv can't infer per-page counts from seq_len
  // alone.
  std::vector<int> pages_valid_of(ContextId id) const;

  // Tokens in the LAST page that are valid for THIS context. Returns
  // 0 if the context has no pages. In the legacy single-branch path
  // this equals seq_len % page_tokens (or page_tokens when aligned);
  // in the branch-frozen path this is simply the n_valid of the
  // current writing page.
  int tail_valid_count(ContextId id) const;

  // Maximum n_tokens the NEXT append on `id` can accept in a single
  // call. Equal to page_tokens when the context is empty, when its
  // tail was just frozen by a branch, or when the last page is full;
  // otherwise page_tokens - page_valid.back(). Callers that prefill
  // long token spans use this to chunk; using seq_len % page_tokens
  // is wrong when an intermediate page is frozen-partial.
  int next_append_capacity(ContextId id) const;

  // Diagnostics: refcount of a specific page (number of contexts
  // that currently include it in their page list). 0 means the
  // page is on the free list. Returns 0 for an out-of-range
  // page id.
  int page_refcount(PageId p) const;

  // Free page count, including never-allocated pages and pages
  // that were freed by release(). For tests + capacity checks.
  std::uint32_t free_page_count() const;

  // ---- metal-compute backend API (Spec::metal != nullptr) ----------
  // These mirror the MLX write_kv/gather_kv surface but hand the exec
  // borrowed metal buffers so it can encode kv_write_paged / sdpa_paged
  // into its own forward command buffer. All return 0/nullptr when the
  // metal backend isn't configured.

  // Tokens per page (clamped, >= 1).
  int page_tokens() const noexcept;
  // Pool cap in pages (derived from max_seq when Spec::max_pages == 0).
  int max_pages() const noexcept;
  // Bytes between consecutive pages in a pool buffer
  // (n_kv_heads * page_tokens * head_dim * sizeof(f16)).
  std::size_t page_stride_bytes() const noexcept;
  // Physical pages currently held by at least one context.
  std::uint32_t pages_in_use() const;
  // True if `layer` is a linear-attention (SSM/GDN) layer.
  bool is_linear_layer(int layer) const noexcept;

  // True when the Contiguous layout routes FULL (global) attention layers
  // through the shared PAGED pool (kpool/vpool/append/fill_page_table) instead
  // of a per-context contiguous buffer -- so branch children share the parent's
  // frozen prefix pages by refcount with no full-KV copy. The exec uses the
  // paged write/attention for full layers and the contiguous ring for sliding
  // layers. False for the Paged layout (Llama/Qwen: every layer paged anyway)
  // and for a Contiguous manager whose full layers couldn't be paged.
  bool full_layers_paged() const noexcept;

  // Borrowed per-layer K/V pool buffers
  // [n_alloc_pages, n_kv_heads, page_tokens, head_dim] f16. nullptr if
  // the layer is out of range or linear (no paged K/V). The pointer is
  // stable only until the next append that grows the pool, so re-fetch
  // each forward pass.
  const metal_compute::SharedBuffer* kpool(int layer) const;
  const metal_compute::SharedBuffer* vpool(int layer) const;

  // Fill `out` with {page_id, n_valid, global_start} per page in `id`'s
  // list (ascending KV-position order); returns the page count. `out`
  // must hold >= page_count_of(id) * 3 int32s.
  int fill_page_table(ContextId id, std::int32_t* out) const;

  // Per-context GDN recurrent-state buffers for a linear layer.
  // conv_state is [(ssm_conv_kernel-1)*ssm_conv_dim] f16; ssm_state is
  // [ssm_num_v_heads*ssm_v_head_dim*ssm_k_head_dim] f32. Zeroed on
  // acquire_root, DEEP-COPIED on branch. The exec reads/updates them in
  // place inside its forward command buffer. nullptr if `layer` isn't
  // linear or `id` is dead.
  const metal_compute::SharedBuffer* conv_state(ContextId id, int layer) const;
  const metal_compute::SharedBuffer* ssm_state(ContextId id, int layer) const;

  // ---- GDN recurrent-state run-ahead ring (pdecode depth>1) -----------
  // A depth>1 pdecode commits a speculative token's forward (advancing the
  // GDN conv/ssm recurrent state IN PLACE) before the host has confirmed the
  // previous token isn't a stop. If it was a stop, that advance must be
  // undone -- but the recurrent state has no "position" to rewind like the
  // paged KV. The ring keeps R = depth+1 physical slots per GDN layer and
  // PING-PONGS: each commit reads the current slot and writes the next, so
  // the previous `depth` states survive in the older slots and rollback is a
  // pure cursor rewind + an O(1) handle swap -- NO per-token copy of the
  // (multi-MB) ssm state. conv1d writes its updated tail to its own out slot
  // (kernel buffer 9); the ssm step kernel already has split read/write
  // bindings (5/7). Logical slot 0 is the canonical buffer (what ssm_state /
  // conv_state return); slots 1..R-1 are shadow buffers.
  //
  // When the ring is OFF (the default, R==1: sync decode, prefill, batched),
  // *_read and *_write both return the canonical slot, so those paths bind
  // read==write and run exactly in place -- byte-identical to before.

  // Buffer to bind as the GDN step / conv1d STATE-IN (read) for `layer`:
  // the current ring slot (canonical when the ring is off). nullptr if not a
  // linear layer / unknown id / not metal-hybrid.
  const metal_compute::SharedBuffer* ssm_read(ContextId id, int layer) const;
  const metal_compute::SharedBuffer* conv_read(ContextId id, int layer) const;
  // Buffer to bind as the STATE-OUT (write): the next ring slot when the ring
  // is on, else the same canonical slot (in-place). Same nullptr conditions.
  const metal_compute::SharedBuffer* ssm_write(ContextId id, int layer) const;
  const metal_compute::SharedBuffer* conv_write(ContextId id, int layer) const;

  // Turn the ring ON for `id` with R = depth+1 slots (depth>=1; depth<=1 is a
  // no-op -- the ring stays off). Allocates the R-1 shadow buffers per GDN
  // layer on first use (kept for the context's lifetime; reused across
  // sessions). The current state stays in logical slot 0 (cursor 0). No-op
  // (returns true) on a non-hybrid / non-metal manager. False on alloc
  // failure or invalid id.
  bool gdn_ring_begin(ContextId id, int depth);
  // Advance the ring cursor one slot (call once per committed forward). The
  // slot the just-encoded commit WROTE becomes the next commit's read slot.
  // No-op when the ring is off.
  void gdn_ring_advance(ContextId id);
  // Rewind the cursor `n` slots to discard `n` speculative advances (n must
  // be <= depth). Returns the new cursor, or -1 on invalid id / n out of the
  // retained history. No-op (returns 0) when the ring is off and n==0.
  int  gdn_ring_rollback(ContextId id, int n);
  // Turn the ring OFF: if the cursor isn't on slot 0, swap each GDN layer's
  // canonical buffer with the cursor slot so the canonical (what ssm_state /
  // conv_state, and hence the next prefill/sync forward, read) holds the
  // final state. Resets R to 1, cursor to 0. Shadow buffers are retained.
  void gdn_ring_end(ContextId id);

  // ---- Contiguous KV layout API (Spec::kv_layout == Contiguous) -----
  // Per-(context, owning layer) contiguous K/V buffers
  // [n_kv_heads, kv_capacity(layer), head_dim(layer)] f16. The exec
  // encodes kv_write_f16 + sdpa_causal_* against these. Shared layers
  // (Spec::kv_source[L] >= 0) return the source layer's buffer; no own
  // storage. nullptr if not the Contiguous layout or the id/layer is
  // invalid.

  // Current logical sequence length of `id` (Contiguous layout).
  int kv_seq_len(ContextId id) const;

  // Reserve `n` tokens at the tail; advances seq_len. Returns the
  // absolute position of the first reserved token, or -1 on overflow /
  // invalid id. (Phase 2 also advances the per-layer ring head.)
  int kv_append(ContextId id, int n);

  // Undo the last `n` kv_append tokens (Contiguous layout): rewinds
  // seq_len by n. The physical ring slots keep stale data but are
  // overwritten by the next append at the same positions, so this is a
  // pure counter rewind -- used to discard a pdecode run-ahead's
  // speculative tail (a stop token's forward KV that the synchronous
  // loop would never have appended). Returns the new seq_len, or -1 on
  // invalid id / non-contiguous layout / n past the start.
  int kv_rollback(ContextId id, int n);

  // Owning-or-source K/V buffer for `layer`. kv_k/kv_v follow
  // Spec::kv_source so a shared layer transparently reads its source.
  const metal_compute::SharedBuffer* kv_k(ContextId id, int layer) const;
  const metal_compute::SharedBuffer* kv_v(ContextId id, int layer) const;

  // Physical token capacity / head_dim / sliding window of `layer`
  // (used as kv_stride / D / window in the SDPA dispatch). 0 when not
  // the Contiguous layout or layer out of range.
  int kv_capacity(int layer) const noexcept;   // tokens per K/V buffer
  int kv_head_dim(int layer) const noexcept;
  int kv_window(int layer) const noexcept;      // 0 = full (no window)
  // Ring capacity for a bounded sliding layer (== kv_capacity when the
  // buffer is a ring smaller than max_seq), else 0. The exec passes this
  // to the SDPA / kv_write kernels: physical slot = logical pos % ring_cap
  // when > 0. 0 means the layer addresses linearly (full or unbounded).
  int kv_ring_cap(int layer) const noexcept;
  // Per-CONTEXT capacity / ring: identical to the per-layer versions unless
  // this context's sliding ring was grown by ensure_sliding_capacity (a
  // one-shot single-pass prefill). The exec MUST use these for the contiguous
  // path so a grown context's bigger sliding stride/ring is honored; other
  // contexts (and global layers) are unaffected.
  int kv_capacity(ContextId id, int layer) const;
  int kv_ring_cap(ContextId id, int layer) const;
  // Grow this context's SLIDING K/V ring to hold `need` tokens (rounded up,
  // capped at max_seq) so a one-shot prefill longer than the default bounded
  // ring runs single-pass (sliding layers stay on the matrix-core SDPA instead
  // of wrapping to the slow scalar path). Reallocates the sliding buffers, so
  // it is ONLY valid on a FRESH context (seq_len==0) -- it discards KV. No-op
  // (returns true) if already large enough; false if not the metal contiguous
  // layout / not fresh / need>max_seq. Global layers are untouched.
  bool ensure_sliding_capacity(ContextId id, int need);


  // Read-only view of the spec the manager was constructed with.
  const Spec& spec() const noexcept;

  // Toggle the per-write_kv eval that materialises K_pool/V_pool
  // immediately. Default true (the round-trip test reads back
  // without any intervening compute and relies on this). The real
  // model flips it off after warmup: in steady state every
  // write_kv is followed by a gather_kv in the same forward pass
  // and the final logits eval forces the whole chain to
  // materialise -- so the per-write sync is redundant overhead
  // that costs 2*n_layers GPU↔CPU syncs per token (~25% of the
  // decode budget on an 8B model).
  void set_eval_per_write(bool b) noexcept;


private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}

#endif
