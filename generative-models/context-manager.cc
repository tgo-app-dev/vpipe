#include "generative-models/context-manager.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"


#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe::genai {

struct ContextManager::Impl {
  Spec spec;
  const SessionContextIntf* session = nullptr;

  // Pre-allocated _pages indexed by PageId.v. _pages[i].refcount is
  // the number of live contexts that include page i in their list;
  // 0 means the page is on the free list.
  struct PageState {
    int refcount = 0;
  };
  vector<PageState> pages;
  vector<PageId>    free_list;   // LIFO: pop_back to allocate

  // Per-context state. Keyed by ContextId.v.
  //
  // page_list and page_valid are parallel arrays. page_list[i] is the
  // page id of the i-th page in this context's sequence; page_valid[i]
  // is the count of valid tokens written into that page from THIS
  // context's perspective. Branching freezes the parent's tail (the
  // last page may be partial), so an inner page can be partial too --
  // gather_kv reads page_valid to slice the right amount per page.
  //
  // tail_frozen == true means the next append on this context must
  // allocate a fresh page even if the current tail page has slack.
  // It's set by branch() on both the parent and every child so the
  // shared tail page is never written to again.
  //
  // ssm_state / conv_state are sized to n_layers; only the slots
  // corresponding to linear-attention layers (Spec::is_linear_layer
  // [L] == true) are ever populated. Full-attention layers keep the
  // optional empty; their state lives in K_pool / V_pool instead.
  struct Context {
    vector<PageId>                          page_list;
    vector<int>                             page_valid;
    int                                     seq_len = 0;
    bool                                    tail_frozen = false;
    // Grown contiguous SLIDING ring capacity for this context (tokens), 0 =
    // use the per-layer default c_cap. Set by ensure_sliding_capacity for a
    // one-shot single-pass prefill that exceeds the default bounded ring; all
    // sliding owning layers share it (same window+chunk). Global/full layers
    // ignore it. Inherited by branches (copy_contig_kv_locked_).
    int                                     sliding_cap = 0;
    // Reusable branch pool (reserve_branches): private pages owned
    // permanently by this context (refcount 1, never auto-freed until
    // release()). append() draws from here -- via retained_cursor -- before
    // the global free list, and rebranch() rewinds the cursor to 0 so the
    // same physical pages are reused every scene. Empty for ordinary
    // contexts (then append() falls through to the free list unchanged).
    vector<PageId>                          retained;
    size_t                                  retained_cursor = 0;
    // metal backend SSM state, sized to n_layers, only linear layers
    // populated (the rest stay empty SharedBuffers). Deep-copied on
    // branch -- the kernel mutates these in place per context.
    //
    // m_conv_state / m_ssm_state are logical slot 0 of the GDN run-ahead ring
    // (the canonical state the ssm_state/conv_state accessors return). The
    // shadows below are slots 1..R-1; gdn_cur is the current logical slot and
    // gdn_ring_R is R (1 == ring off). See ContextManager::gdn_ring_*.
    vector<metal_compute::SharedBuffer>     m_conv_state;
    vector<metal_compute::SharedBuffer>     m_ssm_state;
    // [n_layers][R-1] shadow ring slots (only linear layers populated).
    // Allocated lazily by gdn_ring_begin; retained across pdecode sessions.
    vector<vector<metal_compute::SharedBuffer>> m_conv_shadow;
    vector<vector<metal_compute::SharedBuffer>> m_ssm_shadow;
    int                                     gdn_ring_R = 1;   // R (1 = off)
    int                                     gdn_cur    = 0;   // logical slot
    // Contiguous KV layout (Spec::kv_layout == Contiguous): per-layer
    // K/V, only OWNING layers (kv_source < 0) populated; shared layers
    // read their source's slot. Deep-copied on branch.
    //   metal: [n_kv_heads, cap(L), head_dim(L)] f16 SharedBuffer, written
    //          in place by kv_write / read by sdpa (mutable, pre-alloc'd).
    //   MLX:   [1, n_kv_heads, T, head_dim(L)] arrays grown by the exec
    //          via concatenate (functional); set/get_contig_kv store them.
    vector<metal_compute::SharedBuffer>     c_k;
    vector<metal_compute::SharedBuffer>     c_v;
  };
  unordered_map<uint32_t, Context> contexts;
  uint32_t                         next_ctx_id = 1;

  mutable mutex                    mu;

  // Per-layer K/V page pools. Each populated entry has shape
  // [pool_capacity, n_kv_heads, page_tokens, head_dim] in
  // Spec::kv_dtype, where pool_capacity grows monotonically (one
  // page at a time) from 1 up to spec.max_pages as pages are
  // claimed by allocate_page_locked_. Linear-attention layers keep
  // the optional empty -- write_kv / gather_kv on those layers logs
  // and returns. Functional updates via slice_update replace the
  // array; MLX's compute graph fuses repeated updates.
  bool                                pool_enabled   = false;
  bool                                eval_per_write = true;
  // Current leading-dim of every populated K_pool[L] / V_pool[L]
  // tensor. Starts at 0; bumped by ensure_pool_capacity_locked_ when
  // a fresh page id needs storage. Always <= spec.max_pages.
  uint32_t                            pool_capacity  = 0;
  // True when at least one Spec::is_linear_layer[L] is true; lets
  // acquire_root / branch / release skip the SSM-allocation paths
  // entirely on Llama-style models.
  bool                                has_ssm_layers = false;

  // ---- metal-compute backend (spec.metal != nullptr) ---------------
  // Per-layer growable K/V pools, each
  // [m_pool_capacity, n_kv_heads, page_tokens, head_dim] f16. Linear
  // layers keep their slot empty (no paged K/V). The pools grow by
  // doubling as pages are claimed, mirroring the MLX K_pool path.
  metal_compute::MetalCompute*        mc             = nullptr;
  vector<metal_compute::SharedBuffer> m_kpool;       // [n_layers]
  vector<metal_compute::SharedBuffer> m_vpool;
  uint32_t                            m_pool_capacity = 0;
  size_t                              m_page_bytes   = 0;  // bytes/page
  size_t                              m_conv_bytes   = 0;  // f16
  size_t                              m_ssm_bytes    = 0;  // f32

  // ---- Contiguous KV layout (spec.kv_layout == Contiguous) ---------
  // Per-layer derived tables (computed once in the ctor). c_src[L] = -1
  // for an owning layer, else the source layer index; c_cap[L] = the
  // physical token capacity of the K/V buffer (max_seq for Full; Phase-2
  // window+B for Sliding); c_hd[L] = per-layer head_dim; c_win[L] = the
  // trailing window (0 = full). Owning layers (c_src<0, non-Linear) get a
  // [n_kv_heads, c_cap, c_hd] f16 K and V buffer per context.
  bool                                contig_enabled = false;
  vector<LayerKind>                   c_kind;   // [n_layers]
  vector<int>                         c_src;    // [n_layers]
  vector<int>                         c_cap;    // [n_layers] tokens
  vector<int>                         c_hd;     // [n_layers] head_dim
  vector<int>                         c_nkv;    // [n_layers] n_kv_heads
  vector<int>                         c_win;    // [n_layers] window

  // Contiguous layout, FULL (global) layers -> the shared PAGED pool instead
  // of a per-context contiguous buffer, so branch children share the parent's
  // frozen prefix pages by refcount (no per-branch full-KV copy). On only when
  // every full OWNING layer shares uniform (n_kv_heads, head_dim) -- the pool
  // is one tensor stride (full_nkv, full_hd). Sliding layers keep the bounded
  // contiguous ring (copied on branch, but small + non-growing). Gemma e4b:
  // full nkv=2/D=512, sliding nkv=2/D=256.
  bool                                full_paged = false;
  int                                 full_nkv   = 0;
  int                                 full_hd    = 0;

  bool metal_enabled() const noexcept { return mc != nullptr; }

  // True for a layer that gets its own paged-pool K/V tensor (m_kpool[L]):
  // every non-linear layer in the Paged layout, or a full OWNING layer in the
  // Contiguous-full-paged layout. Shared / sliding / linear layers get none.
  bool
  pool_owns_(int layer) const noexcept
  {
    if (layer < 0 || layer >= spec.n_layers) {
      return false;
    }
    if (contig_enabled) {
      return full_paged && c_src[(size_t)layer] < 0
             && c_kind[(size_t)layer] == LayerKind::Full;
    }
    return !is_linear_(layer);
  }

  // True for a layer that owns a CONTIGUOUS K/V buffer (not shared, not linear,
  // and -- when full layers are paged -- not a full layer). Shared/linear/paged
  // -full layers allocate none.
  bool
  contig_owns_(int layer) const noexcept
  {
    if (!(contig_enabled && layer >= 0 && layer < spec.n_layers
          && c_src[(size_t)layer] < 0
          && c_kind[(size_t)layer] != LayerKind::Linear)) {
      return false;
    }
    if (full_paged && c_kind[(size_t)layer] == LayerKind::Full) {
      return false;   // full owning layer is paged, not contiguous
    }
    return true;
  }

  // Effective token capacity of `layer` for context `c`: the grown sliding
  // ring (c.sliding_cap) when set and the layer is Sliding, else the per-layer
  // default c_cap. Global/full layers always use c_cap.
  int
  ctx_cap_(const Context& c, int layer) const noexcept
  {
    if (c.sliding_cap > 0 && c_kind[(size_t)layer] == LayerKind::Sliding) {
      return c.sliding_cap;
    }
    return c_cap[(size_t)layer];
  }

  // Mirror-tail slots appended to a BOUNDED sliding ring so the decode
  // attention can read its trailing window as ONE contiguous physical span
  // (no `% ring_cap` wrap). The ring modulo stays ctx_cap_; the physical head
  // stride is ctx_cap_ + (window-1). Every ring write that lands in the first
  // (window-1) slots is mirrored into [ring_cap, ring_cap+window-1) so the tail
  // is a live duplicate of the ring head; a window-length scan that crosses the
  // ring end then reads straight into the mirror instead of wrapping. The tail
  // is allocated UNCONDITIONALLY (harmless to the modulo read path -- it just
  // never reads the extra slots); only the flag-gated linear READ uses it.
  // 0 for Full/Linear layers and for a sliding ring grown to max_seq (then
  // ring_cap == 0, addressing is already linear -- no mirror needed).
  int
  mirror_tail_(const Context& c, int layer) const noexcept
  {
    if (c_kind[(size_t)layer] != LayerKind::Sliding) {
      return 0;
    }
    const int win = !c_win.empty() ? c_win[(size_t)layer] : 0;
    const int cap = ctx_cap_(c, layer);
    if (win <= 1 || cap >= spec.max_seq) {
      return 0;                              // unbounded ring -> linear already
    }
    return win - 1;
  }

  // Physical head stride (token slots per K/V head) for context `c`: the ring
  // modulo plus the mirror tail. This is the stride the kernels address with;
  // the ring modulo (ctx_cap_) is passed separately to the write/read kernels.
  int
  ctx_stride_(const Context& c, int layer) const noexcept
  {
    return ctx_cap_(c, layer) + mirror_tail_(c, layer);
  }

  // Bytes of one contiguous K (or V) buffer for an owning layer of context `c`
  // (honors a grown sliding ring + the mirror tail).
  size_t
  contig_bytes_(const Context& c, int layer) const noexcept
  {
    const int nkv = !c_nkv.empty() ? c_nkv[(size_t)layer] : spec.n_kv_heads;
    return (size_t)nkv * (size_t)ctx_stride_(c, layer)
           * (size_t)c_hd[(size_t)layer] * 2;
  }

  // Prepare a fresh context's contiguous K/V slots (owning layers only).
  // metal: allocate+zero the mutable [n_kv, cap, hd] buffers. MLX: size
  // the per-layer optional<array> vectors (the arrays are filled lazily by
  // the exec via set_contig_kv). Returns false on metal alloc failure.
  bool
  init_contig_kv_locked_(Context* c)
  {
    if (!contig_enabled) {
      return true;
    }
    if (metal_enabled()) {
      c->c_k.resize((size_t)spec.n_layers);
      c->c_v.resize((size_t)spec.n_layers);
      for (int L = 0; L < spec.n_layers; ++L) {
        if (!contig_owns_(L)) {
          continue;
        }
        const size_t n = contig_bytes_(*c, L);
        c->c_k[(size_t)L] = mc->make_shared_buffer(n);
        c->c_v[(size_t)L] = mc->make_shared_buffer(n);
        if (c->c_k[(size_t)L].empty() || c->c_v[(size_t)L].empty()) {
          return false;
        }
        std::memset(c->c_k[(size_t)L].contents(), 0, n);
        std::memset(c->c_v[(size_t)L].contents(), 0, n);
      }
    }
    return true;
  }

  // Deep-copy parent's contiguous K/V (the whole prefix) into `child`.
  // metal: memcpy into child's own buffers (already alloc'd by
  // init_contig_kv_locked_). MLX: copy the array handles (functional --
  // immutable, the child's next concat replaces its own slot).
  void
  copy_contig_kv_locked_(const Context& parent, Context* child)
  {
    if (!contig_enabled) {
      return;
    }
    if (metal_enabled()) {
      // Inherit the parent's (possibly grown) sliding ring: if the child's
      // buffers were sized to a different cap (default, or a prior reuse),
      // reallocate the sliding owning layers to match before copying. Common
      // case (both default, sliding_cap==0) skips this -- no realloc.
      if (child->sliding_cap != parent.sliding_cap) {
        child->sliding_cap = parent.sliding_cap;
        for (int L = 0; L < spec.n_layers; ++L) {
          if (!contig_owns_(L) || c_kind[(size_t)L] != LayerKind::Sliding) {
            continue;
          }
          const size_t n = contig_bytes_(parent, L);
          child->c_k[(size_t)L] = mc->make_shared_buffer(n);
          child->c_v[(size_t)L] = mc->make_shared_buffer(n);
        }
      }
      for (int L = 0; L < spec.n_layers; ++L) {
        if (!contig_owns_(L)) {
          continue;
        }
        if ((size_t)L < parent.c_k.size() && !parent.c_k[(size_t)L].empty()
            && !child->c_k[(size_t)L].empty()) {
          const size_t n = contig_bytes_(parent, L);
          std::memcpy(child->c_k[(size_t)L].contents(),
                      parent.c_k[(size_t)L].contents(), n);
          std::memcpy(child->c_v[(size_t)L].contents(),
                      parent.c_v[(size_t)L].contents(), n);
        }
      }
    }
  }

  // Whether layer L runs the SSM branch. Wraps the bool-vector
  // lookup so non-hybrid configs (empty is_linear_layer) cleanly
  // answer false for every layer.
  bool
  is_linear_(int layer) const noexcept
  {
    if (layer < 0 || layer >= spec.n_layers) {
      return false;
    }
    if (static_cast<size_t>(layer) >= spec.is_linear_layer.size()) {
      return false;
    }
    return spec.is_linear_layer[layer];
  }


  // ---- metal-compute backend helpers (caller holds `mu`) -----------

  // Grow the per-FULL-ATTENTION-layer metal pools so they hold at least
  // `need` pages, by doubling + copying existing page data forward.
  // Linear layers carry no paged K/V. Returns false on allocation
  // failure or hard-cap overflow. No-op when the metal pool is off.
  bool
  ensure_metal_pool_capacity_locked_(uint32_t need)
  {
    if (!metal_enabled() || m_kpool.empty()) {
      return true;
    }
    if (need <= m_pool_capacity) {
      return true;
    }
    uint32_t new_cap = m_pool_capacity > 0 ? m_pool_capacity * 2 : 1;
    if (new_cap < need) {
      new_cap = need;
    }
    if (new_cap > spec.max_pages) {
      new_cap = spec.max_pages;
    }
    if (new_cap < need) {
      return false;   // would exceed the hard cap
    }
    const size_t old_bytes =
        static_cast<size_t>(m_pool_capacity) * m_page_bytes;
    const size_t new_bytes = static_cast<size_t>(new_cap) * m_page_bytes;
    for (int L = 0; L < spec.n_layers; ++L) {
      if (!pool_owns_(L)) {
        continue;   // linear / sliding / shared layer: no paged K/V here
      }
      metal_compute::SharedBuffer nk = mc->make_shared_buffer(new_bytes);
      metal_compute::SharedBuffer nv = mc->make_shared_buffer(new_bytes);
      if (nk.empty() || nv.empty()) {
        return false;
      }
      if (old_bytes > 0) {
        std::memcpy(nk.contents(), m_kpool[L].contents(), old_bytes);
        std::memcpy(nv.contents(), m_vpool[L].contents(), old_bytes);
      }
      m_kpool[L] = std::move(nk);
      m_vpool[L] = std::move(nv);
    }
    m_pool_capacity = new_cap;
    return true;
  }

  // Allocate + zero a context's metal SSM state (no-op when not hybrid
  // or not metal). conv_state is sized to n_layers; only linear layers
  // get a buffer, the rest stay empty.
  bool
  init_metal_state_locked_(Context* c)
  {
    if (!metal_enabled() || !has_ssm_layers) {
      return true;
    }
    c->m_conv_state.resize(static_cast<size_t>(spec.n_layers));
    c->m_ssm_state.resize(static_cast<size_t>(spec.n_layers));
    for (int L = 0; L < spec.n_layers; ++L) {
      if (!is_linear_(L)) {
        continue;
      }
      c->m_conv_state[L] = mc->make_shared_buffer(m_conv_bytes);
      c->m_ssm_state[L]  = mc->make_shared_buffer(m_ssm_bytes);
      if (c->m_conv_state[L].empty() || c->m_ssm_state[L].empty()) {
        return false;
      }
      std::memset(c->m_conv_state[L].contents(), 0, m_conv_bytes);
      std::memset(c->m_ssm_state[L].contents(), 0, m_ssm_bytes);
    }
    return true;
  }

  // Deep-copy parent's metal SSM state into `child` (which must already
  // have its own freshly-allocated buffers from init_metal_state_locked_
  // -- the running state diverges, unlike the refcount-shared KV pages).
  void
  copy_metal_state_locked_(const Context& parent, Context* child)
  {
    if (!metal_enabled() || !has_ssm_layers) {
      return;
    }
    for (int L = 0; L < spec.n_layers; ++L) {
      if (!is_linear_(L)) {
        continue;
      }
      if (static_cast<size_t>(L) < parent.m_conv_state.size()
          && !parent.m_conv_state[L].empty()
          && !child->m_conv_state[L].empty()) {
        std::memcpy(child->m_conv_state[L].contents(),
                    parent.m_conv_state[L].contents(), m_conv_bytes);
        std::memcpy(child->m_ssm_state[L].contents(),
                    parent.m_ssm_state[L].contents(), m_ssm_bytes);
      }
    }
  }

  PageId
  allocate_page_locked_()
  {
    if (free_list.empty()) {
      return PageId{};   // invalid
    }
    PageId p = free_list.back();
    free_list.pop_back();
    pages[p.v].refcount = 1;
    // Metal pool grows to cover this page id (doubling). On failure the
    // page is still claimed; kpool()/vpool() return a buffer that simply
    // doesn't cover it, which the exec treats as a soft error.
    if (metal_enabled()) {
      ensure_metal_pool_capacity_locked_(static_cast<uint32_t>(p.v) + 1);
    }
    return p;
  }

  void
  release_page_locked_(PageId p)
  {
    auto& ps = pages[p.v];
    if (ps.refcount <= 0) {
      return;
    }
    --ps.refcount;
    if (ps.refcount == 0) {
      free_list.push_back(p);
    }
  }

  Context*
  find_locked_(ContextId id)
  {
    if (!id.valid()) {
      return nullptr;
    }
    auto it = contexts.find(id.v);
    return it == contexts.end() ? nullptr : &it->second;
  }

  const Context*
  find_locked_(ContextId id) const
  {
    if (!id.valid()) {
      return nullptr;
    }
    auto it = contexts.find(id.v);
    return it == contexts.end() ? nullptr : &it->second;
  }

  // Physical buffer for logical GDN ring slot `s` (0 == canonical, 1..R-1 ==
  // shadow). Falls back to the canonical slot if the shadow for `s` isn't
  // allocated (ring off) so callers never get a null mid-ring.
  const metal_compute::SharedBuffer*
  gdn_ssm_slot_(const Context* c, int L, int s) const
  {
    const size_t l = (size_t)L;
    if (s <= 0 || l >= c->m_ssm_shadow.size()
        || (size_t)(s - 1) >= c->m_ssm_shadow[l].size()) {
      return &c->m_ssm_state[l];
    }
    return &c->m_ssm_shadow[l][(size_t)(s - 1)];
  }
  const metal_compute::SharedBuffer*
  gdn_conv_slot_(const Context* c, int L, int s) const
  {
    const size_t l = (size_t)L;
    if (s <= 0 || l >= c->m_conv_shadow.size()
        || (size_t)(s - 1) >= c->m_conv_shadow[l].size()) {
      return &c->m_conv_state[l];
    }
    return &c->m_conv_shadow[l][(size_t)(s - 1)];
  }
};

ContextManager::ContextManager(Spec spec, const SessionContextIntf* s)
  : SessionMember(s)
  , _impl(make_unique<Impl>())
{
  _impl->spec    = spec;
  _impl->session = s;
  _impl->mc      = spec.metal;

  // Clamp page_tokens to a sane minimum so the page-arithmetic
  // (modulo, ceil) doesn't degenerate. max_pages is taken at face
  // value; out-of-capacity is a recoverable failure (append returns
  // an invalid slot).
  if (_impl->spec.page_tokens < 1) {
    _impl->spec.page_tokens = 1;
  }
  // Metal paged backend: derive a pool cap from max_seq when unset
  // (enough pages for several full-length contexts; branch fanout shares
  // the prefix). Pools still grow lazily up to this cap. The Contiguous
  // layout has no page pool, so skip (its free_list stays trivial).
  if (_impl->metal_enabled()
      && _impl->spec.kv_layout == KvLayout::Paged
      && _impl->spec.max_pages <= 0) {
    const int per_ctx = (_impl->spec.max_seq + _impl->spec.page_tokens - 1)
                        / _impl->spec.page_tokens;
    _impl->spec.max_pages = (per_ctx > 0 ? per_ctx : 1) * 16;
  }
  if (_impl->spec.max_pages < 1) {
    _impl->spec.max_pages = 1;
  }

  _impl->pages.assign(_impl->spec.max_pages, Impl::PageState{0});
  _impl->free_list.reserve(_impl->spec.max_pages);
  // Push descending so pop_back yields ascending ids -- nicer for
  // diagnostics and tests that compare against expected page ids.
  for (uint32_t i = _impl->spec.max_pages; i-- > 0;) {
    _impl->free_list.push_back(PageId{i});
  }

  // Detect hybrid (Qwen3.5-style) configs before we allocate any
  // pool. The is_linear_layer vector being non-empty AND containing
  // at least one `true` entry signals that some layers should NOT
  // get a K/V pool. We sanity-check the size up front: if the spec
  // got it wrong, fall back to treating every layer as full-attn
  // rather than silently misroute writes.
  if (static_cast<int>(_impl->spec.is_linear_layer.size())
        == _impl->spec.n_layers) {
    for (bool b : _impl->spec.is_linear_layer) {
      if (b) {
        _impl->has_ssm_layers = true;
        break;
      }
    }
  } else if (!_impl->spec.is_linear_layer.empty()) {
    if (s) {
      s->warn(fmt(
          "ContextManager: is_linear_layer size {} != n_layers {}; "
          "treating all layers as full-attention",
          _impl->spec.is_linear_layer.size(), _impl->spec.n_layers));
    }
    _impl->spec.is_linear_layer.clear();
  }

  // Allocate the per-layer K/V pools lazily as pages are claimed.
  // Pool layout: [pool_capacity, n_kv_heads, page_tokens, head_dim],
  // with pool_capacity growing from 1 up to spec.max_pages one page
  // at a time. The inner [n_kv_heads, page_tokens, head_dim] block
  // is the per-page K/V tile in mlx-lm's KVCache layout; a
  // single-page gather yields a slice [1, n_kv_heads, page_tokens,
  // head_dim], which IS the [B, H, T, D] shape
  // fast::scaled_dot_product_attention wants -- no reshape+transpose
  // needed at the call site. Linear-attention layers skip the pool
  // entirely: their state lives in the per-context SSM slots.

  // Metal-compute PAGED backend pools + SSM sizing. f16 K/V (2 bytes/
  // elem, covering both f16 and bf16 -- the kernel interprets the
  // bytes), f16 conv_state, f32 ssm_state.
  if (_impl->metal_enabled()
      && _impl->spec.kv_layout == KvLayout::Paged
      && _impl->spec.n_layers > 0
      && _impl->spec.n_kv_heads > 0
      && _impl->spec.head_dim > 0) {
    _impl->m_page_bytes =
        static_cast<size_t>(_impl->spec.n_kv_heads)
        * _impl->spec.page_tokens * _impl->spec.head_dim * 2;
    _impl->m_conv_bytes =
        static_cast<size_t>(_impl->spec.ssm_conv_kernel > 1
                                ? _impl->spec.ssm_conv_kernel - 1 : 0)
        * static_cast<size_t>(_impl->spec.ssm_conv_dim) * 2;
    _impl->m_ssm_bytes =
        static_cast<size_t>(_impl->spec.ssm_num_v_heads)
        * _impl->spec.ssm_v_head_dim * _impl->spec.ssm_k_head_dim * 4;
    _impl->m_kpool.resize(_impl->spec.n_layers);
    _impl->m_vpool.resize(_impl->spec.n_layers);
    // Seed one page so the first forward's command buffer doesn't fold
    // a fresh alloc in with the weight reads.
    _impl->ensure_metal_pool_capacity_locked_(1);
  }

  // CONTIGUOUS layout (Gemma, both backends): build the per-layer derived
  // tables. Storage is per-context (init_contig_kv_locked_ on acquire_root
  // / branch) -- metal allocs mutable buffers, MLX sizes the array slots.
  // cap(L) is max_seq for full/MLX layers; the metal Sliding ring (Phase 2)
  // bounds it to window + chunk headroom.
  if (_impl->spec.kv_layout == KvLayout::Contiguous
      && _impl->spec.n_layers > 0
      && _impl->spec.n_kv_heads > 0) {
    const int NL = _impl->spec.n_layers;
    _impl->c_kind.assign((size_t)NL, LayerKind::Full);
    _impl->c_src.assign((size_t)NL, -1);
    _impl->c_cap.assign((size_t)NL, 0);
    _impl->c_hd.assign((size_t)NL, 0);
    _impl->c_nkv.assign((size_t)NL, _impl->spec.n_kv_heads);
    _impl->c_win.assign((size_t)NL, 0);
    const bool have_kind =
        (int)_impl->spec.layer_kind.size() == NL;
    const bool have_hd =
        (int)_impl->spec.layer_head_dim.size() == NL;
    const bool have_nkv =
        (int)_impl->spec.layer_n_kv_heads.size() == NL;
    const bool have_src =
        (int)_impl->spec.kv_source.size() == NL;
    for (int L = 0; L < NL; ++L) {
      LayerKind k = LayerKind::Full;
      if (have_kind) {
        k = _impl->spec.layer_kind[(size_t)L];
      } else if (_impl->is_linear_(L)) {
        k = LayerKind::Linear;
      }
      _impl->c_kind[(size_t)L] = k;
      _impl->c_src[(size_t)L] = have_src ? _impl->spec.kv_source[(size_t)L]
                                         : -1;
      _impl->c_hd[(size_t)L] = have_hd ? _impl->spec.layer_head_dim[(size_t)L]
                                       : _impl->spec.head_dim;
      _impl->c_nkv[(size_t)L] =
          have_nkv ? _impl->spec.layer_n_kv_heads[(size_t)L]
                   : _impl->spec.n_kv_heads;
      _impl->c_win[(size_t)L] =
          (k == LayerKind::Sliding) ? _impl->spec.sliding_window : 0;
      // Sliding layers with a ring chunk are BOUNDED to window + B (B =
      // sliding_chunk), capped at max_seq; everything else is max_seq.
      int cap = _impl->spec.max_seq;
      if (k == LayerKind::Sliding
          && _impl->spec.sliding_window > 0
          && _impl->spec.sliding_chunk > 0) {
        const int ring = _impl->spec.sliding_window + _impl->spec.sliding_chunk;
        if (ring < cap) {
          cap = ring;
        }
      }
      _impl->c_cap[(size_t)L] = cap;
    }
    _impl->contig_enabled = true;

    // FULL (global) layers -> the shared PAGED pool, so a branch shares the
    // parent's frozen prefix pages by refcount (no per-branch full-KV copy)
    // and grows on demand. Enabled only when every full OWNING layer shares
    // uniform (n_kv_heads, head_dim) -- the pool is a single tensor stride.
    // Sliding layers keep the contiguous ring. (metal backend only.)
    if (_impl->metal_enabled()) {
      int fnkv = -1, fhd = -1;
      bool uniform = true, any = false;
      for (int L = 0; L < NL; ++L) {
        if (_impl->c_kind[(size_t)L] != LayerKind::Full
            || _impl->c_src[(size_t)L] >= 0) {
          continue;                       // not a full OWNING layer
        }
        any = true;
        if (fnkv < 0) {
          fnkv = _impl->c_nkv[(size_t)L];
          fhd  = _impl->c_hd[(size_t)L];
        } else if (_impl->c_nkv[(size_t)L] != fnkv
                   || _impl->c_hd[(size_t)L] != fhd) {
          uniform = false;
        }
      }
      if (any && uniform && fnkv > 0 && fhd > 0) {
        _impl->full_paged  = true;
        _impl->full_nkv    = fnkv;
        _impl->full_hd     = fhd;
        _impl->m_page_bytes =
            static_cast<size_t>(fnkv) * _impl->spec.page_tokens * fhd * 2;
        _impl->m_kpool.resize((size_t)NL);
        _impl->m_vpool.resize((size_t)NL);
        _impl->ensure_metal_pool_capacity_locked_(1);
      }
    }
  }
}

ContextManager::~ContextManager() = default;

const ContextManager::Spec&
ContextManager::spec() const noexcept
{
  return _impl->spec;
}

ContextId
ContextManager::acquire_root()
{
  lock_guard<mutex> lk(_impl->mu);
  ContextId id{_impl->next_ctx_id++};
  auto [it, _ok] = _impl->contexts.emplace(id.v, Impl::Context{});
  if (_impl->metal_enabled()) {
    if (!_impl->init_metal_state_locked_(&it->second)
        || !_impl->init_contig_kv_locked_(&it->second)) {
      _impl->contexts.erase(it);
      if (session()) {
        session()->warn(fmt(
            "ContextManager::acquire_root: metal KV/state alloc failed"));
      }
      return ContextId{};
    }
  }
  (void)it;
  return id;
}

ContextId
ContextManager::branch(ContextId parent_id)
{
  auto v = branch(parent_id, 1);
  if (v.empty()) {
    return ContextId{};
  }
  return v[0];
}

vector<ContextId>
ContextManager::branch(ContextId parent_id, int n_branches)
{
  if (n_branches < 1) {
    return {};
  }
  lock_guard<mutex> lk(_impl->mu);
  auto* parent = _impl->find_locked_(parent_id);
  if (!parent) {
    if (session()) {
      session()->warn(fmt(
          "ContextManager::branch: parent id {} not found",
          parent_id.v));
    }
    return {};
  }

  // Freeze parent's tail: any page already in parent's list (even a
  // partial one) becomes immutable from this point on. Both the
  // parent and every child must allocate a fresh page on their next
  // append. No data copy needed -- the partial tail page is simply
  // shared by refcount; gather_kv consults per-page n_valid so the
  // partial slot count survives the share.
  parent->tail_frozen = true;

  vector<ContextId> out;
  out.reserve(static_cast<size_t>(n_branches));
  for (int b = 0; b < n_branches; ++b) {
    Impl::Context child;
    child.seq_len     = parent->seq_len;
    child.tail_frozen = !parent->page_list.empty();
    child.page_list.reserve(parent->page_list.size());
    child.page_valid.reserve(parent->page_valid.size());

    // Share every page by bumping its refcount. The child's
    // page_valid is a copy of the parent's at branch time.
    for (size_t i = 0; i < parent->page_list.size(); ++i) {
      const PageId p = parent->page_list[i];
      _impl->pages[p.v].refcount += 1;
      child.page_list.push_back(p);
      child.page_valid.push_back(parent->page_valid[i]);
    }

    // Copy SSM recurrent state from parent into child for every
    // linear-attention layer. Independent thereafter: subsequent
    // forward passes update each context's slot in place.
    if (_impl->metal_enabled()) {
      // Allocate the child's OWN buffers (the kernel mutates them in
      // place), then deep-copy the parent's running state into them.
      _impl->init_metal_state_locked_(&child);
      _impl->copy_metal_state_locked_(*parent, &child);
    }
    // Contiguous KV (both backends): deep-copy the whole prefix into the
    // child (metal memcpy into fresh buffers; MLX array-handle copy). Owning
    // layers only; shared layers carry nothing.
    _impl->init_contig_kv_locked_(&child);
    _impl->copy_contig_kv_locked_(*parent, &child);

    ContextId child_id{_impl->next_ctx_id++};
    _impl->contexts.emplace(child_id.v, std::move(child));
    out.push_back(child_id);
  }
  return out;
}

vector<ContextId>
ContextManager::reserve_branches(int n, int pages_each)
{
  if (n < 1) {
    return {};
  }
  if (pages_each < 0) {
    pages_each = 0;
  }
  lock_guard<mutex> lk(_impl->mu);
  vector<ContextId> out;
  out.reserve(static_cast<size_t>(n));
  for (int b = 0; b < n; ++b) {
    Impl::Context child;
    // Per-context SSM/conv + contiguous-KV buffers, allocated ONCE here and
    // reused across every rebranch (mirrors acquire_root's init paths).
    if (_impl->metal_enabled()) {
      if (!_impl->init_metal_state_locked_(&child)
          || !_impl->init_contig_kv_locked_(&child)) {
        if (session()) {
          session()->warn(fmt(
              "ContextManager::reserve_branches: metal KV/state alloc "
              "failed at branch {}", b));
        }
        break;
      }
    }
    // Pre-allocate the branch's private pages; held permanently (refcount 1)
    // and handed out by append() via retained_cursor.
    child.retained.reserve(static_cast<size_t>(pages_each));
    bool ok = true;
    for (int i = 0; i < pages_each; ++i) {
      PageId p = _impl->allocate_page_locked_();
      if (!p.valid()) {
        ok = false;
        break;
      }
      child.retained.push_back(p);
    }
    if (!ok) {
      for (PageId p : child.retained) {
        _impl->release_page_locked_(p);
      }
      if (session()) {
        session()->warn(fmt(
            "ContextManager::reserve_branches: page pool exhausted "
            "reserving {} pages for branch {} (max_pages={})",
            pages_each, b, _impl->spec.max_pages));
      }
      break;
    }
    ContextId child_id{_impl->next_ctx_id++};
    _impl->contexts.emplace(child_id.v, std::move(child));
    out.push_back(child_id);
  }
  return out;
}

bool
ContextManager::rebranch(ContextId child_id, ContextId parent_id)
{
  lock_guard<mutex> lk(_impl->mu);
  auto* parent = _impl->find_locked_(parent_id);
  auto* child  = _impl->find_locked_(child_id);
  if (!parent || !child) {
    if (session()) {
      session()->warn(fmt(
          "ContextManager::rebranch: parent {} or child {} not found",
          parent_id.v, child_id.v));
    }
    return false;
  }

  // Release the child's current NON-retained pages (last scene's shared
  // parent pages + any overflow pages); keep the retained pool owned and
  // rewind its cursor so the same physical pages are reused this scene.
  const auto& retained = child->retained;
  auto is_retained = [&](PageId p) {
    for (PageId r : retained) {
      if (r.v == p.v) { return true; }
    }
    return false;
  };
  for (PageId p : child->page_list) {
    if (!is_retained(p)) {
      _impl->release_page_locked_(p);
    }
  }
  child->page_list.clear();
  child->page_valid.clear();
  child->retained_cursor = 0;

  // Freeze parent's tail and share its pages by refcount (as branch()).
  parent->tail_frozen = true;
  child->seq_len     = parent->seq_len;
  child->tail_frozen = !parent->page_list.empty();
  child->page_list.reserve(parent->page_list.size());
  child->page_valid.reserve(parent->page_valid.size());
  for (size_t i = 0; i < parent->page_list.size(); ++i) {
    const PageId p = parent->page_list[i];
    _impl->pages[p.v].refcount += 1;
    child->page_list.push_back(p);
    child->page_valid.push_back(parent->page_valid[i]);
  }

  // Deep-copy parent SSM/contig state into the child's EXISTING buffers
  // (copy_*_locked_ memcpy in place -- no realloc).
  if (_impl->metal_enabled()) {
    _impl->copy_metal_state_locked_(*parent, child);
  }
  _impl->copy_contig_kv_locked_(*parent, child);
  return true;
}

void
ContextManager::detach_branch(ContextId child_id)
{
  lock_guard<mutex> lk(_impl->mu);
  auto* child = _impl->find_locked_(child_id);
  if (!child) { return; }
  // Release the child's non-retained pages (last scene's shared parent
  // pages + any overflow); keep the retained pool owned + the buffers.
  // Mirrors rebranch()'s reset half, minus the re-share. Once the shared
  // pages drop their refcount here, the parent base context's own
  // destruction frees them back to the pool.
  const auto& retained = child->retained;
  auto is_retained = [&](PageId p) {
    for (PageId r : retained) {
      if (r.v == p.v) { return true; }
    }
    return false;
  };
  for (PageId p : child->page_list) {
    if (!is_retained(p)) {
      _impl->release_page_locked_(p);
    }
  }
  child->page_list.clear();
  child->page_valid.clear();
  child->retained_cursor = 0;
  child->seq_len         = 0;
  child->tail_frozen     = false;
}

void
ContextManager::release(ContextId id)
{
  lock_guard<mutex> lk(_impl->mu);
  auto it = _impl->contexts.find(id.v);
  if (it == _impl->contexts.end()) {
    return;
  }
  const auto& retained = it->second.retained;
  auto is_retained = [&](PageId p) {
    for (PageId r : retained) {
      if (r.v == p.v) { return true; }
    }
    return false;
  };
  // Release pages once each. A retained page may also sit in page_list (it
  // was appended this scene) -- skip it here and free it via the retained
  // loop so its refcount isn't dropped twice.
  for (PageId p : it->second.page_list) {
    if (!is_retained(p)) {
      _impl->release_page_locked_(p);
    }
  }
  for (PageId p : retained) {
    _impl->release_page_locked_(p);
  }
  _impl->contexts.erase(it);
}

ContextManager::AppendSlot
ContextManager::append(ContextId id, int n_tokens)
{
  lock_guard<mutex> lk(_impl->mu);
  auto* ctx = _impl->find_locked_(id);
  if (!ctx) {
    if (session()) {
      session()->warn(fmt(
          "ContextManager::append: context id {} not found", id.v));
    }
    return AppendSlot{};
  }
  if (n_tokens <= 0) {
    return AppendSlot{};
  }
  const int page_tokens = _impl->spec.page_tokens;

  // Decide whether to write into the existing tail page or allocate
  // a fresh one. We need a fresh page when:
  //   * there is no tail page yet (empty context), OR
  //   * the tail page is full (page_valid.back() == page_tokens), OR
  //   * tail_frozen was set by a prior branch() so the shared tail
  //     page must not be mutated.
  const bool need_fresh =
      ctx->page_list.empty()
      || ctx->tail_frozen
      || ctx->page_valid.back() >= page_tokens;

  int cap;
  if (need_fresh) {
    cap = page_tokens;
  } else {
    cap = page_tokens - ctx->page_valid.back();
  }
  if (n_tokens > cap) {
    if (session()) {
      session()->warn(fmt(
          "ContextManager::append: n_tokens={} exceeds tail capacity "
          "{} (seq_len={}, page_tokens={}); caller must chunk",
          n_tokens, cap, ctx->seq_len, page_tokens));
    }
    return AppendSlot{};
  }

  if (need_fresh) {
    // Prefer this context's pre-reserved private pages (reserve_branches):
    // they're already allocated + owned, so reusing them avoids touching the
    // global free list. Only once the reservation is exhausted do we fall
    // back to a fresh allocation (the "request exceeded the pre-allocation"
    // case). Ordinary contexts have no retained pages and skip straight to
    // the free list.
    PageId fresh{};
    if (ctx->retained_cursor < ctx->retained.size()) {
      fresh = ctx->retained[ctx->retained_cursor++];
    } else {
      fresh = _impl->allocate_page_locked_();
    }
    if (!fresh.valid()) {
      if (session()) {
        session()->warn(fmt(
            "ContextManager::append: page pool exhausted "
            "(max_pages={})", _impl->spec.max_pages));
      }
      return AppendSlot{};
    }
    ctx->page_list.push_back(fresh);
    ctx->page_valid.push_back(0);
    // The fresh page belongs to THIS context; subsequent appends on
    // this context can extend it until it fills.
    ctx->tail_frozen = false;
  }

  AppendSlot slot;
  slot.page_id     = ctx->page_list.back();
  slot.slot_offset = ctx->page_valid.back();
  slot.position    = ctx->seq_len;
  slot.n_tokens    = n_tokens;

  ctx->page_valid.back() += n_tokens;
  ctx->seq_len += n_tokens;
  return slot;
}

int
ContextManager::seq_len_of(ContextId id) const
{
  lock_guard<mutex> lk(_impl->mu);
  const auto* ctx = _impl->find_locked_(id);
  return ctx ? ctx->seq_len : 0;
}

int
ContextManager::page_count_of(ContextId id) const
{
  lock_guard<mutex> lk(_impl->mu);
  const auto* ctx = _impl->find_locked_(id);
  return ctx ? static_cast<int>(ctx->page_list.size()) : 0;
}

vector<PageId>
ContextManager::pages_of(ContextId id) const
{
  lock_guard<mutex> lk(_impl->mu);
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx) {
    return {};
  }
  return ctx->page_list;
}

int
ContextManager::tail_valid_count(ContextId id) const
{
  lock_guard<mutex> lk(_impl->mu);
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx || ctx->page_valid.empty()) {
    return 0;
  }
  return ctx->page_valid.back();
}

vector<int>
ContextManager::pages_valid_of(ContextId id) const
{
  lock_guard<mutex> lk(_impl->mu);
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx) {
    return {};
  }
  return ctx->page_valid;
}

int
ContextManager::next_append_capacity(ContextId id) const
{
  lock_guard<mutex> lk(_impl->mu);
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx) {
    return 0;
  }
  const int page_tokens = _impl->spec.page_tokens;
  if (ctx->page_list.empty()
      || ctx->tail_frozen
      || ctx->page_valid.back() >= page_tokens) {
    return page_tokens;
  }
  return page_tokens - ctx->page_valid.back();
}

int
ContextManager::page_refcount(PageId p) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!p.valid() || p.v >= _impl->pages.size()) {
    return 0;
  }
  return _impl->pages[p.v].refcount;
}

uint32_t
ContextManager::free_page_count() const
{
  lock_guard<mutex> lk(_impl->mu);
  return static_cast<uint32_t>(_impl->free_list.size());
}

// ---------------------------------------------------------------------
// metal-compute backend API (Spec::metal != nullptr)
// ---------------------------------------------------------------------

int
ContextManager::page_tokens() const noexcept
{
  return _impl->spec.page_tokens;
}

int
ContextManager::max_pages() const noexcept
{
  return _impl->spec.max_pages;
}

size_t
ContextManager::page_stride_bytes() const noexcept
{
  return _impl->m_page_bytes;
}

uint32_t
ContextManager::pages_in_use() const
{
  lock_guard<mutex> lk(_impl->mu);
  return _impl->spec.max_pages
         - static_cast<uint32_t>(_impl->free_list.size());
}

bool
ContextManager::is_linear_layer(int layer) const noexcept
{
  return _impl->is_linear_(layer);
}

bool
ContextManager::full_layers_paged() const noexcept
{
  return _impl->full_paged;
}

const metal_compute::SharedBuffer*
ContextManager::kpool(int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (layer < 0 || layer >= static_cast<int>(_impl->m_kpool.size())) {
    return nullptr;
  }
  // Contiguous-full-paged: a shared full layer reads its source's pool. (A
  // non-pool layer's slot is an empty SharedBuffer -- the legacy contract;
  // callers route by layer kind and never attend a non-pool layer's pool.)
  const int own =
      (_impl->contig_enabled && _impl->c_src[(size_t)layer] >= 0)
          ? _impl->c_src[(size_t)layer] : layer;
  return &_impl->m_kpool[static_cast<size_t>(own)];
}

const metal_compute::SharedBuffer*
ContextManager::vpool(int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (layer < 0 || layer >= static_cast<int>(_impl->m_vpool.size())) {
    return nullptr;
  }
  const int own =
      (_impl->contig_enabled && _impl->c_src[(size_t)layer] >= 0)
          ? _impl->c_src[(size_t)layer] : layer;
  return &_impl->m_vpool[static_cast<size_t>(own)];
}

int
ContextManager::fill_page_table(ContextId id, std::int32_t* out) const
{
  lock_guard<mutex> lk(_impl->mu);
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx || out == nullptr) {
    return 0;
  }
  int gstart = 0;
  for (size_t i = 0; i < ctx->page_list.size(); ++i) {
    out[i * 3 + 0] = static_cast<std::int32_t>(ctx->page_list[i].v);
    out[i * 3 + 1] = static_cast<std::int32_t>(ctx->page_valid[i]);
    out[i * 3 + 2] = gstart;
    gstart += ctx->page_valid[i];
  }
  return static_cast<int>(ctx->page_list.size());
}

const metal_compute::SharedBuffer*
ContextManager::conv_state(ContextId id, int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->is_linear_(layer)) {
    return nullptr;
  }
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx || static_cast<size_t>(layer) >= ctx->m_conv_state.size()) {
    return nullptr;
  }
  return &ctx->m_conv_state[static_cast<size_t>(layer)];
}

const metal_compute::SharedBuffer*
ContextManager::ssm_state(ContextId id, int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->is_linear_(layer)) {
    return nullptr;
  }
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx || static_cast<size_t>(layer) >= ctx->m_ssm_state.size()) {
    return nullptr;
  }
  return &ctx->m_ssm_state[static_cast<size_t>(layer)];
}

// ---- GDN recurrent-state run-ahead ring ------------------------------

const metal_compute::SharedBuffer*
ContextManager::ssm_read(ContextId id, int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->is_linear_(layer)) { return nullptr; }
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx || static_cast<size_t>(layer) >= ctx->m_ssm_state.size()) {
    return nullptr;
  }
  return _impl->gdn_ssm_slot_(ctx, layer, ctx->gdn_cur);
}

const metal_compute::SharedBuffer*
ContextManager::ssm_write(ContextId id, int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->is_linear_(layer)) { return nullptr; }
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx || static_cast<size_t>(layer) >= ctx->m_ssm_state.size()) {
    return nullptr;
  }
  const int R = ctx->gdn_ring_R;
  const int wslot = (R > 1) ? (ctx->gdn_cur + 1) % R : 0;
  return _impl->gdn_ssm_slot_(ctx, layer, wslot);
}

const metal_compute::SharedBuffer*
ContextManager::conv_read(ContextId id, int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->is_linear_(layer)) { return nullptr; }
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx || static_cast<size_t>(layer) >= ctx->m_conv_state.size()) {
    return nullptr;
  }
  return _impl->gdn_conv_slot_(ctx, layer, ctx->gdn_cur);
}

const metal_compute::SharedBuffer*
ContextManager::conv_write(ContextId id, int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->is_linear_(layer)) { return nullptr; }
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx || static_cast<size_t>(layer) >= ctx->m_conv_state.size()) {
    return nullptr;
  }
  const int R = ctx->gdn_ring_R;
  const int wslot = (R > 1) ? (ctx->gdn_cur + 1) % R : 0;
  return _impl->gdn_conv_slot_(ctx, layer, wslot);
}

const metal_compute::SharedBuffer*
ContextManager::ssm_slot(ContextId id, int layer, int k) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->is_linear_(layer)) { return nullptr; }
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx || static_cast<size_t>(layer) >= ctx->m_ssm_state.size()) {
    return nullptr;
  }
  const int R = ctx->gdn_ring_R;
  const int s = (R > 1) ? ((ctx->gdn_cur + k) % R + R) % R : 0;
  return _impl->gdn_ssm_slot_(ctx, layer, s);
}

const metal_compute::SharedBuffer*
ContextManager::conv_slot(ContextId id, int layer, int k) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->is_linear_(layer)) { return nullptr; }
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx || static_cast<size_t>(layer) >= ctx->m_conv_state.size()) {
    return nullptr;
  }
  const int R = ctx->gdn_ring_R;
  const int s = (R > 1) ? ((ctx->gdn_cur + k) % R + R) % R : 0;
  return _impl->gdn_conv_slot_(ctx, layer, s);
}

bool
ContextManager::gdn_ring_begin(ContextId id, int depth)
{
  lock_guard<mutex> lk(_impl->mu);
  // No-op when there's no GDN state to ring (dense model / non-metal) or the
  // depth is shallow enough that nothing speculative is ever in flight.
  if (!_impl->metal_enabled() || !_impl->has_ssm_layers || depth <= 1) {
    return true;
  }
  auto* ctx = _impl->find_locked_(id);
  if (!ctx) { return false; }
  const int R = depth + 1;
  const size_t want = static_cast<size_t>(R - 1);   // shadow slots per layer
  const int nl = _impl->spec.n_layers;
  ctx->m_conv_shadow.resize(static_cast<size_t>(nl));
  ctx->m_ssm_shadow.resize(static_cast<size_t>(nl));
  for (int L = 0; L < nl; ++L) {
    if (!_impl->is_linear_(L)) { continue; }
    auto& cs = ctx->m_conv_shadow[static_cast<size_t>(L)];
    auto& ss = ctx->m_ssm_shadow[static_cast<size_t>(L)];
    while (cs.size() < want) {
      auto cb = _impl->mc->make_shared_buffer(_impl->m_conv_bytes);
      auto sb = _impl->mc->make_shared_buffer(_impl->m_ssm_bytes);
      if (cb.empty() || sb.empty()) { return false; }
      std::memset(cb.contents(), 0, _impl->m_conv_bytes);
      std::memset(sb.contents(), 0, _impl->m_ssm_bytes);
      cs.push_back(std::move(cb));
      ss.push_back(std::move(sb));
    }
  }
  ctx->gdn_ring_R = R;
  ctx->gdn_cur = 0;            // current state stays in canonical slot 0
  return true;
}

void
ContextManager::gdn_ring_advance(ContextId id)
{
  lock_guard<mutex> lk(_impl->mu);
  auto* ctx = _impl->find_locked_(id);
  if (!ctx || ctx->gdn_ring_R <= 1) { return; }
  ctx->gdn_cur = (ctx->gdn_cur + 1) % ctx->gdn_ring_R;
}

int
ContextManager::gdn_ring_rollback(ContextId id, int n)
{
  lock_guard<mutex> lk(_impl->mu);
  auto* ctx = _impl->find_locked_(id);
  if (!ctx) { return -1; }
  if (n == 0) { return ctx->gdn_cur; }
  const int R = ctx->gdn_ring_R;
  if (R <= 1 || n < 0 || n > R - 1) { return -1; }   // past retained history
  ctx->gdn_cur = (ctx->gdn_cur - n + R) % R;
  return ctx->gdn_cur;
}

void
ContextManager::gdn_ring_end(ContextId id)
{
  lock_guard<mutex> lk(_impl->mu);
  auto* ctx = _impl->find_locked_(id);
  if (!ctx) { return; }
  const int cur = ctx->gdn_cur;
  if (cur != 0 && ctx->gdn_ring_R > 1) {
    // Move the final (post-rollback) state from the cursor slot into the
    // canonical slot 0 via an O(1) handle swap, so the next prefill / sync
    // forward (which read ssm_state / conv_state == slot 0) start correct.
    const size_t k = static_cast<size_t>(cur - 1);
    for (int L = 0; L < _impl->spec.n_layers; ++L) {
      if (!_impl->is_linear_(L)) { continue; }
      const size_t l = static_cast<size_t>(L);
      if (l < ctx->m_ssm_shadow.size() && k < ctx->m_ssm_shadow[l].size()) {
        std::swap(ctx->m_ssm_state[l], ctx->m_ssm_shadow[l][k]);
        std::swap(ctx->m_conv_state[l], ctx->m_conv_shadow[l][k]);
      }
    }
  }
  ctx->gdn_ring_R = 1;
  ctx->gdn_cur = 0;
}

// ---------------------------------------------------------------------
// Contiguous KV layout API (Spec::kv_layout == Contiguous)
// ---------------------------------------------------------------------

int
ContextManager::kv_seq_len(ContextId id) const
{
  lock_guard<mutex> lk(_impl->mu);
  const auto* ctx = _impl->find_locked_(id);
  return ctx ? ctx->seq_len : 0;
}

int
ContextManager::kv_append(ContextId id, int n)
{
  lock_guard<mutex> lk(_impl->mu);
  auto* ctx = _impl->find_locked_(id);
  if (!ctx || n <= 0 || !_impl->contig_enabled) {
    return -1;
  }
  if (_impl->spec.max_seq > 0 && ctx->seq_len + n > _impl->spec.max_seq) {
    return -1;
  }
  const int pos = ctx->seq_len;
  ctx->seq_len += n;
  return pos;
}

int
ContextManager::kv_rollback(ContextId id, int n)
{
  lock_guard<mutex> lk(_impl->mu);
  auto* ctx = _impl->find_locked_(id);
  if (!ctx || n <= 0 || n > ctx->seq_len) {
    return -1;
  }

  // Contiguous (Gemma): the sliding ring slots keep stale data (the next
  // append overwrites them at the same positions), so the ring side is a pure
  // counter rewind. When full layers are paged, also rewind their page tail
  // (drop the speculative pages) -- fall through to the paged rewind below,
  // which rewinds page_valid AND seq_len. Sliding-only contig (no full pool)
  // just rewinds the counter.
  if (_impl->contig_enabled && !_impl->full_paged) {
    ctx->seq_len -= n;
    return ctx->seq_len;
  }

  // Paged (Llama/Qwen): rewind the tail page_valid counts, freeing any page
  // that fully empties (a retained reserve page rewinds the cursor and stays
  // owned; an allocated page returns to the free list). Refuses to roll past
  // a frozen (branch-shared) tail page -- the speculative append that needs
  // rolling back always wrote into this context's own fresh page, so a
  // rollback that would touch a shared page is a misuse, not run-ahead.
  int rem = n;
  while (rem > 0 && !ctx->page_list.empty()) {
    const int v = ctx->page_valid.back();
    if (v > rem) {
      ctx->page_valid.back() = v - rem;
      rem = 0;
      break;
    }
    if (ctx->tail_frozen) {
      return -1;                       // would empty a shared page
    }
    const PageId pg = ctx->page_list.back();
    ctx->page_list.pop_back();
    ctx->page_valid.pop_back();
    rem -= v;
    if (ctx->retained_cursor > 0
        && ctx->retained_cursor <= ctx->retained.size()
        && ctx->retained[ctx->retained_cursor - 1].v == pg.v) {
      --ctx->retained_cursor;          // back to the reserve, still owned
    } else {
      _impl->release_page_locked_(pg);
    }
  }
  if (rem > 0) {
    return -1;                         // underflowed available pages
  }
  ctx->seq_len -= n;
  return ctx->seq_len;
}

const metal_compute::SharedBuffer*
ContextManager::kv_k(ContextId id, int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->contig_enabled || layer < 0 || layer >= _impl->spec.n_layers) {
    return nullptr;
  }
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx) {
    return nullptr;
  }
  const int src = _impl->c_src[(size_t)layer];
  const int own = src >= 0 ? src : layer;
  if (static_cast<size_t>(own) >= ctx->c_k.size()
      || ctx->c_k[(size_t)own].empty()) {
    return nullptr;
  }
  return &ctx->c_k[(size_t)own];
}

const metal_compute::SharedBuffer*
ContextManager::kv_v(ContextId id, int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->contig_enabled || layer < 0 || layer >= _impl->spec.n_layers) {
    return nullptr;
  }
  const auto* ctx = _impl->find_locked_(id);
  if (!ctx) {
    return nullptr;
  }
  const int src = _impl->c_src[(size_t)layer];
  const int own = src >= 0 ? src : layer;
  if (static_cast<size_t>(own) >= ctx->c_v.size()
      || ctx->c_v[(size_t)own].empty()) {
    return nullptr;
  }
  return &ctx->c_v[(size_t)own];
}

int
ContextManager::kv_capacity(int layer) const noexcept
{
  if (!_impl->contig_enabled || layer < 0 || layer >= _impl->spec.n_layers) {
    return 0;
  }
  return _impl->c_cap[(size_t)layer];
}

int
ContextManager::kv_head_dim(int layer) const noexcept
{
  if (!_impl->contig_enabled || layer < 0 || layer >= _impl->spec.n_layers) {
    return 0;
  }
  return _impl->c_hd[(size_t)layer];
}

int
ContextManager::kv_window(int layer) const noexcept
{
  if (!_impl->contig_enabled || layer < 0 || layer >= _impl->spec.n_layers) {
    return 0;
  }
  return _impl->c_win[(size_t)layer];
}

int
ContextManager::kv_ring_cap(int layer) const noexcept
{
  if (!_impl->contig_enabled || layer < 0 || layer >= _impl->spec.n_layers) {
    return 0;
  }
  // A bounded ring iff the capacity was shrunk below max_seq (a Sliding
  // layer with a ring chunk). Full / unbounded layers address linearly.
  const int cap = _impl->c_cap[(size_t)layer];
  return (cap < _impl->spec.max_seq) ? cap : 0;
}

int
ContextManager::kv_capacity(ContextId id, int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->contig_enabled || layer < 0 || layer >= _impl->spec.n_layers) {
    return 0;
  }
  const auto* ctx = _impl->find_locked_(id);
  // The PHYSICAL head stride (ring modulo + mirror tail). The ring modulo is
  // reported separately by kv_ring_cap; the read/write kernels address with
  // this stride and wrap by the modulo (or, flag-on, read the mirror linearly).
  return ctx ? _impl->ctx_stride_(*ctx, layer) : _impl->c_cap[(size_t)layer];
}

int
ContextManager::kv_ring_cap(ContextId id, int layer) const
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->contig_enabled || layer < 0 || layer >= _impl->spec.n_layers) {
    return 0;
  }
  const auto* ctx = _impl->find_locked_(id);
  const int cap = ctx ? _impl->ctx_cap_(*ctx, layer)
                      : _impl->c_cap[(size_t)layer];
  return (cap < _impl->spec.max_seq) ? cap : 0;
}

bool
ContextManager::ensure_sliding_capacity(ContextId id, int need)
{
  lock_guard<mutex> lk(_impl->mu);
  if (!_impl->contig_enabled || !_impl->metal_enabled()) {
    return false;
  }
  auto* ctx = _impl->find_locked_(id);
  if (!ctx) {
    return false;
  }
  const int max_seq = _impl->spec.max_seq;
  if (need <= 0 || (max_seq > 0 && need > max_seq)) {
    return false;
  }
  // Realloc discards KV: only valid before any token is written.
  if (ctx->seq_len != 0) {
    return false;
  }
  int want = ((need + 15) / 16) * 16;          // 16-align (mma2 over-read)
  if (max_seq > 0 && want > max_seq) {
    want = max_seq;
  }
  // Current effective sliding cap (all sliding owning layers share it).
  int cur = ctx->sliding_cap;
  if (cur == 0) {
    for (int L = 0; L < _impl->spec.n_layers; ++L) {
      if (_impl->contig_owns_(L)
          && _impl->c_kind[(size_t)L] == LayerKind::Sliding) {
        cur = _impl->c_cap[(size_t)L];
        break;
      }
    }
  }
  if (want <= cur) {
    return true;                               // ring already holds the prompt
  }
  // Grow every sliding owning layer's K/V buffer to `want` tokens.
  for (int L = 0; L < _impl->spec.n_layers; ++L) {
    if (!_impl->contig_owns_(L)
        || _impl->c_kind[(size_t)L] != LayerKind::Sliding) {
      continue;
    }
    const int nkv = !_impl->c_nkv.empty() ? _impl->c_nkv[(size_t)L]
                                          : _impl->spec.n_kv_heads;
    // Physical stride = grown ring modulo + mirror tail. The tail is dropped
    // once the ring reaches max_seq (then ring_cap == 0 -> linear, no mirror).
    const int win = !_impl->c_win.empty() ? _impl->c_win[(size_t)L] : 0;
    const int tail =
        (win > 1 && want < _impl->spec.max_seq) ? (win - 1) : 0;
    const size_t n = (size_t)nkv * (size_t)(want + tail)
                     * (size_t)_impl->c_hd[(size_t)L] * 2;
    auto k = _impl->mc->make_shared_buffer(n);
    auto v = _impl->mc->make_shared_buffer(n);
    if (k.empty() || v.empty()) {
      return false;
    }
    std::memset(k.contents(), 0, n);
    std::memset(v.contents(), 0, n);
    ctx->c_k[(size_t)L] = std::move(k);
    ctx->c_v[(size_t)L] = std::move(v);
  }
  ctx->sliding_cap = want;
  return true;
}


void
ContextManager::set_eval_per_write(bool b) noexcept
{
  _impl->eval_per_write = b;
}


}
