#ifndef GENERATIVE_MODELS_SHARED_METAL_SOL_ATTENTION_H
#define GENERATIVE_MODELS_SHARED_METAL_SOL_ATTENTION_H

// Sol-Attn on the metal-compute backend: seven dispatches and the
// scratch they share. The method, and the CPU oracle these kernels are
// checked against, are in sol-attention.h.
//
// THE ATTENTION GOES TO THE FLASH KERNEL. That is the whole design. A
// first port ran the entire method in one plain simdgroup kernel and
// measured 281 GF/s against steel's 6656, so the routing's saving was
// buried under a 24x kernel deficit. Here BOTH halves are the tree's
// own flash attention -- the exact blocks through its has_spans path,
// the approximate ones over the summary sequence with the routing's
// flags as a mask -- and only the summaries, the routing and the merge
// are Sol's own code.
//
//   sol_summaries_mma  x2   block centroids; q at the QUERY block size,
//                           k/v at the KEY block size
//   sol_kc_stats_mma        the centroids' spread, for the threshold
//   sol_proxy_mma           the proxy scores, one batched GEMM (M5)
//   sol_route_p_mma         threshold, decision, flag rows, kept counts
//     (or sol_route_mma, which fuses the proxy into the decision)
//   sol_scan_mma            the counts into a CSR offset array
//   sol_emit_mma            the flag rows into the CSR block list
//   attn_steel[_nax] (has_block_mask + export_ml)   the approximate half
//   attn_steel[_nax] (has_spans + export_ml)        the exact half
//   sol_merge_ml_mma        the two partial softmaxes into one output
//     (or sol_merge_mma, beside sol_approx_mma, under
//      VPIPE_SOL_NO_MASKED_APPROX)
//
// WHICH FLASH KERNEL DECIDES THE TILES, and they are not the same: the
// ALU steel entry routes at a 32-row query block and walks a CSR in
// 16-key units, the matrix-core one at 64 and 32. A coarser query block
// keeps slightly more (15.1% against 13.8% at tau 1.0) for the same
// accuracy, and is far faster; VPIPE_SOL_NO_NAX is the A/B.
//
// MEASURED, 56 heads x 20036 rows x 128, bf16, each arm against THE
// SAME kernel run dense:
//
//   arm    dense       sol    speedup   kept
//   ALU    3271 ms   640 ms    5.11x    13.8%
//   NAX    1149 ms   247 ms    4.65x    15.0%
//
// THE NUMBER THAT MATTERS ON AN M5 IS NEITHER COLUMN'S RATIO. The model
// runs its dense attention on the NAX kernel, so before this port Sol
// was 1149 -> 640 ms, i.e. 1.80x, not the 5.11x the ALU column reads:
// the exact half was giving up the faster kernel that the baseline it
// replaces was already using. It is now 4.65x, and 13.2x against the
// ALU kernel run dense.
//
// The key block knee moved with the port and then moved back. On the
// ALU arm 64 wins (5.11x against 4.27 / 4.36 / 2.89 at 32 / 128 / 256);
// with only the exact half on the matrix cores 128 briefly won, the
// simdgroup approximate half having become a third of the call; with
// both halves there it is 64 again -- which is also the more accurate
// block, so the fast choice and the faithful one are the same one.
//
// THE ALU ROW ABOVE PREDATES THE MASK REACHING THE ALU KERNEL, and its
// approximate half was still sol_approx_mma. MEASURED again on an M4
// Pro, same 56 heads x 20036 rows x 128 against the same kernel dense:
//
//   block    dense       sol    speedup     was
//      32   1727 ms   376 ms    4.59x    2.88x
//      64   1729 ms   372 ms    4.65x    3.63x
//     128   1728 ms   448 ms    3.85x    3.44x
//     256   1756 ms   670 ms    2.62x    2.54x
//
// AND THE KNEE IS GONE AT THE TOP. 32 and 64 are within 1% of each
// other now that the approximate half is no longer the term that grows
// when the block shrinks -- so on this hardware the FAITHFUL block is
// free, where it used to cost 26%. The rest of the ordering stands.
//
// AND IT ALLOCATES ALMOST NOTHING. Every buffer here lives for ONE
// call -- summaries, routing, CSR and both partial softmaxes are all
// consumed by the merge that ends it -- so it takes memory the caller
// is not using rather than memory of its own. The DiT has more than
// enough: with Sol on, the fused qkv projection has already been
// transposed out and is dead until the feed-forward, and the attention
// arena's last window is dead until the transpose that follows the
// call. MEASURED at 56 heads x 14861 rows: 861 MB held became 0.
//
// Two of those 861 were never needed at all -- 427 MB of it was the ALU
// arm's fp32 padded partial, allocated on the matrix-core arm too,
// where nothing reads it.
//
// FAMILY-AGNOSTIC ON PURPOSE. Nothing here knows about a DiT: it takes
// head-major [H, T, D] q/k/v and returns the same, which is the shape
// every unfused attention path in this tree already speaks. A model
// adopts it by routing one dispatch through `encode` -- see the
// MiniMax-H3 transformer, which is the first caller.

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/sol-attention.h"

#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MetalSolAttention {
 public:
  // `bf16` selects the bfloat twin; false is f16. Null (with `err` set)
  // when a ComputeFunction does not validate -- never an object that
  // dispatches no-ops, which is the failure this backend makes silent.
  //
  // The steel function is specialised per SEQUENCE LENGTH (align_Q /
  // align_K are function constants), so it is built on first use and
  // rebuilt when the geometry moves, not here.
  static std::unique_ptr<MetalSolAttention>
  load(metal_compute::MetalCompute* mc, bool bf16, std::string* err);

  // Scratch bytes for one call at this geometry, so a caller can size
  // the box before encoding.
  //
  // WHICH IT HAD BETTER DO. This is spent at the first encode, inside
  // the first forward and therefore after every planning decision --
  // and at 56 heads x 14861 rows it was 861 MB. Until now nothing
  // called it at all, so the DiT kept streamed blocks resident against
  // a budget that had never heard of Sol.
  //
  // `nax` and `key_block` change the answer and are not this class's to
  // guess from a static: the matrix-core arm allocates a bf16 partial
  // where the ALU one allocates an fp32 padded one, and the key block
  // sets the summary sequence. uses_matrix_cores() answers the first
  // once an object exists; a planner with none asks
  // MetalCompute::supports_matrix_cores().
  static std::size_t scratch_bytes(int heads, int tokens, int d,
                                   int key_block, bool nax);
  // GQA. `kv_heads` is how many heads K and V carry; `heads` is always
  // the QUERY count, because that is what the routing, the CSR and the
  // output are indexed by. The two-argument form above is this one with
  // kv_heads == heads, which is MHA and what every caller meant before
  // the distinction existed.
  static std::size_t scratch_bytes(int heads, int kv_heads, int tokens,
                                   int d, int key_block, bool nax);

  // ...and what is left AFTER a lend of `lend_a` / `lend_b` bytes: the
  // same greedy carve, alignment and all, rather than a subtraction.
  // This is the number a planner adds to the model's own scratch.
  static std::size_t private_bytes(int heads, int tokens, int d,
                                   int key_block, bool nax,
                                   std::size_t lend_a, std::size_t lend_b);
  static std::size_t private_bytes(int heads, int kv_heads, int tokens,
                                   int d, int key_block, bool nax,
                                   std::size_t lend_a, std::size_t lend_b);

  // Encode summaries -> stats -> threshold -> forward into `enc`.
  //
  // The four are ordered by data dependence and Metal's serial dispatch
  // type orders consecutive dispatches, so no commit is needed between
  // them -- the whole call is one encoder, like every other attention
  // path here.
  bool encode(metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& q,
              const metal_compute::SharedBuffer& k,
              const metal_compute::SharedBuffer& v,
              metal_compute::SharedBuffer& out, int heads, int tokens,
              int d, float scale, const sol::Config& cfg,
              std::string* err);

  // GQA: K and V carry `kv_heads` heads, Q and the output `heads`.
  //
  // The summaries of K and V are per KV head -- that is what there is to
  // summarise -- while the routing decision, the flags, the CSR and both
  // partial softmaxes stay per QUERY head, so a query head reads the
  // group it belongs to. The exact half is steel, which has always been
  // GQA-aware; this sets its `gqa` instead of the 1 it was pinned at.
  //
  // `heads % kv_heads` must be 0. The form above is this one at
  // kv_heads == heads.
  //
  // REFUSED under VPIPE_SOL_NO_MASKED_APPROX: the simdgroup approximate
  // kernel indexes the summary sequence by the query head, so it would
  // read another group's centroids. That path is a bench A/B, and a
  // wrong answer there is worse than not having it.
  bool encode(metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& q,
              const metal_compute::SharedBuffer& k,
              const metal_compute::SharedBuffer& v,
              metal_compute::SharedBuffer& out, int heads, int kv_heads,
              int tokens, int d, float scale, const sol::Config& cfg,
              std::string* err);

  // A RECTANGULAR BAND of a longer sequence.
  //
  // `q`/`out` hold `q_rows` rows per head and this call attends
  // `q_tokens` of them starting at `q_off`; `k`/`v` hold `k_rows` and
  // it reads `k_tokens` from the start. The square forms above are this
  // one with every field equal to the sequence length and `q_off` 0.
  //
  // `q_off` is the query's position IN THE SEQUENCE, which is what the
  // local band is measured against -- not a buffer offset. For a
  // segment of a joint stream the two coincide, and they are named
  // apart because nothing guarantees it.
  struct Band {
    int q_tokens = 0;   // queries this call attends
    int k_tokens = 0;   // keys it attends them against, from row 0
    int q_rows   = 0;   // rows per head in `q` / `out`
    int k_rows   = 0;   // rows per head in `k` / `v`
    int q_off    = 0;   // sequence position of the first query
    // The ROW of `q`/`out` the band starts at. Distinct from `q_off`:
    // that is where the queries sit in the SEQUENCE and is what the
    // local band is measured against, this is where they sit in the
    // BUFFER. They coincide when the buffer holds the whole joint
    // stream -- which is the common case and exactly why conflating
    // them survives every test that does not separate them.
    int q_row0   = 0;
  };

  bool encode(metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& q,
              const metal_compute::SharedBuffer& k,
              const metal_compute::SharedBuffer& v,
              metal_compute::SharedBuffer& out, int heads, int kv_heads,
              const Band& band, int d, float scale, const sol::Config& cfg,
              std::string* err);

  // Blocks kept exact by the LAST completed encode, and the total the
  // routing chose from. Valid only after the command buffer has run.
  // Realized sparsity is a property of the data, so it is read back
  // rather than predicted.
  long long exact_blocks() const;
  long long total_blocks() const { return _total_blocks; }
  void      reset_counts();

  std::size_t resident_bytes() const;

  // LEND THIS MEMORY THAT IS DEAD FOR THE LENGTH OF ONE CALL, and the
  // scratch above stops being an allocation.
  //
  // Every buffer this class holds bar a few hundred bytes is written
  // and read inside ONE encode: the summaries, the routing decision,
  // the CSR and the two partial softmaxes are all consumed by the merge
  // that ends it. So it does not need memory of its own, it needs
  // memory nobody else is using between the projection and the output.
  //
  // THE CALLER HAS EXACTLY THAT, AND MORE THAN ENOUGH OF IT. Sol reads
  // head-major q/k/v, which means the fused [seq, 3*inner] projection
  // has already been transposed out and is dead until the feed-forward
  // writes over it -- three attention windows' worth -- and the
  // attention arena's LAST window is dead too, being written only by
  // the transpose that follows the call. MEASURED at 56 heads x 14861
  // rows: 853 MB dead against 489 MB wanted.
  //
  // Two regions rather than one because those are two allocations.
  // Anything that fits neither is allocated privately, per buffer, so a
  // caller that lends nothing (the tests, the isolated benches) is
  // unchanged.
  //
  // Set it BEFORE the first encode, and lend nothing again before
  // replacing what was lent: the carved buffers are windows that hold
  // the lender's allocation alive, so a caller that reallocates without
  // taking the loan back pays for both.
  void set_arena(const metal_compute::SharedBuffer& a,
                 const metal_compute::SharedBuffer& b);

  // SAGE ON THE EXACT HALF, which is the one that composes.
  //
  // The two are orthogonal by construction: Sol decides WHICH key blocks
  // are attended and Sage how the ones that are get multiplied. The
  // exact half reads the full q/k -- the same tensors a caller's own
  // prologue quantizes -- so it takes the int8 QK with no change to the
  // kernel at all: the int8 operands there are indexed by the ABSOLUTE
  // key block out of `qb_blocks`, not by the advancing pointer the f16
  // path walks, so the sparse jump needs no twin. That was true before
  // anything composed them and is why this is a host-side change only.
  //
  // The APPROXIMATE half is deliberately left alone. It attends the
  // block SUMMARIES, not the keys -- a 1/64-scale product against a
  // tensor this object built itself -- so quantizing it would need a
  // second prologue over `kc` to save a fraction of a fraction.
  //
  // The caller runs the prologue and passes the driver here BEFORE
  // encode(); `nullptr` (the default) is the f16 exact half. Borrowed
  // for the length of the call, and only honoured on the matrix-core
  // arm, where the int8 fragment MMA exists.
  void set_sage(const MetalSageAttention* sage);

  // THE SAGE CONFIG THAT COMPOSES, and it is not the one a caller would
  // pass on its own.
  //
  // Sage quantizes K as K - mean(K), which is EXACT under a softmax
  // because a per-channel shift moves every score in a ROW by the same
  // <q, mean> and softmax does not see a per-row constant. That argument
  // needs the whole row to be inside ONE softmax. Sol's is not: the row
  // is split between the exact half and the approximate one, the
  // approximate half scores against centroids built from the UNSHIFTED
  // keys, and the merge combines two separately normalised partial
  // softmaxes. The shift then does not cancel -- it displaces one half
  // against the other by exp(<q_i, mean>), per row and unbounded.
  //
  // MEASURED in the H3 stack against a dense reference: Sol alone
  // 0.0190, Sage alone 0.0049, composed WITH the smoothing 0.0376 --
  // twice Sol's own error, from a term that should have cancelled -- and
  // composed without it 0.0198. Over 50 layers at video geometry that is
  // the difference between a clip and noise.
  //
  // So a composing caller quantizes K RAW, and pays the int8 range the
  // smoothing was buying (MEASURED elsewhere: cosine 0.9985 against
  // 0.9999 on deliberately biased keys). Recovering it means shifting
  // Sol's own centroids by the same mean so both halves speak one logit
  // space -- routing is invariant under that, since a uniform shift
  // moves the proxy distribution and its threshold together -- and that
  // is the better fix and a larger one.
  static sage::Config compose(sage::Config in)
  {
    in.smooth_k = false;
    return in;
  }

  // Whether the last encode() actually ran the exact half in int8 -- so
  // a caller reports what happened rather than what it asked for, and an
  // A/B cannot measure the same arm twice.
  bool sage_engaged() const noexcept { return _sage_on; }

  // What this object allocates for itself however much is lent: the
  // AttnParams pair, the span params and their bounds, and the routing
  // counter. All host-written or host-read, so none of them can live in
  // memory the caller reuses between calls.
  static std::size_t pinned_bytes();

  // Whether the exact half is on the matrix-core flash kernel, and the
  // query/key block the routing is therefore decided at. A bench that
  // compares against dense has to run the SAME kernel dense or it is
  // measuring the port and calling it the approximation.
  bool uses_matrix_cores() const { return _nax; }
  int  query_block() const { return _bq; }
  int  key_block_unit() const { return _bk; }

  // Removes this object's scratch from the process-wide residency set.
  // Public because the DESTRUCTOR is not the only moment it matters: a
  // lender that is about to take its arena back wants the set to stop
  // holding it, and set_arena() does that on the way through.
  ~MetalSolAttention();

 private:
  MetalSolAttention() = default;
  std::vector<metal_compute::SharedBuffer*> scratch_buffers_();
  // THE OTHER HALF OF residency_add. MTLResidencySet RETAINS what is
  // added to it and the set lives on MetalCompute, which outlives every
  // model -- so an add without a matching remove is not a hint the
  // kernel may ignore, it is an allocation that never comes back.
  //
  // It is worse than its own size, because most of these buffers are
  // WINDOWS carved out of an arena the caller lent (set_arena): a
  // subview shares its parent's MTL::Buffer, so adding one hands the set
  // the whole parent. Sol's few hundred megabytes of scratch that way
  // pinned the DiT's multi-gigabyte forward arena, which then survived
  // `unload_when_idle: destroy` and left the VAE decode with nothing.
  // See metal_compute_residency.a_subview_adds_the_whole_parent.
  void drop_residency_();
  bool ensure_scratch_(int heads, int kv_heads, int q_tokens, int k_tokens,
                       int d, std::string* err);
  void set_band_params_(int heads, int kv_heads, const Band& band, int d);
  bool ensure_steel_(int tokens, std::string* err);
  // The steel entry point for this object's kernel arm AND head width.
  metal_compute::ComputeFunction _lib_for_width_(
      const metal_compute::FunctionConstants& fc) const;

  metal_compute::MetalCompute* _mc = nullptr;
  bool _bf16 = false;
  metal_compute::ComputeLibrary  _lib;
  metal_compute::ComputeLibrary  _lib_steel;
  metal_compute::ComputeLibrary  _lib_nax;
  metal_compute::ComputeFunction _fn_sum, _fn_stats, _fn_route, _fn_scan;
  metal_compute::ComputeFunction _fn_emit, _fn_approx, _fn_merge;
  metal_compute::ComputeFunction _fn_merge_ml;
  metal_compute::ComputeFunction _fn_steel;   // has_spans + export_ml
  // THE APPROXIMATE HALF ON THE SAME FLASH KERNEL, where there is one
  // worth using. It is an attention over the summary sequence with the
  // routing's flag array as a per-(query block, key) mask, so it needs
  // no kernel of its own -- and it was 34% of a routed call once the
  // exact half moved to the matrix cores, being the last piece still
  // running on plain simdgroup matrices at 1/64 of the work but nothing
  // like 1/64 of the time.
  //
  // AND THE SAME IS TRUE WITHOUT MATRIX CORES, which is why the mask is
  // now in both flash kernels rather than only the matrix-core one.
  // MEASURED on an M4 Pro, 56 heads x 20036 rows x 128, the pass timed
  // alone: 123.3 ms on sol_approx_mma against a dense-scaled ideal of
  // ~27, and the same half on attn_steel with the flag mask.
  metal_compute::ComputeFunction _fn_approx_masked;
  // The exact half's int8 twin, and the driver that fills its operands.
  // The pointer is BORROWED -- the caller owns the driver and has
  // already run its prologue into the same encoder.
  const MetalSageAttention*     _sage = nullptr;
  bool                          _sage_on = false;
  // Whether the cached exact function was built for the int8 QK. A tag
  // beside the sequence and the summary length, because turning Sage on
  // changes the function and nothing else about the geometry does.
  int                           _steel_sage = -1;
  int                           _steel_d = 0;
  // Whether the approximate half takes that kernel. True unless
  // VPIPE_SOL_NO_MASKED_APPROX asks for the simdgroup one, which is the
  // A/B and the reason sol_approx_mma is still here.
  bool _masked = true;
  // The routing proxy as one batched GEMM, and the routing kernel that
  // reads it. The fused sol_route_mma stays for the ALU arm: it is the
  // reference the two are checked against, and on a GPU with no matrix
  // units a separate proxy matrix would be pure extra traffic.
  metal_compute::ComputeLibrary  _lib_proxy;
  metal_compute::ComputeFunction _fn_proxy, _fn_route_p;
  int _steel_seq = -1, _steel_nk = -1;
  // THE EXACT HALF'S KERNEL, AND ITS TILE. On a matrix-core GPU the
  // blocks the routing keeps go to attn_steel_nax, whose query block is
  // 64 and key block 32 against the ALU kernel's 32 and 16 -- and those
  // tiles are not this class's to choose: the CSR is in the exact
  // kernel's own units and the routing is decided at its query block,
  // so picking the kernel picks both.
  //
  // A COARSER ROUTING BLOCK IS A REAL DIFFERENCE, not just a faster
  // one: one threshold then serves 64 rows rather than 32, so the two
  // kernels keep slightly different block sets and the outputs differ
  // by more than precision. That is the same trade `key_block` makes on
  // the other axis, and it is why VPIPE_SOL_NO_NAX exists.
  bool _nax = false;
  int  _bq = 0, _bk = 0;

  metal_compute::SharedBuffer _qc, _kc, _vc, _mean, _var;
  metal_compute::SharedBuffer _flags, _kept, _qb_off, _qb_blk;
  metal_compute::SharedBuffer _o_a, _m_a, _l_a, _o_e, _m_e, _l_e;
  // The flash approximate half's output: the tensor dtype and the plain
  // [H, T, D] shape, against sol_approx_mma's fp32 [H, TPAD, D].
  metal_compute::SharedBuffer _o_an, _params_a;
  // [H, NQ, NK] fp32 -- the proxy scores, when they are computed as a
  // GEMM rather than inside the routing kernel.
  metal_compute::SharedBuffer _proxy;
  // Up to two regions the caller has lent (set_arena), each with its own
  // bump cursor. Empty is the ordinary standalone-allocation case.
  metal_compute::SharedBuffer _arena_a, _arena_b;
  std::size_t                 _arena_used[2] = {0, 0};
  // Whether the buffers below are currently in the residency set, so the
  // remove walks the same list the add did and only when there is
  // something to walk.
  bool                        _res_added = false;
  metal_compute::SharedBuffer _counts, _params, _sp_params, _sp_bounds;
  int _heads = 0, _tokens = 0, _d = 0, _nq = 0, _nk = 0, _tpad = 0;
  // The KEY extent the scratch was built for. `_tokens` is the QUERY
  // one; they were the same number until the band form.
  int _k_tokens = 0;
  // Heads K and V carry. Equal to _heads under MHA, which is what every
  // caller was until GQA; part of the geometry tag because the key
  // summaries are sized by it.
  int _kv_heads = 0;
  int _nqa = 0;      // approximate-half tiles: ceil(T / 32), always
  int _blk = 0;
  long long _total_blocks = 0;
};

}  // namespace genai
}  // namespace vpipe

#endif  // GENERATIVE_MODELS_SHARED_METAL_SOL_ATTENTION_H
