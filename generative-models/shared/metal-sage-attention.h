#ifndef GENERATIVE_MODELS_SHARED_METAL_SAGE_ATTENTION_H
#define GENERATIVE_MODELS_SHARED_METAL_SAGE_ATTENTION_H

// SageAttention on metal-compute: the int8 QK prologue and its scratch.
//
// WHAT THIS CLASS IS NOT. It does not encode an attention. The int8 QK
// lives INSIDE the steel flash kernel, behind function constant 306, so
// the caller keeps dispatching its own attention exactly as before --
// same entry point, same tiles, same params. This class owns the four
// operands that constant asks for and the three kernels that fill them.
//
// A caller therefore does three things, and a family that already runs
// attn_steel_nax has all three within a few lines of each other:
//
//   1. build its attention function with .set_bool(306, true)
//   2. call prepare() into the same encoder, before the dispatch
//   3. call bind() on that dispatch
//
// and nothing else about its forward changes. That is deliberate: the
// alternative -- a class that owns the attention -- would have had to
// reproduce every family's mask, span, sink and stride convention, and
// the four families here do not agree on any of them.
//
// THE LAYOUT IT WRITES, which is the one thing a caller must respect.
// q8/k8 are TIGHTLY PACKED [heads, tokens, D], head-major, because that
// is what the kernel indexes. The SOURCE is not assumed to be: q and k
// are usually a run inside a fused qkv projection, so each carries its
// own row stride and element offset. Scales are [heads, blocks] fp32,
// one per attention tile.

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/sage-attention.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// TOOLKIT: compiled into each plugin (libvpipe_toolkit.a), not exported
// by libvpipe -- see the vpipe_toolkit target and plugin/plugin-abi.h.

namespace vpipe {
namespace metal_compute {
class ComputeEncoder;
}

namespace genai {

class MetalSageAttention {
 public:
  // Null (with `err` set) when the box has no matrix cores or a kernel
  // does not validate -- never an object that dispatches no-ops, which
  // is the failure this backend makes silent. `bf16` selects the
  // bfloat twin of the quantize kernels; false is f16.
  static std::unique_ptr<MetalSageAttention>
  load(metal_compute::MetalCompute* mc, bool bf16, std::string* err);

  // THE FOUR FAMILIES' LOAD, in one place. Each of them has the same
  // three-way answer to make and the same way of reporting it, and four
  // hand-written copies of that is four chances to differ on the one
  // case that matters:
  //
  //   config did not ask          -> null, *fatal false
  //   asked, no matrix cores      -> null, *fatal false, said ONCE
  //   asked, kernels did not load -> null, *fatal TRUE
  //
  // The middle case is not a failure. The int8 QK is a matrix-core
  // instruction with no ALU fallback, so refusing on an M4 would make a
  // graph that runs today un-runnable on half the boxes it runs on now.
  // The last case IS one: a model that asked for Sage and quietly ran
  // dense would be reported as a Sage run and its numbers believed.
  static std::unique_ptr<MetalSageAttention>
  load_for_model(metal_compute::MetalCompute* mc, bool bf16,
                 const sage::Config& cfg, const char* who, bool* fatal);

  ~MetalSageAttention();

  // CHEAP, and safe to ask before deciding anything. A caller that has
  // no model yet -- a stage sizing a graph, a family answering
  // declare_resources -- can ask this without loading kernels.
  static bool available(const metal_compute::MetalCompute* mc);

  // The scratch one call needs, so a caller can size the box before it
  // encodes. `bq` and `bk` are the ATTENTION kernel's query and key
  // block sizes, because the scales are one per its tiles; see
  // nax_query_block() / nax_key_block().
  // `kv_heads` is the KEY head count, which is `heads` for MHA and
  // heads / gqa_factor for a grouped family (FLUX.2 and Krea-2 are
  // both grouped; H3 and Qwen-Image are not). K is quantized once per
  // KV head, not once per query head -- the kernel indexes it the same
  // way it indexes the f16 K.
  static std::size_t scratch_bytes(int heads, int kv_heads, int q_tokens,
                                   int k_tokens, int d, int bq, int bk);

  // ...and what it will still allocate for ITSELF once `lend_a` and
  // `lend_b` bytes have been lent, which is the figure a caller's
  // memory plan actually wants. Simulates the same greedy carve
  // ensure_scratch_ runs, alignment included, so the two cannot
  // disagree about which buffers fit.
  static std::size_t private_bytes(int heads, int kv_heads, int q_tokens,
                                   int k_tokens, int d, int bq, int bk,
                                   std::size_t lend_a, std::size_t lend_b);

  // The steel NAX kernel's tiles at head_dim 128, which is what every
  // DiT in this tree runs. Named rather than open-coded at four call
  // sites: they are the kernel's, and a caller that guessed them wrong
  // would size the scale arrays wrong and read past them.
  static int nax_query_block();
  static int nax_key_block();

  // Up to two regions the caller has lent, each with its own bump
  // cursor, exactly as MetalSolAttention::set_arena. Anything that fits
  // neither is allocated privately, per buffer, so a caller that lends
  // nothing is unchanged.
  //
  // Set it BEFORE the first prepare(), and lend nothing again before
  // replacing what was lent: the carved buffers are windows that hold
  // the lender's allocation alive.
  void set_arena(const metal_compute::SharedBuffer& a,
                 const metal_compute::SharedBuffer& b);

  // What this object is holding, arena windows included.
  std::size_t resident_bytes() const;

  // ...and the part of that it allocated ITSELF, the lend having been
  // too small or absent. This is what a memory plan is actually asking
  // for, and it is TALLIED as the carve runs rather than inferred from
  // the device's allocated size -- that figure is process-wide, moves
  // with every other Metal object, and cannot answer a question about
  // one object. private_bytes() predicts this number; they must agree.
  std::size_t private_held() const noexcept { return _own; }

  // The prologue for ONE attention call: the key mean, then Q and K
  // quantized per block. Encoded into `e`, before the caller's own
  // attention dispatch and into the same encoder -- the kernel reads
  // what this wrote, so they are ordered by the encoder and need no
  // barrier of their own.
  //
  // HOW THE SOURCE IS ADDRESSED, which is the part that has to match the
  // caller's own attention params exactly. `Operand` is the same three
  // numbers steel's Q_strides carry, in ELEMENTS of the tensor dtype:
  //
  //   off    the element offset of head 0, row 0, channel 0
  //   row    the stride between consecutive TOKENS
  //   head   the stride between consecutive HEADS
  //
  // For head-major [H, L, D] that is (0, D, L*D). For q/k read in place
  // inside a fused [rows, 3*I] projection -- which is what every DiT in
  // this tree does -- it is (Q_OFF, 3*I, head_dim). The head stride is
  // NOT derived from the row stride: those two are unrelated in the
  // fused layout, and deriving it reads the wrong head silently.
  struct Operand {
    const metal_compute::SharedBuffer* buf = nullptr;
    std::size_t off  = 0;
    int         row  = 0;
    int         head = 0;
  };

  bool prepare(metal_compute::ComputeEncoder& e, const Operand& q,
               const Operand& k, int heads, int kv_heads, int q_tokens,
               int k_tokens, int d, int bq, int bk, const sage::Config& cfg,
               std::string* err);

  // Bind the four operands at buffers 15..18 onto a dispatch whose
  // function was built with constant 306 true. Call AFTER set_function
  // and the caller's own binds, like any other bind.
  //
  // Binding these onto a function built WITHOUT 306 is harmless (the
  // slots are unused); the reverse is not, and is why prepare() refuses
  // rather than warns when its scratch is not there.
  void bind(metal_compute::ComputeEncoder& e) const;

  // Whether the last prepare() produced usable operands. A caller that
  // wants to fall back to dense within one forward checks this rather
  // than assuming; false means bind() would bind nothing.
  bool ready() const noexcept { return _ready; }

  // The geometry the current scratch was built for, for a caller that
  // caches a function per shape and wants to know when to rebuild.
  int heads() const noexcept { return _heads; }
  int q_tokens() const noexcept { return _qt; }
  int k_tokens() const noexcept { return _kt; }

 private:
  MetalSageAttention() = default;
  std::vector<metal_compute::SharedBuffer*> scratch_buffers_();
  bool ensure_scratch_(int heads, int kv_heads, int q_tokens, int k_tokens,
                       int d, int bq, int bk, std::string* err);
  // Every add to the process-wide residency set is owed a remove: the
  // set RETAINS what it holds and outlives every model, and most of
  // these buffers are windows into an arena the caller lent -- a
  // subview shares its parent's MTL::Buffer, so adding one hands the
  // set the whole parent. See
  // metal_compute_residency.a_subview_adds_the_whole_parent.
  void drop_residency_();

  metal_compute::MetalCompute*  _mc = nullptr;
  bool                          _bf16 = false;
  metal_compute::ComputeLibrary _lib;
  metal_compute::ComputeFunction _fn_sum, _fn_reduce, _fn_quant;

  // [H, D] fp32 key mean, and the per-chunk partials it reduces from.
  metal_compute::SharedBuffer _km, _kmp;
  // [H, qL, D] / [H, kL, D] int8, tightly packed.
  metal_compute::SharedBuffer _q8, _k8;
  // [H, NQ] / [H, NK] fp32, one scale per attention tile.
  metal_compute::SharedBuffer _qs, _ks;

  metal_compute::SharedBuffer _arena_a, _arena_b;
  std::size_t                 _arena_used[2] = {0, 0};
  bool                        _res_added = false;
  // Tallied by the carve: what did NOT fit in the lend.
  std::size_t                 _own = 0;

  int  _heads = 0, _kvh = 0, _qt = 0, _kt = 0, _d = 0, _bq = 0, _bk = 0;
  bool _ready = false;
};

}  // namespace genai
}  // namespace vpipe

#endif  // GENERATIVE_MODELS_SHARED_METAL_SAGE_ATTENTION_H
