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
  //
  // Only the int8 QK's three kernels are REQUIRED. The int8 P*V's four
  // came later, and this class is compiled into plugins: one built
  // against a newer SDK runs on hosts whose library lacks them, and
  // refusing there would turn every Sage graph on such a host into a
  // refused model. Without them pv8_available() is false and prepare()
  // makes the int8 QK alone, whatever the config asks.
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
  // `pv8`: with the int8 P*V's operands too (sage::Config::pv8) -- V
  // as uint8 [kv_heads, k_tokens, d], its scale and mean, and Q's scale
  // per token rather than per block.
  static std::size_t scratch_bytes(int heads, int kv_heads, int q_tokens,
                                   int k_tokens, int d, int bq, int bk,
                                   bool pv8 = false);

  // ...and what it will still allocate for ITSELF once `lend_a` and
  // `lend_b` bytes have been lent, which is the figure a caller's
  // memory plan actually wants. Simulates the same greedy carve
  // ensure_scratch_ runs, alignment included, so the two cannot
  // disagree about which buffers fit.
  static std::size_t private_bytes(int heads, int kv_heads, int q_tokens,
                                   int k_tokens, int d, int bq, int bk,
                                   std::size_t lend_a, std::size_t lend_b,
                                   bool pv8 = false);

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

  //
  // `v`, with cfg.pv8: V's operand too, addressed as q and k are, and
  // the int8 P*V's prologue after theirs -- Q then quantized per token,
  // V's mean and per-channel scale over the sequence, V as uint8 + 128.
  // Without either, the int8 QK alone, as before.
  bool prepare(metal_compute::ComputeEncoder& e, const Operand& q,
               const Operand& k, int heads, int kv_heads, int q_tokens,
               int k_tokens, int d, int bq, int bk, const sage::Config& cfg,
               std::string* err, const Operand* v = nullptr);

  // THE SAME PROLOGUE IN TWO HALVES, for a caller that attends BANDS of
  // queries against one set of keys (MiniMax-H3's banded forward):
  // prepare_keys() once -- K's mean and int8, and V's with cfg.pv8 --
  // with the scratch sized for `q_tokens`, the widest band (reserve() it
  // first, so a narrower last band reuses it); then prepare_queries() for
  // each band of up to that many rows before the band's dispatch, and
  // bind() as after prepare(). The same dispatches prepare() issues, so a
  // band starting on a query block quantizes to exactly what prepare()
  // makes of those rows: Q's scales are per block (or, with pv8, per
  // token) and K's and V's the whole sequence's.
  bool prepare_keys(metal_compute::ComputeEncoder& e, const Operand& k,
                    int heads, int kv_heads, int q_tokens, int k_tokens,
                    int d, int bq, int bk, const sage::Config& cfg,
                    std::string* err, const Operand* v = nullptr);
  bool prepare_queries(metal_compute::ComputeEncoder& e, const Operand& q,
                       int q_tokens, std::string* err);

  // Bind the four operands at buffers 15..18 onto a dispatch whose
  // function was built with constant 306 true. Call AFTER set_function
  // and the caller's own binds, like any other bind.
  //
  // Binding these onto a function built WITHOUT 306 is harmless (the
  // slots are unused); the reverse is not, and is why prepare() refuses
  // rather than warns when its scratch is not there.
  //
  // After a pv8 prepare(), V's three at 19..21 as well: the function
  // must then have been built with pv8_constants(fc, true).
  void bind(metal_compute::ComputeEncoder& e) const;
  // ...for a dispatch whose query rows start at `q_row0` of those
  // prepare() quantized (the caller's Q and O bound from there too, and
  // its params counting blocks from there): Q's int8 and per-token scale
  // bound from that row. K's and V's are whole. FALSE, binding nothing,
  // when the last prepare() did not make per-token scales (a pv8 one)
  // and q_row0 is not 0: a per-block scale is indexed by the dispatch's
  // own block numbers, which no longer count from 0, so there is no
  // offset that would be right. The caller must not dispatch then.
  bool bind(metal_compute::ComputeEncoder& e, int q_row0) const;

  // The int8 P*V's function constants (308, 310), on or off,
  // for a caller building the flash function that bind() feeds.
  static void pv8_constants(metal_compute::FunctionConstants& fc, bool on);

  // Whether this host's library carries the int8 P*V's kernels (see
  // load()). The flash kernel's constants 308 and 310 arrived with them,
  // so their presence is also the answer for the kernel.
  bool pv8_available() const noexcept { return _pv8_kernels; }

  // Whether a prepare() under `cfg` will make the int8 P*V's operands:
  // asked for AND available. BUILD THE FLASH FUNCTION FROM THIS, not
  // from cfg.pv8 -- a function with 308 and 310 set over a prologue
  // that made none reads Q's scales in the wrong layout and V's from
  // unbound buffers. (Given V; prepare() without one makes none.)
  bool pv8_for(const sage::Config& cfg) const noexcept {
    return cfg.pv8 && _pv8_kernels;
  }

  // A CAPACITY, for a caller whose attention runs at several row counts
  // in one forward -- FLUX.3 runs each token mode's stack over its own
  // rows and then the joint stack over all of them. Without one, every
  // change of row count rebuilds the scratch (a private free and
  // allocation, two or three a forward there); with one, a prepare()
  // whose geometry it COVERS -- the same heads, kv heads, head dim and
  // tiles, no more query or key tokens, no P*V it was built without --
  // reuses it. Its operands are written and read at the CALL's own
  // geometry either way, so the answer is the same bit for bit.
  //
  // Built EXACTLY at this geometry -- it shrinks as well as grows -- unless
  // the scratch already is that, so call it with the forward's largest
  // attention before its first prepare(), every forward: a size that
  // stays put builds once. A prepare() it does not cover rebuilds at the
  // call's own geometry and drops the reservation. `pv8` builds the int8
  // P*V's buffers too (only where pv8_available()). False, with `err`,
  // when the geometry is refused or the allocation fails.
  bool reserve(int heads, int kv_heads, int q_tokens, int k_tokens, int d,
               int bq, int bk, bool pv8, std::string* err);

  // How many times the scratch has been BUILT (allocated, carved), for a
  // test or a caller that wants to know a forward did not rebuild it.
  int scratch_builds() const noexcept { return _builds; }

  // Build the scratch with the int8 P*V's buffers from the FIRST
  // prepare(), whether or not that call uses them -- for a caller that
  // turns pv8 on part way through a run (MiniMax-H3 keeps its early
  // steps off). Without it, the first pv8 call rebuilds the scratch;
  // either way a scratch that has them serves a call that does not, so
  // a run toggles it once at most. Set it before the first prepare().
  void reserve_pv8(bool on) noexcept { _reserve_pv8 = on; }

  // Whether the last prepare() made the int8 P*V's operands too.
  bool pv8_ready() const noexcept { return _ready && _pv8_ready; }

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
  // `exact` (reserve) builds at this geometry unless it is exactly
  // that; otherwise (prepare) a reservation that covers the call is
  // kept, and anything else is rebuilt at the call's geometry.
  bool ensure_scratch_(int heads, int kv_heads, int q_tokens, int k_tokens,
                       int d, int bq, int bk, bool pv8, bool exact,
                       std::string* err);
  // The geometry checks prepare() and reserve() share; "" when valid.
  static std::string check_geometry_(int heads, int kv_heads, int q_tokens,
                                     int k_tokens, int d, int bq, int bk);
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
  // The int8 P*V's: Q per token, V's absmax over the sequence, V8 + 128.
  metal_compute::ComputeFunction _fn_qtok, _fn_vam, _fn_var, _fn_vq8;

  // [H, D] fp32 key mean, and the per-chunk partials it reduces from.
  metal_compute::SharedBuffer _km, _kmp;
  // [H, qL, D] / [H, kL, D] int8, tightly packed.
  metal_compute::SharedBuffer _q8, _k8;
  // [H, NQ] / [H, NK] fp32, one scale per attention tile ([H, qL] for
  // Q with pv8: per token).
  metal_compute::SharedBuffer _qs, _ks;
  // pv8: V as uint8 [KVH, kL, D]; its scale and mean [KVH, D] fp32; the
  // per-chunk partials both reduce from.
  metal_compute::SharedBuffer _v8, _vs, _vm, _vmp;

  metal_compute::SharedBuffer _arena_a, _arena_b;
  std::size_t                 _arena_used[2] = {0, 0};
  bool                        _res_added = false;
  // Tallied by the carve: what did NOT fit in the lend.
  std::size_t                 _own = 0;

  int  _heads = 0, _kvh = 0, _qt = 0, _kt = 0, _d = 0, _bq = 0, _bk = 0;
  bool _pv8 = false;         // the scratch's geometry has pv8's buffers
  bool _pv8_kernels = false; // the library carries pv8's four kernels
  bool _reserve_pv8 = false; // build pv8's buffers from the first call
  bool _reserved = false;    // the scratch is a reserve()d capacity
  int  _builds = 0;          // scratch builds, see scratch_builds()
  bool _ready = false;
  bool _pv8_ready = false;
  // prepare_keys() has run (and with pv8's V): prepare_queries() may.
  bool _keys_ready = false;
  bool _keys_pv8 = false;
};

}  // namespace genai
}  // namespace vpipe

#endif  // GENERATIVE_MODELS_SHARED_METAL_SAGE_ATTENTION_H
