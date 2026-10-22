#ifndef VPIPE_GENERATIVE_MODELS_QWEN3_METAL_QWEN_RAW_LAYERS_H
#define VPIPE_GENERATIVE_MODELS_QWEN3_METAL_QWEN_RAW_LAYERS_H

// PRIVATE to MetalQwenModel (metal-qwen-model.cc and
// metal-qwen-raw-layers.cc): the raw-source layers behind
// Config::load_quant_bits and Config::stream_decode.
//
// A backbone whose layers are BUILT FROM THE RAW CHECKPOINT by this model
// rather than read as they sit: dense full attention, or (streaming bf16
// only) a gated-DeltaNet HYBRID, whose two kinds of layer share one slot
// pair through a pool (HybLayout). Two features share the mechanism
// because they share its hard part:
//
//   IN-MEMORY QUANTIZATION (load_quant_bits = 8). An unquantized
//   checkpoint's seven projections become affine w8 group-64 on the GPU
//   as each layer is built, in exactly the layout an 8-bit pack loads
//   into (q|k|v fused by rows, gate|up interleaved), so the forward is
//   the affine path unchanged. An autoregressive model re-reads every
//   weight per token, so holding it at ~53% of bf16 is what keeps an 8B
//   backbone inside a 16 GB box at all.
//
//   STREAMING A PACK. An affine w8 group-64 pack (model-quantize's, or an
//   MLX 8-bit conversion) streams through the same slots: its layers are
//   already the products, short of the two fusions the forward reads --
//   q|k|v by rows, gate|up interleaved -- which the build makes by row
//   copies (copy_rows_u32). Half the bytes per streamed read of a raw
//   bf16 source, and a slot a little over half the size.
//
//   DECODE-CAPABLE STREAMING (stream_decode). A layer the box cannot
//   keep is read per use into one of two slots (shared/block-slots.h),
//   refilled in place by pread with the next streamed layer's read issued
//   under this one's GPU work, and its products are rebuilt in the SAME
//   command buffer that runs it. The resident set grows into free RAM by
//   measurement (shared/block-residency.h) and is wired into the pool
//   (shared/wired-pool.h) -- the block-streaming DiTs' contract
//   (docs/MODEL-MEMORY.md mechanisms 5-7), applied to a decoder.
//
// THE RULE THAT MAKES THEM ONE: a layer's products come from exactly one
// function, encode_build(), whether the layer is built once at load or
// rebuilt for a streamed use. Streamed and resident output are therefore
// byte-identical by construction rather than by keeping two builders in
// step.

#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/shared/block-residency.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/shared/wired-pool.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace vpipe::genai {

// The raw tensors of one dense full-attention Qwen3 layer, in slot order.
enum RawTensor : int {
  kRawInLn, kRawPostLn, kRawQNorm, kRawKNorm,
  kRawQ, kRawK, kRawV, kRawO, kRawGate, kRawUp, kRawDown,
  kRawCount
};

// The tensors of one streamed PACK layer, in slot order: the four norms
// first, as in RawTensor, then a (codes, scales, biases) triple per
// projection. o and down are bound as they sit; q, k, v and gate, up are
// copied into the fused products.
enum PackTensor : int {
  kPackInLn, kPackPostLn, kPackQNorm, kPackKNorm,
  kPackQ,    kPackK = kPackQ + 3, kPackV = kPackK + 3, kPackO = kPackV + 3,
  kPackGate = kPackO + 3, kPackUp = kPackGate + 3, kPackDown = kPackUp + 3,
  kPackCount = kPackDown + 3
};

// The tensors of one layer of a HYBRID stack (Qwen3.5: gated-DeltaNet
// layers between gated full-attention ones), by role. Every layer has the
// first five; a full-attention layer adds the self_attn six, a GDN layer
// the linear_attn nine.
enum HybRole : int {
  kHInLn, kHPostLn, kHGate, kHUp, kHDown,
  kHQ, kHK, kHV, kHO, kHQNorm, kHKNorm,
  kHQkv, kHZ, kHA, kHB, kHOut, kHConv, kHALog, kHDtBias, kHGNorm,
  kHRoleCount
};

// A hybrid slot's layout, off the checkpoint. Its two kinds of layer hold
// different tensors, so a slot cannot be laid out by position as a dense
// one is -- and one laid out for either kind would be rebuilt every other
// read (the slots alternate; one in four layers is full attention). So a
// slot is a POOL that serves both: a PROJECTION takes a buffer of exactly
// its byte size, the k-th of that size in its layer the k-th such buffer,
// so the pool holds the most of each size any layer needs (the MLP's
// three, a qkv-or-q, two square ones, ...); a small tensor (a norm, the
// conv, the recurrence's A_log / dt_bias) has a buffer of its own role. A
// refill places bytes in buffers of the right size whichever kind it is.
// The fused projection (q|k|v, or qkv|z|a|b) is built into one arena of
// the wider kind's size.
struct MetalQwenModel::HybLayout {
  std::vector<std::size_t> pool;      // one slot's buffers, bytes each
  // Per layer: each role's pool buffer, -1 where the layer has none.
  std::vector<std::array<int, kHRoleCount>> pos;
  std::vector<char>        full;      // per layer: full attention
  std::vector<std::size_t> fused;     // per layer: its fused projection
  std::vector<std::size_t> kept;      // per layer: what promotion keeps
  std::size_t arena = 0;              // the widest fused projection
  std::size_t raw = 0;                // the stack's bytes on disk
  std::size_t slot() const
  {
    std::size_t n = arena;
    for (std::size_t b : pool) { n += b; }
    return n;
  }
};

// One layer's raw checkpoint tensors and the products the forward binds.
// A resident layer keeps only `prod` (whatever of `src` it aliases stays
// alive through it); a streaming slot keeps both. `src` is in RawTensor
// order for a raw checkpoint, in PackTensor order for a pack, and a
// HybLayout pool for a hybrid -- which also carries the layer it holds
// now and the arena its fused projection is built into.
struct MetalQwenModel::RawLayer {
  std::vector<metal_compute::SharedBuffer> src;   // kRawCount, compute dtype
  Layer prod;
  int   layer = -1;                    // hybrid: the layer `src` holds
  metal_compute::SharedBuffer arena;   // hybrid: the fused projection
};

struct MetalQwenModel::RawStream {
  RawStream(MetalQwenModel* model, metal_compute::MetalCompute* c,
            bool quantize, bool stream, bool pack = false);
  ~RawStream();
  RawStream(const RawStream&) = delete;
  RawStream& operator=(const RawStream&) = delete;

  // Load the build kernel, size the layers, open the pool window and (when
  // streaming) configure the slots. False when the kernel is missing --
  // a model that cannot build its layers must not load.
  bool init();

  // Build layer L resident: read its raw tensors, build the products on
  // the GPU, keep the products.
  bool build_resident(int L);
  // Wire what a PRELOADED model holds -- the checkpoint's cached trunk
  // first, then every layer -- stopping at the pool's first refusal.
  void wire_preloaded();

  // ---- the streamed forward ---------------------------------------------
  // Once per forward (one prefill chunk, one decode step): measure the
  // resident set, shed what the box took back, wire the trunk, and give
  // the slot pair back once every layer it runs is resident and a whole
  // forward has run without it. `end` is
  // one past the last layer the forward runs -- a taps forward stops at
  // its deepest tap -- and is where the prefetch wraps to the next
  // forward's first streamed layer; -1 is the whole stack.
  void begin_forward(int end = -1);
  bool streamed(int L) const
  {
    return streaming && L >= 0 && L < (int)held.size() && !held[(size_t)L];
  }
  // Layer L streams. Commits what `enc` holds so far (the GPU starts on it
  // while the read lands), acquires L's slot, opens a fresh encoder on
  // `st`, encodes the build and binds the products as _layers[L].
  bool enter(int L, metal_compute::CommandStream& st,
             metal_compute::ComputeEncoder& enc);
  // Finish the layer enter() bound, if any: commit it, prefetch the next
  // streamed layer under its GPU work, wait, keep it if the box can hold
  // it, unbind otherwise, and reopen `enc`. Call at the top of every layer
  // and once after the loop, so a break cannot skip it.
  bool finish(metal_compute::CommandStream& st,
              metal_compute::ComputeEncoder& enc);

  // ---- residency + accounting --------------------------------------------
  // The slot pair given back when every layer in [0, the last forward's
  // end) is held. No forward may be running. Returns the bytes.
  std::size_t release_if_saturated();
  std::size_t evict_tail();
  std::size_t resident_bytes() const;
  int resident_count() const;
  void shutdown();

  MetalQwenModel* const              m;
  metal_compute::MetalCompute* const mc;
  const bool                         quant;
  const bool                         streaming;
  const bool                         pack;   // the source is a w8 pack
  bool                               hyb = false;   // a hybrid stack

  int H = 0, qd = 0, qdo = 0, kd = 0, F = 0, D = 0;
  // One built layer: what promotion KEEPS -- the widest kind's, where a
  // hybrid's two differ (kept_(L) is the layer's own).
  std::size_t kept_bytes = 0;
  std::size_t slot_bytes = 0;   // one slot: sources + their products
  bool        failed = false;   // a streamed read failed this forward

 private:
  std::string prefix_(int L) const;
  // The checkpoint names of layer L's sources, in `src` order.
  std::vector<std::string> source_names_(int L) const;
  // One source, read uncached in the form its slot holds: a pack's u32
  // codes as they sit, everything else in the compute dtype.
  metal_compute::SharedBuffer read_source_(const std::string& name) const;
  bool read_sources_(int L, RawLayer& r);
  bool alloc_products_(RawLayer& r);
  void alias_products_(RawLayer& r) const;
  Layer take_products_(RawLayer& r) const;
  static Layer alias_layer_(const Layer& a);
  bool copy_products_(const RawLayer& s, RawLayer& d);
  bool alloc_like_(const RawLayer& s, RawLayer& d);
  void encode_build_(metal_compute::ComputeEncoder& enc,
                     const RawLayer& r) const;
  int next_streamed_(int L) const;
  std::size_t kept_(int L) const;
  // Every buffer a raw-built layer can bind -- the dense fields (bf16, w8,
  // a pack's) and the gated-DeltaNet ones -- for each walk that wires,
  // counts, copies or aliases a layer, so none of them misses a kind.
  static constexpr int kFields = 23;
  static std::array<metal_compute::SharedBuffer*, kFields> fields_(Layer& y);
  // A layer's distinct allocations, summed once: `arena` first (a fused
  // projection is a window onto it), then the fields, then `src`.
  static std::size_t layer_bytes_(
      const Layer& y,
      const std::vector<metal_compute::SharedBuffer>* src = nullptr,
      const metal_compute::SharedBuffer* arena = nullptr);
  // A hybrid role's field; null for a part of the fused projection.
  static metal_compute::SharedBuffer* hyb_field_(Layer& y, int role);
  // ---- a hybrid stack --------------------------------------------------
  // Whether the slot holds `role` in a form the build derives rather than
  // as it sits: a zero-centred norm (1 + w folded), A_log / dt_bias (f32
  // for the recurrence) -- rebuilt per refill, as the resident load
  // builds them, so the bytes are the same.
  bool hyb_derived_(int role) const;
  metal_compute::SharedBuffer hyb_read_(const std::string& name) const;
  bool hyb_fill_(int L, RawLayer& r, bool whole_pool);
  std::size_t wire_layer_(Layer& y, bool on);
  std::size_t wire_trunk_(bool on);
  void resident_pages_(std::size_t* examined, std::size_t* incore,
                       std::size_t* paged_out) const;

  metal_compute::ComputeLibrary  _lib_quant;
  metal_compute::ComputeFunction _fn_quant;   // affine_quant_rows_w8g64
  metal_compute::ComputeFunction _fn_copy;    // copy_f16: bf16 q|k|v fuse
  metal_compute::ComputeFunction _fn_rows;    // copy_rows_u32: pack fuses

  HybLayout            _hl;           // hybrid only
  BlockSlots<RawLayer> _slots;
  BlockResidency       _resid;
  WiredPool            _wire;
  std::vector<char>    held;          // streaming: L promoted resident
  int                  _pending = -1; // streamed L bound, not finished
  int                  _end = 0;      // one past this forward's last layer
  bool                 _used_slots = true;   // this forward acquired one
  bool                 _shut = false;

  friend class MetalQwenModel;
};

}  // namespace vpipe::genai

#endif
