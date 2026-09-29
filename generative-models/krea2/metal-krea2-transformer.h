#ifndef GENERATIVE_MODELS_KREA2_METAL_KREA2_TRANSFORMER_H
#define GENERATIVE_MODELS_KREA2_METAL_KREA2_TRANSFORMER_H

#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/sage-attention.h"
#include "generative-models/shared/metal-sol-attention.h"
#include "generative-models/shared/sol-attention.h"
#include "generative-models/shared/block-residency.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/shared/wired-pool.h"
#include "generative-models/shared/ane-ffn.h"
#include "generative-models/shared/dit-block-progress.h"
#include "generative-models/shared/fp8-expand.h"
#include "generative-models/shared/fp8-layout.h"
#include "generative-models/shared/fp8.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "generative-models/shared/runtime-lora.h"

namespace vpipe {
namespace genai {

class MetalLlamaWeights;   // fwd
class WeightSet;           // generative-models/weight-set.h
class I8GemmContext;       // fwd (shared/i8-gemm.h)


// Krea-2-Turbo denoiser (Krea2Transformer2DModel): a single-stream MMDiT
// flow-matching transformer, run in f16 on the metal-compute backend.
//
// Text conditioning enters as a stack of hidden states tapped from 12 layers
// of the Qwen3-VL text encoder (see the M2 encoder tap). A small text-fusion
// tower (2 layerwise blocks over the layer axis -> a Linear(12->1) projector
// -> 2 refiner blocks over the token axis) collapses that stack into one
// sequence of text features; txt_in projects it to the transformer width.
// (The image path -- img_in, the 28 adaLN/RoPE/gated transformer blocks and
// the final layer -- is added incrementally; M3a implements the text tower.)
class MetalKrea2Transformer {
 public:
  struct Config {
    int   hidden          = 6144;   // attention_head_dim * num_attention_heads
    int   n_heads         = 48;
    int   n_kv_heads      = 12;     // GQA
    int   head_dim        = 128;
    int   ffn             = 16384;
    int   n_layers        = 28;
    int   in_channels     = 64;     // 16 latent ch * patch 2 * patch 2
    int   timestep_dim    = 256;
    // Text-fusion tower.
    int   text_hidden     = 2560;
    int   text_heads      = 20;
    int   text_kv_heads   = 20;     // no GQA in the fusion blocks
    int   text_ffn        = 6912;
    int   n_text_layers   = 12;     // tapped encoder layers
    int   n_layerwise     = 2;
    int   n_refiner       = 2;
    float norm_eps        = 1e-5f;
    float rope_theta      = 1000.0f;
    int   rope_t          = 32;     // axes_dims_rope
    int   rope_h          = 48;
    int   rope_w          = 48;
    // Accelerated mode (LOSSY, opt-in): dynamic-int8 GEMMs for the big
    // block matmuls (see shared/i8-gemm.h). VPIPE_I8_GEMM overrides.
    bool  i8_gemm         = false;
    // SageAttention: the QK^T product in int8. Independent of i8_gemm
    // above -- that one chooses how the block's GEMMs are computed and
    // this one how the attention between them is.
    sage::Config sage;
    // Sol-Attn (LOSSY), from generate-image's `sol_attn*` keys. Off by
    // default. It routes whole KEY BLOCKS: the ones whose centroid beats
    // a per-query-block threshold are attended exactly on the flash
    // kernel, the rest fold in as their centroid.
    //
    // Krea-2's joint [text; image] attention is what it needs -- one
    // self-attention whose keys ARE its queries, at head_dim 128 -- and
    // the text prefix becomes the exact SINK, because a centroid over
    // prompt tokens is a prompt half-read. GQA (48 query heads over 12
    // kv) is why MetalSolAttention grew a kv_heads argument: the key
    // summaries are per KV head.
    //
    // Independent of `sage` beside it: Sol decides WHICH key blocks are
    // attended, Sage how the ones that are get multiplied.
    sol::Config sol;

    // ---- the ANE tier (LOSSY, opt-in) -------------------------------
    //
    // The block feed-forward on the Apple Neural Engine, which on an M4
    // is the only unused compute on the die: the GPU's f16 GEMM already
    // sits at 84-92% of roofline and its integer MAD is a QUARTER of the
    // f16 rate, so there is no int8 lever there. MEASURED ~14 TOPS on a
    // tiled DiT-shaped FFN against the GPU's achieved 6.6-7.3, and wider
    // on a base M4 (same 16-core ANE, roughly half the GPU).
    //
    // DENSE CHECKPOINTS ONLY. The ANE is fp16-only -- int8 there buys
    // footprint under its ~8-10 MB weight buffer, not arithmetic -- so a
    // quantized block is dequantized into the module's fp16 inputs
    // per block instead. On a
    // bf16 checkpoint the bake converts to fp16, which is safe HERE
    // because the FFN is self-contained: only its output `o` re-enters
    // the bf16 residual stream, and that addition stays on the GPU.
    //
    // The ANE runs ONE shared module whose weights are runtime inputs:
    // each block's gate/up/down are converted to fp16, in the checkpoint's
    // own [out,in] layout, into IOSurface slots at the start of the block,
    // on the ANE worker, while the GPU runs that block's attention. So the
    // tier's memory is fixed (ane_runtime_bytes()) and it compiles once
    // per shape. `ane_templates` is unused by it, kept for config
    // compatibility.
    //
    // `ane_rows` is the ANE's share of the FFN's rows, the GPU taking
    // the rest -- rows are independent in a feed-forward, so the split
    // is exact. 0 means auto-balance from the measured times.
    // `ane_layers` caps how many blocks use the tier; 0 means all.
    // The session is how the tier reaches the shared CoreML model
    // manager; without one there is no ANE and the tier stays off.
    const SessionContextIntf* session = nullptr;
    std::string ane_templates;
    float ane_rows   = 0.0f;
    int   ane_layers = 0;
    // PROTOTYPE: the attention's q, k, v and gate projections split onto
    // the ANE too, stacked into ONE matmul module (hidden -> q+k+v+gate
    // wide) with its own auto share and on/off controller. Needs `session`
    // like the rest of the tier. VPIPE_KREA2_ANE_QKV=1 also turns it on.
    bool  ane_qkv    = false;

    int   text_head_dim() const { return text_hidden / text_heads; }  // 128
  };

  // `stream_blocks` (memory-bounded AWQ calibration on small boxes): keep the
  // 28 main transformer_blocks OFF the wired heap -- forward_dit loads each from
  // the retained source mmap just before use and frees it after, so peak block
  // memory is ~1 block (~0.85 GB bf16) instead of all 28 (~24 GB). The small
  // parts (text-fusion tower, conditioning, final layer) are still preloaded.
  //
  // THE PINNED PREFIX IS RETIRED. BlockResidency replaces it.
  //
  // It was a fraction of TOTAL ram decided before the run, so it could
  // not see the machine it landed on: not another process, not this
  // graph's peers, not the moment a peer let go. On a tight box that is
  // where it did most harm -- plan_streaming's own note records a 16 GB
  // run taken from no swap at all to 11 GB resident with 3 GB of swap
  // moving continuously, by a prefix sized exactly that way.
  //
  // What replaces it measures: admission against the live budget, a
  // probe read off the room actually free, doubling while the box stays
  // healthy, and a shed the moment the pages kept are found outside RAM.
  // A runtime adapter. RUNTIME rather than fused because fusing writes
  // A*B into the base weights, and this base is QUANTIZED -- so the sum
  // is rounded back to 4/8-bit affine, which is exactly the precision a
  // small correction cannot spare. Runtime also makes the strength a
  // knob rather than a rebuild.
  //
  // Named at LOAD because it changes how the weights are BUILT: an
  // adapted `ff.gate` or `ff.up` cannot take the fused-SwiGLU kernel,
  // whose epilogue writes silu(gate)*up straight out of the accumulator
  // and leaves nowhere for a pre-activation delta to land. See the
  // `_fuse_ff` gate in the loader.
  struct LoraSpec {
    std::string path;          // one .safetensors of lora_A/B pairs
    float       scale = 1.0f;  // 1.0 = as trained; 0 = exactly off
  };

  // How many adapters may ride the DiT at once. Two, because that is
  // what a real request wants: an identity or few-step adapter beside a
  // style one. They stay SEPARATE slots rather than one merged set of
  // factors -- concatenating them on the rank axis is exact arithmetic
  // and free in the forward, but it folds both strengths into A and so
  // turns the live knob back into a rebuild. See lora::Stack.
  static constexpr int kMaxLoraSlots = genai::lora::Stack::kMax;

  static std::unique_ptr<MetalKrea2Transformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  // Prefer this overload: the set is the manager's shared,
  // reference-counted view of the checkpoint, so two pipelines running
  // this model share its weights instead of loading a copy each. The dir
  // overload opens a PRIVATE set (tests, and callers with no session).
  static std::unique_ptr<MetalKrea2Transformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  // How many projections carry a runtime adapter, summed over the slots
  // that bound (0 = none attached).
  int lora_modules() const;
  // The same for ONE slot, so a caller can report which adapter took.
  int lora_modules(int slot) const;
  // Slots that bound. Indices below this are live; the rest are not.
  int lora_slots() const { return static_cast<int>(_lora_slots.size()); }

  // The adapter's strength, per FORWARD. Live because it can be: the
  // factors are the adapter's and the strength is the request's, so it
  // rides the GEMM as a constant rather than being folded into A.
  // Turning it costs nothing and needs no reload. 0 skips both GEMMs, so
  // "off" is exactly off.
  // Per SLOT, because two adapters are two requests. An index past what
  // bound is ignored.
  void set_lora_scale(int slot, float s);
  float lora_scale(int slot) const;

  // Every projection a runtime adapter can name, and the base dims its
  // factors have to fit, for a model of this shape. Public and static
  // so the names can be checked without loading a checkpoint: a module
  // name that matches nothing binds nothing, which at runtime is
  // indistinguishable from an adapter that had no effect. Derived from
  // the same table bind_lora_ walks, so the two cannot drift.
  struct LoraModule {
    std::string name;          // `<name>.weight` is the base tensor
    int         n = 0, k = 0;  // out, in
  };
  static std::vector<LoraModule> lora_module_list(const Config& cfg);

  // The file-name convention map this family binds through, and the
  // reason it is a function rather than a constant at each use: the
  // BINDER and the FUSED-FF GATE must ask about the same spellings. If
  // the gate did not know the ai-toolkit names, an adapter published
  // that way would bind its `ff.gate` / `ff.up` -- which the file calls
  // `mlp.gate` / `mlp.up` -- and then meet a fused-SwiGLU kernel with
  // nowhere to put the delta, dropping it on the two biggest
  // projections in every block and reporting nothing.
  static genai::lora::Adapter::Rename lora_rename();

  ~MetalKrea2Transformer();   // out-of-line: _ws is a fwd-declared type


  // Cooperative-stop hook for the (long, disk-bound) streaming-blocks forward:
  // forward_dit polls this before each main block and bails early (returns
  // empty) when it returns true. Used by the AWQ calibration collector to honor
  // a pipeline stop within ~one block. No effect on the preloaded path.
  void set_stream_stop(std::function<bool()> stop) { _stream_stop = std::move(stop); }

  // ---- adaptive block residency (streaming mode) -----------------------
  // Grow the resident block set into free RAM so later steps re-read less
  // from disk. Policy and the reasoning behind it live in
  // shared/block-residency.h -- in particular why this is a fixed
  // resident set and not an LRU cache.
  //
  // `bytes` is what must stay free for the rest of the forward. 0 (the
  // default) disables growth, which is exactly today's pure-streaming
  // behaviour -- so this is opt-in per caller rather than a silent
  // change to every graph.
  void set_residency_reserve(std::size_t bytes) { _resid.set_reserve(bytes); }
  // Size the residency rates for the schedule that is about to run. The
  // defaults are tuned for ~30 steps and are wrong for a 5-step turbo
  // one in the same direction -- see BlockResidency::set_schedule. Call
  // AFTER set_residency_reserve (the probe is sized against what is left
  // once the reserve is taken) and before the first forward.
  void set_residency_schedule(int steps);
  std::size_t resident_block_bytes() const { return _resid.bytes(); }
  int resident_block_count() const { return _resid.count(); }
  std::size_t release_resident_blocks(std::size_t bytes);
  // Free the highest resident block; its bytes, or 0 when there is
  // nothing left.
  std::size_t evict_tail_block_();

  // Pages of the resident set examined, and how many are still in RAM.
  void resident_pages_(std::size_t* examined, std::size_t* incore) const;

  // The per-forward activation scratch actually held right now. MEASURED,
  // not estimated: DitScratch is a persistent struct, so this sums what
  // was allocated rather than re-deriving it from the config. That
  // matters -- the equivalent estimator on MiniMax-H3 under-reported by
  // 212 MB until a test compared it against reality, and a reserve that
  // under-reports is how residency growth eats the room the rest of the
  // forward needs.
  std::size_t scratch_resident_bytes() const;

  // What Sol-Attn holds for a `seq`-token forward AFTER the two buffers
  // the forward lends it, which is what a residency reserve has to leave
  // clear. 0 when the tier is off or every layer is dense.
  //
  // MOST OF IT IS NOT AN ALLOCATION AT ALL: Sol's scratch lives for one
  // call, so it carves from memory this forward is not using between the
  // projection and the output. Only what does not fit is its own. See
  // MetalSolAttention::private_bytes.
  std::size_t sol_scratch_bytes(int seq) const;

  // Bytes the ANE feed-forward tier holds, for the whole model, for a
  // sequence of `seq` tokens (image plus text; <= 0 books one chunk).
  //
  // ONE shared runtime-weight module serves every block, so this is fixed
  // however many blocks use it. Three terms, each MEASURED:
  //   slots    the IOSurface weight inputs, one block's fp16 weights W:
  //            576 MB of process footprint for Krea-2, not wired;
  //   staging  CoreML's wired copy of those inputs plus the graph's
  //            activations at the chunk's rows, W + rows*(3*ffn +
  //            2*hidden)*2 -- fitted to 836 / 953 / 1078 MB against 836 /
  //            938 / 1059 measured at 2048 / 3072 / 4096 rows;
  //   host     the fp16 input/output buffers for the ANE's rows, at most
  //            every whole chunk short of the sequence.
  //
  // Static, and takes a CONFIG rather than reading _cfg, because the
  // caller that needs it most is the PLAN, which asks before anything is
  // loaded -- it fills a default Config with the dims and the tiers it is
  // about to ask for.
  //
  // `c.ane_qkv` adds the q|k|v|gate matmul module beside the feed-
  // forward's. TWO modules, booked together: this took a config instead
  // of loose dims precisely because the old (hidden, ffn, seq) form could
  // not see the second tier, so a graph setting it allocated a module no
  // ledger knew about.
  static std::size_t ane_runtime_bytes(const Config& c, int seq) noexcept;
  // The sequence a plan should book for a width x height image (either <= 0
  // books 1024x1024): its tokens plus the longest text the model conditions
  // on.
  static int ane_plan_seq(int width, int height) noexcept;
  // Rows per ANE predict: VPIPE_KREA2_ANE_CHUNK or the default. One reading
  // for the plan and the model, so the two cannot book different shapes.
  static int ane_chunk_rows() noexcept;
  // Whether the tier tried to arm, and whether it did: a plan that booked
  // the module can release the booking when it did not.
  bool ane_attempted() const noexcept { return _ane_tried; }
  bool ane_armed() const noexcept { return _ane != nullptr; }
  // The qkv tier arms separately -- it has its own module and its own
  // eligibility -- so a caller revising what the plan booked has to ask
  // about both.
  bool ane_qkv_armed() const noexcept { return _ane_qkv != nullptr; }

  // M3a: run the text-fusion tower + txt_in on the (text_seq, n_text_layers,
  // text_hidden) f16 encoder-tap stack -> the (text_seq, hidden) fused text
  // conditioning fed to the DiT's joint sequence. `ehs` is laid out
  // [token][layer][text_hidden] (row-major), matching M2's forward_embeddings_
  // taps output reshaped to (seq, 12, 2560). Empty on failure.
  metal_compute::SharedBuffer
  forward_text(const metal_compute::SharedBuffer& ehs, int text_seq);

  // A reference-image conditioning input (Qwen-Image-Edit-2509 multi-reference).
  // `latents` is a pre-VAE-encoded, packed reference latent already in the DiT
  // input space, token-major [seq, in_channels] f16 (the same layout as the
  // generated `latents`). Reference tokens are concatenated AFTER the generated
  // image tokens, share the image-stream modulation and take part in the full
  // joint attention, but carry a per-reference RoPE FRAME (axis-0) offset so
  // each reference occupies a distinct position band (Qwen _compute_video_freqs
  // frame index: generated = 0, ref i = i+1). The model predicts velocity only
  // for the generated tokens; reference tokens are dropped. seq = grid_h*grid_w.
  struct RefImage {
    metal_compute::SharedBuffer latents;   // [seq, in_channels] f16, token-major
    int seq    = 0;
    int grid_h = 0;
    int grid_w = 0;
  };

  // M3b/M3c: one denoiser step. `fused_text` is the (text_seq, hidden) output
  // of forward_text; `latents` the packed (img_seq, in_channels) noisy image;
  // `timestep` the flow-matching time (= sigma = scheduler t / 1000); the
  // (grid_h, grid_w) latent grid drives the image RoPE coordinates. Returns
  // the (img_seq, in_channels) predicted velocity. When `stop_after_block >= 0`
  // it instead returns the joint (text_seq+img_seq, hidden) hidden state after
  // that 0-indexed block (staged verification against d_blk*). Empty on fail.
  // `refs` are optional reference images (Qwen-Image-Edit multi-reference); each
  // is embedded by the same img_in, appended to the image-token stream with its
  // own RoPE frame offset, and attended jointly. Empty (the default) = plain
  // text-to-image. NOTE: reference conditioning is a LEARNED capability -- it
  // only steers generation on a reference-trained (Qwen-Image-Edit-class)
  // checkpoint; on a plain text-to-image distill the tokens run but are inert.
  metal_compute::SharedBuffer
  forward_dit(const metal_compute::SharedBuffer& fused_text, int text_seq,
              const metal_compute::SharedBuffer& latents, int img_seq,
              int grid_h, int grid_w, float timestep,
              int stop_after_block = -1,
              const std::vector<RefImage>& refs = {});

  const Config& config() const { return _cfg; }

  // Per-block progress (see DitBlockProgressFn): fired as each transformer
  // block is entered, so a caller can show movement INSIDE one denoise
  // step -- which at high resolution is seconds of otherwise-silent work.
  void set_block_progress(DitBlockProgressFn fn) {
    _block_progress = std::move(fn);
  }

  // Drop the per-forward scratch: the reusable DitScratch buffers (activations
  // + RoPE), the dequant/split-K scratches, and the i8 accel scratches. All
  // are lazily rebuilt on the next forward_dit (ensure_dit_scratch_ keys on
  // shape; _w_deq/_splitk/i8 re-grow on demand), so this is a pure memory
  // reclaim -- ~1-2 GB at 1024px -- for a memory-bounded box where the idle
  // DiT would otherwise crowd out a large downstream VAE decode. Call only
  // when no forward is in flight (the caller has read back the latent).
  void release_forward_scratch();

  // ---- On-device AWQ calibration ------------------------------------------
  // Accumulate per-input-channel |activation| abs-max at each main-block Linear
  // input while enabled, tapped inside the verified forward_dit (guarded; no
  // effect when off). Unlike an LM, the DiT's distribution depends on the
  // denoising timestep -- so the collector calls forward_dit over prompts x the
  // turbo sigmas and the abs-max accumulates across all of them (col_absmax is
  // read+write). Four taps per block: qkv/gate input (adaLN norm1 out), to_out.0
  // input (gated attn out), ff gate/up input (adaLN norm2 out), ff down input
  // (SwiGLU out). Feeds the DiT AWQ fold. calib_begin zeroes the accumulators.
  void calib_begin();
  void calib_end() { _calib_on = false; }
  bool calibrating() const { return _calib_on; }
  // [n_layers][channels] abs-max (hidden for qkv/o/gateup, ffn for down).
  std::vector<std::vector<float>> calib_qkv() const;
  std::vector<std::vector<float>> calib_o() const;
  std::vector<std::vector<float>> calib_gateup() const;
  std::vector<std::vector<float>> calib_down() const;

 private:
  MetalKrea2Transformer() = default;

  // A Linear weight: either a dense f16 matrix (`w`, [N,K]) or an MLX-affine
  // quantized triple (`codes` U32 packed [N, K*bits/32] + `scales`/`qbias` F16
  // [N, K/group]). `gemm_` dispatches dense_gemm_t or affine_qmm_steel
  // accordingly. Populated by load_qw_ (auto-detects `<name>.scales`).
  struct QWeight {
    metal_compute::SharedBuffer w;                   // f16 dense
    metal_compute::SharedBuffer codes, scales, qbias;  // affine quant
    bool quantized = false;
    int  bits = 0;                                   // 4 or 8 (per-weight; mixed)
    // FP8 storage: `codes` holds the checkpoint's own [N, K] bytes,
    // expanded to bf16 on the GPU right before each GEMM (fp8_dense_).
    // Deliberately NOT `quantized`: no group scales, and every path that
    // reads an affine triple must not be handed one. SCALED FP8 adds
    // `f8_scale`, f32 -- one value, or one per row when `f8_per_row` --
    // multiplied in by the same expansion (shared/fp8-layout.h).
    fp8::Format f8 = fp8::Format::kNone;
    metal_compute::SharedBuffer f8_scale;
    bool f8_per_row = false;
    bool is_fp8() const { return f8 != fp8::Format::kNone; }
    bool empty() const {
      return (quantized || is_fp8()) ? codes.empty() : w.empty();
    }
  };

  // A Krea2 attention+SwiGLU block (used by the fusion tower and the main image
  // blocks). Norms are zero-centered in the checkpoint -> +1 folded at load.
  // `n1/n2` pre-norms; `qn/kn` per-head q/k RMSNorm. Projections may be
  // quantized (4/8-bit affine) or dense f16.
  struct Block {
    metal_compute::SharedBuffer n1, n2;         // pre-attn / pre-ff norms (+1)
    QWeight q, k, v, gate;                       // attn projections (no bias)
    metal_compute::SharedBuffer qn, kn;         // per-head q/k norm (+1)
    QWeight o;                                  // attn to_out.0 (no bias)
    QWeight ff_gate, ff_up, ff_down;            // SwiGLU
    // Fused interleaved gate|up [2*ffn, K] (row 2g = gate g, 2g+1 = up g).
    // Built at load for the MAIN blocks of a quantized checkpoint when
    // _fuse_ff (ff_gate/ff_up are then released); empty otherwise (dense
    // checkpoints, the text tower). The steel path runs it through the fused
    // affine_qmm_swiglu epilogue; the M5 mma path dequants it whole and folds
    // the interleaved matmul2d output with swiglu_interleaved.
    QWeight ff_gu;
    metal_compute::SharedBuffer sst;            // adaLN scale_shift_table (main)
  };

  // Whether a load is RETAINED for the model's life or read once and
  // thrown away. Explicit, not a mode flag: the same loaders serve the
  // preloaded/pinned blocks AND the per-forward streamed ones, and
  // caching a streamed block would silently undo streaming, putting the
  // whole DiT back in RAM on the box least able to hold it.
  enum class Retain { Cached, Streamed };

  // `keep_split` suppresses the release of ff_gate/ff_up after the fused
  // weight is built. A resident block does not need them; a REFILLABLE
  // one does, because the fuse is rebuilt from them after every read.
  bool load_block_(WeightSet& ws, const std::string& pre,
                   Block& b, bool main_block, Retain r,
                   bool keep_split = false);
  // Load a Linear weight: quantized (affine triple) when `<name>.scales` is
  // present, else dense f16.
  QWeight load_qw_(WeightSet& ws, const std::string& name, Retain r);
  // ---- the streamed blocks' reusable destinations --------------------
  //
  // See shared/block-slots.h. NOTE the tensor lists this model carries --
  // block_bytes_, wire_block_ and each_block_tensor_ -- must agree about
  // what a block is made of. They are checked against each other by the
  // streaming equality test rather than by the type system.
  void each_block_tensor_(
      int L, Block& b, const BlockSlots<Block>::TensorFn& fn) const;
  bool clone_block_(const Block& src, Block& dst, bool copy) const;
  bool weave_into_(const QWeight& gate, const QWeight& up,
                   QWeight& dst) const;
  metal_compute::SharedBuffer rebuild_one_(const std::string& nm,
                                           Placement how);
  void configure_slots_();

  BlockSlots<Block> _slots;
  // bf16 view of a checkpoint tensor. Cached ones go through the weight
  // set so a second model over this checkpoint shares them; streamed
  // ones are rebuilt per forward and retained by nobody. `norm` selects
  // the (1 + w) fold the RMSNorm weights need.
  metal_compute::SharedBuffer
  elt_(WeightSet& ws, const std::string& nm, Retain r, bool norm = false);

  // M5 matrix-core GEMM fast path for one Linear: y[M,N] (element offset ye) =
  // xin[M,K] @ dequant(w)[N,K]^T on the hardware matmul2d units. Dense weights
  // feed the tiled kernel directly; affine-quant weights are first expanded into
  // the reusable f16 scratch _w_deq (affine_dequant) -- the same dequant-once ->
  // dense_gemm_mma shape the LM prefill uses. Returns true when it ran; false
  // (not matrix-core capable / M below the tile-amortize threshold / degenerate
  // N) leaves y untouched so the caller runs its steel dense_gemm_t / qmm path.
  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& xin, const QWeight& w,
                 const metal_compute::SharedBuffer& y, std::size_t ye, int M,
                 int N, int K);

  // A projection's FP8 scale as an f32 buffer (one value, or one per
  // row), read fresh -- tiny, and for a streamed block it has to follow
  // the block. Empty on failure.
  metal_compute::SharedBuffer fp8_scale_buf_(const std::string& weight);

  // After a slot refill: does the slot's KIND of each projection (FP8 or
  // not, format, scale shape) match what block L holds? A checkpoint that
  // mixes kinds across blocks (ComfyUI's "mixed ops") would otherwise have
  // one block's bytes read as another's encoding. False sends the slots to
  // their per-block fallback, which builds each block as it is.
  bool fp8_kinds_match_(int L, const Block& b) const;

  // The dense view every GEMM path reads for an FP8 weight: encode its
  // expansion into the reusable _w_deq scratch and return `view` aliasing
  // it, or `w` itself for any other kind. Serial dispatch on one encoder is
  // what makes the one scratch safe across the block's GEMMs -- each
  // expand-then-multiply pair runs before the next expansion writes, the
  // same guarantee gemm_mma_'s affine dequant rests on. Null when the
  // scratch cannot be had (then _fp8_failed is latched and the forward
  // reports failure rather than multiplying by an empty buffer).
  const QWeight* fp8_dense_(metal_compute::ComputeEncoder& enc,
                            const QWeight& w, QWeight& view, int N, int K);

  // Dynamic-int8 accelerated GEMMs (Config::i8_gemm / VPIPE_I8_GEMM);
  // null when off. Tried first in gemm_mma_ for qualifying shapes.
  std::unique_ptr<I8GemmContext> _i8;
  // The int8 QK prologue. Null when the box has no matrix cores or the
  // config did not ask; every use is guarded.
  std::unique_ptr<MetalSageAttention> _sage;
  // Sol-Attn's kernels and scratch. Null unless Config::sol.enabled and
  // the kernels validated; every use is guarded.
  std::unique_ptr<MetalSolAttention> _sol;

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  int _quant_bits = 0;             // 0 = dense; 4 or 8 when quantized
  int _quant_group = 64;

  std::vector<Block> _layerwise;   // text_fusion.layerwise_blocks
  std::vector<Block> _refiner;     // text_fusion.refiner_blocks
  QWeight _projector;                                  // [1, n_text_layers]
  metal_compute::SharedBuffer _txt_norm;               // txt_in.norm (+1)
  QWeight _txt_l1w; metal_compute::SharedBuffer _txt_l1b;   // txt_in.linear_1
  QWeight _txt_l2w; metal_compute::SharedBuffer _txt_l2b;   // txt_in.linear_2

  // ---- DiT image path ----
  QWeight _img_in_w; metal_compute::SharedBuffer _img_in_b;   // img_in (64->H)
  QWeight _te_l1w, _te_l2w;
  metal_compute::SharedBuffer _te_l1b, _te_l2b;              // time_embed
  QWeight _tmp_w; metal_compute::SharedBuffer _tmp_b;        // time_mod_proj
  std::vector<Block> _blocks;                          // 28 transformer_blocks

  // ---- the ANE tier ------------------------------------------------
  //
  // ONE runtime-weight feed-forward module for the whole model, built
  // LAZILY at the first forward because its row count is compiled into
  // its graph and the sequence length is not known until then. Its
  // weights are INPUTS: each block's are staged into its slots at the start
  // of that block, so nothing is ever baked, the tier's memory is fixed,
  // and the graph compiles once per shape.
  //
  // A null module means the tier is off and every block keeps its GPU
  // path, which is what every failure falls back to.
  //
  // The text a plan books beside the image tokens. See ane_plan_seq().
  static constexpr int kAnePlanTextTokens = 512;
  // The chunk: VPIPE_KREA2_ANE_CHUNK or the default. MEASURED on an M4 Pro,
  // 2 blocks, 2048 against 1024 rows: 1024^2 1.417x / 1.414x, 1536^2
  // 1.376x / 1.378x, 2048^2 1.357x / 1.329x. The finer split does reach
  // its balance (100% of the ANE time hidden), but the feed-forward is only
  // about a third of a block, so balancing it better buys ~2% at most --
  // and the ANE runs ~7% fewer rows per ms at 1024. So 2048 stays; the
  // override is for retuning on other hardware.
  std::unique_ptr<AneFeedForward> _ane;
  bool _ane_skip_warned = false;   // one warning for ineligible blocks
  bool _ane_tried       = false;
  // The q/k/v/gate tier (prototype): one stacked matmul module.
  std::unique_ptr<AneFeedForward> _ane_qkv;
  bool _ane_qkv_tried = false;
  bool ane_qkv_setup_(int seq);
  bool ane_qkv_eligible_(int L, const Block& b);
  void ane_qkv_stage_(int L, const Block& b);

  // Whether block L runs its feed-forward on the ANE: the tier is armed,
  // L is under the cap, and every projection is fp16/bf16 or 4/8-bit affine
  // -- gate and up split, or fused into ff_gu.
  bool ane_eligible_(int L, const Block& b);
  // Dispatch block L's weights into the module's slots and return at once,
  // so the conversion runs under the GPU's attention for the same block.
  // Dense weights convert bf16 -> fp16 row for row through a table (~44 ms
  // a Krea-2 block); quantized ones are dequantized into the same fp16
  // slots (4-bit ~21 ms, 8-bit ~50 ms). Takes the LIVE block: under
  // streaming, _blocks[L] is empty.
  void ane_stage_(int L, const Block& b);
  bool ane_setup_(int seq);      // shared one-time part; false == tier off
  // Streaming-blocks mode: _blocks starts EMPTY and holds only what
  // residency has promoted; every other block comes from the weight set (the
  // retained source mmap) on demand in forward_dit and freed after use.
  bool _stream_blocks = false;
  // Zero-copy weights: quantized Linears load as READ-ONLY views into the
  // retained source mmap (newBufferWithBytesNoCopy) rather than owned copies,
  // so the OS reclaims those clean file pages under memory pressure. On by
  // default; disable with VPIPE_KREA2_NO_MMAP_WEIGHTS. Requires retaining
  // the weight set for the model's lifetime (as streaming already does).
  bool _mmap_weights = true;
  // The checkpoint, held for this model's whole life -- it owns the mmap
  // the mapped weights alias AND is where the streamed blocks are read
  // from, so streaming is the manager's business rather than a private
  // mmap this class kept to itself.
  std::shared_ptr<WeightSet> _ws;
  std::function<bool()> _stream_stop;   // polled per block in streaming mode
  BlockResidency _resid;
  // This model's window onto the manager's process-wide wired pool. A
  // kept block is the coldest memory in the process -- read once a step,
  // never written -- so without mlock it is the first thing the
  // compressor takes, and the run spends the schedule admitting, being
  // measured out, shedding and re-admitting the same blocks. See
  // shared/wired-pool.h.
  WiredPool _wire;
  // One block's bytes, from the checkpoint's tensor table rather than
  // from a loaded block -- set_residency_schedule computes it, and the
  // per-forward wire retry is gated on the box having freed at least
  // this much since the refusal. Zero until the schedule is set, which
  // is the honest answer: nothing has been kept yet either.
  std::size_t _wire_block_hint = 0;
  // Wire (mlock) or unwire every buffer of one block / of everything this
  // model holds that is NOT a streamed block, returning the bytes the
  // pool took or gave back.
  std::vector<metal_compute::SharedBuffer*> scratch_buffers_();
  std::size_t wire_block_(Block& b, bool on);
  std::size_t wire_fixed_(bool on);
  std::size_t wire_retry_slack_() const;
  static std::size_t qw_bytes_(const QWeight& w);
  static std::size_t block_bytes_(const Block& b);
  // What promotion KEEPS, as against what the block holds now. A slot
  // holds the split gate/up as well as the fused weight built from
  // them, and drops the split on promotion -- so admission and the
  // wired pool must ask this, not block_bytes_. A grant smaller than
  // the ask is how the pool learns the box refused, and reporting a
  // refusal that never happened clamps the budget at the first
  // promoted block.
  static std::size_t resident_bytes_(const Block& b);
  // What a REFILL reads: the fused gate|up is woven out of the split
  // pair after the read, never read itself. Counting it would report
  // the drive running faster than it is.
  static std::size_t read_bytes_(const Block& b);
  DitBlockProgressFn    _block_progress;
  metal_compute::SharedBuffer _final_sst;              // final_layer (2, hidden)
  metal_compute::SharedBuffer _final_norm;             // final_layer.norm (+1)
  QWeight _final_lw; metal_compute::SharedBuffer _final_lb;   // final_layer.linear

  // One contiguous image-token segment for the RoPE build: all its tokens share
  // the axis-0 (frame/index) coordinate `frame` and tile a grid_h x grid_w
  // latent grid row-major (axis-1 = row, axis-2 = col). Segment 0 is the
  // generated image (frame 0); each reference adds one (frame = index).
  struct ImgSeg { int frame; int grid_h; int grid_w; int seq; };

  // Build the [seq, head_dim] cos/sin RoPE tables (3-axis, adjacent-pair) for
  // the joint sequence (text rows at the origin, then the image segments in
  // order -- the generated grid plus any reference grids, each in its own frame
  // band).
  void build_rope_tables_(int text_seq, const std::vector<ImgSeg>& segs,
                          metal_compute::SharedBuffer& cos_out,
                          metal_compute::SharedBuffer& sin_out);

  // Per-shape persistent scratch for forward_dit: the RoPE tables (which are
  // timestep-INDEPENDENT) + all block scratch. Across the N denoising steps of
  // one image the (seq, grid) shape is identical, so ensure_dit_scratch_ builds
  // this ONCE and reuses it -- the buffers' contents are overwritten each call.
  // Keyed by (seq, grid_h, grid_w, layout-sig); the sig captures the reference
  // segmentation so a changed reference set rebuilds the RoPE (buffers depend
  // only on total seq, but the RoPE depends on the exact split).
  struct DitScratch {
    int seq = -1, gh = -1, gw = -1;               // shape key (-1 = unbuilt)
    std::size_t sig = 0;                           // reference-layout signature
    metal_compute::SharedBuffer te_in, rcos, rsin, joint;
    metal_compute::SharedBuffer te1, temb, tmp, tmod, mod;
    metal_compute::SharedBuffer n1, n2, nm, gate, att, o;
    metal_compute::SharedBuffer q, k, v, qt, kt, vt, atb;
    metal_compute::SharedBuffer g, u, gu, modf;
  };
  DitScratch _dit;
  // `segs` is the full image layout (generated + refs); grid_h/grid_w are the
  // generated grid (the shape-key + buffer-size anchor).
  void ensure_dit_scratch_(int text_seq, int grid_h, int grid_w,
                           const std::vector<ImgSeg>& segs);

  // Libraries + kernel functions.
  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_sdpa,
      _lib_vis, _lib_rope, _lib_qmm;
  metal_compute::ComputeFunction _fn_gemm, _fn_gemm_bias, _fn_rms, _fn_swiglu,
      _fn_mul_sigmoid, _fn_residual, _fn_transpose, _fn_sdpa, _fn_gelu_tanh,
      _fn_rope_table, _fn_adaln, _fn_gated, _fn_qmm4, _fn_qmm8, _fn_bias_add,
      // vec4 twins of adaln/gated (bit-identical; VPIPE_NO_ELT_V4 disables).
      _fn_adaln4, _fn_gated4,
      _fn_colabsmax;
  // Steel MMA flash-attention (attn_steel_h_bd128, head_dim 128, GQA-aware) for
  // the joint-sequence attention. The scalar sdpa_full_f16 is O(seq^2) at <2%
  // FLOP efficiency and DOMINATES at high resolution (~80% of a 1024px step,
  // seq 4096); the steel kernel is the register-resident flash MLX ships, fast
  // on M4 without matrix cores, and handles GQA via gqa_factor. Loaded
  // best-effort; forward_dit falls back to _fn_sdpa when absent or when
  // VPIPE_KREA2_NO_STEEL_ATTN is set (A/B + safety). align_Q/align_K are shape-
  // dependent function constants, so the concrete function is built per-forward.
  metal_compute::ComputeLibrary _lib_attn;
  metal_compute::SharedBuffer _attn_params;   // SteelAttnParams block
  bool _steel_attn_ok = false;
  // M5 matrix-core (matmul2d) NAX steel flash (attn_steel_nax_h_bd128): the
  // vision tower's M5 attention, ported to the DiT's head_dim 128. Preferred
  // over the ALU steel when supports_matrix_cores(); ALU steel is the M4 path
  // + the fallback (VPIPE_KREA2_NO_ATTN_NAX forces it for A/B).
  metal_compute::ComputeLibrary _lib_attn_nax;
  bool _use_attn_nax = false;
  // Larger-BM bf16 GEMM tiles for the block projections (amortize f16 weight
  // bandwidth at M~=seq). _gemm_tile selects: 0 = BM32, 1 = BM64 (default,
  // ~5% faster), 2 = BM64/BN64 (spills, slower); set in load(), overridable via
  // VPIPE_KREA2_GEMM_TILE for A/B.
  metal_compute::ComputeFunction _fn_gemm_bm64, _fn_gemm_bm64bn64;
  int _gemm_tile = 1;
  // Larger-BM steel qmm tile for the quantized block projections (M ~= seq):
  // the same tuning as the dense BM64 above -- each dequantized [BN,BK] weight
  // tile serves 2x the output rows, halving code re-reads + per-FLOP dequant
  // ALU work. g64-only entry points; routed in forward_dit for M >= 128.
  // _qmm_tile: 1 = BM64 (default when the functions are present), 0 = the
  // 32x32 tile, 2 = BM128 (WM=4, 256 threads) for the tall-M block GEMMs at
  // high resolution (M = seq >= 1024). BM128 quarters the fused gate|up code
  // re-reads (that GEMM had stayed at BM=32), but MEASURED ~7-8% SLOWER on
  // M4 Pro at M=4106/1024px across every GEMM section (ff-up 6663->7176,
  // qkv/o/ff-down similarly up; attn -- unchanged kernel -- flat, so not
  // thermal): these GEMMs are compute/occupancy-bound, not weight-bandwidth-
  // bound [[krea2-dit-perf]], so cutting weight re-reads attacks a non-
  // bottleneck while the 128-row/256-thread tile halves the threadgroup count
  // (33 vs 65 M-tiles) and cuts latency-hiding. So BM128 stays opt-in
  // (default BM64); kept only for an M5 matrix-core retest (bigger tiles may
  // feed the matrix units better). Correctness bit-identical to BM64 (rel-L2
  // = 0, w4 + w8; forward_dit_bm128_matches_bm64). VPIPE_KREA2_QMM_TILE=2.
  metal_compute::ComputeFunction _fn_qmm4_bm64, _fn_qmm8_bm64;
  metal_compute::ComputeFunction _fn_qmm4_bm128, _fn_qmm8_bm128;
  int _qmm_tile = 1;
  // FP16-pipe (half-accumulate) twins of the big-GEMM kernels. The default
  // steel BlockMMA widens every fragment to f32 (FP32 pipe); the _acc16
  // variants keep the simdgroup MMA on half8x8 -- worthwhile only where the
  // f16 matrix rate exceeds f32. MEASURED on M4 Pro (gemm_mma.alu_rate
  // probes): f16 10.4 vs f32 10.1 TFLOP/s and mixed slower -- perf-NEUTRAL
  // there, while f16 accumulation costs accuracy (w4 golden 0.0456->0.0506,
  // w8 0.0161->0.0290). So _acc16 stays opt-in (VPIPE_KREA2_ACC16=1),
  // routed only to the tall-M DiT block GEMMs; re-probe before enabling on
  // a new GPU generation.
  metal_compute::ComputeFunction _fn_qmm4_bm64_a16, _fn_qmm8_bm64_a16,
      _fn_qmm_swiglu4_a16, _fn_qmm_swiglu8_a16, _fn_gemm_bm64_a16;
  bool _acc16 = false;
  // Fused SwiGLU FF over the interleaved gate|up weight. Steel path: one
  // affine_qmm_swiglu GEMM (register-local silu(gate)*up in the store
  // epilogue) replaces the two ff GEMMs + the swiglu pass, with no gate/up
  // intermediates. M5 mma path: dequant + one matmul2d over the same
  // interleaved weight -> _dit.gu [seq, 2*ffn], folded to [seq, ffn] by
  // swiglu_interleaved (the LM prefill's fused-FF shape). g64-only kernels.
  // VPIPE_KREA2_NO_FUSED_FF opts out (A/B).
  metal_compute::ComputeFunction _fn_qmm_swiglu4, _fn_qmm_swiglu8,
      _fn_swiglu_inter;
  // BM=128 twins of the fused SwiGLU GEMM (the biggest DiT GEMM, the only
  // quantized tile that had stayed at BM=32). Routed when _qmm_tile == 2 and
  // seq >= 1024.
  metal_compute::ComputeFunction _fn_qmm_swiglu4_bm128, _fn_qmm_swiglu8_bm128;
  bool _fuse_ff = false;

  // ---- runtime LoRA ------------------------------------------------
  // Krea-2's projections are SPLIT (to_q / to_k / to_v / to_out.0,
  // ff.gate / ff.up / ff.down), which is the spelling diffusers-convention
  // adapters ship in -- so modules map one-to-one and nothing has to be
  // de-fused to find them.
  struct BlockLora {
    genai::lora::Factors q, k, v, gate, o, ff_gate, ff_up, ff_down;
    bool any() const
    {
      return !q.empty() || !k.empty() || !v.empty() || !gate.empty() ||
             !o.empty() || !ff_gate.empty() || !ff_up.empty() ||
             !ff_down.empty();
    }
  };
  // Which of the three block containers a row belongs to. All three are
  // the same `Block`, so one BlockLora and one member-pointer type
  // serve them; only the prefix, the dims and the destination vector
  // differ.
  enum class LoraStack { kMain, kLayerwise, kRefiner };
  struct LoraModuleRef {
    std::string name;
    int n = 0, k = 0;
    LoraStack stack = LoraStack::kMain;
    int block = 0;
    genai::lora::Factors BlockLora::*dst = nullptr;
  };
  static std::vector<LoraModuleRef> lora_modules_(const Config& c);
  // ONE attached adapter: its factors for every container it touches,
  // its live strength, and its share of the [rows, rank] scratch.
  //
  // The TEXT-FUSION tower is not an afterthought here: the ai-toolkit
  // convention names those blocks (`txtfusion.{layerwise,refiner}_
  // blocks`) and the identity-edit adapter adapts all three containers,
  // so binding only the main blocks would apply HALF of it -- which is
  // not a cheaper adapter, it is a wrong one that looks like it worked.
  struct LoraSlot {
    std::vector<BlockLora> blocks;      // per main block
    std::vector<BlockLora> layerwise;
    std::vector<BlockLora> refiner;
    int   modules  = 0;
    int   max_rank = 0;
    float scale    = 1.0f;
    // Ranks of the slots BEFORE this one. The scratch is [rows, sum of
    // ranks], so this slot's region starts at `rows * rank_prefix` --
    // which needs the row count, a forward-time number.
    int   rank_prefix = 0;
  };
  bool bind_lora_(const LoraSpec& spec, LoraSlot& slot, std::string* err);
  // The live adapters for one projection of one block, named by the
  // member of BlockLora that holds it.
  genai::lora::Stack lora_at_(LoraStack which, int block,
                              genai::lora::Factors BlockLora::* dst) const;
  std::vector<LoraSlot>  _lora_slots;
  genai::lora::Applier   _lora;
  // Summed max ranks: what the scratch holds for all slots at once, and
  // the stride that turns a rank prefix into an offset.
  int         _lora_rank_total   = 0;
  // Rows the scratch was last sized for. Set with it, read by lora_at_.
  std::size_t _lora_scratch_rows = 0;

  // M5-only matrix-core dense GEMM (matmul2d/NAX) for the DiT projections. Same
  // NAX path the LM prefill + gemma-vision use, gated on supports_matrix_cores()
  // (Apple10+) so M4/older keep the steel dense_gemm_t / affine_qmm_steel
  // (byte-identical). VPIPE_KREA2_NO_MMA2=1 forces steel on M5 (A/B). The
  // quantized DiT dequant-expands each weight into _w_deq before the dense
  // matmul2d; _fn_dequant4/8 are the group-matched affine_dequant kernels
  // (loaded only for a quantized checkpoint). matmul2d rounds bias in a separate
  // bias_add_rows pass (_fn_bias_add), so it is loaded whenever mma is active.
  metal_compute::ComputeLibrary _lib_dense_mma, _lib_dequant;
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep,
      _fn_dense_mma_tn2, _fn_dequant4, _fn_dequant8;
  metal_compute::SharedBuffer _w_deq;   // reusable [N,K] f16 dequant scratch
  // FP8 weights (a checkpoint stored F8_E4M3 / F8_E5M2): the expansion
  // kernels, loaded only when the checkpoint holds one, on EVERY box --
  // unlike the affine dequant above, which only the matrix-core path
  // needs, an FP8 weight has no in-kernel form on either.
  fp8::Expander _fp8x;          // the widening kernels (shared/fp8-expand.h)
  bool _fp8_failed = false;
  // Which checkpoint tensors are FP8 and with what scale; the encoding's
  // own tensors are in `aux`. Scanned once at load, before any read.
  fp8::Layout _f8;
  bool _use_mma2 = false;
  int  _mma_min_m = 64;   // matmul2d only wins once M amortizes the 128-row tile
  // Split-K deep-reduction GEMM for the very deep K (ff-down, K=16384): the
  // single-op full reduction runs ~0.7x the K<=9728 rate, so gemm_mma_ routes
  // K that is a multiple of kSplitKC (>= 2 chunks) through the split-K kernel
  // (grid.z = K/kSplitKC partial planes) + a residual_add fold. _splitk holds
  // the [n_splits, M, N] f16 partials (reused across calls). Gated on M and on
  // VPIPE_KREA2_NO_SPLITK (A/B).
  static constexpr int kSplitKC = 8192;
  metal_compute::ComputeFunction _fn_dense_mma_splitk;
  metal_compute::SharedBuffer _splitk;  // [n_splits, M, N] f16 partial planes
  bool _use_splitk = false;

  // ---- AWQ calibration accumulators (per main block; live while _calib_on) --
  bool _calib_on = false;
  std::vector<metal_compute::SharedBuffer> _cb_qkv, _cb_o, _cb_gu, _cb_dn;
  std::vector<std::vector<float>> read_calib_(
      const std::vector<metal_compute::SharedBuffer>& acc, int dim) const;
};

}  // namespace genai
}  // namespace vpipe

#endif
