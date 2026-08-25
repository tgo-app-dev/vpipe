#ifndef GENERATIVE_MODELS_BOOGU_METAL_BOOGU_TRANSFORMER_H
#define GENERATIVE_MODELS_BOOGU_METAL_BOOGU_TRANSFORMER_H

#include "generative-models/shared/block-residency.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/shared/wired-pool.h"
#include "generative-models/shared/dit-block-progress.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MetalLlamaWeights;   // fwd
class WeightSet;           // generative-models/weight-set.h
class I8GemmContext;       // fwd (shared/i8-gemm.h)

// Boogu-Image denoiser (BooguImageTransformer2DModel): a 10B flow-matching DiT
// in the NextDiT / Lumina-Image-2.0 lineage, run in bf16 on the metal-compute
// backend. Unlike the FLUX.2 / Qwen-Image MMDiTs it has FIVE kinds of block and
// a two-phase stream topology:
//
//   * three REFINER stacks (num_refiner_layers each) that pre-condition the
//     streams separately before they ever meet -- `context_refiner` over the
//     instruction tokens (UNMODULATED: no timestep enters the text stream),
//     `noise_refiner` over the target latent tokens and `ref_image_refiner`
//     over each reference image's tokens (both modulated). Each reference is
//     refined ON ITS OWN (the reference batches them), so a reference attends
//     only itself here.
//   * num_double_stream_layers DUAL-STREAM blocks: image and instruction keep
//     separate q/k/v + output projections and separate FFs, but attend JOINTLY
//     over [instruct; ref; noise] -- and the image stream ALSO runs a second,
//     image-only self-attention in the same block (three adaLN sets on the
//     image side, two on the instruction side).
//   * the remaining (num_layers - num_double_stream_layers) SINGLE-STREAM
//     blocks over the fused [instruct; ref; noise] sequence.
//
// Normalization is RMSNorm throughout (Lumina "RMSNormZero": modulated blocks
// scale the normed input by (1+scale) and gate the residual by TANH(gate) --
// note the tanh, which FLUX/Qwen-Image gates do not have). The final layer is
// LuminaLayerNormContinuous: an affine-free LayerNorm scaled by (1+linear_1(
// SiLU(temb))) then projected by linear_2 to patch^2 * out_channels.
//
// Conditioning enters PRE-COMPUTED: the Qwen3-VL mllm's LAST hidden state
// [instr_seq, 4096] (image-grounded when editing -- the reference picture rides
// the VL tower) is RMSNormed + projected by caption_embedder to the DiT width.
// The timestep is a cos-first sinusoid of t*timestep_scale through a 2-layer
// TimestepEmbedding to min(hidden,1024); every modulation Linear reads THAT
// (not the full hidden), so the modulation vectors are cheap.
//
// RoPE is 3-axis (axes_dim [40,40,40], sum = head_dim 120) applied as
// INTERLEAVED complex pairs (Lumina `apply_rotary_emb(use_real=False)`), which
// is exactly rope_pair_table_ftab_f16's convention. Position ids: text token l
// -> (l,l,l); each reference image -> (band, row, col) with the band advanced
// by max(ref_rows, ref_cols) per reference; the target -> (band, row, col) at
// the final band. All the per-stage tables (context / noise / per-ref /
// combined-image / joint) are contiguous SLICES of one joint table, so the
// forward builds it once and indexes by row offset.
//
// Attention is GQA (28 q-heads over 7 kv-heads on the 10B) at head_dim 120 --
// NOT 128, which no flash kernel supports natively. The default path therefore
// ZERO-PADS each head to 128 and runs the steel bd128 flash attention (exact:
// zero q/k dims add nothing to a dot product and zero v dims contribute nothing
// to the output -- see kPadAttn); the scalar sdpa_full_f16, which handles an
// arbitrary D and is GQA-aware, is the fallback and what
// VPIPE_BOOGU_NO_STEEL_ATTN selects for A/B.
class MetalBooguTransformer {
 public:
  struct Config {
    int   hidden       = 3360;   // num_attention_heads * head_dim
    int   n_heads      = 28;     // num_attention_heads
    int   n_kv_heads   = 7;      // num_kv_heads (GQA)
    int   head_dim     = 120;    // hidden / n_heads == sum(axes_dim_rope)
    int   in_channels  = 16;     // latent channels (x_embedder in = p*p*this)
    int   patch        = 2;      // patch_size
    int   instruct_dim = 4096;   // instruction_feature_configs.instruct_feat_dim
    int   n_double     = 8;      // num_double_stream_layers
    int   n_single     = 32;     // num_layers - num_double_stream_layers
    int   n_refiner    = 2;      // num_refiner_layers (x3 stacks)
    int   multiple_of  = 256;    // LuminaFeedForward rounding
    int   temb_dim     = 1024;   // min(hidden, 1024) -- modulation input width
    int   freq_dim     = 256;    // frequency_embedding_size
    float norm_eps     = 1e-5f;
    float rope_theta   = 10000.0f;
    int   axes_dim[3]  = {40, 40, 40};
    float timestep_scale = 1000.0f;
    int   max_ref_images = 5;    // image_index_embedding rows
    // Derived from the checkpoint at load() (0 = "read from the weight
    // shapes"): the SwiGLU inner width (4*hidden rounded up to multiple_of --
    // 13568 on the 10B) and the patch-space output width (patch^2 *
    // out_channels = 64).
    int   ff_inner     = 0;
    int   out_channels = 0;
    // Accelerated mode (LOSSY, opt-in): dynamic-int8 GEMMs for the big block
    // matmuls (see shared/i8-gemm.h). Env VPIPE_I8_GEMM=0|1 overrides.
    bool  i8_gemm = false;

    int x_in() const { return patch * patch * in_channels; }
  };

  // `stream_blocks` (memory-bounded, for small boxes): the double + single
  // blocks are NOT preloaded; each is re-read from the retained source mmap
  // just before use in forward_dit and freed after, so peak block RAM is ~one
  // block instead of the whole DiT. The three refiner stacks (2 blocks each)
  // and the embedders stay resident -- they are <2% of the weights. ~2-3x
  // slower per step (weights re-read per forward).
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
  static std::unique_ptr<MetalBooguTransformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false);

  // Prefer this overload: the set is the manager's shared,
  // reference-counted view of the checkpoint, so two pipelines running
  // this model share its weights instead of loading a copy each. The dir
  // overload opens a PRIVATE set (tests, and callers with no session).
  static std::unique_ptr<MetalBooguTransformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false);

  ~MetalBooguTransformer();


  // ---- the resident set, grown by measuring -----------------------------
  //
  // See generative-models/shared/block-residency.h. A streamed block is
  // KEPT after it has been used, as free memory allows, so the re-read
  // this model pays every forward shrinks toward nothing on a box with
  // room -- and is given back the moment the pages it kept are found
  // outside RAM. It replaces the pinned prefix, which was a fraction of
  // TOTAL ram decided before the run and blind to the machine.
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

  // Cooperative stop polled per block in streaming mode, so a pipeline stop is
  // honored within ~one block instead of a whole forward.
  void set_stream_stop(std::function<bool()> stop) {
    _stream_stop = std::move(stop);
  }

  // Per-block progress (see DitBlockProgressFn): fired as each transformer
  // block is entered, so a caller can show movement INSIDE one denoise
  // step -- which at high resolution is seconds of otherwise-silent work.
  void set_block_progress(DitBlockProgressFn fn) {
    _block_progress = std::move(fn);
  }

  // A reference-image conditioning input (Boogu edit). `latents` is the
  // VAE-encoded reference, patch-packed token-major [seq, patch^2*in_channels]
  // f16 -- the same layout as the generated `latents`. Reference tokens go
  // through their OWN embedder (ref_image_patch_embedder, not x_embedder), get
  // image_index_embedding[i] added, are refined by ref_image_refiner, and then
  // sit BEFORE the target tokens in the image stream. They take part in the
  // joint + image self attention but no velocity is returned for them.
  // seq = (grid_h/patch) * (grid_w/patch) is the token count; grid_h/grid_w are
  // the LATENT dims (H/8, W/8), so both must be even.
  struct RefImage {
    metal_compute::SharedBuffer latents;   // [seq, patch^2*in_ch] bf16
    int seq    = 0;
    int grid_h = 0;   // latent rows  (pre-patch)
    int grid_w = 0;   // latent cols  (pre-patch)
  };

  // One denoiser step. EVERY tensor here is BF16 -- the conditioning arrives
  // that way from the encoder's final norm (f16 cannot hold its attention-sink
  // outliers) and the latents/velocity match so nothing on the path narrows.
  // `instruct` is the [instr_seq, instruct_dim] mllm last
  // hidden state; `latents` the patch-packed [img_seq, patch^2*in_channels]
  // noisy target; grid_h/grid_w its LATENT dims (H/8, W/8); `timestep` the
  // flow-matching time in Boogu's ASCENDING convention (0 = pure noise, 1 =
  // clean -- the reference scheduler steps x += (t_next - t) * v).
  // Returns the [img_seq, patch^2*out_channels] predicted velocity for the
  // TARGET tokens only. Empty on failure.
  metal_compute::SharedBuffer
  forward_dit(const metal_compute::SharedBuffer& instruct, int instr_seq,
              const metal_compute::SharedBuffer& latents, int img_seq,
              int grid_h, int grid_w, float timestep,
              const std::vector<RefImage>& refs = {});

  const Config& config() const { return _cfg; }

  // ---- On-device AWQ calibration ------------------------------------------
  // Accumulate per-input-channel |activation| abs-max at each quantizable
  // Linear input while enabled, tapped inside forward_dit (guarded; no effect
  // when off). Like Krea-2 / FLUX.2 the DiT distribution is timestep-dependent,
  // so the collector runs forward_dit over prompts x sigmas and the abs-max
  // accumulates. calib_stats() returns {group -> flat [rows*dim]} feeding the
  // clip-only AWQ fold (see metal-boogu-calibration + the quantizer's boogu
  // dit_act). Groups: ref_{attn,ffn}_{ctx,noise,ref}, dbl_{jattn,sattn,ffn}_
  // {img,txt}, sgl_{attn,ffn}, emb_{x,ref,ctx,proj}.
  void calib_begin();
  void calib_end() { _calib_on = false; }
  bool calibrating() const { return _calib_on; }
  std::map<std::string, std::vector<float>> calib_stats() const;

  // ---- Bench hook: switch the matmul2d route without reloading -------------
  // A cross-process end-to-end A/B of these two cannot resolve them: the GPU's
  // clock is gated by the SoC power budget, and the run-to-run spread at
  // seq 4104 (~5-15%) is larger than either effect -- measured both signs of
  // the same comparison on the same build. Two loaded DiTs will not fit on a
  // 16 GB box either (6.5 GB of 4-bit weights each), so an ALTERNATING
  // in-process A/B needs to reroute a live model. `splitk` and `tn2` start
  // from the load-time env knobs (VPIPE_BOOGU_NO_SPLITK / _NO_TN2), which
  // remain the shipping control; this only lets a bench flip them per forward.
  // No effect on a build/GPU where the kernels never loaded.
  void set_gemm_route(bool splitk, bool tn2)
  {
    _splitk_on = splitk && _use_splitk;
    _tn2_on = tn2 && _fn_dense_mma_tn2.valid();
  }
  bool gemm_route_splitk() const { return _splitk_on; }
  bool gemm_route_tn2() const { return _tn2_on; }

 private:
  MetalBooguTransformer() = default;

  // A Linear weight: dense bf16 [N,K] or an affine-quantized triple
  // (codes/scales/qbias). `gemm_` dispatches accordingly. Loaded by load_qw_.
  struct QWeight {
    metal_compute::SharedBuffer w;                     // bf16 dense [N,K]
    metal_compute::SharedBuffer codes, scales, qbias;  // affine quant
    bool quantized = false;
    int  bits = 0;                                     // 4 | 8 (per-weight)
    int  n = 0, k = 0;                                 // out, in dims
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

  // BooguImageTransformerBlock -- the refiner + single-stream shape:
  //   modulated:   x += tanh(gate_msa) * norm2(attn(rms1(x)*(1+scale_msa)))
  //                x += tanh(gate_mlp) * ffn_norm2(ff(ffn_norm1(x)*(1+scale_mlp)))
  //   unmodulated: the same with the gates/scales dropped (context_refiner).
  // `mod` is norm1.linear ([4*hidden, temb_dim] + bias); n1 is norm1.norm.weight
  // when modulated and norm1.weight when not.
  struct Block {
    QWeight q, k, v, o;                     // attn.to_q/k/v/to_out.0
    metal_compute::SharedBuffer qn, kn;     // per-head q/k RMSNorm [head_dim]
    QWeight ff_gate, ff_up, ff_down;        // feed_forward.linear_1/3/2
    QWeight ff_gu;                          // fused: interleaved [2*inner, H]
    QWeight mod;                            // norm1.linear (modulated only)
    metal_compute::SharedBuffer mod_b;      // norm1.linear.bias
    metal_compute::SharedBuffer n1, n2, fn1, fn2;   // RMSNorm weights [hidden]
    bool modulated = false;
  };

  // BooguImageDoubleStreamTransformerBlock. The joint attention's q/k/v and
  // per-stream output projections live on the PROCESSOR (img_instruct_attn.
  // processor.*) -- the diffusers Attention's own to_q/k/v are deleted -- and
  // are followed by a SHARED to_out.0 over the re-joined sequence. The image
  // stream additionally runs img_self_attn over the image tokens alone.
  struct DoubleBlock {
    // joint attention
    QWeight jq_i, jk_i, jv_i;               // processor.img_to_q/k/v
    QWeight jq_t, jk_t, jv_t;               // processor.instruct_to_q/k/v
    QWeight jout_i, jout_t;                 // processor.img_out / instruct_out
    QWeight jo;                             // img_instruct_attn.to_out.0
    metal_compute::SharedBuffer jqn, jkn;   // img_instruct_attn.norm_q/k
    // image self attention
    QWeight sq, sk, sv, so;                 // img_self_attn.to_q/k/v/to_out.0
    metal_compute::SharedBuffer sqn, skn;   // img_self_attn.norm_q/k
    // feed-forwards (img + instruct)
    QWeight iff_gate, iff_up, iff_down, iff_gu;
    QWeight tff_gate, tff_up, tff_down, tff_gu;
    // modulation: img_norm1/2/3 + instruct_norm1/2 (.linear [4H, temb] + bias,
    // .norm.weight [H]).
    QWeight mi1, mi2, mi3, mt1, mt2;
    metal_compute::SharedBuffer mi1_b, mi2_b, mi3_b, mt1_b, mt2_b;
    metal_compute::SharedBuffer ni1, ni2, ni3, nt1, nt2;
    // post-op RMSNorms.
    metal_compute::SharedBuffer i_attn_n, i_self_n, i_ffn1, i_ffn2;
    metal_compute::SharedBuffer t_attn_n, t_ffn1, t_ffn2;
  };

  // Whether a load is RETAINED for the model's life or read once and
  // thrown away. It is an explicit argument, not a mode flag, because
  // the same three loaders serve both the preloaded/pinned blocks and
  // the per-forward streamed ones -- and caching a streamed block would
  // silently undo streaming, putting the whole DiT back in RAM on the
  // box least able to hold it.
  enum class Retain { Cached, Streamed };

  QWeight load_qw_(WeightSet& ws, const std::string& name, Retain r);
  // bf16 view of a checkpoint tensor (f32/f16/bf16 source). Cached ones
  // go through the weight set so a second model over this checkpoint
  // shares them; streamed ones are rebuilt per forward and retained by
  // nobody.
  metal_compute::SharedBuffer
  bf16_(WeightSet& ws, const std::string& nm, Retain r);
  // Build the INTERLEAVED [2*inner, K] gate|up weight (row 2g = gate_g,
  // 2g+1 = up_g) from the separate linear_1 / linear_3 weights, so the
  // fused-SwiGLU GEMM epilogue reads even col = gate, odd col = up. Returns an
  // empty QWeight when the two sides disagree (or either is missing).
  QWeight fuse_gu_(WeightSet& ws, const std::string& key,
                   const QWeight& gate, const QWeight& up, Retain r);
  bool load_block_(WeightSet& ws, const std::string& pre,
                   Block& b, bool modulated, Retain r);
  bool load_double_(WeightSet& ws, const std::string& pre,
                    DoubleBlock& b, Retain r);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  int _quant_bits = 0;             // 0 dense; 4|8 quantized
  int _quant_group = 64;

  // Embedders, refiners, final layer.
  QWeight _x_embed, _ref_embed;                    // + biases below
  metal_compute::SharedBuffer _x_embed_b, _ref_embed_b;
  metal_compute::SharedBuffer _img_index;          // image_index_embedding [5,H]
  QWeight _t_emb1, _t_emb2;                        // timestep_embedder linear_1/2
  metal_compute::SharedBuffer _t_emb1_b, _t_emb2_b;
  metal_compute::SharedBuffer _cap_norm;           // caption_embedder.0 (RMSNorm)
  QWeight _cap_lin;                                // caption_embedder.1 (+bias)
  metal_compute::SharedBuffer _cap_lin_b;
  QWeight _out_lin1, _out_lin2;                    // norm_out.linear_1/2 (+bias)
  metal_compute::SharedBuffer _out_lin1_b, _out_lin2_b;
  std::vector<Block> _ctx_refiner, _noise_refiner, _ref_refiner;
  std::vector<DoubleBlock> _double;   // promoted blocks (all when preloaded)
  std::vector<Block> _single;         // promoted blocks (all when preloaded)

  // Streaming mode: blocks not yet promoted by residency (both stacks
  // single) are loaded on demand from the retained source mmap per forward.
  bool _stream_blocks = false;
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
  std::size_t wire_block_(DoubleBlock& b, bool on);
  std::size_t wire_block_(Block& b, bool on);
  std::size_t wire_fixed_(bool on);
  std::size_t wire_retry_slack_() const;

  static std::size_t qw_bytes_(const QWeight& w);
  static std::size_t double_bytes_(const DoubleBlock& b);
  static std::size_t single_bytes_(const Block& b);
  void resident_pages_(std::size_t* examined, std::size_t* incore,
                       std::size_t* paged_out = nullptr) const;
  // SINGLE blocks first, then double. Both stacks stream, and the single
  // stack is the tail of the forward -- giving back what runs LAST keeps
  // what remains a contiguous prefix of the pass.
  std::size_t evict_tail_block_();

  // ---- the streamed blocks' reusable destinations --------------------
  //
  // See shared/block-slots.h. TWO sets, because this model has two block
  // shapes and one destination can only serve blocks it fits.
  //
  // NOTE the tensor lists this model carries -- the byte counters, the
  // wired-pool walks, and the each_*_tensor_ enumerations below -- must
  // agree about what a block is made of. They are checked against each
  // other by the streaming equality test rather than by the type system.
  void each_single_tensor_(
      int L, Block& b,
      const BlockSlots<Block>::TensorFn& fn) const;
  void each_double_tensor_(
      int L, DoubleBlock& b,
      const BlockSlots<DoubleBlock>::TensorFn& fn) const;
  bool clone_single_(const Block& src, Block& dst, bool copy) const;
  bool clone_double_(const DoubleBlock& src, DoubleBlock& dst,
                     bool copy) const;
  metal_compute::SharedBuffer rebuild_one_(
      const std::string& nm, Placement how);
  // Re-interleave gate|up into a destination that ALREADY exists. The
  // fused weight has no checkpoint name -- it is a transform of two
  // tensors -- so a refill cannot address it and this runs after one.
  bool weave_into_(const QWeight& gate, const QWeight& up, QWeight& dst) const;
  void configure_slots_();

  BlockSlots<Block>       _single_slots;
  BlockSlots<DoubleBlock> _double_slots;
  // Zero-copy mmap of the quantized weight tensors as read-only views aliasing
  // the retained source mmap, so the DiT's resident footprint stays reclaimable
  // under memory pressure (a 1024px VAE decode). On by default when preloading;
  // off in streaming mode and via VPIPE_BOOGU_NO_MMAP_WEIGHTS.
  bool _mmap_weights = false;
  // The checkpoint, held for this model's whole life -- it owns the mmap
  // the mapped weights alias AND is where the streamed blocks are read
  // from, so streaming is the manager's business now rather than a
  // private mmap this class kept to itself.
  std::shared_ptr<WeightSet> _ws;
  std::function<bool()> _stream_stop;
  DitBlockProgressFn    _block_progress;

  // A [hidden] run of zeros, so an adaLN "scale only" modulation (Lumina
  // RMSNormZero has no shift) can reuse adaln_modulate_f16 with a zero shift.
  metal_compute::SharedBuffer _zero_h;

  // One contiguous token segment of the RoPE build. Text segments tile
  // (l,l,l); image segments share the axis-0 band `t_off` and tile a
  // rows x cols patch grid row-major on axes 1/2.
  struct ImgSeg { int t_off; int rows; int cols; int seq; };

  // Build the [seq, head_dim] cos/sin 3-axis RoPE tables for the joint
  // sequence: `text_seq` text rows first, then the image segments in order
  // (references then the target). Rebuilt per shape; f32 tables (the rotation
  // error is structured and compounds over 46 blocks x N steps).
  void build_rope_tables_(int text_seq, const std::vector<ImgSeg>& segs,
                          metal_compute::SharedBuffer& cos_out,
                          metal_compute::SharedBuffer& sin_out);

  // Libraries + kernel functions.
  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_sdpa,
      _lib_rope, _lib_qmm;
  metal_compute::ComputeFunction _fn_gemm, _fn_gemm_bias, _fn_rms, _fn_swiglu,
      _fn_residual, _fn_transpose, _fn_sdpa, _fn_rope_table,
      _fn_transpose_rope, _fn_layernorm, _fn_adaln, _fn_gated,
      _fn_gated_tanh, _fn_bias_add, _fn_qmm4, _fn_qmm8, _fn_headslice,
      _fn_mulsig, _fn_colabsmax;
  // vec4 twins of the adaLN / tanh-gate / residual passes, taken when the row
  // width is a multiple of 4 (always, for this model's hidden and FF widths).
  // One element per thread leaves those kernels at ~37-54 GB/s where the same
  // bytes through a vec4 2-D grid run at 143-181; the arithmetic is unchanged
  // per element, so the result is bit-identical.
  metal_compute::ComputeFunction _fn_adaln4, _fn_gated_tanh4, _fn_residual4;
  // Padded-head-dim twins backing the steel bd128 attention route (see
  // kPadAttn): transpose+rope / transpose with a zero-extended head width, and
  // the inverse that drops the pad off the attention result.
  metal_compute::ComputeFunction _fn_tr_pad, _fn_tr_unpad, _fn_tr_rope_pad;
  // Larger-tile dense GEMM twins (fewer weight re-reads at the DiT's big
  // M = seq). VPIPE_BOOGU_GEMM_TILE / VPIPE_BOOGU_GEMM_ACC16.
  metal_compute::ComputeFunction _fn_gemm_bm64, _fn_gemm_bm64bn64,
      _fn_gemm_bm64_a16;
  int  _gemm_tile = 0;
  bool _acc16 = false;
  // BM128 affine-qmm twins (g64 only). VPIPE_BOOGU_QMM_TILE.
  metal_compute::ComputeFunction _fn_qmm4_bm128, _fn_qmm8_bm128;
  int _qmm_tile = 0;
  // Fused-SwiGLU FF: one GEMM whose register-local epilogue writes
  // silu(gate)*up from the INTERLEAVED gate|up weight, killing the
  // [seq, 2*inner] intermediate + the slice/swiglu passes.
  metal_compute::ComputeFunction _fn_ff_swiglu, _fn_ff_swiglu_a16,
      _fn_qmm_swiglu4_bm64, _fn_qmm_swiglu8_bm64,
      _fn_qmm_swiglu4_bm64_a16, _fn_qmm_swiglu8_bm64_a16;
  bool _fuse_ff = false;
  bool _ff_acc16 = false;

  // Steel flash attention at head_dim 128 with the 120-dim heads ZERO-PADDED
  // (padding q/k adds 0 to every dot product and padding v contributes 0 to the
  // output, so the result is exact). Costs ~7% extra attention bandwidth and
  // one extra scratch pair, but replaces the scalar O(seq^2) sdpa -- which is
  // ~80% of a 1024px step. Requires the padded transpose/rope + unpad kernels;
  // falls back to sdpa_full_f16 otherwise or under VPIPE_BOOGU_NO_STEEL_ATTN.
  static constexpr int kPadAttn = 128;
  metal_compute::ComputeLibrary _lib_attn, _lib_attn_nax;
  bool _steel_attn_ok = false;
  bool _use_attn_nax = false;
  // A steel entry point instantiated at EXACTLY this model's head_dim (see
  // attn_steel.metal's bd120): no zero-pad, no pad/unpad transposes.
  bool _attn_native = false;
  // One specialized function + param block per DISTINCT sequence length, cached
  // ACROSS forwards. The lengths repeat every sampler step, and a specialized
  // ComputeFunction costs a Metal newFunction() call (the PSO behind it is
  // cached, the specialization is not) -- rebuilding them per step is pure
  // serial CPU time before the first dispatch. Each length keeps its own param
  // buffer because a CPU write is not ordered against a later GPU read.
  struct AttnCfg {
    metal_compute::ComputeFunction fn;
    metal_compute::SharedBuffer params;
    unsigned nqb = 0;
  };
  std::map<int, AttnCfg> _attn_cfgs;

  // M5 matrix-core matmul2d for the block/projection GEMMs (mirrors FLUX.2):
  // dense weights feed dense_gemm_mma directly; quantized weights are dequant-
  // expanded into the reusable bf16 scratch _w_deq then run the SAME dense
  // matmul2d. Gated on matrix cores; VPIPE_BOOGU_NO_MMA2 A/B off.
  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& x, std::size_t xe,
                 const QWeight& w, const metal_compute::SharedBuffer& y,
                 std::size_t ye, int M, int N, int K);
  metal_compute::ComputeLibrary _lib_dense_mma, _lib_dequant;
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep,
      _fn_dense_mma_tn2, _fn_dense_mma_splitk, _fn_dequant4, _fn_dequant8;
  std::unique_ptr<I8GemmContext> _i8;
  bool _use_mma2 = false;
  int  _mma_min_m = 64;
  // Split-K for the ff-down GEMM (K = ff_inner = 13568). Boogu's K is not a
  // multiple of the KC=8192 tile Krea-2/FLUX.2 use, so this dispatches the
  // KC=6784 twin (2 planes). Measured 1.14-1.17x over the unsplit 128x256 at
  // every M; the fold is a residual_add, so the result is not bit-identical
  // (rel-L2 ~2.9e-3, one extra bf16 rounding per fold -- the same trade Krea-2
  // already makes). VPIPE_BOOGU_NO_SPLITK opts out.
  bool _use_splitk = false;
  metal_compute::SharedBuffer _splitk;   // [splits, M, N] partial planes
  // Live route (set_gemm_route); initialised from _use_splitk / the tn2 handle.
  bool _splitk_on = false;
  bool _tn2_on = false;
  // The TN=2 tile (a 128x512 N-region per threadgroup, doubling x-reuse) wins
  // only once there is enough work to fill the GPU with HALF as many
  // threadgroups: measured 0.81-0.88x at M=1032, ~1.00-1.05x at M=2271 and
  // 1.08-1.13x at M=4104, and it loses at every M for the narrow k/v shape
  // (N=840). So it gates on BOTH M and N, not on K alone as the inherited
  // Krea-2 rule did. VPIPE_BOOGU_NO_TN2 opts out.
  //
  // kTn2MinN is 8192, which admits ONLY ff-gate/up (N=13568) -- deliberately
  // NOT q/o (N=3360), even though the microbench says TN=2 is 1.078x there at
  // M=4104. In the model it made the projection section 8-16% SLOWER, and the
  // reason is a bias in the bench, not noise: mma_tile_sweep runs the same GEMM
  // 8x inside one encoder, so from the 2nd iteration the weight is already
  // resident, and TN=2's trade -- read x once for two N-tiles, read the weight
  // the same number of times -- is worth most exactly when weight traffic is
  // free. The forward pass touches each weight ONCE, cold. The ff-gate/up
  // weight (13568x3360, 91 MB) cannot stay resident across bench iterations
  // either, so that measurement transfers and was confirmed end-to-end (single-
  // stream FF 3776 -> 3352 ms at seq 4104); q/o's 22.6 MB can, so it did not.
  // Treat any tile A/B on a small weight from that bench as an upper bound.
  static constexpr int kTn2MinM = 2048;
  static constexpr int kTn2MinN = 8192;
  // The K-chunk the split-K kernel above is instantiated at; ff_inner must be
  // exactly 2 of these for the split to fire (13568 = 2 * 6784).
  static constexpr int kSplitKC = 6784;
  metal_compute::SharedBuffer _w_deq;   // reusable [N,K] bf16 dequant scratch

  // AWQ calibration: one bf16 [rows*dim] abs-max accumulator per group (live
  // while _calib_on), tapped by col_absmax_f16 at each quantizable Linear input.
  bool _calib_on = false;
  std::map<std::string, metal_compute::SharedBuffer> _calib_acc;
};

}  // namespace genai
}  // namespace vpipe

#endif
