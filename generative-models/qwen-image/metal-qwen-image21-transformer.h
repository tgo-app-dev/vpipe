#ifndef GENERATIVE_MODELS_QWEN_IMAGE_METAL_QWEN_IMAGE21_TRANSFORMER_H
#define GENERATIVE_MODELS_QWEN_IMAGE_METAL_QWEN_IMAGE21_TRANSFORMER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/qwen-image/qwen-image21-layout.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/shared/ane-ffn.h"
#include "generative-models/shared/block-residency.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/shared/dit-block-progress.h"
#include "generative-models/shared/dit-gpu-progress.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/sage-attention.h"
#include "generative-models/shared/sol-attention.h"
#include "generative-models/shared/wired-pool.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
class FlexData;
namespace genai {

class WeightSet;
class MetalSageAttention;
class MetalSolAttention;
class I8GemmContext;

// Qwen-Image-2.1's DiT (QwenImage21Transformer2DModel), on metal-compute.
//
// A 7B SINGLE-stream stack over one joint sequence of
// [text | condition images | target image]; see qwen-image21-layout.h for
// how that sequence is built. Despite the family name it shares no code
// with MetalQwenImageTransformer, the 20B dual-stream MMDiT next door --
// only the rotary axes and the head dim happen to match.
//
// Per block, and there is no bias anywhere in the model:
//
//   x += tanh(gate1) * attn(LN(x) * (1 + scale1))
//   x += tanh(gate2) * swiglu(LN(x) * (1 + scale2))
//
// with LN a LayerNorm carrying NO affine parameters. Three things about
// that line are worth stating because each is a silent failure:
//
//  * MODULATION IS NOT PER BLOCK. One shared Linear(hidden, 4*hidden)
//    produces (scale1, gate1, scale2, gate2) and all 32 blocks slice the
//    same four vectors. There is nothing per-block to load.
//  * THE GATES GO THROUGH TANH. Dropping it leaves a model that still
//    denoises, just to the wrong place.
//  * UNDER causal_condition THE MODULATION HAS TWO ROWS: the sampled
//    timestep, and t=0. Text and condition-image rows read the t=0 row,
//    target rows read their own. That is what makes the prefix
//    step-independent, and therefore what makes the KV cache below
//    sound -- the two are one mechanism, not two features.
//
// Attention is BLOCK-CAUSAL: `(q >= kv) or same_image_block`. There is no
// mask buffer. The layout's segments decompose it into one dense
// non-causal dispatch per image block (keys [0, end)) plus one causal
// dispatch for the text runs, which is flux2's klein_kv pattern
// generalized from two groups to N+1.
class MetalQwenImage21Transformer {
 public:
  ~MetalQwenImage21Transformer();

  struct Config {
    int hidden      = 4096;    // num_attention_heads * attention_head_dim
    int n_heads     = 32;
    int head_dim    = 128;
    int n_layers    = 32;
    int in_channels = 64;
    int out_channels = 64;
    int txt_dim     = 4096;    // context_in_dim
    int mlp_ratio   = 3;       // SwiGLU inner = hidden * mlp_ratio
    int time_proj   = 256;     // sinusoidal timestep channels
    float norm_eps  = 1e-6f;
    int rope_theta  = 10000;
    int axes[3]     = {16, 56, 56};   // axes_dims_rope; sum == head_dim
    // Modulate text and condition-image rows from t=0. Required for the
    // KV cache, and the reference refuses the pair without it.
    bool causal_condition = true;

    // Accel tiers, read from accel-settings.h by the caller.
    bool i8_gemm = false;
    sage::Config sage;
    sol::Config  sol;
    // THE FEED-FORWARD ON THE NEURAL ENGINE, beside the GPU. A SwiGLU
    // feed-forward is row-independent, so the ANE takes a contiguous
    // band of rows and the GPU the rest, concurrently and exactly.
    bool ane_ffn   = false;
    int  ane_rows  = 0;    // 0 = balance from the two engines' rates
    int  ane_layers = 0;   // 0 = every block
    // The q/k/v projections on the ANE too, as a SECOND module (one
    // matmul hidden -> 3*hidden, the three weights stacked into its slot)
    // split by rows the same way and sharing the one ANE worker. Once
    // the feed-forward is on the ANE, q/k/v is the largest GPU bucket
    // left: 24.6% of a 1024^2 forward on an M4 Pro.
    bool ane_qkv   = false;

    // Read what can be read from <dir>/config.json. False when the file
    // is absent or names a different class.
    static bool read_dims(const std::string& dir, Config* out);
  };

  // The cross-step prefix cache.
  //
  // Text and condition-image rows modulate from t=0, so their K and V do
  // not change between denoising steps: step one computes the whole
  // joint sequence and keeps them, and every later step runs only the
  // target rows as queries against the cached prefix plus themselves.
  //
  // Stored TOKEN-MAJOR [prefix, hidden] and PRE-RoPE, which is flux2's
  // KvCache shape and is load-bearing: a cached step re-runs the
  // ordinary fused transpose+rope over the whole concatenated K/V, so
  // the spliced band picks up exactly the rotation it originally had
  // instead of needing its own table.
  struct KvCache {
    std::vector<metal_compute::SharedBuffer> k, v;   // one per block
    // The geometry this was filled at. A change of any of them refills
    // rather than reusing, so a caller cannot hand a cache from one
    // picture to another.
    int prefix = 0, joint = 0, target = 0;
    bool populated() const { return !k.empty() && prefix > 0; }
    void clear() { k.clear(); v.clear(); prefix = joint = target = 0; }
  };
  enum class KvMode {
    kOff,       // no cache; run the whole joint sequence every step
    kFill,      // run the whole sequence and KEEP the prefix (step one)
    kCached,    // run the target rows only, against the kept prefix
  };

  struct Request {
    // Packed latents for EVERY image block in order, condition images
    // first and the target last: [layout->image_len, in_channels].
    const metal_compute::SharedBuffer* latents = nullptr;
    // The encoder's hidden states, one row per ENCODER-side token (not
    // per joint row): [enc_len, txt_dim].
    const metal_compute::SharedBuffer* txt = nullptr;
    const qi21::Layout* layout = nullptr;
    float timestep = 0.0f;      // sigma in [0, 1], NOT scaled by 1000
    KvCache* kv = nullptr;
    KvMode kv_mode = KvMode::kOff;
  };

  static std::unique_ptr<MetalQwenImage21Transformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false);

  static std::unique_ptr<MetalQwenImage21Transformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false);

  // One forward. Returns the TARGET block's rows only --
  // [layout->target_len, out_channels] f32-convertible f16 -- because
  // that is the whole of what a denoise step consumes; the reference
  // slices the same rows off its joint output. Empty on failure.
  metal_compute::SharedBuffer forward(const Request& req,
                                      std::string* err = nullptr);

  const Config& config() const { return _cfg; }

  // Streaming + residency, the shared contract every DiT here implements.
  void set_stream_stop(std::function<bool()> stop) {
    _stream_stop = std::move(stop);
  }
  void set_block_progress(DitBlockProgressFn fn) {
    _block_progress = std::move(fn);
  }
  void set_residency_reserve(std::size_t bytes) { _resid.set_reserve(bytes); }
  void set_residency_schedule(int steps);
  std::size_t resident_block_bytes() const { return _resid.bytes(); }
  int resident_block_count() const { return _resid.count(); }
  std::size_t release_resident_blocks(std::size_t bytes);
  std::uint64_t resident_bytes() const;

  bool uses_mma2() const { return _use_mma2; }
  // Which dense tile the non-matrix-core arm took: 0 = BM32, 1 = BM64,
  // 2 = BM64/BN64. Exposed so a test can say that a box without matrix
  // cores still lands on a WIDE tile, rather than only that it produced
  // a right answer slowly.
  int gemm_tile() const { return _gemm_tile; }
  bool uses_attn_nax() const { return _use_attn_nax; }
  bool uses_sage_attn() const { return (bool)_sage; }
  bool uses_i8_gemm() const { return (bool)_i8; }
  bool uses_sol_attn() const { return (bool)_sol; }
  bool ane_armed() const noexcept { return _ane != nullptr; }
  bool ane_attempted() const noexcept { return _ane_tried; }
  bool ane_qkv_armed() const noexcept { return _ane_qkv != nullptr; }

  // What the CoreML module holds, for the resource plan. Its bytes are
  // CoreML's and no other ledger in this process can see them, so the
  // plan claims them in UNITS -- see docs/MODEL-MEMORY.md.
  static std::size_t ane_runtime_bytes(const Config& c, int seq) noexcept;
  // The joint length a plan should book for, from the picture.
  static int ane_plan_seq(int width, int height) noexcept;
  bool streaming() const { return _stream_blocks; }
  int quant_bits() const { return _quant_bits; }

 private:
  MetalQwenImage21Transformer() = default;

  enum class Retain { Cached, Streamed };

  // Dense bf16, or an affine group-quantized w4/w8. Same shape as every
  // other DiT's here; not yet shared because each family's loader reads
  // its own names.
  struct QWeight {
    metal_compute::SharedBuffer w;                    // dense [N,K]
    metal_compute::SharedBuffer codes, scales, qbias; // affine quant
    bool quantized = false;
    int  bits = 0;
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

  // Everything a block owns. Biasless throughout, and with NO modulation
  // of its own -- see the class comment.
  struct Block {
    QWeight qw, kw, vw, ow;                       // attention
    metal_compute::SharedBuffer nq, nk;           // q/k RMSNorm [head_dim]
    QWeight gate_w, proj_w, out_w;                // SwiGLU: gate, up, down
  };

  QWeight load_qw_(WeightSet& ws, const std::string& name, Retain r);
  metal_compute::SharedBuffer to_elt_(WeightSet& ws, const std::string& name,
                                      Retain r);
  bool load_block_(WeightSet& ws, int L, Block& b, Retain r);
  bool load_fixed_(WeightSet& ws);

  // BlockSlots' four callbacks; see shared/block-slots.h.
  // NOTE the tensor lists a block has -- each_block_tensor_ and
  // block_bytes_ -- must agree about what a block is made of. Nothing in
  // the type system checks that; the streamed-equals-preloaded test is
  // what does.
  void each_block_tensor_(int L, Block& b,
                          const BlockSlots<Block>::TensorFn& fn) const;
  bool clone_block_(const Block& src, Block& dst, bool copy) const;
  metal_compute::SharedBuffer rebuild_one_(const std::string& nm,
                                           Placement how);
  void configure_slots_();
  static std::size_t qw_bytes_(const QWeight& w);
  static std::size_t block_bytes_(const Block& b);


  // The sinusoid + its two Linears, on the host: 256 channels of one row
  // is not worth a dispatch, and the rounding below wants to be explicit.
  std::vector<float> time_proj_(float sigma) const;
  // One block's cost, read off the checkpoint.
  std::size_t checkpoint_block_bytes_() const;

  // The ANE tier. Created on the first forward, when the row count is
  // finally known; eligibility is per block, because a block whose
  // weights cannot be staged keeps the GPU feed-forward.
  static int ane_chunk_rows() noexcept;
  bool ane_setup_(int seq);
  bool ane_eligible_(int L, const Block& b) const;
  void ane_stage_(int L, const Block& b);
  // The q/k/v tier's three, on the same terms.
  bool ane_qkv_setup_(int seq);
  bool ane_qkv_eligible_(int L, const Block& b) const;
  void ane_qkv_stage_(int L, const Block& b);

  // The matrix-core GEMM. Returns false with NOTHING encoded when this
  // box or this shape is not for it, and the caller keeps its steel
  // path -- so a decline is a fallback, never a hole.
  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& xin, std::size_t xe,
                 const QWeight& w,
                 const metal_compute::SharedBuffer& y, std::size_t ye,
                 int M, int N, int K) const;

  // ---- residency + the wired pool ------------------------------------
  //
  // BlockSlots alone re-reads the whole stack every forward, which at 40
  // steps is the checkpoint forty times over. What stops that is
  // PROMOTION: a streamed block that the box can hold moves out of its
  // slot into _blocks and is read no more. How many it can hold is
  // measured (BlockResidency), not predicted.
  std::size_t wire_block_(Block& b, bool on);
  std::size_t wire_fixed_(bool on);
  std::size_t wire_retry_slack_() const;
  std::size_t evict_tail_block_();
  void resident_pages_(std::size_t* examined, std::size_t* incore,
                       std::size_t* paged_out) const;
  // The checkpoint's per-block figure, for the wiring retry slack. Set
  // with the residency schedule; 0 until then.
  std::size_t _wire_block_hint = 0;
  // cos/sin tables for the joint sequence, f32, [joint, head_dim] each.
  void build_rope_(const qi21::Layout& lay,
                   metal_compute::SharedBuffer* cos_out,
                   metal_compute::SharedBuffer* sin_out) const;

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  int _quant_bits = 0;
  int _quant_group = 64;

  // Model-level weights. All biasless.
  QWeight _img_in;                          // [hidden, in_channels]
  QWeight _txt_in_a, _txt_in_b;             // in_layer, out_layer
  metal_compute::SharedBuffer _txt_norm;    // ZERO-CENTERED: stored as
                                            // scale-1, so the loader adds 1
  QWeight _time_1, _time_2;                 // timestep_embedder
  QWeight _mod;                             // the ONE shared modulation
  QWeight _norm_out;                        // scale-only final AdaLN
  QWeight _proj_out;                        // [out_channels, hidden]

  std::vector<Block> _blocks;               // empty while streaming
  bool _stream_blocks = false;
  BlockSlots<Block> _slots;
  BlockResidency _resid;
  WiredPool _wire;
  std::shared_ptr<WeightSet> _ws;
  std::function<bool()> _stream_stop;
  DitBlockProgressFn _block_progress;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_rope,
      _lib_attn, _lib_attn_nax, _lib_mma, _lib_dequant, _lib_qmm;
  metal_compute::ComputeFunction _fn_ln_mod, _fn_gated_tanh, _fn_rms,
      _fn_swiglu, _fn_gelu_tanh, _fn_tr_rope, _fn_residual, _fn_gemm_t,
      // BOTH widths, picked per WEIGHT rather than per checkpoint: a
      // mixed pack (`mixed`, or an 8-bit modulation over a 4-bit body)
      // carries w4 and w8 tensors side by side, and `load_qw_` already
      // reads each one's bits off its own code/scale shapes.
      _fn_qmm4, _fn_qmm8, _fn_dequant4, _fn_dequant8, _fn_gemm_mma,
      _fn_gemm_mma_deep, _fn_gemm_mma_tn2, _fn_copy_rows,
      _fn_gated_tanh4, _fn_swiglu4, _fn_gemm_bm64, _fn_gemm_bm64bn64,
      _fn_ln_plain, _fn_scale_rows, _fn_silu, _fn_transpose;
  // One params buffer per dispatch GROUP is required, not one per
  // model: the block-causal decomposition encodes several attention
  // dispatches into one stream, and a shared mutable buffer would hand
  // the last group's shape to all of them. Sized lazily in forward.
  metal_compute::SharedBuffer _attn_params;
  bool _use_mma2 = false;
  bool _use_attn_nax = false;
  bool _steel_attn_ok = false;
  int  _mma_min_m = 64;
  // Which dense tile the NON-matrix-core path takes: 0 = BM32 (the
  // 32x32x32 default), 1 = BM64, 2 = BM64/BN64. This is the arm an M4
  // runs for EVERY GEMM in the model, so it is not a corner.
  //
  // MEASURED at 1024^2 on the no-matrix-core arm, interleaved and with
  // the round order reversed (the two rounds agree within 0.2%, which
  // is what says this is the tile and not the order): BM32 35834 ms,
  // BM64 34367, BM64/BN64 32033 -- so 2 is the default, at 1.12x the
  // tile this model used to be the only one to load.
  int  _gemm_tile = 2;
  // The smallest segment worth quantizing for Sage. A text run is a few
  // dozen rows; its quantization pass would cost more than the dense
  // dispatch it replaces, and preparing twice inside one block rebuilds
  // the scratch the first dispatch is still reading.
  static constexpr int kSageMinSegment = 512;
  // The smallest segment worth ROUTING. Sol's own note: under 16 key
  // blocks the local band, the sink and the always-exact tail already
  // cover most of the sequence, so there is nothing left to skip.
  static constexpr int kSolMinSegment = 1024;
  // How many blocks a NON-streamed stack may encode before it drains.
  // A streamed block commits and waits on its own, which bounds a stop
  // by one block; a preloaded one encodes the whole stack into a
  // single command buffer, so every bit of GPU work happens inside one
  // wait that the per-block check -- which races through in
  // microseconds -- can never observe. The bound there is one FORWARD.
  // Only paid when a stop callback is actually installed.
  static constexpr int kStopDrainBlocks = 4;
  // Dequant-once scratch: a quantized weight is expanded to dense bf16
  // ONCE per GEMM and the matrix-core tiles read that, rather than every
  // tile re-decoding the codes. Shared across a block's GEMMs -- serial
  // ordering plus Metal's WAR tracking make that safe, since each
  // dequant/matmul pair runs before the next writes.
  mutable metal_compute::SharedBuffer _w_deq;
  std::unique_ptr<MetalSageAttention> _sage;
  std::unique_ptr<MetalSolAttention> _sol;
  std::unique_ptr<AneFeedForward> _ane;
  bool _ane_tried = false;
  std::unique_ptr<AneFeedForward> _ane_qkv;
  bool _ane_qkv_tried = false;
  std::unique_ptr<I8GemmContext> _i8;
};

}  // namespace genai
}  // namespace vpipe

#endif
