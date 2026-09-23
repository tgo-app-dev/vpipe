#ifndef GENERATIVE_MODELS_Z_IMAGE_METAL_Z_IMAGE_TRANSFORMER_H
#define GENERATIVE_MODELS_Z_IMAGE_METAL_Z_IMAGE_TRANSFORMER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/shared/ane-ffn.h"
#include "generative-models/shared/block-residency.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/shared/dit-block-progress.h"
#include "generative-models/shared/dit-gpu-progress.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/runtime-lora.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/sage-attention.h"
#include "generative-models/shared/sol-attention.h"
#include "generative-models/shared/wired-pool.h"
#include "generative-models/z-image/z-image-layout.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
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

// Z-Image's DiT (ZImageTransformer2DModel), on metal-compute.
//
// A 6B SINGLE-stream stack in the Lumina/NextDiT lineage -- the first
// model of that lineage in this tree, so none of the Qwen-Image or FLUX
// block code applies. See z-image-layout.h for the sequence it runs
// over.
//
// THE BLOCK, and every line of it differs from this tree's other DiTs:
//
//   h = rms(x) * n1 * scale_msa
//   x = x + tanh(gate_msa) * (rms(attn(h)) * n2)
//   h = rms(x) * n3 * scale_mlp
//   x = x + tanh(gate_mlp) * (rms(ffn(h)) * n4)
//
// Four things there are each a silent failure when dropped:
//
//  * EVERY NORM IS AN RMSNorm WITH A LEARNED WEIGHT, not a bare
//    LayerNorm. Only the final layer's norm_final is a LayerNorm, and
//    that one carries no affine at all.
//  * THE BLOCK IS SANDWICH-NORMED. n2 and n4 normalize the attention's
//    and the feed-forward's OUTPUT, inside the residual and under the
//    gate. A post-norm dropped here still denoises, to the wrong place.
//  * THE GATES GO THROUGH TANH and the scales are 1 + x.
//  * adaLN_modulation IS A BARE Linear FOR A BLOCK AND SiLU+Linear FOR
//    THE FINAL LAYER. The reference spells that difference only as the
//    index inside an nn.Sequential -- `.0` against `.1` -- so the
//    checkpoint's tensor names are the only place it is visible.
//
// THE STACK IS 34 BLOCKS OF ONE SHAPE, in three groups:
//
//   context_refiner  2, UNMODULATED, over the caption rows alone
//   noise_refiner    2, modulated, over the image rows alone
//   layers          30, modulated, over the joined sequence
//
// The two refiners touch disjoint row ranges and never see each other's
// output, so their order is free; this runs the caption first, which
// leaves the 32 MODULATED blocks contiguous for the streaming stack and
// the progress report.
//
// ATTENTION IS DENSE AND BIDIRECTIONAL -- no causality, no mask, one
// dispatch per block. At batch 1 the reference's attention mask is
// None, and its padded rows are real learned tokens that attend like
// any other. That makes this the simplest attention of any DiT here and
// it is also the whole of its cost at large sizes.
//
// TIMESTEP RUNS BACKWARDS relative to every other family in this tree.
// The pipeline feeds the model `1 - sigma`, so the modulation at the
// noisiest end is what other models see at the cleanest. `Request`
// takes sigma and converts, so a caller hands this the same number it
// hands Krea-2.
class MetalZImageTransformer {
 public:
  ~MetalZImageTransformer();

  struct Config {
    int hidden      = 3840;    // `dim`
    int n_heads     = 30;
    int head_dim    = 128;     // dim / n_heads; also sum(axes)
    int n_layers    = 30;      // the MAIN stack
    int n_refiner   = 2;       // each of context_refiner and noise_refiner
    int in_channels = 16;      // VAE latent channels
    int patch       = 2;       // all_patch_size[0]
    int f_patch     = 1;       // all_f_patch_size[0]
    int ffn_inner   = 10240;   // int(dim / 3 * 8)
    int cap_dim     = 2560;    // cap_feat_dim, the text encoder's width
    int adaln_dim   = 256;     // ADALN_EMBED_DIM = min(dim, 256)
    int time_mid    = 1024;    // TimestepEmbedder mid_size
    int time_freq   = 256;     // frequency_embedding_size
    float norm_eps  = 1e-5f;   // NOT 1e-6: the RMSNorms and the qk norms
    float rope_theta = 256.0f; // NOT 10000
    float t_scale   = 1000.0f;
    int axes[3]     = {32, 48, 48};        // axes_dims; sum == head_dim
    int axes_lens[3] = {1536, 512, 512};   // rotary table extents
    bool qk_norm    = true;

    // Patch-vector width the x_embedder consumes: pF*pH*pW*C.
    int patch_dim() const {
      return f_patch * patch * patch * in_channels;
    }
    // Blocks carrying adaLN: noise_refiner + layers. The streaming
    // stack is exactly these, in that order.
    int n_mod_blocks() const { return n_refiner + n_layers; }
    int n_blocks() const { return n_refiner + n_mod_blocks(); }

    // Accel tiers, read from accel-settings.h by the caller.
    bool i8_gemm = false;
    sage::Config sage;
    sol::Config  sol;
    bool ane_ffn   = false;
    int  ane_rows  = 0;    // 0 = balance from the two engines' rates
    int  ane_layers = 0;   // 0 = every block
    // q/k/v on the ANE too: a SECOND module, one matmul hidden ->
    // 3*hidden with the three weights stacked in its slot, split by
    // rows like the feed-forward and sharing its worker. Main stack
    // only, for the same reason the feed-forward is.
    bool ane_qkv   = false;

    // Read what can be read from <dir>/config.json. False when the file
    // is absent or names a different class.
    static bool read_dims(const std::string& dir, Config* out);
  };

  struct Request {
    // The image latents, ALREADY PACKED into patch rows:
    // [layout->img_tokens, patch_dim()]. zimage::pack_latents builds
    // this, and the packing order is part of the model -- see there.
    const metal_compute::SharedBuffer* latents = nullptr;
    // The text encoder's hidden states, one row per REAL caption token:
    // [layout->cap_len, cap_dim].
    const metal_compute::SharedBuffer* cap = nullptr;
    const zimage::Layout* layout = nullptr;
    // SIGMA in [0, 1], the same number every other DiT here takes. The
    // forward converts: the model's own input is (1 - sigma) * t_scale.
    float timestep = 0.0f;
  };

  // ---- runtime LoRA --------------------------------------------------
  struct LoraSpec {
    std::string path;          // one .safetensors of lora_A/B pairs
    float       scale = 1.0f;  // 1.0 = as trained; 0 = exactly off
  };
  // Two adapters at once, which is what a real request wants: a
  // distillation or identity adapter beside a style one. Separate
  // slots rather than one merged set of factors -- concatenating on
  // the rank axis is exact and free in the forward, but it folds both
  // strengths into A and turns the live knob back into a rebuild.
  static constexpr int kMaxLoraSlots = genai::lora::Stack::kMax;

  static std::unique_ptr<MetalZImageTransformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  static std::unique_ptr<MetalZImageTransformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  // LIVE. The strength rides the accumulating GEMM as a constant, so
  // it can be turned between steps without rebuilding any factors;
  // exactly 0 encodes nothing at all, so "off" is off rather than a
  // pair of roundings that cancel.
  void set_lora_scale(int slot, float s);
  // How many projections each slot actually bound. 0 is the answer a
  // test has to refuse: an adapter for another model binds nothing and
  // reports success otherwise.
  int lora_modules(int slot) const;

  // One forward. Returns the IMAGE rows only --
  // [layout->img_tokens, patch_dim()] as bf16 -- which is what the
  // reference's unpatchify reads off the front of its joint output.
  // Empty on failure, and ALSO empty with no error on a cooperative
  // stop.
  metal_compute::SharedBuffer forward(const Request& req,
                                      std::string* err = nullptr);

  const Config& config() const { return _cfg; }

  // Streaming + residency, the shared contract every DiT here
  // implements.
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
  int gemm_tile() const { return _gemm_tile; }
  bool uses_attn_nax() const { return _use_attn_nax; }
  bool uses_sage_attn() const { return (bool)_sage; }
  bool uses_i8_gemm() const { return (bool)_i8; }
  bool uses_sol_attn() const { return (bool)_sol; }
  bool ane_armed() const noexcept { return _ane != nullptr; }
  bool ane_attempted() const noexcept { return _ane_tried; }
  bool ane_qkv_armed() const noexcept { return _ane_qkv != nullptr; }
  // True when the checkpoint stores F32 and every block is therefore
  // narrowed on the way in. Turbo does; the undistilled base does not.
  bool narrows_f32() const { return _src_f32; }

  // THE STREAMING FLOOR: trunk + ONE slot pair, in the bytes this model
  // actually HOLDS. It is here rather than on the shared
  // `streaming_floor_bytes` helper because two facts about this family
  // only the model knows, and both move the answer by ~2x:
  //
  //  * the streamed stack is ONE stack spelled under TWO prefixes
  //    (`noise_refiner.` then `layers.`), sharing one slot pair, while
  //    `context_refiner.` is HELD in both modes and belongs to the
  //    trunk. The shared helper books a pair per STEM, so naming both
  //    would promise two pairs where there is one.
  //  * an F32 checkpoint (Turbo) lands as bf16 everywhere, so the
  //    bytes held are HALF the bytes on disk. A floor measured from
  //    `nbytes` over-states by exactly 2x.
  //
  // 0 when the directory cannot be read, which the caller reads as
  // "cannot be reduced".
  static std::size_t streaming_floor_bytes(const std::string& dir);

  static std::size_t ane_runtime_bytes(const Config& c, int seq) noexcept;
  static int ane_plan_seq(int width, int height) noexcept;
  bool streaming() const { return _stream_blocks; }
  int quant_bits() const { return _quant_bits; }

 private:
  MetalZImageTransformer() = default;

  enum class Retain { Cached, Streamed };

  struct QWeight {
    metal_compute::SharedBuffer w;                    // dense [N,K]
    metal_compute::SharedBuffer codes, scales, qbias; // affine quant
    bool quantized = false;
    int  bits = 0;
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

  // Everything a block owns. The attention and feed-forward projections
  // are biasless; adaLN is the one Linear in a block that has a bias,
  // and the two context_refiner blocks do not have it at all.
  struct Block {
    QWeight qw, kw, vw, ow;              // attention (to_q/k/v/to_out.0)
    metal_compute::SharedBuffer nq, nk;  // q/k RMSNorm [head_dim]
    QWeight w1, w3, w2;                  // SwiGLU: gate, up, down
    metal_compute::SharedBuffer an1, an2;  // attention_norm1 / norm2
    metal_compute::SharedBuffer fn1, fn2;  // ffn_norm1 / ffn_norm2
    QWeight ada;                         // adaLN_modulation.0 [4H, 256]
    metal_compute::SharedBuffer ada_b;   // its bias [4H]
  };

  // One slot's factors for one block. The adaLN projection is
  // deliberately NOT adapted: no publisher in this lineage trains it,
  // and binding it would take a file's silence for a zero.
  struct BlockLora {
    lora::Factors q, k, v, o, w1, w3, w2;
    bool empty() const {
      return q.empty() && k.empty() && v.empty() && o.empty() &&
             w1.empty() && w3.empty() && w2.empty();
    }
  };

  // A block's checkpoint prefix, which is the whole of what the three
  // groups differ by. `mod` indexes the streaming stack (noise_refiner
  // then layers); `ctx` the two preloaded caption blocks.
  std::string mod_prefix_(int i) const;
  std::string ctx_prefix_(int i) const;

  QWeight load_qw_(WeightSet& ws, const std::string& name, Retain r);
  metal_compute::SharedBuffer to_elt_(WeightSet& ws, const std::string& name,
                                      Retain r);
  bool load_block_(WeightSet& ws, const std::string& pre, bool with_ada,
                   Block& b, Retain r);
  bool load_fixed_(WeightSet& ws);

  // BlockSlots' callbacks; see shared/block-slots.h.
  void each_block_tensor_(int L, Block& b,
                          const BlockSlots<Block>::TensorFn& fn) const;
  bool clone_block_(const Block& src, Block& dst, bool copy) const;
  metal_compute::SharedBuffer rebuild_one_(const std::string& nm,
                                           Placement how);
  // F32 -> bf16 into a destination that already exists. Turbo's whole
  // checkpoint takes this path: its bytes are twice the width of the
  // buffers this model reads, so a raw read has nowhere to put them.
  bool fill_narrow_(const std::string& nm,
                    const metal_compute::SharedBuffer& dst);
  void configure_slots_();
  static std::size_t qw_bytes_(const QWeight& w);
  static std::size_t block_bytes_(const Block& b);

  // The timestep sinusoid and its two Linears, on the host: 256
  // channels of one row is not worth a dispatch.
  std::vector<float> time_proj_(float sigma) const;
  std::size_t checkpoint_block_bytes_() const;

  static int ane_chunk_rows() noexcept;
  bool ane_setup_(int seq);
  bool ane_eligible_(int L, const Block& b) const;
  void ane_stage_(int L, const Block& b);
  bool ane_qkv_setup_(int seq);
  bool ane_qkv_eligible_(int L, const Block& b) const;
  void ane_qkv_stage_(int L, const Block& b);

  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& xin, std::size_t xe,
                 const QWeight& w,
                 const metal_compute::SharedBuffer& y, std::size_t ye,
                 int M, int N, int K) const;

  std::size_t wire_block_(Block& b, bool on);
  std::size_t wire_fixed_(bool on);
  std::size_t wire_retry_slack_() const;
  std::size_t evict_tail_block_();
  void resident_pages_(std::size_t* examined, std::size_t* incore,
                       std::size_t* paged_out) const;
  std::size_t _wire_block_hint = 0;

  // cos/sin tables for the JOINT sequence, f32, [joint, head_dim] each.
  // The refiners take prefixes of it -- the reference builds the two
  // halves' tables separately and concatenates them in the same order.
  void build_rope_(const zimage::Layout& lay,
                   metal_compute::SharedBuffer* cos_out,
                   metal_compute::SharedBuffer* sin_out) const;

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  int _quant_bits = 0;
  int _quant_group = 64;
  bool _src_f32 = false;

  // Model-level weights.
  QWeight _x_in;                            // all_x_embedder [H, patch_dim]
  metal_compute::SharedBuffer _x_in_b;
  metal_compute::SharedBuffer _cap_norm;    // cap_embedder.0 RMSNorm [cap_dim]
  QWeight _cap_in;                          // cap_embedder.1 [H, cap_dim]
  metal_compute::SharedBuffer _cap_in_b;
  QWeight _time_1, _time_2;                 // t_embedder.mlp.0 / .2
  metal_compute::SharedBuffer _time_1b, _time_2b;
  metal_compute::SharedBuffer _x_pad, _cap_pad;   // the two learned pads
  QWeight _final_ada;                       // all_final_layer adaLN [H, 256]
  metal_compute::SharedBuffer _final_ada_b;
  QWeight _final_lin;                       // [patch_dim, H]
  metal_compute::SharedBuffer _final_lin_b;

  std::vector<Block> _ctx;                  // context_refiner, ALWAYS held
  std::vector<Block> _blocks;               // the modulated stack
  bool _stream_blocks = false;
  BlockSlots<Block> _slots;
  BlockResidency _resid;
  WiredPool _wire;
  std::shared_ptr<WeightSet> _ws;
  std::function<bool()> _stream_stop;
  DitBlockProgressFn _block_progress;
  // The F32 landing scratch for fill_narrow_. One buffer for the run,
  // grown to the largest tensor it is asked for.
  std::vector<std::uint8_t> _narrow_scratch;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_rope,
      _lib_attn, _lib_attn_nax, _lib_mma, _lib_dequant, _lib_qmm;
  metal_compute::ComputeFunction _fn_ln_plain, _fn_adaln, _fn_adaln4,
      _fn_gated_tanh, _fn_gated_tanh4, _fn_rms, _fn_swiglu, _fn_swiglu4,
      _fn_tr_rope, _fn_residual, _fn_residual4, _fn_gemm_t, _fn_qmm,
      _fn_dequant4, _fn_dequant8, _fn_gemm_mma, _fn_gemm_mma_deep,
      _fn_gemm_mma_tn2, _fn_copy_rows, _fn_gemm_bm64, _fn_gemm_bm64bn64,
      _fn_silu, _fn_transpose, _fn_bias_add;
  // The steel entry, specialized per (q-aligned, k-aligned). Memoized
  // because building a pipeline state is not free and a sampler runs
  // the same three sequence lengths on every one of its steps.
  std::map<unsigned, metal_compute::ComputeFunction> _attn_fn;
  bool _use_mma2 = false;
  bool _use_attn_nax = false;
  bool _steel_attn_ok = false;
  int  _mma_min_m = 64;
  int  _gemm_tile = 2;
  // Fold (1 + scale) into the norm's gain instead of modulating the
  // normed tensor. Exact algebra; the flag exists so the A/B can
  // interleave its arms in ONE process rather than compare two builds.
  bool _fold_mod = true;
  static constexpr int kSageMinSegment = 512;
  static constexpr int kSolMinSegment = 1024;
  // How many blocks a NON-streamed stack may encode before it drains,
  // so a stop that arrives while the GPU is busy is seen at all.
  //
  // A streamed block commits and waits on its own, which bounds a stop
  // by one block. A preloaded one does not: without this the whole
  // stack encodes into a single command buffer and every bit of GPU
  // work happens inside one wait that no per-block check can
  // interrupt, so the bound becomes one FORWARD -- ~4.3 s at 512^2 and
  // tens of seconds at 2048^2. Four blocks costs a handful of pipeline
  // bubbles per forward and is only paid when a stop callback is
  // actually installed.
  static constexpr int kStopDrainBlocks = 4;
  mutable metal_compute::SharedBuffer _w_deq;
  std::unique_ptr<MetalSageAttention> _sage;
  std::unique_ptr<MetalSolAttention> _sol;
  std::unique_ptr<AneFeedForward> _ane;
  bool _ane_tried = false;
  std::unique_ptr<AneFeedForward> _ane_qkv;
  bool _ane_qkv_tried = false;
  std::unique_ptr<I8GemmContext> _i8;

  // ---- runtime LoRA ---------------------------------------------------
  //
  // Indexed [slot][block] over the SAME index space mod_prefix_ uses,
  // with the context refiner's two blocks kept separately because they
  // are not in that stack. Bound once at load: the factors are the
  // adapter's, and only the STRENGTH is live.
  lora::Applier _lora;
  std::vector<BlockLora> _lora_mod[kMaxLoraSlots];
  std::vector<BlockLora> _lora_ctx[kMaxLoraSlots];
  float _lora_scale[kMaxLoraSlots] = {0.0f, 0.0f};
  int _lora_bound[kMaxLoraSlots] = {0, 0};
  int _lora_rank_prefix[kMaxLoraSlots] = {0, 0};
  int _lora_rank_total = 0;
  int _lora_slots = 0;
  // Rows the [rows, rank] intermediate is sized for THIS forward, so
  // a slot's base offset is rows * its rank prefix.
  mutable std::size_t _lora_scratch_rows = 0;
  bool bind_loras_(const std::vector<LoraSpec>& loras);
  // The per-projection stack for one block, or an empty one when
  // nothing is attached. `ctx` selects the context refiner's space.
  lora::Stack lora_for_(bool ctx, int L,
                        lora::Factors BlockLora::* which) const;
};

}  // namespace genai
}  // namespace vpipe

#endif
