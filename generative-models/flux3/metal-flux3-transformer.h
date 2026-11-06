#ifndef GENERATIVE_MODELS_FLUX3_METAL_FLUX3_TRANSFORMER_H
#define GENERATIVE_MODELS_FLUX3_METAL_FLUX3_TRANSFORMER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/shared/ane-ffn.h"
#include "generative-models/shared/block-residency.h"
#include "generative-models/shared/block-slots.h"
#include "generative-models/shared/dit-block-progress.h"
#include "generative-models/shared/dit-flops.h"
#include "generative-models/shared/dit-gpu-progress.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/runtime-lora.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/sage-attention.h"
#include "generative-models/shared/sol-attention.h"
#include "generative-models/shared/wired-pool.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;
class MetalSageAttention;
class MetalSolAttention;
class I8GemmContext;

// FLUX.3 Image Klein's DiT (Flux3Transformer2DModel), on metal-compute.
//
// NOT a FLUX.2 descendant in any way the code can use: no double
// blocks, no fused qkv+mlp projection, no shared joint attention in the
// early stack. Three TOKEN MODES -- the image being denoised, the
// reference images ("image_cond") and the text -- each run their OWN
// stack of `n_mode` blocks, attending only within themselves, and then
// all three run `n_single` joint blocks over one sequence.
//
// ONE BLOCK SHAPE serves all of it, which is what makes the whole
// model one streaming stack:
//
//   h = (1 + scale) * LN(x) + shift          LN without affine, 1e-6
//   a = attn_out(attention(rope(rms(q(h))), rope(rms(k(h))), v(h)))
//   f = mlp_out(silu(gate) * up),  [gate | up] = mlp_in(h)
//   x = x + gate_mod * (a + f)
//
// PARALLEL: one norm feeds both branches and one gate scales their SUM.
// No biases anywhere in the model.
//
// THE MODULATION IS SHARED, per mode, across a whole stack: one
// (shift, scale, gate) for all 5 per-mode blocks of a mode and another
// for all 28 joint blocks -- computed once per forward, not per block.
// In the joint blocks every row takes ITS MODE's vectors, so the three
// row ranges are modulated and gated separately.
//
// THE TIMESTEP IS QUANTIZED, and that is the model, not a detail: the
// reference holds t in bf16 and forms 1000 * t in bf16 before the
// sinusoid, so 0.75 reaches the embedding as 752. A forward fed the
// f32 product moves a random model's output by ~0.1. The image mode
// runs at t; the reference and text modes at 0. A FLOW-MAP checkpoint
// (the distilled Klein) also embeds the time the step jumps TO and adds
// it to every mode's embedding.
//
// THE SEQUENCE is [text | image | references], the order diffusers
// builds. Attention is unmasked and the positions ride RoPE, so the
// order is a layout choice -- this one puts the prompt at row 0, where
// Sol's exact sink wants it, and keeps each mode a contiguous range.
// Positions are 4 axes of 32 (t, h, w, l): an image row takes h and w
// at HALF its grid index, reference n sits at l = -(1 + n), and a text
// row at l = its row, padding included.
class MetalFlux3Transformer {
 public:
  ~MetalFlux3Transformer();

  struct Config {
    int hidden      = 3072;    // num_attention_heads * attention_head_dim
    int n_heads     = 24;
    int head_dim    = 128;
    int n_mode      = 5;       // num_layers: per-mode blocks, per mode
    int n_single    = 28;      // num_single_layers
    int in_channels = 128;     // the image tokens (32 latent ch x 2x2)
    int cond_in     = 128;     // cond_in_channels: the reference tokens
    int joint_dim   = 20480;   // the text tokens (8 Qwen3-VL taps x 2560)
    int time_freq   = 256;     // timestep_channels
    int mlp_hidden  = 9216;    // int(hidden * mlp_ratio)
    float norm_eps  = 1e-6f;   // the LayerNorms and the QK RMSNorms
    float rope_theta = 10000.0f;
    int axes[4]     = {32, 32, 32, 32};   // axes_dims_rope; sum == head_dim
    // The step-distilled checkpoint: carries target_timestep_embedder
    // and REQUIRES the step's target time. Read from config.json and
    // confirmed against the tensors at load.
    bool flow_map   = true;

    // Accel tiers, read from accel-settings.h by the caller.
    bool i8_gemm = false;
    // Whether the kernel tuners MEASURE (the accel key `autotune`,
    // accel-keys.h): false takes the choice believed fastest on this
    // GPU, timing nothing.
    bool autotune = true;
    sage::Config sage;
    sol::Config  sol;
    bool ane_ffn   = false;
    // The ANE's share of the rows in [0,1] (accel-keys.h kAneRows): 0
    // balances from the two engines' rates, a share set is pinned.
    float ane_rows = 0.0f;
    int  ane_layers = 0;   // 0 = every joint block
    bool ane_qkv   = false;
    // The ANE's modules in ROTATED INT8 (W8A8; AneFeedForward::Options::
    // int8): generate-image's `ane_int8`. LOSSY -- the GPU token tier's
    // arithmetic on the ANE's rows; a module that cannot take it keeps
    // fp16 and says so. VPIPE_ANE_INT8 overrides it.
    bool ane_int8  = false;
    // ...unrotated: the GPU token_plain tier's arithmetic (generate-image's
    // `ane_int8_tier: token_plain`). Only with ane_int8.
    bool ane_int8_plain = false;

    // Read what can be read from <dir>/config.json. False when the file
    // is absent, names a different class, or describes a geometry this
    // implementation does not run.
    static bool read_dims(const std::string& dir, Config* out);
  };

  // A reference image, already VAE-encoded and patchified: token-major
  // [grid_h * grid_w, cond_in] bf16 -- the same packing the image
  // latents take.
  struct Ref {
    const metal_compute::SharedBuffer* latents = nullptr;
    int grid_h = 0;
    int grid_w = 0;
  };

  struct Request {
    // The image latents, token-major [grid_h * grid_w, in_channels]
    // bf16, row r * grid_w + c for latent cell (r, c); channel
    // c*4 + dy*2 + dx, the FLUX.2 VAE's own packing.
    const metal_compute::SharedBuffer* latents = nullptr;
    int grid_h = 0;
    int grid_w = 0;
    // The conditioner's text tokens [text_rows, joint_dim] bf16, EVERY
    // row it emitted: the reference's prompt is padded to a multiple of
    // 80 and the DiT attends to the pad rows too.
    const metal_compute::SharedBuffer* text = nullptr;
    int text_rows = 0;
    // In order: reference n takes l = -(1 + n).
    std::vector<Ref> refs;
    // The flow time this step starts at, in [0, 1] (1 = noise), and --
    // for a flow-map checkpoint -- the time it jumps to. Both are
    // quantized inside exactly as the reference quantizes them.
    float t = 0.0f;
    float t_next = 0.0f;
  };

  // ---- runtime LoRA --------------------------------------------------
  struct LoraSpec {
    std::string path;          // one .safetensors of lora_A/B pairs
    float       scale = 1.0f;  // 1.0 = as trained; 0 = exactly off
  };
  static constexpr int kMaxLoraSlots = genai::lora::Stack::kMax;

  static std::unique_ptr<MetalFlux3Transformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  static std::unique_ptr<MetalFlux3Transformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  // LIVE: the strength rides the accumulating GEMM as a constant.
  void set_lora_scale(int slot, float s);
  // How many projections each slot actually bound. 0 is the answer a
  // test has to refuse.
  int lora_modules(int slot) const;

  // One forward: the velocity of the image tokens, [grid_h * grid_w,
  // in_channels] bf16 in the latents' own order. Empty on failure, and
  // ALSO empty with no error on a cooperative stop.
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
  // Drop what a forward keeps for the next one -- the dense stand-in,
  // split-K planes and the int8 context's buffers; each regrows on
  // demand. The stage calls it after a
  // generation on a bounded box, before the decode wants the room.
  void release_forward_scratch();
  // A STREAMED stack residency has kept whole gives its slot pair
  // back (BlockSlots::release_if_saturated) -- between generations; a
  // forward does the same at its top. Returns the bytes freed.
  std::size_t release_idle_slots()
  {
    return _slots.release_if_saturated(_blocks);
  }
  // Streamed blocks BUILT into their slot because the slot was laid
  // out for another (a mixed pack's other width; BlockSlots::Ops::
  // same_layout), over the model's life. For a test to show the slots
  // crossed widths -- and did not give up and stream without them.
  int slot_relayouts() const { return _slots.relaid(); }
  // Whether the block projections run in the signed int8 layout (see
  // _sgn) -- what a test asserts before it compares anything.
  bool signed_layout() const { return _sgn; }
  std::size_t release_resident_blocks(std::size_t bytes);
  std::uint64_t resident_bytes() const;

  bool uses_mma2() const { return _use_mma2; }
  int gemm_tile() const { return _gemm_tile; }
  bool uses_attn_nax() const { return _use_attn_nax; }
  bool uses_sage_attn() const { return (bool)_sage; }
  // How many times Sage's scratch was built: once for a run whose size
  // stays put, since each forward reserves its largest attention (the
  // joint stack) and the per-mode stacks reuse it.
  int sage_scratch_builds() const {
    return _sage ? _sage->scratch_builds() : 0;
  }
  bool uses_i8_gemm() const { return (bool)_i8; }
  bool uses_sol_attn() const { return (bool)_sol; }
  // What the forwards have encoded for perf-flops, credited to the GPU
  // as each block retires (generative-models/shared/dit-flops.h).
  const GpuFlopsTally& gpu_flops() const noexcept { return _flops; }
  bool ane_armed() const noexcept { return _ane != nullptr; }
  bool ane_attempted() const noexcept { return _ane_tried; }
  bool ane_qkv_armed() const noexcept { return _ane_qkv != nullptr; }
  // How many of the armed modules run int8 (Config::ane_int8, where the
  // module and the OS could take it): rotated per token, or with `plain`
  // unrotated (Config::ane_int8_plain).
  int ane_int8_modules(bool plain = false) const noexcept
  {
    auto is = [plain](const AneFeedForward* a) {
      return a != nullptr && a->int8() && a->int8_plain() == plain ? 1 : 0;
    };
    return is(_ane.get()) + is(_ane_qkv.get());
  }
  bool ane_gpu_only() const noexcept
  {
    return _ane != nullptr && _ane->gpu_only();
  }
  bool ane_qkv_gpu_only() const noexcept
  {
    return _ane_qkv != nullptr && _ane_qkv->gpu_only();
  }

  // The number of blocks in the streaming stack and how many a forward
  // with `refs` references runs: the reference mode's 5 blocks are
  // skipped -- never read -- when there is no reference.
  int n_blocks() const { return 3 * _cfg.n_mode + _cfg.n_single; }
  static int blocks_run(const Config& c, bool refs)
  {
    return (refs ? 3 : 2) * c.n_mode + c.n_single;
  }

  // THE STREAMING FLOOR: trunk + ONE slot pair -- every block of every
  // stack has one shape, so one pair serves them all. 0 when the
  // directory cannot be read.
  static std::size_t streaming_floor_bytes(const std::string& dir);
  // Every name prefix the stack is stored under, for a caller that
  // books the floor through the shared helper.
  static const std::vector<std::string>& block_stems();

  static std::size_t ane_runtime_bytes(const Config& c, int seq) noexcept;
  static int ane_plan_seq(int width, int height) noexcept;
  // What ONE forward allocates for its activations at `rows` joint rows
  // (text + image + references): the [rows, hidden] planes, the
  // [rows, 3 mlp] feed-forward pair and the f32 rotary tables. Linear
  // in the rows, and the term that dominates at large sizes -- 8.1 GB
  // at 4096^2 against 11.1 GB of bf16 weights. Plus, as `c` enables
  // them, Sage's and Sol's own scratch at the joint stack's rows (c.sage
  // with its pv8, c.sol): both allocate privately here.
  static std::size_t forward_scratch_bytes(const Config& c,
                                           int rows) noexcept;
  bool streaming() const { return _stream_blocks; }
  int quant_bits() const { return _quant_bits; }

 private:
  MetalFlux3Transformer() = default;

  enum class Retain { Cached, Streamed };

  struct QWeight {
    metal_compute::SharedBuffer w;                    // dense [N,K]
    metal_compute::SharedBuffer codes, scales, qbias; // affine quant
    bool quantized = false;
    int  bits = 0;
    // codes/qbias are in the SIGNED layout (I8GemmContext::gemm_signed):
    // q ^ 0x80 / 0x8 and b' = b + 128 s / 8 s. Only the int8 route and
    // the signed dequant read it; see _sgn.
    bool sgn = false;
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

  struct Block {
    QWeight q, k, v, o;                  // attn.to_q/k/v, attn.to_out.0
    metal_compute::SharedBuffer nq, nk;  // QK RMSNorm [head_dim]
    QWeight ff_in;                       // ff.linear_in [2*mlp, H]: gate|up
    QWeight ff_out;                      // ff.linear_out [H, mlp]
  };

  struct BlockLora {
    lora::Factors q, k, v, o, ff_in, ff_out;
    bool empty() const {
      return q.empty() && k.empty() && v.empty() && o.empty() &&
             ff_in.empty() && ff_out.empty();
    }
  };

  // The stack, in one index space:
  //   [0, n_mode)              mode_blocks.image.*
  //   [n_mode, 2 n_mode)       txt_mode_blocks.*
  //   [2 n_mode, 3 n_mode)     mode_blocks.image_cond.*  (skipped without
  //                            references -- last, so they are a range)
  //   [3 n_mode, + n_single)   single_transformer_blocks.*
  std::string prefix_(int L) const;

  // `sign`: a BLOCK projection, which takes the signed layout when _sgn
  // is set; the model-level linears stay as the checkpoint has them.
  QWeight load_qw_(WeightSet& ws, const std::string& name, Retain r,
                   bool sign = false);
  // Convert a quantized weight the model OWNS (a refilled slot) to the
  // signed layout in place, and say so on it. False touches nothing.
  bool sign_(QWeight& w) const;
  // Is block L stored at the widths `b` was built for? (A mixed pack;
  // see BlockSlots::Ops::same_layout.)
  bool same_layout_(int L, const Block& b) const;
  // A signed weight as a DENSE one, dequantized into _w_deq -- for the
  // routes that do not read the layout. `w` itself when it is not
  // signed; null when the scratch cannot be had.
  const QWeight* signed_dense_(metal_compute::ComputeEncoder& enc,
                               const QWeight& w, QWeight& view, int N,
                               int K) const;
  metal_compute::SharedBuffer to_elt_(WeightSet& ws, const std::string& name,
                                      Retain r);
  bool load_block_(WeightSet& ws, int L, Block& b, Retain r);
  bool load_fixed_(WeightSet& ws);

  void each_block_tensor_(int L, Block& b,
                          const BlockSlots<Block>::TensorFn& fn) const;
  bool clone_block_(const Block& src, Block& dst, bool copy) const;
  metal_compute::SharedBuffer rebuild_one_(const std::string& nm,
                                           Placement how);
  bool fill_narrow_(const std::string& nm,
                    const metal_compute::SharedBuffer& dst);
  void configure_slots_();
  static std::size_t qw_bytes_(const QWeight& w);
  static std::size_t block_bytes_(const Block& b);
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

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  int _quant_bits = 0;
  int _quant_group = 64;
  bool _src_f32 = false;

  // Model-level weights.
  QWeight _x_img, _x_cond, _ctx_in;         // the three input projections
  QWeight _time_1, _time_2;                 // timestep_embedder
  QWeight _tgt_1, _tgt_2;                   // target_timestep_embedder
  // early_ / single_stream_modulations, indexed by kModeImage/Cond/Text.
  QWeight _early_mod[3], _single_mod[3];
  QWeight _out_mod;                         // norm_out.linear [2H, H]
  QWeight _proj_out;                        // [in_channels, H]

  std::vector<Block> _blocks;               // the whole stack
  bool _stream_blocks = false;
  BlockSlots<Block> _slots;
  BlockResidency _resid;
  WiredPool _wire;
  std::shared_ptr<WeightSet> _ws;
  std::function<bool()> _stream_stop;
  DitBlockProgressFn _block_progress;
  std::vector<std::uint8_t> _narrow_scratch;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_rope,
      _lib_attn, _lib_attn_nax, _lib_mma, _lib_dequant, _lib_qmm;
  metal_compute::ComputeFunction _fn_ln_mod, _fn_gated, _fn_gated4,
      _fn_rms, _fn_swiglu_halves, _fn_tr_rope, _fn_tr_rms_rope,
      _fn_residual, _fn_residual4,
      _fn_gemm_t, _fn_qmm4, _fn_qmm8, _fn_dequant4, _fn_dequant8,
      _fn_gemm_mma, _fn_gemm_mma_deep, _fn_gemm_mma_tn2, _fn_gemm_bm64,
      _fn_gemm_bm64bn64, _fn_silu, _fn_transpose;
  std::map<unsigned, metal_compute::ComputeFunction> _attn_fn;
  bool _use_mma2 = false;
  bool _use_attn_nax = false;
  bool _steel_attn_ok = false;
  int  _mma_min_m = 64;
  int  _gemm_tile = 2;
  // QK RMSNorm and RoPE in ONE pass with one rounding, as the reference
  // does them; off only for the A/B (VPIPE_FLUX3_UNFUSED_QK_NORM).
  bool _fused_qk = true;
  static constexpr int kSageMinSegment = 512;
  static constexpr int kSolMinSegment = 1024;
  // How many blocks a NON-streamed stack may encode before it drains,
  // so a stop that arrives while the GPU is busy is seen at all (see
  // the forward).
  static constexpr int kStopDrainBlocks = 4;
  mutable metal_compute::SharedBuffer _w_deq;
  // THE SIGNED-CODE LAYOUT for the blocks' quantized projections: with
  // i8_gemm on a matrix-core box they run int8 x int8 off the layout
  // (I8GemmContext::gemm_signed), where they took dequant + requant.
  // Converted once: at load into the entries the set keeps, after every
  // refill in the slot. OFF under the ANE tier, whose stager reads the
  // codes unsigned. VPIPE_I8_SIGNED=0 for the A/B.
  bool _sgn = false;
  std::unique_ptr<MetalSageAttention> _sage;
  // Reserve Sage's scratch at the forward's largest attention (false:
  // VPIPE_FLUX3_SAGE_NO_RESERVE, the A/B -- it rebuilds per row count).
  bool _sage_reserve = true;
  std::unique_ptr<MetalSolAttention> _sol;
  GpuFlopsTally _flops;
  SolFlops _sol_flops;
  std::unique_ptr<AneFeedForward> _ane;
  bool _ane_tried = false;
  std::unique_ptr<AneFeedForward> _ane_qkv;
  bool _ane_qkv_tried = false;
  std::unique_ptr<I8GemmContext> _i8;

  // ---- runtime LoRA ---------------------------------------------------
  // Indexed [slot][L] over the stack's own index space.
  lora::Applier _lora;
  std::vector<BlockLora> _lora_blk[kMaxLoraSlots];
  float _lora_scale[kMaxLoraSlots] = {};
  int _lora_bound[kMaxLoraSlots] = {};
  int _lora_rank_prefix[kMaxLoraSlots] = {};
  int _lora_rank_total = 0;
  int _lora_slots = 0;
  mutable std::size_t _lora_scratch_rows = 0;
  bool bind_loras_(const std::vector<LoraSpec>& loras);
  lora::Stack lora_for_(int L, lora::Factors BlockLora::* which) const;
};

// The reference's sampler, to the bit where it matters.
namespace flux3 {

// The schedule's N + 1 nodes, 1 -> 0: t_k = a s / (1 + (a - 1) s) over
// s = linspace(1, 0, N + 1), all in f32 exactly as torch computes it --
// linspace's two-sided formula included. Bit-exactness is not vanity
// here: the forward rounds every node to bf16, and a node one f32 ulp
// off a bf16 tie lands on the other side of it.
std::vector<float> schedule(int steps, float alpha);

// What the reference's timestep embedding actually sees: t rounded to
// bf16, then 1000 * t computed in f32 and rounded to bf16 again.
float quantized_time(float t);

// The 256-channel sinusoid of `quantized_time(t)`, [cos | sin], each
// value rounded to bf16 as the reference's `.to(t)` does.
std::vector<float> time_embedding(float t, int dim = 256);

// True when `root` is BFL's OWN single-file layout of a FLUX.3 checkpoint
// -- the root model.safetensors + ae.safetensors that BFL's README
// downloads (or a variants/ folder of it) -- rather than the diffusers
// transformer/ + vae/ this tree reads. Recognised so it can be REFUSED
// by name: unrecognised, it falls through to the "krea2" guess and the
// conditioner encodes the prompt as Krea-2's.
bool is_bfl_root_layout(const std::string& root);

// What every stage says when it refuses one.
std::string bfl_root_layout_message(const std::string& root);

}  // namespace flux3

}  // namespace genai
}  // namespace vpipe

#endif
