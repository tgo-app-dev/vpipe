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
#include "generative-models/shared/dit-flops.h"
#include "generative-models/shared/dit-gpu-progress.h"
#include "generative-models/shared/fp8-expand.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/runtime-lora.h"
#include "generative-models/shared/sage-attention.h"
#include "generative-models/shared/sol-attention.h"
#include "generative-models/shared/wired-pool.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
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
    // The int8 GEMM's tier (I8GemmContext::Tier): 0 accurate (groups
    // of 64), 1 fast (the block projections held as rotated int8 at a
    // coarse group), 2 token (int8 per channel x per token, both
    // operands Hadamard-rotated in blocks of 256), 3 token unrotated
    // (the i8x layout: a Draw Things i8x checkpoint's codes are read as
    // stored; rotated, they are requantized). Meaningful only with
    // i8_gemm; VPIPE_I8_TIER and VPIPE_I8_TOKEN_ROT override it either
    // way.
    int i8_tier = 0;
    // A w4 pack's block projections on the fast / token tiers: HOLD
    // their int8 layout (true; twice their bytes, the fastest) or keep
    // the pack and convert it before each GEMM into one scratch (false,
    // the default; a streamed block always does). The engine's call --
    // generate-image sets it where the box has room.
    bool i8_hold_w4 = false;
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
    // The ANE's modules in ROTATED INT8 (W8A8; AneFeedForward::Options::
    // int8): generate-image's `ane_int8`. LOSSY -- the GPU token tier's
    // arithmetic on the ANE's rows; a module that cannot take it keeps
    // fp16 and says so. VPIPE_ANE_INT8 overrides it.
    bool ane_int8  = false;
    // ...unrotated: the GPU token_plain tier's arithmetic (generate-image's
    // `ane_int8_tier: token_plain`). Only with ane_int8.
    bool ane_int8_plain = false;

    // Read what can be read from <dir>/config.json. False when the file
    // is absent or names a different class. `dir` may also be a single
    // .safetensors DiT (is_single_file_dit), whose dimensions are then
    // read off its tensors.
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
    // TOKEN-major [prefix, hidden] (the strided attention path) rather
    // than head-major [heads, prefix, head_dim]; a step in the other
    // layout refuses the cache.
    bool token_major = false;
    bool populated() const { return !k.empty() && prefix > 0; }
    void clear()
    {
      k.clear(); v.clear(); prefix = joint = target = 0;
      token_major = false;
    }
  };
  enum class KvMode {
    kOff,       // no cache; run the whole joint sequence every step
    kFill,      // run the whole sequence and KEEP the prefix (step one)
    kCached,    // run the target rows only, against the kept prefix
  };

  // What the forward leaves for a LoRA trainer's backward. Every buffer is
  // the storage type and the caller's; a null one is not written.
  //
  //   ckpt          [n_layers, rows, hidden] -- each block's INPUT, the
  //                 checkpoint its backward recomputes the block from;
  //   final_hidden  [rows, hidden] -- the stream after the last block,
  //                 norm_out's input;
  //   modb          [mod_rows, 4 * hidden] -- the shared modulation;
  //   nsc           [mod_rows, hidden] -- norm_out's scale rows.
  //
  // A training forward is EXACT: Sol and Sage, the approximate attention
  // tiers, are skipped even when configured, and the KV cache refused.
  //
  // KEEP: every block's intermediates too, each an arena of n_layers
  // slices ([rows, hidden], [rows, ffn], or [heads, rows] f32 for the
  // attention's exported row statistics), so the backward reads them
  // instead of recomputing the block. The feed-forward then runs on the
  // GPU -- the ANE's fused module never materialises gate and up -- while
  // the q/k/v tier, whose outputs ARE the kept projections, still runs.
  struct TrainKeep {
    const metal_compute::SharedBuffer *h1 = nullptr, *qpre = nullptr,
        *kpre = nullptr, *vb = nullptr, *qh = nullptr, *kh = nullptr,
        *vh = nullptr, *ao = nullptr, *af = nullptr, *x1 = nullptr,
        *h2 = nullptr, *g = nullptr, *u = nullptr, *s = nullptr,
        *mlm = nullptr, *mls = nullptr;
  };
  struct TrainTap {
    const metal_compute::SharedBuffer* ckpt = nullptr;
    const metal_compute::SharedBuffer* final_hidden = nullptr;
    const metal_compute::SharedBuffer* modb = nullptr;
    const metal_compute::SharedBuffer* nsc = nullptr;
    const TrainKeep* keep = nullptr;
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
    // TRAINING: what a backward needs from this forward, copied out as
    // the forward runs. Null for inference.
    const TrainTap* train = nullptr;
  };


  // A single .safetensors holding a Qwen-Image-2.1 DiT under its
  // diffusers names -- unsloth's pre-quantized FP8 file, which has no
  // config beside it. Recognised by what the tensors are called once
  // the checkpoint is open (torchao's spelling already translated).
  static bool is_single_file_dit(const std::string& path);

  // ---- runtime LoRA --------------------------------------------------
  struct LoraSpec {
    std::string path;          // one .safetensors of adapter factors
    float       scale = 1.0f;  // 1.0 = as trained; 0 = exactly off
  };
  // Up to eight adapters at once -- a few-step distillation beside style
  // or identity ones -- each in its own slot so its strength stays a
  // live knob.
  static constexpr int kMaxLoraSlots = genai::lora::Stack::kMax;

  static std::unique_ptr<MetalQwenImage21Transformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  static std::unique_ptr<MetalQwenImage21Transformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  // LIVE. The strength rides the accumulating GEMM as a constant, so it
  // can change between steps with no rebuild; exactly 0 encodes
  // nothing, so "off" is off rather than two roundings that cancel.
  void set_lora_scale(int slot, float s);
  // How many projections each slot bound. 0 is what a test must refuse:
  // an adapter for another model binds nothing and says so only here.
  int lora_modules(int slot) const;
  // The slots in use, in the order the adapters were given -- one that
  // could not be applied included, EMPTY, so the indices stay the
  // caller's.
  int lora_slots() const { return _lora_slots; }

  // A TRAINER's adapter in `slot`: for every projection this family can
  // adapt, `fn(module, n, k, &factors)` fills the factors the forward is to
  // read -- live buffers the trainer keeps writing -- or returns false to
  // leave the module unadapted. Strength 1; the trainer folds alpha into
  // its A factors exactly as a file's would be. Returns the modules bound.
  using LiveLoraFn = std::function<bool(const std::string& module, int n,
                                        int k, lora::Factors* out)>;
  int bind_live_lora(int slot, const LiveLoraFn& fn);
  // Unbind `slot` (a trainer going away): nothing in it is read again,
  // and the last slot's place in the scratch is given back.
  void release_live_lora(int slot);
  // The module path of block L's projection `leaf` ("attn.to_q", ...), as
  // adapter files and bind_live_lora name it.
  static std::string block_module(int L, const std::string& leaf);

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
  // Drop what a forward keeps for the next one -- the dense stand-in,
  // split-K planes and the int8 context's buffers; each regrows on
  // demand. The stage calls it after a
  // generation on a bounded box, before the decode wants the room.
  void release_forward_scratch();
  // A STREAMED stack residency has kept whole gives its slot pair
  // back (BlockSlots::release_if_saturated); every forward does the same
  // at its end. Returns the bytes freed.
  std::size_t release_idle_slots()
  {
    return _slots.release_if_saturated(_blocks);
  }
  // Streamed blocks BUILT into their slot because the slot was laid
  // out for another (a mixed pack's other width; BlockSlots::Ops::
  // same_layout), over the model's life. For a test to show the slots
  // crossed widths -- and did not give up and stream without them.
  // Block projections the model holds in the fast or token layout from
  // a pack at `bits` (4 or 8), over the blocks it HOLDS (all of them,
  // preloaded): a test's proof that a tier took every width of a mixed
  // pack rather than leaving one on the accurate tier.
  int fast_weights(int bits) const;
  int slot_relayouts() const { return _slots.relaid(); }
  // Whether the block projections run in the signed int8 layout (see
  // _sgn) -- what a test asserts before it compares anything.
  bool signed_layout() const { return _sgn; }
  // Whether the block projections run the int8 FAST tier (see _fast).
  bool fast_tier() const { return _fast; }
  // ...and at which group: I8GemmContext::kTokenGroup on the token
  // tier, 0 off both.
  int fast_group() const
  {
    return _fast && _i8 ? _i8->fast_group(_cfg.hidden) : 0;
  }
  // ...and the token tiers' GEMM kernel it took ("blocked 128x64", ...;
  // empty without int8).
  std::string i8_token_kernel() const
  {
    return _i8 ? _i8->token_kernel() : std::string();
  }
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
  // Whether a tier LOST rows on some block and has stayed on the GPU
  // since (AneFeedForward::gpu_only): the rows were recomputed on
  // the GPU, and no later block split.
  bool ane_gpu_only() const noexcept
  {
    return _ane != nullptr && _ane->gpu_only();
  }
  bool ane_qkv_gpu_only() const noexcept
  {
    return _ane_qkv != nullptr && _ane_qkv->gpu_only();
  }

  // What the CoreML module holds, for the resource plan. Its bytes are
  // CoreML's and no other ledger in this process can see them, so the
  // plan claims them in UNITS -- see docs/MODEL-MEMORY.md.
  static std::size_t ane_runtime_bytes(const Config& c, int seq) noexcept;
  // The joint length a plan should book for, from the picture.
  static int ane_plan_seq(int width, int height) noexcept;
  // What one denoise step's forwards allocate beyond the weights, for the
  // plan's denoise phase: img_rows image tokens beside txt_rows of prompt,
  // `cfg` for a negative branch (its own K / V cache), `packed` for a
  // pack (the dense stand-in), `matrix_cores` for the tiers that need
  // them (int8, Sage, Sol).
  static std::size_t forward_scratch_bytes(const Config& c, int img_rows,
                                           int txt_rows, bool cfg,
                                           bool packed, bool matrix_cores);
  bool streaming() const { return _stream_blocks; }
  int quant_bits() const { return _quant_bits; }

 private:
  friend class QwenImage21Trainer;
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
    // FP8 storage: `codes` holds the checkpoint's own [N, K] bytes,
    // widened to bf16 right before each GEMM (fp8_dense_). NOT
    // `quantized`: no group scales, and nothing reading an affine triple
    // may be handed one. A scaled weight's f32 scale is `f8_scale`, one
    // value or one per row (shared/fp8-expand.h).
    fp8::Format f8 = fp8::Format::kNone;
    metal_compute::SharedBuffer f8_scale;
    bool f8_per_row = false;
    // codes/qbias are in the SIGNED layout (I8GemmContext::gemm_signed):
    // q ^ 0x80 / 0x8 and b' = b + 128 s / 8 s. Only the int8 route and
    // the signed dequant read it; see _sgn.
    bool sgn = false;
    // THE FAST TIER's layout (I8GemmContext::gemm_fast) when gc > 0:
    // `codes` holds int8 [N, K] in the group-rotated basis and `cscales`
    // [N, K/gc]. A streamed slot also keeps the raw scales / qbias its
    // refill lands, which post_refill converts from. n / k are the
    // weight's shape, which a refill needs and the buffers do not say.
    int gc = 0;
    int n = 0, k = 0;
    metal_compute::SharedBuffer cscales;
    // A w4 pack's weight kept AS STORED (its codes / scales / qbias) and
    // converted per GEMM (fast_operand_); false for a layout held, w4 or
    // w8 (Config::i8_hold_w4).
    bool pc = false;
    // A DRAW THINGS i8x weight held as it is stored (gc ==
    // kTokenGroup, unrotated): `codes` are the file's `<stem>.i8x` and
    // `cscales` its row scales widened to f32 -- nothing to convert
    // after a refill, and a slot refills both by name.
    bool i8x = false;
    // A DENSE checkpoint's weight in the fast layout in a STREAMED slot:
    // `w` keeps the raw bf16 a refill lands (the source) and post_refill
    // quantizes it into codes / cscales, exactly as a preloaded block's
    // entry was built. A promotion keeps only the layout, as the entry
    // does -- the source is the slot's alone.
    bool ds = false;
    bool is_fp8() const { return f8 != fp8::Format::kNone; }
    bool empty() const {
      return (quantized || is_fp8()) ? codes.empty() : w.empty();
    }
  };

  // Everything a block owns. Biasless throughout, and with NO modulation
  // of its own -- see the class comment.
  struct Block {
    QWeight qw, kw, vw, ow;                       // attention
    metal_compute::SharedBuffer nq, nk;           // q/k RMSNorm [head_dim]
    QWeight gate_w, proj_w, out_w;                // SwiGLU: gate, up, down
  };

  // `sign`: a BLOCK projection, which takes the signed layout when _sgn
  // is set; the model-level linears stay as the checkpoint has them.
  QWeight load_qw_(WeightSet& ws, const std::string& name, Retain r,
                   bool sign = false);
  // Convert a quantized weight the model OWNS (a refilled slot) to the
  // signed layout in place, and say so on it. False touches nothing.
  bool sign_(QWeight& w) const;
  // ...to the fast tier's layout, on the GPU, in place (a refilled slot
  // whose build chose it); a weight not in that layout is untouched.
  bool fast_(QWeight& w) const;
  // The same conversion encoded on `e`, for a caller converting a whole
  // block in one command buffer (post_refill). True with nothing encoded
  // for a weight not in that layout.
  bool encode_fast_(metal_compute::ComputeEncoder& e, QWeight& w) const;
  // The fast-layout operands of a weight on the fast / token tier: its
  // own codes and scales, or -- a w4 pack's weight, held as stored -- the
  // context's scratch it is converted into here, on `enc`, for the GEMM
  // encoded right after (I8GemmContext::encode_to_fast_scratch). False
  // with nothing encoded where the conversion declines.
  bool fast_operand_(metal_compute::ComputeEncoder& enc, const QWeight& w,
                     const metal_compute::SharedBuffer** codes,
                     const metal_compute::SharedBuffer** scales) const;
  // A fast-layout weight dequantized into `out` [N, K] -- the values the
  // fast GEMM reads, for a route that does not read the layout.
  bool fast_dense_(metal_compute::ComputeEncoder& enc, const QWeight& w,
                   const metal_compute::SharedBuffer& out, int N,
                   int K) const;
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
  // What a PROMOTION of slot block `b` keeps (clone_block_ with copy):
  // a dense source and dead pack scales dropped, a w4 pack whose layout
  // is held converted into it. What residency and the wired pool are
  // asked for -- a slot-sized ask books bytes nothing keeps.
  std::size_t kept_block_bytes_(const Block& b) const;
  // ...and makes the promoted block hold exactly that, whichever way it
  // was promoted. A slot's clone already did (clone_block_); without
  // slots the per-block fallback is MOVED in whole, source, dead scales
  // and a per-call w4 pack included. Idempotent: after a clone it finds
  // nothing to do.
  bool keep_block_(Block& b) const;
  // post_refill's conversions, on a stream the reader thread keeps (a
  // fresh queue's first commit costs ~1-1.9 ms; seven a block added up).
  mutable metal_compute::CommandStream _cvt_stream;


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

  // y = x W^T (+ the adapters in `lf`) by whichever route this box and
  // shape take -- FP8 widening, matrix cores, int8, the steel tiles or the
  // affine qmm -- with the LoRA accumulated after every route. The ONE
  // GEMM helper: the forward and a trainer's backward both go through it.
  void lin_(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& zero,
            const metal_compute::SharedBuffer& x, std::size_t xe,
            const QWeight& w, const metal_compute::SharedBuffer& y,
            std::size_t ye, int M, int N, int K,
            const lora::Stack& lf = lora::Stack{});

  // THE FAST TIER over one input for several projections (q|k|v,
  // gate|up): one rotated quantizer, then each GEMM -- what lin_ would run
  // per projection, less the repeated quantizer. False with NOTHING
  // encoded unless every one is in the fast layout at one group with no
  // adapter attached (the caller then runs lin_ per projection).
  struct FastOut {
    const QWeight* w;
    const metal_compute::SharedBuffer* y;
    std::size_t ye;
    int n;                       // the projection's output width
    bool adapted;                // a LoRA stack is attached
  };
  bool fast_shared_(metal_compute::ComputeEncoder& enc,
                    const metal_compute::SharedBuffer& x, std::size_t xe,
                    int M, int K, std::initializer_list<FastOut> outs);

  // The matrix-core GEMM. Returns false with NOTHING encoded when this
  // box or this shape is not for it, and the caller keeps its steel
  // path -- so a decline is a fallback, never a hole.
  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& xin, std::size_t xe,
                 const QWeight& w,
                 const metal_compute::SharedBuffer& y, std::size_t ye,
                 int M, int N, int K) const;

  // FP8. The weight `w` holds as a dense operand for this GEMM: `w`
  // itself, or -- for FP8 -- a `view` aliasing _w_deq, which the
  // expansion has just been encoded into. Null (and _fp8_failed latched)
  // when it cannot be widened.
  const QWeight* fp8_dense_(metal_compute::ComputeEncoder& enc,
                            const QWeight& w, QWeight& view, int N,
                            int K) const;
  // After a slot refill: every FP8 weight still the kind the file says.
  bool fp8_kinds_match_(int L, const Block& b) const;
  // Is block L stored as `b` was built -- affine widths (a mixed pack)
  // and FP8 kinds? See BlockSlots::Ops::same_layout.
  bool same_layout_(int L, const Block& b) const;
  metal_compute::SharedBuffer fp8_scale_buf_(const std::string& weight) const;

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
      _fn_ln_plain, _fn_scale_rows, _fn_silu, _fn_transpose,
      // q/k RMSNorm + rope in place, token-major (see forward's `tm`).
      _fn_rms_rope_tm;
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
  // The checkpoint's FP8 weights and the kernels that widen them.
  fp8::Layout   _f8;
  fp8::Expander _fp8x;
  mutable bool  _fp8_failed = false;

  // ---- runtime LoRA ---------------------------------------------------
  //
  // One slot's factors for one block's seven projections, and for the
  // model-level linears -- which this family's adapters DO train:
  // Viggle's turbo adapts the shared modulation and both timestep
  // Linears, and leaving those out would apply a distillation in part
  // and say nothing. All of them run through the forward's one GEMM
  // helper, so they are adapted the same way. Bound once at load; only
  // the strength is live.
  struct BlockLora {
    lora::Factors q, k, v, o, gate, proj, out;
  };
  struct FixedLora {
    lora::Factors img_in, txt_a, txt_b, time_1, time_2, mod, norm_out,
        proj_out;
  };
  lora::Applier _lora;
  std::vector<BlockLora> _lora_blk[kMaxLoraSlots];
  FixedLora _lora_fix[kMaxLoraSlots];
  float _lora_scale[kMaxLoraSlots] = {};
  int _lora_bound[kMaxLoraSlots] = {};
  int _lora_rank_prefix[kMaxLoraSlots] = {};
  int _lora_rank_total = 0;
  int _lora_slots = 0;
  // Rows the [rows, rank] intermediate is sized for THIS forward -- the
  // widest adapted GEMM's -- so a slot's base offset is rows * its rank
  // prefix.
  mutable std::size_t _lora_scratch_rows = 0;
  bool bind_loras_(const std::vector<LoraSpec>& loras);
  // The stack for one projection, empty when nothing is attached.
  lora::Stack lora_blk_(int L, lora::Factors BlockLora::* which) const;
  lora::Stack lora_fix_(lora::Factors FixedLora::* which) const;
  std::unique_ptr<MetalSageAttention> _sage;
  std::unique_ptr<MetalSolAttention> _sol;
  // perf-flops: GEMMs and attention counted as encoded, credited at
  // retire; Sol's exact half at the keep ratio it realizes.
  GpuFlopsTally _flops;
  SolFlops _sol_flops;
  std::unique_ptr<AneFeedForward> _ane;
  bool _ane_tried = false;
  std::unique_ptr<AneFeedForward> _ane_qkv;
  bool _ane_qkv_tried = false;
  std::unique_ptr<I8GemmContext> _i8;
  // Set by the trainer around a backward GEMM it keeps off the int8
  // pipe (QwenImage21Trainer::set_i8_backward).
  bool _i8_hold = false;
  // THE SIGNED-CODE LAYOUT for the blocks' quantized projections: with
  // i8_gemm on a matrix-core box they run int8 x int8 off the layout
  // (I8GemmContext::gemm_signed), where they took dequant + requant.
  // Converted once: at load into the entries the set keeps, after every
  // refill in the slot. OFF under the ANE tiers, whose stagers read the
  // codes unsigned; the trainer's own dequant reads either. A GEMM held
  // off the int8 pipe (_i8_hold) takes the signed dequant instead.
  // VPIPE_I8_SIGNED=0 for the A/B.
  bool _sgn = false;
  // THE FAST OR TOKEN TIER for the block projections (Config::i8_tier;
  // both hold I8GemmContext's fast layout, at gc or kTokenGroup): g64
  // packs converted -- in the derived entry a retained block keeps, in
  // the slot after every refill; a w4 weight is then held at int8's
  // size, twice its nibbles' -- and bf16 checkpoints quantized once
  // into the entry (a streamed bf16 block stays on the accurate tier).
  // Off under the ANE tiers, whose stagers read the codes as affine.
  bool _fast = false;
  bool _fast_failed = false;
};

}  // namespace genai
}  // namespace vpipe

#endif
