#ifndef GENERATIVE_MODELS_FIBO2_METAL_FIBO2_TRANSFORMER_H
#define GENERATIVE_MODELS_FIBO2_METAL_FIBO2_TRANSFORMER_H

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
#include "generative-models/fibo2/fibo2-layout.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
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

// Fibo 2.0's DiT (BriaFibo2Transformer2DModel; Bria, briaai/Fibo-2.0-*),
// on metal-compute.
//
// DERIVED FROM MetalZImageTransformer, because its block IS Z-Image's --
// the same tensors under the same names, the same sandwich norms, tanh
// gates and (1 + scale), the same q/k RMSNorm and interleaved rotary --
// so the machinery around the block (loading, the int8 layouts, streaming
// slots and residency, the wired pool, runtime LoRA, the ANE tiers, Sage
// and Sol) is that class's, and the notes on it below are that class's
// too, measurements included. What differs is the stack around it:
//
//   noise_refiner   2 blocks over the image and context rows
//   perceiver       2 layers turning the text into `gist` tokens:
//                   cross-attention to text bundle 0, self-attention,
//                   a SwiGLU, then an RMSNorm -- unmodulated, and a
//                   function of the TEXT ALONE (set_text runs it once)
//   layers          44 blocks over [image | context | gist]; five of them
//                   (injection_layer_ids) also cross-attend their own
//                   text bundle through a tanh gate, after the attention
//   final_layer     LayerNorm (no affine) * (1 + scale), a Linear, over
//                   the image rows
//
// PER-TOKEN MODULATION. Every block modulates the IMAGE rows with the
// timestep's adaLN and every other row -- context images and gist tokens,
// which are clean -- with t = 0's. The rows are laid out with the image
// first (fibo2-layout.h), so that is two row ranges, not a mask.
//
// THE TIMESTEP IS SIGMA * t_scale, as most families here take it -- NOT
// Z-Image's 1 - sigma.
//
// THE TEXT IS FIXED FOR A GENERATION. The gist tokens and the injection
// blocks' keys and values depend on the text alone, so set_text computes
// them once and every forward reads them; a forward without a text set
// is refused. The reference recomputes them every step. TWO TEXTS ARE
// HELD (kTextSlots): a guided generation (Fibo 2.0 Dev) runs the prompt
// and the negative at every step, each its own gist count, so each has
// its own slot and neither pass recomputes the other's.
//
// STREAMING. The 46 blocks (refiner then layers) are one uniform stack;
// the injection blocks' cross-attention, the perceiver and the model-level
// tensors are the trunk, held in both modes.
class MetalFibo2Transformer {
 public:
  ~MetalFibo2Transformer();

  struct Config {
    int hidden      = 3584;    // `dim`
    int n_heads     = 28;
    int head_dim    = 128;     // dim / n_heads; also sum(axes)
    int n_layers    = 44;      // the MAIN stack
    int n_refiner   = 2;       // noise_refiner
    int in_channels = 32;      // VAE latent channels (FLUX.2's)
    int patch       = 2;
    int ffn_inner   = 9536;    // (dim * 8 // 3) // 64 * 64
    int cap_dim     = 7680;    // cap_feat_dim: three text layers a bundle
    int adaln_dim   = 256;     // ADALN_EMBED_DIM = min(dim, 256)
    int time_mid    = 1024;    // TimestepEmbedder mid_size
    int time_freq   = 256;     // frequency_embedding_size
    float norm_eps  = 1e-5f;
    float rope_theta = 256.0f;
    float t_scale   = 1000.0f;
    int axes[3]     = {32, 48, 48};        // axes_dims; sum == head_dim
    int axes_lens[3] = {2048, 512, 512};   // rotary table extents
    bool qk_norm    = true;
    // The text: injection blocks and the Perceiver.
    std::vector<int> injection_ids = {6, 14, 22, 30, 38};
    int perceiver_layers = 2;
    int min_gist = 256, max_gist = 512, gist_step = 64;
    int gist_min_text = 256, gist_max_text = 2048;

    int patch_dim() const { return patch * patch * in_channels; }
    // The streamed stack: noise_refiner then layers, in that order.
    int n_mod_blocks() const { return n_refiner + n_layers; }
    int n_blocks() const { return n_mod_blocks(); }
    // Text bundles the forward reads: the Perceiver's, then one per
    // injection block in injection_ids order.
    int n_bundles() const { return 1 + (int)injection_ids.size(); }
    // The bundle an injection block (main-stack index L) reads; -1 for a
    // plain block.
    int bundle_of(int L) const
    {
      for (std::size_t i = 0; i < injection_ids.size(); ++i) {
        if (injection_ids[i] == L) { return 1 + (int)i; }
      }
      return -1;
    }

    // Accel tiers, read from accel-settings.h by the caller.
    bool i8_gemm = false;
    // ...and its tier (I8GemmContext::Tier): 0 accurate (groups of 64),
    // 1 fast (the modulated blocks' projections held as rotated int8 at
    // a coarse group), 2 token (int8 per channel x per token, both
    // operands Hadamard-rotated in blocks of 256), 3 token unrotated.
    // The Perceiver and the injection cross-attention stay accurate: they
    // run once a prompt (set_text), at the text's rows.
    // VPIPE_I8_TIER / VPIPE_I8_TOKEN_ROT override it.
    int i8_tier = 0;
    // A w4 pack's block projections on the fast / token tiers: HOLD
    // their int8 layout (true; twice their bytes, the fastest) or keep
    // the pack and convert it before each GEMM into one scratch (false,
    // the default; a streamed block always does). The engine's call --
    // generate-image sets it where the box has room.
    bool i8_hold_w4 = false;
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
    // The image latents and then every context image's, ALREADY PACKED
    // into patch rows: [layout->refiner_len(), patch_dim()], bf16.
    // fibo2::pack_latents builds a picture's rows; the order inside a row
    // is (channel, dy, dx), part of the model.
    const metal_compute::SharedBuffer* latents = nullptr;
    const fibo2::Layout* layout = nullptr;
    // SIGMA in [0, 1]: 1 pure noise. The model's input is sigma * t_scale.
    float timestep = 0.0f;
    // Which text (set_text's slot) this forward reads; the layout's gist
    // count must be that text's.
    int text_slot = 0;
  };

  // The prompt's slot and a guided generation's negative's.
  static constexpr int kTextSlots = 2;

  // THE TEXT of `slot`, for every forward that reads the slot until the
  // next call: bundles [n_bundles, text_len, cap_dim] (bf16), the
  // encoder's output, held by reference until the next forward of the
  // slot has read it. That forward runs the Perceiver into
  // `gist_count(slot)` gist tokens and every injection block's keys and
  // values, on its own stream; every later one reads them. False (with
  // *err) on a buffer of the wrong size or a slot out of range.
  bool set_text(const metal_compute::SharedBuffer& bundles, int text_len,
                std::string* err = nullptr, int slot = 0);
  // Forget a slot's text (and what its pass left): a generation with no
  // negative drops the last one's, which is a prompt's worth of memory.
  void clear_text(int slot);
  // The gist count of the slot's text (0 before one) -- what the
  // layout's `gist` must be -- and the gist rows themselves [gist,
  // hidden] once a forward has made them, for a test.
  int gist_count(int slot = 0) const
  {
    return slot >= 0 && slot < kTextSlots ? _txt[slot].gist_n : 0;
  }
  const metal_compute::SharedBuffer& gist(int slot = 0) const
  {
    return _txt[slot >= 0 && slot < kTextSlots ? slot : 0].gist;
  }

  // ---- runtime LoRA --------------------------------------------------
  struct LoraSpec {
    std::string path;          // one .safetensors of lora_A/B pairs
    float       scale = 1.0f;  // 1.0 = as trained; 0 = exactly off
  };
  // Up to eight adapters at once: a distillation or identity adapter
  // beside style ones, say. Separate slots rather than one merged set
  // of factors -- concatenating on the rank axis is exact and free in
  // the forward, but it folds every strength into A and turns the live
  // knob back into a rebuild.
  static constexpr int kMaxLoraSlots = genai::lora::Stack::kMax;

  static std::unique_ptr<MetalFibo2Transformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  static std::unique_ptr<MetalFibo2Transformer>
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
  // Block projections the model holds in the fast or token layout from
  // a pack at `bits` (4 or 8), over the blocks it HOLDS (all of them,
  // preloaded): a test's proof that a tier took every width of a mixed
  // pack rather than leaving one on the accurate tier.
  int fast_weights(int bits) const;
  int slot_relayouts() const { return _slots.relaid(); }
  // Whether the block projections run in the signed int8 layout (see
  // _sgn) -- what a test asserts before it compares anything.
  bool signed_layout() const { return _sgn; }
  // Whether the modulated blocks' projections run the int8 FAST or TOKEN
  // tier (see _fast), at which group (I8GemmContext::kTokenGroup on the
  // token tier; 0 off both), and the token GEMM kernel the context took.
  bool fast_tier() const { return _fast; }
  int fast_group() const
  {
    return _fast && _i8 ? _i8->fast_group(_cfg.hidden) : 0;
  }
  std::string i8_token_kernel() const
  {
    return _i8 ? _i8->token_kernel() : std::string();
  }
  std::size_t release_resident_blocks(std::size_t bytes);
  std::uint64_t resident_bytes() const;

  bool uses_mma2() const { return _use_mma2; }
  int gemm_tile() const { return _gemm_tile; }
  bool uses_attn_nax() const { return _use_attn_nax; }
  bool uses_sage_attn() const { return (bool)_sage; }
  // How many times Sage's scratch was built: once for a run whose size
  // stays put, since each forward reserves its largest attention (the
  // main stack's) and the refiners reuse it.
  int sage_scratch_builds() const {
    return _sage ? _sage->scratch_builds() : 0;
  }
  // ...and what it holds -- all of it its own (nothing is lent), and
  // what forward_scratch_bytes() books for it.
  std::size_t sage_scratch_held() const {
    return _sage ? _sage->private_held() : 0;
  }
  // Whether the last attention Sage prepared made the int8 P*V's
  // operands (`sage_pv8`): asked for, the host's kernels there, and the
  // flash function built to read them.
  bool sage_pv8_ready() const { return _sage && _sage->pv8_ready(); }
  bool uses_i8_gemm() const { return (bool)_i8; }
  bool uses_sol_attn() const { return (bool)_sol; }
  // What the forwards have encoded for perf-flops, credited to the GPU
  // as each block retires (generative-models/shared/dit-flops.h).
  const GpuFlopsTally& gpu_flops() const noexcept { return _flops; }
  bool ane_armed() const noexcept { return _ane != nullptr; }
  bool ane_attempted() const noexcept { return _ane_tried; }
  bool ane_qkv_armed() const noexcept { return _ane_qkv != nullptr; }
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
  //    the injection blocks' cross-attention sublayers (under
  //    `layers.`) and the Perceiver are HELD in both modes and belong
  //    to the trunk. The shared helper books a pair per STEM, so naming
  //    both would promise two pairs where there is one -- and it would
  //    size the slot by an injection block, a quarter wider than the
  //    rest.
  //  * an F32 checkpoint lands as bf16 everywhere, so the bytes held
  //    are HALF the bytes on disk.
  //
  // 0 when the directory cannot be read, which the caller reads as
  // "cannot be reduced".
  static std::size_t streaming_floor_bytes(const std::string& dir);

  static std::size_t ane_runtime_bytes(const Config& c, int seq) noexcept;
  static int ane_plan_seq(int width, int height) noexcept;
  // What one forward allocates beyond the weights, for the plan's denoise
  // phase: img_rows image and context-image tokens beside a text of
  // txt_rows (<= 0: the longest), `packed` for a pack (the dense
  // stand-in), `matrix_cores` for the tiers that need them (int8, Sage,
  // Sol). A guided generation's negative (neg_txt_rows; < 0 none, 0 the
  // longest) adds what its pass leaves, and its forward's planes where
  // its joint sequence is the longer.
  static std::size_t forward_scratch_bytes(const Config& c, int img_rows,
                                           int txt_rows, bool packed,
                                           bool matrix_cores,
                                           int neg_txt_rows = -1);
  bool streaming() const { return _stream_blocks; }
  int quant_bits() const { return _quant_bits; }

 private:
  MetalFibo2Transformer() = default;

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
    // THE FAST TIER's layout (I8GemmContext::gemm_fast) when gc > 0:
    // `codes` holds int8 [N, K] in the group-rotated basis and `cscales`
    // [N, K/gc] (f32 [N] at kTokenGroup). A streamed slot also keeps the
    // raw scales / qbias its refill lands, which post_refill converts
    // from. A W4 pack's weight (bits 4) is held AS STORED -- codes,
    // scales, qbias of the pack -- and converted per GEMM
    // (fast_operand_). n / k are the weight's shape, which a refill
    // needs and the buffers do not say.
    int gc = 0;
    int n = 0, k = 0;
    metal_compute::SharedBuffer cscales;
    // A w4 pack's weight kept AS STORED (its codes / scales / qbias) and
    // converted per GEMM (fast_operand_); false for a layout held, w4 or
    // w8 (Config::i8_hold_w4).
    bool pc = false;
    // A DENSE checkpoint's weight in the fast layout in a STREAMED slot:
    // `w` keeps the bf16 a refill lands (the source, narrowed from F32
    // on Turbo) and post_refill quantizes it into codes / cscales,
    // exactly as a preloaded block's entry was built. A promotion keeps
    // only the layout, as the entry does -- the source is the slot's.
    bool ds = false;
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

  // Everything a block owns. The attention and feed-forward projections
  // are biasless; adaLN is the one Linear in a block that has a bias. An
  // injection block's cross-attention sublayer is NOT here (Inject).
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

  // A block's checkpoint prefix. `mod` indexes the streaming stack
  // (noise_refiner then layers).
  std::string mod_prefix_(int i) const;

  // `sign`: a BLOCK projection that runs at the sequence's rows, which
  // takes the signed layout when _sgn is set. The model-level linears
  // and each block's adaLN run at a row or two and stay as the
  // checkpoint has them.
  // `fast`: a MODULATED block's projection (noise refiner, main layers),
  // which takes the fast or token layout when _fast is set.
  QWeight load_qw_(WeightSet& ws, const std::string& name, Retain r,
                   bool sign = false, bool fast = false);
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
  // Is block L stored at the widths `b` was built for? (A mixed pack;
  // see BlockSlots::Ops::same_layout.)
  bool same_layout_(int L, const Block& b) const;
  // THE FAST TIER over one input for several projections (q|k|v,
  // w1|w3): one activation quantizer, then each GEMM -- what gemm_mma_
  // would run per projection, less the repeated quantizer. False with
  // NOTHING encoded unless every one is in the fast layout at one group
  // with no adapter attached (the caller then runs each as before).
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
  // A signed weight as a DENSE one, dequantized into _w_deq -- for the
  // routes that do not read the layout. `w` itself when it is not
  // signed; null when the scratch cannot be had.
  const QWeight* signed_dense_(metal_compute::ComputeEncoder& enc,
                               const QWeight& w, QWeight& view, int N,
                               int K) const;
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
  // What a PROMOTION of slot block `b` keeps (clone_block_ with copy):
  // a dense source and dead pack scales dropped, a w4 pack whose layout
  // is held converted into it. What residency and the wired pool are
  // asked for -- a slot-sized ask books bytes nothing keeps.
  std::size_t kept_block_bytes_(const Block& b) const;
  // ...and the same on a block ALREADY promoted, whatever route it came
  // by: with no slot pair (VPIPE_FIBO2_NO_SLOTS, or one that could not
  // be allocated) BlockSlots::promote_into MOVES the per-block fallback
  // rather than cloning it, so clone_block_'s copy-time work never ran.
  // Idempotent: after a clone it finds nothing to do. False when a held
  // layout could not be built (the block then keeps its pack, per call).
  bool keep_block_(Block& b) const;
  // post_refill's conversions, on a stream the reader thread keeps (a
  // fresh queue's first commit costs ~1-1.9 ms; seven a block added up).
  mutable metal_compute::CommandStream _cvt_stream;

  // The timestep sinusoid and its two Linears, on the host: 256
  // channels of one row is not worth a dispatch.
  std::vector<float> time_proj_(float t) const;
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
  void build_rope_(const fibo2::Layout& lay,
                   metal_compute::SharedBuffer* cos_out,
                   metal_compute::SharedBuffer* sin_out) const;

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  int _quant_bits = 0;
  int _quant_group = 64;
  bool _src_f32 = false;

  // Model-level weights.
  QWeight _x_in;                            // x_embedder [H, patch_dim]
  metal_compute::SharedBuffer _x_in_b;
  QWeight _time_1, _time_2;                 // t_embedder.mlp.0 / .2
  metal_compute::SharedBuffer _time_1b, _time_2b;
  QWeight _final_ada;                       // final_layer adaLN [H, 256]
  metal_compute::SharedBuffer _final_ada_b;
  QWeight _final_lin;                       // [patch_dim, H]
  metal_compute::SharedBuffer _final_lin_b;

  // A cross-attention: queries from the hidden width, keys and values
  // from a text bundle (cap_dim wide), q/k RMSNorm, no rotary, no mask.
  struct CrossAttn {
    QWeight qw, kw, vw, ow;
    metal_compute::SharedBuffer nq, nk;
  };
  // An injection block's extra sublayer, HELD in both modes (the trunk):
  // cross_attn_norm1 -> cross_attention -> cross_attn_norm2, into the
  // residual through tanh(cross_attn_gate).
  struct Inject {
    int layer = -1;                         // main-stack index
    CrossAttn ca;
    metal_compute::SharedBuffer n1, n2;     // cross_attn_norm1 / norm2
    metal_compute::SharedBuffer gate;       // [H] filled with the scalar
  };
  std::vector<Inject> _inject;
  // The Perceiver: latent_queries [max_gist, H] and its layers.
  struct PerceiverLayer {
    metal_compute::SharedBuffer n_ca, n_sa, n_ff;
    CrossAttn ca;
    QWeight sq, sk, sv, so;                 // self-attention
    metal_compute::SharedBuffer snq, snk;
    QWeight w1, w3, w2;
  };
  metal_compute::SharedBuffer _queries;
  std::vector<PerceiverLayer> _perceiver;
  metal_compute::SharedBuffer _p_norm_out;
  bool load_cross_(WeightSet& ws, const std::string& p, CrossAttn& ca);
  // What set_text leaves for the forwards, per slot: the gist rows and,
  // per injection block, its keys and values head-major [NH, T, Hd]
  // (normed), at text_len rows.
  struct TextState {
    metal_compute::SharedBuffer gist;
    int gist_n = 0;
    std::vector<metal_compute::SharedBuffer> inj_k, inj_v;
    int text_len = 0;
    // The bundles set_text was given, until a forward has run the text's
    // pass over them.
    metal_compute::SharedBuffer text;
    bool dirty = false;
    std::size_t bytes() const;
  };
  TextState _txt[kTextSlots];

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
      _fn_tr_rope, _fn_residual, _fn_residual4, _fn_gemm_t,
      // BOTH widths, picked per WEIGHT rather than per checkpoint: a
      // mixed pack (`mixed`, or an 8-bit modulation over a 4-bit body)
      // carries w4 and w8 tensors side by side, and `load_qw_` already
      // reads each one's bits off its own code/scale shapes.
      _fn_qmm4, _fn_qmm8,
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
  // THE SIGNED-CODE LAYOUT for the blocks' quantized projections: with
  // i8_gemm on a matrix-core box they run int8 x int8 off the layout
  // (I8GemmContext::gemm_signed), where they took dequant + requant.
  // Converted once: at load into the entries the set keeps, after every
  // refill in the slot. OFF under the ANE tiers, whose stagers read the
  // codes unsigned. VPIPE_I8_SIGNED=0 for the A/B.
  bool _sgn = false;
  // THE FAST OR TOKEN TIER for the MODULATED blocks' projections
  // (Config::i8_tier; both hold I8GemmContext's fast layout, at gc or
  // kTokenGroup): g64 packs converted -- in the derived entry a
  // retained block keeps, in the slot after every refill; a w4 weight is
  // then held at int8's size, twice its nibbles' -- and dense
  // checkpoints (bf16, or F32 narrowed) quantized once into the entry (a
  // streamed dense block stays on the accurate tier). Off under the ANE
  // tiers, whose stagers read the codes as affine. A GEMM the tier
  // declined after its input was quantized latches _fast_failed and the
  // forward reports it.
  bool _fast = false;
  bool _fast_failed = false;
  std::unique_ptr<MetalSageAttention> _sage;
  // Reserve Sage's scratch at the forward's largest attention (false:
  // VPIPE_FIBO2_SAGE_NO_RESERVE, the A/B -- it rebuilds per row count).
  bool _sage_reserve = true;
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

  // ---- runtime LoRA ---------------------------------------------------
  //
  // Indexed [slot][block] over the SAME index space mod_prefix_ uses.
  // Bound once at load: the factors are the adapter's, and only the
  // STRENGTH is live.
  lora::Applier _lora;
  std::vector<BlockLora> _lora_mod[kMaxLoraSlots];
  float _lora_scale[kMaxLoraSlots] = {};
  int _lora_bound[kMaxLoraSlots] = {};
  int _lora_rank_prefix[kMaxLoraSlots] = {};
  int _lora_rank_total = 0;
  int _lora_slots = 0;
  // Rows the [rows, rank] intermediate is sized for THIS forward, so
  // a slot's base offset is rows * its rank prefix.
  mutable std::size_t _lora_scratch_rows = 0;
  bool bind_loras_(const std::vector<LoraSpec>& loras);
  // The per-projection stack for one block, or an empty one when
  // nothing is attached.
  lora::Stack lora_for_(int L, lora::Factors BlockLora::* which) const;
};

}  // namespace genai
}  // namespace vpipe

#endif
