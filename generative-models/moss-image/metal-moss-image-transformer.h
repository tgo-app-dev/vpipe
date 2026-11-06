#ifndef GENERATIVE_MODELS_MOSS_IMAGE_METAL_MOSS_IMAGE_TRANSFORMER_H
#define GENERATIVE_MODELS_MOSS_IMAGE_METAL_MOSS_IMAGE_TRANSFORMER_H

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
#include <initializer_list>
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

// MOSS-Image's DiT (OpenMOSS MOSSImageTransformer), on metal-compute.
//
// A 5B SINGLE-stream flow-matching stack whose reference is the one
// pipeline_moss_image.py the checkpoint ships. The sequence is
//
//   [ image tokens (grid_h * grid_w, row-major) | text tokens ]
//
// IMAGE FIRST. The text half is the encoder's hidden states projected
// to the model width and refined by TWO unmodulated text blocks over the
// text alone; then 42 modulated blocks run over the joint sequence and
// only the image rows reach the head.
//
// FIVE THINGS ABOUT IT ARE EACH A SILENT FAILURE WHEN DROPPED:
//
//  * ONE MODULATION FOR ALL 42 BLOCKS. `shared_block_modulation` maps
//    the time embedding to six vectors (shift, scale, gate) x (attention,
//    feed-forward) ONCE per forward, and every block reads the same six.
//    There is no per-block adaLN at all -- and no SiLU in front of that
//    Linear, nor in front of the head's `video_head_modulation`.
//  * THE GATES ARE PLAIN: x += o * gate, no tanh, no (1 + gate).
//  * THE ACTIVATION IS SITU, not SwiGLU:
//    (4 tanh(g/4) sigmoid(g)) * (25 tanh(h/25)), and the fused
//    ffn_up_gate is [hidden | gate] -- hidden FIRST.
//  * THE ROTARY IS FOUR AXES (r, f, h, w) of head_dim/8 = 16 bands each,
//    NeoX half-split over the WHOLE head (dim p pairs with p + 64). The
//    image rows sit at (0, text_len, h - (H-1)/2, w - (W-1)/2) -- the
//    text length is a COORDINATE of every image row, and h / w are
//    centred, half-integers on an even grid -- and the text rows at
//    (0, i, 0, 0). The reference rounds the table to the activation
//    dtype (bf16), which is what it was trained with; so does this.
//  * THE TIMESTEP SINUSOID IS [cos | sin], cosine first, over the
//    scheduler's own t = sigma * 1000 in f32 (see `Request::timestep`).
//
// ATTENTION IS DENSE AND BIDIRECTIONAL, one dispatch per block over the
// joint sequence -- the reference evaluates each sample at its TRUE text
// length, so there are no pad rows and no mask. The two text blocks
// attend over the text rows alone.
//
// TOKEN-MAJOR ATTENTION. q/k/v stay [rows, heads, 128] as the
// projections wrote them: the per-head RMSNorm and the rotation run in
// place, and the flash kernel reads them -- and writes its output --
// through a head stride and a row stride, so no transpose runs before or
// after it. Sol reads the head-major layout and takes the transposes.
class MetalMossImageTransformer {
 public:
  ~MetalMossImageTransformer();

  struct Config {
    int hidden       = 3072;     // `dim`
    int n_heads      = 24;
    int head_dim     = 128;      // dim / n_heads
    int n_layers     = 42;       // the modulated stack
    int n_text_blocks = 2;       // text_embedder.blocks
    int ffn_inner    = 8192;     // `ffn_dim`
    int latent_channels = 128;   // the FLUX.2 VAE's 32 x 2 x 2
    int text_dim     = 4096;     // the encoder's width
    int freq_dim     = 256;      // the timestep sinusoid
    float rope_base  = 10000.0f;
    float norm_eps   = 1e-6f;    // every RMSNorm in the model

    // Accel tiers, read from accel-settings.h by the caller.
    bool i8_gemm = false;
    // Whether the kernel tuners MEASURE (the accel key `autotune`,
    // accel-keys.h): false takes the choice believed fastest on this
    // GPU, timing nothing.
    bool autotune = true;
    // ...and its tier (I8GemmContext::Tier): 0 accurate (groups of 64),
    // 1 fast (the modulated blocks' projections held as rotated int8 at
    // a coarse group), 2 token (int8 per channel x per token, Hadamard
    // rotated), 3 token unrotated. The text blocks stay accurate: their
    // rows sit under the int8 floor. VPIPE_I8_TIER overrides it.
    int i8_tier = 0;
    // A w4 pack's block projections on the fast / token tiers: HOLD
    // their int8 layout (true; twice their bytes, the fastest) or keep
    // the pack and convert it before each GEMM into one scratch (false,
    // the default; a streamed block always does). The engine's call --
    // generate-image sets it where the box has room.
    bool i8_hold_w4 = false;
    sage::Config sage;
    sol::Config  sol;
    // THE NEURAL ENGINE TIERS (the M4 series' lever): the main blocks'
    // SITU feed-forward, and q / k / v as one stacked matmul (which
    // REQUIRES the feed-forward's). The ANE takes the LAST rows of a
    // block, the GPU the head rows, concurrently. Off the int8 layouts:
    // the stagers read the codes as affine.
    bool ane_ffn    = false;
    // The ANE's share of the rows in [0,1] (accel-keys.h kAneRows): 0
    // balances from the two engines' rates, a share set is pinned.
    float ane_rows = 0.0f;
    int  ane_layers = 0;    // 0 = every main block
    bool ane_qkv    = false;
    // The ANE's modules in ROTATED INT8 (W8A8; AneFeedForward::Options::
    // int8): generate-image's `ane_int8`. LOSSY -- the GPU token tier's
    // arithmetic on the ANE's rows; a module that cannot take it keeps
    // fp16 and says so. VPIPE_ANE_INT8 overrides it.
    bool ane_int8   = false;
    // ...unrotated: the GPU token_plain tier's arithmetic (generate-image's
    // `ane_int8_tier: token_plain`). Only with ane_int8.
    bool ane_int8_plain = false;

    // Read <dir>/config.json. False when the file is absent or names a
    // different class, or the shape is one this does not run.
    static bool read_dims(const std::string& dir, Config* out);
  };

  struct Request {
    // The noisy latent, TOKEN-major: [grid_h * grid_w, latent_channels]
    // bf16, row r * grid_w + c -- what the reference's
    // `flatten(2).transpose(1, 2)` makes of [C, H, W].
    const metal_compute::SharedBuffer* latents = nullptr;
    int grid_h = 0, grid_w = 0;
    // The text encoder's final hidden states, one row per REAL token:
    // [text_rows, text_dim] bf16.
    const metal_compute::SharedBuffer* text = nullptr;
    int text_rows = 0;
    // THE SCHEDULER'S TIMESTEP, sigma * 1000 formed in f32 exactly as
    // diffusers forms it -- not sigma. The sinusoid is taken in f64 off
    // that f32, so the multiply has to be the reference's own.
    float timestep = 0.0f;
    // Main blocks to SKIP, zero-based. The reference's default negative
    // branch runs without block 22; a skipped block is never read, so a
    // streamed stack does not fetch it either.
    std::vector<int> skip_layers;
  };

  // ---- runtime LoRA --------------------------------------------------
  struct LoraSpec {
    std::string path;          // one .safetensors of lora_A/B pairs
    float       scale = 1.0f;  // 1.0 = as trained; 0 = exactly off
  };
  static constexpr int kMaxLoraSlots = genai::lora::Stack::kMax;

  static std::unique_ptr<MetalMossImageTransformer>
  load(const std::string& model_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  static std::unique_ptr<MetalMossImageTransformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  // LIVE; exactly 0 encodes nothing at all.
  void set_lora_scale(int slot, float s);
  int lora_modules(int slot) const;

  // One forward. Returns the IMAGE rows' velocity, TOKEN-major --
  // [grid_h * grid_w, latent_channels] bf16. Empty on failure, and ALSO
  // empty with no error on a cooperative stop.
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
  std::size_t release_resident_blocks(std::size_t bytes);
  std::uint64_t resident_bytes() const;
  bool streaming() const { return _stream_blocks; }
  int quant_bits() const { return _quant_bits; }

  bool signed_layout() const { return _sgn; }
  bool fast_tier() const { return _fast; }
  int fast_group() const
  {
    return _fast && _i8 ? _i8->fast_group(_cfg.hidden) : 0;
  }
  std::string i8_token_kernel() const
  {
    return _i8 ? _i8->token_kernel() : std::string();
  }
  bool uses_mma2() const { return _use_mma2; }
  int gemm_tile() const { return _gemm_tile; }
  bool uses_attn_nax() const { return _use_attn_nax; }
  bool uses_sage_attn() const { return (bool)_sage; }
  bool uses_i8_gemm() const { return (bool)_i8; }
  bool uses_sol_attn() const { return (bool)_sol; }
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
  // What the ANE tiers hold for a `seq`-row forward (the feed-forward's
  // module, plus the q|k|v one when that tier is on), and the joint
  // sequence a picture implies, for the plan's claim.
  static std::size_t ane_runtime_bytes(const Config& c, int seq) noexcept;
  static int ane_plan_seq(int width, int height) noexcept;

  // THE STREAMING FLOOR: the trunk (model-level weights and the two text
  // blocks, held in both modes) + ONE slot pair of main blocks. 0 when
  // the directory cannot be read.
  static std::size_t streaming_floor_bytes(const std::string& dir);
  // The activations one forward allocates at this geometry -- what the
  // denoise phase has to book beside the weights.
  // `packed` books a pack's dense stand-in; `matrix_cores` the int8,
  // Sage and Sol scratch (Config::i8_gemm / sage / sol) held between
  // forwards on a box that runs them.
  static std::size_t forward_scratch_bytes(const Config& c, int grid_h,
                                           int grid_w, int text_rows,
                                           bool sol, bool packed = false,
                                           bool matrix_cores = false);

 private:
  MetalMossImageTransformer() = default;

  enum class Retain { Cached, Streamed };

  struct QWeight {
    metal_compute::SharedBuffer w;                    // dense [N,K]
    metal_compute::SharedBuffer codes, scales, qbias; // affine quant
    bool quantized = false;
    int  bits = 0;
    // The SIGNED layout (I8GemmContext::gemm_signed); see _sgn.
    bool sgn = false;
    // THE FAST TIER's layout (I8GemmContext::gemm_fast) when gc > 0.
    int gc = 0;
    int n = 0, k = 0;
    metal_compute::SharedBuffer cscales;
    // A w4 pack's weight kept AS STORED (its codes / scales / qbias) and
    // converted per GEMM (fast_operand_); false for a layout held, w4 or
    // w8 (Config::i8_hold_w4).
    bool pc = false;
    // A DENSE checkpoint's weight in the fast layout in a STREAMED slot:
    // `w` keeps the raw bf16 a refill lands (the source) and post_refill
    // quantizes it into codes / cscales, exactly as a preloaded block's
    // entry was built. A promotion keeps only the layout, as the entry
    // does -- the source is the slot's alone.
    bool ds = false;
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

  // A main block. Every projection is biasless and nothing in it is
  // modulated by a weight of its own -- the shared modulation is the
  // model's.
  struct Block {
    QWeight qw, kw, vw, ow;                  // proj.q / k / v / o
    QWeight ug;                              // proj.ffn_up_gate [2F, H]
    QWeight dn;                              // proj.ffn_down    [H, F]
    metal_compute::SharedBuffer n1, n2;      // norm1 / norm2 [H]
    metal_compute::SharedBuffer nq, nk;      // q_norm / k_norm [head_dim]
  };

  // A text block. Every Linear carries a bias, q|k|v arrive FUSED in the
  // checkpoint (q, k, v by columns), and the feed-forward keeps ffn_up
  // (the HIDDEN half) and ffn_gate apart -- two GEMMs into the two-input
  // SITU. Always held.
  struct TextBlock {
    QWeight qkv;  metal_compute::SharedBuffer qkv_b;  // [3H, H]
    QWeight proj; metal_compute::SharedBuffer proj_b; // [H, H]
    QWeight up;   metal_compute::SharedBuffer up_b;   // [F, H]
    QWeight gate; metal_compute::SharedBuffer gate_b; // [F, H]
    QWeight down; metal_compute::SharedBuffer down_b; // [H, F]
    metal_compute::SharedBuffer n1, n2, nq, nk;
  };

  struct BlockLora {
    lora::Factors q, k, v, o, ug, dn;
    bool empty() const {
      return q.empty() && k.empty() && v.empty() && o.empty() &&
             ug.empty() && dn.empty();
    }
  };

  std::string prefix_(int L) const;

  QWeight load_qw_(WeightSet& ws, const std::string& name, Retain r,
                   bool sign = false, bool fast = false);
  bool sign_(QWeight& w) const;
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
  struct FastOut {
    const QWeight* w;
    const metal_compute::SharedBuffer* y;
    std::size_t ye;
    int n;
    bool adapted;
  };
  bool fast_shared_(metal_compute::ComputeEncoder& enc,
                    const metal_compute::SharedBuffer& x, std::size_t xe,
                    int M, int K, std::initializer_list<FastOut> outs);
  const QWeight* signed_dense_(metal_compute::ComputeEncoder& enc,
                               const QWeight& w, QWeight& view, int N,
                               int K) const;
  metal_compute::SharedBuffer to_elt_(WeightSet& ws, const std::string& name,
                                      Retain r);
  bool load_block_(WeightSet& ws, int L, Block& b, Retain r);
  bool load_text_block_(WeightSet& ws, int i, TextBlock& b);
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
  // What a PROMOTION of slot block `b` keeps (clone_block_ with copy):
  // a dense source and dead pack scales dropped, a w4 pack whose layout
  // is held converted into it. What residency and the wired pool are
  // asked for -- a slot-sized ask books bytes nothing keeps.
  std::size_t kept_block_bytes_(const Block& b) const;
  // ...and the same on a block ALREADY promoted, for the promotion that
  // MOVES rather than clones (BlockSlots with no slots -- off, or a slot
  // that could not be allocated -- hands its per-block fallback over as
  // is, so clone_block_'s copy-time work never runs): a dense source
  // dropped, dead pack scales dropped, a held w4 layout converted, all in
  // one command buffer. Idempotent -- after a clone it finds nothing.
  bool keep_block_(Block& b) const;
  static std::size_t text_block_bytes_(const TextBlock& b);
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
  // The rotary table rounded to bf16 as the reference does (default), or
  // kept f32 (VPIPE_MOSS_IMAGE_ROPE_F32=1, the A/B).
  bool _rope_bf16 = true;

  // Model-level weights: video_embedder, text_embedder.proj,
  // time_embedding.0 / .2, shared_block_modulation, video_head_modulation,
  // final_norm_video and final_linear_video.
  QWeight _x_in;  metal_compute::SharedBuffer _x_in_b;
  QWeight _t_in;  metal_compute::SharedBuffer _t_in_b;
  QWeight _time_1, _time_2;
  metal_compute::SharedBuffer _time_1b, _time_2b;
  QWeight _mod;   metal_compute::SharedBuffer _mod_b;
  QWeight _head_mod; metal_compute::SharedBuffer _head_mod_b;
  metal_compute::SharedBuffer _final_norm;
  QWeight _final_lin;

  std::vector<TextBlock> _text;     // ALWAYS held
  std::vector<Block> _blocks;       // the modulated stack
  bool _stream_blocks = false;
  BlockSlots<Block> _slots;
  BlockResidency _resid;
  WiredPool _wire;
  std::shared_ptr<WeightSet> _ws;
  std::function<bool()> _stream_stop;
  DitBlockProgressFn _block_progress;
  std::vector<std::uint8_t> _narrow_scratch;
  // post_refill's conversions, on a stream the reader thread keeps (a
  // fresh queue's first commit costs ~1-1.9 ms; six a block added up).
  mutable metal_compute::CommandStream _cvt_stream;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_rope,
      _lib_attn, _lib_attn_nax, _lib_mma, _lib_dequant, _lib_qmm;
  metal_compute::ComputeFunction _fn_adaln, _fn_adaln4, _fn_gated,
      _fn_gated4, _fn_rms, _fn_situ, _fn_situ2, _fn_rms_rope, _fn_residual,
      _fn_residual4, _fn_gemm_t, _fn_qmm4, _fn_qmm8, _fn_dequant4,
      _fn_dequant8, _fn_gemm_mma, _fn_gemm_mma_deep, _fn_gemm_mma_tn2,
      _fn_gemm_bm64, _fn_gemm_bm64bn64, _fn_silu, _fn_transpose,
      _fn_bias_add;
  std::map<unsigned, metal_compute::ComputeFunction> _attn_fn;
  bool _use_mma2 = false;
  bool _use_attn_nax = false;
  bool _steel_attn_ok = false;
  int  _mma_min_m = 64;
  int  _gemm_tile = 2;
  // VPIPE_MOSS_IMAGE_BAND_ROWS: a test's cap on the matmul2d row band.
  int  _band_cap = 0;
  // Token-major attention (default); VPIPE_MOSS_IMAGE_NO_TM=1 for the
  // head-major A/B.
  bool _attn_tm = true;
  static constexpr int kSageMinSegment = 512;
  static constexpr int kSolMinSegment = 1024;
  // How many blocks a NON-streamed stack may encode before it drains,
  // so a stop that arrives while the GPU is busy is seen at all (see
  // Z-Image's twin of this constant for the measurement).
  static constexpr int kStopDrainBlocks = 4;
  mutable metal_compute::SharedBuffer _w_deq;
  bool _sgn = false;
  bool _fast = false;
  bool _fast_failed = false;
  std::unique_ptr<MetalSageAttention> _sage;
  std::unique_ptr<MetalSolAttention> _sol;
  GpuFlopsTally _flops;
  SolFlops _sol_flops;
  std::unique_ptr<I8GemmContext> _i8;
  std::unique_ptr<AneFeedForward> _ane;
  bool _ane_tried = false;
  std::unique_ptr<AneFeedForward> _ane_qkv;
  bool _ane_qkv_tried = false;

  // ---- runtime LoRA ---------------------------------------------------
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

// ---- the reference's sampler arithmetic ------------------------------
namespace moss_image {

// The image is 16 pixels a token: the VAE's 8 times its 2x2 patch.
constexpr int kPixelsPerToken = 16;
// The reference caps a prompt at 4096 tokens (truncating, not refusing).
constexpr int kMaxTextTokens = 4096;

// MOSSImageScheduler.set_timesteps: a shift-`shift` grid over
// linspace(1, lower, steps), snapped to the NEAREST node of the
// training grid linspace(1, 1/train, train) (training_shift 1), with
// consecutive duplicates removed -- then diffusers' Euler appends 0.
// Returns the sigmas in f32, the trailing 0 included, so a schedule of
// S steps is S + 1 long (shorter only if the snap merged nodes).
std::vector<float> sigmas(int steps, double shift, double training_shift = 1.0,
                          int train_steps = 1000);

// The timestep the DiT is handed for a sigma: sigma * 1000 in f32, the
// diffusers tensor multiply.
inline float timestep(float sigma) { return sigma * 1000.0f; }

// The rotary table for one forward, f32 [img + text, head_dim / 2] cos
// and sin: entry (row, p) is axis p / bands at band p % bands, the
// reference's [sin(4 x bands) | cos(4 x bands)] split into two tables.
// `round_bf16` rounds each value to bf16 as the reference's
// `.to(video.dtype)` does.
void rope_tables(int grid_h, int grid_w, int text_rows, int head_dim,
                 float base, bool round_bf16, std::vector<float>* cos_out,
                 std::vector<float>* sin_out);

}  // namespace moss_image

}  // namespace genai
}  // namespace vpipe

#endif
