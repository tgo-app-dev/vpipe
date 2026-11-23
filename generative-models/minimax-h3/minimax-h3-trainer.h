#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_TRAINER_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_TRAINER_H

// LoRA training for MiniMax-H3: one sample's forward, loss and backward
// over a MetalMiniMaxH3Transformer, accumulating into a LoraParams set's
// gradients. The shape is Qwen-Image-2.1's trainer (qwen-image21-trainer.h).
//
// THE FORWARD IS THE MODEL'S OWN. A step runs forward() with a TrainTap
// (Step::train) that keeps each main block's input, each block's
// modulation table, the final hidden state, the final layer's modulation
// and the token refiner's inputs and output -- every route the forward
// has (dense, FP8 / int8 / NVFP4 / W4A8, pruned AdaLN, the int8 tiers,
// the ANE) stays on, Sol and Sage are skipped. The backward then walks
// the blocks in reverse, RECOMPUTING each from its checkpoint with the
// model's own kernels and GEMM route, and running its backward; then,
// when the refiner is trained, the refiner's two blocks over the text
// rows.
//
// THE LOSS (sample's two modalities, each its own mean): the velocity
// target of H3 is v = x0 - noise (the scheduler's x0 = x + sigma v), on
// the GENERATED video rows and the generated audio rows; text and
// conditioning rows never enter it:
//   L = w * (mean_video (v^ - v)^2 + lambda * mean_audio (v^ - v)^2)
// lambda 0 (a clip without its soundtrack) keeps the audio rows in the
// sequence, as inference does, and out of the loss.
//
// FROZEN BASE: every backward GEMM is dX = dY W, run through the model's
// gemm_ on a transposed copy of the weight (widened first where it is
// stored narrow), so the steel tiles and the matrix cores serve the
// backward too. q|k|v is ONE fused projection in the base's own row
// grouping (per head or flat); a LoRA's terms run through train-ops'
// small GEMMs for every slot bound -- a frozen adapter in another slot
// contributes its dX -- with gradients for the trained slot alone.
//
// THE ADAPTER IS THE DIFFUSERS DECOMPOSITION (the spelling H3's binder
// reads and fuses itself, portable across both qkv groupings):
//   transformer_blocks.<L>.attn.to_q | to_k | to_v | to_out.0
//   transformer_blocks.<L>.ff.net.0.proj   (diffusers' [value; gate])
//   transformer_blocks.<L>.ff.net.2
//   token_refiner.refiner_blocks.<R>....  (the same leaves)
// In TRAINING the factors live in the forward's order: q, k and v as ONE
// banded factor (A stacked, B in the base's fused row order -- the
// binder's qkv_fused_row), fc1's B gate first. LoraParams' row maps
// (Module::b_rows) put the file's rows there and back, so an adapter
// saved and bound again by the model's binder reproduces exactly the
// factors the trainer ran.

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/train/lora-params.h"
#include "generative-models/train/train-ops.h"

#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MiniMaxH3Trainer {
 public:
  // One sample: the layout generate-video would build for the request,
  // its rows as the forward reads them.
  struct Sample {
    const minimax_h3::PackedLayout* layout = nullptr;
    // The conditioning, [num_text_rows, text_dim] bf16 (the encoder tap).
    const metal_compute::SharedBuffer* text = nullptr;
    // The GENERATED rows, bf16: video [num_video_rows, 96] and audio
    // [num_audio_rows, audio_channels], each in the layout's own order.
    const metal_compute::SharedBuffer* video_x0 = nullptr;
    const metal_compute::SharedBuffer* video_noise = nullptr;
    const metal_compute::SharedBuffer* audio_x0 = nullptr;
    const metal_compute::SharedBuffer* audio_noise = nullptr;
    // The CONDITIONING rows, bf16, as the forward reads them (already at
    // their level): video [num_condition_video_rows, 96], audio
    // [num_condition_audio_rows, audio_channels]; null when the layout
    // has none.
    const metal_compute::SharedBuffer* video_cond = nullptr;
    const metal_compute::SharedBuffer* audio_cond = nullptr;
    // x_t = (1 - sigma) x0 + sigma noise, per modality; the rows'
    // timesteps are 1 - sigma (build_row_timesteps).
    float sigma_video = 0.0f;
    float sigma_audio = 0.0f;
    // The conditioning rows' timesteps (condition_timestep and
    // condition_audio_timestep).
    float cond_video_t = 1.0f;
    float cond_audio_t = 1.0f;
    float weight = 1.0f;         // the loss weight
    float audio_weight = 1.0f;   // lambda; 0 leaves the audio out
    int out_head = -1;           // Step::out_head (MiniMax-H3-Image)
  };
  struct Result {
    double loss = 0.0;
    double loss_video = 0.0;     // the two means, unweighted
    double loss_audio = 0.0;
    double forward_ms = 0.0;
    double backward_ms = 0.0;
    // The generated rows' velocities, kept when asked (tests).
    metal_compute::SharedBuffer pred_video, pred_audio;
  };

  static std::unique_ptr<MiniMaxH3Trainer>
  create(MetalMiniMaxH3Transformer* model, std::string* err);
  // Unbinds its adapter: the model runs without it afterwards.
  ~MiniMaxH3Trainer();

  // The projections `targets` names -- "attn" (to_q, to_k, to_v,
  // to_out.0), "attn_ff" (and ff.net.0.proj, ff.net.2) -- for every main
  // block and, with `refiner`, both token-refiner blocks, in the order
  // LoraParams must keep them (q, k, v adjacent).
  std::vector<train::ModuleShape> modules(const std::string& targets,
                                          bool refiner = true) const;
  // The module path of block L's projection (`refiner`: a token-refiner
  // block), e.g. "transformer_blocks.3.attn.to_q".
  static std::string module_name(bool refiner, int L, const char* leaf);

  // Bind `params` into an adapter slot after every file's: from here on
  // the forward reads its shadows. Sets the q/k/v and fc1 modules' row
  // maps. Returns the modules bound (0 on failure).
  int bind(train::LoraParams* params, std::string* err);

  // Forward, loss and backward for one sample. ADDS into the parameters'
  // gradients; the optimizer consumes and zeroes them.
  bool step(const Sample& s, Result* out, std::string* err,
            bool keep_pred = false);
  // The loss step() would report, from the inference forward alone: no
  // tap, no backward, no gradient touched.
  bool loss(const Sample& s, Result* out, std::string* err);

  // A cooperative stop, checked between blocks. A stopped step returns
  // false with an EMPTY reason: a stop is not a failure.
  void set_stop(std::function<bool()> stop) { _stop = std::move(stop); }
  // The int8 route for the backward's dX GEMMs (on by default where the
  // forward has one); off keeps them bf16.
  void set_i8_backward(bool on) { _i8_bwd = on; }
  // RECOMPUTE: NEVER. The forward runs each main block through the
  // trainer's own block forward (TrainTap::run_block) and keeps what its
  // backward reads, so the backward recomputes no main block (the token
  // refiner's two blocks over the text rows still are). Costs
  // keep_block_bytes a block; the default recomputes.
  void set_keep_all(bool on)
  {
    if (on != _keep_all) {
      _keep_all = on;
      _seq = _ntext = _nt = 0;   // the arena is laid out again
    }
  }
  bool keep_all() const noexcept { return _keep_all; }

  // Bytes a step's arena holds at `rows` joint rows, `n_text` of them
  // text, `n_t` distinct timesteps and the adapter's `rank`, for the plan
  // (`keep_all`: every main block's activations kept).
  static std::size_t arena_bytes(const MetalMiniMaxH3Transformer::Config& c,
                                 int rows, int n_text, int n_t, int rank,
                                 bool keep_all = false);
  // One main block's kept activations at `rows` rows.
  static std::size_t keep_block_bytes(
      const MetalMiniMaxH3Transformer::Config& c, int rows);

  MetalMiniMaxH3Transformer* model() const noexcept { return _m; }
  int slot() const noexcept { return _slot; }

 private:
  using M = MetalMiniMaxH3Transformer;
  using Linear = M::Linear;
  using Block = M::Block;
  using BlockLora = M::BlockLora;
  using LoraFactors = lora::Factors;

  MiniMaxH3Trainer() = default;

  // The trained slot's factors of one projection: offsets into the
  // arenas, and whether B is the banded q|k|v.
  struct Trained {
    bool on = false;
    std::size_t a_off = 0, b_off = 0;
    bool banded = false;
  };
  Trained& trained_(bool refiner, int L, LoraFactors BlockLora::* which);

  bool ensure_arena_(int seq, int n_text, int n_t);
  // `l` as a dense bf16 [N, K] operand: itself, or widened / dequantized
  // into a scratch (the model's for FP8 and the int8 layouts).
  const metal_compute::SharedBuffer*
  dense_(metal_compute::ComputeEncoder& enc, const Linear& l, int N, int K,
         Linear* view);
  // W^T [K, N] of `l` [N, K] into the scratch, then dX [M, K] (=)
  // dY [M, N] W through the model's gemm_.
  bool back_gemm_(metal_compute::ComputeEncoder& enc, const Linear& l,
                  int N, int K, const metal_compute::SharedBuffer& dy,
                  std::size_t dye, const metal_compute::SharedBuffer& dx,
                  std::size_t dxe, int M);
  // Every bound adapter's backward for one projection of block L: the dX
  // terms of all, the factor gradients of the trained slot.
  void lora_bwd_(metal_compute::ComputeEncoder& enc, bool refiner, int L,
                 LoraFactors BlockLora::* which,
                 const metal_compute::SharedBuffer& x, int ldx,
                 const metal_compute::SharedBuffer& dy, int ldy,
                 const metal_compute::SharedBuffer& dx, int lddx, int M,
                 int N, int K);
  // What a block's backward reads of its forward: the scratch (one
  // block's, recomputed) or a main block's kept set.
  struct Acts {
    metal_compute::SharedBuffer h1, x1, h2, qkvp, qh, kh, vh, oh, ob, gu, sw,
        mlm, mls;
  };
  Acts scratch_acts_() const;
  Acts keep_acts_(int L) const;
  // One block (from `x`, `rows` rows) into `enc`: kRecompute its forward
  // into `a` then its backward; kForward its forward into `a`, the
  // block's output written over x; kBackward its backward from `a`. The
  // residual gradient in `dres` (f32 [rows, H]) is read and replaced.
  enum class Pass { kRecompute, kForward, kBackward };
  bool block_pass_(metal_compute::ComputeEncoder& enc, Pass pass,
                   bool refiner, int L, const Block& b,
                   const metal_compute::SharedBuffer& x, std::size_t xe,
                   int rows, const metal_compute::SharedBuffer& mod,
                   std::size_t me, const metal_compute::SharedBuffer& dres,
                   const Acts& a, std::string* err);
  // A streamed main block, read into a slot (the model's pair) or `local`;
  // and its predecessor's read issued under the current block's work.
  const Block* acquire_(int L, Block& local, std::string* err);
  void prefetch_(int L);
  // x_t and the conditioning into the forward's row buffers; the Step.
  bool inputs_(const Sample& s, std::string* err);

  MetalMiniMaxH3Transformer* _m = nullptr;
  train::TrainOps _ops;
  train::LoraParams* _p = nullptr;
  int _slot = -1;
  std::vector<Trained> _tr;   // [(n_layers + n_refiner) * 4]
  bool _train_refiner = false;
  std::function<bool()> _stop;
  bool _i8_bwd = true;
  bool _keep_all = false;
  // Each main block's kept activations (keep_acts_ views), laid out for
  // _keep_rows rows.
  std::vector<metal_compute::SharedBuffer> _keep;
  int _keep_rows = 0;
  metal_compute::ComputeLibrary _lib_deq;
  metal_compute::ComputeFunction _fn_deq4, _fn_deq8;

  // The streamed predecessor in flight.
  std::future<bool> _pf;
  int _pf_layer = -1, _pf_slot = -1;

  // ---- per step ----------------------------------------------------------
  std::vector<float> _uniq;
  std::vector<int> _row_idx;
  metal_compute::SharedBuffer _xv, _xa;          // the forward's rows
  metal_compute::SharedBuffer _aidx, _tidx;      // adaln / timestep index
  metal_compute::SharedBuffer _rcos, _rsin;      // [seq, rot / 2] f32

  int _seq = 0, _ntext = 0, _nt = 0;
  // forward-side keeps
  metal_compute::SharedBuffer _ckpt, _mods, _fmod, _final, _ref_in,
      _ref_out;
  // one block's recompute
  metal_compute::SharedBuffer _h1, _qkvp, _qkvn, _qh, _kh, _vh, _oh, _ob,
      _ao, _x1, _h2, _gu, _sw, _mlm, _mls;
  // the backward's scratch
  metal_compute::SharedBuffer _dres, _dres_t, _df, _ds, _dgu, _dh, _dob,
      _doh, _dqh, _dk32, _dv32, _lse2, _dsum, _dproj, _dpv, _dpa, _part,
      _wt, _deq, _lu, _lt, _bbd, _dbbd, _zero;
};

}  // namespace genai
}  // namespace vpipe

#endif
