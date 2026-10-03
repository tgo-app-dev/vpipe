#ifndef GENERATIVE_MODELS_QWEN_IMAGE_QWEN_IMAGE21_TRAINER_H
#define GENERATIVE_MODELS_QWEN_IMAGE_QWEN_IMAGE21_TRAINER_H

// LoRA training for Qwen-Image-2.1: one step's forward, loss and backward
// over a MetalQwenImage21Transformer, accumulating into a LoraParams set's
// gradients.
//
// THE FORWARD IS THE MODEL'S OWN. A step runs forward() -- every route it
// has, the ANE and int8 included when configured -- with a TrainTap that
// keeps each block's input. The backward then walks the blocks in
// reverse, RECOMPUTING each from its checkpoint on the GPU path with its
// intermediates kept, and running its backward. So memory holds one
// block's intermediates rather than the stack's, at the price of a second
// forward; the recompute uses the model's own kernels and GEMM routing,
// which makes it bit-identical to the forward wherever the forward ran
// GPU-only.
//
// FROZEN BASE: every backward GEMM is dX = dY W, run through the model's
// forward GEMM route on a transposed copy of the weight, so the steel
// tiles, the matrix cores and int8 serve the backward too. q/k/v and
// gate/up are each ONE stacked GEMM. A LoRA's own terms -- u = dY B, the
// factor gradients, dX += s u A -- go through train/train-ops' small
// GEMMs, for every slot bound (a frozen adapter in another slot still
// contributes its dX), with gradients for the trained slot alone.

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "generative-models/qwen-image/qwen-image21-layout.h"
#include "generative-models/shared/ane-ffn.h"
#include "generative-models/train/lora-params.h"
#include "generative-models/train/train-ops.h"

#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {
namespace genai {

class QwenImage21Trainer {
 public:
  // One sample of a step. Text-to-image: the latents are the target's
  // alone, [target_len, in_channels] packed as the forward takes them.
  struct Sample {
    const metal_compute::SharedBuffer* x0 = nullptr;
    const metal_compute::SharedBuffer* noise = nullptr;
    const metal_compute::SharedBuffer* txt = nullptr;   // [enc_len, txt_dim]
    const qi21::Layout* layout = nullptr;
    float sigma = 0.0f;      // x_t = (1 - sigma) x0 + sigma noise
    float weight = 1.0f;     // the loss weight
  };
  struct Result {
    double loss = 0.0;
    // Wall time of the step's parts: the forward (with its checkpoints),
    // and the backward (the head, then every block's recompute and
    // backward).
    double forward_ms = 0.0;
    double backward_ms = 0.0;
    // The target rows' prediction, kept when asked (tests).
    metal_compute::SharedBuffer pred;
  };

  static std::unique_ptr<QwenImage21Trainer>
  create(MetalQwenImage21Transformer* model, std::string* err);
  // Unbinds its adapter: the model runs without it afterwards.
  ~QwenImage21Trainer();

  // The projections `targets` names -- "attn" (q, k, v, out), "attn_ff"
  // (and gate, proj, down) -- for every block, as LoraParams wants them.
  std::vector<train::ModuleShape> modules(const std::string& targets) const;

  // Bind `params` into the first free adapter slot: from here on the
  // forward reads its shadows. Returns the modules bound (0 on failure).
  int bind(train::LoraParams* params, std::string* err);

  // Forward, loss and backward for one sample. ADDS into the parameters'
  // gradients; the optimizer consumes and zeroes them.
  bool step(const Sample& s, Result* out, std::string* err,
            bool keep_pred = false);
  // The loss step() would report for `s`, from the inference forward
  // alone: no checkpoints, no backward, no gradient touched.
  bool loss(const Sample& s, double* out, std::string* err);

  // A cooperative stop, checked between blocks. A stopped step returns
  // false with an EMPTY reason: a stop is not a failure.
  void set_stop(std::function<bool()> stop) { _stop = std::move(stop); }

  // Bytes the step's arena holds at `rows` joint rows and `rank`, for
  // the resource plan.
  static std::size_t arena_bytes(const MetalQwenImage21Transformer::Config& c,
                                 int rows, int rank, bool keep) noexcept;

  // KEEP ALL: the forward keeps every block's intermediates and the
  // backward reads them, instead of recomputing each block -- about a
  // third of a step saved, for n_layers times the intermediates (see
  // arena_bytes). The forward's feed-forward then stays on the GPU.
  void set_keep_all(bool on) { _keep_all = on; }
  bool keep_all() const noexcept { return _keep_all; }

  // THE ANE FOR THE BACKWARD (the M4 series). The three large backward
  // GEMMs -- the down projection's (dX = dY W_down), the stacked gate|up
  // and the stacked q|k|v -- each split their ROWS with an ANE matmul
  // module that stages the transposed weight, exactly as the forward's
  // tiers split the feed-forward. Gradients are far below fp16's normal
  // range, so the loss is scaled by a power of two for the whole backward
  // (loss_scale) and the factor gradients divided back.
  void set_ane_backward(bool on) { _ane_bwd = on; }
  // I8 FOR THE BACKWARD: the dX GEMMs take the model's int8 route when
  // it has one (i8_gemm on a matrix-core GPU). Off keeps them on bf16
  // while the forward -- and its recompute -- stay on int8.
  void set_i8_backward(bool on) { _i8_bwd = on; }
  // Tiers created and splitting (false before the first step, and on a
  // box or geometry where every tier declined).
  bool ane_backward_armed() const noexcept;
  float loss_scale() const noexcept { return _loss_scale; }

  MetalQwenImage21Transformer* model() const noexcept { return _m; }

 private:
  using QWeight = MetalQwenImage21Transformer::QWeight;
  using BlockLora = MetalQwenImage21Transformer::BlockLora;
  using Block = MetalQwenImage21Transformer::Block;
  // One of a block's seven projections: its module leaf, its adapter in a
  // BlockLora, and whether it is in the feed-forward.
  struct Proj {
    const char* leaf;
    lora::Factors BlockLora::*lora;
    bool ff;
  };
  static constexpr int kNumProj = 7;
  static const Proj* projs_();

  QwenImage21Trainer() = default;

  bool ensure_arena_(int rows, bool keep);
  // The weight as a dense [N, K] operand: itself, or widened / dequantized
  // into a scratch.
  const metal_compute::SharedBuffer*
  dense_(metal_compute::ComputeEncoder& enc, const QWeight& w, int N, int K,
         QWeight* view);
  // Transpose `w` [N, K] into columns [col, col + N) of the [K, ld] W^T
  // scratch.
  bool transpose_in_(metal_compute::ComputeEncoder& enc, const QWeight& w,
                     int N, int K, int ld, int col);
  bool transpose_to_(metal_compute::ComputeEncoder& enc, const QWeight& w,
                     int N, int K, int ld, int col,
                     const metal_compute::SharedBuffer& dst);
  // One backward GEMM's ANE tier: dX [M, out] = dY [M, in] W'^T, W' its
  // own transposed weight [out, in].
  struct AneBack {
    std::unique_ptr<AneFeedForward> t;
    bool tried = false;
    std::string tag;
    int in = 0, out = 0;
    const metal_compute::SharedBuffer* wt = nullptr;   // its W^T
    AneFeedForward::Plan plan = AneFeedForward::Plan::kGpu;
  };
  // The ANE chunk a step of M rows splits in: the default, halved until
  // the sequence holds two -- which is what lets a 512^2 step split at
  // all -- or VPIPE_QWEN_IMAGE21_TRAIN_ANE_CHUNK.
  static int ane_chunk_for_(int M);
  bool ane_back_setup_(AneBack& a, int M, int chunk);
  void ane_back_stage_(AneBack& a, int L);
  // dX = dY W'^T for tier `a`, its rows split as planned: drains what is
  // encoded, starts the ANE on the tail rows, runs the GPU's head rows,
  // and recomputes any rows the ANE lost. `st` / `enc` are replaced.
  bool back_split_(AneBack& a, int L, metal_compute::CommandStream& st,
                   metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& dy,
                   const metal_compute::SharedBuffer& dx, int M,
                   std::string* err);
  // dX (+)= dY W for a stacked W^T already in the scratch ([K, ldw]).
  void back_gemm_(metal_compute::ComputeEncoder& enc,
                  const metal_compute::SharedBuffer& dy, int M, int ldw,
                  int K, const metal_compute::SharedBuffer& dx);
  // Every bound adapter's backward for block L's projection `which`: dX
  // terms for all, factor gradients for the trained slot.
  void lora_bwd_(metal_compute::ComputeEncoder& enc, int L,
                 lora::Factors BlockLora::*which,
                 const metal_compute::SharedBuffer& x, std::size_t xe,
                 int ldx, const metal_compute::SharedBuffer& dy,
                 std::size_t dye, int ldy,
                 const metal_compute::SharedBuffer& dx, std::size_t dxe,
                 int lddx, int M, int N, int K);

  MetalQwenImage21Transformer* _m = nullptr;
  train::TrainOps _ops;
  train::LoraParams* _p = nullptr;
  int _slot = -1;
  // Per (block, projection): the trained module, or null.
  std::vector<const train::LoraParams::Module*> _mod_of;
  std::function<bool()> _stop;
  bool _prof = false;
  bool _ane_bwd = false;
  bool _i8_bwd = true;
  float _loss_scale = 1.0f;
  // One set of the three backward tiers PER CHUNK SIZE: a module is
  // built for one chunk, and a multi-resolution run steps at several
  // sequence lengths -- tiers built at the first would split a 1024^2
  // step in 512-row chunks, measured slower than no split at all. Each
  // set: down (H -> FF), gate|up (2FF -> H), q|k|v (3H -> H). The W^T
  // scratches are shared: one step runs at a time.
  std::map<int, std::array<AneBack, 3>> _ab_sets;
  metal_compute::SharedBuffer _ab_wt[3];
  // Wall-clock accounting of one step's backward (VPIPE_..._TRAIN_TIMES).
  double _t_start_wait = 0.0, _t_drain = 0.0, _t_join = 0.0,
         _t_gpu_half = 0.0, _t_finish = 0.0, _t_end_wait = 0.0;
  std::vector<std::pair<std::string, double>> _pbuckets;
  metal_compute::ComputeLibrary _lib_deq;
  metal_compute::ComputeFunction _fn_deq4, _fn_deq8;

  int _rows = 0;
  bool _keep_all = false;
  bool _keep_alloc = false;
  // forward-side keeps
  metal_compute::SharedBuffer _ckpt, _final, _modb, _nsc, _xt, _dpred,
      _part, _zero, _rcos, _rsin;
  // the intermediates the backward reads: n_layers slices when keeping,
  // one block's when recomputing
  struct Kept {
    metal_compute::SharedBuffer h1, qpre, kpre, qh, kh, vh, ao, af, x1, h2,
        g, u, s, mlm, mls;
  } _k;
  // the recompute's own scratch
  metal_compute::SharedBuffer _vb, _qn, _kn, _tmp;
  // the backward's scratch
  metal_compute::SharedBuffer _dres, _da, _ds, _dgu, _dh, _daf, _do, _dqh,
      _dk32, _dv32, _lse2, _dsum, _dqn, _dkn, _dqkv, _wt, _deq, _lu, _lt,
      _tnorm;
};

}  // namespace genai
}  // namespace vpipe

#endif
