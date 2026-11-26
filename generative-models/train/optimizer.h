#ifndef VPIPE_GENERATIVE_MODELS_TRAIN_OPTIMIZER_H
#define VPIPE_GENERATIVE_MODELS_TRAIN_OPTIMIZER_H

// The update rule and its learning-rate schedule, over a LoraParams set.
//
// OptimizerSpec is what the optimizer-select stage emits and train-lora
// reads, round-tripping through FlexData like FlowSamplerSpec. The
// schedule is called `lr_schedule` because "scheduler" already means the
// noise schedule everywhere else in this tree.
//
// Optimizer runs one update per call to step(): the global gradient norm
// (for clipping, and reported), then the update over the A and B arenas,
// each of which also zeroes its gradient and rewrites the forward's
// shadow factors. Prodigy's step size is a GLOBAL quantity, so it takes a
// second pass; the others are one.

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "generative-models/train/lora-params.h"
#include "generative-models/train/train-ops.h"

#include <functional>
#include <string>
#include <string_view>

namespace vpipe {
namespace genai {
namespace train {

struct OptimizerSpec {
  // adamw | adamw8bit | adafactor | prodigy | lion | sgd. adamw8bit is
  // AdamW with its two moments held as one-byte block-quantized codes (2
  // bytes a parameter against 8; the update itself in f32). adafactor is
  // transformers' Adafactor with a manual learning rate (relative_step,
  // scale_parameter and the first moment off; decay -0.8, eps 1e-30,
  // update clipped to RMS 1): each factor's squared-gradient moment kept
  // as a row and a column vector -- almost no state.
  std::string optimizer = "adamw";
  double lr = 0.0;                        // 0: the optimizer's default
  double beta1 = 0.9;
  double beta2 = 0.999;
  double eps = 1e-8;
  double weight_decay = 0.01;
  // auto | constant | linear | cosine | cosine_restarts. auto is the
  // trainer's to resolve by the dataset: constant for a small one (trained
  // by steps), cosine for a large one (trained by epochs).
  std::string lr_schedule = "auto";
  int warmup_steps = -1;                  // -1: 2% of the run
  double min_lr_ratio = 0.0;
  int restarts = 1;
  double grad_clip = 1.0;                 // global L2; 0 disables
  double ema = 0.0;                       // EMA decay of the weights; 0 off
  double prodigy_d_coef = 1.0;
  double prodigy_d0 = 1e-6;
  double prodigy_growth = 1e30;           // how fast d may grow a step
  bool bias_correction = false;           // Prodigy's use_bias_correction
  // LoRA+ (Hayou et al.): the B factors learn this many times faster
  // than the A factors (kohya's loraplus_lr_ratio; 16 its paper's
  // suggestion). 1: off.
  double lora_plus_ratio = 1.0;

  // The choice lists the stage offers, in the editor's order.
  static constexpr const char* kOptimizers =
      "adamw,adamw8bit,adafactor,prodigy,lion,sgd";
  static constexpr const char* kSchedules =
      "auto,constant,linear,cosine,cosine_restarts";

  // The learning rate a spec with lr == 0 runs at.
  double effective_lr() const noexcept;
  // Warmup steps for a run of `total` steps.
  int effective_warmup(int total) const noexcept;

  FlexData to_flex() const;
  // False, with a reason, for an unknown optimizer or schedule or an
  // out-of-range value; `out` untouched then.
  static bool from_flex(const FlexData& f, OptimizerSpec* out,
                        std::string* err);
  static bool is_optimizer_spec(const FlexData& f);
};

class Optimizer {
 public:
  bool init(const TrainOps* ops, LoraParams* params,
            const OptimizerSpec& spec, int total_steps, std::string* err);

  // The learning rate the schedule gives update number `t` (from 0).
  double lr_at(int t) const noexcept;

  // One update. `accum` is how many micro-steps the gradients hold (they
  // are averaged). Commits and waits twice (the norm, then the update);
  // reports the pre-clip gradient norm and the learning rate used.
  bool step(int accum, double* grad_norm, double* lr_used, std::string* err);

  // A SPLIT RUN's global norm: this rank's sum of squares in, every
  // rank's out -- each rank's share counted once, the shared factors
  // once -- so every rank clips by the same norm. Unset: the local sum.
  using GlobalSq = std::function<bool(double local, double* global,
                                      std::string* err)>;
  void set_global_sq(GlobalSq f) { _global_sq = std::move(f); }

  int steps_taken() const noexcept { return _t; }
  const OptimizerSpec& spec() const noexcept { return _spec; }
  bool has_ema() const noexcept { return !_ema_a.empty(); }
  const metal_compute::SharedBuffer& ema_a() const { return _ema_a; }
  const metal_compute::SharedBuffer& ema_b() const { return _ema_b; }
  // Prodigy's current step-size estimate (1 for the others).
  double prodigy_d() const noexcept { return _d; }

  // Bytes of optimizer state per parameter, for the plan, at the
  // adapter's `rank` (Adafactor's state is per row and per column: at
  // most 8 / rank bytes a parameter, a factor being [rank, k] or
  // [n, rank]; 0 takes rank 1, the bound).
  static std::size_t state_bytes_per_param(const OptimizerSpec& s,
                                           int rank = 0) noexcept;

  // The whole state -- moments, Prodigy's accumulators, EMA, the step
  // count -- so a resumed run continues the same trajectory.
  bool save_state(const std::string& path, std::string* err) const;
  // load_state runs check_state first, so a state that does not fit
  // leaves this optimizer as it was.
  bool load_state(const std::string& path, std::string* err);
  // Whether the state at `path` fits THIS optimizer -- its header, the
  // adapter's shape, which arenas it holds, every length and the file's
  // exact size -- reading only those (a few hundred bytes) and writing
  // nothing. A file cut short (a host gone before its data reached the
  // drive) fails here, before a resume agrees on it.
  bool check_state(const std::string& path, std::string* err) const;

 private:
  const TrainOps* _ops = nullptr;
  LoraParams* _p = nullptr;
  OptimizerSpec _spec;
  int _total = 0;
  int _warmup = 0;
  int _t = 0;
  // Per arena (A, B): first and second moments; Prodigy also keeps s, the
  // step-0 masters and a scratch for its per-element numerator.
  metal_compute::SharedBuffer _m_a, _m_b, _v_a, _v_b;
  metal_compute::SharedBuffer _s_a, _s_b, _x0_a, _x0_b, _num;
  metal_compute::SharedBuffer _ema_a, _ema_b;
  // adamw8bit: the moments' codes (a byte an element), each arena's block
  // scales [2 * blocks] and the 512 code values.
  metal_compute::SharedBuffer _qm_a, _qm_b, _qv_a, _qv_b, _am_a, _am_b;
  metal_compute::SharedBuffer _codes;
  // adafactor, per arena: its matrices (TrainOps::AfMat), their row and
  // column states, and the update's per-matrix sums of squares.
  struct AfArena {
    metal_compute::SharedBuffer mats, state, usq;
    int nmat = 0, rows = 0, cols = 0;
  };
  AfArena _af_a, _af_b;
  bool build_adafactor_(std::string* err);
  metal_compute::SharedBuffer _part;
  GlobalSq _global_sq;
  double _d = 1.0;
  double _d_max = 0.0;
  double _d_num = 0.0;
};

}  // namespace train
}  // namespace genai
}  // namespace vpipe

#endif
