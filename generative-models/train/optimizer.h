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

#include <string>
#include <string_view>

namespace vpipe {
namespace genai {
namespace train {

struct OptimizerSpec {
  std::string optimizer = "adamw";        // adamw | prodigy | lion | sgd
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

  // The choice lists the stage offers, in the editor's order.
  static constexpr const char* kOptimizers = "adamw,prodigy,lion,sgd";
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

  int steps_taken() const noexcept { return _t; }
  const OptimizerSpec& spec() const noexcept { return _spec; }
  bool has_ema() const noexcept { return !_ema_a.empty(); }
  const metal_compute::SharedBuffer& ema_a() const { return _ema_a; }
  const metal_compute::SharedBuffer& ema_b() const { return _ema_b; }
  // Prodigy's current step-size estimate (1 for the others).
  double prodigy_d() const noexcept { return _d; }

  // Bytes of optimizer state per parameter, for the plan.
  static std::size_t state_bytes_per_param(const OptimizerSpec& s) noexcept;

  // The whole state -- moments, Prodigy's accumulators, EMA, the step
  // count -- so a resumed run continues the same trajectory.
  bool save_state(const std::string& path, std::string* err) const;
  bool load_state(const std::string& path, std::string* err);

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
  metal_compute::SharedBuffer _part;
  double _d = 1.0;
  double _d_max = 0.0;
  double _d_num = 0.0;
};

}  // namespace train
}  // namespace genai
}  // namespace vpipe

#endif
