#ifndef VPIPE_GENERATIVE_MODELS_TRAIN_LORA_TRAINER_H
#define VPIPE_GENERATIVE_MODELS_TRAIN_LORA_TRAINER_H

// What a model family implements to be LoRA-trainable -- the whole of the
// model-specific part. Everything else (the dataset cache, the optimizer
// and its schedule, sigma sampling, checkpoints, resume, previews' decode,
// memory planning) is the train-lora stage's and is shared by every
// family, image or video.
//
// A family trains ONE sample per step() call, accumulating into its bound
// LoraParams' gradients; the stage owns the optimizer step and so the
// batch (accumulation over calls).

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "generative-models/train/lora-params.h"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
namespace genai {
namespace train {

// The query_extension id a plug-in family answers with a LoraTrainer
// factory, when it can train.
inline constexpr const char* kLoraTrainerExtension = "vpipe.lora-trainer/1";

// One sample, as the stage hands it to the family.
struct TrainSample {
  // The clean latent and the noise, packed by pack_latent().
  const metal_compute::SharedBuffer* x0 = nullptr;
  const metal_compute::SharedBuffer* noise = nullptr;
  // The conditioning as the conditioner emitted it, and its sideband.
  const metal_compute::SharedBuffer* text = nullptr;
  int text_rows = 0;
  const FlexData* sideband = nullptr;
  // The latent grid (vae-encode's h, w; frames for video).
  int grid_h = 0, grid_w = 0, frames = 1;
  float sigma = 0.0f;       // x_t = (1 - sigma) x0 + sigma noise
  float weight = 1.0f;      // the loss weight
};

struct StepStats {
  double loss = 0.0;
  double forward_ms = 0.0;
  double backward_ms = 0.0;
};

// A preview: the live adapter, from noise to a latent.
struct PreviewRequest {
  const metal_compute::SharedBuffer* text = nullptr;
  int text_rows = 0;
  const FlexData* sideband = nullptr;
  // The unconditional side of CFG (the empty prompt); null runs without.
  const metal_compute::SharedBuffer* neg_text = nullptr;
  int neg_rows = 0;
  const FlexData* neg_sideband = nullptr;
  int grid_h = 0, grid_w = 0;
  int steps = 20;
  float cfg = 4.0f;
  std::uint64_t seed = 0;
};

class LoraTrainer {
 public:
  virtual ~LoraTrainer() = default;

  virtual const char* family() const noexcept = 0;

  // The projections `targets` names ("attn", "attn_ff", ...), each with
  // the module path its adapter file uses.
  virtual std::vector<ModuleShape>
  modules(const std::string& targets) const = 0;
  // The forward reads `params`' shadows from now on. Modules bound.
  virtual int bind(LoraParams* params, std::string* err) = 0;

  // vae-encode's f32 [c, h, w] latent in the layout step() reads.
  virtual metal_compute::SharedBuffer
  pack_latent(const float* chw, int c, int h, int w, std::string* err) = 0;
  virtual int latent_channels() const noexcept = 0;

  // Forward, loss and backward for one sample.
  virtual bool step(const TrainSample& s, StepStats* out,
                    std::string* err) = 0;
  // The same loss, forward only, adding nothing to any gradient: what a
  // held-out sample is scored with.
  virtual bool eval_loss(const TrainSample& s, double* loss,
                         std::string* err) = 0;

  // The flow shift mu this family's inference schedule applies at a
  // grid, which `auto` sigma sampling pushes its draws through; 0 is no
  // shift.
  virtual double schedule_mu(int grid_h, int grid_w) const noexcept = 0;

  // From noise to a latent with the live adapter: f32 [c, h, w].
  virtual bool preview(const PreviewRequest& r, std::vector<float>* chw,
                       std::string* err) = 0;

  // A cooperative stop between blocks and steps.
  virtual void set_stop(std::function<bool()> stop) = 0;

  // Family options, by FlexData key (train::opt::*); unknown keys are
  // ignored, absent keys keep the default.
  virtual void set_options(const FlexData& o) = 0;

  // Bytes a step at this geometry holds, for the resource plan.
  virtual std::size_t arena_bytes(int grid_h, int grid_w, int text_rows,
                                  int rank, bool keep_all) const = 0;
  // What the model itself holds now.
  virtual std::uint64_t resident_bytes() const = 0;

  virtual void* query_extension(std::string_view) { return nullptr; }
};

// Option keys for set_options.
namespace opt {
inline constexpr std::string_view kKeepAll = "keep_all";
inline constexpr std::string_view kAneBackward = "ane_backward";
// A streamed model: bytes to leave free beside it (Int), and the steps of
// the run (Int), for its block residency.
inline constexpr std::string_view kResidencyReserve = "residency_reserve";
inline constexpr std::string_view kResidencySteps = "residency_steps";
}  // namespace opt

}  // namespace train
}  // namespace genai
}  // namespace vpipe

#endif
