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
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;

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
  // The latent grid (vae-encode's h, w), and its latent frames: 1 for a
  // picture, a clip's count for video.
  int grid_h = 0, grid_w = 0, frames = 1;
  float sigma = 0.0f;       // x_t = (1 - sigma) x0 + sigma noise
  float weight = 1.0f;      // the loss weight
  // An audio-video family's SOUNDTRACK, packed by pack_audio(): its clean
  // latent and noise. Null is a sample without one -- a picture, a clip
  // with no track -- whose audio rows the family lays out as inference
  // does and keeps out of the loss.
  const metal_compute::SharedBuffer* audio_x0 = nullptr;
  const metal_compute::SharedBuffer* audio_noise = nullptr;
  float audio_weight = 1.0f; // the soundtrack's weight beside the video's
  // KEYFRAMES (MiniMax-H3 FL2VA's tasks): bit 0 the first frame, bit 1
  // the last, pinned as conditioning; `cond_x0` their still latents
  // packed by pack_latent, first then last.
  int anchors = 0;
  const metal_compute::SharedBuffer* cond_x0 = nullptr;
  // REFERENCES (MiniMax-H3 Ref2VA): a caption encoded with references
  // (video-ref-encoder) names them on its sideband; `cond_x0` then holds
  // their video rows and `cond_audio` their soundtracks' rows, packed in
  // the encoder's order (bf16 [rows, width]).
  const metal_compute::SharedBuffer* cond_audio = nullptr;
  // The sample's own randomness, for what a family draws itself (the
  // noise of rows the stage does not know the shape of): a pure function
  // of (run seed, sample number), so a resumed run draws the same.
  std::uint64_t seed = 0;
};

struct StepStats {
  double loss = 0.0;
  double forward_ms = 0.0;
  double backward_ms = 0.0;
  // An audio-video family's two means, unweighted; negative for a family
  // with one modality.
  double loss_video = -1.0;
  double loss_audio = -1.0;
};

// An audio-video family's preview soundtrack: its latent, the shape
// audio-vae-decode reads, and its rate.
struct PreviewAudio {
  std::vector<float> latent;
  std::vector<std::int64_t> shape;
  double latents_per_second = 0.0;
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
  int frames = 1;           // latent frames (1: a picture)
  int steps = 20;
  float cfg = 4.0f;         // ignored by a guidance-distilled family
  std::uint64_t seed = 0;
  // OUT, when set: an audio-video family's preview soundtrack.
  PreviewAudio* audio = nullptr;
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

  // vae-encode's f32 latent -- [c, h, w] for a picture, [c, t, h, w]
  // for a family whose VAE has a time axis -- in the layout step()
  // reads. `t` is 1 for a picture; a family without a time axis refuses
  // more.
  virtual metal_compute::SharedBuffer
  pack_latent(const float* cthw, int c, int t, int h, int w,
              std::string* err) = 0;
  virtual int latent_channels() const noexcept = 0;
  // An audio-video family's soundtrack latent -- audio-vae-encode's f32
  // [stereo, channels, stride], its first `latents` columns -- in the
  // layout step() reads. Empty (with `*err`) for a family without audio.
  virtual metal_compute::SharedBuffer
  pack_audio(const float* audio, int stereo, int channels, int stride,
             int latents, std::string* err)
  {
    (void)audio; (void)stereo; (void)channels; (void)stride; (void)latents;
    if (err != nullptr) {
      *err = std::string(family()) + " trains without a soundtrack";
    }
    return {};
  }
  // The soundtrack latents (per stereo channel) a clip of `frames` latent
  // frames lays out; 0 for a family without audio.
  virtual int audio_latents(int frames) const
  {
    (void)frames;
    return 0;
  }
  // The shape of a latent this family emits (a preview) and vae-decode
  // reads: [c, h, w], or [c, t, h, w] for a family with a time axis.
  virtual std::vector<std::int64_t>
  latent_shape(int t, int h, int w) const
  {
    (void)t;
    return {latent_channels(), h, w};
  }

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

  // From noise to a latent with the live adapter, f32 in latent_shape().
  virtual bool preview(const PreviewRequest& r, std::vector<float>* chw,
                       std::string* err) = 0;

  // A cooperative stop between blocks and steps.
  virtual void set_stop(std::function<bool()> stop) = 0;

  // Family options, by FlexData key (train::opt::*); unknown keys are
  // ignored, absent keys keep the default.
  virtual void set_options(const FlexData& o) = 0;

  // Bytes a step at this geometry holds, for the resource plan.
  virtual std::size_t arena_bytes(int frames, int grid_h, int grid_w,
                                  int text_rows, int rank,
                                  bool keep_all) const = 0;
  // What the model itself holds now.
  virtual std::uint64_t resident_bytes() const = 0;

  // Whether caption dropout means anything here: a family sampled with
  // guidance trains some samples against the empty prompt so the
  // unconditional pass keeps working; a guidance-distilled one has no
  // such pass, and the stage turns dropout off for it.
  virtual bool trains_unconditional() const { return true; }

  // The adapter file's family metadata beyond the stage's own
  // (vpipe.family, vpipe.targets, ...): e.g. which partition of a model
  // it was trained on.
  virtual std::map<std::string, std::string> metadata() const { return {}; }
  // The models-DB record of a trained adapter: its model_type and the
  // model_type of what it binds on.
  virtual std::string adapter_model_type() const
  {
    return std::string(family()) + "-lora";
  }
  virtual std::string parent_model_type() const { return family(); }

  virtual void* query_extension(std::string_view) { return nullptr; }
};

// ---- the families the stage can train --------------------------------
//
// What the stage needs of a family BEFORE its model loads: whether a
// checkpoint is one, the weights it would hold (resident, and streaming
// at the least), and the step's working set -- the plan the stream /
// keep-all decisions and the claims are made from. Then the load.

// What the stage knows of the run when it asks.
struct RunShape {
  int max_frames = 1;            // the widest sample's latent frames
  int max_grid_h = 0, max_grid_w = 0;
  int max_cells = 0;             // the most frames * h * w of any sample
  int max_text = 0;              // the longest conditioning, in rows
  int max_anchors = 0;           // the most keyframes a sample pins
  int max_cond_rows = 0;         // the most reference rows a caption holds
  int rank = 16;
  std::string targets = "attn_ff";
  // The optimizer's state a parameter (masters, grads and shadows aside).
  std::size_t opt_bytes_per_param = 8;
  // What else shares the box while training: the sample cache, and the
  // headroom the irreversible decisions keep. Against `ram`.
  std::size_t cache_bytes = 0;
  std::size_t headroom = 0;
  std::size_t ram = 0;
  bool ane = false;              // the stage's `ane`
  bool i8 = false;               // the stage's `i8_gemm`
  // The stage's `dit_dir` (the DiT that runs, for a root whose pipeline
  // carries none this build reads: P-H3-Image's) and `base_lora` (frozen
  // adapters beneath the trained one), resolved to paths.
  std::string dit_dir;
  std::vector<std::string> base_loras;
};

struct LoadPlan {
  std::string dit_dir;           // the weight set to open
  std::size_t held_bytes = 0;    // what the model holds resident
  std::size_t floor_bytes = 0;   // ... and streaming, at the least
  // The adapter's masters, grads, shadows and the optimizer's state.
  std::size_t state_bytes = 0;
  // A step's working set at the widest sample: recomputing each block,
  // and keeping every block's activations (0: the family cannot).
  std::size_t arena_recompute = 0;
  std::size_t arena_keep = 0;
  // A line for the log: what the family decided for this box.
  std::string note;
  // A second weight set the model reads (MiniMax-H3-Image's per-step
  // heads), opened by the stage through the manager; empty for none.
  std::string aux_dir;
  // The family's own decisions, by key, handed back to load().
  FlexData opts;
};

class LoraFamily {
 public:
  virtual ~LoraFamily() = default;
  virtual const char* name() const noexcept = 0;
  // Whether `root` (a registry path, a checkpoint root or file) holds a
  // DiT this family trains; `model_type` is the models DB's record of it
  // (may be empty). False without a word when it is another family's;
  // false with `*err` set when it is this family's but not trainable.
  virtual bool claims(const std::string& root, const std::string& model_type,
                      const RunShape& r, std::string* err) const = 0;
  // The plan, off the checkpoint's headers.
  virtual bool plan(const std::string& root, const std::string& model_type,
                    const RunShape& r, LoadPlan* out,
                    std::string* err) const = 0;
  // The weight claim's directory and floor, for declare_resources (before
  // any geometry is known).
  virtual bool claim(const std::string& root, const std::string& model_type,
                     const RunShape& r, std::string* dit_dir,
                     std::size_t* floor) const = 0;
  // Checkpoints a run holds BESIDE the DiT, each claimed at launch and
  // let go with it: what it holds in all, and its floor when the DiT
  // streams. MiniMax-H3: each frozen `base_lora` adapter (the model's own
  // LoRA slots, which the DiT's figure leaves out) and MiniMax-H3-Image's
  // per-step heads (a weight set of their own).
  struct Extra {
    std::string path;
    std::size_t held = 0, floor = 0;
  };
  virtual std::vector<Extra>
  extra_claims(const std::string& /*root*/, const std::string& /*mtype*/,
               const RunShape& /*r*/) const
  {
    return {};
  }

  struct LoadArgs {
    std::shared_ptr<WeightSet> ws;   // `plan.dit_dir`'s weights
    std::shared_ptr<WeightSet> aux_ws;   // `plan.aux_dir`'s, when named
    metal_compute::MetalCompute* mc = nullptr;
    std::string root, model_type;
    const LoadPlan* plan = nullptr;
    bool stream_blocks = false;
    const RunShape* run = nullptr;
  };
  virtual std::unique_ptr<LoraTrainer> load(const LoadArgs& a,
                                            std::string* err) const = 0;
};

// Every family this build trains, in the order they are asked.
const std::vector<const LoraFamily*>& lora_families();

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
