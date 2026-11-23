#ifndef VPIPE_STAGES_TRAIN_LORA_STAGE_H
#define VPIPE_STAGES_TRAIN_LORA_STAGE_H

// train-lora -- trains a LoRA on a diffusion model from an encoded
// dataset, and saves it where generate-image's adapter pickers find it.
//
// The graph around it is a generation graph's, run the other way:
//
//   training-dataset --captions--> diffusion-conditioner --cond--+
//                    --images----> vae-encode ----------latents--+
//                    --manifest----------------------------------+--> train
//   model-select ----------------------------------------model---+
//   optimizer-select ------------------------------------optimizer+
//
// so the adapter trains on exactly the conditioning it will be used with.
//
// The run has two phases. ENCODE: every caption and picture arrives once
// and is cached (in RAM); the conditioner and the VAE release their
// models at the end of the series. TRAIN: the model loads -- only now,
// so it never shares the box with the encoders -- and trains from the
// cache. Randomness is a pure function of (seed, sample number), so a
// resumed run is the run that was stopped.
//
// Outputs: preview latents at every checkpoint (for vae-decode), one beat
// per saved checkpoint, and one metrics beat per step.
//
// Commands (docs/STAGE-COMMANDS.md): `save` writes a checkpoint now and
// `preview` samples the preview prompts now, both between optimizer
// steps, so a run can be inspected and kept without stopping it.

#include "common/job.h"
#include "interfaces/ui-delegate-intf.h"
#include "pipeline/runtime-context.h"
#include "apple-silicon/tensor-beat.h"
#include "pipeline/typed-stage.h"
#include "stages/generation-input.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/train/lora-params.h"
#include "generative-models/train/lora-trainer.h"
#include "generative-models/train/optimizer.h"
#include "generative-models/train/sample-cache.h"
#include "generative-models/train/train-ops.h"
#endif

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

class TrainLoraStage final : public TypedStage<TrainLoraStage> {
public:
  static constexpr const char* kTypeName = "train-lora";

  TrainLoraStage(const SessionContextIntf* session, std::string id,
                 std::vector<InEdge> iports, FlexData config);
  ~TrainLoraStage() override;

  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;
  const StageSpec& spec() const noexcept override;
  void apply_constant(unsigned iport, const FlexData& beat) override;
  std::vector<ResourceClaim> declare_resources() const override;

  // The optimizer steps a run of `samples` pictures (repeats counted in
  // `epoch_len`) takes when `steps` and `epochs` are both automatic.
  // Public for tests.
  static int auto_steps(int samples, int epoch_len, int batch);
  // The sigma of sample number `g` (a pure function of the seed).
  // `mode`: auto (logit-normal pushed through the family's shift mu),
  // uniform, logit_normal, shift. Public for tests.
  static double draw_sigma(const std::string& mode, std::uint64_t seed,
                           std::uint64_t g, double mu, double mean,
                           double std_dev, double shift);

private:
  // ---- config
  std::string _hf_dir;
  std::string _dit_cfg;                 // dit_dir
  std::vector<std::string> _base_loras; // base_lora
  std::string _output_dir;
  std::string _name;
  std::string _targets = "attn_ff";
  std::string _init_from;
  std::string _resume = "auto";
  std::string _ts_mode = "auto";
  std::string _recompute = "auto";
  std::string _cache_cfg = "auto";    // auto | ram | disk
  int _rank = 16;
  double _alpha = 0.0;
  int _steps_cfg = 0;
  int _epochs_cfg = 0;
  int _batch = 1;
  std::uint64_t _seed = 0;
  SeedSequence _seed_seq = SeedSequence::kIncrement;
  double _caption_dropout = 0.05;
  double _dropout = 0.05;     // this run's: 0 for a guidance-distilled model
  double _logit_mean = 0.0;
  double _logit_std = 1.0;
  double _shift = 3.0;
  int _save_every_cfg = 0;
  int _preview_every_cfg = 0;
  int _preview_steps = 20;
  double _preview_cfg = 4.0;
  std::uint64_t _preview_seed = 42;
  bool _ane_forward = false;
  bool _ane_backward = false;
  bool _i8 = false;
  bool _register = true;

  // ---- run state
  enum class Phase : std::uint8_t { kStart, kEncode, kTrain, kDone };
  Phase _phase = Phase::kStart;
  bool _model_latched = false;
  FlexData _manifest;
  FlexData _opt_flex;
  std::string _trigger;
  std::string _base_ref;     // what hf_dir named, for the metadata
  std::chrono::steady_clock::time_point _encode_t0;
  std::uint64_t _run = 0;    // training jobs finished (a sweep's count)
  // What a host's progress view draws: "lora encode" (the encodes taken
  // of those the manifest wants), then "lora train" (the optimizer steps
  // of the run's total, the loss in its detail).
  UiProgress _bar;

  // Commands: `save` and `preview`, served between optimizer steps.
  Job serve_commands_(RuntimeContext& ctx);

#ifdef VPIPE_BUILD_APPLE_SILICON
  // An entry with a cache key and no buffer lives in the cache only: it
  // is read from disk when a step needs it.
  struct Text {
    metal_compute::SharedBuffer buf;
    int rows = 0;
    FlexData sideband;
    std::string key;
    // A caption encoded WITH REFERENCES (video-ref-encoder, Ref2VA): its
    // reference rows, bf16 -- video [nrv, 96], audio [nra, 32] -- always
    // in RAM (a picture's rows are a few MB), cached beside the caption.
    metal_compute::SharedBuffer rv, ra;
    int nrv = 0, nra = 0;
  };
  struct Picture {
    // Packed, or f32 [C,h,w] / [C,t,h,w] until then.
    metal_compute::SharedBuffer x0;
    int t = 1;              // latent frames: 1 for a picture
    int h = 0, w = 0;       // the latent grid
    std::string key;
    // A clip window's SOUNDTRACK: f32 [2, C, n] until the family packs it
    // (then its rows), always in RAM -- a few KB a window.
    bool audio = false;     // the manifest says one comes
    metal_compute::SharedBuffer ax0;
    int ac = 0, an = 0;     // its channels and latents, unpacked
    std::string akey;
    // KEYFRAMES (bit 0 the first frame, bit 1 the last): their still
    // latents f32 [C, 1, h, w] each until packed, then their rows in one
    // buffer (cx0); always in RAM. And the window's own captions,
    // presented with them (into _texts), in place of the sample's.
    int anchors = 0;
    std::vector<metal_compute::SharedBuffer> kx;
    std::vector<std::string> kkeys;
    metal_compute::SharedBuffer cx0;
    std::vector<int> texts;
  };
  struct Sample {
    std::vector<int> texts;          // into _texts
    std::vector<Picture> pics;
    int repeats = 1;
    bool validation = false;         // held out: scored, never trained
  };
  std::vector<Text> _texts;
  std::vector<Sample> _samples;
  int _null_text = -1;
  std::vector<int> _preview_texts;
  std::vector<std::string> _preview_prompts;
  std::vector<int> _epoch_slots;   // sample index per epoch position
  std::size_t _cache_bytes = 0;

  genai::train::SampleCache _cache;
  bool _disk = false;                // training reads the cache per step
  int _latent_c = 0;                 // the latents' channels
  std::vector<int> _val;             // the held-out samples
  double _val_last = -1.0;

  std::unique_ptr<genai::train::LoraTrainer> _trainer;
  genai::train::TrainOps _ops;
  genai::train::LoraParams _params;
  genai::train::Optimizer _opt;
  genai::train::OptimizerSpec _opt_spec;
  std::string _family;
  std::string _dit_dir;
  // The family's extra claims (frozen adapters, a second weight set),
  // withdrawn with the DiT's at the run's end.
  std::vector<std::string> _extra_dirs;
  // The trained adapter's models-DB record: its type and its parent's.
  std::string _adapter_type, _parent_type;
  int _total_steps = 0;
  int _step = 0;
  int _save_every = 0;
  int _preview_every = 0;
  std::uint64_t _run_seed = 0;
  std::size_t _reserve_bytes = 0;
  double _loss_sum = 0.0;
  int _loss_n = 0;
  std::vector<std::string> _saved;

  bool load_model_(std::string* err);
  // dit_dir and base_lora, resolved: what the families plan from.
  genai::train::RunShape run_paths_() const;
  // Encoded bytes into the cache (a no-op without one).
  void cache_put_(const std::string& key, const TensorBeatPayload& tb,
                  bool as_bf16);
  // Residency: what stays in RAM, what is read per step.
  bool settle_cache_(std::string* err);
  // A step's text and packed latent, from RAM or from the cache.
  bool text_(int ti, const Text** out, Text* scratch, std::string* err);
  bool latent_(const Picture& p, const metal_compute::SharedBuffer** out,
               metal_compute::SharedBuffer* scratch, std::string* err);
  // The held-out loss at fixed (sample, sigma, noise).
  bool validate_(double* loss, std::string* err);
  // A checkpoint at this step, emitted on the checkpoints port; `*out`
  // is its beat. `*err` says why when it fails.
  Job checkpoint_(RuntimeContext& ctx, FlexData* out, std::string* err);
  // The preview prompts (`which` < 0: all) with the live adapter, each
  // latent emitted on the previews port; `*n` counts them.
  Job previews_(RuntimeContext& ctx, int which, int steps, double cfg,
                std::uint64_t seed, int* n, std::string* err);
  int _saved_step = -1;
  FlexData _last_checkpoint;

  // The encode phase: every slot laid out from the manifest up front, and
  // the ones the two encoders still owe, in their arrival order.
  struct WantPic {
    int sample = 0, pic = 0;
    int ph = 0, pw = 0;      // the picture the latent must match
    std::string file;
    int anchor = -1;         // a keyframe's still: which of the window's
  };
  std::vector<WantPic> _want_audio;
  std::size_t _got_audio = 0;
  double _audio_weight = 1.0;
  bool _audio_wired = false;       // the audio_latents port
  bool _refs_wired = false;        // the ref_video_rows / ref_audio_rows
  int _audio_unwired = 0;          // soundtracks listed with it unwired
  bool plan_encode_(std::string* err);
  bool _encode_planned = false;
  std::vector<int> _want_text;
  std::vector<WantPic> _want_pic;
  std::size_t _got_text = 0, _got_pic = 0;
  bool start_training_(std::string* err);
  bool save_(bool final_save, std::string* err);
  bool resume_(std::string* err);
  void register_(const std::string& path);
  std::string resume_dir_() const;
#endif
};

}  // namespace vpipe

#endif
