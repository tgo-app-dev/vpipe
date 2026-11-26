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
#include "generative-models/tensor-parallel.h"
#include "generative-models/shared/tp-overlap.h"
#include "pipeline/stage-command.h"
#endif

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
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
  // `tensor_parallel`: this stage as one RANK of a run split across
  // hosts (train-lora-split.cc). Null: the whole DiT here.
  FlexData _tp_cfg;
  // `data_parallel`: this stage as one rank of a run whose steps' SAMPLES
  // are shared between hosts, each with the whole model. At most one of
  // the two.
  FlexData _dp_cfg;

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
  // The run's activations kept (recompute: never) or not, and what the
  // reserve is without them: a rank falls back where the other cannot.
  bool _keep_all = false;
  std::size_t _arena_recompute = 0;
  std::size_t _state_reserve = 0;
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
  // The state at step `at` read back (-1: the newest that loads -- a
  // whole run's); false with `err` empty when there is none.
  bool resume_(std::int64_t at, std::string* err);
  void register_(const std::string& path);
  std::string resume_dir_() const;

  // ---- A RANK OF A SPLIT RUN (`tensor_parallel`; train-lora-split.cc) --
  // Every rank is a train-lora stage of its own, on its own host,
  // stepping together. Rank 0 encodes the dataset and hands it to the
  // others whole, before any model loads -- so every rank trains on the
  // same bytes, and the others need no encoders, no dataset, no cache --
  // with the run's settings, which theirs then are. Then each loads its
  // share; each forward and backward exchanges partial sums (the
  // family's); between steps every rank agrees on what comes next -- a
  // stop, a checkpoint, a preview, whichever rank was asked
  // (tp_control_) -- the shared factors' gradients are summed and the
  // norm taken over every rank's share. Checkpoints are the WHOLE
  // adapter, gathered, written by rank 0; the resume state is written
  // by every rank (its adapter whole, its optimizer state its share's).
  genai::TpShare _tp;
  std::shared_ptr<genai::TpChannel> _tp_ch;
  std::uint64_t _tp_seq = 0;          // exchanges before the model's
  bool _tp_seed_given = false;        // a follower's: rank 0's run seed
  std::uint64_t _tp_seed = 0;
  double _tp_repl_sq = 0.0;           // the shared gradients' sum of squares
  // How long the channel waits on the peer once a stop is asked before
  // it gives up (the peer is gone): the train calls' longest, x 4.
  std::shared_ptr<std::atomic<std::int64_t>> _tp_grace_ms;
  // Commands rank 0 took, answered once every rank has run them.
  std::vector<std::shared_ptr<StageCommand>> _tp_pending;
  // ---- A RANK OF A DATA-PARALLEL RUN (`data_parallel`) ---------------
  // As a split run's -- the dataset handed over, every step agreed, the
  // resume state every rank's -- but each rank holds the WHOLE model and
  // trains its own samples of each step: rank r `_batch` x samples[r] of
  // them, in the one seeded order, the step's batch every rank's summed.
  // After its backward each rank's gradients (and losses) are summed
  // across the ranks, so every rank takes the same optimizer step and
  // holds the same adapter. Rank 0 alone writes it and makes previews.
  int _dp_rank = -1;
  int _dp_k[2] = {1, 1};              // samples[r]: each rank's multiple
  std::unique_ptr<genai::TpOverlap> _dp_pipe;
  std::unique_ptr<genai::TpBulk> _dp_bulk;
  // A rank of either kind of run; one that splits the model; a follower.
  bool tp_rank_() const { return _tp.active() || _dp_rank >= 0; }
  bool tp_split_() const { return _tp.active(); }
  bool dp_rank_() const { return _dp_rank >= 0; }
  bool tp_follower_() const
  {
    return (_tp.active() && _tp.rank > 0) || _dp_rank > 0;
  }
  int rank_index_() const { return _tp.active() ? _tp.rank : _dp_rank; }
  // The run's configuration of either kind (null: neither).
  const FlexData& par_cfg_() const
  {
    return _dp_cfg.is_object() ? _dp_cfg : _tp_cfg;
  }
  const char* par_key_() const
  {
    return _dp_cfg.is_object() ? "data_parallel" : "tensor_parallel";
  }
  // A step's samples, every rank's, and this rank's first and count.
  int global_batch_() const
  {
    return dp_rank_() ? _batch * (_dp_k[0] + _dp_k[1]) : _batch;
  }
  int local_first_() const
  {
    return _dp_rank > 0 ? _batch * _dp_k[0] : 0;
  }
  int local_count_() const
  {
    return dp_rank_() ? _batch * _dp_k[_dp_rank] : _batch;
  }
  // Bytes to and from the other rank between steps: a split run's
  // through the model's channel, a data-parallel run's through its own.
  bool bulk_(const void* mine, void* others, std::size_t bytes,
             genai::TpDType dtype, std::string* err);
  // Every rank's gradients (and `sums`, n of them: the step's losses)
  // summed into each rank's.
  bool dp_sync_gradients_(double* sums, int n, std::string* err);
  // The channel of a data-parallel run (tp_open_ for one).
  bool dp_open_(std::string* err);
  // The share (its widths the family's) and the channel.
  bool tp_open_(std::string* err);
  // Rank 0: the dataset and the run's settings out; a follower: in.
  bool tp_ship_(std::string* err);
  bool tp_receive_(std::string* err);
  // The shared factors' gradients summed across the ranks.
  bool tp_sync_gradients_(std::string* err);
  // What every rank does next, agreed: each rank's flags OR'd, rank 0's
  // preview arguments.
  struct TpControl {
    bool stop = false, save = false, preview = false;
    int prompt = -1, steps = 0;
    double cfg = 0.0;
    std::int64_t seed = -1;
    // What each rank decided alone from state both hold -- a checkpoint
    // or preview due -- agreed rather than assumed.
    bool due_save = false, due_preview = false;
    // Every `check_every` steps, a digest of what both ranks hold alike
    // (masters_digest_): 0 between checks.
    std::uint64_t digest = 0;
  };
  bool tp_control_(const TpControl& mine, TpControl* all, std::string* err);
  // The agreement before a step -- or, a stop asked after this step's,
  // the next one's, held at once -- and what it decides: `*stopped` the
  // run saved for its resume and ended; else the checkpoint and preview
  // the last step was due, and the commands taken, run on every rank.
  Job tp_agree_(RuntimeContext& ctx, bool* stopped, std::string* err);
  bool _tp_due_save = false, _tp_due_preview = false;
  void tp_close_();
  // THE HANDSHAKE, the first exchange of a run's channel: both ranks
  // configured as one run (the protocol, the kind of run, the ranks, the
  // share and its extra), and a transport that copies bytes (its probe).
  // Refused on both, naming the field, on any mismatch.
  bool tp_handshake_(std::string* err);
  // ...and once each has its model and adapter (start_training_): the
  // same DiT (its checkpoint's headers), the same frozen adapters, the
  // same modules laid out alike.
  bool tp_check_layout_(std::string* err);
  // A digest of what every rank holds alike: a split run's shared
  // factors' masters, a data-parallel run's whole adapter.
  std::uint64_t masters_digest_() const;
  int _tp_check_every = 50;           // `check_every`; 0: never
  // The run's first agreements checked whatever check_every says (set
  // by start_training_; counted down alike on every rank).
  int _tp_first_checks = 0;
  // The digests exchanged now, outside an agreement (before a state is
  // written); false, with the parting said, where they differ.
  bool tp_alike_(std::string* err);
  std::string parted_(std::uint64_t mine, std::uint64_t other) const;
  // The gradient sync's host copies, sized once (and booked): macOS
  // keeps a freed large allocation in the footprint.
  std::vector<float> _sync_mine, _sync_others;
  // The bytes a rank's exchanges hold beside the plan's (booked).
  std::size_t split_reserve_bytes_() const;
  // FAULTS, for tests (VPIPE_TRAIN_FAULT="<rank>:<step>:<what>"): this
  // rank fails at the start of that step's train call ("step"), its
  // preview at that step fails ("preview"), or a shared master is nudged
  // there ("part") -- a GPU's arithmetic parting the ranks.
  struct Fault {
    int rank = -1, step = -1;
    std::string what;
  } _fault;
  bool fault_(const char* what) const
  {
    return _fault.rank == rank_index_() && _fault.step == _step &&
           _fault.what == what;
  }
  // A resume state's run.json is this run's (its seed, length, dataset,
  // batch, and -- a rank -- the kind of run, the rank and its share);
  // `why` the first field that is not.
  bool resume_matches_(const FlexData& run, std::string* why) const;
  // What run.json records of a rank (resume_matches_ checks it).
  FlexData rank_record_() const;
  // THE RESUME STATES KEPT -- the last kResumeKept checkpoints' (review
  // 4b): a rank that died between its peer's write and its own still
  // holds the step before, which the ranks agree on (newest held by
  // both) rather than starting over. Each in `step-NNNNNN/` of the
  // resume folder (optimizer.bin + run.json), written under a .tmp name
  // and renamed whole; the folder's own run.json the newest's record.
  // An older layout's state -- the two files at the folder's root -- is
  // read as one more.
  static constexpr int kResumeKept = 2;
  struct ResumeSlot {
    std::int64_t step = -1;
    std::filesystem::path dir;   // where its optimizer.bin and run.json are
  };
  // This run's states here, newest first (run.json matching, both files
  // there, the optimizer's check_state passed); `others` -- if given --
  // set when a state for another run was passed over, `broken` what was
  // passed over as not loading.
  std::vector<ResumeSlot> resume_slots_(
      bool* others = nullptr,
      std::vector<std::string>* broken = nullptr) const;
  // After a save at `upto` (or a resume there): the states past it --
  // an abandoned run's, which a later agreement could pair with a peer's
  // real one -- gone, then all but the newest kResumeKept, the .tmp
  // leftovers and the older layout's files; the folder's run.json the
  // newest kept's.
  void resume_tidy_(std::int64_t upto) const;
#endif
};

}  // namespace vpipe

#endif
