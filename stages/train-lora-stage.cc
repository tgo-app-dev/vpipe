#include "stages/train-lora-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "common/flex-bag.h"
#include "stages/model-memory-host.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "generative-models/gen-input.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/weight-set.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace vpipe {

namespace fs = std::filesystem;

namespace {

constexpr unsigned kCondPort = 0;
constexpr unsigned kLatentPort = 1;
constexpr unsigned kManifestPort = 2;
constexpr unsigned kModelPort = 3;
constexpr unsigned kOptimizerPort = 4;
constexpr unsigned kAudioPort = 5;
constexpr unsigned kRefVideoPort = 6;
constexpr unsigned kRefAudioPort = 7;

#ifdef VPIPE_BUILD_APPLE_SILICON
// The cache keys of a caption's reference rows.
std::string
ref_key_(const std::string& text_key, const char* which)
{
  return text_key.empty() ? std::string()
                          : genai::train::cache_key({text_key, which});
}

// Whether a conditioning's sideband names references (a Ref2VA caption).
bool
names_references_(const FlexData& sb)
{
  if (!sb.is_object()) { return false; }
  FlexData b = sb;
  auto o = b.as_object();
  if (!o.contains(genai::cond_sideband::kReferences)) { return false; }
  const FlexData r = o.at(genai::cond_sideband::kReferences);
  return r.is_array() && r.as_array().size() > 0;
}
#endif

constexpr SpecExtra kTargetChoices[] = {
  {spec_key::kChoices, "attn_ff,attn"},
};
constexpr SpecExtra kTsChoices[] = {
  {spec_key::kChoices, "auto,uniform,logit_normal,shift"},
};
constexpr SpecExtra kRecomputeChoices[] = {
  {spec_key::kChoices, "auto,always,never"},
};
constexpr SpecExtra kResumeChoices[] = {
  {spec_key::kChoices, "auto,never"},
};
constexpr SpecExtra kCacheChoices[] = {
  {spec_key::kChoices, "auto,ram,disk"},
};
constexpr SpecExtra kSeedSequenceExtra[] = {
  {spec_key::kChoices, kSeedSequenceChoices},
};

const ConfigKey kAttrs[] = {
  {.key = "hf_dir",
   .type = ConfigType::String,
   .doc = "the model to train on: a registry key or a checkpoint root "
          "(overridden by a model-select beat on the model port)",
   .suggest_db = "__vpipe_model_registry",
   .suggest_db_type = "qwen-image-21,minimax-h3-fl2va,minimax-h3-ref2va",
   .model_channel = "image-model"},
  {.key = "dit_dir",
   .type = ConfigType::String,
   .doc = "the DiT that runs, when hf_dir's pipeline names one it does not "
          "carry (P-H3-Image: MiniMax-H3 FL2VA's); a registry key or a "
          "directory",
   .suggest_db = "__vpipe_model_registry",
   .suggest_db_type = "minimax-h3-fl2va",
   .is_path = true,
   .path_kind = "dir"},
  {.key = "base_lora",
   .type = ConfigType::Any,
   .doc = "MiniMax-H3: frozen adapter(s) beneath the one trained -- a path "
          "or a list (P-H3-Image's PrunaAI adapter, or a style to stack "
          "on); recorded in the adapter's metadata as what inference must "
          "stack beneath it",
   .suggest_db = "__vpipe_model_registry",
   .suggest_db_type = "minimax-h3-lora",
   .is_path = true,
   .path_filter = "weights"},
  {.key = "output_dir",
   .type = ConfigType::String,
   .required = true,
   .doc = "where the adapter, its checkpoints and the resume state go",
   .is_path = true,
   .path_write = true,
   .path_kind = "dir"},
  {.key = "name",
   .type = ConfigType::String,
   .required = true,
   .doc = "the adapter's name: <name>.safetensors, <name>-000500."
          "safetensors per checkpoint, and the registry key it is offered "
          "under"},
  {.key = "rank",
   .type = ConfigType::Int,
   .doc = "the adapter's rank (default 16)",
   .def_int = 16},
  {.key = "alpha",
   .type = ConfigType::Real,
   .doc = "the adapter's alpha; 0 (default) is the rank, a strength of 1"},
  {.key = "targets",
   .type = ConfigType::String,
   .doc = "which projections train: attn_ff (default -- q, k, v, out and "
          "the feed-forward; what most published adapters train) or attn",
   .def_str = "attn_ff",
   .extra = kTargetChoices},
  {.key = "init_from",
   .type = ConfigType::String,
   .doc = "continue from an existing adapter file of the same targets and "
          "rank (empty: a fresh adapter, which starts as the base model)",
   .is_path = true,
   .path_filter = "weights"},
  {.key = "steps",
   .type = ConfigType::Int,
   .doc = "optimizer steps; 0 (default) is automatic -- 100 per picture "
          "for a small dataset (600 to 3000), two epochs for a large one"},
  {.key = "epochs",
   .type = ConfigType::Int,
   .doc = "passes over the dataset, used when steps is 0"},
  {.key = "batch_size",
   .type = ConfigType::Int,
   .doc = "samples a step accumulates before the optimizer moves "
          "(default 1). They run one after another, so a larger batch "
          "costs time, not memory.",
   .def_int = 1},
  {.key = "seed",
   .type = ConfigType::Int,
   .doc = "the run's seed: sample order, noise, sigmas, dropout"},
  {.key = "seed_sequence",
   .type = ConfigType::String,
   .doc = "the seed of each training job when several optimizer beats "
          "sweep: increment (default), randomize, keep",
   .def_str = "increment",
   .extra = kSeedSequenceExtra},
  {.key = "caption_dropout",
   .type = ConfigType::Real,
   .doc = "the chance a sample trains against the empty prompt instead of "
          "its caption (default 0.05), which keeps guidance working",
   .def_real = 0.05},
  {.key = "timestep_sampling",
   .type = ConfigType::String,
   .doc = "how sigma is drawn: auto (a logit-normal draw pushed through "
          "the model's own inference shift at the picture's size, so "
          "training samples where sampling spends its steps), uniform, "
          "logit_normal, or shift (a static shift)",
   .def_str = "auto",
   .extra = kTsChoices},
  {.key = "logit_mean",
   .type = ConfigType::Real,
   .doc = "logit-normal mean (default 0)"},
  {.key = "logit_std",
   .type = ConfigType::Real,
   .doc = "logit-normal standard deviation (default 1)",
   .def_real = 1.0},
  {.key = "shift",
   .type = ConfigType::Real,
   .doc = "timestep_sampling=shift: the static shift (default 3)",
   .def_real = 3.0},
  {.key = "save_every",
   .type = ConfigType::Int,
   .doc = "steps between checkpoints; 0 (default) is automatic (every 250 "
          "for a small dataset)"},
  {.key = "preview_every",
   .type = ConfigType::Int,
   .doc = "steps between preview samples; 0 (default) is at every "
          "checkpoint, -1 never"},
  {.key = "preview_steps",
   .type = ConfigType::Int,
   .doc = "sampling steps of a preview (default 20)",
   .def_int = 20},
  {.key = "preview_cfg",
   .type = ConfigType::Real,
   .doc = "guidance of a preview against the empty prompt (default 4; 1 "
          "is none)",
   .def_real = 4.0},
  {.key = "preview_seed",
   .type = ConfigType::Int,
   .doc = "the previews' noise seed, the same at every checkpoint so they "
          "compare (default 42)",
   .def_int = 42},
  {.key = "recompute",
   .type = ConfigType::String,
   .doc = "auto (default): keep every block's activations when the box "
          "can hold them, which saves about a quarter of a step, and "
          "recompute them block by block when it cannot. always / never "
          "force it.",
   .def_str = "auto",
   .extra = kRecomputeChoices},
  {.key = "ane",
   .type = ConfigType::Bool,
   .doc = "M4 series: split the large GEMMs of the forward and the "
          "backward with the Neural Engine (MEASURED 1.13x on a 1024^2 "
          "step on an M4 Pro). Do not enable on an M5."},
  {.key = "i8_gemm",
   .type = ConfigType::Bool,
   .doc = "M5 and later: int8 matrix-core GEMMs in the forward and the "
          "backward (lossy, ~1e-2 per GEMM)"},
  {.key = "audio_loss_weight",
   .type = ConfigType::Real,
   .doc = "an audio-video model (MiniMax-H3): the soundtrack's loss beside "
          "the video's, each its own mean (default 1). A clip without a "
          "soundtrack takes none either way",
   .def_real = 1.0},
  {.key = "resume",
   .type = ConfigType::String,
   .doc = "auto (default): continue a run of the same name that was "
          "stopped; never: start over",
   .def_str = "auto",
   .extra = kResumeChoices},
  {.key = "register",
   .type = ConfigType::Bool,
   .doc = "offer the finished adapter in the model registry, so "
          "generate-image's LoRA pickers list it (default on)",
   .def_bool = true},
  {.key = "tensor_parallel",
   .type = ConfigType::Object,
   .required = false,
   .doc = "this stage as one RANK of a run split across two hosts "
          "(MiniMax-H3's video DiT): {rank, ranks: 2, heads: [first, end], "
          "ffn: [first, end] -- unsaid, an even split -- transport: the "
          "name a plugin registered its transport under, params: that "
          "transport's own}. Rank 0 is the graph with the dataset and its "
          "encoders, and hands the encoded dataset and the run's settings "
          "to rank 1, a train-lora stage alone (hf_dir, output_dir, name); "
          "the ranks keep their blocks' activations (recompute) only "
          "where both can; rank 0 writes the checkpoints, whole"},
  {.key = "data_parallel",
   .type = ConfigType::Object,
   .required = false,
   .doc = "this stage as one rank of a run whose steps' samples are shared "
          "between two hosts, each holding the whole model (any family): "
          "{rank, ranks: 2, samples: [rank 0's, rank 1's] -- each a "
          "multiple of batch_size, 1 and 1 unsaid -- transport, params}. "
          "A step is batch_size x (both) samples, in the one seeded order; "
          "each rank trains its own, then the gradients are summed and "
          "every rank takes the same optimizer step. Rank 0 is the graph "
          "with the dataset, handed over as tensor_parallel's is; rank 0 "
          "writes the adapter and makes the previews. Not with "
          "tensor_parallel"},
  {.key = "cache",
   .type = ConfigType::String,
   .doc = "where the encoded dataset lives while training: ram, disk (read "
          "from the training-dataset's cache_dir every step), or auto -- "
          "ram when it fits an eighth of the box, disk otherwise",
   .def_str = "auto",
   .extra = kCacheChoices},
};

const PortSpec kIports[] = {
  {.name = "conditioning",
   .doc = "the diffusion-conditioner's output for every caption, the empty "
          "prompt and the preview prompts, in the manifest's order",
   .type = &typeid(TensorBeatPayload)},
  {.name = "latents",
   .doc = "vae-encode's f32 [C,h,w] latent for every picture, in the "
          "manifest's order",
   .type = &typeid(TensorBeatPayload)},
  {.name = "manifest",
   .doc = "training-dataset's manifest (one beat)",
   .type = &typeid(FlexDataPayload)},
  {.name = "model",
   .doc = "OPTIONAL shared model reference from a model-select source",
   .type = &typeid(FlexDataPayload)},
  {.name = "optimizer",
   .doc = "OPTIONAL optimizer-select spec (default AdamW); several beats "
          "run one training job each",
   .type = &typeid(FlexDataPayload)},
  {.name = "audio_latents",
   .doc = "OPTIONAL: audio-vae-encode's latent of every clip window's "
          "soundtrack the manifest lists ([2, C, n] for MiniMax-H3), in "
          "its order; unwired, clips train without their soundtracks",
   .type = &typeid(TensorBeatPayload)},
  {.name = "ref_video_rows",
   .doc = "OPTIONAL (MiniMax-H3 Ref2VA): video-ref-encoder's ref_video "
          "beside every conditioning it emits -- the references' video "
          "rows, f32 [rows, 96]; wired with ref_audio_rows",
   .type = &typeid(TensorBeatPayload)},
  {.name = "ref_audio_rows",
   .doc = "OPTIONAL (MiniMax-H3 Ref2VA): video-ref-encoder's ref_audio, "
          "the references' soundtrack rows, f32 [rows, 32]",
   .type = &typeid(TensorBeatPayload)},
};

const PortSpec kOports[] = {
  {.name = "previews",
   .doc = "preview latents at each checkpoint, f32 [C,h,w], for vae-decode",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "checkpoints",
   .doc = "one beat per saved adapter {path, step, loss, trigger_word}",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "metrics",
   .doc = "one beat per step {step, total, loss, lr, grad_norm, sec}; an "
          "audio-video model adds loss_video and loss_audio",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "preview_audio",
   .doc = "OPTIONAL: an audio-video model's preview soundtracks, f32 "
          "[2, C, n] beside each preview latent, for audio-vae-decode",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};

const ConfigKey kSaveResults[] = {
  {.key = "path", .type = ConfigType::String,
   .doc = "the checkpoint written"},
  {.key = "step", .type = ConfigType::Int,
   .doc = "the optimizer step it holds"},
  {.key = "loss", .type = ConfigType::Real,
   .doc = "the mean training loss since the previous checkpoint"},
  {.key = "val_loss", .type = ConfigType::Real,
   .doc = "the held-out loss, when the dataset holds samples out"},
};
const ConfigKey kPreviewArgs[] = {
  {.key = "prompt", .type = ConfigType::Int,
   .doc = "which preview prompt, from 0; -1 (default) for all of them",
   .def_int = -1},
  {.key = "steps", .type = ConfigType::Int,
   .doc = "sampling steps; 0 (default) for the stage's preview_steps",
   .def_int = 0},
  {.key = "cfg", .type = ConfigType::Real,
   .doc = "guidance; 0 (default) for the stage's preview_cfg",
   .def_real = 0.0},
  {.key = "seed", .type = ConfigType::Int,
   .doc = "the first prompt's seed; -1 (default) for the stage's "
          "preview_seed",
   .def_int = -1},
};
const ConfigKey kPreviewResults[] = {
  {.key = "step", .type = ConfigType::Int,
   .doc = "the optimizer step the adapter was at"},
  {.key = "previews", .type = ConfigType::Int,
   .doc = "latents emitted on the previews port"},
};
const CommandSpec kCommands[] = {
  {.name = "save",
   .doc = "save a checkpoint now, with its resume state, and emit it on "
          "the checkpoints port; served between optimizer steps",
   .results = kSaveResults},
  {.name = "preview",
   .doc = "sample the preview prompts with the live adapter now and emit "
          "the latents on the previews port; served between steps",
   .args = kPreviewArgs, .results = kPreviewResults},
};

const StageSpec kSpec = {
  .type_name = "train-lora",
  .doc = "Train a LoRA adapter on a diffusion model from a training-"
         "dataset encoded by the model's own conditioner and VAE. Caches "
         "the encoded dataset, loads the model once the encoders are done, "
         "trains with the optimizer-select spec, and saves checkpoints "
         "(with previews) that generate-image's and generate-video's "
         "adapter pickers offer. Qwen-Image-2.1, and MiniMax-H3 (FL2VA or "
         "Ref2VA) on pictures, each a one-frame clip.",
  .display_name = "Train LoRA",
  .category = StageCategory::Generative,
  .iports = kIports,
  .oports = kOports,
  .attrs = kAttrs,
  .commands = kCommands,
};

std::uint64_t
mix_(std::uint64_t z)
{
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

double
unit_(std::uint64_t z)
{
  return (double)(mix_(z) >> 11) * (1.0 / 9007199254740992.0);
}

}  // namespace

TrainLoraStage::TrainLoraStage(const SessionContextIntf* s, std::string id,
                               std::vector<InEdge> iports, FlexData config)
  : TypedStage<TrainLoraStage>(s, std::move(id), std::move(iports),
                               std::move(config))
{
  _hf_dir = attr_str("hf_dir");
  _output_dir = attr_str("output_dir");
  _name = attr_str("name");
  _rank = (int)attr_int("rank");
  _alpha = attr_real("alpha");
  _targets = attr_str("targets");
  _init_from = attr_str("init_from");
  _steps_cfg = (int)attr_int("steps");
  _epochs_cfg = (int)attr_int("epochs");
  _batch = std::max<int>(1, (int)attr_int("batch_size"));
  _seed = (std::uint64_t)attr_int("seed");
  _caption_dropout = attr_real("caption_dropout");
  _ts_mode = attr_str("timestep_sampling");
  _logit_mean = attr_real("logit_mean");
  _logit_std = attr_real("logit_std");
  _shift = attr_real("shift");
  _save_every_cfg = (int)attr_int("save_every");
  _preview_every_cfg = (int)attr_int("preview_every");
  _preview_steps = (int)attr_int("preview_steps");
  _preview_cfg = attr_real("preview_cfg");
  _preview_seed = (std::uint64_t)attr_int("preview_seed");
  _recompute = attr_str("recompute");
  _ane_forward = _ane_backward = attr_bool("ane");
  _i8 = attr_bool("i8_gemm");
  _resume = attr_str("resume");
  _register = attr_bool("register");
  _cache_cfg = attr_str("cache");
  _audio_weight = attr_real("audio_loss_weight");
  _dit_cfg = attr_str("dit_dir");
  if (this->config().as_object().contains("tensor_parallel")) {
    _tp_cfg = this->config().as_object().at("tensor_parallel");
    if (!_tp_cfg.is_object()) { _tp_cfg = FlexData(); }
  }
  if (this->config().as_object().contains("data_parallel")) {
    _dp_cfg = this->config().as_object().at("data_parallel");
    if (!_dp_cfg.is_object()) { _dp_cfg = FlexData(); }
  }
  {
    auto o = this->config().as_object();
    if (o.contains("base_lora")) {
      const FlexData b = o.at("base_lora");
      if (b.is_array()) {
        auto a = b.as_array();
        for (std::size_t i = 0; i < a.size(); ++i) {
          const std::string v(a.at(i).as_string(""));
          if (!v.empty()) { _base_loras.push_back(v); }
        }
      } else if (b.is_string() && !std::string(b.as_string("")).empty()) {
        _base_loras.push_back(std::string(b.as_string("")));
      }
    }
  }
  if (_cache_cfg.empty()) { _cache_cfg = "auto"; }
  if (!parse_seed_sequence(attr_str("seed_sequence"), &_seed_seq)) {
    fail_config(fmt("TrainLoraStage('{}'): seed_sequence must be increment, "
                    "randomize or keep", this->id()));
  }
  if (_output_dir.empty() || _name.empty()) {
    fail_config(fmt("TrainLoraStage('{}'): output_dir and name are "
                    "required", this->id()));
  }
  if (_rank < 1 || _rank > 1024) {
    fail_config(fmt("TrainLoraStage('{}'): rank must be 1..1024", this->id()));
  }
  if (_caption_dropout < 0.0 || _caption_dropout > 1.0) {
    fail_config(fmt("TrainLoraStage('{}'): caption_dropout must be in "
                    "[0, 1]", this->id()));
  }
  if (_ts_mode != "auto" && _ts_mode != "uniform" &&
      _ts_mode != "logit_normal" && _ts_mode != "shift") {
    fail_config(fmt("TrainLoraStage('{}'): timestep_sampling must be auto, "
                    "uniform, logit_normal or shift", this->id()));
  }
  if (_targets != "attn" && _targets != "attn_ff") {
    fail_config(fmt("TrainLoraStage('{}'): targets must be attn_ff or attn",
                    this->id()));
  }
  _base_ref = _hf_dir;
  allocate_oports(spec().oports.size());
}

TrainLoraStage::~TrainLoraStage() = default;

const StageSpec&
TrainLoraStage::spec() const noexcept
{
  return kSpec;
}

void
TrainLoraStage::apply_constant(unsigned iport, const FlexData& beat)
{
  if (iport != kModelPort) { return; }
  if (apply_model_select_beat(beat, _hf_dir)) { _base_ref = _hf_dir; }
}

void
TrainLoraStage::reset_run_state()
{
  _phase = Phase::kStart;
  _model_latched = false;
  _manifest = FlexData{};
  _opt_flex = FlexData{};
  _run = 0;
#ifdef VPIPE_BUILD_APPLE_SILICON
  _texts.clear();
  _samples.clear();
  _null_text = -1;
  _preview_texts.clear();
  _preview_prompts.clear();
  _epoch_slots.clear();
  _cache_bytes = 0;
  _cache.close();
  _disk = false;
  _latent_c = 0;
  _val.clear();
  _val_last = -1.0;
  _saved_step = -1;
  _last_checkpoint = FlexData();
  _encode_planned = false;
  _want_text.clear();
  _want_pic.clear();
  _want_audio.clear();
  _got_text = 0;
  _got_pic = 0;
  _got_audio = 0;
  _step = 0;
  _saved.clear();
  tp_close_();
  _tp = genai::TpShare{};
  _tp_seq = 0;
  _tp_seed_given = false;
  _bar.finish();
#endif
}

int
TrainLoraStage::auto_steps(int samples, int epoch_len, int batch)
{
  if (samples <= 0) { return 0; }
  batch = std::max(batch, 1);
  if (samples < 64) {
    return std::clamp(100 * samples / batch, 600, 3000);
  }
  const int epochs = samples > 8000 ? 1 : 2;
  return std::max(1, epochs * epoch_len / batch);
}

double
TrainLoraStage::draw_sigma(const std::string& mode, std::uint64_t seed,
                           std::uint64_t g, double mu, double mean,
                           double std_dev, double shift)
{
  const std::uint64_t k = mix_(seed ^ mix_(g * 3 + 1));
  double u = unit_(k);
  if (mode == "uniform") {
    // u as drawn
  } else if (mode == "shift") {
    u = shift * u / (1.0 + (shift - 1.0) * u);
  } else {
    // A Box-Muller normal from two uniforms, through the logistic.
    const double u1 = std::max(unit_(k ^ 0x5bd1e995ull), 1e-12);
    const double u2 = unit_(k ^ 0x27d4eb2full);
    const double n = std::sqrt(-2.0 * std::log(u1)) * std::cos(2 * M_PI * u2);
    u = 1.0 / (1.0 + std::exp(-(mean + std_dev * n)));
    if (mode == "auto" && mu != 0.0) {
      // The inference schedule's exponential time shift.
      const double e = std::exp(mu);
      u = e / (e + (1.0 / u - 1.0));
    }
  }
  return std::clamp(u, 1e-4, 1.0 - 1e-4);
}

#ifdef VPIPE_BUILD_APPLE_SILICON
genai::train::RunShape
TrainLoraStage::run_paths_() const
{
  genai::train::RunShape r;
  if (!_dit_cfg.empty()) { r.dit_dir = resolve_model_dir(session(), _dit_cfg); }
  for (const std::string& b : _base_loras) {
    // A registry key names a file (a LoRA record's local_path) or a
    // directory holding one.
    std::string p = resolve_model_dir(session(), b);
    std::error_code ec;
    if (fs::is_directory(p, ec)) {
      for (const auto& e : fs::directory_iterator(p, ec)) {
        if (e.path().extension() == ".safetensors") {
          p = e.path().string();
          break;
        }
      }
    }
    r.base_loras.push_back(p);
  }
  return r;
}
#endif

std::vector<ResourceClaim>
TrainLoraStage::declare_resources() const
{
  // The training phase's holding: the model, at its streaming floor when
  // it can stream. The encoders' phase is the conditioner's and vae-
  // encode's to declare; this stage loads only after they are done.
  std::vector<ResourceClaim> v;
  if (_hf_dir.empty()) { return v; }
#ifdef VPIPE_BUILD_APPLE_SILICON
  const ResolvedModel rm = resolve_model(session(), _hf_dir);
  const std::string root =
      rm.dir.empty() ? resolve_model_dir(session(), _hf_dir) : rm.dir;
  const genai::train::RunShape run = run_paths_();
  for (const genai::train::LoraFamily* f : genai::train::lora_families()) {
    std::string why;
    if (!f->claims(root, rm.model_type, run, &why)) { continue; }
    std::string dit;
    std::size_t floor = 0;
    if (f->claim(root, rm.model_type, run, &dit, &floor)) {
      v.push_back(model_memory::weight_claim_streamable(dit, floor));
    }
    // What the run holds beside the DiT (frozen adapters, a second weight
    // set), each at what it holds.
    for (const auto& e : f->extra_claims(root, rm.model_type, run)) {
      ResourceClaim c = model_memory::weight_claim_streamable(e.path,
                                                              e.floor);
      bag::set_integer(c.extra, model_memory::kClaimHeldBytes,
                       (std::int64_t)e.held);
      v.push_back(std::move(c));
    }
    break;
  }
#endif
  // The step's activations and the adapter's state, before any geometry
  // is known: revised to the measured figures once training starts.
  for (auto& c : model_memory::scratch_claims(
           "train-lora-arena", model_memory::kUnknownArena, {})) {
    v.push_back(std::move(c));
  }
  return v;
}

#ifdef VPIPE_BUILD_APPLE_SILICON

namespace {

metal_compute::SharedBuffer
copy_beat_(metal_compute::MetalCompute* mc, const TensorBeatPayload& tb)
{
  const auto bytes = tb.materialize_contiguous();
  metal_compute::SharedBuffer b = mc->make_shared_buffer(bytes.size());
  if (!b.empty()) { std::memcpy(b.contents(), bytes.data(), bytes.size()); }
  return b;
}

}  // namespace

void
TrainLoraStage::cache_put_(const std::string& key, const TensorBeatPayload& tb,
                           bool as_bf16)
{
  if (!_cache.is_open() || key.empty()) { return; }
  const auto bytes = tb.materialize_contiguous();
  std::vector<std::int64_t> shape(tb.shape.begin(), tb.shape.end());
  if (as_bf16) {
    // An f32 latent, stored in the storage type it is trained in.
    const std::size_t n = bytes.size() / 4;
    const auto* f = reinterpret_cast<const float*>(bytes.data());
    std::vector<std::uint8_t> b(n * 2);
    auto* d = reinterpret_cast<std::uint16_t*>(b.data());
    for (std::size_t i = 0; i < n; ++i) {
      d[i] = genai::train::f32_to_bf16(f[i]);
    }
    _cache.put(key, "BF16", std::move(shape), std::move(b), std::string());
    return;
  }
  const char* dt = tb.dtype == TensorBeat::DType::Bf16  ? "BF16"
                   : tb.dtype == TensorBeat::DType::F32 ? "F32"
                                                        : "F16";
  _cache.put(key, dt, std::move(shape),
             std::vector<std::uint8_t>(bytes.begin(), bytes.end()),
             tb.sideband.is_null() ? std::string() : tb.sideband.to_json());
}

bool
TrainLoraStage::settle_cache_(std::string* err)
{
  auto* mc = session()->services()->metal_compute();
  if (_cache.is_open()) {
    std::string cerr;
    if (!_cache.flush(&cerr)) {
      // Not fatal: this run holds what it encoded in RAM regardless.
      session()->warn(fmt("TrainLoraStage('{}'): the sample cache was not "
                          "written: {}", this->id(), cerr));
    }
  }
  // What the samples need resident, packed.
  std::size_t total = 0;
  for (const Text& t : _texts) {
    if (!t.buf.empty()) { total += t.buf.byte_size(); continue; }
    const auto* e = _cache.find(t.key);
    total += e != nullptr ? (std::size_t)e->bytes : 0;
  }
  for (const Sample& s : _samples) {
    for (const Picture& p : s.pics) {
      total += (std::size_t)std::max(_latent_c, 1) * p.t * p.h * p.w * 2;
      total += (std::size_t)2 * p.ac * p.an * 2;
    }
  }
  const std::size_t budget = model_memory::phys_ram() / 8;
  _disk = false;
  if (_cache.is_open() && _cache.size() > 0) {
    _disk = _cache_cfg == "disk" || (_cache_cfg == "auto" && total > budget);
  } else if (_cache_cfg == "disk") {
    session()->warn(fmt("TrainLoraStage('{}'): cache is 'disk' but the "
                        "dataset has no cache_dir (wire its model port); "
                        "training from RAM", this->id()));
  } else if (total > budget) {
    session()->warn(fmt("TrainLoraStage('{}'): the encoded dataset ({} MB) "
                        "is held in RAM with no cache to read it from; wire "
                        "the training-dataset's model port so it can train "
                        "from disk", this->id(), total >> 20));
  }
  // Texts the previews and dropout use every checkpoint stay resident.
  std::vector<bool> keep(_texts.size(), !_disk);
  if (_null_text >= 0) { keep[(std::size_t)_null_text] = true; }
  for (int i : _preview_texts) { keep[(std::size_t)i] = true; }
  _cache_bytes = 0;
  for (std::size_t i = 0; i < _texts.size(); ++i) {
    Text& t = _texts[i];
    // A cached caption with references: its rows back into RAM.
    if (names_references_(t.sideband) && t.nrv + t.nra == 0 &&
        t.rv.empty() && t.ra.empty()) {
      for (int k = 0; k < 2; ++k) {
        const std::string rk = ref_key_(t.key, k == 0 ? "video" : "audio");
        const auto* e = _cache.find(rk);
        if (e == nullptr || e->shape.size() != 2) {
          *err = "a cached caption's reference rows are not in '" +
                 _cache.dir() + "' -- run again to re-encode them";
          return false;
        }
        metal_compute::SharedBuffer& b = k == 0 ? t.rv : t.ra;
        (k == 0 ? t.nrv : t.nra) = (int)e->shape[0];
        if (e->bytes == 0) { continue; }
        b = mc->make_shared_buffer((std::size_t)e->bytes);
        if (!_cache.read(rk, b.contents(), (std::size_t)e->bytes, err)) {
          return false;
        }
      }
    }
    _cache_bytes += t.rv.byte_size() + t.ra.byte_size();
    // An EMPTY conditioning (MiniMax-H3's empty prompt: no rows) holds
    // nothing, in RAM or the cache.
    if (t.rows == 0) { continue; }
    if (!keep[i]) {
      if (!t.key.empty() && _cache.has(t.key)) { t.buf = {}; }
      if (!t.buf.empty()) { _cache_bytes += t.buf.byte_size(); }
      continue;
    }
    if (t.buf.empty()) {
      const auto* e = _cache.find(t.key);
      if (e == nullptr) {
        *err = "a cached caption vanished from '" + _cache.dir() + "'";
        return false;
      }
      t.buf = mc->make_shared_buffer((std::size_t)e->bytes);
      if (!_cache.read(t.key, t.buf.contents(), (std::size_t)e->bytes,
                       err)) {
        return false;
      }
    }
    _cache_bytes += t.buf.byte_size();
  }
  for (Sample& s : _samples) {
    for (Picture& p : s.pics) {
      // A soundtrack stays in RAM in either mode (a few KB a window):
      // back from the cache when this run did not encode it.
      if (p.audio && p.ax0.empty()) {
        const auto* e = _cache.find(p.akey);
        if (e == nullptr) {
          *err = "a cached soundtrack vanished from '" + _cache.dir() + "'";
          return false;
        }
        std::vector<std::uint16_t> b((std::size_t)e->bytes / 2);
        if (!_cache.read(p.akey, b.data(), (std::size_t)e->bytes, err)) {
          return false;
        }
        p.ax0 = mc->make_shared_buffer(b.size() * 4);
        auto* d = static_cast<float*>(p.ax0.contents());
        for (std::size_t i = 0; i < b.size(); ++i) {
          d[i] = genai::train::bf16_to_f32(b[i]);
        }
      }
      if (p.audio) { _cache_bytes += p.ax0.byte_size() / 2; }
      // The keyframes' stills likewise stay in RAM.
      for (std::size_t k = 0; k < p.kx.size(); ++k) {
        if (p.kx[k].empty()) {
          const auto* e = _cache.find(p.kkeys[k]);
          if (e == nullptr) {
            *err = "a cached keyframe vanished from '" + _cache.dir() + "'";
            return false;
          }
          std::vector<std::uint16_t> b((std::size_t)e->bytes / 2);
          if (!_cache.read(p.kkeys[k], b.data(), (std::size_t)e->bytes,
                           err)) {
            return false;
          }
          p.kx[k] = mc->make_shared_buffer(b.size() * 4);
          auto* d = static_cast<float*>(p.kx[k].contents());
          for (std::size_t i = 0; i < b.size(); ++i) {
            d[i] = genai::train::bf16_to_f32(b[i]);
          }
        }
        _cache_bytes += p.kx[k].byte_size() / 2;
      }
      if (_disk) {
        if (!p.key.empty() && _cache.has(p.key)) { p.x0 = {}; continue; }
        _cache_bytes += p.x0.byte_size() / 2;
        continue;
      }
      if (p.x0.empty()) {
        // Back from the cache into f32 [C,h,w], exactly the latent that
        // was stored, for packing.
        const auto* e = _cache.find(p.key);
        if (e == nullptr) {
          *err = "a cached latent vanished from '" + _cache.dir() + "'";
          return false;
        }
        std::vector<std::uint16_t> b((std::size_t)e->bytes / 2);
        if (!_cache.read(p.key, b.data(), (std::size_t)e->bytes, err)) {
          return false;
        }
        p.x0 = mc->make_shared_buffer(b.size() * 4);
        auto* d = static_cast<float*>(p.x0.contents());
        for (std::size_t i = 0; i < b.size(); ++i) {
          d[i] = genai::train::bf16_to_f32(b[i]);
        }
      }
      _cache_bytes += p.x0.byte_size() / 2;
    }
  }
  for (std::size_t i = 0; i < _samples.size(); ++i) {
    if (_samples[i].validation) { _val.push_back((int)i); }
  }
  return true;
}

bool
TrainLoraStage::text_(int ti, const Text** out, Text* scratch,
                      std::string* err)
{
  const Text& t = _texts[(std::size_t)ti];
  if (!t.buf.empty() || t.rows == 0) { *out = &t; return true; }
  auto* mc = session()->services()->metal_compute();
  const auto* e = _cache.find(t.key);
  if (e == nullptr) {
    *err = "caption " + t.key + " is in neither RAM nor the cache";
    return false;
  }
  scratch->rows = t.rows;
  scratch->sideband = t.sideband;
  scratch->key = t.key;
  scratch->nrv = t.nrv;
  scratch->nra = t.nra;
  scratch->buf = mc->make_shared_buffer((std::size_t)e->bytes);
  if (!_cache.read(t.key, scratch->buf.contents(), (std::size_t)e->bytes,
                   err)) {
    return false;
  }
  *out = scratch;
  return true;
}

bool
TrainLoraStage::latent_(const Picture& p,
                        const metal_compute::SharedBuffer** out,
                        metal_compute::SharedBuffer* scratch,
                        std::string* err)
{
  if (!p.x0.empty()) { *out = &p.x0; return true; }
  const auto* e = _cache.find(p.key);
  if (e == nullptr || (e->shape.size() != 3 && e->shape.size() != 4)) {
    *err = "latent " + p.key + " is in neither RAM nor the cache";
    return false;
  }
  std::vector<std::uint16_t> b((std::size_t)e->bytes / 2);
  if (!_cache.read(p.key, b.data(), (std::size_t)e->bytes, err)) {
    return false;
  }
  std::vector<float> chw(b.size());
  for (std::size_t i = 0; i < b.size(); ++i) {
    chw[i] = genai::train::bf16_to_f32(b[i]);
  }
  *scratch = _trainer->pack_latent(chw.data(), (int)e->shape[0], p.t, p.h,
                                  p.w, err);
  *out = scratch;
  return !scratch->empty();
}

bool
TrainLoraStage::validate_(double* loss, std::string* err)
{
  auto* mc = session()->services()->metal_compute();
  double sum = 0.0;
  int n = 0;
  for (std::size_t i = 0; i < _val.size(); ++i) {
    const Sample& s = _samples[(std::size_t)_val[i]];
    if (s.pics.empty() || s.texts.empty()) { continue; }
    const Picture& p = s.pics.front();
    const std::vector<int>& tl = p.texts.empty() ? s.texts : p.texts;
    Text scratch;
    const Text* t = nullptr;
    metal_compute::SharedBuffer xs;
    const metal_compute::SharedBuffer* xp = nullptr;
    if (!text_(tl.front(), &t, &scratch, err) ||
        !latent_(p, &xp, &xs, err)) {
      return false;
    }
    const metal_compute::SharedBuffer& x0 = *xp;
    // FIXED: sigma by stratum, noise by index -- the same triple at every
    // checkpoint of every run, so the curve moves only with the adapter.
    const double sigma = ((double)(i % 4) + 0.5) / 4.0;
    const std::size_t ne = x0.byte_size() / 2;
    metal_compute::SharedBuffer noise = mc->make_shared_buffer(ne * 2);
    {
      std::mt19937_64 rng(mix_(0x76616c69646174ull ^ mix_(i)));
      std::normal_distribution<float> nd(0.0f, 1.0f);
      auto* d = static_cast<std::uint16_t*>(noise.contents());
      for (std::size_t j = 0; j < ne; ++j) {
        d[j] = genai::train::f32_to_bf16(nd(rng));
      }
    }
    genai::train::TrainSample ts;
    ts.x0 = &x0;
    ts.noise = &noise;
    ts.text = &t->buf;
    ts.text_rows = t->rows;
    ts.sideband = &t->sideband;
    ts.grid_h = p.h;
    ts.grid_w = p.w;
    ts.frames = p.t;
    ts.sigma = (float)sigma;
    ts.seed = mix_(0x76616c69646174ull ^ mix_(i * 5 + 1));
    if (p.anchors != 0) {
      ts.anchors = p.anchors;
      ts.cond_x0 = &p.cx0;
    } else if (t->nrv + t->nra > 0) {
      // The caption's references (in RAM on its entry, never a scratch).
      const Text& tr = _texts[(std::size_t)tl.front()];
      ts.cond_x0 = &tr.rv;
      ts.cond_audio = &tr.ra;
    }
    metal_compute::SharedBuffer anoise;
    if (!p.ax0.empty()) {
      const std::size_t an = p.ax0.byte_size() / 2;
      anoise = mc->make_shared_buffer(an * 2);
      std::mt19937_64 rng(mix_(0x61756469306f6eull ^ mix_(i)));
      std::normal_distribution<float> nd(0.0f, 1.0f);
      auto* d = static_cast<std::uint16_t*>(anoise.contents());
      for (std::size_t j = 0; j < an; ++j) {
        d[j] = genai::train::f32_to_bf16(nd(rng));
      }
      ts.audio_x0 = &p.ax0;
      ts.audio_noise = &anoise;
      ts.audio_weight = (float)_audio_weight;
    }
    double l = 0.0;
    if (!_trainer->eval_loss(ts, &l, err)) { return false; }
    sum += l;
    ++n;
  }
  *loss = n > 0 ? sum / n : 0.0;
  return true;
}

std::string
TrainLoraStage::resume_dir_() const
{
  // A follower's is named for its rank: two hosts sharing an output
  // folder (a network share) keep theirs apart, and a relaunch that swaps
  // the ranks finds the other's state recorded as another rank's
  // (resume_matches_). Rank 0's -- and a whole run's -- as before.
  if (tp_follower_()) {
    return (fs::path(_output_dir) /
            fmt("{}.rank{}.resume", _name, rank_index_())()).string();
  }
  return (fs::path(_output_dir) / (_name + ".resume")).string();
}

bool
TrainLoraStage::load_model_(std::string* err)
{
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) {
    *err = "no metal-compute backend";
    return false;
  }
  if (_hf_dir.empty()) {
    *err = "no model -- set hf_dir or wire a model-select source";
    return false;
  }
  const ResolvedModel rm = resolve_model(session(), _hf_dir);
  const std::string root =
      rm.dir.empty() ? resolve_model_dir(session(), _hf_dir) : rm.dir;
  // WHICH FAMILY: the first that claims the checkpoint. One that knows it
  // but cannot train it says why.
  const genai::train::LoraFamily* fam = nullptr;
  genai::train::RunShape run = run_paths_();
  for (const genai::train::LoraFamily* f : genai::train::lora_families()) {
    std::string why;
    if (f->claims(root, rm.model_type, run, &why)) { fam = f; break; }
    if (!why.empty()) {
      *err = "'" + _hf_dir + "': " + why;
      return false;
    }
  }
  if (fam == nullptr) {
    *err = "'" + _hf_dir + "' is not a model this stage can train (today: "
           "Qwen-Image-2.1, MiniMax-H3 FL2VA / Ref2VA)";
    return false;
  }
  _family = fam->name();
  _extra_dirs.clear();
  for (const auto& e : fam->extra_claims(root, rm.model_type, run)) {
    _extra_dirs.push_back(e.path);
  }

  // The widest sample this run trains, for the plan below.
  for (const Sample& s : _samples) {
    for (const Picture& p : s.pics) {
      run.max_cells = std::max(run.max_cells, p.t * p.h * p.w);
      run.max_frames = std::max(run.max_frames, p.t);
      run.max_grid_h = std::max(run.max_grid_h, p.h);
      run.max_grid_w = std::max(run.max_grid_w, p.w);
      run.max_anchors =
          std::max(run.max_anchors, __builtin_popcount((unsigned)p.anchors));
    }
  }
  for (const Text& t : _texts) {
    run.max_text = std::max(run.max_text, t.rows);
    run.max_cond_rows = std::max(run.max_cond_rows, t.nrv + t.nra);
  }
  run.rank = _rank;
  run.targets = _targets;
  run.opt_bytes_per_param =
      genai::train::Optimizer::state_bytes_per_param(_opt_spec, _rank);
  run.cache_bytes = _cache_bytes;
  run.headroom = model_memory::kStreamHeadroom;
  run.ram = model_memory::phys_ram();
  run.ane = _ane_forward;
  run.i8 = _i8;
  run.tp = _tp;
  genai::train::LoadPlan plan;
  if (!fam->plan(root, rm.model_type, run, &plan, err)) { return false; }
  _dit_dir = plan.dit_dir;
  // STREAMING, by training's own rule. The generation rule sizes a DiT
  // against the text encoder beside it; here the encoders are done and
  // released before this loads, so what shares the box is the step's
  // transient -- activations and the adapter's state -- and the cache.
  // Irreversible, so the larger headroom.
  bool stream = plan.held_bytes + plan.arena_recompute + plan.state_bytes +
                    _cache_bytes + model_memory::kStreamHeadroom >
                model_memory::phys_ram();
  if (const char* e = std::getenv("VPIPE_TRAIN_STREAM")) {
    stream = std::atoi(e) != 0;
  }
  genai::train::LoraFamily::LoadArgs a;
  a.ws = genai::open_weight_set(_dit_dir, session());
  if (!plan.aux_dir.empty()) {
    a.aux_ws = genai::open_weight_set(plan.aux_dir, session());
  }
  a.mc = mc;
  a.root = root;
  a.model_type = rm.model_type;
  a.plan = &plan;
  a.stream_blocks = stream;
  a.run = &run;
  auto t = fam->load(a, err);
  if (t == nullptr) { return false; }
  // KEEP ALL when the box holds it -- about a quarter of a step saved --
  // where the family can.
  const std::size_t held = stream ? plan.floor_bytes : plan.held_bytes;
  bool keep = false;
  if (plan.arena_keep > 0) {
    if (_recompute == "never") {
      keep = true;
    } else if (_recompute == "auto") {
      keep = held + plan.arena_keep + plan.state_bytes + _cache_bytes +
                 model_memory::kHeadroom <= model_memory::phys_ram();
    }
  }
  const std::size_t arena = keep ? plan.arena_keep : plan.arena_recompute;
  _keep_all = keep;
  _arena_recompute = plan.arena_recompute;
  FlexData o = FlexData::make_object();
  o.as_object().insert_or_assign(genai::train::opt::kKeepAll,
                                 FlexData::make_bool(keep));
  o.as_object().insert_or_assign(genai::train::opt::kAneBackward,
                                 FlexData::make_bool(_ane_backward));
  t->set_options(o);
  _state_reserve = plan.state_bytes + _cache_bytes + model_memory::kHeadroom;
  _reserve_bytes = arena + _state_reserve;
  session()->info(fmt(
      "TrainLoraStage('{}'): {} DiT {} ({} MB held, {} MB streaming at the "
      "least){}, activations {} ({} MB at the widest sample: {} frame(s) "
      "of {}x{} latents, {} text rows), adapter state {} MB, {} MB of "
      "cached samples", this->id(), _family,
      stream ? "STREAMED" : "resident", plan.held_bytes >> 20,
      plan.floor_bytes >> 20, plan.note.empty() ? "" : "; " + plan.note,
      keep ? "kept for every block" : "recomputed per block", arena >> 20,
      run.max_frames, run.max_grid_w, run.max_grid_h, run.max_text,
      plan.state_bytes >> 20, _cache_bytes >> 20));
  _adapter_type = t->adapter_model_type();
  _parent_type = t->parent_model_type();
  _trainer = std::move(t);
  if (auto* mgr = session()->services()->generative_model_manager()) {
    // The step's activations, the adapter's state, and the encoded
    // samples this run keeps in RAM (all of them, or what a disk cache
    // leaves resident: the previews' texts, soundtracks, keyframes and
    // reference rows) -- held from here to the run's end.
    mgr->revise_scratch("train-lora-arena",
                        arena + plan.state_bytes + _cache_bytes);
    // What the model HOLDS, which the claim made at launch from the
    // checkpoint could not know: a DiT pruned at load keeps its AdaLN as
    // 174 MB where the checkpoint carries 26 GB, and a streamed one its
    // floor (the manager holds a revision there at the least).
    mgr->revise_declaration(_dit_dir,
                            (std::size_t)_trainer->resident_bytes());
  }
  return true;
}

bool
TrainLoraStage::start_training_(std::string* err)
{
  auto* mc = session()->services()->metal_compute();
  if (!_ops.valid() && !_ops.init(mc, true, err)) { return false; }
  _fault = Fault{};
  if (const char* e = std::getenv("VPIPE_TRAIN_FAULT")) {
    int r = -1, st = -1;
    char what[16] = {};
    if (std::sscanf(e, "%d:%d:%15s", &r, &st, what) == 3) {
      _fault = Fault{r, st, what};
    }
  }
  // The seed of this training job: a sweep's jobs follow seed_sequence.
  _run_seed = _tp_seed_given ? _tp_seed
                             : generation_seed(_seed_seq, _seed, _run);
  // Caption dropout keeps a guided model's unconditional pass working; a
  // guidance-distilled family has none.
  _dropout = _caption_dropout;
  if (!_trainer->trains_unconditional() && _caption_dropout > 0.0) {
    session()->info(fmt("TrainLoraStage('{}'): {} is guidance-distilled -- "
                        "no unconditional pass to keep -- so caption "
                        "dropout is off", this->id(), _family));
    _dropout = 0.0;
  }
  std::vector<genai::train::ModuleShape> mods = _trainer->modules(_targets);
  const float alpha = _alpha > 0.0 ? (float)_alpha : (float)_rank;
  if (!_params.create(mc, mods, _rank, alpha, err)) { return false; }
  // A RANK: its adapter is its share of the whole (each module's cut), so
  // it starts where the unsplit run would and saves the whole.
  if (tp_split_()) {
    std::map<std::string, genai::train::ModuleCut> cuts;
    if (!_trainer->split_cuts(mods, &cuts, err) ||
        !_params.set_cuts(cuts, err)) {
      return false;
    }
    if (_opt_spec.optimizer == "prodigy") {
      *err = "tensor_parallel: Prodigy's step size is a whole adapter's "
             "estimate; a split run takes another optimizer";
      return false;
    }
    // adamw8bit scales its moments' codes per block of the arena: a block
    // holding a SHARED factor's parameters and a cut one's would code the
    // shared ones by a rank's own maximum, and the ranks' copies -- alike
    // by construction otherwise -- would part.
    if (_opt_spec.optimizer == "adamw8bit") {
      const std::size_t blk = genai::train::TrainOps::kOpt8Block;
      for (const auto& sp : _params.shared_grad_spans()) {
        const std::size_t end = sp.off + sp.n;
        const std::size_t len =
            sp.b ? _params.b_elems() : _params.a_elems();
        if (sp.off % blk != 0 || (end % blk != 0 && end != len)) {
          *err = fmt("tensor_parallel: adamw8bit codes its moments in "
                     "blocks of {}, and this share at rank {} puts a "
                     "shared factor and a cut one in one block (rank x "
                     "the share's heads must be even, rank x its "
                     "feed-forward rows a multiple of {}); another "
                     "optimizer or share", blk, _rank, blk)();
          return false;
        }
      }
    }
  }
  _params.init_random(_run_seed);
  if (!_init_from.empty()) {
    int loaded = 0;
    if (!_params.load(_init_from, &loaded, err)) { return false; }
    session()->info(fmt("TrainLoraStage('{}'): continuing from '{}' ({} of "
                        "{} modules)", this->id(), _init_from, loaded,
                        mods.size()));
  }
  const int bound = _trainer->bind(&_params, err);
  if (bound != (int)mods.size()) {
    if (err->empty()) { *err = "the adapter did not bind"; }
    return false;
  }
  // The run's channel, from where the dataset's exchanges left it.
  if (tp_split_() && !_trainer->split_begin(_tp_ch, _tp_seq, err)) {
    return false;
  }
  // Both ranks loaded the same model and laid the adapter out alike; and
  // what their exchanges hold beside the plan, booked.
  if (tp_rank_()) {
    if (!tp_check_layout_(err)) { return false; }
    const std::size_t sb = split_reserve_bytes_();
    _state_reserve += sb;
    _reserve_bytes += sb;
    if (auto* mgr = session()->services()->generative_model_manager()) {
      mgr->revise_scratch("train-lora-arena",
                          _reserve_bytes - model_memory::kHeadroom);
    }
  }
  // A RANK keeps its blocks' activations only where every rank does: a
  // kept block's forward and a recomputed one's backward exchange
  // differently, so the ranks' exchanges would pair the wrong sums.
  if (tp_split_()) {
    const std::int64_t mine = _keep_all ? 1 : 0;
    std::int64_t other = 0;
    if (!_trainer->split_bulk(&mine, &other, sizeof mine,
                              genai::TpDType::kF32, err)) {
      return false;
    }
    if (_keep_all && other == 0) {
      _keep_all = false;
      FlexData o = FlexData::make_object();
      o.as_object().insert_or_assign(genai::train::opt::kKeepAll,
                                     FlexData::make_bool(false));
      _trainer->set_options(o);
      _reserve_bytes = _arena_recompute + _state_reserve;
      session()->info(fmt("TrainLoraStage('{}'): the other rank recomputes "
                          "its blocks; so does this one", this->id()));
    }
  }
  // The schedule.
  int epoch_len = 0;
  _epoch_slots.clear();
  int trained = 0;
  for (std::size_t i = 0; i < _samples.size(); ++i) {
    if (_samples[i].validation) { continue; }
    ++trained;
    for (int r = 0; r < _samples[i].repeats; ++r) {
      _epoch_slots.push_back((int)i);
    }
  }
  epoch_len = (int)_epoch_slots.size();
  if (epoch_len == 0) {
    *err = "every sample is held out for validation; nothing to train on";
    return false;
  }
  if (_steps_cfg > 0) {
    _total_steps = _steps_cfg;
  } else if (_epochs_cfg > 0) {
    _total_steps = std::max(1, _epochs_cfg * epoch_len / global_batch_());
  } else {
    _total_steps = auto_steps(trained, epoch_len, global_batch_());
  }
  _save_every = _save_every_cfg > 0
      ? _save_every_cfg
      : (trained < 64 ? 250
                      : std::max(100, (_total_steps / 8 / 100) * 100));
  _preview_every = _preview_every_cfg == 0 ? _save_every : _preview_every_cfg;
  // The regime decides an automatic schedule: a small set trains by steps
  // at a constant rate, a large one by epochs down a cosine.
  if (_opt_spec.lr_schedule == "auto") {
    _opt_spec.lr_schedule = trained >= 64 ? "cosine" : "constant";
  }
  if (!_opt.init(&_ops, &_params, _opt_spec, _total_steps, err)) {
    return false;
  }
  // A RANK's norm is every rank's: each share once, the shared factors --
  // summed already, held alike -- once. (Cleared otherwise: _opt outlives
  // a run.)
  _opt.set_global_sq({});
  if (tp_split_()) {
    _opt.set_global_sq([this](double local, double* global,
                              std::string* e) {
      double other = 0.0;
      if (!_trainer->split_bulk(&local, &other, sizeof local,
                                genai::TpDType::kF32, e)) {
        return false;
      }
      *global = local + other - _tp_repl_sq;
      return true;
    });
  }
  // A STREAMED model keeps nothing until it is told what else needs the
  // box: residency refuses every block until a reserve is declared, and a
  // training step visits every block twice. The reserve is the step's own
  // transient plus the cache; the schedule is the run.
  {
    FlexData o = FlexData::make_object();
    auto oo = o.as_object();
    oo.insert_or_assign(genai::train::opt::kResidencyReserve,
                        FlexData::make_int((std::int64_t)_reserve_bytes));
    oo.insert_or_assign(genai::train::opt::kResidencySteps,
                        FlexData::make_int(_total_steps));
    _trainer->set_options(o);
  }
  _step = 0;
  _loss_sum = 0.0;
  _loss_n = 0;
  // A RANK resumes only where every rank would, at the same step: the
  // newest step BOTH hold (each keeps its last kResumeKept states, so a
  // rank that died between its peer's write and its own goes back one
  // checkpoint, not to the start). EVERY rank exchanges, whatever its own
  // `resume` says (rank 0's settings do not carry it): an exchange one
  // rank skipped would pair each later one with the wrong partner, their
  // counts -- the transport's check -- in step. -1 none; -2 `resume:
  // never` here.
  bool resume = _resume == "auto";
  std::int64_t at = -1;
  if (tp_rank_()) {
    std::int64_t mine[kResumeKept], other[kResumeKept];
    for (int i = 0; i < kResumeKept; ++i) {
      mine[i] = resume ? -1 : -2;
      other[i] = -1;
    }
    if (resume) {
      std::vector<std::string> broken;
      const std::vector<ResumeSlot> held = resume_slots_(nullptr, &broken);
      for (const std::string& b : broken) {
        session()->warn(fmt("TrainLoraStage('{}'): {}; passed over",
                            this->id(), b));
      }
      for (std::size_t i = 0; i < held.size() && i < (std::size_t)kResumeKept;
           ++i) {
        mine[i] = held[i].step;
      }
    }
    if (!bulk_(mine, other, sizeof mine, genai::TpDType::kF32, err)) {
      return false;
    }
    if (mine[0] == -2 || other[0] == -2) {
      resume = false;
    } else {
      for (int i = 0; i < kResumeKept && at < 0; ++i) {
        for (int k = 0; k < kResumeKept; ++k) {
          if (mine[i] >= 0 && mine[i] == other[k]) {
            at = mine[i];
            break;
          }
        }
      }
      if (at < 0) { resume = false; }
    }
    auto list = [](const std::int64_t* v) {
      std::string o;
      for (int i = 0; i < kResumeKept; ++i) {
        if (v[i] < 0 && i > 0) { break; }
        o += (o.empty() ? "" : ", ") + std::to_string(v[i]);
      }
      return o;
    };
    if (at < 0 && (mine[0] >= 0 || other[0] >= 0)) {
      session()->warn(fmt("TrainLoraStage('{}'): the ranks hold no resume "
                          "state at one step (this rank {}, the other {}; "
                          "-1 none, -2 resume: never); starting over",
                          this->id(), list(mine), list(other)));
    } else if (at >= 0 && (at != mine[0] || at != other[0])) {
      session()->warn(fmt("TrainLoraStage('{}'): the ranks' newest resume "
                          "states are at steps {} and {}; resuming at {}, "
                          "the newest both hold", this->id(), mine[0],
                          other[0], at));
    }
  }
  if (resume && !resume_(at, err)) {
    if (!err->empty()) { return false; }
  }
  // ...and both did: a rank that read nothing back would train other
  // samples than its peer at every step from here.
  if (tp_rank_()) {
    const std::int64_t at = _step;
    std::int64_t peer = -1;
    if (!bulk_(&at, &peer, sizeof at, genai::TpDType::kF32, err)) {
      return false;
    }
    if (at != peer) {
      *err = fmt("the ranks resumed at steps {} and {}", at, peer)();
      return false;
    }
    // What the two start from, and their first update, compared
    // (tp_agree_).
    _tp_first_checks = 2;
  }
  session()->info(fmt(
      "TrainLoraStage('{}'): training {} modules at rank {} (alpha {}) for "
      "{} steps of {} over {} samples ({} a epoch{}), {} {}; checkpoints "
      "every {}{}", this->id(), bound, _rank, alpha, _total_steps,
      global_batch_(), trained, epoch_len,
      _val.empty() ? std::string()
                   : fmt("; {} held out", _val.size())(),
      _opt_spec.optimizer, _opt_spec.effective_lr(), _save_every,
      dp_rank_() ? fmt("; DATA PARALLEL rank {} of 2: samples [{}, {}) of "
                       "each step's {}", _dp_rank, local_first_(),
                       local_first_() + local_count_(), global_batch_())()
                 : std::string()));
  return true;
}

bool
TrainLoraStage::resume_(std::int64_t at, std::string* err)
{
  err->clear();
  bool others = false;
  std::vector<std::string> broken;
  const std::vector<ResumeSlot> held = resume_slots_(&others, &broken);
  if (at < 0) {   // a whole run (a RANK said so at the exchange)
    for (const std::string& b : broken) {
      session()->warn(fmt("TrainLoraStage('{}'): {}; passed over",
                          this->id(), b));
    }
  }
  if (held.empty()) {
    if (at < 0 && !broken.empty()) {
      session()->warn(fmt("TrainLoraStage('{}'): no resume state in '{}' "
                          "loads; starting over", this->id(),
                          resume_dir_()));
    }
    if (others) {
      session()->warn(fmt(
          "TrainLoraStage('{}'): a resume state is in '{}' but it is for a "
          "different run; starting over", this->id(), resume_dir_()));
    }
    return false;
  }
  // A load that failed past its check (a read error, a file changed
  // since) left part of it in the live optimizer: an older state that
  // loads overwrites all of it, a fresh start would not.
  std::string half_read;
  for (const ResumeSlot& r : held) {
    if (at >= 0 && r.step != at) { continue; }
    const std::string path = (r.dir / "optimizer.bin").string();
    std::string e;
    const bool fits = _opt.check_state(path, &e);
    if (!fits || !_opt.load_state(path, &e)) {
      // A RANK loads the one step both agreed on, or fails: the peer
      // reads its own. A whole run falls back to the older state.
      if (at >= 0) {
        *err = e;
        return false;
      }
      if (fits) { half_read = e; }
      session()->warn(fmt("TrainLoraStage('{}'): the resume state at step "
                          "{} does not load ({}); trying the one before",
                          this->id(), r.step, e));
      continue;
    }
    _step = (int)r.step;
    // What came after it is an abandoned run's: gone, before a later
    // agreement could pair it with the peer's real state at that step.
    resume_tidy_(r.step);
    session()->info(fmt("TrainLoraStage('{}'): resumed at step {} of {}",
                        this->id(), _step, _total_steps));
    return true;
  }
  if (at >= 0) {
    *err = fmt("the resume state at step {} is gone", at)();
  } else if (!half_read.empty()) {
    *err = fmt("no resume state in '{}' loads, and the last read stopped "
               "part way ({}); move the folder aside to start over",
               resume_dir_(), half_read)();
  }
  return false;
}

namespace {

// On the DRIVE, not in its cache: fsync on macOS hands the data to the
// drive, whose cache a power cut loses (F_FULLFSYNC flushes it). A file
// or the folder that names it; best effort -- a save does not fail for
// it.
void
full_sync_(const fs::path& p)
{
  const int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) { return; }
#ifdef F_FULLFSYNC
  if (::fcntl(fd, F_FULLFSYNC) == 0) {
    ::close(fd);
    return;
  }
#endif
  (void)::fsync(fd);
  ::close(fd);
}

}  // namespace

bool
TrainLoraStage::save_(bool final_save, std::string* err)
{
  std::error_code ec;
  fs::create_directories(_output_dir, ec);
  std::map<std::string, std::string> meta;
  meta["modelspec.trigger_phrase"] = _trigger;
  meta["vpipe.trigger_word"] = _trigger;
  meta["vpipe.base_model"] = _base_ref;
  meta["vpipe.family"] = _family;
  meta["vpipe.targets"] = _targets;
  meta["vpipe.steps"] = std::to_string(_step);
  meta["vpipe.samples"] = std::to_string(_samples.size());
  meta["vpipe.optimizer"] = _opt_spec.to_flex().to_json();
  if (_trainer) {
    for (const auto& [k, v] : _trainer->metadata()) { meta[k] = v; }
  }
  char nm[64];
  std::snprintf(nm, sizeof nm, "-%06d", _step);
  const std::string path =
      (fs::path(_output_dir) /
       (_name + (final_save ? std::string() : std::string(nm)) +
        ".safetensors")).string();
  // A RANK's adapter is gathered whole; rank 0 writes it.
  auto save_one = [&](const std::string& at,
                      const metal_compute::SharedBuffer* ea,
                      const metal_compute::SharedBuffer* eb) {
    // A data-parallel follower holds rank 0's adapter, bit for bit:
    // rank 0's file is the run's.
    if (_dp_rank > 0) { return true; }
    if (!_params.split()) { return _params.save(at, meta, err, ea, eb); }
    auto gather = [this](std::vector<float>& v, std::string* e) {
      std::vector<float> o(v.size());
      if (!_trainer->split_bulk(v.data(), o.data(), v.size() * 4,
                                genai::TpDType::kF32, e)) {
        return false;
      }
      for (std::size_t i = 0; i < v.size(); ++i) { v[i] += o[i]; }
      return true;
    };
    return _params.save_gathered(at, meta, _tp.rank == 0, _tp.rank == 0,
                                 gather, err, ea, eb);
  };
  if (!save_one(path, nullptr, nullptr)) { return false; }
  if (_opt.has_ema()) {
    const std::string ep =
        (fs::path(_output_dir) /
         (_name + (final_save ? std::string() : std::string(nm)) +
          "-ema.safetensors")).string();
    if (!save_one(ep, &_opt.ema_a(), &_opt.ema_b())) { return false; }
  }
  // A follower wrote nothing: its beats name no file (rank 0's is on
  // rank 0's disk), and say who wrote it.
  _saved.push_back(tp_follower_() ? std::string() : path);
  if (tp_follower_()) {
    session()->info(fmt("TrainLoraStage('{}'): step {}: the adapter {} "
                        "rank 0, which writes it", id(), _step,
                        dp_rank_() ? "held alike by" : "gathered to"));
  } else {
    session()->info(fmt("TrainLoraStage('{}'): step {}: saved '{}'{}", id(),
                        _step, path, _opt.has_ema() ? " (and its EMA)" : ""));
  }
  // The resume state beside it: the optimizer's whole state and the step,
  // in a folder of its own, written under a .tmp name and renamed whole
  // -- a host that dies mid-write leaves the state before it as it was.
  const fs::path rd = resume_dir_();
  char sn[32];
  std::snprintf(sn, sizeof sn, "step-%06d", _step);
  const fs::path slot = rd / sn;
  const fs::path tmp = rd / (std::string(sn) + ".tmp");
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp, ec);
  if (!_opt.save_state((tmp / "optimizer.bin").string(), err)) {
    return false;
  }
  FlexData run = FlexData::make_object();
  auto ro = run.as_object();
  ro.insert_or_assign("step", FlexData::make_int(_step));
  ro.insert_or_assign("seed", FlexData::make_int((std::int64_t)_run_seed));
  ro.insert_or_assign("total_steps", FlexData::make_int(_total_steps));
  ro.insert_or_assign("samples",
                      FlexData::make_int((std::int64_t)_samples.size()));
  {
    // The kind of run, the batch and -- a rank -- its rank and share.
    const FlexData rr = rank_record_();
    FlexData c = rr;
    auto co = c.as_object();
    for (const auto& k : {"mode", "batch_size", "global_batch", "rank",
                          "share", "dp_samples"}) {
      if (co.contains(k)) { ro.insert_or_assign(k, co.at(k)); }
    }
  }
  {
    std::ofstream o(tmp / "run.json");
    o << run.to_json();
    o.flush();
    if (!o) {
      *err = fmt("could not write '{}'", (tmp / "run.json").string())();
      return false;
    }
  }
  // Both files on the drive BEFORE the rename names them: a rename can
  // reach the drive ahead of the data it names, and a power cut then
  // leaves a whole-looking slot of empty files (check_state passes such
  // a slot over -- for the one before, which this keeps the next save).
  full_sync_(tmp / "optimizer.bin");
  full_sync_(tmp / "run.json");
  fs::remove_all(slot, ec);
  fs::rename(tmp, slot, ec);
  if (ec) {
    *err = fmt("could not keep the resume state '{}': {}", slot.string(),
               ec.message())();
    return false;
  }
  full_sync_(rd);
  resume_tidy_(_step);
  if (final_save) {
    // The run is complete: nothing to resume.
    fs::remove_all(rd, ec);
  }
  return true;
}

void
TrainLoraStage::register_(const std::string& path)
{
  LmdbEnv* env = session()->services()->lmdb_env();
  if (env == nullptr) { return; }
  try {
    FlexData rec = FlexData::make_object();
    auto ro = rec.as_object();
    ro.insert_or_assign("local_path", FlexData::make_string(path));
    ro.insert_or_assign("source", FlexData::make_string(_base_ref));
    ro.insert_or_assign("model_type",
                        FlexData::make_string(_adapter_type.empty()
                                                  ? _family + "-lora"
                                                  : _adapter_type));
    ro.insert_or_assign("parent_model_type",
                        FlexData::make_string(_parent_type.empty()
                                                  ? _family
                                                  : _parent_type));
    ro.insert_or_assign("param_class", FlexData::make_string("LoRA"));
    ro.insert_or_assign("trigger_word", FlexData::make_string(_trigger));
    ro.insert_or_assign("lora_trained", FlexData::make_bool(true));
    LmdbDb db(*env, kModelRegistryDb);
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadWrite);
    db.put(txn, _name, rec.to_binary());
    txn.commit();
    session()->info(fmt("TrainLoraStage('{}'): registered '{}' -> '{}'",
                        this->id(), _name, path));
  } catch (const std::exception& e) {
    session()->warn(fmt("TrainLoraStage('{}'): registry write failed: {}",
                        this->id(), e.what()));
  }
}

#endif  // VPIPE_BUILD_APPLE_SILICON

#ifdef VPIPE_BUILD_APPLE_SILICON

bool
TrainLoraStage::plan_encode_(std::string* err)
{
  FlexData man = _manifest;
  auto mo = man.as_object();
  const FlexData samples = mo.at("samples");
  auto sa = samples.as_array();
  auto str_at = [](const FlexData& arr, std::size_t i) {
    if (!arr.is_array()) { return std::string(); }
    auto a = arr.as_array();
    return i < a.size() ? std::string(a.at(i).as_string("")) : std::string();
  };
  auto bool_at = [](const FlexData& arr, std::size_t i) {
    if (!arr.is_array()) { return false; }
    auto a = arr.as_array();
    return i < a.size() && a.at(i).as_bool(false);
  };
  // A text slot: filled from the cache index now, or by the conditioner.
  auto text_slot = [&](const std::string& key, bool cached,
                       const std::string& what) {
    Text t;
    t.key = key;
    if (cached) {
      const genai::train::SampleCache::Entry* e = _cache.find(key);
      if (e == nullptr || e->shape.size() != 2) {
        *err = what + " is marked cached but is not in '" + _cache.dir() +
               "' -- was the cache changed during the run? Run again to "
               "re-encode it";
        return -1;
      }
      t.rows = (int)e->shape[0];
      if (!e->sideband.empty()) {
        t.sideband = FlexData::from_json(e->sideband);
      }
    }
    _texts.push_back(std::move(t));
    const int idx = (int)_texts.size() - 1;
    if (!cached) { _want_text.push_back(idx); }
    return idx;
  };
  for (std::size_t si = 0; si < sa.size(); ++si) {
    const FlexData e = sa.at(si);
    auto eo = e.as_object();
    const std::string file(eo.at("file").as_string(""));
    Sample s;
    s.repeats = std::max<int>(1, (int)eo.at("repeats").as_int(1));
    s.validation = eo.contains("validation") &&
                   eo.at("validation").as_bool(false);
    const FlexData caps = eo.at("captions");
    const FlexData pics = eo.at("pictures");
    const FlexData ckeys =
        eo.contains("caption_keys") ? eo.at("caption_keys") : FlexData();
    const FlexData ccached =
        eo.contains("caption_cached") ? eo.at("caption_cached") : FlexData();
    auto ca = caps.as_array();
    auto pa = pics.as_array();
    for (std::size_t i = 0; i < ca.size(); ++i) {
      const int idx = text_slot(str_at(ckeys, i), bool_at(ccached, i),
                                fmt("caption {} of '{}'", i, file)());
      if (idx < 0) { return false; }
      s.texts.push_back(idx);
    }
    for (std::size_t i = 0; i < pa.size(); ++i) {
      const FlexData pe = pa.at(i);
      auto po = pe.as_object();
      Picture p;
      p.key = po.contains("key") ? std::string(po.at("key").as_string(""))
                                 : std::string();
      if (po.contains("cached") && po.at("cached").as_bool(false)) {
        const genai::train::SampleCache::Entry* ce = _cache.find(p.key);
        if (ce == nullptr ||
            (ce->shape.size() != 3 && ce->shape.size() != 4)) {
          *err = fmt("picture {} of '{}' is marked cached but is not in "
                     "'{}' -- was the cache changed during the run? Run "
                     "again to re-encode it", i, file, _cache.dir())();
          return false;
        }
        const bool timed = ce->shape.size() == 4;
        _latent_c = (int)ce->shape[0];
        p.t = timed ? (int)ce->shape[1] : 1;
        p.h = (int)ce->shape[timed ? 2 : 1];
        p.w = (int)ce->shape[timed ? 3 : 2];
      } else {
        _want_pic.push_back(WantPic{(int)si, (int)i,
                                    (int)po.at("h").as_int(0),
                                    (int)po.at("w").as_int(0), file});
      }
      // A keyframe window's captions, presented with its keyframes: the
      // conditioner's encodes after the sample's own, in picture order.
      if (po.contains("captions")) {
        const FlexData pcap = po.at("captions");
        const FlexData pk =
            po.contains("caption_keys") ? po.at("caption_keys") : FlexData();
        const FlexData pc = po.contains("caption_cached")
                                ? po.at("caption_cached") : FlexData();
        auto pa2 = pcap.as_array();
        for (std::size_t j = 0; j < pa2.size(); ++j) {
          const int idx = text_slot(
              str_at(pk, j), bool_at(pc, j),
              fmt("presented caption {} of picture {} of '{}'", j, i,
                  file)());
          if (idx < 0) { return false; }
          p.texts.push_back(idx);
        }
      }
      // Its anchors: the stills that follow the window on the latents
      // port, or the cache's.
      if (po.contains("anchors")) {
        const FlexData an = po.at("anchors");
        const FlexData akeys =
            po.contains("anchor_keys") ? po.at("anchor_keys") : FlexData();
        const FlexData acached = po.contains("anchor_cached")
                                     ? po.at("anchor_cached") : FlexData();
        auto aa = an.as_array();
        for (std::size_t k = 0; k < aa.size(); ++k) {
          const std::string end(aa.at(k).as_string(""));
          p.anchors |= end == "first" ? 1 : end == "last" ? 2 : 0;
          p.kkeys.push_back(str_at(akeys, k));
          p.kx.emplace_back();
          if (bool_at(acached, k)) {
            const genai::train::SampleCache::Entry* ke =
                _cache.find(p.kkeys.back());
            if (ke == nullptr) {
              *err = fmt("a keyframe of picture {} of '{}' is marked cached "
                         "but is not in '{}' -- run again to re-encode it",
                         i, file, _cache.dir())();
              return false;
            }
          } else {
            _want_pic.push_back(WantPic{(int)si, (int)i,
                                        (int)po.at("h").as_int(0),
                                        (int)po.at("w").as_int(0), file,
                                        (int)k});
          }
        }
      }
      // A clip window's soundtrack: read back from the cache, or awaited
      // on the audio port in the manifest's order.
      if (po.contains("audio") && po.at("audio").as_bool(false)) {
        if (!_audio_wired) {
          ++_audio_unwired;
        } else {
          p.audio = true;
          p.akey = po.contains("audio_key")
              ? std::string(po.at("audio_key").as_string("")) : std::string();
          if (po.contains("audio_cached") &&
              po.at("audio_cached").as_bool(false)) {
            const genai::train::SampleCache::Entry* ae = _cache.find(p.akey);
            if (ae == nullptr || ae->shape.size() != 3) {
              *err = fmt("the soundtrack of picture {} of '{}' is marked "
                         "cached but is not in '{}' -- run again to "
                         "re-encode it", i, file, _cache.dir())();
              return false;
            }
            p.ac = (int)ae->shape[1];
            p.an = (int)ae->shape[2];
          } else {
            _want_audio.push_back(WantPic{(int)si, (int)i, 0, 0, file});
          }
        }
      }
      s.pics.push_back(std::move(p));
    }
    _samples.push_back(std::move(s));
  }
  // The empty prompt, then the previews.
  {
    const std::string key =
        mo.contains("null_key") ? std::string(mo.at("null_key").as_string(""))
                                : std::string();
    const bool cached =
        mo.contains("null_cached") && mo.at("null_cached").as_bool(false);
    _null_text = text_slot(key, cached, "the empty prompt");
    if (_null_text < 0) { return false; }
  }
  const FlexData pk =
      mo.contains("preview_keys") ? mo.at("preview_keys") : FlexData();
  const FlexData pc =
      mo.contains("preview_cached") ? mo.at("preview_cached") : FlexData();
  for (std::size_t i = 0; i < _preview_prompts.size(); ++i) {
    const int idx = text_slot(str_at(pk, i), bool_at(pc, i),
                              fmt("preview prompt {}", i)());
    if (idx < 0) { return false; }
    _preview_texts.push_back(idx);
  }
  return true;
}

Job
TrainLoraStage::checkpoint_(RuntimeContext& ctx, FlexData* out,
                            std::string* err)
{
  if (!save_(false, err)) {
    *err = "saving a checkpoint failed: " + *err;
    co_return;
  }
  FlexData c = FlexData::make_object();
  auto co = c.as_object();
  co.insert_or_assign("path", FlexData::make_string(_saved.back()));
  if (tp_follower_()) {
    co.insert_or_assign("written_by_rank", FlexData::make_int(0));
  }
  co.insert_or_assign("step", FlexData::make_int(_step));
  co.insert_or_assign("loss",
                      FlexData::make_real(_loss_sum / std::max(_loss_n, 1)));
  if (!_val.empty()) {
    double vl = 0.0;
    if (!validate_(&vl, err)) {
      *err = "scoring the held-out samples failed: " + *err;
      co_return;
    }
    co.insert_or_assign("val_loss", FlexData::make_real(vl));
    session()->info(fmt("TrainLoraStage('{}'): step {}: held-out loss "
                        "{:.4f} over {} samples{}", this->id(), _step, vl,
                        _val.size(),
                        _val_last < 0.0 ? std::string()
                                        : fmt(" (was {:.4f})", _val_last)()));
    _val_last = vl;
  }
  co.insert_or_assign("trigger_word", FlexData::make_string(_trigger));
  _saved_step = _step;
  _last_checkpoint = c;
  *out = c;
  co_await ctx.write(1, make_payload<FlexDataPayload>(std::move(c)));
  _loss_sum = 0.0;
  _loss_n = 0;
}

Job
TrainLoraStage::previews_(RuntimeContext& ctx, int which, int steps,
                          double cfg, std::uint64_t seed, int* n,
                          std::string* err)
{
  *n = 0;
  if (_preview_texts.empty() || _samples.empty() ||
      _samples.front().pics.empty()) {
    *err = "there are no preview prompts";
    co_return;
  }
  if (which >= (int)_preview_texts.size()) {
    *err = fmt("there are {} preview prompts; {} is not one of them",
               _preview_texts.size(), which)();
    co_return;
  }
  // The preview geometry: the first sample's first picture.
  const Picture& gp = _samples.front().pics.front();
  for (std::size_t i = 0; i < _preview_texts.size(); ++i) {
    if (which >= 0 && (int)i != which) { continue; }
    const Text& t = _texts[(std::size_t)_preview_texts[i]];
    genai::train::PreviewRequest r;
    r.text = &t.buf;
    r.text_rows = t.rows;
    r.sideband = &t.sideband;
    if (_null_text >= 0 && cfg != 1.0 &&
        _texts[(std::size_t)_null_text].rows > 0) {
      const Text& nt = _texts[(std::size_t)_null_text];
      r.neg_text = &nt.buf;
      r.neg_rows = nt.rows;
      r.neg_sideband = &nt.sideband;
    }
    r.grid_h = gp.h;
    r.grid_w = gp.w;
    r.frames = gp.t;
    r.steps = steps;
    r.cfg = (float)cfg;
    r.seed = seed + i;
    // An audio-video family's soundtrack too, when something reads it.
    genai::train::PreviewAudio pa;
    const bool want_audio = ctx.num_oports() > 3 && ctx.has_consumers(3);
    if (want_audio) { r.audio = &pa; }
    std::vector<float> chw;
    if (!_trainer->preview(r, &chw, err)) { co_return; }
    auto tb = std::make_unique<TensorBeatPayload>();
    tb->dtype = TensorBeat::DType::F32;
    const std::vector<std::int64_t> shape =
        _trainer->latent_shape(gp.t, gp.h, gp.w);
    tb->shape.assign(shape.begin(), shape.end());
    tb->resize_contiguous(chw.size());
    std::memcpy(tb->as_f32(), chw.data(), chw.size() * 4);
    FlexData sb = FlexData::make_object();
    auto so = sb.as_object();
    so.insert_or_assign("step", FlexData::make_int(_step));
    so.insert_or_assign("prompt", FlexData::make_string(_preview_prompts[i]));
    tb->sideband = sb;
    co_await ctx.write(0, std::move(tb));
    if (want_audio && !pa.latent.empty()) {
      auto ab = std::make_unique<TensorBeatPayload>();
      ab->dtype = TensorBeat::DType::F32;
      ab->shape.assign(pa.shape.begin(), pa.shape.end());
      ab->resize_contiguous(pa.latent.size());
      std::memcpy(ab->as_f32(), pa.latent.data(), pa.latent.size() * 4);
      if (pa.latents_per_second > 0.0) {
        sb.as_object().insert_or_assign(
            sideband::kLatentsPerSecond,
            FlexData::make_real(pa.latents_per_second));
      }
      ab->sideband = std::move(sb);
      co_await ctx.write(3, std::move(ab));
    }
    ++*n;
  }
}

#endif  // VPIPE_BUILD_APPLE_SILICON

Job
TrainLoraStage::serve_commands_(RuntimeContext& ctx)
{
  while (auto cmd = ctx.try_command()) {
#ifdef VPIPE_BUILD_APPLE_SILICON
    if (_phase != Phase::kTrain || !_trainer) {
      cmd->fail(_phase == Phase::kDone
                    ? "training has finished"
                    : "training has not started: the dataset is still "
                      "encoding, or the model loading");
      continue;
    }
    const std::string name(cmd->name());
    std::string err;
    // A RANK runs a command where every rank does, between steps: taken
    // now, run at the next step's agreement (tp_control_), answered then.
    if (tp_rank_() && (name == "save" || name == "preview")) {
      _tp_pending.push_back(cmd);
      continue;
    }
    if (name == "save") {
      // Twice at one step is one checkpoint.
      FlexData r;
      if (_saved_step == _step && _last_checkpoint.is_object()) {
        r = _last_checkpoint;
      } else {
        co_await checkpoint_(ctx, &r, &err);
      }
      if (!err.empty()) { cmd->fail(err); continue; }
      cmd->reply(std::move(r));
    } else if (name == "preview") {
      const FlexData a = cmd->args();
      auto ao = a.as_object();
      const int which = (int)ao.at("prompt").as_int(-1);
      const int steps = (int)ao.at("steps").as_int(0);
      const double cfg = ao.at("cfg").as_real(0.0);
      const std::int64_t seed = ao.at("seed").as_int(-1);
      int n = 0;
      co_await previews_(ctx, which, steps > 0 ? steps : _preview_steps,
                         cfg > 0.0 ? cfg : _preview_cfg,
                         seed >= 0 ? (std::uint64_t)seed : _preview_seed, &n,
                         &err);
      if (!err.empty()) { cmd->fail(err); continue; }
      FlexData r = FlexData::make_object();
      r.as_object().insert_or_assign("step", FlexData::make_int(_step));
      r.as_object().insert_or_assign("previews", FlexData::make_int(n));
      cmd->reply(std::move(r));
    } else {
      cmd->fail("unknown command '" + name + "'");
    }
#else
    cmd->fail("training needs Apple Silicon");
#endif
  }
}

Job
TrainLoraStage::process(RuntimeContext& ctx)
{
#ifndef VPIPE_BUILD_APPLE_SILICON
  ctx.signal_done();
  co_return;
#else
  auto* mc = session()->services()->metal_compute();
  auto fail = [&](const std::string& m) {
    // A RANK lets go of the run's channel BEFORE it raises (error()
    // throws, so nothing after it runs): the other rank's exchange then
    // returns at once (TpChannel: a channel let go) where it would wait
    // on a peer that is gone -- with a stop that never fires mid-step.
    if (tp_rank_() || _tp_ch) { tp_close_(); }
    session()->error(fmt("TrainLoraStage('{}'): {}", this->id(), m));
    _bar.finish();
    _phase = Phase::kDone;
    ctx.signal_done();
  };

  // Commands, between whatever this call does: a step, an encode.
  co_await serve_commands_(ctx);

  // ---- the model, the optimizer and the manifest ----------------------
  if (_phase == Phase::kStart) {
    if (!_model_latched && ctx.num_iports() > kModelPort &&
        ctx.iport_connected(kModelPort)) {
      auto mb = co_await ctx.read(kModelPort);
      _model_latched = true;
      if (const auto* f =
              mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
        if (apply_model_select_beat(f->data, _hf_dir)) { _base_ref = _hf_dir; }
      }
    }
    _opt_spec = genai::train::OptimizerSpec{};
    if (ctx.num_iports() > kOptimizerPort &&
        ctx.iport_connected(kOptimizerPort)) {
      auto ob = co_await ctx.read(kOptimizerPort);
      if (const auto* f =
              ob ? dynamic_cast<const FlexDataPayload*>(ob.get()) : nullptr) {
        std::string err;
        if (!genai::train::OptimizerSpec::from_flex(f->data, &_opt_spec,
                                                    &err)) {
          fail("the optimizer spec is not one this stage reads: " + err);
          co_return;
        }
      }
    }
    // A FOLLOWER of a split run: no dataset here -- rank 0's, encoded,
    // with the run's settings, over the run's channel.
    if (_tp_cfg.is_object() && _dp_cfg.is_object()) {
      fail("tensor_parallel and data_parallel: a run is one or the other");
      co_return;
    }
    if (par_cfg_().is_object() &&
        par_cfg_().as_object().contains("rank") &&
        par_cfg_().as_object().at("rank").as_int(0) > 0) {
      // Rank 0 refuses init_from for the run; one given here would start
      // this rank where rank 0 does not (a shared factor apart from the
      // first step).
      if (!_init_from.empty()) {
        fail(std::string(par_key_()) + ": init_from is not shared between "
             "hosts yet");
        co_return;
      }
      std::string err;
      if (!tp_open_(&err)) {
        fail(std::string(par_key_()) + ": " + err);
        co_return;
      }
      _tp_ch->set_stop([&ctx]() { return ctx.stop_requested(); });
      _bar = session()->open_progress("lora encode");
      _bar.update(0, 0, "waiting for rank 0's dataset");
      _encode_t0 = std::chrono::steady_clock::now();
      const bool got = tp_receive_(&err);
      _tp_ch->set_stop({});
      if (!got) { fail(err); co_return; }
      _encode_planned = true;
      _phase = Phase::kEncode;
      co_return;
    }
    if (!ctx.iport_connected(kManifestPort)) {
      fail("the manifest port is not wired -- connect a training-dataset");
      co_return;
    }
    auto man = co_await ctx.read(kManifestPort);
    const auto* mf =
        man ? dynamic_cast<const FlexDataPayload*>(man.get()) : nullptr;
    if (mf == nullptr || !mf->data.is_object()) {
      fail("no manifest arrived from the training-dataset");
      co_return;
    }
    _manifest = mf->data;
    {
      auto o = _manifest.as_object();
      _trigger = o.contains("trigger_word")
          ? std::string(o.at("trigger_word").as_string("")) : std::string();
      if (o.contains("preview_prompts")) {
        const FlexData pv = o.at("preview_prompts");
        auto a = pv.as_array();
        for (std::size_t i = 0; i < a.size(); ++i) {
          _preview_prompts.push_back(std::string(a.at(i).as_string("")));
        }
      }
    }
    {
      auto o = _manifest.as_object();
      const std::string cdir =
          o.contains("cache_dir")
              ? std::string(o.at("cache_dir").as_string(""))
              : std::string();
      if (!cdir.empty()) {
        std::string cerr;
        if (!_cache.open(cdir, &cerr)) {
          session()->warn(fmt("TrainLoraStage('{}'): {}; the encoded "
                              "dataset will not be cached", this->id(),
                              cerr));
        }
      }
    }
    _phase = Phase::kEncode;
    _encode_t0 = std::chrono::steady_clock::now();
    _bar = session()->open_progress("lora encode");
    co_return;
  }

  // ---- ENCODE: drain both encoders' streams as they arrive ------------
  //
  // The manifest says what arrives on each port and in which order: the
  // conditioner's beats are the uncached captions, sample by sample,
  // then the empty prompt and the previews; the VAE's are the uncached
  // pictures. The two streams are read INDEPENDENTLY, whichever has a
  // beat, and never in lockstep: a conditioner that batches its prompts
  // holds dozens of captions back, and a reader waiting on sample 0's
  // caption while the pictures pile up behind it would stall the dataset
  // that feeds both -- the batch would never fill.
  if (_phase == Phase::kEncode) {
    if (!_encode_planned) {
      _encode_planned = true;
      _audio_wired = ctx.num_iports() > kAudioPort &&
                     ctx.iport_connected(kAudioPort);
      const bool rv = ctx.num_iports() > kRefVideoPort &&
                      ctx.iport_connected(kRefVideoPort);
      const bool ra = ctx.num_iports() > kRefAudioPort &&
                      ctx.iport_connected(kRefAudioPort);
      if (rv != ra) {
        fail("ref_video_rows and ref_audio_rows are wired together "
             "(video-ref-encoder's oports 1 and 2) or not at all");
        co_return;
      }
      _refs_wired = rv;
      std::string perr;
      if (!plan_encode_(&perr)) { fail(perr); co_return; }
      if (_audio_unwired > 0) {
        session()->info(fmt("TrainLoraStage('{}'): the manifest lists {} "
                            "soundtracks and the audio_latents port is not "
                            "wired; the clips train without them",
                            this->id(), _audio_unwired));
      }
    }
    // One conditioning beat into its slot, RAM and the cache.
    auto take_text = [&](const BeatPayloadIntf* cb, int idx) {
      const auto* tb = cb ? dynamic_cast<const TensorBeatPayload*>(cb)
                          : nullptr;
      if (tb == nullptr || tb->shape.size() != 2) { return false; }
      Text& t = _texts[(std::size_t)idx];
      t.rows = (int)tb->shape[0];
      if (t.rows > 0) { t.buf = copy_beat_(mc, *tb); }
      t.sideband = tb->sideband;
      _cache_bytes += t.buf.byte_size();
      cache_put_(t.key, *tb, false);
      return true;
    };
    // A caption's reference rows (video-ref-encoder's oports 1 and 2,
    // emitted beside its conditioning): f32 rows, rounded to bf16 as
    // the forward reads them, kept in RAM and cached beside the caption.
    auto take_refs = [&](const BeatPayloadIntf* vb, const BeatPayloadIntf* ab,
                         int idx, std::string* why) {
      Text& t = _texts[(std::size_t)idx];
      auto one = [&](const BeatPayloadIntf* b, int width, const char* which,
                     metal_compute::SharedBuffer* dst, int* n) {
        const auto* tb = b ? dynamic_cast<const TensorBeatPayload*>(b)
                           : nullptr;
        if (tb == nullptr || tb->dtype != TensorBeat::DType::F32 ||
            tb->shape.size() != 2 ||
            (tb->shape[0] > 0 && tb->shape[1] != width)) {
          *why = fmt("a caption's reference {} rows did not arrive as f32 "
                     "[rows, {}] -- is video-ref-encoder's oport {} wired "
                     "here?", which, width,
                     std::string(which) == "video" ? 1 : 2)();
          return false;
        }
        *n = (int)tb->shape[0];
        // Cached even when empty (an audio-only request's video rows), so
        // a rerun reading the caption back finds both halves.
        cache_put_(ref_key_(t.key, which), *tb, true);
        if (*n == 0) { return true; }
        const auto bytes = tb->materialize_contiguous();
        const std::size_t m = bytes.size() / 4;
        *dst = mc->make_shared_buffer(m * 2);
        const auto* src = reinterpret_cast<const float*>(bytes.data());
        auto* d = static_cast<std::uint16_t*>(dst->contents());
        for (std::size_t j = 0; j < m; ++j) {
          d[j] = genai::train::f32_to_bf16(src[j]);
        }
        _cache_bytes += m * 2;
        return true;
      };
      return one(vb, 96, "video", &t.rv, &t.nrv) &&
             one(ab, 32, "audio", &t.ra, &t.nra);
    };
    // One latent into its picture: f32 [C,h,w], rounded to the storage
    // type now, exactly as packing would -- so a latent read back from
    // the cache is the SAME latent.
    auto take_latent = [&](const BeatPayloadIntf* lb, const WantPic& w,
                           std::string* why) {
      const auto* tb = lb ? dynamic_cast<const TensorBeatPayload*>(lb)
                          : nullptr;
      if (tb == nullptr || tb->dtype != TensorBeat::DType::F32 ||
          (tb->shape.size() != 3 && tb->shape.size() != 4)) {
        *why = fmt("the latent of sample {} ('{}') did not arrive as f32 "
                   "[C,h,w] (or a video VAE's [C,t,h,w]) -- vae-encode and "
                   "the training-dataset disagree", w.sample, w.file)();
        return false;
      }
      // A video VAE's latent carries a time axis: one frame for a picture.
      const bool timed = tb->shape.size() == 4;
      const int lc = (int)tb->shape[0];
      const int lt = timed ? (int)tb->shape[1] : 1;
      const int lh = (int)tb->shape[timed ? 2 : 1];
      const int lw = (int)tb->shape[timed ? 3 : 2];
      // The picture and its latent agree on a scale, or the VAE was
      // handed something else.
      if (lh <= 0 || lw <= 0 || w.ph % lh != 0 || w.pw % lw != 0 ||
          w.ph / lh != w.pw / lw) {
        *why = fmt("sample {}'s latent [{}, {}, {}] is not a picture of "
                   "{}x{}", w.sample, lc, lh, lw, w.pw, w.ph)();
        return false;
      }
      const auto bytes = tb->materialize_contiguous();
      const std::size_t n = bytes.size() / 4;
      Picture& p =
          _samples[(std::size_t)w.sample].pics[(std::size_t)w.pic];
      if (w.anchor >= 0) {
        // A keyframe's still: one latent frame of the window's grid.
        if (lt != 1) {
          *why = fmt("a keyframe of sample {} ('{}') arrived as {} latent "
                     "frames, not a still's one", w.sample, w.file, lt)();
          return false;
        }
        metal_compute::SharedBuffer& k = p.kx[(std::size_t)w.anchor];
        k = mc->make_shared_buffer(bytes.size());
        const auto* src = reinterpret_cast<const float*>(bytes.data());
        auto* dst = static_cast<float*>(k.contents());
        for (std::size_t j = 0; j < n; ++j) {
          dst[j] = genai::train::bf16_to_f32(genai::train::f32_to_bf16(src[j]));
        }
        _cache_bytes += n * 2;
        cache_put_(p.kkeys[(std::size_t)w.anchor], *tb, true);
        return true;
      }
      p.t = lt;
      p.h = lh;
      p.w = lw;
      _latent_c = lc;
      // Held as f32 [C,h,w] until the family can pack it: the model has
      // not loaded yet, on purpose.
      p.x0 = mc->make_shared_buffer(bytes.size());
      {
        const auto* src = reinterpret_cast<const float*>(bytes.data());
        auto* dst = static_cast<float*>(p.x0.contents());
        for (std::size_t j = 0; j < n; ++j) {
          dst[j] = genai::train::bf16_to_f32(
              genai::train::f32_to_bf16(src[j]));
        }
      }
      _cache_bytes += n * 2;
      cache_put_(p.key, *tb, true);
      return true;
    };

    // One soundtrack latent into its clip window: f32 [2, C, n], rounded
    // to the storage type now, as a latent is.
    auto take_audio = [&](const BeatPayloadIntf* ab, const WantPic& w,
                          std::string* why) {
      const auto* tb = ab ? dynamic_cast<const TensorBeatPayload*>(ab)
                          : nullptr;
      if (tb == nullptr || tb->dtype != TensorBeat::DType::F32 ||
          tb->shape.size() != 3 || tb->shape[0] != 2) {
        *why = fmt("the soundtrack of sample {} ('{}') did not arrive as "
                   "f32 [2, C, n] -- audio-vae-encode and the "
                   "training-dataset disagree", w.sample, w.file)();
        return false;
      }
      Picture& p =
          _samples[(std::size_t)w.sample].pics[(std::size_t)w.pic];
      p.ac = (int)tb->shape[1];
      p.an = (int)tb->shape[2];
      const auto bytes = tb->materialize_contiguous();
      const std::size_t n = bytes.size() / 4;
      p.ax0 = mc->make_shared_buffer(bytes.size());
      const auto* src = reinterpret_cast<const float*>(bytes.data());
      auto* dst = static_cast<float*>(p.ax0.contents());
      for (std::size_t j = 0; j < n; ++j) {
        dst[j] = genai::train::bf16_to_f32(genai::train::f32_to_bf16(src[j]));
      }
      _cache_bytes += n * 2;
      cache_put_(p.akey, *tb, true);
      return true;
    };

    const bool want_t = _got_text < _want_text.size();
    const bool want_p = _got_pic < _want_pic.size();
    const bool want_a = _got_audio < _want_audio.size();
    if (want_t || want_p || want_a) {
      _bar.update(_got_text + _got_pic + _got_audio,
                  _want_text.size() + _want_pic.size() + _want_audio.size(),
                  fmt("{} captions, {} pictures{}", _want_text.size(),
                      _want_pic.size(),
                      _want_audio.empty()
                          ? std::string()
                          : fmt(", {} soundtracks", _want_audio.size())())());
      std::vector<unsigned> ports;
      if (want_t && !ctx.eos(kCondPort)) { ports.push_back(kCondPort); }
      if (want_p && !ctx.eos(kLatentPort)) { ports.push_back(kLatentPort); }
      if (want_a && !ctx.eos(kAudioPort)) { ports.push_back(kAudioPort); }
      if (want_a && ctx.eos(kAudioPort)) {
        fail(fmt("the soundtracks ended after {} of {} -- audio-vae-encode "
                 "and the training-dataset disagree (is audio-vae-encode "
                 "current: it must encode a batch beat by beat?)",
                 _got_audio, _want_audio.size())());
        co_return;
      }
      if (want_t && ctx.eos(kCondPort)) {
        fail(fmt("the conditioning ended after {} of {} encodes -- the "
                 "conditioner (diffusion-conditioner or video-ref-encoder) "
                 "and the training-dataset disagree about how many captions "
                 "there are: it skipped one (see its warning above), or it "
                 "is older than allow_empty / batch prompts", _got_text,
                 _want_text.size())());
        co_return;
      }
      if (want_p && ctx.eos(kLatentPort)) {
        fail(fmt("the latents ended after {} of {} pictures -- vae-encode "
                 "and the training-dataset disagree", _got_pic,
                 _want_pic.size())());
        co_return;
      }
      co_await ctx.read_any(std::move(ports), true);
      while (_got_text < _want_text.size() && ctx.backlog(kCondPort) > 0) {
        auto cb = co_await ctx.read(kCondPort);
        const int idx = _want_text[_got_text];
        if (!take_text(cb.get(), idx)) {
          fail(fmt("conditioning {} of {} did not arrive as a [rows, dim] "
                   "tensor", _got_text, _want_text.size())());
          co_return;
        }
        if (_refs_wired) {
          // Its reference rows: written ahead of the conditioning by the
          // same request, so these reads do not wait.
          auto vb = co_await ctx.read(kRefVideoPort);
          auto ab = co_await ctx.read(kRefAudioPort);
          std::string why;
          if (!take_refs(vb.get(), ab.get(), idx, &why)) {
            fail(why);
            co_return;
          }
        } else if (names_references_(_texts[(std::size_t)idx].sideband)) {
          fail("a caption arrived encoded with references and "
               "ref_video_rows / ref_audio_rows are not wired: wire "
               "video-ref-encoder's oports 1 and 2 to train-lora's iports 6 "
               "and 7");
          co_return;
        }
        ++_got_text;
      }
      while (_got_pic < _want_pic.size() && ctx.backlog(kLatentPort) > 0) {
        auto lb = co_await ctx.read(kLatentPort);
        std::string why;
        if (!take_latent(lb.get(), _want_pic[_got_pic], &why)) {
          fail(why);
          co_return;
        }
        ++_got_pic;
        if (_cache.pending_bytes() >= ((std::size_t)256 << 20)) {
          std::string cerr;
          if (!_cache.flush(&cerr)) {
            session()->warn(fmt("TrainLoraStage('{}'): the sample cache was "
                                "not written: {}", this->id(), cerr));
          }
        }
      }
      while (_got_audio < _want_audio.size() &&
             ctx.backlog(kAudioPort) > 0) {
        auto ab = co_await ctx.read(kAudioPort);
        std::string why;
        if (!take_audio(ab.get(), _want_audio[_got_audio], &why)) {
          fail(why);
          co_return;
        }
        ++_got_audio;
      }
      if (_got_text < _want_text.size() || _got_pic < _want_pic.size() ||
          _got_audio < _want_audio.size()) {
        co_return;
      }
    }
    _bar.update(_got_text + _got_pic + _got_audio,
                _want_text.size() + _want_pic.size() + _want_audio.size(),
                fmt("{} captions, {} pictures", _want_text.size(),
                    _want_pic.size())());
    // Then the END of both streams. The encoders let go of their models
    // before they signal it, and the model below must not load beside
    // them: a trainer that starts while the conditioner still holds its
    // encoder runs its whole course with that much less room. A beat
    // past what the manifest promised is a disagreement, not a spare.
    for (const unsigned p : {kCondPort, kLatentPort, kAudioPort,
                             kRefVideoPort, kRefAudioPort}) {
      if (p == kAudioPort && !_audio_wired) { continue; }
      if ((p == kRefVideoPort || p == kRefAudioPort) && !_refs_wired) {
        continue;
      }
      while (ctx.iport_connected(p) && !ctx.eos(p)) {
        auto extra = co_await ctx.read(p);
        if (extra) {
          fail(fmt("more {} arrived than the training-dataset's manifest "
                   "lists", p == kCondPort    ? "conditionings"
                            : p == kLatentPort ? "latents"
                            : p == kAudioPort  ? "soundtracks"
                                               : "reference rows")());
          co_return;
        }
      }
    }

    std::string err;
    if (!settle_cache_(&err)) { fail(err); co_return; }
    // RANK 0 of a split run: the encoded dataset to the other rank,
    // before either loads its share.
    if (par_cfg_().is_object() && !tp_follower_()) {
      if (!_init_from.empty()) {
        fail(std::string(par_key_()) + ": init_from is not shared between "
             "hosts yet");
        co_return;
      }
      if (!tp_open_(&err)) {
        fail(std::string(par_key_()) + ": " + err);
        co_return;
      }
      _tp_ch->set_stop([&ctx]() { return ctx.stop_requested(); });
      const bool shipped = tp_ship_(&err);
      _tp_ch->set_stop({});
      if (!shipped) { fail(err); co_return; }
    }
    session()->info(fmt(
        "TrainLoraStage('{}'): encoded {} samples ({} captions, {} MB "
        "{}) in {:.1f} s; loading the model", this->id(),
        _samples.size(), _texts.size(), _cache_bytes >> 20,
        _disk ? "read from the cache every step" : "in RAM",
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      _encode_t0).count()));
    if (auto* mgr = session()->services()->generative_model_manager()) {
      // What the encoders left POOLED -- held by nobody, kept for a
      // relaunch -- is given back before the model loads: parked bytes
      // are still allocated, and the GPU's working set is what a
      // preview's decode has to fit in beside the trainer. A preview's
      // VAE decode reloads the half it needs.
      const std::size_t freed = mgr->pool_evict(mgr->pooled_bytes());
      if (freed > 0) {
        session()->info(fmt("TrainLoraStage('{}'): gave back {} MB the "
                            "encoders left pooled", this->id(),
                            freed >> 20));
      }
    }
    const auto mb0 = mc->memory_budget();
    if (auto* mgr = session()->services()->generative_model_manager()) {
      // Whatever checkpoint is still held as training starts, by whom.
      for (const auto& u : mgr->weight_report()) {
        if (u.bytes < ((std::size_t)64 << 20)) { continue; }
        session()->info(fmt(
            "TrainLoraStage('{}'): still held as training starts: {} ({} "
            "MB, {} holder(s){})", this->id(), u.dir, u.bytes >> 20,
            u.holders, u.parked ? ", parked" : ""));
      }
    }
    // The load streams a DiT of many GB in: the train report opens
    // here, indeterminate, rather than leave the encode's at its end.
    _bar = session()->open_progress("lora train");
    _bar.update(0, 0, "loading the model");
    if (!load_model_(&err)) { fail(err); co_return; }
    {
      // What the GPU's working set holds before the model and after it:
      // the first is everything else in the process -- the encoders'
      // leftovers above all -- which a preview's decode has to fit
      // beside.
      const auto mb1 = mc->memory_budget();
      session()->info(fmt(
          "TrainLoraStage('{}'): GPU working set {} MB before the model, "
          "{} MB after, of {} MB", this->id(), mb0.allocated >> 20,
          mb1.allocated >> 20, mb1.recommended >> 20));
    }
    // Pack every latent held in RAM now that the family can.
    const int C = _trainer->latent_channels();
    for (Sample& s : _samples) {
      for (Picture& p : s.pics) {
        if (p.x0.empty()) { continue; }
        const auto* chw = static_cast<const float*>(p.x0.contents());
        std::string perr;
        metal_compute::SharedBuffer packed =
            _trainer->pack_latent(chw, C, p.t, p.h, p.w, &perr);
        if (packed.empty()) { fail(perr); co_return; }
        p.x0 = std::move(packed);
      }
    }
    // Every keyframe's still, into one buffer of rows a window, first
    // then last -- the conditioning rows of its layout.
    for (Sample& s : _samples) {
      for (Picture& p : s.pics) {
        if (p.kx.empty()) { continue; }
        std::vector<metal_compute::SharedBuffer> rows;
        std::size_t total = 0;
        for (const metal_compute::SharedBuffer& k : p.kx) {
          std::string perr;
          metal_compute::SharedBuffer r = _trainer->pack_latent(
              static_cast<const float*>(k.contents()), C, 1, p.h, p.w,
              &perr);
          if (r.empty()) { fail("a keyframe: " + perr); co_return; }
          total += r.byte_size();
          rows.push_back(std::move(r));
        }
        p.cx0 = mc->make_shared_buffer(total);
        std::size_t off = 0;
        for (const metal_compute::SharedBuffer& r : rows) {
          std::memcpy(static_cast<std::uint8_t*>(p.cx0.contents()) + off,
                      r.contents(), r.byte_size());
          off += r.byte_size();
        }
        p.kx.clear();
      }
    }
    // And every soundtrack, at the latents the family lays out for its
    // clip: a window's audio latent can come one short of the count the
    // layout rounds to (the codec's hop), which is held from its last.
    {
      int dropped = 0;
      for (Sample& s : _samples) {
        for (Picture& p : s.pics) {
          if (!p.audio || p.ax0.empty()) { continue; }
          const int want = _trainer->audio_latents(p.t);
          if (want <= 0) {
            p.audio = false;
            p.ax0 = {};
            ++dropped;
            continue;
          }
          if (p.an < want - 1) {
            fail(fmt("a soundtrack latent of {} frames is too short for "
                     "its {}-latent-frame clip ({} wanted)", p.an, p.t,
                     want)());
            co_return;
          }
          const auto* src = static_cast<const float*>(p.ax0.contents());
          std::vector<float> fit((std::size_t)2 * p.ac * want);
          for (int r = 0; r < 2 * p.ac; ++r) {
            for (int i = 0; i < want; ++i) {
              fit[(std::size_t)r * want + i] =
                  src[(std::size_t)r * p.an + std::min(i, p.an - 1)];
            }
          }
          std::string perr;
          metal_compute::SharedBuffer packed = _trainer->pack_audio(
              fit.data(), 2, p.ac, want, want, &perr);
          if (packed.empty()) { fail(perr); co_return; }
          p.ax0 = std::move(packed);
        }
      }
      if (dropped > 0) {
        session()->info(fmt("TrainLoraStage('{}'): {} trains without "
                            "soundtracks; {} were dropped", this->id(),
                            _family, dropped));
      }
    }
    if (!start_training_(&err)) { fail(err); co_return; }
    // A RANK stops between steps alone, where every rank agrees to
    // (tp_control_): one stopping inside a step would leave the others'
    // exchanges waiting on partials that never come.
    if (tp_rank_()) {
      _trainer->set_stop([]() { return false; });
      // ...but the CHANNEL gives a wait up once a stop has been asked and
      // has stood for kPeerGrace: a live peer meets this rank at the next
      // agreement well inside it, so the other rank is gone, and this
      // one's user can still end it (the run fails; its last checkpoint
      // resumes). Without it the channel polls the stop above: never.
      if (_tp_ch) {
        // At least 10 minutes, and 4x the longest train call (agreement,
        // checkpoint, previews, step) this rank has made -- its waits
        // hold the peer's slower work, so a live peer stays inside it.
        _tp_grace_ms = std::make_shared<std::atomic<std::int64_t>>(600000);
        auto since = std::make_shared<std::atomic<std::int64_t>>(-1);
        auto stop = [&ctx, since, grace = _tp_grace_ms]() {
          if (!ctx.stop_requested()) { return false; }
          const std::int64_t now =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                  .count();
          std::int64_t t0 = -1;
          (void)since->compare_exchange_strong(t0, now);
          return now - since->load() > grace->load();
        };
        try {
          _tp_ch->set_stop(stop);
        } catch (...) {
        }
      }
    } else {
      _trainer->set_stop([&ctx]() { return ctx.stop_requested(); });
    }
    _phase = Phase::kTrain;
    _bar.update(_step, (std::uint64_t)_total_steps);
    co_return;
  }

  // ---- TRAIN: one optimizer step per call ------------------------------
  if (_phase == Phase::kTrain) {
    // A RANK's longest call, for its channel's patience (_tp_grace_ms).
    struct CallTime {
      std::shared_ptr<std::atomic<std::int64_t>> g;
      std::chrono::steady_clock::time_point t0 =
          std::chrono::steady_clock::now();
      ~CallTime()
      {
        if (!g) { return; }
        const std::int64_t ms = 4 *
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
        if (ms > g->load()) { g->store(ms); }
      }
    } call_time{_tp_grace_ms};
    // A RANK: what every rank does now -- stop, save, preview -- agreed
    // before each step (and the run's end), whichever rank was asked.
    if (tp_rank_()) {
      bool stopped = false;
      std::string err;
      co_await tp_agree_(ctx, &stopped, &err);
      if (!err.empty()) { fail(err); co_return; }
      if (stopped) { co_return; }
      // FAULTS for tests: this rank gone mid-run, or its copy of what
      // both hold parted from the other's.
      if (fault_("step")) {
        fail("an injected fault at step " + std::to_string(_step));
        co_return;
      }
      if (fault_("part")) {
        auto* m = static_cast<float*>(_params.a_master().contents());
        const auto sp = _params.shared_grad_spans();
        const std::size_t at = sp.empty() ? 0 : sp.front().off;
        if (_params.a_elems() > at) { m[at] += 1e-3f; }
      }
    }
    if (_step >= _total_steps) {
      std::string err;
      if (!save_(true, &err)) {
        fail("saving the adapter failed: " + err);
        co_return;
      }
      if (_register && !_saved.empty() && !tp_follower_()) {
        register_(_saved.back());
      }
      FlexData done = FlexData::make_object();
      auto d = done.as_object();
      d.insert_or_assign("path", FlexData::make_string(_saved.back()));
      if (tp_follower_()) {
        d.insert_or_assign("written_by_rank", FlexData::make_int(0));
      }
      d.insert_or_assign("step", FlexData::make_int(_step));
      d.insert_or_assign("final", FlexData::make_bool(true));
      if (!_val.empty()) {
        double vl = 0.0;
        if (validate_(&vl, &err)) {
          d.insert_or_assign("val_loss", FlexData::make_real(vl));
          session()->info(fmt("TrainLoraStage('{}'): final held-out loss "
                              "{:.4f} over {} samples{}", this->id(), vl,
                              _val.size(),
                              _val_last < 0.0
                                  ? std::string()
                                  : fmt(" (was {:.4f})", _val_last)()));
        }
      }
      d.insert_or_assign("trigger_word", FlexData::make_string(_trigger));
      co_await ctx.write(1, make_payload<FlexDataPayload>(std::move(done)));
      ++_run;
      // A sweep: another optimizer beat trains another adapter on the
      // same cache. A rank's run is its channel's: let go with it.
      tp_close_();
      _trainer.reset();
      if (auto* mgr = session()->services()->generative_model_manager()) {
        mgr->revise_declaration(_dit_dir, 0);
        for (const std::string& x : _extra_dirs) {
          mgr->revise_declaration(x, 0);
        }
      }
      _bar.finish();
      _phase = Phase::kDone;
      ctx.signal_done();
      co_return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int E = (int)_epoch_slots.size();
    double loss = 0.0, fwd = 0.0, bwd = 0.0;
    // An audio-video family's two means, unweighted.
    double loss_v = 0.0, loss_a = 0.0;
    int parts = 0, aparts = 0;
    // This rank's samples of the step (all of them, unless the step's
    // samples are shared between hosts: data_parallel).
    const int gb = global_batch_(), b0 = local_first_();
    for (int b = b0; b < b0 + local_count_(); ++b) {
      const std::uint64_t g = (std::uint64_t)_step * gb + b;
      const int epoch = (int)(g / (std::uint64_t)E);
      const int pos = (int)(g % (std::uint64_t)E);
      // This epoch's order: a seeded permutation, so resuming at any step
      // recomputes the same one.
      std::vector<int> order(E);
      std::iota(order.begin(), order.end(), 0);
      std::mt19937_64 perm(mix_(_run_seed ^ mix_((std::uint64_t)epoch)));
      std::shuffle(order.begin(), order.end(), perm);
      const Sample& s = _samples[(std::size_t)_epoch_slots[order[pos]]];
      const std::uint64_t k = mix_(_run_seed ^ mix_(g * 7 + 3));
      const Picture& p = s.pics[(std::size_t)(k % s.pics.size())];
      // A keyframe window's captions are its own (presented with them).
      const std::vector<int>& tl = p.texts.empty() ? s.texts : p.texts;
      int ti = tl[(std::size_t)((k >> 16) % tl.size())];
      if (unit_(k ^ 0xabcdefull) < _dropout && _null_text >= 0 &&
          _texts[(std::size_t)_null_text].rows > 0) {
        ti = _null_text;
      }
      Text tscratch;
      const Text* tp = nullptr;
      metal_compute::SharedBuffer xs;
      const metal_compute::SharedBuffer* xp = nullptr;
      {
        std::string rerr;
        if (!text_(ti, &tp, &tscratch, &rerr) ||
            !latent_(p, &xp, &xs, &rerr)) {
          fail("reading a sample back failed: " + rerr);
          co_return;
        }
      }
      const Text& t = *tp;
      const metal_compute::SharedBuffer& x0 = *xp;
      const double sigma = draw_sigma(
          _ts_mode, _run_seed, g, _trainer->schedule_mu(p.h, p.w),
          _logit_mean, _logit_std, _shift);
      // The noise, from (seed, sample number).
      const std::size_t n = x0.byte_size() / 2;
      metal_compute::SharedBuffer noise = mc->make_shared_buffer(n * 2);
      {
        std::mt19937_64 rng(mix_(_run_seed ^ mix_(g * 11 + 5)));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        auto* d = static_cast<std::uint16_t*>(noise.contents());
        for (std::size_t i = 0; i < n; ++i) {
          d[i] = genai::train::f32_to_bf16(nd(rng));
        }
      }
      genai::train::TrainSample ts;
      ts.x0 = &x0;
      ts.noise = &noise;
      ts.text = &t.buf;
      ts.text_rows = t.rows;
      ts.sideband = &t.sideband;
      ts.grid_h = p.h;
      ts.grid_w = p.w;
      ts.frames = p.t;
      ts.sigma = (float)sigma;
      // What the family draws itself (a missing soundtrack's noise).
      ts.seed = mix_(_run_seed ^ mix_(g * 13 + 7));
      if (p.anchors != 0) {
        ts.anchors = p.anchors;
        ts.cond_x0 = &p.cx0;
      } else if (t.nrv + t.nra > 0) {
        // The caption's references: its conditioning rows (in RAM on its
        // entry, never a scratch).
        const Text& tr = _texts[(std::size_t)ti];
        ts.cond_x0 = &tr.rv;
        ts.cond_audio = &tr.ra;
      }
      // The window's soundtrack and its noise, from (seed, sample number).
      metal_compute::SharedBuffer anoise;
      if (!p.ax0.empty()) {
        const std::size_t an = p.ax0.byte_size() / 2;
        anoise = mc->make_shared_buffer(an * 2);
        std::mt19937_64 rng(mix_(_run_seed ^ mix_(g * 17 + 9)));
        std::normal_distribution<float> nd(0.0f, 1.0f);
        auto* d = static_cast<std::uint16_t*>(anoise.contents());
        for (std::size_t i = 0; i < an; ++i) {
          d[i] = genai::train::f32_to_bf16(nd(rng));
        }
        ts.audio_x0 = &p.ax0;
        ts.audio_noise = &anoise;
        ts.audio_weight = (float)_audio_weight;
      }
      genai::train::StepStats st;
      std::string err;
      if (!_trainer->step(ts, &st, &err)) {
        if (err.empty() && ctx.stop_requested()) {
          // A stop is not a failure: keep what the run has learned.
          std::string serr;
          (void)save_(false, &serr);
          session()->info(fmt("TrainLoraStage('{}'): stopped at step {}; "
                              "the run resumes from here", this->id(),
                              _step));
          _bar.finish();
          _phase = Phase::kDone;
          ctx.signal_done();
          co_return;
        }
        fail("training step failed: " + err);
        co_return;
      }
      loss += st.loss;
      fwd += st.forward_ms;
      bwd += st.backward_ms;
      if (st.loss_video >= 0.0) { loss_v += st.loss_video; ++parts; }
      if (st.loss_audio >= 0.0) { loss_a += st.loss_audio; ++aparts; }
    }
    double gnorm = 0.0, lr = 0.0;
    std::string err;
    // A RANK: the shared factors' gradients are every rank's partial.
    if (tp_split_() && !tp_sync_gradients_(&err)) {
      fail("tensor parallel: summing the gradients: " + err);
      co_return;
    }
    // A DATA-PARALLEL rank: every gradient, and the losses, every rank's
    // samples' -- the step's, summed.
    if (dp_rank_()) {
      double sums[5] = {loss, loss_v, (double)parts, loss_a,
                        (double)aparts};
      if (!dp_sync_gradients_(sums, 5, &err)) {
        fail("data parallel: summing the gradients: " + err);
        co_return;
      }
      loss = sums[0];
      loss_v = sums[1];
      parts = (int)std::lround(sums[2]);
      loss_a = sums[3];
      aparts = (int)std::lround(sums[4]);
    }
    if (!_opt.step(gb, &gnorm, &lr, &err)) {
      fail("optimizer step failed: " + err);
      co_return;
    }
    ++_step;
    loss /= gb;
    _loss_sum += loss;
    ++_loss_n;
    const double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    // A RANK stopped: nothing written now -- a stop closes the outputs,
    // and a write to one would end this stage before the next step's
    // agreement saves the run with the other rank.
    const bool tp_stopping = tp_rank_() && ctx.stop_requested();
    if (!tp_stopping) {
      FlexData m = FlexData::make_object();
      auto o = m.as_object();
      o.insert_or_assign("step", FlexData::make_int(_step));
      o.insert_or_assign("loss", FlexData::make_real(loss));
      o.insert_or_assign("lr", FlexData::make_real(lr));
      o.insert_or_assign("grad_norm", FlexData::make_real(gnorm));
      o.insert_or_assign("sec", FlexData::make_real(sec));
      o.insert_or_assign("total", FlexData::make_int(_total_steps));
      if (parts > 0) {
        o.insert_or_assign("loss_video", FlexData::make_real(loss_v / parts));
      }
      // Over the samples that carried a soundtrack.
      if (aparts > 0) {
        o.insert_or_assign("loss_audio",
                           FlexData::make_real(loss_a / aparts));
      }
      co_await ctx.write(2, make_payload<FlexDataPayload>(std::move(m)));
    }
    {
      _bar.update((std::uint64_t)_step, (std::uint64_t)_total_steps,
                  fmt("loss {:.4f} (mean {:.4f})", loss,
                      _loss_sum / std::max(_loss_n, 1))());
    }
    if (_step == 1 || _step % 10 == 0 || _step == _total_steps) {
      session()->info(fmt(
          "TrainLoraStage('{}'): step {}/{} loss {:.4f} (mean {:.4f}) lr "
          "{:.2e} |g| {:.3f} {:.1f} s (forward {:.1f}, backward {:.1f})",
          this->id(), _step, _total_steps, loss,
          _loss_sum / std::max(_loss_n, 1), lr, gnorm, sec, fwd / 1000.0,
          bwd / 1000.0));
    }
    // ...and the next agreement -- the one the other rank holds before
    // its next step -- held now: no further call comes after a stop.
    if (tp_stopping) {
      bool stopped = false;
      co_await tp_agree_(ctx, &stopped, &err);
      if (!err.empty()) { fail(err); }
      co_return;
    }
    const bool save_now = _step % _save_every == 0 && _step < _total_steps;
    // A RANK's scheduled checkpoint and preview after the next agreement
    // (tp_agree_): a stop asked meanwhile is the other rank's too, and
    // neither then saves or previews alone.
    if (save_now && tp_rank_()) {
      _tp_due_save = true;
    } else if (save_now) {
      FlexData c;
      co_await checkpoint_(ctx, &c, &err);
      if (!err.empty()) { fail(err); co_return; }
    }
    const bool preview_now =
        _preview_every > 0 && !_preview_texts.empty() &&
        (_step % _preview_every == 0 || _step == _total_steps);
    if (preview_now && tp_rank_()) {
      _tp_due_preview = true;
    } else if (preview_now) {
      int n = 0;
      co_await previews_(ctx, -1, _preview_steps, _preview_cfg,
                         _preview_seed, &n, &err);
      if (!err.empty()) {
        session()->warn(fmt("TrainLoraStage('{}'): preview failed: {}",
                            this->id(), err));
      }
    }
    co_return;
  }
  ctx.signal_done();
#endif
}

VPIPE_REGISTER_STAGE(TrainLoraStage)
VPIPE_REGISTER_SPEC(TrainLoraStage, kSpec)

}  // namespace vpipe
