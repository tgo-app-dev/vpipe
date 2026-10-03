#include "stages/train-lora-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/qwen-image/qwen-image21-lora-trainer.h"
#include "generative-models/weight-set.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <utility>

namespace vpipe {

namespace fs = std::filesystem;

namespace {

constexpr unsigned kCondPort = 0;
constexpr unsigned kLatentPort = 1;
constexpr unsigned kManifestPort = 2;
constexpr unsigned kModelPort = 3;
constexpr unsigned kOptimizerPort = 4;

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
   .suggest_db_type = "qwen-image-21",
   .model_channel = "image-model"},
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
};

const PortSpec kOports[] = {
  {.name = "previews",
   .doc = "preview latents at each checkpoint, f32 [C,h,w], for vae-decode",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "checkpoints",
   .doc = "one beat per saved adapter {path, step, loss, trigger_word}",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "metrics",
   .doc = "one beat per step {step, loss, lr, grad_norm, sec}",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = "train-lora",
  .doc = "Train a LoRA adapter on a diffusion model from a training-"
         "dataset encoded by the model's own conditioner and VAE. Caches "
         "the encoded dataset, loads the model once the encoders are done, "
         "trains with the optimizer-select spec, and saves checkpoints "
         "(with previews) that generate-image's adapter pickers offer. "
         "Qwen-Image-2.1 today.",
  .display_name = "Train LoRA",
  .category = StageCategory::Generative,
  .iports = kIports,
  .oports = kOports,
  .attrs = kAttrs,
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
  _next_sample = 0;
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
  _step = 0;
  _saved.clear();
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

std::vector<ResourceClaim>
TrainLoraStage::declare_resources() const
{
  // The training phase's holding: the model, at its streaming floor when
  // it can stream. The encoders' phase is the conditioner's and vae-
  // encode's to declare; this stage loads only after they are done.
  std::vector<ResourceClaim> v;
  if (_hf_dir.empty()) { return v; }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const std::string dit = (fs::path(root) / "transformer").string();
  static const std::vector<std::string_view> kStems = {
      "transformer_blocks."};
  v.push_back(model_memory::weight_claim_streamable(
      dit, model_memory::streaming_floor_bytes(dit, kStems)));
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
      total += (std::size_t)std::max(_latent_c, 1) * p.h * p.w * 2;
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
  if (!t.buf.empty()) { *out = &t; return true; }
  auto* mc = session()->services()->metal_compute();
  const auto* e = _cache.find(t.key);
  if (e == nullptr) {
    *err = "caption " + t.key + " is in neither RAM nor the cache";
    return false;
  }
  scratch->rows = t.rows;
  scratch->sideband = t.sideband;
  scratch->key = t.key;
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
  if (e == nullptr || e->shape.size() != 3) {
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
  *scratch = _trainer->pack_latent(chw.data(), (int)e->shape[0], p.h, p.w,
                                  err);
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
    Text scratch;
    const Text* t = nullptr;
    metal_compute::SharedBuffer xs;
    const metal_compute::SharedBuffer* xp = nullptr;
    if (!text_(s.texts.front(), &t, &scratch, err) ||
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
    ts.sigma = (float)sigma;
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
  const std::string root = resolve_model_dir(session(), _hf_dir);
  _dit_dir = (fs::path(root) / "transformer").string();
  if (!genai::QwenImage21LoraTrainer::claims(_dit_dir)) {
    *err = "'" + _hf_dir + "' is not a model this stage can train (today: "
           "Qwen-Image-2.1)";
    return false;
  }
  _family = "qwen-image-21";

  // The largest geometry this run will train at, for the plan below.
  int max_rows = 0, max_text = 0;
  for (const Sample& s : _samples) {
    for (const Picture& p : s.pics) {
      max_rows = std::max(max_rows, p.h * p.w);
    }
  }
  for (const Text& t : _texts) { max_text = std::max(max_text, t.rows); }

  genai::QwenImage21LoraTrainer::LoadArgs a;
  a.mc = mc;
  if (!genai::MetalQwenImage21Transformer::Config::read_dims(_dit_dir,
                                                             &a.cfg)) {
    *err = "'" + _dit_dir + "' has no Qwen-Image-2.1 config";
    return false;
  }
  a.cfg.ane_ffn = _ane_forward;
  a.cfg.ane_qkv = _ane_forward;
  a.cfg.i8_gemm = _i8;
  // THE ADAPTER'S STATE: masters, gradients, shadows and the optimizer's
  // moments, per parameter -- from the dims, before anything loads.
  const std::size_t H = (std::size_t)a.cfg.hidden;
  const std::size_t FF = H * (std::size_t)a.cfg.mlp_ratio;
  const std::size_t per_block = _targets == "attn"
      ? 8 * H : 8 * H + 3 * (H + FF);
  const std::size_t params =
      per_block * (std::size_t)_rank * (std::size_t)a.cfg.n_layers;
  const std::size_t state =
      params * (10 + genai::train::Optimizer::state_bytes_per_param(
                         _opt_spec));
  const std::size_t dit_bytes = model_memory::dir_weights_bytes(_dit_dir);
  // The recompute arena at the widest bucket, from the dims alone.
  const std::size_t arena_est =
      genai::QwenImage21Trainer::arena_bytes(a.cfg, max_rows + max_text,
                                             _rank, false);
  // STREAMING, by training's own rule. The generation rule sizes a DiT
  // against the text encoder beside it; here the encoders are done and
  // released before this loads, so what shares the box is the step's
  // transient -- activations and the adapter's state -- and the cache.
  // Irreversible, so the larger headroom.
  bool stream = dit_bytes + arena_est + state + _cache_bytes +
                    model_memory::kStreamHeadroom >
                model_memory::phys_ram();
  if (const char* e = std::getenv("VPIPE_TRAIN_STREAM")) {
    stream = std::atoi(e) != 0;
  }
  a.stream_blocks = stream;
  a.ws = genai::open_weight_set(_dit_dir, session());
  auto t = genai::QwenImage21LoraTrainer::load(a, err);
  if (t == nullptr) { return false; }
  // KEEP ALL when the box holds it -- about a quarter of a step saved.
  const std::size_t keep_arena =
      genai::QwenImage21Trainer::arena_bytes(a.cfg, max_rows + max_text,
                                             _rank, true);
  const std::size_t held = stream ? model_memory::streaming_floor_bytes(
                                        _dit_dir, {"transformer_blocks."})
                                  : dit_bytes;
  bool keep = false;
  if (_recompute == "never") {
    keep = true;
  } else if (_recompute == "auto") {
    keep = held + keep_arena + state + _cache_bytes +
               model_memory::kHeadroom <= model_memory::phys_ram();
  }
  FlexData o = FlexData::make_object();
  o.as_object().insert_or_assign(genai::train::opt::kKeepAll,
                                 FlexData::make_bool(keep));
  o.as_object().insert_or_assign(genai::train::opt::kAneBackward,
                                 FlexData::make_bool(_ane_backward));
  t->set_options(o);
  _reserve_bytes = (keep ? keep_arena : arena_est) + state + _cache_bytes +
                   model_memory::kHeadroom;
  session()->info(fmt(
      "TrainLoraStage('{}'): {} DiT {} ({} MB on disk), activations {} "
      "({} MB at the widest bucket, {} rows), {} MB of cached samples",
      this->id(), _family, stream ? "STREAMED" : "resident", dit_bytes >> 20,
      keep ? "kept for every block" : "recomputed per block",
      (keep ? keep_arena : arena_est) >> 20, max_rows + max_text,
      _cache_bytes >> 20));
  _trainer = std::move(t);
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_scratch("train-lora-arena",
                        (keep ? keep_arena : arena_est) + state);
  }
  return true;
}

bool
TrainLoraStage::start_training_(std::string* err)
{
  auto* mc = session()->services()->metal_compute();
  if (!_ops.valid() && !_ops.init(mc, true, err)) { return false; }
  // The seed of this training job: a sweep's jobs follow seed_sequence.
  _run_seed = generation_seed(_seed_seq, _seed, _run);
  std::vector<genai::train::ModuleShape> mods = _trainer->modules(_targets);
  const float alpha = _alpha > 0.0 ? (float)_alpha : (float)_rank;
  if (!_params.create(mc, mods, _rank, alpha, err)) { return false; }
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
    _total_steps = std::max(1, _epochs_cfg * epoch_len / _batch);
  } else {
    _total_steps = auto_steps(trained, epoch_len, _batch);
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
  if (_resume == "auto" && !resume_(err)) {
    if (!err->empty()) { return false; }
  }
  session()->info(fmt(
      "TrainLoraStage('{}'): training {} modules at rank {} (alpha {}) for "
      "{} steps of {} over {} samples ({} a epoch{}), {} {}; checkpoints "
      "every {}", this->id(), bound, _rank, alpha, _total_steps, _batch,
      trained, epoch_len,
      _val.empty() ? std::string()
                   : fmt("; {} held out", _val.size())(),
      _opt_spec.optimizer, _opt_spec.effective_lr(), _save_every));
  return true;
}

bool
TrainLoraStage::resume_(std::string* err)
{
  err->clear();
  const fs::path dir = resume_dir_();
  std::error_code ec;
  if (!fs::is_regular_file(dir / "run.json", ec)) { return false; }
  FlexData run;
  try {
    std::ifstream in(dir / "run.json");
    run = FlexData::from_json(in);
  } catch (...) {
    return false;
  }
  if (!run.is_object()) { return false; }
  auto o = run.as_object();
  const int step = (int)o.at("step").as_int(0);
  const std::uint64_t seed = (std::uint64_t)o.at("seed").as_int(0);
  const int total = (int)o.at("total_steps").as_int(0);
  const std::int64_t samples = o.at("samples").as_int(0);
  if (seed != _run_seed || total != _total_steps ||
      samples != (std::int64_t)_samples.size()) {
    session()->warn(fmt(
        "TrainLoraStage('{}'): a resume state is in '{}' but it is for a "
        "different run (seed, length or dataset); starting over",
        this->id(), dir.string()));
    return false;
  }
  if (!_opt.load_state((dir / "optimizer.bin").string(), err)) {
    return false;
  }
  _step = step;
  session()->info(fmt("TrainLoraStage('{}'): resumed at step {} of {}",
                      this->id(), _step, _total_steps));
  return true;
}

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
  char nm[64];
  std::snprintf(nm, sizeof nm, "-%06d", _step);
  const std::string path =
      (fs::path(_output_dir) /
       (_name + (final_save ? std::string() : std::string(nm)) +
        ".safetensors")).string();
  if (!_params.save(path, meta, err)) { return false; }
  if (_opt.has_ema()) {
    const std::string ep =
        (fs::path(_output_dir) /
         (_name + (final_save ? std::string() : std::string(nm)) +
          "-ema.safetensors")).string();
    if (!_params.save(ep, meta, err, &_opt.ema_a(), &_opt.ema_b())) {
      return false;
    }
  }
  _saved.push_back(path);
  session()->info(fmt("TrainLoraStage('{}'): step {}: saved '{}'{}", id(),
                      _step, path, _opt.has_ema() ? " (and its EMA)" : ""));
  // The resume state beside it: the optimizer's whole state and the step.
  const fs::path rd = resume_dir_();
  fs::create_directories(rd, ec);
  if (!_opt.save_state((rd / "optimizer.bin").string(), err)) { return false; }
  FlexData run = FlexData::make_object();
  auto ro = run.as_object();
  ro.insert_or_assign("step", FlexData::make_int(_step));
  ro.insert_or_assign("seed", FlexData::make_int((std::int64_t)_run_seed));
  ro.insert_or_assign("total_steps", FlexData::make_int(_total_steps));
  ro.insert_or_assign("samples",
                      FlexData::make_int((std::int64_t)_samples.size()));
  {
    std::ofstream o(rd / "run.json");
    o << run.to_json();
  }
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
                        FlexData::make_string(_family + "-lora"));
    ro.insert_or_assign("parent_model_type", FlexData::make_string(_family));
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

Job
TrainLoraStage::process(RuntimeContext& ctx)
{
#ifndef VPIPE_BUILD_APPLE_SILICON
  ctx.signal_done();
  co_return;
#else
  auto* mc = session()->services()->metal_compute();
  auto fail = [&](const std::string& m) {
    session()->error(fmt("TrainLoraStage('{}'): {}", this->id(), m));
    _phase = Phase::kDone;
    ctx.signal_done();
  };

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
    co_return;
  }

  // ---- ENCODE: one sample's beats per call, in the manifest's order -----
  //
  // A caption or picture the manifest marks CACHED arrives from the cache,
  // not from the conditioner or the VAE: the dataset stage did not emit
  // it. Everything that does arrive goes into the cache too.
  if (_phase == Phase::kEncode) {
    FlexData man = _manifest;
    auto mo = man.as_object();
    const FlexData samples = mo.at("samples");
    auto sa = samples.as_array();
    auto str_at = [](const FlexData& arr, std::size_t i) {
      if (!arr.is_array()) { return std::string(); }
      auto a = arr.as_array();
      return i < a.size() ? std::string(a.at(i).as_string(""))
                          : std::string();
    };
    auto bool_at = [](const FlexData& arr, std::size_t i) {
      if (!arr.is_array()) { return false; }
      auto a = arr.as_array();
      return i < a.size() && a.at(i).as_bool(false);
    };
    // A cached text: its shape and sideband from the index, its bytes
    // later (settle_cache_ or the step that needs them).
    auto cached_text = [&](const std::string& key, int* idx) {
      const genai::train::SampleCache::Entry* e = _cache.find(key);
      if (e == nullptr || e->shape.size() != 2) { return false; }
      Text t;
      t.key = key;
      t.rows = (int)e->shape[0];
      if (!e->sideband.empty()) {
        t.sideband = FlexData::from_json(e->sideband);
      }
      _texts.push_back(std::move(t));
      *idx = (int)_texts.size() - 1;
      return true;
    };
    // One conditioning beat into RAM and the cache; false for anything
    // else.
    auto take_text = [&](const BeatPayloadIntf* cb, const std::string& key,
                         int* idx) {
      const auto* tb = cb ? dynamic_cast<const TensorBeatPayload*>(cb)
                          : nullptr;
      if (tb == nullptr || tb->shape.size() != 2) { return false; }
      Text t;
      t.buf = copy_beat_(mc, *tb);
      t.rows = (int)tb->shape[0];
      t.sideband = tb->sideband;
      t.key = key;
      _cache_bytes += t.buf.byte_size();
      cache_put_(key, *tb, false);
      _texts.push_back(std::move(t));
      *idx = (int)_texts.size() - 1;
      return true;
    };
    auto missing = [&](const std::string& what) {
      fail(fmt("{} is marked cached but is not in '{}' -- was the cache "
               "changed during the run? Run again to re-encode it", what,
               _cache.dir())());
    };
    if (_next_sample < sa.size()) {
      const FlexData e = sa.at(_next_sample);
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
      const FlexData ccached = eo.contains("caption_cached")
                                   ? eo.at("caption_cached")
                                   : FlexData();
      auto ca = caps.as_array();
      auto pa = pics.as_array();
      for (std::size_t i = 0; i < ca.size(); ++i) {
        int idx = -1;
        const std::string key = str_at(ckeys, i);
        if (bool_at(ccached, i)) {
          if (!cached_text(key, &idx)) {
            missing(fmt("caption {} of '{}'", i, file)());
            co_return;
          }
          s.texts.push_back(idx);
          continue;
        }
        auto cb = co_await ctx.read(kCondPort);
        if (!take_text(cb.get(), key, &idx)) {
          fail(fmt("the conditioning of sample {} ('{}') did not arrive -- "
                   "the diffusion-conditioner and the training-dataset "
                   "disagree about how many captions there are",
                   _next_sample, file)());
          co_return;
        }
        s.texts.push_back(idx);
      }
      for (std::size_t i = 0; i < pa.size(); ++i) {
        const FlexData pe = pa.at(i);
        auto po = pe.as_object();
        const std::string key =
            po.contains("key") ? std::string(po.at("key").as_string(""))
                               : std::string();
        if (po.contains("cached") && po.at("cached").as_bool(false)) {
          const genai::train::SampleCache::Entry* ce = _cache.find(key);
          if (ce == nullptr || ce->shape.size() != 3) {
            missing(fmt("picture {} of '{}'", i, file)());
            co_return;
          }
          Picture p;
          p.key = key;
          _latent_c = (int)ce->shape[0];
          p.h = (int)ce->shape[1];
          p.w = (int)ce->shape[2];
          s.pics.push_back(std::move(p));
          continue;
        }
        auto lb = co_await ctx.read(kLatentPort);
        const auto* tb =
            lb ? dynamic_cast<const TensorBeatPayload*>(lb.get()) : nullptr;
        if (tb == nullptr || tb->dtype != TensorBeat::DType::F32 ||
            tb->shape.size() != 3) {
          fail(fmt("the latent of sample {} ('{}') did not arrive as f32 "
                   "[C,h,w] -- vae-encode and the training-dataset disagree",
                   _next_sample, file)());
          co_return;
        }
        const int ph = (int)po.at("h").as_int(0);
        const int pw = (int)po.at("w").as_int(0);
        const int lc = (int)tb->shape[0], lh = (int)tb->shape[1],
                  lw = (int)tb->shape[2];
        // The picture and its latent agree on a scale, or the VAE was
        // handed something else.
        if (lh <= 0 || lw <= 0 || ph % lh != 0 || pw % lw != 0 ||
            ph / lh != pw / lw) {
          fail(fmt("sample {}'s latent [{}, {}, {}] is not a picture of "
                   "{}x{}", _next_sample, lc, lh, lw, pw, ph)());
          co_return;
        }
        // Rounded to the storage type now, exactly as packing would: a
        // latent read back from the cache is then the SAME latent.
        const auto bytes = tb->materialize_contiguous();
        const std::size_t n = bytes.size() / 4;
        Picture p;
        p.h = lh;
        p.w = lw;
        p.key = key;
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
        cache_put_(key, *tb, true);
        s.pics.push_back(std::move(p));
      }
      _samples.push_back(std::move(s));
      ++_next_sample;
      if (_cache.pending_bytes() >= ((std::size_t)256 << 20)) {
        std::string cerr;
        if (!_cache.flush(&cerr)) {
          session()->warn(fmt("TrainLoraStage('{}'): the sample cache was "
                              "not written: {}", this->id(), cerr));
        }
      }
      co_return;
    }
    // The empty prompt, then the previews.
    {
      const std::string key =
          mo.contains("null_key") ? std::string(mo.at("null_key").as_string(""))
                                  : std::string();
      if (mo.contains("null_cached") && mo.at("null_cached").as_bool(false)) {
        if (!cached_text(key, &_null_text)) {
          missing("the empty prompt");
          co_return;
        }
      } else {
        auto nb = co_await ctx.read(kCondPort);
        if (!take_text(nb.get(), key, &_null_text)) {
          fail("the empty prompt's conditioning did not arrive -- is the "
               "diffusion-conditioner current (it must accept allow_empty)?");
          co_return;
        }
      }
    }
    {
      const FlexData pk =
          mo.contains("preview_keys") ? mo.at("preview_keys") : FlexData();
      const FlexData pc = mo.contains("preview_cached")
                              ? mo.at("preview_cached")
                              : FlexData();
      for (std::size_t i = 0; i < _preview_prompts.size(); ++i) {
        int idx = -1;
        const std::string key = str_at(pk, i);
        if (bool_at(pc, i)) {
          if (!cached_text(key, &idx)) {
            missing(fmt("preview prompt {}", i)());
            co_return;
          }
        } else {
          auto pb = co_await ctx.read(kCondPort);
          if (!take_text(pb.get(), key, &idx)) {
            fail("a preview prompt's conditioning did not arrive");
            co_return;
          }
        }
        _preview_texts.push_back(idx);
      }
    }
    std::string err;
    if (!settle_cache_(&err)) { fail(err); co_return; }
    session()->info(fmt(
        "TrainLoraStage('{}'): encoded {} samples ({} captions, {} MB "
        "{}) in {:.1f} s; loading the model", this->id(),
        _samples.size(), _texts.size(), _cache_bytes >> 20,
        _disk ? "read from the cache every step" : "in RAM",
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      _encode_t0).count()));
    if (!load_model_(&err)) { fail(err); co_return; }
    // Pack every latent held in RAM now that the family can.
    const int C = _trainer->latent_channels();
    for (Sample& s : _samples) {
      for (Picture& p : s.pics) {
        if (p.x0.empty()) { continue; }
        const auto* chw = static_cast<const float*>(p.x0.contents());
        std::string perr;
        metal_compute::SharedBuffer packed =
            _trainer->pack_latent(chw, C, p.h, p.w, &perr);
        if (packed.empty()) { fail(perr); co_return; }
        p.x0 = std::move(packed);
      }
    }
    if (!start_training_(&err)) { fail(err); co_return; }
    _trainer->set_stop([&ctx]() { return ctx.stop_requested(); });
    _phase = Phase::kTrain;
    co_return;
  }

  // ---- TRAIN: one optimizer step per call ------------------------------
  if (_phase == Phase::kTrain) {
    if (_step >= _total_steps) {
      std::string err;
      if (!save_(true, &err)) {
        fail("saving the adapter failed: " + err);
        co_return;
      }
      if (_register && !_saved.empty()) { register_(_saved.back()); }
      FlexData done = FlexData::make_object();
      auto d = done.as_object();
      d.insert_or_assign("path", FlexData::make_string(_saved.back()));
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
      // same cache.
      _trainer.reset();
      _phase = Phase::kDone;
      ctx.signal_done();
      co_return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int E = (int)_epoch_slots.size();
    double loss = 0.0, fwd = 0.0, bwd = 0.0;
    for (int b = 0; b < _batch; ++b) {
      const std::uint64_t g = (std::uint64_t)_step * _batch + b;
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
      int ti = s.texts[(std::size_t)((k >> 16) % s.texts.size())];
      if (unit_(k ^ 0xabcdefull) < _caption_dropout && _null_text >= 0) {
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
      ts.sigma = (float)sigma;
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
    }
    double gnorm = 0.0, lr = 0.0;
    std::string err;
    if (!_opt.step(_batch, &gnorm, &lr, &err)) {
      fail("optimizer step failed: " + err);
      co_return;
    }
    ++_step;
    loss /= _batch;
    _loss_sum += loss;
    ++_loss_n;
    const double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    {
      FlexData m = FlexData::make_object();
      auto o = m.as_object();
      o.insert_or_assign("step", FlexData::make_int(_step));
      o.insert_or_assign("loss", FlexData::make_real(loss));
      o.insert_or_assign("lr", FlexData::make_real(lr));
      o.insert_or_assign("grad_norm", FlexData::make_real(gnorm));
      o.insert_or_assign("sec", FlexData::make_real(sec));
      co_await ctx.write(2, make_payload<FlexDataPayload>(std::move(m)));
    }
    if (_step == 1 || _step % 10 == 0 || _step == _total_steps) {
      session()->info(fmt(
          "TrainLoraStage('{}'): step {}/{} loss {:.4f} (mean {:.4f}) lr "
          "{:.2e} |g| {:.3f} {:.1f} s (forward {:.1f}, backward {:.1f})",
          this->id(), _step, _total_steps, loss,
          _loss_sum / std::max(_loss_n, 1), lr, gnorm, sec, fwd / 1000.0,
          bwd / 1000.0));
    }
    const bool save_now = _step % _save_every == 0 && _step < _total_steps;
    if (save_now) {
      if (!save_(false, &err)) {
        fail("saving a checkpoint failed: " + err);
        co_return;
      }
      FlexData c = FlexData::make_object();
      auto co = c.as_object();
      co.insert_or_assign("path", FlexData::make_string(_saved.back()));
      co.insert_or_assign("step", FlexData::make_int(_step));
      co.insert_or_assign("loss",
                          FlexData::make_real(_loss_sum /
                                              std::max(_loss_n, 1)));
      if (!_val.empty()) {
        double vl = 0.0;
        if (!validate_(&vl, &err)) {
          fail("scoring the held-out samples failed: " + err);
          co_return;
        }
        co.insert_or_assign("val_loss", FlexData::make_real(vl));
        session()->info(fmt("TrainLoraStage('{}'): step {}: held-out loss "
                            "{:.4f} over {} samples{}", this->id(), _step,
                            vl, _val.size(),
                            _val_last < 0.0
                                ? std::string()
                                : fmt(" (was {:.4f})", _val_last)()));
        _val_last = vl;
      }
      co.insert_or_assign("trigger_word", FlexData::make_string(_trigger));
      co_await ctx.write(1, make_payload<FlexDataPayload>(std::move(c)));
      _loss_sum = 0.0;
      _loss_n = 0;
    }
    const bool preview_now =
        _preview_every > 0 && !_preview_texts.empty() &&
        (_step % _preview_every == 0 || _step == _total_steps);
    if (preview_now) {
      // The preview geometry: the first sample's first picture.
      const Picture& gp = _samples.front().pics.front();
      for (std::size_t i = 0; i < _preview_texts.size(); ++i) {
        const Text& t = _texts[(std::size_t)_preview_texts[i]];
        genai::train::PreviewRequest r;
        r.text = &t.buf;
        r.text_rows = t.rows;
        r.sideband = &t.sideband;
        if (_null_text >= 0 && _preview_cfg != 1.0) {
          const Text& nt = _texts[(std::size_t)_null_text];
          r.neg_text = &nt.buf;
          r.neg_rows = nt.rows;
          r.neg_sideband = &nt.sideband;
        }
        r.grid_h = gp.h;
        r.grid_w = gp.w;
        r.steps = _preview_steps;
        r.cfg = (float)_preview_cfg;
        r.seed = _preview_seed + i;
        std::vector<float> chw;
        std::string perr;
        if (!_trainer->preview(r, &chw, &perr)) {
          if (!perr.empty()) {
            session()->warn(fmt("TrainLoraStage('{}'): preview failed: {}",
                                this->id(), perr));
          }
          break;
        }
        auto tb = std::make_unique<TensorBeatPayload>();
        tb->dtype = TensorBeat::DType::F32;
        tb->shape = {_trainer->latent_channels(), gp.h, gp.w};
        tb->resize_contiguous(chw.size());
        std::memcpy(tb->as_f32(), chw.data(), chw.size() * 4);
        FlexData sb = FlexData::make_object();
        auto so = sb.as_object();
        so.insert_or_assign("step", FlexData::make_int(_step));
        so.insert_or_assign("prompt",
                            FlexData::make_string(_preview_prompts[i]));
        tb->sideband = std::move(sb);
        co_await ctx.write(0, std::move(tb));
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
