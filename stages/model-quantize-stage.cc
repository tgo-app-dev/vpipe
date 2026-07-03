#include "stages/model-quantize-stage.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/quantize/arch-detect.h"
#include "generative-models/quantize/calibration.h"
#include "generative-models/quantize/model-quantizer.h"
#include "generative-models/tokenizer.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

ModelQuantizeStage::ModelQuantizeStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  config)
  : TypedStage<ModelQuantizeStage>(s, std::move(id), std::move(iports),
                                   std::move(config))
{
  // Validation is deferred to launch (see Stage::fail_config). Attribute
  // defaults live in kSpec.attrs; attr_* resolves the configured value else
  // the default.
  _src_model     = attr_str("src_model");
  _output_name   = attr_str("output_name");
  _models_db     = attr_str("models_db");
  _arch          = attr_str("arch");
  _bits          = static_cast<int>(attr_uint("bits"));
  _group_size    = static_cast<int>(attr_uint("group_size"));
  _skip_existing = attr_bool("skip_existing");
  _awq           = attr_bool("awq");
  _awq_clip      = attr_bool("awq_clip");
  _calib_dir     = attr_str("calib_dir");
  _mixed         = attr_bool("mixed");
  _high_bits     = static_cast<int>(attr_uint("high_bits"));
  _mixed_frac    = static_cast<float>(attr_real("mixed_frac"));
  _layer_prefix  = attr_str("layer_prefix");
  _n_layers      = static_cast<int>(attr_uint("n_layers"));
  if (_models_db.empty()) { _models_db = "models"; }

  if (_src_model.empty()) {
    fail_config(fmt("ModelQuantizeStage('{}'): config.src_model is required",
                    this->id()));
  }
  if (_output_name.empty()) {
    fail_config(fmt("ModelQuantizeStage('{}'): config.output_name is required",
                    this->id()));
  }
  if (_bits != 4 && _bits != 8) {
    fail_config(fmt("ModelQuantizeStage('{}'): bits must be 4 or 8 (got {})",
                    this->id(), _bits));
  }
  if (_group_size != 32 && _group_size != 64) {
    fail_config(fmt(
        "ModelQuantizeStage('{}'): group_size must be 32 or 64 (got {})",
        this->id(), _group_size));
  }
  if (_awq_clip && !_awq) {
    fail_config(fmt(
        "ModelQuantizeStage('{}'): awq_clip requires awq=true", this->id()));
  }
  // The mixed-affine decode kernels are w4g64 + w8g64 only.
  if (_mixed && (_bits != 4 || _high_bits != 8 || _group_size != 64)) {
    fail_config(fmt(
        "ModelQuantizeStage('{}'): mixed requires bits=4, high_bits=8, "
        "group_size=64", this->id()));
  }
  // NOTE: arch / n_layers / layer_prefix and the awq-calibratable check are
  // resolved at run time (quantize_once) -- they auto-detect from the source
  // config.json, which is only available after src_model is resolved.

  allocate_oports(spec().oports.size());
}

namespace {

// On-device auto-calibration corpus shape: an effective AWQ pass at 4-bit
// needs ~128 sequences of ~512 tokens. The text comes from the curated
// built-in corpus (genai::build_builtin_calibration_corpus).
constexpr int kCalibSeqs   = 128;
constexpr int kCalibSeqLen = 512;

constexpr ConfigKey kAttrs[] = {
  {.key = "src_model", .type = ConfigType::String, .required = true,
   .doc = "source model: a models-DB key (registered by model-fetch) or a "
          "bf16/f16 safetensors directory path",
   .suggest_db = "models"},
  {.key = "output_name", .type = ConfigType::String, .required = true,
   .doc = "result name -> <cwd>/models/<output_name> (registered in the "
          "models DB under this key), or an explicit path (\"/..\", \"./..\") "
          "used verbatim and not registered"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db for src lookup + output registration", .def_str = "models"},
  {.key = "bits", .type = ConfigType::Uint,
   .doc = "backbone affine bit-width (4 | 8)", .def_uint = 8},
  {.key = "group_size", .type = ConfigType::Uint,
   .doc = "affine group size (32 | 64)", .def_uint = 64},
  {.key = "arch", .type = ConfigType::String,
   .doc = "model family tag; empty => auto-detect from config.json model_type",
   .def_str = ""},
  {.key = "skip_existing", .type = ConfigType::Bool,
   .doc = "skip if the output config.json already exists", .def_bool = true},
  {.key = "awq", .type = ConfigType::Bool,
   .doc = "AWQ activation-aware smoothing (per-layer fp-equivalent scale "
          "search). For standard-layernorm stacks -- full attention AND "
          "Qwen3.5 gated-DeltaNet (Llama / Qwen2 / Qwen3 / Qwen3.5 / MOSS); "
          "blocked for Gemma FFN norms and MoE (use mixed or plain). "
          "Calibration: supply calib_dir, else it auto-calibrates on-device "
          "for the Qwen3 family (dense or Qwen3.5 hybrid)",
   .def_bool = false},
  {.key = "awq_clip", .type = ConfigType::Bool,
   .doc = "paired AWQ per-group weight clip search (needs awq); OFF by "
          "default -- can regress end drift on small calibration sets",
   .def_bool = false},
  {.key = "calib_dir", .type = ConfigType::String,
   .doc = "dir with calib_{qkv,o,gateup,down}.f32 activation stats; empty "
          "(default) => on-device auto-calibration for a known arch",
   .def_str = ""},
  {.key = "mixed", .type = ConfigType::Bool,
   .doc = "unsloth-style PER-LAYER mixed precision: promote the most-"
          "sensitive mixed_frac of layers to high_bits. Works for any "
          "standard-named family (skips non-matching layers). Requires "
          "bits=4, high_bits=8, group_size=64", .def_bool = false},
  {.key = "high_bits", .type = ConfigType::Uint,
   .doc = "promoted bit-width for mixed precision (8)", .def_uint = 8},
  {.key = "mixed_frac", .type = ConfigType::Real,
   .doc = "fraction of LAYERS promoted to high_bits", .def_real = 0.25},
  {.key = "layer_prefix", .type = ConfigType::String,
   .doc = "arch layer-root prefix for awq/mixed per-layer tensors; empty => "
          "auto-detect from config.json", .def_str = ""},
  {.key = "n_layers", .type = ConfigType::Uint,
   .doc = "layer count for awq/mixed; 0 => auto-detect from config.json",
   .def_uint = 0},
};
const StageSpec kSpec = {
  .type_name = "model-quantize",
  .doc       = "Source: one-shot offline quantization of a safetensors model "
               "to the MLX-affine group-quant format, across the families "
               "vpipe supports (Llama / Qwen2 / Qwen3 / Qwen3.5 / Gemma-4 / "
               "MOSS). arch / layer_prefix / n_layers auto-detect from the "
               "source. Linears are quantized; embeddings/heads/norms/aux "
               "modules pass through. 0 in / 0 out.",
  .display_name = "Model Quantize",
  .category  = StageCategory::Preparation,
  .iports    = {},
  .oports    = {},
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
ModelQuantizeStage::spec() const noexcept
{
  return kSpec;
}

// Register the quantized result in the models DB under `key` so downstream
// stages (e.g. text-to-speech) can reference it by key. Non-fatal: a registry
// failure just logs (the model is on disk regardless).
void
ModelQuantizeStage::register_output_(const std::string& key,
                                     const std::string& dir,
                                     const std::string& arch, int bits)
{
  LmdbEnv* env = session() ? session()->lmdb_env() : nullptr;
  if (env == nullptr) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): no lmdb_env(); not registering '{}'",
        this->id(), key));
    return;
  }
  try {
    FlexData rec = FlexData::make_object();
    auto ro = rec.as_object();
    ro.insert_or_assign("local_path", FlexData::make_string(dir));
    ro.insert_or_assign("source", FlexData::make_string(_src_model));
    ro.insert_or_assign("quantized", FlexData::make_bool(true));
    ro.insert_or_assign("bits", FlexData::make_uint((std::uint64_t)bits));
    if (!arch.empty()) {
      ro.insert_or_assign("model_type", FlexData::make_string(arch));
    }
    LmdbDb  db(*env, _models_db);
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadWrite);
    const std::string bytes = rec.to_binary();
    db.put(txn, key, bytes);
    txn.commit();
    session()->info(fmt(
        "ModelQuantizeStage('{}'): registered '{}' -> '{}' in sub-db '{}'",
        this->id(), key, dir, _models_db));
  } catch (const std::exception& e) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): registry write for '{}' failed: {}",
        this->id(), key, e.what()));
  }
}

bool
ModelQuantizeStage::quantize_once(const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;

  // Resolve the source: a models-DB key wins over a same-named path.
  const std::string src_dir =
      resolve_model_dir(session(), _models_db, _src_model);

  // Resolve the output. A bare name (or "org/name") -> <cwd>/models/<name>,
  // registered under that key; an absolute/relative path ("/..", "./..") is
  // used verbatim and NOT registered.
  const bool explicit_path =
      _output_name[0] == '/' ||
      _output_name.rfind("./", 0) == 0 || _output_name.rfind("../", 0) == 0;
  const std::string out_dir = explicit_path
      ? _output_name
      : (fs::current_path() / "models" / _output_name).string();

  if (_skip_existing && fs::exists(fs::path(out_dir) / "config.json", ec)) {
    session()->info(fmt(
        "ModelQuantizeStage('{}'): output '{}' already exists; skipping",
        this->id(), out_dir));
    return true;
  }

  metal_compute::MetalCompute* mc = session()->metal_compute();
  if (mc == nullptr) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): no metal-compute backend", this->id()));
    return false;
  }

  // Auto-detect arch / n_layers / layer_prefix + capabilities from the source
  // (config.json + a probe of the safetensors layer layout); explicit config
  // wins over the detection.
  const genai::QuantArchInfo meta = genai::detect_quant_arch(session(), src_dir);
  const std::string eff_arch = _arch.empty() ? meta.arch : _arch;
  const std::string eff_layer_prefix =
      _layer_prefix.empty() ? meta.layer_prefix : _layer_prefix;
  const int eff_n_layers = _n_layers > 0 ? _n_layers : meta.n_layers;
  session()->info(fmt(
      "ModelQuantizeStage('{}'): arch='{}' layer_prefix='{}' n_layers={} "
      "(attn {}) awq_ok={} calib_ok={}", this->id(), eff_arch,
      eff_layer_prefix, eff_n_layers, meta.n_attn_layers, meta.awq_ok,
      meta.calib_ok));
  session()->log_verbose(fmt(
      "ModelQuantizeStage('{}'): calib_dir='{}' awq={} mixed={}",
      this->id(), _calib_dir, _awq, _mixed));

  // AWQ / mixed need the per-layer geometry; validate it resolved (here, not
  // in the ctor -- it depends on the resolved source).
  if (_awq || _mixed) {
    if (eff_n_layers <= 0 || eff_layer_prefix.empty()) {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): awq/mixed need n_layers + layer_prefix; "
          "auto-detect from '{}' failed -- set them explicitly (got "
          "n_layers={}, layer_prefix='{}')",
          this->id(), src_dir, eff_n_layers, eff_layer_prefix));
      return false;
    }
  }
  // AWQ's fp-equivalent fold needs every layer to be a foldable block rooted
  // at input_layernorm (full attention OR Qwen3.5 gated-DeltaNet) with the
  // standard layernorm pair and a dense MLP. Block it for Gemma-style FFN
  // norms and MoE MLPs, where it would mis-fold or hard-fail -- mixed
  // precision and plain quant still work for those.
  if (_awq && meta.detected && !meta.awq_ok) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): AWQ is not supported for arch '{}' (not a "
        "uniform foldable stack -- Gemma FFN norms or a non-dense/MoE MLP). "
        "Use mixed=true or plain quantization instead.",
        this->id(), eff_arch));
    return false;
  }

  // On-device auto-calibration: when AWQ is on and no calib_dir was supplied,
  // produce the activation stats here (8-bit base + tapped forward over a
  // built-in corpus) instead of needing the offline HF script.
  std::string eff_calib = _calib_dir;
  fs::path tmp_q8, tmp_cal;
  if (stop()) { return false; }
  if (_awq && _calib_dir.empty()) {
    if (!meta.calib_ok) {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): on-device auto-calibration is not "
          "available for arch '{}' (only dense full-attention Qwen3-family "
          "backbones run through MetalQwenModel); supply calib_dir",
          this->id(), eff_arch));
      return false;
    }
    genai::MetalQwenModel::Config bcfg = meta.backbone;
    bcfg.n_layers = eff_n_layers;   // honor an explicit n_layers override
    auto tok = genai::Tokenizer::from_huggingface_json(
        src_dir + "/tokenizer.json", session());
    if (!tok) {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): auto-calibration needs "
          "'{}/tokenizer.json'", this->id(), src_dir));
      return false;
    }
    // Built-in corpus: ~128 sequences of ~512 tokens from the curated mlx-lm
    // calibration set, chat-template-wrapped so control tokens are exercised.
    std::vector<std::vector<std::int32_t>> corpus =
        genai::build_builtin_calibration_corpus(
            *tok, kCalibSeqs, kCalibSeqLen, /*apply_chat_template=*/true);
    if (corpus.empty()) {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): built-in calibration corpus produced "
          "no sequences", this->id()));
      return false;
    }
    tmp_q8  = fs::temp_directory_path() / ("vpipe-mqcal8-" + this->id());
    tmp_cal = fs::temp_directory_path() / ("vpipe-mqcal-" + this->id());
    fs::remove_all(tmp_q8, ec);
    fs::remove_all(tmp_cal, ec);
    if (bcfg.is_moe()) {
      // MoE: memory-safe layer-by-layer streaming forward over the bf16 SOURCE
      // (never resides the whole expert stack; no 8-bit base needed). Taps the
      // PER-EXPERT gate/up + down stats the MoE AWQ folds consume.
      session()->info(fmt(
          "ModelQuantizeStage('{}'): auto-calibrating MoE (streaming per-layer "
          "forward over {} corpus seqs)", this->id(), corpus.size()));
      std::string ce;
      if (!genai::collect_backbone_calibration_streaming(
              mc, src_dir, bcfg, corpus, tmp_cal.string(), &ce,
              (std::uint64_t)8 << 30, stop)) {
        if (stop()) {
          session()->info(fmt(
              "ModelQuantizeStage('{}'): quantize stopped during "
              "calibration", this->id()));
        } else {
          session()->warn(fmt("ModelQuantizeStage('{}'): {}",
                              this->id(), ce));
        }
        fs::remove_all(tmp_cal, ec);
        return false;
      }
      eff_calib = tmp_cal.string();
    } else {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): auto-calibrating (8-bit base + on-device "
          "taps over {} corpus seqs)", this->id(), corpus.size()));
      {
        genai::QuantizeOptions q8; q8.bits = 8; q8.group = 64;
        genai::ModelQuantizer mq8(mc);
        std::string e8;
        if (!mq8.run(src_dir, tmp_q8.string(), q8, &e8, stop)) {
          if (stop()) {
            session()->info(fmt(
                "ModelQuantizeStage('{}'): quantize stopped during "
                "calibration", this->id()));
          } else {
            session()->warn(fmt(
                "ModelQuantizeStage('{}'): calib 8-bit base: {}",
                this->id(), e8));
          }
          fs::remove_all(tmp_q8, ec);
          return false;
        }
      }
      std::string ce;
      if (!genai::collect_backbone_calibration(
              mc, tmp_q8.string(), bcfg, corpus, tmp_cal.string(), &ce,
              stop)) {
        if (stop()) {
          session()->info(fmt(
              "ModelQuantizeStage('{}'): quantize stopped during "
              "calibration", this->id()));
        } else {
          session()->warn(fmt("ModelQuantizeStage('{}'): {}",
                              this->id(), ce));
        }
        fs::remove_all(tmp_q8, ec);
        fs::remove_all(tmp_cal, ec);
        return false;
      }
      fs::remove_all(tmp_q8, ec);   // the 8-bit base is consumed
      eff_calib = tmp_cal.string();
    }
  }

  genai::QuantizeOptions opt;
  opt.bits  = _bits;
  opt.group = _group_size;
  // Standard LMs (Qwen/Llama/Gemma) quantize the embed table + lm_head (MLX
  // convention) so they reload through the affine inference path. MOSS keeps
  // bf16 embeddings (its wrapper gathers them host-side), so leave it off.
  opt.quant_embeddings = !eff_arch.empty() && eff_arch != "moss-tts-local";
  // Qwen3.5/3.6 store zero-centered RMSNorm weights and apply (1 + weight)
  // (Gemma-style); vpipe's RMSNorm kernel multiplies by `weight` directly (as
  // MLX-converted checkpoints, which pre-fold the +1), so fold +1 into the
  // affected norms when quantizing a raw-HF qwen3.5 checkpoint. Other families
  // (Qwen3/Llama standard norms; Gemma handled in the runtime) leave it off.
  opt.norm_offset = (eff_arch == "qwen3.5");
  // Gemma-4 (e4b / E2B PLE family) also quantizes the per-layer-input tensors
  // and the per-layer embed table (MLX convention): the affine gemma inference
  // path binds embed_tokens_per_layer / per_layer_input_gate /
  // per_layer_projection as quantized triples (per_layer_model_projection +
  // the norms stay bf16). Extend the default linear set with these gemma leaves
  // so a raw-HF gemma reloads through the affine path. (embed_tokens + tied
  // lm_head are already covered by quant_embeddings above.)
  if (eff_arch == "gemma4") {
    opt.quant_linears = genai::ModelQuantizer::default_quant_linears();
    opt.quant_linears.push_back("embed_tokens_per_layer");
    opt.quant_linears.push_back("per_layer_input_gate");
    opt.quant_linears.push_back("per_layer_projection");
  }
  if (_awq) {
    opt.smoothquant  = true;       // the SQ pass runs the AWQ scale search
    opt.awq_clip     = _awq_clip;
    opt.calib_dir    = eff_calib;
    opt.layer_prefix = eff_layer_prefix;
    opt.n_layers     = eff_n_layers;
  }
  if (_mixed) {
    opt.mixed      = true;
    opt.high_bits  = _high_bits;
    opt.mixed_frac = _mixed_frac;
    opt.layer_prefix = eff_layer_prefix;   // mixed needs it too (+ n_layers)
    opt.n_layers   = eff_n_layers;
  }

  if (stop()) {
    if (!tmp_cal.empty()) { fs::remove_all(tmp_cal, ec); }
    return false;
  }
  genai::ModelQuantizer mq(mc);
  std::string err;
  const bool ran = mq.run(src_dir, out_dir, opt, &err, stop);
  if (!tmp_cal.empty()) { fs::remove_all(tmp_cal, ec); }   // drop temp calib
  if (!ran) {
    // On a stop request the run aborted mid-pass: config.json is written only
    // at the very end, so the output is incomplete (partial shards are inert).
    // Leave it in place but flag it for removal -- info, not the scary warn.
    if (stop()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): quantize stopped; output '{}' is "
          "incomplete -- remove it before reuse", this->id(), out_dir));
      return false;
    }
    // warn (non-fatal): error() throws by design; an offline quantize
    // failure should surface and return false, not abort the process.
    session()->warn(fmt("ModelQuantizeStage('{}'): {}", this->id(), err));
    return false;
  }

  session()->log_normal(fmt(
      "ModelQuantizeStage('{}'): quantized '{}' -> '{}' ({}-bit g{})",
      this->id(), src_dir, out_dir, _bits, _group_size));

  // Register the result so it is referenceable by key (skip explicit paths).
  if (!explicit_path) {
    register_output_(_output_name, out_dir, eff_arch, _bits);
  }
  return true;
}

Job
ModelQuantizeStage::process(RuntimeContext& ctx)
{
  if (ctx.stop_requested()) {
    ctx.signal_done();
    co_return;
  }
  session()->info(fmt(
      "ModelQuantizeStage('{}'): quantizing '{}' -> '{}' ({}-bit g{}{}{})",
      this->id(), _src_model, _output_name, _bits, _group_size,
      _awq ? (_awq_clip ? " awq+clip" : " awq") : "",
      _mixed ? " mixed" : ""));
  quantize_once([&ctx] { return ctx.stop_requested(); });
  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(ModelQuantizeStage)
VPIPE_REGISTER_SPEC(ModelQuantizeStage, kSpec)

}  // namespace vpipe
