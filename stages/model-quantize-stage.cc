#include "stages/model-quantize-stage.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/temp-root.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "generative-models/flux2/metal-flux2-calibration.h"
#include "generative-models/krea2/metal-krea2-calibration.h"
#include "generative-models/qwen-image/metal-qwen-image-calibration.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/quantize/arch-detect.h"
#include "generative-models/quantize/calibration.h"
#include "generative-models/quantize/model-quantizer.h"
#include "generative-models/tokenizer.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {
// Defined below; forward-declared so the constructor can validate `target`.
std::string t2i_target_subdir_(const std::string& target);
}

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
  _target        = attr_str("target");
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
  _quant_modulation = attr_bool("quant_modulation");
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
  // `target` is validated at run time: its valid values depend on the model
  // (Krea-2 components dit|text_encoder|vae vs a general-LLM submodule scope
  // all|text|vision|audio|<prefix>), which is only known once src is resolved.
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

// The text-to-image DiTs (Krea-2's Qwen-Image MMDiT and FLUX.2's FLUX-topology
// transformer) are diffusion transformers, not LMs: the config.json carries a
// `_class_name` and no LM stack (no model_type / layers.N / embed_tokens), so
// the LM arch-detect + AWQ/mixed/embedding machinery doesn't apply. They
// quantize as a plain group-affine linear pass over a per-family DiT leaf set.
//
// Map a transformer config `_class_name` to the vpipe family tag, or "" when it
// is not a recognised text-to-image DiT. Krea-2 and FLUX.2 ship the SAME
// diffusers pipeline shape (transformer/ + text_encoder/ + vae/ + tokenizer/ +
// scheduler/), so they share the self-contained, chainable pipeline-quantize
// path below; only the DiT class-name and the DiT quant leaf set differ.
std::string
dit_class_family_(const std::string& class_name)
{
  if (class_name == "Krea2Transformer2DModel") { return "krea2"; }
  if (class_name == "Flux2Transformer2DModel") { return "flux2"; }
  if (class_name == "QwenImageTransformer2DModel") { return "qwen-image-edit"; }
  return {};
}

// Resolve the DiT weight dir + family from what the user pointed `src_model`
// at: either the transformer/ dir itself (its config.json IS the DiT config),
// or the stock pipeline ROOT (a diffusers layout with no top-level config.json
// -- the DiT lives under transformer/, siblings text_encoder/, tokenizer/,
// vae/). Sets *family to the vpipe tag and returns the DiT dir, or "" if it's
// not a text-to-image DiT. The rest of the DiT path relies on parent_path()
// being the pipeline root (for AWQ calibration, which loads the encoder +
// tokenizer), so returning the transformer/ subdir keeps that invariant for the
// root case too.
std::string
resolve_t2i_dit_dir_(const std::string& src_dir, std::string* family)
{
  namespace fs = std::filesystem;
  auto family_of = [](const fs::path& cfg) -> std::string {
    std::ifstream in(cfg);
    if (!in) { return {}; }
    FlexData c = FlexData::from_json(in);
    if (!c.is_object()) { return {}; }
    auto o = c.as_object();
    if (!o.contains("_class_name")) { return {}; }
    return dit_class_family_(std::string(o.at("_class_name").as_string("")));
  };
  std::string fam = family_of(fs::path(src_dir) / "config.json");
  if (!fam.empty()) { if (family) { *family = fam; } return src_dir; }
  const fs::path sub = fs::path(src_dir) / "transformer";
  fam = family_of(sub / "config.json");
  if (!fam.empty()) { if (family) { *family = fam; } return sub.string(); }
  // Fallback: the diffusers pipeline root carries a model_index.json whose
  // `transformer` entry names the DiT class ([library, class_name]). Use it when
  // transformer/config.json is absent or has lost its _class_name (e.g. a
  // re-exported checkpoint) -- it is the canonical pipeline descriptor.
  {
    std::ifstream in(fs::path(src_dir) / "model_index.json");
    if (in) {
      FlexData mi = FlexData::from_json(in);
      if (mi.is_object()) {
        auto o = mi.as_object();
        if (o.contains("transformer")) {
          FlexData t = o.at("transformer");
          if (t.is_array()) {
            auto arr = t.as_array();
            if (arr.size() >= 2) {
              fam = dit_class_family_(std::string(arr[1].as_string("")));
              if (!fam.empty() && fs::is_directory(sub)) {
                if (family) { *family = fam; }
                return sub.string();
              }
            }
          }
        }
      }
    }
  }
  return {};
}

// Canonicalise the `target` config value to a pipeline component sub-dir name
// (transformer | text_encoder | vae), or "" if unrecognised. Empty input
// defaults to the DiT (the transformer). Family-agnostic (both share layout).
std::string
t2i_target_subdir_(const std::string& target)
{
  if (target.empty() || target == "dit" || target == "transformer") {
    return "transformer";
  }
  if (target == "text_encoder" || target == "text-encoder" ||
      target == "encoder" || target == "text") {
    return "text_encoder";
  }
  if (target == "vae") { return "vae"; }
  return {};
}

// Recursively materialise `src` at `dst`, hard-linking regular files when the
// two live on one filesystem (instant, no extra space -- the copies are
// read-only) and falling back to a byte copy across devices. Used to assemble
// a self-contained Krea-2 / FLUX.2 / Qwen-Image-Edit output from the un-
// quantized (or already-quantized) components this pass does not touch.
//
// Polls `stop` before every file/subdir so a pipeline stop is honored PER FILE
// (not just between top-level components): assembling a self-contained output
// copies the sibling sub-models, and on a cross-device dst that is a real byte
// copy of large shards (e.g. the ~20 GB w4 DiT when quantizing the text encoder
// on top). Returns false as soon as the stop fires (a partial `dst` is left for
// the caller to discard), true on completion.
bool
link_or_copy_tree_(const std::filesystem::path& src,
                   const std::filesystem::path& dst, std::error_code& ec,
                   const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  if (stop()) { return false; }
  if (fs::is_directory(src)) {
    fs::create_directories(dst, ec);
    for (const auto& e : fs::directory_iterator(src, ec)) {
      if (!link_or_copy_tree_(e.path(), dst / e.path().filename(), ec, stop)) {
        return false;
      }
    }
    return true;
  }
  if (!fs::is_regular_file(src)) { return true; }
  fs::remove(dst, ec);
  std::error_code le;
  fs::create_hard_link(src, dst, le);
  if (le) {   // cross-device / unsupported -> real copy
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  }
  return true;
}

// Resolve a general-LLM `target` to a submodule quantization SCOPE. Sets
// *scope (a tensor-name substring; "" = the whole model) and *all_in_scope
// (quantize every 2D linear in scope -- for towers whose linear leaves are
// non-standard). Named aliases (text/language, vision, audio) resolve against
// the model's ACTUAL tensor names; an unrecognised value is used as a literal
// prefix. Returns false + sets *err when a named submodule is absent.
bool
resolve_llm_target_(const std::string& src_dir, const std::string& target,
                    std::string* scope, bool* all_in_scope, std::string* err)
{
  *scope = "";
  *all_in_scope = false;
  std::string t;
  for (char c : target) { t.push_back((char)std::tolower((unsigned char)c)); }
  if (t.empty() || t == "all" || t == "model" || t == "full" ||
      t == "everything") {
    return true;                       // whole model, standard leaf set
  }

  std::vector<std::string> cands;
  bool broad = false, backbone = false;
  if (t == "text" || t == "language" || t == "language_model" ||
      t == "backbone" || t == "llm" || t == "decoder") {
    cands = {"language_model.", "model.language_model."};
    backbone = true;                   // standard leaves -> leaf-gated
  } else if (t == "vision" || t == "visual" || t == "image") {
    cands = {"visual.", "vision_tower.", "vision_model.", "vision."};
    broad = true;
  } else if (t == "audio" || t == "speech") {
    cands = {"audio_tower.", "audio_model.", "audio."};
    broad = true;
  } else {
    cands = {target};                  // explicit tensor-name prefix
    broad = true;
  }

  std::vector<std::string> names;
  auto wts = genai::MetalLlamaWeights::open_model(src_dir);
  if (wts.has_value()) { names = wts->tensor_names(); }
  auto present = [&](const std::string& pre) {
    for (const auto& n : names) {
      if (n.find(pre) != std::string::npos) { return true; }
    }
    return false;
  };
  for (const auto& c : cands) {
    if (present(c)) { *scope = c; *all_in_scope = broad; return true; }
  }
  if (backbone) {
    return true;   // a plain LLM has no separate backbone submodule => all
  }
  *err = fmt("no submodule matching target '{}' found in the model",
             target)();
  return false;
}

// The DiT's quantizable Linear leaves (the segment before ".weight"), per
// family. Krea-2 (Qwen-Image MMDiT): the 28 blocks' attn (to_q/k/v/gate +
// to_out.0 -> leaf "0") and SwiGLU (gate/up/down), the text-fusion blocks,
// txt_in / time_embed (linear_1/2), img_in, time_mod_proj and
// final_layer.linear. `projector` (K=12) stays dense (not group-divisible).
// "0" uniquely matches attn.to_out.0 in the DiT.
//
// FLUX.2 (FLUX topology, Flux2Transformer2DModel): the double-stream blocks'
// attn (to_q/k/v + to_out.0 + to_add_out, plus add_{q,k,v}_proj when present)
// and gated MLP (ff.linear_in/out, ff_context.linear_in/out); the single-stream
// blocks' fused attn+MLP (to_qkv_mlp_proj in, to_out out); and the embedders
// (x_embedder, context_embedder, final proj_out). The small AdaLayerNorm
// modulation linears (norm*.linear) are LEFT bf16 (matched by nothing here) --
// they are cheap and quality-sensitive. NOTE: verify against the actual
// safetensors tensor names on first run + keep in lock-step with the flux2 DiT
// loader's expected quant leaf set.
std::vector<std::string>
dit_quant_linears_(const std::string& family)
{
  if (family == "flux2") {
    // The big per-block compute Linears only. The embedders (x_embedder K=128
    // -> only 2 groups/row at g64; context_embedder; final proj_out) are left
    // bf16: they are precision-sensitive input/output projections that all image
    // info flows through, and 4-bit them dominates the DiT quant error.
    return {"to_q", "to_k", "to_v", "to_out", "to_add_out",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_qkv_mlp_proj",
            "linear_in", "linear_out"};
  }
  if (family == "qwen-image-edit") {
    // Dual-stream QwenImageTransformer2DModel. The quantizer matches the LAST
    // dot-component before ".weight", so use those: to_out is "attn.to_out.0"
    // -> "0" (60, unique); both FeedForward up-projs "*_mlp.net.0.proj" ->
    // "proj" (120); both down-projs "*_mlp.net.2" -> "2" (120). The adaLN
    // modulation ("*_mod.1" -> "1"), img_in/txt_in and the norm_out/proj_out
    // head stay bf16 (precision-sensitive; the residual reaches ~1e7).
    return {"to_q", "to_k", "to_v", "0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            "proj", "2"};
  }
  return {"to_q", "to_k", "to_v", "to_gate", "0", "gate", "up", "down",
          "linear_1", "linear_2", "img_in", "time_mod_proj", "linear"};
}

// The DiT's main-block count (config.json num_layers; default 28) -- the
// per-layer mixed-precision ranking runs over transformer_blocks.{0..N-1}.
int
dit_num_layers_(const std::string& src_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(src_dir) / "config.json");
  if (in) {
    FlexData cfg = FlexData::from_json(in);
    if (cfg.is_object()) {
      auto obj = cfg.as_object();
      if (obj.contains("num_layers")) {
        const int n = (int)obj.at("num_layers").as_int(0);
        if (n > 0) { return n; }
      }
    }
  }
  return 28;
}

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
  {.key = "target", .type = ConfigType::String,
   .doc = "which part of the model to quantize. Text-to-image (Krea-2 / "
          "FLUX.2): a component -- dit (default) | text_encoder | vae -- the "
          "output is a SELF-CONTAINED pipeline (all copied, the target "
          "quantized) usable as an hf_dir; chain passes. General / multi-modal "
          "LLM: a submodule "
          "SCOPE -- all (default, whole model) | text (language backbone only) "
          "| vision | audio | an explicit tensor-name prefix -- quantizes just "
          "that submodule, leaving the rest bf16 (e.g. quantize the LM "
          "backbone, keep the vision tower full precision)",
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
  {.key = "quant_modulation", .type = ConfigType::Bool,
   .doc = "Qwen-Image-Edit DiT only: also quantize the AdaLN modulation "
          "projections (*_mod.1). They are the largest weights (~13 GB bf16 -> "
          "~3.4 GB at 4-bit, which lets the whole DiT fit a 16 GB box) but "
          "precision-sensitive, so they are kept bf16 by default -- opt in here",
   .def_bool = false},
};
// Trigger iport (optional, any beat) + summary oport -- see model-fetch
// for the shared "preparation recipe" rationale.
const PortSpec kIports[] = {
  {.name = "trigger",
   .doc  = "optional pacing trigger (any beat type); when wired, the work "
           "waits for one beat before running -- lets these preparation "
           "stages cascade into a recipe",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "summary",
   .doc  = "FlexData summary of the completed work; its `text` field "
           "renders a report via save-text, and the beat also triggers "
           "the next stage in a recipe",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "model-quantize",
  .doc       = "Source: one-shot offline quantization of a safetensors model "
               "to the MLX-affine group-quant format, across the families "
               "vpipe supports (Llama / Qwen2 / Qwen3 / Qwen3.5 / Gemma-4 / "
               "MOSS). arch / layer_prefix / n_layers auto-detect from the "
               "source. Linears are quantized; embeddings/heads/norms/aux "
               "modules pass through. For a Krea-2 (text-to-image) model, "
               "`target` selects the component (dit | text_encoder | vae) and "
               "the output is a SELF-CONTAINED pipeline (all components copied, "
               "the target quantized) usable directly as a text-to-image "
               "hf_dir -- chain passes to quantize more than one component. "
               "Optional trigger in / summary out.",
  .display_name = "Model Quantize",
  .category  = StageCategory::Preparation,
  .iports    = kIports,
  .oports    = kOports,
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

  // ---- Text-to-image (Krea-2 / FLUX.2): a multi-component pipeline. ----
  std::string t2i_family;
  const std::string dit_dir = resolve_t2i_dit_dir_(src_dir, &t2i_family);
  if (!dit_dir.empty()) {
    const bool is_root = (dit_dir != src_dir);   // src is the pipeline ROOT
    if (is_root) {
      // Self-contained pass: copy every component, quantize the `target` one
      // (default the DiT), register as a full pipeline usable as an hf_dir.
      // Chainable across passes (DiT, then text_encoder, ...).
      return quantize_t2i_pipeline_(src_dir, t2i_family, out_dir, stop);
    }
    // src is a bare transformer dir -> transformer-only output for the
    // text-to-image `dit_dir` override (registered "<family>-dit"). Only the
    // DiT is present, so `target` other than dit doesn't apply.
    if (!_target.empty() && t2i_target_subdir_(_target) != "transformer") {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): src '{}' is a bare DiT dir; target='{}' "
          "needs the full pipeline root (with {}/) -- point src_model at the "
          "model root", this->id(), src_dir, _target,
          t2i_target_subdir_(_target)));
      return false;
    }
    if (!quantize_dit_component_(dit_dir, out_dir, t2i_family,
                                 fs::path(dit_dir).parent_path().string(),
                                 stop)) {
      return false;
    }
    session()->log_normal(fmt(
        "ModelQuantizeStage('{}'): quantized {} DiT '{}' -> '{}' ({}-bit "
        "g{}{}{})", this->id(), t2i_family, src_dir, out_dir, _bits,
        _group_size, _mixed ? " mixed" : "", _awq ? " awq-clip" : ""));
    // "<family>-dit" (transformer-only) -> the text-to-image dit_dir slot.
    if (!explicit_path) {
      register_output_(_output_name, out_dir, t2i_family + "-dit", _bits);
    }
    return true;
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

  // Submodule scope (general / multi-modal LLM `target`): restrict the pass to
  // one part of the model (e.g. quantize the language backbone, keep the
  // vision tower bf16). Resolved against the model's tensor names.
  std::string quant_scope;
  bool quant_all_in_scope = false;
  {
    std::string serr;
    if (!resolve_llm_target_(src_dir, _target, &quant_scope,
                             &quant_all_in_scope, &serr)) {
      session()->warn(fmt("ModelQuantizeStage('{}'): {}", this->id(), serr));
      return false;
    }
    if (!quant_scope.empty() && quant_all_in_scope && (_awq || _mixed)) {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): awq/mixed operate on the language "
          "backbone; target '{}' (a vision/audio/explicit submodule) supports "
          "plain quantization only", this->id(), _target));
      return false;
    }
    if (!quant_scope.empty()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): target '{}' -> quantizing only the "
          "submodule scope '{}'{}", this->id(), _target, quant_scope,
          quant_all_in_scope ? " (all linears in scope)" : ""));
    }
  }

  // AWQ / mixed need the per-layer geometry; validate it resolved (here, not
  // in the ctor -- it depends on the resolved source).
  if (_awq || _mixed) {
    if (eff_n_layers <= 0 || eff_layer_prefix.empty()) {
      // A diffusion pipeline reaching the general LM path means its DiT was not
      // recognised as a text-to-image transformer (resolve_t2i_dit_dir_ returned
      // empty) -- typically a build that predates the model's DiT-quant support,
      // or a transformer/ config whose _class_name is unknown. Give that hint
      // instead of the bare auto-detect failure.
      const bool looks_t2i =
          fs::exists(fs::path(src_dir) / "model_index.json") ||
          fs::exists(fs::path(src_dir) / "transformer" / "config.json");
      if (looks_t2i) {
        session()->warn(fmt(
            "ModelQuantizeStage('{}'): '{}' looks like a text-to-image pipeline "
            "(has model_index.json / transformer/), but its DiT was not routed "
            "to the DiT-quantize path -- the build likely predates this model's "
            "support, or its transformer _class_name is unrecognised. Rebuild "
            "from a current tree (or point src_model at the transformer/ dir).",
            this->id(), src_dir));
        return false;
      }
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
  // On-device AWQ auto-calibration (8-bit base + tapped forward over the
  // built-in text corpus) when AWQ is on and no calib_dir was supplied.
  std::string eff_calib = _calib_dir;
  fs::path auto_calib;
  if (stop()) { return false; }
  if (_awq && _calib_dir.empty()) {
    eff_calib = auto_calibrate_backbone_(src_dir, meta, eff_n_layers,
                                         src_dir + "/tokenizer.json", stop);
    if (eff_calib.empty()) { return false; }   // logged inside
    auto_calib = eff_calib;
  }

  genai::QuantizeOptions opt;
  opt.bits  = _bits;
  opt.group = _group_size;
  // Standard LMs (Qwen/Llama/Gemma) quantize the embed table + lm_head (MLX
  // convention) so they reload through the affine inference path. Every MOSS
  // variant (moss-tts / moss-tts-local / moss-tts-realtime) keeps bf16
  // embeddings + heads (its wrapper gathers them host-side, and the realtime
  // tied text head reads language_model.embed_tokens), so leave it off.
  const bool moss = eff_arch.rfind("moss-tts", 0) == 0;
  opt.quant_embeddings = !eff_arch.empty() && !moss;
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
  opt.quant_scope        = quant_scope;         // "" => the whole model
  opt.quant_all_in_scope = quant_all_in_scope;

  if (stop()) {
    if (!auto_calib.empty()) { fs::remove_all(auto_calib, ec); }
    return false;
  }
  genai::ModelQuantizer mq(mc);
  std::string err;
  const bool ran = mq.run(src_dir, out_dir, opt, &err, stop);
  if (!auto_calib.empty()) { fs::remove_all(auto_calib, ec); }   // temp calib
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

bool
ModelQuantizeStage::quantize_dit_component_(
    const std::string& dit_dir, const std::string& out_dir,
    const std::string& family, const std::string& calib_root,
    const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  metal_compute::MetalCompute* mc = session()->metal_compute();
  if (mc == nullptr) { return false; }

  const bool is_flux2 = (family == "flux2");
  const bool is_qie   = (family == "qwen-image-edit");
  const bool awq = _awq;

  // Plain group-affine over the DiT leaf set (no LM arch-detect / embedding
  // quant; the DiT loader folds the zero-centered norms' +1 itself, so
  // norm_offset stays off). dit_family lets the quantizer pick the flux2
  // AWQ-calib mapping + smoothing fold + two-prefix mixed ranking.
  genai::QuantizeOptions opt;
  opt.bits  = _bits;
  opt.group = _group_size;
  opt.quant_linears    = dit_quant_linears_(family);
  // Opt-in: also quantize the QIE AdaLN modulation (*_mod.1 -> leaf "1", the
  // largest weights, kept bf16 by default for precision). The DiT loader runs a
  // quantized modulation through gemm_bias_q, so a model built with this flag
  // loads + infers on a 16 GB box.
  if (is_qie && _quant_modulation) {
    opt.quant_linears.push_back("1");
    // The modulation carries large-magnitude scale/gate values (they drive the
    // ~1e7 residual), so 4-bit wrecks it (block-0 rel-L2 ~0.75) while 8-bit is
    // fine. Force w8 for the modulation regardless of the body's bit-width; the
    // DiT loader auto-detects per-tensor bits, so body @ bits, modulation @ 8.
    opt.high_bit_leaves.push_back("1");
    opt.high_bits = 8;
    session()->info(fmt(
        "ModelQuantizeStage('{}'): quantizing the AdaLN modulation (*_mod.1) "
        "at 8-bit (precision-sensitive; body stays {}-bit)", this->id(),
        _bits));
  }
  opt.quant_embeddings = false;
  opt.norm_offset      = false;
  opt.dit_family       = family;   // "krea2" | "flux2"
  const int dit_layers = dit_num_layers_(dit_dir);
  fs::path tmp_cal;
  if (_mixed) {
    // Per-block mixed precision (bits=4, high_bits=8, group=64 -- enforced in
    // the ctor). Krea-2 ranks the single transformer_blocks prefix; FLUX.2 ranks
    // both block prefixes (flagged via dit_family in the quantizer).
    opt.mixed      = true;
    opt.high_bits  = _high_bits;
    opt.mixed_frac = _mixed_frac;
    opt.n_layers   = dit_layers;
    if (!is_flux2) { opt.layer_prefix = "transformer_blocks."; }
  }
  if (awq) {
    // DiT AWQ = activation-aware weight CLIPPING. On-device auto-calibrate over
    // prompts x sigmas (reading the encoder + tokenizer from calib_root) when
    // no calib_dir is supplied. Krea-2 and FLUX.2 use family-specific collectors
    // (different encoder / template / tap groups); the quantizer reads the
    // right calib layout via opt.dit_family.
    if (!_calib_dir.empty()) {
      opt.calib_dir = _calib_dir;
    } else {
      tmp_cal = vpipe::temp_root() /
                ((is_flux2 ? "vpipe-flux2-ditcal-"
                  : is_qie ? "vpipe-qie-ditcal-"
                  : "vpipe-krea2-ditcal-") + this->id());
      fs::remove_all(tmp_cal, ec);
      session()->info(fmt(
          "ModelQuantizeStage('{}'): on-device {} DiT AWQ calibration "
          "(prompts x sigmas)", this->id(), family));
      std::string ce;
      const bool ok = is_flux2
          ? genai::collect_flux2_calibration(
                mc, calib_root, genai::default_dit_calibration_prompts(), 8, 256,
                256, 0, tmp_cal.string(), &ce, stop)
          : is_qie
          ? genai::collect_qwen_image_calibration(
                mc, calib_root, genai::default_dit_calibration_prompts(), 8, 256,
                256, 0, tmp_cal.string(), &ce, stop)
          : genai::collect_dit_calibration(
                mc, calib_root, genai::default_dit_calibration_prompts(), 8, 256,
                256, 0, tmp_cal.string(), &ce, stop);
      if (!ok) {
        fs::remove_all(tmp_cal, ec);
        if (stop()) {
          session()->info(fmt(
              "ModelQuantizeStage('{}'): stopped during DiT calibration",
              this->id()));
        } else {
          session()->warn(fmt("ModelQuantizeStage('{}'): DiT calib: {}",
                              this->id(), ce));
        }
        return false;
      }
      opt.calib_dir = tmp_cal.string();
    }
    opt.dit_awq    = true;
    opt.dit_family = family;   // "krea2" | "flux2"
    opt.n_layers   = dit_layers;
  }
  genai::ModelQuantizer mq(mc);
  std::string err;
  const bool ran = mq.run(dit_dir, out_dir, opt, &err, stop);
  if (!tmp_cal.empty()) { fs::remove_all(tmp_cal, ec); }
  if (!ran) {
    if (stop()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): quantize stopped; output '{}' is "
          "incomplete", this->id(), out_dir));
    } else {
      session()->warn(fmt("ModelQuantizeStage('{}'): {}", this->id(), err));
    }
    return false;
  }
  return true;
}

bool
ModelQuantizeStage::quantize_text_encoder_(
    const std::string& enc_dir, const std::string& out_dir,
    const std::string& root, const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  metal_compute::MetalCompute* mc = session()->metal_compute();
  if (mc == nullptr) { return false; }

  // The text encoder is a dense Qwen3-VL backbone under the language_model.
  // prefix (q/k/v/o_proj, gate/up/down_proj) -- the default linear set matches
  // it prefix-agnostically. CRITICAL: keep embed_tokens bf16 (quant_embeddings
  // off) -- the text-to-image stage host-gathers it as a plain table.
  const genai::QuantArchInfo meta =
      genai::detect_quant_arch(session(), enc_dir);
  genai::QuantizeOptions opt;
  opt.bits  = _bits;
  opt.group = _group_size;
  opt.quant_embeddings = false;
  opt.norm_offset      = false;   // qwen3_vl uses standard RMSNorm
  session()->info(fmt(
      "ModelQuantizeStage('{}'): text encoder '{}' (arch '{}', {} layers, "
      "prefix '{}')", this->id(), enc_dir, meta.arch, meta.n_layers,
      meta.layer_prefix));
  if (_mixed) {
    if (meta.n_layers <= 0 || meta.layer_prefix.empty()) {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): text-encoder mixed precision needs "
          "n_layers + layer_prefix; auto-detect from '{}' failed", this->id(),
          enc_dir));
      return false;
    }
    opt.mixed        = true;
    opt.high_bits    = _high_bits;
    opt.mixed_frac   = _mixed_frac;
    opt.layer_prefix = meta.layer_prefix;
    opt.n_layers     = meta.n_layers;
  }
  fs::path auto_calib;
  if (_awq) {
    if (meta.n_layers <= 0 || meta.layer_prefix.empty()) {
      session()->warn(fmt(
          "ModelQuantizeStage('{}'): text-encoder AWQ needs n_layers + "
          "layer_prefix; auto-detect from '{}' failed", this->id(), enc_dir));
      return false;
    }
    std::string calib = _calib_dir;
    if (calib.empty()) {
      // On-device auto-calibration. The text encoder is a dense Qwen3
      // backbone, so a plain text corpus exercises exactly the encoded path;
      // its tokenizer lives in the pipeline root (the sub-dir has none).
      // The fast tokenizer.json lives in the encoder dir, the pipeline
      // tokenizer/ (Krea-2 / FLUX.2), or processor/ (Qwen-Image-Edit, whose
      // tokenizer/ holds only the slow vocab.json + merges).
      std::string tok = (fs::path(enc_dir) / "tokenizer.json").string();
      if (!fs::exists(tok, ec)) {
        tok = (fs::path(root) / "tokenizer" / "tokenizer.json").string();
      }
      if (!fs::exists(tok, ec)) {
        tok = (fs::path(root) / "processor" / "tokenizer.json").string();
      }
      calib = auto_calibrate_backbone_(enc_dir, meta, meta.n_layers, tok, stop);
      if (calib.empty()) { return false; }   // logged inside
      auto_calib = calib;
    }
    opt.smoothquant  = true;
    opt.awq_clip     = _awq_clip;
    opt.calib_dir    = calib;
    opt.layer_prefix = meta.layer_prefix;
    opt.n_layers     = meta.n_layers;
  }
  genai::ModelQuantizer mq(mc);
  std::string err;
  const bool ran = mq.run(enc_dir, out_dir, opt, &err, stop);
  if (!auto_calib.empty()) { fs::remove_all(auto_calib, ec); }
  if (!ran) {
    if (stop()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): text-encoder quantize stopped; '{}' is "
          "incomplete", this->id(), out_dir));
    } else {
      session()->warn(fmt("ModelQuantizeStage('{}'): text-encoder: {}",
                          this->id(), err));
    }
    return false;
  }
  return true;
}

std::string
ModelQuantizeStage::auto_calibrate_backbone_(
    const std::string& src_dir, const genai::QuantArchInfo& meta,
    int n_layers, const std::string& tok_path,
    const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  metal_compute::MetalCompute* mc = session()->metal_compute();
  if (mc == nullptr) { return {}; }
  if (!meta.calib_ok) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): on-device auto-calibration is not available "
        "for arch '{}' (only dense full-attention Qwen3-family backbones run "
        "through MetalQwenModel); supply calib_dir", this->id(), meta.arch));
    return {};
  }
  genai::MetalQwenModel::Config bcfg = meta.backbone;
  bcfg.n_layers = n_layers;   // honor an explicit n_layers override
  auto tok = genai::Tokenizer::from_huggingface_json(tok_path, session());
  if (!tok) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): auto-calibration needs a tokenizer at '{}'",
        this->id(), tok_path));
    return {};
  }
  // Built-in corpus: ~128 sequences of ~512 tokens from the curated mlx-lm
  // calibration set, chat-template-wrapped so control tokens are exercised.
  std::vector<std::vector<std::int32_t>> corpus =
      genai::build_builtin_calibration_corpus(
          *tok, kCalibSeqs, kCalibSeqLen, /*apply_chat_template=*/true);
  if (corpus.empty()) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): built-in calibration corpus produced no "
        "sequences", this->id()));
    return {};
  }
  const fs::path tmp_q8 =
      vpipe::temp_root() / ("vpipe-mqcal8-" + this->id());
  const fs::path tmp_cal =
      vpipe::temp_root() / ("vpipe-mqcal-" + this->id());
  fs::remove_all(tmp_q8, ec);
  fs::remove_all(tmp_cal, ec);
  auto on_fail = [&](const std::string& what, const std::string& detail) {
    if (stop()) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): quantize stopped during calibration",
          this->id()));
    } else if (!detail.empty()) {
      session()->warn(fmt("ModelQuantizeStage('{}'): {}{}", this->id(), what,
                          detail));
    }
  };
  if (bcfg.is_moe()) {
    // MoE: memory-safe layer-by-layer streaming forward over the bf16 SOURCE
    // (never resides the whole expert stack; no 8-bit base needed).
    session()->info(fmt(
        "ModelQuantizeStage('{}'): auto-calibrating MoE (streaming per-layer "
        "forward over {} corpus seqs)", this->id(), corpus.size()));
    std::string ce;
    if (!genai::collect_backbone_calibration_streaming(
            mc, src_dir, bcfg, corpus, tmp_cal.string(), &ce,
            (std::uint64_t)8 << 30, stop)) {
      on_fail("", ce);
      fs::remove_all(tmp_cal, ec);
      return {};
    }
    return tmp_cal.string();
  }
  session()->info(fmt(
      "ModelQuantizeStage('{}'): auto-calibrating (8-bit base + on-device taps "
      "over {} corpus seqs)", this->id(), corpus.size()));
  {
    genai::QuantizeOptions q8; q8.bits = 8; q8.group = 64;
    genai::ModelQuantizer mq8(mc);
    std::string e8;
    if (!mq8.run(src_dir, tmp_q8.string(), q8, &e8, stop)) {
      on_fail("calib 8-bit base: ", e8);
      fs::remove_all(tmp_q8, ec);
      return {};
    }
  }
  std::string ce;
  if (!genai::collect_backbone_calibration(
          mc, tmp_q8.string(), bcfg, corpus, tmp_cal.string(), &ce, stop)) {
    on_fail("", ce);
    fs::remove_all(tmp_q8, ec);
    fs::remove_all(tmp_cal, ec);
    return {};
  }
  fs::remove_all(tmp_q8, ec);   // the 8-bit base is consumed
  return tmp_cal.string();
}

bool
ModelQuantizeStage::quantize_t2i_pipeline_(
    const std::string& root, const std::string& family,
    const std::string& out_dir, const std::function<bool()>& stop)
{
  namespace fs = std::filesystem;
  std::error_code ec;

  const std::string tgt_sub = t2i_target_subdir_(_target);
  if (!_target.empty() && tgt_sub.empty()) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): target '{}' is not a {} component "
        "(want dit | text_encoder | vae)", this->id(), _target, family));
    return false;
  }
  const std::string tgt = tgt_sub.empty() ? std::string("transformer") : tgt_sub;
  const fs::path tgt_src = fs::path(root) / tgt;
  if (!fs::is_directory(tgt_src, ec)) {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): {} root '{}' has no '{}' component to "
        "quantize", this->id(), family, root, tgt));
    return false;
  }
  if (tgt == "vae") {
    session()->warn(fmt(
        "ModelQuantizeStage('{}'): VAE quantization is not supported yet (the "
        "diffusion VAE is a conv net, not a Linear stack); no pass performed",
        this->id()));
    return false;
  }
  if (_skip_existing &&
      fs::exists(fs::path(out_dir) / tgt / "config.json", ec)) {
    session()->info(fmt(
        "ModelQuantizeStage('{}'): output '{}' already has a quantized {}; "
        "skipping", this->id(), out_dir, tgt));
    return true;
  }

  session()->info(fmt(
      "ModelQuantizeStage('{}'): {} pipeline '{}' -> '{}' (quantizing {}, "
      "copying the other components)", this->id(), family, root, out_dir, tgt));

  // 1. Assemble the self-contained output: hard-link/copy every component
  //    except the target (quantized next), so the result carries all of
  //    tokenizer/scheduler/model_index.json + the other (un- or already-
  //    quantized) sub-models -- usable directly as a text-to-image hf_dir.
  fs::create_directories(out_dir, ec);
  for (const auto& e : fs::directory_iterator(root, ec)) {
    if (e.path().filename().string() == tgt) { continue; }
    if (!link_or_copy_tree_(e.path(), fs::path(out_dir) / e.path().filename(),
                            ec, stop)) {
      session()->info(fmt(
          "ModelQuantizeStage('{}'): stopped while assembling '{}'",
          this->id(), out_dir));
      return false;
    }
  }
  session()->log_debug(fmt(
      "ModelQuantizeStage('{}'): copied {} components (all but {}) into '{}'",
      this->id(), family, tgt, out_dir));

  // 2. Quantize the target component into out_dir/<tgt>.
  const std::string tgt_out = (fs::path(out_dir) / tgt).string();
  bool ok = false;
  if (tgt == "transformer") {
    ok = quantize_dit_component_(tgt_src.string(), tgt_out, family, root, stop);
  } else if (tgt == "text_encoder") {
    ok = quantize_text_encoder_(tgt_src.string(), tgt_out, root, stop);
  }
  if (!ok) { return false; }

  session()->log_normal(fmt(
      "ModelQuantizeStage('{}'): quantized {} {} '{}' -> self-contained "
      "'{}' ({}-bit g{}{}{})", this->id(), family, tgt, root, out_dir, _bits,
      _group_size, _mixed ? " mixed" : "", _awq ? " awq" : ""));

  // 3. Register as a full pipeline (usable directly as an hf_dir).
  const bool explicit_path =
      _output_name[0] == '/' ||
      _output_name.rfind("./", 0) == 0 || _output_name.rfind("../", 0) == 0;
  if (!explicit_path) {
    register_output_(_output_name, out_dir, family, _bits);
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
  // Optional trigger (see model-fetch): gate the work on one beat when the
  // iport is wired so this stage can cascade in a recipe.
  if (ctx.iport_connected(0)) {
    auto trig = co_await ctx.read(0);
    if (!trig) {
      ctx.signal_done();
      co_return;
    }
  }
  // Source availability is checked HERE, AFTER the trigger -- in a recipe
  // the upstream model-fetch may not have downloaded it at config time.
  // Missing source => log + halt WITHOUT emitting a summary, so the
  // cascade stops here instead of quantizing a nonexistent model.
  if (!model_dir_available(session(), _models_db, _src_model)) {
    session()->error(fmt(
        "ModelQuantizeStage('{}'): source model '{}' is not available "
        "(not downloaded yet?); skipping quantization",
        this->id(), _src_model));
    ctx.signal_done();
    co_return;
  }
  session()->info(fmt(
      "ModelQuantizeStage('{}'): quantizing '{}' -> '{}' ({}-bit g{}{}{})",
      this->id(), _src_model, _output_name, _bits, _group_size,
      _awq ? (_awq_clip ? " awq+clip" : " awq") : "",
      _mixed ? " mixed" : ""));
  const bool ok = quantize_once([&ctx] { return ctx.stop_requested(); });
  // Emit the summary only on success, so a failed quantization halts the
  // cascade (mirrors model-benchmark / -eval, which emit only on success).
  if (ok && ctx.has_consumers(0)) {
    FlexData summary = FlexData::make_object();
    auto so = summary.as_object();
    so.insert_or_assign("stage", FlexData::make_string("model-quantize"));
    so.insert_or_assign("source", FlexData::make_string(_src_model));
    so.insert_or_assign("output", FlexData::make_string(_output_name));
    so.insert_or_assign("models_db", FlexData::make_string(_models_db));
    so.insert_or_assign("bits", FlexData::make_int(_bits));
    so.insert_or_assign("group_size", FlexData::make_int(_group_size));
    so.insert_or_assign("quantized", FlexData::make_bool(ok));
    so.insert_or_assign("text", FlexData::make_string(
        fmt("[model-quantize] {} -> {} ({}-bit, group {}) [{}]",
            _src_model, _output_name, _bits, _group_size,
            ok ? "ok" : "failed")()));
    co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(summary)));
  }
  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(ModelQuantizeStage)
VPIPE_REGISTER_SPEC(ModelQuantizeStage, kSpec)

}  // namespace vpipe
