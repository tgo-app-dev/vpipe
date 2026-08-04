#include "stages/model-detect.h"

#include "common/flex-data.h"
#include "stages/model-catalog.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>

namespace vpipe {

namespace {

namespace fs = std::filesystem;

std::string
lower_(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return s;
}

bool
has_(const std::string& hay_lc, const char* needle)
{
  return hay_lc.find(needle) != std::string::npos;
}

// Read a JSON file into a FlexData object. Returns a Null FlexData when
// the file is absent / unreadable / not an object -- callers test
// is_object() and move on, so a malformed config never throws out of
// detection.
FlexData
read_json_(const fs::path& p)
{
  std::error_code ec;
  if (!fs::exists(p, ec) || ec) {
    return {};
  }
  std::ifstream in(p);
  if (!in) {
    return {};
  }
  try {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      return fd;
    }
  } catch (...) {
  }
  return {};
}

std::string
str_field_(const FlexData& obj, const char* key)
{
  if (!obj.is_object()) {
    return {};
  }
  auto o = obj.as_object();
  return o.contains(key) ? std::string(o.at(key).as_string("")) : std::string();
}

bool
has_field_(const FlexData& obj, const char* key)
{
  return obj.is_object() && obj.as_object().contains(key);
}

// Split a name on the separators HF repos use, so "gemma-4-e4b-it-4bit"
// becomes {gemma,4,e4b,it,4bit}.
std::vector<std::string>
tokens_(const std::string& name)
{
  std::vector<std::string> out;
  std::string cur;
  for (char c : name) {
    if (c == '-' || c == '_' || c == '.' || c == ' ') {
      if (!cur.empty()) { out.push_back(cur); cur.clear(); }
      // '.' also separates, but "3.5" / "1.7B" must survive: re-join
      // below when both sides are numeric.
      if (c == '.') { out.push_back("."); }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) { out.push_back(cur); }
  // Re-join "1" "." "7B" -> "1.7B".
  std::vector<std::string> joined;
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == "." && !joined.empty() && i + 1 < out.size()) {
      joined.back() += "." + out[i + 1];
      ++i;
    } else if (out[i] != ".") {
      joined.push_back(out[i]);
    }
  }
  return joined;
}

// A parameter-count token: 4B, 1.7B, 27B, E4B (Gemma's effective sizes),
// optionally followed by the MoE active-parameter token (35B + A3B ->
// "35B-A3B", the spelling the catalogue uses).
bool
param_token_(const std::string& t)
{
  if (t.size() < 2) { return false; }
  size_t i = 0;
  if (t[0] == 'e' || t[0] == 'E') { i = 1; }
  bool digit = false;
  for (; i + 1 < t.size(); ++i) {
    if (std::isdigit((unsigned char)t[i])) { digit = true; }
    else if (t[i] != '.') { return false; }
  }
  return digit && (t.back() == 'b' || t.back() == 'B');
}

bool
active_param_token_(const std::string& t)
{
  if (t.size() < 3 || (t[0] != 'a' && t[0] != 'A')) { return false; }
  return param_token_(t.substr(1));
}

std::string
upper_(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::toupper(c); });
  return s;
}

// "Qwen3.5-4B-MLX-4bit" -> "4B"; "gemma-4-e4b-it" -> "E4B";
// "Qwen3.6-35B-A3B-OptiQ" -> "35B-A3B". Empty when the name says nothing.
std::string
param_class_from_name_(const std::string& name)
{
  const auto tk = tokens_(name);
  for (size_t i = 0; i < tk.size(); ++i) {
    if (!param_token_(tk[i])) { continue; }
    std::string pc = upper_(tk[i]);
    if (i + 1 < tk.size() && active_param_token_(tk[i + 1])) {
      pc += "-" + upper_(tk[i + 1]);
    }
    return pc;
  }
  return {};
}

// family / version display labels for a runtime tag, matching the
// catalogue's spelling so a registered model sorts beside its
// catalogued siblings in the web-ui browser.
void
family_version_(const std::string& mt, std::string& family,
                std::string& version)
{
  struct Row { const char* mt; const char* family; const char* version; };
  static const Row kRows[] = {
      {"qwen3.5",                "Qwen",        "3.5"},
      {"qwen3.6",                "Qwen",        "3.6"},
      {"qwen3-asr",              "Qwen",        "3-ASR"},
      {"qwen3.5-vision-encoder", "Qwen",        "3.5"},
      {"gemma4",                 "Gemma",       "4"},
      {"gemma4_unified",         "Gemma",       "4"},
      {"gemma4-vision-encoder",  "Gemma",       "4"},
      {"llama3",                 "Llama",       "3"},
      {"krea2",                  "Krea",        "2"},
      {"krea2-lora",             "Krea",        "2"},
      {"flux2",                  "FLUX",        "2"},
      {"qwen-image-edit",        "Qwen-Image",  "Edit-2511"},
      {"mage-flow",              "Mage-Flow",   "Gen"},
      {"mage-flow-edit",         "Mage-Flow",   "Edit"},
      {"boogu-image",            "Boogu-Image", "0.1"},
      {"boogu-image-edit",       "Boogu-Image", "0.1-Edit"},
      {"moss-tts",               "MOSS",        "TTS"},
      {"moss-tts-local",         "MOSS",        "TTS-Local"},
      {"moss-tts-realtime",      "MOSS",        "TTS-Realtime"},
      {"moss-codec",             "MOSS",        "Audio-Tokenizer"},
      {"moss-codec-v2",          "MOSS",        "Audio-Tokenizer"},
      {"yolo",                   "YOLOX",       "L"},
      {"silero-vad",             "Silero",      "VAD"},
      {"audio-tagging",          "BEATs",       "iter3+"},
  };
  for (const auto& r : kRows) {
    if (mt == r.mt) { family = r.family; version = r.version; return; }
  }
}

// ---- config.json (HF / MLX layout) -> runtime tag --------------------
//
// The config alone is not always decisive, so the directory NAME breaks
// the two ties it cannot:
//   * Qwen3.6 checkpoints still report model_type "qwen3_5" (same
//     architecture class), and
//   * every Gemma-4 reports "gemma4", but the E-sized (E2B/E4B) ones run
//     the per-layer-embedding path ("gemma4") while the 12B/26B/31B run
//     the unified one ("gemma4_unified") -- which is exactly the
//     distinction the catalogue draws by param_class.
std::string
lm_tag_(const std::string& cfg_type, const std::string& name_lc,
        const std::string& param_class)
{
  const std::string mt = lower_(cfg_type);
  auto starts = [&](const char* p) { return mt.rfind(p, 0) == 0; };

  if (starts("qwen3_asr") || starts("qwen3asr")) { return "qwen3-asr"; }
  if (starts("qwen3_5") || starts("qwen3_6")) {
    if (starts("qwen3_6") || has_(name_lc, "qwen3.6")
        || has_(name_lc, "qwen3_6")) {
      return "qwen3.6";
    }
    return "qwen3.5";
  }
  if (starts("gemma")) {
    // E2B / E4B -> the effective (gemma3n-style) path; every other size
    // is the unified one.
    return (!param_class.empty() && param_class[0] == 'E') ? "gemma4"
                                                           : "gemma4_unified";
  }
  if (starts("moss_tts_local"))    { return "moss-tts-local"; }
  if (starts("moss_tts_realtime")) { return "moss-tts-realtime"; }
  if (starts("moss_tts"))          { return "moss-tts"; }
  if (starts("llama"))             { return "llama3"; }
  return {};
}

// ---- transformer/config.json (diffusers layout) -> runtime tag -------
std::string
dit_tag_(const std::string& cls, const std::string& name_lc)
{
  const bool edit = has_(name_lc, "edit");
  // FLUX.2-klein-9b-kv reports this SAME class, and its config.json and tensor
  // names are identical to plain klein-9B's -- the two differ only in the
  // weights and in the attention recipe those weights were distilled for. So
  // "flux2" is the right answer for both, and the recipe is selected by the
  // consuming stage's `klein_kv` key rather than guessed here. Deliberately
  // NOT inferred from the name the way `edit` is below: a wrong `edit` guess
  // picks a sibling checkpoint, while a wrong recipe guess silently produces
  // wrong IMAGES from a model that loads and runs perfectly.
  if (cls == "Flux2Transformer2DModel")      { return "flux2"; }
  if (cls == "Krea2Transformer2DModel")      { return "krea2"; }
  if (cls == "QwenImageTransformer2DModel")  { return "qwen-image-edit"; }
  if (cls == "MageFlow") {
    return edit ? "mage-flow-edit" : "mage-flow";
  }
  if (cls == "BooguImageTransformer2DModel") {
    // The t2i and edit repos ship the SAME transformer config (only the
    // weights differ), so the name is the only signal.
    return edit ? "boogu-image-edit" : "boogu-image";
  }
  return {};
}

// A BARE DiT component directory (diffusers weights + config, no
// pipeline around it -- what model-quantize's DiT-only output is) gets
// the "<family>-dit" tag that text-to-image's `dit_dir` picker filters
// on, NOT the pipeline tag: it is not loadable as a pipeline, and
// claiming otherwise would offer it to hf_dir where it would fail.
std::string
dit_component_tag_(const std::string& cls)
{
  if (cls == "Flux2Transformer2DModel")      { return "flux2-dit"; }
  if (cls == "Krea2Transformer2DModel")      { return "krea2-dit"; }
  if (cls == "QwenImageTransformer2DModel")  { return "qwen-image-edit-dit"; }
  if (cls == "MageFlow")                     { return "mage-flow-dit"; }
  if (cls == "BooguImageTransformer2DModel") { return "boogu-image-dit"; }
  return {};
}

// Quantization / precision label for the record's `variant`.
std::string
variant_(const FlexData& cfg, const std::string& name_lc,
         const std::string& gguf_file)
{
  if (!gguf_file.empty()) {
    // "...-Q4_K_M.gguf" -> "GGUF Q4_K_M".
    const std::string base = fs::path(gguf_file).stem().string();
    const size_t dash = base.rfind('-');
    const std::string q = (dash == std::string::npos) ? std::string()
                                                      : base.substr(dash + 1);
    return q.empty() ? "GGUF" : ("GGUF " + upper_(q));
  }
  std::string bits;
  if (cfg.is_object()) {
    auto o = cfg.as_object();
    if (o.contains("quantization")) {
      FlexData q = o.at("quantization");
      if (q.is_object() && q.as_object().contains("bits")) {
        bits = std::to_string(q.as_object().at("bits").as_int(0));
      }
    }
  }
  const char* style = has_(name_lc, "optiq") ? "OptiQ "
                    : has_(name_lc, "mlx")   ? "MLX "
                                             : "";
  if (!bits.empty() && bits != "0") {
    return std::string(style) + bits + "-bit";
  }
  const std::string dtype = str_field_(cfg, "torch_dtype");
  if (dtype == "bfloat16") { return "bf16"; }
  if (dtype == "float16")  { return "f16"; }
  return std::string(style);
}

// Walk a directory once for the file count + byte total that the
// registry record reports (model-fetch records the same two facts from
// its download).
void
tree_size_(const fs::path& root, std::uint64_t& files, std::uint64_t& bytes)
{
  std::error_code ec;
  if (fs::is_regular_file(root, ec)) {
    files = 1;
    bytes = fs::file_size(root, ec);
    if (ec) { bytes = 0; }
    return;
  }
  fs::recursive_directory_iterator it(
      root, fs::directory_options::skip_permission_denied, ec);
  if (ec) { return; }
  for (const auto& e : it) {
    std::error_code fec;
    if (!e.is_regular_file(fec) || fec) { continue; }
    const auto sz = fs::file_size(e.path(), fec);
    if (!fec) { bytes += sz; }
    ++files;
  }
}

// True when the directory holds at least one file with `ext`.
bool
has_ext_(const fs::path& dir, const char* ext, std::string* found = nullptr)
{
  std::error_code ec;
  fs::directory_iterator it(dir, ec);
  if (ec) { return false; }
  for (const auto& e : it) {
    std::error_code fec;
    if (!e.is_regular_file(fec) || fec) { continue; }
    if (lower_(e.path().extension().string()) == ext) {
      if (found) { *found = e.path().filename().string(); }
      return true;
    }
  }
  return false;
}

// Copy a catalogue entry's curated metadata into the detection result.
void
from_catalog_(const ModelCatalogEntry& e, DetectedModel& d)
{
  d.model_type         = e.model_type;
  d.family             = e.family;
  d.version            = e.version;
  d.param_class        = e.param_class;
  d.variant            = e.variant;
  d.category           = catalog_category(e);
  d.parent_model_type  = e.parent_model_type;
  d.parent_param_class = e.parent_param_class;
  d.inputs             = e.inputs;
  d.outputs            = e.outputs;
  if (d.inputs.empty() && d.outputs.empty()) {
    catalog_default_io(e.model_type, d.inputs, d.outputs);
  }
  d.detected_by = "catalog";
}

// Whether the ABSENCE of a vision_config / audio_config in this family's
// config.json is evidence that the checkpoint drops that modality.
//
// It is evidence only for the families that signal their towers exactly
// that way at the config ROOT (Qwen3.5/3.6: vision_config; Gemma-4: both).
// Qwen3-ASR, for one, wraps everything under thinker_config and carries no
// audio_config at all -- reading its absence as "no audio" would turn a
// transcription model into a text-to-text one. Anything not listed here
// keeps its family defaults.
bool
io_trim_is_meaningful_(const std::string& mt)
{
  return mt == "qwen3.5" || mt == "qwen3.6" || mt == "gemma4"
         || mt == "gemma4_unified";
}

// Drop modalities the checkpoint plainly does not carry. The per-type
// defaults describe a FAMILY at its fullest (a Qwen3.5 repo with a vision
// tower takes images); a text-only dump of the same family does not, and
// claiming otherwise would offer it to a visual-qa field it cannot serve.
void
trim_io_to_config_(const FlexData& cfg, std::vector<std::string>& inputs)
{
  if (!cfg.is_object() || inputs.empty()) { return; }
  const bool vision = has_field_(cfg, "vision_config");
  const bool audio  = has_field_(cfg, "audio_config");
  std::vector<std::string> kept;
  for (const auto& m : inputs) {
    if ((m == "image" || m == "video") && !vision) { continue; }
    if (m == "audio" && !audio) { continue; }
    kept.push_back(m);
  }
  // A model that reads nothing would be nonsense: keep text as the floor.
  if (kept.empty()) { kept.push_back("text"); }
  inputs = kept;
}

}  // namespace

void
record_detected_fields(FlexData::ObjectView& rec, const DetectedModel& d)
{
  auto put = [&](const char* k, const std::string& v) {
    if (!v.empty()) { rec.insert_or_assign(k, FlexData::make_string(v)); }
  };
  put("model_type", d.model_type);
  put("family", d.family);
  put("version", d.version);
  put("param_class", d.param_class);
  put("variant", d.variant);
  put("category", d.category);
  put("parent_model_type", d.parent_model_type);
  put("parent_param_class", d.parent_param_class);
  put("detected_by", d.detected_by);
  auto put_list = [&](const char* k, const std::vector<std::string>& v) {
    if (v.empty()) { return; }
    FlexData arr = FlexData::make_array();
    {
      auto a = arr.as_array();
      for (const auto& s : v) { a.push_back(FlexData::make_string(s)); }
    }
    rec.insert_or_assign(k, std::move(arr));
  };
  put_list("inputs", d.inputs);
  put_list("outputs", d.outputs);
}

std::string
coreml_artifact(const std::string& path)
{
  if (path.empty()) { return {}; }
  auto is_bundle = [](const fs::path& p) {
    const std::string e = lower_(p.extension().string());
    return e == ".mlpackage" || e == ".mlmodelc";
  };
  const fs::path root(path);
  if (is_bundle(root)) { return path; }

  std::error_code ec;
  if (!fs::is_directory(root, ec) || ec) { return {}; }
  // .mlpackage is the shipped form, so it wins outright; a .mlmodelc is
  // remembered and used only if no package turns up, which keeps the
  // answer independent of directory iteration order.
  fs::path compiled;
  for (const auto& e : fs::directory_iterator(root, ec)) {
    if (ec) { break; }
    const fs::path& p = e.path();
    const std::string x = lower_(p.extension().string());
    if (x == ".mlpackage") { return p.string(); }
    if (x == ".mlmodelc" && compiled.empty()) { compiled = p; }
  }
  return compiled.empty() ? std::string() : compiled.string();
}

std::string
hf_path_from_local(const std::string& dir)
{
  fs::path p(dir);
  if (p.filename().empty()) { p = p.parent_path(); }   // trailing slash
  const std::string repo = p.filename().string();
  const std::string owner = p.parent_path().filename().string();
  if (repo.empty() || owner.empty()) { return {}; }
  return owner + "/" + repo;
}

DetectedModel
detect_model_dir(const std::string& dir, const std::string& hf_path_hint)
{
  DetectedModel d;
  std::error_code ec;
  const fs::path root(dir);
  if (!fs::exists(root, ec) || ec) {
    return d;
  }
  d.is_dir = fs::is_directory(root, ec) && !ec;
  tree_size_(root, d.file_count, d.total_bytes);

  const std::string base    = root.filename().string();
  const std::string base_lc = lower_(base);
  const std::string stem    = fs::path(base).stem().string();
  const std::string hf_path =
      hf_path_hint.empty() ? hf_path_from_local(dir) : hf_path_hint;

  // ---- 1. the catalogue, by name then by repo path -------------------
  // A catalogue hit is curated metadata -- always better than probing.
  // `name` first: the vpipe-supplement CoreML models share one hf_path
  // and are told apart only by name (which is their directory /
  // .mlpackage stem on disk).
  if (const ModelCatalogEntry* e = catalog_by_name(stem)) {
    from_catalog_(*e, d);
    return d;
  }
  if (!hf_path.empty()) {
    if (const ModelCatalogEntry* e = catalog_by_path(hf_path)) {
      from_catalog_(*e, d);
      return d;
    }
  }

  // ---- 2. a diffusers pipeline (transformer/ + vae/) ------------------
  const FlexData dit_cfg = read_json_(root / "transformer" / "config.json");
  if (dit_cfg.is_object()) {
    d.model_type = dit_tag_(str_field_(dit_cfg, "_class_name"), base_lc);
    if (!d.model_type.empty()) {
      d.detected_by = "diffusers";
    }
  }

  // ---- 3. a plain HF / MLX checkpoint (config.json at the root) -------
  // Also catches a bare DiT component: a diffusers config carries
  // `_class_name` where an HF one carries `model_type`.
  FlexData cfg;
  if (d.model_type.empty()) {
    cfg = read_json_(root / "config.json");
    if (cfg.is_object()) {
      d.param_class = param_class_from_name_(base);
      const std::string cls = str_field_(cfg, "_class_name");
      if (!cls.empty()) {
        d.model_type = dit_component_tag_(cls);
        if (!d.model_type.empty()) {
          // A component, not a runnable model: no I/O of its own.
          d.category    = "component";
          d.detected_by = "diffusers-component";
        }
      } else {
        d.model_type =
            lm_tag_(str_field_(cfg, "model_type"), base_lc, d.param_class);
        if (!d.model_type.empty()) {
          d.detected_by = "config";
        }
      }
    }
  }

  // ---- 4. a GGUF checkpoint -------------------------------------------
  std::string gguf;
  if (d.model_type.empty() && d.is_dir && has_ext_(root, ".gguf", &gguf)) {
    // The repo NAME carries the family + size ("Qwen3.5-4B-GGUF"); the
    // .gguf file only carries the quant label.
    d.param_class = param_class_from_name_(base);
    if (has_(base_lc, "qwen3.6") || has_(base_lc, "qwen3_6")) {
      d.model_type = "qwen3.6";
    } else if (has_(base_lc, "qwen3.5") || has_(base_lc, "qwen3_5")) {
      d.model_type = "qwen3.5";
    } else if (has_(base_lc, "gemma")) {
      d.model_type = (!d.param_class.empty() && d.param_class[0] == 'E')
                         ? "gemma4" : "gemma4_unified";
    }
    if (!d.model_type.empty()) { d.detected_by = "gguf"; }
  }

  // ---- 5. a CoreML package (a tower / detector / VAD supplement) ------
  // Only the SHAPE is knowable here -- which model it belongs to is not,
  // so this sets the category and leaves model_type to the catalogue
  // (step 1) or the user's explicit override.
  const std::string ext_lc = lower_(root.extension().string());
  if (d.model_type.empty()
      && (ext_lc == ".mlpackage" || ext_lc == ".mlmodelc")) {
    d.category    = "supplement";
    d.detected_by = "coreml";
  }
  // A bare LoRA (weights, no config) is likewise a supplement of a model
  // we cannot name from the file alone.
  if (d.model_type.empty() && d.category.empty() && has_(base_lc, "lora")) {
    d.category = "supplement";
  }

  if (d.param_class.empty()) {
    d.param_class = param_class_from_name_(base);
  }
  d.variant = variant_(cfg, base_lc, gguf);
  if (d.category.empty()) {
    d.category = "model";
  }
  family_version_(d.model_type, d.family, d.version);

  // ---- 6. I/O modalities ----------------------------------------------
  // Derived from the runtime tag (the same table the catalogue uses, so a
  // registered model reads identically to a fetched one), then trimmed to
  // what this particular checkpoint actually carries.
  if (!d.model_type.empty()) {
    catalog_default_io(d.model_type, d.inputs, d.outputs);
    if (io_trim_is_meaningful_(d.model_type)) {
      trim_io_to_config_(cfg, d.inputs);
    }
  }
  return d;
}

}
