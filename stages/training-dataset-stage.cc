#include "stages/training-dataset-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "generative-models/train/sample-cache.h"
#include "stages/image-decode.h"
#include "stages/model-registry.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <utility>

namespace vpipe {

namespace fs = std::filesystem;

namespace {

constexpr SpecExtra kCaptionChoices[] = {
  {spec_key::kChoices, "sidecar,jsonl,none"},
};
constexpr SpecExtra kMissingChoices[] = {
  {spec_key::kChoices, "trigger_only,skip,error"},
};
constexpr SpecExtra kFlipChoices[] = {
  {spec_key::kChoices, "off,random"},
};
constexpr SpecExtra kAlphaChoices[] = {
  {spec_key::kChoices, "drop,keep"},
};

const ConfigKey kAttrs[] = {
  {.key = "dir",
   .type = ConfigType::String,
   .required = true,
   .doc = "the folder of pictures, scanned recursively (png, jpg, jpeg, "
          "webp, bmp, tif). A subfolder named N_name repeats its pictures "
          "N times per epoch, as kohya-style datasets are laid out.",
   .is_path = true,
   .path_kind = "dir"},
  {.key = "captions",
   .type = ConfigType::String,
   .doc = "where captions come from: sidecar (a text file beside each "
          "picture; several lines are several captions), jsonl (a "
          "metadata.jsonl in the folder with file_name and text), or none",
   .def_str = "sidecar",
   .extra = kCaptionChoices},
  {.key = "caption_ext",
   .type = ConfigType::String,
   .doc = "the sidecar caption's extension (default .txt)",
   .def_str = ".txt"},
  {.key = "trigger_word",
   .type = ConfigType::String,
   .doc = "a word the adapter learns to answer to. Joined to every "
          "caption by caption_template unless the caption already names "
          "it; a [trigger] placeholder in a caption is replaced in place. "
          "Recorded in the saved adapter's metadata."},
  {.key = "caption_template",
   .type = ConfigType::String,
   .doc = "how the trigger joins a caption: {trigger} and {caption} are "
          "replaced (default \"{trigger}, {caption}\")",
   .def_str = "{trigger}, {caption}"},
  {.key = "missing_caption",
   .type = ConfigType::String,
   .doc = "a picture with no caption: trigger_only (the trigger alone -- "
          "the usual way to train a subject from a handful of pictures), "
          "skip, or error",
   .def_str = "trigger_only",
   .extra = kMissingChoices},
  {.key = "shuffle_tags",
   .type = ConfigType::Bool,
   .doc = "comma-separated tag captions: shuffle the tags after the first "
          "keep_tags, so the adapter does not learn their order. Every "
          "shuffle is encoded up front (caption_variants of them)."},
  {.key = "keep_tags",
   .type = ConfigType::Int,
   .doc = "tags held in place at the front when shuffling (default 1, the "
          "trigger)",
   .def_int = 1},
  {.key = "caption_variants",
   .type = ConfigType::Int,
   .doc = "encodings kept per caption when shuffling (default 4; the "
          "embeddings are cached, so a caption that changes per epoch has "
          "to be encoded in advance)"},
  {.key = "resolution",
   .type = ConfigType::Any,
   .doc = "the side of the square whose area each bucket keeps: one "
          "number (default 1024) or a list, e.g. [512, 768, 1024] -- every "
          "picture is then encoded at each, which small datasets "
          "generalise better from"},
  {.key = "bucket_step",
   .type = ConfigType::Int,
   .doc = "bucket sides are multiples of this (default 32, Qwen-Image-2.1's "
          "size grid)",
   .def_int = 32},
  {.key = "max_aspect",
   .type = ConfigType::Real,
   .doc = "the widest aspect a bucket keeps (default 2.0); wider pictures "
          "are cropped to it",
   .def_real = 2.0},
  {.key = "flip",
   .type = ConfigType::String,
   .doc = "off, or random: every picture is also encoded mirrored. Off by "
          "default -- text and asymmetric subjects do not survive it.",
   .def_str = "off",
   .extra = kFlipChoices},
  {.key = "alpha",
   .type = ConfigType::String,
   .doc = "drop (RGB pictures) or keep (RGBA, for a VAE that takes four "
          "channels and pictures that carry transparency)",
   .def_str = "drop",
   .extra = kAlphaChoices},
  {.key = "preview_prompts",
   .type = ConfigType::Any,
   .doc = "prompts the trainer samples at every checkpoint, so they can "
          "be compared: a string or a list; {trigger} is expanded. Default "
          "the trigger word alone."},
  {.key = "seed",
   .type = ConfigType::Int,
   .doc = "seeds the tag shuffles and the held-out choice (default 0)"},
  {.key = "cache_dir",
   .type = ConfigType::String,
   .doc = "where the encoded captions and pictures are kept between runs, "
          "so a re-run encodes only what changed: auto (a folder per "
          "dataset under ~/Library/Caches/vpipe/train), a directory, or "
          "none. Needs the model port wired -- the cache is per model. Keep "
          "it on the internal SSD.",
   .def_str = "auto",
   .is_path = true,
   .path_kind = "dir"},
  {.key = "validation",
   .type = ConfigType::Int,
   .doc = "pictures held out of training and scored at fixed noise at "
          "every checkpoint: -1 (auto: 1% of a set of 500 or more, at most "
          "64; none below), 0, or a count",
   .def_int = -1},
};

const PortSpec kIports[] = {
  {.name = "model",
   .doc = "OPTIONAL model-select source: the checkpoint the encoded "
          "dataset is cached for",
   .type = &typeid(FlexDataPayload)},
};

const PortSpec kOports[] = {
  {.name = "images",
   .doc = "planar U8 [C,H,W] pictures fitted to their buckets, for "
          "vae-encode (one per sample, resolution and flip)",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "captions",
   .doc = "caption text for the diffusion-conditioner: every variant, "
          "then the empty prompt, then the preview prompts",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "manifest",
   .doc = "ONE beat describing every sample and the order of the other "
          "two streams, for train-lora",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = "training-dataset",
  .doc = "Source: a folder of pictures and captions as the streams a LoRA "
         "training graph encodes -- pictures fitted to aspect buckets for "
         "vae-encode, captions with the trigger word applied for the "
         "diffusion-conditioner, and a manifest that tells train-lora how "
         "they pair; the encodes are cached per model, so a re-run encodes "
         "only what changed. 0-1 in / 3 out.",
  .display_name = "Training Dataset",
  .category = StageCategory::Generative,
  .iports = kIports,
  .oports = kOports,
  .attrs = kAttrs,
};

bool
is_image_(const fs::path& p)
{
  std::string e = p.extension().string();
  std::transform(e.begin(), e.end(), e.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".webp" ||
         e == ".bmp" || e == ".tif" || e == ".tiff";
}

std::string
trim_(std::string s)
{
  auto ns = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), ns));
  s.erase(std::find_if(s.rbegin(), s.rend(), ns).base(), s.end());
  return s;
}

std::string
lower_(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return s;
}

void
replace_all_(std::string& s, const std::string& from, const std::string& to)
{
  if (from.empty()) { return; }
  std::size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
  }
}

// Whether `word` occurs in `text` as a whole word, ignoring case.
bool
has_word_(const std::string& text, const std::string& word)
{
  if (word.empty()) { return false; }
  const std::string t = lower_(text), w = lower_(word);
  std::size_t p = 0;
  while ((p = t.find(w, p)) != std::string::npos) {
    const bool left = p == 0 || !std::isalnum((unsigned char)t[p - 1]);
    const std::size_t e = p + w.size();
    const bool right = e >= t.size() || !std::isalnum((unsigned char)t[e]);
    if (left && right) { return true; }
    ++p;
  }
  return false;
}

std::vector<std::string>
split_tags_(const std::string& s)
{
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) {
    t = trim_(t);
    if (!t.empty()) { v.push_back(t); }
  }
  return v;
}

std::string
join_tags_(const std::vector<std::string>& v)
{
  std::string o;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i > 0) { o += ", "; }
    o += v[i];
  }
  return o;
}

}  // namespace

TrainingDatasetStage::TrainingDatasetStage(const SessionContextIntf* s,
                                           std::string id,
                                           std::vector<InEdge> iports,
                                           FlexData config)
  : TypedStage<TrainingDatasetStage>(s, std::move(id), std::move(iports),
                                     std::move(config)),
    _libs(s ? s->services()->ffmpeg_libraries() : nullptr)
{
  _dir = attr_str("dir");
  _captions = attr_str("captions");
  _caption_ext = attr_str("caption_ext");
  _trigger = trim_(attr_str("trigger_word"));
  _template = attr_str("caption_template");
  _missing = attr_str("missing_caption");
  _shuffle_tags = attr_bool("shuffle_tags");
  _keep_tags = std::max<int>(0, (int)attr_int("keep_tags"));
  _variants = std::max<int>(0, (int)attr_int("caption_variants"));
  _bucket_step = (int)attr_int("bucket_step");
  _max_aspect = attr_real("max_aspect");
  _flip = attr_str("flip") == "random";
  _keep_alpha = attr_str("alpha") == "keep";
  _seed = (std::uint64_t)attr_int("seed");
  _cache_cfg = attr_str("cache_dir");
  if (_cache_cfg.empty()) { _cache_cfg = "auto"; }
  _validation = (int)attr_int("validation");
  auto o = this->config().as_object();
  if (o.contains("resolution")) {
    const FlexData r = o.at("resolution");
    if (r.is_array()) {
      auto a = r.as_array();
      for (std::size_t i = 0; i < a.size(); ++i) {
        _resolutions.push_back((int)a.at(i).as_int(0));
      }
    } else {
      _resolutions.push_back((int)r.as_int(0));
    }
  }
  if (_resolutions.empty()) { _resolutions.push_back(1024); }
  if (o.contains("preview_prompts")) {
    const FlexData p = o.at("preview_prompts");
    if (p.is_array()) {
      auto a = p.as_array();
      for (std::size_t i = 0; i < a.size(); ++i) {
        _preview.push_back(std::string(a.at(i).as_string("")));
      }
    } else if (p.is_string()) {
      _preview.push_back(std::string(p.as_string("")));
    }
  }
  if (_preview.empty() && !_trigger.empty()) {
    _preview.push_back("{trigger}");
  }
  for (std::string& p : _preview) { replace_all_(p, "{trigger}", _trigger); }

  if (_captions != "sidecar" && _captions != "jsonl" && _captions != "none") {
    fail_config(fmt("TrainingDatasetStage('{}'): captions must be sidecar, "
                    "jsonl or none (got \"{}\")", this->id(), _captions));
  }
  if (_missing != "trigger_only" && _missing != "skip" &&
      _missing != "error") {
    fail_config(fmt("TrainingDatasetStage('{}'): missing_caption must be "
                    "trigger_only, skip or error (got \"{}\")", this->id(),
                    _missing));
  }
  for (int r : _resolutions) {
    if (r < 64 || r > 8192) {
      fail_config(fmt("TrainingDatasetStage('{}'): resolution {} is out of "
                      "range [64, 8192]", this->id(), r));
    }
  }
  if (_bucket_step < 8 || _max_aspect < 1.0) {
    fail_config(fmt("TrainingDatasetStage('{}'): bucket_step must be at "
                    "least 8 and max_aspect at least 1", this->id()));
  }
  allocate_oports(spec().oports.size());
}

TrainingDatasetStage::~TrainingDatasetStage() = default;

const StageSpec&
TrainingDatasetStage::spec() const noexcept
{
  return kSpec;
}

void
TrainingDatasetStage::reset_run_state()
{
  _started = false;
  _samples.clear();
  _next = 0;
  _tail = 0;
  _cache_dir.clear();
  _null_key.clear();
  _null_cached = false;
  _preview_keys.clear();
  _preview_cached.clear();
  _model_latched = false;
}

void
TrainingDatasetStage::apply_constant(unsigned iport, const FlexData& beat)
{
  if (iport == 0) { (void)apply_model_select_beat(beat, _hf_dir); }
}

std::string
TrainingDatasetStage::default_cache_root()
{
  const char* h = std::getenv("HOME");
  return (fs::path(h != nullptr ? h : "/tmp") / "Library" / "Caches" /
          "vpipe" / "train").string();
}

void
TrainingDatasetStage::key_samples(std::vector<Sample>* samples,
                                  bool keep_alpha,
                                  const std::set<std::string>& cached)
{
  using genai::train::cache_key;
  for (Sample& s : *samples) {
    s.caption_keys.clear();
    s.caption_cached.clear();
    for (const std::string& c : s.captions) {
      const std::string k = cache_key({"text", c});
      s.caption_keys.push_back(k);
      s.caption_cached.push_back(cached.count(k) != 0);
    }
    const std::string size = std::to_string(s.size);
    const std::string mt = std::to_string(s.mtime);
    for (Picture& p : s.pictures) {
      p.key = cache_key({"latent", s.rel, size, mt, std::to_string(p.w),
                         std::to_string(p.h), p.flip ? "flip" : "",
                         keep_alpha ? "rgba" : "rgb"});
      p.cached = cached.count(p.key) != 0;
    }
  }
}

int
TrainingDatasetStage::validation_count(int validation, int samples)
{
  if (samples < 2) { return 0; }
  if (validation < 0) {
    return samples >= 500 ? std::min(64, std::max(1, samples / 100)) : 0;
  }
  return std::min(validation, samples - 1);
}

void
TrainingDatasetStage::hold_out(std::vector<Sample>* samples, int n,
                               std::uint64_t seed)
{
  std::vector<std::size_t> idx(samples->size());
  for (std::size_t i = 0; i < idx.size(); ++i) { idx[i] = i; }
  std::mt19937_64 rng(seed ^ 0x76616c6964ull);
  std::shuffle(idx.begin(), idx.end(), rng);
  for (auto& s : *samples) { s.validation = false; }
  for (int i = 0; i < n && i < (int)idx.size(); ++i) {
    (*samples)[idx[(std::size_t)i]].validation = true;
  }
}

void
TrainingDatasetStage::bucket(int w, int h, int resolution, int step,
                             double max_aspect, int* bw, int* bh)
{
  double a = (double)w / (double)std::max(h, 1);
  a = std::clamp(a, 1.0 / max_aspect, max_aspect);
  const double area = (double)resolution * resolution;
  const double fw = std::sqrt(area * a), fh = std::sqrt(area / a);
  *bw = std::max(step, (int)std::lround(fw / step) * step);
  *bh = std::max(step, (int)std::lround(fh / step) * step);
}

std::string
TrainingDatasetStage::apply_trigger(const std::string& caption,
                                    const std::string& trigger,
                                    const std::string& tmpl)
{
  if (trigger.empty()) { return caption; }
  if (caption.find("[trigger]") != std::string::npos) {
    std::string c = caption;
    replace_all_(c, "[trigger]", trigger);
    return c;
  }
  if (has_word_(caption, trigger)) { return caption; }
  if (caption.empty()) { return trigger; }
  std::string out = tmpl;
  replace_all_(out, "{trigger}", trigger);
  replace_all_(out, "{caption}", caption);
  return trim_(out);
}

bool
TrainingDatasetStage::scan(std::vector<Sample>* out, std::string* err) const
{
  out->clear();
  std::error_code ec;
  if (_dir.empty() || !fs::is_directory(_dir, ec)) {
    if (err != nullptr) { *err = "'" + _dir + "' is not a folder"; }
    return false;
  }
  const fs::path root = fs::path(_dir);
  std::vector<fs::path> files;
  for (fs::recursive_directory_iterator it(root, ec), end;
       !ec && it != end; it.increment(ec)) {
    if (it->is_regular_file(ec) && is_image_(it->path())) {
      files.push_back(it->path());
    }
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    if (err != nullptr) { *err = "no pictures under '" + _dir + "'"; }
    return false;
  }

  // metadata.jsonl, by file name relative to the folder.
  std::vector<std::pair<std::string, std::vector<std::string>>> jsonl;
  if (_captions == "jsonl") {
    std::ifstream in(root / "metadata.jsonl");
    std::string line;
    while (std::getline(in, line)) {
      line = trim_(line);
      if (line.empty()) { continue; }
      FlexData row;
      try {
        std::istringstream ls(line);
        row = FlexData::from_json(ls);
      } catch (...) {
        continue;
      }
      if (!row.is_object()) { continue; }
      auto o = row.as_object();
      if (!o.contains("file_name")) { continue; }
      std::vector<std::string> caps;
      for (const char* k : {"text", "caption"}) {
        if (!o.contains(k)) { continue; }
        const FlexData t = o.at(k);
        if (t.is_array()) {
          auto a = t.as_array();
          for (std::size_t i = 0; i < a.size(); ++i) {
            caps.push_back(trim_(std::string(a.at(i).as_string(""))));
          }
        } else {
          caps.push_back(trim_(std::string(t.as_string(""))));
        }
        break;
      }
      jsonl.push_back({std::string(o.at("file_name").as_string("")),
                       std::move(caps)});
    }
  }

  std::mt19937_64 rng(_seed);
  const std::regex rep("^([0-9]+)_(.+)$");
  for (const fs::path& f : files) {
    Sample s;
    s.path = f.string();
    s.rel = fs::relative(f, root, ec).generic_string();
    s.size = (std::uint64_t)fs::file_size(f, ec);
    s.mtime = (std::int64_t)fs::last_write_time(f, ec)
                  .time_since_epoch().count();
    // N_name folders repeat.
    const std::string first = fs::path(s.rel).begin()->string();
    std::smatch m;
    if (fs::path(s.rel).has_parent_path() &&
        std::regex_match(first, m, rep)) {
      s.repeats = std::max(1, std::atoi(m[1].str().c_str()));
    }
    // Captions.
    std::vector<std::string> raw;
    if (_captions == "sidecar") {
      fs::path cp = f;
      cp.replace_extension(_caption_ext);
      std::ifstream in(cp);
      std::string line;
      while (std::getline(in, line)) {
        line = trim_(line);
        if (!line.empty()) { raw.push_back(line); }
      }
    } else if (_captions == "jsonl") {
      for (const auto& j : jsonl) {
        if (j.first == s.rel || j.first == f.filename().string()) {
          for (const std::string& c : j.second) {
            if (!c.empty()) { raw.push_back(c); }
          }
          break;
        }
      }
    }
    if (raw.empty()) {
      if (_missing == "skip") { continue; }
      if (_missing == "error") {
        if (err != nullptr) { *err = "'" + s.rel + "' has no caption"; }
        return false;
      }
      if (_trigger.empty()) {
        if (err != nullptr) {
          *err = "'" + s.rel + "' has no caption and there is no "
                 "trigger_word to stand in for one";
        }
        return false;
      }
      raw.push_back(std::string());
    }
    for (const std::string& c : raw) {
      const std::string base = apply_trigger(c, _trigger, _template);
      s.captions.push_back(base);
      if (_shuffle_tags) {
        std::vector<std::string> tags = split_tags_(base);
        const std::size_t keep =
            std::min<std::size_t>((std::size_t)_keep_tags, tags.size());
        const int n = _variants > 0 ? _variants : 4;
        for (int v = 1; v < n && tags.size() > keep + 1; ++v) {
          std::shuffle(tags.begin() + (long)keep, tags.end(), rng);
          s.captions.push_back(join_tags_(tags));
        }
      }
    }
    // Pictures: one per resolution (and flip), each at its own bucket.
    int w = 0, h = 0;
    std::string perr;
    if (!probe_image_size(_libs, s.path, &w, &h, &perr)) {
      session()->warn(fmt("TrainingDatasetStage('{}'): skipping '{}': {}",
                          this->id(), s.rel, perr));
      continue;
    }
    for (int r : _resolutions) {
      Picture p;
      bucket(w, h, r, _bucket_step, _max_aspect, &p.w, &p.h);
      s.pictures.push_back(p);
      if (_flip) {
        p.flip = true;
        s.pictures.push_back(p);
      }
    }
    out->push_back(std::move(s));
  }
  if (out->empty()) {
    if (err != nullptr) { *err = "no usable pictures under '" + _dir + "'"; }
    return false;
  }
  return true;
}

FlexData
TrainingDatasetStage::manifest(const std::vector<Sample>& samples) const
{
  FlexData m = FlexData::make_object();
  auto o = m.as_object();
  o.insert_or_assign("kind", FlexData::make_string("training-manifest"));
  o.insert_or_assign("version", FlexData::make_int(1));
  o.insert_or_assign("dir", FlexData::make_string(_dir));
  o.insert_or_assign("trigger_word", FlexData::make_string(_trigger));
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  std::int64_t caps = 0, pics = 0;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const Sample& s = samples[i];
    FlexData e = FlexData::make_object();
    auto eo = e.as_object();
    eo.insert_or_assign("id", FlexData::make_int((std::int64_t)i));
    eo.insert_or_assign("file", FlexData::make_string(s.rel));
    eo.insert_or_assign("repeats", FlexData::make_int(s.repeats));
    FlexData ca = FlexData::make_array();
    auto caa = ca.as_array();
    for (const std::string& c : s.captions) {
      caa.push_back(FlexData::make_string(c));
    }
    eo.insert_or_assign("captions", std::move(ca));
    if (!s.caption_keys.empty()) {
      FlexData ck = FlexData::make_array(), cc = FlexData::make_array();
      for (std::size_t j = 0; j < s.caption_keys.size(); ++j) {
        ck.as_array().push_back(FlexData::make_string(s.caption_keys[j]));
        cc.as_array().push_back(FlexData::make_bool(s.caption_cached[j]));
      }
      eo.insert_or_assign("caption_keys", std::move(ck));
      eo.insert_or_assign("caption_cached", std::move(cc));
    }
    if (s.validation) {
      eo.insert_or_assign("validation", FlexData::make_bool(true));
    }
    FlexData pa = FlexData::make_array();
    auto paa = pa.as_array();
    for (const Picture& p : s.pictures) {
      FlexData po = FlexData::make_object();
      auto poo = po.as_object();
      poo.insert_or_assign("w", FlexData::make_int(p.w));
      poo.insert_or_assign("h", FlexData::make_int(p.h));
      poo.insert_or_assign("flip", FlexData::make_bool(p.flip));
      if (!p.key.empty()) {
        poo.insert_or_assign("key", FlexData::make_string(p.key));
        poo.insert_or_assign("cached", FlexData::make_bool(p.cached));
      }
      paa.push_back(std::move(po));
    }
    eo.insert_or_assign("pictures", std::move(pa));
    a.push_back(std::move(e));
    caps += (std::int64_t)s.captions.size();
    pics += (std::int64_t)s.pictures.size();
  }
  o.insert_or_assign("samples", std::move(arr));
  o.insert_or_assign("captions_total", FlexData::make_int(caps));
  o.insert_or_assign("pictures_total", FlexData::make_int(pics));
  // The empty prompt follows the captions, then the previews.
  o.insert_or_assign("null_caption", FlexData::make_bool(true));
  FlexData pv = FlexData::make_array();
  auto pva = pv.as_array();
  for (const std::string& p : _preview) {
    pva.push_back(FlexData::make_string(p));
  }
  o.insert_or_assign("preview_prompts", std::move(pv));
  // The cache: where train-lora reads the cached entries back and writes
  // the new ones, and the keys of the texts after the samples.
  if (!_cache_dir.empty()) {
    o.insert_or_assign("cache_dir", FlexData::make_string(_cache_dir));
    o.insert_or_assign("null_key", FlexData::make_string(_null_key));
    o.insert_or_assign("null_cached", FlexData::make_bool(_null_cached));
    FlexData pk = FlexData::make_array(), pc = FlexData::make_array();
    for (std::size_t i = 0; i < _preview_keys.size(); ++i) {
      pk.as_array().push_back(FlexData::make_string(_preview_keys[i]));
      pc.as_array().push_back(FlexData::make_bool(_preview_cached[i]));
    }
    o.insert_or_assign("preview_keys", std::move(pk));
    o.insert_or_assign("preview_cached", std::move(pc));
  }
  return m;
}

Job
TrainingDatasetStage::process(RuntimeContext& ctx)
{
  auto caption_beat = [](const std::string& text, bool empty_ok) {
    FlexData c = FlexData::make_object();
    auto o = c.as_object();
    o.insert_or_assign(beat::kText, FlexData::make_string(text));
    o.insert_or_assign(beat::kBatch, FlexData::make_bool(true));
    if (empty_ok) {
      o.insert_or_assign(beat::kAllowEmpty, FlexData::make_bool(true));
    }
    return make_payload<FlexDataPayload>(std::move(c));
  };

  if (!_started) {
    _started = true;
    if (!_model_latched && ctx.num_iports() > 0 && ctx.iport_connected(0)) {
      auto mb = co_await ctx.read(0);
      _model_latched = true;
      if (const auto* f =
              mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
        (void)apply_model_select_beat(f->data, _hf_dir);
      }
    }
    std::string err;
    if (!scan(&_samples, &err)) {
      session()->error(fmt("TrainingDatasetStage('{}'): {}", this->id(),
                           err));
      ctx.signal_done();
      co_return;
    }
    hold_out(&_samples, validation_count(_validation, (int)_samples.size()),
             _seed);

    // The cache, per (dataset, model): nothing to key it to without a
    // model, so nothing is cached then.
    std::set<std::string> cached;
    if (_cache_cfg != "none" && !_hf_dir.empty()) {
      std::error_code ec;
      const std::string model =
          fs::weakly_canonical(resolve_model_dir(session(), _hf_dir), ec)
              .string();
      const fs::path root =
          _cache_cfg == "auto"
              ? fs::path(default_cache_root()) /
                    genai::train::cache_key(
                        {"dataset", fs::weakly_canonical(_dir, ec).string()})
              : fs::path(_cache_cfg);
      _cache_dir = (root / genai::train::cache_key({"model", model}))
                       .string();
      cached = genai::train::SampleCache::keys_in(_cache_dir);
      key_samples(&_samples, _keep_alpha, cached);
      _null_key = genai::train::cache_key({"text", ""});
      _null_cached = cached.count(_null_key) != 0;
      for (const std::string& p : _preview) {
        const std::string k = genai::train::cache_key({"text", p});
        _preview_keys.push_back(k);
        _preview_cached.push_back(cached.count(k) != 0);
      }
    } else if (_cache_cfg != "none") {
      session()->info(fmt("TrainingDatasetStage('{}'): no model wired to "
                          "port 0, so the encoded dataset is not cached",
                          this->id()));
    }
    std::size_t caps = 0, pics = 0, held = 0, hits = 0;
    for (const Sample& s : _samples) {
      caps += s.captions.size();
      pics += s.pictures.size();
      held += s.validation ? 1 : 0;
      for (bool c : s.caption_cached) { hits += c ? 1 : 0; }
      for (const Picture& p : s.pictures) { hits += p.cached ? 1 : 0; }
    }
    session()->info(fmt(
        "TrainingDatasetStage('{}'): {} pictures, {} captions and {} "
        "encodes from '{}'{}{}{}", this->id(), _samples.size(), caps, pics,
        _dir, _trigger.empty() ? std::string()
                               : fmt(", trigger \"{}\"", _trigger)(),
        held > 0 ? fmt("; {} held out for validation", held)()
                 : std::string(),
        _cache_dir.empty()
            ? std::string()
            : fmt("; {} of {} cached in '{}'", hits, caps + pics,
                  _cache_dir)()));
    co_await ctx.write(2, make_payload<FlexDataPayload>(manifest(_samples)));
    co_return;
  }

  if (_next < _samples.size()) {
    const Sample& s = _samples[_next++];
    for (std::size_t i = 0; i < s.captions.size(); ++i) {
      if (i < s.caption_cached.size() && s.caption_cached[i]) { continue; }
      co_await ctx.write(1, caption_beat(s.captions[i], false));
    }
    for (const Picture& p : s.pictures) {
      if (p.cached) { continue; }
      ImageDecodeOptions opt;
      opt.keep_alpha = _keep_alpha;
      opt.out_w = p.w;
      opt.out_h = p.h;
      opt.flip = p.flip;
      std::string err;
      auto img = decode_image_file(_libs, s.path, opt, &err);
      if (!img) {
        // The manifest promised this picture: a missing one would shift
        // every later latent onto the wrong caption.
        session()->error(fmt("TrainingDatasetStage('{}'): '{}' decoded at "
                             "scan time and not now: {}", this->id(),
                             s.rel, err));
        ctx.signal_done();
        co_return;
      }
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign(beat::kBatch,
                                      FlexData::make_bool(true));
      img->sideband = std::move(sb);
      co_await ctx.write(0, std::move(img));
    }
    co_return;
  }

  if (_tail == 0) {
    _tail = 1;
    if (!_null_cached) {
      co_await ctx.write(1, caption_beat(std::string(), true));
    }
    for (std::size_t i = 0; i < _preview.size(); ++i) {
      if (i < _preview_cached.size() && _preview_cached[i]) { continue; }
      co_await ctx.write(1, caption_beat(_preview[i], false));
    }
  }
  ctx.signal_done();
}

VPIPE_REGISTER_STAGE(TrainingDatasetStage)
VPIPE_REGISTER_SPEC(TrainingDatasetStage, kSpec)

}  // namespace vpipe
