#include "stages/training-dataset-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "common/media-decode.h"
#include "generative-models/minimax-h3/minimax-h3-references.h"
#include "generative-models/train/sample-cache.h"
#include "generative-models/generative-model-manager.h"
#include "stages/image-decode.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
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
constexpr SpecExtra kMediaChoices[] = {
  {spec_key::kChoices, "auto,pictures,clips"},
};
constexpr SpecExtra kAudioChoices[] = {
  {spec_key::kChoices, "track,none"},
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
  {.key = "media",
   .type = ConfigType::String,
   .doc = "what the folder holds: auto (pictures, and .mp4 .mov .mkv .webm "
          ".avi .m4v files as clips), pictures, or clips. A clip is cut "
          "into windows of a `frames` bucket for a video model (MiniMax-H3)",
   .def_str = "auto",
   .extra = kMediaChoices},
  {.key = "fps",
   .type = ConfigType::Real,
   .doc = "the rate a clip is put on before it is cut, frames held and "
          "dropped whole as ffmpeg's fps filter does (default 24, "
          "MiniMax-H3's)",
   .def_real = 24.0},
  {.key = "frames",
   .type = ConfigType::Any,
   .doc = "a clip window's frame count, or a list of them: each clip takes "
          "the longest it fills (default 39). MiniMax-H3 takes 17n + 5 "
          "(22, 39, 56, 73, ...); its previews decode from 39"},
  {.key = "windows",
   .type = ConfigType::Int,
   .doc = "windows a clip yields, spread evenly over it (default 1, its "
          "start); each step trains one of a clip's windows",
   .def_int = 1},
  {.key = "audio",
   .type = ConfigType::String,
   .doc = "a clip window's soundtrack: track (its own stretch of the "
          "clip's audio, for audio-vae-encode on port 3) or none. A clip "
          "without an audio stream trains without one either way",
   .def_str = "track",
   .extra = kAudioChoices},
  {.key = "task_mix",
   .type = ConfigType::Any,
   .doc = "keyframe tasks for clips, by weight: {t2va, i2va (the first "
          "frame anchored), l2va (the last), fl2va (both)}, e.g. {\"t2va\": "
          "0.5, \"i2va\": 0.3, \"fl2va\": 0.2}; each window draws one by "
          "the seed (default {\"t2va\": 1}). The anchors are the window's "
          "own frames, encoded as stills; with the keyframes port wired to "
          "the conditioner's ref_images, its captions are encoded again with "
          "them shown"},
  {.key = "audio_sample_rate",
   .type = ConfigType::Int,
   .doc = "the rate soundtracks are resampled to (default 32000, "
          "MiniMax-H3's audio VAE)",
   .def_int = 32000},
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
          "streams, for train-lora",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "audio",
   .doc = "OPTIONAL: each clip window's soundtrack, f32 planar [2, N] at "
          "audio_sample_rate, for audio-vae-encode (one per window that "
          "has one, in the manifest's order)",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "keyframes",
   .doc = "OPTIONAL, for the diffusion-conditioner's ref_images: one "
          "picture list per caption beat, in step with the captions port "
          "-- empty for a plain caption, a keyframe window's frames for the "
          "captions presented with them",
   .type = &typeid(TensorListPayload), .tags = "rgb-frames-list",
   .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = "training-dataset",
  .doc = "Source: a folder of pictures and captions as the streams a LoRA "
         "training graph encodes -- pictures fitted to aspect buckets for "
         "vae-encode, captions with the trigger word applied for the "
         "diffusion-conditioner, and a manifest that tells train-lora how "
         "they pair; the encodes are cached per model, so a re-run encodes "
         "only what changed. Clips (for a video model) are cut into "
         "windows, their soundtracks on port 3, keyframes on port 4. 0-1 "
         "in / 5 out.",
  .display_name = "Training Dataset",
  .category = StageCategory::Generative,
  .iports = kIports,
  .oports = kOports,
  .attrs = kAttrs,
};

bool
is_clip_(const fs::path& p)
{
  std::string e = p.extension().string();
  std::transform(e.begin(), e.end(), e.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return e == ".mp4" || e == ".mov" || e == ".mkv" || e == ".webm" ||
         e == ".avi" || e == ".m4v";
}

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
  _media = attr_str("media");
  if (_media.empty()) { _media = "auto"; }
  _fps = attr_real("fps");
  _windows = std::max<int>(1, (int)attr_int("windows"));
  _audio = attr_str("audio") != "none";
  _audio_rate = (int)attr_int("audio_sample_rate");
  auto o = this->config().as_object();
  if (o.contains("frames")) {
    const FlexData r = o.at("frames");
    if (r.is_array()) {
      auto a = r.as_array();
      for (std::size_t i = 0; i < a.size(); ++i) {
        _frames.push_back((int)a.at(i).as_int(0));
      }
    } else {
      _frames.push_back((int)r.as_int(0));
    }
  }
  if (_frames.empty()) { _frames.push_back(39); }
  if (o.contains("task_mix")) {
    const FlexData t = o.at("task_mix");
    if (t.is_object()) {
      FlexData tc = t;
      auto to = tc.as_object();
      for (const char* k : {"t2va", "i2va", "l2va", "fl2va"}) {
        if (!to.contains(k)) { continue; }
        const double w = to.at(k).as_real(0.0);
        if (w > 0.0) { _tasks.push_back({task_anchors(k), w}); }
      }
      for (const auto& [k, v] : to) {
        if (task_anchors(std::string(k)) < 0) {
          fail_config(fmt("TrainingDatasetStage('{}'): task_mix names '{}' "
                          "-- the tasks are t2va, i2va, l2va and fl2va",
                          this->id(), k));
        }
      }
    } else {
      fail_config(fmt("TrainingDatasetStage('{}'): task_mix must be an "
                      "object of task weights", this->id()));
    }
  }
  if (_tasks.empty()) { _tasks.push_back({0, 1.0}); }
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
  if (_media != "auto" && _media != "pictures" && _media != "clips") {
    fail_config(fmt("TrainingDatasetStage('{}'): media must be auto, "
                    "pictures or clips (got \"{}\")", this->id(), _media));
  }
  if (!(_fps > 0.0) || _audio_rate < 8000) {
    fail_config(fmt("TrainingDatasetStage('{}'): fps must be positive and "
                    "audio_sample_rate at least 8000", this->id()));
  }
  for (int f : _frames) {
    if (f < 2) {
      fail_config(fmt("TrainingDatasetStage('{}'): a clip window of {} "
                      "frames is not a clip", this->id(), f));
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
  _decode_booked = 0;
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

int
TrainingDatasetStage::clip_bucket(int clip_frames,
                                  const std::vector<int>& buckets)
{
  int best = 0;
  for (int f : buckets) {
    if (f <= clip_frames) { best = std::max(best, f); }
  }
  return best;
}

std::vector<int>
TrainingDatasetStage::window_starts(int clip_frames, int window, int count)
{
  std::vector<int> v;
  const int last = clip_frames - window;
  if (window <= 0 || last < 0) { return v; }
  count = std::max(1, count);
  for (int i = 0; i < count; ++i) {
    const int st = count == 1
        ? 0 : (int)std::lround((double)i * last / (double)(count - 1));
    if (v.empty() || st != v.back()) { v.push_back(st); }
  }
  return v;
}

int
TrainingDatasetStage::frames_at_fps(long long src_frames, double src_fps,
                                    double seconds, double fps)
{
  if (src_frames > 0 && src_fps > 0.0) {
    const std::vector<int> c = genai::minimax_h3::frame_resample_counts(
        (int)src_frames, src_fps, fps);
    int n = 0;
    for (int k : c) { n += k; }
    return n;
  }
  // No frame count in the container: from its length, a frame short of
  // what it may hold, since the last slot's rounding is not known.
  return std::max(0, (int)std::floor(seconds * fps) - 1);
}

int
TrainingDatasetStage::task_anchors(const std::string& task)
{
  if (task == "t2va") { return 0; }
  if (task == "i2va") { return 1; }
  if (task == "l2va") { return 2; }
  if (task == "fl2va") { return 3; }
  return -1;
}

int
TrainingDatasetStage::draw_task(
    const std::vector<std::pair<int, double>>& tasks, std::uint64_t seed,
    const std::string& window)
{
  double total = 0.0;
  for (const auto& t : tasks) { total += t.second; }
  if (tasks.empty() || !(total > 0.0)) { return 0; }
  std::uint64_t h = 1469598103934665603ull ^ seed;
  for (unsigned char c : window) { h = (h ^ c) * 1099511628211ull; }
  std::mt19937_64 rng(h);
  const double u = std::uniform_real_distribution<double>(0.0, total)(rng);
  double acc = 0.0;
  for (const auto& t : tasks) {
    acc += t.second;
    if (u < acc) { return t.first; }
  }
  return tasks.back().first;
}

void
TrainingDatasetStage::key_samples(std::vector<Sample>* samples,
                                  bool keep_alpha,
                                  const std::set<std::string>& cached,
                                  double fps, int audio_rate)
{
  using genai::train::cache_key;
  char fb[32];
  std::snprintf(fb, sizeof fb, "%.6g", fps);
  for (Sample& s : *samples) {
    s.caption_keys.clear();
    s.caption_cached.clear();
    for (const std::string& c : s.captions) {
      // A caption encoded with references is that request's, keyed by
      // the files and the duration they are cut to.
      const std::string k =
          s.references.empty()
              ? cache_key({"text", c})
              : cache_key({"text", c, "references", s.ref_sig,
                           std::to_string(s.ref_frames)});
      s.caption_keys.push_back(k);
      s.caption_cached.push_back(cached.count(k) != 0);
    }
    const std::string size = std::to_string(s.size);
    const std::string mt = std::to_string(s.mtime);
    for (Picture& p : s.pictures) {
      if (p.frames > 0) {
        // A clip window: which frames, at which rate, and its soundtrack
        // keyed apart (a window can lose or gain its audio alone).
        const std::string fr = std::to_string(p.frames);
        const std::string st = std::to_string(p.start);
        p.key = cache_key({"latent", s.rel, size, mt, std::to_string(p.w),
                           std::to_string(p.h), p.flip ? "flip" : "",
                           keep_alpha ? "rgba" : "rgb", "clip", fr, st, fb});
        p.cached = cached.count(p.key) != 0;
        // The anchors are the window's frames, the presented captions its
        // captions shown them: both keyed off the window.
        p.anchor_keys.clear();
        p.anchor_cached.clear();
        for (int b = 0; b < 2; ++b) {
          if ((p.anchors & (1 << b)) == 0) { continue; }
          p.anchor_keys.push_back(
              cache_key({"anchor", p.key, b == 0 ? "first" : "last"}));
          p.anchor_cached.push_back(cached.count(p.anchor_keys.back()) != 0);
        }
        p.caption_keys.clear();
        p.caption_cached.clear();
        for (const std::string& c : p.captions) {
          p.caption_keys.push_back(cache_key(
              {"text", c, "keyframes", p.key, std::to_string(p.anchors)}));
          p.caption_cached.push_back(cached.count(p.caption_keys.back()) !=
                                     0);
        }
        if (p.audio) {
          p.audio_key = cache_key({"audio", s.rel, size, mt, fr, st, fb,
                                   std::to_string(audio_rate)});
          p.audio_cached = cached.count(p.audio_key) != 0;
        }
        continue;
      }
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
    // A `references` folder holds what samples are conditioned on, not
    // samples.
    if (it->is_directory(ec) && it->path().filename() == "references") {
      it.disable_recursion_pending();
      continue;
    }
    if (!it->is_regular_file(ec)) { continue; }
    const bool pic = is_image_(it->path()), clip = is_clip_(it->path());
    if ((pic && _media != "clips") || (clip && _media != "pictures")) {
      files.push_back(it->path());
    }
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    if (err != nullptr) {
      *err = "no " + std::string(_media == "clips"      ? "clips"
                                 : _media == "pictures" ? "pictures"
                                                        : "pictures or "
                                                          "clips") +
             " under '" + _dir + "'";
    }
    return false;
  }

  // metadata.jsonl, by file name relative to the folder.
  std::vector<std::pair<std::string, std::vector<std::string>>> jsonl;
  std::map<std::string, std::vector<std::string>> jsonl_refs;
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
      if (o.contains("references")) {
        const FlexData r = o.at("references");
        std::vector<std::string> rv;
        if (r.is_string()) {
          rv.push_back(std::string(r.as_string("")));
        } else if (r.is_array()) {
          auto ra = r.as_array();
          for (std::size_t i = 0; i < ra.size(); ++i) {
            rv.push_back(std::string(ra.at(i).as_string("")));
          }
        }
        jsonl_refs[std::string(o.at("file_name").as_string(""))] =
            std::move(rv);
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
    // Its references: a sidecar, or the jsonl row's.
    {
      std::vector<std::string> rr;
      fs::path rp = f;
      rp.replace_extension(".refs");
      if (fs::exists(rp, ec)) {
        std::ifstream in(rp);
        std::string line;
        while (std::getline(in, line)) {
          line = trim_(line);
          if (!line.empty() && line[0] != '#') { rr.push_back(line); }
        }
      } else if (_captions == "jsonl") {
        auto it = jsonl_refs.find(s.rel);
        if (it == jsonl_refs.end()) {
          it = jsonl_refs.find(f.filename().string());
        }
        if (it != jsonl_refs.end()) { rr = it->second; }
      }
      for (const std::string& r0 : rr) {
        if (r0.empty()) { continue; }
        // Absolute, else beside the sample, else under the folder.
        fs::path q = fs::path(r0);
        if (q.is_relative()) {
          q = f.parent_path() / r0;
          if (!fs::exists(q, ec)) { q = root / r0; }
        }
        if (!fs::is_regular_file(q, ec)) {
          if (err != nullptr) {
            *err = "'" + s.rel + "' names a reference '" + r0 +
                   "' that is not a file";
          }
          return false;
        }
        q = fs::weakly_canonical(q, ec);
        s.references.push_back(q.string());
        s.ref_sig += q.string() + "|" +
                     std::to_string(fs::file_size(q, ec)) + "|" +
                     std::to_string((std::int64_t)fs::last_write_time(q, ec)
                                        .time_since_epoch().count()) +
                     ";";
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
    // A CLIP: windows of the longest bucket it fills, each window at
    // every resolution (and flip).
    if (is_clip_(f)) {
      std::string perr;
      const auto pr = probe_media_file(_libs, s.path, &perr);
      if (!pr || !pr->has_video || pr->width <= 0 || pr->height <= 0) {
        session()->warn(fmt("TrainingDatasetStage('{}'): skipping '{}': {}",
                            this->id(), s.rel,
                            pr ? std::string("no video stream") : perr));
        continue;
      }
      const int cf = frames_at_fps(pr->nb_frames, pr->fps,
                                   pr->duration_seconds, _fps);
      const int win = clip_bucket(cf, _frames);
      if (win == 0) {
        session()->warn(fmt("TrainingDatasetStage('{}'): skipping '{}': {} "
                            "frames at {} fps, shorter than the shortest "
                            "window", this->id(), s.rel, cf, _fps));
        continue;
      }
      s.clip = true;
      s.clip_frames = cf;
      s.src_w = pr->width;
      s.src_h = pr->height;
      s.src_fps = pr->fps;
      s.src_seconds = pr->duration_seconds;
      s.ref_frames = win;
      const bool aud = _audio && pr->has_audio;
      for (int st : window_starts(cf, win, _windows)) {
        for (int r : _resolutions) {
          Picture p;
          bucket(pr->width, pr->height, r, _bucket_step, _max_aspect, &p.w,
                 &p.h);
          p.frames = win;
          p.start = st;
          p.audio = aud;
          for (int fl = 0; fl < (_flip ? 2 : 1); ++fl) {
            p.flip = fl == 1;
            // The window's task, by its own key.
            p.anchors = draw_task(
                _tasks, _seed,
                fmt("{}|{}|{}x{}|{}", s.rel, st, p.w, p.h, fl)());
            p.captions.clear();
            if (p.anchors != 0 && _show_keyframes) {
              p.captions = s.captions;
            }
            s.pictures.push_back(p);
          }
        }
      }
      out->push_back(std::move(s));
      continue;
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
  std::int64_t caps = 0, pics = 0, auds = 0;
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
    if (!s.references.empty()) {
      FlexData rl = FlexData::make_array();
      for (const std::string& r : s.references) {
        rl.as_array().push_back(FlexData::make_string(r));
      }
      eo.insert_or_assign("references", std::move(rl));
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
      if (p.frames > 0) {
        poo.insert_or_assign("frames", FlexData::make_int(p.frames));
        poo.insert_or_assign("start", FlexData::make_int(p.start));
        poo.insert_or_assign("audio", FlexData::make_bool(p.audio));
        if (p.audio && !p.audio_key.empty()) {
          poo.insert_or_assign("audio_key",
                               FlexData::make_string(p.audio_key));
          poo.insert_or_assign("audio_cached",
                               FlexData::make_bool(p.audio_cached));
        }
        auds += p.audio ? 1 : 0;
        if (p.anchors != 0) {
          FlexData an = FlexData::make_array();
          if (p.anchors & 1) {
            an.as_array().push_back(FlexData::make_string("first"));
          }
          if (p.anchors & 2) {
            an.as_array().push_back(FlexData::make_string("last"));
          }
          poo.insert_or_assign("anchors", std::move(an));
          if (!p.anchor_keys.empty()) {
            FlexData ak = FlexData::make_array(), ac = FlexData::make_array();
            for (std::size_t j = 0; j < p.anchor_keys.size(); ++j) {
              ak.as_array().push_back(
                  FlexData::make_string(p.anchor_keys[j]));
              ac.as_array().push_back(
                  FlexData::make_bool(p.anchor_cached[j]));
            }
            poo.insert_or_assign("anchor_keys", std::move(ak));
            poo.insert_or_assign("anchor_cached", std::move(ac));
          }
        }
        if (!p.captions.empty()) {
          FlexData pc = FlexData::make_array();
          for (const std::string& c : p.captions) {
            pc.as_array().push_back(FlexData::make_string(c));
          }
          poo.insert_or_assign("captions", std::move(pc));
          if (!p.caption_keys.empty()) {
            FlexData ck = FlexData::make_array(), cc = FlexData::make_array();
            for (std::size_t j = 0; j < p.caption_keys.size(); ++j) {
              ck.as_array().push_back(
                  FlexData::make_string(p.caption_keys[j]));
              cc.as_array().push_back(
                  FlexData::make_bool(p.caption_cached[j]));
            }
            poo.insert_or_assign("caption_keys", std::move(ck));
            poo.insert_or_assign("caption_cached", std::move(cc));
          }
          caps += (std::int64_t)p.captions.size();
        }
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
  o.insert_or_assign("audio_total", FlexData::make_int(auds));
  o.insert_or_assign("fps", FlexData::make_real(_fps));
  o.insert_or_assign("audio_sample_rate", FlexData::make_int(_audio_rate));
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
  auto caption_beat = [](const std::string& text, bool empty_ok,
                         const Sample* refs = nullptr) {
    FlexData c = FlexData::make_object();
    auto o = c.as_object();
    o.insert_or_assign(beat::kText, FlexData::make_string(text));
    o.insert_or_assign(beat::kBatch, FlexData::make_bool(true));
    if (empty_ok) {
      o.insert_or_assign(beat::kAllowEmpty, FlexData::make_bool(true));
    }
    // A sample's references and its duration: video-ref-encoder's
    // request for this caption.
    if (refs != nullptr && !refs->references.empty()) {
      FlexData a = FlexData::make_array();
      for (const std::string& r : refs->references) {
        a.as_array().push_back(FlexData::make_string(r));
      }
      o.insert_or_assign(beat::kReferences, std::move(a));
      o.insert_or_assign(beat::kFrames, FlexData::make_int(refs->ref_frames));
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
    // Keyframes are shown to the conditioner when its ref_images reads
    // them and a task anchors any.
    _show_keyframes = false;
    for (const auto& t : _tasks) {
      _show_keyframes = _show_keyframes || t.first != 0;
    }
    _show_keyframes = _show_keyframes && ctx.num_oports() > 4 &&
                      ctx.has_consumers(4);
    std::string err;
    if (!scan(&_samples, &err)) {
      session()->error(fmt("TrainingDatasetStage('{}'): {}", this->id(),
                           err));
      ctx.signal_done();
      co_return;
    }
    hold_out(&_samples, validation_count(_validation, (int)_samples.size()),
             _seed);
    // No consumer for the soundtracks: the clips train without them, and
    // the manifest says so (train-lora would otherwise wait for them).
    if (ctx.num_oports() <= 3 || !ctx.has_consumers(3)) {
      for (Sample& s : _samples) {
        for (Picture& p : s.pictures) { p.audio = false; }
      }
    }

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
      key_samples(&_samples, _keep_alpha, cached, _fps, _audio_rate);
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
    std::size_t caps = 0, pics = 0, held = 0, hits = 0, auds = 0,
                clips = 0;
    for (const Sample& s : _samples) {
      caps += s.captions.size();
      pics += s.pictures.size();
      held += s.validation ? 1 : 0;
      for (bool c : s.caption_cached) { hits += c ? 1 : 0; }
      for (const Picture& p : s.pictures) {
        hits += p.cached ? 1 : 0;
        hits += p.audio_cached ? 1 : 0;
        auds += p.audio ? 1 : 0;
        clips += p.frames > 0 ? 1 : 0;
      }
    }
    session()->info(fmt(
        "TrainingDatasetStage('{}'): {} samples, {} captions and {} "
        "encodes{} from '{}'{}{}{}", this->id(), _samples.size(), caps, pics,
        clips > 0 ? fmt(" ({} clip windows of them, {} with a soundtrack)",
                        clips, auds)()
                  : std::string(),
        _dir, _trigger.empty() ? std::string()
                               : fmt(", trigger \"{}\"", _trigger)(),
        held > 0 ? fmt("; {} held out for validation", held)()
                 : std::string(),
        _cache_dir.empty()
            ? std::string()
            : fmt("; {} of {} cached in '{}'", hits, caps + pics + auds,
                  _cache_dir)()));
    co_await ctx.write(2, make_payload<FlexDataPayload>(manifest(_samples)));
    co_return;
  }

  // Every caption beat has its keyframe list beside it, when shown.
  auto keyframes_beat = [](std::vector<TensorBeat> items) {
    auto l = std::make_unique<TensorListPayload>();
    l->items = std::move(items);
    return l;
  };

  if (_next < _samples.size()) {
    const Sample& s = _samples[_next++];
    for (std::size_t i = 0; i < s.captions.size(); ++i) {
      if (i < s.caption_cached.size() && s.caption_cached[i]) { continue; }
      co_await ctx.write(1, caption_beat(s.captions[i], false, &s));
      if (_show_keyframes) { co_await ctx.write(4, keyframes_beat({})); }
    }
    if (s.clip) {
      // ONE decode for every window of the clip, from its start to the
      // last window's end.
      bool need = false, need_audio = false;
      int end = 0;
      for (const Picture& p : s.pictures) {
        need = need || !p.cached || (p.audio && !p.audio_cached);
        need_audio = need_audio || (p.audio && !p.audio_cached);
        for (bool c : p.anchor_cached) { need = need || !c; }
        for (bool c : p.caption_cached) { need = need || !c; }
        if (p.anchor_cached.size() <
            (std::size_t)__builtin_popcount((unsigned)p.anchors)) {
          need = true;
        }
        if (p.caption_cached.size() < p.captions.size()) { need = true; }
        end = std::max(end, p.start + p.frames);
      }
      if (!need) { co_return; }
      std::string err;
      // Only the source frames the windows read are KEPT: source frame i
      // covers output slots [slot(i), slot(i + 1)) at `fps` (ffmpeg's fps
      // filter: slot(i) = floor(i * fps / src + 0.5)). Kept whole, the
      // decode held every frame up to the last window's end -- a long
      // 1080p clip's 9-11 GB.
      std::vector<std::pair<int, int>> wins;
      for (const Picture& p : s.pictures) {
        wins.push_back({p.start, p.start + p.frames});
      }
      const double dst = _fps;
      // VPIPE_DATASET_DECODE_ALL=1 keeps every frame (the A/B).
      const bool all = std::getenv("VPIPE_DATASET_DECODE_ALL") != nullptr;
      auto keep = [&wins, dst, all](int i, double src) {
        if (all) { return true; }
        const double sc = src > 0.0 ? dst / src : 1.0;
        const long long a = (long long)std::floor((double)i * sc + 0.5);
        const long long b =
            (long long)std::floor((double)(i + 1) * sc + 0.5);
        if (b <= a) { return false; }   // a dropped frame
        for (const auto& w : wins) {
          if (a < w.second && w.first < b) { return true; }
        }
        return false;
      };
      // What the decode holds: each window's distinct source frames at
      // most (frames x src / fps, and two of the resampler's rounding), at
      // the probed size, and the soundtrack whole -- booked before it runs.
      {
        const double ratio =
            s.src_fps > 0.0 ? std::max(s.src_fps / _fps, 1.0) : 1.0;
        std::size_t frames = 0;
        for (const auto& w : wins) {
          frames += (std::size_t)std::ceil((w.second - w.first) * ratio) + 2;
        }
        const std::size_t bytes =
            frames * 3 * (std::size_t)std::max(s.src_w, 1) *
                (std::size_t)std::max(s.src_h, 1) +
            (need_audio ? (std::size_t)(std::max(s.src_seconds, 0.0) *
                                        _audio_rate) * 2 * 4
                        : 0);
        if (bytes > _decode_booked) {
          _decode_booked = bytes;
          if (auto* mgr = session()->services()->generative_model_manager()) {
            mgr->revise_scratch(decode_label_(), bytes);
          }
        }
      }
      const auto v = decode_video_frames(_libs, s.path, end / _fps + 0.5,
                                         need_audio ? _audio_rate : 0, keep,
                                         &err);
      // On `fps` as ffmpeg's fps filter puts it: every frame held until
      // the next one's slot.
      std::vector<int> idx;
      if (v && v->num_frames > 0) {
        const std::vector<int> cnt = genai::minimax_h3::frame_resample_counts(
            v->num_frames, v->fps > 0.0 ? v->fps : _fps, _fps);
        for (std::size_t i = 0; i < cnt.size(); ++i) {
          for (int k = 0; k < cnt[i]; ++k) { idx.push_back((int)i); }
        }
      }
      if (!v || (int)idx.size() < end) {
        // The manifest promised these windows: a missing one would shift
        // every later latent onto the wrong caption.
        session()->error(fmt(
            "TrainingDatasetStage('{}'): '{}' {}", this->id(), s.rel,
            !v ? "did not decode: " + err
               : fmt("decoded to {} frames at {} fps where its probe "
                     "promised {}", idx.size(), _fps, end)()));
        ctx.signal_done();
        co_return;
      }
      const int H = v->height, W = v->width;
      const std::size_t plane = (std::size_t)3 * H * W;
      // Each source frame's place among the kept ones (-1: not kept).
      std::vector<int> pos((std::size_t)std::max(v->num_frames, 0), -1);
      for (std::size_t k = 0; k < v->kept.size(); ++k) {
        pos[(std::size_t)v->kept[k]] = (int)k;
      }
      bool padded = false;
      // One source frame fitted to a picture's bucket -- the picture
      // rule, cover then center-crop (and mirror) -- into [3, h, w].
      auto fit_frame = [&](const Picture& p, int at, std::uint8_t* dst,
                           genai::minimax_h3::PlanarResize* plan,
                           std::vector<std::uint8_t>* scaled) {
        const double sc = std::max((double)p.w / W, (double)p.h / H);
        const int SW = std::max(p.w, (int)std::lround(W * sc));
        const int SH = std::max(p.h, (int)std::lround(H * sc));
        const int sp = pos[(std::size_t)idx[(std::size_t)at]];
        if (sp < 0) { return false; }   // a frame the windows were not
                                         // computed to read
        const std::uint8_t* fr = v->rgb.data() + (std::size_t)sp * plane;
        if (SW != W || SH != H) {
          if (plan->empty() &&
              !genai::minimax_h3::prepare_planar_resize(H, W, SH, SW, plan)) {
            return false;
          }
          scaled->resize((std::size_t)3 * SH * SW);
          genai::minimax_h3::run_planar_resize(*plan, fr, scaled->data());
          fr = scaled->data();
        }
        const int x0 = (SW - p.w) / 2, y0 = (SH - p.h) / 2;
        for (int c = 0; c < 3; ++c) {
          for (int y = 0; y < p.h; ++y) {
            const std::uint8_t* src =
                fr + ((std::size_t)c * SH + (std::size_t)(y + y0)) * SW +
                (std::size_t)x0;
            std::uint8_t* d = dst + ((std::size_t)c * p.h + y) * p.w;
            if (p.flip) {
              for (int x = 0; x < p.w; ++x) { d[x] = src[p.w - 1 - x]; }
            } else {
              std::memcpy(d, src, (std::size_t)p.w);
            }
          }
        }
        return true;
      };
      for (const Picture& p : s.pictures) {
        genai::minimax_h3::PlanarResize kplan;
        std::vector<std::uint8_t> kscaled;
        // The keyframes, fitted as the window is: the anchors' stills and
        // the pictures shown with the presented captions.
        std::vector<TensorBeat> keys;
        for (int b = 0; b < 2; ++b) {
          if ((p.anchors & (1 << b)) == 0) { continue; }
          TensorBeat k;
          k.dtype = TensorBeat::DType::U8;
          k.shape = {3, p.h, p.w};
          k.resize_contiguous((std::size_t)3 * p.h * p.w);
          if (!fit_frame(p, b == 0 ? p.start : p.start + p.frames - 1,
                         k.as_u8(), &kplan, &kscaled)) {
            session()->error(fmt("TrainingDatasetStage('{}'): '{}' cannot "
                                 "be resized to {}x{}", this->id(), s.rel,
                                 p.w, p.h));
            ctx.signal_done();
            co_return;
          }
          keys.push_back(std::move(k));
        }
        if (!p.cached) {
          // The picture rule: scaled to COVER the bucket, aspect kept,
          // and center-cropped to it -- here by Lanczos, frame by frame.
          genai::minimax_h3::PlanarResize plan;
          std::vector<std::uint8_t> scaled;
          auto tb = std::make_unique<TensorBeatPayload>();
          tb->dtype = TensorBeat::DType::U8;
          tb->shape = {p.frames, 3, p.h, p.w};
          const std::size_t fsz = (std::size_t)3 * p.h * p.w;
          tb->resize_contiguous((std::size_t)p.frames * fsz);
          for (int j = 0; j < p.frames; ++j) {
            if (!fit_frame(p, p.start + j, tb->as_u8() + (std::size_t)j * fsz,
                           &plan, &scaled)) {
              session()->error(fmt("TrainingDatasetStage('{}'): '{}' cannot "
                                   "be resized to {}x{}", this->id(), s.rel,
                                   p.w, p.h));
              ctx.signal_done();
              co_return;
            }
          }
          FlexData sb = FlexData::make_object();
          auto so = sb.as_object();
          so.insert_or_assign(beat::kBatch, FlexData::make_bool(true));
          so.insert_or_assign(sideband::kFrames,
                              FlexData::make_int(p.frames));
          so.insert_or_assign("fps", FlexData::make_real(_fps));
          tb->sideband = std::move(sb);
          co_await ctx.write(0, std::move(tb));
        }
        // Its anchors, as stills, in the order first, last.
        for (std::size_t k = 0; k < keys.size(); ++k) {
          if (k < p.anchor_cached.size() && p.anchor_cached[k]) { continue; }
          auto ab = std::make_unique<TensorBeatPayload>();
          static_cast<TensorBeat&>(*ab) = keys[k];
          FlexData sb = FlexData::make_object();
          sb.as_object().insert_or_assign(beat::kBatch,
                                          FlexData::make_bool(true));
          ab->sideband = std::move(sb);
          co_await ctx.write(0, std::move(ab));
        }
        if (p.audio && !p.audio_cached) {
          // The window's own stretch of the soundtrack, stereo (a mono
          // track repeated, as the model's reference path upmixes).
          const int ch = std::max(v->audio_channels, 0);
          const std::size_t total =
              ch > 0 ? v->pcm.size() / (std::size_t)ch : 0;
          const std::size_t a0 =
              (std::size_t)std::llround(p.start / _fps * _audio_rate);
          const std::size_t n =
              (std::size_t)std::llround(p.frames / _fps * _audio_rate);
          padded = padded || a0 + n > total;
          auto tb = std::make_unique<TensorBeatPayload>();
          tb->dtype = TensorBeat::DType::F32;
          tb->shape = {2, (std::int64_t)n};
          tb->resize_contiguous(2 * n);
          float* d = tb->as_f32();
          for (int c = 0; c < 2; ++c) {
            const int sc2 = ch > 0 ? std::min(c, ch - 1) : 0;
            for (std::size_t i = 0; i < n; ++i) {
              const std::size_t k = a0 + i;
              d[(std::size_t)c * n + i] =
                  ch > 0 && k < total ? v->pcm[(std::size_t)sc2 * total + k]
                                      : 0.0f;
            }
          }
          FlexData sb = FlexData::make_object();
          auto so = sb.as_object();
          so.insert_or_assign(beat::kBatch, FlexData::make_bool(true));
          so.insert_or_assign("sample_rate", FlexData::make_int(_audio_rate));
          tb->sideband = std::move(sb);
          co_await ctx.write(3, std::move(tb));
        }
        // Its captions presented with its keyframes, each with the list.
        for (std::size_t i = 0; i < p.captions.size(); ++i) {
          if (i < p.caption_cached.size() && p.caption_cached[i]) { continue; }
          co_await ctx.write(1, caption_beat(p.captions[i], false));
          co_await ctx.write(4, keyframes_beat(keys));
        }
      }
      if (padded) {
        session()->warn(fmt("TrainingDatasetStage('{}'): '{}''s soundtrack "
                            "ends before its last window; the rest is "
                            "silence", this->id(), s.rel));
      }
      co_return;
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
      if (_show_keyframes) { co_await ctx.write(4, keyframes_beat({})); }
    }
    for (std::size_t i = 0; i < _preview.size(); ++i) {
      if (i < _preview_cached.size() && _preview_cached[i]) { continue; }
      co_await ctx.write(1, caption_beat(_preview[i], false));
      if (_show_keyframes) { co_await ctx.write(4, keyframes_beat({})); }
    }
  }
  // The decodes are over.
  if (_decode_booked > 0) {
    if (auto* mgr = session()->services()->generative_model_manager()) {
      mgr->revise_scratch(decode_label_(), 0);
    }
    _decode_booked = 0;
  }
  ctx.signal_done();
}

std::vector<ResourceClaim>
TrainingDatasetStage::declare_resources() const
{
  // Pictures decode one at a time (a picture's worth); a clip's windows
  // are booked by their size as each decodes.
  if (_media == "pictures") { return {}; }
  return model_memory::scratch_claims(decode_label_(),
                                      model_memory::kUnknownArena,
                                      model_memory::kPhaseCondition);
}

VPIPE_REGISTER_STAGE(TrainingDatasetStage)
VPIPE_REGISTER_SPEC(TrainingDatasetStage, kSpec)

}  // namespace vpipe
