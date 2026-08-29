#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/riffle-rows.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/shared/stream-sizing.h"
#include "generative-models/shared/streamed-refill.h"
#include "generative-models/shared/kernel-autotune.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "generative-models/generative-model-manager.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <cctype>
#include <fstream>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
namespace h3 = vpipe::genai::minimax_h3;

namespace {

using H3Route = MetalMiniMaxH3Transformer::GemmRoute;

// Which routes DEQUANTIZE the whole [N, K] weight into a dense bf16
// buffer and then run a matmul2d tile on it, as against the steel qmm
// routes that walk the quantized weight inline. Three places care --
// which arms the tuner keeps, whether a route's cost has an
// M-independent half, and whether a split-K is even possible -- and they
// have to agree, so they ask here.
bool
route_is_mma_(H3Route r)
{
  return r == H3Route::kMma128 || r == H3Route::kMma128x256 ||
         r == H3Route::kMma128x256Tn2;
}

// The tuner's row floors, and the ceiling they imply.
//
// kFlatTuneRows is about filling the machine and is generous for that:
// the grid is two-dimensional and N alone already supplies 42 tiles at
// the narrowest projection here. kMmaTuneRows is where the dequant has
// fallen to a few percent of the call -- see route_tune_rows_ for the
// ladder that was measured to land on it.
constexpr int kFlatTuneRows = 1024;
constexpr int kMmaTuneRows  = 4096;

// How far behind the leader a candidate may measure on the warm pass and
// still be timed properly. See the note in tune_qmm_ for why it is this
// loose -- the warm pass separates hopeless from plausible and nothing
// finer.
constexpr double kPruneKeep = 0.6;

// VPIPE_H3_TUNE_ROWS overrides both floors. Read at LOAD and kept on the
// model rather than in a process-wide static: a static would be fixed by
// whichever model in the process ran first, which is exactly the ordering
// a test cannot control.
int
tune_rows_override_()
{
  const char* e = std::getenv("VPIPE_H3_TUNE_ROWS");
  return e != nullptr ? std::atoi(e) : 0;
}

// The most rows any route asks to be timed over, in whole 128-row tiles.
//
// This is a CEILING on the measurement, and that makes it the natural
// cache key. tune_qmm_ times at min(M, ceiling), so two geometries at or
// above it run a LITERALLY IDENTICAL measurement -- same rows, same
// shapes, same candidates -- and keying the cache on the caller's own M
// made the second one pay for the first one's answer again. At the
// production geometry that was 16.5 s per distinct sequence length, for
// a result that could not differ.
//
// Below the ceiling the row count really is the caller's, so min() gives
// an exact key there and a shared one above.
int
tune_ceiling_()
{
  const int o = tune_rows_override_();
  const int m = o > 0 ? o : kMmaTuneRows;
  return ((m + 127) / 128) * 128;
}

// Namespace for this class's derived-tensor cache keys. A WeightSet is
// shared by everything reading one checkpoint, so the key has to name
// the transform, not just the tensor it came from.
constexpr const char* kKey = "minimax-h3-dit/bf16|";

// The Comfy-Org single-file DiT keeps the transformer config under
// this `__metadata__` key -- its presence is also what identifies
// the file as that conversion, and therefore as flat-qkv.
constexpr const char* kComfyKey = "config";

inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

inline float
bf16_to_f32_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// A checkpoint tensor as bf16, converted straight out of the source
// bytes. Deliberately NOT via an f32 staging vector: the largest tensor
// here is the 500 MB AdaLN projection, so the intermediate would be the
// peak.
SharedBuffer
to_bf16_(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm)
{
  const auto* info = wts.info(nm);
  if (info == nullptr || info->shape.empty()) { return {}; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { return {}; }
  if (info->dtype == "BF16") { return raw; }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  auto* d = static_cast<std::uint16_t*>(out.contents());
  if (info->dtype == "F32") {
    const auto* s = static_cast<const float*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_(s[i]); }
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_((float)s[i]); }
  } else {
    return {};
  }
  return out;
}

// A checkpoint tensor read straight to a host f32 vector. The timestep
// MLP runs on the host in f32 (see time_embed_), so its four tensors are
// CONSUMED by this conversion -- hence read(), not tensor(): caching
// them would keep a second copy of the bytes alive next to the vector.
bool
to_host_f32_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
             std::vector<float>& out)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr || info->shape.empty()) { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty()) { return false; }
  out.resize(n);
  if (info->dtype == "F32") {
    std::memcpy(out.data(), raw.contents(), n * 4);
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = bf16_to_f32_(s[i]); }
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { out[i] = (float)s[i]; }
  } else {
    return false;
  }
  return true;
}

// One scalar tensor as f32 (a kohya `alpha`). Empty when it is not a
// scalar this can read; the caller then treats the module as
// alpha == rank, which is the no-rescale case.
std::vector<float>
scalar_f32_(const MetalLlamaWeights& w, const std::string& name,
            const MetalLlamaWeights::TensorInfo* ti, MetalCompute* mc)
{
  std::size_t n = 1;
  for (auto d : ti->shape) { n *= (std::size_t)d; }
  if (n != 1) { return {}; }
  SharedBuffer b = w.load(name, mc);
  if (b.empty()) { return {}; }
  if (ti->dtype == "F32") {
    return {*static_cast<const float*>(b.contents())};
  }
  if (ti->dtype == "F16") {
    return {(float)*static_cast<const _Float16*>(b.contents())};
  }
  if (ti->dtype == "BF16") {
    return {bf16_to_f32_(*static_cast<const std::uint16_t*>(b.contents()))};
  }
  return {};
}

// Does this adapter touch any `mlp.fc1`?
//
// Asked by opening the header only -- no tensor is read -- because the
// answer decides how the BLOCKS are built, and that happens before there
// is a model to bind an adapter to. A file that cannot be opened answers
// "no": bind_lora_ runs later and reports the real error, and refusing
// the fusion on a file that turns out to be unreadable would be a
// slowdown chosen for a reason that never materialized.
bool
lora_touches_fc1_(const std::string& path)
{
  auto w = MetalLlamaWeights::open(path);
  if (!w.has_value()) { return false; }
  for (const std::string& n : w->tensor_names()) {
    if (n.find(".mlp.fc1.lora_") != std::string::npos) { return true; }
  }
  return false;
}

std::string
blk_(const char* prefix, int i, const char* rest)
{
  return std::string(prefix) + std::to_string(i) + "." + rest;
}

std::string
lower_(std::string v)
{
  for (char& c : v) { c = (char)std::tolower((unsigned char)c); }
  return v;
}

// Does the FILENAME of `file` carry `needle`? The partition is spelled
// in the name and nowhere else in a repack, so this is how a resolved
// component is checked against the partition that was asked for.
// Case-insensitive, and the DIRECTORY is excluded on purpose: a
// `.../Ref2VA/` parent says nothing about which file was picked out of
// it.
bool
filename_has_(const std::string& file, const std::string& needle)
{
  if (needle.empty()) { return true; }
  const std::string nm = lower_(std::filesystem::path(file).filename()
                                    .string());
  return nm.find(lower_(needle)) != std::string::npos;
}

}  // namespace

// ---- config ----------------------------------------------------------

std::string
MetalMiniMaxH3Transformer::partition_of(const std::string& path,
                                       const std::string& hint)
{
  namespace fs = std::filesystem;
  // A hint from the models DB is authoritative -- see the header. Two
  // records can share one directory exactly so that probing it cannot
  // answer, which is the case this exists for.
  if (!hint.empty()) { return hint; }
  const std::string dit = resolve_dit_dir(path, hint);
  auto lower = [](std::string v) {
    for (char& c : v) { c = (char)std::tolower((unsigned char)c); }
    return v;
  };
  // A repack says it in the filename and nowhere else -- the config it
  // embeds is identical for both. `pruned` is a different MODEL, not a
  // partition, so it is not answered for.
  if (!dit.empty() && !fs::is_directory(fs::path(dit))) {
    const std::string nm = lower(fs::path(dit).filename().string());
    if (nm.find("pruned") != std::string::npos) { return {}; }
    if (nm.find("fl2va") != std::string::npos)  { return "fl2va"; }
    if (nm.find("ref2va") != std::string::npos) { return "ref2va"; }
    return {};
  }
  // A DERIVED checkpoint -- model-quantize's output -- is a directory of
  // shards, so the filename that carried the partition is gone. The
  // producer records it in the config it writes, for the same reason it
  // records `qkv_per_head`: the two partitions are byte-identical in
  // every other respect, so an output that does not SAY which one it is
  // runs the wrong task at full cost and conditions on nothing.
  if (!dit.empty() && fs::is_directory(fs::path(dit))) {
    std::ifstream in(fs::path(dit) / "config.json");
    if (in) {
      try {
        FlexData fd = FlexData::from_json(in);
        if (fd.is_object() &&
            fd.as_object().contains(kPartitionKey)) {
          const FlexData v = fd.as_object().at(kPartitionKey);
          if (v.is_string()) { return std::string(v.as_string()); }
        }
      } catch (...) {
        // Fall through to the manifest walk below.
      }
    }
  }
  // A diffusers checkout says it in the pipeline manifest, which sits
  // beside the transformer directory (the partition root).
  fs::path base(dit.empty() ? path : dit);
  for (int up = 0; up < 3 && !base.empty(); ++up) {
    const fs::path mi = base / "model_index.json";
    std::error_code ec;
    if (fs::exists(mi, ec)) {
      std::ifstream in(mi);
      if (in) {
        try {
          FlexData fd = FlexData::from_json(in);
          if (fd.is_object() && fd.as_object().contains("_minimax_h3")) {
            FlexData mm = fd.as_object().at("_minimax_h3");
            if (mm.is_object() && mm.as_object().contains("partition")) {
              return std::string(mm.as_object().at("partition").as_string());
            }
          }
        } catch (...) {
          return {};
        }
      }
      return {};
    }
    base = base.parent_path();
  }
  return {};
}

std::string
MetalMiniMaxH3Transformer::partition_of_model_type(
    const std::string& model_type)
{
  if (model_type == "minimax-h3-fl2va")  { return "fl2va"; }
  if (model_type == "minimax-h3-ref2va") { return "ref2va"; }
  return {};
}

std::string
MetalMiniMaxH3Transformer::resolve_dit_dir(const std::string& path,
                                          const std::string& partition)
{
  namespace fs = std::filesystem;
  fs::path p(path);
  // A Comfy-Org single-file DiT, named either directly or through the
  // repo root / diffusion_models subdir. Probed FIRST so a repo that is
  // both (a Comfy-Org checkout inside a diffusers tree) resolves to the
  // self-describing file rather than to a config.json that would not
  // describe those bytes' qkv grouping.
  {
    // The asked-for partition, ALONE. The historical `fl2va` preference
    // stays for a caller that asks for nothing, so every graph written
    // before Ref2VA existed keeps resolving the way it did.
    //
    // It must not trail an explicit partition as a fallback, and the
    // check below is the other half of that: resolve_component RANKS by
    // `prefer` but still accepts a file matching none of it, so a repack
    // carrying only the OTHER half answers with it. Those bytes load
    // cleanly -- the two partitions share every name and shape -- and
    // condition on the wrong task, which is the failure this whole
    // partition argument exists to prevent. Resolving to nothing instead
    // fails at load with a missing-config error.
    const std::vector<std::string> prefer =
        partition.empty() ? std::vector<std::string>{"fl2va"}
                          : std::vector<std::string>{partition};
    const std::string f = comfy::resolve_component(path, "diffusion_models",
                                                   kComfyKey, prefer);
    if (!f.empty() && (partition.empty() || filename_has_(f, partition))) {
      return f;
    }
  }
  if (!fs::is_directory(p)) { return path; }
  // A QUANTIZED repack: model-quantize keeps the repack's role subdirs and
  // writes the component it quantized as a directory checkpoint inside its
  // own, so `diffusion_models/` holds config.json + shards rather than one
  // .safetensors. Probed after the repack file (a source repo has no
  // config.json there, so the two never both match) and before the
  // diffusers spellings below.
  if (fs::exists(p / "diffusion_models" / "config.json")) {
    return (p / "diffusion_models").string();
  }
  if (fs::exists(p / "config.json") && !fs::exists(p / "transformer")) {
    return p.string();                       // already the transformer dir
  }
  if (fs::exists(p / "transformer" / "config.json")) {
    return (p / "transformer").string();     // a partition root
  }
  // A repository root: the pipeline lives one level down in a partition
  // subdirectory, and MiniMaxAI publishes BOTH -- `FL2VA/` and `Ref2VA/`
  // are each a complete pipeline, distinguished by nothing but the
  // directory name and their own model_index.json (the transformer
  // configs are byte-identical).
  //
  // So `partition` picks, and neither is a fallback for the other: a
  // repo holding only the one that was not asked for resolves to nothing
  // and fails at load with a missing-config error, rather than running
  // the wrong task at full cost. A caller that asks for nothing PREFERS
  // the historical FL2VA and settles for whatever single partition the
  // checkout has.
  {
    const std::string want = lower_(partition);
    for (const char* part : {"FL2VA", "Ref2VA"}) {
      // Told: that one only. A checkout holding just the other half
      // resolves to nothing, which fails at load, where answering with
      // it would run the wrong task at full cost.
      if (!want.empty() && want != lower_(part)) { continue; }
      const fs::path sub = p / part / "transformer";
      if (fs::exists(sub / "config.json")) { return sub.string(); }
    }
    // Untold, the loop above has already preferred FL2VA and fallen
    // back to Ref2VA -- "whatever this checkout is" -- which is what the
    // repack path does with the same question, and what keeps a graph
    // written before Ref2VA existed resolving as it did.
  }
  return path;
}

MetalMiniMaxH3Transformer::GenerationParams
MetalMiniMaxH3Transformer::GenerationParams::from_flex(const FlexData& fd,
                                                       std::string* err)
{
  GenerationParams p;
  if (!fd.is_object()) {
    if (err != nullptr) { *err = "not a JSON object; using the defaults"; }
    return p;
  }
  auto o = fd.as_object();
  if (o.contains("video_shift")) {
    p.video_shift = o.at("video_shift").as_real(p.video_shift);
  }
  if (o.contains("audio_shift")) {
    p.audio_shift = o.at("audio_shift").as_real(p.audio_shift);
  }
  if (o.contains("condition_timestep")) {
    p.condition_timestep =
        o.at("condition_timestep").as_real(p.condition_timestep);
  }
  if (o.contains("condition_audio_timestep")) {
    p.condition_audio_timestep =
        o.at("condition_audio_timestep").as_real(p.condition_audio_timestep);
  }
  if (o.contains("audio_seconds")) {
    p.audio_seconds = o.at("audio_seconds").as_real(p.audio_seconds);
  }
  // A shift multiplies the schedule, so a non-positive one collapses it
  // to a single sigma and generates noise. The timesteps are levels in
  // t = 1 - sigma and live in [0, 1].
  if (!(p.video_shift > 0.0)) { p.video_shift = 12.0; }
  if (!(p.audio_shift > 0.0)) { p.audio_shift = 3.0; }
  auto clamp01 = [](double v) {
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
  };
  p.condition_timestep = clamp01(p.condition_timestep);
  p.condition_audio_timestep = clamp01(p.condition_audio_timestep);
  if (p.audio_seconds < 0.0) { p.audio_seconds = 0.0; }
  return p;
}

int
MetalMiniMaxH3Transformer::GenerationParams::audio_latents(int num_frames,
                                                           double fps) const
{
  if (audio_seconds > 0.0) {
    return (int)(audio_seconds * (double)minimax_h3::kAudioLatentsPerSecond +
                 0.5);
  }
  return minimax_h3::audio_latent_num_frames(num_frames, fps);
}

bool
MetalMiniMaxH3Transformer::config_from_json(const std::string& dit_dir,
                                            Config& out, std::string* err,
                                            const std::string& partition)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  fs::path p(resolve_dit_dir(dit_dir, partition));
  // Record WHICH partition this config describes, so the load() that
  // follows resolves the same file this was read from. Without it a
  // caller could read a Ref2VA config and then load the FL2VA weights
  // out of the same directory -- the configs are byte-identical, so
  // nothing downstream would notice.
  out.partition =
      partition.empty() ? partition_of(p.string()) : partition;
  const bool is_comfy = comfy::is_component(p.string(), kComfyKey);
  FlexData cfg;
  // ---- the Comfy-Org single file -------------------------------------
  // No config.json: the whole transformer config is a JSON string in the
  // safetensors `__metadata__`, under "config" -> "transformer". The
  // fields are the diffusers ones verbatim, so only the ENVELOPE differs
  // -- and the qkv grouping, which is why this branch exists at all.
  if (is_comfy) {
    FlexData md;
    std::string cerr;
    if (!comfy::metadata_json(p.string(), kComfyKey, md, &cerr)) {
      return fail(cerr);
    }
    if (!md.is_object() || !md.as_object().contains("transformer")) {
      return fail(p.string() + ": __metadata__ config has no 'transformer'");
    }
    cfg = md.as_object().at("transformer");
    if (!cfg.is_object()) {
      return fail(p.string() + ": 'transformer' is not an object");
    }
    // `image_model` is Comfy-Org's own architecture tag and stands in
    // for the diffusers `_class_name` this file does not carry. Refusing
    // an unknown one matters more here than in the directory case: a
    // sibling Comfy-Org repo has the same layout and the same metadata
    // key shape, so without this a Wan file would parse into an H3
    // config and load into the wrong model.
    // at() returns a FlexData BY VALUE and as_string() is a view into
    // it, so the entry has to be bound to a local first.
    auto co = cfg.as_object();
    const FlexData im_fd =
        co.contains("image_model") ? co.at("image_model") : FlexData();
    const std::string im(im_fd.as_string(""));
    if (im != "minimax_h3") {
      return fail("not a minimax_h3 checkpoint (image_model=" + im + ")");
    }
  } else {
    if (fs::is_directory(p)) { p = p / "config.json"; }
    std::ifstream f(p);
    if (!f) { return fail("cannot open " + p.string()); }
    try {
      cfg = FlexData::from_json(f);
    } catch (...) {
      return fail("cannot parse " + p.string());
    }
    if (!cfg.is_object()) {
      return fail(p.string() + " is not a JSON object");
    }
    auto co = cfg.as_object();
    const FlexData cls_fd =
        co.contains("_class_name") ? co.at("_class_name") : FlexData();
    const std::string cls(cls_fd.as_string(""));
    if (cls != "MiniMaxH3DiTModel") {
      return fail("not a MiniMaxH3DiTModel config (_class_name=" + cls + ")");
    }
  }
  auto o = cfg.as_object();
  // The bytes' qkv grouping follows the SOURCE, not the tensors -- both
  // layouts spell the projection identically. See Config::qkv_per_head.
  // An EXPLICIT `qkv_per_head` wins over the source, because a
  // checkpoint DERIVED from a Comfy-Org one (model-quantize's output)
  // is an ordinary directory that has copied the flat order through:
  // there is nothing left in its shape or its path to infer it from,
  // so the producer has to state it and this has to believe it.
  out.qkv_per_head = o.contains("qkv_per_head")
                         ? o.at("qkv_per_head").as_bool(true)
                         : !is_comfy;
  auto gi = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  auto gf = [&](const char* k, float d) {
    return o.contains(k) ? (float)o.at(k).as_real(d) : d;
  };
  out.hidden         = gi("hidden_size", 5376);
  out.n_heads        = gi("num_attention_heads", 56);
  out.head_dim       = gi("attention_head_dim", 128);
  out.n_layers       = gi("num_layers", 50);
  out.n_refiner      = gi("token_refiner_num_layers", 2);
  out.ffn            = gi("ffn_hidden_size", 14336);
  out.video_channels = gi("latents_dim", 24);
  out.audio_channels = gi("audio_latents_dim", 32);
  out.text_dim       = gi("text_dim", 5120);
  out.freq_dim       = gi("timestep_input_dim", 256);
  out.time_hidden    = gi("time_embed_hidden_size", 5376);
  out.time_dim       = gi("time_embed_dim", 2688);
  out.rope_freq_dim  = gi("rope_inv_freq_len", 16);
  out.norm_eps       = gf("norm_eps", 1e-5f);
  out.qk_norm_eps    = gf("qk_norm_eps", 1e-5f);
  out.final_norm_eps = gf("final_norm_eps", 1e-5f);
  if (o.contains("patch_size")) {
    FlexData ps = o.at("patch_size");
    auto arr = ps.as_array();
    if (arr.size() == 3) {
      out.patch_t = (int)arr[0].as_int(1);
      out.patch_h = (int)arr[1].as_int(2);
      out.patch_w = (int)arr[2].as_int(2);
    }
  }
  // The AdaLN width is the load-bearing consistency check: it has to be
  // exactly 6 modulation vectors x 3 modalities. If the checkpoint ever
  // grows a fourth modality, every row index in the packed layout shifts
  // and the model would still run -- reading the wrong table row for
  // every audio token.
  const int adaln = gi("adaln_out_features", out.adaln_out());
  if (adaln != out.adaln_out()) {
    return fail(fmt("adaln_out_features {} is not 6 * hidden {} * {} "
                    "modalities = {}", adaln, out.hidden,
                    h3::kModalityNum, out.adaln_out())());
  }
  const int fadaln = gi("final_adaln_out_features", 2 * out.hidden);
  if (fadaln != 2 * out.hidden) {
    return fail(fmt("final_adaln_out_features {} is not 2 * hidden {}",
                    fadaln, out.hidden)());
  }
  if (out.patch_t != 1) {
    return fail("only patch_size[0] == 1 (no temporal patching) is supported");
  }
  if (out.head_dim != 128) {
    return fail("only attention_head_dim 128 is supported (the steel "
                "flash-attention kernel this model runs on is bd128)");
  }
  if (out.rope_rot() > out.head_dim) {
    return fail("6 * rope_inv_freq_len exceeds attention_head_dim");
  }
  return true;
}

// ---- weight loading --------------------------------------------------

// Copied or Mapped for the bytes this model KEEPS, decided by the
// residency verdict rather than per call.
//
//   PRELOADING -> Mapped. Everything is resident either way, so owning
//     the bytes buys nothing and costs their KIND: clean file pages the
//     kernel drops for free become anonymous memory it can only
//     compress. load_mapped() wrapping the whole shard is the usual
//     objection to mapping and does not apply here -- the whole
//     checkpoint being resident is what preloading MEANS.
//   STREAMING  -> Copied, for the pinned prefix as well as the tail. A
//     prefix the kernel can evict is not a prefix, and a forward is a
//     cyclic scan: the one access pattern an LRU page cache handles
//     worst, dropping each block exactly before it comes round again.
//
//   PRELOADING WITH THE WIRED POOL ON -> Copied, which reverses the
//     first case and is the point of the pool. Mapped weights can be
//     neither wired, parked nor pooled: mlock refuses read-only file
//     mappings (MEASURED on the M4 Pro: ENOMEM after ~4 GB against a
//     49 GB pool and a 53.7 GB vm.user_wire_limit, while the same call
//     wires 12-15 GB of anonymous buffers), and for_each_weight()
//     enumerates cached COPIED entries only. So a mapped preload is
//     invisible to every mechanism in docs/MODEL-MEMORY.md except the
//     accounting -- 33 GB the model holds, the pool cannot protect and
//     a peer cannot reclaim.
//
// MEASURED on the sibling LTX-2.5 DiT at 960x544x121, preloaded on a
// 64 GB box: 409.7 s of denoise mapped against 473.5 s owning the same
// bytes, the copied arm at 36 GB compressed with its own weights 2-3%
// resident.
//
// READ THAT NUMBER WITH ITS OWN SECOND CLAUSE. The copied arm lost
// because it was BEING COMPRESSED -- 36 GB of it, 2-3% resident -- which
// is the exact failure wiring exists to prevent, and it was measured
// with no pool to wire into. It says owning bytes the compressor can
// take is worse than mapping them. It does not say owning bytes the
// compressor CANNOT take is worse, and that is the arm this enables.
//
// RE-MEASURED with the pool on, H3 at 960x544x21 / 4 steps on the M4
// Pro, arms INTERLEAVED (copied, mapped, copied, mapped) so a thermal
// drift cannot be read as a result: 186 / 174 s wired against 198 /
// 238 s mapped -- 1.21x, and the mapped arm's 40 s spread against the
// wired arm's 12 s is the page cache being unpredictable in exactly the
// way wiring removes. The box got HEALTHIER, not tighter: compression
// fell from 2060 to 991 MB across the run and swap fell with it, while
// 35 GB sat wired.
//
// A shard the file cannot map falls back to a copy regardless, so this
// is never worse than Copied -- but the log, not this rule, is what says
// whether a preload actually mapped.
static WeightSet::Residency
kept_residency_(bool stream_blocks, bool wire_resident)
{
  return (stream_blocks || wire_resident) ? WeightSet::Residency::Copied
                                          : WeightSet::Residency::Mapped;
}

// The budget bake_adaln checks its tables against. One definition, so
// the prediction below and the refusal there cannot drift apart.
static std::size_t
adaln_table_budget_()
{
  const char* e = std::getenv("VPIPE_H3_ADALN_BAKE_MAX_MB");
  const long v = (e != nullptr) ? std::atol(e) : 0;
  return (std::size_t)(v > 0 ? v : 1536) << 20;
}

std::size_t
MetalMiniMaxH3Transformer::streaming_floor_bytes(const std::string& dit_dir)
{
  auto wts = MetalLlamaWeights::open_model(dit_dir);
  if (!wts.has_value()) { return 0; }
  // AdaLN excluded: the bake retires it before the first forward, so a
  // slot never carries it for long and counting it would make the floor
  // 1.6x what the model settles at.
  return stream_floor_bytes(*wts, "blocks.", "adaln");
}

std::size_t
MetalMiniMaxH3Transformer::adaln_retired_bytes(const std::string& dit_dir)
{
  auto wts = MetalLlamaWeights::open_model(dit_dir);
  if (!wts.has_value()) { return 0; }
  std::size_t n = 0;
  for (const std::string& nm : wts->tensor_names()) {
    if (nm.find("adaln_proj") == std::string::npos) { continue; }
    const auto* ti = wts->info(nm);
    if (ti != nullptr) { n += (std::size_t)ti->nbytes; }
  }
  return n;
}

bool
MetalMiniMaxH3Transformer::adaln_bake_certain(const Config& cfg, int max_rows)
{
  if (max_rows <= 0) { return false; }
  if (std::getenv("VPIPE_H3_NO_ADALN_BAKE") != nullptr) { return false; }
  // The same arithmetic bake_adaln refuses on, with the caller's WORST
  // case for the row count. Certain only when even that fits.
  const std::size_t T = (std::size_t)max_rows;
  const std::size_t D = (std::size_t)cfg.adaln_out();
  const std::size_t FD = (std::size_t)(2 * cfg.hidden);
  const std::size_t want =
      T * D * 2 * (std::size_t)cfg.n_layers + T * FD * 2;
  return want <= adaln_table_budget_();
}

SharedBuffer
MetalMiniMaxH3Transformer::weight_(WeightSet& ws, const std::string& nm,
                                   Retain r)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr) { return {}; }
  if (info->dtype == "BF16") {
    // Already the forward's dtype, so the model keeps it AS IS; whose
    // memory it is follows kept_residency_ above.
    const auto res = kept_residency_(_stream_blocks, _wire_resident);
    return r == Retain::Streamed
               ? ws.stream_tensor(nm, _mc, res)
               : ws.tensor(nm, _mc, res);
  }
  // F32 in the checkpoint: a genuine transform the model then keeps, and
  // the source bytes are dropped -- which is exactly a derived entry.
  auto build = [&]() -> SharedBuffer { return to_bf16_(ws.src(), _mc, nm); };
  if (r == Retain::Streamed) { return ws.stream_derived(build); }
  return ws.derived(std::string(kKey) + nm, build);
}

MetalMiniMaxH3Transformer::Linear
MetalMiniMaxH3Transformer::linear_(WeightSet& ws, const std::string& nm,
                                   bool bias, Retain r)
{
  Linear l;
  if (bias) { l.b = weight_(ws, nm + ".bias", r); }
  const MetalLlamaWeights& src = ws.src();
  const auto* si = src.info(nm + ".scales");
  const auto* ci = src.info(nm + ".weight");
  if (_quant_bits > 0 && si != nullptr && ci != nullptr &&
      si->shape.size() == 2 && ci->shape.size() == 2) {
    // Per-WEIGHT bit width rather than the config's, so a mixed 4/8
    // checkpoint loads as-is: codes cols = K*bits/32 and scales cols =
    // K/group, so bits = codes_cols*32 / (scales_cols*group).
    const long gcols = ci->shape[1];
    const long scols = si->shape[1];
    const long K = scols * (long)_quant_group;
    const int bits = K > 0 ? (int)(gcols * 32 / K) : 0;
    l.bits = (bits == 8) ? 8 : 4;
    // The packed codes are the dominant bytes of a quantized pack, so
    // they take the same residency rule as everything else the model
    // KEEPS -- see kept_residency_(). Read separately from weight_()
    // only because they are raw u32 rather than the compute dtype.
    l.codes  = r == Retain::Streamed
                   ? ws.stream_tensor(nm + ".weight", _mc,
                                      kept_residency_(_stream_blocks,
                                                      _wire_resident))
                   : ws.tensor(nm + ".weight", _mc,
                               kept_residency_(_stream_blocks,
                                               _wire_resident));
    l.scales = weight_(ws, nm + ".scales", r);
    l.qbias  = weight_(ws, nm + ".biases", r);
    if (!l.codes.empty() && !l.scales.empty() && !l.qbias.empty()) {
      l.quantized = true;
      return l;
    }
    l.codes = {}; l.scales = {}; l.qbias = {};
  }
  l.w = weight_(ws, nm + ".weight", r);
  return l;
}

// fc1 in the interleaved gate/up layout, for a block the model KEEPS.
//
// Through derived() with read() sources, and that combination is the
// whole point of the function. A cached tensor stays cached whether or
// not this model still points at it, so transforming one would hold BOTH
// layouts -- 77 MB per block, 3.9 GB over the stack, four times what the
// fused path saves in scratch. read() does not cache, so only the layout
// the model runs on is retained, and derived() means two models over one
// checkpoint still share the one interleave.
//
// Falls back to the plain cached load whenever the checkpoint is not the
// shape this can permute; the caller then takes the split path, which is
// correct, just unfused.
MetalMiniMaxH3Transformer::Linear
MetalMiniMaxH3Transformer::fc1_gu_(WeightSet& ws, const std::string& nm)
{
  const MetalLlamaWeights& src = ws.src();
  const auto* si = src.info(nm + ".scales");
  const auto* ci = src.info(nm + ".weight");
  if (_quant_bits <= 0 || si == nullptr || ci == nullptr ||
      si->shape.size() != 2 || ci->shape.size() != 2) {
    return linear_(ws, nm, false, Retain::Cached);
  }
  const long N = ci->shape[0];
  const long K = si->shape[1] * (long)_quant_group;
  const int  b0 = K > 0 ? (int)(ci->shape[1] * 32 / K) : 0;
  const int  bits = (b0 == 8) ? 8 : 4;
  if (N != 2 * (long)_cfg.ffn || K != (long)_cfg.hidden) {
    return linear_(ws, nm, false, Retain::Cached);
  }
  const int half = (int)N / 2;
  // The key names the transform AND the width, because the bytes differ
  // between a w4 and a w8 checkpoint of the same tensor.
  const std::string key =
      "h3-fc1-gu/w" + std::to_string(bits) + "/g" +
      std::to_string(_quant_group) + "/";
  auto permute = [&](const char* suffix, std::size_t row_bytes) {
    return ws.derived(key + nm + suffix, [&]() -> SharedBuffer {
      // The SOURCE goes through the same conversion weight_() applies,
      // not through a raw read. This checkpoint stores `.scales` and
      // `.biases` as F16 while the forward is bf16, and the two are the
      // same WIDTH -- so a raw read permutes cleanly, matches every size
      // check here, and hands the kernel numbers off by an exponent
      // bias. That is what this cost the first time: the isolated kernel
      // agreed at 3.3e-3 and the real model came out at rel-L2 1.07.
      const auto* si2 = ws.src().info(nm + suffix);
      if (si2 == nullptr) { return {}; }
      SharedBuffer s =
          si2->dtype == "BF16" || si2->dtype == "U32"
              ? ws.read(nm + suffix, _mc, WeightSet::Residency::Copied)
              : to_bf16_(ws.src(), _mc, nm + suffix);
      if (s.empty() || s.byte_size() < (std::size_t)N * row_bytes) {
        return {};
      }
      SharedBuffer d = _mc->make_shared_buffer((std::size_t)N * row_bytes);
      if (d.empty()) { return {}; }
      const auto* sp = static_cast<const std::uint8_t*>(s.contents());
      auto* dp = static_cast<std::uint8_t*>(d.contents());
      for (int j = 0; j < half; ++j) {
        std::memcpy(dp + (std::size_t)(2 * j) * row_bytes,
                    sp + (std::size_t)j * row_bytes, row_bytes);
        std::memcpy(dp + (std::size_t)(2 * j + 1) * row_bytes,
                    sp + (std::size_t)(half + j) * row_bytes, row_bytes);
      }
      return d;
    });
  };
  Linear l;
  l.bits = bits;
  l.codes  = permute(".weight", (std::size_t)K * (std::size_t)bits / 8);
  l.scales = permute(".scales", (std::size_t)K / _quant_group * 2);
  l.qbias  = permute(".biases", (std::size_t)K / _quant_group * 2);
  if (l.codes.empty() || l.scales.empty() || l.qbias.empty()) {
    return linear_(ws, nm, false, Retain::Cached);
  }
  l.quantized = true;
  l.gu_inter  = true;
  return l;
}

// The same permutation for a block already in hand -- the streamed one
// being promoted to resident. Its buffers came from stream_tensor and are
// cached nowhere, so transforming them holds no second copy and there is
// nothing for derived() to share.
//
// fc1's [all gate | all up] rows -> the [g0,u0,g1,u1,...] pairing the
// fused epilogue reads.
//
// Rows are the OUTER dimension of the codes AND of the per-group scales
// and biases, so this is whole-row copies and no quantization group is
// ever split -- which is the whole reason the fusion can be a load-time
// transform instead of a kernel that understands two layouts.
//
// IN PLACE, through the shared cycle walk (shared/riffle-rows.h).
//
// It used to build three fresh buffers and swap them in, which cost a
// whole second copy of fc1 -- the largest weight in a block -- at the
// one moment a promotion is already asking the box for room. That made
// the interleave fail on memory, per block, which is precisely what
// made `gu_inter` vary between blocks and forced _ff_needs_wide to be
// decided globally.
//
// Writing through the buffers is safe HERE and would not be everywhere:
// this is only ever called on a block that promotion has just cloned
// into the resident set, out of stream_tensor sources that are cached
// nowhere. A weight held through tensor()/derived() is shared with
// every other holder of that checkpoint and must never be written
// (VPIPE_WEIGHT_INTEGRITY=1 catches it); nothing on this path is.
//
// The permutation is the same one FLUX.2's ff.linear_in wants -- gate
// rows and up rows woven into pairs -- which is why the walk is shared
// rather than written twice.
bool
MetalMiniMaxH3Transformer::interleave_gu_(Linear& l) const
{
  if (!l.quantized || l.gu_inter || l.codes.empty()) { return false; }
  const int N = 2 * _cfg.ffn, K = _cfg.hidden;
  const std::size_t row_codes = (std::size_t)K * (std::size_t)l.bits / 8;
  const std::size_t row_grp   = (std::size_t)K / (std::size_t)_quant_group;
  if (l.codes.byte_size() < (std::size_t)N * row_codes ||
      l.scales.byte_size() < (std::size_t)N * row_grp * 2 ||
      l.qbias.byte_size() < (std::size_t)N * row_grp * 2) {
    return false;
  }
  // All three checked before any is touched: they only mean something
  // together, and the walk cannot back out once it has started.
  const std::size_t n = (std::size_t)N;
  if (!genai::riffle_rows_ok(l.codes, n)
      || !genai::riffle_rows_ok(l.scales, n)
      || !genai::riffle_rows_ok(l.qbias, n)) {
    return false;
  }
  genai::riffle_rows(l.codes, n);
  genai::riffle_rows(l.scales, n);
  genai::riffle_rows(l.qbias, n);
  l.gu_inter = true;
  return true;
}

bool
MetalMiniMaxH3Transformer::clone_block_(const Block& src, Block& dst,
                                        bool copy_bytes) const
{
  bool ok = true;
  auto one = [&](const SharedBuffer& s, SharedBuffer& d) {
    if (!ok || s.empty()) { d = SharedBuffer{}; return; }
    d = _mc->make_shared_buffer(s.byte_size());
    if (d.empty()) { ok = false; return; }
    if (copy_bytes) {
      std::memcpy(d.contents(), s.contents(), s.byte_size());
    }
  };
  auto lin = [&](const Linear& s, Linear& d) {
    d.quantized = s.quantized;
    d.bits      = s.bits;
    d.gu_inter  = s.gu_inter;
    one(s.w, d.w);
    one(s.b, d.b);
    one(s.codes, d.codes);
    one(s.scales, d.scales);
    one(s.qbias, d.qbias);
  };
  one(src.n1, dst.n1);
  one(src.n2, dst.n2);
  one(src.qn, dst.qn);
  one(src.kn, dst.kn);
  lin(src.qkv, dst.qkv);
  lin(src.out, dst.out);
  lin(src.fc1, dst.fc1);
  lin(src.fc2, dst.fc2);
  lin(src.adaln, dst.adaln);
  if (!ok) { dst = Block{}; }
  return ok;
}

bool
MetalMiniMaxH3Transformer::refill_block_(WeightSet& ws,
                                         const std::string& prefix, Block& b,
                                         bool with_adaln)
{
  bool ok = true;
  int repaired = 0;
  // PER TENSOR, not per block. A tensor a raw read cannot place is
  // rebuilt the way load_block_ builds it, into the slot, and everything
  // around it still takes the fast path.
  //
  // This used to answer for the whole block, and any refusal anywhere
  // turned the slots off FOR THE RUN -- which sent every block of every
  // forward back through per-block allocations. That fallback is the
  // expensive one: it allocates a block's worth of buffers per block per
  // step, and before the reads were made uncached it also left the whole
  // checkpoint in the buffer cache, which is how a 24 GB box ends up
  // swapping without making progress. An old quantized pack whose blocks
  // are not byte-identical to block 0 is exactly the checkpoint that
  // triggered it, and it does not deserve the whole mechanism.
  //
  // `kUnservable` and `kFailed` are repaired the SAME way here, which
  // reads odd until you notice what the repair is: it does not top the
  // buffer up, it builds a new one and replaces it. That is the correct
  // response to a partly-written destination as well as to a dtype no
  // raw read can place -- and it is what makes a slot whose SIZE no
  // longer fits recover, since the replacement is sized from the
  // checkpoint.
  auto one = [&](const std::string& nm, SharedBuffer& dst, bool raw) {
    if (!ok) { return; }
    if (dst.empty()) { return; }          // this slot has no such tensor
    if (refill_streamed_tensor(ws, nm, dst, RefillDst::kBf16) ==
        Refill::kFilled) {
      return;
    }
    // Codes are the checkpoint's own u32 words and are copied; anything
    // else is read as bf16, which is what weight_() delivers.
    SharedBuffer rebuilt =
        raw ? ws.stream_tensor(
                  nm, _mc, kept_residency_(_stream_blocks, _wire_resident))
            : weight_(ws, nm, Retain::Streamed);
    if (rebuilt.empty()) { ok = false; return; }
    dst = std::move(rebuilt);
    ++repaired;
  };
  auto lin = [&](const std::string& nm, Linear& l) {
    if (!ok) { return; }
    // The one thing the per-tensor path cannot repair, because it is not
    // a tensor: a Linear whose quantized-ness or bit width has changed
    // under the slot. Rebuild the whole Linear so its buffers and its
    // metadata cannot disagree.
    if (!linear_matches_(ws, nm, l)) {
      Linear fresh = linear_(ws, nm, !l.b.empty(), Retain::Streamed);
      const bool built = fresh.quantized
                             ? !fresh.codes.empty()
                             : !fresh.w.empty();
      if (!built) { ok = false; return; }
      l = std::move(fresh);
      ++repaired;
      return;
    }
    if (l.quantized) {
      one(nm + ".weight", l.codes, /*raw=*/true);
      one(nm + ".scales", l.scales, /*raw=*/false);
      one(nm + ".biases", l.qbias, /*raw=*/false);
    } else {
      one(nm + ".weight", l.w, /*raw=*/false);
    }
    one(nm + ".bias", l.b, /*raw=*/false);
  };
  one(prefix + "norm1.weight", b.n1, /*raw=*/false);
  one(prefix + "norm2.weight", b.n2, /*raw=*/false);
  one(prefix + "attn.q_norm.weight", b.qn, /*raw=*/false);
  one(prefix + "attn.k_norm.weight", b.kn, /*raw=*/false);
  lin(prefix + "attn.qkv_proj", b.qkv);
  lin(prefix + "attn.out_proj", b.out);
  lin(prefix + "mlp.fc1", b.fc1);
  lin(prefix + "mlp.fc2", b.fc2);
  if (with_adaln) { lin(prefix + "adaln_proj.linear", b.adaln); }
  // Said once, and at debug: a checkpoint that repairs every block is
  // paying for it on every forward, and that is worth being able to see
  // without it turning into a mechanism that silently switched itself
  // off.
  if (repaired > 0 && !_refill_repaired && _mc->session() != nullptr) {
    _refill_repaired = true;
    _mc->session()->log_debug(fmt(
        "MetalMiniMaxH3Transformer: {} tensor(s) of a streamed block are "
        "not raw-readable into a slot and are rebuilt per forward; the "
        "rest still refill", repaired));
  }
  return ok;
}

bool
MetalMiniMaxH3Transformer::load_block_(WeightSet& ws,
                                       const std::string& prefix, Block& b,
                                       bool with_adaln, Retain r)
{
  b.n1  = weight_(ws, prefix + "norm1.weight", r);
  b.n2  = weight_(ws, prefix + "norm2.weight", r);
  b.qkv = linear_(ws, prefix + "attn.qkv_proj", false, r);
  b.out = linear_(ws, prefix + "attn.out_proj", false, r);
  b.qn  = weight_(ws, prefix + "attn.q_norm.weight", r);
  b.kn  = weight_(ws, prefix + "attn.k_norm.weight", r);
  // A block that is KEPT is built straight into the interleaved layout.
  // A STREAMED one is not: it is re-read every forward, so the
  // permutation would be paid 50 times a step and cost more than the
  // fusion saves. A streamed block later promoted to resident is
  // interleaved at the promotion, where it is paid once.
  b.fc1 = (_fuse_ff && r == Retain::Cached)
              ? fc1_gu_(ws, prefix + "mlp.fc1")
              : linear_(ws, prefix + "mlp.fc1", false, r);
  b.fc2 = linear_(ws, prefix + "mlp.fc2", false, r);
  if (with_adaln) {
    b.adaln = linear_(ws, prefix + "adaln_proj.linear", true, r);
    if (b.adaln.empty() || b.adaln.b.empty()) { return false; }
  }
  return !b.n1.empty() && !b.n2.empty() && !b.qkv.empty() && !b.out.empty() &&
         !b.qn.empty() && !b.kn.empty() && !b.fc1.empty() && !b.fc2.empty();
}

// y += B (A x), as two dense GEMMs through an [M, rank] scratch.
//
// Both factors are used in the checkpoint's own orientation -- A is
// [rank, K] and B is [N, rank] -- which is exactly the transposed-weight
// layout `dense_gemm_t_*` wants, so neither needs a transpose. The
// SECOND one accumulates: the base projection has already written y, and
// the delta lands on top of it in place. Encoded after the base GEMM into
// the same stream, so Metal's hazard tracking orders the two.
bool
MetalMiniMaxH3Transformer::lora_gemm_a_(ComputeEncoder& enc,
                                        const SharedBuffer& x,
                                        std::size_t x_off,
                                        const LoraFactors& lf, int M, int K,
                                        std::size_t lora_off)
{
  const int r = lf.rank;
  const bool mma = _use_mma2 && M >= _mma_min_m;
  const metal_compute::ComputeFunction* fa = mma ? lora_route_a_(r) : nullptr;
  // The strength rides HERE on the matrix-core route and on the second
  // GEMM on the steel one, because that is where each has room for it:
  // steel's epilogue is a scalar expression (acc * s + y) while
  // matmul2d's accumulate is a mode with nowhere to put a coefficient, so
  // the mma pair scales its smaller intermediate instead. Same product
  // either way; what must never happen is both, which is why this
  // reports back which one took it.
  enc.set_function(fa != nullptr ? *fa : _fn_gemm);
  enc.set_buffer(0, x, x_off * 2);
  enc.set_buffer(1, lf.a);
  enc.set_buffer(2, lf.a);          // bias slot unused (has_bias = 0)
  enc.set_buffer(3, _s.lora, lora_off * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, r);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  if (fa != nullptr) {
    enc.set_constant(8, _lora_scale);
    const int BN = (fa == &_fn_lora_a128) ? 128 : 64;
    const unsigned tw = (BN == 128) ? 256u : 128u;
    enc.dispatch({(unsigned)(((r + BN - 1) / BN) * tw),
                  (unsigned)((M + BN - 1) / BN), 1}, {tw, 1, 1});
    return true;
  }
  enc.dispatch({(unsigned)(((r + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
  return false;
}

void
MetalMiniMaxH3Transformer::lora_gemm_b_(ComputeEncoder& enc,
                                        const LoraFactors& lf,
                                        const SharedBuffer& y,
                                        std::size_t y_off, int M, int N,
                                        bool scale_applied,
                                        std::size_t lora_off)
{
  const int r = lf.rank;
  const bool mma = _use_mma2 && M >= _mma_min_m;
  // The accumulating tiles have no scale of their own, so this route is
  // legal only once the FIRST GEMM applied the strength.
  const metal_compute::ComputeFunction* fb =
      (mma && scale_applied) ? lora_route_b_(r, N) : nullptr;
  enc.set_function(fb != nullptr ? *fb : _fn_gemm_acc);
  enc.set_buffer(0, _s.lora, lora_off * 2);
  enc.set_buffer(1, lf.b);
  enc.set_buffer(2, lf.b);
  enc.set_buffer(3, y, y_off * 2);
  enc.set_constant(4, r);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.set_constant(8, scale_applied ? 1.0f : _lora_scale);
  if (fb != nullptr) {
    const int BN = (fb == &_fn_lora_b256) ? 256 : 128;
    enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    return;
  }
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
MetalMiniMaxH3Transformer::lora_apply_(ComputeEncoder& enc,
                                       const SharedBuffer& x,
                                       std::size_t x_off,
                                       const LoraFactors& lf,
                                       const SharedBuffer& y,
                                       std::size_t y_off, int M, int N, int K)
{
  if (lf.empty() || _s.lora.empty() || !_fn_gemm_acc.valid()) { return; }
  // Strength 0 is a legitimate request (an A/B against the un-adapted
  // model) and the cheapest way to serve it is not to encode the two
  // GEMMs at all -- which also makes "off" exactly off rather than a
  // pair of roundings that happen to cancel.
  if (_lora_scale == 0.0f) { return; }
  const bool scaled = lora_gemm_a_(enc, x, x_off, lf, M, K);
  lora_gemm_b_(enc, lf, y, y_off, M, N, scaled);
}

// Which tile runs t = x A^T. Its N is the RANK, so this is purely a rank
// question: the 64-wide tile fits rank 64 without half of it hanging past
// N, and the 128-wide one takes over as soon as there is a second tile's
// worth of rank to fill.
//
// MEASURED on M5 at H3's four projection shapes, 9382 rows
// (minimax_h3_blocks.lora_route_sweep, arms interleaved and all warmed
// before any is timed). ms for the four shapes SUMMED, steel -> best mma:
//
//   rank  64   11.47 -> 7.11 ms  (1.61x, 64-wide)
//   rank 128   21.67 -> 7.42 ms  (2.92x, 128-wide)
//   rank 384   64.75 -> 26.02 ms (2.49x, 128-wide)
//
// This half wins at EVERY rank and every shape, which is what makes it
// the unconditional part: the GEMM reads the whole [M, K] activation to
// produce [M, rank], so its ceiling is that one stream (~9.6 TFLOP/s at
// rank 64 on this machine's bandwidth) and steel's 32-wide output tile
// re-reads x once per tile to get nowhere near it.
//
// The 64/128 threshold is worth its own line because the wrong side of
// it gives most of the win back: at rank 128 the 64-wide tile totals
// 13.77 ms against the 128-wide tile's 7.42, and nearly all of that gap
// is fc2 alone (8.25 vs 3.31), whose K = ffn = 14336 is the deepest
// contraction here. A half-empty tile costs least where K is shallow.
const metal_compute::ComputeFunction*
MetalMiniMaxH3Transformer::lora_route_a_(int rank) const
{
  if (_lora_mma_off) { return nullptr; }
  const metal_compute::ComputeFunction* f =
      (rank <= 64) ? &_fn_lora_a64 : &_fn_lora_a128;
  if (!f->valid()) { f = &_fn_lora_a128; }
  return f->valid() ? f : nullptr;
}

// Which tile runs y += t B^T as its OWN dispatch -- or none, which means
// steel. Reached only where the fold in gemm_mma_ declined: an i8 or
// split-K base, a tn2 tile, a contraction past the fold's K ceiling, or
// no matrix cores at all. Where the fold applies there is no second
// dispatch to route.
//
// Here the contraction depth IS the rank, and at rank 64 that is too
// shallow for the matrix units to beat a kernel that is simply streaming
// y. This GEMM reads and writes the full [M, N] output, so at rank 64 its
// arithmetic intensity is r/2 = 32 flops/byte and steel already runs it
// at ~107 GB/s -- about 70% of what this machine has. There is no
// headroom for a different multiplier to find, and matmul2d's wider tiles
// re-read the operands enough to LOSE:
//
//   rank  64  qkv  steel 7.51 ms | mma128 13.49 | mma256 12.30
//   rank  64  fc1  steel 10.61   | mma128 19.77 | mma256 19.69
//   rank 128  qkv  steel 14.01   | mma128 13.11 | mma256  7.89
//   rank 384  qkv  steel 40.93   | mma128 11.88 | mma256 13.98
//
// so rank 64 stays on steel and rank >= 128 goes to matmul2d.
//
// Not taken, and measured: at rank 64 the two NARROW-N shapes (o and
// fc2, N = 5376) do give matmul2d a 6-9% edge -- it is only the wide
// ones that lose, and they lose by 1.8x. A rule with N in it would
// collect that, and it is left out on purpose: 6-9% of the second GEMM
// at half the shapes is 0.3 ms in a block that spends 694, and the
// second dimension would have to be re-measured on every future GPU
// alongside the first. The tile
// then INVERTS with rank -- 256-wide leads at 128 by 1.17-1.66x over the
// four shapes, 128-wide leads at 384 by 1.15-1.19x -- consistently enough
// at both ranks and all four shapes to be a rank effect rather than
// noise. Where between 128 and 384 it crosses is NOT measured; 256 is the
// midpoint and the only ranks that ship today are 64, 128 and the stacked
// 384, so nothing in the tree currently lands on the guess.
const metal_compute::ComputeFunction*
MetalMiniMaxH3Transformer::lora_route_b_(int rank, int N) const
{
  (void)N;   // shape-independent so far: all four projections agreed.
  if (rank < 128) { return nullptr; }
  const metal_compute::ComputeFunction* f =
      (rank >= 256) ? &_fn_lora_b128 : &_fn_lora_b256;
  return f->valid() ? f : nullptr;
}

// Read the adapter and bind its factors to the modules they name.
//
// The module names are the MODEL's own -- blocks.N.attn.qkv_proj,
// mlp.fc1/fc2, adaln_proj.linear, token_refiner.blocks.N.* -- so this is
// a lookup rather than a remap. A module the checkpoint does not carry is
// SKIPPED and counted, not an error: an adapter trained on a subset of
// the projections is normal, and refusing one would be refusing the
// common case.
bool
MetalMiniMaxH3Transformer::bind_lora_(const LoraSpec& spec, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "minimax-h3 lora: " + m; }
    return false;
  };
  auto w = MetalLlamaWeights::open(spec.path);
  if (!w.has_value()) { return fail("cannot open " + spec.path); }
  _lora_scale = spec.scale;

  // One factor, converted to bf16 and (for A) pre-scaled -- folding the
  // strength into A here keeps the kernels scale-free and costs one pass
  // over the smaller of the two matrices.
  auto take = [&](const std::string& name, float mul) -> SharedBuffer {
    const auto* ti = w->info(name);
    if (ti == nullptr || ti->shape.size() != 2) { return {}; }
    std::size_t n = (std::size_t)ti->shape[0] * (std::size_t)ti->shape[1];
    SharedBuffer src = w->load(name, _mc);
    if (src.empty()) { return {}; }
    SharedBuffer dst = _mc->make_shared_buffer(n * 2);
    if (dst.empty()) { return {}; }
    auto* d = static_cast<std::uint16_t*>(dst.contents());
    if (ti->dtype == "BF16") {
      const auto* p = static_cast<const std::uint16_t*>(src.contents());
      for (std::size_t i = 0; i < n; ++i) {
        d[i] = mul == 1.0f ? p[i] : f32_to_bf16_(bf16_to_f32_(p[i]) * mul);
      }
    } else if (ti->dtype == "F16") {
      const auto* p = static_cast<const _Float16*>(src.contents());
      for (std::size_t i = 0; i < n; ++i) {
        d[i] = f32_to_bf16_((float)p[i] * mul);
      }
    } else if (ti->dtype == "F32") {
      const auto* p = static_cast<const float*>(src.contents());
      for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_(p[i] * mul); }
    } else {
      return {};
    }
    return dst;
  };

  // Publishers disagree about one thing only: whether the module names
  // carry a container prefix. ComfyUI-convention adapters key on
  // `diffusion_model.<module>` where the model's own names have no
  // prefix at all, and the shapes and the ORDER are otherwise identical
  // -- so this is a lookup with two spellings rather than a conversion.
  // An adapter that needs more than a prefix (diffusers' split
  // to_q/to_k/to_v, its value-first ff.net.0.proj) is NOT one of these
  // and must not be coerced into looking like one.
  static const char* const kPrefixes[] = {"", "diffusion_model."};

  int skipped = 0;
  auto bind = [&](const std::string& mod, int N, int K, LoraFactors& out) {
    std::string key;
    const MetalLlamaWeights::TensorInfo* ai = nullptr;
    const MetalLlamaWeights::TensorInfo* bi = nullptr;
    for (const char* pre : kPrefixes) {
      const std::string k = std::string(pre) + mod;
      const auto* a = w->info(k + ".lora_A.weight");
      const auto* b = w->info(k + ".lora_B.weight");
      if (a != nullptr && b != nullptr) { key = k; ai = a; bi = b; break; }
    }
    if (ai == nullptr || bi == nullptr) { return; }
    if (ai->shape.size() != 2 || bi->shape.size() != 2 ||
        ai->shape[1] != K || bi->shape[0] != N ||
        ai->shape[0] != bi->shape[1]) {
      // A shape that does not fit is the one thing worth counting
      // separately from absence: it means the adapter was trained
      // against a DIFFERENT model, and applying the parts that happen to
      // fit would be worse than applying none.
      ++skipped;
      return;
    }
    out.rank = (int)ai->shape[0];
    // kohya / ai-toolkit ship a per-module `alpha` and the update is
    // scaled by alpha/rank. Absent (diffusers, and larryvrh's adapter)
    // means the factors are already at strength, which is the same thing
    // as alpha == rank. This is a property of the FILE, so it folds into
    // A once, here. The caller's strength does NOT: it is a per-forward
    // constant, so it can be turned without rebuilding 33B of weights.
    float mul = 1.0f;
    if (const auto* al = w->info(key + ".alpha")) {
      std::vector<float> v = scalar_f32_(*w, key + ".alpha", al, _mc);
      if (!v.empty() && out.rank > 0) { mul *= v[0] / (float)out.rank; }
    }
    out.a = take(key + ".lora_A.weight", mul);
    out.b = take(key + ".lora_B.weight", 1.0f);
    if (out.empty()) { out = LoraFactors{}; ++skipped; return; }
    _lora_max_rank = std::max(_lora_max_rank, out.rank);
    ++_lora_modules;
  };

  const Config& c = _cfg;
  const int H = c.hidden, I = c.inner(), F = c.ffn;
  _lora_blocks.resize((std::size_t)c.n_layers);
  for (int i = 0; i < c.n_layers; ++i) {
    BlockLora& bl = _lora_blocks[(std::size_t)i];
    const std::string p = blk_("blocks.", i, "");
    bind(p + "attn.qkv_proj", 3 * I, H, bl.qkv);
    bind(p + "attn.out_proj", H, I, bl.out);
    bind(p + "mlp.fc1", 2 * F, H, bl.fc1);
    bind(p + "mlp.fc2", H, F, bl.fc2);
    bind(p + "adaln_proj.linear", c.adaln_out(), c.time_dim, bl.adaln);
  }
  _lora_refiner.resize((std::size_t)c.n_refiner);
  for (int i = 0; i < c.n_refiner; ++i) {
    BlockLora& bl = _lora_refiner[(std::size_t)i];
    const std::string p = blk_("token_refiner.blocks.", i, "");
    bind(p + "attn.qkv_proj", 3 * I, H, bl.qkv);
    bind(p + "attn.out_proj", H, I, bl.out);
    bind(p + "mlp.fc1", 2 * F, H, bl.fc1);
    bind(p + "mlp.fc2", H, F, bl.fc2);
  }
  bind("final_layer.adaln_proj.linear", 2 * H, c.time_dim, _lora_final);

  if (_lora_modules == 0) {
    return fail("'" + spec.path + "' adapts none of this model's modules");
  }
  if (_mc->session() != nullptr) {
    _mc->session()->log_normal(fmt(
        "MetalMiniMaxH3Transformer: runtime LoRA '{}' -- {} modules at "
        "scale {}, rank <= {}{}",
        spec.path, _lora_modules, _lora_scale, _lora_max_rank,
        skipped > 0 ? fmt(", {} SKIPPED (shape mismatch)", skipped)()
                    : std::string()));
  }
  return true;
}

MetalMiniMaxH3Transformer::~MetalMiniMaxH3Transformer()
{
  // GIVE THE POOL BACK. Freeing a wired buffer unwires it in the kernel,
  // so the machine recovers either way -- but the pool's own counter
  // would not, and a DiT that is destroyed after every clip (the
  // ordinary `unload_when_idle: destroy` path) would leak its whole
  // share of the budget per clip until nothing could wire at all.
  if (_wire_resident) {
    wire_fixed_(false);
    for (Block& b : _blocks) { wire_block_(b, false); }
    _slot[0] = Block{};
    _slot[1] = Block{};
  }
}

std::unique_ptr<MetalMiniMaxH3Transformer>
MetalMiniMaxH3Transformer::load(const std::string& dit_dir, MetalCompute* mc,
                                const Config& cfg, bool stream_blocks,
                                const LoraSpec* lora)
{
  return load(WeightSet::open(resolve_dit_dir(dit_dir, cfg.partition),
                             nullptr),
              mc, cfg, stream_blocks, lora);
}

std::unique_ptr<MetalMiniMaxH3Transformer>
MetalMiniMaxH3Transformer::load(std::shared_ptr<WeightSet> ws_in,
                                MetalCompute* mc, const Config& cfg,
                                bool stream_blocks,
                                const LoraSpec* lora)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  WeightSet& ws = *ws_in;
  auto m = std::unique_ptr<MetalMiniMaxH3Transformer>(
      new MetalMiniMaxH3Transformer());
  m->_ws = std::move(ws_in);
  m->_mc = mc;
  m->_cfg = cfg;
  // Honored HERE rather than only in the stage so the tests and the
  // offline tools can force either mode: streaming has to be provably
  // the same function as preloading, and that is checked by running one
  // golden both ways.
  if (const char* e = std::getenv("VPIPE_H3_STREAM")) {
    stream_blocks = (std::atoi(e) != 0);
  }
  m->_stream_blocks = stream_blocks;
  // Everything loaded HERE is kept for the model's life, so it is cached
  // whichever mode we are in. Only the 50 main blocks stream, and they do
  // it in forward().
  const Retain r = Retain::Cached;

  // BF16 metallibs, reached by name with the SAME *_f16 entry points.
  // bf16 for the reason every DiT here runs bf16: f16's 65504 ceiling is
  // inside the range this residual stream reaches, and the reference is
  // bf16 anyway.
  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_fn_gemm      = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  // The accumulating twin, for a runtime LoRA's second factor. Optional:
  // a build without it simply cannot attach one, which bind_lora_ reports
  // rather than silently applying half an adapter.
  m->_fn_gemm_acc  = m->_lib_gemm.function("dense_gemm_t_bm64_acc_f16");
  m->_fn_rms       = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_rms_heads = m->_lib_rope.function("rms_norm_heads_strided_f16");
  m->_fn_trope = m->_lib_rope.function("transpose_rope_half_part_ftab_f16");
  // The transpose-free twin. Optional: a build without it simply keeps
  // the transposing path, which is correct and only slower.
  m->_fn_rope_ip = m->_lib_rope.function("rope_half_part_ftab_inplace_f16");
  m->_fn_modulate  = m->_lib_elt.function("adaln_modulate_idx_f16");
  m->_fn_gated     = m->_lib_elt.function("gated_residual_idx_f16");
  // The fused fc1 is [GATE | up] and the block computes silu(gate)*up.
  // Measured against a ComfyUI golden: gate-first 0.0058, value-first
  // 0.2524. A wrong choice here is silent -- a plausible, wrong
  // activation in every block. The video VAE's decoder is the same.
  //
  // The two references LOOK like they disagree and do not.
  //
  // diffusers implements [value; gate] -- its generic `SwiGLU` chunks
  // the fused projection and returns `value * silu(gate)` -- and it
  // never sees these bytes, because
  // `scripts/convert_minimax_h3_to_diffusers.py` SWAPS the two halves as
  // it writes `mlp.fc1` out as `ff.net.0.proj`. Upstream states the raw
  // layout there in as many words: "the reference computes
  // fc2(silu(gate) * value) from a fused [gate; value]; diffusers'
  // SwiGLU computes value * silu(gate) from a fused [value; gate], so
  // the two halves swap places."
  //
  // So a diffusers-FORMAT checkpoint is value-first, the raw MiniMaxAI
  // release is gate-first, and both are right about their own bytes.
  // Re-checked on diffusers main 175fe6b2 (2026-08-12): unchanged, and
  // nothing to report upstream. What the raw layout is was never in
  // dispute; only our reading of a file that describes the CONVERTED
  // one was.
  m->_fn_swiglu = m->_lib_elt.function("swiglu_split_gate_first_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_residual  = m->_lib_elt.function("residual_add_f16");
  m->_fn_bias_add  = m->_lib_elt.function("bias_add_rows_f16");
  // The blit a baked AdaLN needs when an adapter has to be added on top.
  m->_fn_copy      = m->_lib_elt.function("copy_f16");
  m->_fn_nan_trip  = m->_lib_elt.function("nan_tripwire_f16");
  {
    metal_compute::ComputeLibrary sdpa = mc->load_library("sdpa_bf16");
    m->_fn_sdpa = sdpa.function("sdpa_full_f16");
  }
  if (!m->_fn_gemm.valid() || !m->_fn_rms.valid() ||
      !m->_fn_rms_heads.valid() || !m->_fn_trope.valid() ||
      !m->_fn_modulate.valid() || !m->_fn_gated.valid() ||
      !m->_fn_swiglu.valid() || !m->_fn_transpose.valid() ||
      !m->_fn_bias_add.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_residual.valid()) {
    return nullptr;
  }
  // Steel flash-attention. The scalar sdpa_full_f16 is O(seq^2) at a few
  // percent of peak and this model's sequences are video-sized, so the
  // fallback is a correctness A/B rather than a path anyone should run.
  m->_lib_attn = mc->load_library("attn_steel");
  m->_attn_p_main = mc->make_shared_buffer(sizeof(float) * 64);
  m->_attn_p_text = mc->make_shared_buffer(sizeof(float) * 64);
  m->_steel_ok = m->_lib_attn.valid() && !m->_attn_p_main.empty() &&
                 !m->_attn_p_text.empty() && cfg.head_dim == 128 &&
                 std::getenv("VPIPE_H3_NO_STEEL_ATTN") == nullptr;
  // On M5 the same flash attention runs on the matrix units
  // (attn_steel_nax, bq=64/bk=32, matmul2d QK^T and P*V with the online
  // softmax register-resident) instead of the simdgroup ALU kernel
  // (bq=32/bk=16). IDENTICAL AttnParams, function-constant and
  // threadgroup contract -- only the tile sizes and the kernel differ --
  // which is why this is a function swap and not a second code path.
  // head_dim 128 + bf16 is exactly the entry FLUX.2 and Krea-2 already
  // use. VPIPE_H3_NO_ATTN_NAX forces the ALU kernel (A/B).
  if (m->_steel_ok && mc->supports_matrix_cores() &&
      std::getenv("VPIPE_H3_NO_ATTN_NAX") == nullptr) {
    m->_lib_attn_nax = mc->load_library("attn_steel_nax");
    m->_attn_nax = m->_lib_attn_nax.valid();
  }
  // FUSED ATTENTION: let steel address the activations where they
  // already are instead of transposing four [seq, inner] buffers per
  // block into and out of a head-major layout it never required. See
  // the strides in fill() below, and the header for what each bit costs.
  //
  // The default is the OUT bit alone: the q/k/v half was measured and
  // loses. VPIPE_H3_FUSED_ATTN overrides with an explicit mask (0 for
  // the old path, 3 for both).
  {
    int want = kFusedAttnOut;
    if (const char* e = std::getenv("VPIPE_H3_FUSED_ATTN")) {
      if (*e != '\0') { want = std::atoi(e); }
    }
    m->_fused_attn = 0;
    if (m->_steel_ok) {
      m->_fused_attn = want & kFusedAttnOut;
      // Steel only -- the scalar sdpa_full_f16 fallback reads contiguous
      // [H, seq, D] and has no stride to give it. The head-major windows
      // therefore stay allocated: the fallback is what runs if the steel
      // functions fail to build, and it has to have somewhere to work.
      if ((want & kFusedAttnQkv) != 0 && m->_fn_rope_ip.valid()) {
        m->_fused_attn |= kFusedAttnQkv;
      }
    }
  }

  // Quantization block from the checkpoint's own config.json.
  {
    std::ifstream qin(std::filesystem::path(ws.dir()) / "config.json");
    if (qin) {
      FlexData fd;
      try {
        fd = FlexData::from_json(qin);
      } catch (...) {
        fd = FlexData::make_null();
      }
      if (fd.is_object()) {
        auto o = fd.as_object();
        if (o.contains("quantization")) {
          FlexData q = o.at("quantization");
          if (q.is_object()) {
            auto qo = q.as_object();
            const int b = qo.contains("bits") ? (int)qo.at("bits").as_int(0) : 0;
            if (b == 4 || b == 8) { m->_quant_bits = b; }
            if (qo.contains("group_size")) {
              m->_quant_group = (int)qo.at("group_size").as_int(64);
            }
          }
        }
      }
    }
  }
  if (m->_quant_bits > 0) {
    const std::string g = "g" + std::to_string(m->_quant_group);
    m->_lib_qmm = mc->load_library("affine_qmm_steel_bf16");
    m->_fn_qmm4 = m->_lib_qmm.function("affine_qmm_steel_w4" + g);
    m->_fn_qmm8 = m->_lib_qmm.function("affine_qmm_steel_w8" + g);
    if (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid()) { return nullptr; }
    m->_qmm_tile = 0;
    m->_fn_qmm4_bm64 = m->_lib_qmm.function("affine_qmm_steel_w4" + g + "_bm64");
    m->_fn_qmm8_bm64 = m->_lib_qmm.function("affine_qmm_steel_w8" + g + "_bm64");
    if (m->_fn_qmm4_bm64.valid() && m->_fn_qmm8_bm64.valid()) {
      m->_qmm_tile = 1;
      m->_fn_qmm4_bm128 =
          m->_lib_qmm.function("affine_qmm_steel_w4" + g + "_bm128");
      m->_fn_qmm8_bm128 =
          m->_lib_qmm.function("affine_qmm_steel_w8" + g + "_bm128");
      if (m->_fn_qmm4_bm128.valid() && m->_fn_qmm8_bm128.valid()) {
        m->_qmm_tile = 2;
      }
    }
    if (const char* t = std::getenv("VPIPE_H3_QMM_TILE")) {
      m->_qmm_tile = std::min(m->_qmm_tile, std::atoi(t));
    }
    // The fused-SwiGLU FF. Group 64 only -- that is the width the fused
    // entry points are built for, and a g32 checkpoint has no g32 twin to
    // fall back to. Decided BEFORE any block loads, because load_block_
    // interleaves fc1 when it is set.
    if (m->_quant_group == 64 &&
        std::getenv("VPIPE_H3_NO_FUSED_FF") == nullptr) {
      m->_fn_qmm_swiglu4 = m->_lib_qmm.function("affine_qmm_swiglu_w4g64");
      m->_fn_qmm_swiglu8 = m->_lib_qmm.function("affine_qmm_swiglu_w8g64");
      m->_fn_qmm_swiglu4_bm64 =
          m->_lib_qmm.function("affine_qmm_swiglu_w4g64_bm64");
      m->_fn_qmm_swiglu8_bm64 =
          m->_lib_qmm.function("affine_qmm_swiglu_w8g64_bm64");
      m->_fuse_ff = m->_fn_qmm_swiglu4.valid() && m->_fn_qmm_swiglu8.valid();
    }
    // A runtime LoRA on `mlp.fc1` and the fused SwiGLU are mutually
    // exclusive: the fused epilogue writes silu(gate)*up straight out of
    // the accumulator, so there is no [rows, 2*ffn] pre-activation for a
    // delta to be added to. This has to be settled HERE, before any block
    // loads -- load_block_ interleaves fc1's rows when the fusion is on,
    // and an interleaved weight cannot then take the split path.
    if (m->_fuse_ff && lora != nullptr && lora_touches_fc1_(lora->path)) {
      m->_fuse_ff = false;
      if (mc->session() != nullptr) {
        mc->session()->log_debug(fmt(
            "MetalMiniMaxH3Transformer: fused SwiGLU off -- the LoRA adapts "
            "mlp.fc1, whose delta has to land before the activation"));
      }
    }
  }

  // ---- M5 matrix cores: dequant-once -> dense matmul2d ------------------
  // The block GEMMs are 71% of a step and run at 83-86% of this GPU's ALU
  // ceiling, so on a machine WITHOUT matrix cores there is nothing left to
  // win there. On M5 there is: matmul2d is a different pipe, and the four
  // DiTs already on it (FLUX.2, Krea-2, Boogu, Qwen-Image-Edit) measure
  // 2.4-3.3x over the steel quantized GEMM at these shapes.
  //
  // The design is theirs and is NOT obvious: dequantize the whole weight
  // ONCE into a bf16 scratch and run a plain dense matmul2d on it, rather
  // than dequantizing tiles inside a fused quantized kernel. Fused loses
  // (0.5-0.6x) because manual threadgroup staging defeats matmul2d's
  // internal pipelining -- it wants to drive the tiled loop from device
  // memory itself. The dequant is one pass over the weight against M rows
  // of GEMM, so it amortizes at exactly the M where the tile does.
  //
  // Gated on the capability, never on a model name: supports_matrix_cores()
  // is supportsFamily(Apple10), i.e. "M5 or newer". VPIPE_H3_NO_MMA2 forces
  // steel for the A/B.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_H3_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_fn_dense_mma_tn2 =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");
    m->_fn_lora_a64 =
        m->_lib_dense_mma.function("dense_gemm_mma_t_scaled_f16");
    m->_fn_lora_a128 =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128_scaled_f16");
    m->_fn_lora_b128 =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128_acc_f16");
    m->_fn_lora_b256 =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_acc_f16");
    m->_fn_lora_fused128 =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128_lora_f16");
    m->_fn_lora_fused256 =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_lora_f16");
    // Forces the ADAPTER back onto the steel pair while leaving the base
    // projections on the matrix cores. That separation is the point:
    // VPIPE_H3_NO_MMA2 moves the base too, so the difference it produces
    // is not the adapter's. Read at load rather than per encode so two
    // models in ONE process can take different routes -- which is what
    // minimax_h3_lora.runtime_adapter_routes_agree needs.
    m->_lora_mma_off = std::getenv("VPIPE_H3_NO_LORA_MMA") != nullptr;
    // Keeps the adapter on the matrix cores but encodes its second GEMM
    // as its own dispatch, which is the A/B for the FOLD alone.
    m->_lora_fuse_off = std::getenv("VPIPE_H3_NO_LORA_FUSE") != nullptr;
    if (const char* e = std::getenv("VPIPE_H3_LORA_FUSE_MAX_K")) {
      m->_lora_fuse_max_k = std::atoi(e);
    }
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid();
    // A quantized checkpoint additionally needs the dequant kernels; a
    // dense one feeds the same tiles its bf16 weight directly.
    if (m->_use_mma2 && m->_quant_bits > 0) {
      const std::string g = "g" + std::to_string(m->_quant_group);
      m->_lib_dequant = mc->load_library("affine_dequant_bf16");
      m->_fn_dequant4 = m->_lib_dequant.function("affine_dequant_w4" + g);
      m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant_w8" + g);
      if (!m->_fn_dequant4.valid() || !m->_fn_dequant8.valid()) {
        m->_use_mma2 = false;   // stay on steel qmm
      }
    }
    if (const char* e = std::getenv("VPIPE_H3_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    // Dynamic-int8 accelerated GEMMs (opt-in, LOSSY), tried first in
    // gemm_mma_ once the dense weight exists. bf16 kernels: this DiT's
    // activations, dequant scratch and output are all bf16.
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm, /*bf16=*/true);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
    // Deep-K split for the FF down projection. bf16 libraries: the fold
    // writes this DiT's element type.
    m->_tune_ceiling = tune_ceiling_();
    m->_splitk.load(mc, m->_lib_dense_mma, m->_lib_elt);
    // The split is tuned at the same ceiling the route is, so two
    // geometries above it share one answer instead of re-measuring it.
    m->_splitk.m_bucket = m->_tune_ceiling;
  }
  // The matrix-core path takes precedence and keeps the FF it has. See
  // the _fuse_ff comment in the header for why this is a refusal rather
  // than a second arm: folding the epilogue is exactly what matmul2d
  // cannot do, so fusing there costs the interleaved weight's bytes and
  // returns nothing. Both routes are decided by now, so this is the one
  // place that can see the conflict.
  if (m->_use_mma2) { m->_fuse_ff = false; }

  m->_video_patch = m->linear_(ws, "video_patch_proj", true, r);
  m->_audio_patch = m->linear_(ws, "audio_patch_proj", true, r);
  m->_cond_proj   = m->linear_(ws, "condition_proj", true, r);
  m->_final_norm  = m->weight_(ws, "final_layer.norm.weight", r);
  m->_final_adaln = m->linear_(ws, "final_layer.adaln_proj.linear", true, r);
  m->_video_out   = m->linear_(ws, "final_layer.video_out", true, r);
  m->_audio_out   = m->linear_(ws, "final_layer.audio_out", true, r);
  m->_refiner_final_norm = m->weight_(ws, "token_refiner.final_norm.weight", r);
  if (m->_video_patch.empty() || m->_audio_patch.empty() ||
      m->_cond_proj.empty() || m->_final_norm.empty() ||
      m->_final_adaln.empty() || m->_video_out.empty() ||
      m->_audio_out.empty() || m->_refiner_final_norm.empty()) {
    return nullptr;
  }

  // The timestep MLP goes to the HOST in f32 -- see time_embed_.
  if (!to_host_f32_(ws, mc, "time_embedder.proj_in.weight", m->_time_in_w) ||
      !to_host_f32_(ws, mc, "time_embedder.proj_in.bias", m->_time_in_b) ||
      !to_host_f32_(ws, mc, "time_embedder.proj_out.weight", m->_time_out_w) ||
      !to_host_f32_(ws, mc, "time_embedder.proj_out.bias", m->_time_out_b)) {
    return nullptr;
  }
  // rope.inv_freq is LOADED rather than recomputed from a theta. The
  // config carries no rope_theta at all, so recomputing it would mean
  // guessing the base -- and a wrong base is a wrong model that still
  // produces smooth video.
  if (!to_host_f32_(ws, mc, "rope.inv_freq", m->_inv_freq) ||
      (int)m->_inv_freq.size() != cfg.rope_freq_dim) {
    return nullptr;
  }

  m->_refiner.resize((std::size_t)cfg.n_refiner);
  for (int i = 0; i < cfg.n_refiner; ++i) {
    if (!m->load_block_(ws, blk_("token_refiner.blocks.", i, ""),
                        m->_refiner[(std::size_t)i], false, r)) {
      return nullptr;
    }
  }
  // WHETHER THIS MODEL'S PAGES GO IN THE WIRED POOL.
  //
  // Restored here after the prefix retirement deleted it by accident:
  // the assignment lived inside the block that sized the pinned prefix,
  // that block went, and the five places that READ these two fields
  // stayed. The effect was silent and total -- `_wire_resident` is
  // false-initialised, so wire_fixed_() never ran, no block was ever
  // wired, and the only visible trace was "wire budget 0 MB" in the
  // residency probe line.
  //
  // From the MANAGER now rather than from this model's own percentage.
  // The old form asked what share of RAM one DiT could wire, which is
  // the per-model budget the pool exists to replace: a ceiling each
  // model guesses separately never adds up to the box. The pool is the
  // one accounting, and wire_into_pool() is what enforces it -- this
  // flag only says whether to ask.
  //
  // `_wire_budget` is what is still UNUSED, since the probe below asks
  // how many more blocks can be wired, not how large the pool is.
  {
    const auto* sess = mc != nullptr ? mc->session() : nullptr;
    auto* mgr = sess != nullptr && sess->services() != nullptr
                    ? sess->services()->generative_model_manager()
                    : nullptr;
    const std::size_t lim = mgr != nullptr ? mgr->wired_pool_limit() : 0;
    if (lim > 0) {
      const std::size_t used = mgr->wired_pool_used();
      m->_wire_resident = true;
      m->_wire_budget = lim > used ? lim - used : 0;
    }
    if (const char* e = std::getenv("VPIPE_WIRE_RESIDENT")) {
      m->_wire_resident = std::atoi(e) != 0;
    }
  }
  // The 50 main blocks: preloaded (default), or -- streaming -- only the
  // pinned prefix, with forward() reading and freeing the tail per block.
  if (!stream_blocks) {
    m->_blocks.resize((std::size_t)cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
      if (!m->load_block_(ws, blk_("blocks.", i, ""),
                          m->_blocks[(std::size_t)i], true, r)) {
        return nullptr;
      }
    }
    // NOT WIRED HERE, AND NOT BECAUSE IT CANNOT BE.
    //
    // With the pool on these blocks are Copied (kept_residency_), so
    // they are cached entries of the weight set -- which means
    // wire_fixed_() already reaches them through for_each_weight(), and
    // it runs on the first forward where the SCRATCH exists. That
    // ordering is the reason to leave it there: the scratch goes into
    // the pool first, then the trunk and the blocks, so a pool that
    // runs out runs out on the blocks, which are the half a forward can
    // proceed without.
    //
    // Wiring them here instead inverts exactly that. MEASURED when this
    // was tried at the load site: 4315 MB of blocks wired, the pool
    // collapsed on the first refusal, and NOTHING was left for the
    // trunk and the scratch -- against ~17 GB wired and both protected
    // when the forward does it.
  } else {
    // No prefix: growth fills these as the box allows. See the note on
    // load() -- a fraction of total RAM decided before the run cannot
    // see the machine, and BlockResidency measures instead.
    m->_blocks.resize((std::size_t)cfg.n_layers);
  }
  if (lora != nullptr && !lora->path.empty()) {
    std::string lerr;
    if (!m->bind_lora_(*lora, &lerr)) {
      if (mc->session() != nullptr) { mc->session()->error(fmt("{}", lerr)); }
      return nullptr;
    }
  }
  // The ff scratch can only shrink when NOTHING can present an unfused
  // fc1: streaming can, at any step, and so can a block whose interleave
  // failed to allocate. Asked of the blocks that exist rather than
  // assumed from `_fuse_ff`, because the interleave is allowed to fail.
  m->_ff_needs_wide = !m->_fuse_ff || stream_blocks;
  if (!m->_ff_needs_wide) {
    for (const Block& b : m->_refiner) {
      if (!b.fc1.gu_inter) { m->_ff_needs_wide = true; }
    }
    for (const Block& b : m->_blocks) {
      if (!b.fc1.gu_inter) { m->_ff_needs_wide = true; }
    }
  }
  if (mc->session() != nullptr) {
    mc->session()->log_debug(
        fmt("MetalMiniMaxH3Transformer: {} blocks + {} refiner, hidden {}, "
            "attn inner {}, {}{}",
            cfg.n_layers, cfg.n_refiner, cfg.hidden, cfg.inner(),
            m->_quant_bits > 0
                ? fmt("w{} g{}", m->_quant_bits, m->_quant_group)()
                : std::string("bf16"),
            m->_fuse_ff
                ? fmt(", fused SwiGLU FF ({} ff scratch)",
                      m->_ff_needs_wide ? "wide" : "narrow")()
                : std::string()));
  }
  return m;
}

std::vector<int>
MetalMiniMaxH3Transformer::tune_prune_survivors(
    const std::vector<double>& warm, double keep)
{
  std::vector<int> live;
  const std::size_t n = warm.size();
  // A warm pass that failed to time is not evidence. Neither is a field
  // of two, where there is nothing to prune down to.
  bool timed = n > 2;
  for (double t : warm) { timed = timed && t > 0.0; }
  if (!timed) {
    for (std::size_t i = 0; i < n; ++i) { live.push_back((int)i); }
    return live;
  }
  // Fastest and second-fastest. BOTH start at infinity, so that a list
  // already in ascending order still finds a real second -- seeded from
  // warm[0] it would not, and {1, 2, 3} would keep one arm instead of
  // the two this rule promises.
  double lo = std::numeric_limits<double>::infinity(), lo2 = lo;
  for (double t : warm) {
    if (t < lo)       { lo2 = lo; lo = t; }
    else if (t < lo2) { lo2 = t; }
  }
  for (std::size_t i = 0; i < n; ++i) {
    // The two fastest unconditionally, then anything the warm pass puts
    // within `keep` of the leader.
    if (warm[i] <= lo2 || lo >= keep * warm[i]) { live.push_back((int)i); }
  }
  return live;
}

void
MetalMiniMaxH3Transformer::set_fused_attention(int mask)
{
  // Bits the build cannot honour are dropped rather than faked, so an
  // A/B that reads fused_attention() back sees which arm it ran.
  int want = 0;
  if (_steel_ok) {
    want = mask & kFusedAttnOut;
    if ((mask & kFusedAttnQkv) != 0 && _fn_rope_ip.valid()) {
      want |= kFusedAttnQkv;
    }
  }
  _fused_attn = want;
  // The cached AttnParams may describe another layout now. The pipeline
  // states do not depend on the layout and are left alone.
}

void
MetalMiniMaxH3Transformer::set_qmm_tile(int cap)
{
  const int built = (_fn_qmm4_bm128.valid() && _fn_qmm8_bm128.valid()) ? 2
                    : (_fn_qmm4_bm64.valid() && _fn_qmm8_bm64.valid()) ? 1
                                                                       : 0;
  if (cap < 0) { cap = 0; }
  _qmm_tile = std::min(cap, built);
  // Setting the cap by hand means the caller is driving the choice --
  // an in-process A/B, or a bisect. Stop tuning for this model's life
  // and drop what was already measured, or the tuned answer would keep
  // overriding the cap and every arm of the A/B would run the same
  // kernel while appearing to test three.
  _qmm_manual = true;
  _qmm_tuned.clear();
  _qmm_tuning_desc.clear();
}

void
MetalMiniMaxH3Transformer::set_gemm_route(GemmRoute r)
{
  _forced_route = r;
  // kAuto hands the choice back to the tuner; anything else is the caller
  // driving, for the same reason and with the same consequence as
  // set_qmm_tile -- see the note there, and the header.
  _qmm_manual = (r != GemmRoute::kAuto);
  _qmm_tuned.clear();
  _qmm_tuning_desc.clear();
}

// One quantized GEMM at an EXPLICIT tile height. Split out of gemm_ so
// the autotuner can time the very dispatch the forward will run --
// benching an approximation of it is how a tuner ends up confidently
// picking the wrong kernel.
void
MetalMiniMaxH3Transformer::qmm_dispatch_(ComputeEncoder& enc,
                                         const SharedBuffer& x,
                                         std::size_t x_off, const Linear& l,
                                         const SharedBuffer& y,
                                         std::size_t y_off,
                                         int M, int N, int K, int bm)
{
  enc.set_function(
      l.bits == 8
          ? (bm == 128 ? _fn_qmm8_bm128 : bm == 64 ? _fn_qmm8_bm64 : _fn_qmm8)
          : (bm == 128 ? _fn_qmm4_bm128 : bm == 64 ? _fn_qmm4_bm64
                                                   : _fn_qmm4));
  enc.set_buffer(0, l.codes);
  enc.set_buffer(1, l.scales);
  enc.set_buffer(2, l.qbias);
  enc.set_buffer(3, x, x_off * 2);
  enc.set_buffer(4, y, y_off * 2);
  enc.set_constant(5, K);
  enc.set_constant(6, N);
  enc.set_constant(7, M);
  const unsigned tgz = (bm == 128) ? 4u : 2u;
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + bm - 1) / bm) * 2), tgz}, {32, 2, tgz});
}

// The tile height for one quantized GEMM.
//
// MEASURED (M4 Pro, w4g64, alternating the tiles inside one process --
// across processes this box has ~4-5% of thermal spread, which is wider
// than the effect):
//
//     rows     BM32      BM64      BM128
//     4282    1.000x    1.049x     1.050x
//     9382    1.000x    1.042x     0.991x
//
// So BM64 is the win, and BM128 is a LOSS at the model's own geometry --
// slower than the shortest tile. A staircase that reached for BM128 at
// M >= 8192 (which is what shipped, copied from the Wan DiT's shape)
// therefore cost ~5% at exactly the size this model is built for. That
// is the argument for measuring rather than legislating: the crossover
// is a property of the machine and of the (N, K) of each projection, and
// nothing about the checkpoint predicts it.
//
// `tune_qmm_` fills _qmm_tuned for the sequence length actually running;
// until it has (or when it is disabled, or the shape is not one it
// tuned) this falls back to the measured staircase.
//
// On M5 the candidate list also carries the matrix-core tiles, and the
// same argument applies with more force: NOTHING in this file knows where
// dequant-once + matmul2d overtakes the steel quantized GEMM, because it
// depends on M, on (N, K), and on a GPU generation this code predates.
const char*
MetalMiniMaxH3Transformer::route_name_(GemmRoute r)
{
  switch (r) {
    case GemmRoute::kSteelBm32:     return "bm32";
    case GemmRoute::kSteelBm64:     return "bm64";
    case GemmRoute::kSteelBm128:    return "bm128";
    case GemmRoute::kMma128:        return "mma128";
    case GemmRoute::kMma128x256:    return "mma128x256";
    case GemmRoute::kMma128x256Tn2: return "mma128x256tn2";
    case GemmRoute::kAuto:          break;
  }
  return "auto";
}

bool
MetalMiniMaxH3Transformer::route_ok_(GemmRoute r, int M, int N, int K) const
{
  switch (r) {
    case GemmRoute::kSteelBm32:
      return true;
    case GemmRoute::kSteelBm64:
      return _qmm_tile >= 1;
    case GemmRoute::kSteelBm128:
      return _qmm_tile >= 2;
    case GemmRoute::kMma128:
    case GemmRoute::kMma128x256:
    case GemmRoute::kMma128x256Tn2: {
      // N >= 16 keeps a degenerate projector off a 128-wide tile; M is the
      // real gate, since the dequant pass is paid whatever M is.
      if (!_use_mma2 || M < _mma_min_m || N < 16) { return false; }
      // The dequant kernel's contract, which nothing else checks: it walks
      // one thread per packed u32 word (K/8 words per row at w4, K/4 at w8)
      // and reads one scale per group, so a K that is not a whole number of
      // both would silently under-write the tail of every row. Every H3
      // projection is a multiple of 64, so this never fires today -- it is
      // here so that a future checkpoint with an odd width falls back to
      // steel instead of producing quiet garbage.
      if (_quant_bits > 0) {
        const int per_word = (_quant_bits == 8) ? 4 : 8;
        if (K % per_word != 0 || K % _quant_group != 0) { return false; }
      }
      // The dequant writes a full [N, K] bf16 weight. At the released
      // config the widest is fc1 (28672 x 5376 = 308 MB) -- worth knowing
      // on a 16 GB box, and the reason the scratch is grown once and kept
      // rather than sized per projection.
      if (r == GemmRoute::kMma128x256Tn2) { return _fn_dense_mma_tn2.valid(); }
      return true;
    }
    case GemmRoute::kAuto:
      break;
  }
  return false;
}

MetalMiniMaxH3Transformer::GemmRoute
MetalMiniMaxH3Transformer::gemm_route_(int M, int N, int K) const
{
  if (_forced_route != GemmRoute::kAuto) {
    // A forced route that cannot run this shape falls back rather than
    // failing, so a bench can ask for mma unconditionally and still get a
    // correct (if slower) answer on an M4.
    if (route_ok_(_forced_route, M, N, K)) { return _forced_route; }
  } else {
    const int key = tune_row_key(M);
    for (const QmmTuneSet& set : _qmm_tuned) {
      if (set.m != key) { continue; }
      for (const QmmTune& t : set.shapes) {
        if (t.N == N && t.K == K && route_ok_(t.route, M, N, K)) {
          return t.route;
        }
      }
      break;
    }
  }
  // Untuned fallback (VPIPE_H3_NO_QMM_AUTOTUNE, or a shape the tuner did
  // not cover). MEASURED on M5 at this model's four real shapes, w4, over
  // a 602..19008 row ladder -- minimax_h3_blocks.gemm_route_sweep, arms
  // interleaved and all warmed before any is timed:
  //
  //   * matmul2d beats steel by 1.7-4.0x at EVERY shape and EVERY row
  //     count. Steel sits flat at 2.2-3.5 TFLOP/s whichever tile it uses
  //     (bm64 is 1.0-1.19x bm32, bm128 never wins), so on this GPU the
  //     steel tile that the M4 tuner argues about is a rounding error next
  //     to the question of whether to use the matrix units at all.
  //   * fc2 (N=5376, K=14336) is the one shape with a structural tile
  //     answer: the narrow 128x128 tile collapses to 1.70-2.71x where the
  //     wide one holds 2.36-3.40x. That is the deep-K/narrow-N cliff the
  //     Krea-2, Boogu and OptiQ audits all hit, and it reproduced in both
  //     runs at every row count. Hence the K-only wide-tile rule, which is
  //     shared with the LM families (kMmaDeepK).
  //
  // What is NOT decided here is which mma tile wins for the other three
  // shapes. Between two runs of the same sweep the winner changed and the
  // absolute rates moved 14-17% -- the tn2 tile led every shape at 9382
  // rows in the short run and lost three of four in the long one, which
  // heated the box. Those tiles are within ~5-10% of each other, i.e.
  // inside this machine's power-budget spread, so a constant fitted to
  // either run would be fitted to noise. The tuner picks among them at the
  // sequence actually running, which is the only place that comparison is
  // fair; the fallback just takes the shape-driven answer and stops.
  //
  // The tn2 tile is deliberately never GUESSED: its win is shape-specific
  // and it is the one tile that can address a sub-tile past N (safe now --
  // dense_gemm_mma_impl guards it -- but not something to reach for blind).
  if (route_ok_(GemmRoute::kMma128, M, N, K)) {
    return mma_use_wide_tile(N, K) ? GemmRoute::kMma128x256
                                   : GemmRoute::kMma128;
  }
  if (_qmm_tile >= 1 && M >= 4096) { return GemmRoute::kSteelBm64; }
  return GemmRoute::kSteelBm32;
}

// One projection through `route`, WITHOUT the bias: neither the steel qmm
// nor the dense matmul2d tiles have a bias epilogue, so gemm_ adds it as a
// separate row-broadcast pass. (The plain dense GEMM does have one, and is
// the only arm that applies it here.)
//
// This is the single dispatcher for all four combinations of
// dense/quantized weight and steel/matmul2d route, and the autotune calls
// it too -- so what the tuner measures is by construction what the forward
// runs.
void
MetalMiniMaxH3Transformer::gemm_route_dispatch_(
    ComputeEncoder& enc, const SharedBuffer& x, std::size_t x_off,
    const Linear& l, const SharedBuffer& y, std::size_t y_off,
    int M, int N, int K, GemmRoute route, const LoraFactors* lora,
    bool* lora_folded, std::size_t lora_off)
{
  if (gemm_mma_(enc, x, x_off, l, y, y_off, M, N, K, route, lora,
                lora_folded, lora_off)) {
    return;
  }
  if (l.quantized) {
    const int bm = (route == GemmRoute::kSteelBm128) ? 128
                 : (route == GemmRoute::kSteelBm64)  ? 64
                                                     : 32;
    qmm_dispatch_(enc, x, x_off, l, y, y_off, M, N, K, bm);
    return;
  }
  // Dense weight on a non-matmul2d route. has_bias stays 0 for the same
  // reason the other arms have none: the caller owns the bias, so all four
  // combinations reach gemm_'s single bias pass.
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x, x_off * 2);
  enc.set_buffer(1, l.w);
  enc.set_buffer(2, l.w);
  enc.set_buffer(3, y, y_off * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

// Dequant-once into _w_deq, then one dense matmul2d tile.
//
// The scratch is SHARED across a block's projections and reused across
// blocks. That is safe because the dequant/matmul pairs are encoded in
// order into one command stream and Metal's hazard tracking serializes a
// write-after-read on the same buffer -- each matmul reads _w_deq before
// the next dequant overwrites it. It is also the only affordable choice:
// a per-projection scratch would be ~850 MB of the four shapes at once.
bool
MetalMiniMaxH3Transformer::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                                     std::size_t x_off, const Linear& l,
                                     const SharedBuffer& y, std::size_t y_off,
                                     int M, int N, int K, GemmRoute route,
                                     const LoraFactors* lora,
                                     bool* lora_folded, std::size_t lora_off)
{
  // Only the mma routes belong here, and this test has to come FIRST.
  // route_ok_ answers "can this route run this shape", and for a STEEL
  // route the answer is always yes -- so guarding with route_ok_ alone
  // let every steel route fall into the matrix-core path and run the
  // 128x128 tile anyway. Two consequences, both silent because the two
  // paths agree numerically: the steel arms of the autotune measured mma
  // (so the tuner compared one kernel against itself), and a tiny GEMM
  // took the mma path regardless of _mma_min_m -- the M=2 AdaLN
  // projection dequantized 520 MB to multiply two rows. Caught by
  // scratch_estimate_matches_allocation, which noticed the dequant
  // scratch existing at a geometry where it should not.
  if (route != GemmRoute::kMma128 && route != GemmRoute::kMma128x256 &&
      route != GemmRoute::kMma128x256Tn2) {
    return false;
  }
  if (!route_ok_(route, M, N, K)) { return false; }
  const SharedBuffer* wdense = nullptr;
  if (l.quantized) {
    const metal_compute::ComputeFunction& dq =
        (l.bits == 8) ? _fn_dequant8 : _fn_dequant4;
    if (!dq.valid()) { return false; }
    const std::size_t need = (std::size_t)N * (std::size_t)K * 2;
    if (_w_deq.empty() || _w_deq.byte_size() < need) {
      _w_deq = _mc->make_shared_buffer(need);
      if (_w_deq.empty()) { return false; }   // stay on steel
    }
    // One thread per packed u32 word: w4 packs 8 nibbles per word (K/8
    // words per row), w8 packs 4 bytes (K/4).
    enc.set_function(dq);
    enc.set_buffer(0, l.codes);
    enc.set_buffer(1, l.scales);
    enc.set_buffer(2, l.qbias);
    enc.set_buffer(3, _w_deq);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    const unsigned words = (unsigned)(l.bits == 8 ? (K / 4) : (K / 8));
    enc.dispatch({words, (unsigned)N, 1}, {64, 1, 1});
    wdense = &_w_deq;
  } else {
    wdense = &l.w;
  }
  // Accelerated mode: int8 activations x int8 weight, quantized per call.
  // Takes the same dense weight the matmul2d tile would have read, so it
  // sits after the dequant and before the tile choice; a shape it does not
  // accept falls through to bf16 with nothing encoded.
  if (_i8 && _i8->gemm(enc, x, x_off, *wdense, y, y_off, M, N, K)) {
    return true;
  }
  // Deep contraction (fc2, K = ffn): split it across planes and fold. Same
  // dense weight, so it sits beside the i8 arm and before the single-op
  // tile; declines on every other projection here, whose K is hidden or
  // inner and far short of the floor.
  if (_splitk.encode(_mc, enc, x, *wdense, y, K, N, M, x_off, y_off)) {
    return true;
  }
  // The adapter's second factor, folded into this tile. Deliberately the
  // LAST thing tried: i8 and split-K are worth more on the BASE than the
  // fold is worth on the delta (1.35x and ~2x against a delta that is a
  // few percent of the projection), so a shape either of them accepts
  // keeps them and pays for a separate second GEMM. Only the two plain
  // tiles have a fused twin -- tn2's software TN grid would need the
  // delta sliced per sub-tile, and split-K would have to add it to
  // exactly one plane.
  //
  // The K CEILING is not caution, it is a measured cliff. Carrying a
  // second contraction in the tile costs something that scales with the
  // BASE contraction's depth, not with the delta's -- which is not where
  // one would look for it, since the delta is 64 deep either way.
  // MEASURED at 9382 rows, rank 64, ms added to the projection by the
  // adapter's second GEMM (minimax_h3_blocks.lora_fused_rate):
  //
  //   qkv  N=21504 K= 5376   separate +7.62   fused +3.46   (2.2x)
  //   fc1  N=28672 K= 5376   separate +10.31  fused +3.21   (3.2x)
  //   o    N= 5376 K= 7168   separate +2.15   fused free
  //   fc2  N= 5376 K=14336   separate +0.59   fused +56.17  (a 39% LOSS
  //                                                    on the projection)
  //
  // fc2 also happens to be split-K's shape, so in the tuned model the
  // fold never reaches it -- but "another rule gets there first" is not a
  // guarantee, and split-K declines below its own row floor. So the gate
  // is explicit. 8192 sits just past the deepest K measured GOOD rather
  // than midway to the bad one, because the two sides are not
  // symmetrical: declining a fold costs a few ms, taking a bad one costs
  // 39% of a projection. Where it actually crosses between 7168 and
  // 14336 is unmeasured; VPIPE_H3_LORA_FUSE_MAX_K re-probes it.
  if (lora != nullptr && lora_folded != nullptr && !_lora_fuse_off &&
      K <= _lora_fuse_max_k &&
      (route == GemmRoute::kMma128 || route == GemmRoute::kMma128x256)) {
    const metal_compute::ComputeFunction* ff =
        (route == GemmRoute::kMma128x256) ? &_fn_lora_fused256
                                          : &_fn_lora_fused128;
    if (ff->valid()) {
      const int FRN = (route == GemmRoute::kMma128x256) ? 256 : 128;
      enc.set_function(*ff);
      enc.set_buffer(0, x, x_off * 2);
      enc.set_buffer(1, *wdense);
      enc.set_buffer(2, *wdense);    // bias slot, unread (has_bias = 0)
      enc.set_buffer(3, y, y_off * 2);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.set_buffer(8, _s.lora, lora_off * 2);
      enc.set_buffer(9, lora->b);
      enc.set_constant(10, lora->rank);
      enc.dispatch({(unsigned)(((N + FRN - 1) / FRN) * 256),
                    (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
      *lora_folded = true;
      return true;
    }
  }
  // RN is the N-region one threadgroup owns (TN*BN), which is what the
  // grid divides N by -- 512 for the tn2 tile, else BN.
  int RN = 128;
  const metal_compute::ComputeFunction* fn = &_fn_dense_mma;
  if (route == GemmRoute::kMma128x256) {
    fn = &_fn_dense_mma_deep; RN = 256;
  } else if (route == GemmRoute::kMma128x256Tn2) {
    fn = &_fn_dense_mma_tn2; RN = 512;
  }
  enc.set_function(*fn);
  enc.set_buffer(0, x, x_off * 2);
  enc.set_buffer(1, *wdense);
  enc.set_buffer(2, *wdense);      // bias slot, unread (has_bias = 0)
  enc.set_buffer(3, y, y_off * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  return true;
}

std::string
MetalMiniMaxH3Transformer::qmm_tuning() const
{
  return _qmm_tuning_desc;
}

void
MetalMiniMaxH3Transformer::dispatch_row_bands_(ComputeEncoder& enc,
                                               const SharedBuffer& x,
                                               std::size_t x_off,
                                               const Linear& l,
                                               const SharedBuffer& y,
                                               std::size_t y_off, int M,
                                               int N, int K, GemmRoute route)
{
  const int band = mma_row_band(N, K);
  for (int m0 = 0; m0 < M; m0 += band) {
    gemm_route_dispatch_(enc, x, x_off + (std::size_t)m0 * K, l, y,
                         y_off + (std::size_t)m0 * N,
                         std::min(band, M - m0), N, K, route, nullptr,
                         nullptr);
  }
}

int
MetalMiniMaxH3Transformer::mma_row_band(int N, int K)
{
  static const int kForced = [] {
    const char* e = std::getenv("VPIPE_H3_MMA_ROW_BAND");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  if (kForced > 0) { return kForced; }
  // The rule itself is shared: the Wan VAE reaches the same line from the
  // other side (a fixed row cap that bounds its destination and not its
  // 27-tap im2col source), and two families disagreeing about where 2^31
  // is would be one of them silently wrong.
  return ::vpipe::genai::mma_row_band(N, K);
}

void
MetalMiniMaxH3Transformer::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                                 std::size_t x_off, const Linear& l,
                                 const SharedBuffer& y, std::size_t y_off,
                                 int M, int N, int K, const LoraFactors* lora)
{
  const bool bias = !l.b.empty();
  // An adapter that cannot or should not run is dropped HERE, so nothing
  // downstream has to re-check it. Strength 0 is a legitimate request and
  // the cheapest way to serve it is to encode nothing, which also makes
  // "off" exactly off rather than a pair of roundings that cancel.
  if (lora != nullptr &&
      (lora->empty() || _s.lora.empty() || !_fn_gemm_acc.valid() ||
       _lora_scale == 0.0f)) {
    lora = nullptr;
  }
  // One route for the WHOLE projection, chosen at the full M so a band
  // never lands on a kernel the tuner did not measure, then the rows in
  // bands narrow enough for the mma tiles' 32-bit addressing. Below the
  // 2^31-byte line -- everything this model ran before 1344x768 -- the
  // band IS M and this is the single pass it always was.
  const GemmRoute route = gemm_route_(M, N, K);
  const int band = mma_row_band(N, K);
  for (int m0 = 0; m0 < M; m0 += band) {
    const int rows = std::min(band, M - m0);
    const std::size_t xo = x_off + (std::size_t)m0 * K;
    const std::size_t yo = y_off + (std::size_t)m0 * N;
    const std::size_t lo =
        lora != nullptr ? (std::size_t)m0 * lora->rank : 0;
    // t = [scale *] x A^T, BEFORE the projection: the fused tile reads
    // it, and the unfused path does not care which order the two are in.
    bool scaled = false;
    if (lora != nullptr) {
      scaled = lora_gemm_a_(enc, x, xo, *lora, rows, K, lo);
    }
    bool folded = false;
    gemm_route_dispatch_(enc, x, xo, l, y, yo, rows, N, K, route,
                         // The fused tile has no scale, so offering it an
                         // adapter whose strength never reached t would
                         // run that adapter at 1.0.
                         (lora != nullptr && scaled) ? lora : nullptr,
                         &folded, lo);
    if (lora != nullptr && !folded) {
      lora_gemm_b_(enc, *lora, y, yo, rows, N, scaled, lo);
    }
    if (bias) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, yo * 2);
      enc.set_buffer(1, l.b);
      enc.set_constant(2, N);
      enc.set_constant(3, rows * N);
      enc.dispatch({(unsigned)(rows * N), 1, 1}, {256, 1, 1});
    }
  }
}

// Pick the GEMM route per projection, by MEASURING it at the sequence
// length that is about to run.
//
// The crossover is a function of M, of each projection's (N, K), and of
// the machine -- see the table on gemm_route_. Three of the four are
// unknown until a graph asks for a geometry, and the fourth (the machine)
// is the one a constant in this file can never track: these kernels were
// tuned on a pre-matrix-core GPU, and an M5 will not agree.
//
// On M5 the candidates include the matrix-core tiles, so this one measure-
// ment answers BOTH "which steel tile" and "steel or matmul2d at all".
// Making that a threshold instead would repeat the mistake this tuner was
// written to fix, one generation later: the shipped BM128-at-M>=8192 rule
// was a constant copied from another model, and it was the SLOWEST arm at
// this model's own production geometry.
//
// The mma arms are timed WITH their dequant pass, which is what the
// forward pays: every block carries its own weights, so the dequant is per
// GEMM and not once per model. Timing it separately would flatter the
// matrix-core arm at exactly the short sequences where it is marginal.
//
// Runs ONCE per distinct sequence length, before the forward's command
// stream opens, over the four distinct projection shapes a block uses. It
// costs roughly three blocks' worth of GEMM -- ~10 s at the production
// geometry -- against a denoise loop of 20+ steps x 50 blocks, so under
// 1%. VPIPE_H3_NO_QMM_AUTOTUNE=1 skips it and keeps the fallback rule.

namespace {

// Does this route's RANKING move with the row count, or only its
// absolute time? Marked here, statically, per kernel -- it is a property
// of what the kernel does and not of the checkpoint or the machine.
//
// FALSE (steel). qmm walks the quantized weight inline, one output tile
// at a time, and every byte it touches it touches per tile. Double the
// rows and it does the same thing twice, so two steel routes keep their
// ratio all the way up.
//
// TRUE (mma). The matrix-core routes DEQUANTIZE the whole [N, K] weight
// into a dense bf16 buffer before they multiply -- 308 MB at the widest
// projection here -- and they pay that once per call whatever M is.
// Their time is therefore fixed + proportional, and at too few rows the
// fixed half dominates: MEASURED against the GEMM it is ~1% of a
// production-height call and ~11% of a 1k-row one. Timing them short
// charges them a cost the forward amortizes away, which is exactly how
// a tuner talks itself out of the faster kernel.
bool
route_m_sensitive_(H3Route r)
{
  return route_is_mma_(r);
}

// The rows a route has to be timed over. Both numbers are floors on the
// MEASUREMENT, never on what runs.
//
// 1024 for the flat routes is about filling the machine, and it is
// generous for that: the grid is two-dimensional and N alone already
// supplies 42 tiles at the narrowest projection here.
//
// 4096 for the fixed-cost ones is where the dequant has fallen to a few
// percent of the call -- close enough to its production share that it
// cannot reorder a field the run-to-run noise does not already reorder.
//
// That is also where it stops drifting, MEASURED on an M5 against the
// full-height answer at 16990 rows (67.5 s of tuning):
//
//   4096  qkv/o/fc1 = mma 128x256 tn2, fc2 = mma 128x256 + split7
//         -- the full-height answer exactly, in 16.5 s
//   2048  same tiles, split7 -> split4
//   1024  every shape drops to the plain 128x128 tile
//    512  ...and split-K disappears
//
// So the drift is real and it is one-directional: too few rows and the
// WIDE tiles and the deep split, which are the two things that need
// rows to pay for themselves, both lose to something narrower. A cap
// picked for cheapness rather than measured would have quietly tuned
// this model onto slower kernels and reported a tuned answer.
//
// VPIPE_H3_TUNE_ROWS overrides both, which is how the ladder above was
// taken and how to re-take it on a machine this was not tuned on.
int
route_tune_rows_(H3Route r, int ceiling)
{
  // The ceiling IS the override when there is one -- it is the rounded
  // form of the same number -- so a forced value reaches both floors
  // through here and the two can never disagree.
  if (tune_rows_override_() > 0) { return ceiling; }
  return route_m_sensitive_(r) ? kMmaTuneRows : kFlatTuneRows;
}


}  // namespace

void
MetalMiniMaxH3Transformer::tune_qmm_(int M)
{
  if (_qmm_manual) { return; }   // the caller is driving; see set_qmm_tile
  // Keyed on what will be MEASURED, not on what the caller asked for --
  // see tune_ceiling_. Two video geometries above the ceiling share one
  // entry because they would produce one answer.
  const int key = tune_row_key(M);
  for (const QmmTuneSet& set : _qmm_tuned) {
    if (set.m == key) { return; }
  }
  // Nothing to choose: no wide steel twins built AND no matrix cores.
  if (_qmm_tile < 1 && !_use_mma2) { return; }
  static const bool kOff =
      std::getenv("VPIPE_H3_NO_QMM_AUTOTUNE") != nullptr;
  if (kOff) { return; }

  // A resident block to borrow weights from. With streaming and no
  // pinned prefix the main stack is not in memory here, but the token
  // refiner always is and carries the same projection shapes.
  const Block* b = nullptr;
  if (!_blocks.empty() && !_blocks[0].qkv.empty()) { b = &_blocks[0]; }
  else if (!_refiner.empty() && !_refiner[0].qkv.empty()) {
    b = &_refiner[0];
  }
  if (b == nullptr) { return; }

  const Config& c = _cfg;
  const int H = c.hidden, I = c.inner(), F = c.ffn;
  struct Shape { const Linear* l; int N, K; const char* name; };
  const Shape shapes[] = {
      {&b->qkv, 3 * I,     H, "qkv"},
      {&b->out, H,         I, "o"},
      {&b->fc1, 2 * F,     H, "fc1"},
      {&b->fc2, H,         F, "fc2"},
  };
  // Input and output scratch, already sized for this sequence: s.qkv is
  // [M, 3I] (>= every K here) and s.ff is [M, 2F] (>= every N). Distinct
  // buffers, so no shape aliases its own source.
  const SharedBuffer& xin  = _s.qkv;
  const SharedBuffer& yout = _s.ff;
  if (xin.empty() || yout.empty() || _s.seq != M) { return; }
  // With a fused FF, `yout` is only as wide as 3*inner and fc1 has no
  // route to choose anyway -- it is the fused kernel at its own tile. So
  // it is skipped, and skipping it is REQUIRED, not an optimization:
  // tuning it would write 2*ffn columns into a row that is not that wide.
  const bool skip_fc1 = !_ff_needs_wide;

  static const GemmRoute kAll[] = {
      GemmRoute::kSteelBm32, GemmRoute::kSteelBm64, GemmRoute::kSteelBm128,
      GemmRoute::kMma128,    GemmRoute::kMma128x256,
      GemmRoute::kMma128x256Tn2,
  };


  std::vector<QmmTune> tuned;
  std::string detail;
  int tune_rows = 0;
  std::size_t n_pruned = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (const Shape& sh : shapes) {
    if (skip_fc1 && sh.N == 2 * F) { continue; }
    // Candidates are filtered per SHAPE, not once: route_ok_ takes
    // (M, N, K), and the tn2 tile in particular is legal for some of this
    // model's widths and not others.
    std::vector<GemmRoute> cands;
    for (GemmRoute r : kAll) {
      if (route_ok_(r, M, sh.N, sh.K)) { cands.push_back(r); }
    }
    if (cands.size() < 2) {
      if (!cands.empty()) { tuned.push_back(QmmTune{sh.N, sh.K, cands[0]}); }
      continue;
    }
    // How many rows this set has to be TIMED over. Not the caller's M:
    // at production geometry that is ~17k rows, and timing six routes
    // over four shapes there costs a minute of GPU before the first block
    // of the first step reports anything.
    //
    // Every route tiles the output and loops. Rows past the point where
    // the grid is full multiply every candidate's time by the same
    // factor, so the RANKING stops moving long before the row count does
    // -- and the grid is full from N alone here, since the narrowest tile
    // is 128 wide against a 5376-wide projection.
    //
    // What that argument does NOT cover is a route with an M-INDEPENDENT
    // cost, which is why the floor is a static property of the KERNEL
    // (route_m_sensitive_) rather than one number for the model.
    int tune_m = 0;
    bool any_mma = false;
    for (GemmRoute r : cands) {
      tune_m = std::max(tune_m, route_tune_rows_(r, _tune_ceiling));
      any_mma = any_mma || route_m_sensitive_(r);
    }
    // An mma route below _mma_min_m does not merely score badly -- it is
    // refused and silently dispatched as steel, which would time one
    // kernel in two arms. Never below the floor that decides that.
    if (any_mma) { tune_m = std::max(tune_m, _mma_min_m); }
    tune_m = ((tune_m + 127) / 128) * 128;    // whole tiles, no tail
    if (tune_m > M) { tune_m = M; }
    tune_rows = tune_m;

    // One arm, warm pass and timed round alike.
    //
    // No adapter: the tuner measures the BASE projection. The fold adds
    // ~1% of a tile's work and only to the two plain tiles, which is far
    // short of flipping a route, and giving the tuner an adapter would
    // make its choice depend on which LoRA happened to be loaded.
    //
    // Banded exactly as the forward encodes it. Unbanded, an mma route
    // past 2^31 bytes would silently skip its last tiles and be timed for
    // work it did not do, while steel -- which addresses in 64 bits --
    // did all of it. (At `tune_m` the band never fires; it is here
    // because what is timed has to be what runs.)
    auto time_route = [&](GemmRoute r) {
      return autotune_time(_mc, 1, [&](ComputeEncoder& enc) {
        dispatch_row_bands_(enc, xin, 0, *sh.l, yout, 0, tune_m, sh.N,
                            sh.K, r);
      });
    };

    // PRUNE ON A WARM PASS, which is how this gets cheaper without being
    // told anything about the machine it is on.
    //
    // The point of this tuner is that an M4 Pro, an M5, an M5 Max and
    // whatever comes next each end up on their own best kernel, so the
    // one thing it must not do is carry a constant fitted to the box it
    // was written on. But most of what it spends goes on arms that are
    // not close, and WHICH arms those are is itself something a
    // measurement can answer. So: run every candidate once, drop the ones
    // far behind the leader, and spend the timed rounds on what is left.
    // On this M5 that drops the three steel tiles because they measured
    // 3x behind; on a machine with no matrix cores nothing is dropped,
    // because the three steel tiles are within 1.2x of each other; on a
    // part where steel wins it is the matmul2d arms that go.
    //
    // THE THRESHOLD IS SET BY WHAT THE WARM PASS CAN ACTUALLY RESOLVE,
    // and that is much less than it looks. MEASURED on the M5, warm
    // against the timed mean, as a ratio to the leader:
    //
    //   fc1 route  warm .292 .316 .309 | .865 .932 1.000
    //              mean .289 .312 .306 | .866 .930 1.000
    //   fc2 route  warm .312 .339 .332 | .938 .976 1.000
    //              mean .314 .341 .335 | .992 .989 1.000
    //
    // Far-behind arms are identified exactly. Close ones are NOT: in the
    // split sweep on the same box, warm rated two arms .905 and .851
    // whose means were .990 and .996, and the first of them went on to
    // win the vote. A threshold at .85 would have pruned the winner.
    //
    // Hence 0.6: it removes only the hopeless, never reaches into the
    // cluster where the decision actually lives, and the two fastest
    // survive whatever the ratios say.
    //
    // VPIPE_H3_TUNE_NO_PRUNE=1 measures every candidate, which is how to
    // check that the pruned arms were not deciding anything. MEASURED
    // here: 16.7 s against 11.9, same routes.
    static const bool kNoPrune =
        std::getenv("VPIPE_H3_TUNE_NO_PRUNE") != nullptr;
    if (!kNoPrune && cands.size() > 2) {
      std::vector<double> warm(cands.size(), 0.0);
      bool timed = true;
      for (std::size_t i = 0; i < cands.size() && timed; ++i) {
        warm[i] = time_route(cands[i]);
        timed = warm[i] > 0.0;
      }
      if (timed) {
        std::vector<GemmRoute> keep;
        // Original order, so the vote's tie-break is stable.
        for (int i : tune_prune_survivors(warm, kPruneKeep)) {
          keep.push_back(cands[(std::size_t)i]);
        }
        n_pruned += cands.size() - keep.size();
        cands.swap(keep);
      }
    }

    const int w = autotune_vote((int)cands.size(), /*rounds=*/2,
        /*reps_for_us=*/1,
        [&](int i) { return time_route(cands[(std::size_t)i]); });
    const GemmRoute win = cands[(std::size_t)w];
    tuned.push_back(QmmTune{sh.N, sh.K, win});
    if (!detail.empty()) { detail += " "; }
    detail += std::string(sh.name) + "=" + route_name_(win);
    // Second decision, only where a split is possible at all: with the tile
    // settled, is the contraction better split? Driven through the same
    // dispatcher, so the tuner compares the winning route against itself
    // with and without the split rather than against a stand-in. The route
    // has to be pinned first -- measuring both at once would confound them.
    if (route_is_mma_(win)) {
      // Keyed on the REAL M -- plan() matches the row count the forward
      // asks with -- but MEASURED at tune_m like everything else. The
      // split path already walks M in row blocks of its own choosing
      // (MmaSplitK::rows_per_block), so above one block the decision no
      // longer moves with M; tune_m is at least one such block at every
      // shape here.
      const int sp = _splitk.tune(_mc, sh.K, sh.N, M,
          [&](ComputeEncoder& enc) {
            dispatch_row_bands_(enc, xin, 0, *sh.l, yout, 0, tune_m, sh.N,
                                sh.K, win);
          });
      if (sp > 0) { detail += "+split" + std::to_string(sp); }
    }
  }
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  ++_qmm_tune_count;
  // Bounded: a pathological caller sweeping sequence lengths must not
  // grow this without limit, and the oldest entry is the one least
  // likely to come back.
  if (_qmm_tuned.size() >= 8) { _qmm_tuned.erase(_qmm_tuned.begin()); }
  _qmm_tuned.push_back(QmmTuneSet{key, std::move(tuned)});
  // The row count MEASURED, which is also the cache key. The caller's own
  // row count is in the debug line below and in the first-forward log; it
  // is deliberately not what this string names, because after the
  // bucketing it is not what the answer belongs to.
  _qmm_tuning_desc = std::to_string(key) + " rows: " + detail;
  if (_mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "[h3-dit] qmm autotune for {} rows, timed over {}: {} "
        "({:.0f} ms, {} arm(s) pruned on the warm pass)",
        M, tune_rows, detail, ms, n_pruned));
  }
}

// ---- rope ------------------------------------------------------------

void
MetalMiniMaxH3Transformer::build_rope_(const h3::PackedLayout& L,
                                       SharedBuffer& cos_out,
                                       SharedBuffer& sin_out) const
{
  // Per row: the three axis coordinates each multiply the SAME inv_freq
  // table, and the three blocks are concatenated to 3 * rope_freq_dim.
  // The reference then concatenates that with itself to reach the 96
  // rotated channels -- the second copy is identical, so only the first
  // half is stored and the kernel indexes both halves into it.
  const int F = _cfg.rope_freq_dim;
  const int half = 3 * F;
  auto* cb = static_cast<float*>(cos_out.contents());
  auto* sb = static_cast<float*>(sin_out.contents());
  for (int s = 0; s < L.seq_len; ++s) {
    const double* p = &L.position_ids[(std::size_t)s * 3];
    for (int a = 0; a < 3; ++a) {
      for (int i = 0; i < F; ++i) {
        // Evaluated in double: these are absolute positions scaled by 32
        // over sequences tens of thousands of rows long, so the angles
        // are large and a float accumulation would quantize the grid.
        const double ang = p[a] * (double)_inv_freq[(std::size_t)i];
        const std::size_t o = (std::size_t)s * half + (std::size_t)(a * F + i);
        cb[o] = (float)std::cos(ang);
        sb[o] = (float)std::sin(ang);
      }
    }
  }
}

// ---- the timestep MLP, on the host in f32 ----------------------------

bool
MetalMiniMaxH3Transformer::time_embed_(const std::vector<float>& timesteps,
                                       SharedBuffer& out) const
{
  const int n_t = (int)timesteps.size();
  const int F = _cfg.freq_dim, H = _cfg.time_hidden, D = _cfg.time_dim;
  if (n_t <= 0 || out.byte_size() < (std::size_t)n_t * D * 2) { return false; }
  if ((int)_time_in_w.size() != H * F || (int)_time_out_w.size() != D * H) {
    return false;
  }
  std::vector<float> row((std::size_t)F), h1((std::size_t)H);
  auto* dst = static_cast<std::uint16_t*>(out.contents());
  const int half = F / 2;
  // Diagnostic: the scale the sinusoidal grid sees. At t in [0, 1] every
  // angle is tiny and the embedding is nearly CONSTANT across the
  // schedule, which reads as a model that cannot tell its own noise
  // level. 1000 is the flow-scheduler convention.
  double tscale = 1.0;
  if (const char* ts = std::getenv("VPIPE_H3_TSCALE")) {
    tscale = std::atof(ts);
  }
  for (int t = 0; t < n_t; ++t) {
    // diffusers Timesteps(freq_dim, flip_sin_to_cos=True,
    // downscale_freq_shift=0): cos first, then sin. Timesteps arrive in
    // [0, 1] UNSCALED -- there is no 1000x here, which is the scale
    // every other flow model in this tree conditions on.
    for (int i = 0; i < half; ++i) {
      const double fr = std::exp(-std::log(1e4) * (double)i / (double)half);
      const double ang = (double)timesteps[(std::size_t)t] * tscale * fr;
      row[(std::size_t)i] = (float)std::cos(ang);
      row[(std::size_t)(half + i)] = (float)std::sin(ang);
    }
    for (int o = 0; o < H; ++o) {
      float acc = _time_in_b[(std::size_t)o];
      const float* w = &_time_in_w[(std::size_t)o * F];
      for (int i = 0; i < F; ++i) { acc += w[i] * row[(std::size_t)i]; }
      h1[(std::size_t)o] = acc / (1.0f + std::exp(-acc));   // silu
    }
    for (int o = 0; o < D; ++o) {
      float acc = _time_out_b[(std::size_t)o];
      const float* w = &_time_out_w[(std::size_t)o * H];
      for (int i = 0; i < H; ++i) { acc += w[i] * h1[(std::size_t)i]; }
      // Every AdaLN module applies its OWN silu to temb and casts the
      // result to its projection's dtype, so the activation runs at f32
      // here and only its output rounds to bf16. That order is the
      // reference's and its comment says why: rounding BEFORE the
      // activation biases every block's modulation identically at every
      // step, which then accumulates coherently down the denoising
      // trajectory instead of averaging out.
      const float sv = acc / (1.0f + std::exp(-acc));
      dst[(std::size_t)t * D + (std::size_t)o] = f32_to_bf16_(sv);
    }
  }
  return true;
}

std::size_t
MetalMiniMaxH3Transformer::adaln_table_bytes() const
{
  std::size_t n = _final_adaln_tab.byte_size();
  for (const SharedBuffer& b : _adaln_tab) { n += b.byte_size(); }
  return n;
}

// The AdaLN bake is the one reader in this model whose tensors are all
// the SAME SHAPE -- 50 projections of [96768, time_dim], byte for byte
// identical in size. That is what makes the fast read available to it:
// refill_streamed_tensor needs a destination the caller already owns,
// and the route this replaces allocated a fresh set per block and threw
// it away, so there was never a buffer to refill.
//
// It matters because the bake is the single largest read in a run. At
// the released 8-bit config the projections are 13.86 GB, 39.3% of the
// checkpoint, and MEASURED on an M5 over the real tensors (arms
// interleaved, order rotated, no block read by both) the mapped memcpy
// this used runs at 1.32-1.36 GB/s against 5.97-5.98 GB/s for a pread
// into a buffer that exists. That is ~10 s against ~2 s, before the
// first step reports anything.
//
// Block 0 still goes through linear_(): that is where `quantized` and
// `bits` are derived from the shapes, and one block on the old path
// against forty-nine is not worth a second copy of that reasoning. It
// also makes the fallback trivial -- anything the refill will not serve
// rebuilds here and becomes the destination for the blocks after it, so
// a checkpoint with a shape or dtype this route cannot place is slower
// and not wrong.
bool
MetalMiniMaxH3Transformer::adaln_into_(const std::string& nm, Linear& dst)
{
  if (!dst.empty()) {
    // bf16, because that is what this model reads a projection as: the
    // f16 scales and biases are converted by to_bf16_() on the old path
    // and in place here, to the same bits.
    const SharedBuffer* parts[] = {
        dst.quantized ? &dst.codes : &dst.w,
        &dst.scales, &dst.qbias, &dst.b,
    };
    const char* suffix[] = {".weight", ".scales", ".biases", ".bias"};
    bool all = true;
    for (int i = 0; i < 4 && all; ++i) {
      // A Linear that never had this part (an unquantized projection has
      // no scales) is not a refusal -- there is nothing to refill.
      if (parts[i]->empty()) { continue; }
      all = refill_streamed_tensor(*_ws, nm + suffix[i], *parts[i],
                                   RefillDst::kBf16) == Refill::kFilled;
    }
    if (all) { return true; }
  }
  dst = linear_(*_ws, nm, true, Retain::Streamed);
  return !dst.empty();
}

bool
MetalMiniMaxH3Transformer::bake_adaln(
    const std::vector<std::vector<float>>& schedule, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "minimax-h3 bake_adaln: " + m; }
    return false;
  };
  if (schedule.empty()) { return fail("empty schedule"); }
  if (_mc == nullptr || _ws == nullptr) { return fail("no weights"); }
  const Config& c = _cfg;
  const int D = c.adaln_out();
  const int FD = 2 * c.hidden;

  // A step's timesteps land CONTIGUOUSLY, so a block binds its slice at
  // an offset rather than gathering rows. The condition timestep repeats
  // in every step's group; deduplicating it would save one row in three
  // and cost a gather, which is the wrong trade at these sizes.
  std::vector<float> all;
  _adaln_row0.clear();
  _adaln_nt.clear();
  for (const std::vector<float>& st : schedule) {
    if (st.empty()) { return fail("a step with no timesteps"); }
    _adaln_row0.push_back((int)all.size());
    _adaln_nt.push_back((int)st.size());
    all.insert(all.end(), st.begin(), st.end());
  }
  const int T = (int)all.size();

  // The size grows with the step count, and past some point holding the
  // table is worse than recomputing it -- so this REFUSES rather than
  // silently taking a bad trade. The default budget is generous next to
  // what it replaces (12.9 GB of weights at the released config) and
  // still bounds a long schedule: 32 steps is ~930 MB.
  const std::size_t kBudget = adaln_table_budget_();
  const std::size_t want =
      (std::size_t)T * (std::size_t)D * 2 * (std::size_t)c.n_layers +
      (std::size_t)T * (std::size_t)FD * 2;
  if (want > kBudget) {
    return fail(fmt("{} steps would need {} MB of tables, over the {} MB "
                    "budget (VPIPE_H3_ADALN_BAKE_MAX_MB)",
                    schedule.size(), want >> 20, kBudget >> 20)());
  }

  // The timestep MLP for EVERY row at once, on the host in f32 exactly
  // as a single step would do it.
  SharedBuffer temb =
      _mc->make_shared_buffer((std::size_t)T * (std::size_t)c.time_dim * 2);
  if (temb.empty()) { return fail("temb allocation failed"); }
  if (!time_embed_(all, temb)) { return fail("time embedding failed"); }

  // CAVEAT ON THIS NUMBER, which is a byte count and not a measurement.
  //
  // Releasing a weight that was MAPPED drops a view into the shard and
  // frees no physical memory -- the mapping is whole-shard and stays --
  // so in the preloading arm (see kept_residency_) this over-reports.
  // It is left as a byte count anyway because there is no cheap
  // predicate for "was this backed by a file": SharedBuffer::is_owned()
  // answers a DIFFERENT question (did this handle allocate the bytes, or
  // is it an alias of a cached tensor), and filtering on it reports 0 MB
  // in every arm -- including the streamed one, where the bytes really
  // are released. Measured that mistake before believing it.
  //
  // The bake's other win -- not running 51 projections per step -- is
  // real in both arms and is not what this number is about.
  auto lin_bytes = [](const Linear& l) {
    return l.w.byte_size() + l.b.byte_size() + l.codes.byte_size() +
           l.scales.byte_size() + l.qbias.byte_size();
  };
  std::size_t freed = 0;
  _adaln_tab.clear();
  _adaln_tab.resize((std::size_t)c.n_layers);
  // ONE set of destinations for all 50, refilled in place -- see
  // adaln_into_. Released at the end of the bake; holding 276 MB for the
  // rest of a run on the box this streams for would give back a good
  // part of what the bake just freed.
  Linear held;
  for (int i = 0; i < c.n_layers; ++i) {
    // Take the projection from wherever it is. A resident block already
    // holds it; a streamed one does not, and reading JUST this tensor is
    // the point -- the bake never needs the rest of the block, so it
    // touches the 13.9 GB once and nothing else.
    const Linear* ada = nullptr;
    if ((std::size_t)i < _blocks.size() && !_blocks[(std::size_t)i].adaln.empty()) {
      ada = &_blocks[(std::size_t)i].adaln;
    } else {
      if (!adaln_into_(blk_("blocks.", i, "") + "adaln_proj.linear", held)) {
        return fail(fmt("block {} has no adaln", i)());
      }
      ada = &held;
    }
    SharedBuffer tab =
        _mc->make_shared_buffer((std::size_t)T * (std::size_t)D * 2);
    if (tab.empty()) { return fail(fmt("table {} allocation failed", i)()); }
    {
      CommandStream st = _mc->make_command_stream();
      {
        ComputeEncoder enc = st.begin_compute();
        gemm_(enc, temb, 0, *ada, tab, 0, T, D, c.time_dim);
      }
      // CHECKED, because a discarded status here is the worst kind of
      // silence this model can produce. The table is a FRESH allocation:
      // if the GEMM's command buffer fails -- out of memory is the way,
      // and the video geometries put ~10 GB of scratch beside the
      // weights -- nothing has written it, and every block then
      // modulates every row of every step with whatever those pages
      // held. That is a whole-latent corruption in BOTH modalities from
      // a run that reported success.
      std::string bake_err;
      if (!st.commit().wait_ok(&bake_err)) {
        return fail(fmt("AdaLN bake of block {} failed: {}", i,
                        bake_err.empty() ? std::string("GPU error")
                                         : bake_err)());
      }
    }
    _adaln_tab[(std::size_t)i] = std::move(tab);
    // Release the weights. For a resident model this IS the win; for a
    // streamed one the win is that load_block_ stops asking for them.
    freed += lin_bytes(*ada);
    if ((std::size_t)i < _blocks.size()) {
      _blocks[(std::size_t)i].adaln = Linear{};
    }
  }

  // Same two cases as the blocks, and the final layer needs them for the
  // same reason the blocks do: a SECOND bake (a new schedule on a model
  // that already baked one) finds this released and has to re-read it.
  // Missing that made re-baking silently project against an empty
  // weight, which minimax_h3_dit.denoise_holds_the_anchors caught as two
  // denoise runs disagreeing -- not as a failure at the bake.
  // The blocks are done with it. The final layer is a different shape
  // and reads its own.
  held = Linear{};

  Linear fheld;
  const Linear* fada = &_final_adaln;
  if (_final_adaln.empty()) {
    fheld = linear_(*_ws, "final_layer.adaln_proj.linear", true,
                    Retain::Streamed);
    if (fheld.empty()) { return fail("final_layer has no adaln"); }
    fada = &fheld;
  }
  _final_adaln_tab =
      _mc->make_shared_buffer((std::size_t)T * (std::size_t)FD * 2);
  if (_final_adaln_tab.empty()) { return fail("final table alloc failed"); }
  {
    CommandStream st = _mc->make_command_stream();
    {
      ComputeEncoder enc = st.begin_compute();
      gemm_(enc, temb, 0, *fada, _final_adaln_tab, 0, T, FD, c.time_dim);
    }
    // Same reasoning as the per-block tables above: unwritten here is
    // silent, and it reaches the output layer of every step.
    std::string bake_err;
    if (!st.commit().wait_ok(&bake_err)) {
      return fail(fmt("AdaLN bake of the final layer failed: {}",
                      bake_err.empty() ? std::string("GPU error")
                                       : bake_err)());
    }
  }
  freed += lin_bytes(*fada);
  _final_adaln = Linear{};

  // 13.2 GB of per-step projections just stopped existing, and that is
  // not something the residency policy can see or predict. Anything it
  // concluded about this box a moment ago was measured against a load
  // this model no longer carries, so the ratchet starts over.
  _resid.note_landscape_changed();

  // NOT RECYCLABLE from here on.
  //
  // The bake reads every adaln_proj uncached and replaces it with a
  // table built for THIS schedule, then releases the projections. The
  // checkpoint on disk is untouched -- and if the pool held only the
  // checkpoint this would not matter. What it holds is the set a model
  // was built from, and a relaunch that recycled it would find a model
  // whose tables are right for a step count it does not share, and whose
  // projections are gone.
  //
  // Said here rather than at the pool, because here is the only place
  // that knows the specialisation happened.
  if (_ws) {
    _ws->set_not_recyclable(fmt(
        "AdaLN baked for a {}-step schedule", schedule.size())());
  }

  // Both of BlockResidency's rates are tuned for a 30-step schedule and
  // are wrong for a 5-step turbo one, in the same direction.
  //
  // The per-forward cap exists so the residency MEASUREMENT gets a look
  // before the whole checkpoint has been admitted -- so what it should
  // bound is "blocks per look", and the number of looks is the number of
  // steps. A constant 8 lets a 30-step run reach 50 blocks with 20 looks
  // to spare and caps a 5-step run at 40 it can never use. Sized so the
  // set converges by the penultimate step, every schedule gets the same
  // number of chances to react rather than the same number of blocks.
  //
  // Recovery has the same defect: one block per three quiet forwards
  // cannot undo a shed inside a 5-step run at all, which is how a single
  // sample taken during the bake decided the whole of it.
  //
  // ALL OF THIS IS STREAMING-ONLY. Under preload every block is already
  // resident, so `streaming` is false for every block of every forward
  // and admit() is never reached -- the cap would gate nothing. It was
  // computed and PRINTED anyway, which put "residency probe 26 blocks of
  // 50" under a run that had just announced PRELOAD and read as a
  // streaming model that had pinned half its stack.
  if (_stream_blocks) {
    const int steps = (int)schedule.size();
    // The PROBE is a question about ROOM, not about the schedule.
    //
    // On a box with space free, the right first forward is a large one:
    // a block admitted on forward 1 is reused by every forward after it,
    // one admitted on the last is reused by nothing. So this asks how
    // many blocks fit RIGHT NOW and commits half of them, then
    // BlockResidency doubles per healthy forward -- which reaches the
    // whole stack in two or three forwards on a roomy box and stays
    // small on a tight one, without either answer being written down in
    // advance.
    //
    // THREE QUARTERS rather than all of it, because the measurement
    // that would say this was too much does not exist until the next
    // forward. That is the safe margin -- and it is a margin on top of
    // two others, which is why it can be this generous: the figure is
    // already 90% of RECLAIMABLE physical memory, and already clamped
    // to what can be wired. Spreading over the schedule was not a
    // margin at all, it was just slower everywhere.
    //
    // A ONE-step schedule is the exception: it has no second forward, so
    // nothing kept is ever reused and every admission is a block-sized
    // memcpy plus an mlock paid for nothing.
    int probe = 1;
    if (steps > 1 && _wire_resident) {
      // WIRING ON: NO RATE LIMIT. Take everything the box will give on
      // the first pass.
      //
      // The cap exists because admit()'s gate is arithmetic over
      // `available_physical`, which counts file cache -- on a streaming
      // model mostly the checkpoint that model just read, so the signal
      // says "room" the more it streams. The true signal was the
      // mincore walk, which arrives once per forward, so the cap bounded
      // what could be committed before the first evidence.
      //
      // Wiring replaces that evidence with a SYNCHRONOUS one. mlock
      // either takes the block or refuses it, per block, before it is
      // kept -- and a refusal collapses `_wire_budget` to what was
      // granted, so admissions stop on the spot. There is no window
      // between committing and finding out, which is the only thing the
      // cap was buying. (And since the walk now skips wired buffers,
      // waiting for it would be waiting for a measurement that reports
      // nothing.)
      //
      // What it COSTS is a whole extra pass of reads. A block not
      // admitted on forward 1 was still read from disk, used and
      // dropped -- so capping at 26 of 50 means 24 blocks re-read on
      // forward 2 for nothing, ~9.4 GB at this model's sizes.
      probe = c.n_layers;
    } else if (steps > 1) {
      const std::size_t blk = resident_block_bytes_();
      const auto mb = _mc->memory_budget();
      std::size_t room = (std::size_t)((double)mb.available_physical * 0.9);
      const std::size_t keep = _resid.reserve();
      room = room > keep ? room - keep : 0;
      if (blk > 0) {
        const std::size_t fits = room / blk * 3 / 4;
        probe = fits > (std::size_t)c.n_layers ? c.n_layers
                                               : (int)fits;
      }
      if (probe < 1) { probe = 1; }
    }
    // Forcing the probe is what makes the cap MEASURABLE. Its whole
    // effect is on the first forward or two, so an A/B needs the two
    // arms in one binary -- otherwise the only comparison available is
    // against a different build on a different day.
    if (const char* e = std::getenv("VPIPE_H3_PROBE")) {
      const int v = std::atoi(e);
      if (v > 0) { probe = v > c.n_layers ? c.n_layers : v; }
    }
    _resid.set_per_forward_cap(probe);
    _resid.set_quiet_forwards(steps / 4 > 0 ? (steps / 4 > 3 ? 3 : steps / 4)
                                            : 1);
    if (_mc->session() != nullptr) {
      _mc->session()->info(fmt(
          "MetalMiniMaxH3Transformer: residency probe {} blocks of {} "
          "({} MB each, {} MB reclaimable, wire budget {} MB){}",
          probe, c.n_layers,
          resident_block_bytes_() >> 20,
          _mc->memory_budget().available_physical >> 20,
          _wire_budget >> 20,
          _wire_resident
              ? " -- uncapped, the wire budget is the gate"
              : ", doubling per healthy forward"));
    }
  }

  if (_mc->session() != nullptr) {
    _mc->session()->log_normal(fmt(
        "MetalMiniMaxH3Transformer: baked AdaLN for {} steps ({} rows) -- "
        "{} projections ({} MB of weights) replaced by {} MB of tables{}",
        schedule.size(), T, c.n_layers + 1, freed >> 20,
        adaln_table_bytes() >> 20,
        _stream_blocks ? ", and no longer streamed per step" : ""));
  }
  return true;
}

// ---- scratch ---------------------------------------------------------

// The scratch a forward at this geometry needs, in bytes.
//
// It IS a second opinion about ensure_scratch_ -- the two list the same
// buffers separately -- so what keeps them honest is
// minimax_h3_dit.scratch_estimate_matches_allocation, which allocates at
// a real geometry and compares this against scratch_resident_bytes(). Do
// not add a buffer to one without the other; the test fails loudly if you
// do, which is the point.
//
// That matters more here than it looks: at video sequence lengths this
// scratch is the single largest live allocation the model makes, ~200 KB
// PER ROW -- ~1.9 GB at the 9382-row production layout and ~3.9 GB at 19k
// rows. An estimator that quietly under-reports is how a preflight passes
// and the machine then thrashes, and on a 16 GB box thrashing is not an
// OOM kill, it is a watchdog kernel panic.
//
// The dequant scratch is counted too. It is allocated lazily inside
// gemm_mma_ rather than here, but it is just as resident and just as
// large (the widest projection: 2*ffn x hidden = 308 MB at the released
// config), so leaving it out would understate the peak by that much.
namespace {

// elems is in ELEMENTS of `esz` bytes; one row of the table per buffer
// ensure_scratch_ allocates.
struct ScratchItem { std::size_t mul_seq, mul_text, mul_t, esz; };

// Elements per row of the ATTENTION ARENA: qh|kh|vh|oh|ob end to end,
// with `ff` aliased over the whole of it. The five are dead before the
// FF writes, so what has to be held is the WIDER of the two uses.
//
// max(), not 5*I: nothing guarantees 2*ffn fits inside five inner()s on
// a config this file has not seen. Where it does not, the arena is the
// ff width and the attention buffers sit inside it instead.
std::size_t
attn_arena_elems_(const MetalMiniMaxH3Transformer::Config& c, bool narrow_ff)
{
  const std::size_t I  = (std::size_t)c.inner();
  const std::size_t ff = narrow_ff ? 3 * I : 2 * (std::size_t)c.ffn;
  // ...+ hidden, because `proj` also lives here, parked past ff. It is
  // written only by the FINAL layer -- final norm, modulate, then the
  // video/audio output projections read it -- so it is dead for the
  // whole block loop, and everything else in this arena is dead by the
  // time it is written. Placing it PAST ff rather than at 0 costs
  // nothing on the released config (34048 still under 5*inner) and
  // means the two can never collide even if ff's live range grows.
  return std::max<std::size_t>(5 * I, ff + (std::size_t)c.hidden);
}

std::vector<ScratchItem>
scratch_plan_(const MetalMiniMaxH3Transformer::Config& c, bool narrow_ff)
{
  const std::size_t H = (std::size_t)c.hidden;
  const std::size_t I = (std::size_t)c.inner();
  const std::size_t rot_half = (std::size_t)(3 * c.rope_freq_dim);
  return {
      {rot_half, 0, 0, sizeof(float)},           // rcos
      {rot_half, 0, 0, sizeof(float)},           // rsin
      {H, 0, 0, 2}, {H, 0, 0, 2},                // x, nm (proj is in the arena)
      {3 * I, 0, 0, 2},                          // qkv
      // qh|kh|vh|oh|ob and ff, one arena (see attn_arena_elems_).
      {attn_arena_elems_(c, narrow_ff), 0, 0, 2},
      {0, H, 0, 2},                              // txt
      {0, 0, (std::size_t)c.time_dim, 2},        // temb
      {0, 0, (std::size_t)c.adaln_out(), 2},     // mod
      {0, 0, 2 * H, 2},                          // fmod
      {1, 0, 0, sizeof(int)},                    // adaln_idx
      {1, 0, 0, sizeof(int)},                    // tstep_idx
  };
}

}  // namespace

std::size_t
MetalMiniMaxH3Transformer::scratch_bytes(const Config& c, int seq, int n_text,
                                         int n_t, bool with_dequant,
                                         bool narrow_ff)
{
  if (seq <= 0) { return 0; }
  const std::size_t S = (std::size_t)seq;
  const std::size_t T = (std::size_t)(n_text > 0 ? n_text : 0);
  const std::size_t N = (std::size_t)(n_t > 0 ? n_t : 0);
  std::size_t total = 0;
  for (const ScratchItem& it : scratch_plan_(c, narrow_ff)) {
    total += (S * it.mul_seq + T * it.mul_text + N * it.mul_t) * it.esz;
  }
  if (with_dequant) {
    // _w_deq is grown to the WIDEST projection that actually reaches
    // gemm_mma_, and kept.
    //
    // Only the four whose M is the SEQUENCE length count. The two
    // modulation projections are wider on paper -- the per-block AdaLN is
    // [adaln_out, time_dim] = 496 MB against fc1's 294 -- but their M is
    // the distinct-TIMESTEP count, a handful of rows, so they never clear
    // the tile minimum and always run steel. Including them over-stated
    // this by 212 MB.
    //
    // (That is also the shape of a bug this file had: gemm_mma_ once took
    // the mma path for steel routes too, and the M=2 AdaLN really did
    // dequantize 496 MB to multiply two rows. If _mma_min_m is dropped
    // below the timestep count by hand, this estimate under-states again
    // -- scratch_estimate_matches_allocation pins the default.)
    const std::size_t H = (std::size_t)c.hidden;
    const std::size_t I = (std::size_t)c.inner();
    const std::size_t F = (std::size_t)c.ffn;
    const std::size_t cand[] = {
        3 * I * H,   // qkv
        H * I,       // out
        2 * F * H,   // fc1 -- the widest of these
        H * F,       // fc2
    };
    std::size_t widest = 0;
    for (std::size_t v : cand) { widest = std::max(widest, v); }
    total += widest * 2;
  }
  return total;
}

std::size_t
MetalMiniMaxH3Transformer::scratch_resident_bytes() const
{
  const metal_compute::SharedBuffer* all[] = {
      &_s.rcos, &_s.rsin, &_s.x, &_s.nm, &_s.qkv, &_s.attn,
      &_s.txt, &_s.temb, &_s.mod,
      &_s.fmod, &_s.adaln_idx, &_s.tstep_idx};
  std::size_t total = 0;
  for (const metal_compute::SharedBuffer* b : all) { total += b->byte_size(); }
  return total;
}

std::size_t
MetalMiniMaxH3Transformer::dequant_scratch_bytes() const
{
  return _w_deq.byte_size();
}

std::size_t
MetalMiniMaxH3Transformer::block_bytes_(const Block& b)
{
  const metal_compute::SharedBuffer* all[] = {
      &b.n1, &b.n2, &b.qn, &b.kn,
      &b.qkv.w, &b.qkv.b, &b.qkv.codes, &b.qkv.scales, &b.qkv.qbias,
      &b.out.w, &b.out.b, &b.out.codes, &b.out.scales, &b.out.qbias,
      &b.fc1.w, &b.fc1.b, &b.fc1.codes, &b.fc1.scales, &b.fc1.qbias,
      &b.fc2.w, &b.fc2.b, &b.fc2.codes, &b.fc2.scales, &b.fc2.qbias,
      &b.adaln.w, &b.adaln.b, &b.adaln.codes, &b.adaln.scales,
      &b.adaln.qbias};
  std::size_t n = 0;
  for (const metal_compute::SharedBuffer* p : all) { n += p->byte_size(); }
  return n;
}

// How much of the RESIDENT set is still in RAM.
//
// Every held block, but sampled inside each buffer: a pin either holds
// or it does not, so every 64th page finds it, and at 16 KB pages that
// is one byte of vector per megabyte examined. The pinned prefix counts
// too -- it is the part whose eviction hurts most, since nothing will
// ever re-admit it.
void
MetalMiniMaxH3Transformer::resident_pages_(std::size_t* examined,
                                           std::size_t* incore,
                                           std::size_t* paged_out) const
{
  *examined = 0;
  *incore = 0;
  if (paged_out != nullptr) { *paged_out = 0; }
  for (const Block& b : _blocks) {
    const metal_compute::SharedBuffer* all[] = {
        &b.qkv.w, &b.qkv.codes, &b.out.w, &b.out.codes,
        &b.fc1.w, &b.fc1.codes, &b.fc2.w, &b.fc2.codes,
        &b.adaln.w, &b.adaln.codes};
    for (const metal_compute::SharedBuffer* p : all) {
      if (p->byte_size() == 0) { continue; }
      // A WIRED BUFFER CANNOT HAVE LEFT RAM, so asking is spending the
      // walk to be told what mlock already guarantees. Skipped PER
      // BUFFER rather than per block, because wire_block_ stops at the
      // first refusal and leaves the rest of that block unwired -- the
      // remainder is exactly what still needs measuring.
      //
      // With everything wired `examined` stays 0, and the caller reads
      // that as "no evidence" rather than as a shortfall (it tests
      // examined > 0 first), which is the correct answer: there is
      // nothing this walk could have found.
      //
      // Worth the branch: the walk costs ~57 ms per 4.3 GB, so a fully
      // wired 35 GB stack would pay ~460 ms per look for a guaranteed
      // answer.
      if (p->is_wired()) { continue; }
      const auto r = p->page_residency(64);
      if (!r.valid) { continue; }
      *examined += r.examined;
      *incore += r.incore;
      if (paged_out != nullptr) { *paged_out += r.paged_out; }
    }
  }
}

bool
MetalMiniMaxH3Transformer::linear_matches_(WeightSet& ws,
                                           const std::string& nm,
                                           const Linear& l) const
{
  const MetalLlamaWeights& src = ws.src();
  const auto* si = src.info(nm + ".scales");
  const auto* ci = src.info(nm + ".weight");
  if (ci == nullptr) { return false; }
  const bool ckpt_quant = _quant_bits > 0 && si != nullptr &&
                          si->shape.size() == 2 && ci->shape.size() == 2;
  if (ckpt_quant != l.quantized) { return false; }
  if (!ckpt_quant) { return true; }
  // The same derivation linear_() uses, so the two cannot disagree
  // about what this Linear is.
  const long K = si->shape[1] * (long)_quant_group;
  const int bits = K > 0 ? (int)(ci->shape[1] * 32 / K) : 0;
  return ((bits == 8) ? 8 : 4) == l.bits;
}

std::size_t
MetalMiniMaxH3Transformer::resident_block_bytes_() const
{
  if (!_ws) { return 0; }
  // Block 0 stands for all of them: the stack is uniform, and the one
  // place it is not (a first or last block carrying extra) is what
  // BlockResidency's own largest-seen _block_hint corrects for once real
  // admissions start.
  const std::string pfx = blk_("blocks.", 0, "");
  std::size_t total = 0;
  for (const std::string& n : _ws->src().tensor_names()) {
    if (n.rfind(pfx, 0) != 0) { continue; }
    if (n.find("adaln") != std::string::npos) { continue; }
    const auto* ti = _ws->src().info(n);
    if (ti != nullptr) { total += (std::size_t)ti->nbytes; }
  }
  return total;
}

std::size_t
MetalMiniMaxH3Transformer::wire_block_(Block& b, bool on)
{
  metal_compute::SharedBuffer* all[] = {
      &b.n1, &b.n2, &b.qn, &b.kn,
      &b.qkv.w, &b.qkv.b, &b.qkv.codes, &b.qkv.scales, &b.qkv.qbias,
      &b.out.w, &b.out.b, &b.out.codes, &b.out.scales, &b.out.qbias,
      &b.fc1.w, &b.fc1.b, &b.fc1.codes, &b.fc1.scales, &b.fc1.qbias,
      &b.fc2.w, &b.fc2.b, &b.fc2.codes, &b.fc2.scales, &b.fc2.qbias,
      &b.adaln.w, &b.adaln.b, &b.adaln.codes, &b.adaln.scales,
      &b.adaln.qbias};
  auto* mgr = _mc != nullptr && _mc->session() != nullptr
                  ? _mc->session()->services()->generative_model_manager()
                  : nullptr;
  if (mgr == nullptr) { return 0; }
  std::size_t changed = 0;
  for (metal_compute::SharedBuffer* p : all) {
    if (p->byte_size() == 0 || p->is_wired() == on) { continue; }
    if (!on) {
      mgr->unwire_from_pool(*p);
      changed += p->byte_size();
      continue;
    }
    if (mgr->wire_into_pool(*p) == 0) {
      // The pool is full, or the box refused. STOP, and keep what is
      // already wired rather than unwinding it. A partly wired block is
      // partly protected, which is strictly better than none -- and
      // giving protection back on the way out means competing for it
      // again on the next block, against a pool that just said no.
      break;
    }
    changed += p->byte_size();
  }
  return changed;
}

std::vector<metal_compute::SharedBuffer*>
MetalMiniMaxH3Transformer::scratch_buffers_()
{
  // The ARENA, not its windows: subview() handles are explicitly not
  // wirable, and wiring the one allocation covers all six of them.
  return {&_s.rcos, &_s.rsin, &_s.x, &_s.nm, &_s.qkv, &_s.attn,
          &_s.txt, &_s.temb,
          &_s.mod, &_s.fmod, &_s.adaln_idx, &_s.tstep_idx, &_s.lora};
}

// Wire the TRUNK and the SCRATCH -- everything this model holds that is
// not a streamed block.
//
// They belong in the pool ahead of the blocks, not after: a resident
// block is an optimisation the model can shed and stream instead, while
// the scratch is what a forward cannot proceed without and the trunk is
// read on every block of every forward. Protecting the optional half
// first is how a run ends up with 32 GB of wired blocks beside an
// activation buffer the compressor is free to take.
std::size_t
MetalMiniMaxH3Transformer::wire_fixed_(bool on)
{
  auto* mgr = _mc != nullptr && _mc->session() != nullptr
                  ? _mc->session()->services()->generative_model_manager()
                  : nullptr;
  if (mgr == nullptr) { return 0; }
  std::size_t changed = 0;
  auto one = [&](metal_compute::SharedBuffer& b) {
    if (b.byte_size() == 0 || b.is_wired() == on) { return; }
    if (!on) { mgr->unwire_from_pool(b); changed += b.byte_size(); return; }
    changed += mgr->wire_into_pool(b);
  };
  for (metal_compute::SharedBuffer* b : scratch_buffers_()) { one(*b); }
  // The TRUNK: everything the weight set cached for this model, which
  // for a streaming DiT is the non-block tensors it holds for the whole
  // run. Read on every block of every forward and never shed, so it has
  // a better claim on the pool than any single resident block does.
  if (_ws) {
    _ws->for_each_weight([&](metal_compute::SharedBuffer& b) { one(b); });
  }
  return changed;
}

// Evict from the TAIL down, so what remains stays a contiguous prefix.
// In a cyclic scan every block is worth the same, so there is nothing
// cleverer to choose and a prefix keeps the bookkeeping trivial. The
// pinned prefix is never touched. Rescanning per call is 50 empty tests
// against a disk read, so the descending cursor this replaced bought
// nothing and could not be shared.
std::size_t
MetalMiniMaxH3Transformer::evict_tail_block_()
{
  const int floor = 0;
  for (int i = (int)_blocks.size() - 1; i >= floor; --i) {
    Block& b = _blocks[(std::size_t)i];
    const std::size_t n = block_bytes_(b);
    if (n == 0) { continue; }
    // Before the buffers go: give the wiring back. Dropping a wired
    // mapping would unwire it anyway, but doing it here keeps
    // _wired_bytes honest without having to infer it from destructors.
    const std::size_t unwired = wire_block_(b, false);
    _wired_bytes -= (unwired > _wired_bytes) ? _wired_bytes : unwired;
    b = Block{};
    // Taking one out of the PINNED prefix un-pins it: that prefix was
    // sized at load against what the box was believed to hold, and a
    // measurement saying its pages are no longer in RAM is that belief
    // being wrong. The forward decides resident-or-streamed by whether
    // the slot is EMPTY, not by this count, so it simply streams now.
    return n;
  }
  return 0;
}

std::size_t
MetalMiniMaxH3Transformer::release_resident_blocks(std::size_t bytes)
{
  const std::size_t freed =
      _resid.release(bytes, [this]() -> std::size_t {
        return evict_tail_block_();
      });
  if (freed > 0 && _mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalMiniMaxH3Transformer: released {} MB of resident blocks "
        "({} left)", freed >> 20, _resid.count()));
  }
  return freed;
}

bool
MetalMiniMaxH3Transformer::ensure_scratch_(int seq, int n_text, int n_t)
{
  if (_s.seq == seq && _s.n_text == n_text && _s.n_t == n_t) { return true; }
  const Config& c = _cfg;
  const std::size_t S = (std::size_t)seq, H = (std::size_t)c.hidden;
  const std::size_t I = (std::size_t)c.inner();
  auto mk = [&](std::size_t elems) { return _mc->make_shared_buffer(elems * 2); };
  Scratch s;
  s.seq = seq;
  s.n_text = n_text;
  s.n_t = n_t;
  const std::size_t rot_half = (std::size_t)(3 * c.rope_freq_dim);
  s.rcos = _mc->make_shared_buffer(S * rot_half * sizeof(float));
  s.rsin = _mc->make_shared_buffer(S * rot_half * sizeof(float));
  s.x    = mk(S * H);
  s.nm   = mk(S * H);
  s.qkv  = mk(S * 3 * I);
  // The arena, then five windows into it. subview() keeps the same
  // MTL::Buffer and carries the offset, so binding a window addresses
  // its slice with no copy -- and each window is CONTIGUOUS, which the
  // [rows, I] shapes require.
  s.attn = mk(S * attn_arena_elems_(c, !_ff_needs_wide));
  if (s.attn.empty()) { return false; }
  const std::size_t win = S * I * 2;            // bytes per window
  s.qh   = s.attn.subview(0 * win, win);
  s.kh   = s.attn.subview(1 * win, win);
  s.vh   = s.attn.subview(2 * win, win);
  s.oh   = s.attn.subview(3 * win, win);
  s.ob   = s.attn.subview(4 * win, win);
  // The [seq, 2*ffn] fc1 intermediate -- the single largest scratch this
  // model holds. A fused FF never writes it, so when every block is
  // interleaved it shrinks to what its OTHER user needs: tune_qmm_'s
  // destination, whose widest remaining shape is qkv's 3*inner. The
  // public scratch_bytes() still quotes the wide figure, so the stage's
  // preflight over-estimates rather than under-estimates.
  // ALIASED over the arena, from offset 0: every window above is dead by
  // the time fc1 writes this.
  const std::size_t ff_elems =
      S * (_ff_needs_wide ? 2 * (std::size_t)c.ffn : 3 * I);
  s.ff   = s.attn.subview(0, ff_elems * 2);
  // ...and `proj` past it, for the final layer alone.
  s.proj = s.attn.subview(ff_elems * 2, S * H * 2);
  // A runtime LoRA's [M, rank] intermediate. Rank 64 at 19k rows is
  // 2.4 MB -- the whole reason applying an adapter at runtime is
  // affordable is that this is the only extra buffer it needs.
  s.lora = _lora_max_rank > 0 ? mk(S * (std::size_t)_lora_max_rank)
                              : SharedBuffer{};
  s.txt  = mk((std::size_t)n_text * H);
  s.temb = mk((std::size_t)n_t * (std::size_t)c.time_dim);
  s.mod  = mk((std::size_t)n_t * (std::size_t)c.adaln_out());
  s.fmod = mk((std::size_t)n_t * 2 * H);
  s.adaln_idx = _mc->make_shared_buffer(S * sizeof(int));
  if (_nan_report.empty()) {
    _nan_report = _mc->make_shared_buffer(1024 * sizeof(unsigned));
  }
  s.tstep_idx = _mc->make_shared_buffer(S * sizeof(int));
  if (s.rcos.empty() || s.rsin.empty() || s.x.empty() || s.nm.empty() ||
      s.proj.empty() || s.qkv.empty() || s.qh.empty() || s.kh.empty() ||
      s.vh.empty() || s.oh.empty() || s.ob.empty() || s.ff.empty() ||
      s.txt.empty() || s.temb.empty() || s.mod.empty() || s.fmod.empty() ||
      s.adaln_idx.empty() || s.tstep_idx.empty()) {
    return false;
  }
  // UNWIRE THE OLD ONE FIRST. Destroying a wired buffer unwires it in
  // the kernel but does NOT decrement the pool's counter -- only
  // unwire_from_pool does -- so replacing the scratch on a geometry
  // change left the pool believing those bytes were still held. Every
  // change leaked a scratch's worth, and a pool that has lost budget to
  // bytes nothing holds wires less of what comes next: the resident set
  // shrinks for a reason nothing in the log names.
  //
  // Before the assignment rather than after, because after it the old
  // buffers are gone and there is nothing left to unwire.
  if (_wire_resident) {
    auto* mgr = _mc != nullptr && _mc->session() != nullptr
                    ? _mc->session()->services()->generative_model_manager()
                    : nullptr;
    if (mgr != nullptr) {
      for (metal_compute::SharedBuffer* b : scratch_buffers_()) {
        mgr->unwire_from_pool(*b);
      }
    }
  }
  _s = std::move(s);
  // The new scratch is wired by the forward, which calls wire_fixed_()
  // right after this -- see the note there on why the scratch goes into
  // the pool before any block does.
  return true;
}

// ---- forward ---------------------------------------------------------

MetalMiniMaxH3Transformer::Velocity
MetalMiniMaxH3Transformer::forward(const Step& in, std::string* err)
{
  auto fail = [&](std::string m) -> Velocity {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  if (in.layout == nullptr || in.timesteps == nullptr ||
      in.row_timestep_index == nullptr || in.video == nullptr ||
      in.text == nullptr) {
    return fail("incomplete Step");
  }
  const h3::PackedLayout& L = *in.layout;
  const int seq = L.seq_len;
  const int n_t = (int)in.timesteps->size();
  const int n_video = (int)L.video_indices.size();
  // TOTAL audio rows, reference and generated. `num_audio_rows` counts
  // the GENERATED ones alone, which is all a `t2va` / `fl2va` layout
  // has -- a `ref2va` request also carries a reference block per
  // audio-bearing reference, and every one of them is a row the caller
  // hands in and the head writes back.
  const int n_audio = (int)L.audio_indices.size();
  const int n_text = L.num_text_rows;
  const int n_cond = L.num_condition_rows;
  const int H = c.hidden, I = c.inner(), NH = c.n_heads, HD = c.head_dim;
  const int FF = c.ffn, VPE = c.video_patch_elems(), AC = c.audio_channels;
  // How the fused [rows, 3*I] qkv projection groups its output. Two
  // layouts ship, with the SAME name and the SAME shape, so nothing in
  // the checkpoint distinguishes them -- `qkv_per_head` records where
  // the weights came from (see the Config comment).
  //
  //   per head (MiniMaxAI's release):  [h0(q,k,v) | h1(q,k,v) | ...]
  //     a head is 3*HD apart, q/k/v are HD apart WITHIN it
  //   flat (Comfy-Org's conversion):   [all q | all k | all v]
  //     a head is HD apart, q/k/v are I apart
  const int QKV_HSTRIDE = c.qkv_per_head ? 3 * HD : HD;
  const int Q_OFF = 0;
  const int K_OFF = c.qkv_per_head ? HD : I;
  const int V_OFF = c.qkv_per_head ? 2 * HD : 2 * I;
  if (seq <= 0 || n_t <= 0 || n_text <= 0) { return fail("empty sequence"); }
  if ((int)in.row_timestep_index->size() != seq) {
    return fail("row_timestep_index does not match the layout");
  }
  if (n_audio > 0 && in.audio == nullptr) {
    return fail("the layout has audio rows but no audio latents");
  }
  if (in.video->byte_size() < (std::size_t)n_video * VPE * 2) {
    return fail("video rows are smaller than num_video_rows * patch elems");
  }
  if (in.text->byte_size() < (std::size_t)n_text * c.text_dim * 2) {
    return fail("text conditioning is smaller than num_text_rows * text_dim");
  }
  // What the FIRST forward spends before it can report a single block.
  // It is not small and it was not visible: the bar opens, the AdaLN
  // bake logs, and then nothing says anything until block 0. Every term
  // below is paid once per run, and each is a different thing to fix, so
  // they are reported apart rather than as one number.
  using PClock = std::chrono::steady_clock;
  const auto p_t0 = PClock::now();
  auto p_ms = [](PClock::time_point a, PClock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  if (!ensure_scratch_(seq, n_text, n_t)) {
    return fail("activation allocation failed (out of GPU memory)");
  }
  const auto p_scratch = PClock::now();
  // Re-arm growth for this forward. The per-forward flag is what stops
  // the budget being re-queried for every one of 50 blocks once the
  // answer is no; the RATCHET (_resid_ceiling) is what survives across
  // forwards, so a set that was cut back does not simply refill next
  // step. Scratch is allocated by now, so what the budget reports here
  // already has it subtracted.
  // Also where the box gets to answer back: nothing else in the process
  // can tell this that the OS has started swapping, and by the time it
  // has, every admission since is costing more than it saves.
  // ensure_scratch_() ran above, so the activations the stage told us to
  // reserve for are ALREADY allocated and already out of the budget read
  // on the next line. Declaring them keeps the gate from demanding room
  // for them twice; what is left of the reserve is the part still to be
  // found, which is the peers after the denoise (the VAE decode).
  // The scratch is allocated by now, so it can take its place in the
  // pool BEFORE this forward's block admissions start asking for room.
  // Order matters: the blocks are the shed-able half.
  if (_wire_resident) {
    // RETRY THE WIRED POOL, at the top of each forward.
    //
    // wire_into_pool() collapses the pool to what was granted when mlock
    // refuses, which is right for a box that is full and wrong for one
    // that was momentarily busy -- another process spiking during load.
    // Without a retry the second case holds a small resident set for the
    // whole schedule on the strength of one syscall: MEASURED, a 9 GB
    // pool that granted 5857 MB sat at 16 of 50 blocks for the rest of
    // the run.
    //
    // The flag stays set until a retry actually RAISES the budget, so a
    // box busy at forward 2 is still asked again at forward 5. A single
    // attempt would have spent itself against the same spike that
    // caused the refusal.
    //
    // GATED on the box having demonstrably freed a block's worth since
    // the refusal, so a genuinely full box is never asked: reopening the
    // ceiling makes wired_pool_can_take() pass, and the mlock behind it
    // would then fail and leave that block resident but UNWIRED -- one
    // per forward, exactly the state the wirable gate exists to avoid.
    // A peer holding wired memory shows up in this reading, since its
    // pages are unavailable while it holds them and return when it lets
    // go.
    //
    // A forward is the granularity because it is where the resident set
    // is reconsidered anyway, and the check costs one budget read when
    // the flag is clear.
    const std::size_t avail =
        _wire_retry ? _mc->memory_budget().available_physical : 0;
    if (_wire_retry && avail > _wire_retry_at + resident_block_bytes_()) {
      const std::size_t now = avail;
      auto* mgr = _mc->session() != nullptr &&
                          _mc->session()->services() != nullptr
                      ? _mc->session()->services()->generative_model_manager()
                      : nullptr;
      if (mgr != nullptr) {
        mgr->reopen_wired_pool();
        const std::size_t lim = mgr->wired_pool_limit();
        const std::size_t used = mgr->wired_pool_used();
        const std::size_t room = lim > used ? lim - used : 0;
        // Only ever RAISED here. The budget also bounds what this model
        // has already wired, and lowering it below `_wired_bytes` would
        // read as an over-spend that nothing can give back.
        //
        // CLAMPED TO THE POOL, belt and braces. The arithmetic already
        // gives `_wired_bytes + (lim - used) <= lim` because this
        // model's wired bytes are part of the manager's `used` -- but
        // that holds only while the two counters agree, and the failure
        // mode if they ever drift is not a slow run. Wired memory is the
        // one allocation the kernel cannot reclaim, so an over-budget
        // here panics the box rather than degrading it. One min() is a
        // cheap way to never find out.
        std::size_t want = _wired_bytes + room;
        if (want > lim) { want = lim; }
        if (want > _wire_budget) {
          if (_mc->session() != nullptr) {
            _mc->session()->log_debug(fmt(
                "MetalMiniMaxH3Transformer: retrying the wired pool -- "
                "budget {} -> {} MB", _wire_budget >> 20, want >> 20));
          }
          _wire_budget = want;
          _wire_retry  = false;
          // The residency policy stopped growing when the budget ran
          // out, and it cannot see that the budget moved. This is the
          // same "the ground moved" case the AdaLN bake uses.
          _resid.note_landscape_changed();
        } else {
          // The ceiling did not move after all -- the pool is full
          // rather than the box being busy. Re-arm against the CURRENT
          // reading so the next look asks about a fresh block's worth.
          _wire_retry_at = now;
        }
      }
    }
    wire_fixed_(true);
  }
  _resid.note_reserve_allocated(
      scratch_bytes(c, seq, n_text, n_t, uses_matrix_cores()));
  const auto mbudget = _mc->memory_budget();
  _resid.begin_forward(mbudget, [this]() -> std::size_t {
    return evict_tail_block_();
  });
  // Then the measurement that actually finds the limit: are the blocks
  // we kept still in RAM? Anything less means a pin has failed and this
  // box holds less than it was asked to, so one block goes back.
  //
  // Gated on the cheap signal -- our OWN compressed footprint moving --
  // because the walk costs ~57 ms per 4.3 GB and a healthy run would pay
  // it every step to be told nothing.
  bool shortfall = false;
  if (_resid.count() > 0 &&
      _resid.self_compression_grew(mbudget.self_compressed)) {
    std::size_t examined = 0, incore = 0, paged_out = 0;
    resident_pages_(&examined, &incore, &paged_out);
    if (examined > 0 && incore < examined) {
      shortfall = true;
      std::size_t freed = _resid.note_weight_residency(
          examined, incore, [this]() -> std::size_t {
            return evict_tail_block_();
          });
      // Nothing left outside the pinned prefix and the pages are still
      // leaving RAM: the prefix itself is what does not fit, so give one
      // of it back rather than sit in the thrash it was meant to prevent.
      if (_mc->session() != nullptr) {
        _mc->session()->log_normal(fmt(
            "MetalMiniMaxH3Transformer: resident weights are only {}% in "
            "RAM ({} of {} sampled pages paged out, {} MB wired) -- "
            "released {} MB, now {} blocks resident",
            (int)(100.0 * (double)incore / (double)examined),
            paged_out, examined, _wired_bytes >> 20, freed >> 20,
            _resid.count()));
      }
    }
  }
  // Nothing of ours had left RAM this step -- either the walk said so or
  // there was no compression to make it worth walking. Enough of these
  // in a row lifts the ratchet by a block, so a shed taken during a
  // momentary squeeze (the AdaLN bake is the worst of them, and it ENDS
  // by handing back 13 GB) is not the last word on the whole run.
  if (!shortfall) { _resid.note_healthy_forward(); }
  const auto p_resid = PClock::now();
  Scratch& s = _s;
  // Measure the GEMM tile for THIS sequence length, before the stream
  // opens -- the tuner needs its own command buffers, and the answer has
  // to be in hand before the first block encodes.
  tune_qmm_(seq);
  const auto p_tune = PClock::now();

  Velocity out;
  out.video = _mc->make_shared_buffer((std::size_t)n_video * VPE * 2);
  if (n_audio > 0) {
    out.audio = _mc->make_shared_buffer((std::size_t)n_audio * AC * 2);
  }
  if (out.video.empty() || (n_audio > 0 && out.audio.empty())) {
    return fail("velocity allocation failed");
  }

  build_rope_(L, s.rcos, s.rsin);
  // A baked schedule replaces every AdaLN projection with a slice of a
  // precomputed table. It needs the step to say WHICH slice, so a Step
  // that does not name its schedule entry falls back to computing the
  // modulation, whatever the model holds.
  const bool baked = adaln_baked() && in.schedule_index >= 0 &&
                     (std::size_t)in.schedule_index < _adaln_row0.size() &&
                     _adaln_nt[(std::size_t)in.schedule_index] == n_t;
  if (adaln_baked() && !baked) {
    return fail(fmt("schedule_index {} does not name a baked step with {} "
                    "timesteps (the schedule has {})", in.schedule_index,
                    n_t, _adaln_row0.size())());
  }
  // ALWAYS, even baked. Baking removes the AdaLN projection's need for
  // it, but not the adaln ADAPTER's: that is applied per forward (so
  // lora_scale stays live) and reads temb as its input. Skipping it here
  // left the adapter projecting from an uninitialized scratch -- which
  // produced plausible numbers, not a crash, and showed up only as the
  // baked and unbaked paths disagreeing once an adapter was attached.
  // It costs a host pass over n_t rows, which is nothing.
  if (!time_embed_(*in.timesteps, s.temb)) {
    return fail("timestep embedding failed");
  }
  {
    const std::vector<int> adaln =
        h3::build_adaln_indices(L, *in.row_timestep_index);
    if ((int)adaln.size() != seq) { return fail("adaln index build failed"); }
    std::memcpy(s.adaln_idx.contents(), adaln.data(), (std::size_t)seq * 4);
    std::memcpy(s.tstep_idx.contents(), in.row_timestep_index->data(),
                (std::size_t)seq * 4);
  }

  const float scale = 1.0f / std::sqrt((float)HD);
  // The NAX kernel's tiles: bq=64/bk=32 against the ALU kernel's 32/16.
  // These feed the param block AND the dispatch grid, so they have to move
  // together with the function -- which is the whole of the difference.
  //
  // They are resolved BEFORE the param block is filled, and _attn_nax is
  // cleared here if the NAX function does not build, because a param block
  // filled for one tile and a kernel compiled for the other is not a
  // fallback -- it is a wrong answer. Falling back to the ALU steel kernel
  // (rather than letting the validity check below drop through to the
  // scalar sdpa) matters: the scalar path is O(seq^2) at a few percent of
  // peak, so at this model's sequences it is not a slow path but a hang.
  if (_attn_nax && (_attn_seq != seq || _attn_text != n_text)) {
    metal_compute::FunctionConstants probe;
    probe.set_bool(200, (seq % 64) == 0).set_bool(201, (seq % 32) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false);
    if (!_lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", probe).valid()) {
      _attn_nax = false;
    }
  }
  const int A_BQ = _attn_nax ? 64 : 32;
  const int A_BK = _attn_nax ? 32 : 16;
  bool use_steel = _steel_ok;
  // Read ONCE per forward: the A/B setter can change it between two, and
  // the cached AttnParams below are tagged with the value they were
  // filled for.
  const int fused_attn = _fused_attn;
  // The FUNCTIONS depend on the sequence lengths alone; the PARAMS also
  // depend on which layout attention is reading, and the A/B setter
  // flips that between forwards. Kept apart so toggling the layout does
  // not rebuild two pipeline states -- an A/B that pays a rebuild in one
  // arm and not the other is measuring the rebuild.
  const bool attn_dirty = _attn_seq != seq || _attn_text != n_text;
  if (use_steel && (attn_dirty || _attn_fused != fused_attn)) {
    // C++ mirror of mlx::steel::AttnParams, as in the sibling DiTs. Two
    // shapes only, and both are SQUARE: this model has no
    // cross-attention, so the refiner attends over the text rows and
    // every main block over the whole packed sequence.
    struct P {
      int B, H, D, qL, kL, gqa_factor;
      float scale;
      int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
      std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
    };
    auto fill = [&](SharedBuffer& pb, int qL) {
      auto* p = static_cast<P*>(pb.contents());
      p->B = 1; p->H = NH; p->D = HD;
      p->qL = qL; p->kL = qL;
      p->gqa_factor = 1; p->scale = scale;
      p->NQ = (qL + A_BQ - 1) / A_BQ; p->NK = (qL + A_BK - 1) / A_BK;
      p->NQ_aligned = qL / A_BQ; p->NK_aligned = qL / A_BK;
      p->qL_rem = qL - p->NQ_aligned * A_BQ;
      p->kL_rem = qL - p->NK_aligned * A_BK;
      p->qL_off = 0;
      // Head-major [H, qL, D], the layout the transposes produce.
      const std::int64_t hm[3] = {(std::int64_t)NH * qL * HD,
                                  (std::int64_t)qL * HD, HD};
      if ((fused_attn & kFusedAttnQkv) != 0) {
        // Q, K and V ARE the fused projection, addressed in place: a row
        // is 3*I apart, a head QKV_HSTRIDE apart within the row, and the
        // head_dim channels are contiguous, which is the one thing the
        // BlockLoader actually requires. Which of the two groupings the
        // checkpoint uses is already in QKV_HSTRIDE, so both are covered
        // by the same three numbers. The q/k/v base offsets are applied
        // at the BIND, not here -- steel has no field for them.
        p->Q_strides[0] = (std::int64_t)qL * 3 * I;
        p->Q_strides[1] = QKV_HSTRIDE;
        p->Q_strides[2] = 3 * I;
      } else {
        for (int i = 0; i < 3; ++i) { p->Q_strides[i] = hm[i]; }
      }
      for (int i = 0; i < 3; ++i) {
        p->K_strides[i] = p->Q_strides[i];
        p->V_strides[i] = p->Q_strides[i];
      }
      if ((fused_attn & kFusedAttnOut) != 0) {
        // O goes straight back into [rows, I], head h at column h*HD --
        // the layout the out projection reads. Unlike the q/k/v side
        // this asks nothing of the loaders: it is the kernel's final
        // store, of exactly the bytes it was storing anyway.
        p->O_strides[0] = (std::int64_t)qL * I;
        p->O_strides[1] = HD;
        p->O_strides[2] = I;
      } else {
        for (int i = 0; i < 3; ++i) { p->O_strides[i] = hm[i]; }
      }
    };
    fill(_attn_p_main, seq);
    fill(_attn_p_text, n_text);
    _attn_fused = fused_attn;
    auto build = [&](int qL) {
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (qL % A_BQ) == 0).set_bool(201, (qL % A_BK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      return _attn_nax
                 ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
                 : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
    };
    if (attn_dirty) {
      _fn_attn_main = build(seq);
      _fn_attn_text = build(n_text);
      _attn_seq = seq;
      _attn_text = n_text;
    }
  }
  if (use_steel) {
    use_steel = _fn_attn_main.valid() && _fn_attn_text.valid();
  }
  // The scalar fallback has no strides, so a build that fails to make
  // the steel functions falls all the way back -- head-major buffers,
  // transposes and all. That is why the arena still carries them.
  const int  fused     = use_steel ? fused_attn : 0;
  const bool fused_qkv = (fused & kFusedAttnQkv) != 0;
  const bool fused_out = (fused & kFusedAttnOut) != 0;

  // ---- env-gated per-section GPU timing (VPIPE_H3_DIT_PROFILE) --------
  // The whole forward is ONE deferred stream, so there is nothing to
  // time inside it without splitting: each psplit() ends the encoder,
  // commits, waits and charges the slice to a bucket. That serializes
  // the GPU and inflates the total, so read the SHARE, not the sum.
  const bool prof = std::getenv("VPIPE_H3_DIT_PROFILE") != nullptr;
  double t_in = 0, t_adaln = 0, t_qkv = 0, t_prep = 0, t_attn = 0,
         t_oproj = 0, t_ff = 0, t_elt = 0, t_final = 0;

  // ---- env-gated streaming split (VPIPE_H3_STREAM_PROFILE) ------------
  // How much of a streamed block's wall time is the DISK read and how
  // much is the GPU. Unlike the section profile above this adds NO
  // barriers: the streamed path already loads serially and already
  // commits-and-waits per block (it must, before the weights go away),
  // so both ends are on the critical path whether or not we time them.
  //
  // It answers one question -- what a weight PREFETCH could hide. The
  // read is hideable under the GPU only up to min(read, gpu), so the
  // ceiling on prefetch is read/(read+gpu) and there is no point
  // building it until that number is worth having. Both terms move with
  // the machine: block bytes with the quantization, read rate with the
  // storage (internal NVMe, Thunderbolt, USB), gpu with the geometry.
  const bool sprof = std::getenv("VPIPE_H3_STREAM_PROFILE") != nullptr;
  double sp_read_ms = 0, sp_gpu_ms = 0;
  std::size_t sp_read_bytes = 0;
  int sp_blocks = 0;
  const double sp_alloc0 = sprof && _ws ? _ws->stats().streamed_alloc_ms : 0.0;
  const double sp_fetch0 = sprof && _ws ? _ws->stats().streamed_fetch_ms : 0.0;
  const auto t_begin = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point mark = t_begin;

  if (!_prologue_logged && _mc->session() != nullptr) {
    _prologue_logged = true;
    const auto p_end = PClock::now();
    _mc->session()->log_normal(fmt(
        "[h3-dit] first forward at {} rows: {:.0f} ms before block 0 "
        "(scratch {:.0f}, residency+wire {:.0f}, gemm autotune {:.0f}, "
        "rope/temb/attn setup {:.0f})", seq, p_ms(p_t0, p_end),
        p_ms(p_t0, p_scratch), p_ms(p_scratch, p_resid),
        p_ms(p_resid, p_tune), p_ms(p_tune, p_end)));
  }

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto psplit = [&](double& acc) {
      if (!prof) { return; }
      enc.end();
      stream.commit().wait();
      acc += std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - mark).count();
      stream = _mc->make_command_stream();
      enc = stream.begin_compute();
      mark = std::chrono::steady_clock::now();
    };
    // Per-modality row RMS of the packed activation, read back mid-stream.
    // The three modalities occupy disjoint row ranges, so this says
    // whether the text rows carry signal at all and whether the media
    // rows respond to them. Costs a full GPU drain, so it is opt-in.
    const bool xprobe = std::getenv("VPIPE_H3_XPROBE") != nullptr;
    auto xdump = [&](const char* where) {
      if (!xprobe) { return; }
      enc.end();
      stream.commit().wait();
      const auto* p = static_cast<const std::uint16_t*>(s.x.contents());
      auto band = [&](int r0, int n) {
        double a = 0.0;
        for (int r = 0; r < n; ++r) {
          for (int k = 0; k < H; ++k) {
            const std::uint32_t u =
                (std::uint32_t)p[(std::size_t)(r0 + r) * H + k] << 16;
            float v;
            std::memcpy(&v, &u, 4);
            a += (double)v * v;
          }
        }
        return n > 0 ? std::sqrt(a / ((double)n * H)) : 0.0;
      };
      if (_mc->session() != nullptr) {
        _mc->session()->info(fmt(
            "h3-xprobe {:22s} text[0,{}) {:.5f}  audio[{},+{}) {:.5f}  "
            "video[{},+{}) {:.5f}", where, n_text, band(0, n_text),
            L.audio_start, n_audio, band(L.audio_start, n_audio),
            L.video_start, L.num_video_rows,
            band(L.video_start, L.num_video_rows)));
      }
      stream = _mc->make_command_stream();
      enc = stream.begin_compute();
      mark = std::chrono::steady_clock::now();
    };
    // The same read-back over an ARBITRARY intermediate, so one block's
    // internals can be walked GEMM by GEMM. `bprobe` is set only for the
    // block being dissected -- every call drains the GPU.
    bool bprobe = false;
    // Which block the tripwire tags its findings with. Set in the loop
    // for the same reason bprobe is: the block lambda is defined before
    // the loop variable exists.
    int trip_blk = -1;
    auto bdump = [&](const char* where, const SharedBuffer& buf, int width) {
      if (!bprobe || _mc->session() == nullptr) { return; }
      enc.end();
      stream.commit().wait();
      const auto* p = static_cast<const std::uint16_t*>(buf.contents());
      auto band = [&](int r0, int n) {
        double a = 0.0;
        for (int r = 0; r < n; ++r) {
          for (int k = 0; k < width; ++k) {
            const std::uint32_t u =
                (std::uint32_t)p[(std::size_t)(r0 + r) * width + k] << 16;
            float v;
            std::memcpy(&v, &u, 4);
            a += (double)v * v;
          }
        }
        return n > 0 ? std::sqrt(a / ((double)n * width)) : 0.0;
      };
      _mc->session()->info(fmt(
          "h3-bprobe {:14s} w{:5}  text {:12.5f}  audio {:12.5f}  "
          "video {:12.5f}", where, width, band(0, n_text),
          band(L.audio_start, n_audio),
          band(L.video_start, L.num_video_rows)));
      stream = _mc->make_command_stream();
      enc = stream.begin_compute();
      mark = std::chrono::steady_clock::now();
    };
    // THE TRIPWIRE. One dispatch per checked tensor, no drain, so it can
    // run on EVERY op of EVERY block without changing the timing that
    // this bug turned out to depend on. `tag` is block * 16 + op, and the
    // report is read once after the forward commits.
    static const bool ktrip =
        std::getenv("VPIPE_H3_NAN_TRIP") != nullptr;
    if (ktrip && !_nan_report.empty()) {
      std::memset(_nan_report.contents(), 0, 1024 * sizeof(unsigned));
    }
    auto trip = [&](int blk, int op, const SharedBuffer& b,
                    std::size_t n) {
      if (!ktrip || !_fn_nan_trip.valid() || _nan_report.empty() ||
          b.empty() || n == 0) {
        return;
      }
      enc.set_function(_fn_nan_trip);
      enc.set_buffer(0, b);
      enc.set_buffer(1, _nan_report);
      enc.set_constant(2, (unsigned)n);
      enc.set_constant(3, (unsigned)(blk * 16 + op));
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    };

    auto rms = [&](const SharedBuffer& x, std::size_t x_off,
                   const SharedBuffer& g, const SharedBuffer& y,
                   std::size_t y_off, int rows, int width, float eps) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, x, x_off * 2);
      enc.set_buffer(1, g);
      enc.set_buffer(2, y, y_off * 2);
      enc.set_constant(3, width);
      enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
    };
    // Where this block reads its modulation from. Unbaked that is always
    // the s.mod scratch the AdaLN GEMM just wrote; baked it is a SLICE of
    // the block's table, bound at an offset so nothing is copied. The
    // block lambda sets both before using them.
    const SharedBuffer* mod_buf = &s.mod;
    std::size_t mod_off = 0;
    auto modulate = [&](const SharedBuffer& x, const SharedBuffer& mod,
                        std::size_t mod_byte_off,
                        const SharedBuffer& idx, const SharedBuffer& y,
                        int rows, int stride, int scale_off, int shift_off) {
      enc.set_function(_fn_modulate);
      enc.set_buffer(0, x); enc.set_buffer(1, mod, mod_byte_off);
      enc.set_buffer(2, idx);
      enc.set_buffer(3, y);
      enc.set_constant(4, H);
      enc.set_constant(5, stride);
      enc.set_constant(6, scale_off);
      enc.set_constant(7, shift_off);
      enc.set_constant(8, rows * H);
      enc.dispatch({(unsigned)(rows * H), 1, 1}, {256, 1, 1});
    };
    auto gated = [&](const SharedBuffer& x, const SharedBuffer& sub, int rows,
                     int gate_off) {
      enc.set_function(_fn_gated);
      enc.set_buffer(0, x); enc.set_buffer(1, *mod_buf, mod_off * 2);
      enc.set_buffer(2, s.adaln_idx); enc.set_buffer(3, sub);
      enc.set_constant(4, H);
      enc.set_constant(5, 6 * H);
      enc.set_constant(6, gate_off);
      enc.set_constant(7, rows * H);
      enc.dispatch({(unsigned)(rows * H), 1, 1}, {256, 1, 1});
    };
    auto residual = [&](const SharedBuffer& x, const SharedBuffer& sub,
                        int rows) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, x); enc.set_buffer(1, sub); enc.set_buffer(2, x);
      enc.set_constant(3, rows * H);
      enc.dispatch({(unsigned)(rows * H), 1, 1}, {256, 1, 1});
    };
    // q, k and v all come straight out of the ONE fused [rows, 3*I]
    // projection. `rot` 0 makes this a pure strided transpose, which is
    // what V needs and what the refiner needs for all three.
    auto trope = [&](const SharedBuffer& dst, int rows, int off, int rot) {
      enc.set_function(_fn_trope);
      enc.set_buffer(0, s.qkv); enc.set_buffer(1, dst);
      enc.set_buffer(2, s.rcos); enc.set_buffer(3, s.rsin);
      enc.set_constant(4, NH);
      enc.set_constant(5, rows);
      enc.set_constant(6, HD);
      enc.set_constant(7, rot);
      enc.set_constant(8, 3 * I);
      enc.set_constant(9, off);
      enc.set_constant(10, QKV_HSTRIDE);   // grouping, see below
      enc.dispatch({(unsigned)HD, (unsigned)rows, (unsigned)NH},
                   {(unsigned)HD, 1, 1});
    };
    // The fused path's rope: the same rotation, written back over the
    // values it read. No transpose, and V -- which does not rotate --
    // needs no pass at all.
    auto rope_ip = [&](int rows, int off, int rot) {
      if (rot == 0) { return; }
      enc.set_function(_fn_rope_ip);
      enc.set_buffer(0, s.qkv);
      enc.set_buffer(1, s.rcos); enc.set_buffer(2, s.rsin);
      enc.set_constant(3, NH);
      enc.set_constant(4, rows);
      enc.set_constant(5, HD);
      enc.set_constant(6, rot);
      enc.set_constant(7, 3 * I);
      enc.set_constant(8, off);
      enc.set_constant(9, QKV_HSTRIDE);
      enc.dispatch({(unsigned)HD, (unsigned)rows, (unsigned)NH},
                   {(unsigned)HD, 1, 1});
    };
    auto qk_norm = [&](const SharedBuffer& g, int rows, int off) {
      enc.set_function(_fn_rms_heads);
      enc.set_buffer(0, s.qkv); enc.set_buffer(1, g);
      enc.set_constant(2, rows);
      enc.set_constant(3, NH);
      enc.set_constant(4, HD);
      enc.set_constant(5, 3 * I);
      enc.set_constant(6, off);
      enc.set_constant(7, c.qk_norm_eps);
      enc.set_constant(8, QKV_HSTRIDE);    // grouping, see below
      enc.dispatch({32, (unsigned)(rows * NH), 1}, {32, 1, 1});
    };
    auto attn = [&](int rows, bool main) {
      if (use_steel) {
        enc.set_function(main ? _fn_attn_main : _fn_attn_text);
        if (fused_qkv) {
          enc.set_buffer(0, s.qkv, (std::size_t)Q_OFF * 2);
          enc.set_buffer(1, s.qkv, (std::size_t)K_OFF * 2);
          enc.set_buffer(2, s.qkv, (std::size_t)V_OFF * 2);
        } else {
          enc.set_buffer(0, s.qh); enc.set_buffer(1, s.kh);
          enc.set_buffer(2, s.vh);
        }
        enc.set_buffer(3, fused_out ? s.ob : s.oh);
        enc.set_buffer(4, main ? _attn_p_main : _attn_p_text);
        enc.dispatch({32 * (unsigned)((rows + A_BQ - 1) / A_BQ),
                      4 * (unsigned)NH, 1}, {32, 4, 1});
        return;
      }
      enc.set_function(_fn_sdpa);
      enc.set_buffer(0, s.qh); enc.set_buffer(1, s.kh);
      enc.set_buffer(2, s.vh); enc.set_buffer(3, s.oh);
      enc.set_constant(4, scale); enc.set_constant(5, rows);
      enc.set_constant(6, HD); enc.set_constant(7, NH);
      enc.set_constant(8, NH); enc.set_constant(9, rows);
      enc.set_constant(10, rows);
      enc.dispatch({32, (unsigned)NH, (unsigned)rows}, {32, 1, 1});
    };

    // One block. `modulated` selects the 50 main blocks (per-row AdaLN
    // and rope) from the 2 refiner blocks (neither).
    auto block = [&](const Block& b, const SharedBuffer& x, int rows,
                     bool modulated, const BlockLora* lo) {
      rms(x, 0, b.n1, s.nm, 0, rows, H, c.norm_eps);
      bdump("n1", s.nm, H);
      trip(trip_blk, 0, s.nm, (std::size_t)rows * H);
      if (modulated) {
        modulate(s.nm, *mod_buf, mod_off * 2, s.adaln_idx, s.nm, rows,
                 6 * H, H, 0);
      }
      bdump("n1+mod", s.nm, H);
      psplit(t_elt);
      gemm_(enc, s.nm, 0, b.qkv, s.qkv, 0, rows, 3 * I, H,
            lo != nullptr ? &lo->qkv : nullptr);
      bdump("qkv", s.qkv, 3 * I);
      trip(trip_blk, 2, s.qkv, (std::size_t)rows * 3 * I);
      psplit(t_qkv);
      // Per-HEAD RMS over head_dim, in place on the fused buffer, BEFORE
      // rope -- the reference's order. See QKV_HSTRIDE / Q_OFF above for
      // the two groupings the fused projection ships in.
      qk_norm(b.qn, rows, Q_OFF);
      qk_norm(b.kn, rows, K_OFF);
      if (fused_qkv) {
        rope_ip(rows, Q_OFF, modulated ? c.rope_rot() : 0);
        rope_ip(rows, K_OFF, modulated ? c.rope_rot() : 0);
      } else {
        trope(s.qh, rows, Q_OFF, modulated ? c.rope_rot() : 0);
        trope(s.kh, rows, K_OFF, modulated ? c.rope_rot() : 0);
        trope(s.vh, rows, V_OFF, 0);
      }
      psplit(t_prep);
      attn(rows, modulated);
      psplit(t_attn);
      if (!fused_out) {
        // Head-major [H, rows, D] back to the row-major [rows, I] the
        // out projection reads. The fused path had steel store it that
        // way to begin with.
        enc.set_function(_fn_transpose);
        enc.set_buffer(0, s.oh); enc.set_buffer(1, s.ob);
        enc.set_constant(2, NH); enc.set_constant(3, rows);
        enc.set_constant(4, HD);
        enc.dispatch({(unsigned)HD, (unsigned)rows, (unsigned)NH},
                     {(unsigned)HD, 1, 1});
      }
      bdump("attn_out", s.ob, I);
      trip(trip_blk, 3, s.ob, (std::size_t)rows * I);
      gemm_(enc, s.ob, 0, b.out, s.nm, 0, rows, H, I,
            lo != nullptr ? &lo->out : nullptr);
      bdump("o_proj", s.nm, H);
      trip(trip_blk, 4, s.nm, (std::size_t)rows * H);
      psplit(t_oproj);
      if (modulated) { gated(x, s.nm, rows, 2 * H); }
      else           { residual(x, s.nm, rows); }
      bdump("x_after_attn", x, H);
      trip(trip_blk, 5, x, (std::size_t)rows * H);

      rms(x, 0, b.n2, s.nm, 0, rows, H, c.norm_eps);
      if (modulated) {
        modulate(s.nm, *mod_buf, mod_off * 2, s.adaln_idx, s.nm, rows,
                 6 * H, 4 * H, 3 * H);
      }
      bdump("n2+mod", s.nm, H);
      psplit(t_elt);
      // s.qkv is free by now and is the only scratch wide enough to take
      // the activation, whichever path writes it.
      if (b.fc1.gu_inter) {
        // One GEMM: the epilogue holds each (gate, up) pair in one
        // accumulator fragment and stores silu(gate)*up straight into
        // s.qkv. No [rows, 2*ffn] intermediate is written and none is
        // read back -- which is the whole saving, since the arithmetic is
        // identical either way.
        const bool bm64 = _qmm_tile >= 1 &&
                          (b.fc1.bits == 8 ? _fn_qmm_swiglu8_bm64.valid()
                                           : _fn_qmm_swiglu4_bm64.valid());
        enc.set_function(b.fc1.bits == 8
                             ? (bm64 ? _fn_qmm_swiglu8_bm64 : _fn_qmm_swiglu8)
                             : (bm64 ? _fn_qmm_swiglu4_bm64
                                     : _fn_qmm_swiglu4));
        enc.set_buffer(0, b.fc1.codes); enc.set_buffer(1, b.fc1.scales);
        enc.set_buffer(2, b.fc1.qbias); enc.set_buffer(3, s.nm);
        enc.set_buffer(4, s.qkv);
        enc.set_constant(5, H);
        enc.set_constant(6, 2 * FF);     // the FUSED width, not the output
        enc.set_constant(7, rows);
        const int bm = bm64 ? 64 : 32;
        enc.dispatch({(unsigned)(((2 * FF + 31) / 32) * 32),
                      (unsigned)(((rows + bm - 1) / bm) * 2), 2}, {32, 2, 2});
      } else {
        gemm_(enc, s.nm, 0, b.fc1, s.ff, 0, rows, 2 * FF, H,
              lo != nullptr ? &lo->fc1 : nullptr);
        bdump("fc1", s.ff, 2 * FF);
      trip(trip_blk, 6, s.ff, (std::size_t)rows * 2 * FF);
        // fc1 is FUSED [gate | up], GATE first -- the diffusers SwiGLU
        // convention, not the llama one the rest of this tree follows.
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, s.ff); enc.set_buffer(1, s.qkv);
        enc.set_constant(2, rows);
        enc.set_constant(3, FF);
        enc.dispatch({(unsigned)(rows * FF), 1, 1}, {256, 1, 1});
      }
      bdump("swiglu", s.qkv, FF);
      trip(trip_blk, 7, s.qkv, (std::size_t)rows * FF);
      gemm_(enc, s.qkv, 0, b.fc2, s.nm, 0, rows, H, FF,
            lo != nullptr ? &lo->fc2 : nullptr);
      bdump("fc2", s.nm, H);
      trip(trip_blk, 8, s.nm, (std::size_t)rows * H);
      psplit(t_ff);
      if (modulated) { gated(x, s.nm, rows, 5 * H); }
      else           { residual(x, s.nm, rows); }
      bdump("x_after_ff", x, H);
      trip(trip_blk, 9, x, (std::size_t)rows * H);
      psplit(t_elt);
    };

    // ---- 1. project each modality into the packed sequence -----------
    // Every modality's rows are CONTIGUOUS in the FL2VA layout, so the
    // reference's three index_copy scatters are plain destination
    // offsets here. Video is the one split: its rows are the
    // conditioning block AND the target block, which are not adjacent.
    // Each modality arrives as one buffer in ITS OWN row order and is
    // scattered into the packed sequence run by run. `t2va` / `fl2va`
    // give video two runs (the conditioning block and the target) and
    // audio one; `ref2va` gives one run per reference plus the target,
    // and interleaves the two modalities -- so the destination offsets
    // are read from the layout rather than derived from a start and a
    // count. Every run is still CONTIGUOUS, so this is the same number
    // of GEMMs it always was, just addressed properly.
    {
      std::size_t src = 0;
      for (const h3::RowRun& r : L.video_runs) {
        if (r.count > 0) {
          gemm_(enc, *in.video, src * VPE, _video_patch, s.x,
                (std::size_t)r.start * H, r.count, H, VPE);
        }
        src += (std::size_t)r.count;
      }
    }
    if (in.audio != nullptr) {
      std::size_t src = 0;
      for (const h3::RowRun& r : L.audio_runs) {
        if (r.count > 0) {
          gemm_(enc, *in.audio, src * AC, _audio_patch, s.x,
                (std::size_t)r.start * H, r.count, H, AC);
        }
        src += (std::size_t)r.count;
      }
    }
    gemm_(enc, *in.text, 0, _cond_proj, s.txt, 0, n_text, H, c.text_dim);
    psplit(t_in);

    // ---- 2. the text refiner -----------------------------------------
    for (std::size_t ri = 0; ri < _refiner.size(); ++ri) {
      block(_refiner[ri], s.txt, n_text, false,
            ri < _lora_refiner.size() ? &_lora_refiner[ri] : nullptr);
    }
    // Its final norm writes straight into the packed sequence's text
    // rows, which are rows [0, n_text) -- no separate copy.
    rms(s.txt, 0, _refiner_final_norm, s.x, 0, n_text, H, c.final_norm_eps);
    psplit(t_in);
    xdump("after refiner write");

    // ---- weight prefetch (VPIPE_H3_NO_PREFETCH=1 disables) -----------
    // The streamed path reads block L, encodes it, then commits AND
    // WAITS before the weights free -- so the disk and the GPU take
    // strict turns. MEASURED at 960x544: ~160 ms of read against ~2.6 s
    // of GPU per block, all of it real disk (1.4-1.6 GB/s, on EVERY
    // pass, because 8.9 GB of streamed weights cannot stay cached).
    //
    // Issuing block L+1's read between the commit and the wait overlaps
    // it with GPU work already in flight. Nothing about the encoded work
    // changes -- same bytes, same order, same output. This is
    // scheduling, not arithmetic.
    //
    // Depth is structurally ONE: a single outstanding read into a single
    // spare Block. That is what keeps a tight box safe -- no queue can
    // run away, and the extra live memory is exactly one block.
    //
    // DECLARATION ORDER MATTERS. Members destroy in reverse, so `fut`
    // goes FIRST and its destructor blocks until the reader finishes --
    // which is what keeps `blk`, the buffer that reader is writing into,
    // alive until then. Reordering these is a use-after-free on every
    // early return out of this loop (stop, GPU error, alloc failure).
    //
    // `slot` is which of the two reusable destinations the outstanding
    // read is filling, or -1 when it is filling `blk` -- the per-block
    // allocation this falls back to for a checkpoint the slots cannot
    // serve. The two modes never run at once, but the same outstanding
    // read has to be joinable either way, so both destinations live here.
    struct PrefetchSlot {
      Block             blk;
      int               layer = -1;
      int               slot  = -1;
      std::future<bool> fut;
    } pf;
    const bool pf_on = _stream_blocks &&
                       std::getenv("VPIPE_H3_NO_PREFETCH") == nullptr;
    int pf_started = 0, pf_hit = 0;
    // The next layer that will actually be STREAMED. Resident blocks are
    // skipped -- prefetching one would read bytes the forward already
    // has. Safe to look ahead: promotion only ever adds the block just
    // finished, never one further down the stack.
    auto pf_next = [&](int from) {
      for (int n = from; n < c.n_layers; ++n) {
        const bool h = n < (int)_blocks.size() &&
                       !_blocks[(std::size_t)n].qkv.empty();
        if (!h) { return n; }
      }
      return -1;
    };

    // ---- 3. the block stack ------------------------------------------
    for (int Lx = 0; Lx < c.n_layers; ++Lx) {
      // Cooperative stop, checked EVERY block rather than only on the
      // streamed tail. The denoise loop already stops between steps, but
      // a step here is one pass over 50 blocks of a 33B stack -- tens of
      // seconds -- so between-step is not responsive enough for someone
      // who has pressed stop. Per block it lands in about a block.
      //
      // Returning empty is how forward() reports "no velocity": the
      // caller already treats that as a failed/abandoned step and the
      // denoise unwinds. Anything already encoded is dropped with the
      // stream, and the scratch is reused rather than freed, so bailing
      // mid-stack leaks nothing.
      if (_stream_stop && _stream_stop()) {
        if (err != nullptr) { *err = "stopped"; }
        return {};
      }
      if (_block_progress) { _block_progress(Lx, c.n_layers); }
      // Pinned prefix (Lx < _pinned) is resident in _blocks; the tail is
      // read from the retained weight set into a loop-local Block and
      // freed when the iteration ends. The whole forward is otherwise ONE
      // deferred stream, so the block's work has to be committed and
      // WAITED FOR before its weights go away -- an encoded GEMM holds
      // the buffer by pointer, not by reference.
      // Resident if it was pinned at load OR promoted by a previous pass;
      // `_blocks` is sized to n_layers in streaming mode and an unfilled
      // slot reads as empty.
      const bool held = Lx < (int)_blocks.size() &&
                        !_blocks[(std::size_t)Lx].qkv.empty();
      const bool streaming = _stream_blocks && !held;
      // The per-block allocation, used only when the slots cannot serve
      // this checkpoint. Empty in the ordinary case.
      Block streamed;
      // Whichever destination this block ended up in.
      const Block* bp = nullptr;
      if (streaming) {
        // `with_adaln` false once baked: those tensors are 55% of a
        // block's bytes and nothing reads them any more, so a streaming
        // run stops paying for them on every step. This is the whole
        // point of baking.
        const auto rd0 = sprof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
        // Allocate the slot pair on the first streamed block, and again
        // if `baked` has changed which tensors a block carries since.
        // The first read doubles as the allocation -- load_block_ builds
        // the shapes and fills them, so this costs nothing extra -- and
        // the second slot is cloned from those shapes without bytes.
        // VPIPE_H3_NO_SLOTS forces the per-block allocations back, which
        // is what makes the slot path A/B-able against the thing it
        // replaced -- the two must be the same function, and the only way
        // to show that is to run one forward each way.
        if (_slots_first) {
          _slots_first = false;
          _slots_off = std::getenv("VPIPE_H3_NO_SLOTS") != nullptr;
        }
        if (!_slots_off && (!_slots_ready || _slots_adaln != !baked)) {
          _slot[0] = Block{};
          _slot[1] = Block{};
          _slot_pair = false;
          if (!load_block_(*_ws, blk_("blocks.", Lx, ""), _slot[0], !baked,
                           Retain::Streamed)) {
            return {};
          }
          // The second slot is a durable allocation, so it is asked for
          // on the same terms as any other growth. Refused, this runs
          // single-slot: no prefetch, but still no per-block allocation
          // and no memcpy out of the mapping, which is the larger half.
          // VPIPE_H3_SINGLE_SLOT forces the refusal. Which mode a run
          // takes otherwise depends on LIVE memory, so on a tight box it
          // is not reproducible from outside -- and single-slot is a
          // distinct third path (the pair, this, and no slots at all)
          // that no A/B could reach without a way to ask for it. Read per
          // allocation rather than latched in a static, so a test can put
          // two arms in one process.
          const bool force_single =
              std::getenv("VPIPE_H3_SINGLE_SLOT") != nullptr;
          const auto mbs = _mc->memory_budget();
          _slot_pair = !force_single && mbs.recommended != 0 &&
                       mbs.fits_growth(block_bytes_(_slot[0])) &&
                       clone_block_(_slot[0], _slot[1], false);
          if (!_slot_pair) { _slot[1] = Block{}; }
          _slots_ready = true;
          _slots_adaln = !baked;
          _slot_cur    = 0;
          bp           = &_slot[0];
          if (_mc->session() != nullptr) {
            _mc->session()->log_debug(fmt(
                "MetalMiniMaxH3Transformer: block slots ready ({} x {} MB{})",
                _slot_pair ? 2 : 1, block_bytes_(_slot[0]) >> 20,
                _slot_pair ? "" : ", single -- prefetch off"));
          }
        }
        if (bp == nullptr && _slots_ready && !_slots_off) {
          int use = -1;
          if (pf.layer == Lx && pf.slot >= 0 && pf.fut.valid()) {
            // This block's read was issued under the PREVIOUS block's GPU
            // work, so waiting here costs only the part that did not fit
            // under that window.
            const bool ok = pf.fut.get();
            use      = pf.slot;
            pf.layer = -1;
            pf.slot  = -1;
            // A prefetch that could not serve is not a failed forward:
            // the same refusal on the synchronous path below turns the
            // slots off and falls back, and it means the same thing
            // here. Falling through costs this block a second read and
            // the run its slots -- both of which beat abandoning a step.
            if (!ok) {
              // Said here too, and not only on the synchronous path
              // below. This is the branch that fires FIRST -- a prefetch
              // is issued for the next block before the next block is
              // reached -- so leaving it silent is what turns a
              // checkpoint the slots cannot serve into an unexplained
              // slowdown, which is the one outcome the sticky fallback
              // was meant to avoid.
              _slots_off = true;
              if (_mc->session() != nullptr) {
                _mc->session()->log_debug(fmt(
                    "MetalMiniMaxH3Transformer: block slots cannot serve "
                    "this checkpoint (prefetch of block {}) -- streaming "
                    "per-block allocations, which re-read through the "
                    "shard mapping rather than uncached", Lx));
              }
            }
            else { ++pf_hit; }
          } else {
            // Never the slot an outstanding read is writing into.
            use = pf.slot >= 0 ? (pf.slot ^ 1) : _slot_cur;
            if (!refill_block_(*_ws, blk_("blocks.", Lx, ""), _slot[use],
                               !baked)) {
              // A dtype or a shape no raw read can place. Sticky, and
              // said once: a fallback that came and went would read as an
              // unexplained slowdown rather than a property of the
              // checkpoint.
              _slots_off = true;
              if (_mc->session() != nullptr) {
                _mc->session()->log_debug(fmt(
                    "MetalMiniMaxH3Transformer: block slots cannot serve "
                    "this checkpoint -- streaming per-block allocations"));
              }
            }
          }
          if (!_slots_off) {
            _slot_cur = use;
            bp        = &_slot[use];
          } else {
            // Hand the slots back. They are a block each and nothing
            // will read them again this run.
            //
            // JOIN FIRST. A prefetch may be in flight into one of these,
            // and freeing a buffer a reader thread is writing into is a
            // use-after-free -- the one way this fallback could turn a
            // recoverable refusal into a crash. Its result is discarded
            // either way: the block is about to be rebuilt from scratch.
            if (pf.fut.valid()) { (void)pf.fut.get(); }
            pf.layer   = -1;
            pf.slot    = -1;
            _slot[0]   = Block{};
            _slot[1]   = Block{};
            _slot_pair = false;
          }
        }
        if (bp == nullptr) {
          // The pre-slot path, unchanged.
          bool have = false;
          if (pf.layer == Lx && pf.slot < 0 && pf.fut.valid()) {
            const bool ok = pf.fut.get();
            pf.layer = -1;
            if (!ok) { return {}; }
            streamed = std::move(pf.blk);
            pf.blk = Block{};
            have = true;
            ++pf_hit;
          }
          if (!have && !load_block_(*_ws, blk_("blocks.", Lx, ""), streamed,
                                    !baked, Retain::Streamed)) {
            return {};
          }
          bp = &streamed;
        }
        if (sprof) {
          sp_read_ms += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - rd0).count();
          sp_read_bytes += block_bytes_(*bp);
          ++sp_blocks;
        }
      }
      const Block& b = streaming ? *bp : _blocks[(std::size_t)Lx];
      // The modulation table for this block: [n_t, 6*H*3], which is the
      // same bytes as [n_t*3, 6*H] -- the layout `timestep_index * 3 +
      // tag` addresses. M is the number of DISTINCT timesteps, so this
      // is a very tall, very thin GEMM.
      //
      // Unbaked this reads the block's 260M-parameter AdaLN projection on
      // EVERY denoise step, and those 50 projections are 55% of the 4-bit
      // checkpoint. bake_adaln replaces the GEMM with a slice of a table
      // computed once for the whole schedule -- see its comment for why
      // that is a memory decision rather than a speed one.
      const LoraFactors* ada_lo =
          ((std::size_t)Lx < _lora_blocks.size() &&
           !_lora_blocks[(std::size_t)Lx].adaln.empty())
              ? &_lora_blocks[(std::size_t)Lx].adaln
              : nullptr;
      if (baked) {
        const std::size_t row0 =
            (std::size_t)_adaln_row0[(std::size_t)in.schedule_index] *
            (std::size_t)c.adaln_out();
        if (ada_lo == nullptr) {
          // Nothing to add: point the modulation kernels straight at
          // this step's rows. No GEMM, no copy, no scratch.
          mod_buf = &_adaln_tab[(std::size_t)Lx];
          mod_off = row0;
        } else {
          // An adapted AdaLN still has to be ADDED to something
          // writable, so the slice is copied into the scratch first.
          // That copy is 580 KB against the 258 MB projection it
          // replaced, and it is what keeps `lora_scale` live: the
          // adapter is applied per forward rather than baked in.
          mod_buf = &s.mod;
          mod_off = 0;
          const int n_el = n_t * c.adaln_out();
          enc.set_function(_fn_copy);
          enc.set_buffer(0, _adaln_tab[(std::size_t)Lx], row0 * 2);
          enc.set_buffer(1, s.mod);
          enc.set_constant(2, 0);
          enc.set_constant(3, n_el);
          enc.dispatch({(unsigned)n_el, 1, 1}, {256, 1, 1});
          lora_apply_(enc, s.temb, 0, *ada_lo, s.mod, 0, n_t, c.adaln_out(),
                      c.time_dim);
        }
      } else {
        mod_buf = &s.mod;
        mod_off = 0;
        gemm_(enc, s.temb, 0, b.adaln, s.mod, 0, n_t, c.adaln_out(),
              c.time_dim);
        if (ada_lo != nullptr) {
          lora_apply_(enc, s.temb, 0, *ada_lo, s.mod, 0, n_t, c.adaln_out(),
                      c.time_dim);
        }
      }
      psplit(t_adaln);
      // The modulation VALUES, per modality tag. s.mod is [n_t*3, 6*H]
      // with row = timestep_index*3 + tag, so the three tags are three
      // slices of ONE tensor and comparing them needs no reference: if
      // tag 1's scale/shift dwarf tag 0's, the text rows' 460x blow-up
      // is coming from the numbers, not from the addressing.
      if (xprobe && Lx == 0 && _mc->session() != nullptr) {
        enc.end();
        stream.commit().wait();
        const auto* m = static_cast<const std::uint16_t*>(s.mod.contents());
        for (int tag = 0; tag < h3::kModalityNum; ++tag) {
          const std::size_t row = (std::size_t)tag;   // timestep 0
          double sh = 0.0, sc = 0.0, ga = 0.0;
          double shx = 0.0, scx = 0.0, gax = 0.0;
          for (int k = 0; k < H; ++k) {
            auto rd = [&](int off) {
              const std::uint32_t u =
                  (std::uint32_t)m[row * (6 * H) + off + k] << 16;
              float v; std::memcpy(&v, &u, 4); return (double)v;
            };
            const double a = rd(0), b2 = rd(H), g = rd(2 * H);
            sh += a * a; sc += b2 * b2; ga += g * g;
            shx = std::max(shx, std::fabs(a));
            scx = std::max(scx, std::fabs(b2));
            gax = std::max(gax, std::fabs(g));
          }
          _mc->session()->info(fmt(
              "h3-modprobe tag {} ({})  shift rms {:.4f} max {:.4f}  "
              "scale rms {:.4f} max {:.4f}  gate rms {:.4f} max {:.4f}",
              tag, tag == 0 ? "video" : (tag == 1 ? "text" : "audio"),
              std::sqrt(sh / H), shx, std::sqrt(sc / H), scx,
              std::sqrt(ga / H), gax));
        }
        stream = _mc->make_command_stream();
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
      }
      bprobe = xprobe && Lx == 0;
      trip_blk = Lx;
      block(b, s.x, seq, true,
            (std::size_t)Lx < _lora_blocks.size() ? &_lora_blocks[(std::size_t)Lx]
                                                  : nullptr);
      bprobe = false;
      if (Lx == 0 || Lx == 3 || Lx == 24 || Lx == c.n_layers - 1) {
        xdump(("after block " + std::to_string(Lx)).c_str());
      }
      // A RESIDENT stack encodes all 50 blocks in milliseconds and then
      // runs for a minute, so `_block_progress` -- which fires at the top
      // of each iteration, on the ENCODE thread -- would race to 100% and
      // sit there. The streamed path does not have that problem only
      // because it must already commit-and-wait per block to free the
      // weights, which paces its callbacks against real work by accident.
      //
      // So take the same barrier deliberately when someone is watching.
      // It costs a commit and a re-encode per block -- sub-millisecond
      // against ~1.5 s of GPU per block at production geometry, and there
      // is no CPU work to overlap with anyway, since encoding is the only
      // thing this thread does. Gated on the callback so a run with no UI
      // attached keeps one uninterrupted stream.
      if (!streaming && _block_progress) {
        enc.end();
        std::string blk_err;
        if (!stream.commit().wait_ok(&blk_err)) {
          return fail("block " + std::to_string(Lx) + ": " +
                      (blk_err.empty() ? std::string("GPU error") : blk_err));
        }
        stream = _mc->make_command_stream();
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
      }
      if (streaming) {
        enc.end();
        std::string blk_err;
        const auto gp0 = sprof ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
        metal_compute::CommandStream::Fence fence = stream.commit();
        // Between the commit and the wait is the whole opportunity: the
        // GPU is busy with block Lx and this thread has nothing to do.
        //
        // Gated on the SAME budget question growth asks, because on a
        // box that fits one block the failure mode is not slowness but
        // thrash: a second live block tips the machine into the
        // compressor, the prefetched pages are evicted before the GPU
        // reads them, and a hidden read becomes read + compress +
        // decompress. Asked per block and never queued, so the moment it
        // stops being affordable the next iteration is serial again.
        if (pf_on && pf.layer < 0) {
          const int nxt = pf_next(Lx + 1);
          const bool slots = _slots_ready && !_slots_off && _slot_pair;
          // In slot mode the destination is memory this model ALREADY
          // holds, so there is no growth to gate -- the budget question
          // was asked once, when the second slot was allocated. The gate
          // stays on the fallback path, where a prefetch really does put
          // a second block's worth of buffers in the air.
          //
          // That also removes the reason a prefetch used to come and go
          // between blocks: the gate consults system-wide paging, which
          // moves under a streaming run, so on a tight box it would serve
          // some blocks and not others and the GPU would idle on exactly
          // the ones it refused.
          bool room = slots;
          if (!slots) {
            // SINGLE-SLOT ISSUES NOTHING. The fallback destination is
            // only ever CONSUMED by the pre-slot path below, which runs
            // when the slots are not ready; the slot consumption branch
            // requires pf.slot >= 0. So a fallback read issued while one
            // live slot is serving the stack is picked up by nobody:
            // pf.layer stays set, which refuses every later prefetch, and
            // pf.blk holds a whole block for the rest of the forward.
            //
            // MEASURED before this, on a 4-block stack: the pair reports
            // 3 of 3 reads hit and slots-off 3 of 3, while single-slot
            // reported 0 of 1 -- one read paid for and discarded, then no
            // prefetch at all. Both wrongs land on the run that refused
            // the second slot because memory was tight, which is the last
            // run that can afford a block of dead buffers.
            //
            // Not issuing is what the mode's own log line already claims
            // ("single -- prefetch off") and is strictly better than the
            // behaviour it replaces: the same absent overlap, minus the
            // wasted read and minus the block. Teaching the consumption
            // path to take a fallback read while the slots are live would
            // be better still, and is a separate change -- it puts a
            // second block in the air, which is the growth the budget
            // just refused.
            room = false;
            if (!_slots_ready || _slots_off) {
              const auto mbp = _mc->memory_budget();
              room = mbp.recommended != 0 &&
                     mbp.fits_growth(block_bytes_(*bp));
            }
          }
          if (nxt >= 0 && room) {
            pf.layer = nxt;
            pf.slot  = slots ? (_slot_cur ^ 1) : -1;
            ++pf_started;
            const int dst = pf.slot;
            pf.fut = std::async(
                std::launch::async, [this, &pf, nxt, baked, dst]() {
                  return dst >= 0
                             ? refill_block_(*_ws, blk_("blocks.", nxt, ""),
                                             _slot[dst], !baked)
                             : load_block_(*_ws, blk_("blocks.", nxt, ""),
                                           pf.blk, !baked, Retain::Streamed);
                });
          }
        }
        if (!fence.wait_ok(&blk_err)) {
          return fail("streamed block " + std::to_string(Lx) + ": " +
                      (blk_err.empty() ? std::string("GPU error") : blk_err));
        }
        if (sprof) {
          sp_gpu_ms += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - gp0).count();
        }
        stream = _mc->make_command_stream();
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
        // The commit above has been WAITED for, so nothing encoded still
        // points at this block's buffers -- which is exactly why the
        // promotion happens here and not at the top of the iteration.
        // Keeping it costs nothing extra: the bytes are already resident,
        // and the alternative is dropping them and re-reading the same
        // block from disk on the next of 30-odd steps.
        if (Lx < (int)_blocks.size()) {
          const std::size_t nb = block_bytes_(*bp);
          // Past the wire budget there is nothing to gain: the block
          // would be kept unprotected, the compressor would take it (it
          // is the coldest memory in the process), and the next walk
          // would shed a block and ratchet the ceiling over the whole
          // resident set. Better not to hold it at all.
          const bool wirable = !_wire_resident ||
                               _wired_bytes + nb <= _wire_budget;
          if (wirable && _resid.admit(_mc, nb)) {
            // Out of a SLOT the block has to be copied -- the slot is the
            // streamer's own destination and the next read needs it back.
            // A move would hand it away and leave the stream writing into
            // the resident set. Paid at most once per block for a whole
            // run, against the per-step read it retires.
            //
            // The fallback path owns its block outright, so it still
            // moves.
            bool kept = true;
            if (bp == &streamed) {
              _blocks[(std::size_t)Lx] = std::move(streamed);
            } else {
              kept = clone_block_(*bp, _blocks[(std::size_t)Lx], true);
              if (!kept) { _blocks[(std::size_t)Lx] = Block{}; }
            }
            // A failed copy is simply not a promotion: the slot still
            // holds the block, the forward has already run it, and the
            // next step streams it again. Nothing is booked.
            if (kept) {
              // Now that it is KEPT, the fc1 interleave is paid once
              // instead of per forward, so every later step takes the
              // fused FF for this block. The scratch stays wide (it was
              // sized for a streaming run and blocks below this one may
              // still be unfused), so the two paths coexist.
              if (_fuse_ff) { interleave_gu_(_blocks[(std::size_t)Lx].fc1); }
              // Wired LAST, after every write this block will ever get:
              // mlock pins the pages that exist now, and interleave_gu_
              // above replaces buffers outright.
              if (_wire_resident) {
                const std::size_t got =
                    wire_block_(_blocks[(std::size_t)Lx], true);
                _wired_bytes += got;
                // The percentage is an UP-TO, not a reservation. When
                // mlock refuses, the box is not going to give us the
                // rest of it -- another process holds wired memory, or
                // the system limit is nearer than the fraction implied
                // -- so the pool becomes what was actually granted and
                // growth stops here rather than continuing to admit
                // blocks nothing can protect.
                if (got < nb) {
                  // Hold HERE, for the rest of this forward only. The
                  // box has just said no, so asking again on the next
                  // block would be one failed syscall per block -- but
                  // the refusal may have been another process spiking,
                  // and a run that never asks again holds a small
                  // resident set for the rest of the schedule on the
                  // strength of one syscall. `_wire_retry` is what makes
                  // the next forward ask again.
                  _wire_budget = _wired_bytes;
                  _wire_retry  = true;
                  _wire_retry_at = _mc->memory_budget().available_physical;
                  if (_mc->session() != nullptr) {
                    _mc->session()->log_debug(fmt(
                        "MetalMiniMaxH3Transformer: the box granted {} MB "
                        "of the wired pool and refused more; holding there "
                        "for this forward and retrying on the next",
                        _wired_bytes >> 20));
                  }
                }
              }
              _resid.note_admitted(nb);
              if (_mc->session() != nullptr) {
                const auto mb = _mc->memory_budget();
                _mc->session()->log_debug(fmt(
                    "MetalMiniMaxH3Transformer: block {} resident ({} of {}, "
                    "{} MB, {} MB wired; {} MB idle, {} MB compressed, "
                    "{} MB swap, reserve {} MB)", Lx,
                    _resid.count(), c.n_layers, _resid.bytes() >> 20,
                    _wired_bytes >> 20,
                    mb.free_physical >> 20, mb.compressed >> 20,
                    mb.swap_used >> 20, _resid.reserve() >> 20));
              }
            }
          }
        }
      }
    }

    if (sprof && sp_blocks > 0 && _mc->session() != nullptr) {
      const double tot = sp_read_ms + sp_gpu_ms;
      const double sp_alloc_ms =
        (_ws ? _ws->stats().streamed_alloc_ms : 0.0) - sp_alloc0;
    const double sp_fetch_ms =
        (_ws ? _ws->stats().streamed_fetch_ms : 0.0) - sp_fetch0;
    const double gbs = sp_read_ms > 0.0
          ? (double)sp_read_bytes / (sp_read_ms / 1000.0) / 1e9 : 0.0;
      _mc->session()->log_normal(fmt(
          "MetalMiniMaxH3Transformer: streamed {} of {} blocks -- read "
          "{:.0f} ms ({} MB at {:.2f} GB/s, {:.1f} ms/block), gpu {:.0f} ms "
          "({:.0f} ms/block); prefetch {}/{} hit, a perfect one hides at "
          "most {:.1f}% of "
          "this pass [read = {:.0f} ms alloc + {:.0f} ms fetch at {:.2f} GB/s]", sp_blocks, c.n_layers, sp_read_ms,
          sp_read_bytes >> 20, gbs, sp_read_ms / (double)sp_blocks,
          sp_gpu_ms, sp_gpu_ms / (double)sp_blocks,
          pf_hit, pf_started,
          tot > 0.0 ? 100.0 * sp_read_ms / tot : 0.0,
        sp_alloc_ms, sp_fetch_ms,
        sp_fetch_ms > 0.0
            ? (double)sp_read_bytes / (sp_fetch_ms / 1000.0) / 1e9
            : 0.0));
    }

    // What the prefetch actually managed this pass, kept so a test can
    // see it: `started` counts reads issued, `hit` counts reads a block
    // consumed. Any gap is work paid for and thrown away, and -- on the
    // fallback destination -- a block's worth of buffers left live for
    // the rest of the forward.
    _pf_started = pf_started;
    _pf_hit     = pf_hit;
    // THE FIRE-AND-FORGET TALLY. The encoder splits every 50 dispatches
    // and commits those buffers without waiting, so nothing else is in a
    // position to see one fail; a completion handler latches it and
    // Fence::wait_ok reports it. This says whether that machinery is
    // LIVE. A zero error count next to a zero completed count means
    // nobody was watching; next to a large completed count it means
    // nothing failed, and those are opposite conclusions from the same
    // silence.
    if (_mc->session() != nullptr) {
      unsigned long long ff_done = 0, ff_bad = 0;
      CommandStream::fire_and_forget_stats(&ff_done, &ff_bad);
      _mc->session()->log_debug(fmt(
          "[h3-dit] fire-and-forget command buffers: {} completed, {} in "
          "error (process total)", ff_done, ff_bad));
    }

    // ---- 4. the shared output norm and the two heads ------------------
    const SharedBuffer* fmod_buf = &s.fmod;
    std::size_t fmod_off = 0;
    if (baked) {
      const std::size_t row0 =
          (std::size_t)_adaln_row0[(std::size_t)in.schedule_index] *
          (std::size_t)(2 * H);
      if (_lora_final.empty()) {
        fmod_buf = &_final_adaln_tab;
        fmod_off = row0;
      } else {
        const int n_el = n_t * 2 * H;
        enc.set_function(_fn_copy);
        enc.set_buffer(0, _final_adaln_tab, row0 * 2);
        enc.set_buffer(1, s.fmod);
        enc.set_constant(2, 0);
        enc.set_constant(3, n_el);
        enc.dispatch({(unsigned)n_el, 1, 1}, {256, 1, 1});
        lora_apply_(enc, s.temb, 0, _lora_final, s.fmod, 0, n_t, 2 * H,
                    c.time_dim);
      }
    } else {
      gemm_(enc, s.temb, 0, _final_adaln, s.fmod, 0, n_t, 2 * H, c.time_dim);
      lora_apply_(enc, s.temb, 0, _lora_final, s.fmod, 0, n_t, 2 * H,
                  c.time_dim);
    }
    rms(s.x, 0, _final_norm, s.proj, 0, seq, H, c.final_norm_eps);
    // The final modulation is indexed by the bare TIMESTEP index, not by
    // the AdaLN index: this norm is modality-independent. Its two halves
    // are shift then scale, the order the Wan and LTX output layers use.
    modulate(s.proj, *fmod_buf, fmod_off * 2, s.tstep_idx, s.proj, seq,
             2 * H, H, 0);
    // The inverse gather, run by run, so the velocity comes back in the
    // same row order the caller handed the latents in.
    {
      std::size_t dst = 0;
      for (const h3::RowRun& r : L.video_runs) {
        if (r.count > 0) {
          gemm_(enc, s.proj, (std::size_t)r.start * H, _video_out, out.video,
                dst * VPE, r.count, VPE, H);
        }
        dst += (std::size_t)r.count;
      }
    }
    if (n_audio > 0) {
      std::size_t dst = 0;
      for (const h3::RowRun& r : L.audio_runs) {
        if (r.count > 0) {
          gemm_(enc, s.proj, (std::size_t)r.start * H, _audio_out, out.audio,
                dst * AC, r.count, AC, H);
        }
        dst += (std::size_t)r.count;
      }
    }
    psplit(t_final);
  }
  std::string gpu_err;
  const bool trip_on = !_nan_report.empty() &&
                       std::getenv("VPIPE_H3_NAN_TRIP") != nullptr;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? std::string("MiniMax-H3 DiT forward failed")
                                : gpu_err);
  }
  // THE TRIPWIRE'S VERDICT, read once now that everything has completed.
  // Names the block and the op that first held a non-finite value, the
  // element, and how many there were -- one element or a region, which
  // are different bugs.
  if (trip_on && _mc->session() != nullptr) {
    const auto* r = static_cast<const unsigned*>(_nan_report.contents());
    if (r != nullptr && r[0] != 0) {
      static const char* kOp[] = {"n1", "?1", "qkv", "attn_out", "o_proj",
                                  "x_after_attn", "fc1", "swiglu", "fc2",
                                  "x_after_ff", "?", "?", "?", "?", "?", "?"};
      const unsigned blk = r[1] / 16, op = r[1] % 16;
      const std::uint32_t bits = (std::uint32_t)r[3] << 16;
      float v;
      std::memcpy(&v, &bits, 4);
      _mc->session()->warn(fmt(
          // "first" is whichever thread won the claim, not the lowest
          // index -- and the count is the RUNNING TOTAL over every
          // tripwire in this forward, not this tensor's. Both were
          // mislabelled in the first cut and both invite wrong readings:
          // an arbitrary index reads as a position, and a total reads as
          // a density.
          "h3-nan-trip: earliest CLAIMED non-finite at block {} op '{}' -- "
          "element {} (raw 0x{:04x} = {}); {} non-finite seen so far this "
          "forward, across all checks",
          blk, kOp[op], r[2], r[3], (double)v, r[4]));
      // EVERY call site that saw one, in block/op order. Where the count
      // is small the fault is localised and its inputs are worth
      // capturing; where it is the whole tensor it is already
      // propagation and the site upstream of it is the interesting one.
      for (unsigned t = 0; t < 1008u; ++t) {
        const unsigned c = r[16 + t];
        if (c == 0) { continue; }
        _mc->session()->warn(fmt(
            "h3-nan-trip:   block {:2} op '{}' -- {} non-finite", t / 16,
            kOp[t % 16], c));
      }
    } else {
      _mc->session()->log_debug(fmt("h3-nan-trip: no non-finite anywhere"));
    }
  }

  if (prof && _mc->session() != nullptr) {
    const double tot = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_begin).count();
    const double NL = (double)c.n_layers;
    auto rate = [](double flops, double ms) {
      return ms > 0.0 ? flops / (ms * 1e6) : 0.0;
    };
    const double f_qkv = NL * 2.0 * (double)seq * 3.0 * I * H;
    const double f_op  = NL * 2.0 * (double)seq * H * I;
    const double f_ff  = NL * 2.0 * (double)seq * H * (2.0 * FF + FF);
    const double f_ada = NL * 2.0 * (double)n_t * c.adaln_out() * c.time_dim;
    // Attention is QK^T + AV, both [seq, seq] x inner, and this model
    // masks NOTHING -- one packed sequence attending to itself -- so
    // there is no halving. Reported as a RATE because the share alone
    // cannot distinguish "attention is expensive here because it is
    // quadratic" from "the attention kernel is slow", and at this
    // sequence length those call for completely different work.
    const double f_attn = NL * 4.0 * (double)seq * (double)seq * I;
    _mc->session()->info(fmt(
        "[h3-dit] {} rows ({} text + {} cond + {} audio + {} video), "
        "{} blocks, {} timesteps, {:.0f} ms total\n"
        "  input+refiner {:8.1f} ms   adaln {:8.1f} ms ({:6.0f} GF/s)\n"
        "  qkv   {:8.1f} ms ({:6.0f} GF/s)   o-proj {:8.1f} ms "
        "({:6.0f} GF/s)\n"
        "  ff    {:8.1f} ms ({:6.0f} GF/s)   attn   {:8.1f} ms "
        "({:6.0f} GF/s, {})\n"
        "  prep  {:8.1f} ms   elt {:8.1f} ms   final {:8.1f} ms",
        seq, n_text, n_cond, n_audio, L.num_video_rows, c.n_layers, n_t, tot,
        t_in, t_adaln, rate(f_ada, t_adaln),
        t_qkv, rate(f_qkv, t_qkv), t_oproj, rate(f_op, t_oproj),
        t_ff, rate(f_ff, t_ff), t_attn, rate(f_attn, t_attn),
        use_steel ? "steel flash" : "SCALAR sdpa",
        t_prep, t_elt, t_final));
  }
  return out;
}

}  // namespace genai
}  // namespace vpipe
