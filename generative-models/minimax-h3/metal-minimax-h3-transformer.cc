#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-stream-kv.h"

#include "common/flex-data.h"
#include "common/temp-root.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/riffle-rows.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/shared/stream-sizing.h"
#include "generative-models/shared/streamed-refill.h"
#include "generative-models/shared/kernel-autotune.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/shared/tuned-answers.h"
#include "generative-models/shared/dit-gpu-progress.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "generative-models/generative-model-manager.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <map>
#include <set>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

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
  // FP8, plain or scaled (or U8 codes a comfy_quant record calls FP8), or
  // ComfyUI's int8 (I8, its rotation undone): decoded whole, scale
  // applied, to the bits a dense conversion of the file holds. A U8 or I8
  // tensor no record claims fails here, as it should.
  if (fp8::format_of(info->dtype) != fp8::Format::kNone ||
      info->dtype == "U8" || info->dtype == "I8") {
    SharedBuffer out = mc->make_shared_buffer(n * 2);
    if (out.empty() ||
        !fp8::decode_tensor_bf16(
            wts, nm, static_cast<std::uint16_t*>(out.contents()), n,
            nullptr)) {
      return {};
    }
    return out;
  }
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

// A Draw Things i8x weight's row scales (`<stem>.i8x_scales`, the
// store's element type) widened to the f32 the token GEMM reads.
SharedBuffer
i8x_scales_f32_(const MetalLlamaWeights& wts, MetalCompute* mc,
                const std::string& nm)
{
  const auto* info = wts.info(nm);
  if (info == nullptr || info->shape.size() != 1) { return {}; }
  const std::size_t n = (std::size_t)info->shape[0];
  SharedBuffer raw = wts.load(nm, mc);
  SharedBuffer out = mc->make_shared_buffer(n * 4);
  if (raw.empty() || out.empty()) { return {}; }
  auto* d = static_cast<float*>(out.contents());
  if (info->dtype == "F32") {
    std::memcpy(d, raw.contents(), n * 4);
  } else if (info->dtype == "F16") {
    const auto* sp = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (float)sp[i]; }
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
  if (fp8::format_of(info->dtype) != fp8::Format::kNone ||
      info->dtype == "U8" || info->dtype == "I8") {
    out.resize(n);
    return fp8::decode_tensor_f32(ws.src(), nm, out.data(), n, nullptr);
  }
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
  // The diffusers pipeline at the repository root says it by PLACE:
  // `transformer/` is FL2VA's DiT and `transformer_ref/` Ref2VA's. Its
  // manifest names no partition, because it is both.
  if (!dit.empty()) {
    const std::string pp = minimax_h3::diffusers_pipeline_partition(dit);
    if (!pp.empty()) { return pp; }
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
  // The diffusers pipeline at the repository root holds BOTH DiTs
  // (minimax-h3-diffusers-layout.h), so its `transformer/` is FL2VA's and
  // not the answer for every partition: taken as "a partition root" it
  // handed a Ref2VA graph the FL2VA weights, which load, run at full
  // cost and condition on nothing. Such a root is answered below, after
  // the released partitions.
  const bool pipeline_root =
      minimax_h3::is_diffusers_pipeline_root(p.string());
  if (!pipeline_root && fs::exists(p / "transformer" / "config.json")) {
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
  // The diffusers pipeline's own DiTs, LAST: a checkout that also holds
  // the released partition has just answered with it (the same weights,
  // under the names every other layout here was verified against). Told,
  // the partition's directory only; untold, FL2VA's, as above.
  if (pipeline_root) {
    const std::string d = minimax_h3::diffusers_pipeline_dit(p.string(),
                                                             partition);
    if (!d.empty()) { return d; }
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
  if (o.contains("refine_start_sigma")) {
    p.refine_start_sigma =
        o.at("refine_start_sigma").as_real(p.refine_start_sigma);
  }
  if (o.contains("refine_shift")) {
    p.refine_shift = o.at("refine_shift").as_real(p.refine_shift);
  }
  if (p.refine_start_sigma > 1.0 || p.refine_start_sigma == 0.0) {
    if (err != nullptr) {
      *err = fmt("refine_start_sigma {} is not in (0, 1]; the release's "
                 "grid runs", p.refine_start_sigma)();
    }
    p.refine_start_sigma = -1.0;
  }
  if (!(p.refine_shift > 0.0)) { p.refine_shift = 6.0; }
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
  // The diffusers layout (minimax-h3-diffusers-layout.h): its assembled
  // qkv is FLAT, whatever a config says about grouping.
  bool diffusers = false;
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
  } else if (fs::is_regular_file(p) &&
             (lower_(p.extension().string()) == ".safetensors" ||
              lower_(p.extension().string()) == ".ckpt")) {
    // ---- a single file with no config --------------------------------
    // lightx2v's FP8 turbo DiT (the diffusers layout), a Draw Things
    // checkpoint (`.ckpt`, presented under the release's own names with
    // qkv_proj fused per head -- shared/drawthings-store.h), or a DiT
    // ComfyUI saved (H3's names under its `model.diffusion_model.`, the
    // qkv flat -- minimax-h3-diffusers-layout.h): the H3 architecture's
    // dimensions are read off the tensors, and everything the tensors
    // cannot say (patch, rope, eps) keeps the architecture's defaults.
    auto w = MetalLlamaWeights::open_model(p.string());
    if (!w.has_value()) { return fail("cannot open " + p.string()); }
    const bool split = w->has("blocks.0.attn.qkv_proj.q.weight");
    // ComfyUI's own: under its model prefix (a DiT ComfyUI saved), or
    // BARE but in its PRUNED form -- an `adaln_t_table` exists only in
    // ComfyUI's curve-form model (Comfy-Org's official pruned files carry
    // no prefix and no metadata). Either way its qkv is ComfyUI's FLAT
    // order -- MEASURED on Comfy-Org's minimax_h3_fl2va_pruned_int8_
    // convrot: per-row scale R^2 0.140 / 0.037 / 0.207 grouped flat
    // against 0.001 / 0.001 / 0.004 per head (blocks 0 / 25 / 49).
    const bool comfy_saved =
        !split && (minimax_h3::is_comfy_saved_dit_file(p.string()) ||
                   (w->has("adaln_t_table") &&
                    w->has("blocks.0.attn.qkv_proj.weight")));
    const bool fused =
        !split &&
        (lower_(p.extension().string()) == ".ckpt" || comfy_saved) &&
        w->has("blocks.0.attn.qkv_proj.weight");
    if (!split && !fused) {
      return fail(p.string() + " is neither a Comfy-Org H3 component, an "
                               "H3 DiT ComfyUI saved, nor one in the "
                               "diffusers naming");
    }
    auto dim = [&](const std::string& n, int axis) -> long {
      const auto* ti = w->info(n);
      if (ti == nullptr || (int)ti->shape.size() <= axis) { return -1; }
      long v = (long)ti->shape[(std::size_t)axis];
      // A W4A8 weight's codes are nibbles, two a byte: its logical width
      // is twice the stored one (its relative scales beside it say so).
      if (axis == 1 && ti->dtype == "I8" && n.size() > 7 &&
          n.compare(n.size() - 7, 7, ".weight") == 0 &&
          w->has(n.substr(0, n.size() - 7) + ".weight_s_rel")) {
        v *= 2;
      }
      return v;
    };
    auto count = [&](const std::string& pre) {
      int n = 0;
      while (w->has(pre + std::to_string(n) + ".norm1.weight")) { ++n; }
      return n;
    };
    const long hidden = dim("condition_proj.weight", 0);
    const long vin = dim("video_patch_proj.weight", 1);
    if (hidden <= 0 || vin <= 0 || vin % 4 != 0) {
      return fail(p.string() + ": cannot read the H3 dimensions off it");
    }
    // ComfyUI's PRUNED form has no timestep MLP: the AdaLN input is the
    // width of its `adaln_t_table` (Config::curve_grid). The MLP's two
    // other widths are then the architecture's, and nothing reads them.
    const long grid = dim("adaln_t_table", 0);
    const long tdim = grid > 0 ? dim("adaln_t_table", 1)
                               : dim("time_embedder.proj_out.weight", 0);
    if (grid > 0 && dim("blocks.0.adaln_proj.linear.weight", 1) != tdim) {
      return fail(p.string() + ": its AdaLN projections do not read its "
                               "adaln_t_table's width");
    }
    auto or_def = [](long v, long d) { return v > 0 ? v : d; };
    cfg = FlexData::from_json(fmt(
        "{{\"hidden_size\": {}, \"num_attention_heads\": {}, "
        "\"num_layers\": {}, \"token_refiner_num_layers\": {}, "
        "\"ffn_hidden_size\": {}, \"latents_dim\": {}, "
        "\"audio_latents_dim\": {}, \"text_dim\": {}, "
        "\"timestep_input_dim\": {}, \"time_embed_hidden_size\": {}, "
        "\"time_embed_dim\": {}, \"adaln_curve_grid\": {}}}",
        hidden,
        split ? dim("blocks.0.attn.qkv_proj.q.weight", 0) / 128
              : dim("blocks.0.attn.qkv_proj.weight", 0) / (3 * 128),
        count("blocks."), count("token_refiner.blocks."),
        dim("blocks.0.mlp.fc2.weight", 1), vin / 4,
        dim("audio_patch_proj.weight", 1), dim("condition_proj.weight", 1),
        or_def(dim("time_embedder.proj_in.weight", 1), 256),
        or_def(dim("time_embedder.proj_in.weight", 0), hidden),
        tdim, grid > 0 ? grid : 0)());
    // The diffusers layout's assembled qkv is FLAT, and so is ComfyUI's
    // own; the Draw Things one is gathered per head, as released.
    diffusers = split || comfy_saved;
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
    if (cls == "MiniMaxH3Transformer3DModel") {
      // The diffusers config: the same model under diffusers' own key
      // names, which are moved onto the native ones read below. Only the
      // keys that differ are listed; the rest are spelled alike.
      static const std::pair<const char*, const char*> kKeys[] = {
          {"num_refiner_layers", "token_refiner_num_layers"},
          {"ffn_dim", "ffn_hidden_size"},
          {"in_channels", "latents_dim"},
          {"audio_in_channels", "audio_latents_dim"},
          {"freq_dim", "timestep_input_dim"},
          {"time_embed_hidden_dim", "time_embed_hidden_size"},
          {"rope_freq_dim", "rope_inv_freq_len"},
      };
      std::string j = "{";
      for (const auto& [k, v] : co) {
        std::string key(k);
        for (const auto& [from, to] : kKeys) {
          if (key == from) { key = to; }
        }
        if (j.size() > 1) { j += ", "; }
        j += "\"" + key + "\": " + v.to_json();
      }
      j += "}";
      cfg = FlexData::from_json(j);
      diffusers = true;
    } else if (cls != "MiniMaxH3DiTModel") {
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
  out.qkv_per_head = diffusers ? false
                     : o.contains("qkv_per_head")
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
  out.curve_grid     = std::max(0, gi("adaln_curve_grid", 0));
  out.rope_freq_dim  = gi("rope_inv_freq_len", 16);
  out.rope_theta     = gf("rope_theta", 10000.0f);
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
  // The pruned AdaLN kernel reads points of 8 (h3_adaln_pruned_f16), the
  // width every pruned checkpoint seen carries.
  if (out.pruned() && (out.time_dim != 8 || out.curve_grid < 2)) {
    return fail(fmt("a pruned AdaLN of {} x {} (adaln_t_table) is not the "
                    "8-wide curve this model reads", out.curve_grid,
                    out.time_dim)());
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

namespace {

// fp8_widen_bytes off a surveyed checkpoint: the widest of its FP8 /
// int8 / W4A8 weights as bf16 (a W4A8 weight's K is twice its stored
// width), and the fused q|k|v when a block's projections are among them
// -- a file that stores q, k and v apart is widened fused, as it loads.
std::size_t
widen_of_(const MetalLlamaWeights& w, const fp8::Layout& lay,
          const MetalMiniMaxH3Transformer::Config& c)
{
  std::size_t n = 0;
  bool block = false;
  for (const auto& [nm, fw] : lay.weights) {
    const auto* ti = w.info(nm);
    if (ti == nullptr || ti->shape.size() != 2) { continue; }
    const std::size_t k = (std::size_t)ti->shape[1] *
                          (fw.format == fp8::Format::kW4A8 ? 2 : 1);
    n = std::max(n, (std::size_t)ti->shape[0] * k);
    block = block || nm.compare(0, 7, "blocks.") == 0;
  }
  if (block) {
    n = std::max(n, (std::size_t)3 * (std::size_t)c.inner() *
                        (std::size_t)c.hidden);
  }
  return n * 2;
}

}  // namespace

std::size_t
MetalMiniMaxH3Transformer::fp8_widen_bytes(const std::string& dit_dir,
                                           const Config& c)
{
  auto w = MetalLlamaWeights::open_model(
      resolve_dit_dir(dit_dir, c.partition));
  if (!w.has_value()) { return 0; }
  fp8::Layout lay;
  std::string err;
  if (!fp8::scan(*w, &lay, &err, /*int8=*/true) || !lay.any()) { return 0; }
  return widen_of_(*w, lay, c);
}

std::size_t
MetalMiniMaxH3Transformer::streaming_floor_bytes(const std::string& dit_dir)
{
  auto wts = MetalLlamaWeights::open_model(dit_dir);
  if (!wts.has_value()) { return 0; }
  // AdaLN excluded: the bake retires it before the first forward, so a
  // slot never carries it for long and counting it would make the floor
  // 1.6x what the model settles at.
  const std::size_t f = stream_floor_bytes(*wts, "blocks.", "adaln");
  // PRUNED, nothing is baked: the projections (77 MB at bf16) are held
  // in the trunk as stored, from load, so the floor carries them.
  if (f == 0 || !wts->has("adaln_t_table")) { return f; }
  std::size_t pruned = 0;
  for (const std::string& nm : wts->tensor_names()) {
    if (nm.rfind("blocks.", 0) != 0 ||
        nm.find("adaln_proj") == std::string::npos) {
      continue;
    }
    if (const auto* ti = wts->info(nm)) { pruned += (std::size_t)ti->nbytes; }
  }
  return f + pruned;
}

std::size_t
MetalMiniMaxH3Transformer::adaln_retired_bytes(const std::string& dit_dir)
{
  auto wts = MetalLlamaWeights::open_model(dit_dir);
  if (!wts.has_value()) { return 0; }
  // Pruned, there is no bake (bake_adaln refuses it) and nothing retires.
  if (wts->has("adaln_t_table")) { return 0; }
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
  if (max_rows <= 0 || cfg.pruned()) { return false; }
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
    const auto res = kept_residency_(_stream_blocks, _wire.on());
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
  // Stored in pieces (the diffusers layout): assembled whole.
  if (const auto parts = minimax_h3::assembled_parts(ws.src(), nm);
      !parts.empty()) {
    return assemble_(ws, nm, parts, bias, r);
  }
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
                                                      _wire.on()))
                   : ws.tensor(nm + ".weight", _mc,
                               kept_residency_(_stream_blocks,
                                               _wire.on()));
    l.scales = weight_(ws, nm + ".scales", r);
    l.qbias  = weight_(ws, nm + ".biases", r);
    if (!l.codes.empty() && !l.scales.empty() && !l.qbias.empty()) {
      l.quantized = true;
      return l;
    }
    l.codes = {}; l.scales = {}; l.qbias = {};
  }
  // FP8: the file's codes, kept at their own size and widened per GEMM
  // (fp8_dense_) -- the same residency rule as every other byte the
  // model keeps. An FP8 weight whose codes or scale cannot be read comes
  // back EMPTY, never as a dense read of the codes.
  if (const fp8::Weight* fw = _f8.find(nm + ".weight"); fw != nullptr) {
    if (ci == nullptr || ci->shape.size() != 2) { return l; }
    const auto res = kept_residency_(_stream_blocks, _wire.on());
    l.codes = r == Retain::Streamed
                  ? ws.stream_tensor(nm + ".weight", _mc, res)
                  : ws.tensor(nm + ".weight", _mc, res);
    bool sok = true;
    l.f8_scale = fp8::scale_buffer(src, *fw, (std::size_t)ci->shape[0],
                                   _mc, &sok);
    if (l.codes.empty() || !sok) { return Linear{}; }
    l.f8         = fw->format;
    l.f8_per_row = fw->per_row;
    l.f8_rot     = fw->rot_group;
    // W4A8: its relative scales beside the codes (held as they are, the
    // same way), and the codebook, 16 f32.
    if (fw->format == fp8::Format::kW4A8) {
      l.f8_qgroup = fw->qgroup;
      l.f8_srel = r == Retain::Streamed
                      ? ws.stream_tensor(fw->s_rel, _mc, res)
                      : ws.tensor(fw->s_rel, _mc, res);
      if (l.f8_srel.empty()) { return Linear{}; }
      if (!fw->codebook.empty()) {
        l.f8_cb = _mc->make_shared_buffer(16 * sizeof(float));
        if (l.f8_cb.empty() ||
            !src.read_into(fw->codebook, l.f8_cb.contents(),
                           16 * sizeof(float))) {
          return Linear{};
        }
      }
    }
    return l;
  }
  l.w = weight_(ws, nm + ".weight", r);
  return l;
}

// A ConvRot weight's row scales as the token GEMM reads them: f32, one a
// row -- a per-tensor scale repeated. Empty when they cannot be read.
SharedBuffer
MetalMiniMaxH3Transformer::convrot_row_scales_(const MetalLlamaWeights& src,
                                               const fp8::Weight& fw, int N)
{
  std::vector<float> v;
  if (!fp8::read_scale(src, fw, (std::size_t)N, &v, nullptr) || v.empty()) {
    return {};
  }
  if (v.size() == 1) { v.assign((std::size_t)N, v[0]); }
  if (v.size() != (std::size_t)N) { return {}; }
  SharedBuffer b = _mc->make_shared_buffer(v.size() * sizeof(float));
  if (b.empty()) { return {}; }
  std::memcpy(b.contents(), v.data(), v.size() * sizeof(float));
  return b;
}

MetalMiniMaxH3Transformer::Linear
MetalMiniMaxH3Transformer::i8_linear_(WeightSet& ws, const std::string& nm,
                                      Retain r)
{
  if (!_i8_layout || !_i8) { return Linear{}; }
  const MetalLlamaWeights& src = ws.src();
  // A DRAW THINGS i8x weight on the unrotated token tier: its codes and
  // row scales are that layout as stored, so nothing is quantized again
  // and a streamed block refills them by name. (Rotated, it is quantized
  // from its dequantized `.weight` below, like a dense checkpoint.)
  const auto* xi = src.info(nm + ".i8x");
  if (xi != nullptr && xi->shape.size() == 2 &&
      src.info(nm + ".i8x_scales") != nullptr &&
      _i8->fast_group((int)xi->shape[1]) == I8GemmContext::kTokenGroup &&
      !_i8->token_rotated()) {
    const int N = (int)xi->shape[0], K = (int)xi->shape[1];
    const std::string cn = nm + ".i8x", sn = nm + ".i8x_scales";
    Linear l;
    l.gc = I8GemmContext::kTokenGroup;
    l.n = N;
    l.k = K;
    l.i8x = true;
    if (r == Retain::Streamed) {
      l.codes = ws.stream_tensor(cn, _mc, WeightSet::Residency::Copied);
      l.cscales = ws.stream_derived(
          [&]() { return i8x_scales_f32_(src, _mc, sn); });
    } else {
      const std::size_t cb = (std::size_t)N * K, sb = (std::size_t)N * 4;
      SharedBuffer both = ws.derived(
          std::string("h3-i8/i8x|") + cn, [&]() -> SharedBuffer {
            SharedBuffer sc = i8x_scales_f32_(src, _mc, sn);
            SharedBuffer out = _mc->make_shared_buffer(cb + sb);
            if (sc.empty() || out.empty() ||
                !src.read_into(cn, out.contents(), cb)) {
              return {};
            }
            std::memcpy(static_cast<std::uint8_t*>(out.contents()) + cb,
                        sc.contents(), sb);
            return out;
          });
      if (!both.empty()) {
        l.codes = both.subview(0, cb);
        l.cscales = both.subview(cb, sb);
      }
    }
    if (l.codes.empty() || l.cscales.empty()) { return Linear{}; }
    return l;
  }
  // ComfyUI's int8 CONVROT -- and its W4A8 -- on the TOKEN tier, as
  // stored: the codes already ARE the token layout (int8 per output row
  // over all of K) in ConvRot's basis, so the activation is rotated alike
  // (I8GemmContext::quantize_token_convrot) and nothing is converted; a
  // W4A8 weight's nibbles become those codes per GEMM (gemm_).
  // VPIPE_H3_CONVROT_NATIVE=0 keeps them widened per GEMM, for an A/B.
  if (const fp8::Weight* fw = _f8.find(nm + ".weight");
      fw != nullptr && fw->rot_group == 256 &&
      (fw->format == fp8::Format::kI8 || fw->format == fp8::Format::kW4A8) &&
      _i8->convrot_available() &&
      !(std::getenv("VPIPE_H3_CONVROT_NATIVE") != nullptr &&
        std::atoi(std::getenv("VPIPE_H3_CONVROT_NATIVE")) == 0)) {
    const auto* wi = src.info(nm + ".weight");
    const bool w4 = fw->format == fp8::Format::kW4A8;
    const int N = wi != nullptr && wi->shape.size() == 2 ? (int)wi->shape[0]
                                                          : 0;
    const int K = wi != nullptr && wi->shape.size() == 2
                      ? (int)wi->shape[1] * (w4 ? 2 : 1)
                      : 0;
    if (N >= 16 && N % 4 == 0 && K % 256 == 0 &&
        _i8->fast_group(K) == I8GemmContext::kTokenGroup) {
      Linear l;
      l.gc = I8GemmContext::kTokenGroup;
      l.n = N;
      l.k = K;
      l.cr = fw->rot_group;
      l.cr_w4 = w4;
      const auto res = kept_residency_(_stream_blocks, _wire.on());
      l.codes = r == Retain::Streamed
                    ? ws.stream_tensor(nm + ".weight", _mc, res)
                    : ws.tensor(nm + ".weight", _mc, res);
      l.cscales = convrot_row_scales_(src, *fw, N);
      if (w4) {
        l.f8_qgroup = fw->qgroup;
        l.f8_srel = r == Retain::Streamed
                        ? ws.stream_tensor(fw->s_rel, _mc, res)
                        : ws.tensor(fw->s_rel, _mc, res);
        if (!fw->codebook.empty()) {
          l.f8_cb = _mc->make_shared_buffer(16 * sizeof(float));
          if (l.f8_cb.empty() ||
              !src.read_into(fw->codebook, l.f8_cb.contents(),
                             16 * sizeof(float))) {
            return Linear{};
          }
        }
        if (l.f8_srel.empty() || !_fn_w4a8_i8.valid()) { return Linear{}; }
      }
      if (l.codes.empty() || l.cscales.empty()) { return Linear{}; }
      return l;
    }
  }
  // Anything else is CONVERTED: once, into the entry a kept block holds,
  // straight from the source (a read, so no bf16 copy stays beside it);
  // or, a STREAMED block, after every refill out of what the slot holds
  // (Linear::ds / ps), so a streamed block runs the tier a preloaded one
  // runs. The conversion is a weight pass at the quantizers' rate (~0.1
  // ms a [12288, 4096] weight on an M5 Ultra) against a block's GEMMs at
  // video geometry, so it is paid per refill without regret.
  const auto* ci = src.info(nm + ".weight");
  if (ci == nullptr || ci->shape.size() != 2 ||
      _f8.find(nm + ".weight") != nullptr) {
    return Linear{};
  }
  const auto* si = src.info(nm + ".scales");
  const bool pack = _quant_bits > 0 && si != nullptr && si->shape.size() == 2;
  const int N = (int)ci->shape[0];
  const int K = pack ? (int)(si->shape[1] * _quant_group) : (int)ci->shape[1];
  // g64 packs, w8 or w4.
  const int bits =
      pack && K > 0 ? (int)(ci->shape[1] * 32 / (long)K) : 0;
  if (pack && (_quant_group != 64 || (bits != 8 && bits != 4) ||
               ci->shape[1] * 32 != (long)K * bits)) {
    return Linear{};
  }
  const int gc = _i8->fast_group(K);
  if (gc <= 0 || N < 16 ||
      (gc == I8GemmContext::kTokenGroup && (N % 4) != 0) ||
      (pack && !_i8->fast_converts(bits, gc))) {
    return Linear{};
  }
  // A W4 PACK KEPT AS STORED unless the engine says the box has room
  // for its layout (Config::i8_hold_w4): the layout is twice its size,
  // so gemm_ converts it into the context's scratch before each
  // projection (I8GemmContext::encode_to_fast_scratch). (A streamed one
  // is handled below: it is never cached.)
  if (pack && bits == 4 && !_cfg.i8_hold_w4 && r == Retain::Cached) {
    Linear l;
    l.gc = gc;
    l.n = N;
    l.k = K;
    l.bits = 4;
    l.pc = true;
    l.codes = ws.tensor(nm + ".weight", _mc,
                        kept_residency_(_stream_blocks, _wire.on()));
    l.scales = weight_(ws, nm + ".scales", r);
    l.qbias = weight_(ws, nm + ".biases", r);
    if (l.codes.empty() || l.scales.empty() || l.qbias.empty()) {
      return Linear{};
    }
    return l;
  }
  const std::size_t cb = (std::size_t)N * K;
  const std::size_t sb = I8GemmContext::fast_scale_bytes(N, K, gc);
  if (r == Retain::Streamed) {
    Linear l;
    l.gc = gc;
    l.n = N;
    l.k = K;
    if (pack && bits == 4) {
      // A slot holds a w4 pack as stored, converted per projection; a
      // promotion holds its layout where the engine says (keep_block_).
      l.bits = 4;
      l.pc = true;
    } else if (pack) {
      l.bits = 8;
      l.ps = true;
    } else {
      l.ds = true;
      l.w = weight_(ws, nm + ".weight", r);
      l.codes = _mc->make_shared_buffer(cb);
    }
    if (pack) {
      l.codes = ws.stream_tensor(nm + ".weight", _mc,
                                 WeightSet::Residency::Copied);
      l.scales = weight_(ws, nm + ".scales", r);
      l.qbias = weight_(ws, nm + ".biases", r);
      if (l.scales.empty() || l.qbias.empty()) { return Linear{}; }
    }
    if (!l.pc) { l.cscales = _mc->make_shared_buffer(sb); }
    if (l.codes.empty() || (!l.pc && l.cscales.empty()) ||
        (l.ds && l.w.empty())) {
      return Linear{};
    }
    if (!l.pc) {
      CommandStream st = _mc->make_command_stream();
      bool ok = false;
      {
        ComputeEncoder e = st.begin_compute();
        ok = encode_fast_(e, l);
      }
      if (!ok || !st.commit().wait_ok(nullptr)) { return Linear{}; }
    }
    return l;
  }
  const std::string key = std::string("h3-i8/") + _i8->layout_tag(gc) +
                          (pack ? (bits == 8 ? "w8|" : "w4|") : "bf|") +
                          nm;
  SharedBuffer both = ws.derived(key, [&]() -> SharedBuffer {
    SharedBuffer out = _mc->make_shared_buffer(cb + sb);
    if (out.empty()) { return {}; }
    SharedBuffer c, sc, b, w;
    if (pack) {
      c = ws.read(nm + ".weight", _mc, WeightSet::Residency::Copied);
      sc = to_bf16_(src, _mc, nm + ".scales");
      b = to_bf16_(src, _mc, nm + ".biases");
      if (c.empty() || sc.empty() || b.empty()) { return {}; }
    } else {
      w = to_bf16_(src, _mc, nm + ".weight");
      if (w.empty()) { return {}; }
    }
    CommandStream st = _mc->make_command_stream();
    bool ok = false;
    {
      ComputeEncoder e = st.begin_compute();
      ok = pack ? _i8->encode_to_fast(e, c, sc, b, out.subview(0, cb),
                                      out.subview(cb, sb), gc, N, K, bits)
                : _i8->encode_quant_fast(e, w, out.subview(0, cb),
                                         out.subview(cb, sb), gc, N, K);
    }
    return ok && st.commit().wait_ok(nullptr) ? std::move(out)
                                              : SharedBuffer{};
  });
  if (both.empty()) { return Linear{}; }
  Linear l;
  l.gc = gc;
  l.n = N;
  l.k = K;
  l.codes = both.subview(0, cb);
  l.cscales = both.subview(cb, sb);
  if (l.codes.empty() || l.cscales.empty()) { return Linear{}; }
  return l;
}

bool
MetalMiniMaxH3Transformer::encode_fast_(ComputeEncoder& e, Linear& l) const
{
  if (!l.ds && !l.ps) { return true; }
  if (_i8 == nullptr || l.codes.empty() || l.cscales.empty()) {
    return false;
  }
  // The same calls, at the same group, the preloaded entry is built with.
  if (l.ds) {
    return !l.w.empty() &&
           _i8->encode_quant_fast(e, l.w, l.codes, l.cscales, l.gc, l.n,
                                  l.k);
  }
  return !l.scales.empty() && !l.qbias.empty() &&
         _i8->encode_to_fast(e, l.codes, l.scales, l.qbias, l.codes,
                             l.cscales, l.gc, l.n, l.k);
}

std::size_t
MetalMiniMaxH3Transformer::streamed_holding_bytes() const
{
  if (!_stream_blocks) { return 0; }
  // The factors kept with promoted blocks are the adapter file's own
  // claim's (it books every factor), so they are not counted twice.
  std::size_t n = _resid.bytes() - std::min(_lora_kept, _resid.bytes()) +
                  slot_bytes() + _final_adaln_tab.byte_size();
  for (const SharedBuffer& t : _adaln_tab) { n += t.byte_size(); }
  return n;
}

std::size_t
MetalMiniMaxH3Transformer::release_idle_slots()
{
  if (!_stream_blocks || !_slots_ready ||
      _blocks.size() != (std::size_t)_cfg.n_layers) {
    return 0;
  }
  for (const Block& b : _blocks) {
    if (b.qkv.empty()) { return 0; }
  }
  std::size_t freed = slot_bytes();
  _slot[0] = Block{};
  _slot[1] = Block{};
  _slots_ready = false;
  _slot_pair = false;
  // ...and the set a streamed block's adapter factors were read into:
  // every kept block holds its own (keep_lora_), and fill_lora_ makes the
  // set again should one be given back.
  for (BlockLora& bl : _lora_stream) {
    for (auto m : kLoraMembers) {
      freed += (bl.*m).a.byte_size() + (bl.*m).b.byte_size();
      (bl.*m).a = metal_compute::SharedBuffer{};
      (bl.*m).b = metal_compute::SharedBuffer{};
    }
  }
  _lora_stream_layer = -1;
  if (freed > 0 && _mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalMiniMaxH3Transformer: all {} blocks resident -- the slot "
        "pair and the adapter factors' streamed set ({} MB) are released",
        _cfg.n_layers, freed >> 20));
  }
  return freed;
}

std::size_t
MetalMiniMaxH3Transformer::kept_block_bytes_(const Block& b) const
{
  const bool hold = _cfg.i8_hold_w4 && _i8 != nullptr;
  auto kl = [&](const Linear& l) -> std::size_t {
    if (l.pc && hold) {
      return (std::size_t)l.n * l.k +
             I8GemmContext::fast_scale_bytes(l.n, l.k, l.gc);
    }
    if (l.ds || l.ps) {
      return l.codes.byte_size() + l.cscales.byte_size() + l.b.byte_size();
    }
    return l.w.byte_size() + l.b.byte_size() + l.codes.byte_size() +
           l.scales.byte_size() + l.qbias.byte_size() +
           l.f8_scale.byte_size() + l.f8_srel.byte_size() +
           l.f8_cb.byte_size() + l.cscales.byte_size();
  };
  return b.n1.byte_size() + b.n2.byte_size() + b.qn.byte_size() +
         b.kn.byte_size() + kl(b.qkv) + kl(b.out) + kl(b.fc1) + kl(b.fc2) +
         kl(b.adaln);
}

bool
MetalMiniMaxH3Transformer::keep_block_(Block& b) const
{
  // Conversions land in fresh buffers, swapped in after the wait: the
  // encoder reads the pack until then.
  struct Swap {
    Linear* l;
    SharedBuffer codes, cscales;
  };
  std::vector<Swap> swaps;
  CommandStream cs;
  ComputeEncoder ce;
  bool ok = true;
  const bool hold = _cfg.i8_hold_w4 && _i8 != nullptr;
  for (Linear* l : {&b.qkv, &b.out, &b.fc1, &b.fc2}) {
    if (l->ds) {
      l->w = SharedBuffer{};
      l->ds = false;
    }
    if (l->ps) {
      l->scales = SharedBuffer{};
      l->qbias = SharedBuffer{};
      l->ps = false;
    }
    if (l->pc && hold) {
      Swap sw{l, _mc->make_shared_buffer((std::size_t)l->n * l->k),
              _mc->make_shared_buffer(
                  I8GemmContext::fast_scale_bytes(l->n, l->k, l->gc))};
      if (!cs.valid()) {
        cs = _mc->make_command_stream();
        ce = cs.begin_compute();
      }
      if (!sw.codes.empty() && !sw.cscales.empty() && ce.valid() &&
          _i8->encode_to_fast(ce, l->codes, l->scales, l->qbias, sw.codes,
                              sw.cscales, l->gc, l->n, l->k, l->bits)) {
        swaps.push_back(std::move(sw));
      } else {
        ok = false;
      }
    }
  }
  if (cs.valid()) {
    ce.end();
    ok = cs.commit().wait_ok(nullptr) && ok;
    for (Swap& sw : swaps) {
      if (!ok) { break; }
      sw.l->codes = std::move(sw.codes);
      sw.l->cscales = std::move(sw.cscales);
      sw.l->pc = false;
      sw.l->scales = SharedBuffer{};
      sw.l->qbias = SharedBuffer{};
    }
  }
  return ok;
}

namespace {

// The per-row f32 scale of a weight assembled from `parts`, in output-row
// order: each piece's own scale (one value spread over its rows, or its
// rows' values), 1 for a piece that has none. Empty when no piece is
// scaled -- the assembled weight is then plain FP8.
bool
assembled_scale_(const MetalLlamaWeights& src, const fp8::Layout& f8,
                 const std::vector<minimax_h3::Part>& parts,
                 std::vector<float>* out)
{
  out->clear();
  bool any = false;
  for (const auto& p : parts) {
    const fp8::Weight* fw = f8.find(p.layer + ".weight");
    if (fw != nullptr && !fw->scale.empty()) { any = true; }
  }
  if (!any) { return true; }
  for (const auto& p : parts) {
    const fp8::Weight* fw = f8.find(p.layer + ".weight");
    const auto* ti = src.info(p.layer + ".weight");
    if (fw == nullptr || ti == nullptr || ti->shape.empty()) { return false; }
    std::vector<float> v;
    if (!fp8::read_scale(src, *fw, (std::size_t)ti->shape[0], &v, nullptr)) {
      return false;
    }
    for (std::int64_t r = 0; r < p.rows; ++r) {
      out->push_back(v.empty()       ? 1.0f
                     : v.size() == 1 ? v[0]
                                     : v[(std::size_t)(p.row0 + r)]);
    }
  }
  return true;
}

// Exchange the two halves of `b` in place -- the swapped fc1 of the
// diffusers layout, after a whole-tensor refill. Through a bounded
// stack chunk, since a full-size temporary is the half being avoided.
void
swap_halves_(const SharedBuffer& b)
{
  const std::size_t half = b.byte_size() / 2;
  auto* p = static_cast<std::uint8_t*>(b.contents());
  alignas(64) std::uint8_t tmp[1 << 16];
  for (std::size_t o = 0; o < half; o += sizeof(tmp)) {
    const std::size_t n = std::min(sizeof(tmp), half - o);
    std::memcpy(tmp, p + o, n);
    std::memcpy(p + o, p + half + o, n);
    std::memcpy(p + half + o, tmp, n);
  }
}

}  // namespace

MetalMiniMaxH3Transformer::Linear
MetalMiniMaxH3Transformer::assemble_(
    WeightSet& ws, const std::string& nm,
    const std::vector<minimax_h3::Part>& parts, bool bias, Retain r)
{
  const MetalLlamaWeights& src = ws.src();
  // Every piece has to be a matrix of one width, and the rows it lends
  // have to exist.
  std::int64_t rows = 0, K = -1;
  fp8::Format f = fp8::Format::kNone;
  bool all_f8 = true;
  for (const auto& p : parts) {
    const auto* ti = src.info(p.layer + ".weight");
    if (ti == nullptr || ti->shape.size() != 2 ||
        p.row0 < 0 || p.rows <= 0 || p.row0 + p.rows > ti->shape[0] ||
        (K >= 0 && ti->shape[1] != K)) {
      return Linear{};
    }
    K = ti->shape[1];
    rows += p.rows;
    // int8 pieces are assembled dense (to_bf16_ decodes each with its
    // rotation): stacked codes would mix their scales and groups.
    const fp8::Weight* fw = _f8.find(p.layer + ".weight");
    // W4A8 pieces are not assembled: their nibble rows are half a row,
    // and no layout stores it in pieces.
    if (fw != nullptr && fw->format == fp8::Format::kW4A8) {
      return Linear{};
    }
    if (fw == nullptr || fw->format == fp8::Format::kI8 ||
        (f != fp8::Format::kNone && fw->format != f)) {
      all_f8 = false;
    } else {
      f = fw->format;
    }
  }
  // FP8 codes stacked, or bf16: one element's bytes, and the piece's
  // bytes as this weight stores them.
  const std::size_t elt = all_f8 ? 1 : 2;
  const std::size_t row_bytes = (std::size_t)K * elt;
  auto build = [&]() -> SharedBuffer {
    SharedBuffer d = _mc->make_shared_buffer((std::size_t)rows * row_bytes);
    if (d.empty()) { return {}; }
    auto* dp = static_cast<std::uint8_t*>(d.contents());
    std::size_t off = 0;
    for (const auto& p : parts) {
      // Read, not tensor(): the piece is CONSUMED into the assembly, and
      // caching it would keep a second copy beside the product.
      SharedBuffer piece =
          all_f8 ? ws.read(p.layer + ".weight", _mc,
                           WeightSet::Residency::Copied)
                 : to_bf16_(src, _mc, p.layer + ".weight");
      const std::size_t nb = (std::size_t)p.rows * row_bytes;
      if (piece.empty() ||
          piece.byte_size() < (std::size_t)(p.row0 + p.rows) * row_bytes) {
        return {};
      }
      std::memcpy(dp + off,
                  static_cast<const std::uint8_t*>(piece.contents()) +
                      (std::size_t)p.row0 * row_bytes,
                  nb);
      off += nb;
    }
    return d;
  };
  Linear l;
  const std::string key =
      std::string(all_f8 ? "h3-asm/f8|" : "h3-asm/bf16|") + nm;
  SharedBuffer w = r == Retain::Streamed ? ws.stream_derived(build)
                                         : ws.derived(key, build);
  if (w.empty()) { return Linear{}; }
  if (all_f8) {
    std::vector<float> sc;
    if (!assembled_scale_(src, _f8, parts, &sc)) { return Linear{}; }
    if (!sc.empty()) {
      l.f8_scale = _mc->make_shared_buffer(sc.size() * sizeof(float));
      if (l.f8_scale.empty()) { return Linear{}; }
      std::memcpy(l.f8_scale.contents(), sc.data(), sc.size() * sizeof(float));
      l.f8_per_row = true;
    }
    l.codes = std::move(w);
    l.f8    = f;
  } else {
    l.w = std::move(w);
  }
  // No H3 projection stored in pieces carries a bias; one that did would
  // need its pieces assembled too, and is refused rather than dropped.
  if (bias) {
    for (const auto& p : parts) {
      if (src.has(p.layer + ".bias")) { return Linear{}; }
    }
  }
  return l;
}

bool
MetalMiniMaxH3Transformer::refill_linear_(WeightSet& ws,
                                          const std::string& nm, Linear& l)
{
  const MetalLlamaWeights& src = ws.src();
  const auto parts = minimax_h3::assembled_parts(src, nm);
  if (parts.empty()) {
    if (!l.is_fp8() || l.codes.empty()) { return false; }
    // One FP8 tensor: the codes as they sit, the scale beside them.
    if (refill_streamed_tensor(ws, nm + ".weight", l.codes,
                               RefillDst::kRaw) != Refill::kFilled) {
      return false;
    }
    if (!l.f8_scale.empty()) {
      const fp8::Weight* fw = _f8.find(nm + ".weight");
      const auto* ti = src.info(nm + ".weight");
      std::vector<float> v;
      if (fw == nullptr || ti == nullptr || ti->shape.empty() ||
          !fp8::read_scale(src, *fw, (std::size_t)ti->shape[0], &v,
                           nullptr) ||
          v.size() * sizeof(float) != l.f8_scale.byte_size()) {
        return false;
      }
      std::memcpy(l.f8_scale.contents(), v.data(), l.f8_scale.byte_size());
    }
    if (l.f8 == fp8::Format::kW4A8) {
      const fp8::Weight* fw = _f8.find(nm + ".weight");
      if (fw == nullptr || l.f8_srel.empty() ||
          refill_streamed_tensor(ws, fw->s_rel, l.f8_srel,
                                 RefillDst::kRaw) != Refill::kFilled ||
          (!l.f8_cb.empty() &&
           !src.read_into(fw->codebook, l.f8_cb.contents(),
                          16 * sizeof(float)))) {
        return false;
      }
    }
    if (!l.b.empty() &&
        refill_streamed_tensor(ws, nm + ".bias", l.b, RefillDst::kBf16) !=
            Refill::kFilled) {
      return false;
    }
    return true;
  }
  if (!l.b.empty()) { return false; }
  const bool f8 = l.is_fp8();
  SharedBuffer& dst = f8 ? l.codes : l.w;
  if (dst.empty()) { return false; }
  const RefillDst as = f8 ? RefillDst::kRaw : RefillDst::kBf16;
  const auto* t0 = src.info(parts[0].layer + ".weight");
  if (t0 == nullptr || t0->shape.size() != 2) { return false; }
  const std::size_t row_bytes = (std::size_t)t0->shape[1] * (f8 ? 1 : 2);
  // The two assemblies this layout has: DISTINCT whole tensors stacked
  // (qkv), each refilled straight into its rows; and ONE tensor's halves
  // exchanged (fc1), refilled whole and swapped where it lies.
  const bool swapped = parts.size() == 2 &&
                       parts[0].layer == parts[1].layer &&
                       parts[1].row0 == 0 && parts[0].row0 == parts[1].rows &&
                       parts[0].rows == parts[1].rows;
  if (swapped) {
    if (refill_streamed_tensor(ws, parts[0].layer + ".weight", dst, as) !=
        Refill::kFilled) {
      return false;
    }
    swap_halves_(dst);
  } else {
    std::size_t off = 0;
    for (const auto& p : parts) {
      const auto* ti = src.info(p.layer + ".weight");
      if (ti == nullptr || ti->shape.size() != 2 || p.row0 != 0 ||
          p.rows != ti->shape[0]) {
        return false;
      }
      const std::size_t nb = (std::size_t)p.rows * row_bytes;
      if (off + nb > dst.byte_size() ||
          refill_streamed_tensor(ws, p.layer + ".weight",
                                 dst.subview(off, nb), as) !=
              Refill::kFilled) {
        return false;
      }
      off += nb;
    }
    if (off != dst.byte_size()) { return false; }
  }
  if (f8 && !l.f8_scale.empty()) {
    std::vector<float> sc;
    if (!assembled_scale_(src, _f8, parts, &sc) ||
        sc.size() * sizeof(float) != l.f8_scale.byte_size()) {
      return false;
    }
    std::memcpy(l.f8_scale.contents(), sc.data(), l.f8_scale.byte_size());
  }
  return true;
}

const MetalMiniMaxH3Transformer::Linear*
MetalMiniMaxH3Transformer::fp8_dense_(ComputeEncoder& enc, const Linear& l,
                                      Linear& view, int N, int K)
{
  if (!l.is_fp8()) { return &l; }
  const std::size_t need = (std::size_t)N * (std::size_t)K * 2;
  // The dequant scratch the quantized path widens into, shared the same
  // way: encoded in order into one stream, so each GEMM reads it before
  // the next expansion overwrites it.
  if (_w_deq.empty() || _w_deq.byte_size() < need) {
    _w_deq = _mc->make_shared_buffer(need);
    if (_w_deq.empty()) { _fp8_failed = true; return nullptr; }
  }
  fp8::Codes c;
  c.codes   = &l.codes;
  c.format  = l.f8;
  c.scale   = &l.f8_scale;
  c.per_row = l.f8_per_row;
  c.rot_group = l.f8_rot;
  c.s_rel    = &l.f8_srel;
  c.codebook = &l.f8_cb;
  c.qgroup   = l.f8_qgroup;
  if (!_fp8x.encode(enc, c, _w_deq, (std::size_t)N, (std::size_t)K)) {
    _fp8_failed = true;
    return nullptr;
  }
  // An alias of the scratch (same Metal buffer, retained), not a copy.
  view = Linear{};
  view.w = _w_deq.subview(0, need);
  if (view.w.empty()) { _fp8_failed = true; return nullptr; }
  return &view;
}

// The int8 layout widened into the same dequant scratch, for a shape the
// int8 GEMM declines (under its row floor): the rotation undone, so any
// dense route reads the weight it would have.
const MetalMiniMaxH3Transformer::Linear*
MetalMiniMaxH3Transformer::i8_dense_(ComputeEncoder& enc, const Linear& l,
                                     const SharedBuffer& codes,
                                     const SharedBuffer& cscales,
                                     Linear& view, int N, int K)
{
  const std::size_t need = (std::size_t)N * (std::size_t)K * 2;
  if (_w_deq.empty() || _w_deq.byte_size() < need) {
    _w_deq = _mc->make_shared_buffer(need);
    if (_w_deq.empty()) { return nullptr; }
  }
  if (l.cr > 0) {
    // ConvRot codes: widened as the int8 checkpoint is, the rotation
    // undone (the token layout's own dequant would take Sylvester's).
    fp8::Codes c;
    c.codes = &codes;
    c.format = fp8::Format::kI8;
    c.scale = &cscales;
    c.per_row = true;
    c.rot_group = l.cr;
    if (!_fp8x.encode(enc, c, _w_deq, (std::size_t)N, (std::size_t)K)) {
      return nullptr;
    }
  } else if (_i8 == nullptr ||
             !_i8->encode_dequant_fast(enc, codes, cscales, l.gc, _w_deq, N,
                                       K)) {
    return nullptr;
  }
  view = Linear{};
  view.w = _w_deq.subview(0, need);
  return view.w.empty() ? nullptr : &view;
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
    d.quantized  = s.quantized;
    d.bits       = s.bits;
    d.gu_inter   = s.gu_inter;
    d.f8         = s.f8;
    d.f8_per_row = s.f8_per_row;
    d.f8_rot     = s.f8_rot;
    d.f8_qgroup  = s.f8_qgroup;
    d.gc         = s.gc;
    d.cr         = s.cr;
    d.cr_w4      = s.cr_w4;
    d.n          = s.n;
    d.k          = s.k;
    d.i8x        = s.i8x;
    d.pc         = s.pc;
    // A streamed layout's refill sources are the slot's alone: a second
    // slot (shapes only) needs them, a promotion (bytes) keeps the layout
    // without them -- the rest of what it keeps is keep_block_'s.
    d.ds         = s.ds && !copy_bytes;
    d.ps         = s.ps && !copy_bytes;
    static const SharedBuffer kNone;
    const bool src_dead = copy_bytes && s.ds;
    const bool pack_dead = copy_bytes && s.ps;
    one(s.cscales, d.cscales);
    one(src_dead ? kNone : s.w, d.w);
    one(s.b, d.b);
    one(s.codes, d.codes);
    one(pack_dead ? kNone : s.scales, d.scales);
    one(pack_dead ? kNone : s.qbias, d.qbias);
    one(s.f8_scale, d.f8_scale);
    one(s.f8_srel, d.f8_srel);
    one(s.f8_cb, d.f8_cb);
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
                  nm, _mc, kept_residency_(_stream_blocks, _wire.on()))
            : weight_(ws, nm, Retain::Streamed);
    if (rebuilt.empty()) { ok = false; return; }
    dst = std::move(rebuilt);
    ++repaired;
  };
  // The streamed layouts whose products are rebuilt from what was just
  // refilled, all in one command buffer once the reads are in.
  std::vector<Linear*> convert;
  auto lin = [&](const std::string& nm, Linear& l) {
    if (!ok) { return; }
    // THE INT8 LAYOUT. A dense weight refills its bf16 source (`ds`), a
    // w8 pack its codes, scales and biases (`ps`) -- both converted after
    // the reads -- and a w4 pack converted per projection its pack as
    // stored (`pc`). Draw Things' own codes refill raw by name and the
    // f32 row scales are rebuilt: small, and a slot that kept them would
    // keep the FIRST block's.
    if (l.gc > 0 && (l.ds || l.ps || l.pc)) {
      if (l.ds) {
        one(nm + ".weight", l.w, /*raw=*/false);
      } else {
        one(nm + ".weight", l.codes, /*raw=*/true);
        one(nm + ".scales", l.scales, /*raw=*/false);
        one(nm + ".biases", l.qbias, /*raw=*/false);
      }
      if (!l.pc) { convert.push_back(&l); }
      return;
    }
    // ConvRot codes as stored (cr): the codes -- or a W4A8 weight's
    // nibbles and relative scales -- refilled raw by name; the row scales
    // and the codebook rebuilt, as Draw Things' are.
    if (l.gc > 0 && l.cr > 0) {
      const fp8::Weight* fw = _f8.find(nm + ".weight");
      if (fw == nullptr ||
          refill_streamed_tensor(ws, nm + ".weight", l.codes,
                                 RefillDst::kRaw) != Refill::kFilled ||
          (l.cr_w4 &&
           refill_streamed_tensor(ws, fw->s_rel, l.f8_srel,
                                  RefillDst::kRaw) != Refill::kFilled)) {
        ok = false;
        return;
      }
      if (!l.f8_cb.empty() &&
          !ws.src().read_into(fw->codebook, l.f8_cb.contents(),
                              16 * sizeof(float))) {
        ok = false;
        return;
      }
      l.cscales = convrot_row_scales_(ws.src(), *fw, l.n);
      if (l.cscales.empty()) { ok = false; }
      return;
    }
    if (l.gc > 0) {
      if (!l.i8x) { ok = false; return; }
      if (refill_streamed_tensor(ws, nm + ".i8x", l.codes,
                                 RefillDst::kRaw) != Refill::kFilled) {
        SharedBuffer rb = ws.stream_tensor(nm + ".i8x", _mc,
                                           WeightSet::Residency::Copied);
        if (rb.empty()) { ok = false; return; }
        l.codes = std::move(rb);
        ++repaired;
      }
      l.cscales = ws.stream_derived([&]() {
        return i8x_scales_f32_(ws.src(), _mc, nm + ".i8x_scales");
      });
      if (l.cscales.empty()) { ok = false; }
      return;
    }
    // The one thing the per-tensor path cannot repair, because it is not
    // a tensor: a Linear whose quantized-ness or bit width has changed
    // under the slot. Rebuild the whole Linear so its buffers and its
    // metadata cannot disagree.
    auto rebuild = [&]() {
      Linear fresh = linear_(ws, nm, !l.b.empty(), Retain::Streamed);
      if (fresh.empty()) { ok = false; return; }
      l = std::move(fresh);
      ++repaired;
    };
    if (!linear_matches_(ws, nm, l)) { rebuild(); return; }
    // FP8, and anything stored in pieces, refill as a whole Linear --
    // codes raw, scale beside them, pieces into their rows -- and are
    // rebuilt whole when any piece cannot be placed.
    if (l.is_fp8() || !minimax_h3::assembled_parts(ws.src(), nm).empty()) {
      if (!refill_linear_(ws, nm, l)) { rebuild(); }
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
  // Pruned, the AdaLN lives in the trunk (_padaln), never in a block.
  if (with_adaln && !_cfg.pruned()) {
    lin(prefix + "adaln_proj.linear", b.adaln);
  }
  if (ok && !convert.empty()) {
    if (!_cvt_stream.valid()) { _cvt_stream = _mc->make_command_stream(); }
    bool enc_ok = true;
    {
      ComputeEncoder e = _cvt_stream.begin_compute();
      for (Linear* l : convert) { enc_ok = encode_fast_(e, *l) && enc_ok; }
    }
    ok = _cvt_stream.commit().wait_ok(nullptr) && enc_ok;
  }
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
                                       bool main, bool with_adaln, Retain r)
{
  b.n1  = weight_(ws, prefix + "norm1.weight", r);
  b.n2  = weight_(ws, prefix + "norm2.weight", r);
  // A MAIN block's projections in the int8 layout where the tier asks
  // and the weight allows -- baked or not; the refiner (text rows, under
  // the int8 row floor) keeps its own.
  auto proj = [&](const std::string& nm) {
    Linear l = main ? i8_linear_(ws, nm, r) : Linear{};
    if (l.empty()) { return linear_(ws, nm, false, r); }
    return l;
  };
  b.qkv = proj(prefix + "attn.qkv_proj");
  b.out = proj(prefix + "attn.out_proj");
  b.qn  = weight_(ws, prefix + "attn.q_norm.weight", r);
  b.kn  = weight_(ws, prefix + "attn.k_norm.weight", r);
  // A block that is KEPT is built straight into the interleaved layout.
  // A STREAMED one is not: it is re-read every forward, so the
  // permutation would be paid 50 times a step and cost more than the
  // fusion saves. A streamed block later promoted to resident is
  // interleaved at the promotion, where it is paid once.
  {
    Linear l = main ? i8_linear_(ws, prefix + "mlp.fc1", r) : Linear{};
    b.fc1 = !l.empty() ? std::move(l)
            : (_fuse_ff && r == Retain::Cached)
                ? fc1_gu_(ws, prefix + "mlp.fc1")
                : linear_(ws, prefix + "mlp.fc1", false, r);
  }
  b.fc2 = proj(prefix + "mlp.fc2");
  // Pruned, the AdaLN lives in the trunk (_padaln), never in a block.
  if (with_adaln && !_cfg.pruned()) {
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
                                        float scale, std::size_t lora_off)
{
  const int r = lf.rank;
  _flops.gemm(M, r, K);
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
    enc.set_constant(8, scale);
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
                                        bool scale_applied, float scale,
                                        std::size_t lora_off)
{
  // The second GEMM contracts over the PER-PART rank of a banded pair,
  // reading its window of the [M, lf.rank] t -- see Factors::banded.
  const bool band = lf.banded();
  const int r = lf.b_rank();
  _flops.gemm(M, N, r);   // a banded B reads its group's share of rank
  const bool mma = _use_mma2 && M >= _mma_min_m;
  // The accumulating tiles have no scale of their own, so this route is
  // legal only once the FIRST GEMM applied the strength.
  const metal_compute::ComputeFunction* fb =
      (mma && scale_applied) ? lora_route_b_(r, N) : nullptr;
  if (band && fb != nullptr) {
    // The banded twin of the tile the rank chose, unless its width would
    // straddle a part's group -- the 256-wide tile over q/k/v grouped per
    // 128-wide head -- in which case the 128-wide one serves it.
    const bool wide = fb == &_fn_lora_b256 && lf.group % 256 == 0;
    fb = wide ? &_fn_lora_b256_band : &_fn_lora_b128_band;
    if (!fb->valid() || lf.group % 128 != 0) { fb = nullptr; }
  }
  if (band && fb == nullptr) {
    // Steel, banded: BN 32 divides every group this model has.
    enc.set_function(_fn_gemm_acc_band);
    enc.set_buffer(0, _s.lora, lora_off * 2);
    enc.set_buffer(1, lf.b);
    enc.set_buffer(2, lf.b);
    enc.set_buffer(3, y, y_off * 2);
    enc.set_constant(4, r);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.set_constant(8, scale_applied ? 1.0f : scale);
    enc.set_constant(9, lf.rank);
    enc.set_constant(10, lf.group);
    enc.set_constant(11, lf.parts);
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
    return;
  }
  enc.set_function(fb != nullptr ? *fb : _fn_gemm_acc);
  enc.set_buffer(0, _s.lora, lora_off * 2);
  enc.set_buffer(1, lf.b);
  enc.set_buffer(2, lf.b);
  enc.set_buffer(3, y, y_off * 2);
  enc.set_constant(4, r);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.set_constant(8, scale_applied ? 1.0f : scale);
  if (band) {                         // a banded mma twin, see above
    enc.set_constant(9, lf.rank);
    enc.set_constant(10, lf.group);
    enc.set_constant(11, lf.parts);
  }
  if (fb != nullptr) {
    const int BN =
        (fb == &_fn_lora_b256 || fb == &_fn_lora_b256_band) ? 256 : 128;
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
                                       const LoraStack& st,
                                       const SharedBuffer& y,
                                       std::size_t y_off, int M, int N, int K)
{
  if (st.empty() || _s.lora.empty() || !_fn_gemm_acc.valid()) { return; }
  // Strength 0 is dropped in lora_stack_, so a slot that reaches here
  // has something to add. Each one runs its own pair into its own
  // scratch region and accumulates onto the same y; the order they land
  // in does not matter because addition is what they do.
  for (int i = 0; i < st.n; ++i) {
    const LoraStack::Use& u = st.s[i];
    const bool scaled =
        lora_gemm_a_(enc, x, x_off, *u.f, M, K, u.scale, u.base);
    lora_gemm_b_(enc, *u.f, y, y_off, M, N, scaled, u.scale, u.base);
  }
}

// The live adapters for one projection, in slot order.
//
// An adapter that cannot or should not run is dropped HERE, so nothing
// downstream has to re-check it: a slot the file did not touch, a
// strength of zero (a legitimate request -- an A/B against the
// un-adapted model -- and the cheapest way to serve it is to encode
// nothing, which also makes "off" exactly off rather than a pair of
// roundings that cancel), or a build with no accumulating GEMM.
MetalMiniMaxH3Transformer::LoraStack
MetalMiniMaxH3Transformer::lora_stack_(bool refiner, int layer,
                                       LoraFactors BlockLora::* which) const
{
  LoraStack st;
  if (_s.lora.empty() || !_fn_gemm_acc.valid()) { return st; }
  for (std::size_t si = 0; si < _lora.size(); ++si) {
    const LoraSlot& sl = _lora[si];
    const std::vector<BlockLora>& v = refiner ? sl.refiner : sl.blocks;
    if (layer < 0 || (std::size_t)layer >= v.size()) { continue; }
    const LoraFactors* f = &(v[(std::size_t)layer].*which);
    // Spilled (shape only): the streamed copy read in for this block.
    if (f->empty() && f->rank > 0 && !refiner &&
        layer == _lora_stream_layer && si < _lora_stream.size()) {
      f = &(_lora_stream[si].*which);
    }
    st.add(*f, sl.scale, _lora_scratch_rows * (std::size_t)sl.rank_prefix);
  }
  return st;
}

// ---- adapter factors streamed with their block (see the header) ------

MetalMiniMaxH3Transformer::SpillFile::~SpillFile()
{
  if (fd >= 0) { ::close(fd); }
}

namespace {

bool
pwrite_all(int fd, const void* p, std::size_t n, std::uint64_t off)
{
  const auto* c = static_cast<const char*>(p);
  while (n > 0) {
    const ssize_t w = ::pwrite(fd, c, n, (off_t)off);
    if (w <= 0) { return false; }
    c += w;
    n -= (std::size_t)w;
    off += (std::uint64_t)w;
  }
  return true;
}

bool
pread_all(int fd, void* p, std::size_t n, std::uint64_t off)
{
  auto* c = static_cast<char*>(p);
  while (n > 0) {
    const ssize_t r = ::pread(fd, c, n, (off_t)off);
    if (r <= 0) { return false; }
    c += r;
    n -= (std::size_t)r;
    off += (std::uint64_t)r;
  }
  return true;
}

}  // namespace

bool
MetalMiniMaxH3Transformer::spill_lora_()
{
  if (_lora.empty() || std::getenv("VPIPE_H3_LORA_RESIDENT") != nullptr) {
    return false;
  }
  const std::string tmpl =
      (temp_root() / "vpipe-h3-lora-XXXXXX").string();
  std::vector<char> name(tmpl.begin(), tmpl.end());
  name.push_back('\0');
  const int fd = ::mkstemp(name.data());
  if (fd < 0) { return false; }
  ::unlink(name.data());   // gone with the descriptor, crash or not

  std::vector<std::vector<BlockSpill>> at(_lora.size());
  std::vector<BlockLora> stream(_lora.size());
  std::uint64_t off = 0;
  std::size_t spilled = 0;
  // Into the file first, every slot and block; only then is anything
  // freed, so a failed write leaves the model exactly as bound.
  for (std::size_t si = 0; si < _lora.size(); ++si) {
    LoraSlot& sl = _lora[si];
    at[si].resize(sl.blocks.size());
    for (std::size_t L = 0; L < sl.blocks.size(); ++L) {
      // A block resident now (pinned at load) keeps its factors.
      if (L < _blocks.size() && !_blocks[L].qkv.empty()) { continue; }
      for (std::size_t m = 0; m < std::size(kLoraMembers); ++m) {
        const LoraFactors& f = sl.blocks[L].*kLoraMembers[m];
        if (f.empty()) { continue; }
        LoraSpillAt& w = at[si][L].*kSpillMembers[m];
        w.a_off = off;
        w.a_bytes = f.a.byte_size();
        w.b_off = off + w.a_bytes;
        w.b_bytes = f.b.byte_size();
        if (!pwrite_all(fd, f.a.contents(), w.a_bytes, w.a_off) ||
            !pwrite_all(fd, f.b.contents(), w.b_bytes, w.b_off)) {
          ::close(fd);
          return false;
        }
        off += w.a_bytes + w.b_bytes;
        // The streamed set: one buffer pair per projection, the size of
        // the largest block's.
        LoraFactors& d = stream[si].*kLoraMembers[m];
        if (d.a.byte_size() < w.a_bytes) {
          d.a = _mc->make_shared_buffer(w.a_bytes);
        }
        if (d.b.byte_size() < w.b_bytes) {
          d.b = _mc->make_shared_buffer(w.b_bytes);
        }
        if (d.a.empty() || d.b.empty()) {
          ::close(fd);
          return false;
        }
        ++spilled;
      }
    }
  }
  if (spilled == 0) {
    ::close(fd);
    return false;
  }
  for (std::size_t si = 0; si < _lora.size(); ++si) {
    for (std::size_t L = 0; L < _lora[si].blocks.size(); ++L) {
      for (std::size_t m = 0; m < std::size(kLoraMembers); ++m) {
        if ((at[si][L].*kSpillMembers[m]).a_bytes == 0) { continue; }
        LoraFactors& f = _lora[si].blocks[L].*kLoraMembers[m];
        f.a = metal_compute::SharedBuffer{};   // the shape stays
        f.b = metal_compute::SharedBuffer{};
      }
    }
  }
  std::size_t held = 0;
  for (const BlockLora& b : stream) {
    for (auto m : kLoraMembers) {
      held += (b.*m).a.byte_size() + (b.*m).b.byte_size();
    }
  }
  _lora_spill_file.fd = fd;
  _lora_spill = std::move(at);
  _lora_stream = std::move(stream);
  _lora_stream_layer = -1;
  if (_mc->session() != nullptr) {
    _mc->session()->info(fmt(
        "MetalMiniMaxH3Transformer: adapter factors stream with their "
        "blocks -- {} MB spilled ({} projections), {} MB held for the "
        "streamed block", off >> 20, spilled, held >> 20));
  }
  return true;
}

bool
MetalMiniMaxH3Transformer::fill_lora_(int layer)
{
  if (_lora_spill_file.fd < 0 || layer == _lora_stream_layer) {
    return true;
  }
  for (std::size_t si = 0; si < _lora.size(); ++si) {
    if ((std::size_t)layer >= _lora_spill[si].size()) { continue; }
    const BlockLora& shape = _lora[si].blocks[(std::size_t)layer];
    for (std::size_t m = 0; m < std::size(kLoraMembers); ++m) {
      const LoraSpillAt& w =
          _lora_spill[si][(std::size_t)layer].*kSpillMembers[m];
      if (w.a_bytes == 0) { continue; }
      LoraFactors& d = _lora_stream[si].*kLoraMembers[m];
      const LoraFactors& f = shape.*kLoraMembers[m];
      // Given back with the slot pair once every block was kept
      // (release_idle_slots); made again for a block streamed since.
      if (d.a.byte_size() < w.a_bytes) {
        d.a = _mc->make_shared_buffer(w.a_bytes);
      }
      if (d.b.byte_size() < w.b_bytes) {
        d.b = _mc->make_shared_buffer(w.b_bytes);
      }
      if (d.a.empty() || d.b.empty() ||
          !pread_all(_lora_spill_file.fd, d.a.contents(), w.a_bytes,
                     w.a_off) ||
          !pread_all(_lora_spill_file.fd, d.b.contents(), w.b_bytes,
                     w.b_off)) {
        if (_mc->session() != nullptr) {
          _mc->session()->error(fmt(
              "MetalMiniMaxH3Transformer: adapter factors of block {} "
              "could not be read back", layer));
        }
        _lora_stream_layer = -1;
        return false;
      }
      d.rank = f.rank;
      d.parts = f.parts;
      d.group = f.group;
    }
  }
  _lora_stream_layer = layer;
  return true;
}

void
MetalMiniMaxH3Transformer::keep_lora_(int layer)
{
  if (_lora_spill_file.fd < 0 || layer != _lora_stream_layer) { return; }
  for (std::size_t si = 0; si < _lora.size(); ++si) {
    if ((std::size_t)layer >= _lora_spill[si].size()) { continue; }
    for (std::size_t m = 0; m < std::size(kLoraMembers); ++m) {
      const LoraSpillAt& w =
          _lora_spill[si][(std::size_t)layer].*kSpillMembers[m];
      if (w.a_bytes == 0) { continue; }
      LoraFactors& f =
          _lora[si].blocks[(std::size_t)layer].*kLoraMembers[m];
      const LoraFactors& src = _lora_stream[si].*kLoraMembers[m];
      metal_compute::SharedBuffer a = _mc->make_shared_buffer(w.a_bytes);
      metal_compute::SharedBuffer b = _mc->make_shared_buffer(w.b_bytes);
      if (a.empty() || b.empty()) { continue; }   // stays streamed
      std::memcpy(a.contents(), src.a.contents(), w.a_bytes);
      std::memcpy(b.contents(), src.b.contents(), w.b_bytes);
      f.a = std::move(a);
      f.b = std::move(b);
    }
  }
}

std::size_t
MetalMiniMaxH3Transformer::lora_layer_bytes_(int layer) const
{
  if (_lora_spill_file.fd < 0) { return 0; }
  std::size_t n = 0;
  for (const auto& per_block : _lora_spill) {
    if ((std::size_t)layer >= per_block.size()) { continue; }
    for (auto m : kSpillMembers) {
      const LoraSpillAt& w = per_block[(std::size_t)layer].*m;
      n += (std::size_t)(w.a_bytes + w.b_bytes);
    }
  }
  return n;
}

std::size_t
MetalMiniMaxH3Transformer::wire_lora_(int layer, bool on)
{
  if (_lora_spill_file.fd < 0) { return 0; }
  std::size_t changed = 0;
  for (std::size_t si = 0; si < _lora.size(); ++si) {
    if ((std::size_t)layer >= _lora_spill[si].size()) { continue; }
    for (std::size_t m = 0; m < std::size(kLoraMembers); ++m) {
      if ((_lora_spill[si][(std::size_t)layer].*kSpillMembers[m])
              .a_bytes == 0) {
        continue;
      }
      LoraFactors& f =
          _lora[si].blocks[(std::size_t)layer].*kLoraMembers[m];
      changed += _wire.wire_one(_mc, f.a, on);
      changed += _wire.wire_one(_mc, f.b, on);
    }
  }
  return changed;
}

void
MetalMiniMaxH3Transformer::drop_lora_(int layer)
{
  if (_lora_spill_file.fd < 0) { return; }
  for (std::size_t si = 0; si < _lora.size(); ++si) {
    if ((std::size_t)layer >= _lora_spill[si].size()) { continue; }
    for (std::size_t m = 0; m < std::size(kLoraMembers); ++m) {
      if ((_lora_spill[si][(std::size_t)layer].*kSpillMembers[m])
              .a_bytes == 0) {
        continue;
      }
      LoraFactors& f =
          _lora[si].blocks[(std::size_t)layer].*kLoraMembers[m];
      f.a = metal_compute::SharedBuffer{};
      f.b = metal_compute::SharedBuffer{};
    }
  }
  if (layer == _lora_stream_layer) { _lora_stream_layer = -1; }
}

MetalMiniMaxH3Transformer::LoraStack
MetalMiniMaxH3Transformer::lora_stack_final_() const
{
  LoraStack st;
  if (_s.lora.empty() || !_fn_gemm_acc.valid()) { return st; }
  for (const LoraSlot& sl : _lora) {
    st.add(sl.final_adaln, sl.scale,
           _lora_scratch_rows * (std::size_t)sl.rank_prefix);
  }
  return st;
}

int
MetalMiniMaxH3Transformer::lora_modules() const
{
  int n = 0;
  for (const LoraSlot& sl : _lora) { n += sl.modules; }
  return n;
}

int
MetalMiniMaxH3Transformer::lora_banded(int slot) const
{
  return (slot >= 0 && (std::size_t)slot < _lora.size())
      ? _lora[(std::size_t)slot].banded : 0;
}

std::size_t
MetalMiniMaxH3Transformer::lora_bytes() const
{
  auto fb = [](const LoraFactors& f) {
    return f.a.byte_size() + f.b.byte_size();
  };
  std::size_t n = 0;
  for (const LoraSlot& sl : _lora) {
    for (const auto* v : {&sl.blocks, &sl.refiner}) {
      for (const BlockLora& bl : *v) {
        n += fb(bl.qkv) + fb(bl.out) + fb(bl.fc1) + fb(bl.fc2) +
             fb(bl.adaln);
      }
    }
    n += fb(sl.final_adaln);
  }
  // ...and the set a streamed block's factors are read into.
  for (const BlockLora& bl : _lora_stream) {
    n += fb(bl.qkv) + fb(bl.out) + fb(bl.fc1) + fb(bl.fc2) + fb(bl.adaln);
  }
  return n;
}

namespace {

// The main block an adapter tensor belongs to, or -1: diffusers'
// `transformer_blocks.<i>.`, the release's `blocks.<i>.`, kohya's
// `_blocks_<i>_`. A refiner block is not a main block. `end` is where
// the index ends: the name past it is the projection's in every block.
int
adapter_main_block_(const std::string& n, std::size_t* end)
{
  if (n.find("refiner") != std::string::npos) { return -1; }
  std::size_t p = 0;
  while ((p = n.find("blocks", p)) != std::string::npos) {
    std::size_t q = p + 6;
    p = q;
    if (q + 1 >= n.size() || (n[q] != '.' && n[q] != '_') ||
        std::isdigit((unsigned char)n[q + 1]) == 0) {
      continue;
    }
    int v = 0;
    for (++q; q < n.size() && std::isdigit((unsigned char)n[q]) != 0; ++q) {
      v = v * 10 + (n[q] - '0');
    }
    if (q < n.size() && (n[q] == '.' || n[q] == '_')) {
      *end = q;
      return v;
    }
  }
  return -1;
}

}  // namespace

MetalMiniMaxH3Transformer::LoraFootprint
MetalMiniMaxH3Transformer::lora_footprint(const std::string& path,
                                          int n_layers, const Config* pruned)
{
  LoraFootprint fp;
  auto wts = MetalLlamaWeights::open_model(path);
  if (!wts.has_value()) { return fp; }
  // A PRUNED DiT's AdaLN modules: one f32 [D * 9] delta a projection,
  // whether the file carries factors or `.diff` / `.diff_b` -- held in
  // the trunk, streamed or not -- plus the patch scratch once.
  auto adaln_ = [](const std::string& n) {
    return n.find("adaln_proj.linear") != std::string::npos ||
           n.find("norm_out.linear") != std::string::npos;
  };
  std::set<std::string> deltas;
  if (pruned != nullptr && pruned->pruned()) {
    for (const std::string& nm : wts->tensor_names()) {
      if (!adaln_(nm)) { continue; }
      const std::size_t p = nm.find(".lora_") != std::string::npos
                                ? nm.find(".lora_")
                            : nm.find(".diff") != std::string::npos
                                ? nm.find(".diff")
                                : std::string::npos;
      if (p == std::string::npos) { continue; }
      std::size_t end = 0;
      const int L = adapter_main_block_(nm, &end);
      if (L >= 0 && n_layers > 0 && L >= n_layers) { continue; }
      deltas.insert(nm.substr(0, p));
    }
    for (const std::string& mod : deltas) {
      const bool fin = mod.find("final_layer") != std::string::npos ||
                       mod.find("norm_out") != std::string::npos;
      const std::size_t d =
          (std::size_t)(fin ? 2 * pruned->hidden : pruned->adaln_out());
      fp.whole += d * 9 * 4;
    }
    if (!deltas.empty()) {
      fp.whole += (std::size_t)std::max(pruned->adaln_out(),
                                        2 * pruned->hidden) * 9 * 4;
    }
    // The released MLP, to the host in f32, for an adapter that moves
    // the embedding (its tables are the denoise's: off_curve_bytes).
    if (moves_time_embedding(path) && !pruned->time_mlp_dir.empty()) {
      if (auto m = MetalLlamaWeights::open_model(pruned->time_mlp_dir)) {
        for (const char* nm : {"time_embedder.proj_in.weight",
                               "time_embedder.proj_in.bias",
                               "time_embedder.proj_out.weight",
                               "time_embedder.proj_out.bias"}) {
          if (const auto* ti = m->info(nm)) {
            std::size_t n = 1;
            for (auto x : ti->shape) { n *= (std::size_t)x; }
            fp.whole += n * 4;
          }
        }
      }
    }
  }
  const bool skip_adaln = pruned != nullptr && pruned->pruned();
  // A factor is `<module><mark>...weight`: peft's `.lora_A.[x.]weight`,
  // kohya's `.lora_down.weight`; an FP8 factor's scale is not one.
  static const char* const kA[] = {".lora_A.", ".lora_down.", ".lora_a."};
  static const char* const kB[] = {".lora_B.", ".lora_up.", ".lora_b."};
  auto mark = [](const std::string& n, const char* const* m, std::size_t k) {
    for (std::size_t i = 0; i < k; ++i) {
      const std::size_t p = n.find(m[i]);
      if (p != std::string::npos) { return p; }
    }
    return std::string::npos;
  };
  std::size_t main = 0;
  std::map<std::string, std::size_t> per_role;   // largest across blocks
  std::map<std::string, int> fused;              // q|k|v ranks summed
  for (const std::string& nm : wts->tensor_names()) {
    const auto* ti = wts->info(nm);
    if (ti == nullptr || nm.size() < 7 ||
        nm.compare(nm.size() - 7, 7, ".weight") != 0) {
      continue;
    }
    const std::size_t pa = mark(nm, kA, std::size(kA));
    const std::size_t pb = pa == std::string::npos
                               ? mark(nm, kB, std::size(kB))
                               : std::string::npos;
    if (pa == std::string::npos && pb == std::string::npos) { continue; }
    if (skip_adaln && adaln_(nm)) { continue; }   // counted above
    std::size_t end = 0;
    const int L = adapter_main_block_(nm, &end);
    if (L >= 0 && n_layers > 0 && L >= n_layers) { continue; }
    std::size_t n = 1;
    for (auto d : ti->shape) { n *= (std::size_t)d; }
    const std::size_t b = n * 2;
    fp.whole += b;
    if (L >= 0) {
      main += b;
      std::size_t& most = per_role[nm.substr(end)];
      most = std::max(most, b);
    }
    if (pa != std::string::npos && !ti->shape.empty()) {
      const int r = (int)ti->shape[0];
      const std::string mod = nm.substr(0, pa);
      const bool qkv = mod.size() >= 4 &&
                       (mod.compare(mod.size() - 4, 4, "to_q") == 0 ||
                        mod.compare(mod.size() - 4, 4, "to_k") == 0 ||
                        mod.compare(mod.size() - 4, 4, "to_v") == 0);
      const int w = qkv ? (fused[mod.substr(0, mod.size() - 4)] += r) : r;
      fp.rank = std::max(fp.rank, w);
    }
  }
  std::size_t set = 0;
  for (const auto& [role, b] : per_role) { set += b; }
  fp.streamed = main > 0 ? fp.whole - main + set : fp.whole;
  return fp;
}

bool
MetalMiniMaxH3Transformer::moves_time_embedding(const std::string& path)
{
  for (const char* m : {"time_embedder.linear_", "time_embedder.proj_",
                        "endpoint_time_embedder"}) {
    if (lora::Adapter::file_touches(path, m)) { return true; }
  }
  return false;
}

std::size_t
MetalMiniMaxH3Transformer::off_curve_bytes(const Config& c, int max_rows)
{
  if (!c.pruned() || max_rows <= 0) { return 0; }
  const std::size_t T = (std::size_t)max_rows;
  const std::size_t D = (std::size_t)c.adaln_out();
  const std::size_t FD = (std::size_t)(2 * c.hidden);
  // The released embedding's width: the architecture's, 2688 on H3.
  const std::size_t M = 2688;
  return T * D * 2 * (std::size_t)c.n_layers + T * FD * 2 +   // tables
         D * M * 2 +                                          // one block
         T * M * 2;                                           // the rows
}

int
MetalMiniMaxH3Transformer::lora_modules(int slot) const
{
  return (slot >= 0 && (std::size_t)slot < _lora.size())
      ? _lora[(std::size_t)slot].modules : 0;
}

void
MetalMiniMaxH3Transformer::set_lora_scale(int slot, float s)
{
  if (slot >= 0 && (std::size_t)slot < _lora.size()) {
    _lora[(std::size_t)slot].scale = s;
  }
}

float
MetalMiniMaxH3Transformer::lora_scale(int slot) const
{
  return (slot >= 0 && (std::size_t)slot < _lora.size())
      ? _lora[(std::size_t)slot].scale : 0.0f;
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

void
MetalMiniMaxH3Transformer::curve_coords(const std::vector<float>& table,
                                        int knots,
                                        const std::vector<float>& t,
                                        float* out)
{
  for (std::size_t i = 0; i < t.size(); ++i) {
    // f32 throughout, as the reference's float tensors: pos = t * 1024,
    // lo = floor clamped to knots - 2, w = pos - lo.
    const float pos = t[i] * (float)(knots - 1);
    int lo = (int)std::floor(pos);
    lo = std::max(0, std::min(lo, knots - 2));
    const float w = pos - (float)lo;
    const float* a = table.data() + (std::size_t)lo * 8;
    const float* b = a + 8;
    for (int k = 0; k < 8; ++k) {
      // torch.lerp's two-sided form: exact at both ends.
      const float d = b[k] - a[k];
      out[i * 8 + (std::size_t)k] =
          w < 0.5f ? a[k] + w * d : b[k] - d * (1.0f - w);
    }
  }
}

// AdaLN curve branches (see CurveSet). The file names them in either
// spelling of the model -- `transformer_blocks.N.` (diffusers) or
// `blocks.N.` -- under `transformer.` / `diffusion_model.` or bare, and
// the curve TABLE is a tensor `adaln_t_table` in the file itself or a
// file beside it: `adaln_t_table.bin` (raw little-endian f32, what the
// release's download script writes) or `adaln_t_table.safetensors`.
bool
MetalMiniMaxH3Transformer::bind_curves_(const std::string& path,
                                        LoraSlot& slot, bool* took,
                                        std::string* err)
{
  *took = false;
  auto no = [&](std::string m) {
    if (err != nullptr) { *err = "minimax-h3 curve branches: " + m; }
    return false;
  };
  if (!lora::Adapter::file_touches(path, ".curve_A.")) { return true; }
  std::shared_ptr<WeightSet> ws = WeightSet::open(path, nullptr);
  if (!ws) { return no("cannot open '" + path + "'"); }
  const Config& c = _cfg;
  const int D = c.adaln_out();
  std::vector<std::string> an((std::size_t)c.n_layers),
      bn((std::size_t)c.n_layers);
  const std::string kA = ".adaln_proj.linear.curve_A.weight";
  const std::string kB = ".adaln_proj.linear.curve_B.weight";
  int unplaced = 0;
  for (const std::string& n : ws->src().tensor_names()) {
    const bool is_a = n.size() > kA.size() &&
                      n.compare(n.size() - kA.size(), kA.size(), kA) == 0;
    const bool is_b = n.size() > kB.size() &&
                      n.compare(n.size() - kB.size(), kB.size(), kB) == 0;
    if (!is_a && !is_b) { continue; }
    std::string k = n.substr(0, n.size() - kA.size());
    for (const char* pre : {"model.diffusion_model.", "diffusion_model.",
                            "transformer."}) {
      const std::string ps(pre);
      if (k.compare(0, ps.size(), ps) == 0) { k = k.substr(ps.size()); }
    }
    int layer = -1;
    for (const char* blk : {"transformer_blocks.", "blocks."}) {
      const std::string bs(blk);
      if (k.compare(0, bs.size(), bs) == 0) {
        const std::string idx = k.substr(bs.size());
        if (!idx.empty() &&
            idx.find_first_not_of("0123456789") == std::string::npos) {
          layer = std::atoi(idx.c_str());
        }
      }
    }
    if (layer < 0 || layer >= c.n_layers) { ++unplaced; continue; }
    (is_a ? an : bn)[(std::size_t)layer] = n;
  }
  // Outside this DiT's blocks is SKIPPED and said, as an adapter's
  // unmatched modules are: a DiT built with fewer blocks (a test's) is
  // not a reason to refuse the rest.
  if (unplaced != 0 && _mc->session() != nullptr) {
    _mc->session()->info(fmt("MetalMiniMaxH3Transformer: '{}' names {} "
                             "curve tensors outside this DiT's {} blocks; "
                             "skipped", path, unplaced, c.n_layers));
  }

  CurveSet& cs = slot.curve;
  cs = CurveSet{};
  // B A per block, in double: [D, 8]. A branch whose B is all zero adds
  // nothing at any timestep and is dropped -- Open-H3-Regenerate-2K's 50
  // are all zero (LMS Sharp v1.0's curve_B as shipped), so that file
  // binds as an exact no-op and needs no table.
  std::vector<SharedBuffer> wc((std::size_t)c.n_layers);
  int zero = 0;
  for (int L = 0; L < c.n_layers; ++L) {
    const std::string& a_nm = an[(std::size_t)L];
    const std::string& b_nm = bn[(std::size_t)L];
    if (a_nm.empty() != b_nm.empty()) {
      return no(fmt("block {} has only one of curve_A / curve_B", L)());
    }
    if (a_nm.empty()) { continue; }
    const auto* ai = ws->src().info(a_nm);
    const auto* bi = ws->src().info(b_nm);
    if (ai == nullptr || bi == nullptr || ai->shape.size() != 2 ||
        bi->shape.size() != 2 || ai->shape[1] != 8 ||
        bi->shape[0] != D || bi->shape[1] != ai->shape[0]) {
      return no(fmt("block {}'s curve branch is not A [r, 8], B [{}, r]", L,
                    D)());
    }
    const int r = (int)ai->shape[0];
    std::vector<float> A, B;
    if (!to_host_f32_(*ws, _mc, a_nm, A) || !to_host_f32_(*ws, _mc, b_nm, B)) {
      return no(fmt("cannot read block {}'s curve branch", L)());
    }
    if (std::all_of(B.begin(), B.end(), [](float v) { return v == 0.0f; })) {
      ++zero;
      continue;
    }
    SharedBuffer w = _mc->make_shared_buffer((std::size_t)D * 8 * 4);
    if (w.empty()) { return no("allocation failed"); }
    auto* wp = static_cast<float*>(w.contents());
    for (int j = 0; j < D; ++j) {
      const float* br = B.data() + (std::size_t)j * r;
      for (int k = 0; k < 8; ++k) {
        double acc = 0.0;
        for (int q = 0; q < r; ++q) {
          acc += (double)br[q] * (double)A[(std::size_t)q * 8 + k];
        }
        wp[(std::size_t)j * 8 + k] = (float)acc;
      }
    }
    wc[(std::size_t)L] = std::move(w);
    ++cs.branches;
    cs.rank = std::max(cs.rank, r);
  }
  *took = cs.branches + zero > 0;
  if (cs.branches == 0) {
    if (_mc->session() != nullptr && zero > 0) {
      _mc->session()->info(fmt(
          "MetalMiniMaxH3Transformer: '{}': {} AdaLN curve branches, every "
          "one with B = 0 -- they add nothing, so none is bound", path,
          zero));
    }
    return true;
  }

  // The table, needed only now that something reads it.
  if (ws->src().info("adaln_t_table") != nullptr) {
    if (!to_host_f32_(*ws, _mc, "adaln_t_table", cs.table)) {
      return no("cannot read adaln_t_table in '" + path + "'");
    }
  } else {
    const std::filesystem::path dir =
        std::filesystem::path(path).parent_path();
    const std::filesystem::path bin = dir / "adaln_t_table.bin";
    const std::filesystem::path st = dir / "adaln_t_table.safetensors";
    std::error_code ec;
    if (std::filesystem::is_regular_file(bin, ec)) {
      std::ifstream f(bin, std::ios::binary | std::ios::ate);
      const std::streamsize bytes = f ? (std::streamsize)f.tellg() : 0;
      if (bytes <= 0 || bytes % 32 != 0) {
        return no(fmt("'{}' is {} bytes, not rows of 8 f32", bin.string(),
                      (long long)bytes)());
      }
      cs.table.resize((std::size_t)bytes / 4);
      f.seekg(0);
      f.read(reinterpret_cast<char*>(cs.table.data()), bytes);
    } else if (std::filesystem::is_regular_file(st, ec)) {
      std::shared_ptr<WeightSet> tw = WeightSet::open(st.string(), nullptr);
      if (!tw || !to_host_f32_(*tw, _mc, "adaln_t_table", cs.table)) {
        return no("cannot read adaln_t_table from '" + st.string() + "'");
      }
    } else {
      return no(fmt(
          "'{}' carries AdaLN curve branches but no curve table: they read "
          "an 8-D point off `adaln_t_table` (1025 x 8; ComfyUI's pruned "
          "MiniMax-H3 checkpoints carry it), and neither the file nor "
          "adaln_t_table.bin / .safetensors beside it has one", path)());
    }
  }
  if (cs.table.size() < 16 || cs.table.size() % 8 != 0) {
    return no(fmt("the curve table has {} values, not rows of 8",
                  cs.table.size())());
  }
  cs.knots = (int)(cs.table.size() / 8);
  if (!_fn_curve_add.valid()) {
    if (!_lib_regen.valid()) {
      _lib_regen = _mc->load_library("minimax_h3_regenerate_bf16");
    }
    _fn_curve_add = _lib_regen.function("h3_adaln_curve_add_f16");
    if (!_fn_curve_add.valid()) {
      return no("h3_adaln_curve_add_f16 is missing from this build");
    }
  }
  cs.wc = std::move(wc);
  if (_mc->session() != nullptr) {
    _mc->session()->info(fmt(
        "MetalMiniMaxH3Transformer: '{}': {} AdaLN curve branches (rank {}, "
        "a {}-knot table) at scale {:.3g}, B A held f32 ({} MB)", path,
        cs.branches, cs.rank, cs.knots, slot.scale,
        ((std::size_t)cs.branches * D * 8 * 4) >> 20));
  }
  return true;
}

bool
MetalMiniMaxH3Transformer::curve_live_(int L) const
{
  for (const LoraSlot& sl : _lora) {
    if (!sl.curve.empty() && sl.scale != 0.0f &&
        (std::size_t)L < sl.curve.wc.size() &&
        !sl.curve.wc[(std::size_t)L].empty()) {
      return true;
    }
  }
  return false;
}

void
MetalMiniMaxH3Transformer::curve_apply_(ComputeEncoder& enc, int L,
                                        const SharedBuffer& mod, int n_t)
{
  const int D = _cfg.adaln_out();
  int ci = 0;
  for (const LoraSlot& sl : _lora) {
    if (sl.curve.empty()) { continue; }
    const int at = ci++;
    if (sl.scale == 0.0f || (std::size_t)L >= sl.curve.wc.size() ||
        sl.curve.wc[(std::size_t)L].empty() || n_t != _curve_n_t) {
      continue;
    }
    enc.set_function(_fn_curve_add);
    enc.set_buffer(0, mod);
    enc.set_buffer(1, sl.curve.wc[(std::size_t)L]);
    enc.set_buffer(2, _curve_coords, (std::size_t)at * n_t * 8 * 4);
    enc.set_constant(3, D);
    enc.set_constant(4, n_t);
    enc.set_constant(5, sl.scale);
    enc.dispatch({(unsigned)D, (unsigned)n_t, 1}, {256, 1, 1});
    _flops.add(2ull * (std::uint64_t)D * 8 * (std::uint64_t)n_t);
  }
}

void
MetalMiniMaxH3Transformer::pruned_mod_(ComputeEncoder& enc,
                                       const PrunedAdaln& p, int L,
                                       const SharedBuffer& out, int n_t,
                                       int D, const SharedBuffer* res,
                                       std::size_t res_off)
{
  // The live slots' deltas for this projection, added to the weight and
  // the bias in f32, slot by slot, into the patch scratch -- which the
  // projection then reads as F32. Serial dispatches on one encoder, so a
  // block's patch is read before the next block's overwrites it.
  const std::size_t nw = (std::size_t)D * 8;
  bool first = true;
  for (const LoraSlot& sl : _lora) {
    const SharedBuffer* dp =
        L < 0 ? &sl.pdelta_final
              : (std::size_t)L < sl.pdelta.size() ? &sl.pdelta[(std::size_t)L]
                                                  : nullptr;
    if (dp == nullptr || dp->empty() || sl.scale == 0.0f ||
        _padaln_patch.empty()) {
      continue;
    }
    const SharedBuffer& d = *dp;
    for (int part = 0; part < 2; ++part) {
      const int n = part == 0 ? (int)nw : D;
      const std::size_t off = part == 0 ? 0 : nw * 4;
      enc.set_function(_fn_pruned_patch);
      enc.set_buffer(0, _padaln_patch, off);
      enc.set_buffer(1, part == 0 ? p.w : p.b);
      enc.set_buffer(2, d, off);
      enc.set_constant(3, n);
      enc.set_constant(4, p.dtype);
      enc.set_constant(5, sl.scale);
      enc.set_constant(6, first ? 1 : 0);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    }
    first = false;
  }
  enc.set_function(res != nullptr ? _fn_pruned_res : _fn_pruned_mod);
  enc.set_buffer(0, out);
  if (res != nullptr) { enc.set_buffer(7, *res, res_off); }
  if (first) {
    enc.set_buffer(1, p.w);
    enc.set_buffer(2, p.b);
  } else {
    enc.set_buffer(1, _padaln_patch);
    enc.set_buffer(2, _padaln_patch, nw * 4);
  }
  enc.set_buffer(3, _pcoords);
  enc.set_constant(4, D);
  enc.set_constant(5, n_t);
  enc.set_constant(6, first ? p.dtype : 0);
  enc.dispatch({(unsigned)D, (unsigned)n_t, 1}, {256, 1, 1});
  _flops.add(2ull * (std::uint64_t)D * 8 * (std::uint64_t)n_t);
}

namespace {

// silu(e) rounded to bf16, as the released forward hands it to the AdaLN
// (time_embed_), back in f32.
float
silu_bf16_(float e)
{
  const std::uint32_t u = (std::uint32_t)f32_to_bf16_(e / (1.0f + std::exp(-e)))
                          << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// A in place, the n x n SPD system's Cholesky factor, then B (n x nrhs,
// row-major) solved in place. False when A is not positive definite.
bool
chol_solve_(std::vector<double>& A, int n, double* B, int nrhs)
{
  for (int j = 0; j < n; ++j) {
    double d = A[(std::size_t)j * n + j];
    for (int k = 0; k < j; ++k) {
      d -= A[(std::size_t)j * n + k] * A[(std::size_t)j * n + k];
    }
    if (!(d > 0.0)) { return false; }
    const double l = std::sqrt(d);
    A[(std::size_t)j * n + j] = l;
    for (int i = j + 1; i < n; ++i) {
      double v = A[(std::size_t)i * n + j];
      for (int k = 0; k < j; ++k) {
        v -= A[(std::size_t)i * n + k] * A[(std::size_t)j * n + k];
      }
      A[(std::size_t)i * n + j] = v / l;
    }
  }
  for (int c = 0; c < nrhs; ++c) {
    for (int i = 0; i < n; ++i) {        // L y = b
      double v = B[(std::size_t)i * nrhs + c];
      for (int k = 0; k < i; ++k) {
        v -= A[(std::size_t)i * n + k] * B[(std::size_t)k * nrhs + c];
      }
      B[(std::size_t)i * nrhs + c] = v / A[(std::size_t)i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {   // L^T x = y
      double v = B[(std::size_t)i * nrhs + c];
      for (int k = i + 1; k < n; ++k) {
        v -= A[(std::size_t)k * n + i] * B[(std::size_t)k * nrhs + c];
      }
      B[(std::size_t)i * nrhs + c] = v / A[(std::size_t)i * n + i];
    }
  }
  return true;
}

}  // namespace

// The pruned curve's basis in the RELEASED embedding. A pruned file kept
// only the AdaLN's response along the curve t -> silu(temb(t)), which the
// released MLP traces in an 8-dimensional affine span -- MEASURED on
// MiniMax-H3 FL2VA and Ref2VA: under 3e-9 of the curve's energy outside
// 8 dimensions, and a community pruned file's table spans either
// partition's curve to a relative residual of 1.5e-5 / 2.1e-5. So
// least squares over the table recovers U and m, and with them a
// full-model adapter's AdaLN delta B A silu(temb(t)) is (B A U) c(t) +
// B A m: an 8-wide weight delta and a bias delta, exact to that residual.
bool
MetalMiniMaxH3Transformer::ensure_curve_basis_(std::string* err)
{
  auto no = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (!_basis.empty()) { return true; }
  if (!_cfg.pruned()) { return no("not a pruned DiT"); }
  if (_cfg.time_mlp_dir.empty()) {
    return no("no checkpoint with the released timestep MLP was named "
              "(Config::time_mlp_dir; generate-video names the graph's "
              "own partition)");
  }
  std::shared_ptr<WeightSet> ws = WeightSet::open(_cfg.time_mlp_dir, nullptr);
  if (!ws) { return no("cannot open '" + _cfg.time_mlp_dir + "'"); }
  const auto* in = ws->src().info("time_embedder.proj_in.weight");
  const auto* ou = ws->src().info("time_embedder.proj_out.weight");
  if (in == nullptr || ou == nullptr || in->shape.size() != 2 ||
      ou->shape.size() != 2 || ou->shape[1] != in->shape[0]) {
    return no("'" + _cfg.time_mlp_dir + "' carries no timestep MLP");
  }
  if (!to_host_f32_(*ws, _mc, "time_embedder.proj_in.weight", _time_in_w) ||
      !to_host_f32_(*ws, _mc, "time_embedder.proj_in.bias", _time_in_b) ||
      !to_host_f32_(*ws, _mc, "time_embedder.proj_out.weight",
                    _time_out_w) ||
      !to_host_f32_(*ws, _mc, "time_embedder.proj_out.bias", _time_out_b)) {
    return no("cannot read the timestep MLP of '" + _cfg.time_mlp_dir + "'");
  }
  _cfg.freq_dim = (int)in->shape[1];
  _cfg.time_hidden = (int)in->shape[0];
  const int M = (int)ou->shape[0];
  _basis.mlp = M;   // time_mlp_row_ reads the width through mlp_dim_()
  // S = silu(temb(t)) at every 4th knot (257 points: the curve is smooth
  // and its span 8-dimensional), the base MLP alone, in parallel.
  const int G = _cfg.curve_grid, step = 4;
  std::vector<int> knots;
  for (int g = 0; g < G; g += step) { knots.push_back(g); }
  if (knots.back() != G - 1) { knots.push_back(G - 1); }
  const int n = (int)knots.size();
  std::vector<float> S((std::size_t)n * M);
  {
    const int parts = 8;
    std::vector<std::future<void>> fs;
    for (int pi = 0; pi < parts; ++pi) {
      fs.push_back(std::async(std::launch::async, [&, pi]() {
        for (int i = pi; i < n; i += parts) {
          float* row = S.data() + (std::size_t)i * M;
          time_mlp_row_((double)knots[(std::size_t)i] / (double)(G - 1),
                        false, row, /*base_only=*/true);
          for (int j = 0; j < M; ++j) {
            row[j] = row[j] / (1.0f + std::exp(-row[j]));
          }
        }
      }));
    }
    for (auto& f : fs) { f.get(); }
  }
  // Least squares, S ~ [T | 1] X: the 9 x 9 normal matrix, X [9, M].
  std::vector<double> A(81, 0.0), X((std::size_t)9 * M, 0.0);
  for (int i = 0; i < n; ++i) {
    double ta[9];
    for (int k = 0; k < 8; ++k) {
      ta[k] = _adaln_t_table[(std::size_t)knots[(std::size_t)i] * 8 + k];
    }
    ta[8] = 1.0;
    for (int a = 0; a < 9; ++a) {
      for (int b = 0; b < 9; ++b) {
        A[(std::size_t)a * 9 + b] += ta[a] * ta[b];
      }
      const float* row = S.data() + (std::size_t)i * M;
      double* x = X.data() + (std::size_t)a * M;
      for (int j = 0; j < M; ++j) { x[j] += ta[a] * (double)row[j]; }
    }
  }
  if (!chol_solve_(A, 9, X.data(), M)) {
    _basis = CurveBasis{};
    return no("the curve table is degenerate");
  }
  double rr = 0.0, ss = 0.0;
  for (int i = 0; i < n; ++i) {
    const float* row = S.data() + (std::size_t)i * M;
    for (int j = 0; j < M; ++j) {
      double fit = X[(std::size_t)8 * M + j];
      for (int k = 0; k < 8; ++k) {
        fit += _adaln_t_table[(std::size_t)knots[(std::size_t)i] * 8 + k] *
               X[(std::size_t)k * M + j];
      }
      rr += (row[j] - fit) * (row[j] - fit);
      ss += (double)row[j] * row[j];
    }
  }
  const double residual = ss > 0.0 ? std::sqrt(rr / ss) : 1.0;
  if (residual > 1e-3) {
    _basis = CurveBasis{};
    return no(fmt("the timestep MLP of '{}' does not trace this file's "
                  "curve (relative residual {:.3g} over {} knots)",
                  _cfg.time_mlp_dir, residual, n)());
  }
  CurveBasis b;
  b.mlp = M;
  b.residual = residual;
  b.u.resize((std::size_t)M * 8);
  b.m.resize((std::size_t)M);
  for (int j = 0; j < M; ++j) {
    for (int k = 0; k < 8; ++k) {
      b.u[(std::size_t)j * 8 + k] = (float)X[(std::size_t)k * M + j];
    }
    b.m[(std::size_t)j] = (float)X[(std::size_t)8 * M + j];
  }
  // pinv = (U^T U)^-1 U^T, [8, M].
  std::vector<double> G8(64, 0.0), Ut((std::size_t)8 * M);
  for (int j = 0; j < M; ++j) {
    for (int a = 0; a < 8; ++a) {
      Ut[(std::size_t)a * M + j] = X[(std::size_t)a * M + j];
      for (int c = 0; c < 8; ++c) {
        G8[(std::size_t)a * 8 + c] +=
            X[(std::size_t)a * M + j] * X[(std::size_t)c * M + j];
      }
    }
  }
  if (!chol_solve_(G8, 8, Ut.data(), M)) {
    _basis = CurveBasis{};
    return no("the curve's basis is degenerate");
  }
  b.pinv.assign(Ut.begin(), Ut.end());
  _basis = std::move(b);
  if (_mc->session() != nullptr) {
    _mc->session()->info(fmt(
        "MetalMiniMaxH3Transformer: the pruned curve's basis fit from the "
        "timestep MLP of '{}' ({} -> {} -> {}): relative residual {:.2g} "
        "over {} knots", _cfg.time_mlp_dir, _cfg.freq_dim, _cfg.time_hidden,
        M, residual, n));
  }
  return true;
}

// A PRUNED DiT under an adapter that moves its timestep embedding (one
// on the MLP, or a flow map's blend): the curve carries the embedding's
// projection c = pinv (s - m), s = silu(e), and the rest, (I - P)(s - m),
// is what the file's AdaLN cannot see -- MEASURED for HyperFlow's 8-step
// grid, 5-36% of the embedding and 6-49% of block 0's modulation. So the
// RELEASED AdaLN (Config::time_mlp_dir's blocks) projects that remainder
// once per schedule row, W (I - P)(s - m), into tables the projection
// adds (h3_adaln_pruned_res_f16): exact where the file's AdaLN is the
// released one pruned (W8 = W U, b8 = b + W m), and the released AdaLN's
// view of the remainder where the file merged another. One block's
// weight in memory at a time.
bool
MetalMiniMaxH3Transformer::bake_pruned_off_curve_(
    const std::vector<float>& all, const std::vector<float>* all_r,
    std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "minimax-h3 bake_adaln (pruned): " + m; }
    _adaln_row0.clear();
    _adaln_nt.clear();
    _adaln_tab.clear();
    _final_adaln_tab = SharedBuffer{};
    return false;
  };
  const Config& c = _cfg;
  const int T = (int)all.size(), M = _basis.mlp;
  const int D = c.adaln_out(), FD = 2 * c.hidden;
  std::vector<float> e;
  if (!mlp_embedding_(all, all_r, &e)) {
    return fail("the released embedding could not be computed");
  }
  // The remainder rows, [T, M] bf16.
  SharedBuffer R = _mc->make_shared_buffer((std::size_t)T * M * 2);
  if (R.empty()) { return fail("allocation"); }
  auto* rp = static_cast<std::uint16_t*>(R.contents());
  double off = 0.0, tot = 0.0;
  std::vector<double> d((std::size_t)M);
  for (int t = 0; t < T; ++t) {
    const float* er = e.data() + (std::size_t)t * M;
    for (int j = 0; j < M; ++j) {
      d[(std::size_t)j] =
          (double)silu_bf16_(er[j]) - (double)_basis.m[(std::size_t)j];
    }
    double cc[8];
    for (int k = 0; k < 8; ++k) {
      const float* pk = _basis.pinv.data() + (std::size_t)k * M;
      double acc = 0.0;
      for (int j = 0; j < M; ++j) { acc += (double)pk[j] * d[(std::size_t)j]; }
      cc[k] = acc;
    }
    for (int j = 0; j < M; ++j) {
      double r = d[(std::size_t)j];
      for (int k = 0; k < 8; ++k) {
        r -= (double)_basis.u[(std::size_t)j * 8 + k] * cc[k];
      }
      off += r * r;
      tot += d[(std::size_t)j] * d[(std::size_t)j];
      rp[(std::size_t)t * M + j] = f32_to_bf16_((float)r);
    }
  }
  std::shared_ptr<WeightSet> ws = WeightSet::open(_cfg.time_mlp_dir, nullptr);
  if (!ws) { return fail("cannot open '" + _cfg.time_mlp_dir + "'"); }
  // One projection's weight, as bf16, [n, M] -- unquantized only: this
  // reads it whole, and a packed one has no single format to read it in.
  auto weight = [&](const std::string& nm, int n) -> SharedBuffer {
    const auto* ti = ws->src().info(nm);
    if (ti == nullptr || ti->shape.size() != 2 || ti->shape[0] != n ||
        ti->shape[1] != M ||
        (ti->dtype != "BF16" && ti->dtype != "F16" && ti->dtype != "F32")) {
      return {};
    }
    return ti->dtype == "BF16"
               ? ws->read(nm, _mc, WeightSet::Residency::Copied)
               : to_bf16_(ws->src(), _mc, nm);
  };
  _adaln_tab.clear();
  _adaln_tab.resize((std::size_t)c.n_layers);
  CommandStream cs = _mc->make_command_stream();
  for (int L = -1; L < c.n_layers; ++L) {
    const std::string nm = L < 0 ? std::string("final_layer.adaln_proj."
                                               "linear.weight")
                                 : blk_("blocks.", L,
                                        "adaln_proj.linear.weight");
    const int n = L < 0 ? FD : D;
    Linear lw;
    lw.w = weight(nm, n);
    if (lw.w.empty()) {
      return fail(fmt("'{}' has no unquantized {} [{}, {}] (BF16 / F16 / "
                      "F32) to project the off-curve rows through",
                      _cfg.time_mlp_dir, nm, n, M)());
    }
    SharedBuffer tab = _mc->make_shared_buffer((std::size_t)T * n * 2);
    if (tab.empty()) { return fail("table allocation"); }
    {
      ComputeEncoder enc = cs.begin_compute();
      gemm_(enc, R, 0, lw, tab, 0, T, n, M);
    }
    _flops.flush(cs);
    if (!cs.commit().wait_ok(nullptr)) { return fail("the table GEMM"); }
    (L < 0 ? _final_adaln_tab : _adaln_tab[(std::size_t)L]) = std::move(tab);
  }
  if (_mc->session() != nullptr) {
    _mc->session()->info(fmt(
        "MetalMiniMaxH3Transformer: pruned AdaLN under an adapted timestep "
        "embedding -- {} rows, {:.1f}% of the silu'd embedding off the "
        "curve, projected through the released AdaLN of '{}' ({} MB of "
        "tables)", T, 100.0 * std::sqrt(off / std::max(tot, 1e-30)),
        _cfg.time_mlp_dir, adaln_table_bytes() >> 20));
  }
  return true;
}

bool
MetalMiniMaxH3Transformer::pruned_projects_() const
{
  if (!_cfg.pruned() || _basis.empty()) { return false; }
  if (two_time_active()) { return true; }
  for (const LoraSlot& sl : _lora) {
    if (sl.scale != 0.0f && (!sl.time_in.empty() || !sl.time_out.empty())) {
      return true;
    }
  }
  return false;
}

// A pruned adapter's AdaLN deltas (LoraSlot::pdelta): ComfyUI's `.diff`
// / `.diff_b` patches as the file stores them, then any full-model AdaLN
// module ported through the curve's basis -- both at strength 1, summed
// where a projection has both.
bool
MetalMiniMaxH3Transformer::bind_pruned_adaln_(lora::Adapter* ad,
                                              const std::string& path,
                                              bool diffusers, LoraSlot& slot,
                                              std::string* err)
{
  const Config& c = _cfg;
  const int D = c.adaln_out(), FD = 2 * c.hidden;
  slot.pdelta.clear();
  slot.pdelta.resize((std::size_t)c.n_layers);
  slot.pdelta_final = SharedBuffer{};
  auto delta = [&](int L) -> float* {
    SharedBuffer& b = L < 0 ? slot.pdelta_final : slot.pdelta[(std::size_t)L];
    const std::size_t d = (std::size_t)(L < 0 ? FD : D);
    if (b.empty()) {
      b = _mc->make_shared_buffer(d * 9 * 4);
      if (b.empty()) { return nullptr; }
      std::memset(b.contents(), 0, d * 9 * 4);
    }
    return static_cast<float*>(b.contents());
  };
  int diffs = 0, ported = 0, skipped = 0;
  // ---- ComfyUI weight patches, `<layer>.diff` / `<layer>.diff_b` -----
  if (lora::Adapter::file_touches(path, "adaln_proj.linear.diff")) {
    std::shared_ptr<WeightSet> ws = WeightSet::open(path, nullptr);
    if (!ws) {
      if (err != nullptr) { *err = "minimax-h3 lora: cannot open " + path; }
      return false;
    }
    const std::string kW = ".adaln_proj.linear.diff";
    const std::string kB = ".adaln_proj.linear.diff_b";
    for (const std::string& nm : ws->src().tensor_names()) {
      auto ends = [&](const std::string& x) {
        return nm.size() > x.size() &&
               nm.compare(nm.size() - x.size(), x.size(), x) == 0;
      };
      const bool is_b = ends(kB), is_w = !is_b && ends(kW);
      if (!is_w && !is_b) { continue; }
      std::string k = nm.substr(0, nm.size() - (is_w ? kW : kB).size());
      for (const char* pre : {"model.diffusion_model.", "diffusion_model.",
                              "transformer."}) {
        const std::string ps(pre);
        if (k.compare(0, ps.size(), ps) == 0) { k = k.substr(ps.size()); }
      }
      int L = -2;
      if (k == "final_layer") {
        L = -1;
      } else if (k.compare(0, 7, "blocks.") == 0 &&
                 k.size() > 7 &&
                 k.find_first_not_of("0123456789", 7) == std::string::npos) {
        L = std::atoi(k.c_str() + 7);
        if (L >= c.n_layers) { L = -2; }
      }
      const auto* ti = ws->src().info(nm);
      const int d = L == -1 ? FD : D;
      const bool shape_ok =
          ti != nullptr &&
          (is_w ? (ti->shape.size() == 2 && ti->shape[0] == d &&
                   ti->shape[1] == 8)
                : (ti->shape.size() == 1 && ti->shape[0] == d));
      std::vector<float> v;
      if (L == -2 || !shape_ok || !to_host_f32_(*ws, _mc, nm, v)) {
        ++skipped;
        continue;
      }
      float* dst = delta(L);
      if (dst == nullptr) { return false; }
      dst += is_w ? 0 : (std::size_t)d * 8;
      for (std::size_t i = 0; i < v.size(); ++i) { dst[i] += v[i]; }
      ++diffs;
    }
  }
  // ---- a full-model adapter's AdaLN modules, ported ------------------
  // dW = B (A U), db = B (A m): the delta the module adds on the released
  // curve, B A silu(temb(t)), carried by the curve's coordinates.
  if (ad != nullptr) {
    std::string berr;
    bool basis = false, asked = false;
    for (int L = -1; L < c.n_layers; ++L) {
      const std::string mod =
          L < 0 ? (diffusers ? "norm_out.linear"
                             : "final_layer.adaln_proj.linear")
                : diffusers ? "transformer_blocks." + std::to_string(L) +
                                  ".adaln_proj.linear"
                            : blk_("blocks.", L, "adaln_proj.linear");
      if (!ad->has(mod)) { continue; }
      if (!asked) {
        asked = true;
        basis = ensure_curve_basis_(&berr);
      }
      lora::HostFactors hf;
      const int d = L < 0 ? FD : D;
      if (!basis || !ad->bind_host(mod, d, _basis.mlp, &hf) || hf.empty()) {
        ++skipped;
        continue;
      }
      const int r = hf.rank, M = _basis.mlp;
      std::vector<double> au((std::size_t)r * 8, 0.0), am((std::size_t)r, 0.0);
      for (int q = 0; q < r; ++q) {
        const float* a = hf.a.data() + (std::size_t)q * M;
        for (int j = 0; j < M; ++j) {
          const float* u = _basis.u.data() + (std::size_t)j * 8;
          for (int k = 0; k < 8; ++k) {
            au[(std::size_t)q * 8 + k] += a[j] * u[k];
          }
          am[(std::size_t)q] += (double)a[j] * _basis.m[(std::size_t)j];
        }
      }
      float* dst = delta(L);
      if (dst == nullptr) { return false; }
      for (int j = 0; j < d; ++j) {
        const float* bj = hf.b.data() + (std::size_t)j * r;
        double w[8] = {0}, bb = 0.0;
        for (int q = 0; q < r; ++q) {
          for (int k = 0; k < 8; ++k) {
            w[k] += bj[q] * au[(std::size_t)q * 8 + k];
          }
          bb += bj[q] * am[(std::size_t)q];
        }
        for (int k = 0; k < 8; ++k) {
          dst[(std::size_t)j * 8 + k] += (float)w[k];
        }
        dst[(std::size_t)d * 8 + j] += (float)bb;
      }
      ++ported;
    }
    if (asked && !basis && _mc->session() != nullptr) {
      _mc->session()->warn(fmt(
          "MetalMiniMaxH3Transformer: '{}' adapts the AdaLN of a full-model "
          "DiT, and this DiT is ComfyUI's pruned form; its AdaLN modules are "
          "skipped: {}", path, berr));
    }
  }
  slot.modules += diffs + ported;
  if ((diffs > 0 || ported > 0 || skipped > 0) && _mc->session() != nullptr) {
    _mc->session()->info(fmt(
        "MetalMiniMaxH3Transformer: '{}' on the pruned AdaLN: {} ComfyUI "
        "weight diff tensor(s), {} full-model module(s) ported onto the "
        "curve{}{}", path, diffs, ported,
        ported > 0 ? fmt(" (basis residual {:.2g})", _basis.residual)()
                   : std::string(),
        skipped > 0 ? fmt(", {} skipped", skipped)() : std::string()));
  }
  return true;
}

// Read the adapter and bind its factors to the modules they name.
//
// The module names are the MODEL's own -- blocks.N.attn.qkv_proj,
// mlp.fc1/fc2, adaln_proj.linear, token_refiner.blocks.N.* -- so this is
// a lookup rather than a remap. That holds for kohya / musubi-tuner
// files too (`lora_unet_blocks_0_attn_qkv_proj.lora_down.weight`):
// musubi trains against these same module names and only flattens
// them, and lora::Adapter reads the flattening. A module the
// checkpoint does not carry is SKIPPED and counted, not an error: an
// adapter trained on a subset of the projections is normal, and
// refusing one would be refusing the common case.
bool
MetalMiniMaxH3Transformer::bind_lora_(const LoraSpec& spec, LoraSlot& slot,
                                      std::string* err)
{
  // Reading the FILE is shared with every other family that takes an
  // adapter (shared/runtime-lora.h): the publisher spellings, the
  // alpha/rank rescale, the bf16 conversion, the shape check and the
  // skip counting. What stays here is the MODULE LIST, which is the
  // model -- a family that shared that would bind an adapter to the
  // wrong projections and produce plausible wrong output.
  slot.scale = spec.scale;
  // The sampler a manifest beside the file states (SamplingContract),
  // whichever spelling its factors take. Read first, so a malformed one
  // refuses the adapter with its path before anything is bound.
  {
    std::string cerr;
    if (sampling_contract_of(spec.path, &slot.contract, &cerr)) {
      slot.has_contract = true;
    } else if (!cerr.empty()) {
      if (err != nullptr) { *err = "minimax-h3 lora: " + cerr; }
      return false;
    }
  }
  // AdaLN curve branches first: a file of them alone (Open-H3-
  // Regenerate-2K's) is a slot with no LoRA factors at all.
  bool curved = false;
  if (!bind_curves_(spec.path, slot, &curved, err)) { return false; }
  // ComfyUI WEIGHT DIFFS on the AdaLN (`adaln_proj.linear.diff` /
  // `.diff_b`, added to the weight and the bias) are how the pruned
  // form's Turbo ports carry that half of the distillation, [D, 8] --
  // applied on a pruned DiT (bind_pruned_adaln_). On a released one they
  // fit nothing, and the rest of the adapter would run without them and
  // look like a weaker adapter: said, not skipped quietly.
  if (!_cfg.pruned() &&
      lora::Adapter::file_touches(spec.path, "adaln_proj.linear.diff") &&
      _mc->session() != nullptr) {
    _mc->session()->warn(fmt(
        "MetalMiniMaxH3Transformer: '{}' carries ComfyUI weight diffs on "
        "the AdaLN (adaln_proj.linear.diff / .diff_b) -- the pruned form's "
        "-- and this DiT is the released one, which does not apply them; "
        "its LoRA factors bind without them", spec.path));
  }
  if (curved && !lora::Adapter::file_touches(spec.path, ".lora_")) {
    // Curves alone: a slot with no factors, kept so the next adapter's
    // strength still lands on ITS slot (a slot is its index).
    slot.blocks.resize((std::size_t)_cfg.n_layers);
    slot.refiner.resize((std::size_t)_cfg.n_refiner);
    slot.modules = slot.curve.branches;
    return true;
  }
  std::string oerr;
  auto ad = lora::Adapter::open(spec.path, _mc, &oerr);
  if (!ad) {
    if (err != nullptr) { *err = "minimax-h3 lora: " + oerr; }
    return false;
  }

  // Every fused `attn.qkv_proj` the FILE supplied, so the check at the
  // end of this function can ask what kind of fusion they are.
  std::vector<LoraFactors*> fused_qkv;
  auto bind = [&](const std::string& mod, int N, int K, LoraFactors& out) {
    ad->bind(mod, N, K, &out);
  };

  const Config& c = _cfg;
  const int H = c.hidden, I = c.inner(), F = c.ffn;

  // ---- the DIFFUSERS decomposition ---------------------------------
  //
  // lightx2v publishes each Turbo adapter twice. The ComfyUI copy names
  // this model's own modules and binds below; the diffusers copy is the
  // ORIGINAL peft export and is a different DECOMPOSITION, not a
  // renaming:
  //
  //   transformer_blocks.N.            -> blocks.N.
  //   token_refiner.refiner_blocks.N.  -> token_refiner.blocks.N.
  //   attn.to_q + to_k + to_v          -> attn.qkv_proj  (FUSED)
  //   attn.to_out.0                    -> attn.out_proj
  //   ff.net.0.proj                    -> mlp.fc1  (B halves SWAPPED)
  //   ff.net.2                         -> mlp.fc2
  //
  // and its factors are spelled `.lora_A.default.weight` with ONE alpha
  // in `__metadata__` instead of a tensor per module.
  //
  // Every line of that was VERIFIED tensor-for-tensor against
  // upstream's own ComfyUI conversion of the same adapter, on a main
  // block and on a refiner block: A concatenates on the rank axis
  // exactly, B is exactly block-diagonal, fc1's halves are exactly
  // swapped, out_proj and fc2 are byte-identical, and alpha 8 over rank
  // 128 is the same strength as their fused 24 over 384. It is not
  // inferred from the metadata's description of itself.
  //
  // IT IS THE BETTER FILE TO HOLD, which is what makes this worth
  // doing rather than pointing users at the ComfyUI copy. A PUBLISHED
  // fusion is built for one qkv column grouping -- flat, in every case
  // that exists -- so it is silently wrong on the other publisher's
  // weights. Fusing it HERE builds the grouping this model actually
  // has, so the diffusers copy runs on both releases where the ComfyUI
  // copy runs on one. It is also the smaller download -- 1.38 GB
  // against 1.96, the difference being the two thirds of a
  // block-diagonal B that is zero -- and, bound BANDED below, the
  // smaller model too: the zeros are never built.
  // VDN-H3 wraps each DiT block's attention in a HybridAttention that
  // keeps the original module as `.orig`, so its adapters name
  // `attn.orig.to_q` where a plain diffusers export names `attn.to_q`.
  // The TOKEN REFINER is not wrapped -- only the DiT blocks are -- so
  // its targets keep the plain spelling and the prefix is per call
  // rather than per adapter. Same tensors either way: the branch shares
  // the backbone's projections, which is exactly why one adapter feeds
  // both halves of the hybrid.
  const bool vdn_hybrid = ad->has("transformer_blocks.0.attn.orig.to_q");
  if (vdn_hybrid || ad->has("transformer_blocks.0.attn.to_q")) {
    const int HD = c.head_dim;
    const bool per_head = c.qkv_per_head;
    // Where part `pt` (0 = q, 1 = k, 2 = v) row `r` lands in the fused
    // [3 * inner] output. FLAT is [all q | all k | all v]; PER-HEAD
    // interleaves the three within each head, which is what every
    // MiniMaxAI release does and what no published fusion targets.
    const lora::Adapter::RowMap qkv_rows =
        [per_head, HD, I](int pt, int r) {
          return qkv_fused_row(pt, r, I, HD, per_head);
        };
    // Diffusers' SwiGLU is [value; gate] where fc1 here is read
    // [gate; value] -- see the fc1 half-order note in the forward.
    const lora::Adapter::RowMap swap_halves =
        [F](int, int r) { return r < F ? r + F : r - F; };
    // BANDED where it can be (see lora::Factors::banded): the three
    // adapters' B rows stored side by side with no zeros between them,
    // and each output tile contracting over its own part's rank. The
    // group is how many consecutive fused rows one of q/k/v owns -- a
    // head's worth per head, the whole inner width flat -- and every
    // second-factor kernel tiles at a width that divides both. The
    // block-diagonal bind stays as the fallback: the same arithmetic
    // with (parts-1)/parts of the second GEMM spent on zeros.
    const bool band = !_lora_band_off && _fn_gemm_acc_band.valid();
    const int qkv_group = per_head ? HD : I;
    int banded = 0;
    // out_proj and fc2 are byte-identical between the two copies, so
    // they take the plain bind and not a scatter that would copy every
    // row to its own index.
    auto bind_block = [&](const std::string& dp, const std::string& ap,
                          BlockLora& bl) {
      const std::vector<std::string> qkv = {dp + ap + "to_q",
                                            dp + ap + "to_k",
                                            dp + ap + "to_v"};
      if (band && ad->bind_banded(qkv, 3 * I, H, qkv_group, &bl.qkv,
                                  qkv_rows)) {
        ++banded;
      } else {
        ad->bind_fused(qkv, 3 * I, H, &bl.qkv, qkv_rows);
      }
      ad->bind(dp + ap + "to_out.0", H, I, &bl.out);
      ad->bind_fused({dp + "ff.net.0.proj"}, 2 * F, H, &bl.fc1, swap_halves);
      ad->bind(dp + "ff.net.2", H, F, &bl.fc2);
    };
    const std::string block_ap = vdn_hybrid ? "attn.orig." : "attn.";
    slot.blocks.resize((std::size_t)c.n_layers);
    for (int i = 0; i < c.n_layers; ++i) {
      bind_block("transformer_blocks." + std::to_string(i) + ".", block_ap,
                 slot.blocks[(std::size_t)i]);
    }
    slot.refiner.resize((std::size_t)c.n_refiner);
    for (int i = 0; i < c.n_refiner; ++i) {
      bind_block("token_refiner.refiner_blocks." + std::to_string(i) + ".",
                 "attn.", slot.refiner[(std::size_t)i]);
    }
    // The AdaLN projections, which the plain diffusers and ComfyUI
    // exports do NOT adapt -- bind() simply finds nothing there, which
    // is why this costs those adapters nothing. VDN's turbo adapter is
    // the one that does: `transformer_blocks.N.adaln_proj.linear`
    // [96768, 2688] and `norm_out.linear` [10752, 2688], both at rank
    // 16 where everything else is 64. Left unbound they are the LARGEST
    // half of an 8-step distillation silently absent, and the model
    // still renders.
    //
    // SCALE: none of the three VDN adapters carries a `__metadata__`
    // alpha or a per-module `.alpha` tensor -- their alphas live only in
    // adapter_spec.json, which nothing here reads -- so every module
    // takes the mul = 1.0 fallback. That is EXACTLY right for these
    // files and by arithmetic rather than by design: alpha equals rank
    // throughout (64/64 for the two `default` adapters, and 64/64 plus
    // 16/16 for turbo's adaln group). An adapter of this shape whose
    // alpha differed from its rank would be applied at the wrong
    // strength with nothing to say so.
    for (int i = 0; i < c.n_layers; ++i) {
      ad->bind("transformer_blocks." + std::to_string(i)
                   + ".adaln_proj.linear",
               c.adaln_out(), c.time_dim,
               &slot.blocks[(std::size_t)i].adaln);
    }
    ad->bind("norm_out.linear", 2 * H, c.time_dim, &slot.final_adaln);
    // The timestep MLP (diffusers' `time_embedder.linear_1/2`) and, for
    // a flow-map adapter, its endpoint twin -- on the host, in f32.
    if (!bind_flow_map_(*ad, spec.path, slot, err)) { return false; }
    slot.modules  = ad->modules() + slot.curve.branches;
    slot.banded   = banded;
    slot.max_rank = ad->max_rank();
    if (c.pruned() &&
        !bind_pruned_adaln_(ad.get(), spec.path, /*diffusers=*/true, slot,
                            err)) {
      return false;
    }
    if (slot.modules == 0) {
      if (err != nullptr) {
        *err = "minimax-h3 lora: '" + spec.path +
               "' looks like the diffusers decomposition but none of its "
               "modules fit this DiT";
      }
      return false;
    }
    if (_mc->session() != nullptr) {
      _mc->session()->info(fmt(
          "MetalMiniMaxH3Transformer: {} (diffusers decomposition, qkv "
          "fused HERE for this DiT's {} grouping, {} of {} banded; "
          "matrix-core band tiles {})",
          ad->summary(spec.path, slot.scale),
          per_head ? "PER-HEAD" : "flat", banded,
          c.n_layers + c.n_refiner,
          (_fn_lora_b128_band.valid() && _fn_lora_fused128_band.valid())
              ? "ready" : "absent"));
    }
    return true;
  }

  slot.blocks.resize((std::size_t)c.n_layers);
  for (int i = 0; i < c.n_layers; ++i) {
    BlockLora& bl = slot.blocks[(std::size_t)i];
    const std::string p = blk_("blocks.", i, "");
    bind(p + "attn.qkv_proj", 3 * I, H, bl.qkv);
    if (!bl.qkv.empty()) { fused_qkv.push_back(&bl.qkv); }
    bind(p + "attn.out_proj", H, I, bl.out);
    bind(p + "mlp.fc1", 2 * F, H, bl.fc1);
    bind(p + "mlp.fc2", H, F, bl.fc2);
    bind(p + "adaln_proj.linear", c.adaln_out(), c.time_dim, bl.adaln);
  }
  slot.refiner.resize((std::size_t)c.n_refiner);
  for (int i = 0; i < c.n_refiner; ++i) {
    BlockLora& bl = slot.refiner[(std::size_t)i];
    const std::string p = blk_("token_refiner.blocks.", i, "");
    bind(p + "attn.qkv_proj", 3 * I, H, bl.qkv);
    if (!bl.qkv.empty()) { fused_qkv.push_back(&bl.qkv); }
    bind(p + "attn.out_proj", H, I, bl.out);
    bind(p + "mlp.fc1", 2 * F, H, bl.fc1);
    bind(p + "mlp.fc2", H, F, bl.fc2);
  }
  bind("final_layer.adaln_proj.linear", 2 * H, c.time_dim,
       slot.final_adaln);
  // Through the same door as the diffusers branch, so a flow-map adapter
  // in THIS spelling is bound whole or refused -- never bound as a plain
  // LoRA with its endpoint embedder silently left behind.
  if (!bind_flow_map_(*ad, spec.path, slot, err)) { return false; }

  slot.modules  = ad->modules() + slot.curve.branches;
  slot.max_rank = ad->max_rank();
  // PRUNED: the AdaLN's `.diff` patches and full-model modules, as f32
  // deltas on the trunk's projections (counted into `modules`).
  if (c.pruned() &&
      !bind_pruned_adaln_(ad.get(), spec.path, /*diffusers=*/false, slot,
                          err)) {
    return false;
  }
  if (slot.modules == 0) {
    // NAME what would have matched. "none of this model's modules" is
    // the whole of what a user sees when a trainer's naming is one this
    // reader does not know, and it reads as a broken file -- so the
    // conventions it DOES know go in the message, with one module in
    // each spelling, which is what lets the next unknown one be
    // recognised as a naming question rather than a bad download.
    if (err != nullptr) {
      *err = "minimax-h3 lora: '" + spec.path +
             "' adapts none of this model's modules. Module names are "
             "read as this DiT's own (`blocks.0.attn.qkv_proj`, bare or "
             "under `diffusion_model.` / `transformer.`), as kohya / "
             "musubi-tuner's (`lora_unet_blocks_0_attn_qkv_proj`), or as "
             "the diffusers decomposition "
             "(`transformer_blocks.0.attn.to_q`); factors as lora_A / "
             "lora_B or lora_down / lora_up";
    }
    return false;
  }
  if (_mc->session() != nullptr) {
    _mc->session()->info(fmt(
        "MetalMiniMaxH3Transformer: {}",
        ad->summary(spec.path, slot.scale)));
  }
  // A FUSED qkv adapter carries a ROW ORDER, and it is adapted to this
  // DiT's rather than reported as a mismatch.
  //
  // `attn.qkv_proj` writes one [rows, 3*inner] block whose columns
  // Comfy-Org's repack groups FLAT ([all q | all k | all v]) and every
  // MiniMaxAI release groups PER HEAD ([q0 k0 v0 | q1 k1 v1 | ...]).
  // This tree re-orders neither publisher's weights -- it points the
  // attention at the right offsets -- so the delta has to be in the
  // base's order, and one built for the other publisher lands on the
  // wrong channels in all 50 blocks.
  //
  // THE MAP IS PROVEN, not derived: Comfy-Org's flat row r and
  // MiniMaxAI's per-head row qkv_fused_row(r/I, r%I, ..., true) are the
  // SAME WEIGHTS. MEASURED on the two published FL2VA checkpoints,
  // blocks.0.attn.qkv_proj: 156 sampled rows byte-identical under that
  // map, 3 of 156 under the identity (its fixed points).
  //
  // Only the fused projection is affected. out_proj, mlp.fc1/fc2 and
  // adaln_proj carry no grouping and were always applied correctly,
  // which is why a mismatch reads as "Turbo, but not quite" instead of
  // as a broken model -- and why it survived so long unnoticed.
  if (!fused_qkv.empty() && c.qkv_per_head &&
      spec.qkv_layout != LoraSpec::QkvLayout::kPerHead) {
    const int HD = c.head_dim;
    const lora::Adapter::RowMap to_per_head = [HD, I](int, int r) {
      return qkv_fused_row(r / I, r % I, I, HD, true);
    };
    int fixed = 0;
    for (LoraFactors* f : fused_qkv) {
      if (lora::Adapter::permute_b_rows(f, 3 * I, to_per_head)) { ++fixed; }
    }
    // MEASURED says the layout was read off the factors; ASSUMED says
    // it was taken on the evidence that every published adapter is
    // flat. Both permute; the log says which, because only one of them
    // is something the bytes could confirm.
    const bool measured =
        lora::Adapter::block_diagonal_b(*fused_qkv[0], 3, 3 * I);
    if (_mc->session() != nullptr) {
      _mc->session()->info(fmt(
          "MetalMiniMaxH3Transformer: '{}' adapts the FUSED attn.qkv_proj "
          "in the flat [q|k|v] order ({}) and this DiT groups qkv PER "
          "HEAD -- {} of {} blocks re-ordered to match. Set "
          "`lora_qkv_layout: per_head` if this adapter was trained on "
          "MiniMaxAI's weights", spec.path,
          measured ? "block-diagonal B, so read off the factors"
                   : "assumed: every published H3 adapter is",
          fixed, (int)fused_qkv.size()));
    }
  }
  return true;
}

bool
MetalMiniMaxH3Transformer::sampling_contract_of(const std::string& file,
                                                SamplingContract* out,
                                                std::string* err)
{
  namespace fs = std::filesystem;
  if (err != nullptr) { err->clear(); }
  std::error_code ec;
  const fs::path mp = fs::path(file).parent_path() / "manifest.json";
  if (!fs::is_regular_file(mp, ec)) { return false; }
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "'" + mp.string() + "': " + m; }
    return false;
  };
  std::ifstream f(mp);
  std::stringstream ss;
  ss << f.rdbuf();
  const FlexData doc = FlexData::from_json(ss.str());
  if (!doc.is_object()) { return false; }   // someone else's manifest
  const auto o = doc.as_object();   // a view: `doc` owns it
  if (!o.contains("sampling")) { return false; }
  const FlexData smp = o.at("sampling");
  if (!smp.is_object()) { return fail("`sampling` is not an object"); }
  const auto so = smp.as_object();
  auto str = [](const FlexData& d) {
    return d.is_string() ? std::string(d.as_string("")) : std::string();
  };
  SamplingContract c;
  if (o.contains("release_id")) { c.release = str(o.at("release_id")); }
  if (o.contains("task")) { c.task = str(o.at("task")); }
  if (o.contains("trunk")) { c.trunk = str(o.at("trunk")); }
  if (so.contains("knots") && so.at("knots").is_array()) {
    const FlexData kn = so.at("knots");
    const auto ka = kn.as_array();
    for (std::size_t i = 0; i < ka.size(); ++i) {
      c.sigmas.push_back(ka[i].as_real(-1.0) / 1000.0);
    }
  }
  // diffusers' own check on a grid it is handed: strictly decreasing,
  // at most 1, ending at exactly 0.
  bool ok = c.sigmas.size() >= 2 && c.sigmas.front() <= 1.0 &&
            c.sigmas.back() == 0.0;
  for (std::size_t i = 1; ok && i < c.sigmas.size(); ++i) {
    ok = c.sigmas[i] < c.sigmas[i - 1];
  }
  if (!ok) {
    return fail("`sampling.knots` is not a grid strictly decreasing from "
                "at most 1000 to exactly 0");
  }
  auto num = [&](const char* k) {
    return so.contains(k) ? so.at(k).as_real(0.0) : 0.0;
  };
  c.video_shift = num("video_shift");
  c.audio_shift = num("audio_shift");
  const std::string method =
      so.contains("method") ? str(so.at("method")) : std::string();
  if (method == "renoise") {
    c.renoise = true;
  } else if (!method.empty() && method != "euler") {
    // A sampler this loop does not implement is refused BY NAME: run as
    // Euler it would be exactly the generic sampling the release warns
    // is not equivalent.
    return fail("`sampling.method` '" + method + "' is not one this "
                "sampler implements (renoise, euler)");
  }
  const std::string np =
      o.contains("numerics_policy") ? str(o.at("numerics_policy"))
                                    : std::string();
  if (np == "bf16_conditioning_v1") {
    c.bf16_conditioning = true;
  } else if (!np.empty() && np != "fp32_conditioning_v1") {
    return fail("`numerics_policy` '" + np + "' is not one this model "
                "implements (bf16_conditioning_v1, fp32_conditioning_v1)");
  }
  if (!c.trunk.empty() && contract_partition(c).empty()) {
    return fail("`trunk` '" + c.trunk + "' names no MiniMax-H3 DiT "
                "(transformer, transformer_ref)");
  }
  if (out != nullptr) { *out = std::move(c); }
  return true;
}

std::string
MetalMiniMaxH3Transformer::contract_partition(const SamplingContract& c)
{
  // The diffusers pipeline's two DiT folders: `transformer/` is FL2VA's,
  // `transformer_ref/` Ref2VA's (minimax-h3-diffusers-layout.h).
  if (c.trunk == "transformer") { return "fl2va"; }
  if (c.trunk == "transformer_ref") { return "ref2va"; }
  return {};
}

const MetalMiniMaxH3Transformer::SamplingContract*
MetalMiniMaxH3Transformer::sampling_contract() const
{
  if (_contract.slot < 0 || (std::size_t)_contract.slot >= _lora.size() ||
      _lora[(std::size_t)_contract.slot].scale == 0.0f) {
    return nullptr;
  }
  return &_contract;
}

// HyperFlow's file layout, which is the only flow-map adapter there is:
//
//   header   hyperflow = "true", hyperflow_gate, hyperflow_sigmas (JSON
//            list, raw grid), hyperflow_video_shift / _audio_shift,
//            hyperflow_version, lora_alpha / lora_rank
//   tensors  transformer.endpoint_time_embedder.linear_{1,2}.lora_{A,B}
//            beside the ordinary per-block factors, f32
//
// The endpoint embedder has NO base weights in the file: upstream builds
// it as a deep copy of the checkpoint's own `time_embedder` taken BEFORE
// any adapter is injected, then gives it its own adapter. So here it is
// the base proj_in/proj_out plus end_in/end_out -- and NOT plus the base
// embedder's own adapter (time_in/time_out), which upstream's copy never
// saw.
bool
MetalMiniMaxH3Transformer::bind_flow_map_(lora::Adapter& ad,
                                          const std::string& path,
                                          LoraSlot& slot, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "minimax-h3 lora: '" + path + "': " + m; }
    return false;
  };
  const Config& c = _cfg;
  // PRUNED (Config::curve_grid) the file has no timestep MLP. One that
  // an adapter adapts -- or a flow-map adapter's two -- runs on the
  // RELEASED MLP (Config::time_mlp_dir) and is projected onto the
  // curve's coordinates, the part it moves OFF the curve baked from the
  // released AdaLN (bake_adaln). Without that MLP its modules are
  // skipped and a flow map refused.
  if (c.pruned()) {
    const bool flow = ad.metadata("hyperflow") == "true" ||
                      ad.has("endpoint_time_embedder.linear_1") ||
                      ad.has("endpoint_time_embedder.proj_in");
    const bool mlp_mods = ad.has("time_embedder.linear_1") ||
                          ad.has("time_embedder.linear_2") ||
                          ad.has("time_embedder.proj_in") ||
                          ad.has("time_embedder.proj_out");
    if (!flow && !mlp_mods) { return true; }
    std::string berr;
    if (!ensure_curve_basis_(&berr)) {
      if (flow) {
        return fail("a flow-map adapter blends two timestep MLPs, and this "
                    "DiT is ComfyUI's pruned form, which has none: " + berr);
      }
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "MetalMiniMaxH3Transformer: '{}' adapts the timestep MLP, and "
            "this DiT is ComfyUI's pruned form; those modules are skipped: "
            "{}", path, berr));
      }
      return true;
    }
  }
  // The timestep MLP, whichever adapter this is: diffusers' names
  // (`linear_1/2`) or this checkpoint's own (`proj_in/out`). It runs on
  // the host in f32 (see time_embed_), so these factors stay f32 on the
  // host too -- a flow-map adapter ships them in f32 deliberately, and
  // they would lose exactly what that keeps through the bf16 GPU path.
  auto mlp = [&](const std::string& pre, lora::HostFactors* in,
                 lora::HostFactors* out) {
    const bool diffusers = ad.has(pre + "linear_1") ||
                           ad.has(pre + "linear_2");
    ad.bind_host(pre + (diffusers ? "linear_1" : "proj_in"), c.time_hidden,
                 c.freq_dim, in);
    ad.bind_host(pre + (diffusers ? "linear_2" : "proj_out"), mlp_dim_(),
                 c.time_hidden, out);
  };
  mlp("time_embedder.", &slot.time_in, &slot.time_out);
  const bool header = ad.metadata("hyperflow") == "true";
  const bool tensors = ad.has("endpoint_time_embedder.linear_1") ||
                       ad.has("endpoint_time_embedder.proj_in");
  if (!header && !tensors) { return true; }   // an ordinary adapter
  if (!header || !tensors) {
    return fail(header
        ? "the header says HyperFlow but the file has no "
          "endpoint_time_embedder -- half a flow-map adapter"
        : "the file has an endpoint_time_embedder but no HyperFlow "
          "header, so its gate and grid are unknown");
  }
  mlp("endpoint_time_embedder.", &slot.end_in, &slot.end_out);
  // BOTH embedders fully adapted, or refuse. A flow-map file that bound
  // only some of them is not a weaker adapter but a different function
  // of (t, r) -- the gate blends two embeddings the model was trained
  // to see TOGETHER.
  if (slot.end_in.empty() || slot.end_out.empty() ||
      slot.time_in.empty() || slot.time_out.empty()) {
    return fail(fmt("the timestep embedders did not bind (base {}/{}, "
                    "endpoint {}/{}) -- shapes do not fit this DiT's "
                    "{} -> {} -> {} MLP",
                    !slot.time_in.empty(), !slot.time_out.empty(),
                    !slot.end_in.empty(), !slot.end_out.empty(),
                    c.freq_dim, c.time_hidden, mlp_dim_())());
  }
  FlowMap f;
  try {
    f.gate = std::stof(ad.metadata("hyperflow_gate"));
  } catch (...) {
    return fail("no readable `hyperflow_gate` in the header");
  }
  const std::string sv = ad.metadata("hyperflow_sigmas");
  if (!sv.empty()) {
    const FlexData arr = FlexData::from_json(sv);
    if (arr.is_array()) {
      const auto av = arr.as_array();   // a view: `arr` owns it
      for (std::size_t i = 0; i < av.size(); ++i) {
        f.sigmas.push_back((float)av[i].as_real(-1.0));
      }
    }
    // The contract diffusers' scheduler enforces on a grid it is handed:
    // strictly decreasing, at most 1, ending at EXACTLY 0.
    bool ok = f.sigmas.size() >= 2 && f.sigmas.front() <= 1.0f &&
              f.sigmas.back() == 0.0f;
    for (std::size_t i = 1; ok && i < f.sigmas.size(); ++i) {
      ok = f.sigmas[i] < f.sigmas[i - 1];
    }
    if (!ok) { return fail("`hyperflow_sigmas` is not a valid grid: " + sv); }
  }
  auto num = [&](const char* k) {
    try {
      const std::string v = ad.metadata(k);
      return v.empty() ? 0.0 : std::stod(v);
    } catch (...) {
      return 0.0;
    }
  };
  f.video_shift = num("hyperflow_video_shift");
  f.audio_shift = num("hyperflow_audio_shift");
  f.version     = ad.metadata("hyperflow_version");
  slot.flow     = std::move(f);
  slot.two_time = true;
  if (_mc->session() != nullptr) {
    _mc->session()->info(fmt(
        "MetalMiniMaxH3Transformer: '{}' is a FLOW-MAP adapter (HyperFlow "
        "{}): two-time timestep embedding, gate {:.4g}, {}-step grid, "
        "trained at shifts {:.4g}/{:.4g}", path, slot.flow.version,
        slot.flow.gate,
        slot.flow.sigmas.empty() ? 0 : (int)slot.flow.sigmas.size() - 1,
        slot.flow.video_shift, slot.flow.audio_shift));
  }
  return true;
}

int
MetalMiniMaxH3Transformer::flow_map_steps(const std::string& adapter_file)
{
  // The safetensors header is `u64 length` + that many bytes of JSON,
  // with the file's own facts under `__metadata__` as strings.
  std::ifstream f(adapter_file, std::ios::binary);
  if (!f) { return 0; }
  std::uint64_t n = 0;
  f.read(reinterpret_cast<char*>(&n), 8);
  if (!f || n == 0 || n > (std::uint64_t)64 << 20) { return 0; }
  std::string js((std::size_t)n, '\0');
  f.read(js.data(), (std::streamsize)n);
  if (!f) { return 0; }
  const FlexData hdr = FlexData::from_json(js);
  if (!hdr.is_object()) { return 0; }
  const auto o = hdr.as_object();
  if (!o.contains("__metadata__")) { return 0; }
  const FlexData md = o.at("__metadata__");
  if (!md.is_object()) { return 0; }
  const auto mo = md.as_object();
  if (!mo.contains("hyperflow") ||
      std::string(mo.at("hyperflow").as_string("")) != "true" ||
      !mo.contains("hyperflow_sigmas")) {
    return 0;
  }
  const FlexData g = FlexData::from_json(
      std::string(mo.at("hyperflow_sigmas").as_string("")));
  if (!g.is_array()) { return 0; }
  const std::size_t pts = g.as_array().size();
  return pts >= 2 ? (int)pts - 1 : 0;
}

bool
MetalMiniMaxH3Transformer::two_time_active() const
{
  return _flow.slot >= 0 && (std::size_t)_flow.slot < _lora.size() &&
         _lora[(std::size_t)_flow.slot].scale != 0.0f;
}

MetalMiniMaxH3Transformer::~MetalMiniMaxH3Transformer()
{
  // GIVE THE POOL BACK. Freeing a wired buffer unwires it in the kernel,
  // so the machine recovers either way -- but the pool's own counter
  // would not, and a DiT that is destroyed after every clip (the
  // ordinary `unload_when_idle: destroy` path) would leak its whole
  // share of the budget per clip until nothing could wire at all.
  if (_wire.on()) {
    wire_fixed_(false);
    for (std::size_t L = 0; L < _blocks.size(); ++L) {
      wire_block_(_blocks[L], false);
      wire_lora_((int)L, false);
    }
    _slot[0] = Block{};
    _slot[1] = Block{};
  }
}

std::unique_ptr<MetalMiniMaxH3Transformer>
MetalMiniMaxH3Transformer::load(const std::string& dit_dir, MetalCompute* mc,
                                const Config& cfg, bool stream_blocks,
                                const std::vector<LoraSpec>& loras)
{
  return load(WeightSet::open(resolve_dit_dir(dit_dir, cfg.partition),
                             nullptr),
              mc, cfg, stream_blocks, loras);
}

std::unique_ptr<MetalMiniMaxH3Transformer>
MetalMiniMaxH3Transformer::load(std::shared_ptr<WeightSet> ws_in,
                                MetalCompute* mc, const Config& cfg,
                                bool stream_blocks,
                                const std::vector<LoraSpec>& loras)
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
  m->_fn_gemm_acc_band =
      m->_lib_gemm.function("dense_gemm_t_bm64_acc_band_f16");
  m->_lora_band_off = std::getenv("VPIPE_H3_LORA_NO_BAND") != nullptr;
  m->_fn_rms       = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_rms_heads = m->_lib_rope.function("rms_norm_heads_strided_f16");
  m->_fn_trope = m->_lib_rope.function("transpose_rope_half_part_ftab_f16");
  // The transpose-free twin. Optional: a build without it simply keeps
  // the transposing path, which is correct and only slower.
  m->_fn_rope_ip = m->_lib_rope.function("rope_half_part_ftab_inplace_f16");
  // The banded forward's (band_rows); without it the sequence runs whole.
  m->_fn_trope_band =
      m->_lib_rope.function("transpose_rope_half_part_ftab_band_f16");
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
  m->_fn_copy_rect = m->_lib_elt.function("copy_rect_f16");
  m->_fn_nan_trip  = m->_lib_elt.function("nan_tripwire_f16");
  {
    m->_lib_sdpa = mc->load_library("sdpa_bf16");
    m->_fn_sdpa = m->_lib_sdpa.function("sdpa_full_f16");
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
  // The checkpoint's LAYOUT and its FP8 / int8 weights, surveyed before
  // anything loads: linear_ asks the survey what it is reading. An
  // encoding this build cannot decode (block scales, NVFP4, int8 with no
  // record) and a diffusers-named file with a tensor the translation
  // cannot place are refused HERE, by name -- either would otherwise
  // surface as a missing tensor several lines on, which names the wrong
  // cause.
  {
    const SessionContextIntf* sess = mc->session();
    const MetalLlamaWeights& src = ws.src();
    if (const std::string u = minimax_h3::untranslatable_tensor(src);
        !u.empty()) {
      if (sess != nullptr) {
        sess->warn(fmt("minimax-h3: '{}' is an H3 DiT in the diffusers "
                       "naming, but tensor '{}' has no H3 name, so the "
                       "checkpoint cannot be read whole", ws.dir(), u));
      }
      return nullptr;
    }
    // ComfyUI's int8 too (a community checkpoint saved by ComfyUI): every
    // widening here carries its rotation (Linear::f8_rot), and the ANE's
    // staging, which cannot, leaves those projections to the GPU.
    std::string ferr;
    if (!fp8::scan(src, &m->_f8, &ferr, /*int8=*/true)) {
      if (sess != nullptr) {
        sess->warn(fmt("minimax-h3: '{}': {}", ws.dir(), ferr));
      }
      return nullptr;
    }
    if (m->_f8.any()) {
      m->_f8_widen = widen_of_(src, m->_f8, m->_cfg);
      // A native W4A8 weight's per-GEMM decode (gemm_); optional, as the
      // ConvRot quantizer is: without it W4A8 widens per GEMM instead.
      m->_lib_dq = mc->load_library("affine_dequant_bf16");
      m->_fn_w4a8_i8 = m->_lib_dq.function("w4a8_to_i8");
      if (!m->_fp8x.load(mc)) {
        if (sess != nullptr) {
          sess->warn(fmt("minimax-h3: '{}' stores FP8 weights and the FP8 "
                         "expansion kernels did not load", ws.dir()));
        }
        return nullptr;
      }
      std::size_t fp8_bytes = 0, n_scaled = 0, n_i8 = 0, n_rot = 0;
      std::size_t n_w4 = 0;
      int rot = 0;
      std::string first_i8, first_w4;
      for (const auto& [n, fw] : m->_f8.weights) {
        if (const auto* ti = src.info(n); ti != nullptr) {
          fp8_bytes += (std::size_t)ti->nbytes;
        }
        if (!fw.scale.empty()) { ++n_scaled; }
        if (fw.format == fp8::Format::kI8) {
          ++n_i8;
          if (first_i8.empty()) { first_i8 = n; }
        }
        if (fw.format == fp8::Format::kW4A8) {
          ++n_w4;
          if (first_w4.empty()) { first_w4 = n; }
          for (const std::string& a : {fw.s_rel, fw.codebook}) {
            if (const auto* ai = src.info(a); ai != nullptr) {
              fp8_bytes += (std::size_t)ai->nbytes;
            }
          }
        }
        if (fw.rot_group > 0) {
          ++n_rot;
          rot = std::max(rot, fw.rot_group);
        }
      }
      // The int8 widening kernels are optional host kernels: a host
      // older than them runs FP8 and refuses int8 here, by name.
      if (n_i8 > 0 && !m->_fp8x.int8_valid()) {
        if (sess != nullptr) {
          sess->warn(fmt("minimax-h3: '{}' stores int8 weights ('{}' among "
                         "{}) and this host has no int8 widening kernels",
                         ws.dir(), first_i8, n_i8));
        }
        return nullptr;
      }
      if (n_w4 > 0 && !m->_fp8x.w4a8_valid()) {
        if (sess != nullptr) {
          sess->warn(fmt("minimax-h3: '{}' stores W4A8 weights ('{}' among "
                         "{}) and this host has no W4A8 widening kernel",
                         ws.dir(), first_w4, n_w4));
        }
        return nullptr;
      }
      if (sess != nullptr) {
        sess->info(fmt(
            "MetalMiniMaxH3Transformer: {} checkpoint{} -- {} tensors "
            "({} MB){} kept as {} codes and widened per GEMM{}",
            n_w4 == m->_f8.weights.size() ? "W4A8"
            : n_i8 == m->_f8.weights.size() ? "int8"
            : n_i8 + n_w4 > 0             ? "FP8 / int8"
                                          : "FP8",
            src.has("blocks.0.attn.qkv_proj.q.weight")
                ? std::string(" (diffusers layout, read under H3's names)")
                : std::string(),
            m->_f8.weights.size(), fp8_bytes >> 20,
            n_scaled > 0 ? fmt(", {} of them scaled", n_scaled)()
                         : std::string(),
            n_w4 > 0 ? "4-bit" : n_i8 > 0 ? "8-bit" : "FP8",
            n_rot > 0 ? fmt(" ({} ConvRot-rotated, groups up to {}, "
                            "un-rotated as they widen)", n_rot, rot)()
                      : std::string()));
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
    //
    // BOTH spellings, and EVERY slot. bind_lora_ reaches fc1 from the
    // diffusers decomposition too (`ff.net.0.proj`, halves swapped), and
    // a file asked only about `.mlp.fc1.` would leave the fusion on and
    // that adapter's fc1 delta silently dropped -- visible only on a
    // machine without matrix cores, since `_use_mma2` turns the fusion
    // off anyway. One adapter needing the split path settles it for all
    // of them: the weight layout is the block's, not the adapter's.
    for (const LoraSpec& sp : loras) {
      if (!m->_fuse_ff || sp.path.empty()) { continue; }
      if (lora::Adapter::file_touches(sp.path, ".mlp.fc1.lora_") ||
          lora::Adapter::file_touches(sp.path, ".ff.net.0.proj.lora_")) {
        m->_fuse_ff = false;
        if (mc->session() != nullptr) {
          mc->session()->log_debug(fmt(
              "MetalMiniMaxH3Transformer: fused SwiGLU off -- a LoRA adapts "
              "mlp.fc1, whose delta has to land before the activation"));
        }
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
  // THE TUNE CEILING IS NOT A MATRIX-CORE FACT, and it used to be set as
  // if it were -- inside the block below, so a box without matrix units
  // kept the member's default while tune_qmm_(), which runs on EVERY
  // box, read it. Two consequences, and the second is the one that
  // matters: VPIPE_H3_TUNE_ROWS silently did nothing there, and the
  // ceiling differed between an M4 and an M5 without anybody asking it
  // to -- so the same geometry shared a tuned route on one box and
  // re-measured on the other. Hoisted; the split's bucket below stays
  // inside, because THAT is matrix-core only.
  m->_tune_ceiling = tune_ceiling_();
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
    m->_fn_lora_b128_band =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128_acc_band_f16");
    m->_fn_lora_b256_band =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_acc_band_f16");
    m->_fn_lora_fused128_band =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128_lora_band_f16");
    m->_fn_lora_fused256_band =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_lora_band_f16");
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
    i8->set_tier(cfg.i8_tier >= 2   ? I8GemmContext::Tier::kToken
                 : cfg.i8_tier == 1 ? I8GemmContext::Tier::kFast
                                    : I8GemmContext::Tier::kAccurate);
    i8->set_token_rotation(cfg.i8_tier == 2);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
    // The int8 LAYOUT for the main blocks (see Config::i8_tier), decided
    // here because load_block_ reads it. Not under the ANE tier, whose
    // stagers read the checkpoint's weights as they are.
    m->_i8_layout = m->_i8 != nullptr &&
                    m->_i8->tier() != I8GemmContext::Tier::kAccurate &&
                    !cfg.ane_ffn && m->_i8->fast_group(cfg.hidden) > 0;
    if (m->_i8_layout && mc->session() != nullptr) {
      const int fg = m->_i8->fast_group(cfg.hidden);
      // ComfyUI ConvRot-256 codes the token tier runs as stored (the
      // test i8_linear_ makes per projection).
      std::size_t n_cr = 0;
      if (fg == I8GemmContext::kTokenGroup && m->_i8->convrot_available() &&
          !(std::getenv("VPIPE_H3_CONVROT_NATIVE") != nullptr &&
            std::atoi(std::getenv("VPIPE_H3_CONVROT_NATIVE")) == 0)) {
        for (const auto& [n, fw] : m->_f8.weights) {
          n_cr += n.compare(0, 7, "blocks.") == 0 && fw.rot_group == 256 &&
                  (fw.format == fp8::Format::kI8 ||
                   fw.format == fp8::Format::kW4A8);
        }
      }
      mc->session()->info(fmt(
          "MetalMiniMaxH3Transformer: the main blocks' projections take the "
          "{} int8 tier ({}){}",
          fg == I8GemmContext::kTokenGroup ? "TOKEN" : "FAST",
          fg != I8GemmContext::kTokenGroup
              ? fmt("rotated, group {}", fg)()
          : m->_i8->token_rotated()
              ? "per channel x per token, Hadamard-256 rotated; " +
                    m->_i8->token_kernel() + " GEMM"
          : "per channel x per token; a Draw Things i8x checkpoint's "
            "codes as stored; " + m->_i8->token_kernel() + " GEMM",
          n_cr > 0 ? fmt("; {} ComfyUI ConvRot projection(s) run their "
                         "codes as stored", n_cr)()
                   : std::string()));
    }
    // Deep-K split for the FF down projection. bf16 libraries: the fold
    // writes this DiT's element type.
    m->_splitk.load(mc, m->_lib_dense_mma, m->_lib_elt);
    // The split is tuned at the same ceiling the route is, so two
    // geometries above it share one answer instead of re-measuring it.
    m->_splitk.m_bucket = m->_tune_ceiling;
  }
  // Sol-Attn (opt-in, LOSSY): approximate attention by routing whole KV
  // blocks. Loaded here rather than inside the matrix-core block above
  // because it has nothing to do with matmul2d -- the kernels are plain
  // simdgroup code and run on any box.
  //
  // A FAILED LOAD REFUSES THE MODEL rather than falling back to dense.
  // Dense attention under a config that asked for Sol is not a slower
  // answer, it is a different one at a different memory cost, and a run
  // that silently produced it would be reported as a Sol run.
  if (cfg.sol.enabled) {
    std::string serr;
    m->_sol = MetalSolAttention::load(mc, /*bf16=*/true, &serr);
    if (!m->_sol) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt(
            "MetalMiniMaxH3Transformer: sol_attn requested but {}", serr));
      }
      return nullptr;
    }
    if (mc->session() != nullptr) {
      mc->session()->log_normal(fmt(
          "MetalMiniMaxH3Transformer: Sol-Attn ON -- tau {:.2f}, {} "
          "threshold, layers {}+, local radius {}",
          (double)cfg.sol.tau, sol::threshold_name(cfg.sol.thresh),
          cfg.sol.dense_layers, cfg.sol.local_radius));
    }
  }

  // SAGE, on the same terms as Sol above: a run that asked for it and
  // quietly got dense attention would be reported as a Sage run and its
  // numbers believed. The three-way answer (and the one case that is
  // NOT a failure) lives in load_for_model.
  {
    bool sage_fatal = false;
    m->_sage = MetalSageAttention::load_for_model(
        mc, /*bf16=*/true, cfg.sage, "MetalMiniMaxH3Transformer",
        &sage_fatal);
    if (sage_fatal) { return nullptr; }
    // Its int8 fragment exists only on the matrix-core flash entry.
    // load_for_model already reports a GPU with no matrix cores and returns
    // null; this is the other way to lack the entry -- matrix cores present
    // but the NAX flash library not loaded -- which would otherwise keep the
    // setting and run the dense kernel. Said, and dropped, as generate-image's
    // Qwen-Image does.
    if (m->_sage && !m->_attn_nax) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt(
            "MetalMiniMaxH3Transformer: sage_attn is off -- the matrix-core "
            "flash entry is unavailable here, so attention runs the dense "
            "kernel"));
      }
      m->_sage.reset();
    }
    // The early steps run without the int8 P*V (sage_pv8_from_step), so
    // its buffers are built from the first prepare(): toggling it would
    // otherwise rebuild the scratch mid-denoise, a private free and
    // allocation of ~3 bytes a q/k/v element where nothing is lent.
    if (m->_sage) { m->_sage->reserve_pv8(cfg.sage.pv8); }
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
  if (!cfg.pruned()) {
    m->_final_adaln =
        m->linear_(ws, "final_layer.adaln_proj.linear", true, r);
  }
  m->_video_out   = m->linear_(ws, "final_layer.video_out", true, r);
  m->_audio_out   = m->linear_(ws, "final_layer.audio_out", true, r);
  m->_refiner_final_norm = m->weight_(ws, "token_refiner.final_norm.weight", r);
  if (m->_video_patch.empty() || m->_audio_patch.empty() ||
      m->_cond_proj.empty() || m->_final_norm.empty() ||
      (m->_final_adaln.empty() && !cfg.pruned()) || m->_video_out.empty() ||
      m->_audio_out.empty() || m->_refiner_final_norm.empty()) {
    return nullptr;
  }

  if (cfg.pruned()) {
    // ComfyUI's PRUNED form: the curve table to the host (time_embed_
    // reads points off it per forward) and every AdaLN projection held
    // AS STORED in the trunk -- 1.5 MB a block, read once, never
    // streamed; the kernel widens it exactly, as ComfyUI casts it to
    // f32 (h3_adaln_pruned_f16).
    if (!to_host_f32_(ws, mc, "adaln_t_table", m->_adaln_t_table) ||
        m->_adaln_t_table.size() !=
            (std::size_t)cfg.curve_grid * (std::size_t)cfg.time_dim) {
      if (mc->session() != nullptr) {
        mc->session()->error(
            fmt("MetalMiniMaxH3Transformer: the pruned DiT's adaln_t_table "
                "cannot be read"));
      }
      return nullptr;
    }
    const auto res = kept_residency_(m->_stream_blocks, m->_wire.on());
    auto held = [&](const std::string& nm, int D, PrunedAdaln& p) {
      const auto* wi = ws.src().info(nm + ".weight");
      const auto* bi = ws.src().info(nm + ".bias");
      if (wi == nullptr || bi == nullptr || wi->dtype != bi->dtype ||
          wi->shape.size() != 2 || wi->shape[0] != D ||
          wi->shape[1] != cfg.time_dim || bi->shape.size() != 1 ||
          bi->shape[0] != D) {
        return false;
      }
      p.dtype = wi->dtype == "F32"   ? 0
                : wi->dtype == "F16" ? 1
                : wi->dtype == "BF16" ? 2
                                      : -1;
      if (p.dtype < 0) { return false; }
      p.w = ws.tensor(nm + ".weight", mc, res);
      p.b = ws.tensor(nm + ".bias", mc, res);
      return !p.w.empty() && !p.b.empty();
    };
    m->_padaln.resize((std::size_t)cfg.n_layers);
    bool ok = held("final_layer.adaln_proj.linear", 2 * cfg.hidden,
                   m->_final_padaln);
    for (int i = 0; ok && i < cfg.n_layers; ++i) {
      ok = held(blk_("blocks.", i, "adaln_proj.linear"), cfg.adaln_out(),
                m->_padaln[(std::size_t)i]);
    }
    if (ok && !m->_lib_regen.valid()) {
      m->_lib_regen = mc->load_library("minimax_h3_regenerate_bf16");
    }
    m->_fn_pruned_mod = m->_lib_regen.function("h3_adaln_pruned_f16");
    m->_fn_pruned_patch = m->_lib_regen.function("h3_adaln_patch_f32");
    m->_fn_pruned_res = m->_lib_regen.function("h3_adaln_pruned_res_f16");
    if (!ok || !m->_fn_pruned_mod.valid() || !m->_fn_pruned_patch.valid() ||
        !m->_fn_pruned_res.valid()) {
      if (mc->session() != nullptr) {
        mc->session()->error(fmt(
            "MetalMiniMaxH3Transformer: the pruned DiT's AdaLN projections "
            "{}", ok ? "cannot run: h3_adaln_pruned_f16 / _patch_f32 are "
                       "missing from this build"
                     : "are not [D, 8] F32 / F16 / BF16 pairs"));
      }
      return nullptr;
    }
  } else if (
      // The timestep MLP goes to the HOST in f32 -- see time_embed_.
      !to_host_f32_(ws, mc, "time_embedder.proj_in.weight", m->_time_in_w) ||
      !to_host_f32_(ws, mc, "time_embedder.proj_in.bias", m->_time_in_b) ||
      !to_host_f32_(ws, mc, "time_embedder.proj_out.weight", m->_time_out_w) ||
      !to_host_f32_(ws, mc, "time_embedder.proj_out.bias", m->_time_out_b)) {
    return nullptr;
  }
  // rope.inv_freq is LOADED rather than recomputed from a theta. The
  // native config carries no rope_theta at all, so recomputing it would
  // mean guessing the base -- and a wrong base is a wrong model that
  // still produces smooth video.
  //
  // The diffusers layout has no such tensor and states the base instead
  // (`rope_theta`, config_from_json), so there it is computed -- in f32,
  // as 1 / theta^(2i / (2 * rope_freq_dim)), which reproduces the native
  // table BIT FOR BIT (checked against the Comfy-Org file; a double
  // evaluation rounds 4 of the 16 entries differently). A Draw Things
  // `.ckpt` carries no table either and runs on the architecture's base,
  // the diffusers config's 10000.
  const bool dt_ckpt =
      lower_(std::filesystem::path(ws.dir()).extension().string()) ==
      ".ckpt";
  if (ws.src().has("rope.inv_freq")) {
    if (!to_host_f32_(ws, mc, "rope.inv_freq", m->_inv_freq) ||
        (int)m->_inv_freq.size() != cfg.rope_freq_dim) {
      return nullptr;
    }
  } else if ((ws.src().has("blocks.0.attn.qkv_proj.q.weight") || dt_ckpt) &&
             cfg.rope_theta > 0.0f && cfg.rope_freq_dim > 0) {
    m->_inv_freq.resize((std::size_t)cfg.rope_freq_dim);
    const float dim = (float)(2 * cfg.rope_freq_dim);
    for (int i = 0; i < cfg.rope_freq_dim; ++i) {
      m->_inv_freq[(std::size_t)i] =
          1.0f / std::pow(cfg.rope_theta, (float)(2 * i) / dim);
    }
  } else {
    return nullptr;
  }

  m->_refiner.resize((std::size_t)cfg.n_refiner);
  for (int i = 0; i < cfg.n_refiner; ++i) {
    if (!m->load_block_(ws, blk_("token_refiner.blocks.", i, ""),
                        m->_refiner[(std::size_t)i], /*main=*/false,
                        /*with_adaln=*/false, r)) {
      return nullptr;
    }
  }
  // WHETHER THIS MODEL'S PAGES GO IN THE WIRED POOL.
  //
  // Restored here after the prefix retirement deleted it by accident:
  // the assignment lived inside the block that sized the pinned prefix,
  // that block went, and the five places that READ these two fields
  // stayed. The effect was silent and total -- the pool flag was
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
  m->_wire.open(mc);
  // The 50 main blocks: preloaded (default), or -- streaming -- only the
  // pinned prefix, with forward() reading and freeing the tail per block.
  if (!stream_blocks) {
    m->_blocks.resize((std::size_t)cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
      if (!m->load_block_(ws, blk_("blocks.", i, ""),
                          m->_blocks[(std::size_t)i], /*main=*/true,
                          /*with_adaln=*/true, r)) {
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
  // Slots bind in the order given and keep it: `lora_scale(0)` is the
  // first adapter the caller named, whatever it turned out to be.
  for (const LoraSpec& sp : loras) {
    if (sp.path.empty()) { continue; }
    if ((int)m->_lora.size() >= kMaxLoraSlots) {
      if (mc->session() != nullptr) {
        mc->session()->warn(fmt(
            "MetalMiniMaxH3Transformer: {} adapters given, {} slots -- "
            "'{}' is IGNORED", loras.size(), kMaxLoraSlots, sp.path));
      }
      break;
    }
    LoraSlot slot;
    std::string lerr;
    if (!m->bind_lora_(sp, slot, &lerr)) {
      if (mc->session() != nullptr) { mc->session()->error(fmt("{}", lerr)); }
      return nullptr;
    }
    // Each slot owns its own rows of the [seq, rank] scratch, so two
    // adapters' first GEMMs do not have to be ordered against each
    // other's second.
    slot.rank_prefix = m->_lora_rank_total;
    m->_lora_rank_total += slot.max_rank;
    // A flow-map adapter REPLACES the timestep embedding rather than
    // adding to it, so two of them have no meaning together -- the
    // second would blend an endpoint embedder trained against a
    // different one. Refused rather than resolved by order.
    if (slot.two_time) {
      if (m->_flow.slot >= 0) {
        if (mc->session() != nullptr) {
          mc->session()->error(fmt(
              "MetalMiniMaxH3Transformer: '{}' is a second flow-map "
              "(two-time) adapter; slot {} already carries one, and two "
              "cannot share the timestep embedding", sp.path,
              m->_flow.slot));
        }
        return nullptr;
      }
      m->_flow = slot.flow;
      m->_flow.slot = (int)m->_lora.size();
    }
    // A sampling contract names THE sampler, so two of them -- or one
    // beside a flow-map adapter, which names its own grid -- cannot both
    // hold. And the adapter was trained on one of the two DiTs: a FL2VA
    // adapter on the Ref2VA weights is a different model that runs.
    if (slot.has_contract) {
      const SamplingContract& sc = slot.contract;
      std::string why;
      if (m->_contract.slot >= 0) {
        why = fmt("slot {} already states a sampling contract",
                  m->_contract.slot)();
      } else if (m->_flow.slot >= 0) {
        why = fmt("slot {} is a flow-map adapter, which names its own "
                  "grid", m->_flow.slot)();
      } else {
        const std::string want = contract_partition(sc);
        if (!want.empty() && !cfg.partition.empty() &&
            want != cfg.partition) {
          why = fmt("it was trained on the {} DiT (`{}`) and this is the "
                    "{} one", want, sc.trunk, cfg.partition)();
        }
      }
      if (!why.empty()) {
        if (mc->session() != nullptr) {
          mc->session()->error(fmt(
              "MetalMiniMaxH3Transformer: '{}' ({}) is refused: {}",
              sp.path, sc.release, why));
        }
        return nullptr;
      }
      m->_contract = sc;
      m->_contract.slot = (int)m->_lora.size();
      if (mc->session() != nullptr) {
        mc->session()->info(fmt(
            "MetalMiniMaxH3Transformer: '{}' states a sampling contract "
            "({}): {} steps on its own grid, {}, shifts {:.4g}/{:.4g}{}",
            sp.path, sc.release.empty() ? "unnamed" : sc.release,
            (int)sc.sigmas.size() - 1,
            sc.renoise ? "fresh noise each step" : "Euler",
            sc.video_shift, sc.audio_shift,
            sc.bf16_conditioning ? ", keyframes rounded to bf16" : ""));
      }
    }
    m->_lora.push_back(std::move(slot));
  }
  // PRUNED: the scratch a patched projection is built in (pruned_mod_),
  // the widest of a block's and the final layer's [D * 8 + D] in f32.
  if (cfg.pruned()) {
    bool deltas = false;
    for (const LoraSlot& sl : m->_lora) {
      deltas = deltas || !sl.pdelta_final.empty();
      for (const SharedBuffer& d : sl.pdelta) { deltas = deltas || !d.empty(); }
    }
    if (deltas) {
      const std::size_t D =
          (std::size_t)std::max(cfg.adaln_out(), 2 * cfg.hidden);
      m->_padaln_patch = mc->make_shared_buffer(D * 9 * 4);
      if (m->_padaln_patch.empty()) { return nullptr; }
    }
  }
  // A streamed DiT's adapter factors stream with their blocks.
  if (stream_blocks) { (void)m->spill_lora_(); }
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
  // Accelerated mode, NATIVE: the int8 activation against the checkpoint's
  // own codes, before any dequant exists. A shape it declines falls
  // through to the dequant + requant form below, nothing encoded.
  if (_i8 && l.quantized && _quant_group == 64 &&
      _i8->gemm_affine(enc, x, x_off, l.codes, l.scales, l.qbias, l.bits, y,
                       y_off, M, N, K)) {
    return true;
  }
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
    const bool wide = route == GemmRoute::kMma128x256;
    const metal_compute::ComputeFunction* ff =
        wide ? &_fn_lora_fused256 : &_fn_lora_fused128;
    // A banded pair needs the banded tile, and the BASE's tile width is
    // fixed by the route -- so where it would straddle a part's group
    // (256 wide over q/k/v grouped per 128-wide head) there is no fold
    // and the second GEMM runs on its own, as for any other decline.
    if (lora->banded()) {
      ff = wide ? &_fn_lora_fused256_band : &_fn_lora_fused128_band;
      if (lora->group % (wide ? 256 : 128) != 0) { ff = nullptr; }
    }
    if (ff != nullptr && ff->valid()) {
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
      enc.set_constant(10, lora->b_rank());
      if (lora->banded()) {
        enc.set_constant(11, lora->rank);
        enc.set_constant(12, lora->group);
        enc.set_constant(13, lora->parts);
      }
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
  // An FP8 weight is timed the way the forward runs it: widened once,
  // then the tile. Measuring the codes as a dense operand would read
  // an empty `w`.
  Linear fv;
  const Linear* wl = fp8_dense_(enc, l, fv, N, K);
  if (wl == nullptr) { return; }
  const int band = mma_row_band(N, K);
  for (int m0 = 0; m0 < M; m0 += band) {
    gemm_route_dispatch_(enc, x, x_off + (std::size_t)m0 * K, *wl, y,
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
                                 int M, int N, int K, const LoraStack* lora)
{
  const bool bias = !l.b.empty();
  // FP8: widened ONCE for the whole projection, ahead of every band,
  // into the dequant scratch -- `wl` is then an ordinary dense weight to
  // every route, i8 and LoRA fold below. The bias stays `l`'s.
  Linear fv;
  const Linear* wl = l.gc > 0 ? &l : fp8_dense_(enc, l, fv, N, K);
  if (wl == nullptr) { return; }
  // The int8 layout's operands: its own, or -- a w4 pack held as stored
  // -- the context's scratch, converted ONCE for all of the bands (and
  // for the dense stand-in below, which widens the same values).
  const SharedBuffer* fc = &wl->codes;
  const SharedBuffer* fcs = &wl->cscales;
  // A native W4A8 weight's int8 codes, decoded ONCE for all the bands.
  if (wl->gc > 0 && wl->cr_w4) {
    const std::size_t n = (std::size_t)N * (std::size_t)K;
    if (_cr_codes.empty() || _cr_codes.byte_size() < n) {
      _cr_codes = _mc->make_shared_buffer(n);
      if (_cr_codes.empty()) { _fp8_failed = true; return; }
    }
    const bool cb = !wl->f8_cb.empty();
    enc.set_function(_fn_w4a8_i8);
    enc.set_buffer(0, wl->codes);
    enc.set_buffer(1, _cr_codes);
    enc.set_constant(2, (std::uint32_t)n);
    enc.set_buffer(3, wl->f8_srel);
    enc.set_buffer(4, cb ? wl->f8_cb : wl->cscales);
    enc.set_constant(5, (std::uint32_t)wl->f8_qgroup);
    enc.set_constant(6, (std::uint32_t)(cb ? 1 : 0));
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    fc = &_cr_codes;
  }
  if (wl->gc > 0 && wl->pc) {
    if (!_i8 || !_i8->encode_to_fast_scratch(enc, wl->codes, wl->scales,
                                             wl->qbias, wl->gc, N, K)) {
      return;
    }
    fc = &_i8->fast_scratch_codes();
    fcs = &_i8->fast_scratch_scales();
  }
  _flops.gemm(M, N, K);
  // The int8 layout's dense stand-in, widened only if a band needs it.
  Linear iv;
  const Linear* idense = nullptr;
  // Whether an adapter can run at all was settled in lora_stack_, so an
  // empty stack here simply means none is attached.
  const int nl = lora != nullptr ? lora->n : 0;
  // One route for the WHOLE projection, chosen at the full M so a band
  // never lands on a kernel the tuner did not measure, then the rows in
  // bands narrow enough for the mma tiles' 32-bit addressing. Below the
  // 2^31-byte line -- everything this model ran before 1344x768 -- the
  // band IS M and this is the single pass it always was. A banded
  // forward's band takes the whole sequence's route (_route_rows): the
  // tuner measured at the sequence, and a band looked up at its own rows
  // met the untuned fallback -- another kernel, and other bits.
  GemmRoute route = gemm_route_(_route_rows > 0 ? _route_rows : M, N, K);
  if (!route_ok_(route, M, N, K)) { route = gemm_route_(M, N, K); }
  const int band = mma_row_band(N, K);
  for (int m0 = 0; m0 < M; m0 += band) {
    const int rows = std::min(band, M - m0);
    const std::size_t xo = x_off + (std::size_t)m0 * K;
    const std::size_t yo = y_off + (std::size_t)m0 * N;
    // t = [scale *] x A^T for EVERY adapter, BEFORE the projection: the
    // fused tile reads one of them, and the unfused path does not care
    // which order the two halves are in. Each writes into its own slot
    // region, so these do not serialise on the scratch.
    std::size_t lo[kMaxLoraSlots]   = {};
    bool        scaled[kMaxLoraSlots] = {};
    for (int i = 0; i < nl; ++i) {
      const LoraStack::Use& u = lora->s[i];
      lo[i] = u.base + (std::size_t)m0 * u.f->rank;
      scaled[i] =
          lora_gemm_a_(enc, x, xo, *u.f, rows, K, u.scale, lo[i]);
    }
    // The base tile can absorb ONE adapter's second GEMM. It carries no
    // scale of its own, so only a slot whose strength already reached t
    // is eligible -- offering it another would run that adapter at 1.0.
    // The rest get their own dispatch, which is what a second adapter
    // costs and the only place it costs anything.
    int fold = -1;
    for (int i = 0; i < nl && fold < 0; ++i) {
      if (scaled[i]) { fold = i; }
    }
    bool folded = false;
    // THE INT8 LAYOUT (Linear::gc): I8GemmContext, its scratch, nothing
    // to fold. A band it declines -- the int8 route's row floor -- runs
    // dense from the layout widened into the dequant scratch.
    // ConvRot codes (cr): the activation rotated alike, then the same
    // token GEMM over the codes as stored.
    if (wl->gc > 0 && _i8 &&
        (wl->cr > 0
             ? _i8->fast_accepts(rows, N, K) &&
                   _i8->quantize_token_convrot(enc, x, xo, rows, K) &&
                   _i8->gemm_fast_quantized(enc, *fc, *fcs, wl->gc,
                                            SharedBuffer{}, y, yo, rows, N,
                                            K)
             : _i8->gemm_fast(enc, x, xo, *fc, *fcs, wl->gc, SharedBuffer{},
                              y, yo, rows, N, K))) {
      if (wl->cr > 0) { ++_cr_gemms; }
    } else {
      const Linear* dl = wl;
      if (wl->gc > 0) {
        if (idense == nullptr) {
          idense = i8_dense_(enc, *wl, *fc, *fcs, iv, N, K);
        }
        if (idense == nullptr) { return; }
        dl = idense;
      }
      gemm_route_dispatch_(enc, x, xo, *dl, y, yo, rows, N, K, route,
                           fold >= 0 ? lora->s[fold].f : nullptr,
                           &folded, fold >= 0 ? lo[fold] : 0);
    }
    for (int i = 0; i < nl; ++i) {
      if (folded && i == fold) { continue; }
      const LoraStack::Use& u = lora->s[i];
      lora_gemm_b_(enc, *u.f, y, yo, rows, N, scaled[i], u.scale, lo[i]);
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

// THE ATTENTION QUERY TILE, measured ON THE REAL WORKLOAD.
//
// The NAX attention holds a BQ-row query tile in registers and streams the
// whole K/V once per tile, so intensity is ~BQ flops per byte at any length.
// BQ = WM * 16 with TQ == 1, so BQ 128 is the same kernel with 8 simdgroups
// instead of 4: half the K/V re-read, half as many threadgroups in flight.
// Which way that trades is a property of the MACHINE and the SHAPE -- MEASURED
// bit-identical either way, an M5 Pro gains 1.13x at 19k rows and 1.22x at
// 100k, a 10-core M5 Air loses 5% at 9.4k and gains 8-13% at 19k-38k, and a
// 48q/12kv GQA shape gains nothing on either.
//
// So it is MEASURED HERE, on this model's own blocks, the way the ANE tier
// measures its split: the streamed path already commits and waits per block,
// so a block's wall time is there for the taking. Warm two blocks on the
// incumbent, alternate whole blocks three-and-three, keep the median winner.
// No synthetic dispatch, no truncated stand-in shape, and the samples are the
// real sequence at the real head count.
//
// The RESIDENT path has no per-block boundary -- the whole forward is one
// deferred stream -- so it cannot sample and keeps BQ 64.
void
MetalMiniMaxH3Transformer::tune_attn_tile_(int seq, bool dense)
{
  if (_attn_tile_latched || !_attn_nax) { return; }
  // Nothing to choose: too short for the wide tile to pay, or a forward whose
  // spans / Sol / Sage blocks are built for one tile size.
  if (!dense || seq < kAttnBqMinRows) { return; }
  if (const char* e = std::getenv("VPIPE_H3_ATTN_BQ")) {
    const int v = std::atoi(e);
    if (v == 64 || v == 128) {              // pinned (A/B)
      _attn_bq = v;
      _attn_tile_latched = true;
      return;
    }
  }
  if (std::getenv("VPIPE_H3_NO_ATTN_TILE_AUTOTUNE") != nullptr) {
    _attn_tile_latched = true;              // BQ 64 stands, unmeasured
    return;
  }
  // What an earlier load of this model measured at a length like this
  // one (tuned-answers.h): the probe's six blocks run half of them on the
  // losing tile, so a model reloaded for every clip lost that each time.
  // A HALF-OCTAVE band of the length, not the length: the tiles' ranking
  // moves slowly with it (BQ 128 gained 1.13x at 19k rows and 1.22x at
  // 100k on an M5 Pro), so clips of a similar size share one answer.
  int kept = 0;
  if (find_tuned(tuning_cache(_mc), attn_tile_key_(seq), kept) &&
      (kept == 64 || kept == 128)) {
    _attn_bq = kept;
    _attn_tile_latched = true;
    if (_mc->session() != nullptr) {
      _mc->session()->log_debug(fmt(
          "MetalMiniMaxH3Transformer: attention query tile -> BQ {} (the "
          "session's answer for {} rows)", _attn_bq, seq));
    }
    return;
  }
  _attn_tile_seq = seq;
}

TuneKey
MetalMiniMaxH3Transformer::attn_tile_key_(int seq) const
{
  const int band = seq > 0 ? (int)std::floor(2.0 * std::log2((double)seq))
                           : 0;
  return TuneKey("h3-attn-tile/1")("heads", _cfg.n_heads)
      ("head_dim", _cfg.head_dim)("stream", (int)_stream_blocks)
      ("seq_band", band);
}

// The tile the NEXT main block runs. Advances the probe only when a sample
// can actually come back (a streamed block, whose commit is waited for), so a
// plan and a note stay one to one.
int
MetalMiniMaxH3Transformer::plan_attn_tile_(bool wide_available, bool streaming)
{
  _attn_tile_probing = -1;
  if (!wide_available) { return 64; }
  if (_attn_tile_latched || !streaming) { return _attn_bq; }
  const int n = _attn_tile_blocks++;
  if (n < kTileWarmBlocks) { return 64; }   // clocks and caches settle first
  const int k = n - kTileWarmBlocks;
  if (k < kTileProbeBlocks) {
    const int tile = (k % 2 == 0) ? 64 : 128;
    _attn_tile_probing = (tile == 128) ? 1 : 0;
    return tile;
  }
  return _attn_bq;
}

// One block's wall time, charged to whichever tile it ran. Latches as soon as
// both rings are full: three samples each is what the alternation produces.
void
MetalMiniMaxH3Transformer::note_attn_tile_(double block_ms)
{
  const int a = _attn_tile_probing;
  _attn_tile_probing = -1;
  if (a < 0 || a > 1 || block_ms <= 0.0) { return; }
  _attn_tile_ms[a][_attn_tile_n[a] % kTileRing] = block_ms;
  ++_attn_tile_n[a];
  if (_attn_tile_n[0] < kTileRing || _attn_tile_n[1] < kTileRing) { return; }
  auto median = [&](int i) {
    double t[kTileRing];
    for (int j = 0; j < kTileRing; ++j) { t[j] = _attn_tile_ms[i][j]; }
    for (int x = 0; x < kTileRing; ++x) {
      for (int y = x + 1; y < kTileRing; ++y) {
        if (t[y] < t[x]) { const double s = t[x]; t[x] = t[y]; t[y] = s; }
      }
    }
    return t[kTileRing / 2];
  };
  const double m64 = median(0), m128 = median(1);
  _attn_bq = (m128 > 0.0 && m128 < m64) ? 128 : 64;
  _attn_tile_latched = true;
  if (_attn_tile_seq > 0) {
    store_tuned(tuning_cache(_mc), attn_tile_key_(_attn_tile_seq), _attn_bq);
  }
  if (_mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_normal(fmt(
        "MetalMiniMaxH3Transformer: attention query tile -> BQ {} (per-block "
        "median: BQ 64 {:.1f} ms, BQ 128 {:.1f} ms over {} probe blocks)",
        _attn_bq, m64, m128, kTileProbeBlocks));
  }
}

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
  // READ HERE, not once per process. A static latches whichever value
  // was set when the FIRST model in the process tuned -- so a test
  // binary that pins the tuner for one case and not another gets
  // whichever ran first, and the pinned case then autotunes anyway.
  // This runs once per (model, sequence length); a getenv is nothing
  // beside the sweep it guards.
  if (std::getenv("VPIPE_H3_NO_QMM_AUTOTUNE") != nullptr) { return; }

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

  // THE SESSION'S ANSWERS (tuned-answers.h). Every load of this model --
  // a host unloads it between clips -- measured all of the below again:
  // 2.8 s on an M5 Pro at 832x480, 12-16 s on the plain M5, before the
  // first block reported anything. A route is filed per shape under
  // everything that decides which arms run and what they cost: the
  // projection's shape and weight format, the rows timed, the tiles built
  // (steel twins, matmul2d, tn2), the mma floor, the i8 group the mma arms
  // go through (gemm_mma_ tries int8 first), and the pruning switch.
  KernelTuningCache* const answers = tuning_cache(_mc);
  static const bool kNoPrune =
      std::getenv("VPIPE_H3_TUNE_NO_PRUNE") != nullptr;
  const std::string sig =
      "qmm_tile=" + std::to_string(_qmm_tile) +
      " mma2=" + std::to_string((int)_use_mma2) +
      " tn2=" + std::to_string((int)_fn_dense_mma_tn2.valid()) +
      " mma_min_m=" + std::to_string(_mma_min_m) +
      " i8=" + std::to_string(_i8 ? _i8->group() : 0) +
      " prune=" + std::to_string((int)!kNoPrune);
  auto weight_format = [&](const Linear& l) {
    if (l.quantized) {
      return "w" + std::to_string(l.bits) + "g" +
             std::to_string(_quant_group);
    }
    if (l.is_fp8()) { return "f8_" + std::to_string((int)l.f8); }
    return std::string("dense");
  };

  std::vector<QmmTune> tuned;
  std::string detail;
  int tune_rows = 0;
  std::size_t n_pruned = 0;
  int n_measured = 0, n_kept = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (const Shape& sh : shapes) {
    if (skip_fc1 && sh.N == 2 * F) { continue; }
    // A projection in the int8 layout has no route to choose.
    if (sh.l->gc > 0) { continue; }
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
    //
    // The session may already hold this shape's answer: then none of it
    // runs. The kept route must still be one of this shape's candidates
    // -- it always is under the same key; the check costs nothing and
    // keeps a bad entry from naming a kernel that cannot run here.
    const std::string wfmt = weight_format(*sh.l);
    const TuneKey rkey = TuneKey("h3-dit-route/1")("tiles", sig)
        ("fmt", wfmt)("rows", key)("tune_m", tune_m)
        ("N", sh.N)("K", sh.K);
    int kept = -1;
    bool have = find_tuned(answers, rkey, kept);
    if (have) {
      have = false;
      for (GemmRoute r : cands) { have = have || (int)r == kept; }
    }
    if (have) {
      ++n_kept;
    } else {
      ++n_measured;
    }
    if (!have && !kNoPrune && cands.size() > 2) {
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

    GemmRoute win = (GemmRoute)kept;
    if (!have) {
      const int w = autotune_vote((int)cands.size(), /*rounds=*/2,
          /*reps_for_us=*/1,
          [&](int i) { return time_route(cands[(std::size_t)i]); });
      win = cands[(std::size_t)w];
      store_tuned(answers, rkey, (int)win);
    }
    tuned.push_back(QmmTune{sh.N, sh.K, win});
    if (!detail.empty()) { detail += " "; }
    detail += std::string(sh.name) + "=" + route_name_(win);
    // Second decision, only where a split is possible at all: with the tile
    // settled, is the contraction better split? Driven through the same
    // dispatcher, so the tuner compares the winning route against itself
    // with and without the split rather than against a stand-in. The route
    // has to be pinned first -- measuring both at once would confound them.
    // Keyed on the REAL M -- plan() matches the row count the forward
    // asks with -- but MEASURED at tune_m like everything else. Both
    // split paths already walk M in row blocks of their own choosing, so
    // above one block the decision no longer moves with M; tune_m is at
    // least one such block at every shape here.
    auto run_route = [&](ComputeEncoder& enc) {
      dispatch_row_bands_(enc, xin, 0, *sh.l, yout, 0, tune_m, sh.N, sh.K,
                          win);
    };
    // What `run_route` encodes, for the split tuners' session keys: they
    // see a closure, not the route, the format or the rows it times.
    const std::string caller = "h3-dit/1 " + sig + " fmt=" + wfmt +
        " route=" + route_name_(win) + " tune_m=" + std::to_string(tune_m);
    // WHICH SPLIT TO TUNE FOLLOWS FROM WHICH ARM WILL RUN. gemm_ tries
    // the int8 GEMM before MmaSplitK, so on a shape int8 accepts the
    // dense split never runs -- and tuning it there would compare
    // identical work and record a winner that means nothing. The two are
    // mutually exclusive for exactly that reason, not merely by taste.
    if (_i8 && _i8->accepts(tune_m, sh.N, sh.K) &&
        _i8->split_available()) {
      const int sp = _i8->tune(_mc, sh.K, sh.N, M, run_route, caller);
      if (sp > 0) { detail += "+i8split" + std::to_string(sp); }
    } else if (route_is_mma_(win)) {
      const int sp = _splitk.tune(_mc, sh.K, sh.N, M, run_route, caller);
      if (sp > 0) { detail += "+split" + std::to_string(sp); }
    }
  }
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  // A set the session answered whole was not measured: the count is of
  // measurements (qmm_tune_count's contract), and a test reads it so. A
  // set whose every shape had one candidate counts, as it always did.
  if (n_measured > 0 || n_kept == 0) { ++_qmm_tune_count; }
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
        "({:.0f} ms, {} arm(s) pruned on the warm pass; {} shape(s) "
        "measured, {} from the session's answers)",
        M, tune_rows, detail, ms, n_pruned, n_measured, n_kept));
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
  //
  // PRECISION IS THE REFERENCE'S, NOT THE BEST AVAILABLE. Both ComfyUI
  // and diffusers cast the position to fp32 before multiplying, and then
  // round the rotation table to the block stack's dtype -- bf16 -- before
  // it is applied (`table.to(dtype)` / `cos.to(hidden_states.dtype)`).
  // This evaluated the angle in double and kept an fp32 table, which is
  // MORE exact than anything the checkpoint has ever been fed.
  //
  // MEASURED on a real 36556-row request: the bf16 cast moves the table
  // by 6.0e-4 mean / 2.0e-3 max per element, which is 320x the 1.9e-6
  // the double-vs-fp32 angle was worth. Being more exact is not neutral
  // when the quantization may be baked into what the model learned, so
  // the default reproduces it; `VPIPE_H3_ROPE_FP32=1` restores the old
  // table for an A/B.
  const int F = _cfg.rope_freq_dim;
  const int half = 3 * F;
  auto* cb = static_cast<float*>(cos_out.contents());
  auto* sb = static_cast<float*>(sin_out.contents());
  const char* fp32_env = std::getenv("VPIPE_H3_ROPE_FP32");
  const bool  keep_fp32 = fp32_env != nullptr && fp32_env[0] == '1';
  // Once per process: a run has to say which table it used, because the
  // two are indistinguishable in the output and the difference is the
  // kind that only shows up in a careful measurement weeks later.
  static bool said = false;
  if (!said && _mc != nullptr && _mc->session() != nullptr) {
    said = true;
    _mc->session()->log_normal(fmt(
        "MetalMiniMaxH3Transformer: rope table {}",
        keep_fp32 ? "fp32 (angles in double) -- MORE exact than the reference"
                  : "bf16, angles fp32 -- matching ComfyUI / diffusers"));
  }
  // Round-to-nearest-EVEN on the top 16 bits, which is what torch's
  // `.to(torch.bfloat16)` does. The buffer stays fp32; only the VALUE is
  // quantized, so the kernels are untouched.
  auto bf16_rne = [](float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    const std::uint32_t lsb  = (u >> 16) & 1u;
    const std::uint32_t bias = 0x7fffu + lsb;
    u += bias;
    u &= 0xffff0000u;
    float o;
    std::memcpy(&o, &u, sizeof o);
    return o;
  };
  for (int s = 0; s < L.seq_len; ++s) {
    const double* p = &L.position_ids[(std::size_t)s * 3];
    for (int a = 0; a < 3; ++a) {
      for (int i = 0; i < F; ++i) {
        const std::size_t o = (std::size_t)s * half + (std::size_t)(a * F + i);
        if (keep_fp32) {
          // The old path. Evaluated in double: the positions are absolute
          // and scaled by 32 over tens of thousands of rows, so the
          // angles are large.
          const double ang = p[a] * (double)_inv_freq[(std::size_t)i];
          cb[o] = (float)std::cos(ang);
          sb[o] = (float)std::sin(ang);
        } else {
          const float ang = (float)p[a] * _inv_freq[(std::size_t)i];
          cb[o] = bf16_rne(std::cos(ang));
          sb[o] = bf16_rne(std::sin(ang));
        }
      }
    }
  }
}

// ---- the timestep MLP, on the host in f32 ----------------------------

void
MetalMiniMaxH3Transformer::time_mlp_row_(double t, bool endpoint,
                                         float* out, bool base_only) const
{
  const int F = _cfg.freq_dim, H = _cfg.time_hidden, D = mlp_dim_();
  std::vector<float> row((std::size_t)F), h1((std::size_t)H);
  const int half = F / 2;
  // Diagnostic: the scale the sinusoidal grid sees. At t in [0, 1] every
  // angle is tiny and the embedding is nearly CONSTANT across the
  // schedule, which reads as a model that cannot tell its own noise
  // level. 1000 is the flow-scheduler convention.
  double tscale = 1.0;
  if (const char* ts = std::getenv("VPIPE_H3_TSCALE")) {
    tscale = std::atof(ts);
  }
  // diffusers Timesteps(freq_dim, flip_sin_to_cos=True,
  // downscale_freq_shift=0): cos first, then sin. Timesteps arrive in
  // [0, 1] UNSCALED -- there is no 1000x here, which is the scale every
  // other flow model in this tree conditions on. An ENDPOINT goes
  // through the same projection: upstream copies `time_proj` for it.
  for (int i = 0; i < half; ++i) {
    const double fr = std::exp(-std::log(1e4) * (double)i / (double)half);
    const double ang = t * tscale * fr;
    row[(std::size_t)i] = (float)std::cos(ang);
    row[(std::size_t)(half + i)] = (float)std::sin(ang);
  }
  for (int o = 0; o < H; ++o) {
    float acc = _time_in_b[(std::size_t)o];
    const float* w = &_time_in_w[(std::size_t)o * F];
    for (int i = 0; i < F; ++i) { acc += w[i] * row[(std::size_t)i]; }
    h1[(std::size_t)o] = acc;
  }
  // Each live slot's adapter on proj_in, PRE-activation. The endpoint
  // embedder is the flow-map slot's own copy and takes only its own
  // factors (see bind_flow_map_).
  for (std::size_t si = 0; !base_only && si < _lora.size(); ++si) {
    const LoraSlot& sl = _lora[si];
    if (endpoint) {
      if ((int)si == _flow.slot) {
        sl.end_in.apply(row.data(), sl.scale, h1.data());
      }
    } else {
      sl.time_in.apply(row.data(), sl.scale, h1.data());
    }
  }
  for (int o = 0; o < H; ++o) {
    const float a = h1[(std::size_t)o];
    h1[(std::size_t)o] = a / (1.0f + std::exp(-a));   // silu
  }
  for (int o = 0; o < D; ++o) {
    float acc = _time_out_b[(std::size_t)o];
    const float* w = &_time_out_w[(std::size_t)o * H];
    for (int i = 0; i < H; ++i) { acc += w[i] * h1[(std::size_t)i]; }
    out[o] = acc;
  }
  for (std::size_t si = 0; !base_only && si < _lora.size(); ++si) {
    const LoraSlot& sl = _lora[si];
    if (endpoint) {
      if ((int)si == _flow.slot) {
        sl.end_out.apply(h1.data(), sl.scale, out);
      }
    } else {
      sl.time_out.apply(h1.data(), sl.scale, out);
    }
  }
}

bool
MetalMiniMaxH3Transformer::time_embedding(const std::vector<float>& timesteps,
                                          const std::vector<float>* endpoints,
                                          std::vector<float>* out) const
{
  const int n_t = (int)timesteps.size();
  const int D = mlp_dim_();
  if (out == nullptr || n_t <= 0) { return false; }
  // PRUNED: the point of the curve each timestep lands on, t clamped to
  // [0, 1] as ComfyUI clamps it (a no-op for every timestep H3 makes) --
  // unless an adapter moves the embedding (pruned_projects_), which is
  // then the released one, projected below.
  const bool project = pruned_projects_();
  if (_cfg.pruned() && !project) {
    std::vector<float> t(timesteps);
    for (float& v : t) { v = std::min(1.0f, std::max(0.0f, v)); }
    out->assign((std::size_t)n_t * 8, 0.0f);
    curve_coords(_adaln_t_table, _cfg.curve_grid, t, out->data());
    return true;
  }
  if (!mlp_embedding_(timesteps, endpoints, out)) { return false; }
  if (project) {
    // The released embedding, silu'd and rounded to bf16 as the released
    // AdaLN reads it (time_embed_), onto the curve's coordinates:
    // c = pinv (s - m). What it leaves off the curve is the bake's
    // (bake_adaln, pruned) -- of the same rounded s, so the two parts
    // add up to the released W s + b.
    std::vector<float> pts((std::size_t)n_t * 8);
    for (int t = 0; t < n_t; ++t) {
      const float* e = out->data() + (std::size_t)t * D;
      for (int k = 0; k < 8; ++k) {
        const float* pk = _basis.pinv.data() + (std::size_t)k * D;
        double acc = 0.0;
        for (int j = 0; j < D; ++j) {
          acc += (double)pk[j] *
                 ((double)silu_bf16_(e[j]) - (double)_basis.m[(std::size_t)j]);
        }
        pts[(std::size_t)t * 8 + k] = (float)acc;
      }
    }
    *out = std::move(pts);
  }
  return true;
}

bool
MetalMiniMaxH3Transformer::mlp_embedding_(
    const std::vector<float>& timesteps, const std::vector<float>* endpoints,
    std::vector<float>* out) const
{
  const int n_t = (int)timesteps.size();
  const int F = _cfg.freq_dim, H = _cfg.time_hidden, D = mlp_dim_();
  if (out == nullptr || n_t <= 0 || (int)_time_in_w.size() != H * F ||
      (int)_time_out_w.size() != D * H) {
    return false;
  }
  // The flow-map blend, or the plain embedder. Off at strength 0 in
  // BOTH halves -- see two_time_active() -- so an endpoint handed to a
  // model with no live flow-map adapter is ignored rather than refused:
  // the caller may not know the strength was turned down.
  const bool two = two_time_active();
  if (two && (endpoints == nullptr || (int)endpoints->size() != n_t)) {
    return false;
  }
  out->assign((std::size_t)n_t * (std::size_t)D, 0.0f);
  std::vector<float> er(two ? (std::size_t)D : 0);
  for (int t = 0; t < n_t; ++t) {
    float* dst = out->data() + (std::size_t)t * D;
    time_mlp_row_((double)timesteps[(std::size_t)t], false, dst);
    if (!two) { continue; }
    time_mlp_row_((double)(*endpoints)[(std::size_t)t], true, er.data());
    // emb_t + gate * (emb_r - emb_t), term for term as upstream writes
    // it: rearranged to (1-g) emb_t + g emb_r it rounds differently in
    // f32, and this module is the one place that rounding is coherent
    // across every block and every step.
    const float g = _flow.gate;
    for (int o = 0; o < D; ++o) {
      dst[o] = dst[o] + g * (er[(std::size_t)o] - dst[o]);
    }
  }
  return true;
}

bool
MetalMiniMaxH3Transformer::time_embed_(const std::vector<float>& timesteps,
                                       const std::vector<float>* endpoints,
                                       SharedBuffer& out) const
{
  const int n_t = (int)timesteps.size();
  const int D = _cfg.time_dim;
  if (n_t <= 0 || out.byte_size() < (std::size_t)n_t * D * 2) { return false; }
  std::vector<float> temb;
  if (!time_embedding(timesteps, endpoints, &temb)) { return false; }
  auto* dst = static_cast<std::uint16_t*>(out.contents());
  for (std::size_t i = 0; i < temb.size(); ++i) {
    // Every AdaLN module applies its OWN silu to temb and casts the
    // result to its projection's dtype, so the activation runs at f32
    // here and only its output rounds to bf16. That order is the
    // reference's and its comment says why: rounding BEFORE the
    // activation biases every block's modulation identically at every
    // step, which then accumulates coherently down the denoising
    // trajectory instead of averaging out.
    const float acc = temb[i];
    // Pruned, the projections read the point as it is (no silu); this
    // copy is what an adapter's AdaLN modules read (pruned_mod_ reads
    // the f32 point).
    dst[i] = f32_to_bf16_(_cfg.pruned() ? acc
                                        : acc / (1.0f + std::exp(-acc)));
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
  if (!dst.empty() &&
      (dst.is_fp8() || !minimax_h3::assembled_parts(_ws->src(), nm).empty())) {
    if (linear_matches_(*_ws, nm, dst) && refill_linear_(*_ws, nm, dst)) {
      return true;
    }
  } else if (!dst.empty()) {
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
    const std::vector<std::vector<float>>& schedule, std::string* err,
    const std::vector<std::vector<float>>* endpoints)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = "minimax-h3 bake_adaln: " + m; }
    return false;
  };
  if (schedule.empty()) { return fail("empty schedule"); }
  if (_mc == nullptr || _ws == nullptr) { return fail("no weights"); }
  const Config& c = _cfg;
  // Pruned, a projection is [D, 8] -- 1.5 MB, read once at load and held
  // -- so a table of its rows would hold more than it retires. What IS
  // baked there is the part an adapted embedding leaves off the curve
  // (below), which only an adapter on the timestep MLP makes.
  if (c.pruned() && !pruned_projects_()) {
    return fail("the pruned AdaLN reads 8 columns and is held from load; "
                "nothing to bake");
  }
  const int D = c.adaln_out();
  const int FD = 2 * c.hidden;

  // A step's timesteps land CONTIGUOUSLY, so a block binds its slice at
  // an offset rather than gathering rows. The condition timestep repeats
  // in every step's group; deduplicating it would save one row in three
  // and cost a gather, which is the wrong trade at these sizes.
  // Under a live flow-map adapter the table is a function of (t, r), so
  // the endpoints are concatenated in the same order and a schedule
  // without them is refused -- a table built from t alone would be the
  // base model's modulation under the adapter's blocks.
  const bool two = two_time_active();
  if (two && (endpoints == nullptr || endpoints->size() != schedule.size())) {
    return fail("a flow-map adapter is live and the schedule carries no "
                "endpoints");
  }
  std::vector<float> all, all_r;
  _adaln_row0.clear();
  _adaln_nt.clear();
  for (std::size_t si = 0; si < schedule.size(); ++si) {
    const std::vector<float>& st = schedule[si];
    if (st.empty()) { return fail("a step with no timesteps"); }
    _adaln_row0.push_back((int)all.size());
    _adaln_nt.push_back((int)st.size());
    all.insert(all.end(), st.begin(), st.end());
    if (two) {
      const std::vector<float>& er = (*endpoints)[si];
      if (er.size() != st.size()) {
        return fail(fmt("step {} has {} timesteps and {} endpoints", si,
                        st.size(), er.size())());
      }
      all_r.insert(all_r.end(), er.begin(), er.end());
    }
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

  if (c.pruned()) {
    return bake_pruned_off_curve_(all, two ? &all_r : nullptr, err);
  }

  // The timestep MLP for EVERY row at once, on the host in f32 exactly
  // as a single step would do it.
  SharedBuffer temb =
      _mc->make_shared_buffer((std::size_t)T * (std::size_t)c.time_dim * 2);
  if (temb.empty()) { return fail("temb allocation failed"); }
  if (!time_embed_(all, two ? &all_r : nullptr, temb)) {
    return fail("time embedding failed");
  }

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
           l.scales.byte_size() + l.qbias.byte_size() +
           l.f8_scale.byte_size() + l.f8_srel.byte_size() +
           l.f8_cb.byte_size();
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
      _flops.flush(st);
      // CHECKED, because a discarded status here is the worst kind of
      // silence this model can produce. The table is a FRESH allocation:
      // if the GEMM's command buffer fails -- out of memory is the way,
      // and the video geometries put ~10 GB of scratch beside the
      // weights -- nothing has written it, and every block then
      // modulates every row of every step with whatever those pages
      // held. That is a whole-latent corruption in BOTH modalities from
      // a run that reported success.
      std::string bake_err;
      if (_fp8_failed) {
        _fp8_failed = false;
        return fail(fmt("AdaLN bake of block {}: its FP8 projection could "
                        "not be widened", i)());
      }
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
    _flops.flush(st);
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

  // A new run: a pool ceiling some earlier model collapsed may be asked
  // about again. See WiredPool::retry.
  _wire.new_run();

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
    if (steps > 1 && _wire.on()) {
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
      // kept -- and a refusal collapses the pool to what was
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
          _wire.budget() >> 20,
          _wire.on()
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
                                         bool narrow_ff,
                                         std::size_t arena_floor, int band,
                                         std::size_t widen)
{
  if (seq <= 0) { return 0; }
  const std::size_t S = (std::size_t)seq;
  const std::size_t T = (std::size_t)(n_text > 0 ? n_text : 0);
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
  //
  // An FP8 / int8 / W4A8 weight is widened into the same scratch ahead
  // of EVERY route, on every GPU, the AdaLN's included: `widen` is that.
  const auto dequant = [&]() -> std::size_t {
    const std::size_t H = (std::size_t)c.hidden;
    const std::size_t I = (std::size_t)c.inner();
    const std::size_t F = (std::size_t)c.ffn;
    const std::size_t four =
        with_dequant ? std::max({3 * I * H, H * I, 2 * F * H, H * F}) * 2
                     : 0;
    return std::max(four, widen);
  };
  if (band > 0 && band < seq) {
    // The banded layout (ensure_scratch_): per row of the sequence the
    // rope tables, x, qh | kh | vh and the two indices; per row of a band
    // nm, qkv and ob | oh | qb or ff, the wider; proj lies over qh.
    const std::size_t B = (std::size_t)band;
    const std::size_t H = (std::size_t)c.hidden;
    const std::size_t I = (std::size_t)c.inner();
    const std::size_t rot_half = (std::size_t)(3 * c.rope_freq_dim);
    const std::size_t N = (std::size_t)(n_t > kTimestepSlots
                                            ? n_t
                                            : kTimestepSlots);
    const std::size_t ff = narrow_ff ? 3 * I : 2 * (std::size_t)c.ffn;
    std::size_t b = S * (2 * rot_half * 4 + H * 2 + 3 * I * 2 +
                         2 * sizeof(int));
    b += B * (H + 3 * I + std::max(3 * I, ff)) * 2;
    b += T * H * 2 + N * ((std::size_t)c.time_dim + c.adaln_out() + 2 * H) * 2;
    return b + dequant();
  }
  // THE SLOT FLOOR, applied HERE and not left to the caller. The scratch
  // allocates max(n_t, kTimestepSlots) timestep slots so that the key in
  // ensure_scratch_ settles on the first forward instead of moving on the
  // second -- and an estimate that took `n_t` at face value would report
  // less than was allocated for every caller who passes the count it
  // actually has. generate-video already passes kTimestepSlots, so its
  // preflight was right by hand; nothing else was, and a preflight that
  // is right only when the caller knows to lie about its argument is one
  // rebuild away from being wrong again.
  const std::size_t N = (std::size_t)(n_t > kTimestepSlots
                                          ? n_t
                                          : kTimestepSlots);
  std::size_t total = 0;
  for (const ScratchItem& it : scratch_plan_(c, narrow_ff)) {
    total += (S * it.mul_seq + T * it.mul_text + N * it.mul_t) * it.esz;
  }
  // The arena is grown to hold the VDN branch's scratch when there is
  // one, so what the branch costs is the OVERSHOOT and not its size.
  // Zero at both geometries measured -- the attention's five [seq,
  // inner] windows are wider per row than the branch's whole scratch --
  // and this term is what says so when a geometry comes along where
  // they are not.
  const std::size_t arena = S * attn_arena_elems_(c, narrow_ff) * 2;
  if (arena_floor > arena) { total += arena_floor - arena; }
  total += dequant();
  return total;
}

int
MetalMiniMaxH3Transformer::band_rows(const Config& c, int seq)
{
  int band = kBandRows;
  if (const char* e = std::getenv("VPIPE_H3_BAND_ROWS")) {
    // Whole multiples of the widest query tile, so every band starts on
    // a tile the whole sequence's attention would start one on too.
    band = std::max(0, std::atoi(e)) / 128 * 128;
  }
  if (band <= 0 || seq < 2 * band) { return 0; }
  if (c.ane_ffn || c.ane_qkv) { return 0; }
  // BALANCED BANDS: every band but the last takes `step` rows (at most
  // `band`, a whole number of query tiles), the last the rest -- never
  // more than step, so the band-sized scratch and the two banded
  // pipelines (full, last) hold. Cutting `band` rows until the sequence
  // ran out left whatever remained: 4,120 rows in bands of 1,024 left a
  // 24-row last band, and a band that short is not computed as the
  // whole sequence's rows are -- the matrix-core GEMMs (the adapter's
  // included) start at 64 rows, int8 at 1,024, and the route tuner and
  // the int8 split key their answers on row buckets up to 4,096. So the
  // last band must clear all of those (half a band, for a small band).
  // At the default band that holds to ~250k rows; past it the split
  // with the longest last band is taken -- correct, if a few rows then
  // round as a short band's do.
  const int floor = std::min(std::max(tune_ceiling_(), 4096), band / 2);
  const int n0 = (seq + band - 1) / band;
  int best = 0;
  int best_last = 0;
  for (int n = n0; n < n0 + 16; ++n) {
    const int step = ((seq + n - 1) / n + 127) / 128 * 128;
    const int last = seq - (n - 1) * step;
    if (step > band || last <= 0) { continue; }
    if (last >= floor) { return step; }
    if (last > best_last) {
      best = step;
      best_last = last;
    }
  }
  return best;
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
      &b.adaln.qbias,
      &b.qkv.f8_scale, &b.out.f8_scale, &b.fc1.f8_scale, &b.fc2.f8_scale,
      &b.adaln.f8_scale, &b.qkv.cscales, &b.out.cscales, &b.fc1.cscales,
      &b.fc2.cscales, &b.adaln.cscales,
      &b.qkv.f8_srel, &b.out.f8_srel, &b.fc1.f8_srel, &b.fc2.f8_srel,
      &b.qkv.f8_cb, &b.out.f8_cb, &b.fc1.f8_cb, &b.fc2.f8_cb};
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
      // Nor one below the pool's minimum: never wired, heap-owned pages
      // that read partly out of RAM for reasons unrelated to this block.
      // See GenerativeModelManager::pool_wirable.
      if (!GenerativeModelManager::pool_wirable(*p)) { continue; }
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
  // Stored in pieces: FP8 exactly when assemble_ would make it FP8 --
  // every piece FP8 of one format -- and never quantized.
  if (const auto parts = minimax_h3::assembled_parts(src, nm);
      !parts.empty()) {
    fp8::Format f = fp8::Format::kNone;
    bool all = true;
    for (const auto& p : parts) {
      const fp8::Weight* fw = _f8.find(p.layer + ".weight");
      if (fw == nullptr || fw->format == fp8::Format::kI8 ||
          (f != fp8::Format::kNone && fw->format != f)) {
        all = false;
      } else {
        f = fw->format;
      }
    }
    return !l.quantized && (all ? l.f8 == f : !l.is_fp8());
  }
  const auto* si = src.info(nm + ".scales");
  const auto* ci = src.info(nm + ".weight");
  if (ci == nullptr) { return false; }
  if (const fp8::Weight* fw = _f8.find(nm + ".weight"); fw != nullptr) {
    return l.is_fp8() && l.f8 == fw->format &&
           l.f8_scale.empty() == fw->scale.empty() &&
           l.f8_per_row == fw->per_row && l.f8_rot == fw->rot_group &&
           l.f8_qgroup == fw->qgroup &&
           l.f8_cb.empty() == fw->codebook.empty();
  }
  if (l.is_fp8()) { return false; }
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
    // The i8x code / scale views of a Draw Things weight are the same
    // weight again (stream_sizing_alias): counted, a block read 3 bytes
    // an element where it holds 2 (dense) or 1 (token_plain).
    if (stream_sizing_alias(n)) { continue; }
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
      &b.adaln.qbias,
      &b.qkv.f8_scale, &b.out.f8_scale, &b.fc1.f8_scale, &b.fc2.f8_scale,
      &b.adaln.f8_scale, &b.qkv.cscales, &b.out.cscales, &b.fc1.cscales,
      &b.fc2.cscales, &b.adaln.cscales,
      &b.qkv.f8_srel, &b.out.f8_srel, &b.fc1.f8_srel, &b.fc2.f8_srel,
      &b.qkv.f8_cb, &b.out.f8_cb, &b.fc1.f8_cb, &b.fc2.f8_cb};
  std::size_t changed = 0;
  for (metal_compute::SharedBuffer* p : all) {
    const std::size_t n = _wire.wire_one(_mc, *p, on);
    // A buffer that would not wire for a reason of its own -- too small to
    // be a pool question, or its own mlock property -- leaves the pool able
    // to take the next: go on. Only a pool that can no longer take it STOPS
    // the block, keeping what is already wired rather than unwinding it: a
    // partly wired block is partly protected, and giving it back means
    // competing for it again against a pool that just said no.
    if (on && n == 0 && p->byte_size() > 0 && !p->is_wired() &&
        _wire.refused(_mc, *p)) {
      break;
    }
    changed += n;
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
  std::size_t changed = 0;
  for (metal_compute::SharedBuffer* b : scratch_buffers_()) {
    changed += _wire.wire_one(_mc, *b, on);
  }
  // The TRUNK: everything the weight set cached for this model, which
  // for a streaming DiT is the non-block tensors it holds for the whole
  // run. Read on every block of every forward and never shed, so it has
  // a better claim on the pool than any single resident block does.
  if (_ws) {
    _ws->for_each_weight([&](metal_compute::SharedBuffer& b) {
      changed += _wire.wire_one(_mc, b, on);
    });
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
    const std::size_t nb = block_bytes_(b);
    if (nb == 0) { continue; }
    // What its promotion booked: the block and its adapter factors.
    const std::size_t nl = lora_layer_bytes_(i);
    const std::size_t n = nb + nl;
    // Before the buffers go: give the wiring back. Dropping a wired
    // mapping would unwire it anyway, but doing it here keeps the pool's
    // counter honest without having to infer it from destructors.
    _wire.note_unwired(wire_block_(b, false) + wire_lora_(i, false));
    b = Block{};
    drop_lora_(i);
    _lora_kept -= std::min(nl, _lora_kept);
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
MetalMiniMaxH3Transformer::ensure_scratch_(int seq, int n_text, int n_t,
                                           std::size_t arena_floor, int band)
{
  // KEPT IF IT IS BIG ENOUGH, not if the geometry is identical.
  //
  // `n_t` is the count of DISTINCT timesteps in one forward, and it
  // moves WITHIN a denoise -- 1 on the first step and 2 on the second,
  // because the video and audio schedules only coincide at the start.
  // Nothing about the layout depends on it: three small buffers are
  // sized by it (temb, mod, fmod -- a few hundred KB between 1 and 2)
  // and every kernel dispatches over the forward's own count. So a
  // rebuild for a larger n_t rebuilt every activation buffer in the
  // model to grow three of them.
  //
  // THAT IS AN OUT-OF-MEMORY AND NOT A SLOW PATH. The new set is built
  // in full before the old one is dropped, so the peak is both at once
  // -- at 99128 rows, 13 GB twice on a 24 GB box, which is exactly how
  // a 1344x768 328-frame run died at step 1 with
  // kIOGPUCommandBufferCallbackErrorOutOfMemory five minutes in.
  //
  // The floor is a MINIMUM for the same reason: an arena that already
  // clears it is not a reason to reallocate.
  // ...and a q|k|v scratch wide enough for a stream forward's staging.
  const bool qkv_fits = _s.qkv.byte_size() >= _qkv_floor_elems * 2;
  if (_s.seq == seq && _s.n_text == n_text && _s.n_t >= n_t
      && _s.band == band && _s.attn.byte_size() >= arena_floor && qkv_fits) {
    return true;
  }
  // Under a floor (set_scratch_floor), a scratch that FITS is kept, and
  // one that does not is rebuilt at the floor -- so a run that knows its
  // longest forward pays for one allocation.
  if (_scratch_floor_seq > 0) {
    if (_s.seq >= seq && _s.n_text >= n_text && _s.n_t >= n_t &&
        _s.band == band && _s.attn.byte_size() >= arena_floor && qkv_fits) {
      return true;
    }
    seq    = std::max(seq, _scratch_floor_seq);
    n_text = std::max(n_text, _scratch_floor_text);
  }
  // WHAT MOVED, and it is worth a line: this reallocates every activation
  // buffer, and at video geometry that is over 13 GB -- built in full
  // BEFORE the old set is dropped, so the peak is both at once. A
  // rebuild in the middle of a denoise is therefore not a slow path, it
  // is an out-of-memory waiting for a big enough clip.
  if (_mc != nullptr && _mc->session() != nullptr && _s.seq > 0) {
    _mc->session()->log_normal(fmt(
        "MetalMiniMaxH3Transformer: activation scratch REBUILT -- "
        "(seq {}, n_text {}, n_t {}) -> (seq {}, n_text {}, n_t {})",
        _s.seq, _s.n_text, _s.n_t, seq, n_text, n_t));
  }
  // TAKE THE ARENA BACK BEFORE ALLOCATING THE NEXT ONE. The branch
  // holds a window over it, which keeps it alive -- so a geometry
  // change that allocated first would have two arenas resident at once,
  // which is the peak this whole scheme exists to avoid.
  // `_vdn_proj` is a window over the same arena and holds it alive for
  // exactly the same reason; ensure_vdn_ rebuilds it every forward.
  if (_vdn) { _vdn->set_arena(metal_compute::SharedBuffer{}); }
  // Sol borrows two of these buffers for the length of each of its
  // calls, and its windows hold them alive; same rule.
  // Sage's windows hold the same arena alive; same rule, same reason.
  if (_sage) {
    _sage->set_arena(metal_compute::SharedBuffer{},
                     metal_compute::SharedBuffer{});
  }
  if (_sol) {
    _sol->set_arena(metal_compute::SharedBuffer{},
                    metal_compute::SharedBuffer{});
  }
  if (_sol_tail) {
    _sol_tail->set_arena(metal_compute::SharedBuffer{},
                         metal_compute::SharedBuffer{});
  }
  _vdn_proj = metal_compute::SharedBuffer{};
  const Config& c = _cfg;
  const std::size_t S = (std::size_t)seq, H = (std::size_t)c.hidden;
  const std::size_t I = (std::size_t)c.inner();
  auto mk = [&](std::size_t elems) { return _mc->make_shared_buffer(elems * 2); };
  Scratch s;
  s.seq = seq;
  s.n_text = n_text;
  // SLOTS, not the count this forward happens to have: sized for the
  // most a schedule can present so the key above settles on the first
  // forward instead of moving on the second. Matches the bound
  // GenerateVideoStage's preflight already sizes the box against
  // (kTimestepsUpperBound) -- the two are the same claim and a larger
  // one here would make the preflight an under-estimate.
  const int nt_alloc = n_t > kTimestepSlots ? n_t : kTimestepSlots;
  s.n_t = nt_alloc;
  const std::size_t rot_half = (std::size_t)(3 * c.rope_freq_dim);
  s.rcos = _mc->make_shared_buffer(S * rot_half * sizeof(float));
  s.rsin = _mc->make_shared_buffer(S * rot_half * sizeof(float));
  // Rows of the per-band buffers: the whole sequence's, or one band's
  // (the BANDED layout; see Scratch::band).
  const std::size_t R = band > 0 ? (std::size_t)band : S;
  s.band = band;
  s.x    = mk(S * H);
  s.nm   = mk(R * H);
  s.qkv  = mk(std::max(R * 3 * I, _qkv_floor_elems));
  // The arena, then five windows into it. subview() keeps the same
  // MTL::Buffer and carries the offset, so binding a window addresses
  // its slice with no copy -- and each window is CONTIGUOUS, which the
  // [rows, I] shapes require.
  // ...and a THIRD user, when the hybrid is on: the VDN branch carves
  // its whole scratch out of this, because the stretch it runs in --
  // after the qkv projection, before qk_norm -- is a stretch in which
  // neither the attention windows nor the FF intermediate is live. See
  // MetalVdnBranch::set_arena.
  if (band > 0) {
    // qh | kh | vh for every row, then one band's ob | oh | qb (Sol's
    // head-major output and gathered queries) with ff over them -- the
    // attention output is folded into nm before fc1 writes; the final
    // layer's proj over qh | kh, dead once the blocks are.
    const std::size_t win = S * I * 2;
    const std::size_t bw = R * I * 2;           // one band's [n, inner]
    const std::size_t ff_elems =
        R * (_ff_needs_wide ? 2 * (std::size_t)c.ffn : 3 * I);
    const std::size_t tail = std::max(3 * bw, ff_elems * 2);
    s.attn = _mc->make_shared_buffer(std::max(3 * win + tail, arena_floor));
    if (s.attn.empty()) { return false; }
    s.qh   = s.attn.subview(0 * win, win);
    s.kh   = s.attn.subview(1 * win, win);
    s.vh   = s.attn.subview(2 * win, win);
    s.ob   = s.attn.subview(3 * win, bw);
    s.oh   = s.attn.subview(3 * win + bw, bw);
    s.qb   = s.attn.subview(3 * win + 2 * bw, bw);
    s.ff   = s.attn.subview(3 * win, ff_elems * 2);
    s.proj = s.attn.subview(0, S * H * 2);
  } else {
    s.attn = _mc->make_shared_buffer(std::max(
        S * attn_arena_elems_(c, !_ff_needs_wide) * 2, arena_floor));
    if (s.attn.empty()) { return false; }
    const std::size_t win = S * I * 2;          // bytes per window
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
  }
  // The runtime LoRAs' [M, rank] intermediates, one region per SLOT so
  // two adapters do not serialise on the same rows. Rank 64 at 19k rows
  // is 2.4 MB -- the whole reason applying an adapter at runtime is
  // affordable is that this is the only extra buffer it needs, and a
  // second one is another 2.4.
  s.lora = _lora_rank_total > 0 ? mk(R * (std::size_t)_lora_rank_total)
                                : SharedBuffer{};
  _lora_scratch_rows = R;
  s.txt  = mk((std::size_t)n_text * H);
  s.temb = mk((std::size_t)nt_alloc * (std::size_t)c.time_dim);
  s.mod  = mk((std::size_t)nt_alloc * (std::size_t)c.adaln_out());
  s.fmod = mk((std::size_t)nt_alloc * 2 * H);
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
  if (_wire.on()) {
    for (metal_compute::SharedBuffer* b : scratch_buffers_()) {
      _wire.wire_one(_mc, *b, false);
    }
  }
  _s = std::move(s);
  ++_scratch_rebuilds;
  // The new scratch is wired by the forward, which calls wire_fixed_()
  // right after this -- see the note there on why the scratch goes into
  // the pool before any block does.
  return true;
}

// ---- forward ---------------------------------------------------------

// ---- VDN-H3's hybrid attention ---------------------------------------

bool
MetalMiniMaxH3Transformer::attach_linear_branch(const std::string& dir,
                                                std::string* err)
{
  auto fail = [&](const std::string& what) {
    if (err != nullptr) { *err = what; }
    return false;
  };
  if (!minimax_h3::vdn::load_config(dir, &_vdn_cfg, err)) { return false; }
  if (!_vdn_cfg.supported(err)) { return false; }

  // THROUGH THE MANAGER, like every other checkpoint: the linear branch
  // is a separate 4.28 GB safetensors next to this model's, and a set
  // opened privately here would be invisible to the only thing with a
  // session-wide view of memory.
  std::shared_ptr<WeightSet> ws =
      open_weight_set(dir + "/linear_branch",
                      _mc != nullptr ? _mc->session() : nullptr);
  if (!ws) { return fail("cannot open " + dir + "/linear_branch"); }

  minimax_h3::MetalVdnBranch::Dims dims;
  dims.heads    = _cfg.n_heads;
  dims.head_dim = _cfg.head_dim;
  dims.hidden   = _cfg.hidden;
  dims.n_layers = _cfg.n_layers;
  dims.matrix_cores = _mc != nullptr && _mc->supports_matrix_cores();
  std::unique_ptr<minimax_h3::MetalVdnBranch> br =
      minimax_h3::MetalVdnBranch::load(ws, _mc, _vdn_cfg, dims, err);
  if (!br) { return false; }

  // The windowed softmax. Refused, the hybrid is not merely slower --
  // the linear half carries the window's complement and would double
  // count everything -- so this is a load failure and not a fallback.
  _fn_sdpa_spans = _lib_sdpa.function("sdpa_spans_f16");
  if (!_fn_sdpa_spans.valid()) {
    return fail("sdpa_spans_f16 did not validate");
  }
  // THE BRANCH FOLLOWS THE DiT. If this stack streams its own blocks it
  // is because the box could not hold them, and adding 4.28 GB of branch
  // beside it would undo the decision that was just made -- so the
  // branch streams too, and its blocks are released after the same
  // per-block commit the DiT's are. A resident stack keeps both.
  br->set_stream_blocks(_stream_blocks);
  _vdn = std::move(br);
  _vdn->set_flops(&_flops);   // credited at this model's commits
  if (_mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalMiniMaxH3Transformer: VDN hybrid attention attached ({} "
        "blocks, window radius {} chunk {})",
        _cfg.n_layers, _vdn_cfg.radius, _vdn_cfg.chunk));
  }
  return true;
}

// The per-forward VDN state: the window as CSR spans, and the two
// buffers the readout passes through.
//
// Rebuilt only when the geometry moves. The spans are a few hundred
// kilobytes but building them walks every (query frame, key frame)
// pair, and a denoise loop asks the same question 50 times a step.
int
MetalMiniMaxH3Transformer::forward_band(int seq) const
{
  // The forward's own rule (see where it decides), less what only a
  // request says (streaming attention).
  if (_vdn || !_steel_ok || !_fn_trope_band.valid() ||
      (_fused_attn & kFusedAttnQkv) != 0 ||
      (_fused_attn & kFusedAttnOut) == 0) {
    return 0;
  }
  return band_rows(_cfg, seq);
}

double
MetalMiniMaxH3Transformer::sol_kept_fraction() const
{
  if (!_sol) { return -1.0; }
  const long long layers = _cfg.n_layers - _cfg.sol.dense_layers;
  const long long tot =
      _sol_routed_total > 0
          ? _sol_routed_total
          : _sol->total_blocks() * (layers > 0 ? layers : 1);
  const long long got =
      _sol->exact_blocks() + (_sol_tail ? _sol_tail->exact_blocks() : 0);
  return tot > 0 && got > 0 ? (double)got / (double)tot : -1.0;
}

std::size_t
MetalMiniMaxH3Transformer::sol_held_bytes() const
{
  std::size_t n = 0;
  for (const MetalSolAttention* o : {_sol.get(), _sol_tail.get()}) {
    if (o != nullptr) {
      n += o->resident_bytes() + MetalSolAttention::pinned_bytes();
    }
  }
  return n;
}

std::size_t
MetalMiniMaxH3Transformer::sol_scratch_bytes(int seq) const
{
  if (!_sol || seq <= 0) { return 0; }
  // The same condition forward() applies: every layer dense means the
  // routed path is never reached and nothing is allocated for it.
  if (_cfg.sol.dense_layers >= _cfg.n_layers) { return 0; }
  const std::size_t I = (std::size_t)_cfg.inner();
  const int band = forward_band(seq);
  if (band > 0) {
    // Two objects, a band's calls each, lent the band's qkv buffer.
    const std::size_t lend = (std::size_t)band * 3 * I * 2;
    std::size_t n = MetalSolAttention::band_private_bytes(
        _cfg.n_heads, _cfg.n_heads, band, seq, _cfg.head_dim,
        _cfg.sol.key_block, _sol->uses_matrix_cores(), lend, 0);
    if (seq % band != 0) {
      n += MetalSolAttention::band_private_bytes(
          _cfg.n_heads, _cfg.n_heads, seq % band, seq, _cfg.head_dim,
          _cfg.sol.key_block, _sol->uses_matrix_cores(), lend, 0);
    }
    // ...and the layer's keys, summarised once for every band's call
    // (MetalSolAttention::prepare_keys): the first object's own.
    n += MetalSolAttention::prepared_keys_bytes(
        _cfg.n_heads, seq, _cfg.head_dim, _cfg.sol.key_block);
    return n;
  }
  const std::size_t win = (std::size_t)seq * I * 2;
  return MetalSolAttention::private_bytes(
      _cfg.n_heads, seq, _cfg.head_dim, _cfg.sol.key_block,
      _sol->uses_matrix_cores(), 3 * win, win);
}

std::size_t
MetalMiniMaxH3Transformer::sage_lend_bytes_(int seq) const
{
  if (seq <= 0) { return 0; }
  // SOL TAKES `_s.qkv` WHENEVER IT IS ON -- three attention windows of
  // it, which at video geometry is the whole thing -- and the fused
  // attention spellings READ q/k/v out of it in place, so a Sage carve
  // there would be quantizing the buffer the attention is reading.
  // Spare in exactly one case, and this is the only place that says so.
  const bool sol_on = (bool)_sol && _cfg.sol.dense_layers < _cfg.n_layers;
  if (sol_on || (_fused_attn & kFusedAttnQkv) != 0) { return 0; }
  return 3 * (std::size_t)seq * (std::size_t)_cfg.inner() * 2;
}

std::size_t
MetalMiniMaxH3Transformer::sage_scratch_bytes(int seq) const
{
  if (!_sage || seq <= 0) { return 0; }
  // The same condition forward() applies: every layer dense means the
  // int8 path is never reached and nothing is allocated for it.
  if (_cfg.sage.dense_layers >= _cfg.n_layers) { return 0; }
  // H3 is MHA, so the KV head count is the query head count. The blocks
  // are the flash kernel's, not Sage's own choice: the scale arrays are
  // one entry per its tiles. Banded: one band's queries against the
  // whole sequence's keys (and V), lent nothing -- those outlive every
  // band of a layer.
  const int band = forward_band(seq);
  return MetalSageAttention::private_bytes(
      _cfg.n_heads, _cfg.n_heads, band > 0 ? band : seq, seq,
      _cfg.head_dim, MetalSageAttention::nax_query_block(),
      MetalSageAttention::nax_key_block(),
      band > 0 ? 0 : sage_lend_bytes_(seq), 0,
      _sage->pv8_for(_cfg.sage));
}

std::size_t
MetalMiniMaxH3Transformer::i8_scratch_bytes(int seq) const
{
  if (!_i8 || seq <= 0) { return 0; }
  // fc2's K is the widest activation, fc1 [2 ffn, hidden] the widest
  // weight; the row bands gemm_ cuts are at most the sequence -- or, in
  // the banded forward, one band.
  const int band = forward_band(seq);
  return I8GemmContext::forward_bytes_bound(
      band > 0 ? band : seq, _cfg.ffn, 2 * _cfg.ffn, _cfg.hidden,
      _i8->tier() == I8GemmContext::Tier::kAccurate);
}

namespace {

// The A/B for the arena carve. Set VPIPE_H3_NO_VDN_ARENA and the branch
// allocates its own scratch and the readout projection gets its own
// buffer again -- which is the same run costing a gigabyte more, and
// the way to check that the sharing changed nothing but the footprint.
bool
vdn_arena_enabled_()
{
  static const bool on = std::getenv("VPIPE_H3_NO_VDN_ARENA") == nullptr;
  return on;
}

}  // namespace

std::size_t
MetalMiniMaxH3Transformer::vdn_scratch_bytes(const minimax_h3::PackedLayout& L,
                                             int grid_h, int grid_w,
                                             int n_text,
                                             std::size_t* arena_floor) const
{
  if (arena_floor != nullptr) { *arena_floor = 0; }
  if (!_vdn || grid_h <= 0 || grid_w <= 0) { return 0; }
  const int tpf = grid_h * grid_w;
  if (L.num_video_rows <= 0 || L.num_video_rows % tpf != 0) { return 0; }
  minimax_h3::MetalVdnBranch::Dims dims;
  dims.heads    = _cfg.n_heads;
  dims.head_dim = _cfg.head_dim;
  dims.hidden   = _cfg.hidden;
  dims.n_layers = _cfg.n_layers;
  dims.matrix_cores = _mc != nullptr && _mc->supports_matrix_cores();
  minimax_h3::MetalVdnBranch::Geometry g;
  g.frames   = L.num_video_rows / tpf;
  g.grid_h   = grid_h;
  g.grid_w   = grid_w;
  g.text_len = n_text;
  // A window that covers the clip turns the linear half off entirely,
  // and then reserve() allocates nothing -- so neither should this.
  // Asked BEFORE ensure_vdn_ has run, though, so the cover has to be
  // recomputed rather than read off _vdn_full_cover.
  const std::vector<minimax_h3::vdn::Bound> bounds =
      minimax_h3::vdn::window_bounds(g.frames, _vdn_cfg.radius,
                                     _vdn_cfg.chunk);
  bool covers = true;
  for (const minimax_h3::vdn::Bound& b : bounds) {
    if (b.lo > 0 || b.hi < g.frames - 1) { covers = false; break; }
  }
  // THE READOUT IS OWED WHETHER OR NOT THE LINEAR HALF RUNS: the gate
  // needs the branch attached, so ensure_vdn_ allocates it under full
  // cover too. [video_rows, inner], bf16 -- 270 MB at 37 frames and
  // 673 at 92, and until now nothing counted it either.
  const std::size_t I = (std::size_t)(_cfg.n_heads * _cfg.head_dim);
  const std::size_t ro = (std::size_t)L.num_video_rows * I * 2;
  if (covers) { return ro; }
  if (!vdn_arena_enabled_()) {
    // The A/B leg: nothing is shared, so the branch's whole scratch is
    // a second allocation again and the projection is its own buffer.
    return ro + (std::size_t)L.num_video_rows * (std::size_t)_cfg.hidden * 2
           + minimax_h3::MetalVdnBranch::scratch_bytes(dims, _vdn_cfg, g);
  }
  if (arena_floor != nullptr) {
    *arena_floor = minimax_h3::MetalVdnBranch::arena_bytes(dims, _vdn_cfg, g);
  }
  return ro + minimax_h3::MetalVdnBranch::pinned_bytes(dims, _vdn_cfg);
}

bool
MetalMiniMaxH3Transformer::ensure_vdn_(const Step& in,
                                       const minimax_h3::PackedLayout& L,
                                       int bq, int bk, std::string* err)
{
  auto fail = [&](const std::string& what) {
    if (err != nullptr) { *err = what; }
    return false;
  };
  const int tpf = in.video_grid_h * in.video_grid_w;
  if (tpf <= 0 || L.num_video_rows % tpf != 0) {
    return fail("the video grid does not divide the video rows");
  }
  const int frames = L.num_video_rows / tpf;

  // THE WINDOW FIRST, because whether the linear half runs at all
  // decides whether its buffers are worth allocating -- and at
  // production geometry they are gigabytes.
  if (_vdn_frames != frames || _vdn_tpf != tpf
      || _vdn_vstart != L.video_start || _vdn_seq != L.seq_len) {
    minimax_h3::vdn::WindowMask mask;
    mask.seq_len          = L.seq_len;
    mask.video_start      = L.video_start;
    mask.num_frames       = frames;
    mask.tokens_per_frame = tpf;
    mask.anchors          = _vdn_cfg.anchors;
    mask.bounds = minimax_h3::vdn::window_bounds(frames, _vdn_cfg.radius,
                                                 _vdn_cfg.chunk);
    const minimax_h3::vdn::WindowSpans sp =
        minimax_h3::vdn::build_window_spans(mask);
    auto ib = [&](const std::vector<int>& v) {
      metal_compute::SharedBuffer b =
          _mc->make_shared_buffer(v.size() * sizeof(int));
      if (!b.empty()) {
        std::memcpy(b.contents(), v.data(), v.size() * sizeof(int));
      }
      return b;
    };
    _vdn_span_off = ib(sp.off);
    _vdn_span_lo  = ib(sp.start);
    _vdn_span_hi  = ib(sp.end);
    if (_vdn_span_off.empty() || _vdn_span_lo.empty()
        || _vdn_span_hi.empty()) {
      return fail("cannot allocate the VDN window spans");
    }
    _vdn_bounds = mask.bounds;
    // The reference's own predicate, and it must stay that predicate:
    // the two halves partition the keys, so anything that makes them
    // disagree about the window's edge double counts or drops frames.
    _vdn_full_cover = true;
    for (const minimax_h3::vdn::Bound& b : mask.bounds) {
      if (b.lo > 0 || b.hi < frames - 1) { _vdn_full_cover = false; break; }
    }
    _vdn_frames = frames;
    _vdn_tpf    = tpf;
    _vdn_vstart = L.video_start;
    _vdn_seq    = L.seq_len;
  }
  if (_vdn_full_cover) { return true; }

  // The block-sparse form of the same window, for whichever steel kernel
  // the caller resolved: 32x16 for the ALU one, 64x32 for the matrix-core
  // one. Both carry the has_spans specialisation, so the tiles are the
  // caller's to choose -- but a list built for one grid read against the
  // other is a different window and not a slower answer, which is why
  // the tag below carries them.
  if (bq <= 0 || bk <= 0) { return fail("no block tiling for the window"); }
  if (_vdn_bspan_seq != L.seq_len || _vdn_bspan_bq != bq
      || _vdn_bspan_bk != bk) {
    minimax_h3::vdn::WindowMask bm;
    bm.seq_len          = L.seq_len;
    bm.video_start      = L.video_start;
    bm.num_frames       = frames;
    bm.tokens_per_frame = tpf;
    bm.anchors          = _vdn_cfg.anchors;
    bm.bounds           = _vdn_bounds;
    const minimax_h3::vdn::BlockSpans bs =
        minimax_h3::vdn::build_block_spans(bm, bq, bk);
    auto ib = [&](const std::vector<int>& v) {
      metal_compute::SharedBuffer b =
          _mc->make_shared_buffer(v.size() * sizeof(int));
      if (!b.empty()) {
        std::memcpy(b.contents(), v.data(), v.size() * sizeof(int));
      }
      return b;
    };
    _vdn_qb_off    = ib(bs.off);
    _vdn_qb_blocks = ib(bs.blocks);
    // Five ints now: AttnSpanParams gained qb_stride.
    _vdn_sp_params = _mc->make_shared_buffer(5 * sizeof(int));
    _vdn_sp_bounds =
        _mc->make_shared_buffer((std::size_t)frames * 2 * sizeof(int));
    if (_vdn_qb_off.empty() || _vdn_qb_blocks.empty()
        || _vdn_sp_params.empty() || _vdn_sp_bounds.empty()) {
      return fail("cannot allocate the VDN block spans");
    }
    int* sp = static_cast<int*>(_vdn_sp_params.contents());
    sp[0] = L.video_start;
    sp[1] = tpf;
    sp[2] = frames;
    // Bit 0 is anchor COLUMNS, bit 1 anchor ROWS -- the kernel's
    // spelling of AnchorFrames, which it cannot see.
    sp[3] = (_vdn_cfg.anchors == minimax_h3::vdn::AnchorFrames::kColumns
             || _vdn_cfg.anchors == minimax_h3::vdn::AnchorFrames::kBoth)
                ? 1
                : 0;
    sp[3] |= (_vdn_cfg.anchors == minimax_h3::vdn::AnchorFrames::kRows
              || _vdn_cfg.anchors == minimax_h3::vdn::AnchorFrames::kBoth)
                 ? 2
                 : 0;
    // qb_stride 0: VDN's window is geometric, so every head walks the
    // SAME block list. Sol-Attn is the caller that sets this.
    sp[4] = 0;
    int* bnd = static_cast<int*>(_vdn_sp_bounds.contents());
    for (int f = 0; f < frames; ++f) {
      bnd[2 * f]     = _vdn_bounds[(std::size_t)f].lo;
      bnd[2 * f + 1] = _vdn_bounds[(std::size_t)f].hi;
    }
    _vdn_bspan_seq = L.seq_len;
    _vdn_bspan_bq  = bq;
    _vdn_bspan_bk  = bk;
    _vdn_kb_sparse = bs.blocks.size();
    _vdn_kb_dense = bs.off.size() > 1
                        ? (bs.off.size() - 1)
                              * (std::size_t)((L.seq_len + bk - 1) / bk)
                        : 0;
    // INFO and not debug: this is the line that says whether the hybrid
    // is doing anything at all at this geometry. Full cover turns the
    // linear half off and makes the run the stock model plus a gate --
    // correct, and indistinguishable from the outside -- so without a
    // report a short clip looks like a VDN generation and is not one.
    // Once per geometry, not per forward.
    // A KEYFRAME ANCHOR IS OUTSIDE THE WINDOW, and that is the layout
    // upstream runs it in. The branch was trained text-to-video, and
    // OpenVDN has since published first-frame, last-frame and
    // first-and-last conditioning on the SAME checkpoints (2026-09-08):
    // the keyframes are packed ahead of the generated video, held at
    // t = 0.999, and treated like the prompt and the soundtrack --
    // dense both ways in the window softmax and absent from the linear
    // scan. That is what falls out here too: the conditioning rows sit
    // below video_start, so the mask keeps them GLOBAL, and the branch,
    // sized from num_video_rows, never sees them. So it is reported,
    // not warned about.
    //
    // A REFERENCE is the same GEOMETRY and a different claim. Upstream's
    // "Ref2VA-like" mode (2026-09-17) packs references through the FL2VA
    // weights and renders with them, so on this partition the pairing is
    // published and reported. On the Ref2VA partition the branch would
    // be attached to blocks it was not trained with, which nothing
    // published does -- that one still warns.
    //
    // Which of the two this is comes off the LAYOUT, not off the
    // partition, and that distinction is the whole point of the mode:
    // references now run on FL2VA weights, so the partition no longer
    // says what the conditioning rows are. It also keeps the wording
    // honest -- a reference has no frame count, since each one carries
    // its own geometry, so dividing the rows by the target's rows-per-
    // frame would print a number that means nothing.
    if (_mc->session() != nullptr && L.num_condition_rows > 0) {
      if (L.ref2va && _cfg.partition == "ref2va") {
        _mc->session()->warn(fmt(
            "MetalMiniMaxH3Transformer: VDN's branch is trained on the "
            "FL2VA partition's blocks and this is Ref2VA; its {} reference "
            "rows are attended densely, outside the window, and the linear "
            "branch does not see them -- no published configuration pairs "
            "the two", L.num_condition_rows));
      } else if (L.ref2va) {
        _mc->session()->info(fmt(
            "MetalMiniMaxH3Transformer: VDN with {} reference rows "
            "(Ref2VA-like on the FL2VA partition), attended densely "
            "outside the window and skipped by the linear branch, as "
            "upstream conditions on them", L.num_condition_rows));
      } else {
        const int kf = tpf > 0 ? L.num_condition_rows / tpf : 0;
        _mc->session()->info(fmt(
            "MetalMiniMaxH3Transformer: VDN with {} keyframe {} ({} "
            "conditioning rows), attended densely outside the window and "
            "skipped by the linear branch, as upstream conditions on them",
            kf, kf == 1 ? "frame" : "frames", L.num_condition_rows));
      }
    }
    if (_mc->session() != nullptr && _vdn_kb_dense > 0) {
      _mc->session()->info(fmt(
          "MetalMiniMaxH3Transformer: VDN window over {} frames of {} "
          "tokens -- {}, {} of {} key blocks ({:.1f}% of dense)",
          frames, tpf,
          _vdn_full_cover ? "COVERS THE CLIP, so the linear half is off"
                          : "windowed",
          _vdn_kb_sparse, _vdn_kb_dense,
          100.0 * (double)_vdn_kb_sparse / (double)_vdn_kb_dense));
    }
  }

  const int I = _cfg.n_heads * _cfg.head_dim;
  const std::size_t vrows = (std::size_t)L.num_video_rows;
  if (_vdn_ro.byte_size() < vrows * (std::size_t)I * 2) {
    _vdn_ro = _mc->make_shared_buffer(vrows * (std::size_t)I * 2);
    if (_vdn_ro.empty()) { return fail("cannot allocate the VDN readout"); }
    // ZEROED, and it stays that way where it matters. Under anchors
    // `both` the branch never WRITES frames 0 and F-1 -- the softmax
    // side covers them in both directions, so the partition makes their
    // readout exactly zero -- and this buffer is reused across all 50
    // blocks. Left unzeroed those two frames would project whatever the
    // allocator handed over into the residual stream, once per block:
    // not a crash and not a NaN, just two frames of noise.
    std::memset(_vdn_ro.contents(), 0, _vdn_ro.byte_size());
  }
  // THE PROJECTION IS A WINDOW OVER `ob`, not an allocation. It is
  // written by the branch's out projection, which is encoded after the
  // model's own -- and the model's own is the last reader of `ob`. The
  // residual add consumes it on the next dispatch, long before the FF
  // writes over the front of the arena.
  //
  // It fits the window by construction: video rows are a run inside the
  // packed sequence, so vrows <= seq, and hidden < inner. Asserted
  // rather than assumed, because the two shapes come from different
  // places and a config where they crossed would corrupt the residual
  // stream quietly.
  const std::size_t projb = vrows * (std::size_t)_cfg.hidden * 2;
  const std::size_t win = (std::size_t)_s.seq * (std::size_t)I * 2;
  if (!vdn_arena_enabled_()) {
    if (_vdn_proj.byte_size() < projb) {
      _vdn_proj = _mc->make_shared_buffer(projb);
      if (_vdn_proj.empty()) {
        return fail("cannot allocate the VDN projection");
      }
    }
    return true;
  }
  if (_s.attn.byte_size() < 4 * win + projb) {
    return fail("the attention arena cannot hold the VDN projection");
  }
  _vdn_proj = _s.attn.subview(4 * win, projb);
  // And the branch's own scratch out of the same arena, ahead of that
  // window -- see MetalVdnBranch::set_arena for why nothing in it is
  // live while the branch runs.
  _vdn->set_arena(_s.attn);
  return true;
}


// ---- the ANE tier ---------------------------------------------------

bool
MetalMiniMaxH3Transformer::ane_setup_(int seq)
{
  if (_ane_tried) { return _ane != nullptr; }
  _ane_tried = true;
  if (_mc->session() == nullptr || seq <= 0) { return false; }
  // `ane_rows` 0 means auto-balance from the two engines' measured times.
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "minimax-h3";
  o.hidden  = _cfg.hidden;
  o.ffn     = _cfg.ffn;
  o.seq     = seq;
  o.rows    = _cfg.ane_rows;
  o.chunk   = ane_chunk_rows();
  o.profile = std::getenv("VPIPE_H3_ANE_PROFILE") != nullptr;
  _ane = AneFeedForward::create(o);
  return _ane != nullptr;
}

bool
MetalMiniMaxH3Transformer::ane_eligible_(int L, const Block& b, bool adapted)
{
  if (_ane == nullptr) { return false; }
  const int cap = (_cfg.ane_layers > 0)
                      ? std::min(_cfg.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  if (L >= cap) { return false; }
  // The module has no bias: a projection carrying one keeps the GPU. Dense
  // bf16 or 4/8-bit affine with all three parts. An adapter is merged into
  // the staged rows, which needs its rows in fc1's own order -- true of the
  // [gate; up] halves, not of the interleaved pairs.
  // ComfyUI's int8 is not a source the staging reads (AneFfnSource takes
  // FP8 codes, and a ConvRot weight is no elementwise code at all).
  auto stageable = [](const Linear& l) {
    if (l.empty() || !l.b.empty() || l.f8 == fp8::Format::kI8 ||
        l.f8 == fp8::Format::kW4A8) {
      return false;
    }
    if (!l.quantized) { return true; }
    return (l.bits == 4 || l.bits == 8) && !l.scales.empty() &&
           !l.qbias.empty();
  };
  const bool ok = !(adapted && b.fc1.gu_inter) && stageable(b.fc1) &&
                  stageable(b.fc2);
  if (!ok && !_ane_skip_warned && _mc->session() != nullptr) {
    _ane_skip_warned = true;
    _mc->session()->warn(fmt(
        "minimax-h3: block {} {}, so it (and any like it) keeps the GPU "
        "feed-forward", L,
        adapted && b.fc1.gu_inter
            ? "has an adapter on an interleaved fc1"
        : b.fc1.f8 == fp8::Format::kI8 || b.fc2.f8 == fp8::Format::kI8 ||
                  b.fc1.f8 == fp8::Format::kW4A8 ||
                  b.fc2.f8 == fp8::Format::kW4A8
            ? "stores its feed-forward as int8 / W4A8"
            : "has no bias-free fp16 or 4/8-bit affine feed-forward"));
  }
  return ok;
}

void
MetalMiniMaxH3Transformer::ane_stage_(int L, const Block& b,
                                      const LoraStack& fc1,
                                      const LoraStack& fc2)
{
  auto view = [](const Linear& l, std::size_t stride, std::size_t offset,
                 const LoraStack& st) {
    AneFfnSource s;
    s.w         = &l.w;
    s.codes     = &l.codes;
    s.scales    = &l.scales;
    s.qbias     = &l.qbias;
    s.quantized = l.quantized;
    s.bits      = l.bits;
    // FP8: the codes, one per element, and the f32 scale in `scales`,
    // which a dense source has no other use for (shared/ane-ffn.h).
    if (l.is_fp8()) {
      s.w         = &l.codes;
      s.bits      = l.f8 == fp8::Format::kE5M2 ? AneFfnSource::kFp8E5M2
                                               : AneFfnSource::kFp8E4M3;
      s.scales    = l.f8_scale.empty() ? nullptr : &l.f8_scale;
    }
    s.stride    = stride;
    s.offset    = offset;
    // B is [out rows, rank] in the weight's own row order, so the source
    // row that indexes the weight indexes B too.
    for (int i = 0; i < st.n && s.deltas < AneFfnSource::kMaxDeltas; ++i) {
      AneFfnSource::Delta& d = s.delta[s.deltas++];
      d.a     = &st.s[i].f->a;
      d.b     = &st.s[i].f->b;
      d.rank  = st.s[i].f->b_rank();
      d.group = st.s[i].f->group;
      d.parts = st.s[i].f->parts;
      d.scale = st.s[i].scale;
      d.b_stride = stride;        // the fc1 adapter is laid out like fc1
      d.b_offset = offset;
    }
    return s;
  };
  // fc1 is [gate; up], GATE first -- two halves as loaded, or (g, u) pairs
  // once interleaved for the fused kernel.
  const std::size_t FF = (std::size_t)_cfg.ffn;
  const bool inter = b.fc1.gu_inter;
  _ane->stage(L, inter ? view(b.fc1, 2, 0, fc1) : view(b.fc1, 1, 0, fc1),
              inter ? view(b.fc1, 2, 1, fc1) : view(b.fc1, 1, FF, fc1),
              view(b.fc2, 1, 0, fc2), _quant_group);
}

bool
MetalMiniMaxH3Transformer::ane_qkv_setup_(int seq)
{
  if (_ane_qkv_tried) { return _ane_qkv != nullptr; }
  _ane_qkv_tried = true;
  if (_mc->session() == nullptr || seq <= 0) { return false; }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "minimax-h3-qkv";
  o.hidden  = _cfg.hidden;
  o.ffn     = 3 * _cfg.inner();          // output width
  o.seq     = seq;
  o.chunk   = ane_chunk_rows();
  o.matmul  = true;
  o.profile = std::getenv("VPIPE_H3_ANE_PROFILE") != nullptr;
  if (const char* r = std::getenv("VPIPE_H3_ANE_QKV_ROWS")) {
    o.rows = (float)std::atof(r);
  }
  _ane_qkv = AneFeedForward::create(o);
  return _ane_qkv != nullptr;
}

bool
MetalMiniMaxH3Transformer::ane_qkv_eligible_(int L, const Block& b)
{
  if (_ane_qkv == nullptr) { return false; }
  const int cap = (_cfg.ane_layers > 0)
                      ? std::min(_cfg.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  if (L >= cap) { return false; }
  const Linear& l = b.qkv;
  // int8 keeps the GPU, as in ane_eligible_.
  if (l.empty() || !l.b.empty() || l.f8 == fp8::Format::kI8 ||
      l.f8 == fp8::Format::kW4A8) {
    return false;
  }
  if (!l.quantized) { return true; }
  return (l.bits == 4 || l.bits == 8) && !l.scales.empty() &&
         !l.qbias.empty();
}

void
MetalMiniMaxH3Transformer::ane_qkv_stage_(int L, const Block& b,
                                          const LoraStack& lora)
{
  AneFfnSource s;
  s.w         = &b.qkv.w;
  s.codes     = &b.qkv.codes;
  s.scales    = &b.qkv.scales;
  s.qbias     = &b.qkv.qbias;
  s.quantized = b.qkv.quantized;
  s.bits      = b.qkv.bits;
  if (b.qkv.is_fp8()) {
    s.w      = &b.qkv.codes;
    s.bits   = b.qkv.f8 == fp8::Format::kE5M2 ? AneFfnSource::kFp8E5M2
                                              : AneFfnSource::kFp8E4M3;
    s.scales = b.qkv.f8_scale.empty() ? nullptr : &b.qkv.f8_scale;
  }
  // The qkv adapter's B rows were permuted to the projection's own head
  // grouping at bind, so they map 1:1 (the Delta defaults).
  for (int i = 0; i < lora.n && s.deltas < AneFfnSource::kMaxDeltas; ++i) {
    AneFfnSource::Delta& d = s.delta[s.deltas++];
    d.a     = &lora.s[i].f->a;
    d.b     = &lora.s[i].f->b;
    // A banded q/k/v pair merges part by part (see Delta::group).
    d.rank  = lora.s[i].f->b_rank();
    d.group = lora.s[i].f->group;
    d.parts = lora.s[i].f->parts;
    d.scale = lora.s[i].scale;
  }
  _ane_qkv->stage(L, s, _quant_group);
}

std::size_t
MetalMiniMaxH3Transformer::ane_runtime_bytes(const Config& c, int seq) noexcept
{
  std::size_t n =
      AneFeedForward::runtime_bytes(c.hidden, c.ffn, seq, ane_chunk_rows());
  // The fused qkv projection's module, when the graph asked for it. The
  // preflight in generate-video booked this term long before the CLAIM
  // did, so the gate and the plan disagreed about the same tier; they
  // read the same function now.
  if (c.ane_qkv || std::getenv("VPIPE_H3_ANE_QKV") != nullptr) {
    n += AneFeedForward::matmul_runtime_bytes(c.hidden, 3 * c.inner(), seq,
                                              ane_chunk_rows());
  }
  return n;
}

int
MetalMiniMaxH3Transformer::ane_chunk_rows() noexcept
{
  return AneFeedForward::chunk_rows("VPIPE_H3_ANE_CHUNK");
}

MetalMiniMaxH3Transformer::Velocity
MetalMiniMaxH3Transformer::forward(const Step& in, std::string* err)
{
  auto fail = [&](std::string m) -> Velocity {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  // Sage's int8 P*V for THIS forward: configured, its kernels on this
  // host (pv8_for), and not one of the early steps the loop keeps on the
  // int8 QK alone. `sage8` is the config the prologues are handed,
  // saying so.
  const bool pv8_on =
      _sage && _sage->pv8_for(c.sage) && !in.sage_pv8_off;
  sage::Config sage8 = c.sage;
  sage8.pv8 = pv8_on;
  if (in.layout == nullptr || in.timesteps == nullptr ||
      in.row_timestep_index == nullptr || in.text == nullptr) {
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
  // MUST agree with qkv_fused_row(), which places a split adapter's
  // rows into this same layout; the two are pinned together by a test.
  const int QKV_HSTRIDE = c.qkv_per_head ? 3 * HD : HD;
  const int Q_OFF = 0;
  const int K_OFF = c.qkv_per_head ? HD : I;
  const int V_OFF = c.qkv_per_head ? 2 * HD : 2 * I;
  if (seq <= 0 || n_t <= 0 || n_text <= 0) { return fail("empty sequence"); }
  if ((int)in.row_timestep_index->size() != seq) {
    return fail("row_timestep_index does not match the layout");
  }
  if (two_time_active() &&
      (in.endpoints == nullptr ||
       in.endpoints->size() != in.timesteps->size())) {
    // Refused, not defaulted. A flow-map adapter's blocks were trained
    // under the blended embedding; run with emb(t) alone they produce a
    // plausible, wrong velocity -- the one failure nobody would notice.
    return fail("a flow-map (HyperFlow) adapter is live, so every forward "
                "needs its timesteps' ENDPOINTS (Step::endpoints, from "
                "build_row_time_pairs); set its lora scale to 0 to run the "
                "base model");
  }
  if (n_audio > 0 && in.audio == nullptr) {
    return fail("the layout has audio rows but no audio latents");
  }
  // A layout with NO video rows is legal: TaoMate-H3's audio teacher is a
  // text + audio sequence, and the reference runs it as one.
  if (n_video > 0 && in.video == nullptr) {
    return fail("the layout has video rows but no video latents");
  }
  if (n_audio <= 0 && n_video <= 0) {
    return fail("the layout has neither video nor audio rows");
  }
  if (n_video > 0 &&
      in.video->byte_size() < (std::size_t)n_video * VPE * 2) {
    return fail("video rows are smaller than num_video_rows * patch elems");
  }
  // ---- the streaming attention's preconditions ------------------------
  minimax_h3::StreamKv* const skv = in.stream_kv;
  bool stage_in_qkv = false;
  const bool sattn = skv != nullptr;
  const bool stream_commit = sattn && in.stream_commit;
  minimax_h3::StreamKv::CommitPlan splan;
  if (in.stream_commit && !sattn) {
    return fail("stream_commit names no stream_kv");
  }
  if (sattn) {
    if (!skv->ready() || skv->layers() != c.n_layers) {
      return fail(fmt("streaming attention: the cache holds {} layers, the "
                      "model {}", skv->layers(), c.n_layers)());
    }
    // [text | audio | video] and nothing else: the split attention reads
    // rows [0, n_text) as the prompt and everything after as media.
    if (n_cond > 0 || L.num_condition_audio_rows > 0 ||
        L.audio_start != n_text ||
        (n_video > 0 && L.video_start != n_text + n_audio) ||
        n_text + n_audio + n_video != seq) {
      return fail("streaming attention needs a [text | audio | video] "
                  "layout with no conditioning rows");
    }
    if (!_steel_ok || !_fn_copy_rect.valid()) {
      return fail("streaming attention needs the steel flash kernel and "
                  "copy_rect_f16");
    }
    // Staging lives in the q|k|v scratch (StreamKv::adopt_staging) when
    // nothing else writes that scratch inside the attention window -- the
    // ANE's q|k|v rows land in it asynchronously -- else apart. A run
    // names the width up front (set_scratch_floor); a forward only ever
    // raises it.
    stage_in_qkv = _ane_qkv == nullptr &&
                   std::getenv("VPIPE_H3_STAGING_APART") == nullptr;
    if (stage_in_qkv) {
      _qkv_floor_elems =
          std::max(_qkv_floor_elems, skv->staging_bytes(seq) / 2);
    } else if (!skv->ensure_staging(seq)) {
      return fail("streaming attention: cannot allocate the K/V staging");
    }
    if (stream_commit) {
      std::string perr;
      if (!skv->plan_commit(n_text, n_audio, n_video, &splan, &perr)) {
        return fail(perr);
      }
    }
    // A cache on disk starts reading block 0's rows now, under everything
    // the forward does before its first block.
    {
      std::string perr;
      if (!skv->begin_forward(stream_commit ? &splan : nullptr, &perr)) {
        return fail(perr);
      }
    }
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
  // The VDN branch carves its scratch out of the attention arena, so
  // the arena has to be sized for it -- and HERE is the only place that
  // can still happen, since the next line allocates it. Zero with no
  // branch, with no video grid, and when the window covers the clip and
  // the linear half does not run.
  std::size_t vdn_arena = 0;
  (void)vdn_scratch_bytes(L, in.video_grid_h, in.video_grid_w, n_text,
                          &vdn_arena);
  // A LONG SEQUENCE'S MAIN BLOCKS RUN IN ROW BANDS (band_rows): every
  // row's residual and q / k / v held, one band of the rest. Decided here,
  // with the layout: a path that needs the whole sequence's buffers keeps
  // them -- streaming attention, VDN's branch, the in-place q/k/v layout,
  // a head-major output, no steel kernels.
  int band = forward_band(seq);
  if (sattn || n_text > band) { band = 0; }
  _last_band = band;
  if (!ensure_scratch_(seq, n_text, n_t, vdn_arena, band)) {
    return fail("activation allocation failed (out of GPU memory)");
  }
  if (stage_in_qkv &&
      !skv->adopt_staging(_s.qkv, skv->capacity() + seq)) {
    return fail("streaming attention: the q|k|v scratch cannot hold the "
                "K/V staging");
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
  if (_wire.on()) {
    // RETRY A COLLAPSED POOL at the top of each forward -- after this
    // model's own refusal once the box has freed a block's worth, and a
    // ceiling someone else collapsed once per run. See WiredPool::retry.
    // The residency policy stopped growing when the pool ran out, and it
    // cannot see that the pool moved: the same "the ground moved" case the
    // AdaLN bake uses.
    if (_wire.retry(_mc, resident_block_bytes_())) {
      _resid.note_landscape_changed();
    }
    wire_fixed_(true);
  }
  // A stack residency has kept whole needs its slots no more.
  (void)release_idle_slots();
  _resid.note_reserve_allocated(
      scratch_bytes(c, seq, n_text, n_t, uses_dequant_scratch(), false, 0,
                    0, _f8_widen));
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
            paged_out, examined, _wire.wired_bytes() >> 20, freed >> 20,
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
  // A commit writes no velocity at all; an audio-only forward has none
  // for video.
  if (!stream_commit) {
    if (n_video > 0) {
      out.video = _mc->make_shared_buffer((std::size_t)n_video * VPE * 2);
    }
    if (n_audio > 0) {
      out.audio = _mc->make_shared_buffer((std::size_t)n_audio * AC * 2);
    }
    if ((n_video > 0 && out.video.empty()) ||
        (n_audio > 0 && out.audio.empty())) {
      return fail("velocity allocation failed");
    }
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
  // The step's first row in every baked table.
  const std::size_t brow =
      baked ? (std::size_t)_adaln_row0[(std::size_t)in.schedule_index] : 0;
  // ALWAYS, even baked. Baking removes the AdaLN projection's need for
  // it, but not the adaln ADAPTER's: that is applied per forward (so
  // lora_scale stays live) and reads temb as its input. Skipping it here
  // left the adapter projecting from an uninitialized scratch -- which
  // produced plausible numbers, not a crash, and showed up only as the
  // baked and unbaked paths disagreeing once an adapter was attached.
  // It costs a host pass over n_t rows, which is nothing.
  // An adapted embedding on a PRUNED DiT is only half on the curve; the
  // rest is the bake's, and without it the model runs a different AdaLN.
  if (c.pruned() && pruned_projects_() && !baked) {
    return fail("an adapter moves this pruned DiT's timestep embedding off "
                "its curve, and that part comes from a baked schedule "
                "(bake_adaln), which this step does not name");
  }
  if (!time_embed_(*in.timesteps, in.endpoints, s.temb)) {
    return fail("timestep embedding failed");
  }
  // PRUNED: the curve points the AdaLN projections read, in f32 -- fresh
  // every forward, as the curve branches' are below.
  if (c.pruned()) {
    std::vector<float> pts;
    if (!time_embedding(*in.timesteps, in.endpoints, &pts)) {
      return fail("pruned AdaLN points failed");
    }
    _pcoords = _mc->make_shared_buffer(pts.size() * 4);
    if (_pcoords.empty()) { return fail("pruned AdaLN points allocation"); }
    std::memcpy(_pcoords.contents(), pts.data(), pts.size() * 4);
  }
  // The curve branches' c(t), one block of n_t rows per curve slot. The
  // buffer is fresh every forward: the last one's command buffers may
  // still be reading the old.
  {
    int n_curve = 0;
    for (const LoraSlot& sl : _lora) { n_curve += sl.curve.empty() ? 0 : 1; }
    _curve_n_t = 0;
    if (n_curve > 0) {
      _curve_coords = _mc->make_shared_buffer((std::size_t)n_curve * n_t * 32);
      if (_curve_coords.empty()) { return fail("curve coords allocation"); }
      auto* cp = static_cast<float*>(_curve_coords.contents());
      int ci = 0;
      for (const LoraSlot& sl : _lora) {
        if (sl.curve.empty()) { continue; }
        curve_coords(sl.curve.table, sl.curve.knots, *in.timesteps,
                     cp + (std::size_t)(ci++) * n_t * 8);
      }
      _curve_n_t = n_t;
    }
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
  // VDN's hybrid attention, for this forward. Off unless a branch is
  // attached AND the caller named the target video's grid: the window
  // is over whole frames and the short conv is a (t, h, w) stencil, and
  // neither can be recovered from a row count.
  const bool vdn_on = (bool)_vdn && in.video_grid_h > 0
                      && in.video_grid_w > 0 && L.num_video_rows > 0
                      && !sattn;
  // TWO SWITCHES, not one. The per-head softmax gate is part of the
  // hybrid's attention whatever the window does; the WINDOW and the
  // linear branch are a matched pair that turn off together when the
  // window reaches every frame. See _vdn_full_cover.
  const bool spans_off = std::getenv("VPIPE_VDN_NO_STEEL_SPANS") != nullptr;
  // THE WINDOW DECIDES THE TILES, so it is resolved before them -- the
  // block list, the param block and the dispatch grid are all built for
  // one BQ/BK and there is no arrangement in which they may disagree.
  //
  // Both steel kernels carry the has_spans specialisation now, so a
  // windowed forward no longer costs the matrix cores: the ALU kernel
  // masks 32x16 blocks and the NAX one 64x32, from the same closed-form
  // predicate. What is NOT arranged for is mixing them -- if the NAX
  // specialisation does not build, the whole forward drops to the ALU
  // tiles rather than running the window at one block size and the text
  // rows at another out of one param block.
  //
  // On a clip the window covers this builds a pipeline the forward will
  // not use, because full cover is only known after ensure_vdn_ below
  // and that needs the tiles this decides. It is one pipeline object per
  // geometry, and the alternative is a forward whose tile size depends
  // on a value computed from the tile size.
  if (vdn_on && !spans_off && _attn_nax
      && (_attn_spans_seq != seq || _attn_spans_nax != 1)) {
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (seq % 64) == 0).set_bool(201, (seq % 32) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(303, true);
    _fn_attn_spans =
        _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc);
    if (_fn_attn_spans.valid()) {
      _attn_spans_seq = seq;
      _attn_spans_nax = 1;
    } else {
      _attn_nax = false;
    }
  }
  const int A_BQ = _attn_nax ? 64 : 32;
  const int A_BK = _attn_nax ? 32 : 16;
  if (vdn_on && !ensure_vdn_(in, L, A_BQ, A_BK, err)) { return {}; }
  const bool vdn_linear = vdn_on && !_vdn_full_cover;
  // SOL-ATTN AND THE VDN LINEAR BRANCH ARE EXCLUSIVE, and it is a
  // correctness rule rather than a policy. VDN's two halves PARTITION
  // the keys -- a windowed softmax plus a linear recurrence carrying
  // exactly its complement -- so approximating the softmax half would
  // leave the two describing different key sets, and the frames the
  // routing dropped would be counted by neither.
  //
  // ...AND IT MUST BE OFF WHEN IT WOULD ROUTE NOTHING. `sol_on` forces
  // the unfused head-major path below, so leaving it true with every
  // layer excluded would move the whole stack off the steel kernel to
  // run exactly the same dense attention by a slower route and with
  // different rounding. `dense_layers >= n_layers` is a legitimate
  // setting (it is the ablation that says the filter works), so it has
  // to mean "unchanged model", not "same answer, different kernel".
  const bool sol_on = (bool)_sol && !vdn_linear && !in.dense_attention &&
                      c.sol.dense_layers < c.n_layers && !sattn;
  _sol_fwd = c.sol;
  if (in.sol_routing) {
    _sol_fwd.tau = in.sol_tau;
    _sol_fwd.thresh =
        in.sol_exact ? sol::Threshold::kExact : sol::Threshold::kDiag;
  }
  // SAGE IS A KERNEL VARIANT, not a second attention: it needs the
  // matrix-core steel entry, because the int8 fragment MMA only exists
  // there. `_attn_nax` is resolved further down, so this is re-tested at
  // the point the functions are built rather than assumed here.
  const bool sage_want = (bool)_sage && c.sage.enabled &&
                         c.sage.dense_layers < c.n_layers && !sattn;
  // THE QUERY TILE, once the three specialisations that own the tile size are
  // known. DENSE-ONLY: the spans block list, Sage's int8 scale arrays and
  // Sol's blocks are each built for one query tile, so a forward using any of
  // them stays at BQ 64. The probe itself runs at most once per model, on the
  // first forward long enough for the wide tile to matter.
  const bool tile_dense = !vdn_on && !sol_on && !sage_want && !sattn;
  tune_attn_tile_(seq, tile_dense);
  // BUILT whenever the wide tile may run: latched to 128, or still probing,
  // which alternates whole blocks between the two and so needs both pairs.
  const bool wide_tile = _attn_nax && tile_dense && seq >= kAttnBqMinRows &&
                         (_attn_bq == 128 || !_attn_tile_latched);
  if (sol_on) {
    // LEND SOL THE TWO BUFFERS THAT ARE DEAD WHILE IT RUNS, and it
    // stops allocating a second attention's worth of scratch.
    //
    // Sol reads HEAD-MAJOR q/k/v, so with it on the fused spellings are
    // off and the trope has already copied the projection out by the
    // time it is called: `s.qkv`, three attention windows wide, is then
    // dead until the feed-forward writes over it -- which is the same
    // observation the FF below already makes about it. And the arena's
    // LAST window is dead too: Sol writes its output into `oh`, and
    // `ob` is not touched until the transpose that follows.
    //
    // MEASURED at 56 heads x 14861 rows: 852 MB lent against 489 MB
    // wanted, so what Sol costs the process is its pinned few hundred
    // bytes. See MetalSolAttention::set_arena.
    const std::size_t win = (std::size_t)seq * (std::size_t)I * 2;
    _sol->set_arena(_s.qkv,
                    _s.attn.byte_size() >= 5 * win
                        ? _s.attn.subview(4 * win, win)
                        : metal_compute::SharedBuffer{});
  }
  // AND SAGE'S, out of the same buffer when Sol has not taken it. The
  // size comes from sage_lend_bytes_ so the preflight and the carve
  // agree; 0 there means it allocates its own and the estimate says so.
  if (_sage) {
    const std::size_t lend = sage_lend_bytes_(seq);
    _sage->set_arena(lend > 0 && _s.qkv.byte_size() >= lend
                         ? _s.qkv.subview(0, lend)
                         : metal_compute::SharedBuffer{},
                     metal_compute::SharedBuffer{});
  }
  const bool steel_spans = vdn_linear && _steel_ok && !spans_off;
  // WHAT THE ROUTING KEPT, reported here and reset here, and both for
  // the same reason: this is the only point at which the counter is both
  // COMPLETE and FINISHED. It is written by the GPU and read on the
  // ENCODE thread, so mid-forward it holds a partial total; and it
  // accumulates over every routed layer, so it has to be read before it
  // is cleared. Reading it at the last block instead reported one
  // layer's total against every layer's count -- 78.4% for three layers
  // that each kept 26.1%. What is printed is therefore the PREVIOUS
  // forward, complete; a diagnostic may lag, it may not misattribute.
  //
  // Realized sparsity is a property of the ACTIVATIONS, not of tau: a
  // clustered synthetic fixture routes to 14% where a real H3 block
  // routes to 26% at the same tau, and that gap is the difference
  // between a projection and a measurement.
  if (sol_on && _sol) {
    const long long layers = c.n_layers - c.sol.dense_layers;
    // A banded forward routes several calls a layer, so the total is
    // what the calls themselves routed; a whole one, one call a layer.
    const long long tot =
        _sol_routed_total > 0
            ? _sol_routed_total
            : _sol->total_blocks() * (layers > 0 ? layers : 1);
    const long long got =
        _sol->exact_blocks() + (_sol_tail ? _sol_tail->exact_blocks() : 0);
    if (tot > 0 && got > 0 && _mc->session() != nullptr) {
      _mc->session()->log_normal(fmt(
          "[h3-dit] Sol-Attn kept {} of {} key blocks exact ({:.1f}%) over "
          "{} routed layers of the PREVIOUS forward", got, tot,
          100.0 * (double)got / (double)tot, layers));
    }
    _sol->reset_counts();
    if (_sol_tail) { _sol_tail->reset_counts(); }
  }
  _sol_routed_total = 0;
  if (_sol) { _sol_flops.begin(); }
  _vdn_blocks_run = 0;
  bool use_steel = _steel_ok;
  // Without the block-sparse kernel the window falls to sdpa_spans_f16,
  // which reads HEAD-MAJOR q/k/v -- the unfused layout -- so that arm
  // has to give up steel's fused spellings too. It is kept reachable
  // (VPIPE_VDN_NO_STEEL_SPANS) because it is the verified reference the
  // steel path is checked against, and the two must be A/B-able in one
  // process.
  if (vdn_linear && !steel_spans) { use_steel = false; }
  // Read ONCE per forward: the A/B setter can change it between two, and
  // the cached AttnParams below are tagged with the value they were
  // filled for.
  // SOL AND i8_gemm ARE INDEPENDENT, and this is the line where someone
  // would expect them to collide. They do not: i8_gemm chooses how the
  // block's GEMMs are computed -- qkv, the FF pair, the out projection --
  // and Sol chooses how the ATTENTION between them is computed. Neither
  // reads the other's state, and `fused_attn` below is an attention-side
  // LAYOUT, not a GEMM one: the qkv GEMM writes s.qkv the same way
  // either way, and the trope after it is what Sol needs. So the two can
  // be enabled together, and the only thing Sol costs is the in-place
  // fused qkv read, which is an attention optimisation i8_gemm never
  // used.
  //
  // SOL UNFUSES THE PROJECTION; IT DOES NOT TURN STEEL OFF. Sol reads and
  // writes HEAD-MAJOR, so the trope has to run -- but the layers Sol
  // does NOT route still want the steel kernel, and steel reads
  // head-major perfectly well (its strides are a parameter).
  //
  // Turning `use_steel` off instead, which is what this did first, sends
  // every dense layer to the SCALAR sdpa kernel. MEASURED: 3459 ms a
  // block became 66000, and the cost is entirely in the layers Sol never
  // touched -- skipping every one of Sol's own dispatches changed
  // nothing. A knob that makes the layers it excludes 19x slower is not
  // an accelerator, and it looked exactly like the new kernels being
  // slow.
  // The streaming split reads HEAD-MAJOR q/k/v and writes a head-major
  // o, exactly as Sol does, so it unfuses the same way.
  const int fused_attn = (sol_on || sattn) ? 0 : _fused_attn;
  // The FUNCTIONS depend on the sequence lengths alone; the PARAMS also
  // depend on which layout attention is reading, and the A/B setter
  // flips that between forwards. Kept apart so toggling the layout does
  // not rebuild two pipeline states -- an A/B that pays a rebuild in one
  // arm and not the other is measuring the rebuild.
  // Resolved HERE and not where sage_want was: the int8 QK exists only
  // in the matrix-core entry point, and `_attn_nax` is settled above.
  const bool sage_on = sage_want && _attn_nax;
  const bool attn_dirty = _attn_seq != seq || _attn_text != n_text
                          || _attn_nax_built != (_attn_nax ? 1 : 0)
                          || _attn_sage_built != (sage_on ? 1 : 0)
                          || _attn_pv8_built != (pv8_on ? 1 : 0)
                          // ...and the band's first video block, the layout's
                          || (pv8_on && _attn_band_nb != 0 &&
                              _attn_band_nb !=
                                  (L.video_start + A_BQ - 1) / A_BQ)
                          // ...AND the query tile, which is NOT a function of
                          // the sequence: a spans / Sol / Sage forward at some
                          // length leaves the wide pair unbuilt, and a later
                          // DENSE forward at that same length would otherwise
                          // find `_attn_bq128_seq` stale and quietly stay at
                          // BQ 64 for the rest of the run.
                          || _attn_wide_built != (wide_tile ? 1 : 0);
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
    // The tile `fill` writes for. A_BQ for the dense/text/spans block; set
    // to 128 around the wide pair's fill below.
    int A_BQ_used = A_BQ;
    auto fill = [&](SharedBuffer& pb, int qL) {
      auto* p = static_cast<P*>(pb.contents());
      p->B = 1; p->H = NH; p->D = HD;
      p->qL = qL; p->kL = qL;
      p->gqa_factor = 1; p->scale = scale;
      p->NQ = (qL + A_BQ_used - 1) / A_BQ_used;
      p->NK = (qL + A_BK - 1) / A_BK;
      p->NQ_aligned = qL / A_BQ_used; p->NK_aligned = qL / A_BK;
      p->qL_rem = qL - p->NQ_aligned * A_BQ_used;
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
    auto build = [&](int qL, bool spans, bool i8) {
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (qL % A_BQ) == 0).set_bool(201, (qL % A_BK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false)
          .set_bool(303, spans)
          .set_bool(sage::kQkInt8Constant, i8);
      // The int8 P*V rides on the int8 QK's function only: its operands
      // come from the same prologue, and only when it ran.
      MetalSageAttention::pv8_constants(fc, i8 && pv8_on);
      return _attn_nax
                 ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
                 : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
    };
    if (attn_dirty) {
      // TWO instantiations of the main entry, not one. The leading
      // `dense_layers` blocks run the f16 kernel and the rest the int8
      // one, so both have to exist for the same sequence -- and the
      // refiner's text attention is never routed to Sage (it is a
      // prompt, not a video sequence), so it is built dense only.
      _fn_attn_main = build(seq, false, false);
      _fn_attn_text = build(n_text, false, false);
      // THE WIDE TILE'S OWN PAIR. Its NQ / NQ_aligned / qL_rem differ from
      // the BQ 64 block, and the text and spans paths read that one, so it
      // gets a parameter buffer of its own rather than a shared one edited
      // per dispatch.
      if (wide_tile) {
        if (_attn_p_main_bq128.empty()) {
          _attn_p_main_bq128 = _mc->make_shared_buffer(sizeof(P));
        }
        if (!_attn_p_main_bq128.empty()) {
          const int save_bq = A_BQ_used;
          A_BQ_used = 128;
          fill(_attn_p_main_bq128, seq);
          A_BQ_used = save_bq;
          metal_compute::FunctionConstants fc128;
          fc128.set_bool(200, (seq % 128) == 0)
              .set_bool(201, (seq % A_BK) == 0)
              .set_bool(300, false).set_bool(301, false).set_bool(302, false)
              .set_bool(303, false)
              .set_bool(sage::kQkInt8Constant, false);
          _fn_attn_main_bq128 = _lib_attn_nax.function(
              "attn_steel_nax_h_bd128_bq128_bf16", fc128);
          _attn_bq128_seq = _fn_attn_main_bq128.valid() ? seq : -1;
        }
      } else {
        _fn_attn_main_bq128 = metal_compute::ComputeFunction{};
        _attn_bq128_seq = -1;
      }
      _fn_attn_main_i8 =
          sage_on ? build(seq, false, true) : metal_compute::ComputeFunction{};
      // The int8 P*V's band (see _fn_attn_main_i8_qk): the prefix rows'
      // own function, and the video rows' params from their first block.
      _fn_attn_main_i8_qk = metal_compute::ComputeFunction{};
      _attn_band_nb = 0;
      const bool band_on = !c.sage.pv8_all_rows;
      const int nb = (L.video_start + A_BQ - 1) / A_BQ;
      if (sage_on && pv8_on && band_on && nb > 0 &&
          nb < (seq + A_BQ - 1) / A_BQ) {
        metal_compute::FunctionConstants fq;
        fq.set_bool(200, (seq % A_BQ) == 0).set_bool(201, (seq % A_BK) == 0)
            .set_bool(300, false).set_bool(301, false).set_bool(302, false)
            .set_bool(303, false)
            .set_bool(sage::kQkInt8Constant, true);
        MetalSageAttention::pv8_constants(fq, false);
        fq.set_bool(sage::kQTokenScaleConstant, true);
        _fn_attn_main_i8_qk = _lib_attn_nax.function(
            "attn_steel_nax_h_bd128_bf16", fq);
        if (_attn_p_band.empty()) {
          _attn_p_band = _mc->make_shared_buffer(sizeof(P));
        }
        if (_fn_attn_main_i8_qk.valid() && !_attn_p_band.empty()) {
          std::memcpy(_attn_p_band.contents(), _attn_p_main.contents(),
                      sizeof(P));
          static_cast<P*>(_attn_p_band.contents())->NQ_aligned -= nb;
          _attn_band_nb = nb;
        }
      }
      // The same fallback as at load, for a sequence the matrix-core entry
      // refused: said once, so a Sage run that went dense is not reported
      // as a Sage run.
      if (sage_want && !_fn_attn_main_i8.valid() && !_sage_off_noted &&
          _mc->session() != nullptr) {
        _sage_off_noted = true;
        _mc->session()->log_normal(fmt(
            "MetalMiniMaxH3Transformer: sage_attn is off at {} rows -- the "
            "matrix-core flash entry does not build for this sequence, so "
            "attention runs the dense kernel", seq));
      }
      _attn_seq = seq;
      _attn_text = n_text;
      _attn_nax_built = _attn_nax ? 1 : 0;
      _attn_sage_built = sage_on ? 1 : 0;
      _attn_pv8_built = pv8_on ? 1 : 0;
      _attn_wide_built = wide_tile ? 1 : 0;
    }
  }
  // THE BANDED ATTENTION: a band's queries -- head-major in the whole-
  // sequence window, from the band's first row -- over every row's keys,
  // its output row-major into the band's own buffer. The dense kernel and
  // the dense parameters but for qL and the strides, so a query row's
  // arithmetic is the whole sequence's. One block per band size (full,
  // last) and, under Sage's int8 P*V, one for band 0 past its prompt and
  // soundtrack rows, which keep the int8 QK alone as the whole sequence's
  // do (`_attn_p_band`): qL kept, NQ_aligned short of those blocks.
  // Pipelines per band size: dense, and under Sage its three variants.
  const int band_nb =
      band > 0 && sage_on && pv8_on && !c.sage.pv8_all_rows
          ? (L.video_start + A_BQ - 1) / A_BQ
          : 0;
  if (band > 0 && (_attn_bandq_seq != seq || _attn_bandq_rows != band ||
                   _attn_bandq_nax != (_attn_nax ? 1 : 0) ||
                   _attn_bandq_sage != (sage_on ? 1 : 0) ||
                   _attn_bandq_nb != band_nb)) {
    struct P {
      int B, H, D, qL, kL, gqa_factor;
      float scale;
      int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
      std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
    };
    auto fill = [&](SharedBuffer& pb, int qL) {
      if (pb.empty()) { pb = _mc->make_shared_buffer(sizeof(P)); }
      if (pb.empty()) { return; }
      auto* p = static_cast<P*>(pb.contents());
      p->B = 1; p->H = NH; p->D = HD;
      p->qL = qL; p->kL = seq;
      p->gqa_factor = 1; p->scale = scale;
      p->NQ = (qL + A_BQ - 1) / A_BQ;
      p->NK = (seq + A_BK - 1) / A_BK;
      p->NQ_aligned = qL / A_BQ; p->NK_aligned = seq / A_BK;
      p->qL_rem = qL - p->NQ_aligned * A_BQ;
      p->kL_rem = seq - p->NK_aligned * A_BK;
      p->qL_off = 0;
      const std::int64_t hm[3] = {(std::int64_t)NH * seq * HD,
                                  (std::int64_t)seq * HD, HD};
      for (int i = 0; i < 3; ++i) {
        p->Q_strides[i] = hm[i];
        p->K_strides[i] = hm[i];
        p->V_strides[i] = hm[i];
      }
      p->O_strides[0] = (std::int64_t)qL * I;
      p->O_strides[1] = HD;
      p->O_strides[2] = I;
    };
    auto build = [&](int qL, int variant) {
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (qL % A_BQ) == 0).set_bool(201, (seq % A_BK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false)
          .set_bool(303, false)
          .set_bool(sage::kQkInt8Constant, variant >= 1);
      MetalSageAttention::pv8_constants(fc, variant == 3);
      if (variant == 2) { fc.set_bool(sage::kQTokenScaleConstant, true); }
      return _attn_nax
                 ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
                 : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
    };
    for (int k = 0; k < 2; ++k) {
      const int qL = k == 0 ? band : seq % band;
      for (int v = 0; v < 4; ++v) {
        _fn_attn_bandv[k][v] = metal_compute::ComputeFunction{};
      }
      if (qL == 0) { continue; }
      fill(k == 0 ? _attn_p_bandq : _attn_p_bandq_tail, qL);
      _fn_attn_bandv[k][0] = build(qL, 0);
      if (sage_on) {
        for (int v = 1; v < 4; ++v) { _fn_attn_bandv[k][v] = build(qL, v); }
      }
    }
    if (band_nb > 0 && !_attn_p_bandq.empty()) {
      fill(_attn_p_bandq_rest, band);
      if (!_attn_p_bandq_rest.empty()) {
        static_cast<P*>(_attn_p_bandq_rest.contents())->NQ_aligned -=
            band_nb;
      }
    }
    _attn_bandq_seq = seq;
    _attn_bandq_rows = band;
    _attn_bandq_nax = _attn_nax ? 1 : 0;
    _attn_bandq_sage = sage_on ? 1 : 0;
    _attn_bandq_nb = band_nb;
  }
  if (band > 0 &&
      (!_fn_attn_bandv[0][0].valid() ||
       (seq % band != 0 && !_fn_attn_bandv[1][0].valid()))) {
    return fail("banded attention: the steel pipelines did not build");
  }
  // Sol's last band on an object of its own (see _sol_tail).
  if (band > 0 && sol_on && seq % band != 0 && !_sol_tail) {
    std::string serr;
    _sol_tail = MetalSolAttention::load(_mc, /*bf16=*/true, &serr);
    if (!_sol_tail) {
      return fail("banded attention: Sol for the last band: " + serr);
    }
  }
  if (band > 0 && sol_on && _sol_tail) {
    _sol_tail->set_arena(_s.qkv, metal_compute::SharedBuffer{});
  }
  // The SAME entry point as the dense pair, specialised on has_spans --
  // so it is built here rather than in the block above, which only runs
  // when that pair is dirty. Its tag is the sequence length AND the
  // kernel, because the tiles differ between the two.
  //
  // The matrix-core spelling was resolved before the tiles were chosen
  // (it is what chooses them); this is the ALU one, which is what an M4
  // runs and what VPIPE_H3_NO_ATTN_NAX asks for.
  if (steel_spans && !_attn_nax
      && (_attn_spans_seq != seq || _attn_spans_nax != 0)) {
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (seq % A_BQ) == 0).set_bool(201, (seq % A_BK) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(303, true);
    _fn_attn_spans = _lib_attn.function("attn_steel_h_bd128_bf16", fc);
    _attn_spans_seq = seq;
    _attn_spans_nax = 0;
  }
  if (use_steel) {
    use_steel = _fn_attn_main.valid() && _fn_attn_text.valid();
  }
  // ---- the streaming split: two parameter blocks, two pipelines ------
  //
  // Filled per forward: the cache's row count moves with every commit,
  // and with it kL. Both attentions are the same steel entry the dense
  // path runs -- only the strides differ: queries and outputs are rows of
  // the head-major [NH, seq, HD] scratch, keys and values rows of the
  // [NH, cache + seq, HD] staging the cache gathers into.
  const int s_rows = sattn ? skv->rows() : 0;
  const int s_media = seq - n_text;
  const metal_compute::ComputeFunction* s_fn_text = nullptr;
  const metal_compute::ComputeFunction* s_fn_media = nullptr;
  if (sattn) {
    if (!use_steel) {
      return fail("streaming attention: the steel flash kernel is "
                  "unavailable");
    }
    struct P {
      int B, H, D, qL, kL, gqa_factor;
      float scale;
      int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
      std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
    };
    for (SharedBuffer* pb : {&_sattn_p_text, &_sattn_p_media}) {
      if (pb->empty()) { *pb = _mc->make_shared_buffer(sizeof(P)); }
      if (pb->empty()) { return fail("streaming attention: params"); }
    }
    const int SR = s_rows + seq;
    auto fill = [&](SharedBuffer& pb, int qL, int kL) {
      auto* p = static_cast<P*>(pb.contents());
      p->B = 1; p->H = NH; p->D = HD;
      p->qL = qL; p->kL = kL;
      p->gqa_factor = 1; p->scale = scale;
      p->NQ = (qL + A_BQ - 1) / A_BQ;
      p->NK = (kL + A_BK - 1) / A_BK;
      p->NQ_aligned = qL / A_BQ; p->NK_aligned = kL / A_BK;
      p->qL_rem = qL - p->NQ_aligned * A_BQ;
      p->kL_rem = kL - p->NK_aligned * A_BK;
      p->qL_off = 0;
      const std::int64_t q[3] = {(std::int64_t)NH * seq * HD,
                                 (std::int64_t)seq * HD, HD};
      const std::int64_t k[3] = {(std::int64_t)NH * SR * HD,
                                 (std::int64_t)SR * HD, HD};
      for (int i = 0; i < 3; ++i) {
        p->Q_strides[i] = q[i];
        p->O_strides[i] = q[i];
        p->K_strides[i] = k[i];
        p->V_strides[i] = k[i];
      }
    };
    fill(_sattn_p_text, n_text, n_text);
    fill(_sattn_p_media, s_media, SR);
    auto pipeline = [&](int qL, int kL) -> metal_compute::ComputeFunction* {
      const int nx = _attn_nax ? 1 : 0;
      const int aq = (qL % A_BQ) == 0 ? 1 : 0;
      const int ak = (kL % A_BK) == 0 ? 1 : 0;
      metal_compute::ComputeFunction& f = _fn_sattn[nx][aq][ak];
      if (!f.valid()) {
        metal_compute::FunctionConstants fc;
        fc.set_bool(200, aq != 0).set_bool(201, ak != 0)
            .set_bool(300, false).set_bool(301, false).set_bool(302, false)
            .set_bool(303, false)
            .set_bool(sage::kQkInt8Constant, false);
        f = _attn_nax
                ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
                : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
      }
      return &f;
    };
    s_fn_text = pipeline(n_text, n_text);
    s_fn_media = pipeline(s_media, SR);
    if (!s_fn_text->valid() || !s_fn_media->valid()) {
      return fail("streaming attention: the steel pipelines did not build");
    }
  }
  // A block-sparse kernel that did not validate must not fall back to
  // the DENSE one: dense attention over a windowed model is not a
  // slower answer, it is a different one, and the linear branch beside
  // it would then double count. Fall to sdpa_spans_f16 instead, which
  // is the same mask by another route.
  const bool spans_ok = steel_spans && _fn_attn_spans.valid();
  if (vdn_linear && !spans_ok) { use_steel = false; }
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
         t_oproj = 0, t_ff = 0, t_elt = 0, t_final = 0, t_vdn = 0, t_vdn_read = 0, t_sgate = 0;
  // A failure INSIDE the block lambda, which returns void. Checked after
  // every call: a branch that could not encode has left a partial run of
  // dispatches, and running the stack on top of it would produce a
  // plausible video with one block's attention missing.
  std::string vdn_err;
  std::string sol_err;

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

  // perf-flops: what an aborted forward added and never flushed is
  // dropped.
  _flops.begin();
  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto psplit = [&](double& acc) {
      if (!prof) { return; }
      enc.end();
      _flops.flush(stream);
      stream.commit().wait();
      acc += std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - mark).count();
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
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
      _flops.flush(stream);
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
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
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
      _flops.flush(stream);
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
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
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
    // `layer` is the block index, because Sol leaves the leading
    // `dense_layers` alone: the first blocks are where the residual
    // stream is least redundant, and the published profile keeps one of
    // 48 exact for that reason.
    auto attn = [&](int rows, bool main, int layer) {
      // perf-flops: every arm below counts the (query, key) pairs it
      // attends -- the window's share of them on the windowed arms.
      const long long dense = (long long)rows * rows;
      auto windowed = [&]() {
        return _vdn_kb_dense > 0
                   ? (long long)((double)dense * (double)_vdn_kb_sparse /
                                 (double)_vdn_kb_dense)
                   : dense;
      };
      // TaoMate-H3's split, on the main blocks. The refiner below is text
      // alone either way.
      if (sattn && main) {
        // Text over the text, the media over the cache and the whole
        // sequence.
        _flops.attention(NH,
                         (long long)n_text * n_text +
                             (long long)s_media * (s_rows + seq),
                         HD);
        skv->encode_gather(enc, _fn_copy_rect, layer, s.kh, s.vh, rows);
        const std::size_t kv_text = (std::size_t)s_rows * HD * 2;
        const std::size_t q_media = (std::size_t)n_text * HD * 2;
        // Text rows over the text keys alone: staging rows past the cache.
        enc.set_function(*s_fn_text);
        enc.set_buffer(0, s.qh);
        enc.set_buffer(1, skv->staging_k(), kv_text);
        enc.set_buffer(2, skv->staging_v(), kv_text);
        enc.set_buffer(3, s.oh);
        enc.set_buffer(4, _sattn_p_text);
        enc.dispatch({32 * (unsigned)((n_text + A_BQ - 1) / A_BQ),
                      4 * (unsigned)NH, 1}, {32, 4, 1});
        // Media rows over [cache ; text ; media]: the whole staging.
        enc.set_function(*s_fn_media);
        enc.set_buffer(0, s.qh, q_media);
        enc.set_buffer(1, skv->staging_k());
        enc.set_buffer(2, skv->staging_v());
        enc.set_buffer(3, s.oh, q_media);
        enc.set_buffer(4, _sattn_p_media);
        enc.dispatch({32 * (unsigned)((s_media + A_BQ - 1) / A_BQ),
                      4 * (unsigned)NH, 1}, {32, 4, 1});
        // The commit reads the OLD cache out of staging, so it can only
        // come after both attentions have read it -- which the encoder's
        // order guarantees.
        if (stream_commit) {
          skv->encode_commit(enc, _fn_copy_rect, layer, splan, rows);
        }
        return;
      }
      // `main` only: the other two blocks are H3's TOKEN REFINER, which
      // attends over the prompt alone. There is no video sequence there
      // to be sparse in, and the published profile likewise leaves
      // everything but video self-attention dense.
      if (sol_on && main && layer >= c.sol.dense_layers) {
        // The SINK is the layout's, not the config's: everything below
        // video_start is prompt and soundtrack, and those rows have to
        // be read exactly. Rounded outward to whole blocks inside the
        // kernel, so a ragged boundary keeps one extra block exact
        // rather than half a block approximate.
        sol::Config sc = _sol_fwd;
        sc.sink_start  = 0;
        sc.sink_tokens = main ? L.video_start : 0;
        std::string serr;
        // SAGE ON SOL'S EXACT HALF. The two are orthogonal -- Sol picks
        // which key blocks are attended, Sage how the ones that are get
        // multiplied -- and before this they simply did not meet: the
        // branch returns here, so a graph naming both ran Sage on the
        // ONE block Sol leaves dense and nothing said so.
        //
        // The prologue goes into the same encoder ahead of Sol's own
        // passes, over the head-major q/k Sol is about to read. Its
        // blocks are the EXACT half's, asked of Sol rather than assumed:
        // the scale arrays are one entry per that kernel's tiles.
        const bool sage_here =
            sage_on && _sage && layer >= c.sage.dense_layers &&
            _sol->uses_matrix_cores();
        bool sage_ok = false;
        if (sage_here) {
          const MetalSageAttention::Operand qo{&s.qh, 0, HD, rows * HD};
          const MetalSageAttention::Operand ko{&s.kh, 0, HD, rows * HD};
          const MetalSageAttention::Operand vo{&s.vh, 0, HD, rows * HD};
          std::string gerr;
          // MetalSolAttention::compose, not c.sage: the key smoothing
          // is exact under ONE softmax and Sol's row is split across
          // two, so composing has to quantize K raw. See that function.
          sage_ok = _sage->prepare(enc, qo, ko, NH, NH, rows, rows, HD,
                                   _sol->query_block(),
                                   _sol->key_block_unit(),
                                   MetalSolAttention::compose(sage8),
                                   &gerr, &vo);
          if (!sage_ok && sol_err.empty()) {
            sol_err = "sage_attn: " + gerr;
          }
        }
        _sol->set_sage(sage_ok ? _sage.get() : nullptr);
        if (!_sol->encode(enc, s.qh, s.kh, s.vh, s.oh, NH, rows, HD, scale,
                          sc, &serr)) {
          sol_err = "sol_attn: " + serr;
        } else {
          _flops.add(_sol_flops.routed(attention_flops(NH, dense, HD),
                                       _sol->total_blocks()));
        }
        return;
      }
      // The hybrid's softmax half: exact attention, but (video, video)
      // pairs restricted to the query frame's chunk window. Everything
      // touching a global row -- the prompt, the soundtrack -- stays
      // dense in both directions, which is what the spans' group 0 is.
      // The refiner blocks are not video and keep full attention.
      // The block-sparse steel kernel: the same MMA flash attention the
      // dense path runs, walking only the key blocks this query block's
      // window admits. `has_spans` is a function constant, so the dense
      // instantiations are unchanged -- the specialisation is resolved
      // before codegen.
      if (spans_ok && main) {
        enc.set_function(_fn_attn_spans);
        if (fused_qkv) {
          enc.set_buffer(0, s.qkv, (std::size_t)Q_OFF * 2);
          enc.set_buffer(1, s.qkv, (std::size_t)K_OFF * 2);
          enc.set_buffer(2, s.qkv, (std::size_t)V_OFF * 2);
        } else {
          enc.set_buffer(0, s.qh); enc.set_buffer(1, s.kh);
          enc.set_buffer(2, s.vh);
        }
        enc.set_buffer(3, fused_out ? s.ob : s.oh);
        enc.set_buffer(4, _attn_p_main);
        enc.set_buffer(8, _vdn_qb_off);
        enc.set_buffer(9, _vdn_qb_blocks);
        enc.set_buffer(10, _vdn_sp_params);
        enc.set_buffer(11, _vdn_sp_bounds);
        enc.dispatch({32 * (unsigned)((rows + A_BQ - 1) / A_BQ),
                      4 * (unsigned)NH, 1}, {32, 4, 1});
        _flops.attention(NH, windowed(), HD);
        return;
      }
      if (vdn_linear && main) {
        enc.set_function(_fn_sdpa_spans);
        enc.set_buffer(0, s.qh); enc.set_buffer(1, s.kh);
        enc.set_buffer(2, s.vh); enc.set_buffer(3, s.oh);
        enc.set_constant(4, scale); enc.set_constant(5, rows);
        enc.set_constant(6, HD); enc.set_constant(7, NH);
        enc.set_constant(8, NH); enc.set_constant(9, rows);
        enc.set_constant(10, rows);
        enc.set_buffer(11, _vdn_span_off);
        enc.set_buffer(12, _vdn_span_lo);
        enc.set_buffer(13, _vdn_span_hi);
        enc.set_constant(14, L.video_start);
        enc.set_constant(15, _vdn_tpf);
        enc.set_constant(16, _vdn_frames);
        enc.dispatch({32, (unsigned)NH, (unsigned)rows}, {32, 1, 1});
        _flops.attention(NH, windowed(), HD);
        return;
      }
      if (use_steel) {
        // SAGE: the int8 QK, on the MAIN blocks past `dense_layers`.
        //
        // Not the refiner's text attention -- that is a prompt, not a
        // video sequence, and it is short enough that the prologue would
        // cost more than the product it accelerates.
        //
        // The prologue goes into the SAME encoder, immediately before
        // the dispatch that reads it: the encoder orders them, so there
        // is no barrier to get wrong. It reads q and k exactly where the
        // attention will -- the fused projection in place, or the
        // head-major transposes -- so the two cannot disagree about
        // which head is which.
        const bool i8 = sage_on && main && _fn_attn_main_i8.valid() &&
                        layer >= c.sage.dense_layers;
        bool i8_ok = false;
        if (i8) {
          MetalSageAttention::Operand qo, ko, vo;
          if (fused_qkv) {
            qo = {&s.qkv, (std::size_t)Q_OFF, 3 * I, QKV_HSTRIDE};
            ko = {&s.qkv, (std::size_t)K_OFF, 3 * I, QKV_HSTRIDE};
            vo = {&s.qkv, (std::size_t)V_OFF, 3 * I, QKV_HSTRIDE};
          } else {
            qo = {&s.qh, 0, HD, rows * HD};
            ko = {&s.kh, 0, HD, rows * HD};
            vo = {&s.vh, 0, HD, rows * HD};
          }
          std::string gerr;
          // NH kv heads as well: H3 is MHA, gqa_factor 1.
          i8_ok = _sage->prepare(enc, qo, ko, NH, NH, rows, rows, HD, A_BQ,
                                 A_BK, sage8, &gerr, &vo);
          // The function was built with the int8 P*V's constants iff
          // pv8_on; a prologue that made no V would leave its buffers
          // unbound under them. Refused, never run so.
          if (i8_ok && pv8_on != _sage->pv8_ready()) {
            i8_ok = false;
            if (sol_err.empty()) {
              sol_err = "sage_attn: the int8 P*V's operands were not made";
            }
          }
          if (!i8_ok && sol_err.empty()) {
            sol_err = "sage_attn: " + gerr;
          }
        }
        // THE WIDE QUERY TILE, dense main path only: 8 simdgroups covering
        // 128 query rows per threadgroup, so the whole-K/V re-read happens
        // half as often. Its own pipeline and parameter block; everything
        // else (text rows, spans, Sage's int8 twin) stays at BQ 64.
        const bool wide_avail = wide_tile && main && !i8_ok &&
                                _attn_bq128_seq == rows &&
                                _fn_attn_main_bq128.valid() &&
                                !_attn_p_main_bq128.empty();
        // The probe runs HERE, one verdict per block, and the block's own
        // wall time comes back at the streamed commit below.
        const bool use_wide =
            plan_attn_tile_(wide_avail, _stream_blocks) == 128;
        if (use_wide) {
          enc.set_function(_fn_attn_main_bq128);
        } else {
          enc.set_function(i8_ok ? _fn_attn_main_i8
                                 : (main ? _fn_attn_main : _fn_attn_text));
        }
        if (fused_qkv) {
          enc.set_buffer(0, s.qkv, (std::size_t)Q_OFF * 2);
          enc.set_buffer(1, s.qkv, (std::size_t)K_OFF * 2);
          enc.set_buffer(2, s.qkv, (std::size_t)V_OFF * 2);
        } else {
          enc.set_buffer(0, s.qh); enc.set_buffer(1, s.kh);
          enc.set_buffer(2, s.vh);
        }
        enc.set_buffer(3, fused_out ? s.ob : s.oh);
        enc.set_buffer(4, use_wide ? _attn_p_main_bq128
                                   : (main ? _attn_p_main : _attn_p_text));
        if (i8_ok) { _sage->bind(enc); }
        const unsigned bq = use_wide ? 128u : (unsigned)A_BQ;
        const unsigned wm = use_wide ? 8u : 4u;
        // Sage's int8 P*V on the video rows only (_fn_attn_main_i8_qk):
        // the prefix's blocks first on the int8 QK, then the video's
        // from their first block -- Q and O bound from its row, K and V
        // whole.
        const int nq_all = (rows + A_BQ - 1) / A_BQ;
        if (i8_ok && !use_wide && pv8_on && _sage->pv8_ready() &&
            _attn_band_nb > 0 && _attn_band_nb < nq_all &&
            _fn_attn_main_i8_qk.valid()) {
          const int nb = _attn_band_nb;
          const std::size_t r0 = (std::size_t)nb * A_BQ;
          enc.set_function(_fn_attn_main_i8_qk);
          enc.dispatch({32u * (unsigned)nb, 4u * (unsigned)NH, 1},
                       {32, 4, 1});
          enc.set_function(_fn_attn_main_i8);
          const std::size_t qrow = fused_qkv ? (std::size_t)3 * I
                                             : (std::size_t)HD;
          const std::size_t orow = fused_out ? (std::size_t)I
                                             : (std::size_t)HD;
          if (fused_qkv) {
            enc.set_buffer(0, s.qkv, ((std::size_t)Q_OFF + r0 * qrow) * 2);
          } else {
            enc.set_buffer(0, s.qh, r0 * qrow * 2);
          }
          enc.set_buffer(3, fused_out ? s.ob : s.oh, r0 * orow * 2);
          enc.set_buffer(4, _attn_p_band);
          // pv8_ready() above is what makes this bind; a false here
          // would leave row 0's binding on the video rows' dispatch.
          if (!_sage->bind(enc, (int)r0)) {
            if (sol_err.empty()) {
              sol_err = "sage_attn: the video rows' operands did not bind";
            }
            return;
          }
          enc.dispatch({32u * (unsigned)(nq_all - nb), 4u * (unsigned)NH,
                        1}, {32, 4, 1});
          _flops.attention(NH, dense, HD);
          return;
        }
        enc.dispatch({32 * (unsigned)(((unsigned)rows + bq - 1) / bq),
                      wm * (unsigned)NH, 1}, {32, wm, 1});
        _flops.attention(NH, dense, HD);
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
      _flops.attention(NH, dense, HD);
    };

    // One block. `modulated` selects the 50 main blocks (per-row AdaLN
    // and rope) from the 2 refiner blocks (neither).
    // `lora_layer` indexes the adapters' per-block factors -- the same
    // stack `modulated` selects, since the refiner blocks are exactly the
    // unmodulated ones.
    // The ANE feed-forward tier, armed once from the first forward's row
    // count. `ane_on_commit` is how a streamed block lets the split issue
    // its successor's read early; see the ANE split in the block.
    const bool ane_on = _cfg.ane_ffn && ane_setup_(seq);
    const bool qkv_on =
        (_cfg.ane_qkv ||
         (!_ane_disabled && std::getenv("VPIPE_H3_ANE_QKV") != nullptr)) &&
        ane_qkv_setup_(seq);
    std::function<void()> ane_on_commit;
    // The GPU's feed-forward over rows [r0, r0 + n) of s.nm, IN PLACE: its
    // share of a split, a band of the banded block (from row 0 of its
    // band-sized s.nm), and the rows the ANE lost -- which still hold
    // their input, because the ANE writes a chunk back only once all of
    // it came out finite. The scratch is used from its first row.
    auto ff_gpu = [&](const Block& b, const LoraStack& lo_fc1,
                      const LoraStack& lo_fc2, int r0, int n) {
      if (n <= 0) { return; }
      const std::size_t e0 = (std::size_t)r0 * H;
      const bool head = r0 == 0;          // the dumps describe the head
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
        enc.set_buffer(2, b.fc1.qbias); enc.set_buffer(3, s.nm, e0 * 2);
        enc.set_buffer(4, s.qkv);
        enc.set_constant(5, H);
        enc.set_constant(6, 2 * FF);     // the FUSED width, not the output
        enc.set_constant(7, n);
        const int bm = bm64 ? 64 : 32;
        enc.dispatch({(unsigned)(((2 * FF + 31) / 32) * 32),
                      (unsigned)(((n + bm - 1) / bm) * 2), 2}, {32, 2, 2});
      } else {
        gemm_(enc, s.nm, e0, b.fc1, s.ff, 0, n, 2 * FF, H,
              &lo_fc1);
        if (head) {
          bdump("fc1", s.ff, 2 * FF);
          trip(trip_blk, 6, s.ff, (std::size_t)n * 2 * FF);
        }
        // fc1 is FUSED [gate | up], GATE first -- the diffusers SwiGLU
        // convention, not the llama one the rest of this tree follows.
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, s.ff); enc.set_buffer(1, s.qkv);
        enc.set_constant(2, n);
        enc.set_constant(3, FF);
        enc.dispatch({(unsigned)(n * FF), 1, 1}, {256, 1, 1});
      }
      if (head) {
        bdump("swiglu", s.qkv, FF);
        trip(trip_blk, 7, s.qkv, (std::size_t)n * FF);
      }
      gemm_(enc, s.qkv, 0, b.fc2, s.nm, e0, n, H, FF,
            &lo_fc2);
    };
    auto block = [&](const Block& b, const SharedBuffer& x, int rows,
                     bool modulated, int lora_layer) {
      const bool lref = !modulated;
      const LoraStack lo_qkv = lora_stack_(lref, lora_layer,
                                           &BlockLora::qkv);
      const LoraStack lo_out = lora_stack_(lref, lora_layer,
                                           &BlockLora::out);
      const LoraStack lo_fc1 = lora_stack_(lref, lora_layer,
                                           &BlockLora::fc1);
      const LoraStack lo_fc2 = lora_stack_(lref, lora_layer,
                                           &BlockLora::fc2);
      // Stage this block's feed-forward weights for the ANE NOW, on the ANE
      // worker, so the conversion runs under the attention encoded below;
      // the split joins it. Refiner blocks run the text rows alone.
      const bool ane_ok =
          modulated && ane_on &&
          ane_eligible_(lora_layer, b, !lo_fc1.empty() || !lo_fc2.empty());
      const AneFeedForward::Plan ane_plan =
          ane_ok ? _ane->plan_block() : AneFeedForward::Plan::kGpu;
      const bool ane_block = ane_plan == AneFeedForward::Plan::kSplit;
      const bool ane_probe = ane_plan == AneFeedForward::Plan::kProbe;
      if ((ane_block || ane_probe) &&
          (lora_layer == 0 || _ane->needs_barrier())) {
        // A measured block after unsplit work -- GPU-mode blocks, or, for
        // block 0, the refiner, projections and modulation this forward
        // queued ahead of it: drain that first, so this block's split point
        // times its own work. Without the block-0 case every step's first
        // sample ran ~25% long and flipped an M5 Pro run off the split. See
        // AneFeedForward::needs_barrier.
        enc.end();
        _flops.flush(stream);
        stream.commit().wait();
        // The same stream: a fresh one is a new queue, and its first
        // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
        // stream_per_commit_costs).
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
      }
      // The qkv tier, planned like the FF's. When qkv splits, ITS weights
      // are staged first -- they are needed before the attention -- and the
      // FF's move to just after the qkv join, where they still run under
      // the attention: the two share one ANE worker.
      const bool qkv_ok =
          modulated && qkv_on && ane_qkv_eligible_(lora_layer, b);
      const AneFeedForward::Plan qkv_plan =
          qkv_ok ? _ane_qkv->plan_block() : AneFeedForward::Plan::kGpu;
      const bool qkv_block = qkv_plan == AneFeedForward::Plan::kSplit;
      const bool qkv_probe = qkv_plan == AneFeedForward::Plan::kProbe;
      if ((qkv_block || qkv_probe) &&
          (lora_layer == 0 || _ane_qkv->needs_barrier())) {
        enc.end();
        _flops.flush(stream);
        stream.commit().wait();
        // The same stream: a fresh one is a new queue, and its first
        // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
        // stream_per_commit_costs).
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
      }
      if (qkv_block) {
        ane_qkv_stage_(lora_layer, b, lo_qkv);
      } else if (ane_block) {
        ane_stage_(lora_layer, b, lo_fc1, lo_fc2);
      }
      rms(x, 0, b.n1, s.nm, 0, rows, H, c.norm_eps);
      bdump("n1", s.nm, H);
      trip(trip_blk, 0, s.nm, (std::size_t)rows * H);
      if (modulated) {
        modulate(s.nm, *mod_buf, mod_off * 2, s.adaln_idx, s.nm, rows,
                 6 * H, H, 0);
      }
      bdump("n1+mod", s.nm, H);
      psplit(t_elt);
      // ---- the qkv split: the ANE's rows of s.nm into the tail of s.qkv,
      // the GPU's head rows beside them, joined before anything reads it.
      int q_rows = 0;
      double q_drain_ms = 0.0;
      if (qkv_block || qkv_probe) {
        const auto t_q0 = std::chrono::steady_clock::now();
        enc.end();
        _flops.flush(stream);
        metal_compute::CommandStream::Fence qfence = stream.commit();
        const bool staged = qkv_block && _ane_qkv->join_stage(lora_layer);
        qfence.wait();
        q_drain_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_q0).count();
        if (staged) {
          q_rows = _ane_qkv->begin(s.nm, s.qkv, rows);
        } else if (qkv_block && _mc->session() != nullptr) {
          _mc->session()->warn(fmt(
              "minimax-h3: staging block {}'s qkv for the ANE failed; it "
              "keeps the GPU", lora_layer));
        }
        // The same stream: a fresh one is a new queue, and its first
        // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
        // stream_per_commit_costs).
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
      }
      const auto t_qg0 = std::chrono::steady_clock::now();
      gemm_(enc, s.nm, 0, b.qkv, s.qkv, 0, rows - q_rows, 3 * I, H,
            &lo_qkv);
      if (q_rows > 0 || qkv_probe) {
        enc.end();
        _flops.flush(stream);
        stream.commit().wait();
        const double qg_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_qg0).count();
        int q_lost = -1;
        if (q_rows > 0) {
          if (!_ane_qkv->finish(lora_layer, rows, qg_ms, q_drain_ms)) {
            q_lost = _ane_qkv->lost_row0();
          }
        } else {
          _ane_qkv->note_probe(q_drain_ms, qg_ms);
        }
        // The same stream: a fresh one is a new queue, and its first
        // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
        // stream_per_commit_costs).
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
        // THE GPU FALLBACK: `s.nm` is only the tier's input, so the rows it
        // lost are projected again from it before anything reads s.qkv.
        if (q_lost >= 0 && q_lost < rows) {
          const std::size_t r0 = (std::size_t)q_lost;
          gemm_(enc, s.nm, r0 * H, b.qkv, s.qkv, r0 * 3 * I, rows - q_lost,
                3 * I, H, &lo_qkv);
        }
      }
      // The FF's staging, deferred from the block start when qkv split.
      if (qkv_block && ane_block) {
        ane_stage_(lora_layer, b, lo_fc1, lo_fc2);
      }
      bdump("qkv", s.qkv, 3 * I);
      trip(trip_blk, 2, s.qkv, (std::size_t)rows * 3 * I);
      psplit(t_qkv);
      // THE GATE'S WEIGHTS LIVE IN THE BRANCH'S BLOCK, so the block is
      // brought in whenever the hybrid is on at all -- not only when the
      // linear half runs. Under full cover this loads a block to use two
      // small tensors from it, which is the cheap side of a decision
      // whose expensive side would be a second accessor for the gate
      // alone; full cover only happens on clips of at most
      // (radius + 1) * chunk frames, which is not the geometry anyone
      // pays this for.
      if (modulated && vdn_on) {
        std::string verr;
        if (!_vdn->ensure_block(lora_layer, &verr)) {
          vdn_err = "VDN block " + std::to_string(lora_layer) + ": " + verr;
          return;
        }
        // Its own bucket: on the streamed path this is 89 MB of DISK per
        // block, which is not the branch's arithmetic and must not be
        // reported as it. Nothing is encoded here, so under the profile
        // the split measures the read alone.
        psplit(t_vdn_read);
      }
      // ---- VDN: the linear branch, on the RAW projection -------------
      //
      // HERE AND NOT LATER, because qk_norm and rope below rewrite
      // s.qkv IN PLACE and the branch shares this model's q/k/v. The
      // alternative -- snapshotting the three -- is 8.67 GB a block at
      // production geometry; running the whole branch first costs
      // nothing, since its readout is not wanted until after the out
      // projection anyway.
      //
      // Nothing is copied out: the strides below hand the branch H3's
      // own fused, per-head-grouped bf16 buffer where it lies, and the
      // video rows are addressed as the run they are inside the packed
      // sequence. See MetalVdnBranch::Inputs.
      if (modulated && vdn_linear) {
        minimax_h3::MetalVdnBranch::Inputs vi;
        vi.x = &s.nm;
        vi.q_raw = vi.k_raw = vi.v_raw = &s.qkv;
        vi.text_x = &s.nm;
        vi.text_k = vi.text_v = &s.qkv;
        vi.qkv_row_stride  = 3 * I;
        vi.qkv_head_stride = QKV_HSTRIDE;
        vi.qkv_bf16 = true;
        vi.x_bf16   = true;
        vi.out_bf16 = true;
        const std::size_t v0 = (std::size_t)L.video_start * 3 * I;
        vi.q_off = (v0 + (std::size_t)Q_OFF) * 2;
        vi.k_off = (v0 + (std::size_t)K_OFF) * 2;
        vi.v_off = (v0 + (std::size_t)V_OFF) * 2;
        // The prompt is rows [0, n_text), so its own offsets are the
        // field offsets alone.
        vi.text_x_off = 0;
        vi.text_k_off = (std::size_t)K_OFF * 2;
        vi.text_v_off = (std::size_t)V_OFF * 2;
        vi.x_off = (std::size_t)L.video_start * H * 2;
        minimax_h3::MetalVdnBranch::Geometry vg;
        vg.frames   = _vdn_frames;
        vg.grid_h   = in.video_grid_h;
        vg.grid_w   = in.video_grid_w;
        vg.text_len = n_text;
        std::string verr;
        if (!_vdn->encode(enc, lora_layer, vg, _vdn_bounds, vi, _vdn_ro,
                          &verr)) {
          vdn_err = "VDN block " + std::to_string(lora_layer) + ": " + verr;
          return;
        }
        ++_vdn_blocks_run;
        psplit(t_vdn);
      }
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
      attn(rows, modulated, lora_layer);
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
      // The softmax half's per-head gate. It scales a distribution that
      // renormalised to 1 over however little of the sequence it saw,
      // back toward the share it actually captured -- so it belongs
      // BEFORE the out projection, and it reads the same x the branch
      // did, which s.nm still holds until the GEMM below overwrites it.
      if (modulated && vdn_on) {
        std::string verr;
        if (!_vdn->encode_softmax_gate(enc, lora_layer, rows, s.nm, 0, true,
                                       H, s.ob, 0, &verr)) {
          vdn_err = "VDN gate " + std::to_string(lora_layer) + ": " + verr;
          return;
        }
        // Its OWN bucket. Without this the gate is charged to the out
        // projection below -- the next psplit -- which is why VDN's
        // o-proj read higher than the baseline's and looked like the
        // second adapter alone.
        psplit(t_sgate);
      }
      bdump("attn_out", s.ob, I);
      trip(trip_blk, 3, s.ob, (std::size_t)rows * I);
      gemm_(enc, s.ob, 0, b.out, s.nm, 0, rows, H, I,
            &lo_out);
      // DRAIN THE OUT PROJECTION HERE, before the branch's own
      // projection is encoded. Under VPIPE_H3_DIT_PROFILE a psplit
      // charges everything since the last one, so the VDN block below
      // was absorbing this GEMM and reporting it as branch cost --
      // which showed as an o-proj at 9.7 PFLOP/s, a number no kernel
      // can reach and the only reason the mistake was visible.
      // Inert without the profile.
      psplit(t_oproj);
      // ---- VDN: the linear readout into the residual stream ----------
      //
      // out = orig.to_out(gate * window_softmax) + to_out_linear(readout)
      // on VIDEO rows, which is the line the whole component exists to
      // add. The projection is the BRANCH's weight run by THIS model's
      // GEMM -- that is where the quantisation, the LoRA stack and the
      // steel routing live -- and the add is masked to the video run
      // because the window covers the global rows exactly, so anything
      // the branch put there would be counted twice.
      if (modulated && vdn_linear) {
        const metal_compute::SharedBuffer* wl =
            _vdn->out_linear(lora_layer);
        if (wl == nullptr || wl->empty()) {
          vdn_err = "VDN block " + std::to_string(lora_layer)
                    + ": no output projection";
          return;
        }
        Linear vl;
        vl.w = wl->subview(0, wl->byte_size());
        gemm_(enc, _vdn_ro, 0, vl, _vdn_proj, 0, L.num_video_rows, H, I,
              nullptr);
        const std::size_t vo = (std::size_t)L.video_start * H * 2;
        enc.set_function(_fn_residual);
        enc.set_buffer(0, s.nm, vo);
        enc.set_buffer(1, _vdn_proj);
        enc.set_buffer(2, s.nm, vo);
        enc.set_constant(3, L.num_video_rows * H);
        enc.dispatch({(unsigned)(L.num_video_rows * H), 1, 1}, {256, 1, 1});
        psplit(t_vdn);
      }
      bdump("o_proj", s.nm, H);
      trip(trip_blk, 4, s.nm, (std::size_t)rows * H);
      psplit(t_oproj);   // the dumps and tripwires only, by now
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
      // ---- the ANE split ----------------------------------------------
      //
      // The ANE takes the TAIL rows and the GPU the HEAD: every kernel
      // below reads from row 0, and rows are independent in a feed-forward,
      // so the split is exact. s.nm is the input and the output of both
      // halves, on disjoint rows.
      //
      // COMMIT the attention WITHOUT waiting, and only then join the
      // staging job: a resident stack otherwise commits only at the end of
      // the forward, so joining first would overlap nothing but the encode.
      // The ANE starts once s.nm is readable and BEFORE the GPU's half is
      // encoded, so it hides behind that whole half -- MEASURED on Krea-2,
      // encoding the GPU's gate/up first left only 48-62% of it hidden.
      int a_rows = 0;
      double ane_drain_ms = 0.0;
      if (ane_block || ane_probe) {
        const auto t_d0 = std::chrono::steady_clock::now();
        enc.end();
        _flops.flush(stream);
        metal_compute::CommandStream::Fence afence = stream.commit();
        // A streamed block issues its successor's read here, under this
        // block's attention and both halves of its feed-forward -- at the
        // block's end it would hide behind little more than the gated add.
        // Safe for the reason that one is: it lands in the OTHER slot.
        if (ane_on_commit) { ane_on_commit(); }
        const bool staged = ane_block && _ane->join_stage(lora_layer);
        afence.wait();
        ane_drain_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_d0).count();
        if (staged) {
          a_rows = _ane->begin(s.nm, s.nm, rows);
        } else if (ane_block && _mc->session() != nullptr) {
          _mc->session()->warn(fmt(
              "minimax-h3: staging block {}'s weights for the ANE failed; "
              "it keeps the GPU feed-forward", lora_layer));
        }
        // The same stream: a fresh one is a new queue, and its first
        // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
        // stream_per_commit_costs).
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
      }
      // The block feed-forward over rows [r0, r0 + n) (ff_gpu).
      auto gpu_ff = [&](int r0, int n) {
        ff_gpu(b, lo_fc1, lo_fc2, r0, n);
      };
      // The ANE always leaves the GPU at least one row.
      const int g_rows = rows - a_rows;
      const auto t_g0 = std::chrono::steady_clock::now();
      gpu_ff(0, g_rows);
      if (a_rows > 0) {
        // The GPU's half, drained, then the ANE joined behind it.
        enc.end();
        _flops.flush(stream);
        stream.commit().wait();
        const double t_gpu =
            _ane->timing()
                ? std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t_g0).count()
                : 0.0;
        const bool ane_ok =
            _ane->finish(lora_layer, rows, t_gpu, ane_drain_ms);
        // The same stream: a fresh one is a new queue, and its first
        // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
        // stream_per_commit_costs).
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
        // THE GPU FALLBACK, in place over the rows the ANE lost.
        const int lost = ane_ok ? -1 : _ane->lost_row0();
        if (lost >= 0 && lost < rows) { gpu_ff(lost, rows - lost); }
      } else if (ane_probe) {
        // The GPU over every row, drained, for the on/off measurement.
        enc.end();
        _flops.flush(stream);
        stream.commit().wait();
        _ane->note_probe(ane_drain_ms,
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t_g0).count());
        // The same stream: a fresh one is a new queue, and its first
        // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
        // stream_per_commit_costs).
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
      }
      bdump("fc2", s.nm, H);
      trip(trip_blk, 8, s.nm, (std::size_t)rows * H);
      psplit(t_ff);
      if (modulated) { gated(x, s.nm, rows, 5 * H); }
      else           { residual(x, s.nm, rows); }
      bdump("x_after_ff", x, H);
      trip(trip_blk, 9, x, (std::size_t)rows * H);
      psplit(t_elt);
    };

    // THE BANDED BLOCK (band_rows: a long sequence's main blocks). Pass 1
    // projects each band and writes its q / k / v into the whole-sequence
    // head-major windows; pass 2 attends each band's queries over every
    // key, projects out, adds, and runs the feed-forward -- band by band,
    // in the band-sized nm / qkv / ob | ff. Every row's arithmetic is
    // block()'s, in block()'s order.
    auto modulate_band = [&](int r0, int n, int scale_off, int shift_off) {
      enc.set_function(_fn_modulate);
      enc.set_buffer(0, s.nm); enc.set_buffer(1, *mod_buf, mod_off * 2);
      enc.set_buffer(2, s.adaln_idx, (std::size_t)r0 * sizeof(int));
      enc.set_buffer(3, s.nm);
      enc.set_constant(4, H);
      enc.set_constant(5, 6 * H);
      enc.set_constant(6, scale_off);
      enc.set_constant(7, shift_off);
      enc.set_constant(8, n * H);
      enc.dispatch({(unsigned)(n * H), 1, 1}, {256, 1, 1});
    };
    auto gated_band = [&](int r0, int n, int gate_off) {
      enc.set_function(_fn_gated);
      enc.set_buffer(0, s.x, (std::size_t)r0 * H * 2);
      enc.set_buffer(1, *mod_buf, mod_off * 2);
      enc.set_buffer(2, s.adaln_idx, (std::size_t)r0 * sizeof(int));
      enc.set_buffer(3, s.nm);
      enc.set_constant(4, H);
      enc.set_constant(5, 6 * H);
      enc.set_constant(6, gate_off);
      enc.set_constant(7, n * H);
      enc.dispatch({(unsigned)(n * H), 1, 1}, {256, 1, 1});
    };
    const std::size_t rope_row =
        (std::size_t)(3 * c.rope_freq_dim) * sizeof(float);
    auto trope_band = [&](const SharedBuffer& dst, int r0, int n, int off,
                          int rot) {
      enc.set_function(_fn_trope_band);
      enc.set_buffer(0, s.qkv); enc.set_buffer(1, dst);
      enc.set_buffer(2, s.rcos, (std::size_t)r0 * rope_row);
      enc.set_buffer(3, s.rsin, (std::size_t)r0 * rope_row);
      enc.set_constant(4, NH);
      enc.set_constant(5, n);
      enc.set_constant(6, HD);
      enc.set_constant(7, rot);
      enc.set_constant(8, 3 * I);
      enc.set_constant(9, off);
      enc.set_constant(10, QKV_HSTRIDE);
      enc.set_constant(11, seq);
      enc.set_constant(12, r0);
      enc.dispatch({(unsigned)HD, (unsigned)n, (unsigned)NH},
                   {(unsigned)HD, 1, 1});
    };
    auto block_banded = [&](const Block& b, int layer) {
      const LoraStack lo_qkv = lora_stack_(false, layer, &BlockLora::qkv);
      const LoraStack lo_out = lora_stack_(false, layer, &BlockLora::out);
      const LoraStack lo_fc1 = lora_stack_(false, layer, &BlockLora::fc1);
      const LoraStack lo_fc2 = lora_stack_(false, layer, &BlockLora::fc2);
      const int rot = c.rope_rot();
      for (int r0 = 0; r0 < seq; r0 += band) {
        const int n = std::min(band, seq - r0);
        rms(s.x, (std::size_t)r0 * H, b.n1, s.nm, 0, n, H, c.norm_eps);
        modulate_band(r0, n, H, 0);
        psplit(t_elt);
        gemm_(enc, s.nm, 0, b.qkv, s.qkv, 0, n, 3 * I, H, &lo_qkv);
        psplit(t_qkv);
        qk_norm(b.qn, n, Q_OFF);
        qk_norm(b.kn, n, K_OFF);
        trope_band(s.qh, r0, n, Q_OFF, rot);
        trope_band(s.kh, r0, n, K_OFF, rot);
        trope_band(s.vh, r0, n, V_OFF, 0);
        psplit(t_prep);
      }
      // The layer's attention, chosen as block() chooses it: Sol on its
      // routed layers (Sage on its exact half where both are on), Sage's
      // int8 kernels past its dense layers, else the dense kernel. Sage's
      // keys -- and V, under the int8 P*V -- once for the layer, the
      // band's queries before each band.
      const bool sol_here = sol_on && layer >= c.sol.dense_layers;
      const bool sage_here =
          sage_on && _sage && layer >= c.sage.dense_layers &&
          (sol_here ? _sol->uses_matrix_cores()
                    : _fn_attn_bandv[0][pv8_on ? 3 : 1].valid());
      bool keys_ok = false;
      if (sage_here) {
        const MetalSageAttention::Operand ko{&s.kh, 0, HD, seq * HD};
        const MetalSageAttention::Operand vo{&s.vh, 0, HD, seq * HD};
        std::string gerr;
        keys_ok = _sage->prepare_keys(
            enc, ko, NH, NH, band, seq, HD,
            sol_here ? _sol->query_block() : A_BQ,
            sol_here ? _sol->key_block_unit() : A_BK,
            sol_here ? MetalSolAttention::compose(sage8) : sage8, &gerr,
            &vo);
        if (!keys_ok && sol_err.empty()) { sol_err = "sage_attn: " + gerr; }
      }
      // ...and Sol's: its key summaries, their statistics and the exact
      // threshold's covariance once for the layer -- every band attends
      // the same keys -- read by each band's call through Band::keys
      // (both Sol objects: the last band's reads the first's set).
      bool sol_keys = false;
      if (sol_here) {
        std::string kerr;
        sol_keys = _sol->prepare_keys(enc, s.kh, s.vh, NH, seq, seq, HD,
                                      _sol_fwd, &kerr);
        if (!sol_keys && sol_err.empty()) { sol_err = "sol_attn: " + kerr; }
      }
      for (int r0 = 0; r0 < seq; r0 += band) {
        const int n = std::min(band, seq - r0);
        const bool full = n == band;
        const int fi = full ? 0 : 1;
        const int nqb = (n + A_BQ - 1) / A_BQ;
        bool sage_ok = false;
        if (keys_ok) {
          const MetalSageAttention::Operand qo{
              &s.qh, (std::size_t)r0 * HD, HD, seq * HD};
          std::string gerr;
          sage_ok = _sage->prepare_queries(enc, qo, n, &gerr);
          if (sage_ok && pv8_on != _sage->pv8_ready()) {
            sage_ok = false;
            gerr = "the int8 P*V's operands were not made";
          }
          if (!sage_ok && sol_err.empty()) {
            sol_err = "sage_attn: " + gerr;
          }
        }
        if (sol_here) {
          // Sol reads and writes a band's rows in ONE layout, so the
          // band's queries are gathered into their own [H, n, D].
          MetalSolAttention* so = full ? _sol.get() : _sol_tail.get();
          enc.set_function(_fn_copy_rect);
          enc.set_buffer(0, s.qh, (std::size_t)r0 * HD * 2);
          enc.set_buffer(1, s.qb);
          enc.set_constant(2, 0);
          enc.set_constant(3, 0);
          enc.set_constant(4, NH);
          enc.set_constant(5, n * HD);
          enc.set_constant(6, seq * HD);
          enc.set_constant(7, n * HD);
          enc.dispatch({(unsigned)(NH * n * HD), 1, 1}, {256, 1, 1});
          sol::Config sc = _sol_fwd;
          sc.sink_start  = 0;
          sc.sink_tokens = L.video_start;
          so->set_sage(sage_ok ? _sage.get() : nullptr);
          MetalSolAttention::Band bd;
          bd.q_tokens = n;
          bd.k_tokens = seq;
          bd.q_rows   = n;
          bd.k_rows   = seq;
          bd.q_off    = r0;
          bd.q_row0   = 0;
          bd.keys     = sol_keys ? _sol.get() : nullptr;
          std::string serr;
          if (!so->encode(enc, s.qb, s.kh, s.vh, s.oh, NH, NH, bd, HD, scale,
                          sc, &serr)) {
            if (sol_err.empty()) { sol_err = "sol_attn: " + serr; }
          } else {
            _flops.add(_sol_flops.routed(
                attention_flops(NH, (long long)n * seq, HD),
                so->total_blocks()));
            _sol_routed_total += so->total_blocks();
          }
          enc.set_function(_fn_transpose);
          enc.set_buffer(0, s.oh); enc.set_buffer(1, s.ob);
          enc.set_constant(2, NH); enc.set_constant(3, n);
          enc.set_constant(4, HD);
          enc.dispatch({(unsigned)HD, (unsigned)n, (unsigned)NH},
                       {(unsigned)HD, 1, 1});
        } else if (sage_ok && r0 == 0 && _attn_bandq_nb > 0 &&
                   _attn_bandq_nb < nqb && _fn_attn_bandv[0][2].valid() &&
                   !_attn_p_bandq_rest.empty()) {
          // Band 0 under the int8 P*V: its prompt and soundtrack rows on
          // the int8 QK alone, as the whole sequence's are.
          const int nb = _attn_bandq_nb;
          const std::size_t qr = (std::size_t)nb * A_BQ;
          enc.set_function(_fn_attn_bandv[0][2]);
          enc.set_buffer(0, s.qh);
          enc.set_buffer(1, s.kh);
          enc.set_buffer(2, s.vh);
          enc.set_buffer(3, s.ob);
          enc.set_buffer(4, _attn_p_bandq);
          _sage->bind(enc);
          enc.dispatch({32u * (unsigned)nb, 4u * (unsigned)NH, 1},
                       {32, 4, 1});
          enc.set_function(_fn_attn_bandv[0][3]);
          enc.set_buffer(0, s.qh, qr * HD * 2);
          enc.set_buffer(1, s.kh);
          enc.set_buffer(2, s.vh);
          enc.set_buffer(3, s.ob, qr * I * 2);
          enc.set_buffer(4, _attn_p_bandq_rest);
          if (!_sage->bind(enc, (int)qr)) {
            if (sol_err.empty()) {
              sol_err = "sage_attn: the video rows' operands did not bind";
            }
          } else {
            enc.dispatch({32u * (unsigned)(nqb - nb), 4u * (unsigned)NH, 1},
                         {32, 4, 1});
          }
          _flops.attention(NH, (long long)n * seq, HD);
        } else {
          enc.set_function(
              _fn_attn_bandv[fi][sage_ok ? (pv8_on ? 3 : 1) : 0]);
          enc.set_buffer(0, s.qh, (std::size_t)r0 * HD * 2);
          enc.set_buffer(1, s.kh);
          enc.set_buffer(2, s.vh);
          enc.set_buffer(3, s.ob);
          enc.set_buffer(4, full ? _attn_p_bandq : _attn_p_bandq_tail);
          if (sage_ok) { _sage->bind(enc); }
          enc.dispatch({32 * (unsigned)nqb, 4 * (unsigned)NH, 1},
                       {32, 4, 1});
          _flops.attention(NH, (long long)n * seq, HD);
        }
        psplit(t_attn);
        gemm_(enc, s.ob, 0, b.out, s.nm, 0, n, H, I, &lo_out);
        psplit(t_oproj);
        gated_band(r0, n, 2 * H);
        rms(s.x, (std::size_t)r0 * H, b.n2, s.nm, 0, n, H, c.norm_eps);
        modulate_band(r0, n, 4 * H, 3 * H);
        psplit(t_elt);
        ff_gpu(b, lo_fc1, lo_fc2, 0, n);
        psplit(t_ff);
        gated_band(r0, n, 5 * H);
        psplit(t_elt);
      }
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
      block(_refiner[ri], s.txt, n_text, false, (int)ri);
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
    // A streaming attention whose cache is ON DISK reads block L's rows
    // while the GPU runs block L - 1, so every block -- resident ones too
    // -- is committed and waited for: that wait is what frees a slot for
    // the next read and hands a clean forward's new rows to their file.
    const bool kv_io = sattn && skv->on_disk();
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
      // On the GPU's clock. A resident stack encodes all 50 blocks in
      // milliseconds and then runs for a minute, so an inline report
      // would race to 100% and sit there. See
      // shared/dit-gpu-progress.h.
      // The blocks before this one, credited when the buffer now open
      // retires -- the checkpoint the progress report rides too.
      _flops.flush(stream);
      report_block(stream, _block_progress, Lx, c.n_layers);
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
      // Declared after `streamed`, so it destroys FIRST: a block that exits
      // between staging its weights for the ANE and the split that joins
      // them must not free what the staging job is still reading.
      struct AneJoin {
        AneFeedForward* a;
        ~AneJoin() { if (a != nullptr) { a->join(); } }
      } ane_join{_ane != nullptr ? _ane.get() : _ane_qkv.get()};
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
          if (!load_block_(*_ws, blk_("blocks.", Lx, ""), _slot[0],
                           /*main=*/true, !baked, Retain::Streamed)) {
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
                                    /*main=*/true, !baked,
                                    Retain::Streamed)) {
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
      // Its adapters' factors, spilled at load, read in beside it.
      if (streaming && !fill_lora_(Lx)) { return {}; }
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
      const LoraStack ada_lo =
          lora_stack_(/*refiner=*/false, Lx, &BlockLora::adaln);
      const bool curved = curve_live_(Lx);
      if (baked && !c.pruned()) {
        const std::size_t row0 =
            (std::size_t)_adaln_row0[(std::size_t)in.schedule_index] *
            (std::size_t)c.adaln_out();
        if (ada_lo.empty() && !curved) {
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
          lora_apply_(enc, s.temb, 0, ada_lo, s.mod, 0, n_t, c.adaln_out(),
                      c.time_dim);
          if (curved) { curve_apply_(enc, Lx, s.mod, n_t); }
        }
      } else {
        mod_buf = &s.mod;
        mod_off = 0;
        if (c.pruned()) {
          // Baked, a pruned DiT's tables are the rows an adapted
          // embedding leaves off the curve, added in the projection.
          pruned_mod_(enc, _padaln[(std::size_t)Lx], Lx, s.mod, n_t,
                      c.adaln_out(),
                      baked ? &_adaln_tab[(std::size_t)Lx] : nullptr,
                      brow * (std::size_t)c.adaln_out() * 2);
        } else {
          gemm_(enc, s.temb, 0, b.adaln, s.mod, 0, n_t, c.adaln_out(),
                c.time_dim);
        }
        lora_apply_(enc, s.temb, 0, ada_lo, s.mod, 0, n_t, c.adaln_out(),
                    c.time_dim);
        if (curved) { curve_apply_(enc, Lx, s.mod, n_t); }
      }
      psplit(t_adaln);
      // The modulation VALUES, per modality tag. s.mod is [n_t*3, 6*H]
      // with row = timestep_index*3 + tag, so the three tags are three
      // slices of ONE tensor and comparing them needs no reference: if
      // tag 1's scale/shift dwarf tag 0's, the text rows' 460x blow-up
      // is coming from the numbers, not from the addressing.
      if (xprobe && Lx == 0 && _mc->session() != nullptr) {
        enc.end();
        _flops.flush(stream);
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
        // The same stream: a fresh one is a new queue, and its first
        // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
        // stream_per_commit_costs).
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
      }
      // Block Lx+1's read, issued under GPU work already in flight: at the
      // end of this block, or earlier from the ANE split, which commits the
      // attention mid-block. A no-op while a read is outstanding.
      auto issue_prefetch = [&]() {
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
                                           pf.blk, /*main=*/true, !baked,
                                           Retain::Streamed);
                });
          }
        }
      };
      ane_on_commit = streaming ? std::function<void()>(issue_prefetch)
                                : std::function<void()>();
      if (kv_io) {
        std::string kerr;
        if (!skv->acquire(Lx, &kerr)) { return fail(kerr); }
      }
      bprobe = xprobe && Lx == 0;
      trip_blk = Lx;
      if (band > 0) {
        _route_rows = seq;
        block_banded(b, Lx);
        _route_rows = 0;
      } else {
        block(b, s.x, seq, true, Lx);
      }
      ane_on_commit = nullptr;
      if (!vdn_err.empty()) { return fail(vdn_err); }
      if (!sol_err.empty()) { return fail(sol_err); }
      bprobe = false;
      if (Lx == 0 || Lx == 3 || Lx == 24 || Lx == c.n_layers - 1) {
        xdump(("after block " + std::to_string(Lx)).c_str());
      }
      // NO BARRIER HERE ANY MORE. A resident stack used to commit and
      // re-encode per block whenever a bar was attached, purely so the
      // per-block report was paced by something real. report_block()
      // above does that from a completion handler instead, which costs
      // no commit, no re-encode and no pipeline bubble -- so a watched
      // run and an unwatched one now encode identically.
      //
      // What goes with it is a GPU error being attributed to the block
      // that caused it. The terminal commit still REPORTS the failure,
      // it just cannot name the block any more -- and it never could on
      // an unwatched run, so this drops a diagnostic that was only
      // present when a bar happened to be attached rather than one
      // anything relied on.
      if (streaming || kv_io) {
        enc.end();
        // This block's FLOPs ride the buffer committed below, which it
        // waits on: flushed at the next block's start they would land on
        // the next block's buffer and be credited a block late.
        _flops.flush(stream);
        std::string blk_err;
        // Taken unconditionally: the query-tile probe charges this block's
        // wall time to whichever tile it ran, and one clock read per block is
        // nothing beside the block.
        const auto gp0 = std::chrono::steady_clock::now();
        metal_compute::CommandStream::Fence fence = stream.commit();
        // The cache's next block, into the slot block Lx - 1 used.
        if (kv_io) { skv->prefetch(Lx + 1); }
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
        if (streaming) { issue_prefetch(); }
        if (!fence.wait_ok(&blk_err)) {
          return fail("streamed block " + std::to_string(Lx) + ": " +
                      (blk_err.empty() ? std::string("GPU error") : blk_err));
        }
        if (kv_io) { skv->layer_done(Lx); }
        const double blk_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gp0).count();
        if (sprof && streaming) { sp_gpu_ms += blk_ms; }
        if (streaming) { note_attn_tile_(blk_ms); }
        // The same stream: a fresh one is a new queue, and its first
        // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
        // stream_per_commit_costs).
        enc = stream.begin_compute();
        mark = std::chrono::steady_clock::now();
      }
      if (streaming) {
        // The VDN branch follows the DiT, and HERE is the only place it
        // can: the commit above has been waited for, so nothing encoded
        // still points at this block's branch buffers. On the resident
        // path there is no such point -- the whole forward is one
        // deferred stream -- which is why this sits inside `streaming`
        // and not at the end of the block lambda.
        if (vdn_on && _vdn->stream_blocks()) {
          _vdn->release_block(Lx);
        }
        // The commit above has been WAITED for, so nothing encoded still
        // points at this block's buffers -- which is exactly why the
        // promotion happens here and not at the top of the iteration.
        // Keeping it costs nothing extra: the bytes are already resident,
        // and the alternative is dropping them and re-reading the same
        // block from disk on the next of 30-odd steps.
        if (Lx < (int)_blocks.size()) {
          // KEPT bytes, not the slot's: a promotion drops a streamed
          // layout's refill sources and converts a held w4 layout -- and
          // brings the block's spilled adapter factors back (keep_lora_
          // below; ~124 MB a block of the Open-H3-Regenerate merge).
          const std::size_t nl = lora_layer_bytes_(Lx);
          const std::size_t nb = kept_block_bytes_(*bp) + nl;
          // Past the wire budget there is nothing to gain: the block
          // would be kept unprotected, the compressor would take it (it
          // is the coldest memory in the process), and the next walk
          // would shed a block and ratchet the ceiling over the whole
          // resident set. Better not to hold it at all.
          const bool wirable = _wire.wirable(nb);
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
            // What a preloaded block holds, cloned or MOVED: the move
            // brings the fallback's refill sources and per-call pack. A
            // held layout that cannot be built leaves the pack converting
            // per projection, which is still the same values.
            if (kept) { (void)keep_block_(_blocks[(std::size_t)Lx]); }
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
              // Its adapters' factors stay with it.
              keep_lora_(Lx);
              // Wired LAST, after every write this block will ever get:
              // mlock pins the pages that exist now, and interleave_gu_
              // above replaces buffers outright.
              if (_wire.on()) {
                _wire.note_wired(
                    _mc,
                    wire_block_(_blocks[(std::size_t)Lx], true) +
                        wire_lora_(Lx, true),
                    nb);
              }
              _resid.note_admitted(nb);
              _lora_kept += nl;
              if (_mc->session() != nullptr) {
                const auto mb = _mc->memory_budget();
                _mc->session()->log_debug(fmt(
                    "MetalMiniMaxH3Transformer: block {} resident ({} of {}, "
                    "{} MB, {} MB wired; {} MB idle, {} MB compressed, "
                    "{} MB swap, reserve {} MB)", Lx,
                    _resid.count(), c.n_layers, _resid.bytes() >> 20,
                    _wire.wired_bytes() >> 20,
                    mb.free_physical >> 20, mb.compressed >> 20,
                    mb.swap_used >> 20, _resid.reserve() >> 20));
              }
            }
          }
        }
      }
    }

    // Every read and write of the cache's files, joined: a clean
    // forward's chunk is whole on disk before finish_commit names it.
    if (kv_io) {
      std::string kerr;
      if (!skv->end_forward(&kerr)) { return fail(kerr); }
      if (_mc->session() != nullptr) {
        const minimax_h3::StreamKv::IoStats io = skv->io_stats();
        _mc->session()->log_debug(fmt(
            "MetalMiniMaxH3Transformer: stream K/V from disk -- {} rows, "
            "{} MB read, {} MB written, {:.0f} ms waited for reads",
            s_rows, io.read >> 20, io.written >> 20, io.wait_ms));
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
    // A streaming COMMIT wanted the blocks' keys and values, which the
    // attention has already written into the cache; it has no velocity.
    if (!stream_commit) {
    const SharedBuffer* fmod_buf = &s.fmod;
    std::size_t fmod_off = 0;
    const LoraStack fin_lo = lora_stack_final_();
    if (baked && !c.pruned()) {
      const std::size_t row0 =
          (std::size_t)_adaln_row0[(std::size_t)in.schedule_index] *
          (std::size_t)(2 * H);
      if (fin_lo.empty()) {
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
        lora_apply_(enc, s.temb, 0, fin_lo, s.fmod, 0, n_t, 2 * H,
                    c.time_dim);
      }
    } else {
      if (c.pruned()) {
        pruned_mod_(enc, _final_padaln, -1, s.fmod, n_t, 2 * H,
                    baked ? &_final_adaln_tab : nullptr,
                    brow * (std::size_t)(2 * H) * 2);
      } else {
        gemm_(enc, s.temb, 0, _final_adaln, s.fmod, 0, n_t, 2 * H,
              c.time_dim);
      }
      lora_apply_(enc, s.temb, 0, fin_lo, s.fmod, 0, n_t, 2 * H,
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
    }   // !stream_commit
    psplit(t_final);
  }
  std::string gpu_err;
  const bool trip_on = !_nan_report.empty() &&
                       std::getenv("VPIPE_H3_NAN_TRIP") != nullptr;
  _flops.flush(stream);
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? std::string("MiniMax-H3 DiT forward failed")
                                : gpu_err);
  }
  // Every routed layer has run, so Sol's count is this forward's whole.
  // (The log at the next forward's start reads the same count.)
  if (_sol) {
    _sol_flops.finish(_sol->exact_blocks() +
                      (_sol_tail ? _sol_tail->exact_blocks() : 0));
  }
  // An FP8 weight that could not be widened left its projection
  // unwritten: the forward ran and its output is not the model's.
  if (_fp8_failed) {
    _fp8_failed = false;
    return fail("an FP8 weight could not be widened for its GEMM (dequant "
                "scratch allocation or buffer size); the forward is void");
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
        "  prep  {:8.1f} ms   elt {:8.1f} ms   final {:8.1f} ms\n"
        "  vdn   {:8.1f} ms ({}), weights {:8.1f} ms, "
        "softmax gate {:8.1f} ms",
        seq, n_text, n_cond, n_audio, L.num_video_rows, c.n_layers, n_t, tot,
        t_in, t_adaln, rate(f_ada, t_adaln),
        t_qkv, rate(f_qkv, t_qkv), t_oproj, rate(f_op, t_oproj),
        t_ff, rate(f_ff, t_ff), t_attn, rate(f_attn, t_attn),
        // WHICH attention actually ran, because the three are different
        // functions and not three speeds of one. A profile that says
        // "steel flash" for a windowed run reads as the dense kernel.
        !vdn_linear
            ? (use_steel ? "steel flash" : "SCALAR sdpa")
            : (spans_ok ? "steel flash, WINDOWED (block-sparse)"
                        : "SCALAR sdpa_spans, windowed"),
        t_prep, t_elt, t_final,
        // The linear branch: its features, statistics, solve, scan,
        // gather and readout, plus the to_out_linear projection. Zero
        // when no branch is attached, and zero under full cover, where
        // the window reaches every frame and the linear half is
        // correctly off.
        t_vdn,
        !_vdn ? "no branch attached"
              : (_vdn_full_cover ? "full cover -- linear half OFF"
                                 : "linear branch on video rows"),
        t_vdn_read, t_sgate));
  }
  // The GPU has run the whole forward, write-backs included, so the
  // cache's bookkeeping can now say what its buffers hold.
  if (stream_commit) { skv->finish_commit(splan); }
  out.ok = true;
  return out;
}

}  // namespace genai
}  // namespace vpipe
