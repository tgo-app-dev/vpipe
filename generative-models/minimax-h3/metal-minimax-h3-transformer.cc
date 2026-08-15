#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/shared/stream-pin.h"
#include "generative-models/shared/kernel-autotune.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
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
    // The asked-for partition ranks first; the historical `fl2va`
    // preference stays as the fallback so every graph written before
    // Ref2VA existed keeps resolving the way it did.
    const std::vector<std::string> prefer =
        partition.empty() ? std::vector<std::string>{"fl2va"}
                          : std::vector<std::string>{partition, "fl2va"};
    const std::string f = comfy::resolve_component(path, "diffusion_models",
                                                   kComfyKey, prefer);
    if (!f.empty()) { return f; }
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
  // subdirectory. FL2VA is the partition this class implements; Ref2VA is
  // a different task with a different packed layout, so it is NOT a
  // fallback -- a repo with only Ref2VA resolves to nothing and fails at
  // load with a missing-config error rather than running the wrong
  // layout.
  if (fs::exists(p / "FL2VA" / "transformer" / "config.json")) {
    return (p / "FL2VA" / "transformer").string();
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

SharedBuffer
MetalMiniMaxH3Transformer::weight_(WeightSet& ws, const std::string& nm,
                                   Retain r)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr) { return {}; }
  if (info->dtype == "BF16") {
    // Already the forward's dtype, so the model keeps it AS IS -- a
    // plain cached tensor(). Copied rather than Mapped: load_mapped()
    // wraps the whole shard, and at 66 GB over 13 shards mapping one
    // tensor would make the entire checkpoint resident.
    return r == Retain::Streamed
               ? ws.stream_tensor(nm, _mc, WeightSet::Residency::Copied)
               : ws.tensor(nm, _mc, WeightSet::Residency::Copied);
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
    l.codes  = r == Retain::Streamed
                   ? ws.stream_tensor(nm + ".weight", _mc,
                                      WeightSet::Residency::Copied)
                   : ws.tensor(nm + ".weight", _mc,
                               WeightSet::Residency::Copied);
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
// Into FRESH buffers rather than in place: the source may be a cached
// WeightSet tensor, which is shared with every other holder of that
// checkpoint and must never be written through (see the WeightSet
// immutability contract; VPIPE_WEIGHT_INTEGRITY=1 catches a violation).
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
  SharedBuffer codes = _mc->make_shared_buffer((std::size_t)N * row_codes);
  SharedBuffer scales = _mc->make_shared_buffer((std::size_t)N * row_grp * 2);
  SharedBuffer qbias = _mc->make_shared_buffer((std::size_t)N * row_grp * 2);
  if (codes.empty() || scales.empty() || qbias.empty()) { return false; }
  const auto* sc = static_cast<const std::uint8_t*>(l.codes.contents());
  const auto* ss = static_cast<const std::uint16_t*>(l.scales.contents());
  const auto* sb = static_cast<const std::uint16_t*>(l.qbias.contents());
  auto* dc = static_cast<std::uint8_t*>(codes.contents());
  auto* ds = static_cast<std::uint16_t*>(scales.contents());
  auto* db = static_cast<std::uint16_t*>(qbias.contents());
  const int half = N / 2;
  for (int j = 0; j < half; ++j) {
    const int from[2] = {j, half + j};          // gate row, then up row
    for (int h = 0; h < 2; ++h) {
      const std::size_t d = (std::size_t)(2 * j + h);
      const std::size_t s = (std::size_t)from[h];
      std::memcpy(dc + d * row_codes, sc + s * row_codes, row_codes);
      std::memcpy(ds + d * row_grp, ss + s * row_grp, row_grp * 2);
      std::memcpy(db + d * row_grp, sb + s * row_grp, row_grp * 2);
    }
  }
  l.codes = std::move(codes);
  l.scales = std::move(scales);
  l.qbias = std::move(qbias);
  l.gu_inter = true;
  return true;
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

MetalMiniMaxH3Transformer::~MetalMiniMaxH3Transformer() = default;

std::unique_ptr<MetalMiniMaxH3Transformer>
MetalMiniMaxH3Transformer::load(const std::string& dit_dir, MetalCompute* mc,
                                const Config& cfg, bool stream_blocks,
                                double pin_frac, const LoraSpec* lora)
{
  return load(WeightSet::open(resolve_dit_dir(dit_dir, cfg.partition),
                             nullptr),
              mc, cfg, stream_blocks, pin_frac, lora);
}

std::unique_ptr<MetalMiniMaxH3Transformer>
MetalMiniMaxH3Transformer::load(std::shared_ptr<WeightSet> ws_in,
                                MetalCompute* mc, const Config& cfg,
                                bool stream_blocks, double pin_frac,
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
    if (!stream_blocks) { pin_frac = 0.0; }
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
    m->_splitk.load(mc, m->_lib_dense_mma, m->_lib_elt);
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
  } else {
    if (pin_frac > 0.0) {
      std::vector<std::string> prefixes((std::size_t)cfg.n_layers);
      for (int i = 0; i < cfg.n_layers; ++i) {
        prefixes[(std::size_t)i] = blk_("blocks.", i, "");
      }
      m->_pinned = stream_pin_count(ws.src(), prefixes, pin_frac);
      if (m->_pinned > cfg.n_layers) { m->_pinned = cfg.n_layers; }
    }
    // Sized to the FULL depth even though only the pinned prefix is
    // filled: the empty slots are where forward() promotes streamed
    // blocks into as free memory allows (see set_residency_reserve).
    // An unfilled slot reads as empty, which is what `held` tests.
    m->_blocks.resize((std::size_t)cfg.n_layers);
    for (int i = 0; i < m->_pinned; ++i) {
      if (!m->load_block_(ws, blk_("blocks.", i, ""),
                          m->_blocks[(std::size_t)i], true, r)) {
        return nullptr;
      }
    }
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
    for (const QmmTuneSet& set : _qmm_tuned) {
      if (set.m != M) { continue; }
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
  const int band = mma_row_band_(N);
  for (int m0 = 0; m0 < M; m0 += band) {
    gemm_route_dispatch_(enc, x, x_off + (std::size_t)m0 * K, l, y,
                         y_off + (std::size_t)m0 * N,
                         std::min(band, M - m0), N, K, route, nullptr,
                         nullptr);
  }
}

int
MetalMiniMaxH3Transformer::mma_row_band_(int N) const
{
  static const int kForced = [] {
    const char* e = std::getenv("VPIPE_H3_MMA_ROW_BAND");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  if (kForced > 0) { return kForced; }
  if (N <= 0) { return 1 << 30; }
  // The last row a 32-bit BYTE offset can reach, floored to the 128-row
  // tile the mma kernels step in -- a band that ended mid-tile would
  // still hand one tile an offset past the line.
  const long long lim = (((long long)1 << 31) - 1) / ((long long)N * 2);
  long long band = (lim / 128) * 128;
  if (band < 128) { band = 128; }
  if (band > (1 << 30)) { band = 1 << 30; }
  return (int)band;
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
  const int band = mma_row_band_(N);
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
void
MetalMiniMaxH3Transformer::tune_qmm_(int M)
{
  if (_qmm_manual) { return; }   // the caller is driving; see set_qmm_tile
  for (const QmmTuneSet& set : _qmm_tuned) {
    if (set.m == M) { return; }
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
    const int w = autotune_vote((int)cands.size(), /*rounds=*/2,
        /*reps_for_us=*/1,
        [&](int i) {
          return autotune_time(_mc, 1, [&](ComputeEncoder& enc) {
            // No adapter: the tuner measures the BASE projection. The
            // fold adds ~1% of a tile's work and only to the two plain
            // tiles, which is far short of flipping a route, and giving
            // the tuner an adapter would make its choice depend on which
            // LoRA happened to be loaded.
            //
            // Banded exactly as the forward encodes it. Unbanded, an mma
            // route past 2^31 bytes would silently skip its last tiles
            // and be timed for work it did not do, while steel -- which
            // addresses in 64 bits -- did all of it.
            dispatch_row_bands_(enc, xin, 0, *sh.l, yout, 0, M, sh.N, sh.K,
                                cands[(std::size_t)i]);
          });
        });
    const GemmRoute win = cands[(std::size_t)w];
    tuned.push_back(QmmTune{sh.N, sh.K, win});
    if (!detail.empty()) { detail += " "; }
    detail += std::string(sh.name) + "=" + route_name_(win);
    // Second decision, only where a split is possible at all: with the tile
    // settled, is the contraction better split? Driven through the same
    // dispatcher, so the tuner compares the winning route against itself
    // with and without the split rather than against a stand-in. The route
    // has to be pinned first -- measuring both at once would confound them.
    if (win == GemmRoute::kMma128 || win == GemmRoute::kMma128x256 ||
        win == GemmRoute::kMma128x256Tn2) {
      const int sp = _splitk.tune(_mc, sh.K, sh.N, M,
          [&](ComputeEncoder& enc) {
            dispatch_row_bands_(enc, xin, 0, *sh.l, yout, 0, M, sh.N, sh.K,
                                win);
          });
      if (sp > 0) { detail += "+split" + std::to_string(sp); }
    }
  }
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  // Bounded: a pathological caller sweeping sequence lengths must not
  // grow this without limit, and the oldest entry is the one least
  // likely to come back.
  if (_qmm_tuned.size() >= 8) { _qmm_tuned.erase(_qmm_tuned.begin()); }
  _qmm_tuned.push_back(QmmTuneSet{M, std::move(tuned)});
  _qmm_tuning_desc = std::to_string(M) + ": " + detail;
  if (_mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "[h3-dit] qmm autotune at {} rows: {} ({:.0f} ms)", M, detail, ms));
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
  static const std::size_t kBudget = []() {
    const char* e = std::getenv("VPIPE_H3_ADALN_BAKE_MAX_MB");
    const long v = (e != nullptr) ? std::atol(e) : 0;
    return (std::size_t)(v > 0 ? v : 1536) << 20;
  }();
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

  auto lin_bytes = [](const Linear& l) {
    return l.w.byte_size() + l.b.byte_size() + l.codes.byte_size() +
           l.scales.byte_size() + l.qbias.byte_size();
  };
  std::size_t freed = 0;
  _adaln_tab.clear();
  _adaln_tab.resize((std::size_t)c.n_layers);
  for (int i = 0; i < c.n_layers; ++i) {
    // Take the projection from wherever it is. A resident block already
    // holds it; a streamed one does not, and reading JUST this tensor is
    // the point -- the bake never needs the rest of the block, so it
    // touches the 12.9 GB once and nothing else.
    Linear held;
    const Linear* ada = nullptr;
    if ((std::size_t)i < _blocks.size() && !_blocks[(std::size_t)i].adaln.empty()) {
      ada = &_blocks[(std::size_t)i].adaln;
    } else {
      held = linear_(*_ws, blk_("blocks.", i, "") + "adaln_proj.linear", true,
                     Retain::Streamed);
      if (held.empty()) { return fail(fmt("block {} has no adaln", i)()); }
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
      st.commit().wait();
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
    st.commit().wait();
  }
  freed += lin_bytes(*fada);
  _final_adaln = Linear{};

  if (_mc->session() != nullptr) {
    _mc->session()->log_normal(fmt(
        "MetalMiniMaxH3Transformer: baked AdaLN for {} steps ({} rows) -- "
        "{} projections ({} MB) replaced by {} MB of tables{}",
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

std::vector<ScratchItem>
scratch_plan_(const MetalMiniMaxH3Transformer::Config& c, bool narrow_ff)
{
  const std::size_t H = (std::size_t)c.hidden;
  const std::size_t I = (std::size_t)c.inner();
  const std::size_t rot_half = (std::size_t)(3 * c.rope_freq_dim);
  return {
      {rot_half, 0, 0, sizeof(float)},           // rcos
      {rot_half, 0, 0, sizeof(float)},           // rsin
      {H, 0, 0, 2}, {H, 0, 0, 2}, {H, 0, 0, 2},  // x, nm, proj
      {3 * I, 0, 0, 2},                          // qkv
      {I, 0, 0, 2}, {I, 0, 0, 2}, {I, 0, 0, 2},  // qh, kh, vh
      {I, 0, 0, 2}, {I, 0, 0, 2},                // oh, ob
      {narrow_ff ? 3 * I : 2 * (std::size_t)c.ffn, 0, 0, 2},   // ff
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
      &_s.rcos, &_s.rsin, &_s.x, &_s.nm, &_s.qkv, &_s.qh, &_s.kh, &_s.vh,
      &_s.oh, &_s.ob, &_s.ff, &_s.proj, &_s.txt, &_s.temb, &_s.mod,
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

std::size_t
MetalMiniMaxH3Transformer::release_resident_blocks(std::size_t bytes)
{
  // Evict from the TAIL down, so what remains stays a contiguous prefix.
  // In a cyclic scan every block is worth the same, so there is nothing
  // cleverer to choose and a prefix keeps the bookkeeping trivial. The
  // pinned prefix is never touched.
  int next = (int)_blocks.size() - 1;
  const std::size_t freed = _resid.release(bytes, [&]() -> std::size_t {
    for (; next >= _pinned; --next) {
      Block& b = _blocks[(std::size_t)next];
      const std::size_t n = block_bytes_(b);
      if (n == 0) { continue; }
      b = Block{};
      --next;
      return n;
    }
    return 0;
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
  s.proj = mk(S * H);
  s.qkv  = mk(S * 3 * I);
  s.qh   = mk(S * I);
  s.kh   = mk(S * I);
  s.vh   = mk(S * I);
  s.oh   = mk(S * I);
  s.ob   = mk(S * I);
  // The [seq, 2*ffn] fc1 intermediate -- the single largest scratch this
  // model holds. A fused FF never writes it, so when every block is
  // interleaved it shrinks to what its OTHER user needs: tune_qmm_'s
  // destination, whose widest remaining shape is qkv's 3*inner. The
  // public scratch_bytes() still quotes the wide figure, so the stage's
  // preflight over-estimates rather than under-estimates.
  s.ff   = mk(S * (_ff_needs_wide ? 2 * (std::size_t)c.ffn : 3 * I));
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
  s.tstep_idx = _mc->make_shared_buffer(S * sizeof(int));
  if (s.rcos.empty() || s.rsin.empty() || s.x.empty() || s.nm.empty() ||
      s.proj.empty() || s.qkv.empty() || s.qh.empty() || s.kh.empty() ||
      s.vh.empty() || s.oh.empty() || s.ob.empty() || s.ff.empty() ||
      s.txt.empty() || s.temb.empty() || s.mod.empty() || s.fmod.empty() ||
      s.adaln_idx.empty() || s.tstep_idx.empty()) {
    return false;
  }
  _s = std::move(s);
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
  if (!ensure_scratch_(seq, n_text, n_t)) {
    return fail("activation allocation failed (out of GPU memory)");
  }
  // Re-arm growth for this forward. The per-forward flag is what stops
  // the budget being re-queried for every one of 50 blocks once the
  // answer is no; the RATCHET (_resid_ceiling) is what survives across
  // forwards, so a set that was cut back does not simply refill next
  // step. Scratch is allocated by now, so what the budget reports here
  // already has it subtracted.
  _resid.begin_forward();
  Scratch& s = _s;
  // Measure the GEMM tile for THIS sequence length, before the stream
  // opens -- the tuner needs its own command buffers, and the answer has
  // to be in hand before the first block encodes.
  tune_qmm_(seq);

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
  if (use_steel && (_attn_seq != seq || _attn_text != n_text)) {
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
      p->Q_strides[0] = (std::int64_t)NH * qL * HD;
      p->Q_strides[1] = (std::int64_t)qL * HD;
      p->Q_strides[2] = HD;
      for (int i = 0; i < 3; ++i) {
        p->K_strides[i] = p->Q_strides[i];
        p->V_strides[i] = p->Q_strides[i];
        p->O_strides[i] = p->Q_strides[i];
      }
    };
    fill(_attn_p_main, seq);
    fill(_attn_p_text, n_text);
    auto build = [&](int qL) {
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (qL % A_BQ) == 0).set_bool(201, (qL % A_BK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      return _attn_nax
                 ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
                 : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
    };
    _fn_attn_main = build(seq);
    _fn_attn_text = build(n_text);
    _attn_seq = seq;
    _attn_text = n_text;
  }
  if (use_steel) {
    use_steel = _fn_attn_main.valid() && _fn_attn_text.valid();
  }

  // ---- env-gated per-section GPU timing (VPIPE_H3_DIT_PROFILE) --------
  // The whole forward is ONE deferred stream, so there is nothing to
  // time inside it without splitting: each psplit() ends the encoder,
  // commits, waits and charges the slice to a bucket. That serializes
  // the GPU and inflates the total, so read the SHARE, not the sum.
  const bool prof = std::getenv("VPIPE_H3_DIT_PROFILE") != nullptr;
  double t_in = 0, t_adaln = 0, t_qkv = 0, t_prep = 0, t_attn = 0,
         t_oproj = 0, t_ff = 0, t_elt = 0, t_final = 0;
  const auto t_begin = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point mark = t_begin;

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
        enc.set_buffer(0, s.qh); enc.set_buffer(1, s.kh);
        enc.set_buffer(2, s.vh); enc.set_buffer(3, s.oh);
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
      if (modulated) {
        modulate(s.nm, *mod_buf, mod_off * 2, s.adaln_idx, s.nm, rows,
                 6 * H, H, 0);
      }
      bdump("n1+mod", s.nm, H);
      psplit(t_elt);
      gemm_(enc, s.nm, 0, b.qkv, s.qkv, 0, rows, 3 * I, H,
            lo != nullptr ? &lo->qkv : nullptr);
      bdump("qkv", s.qkv, 3 * I);
      psplit(t_qkv);
      // Per-HEAD RMS over head_dim, in place on the fused buffer, BEFORE
      // rope -- the reference's order. See QKV_HSTRIDE / Q_OFF above for
      // the two groupings the fused projection ships in.
      qk_norm(b.qn, rows, Q_OFF);
      qk_norm(b.kn, rows, K_OFF);
      trope(s.qh, rows, Q_OFF, modulated ? c.rope_rot() : 0);
      trope(s.kh, rows, K_OFF, modulated ? c.rope_rot() : 0);
      trope(s.vh, rows, V_OFF, 0);
      psplit(t_prep);
      attn(rows, modulated);
      psplit(t_attn);
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, s.oh); enc.set_buffer(1, s.ob);
      enc.set_constant(2, NH); enc.set_constant(3, rows);
      enc.set_constant(4, HD);
      enc.dispatch({(unsigned)HD, (unsigned)rows, (unsigned)NH},
                   {(unsigned)HD, 1, 1});
      bdump("attn_out", s.ob, I);
      gemm_(enc, s.ob, 0, b.out, s.nm, 0, rows, H, I,
            lo != nullptr ? &lo->out : nullptr);
      bdump("o_proj", s.nm, H);
      psplit(t_oproj);
      if (modulated) { gated(x, s.nm, rows, 2 * H); }
      else           { residual(x, s.nm, rows); }
      bdump("x_after_attn", x, H);

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
        // fc1 is FUSED [gate | up], GATE first -- the diffusers SwiGLU
        // convention, not the llama one the rest of this tree follows.
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, s.ff); enc.set_buffer(1, s.qkv);
        enc.set_constant(2, rows);
        enc.set_constant(3, FF);
        enc.dispatch({(unsigned)(rows * FF), 1, 1}, {256, 1, 1});
      }
      bdump("swiglu", s.qkv, FF);
      gemm_(enc, s.qkv, 0, b.fc2, s.nm, 0, rows, H, FF,
            lo != nullptr ? &lo->fc2 : nullptr);
      bdump("fc2", s.nm, H);
      psplit(t_ff);
      if (modulated) { gated(x, s.nm, rows, 5 * H); }
      else           { residual(x, s.nm, rows); }
      bdump("x_after_ff", x, H);
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
      Block streamed;
      if (streaming) {
        // `with_adaln` false once baked: those tensors are 55% of a
        // block's bytes and nothing reads them any more, so a streaming
        // run stops paying for them on every step. This is the whole
        // point of baking.
        if (!load_block_(*_ws, blk_("blocks.", Lx, ""), streamed, !baked,
                         Retain::Streamed)) {
          return {};
        }
      }
      const Block& b = streaming ? streamed : _blocks[(std::size_t)Lx];
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
        if (!stream.commit().wait_ok(&blk_err)) {
          return fail("streamed block " + std::to_string(Lx) + ": " +
                      (blk_err.empty() ? std::string("GPU error") : blk_err));
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
          const std::size_t nb = block_bytes_(streamed);
          if (_resid.admit(_mc, nb)) {
            _blocks[(std::size_t)Lx] = std::move(streamed);
            // Now that it is KEPT, the fc1 interleave is paid once
            // instead of per forward, so every later step takes the
            // fused FF for this block. The scratch stays wide (it was
            // sized for a streaming run and blocks below this one may
            // still be unfused), so the two paths coexist.
            if (_fuse_ff) { interleave_gu_(_blocks[(std::size_t)Lx].fc1); }
            _resid.note_admitted(nb);
            if (_mc->session() != nullptr) {
              const auto mb = _mc->memory_budget();
              _mc->session()->log_debug(fmt(
                  "MetalMiniMaxH3Transformer: block {} resident ({} of {}, "
                  "{} MB; {} MB reclaimable, reserve {} MB)", Lx,
                  _resid.count() + _pinned, c.n_layers, _resid.bytes() >> 20,
                  mb.available_physical >> 20, _resid.reserve() >> 20));
            }
          }
        }
      }
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
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? std::string("MiniMax-H3 DiT forward failed")
                                : gpu_err);
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
