#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"

#include "generative-models/shared/kernel-autotune.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/minimax-h3/minimax-h3-diffusers-layout.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/shared/tuned-answers.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

constexpr const char* kKey = "minimax-h3-vvae/bf16|";

// Elements one elementwise dispatch is handed. The kernels take their
// count as an int, so the band keeps that honest at any extent; it is
// far above every shape that tiles, so nothing real splits.
constexpr std::size_t kEltBand = 1u << 30;

// The hardware conv's tiles (conv3d_hw_causal_s*_<name>_f16): the
// tuned candidates in order, then the partial-tile entry. Channels x
// width x height of the destination tile, and simdgroups.
struct C3Geom { int tc, tw, th, sg; };
constexpr const char* kC3Names[] = {"c128", "c128d", "c128w16d", "tail",
                                    "c16"};
constexpr C3Geom kC3Geom[] = {{128, 8, 8, 4}, {128, 8, 8, 4},
                              {128, 16, 8, 8}, {128, 8, 8, 4},
                              {16, 8, 8, 4}};

// Rows per threadgroup of the three-pass norm's statistics pass. A
// 256x256 frame is 128 bands, so a 17-frame clip is ~2000 threadgroups.
constexpr int kGnBand = 512;

// VPIPE_H3_VVAE_ENC_PROFILE, read once.
bool
enc_profile_on_()
{
  static const bool on = [] {
    const char* e = std::getenv("VPIPE_H3_VVAE_ENC_PROFILE");
    return e != nullptr && *e != '\0' && *e != '0';
  }();
  return on;
}

double
ms_since_(std::chrono::steady_clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
}

// The Comfy-Org single-file video VAE keeps the wrapper config (and,
// nested under "source_config", the net's) under this `__metadata__`
// key; its presence identifies the file.
constexpr const char* kComfyKey = "minimax_h3_video_vae";

inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

inline float
bf16_to_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

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

std::string
blk_(int i, const char* rest)
{
  return "decoder.transformer_blocks." + std::to_string(i) + "." + rest;
}

// A linear the checkpoint stores in pieces (the diffusers layout's split
// attention and swapped feed-forward), assembled into the bf16 tensor
// the released file would have converted to: the same rounding, so the
// two layouts load bit-identical weights. `suffix` is ".weight" or
// ".bias"; empty when a piece lacks it. Each piece is read ONCE and
// UNCACHED -- the assembly is what the model keeps.
SharedBuffer
assemble_bf16_(WeightSet& ws, MetalCompute* mc, const std::string& key,
               const std::vector<minimax_h3::Part>& parts,
               const char* suffix, const std::string& part)
{
  const MetalLlamaWeights& src = ws.src();
  std::int64_t rows = 0, width = -1;
  for (const minimax_h3::Part& pc : parts) {
    const auto* ti = src.info(pc.layer + suffix);
    if (ti == nullptr || ti->shape.empty()) { return {}; }
    std::int64_t w = 1;
    for (std::size_t d = 1; d < ti->shape.size(); ++d) { w *= ti->shape[d]; }
    if (width >= 0 && w != width) { return {}; }
    width = w;
    rows += pc.rows;
  }
  return ws.derived(key, [&]() -> SharedBuffer {
    std::map<std::string, SharedBuffer> raw;
    for (const minimax_h3::Part& pc : parts) {
      const std::string nm = pc.layer + suffix;
      if (raw.count(nm) == 0) {
        raw[nm] = ws.read(nm, mc, WeightSet::Residency::Copied);
        if (raw[nm].empty()) { return {}; }
      }
    }
    SharedBuffer out =
        mc->make_shared_buffer((std::size_t)rows * (std::size_t)width * 2);
    if (out.empty()) { return {}; }
    auto* d = static_cast<std::uint16_t*>(out.contents());
    for (const minimax_h3::Part& pc : parts) {
      const std::string nm = pc.layer + suffix;
      const std::string& dt = src.info(nm)->dtype;
      const void* base = raw[nm].contents();
      const std::size_t n = (std::size_t)pc.rows * (std::size_t)width;
      const std::size_t at = (std::size_t)pc.row0 * (std::size_t)width;
      for (std::size_t i = 0; i < n; ++i) {
        if (dt == "F32") {
          d[i] = f32_to_bf16_(static_cast<const float*>(base)[at + i]);
        } else if (dt == "F16") {
          d[i] = f32_to_bf16_(
              (float)static_cast<const _Float16*>(base)[at + i]);
        } else if (dt == "BF16") {
          d[i] = static_cast<const std::uint16_t*>(base)[at + i];
        } else {
          return {};
        }
      }
      d += n;
    }
    return out;
  }, part);
}

// The checks a config meets once its fields are read, whichever layout
// they came from: the derived widths, the shapes this decoder
// implements, and a tiling that covers the image.
bool
finish_config_(MetalMiniMaxH3VideoVae::Config& out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  out.dim = out.n_heads * out.head_dim;
  // FeedForward(dim, mult=4): the diffusers default this decoder is
  // constructed with, so the inner width is not in the config.
  out.ffn_inner = 4 * out.dim;
  if (out.head_dim != 64) {
    return fail("only dim_head 64 is supported (the steel flash-attention "
                "entry point this decoder runs on is bd64)");
  }
  if (out.rope_dim() % 6 != 0) {
    return fail("int(dim_head * rope_dim_ratio) must be divisible by 6");
  }
  // Tile size, overridable for MEASUREMENT. The checkpoint's 256 is the
  // reference's number, and the cost it buys is not obvious: at 960x544
  // it lays 15 tiles whose union decodes 1.88x the pixels that exist, so
  // 88% of the decode is overlap -- but tiles are also where attention is
  // quadratic, so bigger tiles trade redundant GEMM for superlinear
  // attention and the balance is an empirical question per machine.
  //
  // NOT a tuning knob to ship a different default behind: the tiling
  // decides where the seams fall and what the cross-fade blends, so a
  // different size is a different OUTPUT, not the same one computed
  // faster. Anything found here has to clear the decode golden before it
  // becomes a default.
  if (const char* e = std::getenv("VPIPE_H3_VVAE_TILE")) {
    const int t = std::atoi(e);
    if (t >= out.patch && t % out.patch == 0) { out.tile_size = t; }
  }

  // ---- the tiling must COVER the axis exactly ------------------------
  // `split_tiles` pays the leftover slack back in whole `ratio` steps,
  // so the union it lays overruns the axis by `slack % ratio` -- and
  // both tilers and `stitch_` trust the coverage. A tile that ends past
  // the image is an out-of-bounds READ in encode_tiled_/decode_tiled_
  // and an out-of-bounds WRITE in stitch_, which is memory corruption
  // rather than a wrong picture.
  //
  // The slack is `tile_size*n - overlap*(n-1) - length` for whichever
  // `n` the split lands on, and H/W are already multiples of `patch`,
  // so it is a multiple of `patch` for EVERY n exactly when both the
  // tile and the minimum overlap are. Checked here rather than in
  // split_tiles because this is where a checkpoint's numbers arrive,
  // and refusing a config beats silently retiling one.
  if (out.tile_size % out.patch != 0 ||
      out.tile_overlap_min % out.patch != 0) {
    return fail("vae_tile_size " + std::to_string(out.tile_size) +
                " and vae_tile_overlap_min " +
                std::to_string(out.tile_overlap_min) +
                " must both be multiples of the " +
                std::to_string(out.patch) +
                "-pixel patch, or the tiling overruns the image");
  }
  return true;
}

// The DIFFUSERS config (AutoencoderKLMiniMaxH3, at the repository root's
// `vae/`): one flat file where the released layout keeps two -- the
// wrapper's clip geometry and latent whitening, and the source net's
// shape -- under other names. What the released config states as flags
// (a ViT decoder, gated SiLU, causal 3D encoder, per-frame group norm)
// the diffusers class has no other form of, so there is nothing to
// check for them; what it states as numbers is read, and what this
// decoder hard-codes is refused when it differs.
bool
diffusers_config_(const std::string& dir,
                  MetalMiniMaxH3VideoVae::Config& out, std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const fs::path p = fs::path(dir) / "config.json";
  FlexData cfg;
  {
    std::ifstream f(p);
    if (!f) { return fail("cannot open " + p.string()); }
    try {
      cfg = FlexData::from_json(f);
    } catch (...) {
      return fail("cannot parse " + p.string());
    }
  }
  if (!cfg.is_object()) { return fail(p.string() + " is not a JSON object"); }
  auto o = cfg.as_object();
  auto gi = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  auto gr = [&](const char* k, double d) {
    return o.contains(k) ? o.at(k).as_real(d) : d;
  };
  auto ints = [&](const char* k, std::vector<int>& dst) {
    if (!o.contains(k)) { return; }
    FlexData v = o.at(k);
    if (!v.is_array()) { return; }
    auto a = v.as_array();
    std::vector<int> got;
    for (std::size_t i = 0; i < a.size(); ++i) {
      got.push_back((int)a.at(i).as_int(0));
    }
    if (!got.empty()) { dst = std::move(got); }
  };
  out.z_channels   = gi("latent_channels", out.z_channels);
  out.in_channels  = gi("in_channels", out.in_channels);
  out.out_channels = gi("out_channels", out.out_channels);
  out.res_blocks   = gi("layers_per_block", out.res_blocks);
  out.gn_groups    = gi("norm_num_groups", out.gn_groups);
  out.enc_norm_eps = (float)gr("norm_eps", out.enc_norm_eps);
  out.n_layers     = gi("decoder_num_layers", out.n_layers);
  out.n_heads      = gi("decoder_num_attention_heads", out.n_heads);
  out.head_dim     = gi("decoder_attention_head_dim", out.head_dim);
  out.n_register   = gi("decoder_num_register_tokens", out.n_register);
  out.rope_theta   = gr("decoder_rope_theta", out.rope_theta);
  out.rope_dim_ratio = gr("decoder_rope_dim_ratio", out.rope_dim_ratio);
  out.norm_eps     = (float)gr("decoder_norm_eps", out.norm_eps);
  out.clip_length  = gi("clip_length", out.clip_length);
  out.token_drop   = gi("token_drop", out.token_drop);
  // The encoder's widths as diffusers states them: one per level, where
  // the released config gives a base and multipliers.
  std::vector<int> widths;
  ints("block_out_channels", widths);
  if (!widths.empty()) {
    out.base_ch = widths.front();
    out.ch_mult.clear();
    for (int w : widths) {
      if (out.base_ch <= 0 || w % out.base_ch != 0) {
        return fail(p.string() + ": block_out_channels are not multiples "
                    "of the first");
      }
      out.ch_mult.push_back(w / out.base_ch);
    }
  }
  ints("spatial_downsample_factors", out.space_down);
  ints("temporal_downsample_factors", out.time_down);
  if (out.ch_mult.size() != out.space_down.size() ||
      out.ch_mult.size() != out.time_down.size()) {
    return fail(p.string() + ": block_out_channels and the spatial and "
                "temporal downsample factors must have one entry per level");
  }
  // No separate statement of the compression to hold the factors
  // against: here they ARE the statement.
  out.patch = 1;
  out.patch_t = 1;
  for (std::size_t i = 0; i < out.ch_mult.size(); ++i) {
    out.patch *= out.space_down[i];
    out.patch_t *= out.time_down[i];
  }
  if (o.contains("spatial_padding_mode")) {
    const FlexData v = o.at("spatial_padding_mode");   // a VIEW: bind it
    const std::string pm(v.as_string("reflect"));
    if (pm != "reflect") {
      return fail("encoder spatial_padding_mode '" + pm + "' is not reflect");
    }
  }
  if (gi("decoder_ffn_mult", 4) != 4) {
    return fail("decoder_ffn_mult is not 4: the feed-forward here is "
                "4 * dim wide");
  }
  auto reals = [&](const char* k, std::vector<float>& dst) {
    if (!o.contains(k)) { return; }
    FlexData v = o.at(k);
    if (!v.is_array()) { return; }
    auto a = v.as_array();
    std::vector<float> got;
    for (std::size_t i = 0; i < a.size(); ++i) {
      got.push_back((float)a.at(i).as_real(0.0));
    }
    if ((int)got.size() == out.z_channels) { dst = std::move(got); }
  };
  reals("latents_mean", out.latents_mean);
  reals("latents_std", out.latents_std);
  // vae_tile_size / vae_tile_overlap_min are not stated; the defaults
  // are the released 256 / 64, which this class and the reference's
  // diffusers wrapper share.
  return true;
}

}  // namespace

std::string
MetalMiniMaxH3VideoVae::resolve_vae_dir(const std::string& path)
{
  namespace fs = std::filesystem;
  fs::path p(path);
  // The Comfy-Org single file, probed first for the same reason as the
  // DiT's: it is self-describing, where a directory beside it is not.
  // Unlike the DiT, this conversion does NOT repack the tensors -- it is
  // the released net, cast to fp16 -- so the only thing that changes
  // here is where the config comes from.
  {
    const std::string f =
        comfy::resolve_component(path, "vae", kComfyKey, {"video_vae"});
    if (!f.empty()) { return f; }
  }
  if (!fs::is_directory(p)) { return path; }
  if (fs::exists(p / "model.safetensors") && fs::exists(p / "config.json") &&
      !fs::exists(p / "source")) {
    return p.string();                                 // already source/
  }
  // The diffusers VAE named directly: its weights sit beside its config.
  if (minimax_h3::is_diffusers_vae_dir(p.string(),
                                       minimax_h3::kDiffusersVideoVaeClass)) {
    return p.string();
  }
  if (fs::exists(p / "source" / "model.safetensors")) {
    return (p / "source").string();                    // video_vae/
  }
  if (fs::exists(p / "video_vae" / "source" / "model.safetensors")) {
    return (p / "video_vae" / "source").string();      // a partition root
  }
  // Either partition's copy -- see the audio VAE: the two subtrees are
  // complete pipelines over the same VAEs, so Ref2VA must resolve and
  // either copy is the same net.
  for (const char* part : {"FL2VA", "Ref2VA"}) {
    const fs::path src = p / part / "video_vae" / "source";
    if (fs::exists(src / "model.safetensors")) { return src.string(); }
  }
  // The diffusers pipeline at the repository's root, LAST: a checkout
  // that also holds a partition has just answered with the released
  // copy, the one every graph already shares (the same numbers; see
  // minimax-h3-diffusers-layout.h).
  if (minimax_h3::is_diffusers_vae_dir((p / "vae").string(),
                                       minimax_h3::kDiffusersVideoVaeClass)) {
    return (p / "vae").string();
  }
  return path;
}

bool
MetalMiniMaxH3VideoVae::config_from_json(const std::string& vae_dir,
                                         Config& out, std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  fs::path p(resolve_vae_dir(vae_dir));
  if (minimax_h3::is_diffusers_vae_dir(p.string(),
                                       minimax_h3::kDiffusersVideoVaeClass)) {
    return diffusers_config_(p.string(), out, err) &&
           finish_config_(out, err);
  }
  // `cfg` is the source net's config (diffusers `source/config.json`);
  // `wrap` is the wrapper's, which is where the clip/tile geometry and
  // the latent whitening live. The released layout keeps them in two
  // files one directory apart; the Comfy-Org file carries both in one
  // metadata blob, with the source config nested under "source_config".
  FlexData cfg, wrap;
  std::string comfy_file;
  if (comfy::is_component(p.string(), kComfyKey)) {
    std::string cerr;
    if (!comfy::metadata_json(p.string(), kComfyKey, wrap, &cerr)) {
      return fail(cerr);
    }
    if (!wrap.is_object() || !wrap.as_object().contains("source_config")) {
      return fail(p.string() + ": __metadata__ has no 'source_config'");
    }
    cfg = wrap.as_object().at("source_config");
    if (!cfg.is_object()) {
      return fail(p.string() + ": 'source_config' is not an object");
    }
    // NOTE: vae_tile_size / vae_tile_overlap_min are NOT in this blob.
    // They fall back to the Config defaults, which are the released
    // 256 / 64 -- and they must, because a tile's rope is built from its
    // own extent, so a different tiling is a different function rather
    // than a different seam.
    //
    comfy_file = p.string();
  } else {
    if (fs::is_directory(p)) { p = p / "config.json"; }
    std::ifstream f(p);
    if (!f) { return fail("cannot open " + p.string()); }
    try {
      cfg = FlexData::from_json(f);
    } catch (...) {
      return fail("cannot parse " + p.string());
    }
    if (!cfg.is_object()) { return fail(p.string() + " is not a JSON object"); }
    std::ifstream pf(p.parent_path().parent_path() / "config.json");
    if (pf) {
      try {
        wrap = FlexData::from_json(pf);
      } catch (...) {
        wrap = FlexData::make_null();
      }
    }
  }
  auto o = cfg.as_object();
  auto gi = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  out.z_channels   = gi("z_channels", 24);
  out.out_channels = gi("out_ch", 3);
  out.patch        = gi("vae_ratio", 16);
  out.patch_t      = gi("vae_ratio_t", 4);
  // The decoder MUST be the ViT one: the CNN decoder this class does not
  // implement reports the same class name and the same latent geometry,
  // so running it here would load a disjoint set of tensors and fail on
  // a missing name rather than on the thing that is actually wrong.
  if (o.contains("use_vit_decoder") && !o.at("use_vit_decoder").as_bool(true)) {
    return fail("this checkpoint has a CNN decoder (use_vit_decoder false), "
                "which is not implemented here");
  }
  if (o.contains("vit_decoder_kwargs")) {
    FlexData v = o.at("vit_decoder_kwargs");
    if (v.is_object()) {
      auto vo = v.as_object();
      auto vi = [&](const char* k, int d) {
        return vo.contains(k) ? (int)vo.at(k).as_int(d) : d;
      };
      out.n_layers  = vi("num_layers", 36);
      out.n_heads   = vi("heads", 32);
      out.head_dim  = vi("dim_head", 64);
      if (vo.contains("rope_theta")) {
        out.rope_theta = vo.at("rope_theta").as_real(100.0);
      }
      if (vo.contains("rope_dim_ratio")) {
        out.rope_dim_ratio = vo.at("rope_dim_ratio").as_real(0.75);
      }
      // The released config gates the feed-forward shape on two flags
      // rather than stating it. Anything but gated SiLU is a different
      // activation on the same weights.
      if (vo.contains("ffn_use_gated") && !vo.at("ffn_use_gated").as_bool(true)) {
        return fail("ffn_use_gated is false: this decoder's feed-forward is "
                    "not the gated one implemented here");
      }
      if (vo.contains("ffn_activation_fn")) {
        const std::string a(vo.at("ffn_activation_fn").as_string("silu"));
        if (a != "silu") {
          return fail("ffn_activation_fn '" + a + "' is not silu");
        }
      }
    }
  }
  // ---- the CNN encoder ------------------------------------------------
  out.in_channels = gi("in_channels", 3);
  out.base_ch     = gi("ch", 128);
  out.res_blocks  = gi("num_res_blocks", 2);
  auto ints = [&](const char* k, std::vector<int>& dst) {
    if (!o.contains(k)) { return; }
    FlexData v = o.at(k);
    if (!v.is_array()) { return; }
    auto a = v.as_array();
    std::vector<int> got;
    for (std::size_t i = 0; i < a.size(); ++i) {
      got.push_back((int)a.at(i).as_int(1));
    }
    if (!got.empty()) { dst = std::move(got); }
  };
  ints("ch_mult", out.ch_mult);
  ints("space_down", out.space_down);
  ints("time_down", out.time_down);
  if (out.ch_mult.size() != out.space_down.size() ||
      out.ch_mult.size() != out.time_down.size()) {
    return fail("ch_mult, space_down and time_down must have one entry per "
                "level");
  }
  // The per-level factors are the only statement of where a
  // downsampling convolution exists, so a product that disagrees with
  // the advertised compression ratio means the levels are being built
  // from the wrong list -- which loads a plausible but different set of
  // tensors and only shows up as a wrong latent.
  int sprod = 1, tprod = 1;
  for (std::size_t i = 0; i < out.ch_mult.size(); ++i) {
    sprod *= out.space_down[i];
    tprod *= out.time_down[i];
  }
  if (sprod != out.patch || tprod != out.patch_t) {
    return fail("space_down/time_down multiply out to " +
                std::to_string(sprod) + "x" + std::to_string(tprod) +
                ", not the configured " + std::to_string(out.patch) + "x" +
                std::to_string(out.patch_t));
  }
  // Three encoder behaviours are configurable in the reference and the
  // implementation here hard-codes all three. Each fails silently: zero
  // padding only darkens the frame border, whole-clip group norm is a
  // good approximation of the per-frame one, and a non-causal encoder
  // merely shifts the temporal alignment.
  if (o.contains("padding_mode")) {
    const std::string pm(o.at("padding_mode").as_string("reflect"));
    if (pm != "reflect") {
      return fail("encoder padding_mode '" + pm + "' is not reflect");
    }
  }
  if (o.contains("use_t_isolated_gn") &&
      !o.at("use_t_isolated_gn").as_bool(true)) {
    return fail("use_t_isolated_gn is false: this encoder's group norm is "
                "not the per-frame one implemented here");
  }
  if (o.contains("causal_encoder") && !o.at("causal_encoder").as_bool(true)) {
    return fail("causal_encoder is false: this encoder's temporal padding "
                "is not the causal one implemented here");
  }
  if (o.contains("use_3d_conv") && !o.at("use_3d_conv").as_bool(true)) {
    return fail("use_3d_conv is false: this encoder is not the 3D one "
                "implemented here");
  }
  // The clip length, the token drop and the latent whitening live in the
  // WRAPPER's config, not the source net's (see above for where each
  // layout keeps it).
  if (wrap.is_object()) {
    auto wo = wrap.as_object();
    if (wo.contains("vae_clip_length")) {
      out.clip_length = (int)wo.at("vae_clip_length").as_int(17);
    }
    if (wo.contains("vae_token_drop")) {
      out.token_drop = (int)wo.at("vae_token_drop").as_int(3);
    }
    if (wo.contains("vae_tile_size")) {
      out.tile_size = (int)wo.at("vae_tile_size").as_int(256);
    }
    if (wo.contains("vae_tile_overlap_min")) {
      out.tile_overlap_min =
          (int)wo.at("vae_tile_overlap_min").as_int(64);
    }
    auto reals = [&](const char* k, std::vector<float>& dst) {
      if (!wo.contains(k)) { return; }
      FlexData v = wo.at(k);
      if (!v.is_array()) { return; }
      auto a = v.as_array();
      std::vector<float> got;
      for (std::size_t i = 0; i < a.size(); ++i) {
        got.push_back((float)a.at(i).as_real(0.0));
      }
      if ((int)got.size() == out.z_channels) { dst = std::move(got); }
    };
    reals("latents_mean", out.latents_mean);
    reals("latents_std", out.latents_std);
  }

  if (!finish_config_(out, err)) { return false; }

  // ---- the single-frame image capability ------------------------------
  // Read LAST, because validating the slice needs `patch_t`, and that is
  // parsed above. It sits BESIDE the wrapper blob rather than inside it:
  // the source net is the same net, and what a `t1_direct` checkpoint
  // changed is which weights its decoder was left with. The entries are
  // bare strings -- "true" and "3" -- where `source_config` is JSON, so
  // they are read off the raw metadata rather than parsed as JSON.
  //
  // Only the single-file packing can carry it. The diffusers layout has
  // nowhere to put it that is not the source net's own config.json, and
  // inventing a key there would claim the reference publishes one.
  if (!comfy_file.empty()) {
    FlexData meta;
    if (comfy::read_metadata(comfy_file, meta, nullptr) && meta.is_object()) {
      auto mo = meta.as_object();
      auto str = [&](const char* k) -> std::string {
        if (!mo.contains(k)) { return {}; }
        const FlexData v = mo.at(k);   // as_string() is a VIEW: bind it
        return v.is_string() ? std::string(v.as_string("")) : std::string();
      };
      const std::string direct = str("h3_t1_direct");
      out.t1_direct = (direct == "true" || direct == "1");
      if (out.t1_direct) {
        // Absent, default to the pad the chunked path already discards,
        // which is where the released checkpoint puts it. Present and
        // outside the block, REFUSE rather than clamp: the file is
        // describing geometry this build did not parse, and quietly
        // decoding a different frame is the failure worth refusing.
        const std::string slice = str("h3_t1_output_slice");
        out.t1_output_slice =
            slice.empty() ? out.frame_pre_pad() : std::atoi(slice.c_str());
        if (out.t1_output_slice < 0 || out.t1_output_slice >= out.patch_t) {
          return fail(comfy_file + ": h3_t1_output_slice '" + slice +
                      "' is not one of the " + std::to_string(out.patch_t) +
                      " pixel frames a latent frame decodes to");
        }
      }
    }
  }
  return true;
}

SharedBuffer
MetalMiniMaxH3VideoVae::weight_(WeightSet& ws, const std::string& nm)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr) { return {}; }
  if (info->dtype == "BF16") {
    return ws.tensor(nm, _mc, WeightSet::Residency::Copied, _part);
  }
  return ws.derived(std::string(kKey) + nm, [&]() -> SharedBuffer {
    return to_bf16_(ws.src(), _mc, nm);
  }, _part);
}

MetalMiniMaxH3VideoVae::Linear
MetalMiniMaxH3VideoVae::linear_(WeightSet& ws, const std::string& nm)
{
  Linear l;
  // Stored in pieces (the diffusers layout): assembled whole.
  if (const auto parts =
          minimax_h3::vae_assembled_parts(ws.src(), nm, _cfg.head_dim);
      !parts.empty()) {
    const std::string key = std::string(kKey) + "asm|" + nm;
    l.w = assemble_bf16_(ws, _mc, key + ".weight", parts, ".weight", _part);
    l.b = assemble_bf16_(ws, _mc, key + ".bias", parts, ".bias", _part);
    return l;
  }
  l.b = weight_(ws, nm + ".bias");
  const MetalLlamaWeights& src = ws.src();
  const auto* si = src.info(nm + ".scales");
  const auto* ci = src.info(nm + ".weight");
  if (_quant_bits > 0 && si != nullptr && ci != nullptr &&
      si->shape.size() == 2 && ci->shape.size() == 2) {
    const long gcols = ci->shape[1];
    const long scols = si->shape[1];
    const long K = scols * (long)_quant_group;
    const int bits = K > 0 ? (int)(gcols * 32 / K) : 0;
    l.bits = (bits == 8) ? 8 : 4;
    l.codes  = ws.tensor(nm + ".weight", _mc, WeightSet::Residency::Copied,
                         _part);
    l.scales = weight_(ws, nm + ".scales");
    l.qbias  = weight_(ws, nm + ".biases");
    if (!l.codes.empty() && !l.scales.empty() && !l.qbias.empty()) {
      l.quantized = true;
      return l;
    }
    l.codes = {}; l.scales = {}; l.qbias = {};
  }
  l.w = weight_(ws, nm + ".weight");
  return l;
}

// A causal conv3d as a dense-GEMM weight. The 3x3x3 kernels flatten to
// [cout, 27*cin] ordered (kt, ky, kx, cin), which is the layout
// im2col_hwc_3x3x3_reflect_tiled emits; the 1x1x1 shortcut and quant
// convolutions are already a Linear and go through linear_() untouched.
MetalMiniMaxH3VideoVae::Conv3d
MetalMiniMaxH3VideoVae::conv3d_(WeightSet& ws, const std::string& nm,
                                bool hw)
{
  Conv3d c;
  c.name = nm;
  const auto* info = ws.src().info(nm + ".weight");
  if (info == nullptr || info->shape.size() != 5) { return c; }
  const auto& sh = info->shape;
  c.cout = (int)sh[0];
  c.cin  = (int)sh[1];
  const int kt = (int)sh[2], kh = (int)sh[3], kw = (int)sh[4];
  if (kt == 1 && kh == 1 && kw == 1) {
    c.spatial = false;
    c.k = c.cin;
    c.l = linear_(ws, nm);
    return c;
  }
  if (kt != 3 || kh != 3 || kw != 3) { return Conv3d{}; }
  c.spatial = true;
  c.k = 27 * c.cin;
  c.l.b = weight_(ws, nm + ".bias");
  // The kernel's output tile is 128 channels (or 16, at stride 1 only:
  // see hw_route_), and it always adds a bias.
  const bool hwio = hw && c.cout % 16 == 0 && !c.l.b.empty();
  SharedBuffer w = c3d_weight_(ws, nm, c.cin, c.cout, hwio);
  if (w.empty()) { return Conv3d{}; }
  (hwio ? c.whwio : c.l.w) = std::move(w);
  return c;
}

SharedBuffer
MetalMiniMaxH3VideoVae::c3d_weight_(WeightSet& ws, const std::string& nm,
                                    int cin, int cout, bool hwio)
{
  const std::string key =
      std::string(kKey) + (hwio ? "c3d-hwio|" : "c3d|") + nm;
  return ws.derived(key, [&]() -> SharedBuffer {
    // Read UNCACHED: the permuted bf16 copy below is what the model
    // keeps, and caching the f32 original next to it would hold a
    // second, four-times-larger copy of every convolution.
    SharedBuffer raw =
        ws.read(nm + ".weight", _mc, WeightSet::Residency::Copied);
    if (raw.empty()) { return {}; }
    const std::size_t n = (std::size_t)cout * cin * 27;
    // The released checkpoint stores these f32; Comfy-Org's repack
    // stores the same net fp16. Read the dtype rather than assuming,
    // because the permutation below indexes the source directly -- an
    // f32 read of fp16 bytes does not fail, it walks off the end (or,
    // with a shorter tensor, silently permutes garbage).
    const auto* wi = ws.src().info(nm + ".weight");
    const std::string dt = wi != nullptr ? wi->dtype : std::string();
    const std::size_t esz =
        dt == "F32" ? 4u : (dt == "F16" || dt == "BF16" ? 2u : 0u);
    if (esz == 0 || raw.byte_size() < n * esz) { return {}; }
    const auto* s32 = static_cast<const float*>(raw.contents());
    const auto* s16 = static_cast<const _Float16*>(raw.contents());
    const auto* sb16 = static_cast<const std::uint16_t*>(raw.contents());
    auto src = [&](std::size_t i) -> float {
      if (dt == "F32")  { return s32[i]; }
      if (dt == "F16")  { return (float)s16[i]; }
      return bf16_to_f32_(sb16[i]);
    };
    SharedBuffer out = _mc->make_shared_buffer(n * 2);
    if (out.empty()) { return {}; }
    auto* d = static_cast<std::uint16_t*>(out.contents());
    for (int o = 0; o < cout; ++o) {
      for (int t = 0; t < 3; ++t) {
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            const int tap = (t * 3 + ky) * 3 + kx;
            for (int i = 0; i < cin; ++i) {
              const std::size_t si =
                  ((((std::size_t)o * cin + i) * 3 + t) * 3 + ky) * 3 + kx;
              const std::size_t di =
                  hwio ? ((std::size_t)tap * cin + i) * cout + o
                       : ((std::size_t)o * 27 + tap) * cin + i;
              d[di] = f32_to_bf16_(src(si));
            }
          }
        }
      }
    }
    return out;
  }, _part);
}

SharedBuffer
MetalMiniMaxH3VideoVae::gemm_weight_(const Conv3d& c)
{
  if (!_ws) { return {}; }
  // Attributed to the encoder's part, like everything else it loads, so
  // releasing the encoder drops it too.
  const std::string saved = _part;
  _part = "encoder";
  SharedBuffer w = c3d_weight_(*_ws, c.name, c.cin, c.cout, false);
  _part = saved;
  return w;
}

bool
MetalMiniMaxH3VideoVae::load_encoder_(WeightSet& ws)
{
  const Config& c = _cfg;
  _enc_conv_in = conv3d_(ws, "encoder.conv_in", _use_hwconv);
  _enc_conv_out = conv3d_(ws, "encoder.conv_out", _use_hwconv);
  _quant_conv   = conv3d_(ws, "quant_conv", false);
  _enc_norm_w   = weight_(ws, "encoder.norm_out.weight");
  _enc_norm_b   = weight_(ws, "encoder.norm_out.bias");
  if (_enc_conv_in.empty() || _enc_conv_out.empty() || _quant_conv.empty() ||
      _enc_norm_w.empty() || _enc_norm_b.empty()) {
    return false;
  }
  const std::size_t levels = c.ch_mult.size();
  _enc_levels.clear();
  _enc_levels.resize(levels);
  int prev = c.base_ch;
  for (std::size_t i = 0; i < levels; ++i) {
    EncLevel& lv = _enc_levels[i];
    lv.space = c.space_down[i];
    lv.time  = c.time_down[i];
    lv.cin   = prev;
    lv.cout  = c.base_ch * c.ch_mult[i];
    const std::string base = "encoder.down." + std::to_string(i) + ".";
    for (int j = 0; j < c.res_blocks; ++j) {
      EncResnet r;
      const std::string p = base + "block." + std::to_string(j) + ".";
      r.cin  = (j == 0) ? lv.cin : lv.cout;
      r.cout = lv.cout;
      r.n1w = weight_(ws, p + "norm1.weight");
      r.n1b = weight_(ws, p + "norm1.bias");
      r.n2w = weight_(ws, p + "norm2.weight");
      r.n2b = weight_(ws, p + "norm2.bias");
      r.c1  = conv3d_(ws, p + "conv1", _use_hwconv);
      r.c2  = conv3d_(ws, p + "conv2", _use_hwconv);
      if (r.cin != r.cout) {
        r.skip = conv3d_(ws, p + "nin_shortcut", false);
        if (r.skip.empty()) { return false; }
      }
      if (r.n1w.empty() || r.n1b.empty() || r.n2w.empty() || r.n2b.empty() ||
          r.c1.empty() || r.c2.empty()) {
        return false;
      }
      lv.res.push_back(std::move(r));
    }
    if (lv.space * lv.time > 1) {
      lv.down = conv3d_(ws, base + "downsample.conv", _use_hwconv);
      if (lv.down.empty()) { return false; }
    }
    prev = lv.cout;
  }
  return true;
}

bool
MetalMiniMaxH3VideoVae::load_decoder_(WeightSet& ws)
{
  const Config& cfg = _cfg;
  _post_quant = linear_(ws, "post_quant_conv");
  _proj_in    = linear_(ws, "decoder.x_embedder");
  _proj_out   = linear_(ws, "decoder.proj_out");
  _register_tokens = weight_(ws, "decoder.register_tokens");
  _norm_out_w = weight_(ws, "decoder.norm_out.weight");
  _norm_out_b = weight_(ws, "decoder.norm_out.bias");
  if (_post_quant.empty() || _proj_in.empty() || _proj_out.empty() ||
      _register_tokens.empty() || _norm_out_w.empty() ||
      _norm_out_b.empty()) {
    return false;
  }
  _blocks.clear();
  _blocks.resize((std::size_t)cfg.n_layers);
  for (int i = 0; i < cfg.n_layers; ++i) {
    Block& b = _blocks[(std::size_t)i];
    b.n1  = weight_(ws, blk_(i, "norm1.weight"));
    b.n2  = weight_(ws, blk_(i, "norm2.weight"));
    b.s1  = weight_(ws, blk_(i, "scale1"));
    b.s2  = weight_(ws, blk_(i, "scale2"));
    b.qkv = linear_(ws, blk_(i, "attn.to_qkv"));
    b.out = linear_(ws, blk_(i, "attn.to_out"));
    b.w1  = linear_(ws, blk_(i, "ff.w1"));
    b.w2  = linear_(ws, blk_(i, "ff.w2"));
    if (b.n1.empty() || b.n2.empty() || b.s1.empty() || b.s2.empty() ||
        b.qkv.empty() || b.out.empty() || b.w1.empty() || b.w2.empty()) {
      return false;
    }
  }
  if (_mc->session() != nullptr) {
    _mc->session()->log_debug(
        fmt("MetalMiniMaxH3VideoVae: ViT decoder, {} blocks, dim {}, "
            "{} heads x {}, patch {}x{}x{}",
            cfg.n_layers, cfg.dim, cfg.n_heads, cfg.head_dim, cfg.patch_t,
            cfg.patch, cfg.patch));
  }
  return true;
}

std::size_t
MetalMiniMaxH3VideoVae::held_bytes(const std::string& vae_dir, Half half)
{
  auto wts = MetalLlamaWeights::open_model(resolve_vae_dir(vae_dir));
  if (!wts.has_value()) { return 0; }
  std::size_t b = 0;
  for (const std::string& nm : wts->tensor_names()) {
    const bool dec = nm.rfind("decoder.", 0) == 0 ||
                     nm.rfind("post_quant_conv.", 0) == 0;
    const bool enc = nm.rfind("encoder.", 0) == 0 ||
                     nm.rfind("quant_conv.", 0) == 0;
    if (!(half == Half::kDecoder ? dec : enc)) { continue; }
    // Training's alone: nothing decodes with it.
    if (nm == "decoder.mask_token") { continue; }
    const auto* ti = wts->info(nm);
    if (ti == nullptr) { continue; }
    std::size_t n = 1;
    for (auto d : ti->shape) { n *= (std::size_t)d; }
    // Held bf16 whatever the file stores, or as packed when packed.
    const std::size_t stored = (std::size_t)ti->nbytes;
    b += ti->dtype == "F32" || ti->dtype == "F16" || ti->dtype == "BF16"
             ? n * 2
             : stored;
  }
  return b;
}

std::size_t
MetalMiniMaxH3VideoVae::decode_working_bytes(const Config& c, int LT, int lh,
                                             int lw)
{
  const int tcs = c.tokens_per_chunk();
  if (LT <= 0 || lh <= 0 || lw <= 0 || tcs <= 0) { return 0; }
  const int td = c.token_drop, tr = c.patch_t;
  const int span = tcs + c.token_overlap();
  const int num_tokens = LT + td;
  const int pad_tokens = ((-num_tokens % tcs) + tcs) % tcs;
  const std::size_t ZC = (std::size_t)c.z_channels;
  const std::size_t OC = (std::size_t)c.out_channels;
  const std::size_t H = (std::size_t)lh * c.patch;
  const std::size_t W = (std::size_t)lw * c.patch;
  const std::size_t cf = (std::size_t)span * tr;
  // The padded latent and one chunk of it.
  std::size_t b = ZC * (LT + pad_tokens) * lh * lw * 2;
  b += ZC * span * lh * lw * 2;
  // A chunk: the stitch, and its tiles' decodes -- the previous row of
  // tiles and the one being decoded (decode_tiled_ stitches each as it
  // finishes).
  const auto ys = minimax_h3::split_tiles((int)H, c.tile_size,
                                          c.tile_overlap_min, c.patch);
  const auto xs = minimax_h3::split_tiles((int)W, c.tile_size,
                                          c.tile_overlap_min, c.patch);
  std::size_t row_most = 0, widest = 0;
  for (int th : ys.length) {
    std::size_t row = 0;
    for (int tw : xs.length) {
      row += OC * cf * (std::size_t)th * tw * 2;
      widest = std::max(widest, (std::size_t)th * tw);
    }
    row_most = std::max(row_most, row);
  }
  const std::size_t tiles = 2 * row_most;
  const bool one = ys.length.size() <= 1 && xs.length.size() <= 1;
  if (one) { widest = H * W; }
  b += (one ? 0 : tiles + ZC * span * (widest / (c.patch * c.patch)) * 2);
  b += OC * cf * H * W * 2;
  // The ViT scratch at the widest tile's rows (ensure_scratch_).
  const std::size_t voxels =
      (std::size_t)span * (widest / ((std::size_t)c.patch * c.patch));
  const std::size_t R = voxels + (std::size_t)c.n_register + 1;
  const std::size_t D = (std::size_t)c.dim, F = (std::size_t)c.ffn_inner;
  b += R * (std::size_t)(3 * c.rope_freqs()) * sizeof(float) * 2;
  b += R * D * 2 * 7;
  b += R * std::max(3 * D, F) * 2 + R * 2 * F * 2;
  b += R * (std::size_t)c.patch_elems() * 2 + voxels * ZC * 2;
  // The cross-fade partner.
  b += OC * (std::size_t)(c.token_overlap() * tr) * H * W * 2;
  return b;
}

std::size_t
MetalMiniMaxH3VideoVae::decode_video_bytes(const Config& c, int LT, int lh,
                                           int lw)
{
  if (LT <= 0 || lh <= 0 || lw <= 0) { return 0; }
  const std::size_t frames =
      (std::size_t)std::max(0, decoded_frames_for(c, LT));
  return (std::size_t)c.out_channels * frames * lh * c.patch * lw *
             c.patch * 2 +
         decode_working_bytes(c, LT, lh, lw);
}

MetalMiniMaxH3VideoVae::~MetalMiniMaxH3VideoVae() = default;

std::unique_ptr<MetalMiniMaxH3VideoVae>
MetalMiniMaxH3VideoVae::load(const std::string& vae_dir, MetalCompute* mc,
                             const Config& cfg)
{
  return load(WeightSet::open(resolve_vae_dir(vae_dir), nullptr), mc, cfg);
}

std::unique_ptr<MetalMiniMaxH3VideoVae>
MetalMiniMaxH3VideoVae::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                             const Config& cfg)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  WeightSet& ws = *ws_in;
  auto m =
      std::unique_ptr<MetalMiniMaxH3VideoVae>(new MetalMiniMaxH3VideoVae());
  m->_ws = std::move(ws_in);
  m->_mc = mc;
  m->_cfg = cfg;

  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_fn_gemm      = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_rms       = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_rms_heads = m->_lib_rope.function("rms_norm_heads_strided_f16");
  m->_fn_trope = m->_lib_rope.function("transpose_rope_half_part_ftab_f16");
  m->_fn_gated     = m->_lib_elt.function("gated_residual_f16");
  // The decoder's FFN is `w2(silu(gate) * up)` with w1 emitting
  // [GATE | up] -- gate FIRST. (The value-first spelling is the other
  // convention for the same weights and silently halves the FFN into
  // nonsense rather than failing.)
  //
  // Same swap as the DiT's fc1: diffusers implements [up; gate] and
  // `scripts/convert_minimax_h3_to_diffusers.py` reorders the halves as
  // it converts, stating it for this tensor too -- "the two halves of
  // `w1` are swapped because diffusers' SwiGLU reads [up; gate] where
  // the reference stores [gate; up]". Which is independent confirmation
  // of the gate-first reading here, from the same file.
  m->_fn_swiglu    = m->_lib_elt.function("swiglu_split_gate_first_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_ln        = m->_lib_elt.function("layer_norm_affine_f16");
  m->_fn_bias_add  = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_gn_frames = m->_lib_elt.function("group_norm_frames_f16");
  m->_fn_im2col_r  =
      m->_lib_elt.function("im2col_hwc_3x3x3_reflect_tiled_f16");
  m->_fn_residual  = m->_lib_elt.function("residual_add_f16");
  {
    metal_compute::ComputeLibrary sdpa = mc->load_library("sdpa_bf16");
    m->_fn_sdpa = sdpa.function("sdpa_full_f16");
  }
  if (!m->_fn_gemm.valid() || !m->_fn_rms.valid() ||
      !m->_fn_rms_heads.valid() || !m->_fn_trope.valid() ||
      !m->_fn_gated.valid() || !m->_fn_swiglu.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_ln.valid() ||
      !m->_fn_bias_add.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_gn_frames.valid() || !m->_fn_im2col_r.valid() ||
      !m->_fn_residual.valid()) {
    return nullptr;
  }
  // The coalesced three-pass GroupNorm, on every GPU (see
  // group_norm_frames_stats_f16 for why the one-pass kernel is slow).
  m->_fn_gn_stats = m->_lib_elt.function("group_norm_frames_stats_f16");
  m->_fn_gn_fin   = m->_lib_elt.function("group_norm_frames_finalize_f16");
  m->_fn_gn_apply = m->_lib_elt.function("group_norm_frames_apply_f16");
  m->_gn3 = m->_fn_gn_stats.valid() && m->_fn_gn_fin.valid() &&
            m->_fn_gn_apply.valid() &&
            std::getenv("VPIPE_H3_VVAE_GN_ONEPASS") == nullptr;
  // M5: the encoder's 3x3x3 convs on the hardware convolution op. Decided
  // here because the encoder's loader builds each conv's weight in the
  // layout of its route. It reads frames the norm REFLECT-PADS, so it
  // needs the three-pass norm too. VPIPE_VAE_NO_HWCONV, as the other VAEs.
  if (mc->supports_matrix_cores() && m->_gn3 &&
      std::getenv("VPIPE_VAE_NO_HWCONV") == nullptr) {
    m->_lib_convhw = mc->load_library("conv2d_mma_bf16");
    m->_fn_pad = m->_lib_elt.function("reflect_pad_hwc_f16");
    m->_use_hwconv = m->_fn_pad.valid();
    for (int st = 0; st < 2; ++st) {
      // The 16-channel entry is stride 1 only.
      const int n = (st == 0) ? kC3C16 : kC3Tail;
      for (int k = 0; k <= n; ++k) {
        const std::string nm = std::string("conv3d_hw_causal_s") +
                               (st == 0 ? "1_" : "2_") + kC3Names[k] +
                               "_f16";
        m->_fn_c3hw[st][k] = m->_lib_convhw.function(nm);
        m->_use_hwconv = m->_use_hwconv && m->_fn_c3hw[st][k].valid();
      }
    }
  }
  m->_lib_attn = mc->load_library("attn_steel");
  m->_attn_p = mc->make_shared_buffer(sizeof(float) * 64);
  m->_steel_ok = m->_lib_attn.valid() && !m->_attn_p.empty() &&
                 std::getenv("VPIPE_H3_NO_STEEL_ATTN") == nullptr;
  // M5: the matrix-core (NAX) flash attention, same contract as the ALU
  // steel kernel with bq/bk 64/32 instead of 32/16. This tower is head_dim
  // 64 AND bf16, a combination NAX did not carry until it was instantiated
  // for this model -- the f16 bd64 and bf16 bd128 entries are the ones the
  // vision towers and the image DiTs use. VPIPE_H3_NO_ATTN_NAX forces ALU.
  if (m->_steel_ok && mc->supports_matrix_cores() &&
      std::getenv("VPIPE_H3_NO_ATTN_NAX") == nullptr) {
    m->_lib_attn_nax = mc->load_library("attn_steel_nax");
    m->_attn_nax = m->_lib_attn_nax.valid();
  }

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
  }

  // The same dequant-once + dense matmul2d route the DiT takes; see the
  // long note on MetalMiniMaxH3Transformer's loader for why the dequant is
  // a separate pass rather than fused into a quantized matmul2d.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_H3_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid();
    m->_splitk.load(mc, m->_lib_dense_mma, m->_lib_elt);
    if (m->_use_mma2 && m->_quant_bits > 0) {
      const std::string g = "g" + std::to_string(m->_quant_group);
      m->_lib_dequant = mc->load_library("affine_dequant_bf16");
      m->_fn_dequant4 = m->_lib_dequant.function("affine_dequant_w4" + g);
      m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant_w8" + g);
      if (!m->_fn_dequant4.valid() || !m->_fn_dequant8.valid()) {
        m->_use_mma2 = false;
      }
    }
    if (const char* e = std::getenv("VPIPE_H3_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
  }

  // THE DECODER LOADS AT THE FIRST DECODE (load_decoder_), as the
  // encoder does at the first encode: an encoding stage -- vae-encode,
  // video-ref-encoder -- held 4.6 GB of decoder it never ran. A
  // checkpoint without one is still refused here, by its headers.
  if (ws.src().info("post_quant_conv.weight") == nullptr ||
      ws.src().info("decoder.x_embedder.weight") == nullptr ||
      ws.src().info("decoder.proj_out.weight") == nullptr) {
    return nullptr;
  }
  return m;
}

int
MetalMiniMaxH3VideoVae::row_band_(int widest) const
{
  // VPIPE_H3_VVAE_ROW_BAND forces a band, which is how a shape that
  // fits can be tested against its own banded self.
  //
  // CLAMPED to what the band is for, because it is otherwise a way to
  // reintroduce exactly the overflow this function exists to prevent:
  // a forced band wide enough puts `M * N` past the int it is narrowed
  // to at the bias fold, which then reads as a huge unsigned `total`
  // AND as a 1-D grid of the same wrong size. A forced band may be
  // SMALLER than the computed one (that is the A/B it is for) and never
  // larger.
  static const int kForced = [] {
    const char* e = std::getenv("VPIPE_H3_VVAE_ROW_BAND");
    return e != nullptr ? std::atoi(e) : 0;
  }();
  if (widest <= 0) { return 1 << 30; }
  // The last row a 32-bit BYTE offset reaches, floored to the 128-row
  // tile the mma kernels step in: a band ending mid-tile would still
  // hand one tile a base past the line.
  const long long lim = (((long long)1 << 31) - 1) / ((long long)widest * 2);
  long long band = (lim / 128) * 128;
  if (band < 128) { band = 128; }
  if (band > (1 << 30)) { band = 1 << 30; }
  if (kForced > 0 && (long long)kForced < band) { band = kForced; }
  return (int)band;
}

void
MetalMiniMaxH3VideoVae::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                              std::size_t x_off, const Linear& l,
                              const SharedBuffer& y, std::size_t y_off, int M,
                              int N, int K)
{
  _flops.gemm(M, N, K);
  gemm_uncounted_(enc, x, x_off, l, y, y_off, M, N, K);
}

void
MetalMiniMaxH3VideoVae::gemm_uncounted_(ComputeEncoder& enc,
                                        const SharedBuffer& x,
                                        std::size_t x_off, const Linear& l,
                                        const SharedBuffer& y,
                                        std::size_t y_off, int M, int N,
                                        int K)
{
  // Rows in bands narrow enough that no dispatch addresses past 2^31,
  // each rebased through the buffer offset so the kernel's own
  // arithmetic stays small. Below the line -- the decode, and every
  // tiled encode -- the band IS M and this is the single pass it was.
  //
  // THE WIDEST OF THE TWO ROW STRIDES, not the destination's. The mma
  // kernels wrap EVERY operand in int32 extents, so the X operand is
  // `band * K * 2` bytes where the destination is `band * N * 2`, and
  // banding on N alone leaves X past the line whenever K > N. Nothing
  // reaches it today -- the only GEMMs that see a large unbanded M are
  // the 1x1x1 convs, where K = cin <= N = cout -- but the decoder's
  // `w2` is K 8192 against N 2048, so the shape that breaks it is
  // already in the model and is merely never handed enough rows.
  // Banding on the max costs nothing when K <= N, which is every case
  // that runs.
  const int band = row_band_(std::max(N, K));
  if (M > band) {
    for (int m0 = 0; m0 < M; m0 += band) {
      gemm_uncounted_(enc, x, x_off + (std::size_t)m0 * K, l, y,
                      y_off + (std::size_t)m0 * N, std::min(band, M - m0),
                      N, K);
    }
    return;
  }
  const bool bias = !l.b.empty();
  if (gemm_mma_(enc, x, x_off, l, y, y_off, M, N, K)) {
    if (bias) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, y_off * 2);
      enc.set_buffer(1, l.b);
      enc.set_constant(2, N);
      enc.set_constant(3, M * N);
      enc.dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
    }
    return;
  }
  if (l.quantized) {
    enc.set_function(l.bits == 8 ? _fn_qmm8 : _fn_qmm4);
    enc.set_buffer(0, l.codes);
    enc.set_buffer(1, l.scales);
    enc.set_buffer(2, l.qbias);
    enc.set_buffer(3, x, x_off * 2);
    enc.set_buffer(4, y, y_off * 2);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.set_constant(7, M);
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32), (unsigned)(((M + 31) / 32) * 2),
                  2}, {32, 2, 2});
    if (bias) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, y_off * 2);
      enc.set_buffer(1, l.b);
      enc.set_constant(2, N);
      enc.set_constant(3, M * N);
      enc.dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x, x_off * 2);
  enc.set_buffer(1, l.w);
  enc.set_buffer(2, bias ? l.b : l.w);
  enc.set_buffer(3, y, y_off * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias ? 1 : 0);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

// The DiT's gemm_mma_ without the route enum: this tower has one hot shape
// per stage rather than four competing ones, and it runs once per clip, so
// the shared K-only tile rule is used directly instead of a tuner. Same
// scratch discipline -- one [N, K] bf16 buffer, grown on demand and reused,
// safe because the dequant/matmul pairs are encoded in order and Metal
// serializes the write-after-read on it.
bool
MetalMiniMaxH3VideoVae::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                                  std::size_t x_off, const Linear& l,
                                  const SharedBuffer& y, std::size_t y_off,
                                  int M, int N, int K)
{
  if (!_use_mma2 || M < _mma_min_m || N < 16) { return false; }
  const SharedBuffer* wdense = nullptr;
  if (l.quantized) {
    const metal_compute::ComputeFunction& dq =
        (l.bits == 8) ? _fn_dequant8 : _fn_dequant4;
    if (!dq.valid()) { return false; }
    // The dequant kernel's K contract -- see the note in the DiT's
    // route_ok_. A width that is not a whole number of packed words AND of
    // quant groups would under-write each row's tail, so it stays on steel.
    const int per_word = (l.bits == 8) ? 4 : 8;
    if (K % per_word != 0 || K % _quant_group != 0) { return false; }
    const std::size_t need = (std::size_t)N * (std::size_t)K * 2;
    if (_w_deq.empty() || _w_deq.byte_size() < need) {
      _w_deq = _mc->make_shared_buffer(need);
      if (_w_deq.empty()) { return false; }
    }
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
  // Deep contraction: split it across planes and fold. Declines unless a
  // tuned entry says otherwise, which for this tower is the usual answer --
  // see the tune loop in encode() for why the band size decides it.
  if (_splitk.encode(_mc, enc, x, *wdense, y, K, N, M, x_off, y_off)) {
    return true;
  }
  bool wide = mma_use_wide_tile(N, K);
  if (_force_tile != 0) {
    wide = (_force_tile == 2);
  } else {
    for (const TileTune& t : _tile_tuned) {
      if (t.N == N && t.K == K && t.M == M) { wide = t.wide; break; }
    }
  }
  const int RN = wide ? 256 : 128;
  enc.set_function(wide ? _fn_dense_mma_deep : _fn_dense_mma);
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

void
MetalMiniMaxH3VideoVae::build_rope_(int T, int h, int w, SharedBuffer& cos_out,
                                    SharedBuffer& sin_out) const
{
  // Coordinates are length-normalized to [-1, 1) per axis -- CELL
  // CENTRES, `2*(i + 0.5)/size - 1` -- and the angle carries an explicit
  // 2*pi. So the grid depends on the tile's extent, not on absolute
  // position: the same voxel decoded inside a different tile size sees
  // different angles.
  const int F = _cfg.rope_freqs();
  const int half = 3 * F;
  const int voxels = T * h * w;
  const int rows = voxels + _cfg.n_register + 1;
  auto* cb = static_cast<float*>(cos_out.contents());
  auto* sb = static_cast<float*>(sin_out.contents());
  std::vector<double> inv((std::size_t)F);
  for (int i = 0; i < F; ++i) {
    // 1 / theta^arange(0, 1, 2*num_axes/rope_dim): the exponent grid is
    // built by STEP, not by count, so it is i * (6 / rope_dim).
    inv[(std::size_t)i] =
        1.0 / std::pow(_cfg.rope_theta,
                       (double)i * 6.0 / (double)_cfg.rope_dim());
  }
  // Diagnostic only: neutralize the rope (cos 1, sin 0) so a mismatch
  // can be attributed to the rope or to the rest of the block.
  const bool no_rope = std::getenv("VPIPE_H3_VVAE_NO_ROPE") != nullptr;
  const int sizes[3] = {T, h, w};
  for (int r = 0; r < rows; ++r) {
    double coord[3] = {0.0, 0.0, 0.0};
    if (r < voxels) {
      const int idx[3] = {r / (h * w), (r / w) % h, r % w};
      for (int a = 0; a < 3; ++a) {
        coord[a] = 2.0 * (((double)idx[a] + 0.5) / (double)sizes[a]) - 1.0;
      }
    }
    // The register tokens and the trailing zero token all sit at
    // position 0, which is what `coord` already is.
    for (int a = 0; a < 3; ++a) {
      for (int i = 0; i < F; ++i) {
        const double ang =
            2.0 * M_PI * coord[a] * inv[(std::size_t)i];
        const std::size_t o = (std::size_t)r * half + (std::size_t)(a * F + i);
        cb[o] = no_rope ? 1.0f : (float)std::cos(ang);
        sb[o] = no_rope ? 0.0f : (float)std::sin(ang);
      }
    }
  }
}

bool
MetalMiniMaxH3VideoVae::ensure_scratch_(int rows, int voxels)
{
  if (_s.rows == rows) { return true; }
  const Config& c = _cfg;
  const std::size_t R = (std::size_t)rows, D = (std::size_t)c.dim;
  auto mk = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  Scratch s;
  s.rows = rows;
  const std::size_t half = (std::size_t)(3 * c.rope_freqs());
  s.rcos = _mc->make_shared_buffer(R * half * sizeof(float));
  s.rsin = _mc->make_shared_buffer(R * half * sizeof(float));
  s.x   = mk(R * D);
  s.nm  = mk(R * D);
  // Sized for the WIDER of its two uses. s.qkv holds the fused q|k|v
  // (3*dim = 6144) and is then reused for the SwiGLU output (ffn_inner =
  // 8192), which is larger here -- the DiT's ordering is the other way
  // round, and carrying that assumption over writes past the end of the
  // buffer and feeds the second feed-forward GEMM garbage.
  s.qkv = mk(R * (std::size_t)std::max(3 * c.dim, c.ffn_inner));
  s.qh  = mk(R * D);
  s.kh  = mk(R * D);
  s.vh  = mk(R * D);
  s.oh  = mk(R * D);
  s.ob  = mk(R * D);
  s.ff  = mk(R * 2 * (std::size_t)c.ffn_inner);
  s.patches = mk(R * (std::size_t)c.patch_elems());
  s.zt  = mk((std::size_t)voxels * (std::size_t)c.z_channels);
  // The q/k norms have NO affine (elementwise_affine=False), so they
  // multiply by 1. A ones vector keeps the shared per-head RMS kernel
  // rather than forking it for a missing gamma.
  s.ones = mk((std::size_t)c.head_dim);
  if (s.rcos.empty() || s.rsin.empty() || s.x.empty() || s.nm.empty() ||
      s.qkv.empty() || s.qh.empty() || s.kh.empty() || s.vh.empty() ||
      s.oh.empty() || s.ob.empty() || s.ff.empty() || s.patches.empty() ||
      s.zt.empty() || s.ones.empty()) {
    return false;
  }
  auto* o = static_cast<std::uint16_t*>(s.ones.contents());
  for (int i = 0; i < c.head_dim; ++i) { o[i] = f32_to_bf16_(1.0f); }
  _s = std::move(s);
  return true;
}

// ---- the ANE tier ---------------------------------------------------

std::size_t
MetalMiniMaxH3VideoVae::ane_runtime_bytes(const Config& c) noexcept
{
  // One decode() tile: a chunk's latent frames over one spatial tile, plus
  // the register and zero rows.
  const int t = c.tokens_per_chunk() + c.token_overlap();
  const int e = std::max(1, c.tile_size / std::max(1, c.patch));
  const int rows = t * e * e + c.n_register + 1;
  const std::size_t biases =
      (std::size_t)(2 * c.ffn_inner + c.dim) * 2 * 2;   // slots + staging
  return AneFeedForward::runtime_bytes(c.dim, c.ffn_inner, rows,
                                       ane_chunk_rows()) + biases;
}

int
MetalMiniMaxH3VideoVae::ane_chunk_rows() noexcept
{
  constexpr const char* kEnv = "VPIPE_H3_VVAE_ANE_CHUNK";
  return std::getenv(kEnv) != nullptr ? AneFeedForward::chunk_rows(kEnv)
                                      : 1024;
}

bool
MetalMiniMaxH3VideoVae::ane_setup_(int rows)
{
  if (_ane_tried) { return _ane != nullptr; }
  _ane_tried = true;
  if (_mc->session() == nullptr || rows <= 0) { return false; }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "minimax-h3-vae";
  o.hidden  = _cfg.dim;
  o.ffn     = _cfg.ffn_inner;
  o.seq     = rows;
  o.rows    = _cfg.ane_rows;
  o.chunk   = ane_chunk_rows();
  o.biases  = true;
  o.profile = std::getenv("VPIPE_H3_VVAE_ANE_PROFILE") != nullptr;
  _ane = AneFeedForward::create(o);
  return _ane != nullptr;
}

bool
MetalMiniMaxH3VideoVae::ane_eligible_(int L, const Block& b)
{
  if (_ane == nullptr) { return false; }
  const int cap = (_cfg.ane_layers > 0)
                      ? std::min(_cfg.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  if (L >= cap) { return false; }
  auto stageable = [](const Linear& l) {
    if (l.empty()) { return false; }
    if (!l.quantized) { return true; }
    return (l.bits == 4 || l.bits == 8) && !l.scales.empty() &&
           !l.qbias.empty();
  };
  const bool ok = stageable(b.w1) && stageable(b.w2);
  if (!ok && !_ane_skip_warned && _mc->session() != nullptr) {
    _ane_skip_warned = true;
    _mc->session()->warn(fmt(
        "minimax-h3-vae: block {} has no fp16 or 4/8-bit affine feed-forward "
        "to stage, so it (and any like it) keeps the GPU feed-forward", L));
  }
  return ok;
}

void
MetalMiniMaxH3VideoVae::ane_stage_(int L, const Block& b)
{
  auto view = [](const Linear& l, std::size_t offset) {
    AneFfnSource s;
    s.w         = &l.w;
    s.codes     = &l.codes;
    s.scales    = &l.scales;
    s.qbias     = &l.qbias;
    s.quantized = l.quantized;
    s.bits      = l.bits;
    s.offset    = offset;
    s.bias      = l.b.empty() ? nullptr : &l.b;
    return s;
  };
  _ane->stage(L, view(b.w1, 0), view(b.w1, (std::size_t)_cfg.ffn_inner),
              view(b.w2, 0), _quant_group);
}

SharedBuffer
MetalMiniMaxH3VideoVae::decode(const SharedBuffer& z, int T, int h, int w,
                               std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  const int D = c.dim, NH = c.n_heads, HD = c.head_dim, FF = c.ffn_inner;
  const int ZC = c.z_channels, PE = c.patch_elems();
  if (T <= 0 || h <= 0 || w <= 0) { return fail("empty latent grid"); }
  const int voxels = T * h * w;
  const int rows = voxels + c.n_register + 1;
  if (z.byte_size() < (std::size_t)ZC * voxels * 2) {
    return fail("latents are smaller than z_channels * T * h * w");
  }
  if (!_ws) { return fail("no weight set"); }
  if (!_ws->ensure_part("decoder", [&]() {
        _part = "decoder";
        const bool ok = load_decoder_(*_ws);
        _part.clear();
        return ok;
      })) {
    return fail("the decoder half of this checkpoint failed to load");
  }
  if (!ensure_scratch_(rows, voxels)) {
    return fail("activation allocation failed (out of GPU memory)");
  }
  Scratch& s = _s;
  build_rope_(T, h, w, s.rcos, s.rsin);

  // Channel-first [C, T, h, w] -> one row per voxel [T*h*w, C]. On the
  // host: it is a transpose of a few megabytes at tile sizes, and the
  // GPU alternative would be a kernel that exists only for this.
  {
    const auto* src = static_cast<const std::uint16_t*>(z.contents());
    auto* dst = static_cast<std::uint16_t*>(s.zt.contents());
    for (int v = 0; v < voxels; ++v) {
      for (int ch = 0; ch < ZC; ++ch) {
        dst[(std::size_t)v * ZC + ch] =
            src[(std::size_t)ch * voxels + (std::size_t)v];
      }
    }
  }

  const float scale = 1.0f / std::sqrt((float)HD);
  // As in the DiT: resolve the kernel BEFORE the tiles, because a param
  // block filled for one tile and a kernel compiled for the other is a
  // wrong answer, not a fallback. Dropping to the ALU steel kernel (rather
  // than to the scalar sdpa the validity check below would reach) keeps the
  // failure a small one.
  if (_attn_nax && _attn_rows != rows) {
    metal_compute::FunctionConstants probe;
    probe.set_bool(200, (rows % 64) == 0).set_bool(201, (rows % 32) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false);
    if (!_lib_attn_nax.function("attn_steel_nax_h_bd64_bf16", probe).valid()) {
      _attn_nax = false;
    }
  }
  const int A_BQ = _attn_nax ? 64 : 32;
  const int A_BK = _attn_nax ? 32 : 16;
  bool use_steel = _steel_ok;
  if (use_steel && _attn_rows != rows) {
    struct P {
      int B, H, D, qL, kL, gqa_factor;
      float scale;
      int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
      std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
    };
    auto* p = static_cast<P*>(_attn_p.contents());
    p->B = 1; p->H = NH; p->D = HD;
    p->qL = rows; p->kL = rows;
    p->gqa_factor = 1; p->scale = scale;
    p->NQ = (rows + A_BQ - 1) / A_BQ; p->NK = (rows + A_BK - 1) / A_BK;
    p->NQ_aligned = rows / A_BQ; p->NK_aligned = rows / A_BK;
    p->qL_rem = rows - p->NQ_aligned * A_BQ;
    p->kL_rem = rows - p->NK_aligned * A_BK;
    p->qL_off = 0;
    p->Q_strides[0] = (std::int64_t)NH * rows * HD;
    p->Q_strides[1] = (std::int64_t)rows * HD;
    p->Q_strides[2] = HD;
    for (int i = 0; i < 3; ++i) {
      p->K_strides[i] = p->Q_strides[i];
      p->V_strides[i] = p->Q_strides[i];
      p->O_strides[i] = p->Q_strides[i];
    }
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (rows % A_BQ) == 0).set_bool(201, (rows % A_BK) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false);
    _fn_attn = _attn_nax
                   ? _lib_attn_nax.function("attn_steel_nax_h_bd64_bf16", fc)
                   : _lib_attn.function("attn_steel_h_bd64_bf16", fc);
    _attn_rows = rows;
  }
  if (use_steel) { use_steel = _fn_attn.valid(); }

  // The ANE feed-forward tier, armed once from the first tile's rows.
  const bool ane_on = _cfg.ane_ffn && ane_setup_(rows);
  using Clk = std::chrono::steady_clock;
  _flops.begin();
  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // 1. post_quant_conv is a 1x1x1 convolution, i.e. a Linear over the
    // channel axis, then the patch embedding.
    gemm_(enc, s.zt, 0, _post_quant, s.nm, 0, voxels, ZC, ZC);
    gemm_(enc, s.nm, 0, _proj_in, s.x, 0, voxels, D, ZC);
    // 2. The suffix rows: `n_register` LEARNED register tokens and one
    // all-zero token. They never reach the patch projection, but they
    // are attended over by every real token, so they are part of the
    // function and not padding -- and s.x is reused across calls, so
    // leaving them uninitialized leaves the previous decode's tail in
    // the attention. Written on the host because the projection GEMM
    // above writes rows [0, voxels) only, so the ranges are disjoint and
    // the host store lands before the stream is committed.
    std::memcpy(static_cast<std::uint16_t*>(s.x.contents()) +
                    (std::size_t)voxels * D,
                _register_tokens.contents(),
                (std::size_t)c.n_register * D * 2);
    std::memset(static_cast<std::uint16_t*>(s.x.contents()) +
                    (std::size_t)(voxels + c.n_register) * D,
                0, (std::size_t)D * 2);

    for (int L = 0; L < c.n_layers; ++L) {
      const Block& b = _blocks[(std::size_t)L];
      // Stage this block's feed-forward for the ANE now, on its worker, so
      // the conversion runs under the attention encoded below.
      const AneFeedForward::Plan ane_plan =
          ane_on && ane_eligible_(L, b) ? _ane->plan_block()
                                        : AneFeedForward::Plan::kGpu;
      const bool ane_block = ane_plan == AneFeedForward::Plan::kSplit;
      const bool ane_probe = ane_plan == AneFeedForward::Plan::kProbe;
      if ((ane_block || ane_probe) && (L == 0 || _ane->needs_barrier())) {
        // A measured layer after unsplit work -- GPU-mode layers, or the
        // embedding ahead of layer 0: drain it first. See
        // AneFeedForward::needs_barrier.
        enc.end();
        _flops.flush(stream);
        stream.commit().wait();
        enc = stream.begin_compute();
      }
      if (ane_block) { ane_stage_(L, b); }
      enc.set_function(_fn_rms);
      enc.set_buffer(0, s.x); enc.set_buffer(1, b.n1); enc.set_buffer(2, s.nm);
      enc.set_constant(3, D); enc.set_constant(4, c.norm_eps);
      enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
      gemm_(enc, s.nm, 0, b.qkv, s.qkv, 0, rows, 3 * D, D);
      // Per-head RMS with NO affine, then the partial rotate-half rope
      // over 48 of the 64 head channels. Both read straight out of the
      // fused projection.
      // The fused to_qkv is grouped PER HEAD -- [h0(q,k,v) | h1(q,k,v) |
      // ...] -- so a head sits 3*HD apart and q/k/v are HD apart WITHIN
      // it. Reading it as [all q | all k | all v] gives each head a
      // blend of its own q, k and v: attention then relates no token to
      // its neighbours, and every 16x16 patch decodes on its own, which
      // reads as a tile grid rather than as noise.
      for (int qk = 0; qk < 2; ++qk) {
        enc.set_function(_fn_rms_heads);
        enc.set_buffer(0, s.qkv); enc.set_buffer(1, s.ones);
        enc.set_constant(2, rows); enc.set_constant(3, NH);
        enc.set_constant(4, HD); enc.set_constant(5, 3 * D);
        enc.set_constant(6, qk * HD); enc.set_constant(7, c.norm_eps);
        enc.set_constant(8, 3 * HD);
        enc.dispatch({32, (unsigned)(rows * NH), 1}, {32, 1, 1});
      }
      for (int i = 0; i < 3; ++i) {
        enc.set_function(_fn_trope);
        enc.set_buffer(0, s.qkv);
        enc.set_buffer(1, i == 0 ? s.qh : i == 1 ? s.kh : s.vh);
        enc.set_buffer(2, s.rcos); enc.set_buffer(3, s.rsin);
        enc.set_constant(4, NH); enc.set_constant(5, rows);
        enc.set_constant(6, HD);
        enc.set_constant(7, i < 2 ? c.rope_dim() : 0);
        enc.set_constant(8, 3 * D); enc.set_constant(9, i * HD);
        enc.set_constant(10, 3 * HD);   // per-head grouping, see above
        enc.dispatch({(unsigned)HD, (unsigned)rows, (unsigned)NH},
                     {(unsigned)HD, 1, 1});
      }
      _flops.attention(NH, (long long)rows * (long long)rows, HD);
      if (use_steel) {
        enc.set_function(_fn_attn);
        enc.set_buffer(0, s.qh); enc.set_buffer(1, s.kh);
        enc.set_buffer(2, s.vh); enc.set_buffer(3, s.oh);
        enc.set_buffer(4, _attn_p);
        enc.dispatch({32 * (unsigned)((rows + A_BQ - 1) / A_BQ),
                      4 * (unsigned)NH, 1}, {32, 4, 1});
      } else {
        enc.set_function(_fn_sdpa);
        enc.set_buffer(0, s.qh); enc.set_buffer(1, s.kh);
        enc.set_buffer(2, s.vh); enc.set_buffer(3, s.oh);
        enc.set_constant(4, scale); enc.set_constant(5, rows);
        enc.set_constant(6, HD); enc.set_constant(7, NH);
        enc.set_constant(8, NH); enc.set_constant(9, rows);
        enc.set_constant(10, rows);
        enc.dispatch({32, (unsigned)NH, (unsigned)rows}, {32, 1, 1});
      }
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, s.oh); enc.set_buffer(1, s.ob);
      enc.set_constant(2, NH); enc.set_constant(3, rows);
      enc.set_constant(4, HD);
      enc.dispatch({(unsigned)HD, (unsigned)rows, (unsigned)NH},
                   {(unsigned)HD, 1, 1});
      gemm_(enc, s.ob, 0, b.out, s.nm, 0, rows, D, D);
      // The residual gate is a learned per-CHANNEL vector (zero-init at
      // training time), not a per-row modulation -- so the DiT's indexed
      // gate is the wrong kernel and the plain broadcast one is right.
      enc.set_function(_fn_gated);
      enc.set_buffer(0, s.x); enc.set_buffer(1, b.s1); enc.set_buffer(2, s.nm);
      enc.set_constant(3, D); enc.set_constant(4, rows * D);
      enc.dispatch({(unsigned)(rows * D), 1, 1}, {256, 1, 1});

      enc.set_function(_fn_rms);
      enc.set_buffer(0, s.x); enc.set_buffer(1, b.n2); enc.set_buffer(2, s.nm);
      enc.set_constant(3, D); enc.set_constant(4, c.norm_eps);
      enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
      // ---- the ANE split -------------------------------------------
      // The ANE takes the TAIL rows of s.nm and the GPU the head: every
      // kernel below reads from row 0 and the feed-forward is
      // row-independent, so the split is exact. s.nm is input and output of
      // both halves, on disjoint rows. The attention above is committed
      // WITHOUT waiting before the staging job is joined, so the two
      // overlap; the ANE starts before the GPU's half is encoded, so it
      // hides behind that whole half.
      int a_rows = 0;
      double ane_drain_ms = 0.0;
      if (ane_block || ane_probe) {
        const auto t_d0 = Clk::now();
        enc.end();
        _flops.flush(stream);
        CommandStream::Fence afence = stream.commit();
        const bool staged = ane_block && _ane->join_stage(L);
        afence.wait();
        ane_drain_ms = std::chrono::duration<double, std::milli>(
            Clk::now() - t_d0).count();
        if (staged) {
          a_rows = _ane->begin(s.nm, s.nm, rows);
        } else if (ane_block && _mc->session() != nullptr) {
          _mc->session()->warn(fmt(
              "minimax-h3-vae: staging block {}'s feed-forward for the ANE "
              "failed; it keeps the GPU", L));
        }
        enc = stream.begin_compute();
      }
      // The GPU's feed-forward over rows [r0, r0 + n) of s.nm, in place:
      // its share of a split, and the rows the ANE lost -- which still hold
      // their input, because the ANE writes a chunk back only once all of
      // it came out finite. The scratch is used from its first row.
      auto gpu_ff = [&](int r0, int n) {
        if (n <= 0) { return; }
        const std::size_t e0 = (std::size_t)r0 * D;
        gemm_(enc, s.nm, e0, b.w1, s.ff, 0, n, 2 * FF, D);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, s.ff); enc.set_buffer(1, s.qkv);
        enc.set_constant(2, n); enc.set_constant(3, FF);
        enc.dispatch({(unsigned)(n * FF), 1, 1}, {256, 1, 1});
        gemm_(enc, s.qkv, 0, b.w2, s.nm, e0, n, D, FF);
      };
      const int g_rows = rows - a_rows;      // the ANE leaves the GPU >= 1
      const auto t_g0 = Clk::now();
      gpu_ff(0, g_rows);
      if (a_rows > 0) {
        enc.end();
        _flops.flush(stream);
        stream.commit().wait();
        const double gpu_ms =
            _ane->timing() ? std::chrono::duration<double, std::milli>(
                                 Clk::now() - t_g0).count()
                           : 0.0;
        const bool ane_ok = _ane->finish(L, rows, gpu_ms, ane_drain_ms);
        enc = stream.begin_compute();
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
                             Clk::now() - t_g0).count());
        enc = stream.begin_compute();
      }
      enc.set_function(_fn_gated);
      enc.set_buffer(0, s.x); enc.set_buffer(1, b.s2); enc.set_buffer(2, s.nm);
      enc.set_constant(3, D); enc.set_constant(4, rows * D);
      enc.dispatch({(unsigned)(rows * D), 1, 1}, {256, 1, 1});
    }

    // The output norm is a LAYER norm (mean subtracted), unlike every
    // norm inside the blocks.
    enc.set_function(_fn_ln);
    enc.set_buffer(0, s.x); enc.set_buffer(1, _norm_out_w);
    enc.set_buffer(2, _norm_out_b); enc.set_buffer(3, s.nm);
    enc.set_constant(4, D); enc.set_constant(5, c.norm_eps);
    enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
    // Only the voxel rows reach the patch projection; the register and
    // cls rows are dropped.
    gemm_(enc, s.nm, 0, _proj_out, s.patches, 0, voxels, PE, D);
  }
  std::string gpu_err;
  _flops.flush(stream);
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? std::string("MiniMax-H3 video VAE decode "
                                              "failed")
                                : gpu_err);
  }

  // Unpatchify: each row is (out_channels, patch_t, patch, patch) with
  // the CHANNEL slowest, so the pixel index is
  // ((ch*pt + kt)*p + ky)*p + kx.
  const int PT = c.patch_t, P = c.patch, OC = c.out_channels;
  const int oh = h * P, ow = w * P, ot = T * PT;
  SharedBuffer out =
      _mc->make_shared_buffer((std::size_t)OC * ot * oh * ow * 2);
  if (out.empty()) { return fail("output allocation failed"); }
  {
    const auto* src = static_cast<const std::uint16_t*>(s.patches.contents());
    auto* dst = static_cast<std::uint16_t*>(out.contents());
    const std::size_t plane = (std::size_t)oh * ow;
    for (int t = 0; t < T; ++t) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const std::uint16_t* row =
              src + ((std::size_t)((t * h + y) * w + x)) * PE;
          for (int ch = 0; ch < OC; ++ch) {
            for (int kt = 0; kt < PT; ++kt) {
              for (int ky = 0; ky < P; ++ky) {
                for (int kx = 0; kx < P; ++kx) {
                  dst[((std::size_t)ch * ot + (t * PT + kt)) * plane +
                      (std::size_t)(y * P + ky) * ow + (x * P + kx)] =
                      row[((ch * PT + kt) * P + ky) * P + kx];
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

// ---- the CNN encoder ---------------------------------------------------

int
MetalMiniMaxH3VideoVae::encoded_frames(int T) const
{
  return encoded_frames_for(_cfg, T);
}

int
MetalMiniMaxH3VideoVae::encoded_frames_for(const Config& cfg, int T)
{
  if (T <= 0) { return 0; }
  int t = T;
  for (std::size_t i = 0; i < cfg.time_down.size(); ++i) {
    const int s = cfg.time_down[i];
    // Causal padding prepends 2 frames, so a stride-s convolution over
    // T frames emits floor((T-1)/s)+1 -- 17 -> 9 -> 5, where a plain
    // division would say 17 -> 8 -> 4 and silently lose the last frame.
    if (s > 1) { t = (t - 1) / s + 1; }
  }
  return t;
}

bool
MetalMiniMaxH3VideoVae::gn_coalesced_(int C) const
{
  if (!_gn3 || C <= 0 || C % 4 != 0) { return false; }
  const int C4 = C / 4;
  return C4 <= 256 && 256 % C4 == 0 && C4 % _cfg.gn_groups == 0;
}

void
MetalMiniMaxH3VideoVae::enc_gn_(ComputeEncoder& enc, const SharedBuffer& in,
                                const SharedBuffer& gamma,
                                const SharedBuffer& beta,
                                const SharedBuffer& out, int T, int H, int W,
                                int C, bool silu, bool pad)
{
  const int rows = H * W;
  const int G = _cfg.gn_groups;
  if (!gn_coalesced_(C)) {
    enc.set_function(_fn_gn_frames);
    enc.set_buffer(0, in);
    enc.set_buffer(1, gamma);
    enc.set_buffer(2, beta);
    enc.set_buffer(3, out);
    enc.set_constant(4, rows);
    enc.set_constant(5, C);
    enc.set_constant(6, G);
    enc.set_constant(7, _cfg.enc_norm_eps);
    enc.set_constant(8, silu ? 1 : 0);
    enc.dispatch({256, (unsigned)G, (unsigned)T}, {256, 1, 1});
    return;
  }
  const int nband = (rows + kGnBand - 1) / kGnBand;
  enc.set_function(_fn_gn_stats);
  enc.set_buffer(0, in);
  enc.set_buffer(1, _es.gn_part);
  enc.set_constant(2, rows);
  enc.set_constant(3, C);
  enc.set_constant(4, G);
  enc.set_constant(5, kGnBand);
  enc.dispatch({256, (unsigned)nband, (unsigned)T}, {256, 1, 1});
  enc.set_function(_fn_gn_fin);
  enc.set_buffer(0, _es.gn_part);
  enc.set_buffer(1, _es.gn_stats);
  enc.set_constant(2, nband);
  enc.set_constant(3, G);
  enc.set_constant(4, rows * (C / G));
  enc.set_constant(5, _cfg.enc_norm_eps);
  enc.dispatch({(unsigned)G, (unsigned)T, 1}, {(unsigned)G, 1, 1});
  enc.set_function(_fn_gn_apply);
  enc.set_buffer(0, in);
  enc.set_buffer(1, _es.gn_stats);
  enc.set_buffer(2, gamma);
  enc.set_buffer(3, beta);
  enc.set_buffer(4, out);
  enc.set_constant(5, rows);
  enc.set_constant(6, C);
  enc.set_constant(7, G);
  enc.set_constant(8, silu ? 1 : 0);
  enc.set_constant(9, W);
  enc.set_constant(10, H);
  enc.set_constant(11, pad ? 1 : 0);
  enc.dispatch({(unsigned)(rows * (C / 4)), (unsigned)T, 1}, {256, 1, 1});
}

void
MetalMiniMaxH3VideoVae::enc_pad_(ComputeEncoder& enc, const SharedBuffer& in,
                                 int T, int H, int W, int C, int lo, int hi,
                                 const SharedBuffer& out)
{
  const int Hp = H + lo + hi, Wp = W + lo + hi;
  enc.set_function(_fn_pad);
  enc.set_buffer(0, in);
  enc.set_buffer(1, out);
  enc.set_constant(2, H);
  enc.set_constant(3, W);
  enc.set_constant(4, C);
  enc.set_constant(5, lo);
  enc.set_constant(6, hi);
  enc.dispatch({(unsigned)(Wp * C), (unsigned)Hp, (unsigned)T}, {256, 1, 1});
}

void
MetalMiniMaxH3VideoVae::c3_dispatch_(ComputeEncoder& enc, const Conv3d& c,
                                     int cand, const SharedBuffer& in,
                                     int Ti, int H, int W,
                                     const SharedBuffer& out, int stride_t,
                                     int stride_s, const SharedBuffer* res)
{
  const int Ho = (stride_s == 2) ? (H - 2) / 2 + 1 : H;
  const int Wo = (stride_s == 2) ? (W - 2) / 2 + 1 : W;
  const int pad = (stride_s == 2) ? 1 : 2;
  const int To = (Ti - 1) / stride_t + 1;
  const C3Geom& g = kC3Geom[cand];
  enc.set_function(_fn_c3hw[stride_s == 2 ? 1 : 0][cand]);
  enc.set_buffer(0, in);
  enc.set_buffer(1, c.whwio);
  enc.set_buffer(2, out);
  enc.set_buffer(3, c.l.b);
  enc.set_buffer(4, res != nullptr ? *res : out);
  enc.set_constant(5, W + pad);
  enc.set_constant(6, H + pad);
  enc.set_constant(7, c.cin);
  enc.set_constant(8, c.cout);
  enc.set_constant(9, Wo);
  enc.set_constant(10, Ho);
  enc.set_constant(11, Ti);
  enc.set_constant(12, stride_t);
  enc.set_constant(13, res != nullptr ? 1 : 0);
  enc.set_constant(14, 0);
  const unsigned tg = (unsigned)(g.sg * 32);
  enc.dispatch({(unsigned)((Wo + g.tw - 1) / g.tw) * tg,
                (unsigned)((Ho + g.th - 1) / g.th),
                (unsigned)(To * (c.cout / g.tc))}, {tg, 1, 1});
}

int
MetalMiniMaxH3VideoVae::c3_pick_(int stride, int cin, int cout, int H,
                                 int W) const
{
  for (const C3Tune& t : _c3_tuned) {
    if (t.stride == stride && t.cin == cin && t.cout == cout && t.H == H &&
        t.W == W) {
      return t.pick;
    }
  }
  return -1;
}

void
MetalMiniMaxH3VideoVae::c3_tune_(const Conv3d& c, int Ti, int H, int W,
                                 int stride_t, int stride_s)
{
  if (c.cout % 128 != 0 || c3_pick_(stride_s, c.cin, c.cout, H, W) >= 0) {
    return;
  }
  // A few frames are enough to rank the tiles -- the shape that decides
  // is the frame's -- and keep the tuning to milliseconds a shape. The
  // trial reads `_es.b` and writes `_es.c`: `_es.a` holds this encode's
  // input. Their contents are garbage, which a timing does not mind.
  const int ti = std::min(Ti, 3);
  // Kept for the session (tuned-answers.h): the encoder is loaded again
  // for every clip it encodes, at the same few shapes.
  KernelTuningCache* const answers = tuning_cache(_mc);
  const TuneKey key = TuneKey("h3-vvae-c3/1")("stride_t", stride_t)
      ("stride_s", stride_s)("cin", c.cin)("cout", c.cout)("H", H)
      ("W", W)("frames", ti);
  int pick = -1;
  if (!find_tuned(answers, key, pick) || pick < 0 || pick >= kC3Cands) {
    pick = autotune_vote(kC3Cands, /*rounds=*/3, /*reps=*/1,
        [&](int i) {
          return autotune_time(_mc, 1, [&](ComputeEncoder& enc) {
            c3_dispatch_(enc, c, i, _es.b, ti, H, W, _es.c, stride_t,
                         stride_s, nullptr);
          });
        });
    store_tuned(answers, key, pick);
  }
  _c3_tuned.push_back(C3Tune{stride_s, c.cin, c.cout, H, W, pick});
  if (_mc->session() != nullptr) {
    const std::string nm = kC3Names[pick];
    _mc->session()->log_debug(fmt(
        "MiniMax-H3 video VAE: hardware conv s{} {}->{} at {}x{} takes "
        "the {} tile", stride_s, c.cin, c.cout, W, H, nm));
  }
}

bool
MetalMiniMaxH3VideoVae::hw_route_(const Conv3d& c, int H, int W,
                                  int stride_s, bool from_norm) const
{
  if (!_use_hwconv || c.whwio.empty() || H < 2 || W < 2) { return false; }
  // The 16-channel tile has no stride-2 twin.
  if (c.cout % 128 != 0 && stride_s != 1) { return false; }
  if (from_norm && !gn_coalesced_(c.cin)) { return false; }
  // The op addresses one padded FRAME as its source, in int32 bytes.
  const int pad = (stride_s == 2) ? 1 : 2;
  return tensor_op_operand_fits(
      (std::size_t)(H + pad) * (std::size_t)(W + pad) * (std::size_t)c.cin,
      2);
}

int
MetalMiniMaxH3VideoVae::enc_conv_hw_(ComputeEncoder& enc, const Conv3d& c,
                                     const SharedBuffer& in, int Ti, int H,
                                     int W, const SharedBuffer& out,
                                     int stride_t, int stride_s,
                                     const SharedBuffer* res)
{
  if (Ti <= 0 || H <= 0 || W <= 0) { return 0; }
  const int Ho = (stride_s == 2) ? (H - 2) / 2 + 1 : H;
  const int Wo = (stride_s == 2) ? (W - 2) / 2 + 1 : W;
  const int To = (Ti - 1) / stride_t + 1;
  // Counted by LIVE taps, exactly as the gather route counts them, so the
  // two report the same arithmetic for the same conv.
  for (int t = 0; t < To; ++t) {
    int live = 0;
    for (int kt = 0; kt < 3; ++kt) {
      const int si = t * stride_t + kt - 2;
      live += (si >= 0 && si < Ti) ? 1 : 0;
    }
    _flops.gemm((long long)Ho * Wo, c.cout, 9LL * live * c.cin);
  }
  // A grid with a partial tile takes the guarded entry; otherwise the
  // tile this shape measured fastest on (tuned in encode()).
  const bool partial = (Ho % 8) != 0 || (Wo % 16) != 0;
  int cand = partial ? kC3Tail : c3_pick_(stride_s, c.cin, c.cout, H, W);
  if (cand < 0) { cand = 0; }
  if (c.cout % 128 != 0) { cand = kC3C16; }
  c3_dispatch_(enc, c, cand, in, Ti, H, W, out, stride_t, stride_s, res);
  return To;
}

int
MetalMiniMaxH3VideoVae::enc_conv_(ComputeEncoder& enc, const Conv3d& c,
                                  const SharedBuffer& in, int Ti, int H, int W,
                                  const SharedBuffer& out, int stride_t,
                                  int stride_s)
{
  if (Ti <= 0 || H <= 0 || W <= 0) { return 0; }
  if (!c.spatial) {
    // 1x1x1: no spatial gather and no temporal reach, so it is a plain
    // Linear over every voxel of the clip at once. The voxel count is
    // the ONE quantity here that an extent could push past an int on
    // its own -- gemm_ bands everything downstream of it -- so it is
    // computed wide and refused rather than wrapped.
    const std::size_t voxels = (std::size_t)Ti * H * W;
    if (voxels > (std::size_t)INT_MAX) { return 0; }
    gemm_(enc, in, 0, c.l, out, 0, (int)voxels, c.cout, c.cin);
    return Ti;
  }
  // A conv loaded for the hardware route has no GEMM weight until an
  // extent that route declines asks for one.
  Linear lazy;
  const Linear* lp = &c.l;
  if (c.l.w.empty()) {
    lazy.w = gemm_weight_(c);
    if (lazy.w.empty()) { return 0; }
    lazy.b = c.l.b.subview(0, c.l.b.byte_size());
    lp = &lazy;
  }
  const Linear& l = *lp;
  const int Ho = (stride_s == 2) ? (H - 2) / 2 + 1 : H;
  const int Wo = (stride_s == 2) ? (W - 2) / 2 + 1 : W;
  // A stride-2 convolution gets its padding from the reference's
  // separate asymmetric bottom/right pad, so it carries none of its own.
  const int pad_lo = (stride_s == 2) ? 0 : 1;
  const int To = (Ti - 1) / stride_t + 1;
  const std::size_t ohw = (std::size_t)Ho * Wo;
  const std::size_t ihw = (std::size_t)H * W;
  const std::size_t per_row = (std::size_t)27 * c.cin;
  std::size_t band = per_row > 0 ? _es.col_cap / per_row : ohw;
  band = std::min(std::max<std::size_t>(band, 1), ohw);
  band = std::min<std::size_t>(band, 4096);

  for (int t = 0; t < To; ++t) {
    // Causal taps: output frame t reads input frames t*stride - 2 ..
    // t*stride, and anything before the start of the clip reads as zero.
    int mask = 0, live = 0;
    std::size_t off[3] = {0, 0, 0};
    for (int kt = 0; kt < 3; ++kt) {
      const int si = t * stride_t + kt - 2;
      if (si >= 0 && si < Ti) {
        mask |= 1 << kt;
        ++live;
        off[kt] = (std::size_t)si * ihw * (std::size_t)c.cin;
      }
    }
    // The frame's arithmetic is its LIVE taps: a tap before the clip is
    // causal zero padding, which the gather below multiplies but the
    // model does not (the same rule as the Wan VAE's).
    _flops.gemm((long long)ohw, c.cout, 9LL * live * c.cin);
    for (std::size_t r0 = 0; r0 < ohw; r0 += band) {
      const int mrows = (int)std::min(band, ohw - r0);
      enc.set_function(_fn_im2col_r);
      for (int kt = 0; kt < 3; ++kt) {
        enc.set_buffer((unsigned)kt, in, off[kt] * 2);
      }
      enc.set_buffer(3, _es.col);
      enc.set_constant(4, H);
      enc.set_constant(5, W);
      enc.set_constant(6, c.cin);
      enc.set_constant(7, (int)r0);
      enc.set_constant(8, mrows);
      enc.set_constant(9, mask);
      enc.set_constant(10, Wo);
      enc.set_constant(11, stride_s);
      enc.set_constant(12, pad_lo);
      enc.dispatch({(unsigned)(27 * c.cin), (unsigned)mrows, 1}, {64, 1, 1});
      gemm_uncounted_(enc, _es.col, 0, l, out,
                      ((std::size_t)t * ohw + r0) * (std::size_t)c.cout,
                      mrows, c.cout, 27 * c.cin);
    }
  }
  return To;
}

bool
MetalMiniMaxH3VideoVae::ensure_enc_scratch_(int T, int H, int W)
{
  if (_es.T == T && _es.H == H && _es.W == W) { return true; }
  const Config& c = _cfg;
  // The widest activation the encode ever holds. It is almost always
  // level 0's -- the spatial extent falls by 4x per level while the
  // channel count only doubles -- but derive it rather than assume it,
  // because a checkpoint with a different `space_down` would move it.
  std::size_t need =
      (std::size_t)T * H * W * (std::size_t)std::max(c.in_channels, c.base_ch);
  // The hardware route's inputs are reflect-padded frames, one pixel
  // all round -- the widest pad either stride uses.
  const int pad = _use_hwconv ? 2 : 0;
  // The three-pass norm's partials: every frame's bands at every level.
  std::size_t gn_part = 0, gn_stats = 0;
  const std::size_t G = (std::size_t)c.gn_groups;
  int t = T, h = H, w = W;
  for (std::size_t i = 0; i < _enc_levels.size(); ++i) {
    const EncLevel& lv = _enc_levels[i];
    need = std::max(need, (std::size_t)t * (h + pad) * (w + pad) *
                              (std::size_t)std::max(lv.cin, lv.cout));
    const std::size_t nband =
        ((std::size_t)h * w + kGnBand - 1) / kGnBand;
    gn_part = std::max(gn_part, (std::size_t)t * nband * G * 2);
    gn_stats = std::max(gn_stats, (std::size_t)t * G * 2);
    if (lv.space > 1) { h = (h - 2) / 2 + 1; w = (w - 2) / 2 + 1; }
    if (lv.time > 1) { t = (t - 1) / lv.time + 1; }
    need = std::max(need, (std::size_t)t * h * w * (std::size_t)lv.cout);
  }
  need = std::max(need,
                  (std::size_t)t * h * w * (std::size_t)(2 * c.z_channels));
  EncScratch s;
  s.T = T; s.H = H; s.W = W;
  // The im2col band. The gather is 27x the activation, so a full frame of it
  // would be larger than the whole encode's working set at any real tile
  // size; banding is what keeps it a fixed cost.
  //
  // 32 MB, not the 16 MB this shipped with. The band is col_cap/(27*cin), so
  // the cap sets the GEMM's M, and M has an optimum rather than a direction:
  // MEASURED at the deepest level (cin=1024, K=27648), best GF/s by band --
  // 303: 8880, 606: 10216, 808: 10150, 1213: 9456, 2427: 7960, 4096: 6649.
  // The band is itself an operand competing for cache, so past ~800 rows it
  // costs more than the better shape buys. 32 MB puts cin=1024 at 606, its
  // peak. It also moves cin=512 from 606 to 1213, which measures ~5% below
  // ITS peak -- one cap cannot put both levels at 606, and the deeper level
  // is the one that was furthest from its optimum. A per-level band is the
  // real answer and wants its own tuner; this is the bounded version.
  s.col_cap = 16u << 20;
  s.a = _mc->make_shared_buffer(need * 2);
  s.b = _mc->make_shared_buffer(need * 2);
  s.c = _mc->make_shared_buffer(need * 2);
  s.col = _mc->make_shared_buffer(s.col_cap * 2);
  s.gn_part  = _mc->make_shared_buffer(std::max<std::size_t>(gn_part, 2) * 4);
  s.gn_stats = _mc->make_shared_buffer(std::max<std::size_t>(gn_stats, 2) *
                                       4);
  if (s.a.empty() || s.b.empty() || s.c.empty() || s.col.empty() ||
      s.gn_part.empty() || s.gn_stats.empty()) {
    return false;
  }
  _es = std::move(s);
  return true;
}

SharedBuffer
MetalMiniMaxH3VideoVae::encode(const SharedBuffer& x, int T, int H, int W,
                               std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  const int IC = c.in_channels;
  if (T <= 0 || H <= 0 || W <= 0) { return fail("empty clip"); }
  if (H % c.patch != 0 || W % c.patch != 0) {
    return fail("H and W must be multiples of " + std::to_string(c.patch));
  }
  if (x.byte_size() < (std::size_t)IC * T * H * W * 2) {
    return fail("input is smaller than in_channels * T * H * W");
  }
  if (!_ws) { return fail("no weight set"); }
  if (!_ws->ensure_part("encoder", [&]() {
        _part = "encoder";
        const bool ok = load_encoder_(*_ws);
        _part.clear();
        return ok;
      })) {
    return fail("the encoder half of this checkpoint failed to load");
  }
  if (!ensure_enc_scratch_(T, H, W)) {
    return fail("activation allocation failed (out of GPU memory)");
  }

  const bool prof = enc_profile_on_();
  auto pclock = std::chrono::steady_clock::now();
  // Channel-first [C, T, H, W] -> per-frame channel-last [T, H*W, C],
  // the layout every kernel below reads. Done on the host for the same
  // reason as the decoder's: it is one pass over the input and the GPU
  // alternative is a kernel that exists only here.
  {
    const auto* src = static_cast<const std::uint16_t*>(x.contents());
    auto* dst = static_cast<std::uint16_t*>(_es.a.contents());
    const std::size_t hw = (std::size_t)H * W;
    for (int t = 0; t < T; ++t) {
      for (std::size_t p = 0; p < hw; ++p) {
        for (int ch = 0; ch < IC; ++ch) {
          dst[((std::size_t)t * hw + p) * IC + ch] =
              src[((std::size_t)ch * T + t) * hw + p];
        }
      }
    }
  }

  // Three interchangeable buffers. `cur` is the live activation; the
  // other two are spares, and every op writes to a spare so nothing is
  // ever read and written in the same dispatch.
  SharedBuffer* buf[3] = {&_es.a, &_es.b, &_es.c};
  int cur = 0;
  auto spare = [&](int n) { return (cur + 1 + n) % 3; };

  // Depth override, for bisecting a mismatch: run only the first N
  // levels and return that activation raw, with no output head. A
  // 0-level run is conv_in alone, which separates the padding and the
  // host transposes from everything the levels do.
  int levels = (int)_enc_levels.size();
  int out_ch = 2 * c.z_channels;
  if (const char* lv = std::getenv("VPIPE_H3_VVAE_ENC_LEVELS")) {
    const int n = std::atoi(lv);
    if (n >= 0 && n < levels) {
      levels = n;
      out_ch = (n == 0) ? c.base_ch : _enc_levels[(std::size_t)n - 1].cout;
    }
  }
  const bool head = out_ch == 2 * c.z_channels;
  const int ZC2o = out_ch;

  int t = T, h = H, w = W;
  // Split-K tuning, before the stream opens (it runs its own).
  //
  // The GEMM this tower cares about is a 3x3x3 conv's im2col matmul, whose
  // K is 27*cin -- 27648 at the deepest level, deeper than anything in the
  // DiT. What decides whether splitting that pays is not K, though, but M,
  // and M here is the IM2COL BAND, not the level's voxel count: the band is
  // capped at col_cap/(27*cin) so the gather stays a fixed 16 MB, which
  // means the deeper the level, the FEWER rows per GEMM. At cin=1024 that
  // is ~303 rows, and a split wants ~512+.
  //
  // So the banding inverts the regime: the levels whose weights are large
  // enough to want a split are exactly the ones whose band is too short to
  // get one. That is a real result rather than a reason to skip the tuner --
  // it is measured per shape here, so raising col_cap (or a level's band)
  // flips these decisions automatically instead of needing this comment to
  // be re-read.
  if (_use_mma2 && _splitk.enabled) {
    KernelTuningCache* const answers = tuning_cache(_mc);
    int tt = T, hh = H, ww = W;
    for (const EncLevel& lv : _enc_levels) {
      for (const EncResnet& r : lv.res) {
        for (const Conv3d* cv : {&r.c1, &r.c2}) {
          // The 3x3x3 GEMMs only: `k` is the GEMM's depth (27*cin), so
          // the test this replaces -- `k != 3` -- skipped every conv and
          // nothing here ever ran. A conv on the hardware route has no
          // GEMM to tune.
          if (cv->l.w.empty() || !cv->spatial) { continue; }
          const std::size_t per_row = (std::size_t)27 * cv->cin;
          const std::size_t ohw = (std::size_t)hh * ww;
          std::size_t band = per_row > 0 ? _es.col_cap / per_row : ohw;
          band = std::min(std::max<std::size_t>(band, 1), ohw);
          band = std::min<std::size_t>(band, 4096);
          const int M = (int)band, N = cv->cout, K = (int)per_row;
          // The trial GEMMs write `_es.b`: `_es.a` already holds the
          // input this encode is about to read.
          if (M < 128 || _es.b.byte_size() < (std::size_t)M * N * 2) {
            continue;
          }
          auto run = [&](ComputeEncoder& enc) {
            gemm_mma_(enc, _es.col, 0, cv->l, _es.b, 0, M, N, K);
          };
          // Tile first, with the split held off: the two decisions are not
          // independent (a split changes how much work each tile does), and
          // measuring them together would confound them. Same order the DiT
          // tuner uses.
          bool have = false;
          int wt = -1;
          for (const TileTune& t : _tile_tuned) {
            if (t.N == N && t.K == K && t.M == M) {
              have = true;
              wt = t.wide ? 1 : 0;
              break;
            }
          }
          // Both answers kept for the session (tuned-answers.h), under
          // the GEMM's shape and its weight's format.
          const std::string wfmt = cv->l.quantized
              ? "w" + std::to_string(cv->l.bits) : std::string("dense");
          const TuneKey tkey = TuneKey("h3-vvae-tile/1")("fmt", wfmt)
              ("N", N)("K", K)("M", M);
          if (!have &&
              (!find_tuned(answers, tkey, wt) || wt < 0 || wt > 1)) {
            _splitk.bypass = true;
            wt = autotune_vote(2, /*rounds=*/3, /*reps_for_us=*/1,
                [&](int i) {
                  _force_tile = (i == 0) ? 1 : 2;
                  const double t = autotune_time(_mc, 1, run);
                  _force_tile = 0;
                  return t;
                });
            _splitk.bypass = false;
            store_tuned(answers, tkey, wt);
          }
          if (!have) { _tile_tuned.push_back(TileTune{N, K, M, wt == 1}); }
          _splitk.tune(_mc, K, N, M, run,
                       "h3-vvae/1 fmt=" + wfmt + " tile=" +
                           std::to_string(wt));
        }
      }
      if (lv.space > 1) { hh = (hh - 2) / 2 + 1; ww = (ww - 2) / 2 + 1; }
      if (lv.time > 1) { tt = (tt - 1) / lv.time + 1; }
    }
    (void)tt;
  }
  // The hardware conv's tile, per shape, before the stream opens.
  if (_use_hwconv) {
    int tt = T, hh = H, ww = W;
    if (hw_route_(_enc_conv_in, hh, ww, 1, false)) {
      c3_tune_(_enc_conv_in, tt, hh, ww, 1, 1);
    }
    for (const EncLevel& lv : _enc_levels) {
      for (const EncResnet& r : lv.res) {
        for (const Conv3d* cv : {&r.c1, &r.c2}) {
          if (hw_route_(*cv, hh, ww, 1, true)) {
            c3_tune_(*cv, tt, hh, ww, 1, 1);
          }
        }
      }
      if (!lv.down.empty() && hw_route_(lv.down, hh, ww, lv.space, false)) {
        c3_tune_(lv.down, tt, hh, ww, lv.time, lv.space);
      }
      if (lv.space > 1) { hh = (hh - 2) / 2 + 1; ww = (ww - 2) / 2 + 1; }
      if (lv.time > 1) { tt = (tt - 1) / lv.time + 1; }
    }
  }
  _flops.begin();
  if (prof) { prof_add_("host: transpose in", ms_since_(pclock)); }
  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // A profile bucket ends here: commit, wait, and charge the time since
    // the last mark to `bucket`.
    pclock = std::chrono::steady_clock::now();
    auto mark = [&](const std::string& bucket) {
      if (!prof) { return; }
      enc.end();
      _flops.flush(stream);
      std::string ge;
      (void)stream.commit().wait_ok(&ge);
      prof_add_(bucket, ms_since_(pclock));
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
      pclock = std::chrono::steady_clock::now();
    };
    if (hw_route_(_enc_conv_in, h, w, 1, false)) {
      enc_pad_(enc, *buf[cur], t, h, w, IC, 1, 1, *buf[spare(1)]);
      t = enc_conv_hw_(enc, _enc_conv_in, *buf[spare(1)], t, h, w,
                       *buf[spare(0)], 1, 1, nullptr);
    } else {
      t = enc_conv_(enc, _enc_conv_in, *buf[cur], t, h, w, *buf[spare(0)], 1,
                    1);
    }
    cur = spare(0);
    if (t == 0) { return fail("conv_in produced no frames"); }
    mark("conv_in");

    for (std::size_t li = 0; li < (std::size_t)levels; ++li) {
      const EncLevel& lv = _enc_levels[li];
      const std::string lname = "L" + std::to_string(li);
      for (const EncResnet& r : lv.res) {
        const int s1 = spare(0), s2 = spare(1);
        const std::size_t rows = (std::size_t)h * w;
        // norm -> silu -> conv1 -> norm -> silu -> conv2, then the skip.
        // On the hardware route each norm writes its conv's input
        // reflect-padded, and conv2 adds the residual in its epilogue.
        const bool hw1 = hw_route_(r.c1, h, w, 1, true);
        const bool hw2 = hw_route_(r.c2, h, w, 1, true);
        enc_gn_(enc, *buf[cur], r.n1w, r.n1b, *buf[s1], t, h, w, r.cin,
                true, hw1);
        mark("norm");
        if (hw1) {
          enc_conv_hw_(enc, r.c1, *buf[s1], t, h, w, *buf[s2], 1, 1,
                       nullptr);
        } else {
          enc_conv_(enc, r.c1, *buf[s1], t, h, w, *buf[s2], 1, 1);
        }
        mark(lname + " conv3");
        enc_gn_(enc, *buf[s2], r.n2w, r.n2b, *buf[s1], t, h, w, r.cout,
                true, hw2);
        mark("norm");
        if (hw2) {
          if (r.skip.empty()) {
            enc_conv_hw_(enc, r.c2, *buf[s1], t, h, w, *buf[s2], 1, 1,
                         buf[cur]);
            mark(lname + " conv3");
            cur = s2;
          } else {
            // conv1's output is spent (the norm read it), so the
            // shortcut takes its buffer, and the sum lands in `cur`,
            // whose last reader was the shortcut.
            enc_conv_(enc, r.skip, *buf[cur], t, h, w, *buf[s2], 1, 1);
            mark("conv1 shortcut");
            enc_conv_hw_(enc, r.c2, *buf[s1], t, h, w, *buf[cur], 1, 1,
                         buf[s2]);
            mark(lname + " conv3");
          }
          continue;
        }
        enc_conv_(enc, r.c2, *buf[s1], t, h, w, *buf[s2], 1, 1);
        mark(lname + " conv3");
        const std::size_t n = (std::size_t)t * rows * (std::size_t)r.cout;
        // Elementwise over the whole activation, which at a full-frame
        // encode is 2.2e9 elements -- past the int this used to narrow
        // to. Banded, so the count the kernel is handed is always one
        // it can index.
        auto residual = [&](int a, int b, int o) {
          for (std::size_t i = 0; i < n; i += kEltBand) {
            const int m = (int)std::min((std::size_t)kEltBand, n - i);
            enc.set_function(_fn_residual);
            enc.set_buffer(0, *buf[a], i * 2);
            enc.set_buffer(1, *buf[b], i * 2);
            enc.set_buffer(2, *buf[o], i * 2);
            enc.set_constant(3, m);
            enc.dispatch({(unsigned)m, 1, 1}, {256, 1, 1});
          }
        };
        if (r.skip.empty()) {
          residual(cur, s2, s1);
          mark("residual");
        } else {
          // The 1x1x1 shortcut has to run BEFORE the add, and it reads
          // the resnet's input -- which is why three buffers are the
          // minimum here and not two.
          enc_conv_(enc, r.skip, *buf[cur], t, h, w, *buf[s1], 1, 1);
          mark("conv1 shortcut");
          residual(s1, s2, cur);
          mark("residual");
          continue;                    // the sum landed back in `cur`
        }
        cur = s1;
      }
      if (!lv.down.empty()) {
        const int s1 = spare(0), s2 = spare(1);
        int nt = 0;
        if (hw_route_(lv.down, h, w, lv.space, false)) {
          // Stride 2 pads the bottom and right only; a stride-1 temporal
          // downsample would pad all round, as the norm does.
          const int lo = (lv.space == 2) ? 0 : 1;
          enc_pad_(enc, *buf[cur], t, h, w, lv.cout, lo, 1, *buf[s2]);
          nt = enc_conv_hw_(enc, lv.down, *buf[s2], t, h, w, *buf[s1],
                            lv.time, lv.space, nullptr);
        } else {
          nt = enc_conv_(enc, lv.down, *buf[cur], t, h, w, *buf[s1],
                         lv.time, lv.space);
        }
        if (nt == 0) { return fail("a downsample produced no frames"); }
        mark(lname + " down");
        if (lv.space > 1) { h = (h - 2) / 2 + 1; w = (w - 2) / 2 + 1; }
        t = nt;
        cur = s1;
      }
    }

    if (head) {
      const int s1 = spare(0), s2 = spare(1);
      const bool hwo = hw_route_(_enc_conv_out, h, w, 1, true);
      enc_gn_(enc, *buf[cur], _enc_norm_w, _enc_norm_b, *buf[s1], t, h, w,
              _enc_levels.back().cout, true, hwo);
      if (hwo) {
        enc_conv_hw_(enc, _enc_conv_out, *buf[s1], t, h, w, *buf[s2], 1, 1,
                     nullptr);
      } else {
        enc_conv_(enc, _enc_conv_out, *buf[s1], t, h, w, *buf[s2], 1, 1);
      }
      enc_conv_(enc, _quant_conv, *buf[s2], t, h, w, *buf[s1], 1, 1);
      cur = s1;
      mark("head");
    }
  }
  std::string gpu_err;
  _flops.flush(stream);
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty()
                    ? std::string("MiniMax-H3 video VAE encode failed")
                    : gpu_err);
  }
  pclock = std::chrono::steady_clock::now();

  // Back to channel-first [2*z, To, h, w], matching what decode() takes.
  SharedBuffer out =
      _mc->make_shared_buffer((std::size_t)ZC2o * t * h * w * 2);
  if (out.empty()) { return fail("output allocation failed"); }
  {
    const auto* src = static_cast<const std::uint16_t*>(buf[cur]->contents());
    auto* dst = static_cast<std::uint16_t*>(out.contents());
    const std::size_t hw = (std::size_t)h * w;
    for (int f = 0; f < t; ++f) {
      for (std::size_t p = 0; p < hw; ++p) {
        for (int ch = 0; ch < ZC2o; ++ch) {
          dst[((std::size_t)ch * t + f) * hw + p] =
              src[((std::size_t)f * hw + p) * ZC2o + ch];
        }
      }
    }
  }
  if (prof) { prof_add_("host: transpose out", ms_since_(pclock)); }
  return out;
}

void
MetalMiniMaxH3VideoVae::prof_add_(const std::string& name, double ms)
{
  for (ProfBucket& b : _enc_prof) {
    if (b.name == name) { b.ms += ms; ++b.marks; return; }
  }
  _enc_prof.push_back(ProfBucket{name, ms, 1});
}

void
MetalMiniMaxH3VideoVae::prof_log_(int T, int H, int W)
{
  if (_enc_prof.empty() || _mc->session() == nullptr) { return; }
  double tot = 0.0;
  for (const ProfBucket& b : _enc_prof) { tot += b.ms; }
  std::string line;
  for (const ProfBucket& b : _enc_prof) {
    line += std::format("\n  {:<20} {:>9.1f} ms {:>5.1f}%  ({} marks)",
                        b.name, b.ms,
                        tot > 0.0 ? 100.0 * b.ms / tot : 0.0, b.marks);
  }
  std::string route = _use_hwconv ? "hardware conv" : "im2col + GEMM";
  for (const C3Tune& t : _c3_tuned) {
    route += std::format("; s{} {}->{} {}x{} {}", t.stride, t.cin, t.cout,
                         t.W, t.H, kC3Names[t.pick]);
  }
  _mc->session()->log_normal(fmt(
      "MiniMax-H3 video VAE encode profile, {}x{}x{} ({}; each GPU "
      "bucket is a commit AND a wait): {:.1f} ms{}",
      T, W, H, route, tot, line));
  _enc_prof.clear();
}

// ---- tiling and clip chunking ------------------------------------------

void
MetalMiniMaxH3VideoVae::stitch_tile_(std::uint16_t* dst,
                                     const std::uint16_t* cur,
                                     const std::uint16_t* up,
                                     const std::uint16_t* lf, int i, int j,
                                     const minimax_h3::TileSplit& ys,
                                     const minimax_h3::TileSplit& xs,
                                     int ratio, int C, int T, int out_h,
                                     int out_w, std::vector<float>& tmp)
{
  const int rows = (int)ys.start.size();
  const int cols = (int)xs.start.size();
  const int th = ys.length[(std::size_t)i] / ratio;
  const int oy = ys.start[(std::size_t)i] / ratio;
  const int tw = xs.length[(std::size_t)j] / ratio;
  const int ox = xs.start[(std::size_t)j] / ratio;
  const std::size_t oplane = (std::size_t)out_h * out_w;
  const std::size_t plane = (std::size_t)th * tw;
  const std::size_t n = (std::size_t)C * T * plane;
  tmp.resize(n);
  for (std::size_t k = 0; k < n; ++k) { tmp[k] = bf16_to_f32_(cur[k]); }

  // Cross-fade against the tile ABOVE, then against the one to the
  // LEFT. Both read the neighbour's ORIGINAL bytes, not a version
  // already blended this pass -- the reference rebinds a local and
  // leaves the tile grid untouched, so a corner cell mixes three
  // unmodified tiles rather than a chain of two.
  if (i > 0 && up != nullptr) {
    const int ph = ys.length[(std::size_t)(i - 1)] / ratio;
    int ext = ys.overlap[(std::size_t)(i - 1)] / ratio;
    ext = std::min(std::min(ext, ph), th);
    for (int c = 0; c < C; ++c) {
      for (int t = 0; t < T; ++t) {
        const std::size_t cb = ((std::size_t)c * T + t) * plane;
        const std::size_t pb = ((std::size_t)c * T + t) * ph * tw;
        for (int r = 0; r < ext; ++r) {
          const float wb = (float)r / (float)ext;
          for (int x = 0; x < tw; ++x) {
            const std::size_t k = cb + (std::size_t)r * tw + x;
            tmp[k] = (1.0f - wb) *
                         bf16_to_f32_(up[pb + (std::size_t)(ph - ext + r) *
                                                 tw + x]) +
                     wb * tmp[k];
          }
        }
      }
    }
  }
  if (j > 0 && lf != nullptr) {
    const int pw = xs.length[(std::size_t)(j - 1)] / ratio;
    int ext = xs.overlap[(std::size_t)(j - 1)] / ratio;
    ext = std::min(std::min(ext, pw), tw);
    for (int c = 0; c < C; ++c) {
      for (int t = 0; t < T; ++t) {
        const std::size_t cb = ((std::size_t)c * T + t) * plane;
        const std::size_t pb = ((std::size_t)c * T + t) * th * pw;
        for (int r = 0; r < th; ++r) {
          for (int x = 0; x < ext; ++x) {
            const float wb = (float)x / (float)ext;
            const std::size_t k = cb + (std::size_t)r * tw + x;
            tmp[k] = (1.0f - wb) *
                         bf16_to_f32_(lf[pb + (std::size_t)r * pw +
                                         (pw - ext + x)]) +
                     wb * tmp[k];
          }
        }
      }
    }
  }

  // A non-last tile contributes everything but the overlap it hands
  // to its successor, so the contributions tile the axis exactly.
  const int wh = (i + 1 < rows)
                     ? th - ys.overlap[(std::size_t)i] / ratio
                     : th;
  const int ww = (j + 1 < cols)
                     ? tw - xs.overlap[(std::size_t)j] / ratio
                     : tw;
  for (int c = 0; c < C; ++c) {
    for (int t = 0; t < T; ++t) {
      const std::size_t cb = ((std::size_t)c * T + t) * plane;
      const std::size_t ob = ((std::size_t)c * T + t) * oplane;
      for (int r = 0; r < wh; ++r) {
        for (int x = 0; x < ww; ++x) {
          dst[ob + (std::size_t)(oy + r) * out_w + (ox + x)] =
              f32_to_bf16_(tmp[cb + (std::size_t)r * tw + x]);
        }
      }
    }
  }
}

SharedBuffer
MetalMiniMaxH3VideoVae::stitch_(const std::vector<SharedBuffer>& tiles,
                                const minimax_h3::TileSplit& ys,
                                const minimax_h3::TileSplit& xs, int ratio,
                                int C, int T, int out_h, int out_w)
{
  const int rows = (int)ys.start.size();
  const int cols = (int)xs.start.size();
  if (ratio <= 0 || rows == 0 || cols == 0 ||
      (int)tiles.size() != rows * cols) {
    return {};
  }
  SharedBuffer out =
      _mc->make_shared_buffer((std::size_t)C * T * out_h * out_w * 2);
  if (out.empty()) { return {}; }
  auto* dst = static_cast<std::uint16_t*>(out.contents());
  auto px = [&](int r, int q) {
    return static_cast<const std::uint16_t*>(
        tiles[(std::size_t)(r * cols + q)].contents());
  };
  std::vector<float> tmp;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const auto& tb = tiles[(std::size_t)(i * cols + j)];
      const std::size_t n = (std::size_t)C * T *
                            (ys.length[(std::size_t)i] / ratio) *
                            (xs.length[(std::size_t)j] / ratio);
      if (tb.empty() || tb.byte_size() < n * 2) { return {}; }
      stitch_tile_(dst, px(i, j), i > 0 ? px(i - 1, j) : nullptr,
                   j > 0 ? px(i, j - 1) : nullptr, i, j, ys, xs, ratio, C,
                   T, out_h, out_w, tmp);
    }
  }
  return out;
}

SharedBuffer
MetalMiniMaxH3VideoVae::encode_tiled_(const SharedBuffer& x, int T, int H,
                                      int W, std::string* err)
{
  const Config& c = _cfg;
  const auto ys = minimax_h3::split_tiles(H, c.tile_size, c.tile_overlap_min,
                                          c.patch);
  const auto xs = minimax_h3::split_tiles(W, c.tile_size, c.tile_overlap_min,
                                          c.patch);
  if (ys.start.size() <= 1 && xs.start.size() <= 1) {
    // One tile is still a unit of work, so it ticks -- for the same
    // reason decode_tiled_'s untiled path does.
    if (_prog_total <= 0) { _prog_total = 1; _prog_done = 0; }
    if (_tile_progress) { _tile_progress(_prog_done, _prog_total); }
    if (stopping()) { return {}; }
    SharedBuffer one = encode(x, T, H, W, err);
    if (!one.empty()) { ++_prog_done; }
    if (_tile_progress) { _tile_progress(_prog_done, _prog_total); }
    return one;
  }
  if (_prog_total <= 0) {
    _prog_total = (int)(ys.start.size() * xs.start.size());
    _prog_done  = 0;
  }
  const int IC = c.in_channels;
  const std::size_t plane = (std::size_t)H * W;
  const auto* src = static_cast<const std::uint16_t*>(x.contents());
  std::vector<SharedBuffer> tiles;
  tiles.reserve(ys.start.size() * xs.start.size());
  for (std::size_t i = 0; i < ys.start.size(); ++i) {
    for (std::size_t j = 0; j < xs.start.size(); ++j) {
      // Between tiles, with the GPU idle: the previous tile committed
      // and waited. Empty and NO `err` -- a stop is not a failure.
      if (stopping()) { return {}; }
      const int y0 = ys.start[i], th = ys.length[i];
      const int x0 = xs.start[j], tw = xs.length[j];
      const auto pc = std::chrono::steady_clock::now();
      SharedBuffer sub =
          _mc->make_shared_buffer((std::size_t)IC * T * th * tw * 2);
      if (sub.empty()) {
        if (err != nullptr) { *err = "tile allocation failed"; }
        return {};
      }
      auto* d = static_cast<std::uint16_t*>(sub.contents());
      for (int ch = 0; ch < IC; ++ch) {
        for (int t = 0; t < T; ++t) {
          const std::size_t sb = ((std::size_t)ch * T + t) * plane;
          const std::size_t db = ((std::size_t)ch * T + t) * th * tw;
          for (int r = 0; r < th; ++r) {
            std::memcpy(d + db + (std::size_t)r * tw,
                        src + sb + (std::size_t)(y0 + r) * W + x0,
                        (std::size_t)tw * 2);
          }
        }
      }
      if (enc_profile_on_()) { prof_add_("host: tile copy", ms_since_(pc)); }
      if (_tile_progress) { _tile_progress(_prog_done, _prog_total); }
      SharedBuffer enc = encode(sub, T, th, tw, err);
      if (enc.empty()) { return {}; }
      ++_prog_done;
      if (_tile_progress) { _tile_progress(_prog_done, _prog_total); }
      tiles.push_back(std::move(enc));
    }
  }
  const auto pc = std::chrono::steady_clock::now();
  SharedBuffer out =
      stitch_(tiles, ys, xs, c.patch, 2 * c.z_channels, encoded_frames(T),
              H / c.patch, W / c.patch);
  if (enc_profile_on_()) { prof_add_("host: stitch", ms_since_(pc)); }
  if (out.empty() && err != nullptr) { *err = "tile stitch failed"; }
  return out;
}

SharedBuffer
MetalMiniMaxH3VideoVae::decode_tiled_(const SharedBuffer& z, int LT, int lh,
                                      int lw, std::string* err)
{
  const Config& c = _cfg;
  // The tiles are laid out in PIXEL space and mapped back onto the
  // latent grid, which is what keeps a decode's boundaries on the same
  // pixels the matching encode used.
  const int H = lh * c.patch, W = lw * c.patch;
  const auto ys = minimax_h3::split_tiles(H, c.tile_size, c.tile_overlap_min,
                                          c.patch);
  const auto xs = minimax_h3::split_tiles(W, c.tile_size, c.tile_overlap_min,
                                          c.patch);
  if (ys.start.size() <= 1 && xs.start.size() <= 1) {
    // A clip small enough not to tile still ticks -- once per CHUNK,
    // which is what a "tile" degenerates to here. Leaving this path
    // silent is what made the bar render with no percentage at 256x256:
    // the whole frame is one tile, so the loop below never runs, and the
    // only geometry with no reporting was the small one where the
    // untiled decode is also the only thing happening.
    if (_prog_total <= 0) { _prog_total = 1; _prog_done = 0; }
    if (_tile_progress) { _tile_progress(_prog_done, _prog_total); }
    SharedBuffer one = decode(z, LT, lh, lw, err);
    if (!one.empty()) { ++_prog_done; }
    if (_tile_progress) { _tile_progress(_prog_done, _prog_total); }
    return one;
  }
  const int ZC = c.z_channels;
  // Called directly rather than through decode_video: report against this
  // call's own tiles, so a bare decode still paces a bar instead of
  // reporting against a total nobody set.
  if (_prog_total <= 0) {
    _prog_total = (int)(ys.start.size() * xs.start.size());
    _prog_done  = 0;
  }
  const std::size_t plane = (std::size_t)lh * lw;
  const auto* src = static_cast<const std::uint16_t*>(z.contents());
  // EACH TILE STITCHED AS IT FINISHES: its region reads only itself, the
  // tile above and the one to its left (stitch_tile_), so the previous
  // row of tiles and this one are all that is held -- where every tile
  // of the clip was kept until the end.
  const int OC = c.out_channels, OT = LT * c.patch_t;
  const int rows = (int)ys.start.size(), cols = (int)xs.start.size();
  SharedBuffer out =
      _mc->make_shared_buffer((std::size_t)OC * OT * H * W * 2);
  if (out.empty()) {
    if (err != nullptr) { *err = "tile allocation failed"; }
    return {};
  }
  auto* dst = static_cast<std::uint16_t*>(out.contents());
  std::vector<SharedBuffer> above((std::size_t)cols), row((std::size_t)cols);
  std::vector<float> tmp;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int y0 = ys.start[(std::size_t)i] / c.patch;
      const int th = ys.length[(std::size_t)i] / c.patch;
      const int x0 = xs.start[(std::size_t)j] / c.patch;
      const int tw = xs.length[(std::size_t)j] / c.patch;
      SharedBuffer sub =
          _mc->make_shared_buffer((std::size_t)ZC * LT * th * tw * 2);
      if (sub.empty()) {
        if (err != nullptr) { *err = "tile allocation failed"; }
        return {};
      }
      auto* d = static_cast<std::uint16_t*>(sub.contents());
      for (int ch = 0; ch < ZC; ++ch) {
        for (int t = 0; t < LT; ++t) {
          const std::size_t sb = ((std::size_t)ch * LT + t) * plane;
          const std::size_t db = ((std::size_t)ch * LT + t) * th * tw;
          for (int r = 0; r < th; ++r) {
            std::memcpy(d + db + (std::size_t)r * tw,
                        src + sb + (std::size_t)(y0 + r) * lw + x0,
                        (std::size_t)tw * 2);
          }
        }
      }
      if (_tile_progress) { _tile_progress(_prog_done, _prog_total); }
      SharedBuffer dec = decode(sub, LT, th, tw, err);
      if (dec.empty()) { return {}; }
      ++_prog_done;
      if (_tile_progress) { _tile_progress(_prog_done, _prog_total); }
      const std::size_t n = (std::size_t)OC * OT * (th * c.patch) *
                            (tw * c.patch);
      if (dec.byte_size() < n * 2) {
        if (err != nullptr) { *err = "tile stitch failed"; }
        return {};
      }
      auto bytes = [](const SharedBuffer& b) {
        return b.empty() ? nullptr
                         : static_cast<const std::uint16_t*>(b.contents());
      };
      stitch_tile_(dst, bytes(dec),
                   i > 0 ? bytes(above[(std::size_t)j]) : nullptr,
                   j > 0 ? bytes(row[(std::size_t)(j - 1)]) : nullptr, i, j,
                   ys, xs, 1, OC, OT, H, W, tmp);
      row[(std::size_t)j] = std::move(dec);
    }
    above.swap(row);
    for (SharedBuffer& b : row) { b = SharedBuffer{}; }
  }
  return out;
}

int
MetalMiniMaxH3VideoVae::video_latent_frames(int T) const
{
  return video_latent_frames_for(_cfg, T);
}

int
MetalMiniMaxH3VideoVae::video_latent_frames_for(const Config& cfg, int T)
{
  if (T <= 0) { return 0; }
  // A single frame has no temporal extent to chunk, and no tail is
  // dropped from it.
  if (T == 1) { return encoded_frames_for(cfg, 1); }
  const int cl = cfg.clip_length;
  if (cl <= 0) { return 0; }
  const int pad = ((-T % cl) + cl) % cl;
  const int n = (T + pad) / cl;
  const int per = encoded_frames_for(cfg, cl);
  const int total = n * per - cfg.token_drop;
  return total > 0 ? total : 0;
}

std::size_t
MetalMiniMaxH3VideoVae::latent_bytes(int width, int height, int frames)
{
  if (width <= 0 || height <= 0 || frames <= 0) { return 0; }
  const Config cfg;                       // the model's own ratios
  if (cfg.patch <= 0) { return 0; }
  const int lf = video_latent_frames_for(cfg, frames);
  if (lf <= 0) { return 0; }
  const std::size_t w = (std::size_t)((width + cfg.patch - 1) / cfg.patch);
  const std::size_t h = (std::size_t)((height + cfg.patch - 1) / cfg.patch);
  // bf16, and z_channels rather than 2*z_channels: an ENCODE emits mean
  // and logvar, but what travels between the denoise and the decodes is
  // the sampled latent, which is half of that.
  return w * h * (std::size_t)lf * (std::size_t)cfg.z_channels * 2;
}

int
MetalMiniMaxH3VideoVae::decoded_frames(int LT) const
{
  return decoded_frames_for(_cfg, LT);
}

int
MetalMiniMaxH3VideoVae::decoded_frames_for(const Config& c, int LT)
{
  const int tcs = c.tokens_per_chunk();
  if (LT <= 0 || tcs <= 0) { return 0; }
  const int td = c.token_drop;
  const int num_tokens = LT + td;
  const int pad_tokens = ((-num_tokens % tcs) + tcs) % tcs;
  const int chunks = (num_tokens + pad_tokens) / tcs - (td > 0 ? 1 : 0);
  if (chunks <= 0) { return 0; }
  const int cnf  = tcs * c.patch_t;
  const int clip = (tcs + c.token_overlap()) * c.patch_t;
  const int fpp  = c.frame_pre_pad();
  const int head = std::min(cnf, clip) - fpp;
  const int tail = td > 0 ? std::max(std::min(cnf, clip - cnf) - fpp, 0) : 0;
  int frames = chunks * head + tail;
  if (pad_tokens > 0) {
    // The repeated latent frames produced pixel frames nobody asked
    // for. A chunk's LAST latent frame only covers
    // `clip_length % patch_t` of them; every other covers `patch_t`.
    const int intra = c.clip_length % c.patch_t;
    for (int k = 0; k < pad_tokens; ++k) {
      frames -= (intra != 0 && (LT + k) % tcs == 0) ? intra : c.patch_t;
    }
  }
  return frames > 0 ? frames : 0;
}

SharedBuffer
MetalMiniMaxH3VideoVae::encode_video(const SharedBuffer& x, int T, int H,
                                     int W, int* latent_frames,
                                     std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  if (T <= 0) { return fail("empty video"); }
  // The counters belong to THIS call. Left set, a later bare
  // encode_tiled_ / decode_tiled_ would report against this clip's total
  // rather than its own, which reads as a bar that starts part-full.
  struct ProgReset {
    int& done;
    int& total;
    ~ProgReset() { done = 0; total = 0; }
  } prog_reset{_prog_done, _prog_total};
  // One bar for the WHOLE clip (clips x tiles), as decode_video does: a
  // clip boundary is invisible to whoever is watching. The tiling is the
  // same for every clip -- they all have the same extent.
  {
    const auto ys0 = minimax_h3::split_tiles(H, c.tile_size,
                                             c.tile_overlap_min, c.patch);
    const auto xs0 = minimax_h3::split_tiles(W, c.tile_size,
                                             c.tile_overlap_min, c.patch);
    const int tiles =
        std::max(1, (int)(ys0.start.size() * xs0.start.size()));
    const int cl0 = std::max(1, c.clip_length);
    const int calls = T == 1 ? 1 : (T + cl0 - 1) / cl0;
    _prog_total = calls * tiles;
    _prog_done  = 0;
  }
  _enc_prof.clear();
  if (T == 1) {
    SharedBuffer out = encode_tiled_(x, 1, H, W, err);
    if (!out.empty() && latent_frames != nullptr) { *latent_frames = 1; }
    prof_log_(1, H, W);
    return out;
  }
  const int cl = c.clip_length, IC = c.in_channels;
  if (cl <= 0) { return fail("clip_length is not positive"); }
  const int total = video_latent_frames(T);
  if (total <= 0) {
    return fail("too few frames to encode: " + std::to_string(T) +
                " leaves nothing after token_drop");
  }
  const int pad = ((-T % cl) + cl) % cl;
  const int nclips = (T + pad) / cl;
  const int per = encoded_frames(cl);
  const int lh = H / c.patch, lw = W / c.patch;

  SharedBuffer clip =
      _mc->make_shared_buffer((std::size_t)IC * cl * H * W * 2);
  SharedBuffer out = _mc->make_shared_buffer((std::size_t)(2 * c.z_channels) *
                                             total * lh * lw * 2);
  if (clip.empty() || out.empty()) { return fail("allocation failed"); }
  const std::size_t plane = (std::size_t)H * W;
  const std::size_t lplane = (std::size_t)lh * lw;
  const auto* src = static_cast<const std::uint16_t*>(x.contents());
  auto* dst = static_cast<std::uint16_t*>(out.contents());
  const int ZC2 = 2 * c.z_channels;

  for (int n = 0; n < nclips; ++n) {
    if (stopping()) { return {}; }
    const auto pc0 = std::chrono::steady_clock::now();
    auto* cd = static_cast<std::uint16_t*>(clip.contents());
    for (int ch = 0; ch < IC; ++ch) {
      for (int t = 0; t < cl; ++t) {
        // Past the end the LAST REAL frame repeats. Repeating it is
        // what the reference pads with; zeros would put a hard cut into
        // the causal window of the final latents.
        const int st = std::min(n * cl + t, T - 1);
        std::memcpy(cd + ((std::size_t)ch * cl + t) * plane,
                    src + ((std::size_t)ch * T + st) * plane, plane * 2);
      }
    }
    if (enc_profile_on_()) { prof_add_("host: clip copy", ms_since_(pc0)); }
    SharedBuffer enc = encode_tiled_(clip, cl, H, W, err);
    if (enc.empty()) { return {}; }
    const auto pc = std::chrono::steady_clock::now();
    const auto* ed = static_cast<const std::uint16_t*>(enc.contents());
    for (int ch = 0; ch < ZC2; ++ch) {
      for (int f = 0; f < per; ++f) {
        const int g = n * per + f;
        if (g >= total) { continue; }        // the token_drop tail
        std::memcpy(dst + ((std::size_t)ch * total + g) * lplane,
                    ed + ((std::size_t)ch * per + f) * lplane, lplane * 2);
      }
    }
    if (enc_profile_on_()) { prof_add_("host: clip copy", ms_since_(pc)); }
  }
  if (latent_frames != nullptr) { *latent_frames = total; }
  prof_log_(T, H, W);
  return out;
}

SharedBuffer
MetalMiniMaxH3VideoVae::decode_video(const SharedBuffer& z, int LT, int lh,
                                     int lw, int* out_frames,
                                     std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  // See encode_video: the tile counters are this call's, and must not
  // outlive it into a later decode_image.
  struct ProgReset {
    int& done;
    int& total;
    ~ProgReset() { done = 0; total = 0; }
  } prog_reset{_prog_done, _prog_total};
  const int tcs = c.tokens_per_chunk();
  if (LT <= 0 || lh <= 0 || lw <= 0 || tcs <= 0) {
    return fail("empty latent video");
  }
  const int td = c.token_drop, tr = c.patch_t;
  const int tov = c.token_overlap(), fpp = c.frame_pre_pad();
  const int fov = c.frame_overlap(), cnf = tcs * tr;
  const int num_tokens = LT + td;
  const int pad_tokens = ((-num_tokens % tcs) + tcs) % tcs;
  const int chunks = (num_tokens + pad_tokens) / tcs - (td > 0 ? 1 : 0);
  if (chunks <= 0) {
    // Below one chunk there is nothing to decode: `token_drop` already
    // removed more than a chunk's worth of the sequence's tail.
    return fail("only " + std::to_string(LT) + " latent frames: fewer than "
                "the " + std::to_string(tcs + td) + " one chunk needs");
  }
  const int ZC = c.z_channels;
  const int H = lh * c.patch, W = lw * c.patch;
  const std::size_t lplane = (std::size_t)lh * lw;
  const std::size_t plane  = (std::size_t)H * W;
  const int OC = c.out_channels;

  // Repeat the last latent frame up to a whole number of chunks; the
  // pixel frames it invents are cut off again at the end.
  const int zt = LT + pad_tokens;
  SharedBuffer zp = _mc->make_shared_buffer((std::size_t)ZC * zt * lplane * 2);
  if (zp.empty()) { return fail("allocation failed"); }
  {
    const auto* s = static_cast<const std::uint16_t*>(z.contents());
    auto* d = static_cast<std::uint16_t*>(zp.contents());
    for (int ch = 0; ch < ZC; ++ch) {
      for (int t = 0; t < zt; ++t) {
        const int st = std::min(t, LT - 1);
        std::memcpy(d + ((std::size_t)ch * zt + t) * lplane,
                    s + ((std::size_t)ch * LT + st) * lplane, lplane * 2);
      }
    }
  }

  const int span = tcs + tov;
  // THE CLIP ONCE, each chunk's frames written into it as the chunk is
  // decoded: a chunk yields its own frames and, when token_drop is on, a
  // trailing piece that is the NEXT chunk's cross-fade partner (and, after
  // the last chunk, the clip's tail). Every piece used to be kept and
  // copied into a second whole clip at the end -- two copies of the
  // decode at the peak. The frame count is the chunks' geometry alone.
  const int cf = span * tr;
  const int n_main = cf > fpp ? std::min(cnf, cf) - fpp : 0;
  const int n_tail =
      td > 0 && cf - cnf > fpp ? std::min(cnf, cf - cnf) - fpp : 0;
  const int frames = chunks * n_main + n_tail;
  int drop = 0;
  if (pad_tokens > 0) {
    const int intra = c.clip_length % tr;
    for (int k = 0; k < pad_tokens; ++k) {
      drop += (intra != 0 && (LT + k) % tcs == 0) ? intra : tr;
    }
  }
  const int keep = frames - drop;
  if (keep <= 0) { return fail("the pad trim removed every decoded frame"); }
  SharedBuffer out =
      _mc->make_shared_buffer((std::size_t)OC * keep * plane * 2);
  if (out.empty()) { return fail("output allocation failed"); }
  auto* dst = static_cast<std::uint16_t*>(out.contents());
  // Frames [at, at + n) of the clip from `src` ([OC, n_src, H, W], frames
  // from `f0`); those past the pad trim are not kept.
  auto put = [&](const std::uint16_t* src, int n_src, int f0, int at,
                 int n) {
    for (int f = 0; f < n && at + f < keep; ++f) {
      for (int ch = 0; ch < OC; ++ch) {
        std::memcpy(dst + ((std::size_t)ch * keep + at + f) * plane,
                    src + ((std::size_t)ch * n_src + f0 + f) * plane,
                    plane * 2);
      }
    }
  };
  // The previous chunk's trailing piece, kept for the cross-fade.
  SharedBuffer overlap;
  int overlap_n = 0;
  int at = 0;
  SharedBuffer sub =
      _mc->make_shared_buffer((std::size_t)ZC * span * lplane * 2);
  if (sub.empty()) { return fail("allocation failed"); }

  // The bar spans the WHOLE clip, not one chunk: a chunk boundary is
  // invisible to whoever is watching, and a bar that refilled per chunk
  // would read as progress restarting. Counted here because this is the
  // only place that knows both factors -- the tiling is decided per chunk
  // but is identical across them, since every chunk has the same extent.
  {
    const int H0 = lh * c.patch, W0 = lw * c.patch;
    const auto ys0 = minimax_h3::split_tiles(H0, c.tile_size,
                                             c.tile_overlap_min, c.patch);
    const auto xs0 = minimax_h3::split_tiles(W0, c.tile_size,
                                             c.tile_overlap_min, c.patch);
    _prog_total = chunks * (int)(ys0.start.size() * xs0.start.size());
    _prog_done  = 0;
  }
  for (int i = 0; i < chunks; ++i) {
    const int start = i * tcs;
    {
      const auto* s = static_cast<const std::uint16_t*>(zp.contents());
      auto* d = static_cast<std::uint16_t*>(sub.contents());
      for (int ch = 0; ch < ZC; ++ch) {
        for (int t = 0; t < span; ++t) {
          const int st = std::min(start + t, zt - 1);
          std::memcpy(d + ((std::size_t)ch * span + t) * lplane,
                      s + ((std::size_t)ch * zt + st) * lplane, lplane * 2);
        }
      }
    }
    SharedBuffer clip = decode_tiled_(sub, span, lh, lw, err);
    if (clip.empty()) { return {}; }
    const auto* cd = static_cast<const std::uint16_t*>(clip.contents());
    if (n_main > 0) {
      put(cd, cf, fpp, at, n_main);
      if (overlap_n > 0) {
        // Linear cross-fade over the frames the two chunks share, in the
        // clip itself: the same values in the same order as a piece's.
        const int ext = std::min(std::min(overlap_n, n_main), fov);
        const auto* od =
            static_cast<const std::uint16_t*>(overlap.contents());
        for (int ch = 0; ch < OC; ++ch) {
          for (int f = 0; f < ext && at + f < keep; ++f) {
            const float wb = (float)f / (float)ext;
            std::uint16_t* pd =
                dst + ((std::size_t)ch * keep + at + f) * plane;
            const std::size_t ob =
                ((std::size_t)ch * overlap_n + (overlap_n - ext + f)) *
                plane;
            for (std::size_t q = 0; q < plane; ++q) {
              pd[q] = f32_to_bf16_((1.0f - wb) * bf16_to_f32_(od[ob + q]) +
                                   wb * bf16_to_f32_(pd[q]));
            }
          }
        }
        overlap = SharedBuffer{};
        overlap_n = 0;
      }
      at += n_main;
    }
    if (n_tail > 0) {
      overlap = _mc->make_shared_buffer((std::size_t)OC * n_tail * plane * 2);
      if (overlap.empty()) { return fail("allocation failed"); }
      auto* tp = static_cast<std::uint16_t*>(overlap.contents());
      for (int ch = 0; ch < OC; ++ch) {
        std::memcpy(tp + (std::size_t)ch * n_tail * plane,
                    cd + ((std::size_t)ch * cf + cnf + fpp) * plane,
                    (std::size_t)n_tail * plane * 2);
      }
      overlap_n = n_tail;
    }
  }
  if (overlap_n > 0) {
    put(static_cast<const std::uint16_t*>(overlap.contents()), overlap_n, 0,
        at, overlap_n);
  }
  if (out_frames != nullptr) { *out_frames = keep; }
  return out;
}

SharedBuffer
MetalMiniMaxH3VideoVae::decode_image(const SharedBuffer& z, int h, int w,
                                     std::string* err)
{
  auto fail = [&](const std::string& m) -> SharedBuffer {
    if (err != nullptr) { *err = m; }
    return {};
  };
  const Config& c = _cfg;
  // Refused rather than attempted, and the message names what is
  // missing: every H3 video VAE has the tensors to run this and would
  // return a picture-shaped buffer of the wrong picture, so a caller who
  // pointed at the wrong checkpoint needs to be told which one to use
  // rather than left to judge the output.
  if (!c.t1_direct) {
    return fail("this checkpoint does not decode a single latent frame: it "
                "declares no 'h3_t1_direct' in its metadata. The chunked "
                "video decoder needs " +
                std::to_string(c.tokens_per_chunk() + c.token_drop) +
                " latent frames; a still is 1, which only an image-trained "
                "H3 VAE decodes");
  }
  if (h <= 0 || w <= 0) { return fail("empty latent grid"); }

  // One latent frame in, one whole `patch_t` block out. Tiled exactly as
  // a clip is, so a picture and a frame of a clip at the same canvas see
  // the same rope and the same seams.
  SharedBuffer block = decode_tiled_(z, 1, h, w, err);
  if (block.empty()) { return {}; }

  const int OC = c.out_channels;
  const int H  = h * c.patch;
  const int W  = w * c.patch;
  const std::size_t plane = (std::size_t)H * W;
  const int slice = c.t1_output_slice;
  // config_from_json bounds this, but decode_image is public and a Config
  // can also be hand-built, so the read is checked where it is used.
  if (slice < 0 || slice >= c.patch_t) {
    return fail("t1_output_slice " + std::to_string(slice) + " is not one of "
                "the " + std::to_string(c.patch_t) + " pixel frames a latent "
                "frame decodes to");
  }

  SharedBuffer out = _mc->make_shared_buffer((std::size_t)OC * plane * 2);
  if (out.empty()) { return fail("output allocation failed"); }
  const auto* src = static_cast<const std::uint16_t*>(block.contents());
  auto*       dst = static_cast<std::uint16_t*>(out.contents());
  for (int ch = 0; ch < OC; ++ch) {
    std::memcpy(dst + (std::size_t)ch * plane,
                src + ((std::size_t)ch * c.patch_t + slice) * plane,
                plane * 2);
  }
  return out;
}

}  // namespace genai
}  // namespace vpipe
