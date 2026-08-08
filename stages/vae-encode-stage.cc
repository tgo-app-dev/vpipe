#include "stages/vae-encode-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-detect.h"
#include "stages/model-registry.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

VaeEncodeStage::VaeEncodeStage(const SessionContextIntf* s,
                               std::string               id,
                               std::vector<InEdge>       iports,
                               FlexData                  config)
  : TypedStage<VaeEncodeStage>(s, std::move(id), std::move(iports),
                               std::move(config))
{
  // Deferred validation (see Stage::fail_config): construct for any config so
  // a graph can be built/edited before hf_dir is supplied. hf_dir is OPTIONAL
  // now -- a model-select source on the model iport can supply it instead --
  // so the "no model at all" case is reported at initialize()/process() time.
  _hf_dir    = attr_str("hf_dir");

  // Optional letterbox resize: target_width + target_height (both required
  // together, both multiples of 8 -- the VAE downsamples by 8, and the
  // downstream generate-image latent grid needs an even latent H/W).
  _frames   = (int)attr_int("frames");
  if (_frames <= 0) { _frames = 81; }
  _target_w = (int)attr_int("target_width");
  _target_h = (int)attr_int("target_height");
  if ((_target_w > 0) != (_target_h > 0)) {
    fail_config(fmt(
        "VaeEncodeStage('{}'): target_width and target_height must be set "
        "together (got {}x{})", this->id(), _target_w, _target_h));
  }
  if (_target_w > 0 && (_target_w % 8 != 0 || _target_h % 8 != 0)) {
    fail_config(fmt(
        "VaeEncodeStage('{}'): target_width/target_height must be multiples "
        "of 8 (got {}x{})", this->id(), _target_w, _target_h));
  }

  // Letterbox pad color: an [r,g,b] array (0..255), a named color, or a single
  // gray level. Defaults to black.
  parse_pad_color_();
#ifdef VPIPE_BUILD_APPLE_SILICON
  {
    bool bad = false;
    _unload_cfg = model_memory::parse_unload_policy(
        attr_str("unload_when_idle"), &bad);
    if (bad) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): unload_when_idle '{}' is not auto|always|never; "
          "using auto", this->id(), attr_str("unload_when_idle")));
    }
  }
#endif
  allocate_oports(spec().oports.size());
}

VaeEncodeStage::~VaeEncodeStage() = default;

namespace {

// MiniMax-H3's video VAE works in IMAGENET-normalized pixel space. The
// reference's own constants; vae-decode undoes this on the way out.
[[maybe_unused]] constexpr float kImagenetMean[3] = {0.485f, 0.456f,
                                                     0.406f};
[[maybe_unused]] constexpr float kImagenetStd[3]  = {0.229f, 0.224f,
                                                     0.225f};

// Clamp a numeric channel value into a 0..255 int.
int
clamp_u8_(double v)
{
  return (int)std::min(255.0, std::max(0.0, std::round(v)));
}

}  // namespace

void
VaeEncodeStage::parse_pad_color_()
{
  FlexData pc = attr("pad_color");
  if (pc.is_array()) {
    auto arr = pc.as_array();
    if (arr.size() >= 3) {
      _pad_r = clamp_u8_(arr[0].as_real(0.0));
      _pad_g = clamp_u8_(arr[1].as_real(0.0));
      _pad_b = clamp_u8_(arr[2].as_real(0.0));
    } else if (arr.size() == 1) {
      _pad_r = _pad_g = _pad_b = clamp_u8_(arr[0].as_real(0.0));
    }
    return;
  }
  if (pc.is_int() || pc.is_uint() || pc.is_real()) {
    _pad_r = _pad_g = _pad_b = clamp_u8_(pc.as_real(0.0));
    return;
  }
  if (pc.is_string()) {
    std::string s(pc.as_string(""));
    for (char& c : s) { c = (char)std::tolower((unsigned char)c); }
    if (s == "white") {
      _pad_r = _pad_g = _pad_b = 255;
    } else if (s == "gray" || s == "grey") {
      _pad_r = _pad_g = _pad_b = 128;
    } else {
      _pad_r = _pad_g = _pad_b = 0;   // "black" / unrecognized => black
    }
    return;
  }
  // Unset / null: leave the black default.
}

namespace {
// The model iport (a model-select source) overrides hf_dir. Appended after the
// primary `image` input, so it is iport1. (Referenced only from the Apple-gated
// code below; marked maybe_unused for the inert non-Apple build.)
[[maybe_unused]] constexpr unsigned kModelPort = 1;

const ConfigKey kAttrs[] = {
  {.key = "frames", .type = ConfigType::Int, .required = false,
   .doc = "VIDEO VAE only: how many video frames the image-to-video "
          "conditioning clip spans. The latent is the encoding of the "
          "conditioning image followed by that many minus one BLANK frames "
          "-- not of the image alone, because the VAE's temporal convolutions "
          "mix neighbouring frames, so a 1-frame encode is a different tensor. "
          "MUST match the generate-video stage's `frames`",
   .def_int = 81},
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "Krea-2-Turbo / FLUX.2 / Qwen-Image-Edit / Mage-Flow model dir (VAE "
          "read from <hf_dir>/vae). OPTIONAL: a model-select source on the "
          "model iport overrides it",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "krea2,flux2,qwen-image-edit,mage-flow,mage-flow-edit,"
       "boogu-image,boogu-image-edit"},
  {.key = "target_width", .type = ConfigType::Int, .required = false,
   .doc = "letterbox-resize the input to this width before encoding (multiple "
          "of 8; requires target_height). 0/unset = encode at native size"},
  {.key = "target_height", .type = ConfigType::Int, .required = false,
   .doc = "letterbox-resize target height (multiple of 8; requires "
          "target_width)"},
  {.key = "pad_color", .type = ConfigType::Any, .required = false,
   .doc = "letterbox pad color: [r,g,b] 0..255, a name "
          "(black/white/gray), or a single gray level. Default black"},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "drop the VAE weights after each beat and reload on the next one. "
          "This stage is idle for the whole denoise, so on a memory-bounded box "
          "releasing it (weights AND the decode working set) is what lets a "
          "large DiT run in the same machine. \"auto\" (default) decides from "
          "physical RAM vs the pipeline's weight bytes; \"always\" / "
          "\"never\" force it",
   .def_str = "auto"},
};
const PortSpec kIports[] = {
  {.name = "image", .doc = "U8 or f32 RGB image [3,H,W] (channel-first, U8 "
                           "0..255 or f32 [-1,1])",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
  {.name = "model", .doc = "OPTIONAL shared model reference from a model-select "
                           "source; overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "latent", .doc = "f32 whitened latent [z_dim, H/8, W/8] (unpacked)",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "vae-encode",
  .doc       = "Encodes an RGB image into a Krea-2-Turbo (Qwen-Image VAE) "
               "whitened latent on the metal-compute backend. The mirror of "
               "vae-decode; feeds the generate-image `latent` port (img2img).",
  .display_name = "VAE Encode",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

#ifdef VPIPE_BUILD_APPLE_SILICON
// VAE family from the vae config.json `_class_name` ("AutoencoderKLFlux2" ->
// "flux2"; "MageVAE" -> "mage"; else "krea2").
std::string
vae_family_(const std::string& vae_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(vae_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto obj = fd.as_object();
      if (obj.contains("_class_name")) {
        const std::string cls(obj.at("_class_name").as_string(""));
        if (cls == "AutoencoderKLFlux2") { return "flux2"; }
        // The PLAIN diffusers AutoencoderKL (the FLUX.1 VAE Boogu-Image uses)
        // runs on the SAME code path -- MetalFlux2Vae is config-driven, and at
        // patch 1 with a scalar shift/scale whitening it IS a plain
        // AutoencoderKL. Same family string, so the branches below are shared.
        if (cls == "AutoencoderKL") { return "flux2"; }
        if (cls == "MageVAE") { return "mage"; }
        if (cls == "AutoencoderKLWan") { return "wan"; }
        if (cls == "MiniMaxH3VideoVAE") { return "minimax-h3"; }
      }
    }
  }
  return "krea2";
}

// MageVAE geometry from vae/config.json (see the vae-decode twin).
genai::MetalMageVae::Config
mage_vae_config_(const std::string& vae_dir)
{
  genai::MetalMageVae::Config c;
  std::ifstream in(std::filesystem::path(vae_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      if (o.contains("latent_channels")) {
        c.latent_channels =
            (int)o.at("latent_channels").as_int(c.latent_channels);
      }
      if (o.contains("downsample_factor")) {
        c.patch = (int)o.at("downsample_factor").as_int(c.patch);
      }
    }
  }
  return c;
}
#endif
}  // namespace

const StageSpec&
VaeEncodeStage::spec() const noexcept
{
  return kSpec;
}

#ifdef VPIPE_BUILD_APPLE_SILICON

void
VaeEncodeStage::reset_run_state()
{
  // If a previous run left the weights UNLOADED (the idle-unload
  // policy drops them between beats), let this launch load them again:
  // ensure_loaded_'s once-only guard is per-Stage, not per-launch, so
  // without this the stage stays inert for the whole run. When the
  // weights are still held we deliberately leave the guard set --
  // reloading on top of a resident copy is exactly what doubles peak
  // memory.
  if (!_vae && !_flux2_vae && !_mage_vae) {
    _load_attempted = false;
    _unloaded       = false;
  }

  // Per-launch reset: the stage survives a stop/relaunch, and the
  // select sources upstream re-emit on every launch. Without this the
  // re-emitted beat is never latched and this stage keeps the previous
  // run's selection.
  _model_latched = false;

}

std::vector<ResourceClaim>
VaeEncodeStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  return model_memory::weight_claims({resolve_vae_dir(root)});
}

Job
VaeEncodeStage::initialize(RuntimeContext& ctx)
{
  // Nothing is loaded here, deliberately. This stage encodes REFERENCE
  // images, and a graph that is wired for one but never sent one -- a
  // text-to-image run through an edit-capable pipeline -- must not pay
  // for the encoder's weights. The load happens on the first image
  // instead (see process()), so "no reference image" costs nothing.
  //
  // The cost of deferring is that the load lands inside the first beat
  // rather than at the init barrier. It is the same work either way, and
  // an edit run needs it before it can produce anything regardless.
  (void)ctx;
  co_return;
}

// Loading chatter: info on the first load, debug on an idle-unload reload (a
// bounded box reloads per beat, and one line per frame is noise).
void
VaeEncodeStage::load_note_(const VpipeFormat& msg) const
{
  if (_quiet_reload) { session()->log_debug(msg); }
  else               { session()->info(msg); }
}

void
VaeEncodeStage::unload_vae_()
{
  if (!_vae && !_flux2_vae && !_mage_vae) { return; }
  _vae.reset();
  _flux2_vae.reset();
  _mage_vae.reset();
  _unloaded = true;
  _quiet_reload = true;
  session()->log_debug(fmt("VaeEncodeStage('{}'): VAE encoder unloaded (idle)",
                           this->id()));
}

void
VaeEncodeStage::reload_vae_()
{
  if (!_unloaded) { return; }
  _unloaded = false;
  _load_attempted = false;      // let ensure_loaded_ run its body again
  ensure_loaded_();
}

void
VaeEncodeStage::ensure_loaded_()
{
  if (_load_attempted) { return; }   // idempotent: load at most once
  _load_attempted = true;
  if (_hf_dir.empty()) {
    session()->error(fmt(
        "VaeEncodeStage('{}'): no model -- set config.hf_dir or wire a "
        "model-select source to the model iport; inert", this->id()));
    return;
  }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "VaeEncodeStage('{}'): no metal-compute backend on this session; "
        "the stage is inert", this->id()));
    return;
  }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  // Idle-unload decision (auto: the DiT + text encoder of this same pipeline are
  // resident during a run, so size the box against THEIR weights -- the VAE's
  // own are small, it is the decode working set that has to fit beside them).
  if (!_unload_resolved) {
    _unload_resolved = true;
    const std::vector<std::string> peers = {
        (std::filesystem::path(root) / "transformer").string(),
        (std::filesystem::path(root) / "text_encoder").string(),
        (std::filesystem::path(root) / "mllm").string()};
    const std::size_t fp = model_memory::weight_footprint(session(), peers);
    switch (_unload_cfg) {
      case model_memory::UnloadPolicy::kAlways: _unload_idle = true;  break;
      case model_memory::UnloadPolicy::kNever:  _unload_idle = false; break;
      default:
        _unload_idle =
            model_memory::bounded(session(), peers, model_memory::kHeadroom);
        break;
    }
    session()->log_debug(fmt(
        "VaeEncodeStage('{}'): peer footprint {} MB + {} MB headroom vs {} MB "
        "RAM, unload_when_idle={} -> {}", this->id(), fp >> 20,
        model_memory::kHeadroom >> 20, model_memory::phys_ram() >> 20,
        model_memory::unload_policy_name(_unload_cfg),
        _unload_idle ? "UNLOAD after each beat" : "keep resident"));
  }
  namespace fs = std::filesystem;
  // `vae/` for every diffusers checkpoint, `video_vae/source/` for
  // MiniMax-H3. Shared with vae-encode so the two halves of one model
  // can never resolve to different directories.
  const std::string vae_dir = resolve_vae_dir(root);

  _family = vae_family_(vae_dir);
  // One shared, reference-counted view of this checkpoint. The peer VAE
  // stage in the same graph (encode beside decode) names the same
  // directory and gets the SAME set, so the two halves of the model are
  // loaded once between them instead of once each -- and two pipelines
  // running the same model share it too.
  // Falls back to a private set when the session has no manager; see
  // open_weight_set().
  // MiniMax-H3 keeps the WEIGHTS one level below the config that named
  // the family, so the set is opened on the resolved leaf. Every other
  // family's resolver returns what it was handed.
  const std::string ws_dir =
      (_family == "minimax-h3")
          ? genai::MetalMiniMaxH3VideoVae::resolve_vae_dir(vae_dir)
          : vae_dir;
  std::shared_ptr<genai::WeightSet> ws =
      genai::open_weight_set(ws_dir, session());
  if (!ws) {
    session()->error(fmt(
        "{}('{}'): no readable checkpoint under '{}'; inert",
        "VaeEncodeStage", this->id(), vae_dir));
    return;
  }
  if (_family == "flux2") {
    genai::MetalFlux2Vae::Config fcfg;
    std::ifstream in(fs::path(vae_dir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto obj = fd.as_object();
        if (obj.contains("latent_channels")) {
          fcfg.latent_channels =
              (int)obj.at("latent_channels").as_int(fcfg.latent_channels);
        }
        if (obj.contains("norm_num_groups")) {
          fcfg.norm_groups =
              (int)obj.at("norm_num_groups").as_int(fcfg.norm_groups);
        }
        if (obj.contains("block_out_channels")) {
          FlexData bo = obj.at("block_out_channels");
          auto sp = bo.as_real_span();
          for (int i = 0; i < 4 && i < (int)sp.size(); ++i) {
            fcfg.block_out[i] = (int)sp[(std::size_t)i];
          }
        }
        if (obj.contains("layers_per_block")) {
          fcfg.layers_per_block =
              (int)obj.at("layers_per_block").as_int(fcfg.layers_per_block);
        }
      }
    }
    load_note_(fmt(
        "VaeEncodeStage('{}'): loading AutoencoderKL encoder from '{}'", this->id(),
        vae_dir));
    _flux2_vae = genai::MetalFlux2Vae::load(ws, mc, fcfg,
                                            /*with_encoder=*/true);
    if (!_flux2_vae || !_flux2_vae->has_encoder()) {
      session()->error(fmt(
          "VaeEncodeStage('{}'): failed to load the AutoencoderKL encoder from "
          "'{}'; inert", this->id(), vae_dir));
      _flux2_vae.reset();
    }
    return;
  }

  if (_family == "minimax-h3") {
    genai::MetalMiniMaxH3VideoVae::Config h3cfg;
    std::string h3err;
    if (!genai::MetalMiniMaxH3VideoVae::config_from_json(vae_dir, h3cfg,
                                                         &h3err)) {
      session()->error(fmt("VaeEncodeStage('{}'): {}; inert", this->id(),
                           h3err));
      return;
    }
    load_note_(fmt("VaeEncodeStage('{}'): loading the MiniMax-H3 video VAE "
                   "(encoder) from '{}'", this->id(), vae_dir));
    _h3_vae = genai::MetalMiniMaxH3VideoVae::load(vae_dir, mc, h3cfg);
    if (!_h3_vae) {
      session()->error(fmt(
          "VaeEncodeStage('{}'): failed to load the MiniMax-H3 video VAE "
          "from '{}'; inert", this->id(), vae_dir));
    }
    return;
  }

  if (_family == "wan") {
    genai::MetalWanVae::Config wcfg;
    std::string werr;
    if (!genai::MetalWanVae::config_from_json(vae_dir, wcfg, &werr)) {
      session()->error(fmt("VaeEncodeStage('{}'): {}; inert", this->id(),
                           werr));
      return;
    }
    load_note_(fmt("VaeEncodeStage('{}'): loading Wan video VAE (encoder) "
                   "from '{}'", this->id(), vae_dir));
    _wan_vae = genai::MetalWanVae::load(ws, mc, wcfg, /*with_encoder=*/true);
    if (!_wan_vae) {
      session()->error(fmt(
          "VaeEncodeStage('{}'): failed to load the Wan VAE from '{}'; inert",
          this->id(), vae_dir));
    }
    return;
  }

  if (_family == "mage") {
    load_note_(fmt("VaeEncodeStage('{}'): loading MageVAE encoder from "
                        "'{}'", this->id(), vae_dir));
    _mage_vae = genai::MetalMageVae::load(ws, mc,
                                          mage_vae_config_(vae_dir),
                                          /*with_encoder=*/true);
    if (!_mage_vae || !_mage_vae->has_encoder()) {
      session()->error(fmt(
          "VaeEncodeStage('{}'): failed to load the MageVAE encoder from '{}'; "
          "inert", this->id(), vae_dir));
      _mage_vae.reset();
    }
    return;
  }

  genai::MetalKrea2Vae::Config cfg;   // Qwen-Image VAE defaults
  // Read z_dim / base_dim / num_res_blocks + the per-channel latent statistics
  // (latents_mean / latents_std, needed to whiten the encoded latent).
  {
    std::ifstream in(fs::path(vae_dir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto obj = fd.as_object();
        if (obj.contains("z_dim")) {
          cfg.z_dim = (int)obj.at("z_dim").as_int(cfg.z_dim);
        }
        if (obj.contains("base_dim")) {
          cfg.base_dim = (int)obj.at("base_dim").as_int(cfg.base_dim);
        }
        if (obj.contains("num_res_blocks")) {
          cfg.num_res_blocks =
              (int)obj.at("num_res_blocks").as_int(cfg.num_res_blocks);
        }
        if (obj.contains("latents_mean")) {
          FlexData lm = obj.at("latents_mean");
          for (auto v : lm.as_real_span()) { cfg.latents_mean.push_back((float)v); }
        }
        if (obj.contains("latents_std")) {
          FlexData ls = obj.at("latents_std");
          for (auto v : ls.as_real_span()) { cfg.latents_std.push_back((float)v); }
        }
      }
    }
  }
  if ((int)cfg.latents_mean.size() != cfg.z_dim ||
      (int)cfg.latents_std.size() != cfg.z_dim) {
    session()->error(fmt(
        "VaeEncodeStage('{}'): vae config.json is missing latents_mean/"
        "latents_std ({}/{} of z_dim {}); the stage is inert", this->id(),
        cfg.latents_mean.size(), cfg.latents_std.size(), cfg.z_dim));
    return;
  }

  load_note_(fmt(
      "VaeEncodeStage('{}'): loading Qwen-Image VAE encoder from '{}'",
      this->id(), vae_dir));
  _vae = genai::MetalKrea2Vae::load(ws, mc, cfg, /*with_encoder=*/true);
  if (!_vae || !_vae->has_encoder()) {
    session()->error(fmt(
        "VaeEncodeStage('{}'): failed to load the VAE encoder from '{}'; "
        "inert", this->id(), vae_dir));
    _vae.reset();
    return;
  }
  session()->log_debug(fmt(
      "VaeEncodeStage('{}'): VAE encoder ready from '{}' (z_dim {}, base_dim "
      "{}, num_res_blocks {})", this->id(), vae_dir, cfg.z_dim, cfg.base_dim,
      cfg.num_res_blocks));
}

namespace {

// Anti-aliased separable resample weights for one axis (srcN -> dstN). A
// triangle (linear) filter whose support scales with the downscale ratio: on
// downscale it averages the full source footprint (no aliasing), and on upscale
// (ratio <= 1) it reduces EXACTLY to 2-tap bilinear. Each entry gives the first
// source index and its contiguous, normalized weights.
struct AxisContrib { int i0; std::vector<float> w; };
static std::vector<AxisContrib>
build_resample_(int srcN, int dstN)
{
  std::vector<AxisContrib> c((std::size_t)dstN);
  const double ratio = (double)srcN / (double)dstN;
  const double fscale = std::max(1.0, ratio);   // filter scale, in source px
  for (int o = 0; o < dstN; ++o) {
    const double center = (o + 0.5) * ratio - 0.5;
    const int i0 = (int)std::ceil(center - fscale);
    const int i1 = (int)std::floor(center + fscale);
    AxisContrib& e = c[(std::size_t)o];
    e.i0 = i0;
    double sum = 0.0;
    for (int i = i0; i <= i1; ++i) {
      const double t = std::fabs((i - center) / fscale);
      const double w = t < 1.0 ? (1.0 - t) : 0.0;
      e.w.push_back((float)w);
      sum += w;
    }
    if (sum > 0.0) { for (float& w : e.w) { w = (float)(w / sum); } }
  }
  return c;
}

// Produce a normalized f32 [-1,1] planar RGB [3,outH,outW] from the source
// image bytes (channel-first [3,sH,sW], U8 0..255 or f32 already in [-1,1]).
// When the target differs from the source the picture is scaled to fit with its
// original aspect ratio (anti-aliased separable resample), centered, and the
// leftover border is filled with `pad` (per-channel, already mapped to [-1,1]).
// Same size => straight normalize, no resample.
std::vector<float>
normalize_and_fit_(const std::uint8_t* src, bool is_u8, int sH, int sW,
                   int outH, int outW, const float pad[3])
{
  auto src_val = [&](int c, int y, int x) -> float {
    const std::size_t idx = ((std::size_t)c * sH + y) * sW + x;
    if (is_u8) { return (float)src[idx] / 255.0f * 2.0f - 1.0f; }
    return reinterpret_cast<const float*>(src)[idx];
  };
  std::vector<float> out((std::size_t)3 * outH * outW);
  if (sH == outH && sW == outW) {
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
          out[((std::size_t)c * outH + y) * outW + x] = src_val(c, y, x);
        }
      }
    }
    return out;
  }

  // Aspect-preserving fit + centered placement.
  const double scale = std::min((double)outW / sW, (double)outH / sH);
  const int newW = std::max(1, (int)std::lround(sW * scale));
  const int newH = std::max(1, (int)std::lround(sH * scale));
  const int offX = (outW - newW) / 2;
  const int offY = (outH - newH) / 2;
  for (int c = 0; c < 3; ++c) {
    const std::size_t base = (std::size_t)c * outH * outW;
    for (std::size_t i = 0; i < (std::size_t)outH * outW; ++i) {
      out[base + i] = pad[c];
    }
  }
  // Separable anti-aliased resample: horizontal (sW -> newW) into a scratch,
  // then vertical (sH -> newH) into the centered output window. The triangle
  // filter's footprint = the downscale ratio, so shrinking a high-res photo no
  // longer aliases fine detail (pleats, hair); an upscale stays 2-tap bilinear.
  const std::vector<AxisContrib> cx = build_resample_(sW, newW);
  const std::vector<AxisContrib> cy = build_resample_(sH, newH);
  std::vector<float> tmp((std::size_t)3 * sH * newW);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < sH; ++y) {
      for (int ox = 0; ox < newW; ++ox) {
        const AxisContrib& e = cx[(std::size_t)ox];
        float acc = 0.0f;
        for (std::size_t k = 0; k < e.w.size(); ++k) {
          const int sx = std::min(std::max(e.i0 + (int)k, 0), sW - 1);
          acc += e.w[k] * src_val(c, y, sx);
        }
        tmp[((std::size_t)c * sH + y) * newW + ox] = acc;
      }
    }
  }
  for (int c = 0; c < 3; ++c) {
    for (int oy = 0; oy < newH; ++oy) {
      const AxisContrib& e = cy[(std::size_t)oy];
      for (int ox = 0; ox < newW; ++ox) {
        float acc = 0.0f;
        for (std::size_t k = 0; k < e.w.size(); ++k) {
          const int sy = std::min(std::max(e.i0 + (int)k, 0), sH - 1);
          acc += e.w[k] * tmp[((std::size_t)c * sH + sy) * newW + ox];
        }
        out[((std::size_t)c * outH + (offY + oy)) * outW + (offX + ox)] = acc;
      }
    }
  }
  return out;
}

}  // namespace

Job
VaeEncodeStage::process(RuntimeContext& ctx)
{
  // Latch the shared model (iport1) once -- a model-select source overrides the
  // hf_dir config -- then lazily load the VAE before the first encode.
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      // Records the selection only -- the weights still wait for an
      // actual image, so a model-select beat alone never loads anything.
      apply_model_select_beat(mfd->data, _hf_dir);
    }
  }
  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }   // upstream EOS -> close oport
  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (tbp == nullptr || tbp->shape.size() != 3 || tbp->shape[0] != 3 ||
      (tbp->dtype != TensorBeat::DType::U8 &&
       tbp->dtype != TensorBeat::DType::F32)) {
    session()->warn(fmt(
        "VaeEncodeStage('{}'): expected a U8/f32 RGB [3,H,W] TensorBeat, got "
        "{}; skipping", this->id(), in->describe()));
    co_return;
  }
  // THIS is where the encoder's weights are read: a real reference image
  // has arrived, so they will actually be used. Idempotent after the
  // first beat.
  ensure_loaded_();
  // A previous beat may have dropped the encoder to leave the DiT room.
  if (_unloaded) { reload_vae_(); }

  // ---- 2D AutoencoderKL: encode to [dit_channels, H/px, W/px] with px =
  // 8*patch -- 16 on FLUX.2 (8x VAE + 2x patch), 8 on the plain FLUX.1 VAE
  // Boogu uses. The multiple-of-16 requirement holds for both: Boogu's DiT
  // patches its H/8 latent 2x2, so the pixel dims still have to be even
  // multiples of 8, and its own image processor uses vae_scale_factor 16. ----
  if (_family == "flux2") {
    if (!_flux2_vae) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): AutoencoderKL encoder not loaded; skipping",
          this->id()));
      co_return;
    }
    const int sH = (int)tbp->shape[1], sW = (int)tbp->shape[2];
    if (sH <= 0 || sW <= 0) { co_return; }
    const bool resize = _target_w > 0 && _target_h > 0;
    const int H = resize ? _target_h : sH;
    const int W = resize ? _target_w : sW;
    if (!resize && ((H % 16) != 0 || (W % 16) != 0)) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): image [{}x{}] must be a positive "
          "multiple of 16 (or set target_width/height); skipping", this->id(),
          sW, sH));
      co_return;
    }
    auto* mc = session()->services()->metal_compute();
    const auto img = tbp->materialize_contiguous();
    const bool is_u8 = tbp->dtype == TensorBeat::DType::U8;
    const float pad[3] = {
      (float)_pad_r / 255.0f * 2.0f - 1.0f,
      (float)_pad_g / 255.0f * 2.0f - 1.0f,
      (float)_pad_b / 255.0f * 2.0f - 1.0f,
    };
    const std::vector<float> norm =
        normalize_and_fit_(img.data(), is_u8, sH, sW, H, W, pad);
    const std::size_t n = (std::size_t)3 * H * W;
    metal_compute::SharedBuffer imgbuf = mc->make_shared_buffer(n * 2);
    if (imgbuf.empty()) { co_return; }
    { auto* d = static_cast<_Float16*>(imgbuf.contents());
      for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)norm[i]; } }
    metal_compute::SharedBuffer lat;
    {   // LLM-lane perf event: one VAE encode per reference image.
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae,
                         kPerfLlmVaeBegin, (std::uint64_t)H * W);
      lat = _flux2_vae->encode(imgbuf, H, W);
    }
    if (lat.empty()) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): AutoencoderKL encode failed; skipping",
          this->id()));
      co_return;
    }
    const int Cdit = _flux2_vae->config().dit_channels();
    // Pixels per latent cell: 8x conv trunk times the patch factor (2 on
    // AutoencoderKLFlux2, 1 on the plain AutoencoderKL Boogu uses).
    const int px = 8 * _flux2_vae->config().patch;
    const int lh = H / px, lw = W / px;
    const std::size_t nz = (std::size_t)Cdit * lh * lw;
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {Cdit, lh, lw};
    out->resize_contiguous(nz);
    const auto* lp = static_cast<const _Float16*>(lat.contents());
    float* op = out->as_f32();
    for (std::size_t i = 0; i < nz; ++i) { op[i] = (float)lp[i]; }
    ++_latents_emitted;
    session()->log_debug(fmt(
        "VaeEncodeStage('{}'): AutoencoderKL encoded latent #{} [{}, {}, {}]",
        this->id(), _latents_emitted, Cdit, lh, lw));
    if (_unload_idle) { unload_vae_(); }
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Mage-Flow MageVAE: encode to [128, H/16, W/16] (16x, patch_size 1 in
  // the DiT), so the image must be a multiple of 16. The posterior is NOT
  // sampled (vae/config.json sample_posterior:false) -- encode returns the
  // MEAN, so this is deterministic and needs no whitening. ----
  if (_family == "mage") {
    if (!_mage_vae) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): MageVAE encoder not loaded; skipping",
          this->id()));
      co_return;
    }
    const int P = _mage_vae->config().patch;
    const int sH = (int)tbp->shape[1], sW = (int)tbp->shape[2];
    if (sH <= 0 || sW <= 0) { co_return; }
    const bool resize = _target_w > 0 && _target_h > 0;
    const int H = resize ? _target_h : sH;
    const int W = resize ? _target_w : sW;
    if ((H % P) != 0 || (W % P) != 0) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): MageVAE image [{}x{}] must be a positive "
          "multiple of {} (or set target_width/height); skipping", this->id(),
          W, H, P));
      co_return;
    }
    auto* mc = session()->services()->metal_compute();
    const auto img = tbp->materialize_contiguous();
    const bool is_u8 = tbp->dtype == TensorBeat::DType::U8;
    const float pad[3] = {
      (float)_pad_r / 255.0f * 2.0f - 1.0f,
      (float)_pad_g / 255.0f * 2.0f - 1.0f,
      (float)_pad_b / 255.0f * 2.0f - 1.0f,
    };
    const std::vector<float> norm =
        normalize_and_fit_(img.data(), is_u8, sH, sW, H, W, pad);
    const std::size_t n = (std::size_t)3 * H * W;
    metal_compute::SharedBuffer imgbuf = mc->make_shared_buffer(n * 2);
    if (imgbuf.empty()) { co_return; }
    { auto* d = static_cast<_Float16*>(imgbuf.contents());
      for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)norm[i]; } }
    std::string eerr;
    metal_compute::SharedBuffer lat;
    {
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae,
                         kPerfLlmVaeBegin, (std::uint64_t)H * W);
      lat = _mage_vae->encode(imgbuf, H, W, &eerr);
    }
    if (lat.empty()) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): MageVAE encode failed ({}); skipping",
          this->id(), eerr.empty() ? "unknown error" : eerr));
      co_return;
    }
    const int Cz = _mage_vae->config().latent_channels;
    const int lh = H / P, lw = W / P;
    const std::size_t nz = (std::size_t)Cz * lh * lw;
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {Cz, lh, lw};
    out->resize_contiguous(nz);
    const auto* lp = static_cast<const _Float16*>(lat.contents());
    float* op = out->as_f32();
    for (std::size_t i = 0; i < nz; ++i) { op[i] = (float)lp[i]; }
    ++_latents_emitted;
    session()->log_debug(fmt(
        "VaeEncodeStage('{}'): MageVAE encoded latent #{} [{}, {}, {}]",
        this->id(), _latents_emitted, Cz, lh, lw));
    if (_unload_idle) { unload_vae_(); }
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Wan video VAE: the IMAGE-TO-VIDEO conditioning latent ---------
  // The reference does not encode the conditioning image on its own. It
  // builds a clip -- the image followed by `frames - 1` blank frames --
  // and encodes THAT, because the VAE's causal temporal convolutions mix
  // each output frame with its two predecessors. A 1-frame encode is a
  // different tensor, and the DiT would be conditioned on something it was
  // never trained against.
  // ---- MiniMax-H3: one KEYFRAME anchor ------------------------------
  // FL2VA conditions on real encoded frames rather than on a masked
  // blank clip the way Wan's i2v does, so this encodes the image ALONE
  // and emits a one-latent-frame anchor. The packing convention
  // downstream is "one latent frame = one anchor", so a two-frame beat
  // is a first-AND-last request; this stage sees one image and emits
  // one.
  if (_family == "minimax-h3") {
    if (!_h3_vae) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): MiniMax-H3 VAE not loaded; skipping",
          this->id()));
      co_return;
    }
    const auto& vc = _h3_vae->config();
    const int sH = (int)tbp->shape[1], sW = (int)tbp->shape[2];
    const bool rs = _target_w > 0 && _target_h > 0;
    const int H = rs ? _target_h : sH;
    const int W = rs ? _target_w : sW;
    if (H <= 0 || W <= 0 || (H % vc.patch) != 0 || (W % vc.patch) != 0) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): image [{}x{}] must be a positive multiple of "
          "{} (or set target_width/target_height); skipping", this->id(), sW,
          sH, vc.patch));
      co_return;
    }
    auto* mc = session()->services()->metal_compute();
    const auto img = tbp->materialize_contiguous();
    const bool u8 = tbp->dtype == TensorBeat::DType::U8;
    const float pad[3] = {
      (float)_pad_r / 255.0f * 2.0f - 1.0f,
      (float)_pad_g / 255.0f * 2.0f - 1.0f,
      (float)_pad_b / 255.0f * 2.0f - 1.0f,
    };
    const std::vector<float> norm =
        normalize_and_fit_(img.data(), u8, sH, sW, H, W, pad);
    const std::size_t plane = (std::size_t)H * W;
    metal_compute::SharedBuffer frame =
        mc->make_shared_buffer((std::size_t)3 * plane * 2);
    if (frame.empty()) { co_return; }
    {
      auto* d = static_cast<std::uint16_t*>(frame.contents());
      // This VAE's pixel space is IMAGENET-NORMALIZED, not [-1, 1]:
      // the reference encoder's first act is `(x + 1)/2` then
      // `(. - mean)/std`. `normalize_and_fit_` only gets as far as
      // [-1, 1], so without this the encoder sees an input ~4.4x too
      // wide and the anchor latent it produces is wrong. vae-decode
      // undoes exactly this on the way out.
      for (int c = 0; c < 3; ++c) {
        const float pm = kImagenetMean[c], ps = kImagenetStd[c];
        for (std::size_t i = 0; i < plane; ++i) {
          const std::size_t k = (std::size_t)c * plane + i;
          const float v = ((norm[k] + 1.0f) * 0.5f - pm) / ps;
          std::uint32_t u;
          std::memcpy(&u, &v, 4);
          d[k] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
        }
      }
    }
    int lf = 0;
    std::string eerr;
    metal_compute::SharedBuffer mom;
    {
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae,
                         kPerfLlmVaeBegin, (std::uint64_t)H * W);
      mom = _h3_vae->encode_video(frame, 1, H, W, &lf, &eerr);
    }
    if (mom.empty() || lf <= 0) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): keyframe encode failed ({}); skipping",
          this->id(), eerr.empty() ? "unknown error" : eerr));
      co_return;
    }
    // The MEAN half of the moments, then WHITENED into the normalized
    // space the DiT generates in -- the same transform vae-decode
    // undoes. Sampling the posterior instead would put noise into an
    // anchor whose whole job is to be exact.
    const int Cz = vc.z_channels;
    const int lh = H / vc.patch, lw = W / vc.patch;
    const std::size_t vox = (std::size_t)lf * lh * lw;
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {Cz, lf, lh, lw};
    out->resize_contiguous((std::size_t)Cz * vox);
    const auto* mp = static_cast<const std::uint16_t*>(mom.contents());
    float* op = out->as_f32();
    const bool whiten = (int)vc.latents_mean.size() == Cz &&
                        (int)vc.latents_std.size() == Cz;
    for (int c = 0; c < Cz; ++c) {
      const float mu = whiten ? vc.latents_mean[(std::size_t)c] : 0.0f;
      const float sd = whiten ? vc.latents_std[(std::size_t)c] : 1.0f;
      for (std::size_t i = 0; i < vox; ++i) {
        const std::size_t k = (std::size_t)c * vox + i;
        const std::uint32_t u = (std::uint32_t)mp[k] << 16;
        float x;
        std::memcpy(&x, &u, 4);
        op[k] = sd != 0.0f ? (x - mu) / sd : (x - mu);
      }
    }
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("anchors", FlexData::make_string("first"));
    out->sideband = std::move(sb);
    session()->log_debug(fmt(
        "VaeEncodeStage('{}'): keyframe [{}x{}] -> anchor latent "
        "[{}, {}, {}, {}]", this->id(), W, H, Cz, lf, lh, lw));
    ++_latents_emitted;
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  if (_family == "wan") {
    if (!_wan_vae || !_wan_vae->has_encoder()) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): Wan VAE encoder not loaded; skipping",
          this->id()));
      co_return;
    }
    const int sH = (int)tbp->shape[1], sW = (int)tbp->shape[2];
    const bool rs = _target_w > 0 && _target_h > 0;
    const int H = rs ? _target_h : sH;
    const int W = rs ? _target_w : sW;
    if (H <= 0 || W <= 0 || (H % 8) != 0 || (W % 8) != 0) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): image [{}x{}] must be a positive multiple of "
          "8 (or set target_width/target_height); skipping", this->id(), sW,
          sH));
      co_return;
    }
    if ((_frames % 4) != 1 || _frames <= 0) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): frames {} is not 4k+1; the video VAE "
          "compresses time in 4-frame chunks after a 1-frame first chunk, so "
          "use 81, 121, ...; skipping", this->id(), _frames));
      co_return;
    }
    auto* mc = session()->services()->metal_compute();
    const auto img = tbp->materialize_contiguous();
    const bool u8 = tbp->dtype == TensorBeat::DType::U8;
    const float pad[3] = {
      (float)_pad_r / 255.0f * 2.0f - 1.0f,
      (float)_pad_g / 255.0f * 2.0f - 1.0f,
      (float)_pad_b / 255.0f * 2.0f - 1.0f,
    };
    const std::vector<float> norm =
        normalize_and_fit_(img.data(), u8, sH, sW, H, W, pad);
    // [3, F, H, W] channel-first: frame 0 is the image, the rest are the
    // ZERO of the [-1,1] range (mid grey), which is what the reference's
    // new_zeros produces on an already-normalized tensor.
    const std::size_t plane = (std::size_t)H * W;
    metal_compute::SharedBuffer vid =
        mc->make_shared_buffer((std::size_t)3 * _frames * plane * 2);
    if (vid.empty()) {
      session()->warn(fmt("VaeEncodeStage('{}'): conditioning-clip alloc "
                          "failed; skipping", this->id()));
      co_return;
    }
    {
      auto* d = static_cast<_Float16*>(vid.contents());
      std::memset(d, 0, (std::size_t)3 * _frames * plane * 2);
      for (int c = 0; c < 3; ++c) {
        _Float16* dst = d + (std::size_t)c * _frames * plane;
        const float* src = norm.data() + (std::size_t)c * plane;
        for (std::size_t i = 0; i < plane; ++i) { dst[i] = (_Float16)src[i]; }
      }
    }
    std::string eerr;
    metal_compute::SharedBuffer lat;
    {
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae,
                         kPerfLlmVaeBegin,
                         (std::uint64_t)_frames * H * W);
      lat = _wan_vae->encode(vid, _frames, H, W, &eerr);
    }
    if (lat.empty()) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): video encode failed ({}); skipping",
          this->id(), eerr.empty() ? "unknown error" : eerr));
      co_return;
    }
    const int Cz = _wan_vae->config().z_dim;
    const int T = genai::MetalWanVae::latent_frames(_frames);
    const std::size_t nz = (std::size_t)Cz * T * (H / 8) * (W / 8);
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {Cz, T, H / 8, W / 8};
    out->resize_contiguous(nz);
    const auto* lp = static_cast<const _Float16*>(lat.contents());
    float* op = out->as_f32();
    for (std::size_t i = 0; i < nz; ++i) { op[i] = (float)lp[i]; }
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("frames",
                                    FlexData::make_int((std::int64_t)_frames));
    out->sideband = std::move(sb);
    ++_latents_emitted;
    session()->log_debug(fmt(
        "VaeEncodeStage('{}'): Wan conditioning latent #{} [{}, {}, {}, {}] "
        "from a {}-frame clip", this->id(), _latents_emitted, Cz, T, H / 8,
        W / 8, _frames));
    if (_unload_idle) { unload_vae_(); }
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  if (!_vae) {
    session()->warn(fmt(
        "VaeEncodeStage('{}'): VAE encoder not loaded; skipping", this->id()));
    co_return;
  }
  const int sH = (int)tbp->shape[1];
  const int sW = (int)tbp->shape[2];
  if (sH <= 0 || sW <= 0) {
    session()->warn(fmt(
        "VaeEncodeStage('{}'): image [{}x{}] has invalid dimensions; skipping",
        this->id(), sW, sH));
    co_return;
  }
  // Target size: the letterbox target when configured, else the source size.
  // When encoding at native size the source must already be a multiple of 8
  // (the VAE downsamples by 8); the letterbox target is validated at config.
  const bool resize = _target_w > 0 && _target_h > 0;
  const int H = resize ? _target_h : sH;
  const int W = resize ? _target_w : sW;
  if (!resize && ((H % 8) != 0 || (W % 8) != 0)) {
    session()->warn(fmt(
        "VaeEncodeStage('{}'): image [{}x{}] must be a positive multiple of 8 "
        "(or set target_width/target_height to letterbox-resize); skipping",
        this->id(), sW, sH));
    co_return;
  }
  auto* mc = session()->services()->metal_compute();
  session()->log_debug(fmt(
      "VaeEncodeStage('{}'): beat received, image [3, {}, {}] -> encode [3, {}, "
      "{}]{} -> latent [{}, {}, {}]", this->id(), sH, sW, H, W,
      resize ? " (letterbox)" : "", _vae->config().z_dim, H / 8, W / 8));

  // Materialise the source (handles a padded/strided beat), then normalize to
  // [-1,1] and letterbox-fit into the target [3,H,W] (pad in [-1,1]). U8 maps
  // 0..255 -> [-1,1]; f32 is assumed already in [-1,1].
  const auto img = tbp->materialize_contiguous();
  const bool is_u8 = tbp->dtype == TensorBeat::DType::U8;
  const float pad[3] = {
    (float)_pad_r / 255.0f * 2.0f - 1.0f,
    (float)_pad_g / 255.0f * 2.0f - 1.0f,
    (float)_pad_b / 255.0f * 2.0f - 1.0f,
  };
  const std::vector<float> norm =
      normalize_and_fit_(img.data(), is_u8, sH, sW, H, W, pad);

  const std::size_t n = (std::size_t)3 * H * W;
  metal_compute::SharedBuffer imgbuf = mc->make_shared_buffer(n * 2);
  if (imgbuf.empty()) {
    session()->warn(fmt(
        "VaeEncodeStage('{}'): image upload alloc failed; skipping",
        this->id()));
    co_return;
  }
  {
    auto* d = static_cast<_Float16*>(imgbuf.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)norm[i]; }
  }

  metal_compute::SharedBuffer lat;
  {
    PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae, kPerfLlmVaeBegin,
                       (std::uint64_t)H * W);
    lat = _vae->encode(imgbuf, H, W);
  }
  if (lat.empty()) {
    session()->warn(fmt(
        "VaeEncodeStage('{}'): encode failed; skipping", this->id()));
    co_return;
  }

  // f16 whitened latent [z_dim, H/8, W/8] -> f32 TensorBeat.
  const int Cz = _vae->config().z_dim, lh = H / 8, lw = W / 8;
  const std::size_t nz = (std::size_t)Cz * lh * lw;
  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  out->shape = {Cz, lh, lw};
  out->resize_contiguous(nz);
  const auto* lp = static_cast<const _Float16*>(lat.contents());
  float* op = out->as_f32();
  for (std::size_t i = 0; i < nz; ++i) { op[i] = (float)lp[i]; }
  ++_latents_emitted;
  session()->log_debug(fmt(
      "VaeEncodeStage('{}'): encoded + emitted latent #{} [16, {}, {}]",
      this->id(), _latents_emitted, lh, lw));
  if (_unload_idle) { unload_vae_(); }
  co_await ctx.write(0, std::move(out));
}

#else   // !VPIPE_BUILD_APPLE_SILICON

Job
VaeEncodeStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (session()) {
    session()->error(fmt(
        "VaeEncodeStage('{}'): built without VPIPE_BUILD_APPLE_SILICON; the "
        "metal VAE is unavailable, the stage is inert", this->id()));
  }
  co_return;
}

Job
VaeEncodeStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  (void)in;
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(VaeEncodeStage)
VPIPE_REGISTER_SPEC(VaeEncodeStage, kSpec)

}
