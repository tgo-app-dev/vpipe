#include "stages/vae-encode-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
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
  // a graph can be built/edited before hf_dir is supplied.
  _hf_dir    = attr_str("hf_dir");
  _models_db = attr_str("models_db");
  if (_models_db.empty()) { _models_db = "models"; }
  if (_hf_dir.empty()) {
    fail_config(fmt(
        "VaeEncodeStage('{}'): config.hf_dir is required (the Krea-2-Turbo "
        "model dir; the VAE is read from <hf_dir>/vae)", this->id()));
  }

  // Optional letterbox resize: target_width + target_height (both required
  // together, both multiples of 8 -- the VAE downsamples by 8, and the
  // downstream text-to-image latent grid needs an even latent H/W).
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
  allocate_oports(spec().oports.size());
}

VaeEncodeStage::~VaeEncodeStage() = default;

namespace {

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
const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "Krea-2-Turbo / FLUX.2 model dir (VAE read from <hf_dir>/vae)",
   .suggest_db = "models", .suggest_db_type = "krea2,flux2"},
  {.key = "models_db", .type = ConfigType::String, .required = false,
   .doc = "model registry db for resolve_model_dir (default \"models\")"},
  {.key = "target_width", .type = ConfigType::Int, .required = false,
   .doc = "letterbox-resize the input to this width before encoding (multiple "
          "of 8; requires target_height). 0/unset = encode at native size"},
  {.key = "target_height", .type = ConfigType::Int, .required = false,
   .doc = "letterbox-resize target height (multiple of 8; requires "
          "target_width)"},
  {.key = "pad_color", .type = ConfigType::Any, .required = false,
   .doc = "letterbox pad color: [r,g,b] 0..255, a name "
          "(black/white/gray), or a single gray level. Default black"},
};
const PortSpec kIports[] = {
  {.name = "image", .doc = "U8 or f32 RGB image [3,H,W] (channel-first, U8 "
                           "0..255 or f32 [-1,1])",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
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
               "vae-decode; feeds the text-to-image `latent` port (img2img).",
  .display_name = "VAE Encode",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

#ifdef VPIPE_BUILD_APPLE_SILICON
// VAE family from the vae config.json `_class_name` ("AutoencoderKLFlux2" ->
// "flux2"; else "krea2").
std::string
vae_family_(const std::string& vae_dir)
{
  namespace fs = std::filesystem;
  std::ifstream in(fs::path(vae_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto obj = fd.as_object();
      if (obj.contains("_class_name") &&
          std::string(obj.at("_class_name").as_string("")) ==
              "AutoencoderKLFlux2") {
        return "flux2";
      }
    }
  }
  return "krea2";
}
#endif
}  // namespace

const StageSpec&
VaeEncodeStage::spec() const noexcept
{
  return kSpec;
}

#ifdef VPIPE_BUILD_APPLE_SILICON

Job
VaeEncodeStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (_hf_dir.empty()) { co_return; }   // ctor already recorded the error.
  auto* mc = session() ? session()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "VaeEncodeStage('{}'): no metal-compute backend on this session; "
        "the stage is inert", this->id()));
    co_return;
  }
  const std::string root = resolve_model_dir(session(), _models_db, _hf_dir);
  namespace fs = std::filesystem;
  std::string vae_dir = root;
  if (fs::exists(fs::path(root) / "vae" / "config.json")) {
    vae_dir = (fs::path(root) / "vae").string();
  }

  _family = vae_family_(vae_dir);
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
    session()->info(fmt(
        "VaeEncodeStage('{}'): loading FLUX.2 VAE encoder from '{}'", this->id(),
        vae_dir));
    _flux2_vae = genai::MetalFlux2Vae::load(vae_dir, mc, fcfg,
                                            /*with_encoder=*/true);
    if (!_flux2_vae || !_flux2_vae->has_encoder()) {
      session()->error(fmt(
          "VaeEncodeStage('{}'): failed to load the FLUX.2 VAE encoder from "
          "'{}'; inert", this->id(), vae_dir));
      _flux2_vae.reset();
    }
    co_return;
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
    co_return;
  }

  session()->info(fmt(
      "VaeEncodeStage('{}'): loading Qwen-Image VAE encoder from '{}'",
      this->id(), vae_dir));
  _vae = genai::MetalKrea2Vae::load(vae_dir, mc, cfg, /*with_encoder=*/true);
  if (!_vae || !_vae->has_encoder()) {
    session()->error(fmt(
        "VaeEncodeStage('{}'): failed to load the VAE encoder from '{}'; "
        "inert", this->id(), vae_dir));
    _vae.reset();
    co_return;
  }
  session()->log_debug(fmt(
      "VaeEncodeStage('{}'): VAE encoder ready from '{}' (z_dim {}, base_dim "
      "{}, num_res_blocks {})", this->id(), vae_dir, cfg.z_dim, cfg.base_dim,
      cfg.num_res_blocks));
}

namespace {

// Produce a normalized f32 [-1,1] planar RGB [3,outH,outW] from the source
// image bytes (channel-first [3,sH,sW], U8 0..255 or f32 already in [-1,1]).
// When the target differs from the source the picture is scaled to fit with
// its original aspect ratio (bilinear), centered, and the leftover border is
// filled with `pad` (per-channel, already mapped to [-1,1]). Same size =>
// straight normalize, no resample.
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
  for (int c = 0; c < 3; ++c) {
    for (int oy = 0; oy < newH; ++oy) {
      const double sy = ((oy + 0.5) * sH / newH) - 0.5;
      const int y0 = (int)std::floor(sy);
      const float fy = (float)(sy - y0);
      const int y0c = std::min(std::max(y0, 0), sH - 1);
      const int y1c = std::min(std::max(y0 + 1, 0), sH - 1);
      for (int ox = 0; ox < newW; ++ox) {
        const double sx = ((ox + 0.5) * sW / newW) - 0.5;
        const int x0 = (int)std::floor(sx);
        const float fx = (float)(sx - x0);
        const int x0c = std::min(std::max(x0, 0), sW - 1);
        const int x1c = std::min(std::max(x0 + 1, 0), sW - 1);
        const float v00 = src_val(c, y0c, x0c);
        const float v01 = src_val(c, y0c, x1c);
        const float v10 = src_val(c, y1c, x0c);
        const float v11 = src_val(c, y1c, x1c);
        const float top = v00 + fx * (v01 - v00);
        const float bot = v10 + fx * (v11 - v10);
        out[((std::size_t)c * outH + (offY + oy)) * outW + (offX + ox)] =
            top + fy * (bot - top);
      }
    }
  }
  return out;
}

}  // namespace

Job
VaeEncodeStage::process(RuntimeContext& ctx)
{
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

  // ---- FLUX.2: encode to [dit_channels, H/16, W/16] (8x VAE + 2x patch), so
  // the native image must be a multiple of 16. ----
  if (_family == "flux2") {
    if (!_flux2_vae) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): FLUX.2 VAE encoder not loaded; skipping",
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
          "VaeEncodeStage('{}'): FLUX.2 image [{}x{}] must be a positive "
          "multiple of 16 (or set target_width/height); skipping", this->id(),
          sW, sH));
      co_return;
    }
    auto* mc = session()->metal_compute();
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
    metal_compute::SharedBuffer lat = _flux2_vae->encode(imgbuf, H, W);
    if (lat.empty()) {
      session()->warn(fmt(
          "VaeEncodeStage('{}'): FLUX.2 encode failed; skipping", this->id()));
      co_return;
    }
    const int Cdit = _flux2_vae->config().dit_channels();
    const int lh = H / 16, lw = W / 16;
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
        "VaeEncodeStage('{}'): FLUX.2 encoded latent #{} [{}, {}, {}]",
        this->id(), _latents_emitted, Cdit, lh, lw));
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
  auto* mc = session()->metal_compute();
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

  metal_compute::SharedBuffer lat = _vae->encode(imgbuf, H, W);
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
