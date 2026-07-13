#include "stages/vae-decode-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

VaeDecodeStage::VaeDecodeStage(const SessionContextIntf* s,
                               std::string               id,
                               std::vector<InEdge>       iports,
                               FlexData                  config)
  : TypedStage<VaeDecodeStage>(s, std::move(id), std::move(iports),
                               std::move(config))
{
  // Deferred validation (see Stage::fail_config): construct for any config so
  // a graph can be built/edited before hf_dir is supplied.
  _hf_dir    = attr_str("hf_dir");
  _models_db = attr_str("models_db");
  if (_models_db.empty()) { _models_db = "models"; }
  if (_hf_dir.empty()) {
    fail_config(fmt(
        "VaeDecodeStage('{}'): config.hf_dir is required (the Krea-2-Turbo "
        "model dir; the VAE is read from <hf_dir>/vae)", this->id()));
  }
  allocate_oports(spec().oports.size());
}

VaeDecodeStage::~VaeDecodeStage() = default;

namespace {
const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "Krea-2-Turbo / FLUX.2 model dir (VAE read from <hf_dir>/vae)",
   .suggest_db = "models", .suggest_db_type = "krea2,flux2"},
  {.key = "models_db", .type = ConfigType::String, .required = false,
   .doc = "model registry db for resolve_model_dir (default \"models\")"},
};
const PortSpec kIports[] = {
  {.name = "latent", .doc = "f32 latent [z_dim, H/8, W/8] (unpacked, whitened)",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "image", .doc = "decoded image as planar U8 RGB TensorBeat [3,H,W]",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "vae-decode",
  .doc       = "Decodes a text-to-image (Krea-2 Qwen-Image / FLUX.2) VAE latent "
               "into a planar U8 RGB image on the metal-compute backend. The "
               "second half of the text-to-image split.",
  .display_name = "VAE Decode",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

#ifdef VPIPE_BUILD_APPLE_SILICON
// Detect the VAE family from the vae config.json `_class_name`:
// "AutoencoderKLFlux2" -> "flux2"; anything else (AutoencoderKLQwenImage) ->
// "krea2".
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
VaeDecodeStage::spec() const noexcept
{
  return kSpec;
}

#ifdef VPIPE_BUILD_APPLE_SILICON

Job
VaeDecodeStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (_hf_dir.empty()) { co_return; }   // ctor already recorded the error.
  auto* mc = session() ? session()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "VaeDecodeStage('{}'): no metal-compute backend on this session; "
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
    // FLUX.2 AutoencoderKLFlux2: read the geometry from config.json; the latent
    // whitening (patch + BatchNorm) is internal, so no latents_mean/std here.
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
        "VaeDecodeStage('{}'): loading FLUX.2 VAE from '{}'", this->id(),
        vae_dir));
    _flux2_vae = genai::MetalFlux2Vae::load(vae_dir, mc, fcfg);
    if (!_flux2_vae) {
      session()->error(fmt(
          "VaeDecodeStage('{}'): failed to load the FLUX.2 VAE from '{}'; inert",
          this->id(), vae_dir));
    }
    co_return;
  }

  genai::MetalKrea2Vae::Config cfg;   // Qwen-Image VAE defaults
  // Read the per-channel latent statistics (and z_dim / base_dim) from the
  // vae config.json; the un-whiten needs latents_mean / latents_std.
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
        "VaeDecodeStage('{}'): vae config.json is missing latents_mean/"
        "latents_std ({}/{} of z_dim {}); the stage is inert", this->id(),
        cfg.latents_mean.size(), cfg.latents_std.size(), cfg.z_dim));
    co_return;
  }

  session()->info(fmt(
      "VaeDecodeStage('{}'): loading Qwen-Image VAE from '{}'", this->id(),
      vae_dir));
  _vae = genai::MetalKrea2Vae::load(vae_dir, mc, cfg);
  if (!_vae) {
    session()->error(fmt(
        "VaeDecodeStage('{}'): failed to load the VAE from '{}'; inert",
        this->id(), vae_dir));
    co_return;
  }
  session()->log_debug(fmt(
      "VaeDecodeStage('{}'): VAE ready from '{}' (z_dim {}, base_dim {}, "
      "num_res_blocks {})", this->id(), vae_dir, cfg.z_dim, cfg.base_dim,
      cfg.num_res_blocks));
}

Job
VaeDecodeStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }   // upstream EOS -> close oport
  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (tbp == nullptr || tbp->dtype != TensorBeat::DType::F32 ||
      tbp->shape.size() != 3) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): expected an f32 [z,H/8,W/8] latent TensorBeat, "
        "got {}; skipping", this->id(), in->describe()));
    co_return;
  }
  auto* mc = session()->metal_compute();

  // ---- FLUX.2: input [dit_channels, H/16, W/16]; decode directly (the patch +
  // BatchNorm whitening is internal), no per-channel un-whiten. ----
  if (_family == "flux2") {
    if (!_flux2_vae) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): FLUX.2 VAE not loaded; skipping", this->id()));
      co_return;
    }
    const int Cdit = (int)tbp->shape[0];
    const int h16 = (int)tbp->shape[1];
    const int w16 = (int)tbp->shape[2];
    if (Cdit != _flux2_vae->config().dit_channels() || h16 <= 0 || w16 <= 0) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): latent [{}, {}, {}] does not match "
          "dit_channels {}; skipping", this->id(), Cdit, h16, w16,
          _flux2_vae->config().dit_channels()));
      co_return;
    }
    const std::size_t nz = (std::size_t)Cdit * h16 * w16;
    metal_compute::SharedBuffer z = mc->make_shared_buffer(nz * 2);
    { auto* d = static_cast<_Float16*>(z.contents());
      const float* s = tbp->as_f32();
      for (std::size_t i = 0; i < nz; ++i) { d[i] = (_Float16)s[i]; } }
    std::string derr;
    metal_compute::SharedBuffer rgb = _flux2_vae->decode(z, h16, w16, &derr);
    if (rgb.empty()) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): FLUX.2 decode failed ({}); skipping",
          this->id(), derr.empty() ? "unknown error" : derr));
      co_return;
    }
    const int H = h16 * 16, W = w16 * 16;
    const std::size_t n = (std::size_t)3 * H * W;
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::U8;
    out->shape = {3, H, W};
    out->resize_contiguous(n);
    const auto* rp = static_cast<const _Float16*>(rgb.contents());
    std::uint8_t* op = out->as_u8();
    for (std::size_t i = 0; i < n; ++i) {
      float v = ((float)rp[i] + 1.0f) * 0.5f * 255.0f;
      v = std::round(v);
      if (v < 0.0f) { v = 0.0f; }
      if (v > 255.0f) { v = 255.0f; }
      op[i] = (std::uint8_t)v;
    }
    ++_images_emitted;
    session()->log_debug(fmt(
        "VaeDecodeStage('{}'): FLUX.2 decoded + emitted image #{} [3, {}, {}]",
        this->id(), _images_emitted, H, W));
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  if (!_vae) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): VAE not loaded; skipping", this->id()));
    co_return;
  }
  const int Cz = (int)tbp->shape[0];
  const int h8 = (int)tbp->shape[1];
  const int w8 = (int)tbp->shape[2];
  session()->log_debug(fmt(
      "VaeDecodeStage('{}'): beat received, latent [{}, {}, {}] -> image "
      "[{}, {}]", this->id(), Cz, h8, w8, h8 * 8, w8 * 8));
  if (Cz != _vae->config().z_dim || h8 <= 0 || w8 <= 0) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): latent [{}, {}, {}] does not match z_dim {}; "
        "skipping", this->id(), Cz, h8, w8, _vae->config().z_dim));
    co_return;
  }

  // f32 latent -> f16 channel-first buffer -> un-whiten -> decode.
  const std::size_t nz = (std::size_t)Cz * h8 * w8;
  metal_compute::SharedBuffer z = mc->make_shared_buffer(nz * 2);
  { auto* d = static_cast<_Float16*>(z.contents());
    const float* s = tbp->as_f32();
    for (std::size_t i = 0; i < nz; ++i) { d[i] = (_Float16)s[i]; } }

  metal_compute::SharedBuffer zw = _vae->unwhiten(z, h8, w8);
  if (zw.empty()) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): un-whiten failed; skipping", this->id()));
    co_return;
  }
  std::string derr;
  metal_compute::SharedBuffer rgb = _vae->decode(zw, h8, w8, &derr);
  if (rgb.empty()) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): decode failed ({}); skipping", this->id(),
        derr.empty() ? "unknown error" : derr));
    co_return;
  }

  // f16 [3,H,W] in [-1,1] -> planar U8 RGB (x+1)/2*255, rounded + clamped.
  const int H = h8 * 8, W = w8 * 8;
  const std::size_t n = (std::size_t)3 * H * W;
  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::U8;
  out->shape = {3, H, W};
  out->resize_contiguous(n);
  const auto* rp = static_cast<const _Float16*>(rgb.contents());
  std::uint8_t* op = out->as_u8();
  for (std::size_t i = 0; i < n; ++i) {
    float v = ((float)rp[i] + 1.0f) * 0.5f * 255.0f;
    v = std::round(v);
    if (v < 0.0f) { v = 0.0f; }
    if (v > 255.0f) { v = 255.0f; }
    op[i] = (std::uint8_t)v;
  }
  ++_images_emitted;
  session()->log_debug(fmt(
      "VaeDecodeStage('{}'): decoded + emitted image #{} planar U8 RGB "
      "[3, {}, {}]", this->id(), _images_emitted, H, W));
  co_await ctx.write(0, std::move(out));
}

#else   // !VPIPE_BUILD_APPLE_SILICON

Job
VaeDecodeStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (session()) {
    session()->error(fmt(
        "VaeDecodeStage('{}'): built without VPIPE_BUILD_APPLE_SILICON; the "
        "metal VAE is unavailable, the stage is inert", this->id()));
  }
  co_return;
}

Job
VaeDecodeStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  (void)in;
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(VaeDecodeStage)
VPIPE_REGISTER_SPEC(VaeDecodeStage, kSpec)

}
