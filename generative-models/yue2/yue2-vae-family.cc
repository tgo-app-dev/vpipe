// The YuE2 Oobleck VAE as a VaeModelFamily, so `audio-vae-decode` decodes
// generate-audio's latents through its registry path -- no YuE2 branch in
// the stage. Registered at load like the resource planners.
//
// The VAE is its own checkpoint (m-a-p/YuE2-Vae or YuE2-Vae-legacy), so
// the family's `root` IS the VAE: there is no DiT beside it to keep out
// of the claim.

#include "generative-models/vae-model-registry.h"

#include "common/vpipe-format.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/weight-set.h"
#include "generative-models/yue2/metal-oobleck-decoder.h"
#include "interfaces/session-context-intf.h"

#include <filesystem>
#include <memory>
#include <string>

namespace vpipe::genai {

namespace {

// A YuE2 VAE export: `root` itself, or a `vae/` beside an LM checkpoint.
std::string
yue2_vae_dir_(const std::string& root)
{
  if (MetalOobleckDecoder::is_oobleck_dir(root)) { return root; }
  const std::string sub = (std::filesystem::path(root) / "vae").string();
  if (MetalOobleckDecoder::is_oobleck_dir(sub)) { return sub; }
  return {};
}

class Yue2AudioDecoder final : public AudioVaeDecoder {
public:
  Yue2AudioDecoder(std::unique_ptr<MetalOobleckDecoder> d,
                   const SessionContextIntf* session)
      : _d(std::move(d)), _session(session) {}

  int sample_rate() const override { return _d->config().sample_rate; }

  bool decode(const AudioVaeDecodeRequest& req, std::vector<float>* pcm,
              std::vector<int>* shape, std::string* err) override
  {
    // SAY SO WHEN A KEY CANNOT BE HONOURED. audio-vae-decode offers
    // `i8_gemm` and hands it to whichever family claimed the checkpoint,
    // and the Oobleck decoder has no int8 route at all -- so taking the
    // flag and doing nothing would read exactly like taking it and
    // working. Once per decoder, at info: it is a knob that did nothing,
    // not an error.
    if (!_i8_said && accel::flag(req.accel, accel::kI8Gemm)) {
      _i8_said = true;
      if (_session != nullptr) {
        _session->info(fmt("yue2 VAE: i8_gemm has no effect here -- the "
                           "Oobleck decoder has no int8 route; its GEMMs "
                           "run on the matrix cores where there are any, "
                           "and the steel tiles otherwise"));
      }
    }
    // generate-audio sends [1, latent_dim, frames]; a bare
    // [latent_dim, frames] is the same thing without the batch axis.
    const int Z = _d->config().latent_dim;
    const std::vector<int>& s = req.shape;
    int frames = 0;
    if (s.size() == 3 && s[0] == 1 && s[1] == Z) {
      frames = s[2];
    } else if (s.size() == 2 && s[0] == Z) {
      frames = s[1];
    }
    if (req.latent == nullptr || frames <= 0) {
      if (err != nullptr) {
        *err = "YuE2 VAE wants a [1, " + std::to_string(Z) +
               ", frames] latent";
      }
      return false;
    }
    if (!_d->decode(req.latent, frames, pcm, /*clamp=*/true, req.progress,
                    err)) {
      return false;
    }
    const int C = _d->config().out_channels;
    *shape = {C, (int)(pcm->size() / (std::size_t)C)};
    return true;
  }

  std::uint64_t resident_bytes() const override
  {
    return _d->resident_bytes();
  }

private:
  std::unique_ptr<MetalOobleckDecoder> _d;
  const SessionContextIntf*            _session = nullptr;
  bool                                 _i8_said = false;
};

class Yue2VaeFamily final : public VaeModelFamily {
public:
  std::string_view tag() const noexcept override { return "yue2"; }

  bool claims(const std::string& root, const std::string& /*vae_dir*/,
              const std::string& /*model_type*/) const override
  {
    return !yue2_vae_dir_(root).empty();
  }

  std::string vae_path(const std::string& root,
                       std::string_view role) const override
  {
    return role == kRoleAudio ? yue2_vae_dir_(root) : std::string();
  }

  std::vector<std::string> idle_peers(const std::string& /*root*/)
      const override
  {
    // The LM is a different checkpoint, held by generate-audio, so
    // nothing under this root sits beside the decode.
    return {};
  }

  std::unique_ptr<VaeDecoder>
  load_decoder(const VaeModelCreateArgs& /*args*/) override
  {
    return nullptr;                 // an audio VAE: no pictures
  }

  std::unique_ptr<AudioVaeDecoder>
  load_audio_decoder(const VaeModelCreateArgs& args) override
  {
    const std::string dir = yue2_vae_dir_(args.root);
    MetalOobleckDecoder::Config cfg;
    std::string err;
    if (dir.empty() || !MetalOobleckDecoder::config_from_dir(dir, &cfg,
                                                             &err)) {
      if (args.session != nullptr) {
        args.session->warn(fmt("yue2 VAE: {}", err));
      }
      return nullptr;
    }
    auto d = MetalOobleckDecoder::load(open_weight_set(dir, args.session),
                                       args.metal, cfg, &err);
    if (!d) {
      if (args.session != nullptr) {
        args.session->warn(fmt("yue2 VAE: could not load '{}': {}", dir,
                               err));
      }
      return nullptr;
    }
    return std::make_unique<Yue2AudioDecoder>(std::move(d),
                                             args.session);
  }
};

[[maybe_unused]] const bool kRegistered =
    VaeModelRegistry::get().add(std::make_unique<Yue2VaeFamily>());

}  // namespace

}  // namespace vpipe::genai
