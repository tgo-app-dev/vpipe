// MiniMax-Music3's vocoder as a VaeModelFamily, so `audio-vae-decode`
// decodes generate-audio's latents through its registry path -- no
// Music3 branch in the stage.
//
// The vocoder lives in the model's own repository (`vocoder/`, a DAC
// decoder, MiniMaxMusic3Vocoder) beside the four parts generate-audio
// holds, so `root` here is the whole repository and every question below
// names the one directory this decoder owns.

#include "generative-models/vae-model-registry.h"

#include "common/vpipe-format.h"
#include "generative-models/minimax-music3/music3-protocol.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/weight-set.h"
#include "generative-models/yue2/metal-oobleck-decoder.h"
#include "interfaces/session-context-intf.h"

#include <filesystem>
#include <memory>
#include <string>

namespace vpipe::genai {

namespace {

// The vocoder: `root/vocoder`, or `root` itself when that is the vocoder.
std::string
music3_vocoder_dir_(const std::string& root)
{
  const std::string sub = (std::filesystem::path(root) / "vocoder").string();
  if (MetalOobleckDecoder::is_music3_vocoder_dir(sub)) { return sub; }
  if (MetalOobleckDecoder::is_music3_vocoder_dir(root)) { return root; }
  return {};
}

class Music3AudioDecoder final : public AudioVaeDecoder {
public:
  Music3AudioDecoder(std::unique_ptr<MetalOobleckDecoder> d,
                     const SessionContextIntf* session)
      : _d(std::move(d)), _session(session) {}

  int sample_rate() const override { return _d->config().sample_rate; }

  bool decode(const AudioVaeDecodeRequest& req, std::vector<float>* pcm,
              std::vector<int>* shape, std::string* err) override
  {
    if (!_i8_said && accel::flag(req.accel, accel::kI8Gemm)) {
      _i8_said = true;
      if (_session != nullptr) {
        _session->info(fmt("minimax-music3 vocoder: i8_gemm has no effect "
                           "here -- the decoder has no int8 route"));
      }
    }
    // generate-audio sends [1, 128, latents] -- the song's windows already
    // stitched (see music3-protocol.h); [128, latents] is the same thing
    // without the batch axis.
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
        *err = "the MiniMax-Music3 vocoder wants a [1, " +
               std::to_string(Z) + ", latents] latent";
      }
      return false;
    }
    if (req.latents_per_second > 0.0 &&
        std::abs(req.latents_per_second - music3::kLatentsPerSecond) >
            1e-3) {
      if (err != nullptr) {
        *err = fmt("a latent at {} a second is not MiniMax-Music3's ({})",
                   req.latents_per_second, music3::kLatentsPerSecond)();
      }
      return false;
    }
    if (!_d->decode(req.latent, frames, pcm, /*clamp=*/true, req.progress,
                    err)) {
      return false;
    }
    const int C = _d->config().out_channels * _d->config().streams;
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

class Music3VaeFamily final : public VaeModelFamily {
public:
  std::string_view tag() const noexcept override { return "minimax-music3"; }

  bool claims(const std::string& root, const std::string& /*vae_dir*/,
              const std::string& /*model_type*/) const override
  {
    return !music3_vocoder_dir_(root).empty();
  }

  std::string vae_path(const std::string& root,
                       std::string_view role) const override
  {
    return role == kRoleAudio ? music3_vocoder_dir_(root) : std::string();
  }

  std::vector<std::string> idle_peers(const std::string& root)
      const override
  {
    // What generate-audio holds beside the decode: the language model,
    // the depth decoder, the condition encoder and the DiT.
    std::vector<std::string> v;
    for (const char* sub : {"language_model", "rvq_depth_decoder",
                            "condition_encoder", "transformer"}) {
      const auto p = std::filesystem::path(root) / sub;
      if (std::filesystem::is_directory(p)) { v.push_back(p.string()); }
    }
    return v;
  }

  std::unique_ptr<VaeDecoder>
  load_decoder(const VaeModelCreateArgs& /*args*/) override
  {
    return nullptr;                 // an audio VAE: no pictures
  }

  std::unique_ptr<AudioVaeDecoder>
  load_audio_decoder(const VaeModelCreateArgs& args) override
  {
    const std::string dir = music3_vocoder_dir_(args.root);
    MetalOobleckDecoder::Config cfg;
    std::string err;
    if (dir.empty() || !MetalOobleckDecoder::config_from_dir(dir, &cfg,
                                                             &err)) {
      if (args.session != nullptr) {
        args.session->warn(fmt("minimax-music3 vocoder: {}",
                               err.empty() ? "no vocoder/ here" : err));
      }
      return nullptr;
    }
    auto d = MetalOobleckDecoder::load(open_weight_set(dir, args.session),
                                       args.metal, cfg, &err);
    if (!d) {
      if (args.session != nullptr) {
        args.session->warn(fmt("minimax-music3 vocoder: could not load "
                               "'{}': {}", dir, err));
      }
      return nullptr;
    }
    return std::make_unique<Music3AudioDecoder>(std::move(d),
                                               args.session);
  }
};

[[maybe_unused]] const bool kRegistered =
    VaeModelRegistry::get().add(std::make_unique<Music3VaeFamily>());

}  // namespace

}  // namespace vpipe::genai
