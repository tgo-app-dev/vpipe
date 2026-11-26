// MAGI-2-preview's decoders as a registered VAE family: vae-decode runs
// the turbo video decoder (turbo_vae/), audio-vae-decode Stable Audio
// Open's autoencoder (stable-audio-open-1.0/) with the release's 441/512
// stretch to 44.1 kHz. Both under the MAGI-2 root, which is what the
// graph names.

#include "common/flex-bag.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/magi2/magi2-audio.h"
#include "generative-models/magi2/magi2-image-encoder.h"
#include "generative-models/magi2/magi2-layout.h"
#include "generative-models/magi2/metal-magi2-turbo.h"
#include "generative-models/vae-model-registry.h"
#include "generative-models/vae-host-ext.h"
#include "generative-models/weight-set.h"
#include "generative-models/yue2/metal-oobleck-decoder.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-memory-host.h"
#include "stages/model-memory.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::genai {
namespace {

std::string
turbo_ckpt_(const std::string& dir)
{
  return (std::filesystem::path(dir) / "checkpoint.ckpt").string();
}

class Magi2VideoDecoder final : public VaeDecoder, public VaeWorkingBytes {
 public:
  explicit Magi2VideoDecoder(std::unique_ptr<MetalMagi2Turbo> d)
      : _d(std::move(d)) {}

  int latent_channels() const override { return 48; }
  int spatial_compression() const override { return 16; }
  int temporal_compression() const override { return 4; }

  bool decode(const VaeDecodeRequest& req, const VaeFrameSink& sink,
              std::string* err) override
  {
    const std::vector<int>& s = req.shape;
    if (req.latent == nullptr || s.size() != 4 || s[0] != 48) {
      if (err != nullptr) {
        *err = "the MAGI-2 turbo decoder wants a [48, T, h, w] latent";
      }
      return false;
    }
    const int T = s[1], h = s[2], w = s[3];
    std::vector<float> rgb;
    if (!_d->decode(req.latent, T, h, w, &rgb, err)) {
      _d->release_scratch();
      return false;
    }
    _d->release_scratch();
    VaeFrameChunk c;
    c.rgb = rgb.data();
    c.channels = 3;
    c.frame0 = 0;
    c.n = MetalMagi2Turbo::frames_of(T);
    c.height = 16 * h;
    c.width = 16 * w;
    c.frames_total = c.n;
    return sink(c);
  }

  std::uint64_t resident_bytes() const override { return _d->weight_bytes(); }

  std::size_t working_bytes(const std::vector<int>& shape) const override
  {
    if (shape.size() != 4) { return 0; }
    return _d->scratch_bytes(shape[1], shape[2], shape[3]);
  }

  void* query_extension(std::string_view id) noexcept override
  {
    if (id == kVaeWorkingBytesExt) {
      return static_cast<VaeWorkingBytes*>(this);
    }
    return nullptr;
  }

 private:
  std::unique_ptr<MetalMagi2Turbo> _d;
};

// Image-to-video's picture: the Wan2.2 VAE's encoder on one frame.
class Magi2PictureEncoder final : public VaeEncoder {
 public:
  explicit Magi2PictureEncoder(std::unique_ptr<Magi2ImageEncoder> e)
      : _e(std::move(e)) {}

  int latent_channels() const override { return 48; }
  int spatial_compression() const override { return 16; }
  int temporal_compression() const override { return 4; }

  bool encode(const VaeEncodeRequest& req, std::vector<float>* out,
              std::vector<int>* shape, std::string* err) override
  {
    if (req.pixels == nullptr || req.frames != 1) {
      if (err != nullptr) {
        *err = "MAGI-2 encodes one picture (image-to-video's first frame)";
      }
      return false;
    }
    int h = 0, w = 0;
    if (!_e->encode(req.pixels, req.height, req.width, out, &h, &w, err)) {
      return false;
    }
    *shape = {48, 1, h, w};
    return true;
  }

  std::uint64_t resident_bytes() const override { return _e->weight_bytes(); }

 private:
  std::unique_ptr<Magi2ImageEncoder> _e;
};

class Magi2AudioDecoder final : public AudioVaeDecoder {
 public:
  explicit Magi2AudioDecoder(std::unique_ptr<MetalOobleckDecoder> d)
      : _d(std::move(d)) {}

  int sample_rate() const override { return magi2::kAudioSampleRate; }

  bool decode(const AudioVaeDecodeRequest& req, std::vector<float>* pcm,
              std::vector<int>* shape, std::string* err) override
  {
    const int Z = _d->config().latent_dim;
    const std::vector<int>& s = req.shape;
    int L = 0;
    if (s.size() == 3 && s[0] == 1 && s[1] == Z) {
      L = s[2];
    } else if (s.size() == 2 && s[0] == Z) {
      L = s[1];
    }
    if (req.latent == nullptr || L <= 0) {
      if (err != nullptr) {
        *err = "MAGI-2's soundtrack wants a [1, " + std::to_string(Z) +
               ", L] latent";
      }
      return false;
    }
    // Whole, as the release decodes it (one tile: the edges are the
    // clip's own, zero-padded).
    _d->set_core_frames(L);
    std::vector<float> raw;
    if (!_d->decode(req.latent, L, &raw, /*clamp=*/false, req.progress,
                    err)) {
      return false;
    }
    const int C = _d->config().out_channels;
    const long n = (long)(raw.size() / (std::size_t)C);
    const long num = magi2::stretched_length(n);
    *pcm = magi2::fft_resample(raw.data(), C, n, num);
    // The release writes it as 16-bit PCM, which clips.
    for (float& v : *pcm) { v = std::clamp(v, -1.0f, 1.0f); }
    *shape = {C, (int)num};
    return true;
  }

  std::uint64_t resident_bytes() const override
  {
    return _d->resident_bytes();
  }

 private:
  std::unique_ptr<MetalOobleckDecoder> _d;
};

class Magi2VaeFamily final : public VaeModelFamily,
                             public VaeEncodeWorkingBytes {
 public:
  std::string_view tag() const noexcept override { return "magi2"; }

  void* query_extension(std::string_view id) noexcept override
  {
    if (id == kVaeEncodeWorkingBytesExt) {
      return static_cast<VaeEncodeWorkingBytes*>(this);
    }
    return nullptr;
  }

  // ---- VaeEncodeWorkingBytes: the host picture encoder's --------------
  std::size_t encode_working_bytes(const std::string&, int frames,
                                   int height, int width) const override
  {
    return frames == 1 ? Magi2ImageEncoder::working_bytes(height, width)
                       : 0;
  }

  bool claims(const std::string& root, const std::string& /*vae_dir*/,
              const std::string& /*model_type*/) const override
  {
    return magi2::is_root(root);
  }

  std::string vae_path(const std::string& root,
                       std::string_view role) const override
  {
    magi2::Layout l;
    if (!magi2::resolve_layout(root, &l)) { return {}; }
    // The checkpoint FILE the decoder opens, which is what its claim, its
    // weight set and its release all name.
    if (role == kRoleVideo) { return turbo_ckpt_(l.turbo); }
    if (role == kRoleAudio) { return l.audio; }
    if (role == kRoleVideoEncode) { return l.wan_vae; }
    return {};
  }

  std::vector<ResourceClaim>
  declare_resources(const std::string& root,
                    const std::string& /*vae_dir*/) const override
  {
    magi2::Layout l;
    std::size_t weights = 0;
    if (!magi2::resolve_layout(root, &l) ||
        !MetalMagi2Turbo::plan(l.turbo, 1, 1, 1, &weights, nullptr)) {
      return {};
    }
    // The checkpoint is a TRAINING STATE (the raw model, its EMA and the
    // optimizer's moments: 1.8 GB on disk); the decoder holds the EMA's
    // decoder in bf16. Booked at that, under the file it opens.
    std::vector<ResourceClaim> out =
        model_memory::weight_claims({turbo_ckpt_(l.turbo)});
    for (ResourceClaim& c : out) {
      bag::set_integer(c.extra, model_memory::kClaimHeldBytes,
                       (std::int64_t)weights);
    }
    return out;
  }

  // Each codec at what its load HOLDS, under the name it opens: the
  // turbo decoder (the EMA's decoder of a training state), Stable Audio
  // Open's decoder (the file is the whole model) and the Wan2.2 VAE's
  // encoder half, on the host. The stages book their claims at these.
  std::vector<StageHolding>
  declare_holdings(const std::string& root,
                   std::string_view role) const override
  {
    magi2::Layout l;
    if (!magi2::resolve_layout(root, &l)) { return {}; }
    StageHolding h;
    if (role == kRoleVideo) {
      if (!MetalMagi2Turbo::plan(l.turbo, 1, 1, 1, &h.preload, nullptr)) {
        return {};
      }
      h.source = turbo_ckpt_(l.turbo);
    } else if (role == kRoleAudio) {
      MetalOobleckDecoder::Config c;
      const auto w = MetalLlamaWeights::open_model(l.audio);
      if (!w.has_value() ||
          !MetalOobleckDecoder::config_from_dir(l.audio, &c, nullptr)) {
        return {};
      }
      h.source = l.audio;
      h.preload = MetalOobleckDecoder::held_bytes(*w, c);
    } else if (role == kRoleVideoEncode && !l.wan_vae.empty()) {
      const auto w = MetalLlamaWeights::open_model(l.wan_vae);
      if (!w.has_value()) { return {}; }
      h.source = l.wan_vae;
      h.preload = Magi2ImageEncoder::held_bytes(*w);
    } else {
      return {};
    }
    if (h.preload == 0) { return {}; }
    return {h};
  }

  std::vector<std::string> idle_peers(const std::string& root) const override
  {
    magi2::Layout l;
    if (!magi2::resolve_layout(root, &l)) { return {root}; }
    std::vector<std::string> v{l.preview, l.text_encoder};
    if (!l.refiner.empty()) { v.push_back(l.refiner); }
    return v;
  }

  std::unique_ptr<VaeDecoder>
  load_decoder(const VaeModelCreateArgs& args) override
  {
    magi2::Layout l;
    if (!magi2::resolve_layout(args.root, &l)) { return nullptr; }
    std::string err;
    auto d = MetalMagi2Turbo::load(
        open_weight_set(turbo_ckpt_(l.turbo), args.session), l.turbo,
        args.metal, &err);
    if (!d) {
      if (args.session != nullptr) {
        args.session->warn(fmt("magi2 turbo decoder: {}", err));
      }
      return nullptr;
    }
    return std::make_unique<Magi2VideoDecoder>(std::move(d));
  }

  std::unique_ptr<VaeEncoder>
  load_encoder(const VaeModelCreateArgs& args) override
  {
    magi2::Layout l;
    if (!magi2::resolve_layout(args.root, &l) || l.wan_vae.empty()) {
      if (args.session != nullptr) {
        args.session->warn(fmt("magi2: no vae/Wan2.2_VAE.pth under '{}' -- "
                               "image-to-video's picture has no encoder",
                               args.root));
      }
      return nullptr;
    }
    std::string err;
    auto e = Magi2ImageEncoder::load(open_weight_set(l.wan_vae, args.session),
                                     args.metal, &err);
    if (!e) {
      if (args.session != nullptr) {
        args.session->warn(fmt("magi2 picture encoder: {}", err));
      }
      return nullptr;
    }
    return std::make_unique<Magi2PictureEncoder>(std::move(e));
  }

  std::unique_ptr<AudioVaeDecoder>
  load_audio_decoder(const VaeModelCreateArgs& args) override
  {
    magi2::Layout l;
    if (!magi2::resolve_layout(args.root, &l)) { return nullptr; }
    MetalOobleckDecoder::Config cfg;
    std::string err;
    if (!MetalOobleckDecoder::config_from_dir(l.audio, &cfg, &err)) {
      if (args.session != nullptr) {
        args.session->warn(fmt("magi2 audio decoder: {}", err));
      }
      return nullptr;
    }
    auto d = MetalOobleckDecoder::load(open_weight_set(l.audio, args.session),
                                       args.metal, cfg, &err);
    if (!d) {
      if (args.session != nullptr) {
        args.session->warn(fmt("magi2 audio decoder: could not load '{}': "
                               "{}", l.audio, err));
      }
      return nullptr;
    }
    return std::make_unique<Magi2AudioDecoder>(std::move(d));
  }
};

[[maybe_unused]] const bool kRegistered =
    VaeModelRegistry::get().add(std::make_unique<Magi2VaeFamily>());

}  // namespace
}  // namespace vpipe::genai
