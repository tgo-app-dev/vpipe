#ifndef VPIPE_GENERATIVE_MODELS_VAE_HOST_EXT_H
#define VPIPE_GENERATIVE_MODELS_VAE_HOST_EXT_H

// What the HOST asks of a registered VAE family beyond the installed
// interface. HOST-ONLY: not installed, so a plugin cannot name any of it
// and nothing about the plugin ABI moves.
//
// A decoder's WORKING SET, through VaeDecoder::query_extension:
//
// vae-decode revises its decode arena on every beat to the frames the
// decode emits -- which for a registered decoder was ALL it could book:
// the interface has no way to say what the decoder holds while it runs.
// An in-tree family that knows answers here, and the stage adds it.

#include <cstddef>
#include <string_view>
#include <vector>

namespace vpipe::genai {

inline constexpr std::string_view kVaeWorkingBytesExt =
    "vpipe.vae-working-bytes/1";

class VaeWorkingBytes {
public:
  virtual ~VaeWorkingBytes() = default;
  // Bytes a decode of a latent of this shape allocates beyond the frames
  // it hands back (the beat's shape, as VaeDecodeRequest::shape).
  virtual std::size_t working_bytes(const std::vector<int>& shape) const = 0;
};

// AN ENCODE'S WORKING SET, through VaeModelFamily::query_extension -- the
// FAMILY, so vae-encode can book it at launch, before its encoder loads,
// and revise it per picture: what encoding `frames` pictures of `height`
// x `width` holds beside the encoder's weights (MAGI-2's picture encoder
// runs on the host: its activations and the middle's attention scores,
// which grow with the picture's area squared). 0: cannot say.
inline constexpr std::string_view kVaeEncodeWorkingBytesExt =
    "vpipe.vae-encode-working-bytes/1";

class VaeEncodeWorkingBytes {
public:
  virtual ~VaeEncodeWorkingBytes() = default;
  virtual std::size_t encode_working_bytes(const std::string& root,
                                           int frames, int height,
                                           int width) const = 0;
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_VAE_HOST_EXT_H
