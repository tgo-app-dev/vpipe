#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_LAYOUT_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_LAYOUT_H

// Where MAGI-2-preview's parts live under the release's root (sand-ai/
// MAGI-2-preview): the preview DiT (preview/, sharded safetensors, no
// config -- its shapes are read off the tensors), the refiner
// (refiner/), Qwen3.5-27B (text_encoder/), the turbo video decoder
// (turbo_vae/: a config JSON and checkpoint.ckpt), Stable Audio Open's
// autoencoder (stable-audio-open-1.0/) and the Wan2.2 VAE
// (vae/Wan2.2_VAE.pth, whose encoder image-to-video needs).
//
// The root is told by its files -- the preview's index naming the
// preview's own tensors, and the decoders beside it -- because nothing
// in it carries a class name.

#include <string>

namespace vpipe {
namespace genai {
namespace magi2 {

struct Layout {
  std::string root, preview, refiner, text_encoder, turbo, audio, wan_vae;
};

// True (and `out` filled; the optional parts empty where absent) for a
// MAGI-2-preview root. Cheap: a few stats and one index read, memoized
// per root.
bool resolve_layout(const std::string& root, Layout* out);
inline bool
is_root(const std::string& root)
{
  return resolve_layout(root, nullptr);
}

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_MAGI2_LAYOUT_H
