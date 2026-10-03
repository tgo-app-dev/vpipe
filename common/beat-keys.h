#ifndef VPIPE_COMMON_BEAT_KEYS_H
#define VPIPE_COMMON_BEAT_KEYS_H

// Keys of the FlexData that travels WITH beats: the object a text or
// model-select beat is, and the `sideband` a TensorBeat carries beside
// its data. A plugin stage meets these whenever it is wired to a host
// stage, and a model family meets the latent's sideband again in
// VaeDecodeRequest::sideband -- so they are the host's contract, added
// and never renamed or repurposed, with an absent key meaning what it
// meant before the key existed.
//
// Keys a family defines for ITSELF (its own model_config knobs, a
// sideband only its own stages read) are the family's business and do
// not belong here. The conditioning sideband a generating stage hands a
// family is in generative-models/gen-input.h.

#include <string_view>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {

// ---- the object a beat IS -----------------------------------------------
namespace beat {

// Text beats -- a caption, a transcript, a chat reply -- as an object.
// (`text-prompt` emits the bare string; a reader takes both.)
inline constexpr std::string_view kText = "text";
// A `model-select` beat: the model directory or registry key it chose.
inline constexpr std::string_view kHfDir = "hf_dir";
// Bool, on a text beat (or a TensorBeat's sideband): this beat is one of a
// SERIES that ends with the stream -- a training dataset. A consumer that
// loads a model for it holds the model across the series and releases it
// at end of stream, rather than after every beat as a generation graph's
// idle policy would; absent or false is the old behaviour.
inline constexpr std::string_view kBatch = "batch";
// Bool, on a text beat: encode the text even when it is EMPTY -- the empty
// prompt's own conditioning, which caption dropout trains against. Absent
// or false, an empty prompt produces nothing, as before.
inline constexpr std::string_view kAllowEmpty = "allow_empty";

}  // namespace beat

// ---- a TensorBeat's sideband --------------------------------------------
namespace sideband {

// The model a beat was made by, as the user named it (a registry key or
// a path). Carried from latent to frames so a saved file can say.
inline constexpr std::string_view kModelName = "model_name";

// Frames per second of a clip, or of the clip a latent decodes to. Real.
inline constexpr std::string_view kFps = "fps";
// Frames in the WHOLE clip a beat belongs to. Int.
inline constexpr std::string_view kFrames = "frames";
// This beat's frame index within that clip, from 0. Int.
inline constexpr std::string_view kFrame = "frame";
// Latent frames per second of an AUDIO latent. Real.
inline constexpr std::string_view kLatentsPerSecond = "latents_per_second";

// The content screen refused the prompt. Bool. Set on the conditioning
// and carried to the latent, where a decoder emits a refusal card of
// kRefusalWidth x kRefusalHeight (ints) instead of decoding.
inline constexpr std::string_view kContentBlocked = "content_blocked";
inline constexpr std::string_view kRefusalWidth   = "refusal_width";
inline constexpr std::string_view kRefusalHeight  = "refusal_height";

}  // namespace sideband

}  // namespace vpipe

VPIPE_API_END

#endif
