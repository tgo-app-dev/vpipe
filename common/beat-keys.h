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
// The frame rate as a fraction, when it is exactly one (29.97 is
// 30000/1001). Uint each. (The capture and decode stages have written
// these spellings since before they were named here.)
inline constexpr std::string_view kFpsNum = "fps_num";
inline constexpr std::string_view kFpsDen = "fps_den";
// This frame's presentation time in the file it was read from,
// microseconds from the file's start. Int.
inline constexpr std::string_view kPtsUs = "pts_us";
// Latent frames per second of an AUDIO latent. Real.
inline constexpr std::string_view kLatentsPerSecond = "latents_per_second";

// The content screen refused the prompt. Bool. Set on the conditioning
// and carried to the latent, where a decoder emits a refusal card of
// kRefusalWidth x kRefusalHeight (ints) instead of decoding.
inline constexpr std::string_view kContentBlocked = "content_blocked";
inline constexpr std::string_view kRefusalWidth   = "refusal_width";
inline constexpr std::string_view kRefusalHeight  = "refusal_height";

// ---- what a picture's samples MEAN --------------------------------------
//
// A picture beat ([C,H,W] U8, F16 or F32) says how its samples are encoded,
// as its source declared it, so a writer can tag the file it makes and
// nothing is re-interpreted on the way. Absent keys mean what a picture
// beat always meant: 8-bit RGB in sRGB (BT.709 primaries), straight
// alpha. F16 and F32 samples run 0..1 over the source's code range in its own
// transfer; a linear (scene- or display-linear) picture may exceed 1.
//
// ITU-T H.273 (CICP) code points, Uint, 2 = unspecified:
// primaries 1 BT.709, 9 BT.2020, 12 Display P3; transfer 1 BT.709,
// 8 linear, 13 sRGB, 16 PQ (ST 2084), 18 HLG; matrix -- the YCbCr matrix
// the samples were decoded from, kept for re-encoding -- 0 RGB,
// 1 BT.709, 6 BT.601, 9 BT.2020 NCL.
inline constexpr std::string_view kColorPrimaries = "color_primaries";
inline constexpr std::string_view kColorTransfer  = "color_transfer";
inline constexpr std::string_view kColorMatrix    = "color_matrix";
// Bool: the source's YCbCr was full range (the RGB samples always are).
inline constexpr std::string_view kColorFullRange = "color_full_range";
// "straight" | "premultiplied": how a 4-channel picture's colour relates
// to its alpha. Absent = straight.
inline constexpr std::string_view kAlphaMode = "alpha_mode";
// Bits per sample of the source (8, 10, 12, 16; 16 or 32 for float
// sources) -- what a writer can keep. Uint, informative.
inline constexpr std::string_view kSourceBits = "source_bits";
// HDR10 static metadata, as the source declared it. Mastering display
// (SMPTE ST 2086): an array of 10 numbers -- red, green, blue, white
// point x,y in units of 0.00002, then max and min luminance in units of
// 0.0001 cd/m^2. Content light (CTA-861.3): [max_cll, max_fall] in
// cd/m^2.
inline constexpr std::string_view kMasteringDisplay = "mastering_display";
inline constexpr std::string_view kContentLight     = "content_light";

}  // namespace sideband

}  // namespace vpipe

VPIPE_API_END

#endif
