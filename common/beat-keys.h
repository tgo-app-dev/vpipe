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

// ---- one SOURCE over several beats ---------------------------------------
//
// A producer that restores a long source in pieces, carrying its state
// across the cuts, says so on every beat of the stream, so a consumer
// that keeps causal state of its own -- a video VAE's decode carries --
// continues across the cut too rather than opening cold. Absent: the beat
// stands alone, as every beat did before these existed.
//
// Bool: this beat continues the one before it on the same stream.
inline constexpr std::string_view kStreamContinues = "stream_continues";
// Int, on a group of frames (temporal-stack): how many of its leading
// frames the group before already carried -- its overlap. A consumer
// that stitches groups back into one source drops them.
inline constexpr std::string_view kRepeated = "repeated";
// Bool, on every beat of a stream: true on its last. A stage holding a
// model or carries for the stream keeps them until then.
inline constexpr std::string_view kStreamFinal = "stream_final";
// Int: of the frames this beat decodes to, how many are real. The end of
// a stream padded out to its producer's frame rule decodes to more, and a
// decoder drops the rest. Absent: all of them.
inline constexpr std::string_view kValidFrames = "valid_frames";
// Ints, for pieces that OVERLAP rather than continue: a producer that
// restores a long source as independent clips re-restores a few frames
// at each cut and has the decoder cross-fade them. `blend_head`: this
// beat's first N decoded frames are the same source frames as the last N
// of the beat before; a decoder blends the two (a raised cosine from the
// old to the new) instead of emitting both. `blend_tail`: this beat's
// last N frames will be repeated by the next beat; a decoder holds them
// back for that blend, and emits them as they are if no such beat comes.
inline constexpr std::string_view kBlendHead = "blend_head";
inline constexpr std::string_view kBlendTail = "blend_tail";
// Per decoded frame, the channel statistics of the picture it restores:
// [mean_r, mean_g, mean_b, std_r, std_g, std_b] over samples in [-1, 1].
// A decoder rescales each channel of each frame to them (AdaIN) -- the
// colour correction a restoration model's reference applies against its
// own input. Absent: no correction.
inline constexpr std::string_view kColorRef = "color_ref";
// Reals, from a producer that fed its model each source frame SEVERAL
// times -- FlashVSR+'s temporal compression 2 or 1 repeats every frame
// but the first 2 or 4 times, giving fast motion more latent frames --
// the N weights a decoder merges each run of N decoded frames back into
// one frame with, after the first `merge_lead` decoded frames, which pass
// through alone. `valid_frames` counts DECODED frames; `color_ref`,
// `color_ref_frames` and the blend counts count the merged ones. Absent:
// every decoded frame is a frame.
inline constexpr std::string_view kMergeWeights = "merge_weights";
// Int: decoded frames at the start of this beat that are not merged (the
// source's first frame, which its producer fed once). Absent: 0.
inline constexpr std::string_view kMergeLead = "merge_lead";
// Int: for this beat a decoder's colour-reference input carries a beat
// of this many source pictures, one per output frame at the output size
// (planar [n, 3, H, W]), and the decoder moves each frame's LOW
// FREQUENCIES to its picture's while keeping its own detail: the wavelet
// colour fix, `clamp(x - low(x) + low(ref))` with `low` five passes (or
// `color_ref_levels`) of a 3x3 Gaussian dilated 1, 2, 4, ... with
// edge-replicate padding, over samples in [-1, 1]. Absent: nothing to
// read.
inline constexpr std::string_view kColorRefFrames = "color_ref_frames";
// Int: the wavelet colour fix's levels. Absent: 5.
inline constexpr std::string_view kColorRefLevels = "color_ref_levels";
// String: on an IMAGE latent ([z, h, w]) that announces
// `color_ref_frames` 1, how its decode is corrected against the one
// picture its decoder's colour-reference input then carries ([3, H, W]
// at the output size, samples 0..1): "gaussian" -- clamp(x - low(x) +
// low(ref)) with `low` ONE Gaussian blur of `color_ref_sigma` (OpenCV's
// kernel: cvRound(8 sigma + 1) | 1 taps, reflect-101 edges; VOSR's
// `wavelet_color_fix`); "adain" -- each channel's mean and unbiased
// spread (+ 1e-5) moved to the picture's (VOSR's `adain_color_fix`);
// "none" -- the picture read and left, so the pairing holds. Over
// samples in 0..1, clamped there. Absent: "wavelet", the stack above,
// which an image decoder does not run.
inline constexpr std::string_view kColorRefMethod = "color_ref_method";
// Real: the "gaussian" colour fix's sigma, in output pixels. Absent: 5.
inline constexpr std::string_view kColorRefSigma = "color_ref_sigma";
// String: which of its model's decoders a latent asks to be decoded by.
// "tiny": the model's small CONDITIONED decoder in place of its VAE --
// FlashVSR's TCDecoder, which reads the restored clip's own source beside
// the latent (`decoder_cond_frames`). Absent or "vae": the VAE. A decoder
// with no such model decodes with the VAE, and says so.
inline constexpr std::string_view kDecoder = "decoder";
// Int: for this beat a decoder's colour-reference input carries a beat of
// this many SOURCE pictures at the output size (planar [n, 3, H, W]): the
// frames the latent's decode covers, from its first decoded frame, one
// per source frame. A conditioned decoder reads them beside the latent --
// decoded frame d takes picture d up to `merge_lead`, then lead + (d -
// lead) / r for runs of r (`merge_weights`), the last picture standing in
// past the end. Sent after the colour fix's pictures (`color_ref_frames`)
// when both are announced. Absent: nothing to read.
inline constexpr std::string_view kDecoderCondFrames = "decoder_cond_frames";
// The picture's own rectangle on a canvas it was padded out to, as Ints
// [left, top, right, bottom] in pixels (right and bottom exclusive) -- of
// the picture, or on a latent of the picture it decodes to.
// image-resample's `pad_to_grid` sets it; vae-encode carries it to the
// latent, generate-image to its output when that output takes the
// reference's size, and vae-decode cuts the decoded picture to it: BFL's
// single-reference edit, generated on the padded canvas and cropped back.
// Absent: all of it.
inline constexpr std::string_view kCropBox = "crop_box";

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
