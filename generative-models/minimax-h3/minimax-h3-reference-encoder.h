#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_REFERENCE_ENCODER_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_REFERENCE_ENCODER_H

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
namespace genai {

class MetalMiniMaxH3AudioVae;
class MetalMiniMaxH3VideoVae;
class MetalQwenVisionEncoder;
class MiniMaxH3TextEncoder;

namespace minimax_h3 {

// Turning a `ref2va` request's decoded media into everything the DiT
// needs: the conditioning rows, the reference latent rows of both
// modalities, and the per-reference geometry the packed layout is built
// from.
//
// This is the glue over four pieces that already exist -- the reference
// normalization (minimax-h3-references.h), the Qwen3-VL vision tower,
// the video VAE encoder and the audio VAE encoder -- and it is its own
// file because the ORDER and the pairing are the parts that fail
// silently. Two of them in particular:
//
//   * a reference is read TWICE, by two models with different rules.
//     The vision tower sees it at 2 fps and at its own smart-resized
//     canvas; the video VAE sees the full 24 fps clip at MiniMax-H3's
//     canvas. Feeding either one the other's pixels still produces a
//     well-shaped tensor.
//   * every reference contributes to a shared, ORDERED clock. The
//     layout numbers the `<Picture i>` / `<Audio j>` / `<Video k>`
//     labels and advances the rotary origin in the order the references
//     are read, so the rows emitted here and the runs the layout
//     reserves have to be built from one traversal, not two.
//
// Nothing here opens a media file. The caller decodes (common/
// media-decode.h) and brings the RATES along -- a frame rate lost on the
// way in is a reference conditioned at the wrong speed, with nothing to
// raise about it.

// One decoded reference, in the order the model should read it.
struct MediaReference {
  enum class Kind { kImage, kVideo, kAudio };
  Kind kind = Kind::kImage;

  // Pixels: `num_frames` consecutive planar [3, height, width] u8
  // planes, at `fps`. One frame for kImage; unused by kAudio.
  std::vector<std::uint8_t> rgb;
  int    num_frames = 0;
  int    height     = 0;
  int    width      = 0;
  double fps        = kFps;

  // The soundtrack: planar [channels, samples] f32 at `sample_rate`,
  // ALREADY at the audio VAE's rate (see normalize_audio_reference on
  // why the resample belongs to the decoder). Empty for a reference
  // that carries none; a kAudio reference is nothing but this.
  std::vector<float> pcm;
  int channels    = 0;
  int sample_rate = 0;

  bool has_audio() const { return channels > 0 && !pcm.empty(); }

  // This reference's own canvas short edge, overriding the plan's for
  // this one reference. Negative uses the plan's; 0 is the never-
  // upsample rule (resolve_canvas_within).
  //
  // Per REFERENCE and not per kind, because the reason to set it is a
  // property of the picture rather than of the checkpoint: a caller that
  // has already sized one -- through a resample stage, a crop, or a
  // generator -- is telling the encoder to leave that one alone, and it
  // may be telling it about only one of nine.
  int short_edge = -1;
};

// What the encoder had to do to a reference to make it fit.
//
// REPORTED here rather than warned about, because this is a library:
// which of these deserves a user's attention is the caller's judgement,
// and a test wants the numbers rather than a sentence. A stage turns
// them into warnings.
//
// The temporal counts are kept separate on purpose -- there are three
// distinct reductions and they have different remedies. A 30 fps clip
// loses frames to the RATE resample whether or not it is also too long;
// `target_frames` then truncates; and the VAE additionally snaps down to
// a whole number of `17n + 5` chunks. Collapsing them into one
// percentage tells a user that something was lost without telling them
// which knob gets it back.
struct ReferenceFit {
  MediaReference::Kind kind = MediaReference::Kind::kImage;

  // Pixels in, pixels encoded. Equal when the reference was already on
  // the grid -- the pass-through route, and the only one that reproduces
  // the caller's pixels exactly.
  int src_h = 0, src_w = 0;
  int canvas_h = 0, canvas_w = 0;

  double src_fps     = 0.0;
  int    src_frames  = 0;   // as handed in
  int    rate_frames = 0;   // after the 24 fps resample
  int    used_frames = 0;   // after truncation to target_frames
  int    vae_frames  = 0;   // after the 17n+5 snap, what the VAE saw

  double audio_src_seconds  = 0.0;
  double audio_kept_seconds = 0.0;

  bool rescaled() const
  {
    return canvas_h > 0 && (canvas_h != src_h || canvas_w != src_w);
  }
  bool upscaled() const
  {
    return canvas_h > src_h || canvas_w > src_w;
  }
};

// The limits MiniMax-H3 documents for the released checkpoint. They
// bound validation and nothing else, so a fine-tune that packs more
// raises them rather than patching around them.
struct ReferenceLimits {
  int max_images     = 9;
  int max_videos     = 3;
  int max_audios     = 3;
  int max_references = 12;
};

// Everything the encoders need that is not a model or a reference.
struct ReferencePlan {
  // The GENERATED frame count, already snapped to `17n + 5`. Both the
  // video and the audio truncation are measured against it: a reference
  // is cut to the duration of what is being generated.
  int target_frames = 0;

  // The reference-image rule (its own short edge, no area cap) and the
  // canvas rule a reference VIDEO shares with the target.
  //
  // `canvas_short_edge` of 0 means the source clip's own short edge --
  // the never-upscale route, which is not a different rule but the same
  // one started from what the file actually carries. See
  // normalize_video_reference; the area cap still binds a large source
  // either way.
  //
  // It moves THREE things, not two. The reference rows the DiT attends
  // to and the VAE encode that produced them are the obvious pair; the
  // third is the CONDITIONER, because the vision tower is handed the
  // same normalized pixels and smart-resizes from them, so this canvas
  // also decides how many vision tokens a clip contributes. MEASURED on
  // the two-reference ref2va example, 1344x768 -> 960x544: 13120 -> 7144
  // reference rows AND 3115 -> 2119 conditioning rows. Lowering it is a
  // fidelity decision on both channels, not a pure cost one.
  int          reference_image_short_edge = 2048;
  int          canvas_multiple            = 32;
  int          canvas_short_edge          = 768;
  std::int64_t canvas_max_pixels          = 768LL * 1344LL;
  // The image path's area cap. 0 -- uncapped -- is the released
  // checkpoint's rule, where `reference_image_short_edge` is the only
  // bound. It stops being a bound the moment a caller says 0 to take a
  // picture as it arrived, and an uncapped 4K still is 8160 DiT rows and
  // ~220 VAE tiles, so a graph that feeds raw pictures wants this set.
  std::int64_t reference_image_max_pixels = 0;

  // The rate the CONDITIONER reads a video reference at -- every
  // `fps / video_sample_fps`-th of the normalized frames. Not the rate
  // the VAE encodes it at, which is the full 24.
  double video_sample_fps = 2.0;
  int    temporal_patch   = 2;   // Qwen3-VL's, for the block merge

  // The DiT's spatial patch, for packing latents into rows.
  int patch_h = 2;
  int patch_w = 2;
};

// What the encoders produced, in packed order.
struct EncodedReferences {
  // One entry per reference, in read order -- what
  // build_ref2va_packed_sequence takes.
  std::vector<Reference> layout;

  // The reference VIDEO rows: [rows, video_channels * patch_h *
  // patch_w] f32, whitened, concatenated in reference order. This is
  // the leading block of the denoise loop's video buffer.
  std::vector<float> video_rows;
  int                video_row_elems = 0;

  // The reference AUDIO rows: [rows, audio_latent_channels] f32,
  // whitened, channel-major within a reference (the stereo pair is the
  // outer axis) and concatenated in reference order.
  std::vector<float> audio_rows;
  int                audio_row_elems = 0;

  // The conditioning the DiT's text rows read: bf16 [n_tokens,
  // text_dim], the presentation of the whole request. Empty when the
  // caller passed no text encoder.
  metal_compute::SharedBuffer conditioning;
  std::vector<int>            token_tags;
  int                         n_tokens = 0;

  // One per reference, in read order, parallel to `layout`.
  std::vector<ReferenceFit> fits;
};

// The models. Each may be null when the corresponding work is not
// wanted, which is what makes this testable a piece at a time: an
// audio-only pass needs the audio VAE alone.
struct ReferenceEncoders {
  metal_compute::MetalCompute* mc = nullptr;
  MetalQwenVisionEncoder* vision    = nullptr;
  MetalMiniMaxH3VideoVae* video_vae = nullptr;
  MetalMiniMaxH3AudioVae* audio_vae = nullptr;
  MiniMaxH3TextEncoder*   text      = nullptr;

  // Called as each reference is finished, and once more when the
  // presentation is built. Optional.
  //
  // `total` is therefore refs.size() + 1, not refs.size(): the
  // presentation is ONE conditioner call over the whole request, made
  // AFTER the last reference has reported, so a total that stopped at
  // the references would reach 100% and then leave the caller waiting at
  // it. Counting it keeps that last stretch legible as unfinished work
  // rather than as a hang.
  std::function<void(int done, int total)> progress;
};

// Which frames of a normalized 24 fps reference the CONDITIONER reads,
// and the timestamp of every vision block they merge into.
//
// The stride is `fps / sample_fps` and the cursor walks it in floating
// point, so the indices are not an arithmetic sequence: at 24 -> 2 fps
// they are 0, 12, 24, ..., but a source at a rate that does not divide
// evenly drifts and the DEDUPLICATION (`round(cursor) > last`) is what
// keeps a frame from being read twice.
//
// Qwen3-VL then merges the sampled frames in groups of `temporal_patch`
// -- repeating the last timestamp when the count does not divide -- and
// labels a group with the MEAN of its two timestamps. At 2 fps that
// makes the first block 0.25 s, which "%.1f" renders as "0.2" rather
// than "0.3": both C's and Python's formatting round half to even.
//
// False when the clip is too short to fill one merged group, with the
// minimum length in `err` -- a case worth naming, since the tower would
// otherwise silently pad it.
bool condition_frame_indices(int num_frames, double fps, double sample_fps,
                             int temporal_patch, std::vector<int>* indices,
                             std::vector<float>* block_seconds,
                             std::string* err = nullptr);

// Check a request against the released checkpoint's limits.
//
// The one rule that is not a count: an AUDIO reference may not be the
// only kind. A soundtrack never reaches the conditioner, so a request
// of nothing but audio has no visual reference at all -- it is a `t2va`
// request wearing a `ref2va` shape, and it packs a sequence whose text
// rows describe references the model cannot see.
bool validate_reference_request(const std::vector<MediaReference>& refs,
                                const ReferenceLimits& limits,
                                std::string* err = nullptr);

// Normalize and encode a whole `ref2va` request.
//
// The references are traversed ONCE, in read order, and each is
// normalized onto MiniMax-H3's rates before either model sees it. A
// reference's soundtrack is encoded BEFORE its pixels, mirroring the
// order its rows are packed in.
//
// False on the first failure, with a reason in `err`. Partial output is
// not left behind: `out` is only written on success.
bool encode_references(const std::vector<MediaReference>& refs,
                       std::string_view prompt, const ReferencePlan& plan,
                       const ReferenceEncoders& models,
                       EncodedReferences* out, std::string* err = nullptr);

}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
