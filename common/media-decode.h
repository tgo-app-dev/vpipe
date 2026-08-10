#ifndef VPIPE_COMMON_MEDIA_DECODE_H
#define VPIPE_COMMON_MEDIA_DECODE_H

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// One-shot FFmpeg decode helpers for single media items -- the glue
// between a media-line attachment (a file path or an in-memory byte
// payload; see common/media-line.h) and what the model encoders eat:
//   * images -> planar RGB u8 [3,H,W], channel order R,G,B, contiguous
//     (the same layout load-image / video-to-rgb emit and every vision
//     encoder's encode(rgb,H,W) takes);
//   * audio  -> f32 PCM in [-1,1] resampled to `target_sample_rate`
//     (16 kHz for every audio encoder in the tree), mono by default and
//     PLANAR when more than one channel is asked for;
//   * video  -> `num_frames` consecutive planar RGB u8 [3,H,W] planes at
//     the rate the container reports, with the soundtrack alongside.
// Any container/codec FFmpeg can demux+decode works (jpg/png/webp/...,
// wav/mp3/m4a/flac/...). The *_bytes variants read from memory via a
// custom AVIO context, so a base64 web attachment never touches disk.
//
// Pure software decode, synchronous, no session dependency beyond the
// dlopen'ed FFmpegLibraries. On failure returns nullopt and, when
// `error` is non-null, stores a one-line reason.
namespace vpipe {

class FFmpegLibraries;

struct DecodedImage {
  int                       width  = 0;
  int                       height = 0;
  std::vector<std::uint8_t> rgb;   // planar [3,H,W], contiguous
};

struct DecodedAudio {
  int                sample_rate = 0;   // == requested target rate
  // PLANAR f32: `channels` consecutive runs of `pcm.size() / channels`
  // samples. Mono (the default) makes that one run, so a caller that
  // never asks for more sees the flat waveform it always did.
  int                channels    = 1;
  std::vector<float> pcm;
};

struct DecodedVideo {
  int    width      = 0;
  int    height     = 0;
  int    num_frames = 0;
  // The rate the CONTAINER reports. Carried out because a model that
  // resamples a clip onto its own clock (MiniMax-H3's 24 fps) has no
  // other way to know what it is resampling FROM -- a rate lost here is
  // a reference conditioned at the wrong speed, with nothing to raise
  // about it.
  double fps        = 0.0;
  // `num_frames` consecutive planar [3,H,W] u8 planes.
  std::vector<std::uint8_t> rgb;

  // The soundtrack, when the container has one. Planar, like
  // DecodedAudio, and 0 channels when there is none.
  int                audio_channels    = 0;
  int                audio_sample_rate = 0;
  std::vector<float> pcm;
};

std::optional<DecodedImage>
decode_image_file(const FFmpegLibraries* libs,
                  const std::string&     path,
                  std::string*           error = nullptr);

std::optional<DecodedImage>
decode_image_bytes(const FFmpegLibraries*         libs,
                   std::span<const std::uint8_t>  bytes,
                   std::string*                   error = nullptr);

// `channels` is 1 (mono, the historical behaviour) or 2 (stereo,
// PLANAR). A source with a different channel count is downmixed or
// upmixed by swresample, in the one pass that also does the rate
// conversion -- resampling twice is how a waveform ends up filtered
// twice.
std::optional<DecodedAudio>
decode_audio_file(const FFmpegLibraries* libs,
                  const std::string&     path,
                  int                    target_sample_rate,
                  std::string*           error    = nullptr,
                  int                    channels = 1);

// Decode a video file's frames, its frame rate and its soundtrack.
//
// `max_seconds` bounds the decode: frames past it are never read, which
// is not an optimization but a memory bound -- 15 seconds of 1080p is
// 2.2 GB of planar RGB, and a caller that truncates the clip afterwards
// would pay for all of it first. 0 or negative means the whole file, up
// to the same internal pixel cap.
//
// `audio_sample_rate` is what the soundtrack is resampled to; pass 0 to
// skip audio entirely (a motion reference conditioned on pixels alone).
// A file with no audio stream is not an error -- `audio_channels` comes
// back 0.
std::optional<DecodedVideo>
decode_video_file(const FFmpegLibraries* libs,
                  const std::string&     path,
                  double                 max_seconds       = 0.0,
                  int                    audio_sample_rate = 0,
                  std::string*           error             = nullptr);

std::optional<DecodedAudio>
decode_audio_bytes(const FFmpegLibraries*         libs,
                   std::span<const std::uint8_t>  bytes,
                   int                            target_sample_rate,
                   std::string*                   error    = nullptr,
                   int                            channels = 1);

}

#endif
