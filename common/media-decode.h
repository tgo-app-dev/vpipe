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
//   * audio  -> mono f32 PCM in [-1,1] resampled to `target_sample_rate`
//     (16 kHz for every audio encoder in the tree).
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
  std::vector<float> pcm;              // mono f32
};

std::optional<DecodedImage>
decode_image_file(const FFmpegLibraries* libs,
                  const std::string&     path,
                  std::string*           error = nullptr);

std::optional<DecodedImage>
decode_image_bytes(const FFmpegLibraries*         libs,
                   std::span<const std::uint8_t>  bytes,
                   std::string*                   error = nullptr);

std::optional<DecodedAudio>
decode_audio_file(const FFmpegLibraries* libs,
                  const std::string&     path,
                  int                    target_sample_rate,
                  std::string*           error = nullptr);

std::optional<DecodedAudio>
decode_audio_bytes(const FFmpegLibraries*         libs,
                   std::span<const std::uint8_t>  bytes,
                   int                            target_sample_rate,
                   std::string*                   error = nullptr);

}

#endif
