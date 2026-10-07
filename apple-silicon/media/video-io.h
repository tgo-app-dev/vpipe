#ifndef VPIPE_APPLE_MEDIA_VIDEO_IO_H
#define VPIPE_APPLE_MEDIA_VIDEO_IO_H

#include "apple-silicon/media/color-tags.h"
#include "apple-silicon/tensor-beat.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace vpipe::apple_media {

// Video through AVFoundation and VideoToolbox: ProRes (422 and 4444,
// with its alpha) and HEVC Main10 (PQ / HLG, with its HDR metadata) on
// the hardware codecs, frames as planar pictures in F16 -- 10 and 12
// bits survive -- or U8, tagged with the colour their file declares.
// Video only: a file's audio is not read or written here.

struct VideoInfo {
  int         width = 0;
  int         height = 0;
  int         fps_num = 0;   // the frame rate as a fraction
  int         fps_den = 1;
  double      duration_s = 0;
  std::int64_t frames = 0;   // estimated from duration x rate
  std::string codec;         // FourCC: "ap4h", "hvc1", "avc1", ...
  bool        has_alpha = false;
  ColorTags   color;         // as the track declares it
};

struct VideoReadOptions {
  TensorBeat::DType dtype = TensorBeat::DType::F16;  // or U8, F32
  bool   keep_alpha = true;   // 4 channels when the track has alpha
  double start_s = 0;
  double duration_s = 0;      // 0 = to the end
};

class VideoReader {
public:
  VideoReader();
  ~VideoReader();
  VideoReader(const VideoReader&) = delete;
  VideoReader& operator=(const VideoReader&) = delete;

  bool open(const std::string& path, const VideoReadOptions& opt,
            std::string* err);
  const VideoInfo& info() const;
  // The next frame into `out` (planar, tagged), with its presentation
  // time. False at the end of the range (`err` left empty) or on an
  // error (`err` set).
  bool read(TensorBeat* out, std::int64_t* pts_us, std::string* err);

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

// Codecs VideoWriter knows, by the name a graph uses:
//   prores4444, prores4444xq  ProRes 4444 (XQ); keeps alpha
//   prores422hq, prores422, prores422lt, prores422proxy
//   hevc   HEVC Main10 (10-bit; HDR when the frames are PQ or HLG)
//   hevc8  HEVC Main (8-bit)
//   h264   H.264 High (8-bit)
bool video_codec_known(std::string_view codec);

struct VideoWriteOptions {
  std::string codec = "prores4444";
  int         width = 0;
  int         height = 0;
  int         fps_num = 24;
  int         fps_den = 1;
  bool        alpha = false;    // keep a 4-channel picture's alpha
  ColorTags   color;            // what the frames' samples mean
  double      quality = 0;      // HEVC / H.264: 0..1; 0 = the encoder's
  std::int64_t bitrate = 0;     // HEVC / H.264 bits per second; 0 = auto
  // HEVC / H.264 rate and structure -- what a delivery spec asks for. 0,
  // -1 or "" leaves each to the encoder.
  std::int64_t max_bitrate = 0; // the most bits in any one second
  int         keyframe_interval = 0;  // the most frames between keyframes
  int         frame_reordering = -1;  // B-frames: 1 allowed, 0 not
  // H.264 only: "baseline" | "main" | "high" ("" high), its level ("3.0"
  // .. "5.2", "" auto) and entropy coding ("cabac" | "cavlc"; baseline
  // has CAVLC alone, and no B-frames).
  std::string profile;
  std::string level;
  std::string entropy;
};

// Is `profile` / `level` / `entropy` one VideoWriter takes for H.264
// ("" each: the default)?
bool h264_settings_known(const std::string& profile,
                         const std::string& level,
                         const std::string& entropy);

class VideoWriter {
public:
  VideoWriter();
  ~VideoWriter();
  VideoWriter(const VideoWriter&) = delete;
  VideoWriter& operator=(const VideoWriter&) = delete;

  // A QuickTime movie (.mov), or MPEG-4 (.mp4, not for ProRes).
  bool open(const std::string& path, const VideoWriteOptions& opt,
            std::string* err);
  // Frame `index` (its time is index / fps) from a planar picture of the
  // writer's size, U8, F16 or F32 (0..1).
  bool write(const TensorBeat& pic, std::int64_t index, std::string* err);
  // Closes the file; true when it was written whole.
  bool finish(std::string* err);

private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

// The frame rate as a fraction: the common NTSC rates exactly
// (23.976 -> 24000/1001), others to a millisecond.
void fps_fraction(double fps, int* num, int* den);

}  // namespace vpipe::apple_media

#endif
