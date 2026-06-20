#ifndef ENCODED_SEGMENT_H
#define ENCODED_SEGMENT_H

#include "common/beat-payload-intf.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// One chunk of contiguous encoded media from a capture stream, plus
// the parallel metadata a consumer needs to correlate it with the
// <camera>-videos LMDB row.
//
// `data` granularity depends on the producer:
//   * rtsp-capture (live) emits one Beat per H.264 access unit
//     (i.e. per encoded video frame) on oport 0, and one Beat per
//     audio packet (one AAC frame) on oport 1. `data` is annex-B
//     framed for video (start-code prefixed NALs, the framing
//     FFmpeg's rtp depacketizer / global-header encoder emits) and
//     raw AAC frames for audio. `extradata` may be empty when the
//     source delivers SPS/PPS in-band on every IDR (cameras
//     commonly do); the H.264 decoder then picks them up from the
//     IDR's inline NALs.
//   * extract_encoded_segments (file → Beat helper used in tests
//     and any catch-up consumer) emits one Beat per whole mp4
//     file. `data` is AVCC-framed video (4-byte BE length prefix
//     per NAL) and raw AAC frames for audio. `extradata` carries
//     AVCDecoderConfigurationRecord / AudioSpecificConfig.
//
// Consumers (e.g. video-to-rgb) sniff the leading bytes of `data`
// to pick the matching access-unit splitter at decode time.
//
// `db_key` identifies the parent segment in the
// `<camera_name>-videos` LMDB sub-db. For per-GOP Beats `start_utc`
// / `end_utc` / `duration_us` describe the GOP's wall-clock window;
// `path` is the parent segment's mp4 path (consumers can also look
// it up via `db_key`).
struct EncodedSegment {
  enum class Kind { Video, Audio };

  Kind        kind = Kind::Video;

  // 8-byte big-endian microseconds-since-epoch. Matches the
  // <camera>-videos sub-db key the capture stage already writes.
  std::string db_key;

  std::string camera_name;
  std::string path;                       // source mp4 absolute path
  std::chrono::system_clock::time_point start_utc{};
  std::chrono::system_clock::time_point end_utc{};
  int64_t     duration_us = 0;

  unsigned    codec_id    = 0;            // AV_CODEC_ID_* value
  unsigned    width       = 0;            // video only
  unsigned    height      = 0;            // video only
  unsigned    sample_rate = 0;            // audio only
  unsigned    channels    = 0;            // audio only

  std::vector<uint8_t> extradata;
  std::vector<uint8_t> data;
};

// Pipeline-edge payload form. Inherits the verbatim segment fields
// and adds the BeatPayloadIntf virtuals.
class EncodedSegmentPayload : public EncodedSegment,
                              public BeatPayloadIntf {
public:
  EncodedSegmentPayload() = default;
  explicit
  EncodedSegmentPayload(const EncodedSegment& s)
    : EncodedSegment(s)
  {}
  explicit
  EncodedSegmentPayload(EncodedSegment&& s) noexcept
    : EncodedSegment(std::move(s))
  {}

  std::unique_ptr<BeatPayloadIntf>
  clone() const override
  {
    return std::make_unique<EncodedSegmentPayload>(
        static_cast<const EncodedSegment&>(*this));
  }

  std::string
  describe() const override
  {
    std::string s = "EncodedSegment ";
    s += (kind == Kind::Audio ? "audio" : "video");
    s += " codec=";
    s += std::to_string(codec_id);
    s += " ";
    s += std::to_string(data.size());
    s += "B";
    return s;
  }
};

}

#endif
