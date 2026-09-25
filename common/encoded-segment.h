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
// Metadata key under which a container vpipe wrote records the TRUE
// (unpadded) audio sample count. Reverse-DNS, because mp4 carries an
// arbitrary key only in the QuickTime `mdta` form.
inline constexpr const char* kAudioSamplesMetaKey =
    "com.tgo.vpipe.audio_samples";

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
  // Source frame rate as a rational (fps = fps_num / fps_den), video
  // only. Taken from the input stream's avg_frame_rate (else
  // r_frame_rate); 0/0 when the source did not advertise one.
  // Downstream decoders (video-to-rgb) forward this onto each emitted
  // frame's TensorBeat sideband so a sink (hls-broadcast) can adopt the
  // original cadence without a hard-coded default.
  unsigned    fps_num     = 0;            // video only
  unsigned    fps_den     = 0;            // video only
  unsigned    sample_rate = 0;            // audio only
  unsigned    channels    = 0;            // audio only

  // AUDIO GAPLESS BOOKKEEPING, in samples at `sample_rate`.
  //
  // A block codec cannot encode an arbitrary length. AAC emits whole
  // 1024-sample frames and primes its decoder with a frame of silence,
  // so a decoded stream carries padding at BOTH ends: `skip_head`
  // samples of encoder priming before the first real sample, and
  // whatever it took to round the tail up to a whole frame after the
  // last one. Played alone that is inaudible; CONCATENATED it is a gap
  // at every join, which is what a multi-part video hears.
  //
  // The container knows the priming (`initial_padding`) but NOT the
  // true length -- an mp4's audio duration is the PADDED count. So the
  // writer records the true count and the reader trims to it. 0 means
  // "not known", and nothing is trimmed at that end.
  // Which JOINED part this packet came from, 0 when there is only one.
  // `skip_head` / `total_samples` describe THAT part, so a consumer
  // counting samples has to restart its count when this changes.
  unsigned     part_index   = 0;
  std::int64_t skip_head    = 0;          // audio only, encoder priming
  std::int64_t total_samples = 0;         // audio only, true length

  // Whether this packet can START a decode, i.e. the container's
  // AV_PKT_FLAG_KEY. Video only.
  //
  // For H.264 the consumer does NOT read this -- it scans the access
  // unit for an IDR NAL instead, because the live source (rtsp-capture)
  // has no container to ask and delivers SPS/PPS/IDR in-band. For every
  // OTHER codec that scan is meaningless: FFV1 has no NAL framing at
  // all, so a gate written against H.264's returns false forever and
  // the decoder never starts. See video-to-rgb's sync gate.
  //
  // DEFAULTS TRUE, and the default is the safe direction: a producer
  // that does not know must not be able to stall a decoder for the
  // length of a stream. The cost of guessing true on a long-GOP codec
  // is a few bad frames until the next real keyframe; the cost of
  // guessing false is silence.
  bool       key_frame = true;

  // WHAT THE CONTAINER SAYS THE YUV MEANS, video only, as AVCOL_* ints.
  //
  // Carried because the bitstream is not the only place it lives and a
  // decoder built from `extradata` alone can only ever see the other
  // one. H.264 may put a colour description in the SPS VUI, and when it
  // does that wins -- but plenty of files tag only the container (mp4's
  // `colr` box, matroska's colour elements), and for those the VUI is
  // silent and this is the whole of the answer. ffmpeg's own decode
  // seeds the codec context from the container and lets the VUI
  // overwrite it, which is the behaviour these fields exist to
  // reproduce.
  //
  // UNSPECIFIED is the honest default and what every producer that does
  // not know writes: the reader then falls back on convention, which is
  // where a wrong answer becomes a hue tilt nothing reports.
  int color_range     = 0;   // AVCOL_RANGE_UNSPECIFIED
  int colorspace      = 2;   // AVCOL_SPC_UNSPECIFIED
  int color_primaries = 2;   // AVCOL_PRI_UNSPECIFIED
  int color_trc       = 2;   // AVCOL_TRC_UNSPECIFIED

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
