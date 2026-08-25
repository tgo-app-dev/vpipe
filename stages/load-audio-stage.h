#ifndef LOAD_AUDIO_STAGE_H
#define LOAD_AUDIO_STAGE_H

#include "common/encoded-segment.h"
#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>

namespace vpipe {

// Reads an audio file (or network URL) and emits its ENCODED packets,
// one Beat per packet -- the file-based sibling of `rtsp-capture`'s
// audio oport, and the upstream `audio-to-pcm` was written for.
//
//   oport0  EncodedSegmentPayload, kind = Audio, ONE PACKET per beat,
//           carrying `codec_id` / `sample_rate` / `channels` /
//           `extradata` from the stream's parameters. That is exactly
//           what audio-to-pcm's `ensure_decoder_` reads, so any codec
//           FFmpeg can decode works -- mp3, AAC, FLAC, ALAC, Opus,
//           Vorbis, or the PCM in a .wav.
//
//   load-audio -> audio-to-pcm -> (audio-transcribe | audio-tagging |
//                                  audio-segment | audio-vae-encode)
//
// ---- WHY IT DEMUXES AND DOES NOT DECODE ----
//
// `audio-to-pcm` already owns the decode, the resample to a configured
// rate and the accumulation into chunk-sized beats, and it is the stage
// every PCM consumer in the tree is written against. Decoding here would
// either duplicate that or produce a second, subtly different PCM
// contract.
//
// `load-video` stops at packets for the same reason, and it did not
// always: it decoded to FrameRefs, which no tensor-side stage could
// read. Both file sources now publish rtsp-capture's EncodedSegments, so
// a file and a camera are interchangeable inputs and the decode lives in
// one place per modality.
//
// It is also why the beat granularity is ONE PACKET rather than one
// file: audio-to-pcm feeds `seg.data` to the decoder as a single
// AVPacket, so a beat holding a whole file's concatenated frames would
// be one malformed packet.
//
// ---- TIMESTAMPS ARE MEDIA TIME ----
//
// `start_utc` / `end_utc` are the packet's presentation time measured
// from the START OF THE FILE, expressed as an epoch offset because that
// is the field's type. A file has no wall clock, and inventing "now"
// would make two runs over the same file disagree about when its
// samples happened. audio-to-pcm forwards this to `timestamp_us` on the
// PCM beat, and `audio-segment` / `audio-transcribe` only ever compare
// those to each other -- so a consistent media clock is what they need.
//
// Configuration (FlexData object on the 4th constructor parameter):
//   input_url        (string, required)  -- file path or URL
//   stream_index     (int, default -1)   -- ABSOLUTE stream index, video
//                                           streams included; -1 = the first
//                                           audio stream
//   read_timeout_ms  (int, default 0)    -- network open/read timeout
//   options          (object<string,string>) -- av_dict for the open
class LoadAudioStage final : public TypedStage<LoadAudioStage> {
public:
  static constexpr const char* kTypeName = "load-audio";

  LoadAudioStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);

  ~LoadAudioStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  std::uint64_t packets_emitted() const noexcept { return _packets; }
  unsigned      codec_id()        const noexcept { return _codec_id; }
  unsigned      sample_rate()     const noexcept { return _sample_rate; }
  unsigned      channels()        const noexcept { return _channels; }

private:
  void open_input_();
  std::string av_err_(int rc) const;

  const FFmpegLibraries* _libs = nullptr;

  std::string _input_url;
  int         _stream_index   = -1;
  int         _read_timeout_ms = 0;
  FlexData    _open_options;

  AVFormatContext* _fctx = nullptr;
  AVPacket*        _pkt  = nullptr;
  int              _idx  = -1;      // the chosen stream
  AVRational       _tb   = {0, 1};  // its time base

  // Cached stream parameters, copied onto every segment. Read once:
  // they are the same for every packet, and a consumer that opened a
  // decoder from the first beat must not see them change.
  unsigned             _codec_id    = 0;
  unsigned             _sample_rate = 0;
  unsigned             _channels    = 0;
  std::vector<uint8_t> _extradata;

  bool          _eof     = false;
  std::uint64_t _packets = 0;
  // The last packet's media time, carried forward for one with no pts
  // (some raw formats stamp none). Repeating the previous timestamp
  // keeps the stream monotonic; falling back to zero would make time
  // jump to the start of the file mid-clip.
  std::int64_t  _last_us = 0;
};

}  // namespace vpipe

#endif
