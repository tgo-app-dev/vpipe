#ifndef VIDEO_AUDIO_TO_PCM_STAGE_H
#define VIDEO_AUDIO_TO_PCM_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

namespace vpipe {

struct EncodedSegment;

// Audio analog of `video-to-rgb`. Consumes Beat<EncodedSegment> with
// `kind=Audio` on iport 0 (typically wired to `rtsp-capture` oport 1
// for live capture — one beat per AAC frame), decodes via FFmpeg's
// AAC decoder, resamples to `output_sample_rate` Hz mono float32 via
// libswresample, and accumulates the resulting PCM into a rolling
// buffer. Once the buffered duration meets or exceeds
// `chunk_duration_s` the stage emits one Beat<TensorBeatPayload> on
// oport 0:
//
//   * shape  : [N] (1-D mono PCM)
//   * dtype  : F32 in [-1, 1]
//   * sideband:
//       timestamp_us  - UTC microseconds of the FIRST audio sample in
//                       the chunk (matches the source EncodedSegment's
//                       start_utc; matches what audio-transcribe-stage
//                       looks for).
//       sample_rate   - copy of output_sample_rate so downstream
//                       consumers don't need to read this stage's
//                       config.
//       duration_us   - chunk wall-clock duration in microseconds.
//
// Clock-domain crosser, same shape as `video-to-rgb`: iport runs at
// packet rate (~43 AAC frames / s @ 22 ms each), oport at chunk rate
// (~1 / chunk_duration_s).
//
// Config (FlexData object):
//   output_sample_rate   (int,   default 16000)  -- target sample rate
//                                                   for the resampler.
//                                              audio-transcribe-stage's
//                                                   Qwen3-ASR encoder
//                                                   expects 16 kHz.
//   chunk_duration_s     (real,  default 10.0)   -- minimum duration of
//                                                   each emitted chunk
//                                                   in seconds. A
//                                                   larger value gives
//                                                   the ASR more
//                                                   context per call
//                                                   at the cost of
//                                                   higher latency.
//                                                   For STREAMING ASR
//                                                   set this small
//                                                   (e.g. 0.032 = one
//                                                   AAC frame) so the
//                                                   stage emits tiny
//                                                   PCM frames at line
//                                                   rate. The
//                                                   audio-segment +
//                                                   streaming audio-
//                                                   transcribe pair
//                                                   then drive the
//                                                   utterance boundary.
//   max_chunk_duration_s (real,  default 30.0)   -- hard cap; even if
//                                                   the consumer is
//                                                   slow the stage
//                                                   flushes at this
//                                                   duration so a
//                                                   single chunk
//                                                   doesn't outgrow
//                                                   the encoder's
//                                                   max_source_positions
//                                                   ceiling.
//   oport_capacity       (int,   default 4)      -- oport buffer depth
//                                                   in chunks.
//   flush_on_eos         (bool,  default true)   -- emit a partial
//                                                   chunk on drain
//                                                   when the source
//                                                   closes; turn off
//                                                   if you only want
//                                                   full-duration
//                                                   chunks.
//   emit_log_every       (int,   default 1)      -- emit the per-chunk
//                                                   INFO log every Nth
//                                                   chunk. Use a high
//                                                   value (e.g. 31 for
//                                                   ~one log/s @ 32 ms
//                                                   frames) to keep
//                                                   small-frame
//                                                   streaming
//                                                   pipelines from
//                                                   spamming the log
//                                                   while still seeing
//                                                   periodic health
//                                                   pings. >= 1.
//
// On a codec / sample rate / channel layout change between segments
// the decoder and resampler are torn down and rebuilt on the next
// packet -- same recovery shape as the video stage's codec switch.
class AudioToPcmStage final : public TypedStage<AudioToPcmStage> {
public:
  static constexpr const char* kTypeName = "audio-to-pcm";

  AudioToPcmStage(const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);
  ~AudioToPcmStage() override;

  Job process(RuntimeContext& ctx) override;
  Job drain  (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;
  // Clock-domain crosser: iport (packet rate) clock 0, oport (chunk
  // rate) clock 1 -- declared per-port in kSpec.

  // Test-only accessors.
  int    output_sample_rate() const noexcept { return _output_sample_rate; }
  double chunk_duration_s()   const noexcept { return _chunk_duration_s; }
  double max_chunk_duration_s() const noexcept { return _max_chunk_duration_s; }

private:
  // Decode a single AAC frame's worth of bytes through the FFmpeg
  // decoder + swresample pipeline; appends mono-f32-at-output_sr
  // samples to `_chunk_buf`. Returns false on hard decoder error;
  // the caller drops the beat and continues.
  bool decode_one_(const EncodedSegment& seg);
  // Emit the buffered chunk on oport 0. Resets the buffer + chunk
  // start timestamp. No-op when `_chunk_buf` is empty.
  Job  emit_chunk_(RuntimeContext& ctx);
  // (Re-)open the AAC decoder for `seg`'s codec_id / extradata.
  // Returns true on success.
  bool ensure_decoder_(const EncodedSegment& seg);
  // (Re-)allocate the swr context when input format changes.
  bool ensure_resampler_(int             in_sample_rate,
                         int             in_channels,
                         enum AVSampleFormat in_fmt);
  void teardown_decoder_();
  void teardown_resampler_();

  // ---- config attributes; defaults live in kSpec.attrs and are read
  // in the constructor via attr_*. Declarations carry no non-zero
  // default. ----
  int    _output_sample_rate{};
  double _chunk_duration_s{};
  double _max_chunk_duration_s{};
  unsigned _oport_capacity{};
  bool     _flush_on_eos{};
  unsigned _emit_log_every{};

  // ---- FFmpeg state ----
  const FFmpegLibraries* _libs = nullptr;
  AVCodecContext*        _dctx = nullptr;
  AVPacket*              _pkt  = nullptr;
  AVFrame*               _frame = nullptr;
  SwrContext*            _swr  = nullptr;

  // Last-seen input format for cache invalidation.
  unsigned        _last_codec_id        = 0;
  int             _last_in_sample_rate  = 0;
  int             _last_in_channels     = 0;
  enum AVSampleFormat _last_in_sample_fmt = AV_SAMPLE_FMT_NONE;

  // Rolling chunk buffer (mono f32 at _output_sample_rate).
  std::vector<float> _chunk_buf;
  // UTC microseconds of the FIRST sample in `_chunk_buf`. Set on the
  // first appended packet of a chunk; reset to 0 after emit.
  std::uint64_t      _chunk_first_ts_us = 0;
  bool               _chunk_has_ts      = false;

  // Bookkeeping for test / log.
  std::uint64_t      _chunks_emitted    = 0;
  std::uint64_t      _packets_decoded   = 0;
  bool               _decoder_warned    = false;
};

}

#endif
