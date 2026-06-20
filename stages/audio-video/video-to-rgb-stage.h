#ifndef VIDEO_TO_RGB_STAGE_H
#define VIDEO_TO_RGB_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/ffmpeg-libraries.h"
#include "common/ffmpeg-log-tap.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include <mutex>
#include <string>

extern "C" {
#include <libswscale/swscale.h>
}

namespace vpipe {

namespace metal_compute { class MetalCompute; }
struct EncodedSegment;

// Consumes Beat<EncodedSegment> (AVCC H.264 + extradata) on iport 0,
// decodes each contained frame via FFmpeg's software H.264 decoder,
// chroma-upsamples yuv420p -> yuv444p (bicubic), converts yuv444p ->
// planar RGB (gbrp), and emits one Beat<TensorBeat> per frame on
// oport 0 with shape [3, height, width].
//
// Clock-domain crosser: iport runs at segment rate (one beat per
// finalised RTSP capture, ~5s), oport runs at frame rate (~10-30
// beats/sec). Declared via differing iport/oport_clock_group.
//
// Output layout: TensorBeat carries pitch-padded strides reflecting
// FFmpeg's per-plane `linesize[p]` (SIMD alignment). Consumers that
// don't yet understand strides should call
// TensorBeat::materialize_contiguous() or check is_contiguous() per
// the recipe in apple-silicon/tensor-beat.h.
//
// Config (FlexData object):
//   normalize       (bool,   default true)  -- F32 only: divide
//                                              bytes by 255.0 so the
//                                              output is in [0, 1].
//                                              Ignored when
//                                              output_dtype == "u8".
//   oport_capacity  (int,    default 4)     -- oport buffer depth
//   hwaccel         (string, default "auto") -- "auto" | "videotoolbox"
//                                              | "none". "auto" tries
//                                              videotoolbox and falls
//                                              back to software if the
//                                              platform/codec can't.
//   output_dtype    (string, default "f32") -- "f32" | "u8". "u8"
//                                              emits raw RGB bytes
//                                              in [0, 255] with no
//                                              float conversion --
//                                              cheaper to produce and
//                                              4x smaller per frame.
//   output_width    (int,    optional)      -- if set together with
//   output_height   (int,    optional)         output_height, the
//                                              decoder's frames are
//                                              center-cropped to the
//                                              output aspect ratio
//                                              and bilinearly rescaled
//                                              to (output_width,
//                                              output_height). Both
//                                              keys must be set
//                                              together (or both
//                                              omitted). When unset
//                                              the stage emits frames
//                                              at the decoder's
//                                              native size. On the
//                                              Metal fast path the
//                                              crop+rescale+RGB
//                                              conversion runs in one
//                                              compute kernel.
// Fallback behaviour (when hwaccel is engaged): the stage keeps a
// hardware decoder context (`_vctx_hw`) plus a lazily-created
// software decoder context (`_vctx_sw`). Each GOP starts by trying
// the HW decoder; if the HW decoder fails on any AU in the GOP,
// the stage marks `_hw_broken_this_gop = true` and routes the
// rest of the GOP through SW (which catches up at the next IDR
// since SW also needs an IDR to sync). At every IDR boundary the
// HW flag resets, so HW is re-tried — recovering automatically
// when the underlying VT issue clears. Some 4K H.264 streams
// (notably certain Reolink models) periodically fail VT
// VTDecompressionSessionDecodeFrame() with OSStatus errors that
// FFmpeg surfaces as AVERROR_UNKNOWN; the per-GOP fallback keeps
// the pipeline live across those windows.
//
// An FFmpeg log tap is installed at construction so the real
// videotoolbox / h264 error message (lost in
// AVERROR_UNKNOWN by libavcodec's hwaccel wrapper) gets
// captured and surfaced in the stage's WARN logs whenever a
// decode attempt fails. Remove via destructor.
class VideoToRgbStage final : public TypedStage<VideoToRgbStage> {
public:
  static constexpr const char* kTypeName = "video-to-rgb";

  enum class HwMode { Auto, Videotoolbox, None };

  VideoToRgbStage(const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);
  ~VideoToRgbStage() override;

  Job process(RuntimeContext& ctx) override;
  Job drain  (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;
  // Clock-domain crosser: iport (segment rate) clock 0, oport (frame
  // rate) clock 1 -- declared per-port in kSpec.

private:
  // Per-context decoder state. Lives in the stage (one for HW,
  // one for SW). Both contexts decode the same bitstream
  // independently when needed; on a clean run only the HW one is
  // active. `synced` is the same gate `_decoder_synced` used to
  // be — false until we've successfully decoded a frame on this
  // context since the last open/flush.
  struct CodecSlot {
    AVCodecContext* ctx       = nullptr;
    bool            synced    = false;
    // True only for the HW slot (set when get_format negotiated
    // to a videotoolbox pix_fmt).
    bool            hw_active = false;
    bool            hw_logged = false;
  };

  Job  decode_segment_(RuntimeContext& ctx, const EncodedSegment& seg);
  Job  flush_decoder_ (RuntimeContext& ctx);
  void ensure_codec_hw_(const EncodedSegment& seg);
  void ensure_codec_sw_(const EncodedSegment& seg);
  void ensure_sws_    (int w, int h, int dec_pix_fmt);
  void teardown_codec_();
  void teardown_sws_  ();
  std::unique_ptr<BeatPayloadIntf>
  frame_to_tensor_beat_(const AVFrame*   src,
                        std::uint64_t    timestamp_us,
                        std::string_view camera_name);

  // (out_w, out_h) of the emitted TensorBeat for a src frame at
  // (src_w, src_h). When rescaling is disabled (both _output_width
  // and _output_height are 0) returns (src_w, src_h). Otherwise
  // returns (_output_width, _output_height).
  void emitted_dims_(int src_w, int src_h,
                     int* out_w, int* out_h) const noexcept;
  // Attempts to decode the single AU into a frame via `slot`.
  // Returns the number of frames emitted on this slot (0 or 1 in
  // the per-AU path), or -1 on a hard error (slot was flushed and
  // marked unsynced). `has_idr` controls whether an unsynced slot
  // is allowed to attempt the AU at all.
  Job  try_decode_au_(RuntimeContext& ctx,
                      CodecSlot&      slot,
                      const std::uint8_t* data,
                      std::size_t     len,
                      bool            has_idr,
                      std::uint64_t   timestamp_us,
                      std::string_view camera_name,
                      int*            out_rc,
                      std::size_t*    out_emitted);

  static enum AVPixelFormat
  get_format_(AVCodecContext* ctx,
              const enum AVPixelFormat* fmts) noexcept;

  std::string av_err_(int rc) const;

  const FFmpegLibraries* _libs = nullptr;

  CodecSlot       _hw;        // hardware-accelerated path (may downgrade
                              // to software inside ensure_codec_hw_ if
                              // hwaccel is unavailable for this codec)
  CodecSlot       _sw;        // pure-software fallback, lazily opened
                              // when _hw fails
  AVPacket*       _pkt        = nullptr;
  AVFrame*        _yuv_in     = nullptr;
  AVFrame*        _sw_frame   = nullptr;
  AVFrame*        _yuv444     = nullptr;
  AVFrame*        _gbrp       = nullptr;
  SwsContext*     _sws_chroma = nullptr;
  SwsContext*     _sws_rgb    = nullptr;

  int      _last_w           = 0;
  int      _last_h           = 0;
  int      _last_dec_pix_fmt = -1;
  unsigned _last_codec_id    = 0;

  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  bool                _normalize{};
  unsigned            _oport_capacity{};
  TensorBeat::DType   _output_dtype{};
  // Rescale-to dimensions. Both 0 ⇒ pass-through (no rescale).
  // Validated at config-parse time: both must be > 0 if either is.
  int                 _output_width  = 0;
  int                 _output_height = 0;

  HwMode             _hw_mode{};
  AVBufferRef*       _hw_device_ctx = nullptr;
  enum AVPixelFormat _hw_pix_fmt    = AV_PIX_FMT_NONE;
  metal_compute::MetalCompute* _mc = nullptr;   // session-owned
  bool               _metal_logged  = false;

  // Per-GOP fallback policy: cleared on every IDR boundary, set
  // when the HW slot fails any AU in the current GOP. While true,
  // P-slices of this GOP are routed through the SW slot only.
  // The next IDR clears the flag and HW is retried — recovering
  // automatically when the underlying VT issue goes away.
  bool      _hw_broken_this_gop  = false;
  // Counters used by INFO logs to surface the wait/recovery on
  // operator displays; reset on each successful sync.
  uint64_t  _hw_drops_since_desync = 0;
  uint64_t  _sw_drops_since_desync = 0;

  // FFmpeg log tap: captures the most recent codec-related
  // warning/error (tag "videotoolbox" / "h264" / "AVCodecContext")
  // into _last_ffmpeg_log under _ffmpeg_log_mtx. We surface that
  // text in our own WARN whenever a send_packet error fires, so
  // the AVERROR_UNKNOWN wrapper gets paired with the actual VT
  // OSStatus / h264 error string for triage.
  LogTapHandle  _ffmpeg_log_tap;
  mutable std::mutex  _ffmpeg_log_mtx;
  std::string         _last_ffmpeg_log;     // most recent line

  // Last seen EncodedSegment.camera_name. Propagated as a sideband
  // field on every emitted TensorBeat so downstream stages can
  // namespace per-camera state. Cached on the stage so flush_decoder_
  // (which runs at EOS without a source segment) can still stamp the
  // reorder-drain frames with the correct name.
  std::string         _last_camera_name;
};

}

#endif
