#include "common/media-decode.h"

#include "common/ffmpeg-libraries.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

// Hard cap on decoded audio (samples at the target rate): 20 minutes.
// A runaway/corrupt stream must not balloon host memory; anything a
// chat-attached clip legitimately needs sits far below this.
constexpr size_t kMaxAudioSeconds = 20 * 60;

// The same idea for video, where the cost per unit is far higher:
// planar RGB is 6 MB a 1080p frame, so a whole-file decode of a long
// clip is measured in gigabytes. Callers that know their bound pass
// `max_seconds`; these are the backstop for the ones that do not.
constexpr int64_t kMaxVideoFrames          = 4096;
constexpr int64_t kMaxVideoPixelsPerFrame  = 4096LL * 4096LL;

string
av_err_(const FFmpegLibraries* libs, int rc)
{
  char buf[256] = {0};
  libs->avutil().api.strerror(rc, buf, sizeof buf);
  return string(buf);
}

void
set_err_(string* error, string msg)
{
  if (error) {
    *error = std::move(msg);
  }
}

// ---- RAII deleters (mirror load-image-stage's per-resource pattern) --

struct CctxDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(AVCodecContext* p) const noexcept
  {
    if (p) {
      libs->avcodec().api.free_context(&p);
    }
  }
};
using CctxPtr = unique_ptr<AVCodecContext, CctxDeleter>;

struct PktDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(AVPacket* p) const noexcept
  {
    if (p) {
      libs->avcodec().api.packet_free(&p);
    }
  }
};
using PktPtr = unique_ptr<AVPacket, PktDeleter>;

struct FrameDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(AVFrame* p) const noexcept
  {
    if (p) {
      libs->avutil().api.frame_free(&p);
    }
  }
};
using FramePtr = unique_ptr<AVFrame, FrameDeleter>;

struct SwsDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(SwsContext* p) const noexcept
  {
    if (p) {
      libs->swscale().api.free_context(p);
    }
  }
};
using SwsPtr = unique_ptr<SwsContext, SwsDeleter>;

struct SwrDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(SwrContext* p) const noexcept
  {
    if (p) {
      libs->swresample().api.free(&p);
    }
  }
};
using SwrPtr = unique_ptr<SwrContext, SwrDeleter>;

// ---- in-memory AVIO source -------------------------------------------

struct MemReader {
  const uint8_t* data = nullptr;
  size_t         size = 0;
  size_t         pos  = 0;
};

int
mem_read_(void* opaque, uint8_t* buf, int buf_size)
{
  auto* r = static_cast<MemReader*>(opaque);
  if (r->pos >= r->size) {
    return AVERROR_EOF;
  }
  const size_t n = std::min(static_cast<size_t>(buf_size),
                            r->size - r->pos);
  std::memcpy(buf, r->data + r->pos, n);
  r->pos += n;
  return static_cast<int>(n);
}

int64_t
mem_seek_(void* opaque, int64_t offset, int whence)
{
  auto* r = static_cast<MemReader*>(opaque);
  if (whence & AVSEEK_SIZE) {
    return static_cast<int64_t>(r->size);
  }
  int64_t base = 0;
  switch (whence & ~AVSEEK_FORCE) {
  case SEEK_SET: base = 0; break;
  case SEEK_CUR: base = static_cast<int64_t>(r->pos); break;
  case SEEK_END: base = static_cast<int64_t>(r->size); break;
  default:       return AVERROR(EINVAL);
  }
  const int64_t np = base + offset;
  if (np < 0 || np > static_cast<int64_t>(r->size)) {
    return AVERROR(EINVAL);
  }
  r->pos = static_cast<size_t>(np);
  return np;
}

// Demuxer handle over either a path/URL or an in-memory byte span.
// avformat_open_input with a preset custom pb sets AVFMT_FLAG_CUSTOM_IO,
// so close_input never frees the AVIOContext -- the dtor does (its
// buffer via av_freep first: FFmpeg may have reallocated it, so free
// whatever the context currently holds, not the original allocation).
struct OpenedInput {
  const FFmpegLibraries* libs = nullptr;
  AVFormatContext*       fctx = nullptr;
  AVIOContext*           avio = nullptr;
  unique_ptr<MemReader>  mem;

  ~OpenedInput()
  {
    if (fctx) {
      libs->avformat().api.close_input(&fctx);
    }
    if (avio) {
      libs->avutil().api.freep(&avio->buffer);
      libs->avformat().api.avio_context_free(&avio);
    }
  }
};

bool
open_input_(const FFmpegLibraries*  libs,
            const string*           path,
            span<const uint8_t>     bytes,
            OpenedInput*            in,
            string*                 error)
{
  in->libs = libs;
  int rc = 0;
  if (path) {
    rc = libs->avformat().api.open_input(&in->fctx, path->c_str(),
                                         nullptr, nullptr);
  } else {
    in->mem       = make_unique<MemReader>();
    in->mem->data = bytes.data();
    in->mem->size = bytes.size();
    constexpr int kIoBuf = 1 << 16;
    auto* iobuf = static_cast<uint8_t*>(
        libs->avutil().api.malloc(kIoBuf));
    if (!iobuf) {
      set_err_(error, "avio buffer alloc failed");
      return false;
    }
    in->avio = libs->avformat().api.avio_alloc_context(
        iobuf, kIoBuf, /*write_flag=*/0, in->mem.get(),
        &mem_read_, nullptr, &mem_seek_);
    if (!in->avio) {
      libs->avutil().api.freep(&iobuf);
      set_err_(error, "avio_alloc_context failed");
      return false;
    }
    in->fctx = libs->avformat().api.alloc_context();
    if (!in->fctx) {
      set_err_(error, "avformat_alloc_context failed");
      return false;
    }
    in->fctx->pb = in->avio;
    rc = libs->avformat().api.open_input(&in->fctx, nullptr,
                                         nullptr, nullptr);
  }
  if (rc < 0) {
    // On failure open_input freed the AVFormatContext and nulled it.
    set_err_(error, "open_input failed: " + av_err_(libs, rc));
    return false;
  }
  rc = libs->avformat().api.find_stream_info(in->fctx, nullptr);
  if (rc < 0) {
    set_err_(error, "find_stream_info failed: " + av_err_(libs, rc));
    return false;
  }
  return true;
}

int
find_stream_(const AVFormatContext* fctx, AVMediaType type)
{
  for (unsigned i = 0; i < fctx->nb_streams; ++i) {
    if (fctx->streams[i]->codecpar->codec_type == type) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

CctxPtr
open_decoder_(const FFmpegLibraries* libs,
              AVStream*              stream,
              string*                error)
{
  const AVCodec* codec =
      libs->avcodec().api.find_decoder(stream->codecpar->codec_id);
  if (!codec) {
    set_err_(error, "no decoder for codec_id "
        + to_string(static_cast<int>(stream->codecpar->codec_id)));
    return CctxPtr(nullptr, CctxDeleter{libs});
  }
  CctxPtr cctx(libs->avcodec().api.alloc_context3(codec),
               CctxDeleter{libs});
  if (!cctx) {
    set_err_(error, "avcodec_alloc_context3 failed");
    return cctx;
  }
  int rc = libs->avcodec().api.parameters_to_context(
      cctx.get(), stream->codecpar);
  if (rc < 0) {
    set_err_(error, "parameters_to_context failed: "
        + av_err_(libs, rc));
    cctx.reset();
    return CctxPtr(nullptr, CctxDeleter{libs});
  }
  rc = libs->avcodec().api.open2(cctx.get(), codec, nullptr);
  if (rc < 0) {
    set_err_(error, "avcodec_open2 failed: " + av_err_(libs, rc));
    cctx.reset();
    return CctxPtr(nullptr, CctxDeleter{libs});
  }
  return cctx;
}

// ---- image -----------------------------------------------------------

optional<DecodedImage>
decode_image_impl_(const FFmpegLibraries*  libs,
                   const string*           path,
                   span<const uint8_t>     bytes,
                   string*                 error)
{
  if (!libs || !libs->valid()) {
    set_err_(error, "FFmpeg libraries unavailable");
    return nullopt;
  }
  OpenedInput in;
  if (!open_input_(libs, path, bytes, &in, error)) {
    return nullopt;
  }
  const int v_idx = find_stream_(in.fctx, AVMEDIA_TYPE_VIDEO);
  if (v_idx < 0) {
    set_err_(error, "no video/image stream");
    return nullopt;
  }
  CctxPtr cctx = open_decoder_(libs, in.fctx->streams[v_idx], error);
  if (!cctx) {
    return nullopt;
  }
  PktPtr pkt(libs->avcodec().api.packet_alloc(), PktDeleter{libs});
  FramePtr frame(libs->avutil().api.frame_alloc(), FrameDeleter{libs});
  if (!pkt || !frame) {
    set_err_(error, "packet/frame alloc failed");
    return nullopt;
  }

  // Read packets until the decoder emits the first frame (stills are
  // one packet; tolerate a few non-video packets first).
  bool got_frame = false;
  bool flushed   = false;
  while (!got_frame) {
    int rc = libs->avformat().api.read_frame(in.fctx, pkt.get());
    if (rc < 0) {
      if (!flushed) {
        libs->avcodec().api.send_packet(cctx.get(), nullptr);
        flushed = true;
      }
    } else {
      if (pkt->stream_index != v_idx) {
        libs->avcodec().api.packet_unref(pkt.get());
        continue;
      }
      const int srp =
          libs->avcodec().api.send_packet(cctx.get(), pkt.get());
      libs->avcodec().api.packet_unref(pkt.get());
      if (srp < 0 && srp != AVERROR(EAGAIN)) {
        set_err_(error, "send_packet failed: " + av_err_(libs, srp));
        return nullopt;
      }
    }
    const int rrf =
        libs->avcodec().api.receive_frame(cctx.get(), frame.get());
    if (rrf == 0) {
      got_frame = true;
    } else if (rrf == AVERROR(EAGAIN)) {
      if (flushed) {
        set_err_(error, "no decoded frame produced");
        return nullopt;
      }
    } else {
      set_err_(error, "receive_frame failed: " + av_err_(libs, rrf));
      return nullopt;
    }
  }

  const int W = frame->width;
  const int H = frame->height;
  if (W <= 0 || H <= 0) {
    set_err_(error, "decoded frame has invalid dimensions");
    return nullopt;
  }
  // Single sws_scale: source pix_fmt -> planar GBRP, then repack to
  // contiguous channel order R,G,B (GBRP plane 2=R, 0=G, 1=B).
  SwsPtr sws(libs->swscale().api.get_context(
                 W, H, static_cast<AVPixelFormat>(frame->format),
                 W, H, AV_PIX_FMT_GBRP,
                 SWS_BILINEAR, nullptr, nullptr, nullptr),
             SwsDeleter{libs});
  if (!sws) {
    set_err_(error, "sws_getContext failed");
    return nullopt;
  }
  FramePtr gbrp(libs->avutil().api.frame_alloc(), FrameDeleter{libs});
  if (!gbrp) {
    set_err_(error, "frame_alloc(gbrp) failed");
    return nullopt;
  }
  gbrp->format = AV_PIX_FMT_GBRP;
  gbrp->width  = W;
  gbrp->height = H;
  int rc = libs->avutil().api.frame_get_buffer(gbrp.get(), 32);
  if (rc < 0) {
    set_err_(error, "frame_get_buffer failed: " + av_err_(libs, rc));
    return nullopt;
  }
  libs->swscale().api.scale(sws.get(), frame->data, frame->linesize,
                            0, H, gbrp->data, gbrp->linesize);

  DecodedImage out;
  out.width  = W;
  out.height = H;
  out.rgb.assign(static_cast<size_t>(3) * H * W, 0);
  const int src_plane_for_channel[3] = {2, 0, 1};
  for (int c = 0; c < 3; ++c) {
    const int      sp        = src_plane_for_channel[c];
    const uint8_t* src       = gbrp->data[sp];
    const int      src_pitch = gbrp->linesize[sp];
    uint8_t*       dst_plane =
        out.rgb.data() + static_cast<size_t>(c) * H * W;
    for (int y = 0; y < H; ++y) {
      std::memcpy(dst_plane + static_cast<size_t>(y) * W,
                  src + static_cast<size_t>(y) * src_pitch,
                  static_cast<size_t>(W));
    }
  }
  return out;
}

// ---- audio -----------------------------------------------------------

optional<DecodedAudio>
decode_audio_impl_(const FFmpegLibraries*  libs,
                   const string*           path,
                   span<const uint8_t>     bytes,
                   int                     target_sample_rate,
                   int                     out_channels,
                   string*                 error)
{
  if (!libs || !libs->valid()) {
    set_err_(error, "FFmpeg libraries unavailable");
    return nullopt;
  }
  if (out_channels != 1 && out_channels != 2) {
    set_err_(error, "channels must be 1 (mono) or 2 (stereo)");
    return nullopt;
  }
  if (target_sample_rate < 1000 || target_sample_rate > 384000) {
    set_err_(error, "target_sample_rate outside [1000, 384000]");
    return nullopt;
  }
  OpenedInput in;
  if (!open_input_(libs, path, bytes, &in, error)) {
    return nullopt;
  }
  const int a_idx = find_stream_(in.fctx, AVMEDIA_TYPE_AUDIO);
  if (a_idx < 0) {
    set_err_(error, "no audio stream");
    return nullopt;
  }
  CctxPtr cctx = open_decoder_(libs, in.fctx->streams[a_idx], error);
  if (!cctx) {
    return nullopt;
  }
  PktPtr pkt(libs->avcodec().api.packet_alloc(), PktDeleter{libs});
  FramePtr frame(libs->avutil().api.frame_alloc(), FrameDeleter{libs});
  if (!pkt || !frame) {
    set_err_(error, "packet/frame alloc failed");
    return nullopt;
  }

  DecodedAudio out;
  out.sample_rate = target_sample_rate;
  out.channels    = out_channels;
  const size_t max_samples =
      static_cast<size_t>(target_sample_rate) * kMaxAudioSeconds;
  // Accumulated INTERLEAVED and de-interleaved once at the end: the
  // planar form swresample would write needs one contiguous plane per
  // channel up front, which a stream of unknown length has not got.
  vector<float> inter;

  // Resampler to f32 @ target rate, built lazily on the first decoded
  // frame (the true sample format/layout may only be known then) and
  // rebuilt on a mid-stream format change.
  SwrPtr swr(nullptr, SwrDeleter{libs});
  int in_rate = 0, in_ch = 0;
  enum AVSampleFormat in_fmt = AV_SAMPLE_FMT_NONE;

  auto ensure_swr = [&](const AVFrame* f) -> bool {
    const auto ffmt = static_cast<enum AVSampleFormat>(f->format);
    if (swr && f->sample_rate == in_rate
        && f->ch_layout.nb_channels == in_ch && ffmt == in_fmt) {
      return true;
    }
    swr.reset();
    AVChannelLayout in_layout = AV_CHANNEL_LAYOUT_MONO;
    if (f->ch_layout.nb_channels == 2) {
      in_layout = AV_CHANNEL_LAYOUT_STEREO;
    } else if (f->ch_layout.nb_channels > 2) {
      in_layout.nb_channels = f->ch_layout.nb_channels;
    }
    AVChannelLayout out_layout = AV_CHANNEL_LAYOUT_MONO;
    if (out_channels == 2) {
      const AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
      out_layout = stereo;
    }
    SwrContext* sw = nullptr;
    int rc = libs->swresample().api.alloc_set_opts2(
        &sw,
        &out_layout, AV_SAMPLE_FMT_FLT, target_sample_rate,
        &in_layout,  ffmt,              f->sample_rate,
        0, nullptr);
    if (rc < 0 || !sw) {
      set_err_(error, "swr_alloc_set_opts2 failed");
      return false;
    }
    rc = libs->swresample().api.init(sw);
    if (rc < 0) {
      libs->swresample().api.free(&sw);
      set_err_(error, "swr_init failed: " + av_err_(libs, rc));
      return false;
    }
    swr.reset(sw);
    in_rate = f->sample_rate;
    in_ch   = f->ch_layout.nb_channels;
    in_fmt  = ffmt;
    return true;
  };

  auto convert_append = [&](const AVFrame* f) -> bool {
    // Upper bound: swr delay + rate-scaled input + slack.
    const int64_t delay =
        libs->swresample().api.get_delay(swr.get(), in_rate);
    const int64_t max_out = delay
        + (static_cast<int64_t>(f ? f->nb_samples : 0)
           * target_sample_rate) / std::max(1, in_rate)
        + 32;
    const size_t prev = inter.size();
    inter.resize(prev + static_cast<size_t>(max_out) * out_channels);
    float* dst = inter.data() + prev;
    const int n = libs->swresample().api.convert(
        swr.get(), reinterpret_cast<uint8_t**>(&dst),
        static_cast<int>(max_out),
        f ? const_cast<const uint8_t**>(f->extended_data) : nullptr,
        f ? f->nb_samples : 0);
    if (n < 0) {
      inter.resize(prev);
      set_err_(error, "swr_convert failed: " + av_err_(libs, n));
      return false;
    }
    inter.resize(prev + static_cast<size_t>(n) * out_channels);
    if (inter.size() / static_cast<size_t>(out_channels) > max_samples) {
      set_err_(error, "audio longer than the "
          + to_string(kMaxAudioSeconds) + " s decode cap");
      return false;
    }
    return true;
  };

  bool eof = false;
  while (!eof) {
    int rc = libs->avformat().api.read_frame(in.fctx, pkt.get());
    if (rc < 0) {
      libs->avcodec().api.send_packet(cctx.get(), nullptr);  // flush
      eof = true;
    } else {
      if (pkt->stream_index != a_idx) {
        libs->avcodec().api.packet_unref(pkt.get());
        continue;
      }
      const int srp =
          libs->avcodec().api.send_packet(cctx.get(), pkt.get());
      libs->avcodec().api.packet_unref(pkt.get());
      if (srp < 0 && srp != AVERROR(EAGAIN)) {
        set_err_(error, "send_packet failed: " + av_err_(libs, srp));
        return nullopt;
      }
    }
    while (true) {
      const int rrf =
          libs->avcodec().api.receive_frame(cctx.get(), frame.get());
      if (rrf == AVERROR(EAGAIN) || rrf == AVERROR_EOF) {
        break;
      }
      if (rrf < 0) {
        set_err_(error, "receive_frame failed: " + av_err_(libs, rrf));
        return nullopt;
      }
      const bool ok = ensure_swr(frame.get())
          && convert_append(frame.get());
      libs->avutil().api.frame_unref(frame.get());
      if (!ok) {
        return nullopt;
      }
    }
  }
  // Drain the resampler's tail (rate-conversion latency).
  if (swr) {
    if (!convert_append(nullptr)) {
      return nullopt;
    }
  }
  if (inter.empty()) {
    set_err_(error, "no audio samples decoded");
    return nullopt;
  }
  const size_t frames = inter.size() / static_cast<size_t>(out_channels);
  out.pcm.assign(inter.size(), 0.0f);
  for (int c = 0; c < out_channels; ++c) {
    float* dst = out.pcm.data() + static_cast<size_t>(c) * frames;
    for (size_t i = 0; i < frames; ++i) {
      dst[i] = inter[i * static_cast<size_t>(out_channels) + c];
    }
  }
  return out;
}

// ---- video -----------------------------------------------------------

optional<DecodedVideo>
decode_video_impl_(const FFmpegLibraries* libs,
                   const string&          path,
                   double                 max_seconds,
                   int                    audio_sample_rate,
                   string*                error)
{
  if (!libs || !libs->valid()) {
    set_err_(error, "FFmpeg libraries unavailable");
    return nullopt;
  }
  OpenedInput in;
  if (!open_input_(libs, &path, {}, &in, error)) { return nullopt; }
  const int v_idx = find_stream_(in.fctx, AVMEDIA_TYPE_VIDEO);
  if (v_idx < 0) {
    set_err_(error, "no video stream");
    return nullopt;
  }
  AVStream* vs = in.fctx->streams[v_idx];
  CctxPtr cctx = open_decoder_(libs, vs, error);
  if (!cctx) { return nullopt; }

  DecodedVideo out;
  // avg_frame_rate is what the container reports for the stream as a
  // whole; r_frame_rate is the finest tick it can express, which for a
  // variable-rate file is far above the real rate. Prefer the average
  // and fall back only when it is absent.
  const AVRational fr = (vs->avg_frame_rate.num > 0 &&
                         vs->avg_frame_rate.den > 0)
                            ? vs->avg_frame_rate
                            : vs->r_frame_rate;
  if (fr.num <= 0 || fr.den <= 0) {
    set_err_(error, "the container reports no frame rate");
    return nullopt;
  }
  out.fps = static_cast<double>(fr.num) / static_cast<double>(fr.den);

  // How many frames to read. `max_seconds` is a MEMORY bound, not a
  // nicety: planar RGB is 6 MB a frame at 1080p, so a caller that
  // truncates the clip afterwards would pay for the whole file first.
  int64_t frame_cap = kMaxVideoFrames;
  if (max_seconds > 0.0) {
    const double want = max_seconds * out.fps;
    // Two frames of slack: the rate resample downstream places output
    // slots by rounding, so the last one can fall just past the cut.
    frame_cap = min<int64_t>(frame_cap,
                             static_cast<int64_t>(want) + 2);
  }

  PktPtr pkt(libs->avcodec().api.packet_alloc(), PktDeleter{libs});
  FramePtr frame(libs->avutil().api.frame_alloc(), FrameDeleter{libs});
  FramePtr gbrp(libs->avutil().api.frame_alloc(), FrameDeleter{libs});
  if (!pkt || !frame || !gbrp) {
    set_err_(error, "packet/frame alloc failed");
    return nullopt;
  }
  SwsPtr sws(nullptr, SwsDeleter{libs});
  int sw_w = 0, sw_h = 0, sw_fmt = -1;

  // One decoded frame -> one planar [3,H,W] u8 plane appended to `rgb`.
  auto append_frame = [&](const AVFrame* f) -> bool {
    if (f->width <= 0 || f->height <= 0) {
      set_err_(error, "decoded frame has invalid dimensions");
      return false;
    }
    if (out.num_frames == 0) {
      out.width  = f->width;
      out.height = f->height;
      const int64_t px = static_cast<int64_t>(out.width) * out.height;
      if (px > kMaxVideoPixelsPerFrame) {
        set_err_(error, "frame is larger than the decode cap");
        return false;
      }
    } else if (f->width != out.width || f->height != out.height) {
      // A mid-stream resolution change would make `rgb` ragged, and
      // every consumer here indexes it by a fixed plane size.
      set_err_(error, "the video changes resolution mid-stream");
      return false;
    }
    const int W = out.width, H = out.height;
    if (!sws || sw_w != W || sw_h != H || sw_fmt != f->format) {
      sws.reset(libs->swscale().api.get_context(
          W, H, static_cast<AVPixelFormat>(f->format),
          W, H, AV_PIX_FMT_GBRP, SWS_BILINEAR, nullptr, nullptr,
          nullptr));
      if (!sws) {
        set_err_(error, "sws_getContext failed");
        return false;
      }
      sw_w = W; sw_h = H; sw_fmt = f->format;
      libs->avutil().api.frame_unref(gbrp.get());
      gbrp->format = AV_PIX_FMT_GBRP;
      gbrp->width  = W;
      gbrp->height = H;
      const int rc = libs->avutil().api.frame_get_buffer(gbrp.get(), 32);
      if (rc < 0) {
        set_err_(error, "frame_get_buffer failed: " + av_err_(libs, rc));
        return false;
      }
    }
    libs->swscale().api.scale(sws.get(), f->data, f->linesize, 0, H,
                              gbrp->data, gbrp->linesize);
    const size_t plane = static_cast<size_t>(3) * H * W;
    out.rgb.resize(plane * static_cast<size_t>(out.num_frames + 1));
    uint8_t* base = out.rgb.data() + plane * out.num_frames;
    const int src_plane_for_channel[3] = {2, 0, 1};   // GBRP -> R,G,B
    for (int c = 0; c < 3; ++c) {
      const int      sp    = src_plane_for_channel[c];
      const uint8_t* src   = gbrp->data[sp];
      const int      pitch = gbrp->linesize[sp];
      uint8_t*       dst   = base + static_cast<size_t>(c) * H * W;
      for (int y = 0; y < H; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * W,
                    src + static_cast<size_t>(y) * pitch,
                    static_cast<size_t>(W));
      }
    }
    ++out.num_frames;
    return true;
  };

  bool eof = false, stop = false;
  while (!eof && !stop) {
    int rc = libs->avformat().api.read_frame(in.fctx, pkt.get());
    if (rc < 0) {
      libs->avcodec().api.send_packet(cctx.get(), nullptr);   // flush
      eof = true;
    } else {
      if (pkt->stream_index != v_idx) {
        libs->avcodec().api.packet_unref(pkt.get());
        continue;
      }
      const int srp = libs->avcodec().api.send_packet(cctx.get(),
                                                      pkt.get());
      libs->avcodec().api.packet_unref(pkt.get());
      if (srp < 0 && srp != AVERROR(EAGAIN)) {
        set_err_(error, "send_packet failed: " + av_err_(libs, srp));
        return nullopt;
      }
    }
    while (true) {
      const int rrf = libs->avcodec().api.receive_frame(cctx.get(),
                                                        frame.get());
      if (rrf == AVERROR(EAGAIN) || rrf == AVERROR_EOF) { break; }
      if (rrf < 0) {
        set_err_(error, "receive_frame failed: " + av_err_(libs, rrf));
        return nullopt;
      }
      const bool ok = append_frame(frame.get());
      libs->avutil().api.frame_unref(frame.get());
      if (!ok) { return nullopt; }
      if (out.num_frames >= frame_cap) { stop = true; break; }
    }
  }
  if (out.num_frames == 0) {
    set_err_(error, "no frames decoded");
    return nullopt;
  }

  // The soundtrack, from the SAME file. A video reference conditions on
  // its own audio, so the two have to come out of one decode -- pairing
  // a clip with a separately-opened waveform is how they end up out of
  // sync.
  if (audio_sample_rate > 0 &&
      find_stream_(in.fctx, AVMEDIA_TYPE_AUDIO) >= 0) {
    string aerr;
    auto a = decode_audio_impl_(libs, &path, {}, audio_sample_rate, 2,
                                &aerr);
    if (a) {
      out.audio_channels    = a->channels;
      out.audio_sample_rate = a->sample_rate;
      out.pcm               = std::move(a->pcm);
    }
    // A soundtrack that fails to decode is NOT an error: the clip still
    // conditions on motion alone, which is a legitimate request.
  }
  return out;
}

}  // namespace

string_view
media_kind_name(MediaKind k) noexcept
{
  switch (k) {
    case MediaKind::Image: return "image";
    case MediaKind::Video: return "video";
    case MediaKind::Audio: return "audio";
    case MediaKind::Unknown: break;
  }
  return "unknown";
}

optional<MediaProbe>
probe_media_file(const FFmpegLibraries* libs, const string& path,
                 string* error)
{
  if (!libs || !libs->valid()) {
    set_err_(error, "FFmpeg libraries unavailable");
    return nullopt;
  }
  OpenedInput in;
  if (!open_input_(libs, &path, {}, &in, error)) { return nullopt; }

  MediaProbe p;
  const int v_idx = find_stream_(in.fctx, AVMEDIA_TYPE_VIDEO);
  const int a_idx = find_stream_(in.fctx, AVMEDIA_TYPE_AUDIO);
  p.has_video = v_idx >= 0;
  p.has_audio = a_idx >= 0;
  if (in.fctx->duration > 0) {
    p.duration_seconds = (double)in.fctx->duration / (double)AV_TIME_BASE;
  }
  if (v_idx >= 0) {
    const AVStream* vs = in.fctx->streams[v_idx];
    p.width  = vs->codecpar->width;
    p.height = vs->codecpar->height;
    p.nb_frames = (long long)vs->nb_frames;
    const AVRational fr = (vs->avg_frame_rate.num > 0 &&
                           vs->avg_frame_rate.den > 0)
                              ? vs->avg_frame_rate
                              : vs->r_frame_rate;
    if (fr.num > 0 && fr.den > 0) {
      p.fps = (double)fr.num / (double)fr.den;
    }
  }

  // A still image IS a one-frame video stream to FFmpeg, so the question
  // is never "is there video" but "does it move". Three signals, in
  // order of how much they can be trusted:
  //
  //   1. The DEMUXER. FFmpeg routes stills through dedicated image
  //      demuxers -- `image2`, `png_pipe`, `jpeg_pipe`, `webp_pipe`,
  //      ... -- and picking one is a statement about the file, not a
  //      guess from its name: it comes from probing the bytes.
  //   2. nb_frames == 1. A container that counted its frames and got 1.
  //   3. No rate and no duration. What is left when the container says
  //      nothing at all, which is the case for a raw single frame.
  //
  // The `_pipe` demuxers are the interesting case: they also serve
  // ANIMATED webp/gif, so the demuxer alone would call a short animation
  // a still. It is therefore only decisive when the frame count agrees
  // -- 0 (not counted) or 1 -- and a demuxer that reports 12 frames wins
  // over its own name.
  if (v_idx >= 0) {
    const char* fmt = (in.fctx->iformat && in.fctx->iformat->name)
                          ? in.fctx->iformat->name : "";
    const string fname(fmt);
    const bool image_demuxer =
        fname == "image2" || fname == "image2pipe" ||
        (fname.size() > 5 &&
         fname.compare(fname.size() - 5, 5, "_pipe") == 0);
    const bool says_one   = p.nb_frames == 1;
    const bool says_none  = p.nb_frames <= 0 && p.duration_seconds <= 0.0;
    p.kind = ((image_demuxer && p.nb_frames <= 1) || says_one || says_none)
                 ? MediaKind::Image
                 : MediaKind::Video;
    // A still with a soundtrack is not a thing; if the container really
    // carries audio, believe the audio and call it a video.
    if (p.kind == MediaKind::Image && p.has_audio) {
      p.kind = MediaKind::Video;
    }
  } else if (a_idx >= 0) {
    p.kind = MediaKind::Audio;
  }
  return p;
}

optional<DecodedImage>
decode_image_file(const FFmpegLibraries* libs,
                  const string&          path,
                  string*                error)
{
  return decode_image_impl_(libs, &path, {}, error);
}

optional<DecodedImage>
decode_image_bytes(const FFmpegLibraries*  libs,
                   span<const uint8_t>     bytes,
                   string*                 error)
{
  return decode_image_impl_(libs, nullptr, bytes, error);
}

optional<DecodedAudio>
decode_audio_file(const FFmpegLibraries* libs,
                  const string&          path,
                  int                    target_sample_rate,
                  string*                error,
                  int                    channels)
{
  return decode_audio_impl_(libs, &path, {}, target_sample_rate,
                            channels, error);
}

optional<DecodedAudio>
decode_audio_bytes(const FFmpegLibraries*  libs,
                   span<const uint8_t>     bytes,
                   int                     target_sample_rate,
                   string*                 error,
                   int                     channels)
{
  return decode_audio_impl_(libs, nullptr, bytes, target_sample_rate,
                            channels, error);
}

optional<DecodedVideo>
decode_video_file(const FFmpegLibraries* libs,
                  const string&          path,
                  double                 max_seconds,
                  int                    audio_sample_rate,
                  string*                error)
{
  return decode_video_impl_(libs, path, max_seconds, audio_sample_rate,
                            error);
}

}
