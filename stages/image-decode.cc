#include "stages/image-decode.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace vpipe {

namespace {

// One small unique_ptr deleter per FFmpeg resource type, all carrying
// the FFmpegLibraries* so destruction routes through the right api table.
struct FctxDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(AVFormatContext* p) const noexcept
  {
    if (p) { libs->avformat().api.close_input(&p); }
  }
};
using FctxPtr = std::unique_ptr<AVFormatContext, FctxDeleter>;

struct CctxDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(AVCodecContext* p) const noexcept
  {
    if (p) { libs->avcodec().api.free_context(&p); }
  }
};
using CctxPtr = std::unique_ptr<AVCodecContext, CctxDeleter>;

struct PktDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(AVPacket* p) const noexcept
  {
    if (p) { libs->avcodec().api.packet_free(&p); }
  }
};
using PktPtr = std::unique_ptr<AVPacket, PktDeleter>;

struct FrameDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(AVFrame* p) const noexcept
  {
    if (p) { libs->avutil().api.frame_free(&p); }
  }
};
using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;

struct SwsDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(SwsContext* p) const noexcept
  {
    if (p) { libs->swscale().api.free_context(p); }
  }
};
using SwsPtr = std::unique_ptr<SwsContext, SwsDeleter>;

std::string
av_err_(const FFmpegLibraries* libs, int rc)
{
  char buf[256];
  libs->avutil().api.strerror(rc, buf, sizeof buf);
  return std::string(buf);
}

bool
fail_(std::string* err, const std::string& m)
{
  if (err != nullptr) { *err = m; }
  return false;
}

// Open `url` and find its first video (image) stream.
bool
open_(const FFmpegLibraries* libs, const std::string& url, FctxPtr* fctx,
      int* v_idx, std::string* err)
{
  AVFormatContext* raw = nullptr;
  AVDictionary* opts = nullptr;
  int rc = libs->avformat().api.open_input(&raw, url.c_str(), nullptr, &opts);
  libs->avutil().api.dict_free(&opts);
  if (rc < 0) {
    return fail_(err, "open_input('" + url + "') failed: " +
                          av_err_(libs, rc));
  }
  *fctx = FctxPtr(raw, FctxDeleter{libs});
  rc = libs->avformat().api.find_stream_info(fctx->get(), nullptr);
  if (rc < 0) {
    return fail_(err, "find_stream_info('" + url + "') failed: " +
                          av_err_(libs, rc));
  }
  *v_idx = -1;
  for (unsigned i = 0; i < (*fctx)->nb_streams; ++i) {
    if ((*fctx)->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      *v_idx = (int)i;
      break;
    }
  }
  if (*v_idx < 0) {
    return fail_(err, "no video/image stream in '" + url + "'");
  }
  return true;
}

}  // namespace

bool
probe_image_size(const FFmpegLibraries* libs, const std::string& url,
                 int* w, int* h, std::string* err)
{
  if (libs == nullptr || !libs->valid()) {
    return fail_(err, "FFmpeg is not available");
  }
  FctxPtr fctx(nullptr, FctxDeleter{libs});
  int v_idx = -1;
  if (!open_(libs, url, &fctx, &v_idx, err)) { return false; }
  const AVCodecParameters* cp = fctx->streams[v_idx]->codecpar;
  *w = cp->width;
  *h = cp->height;
  if (*w <= 0 || *h <= 0) {
    return fail_(err, "'" + url + "' has no stated size");
  }
  return true;
}

std::unique_ptr<TensorBeatPayload>
decode_image_file(const FFmpegLibraries* libs, const std::string& url,
                  const ImageDecodeOptions& opt, std::string* err)
{
  if (libs == nullptr || !libs->valid()) {
    fail_(err, "FFmpeg is not available");
    return nullptr;
  }
  FctxPtr fctx(nullptr, FctxDeleter{libs});
  int v_idx = -1;
  if (!open_(libs, url, &fctx, &v_idx, err)) { return nullptr; }

  AVStream* stream = fctx->streams[v_idx];
  const AVCodec* codec =
      libs->avcodec().api.find_decoder(stream->codecpar->codec_id);
  if (!codec) {
    fail_(err, "no decoder for codec_id " +
                   std::to_string((int)stream->codecpar->codec_id) +
                   " in '" + url + "'");
    return nullptr;
  }
  AVCodecContext* raw_cctx = libs->avcodec().api.alloc_context3(codec);
  if (!raw_cctx) {
    fail_(err, "avcodec_alloc_context3 failed");
    return nullptr;
  }
  CctxPtr cctx(raw_cctx, CctxDeleter{libs});
  int rc = libs->avcodec().api.parameters_to_context(cctx.get(),
                                                     stream->codecpar);
  if (rc < 0) {
    fail_(err, "parameters_to_context failed for '" + url + "': " +
                   av_err_(libs, rc));
    return nullptr;
  }
  rc = libs->avcodec().api.open2(cctx.get(), codec, nullptr);
  if (rc < 0) {
    fail_(err, "avcodec_open2 failed for '" + url + "': " +
                   av_err_(libs, rc));
    return nullptr;
  }

  PktPtr pkt(libs->avcodec().api.packet_alloc(), PktDeleter{libs});
  FramePtr frame(libs->avutil().api.frame_alloc(), FrameDeleter{libs});
  if (!pkt || !frame) {
    fail_(err, "packet/frame allocation failed");
    return nullptr;
  }

  // Read packets until the decoder emits a frame. For still images
  // (jpg/png/webp/...) one packet is typically enough; short streams
  // whose first frame follows a few non-video packets work too.
  bool got_frame = false;
  bool flushed = false;
  while (!got_frame) {
    rc = libs->avformat().api.read_frame(fctx.get(), pkt.get());
    if (rc == AVERROR_EOF || rc < 0) {
      if (!flushed) {
        libs->avcodec().api.send_packet(cctx.get(), nullptr);
        flushed = true;
      }
    } else {
      if (pkt->stream_index != v_idx) {
        libs->avcodec().api.packet_unref(pkt.get());
        continue;
      }
      const int srp = libs->avcodec().api.send_packet(cctx.get(), pkt.get());
      libs->avcodec().api.packet_unref(pkt.get());
      if (srp < 0 && srp != AVERROR(EAGAIN)) {
        fail_(err, "send_packet failed for '" + url + "': " +
                       av_err_(libs, srp));
        return nullptr;
      }
    }
    const int rrf = libs->avcodec().api.receive_frame(cctx.get(),
                                                      frame.get());
    if (rrf == 0) {
      got_frame = true;
    } else if (rrf == AVERROR(EAGAIN)) {
      if (flushed) {
        fail_(err, "no decoded frame produced for '" + url + "'");
        return nullptr;
      }
    } else if (rrf == AVERROR_EOF) {
      fail_(err, "decoder drained without a frame for '" + url + "'");
      return nullptr;
    } else {
      fail_(err, "receive_frame failed for '" + url + "': " +
                     av_err_(libs, rrf));
      return nullptr;
    }
  }

  const int W = frame->width;
  const int H = frame->height;
  if (W <= 0 || H <= 0) {
    fail_(err, "decoded frame from '" + url + "' has invalid dimensions");
    return nullptr;
  }

  // The scaled size: the decoded one, or the smallest that COVERS the
  // target with the aspect kept, cropped to it below.
  int SW = W, SH = H, OW = W, OH = H;
  if (opt.out_w > 0 && opt.out_h > 0) {
    OW = opt.out_w;
    OH = opt.out_h;
    const double s = std::max((double)OW / W, (double)OH / H);
    SW = std::max(OW, (int)std::lround(W * s));
    SH = std::max(OH, (int)std::lround(H * s));
  }
  const bool resize = SW != W || SH != H;

  // Single sws_scale: arbitrary source pix_fmt -> planar gbrp (gbrap with
  // alpha). Bilinear at the decoded size, as load-image always did; an
  // area filter when shrinking to a bucket, which is what keeps fine
  // texture from aliasing, and Lanczos when enlarging.
  const int CH = opt.keep_alpha ? 4 : 3;
  const AVPixelFormat planar =
      opt.keep_alpha ? AV_PIX_FMT_GBRAP : AV_PIX_FMT_GBRP;
  int filter = SWS_BILINEAR;
  if (resize) { filter = (SW < W || SH < H) ? SWS_AREA : SWS_LANCZOS; }
  SwsPtr sws(libs->swscale().api.get_context(
                 W, H, (AVPixelFormat)frame->format, SW, SH, planar, filter,
                 nullptr, nullptr, nullptr),
             SwsDeleter{libs});
  if (!sws) {
    fail_(err, "sws_getContext failed for '" + url + "'");
    return nullptr;
  }
  FramePtr gbrp(libs->avutil().api.frame_alloc(), FrameDeleter{libs});
  if (!gbrp) {
    fail_(err, "frame_alloc(gbrp) failed");
    return nullptr;
  }
  gbrp->format = planar;
  gbrp->width = SW;
  gbrp->height = SH;
  rc = libs->avutil().api.frame_get_buffer(gbrp.get(), 32);
  if (rc < 0) {
    fail_(err, "frame_get_buffer(gbrp) failed for '" + url + "': " +
                   av_err_(libs, rc));
    return nullptr;
  }
  libs->swscale().api.scale(sws.get(), frame->data, frame->linesize, 0, H,
                            gbrp->data, gbrp->linesize);

  // Pick a uniform per-row pitch P that fits every plane's linesize, as
  // load-image always has: planes can come back with distinct linesizes.
  // With a crop the output is packed at its own width instead.
  const int x0 = (SW - OW) / 2, y0 = (SH - OH) / 2;
  int P = gbrp->linesize[0];
  for (int i = 1; i < CH; ++i) { P = std::max(P, gbrp->linesize[i]); }
  const bool packed = resize || opt.flip || P == OW;
  const std::size_t row_stride = packed ? (std::size_t)OW : (std::size_t)P;

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::U8;
  out->shape = {CH, OH, OW};
  if (!packed) {
    out->strides = {(std::int64_t)OH * P, P, 1};
  }
  out->data.assign((std::size_t)CH * OH * row_stride, 0);

  // GBRP plane indices: 0=G, 1=B, 2=R -- and GBRAP appends A as plane 3.
  // TensorBeat wants channels ordered R, G, B(, A).
  const int src_plane_for_channel[4] = {2, 0, 1, 3};
  std::uint8_t* dst_base = out->as_u8();
  for (int c = 0; c < CH; ++c) {
    const int sp = src_plane_for_channel[c];
    const std::uint8_t* src = gbrp->data[sp];
    const int src_pitch = gbrp->linesize[sp];
    std::uint8_t* dst_plane = dst_base + (std::size_t)c * OH * row_stride;
    for (int y = 0; y < OH; ++y) {
      const std::uint8_t* s =
          src + (std::size_t)(y + y0) * src_pitch + (std::size_t)x0;
      std::uint8_t* d = dst_plane + (std::size_t)y * row_stride;
      if (opt.flip) {
        for (int x = 0; x < OW; ++x) { d[x] = s[OW - 1 - x]; }
      } else {
        std::memcpy(d, s, (std::size_t)OW);
      }
    }
  }
  return out;
}

}  // namespace vpipe
