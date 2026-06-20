#include "stages/load-image-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

// One small unique_ptr deleter per FFmpeg resource type, all carrying
// the FFmpegLibraries* so destruction routes through the right api
// table. The deleters live in the cc only because no other TU needs
// them.
struct FctxDeleter {
  const FFmpegLibraries* libs;
  void
  operator()(AVFormatContext* p) const noexcept
  {
    if (p) {
      libs->avformat().api.close_input(&p);
    }
  }
};
using FctxPtr = unique_ptr<AVFormatContext, FctxDeleter>;

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

}

LoadImageStage::LoadImageStage(const SessionContextIntf* s,
                               string                    id,
                               vector<InEdge>            iports,
                               FlexData                  config)
  : TypedStage<LoadImageStage>(s, std::move(id), std::move(iports),
                               std::move(config))
  , _libs(s ? s->ffmpeg_libraries() : nullptr)
{
  // Validation is deferred to launch (see Stage::fail_config): the
  // stage must construct for any config so a graph can be built/edited
  // before a url is supplied.
  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("url")) {
      FlexData u = root.at("url");
      if (u.is_string()) {
        _urls.emplace_back(string(u.as_string("")));
      } else if (u.is_array()) {
        for (FlexData e : u.as_array()) {
          if (e.is_string()) {
            _urls.emplace_back(string(e.as_string("")));
          } else {
            fail_config(fmt(
                "LoadImageStage('{}'): config.url array entries must be "
                "strings", this->id()));
          }
        }
      } else {
        fail_config(fmt(
            "LoadImageStage('{}'): config.url must be a string or array "
            "of strings", this->id()));
      }
    }
  }
  if (_urls.empty()) {
    fail_config(fmt(
        "LoadImageStage('{}'): config.url is required "
        "(non-empty string or array of strings)", this->id()));
  }
  for (const auto& url : _urls) {
    if (url.empty()) {
      fail_config(fmt(
          "LoadImageStage('{}'): config.url contains an empty entry",
          this->id()));
    }
  }
  if (!_libs || !_libs->valid()) {
    fail_config(fmt(
        "LoadImageStage('{}'): FFmpegLibraries unavailable",
        this->id()));
  }
  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "url", .type = ConfigType::Any, .required = true,
   .doc = "image path/URL string or array of strings"},
};
const PortSpec kIports[] = {
  {.name = "trigger", .doc = "optional pacing beat (e.g. chrono); each "
                             "one decodes + emits the next image",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "image", .doc = "decoded image as planar U8 RGB TensorBeat "
                           "[3,H,W]",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "load-image",
  .doc       = "Source: decodes still images from files/URLs (FFmpeg) to "
               "planar U8 RGB TensorBeats. With a wired trigger iport one "
               "beat emits one image; unwired it emits all then ends.",
  .display_name = "Load Image",
  .category  = StageCategory::Video,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
LoadImageStage::spec() const noexcept
{
  return kSpec;
}

LoadImageStage::~LoadImageStage() = default;

string
LoadImageStage::av_err_(int rc) const
{
  char buf[256];
  _libs->avutil().api.strerror(rc, buf, sizeof buf);
  return string(buf);
}

unique_ptr<BeatPayloadIntf>
LoadImageStage::decode_url_(const string& url) const
{
  AVFormatContext* raw_fctx = nullptr;
  AVDictionary*    opts     = nullptr;
  int rc = _libs->avformat().api.open_input(
      &raw_fctx, url.c_str(), nullptr, &opts);
  _libs->avutil().api.dict_free(&opts);
  if (rc < 0) {
    session()->warn(fmt(
        "LoadImageStage('{}'): open_input('{}') failed: {}",
        this->id(), url, av_err_(rc)));
    return nullptr;
  }
  FctxPtr fctx(raw_fctx, FctxDeleter{_libs});

  rc = _libs->avformat().api.find_stream_info(fctx.get(), nullptr);
  if (rc < 0) {
    session()->warn(fmt(
        "LoadImageStage('{}'): find_stream_info('{}') failed: {}",
        this->id(), url, av_err_(rc)));
    return nullptr;
  }

  int v_idx = -1;
  for (unsigned i = 0; i < fctx->nb_streams; ++i) {
    if (fctx->streams[i]->codecpar->codec_type
        == AVMEDIA_TYPE_VIDEO) {
      v_idx = static_cast<int>(i);
      break;
    }
  }
  if (v_idx < 0) {
    session()->warn(fmt(
        "LoadImageStage('{}'): no video/image stream in '{}'",
        this->id(), url));
    return nullptr;
  }

  AVStream*      stream = fctx->streams[v_idx];
  const AVCodec* codec  = _libs->avcodec().api.find_decoder(
      stream->codecpar->codec_id);
  if (!codec) {
    session()->warn(fmt(
        "LoadImageStage('{}'): no decoder for codec_id {} in '{}'",
        this->id(),
        static_cast<int>(stream->codecpar->codec_id), url));
    return nullptr;
  }
  AVCodecContext* raw_cctx =
      _libs->avcodec().api.alloc_context3(codec);
  if (!raw_cctx) {
    session()->warn(fmt(
        "LoadImageStage('{}'): avcodec_alloc_context3 failed",
        this->id()));
    return nullptr;
  }
  CctxPtr cctx(raw_cctx, CctxDeleter{_libs});

  rc = _libs->avcodec().api.parameters_to_context(cctx.get(),
                                                  stream->codecpar);
  if (rc < 0) {
    session()->warn(fmt(
        "LoadImageStage('{}'): parameters_to_context failed for "
        "'{}': {}", this->id(), url, av_err_(rc)));
    return nullptr;
  }
  rc = _libs->avcodec().api.open2(cctx.get(), codec, nullptr);
  if (rc < 0) {
    session()->warn(fmt(
        "LoadImageStage('{}'): avcodec_open2 failed for '{}': {}",
        this->id(), url, av_err_(rc)));
    return nullptr;
  }

  PktPtr pkt(_libs->avcodec().api.packet_alloc(),
             PktDeleter{_libs});
  if (!pkt) {
    session()->warn(fmt(
        "LoadImageStage('{}'): packet_alloc failed", this->id()));
    return nullptr;
  }
  FramePtr frame(_libs->avutil().api.frame_alloc(),
                 FrameDeleter{_libs});
  if (!frame) {
    session()->warn(fmt(
        "LoadImageStage('{}'): frame_alloc failed", this->id()));
    return nullptr;
  }

  // Read packets until the decoder emits a frame. For still images
  // (jpg/png/webp/...) one packet is typically enough; we also
  // gracefully handle short streams whose first frame comes after a
  // few non-video packets.
  bool got_frame = false;
  bool flushed   = false;
  while (!got_frame) {
    rc = _libs->avformat().api.read_frame(fctx.get(), pkt.get());
    if (rc == AVERROR_EOF || rc < 0) {
      if (!flushed) {
        _libs->avcodec().api.send_packet(cctx.get(), nullptr);
        flushed = true;
      }
    } else {
      if (pkt->stream_index != v_idx) {
        _libs->avcodec().api.packet_unref(pkt.get());
        continue;
      }
      int srp = _libs->avcodec().api.send_packet(cctx.get(),
                                                 pkt.get());
      _libs->avcodec().api.packet_unref(pkt.get());
      if (srp < 0 && srp != AVERROR(EAGAIN)) {
        session()->warn(fmt(
            "LoadImageStage('{}'): send_packet failed for '{}': {}",
            this->id(), url, av_err_(srp)));
        return nullptr;
      }
    }
    int rrf = _libs->avcodec().api.receive_frame(cctx.get(),
                                                 frame.get());
    if (rrf == 0) {
      got_frame = true;
    } else if (rrf == AVERROR(EAGAIN)) {
      if (flushed) {
        session()->warn(fmt(
            "LoadImageStage('{}'): no decoded frame produced for "
            "'{}'", this->id(), url));
        return nullptr;
      }
    } else if (rrf == AVERROR_EOF) {
      session()->warn(fmt(
          "LoadImageStage('{}'): decoder drained without a frame "
          "for '{}'", this->id(), url));
      return nullptr;
    } else {
      session()->warn(fmt(
          "LoadImageStage('{}'): receive_frame failed for '{}': {}",
          this->id(), url, av_err_(rrf)));
      return nullptr;
    }
  }

  const int W = frame->width;
  const int H = frame->height;
  if (W <= 0 || H <= 0) {
    session()->warn(fmt(
        "LoadImageStage('{}'): decoded frame from '{}' has invalid "
        "dimensions {}x{}", this->id(), url, W, H));
    return nullptr;
  }

  // Single sws_scale: arbitrary source pix_fmt → planar gbrp. The
  // bilinear kernel is plenty for stills; chroma fidelity matters
  // less than for video.
  SwsPtr sws(_libs->swscale().api.get_context(
                W, H, static_cast<AVPixelFormat>(frame->format),
                W, H, AV_PIX_FMT_GBRP,
                SWS_BILINEAR, nullptr, nullptr, nullptr),
             SwsDeleter{_libs});
  if (!sws) {
    session()->warn(fmt(
        "LoadImageStage('{}'): sws_getContext failed for '{}'",
        this->id(), url));
    return nullptr;
  }

  FramePtr gbrp(_libs->avutil().api.frame_alloc(),
                FrameDeleter{_libs});
  if (!gbrp) {
    session()->warn(fmt(
        "LoadImageStage('{}'): frame_alloc(gbrp) failed",
        this->id()));
    return nullptr;
  }
  gbrp->format = AV_PIX_FMT_GBRP;
  gbrp->width  = W;
  gbrp->height = H;
  rc = _libs->avutil().api.frame_get_buffer(gbrp.get(), 32);
  if (rc < 0) {
    session()->warn(fmt(
        "LoadImageStage('{}'): frame_get_buffer(gbrp) failed for "
        "'{}': {}", this->id(), url, av_err_(rc)));
    return nullptr;
  }
  _libs->swscale().api.scale(sws.get(),
                             frame->data, frame->linesize,
                             0, H,
                             gbrp->data, gbrp->linesize);

  // Pick a uniform per-row pitch P that fits every plane's linesize.
  // GBRP planes typically share linesize but FFmpeg can hand back
  // distinct values; round up to the wider so all three planes copy
  // through the same row stride.
  int P = gbrp->linesize[0];
  if (gbrp->linesize[1] > P) {
    P = gbrp->linesize[1];
  }
  if (gbrp->linesize[2] > P) {
    P = gbrp->linesize[2];
  }

  auto out = make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::U8;
  out->shape = {3, H, W};
  const size_t row_stride =
      (P == W) ? static_cast<size_t>(W) : static_cast<size_t>(P);
  if (P == W) {
    out->data.assign(static_cast<size_t>(3) * H * W, 0);
  } else {
    out->strides = {static_cast<int64_t>(H) * P, P, 1};
    out->data.assign(static_cast<size_t>(3) * H * P, 0);
  }

  // GBRP plane indices: 0=G, 1=B, 2=R. TensorBeat wants channels
  // ordered R, G, B (matches the rest of the apple-silicon pipeline).
  const int src_plane_for_channel[3] = {2, 0, 1};
  uint8_t* dst_base = out->as_u8();
  for (int c = 0; c < 3; ++c) {
    const int      sp        = src_plane_for_channel[c];
    const uint8_t* src       = gbrp->data[sp];
    const int      src_pitch = gbrp->linesize[sp];
    uint8_t*       dst_plane = dst_base
        + static_cast<size_t>(c) * H * row_stride;
    for (int y = 0; y < H; ++y) {
      std::memcpy(dst_plane + static_cast<size_t>(y) * row_stride,
                  src + static_cast<size_t>(y) * src_pitch,
                  static_cast<size_t>(W));
    }
  }
  return out;
}

Job
LoadImageStage::process(RuntimeContext& ctx)
{
  if (_next >= _urls.size()) {
    ctx.signal_done();
    co_return;
  }

  // Pacing: when an iport is wired, gate each emission on one beat
  // upstream (typically a chrono tick). The payload type doesn't
  // matter -- only the fact of receipt does. EOS upstream means we
  // stop emitting too.
  if (ctx.num_iports() >= 1) {
    auto t = co_await ctx.read(0);
    if (!t) {
      ctx.signal_done();
      co_return;
    }
  }

  const string url = _urls[_next++];
  auto payload = decode_url_(url);
  if (payload) {
    co_await ctx.write(0, std::move(payload));
  }

  // No iport => batch mode: signal done as soon as we've emitted (or
  // tried to emit) the last URL so the driver closes our oport. With
  // an iport wired the driver tears us down when the upstream source
  // EOSes; we never self-signal in that case.
  if (_next >= _urls.size() && ctx.num_iports() == 0) {
    ctx.signal_done();
  }
}

VPIPE_REGISTER_STAGE(LoadImageStage)
VPIPE_REGISTER_SPEC(LoadImageStage, kSpec)

}
