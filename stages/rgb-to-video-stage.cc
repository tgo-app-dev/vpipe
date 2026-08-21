#include "stages/rgb-to-video-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/audio-video/video-tokens.h"
#include "stages/model-provenance.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

const ConfigKey kAttrs[] = {
  {.key = "fps", .type = ConfigType::Real, .required = false,
   .doc = "frame rate to declare when the producer's sideband carries none. "
          "16 is the Wan video default",
   .def_real = 16.0},
  {.key = "pix_fmt", .type = ConfigType::String, .required = false,
   .doc = "pixel format of the emitted frames: \"yuv420p\" (what H.264 "
          "wants) or \"rgb24\" (no conversion, for a lossless codec)",
   .def_str = "yuv420p"},
};
const PortSpec kIports[] = {
  {.name = "image", .doc = "planar U8 RGB TensorBeat [3, H, W], one per frame "
                           "in presentation order",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "video", .doc = "one VideoStreamParams header then one FrameRef per "
                           "frame -- the contract save-video reads",
   .type = &typeid(FrameRefPayload),
   .tags = "video", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "rgb-to-video",
  .doc       = "Adapts planar U8 RGB image beats into the VideoStreamParams + "
               "FrameRef stream a save-video encoder reads. The seam between "
               "the generative image format and ffmpeg.",
  .display_name = "RGB to Video",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

// BT.601 studio-swing RGB -> YUV, which is what an SDR H.264 clip at this
// size is assumed to be by every player that will open it. Full-swing
// (JPEG) coefficients would look washed out through the same pipeline.
inline void
rgb_to_yuv_(float r, float g, float b, float& y, float& u, float& v)
{
  y = 16.0f  + 0.256788f * r + 0.504129f * g + 0.097906f * b;
  u = 128.0f - 0.148223f * r - 0.290993f * g + 0.439216f * b;
  v = 128.0f + 0.439216f * r - 0.367788f * g - 0.071427f * b;
}

inline std::uint8_t
clamp_u8_(float v)
{
  v = std::round(v);
  if (v < 0.0f) { return 0; }
  if (v > 255.0f) { return 255; }
  return (std::uint8_t)v;
}

}  // namespace

RgbToVideoStage::RgbToVideoStage(const SessionContextIntf* s,
                                 std::string               id,
                                 std::vector<InEdge>       iports,
                                 FlexData                  config)
  : TypedStage<RgbToVideoStage>(s, std::move(id), std::move(iports),
                                std::move(config))
  , _libs(s != nullptr && s->services() != nullptr
              ? s->services()->ffmpeg_libraries() : nullptr)
{
  _fps = attr_real("fps");
  if (!(_fps > 0.0)) { _fps = 16.0; }
  _pix_fmt_name = attr_str("pix_fmt");
  if (_pix_fmt_name.empty()) { _pix_fmt_name = "yuv420p"; }
  if (_pix_fmt_name == "yuv420p") {
    _pix_fmt = AV_PIX_FMT_YUV420P;
  } else if (_pix_fmt_name == "rgb24") {
    _pix_fmt = AV_PIX_FMT_RGB24;
  } else {
    // Deferred validation: the constructor never throws, so an unusable
    // pix_fmt is reported here and the runtime skips the stage at launch.
    fail_config(fmt("pix_fmt '{}' is not yuv420p or rgb24", _pix_fmt_name));
  }
  allocate_oports(spec().oports.size());
}

RgbToVideoStage::~RgbToVideoStage() = default;

void
RgbToVideoStage::reset_run_state()
{
  _frames = 0;
  _header_sent = false;
  _w = 0;
  _h = 0;
}

const StageSpec&
RgbToVideoStage::spec() const noexcept
{
  return kSpec;
}

Job
RgbToVideoStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }
  if (_libs == nullptr) {
    session()->warn(fmt(
        "RgbToVideoStage('{}'): no ffmpeg libraries; dropping the frame",
        this->id()));
    co_return;
  }
  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (tbp == nullptr || tbp->dtype != TensorBeat::DType::U8 ||
      tbp->shape.size() != 3 || tbp->shape[0] != 3) {
    session()->warn(fmt(
        "RgbToVideoStage('{}'): expected a planar U8 RGB [3,H,W] TensorBeat, "
        "got {}; skipping", this->id(), in->describe()));
    co_return;
  }
  const int H = (int)tbp->shape[1], W = (int)tbp->shape[2];
  if (H <= 0 || W <= 0) { co_return; }

  // The producer knows the clip's rate when it made the frames; the config
  // is only the fallback for one that does not.
  double fps = _fps;
  if (tbp->sideband.is_object()) {
    FlexData sb = tbp->sideband;          // as_object() is a view: keep it
    auto o = sb.as_object();
    if (o.contains("fps")) {
      const double f = o.at("fps").as_real(0.0);
      if (f > 0.0) { fps = f; }
    }
  }

  if (!_header_sent) {
    _w = W;
    _h = H;
    // yuv420p subsamples chroma by two in each axis, so an odd dimension
    // has no representation. Refuse rather than round: the encoder would
    // otherwise either fail deep inside libx264 or silently drop a line.
    if (_pix_fmt == AV_PIX_FMT_YUV420P && ((W & 1) != 0 || (H & 1) != 0)) {
      session()->error(fmt(
          "RgbToVideoStage('{}'): {}x{} is odd and yuv420p subsamples chroma "
          "2x2; use pix_fmt rgb24 or an even frame size", this->id(), W, H));
      co_return;
    }
    VideoStreamParams p;
    p.width = W;
    p.height = H;
    p.pix_fmt = _pix_fmt;
    // A rational rate from the double, denominator 1000, so 16 stays 16/1
    // and 23.976 survives as 23976/1000 rather than becoming 24.
    const long long num = (long long)std::llround(fps * 1000.0);
    p.frame_rate = AVRational{(int)num, 1000};
    p.time_base  = AVRational{1000, (int)num};
    auto hdr = std::make_unique<VideoStreamParamsPayload>(p);
    // Provenance rides on the HEADER because there is nowhere else for
    // it: a FrameRef is a bare AVFrame with no sideband, and the sink
    // needs the name before it writes the container header anyway.
    hdr->model_name = provenance::model_name(tbp->sideband);
    co_await ctx.write(0, std::move(hdr));
    _header_sent = true;
    session()->log_debug(fmt(
        "RgbToVideoStage('{}'): stream header {}x{} {} @ {:.3f} fps",
        this->id(), W, H, _pix_fmt_name, fps));
  } else if (W != _w || H != _h) {
    // A clip whose frame size changes partway is not something the encoder
    // can be handed -- the header is already out and describes the first
    // size. Dropping the odd frame keeps the stream valid and says so.
    session()->warn(fmt(
        "RgbToVideoStage('{}'): frame {} is {}x{} but the stream is {}x{}; "
        "dropping it", this->id(), _frames, W, H, _w, _h));
    co_return;
  }

  AVFrame* f = _libs->avutil().api.frame_alloc();
  if (f == nullptr) {
    session()->warn(fmt("RgbToVideoStage('{}'): frame_alloc failed",
                        this->id()));
    co_return;
  }
  f->format = _pix_fmt;
  f->width  = W;
  f->height = H;
  if (_libs->avutil().api.frame_get_buffer(f, 0) < 0) {
    _libs->avutil().api.frame_free(&f);
    session()->warn(fmt("RgbToVideoStage('{}'): frame_get_buffer failed",
                        this->id()));
    co_return;
  }
  f->pts = (std::int64_t)_frames;
  // One tick per frame, in the time base the header declares. Stating
  // it is what gives the LAST sample a length in the container: a
  // muxer infers every other frame's duration from the next frame's
  // timestamp and has nothing to infer the final one from.
  f->duration = 1;

  const std::uint8_t* rp = tbp->as_u8();
  const std::uint8_t* gp = rp + (std::size_t)H * W;
  const std::uint8_t* bp = gp + (std::size_t)H * W;
  if (_pix_fmt == AV_PIX_FMT_RGB24) {
    for (int y = 0; y < H; ++y) {
      std::uint8_t* row = f->data[0] + (std::size_t)y * f->linesize[0];
      for (int x = 0; x < W; ++x) {
        const std::size_t i = (std::size_t)y * W + x;
        row[3 * x + 0] = rp[i];
        row[3 * x + 1] = gp[i];
        row[3 * x + 2] = bp[i];
      }
    }
  } else {
    // Luma at full resolution; chroma averaged over each 2x2 block, which
    // is the box filter yuv420p's sample siting expects. Averaging in the
    // RGB domain and converting once per block would be cheaper and wrong
    // at edges, so the conversion runs per pixel and the AVERAGE is taken
    // in the chroma domain.
    for (int y = 0; y < H; ++y) {
      std::uint8_t* yr = f->data[0] + (std::size_t)y * f->linesize[0];
      for (int x = 0; x < W; ++x) {
        const std::size_t i = (std::size_t)y * W + x;
        float yy, uu, vv;
        rgb_to_yuv_((float)rp[i], (float)gp[i], (float)bp[i], yy, uu, vv);
        yr[x] = clamp_u8_(yy);
      }
    }
    for (int y = 0; y < H; y += 2) {
      std::uint8_t* ur = f->data[1] + (std::size_t)(y / 2) * f->linesize[1];
      std::uint8_t* vr = f->data[2] + (std::size_t)(y / 2) * f->linesize[2];
      for (int x = 0; x < W; x += 2) {
        float su = 0.0f, sv = 0.0f;
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            const std::size_t i = (std::size_t)(y + dy) * W + (x + dx);
            float yy, uu, vv;
            rgb_to_yuv_((float)rp[i], (float)gp[i], (float)bp[i], yy, uu, vv);
            su += uu;
            sv += vv;
          }
        }
        ur[x / 2] = clamp_u8_(su * 0.25f);
        vr[x / 2] = clamp_u8_(sv * 0.25f);
      }
    }
  }

  auto sp = FrameRef(f, [api = &_libs->avutil().api](AVFrame* x) {
    api->frame_free(&x);
  });
  ++_frames;
  co_await ctx.write(0, std::make_unique<FrameRefPayload>(std::move(sp)));
}

VPIPE_REGISTER_STAGE(RgbToVideoStage)
VPIPE_REGISTER_SPEC(RgbToVideoStage, kSpec)

}  // namespace vpipe
