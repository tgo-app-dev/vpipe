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

// The CLOSED SETS, for the editor's dropdown. Every one of these keys is
// already closed in resolve_config_(), which fails the stage on anything
// else and stays the authority -- what these add is that the editor
// offers the values instead of leaving the user to find them in the doc
// and then discover a typo at launch.
constexpr SpecExtra kPixFmtChoices[] = {
  {"choices", "yuv420p,yuv422p,yuv444p,rgb24"},
};
constexpr SpecExtra kColorRangeChoices[] = {
  {"choices", "limited,full"},
};
constexpr SpecExtra kColorspaceChoices[] = {
  {"choices", "bt601,bt709"},
};

const ConfigKey kAttrs[] = {
  {.key = "fps", .type = ConfigType::Real, .required = false,
   .doc = "frame rate to declare when the producer's sideband carries none. "
          "16 is the Wan video default",
   .def_real = 16.0},
  {.key = "pix_fmt", .type = ConfigType::String, .required = false,
   .doc = "pixel format of the emitted frames. \"yuv420p\" (the "
          "default) is what an SDR H.264 clip is and what every encoder "
          "takes. \"yuv422p\" and \"yuv444p\" keep more chroma -- 444 "
          "keeps all of it, so an RGB round trip through this stage and "
          "back is limited only by the codec -- which is worth having on "
          "an INTERMEDIATE this graph will read back itself, and costs "
          "bitrate on a delivery file. \"rgb24\" does no conversion at "
          "all, for a lossless codec. CHECK THE ENCODER TAKES IT: "
          "h264_videotoolbox, the default, is yuv420p ONLY, so 422 and "
          "444 need video_codec set to something that accepts them "
          "(libx264, ffv1). A subsampled axis may not be odd -- 420 "
          "needs both even, 422 the width -- and this stage refuses "
          "rather than rounding",
   .def_str = "yuv420p", .extra = kPixFmtChoices},
  {.key = "color_range", .type = ConfigType::String, .required = false,
   .doc = "the YUV swing this stage writes, and TAGS the file with. "
          "\"limited\" (the default) is 16..235 studio swing, what every "
          "player assumes of an SDR H.264 clip and therefore the safe "
          "delivery choice. \"full\" is 0..255: it keeps all 256 code "
          "values instead of 219, which is worth having on an "
          "INTERMEDIATE this graph will read back itself -- a multi-part "
          "video chain handing each clip to the next -- but a player that "
          "ignores the range tag renders it with crushed blacks. Either "
          "way the file says which it is, so vpipe reads it back "
          "unchanged; the choice is about what ELSE opens it",
   .def_str = "limited", .extra = kColorRangeChoices},
  {.key = "colorspace", .type = ConfigType::String, .required = false,
   .doc = "the YUV MATRIX this stage writes, and TAGS the file with. "
          "\"bt601\" (the default) is what this stage has always "
          "emitted and what a player assumes of a clip that does not "
          "say; \"bt709\" is the correct matrix for HD delivery and "
          "what a player assumes of HD when the file is silent. It is a "
          "different question from color_range -- the matrix tilts HUE "
          "and saturation where the range moves CONTRAST -- and the two "
          "errors do not cancel. Changing it changes the pixels, not "
          "just the tag: the file says which matrix it is either way, so "
          "vpipe reads it back unchanged whichever you pick",
   .def_str = "bt601", .extra = kColorspaceChoices},
};
const PortSpec kIports[] = {
  {.name = "image", .doc = "planar U8 RGB [3, H, W] or RGBA [4, H, W] "
                           "TensorBeat, one per frame in presentation "
                           "order. No video pixel format here carries "
                           "alpha, so a fourth plane is composited over "
                           "white",
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
  .doc       = "Adapts planar U8 RGB or RGBA image beats into the "
               "VideoStreamParams + FrameRef stream a save-video encoder "
               "reads. The seam between the generative image format and "
               "ffmpeg. An RGBA frame is flattened onto white, since "
               "neither yuv420p nor rgb24 carries alpha; put an alpha-mix "
               "ahead of this stage to composite over anything else.",
  .display_name = "RGB to Video",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

// RGB -> YUV, for either matrix at either swing.
//
// DERIVED FROM THE TWO LUMA COEFFICIENTS, not written out four times.
// BT.601 and BT.709 differ only in Kr/Kb, and the swing only in two
// scale factors, so a table of four hand-typed matrices would be three
// chances to transpose a digit into a hue tilt nothing reports. The
// arithmetic below reproduces the published BT.601 constants exactly --
// which is asserted in rgb_to_video_colorspace, so a change here that
// moves the numbers is a failing test rather than a quietly different
// file.
//
//   Y'  = Kr*R' + Kg*G' + Kb*B'                  (Kg = 1 - Kr - Kb)
//   Cb  = (B' - Y') / (2 * (1 - Kb))             in [-0.5, +0.5]
//   Cr  = (R' - Y') / (2 * (1 - Kr))
//
// Studio swing puts Y' in 16..235 (219 codes) and the chroma in 16..240
// (224); full swing uses all 256 for both.
struct YuvMatrix {
  float y0, yr, yg, yb;
  float ur, ug, ub;
  float vr, vg, vb;
};

inline YuvMatrix
yuv_matrix_(float kr, float kb, bool full)
{
  const float kg = 1.0f - kr - kb;
  const float ys = full ? 1.0f : 219.0f / 255.0f;
  // 255 (full) or 224 (studio) code values spread over the [-0.5, +0.5]
  // of Cb/Cr, and the /255 that takes R,G,B from bytes to [0,1].
  const float cs = (full ? 255.0f : 224.0f) / (2.0f * 255.0f);
  YuvMatrix m{};
  m.y0 = full ? 0.0f : 16.0f;
  m.yr = ys * kr;  m.yg = ys * kg;  m.yb = ys * kb;
  const float cb = cs / (1.0f - kb);
  m.ur = -cb * kr; m.ug = -cb * kg; m.ub =  cb * (1.0f - kb);
  const float cr = cs / (1.0f - kr);
  m.vr =  cr * (1.0f - kr); m.vg = -cr * kg; m.vb = -cr * kb;
  return m;
}

// The two matrices this stage can write. BT.601 is the default because
// it is what every player assumes of an SDR clip that does not say, and
// what this stage has always emitted; BT.709 is the correct one for HD
// delivery and is what a player assumes of HD when the file is silent.
// Either way the file is TAGGED, so the choice only decides what a
// reader that ignores the tag makes of it.
inline YuvMatrix bt601_matrix_(bool full) {
  return yuv_matrix_(0.299f, 0.114f, full);
}
inline YuvMatrix bt709_matrix_(bool full) {
  return yuv_matrix_(0.2126f, 0.0722f, full);
}

inline void
rgb_to_yuv_(const YuvMatrix& m, float r, float g, float b,
            float& y, float& u, float& v)
{
  y = m.y0  + m.yr * r + m.yg * g + m.yb * b;
  u = 128.0f + m.ur * r + m.ug * g + m.ub * b;
  v = 128.0f + m.vr * r + m.vg * g + m.vb * b;
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
  const std::string cs = attr_str("colorspace");
  if (cs == "bt709") {
    _bt709 = true;
  } else if (cs == "bt601" || cs.empty()) {
    _bt709 = false;
  } else {
    fail_config(fmt("RgbToVideoStage('{}'): colorspace '{}' is neither "
                    "'bt601' nor 'bt709'", this->id(), cs));
  }
  const std::string cr = attr_str("color_range");
  _full_range = (cr == "full");
  if (!cr.empty() && cr != "full" && cr != "limited") {
    fail_config(fmt("RgbToVideoStage('{}'): color_range '{}' is neither "
                    "\"limited\" nor \"full\"", this->id(), cr));
  }
  if (_pix_fmt_name == "yuv420p") {
    _pix_fmt = AV_PIX_FMT_YUV420P;
    _chroma_x = 2; _chroma_y = 2;
  } else if (_pix_fmt_name == "yuv422p") {
    _pix_fmt = AV_PIX_FMT_YUV422P;
    _chroma_x = 2; _chroma_y = 1;
  } else if (_pix_fmt_name == "yuv444p") {
    _pix_fmt = AV_PIX_FMT_YUV444P;
    _chroma_x = 1; _chroma_y = 1;
  } else if (_pix_fmt_name == "rgb24") {
    _pix_fmt = AV_PIX_FMT_RGB24;
    _chroma_x = 1; _chroma_y = 1;      // unused; no chroma plane
  } else {
    // Deferred validation: the constructor never throws, so an unusable
    // pix_fmt is reported here and the runtime skips the stage at launch.
    fail_config(fmt("pix_fmt '{}' is not yuv420p, yuv422p, yuv444p or "
                    "rgb24", _pix_fmt_name));
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
      tbp->shape.size() != 3 ||
      (tbp->shape[0] != 3 && tbp->shape[0] != 4)) {
    session()->warn(fmt(
        "RgbToVideoStage('{}'): expected a planar U8 RGB [3,H,W] or RGBA "
        "[4,H,W] TensorBeat, got {}; skipping",
        this->id(), in->describe()));
    co_return;
  }
  const int CH = (int)tbp->shape[0];
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
    // A SUBSAMPLED AXIS HAS NO ODD SIZE. Refuse rather than round: the
    // encoder would otherwise either fail deep inside libx264 or
    // silently drop a line. Which axes are subsampled is the format's,
    // so the check is too -- 4:2:0 needs both even, 4:2:2 only the
    // width, 4:4:4 and rgb24 neither, and a check written for 4:2:0
    // alone would refuse a perfectly good odd-height 4:2:2 frame.
    if ((_chroma_x > 1 && (W % _chroma_x) != 0)
        || (_chroma_y > 1 && (H % _chroma_y) != 0)) {
      session()->error(fmt(
          "RgbToVideoStage('{}'): {}x{} does not divide {}'s {}x{} chroma "
          "subsampling; use an even {} or a pix_fmt that subsamples less "
          "({} or rgb24)", this->id(), W, H, _pix_fmt_name, _chroma_x,
          _chroma_y,
          (_chroma_x > 1 && (W % _chroma_x) != 0) ? "width" : "height",
          _chroma_y > 1 ? "yuv422p, yuv444p" : "yuv444p"));
      co_return;
    }
    VideoStreamParams p;
    p.width = W;
    p.height = H;
    p.pix_fmt = _pix_fmt;
    // DECLARE what rgb_to_yuv_ actually wrote, so the sink can tag the
    // file and no reader has to fall back on convention. Saying so is
    // what lets a decode come back with the contrast and hue it went in
    // with -- and it is the config's two keys that decide it, never an
    // assumption about the size.
    //
    // PRIMARIES and TRANSFER travel too. They were left UNSPECIFIED,
    // which is a smaller error than a wrong matrix and the same kind: a
    // reader falls back on convention, and the convention for an
    // untagged HD file is BT.709 whatever the matrix says.
    if (_pix_fmt != AV_PIX_FMT_RGB24) {
      p.color_range = _full_range ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
      p.colorspace  = _bt709 ? AVCOL_SPC_BT709 : AVCOL_SPC_SMPTE170M;
      p.color_primaries = _bt709 ? AVCOL_PRI_BT709 : AVCOL_PRI_SMPTE170M;
      p.color_trc   = _bt709 ? AVCOL_TRC_BT709 : AVCOL_TRC_SMPTE170M;
    }
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

  // A PADDED ROW PITCH shears every row here, because the reads below
  // index the planes as tight -- and load-image forwards FFmpeg's
  // 32-byte-aligned GBRP linesize, so a width that is not a multiple of
  // 32 arrives strided. Materialise when it is not contiguous.
  AlignedVector<std::uint8_t> contig;
  const std::uint8_t* src = tbp->as_u8();
  if (!tbp->is_contiguous()) {
    contig = tbp->materialize_contiguous();
    src = contig.data();
  }
  const std::size_t plane = (std::size_t)H * W;

  // COMPOSITE OVER WHITE, since neither yuv420p nor rgb24 carries alpha.
  // Truncating the fourth plane is NOT the neutral choice: under a
  // transparent pixel the stored colour is arbitrary, and on a generated
  // RGBA frame it is usually black -- so dropping the plane fringes every
  // edge in black, which reads as a bad model rather than a lost channel.
  // out = a*rgb + (1-a)*255, the same arithmetic save-image flattens a
  // transparent PNG with. Put an `alpha-mix` ahead of this stage to
  // composite over any other colour, or over another clip.
  std::vector<std::uint8_t> flat;
  if (CH == 4) {
    flat.resize(3 * plane);
    const std::uint8_t* ap = src + 3 * plane;
    for (int c = 0; c < 3; ++c) {
      const std::uint8_t* sp = src + (std::size_t)c * plane;
      std::uint8_t* dp = flat.data() + (std::size_t)c * plane;
      for (std::size_t i = 0; i < plane; ++i) {
        const int a = ap[i];
        dp[i] = (std::uint8_t)((sp[i] * a + 255 * (255 - a) + 127) / 255);
      }
    }
    src = flat.data();
  }

  const std::uint8_t* rp = src;
  const std::uint8_t* gp = rp + plane;
  const std::uint8_t* bp = gp + plane;
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
    // Once per frame, not once per pixel: the loops below run millions
    // of times and have no business re-deciding which matrix this is.
    const YuvMatrix mat =
        _bt709 ? bt709_matrix_(_full_range) : bt601_matrix_(_full_range);
    // Luma at full resolution; chroma averaged over each subsampling
    // block -- 2x2 for 4:2:0, 2x1 for 4:2:2, 1x1 (i.e. no averaging) for
    // 4:4:4 -- which is the box filter those formats' sample siting
    // expects. Averaging in the RGB domain and converting once per block
    // would be cheaper and wrong at edges, so the conversion runs per
    // pixel and the AVERAGE is taken in the chroma domain.
    for (int y = 0; y < H; ++y) {
      std::uint8_t* yr = f->data[0] + (std::size_t)y * f->linesize[0];
      for (int x = 0; x < W; ++x) {
        const std::size_t i = (std::size_t)y * W + x;
        float yy, uu, vv;
        rgb_to_yuv_(mat, (float)rp[i], (float)gp[i], (float)bp[i],
                    yy, uu, vv);
        yr[x] = clamp_u8_(yy);
      }
    }
    const int hx = _chroma_x, hy = _chroma_y;
    const float inv = 1.0f / (float)(hx * hy);
    for (int y = 0; y < H; y += hy) {
      std::uint8_t* ur = f->data[1] + (std::size_t)(y / hy) * f->linesize[1];
      std::uint8_t* vr = f->data[2] + (std::size_t)(y / hy) * f->linesize[2];
      for (int x = 0; x < W; x += hx) {
        float su = 0.0f, sv = 0.0f;
        for (int dy = 0; dy < hy; ++dy) {
          for (int dx = 0; dx < hx; ++dx) {
            const std::size_t i = (std::size_t)(y + dy) * W + (x + dx);
            float yy, uu, vv;
            rgb_to_yuv_(mat, (float)rp[i], (float)gp[i], (float)bp[i],
                        yy, uu, vv);
            su += uu;
            sv += vv;
          }
        }
        // At 4:4:4 the block is one pixel and `inv` is 1, so this is the
        // conversion with no averaging at all -- the same code path
        // rather than a special case that could disagree with it.
        ur[x / hx] = clamp_u8_(su * inv);
        vr[x / hx] = clamp_u8_(sv * inv);
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
