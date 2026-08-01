#include "stages/audio-video/compare-image-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/stage-spec.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

using namespace std;

namespace vpipe {

namespace {

// Parse "#RRGGBB" / "RRGGBB" into r,g,b bytes. Defaults to BLACK on a
// malformed value -- a comparison pads with black so the padded border
// reads as "no image here", not as part of the picture (image-resample
// defaults to 114 grey instead, matching the historical letterbox).
void
parse_hex_color_(string s, uint8_t* r, uint8_t* g, uint8_t* b)
{
  *r = *g = *b = 0;
  if (!s.empty() && s[0] == '#') { s = s.substr(1); }
  if (s.size() >= 6) {
    auto hx = [&](size_t off) -> long {
      return strtol(s.substr(off, 2).c_str(), nullptr, 16);
    };
    *r = static_cast<uint8_t>(hx(0));
    *g = static_cast<uint8_t>(hx(2));
    *b = static_cast<uint8_t>(hx(4));
  }
}

inline uint8_t
clamp_byte_(float v)
{
  if (v <= 0.0f)   { return 0; }
  if (v >= 255.0f) { return 255; }
  return static_cast<uint8_t>(lrintf(v));
}

constexpr ConfigKey kAttrs[] = {
  {.key = "input_normalized", .type = ConfigType::Bool,
   .doc = "F32 input in [0,1] vs [0,255]", .def_bool = true},
  {.key = "title", .type = ConfigType::String,
   .doc = "optional label shown in the web-ui compare picker; empty = "
          "the stage id", .def_str = ""},
  {.key = "pad_color", .type = ConfigType::String,
   .doc = "#RRGGBB fill for the padded border of whichever image is "
          "smaller than the common size", .def_str = "#000000"},
};
const PortSpec kIports[] = {
  {.name = "a", .doc = "image A: planar RGB TensorBeat [3,H,W] (F32 or "
                       "U8). Optional -- unwired or silent shows black.",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
  {.name = "b", .doc = "image B: planar RGB TensorBeat [3,H,W] (F32 or "
                       "U8). Optional -- unwired or silent shows black.",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "compare-image",
  .doc       = "Sink: pairs two images for comparison in its own web-ui "
               "view -- A only, B only, side-by-side (left/right or "
               "top/bottom) with a centre divider, or a draggable "
               "vertical / horizontal wipe. Zoom and pan stay in sync "
               "across both. Mismatched resolutions are matched to "
               "max(Wa,Wb) x max(Ha,Hb) with the pad policy (fit, "
               "centre, pad_color border). A missing input shows black. "
               "0 oports.",
  .display_name = "Compare Images",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};

}  // namespace

CompareImageStage::CompareImageStage(const SessionContextIntf* s,
                                     string                    id,
                                     vector<InEdge>            iports,
                                     FlexData                  config)
  : TypedStage<CompareImageStage>(s, std::move(id), std::move(iports),
                                  std::move(config))
  , _libs(s->ffmpeg_libraries())
{
  _input_normalized = attr_bool("input_normalized");
  _title            = attr_str("title");
  parse_hex_color_(attr_str("pad_color"), &_pad_r, &_pad_g, &_pad_b);

  allocate_oports(spec().oports.size());   // sink: 0 oports
  _channel = std::make_shared<CompareImageChannel>();

  session()->info(fmt(
      "compare-image('{}'): image A/B comparison view", this->id()));
}

CompareImageStage::~CompareImageStage()
{
  teardown_();
}

const StageSpec&
CompareImageStage::spec() const noexcept
{
  return kSpec;
}

// Only the DESTRUCTOR closes the channel. Running out of input is not a
// reason to: a comparison is a still result, and the last published pair
// stays worth looking at after the pipeline has drained -- closing here
// would drop a mounted panel back to "waiting" the instant the images it
// was asked to compare became final. A live video preview closes on
// teardown because a stopped stream has nothing left to show; this is
// the opposite case.
void
CompareImageStage::teardown_()
{
  if (_torn) { return; }
  _torn = true;
  if (_channel) { _channel->close(); }
  if (_png)     { _libs->avcodec().api.free_context(&_png); }
  if (_png_frm) { _libs->avutil().api.frame_free(&_png_frm); }
  if (_png_pkt) { _libs->avcodec().api.packet_free(&_png_pkt); }
}

void
CompareImageStage::resolve_roles_(RuntimeContext& ctx)
{
  if (_roles_resolved) { return; }
  _want_a = ctx.num_iports() >= 1 && ctx.iport_connected(0);
  _want_b = ctx.num_iports() >= 2 && ctx.iport_connected(1);
  _roles_resolved = true;
  session()->info(fmt(
      "compare-image('{}'): roles a={} b={}", this->id(), _want_a, _want_b));
}

bool
CompareImageStage::unpack_(const TensorBeat& tb, Side& side) const
{
  if (tb.shape.size() != 3 || tb.shape[0] != 3) { return false; }
  const int H = static_cast<int>(tb.shape[1]);
  const int W = static_cast<int>(tb.shape[2]);
  if (H <= 0 || W <= 0) { return false; }

  const size_t plane   = static_cast<size_t>(H) * W;
  const size_t expect  = 3 * plane;
  vector<uint8_t>       out(expect);

  if (tb.dtype == TensorBeat::DType::U8) {
    const uint8_t*         src = nullptr;
    AlignedVector<uint8_t> tmp;
    if (tb.is_contiguous() && tb.byte_size() == expect) {
      src = tb.as_u8();
    } else {
      tmp = tb.materialize_contiguous();
      src = tmp.data();
    }
    for (size_t i = 0; i < plane; ++i) {
      out[i * 3 + 0] = src[i];
      out[i * 3 + 1] = src[plane + i];
      out[i * 3 + 2] = src[2 * plane + i];
    }
  } else if (tb.dtype == TensorBeat::DType::F32) {
    const float*         src = nullptr;
    AlignedVector<float> tmp;
    if (tb.is_contiguous() && tb.byte_size() == expect * sizeof(float)) {
      src = tb.as_f32();
    } else {
      tmp = tb.materialize_contiguous_as<float>();
      src = tmp.data();
    }
    const float scale = _input_normalized ? 255.0f : 1.0f;
    for (size_t i = 0; i < plane; ++i) {
      out[i * 3 + 0] = clamp_byte_(src[i] * scale);
      out[i * 3 + 1] = clamp_byte_(src[plane + i] * scale);
      out[i * 3 + 2] = clamp_byte_(src[2 * plane + i] * scale);
    }
  } else {
    return false;   // unsupported dtype: keep whatever we had
  }

  side.rgb   = std::move(out);
  side.w     = W;
  side.h     = H;
  side.valid = true;
  return true;
}

void
CompareImageStage::pad_fit_(const Side& src, int dst_w, int dst_h,
                            vector<uint8_t>* dst) const
{
  dst->assign(static_cast<size_t>(dst_w) * dst_h * 3, 0);
  // Fill with pad_color first; the fitted image overwrites its box.
  for (size_t i = 0; i < dst->size(); i += 3) {
    (*dst)[i + 0] = _pad_r;
    (*dst)[i + 1] = _pad_g;
    (*dst)[i + 2] = _pad_b;
  }
  if (!src.valid || src.w <= 0 || src.h <= 0) { return; }

  // Already the target size -> verbatim copy, no resample. This is the
  // common case (both inputs agree) and must not go through the
  // interpolator, which would soften a pixel-exact image.
  if (src.w == dst_w && src.h == dst_h) {
    std::memcpy(dst->data(), src.rgb.data(), src.rgb.size());
    return;
  }

  // pad policy: largest fit preserving aspect ratio, centred.
  const double sx = static_cast<double>(dst_w) / src.w;
  const double sy = static_cast<double>(dst_h) / src.h;
  const double s  = std::min(sx, sy);
  int fit_w = static_cast<int>(std::lround(src.w * s));
  int fit_h = static_cast<int>(std::lround(src.h * s));
  fit_w = std::clamp(fit_w, 1, dst_w);
  fit_h = std::clamp(fit_h, 1, dst_h);
  const int off_x = (dst_w - fit_w) / 2;
  const int off_y = (dst_h - fit_h) / 2;

  // Bilinear sample, matching image-resample's default algorithm.
  for (int y = 0; y < fit_h; ++y) {
    const double fy = (y + 0.5) * src.h / fit_h - 0.5;
    const int    y0 = static_cast<int>(std::floor(fy));
    const double wy = fy - y0;
    const int    y0c = std::clamp(y0, 0, src.h - 1);
    const int    y1c = std::clamp(y0 + 1, 0, src.h - 1);
    uint8_t* drow =
        dst->data() + (static_cast<size_t>(y + off_y) * dst_w + off_x) * 3;
    for (int x = 0; x < fit_w; ++x) {
      const double fx = (x + 0.5) * src.w / fit_w - 0.5;
      const int    x0 = static_cast<int>(std::floor(fx));
      const double wx = fx - x0;
      const int    x0c = std::clamp(x0, 0, src.w - 1);
      const int    x1c = std::clamp(x0 + 1, 0, src.w - 1);
      for (int c = 0; c < 3; ++c) {
        const double p00 =
            src.rgb[(static_cast<size_t>(y0c) * src.w + x0c) * 3 + c];
        const double p01 =
            src.rgb[(static_cast<size_t>(y0c) * src.w + x1c) * 3 + c];
        const double p10 =
            src.rgb[(static_cast<size_t>(y1c) * src.w + x0c) * 3 + c];
        const double p11 =
            src.rgb[(static_cast<size_t>(y1c) * src.w + x1c) * 3 + c];
        const double top = p00 + (p01 - p00) * wx;
        const double bot = p10 + (p11 - p10) * wx;
        drow[x * 3 + c] =
            clamp_byte_(static_cast<float>(top + (bot - top) * wy));
      }
    }
  }
}

CompareImageChannel::Png
CompareImageStage::encode_png_(const vector<uint8_t>& rgb, int w, int h)
{
  if (_png_bad || _libs == nullptr) { return nullptr; }

  // (Re)build the encoder when the common size changes -- an AVCodec
  // context's dimensions are fixed once opened.
  if (_png != nullptr && (_png_w != w || _png_h != h)) {
    _libs->avcodec().api.free_context(&_png);
    _libs->avutil().api.frame_free(&_png_frm);
    _png = nullptr;
    _png_frm = nullptr;
  }
  if (_png == nullptr) {
    const AVCodec* codec = _libs->avcodec().api.find_encoder_by_name("png");
    if (codec == nullptr) {
      session()->warn(fmt(
          "compare-image('{}'): PNG encoder unavailable; view disabled",
          this->id()));
      _png_bad = true;
      return nullptr;
    }
    _png = _libs->avcodec().api.alloc_context3(codec);
    if (_png == nullptr) { _png_bad = true; return nullptr; }
    _png->width     = w;
    _png->height    = h;
    _png->pix_fmt   = AV_PIX_FMT_RGB24;
    _png->time_base = AVRational{1, 1};
    const int rc = _libs->avcodec().api.open2(_png, codec, nullptr);
    if (rc < 0) {
      session()->warn(fmt(
          "compare-image('{}'): PNG avcodec_open2 failed", this->id()));
      _libs->avcodec().api.free_context(&_png);
      _png_bad = true;
      return nullptr;
    }
    _png_w = w;
    _png_h = h;
  }
  if (_png_frm == nullptr) {
    _png_frm = _libs->avutil().api.frame_alloc();
    if (_png_frm == nullptr) { _png_bad = true; return nullptr; }
    _png_frm->format = AV_PIX_FMT_RGB24;
    _png_frm->width  = w;
    _png_frm->height = h;
    if (_libs->avutil().api.frame_get_buffer(_png_frm, 32) < 0) {
      _libs->avutil().api.frame_free(&_png_frm);
      _png_bad = true;
      return nullptr;
    }
  }
  if (_png_pkt == nullptr) {
    _png_pkt = _libs->avcodec().api.packet_alloc();
    if (_png_pkt == nullptr) { _png_bad = true; return nullptr; }
  }

  // Copy row-wise: the frame's linesize is padded for alignment and is
  // not necessarily w*3.
  const int ls = _png_frm->linesize[0];
  for (int y = 0; y < h; ++y) {
    std::memcpy(_png_frm->data[0] + static_cast<size_t>(y) * ls,
                rgb.data() + static_cast<size_t>(y) * w * 3,
                static_cast<size_t>(w) * 3);
  }
  _png_frm->pts = 0;

  if (_libs->avcodec().api.send_frame(_png, _png_frm) < 0) {
    return nullptr;
  }
  auto out = make_shared<vector<uint8_t>>();
  for (;;) {
    const int rc = _libs->avcodec().api.receive_packet(_png, _png_pkt);
    if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { break; }
    if (rc < 0) { break; }
    if (_png_pkt->size > 0) {
      out->insert(out->end(), _png_pkt->data,
                  _png_pkt->data + _png_pkt->size);
    }
    _libs->avcodec().api.packet_unref(_png_pkt);
  }
  if (out->empty()) { return nullptr; }
  return out;
}

void
CompareImageStage::publish_()
{
  if (!_a.valid && !_b.valid) {
    _pub_w = 0;
    _pub_h = 0;
    _channel->publish(nullptr, nullptr, 0, 0);
    return;
  }

  // Common size = the larger extent on each axis, so neither image is
  // ever downscaled (a comparison must not lose detail it was given).
  const int w = std::max(_a.valid ? _a.w : 0, _b.valid ? _b.w : 0);
  const int h = std::max(_a.valid ? _a.h : 0, _b.valid ? _b.h : 0);
  if (w <= 0 || h <= 0) { return; }
  _pub_w = w;
  _pub_h = h;

  CompareImageChannel::Png pa;
  CompareImageChannel::Png pb;
  vector<uint8_t> canvas;
  if (_a.valid) {
    pad_fit_(_a, w, h, &canvas);
    pa = encode_png_(canvas, w, h);
  }
  if (_b.valid) {
    pad_fit_(_b, w, h, &canvas);
    pb = encode_png_(canvas, w, h);
  }
  _channel->publish(std::move(pa), std::move(pb), w, h);
}

Job
CompareImageStage::process(RuntimeContext& ctx)
{
  resolve_roles_(ctx);

  if (!_want_a && !_want_b) {
    // Nothing wired: publish the empty pair once so a mounted view shows
    // black rather than waiting, then retire. The channel stays OPEN
    // (see teardown_) so the panel reports the empty result rather than
    // dropping back to "waiting".
    publish_();
    ctx.signal_done();
    co_return;
  }

  // Wake on whichever input arrives; both are optional and they run in
  // separate clock groups, so blocking on one would stall the other.
  vector<unsigned> ports;
  if (_want_a) { ports.push_back(0); }
  if (_want_b) { ports.push_back(1); }
  co_await ctx.read_any(ports);

  // Drain both backlogs, keeping only the newest frame per side -- an
  // unread backlog is still consumed so the producer isn't
  // back-pressured.
  bool changed = false;
  for (unsigned p : ports) {
    const uint32_t avail = ctx.backlog(p);
    for (uint32_t i = 0; i < avail; ++i) {
      auto beat = co_await ctx.read(p);
      if (!beat) { break; }
      const auto* tb = dynamic_cast<const TensorBeatPayload*>(beat.get());
      if (tb == nullptr) { continue; }
      if (unpack_(*tb, p == 0 ? _a : _b)) { changed = true; }
    }
  }
  if (changed) { publish_(); }

  // read_any treats a closed port as perpetually ready, so a loop that
  // only re-armed would spin once every input is drained and closed.
  bool all_eos = true;
  for (unsigned p : ports) {
    if (!ctx.eos(p)) { all_eos = false; break; }
  }
  if (all_eos || ctx.stop_requested()) {
    // Retire, but leave the channel open holding the final pair -- the
    // comparison is the output, and it outlives the inputs that made it.
    ctx.signal_done();
    co_return;
  }
  co_return;
}

VPIPE_REGISTER_STAGE(CompareImageStage)
VPIPE_REGISTER_SPEC(CompareImageStage, kSpec)

}
