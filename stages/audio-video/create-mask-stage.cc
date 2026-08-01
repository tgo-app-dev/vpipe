#include "stages/audio-video/create-mask-stage.h"

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

// How long one interactive process() call waits on the editor before
// coming back round. Short enough that a stop request and a newly
// arrived reference image are both picked up promptly, long enough that
// an idle editor costs nothing measurable.
constexpr int kCommitPollMs = 200;

// Fallback palette, used when `class_colors` is unset. Index 0 is
// background and is never painted, so its entry only has to exist.
constexpr uint32_t kDefaultPalette[] = {
  0x000000, 0xff3b30, 0x34c759, 0x0a84ff, 0xffd60a,
  0xbf5af2, 0xff9f0a, 0x64d2ff, 0xff2d55, 0x30d158,
};

uint32_t
parse_hex_color_(string s, uint32_t fallback)
{
  if (!s.empty() && s[0] == '#') { s = s.substr(1); }
  if (s.size() < 6) { return fallback; }
  char* end = nullptr;
  const long v = strtol(s.substr(0, 6).c_str(), &end, 16);
  if (end == nullptr || *end != '\0') { return fallback; }
  return static_cast<uint32_t>(v) & 0xffffffu;
}

// Split "#aaa,#bbb" into colours. An empty / malformed entry falls back
// to the built-in palette slot for its position, so a partial list is
// still usable rather than a configuration error.
vector<uint32_t>
parse_palette_(const string& spec, int n)
{
  vector<uint32_t> out;
  out.reserve(static_cast<size_t>(max(1, n)));
  size_t pos = 0;
  for (int i = 0; i < max(1, n); ++i) {
    const size_t k = static_cast<size_t>(i);
    const uint32_t def =
        kDefaultPalette[k % (sizeof(kDefaultPalette) / sizeof(uint32_t))];
    string item;
    if (pos <= spec.size() && pos != string::npos && !spec.empty()) {
      const size_t comma = spec.find(',', pos);
      item = spec.substr(pos, comma == string::npos ? string::npos
                                                    : comma - pos);
      pos  = (comma == string::npos) ? string::npos : comma + 1;
      while (!item.empty() && (item.front() == ' ' || item.front() == '\t')) {
        item.erase(item.begin());
      }
      while (!item.empty() && (item.back() == ' ' || item.back() == '\t')) {
        item.pop_back();
      }
    }
    out.push_back(item.empty() ? def : parse_hex_color_(item, def));
  }
  return out;
}

inline uint8_t
clamp_byte_(float v)
{
  if (v <= 0.0f)   { return 0; }
  if (v >= 255.0f) { return 255; }
  return static_cast<uint8_t>(lrintf(v));
}

constexpr ConfigKey kAttrs[] = {
  {.key = "mask_mode", .type = ConfigType::String,
   .doc = "\"binary\" (two-state 0/255), \"alpha\" (soft 0..255 coverage, "
          "brush hardness shapes the falloff), or \"class\" (multi-class "
          "index map)", .def_str = "binary"},
  {.key = "classes", .type = ConfigType::Int,
   .doc = "class count INCLUDING background (index 0); \"class\" mode only",
   .def_int = 3},
  {.key = "class_colors", .type = ConfigType::String,
   .doc = "comma-separated #RRGGBB, one per class. Presentation only -- it "
          "drives the editor and the overlay, never the index map that "
          "leaves the stage. Empty = a built-in palette", .def_str = ""},
  {.key = "output", .type = ConfigType::String,
   .doc = "\"mask\" emits the mask [1,H,W]; \"overlay\" emits the reference "
          "image with the mask painted on it [3,H,W]", .def_str = "mask"},
  {.key = "overlay_color", .type = ConfigType::String,
   .doc = "#RRGGBB the mask is painted with in binary / alpha mode; class "
          "mode uses class_colors", .def_str = "#ff3b30"},
  {.key = "overlay_opacity", .type = ConfigType::Real,
   .doc = "0..1 strength the mask is painted over the reference image with",
   .def_real = 0.5},
  {.key = "output_dtype", .type = ConfigType::String,
   .doc = "\"u8\" (0..255, or a class index) or \"f32\" (coverage in [0,1]; "
          "a class index stays an index)", .def_str = "u8"},
  {.key = "width", .type = ConfigType::Int,
   .doc = "mask canvas width; 0 infers it from `height` and the reference "
          "image's aspect ratio, or matches the reference image",
   .def_int = 0},
  {.key = "height", .type = ConfigType::Int,
   .doc = "mask canvas height; 0 infers it the same way as `width`",
   .def_int = 0},
  {.key = "interactive", .type = ConfigType::Bool,
   .doc = "run the mask editor and emit one beat per commit. False = no "
          "GUI: the mask comes from in-mask and is emitted as it arrives",
   .def_bool = true},
  {.key = "input_normalized", .type = ConfigType::Bool,
   .doc = "F32 input in [0,1] vs [0,255]", .def_bool = true},
  {.key = "title", .type = ConfigType::String,
   .doc = "optional label shown in the web-ui mask-editor picker; empty = "
          "the stage id", .def_str = ""},
};
const PortSpec kIports[] = {
  {.name = "ref-image",
   .doc = "reference image: planar RGB TensorBeat [3,H,W] (F32 or U8). "
          "Optional -- the editor's background and the overlay's base; "
          "unwired, the editor paints over black.",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
  {.name = "in-mask",
   .doc = "mask to start from: planar TensorBeat [1,H,W] or [H,W] (F32 or "
          "U8). Optional -- it seeds the editor, and is the whole input "
          "when `interactive` is false.",
   .type = &typeid(TensorBeatPayload),
   .tags = "mask-frames", .clock_group = 1},
};
const PortSpec kOports[] = {
  {.name = "out",
   .doc = "the mask [1,H,W] at the canvas resolution, or the reference "
          "image with the mask painted on it [3,H,W] at the reference "
          "resolution -- whichever `output` selects.",
   .type = &typeid(TensorBeatPayload),
   .tags = "mask-frames,rgb-frames", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "create-mask",
  .doc       = "Authors a mask -- painted by hand in its own web-ui editor "
               "with a round brush (adjustable radius, and hardness in "
               "alpha mode), or passed through from an input. Two-state, "
               "soft alpha, or multi-class. Emits ONE beat per commit: "
               "either the mask [1,H,W] or the reference image with the "
               "mask painted on top [3,H,W]. With `interactive` false it "
               "runs headless, overlaying an incoming mask as it arrives. "
               "2 optional iports, 1 oport.",
  .display_name = "Create Mask",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

CreateMaskStage::CreateMaskStage(const SessionContextIntf* s,
                                 string                    id,
                                 vector<InEdge>            iports,
                                 FlexData                  config)
  : TypedStage<CreateMaskStage>(s, std::move(id), std::move(iports),
                                std::move(config))
  , _libs(s->ffmpeg_libraries())
{
  const string mode = attr_str("mask_mode");
  if (mode == "alpha") {
    _mode = MaskEditorChannel::Mode::Alpha;
  } else if (mode == "class") {
    _mode = MaskEditorChannel::Mode::Class;
  } else if (mode == "binary" || mode.empty()) {
    _mode = MaskEditorChannel::Mode::Binary;
  } else {
    fail_config(fmt("create-mask('{}'): mask_mode must be \"binary\", "
                    "\"alpha\" or \"class\" (got \"{}\")",
                    this->id(), mode));
  }

  _classes = static_cast<int>(attr_int("classes"));
  if (_mode == MaskEditorChannel::Mode::Class) {
    // 256 because a class index has to survive the one-byte canvas the
    // brush paints into and the one-channel beat that leaves the stage.
    if (_classes < 2 || _classes > 256) {
      fail_config(fmt("create-mask('{}'): classes must be in [2,256] "
                      "(got {})", this->id(), _classes));
      _classes = 3;
    }
    _colors = parse_palette_(attr_str("class_colors"), _classes);
  } else {
    _classes = 2;
    _colors  = {parse_hex_color_(attr_str("overlay_color"), 0xff3b30)};
  }

  const string out = attr_str("output");
  if (out == "overlay") {
    _overlay_out = true;
  } else if (out != "mask" && !out.empty()) {
    fail_config(fmt("create-mask('{}'): output must be \"mask\" or "
                    "\"overlay\" (got \"{}\")", this->id(), out));
  }

  _overlay_opacity =
      static_cast<float>(clamp(attr_real("overlay_opacity"), 0.0, 1.0));

  const string dt = attr_str("output_dtype");
  if (dt == "f32") {
    _f32_out = true;
  } else if (dt != "u8" && !dt.empty()) {
    fail_config(fmt("create-mask('{}'): output_dtype must be \"u8\" or "
                    "\"f32\" (got \"{}\")", this->id(), dt));
  }

  _cfg_w = static_cast<int>(attr_int("width"));
  _cfg_h = static_cast<int>(attr_int("height"));
  if (_cfg_w < 0 || _cfg_h < 0) {
    fail_config(fmt("create-mask('{}'): width/height must not be negative",
                    this->id()));
    _cfg_w = _cfg_h = 0;
  }

  _interactive      = attr_bool("interactive");
  _input_normalized = attr_bool("input_normalized");
  _title            = attr_str("title");

  allocate_oports(spec().oports.size());
  _channel = make_shared<MaskEditorChannel>();

  session()->info(fmt(
      "create-mask('{}'): {} mask, {} output, {}",
      this->id(),
      _mode == MaskEditorChannel::Mode::Alpha ? "alpha"
        : (_mode == MaskEditorChannel::Mode::Class ? "class" : "binary"),
      _overlay_out ? "overlay" : "mask",
      _interactive ? "interactive editor" : "headless"));
}

CreateMaskStage::~CreateMaskStage()
{
  teardown_();
}

const StageSpec&
CreateMaskStage::spec() const noexcept
{
  return kSpec;
}

void
CreateMaskStage::free_encoder_(PngEncoder* enc)
{
  if (enc->ctx != nullptr)   { _libs->avcodec().api.free_context(&enc->ctx); }
  if (enc->frame != nullptr) { _libs->avutil().api.frame_free(&enc->frame); }
  enc->ctx   = nullptr;
  enc->frame = nullptr;
}

// Only the DESTRUCTOR closes the channel -- the same call the
// compare-image view makes, and for the same reason: a mask is a still
// result that outlives the inputs that framed it, and closing on EOS
// would drop a mounted editor back to "waiting" the instant its
// reference image became final.
void
CreateMaskStage::teardown_()
{
  if (_torn) { return; }
  _torn = true;
  if (_channel) { _channel->close(); }
  if (_libs == nullptr) { return; }
  free_encoder_(&_rgb_png);
  free_encoder_(&_gray_png);
  if (_pkt != nullptr) { _libs->avcodec().api.packet_free(&_pkt); }
  if (_dec != nullptr) { _libs->avcodec().api.free_context(&_dec); }
  if (_sws != nullptr) {
    _libs->swscale().api.free_context(_sws);
    _sws = nullptr;
  }
}

void
CreateMaskStage::resolve_roles_(RuntimeContext& ctx)
{
  if (_roles_resolved) { return; }
  _want_ref  = ctx.num_iports() >= 1 && ctx.iport_connected(0);
  _want_mask = ctx.num_iports() >= 2 && ctx.iport_connected(1);
  _roles_resolved = true;
  session()->info(fmt("create-mask('{}'): roles ref={} mask={}",
                      this->id(), _want_ref, _want_mask));
}

void
CreateMaskStage::resolve_canvas_(int src_w, int src_h,
                                 int* out_w, int* out_h) const noexcept
{
  const bool have_src = src_w > 0 && src_h > 0;
  if (_cfg_w > 0 && _cfg_h > 0) {
    *out_w = _cfg_w;
    *out_h = _cfg_h;
    return;
  }
  if (_cfg_w > 0) {
    *out_w = _cfg_w;
    // No source to take an aspect ratio from: a square is the only
    // answer that does not invent one.
    *out_h = have_src
        ? max(1, static_cast<int>(lround(
              static_cast<double>(_cfg_w) * src_h / src_w)))
        : _cfg_w;
    return;
  }
  if (_cfg_h > 0) {
    *out_h = _cfg_h;
    *out_w = have_src
        ? max(1, static_cast<int>(lround(
              static_cast<double>(_cfg_h) * src_w / src_h)))
        : _cfg_h;
    return;
  }
  *out_w = max(0, src_w);
  *out_h = max(0, src_h);
}

bool
CreateMaskStage::unpack_ref_(const TensorBeat& tb)
{
  if (tb.shape.size() != 3 || tb.shape[0] != 3) { return false; }
  const int H = static_cast<int>(tb.shape[1]);
  const int W = static_cast<int>(tb.shape[2]);
  if (H <= 0 || W <= 0) { return false; }

  const size_t plane  = static_cast<size_t>(H) * W;
  const size_t expect = 3 * plane;
  vector<uint8_t> out(expect);

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

  _ref       = std::move(out);
  _rw        = W;
  _rh        = H;
  _ref_valid = true;
  return true;
}

bool
CreateMaskStage::unpack_mask_(const TensorBeat& tb, vector<uint8_t>* out,
                              int* w, int* h) const
{
  int H = 0;
  int W = 0;
  if (tb.shape.size() == 3 && tb.shape[0] == 1) {
    H = static_cast<int>(tb.shape[1]);
    W = static_cast<int>(tb.shape[2]);
  } else if (tb.shape.size() == 2) {
    H = static_cast<int>(tb.shape[0]);
    W = static_cast<int>(tb.shape[1]);
  } else {
    return false;
  }
  if (H <= 0 || W <= 0) { return false; }

  const size_t n = static_cast<size_t>(H) * W;
  out->assign(n, 0);

  if (tb.dtype == TensorBeat::DType::U8) {
    const uint8_t*         src = nullptr;
    AlignedVector<uint8_t> tmp;
    if (tb.is_contiguous() && tb.byte_size() == n) {
      src = tb.as_u8();
    } else {
      tmp = tb.materialize_contiguous();
      src = tmp.data();
    }
    std::memcpy(out->data(), src, n);
  } else if (tb.dtype == TensorBeat::DType::F32) {
    const float*         src = nullptr;
    AlignedVector<float> tmp;
    if (tb.is_contiguous() && tb.byte_size() == n * sizeof(float)) {
      src = tb.as_f32();
    } else {
      tmp = tb.materialize_contiguous_as<float>();
      src = tmp.data();
    }
    // A class map is an INDEX whatever the dtype, so it is never
    // scaled -- only coverage is.
    const float scale =
        (_mode == MaskEditorChannel::Mode::Class || !_input_normalized)
            ? 1.0f : 255.0f;
    for (size_t i = 0; i < n; ++i) {
      (*out)[i] = clamp_byte_(src[i] * scale);
    }
  } else {
    return false;
  }

  // Binary tolerates any non-zero input as "set", so a mask that came in
  // as 0/1 rather than 0/255 still means what it says.
  if (_mode == MaskEditorChannel::Mode::Binary) {
    for (uint8_t& v : *out) { v = v != 0 ? 255 : 0; }
  }
  *w = W;
  *h = H;
  return true;
}

void
CreateMaskStage::resample_mask_(const vector<uint8_t>& src, int sw, int sh,
                                int dw, int dh, vector<uint8_t>* dst) const
{
  dst->assign(static_cast<size_t>(max(0, dw)) * max(0, dh), 0);
  if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) { return; }
  if (src.size() < static_cast<size_t>(sw) * sh) { return; }
  if (sw == dw && sh == dh) {
    std::memcpy(dst->data(), src.data(), src.size());
    return;
  }

  // Only coverage may be interpolated. A two-state mask and a class map
  // have no meaningful value BETWEEN their samples -- the average of
  // class 1 and class 3 is not class 2 -- so both take the nearest.
  const bool smooth = _mode == MaskEditorChannel::Mode::Alpha;

  for (int y = 0; y < dh; ++y) {
    const double fy = (y + 0.5) * sh / dh - 0.5;
    uint8_t* drow = dst->data() + static_cast<size_t>(y) * dw;
    if (!smooth) {
      const int sy = clamp(static_cast<int>(lround(fy)), 0, sh - 1);
      const uint8_t* srow = src.data() + static_cast<size_t>(sy) * sw;
      for (int x = 0; x < dw; ++x) {
        const double fx = (x + 0.5) * sw / dw - 0.5;
        drow[x] = srow[clamp(static_cast<int>(lround(fx)), 0, sw - 1)];
      }
      continue;
    }
    const int    y0 = static_cast<int>(floor(fy));
    const double wy = fy - y0;
    const uint8_t* r0 =
        src.data() + static_cast<size_t>(clamp(y0, 0, sh - 1)) * sw;
    const uint8_t* r1 =
        src.data() + static_cast<size_t>(clamp(y0 + 1, 0, sh - 1)) * sw;
    for (int x = 0; x < dw; ++x) {
      const double fx  = (x + 0.5) * sw / dw - 0.5;
      const int    x0  = static_cast<int>(floor(fx));
      const double wx  = fx - x0;
      const int    x0c = clamp(x0, 0, sw - 1);
      const int    x1c = clamp(x0 + 1, 0, sw - 1);
      const double top = r0[x0c] + (r0[x1c] - r0[x0c]) * wx;
      const double bot = r1[x0c] + (r1[x1c] - r1[x0c]) * wx;
      drow[x] = clamp_byte_(static_cast<float>(top + (bot - top) * wy));
    }
  }
}

bool
CreateMaskStage::sample_color_(uint8_t v, uint8_t* r, uint8_t* g, uint8_t* b,
                               float* a) const
{
  if (v == 0) { return false; }        // background paints nothing
  uint32_t c = 0;
  if (_mode == MaskEditorChannel::Mode::Class) {
    if (static_cast<size_t>(v) >= _colors.size()) { return false; }
    c  = _colors[v];
    *a = _overlay_opacity;
  } else {
    c  = _colors.empty() ? 0xff3b30u : _colors[0];
    *a = _overlay_opacity
       * (_mode == MaskEditorChannel::Mode::Alpha ? v / 255.0f : 1.0f);
  }
  *r = static_cast<uint8_t>((c >> 16) & 0xff);
  *g = static_cast<uint8_t>((c >> 8) & 0xff);
  *b = static_cast<uint8_t>(c & 0xff);
  return *a > 0.0f;
}

void
CreateMaskStage::overlay_(const vector<uint8_t>& base, int bw, int bh,
                          vector<uint8_t>* dst) const
{
  const size_t n = static_cast<size_t>(max(0, bw)) * max(0, bh);
  dst->assign(n * 3, 0);
  if (bw <= 0 || bh <= 0) { return; }
  if (base.size() >= n * 3) {
    std::memcpy(dst->data(), base.data(), n * 3);
  }

  // Stretch the canvas onto the base once, then blend 1:1 -- so the
  // resample rule (nearest vs bilinear) is the SAME one the mask output
  // goes through, and an overlay can never disagree with the mask that
  // produced it.
  vector<uint8_t> m;
  resample_mask_(_mask, _mw, _mh, bw, bh, &m);
  if (m.size() < n) { return; }

  uint8_t* d = dst->data();
  for (size_t i = 0; i < n; ++i) {
    uint8_t r = 0, g = 0, b = 0;
    float   a = 0.0f;
    if (!sample_color_(m[i], &r, &g, &b, &a)) { continue; }
    d[i * 3 + 0] = clamp_byte_(d[i * 3 + 0] * (1.0f - a) + r * a);
    d[i * 3 + 1] = clamp_byte_(d[i * 3 + 1] * (1.0f - a) + g * a);
    d[i * 3 + 2] = clamp_byte_(d[i * 3 + 2] * (1.0f - a) + b * a);
  }
}

MaskEditorChannel::Bytes
CreateMaskStage::encode_png_(PngEncoder* enc, const uint8_t* src,
                             int w, int h, int comps)
{
  if (enc->bad || _libs == nullptr || src == nullptr) { return nullptr; }
  if (w <= 0 || h <= 0) { return nullptr; }

  // An AVCodecContext's dimensions are fixed once opened, so a geometry
  // change means a fresh encoder.
  if (enc->ctx != nullptr && (enc->w != w || enc->h != h)) {
    free_encoder_(enc);
  }
  const AVPixelFormat pix =
      comps == 1 ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_RGB24;
  if (enc->ctx == nullptr) {
    const AVCodec* codec = _libs->avcodec().api.find_encoder_by_name("png");
    if (codec == nullptr) {
      session()->warn(fmt(
          "create-mask('{}'): PNG encoder unavailable; editor disabled",
          this->id()));
      enc->bad = true;
      return nullptr;
    }
    enc->ctx = _libs->avcodec().api.alloc_context3(codec);
    if (enc->ctx == nullptr) { enc->bad = true; return nullptr; }
    enc->ctx->width     = w;
    enc->ctx->height    = h;
    enc->ctx->pix_fmt   = pix;
    enc->ctx->time_base = AVRational{1, 1};
    if (_libs->avcodec().api.open2(enc->ctx, codec, nullptr) < 0) {
      session()->warn(fmt("create-mask('{}'): PNG avcodec_open2 failed",
                          this->id()));
      _libs->avcodec().api.free_context(&enc->ctx);
      enc->bad = true;
      return nullptr;
    }
    enc->w = w;
    enc->h = h;
  }
  if (enc->frame == nullptr) {
    enc->frame = _libs->avutil().api.frame_alloc();
    if (enc->frame == nullptr) { enc->bad = true; return nullptr; }
    enc->frame->format = pix;
    enc->frame->width  = w;
    enc->frame->height = h;
    if (_libs->avutil().api.frame_get_buffer(enc->frame, 32) < 0) {
      _libs->avutil().api.frame_free(&enc->frame);
      enc->bad = true;
      return nullptr;
    }
  }
  if (_pkt == nullptr) {
    _pkt = _libs->avcodec().api.packet_alloc();
    if (_pkt == nullptr) { enc->bad = true; return nullptr; }
  }

  // Row-wise: the frame's linesize is padded for alignment and is not
  // necessarily w * comps.
  const int ls = enc->frame->linesize[0];
  for (int y = 0; y < h; ++y) {
    std::memcpy(enc->frame->data[0] + static_cast<size_t>(y) * ls,
                src + static_cast<size_t>(y) * w * comps,
                static_cast<size_t>(w) * comps);
  }
  enc->frame->pts = 0;

  if (_libs->avcodec().api.send_frame(enc->ctx, enc->frame) < 0) {
    return nullptr;
  }
  auto out = make_shared<vector<uint8_t>>();
  for (;;) {
    const int rc = _libs->avcodec().api.receive_packet(enc->ctx, _pkt);
    if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF || rc < 0) { break; }
    if (_pkt->size > 0) {
      out->insert(out->end(), _pkt->data, _pkt->data + _pkt->size);
    }
    _libs->avcodec().api.packet_unref(_pkt);
  }
  if (out->empty()) { return nullptr; }
  return out;
}

bool
CreateMaskStage::decode_commit_(const vector<uint8_t>& png,
                                vector<uint8_t>* out, int* w, int* h)
{
  if (_dec_bad || _libs == nullptr || png.empty()) { return false; }

  if (_dec == nullptr) {
    const AVCodec* codec =
        _libs->avcodec().api.find_decoder(AV_CODEC_ID_PNG);
    if (codec == nullptr) {
      session()->warn(fmt(
          "create-mask('{}'): PNG decoder unavailable; commits ignored",
          this->id()));
      _dec_bad = true;
      return false;
    }
    _dec = _libs->avcodec().api.alloc_context3(codec);
    if (_dec == nullptr) { _dec_bad = true; return false; }
    if (_libs->avcodec().api.open2(_dec, codec, nullptr) < 0) {
      _libs->avcodec().api.free_context(&_dec);
      _dec_bad = true;
      return false;
    }
  }
  if (_pkt == nullptr) {
    _pkt = _libs->avcodec().api.packet_alloc();
    if (_pkt == nullptr) { _dec_bad = true; return false; }
  }

  // A PNG is one packet. The data is borrowed, not reference-counted;
  // packet_unref on a buf-less packet is well defined and just clears
  // the fields.
  _libs->avcodec().api.packet_unref(_pkt);
  _pkt->data = const_cast<uint8_t*>(png.data());
  _pkt->size = static_cast<int>(png.size());
  const int src = _libs->avcodec().api.send_packet(_dec, _pkt);
  _pkt->data = nullptr;
  _pkt->size = 0;
  if (src < 0) { return false; }

  AVFrame* f = _libs->avutil().api.frame_alloc();
  if (f == nullptr) { return false; }
  bool ok = false;
  if (_libs->avcodec().api.receive_frame(_dec, f) >= 0
      && f->width > 0 && f->height > 0) {
    const int fw = f->width;
    const int fh = f->height;
    out->assign(static_cast<size_t>(fw) * fh, 0);

    if (f->format == AV_PIX_FMT_GRAY8) {
      // Already one byte per sample: take it verbatim. Going through
      // swscale here would invite it to treat the plane as LUMA and
      // range-expand it, which would quietly corrupt a class index.
      for (int y = 0; y < fh; ++y) {
        std::memcpy(out->data() + static_cast<size_t>(y) * fw,
                    f->data[0] + static_cast<size_t>(y) * f->linesize[0],
                    static_cast<size_t>(fw));
      }
      ok = true;
    } else {
      // Anything else -- and a browser canvas hands us RGBA or RGB24 --
      // is repacked to RGB24 and read from the R channel. RGB-to-RGB is
      // a pure repack, so no colour maths touches the samples; the view
      // writes R = G = B = value precisely so this is exact.
      _sws = _libs->swscale().api.get_cached_context(
          _sws, fw, fh, static_cast<AVPixelFormat>(f->format),
          fw, fh, AV_PIX_FMT_RGB24, SWS_POINT, nullptr, nullptr, nullptr);
      if (_sws != nullptr) {
        vector<uint8_t> rgb(static_cast<size_t>(fw) * fh * 3, 0);
        uint8_t* dst[4]   = {rgb.data(), nullptr, nullptr, nullptr};
        int      lines[4] = {fw * 3, 0, 0, 0};
        if (_libs->swscale().api.scale(_sws, f->data, f->linesize, 0, fh,
                                       dst, lines) > 0) {
          for (size_t i = 0; i < static_cast<size_t>(fw) * fh; ++i) {
            (*out)[i] = rgb[i * 3];
          }
          ok = true;
        }
      }
    }
    if (ok) {
      *w = fw;
      *h = fh;
    }
  }
  _libs->avutil().api.frame_free(&f);

  if (ok && _mode == MaskEditorChannel::Mode::Binary) {
    for (uint8_t& v : *out) { v = v != 0 ? 255 : 0; }
  }
  return ok;
}

void
CreateMaskStage::publish_()
{
  MaskEditorChannel::Frame f;
  f.width     = _mw;
  f.height    = _mh;
  f.bg_width  = _ref_valid ? _rw : 0;
  f.bg_height = _ref_valid ? _rh : 0;
  f.editor.mode            = _mode;
  f.editor.classes         = _classes;
  f.editor.colors          = _colors;
  f.editor.overlay_opacity = _overlay_opacity;
  f.editor.interactive     = _interactive;

  if (_ref_valid && !_ref.empty()) {
    f.background = encode_png_(&_rgb_png, _ref.data(), _rw, _rh, 3);
  }
  if (!_mask.empty() && _mw > 0 && _mh > 0) {
    f.mask = encode_png_(&_gray_png, _mask.data(), _mw, _mh, 1);
  }
  _channel->publish(std::move(f));
}

unique_ptr<BeatPayloadIntf>
CreateMaskStage::make_output_() const
{
  if (_mw <= 0 || _mh <= 0 || _mask.empty()) { return nullptr; }

  if (!_overlay_out) {
    TensorBeat tb;
    tb.shape = {1, _mh, _mw};
    const size_t n = static_cast<size_t>(_mw) * _mh;
    if (_f32_out) {
      tb.dtype = TensorBeat::DType::F32;
      tb.resize_contiguous(n);
      float* p = tb.as_f32();
      // A class index stays an index; only coverage is normalised.
      const float inv =
          _mode == MaskEditorChannel::Mode::Class ? 1.0f : 1.0f / 255.0f;
      for (size_t i = 0; i < n; ++i) { p[i] = _mask[i] * inv; }
    } else {
      tb.dtype = TensorBeat::DType::U8;
      tb.resize_contiguous(n);
      std::memcpy(tb.as_u8(), _mask.data(), n);
    }
    return make_payload<TensorBeatPayload>(std::move(tb));
  }

  // Overlay: the reference image's own resolution, or the canvas when
  // there is no reference (the mask painted over black -- defined, and
  // more useful than emitting nothing).
  const int bw = _ref_valid ? _rw : _mw;
  const int bh = _ref_valid ? _rh : _mh;
  vector<uint8_t> rgb;
  overlay_(_ref_valid ? _ref : vector<uint8_t>{}, bw, bh, &rgb);
  if (rgb.empty()) { return nullptr; }

  const size_t plane = static_cast<size_t>(bw) * bh;
  TensorBeat tb;
  tb.shape = {3, bh, bw};
  if (_f32_out) {
    tb.dtype = TensorBeat::DType::F32;
    tb.resize_contiguous(plane * 3);
    float* p = tb.as_f32();
    for (size_t i = 0; i < plane; ++i) {
      p[i]             = rgb[i * 3 + 0] / 255.0f;
      p[plane + i]     = rgb[i * 3 + 1] / 255.0f;
      p[2 * plane + i] = rgb[i * 3 + 2] / 255.0f;
    }
  } else {
    tb.dtype = TensorBeat::DType::U8;
    tb.resize_contiguous(plane * 3);
    uint8_t* p = tb.as_u8();
    for (size_t i = 0; i < plane; ++i) {
      p[i]             = rgb[i * 3 + 0];
      p[plane + i]     = rgb[i * 3 + 1];
      p[2 * plane + i] = rgb[i * 3 + 2];
    }
  }
  return make_payload<TensorBeatPayload>(std::move(tb));
}

// Bring the canvas up to date with what is now known about the geometry,
// carrying any existing mask across a size change rather than dropping
// it -- a reference image that arrives after the user has started
// painting must not erase the work.
void
CreateMaskStage::refresh_canvas_(int fallback_w, int fallback_h)
{
  int w = 0;
  int h = 0;
  resolve_canvas_(_ref_valid ? _rw : fallback_w,
                  _ref_valid ? _rh : fallback_h, &w, &h);
  if (w <= 0 || h <= 0) { return; }
  const bool had = !_mask.empty() && _mw > 0 && _mh > 0;
  if (had && w == _mw && h == _mh) { return; }
  if (had) {
    vector<uint8_t> scaled;
    resample_mask_(_mask, _mw, _mh, w, h, &scaled);
    _mask = std::move(scaled);
  } else {
    _mask.assign(static_cast<size_t>(w) * h, 0);
  }
  _mw = w;
  _mh = h;
}

Job
CreateMaskStage::process(RuntimeContext& ctx)
{
  resolve_roles_(ctx);

  // Headless with no mask input: nothing can ever drive this stage.
  if (!_interactive && !_want_mask) {
    if (!_warned_idle) {
      _warned_idle = true;
      session()->warn(fmt(
          "create-mask('{}'): interactive=false with no in-mask input -- "
          "nothing can produce a mask; retiring", this->id()));
    }
    ctx.signal_done();
    co_return;
  }

  vector<unsigned> ports;
  if (_want_ref)  { ports.push_back(0); }
  if (_want_mask) { ports.push_back(1); }

  // Headless runs at the pace of its inputs, so it blocks on them. The
  // editor must NOT: the user decides when something happens, so that
  // path takes only what is already queued and then waits on the commit
  // instead. Blocking on an input there would stall an editor whose
  // reference image is a single still that has already arrived.
  if (!_interactive && !ports.empty()) {
    co_await ctx.read_any(ports);
  }

  bool changed    = false;
  bool mask_moved = false;
  for (unsigned p : ports) {
    // Drain the whole backlog keeping the newest -- an unread backlog is
    // still consumed so the producer is not back-pressured.
    const uint32_t avail = ctx.backlog(p);
    for (uint32_t i = 0; i < avail; ++i) {
      auto beat = co_await ctx.read(p);
      if (!beat) { break; }
      const auto* tb = dynamic_cast<const TensorBeatPayload*>(beat.get());
      if (tb == nullptr) { continue; }
      if (p == 0) {
        if (unpack_ref_(*tb)) { changed = true; }
      } else {
        vector<uint8_t> m;
        int mw = 0, mh = 0;
        if (unpack_mask_(*tb, &m, &mw, &mh)) {
          refresh_canvas_(mw, mh);
          resample_mask_(m, mw, mh, _mw, _mh, &_mask);
          changed = mask_moved = true;
        }
      }
    }
  }
  if (changed) {
    refresh_canvas_(_mw, _mh);
    publish_();
  }

  if (!_interactive) {
    // Input-driven: every mask that arrives is one beat out.
    if (mask_moved) {
      if (auto out = make_output_()) {
        co_await ctx.write(0, std::move(out));
      }
    }
    bool all_eos = true;
    for (unsigned p : ports) {
      if (!ctx.eos(p)) { all_eos = false; break; }
    }
    if (all_eos || ctx.stop_requested()) { ctx.signal_done(); }
    co_return;
  }

  // Interactive: one bounded wait per call, so the runtime's process
  // loop keeps its grip on stop_requested and a reference image that
  // lands mid-wait is picked up on the next turn. One commit is one
  // beat -- the sequence number is what keeps that true when two
  // commits land inside the same window.
  const auto c = _channel->wait_commit(_seen_commit, kCommitPollMs);
  if (c.seq == _seen_commit || !c.png) { co_return; }
  _seen_commit = c.seq;

  vector<uint8_t> m;
  int cw = 0, ch = 0;
  if (!decode_commit_(*c.png, &m, &cw, &ch)) {
    session()->warn(fmt("create-mask('{}'): undecodable commit dropped",
                        this->id()));
    co_return;
  }
  refresh_canvas_(cw, ch);
  resample_mask_(m, cw, ch, _mw, _mh, &_mask);

  if (auto out = make_output_()) {
    co_await ctx.write(0, std::move(out));
  }
  // Re-latch, so an editor that mounts later opens on the mask that was
  // actually committed rather than the one the stage started from.
  publish_();
  co_return;
}

VPIPE_REGISTER_STAGE(CreateMaskStage)
VPIPE_REGISTER_SPEC(CreateMaskStage, kSpec)

}
