#include "stages/vision/detection-overlay-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/oport-policy.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/vision/detection-overlay-font5x7.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

// Golden-ratio increment in [0, 1) gives well-spaced hues for
// adjacent class ids, which is the visually-distinguishable
// property we want.
constexpr double kHueStep = 0.61803398874989485;

array<float, 3>
hsv_to_rgb_(double h, double s, double v)
{
  if (s <= 0.0) {
    auto vv = static_cast<float>(v);
    return { vv, vv, vv };
  }
  double hh = h - floor(h);  // wrap to [0, 1)
  hh *= 6.0;
  int    i = static_cast<int>(floor(hh));
  double f = hh - i;
  double p = v * (1.0 - s);
  double q = v * (1.0 - s * f);
  double t = v * (1.0 - s * (1.0 - f));
  switch (i % 6) {
    case 0: return { (float)v, (float)t, (float)p };
    case 1: return { (float)q, (float)v, (float)p };
    case 2: return { (float)p, (float)v, (float)t };
    case 3: return { (float)p, (float)q, (float)v };
    case 4: return { (float)t, (float)p, (float)v };
    default: return { (float)v, (float)p, (float)q };
  }
}

// Snap a floating-point bbox edge to a pixel index in [0, W-1].
int
clamp_(int v, int lo, int hi)
{
  if (v < lo) { return lo; }
  if (v > hi) { return hi; }
  return v;
}

// Format microseconds-since-UTC-epoch as
// "YYYY-MM-DD HH:MM:SS.fff" (millisecond precision). UTC when
// `use_utc`; otherwise the host's local time zone.
string
format_timestamp_us_(uint64_t ts_us, bool use_utc)
{
  const time_t   secs = static_cast<time_t>(ts_us / 1'000'000ull);
  const unsigned ms   =
      static_cast<unsigned>((ts_us / 1000ull) % 1000ull);
  struct tm tmv{};
  if (use_utc) {
    gmtime_r(&secs, &tmv);
  } else {
    localtime_r(&secs, &tmv);
  }
  char buf[40];
  std::snprintf(buf, sizeof buf,
                "%04d-%02d-%02d %02d:%02d:%02d.%03u",
                tmv.tm_year + 1900,
                tmv.tm_mon + 1,
                tmv.tm_mday,
                tmv.tm_hour,
                tmv.tm_min,
                tmv.tm_sec,
                ms);
  return string(buf);
}

string
format_score_(double s, int precision)
{
  if (precision < 0) { precision = 0; }
  if (precision > 6) { precision = 6; }
  char buf[32];
  snprintf(buf, sizeof buf, "%.*f", precision, s);
  return string(buf);
}

// Per-pixel-type Canvas. T is float (for F32 TensorBeats) or
// uint8_t (for U8 TensorBeats). Layout is plane-major: r-plane,
// then g-plane, then b-plane, each row-major H x W.
template <typename T>
struct Canvas {
  T*  data;
  int h;
  int w;
};

// Convert a normalized [0, 1] float RGB triple to the encoding of
// the destination buffer. For float buffers we honour
// `input_normalized`; for uint8_t buffers we always map to
// [0, 255] (input_normalized has no meaning for u8 — the bytes
// already are integers in the canonical range).
template <typename T>
array<T, 3>
scale_to_buffer_t_(const array<float, 3>& rgb,
                   bool                   input_normalized);

template <>
array<float, 3>
scale_to_buffer_t_<float>(const array<float, 3>& rgb,
                          bool                   input_normalized)
{
  if (input_normalized) {
    return rgb;
  }
  return { rgb[0] * 255.0f, rgb[1] * 255.0f, rgb[2] * 255.0f };
}

template <>
array<uint8_t, 3>
scale_to_buffer_t_<uint8_t>(const array<float, 3>& rgb,
                            bool                   /*input_normalized*/)
{
  auto to_byte = [](float v) -> uint8_t {
    float s = v * 255.0f;
    if (s <= 0.0f)   { return 0; }
    if (s >= 255.0f) { return 255; }
    return static_cast<uint8_t>(lroundf(s));
  };
  return { to_byte(rgb[0]), to_byte(rgb[1]), to_byte(rgb[2]) };
}

template <typename T>
void
set_pixel_t_(Canvas<T>& c, int x, int y, const array<T, 3>& rgb)
{
  if (x < 0 || y < 0 || x >= c.w || y >= c.h) {
    return;
  }
  const size_t plane = static_cast<size_t>(c.h) * c.w;
  const size_t off   = static_cast<size_t>(y) * c.w + x;
  c.data[off]              = rgb[0];
  c.data[plane     + off]  = rgb[1];
  c.data[2 * plane + off]  = rgb[2];
}

template <typename T>
void
fill_rect_t_(Canvas<T>& c, int x0, int y0, int x1, int y1,
             const array<T, 3>& rgb)
{
  if (x0 > x1 || y0 > y1) { return; }
  x0 = clamp_(x0, 0, c.w - 1);
  x1 = clamp_(x1, 0, c.w - 1);
  y0 = clamp_(y0, 0, c.h - 1);
  y1 = clamp_(y1, 0, c.h - 1);
  const size_t plane = static_cast<size_t>(c.h) * c.w;
  const size_t w_run = static_cast<size_t>(x1 - x0 + 1);
  // Three single-channel runs per row (one wide store stream each)
  // beat one pixel loop that interleaves three pointers. The three
  // plane pointers are non-aliasing slices of the same buffer; the
  // `__restrict__` qualifier makes that explicit to the optimiser.
  for (int y = y0; y <= y1; ++y) {
    T* __restrict__ row_r =
        c.data + static_cast<size_t>(y) * c.w + x0;
    T* __restrict__ row_g = row_r + plane;
    T* __restrict__ row_b = row_g + plane;
    if constexpr (std::is_same_v<T, std::uint8_t>) {
      std::memset(row_r, rgb[0], w_run);
      std::memset(row_g, rgb[1], w_run);
      std::memset(row_b, rgb[2], w_run);
    } else {
      std::fill_n(row_r, w_run, rgb[0]);
      std::fill_n(row_g, w_run, rgb[1]);
      std::fill_n(row_b, w_run, rgb[2]);
    }
  }
}

template <typename T>
void
draw_bbox_t_(Canvas<T>& c, int x1, int y1, int x2, int y2,
             const array<T, 3>& col, int thickness)
{
  fill_rect_t_(c, x1, y1, x2, y1 + thickness - 1, col);
  fill_rect_t_(c, x1, y2 - thickness + 1, x2, y2, col);
  fill_rect_t_(c, x1, y1, x1 + thickness - 1, y2, col);
  fill_rect_t_(c, x2 - thickness + 1, y1, x2, y2, col);
}

template <typename T>
void
draw_glyph_t_(Canvas<T>& c, int px, int py, char ch,
              const array<T, 3>& fg, int font_scale)
{
  const FontGlyph& g = font5x7_glyph(ch);
  for (int gy = 0; gy < kFontGlyphH; ++gy) {
    uint8_t row = g[gy];
    for (int gx = 0; gx < kFontGlyphW; ++gx) {
      if (row & (1u << (kFontGlyphW - 1 - gx))) {
        const int bx = px + gx * font_scale;
        const int by = py + gy * font_scale;
        for (int sy = 0; sy < font_scale; ++sy) {
          for (int sx = 0; sx < font_scale; ++sx) {
            set_pixel_t_(c, bx + sx, by + sy, fg);
          }
        }
      }
    }
  }
}

// Render a class-name-only tile into `out_bytes` (allocated to
// 3 × th × tw). Glyph foreground (white) composited on class bg.
// No surrounding padding; the consumer adds that as part of the
// label background fill it draws first.
template <typename T>
void
render_class_name_tile_t_(string_view             text,
                          const array<T, 3>&      bg,
                          const array<T, 3>&      fg,
                          int                     font_scale,
                          int&                    out_tw,
                          int&                    out_th,
                          std::vector<T>&         out_bytes)
{
  const int n_chars = static_cast<int>(text.size());
  const int glyph_w = kFontGlyphW * font_scale;
  const int glyph_h = kFontGlyphH * font_scale;
  out_tw = std::max(n_chars * glyph_w, 0);
  out_th = glyph_h;
  out_bytes.assign(
      static_cast<size_t>(3) * out_th * out_tw, T{});
  if (out_tw == 0 || out_th == 0) {
    return;
  }
  // Fill bg per channel (one wide store per row × plane).
  const size_t plane = static_cast<size_t>(out_th) * out_tw;
  for (int ch = 0; ch < 3; ++ch) {
    T* base = out_bytes.data() + ch * plane;
    if constexpr (std::is_same_v<T, std::uint8_t>) {
      std::memset(base, bg[ch], plane);
    } else {
      std::fill_n(base, plane, bg[ch]);
    }
  }
  // Draw glyphs into a synthetic canvas pointing at the tile bytes.
  Canvas<T> tile_canvas{ out_bytes.data(), out_th, out_tw };
  int px = 0;
  for (int i = 0; i < n_chars; ++i) {
    draw_glyph_t_(tile_canvas, px, 0, text[i], fg, font_scale);
    px += glyph_w;
  }
}

// Block-transfer a pre-rendered planar-RGB tile onto a Canvas at
// (dst_x, dst_y_top). The tile bytes live as 3 contiguous planes
// (R, G, B), each th × tw row-major. The blit copies row-by-row per
// channel, clipping at the canvas edges. Vastly cheaper than
// re-running the glyph rasteriser every frame.
template <typename T>
void
blit_tile_rgb_t_(Canvas<T>& c, int dst_x, int dst_y_top,
                 int tile_th, int tile_tw,
                 const T* __restrict__ tile_bytes)
{
  if (tile_th <= 0 || tile_tw <= 0) { return; }
  // Compute the clipped destination rect.
  int dx0 = dst_x;
  int dy0 = dst_y_top;
  int dx1 = dst_x + tile_tw - 1;
  int dy1 = dst_y_top + tile_th - 1;
  // Clip; track the source offsets that go with the clip.
  int src_x0 = 0, src_y0 = 0;
  if (dx0 < 0)         { src_x0 += -dx0; dx0 = 0; }
  if (dy0 < 0)         { src_y0 += -dy0; dy0 = 0; }
  if (dx1 > c.w - 1)   { dx1 = c.w - 1; }
  if (dy1 > c.h - 1)   { dy1 = c.h - 1; }
  if (dx0 > dx1 || dy0 > dy1) { return; }

  const size_t row_run   = static_cast<size_t>(dx1 - dx0 + 1);
  const size_t c_plane   = static_cast<size_t>(c.h) * c.w;
  const size_t t_plane   = static_cast<size_t>(tile_th) * tile_tw;
  for (int dy = dy0; dy <= dy1; ++dy) {
    const int sy = src_y0 + (dy - dy0);
    for (int ch = 0; ch < 3; ++ch) {
      T* __restrict__ dst_row =
          c.data + ch * c_plane
                 + static_cast<size_t>(dy) * c.w + dx0;
      const T* __restrict__ src_row =
          tile_bytes + ch * t_plane
                     + static_cast<size_t>(sy) * tile_tw + src_x0;
      if constexpr (std::is_same_v<T, std::uint8_t>) {
        std::memcpy(dst_row, src_row, row_run);
      } else {
        std::copy_n(src_row, row_run, dst_row);
      }
    }
  }
}

template <typename T>
void
draw_label_t_(Canvas<T>& c, int x, int y_top, string_view text,
              const array<T, 3>& bg, const array<T, 3>& fg,
              int font_scale, int label_padding)
{
  if (text.empty()) { return; }
  const int  pad     = label_padding;
  const int  n_chars = static_cast<int>(text.size());
  const int  glyph_w = kFontGlyphW * font_scale;
  const int  glyph_h = kFontGlyphH * font_scale;
  const int  bg_w    = n_chars * glyph_w + 2 * pad;
  const int  bg_h    = glyph_h + 2 * pad;

  fill_rect_t_(c, x, y_top, x + bg_w - 1, y_top + bg_h - 1, bg);

  int px = x + pad;
  const int py = y_top + pad;
  for (int i = 0; i < n_chars; ++i) {
    if (px + glyph_w > c.w) { break; }
    draw_glyph_t_(c, px, py, text[i], fg, font_scale);
    px += glyph_w;
  }
}

// Render `text` anchored to the bottom-right of the canvas, with a
// filled background rectangle and a small inset from the frame edge.
// When the rendered text is wider than the canvas, the left edge is
// clamped to 0 so glyphs are clipped against the right edge instead
// of disappearing entirely.
template <typename T>
void
draw_bottom_right_label_t_(Canvas<T>& c, string_view text,
                           const array<T, 3>& bg,
                           const array<T, 3>& fg,
                           int font_scale, int label_padding,
                           int inset, int y_extra = 0)
{
  if (text.empty()) { return; }
  const int n_chars = static_cast<int>(text.size());
  const int glyph_w = kFontGlyphW * font_scale;
  const int glyph_h = kFontGlyphH * font_scale;
  const int bg_w    = n_chars * glyph_w + 2 * label_padding;
  const int bg_h    = glyph_h + 2 * label_padding;
  int x_left = c.w - bg_w - inset;
  // `y_extra` lifts the label off the bottom edge by that many pixels
  // so a second line (the audio class) can stack above the timestamp.
  int y_top  = c.h - bg_h - inset - y_extra;
  if (x_left < 0) { x_left = 0; }
  if (y_top  < 0) { y_top  = 0; }
  draw_label_t_(c, x_left, y_top, text, bg, fg,
                font_scale, label_padding);
}

// Top (highest-score) class label from an audio-tagging FlexData
// object -- but only when that top tag's score is strictly greater than
// `min_score`. The "tags" array is emitted sorted by descending score,
// so element 0 is the dominant class. Empty on a missing/malformed
// payload, or when the dominant class is below the display threshold.
string
audio_top_label_(const FlexData& fd, double min_score)
{
  if (!fd.is_object()) { return {}; }
  auto root = fd.as_object();
  if (!root.contains("tags")) { return {}; }
  FlexData tags = root.at("tags");
  if (!tags.is_array()) { return {}; }
  auto arr = tags.as_array();
  if (arr.empty()) { return {}; }
  FlexData t0 = arr.at(0);
  if (!t0.is_object()) { return {}; }
  auto to = t0.as_object();
  if (!to.contains("label")) { return {}; }
  const double score = to.contains("score")
      ? to.at("score").as_real(0.0) : 0.0;
  if (score <= min_score) { return {}; }
  return string(to.at("label").as_string(""));
}

}  // namespace

DetectionOverlayStage::DetectionOverlayStage(
    const SessionContextIntf* s,
    string                    id,
    vector<InEdge>            iports,
    FlexData                  config)
  : TypedStage<DetectionOverlayStage>(s, std::move(id),
                                      std::move(iports),
                                      std::move(config))
{
  // Scalar attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default. Clamps repair out-of-range
  // overrides.
  _top_n = static_cast<int>(attr_int("top_n"));
  if (_top_n < 0) { _top_n = 0; }
  _score_threshold = attr_real("score_threshold");
  _box_thickness = static_cast<int>(attr_int("box_thickness"));
  if (_box_thickness < 1)  { _box_thickness = 1; }
  if (_box_thickness > 16) { _box_thickness = 16; }
  _font_scale = static_cast<int>(attr_int("font_scale"));
  if (_font_scale < 1) { _font_scale = 1; }
  if (_font_scale > 8) { _font_scale = 8; }
  _label_padding = static_cast<int>(attr_int("label_padding"));
  if (_label_padding < 0)  { _label_padding = 0; }
  if (_label_padding > 32) { _label_padding = 32; }
  _draw_label = attr_bool("draw_label");
  _draw_score = attr_bool("draw_score");
  _score_precision = static_cast<int>(attr_int("score_precision"));
  if (_score_precision < 0) { _score_precision = 0; }
  if (_score_precision > 6) { _score_precision = 6; }
  _input_normalized = attr_bool("input_normalized");
  _draw_timestamp = attr_bool("draw_timestamp");
  _timestamp_use_local = attr_bool("timestamp_use_local");
  _audio_label_threshold = attr_real("audio_label_threshold");
  if (_audio_label_threshold < 0.0) { _audio_label_threshold = 0.0; }
  if (_audio_label_threshold > 1.0) { _audio_label_threshold = 1.0; }
  _oport_capacity = static_cast<unsigned>(attr_uint("oport_capacity"));

  // class_names: composite array, parsed from config directly.
  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("class_names")) {
      auto arr = root.at("class_names");
      if (arr.is_array()) {
        auto av = arr.as_array();
        _class_names.reserve(av.size());
        for (size_t i = 0, n = av.size(); i < n; ++i) {
          _class_names.emplace_back(av.at(i).as_string(""));
        }
      } else {
        session()->warn(fmt(
          "detection-overlay('{}'): class_names is not an array; "
          "ignoring", this->id()));
      }
    }
  }
  if (_oport_capacity == 0) {
    _oport_capacity = 4;
  }

  allocate_oports(spec().oports.size());
  set_oport_policy(0, { _oport_capacity, OverrunPolicy::Backpressure });
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "top_n", .type = ConfigType::Int, .doc = "max boxes to draw; 0 = all",
   .def_int = 10},
  {.key = "score_threshold", .type = ConfigType::Real,
   .doc = "drop detections below this score", .def_real = 0.0},
  {.key = "box_thickness", .type = ConfigType::Int,
   .doc = "border px, clamped [1,16]", .def_int = 2},
  {.key = "font_scale", .type = ConfigType::Int,
   .doc = "integer glyph scale, clamped [1,8]", .def_int = 2},
  {.key = "label_padding", .type = ConfigType::Int,
   .doc = "px around label glyphs, clamped [0,32]", .def_int = 2},
  {.key = "draw_label", .type = ConfigType::Bool, .def_bool = true},
  {.key = "draw_score", .type = ConfigType::Bool, .def_bool = true},
  {.key = "score_precision", .type = ConfigType::Int,
   .doc = "digits after decimal, clamped [0,6]", .def_int = 2},
  {.key = "input_normalized", .type = ConfigType::Bool,
   .doc = "F32 only; ignored for U8", .def_bool = true},
  {.key = "draw_timestamp", .type = ConfigType::Bool,
   .doc = "render sideband timestamp in bottom-right", .def_bool = false},
  {.key = "timestamp_use_local", .type = ConfigType::Bool,
   .doc = "format timestamp in local time vs UTC", .def_bool = false},
  {.key = "audio_label_threshold", .type = ConfigType::Real,
   .doc = "only show the audio tag when its score exceeds this, [0,1]",
   .def_real = 0.6},
  {.key = "class_names", .type = ConfigType::Array,
   .doc = "label lookup keyed by class_id"},
  {.key = "oport_capacity", .type = ConfigType::Uint,
   .doc = "output ring capacity", .def_uint = 4},
};
const PortSpec kIports[] = {
  {.name = "frames", .doc = "RGB TensorBeat [3,H,W] (F32 or U8)",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "detections", .doc = "FlexData YOLO/tracker detections, 1:1 "
                                "with frames",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "audio_tags", .doc = "optional FlexData audio-tagging stream "
                                "(separate clock, timestamp-gated)",
   .type = &typeid(FlexDataPayload), .clock_group = 1},
};
const PortSpec kOports[] = {
  {.name = "frames", .doc = "TensorBeat with boxes + labels drawn "
                            "(same dtype, contiguous)",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "detection-overlay",
  .doc       = "Draws top-N detection boxes + labels (and an optional "
               "audio tag + timestamp) onto each RGB frame. iport0/1 + "
               "oport0 share the video clock; iport2 (audio) is separate.",
  .display_name = "Detection Overlay",
  .category  = StageCategory::Video,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
DetectionOverlayStage::spec() const noexcept
{
  return kSpec;
}

namespace {

// Deterministic palette keyed on an arbitrary non-negative integer
// (class_id or track_id). Same key always produces the same RGB.
// Used both by class-coloured overlays (when there's no tracker
// upstream) and by track-coloured overlays (when each detection
// carries a stable track_id across frames).
array<float, 3>
color_for_key_(int key)
{
  if (key < 0) { key = 0; }
  double h = kHueStep * static_cast<double>(key);
  h -= floor(h);
  return hsv_to_rgb_(h, 0.85, 1.0);
}

}  // namespace

// Picks the right per-dtype label-tile cache off the stage. We can't
// templatize unordered_map<int, TileT> as a single dependent type
// without leaking the tile struct templates into the header; instead
// the stage exposes two concrete maps and we route by `T`.
template <typename T>
static auto&
stage_cache_(DetectionOverlayStage* s);

template <>
auto&
stage_cache_<std::uint8_t>(DetectionOverlayStage* s)
{
  return s->u8_label_tiles();
}

template <>
auto&
stage_cache_<float>(DetectionOverlayStage* s)
{
  return s->f32_label_tiles();
}

// Templated overlay-drawing routine. Walks the detection list once
// (filter -> sort -> top-N -> draw), invoking the per-pixel-type
// drawing helpers. Defined as a free template here so it gets
// instantiated once per pixel type at the bottom of the file.
template <typename T>
static void
draw_overlays_impl_(Canvas<T>&                       c,
                    const FlexData&                  fd,
                    DetectionOverlayStage*           stage,
                    int                              top_n,
                    double                           score_threshold,
                    int                              box_thickness,
                    int                              font_scale,
                    int                              label_padding,
                    int                              score_precision,
                    bool                             draw_label,
                    bool                             draw_score,
                    bool                             input_normalized,
                    const std::vector<std::string>&  class_names,
                    int&                             last_warn_w,
                    int&                             last_warn_h,
                    const std::string&               stage_id)
{
  if (!fd.is_object()) { return; }
  auto root = fd.as_object();
  if (!root.contains("detections")) { return; }
  FlexData dets_fd = root.at("detections");
  if (!dets_fd.is_array()) { return; }
  auto dets = dets_fd.as_array();

  double sx = 1.0, sy = 1.0;
  if (root.contains("frame_width") && root.contains("frame_height")) {
    int fw = static_cast<int>(root.at("frame_width").as_int(c.w));
    int fh = static_cast<int>(root.at("frame_height").as_int(c.h));
    if (fw > 0 && fh > 0 && (fw != c.w || fh != c.h)) {
      if (fw != last_warn_w || fh != last_warn_h) {
        stage->session()->warn(fmt(
            "detection-overlay('{}'): FlexData frame {}x{} != tensor "
            "{}x{} — rescaling bbox coords",
            stage_id, fw, fh, c.w, c.h));
        last_warn_w = fw;
        last_warn_h = fh;
      }
      sx = static_cast<double>(c.w) / static_cast<double>(fw);
      sy = static_cast<double>(c.h) / static_cast<double>(fh);
    }
  }

  struct Pick { size_t idx; double score; };
  vector<Pick> picks;
  picks.reserve(dets.size());
  for (size_t i = 0; i < dets.size(); ++i) {
    FlexData entry = dets.at(i);
    if (!entry.is_object()) { continue; }
    double score = entry.as_object().contains("score")
                 ? entry.as_object().at("score").as_real(0.0)
                 : 0.0;
    if (score < score_threshold) { continue; }
    picks.push_back({ i, score });
  }
  sort(picks.begin(), picks.end(),
       [](const Pick& a, const Pick& b) { return a.score > b.score; });
  size_t keep = picks.size();
  if (top_n > 0 && keep > static_cast<size_t>(top_n)) {
    keep = static_cast<size_t>(top_n);
  }

  // Track-keyed (positive) cache entries drawn this frame. Used after
  // the loop to evict tiles for objects that are no longer present so
  // the cache can't grow unbounded as the upstream tracker mints new
  // track_ids over a long-lived stream.
  std::vector<int> live_track_keys;
  live_track_keys.reserve(keep);

  for (size_t k = 0; k < keep; ++k) {
    FlexData entry = dets.at(picks[k].idx);
    auto d = entry.as_object();
    int    cls = static_cast<int>(
        d.contains("class_id") ? d.at("class_id").as_int(-1) : -1);
    // Tracker upstream annotates each detection with a stable
    // integer `track_id`. When present, color and label are keyed
    // on it (so the same physical object keeps its colour across
    // frames even if its class_id shifts). When absent, fall back
    // to class-coloured rendering.
    int    track_id = static_cast<int>(
        d.contains("track_id") ? d.at("track_id").as_int(-1) : -1);
    double x1  = d.contains("x1") ? d.at("x1").as_real(0.0) : 0.0;
    double y1  = d.contains("y1") ? d.at("y1").as_real(0.0) : 0.0;
    double x2  = d.contains("x2") ? d.at("x2").as_real(0.0) : 0.0;
    double y2  = d.contains("y2") ? d.at("y2").as_real(0.0) : 0.0;
    x1 *= sx; x2 *= sx;
    y1 *= sy; y2 *= sy;

    int ix1 = clamp_(static_cast<int>(lround(x1)), 0, c.w - 1);
    int iy1 = clamp_(static_cast<int>(lround(y1)), 0, c.h - 1);
    int ix2 = clamp_(static_cast<int>(lround(x2)), 0, c.w - 1);
    int iy2 = clamp_(static_cast<int>(lround(y2)), 0, c.h - 1);
    if (ix2 <= ix1 || iy2 <= iy1) {
      continue;  // degenerate
    }

    const int  color_key = (track_id >= 0) ? track_id : cls;
    const auto rgb01     = color_for_key_(color_key);
    const auto rgb_t     = scale_to_buffer_t_<T>(rgb01, input_normalized);
    draw_bbox_t_(c, ix1, iy1, ix2, iy2, rgb_t, box_thickness);

    if (draw_label) {
      // Label-text precedence: per-detection class_name from
      // FlexData, then the stage's class_names config (keyed by
      // class_id), then a "cls<id>" fallback. When the detection
      // carries a track_id we prepend "#N " so each tracked object
      // is unambiguous on screen.
      string class_text;
      if (d.contains("class_name")) {
        class_text.assign(d.at("class_name").as_string(""));
      }
      if (class_text.empty()
          && cls >= 0
          && cls < static_cast<int>(class_names.size())
          && !class_names[cls].empty()) {
        class_text = class_names[cls];
      }
      if (class_text.empty()) {
        class_text = "cls" + to_string(cls);
      }
      if (track_id >= 0) {
        class_text = "#" + to_string(track_id) + " " + class_text;
      }

      string score_str;
      if (draw_score) {
        score_str = format_score_(picks[k].score, score_precision);
      }

      const int glyph_w = kFontGlyphW * font_scale;
      const int glyph_h = kFontGlyphH * font_scale;
      const int class_w = static_cast<int>(class_text.size()) * glyph_w;
      const int sep_w   = score_str.empty() ? 0 : glyph_w;
      const int score_w = static_cast<int>(score_str.size()) * glyph_w;
      const int bg_w    = class_w + sep_w + score_w + 2 * label_padding;
      const int bg_h    = glyph_h + 2 * label_padding;

      int y_top = iy1 - bg_h;
      if (y_top < 0) { y_top = iy1; }

      const array<float, 3> fg01{ 1.0f, 1.0f, 1.0f };
      const auto fg_t = scale_to_buffer_t_<T>(fg01, input_normalized);

      // 1. Background fill across the full label area.
      fill_rect_t_(c, ix1, y_top, ix1 + bg_w - 1,
                                  y_top + bg_h - 1, rgb_t);

      // 2. Blit the pre-rendered class-name tile. Cached per
      // visible-label identity: keyed on track_id when present
      // (so each tracked object owns one cache slot) or on
      // class_id otherwise. Track-keyed entries get a positive
      // key, class-keyed entries get a negative key, so the two
      // namespaces never collide. The text-mismatch check still
      // covers the config-change case.
      const int cache_key =
          (track_id >= 0) ? track_id : -(cls + 1);
      if (cache_key >= 0) {
        live_track_keys.push_back(cache_key);
      }
      auto& cache = stage_cache_<T>(stage);
      auto it = cache.find(cache_key);
      bool need_render =
          (it == cache.end()) || (it->second.text != class_text);
      if (need_render) {
        typename std::decay_t<decltype(cache)>::mapped_type tile;
        tile.text = class_text;
        render_class_name_tile_t_<T>(
            class_text, rgb_t, fg_t, font_scale,
            tile.tw, tile.th, tile.bytes);
        it = cache.insert_or_assign(cache_key,
                                    std::move(tile)).first;
      }
      const auto& tile = it->second;
      blit_tile_rgb_t_<T>(c,
                          ix1 + label_padding,
                          y_top + label_padding,
                          tile.th, tile.tw, tile.bytes.data());

      // 3. Draw the score portion per-frame (digits + decimal point
      // change every frame, so caching wouldn't help). Render at
      // the right side of the class-name tile, preceded by one
      // glyph-width separator gap.
      if (!score_str.empty()) {
        int px = ix1 + label_padding + class_w + sep_w;
        const int py = y_top + label_padding;
        const int n = static_cast<int>(score_str.size());
        for (int i = 0; i < n; ++i) {
          if (px + glyph_w > c.w) { break; }
          draw_glyph_t_(c, px, py, score_str[i], fg_t, font_scale);
          px += glyph_w;
        }
      }
    }
  }

  // Evict track-keyed (positive) tiles for objects not drawn this
  // frame. Without this the cache gains one entry per distinct
  // track_id the upstream tracker ever emits and never releases it --
  // a slow but unbounded leak on an indefinite stream. Class-keyed
  // (negative) tiles are bounded by the class count, so keep them.
  // Runs only when labels are drawn (the only path that fills the
  // cache); with `keep == 0` this evicts every track tile, which is
  // correct (no tracked objects are visible).
  if (draw_label) {
    auto& cache = stage_cache_<T>(stage);
    for (auto it = cache.begin(); it != cache.end(); ) {
      if (it->first >= 0
          && std::find(live_track_keys.begin(),
                       live_track_keys.end(),
                       it->first) == live_track_keys.end()) {
        it = cache.erase(it);
      } else {
        ++it;
      }
    }
  }
}

Job
DetectionOverlayStage::consume_audio_(RuntimeContext& ctx,
                                      uint64_t        boundary_ts_us)
{
  if (ctx.num_iports() < 3 || _audio_eos) {
    co_return;
  }
  // Peek the backlog in arrival (== timestamp) order; consume every
  // entry strictly before the frame's timestamp and adopt the most
  // recent window's top class as the current label. A window whose
  // tags are all below the threshold yields an empty label and CLEARS
  // the current one (the field is then skipped), so a stale label
  // doesn't linger once the sound that produced it has stopped. Leave
  // anything at/after the frame time queued.
  const std::uint32_t avail = ctx.backlog(2);
  std::uint32_t consumed = 0;
  for (std::uint32_t i = 0; i < avail; ++i) {
    const BeatPayloadIntf* p = co_await ctx.peek(2, i);
    if (!p) {
      _audio_eos = true;   // iport2 closed; audio is auxiliary.
      break;
    }
    const auto* fdp = dynamic_cast<const FlexDataPayload*>(p);
    if (fdp && fdp->data.is_object()) {
      auto root = fdp->data.as_object();
      const uint64_t ts = root.contains("timestamp_us")
          ? root.at("timestamp_us").as_uint(0)
          : 0;
      if (ts >= boundary_ts_us) {
        break;
      }
      // Adopt this window's gated top class. An empty result (all tags
      // below the threshold) clears the current label so the overlay
      // field is dropped rather than showing a stale tag.
      _cur_audio_label = audio_top_label_(fdp->data, _audio_label_threshold);
    }
    consumed = i + 1;
  }
  if (consumed > 0) {
    ctx.release_read(2, consumed);
  }
  co_return;
}

Job
DetectionOverlayStage::process(RuntimeContext& ctx)
{
  auto img_opt = co_await ctx.read(0);
  if (!img_opt) {
    ctx.signal_done();
    co_return;
  }
  auto det_opt = co_await ctx.read(1);
  if (!det_opt) {
    ctx.signal_done();
    co_return;
  }

  const auto* tin = dynamic_cast<const TensorBeatPayload*>(img_opt.get());
  const auto* fdp = dynamic_cast<const FlexDataPayload*>(det_opt.get());
  const FlexData* fd = fdp ? &fdp->data : nullptr;
  if (!tin || tin->shape.size() != 3 || tin->shape[0] != 3) {
    session()->warn(fmt(
        "detection-overlay('{}'): iport0 not a [3,H,W] TensorBeat — "
        "dropping frame", this->id()));
    co_return;
  }
  if (tin->dtype != TensorBeat::DType::F32
      && tin->dtype != TensorBeat::DType::U8) {
    session()->warn(fmt(
        "detection-overlay('{}'): unsupported dtype '{}' — only "
        "f32/u8 are drawable; dropping frame",
        this->id(), tin->dtype_name()));
    co_return;
  }
  if (!fd) {
    session()->warn(fmt(
        "detection-overlay('{}'): iport1 not a FlexData — emitting "
        "frame without overlay", this->id()));
  }

  TensorBeat tout;
  tout.dtype = tin->dtype;
  tout.shape = tin->shape;
  if (tin->is_contiguous()) {
    // Always materialise into a CpuCached AlignedVector here -- the
    // overlay mutates the bytes in place (drawing) so it needs its
    // own writable buffer. materialize_contiguous() reads through
    // bytes_() and handles both CpuCached and Shared inputs.
    tout.data = tin->materialize_contiguous();
    const size_t expected_bytes =
        static_cast<size_t>(tin->shape[0])
      * static_cast<size_t>(tin->shape[1])
      * static_cast<size_t>(tin->shape[2])
      * tin->element_byte_size();
    if (tout.data.size() != expected_bytes) {
      // Defensive: shouldn't fire for is_contiguous inputs.
      tout.data = tin->materialize_contiguous();
    }
  } else {
    tout.data = tin->materialize_contiguous();
  }
  // tout.strides stays empty -> output is row-major contiguous.

  const int H = static_cast<int>(tin->shape[1]);
  const int W = static_cast<int>(tin->shape[2]);

  // Frame timestamp drives both the drawn clock and the audio gate.
  std::uint64_t frame_ts_us = 0;
  bool          frame_has_ts = false;
  if (tin->sideband.is_object()) {
    auto sb = tin->sideband.as_object();
    if (sb.contains("timestamp_us")) {
      frame_ts_us  = sb.at("timestamp_us").as_uint(0);
      frame_has_ts = true;
    }
  }
  // iport2 (audio, separate clock domain): pull every classification up
  // to this frame's timestamp; the latest top class becomes the label
  // drawn above the timestamp. When the frame has no timestamp we can't
  // gate, so drain whatever is queued (UINT64_MAX boundary).
  co_await consume_audio_(ctx,
      frame_has_ts ? frame_ts_us : UINT64_MAX);

  if (fd) {
    if (tout.dtype == TensorBeat::DType::F32) {
      Canvas<float> canvas{ tout.as_f32(), H, W };
      draw_overlays_impl_<float>(
          canvas, *fd, this,
          _top_n, _score_threshold, _box_thickness, _font_scale,
          _label_padding, _score_precision,
          _draw_label, _draw_score, _input_normalized,
          _class_names,
          _last_warn_w, _last_warn_h, this->id());
    } else {
      Canvas<uint8_t> canvas{ tout.as_u8(), H, W };
      draw_overlays_impl_<uint8_t>(
          canvas, *fd, this,
          _top_n, _score_threshold, _box_thickness, _font_scale,
          _label_padding, _score_precision,
          _draw_label, _draw_score, _input_normalized,
          _class_names,
          _last_warn_w, _last_warn_h, this->id());
    }
  }

  if (_draw_timestamp && frame_has_ts) {
    const string text =
        format_timestamp_us_(frame_ts_us, !_timestamp_use_local);
    constexpr int kInset = 4;
    // Black background, white foreground — readable across most
    // scene colours without configurable tuning.
    const array<float, 3> bg01{ 0.0f, 0.0f, 0.0f };
    const array<float, 3> fg01{ 1.0f, 1.0f, 1.0f };
    // The audio class (when present) stacks one text row above the
    // timestamp, right-aligned to the same edge.
    const int row_h =
        kFontGlyphH * _font_scale + 2 * _label_padding;
    const int audio_y_extra = row_h + 2;
    const bool have_audio = !_cur_audio_label.empty();
    if (tout.dtype == TensorBeat::DType::F32) {
      Canvas<float> canvas{ tout.as_f32(), H, W };
      const auto bg_t =
          scale_to_buffer_t_<float>(bg01, _input_normalized);
      const auto fg_t =
          scale_to_buffer_t_<float>(fg01, _input_normalized);
      draw_bottom_right_label_t_<float>(
          canvas, text, bg_t, fg_t,
          _font_scale, _label_padding, kInset);
      if (have_audio) {
        draw_bottom_right_label_t_<float>(
            canvas, _cur_audio_label, bg_t, fg_t,
            _font_scale, _label_padding, kInset, audio_y_extra);
      }
    } else {
      Canvas<uint8_t> canvas{ tout.as_u8(), H, W };
      const auto bg_t =
          scale_to_buffer_t_<uint8_t>(bg01, false);
      const auto fg_t =
          scale_to_buffer_t_<uint8_t>(fg01, false);
      draw_bottom_right_label_t_<uint8_t>(
          canvas, text, bg_t, fg_t,
          _font_scale, _label_padding, kInset);
      if (have_audio) {
        draw_bottom_right_label_t_<uint8_t>(
            canvas, _cur_audio_label, bg_t, fg_t,
            _font_scale, _label_padding, kInset, audio_y_extra);
      }
    }
  }

  // Propagate the input sideband to the output so downstream stages
  // see the timestamp + any other metadata; the overlay does not own
  // this data, it just decorates the pixels.
  tout.sideband = tin->sideband;

  co_await ctx.write(0,
      make_payload<TensorBeatPayload>(std::move(tout)));
}

VPIPE_REGISTER_STAGE(DetectionOverlayStage)
VPIPE_REGISTER_SPEC(DetectionOverlayStage, kSpec)

}
