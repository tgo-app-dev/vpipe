#include "stages/audio-video/content-aware-continuation-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

constexpr unsigned kFramesPort = 0;
constexpr unsigned kRefPort    = 1;

// Block matching. Blocks sit on a grid over the reference; each is
// searched for in the continuation, coarsely on a half-size copy and
// then at full size, and a block whose best match is weak is not
// evidence. `kMinNcc` is high on purpose: this is for two frames of the
// same held shot, and a block that matches at 0.8 is a block that moved.
constexpr int    kGridX      = 12;
constexpr int    kGridY      = 8;
constexpr double kMinNcc     = 0.90;
constexpr double kMinStd     = 2.0;    // codes; flat blocks carry no shift
constexpr int    kMinBlocks  = 8;

// Tone: 32 bins of 8 codes on the continuation's level, a bin needs this
// many pixels before its median is trusted.
constexpr int kBins        = 32;
constexpr int kMinBinCount = 200;

float
luma_(float r, float g, float b)
{
  return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

vector<float>
luma_plane_(const vector<float>& rgb, int w, int h)
{
  const size_t n = (size_t)w * h;
  vector<float> y(n);
  for (size_t i = 0; i < n; ++i) {
    y[i] = luma_(rgb[i], rgb[n + i], rgb[2 * n + i]);
  }
  return y;
}

vector<float>
half_(const vector<float>& p, int w, int h, int* ow, int* oh)
{
  const int w2 = w / 2, h2 = h / 2;
  vector<float> o((size_t)w2 * h2);
  for (int y = 0; y < h2; ++y) {
    for (int x = 0; x < w2; ++x) {
      const size_t a = (size_t)(2 * y) * w + 2 * x;
      o[(size_t)y * w2 + x] = 0.25f * (p[a] + p[a + 1] + p[a + w] +
                                       p[a + w + 1]);
    }
  }
  *ow = w2;
  *oh = h2;
  return o;
}

// NCC of the bs x bs block of `a` at (x0, y0) against `b` at the same
// block moved by (dx, dy). -2 when a block leaves the frame or is flat.
double
ncc_(const vector<float>& a, const vector<float>& b, int w, int h, int x0,
     int y0, int bs, int dx, int dy)
{
  if (x0 + dx < 0 || y0 + dy < 0 || x0 + dx + bs > w || y0 + dy + bs > h) {
    return -2.0;
  }
  double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (int y = 0; y < bs; ++y) {
    const float* pa = &a[(size_t)(y0 + y) * w + x0];
    const float* pb = &b[(size_t)(y0 + dy + y) * w + x0 + dx];
    for (int x = 0; x < bs; ++x) {
      const double u = pa[x], v = pb[x];
      sa += u; sb += v; saa += u * u; sbb += v * v; sab += u * v;
    }
  }
  const double n = (double)bs * bs;
  const double va = saa - sa * sa / n, vb = sbb - sb * sb / n;
  if (va <= 1e-9 || vb <= 1e-9) { return -2.0; }
  return (sab - sa * sb / n) / std::sqrt(va * vb);
}

double
block_std_(const vector<float>& a, int w, int x0, int y0, int bs)
{
  double s = 0, ss = 0;
  for (int y = 0; y < bs; ++y) {
    for (int x = 0; x < bs; ++x) {
      const double v = a[(size_t)(y0 + y) * w + x0 + x];
      s += v; ss += v * v;
    }
  }
  const double n = (double)bs * bs;
  return std::sqrt(std::max(0.0, ss / n - (s / n) * (s / n)));
}

// Exhaustive search of offsets within `rad` of (cx, cy), step 1.
void
search_(const vector<float>& a, const vector<float>& b, int w, int h, int x0,
        int y0, int bs, int cx, int cy, int rad, double* best, int* bx,
        int* by)
{
  *best = -2.0;
  *bx = cx;
  *by = cy;
  for (int dy = cy - rad; dy <= cy + rad; ++dy) {
    for (int dx = cx - rad; dx <= cx + rad; ++dx) {
      const double r = ncc_(a, b, w, h, x0, y0, bs, dx, dy);
      if (r > *best) { *best = r; *bx = dx; *by = dy; }
    }
  }
}

// Sub-pixel vertex of a parabola through three samples; 0 when flat.
double
parabola_(double l, double c, double r)
{
  const double d = l - 2.0 * c + r;
  if (std::fabs(d) < 1e-12) { return 0.0; }
  const double v = 0.5 * (l - r) / d;
  return std::max(-0.5, std::min(0.5, v));
}

// Least squares d = k * p + t, dropping outliers once.
bool
fit_axis_(const vector<double>& p, const vector<double>& d,
          const vector<double>& wt, double* k, double* t)
{
  vector<char> use(p.size(), 1);
  for (int pass = 0; pass < 2; ++pass) {
    double sw = 0, sp = 0, sd = 0, spp = 0, spd = 0;
    for (size_t i = 0; i < p.size(); ++i) {
      if (!use[i]) { continue; }
      sw += wt[i]; sp += wt[i] * p[i]; sd += wt[i] * d[i];
      spp += wt[i] * p[i] * p[i]; spd += wt[i] * p[i] * d[i];
    }
    const double det = sw * spp - sp * sp;
    if (sw <= 0 || std::fabs(det) < 1e-9) { return false; }
    *k = (sw * spd - sp * sd) / det;
    *t = (sd - *k * sp) / sw;
    if (pass == 1) { break; }
    // Residuals, then a MAD gate: a block that followed something moving
    // in the scene is not part of the frame's geometry.
    vector<double> res;
    for (size_t i = 0; i < p.size(); ++i) {
      res.push_back(std::fabs(d[i] - (*k * p[i] + *t)));
    }
    vector<double> s = res;
    std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
    const double gate = std::max(1.0, 3.0 * 1.4826 * s[s.size() / 2]);
    for (size_t i = 0; i < p.size(); ++i) { use[i] = res[i] <= gate; }
  }
  return true;
}

// Pool-adjacent-violators: the weighted monotone non-decreasing fit.
void
isotonic_(vector<double>* y, const vector<double>& w)
{
  struct Blk { double v, w; int n; };
  vector<Blk> st;
  for (size_t i = 0; i < y->size(); ++i) {
    st.push_back({(*y)[i], w[i], 1});
    while (st.size() > 1 && st[st.size() - 2].v > st.back().v) {
      Blk b = st.back(); st.pop_back();
      Blk& a = st.back();
      a.v = (a.v * a.w + b.v * b.w) / (a.w + b.w);
      a.w += b.w;
      a.n += b.n;
    }
  }
  size_t i = 0;
  for (const Blk& b : st) {
    for (int j = 0; j < b.n; ++j) { (*y)[i++] = b.v; }
  }
}

float
sample_(const float* p, int w, int h, double x, double y)
{
  x = std::max(0.0, std::min((double)w - 1, x));
  y = std::max(0.0, std::min((double)h - 1, y));
  const int x0 = (int)x, y0 = (int)y;
  const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const float fx = (float)(x - x0), fy = (float)(y - y0);
  const float a = p[(size_t)y0 * w + x0], b = p[(size_t)y0 * w + x1];
  const float c = p[(size_t)y1 * w + x0], d = p[(size_t)y1 * w + x1];
  return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
}

float
lut_at_(const std::array<float, 256>& lut, float v)
{
  v = std::max(0.0f, std::min(255.0f, v));
  const int i = (int)v;
  if (i >= 255) { return lut[255]; }
  const float f = v - (float)i;
  return lut[(size_t)i] + (lut[(size_t)i + 1] - lut[(size_t)i]) * f;
}

// A beat as planar float on a 0..255 scale. [3,H,W] or the frames
// [first, first + count) of a [T,3,H,W] clip, averaged.
bool
to_planar_(const TensorBeatPayload& t, int first, int count,
           vector<float>* out, int* w, int* h)
{
  const bool clip = t.shape.size() == 4;
  if (!(clip || t.shape.size() == 3)) { return false; }
  const int c = (int)t.shape[clip ? 1 : 0];
  if (c != 3) { return false; }
  *h = (int)t.shape[clip ? 2 : 1];
  *w = (int)t.shape[clip ? 3 : 2];
  const size_t plane = (size_t)(*w) * (*h), frame = 3 * plane;
  const int frames = clip ? (int)t.shape[0] : 1;
  if (!clip) { first = 0; count = 1; }
  first = std::max(0, std::min(first, frames - 1));
  count = std::max(1, std::min(count, frames - first));
  const bool u8 = t.dtype == TensorBeat::DType::U8;
  if (!u8 && t.dtype != TensorBeat::DType::F32) { return false; }
  const auto bytes = t.materialize_contiguous();
  out->assign(frame, 0.0f);
  for (int f = first; f < first + count; ++f) {
    for (size_t i = 0; i < frame; ++i) {
      const size_t k = (size_t)f * frame + i;
      (*out)[i] += u8 ? (float)bytes[k]
                      : reinterpret_cast<const float*>(bytes.data())[k] *
                            255.0f;
    }
  }
  for (float& v : *out) { v /= (float)count; }
  return true;
}

void
resize_(const vector<float>& src, int sw, int sh, int dw, int dh,
        vector<float>* out)
{
  out->resize((size_t)3 * dw * dh);
  for (int c = 0; c < 3; ++c) {
    const float* p = &src[(size_t)c * sw * sh];
    for (int y = 0; y < dh; ++y) {
      for (int x = 0; x < dw; ++x) {
        (*out)[(size_t)c * dw * dh + (size_t)y * dw + x] = sample_(
            p, sw, sh, (x + 0.5) * sw / dw - 0.5, (y + 0.5) * sh / dh - 0.5);
      }
    }
  }
}

const PortSpec kIports[] = {
  {.name = "frames",
   .doc = "the continuation's frames, planar RGB [3,H,W] u8 or f32, in "
          "order -- what vae-decode emits",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
  {.name = "reference",
   .doc = "the clip being continued: a stacked [T,3,H,W] clip (what "
          "temporal-stack emits for a guide) or one [3,H,W] frame. Only "
          "its LAST `window` frames are read",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "frames", .doc = "the same frames, same shape and dtype, with "
                            "the seam's tone and geometry taken out",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};

constexpr ConfigKey kAttrs[] = {
  {.key = "tone", .type = ConfigType::Bool, .required = false,
   .doc = "match the continuation's tone to the reference: one monotone "
          "curve per channel, fitted to the aligned pixels either side of "
          "the seam. A curve and not a gain and offset, because the step "
          "a continuation takes is largest in the shadows",
   .def_bool = true},
  {.key = "geometry", .type = ConfigType::Bool, .required = false,
   .doc = "match the continuation's geometry to the reference: a scale "
          "about the frame centre and a shift, per axis, fitted to a "
          "block-matched displacement field. Takes out a continuation "
          "that came back a fraction of a percent taller, shorter or "
          "offset",
   .def_bool = true},
  {.key = "window", .type = ConfigType::Int, .required = false,
   .doc = "frames averaged on each side of the seam: the reference's last "
          "`window` and the continuation's first. More frames average out "
          "noise when the seam falls on a held shot, which is where a "
          "continuation's seam should fall. The stage holds this many "
          "frames before it emits the first",
   .def_int = 4},
  {.key = "max_scale", .type = ConfigType::Real, .required = false,
   .doc = "the largest scale error, as a fraction, still taken as seam "
          "drift (0.05 = 5%). Beyond it the geometry is refused and "
          "reported rather than applied -- a continuation that far off "
          "is a different shot, not a drifted one",
   .def_real = 0.05},
  {.key = "max_tone", .type = ConfigType::Real, .required = false,
   .doc = "the largest shift the tone curve may make at any level, in "
          "codes on a 0..255 scale. Beyond it the tone match is refused "
          "and reported, for the same reason",
   .def_real = 32.0},
  {.key = "fade_frames", .type = ConfigType::Int, .required = false,
   .doc = "0 (the default) applies the correction to EVERY frame of the "
          "continuation, which is right when the step holds for the whole "
          "clip -- the case this stage is for. N > 0 fades it out over "
          "the first N frames instead, handing the picture back to the "
          "model's own look after the seam",
   .def_int = 0},
};

const StageSpec kSpec = {
  .type_name = "content-aware-continuation",
  .doc       = "Matches a continuation clip to the clip it continues: "
               "measures the tone and geometry step at the seam -- the "
               "continuation's first frames against the reference's last "
               "-- and takes it out of every frame. Refuses, and says so, "
               "when the two do not look like one held shot.",
  .display_name = "Content-Aware Continuation",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

bool
ContentAwareContinuationStage::Geometry::identity() const noexcept
{
  return !ok || (std::fabs(sx - 1.0) < 2e-4 && std::fabs(sy - 1.0) < 2e-4 &&
                 std::fabs(tx) < 0.1 && std::fabs(ty) < 0.1);
}

bool
ContentAwareContinuationStage::Tone::identity() const noexcept
{
  return !ok || max_shift < 0.5;
}

ContentAwareContinuationStage::ContentAwareContinuationStage(
    const SessionContextIntf* session, string id, vector<InEdge> iports,
    FlexData config)
  : TypedStage<ContentAwareContinuationStage>(session, std::move(id),
                                              std::move(iports),
                                              std::move(config))
{
  _tone      = attr_bool("tone");
  _geometry  = attr_bool("geometry");
  _window    = (int)attr_int("window");
  _max_scale = attr_real("max_scale");
  _max_tone  = attr_real("max_tone");
  _fade      = (int)attr_int("fade_frames");
  if (_window < 1) {
    fail_config(fmt("ContentAwareContinuationStage('{}'): window must be "
                    ">= 1, got {}", this->id(), _window));
  }
  if (!(_max_scale > 0.0) || !(_max_tone > 0.0)) {
    fail_config(fmt("ContentAwareContinuationStage('{}'): max_scale and "
                    "max_tone must be > 0", this->id()));
  }
  if (_fade < 0) {
    fail_config(fmt("ContentAwareContinuationStage('{}'): fade_frames must "
                    "be >= 0, got {}", this->id(), _fade));
  }
  allocate_oports(1);
}

const StageSpec&
ContentAwareContinuationStage::spec() const noexcept
{
  return kSpec;
}

ContentAwareContinuationStage::Geometry
ContentAwareContinuationStage::estimate_geometry(const vector<float>& ref,
                                                 const vector<float>& cont,
                                                 int w, int h)
{
  Geometry g;
  const vector<float> a = luma_plane_(ref, w, h);
  const vector<float> b = luma_plane_(cont, w, h);
  int w2 = 0, h2 = 0;
  const vector<float> a2 = half_(a, w, h, &w2, &h2);
  const vector<float> b2 = half_(b, w, h, &w2, &h2);
  // A block an eighth of the short side, and a search that reaches past
  // a 5% scale at the frame's edge.
  const int bs = std::max(16, std::min(w, h) / 8);
  const int rad = (int)std::ceil(0.06 * std::max(w, h) / 2.0) + 4;
  const double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
  vector<double> px, py, dxs, dys, wt;
  for (int gy = 0; gy < kGridY; ++gy) {
    for (int gx = 0; gx < kGridX; ++gx) {
      const int x0 = rad + (int)((w - bs - 2 * rad) * (gx + 0.5) / kGridX);
      const int y0 = rad + (int)((h - bs - 2 * rad) * (gy + 0.5) / kGridY);
      if (x0 < 0 || y0 < 0 || x0 + bs > w || y0 + bs > h) { continue; }
      ++g.blocks_total;
      if (block_std_(a, w, x0, y0, bs) < kMinStd) { continue; }
      // Coarse on the half-size copy, then full size about twice that.
      double r = 0;
      int bx = 0, by = 0;
      search_(a2, b2, w2, h2, x0 / 2, y0 / 2, bs / 2, 0, 0, rad / 2 + 1, &r,
              &bx, &by);
      search_(a, b, w, h, x0, y0, bs, 2 * bx, 2 * by, 2, &r, &bx, &by);
      if (r < kMinNcc) { continue; }
      const double fx = parabola_(ncc_(a, b, w, h, x0, y0, bs, bx - 1, by), r,
                                  ncc_(a, b, w, h, x0, y0, bs, bx + 1, by));
      const double fy = parabola_(ncc_(a, b, w, h, x0, y0, bs, bx, by - 1), r,
                                  ncc_(a, b, w, h, x0, y0, bs, bx, by + 1));
      px.push_back(x0 + bs / 2.0 - 0.5 - cx);
      py.push_back(y0 + bs / 2.0 - 0.5 - cy);
      dxs.push_back(bx + fx);
      dys.push_back(by + fy);
      wt.push_back(r);
    }
  }
  g.blocks_used = (int)px.size();
  if (g.blocks_used < kMinBlocks) { return g; }
  double kx = 0, tx = 0, ky = 0, ty = 0;
  if (!fit_axis_(px, dxs, wt, &kx, &tx) || !fit_axis_(py, dys, wt, &ky, &ty)) {
    return g;
  }
  g.sx = 1.0 + kx;
  g.sy = 1.0 + ky;
  g.tx = tx;
  g.ty = ty;
  g.ok = true;
  return g;
}

void
ContentAwareContinuationStage::warp(const vector<float>& src, int w, int h,
                                    const Geometry& g, vector<float>* out)
{
  out->resize(src.size());
  const double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
  const size_t plane = (size_t)w * h;
  for (int c = 0; c < 3; ++c) {
    const float* p = &src[(size_t)c * plane];
    float* o = &(*out)[(size_t)c * plane];
    for (int y = 0; y < h; ++y) {
      const double sy = g.sy * (y - cy) + cy + g.ty;
      for (int x = 0; x < w; ++x) {
        o[(size_t)y * w + x] =
            sample_(p, w, h, g.sx * (x - cx) + cx + g.tx, sy);
      }
    }
  }
}

ContentAwareContinuationStage::Tone
ContentAwareContinuationStage::estimate_tone(const vector<float>& ref,
                                             const vector<float>& cont,
                                             int w, int h, const Geometry& g)
{
  Tone t;
  vector<float> aligned;
  if (g.ok && !g.identity()) {
    warp(cont, w, h, g, &aligned);
  } else {
    aligned = cont;
  }
  // Skip the border the warp clamped: those pixels repeat an edge and
  // pair it with whatever the reference has there.
  const int mx = g.ok ? (int)std::ceil(std::fabs(g.sx - 1.0) * w / 2.0 +
                                       std::fabs(g.tx)) + 2 : 2;
  const int my = g.ok ? (int)std::ceil(std::fabs(g.sy - 1.0) * h / 2.0 +
                                       std::fabs(g.ty)) + 2 : 2;
  const size_t plane = (size_t)w * h;
  t.ok = true;
  // Differences are histogrammed in quarter codes over +-kSpan.
  constexpr int kSpan = 64, kSteps = 4 * 2 * kSpan + 1;
  for (int c = 0; c < 3; ++c) {
    // Per continuation-level bin, a histogram of the DIFFERENCE reference
    // minus continuation, so the bin's median shift is exact without
    // keeping the samples. The median of the differences and not the
    // difference of two medians: the latter compares two statistics of
    // the level spread inside one bin and reads a flat bin as a shift
    // even when nothing moved.
    vector<vector<std::uint32_t>> hist(kBins,
                                       vector<std::uint32_t>(kSteps, 0));
    vector<double> sum(kBins, 0.0);
    vector<std::uint32_t> cnt(kBins, 0);
    for (int y = my; y < h - my; ++y) {
      for (int x = mx; x < w - mx; ++x) {
        const size_t i = (size_t)c * plane + (size_t)y * w + x;
        const float a = std::max(0.0f, std::min(255.0f, aligned[i]));
        const float d = std::max(-(float)kSpan,
                                 std::min((float)kSpan, ref[i] - a));
        const int bin = std::min(kBins - 1, (int)(a / (256.0f / kBins)));
        const int k = (int)std::lround((d + kSpan) * 4.0f);
        ++hist[(size_t)bin][(size_t)k];
        sum[(size_t)bin] += a;
        ++cnt[(size_t)bin];
      }
    }
    vector<double> cen, off, wts;
    for (int b = 0; b < kBins; ++b) {
      if (cnt[(size_t)b] < (std::uint32_t)kMinBinCount) { continue; }
      const std::uint32_t half = cnt[(size_t)b] / 2;
      std::uint32_t acc = 0;
      int med = 0;
      for (int k = 0; k < kSteps; ++k) {
        acc += hist[(size_t)b][(size_t)k];
        if (acc > half) { med = k; break; }
      }
      const double m = sum[(size_t)b] / cnt[(size_t)b];
      cen.push_back(m);
      off.push_back(m + (double)med / 4.0 - kSpan);   // the target level
      wts.push_back((double)cnt[(size_t)b]);
    }
    if (cen.empty()) { t.ok = false; return t; }
    // Monotone: a brighter continuation level never maps darker.
    isotonic_(&off, wts);
    for (size_t i = 0; i < off.size(); ++i) { off[i] -= cen[i]; }
    // Offsets interpolated between bin centres, held flat past the ends.
    auto& lut = t.lut[(size_t)c];
    for (int v = 0; v < 256; ++v) {
      double o;
      if (v <= cen.front()) {
        o = off.front();
      } else if (v >= cen.back()) {
        o = off.back();
      } else {
        size_t j = 1;
        while (j < cen.size() && cen[j] < v) { ++j; }
        const double f = (v - cen[j - 1]) / (cen[j] - cen[j - 1]);
        o = off[j - 1] + (off[j] - off[j - 1]) * f;
      }
      lut[(size_t)v] = (float)std::max(0.0, std::min(255.0, v + o));
    }
    for (int v = 1; v < 256; ++v) {
      lut[(size_t)v] = std::max(lut[(size_t)v], lut[(size_t)v - 1]);
    }
    for (int v = (int)std::ceil(cen.front()); v <= (int)cen.back(); ++v) {
      t.max_shift = std::max(t.max_shift,
                             (double)std::fabs(lut[(size_t)v] - (float)v));
    }
  }
  return t;
}

Job
ContentAwareContinuationStage::process(RuntimeContext& ctx)
{
  if (!_have_ref && !_ref_missing) {
    if (ctx.num_iports() > kRefPort && ctx.iport_connected(kRefPort)) {
      auto rb = co_await ctx.read(kRefPort);
      const auto* rt = rb ? dynamic_cast<const TensorBeatPayload*>(rb.get())
                          : nullptr;
      int first = 0;
      if (rt != nullptr && rt->shape.size() == 4) {
        first = std::max(0, (int)rt->shape[0] - _window);
      }
      if (rt != nullptr &&
          to_planar_(*rt, first, _window, &_ref, &_rw, &_rh)) {
        _have_ref = true;
      } else {
        _ref_missing = true;
        session()->warn(fmt(
            "ContentAwareContinuationStage('{}'): the reference is not a "
            "[T,3,H,W] or [3,H,W] RGB beat; passing frames through "
            "unchanged", this->id()));
      }
    } else {
      _ref_missing = true;
      session()->warn(fmt(
          "ContentAwareContinuationStage('{}'): nothing is wired to the "
          "reference iport; passing frames through unchanged", this->id()));
    }
  }

  auto in = co_await ctx.read(kFramesPort);
  const bool eos = !in;
  if (!eos) {
    const auto* t = dynamic_cast<const TensorBeatPayload*>(in.get());
    if (t == nullptr || t->shape.size() != 3 || t->shape[0] != 3 ||
        (t->dtype != TensorBeat::DType::U8 &&
         t->dtype != TensorBeat::DType::F32)) {
      session()->warn(fmt(
          "ContentAwareContinuationStage('{}'): expected planar RGB [3,H,W] "
          "u8/f32 frames; dropping beat", this->id()));
      co_return;
    }
  }

  // ---- take the estimate once `window` frames are in hand -------------
  if (!_estimated) {
    if (!eos) { _pending.push_back(std::move(in)); }
    const bool enough = (int)_pending.size() >= _window;
    if (!enough && !eos) { co_return; }
    _estimated = true;
    if (_have_ref && !_pending.empty()) {
      // The continuation's first frames, averaged.
      const auto* f0 =
          static_cast<const TensorBeatPayload*>(_pending.front().get());
      const int fw = (int)f0->shape[2], fh = (int)f0->shape[1];
      vector<float> cont((size_t)3 * fw * fh, 0.0f);
      for (const auto& p : _pending) {
        vector<float> one;
        int w = 0, h = 0;
        to_planar_(*static_cast<const TensorBeatPayload*>(p.get()), 0, 1,
                   &one, &w, &h);
        if (w != fw || h != fh) { continue; }
        for (size_t i = 0; i < cont.size(); ++i) { cont[i] += one[i]; }
      }
      for (float& v : cont) { v /= (float)_pending.size(); }
      vector<float> ref;
      if (_rw != fw || _rh != fh) {
        resize_(_ref, _rw, _rh, fw, fh, &ref);
      } else {
        ref = _ref;
      }

      Geometry g;
      if (_geometry) {
        g = estimate_geometry(ref, cont, fw, fh);
        if (!g.ok) {
          session()->warn(fmt(
              "ContentAwareContinuationStage('{}'): geometry NOT matched -- "
              "{} of {} blocks matched the reference; the continuation does "
              "not look like the same shot", this->id(), g.blocks_used,
              g.blocks_total));
          g = Geometry{};
        } else if (std::fabs(g.sx - 1.0) > _max_scale ||
                   std::fabs(g.sy - 1.0) > _max_scale) {
          session()->warn(fmt(
              "ContentAwareContinuationStage('{}'): geometry NOT matched -- "
              "scale x {:.4f} y {:.4f} is beyond max_scale {}", this->id(),
              g.sx, g.sy, _max_scale));
          g = Geometry{};
        }
      }
      _geo = g;
      if (_tone) {
        Tone tn = estimate_tone(ref, cont, fw, fh, _geo);
        if (!tn.ok) {
          session()->warn(fmt(
              "ContentAwareContinuationStage('{}'): tone NOT matched -- no "
              "level had enough pixels to fit", this->id()));
          tn = Tone{};
        } else if (tn.max_shift > _max_tone) {
          session()->warn(fmt(
              "ContentAwareContinuationStage('{}'): tone NOT matched -- the "
              "curve would move a level by {:.1f} codes, beyond max_tone {}",
              this->id(), tn.max_shift, _max_tone));
          tn = Tone{};
        }
        _curve = tn;
      }
      // What was found, in the units a person grading would use.
      auto shift_at = [this](int v) {
        double s = 0;
        for (int c = 0; c < 3; ++c) {
          s += _curve.lut[(size_t)c][(size_t)v] - v;
        }
        return s / 3.0;
      };
      session()->info(fmt(
          "ContentAwareContinuationStage('{}'): seam matched on {} frames -- "
          "geometry {} (scale x {:.4f} y {:.4f}, shift {:+.2f}, {:+.2f} px; "
          "{} of {} blocks), tone {} (shift at 16/64/128/224: "
          "{:+.1f} {:+.1f} {:+.1f} {:+.1f} codes)", this->id(),
          _pending.size(),
          _geo.identity() ? "unchanged" : "corrected", _geo.sx, _geo.sy,
          _geo.tx, _geo.ty, _geo.blocks_used, _geo.blocks_total,
          _curve.identity() ? "unchanged" : "corrected",
          _curve.ok ? shift_at(16) : 0.0, _curve.ok ? shift_at(64) : 0.0,
          _curve.ok ? shift_at(128) : 0.0, _curve.ok ? shift_at(224) : 0.0));
    }
    // Emit what was held, then fall through to EOS handling.
    auto held = std::move(_pending);
    _pending.clear();
    for (auto& p : held) {
      const auto* t = static_cast<const TensorBeatPayload*>(p.get());
      auto out = correct_(*t);
      if (out) {
        co_await ctx.write(0, std::move(out));
      } else {
        co_await ctx.write(0, std::move(p));
      }
    }
    if (eos) { ctx.signal_done(); }
    co_return;
  }

  if (eos) { ctx.signal_done(); co_return; }
  const auto* t = static_cast<const TensorBeatPayload*>(in.get());
  auto out = correct_(*t);
  if (out) {
    co_await ctx.write(0, std::move(out));
  } else {
    co_await ctx.write(0, std::move(in));
  }
}

unique_ptr<BeatPayloadIntf>
ContentAwareContinuationStage::correct_(const TensorBeatPayload& t)
{
  const std::int64_t idx = _frames_out++;
  // The fade: full correction at the seam, none after `_fade` frames.
  double wgt = 1.0;
  if (_fade > 0) {
    wgt = std::max(0.0, 1.0 - (double)idx / (double)_fade);
  }
  const bool geo = !_geo.identity() && wgt > 0.0;
  const bool tone = !_curve.identity() && wgt > 0.0;
  if (!geo && !tone) { return nullptr; }   // forward the beat untouched

  vector<float> px;
  int w = 0, h = 0;
  if (!to_planar_(t, 0, 1, &px, &w, &h)) { return nullptr; }
  if (geo) {
    Geometry g = _geo;
    g.sx = 1.0 + wgt * (g.sx - 1.0);
    g.sy = 1.0 + wgt * (g.sy - 1.0);
    g.tx *= wgt;
    g.ty *= wgt;
    vector<float> o;
    warp(px, w, h, g, &o);
    px.swap(o);
  }
  const size_t plane = (size_t)w * h;
  if (tone) {
    for (int c = 0; c < 3; ++c) {
      float* p = &px[(size_t)c * plane];
      const auto& lut = _curve.lut[(size_t)c];
      for (size_t i = 0; i < plane; ++i) {
        const float v = p[i];
        p[i] = v + (float)wgt * (lut_at_(lut, v) - v);
      }
    }
  }
  TensorBeat tb;
  tb.dtype    = t.dtype;
  tb.shape    = t.shape;
  tb.sideband = t.sideband;
  tb.resize_contiguous(px.size());
  if (t.dtype == TensorBeat::DType::U8) {
    std::uint8_t* d = tb.as_u8();
    for (size_t i = 0; i < px.size(); ++i) {
      d[i] = (std::uint8_t)std::lround(std::max(0.0f,
                                                std::min(255.0f, px[i])));
    }
  } else {
    float* d = tb.as_f32();
    for (size_t i = 0; i < px.size(); ++i) { d[i] = px[i] / 255.0f; }
  }
  return make_payload<TensorBeatPayload>(std::move(tb));
}

VPIPE_REGISTER_STAGE(ContentAwareContinuationStage)

}  // namespace vpipe
