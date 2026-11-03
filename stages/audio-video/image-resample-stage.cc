#include "stages/audio-video/image-resample-stage.h"

#include "apple-silicon/metal-compute/image-ops.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/size-about-area.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "pipeline/runtime-context.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {
// Parse "#RRGGBB" / "RRGGBB" into r,g,b bytes. Defaults to 114 grey on a
// malformed value (matching the historical letterbox pad).
void parse_hex_color_(string s, uint8_t* r, uint8_t* g, uint8_t* b)
{
  *r = *g = *b = 114;
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
}  // namespace

ImageResampleStage::ImageResampleStage(const SessionContextIntf* session,
                                       string                    id,
                                       vector<InEdge>            iports,
                                       FlexData                  config)
  : TypedStage<ImageResampleStage>(session, std::move(id), std::move(iports),
                                   std::move(config))
{
  _out_w  = static_cast<int>(attr_int("width"));
  _out_h  = static_cast<int>(attr_int("height"));
  _src_x  = static_cast<int>(attr_int("src_x"));
  _src_y  = static_cast<int>(attr_int("src_y"));
  _scale  = attr_real("scale");
  _gpu    = attr_bool("gpu");
  parse_hex_color_(attr_str("pad_color"), &_pad_r, &_pad_g, &_pad_b);

  _grid       = static_cast<int>(attr_int("grid"));
  _max_pixels = attr_int("max_pixels");
  _min_side   = static_cast<int>(attr_int("min_side"));
  _max_side   = static_cast<int>(attr_int("max_side"));
  _megapixels = attr_real("megapixels");

  const string fit = attr_str("fit");
  if      (fit == "crop")         { _mode = 1; }
  else if (fit == "stretch")      { _mode = 2; }
  else if (fit == "manual")       { _mode = 3; }
  else if (fit == "pad_to_grid")  { _mode = 4; }
  else if (fit == "crop_to_grid") { _mode = 5; }
  else if (fit == "crop_to_area") { _mode = 6; }
  else                            { _mode = 0; }   // pad (default)

  const string alg = attr_str("algorithm");
  // LANCZOS unless bilinear is asked for by name. The fallback is the
  // DEFAULT, so it has to be the declared one -- attr_str() hands back
  // `def_str` for an absent key, but a key set to "" reaches here empty
  // and would otherwise land on whichever branch the else happened to be.
  if      (alg == "bilinear") { _alg = 0; }
  else if (alg == "bicubic")  { _alg = 2; }
  else                        { _alg = 1; }   // lanczos (default)
  // Deferred validation: the ctor never throws (Stage::fail_config).
  // At least one of width / height must be positive; a missing (<= 0)
  // axis is inferred per-frame from the source aspect ratio.
  if (_mode == 6) {
    // Sized from the source too. A grid and an area no size fits are
    // refused here rather than as a dropped beat.
    const long long area = std::llround(_megapixels * 1048576.0);
    if (_grid <= 0 || !(_megapixels > 0.0) ||
        size_about_area(1.0, area, _grid).first <= 0) {
      fail_config(fmt("ImageResampleStage('{}'): fit 'crop_to_area' wants "
                      "grid > 0 and megapixels > 0 with sizes on the grid "
                      "near that area (got grid {}, megapixels {})",
                      this->id(), _grid, _megapixels));
    }
  } else if (_mode >= 4) {
    // The grid fits size the output from the source; width / height are
    // not read, so nothing about them can be wrong.
    if (_grid <= 0 || _max_pixels <= 0 || _min_side < 0 ||
        _max_side < 0 || (_max_side > 0 && _max_side < _min_side)) {
      fail_config(fmt("ImageResampleStage('{}'): fit '{}' wants grid > 0, "
                      "max_pixels > 0 and 0 <= min_side <= max_side (got "
                      "grid {}, max_pixels {}, sides {}..{})", this->id(),
                      fit, _grid, _max_pixels, _min_side, _max_side));
    }
  } else if (_out_w <= 0 && _out_h <= 0) {
    fail_config(fmt("ImageResampleStage('{}'): at least one of width / "
                    "height must be > 0 (got {}x{})",
                    this->id(), _out_w, _out_h));
  } else if (!alg.empty() && alg != "bilinear" && alg != "lanczos" &&
             alg != "bicubic") {
    fail_config(fmt("ImageResampleStage('{}'): unknown algorithm '{}' "
                    "(supported: 'bilinear', 'lanczos', 'bicubic')",
                    this->id(), alg));
  }
  allocate_oports(spec().oports.size());
}

ImageResampleStage::~ImageResampleStage() = default;

namespace {
// The closed value sets, for the editor's dropdowns (see SpecExtra's
// "choices" in stage-config.h). Named arrays because `extra` is a span:
// a braced temporary would not outlive the declaration it is written in.
//
// These must agree with the ctor's own branch chains above, which stay
// the authority on what is accepted.
constexpr SpecExtra kFitChoices[] = {
  {spec_key::kChoices,
   "pad,crop,stretch,manual,pad_to_grid,crop_to_grid,crop_to_area"},
};
constexpr SpecExtra kAlgorithmChoices[] = {
  {spec_key::kChoices, "lanczos,bilinear,bicubic"},
};
constexpr ConfigKey kAttrs[] = {
  {.key = "width",  .type = ConfigType::Int,
   .doc = "output width in pixels; omit (or set <= 0) to infer it from "
          "height, preserving the source aspect ratio",
   .def_int = 0},
  {.key = "height", .type = ConfigType::Int,
   .doc = "output height in pixels; omit (or set <= 0) to infer it from "
          "width, preserving the source aspect ratio",
   .def_int = 0},
  {.key = "fit", .type = ConfigType::String,
   .doc = "aspect-ratio handling: pad (match long side + pad_color) | "
          "crop (match short side + centre-crop) | stretch (change AR) | "
          "manual (sample from src_x,src_y at scale, pad the rest). Two "
          "more size the output from the source and ignore width/height, "
          "for a model that wants sides on a grid: pad_to_grid (scale down "
          "until the picture, rounded UP to `grid` per side, is at most "
          "max_pixels; centre it and mirror-pad to that canvas; the "
          "picture's rectangle rides on as `crop_box`, so vae-decode cuts "
          "the edited result back to it -- FLUX.3's single-reference edit) "
          "| crop_to_grid (scale down to at most max_pixels, centre-crop "
          "each side DOWN to `grid` -- FLUX.3's rule for each of several "
          "references) | crop_to_area (the size on `grid` within 10% of "
          "`megapixels` whose aspect crops least from the picture's, "
          "covered and centre-cropped, up or down -- Fibo 2.0's trained "
          "sizes, Bria's get_image_size)",
   .def_str = "pad", .extra = kFitChoices},
  {.key = "grid", .type = ConfigType::Int,
   .doc = "pad_to_grid / crop_to_grid / crop_to_area: the multiple every "
          "output side lands on (16: a FLUX VAE latent cell)",
   .def_int = 16},
  {.key = "megapixels", .type = ConfigType::Real,
   .doc = "crop_to_area: the area the size is chosen about, in units of "
          "1024 x 1024. Fibo 2.0 was trained about 1; Bria's service also "
          "offers 4 (1:1 2048 x 2048, 16:9 2816 x 1584)",
   .def_real = 1.0},
  {.key = "max_pixels", .type = ConfigType::Int,
   .doc = "pad_to_grid / crop_to_grid: the most pixels the output may "
          "have (pad_to_grid counts the padded canvas). FLUX.3 uses "
          "4194304 (4 MiB) for a single reference and 1048576 (1 MiB) for "
          "each of several",
   .def_int = 4194304},
  {.key = "min_side", .type = ConfigType::Int,
   .doc = "pad_to_grid: the shortest side the canvas may have; a smaller "
          "picture is padded out to it",
   .def_int = 64},
  {.key = "max_side", .type = ConfigType::Int,
   .doc = "pad_to_grid: the longest side the canvas may have (0: no "
          "limit beyond max_pixels)",
   .def_int = 4096},
  {.key = "pad_color", .type = ConfigType::String,
   .doc = "#RRGGBB solid pad colour for pad / manual (f16 / f32 frames "
          "treat it as 0..1 normalised)",
   .def_str = "#727272"},
  {.key = "src_x", .type = ConfigType::Int,
   .doc = "manual: source origin x", .def_int = 0},
  {.key = "src_y", .type = ConfigType::Int,
   .doc = "manual: source origin y", .def_int = 0},
  {.key = "scale", .type = ConfigType::Real,
   .doc = "manual: resample ratio (output px per source px, > 0)",
   .def_real = 1.0},
  {.key = "algorithm", .type = ConfigType::String,
   .doc = "interpolation algorithm: 'lanczos' (the DEFAULT -- Lanczos-3, "
          "anti-aliased, matches PIL LANCZOS) | 'bilinear' (cheaper, and "
          "aliases when downscaling) | 'bicubic' (matches PIL BICUBIC). "
          "Most traffic through this stage is a downscale onto a model's "
          "input canvas, where bilinear keeps the high frequencies it "
          "should be removing -- so the good filter is the default and the "
          "fast one is opted into. Pick 'bicubic' when a model's own "
          "preprocessing used it and the pixels are what it conditions on: "
          "a VOSR restoration upscales its input this way before anything "
          "sees it, and a different filter there is a different input. It "
          "runs on the GPU for f16 and f32 frames and on the CPU for u8",
   .def_str = "lanczos", .extra = kAlgorithmChoices},
  {.key = "gpu", .type = ConfigType::Bool,
   .doc = "resample on the GPU where a kernel exists (the default): "
          "every filter, fit and channel count for f16 and f32 frames; "
          "bilinear and lanczos for u8 RGB. false pins the CPU path, which "
          "implements all of them too -- what to set when comparing the "
          "two",
   .def_bool = true},
};
const PortSpec kIports[] = {
  {.name = "frames", .doc = "planar RGB [3,H,W] or RGBA [4,H,W] "
                            "TensorBeat (u8, f16 or f32)",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "frames",
   .doc = "resampled planar TensorBeat [3|4,height,width] (same dtype)",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "image-resample",
  .doc       = "Resample rgb-frames to a fixed width x height, with a "
               "configurable aspect-ratio fit (pad / crop / stretch / "
               "manual) and pad colour -- or, with pad_to_grid / "
               "crop_to_grid, to the source's own aspect on a side grid "
               "(FLUX.3's reference rules), or with crop_to_area to the "
               "size about an area its aspect crops least to (Fibo 2.0's "
               "trained sizes). Resamples with Lanczos-3 by "
               "default; 'bilinear' is cheaper and 'bicubic' matches PIL. "
               "f16 and f32 frames, RGB or RGBA, resample on the GPU with "
               "every filter; u8 RGB with bilinear or lanczos; the rest on "
               "the CPU. iport and oport share one clock domain (1:1).",
  .display_name = "Resample",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
ImageResampleStage::spec() const noexcept
{
  return kSpec;
}

Job
ImageResampleStage::initialize(RuntimeContext&)
{
  _mc = session() ? session()->services()->metal_compute() : nullptr;
  const string ws = _out_w > 0 ? std::to_string(_out_w) : string("auto");
  const string hs = _out_h > 0 ? std::to_string(_out_h) : string("auto");
  // Name the FILTER and where it runs. "metal ok" alone reported the
  // backend's availability, which is not the same question: bicubic has
  // no GPU kernel and always takes the CPU path, so a graph that asked
  // for it was told metal was fine and quietly did not use it.
  //
  // THIS RUNS BEFORE ANY BEAT, so it cannot know the dtype or the
  // channel count -- and for u8 they decide it: the u8 kernels have
  // three planes and no bicubic, so u8 RGBA and u8 bicubic take the CPU.
  // f16 and f32 have kernels for all of it. Said here rather than left
  // to contradict a later log line.
  const char* alg = _alg == 0 ? "bilinear" : _alg == 2 ? "bicubic"
                                                       : "lanczos";
  const bool gpu = _gpu && _mc && _mc->valid();
  session()->info(fmt(
      "ImageResampleStage('{}'): -> {}x{}, fit={}, {} on the {}{}",
      this->id(), ws, hs, _mode, alg, gpu ? "GPU" : "CPU",
      !gpu ? "" : _alg == 2 ? " for f16/f32 frames (u8: the CPU)"
                            : " (a u8 RGBA beat takes the CPU)"));
  co_return;
}

void
ImageResampleStage::resolve_out_dims_(int in_w, int in_h,
                                      int* out_w, int* out_h) const noexcept
{
  int ow = _out_w, oh = _out_h;
  if (ow <= 0 && oh <= 0) {          // both unset -> pass the source through
    ow = in_w; oh = in_h;
  } else if (ow <= 0) {              // infer width from height (keep AR)
    ow = static_cast<int>(
        llround(static_cast<double>(in_w) * oh / in_h));
  } else if (oh <= 0) {              // infer height from width (keep AR)
    oh = static_cast<int>(
        llround(static_cast<double>(in_h) * ow / in_w));
  }
  *out_w = ow < 1 ? 1 : ow;
  *out_h = oh < 1 ? 1 : oh;
}

void
ImageResampleStage::cpu_resample_(const uint8_t* src, int in_w, int in_h,
                                  int out_w, int out_h,
                                  uint8_t* dst, bool is_f32, int C) const
{
  const metal_compute::ResampleGeom g =
      metal_compute::compute_resample_geom(in_w, in_h, out_w, out_h,
          geom_mode_(), _src_x, _src_y, static_cast<float>(_scale));
  const int planeS = in_w * in_h;
  const int planeD = out_w * out_h;
  const float div = is_f32 ? 255.0f : 1.0f;   // f32 frames are 0..1
  // A 4th pad entry for RGBA, and it is TRANSPARENT: a border invented
  // to make an aspect fit is not part of the picture, so saying it is
  // opaque would put a frame of `pad_rgb` around every letterboxed
  // image instead of leaving it empty.
  const float pad[4] = { _pad_r / div, _pad_g / div, _pad_b / div, 0.0f };
  const float* sf = reinterpret_cast<const float*>(src);
  float* df = reinterpret_cast<float*>(dst);
  auto rd = [&](int c, int x, int y) -> float {
    const int i = c * planeS + y * in_w + x;
    return is_f32 ? sf[i] : static_cast<float>(src[i]);
  };
  auto wr = [&](int c, int x, int y, float v) {
    const int i = c * planeD + y * out_w + x;
    if (is_f32) { df[i] = v; }
    else { dst[i] = static_cast<uint8_t>(clamp(v + 0.5f, 0.0f, 255.0f)); }
  };
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      const bool out = x < g.pad_x || x >= g.pad_x + g.new_w
                    || y < g.pad_y || y >= g.pad_y + g.new_h;
      if (out) {
        for (int c = 0; c < C; ++c) { wr(c, x, y, pad[c]); }
        continue;
      }
      const float sxf =
          (x - g.pad_x + 0.5f) * g.inv_x - 0.5f + g.src_x0;
      const float syf =
          (y - g.pad_y + 0.5f) * g.inv_y - 0.5f + g.src_y0;
      int ix0 = static_cast<int>(floorf(sxf));
      int iy0 = static_cast<int>(floorf(syf));
      const float fx = sxf - ix0, fy = syf - iy0;
      int ix1 = ix0 + 1, iy1 = iy0 + 1;
      ix0 = clamp(ix0, 0, in_w - 1); ix1 = clamp(ix1, 0, in_w - 1);
      iy0 = clamp(iy0, 0, in_h - 1); iy1 = clamp(iy1, 0, in_h - 1);
      for (int c = 0; c < C; ++c) {
        const float v00 = rd(c, ix0, iy0), v01 = rd(c, ix1, iy0);
        const float v10 = rd(c, ix0, iy1), v11 = rd(c, ix1, iy1);
        const float v0 = v00 + (v01 - v00) * fx;
        const float v1 = v10 + (v11 - v10) * fx;
        wr(c, x, y, v0 + (v1 - v0) * fy);
      }
    }
  }
}

void
ImageResampleStage::cpu_lanczos_(const uint8_t* src, int in_w, int in_h,
                                 int out_w, int out_h,
                                 uint8_t* dst, bool is_f32, bool cubic,
                                 int C) const
{
  const metal_compute::ResampleGeom g =
      metal_compute::compute_resample_geom(in_w, in_h, out_w, out_h,
          geom_mode_(), _src_x, _src_y, static_cast<float>(_scale));
  const int planeS = in_w * in_h;
  const int planeD = out_w * out_h;
  const float div = is_f32 ? 255.0f : 1.0f;   // f32 frames are 0..1
  // A 4th pad entry for RGBA, and it is TRANSPARENT: a border invented
  // to make an aspect fit is not part of the picture, so saying it is
  // opaque would put a frame of `pad_rgb` around every letterboxed
  // image instead of leaving it empty.
  const float pad[4] = { _pad_r / div, _pad_g / div, _pad_b / div, 0.0f };
  const float* sf = reinterpret_cast<const float*>(src);
  float* df = reinterpret_cast<float*>(dst);
  auto rd = [&](int c, int x, int y) -> float {
    const int i = c * planeS + y * in_w + x;
    return is_f32 ? sf[i] : static_cast<float>(src[i]);
  };
  auto wr = [&](int c, int x, int y, float v) {
    const int i = c * planeD + y * out_w + x;
    if (is_f32) { df[i] = v; }
    else { dst[i] = static_cast<uint8_t>(clamp(v + 0.5f, 0.0f, 255.0f)); }
  };
  if (g.new_w <= 0 || g.new_h <= 0) {           // degenerate -> all pad
    for (int y = 0; y < out_h; ++y) {
      for (int x = 0; x < out_w; ++x) {
        for (int c = 0; c < C; ++c) { wr(c, x, y, pad[c]); }
      }
    }
    return;
  }
  std::vector<int> bx, by;
  std::vector<float> wx, wy;
  const int ksx = cubic
      ? metal_compute::build_cubic_coeffs(in_w, g.src_x0, g.new_w, g.inv_x,
                                          bx, wx)
      : metal_compute::build_lanczos_coeffs(in_w, g.src_x0, g.new_w, g.inv_x,
                                            bx, wx);
  const int ksy = cubic
      ? metal_compute::build_cubic_coeffs(in_h, g.src_y0, g.new_h, g.inv_y,
                                          by, wy)
      : metal_compute::build_lanczos_coeffs(in_h, g.src_y0, g.new_h, g.inv_y,
                                            by, wy);
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      const bool out = x < g.pad_x || x >= g.pad_x + g.new_w
                    || y < g.pad_y || y >= g.pad_y + g.new_h;
      if (out) {
        for (int c = 0; c < C; ++c) { wr(c, x, y, pad[c]); }
        continue;
      }
      const int cx = x - g.pad_x, cy = y - g.pad_y;
      const int xmin = bx[static_cast<size_t>(cx)];
      const int ymin = by[static_cast<size_t>(cy)];
      for (int c = 0; c < C; ++c) {
        float acc = 0.0f;
        for (int ty = 0; ty < ksy; ++ty) {
          const float wyt = wy[static_cast<size_t>(cy) * ksy + ty];
          if (wyt == 0.0f) { continue; }
          const int sy = clamp(ymin + ty, 0, in_h - 1);
          float racc = 0.0f;
          for (int tx = 0; tx < ksx; ++tx) {
            const int sx = clamp(xmin + tx, 0, in_w - 1);
            racc += wx[static_cast<size_t>(cx) * ksx + tx] * rd(c, sx, sy);
          }
          acc += wyt * racc;
        }
        wr(c, x, y, acc);
      }
    }
  }
}

Job
ImageResampleStage::process(RuntimeContext& ctx)
{
  auto in0 = co_await ctx.read(0);
  if (!in0) { ctx.signal_done(); co_return; }

  const auto* tin = dynamic_cast<const TensorBeatPayload*>(in0.get());
  if (tin == nullptr || tin->shape.size() != 3
      || (tin->shape[0] != 3 && tin->shape[0] != 4)
      || (tin->dtype != TensorBeat::DType::U8
          && tin->dtype != TensorBeat::DType::F16
          && tin->dtype != TensorBeat::DType::F32)) {
    session()->warn(fmt(
        "ImageResampleStage('{}'): expected planar RGB [3,H,W] or RGBA "
        "[4,H,W] u8/f16/f32 TensorBeat; dropping beat", this->id()));
    co_return;
  }
  const int CH = static_cast<int>(tin->shape[0]);
  const int in_h = static_cast<int>(tin->shape[1]);
  const int in_w = static_cast<int>(tin->shape[2]);
  if (in_w <= 0 || in_h <= 0) { co_return; }

  if (_mode == 6) {
    const auto [aw, ah] = area_fit(in_w, in_h);
    if (aw <= 0 || ah <= 0) {
      session()->warn(fmt(
          "ImageResampleStage('{}'): no size on the {}-pixel grid lies "
          "within 10% of {} megapixels for a {}x{} picture; dropping beat",
          this->id(), _grid, _megapixels, in_w, in_h));
      co_return;
    }
    auto out = resample_(*tin, aw, ah);
    if (!out) { co_return; }
    if (out->sideband.is_object()) {
      out->sideband.as_object().erase(sideband::kCropBox);
    }
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  if (_mode < 4) {
    // Resolve the output size against this frame: an axis left at 0 is
    // inferred from the other so the source aspect ratio is preserved.
    int out_w = 0, out_h = 0;
    resolve_out_dims_(in_w, in_h, &out_w, &out_h);
    auto out = resample_(*tin, out_w, out_h);
    if (!out) { co_return; }
    // A rectangle on the picture that came in is not one on this one.
    if (out->sideband.is_object()) {
      out->sideband.as_object().erase(sideband::kCropBox);
    }
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // pad_to_grid / crop_to_grid: resample to the fit (nothing to do when
  // the source already is it), then pad the canvas out or cut it in.
  const GridFit g = grid_fit(in_w, in_h);
  if (g.out_w <= 0 || g.out_h <= 0) {
    session()->warn(fmt(
        "ImageResampleStage('{}'): a {}x{} picture is smaller than one "
        "{}-pixel grid cell; dropping beat", this->id(), in_w, in_h,
        _grid));
    co_return;
  }
  std::unique_ptr<TensorBeatPayload> fitted;
  const TensorBeat* src = tin;
  if (g.fit_w != in_w || g.fit_h != in_h) {
    fitted = resample_(*tin, g.fit_w, g.fit_h);
    if (!fitted) { co_return; }
    src = fitted.get();
  }
  const auto bytes = src->materialize_contiguous();
  const size_t esz = TensorBeat::byte_size_of(src->dtype);
  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = src->dtype;
  out->shape = { CH, g.out_h, g.out_w };
  out->sideband = tin->sideband;
  out->resize_contiguous(static_cast<size_t>(CH) * g.out_w * g.out_h);
  uint8_t* dst = out->bytes_();
  if (_mode == 5) {
    // Cut the canvas out of the picture at (x0, y0).
    for (int c = 0; c < CH; ++c) {
      for (int y = 0; y < g.out_h; ++y) {
        const size_t s0 = ((static_cast<size_t>(c) * g.fit_h + g.y0 + y) *
                               g.fit_w + g.x0) * esz;
        const size_t d0 =
            ((static_cast<size_t>(c) * g.out_h + y) * g.out_w) * esz;
        std::memcpy(dst + d0, bytes.data() + s0,
                    static_cast<size_t>(g.out_w) * esz);
      }
    }
  } else {
    // Mirror the picture out to the canvas. numpy's np.pad modes, which
    // BFL pads with: `reflect` (the edge sample not repeated) when the
    // picture's shorter side is longer than the grid, else `symmetric`
    // (repeated) -- periodic, so a pad longer than the side keeps
    // mirroring rather than running out.
    const bool refl = std::min(g.fit_w, g.fit_h) > _grid;
    auto mirror = [refl](int i, int n) {
      if (refl) {
        if (n == 1) { return 0; }
        const int m = 2 * (n - 1);
        i %= m;
        if (i < 0) { i += m; }
        return i >= n ? m - i : i;
      }
      const int m = 2 * n;
      i %= m;
      if (i < 0) { i += m; }
      return i >= n ? m - 1 - i : i;
    };
    std::vector<int> xs(static_cast<size_t>(g.out_w));
    for (int x = 0; x < g.out_w; ++x) {
      xs[static_cast<size_t>(x)] = mirror(x - g.x0, g.fit_w);
    }
    for (int c = 0; c < CH; ++c) {
      for (int y = 0; y < g.out_h; ++y) {
        const int sy = mirror(y - g.y0, g.fit_h);
        const uint8_t* srow = bytes.data() +
            ((static_cast<size_t>(c) * g.fit_h + sy) * g.fit_w) * esz;
        uint8_t* drow = dst +
            ((static_cast<size_t>(c) * g.out_h + y) * g.out_w) * esz;
        for (int x = 0; x < g.out_w; ++x) {
          std::memcpy(drow + static_cast<size_t>(x) * esz,
                      srow + static_cast<size_t>(xs[static_cast<size_t>(x)])
                                 * esz, esz);
        }
      }
    }
  }
  // The picture's own rectangle on the canvas, for whatever generates on
  // the canvas to be cut back to (vae-decode). Only when there IS a pad:
  // BFL crops nothing otherwise, and neither does a box that is all of it.
  if (!out->sideband.is_object()) { out->sideband = FlexData::make_object(); }
  out->sideband.as_object().erase(sideband::kCropBox);
  if (_mode == 4 && (g.fit_w != g.out_w || g.fit_h != g.out_h)) {
    FlexData box = FlexData::make_array();
    for (int v : { g.x0, g.y0, g.x0 + g.fit_w, g.y0 + g.fit_h }) {
      box.as_array().push_back(FlexData::make_int(v));
    }
    out->sideband.as_object().insert(sideband::kCropBox, std::move(box));
  }
  session()->log_debug(fmt(
      "ImageResampleStage('{}'): {}x{} -> {}x{} on a {}x{} canvas at "
      "({}, {})", this->id(), in_w, in_h, g.fit_w, g.fit_h, g.out_w,
      g.out_h, g.x0, g.y0));
  co_await ctx.write(0, std::move(out));
}

std::pair<int, int>
ImageResampleStage::area_fit(int in_w, int in_h) const noexcept
{
  if (in_w <= 0 || in_h <= 0) { return {0, 0}; }
  return size_about_area((double)in_w / (double)in_h,
                         std::llround(_megapixels * 1048576.0), _grid);
}

ImageResampleStage::GridFit
ImageResampleStage::grid_fit(int in_w, int in_h) const noexcept
{
  GridFit g;
  if (in_w <= 0 || in_h <= 0 || _grid <= 0 || _max_pixels <= 0) {
    return g;
  }
  const double area = static_cast<double>(in_w) * in_h;
  const double cap = static_cast<double>(_max_pixels);
  if (_mode == 5) {
    // flux3.pipeline.prepare_multi_reference: sides FLOORED under the
    // cap, then each cut down to the grid, centred.
    int w = in_w, h = in_h;
    if (area > cap) {
      const double sc = std::sqrt(cap / area);
      w = static_cast<int>(in_w * sc);
      h = static_cast<int>(in_h * sc);
    }
    g.fit_w = w; g.fit_h = h;
    g.out_w = w / _grid * _grid;
    g.out_h = h / _grid * _grid;
    if (g.out_w <= 0 || g.out_h <= 0) { g.out_w = g.out_h = 0; return g; }
    g.x0 = (w - g.out_w) / 2;
    g.y0 = (h - g.out_h) / 2;
    return g;
  }
  // flux3.pipeline.reference_canvas: never up; the CANVAS -- each side
  // rounded up to the grid, held to [min_side, max_side] -- at most the
  // cap, stepping the first guess down by 0.5% until it is. Sides round
  // half to even, as Python's round() does.
  double scale = std::min(1.0, std::sqrt(cap / area));
  if (_max_side > 0) {
    scale = std::min(scale, static_cast<double>(_max_side) /
                                std::max(in_w, in_h));
  }
  auto side = [&](int v) {
    int c = (v + _grid - 1) / _grid * _grid;
    c = std::max(c, _min_side);
    return _max_side > 0 ? std::min(c, _max_side) : c;
  };
  for (int it = 0; it < 4096; ++it) {
    const int fw = std::max(1, static_cast<int>(std::nearbyint(in_w * scale)));
    const int fh = std::max(1, static_cast<int>(std::nearbyint(in_h * scale)));
    const int cw = side(fw), ch = side(fh);
    if (static_cast<double>(cw) * ch <= cap) {
      g.fit_w = fw; g.fit_h = fh; g.out_w = cw; g.out_h = ch;
      g.x0 = (cw - fw) / 2;
      g.y0 = (ch - fh) / 2;
      return g;
    }
    scale *= 0.995;
  }
  return g;
}

std::unique_ptr<TensorBeatPayload>
ImageResampleStage::resample_(const TensorBeatPayload& in, int out_w,
                              int out_h)
{
  using DType = TensorBeat::DType;
  const TensorBeatPayload* tin = &in;
  const int CH = static_cast<int>(tin->shape[0]);
  const int in_h = static_cast<int>(tin->shape[1]);
  const int in_w = static_cast<int>(tin->shape[2]);

  // Both the resample kernel and the CPU fallback assume a tightly
  // packed planar source (row pitch == in_w). Producers can hand us
  // padded rows -- load-image forwards FFmpeg's GBRP linesize, which is
  // 32-byte aligned, so a width that isn't a multiple of 32 gives
  // strides with pitch > in_w. Reading that as tight shears every row
  // (slanted colour bands). Materialise a contiguous copy when needed.
  AlignedVector<uint8_t> contig;
  const uint8_t* src_tight = tin->bytes_();
  const bool contiguous = tin->is_contiguous();
  if (!contiguous) {
    contig = tin->materialize_contiguous();
    src_tight = contig.data();
  }

  // The source as the GPU reads it: the beat's own Metal buffer when it
  // has one and is tight, else a reused staging buffer.
  const bool gpu = _gpu && _mc && _mc->valid();
  const size_t esz = TensorBeat::byte_size_of(tin->dtype);
  const size_t src_bytes = static_cast<size_t>(CH) * in_w * in_h * esz;
  auto gpu_source = [&]() -> const ExternalStorageHandle* {
    if (tin->external && contiguous) { return tin->external.get(); }
    if (!_src_stage || _stage_bytes < src_bytes) {
      _src_stage =
          metal_compute::make_shared_storage(*_mc, src_bytes, session());
      _stage_bytes = _src_stage ? src_bytes : 0;
    }
    if (!_src_stage) { return nullptr; }
    std::memcpy(_src_stage->contents, src_tight, src_bytes);
    return _src_stage.get();
  };

  // GPU, FLOAT FRAMES: every filter, every fit, RGB or RGBA (resampled
  // premultiplied inside the kernel), in the frame's own element type --
  // nothing passes through 8 bits or through a CPU conversion.
  if (gpu && tin->dtype != DType::U8) {
    const ExternalStorageHandle* src_h = gpu_source();
    auto dst = metal_compute::make_shared_storage(
        *_mc, static_cast<size_t>(CH) * out_w * out_h * esz, session());
    // A border invented to make an aspect fit is not part of the
    // picture, so an RGBA pad is TRANSPARENT (the CPU path's rule).
    const float pad[4] = { _pad_r / 255.0f, _pad_g / 255.0f,
                           _pad_b / 255.0f, 0.0f };
    if (src_h && dst
        && metal_compute::resample_planar_float(
               *_mc, *src_h,
               tin->dtype == DType::F16 ? metal_compute::PlanarFloat::F16
                                        : metal_compute::PlanarFloat::F32,
               CH, in_w, in_h, *dst, out_w, out_h, geom_mode_(), _src_x, _src_y,
               static_cast<float>(_scale), _alg, pad, session())) {
      auto tb = std::make_unique<TensorBeatPayload>();
      tb->dtype = tin->dtype;
      tb->shape = { CH, out_h, out_w };
      tb->sideband = tin->sideband;
      tb->external = std::move(dst);
      return tb;
    }
    // fall through to the CPU path on any GPU failure
  }

  // GPU, U8 FRAMES via the (generalised) letterbox and Lanczos kernels.
  // RGB ONLY -- they take three pad bytes and three planes, and an RGBA
  // resample additionally has to run PREMULTIPLIED, which they have no
  // notion of. Four channels take the CPU path below; a still is not a
  // video loop, so what that costs is milliseconds.
  // No u8 bicubic kernel either: _alg 2 takes the CPU path below.
  if (gpu && CH == 3 && tin->dtype == DType::U8 && _alg != 2) {
    const ExternalStorageHandle* src_h = gpu_source();
    if (src_h) {
      auto dst = metal_compute::make_shared_storage(
          *_mc, static_cast<size_t>(3) * out_w * out_h, session());
      const bool ok = dst && (_alg == 1
          ? metal_compute::resample_lanczos_planar_u8_to_u8(
                *_mc, *src_h, in_w, in_h, *dst, out_w, out_h,
                geom_mode_(), _src_x, _src_y, static_cast<float>(_scale),
                _pad_r, _pad_g, _pad_b, session())
          : metal_compute::resample_planar_u8_to_u8(
                *_mc, *src_h, in_w, in_h, *dst, out_w, out_h,
                geom_mode_(), _src_x, _src_y, static_cast<float>(_scale),
                _pad_r, _pad_g, _pad_b, session()));
      if (ok) {
        auto tb = std::make_unique<TensorBeatPayload>();
        tb->dtype = TensorBeat::DType::U8;
        tb->shape = { 3, out_h, out_w };
        tb->sideband = tin->sideband;
        tb->external = std::move(dst);
        return tb;
      }
    }
    // fall through to the CPU path on any GPU failure
  }

  // CPU path: `gpu: false`, no-metal builds, u8 RGBA or bicubic, or a
  // GPU miss. F16 runs as F32 here -- widened in, narrowed back out --
  // so the CPU code has two element types, not three.
  const bool is_f16 = tin->dtype == DType::F16;
  AlignedVector<uint8_t> wide;
  if (is_f16) {
    const size_t n = static_cast<size_t>(CH) * in_w * in_h;
    wide.resize(n * sizeof(float));
    const auto* h = reinterpret_cast<const _Float16*>(src_tight);
    float* f = reinterpret_cast<float*>(wide.data());
    for (size_t i = 0; i < n; ++i) { f[i] = static_cast<float>(h[i]); }
    src_tight = wide.data();
  }
  TensorBeat tb;
  tb.dtype = is_f16 ? DType::F32 : tin->dtype;
  tb.shape = { CH, out_h, out_w };
  tb.sideband = tin->sideband;
  tb.resize_contiguous(static_cast<size_t>(CH) * out_w * out_h);
  const bool is_f32 = tb.dtype == DType::F32;

  // RGBA RESAMPLES PREMULTIPLIED. Averaging colour and alpha
  // independently pulls the colour of fully transparent pixels into
  // the edge -- a dark halo around everything on an image whose
  // transparent region is black, which is what a generated RGBA
  // picture usually has. Premultiply, resample all four planes, then
  // divide back out.
  AlignedVector<uint8_t> pm;
  if (CH == 4) {
    const size_t plane = static_cast<size_t>(in_w) * in_h;
    pm.resize(4 * plane * (is_f32 ? sizeof(float) : 1));
    const float amax = is_f32 ? 1.0f : 255.0f;
    const float* sf = reinterpret_cast<const float*>(src_tight);
    float* pf = reinterpret_cast<float*>(pm.data());
    for (size_t i = 0; i < plane; ++i) {
      const float a = is_f32 ? sf[3 * plane + i]
                             : static_cast<float>(src_tight[3 * plane + i]);
      const float a01 = amax > 0.0f ? a / amax : 0.0f;
      for (int c = 0; c < 3; ++c) {
        const float v = is_f32 ? sf[c * plane + i]
                               : static_cast<float>(src_tight[c * plane + i]);
        if (is_f32) { pf[c * plane + i] = v * a01; }
        else {
          pm[c * plane + i] =
              static_cast<uint8_t>(clamp(v * a01 + 0.5f, 0.0f, 255.0f));
        }
      }
      if (is_f32) { pf[3 * plane + i] = a; }
      else { pm[3 * plane + i] = src_tight[3 * plane + i]; }
    }
    src_tight = pm.data();
  }

  if (_alg == 1 || _alg == 2) {
    cpu_lanczos_(src_tight, in_w, in_h, out_w, out_h, tb.bytes_(), is_f32,
                 _alg == 2, CH);
  } else {
    cpu_resample_(src_tight, in_w, in_h, out_w, out_h, tb.bytes_(), is_f32,
                  CH);
  }
  if (CH == 4) {
    const size_t plane = static_cast<size_t>(out_w) * out_h;
    const float amax = is_f32 ? 1.0f : 255.0f;
    uint8_t* d8 = tb.bytes_();
    float* df = reinterpret_cast<float*>(d8);
    for (size_t i = 0; i < plane; ++i) {
      const float a = is_f32 ? df[3 * plane + i]
                             : static_cast<float>(d8[3 * plane + i]);
      const float a01 = amax > 0.0f ? a / amax : 0.0f;
      if (a01 <= 1e-4f) { continue; }        // nothing to recover
      for (int c = 0; c < 3; ++c) {
        if (is_f32) { df[c * plane + i] /= a01; }
        else {
          d8[c * plane + i] = static_cast<uint8_t>(
              clamp(d8[c * plane + i] / a01 + 0.5f, 0.0f, 255.0f));
        }
      }
    }
  }
  if (is_f16) {
    auto narrow = std::make_unique<TensorBeatPayload>();
    narrow->dtype = DType::F16;
    narrow->shape = tb.shape;
    narrow->sideband = std::move(tb.sideband);
    const size_t n = static_cast<size_t>(CH) * out_w * out_h;
    narrow->resize_contiguous(n);
    const float* f = tb.as_f32();
    auto* h = reinterpret_cast<_Float16*>(narrow->bytes_());
    for (size_t i = 0; i < n; ++i) { h[i] = static_cast<_Float16>(f[i]); }
    return narrow;
  }
  return std::make_unique<TensorBeatPayload>(std::move(tb));
}

VPIPE_REGISTER_STAGE(ImageResampleStage)
VPIPE_REGISTER_SPEC(ImageResampleStage, kSpec)

}
