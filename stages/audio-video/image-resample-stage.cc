#include "stages/audio-video/image-resample-stage.h"

#include "apple-silicon/metal-compute/image-ops.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

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
  parse_hex_color_(attr_str("pad_color"), &_pad_r, &_pad_g, &_pad_b);

  const string fit = attr_str("fit");
  if      (fit == "crop")    { _mode = 1; }
  else if (fit == "stretch") { _mode = 2; }
  else if (fit == "manual")  { _mode = 3; }
  else                       { _mode = 0; }   // pad (default)

  const string alg = attr_str("algorithm");
  // Deferred validation: the ctor never throws (Stage::fail_config).
  if (_out_w <= 0 || _out_h <= 0) {
    fail_config(fmt("ImageResampleStage('{}'): width and height must be "
                    "> 0 (got {}x{})", this->id(), _out_w, _out_h));
  } else if (!alg.empty() && alg != "bilinear") {
    fail_config(fmt("ImageResampleStage('{}'): unknown algorithm '{}' "
                    "(only 'bilinear' is supported)", this->id(), alg));
  }
  allocate_oports(spec().oports.size());
}

ImageResampleStage::~ImageResampleStage() = default;

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "width",  .type = ConfigType::Int, .required = true,
   .doc = "output width in pixels (> 0)"},
  {.key = "height", .type = ConfigType::Int, .required = true,
   .doc = "output height in pixels (> 0)"},
  {.key = "fit", .type = ConfigType::String,
   .doc = "aspect-ratio handling: pad (match long side + pad_color) | "
          "crop (match short side + centre-crop) | stretch (change AR) | "
          "manual (sample from src_x,src_y at scale, pad the rest)",
   .def_str = "pad"},
  {.key = "pad_color", .type = ConfigType::String,
   .doc = "#RRGGBB solid pad colour for pad / manual (f32 frames treat it "
          "as 0..1 normalised)",
   .def_str = "#727272"},
  {.key = "src_x", .type = ConfigType::Int,
   .doc = "manual: source origin x", .def_int = 0},
  {.key = "src_y", .type = ConfigType::Int,
   .doc = "manual: source origin y", .def_int = 0},
  {.key = "scale", .type = ConfigType::Real,
   .doc = "manual: resample ratio (output px per source px, > 0)",
   .def_real = 1.0},
  {.key = "algorithm", .type = ConfigType::String,
   .doc = "interpolation algorithm (only 'bilinear' for now)",
   .def_str = "bilinear"},
};
const PortSpec kIports[] = {
  {.name = "frames", .doc = "planar RGB TensorBeat [3,H,W] (u8 or f32)",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "frames",
   .doc = "resampled planar RGB TensorBeat [3,height,width] (same dtype)",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "image-resample",
  .doc       = "Resample rgb-frames to a fixed width x height, with a "
               "configurable aspect-ratio fit (pad / crop / stretch / "
               "manual) and pad colour. u8 frames use the letterbox GPU "
               "kernel; f32 uses a CPU bilinear fallback. iport and oport "
               "share one clock domain (1:1).",
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
  _mc = session() ? session()->metal_compute() : nullptr;
  session()->info(fmt(
      "ImageResampleStage('{}'): -> {}x{}, fit={}, metal {}",
      this->id(), _out_w, _out_h, _mode,
      (_mc && _mc->valid()) ? "ok" : "cpu-fallback"));
  co_return;
}

void
ImageResampleStage::cpu_resample_(const uint8_t* src, int in_w, int in_h,
                                  uint8_t* dst, bool is_f32) const
{
  const metal_compute::ResampleGeom g =
      metal_compute::compute_resample_geom(in_w, in_h, _out_w, _out_h,
          _mode, _src_x, _src_y, static_cast<float>(_scale));
  const int planeS = in_w * in_h;
  const int planeD = _out_w * _out_h;
  const float div = is_f32 ? 255.0f : 1.0f;   // f32 frames are 0..1
  const float pad[3] = { _pad_r / div, _pad_g / div, _pad_b / div };
  const float* sf = reinterpret_cast<const float*>(src);
  float* df = reinterpret_cast<float*>(dst);
  auto rd = [&](int c, int x, int y) -> float {
    const int i = c * planeS + y * in_w + x;
    return is_f32 ? sf[i] : static_cast<float>(src[i]);
  };
  auto wr = [&](int c, int x, int y, float v) {
    const int i = c * planeD + y * _out_w + x;
    if (is_f32) { df[i] = v; }
    else { dst[i] = static_cast<uint8_t>(clamp(v + 0.5f, 0.0f, 255.0f)); }
  };
  for (int y = 0; y < _out_h; ++y) {
    for (int x = 0; x < _out_w; ++x) {
      const bool out = x < g.pad_x || x >= g.pad_x + g.new_w
                    || y < g.pad_y || y >= g.pad_y + g.new_h;
      if (out) {
        for (int c = 0; c < 3; ++c) { wr(c, x, y, pad[c]); }
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
      for (int c = 0; c < 3; ++c) {
        const float v00 = rd(c, ix0, iy0), v01 = rd(c, ix1, iy0);
        const float v10 = rd(c, ix0, iy1), v11 = rd(c, ix1, iy1);
        const float v0 = v00 + (v01 - v00) * fx;
        const float v1 = v10 + (v11 - v10) * fx;
        wr(c, x, y, v0 + (v1 - v0) * fy);
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
  if (tin == nullptr || tin->shape.size() != 3 || tin->shape[0] != 3
      || (tin->dtype != TensorBeat::DType::U8
          && tin->dtype != TensorBeat::DType::F32)) {
    session()->warn(fmt(
        "ImageResampleStage('{}'): expected planar RGB [3,H,W] u8/f32 "
        "TensorBeat; dropping beat", this->id()));
    co_return;
  }
  const int in_h = static_cast<int>(tin->shape[1]);
  const int in_w = static_cast<int>(tin->shape[2]);
  if (in_w <= 0 || in_h <= 0) { co_return; }

  // GPU fast path: u8 frames via the (generalised) letterbox kernel.
  if (tin->dtype == TensorBeat::DType::U8 && _mc && _mc->valid()) {
    const ExternalStorageHandle* src_h = nullptr;
    if (tin->external) {
      src_h = tin->external.get();               // already GPU-resident
    } else {
      const size_t need = static_cast<size_t>(3) * in_w * in_h;
      if (!_src_stage || _stage_in_w != in_w || _stage_in_h != in_h) {
        _src_stage =
            metal_compute::make_shared_storage(*_mc, need, session());
        _stage_in_w = in_w; _stage_in_h = in_h;
      }
      if (_src_stage) {
        std::memcpy(_src_stage->contents, tin->as_u8(), need);
        src_h = _src_stage.get();
      }
    }
    if (src_h) {
      auto dst = metal_compute::make_shared_storage(
          *_mc, static_cast<size_t>(3) * _out_w * _out_h, session());
      if (dst && metal_compute::resample_planar_u8_to_u8(
              *_mc, *src_h, in_w, in_h, *dst, _out_w, _out_h,
              _mode, _src_x, _src_y, static_cast<float>(_scale),
              _pad_r, _pad_g, _pad_b, session())) {
        TensorBeat tb;
        tb.dtype = TensorBeat::DType::U8;
        tb.shape = { 3, _out_h, _out_w };
        tb.sideband = tin->sideband;
        tb.external = std::move(dst);
        co_await ctx.write(0,
            make_payload<TensorBeatPayload>(std::move(tb)));
        co_return;
      }
    }
    // fall through to the CPU path on any GPU failure
  }

  // CPU fallback: f32 frames, no-metal builds, or a GPU miss.
  TensorBeat tb;
  tb.dtype = tin->dtype;
  tb.shape = { 3, _out_h, _out_w };
  tb.sideband = tin->sideband;
  tb.resize_contiguous(static_cast<size_t>(3) * _out_w * _out_h);
  cpu_resample_(tin->bytes_(), in_w, in_h, tb.bytes_(),
                tin->dtype == TensorBeat::DType::F32);
  co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
}

VPIPE_REGISTER_STAGE(ImageResampleStage)
VPIPE_REGISTER_SPEC(ImageResampleStage, kSpec)

}
