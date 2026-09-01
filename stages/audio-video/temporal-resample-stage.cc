#include "stages/audio-video/temporal-resample-stage.h"

#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/pixfmt.h>
}

namespace vpipe {

namespace {

// Planar RGB [3, H, W] is not a pixel format FFmpeg names, but GBRP is
// the same three planes in a different ORDER -- so the mapping is a
// pointer permutation and never a copy. G is plane 0 of GBRP and plane 1
// of ours, B is 1 and 2, R is 2 and 0.
constexpr int kPlaneOfGbrp[3] = {1, 2, 0};   // gbrp index -> our index

double
sb_num_(const FlexData& sb, const char* key, double* out)
{
  if (!sb.is_object()) { return false; }
  const auto o = sb.as_object();
  if (!o.contains(key)) { return false; }
  const FlexData v = o.at(key);
  if (v.is_int() || v.is_uint() || v.is_real()) {
    *out = v.as_real(0.0);
    return true;
  }
  return false;
}

constexpr ConfigKey kAttrs[] = {
  {.key = "output_fps", .type = ConfigType::Real, .required = true,
   .doc = "the frame rate to resample TO. Required and positive -- it is "
          "the whole request, and a stage that defaulted it would be a "
          "pass-through wearing a resampler's name. Converted to the "
          "RATIONAL it is by continued fractions, so 29.97 reaches the "
          "filter as 30000/1001 and not as 2997/100: it is what the "
          "chain divides by, and a rate a hair off drops a frame every "
          "few thousand and says nothing",
   .def_real = 0.0},
  {.key = "input_fps", .type = ConfigType::Real, .required = false,
   .doc = "the SOURCE rate, when the beats do not state one. A clip beat "
          "carries `fps` and frame beats carry `fps_num`/`fps_den`, but "
          "only when the source knew them; this is the fallback and it "
          "OVERRIDES the sideband when set. Getting it wrong changes the "
          "ratio, so the clip comes out the wrong LENGTH with every "
          "frame in it individually fine. 0 (default): take the beat's",
   .def_real = 0.0},
  {.key = "method", .type = ConfigType::String, .required = false,
   .doc = "what happens to the information the rate change discards. "
          "\"nearest\" (default) drops and duplicates frames (ffmpeg "
          "`fps`) -- cheapest, and it ALIASES: a downscale by a large "
          "ratio judders and reverses wheels. \"blend\" mixes the two "
          "source frames each output frame falls between (`framerate`). "
          "\"average\" box-averages the source frames inside each output "
          "frame's interval and then decimates (`tmix` + `fps`) -- the "
          "temporal analogue of area-averaging a downscale, and it reads "
          "as motion blur because that is what it is. \"motion\" "
          "estimates motion and synthesises the in-between frames "
          "(`minterpolate`), which is the one that helps going UP in "
          "rate, where there is nothing to average. Cost rises in that "
          "order and `motion` is far the most expensive",
   .def_str = "nearest"},
  {.key = "average_frames", .type = ConfigType::Int, .required = false,
   .doc = "the `average` window, in SOURCE frames. 0 (default) derives "
          "it from the ratio -- round(input_fps / output_fps), which is "
          "the interval an output frame covers. Set it higher for a "
          "longer shutter, lower for a shorter one",
   .def_int = 0},
  {.key = "stacked", .type = ConfigType::Bool, .required = false,
   .doc = "false (default): one FRAME per beat, [3, H, W], and the oport "
          "is on a clock of its own because a rate change does not emit "
          "one beat per beat. true: the whole clip in ONE beat, "
          "[T, 3, H, W] -- what `temporal-stack` builds -- so the stage "
          "is 1:1 and its oport shares the iport's clock, crossing no "
          "domain and able to sit inside a feedback loop. STATED rather "
          "than sensed: the clock analysis runs at launch, before any "
          "beat exists",
   .def_bool = false},
};

const PortSpec kIports[] = {
  {.name = "frames",
   .doc = "planar u8 RGB TensorBeat: [3, H, W] per beat, or ONE "
          "[T, 3, H, W] clip when `stacked`",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames, rgb-clip", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "frames",
   .doc = "the same shape at `output_fps`, with the rate rewritten on "
          "the sideband. The DECLARED group is the streaming answer; the "
          "real one is resolved in oport_clock_group() -- a stacked clip "
          "is 1:1 and shares the iport's clock",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames, rgb-clip", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "temporal-resample",
  .doc       = "Resamples planar RGB frames from one frame rate to "
               "another through an FFmpeg filter chain, with a choice of "
               "how the discarded information is filtered: drop/dup, "
               "blend, box-average (motion blur) or motion-compensated "
               "interpolation. The temporal half of image-resample.",
  .display_name = "Temporal Resample",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

TemporalResampleStage::TemporalResampleStage(const SessionContextIntf* s,
                                             std::string               id,
                                             std::vector<InEdge>       ip,
                                             FlexData                  cfg)
  : TypedStage<TemporalResampleStage>(s, std::move(id), std::move(ip),
                                      std::move(cfg))
{
  allocate_oports(spec().oports.size());

  _out_fps    = attr_real("output_fps");
  _in_fps_cfg = attr_real("input_fps");
  _avg_frames = (int)attr_int("average_frames");
  _stacked    = attr_bool("stacked");

  if (!(_out_fps > 0.0)) {
    fail_config(fmt("TemporalResampleStage('{}'): output_fps must be "
                    "positive, got {}", this->id(), _out_fps));
    return;
  }
  if (_in_fps_cfg < 0.0) {
    fail_config(fmt("TemporalResampleStage('{}'): input_fps must not be "
                    "negative, got {}", this->id(), _in_fps_cfg));
    return;
  }
  if (_avg_frames < 0 || _avg_frames > 1024) {
    fail_config(fmt("TemporalResampleStage('{}'): average_frames {} is "
                    "outside [0, 1024]", this->id(), _avg_frames));
    return;
  }
  const std::string m = attr_str("method");
  if (m == "nearest")       { _method = Method::kNearest; }
  else if (m == "blend")    { _method = Method::kBlend; }
  else if (m == "average")  { _method = Method::kAverage; }
  else if (m == "motion")   { _method = Method::kMotion; }
  else {
    fail_config(fmt("TemporalResampleStage('{}'): method '{}' is not one "
                    "of nearest|blend|average|motion", this->id(), m));
    return;
  }
}

TemporalResampleStage::~TemporalResampleStage()
{
  const FFmpegLibraries* libs = session()->services()->ffmpeg_libraries();
  if (libs != nullptr && libs->avutil().api.frame_free != nullptr) {
    if (_in_frame  != nullptr) { libs->avutil().api.frame_free(&_in_frame); }
    if (_out_frame != nullptr) { libs->avutil().api.frame_free(&_out_frame); }
  }
  _graph.close();
}

const StageSpec&
TemporalResampleStage::spec() const noexcept
{
  return kSpec;
}

unsigned
TemporalResampleStage::oport_clock_group(unsigned p) const noexcept
{
  // One clip in, one clip out: the beat rate is unchanged, so the
  // output is on the input's clock and the graph crosses nothing. A
  // frame STREAM is the other case by construction -- a rate change is
  // exactly "a different number of beats" -- and gets its own group.
  (void)p;
  return _stacked ? 0u : 1u;
}

const char*
TemporalResampleStage::method_name(Method m) noexcept
{
  switch (m) {
    case Method::kNearest: return "nearest";
    case Method::kBlend:   return "blend";
    case Method::kAverage: return "average";
    case Method::kMotion:  return "motion";
  }
  return "nearest";
}

std::string
TemporalResampleStage::chain_for(Method m, double in_fps, double out_fps,
                                 int avg_frames)
{
  if (!(in_fps > 0.0) || !(out_fps > 0.0)) { return {}; }
  const AVRational r = av_rational_from_double(out_fps);
  if (r.num <= 0 || r.den <= 0) { return {}; }
  const std::string rate =
      std::to_string(r.num) + "/" + std::to_string(r.den);
  // Equal rates and a method with nothing to add: say so with an empty
  // chain so the caller can forward the beat untouched. `blend`,
  // `average` and `motion` still have work to do at equal rates (they
  // are filters, not just resamplers), so only `nearest` opts out.
  const double ratio = in_fps / out_fps;
  if (m == Method::kNearest && std::fabs(ratio - 1.0) < 1e-9) { return {}; }

  switch (m) {
    case Method::kNearest:
      return "fps=" + rate;
    case Method::kBlend:
      // `framerate` interpolates between the two neighbours; it is the
      // cheap anti-alias and it does its own rate change.
      return "framerate=fps=" + rate;
    case Method::kAverage: {
      // The box window is the interval one output frame covers, in
      // SOURCE frames. Below 2 there is nothing to average and tmix
      // would be an identity that still costs a pass.
      int n = avg_frames > 0 ? avg_frames : (int)std::lround(ratio);
      if (n > 1024) { n = 1024; }
      if (n < 2) { return "fps=" + rate; }
      return "tmix=frames=" + std::to_string(n) + ",fps=" + rate;
    }
    case Method::kMotion:
      // FFmpeg's own documented motion-compensated preset:
      // bidirectional estimation, adaptive overlapped-block
      // compensation, and variable-size blocks so flat regions cost
      // less. Left as a preset rather than five more config keys --
      // anyone who wants to tune minterpolate wants all of it, and
      // this stage is not the place to re-expose another tool's
      // manual.
      return "minterpolate=fps=" + rate +
             ":mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1";
  }
  return {};
}

double
TemporalResampleStage::input_fps_of_(const TensorBeat& tb) const
{
  // Config WINS. It is the escape hatch for a source whose sideband is
  // wrong, and a fallback that could be overridden by bad metadata
  // would not be one.
  if (_in_fps_cfg > 0.0) { return _in_fps_cfg; }
  double v = 0.0;
  if (sb_num_(tb.sideband, "fps", &v) && v > 0.0) { return v; }
  double num = 0.0, den = 0.0;
  if (sb_num_(tb.sideband, "fps_num", &num) &&
      sb_num_(tb.sideband, "fps_den", &den) && num > 0.0 && den > 0.0) {
    return num / den;
  }
  return 0.0;
}

Job
TemporalResampleStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  const FFmpegLibraries* libs = session()->services()->ffmpeg_libraries();
  if (libs == nullptr || !libs->avfilter().valid()) {
    session()->warn(fmt(
        "TemporalResampleStage('{}'): libavfilter is not loaded, so this "
        "stage can only forward beats unchanged. Install a full FFmpeg "
        "or drop the stage from the graph", this->id()));
    co_return;
  }
  session()->info(fmt(
      "TemporalResampleStage('{}'): -> {:.4f} fps, method '{}', {} input"
      "{}", this->id(), _out_fps, method_name(_method),
      _stacked ? "one stacked clip per beat" : "one frame per beat",
      _in_fps_cfg > 0.0 ? fmt(", source rate forced to {:.4f}",
                              _in_fps_cfg)() : std::string()));
  co_return;
}

void
TemporalResampleStage::reset_run_state()
{
  // The graph carries a filter's internal history (tmix's window,
  // minterpolate's previous frame) and the pts clock. A relaunch is a
  // new stream, so none of it may survive -- the alternative is the
  // first frames of run 2 blended with the last frames of run 1.
  _graph.close();
  _gw = _gh = 0;
  _gfps = 0.0;
  _pts = 0;
  _out_index = 0;
  _emitted = 0;
  _have_seed = false;
  _sb_seed = FlexData{};
  _have_ts = false;
  _first_ts_us = 0;
  _shape_warned = false;
}

bool
TemporalResampleStage::ensure_graph_(int w, int h, double in_fps,
                                     int pix_fmt)
{
  if (_graph.is_open() && _gw == w && _gh == h && _gfps == in_fps &&
      _gfmt == pix_fmt) {
    return true;
  }
  // A geometry or rate change mid-stream rebuilds without flushing, so
  // whatever the old chain was holding is dropped. That is a frame or
  // two at a boundary the source itself created, and flushing would
  // emit them at the OLD size into a stream that has just changed.
  const FFmpegLibraries* libs = session()->services()->ffmpeg_libraries();
  if (libs == nullptr || !libs->avfilter().valid()) { return false; }
  if (_in_frame == nullptr)  { _in_frame  = libs->avutil().api.frame_alloc(); }
  if (_out_frame == nullptr) { _out_frame = libs->avutil().api.frame_alloc(); }
  if (_in_frame == nullptr || _out_frame == nullptr) { return false; }

  _graph.close();
  _pts = 0;
  AvFilterGraph::VideoIn in;
  in.width      = w;
  in.height     = h;
  in.pix_fmt    = pix_fmt;
  in.frame_rate = av_rational_from_double(in_fps);
  const std::string chain =
      chain_for(_method, in_fps, _out_fps, _avg_frames);
  std::string err;
  if (!_graph.open(libs, in, chain, &err)) {
    session()->warn(fmt("TemporalResampleStage('{}'): {}", this->id(), err));
    return false;
  }
  _gw = w; _gh = h; _gfps = in_fps; _gfmt = pix_fmt;
  _gesz = pix_fmt == AV_PIX_FMT_GBRPF32 ? 4 : 1;
  session()->log_normal(fmt(
      "TemporalResampleStage('{}'): {}x{} {:.4f} -> {:.4f} fps, {} via "
      "'{}'", this->id(), w, h, in_fps, _out_fps,
      _gesz == 4 ? "f32" : "u8", chain));
  return true;
}

bool
TemporalResampleStage::push_frame_(const std::uint8_t* rgb, int w, int h,
                                   std::int64_t pts)
{
  const std::size_t plane = (std::size_t)w * h * _gesz;
  _in_frame->format = _gfmt;
  _in_frame->width  = w;
  _in_frame->height = h;
  _in_frame->pts    = pts;
  for (int i = 0; i < 3; ++i) {
    // Not a copy: the frame is unreferenced and buffersrc's KEEP_REF
    // copies it in, so pointing at the beat's own bytes is both safe
    // and free.
    _in_frame->data[i] = const_cast<std::uint8_t*>(
        rgb + (std::size_t)kPlaneOfGbrp[i] * plane);
    _in_frame->linesize[i] = w * _gesz;
  }
  std::string err;
  if (!_graph.push(_in_frame, &err)) {
    session()->warn(fmt("TemporalResampleStage('{}'): {}", this->id(), err));
    return false;
  }
  return true;
}

bool
TemporalResampleStage::drain_(
    std::vector<std::vector<std::uint8_t>>* frames)
{
  while (true) {
    std::string err;
    const auto r = _graph.pull(_out_frame, &err);
    if (r == AvFilterGraph::Pull::kAgain ||
        r == AvFilterGraph::Pull::kEof) {
      return true;
    }
    if (r == AvFilterGraph::Pull::kError) {
      session()->warn(fmt("TemporalResampleStage('{}'): {}", this->id(),
                          err));
      return false;
    }
    const int w = _out_frame->width, h = _out_frame->height;
    const std::size_t row = (std::size_t)w * _gesz;
    std::vector<std::uint8_t> px((std::size_t)3 * h * row);
    for (int i = 0; i < 3; ++i) {
      const std::uint8_t* src = _out_frame->data[i];
      const int ls = _out_frame->linesize[i];
      std::uint8_t* dst = px.data() + (std::size_t)kPlaneOfGbrp[i] * h * row;
      // Row by row: the sink's linesize is aligned and is not W.
      for (int y = 0; y < h; ++y) {
        std::memcpy(dst + (std::size_t)y * row,
                    src + (std::size_t)y * ls, row);
      }
    }
    frames->push_back(std::move(px));
  }
}

// The output rate as a rational, so the sideband states it the way
// video-to-rgb does and a sink can adopt it as its encode cadence.
FlexData
TemporalResampleStage::frame_sideband_()
{
  FlexData sb = _have_seed ? _sb_seed : FlexData::make_object();
  if (!sb.is_object()) { sb = FlexData::make_object(); }
  auto o = sb.as_object();
  const AVRational r = av_rational_from_double(_out_fps);
  o.insert_or_assign("fps_num", FlexData::make_uint((std::uint64_t)r.num));
  o.insert_or_assign("fps_den", FlexData::make_uint((std::uint64_t)r.den));
  // A clip key on a frame beat would say the frame is a whole clip.
  o.erase("fps");
  o.erase("stacked");
  if (_have_ts) {
    const double us =
        (double)_out_index * 1e6 * (double)r.den / (double)r.num;
    o.insert_or_assign(
        "timestamp_us",
        FlexData::make_uint(_first_ts_us + (std::uint64_t)(us + 0.5)));
  }
  ++_out_index;
  return sb;
}

FlexData
TemporalResampleStage::clip_sideband_(int frames)
{
  FlexData sb = _have_seed ? _sb_seed : FlexData::make_object();
  if (!sb.is_object()) { sb = FlexData::make_object(); }
  auto o = sb.as_object();
  o.insert_or_assign("fps", FlexData::make_real(_out_fps));
  o.insert_or_assign("stacked", FlexData::make_int(frames));
  o.erase("fps_num");
  o.erase("fps_den");
  if (_have_ts) {
    o.insert_or_assign("timestamp_us", FlexData::make_uint(_first_ts_us));
  }
  return sb;
}

Job
TemporalResampleStage::process(RuntimeContext& ctx)
{
  auto p = co_await ctx.read(0);
  if (!p) {
    // EOS. A frame stream still has whatever the chain is holding --
    // minterpolate keeps a frame, tmix a window -- and dropping it
    // truncates the clip by exactly the filter's latency.
    if (!_stacked && _graph.is_open()) {
      std::string err;
      std::vector<std::vector<std::uint8_t>> tail;
      if (_graph.push(nullptr, &err) && drain_(&tail)) {
        for (auto& px : tail) {
          auto t = std::make_unique<TensorBeatPayload>();
          t->dtype = _gfmt == AV_PIX_FMT_GBRPF32 ? TensorBeat::DType::F32
                                                 : TensorBeat::DType::U8;
          t->shape = {3, _gh, _gw};
          t->data.assign(px.begin(), px.end());
          t->sideband = frame_sideband_();
          ++_emitted;
          co_await ctx.write(0, std::move(t));
        }
      }
      _graph.close();
    }
    ctx.signal_done();
    co_return;
  }
  const auto* tb = dynamic_cast<const TensorBeatPayload*>(p.get());
  if (tb == nullptr) {
    session()->warn(fmt(
        "TemporalResampleStage('{}'): expected a TensorBeat, got {}; "
        "skipping", this->id(), p->describe()));
    co_return;
  }
  const FFmpegLibraries* libs = session()->services()->ffmpeg_libraries();
  if (libs == nullptr || !libs->avfilter().valid()) {
    co_await ctx.write(0, std::move(p));      // already warned at launch
    co_return;
  }
  // Both of `video-to-rgb`'s outputs, and it defaults to f32. GBRPF32
  // is the same three planes in float, so the only difference below is
  // the element size -- and filtering a normalised [0,1] clip in float
  // is if anything the better arithmetic for `blend` and `average`.
  if (tb->dtype != TensorBeat::DType::U8 &&
      tb->dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "TemporalResampleStage('{}'): frames must be planar RGB, u8 or "
        "f32, got {}; skipping", this->id(),
        TensorBeat::name_of(tb->dtype)));
    co_return;
  }
  const int pix_fmt = tb->dtype == TensorBeat::DType::F32
                          ? AV_PIX_FMT_GBRPF32
                          : AV_PIX_FMT_GBRP;

  const int rank = (int)tb->shape.size();
  const int want = _stacked ? 4 : 3;
  if (rank != want) {
    if (!_shape_warned) {
      _shape_warned = true;
      session()->warn(fmt(
          "TemporalResampleStage('{}'): `stacked` is {} so this stage "
          "expects a rank-{} beat ({}), but a rank-{} one arrived. The "
          "setting decides the CLOCK DOMAIN at launch, so it cannot "
          "follow the beat -- change `stacked` to {}; skipping",
          this->id(), _stacked ? "true" : "false", want,
          _stacked ? "[T, 3, H, W]" : "[3, H, W]", rank,
          _stacked ? "false" : "true"));
    }
    co_return;
  }
  const int frames_in = _stacked ? (int)tb->shape[0] : 1;
  const int c = (int)tb->shape[(std::size_t)rank - 3];
  const int h = (int)tb->shape[(std::size_t)rank - 2];
  const int w = (int)tb->shape[(std::size_t)rank - 1];
  if (c != 3 || h <= 0 || w <= 0 || frames_in <= 0) {
    session()->warn(fmt(
        "TemporalResampleStage('{}'): expected planar RGB, so 3 channels "
        "over a non-empty frame; got {} channels of {}x{} over {} "
        "frame(s); skipping", this->id(), c, w, h, frames_in));
    co_return;
  }

  const double in_fps = input_fps_of_(*tb);
  if (!(in_fps > 0.0)) {
    session()->warn(fmt(
        "TemporalResampleStage('{}'): the source rate is unknown -- the "
        "beat carries no `fps` (a clip) or `fps_num`/`fps_den` (frames) "
        "and `input_fps` is not set. It is what the resample ratio "
        "divides by and cannot be guessed; skipping", this->id()));
    co_return;
  }

  if (!_have_seed || _stacked) {
    _sb_seed   = tb->sideband;
    _have_seed = true;
    double ts = 0.0;
    _have_ts = sb_num_(tb->sideband, "timestamp_us", &ts) && ts >= 0.0;
    _first_ts_us = _have_ts ? (std::uint64_t)ts : 0;
    if (_stacked) { _out_index = 0; }
  }

  // A stacked clip gets a FRESH graph per beat and is flushed at the
  // end of it: the flush is what makes the last frames come out, and a
  // flushed graph is finished. A frame stream keeps one for the run.
  if (_stacked) { _graph.close(); _gw = 0; }
  if (!ensure_graph_(w, h, in_fps, pix_fmt)) { co_return; }

  const std::uint8_t* base = tb->as_u8();
  // Elements and BYTES are not the same number once the pixels are
  // f32: resize_contiguous counts elements and memcpy counts bytes.
  const std::size_t step_el = (std::size_t)3 * h * w;
  const std::size_t step_by = step_el * (std::size_t)_gesz;
  std::vector<std::vector<std::uint8_t>> out;
  for (int i = 0; i < frames_in; ++i) {
    if (!push_frame_(base + (std::size_t)i * step_by, w, h, _pts++)) {
      co_return;
    }
    if (!drain_(&out)) { co_return; }
  }
  if (_stacked) {
    std::string err;
    if (!_graph.push(nullptr, &err)) {
      session()->warn(fmt("TemporalResampleStage('{}'): {}", this->id(),
                          err));
      co_return;
    }
    if (!drain_(&out)) { co_return; }
    _graph.close();
    _gw = 0;

    auto t = std::make_unique<TensorBeatPayload>();
    t->dtype = tb->dtype;
    t->shape = {(std::int64_t)out.size(), 3, h, w};
    t->resize_contiguous(out.size() * step_el);
    std::uint8_t* dst = t->as_u8();
    for (std::size_t i = 0; i < out.size(); ++i) {
      std::memcpy(dst + i * step_by, out[i].data(), step_by);
    }
    t->sideband = clip_sideband_((int)out.size());
    ++_emitted;
    session()->log_debug(fmt(
        "TemporalResampleStage('{}'): clip {} -> {} frames "
        "({:.4f} -> {:.4f} fps)", this->id(), frames_in, out.size(),
        in_fps, _out_fps));
    co_await ctx.write(0, std::move(t));
    co_return;
  }
  for (auto& px : out) {
    auto t = std::make_unique<TensorBeatPayload>();
    t->dtype = tb->dtype;
    t->shape = {3, h, w};
    t->data.assign(px.begin(), px.end());
    t->sideband = frame_sideband_();
    ++_emitted;
    co_await ctx.write(0, std::move(t));
  }
}

VPIPE_REGISTER_STAGE(TemporalResampleStage)
VPIPE_REGISTER_SPEC(TemporalResampleStage, kSpec)

}  // namespace vpipe
