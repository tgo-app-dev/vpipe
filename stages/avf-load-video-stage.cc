#include "stages/avf-load-video-stage.h"

#include "apple-silicon/media/video-io.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <utility>

namespace vpipe {

namespace {

constexpr SpecExtra kDtypeChoices[] = {
  {spec_key::kChoices, "f16,u8,f32"},
};
constexpr SpecExtra kAlphaChoices[] = {
  {spec_key::kChoices, "keep,drop"},
};
constexpr ConfigKey kAttrs[] = {
  {.key = "url", .type = ConfigType::String, .required = true,
   .doc = "the movie file (.mov, .mp4, .m4v): ProRes, HEVC (Main10 and "
          "HDR), H.264 -- whatever AVFoundation decodes",
   .is_path = true, .path_filter = "video"},
  {.key = "dtype", .type = ConfigType::String, .required = false,
   .doc = "f16 | u8 | f32. f16 (the default) keeps 10 and 12 bits and "
          "an HDR transfer's code values (0..1); u8 is what the 8-bit "
          "consumers read; f32 (0..1) is video-to-rgb's default format, "
          "for consumers built around it",
   .def_str = "f16", .extra = kDtypeChoices},
  {.key = "alpha", .type = ConfigType::String, .required = false,
   .doc = "keep | drop. keep (the default) emits [4,H,W] when the track "
          "has alpha (ProRes 4444), straight unless the file says it is "
          "premultiplied (sideband alpha_mode)",
   .def_str = "keep", .extra = kAlphaChoices},
  {.key = "start_s", .type = ConfigType::Real, .required = false,
   .doc = "seconds into the file to start at", .def_real = 0},
  {.key = "duration_s", .type = ConfigType::Real, .required = false,
   .doc = "seconds to read; 0 (the default) = to the end",
   .def_real = 0},
  {.key = "max_frames", .type = ConfigType::Uint, .required = false,
   .doc = "stop after this many frames; 0 (the default) = no limit",
   .def_uint = 0},
};
const PortSpec kOports[] = {
  {.name = "frames",
   .doc = "one planar TensorBeat [3|4,H,W] per frame, F16, U8 or F32, its "
          "sideband carrying the file's colour (color_primaries, "
          "color_transfer, color_matrix, alpha_mode, source_bits, "
          "mastering_display, content_light) and timing (fps, fps_num, "
          "fps_den, frame, frames, pts_us)",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "avf-load-video",
  .doc = "Source: a movie's frames through AVFoundation / VideoToolbox "
         "(Apple) -- ProRes 4444 with alpha, HEVC Main10 with PQ/HLG and "
         "its HDR10 metadata -- as F16 (or U8) planar pictures tagged with "
         "the colour the file declares. Video only.",
  .display_name = "Load Video (AVFoundation)",
  .category = StageCategory::Visual,
  .iports = {},
  .oports = kOports,
  .attrs = kAttrs,
};

}  // namespace

AvfLoadVideoStage::AvfLoadVideoStage(const SessionContextIntf* s,
                                     std::string id,
                                     std::vector<InEdge> iports,
                                     FlexData config)
  : TypedStage<AvfLoadVideoStage>(s, std::move(id), std::move(iports),
                                  std::move(config))
{
  _url = attr_path("url", /*for_write=*/false);
  if (_url.rfind("file://", 0) == 0) { _url.erase(0, 7); }
  if (_url.empty()) {
    fail_config(fmt("AvfLoadVideoStage('{}'): config.url is required",
                    this->id()));
  }
  const std::string dtype = attr_str("dtype");
  if (dtype == "u8") {
    _dtype = TensorBeat::DType::U8;
  } else if (dtype == "f32") {
    _dtype = TensorBeat::DType::F32;
  } else if (dtype != "f16") {
    fail_config(fmt("AvfLoadVideoStage('{}'): dtype must be f16, u8 or "
                    "f32", this->id()));
  }
  _keep_alpha = attr_str("alpha") != "drop";
  _start_s = attr_real("start_s");
  _duration_s = attr_real("duration_s");
  _max_frames = attr_uint("max_frames");
  allocate_oports(kSpec.oports.size());
}

AvfLoadVideoStage::~AvfLoadVideoStage() = default;

const StageSpec&
AvfLoadVideoStage::spec() const noexcept
{
  return kSpec;
}

void
AvfLoadVideoStage::reset_run_state()
{
  _reader.reset();
  _failed = false;
  _emitted = 0;
}

Job
AvfLoadVideoStage::process(RuntimeContext& ctx)
{
  if (_failed || (_max_frames > 0 && _emitted >= _max_frames)) {
    ctx.signal_done();
    co_return;
  }
  if (!_reader) {
    _reader = std::make_unique<apple_media::VideoReader>();
    apple_media::VideoReadOptions opt;
    opt.dtype = _dtype;
    opt.keep_alpha = _keep_alpha;
    opt.start_s = _start_s;
    opt.duration_s = _duration_s;
    std::string err;
    if (!_reader->open(_url, opt, &err)) {
      session()->error(fmt("AvfLoadVideoStage('{}'): {}", this->id(), err));
      _failed = true;
      ctx.signal_done();
      co_return;
    }
    const auto& vi = _reader->info();
    session()->log_verbose(fmt(
        "AvfLoadVideoStage('{}'): {} {}x{} {}/{} fps, {} bits, alpha {}",
        this->id(), vi.codec, vi.width, vi.height, vi.fps_num, vi.fps_den,
        vi.color.source_bits, vi.has_alpha));
  }
  auto frame = std::make_unique<TensorBeatPayload>();
  std::int64_t pts_us = 0;
  std::string err;
  if (!_reader->read(frame.get(), &pts_us, &err)) {
    if (!err.empty()) {
      session()->error(fmt("AvfLoadVideoStage('{}'): {}", this->id(), err));
    }
    ctx.signal_done();
    co_return;
  }
  const auto& vi = _reader->info();
  auto o = frame->sideband.as_object();
  o.insert_or_assign(sideband::kFps,
                     FlexData::make_real((double)vi.fps_num / vi.fps_den));
  o.insert_or_assign(sideband::kFpsNum,
                     FlexData::make_uint((std::uint64_t)vi.fps_num));
  o.insert_or_assign(sideband::kFpsDen,
                     FlexData::make_uint((std::uint64_t)vi.fps_den));
  o.insert_or_assign(sideband::kFrame,
                     FlexData::make_int((std::int64_t)_emitted));
  o.insert_or_assign(sideband::kFrames, FlexData::make_int(vi.frames));
  o.insert_or_assign(sideband::kPtsUs, FlexData::make_int(pts_us));
  ++_emitted;
  co_await ctx.write(0, std::move(frame));
}

VPIPE_REGISTER_STAGE(AvfLoadVideoStage)
VPIPE_REGISTER_SPEC(AvfLoadVideoStage, kSpec)

}  // namespace vpipe
