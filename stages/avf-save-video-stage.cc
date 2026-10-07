#include "stages/avf-save-video-stage.h"

#include "apple-silicon/media/color-tags.h"
#include "apple-silicon/media/video-io.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <filesystem>
#include <utility>

namespace vpipe {

namespace {

constexpr SpecExtra kCodecChoices[] = {
  {spec_key::kChoices,
   "prores4444,prores4444xq,prores422hq,prores422,prores422lt,"
   "prores422proxy,hevc,hevc8,h264"},
};
constexpr SpecExtra kAlphaChoices[] = {
  {spec_key::kChoices, "keep,drop"},
};
constexpr SpecExtra kProfileChoices[] = {
  {spec_key::kChoices, ",baseline,main,high"},
};
constexpr SpecExtra kLevelChoices[] = {
  {spec_key::kChoices, ",auto,3.0,3.1,3.2,4.0,4.1,4.2,5.0,5.1,5.2"},
};
constexpr SpecExtra kEntropyChoices[] = {
  {spec_key::kChoices, ",auto,cabac,cavlc"},
};
constexpr SpecExtra kReorderChoices[] = {
  {spec_key::kChoices, "auto,on,off"},
};
constexpr ConfigKey kAttrs[] = {
  {.key = "path", .type = ConfigType::String, .required = true,
   .doc = "the movie to write: .mov (any codec) or .mp4 / .m4v (HEVC, "
          "H.264). An existing file is replaced",
   .is_path = true, .path_write = true, .path_filter = "video"},
  {.key = "codec", .type = ConfigType::String, .required = false,
   .doc = "prores4444 | prores4444xq | prores422hq | prores422 | "
          "prores422lt | prores422proxy | hevc (Main10; HDR when the "
          "frames are PQ or HLG) | hevc8 | h264. Empty: prores4444 for "
          ".mov, hevc for .mp4",
   .def_str = "", .extra = kCodecChoices},
  {.key = "fps", .type = ConfigType::Real, .required = false,
   .doc = "frames per second; 0 (the default) takes the frames' own "
          "(sideband fps_num/fps_den or fps), else 24",
   .def_real = 0},
  {.key = "alpha", .type = ConfigType::String, .required = false,
   .doc = "keep | drop. keep (the default) writes a 4-channel picture's "
          "alpha into ProRes 4444; other codecs have none",
   .def_str = "keep", .extra = kAlphaChoices},
  {.key = "quality", .type = ConfigType::Real, .required = false,
   .doc = "HEVC / H.264 quality 0..1; 0 (the default) = the encoder's",
   .def_real = 0},
  {.key = "bitrate", .type = ConfigType::Int, .required = false,
   .doc = "HEVC / H.264 average bits per second; 0 (the default) = auto",
   .def_int = 0},
  {.key = "max_bitrate", .type = ConfigType::Int, .required = false,
   .doc = "HEVC / H.264: the most bits in any one second (a peak, as "
          "a delivery spec caps it); 0 (the default) = none",
   .def_int = 0},
  {.key = "keyframe_interval", .type = ConfigType::Uint, .required = false,
   .doc = "HEVC / H.264: the most frames from one keyframe to the next "
          "(a GOP's length); 0 (the default) = the encoder's",
   .def_uint = 0},
  {.key = "frame_reordering", .type = ConfigType::String,
   .required = false,
   .doc = "HEVC / H.264 B-frames: auto (the default, the encoder's) | on "
          "| off. H.264 baseline has none",
   .def_str = "auto", .extra = kReorderChoices},
  {.key = "profile", .type = ConfigType::String, .required = false,
   .doc = "H.264's profile: baseline | main | high; empty (the default) "
          "= high. HEVC's is its codec's (hevc Main10, hevc8 Main)",
   .def_str = "", .extra = kProfileChoices},
  {.key = "level", .type = ConfigType::String, .required = false,
   .doc = "H.264's level: 3.0 .. 5.2; empty or auto (the default) = the "
          "encoder's for the size and rate",
   .def_str = "", .extra = kLevelChoices},
  {.key = "entropy", .type = ConfigType::String, .required = false,
   .doc = "H.264's entropy coding: cabac | cavlc; empty or auto (the "
          "default) = the encoder's (CABAC; baseline has CAVLC alone)",
   .def_str = "", .extra = kEntropyChoices},
  {.key = "color_primaries", .type = ConfigType::Uint, .required = false,
   .doc = "the colour to declare, overriding the frames' tags (CICP: 1 "
          "BT.709, 9 BT.2020, 12 Display P3); 0 (the default) = the "
          "frames' own",
   .def_uint = 0},
  {.key = "color_transfer", .type = ConfigType::Uint, .required = false,
   .doc = "CICP transfer to declare (1 BT.709, 16 PQ, 18 HLG, 8 linear); "
          "0 (the default) = the frames' own. It says what the samples "
          "ARE -- nothing is converted",
   .def_uint = 0},
  {.key = "color_matrix", .type = ConfigType::Uint, .required = false,
   .doc = "CICP YCbCr matrix (1 BT.709, 9 BT.2020); 0 (the default) = "
          "the frames' own, else the one their primaries imply",
   .def_uint = 0},
  {.key = "frames", .type = ConfigType::Uint, .required = false,
   .doc = "the frames to expect, for the progress report (\"save "
          "video\": frames written of these); 0 (the default) = the "
          "frames' sideband `frames`, else uncounted. Nothing is cut or "
          "padded to it",
   .def_uint = 0},
};
const PortSpec kIports[] = {
  {.name = "frames",
   .doc = "planar TensorBeat [3|4,H,W] per frame, F16, U8 or F32 (0..1, "
          "video-to-rgb's default), all one size (avf-load-video / "
          "video-to-rgb / vae-decode format)",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "avf-save-video",
  .doc = "Sink: pictures into a movie through AVFoundation / VideoToolbox "
         "(Apple) -- ProRes 4444 with alpha, ProRes 422, HEVC Main10 with "
         "PQ/HLG and HDR10 metadata, H.264 -- the file tagged with the "
         "frames' colour. Finished at end of stream. Video only.",
  .display_name = "Save Video (AVFoundation)",
  .category = StageCategory::Visual,
  .iports = kIports,
  .oports = {},
  .attrs = kAttrs,
};

}  // namespace

AvfSaveVideoStage::AvfSaveVideoStage(const SessionContextIntf* s,
                                     std::string id,
                                     std::vector<InEdge> iports,
                                     FlexData config)
  : TypedStage<AvfSaveVideoStage>(s, std::move(id), std::move(iports),
                                  std::move(config))
{
  _path = attr_path("path", /*for_write=*/true);
  if (_path.empty()) {
    fail_config(fmt("AvfSaveVideoStage('{}'): config.path is required",
                    this->id()));
  }
  std::string ext = std::filesystem::path(_path).extension().string();
  for (char& c : ext) { c = (char)tolower((unsigned char)c); }
  const bool mp4 = ext == ".mp4" || ext == ".m4v";
  _codec = attr_str("codec");
  if (_codec.empty()) { _codec = mp4 ? "hevc" : "prores4444"; }
  if (!apple_media::video_codec_known(_codec)) {
    fail_config(fmt("AvfSaveVideoStage('{}'): unknown codec '{}'",
                    this->id(), _codec));
  } else if (mp4 && _codec.rfind("prores", 0) == 0) {
    fail_config(fmt("AvfSaveVideoStage('{}'): ProRes goes in a .mov",
                    this->id()));
  }
  _fps = attr_real("fps");
  _keep_alpha = attr_str("alpha") != "drop";
  _quality = attr_real("quality");
  _bitrate = attr_int("bitrate");
  _max_bitrate = attr_int("max_bitrate");
  _keyframe_interval = (int)attr_uint("keyframe_interval");
  const std::string reorder = attr_str("frame_reordering");
  _frame_reordering = reorder == "on" ? 1 : reorder == "off" ? 0 : -1;
  _profile = attr_str("profile");
  _level = attr_str("level");
  _entropy = attr_str("entropy");
  if (_entropy == "auto") { _entropy.clear(); }
  if (_codec == "h264" &&
      !apple_media::h264_settings_known(_profile, _level, _entropy)) {
    fail_config(fmt("AvfSaveVideoStage('{}'): no H.264 profile '{}' at "
                    "level '{}' with entropy '{}'", this->id(), _profile,
                    _level, _entropy));
  }
  _primaries = (int)attr_uint("color_primaries");
  _transfer = (int)attr_uint("color_transfer");
  _matrix = (int)attr_uint("color_matrix");
  _expected = attr_uint("frames");
}

AvfSaveVideoStage::~AvfSaveVideoStage() = default;

const StageSpec&
AvfSaveVideoStage::spec() const noexcept
{
  return kSpec;
}

void
AvfSaveVideoStage::reset_run_state()
{
  _writer.reset();
  _failed = false;
  _finished = false;
  _written = 0;
  _bar.finish();
  _total = 0;
}

void
AvfSaveVideoStage::finish_()
{
  if (!_writer || _finished) { return; }
  std::string err;
  if (_writer->finish(&err)) {
    _finished = true;
    session()->log_verbose(fmt("AvfSaveVideoStage('{}'): wrote {} frames "
                               "to '{}'", this->id(), _written, _path));
  } else {
    session()->error(fmt("AvfSaveVideoStage('{}'): {}", this->id(), err));
  }
  _writer.reset();
  _bar.finish();
}

Job
AvfSaveVideoStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  if (!in) {  // end of stream: the movie is finished
    finish_();
    ctx.signal_done();
    co_return;
  }
  if (_failed) { co_return; }
  const auto* pic = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (!pic || pic->shape.size() != 3 ||
      (pic->shape[0] != 3 && pic->shape[0] != 4) ||
      (pic->dtype != TensorBeat::DType::U8 &&
       pic->dtype != TensorBeat::DType::F16 &&
       pic->dtype != TensorBeat::DType::F32)) {
    session()->warn(fmt("AvfSaveVideoStage('{}'): expected a U8, F16 or "
                        "F32 [3|4,H,W] picture, got {}; dropping it",
                        this->id(), in->describe()));
    co_return;
  }
  if (!_writer) {
    // The first frame sets the movie: its size, its rate, its colour.
    apple_media::VideoWriteOptions opt;
    opt.codec = _codec;
    opt.width = (int)pic->shape[2];
    opt.height = (int)pic->shape[1];
    opt.alpha = _keep_alpha && pic->shape[0] == 4;
    opt.quality = _quality;
    opt.bitrate = _bitrate;
    opt.max_bitrate = _max_bitrate;
    opt.keyframe_interval = _keyframe_interval;
    opt.frame_reordering = _frame_reordering;
    opt.profile = _profile;
    opt.level = _level;
    opt.entropy = _entropy;
    opt.color = apple_media::ColorTags::read(pic->sideband);
    if (_primaries) { opt.color.primaries = _primaries; }
    if (_transfer) { opt.color.transfer = _transfer; }
    if (_matrix) { opt.color.matrix = _matrix; }
    FlexData sb = pic->sideband;  // as_object() is a view
    std::uint64_t num = 0, den = 0;
    double fps = _fps;
    if (fps <= 0 && sb.is_object()) {
      auto so = sb.as_object();
      if (so.contains(sideband::kFpsNum) && so.contains(sideband::kFpsDen)) {
        num = so.at(sideband::kFpsNum).as_uint(0);
        den = so.at(sideband::kFpsDen).as_uint(0);
      }
      if (so.contains(sideband::kFps)) {
        fps = so.at(sideband::kFps).as_real(0);
      }
    }
    if (_fps <= 0 && num > 0 && den > 0) {
      opt.fps_num = (int)num;
      opt.fps_den = (int)den;
    } else {
      apple_media::fps_fraction(fps > 0 ? fps : 24.0, &opt.fps_num,
                                &opt.fps_den);
    }
    _writer = std::make_unique<apple_media::VideoWriter>();
    std::string err;
    if (!_writer->open(_path, opt, &err)) {
      session()->error(fmt("AvfSaveVideoStage('{}'): {}", this->id(), err));
      _writer.reset();
      _failed = true;
      co_return;
    }
    // A long movie -- an hour's export -- reports its frames as it goes.
    _total = _expected;
    if (_total == 0 && sb.is_object() &&
        sb.as_object().contains(sideband::kFrames)) {
      const std::int64_t n =
          sb.as_object().at(sideband::kFrames).as_int(0);
      _total = n > 0 ? (std::uint64_t)n : 0;
    }
    _bar = session()->open_progress("save video");
    _bar.update(0, _total);
  }
  std::string err;
  if (!_writer->write(*pic, (std::int64_t)_written, &err)) {
    session()->error(fmt("AvfSaveVideoStage('{}'): {}", this->id(), err));
    _writer.reset();
    _bar.finish();
    _failed = true;
    co_return;
  }
  ++_written;
  // Past what it was told (an estimate from a duration): still counted.
  _bar.update(_written, _total > 0 ? std::max(_total, _written) : 0);
}

VPIPE_REGISTER_STAGE(AvfSaveVideoStage)
VPIPE_REGISTER_SPEC(AvfSaveVideoStage, kSpec)

}  // namespace vpipe
