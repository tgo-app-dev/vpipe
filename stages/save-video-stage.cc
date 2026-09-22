#include "stages/save-video-stage.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "common/encoded-segment.h"
#include "stages/model-provenance.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
}

using namespace std;

namespace vpipe {

namespace {

void
fill_dict_from_options_(const FlexData& opts,
                        const LibAvUtil& avu,
                        AVDictionary**   out)
{
  if (!opts.is_object()) {
    return;
  }
  for (auto entry : opts.as_object()) {
    if (!entry.second.is_string()) {
      continue;
    }
    string key(entry.first);
    string val(entry.second.get_string());
    avu.api.dict_set(out, key.c_str(), val.c_str(), 0);
  }
}

// Where the provenance string lands. Reverse-DNS, which is how
// QuickTime's own vendor metadata is named (com.apple.quicktime.software
// is the tag this one stands in for), under the same prefix as the macOS
// app bundle -- see VPIPE_APP_BUNDLE_ID in apps/macos-app/CMakeLists.txt.
// Changing it orphans the tag in every file already written.
constexpr const char* kProvenanceKey = "com.tgous.vpipe.software";

// The muxers that share movenc.c, and so the `use_metadata_tags` flag
// below. Matched by name rather than probed: the option exists only on
// this family, and elsewhere it is ignored SILENTLY, which would leave a
// flag in the dict that reads as doing something and does not.
bool
is_mov_family_(const AVOutputFormat* f)
{
  if (f == nullptr || f->name == nullptr) {
    return false;
  }
  const string n(f->name);
  return n == "mp4" || n == "mov" || n == "ipod" || n == "3gp" ||
         n == "3g2" || n == "psp" || n == "f4v" || n == "ismv";
}

}

SaveVideoStage::SaveVideoStage
  (const SessionContextIntf* s,
   string                    id,
   vector<InEdge>            iports_in,
   FlexData                  config)
  : TypedStage<SaveVideoStage>(s, std::move(id),
                                      std::move(iports_in),
                                      std::move(config))
  , _libs(s->services()->ffmpeg_libraries())
{
  // Nested video/audio defaults: no flat ConfigKey representation, so
  // they are seeded here (the single source) and overridden from the
  // video.* / audio.* sub-objects below.
  // h264_videotoolbox, matching hls-broadcast: it is part of the OS,
  // hardware accelerated, and carries no licence obligation, where
  // libx264 is GPL and so cannot be in an FFmpeg build this project
  // redistributes. Non-Apple builds fall back to libx264 in
  // init_video_encoder_().
  _video_codec    = "h264_videotoolbox";
  _video_bitrate  = 2'000'000;
  _video_preset   = "medium";
  _video_gop_size = 60;
  _audio_codec    = "aac";
  _audio_bitrate  = 128'000;

  // Flat attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _output_url   = attr_path("output_url", true);
  _format       = attr_str("format");
  _enable_video = attr_bool("enable_video");
  _enable_audio = attr_bool("enable_audio");

  const FlexData& cfg = this->config();
  // Whether a key was WRITTEN, as opposed to resolving to its default.
  // The flat keys have to override the superseded nesting only where
  // they were actually set, or a graph using the old spelling would
  // have every setting overwritten by a default it never asked for.
  const auto written = [this](const char* k) {
    const FlexData& c = this->config();
    if (!c.is_object()) { return false; }
    return c.as_object().contains(k);
  };
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("video")) {
      FlexData v_val = root.at("video");
      if (v_val.is_object()) {
        auto vo = v_val.as_object();
        if (vo.contains("codec")) {
          _video_codec = string(vo.at("codec").as_string(_video_codec));
        }
        if (vo.contains("bitrate")) {
          _video_bitrate = vo.at("bitrate").as_int(_video_bitrate);
        }
        if (vo.contains("preset")) {
          _video_preset =
            string(vo.at("preset").as_string(_video_preset));
        }
        if (vo.contains("crf")) {
          _video_has_crf = true;
          _video_crf = static_cast<int>(vo.at("crf").as_int(0));
        }
        if (vo.contains("gop_size")) {
          _video_gop_size =
            static_cast<int>(vo.at("gop_size").as_int(_video_gop_size));
        }
        if (vo.contains("options")) {
          _video_options = vo.at("options");
        }
      }
    }
    if (root.contains("audio")) {
      FlexData a_val = root.at("audio");
      if (a_val.is_object()) {
        auto ao = a_val.as_object();
        if (ao.contains("codec")) {
          _audio_codec = string(ao.at("codec").as_string(_audio_codec));
        }
        if (ao.contains("bitrate")) {
          _audio_bitrate = ao.at("bitrate").as_int(_audio_bitrate);
        }
        if (ao.contains("options")) {
          _audio_options = ao.at("options");
        }
      }
    }
    if (root.contains("muxer_options")) {
      _muxer_options = root.at("muxer_options");
    }
    if (root.contains("video") || root.contains("audio")) {
      session()->warn(fmt(
          "SaveVideoStage('{}'): the nested `video` / `audio` config "
          "objects are superseded by flat keys (video_codec, "
          "video_bitrate, video_preset, video_crf, video_gop_size, "
          "video_options; audio_codec, audio_bitrate, audio_options). "
          "Still honoured, so this graph runs unchanged -- but only the "
          "flat keys carry a default and a description an editor can "
          "show", this->id()));
    }
  }

  // The flat keys LAST, so an explicitly-set one wins over the same
  // setting arriving through the superseded nesting. Read through
  // `written` rather than unconditionally: attr_* cannot distinguish a
  // configured value from the spec default, and applying the default
  // here would silently undo a nested setting.
  if (written("video_codec"))   { _video_codec   = attr_str("video_codec"); }
  if (written("video_bitrate")) { _video_bitrate = attr_int("video_bitrate"); }
  if (written("video_preset"))  { _video_preset  = attr_str("video_preset"); }
  if (written("video_gop_size")) {
    _video_gop_size = static_cast<int>(attr_int("video_gop_size"));
  }
  if (written("video_crf")) {
    // NEGATIVE means unset, and that distinction is load-bearing: crf 0
    // is lossless, a legal request, so absence cannot be spelled as 0.
    const std::int64_t v = attr_int("video_crf");
    _video_has_crf = v >= 0;
    if (_video_has_crf) { _video_crf = static_cast<int>(v); }
  }
  if (written("video_options")) { _video_options = attr("video_options"); }
  if (written("audio_codec"))   { _audio_codec   = attr_str("audio_codec"); }
  if (written("audio_bitrate")) { _audio_bitrate = attr_int("audio_bitrate"); }
  if (written("audio_options")) { _audio_options = attr("audio_options"); }

  // Validation is deferred to launch (see Stage::fail_config).
  if (_output_url.empty()) {
    fail_config(fmt(
      "SaveVideoStage('{}'): config.output_url is required",
      this->id()));
  }
  if (!_enable_video && !_enable_audio) {
    fail_config(fmt(
      "SaveVideoStage('{}'): at least one of enable_video / "
      "enable_audio must be true",
      this->id()));
  }

  // Assign port indices.
  unsigned next = 0;
  if (_enable_video) {
    _video_port = static_cast<int>(next++);
  }
  if (_enable_audio) {
    _audio_port = static_cast<int>(next++);
  }
  if (this->num_iports() != next) {
    fail_config(fmt(
      "SaveVideoStage('{}'): expected {} input edge(s) "
      "(enable_video={}, enable_audio={}) but got {}",
      this->id(), next, _enable_video, _enable_audio,
      this->num_iports()));
  }

  allocate_oports(spec().oports.size());   // sink: 0 oports

  // Allocate scratch packet for receive_packet drains.
  _enc_pkt = _libs->avcodec().api.packet_alloc();
  if (!_enc_pkt) {
    fail_config(fmt(
      "SaveVideoStage('{}'): av_packet_alloc failed",
      this->id()));
  }
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "output_url", .type = ConfigType::String, .required = true,
   .doc = "output file path / URL",
   .is_path = true, .path_write = true, .path_filter = "video"},
  {.key = "format", .type = ConfigType::String,
   .doc = "container format; \"\" = inferred", .def_str = ""},
  {.key = "enable_video", .type = ConfigType::Bool,
   .doc = "encode a video stream", .def_bool = true},
  {.key = "enable_audio", .type = ConfigType::Bool,
   .doc = "encode an audio stream", .def_bool = true},
  // FLAT, one key per setting, because an editor can only show a default
  // and a doc for something the spec names. These lived in a `video` /
  // `audio` Object whose doc was a slash-separated list of sub-keys --
  // readable in source, and nothing the web-ui could turn into a form:
  // an Object is one opaque field, so every default and every
  // explanation below was invisible to whoever was filling it in.
  {.key = "video_codec", .type = ConfigType::String,
   .doc = "video encoder. The default is part of macOS, hardware "
          "accelerated, and carries no licence obligation -- libx264 is "
          "GPL and cannot ship in an FFmpeg build this project "
          "redistributes. A non-Apple build falls back to libx264 at "
          "init. Anything the linked libavcodec provides is accepted",
   .def_str = "h264_videotoolbox"},
  {.key = "video_bitrate", .type = ConfigType::Int,
   .doc = "target video bitrate in bits per second. Ignored by an "
          "encoder run in constant-quality mode -- see video_crf",
   .def_int = 2000000},
  {.key = "video_preset", .type = ConfigType::String,
   .doc = "encoder speed/size preset (ultrafast .. veryslow), for the "
          "encoders that take one; ignored by those that do not",
   .def_str = "medium"},
  {.key = "video_crf", .type = ConfigType::Int,
   .doc = "constant-quality target instead of a bitrate: lower is "
          "better, 0 is lossless, ~18-28 is the usual range. -1 (the "
          "default) leaves it UNSET, and video_bitrate decides. Set by "
          "the encoders that support crf; ignored by the others",
   .def_int = -1},
  {.key = "video_gop_size", .type = ConfigType::Int,
   .doc = "keyframe interval in frames. Smaller seeks and recovers "
          "faster and costs size",
   .def_int = 60},
  {.key = "video_options", .type = ConfigType::Object,
   .doc = "extra av_dict options passed to the video encoder, for "
          "anything the keys above do not name"},
  {.key = "audio_codec", .type = ConfigType::String,
   .doc = "audio encoder", .def_str = "aac"},
  {.key = "audio_bitrate", .type = ConfigType::Int,
   .doc = "target audio bitrate in bits per second", .def_int = 128000},
  {.key = "audio_options", .type = ConfigType::Object,
   .doc = "extra av_dict options passed to the audio encoder"},
  {.key = "muxer_options", .type = ConfigType::Object,
   .doc = "av_dict string opts for the muxer. NOT flattened: unlike the "
          "encoder settings above these have no fixed set of names -- "
          "they are whatever the chosen container takes"},

  // The superseded nesting. Still READ, so a pipeline written against it
  // keeps working, and warned about so it does not keep working
  // silently: an unknown key is not an error anywhere in this runtime,
  // which is what makes a quiet removal the wrong way to do this. Kept
  // in the spec rather than dropped so an editor showing an old
  // pipeline can say what the field is instead of leaving it unnamed.
  {.key = "video", .type = ConfigType::Object,
   .doc = "SUPERSEDED by video_codec / video_bitrate / video_preset / "
          "video_crf / video_gop_size / video_options. Still read, and "
          "warned about; a flat key set alongside it wins"},
  {.key = "audio", .type = ConfigType::Object,
   .doc = "SUPERSEDED by audio_codec / audio_bitrate / audio_options. "
          "Still read, and warned about; a flat key set alongside it "
          "wins"},
};
// Canonical iports (video first, audio second); either may be disabled
// via config, so the live count comes from the wiring. Each port
// carries a StreamParams header then FrameRef beats (mixed -> untyped).
// Clock groups are authoritatively reported by iport_clock_group()
// (video 0, audio 1).
const PortSpec kIports[] = {
  {.name = "video", .doc = "VideoStreamParams header then video FrameRefs",
   .type = nullptr, .clock_group = 0},
  {.name = "audio",
   .doc = "AudioStreamParams header then audio FrameRefs, OR f32 PCM "
          "TensorBeats [channels, n_samples] with a sample_rate sideband",
   .type = nullptr, .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "save-video",
  .doc       = "Sink: encodes incoming video/audio frames (H.264 + AAC "
               "by default) and muxes them into a container file. 0 "
               "oports; iports run on independent per-stream clocks.",
  .display_name = "Save Video",
  .category  = StageCategory::Visual,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
SaveVideoStage::spec() const noexcept
{
  return kSpec;
}

SaveVideoStage::~SaveVideoStage()
{
  release_media_();
  // Ctor-owned scratch, not per-run state -- freed only here.
  if (_enc_pkt) {
    _libs->avcodec().api.packet_free(&_enc_pkt);
  }
}

void
SaveVideoStage::release_media_()
{
  if (!_finalized) {
    // Best-effort cleanup if the pipeline never reached the EOS path.
    finalize_();
  }
  if (_venc) {
    _libs->avcodec().api.free_context(&_venc);
  }
  if (_aenc) {
    _libs->avcodec().api.free_context(&_aenc);
  }
  if (_ofctx) {
    if (_ofctx->pb) {
      _libs->avformat().api.avio_closep(&_ofctx->pb);
    }
    _libs->avformat().api.free_context(_ofctx);
    _ofctx = nullptr;
  }
  if (_apcm_frame) {
    _libs->avutil().api.frame_free(&_apcm_frame);
    _apcm_frame = nullptr;
  }
  _vstream = nullptr;
  _astream = nullptr;
}

void
SaveVideoStage::reset_run_state()
{
  // Free the PREVIOUS run's muxer + encoders before clearing the flags
  // that say they exist -- otherwise this launch rebuilds them and the
  // old ones leak until the stage is destroyed.
  release_media_();
  _video_initialized = false;
  _audio_initialized = false;
  _video_eos         = false;
  _audio_eos         = false;
  _header_written    = false;
  _finalized         = false;
  _audio_pcm         = false;
  _audio_pts         = 0;
  _apcm_carry.clear();
  _apcm_carry_n      = 0;
  _next_port         = 0;
  _model_name.clear();
}

string
SaveVideoStage::av_err_(int rc) const
{
  char buf[256];
  _libs->avutil().api.strerror(rc, buf, sizeof buf);
  return string(buf);
}

void
SaveVideoStage::ensure_output_format_()
{
  if (_ofctx) {
    return;
  }
  AVFormatContext* ofctx = nullptr;
  const char* fmt_name = _format.empty() ? nullptr : _format.c_str();
  int rc = _libs->avformat().api.alloc_output_context2(
    &ofctx, nullptr, fmt_name, _output_url.c_str());
  if (rc < 0 || !ofctx) {
    session()->error(fmt(
        "SaveVideoStage('{}'): alloc_output_context2 failed: "
        "{}", this->id(), av_err_(rc)));
  }
  _ofctx = ofctx;
}

void
SaveVideoStage::init_video_encoder_(const VideoStreamParams& p)
{
  ensure_output_format_();

  const AVCodec* codec = _libs->avcodec().api.find_encoder_by_name(
    _video_codec.c_str());

  // The default is h264_videotoolbox, which exists only on Apple
  // platforms. Fall back to libx264 so a Linux build of an
  // unconfigured graph still encodes.
  //
  // Only for the DEFAULT: an explicitly configured encoder is a
  // deliberate choice, and quietly substituting another would be worse
  // than saying it is missing.
  if (!codec && _video_codec == "h264_videotoolbox") {
    codec = _libs->avcodec().api.find_encoder_by_name("libx264");
    if (codec) {
      session()->log_normal(fmt(
          "SaveVideoStage('{}'): h264_videotoolbox is not in this FFmpeg "
          "build; encoding with libx264 instead",
          this->id()));
      _video_codec = "libx264";
    }
  }

  if (!codec) {
    session()->error(fmt(
        "SaveVideoStage('{}'): video encoder '{}' not found",
        this->id(), _video_codec));
  }

  _venc = _libs->avcodec().api.alloc_context3(codec);
  if (!_venc) {
    session()->error(fmt(
        "SaveVideoStage('{}'): alloc_context3 (video) failed",
        this->id()));
  }

  _venc->width     = p.width;
  _venc->height    = p.height;
  _venc->pix_fmt   = static_cast<AVPixelFormat>(p.pix_fmt);
  // TAG the stream with what the producer says it wrote. Untagged, the
  // file is read by convention, and a convention that disagrees with
  // the producer costs 219/255 of the contrast plus a hue tilt -- which
  // is invisible on one decode and compounds in a graph that re-reads
  // its own output.
  _venc->color_range = static_cast<AVColorRange>(p.color_range);
  _venc->colorspace  = static_cast<AVColorSpace>(p.colorspace);
  _venc->bit_rate  = _video_bitrate;
  _venc->gop_size  = _video_gop_size;

  // Encoder time_base: prefer 1/frame_rate; fall back to input
  // stream's time_base; final fallback 1/25.
  if (p.frame_rate.num > 0 && p.frame_rate.den > 0) {
    _venc->time_base = AVRational{p.frame_rate.den, p.frame_rate.num};
    _venc->framerate = p.frame_rate;
  } else if (p.time_base.num > 0 && p.time_base.den > 0) {
    _venc->time_base = p.time_base;
  } else {
    _venc->time_base = AVRational{1, 25};
  }

  if (_ofctx->oformat
      && (_ofctx->oformat->flags & AVFMT_GLOBALHEADER)) {
    _venc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  AVDictionary* opts = nullptr;
  _libs->avutil().api.dict_set(&opts, "preset",
                              _video_preset.c_str(), 0);
  if (_video_has_crf) {
    string crf_s = to_string(_video_crf);
    _libs->avutil().api.dict_set(&opts, "crf", crf_s.c_str(), 0);
  }
  fill_dict_from_options_(_video_options, _libs->avutil(), &opts);

  int rc = _libs->avcodec().api.open2(_venc, codec, &opts);
  _libs->avutil().api.dict_free(&opts);
  if (rc < 0) {
    session()->error(fmt(
        "SaveVideoStage('{}'): avcodec_open2 (video) failed: "
        "{}", this->id(), av_err_(rc)));
  }

  _vstream = _libs->avformat().api.new_stream(_ofctx, codec);
  if (!_vstream) {
    session()->error(fmt(
        "SaveVideoStage('{}'): avformat_new_stream (video) "
        "failed", this->id()));
  }
  rc = _libs->avcodec().api.parameters_from_context(_vstream->codecpar,
                                                   _venc);
  if (rc < 0) {
    session()->error(fmt(
        "SaveVideoStage('{}'): parameters_from_context "
        "(video) failed: {}", this->id(), av_err_(rc)));
  }
  _vstream->time_base = _venc->time_base;
}

void
SaveVideoStage::init_audio_encoder_(const AudioStreamParams& p)
{
  ensure_output_format_();

  const AVCodec* codec = _libs->avcodec().api.find_encoder_by_name(
    _audio_codec.c_str());
  if (!codec) {
    session()->error(fmt(
        "SaveVideoStage('{}'): audio encoder '{}' not found",
        this->id(), _audio_codec));
  }

  _aenc = _libs->avcodec().api.alloc_context3(codec);
  if (!_aenc) {
    session()->error(fmt(
        "SaveVideoStage('{}'): alloc_context3 (audio) failed",
        this->id()));
  }

  _aenc->sample_rate = p.sample_rate;
  _aenc->sample_fmt  = static_cast<AVSampleFormat>(p.sample_fmt);
  // Bitwise copy is correct for AV_CHANNEL_ORDER_NATIVE (mask is a
  // value type). For CUSTOM the u.map pointer aliases the
  // upstream-owned buffer; safe as long as the producer outlives this
  // encoder, which is true within a single PipelineRuntime lifetime.
  _aenc->ch_layout   = p.ch_layout;
  _aenc->bit_rate    = _audio_bitrate;
  _aenc->time_base   = AVRational{1, p.sample_rate};

  if (_ofctx->oformat
      && (_ofctx->oformat->flags & AVFMT_GLOBALHEADER)) {
    _aenc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  // ASK THE ENCODER which sample format it takes, by trying. The PCM
  // arrives planar f32 and AAC accepts that as it stands, so this stage
  // handed the incoming format straight over -- and every LOSSLESS codec
  // then failed the open with a bare EINVAL, because ALAC and FLAC want
  // s16/s32. A table of what each codec accepts is the other way to do
  // it, but FFmpeg 8 removed `AVCodec::sample_fmts` in favour of a
  // query, so trying is both simpler and more honest than a list this
  // file would have to keep true. It costs one open per candidate, once
  // per run, and only when the first choice is refused.
  static constexpr AVSampleFormat kCandidates[] = {
    AV_SAMPLE_FMT_S32P, AV_SAMPLE_FMT_S32,
    AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_S16,
    AV_SAMPLE_FMT_FLT,
  };
  int rc = 0;
  for (int attempt = 0; ; ++attempt) {
    AVDictionary* opts = nullptr;
    fill_dict_from_options_(_audio_options, _libs->avutil(), &opts);
    rc = _libs->avcodec().api.open2(_aenc, codec, &opts);
    _libs->avutil().api.dict_free(&opts);
    if (rc >= 0) { break; }
    if (attempt >= (int)(sizeof kCandidates / sizeof *kCandidates)) {
      session()->error(fmt(
          "SaveVideoStage('{}'): avcodec_open2 (audio) failed for every "
          "sample format '{}' was offered: {}", this->id(), _audio_codec,
          av_err_(rc)));
      break;
    }
    // A refused open can leave the context unusable, so each attempt
    // gets a fresh one carrying the same stream parameters.
    _libs->avcodec().api.free_context(&_aenc);
    _aenc = _libs->avcodec().api.alloc_context3(codec);
    if (!_aenc) {
      session()->error(fmt(
          "SaveVideoStage('{}'): alloc_context3 (audio) failed",
          this->id()));
      break;
    }
    _aenc->sample_rate = p.sample_rate;
    _aenc->sample_fmt  = kCandidates[attempt];
    _aenc->ch_layout   = p.ch_layout;
    _aenc->bit_rate    = _audio_bitrate;
    _aenc->time_base   = AVRational{1, p.sample_rate};
    if (_ofctx->oformat
        && (_ofctx->oformat->flags & AVFMT_GLOBALHEADER)) {
      _aenc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
  }
  if (rc >= 0 && _aenc->sample_fmt != (AVSampleFormat)p.sample_fmt) {
    session()->info(fmt(
        "SaveVideoStage('{}'): '{}' takes {}, converting the planar f32 "
        "PCM into it", this->id(), _audio_codec,
        _libs->avutil().api.get_sample_fmt_name(_aenc->sample_fmt)));
  }

  _astream = _libs->avformat().api.new_stream(_ofctx, codec);
  if (!_astream) {
    session()->error(fmt(
        "SaveVideoStage('{}'): avformat_new_stream (audio) "
        "failed", this->id()));
  }
  rc = _libs->avcodec().api.parameters_from_context(_astream->codecpar,
                                                   _aenc);
  if (rc < 0) {
    session()->error(fmt(
        "SaveVideoStage('{}'): parameters_from_context "
        "(audio) failed: {}", this->id(), av_err_(rc)));
  }
  _astream->time_base = _aenc->time_base;
}

// Derive the audio stream header from a PCM TensorBeat. The rate must
// come from the beat: nothing else in a generative graph knows it, and
// a config default would silently retime the whole clip.
bool
SaveVideoStage::audio_params_from_pcm_(const TensorBeatPayload& t,
                                       AudioStreamParams* out)
{
  int channels = 1;
  int64_t samples = 0;
  if (t.shape.size() == 2) {
    channels = static_cast<int>(t.shape[0]);
    samples  = t.shape[1];
  } else if (t.shape.size() == 1) {
    samples = t.shape[0];
  } else {
    session()->warn(fmt(
      "encoder('{}'): audio PCM must be rank-1 [N] or rank-2 "
      "[channels,N], got rank {}", this->id(), t.shape.size()));
    return false;
  }
  int rate = 0;
  if (t.sideband.is_object()) {
    FlexData sb = t.sideband;              // as_object() is a view
    auto o = sb.as_object();
    if (o.contains("sample_rate")) {
      rate = static_cast<int>(o.at("sample_rate").as_int(0));
    }
    // The producer may state the channel count too. When it does it
    // WINS over the shape, so a mono clip shaped [1, N] and one shaped
    // [N] describe themselves identically.
    if (o.contains("channels")) {
      const int c = static_cast<int>(o.at("channels").as_int(0));
      if (c > 0) { channels = c; }
    }
  }
  if (rate <= 0) {
    session()->warn(fmt(
      "encoder('{}'): audio PCM beat carries no sample_rate sideband; "
      "refusing to guess one", this->id()));
    return false;
  }
  if (channels <= 0 || samples <= 0) {
    session()->warn(fmt(
      "encoder('{}'): audio PCM beat is {} x {}; nothing to encode",
      this->id(), channels, samples));
    return false;
  }
  out->sample_rate = rate;
  out->sample_fmt  = AV_SAMPLE_FMT_FLTP;   // planar f32: our layout
  out->time_base   = AVRational{1, rate};
  if (channels == 2) {
    const AVChannelLayout st = AV_CHANNEL_LAYOUT_STEREO;
    out->ch_layout = st;
  } else if (channels == 1) {
    const AVChannelLayout mo = AV_CHANNEL_LAYOUT_MONO;
    out->ch_layout = mo;
  } else {
    out->ch_layout = AVChannelLayout{};
    out->ch_layout.order       = AV_CHANNEL_ORDER_UNSPEC;
    out->ch_layout.nb_channels = channels;
  }
  return true;
}

// Encode a whole PCM beat. The encoder wants exactly frame_size
// samples per frame, so this chunks; the tail is zero-padded, which is
// why a producer should send one beat per clip rather than slicing a
// clip across beats.
void
SaveVideoStage::encode_pcm_(const TensorBeatPayload& t)
{
  if (!_aenc) { return; }
  const int channels = _aenc->ch_layout.nb_channels;
  const int64_t samples =
    (t.shape.size() == 2) ? t.shape[1]
                          : (t.shape.empty() ? 0 : t.shape[0]);
  if (samples <= 0 || channels <= 0) { return; }
  if (t.element_count() <
      static_cast<size_t>(samples) * static_cast<size_t>(channels)) {
    session()->warn(fmt(
      "encoder('{}'): audio PCM beat holds {} elements, its shape needs "
      "{}; dropping", this->id(), t.element_count(),
      static_cast<size_t>(samples) * channels));
    return;
  }
  const int frame_size =
    _aenc->frame_size > 0 ? _aenc->frame_size : 1024;

  if (!_apcm_frame) {
    _apcm_frame = _libs->avutil().api.frame_alloc();
    if (!_apcm_frame) { return; }
    _apcm_frame->nb_samples = frame_size;
    _apcm_frame->format     = _aenc->sample_fmt;
    _apcm_frame->ch_layout  = _aenc->ch_layout;
    _apcm_frame->sample_rate = _aenc->sample_rate;
    if (_libs->avutil().api.frame_get_buffer(_apcm_frame, 0) < 0) {
      _libs->avutil().api.frame_free(&_apcm_frame);
      _apcm_frame = nullptr;
      session()->warn(fmt(
        "encoder('{}'): could not allocate the audio PCM frame",
        this->id()));
      return;
    }
  }

  const float* src = t.as_f32();

  // WHOLE FRAMES ONLY, and the remainder is CARRIED to the next beat.
  //
  // A block codec takes a short frame only as the stream's LAST: FFmpeg
  // pads it, latches `last_audio_frame`, and answers the next
  // send_frame with EINVAL ("frame_size was not respected for a
  // non-last frame"). This used to end every BEAT with a short frame,
  // which a one-beat-per-clip graph never notices because its only
  // short frame really is the last. A beat boundary is not the end of
  // the stream, though: `audio-to-pcm` emits one per chunk and a joined
  // `load-video` one per file. MEASURED on the 5-part 896x512 story --
  // 5 x 324000 samples is 317 frames each, 1585 in all, and the encoder
  // took 627 then refused 958, writing 20 s of audio under 50 s of
  // video and still exiting 0.
  //
  // Carrying also keeps the old fix that motivated the short frame:
  // nothing is zero-padded mid-stream, so no silence is welded into a
  // seam. Only `flush_pcm_tail_` may send a partial frame.
  // Carry ++ beat, as one planar block. The carry is almost always
  // empty -- a graph that sends one beat per clip never fills it -- and
  // then the beat is encoded in place with no copy; `_apcm_join` is
  // touched only on the joined path that needs it. Note the channel
  // stride CHANGES when the two are spliced, so this cannot be done by
  // growing `_apcm_carry` in place: channel c moves.
  const int64_t have   = _apcm_carry_n + samples;
  const float*  base   = src;
  int64_t       stride = samples;
  if (_apcm_carry_n > 0) {
    _apcm_join.resize((size_t)channels * (size_t)have);
    for (int c = 0; c < channels; ++c) {
      float* d = _apcm_join.data() + (size_t)c * have;
      std::memcpy(d, _apcm_carry.data() + (size_t)c * _apcm_carry_n,
                  (size_t)_apcm_carry_n * sizeof(float));
      std::memcpy(d + _apcm_carry_n, src + (size_t)c * samples,
                  (size_t)samples * sizeof(float));
    }
    base   = _apcm_join.data();
    stride = have;
  }

  // `whole` is 0 when the two together still do not fill a frame, which
  // is the short-beat case and needs no branch of its own.
  const int64_t whole = (have / frame_size) * frame_size;
  emit_pcm_frames_(base, stride, channels, 0, whole, frame_size);

  const int64_t left = have - whole;
  std::vector<float> keep((size_t)channels * (size_t)left);
  for (int c = 0; c < channels; ++c) {
    std::memcpy(keep.data() + (size_t)c * left,
                base + (size_t)c * stride + whole,
                (size_t)left * sizeof(float));
  }
  _apcm_carry.swap(keep);
  _apcm_carry_n = left;
}

void
SaveVideoStage::emit_pcm_frames_(const float* src, int64_t samples,
                                 int channels, int64_t off, int64_t count,
                                 int frame_size)
{
  for (int64_t i = 0; i < count; i += frame_size) {
    _apcm_frame->nb_samples = frame_size;
    write_pcm_frame_(src, samples, channels, off + i, frame_size);
    _apcm_frame->pts = _audio_pts;
    _audio_pts += frame_size;
    encode_and_mux_(static_cast<unsigned>(_audio_port), _apcm_frame);
  }
}

// The stream's one legitimate short frame. `_audio_pts` still counts
// only REAL samples, so the true-length metadata the seam fix relies on
// stays exact even though this frame is padded on the way out.
void
SaveVideoStage::flush_pcm_tail_()
{
  if (_aenc == nullptr || _apcm_frame == nullptr || _apcm_carry_n <= 0) {
    return;
  }
  const int channels = _aenc->ch_layout.nb_channels;
  const int n        = (int)_apcm_carry_n;
  _apcm_frame->nb_samples = n;
  write_pcm_frame_(_apcm_carry.data(), _apcm_carry_n, channels, 0, n);
  _apcm_frame->pts = _audio_pts;
  _audio_pts += n;
  encode_and_mux_(static_cast<unsigned>(_audio_port), _apcm_frame);
  _apcm_carry.clear();
  _apcm_carry_n = 0;
}

// Planar f32 -- the layout every generative graph in this tree carries
// PCM in -- into whatever the chosen encoder accepts. AAC takes fltp as
// it stands; the LOSSLESS codecs do not, and they are the reason a
// multi-part graph can join its parts without a seam.
void
SaveVideoStage::write_pcm_frame_(const float* src, std::int64_t samples,
                                 int channels, std::int64_t off, int n)
{
  const auto fmt = static_cast<AVSampleFormat>(_apcm_frame->format);
  auto ch_src = [&](int c) { return src + (size_t)c * samples + off; };
  // Scale to the integer range as FFmpeg does: full-scale float 1.0 maps
  // to the type's maximum, and everything is clamped rather than wrapped
  // -- a sample a hair over 1.0 must clip, not invert.
  auto clampf = [](float v, double peak) {
    const double x = (double)v * peak;
    return x >  peak - 1 ?  peak - 1 : (x < -peak ? -peak : x);
  };
  switch (fmt) {
  case AV_SAMPLE_FMT_FLTP:
    for (int c = 0; c < channels; ++c) {
      std::memcpy(_apcm_frame->data[c], ch_src(c), (size_t)n * sizeof(float));
    }
    break;
  case AV_SAMPLE_FMT_S32P:
    for (int c = 0; c < channels; ++c) {
      auto* d = reinterpret_cast<std::int32_t*>(_apcm_frame->data[c]);
      const float* s = ch_src(c);
      for (int i = 0; i < n; ++i) { d[i] = (std::int32_t)clampf(s[i], 2147483648.0); }
    }
    break;
  case AV_SAMPLE_FMT_S16P:
    for (int c = 0; c < channels; ++c) {
      auto* d = reinterpret_cast<std::int16_t*>(_apcm_frame->data[c]);
      const float* s = ch_src(c);
      for (int i = 0; i < n; ++i) { d[i] = (std::int16_t)clampf(s[i], 32768.0); }
    }
    break;
  case AV_SAMPLE_FMT_FLT: {
    auto* d = reinterpret_cast<float*>(_apcm_frame->data[0]);
    for (int c = 0; c < channels; ++c) {
      const float* s = ch_src(c);
      for (int i = 0; i < n; ++i) { d[(size_t)i * channels + c] = s[i]; }
    }
    break;
  }
  case AV_SAMPLE_FMT_S32: {
    auto* d = reinterpret_cast<std::int32_t*>(_apcm_frame->data[0]);
    for (int c = 0; c < channels; ++c) {
      const float* s = ch_src(c);
      for (int i = 0; i < n; ++i) {
        d[(size_t)i * channels + c] = (std::int32_t)clampf(s[i], 2147483648.0);
      }
    }
    break;
  }
  case AV_SAMPLE_FMT_S16: {
    auto* d = reinterpret_cast<std::int16_t*>(_apcm_frame->data[0]);
    for (int c = 0; c < channels; ++c) {
      const float* s = ch_src(c);
      for (int i = 0; i < n; ++i) {
        d[(size_t)i * channels + c] = (std::int16_t)clampf(s[i], 32768.0);
      }
    }
    break;
  }
  default:
    break;   // refused at open, so unreachable
  }
}

bool
SaveVideoStage::ready_to_write_header_() const noexcept
{
  // A port that reached EOS without ever delivering a header counts as
  // settled, not as still-pending. Otherwise one failed or empty stream
  // holds the header back forever and the file is never opened AT ALL
  // -- losing the picture because the soundtrack was unusable, with
  // nothing written to say so.
  bool v_ok = (!_enable_video) || _video_initialized || _video_eos;
  bool a_ok = (!_enable_audio) || _audio_initialized || _audio_eos;
  // ...but there has to be at least one real stream to write.
  return v_ok && a_ok && (_video_initialized || _audio_initialized);
}

void
SaveVideoStage::write_provenance_(AVDictionary** mux_opts)
{
  // Only for a clip a model made. A plain load-video -> save-video
  // transcode must not come out claiming vpipe authored the footage --
  // and saying nothing also leaves the container's metadata layout
  // exactly as it would have been.
  if (_ofctx == nullptr) {
    return;
  }
  // The audio sample count is an arbitrary key too, and it is written
  // at TRAILER time -- so the mdta form has to be asked for here, up
  // front, even when there is no provenance string to write.
  const bool want_mdta = _audio_port >= 0;
  if (_model_name.empty() && !want_mdta) {
    return;
  }
  if (_model_name.empty()) {
    if (is_mov_family_(_ofctx->oformat)) {
      _libs->avutil().api.dict_set(mux_opts, "movflags",
                                   "+use_metadata_tags", AV_DICT_APPEND);
    }
    return;
  }
  const string sw = provenance::software_string(_model_name);
  _libs->avutil().api.dict_set(&_ofctx->metadata, kProvenanceKey,
                               sw.c_str(), 0);
  if (!is_mov_family_(_ofctx->oformat)) {
    // Matroska needs nothing further: it carries arbitrary tag names
    // natively. Containers that carry none drop the entry, which is
    // the same as not having written it.
    return;
  }
  // MP4 will not carry a key it does not recognise. Its default `mdir`
  // metadata holds a fixed set of iTunes atoms and drops everything
  // else without a word; `use_metadata_tags` switches moov/udta/meta to
  // the QuickTime `mdta` form, whose `keys` box holds arbitrary
  // reverse-DNS names.
  //
  // The obvious slot -- (c)too, "encoding tool", which is where an
  // `encoder` tag goes and the nearest thing MP4 has to EXIF Software
  // -- is NOT available: avformat_write_header overwrites that tag
  // with its own Lavf<version> whatever the caller set.
  //
  // APPEND rather than assign, because movflags is a single string and
  // the muxer_options config may already have put faststart in it.
  _libs->avutil().api.dict_set(mux_opts, "movflags",
                               "+use_metadata_tags", AV_DICT_APPEND);
}

void
SaveVideoStage::open_output_and_write_header_()
{
  // For non-NOFILE muxers we need to open the IO context.
  if (_ofctx->oformat
      && !(_ofctx->oformat->flags & AVFMT_NOFILE))
  {
    int rc = _libs->avformat().api.avio_open(&_ofctx->pb,
                                             _output_url.c_str(),
                                             AVIO_FLAG_WRITE);
    if (rc < 0) {
      session()->error(fmt(
          "SaveVideoStage('{}'): avio_open('{}') failed: {}",
          this->id(), _output_url, av_err_(rc)));
    }
  }

  AVDictionary* mux_opts = nullptr;
  fill_dict_from_options_(_muxer_options, _libs->avutil(), &mux_opts);
  // After the config's own options, so the APPEND inside can add to a
  // movflags the user already set rather than replacing it.
  write_provenance_(&mux_opts);
  int rc = _libs->avformat().api.write_header(_ofctx, &mux_opts);
  _libs->avutil().api.dict_free(&mux_opts);
  if (rc < 0) {
    session()->error(fmt(
        "SaveVideoStage('{}'): avformat_write_header failed: "
        "{}", this->id(), av_err_(rc)));
  }
}

void
SaveVideoStage::drain_encoder_(AVCodecContext* enc, AVStream* st)
{
  while (true) {
    int rc = _libs->avcodec().api.receive_packet(enc, _enc_pkt);
    if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
      break;
    }
    if (rc < 0) {
      session()->warn(fmt(
        "encoder('{}'): receive_packet: {}",
        this->id(), av_err_(rc)));
      break;
    }
    _enc_pkt->stream_index = st->index;
    // An encoder that leaves duration at 0 costs the LAST sample its
    // length: the mov muxer derives every other sample's duration from
    // the next packet's dts, and there is no next packet for the final
    // one. The track then ends one frame early, the edit list is
    // written to that short duration, and a conformant demuxer trims
    // the final frame -- a 33-frame clip plays back 32. Fill it in
    // from the encoder's own rate before the rescale, which carries
    // duration across with the timestamps.
    if (_enc_pkt->duration <= 0) {
      if (enc->codec_type == AVMEDIA_TYPE_VIDEO
          && enc->framerate.num > 0 && enc->framerate.den > 0)
      {
        _enc_pkt->duration = _libs->avutil().api.rescale_q(
            1, av_inv_q(enc->framerate), enc->time_base);
      } else if (enc->codec_type == AVMEDIA_TYPE_AUDIO
                 && enc->frame_size > 0 && enc->sample_rate > 0)
      {
        _enc_pkt->duration = _libs->avutil().api.rescale_q(
            enc->frame_size, AVRational{1, enc->sample_rate},
            enc->time_base);
      }
    }
    _libs->avcodec().api.packet_rescale_ts(_enc_pkt,
                                          enc->time_base,
                                          st->time_base);
    rc = _libs->avformat().api.interleaved_write_frame(_ofctx,
                                                      _enc_pkt);
    if (rc < 0) {
      session()->warn(fmt(
        "encoder('{}'): interleaved_write_frame: {}",
        this->id(), av_err_(rc)));
    }
    // interleaved_write_frame consumes the packet (unrefs).
  }
}

void
SaveVideoStage::encode_and_mux_(unsigned port, AVFrame* frame)
{
  AVCodecContext* enc = nullptr;
  AVStream*       st  = nullptr;
  if (static_cast<int>(port) == _video_port) {
    enc = _venc;
    st  = _vstream;
  } else if (static_cast<int>(port) == _audio_port) {
    enc = _aenc;
    st  = _astream;
  }
  if (!enc || !st) {
    return;
  }
  int rc = _libs->avcodec().api.send_frame(enc, frame);
  if (rc < 0) {
    session()->warn(fmt(
      "encoder('{}'): send_frame port {}: {}",
      this->id(), port, av_err_(rc)));
    return;
  }
  drain_encoder_(enc, st);
}

void
SaveVideoStage::finalize_()
{
  if (_finalized) {
    return;
  }
  if (_header_written) {
    if (_venc) {
      _libs->avcodec().api.send_frame(_venc, nullptr);
      drain_encoder_(_venc, _vstream);
    }
    if (_aenc) {
      // The samples that never filled a frame. HERE is the one place a
      // short frame is legal, and it has to go in before the drain or
      // the tail of the last beat is simply lost.
      flush_pcm_tail_();
      _libs->avcodec().api.send_frame(_aenc, nullptr);
      drain_encoder_(_aenc, _astream);
    }
    // RECORD THE TRUE AUDIO LENGTH. `_audio_pts` counted the real
    // samples handed to the encoder, which is the one number the
    // container will not otherwise carry: a block codec rounds the tail
    // up to a whole frame and primes its decoder with another, so an
    // mp4's audio duration is the PADDED count. A reader that trims to
    // this can join two parts without the gap those two paddings would
    // otherwise weld into the seam. Set before the trailer because that
    // is when the mov muxer writes moov.
    if (_aenc != nullptr && _audio_pts > 0) {
      const std::string n = std::to_string((long long)_audio_pts);
      _libs->avutil().api.dict_set(&_ofctx->metadata,
                                   kAudioSamplesMetaKey, n.c_str(), 0);
    }
    int rc = _libs->avformat().api.write_trailer(_ofctx);
    if (rc < 0) {
      session()->warn(fmt(
        "encoder('{}'): write_trailer: {}",
        this->id(), av_err_(rc)));
    }
  }
  if (_ofctx && _ofctx->pb
      && _ofctx->oformat
      && !(_ofctx->oformat->flags & AVFMT_NOFILE))
  {
    _libs->avformat().api.avio_closep(&_ofctx->pb);
  }
  _finalized = true;
}

Job
SaveVideoStage::initialize(RuntimeContext& /*ctx*/)
{
  // Output context, encoder contexts, and the file IO context all
  // depend on stream parameters that arrive as the first token on
  // each input port. Setup is therefore deferred to process().
  co_return;
}

Job
SaveVideoStage::drain(RuntimeContext& /*ctx*/)
{
  // Flush each encoder, write the muxer trailer, close the IO
  // context. Idempotent via _finalized.
  finalize_();
  co_return;
}

Job
SaveVideoStage::process(RuntimeContext& ctx)
{
  // Are we entirely done?
  bool v_done = (_video_port < 0) || _video_eos;
  bool a_done = (_audio_port < 0) || _audio_eos;
  if (v_done && a_done) {
    ctx.signal_done();
    co_return;
  }

  // Pick a non-EOS port via round-robin.
  int target = _next_port;
  for (int tries = 0; tries < 2; ++tries) {
    if (target == _video_port && !_video_eos && _video_port >= 0) {
      break;
    }
    if (target == _audio_port && !_audio_eos && _audio_port >= 0) {
      break;
    }
    target = (target + 1) % static_cast<int>(this->num_iports());
  }
  unsigned port = static_cast<unsigned>(target);

  auto t = co_await ctx.read(port);
  if (!t) {
    if (static_cast<int>(port) == _video_port) {
      _video_eos = true;
    } else if (static_cast<int>(port) == _audio_port) {
      _audio_eos = true;
    }
    // This EOS may be what unblocks the header: the other stream can
    // be initialized and waiting on a peer that turned out to be empty.
    if (!_header_written && ready_to_write_header_()) {
      try {
        open_output_and_write_header_();
        _header_written = true;
      } catch (const exception& e) {
        session()->warn(fmt(
          "encoder('{}'): writing the header after port {} EOS: {}",
          this->id(), port, e.what()));
      }
    }
    if (this->num_iports() > 1) {
      _next_port =
        (target + 1) % static_cast<int>(this->num_iports());
    }
    co_return;
  }

  bool& init = (static_cast<int>(port) == _video_port)
                 ? _video_initialized : _audio_initialized;
  if (!init) {
    try {
      if (static_cast<int>(port) == _video_port) {
        const auto* p = dynamic_cast<const VideoStreamParamsPayload*>(
            t.get());
        if (!p) {
          session()->error(fmt(
              "SaveVideoStage('{}'): expected "
              "VideoStreamParams header on video port",
              this->id()));
        }
        // Before the encoder, because init_video_encoder_ takes the
        // params SLICE and provenance is not in it.
        if (!p->model_name.empty()) { _model_name = p->model_name; }
        init_video_encoder_(*p);
      } else if (const auto* pcm =
                     dynamic_cast<const TensorBeatPayload*>(t.get())) {
        // Raw PCM: this beat is BOTH the header and the first samples.
        // A generative graph has no ffmpeg decoder upstream to emit a
        // real header, so the rate and channel count are read off the
        // beat itself.
        AudioStreamParams ap{};
        if (pcm->dtype != TensorBeat::DType::F32 ||
            !audio_params_from_pcm_(*pcm, &ap)) {
          session()->warn(fmt(
            "encoder('{}'): unusable audio PCM beat; writing the file "
            "WITHOUT a soundtrack", this->id()));
          _audio_eos = true;
          if (!_header_written && ready_to_write_header_()) {
            open_output_and_write_header_();
            _header_written = true;
          }
          co_return;
        }
        _audio_pcm = true;
        // The audio port is the ONLY source of provenance when the
        // picture is disabled, so read it here too. First non-empty
        // wins; the two ports of one graph carry the same name.
        if (_model_name.empty()) {
          _model_name = provenance::model_name(pcm->sideband);
        }
        init_audio_encoder_(ap);
        init = true;
        if (ready_to_write_header_() && !_header_written) {
          open_output_and_write_header_();
          _header_written = true;
        }
        if (_header_written) { encode_pcm_(*pcm); }
        if (this->num_iports() > 1) {
          _next_port = (target + 1) % static_cast<int>(this->num_iports());
        }
        co_return;
      } else {
        const auto* p = dynamic_cast<const AudioStreamParamsPayload*>(
            t.get());
        if (!p) {
          session()->error(fmt(
              "SaveVideoStage('{}'): expected an AudioStreamParams "
              "header or an f32 PCM TensorBeat on the audio port",
              this->id()));
        }
        init_audio_encoder_(*p);
      }
      init = true;
      if (ready_to_write_header_() && !_header_written) {
        open_output_and_write_header_();
        _header_written = true;
      }
    } catch (const exception& e) {
      session()->warn(fmt(
        "encoder('{}'): init port {}: {}; stopping",
        this->id(), port, e.what()));
      ctx.signal_done();
      co_return;
    }
  } else {
    if (!_header_written) {
      // Producer-contract violation: frame before paired header.
      session()->warn(fmt(
        "encoder('{}'): port {} produced a frame before all paired "
        "headers arrived; dropping",
        this->id(), port));
    } else if (_audio_pcm && static_cast<int>(port) == _audio_port) {
      const auto* pcm = dynamic_cast<const TensorBeatPayload*>(t.get());
      if (!pcm || pcm->dtype != TensorBeat::DType::F32) {
        session()->warn(fmt(
          "encoder('{}'): non-PCM token on the raw-audio port; dropping",
          this->id()));
      } else {
        encode_pcm_(*pcm);
      }
    } else {
      const auto* fp = dynamic_cast<const FrameRefPayload*>(t.get());
      if (!fp || !fp->ref) {
        session()->warn(fmt(
          "encoder('{}'): non-FrameRef token on port {}; dropping",
          this->id(), port));
      } else {
        encode_and_mux_(port, fp->ref.get());
      }
    }
  }

  if (this->num_iports() > 1) {
    _next_port = (target + 1) % static_cast<int>(this->num_iports());
  }
}

VPIPE_REGISTER_STAGE(SaveVideoStage)
VPIPE_REGISTER_SPEC(SaveVideoStage, kSpec)

}
