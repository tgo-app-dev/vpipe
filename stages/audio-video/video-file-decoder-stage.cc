#include "stages/audio-video/video-file-decoder-stage.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
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

// Reads an object<string,string> from FlexData and forwards each pair
// into the AVDictionary via the avutil api.
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

}

VideoFileDecoderStage::VideoFileDecoderStage
  (const SessionContextIntf* s,
   string                    id,
   vector<InEdge>            iports,
   FlexData                  config)
  : TypedStage<VideoFileDecoderStage>(s, std::move(id),
                                      std::move(iports),
                                      std::move(config))
  , _libs(s->ffmpeg_libraries())
{
  // Stage's _config now holds the moved-in FlexData. Read everything
  // we need up-front; throw via session()->error if `input_url` is
  // missing because there's no graceful default.
  // Attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _input_url          = attr_str("input_url");
  _format             = attr_str("format");
  _enable_video       = attr_bool("enable_video");
  _enable_audio       = attr_bool("enable_audio");
  _video_stream_index = static_cast<int>(attr_int("video_stream_index"));
  _audio_stream_index = static_cast<int>(attr_int("audio_stream_index"));
  _read_timeout_ms    = static_cast<int>(attr_int("read_timeout_ms"));
  _open_options       = attr("options");

  // Validation is deferred to launch (see Stage::fail_config).
  if (_input_url.empty()) {
    fail_config(fmt(
      "VideoFileDecoderStage('{}'): config.input_url is required",
      this->id()));
  }
  if (!_enable_video && !_enable_audio) {
    fail_config(fmt(
      "VideoFileDecoderStage('{}'): at least one of enable_video / "
      "enable_audio must be true",
      this->id()));
  }

  // Assign port indices: video first, then audio.
  unsigned next = 0;
  if (_enable_video) {
    _video_port = static_cast<int>(next++);
  }
  if (_enable_audio) {
    _audio_port = static_cast<int>(next++);
  }
  allocate_oports(next);
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "input_url", .type = ConfigType::String, .required = true,
   .doc = "file path or network URL (rtsp/http/...)"},
  {.key = "format", .type = ConfigType::String,
   .doc = "forced demuxer; \"\" = autodetect", .def_str = ""},
  {.key = "enable_video", .type = ConfigType::Bool,
   .doc = "emit video oport", .def_bool = true},
  {.key = "enable_audio", .type = ConfigType::Bool,
   .doc = "emit audio oport", .def_bool = true},
  {.key = "video_stream_index", .type = ConfigType::Int,
   .doc = "stream to use; -1 = first video", .def_int = -1},
  {.key = "audio_stream_index", .type = ConfigType::Int,
   .doc = "stream to use; -1 = first audio", .def_int = -1},
  {.key = "read_timeout_ms", .type = ConfigType::Int,
   .doc = "network read timeout ms; 0 = none", .def_int = 0},
  {.key = "options", .type = ConfigType::Object,
   .doc = "av_dict of string opts for input open"},
};
// Canonical oports (video first, audio second); either may be disabled
// via config, so the live count is assigned dynamically in the ctor.
// Each port carries a StreamParams header then FrameRef beats, so the
// payload type is mixed (untyped). Clock groups are authoritatively
// reported by oport_clock_group() (video 0, audio 1).
const PortSpec kOports[] = {
  {.name = "video", .doc = "VideoStreamParams header then video FrameRefs",
   .type = nullptr, .clock_group = 0},
  {.name = "audio", .doc = "AudioStreamParams header then audio FrameRefs",
   .type = nullptr, .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "video-file-decoder",
  .doc       = "Source: demuxes + decodes a video file or network URL "
               "and emits decoded video/audio frames (header + FrameRefs) "
               "on independent per-stream clocks.",
  .display_name = "Video File Reader",
  .category  = StageCategory::Video,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
VideoFileDecoderStage::spec() const noexcept
{
  return kSpec;
}

VideoFileDecoderStage::~VideoFileDecoderStage()
{
  if (_pkt) {
    _libs->avcodec().api.packet_free(&_pkt);
  }
  if (_vctx) {
    _libs->avcodec().api.free_context(&_vctx);
  }
  if (_actx) {
    _libs->avcodec().api.free_context(&_actx);
  }
  if (_fctx) {
    _libs->avformat().api.close_input(&_fctx);
  }
}

string
VideoFileDecoderStage::av_err_(int rc) const
{
  char buf[256];
  _libs->avutil().api.strerror(rc, buf, sizeof buf);
  return string(buf);
}

int
VideoFileDecoderStage::pick_stream_(int media_type,
                                    int requested) const noexcept
{
  if (!_fctx) {
    return -1;
  }
  for (unsigned i = 0; i < _fctx->nb_streams; ++i) {
    AVStream* s = _fctx->streams[i];
    if (s->codecpar->codec_type != media_type) {
      continue;
    }
    if (requested < 0) {
      return static_cast<int>(i);
    }
    if (static_cast<int>(i) == requested) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void
VideoFileDecoderStage::open_codec_(int stream_idx, AVCodecContext** out)
{
  AVStream* s = _fctx->streams[stream_idx];
  const AVCodec* codec =
    _libs->avcodec().api.find_decoder(s->codecpar->codec_id);
  if (!codec) {
    session()->error(fmt(
        "VideoFileDecoderStage('{}'): no decoder for codec_id {}",
        this->id(),
        static_cast<int>(s->codecpar->codec_id)));
  }
  AVCodecContext* cctx = _libs->avcodec().api.alloc_context3(codec);
  if (!cctx) {
    session()->error(fmt(
        "VideoFileDecoderStage('{}'): avcodec_alloc_context3 failed",
        this->id()));
  }
  int rc = _libs->avcodec().api.parameters_to_context(cctx,
                                                     s->codecpar);
  if (rc < 0) {
    _libs->avcodec().api.free_context(&cctx);
    session()->error(fmt(
        "VideoFileDecoderStage('{}'): parameters_to_context failed: "
        "{}", this->id(), av_err_(rc)));
  }
  rc = _libs->avcodec().api.open2(cctx, codec, nullptr);
  if (rc < 0) {
    _libs->avcodec().api.free_context(&cctx);
    session()->error(fmt(
        "VideoFileDecoderStage('{}'): avcodec_open2 failed: {}",
        this->id(), av_err_(rc)));
  }
  *out = cctx;
}

void
VideoFileDecoderStage::open_input_()
{
  AVDictionary* opts = nullptr;
  fill_dict_from_options_(_open_options, _libs->avutil(), &opts);
  if (_read_timeout_ms > 0) {
    string val = to_string(static_cast<long long>(_read_timeout_ms)
                            * 1000LL);
    _libs->avutil().api.dict_set(&opts, "stimeout", val.c_str(), 0);
  }

  AVFormatContext* fctx = nullptr;
  // Forced input format support is intentionally minimal here: we
  // ignore _format unless explicitly extended later. Demuxer
  // autodetection covers all the formats the curated symbol set
  // exercises in practice.
  int rc = _libs->avformat().api.open_input(&fctx, _input_url.c_str(),
                                            nullptr, &opts);
  _libs->avutil().api.dict_free(&opts);
  if (rc < 0) {
    session()->error(fmt(
        "VideoFileDecoderStage('{}'): avformat_open_input('{}') "
        "failed: {}", this->id(), _input_url, av_err_(rc)));
  }
  _fctx = fctx;

  rc = _libs->avformat().api.find_stream_info(_fctx, nullptr);
  if (rc < 0) {
    session()->error(fmt(
        "VideoFileDecoderStage('{}'): find_stream_info failed: {}",
        this->id(), av_err_(rc)));
  }

  if (_video_port >= 0) {
    _v_stream_idx = pick_stream_(AVMEDIA_TYPE_VIDEO,
                                 _video_stream_index);
    if (_v_stream_idx < 0) {
      session()->warn(fmt(
        "VideoFileDecoderStage('{}'): no video stream in '{}'; "
        "video oport will be closed immediately",
        this->id(), _input_url));
    } else {
      open_codec_(_v_stream_idx, &_vctx);
    }
  }
  if (_audio_port >= 0) {
    _a_stream_idx = pick_stream_(AVMEDIA_TYPE_AUDIO,
                                 _audio_stream_index);
    if (_a_stream_idx < 0) {
      session()->warn(fmt(
        "VideoFileDecoderStage('{}'): no audio stream in '{}'; "
        "audio oport will be closed immediately",
        this->id(), _input_url));
    } else {
      open_codec_(_a_stream_idx, &_actx);
    }
  }

  _pkt = _libs->avcodec().api.packet_alloc();
  if (!_pkt) {
    session()->error(fmt(
        "VideoFileDecoderStage('{}'): av_packet_alloc failed",
        this->id()));
  }
}

VideoStreamParams
VideoFileDecoderStage::make_video_params_() const noexcept
{
  VideoStreamParams p;
  if (_v_stream_idx < 0) {
    return p;
  }
  AVStream* s = _fctx->streams[_v_stream_idx];
  p.width      = s->codecpar->width;
  p.height     = s->codecpar->height;
  p.pix_fmt    = s->codecpar->format;
  p.time_base  = s->time_base;
  p.frame_rate = s->avg_frame_rate;
  return p;
}

AudioStreamParams
VideoFileDecoderStage::make_audio_params_() const noexcept
{
  AudioStreamParams p;
  if (_a_stream_idx < 0) {
    return p;
  }
  AVStream* s = _fctx->streams[_a_stream_idx];
  p.sample_rate = s->codecpar->sample_rate;
  p.sample_fmt  = s->codecpar->format;
  // For AV_CHANNEL_ORDER_NATIVE the layout is value-typed (a u64
  // mask). For CUSTOM the u.map pointer aliases _fctx-owned memory;
  // safe as long as the stage outlives the consumer's read of this
  // header, which is true for every sensible pipeline shape.
  p.ch_layout   = s->codecpar->ch_layout;
  p.time_base   = s->time_base;
  return p;
}

Job
VideoFileDecoderStage::drain_codec_(RuntimeContext& ctx,
                                    AVCodecContext* cctx,
                                    unsigned        port,
                                    AVPacket*       pkt)
{
  int rc = _libs->avcodec().api.send_packet(cctx, pkt);
  if (rc < 0 && rc != AVERROR(EAGAIN) && rc != AVERROR_EOF) {
    session()->warn(fmt(
      "decoder('{}'): send_packet on port {}: {}",
      this->id(), port, av_err_(rc)));
    co_return;
  }
  while (true) {
    AVFrame* f = _libs->avutil().api.frame_alloc();
    if (!f) {
      session()->warn(fmt(
        "decoder('{}'): frame_alloc returned null", this->id()));
      co_return;
    }
    rc = _libs->avcodec().api.receive_frame(cctx, f);
    if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
      _libs->avutil().api.frame_free(&f);
      break;
    }
    if (rc < 0) {
      _libs->avutil().api.frame_free(&f);
      session()->warn(fmt(
        "decoder('{}'): receive_frame on port {}: {}",
        this->id(), port, av_err_(rc)));
      break;
    }
    auto sp = FrameRef(f, [api = &_libs->avutil().api](AVFrame* x) {
      api->frame_free(&x);
    });
    co_await ctx.write(port,
        make_payload<FrameRefPayload>(std::move(sp)));
  }
}

Job
VideoFileDecoderStage::initialize(RuntimeContext& ctx)
{
  try {
    open_input_();
  } catch (const exception& e) {
    session()->warn(fmt(
      "decoder('{}'): {}; stopping", this->id(), e.what()));
    ctx.signal_done();
    co_return;
  }

  // Emit StreamParams headers on each enabled port that actually
  // has a matching stream in the input.
  if (_video_port >= 0 && _v_stream_idx >= 0) {
    co_await ctx.write(static_cast<unsigned>(_video_port),
      make_payload<VideoStreamParamsPayload>(make_video_params_()));
  }
  if (_audio_port >= 0 && _a_stream_idx >= 0) {
    co_await ctx.write(static_cast<unsigned>(_audio_port),
      make_payload<AudioStreamParamsPayload>(make_audio_params_()));
  }
}

Job
VideoFileDecoderStage::process(RuntimeContext& ctx)
{
  if (!_fctx) {
    // initialize() failed earlier; nothing to do.
    ctx.signal_done();
    co_return;
  }
  if (_eof) {
    // No more packets to read; let the driver fall through to
    // drain() which flushes the decoders.
    ctx.signal_done();
    co_return;
  }

  int rc = _libs->avformat().api.read_frame(_fctx, _pkt);
  if (rc == AVERROR_EOF) {
    _eof = true;
    co_return;
  }
  if (rc < 0) {
    session()->warn(fmt(
      "decoder('{}'): read_frame: {}", this->id(), av_err_(rc)));
    ctx.signal_done();
    co_return;
  }

  if (_pkt->stream_index == _v_stream_idx
      && _video_port >= 0
      && _vctx)
  {
    co_await drain_codec_(ctx, _vctx,
                          static_cast<unsigned>(_video_port), _pkt);
  } else if (_pkt->stream_index == _a_stream_idx
             && _audio_port >= 0
             && _actx)
  {
    co_await drain_codec_(ctx, _actx,
                          static_cast<unsigned>(_audio_port), _pkt);
  }
  _libs->avcodec().api.packet_unref(_pkt);
}

Job
VideoFileDecoderStage::drain(RuntimeContext& ctx)
{
  // Flush each decoder once (NULL packet) so any frames buffered
  // inside the codec are emitted downstream before the driver
  // closes our output edges.
  if (_vctx && _video_port >= 0) {
    co_await drain_codec_(ctx, _vctx,
                          static_cast<unsigned>(_video_port),
                          nullptr);
  }
  if (_actx && _audio_port >= 0) {
    co_await drain_codec_(ctx, _actx,
                          static_cast<unsigned>(_audio_port),
                          nullptr);
  }
}

VPIPE_REGISTER_STAGE(VideoFileDecoderStage)
VPIPE_REGISTER_SPEC(VideoFileDecoderStage, kSpec)

}
