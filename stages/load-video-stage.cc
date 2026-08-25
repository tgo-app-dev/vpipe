#include "stages/load-video-stage.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
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

LoadVideoStage::LoadVideoStage
  (const SessionContextIntf* s,
   string                    id,
   vector<InEdge>            iports,
   FlexData                  config)
  : TypedStage<LoadVideoStage>(s, std::move(id),
                                      std::move(iports),
                                      std::move(config))
  , _libs(s->services()->ffmpeg_libraries())
{
  // Stage's _config now holds the moved-in FlexData. Read everything
  // we need up-front; throw via session()->error if `input_url` is
  // missing because there's no graceful default.
  // Attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _input_url          = attr_path("input_url", false);
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
      "LoadVideoStage('{}'): config.input_url is required",
      this->id()));
  }
  if (!_enable_video && !_enable_audio) {
    fail_config(fmt(
      "LoadVideoStage('{}'): at least one of enable_video / "
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
   .doc = "file path or network URL (rtsp/http/...)",
   .is_path = true, .path_filter = "video"},
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
// The payload is rtsp-capture's, so a file and a camera are the same
// input to everything downstream. Clock groups are authoritatively
// reported by oport_clock_group() (video 0, audio 1) -- unlike the
// camera's single capture clock, a file's two streams advance
// independently.
const PortSpec kOports[] = {
  {.name = "video", .doc = "ONE EncodedSegment per video packet, carrying "
                           "the stream's codec_id / width / height / "
                           "fps_num / fps_den / extradata. Wire it to "
                           "video-to-rgb, which owns the decode",
   .type = &typeid(EncodedSegmentPayload),
   .tags = "video-encoder-segments", .clock_group = 0},
  {.name = "audio", .doc = "ONE EncodedSegment per audio packet, carrying "
                           "the stream's codec_id / sample_rate / channels "
                           "/ extradata. Wire it to audio-to-pcm, which "
                           "owns the decode and the resample",
   .type = &typeid(EncodedSegmentPayload),
   .tags = "audio-encoder-segments", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "load-video",
  .doc       = "Source: demuxes a video file or network URL and emits its "
               "encoded video/audio packets on independent per-stream "
               "clocks -- the file-based sibling of rtsp-capture. Feeds "
               "video-to-rgb / audio-to-pcm, which own the decode.",
  .display_name = "Load Video",
  .category  = StageCategory::Visual,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
LoadVideoStage::spec() const noexcept
{
  return kSpec;
}

LoadVideoStage::~LoadVideoStage()
{
  if (_pkt) {
    _libs->avcodec().api.packet_free(&_pkt);
  }
  if (_fctx) {
    _libs->avformat().api.close_input(&_fctx);
  }
}

string
LoadVideoStage::av_err_(int rc) const
{
  char buf[256];
  _libs->avutil().api.strerror(rc, buf, sizeof buf);
  return string(buf);
}

int
LoadVideoStage::pick_stream_(int media_type,
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
LoadVideoStage::cache_stream_(int stream_idx, bool video)
{
  if (_fctx == nullptr || stream_idx < 0) { return; }
  AVStream* st = _fctx->streams[stream_idx];
  StreamMeta& m = video ? _vmeta : _ameta;
  m.codec_id  = (unsigned)st->codecpar->codec_id;
  m.time_base = st->time_base;
  if (st->codecpar->extradata != nullptr &&
      st->codecpar->extradata_size > 0) {
    m.extradata.assign(
        st->codecpar->extradata,
        st->codecpar->extradata + st->codecpar->extradata_size);
  }
  if (video) {
    m.width  = (unsigned)st->codecpar->width;
    m.height = (unsigned)st->codecpar->height;
    // avg_frame_rate first, r_frame_rate as the fallback, 0/0 when the
    // container advertises neither -- the same three-way answer
    // rtsp-capture gives, so video-to-rgb's sideband means one thing
    // whichever source it is reading.
    AVRational fr = st->avg_frame_rate;
    if (fr.num <= 0 || fr.den <= 0) { fr = st->r_frame_rate; }
    if (fr.num > 0 && fr.den > 0) {
      m.fps_num = (unsigned)fr.num;
      m.fps_den = (unsigned)fr.den;
    }
  } else {
    m.sample_rate = (unsigned)st->codecpar->sample_rate;
    m.channels    = (unsigned)st->codecpar->ch_layout.nb_channels;
  }
}

std::unique_ptr<EncodedSegmentPayload>
LoadVideoStage::segment_(bool video)
{
  StreamMeta& m = video ? _vmeta : _ameta;
  auto seg = std::make_unique<EncodedSegmentPayload>();
  seg->kind = video ? EncodedSegment::Kind::Video
                    : EncodedSegment::Kind::Audio;
  seg->path      = _input_url;
  seg->codec_id  = m.codec_id;
  seg->extradata = m.extradata;
  if (video) {
    seg->width   = m.width;
    seg->height  = m.height;
    seg->fps_num = m.fps_num;
    seg->fps_den = m.fps_den;
  } else {
    seg->sample_rate = m.sample_rate;
    seg->channels    = m.channels;
  }
  seg->data.assign(_pkt->data, _pkt->data + _pkt->size);

  const AVRational us = {1, 1000000};
  std::int64_t t = m.last_us;
  if (_pkt->pts != AV_NOPTS_VALUE) {
    t = _libs->avutil().api.rescale_q(_pkt->pts, m.time_base, us);
  }
  std::int64_t d = 0;
  if (_pkt->duration > 0) {
    d = _libs->avutil().api.rescale_q(_pkt->duration, m.time_base, us);
  }
  m.last_us = t + d;
  seg->duration_us = d;
  seg->start_utc =
      std::chrono::system_clock::time_point(std::chrono::microseconds(t));
  seg->end_utc =
      std::chrono::system_clock::time_point(
          std::chrono::microseconds(t + d));
  return seg;
}

void
LoadVideoStage::open_input_()
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
        "LoadVideoStage('{}'): avformat_open_input('{}') "
        "failed: {}", this->id(), _input_url, av_err_(rc)));
  }
  _fctx = fctx;

  rc = _libs->avformat().api.find_stream_info(_fctx, nullptr);
  if (rc < 0) {
    session()->error(fmt(
        "LoadVideoStage('{}'): find_stream_info failed: {}",
        this->id(), av_err_(rc)));
  }

  if (_video_port >= 0) {
    _v_stream_idx = pick_stream_(AVMEDIA_TYPE_VIDEO,
                                 _video_stream_index);
    if (_v_stream_idx < 0) {
      session()->warn(fmt(
        "LoadVideoStage('{}'): no video stream in '{}'; "
        "video oport will be closed immediately",
        this->id(), _input_url));
    } else {
      cache_stream_(_v_stream_idx, /*video=*/true);
    }
  }
  if (_audio_port >= 0) {
    _a_stream_idx = pick_stream_(AVMEDIA_TYPE_AUDIO,
                                 _audio_stream_index);
    if (_a_stream_idx < 0) {
      session()->warn(fmt(
        "LoadVideoStage('{}'): no audio stream in '{}'; "
        "audio oport will be closed immediately",
        this->id(), _input_url));
    } else {
      cache_stream_(_a_stream_idx, /*video=*/false);
    }
  }

  _pkt = _libs->avcodec().api.packet_alloc();
  if (!_pkt) {
    session()->error(fmt(
        "LoadVideoStage('{}'): av_packet_alloc failed",
        this->id()));
  }
}

Job
LoadVideoStage::initialize(RuntimeContext& ctx)
{
  try {
    open_input_();
  } catch (const exception& e) {
    session()->warn(fmt(
      "decoder('{}'): {}; stopping", this->id(), e.what()));
    ctx.signal_done();
    co_return;
  }

  // No header beat. Every field a StreamParams header used to carry --
  // geometry, cadence, rate, channels -- rides on EVERY segment, which
  // is what lets a consumer join a stream late and what makes this
  // interchangeable with rtsp-capture. A header would also have to be
  // distinguished by try_get<T> from the beats after it, and a consumer
  // that forgot would read it as a packet.
  co_return;
}

Job
LoadVideoStage::process(RuntimeContext& ctx)
{
  if (!_fctx || !_pkt) {
    // initialize() failed earlier; nothing to do.
    ctx.signal_done();
    co_return;
  }
  if (_eof) { ctx.signal_done(); co_return; }

  const int rc = _libs->avformat().api.read_frame(_fctx, _pkt);
  if (rc == AVERROR_EOF) {
    _eof = true;
    session()->info(fmt(
        "LoadVideoStage('{}'): end of '{}' after {} video + {} audio "
        "packet(s)", this->id(), _input_url, _v_packets, _a_packets));
    ctx.signal_done();
    co_return;
  }
  if (rc < 0) {
    session()->warn(fmt(
      "LoadVideoStage('{}'): read_frame: {}", this->id(), av_err_(rc)));
    ctx.signal_done();
    co_return;
  }

  // Which port, if any. A packet from a stream nobody asked for -- a
  // second language, a cover image, the video track when enable_video
  // is off -- is dropped, and the driver calls again.
  int port = -1;
  bool video = false;
  if (_pkt->stream_index == _v_stream_idx && _video_port >= 0) {
    port = _video_port;
    video = true;
  } else if (_pkt->stream_index == _a_stream_idx && _audio_port >= 0) {
    port = _audio_port;
  }
  if (port < 0 || _pkt->size <= 0) {
    _libs->avcodec().api.packet_unref(_pkt);
    co_return;
  }
  auto seg = segment_(video);
  if (video) { ++_v_packets; } else { ++_a_packets; }
  _libs->avcodec().api.packet_unref(_pkt);
  // The payload is bound to its own local before the suspend: a
  // temporary built inside the co_await argument has its lifetime
  // tangled with the frame layout across the resume.
  co_await ctx.write(static_cast<unsigned>(port), std::move(seg));
  co_return;
}

VPIPE_REGISTER_STAGE(LoadVideoStage)
VPIPE_REGISTER_SPEC(LoadVideoStage, kSpec)

}
