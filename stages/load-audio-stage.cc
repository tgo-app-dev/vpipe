#include "stages/load-audio-stage.h"

#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/error.h>
}

namespace vpipe {

namespace {

// Reads an object<string,string> from FlexData into an AVDictionary.
// The same shape load-video uses; kept local rather than shared because
// the two are three lines each and a header for them would be the only
// thing they had in common.
void
fill_dict_from_options_(const FlexData& opts, const LibAvUtil& avu,
                        AVDictionary** out)
{
  if (!opts.is_object()) { return; }
  for (auto entry : opts.as_object()) {
    if (!entry.second.is_string()) { continue; }
    const std::string key(entry.first);
    const std::string val(entry.second.get_string());
    avu.api.dict_set(out, key.c_str(), val.c_str(), 0);
  }
}

constexpr ConfigKey kAttrs[] = {
  {.key = "input_url", .type = ConfigType::String, .required = true,
   .doc = "audio file path or network URL. Any container and codec the "
          "installed FFmpeg can demux -- mp3, m4a/aac, flac, wav, ogg, opus, "
          "or the audio track of a video file",
   .is_path = true, .path_filter = "audio"},
  {.key = "stream_index", .type = ConfigType::Int, .required = false,
   .doc = "ABSOLUTE stream index to read (as ffprobe numbers them, video "
          "included), not the Nth audio stream; -1 (default) takes the first "
          "audio stream in the file",
   .def_int = -1},
  {.key = "start_s", .type = ConfigType::Real, .required = false,
   .doc = "seek to this media time (seconds from the start of the file) "
          "before reading the first packet; 0 (default) starts at the "
          "beginning. PACKET-accurate: the first sample lands within one "
          "packet of what was asked for",
   .def_real = 0.0},
  {.key = "duration_s", .type = ConfigType::Real, .required = false,
   .doc = "stop this many seconds past start_s; 0 (default) reads to the "
          "end of the file. The packet straddling the end is emitted "
          "whole, so the window may overshoot by one packet",
   .def_real = 0.0},
  {.key = "read_timeout_ms", .type = ConfigType::Int, .required = false,
   .doc = "network open/read timeout in milliseconds; 0 = none",
   .def_int = 0},
  {.key = "options", .type = ConfigType::Object, .required = false,
   .doc = "extra av_dict options passed to the demuxer open"},
};

const PortSpec kOports[] = {
  {.name = "audio",
   .doc = "ONE ENCODED PACKET per beat as an EncodedSegment with "
          "kind=Audio, carrying the stream's codec_id / sample_rate / "
          "channels / extradata. Wire it to audio-to-pcm, which owns the "
          "decode and the resample",
   .type = &typeid(EncodedSegmentPayload), .clock_group = 0},
};

const StageSpec kSpec = {
  .type_name = "load-audio",
  .doc       = "Source: demuxes an audio file or URL and emits its encoded "
               "packets, one per beat -- the file-based sibling of "
               "rtsp-capture's audio oport. Feeds audio-to-pcm, which "
               "decodes and resamples.",
  .display_name = "Load Audio",
  .category  = StageCategory::Audio,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

LoadAudioStage::LoadAudioStage(const SessionContextIntf* s,
                               std::string               id,
                               std::vector<InEdge>       iports,
                               FlexData                  config)
  : TypedStage<LoadAudioStage>(s, std::move(id), std::move(iports),
                               std::move(config))
  , _libs(s->services()->ffmpeg_libraries())
{
  _input_url       = attr_path("input_url", false);
  _stream_index    = static_cast<int>(attr_int("stream_index"));
  _read_timeout_ms = static_cast<int>(attr_int("read_timeout_ms"));
  _start_s         = attr_real("start_s");
  _duration_s      = attr_real("duration_s");
  _open_options    = attr("options");

  if (_start_s < 0.0) {
    fail_config(fmt("LoadAudioStage('{}'): start_s must be >= 0, got {}",
                    this->id(), _start_s));
  }
  if (_duration_s < 0.0) {
    fail_config(fmt("LoadAudioStage('{}'): duration_s must be >= 0 (0 = to "
                    "the end of the file), got {}", this->id(), _duration_s));
  }
  _start_us = static_cast<std::int64_t>(_start_s * 1e6);
  _has_stop = _duration_s > 0.0;
  _stop_us  = _has_stop
                  ? _start_us + static_cast<std::int64_t>(_duration_s * 1e6)
                  : 0;

  // Deferred validation: the ctor never throws, so a graph can be built
  // and edited before the file exists.
  if (_input_url.empty()) {
    fail_config(fmt("LoadAudioStage('{}'): config.input_url is required",
                    this->id()));
  }
  allocate_oports(spec().oports.size());
}

LoadAudioStage::~LoadAudioStage()
{
  if (_pkt != nullptr && _libs != nullptr) {
    _libs->avcodec().api.packet_free(&_pkt);
  }
  if (_fctx != nullptr && _libs != nullptr) {
    _libs->avformat().api.close_input(&_fctx);
  }
}

const StageSpec&
LoadAudioStage::spec() const noexcept
{
  return kSpec;
}

// The configured window, for the open log. Empty when the whole file is
// being read, so the common case says nothing extra.
std::string
LoadAudioStage::window_doc_() const
{
  if (_start_us <= 0 && !_has_stop) { return std::string(); }
  if (!_has_stop) {
    return fmt(", from {:.3f}s to the end", _start_s)();
  }
  return fmt(", window [{:.3f}s, {:.3f}s)", _start_s,
             _start_s + _duration_s)();
}

std::string
LoadAudioStage::av_err_(int rc) const
{
  char buf[256] = {0};
  _libs->avutil().api.strerror(rc, buf, sizeof buf);
  return std::string(buf);
}

void
LoadAudioStage::open_input_()
{
  AVDictionary* opts = nullptr;
  fill_dict_from_options_(_open_options, _libs->avutil(), &opts);
  if (_read_timeout_ms > 0) {
    const std::string val =
        std::to_string(static_cast<long long>(_read_timeout_ms) * 1000LL);
    _libs->avutil().api.dict_set(&opts, "stimeout", val.c_str(), 0);
  }
  AVFormatContext* fctx = nullptr;
  const int rc = _libs->avformat().api.open_input(&fctx, _input_url.c_str(),
                                                  nullptr, &opts);
  _libs->avutil().api.dict_free(&opts);
  if (rc < 0) {
    session()->error(fmt(
        "LoadAudioStage('{}'): could not open '{}': {}", this->id(),
        _input_url, av_err_(rc)));
    return;
  }
  _fctx = fctx;
  const int rc2 = _libs->avformat().api.find_stream_info(_fctx, nullptr);
  if (rc2 < 0) {
    session()->error(fmt(
        "LoadAudioStage('{}'): find_stream_info failed: {}", this->id(),
        av_err_(rc2)));
    return;
  }

  for (unsigned i = 0; i < _fctx->nb_streams; ++i) {
    AVStream* st = _fctx->streams[i];
    if (st->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) { continue; }
    if (_stream_index >= 0 && static_cast<int>(i) != _stream_index) {
      continue;
    }
    _idx = static_cast<int>(i);
    _tb  = st->time_base;
    _codec_id    = static_cast<unsigned>(st->codecpar->codec_id);
    _sample_rate = static_cast<unsigned>(st->codecpar->sample_rate);
    _channels    =
        static_cast<unsigned>(st->codecpar->ch_layout.nb_channels);
    if (st->codecpar->extradata != nullptr &&
        st->codecpar->extradata_size > 0) {
      _extradata.assign(
          st->codecpar->extradata,
          st->codecpar->extradata + st->codecpar->extradata_size);
    }
    break;
  }
  if (_idx < 0) {
    // A file with no audio is a configuration mistake, not a runtime
    // one: every consumer downstream exists to process audio, so the
    // graph would sit waiting for beats that cannot come.
    session()->error(fmt(
        "LoadAudioStage('{}'): '{}' has no audio stream{}; inert",
        this->id(), _input_url,
        _stream_index >= 0
            ? fmt(" at index {}", _stream_index)()
            : std::string()));
    return;
  }
  // Seek AFTER the stream is chosen: the target is expressed in that
  // stream's time base, and AVSEEK_FLAG_BACKWARD asks for the packet at
  // or before it so nothing between the seek point and `start_s` is
  // lost. process() drops what precedes the window; here we only get
  // near it.
  if (_start_us > 0) {
    const AVRational us = {1, 1000000};
    const std::int64_t ts =
        _libs->avutil().api.rescale_q(_start_us, us, _tb);
    const int rc = _libs->avformat().api.seek_frame(_fctx, _idx, ts,
                                                    AVSEEK_FLAG_BACKWARD);
    if (rc < 0) {
      // Not fatal. A stream with no index (a raw bitstream, some live
      // URLs) refuses the seek, and reading from the start is the
      // honest fallback -- process() still drops everything before
      // `start_s`, so the WINDOW is right and only the work to reach it
      // is wasted.
      session()->warn(fmt(
          "LoadAudioStage('{}'): seek to {:.3f}s failed ({}); reading from "
          "the start and skipping forward instead", this->id(), _start_s,
          av_err_(rc)));
    }
  }
  _last_us = _start_us;

  session()->info(fmt(
      "LoadAudioStage('{}'): '{}' stream {} -- codec_id {}, {} Hz, {} "
      "channel(s), {} bytes of extradata{}", this->id(), _input_url, _idx,
      _codec_id, _sample_rate, _channels, _extradata.size(),
      window_doc_()));
}

void
LoadAudioStage::reset_run_state()
{
  if (_pkt != nullptr && _libs != nullptr) {
    _libs->avcodec().api.packet_free(&_pkt);
  }
  if (_fctx != nullptr && _libs != nullptr) {
    _libs->avformat().api.close_input(&_fctx);
  }
  _pkt     = nullptr;
  _fctx    = nullptr;
  _idx     = -1;
  _eof     = false;
  _packets = 0;
  // Back to the window's start, not to 0: open_input_() sets it again
  // after the seek, and a launch that fails before then should still
  // stamp a packet with no pts inside the window it was asked for.
  _last_us = _start_us;
}

Job
LoadAudioStage::initialize(RuntimeContext& ctx)
{
  open_input_();
  if (_fctx == nullptr || _idx < 0) {
    ctx.signal_done();
    co_return;
  }
  _pkt = _libs->avcodec().api.packet_alloc();
  if (_pkt == nullptr) {
    session()->error(fmt("LoadAudioStage('{}'): packet_alloc failed",
                         this->id()));
    ctx.signal_done();
  }
  co_return;
}

Job
LoadAudioStage::process(RuntimeContext& ctx)
{
  if (_fctx == nullptr || _idx < 0 || _pkt == nullptr || _eof) {
    ctx.signal_done();
    co_return;
  }

  const int rc = _libs->avformat().api.read_frame(_fctx, _pkt);
  if (rc == AVERROR_EOF) {
    _eof = true;
    session()->info(fmt(
        "LoadAudioStage('{}'): end of '{}' after {} packet(s)", this->id(),
        _input_url, _packets));
    ctx.signal_done();
    co_return;
  }
  if (rc < 0) {
    session()->warn(fmt("LoadAudioStage('{}'): read_frame: {}", this->id(),
                        av_err_(rc)));
    ctx.signal_done();
    co_return;
  }
  if (_pkt->stream_index != _idx || _pkt->size <= 0) {
    // Another stream's packet (a cover image, a second language, the
    // video track of an mp4). Dropped, and the driver calls again.
    _libs->avcodec().api.packet_unref(_pkt);
    co_return;
  }

  // MEDIA time, from the start of the file -- see the header. A packet
  // with no pts (some raw formats stamp none) repeats the previous
  // one's, which keeps the stream monotonic; leaving it at zero would
  // make time jump back to the start of the file mid-clip.
  //
  // Computed BEFORE the segment is built so a packet outside the window
  // costs no copy of its bytes.
  const AVRational us = {1, 1000000};
  std::int64_t t = _last_us;
  if (_pkt->pts != AV_NOPTS_VALUE) {
    t = _libs->avutil().api.rescale_q(_pkt->pts, _tb, us);
  }
  std::int64_t d = 0;
  if (_pkt->duration > 0) {
    d = _libs->avutil().api.rescale_q(_pkt->duration, _tb, us);
  }
  _last_us = t + d;

  // Past the window: done. Tested on the packet's START, so the packet
  // straddling the end is emitted whole rather than truncated -- this
  // stage does not decode and so cannot cut inside one.
  if (_has_stop && t >= _stop_us) {
    _libs->avcodec().api.packet_unref(_pkt);
    _eof = true;
    session()->info(fmt(
        "LoadAudioStage('{}'): reached {:.3f}s of '{}' after {} packet(s)",
        this->id(), _start_s + _duration_s, _input_url, _packets));
    ctx.signal_done();
    co_return;
  }
  // Before it: the seek landed at or before `start_s`, so drop what
  // ends at or before the window opens. Safe here and NOT in
  // load-video, whose video packets depend on the keyframe ahead of
  // them (see that header).
  if (t + d <= _start_us) {
    _libs->avcodec().api.packet_unref(_pkt);
    co_return;
  }

  auto seg = std::make_unique<EncodedSegmentPayload>();
  seg->kind        = EncodedSegment::Kind::Audio;
  seg->path        = _input_url;
  seg->codec_id    = _codec_id;
  seg->sample_rate = _sample_rate;
  seg->channels    = _channels;
  seg->extradata   = _extradata;
  seg->data.assign(_pkt->data, _pkt->data + _pkt->size);

  seg->duration_us = d;
  seg->start_utc =
      std::chrono::system_clock::time_point(std::chrono::microseconds(t));
  seg->end_utc =
      std::chrono::system_clock::time_point(std::chrono::microseconds(t + d));

  _libs->avcodec().api.packet_unref(_pkt);
  ++_packets;
  co_await ctx.write(0, std::move(seg));
}

VPIPE_REGISTER_STAGE(LoadAudioStage)
VPIPE_REGISTER_SPEC(LoadAudioStage, kSpec)

}  // namespace vpipe
