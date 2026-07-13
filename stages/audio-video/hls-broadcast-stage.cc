#include "stages/audio-video/hls-broadcast-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/host-net.h"
#include "common/thread-pool.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <chrono>
#include <cmath>
#include <coroutine>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
#include <libavutil/samplefmt.h>
}

using namespace std;

namespace vpipe {

namespace {

// Re-schedules the awaiting coroutine on the pool after `delay` instead
// of blocking the worker with a sleep (same pattern as ChronoStage). The
// audio-only silence pump uses it to nap ~one frame between silence
// frames without pinning a pipeline worker for the duration of a stall.
struct TimerAwaiter {
  ThreadPool*                          pool;
  std::chrono::steady_clock::duration  delay;

  bool
  await_ready() const noexcept
  {
    return pool == nullptr ||
           delay <= std::chrono::steady_clock::duration::zero();
  }

  void
  await_suspend(std::coroutine_handle<> h) const noexcept
  {
    // Hoist members into locals before publishing the handle (the timer
    // can resume it on another worker the instant it fires).
    ThreadPool* p = pool;
    auto        d = delay;
    p->schedule_after(d, h);
  }

  void await_resume() const noexcept {}
};

// GBRP plane indices: G=0, B=1, R=2. RGB channel c maps here.
const int kGbrpPlaneForChannel[3] = { 2, 0, 1 };

uint8_t
clamp_byte_(float v)
{
  if (v <= 0.0f) { return 0; }
  if (v >= 255.0f) { return 255; }
  return static_cast<uint8_t>(lroundf(v));
}

// Extract the basename of a path-like url ("a/b/c.m3u8" -> "c.m3u8";
// "stream0.ts" -> "stream0.ts"). The HLS muxer may pass either,
// depending on whether the playlist was opened with a directory
// prefix. We strip directories so the in-memory path is stable.
string
basename_(string_view url)
{
  auto slash = url.find_last_of('/');
  if (slash == string_view::npos) {
    return string(url);
  }
  return string(url.substr(slash + 1));
}

// Strip a trailing ".tmp" suffix if present (the HLS muxer uses it
// for atomic playlist rewrites when `hls_flags +temp_file` is set).
// Treats "stream.m3u8.tmp" as "stream.m3u8" so the published HTTP
// path is the user-visible name in both modes.
string
strip_tmp_suffix_(string s)
{
  constexpr string_view kSfx = ".tmp";
  if (s.size() >= kSfx.size()
      && s.compare(s.size() - kSfx.size(), kSfx.size(), kSfx) == 0) {
    s.resize(s.size() - kSfx.size());
  }
  return s;
}

bool
ends_with_(string_view s, string_view sfx)
{
  return s.size() >= sfx.size()
      && s.compare(s.size() - sfx.size(), sfx.size(), sfx) == 0;
}

// Remove a single flag token from a `+`-joined ffmpeg hls_flags string
// (e.g. "delete_segments+omit_endlist" minus "delete_segments" ->
// "omit_endlist"). Returns the rejoined string; `removed` reports
// whether the token was present.
string
strip_flag_token_(string_view flags, string_view token, bool* removed)
{
  string out;
  bool   found = false;
  size_t i = 0;
  while (i < flags.size()) {
    size_t plus = flags.find('+', i);
    size_t end  = (plus == string_view::npos) ? flags.size() : plus;
    string_view tok = flags.substr(i, end - i);
    if (tok == token) {
      found = true;
    } else if (!tok.empty()) {
      if (!out.empty()) { out.push_back('+'); }
      out.append(tok);
    }
    if (plus == string_view::npos) { break; }
    i = plus + 1;
  }
  if (removed) { *removed = found; }
  return out;
}

}

// Sink for one AVIOContext opened by the HLS muxer. Buffers writes
// (with seek support so .tmp / header-patching flows work) until
// io_close, at which point the bytes are atomically published into
// the stage's blob store. The buffer is shared via shared_ptr so a
// slow HTTP client reading the previous playlist doesn't block the
// next playlist write.
struct HlsBroadcastStage::InMemSink {
  vector<uint8_t> bytes;
  int64_t         pos = 0;
  string          url;

  static int
  write_packet_(void* opaque, const uint8_t* buf, int buf_size)
  {
    auto* self = static_cast<InMemSink*>(opaque);
    if (buf_size <= 0) { return 0; }
    const size_t end = static_cast<size_t>(self->pos)
                     + static_cast<size_t>(buf_size);
    if (end > self->bytes.size()) {
      self->bytes.resize(end);
    }
    std::memcpy(self->bytes.data() + self->pos, buf, buf_size);
    self->pos += buf_size;
    return buf_size;
  }

  static int64_t
  seek_(void* opaque, int64_t offset, int whence)
  {
    auto* self = static_cast<InMemSink*>(opaque);
    if ((whence & AVSEEK_SIZE) == AVSEEK_SIZE) {
      return static_cast<int64_t>(self->bytes.size());
    }
    int64_t base = 0;
    switch (whence & ~AVSEEK_FORCE) {
      case SEEK_SET: base = 0; break;
      case SEEK_CUR: base = self->pos; break;
      case SEEK_END:
        base = static_cast<int64_t>(self->bytes.size());
        break;
      default:       return -1;
    }
    const int64_t np = base + offset;
    if (np < 0) { return -1; }
    self->pos = np;
    return np;
  }
};

namespace {

// FFmpeg's AVFormatContext.io_open / io_close2 are plain C function
// pointers, so we dispatch via a static thunk that recovers the
// stage from `s->opaque`. Defined once here to avoid duplicating the
// trampoline boilerplate for each callback.
extern "C" int
hls_io_open_thunk_(AVFormatContext* s, AVIOContext** pb,
                   const char* url, int /*flags*/,
                   AVDictionary** /*options*/)
{
  auto* stage = static_cast<HlsBroadcastStage*>(s->opaque);
  return stage->io_open_(pb, url ? string(url) : string());
}

extern "C" int
hls_io_close_thunk_(AVFormatContext* s, AVIOContext* pb)
{
  auto* stage = static_cast<HlsBroadcastStage*>(s->opaque);
  return stage->io_close_(pb);
}

}

HlsBroadcastStage::HlsBroadcastStage(const SessionContextIntf* s,
                                     string                    id,
                                     vector<InEdge>            iports,
                                     FlexData                  config)
  : TypedStage<HlsBroadcastStage>(s, std::move(id), std::move(iports),
                                  std::move(config))
  , _libs(s->ffmpeg_libraries())
{
  // Attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default. Post-read clamps below repair
  // out-of-range overrides.
  _playlist_name     = attr_str("playlist_name");
  _segment_duration  = attr_real("segment_duration");
  _live_start_offset = attr_real("live_start_offset");
  _playlist_max_size = static_cast<int>(attr_int("playlist_max_size"));
  _hls_flags         = attr_str("hls_flags");
  // `delete_segments` is meaningless for this stage: every segment +
  // playlist write is routed through in-memory AVIO callbacks
  // (io_open_/io_close_), so the muxer's on-disk unlink of aged-out
  // .ts files always fails ("failed to delete old segment ...: No such
  // file or directory"). Eviction is handled internally by the bounded
  // FIFO in write_packet_, so strip the flag before it reaches the
  // muxer -- regardless of whether it came from the default or config.
  {
    bool had_delete = false;
    _hls_flags = strip_flag_token_(_hls_flags, "delete_segments",
                                   &had_delete);
    if (had_delete) {
      session()->log_verbose(fmt(
          "hls-broadcast('{}'): ignoring 'delete_segments' hls_flag "
          "(segments are in-memory; eviction is handled internally)",
          this->id()));
    }
  }
  _codec_name        = attr_str("codec");
  _bitrate           = attr_int("bitrate");
  _gop_size          = static_cast<int>(attr_int("gop_size"));
  _preset            = attr_str("preset");
  _tune              = attr_str("tune");
  // fps defaults to 0 ("auto"): the encode cadence is taken from the
  // first frame's sideband (fps_num/fps_den, set by video-to-rgb from
  // the capture stream) when the config doesn't pin it. An explicit
  // fps_num > 0 in config always wins; a lone fps_num with fps_den
  // unset implies /1. Final resolution (including the 30/1 last-resort
  // fallback) happens in resolve_default_fps_ on the first frame.
  _fps_num           = static_cast<int>(attr_int("fps_num"));
  _fps_den           = static_cast<int>(attr_int("fps_den"));
  _fps_from_config   = (_fps_num > 0);
  if (_fps_from_config && _fps_den <= 0) { _fps_den = 1; }
  _input_normalized  = attr_bool("input_normalized");
  _realtime          = attr_bool("realtime");
  _log_input_stats_every =
      static_cast<int>(attr_int("log_input_stats_every"));
  if (_log_input_stats_every < 0) { _log_input_stats_every = 0; }
  _serve_http        = attr_bool("serve_http");
  _bind_address      = attr_str("bind_address");
  _http_port         = static_cast<int>(attr_int("port"));
  _audio_codec_name  = attr_str("audio_codec");
  _audio_bitrate     = attr_int("audio_bitrate");
  _audio_sample_rate = static_cast<int>(attr_int("audio_sample_rate"));
  _audio_channels    = static_cast<int>(attr_int("audio_channels"));
  _audio_buffer_seconds = attr_real("audio_buffer_seconds");
  _prime_silence = attr_bool("prime_silence");
  if (_audio_sample_rate <= 0)     { _audio_sample_rate = 48000; }
  if (_audio_channels    <= 0)     { _audio_channels    = 1; }
  if (_audio_bitrate     < 8'000)  { _audio_bitrate     = 128'000; }
  if (_audio_buffer_seconds < 0.0) { _audio_buffer_seconds = 0.0; }

  // An empty bind_address (the default) means "auto": bind, in order of
  // preference, to (1) the address the web-ui is served on -- so the
  // HLS stream is reachable exactly where the UI is, matching whatever
  // interface the operator chose -- else (2) this machine's LAN address
  // (en0), so other devices on the network can play the stream out of
  // the box, else (3) 0.0.0.0 (all interfaces) as a last resort. An
  // explicit bind_address in config always wins (including "0.0.0.0").
  if (_bind_address.empty()) {
    const string web_ui = session()->web_ui_bind_address();
    if (!web_ui.empty()) {
      _bind_address = web_ui;
    } else {
      const string lan = netx::primary_ipv4();
      _bind_address = lan.empty() ? string("0.0.0.0") : lan;
    }
  }

  // fps is resolved on the first frame (resolve_default_fps_), not
  // clamped here -- 0/0 stays as the "auto" sentinel until then.
  if (_gop_size <= 0)           { _gop_size = 60; }
  if (!(_segment_duration > 0.0)) { _segment_duration = 2.0; }
  if (_live_start_offset < 0.0) { _live_start_offset = 0.0; }
  if (_playlist_max_size < 0)   { _playlist_max_size = 0; }
  if (_bitrate < 64'000)        { _bitrate = 64'000; }
  if (_http_port < 0 || _http_port > 65535) { _http_port = 8080; }

  allocate_oports(spec().oports.size());   // sink: 0 oports

  const string url_host =
      (_bind_address == "0.0.0.0" || _bind_address.empty())
          ? "localhost" : _bind_address;
  _playlist_url =
      fmt("http://{}:{}/{}", url_host, _http_port, _playlist_name)();

  session()->info(fmt(
      "hls-broadcast('{}'): in-memory HLS, playlist={}, "
      "codec={}, bitrate={}, gop={}, segment={}s, "
      "live_start_offset={}s, serve_http={}, url={}",
      this->id(), _playlist_name, _codec_name, _bitrate,
      _gop_size, _segment_duration, _live_start_offset, _serve_http,
      _serve_http ? _playlist_url : string("(disabled)")));
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "playlist_name", .type = ConfigType::String,
   .doc = "HLS .m3u8 playlist file name", .def_str = "stream.m3u8"},
  {.key = "segment_duration", .type = ConfigType::Real,
   .doc = "hls_time seconds (may be fractional)", .def_real = 2.0},
  {.key = "live_start_offset", .type = ConfigType::Real,
   .doc = "EXT-X-START offset seconds; 0 = off", .def_real = 0.0},
  {.key = "playlist_max_size", .type = ConfigType::Int,
   .doc = "hls_list_size; 0 = unbounded", .def_int = 5},
  {.key = "hls_flags", .type = ConfigType::String,
   .doc = "muxer hls_flags string ('delete_segments' is ignored -- "
          "segments are in-memory and evicted internally)",
   .def_str = "omit_endlist"},
  {.key = "codec", .type = ConfigType::String,
   .doc = "video encoder name", .def_str = "h264_videotoolbox"},
  {.key = "bitrate", .type = ConfigType::Int,
   .doc = "target bps, min 64000", .def_int = 2'000'000},
  {.key = "gop_size", .type = ConfigType::Int,
   .doc = "keyframe interval in frames", .def_int = 60},
  {.key = "preset", .type = ConfigType::String,
   .doc = "libx264 preset", .def_str = "veryfast"},
  {.key = "tune", .type = ConfigType::String,
   .doc = "libx264 tune", .def_str = "zerolatency"},
  {.key = "fps_num", .type = ConfigType::Int,
   .doc = "frame-rate numerator; 0 = auto (adopt the input sideband's "
          "fps, else 30)", .def_int = 0},
  {.key = "fps_den", .type = ConfigType::Int,
   .doc = "frame-rate denominator; 0 = auto (paired with fps_num)",
   .def_int = 0},
  {.key = "input_normalized", .type = ConfigType::Bool,
   .doc = "F32 input in [0,1] vs [0,255]", .def_bool = true},
  {.key = "realtime", .type = ConfigType::Bool,
   .doc = "PTS tracks wall-clock + forced keyframes", .def_bool = true},
  {.key = "log_input_stats_every", .type = ConfigType::Int,
   .doc = "log min/mean/max every Nth frame; 0 = off", .def_int = 0},
  {.key = "serve_http", .type = ConfigType::Bool,
   .doc = "run in-process static HTTP server", .def_bool = true},
  {.key = "bind_address", .type = ConfigType::String,
   .doc = "HTTP server bind address; empty = auto (web-ui address if "
          "this session is served by the web-ui, else en0's LAN IP, "
          "else 0.0.0.0)",
   .def_str = ""},
  {.key = "port", .type = ConfigType::Int,
   .doc = "HTTP server port, [0,65535]", .def_int = 8080},
  {.key = "audio_codec", .type = ConfigType::String,
   .doc = "audio encoder name for the optional audio stream",
   .def_str = "aac"},
  {.key = "audio_bitrate", .type = ConfigType::Int,
   .doc = "audio target bps", .def_int = 128000},
  {.key = "audio_sample_rate", .type = ConfigType::Int,
   .doc = "audio encoder output rate; 0 = auto (48000)", .def_int = 0},
  {.key = "audio_channels", .type = ConfigType::Int,
   .doc = "audio encoder output channel count", .def_int = 1},
  {.key = "audio_buffer_seconds", .type = ConfigType::Real,
   .doc = "initial jitter buffer (seconds) for timestamp-less audio; "
          "0 = off", .def_real = 0.5},
  {.key = "prime_silence", .type = ConfigType::Bool,
   .doc = "audio-only realtime: go live with a silent track at startup so "
          "viewers can attach before the first PCM arrives; the self-clocked "
          "cadence advances it and real audio resumes gaplessly. Default on; "
          "off keeps the old behavior (stream appears on the first audio beat)",
   .def_bool = true},
};
const PortSpec kIports[] = {
  {.name = "frames", .doc = "video RGB TensorBeat [3,H,W] (F32 or U8); "
                            "stable resolution for the stage's lifetime. "
                            "For an audio-only stream, wire the PCM "
                            "producer here instead (detected by rank).",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "audio", .doc = "OPTIONAL audio PCM TensorBeat: F32 rank-1 "
                           "[n] (mono) or rank-2 [channels, n]. "
                           "sideband.sample_rate + timestamp_us honoured.",
   .type = &typeid(TensorBeatPayload), .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "hls-broadcast",
  .doc       = "Sink: encodes incoming RGB frames (VideoToolbox/libx264) "
               "and/or PCM audio (AAC) into a rolling in-memory HLS "
               "playlist + segments and serves them over an in-process "
               "HTTP server. 0 oports.",
  .display_name = "HLS Broadcast",
  .category  = StageCategory::Network,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
HlsBroadcastStage::spec() const noexcept
{
  return kSpec;
}

HlsBroadcastStage::~HlsBroadcastStage()
{
  teardown_();
}

string
HlsBroadcastStage::av_err_(int rc) const
{
  char buf[256];
  _libs->avutil().api.strerror(rc, buf, sizeof buf);
  return string(buf);
}

int
HlsBroadcastStage::io_open_(AVIOContext** pb, const string& url)
{
  if (!pb) { return AVERROR(EINVAL); }
  auto sink = std::make_unique<InMemSink>();
  sink->url = url;

  // 4 KiB AVIO buffer. FFmpeg drains it via write_packet_ whenever
  // it fills; the muxer writes mpegts in TS-packet-sized chunks
  // (188 B), so a small buffer keeps memory pressure low without
  // adding meaningful syscall overhead (everything stays in
  // userspace).
  constexpr int kIoBuf = 4096;
  auto* buf = static_cast<unsigned char*>(
      _libs->avutil().api.malloc(kIoBuf));
  if (!buf) { return AVERROR(ENOMEM); }

  AVIOContext* ctx = _libs->avformat().api.avio_alloc_context(
      buf, kIoBuf, /*write_flag*/ 1, sink.get(),
      /*read_packet*/ nullptr,
      &InMemSink::write_packet_,
      &InMemSink::seek_);
  if (!ctx) {
    _libs->avutil().api.freep(&buf);
    return AVERROR(ENOMEM);
  }

  _open_sinks.emplace(ctx, std::move(sink));
  *pb = ctx;
  return 0;
}

int
HlsBroadcastStage::io_close_(AVIOContext* pb)
{
  if (!pb) { return 0; }
  _libs->avformat().api.avio_flush(pb);

  auto it = _open_sinks.find(pb);
  std::unique_ptr<InMemSink> sink;
  if (it != _open_sinks.end()) {
    sink = std::move(it->second);
    _open_sinks.erase(it);
  }

  // Detach the AVIO buffer before freeing the context (FFmpeg may
  // have realloc'd it under us; we always have to free whatever it
  // currently points at).
  unsigned char* iobuf = pb->buffer;
  AVIOContext*   tmp   = pb;
  _libs->avformat().api.avio_context_free(&tmp);
  if (iobuf) { _libs->avutil().api.freep(&iobuf); }

  if (!sink) { return 0; }

  const string http_path = "/" + strip_tmp_suffix_(basename_(sink->url));
  string mime = StaticFileServer::mime_type_for(http_path);

  // For .m3u8 publications, optionally inject
  //   #EXT-X-START:TIME-OFFSET=-N,PRECISE=YES
  // so compliant players anchor playback N seconds behind the live
  // edge instead of their default ~3 × target_duration. We splice
  // the tag right after the leading "#EXTM3U" line so the rest of
  // the playlist (VERSION, TARGETDURATION, MEDIA-SEQUENCE, EXTINF
  // entries) is untouched. No-op when the user leaves
  // live_start_offset at 0.
  if (_live_start_offset > 0.0
      && ends_with_(http_path, ".m3u8")
      && !sink->bytes.empty()) {
    auto& bytes = sink->bytes;
    // Locate the end of the first line. Tolerate both LF and CRLF
    // line endings — FFmpeg uses LF on Unix but the spec allows
    // either, and we don't want to break if that ever changes.
    size_t insert_at = 0;
    for (size_t i = 0; i + 1 < bytes.size(); ++i) {
      if (bytes[i] == '\n') {
        insert_at = i + 1;
        break;
      }
    }
    if (insert_at > 0) {
      const string tag = fmt(
          "#EXT-X-START:TIME-OFFSET=-{},PRECISE=YES\n",
          _live_start_offset)();
      bytes.insert(bytes.begin()
                       + static_cast<std::ptrdiff_t>(insert_at),
                   tag.begin(), tag.end());
    }
  }

  _blobs.put(http_path, std::move(sink->bytes), std::move(mime));

  // Bounded-FIFO eviction of mpegts segments. Playlist files are
  // continuously rewritten and don't accumulate.
  if (ends_with_(http_path, ".ts") && _playlist_max_size > 0) {
    _segment_paths.push_back(http_path);
    const size_t cap = static_cast<size_t>(_playlist_max_size) + 1;
    while (_segment_paths.size() > cap) {
      _blobs.erase(_segment_paths.front());
      _segment_paths.pop_front();
    }
  }
  return 0;
}

void
HlsBroadcastStage::resolve_default_fps_(const TensorBeat& tb)
{
  // Adopt the source cadence from the frame's sideband when present;
  // otherwise fall back to 30/1. Called once, before the encoder is
  // built, only when config didn't pin fps.
  int num = 0;
  int den = 0;
  if (tb.sideband.is_object()) {
    auto sb = tb.sideband.as_object();
    if (sb.contains("fps_num") && sb.contains("fps_den")) {
      num = static_cast<int>(sb.at("fps_num").as_uint(0));
      den = static_cast<int>(sb.at("fps_den").as_uint(0));
    }
  }
  if (num > 0 && den > 0) {
    _fps_num = num;
    _fps_den = den;
    session()->info(fmt(
        "hls-broadcast('{}'): adopting source frame rate {}/{} fps "
        "from input sideband", this->id(), _fps_num, _fps_den));
  } else {
    _fps_num = 30;
    _fps_den = 1;
  }
}

bool
HlsBroadcastStage::ensure_encoder_(int H, int W)
{
  if (_enc) { return true; }

  const AVCodec* codec =
      _libs->avcodec().api.find_encoder_by_name(_codec_name.c_str());
  if (!codec) {
    session()->error(fmt(
        "hls-broadcast('{}'): video encoder '{}' not found — check "
        "the 'codec' config (e.g. libx264 / mpeg4)",
        this->id(), _codec_name));
    return false;
  }

  _enc = _libs->avcodec().api.alloc_context3(codec);
  if (!_enc) {
    session()->error(fmt(
        "hls-broadcast('{}'): alloc_context3 failed", this->id()));
    return false;
  }
  _enc->width        = W;
  _enc->height       = H;
  _enc->pix_fmt      = AV_PIX_FMT_YUV420P;
  _enc->time_base    = AVRational{_fps_den, _fps_num};
  _enc->framerate    = AVRational{_fps_num, _fps_den};
  _enc->bit_rate     = _bitrate;
  _enc->gop_size     = _gop_size;
  _enc->max_b_frames = 0;

  // Per-codec AVOption set. VideoToolbox encoders don't understand
  // libx264's preset/tune; libx264 doesn't understand VT's realtime/
  // allow_sw. Sending the wrong knobs just emits "Unknown option"
  // warnings from FFmpeg -- harmless but noisy. Branch explicitly.
  const bool is_vt =
      _codec_name == "h264_videotoolbox"
   || _codec_name == "hevc_videotoolbox";

  AVDictionary* opts = nullptr;
  if (is_vt) {
    _libs->avutil().api.dict_set(&opts, "realtime",   "1", 0);
    _libs->avutil().api.dict_set(&opts, "allow_sw",   "1", 0);
    _libs->avutil().api.dict_set(&opts, "prio_speed", "1", 0);
  } else {
    if (!_preset.empty()) {
      _libs->avutil().api.dict_set(&opts, "preset", _preset.c_str(), 0);
    }
    if (!_tune.empty()) {
      _libs->avutil().api.dict_set(&opts, "tune", _tune.c_str(), 0);
    }
  }
  int rc = _libs->avcodec().api.open2(_enc, codec, &opts);
  _libs->avutil().api.dict_free(&opts);
  if (rc < 0) {
    session()->error(fmt(
        "hls-broadcast('{}'): avcodec_open2: {}",
        this->id(), av_err_(rc)));
    _libs->avcodec().api.free_context(&_enc);
    return false;
  }

  _pkt = _libs->avcodec().api.packet_alloc();
  if (!_pkt) {
    session()->error(fmt(
        "hls-broadcast('{}'): av_packet_alloc failed", this->id()));
    _libs->avcodec().api.free_context(&_enc);
    return false;
  }

  _enc_w = W;
  _enc_h = H;
  return true;
}

bool
HlsBroadcastStage::ensure_sws_(int H, int W)
{
  if (_sws && _gbrp_scratch
      && _gbrp_scratch->width == W && _gbrp_scratch->height == H) {
    return true;
  }
  if (_sws) {
    _libs->swscale().api.free_context(_sws);
    _sws = nullptr;
  }
  if (_gbrp_scratch) {
    _libs->avutil().api.frame_free(&_gbrp_scratch);
  }

  _sws = _libs->swscale().api.get_context(
      W, H, AV_PIX_FMT_GBRP,
      W, H, AV_PIX_FMT_YUV420P,
      SWS_BICUBIC, nullptr, nullptr, nullptr);
  if (!_sws) {
    session()->error(fmt(
        "hls-broadcast('{}'): sws_getContext failed", this->id()));
    return false;
  }

  _gbrp_scratch = _libs->avutil().api.frame_alloc();
  if (!_gbrp_scratch) {
    session()->error(fmt(
        "hls-broadcast('{}'): frame_alloc (gbrp) failed",
        this->id()));
    return false;
  }
  _gbrp_scratch->format = AV_PIX_FMT_GBRP;
  _gbrp_scratch->width  = W;
  _gbrp_scratch->height = H;
  int rc = _libs->avutil().api.frame_get_buffer(_gbrp_scratch, 32);
  if (rc < 0) {
    session()->error(fmt(
        "hls-broadcast('{}'): frame_get_buffer (gbrp): {}",
        this->id(), av_err_(rc)));
    _libs->avutil().api.frame_free(&_gbrp_scratch);
    return false;
  }
  return true;
}

bool
HlsBroadcastStage::open_output_()
{
  if (_ofctx) { return true; }

  AVFormatContext* fctx = nullptr;
  int rc = _libs->avformat().api.alloc_output_context2(
      &fctx, nullptr, "hls", _playlist_name.c_str());
  if (rc < 0 || !fctx) {
    session()->error(fmt(
        "hls-broadcast('{}'): alloc_output_context2('hls', '{}'): "
        "{}", this->id(), _playlist_name, av_err_(rc)));
    return false;
  }

  // Route the muxer's file I/O through our io_open/io_close2
  // callbacks; the hls muxer opens both the .m3u8 and each .ts
  // through these, so all bytes land in `_blobs` instead of the
  // filesystem. `opaque` carries the back-pointer used by the
  // thunks to recover the stage instance.
  fctx->opaque    = this;
  fctx->io_open   = &hls_io_open_thunk_;
  fctx->io_close2 = &hls_io_close_thunk_;

  // Add the wanted stream(s). Video first (stream 0) when present, so
  // its index stays 0 as before; audio follows. At least one is added.
  AVStream* vst = nullptr;
  AVStream* ast = nullptr;
  if (_want_video && _enc) {
    vst = _libs->avformat().api.new_stream(fctx, nullptr);
    if (!vst) {
      _libs->avformat().api.free_context(fctx);
      session()->error(fmt(
          "hls-broadcast('{}'): avformat_new_stream (video) failed",
          this->id()));
      return false;
    }
    rc = _libs->avcodec().api.parameters_from_context(vst->codecpar,
                                                      _enc);
    if (rc < 0) {
      _libs->avformat().api.free_context(fctx);
      session()->error(fmt(
          "hls-broadcast('{}'): parameters_from_context (video): {}",
          this->id(), av_err_(rc)));
      return false;
    }
    vst->time_base = AVRational{_fps_den, _fps_num};
  }
  if (_want_audio && _aenc) {
    ast = _libs->avformat().api.new_stream(fctx, nullptr);
    if (!ast) {
      _libs->avformat().api.free_context(fctx);
      session()->error(fmt(
          "hls-broadcast('{}'): avformat_new_stream (audio) failed",
          this->id()));
      return false;
    }
    rc = _libs->avcodec().api.parameters_from_context(ast->codecpar,
                                                      _aenc);
    if (rc < 0) {
      _libs->avformat().api.free_context(fctx);
      session()->error(fmt(
          "hls-broadcast('{}'): parameters_from_context (audio): {}",
          this->id(), av_err_(rc)));
      return false;
    }
    ast->time_base = AVRational{1, _audio_sample_rate};
  }
  if (!vst && !ast) {
    _libs->avformat().api.free_context(fctx);
    session()->error(fmt(
        "hls-broadcast('{}'): no streams to write (neither video nor "
        "audio ready)", this->id()));
    return false;
  }

  AVDictionary* mux_opts = nullptr;
  _libs->avutil().api.dict_set(
      &mux_opts, "hls_time",
      to_string(_segment_duration).c_str(), 0);
  _libs->avutil().api.dict_set(
      &mux_opts, "hls_list_size",
      to_string(_playlist_max_size).c_str(), 0);
  if (!_hls_flags.empty()) {
    _libs->avutil().api.dict_set(&mux_opts, "hls_flags",
                                _hls_flags.c_str(), 0);
  }

  rc = _libs->avformat().api.write_header(fctx, &mux_opts);
  _libs->avutil().api.dict_free(&mux_opts);
  if (rc < 0) {
    _libs->avformat().api.free_context(fctx);
    session()->error(fmt(
        "hls-broadcast('{}'): write_header (hls): {}",
        this->id(), av_err_(rc)));
    return false;
  }

  _ofctx          = fctx;
  _vstream        = vst;
  _astream        = ast;
  _video_muxed    = _video_muxed || (vst != nullptr);
  _audio_muxed    = _audio_muxed || (ast != nullptr);
  _header_written = true;
  return true;
}

AVFrame*
HlsBroadcastStage::tensor_to_yuv_(const TensorBeat& tb)
{
  const int H = _enc_h;
  const int W = _enc_w;
  const size_t expected_elems = static_cast<size_t>(3) * H * W;

  // Fill the gbrp scratch. Two paths:
  //   * U8 input: bytes are already in [0, 255] — straight memcpy
  //     into each gbrp plane (channel-permuted to GBR order).
  //   * F32 input: scale to [0, 255] (or pass-through if the
  //     producer is already emitting that range) with rounding +
  //     clamp, byte by byte.
  if (tb.dtype == TensorBeat::DType::U8) {
    AlignedVector<uint8_t> tmp;
    const uint8_t* src = nullptr;
    if (tb.is_contiguous() && tb.byte_size() == expected_elems) {
      src = tb.as_u8();
    } else {
      tmp = tb.materialize_contiguous();
      src = tmp.data();
    }
    for (int c = 0; c < 3; ++c) {
      const int      plane = kGbrpPlaneForChannel[c];
      uint8_t*       dst   = _gbrp_scratch->data[plane];
      const int      ls    = _gbrp_scratch->linesize[plane];
      const uint8_t* src_c = src + static_cast<size_t>(c) * H * W;
      for (int y = 0; y < H; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * ls,
                    src_c + static_cast<size_t>(y) * W,
                    static_cast<size_t>(W));
      }
    }
  } else if (tb.dtype == TensorBeat::DType::F32) {
    AlignedVector<float> tmp;
    const float* src = nullptr;
    if (tb.is_contiguous()
        && tb.byte_size() == expected_elems * sizeof(float)) {
      src = tb.as_f32();
    } else {
      tmp = tb.materialize_contiguous_as<float>();
      src = tmp.data();
    }
    const float scale = _input_normalized ? 255.0f : 1.0f;
    for (int c = 0; c < 3; ++c) {
      const int     plane = kGbrpPlaneForChannel[c];
      uint8_t*      dst   = _gbrp_scratch->data[plane];
      const int     ls    = _gbrp_scratch->linesize[plane];
      const float*  src_c = src + static_cast<size_t>(c) * H * W;
      for (int y = 0; y < H; ++y) {
        uint8_t*     drow = dst + static_cast<size_t>(y) * ls;
        const float* srow = src_c + static_cast<size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
          drow[x] = clamp_byte_(srow[x] * scale);
        }
      }
    }
  } else {
    session()->warn(fmt(
        "hls-broadcast('{}'): unsupported input dtype '{}' "
        "(only f32/u8 are supported) — emitting black frame",
        this->id(), tb.dtype_name()));
    for (int c = 0; c < 3; ++c) {
      const int plane = kGbrpPlaneForChannel[c];
      uint8_t*  dst   = _gbrp_scratch->data[plane];
      const int ls    = _gbrp_scratch->linesize[plane];
      for (int y = 0; y < H; ++y) {
        std::memset(dst + static_cast<size_t>(y) * ls, 0,
                    static_cast<size_t>(W));
      }
    }
  }

  AVFrame* yuv = _libs->avutil().api.frame_alloc();
  if (!yuv) {
    session()->warn(fmt(
        "hls-broadcast('{}'): frame_alloc (yuv420p) failed",
        this->id()));
    return nullptr;
  }
  yuv->format = AV_PIX_FMT_YUV420P;
  yuv->width  = W;
  yuv->height = H;
  int rc = _libs->avutil().api.frame_get_buffer(yuv, 32);
  if (rc < 0) {
    session()->warn(fmt(
        "hls-broadcast('{}'): frame_get_buffer (yuv420p): {}",
        this->id(), av_err_(rc)));
    _libs->avutil().api.frame_free(&yuv);
    return nullptr;
  }
  _libs->swscale().api.scale(
      _sws,
      _gbrp_scratch->data, _gbrp_scratch->linesize, 0, H,
      yuv->data, yuv->linesize);

  // Assign PTS + decide whether this frame should be an IDR.
  //
  // In _realtime mode, media time tracks wall-clock so HLS segments
  // close every `segment_duration` wall-clock seconds regardless of
  // how fast frames arrive. The encoder's own gop_size logic
  // schedules keyframes by *frame count*, which gives the wrong
  // wall-clock cadence when the upstream rate is below the declared
  // fps (e.g. yolov6l at ~1 fps with fps_num=30 would produce a
  // keyframe every gop_size frames == many wall-clock seconds). So
  // we also force AV_PICTURE_TYPE_I on the frame that crosses each
  // segment_duration boundary; the HLS muxer can then close
  // segments on schedule.
  //
  // When _realtime is false (deterministic tests), PTS is just a
  // monotonic frame index and we leave keyframe scheduling to the
  // encoder.
  // A/V ts-sync: when audio is also present and this frame carries a
  // wall-clock `timestamp_us`, derive its PTS from the shared UTC epoch
  // so video and audio share one clock. (When there's no audio we keep
  // the legacy wall-clock/frame-index paths so behaviour is unchanged.)
  uint64_t ts_us  = 0;
  bool     has_ts = false;
  if (tb.sideband.is_object()) {
    auto sb = tb.sideband.as_object();
    if (sb.contains("timestamp_us")) {
      ts_us  = sb.at("timestamp_us").as_uint(0);
      has_ts = true;
    }
  }
  const bool ts_sync = _want_audio && has_ts;
  if (ts_sync) { note_epoch_(ts_us); }

  auto now = std::chrono::steady_clock::now();
  int64_t new_pts    = 0;
  int64_t media_us   = 0;
  bool    force_kf   = false;
  if (_realtime) {
    if (!_clock_started) {
      _epoch         = now;
      _last_kf_time  = now;
      _clock_started = true;
      force_kf = true;            // first frame must be IDR
    }
    if (ts_sync && _av_epoch_set) {
      media_us = static_cast<int64_t>(ts_us)
               - static_cast<int64_t>(_av_epoch_us);
      if (media_us < 0) { media_us = 0; }
    } else {
      media_us = std::chrono::duration_cast<std::chrono::microseconds>(
          now - _epoch).count();
    }
    new_pts = _libs->avutil().api.rescale_q(
        media_us, AVRational{1, 1'000'000}, _enc->time_base);
    auto since_kf = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - _last_kf_time).count();
    // Truncation guard: segment_duration is `double` (can be 0.5);
    // round to ms so the keyframe-pacing threshold is sane for
    // sub-second segments.
    const int64_t target_ms =
        static_cast<int64_t>(_segment_duration * 1000.0);
    if (since_kf >= target_ms) {
      force_kf = true;
      _last_kf_time = now;
    }
  } else {
    new_pts = _pts++;
  }
  if (new_pts <= _last_pts) {
    new_pts = _last_pts + 1;
  }
  _last_pts = new_pts;
  // Track the video media clock (microseconds from epoch) so ts-less
  // audio can anchor its start to the current video edge.
  if (_realtime) {
    _video_media_us = media_us;
  } else {
    _video_media_us = _libs->avutil().api.rescale_q(
        new_pts, _enc->time_base, AVRational{1, 1'000'000});
  }
  yuv->pts  = new_pts;
  yuv->pict_type = force_kf ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
  return yuv;
}

int
HlsBroadcastStage::drain_packets_()
{
  while (true) {
    int rc = _libs->avcodec().api.receive_packet(_enc, _pkt);
    if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
      return 0;
    }
    if (rc < 0) {
      session()->warn(fmt(
          "hls-broadcast('{}'): receive_packet: {}",
          this->id(), av_err_(rc)));
      return -1;
    }
    _pkt->stream_index = _vstream->index;
    _libs->avcodec().api.packet_rescale_ts(_pkt, _enc->time_base,
                                           _vstream->time_base);
    rc = _libs->avformat().api.interleaved_write_frame(_ofctx, _pkt);
    _libs->avcodec().api.packet_unref(_pkt);
    if (rc < 0) {
      session()->warn(fmt(
          "hls-broadcast('{}'): interleaved_write_frame: {}",
          this->id(), av_err_(rc)));
      return -1;
    }
  }
}

int
HlsBroadcastStage::send_and_mux_(AVFrame* yuv)
{
  int rc = _libs->avcodec().api.send_frame(_enc, yuv);
  if (rc < 0) {
    session()->warn(fmt(
        "hls-broadcast('{}'): send_frame: {}",
        this->id(), av_err_(rc)));
    return -1;
  }
  return drain_packets_();
}

// ------------------------------------------------------------------
// Role resolution + muxer-open orchestration
// ------------------------------------------------------------------

void
HlsBroadcastStage::resolve_roles_(RuntimeContext& ctx)
{
  if (_roles_resolved) { return; }
  // Strict positional roles: iport 0 is video, iport 1 is audio. Either
  // may be left disconnected (an optional input); connectedness alone
  // decides which streams exist -- for an audio-only stream leave
  // iport 0 unwired and wire the PCM producer to iport 1.
  _want_video = ctx.iport_connected(0);
  _want_audio = ctx.num_iports() >= 2 && ctx.iport_connected(1);
  _roles_resolved = true;
  session()->info(fmt(
      "hls-broadcast('{}'): inputs resolved -> video={}, audio={}",
      this->id(), _want_video, _want_audio));
}

void
HlsBroadcastStage::start_http_()
{
  if (!_serve_http || _http) { return; }
  _http = std::make_unique<StaticFileServer>(
      session(), /*doc_root*/ "", _bind_address, _http_port, &_blobs);
  if (!_http->start()) {
    session()->warn(fmt(
        "hls-broadcast('{}'): static HTTP server failed to start on "
        "{}:{}; the in-memory blob registry is still being populated "
        "and is accessible via the stage's blobs() inspector for "
        "tests.", this->id(), _bind_address, _http_port));
    _http.reset();
  }
}

bool
HlsBroadcastStage::maybe_open_muxer_()
{
  if (_ofctx) { return true; }
  if (!_roles_resolved) { return false; }
  // Video needs its encoder (built from the first frame's W/H).
  if (_want_video && !_enc) { return false; }
  // Audio needs its encoder; give up on audio if it can't be built
  // rather than blocking the whole output.
  if (_want_audio && !_aenc) {
    if (!ensure_audio_encoder_()) { _want_audio = false; }
  }
  if (!_want_video && !_want_audio) { return false; }
  if (!open_output_()) { return false; }
  start_http_();
  return true;
}

// ------------------------------------------------------------------
// Video handling (encode + mux one frame)
// ------------------------------------------------------------------

void
HlsBroadcastStage::handle_video_(const TensorBeat& tb)
{
  const int H = static_cast<int>(tb.shape[1]);
  const int W = static_cast<int>(tb.shape[2]);

  if (!_enc) {
    // Resolve the "auto" fps default from the first frame's sideband
    // (video-to-rgb forwards the capture stream's rate) before the
    // encoder is built; an explicit fps in config skips this.
    if (!_fps_from_config) { resolve_default_fps_(tb); }
    if (!ensure_encoder_(H, W) || !ensure_sws_(H, W)) {
      _fatal = true;
      return;
    }
  } else if (H != _enc_h || W != _enc_w) {
    session()->warn(fmt(
        "hls-broadcast('{}'): resolution changed mid-stream "
        "({}x{} -> {}x{}); dropping frame",
        this->id(), _enc_w, _enc_h, W, H));
    return;
  }

  if (!maybe_open_muxer_()) { return; }

  // Diagnostic: log min/mean/max of the input tensor periodically.
  // A frame whose mean is essentially 0 and whose range is < a few
  // percent is the smoking gun for "all-black playback".
  if (_log_input_stats_every > 0
      && (_frames_in % static_cast<uint64_t>(_log_input_stats_every))
             == 0) {
    double mn = 0.0, mx = 0.0, sum = 0.0;
    size_t n   = 0;
    bool   any = false;
    auto accum = [&](double v) {
      if (!any) { mn = mx = v; any = true; }
      else      { if (v < mn) { mn = v; } if (v > mx) { mx = v; } }
      sum += v;
    };
    if (tb.dtype == TensorBeat::DType::F32) {
      AlignedVector<float> tmp;
      const float* data = nullptr;
      if (tb.is_contiguous()) { data = tb.as_f32(); n = tb.element_count(); }
      else { tmp = tb.materialize_contiguous_as<float>();
             data = tmp.data(); n = tmp.size(); }
      for (size_t i = 0; i < n; ++i) { accum(static_cast<double>(data[i])); }
    } else if (tb.dtype == TensorBeat::DType::U8) {
      AlignedVector<uint8_t> tmp;
      const uint8_t* data = nullptr;
      if (tb.is_contiguous()) { data = tb.as_u8(); n = tb.element_count(); }
      else { tmp = tb.materialize_contiguous();
             data = tmp.data(); n = tmp.size(); }
      for (size_t i = 0; i < n; ++i) { accum(static_cast<double>(data[i])); }
    }
    session()->info(fmt(
        "hls-broadcast('{}'): input frame #{} dtype={} shape=[3,{},{}] "
        "elements={} min={:.4f} mean={:.4f} max={:.4f}",
        this->id(), _frames_in, tb.dtype_name(), H, W, n,
        mn, n > 0 ? sum / static_cast<double>(n) : 0.0, mx));
  }

  AVFrame* yuv = tensor_to_yuv_(tb);
  if (yuv) {
    send_and_mux_(yuv);
    _libs->avutil().api.frame_free(&yuv);
  }
  // Drain any audio buffered while the muxer was still coming up.
  if (_want_audio) { pump_audio_(false); }
}

// ------------------------------------------------------------------
// Audio handling (resample -> FIFO -> AAC -> mux)
// ------------------------------------------------------------------

namespace {
// Build a native AVChannelLayout for a plain N-channel stream. Mono /
// stereo use the well-known masks; anything else gets the low-N bits.
AVChannelLayout
chlayout_for_(int ch)
{
  AVChannelLayout l = AV_CHANNEL_LAYOUT_MONO;
  if (ch == 2) {
    l = AV_CHANNEL_LAYOUT_STEREO;
  } else if (ch > 2) {
    l = AV_CHANNEL_LAYOUT_MASK(ch, (1ULL << ch) - 1ULL);
  }
  return l;
}
}  // namespace

bool
HlsBroadcastStage::ensure_audio_encoder_()
{
  if (_aenc) { return true; }
  const auto& cdc = _libs->avcodec().api;

  const AVCodec* codec =
      cdc.find_encoder_by_name(_audio_codec_name.c_str());
  if (!codec) { codec = cdc.find_encoder(AV_CODEC_ID_AAC); }
  if (!codec) {
    session()->error(fmt(
        "hls-broadcast('{}'): audio encoder '{}' not found "
        "(and no AAC fallback)", this->id(), _audio_codec_name));
    return false;
  }

  _aenc = cdc.alloc_context3(codec);
  if (!_aenc) {
    session()->error(fmt(
        "hls-broadcast('{}'): audio alloc_context3 failed", this->id()));
    return false;
  }
  _aenc->sample_rate = _audio_sample_rate;
  _aenc->ch_layout   = chlayout_for_(_audio_channels);
  _aenc->sample_fmt  = AV_SAMPLE_FMT_FLTP;
  _aenc->bit_rate    = _audio_bitrate;
  _aenc->time_base   = AVRational{1, _audio_sample_rate};

  int rc = cdc.open2(_aenc, codec, nullptr);
  if (rc < 0) {
    session()->error(fmt(
        "hls-broadcast('{}'): audio avcodec_open2: {}",
        this->id(), av_err_(rc)));
    cdc.free_context(&_aenc);
    return false;
  }

  _apkt = cdc.packet_alloc();
  if (!_apkt) {
    session()->error(fmt(
        "hls-broadcast('{}'): audio packet_alloc failed", this->id()));
    cdc.free_context(&_aenc);
    return false;
  }
  // AAC has a fixed 1024-sample frame; fall back to that if the codec
  // reports variable (0) so the FIFO has a definite chunk size.
  _aenc_frame_size = _aenc->frame_size > 0 ? _aenc->frame_size : 1024;

  _aframe = _libs->avutil().api.frame_alloc();
  if (!_aframe) {
    session()->error(fmt(
        "hls-broadcast('{}'): audio frame_alloc failed", this->id()));
    cdc.free_context(&_aenc);
    return false;
  }
  _aframe->format      = AV_SAMPLE_FMT_FLTP;
  _aframe->ch_layout   = _aenc->ch_layout;
  _aframe->sample_rate = _audio_sample_rate;
  _aframe->nb_samples  = _aenc_frame_size;
  rc = _libs->avutil().api.frame_get_buffer(_aframe, 0);
  if (rc < 0) {
    session()->error(fmt(
        "hls-broadcast('{}'): audio frame_get_buffer: {}",
        this->id(), av_err_(rc)));
    _libs->avutil().api.frame_free(&_aframe);
    cdc.free_context(&_aenc);
    return false;
  }
  _afifo.assign(static_cast<size_t>(_audio_channels), {});
  return true;
}

bool
HlsBroadcastStage::ensure_swr_(int in_rate, int in_ch)
{
  if (_swr && _swr_in_rate == in_rate && _swr_in_ch == in_ch) {
    return true;
  }
  if (_swr) { _libs->swresample().api.free(&_swr); }
  AVChannelLayout inl  = chlayout_for_(in_ch);
  AVChannelLayout outl = _aenc->ch_layout;
  int rc = _libs->swresample().api.alloc_set_opts2(
      &_swr,
      &outl, AV_SAMPLE_FMT_FLTP, _audio_sample_rate,
      &inl,  AV_SAMPLE_FMT_FLTP, in_rate,
      0, nullptr);
  if (rc < 0 || !_swr) {
    session()->warn(fmt(
        "hls-broadcast('{}'): swr_alloc_set_opts2 failed ({})",
        this->id(), rc));
    return false;
  }
  rc = _libs->swresample().api.init(_swr);
  if (rc < 0) {
    session()->warn(fmt(
        "hls-broadcast('{}'): swr_init failed ({})", this->id(), rc));
    _libs->swresample().api.free(&_swr);
    return false;
  }
  _swr_in_rate = in_rate;
  _swr_in_ch   = in_ch;
  return true;
}

void
HlsBroadcastStage::afifo_push_(const uint8_t* const* planes, int n)
{
  for (int c = 0; c < _audio_channels; ++c) {
    const float* f = reinterpret_cast<const float*>(planes[c]);
    _afifo[static_cast<size_t>(c)].insert(
        _afifo[static_cast<size_t>(c)].end(), f, f + n);
  }
}

int
HlsBroadcastStage::afifo_filled_() const
{
  return _afifo.empty()
       ? 0
       : static_cast<int>(_afifo.front().size());
}

bool
HlsBroadcastStage::afifo_pull_frame_(uint8_t* const* dst)
{
  if (afifo_filled_() < _aenc_frame_size) { return false; }
  for (int c = 0; c < _audio_channels; ++c) {
    auto& src = _afifo[static_cast<size_t>(c)];
    std::memcpy(dst[c], src.data(),
                static_cast<size_t>(_aenc_frame_size) * sizeof(float));
    src.erase(src.begin(), src.begin() + _aenc_frame_size);
  }
  return true;
}

int
HlsBroadcastStage::afifo_pull_padded_(uint8_t* const* dst)
{
  const int have = afifo_filled_();
  if (have <= 0) { return 0; }
  const int real = std::min(have, _aenc_frame_size);
  for (int c = 0; c < _audio_channels; ++c) {
    auto& src = _afifo[static_cast<size_t>(c)];
    std::memcpy(dst[c], src.data(),
                static_cast<size_t>(real) * sizeof(float));
    if (real < _aenc_frame_size) {
      std::memset(reinterpret_cast<float*>(dst[c]) + real, 0,
                  static_cast<size_t>(_aenc_frame_size - real)
                      * sizeof(float));
    }
    src.erase(src.begin(), src.begin() + real);
  }
  return real;
}

void
HlsBroadcastStage::push_audio_samples_(const TensorBeat& tb)
{
  if (!_aenc) { return; }
  if (tb.dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "hls-broadcast('{}'): audio beat dtype '{}' not f32 -- dropping",
        this->id(), tb.dtype_name()));
    return;
  }
  int in_ch = 0;
  int n     = 0;
  if (tb.shape.size() == 1) {
    in_ch = 1;
    n     = static_cast<int>(tb.shape[0]);
  } else if (tb.shape.size() == 2) {
    in_ch = static_cast<int>(tb.shape[0]);
    n     = static_cast<int>(tb.shape[1]);
  } else {
    return;
  }
  if (n <= 0 || in_ch <= 0) { return; }

  int in_rate = _audio_sample_rate;
  if (tb.sideband.is_object()) {
    auto sb = tb.sideband.as_object();
    if (sb.contains("sample_rate")) {
      const int r = static_cast<int>(sb.at("sample_rate").as_int(0));
      if (r > 0) { in_rate = r; }
    }
  }
  if (!ensure_swr_(in_rate, in_ch)) { return; }

  // Planar f32 view: rank-2 [ch, n] is already planar (row c = plane c);
  // rank-1 [n] is a single mono plane.
  const AlignedVector<float> buf = tb.materialize_contiguous_as<float>();
  std::vector<const uint8_t*> src(static_cast<size_t>(in_ch));
  for (int c = 0; c < in_ch; ++c) {
    src[static_cast<size_t>(c)] = reinterpret_cast<const uint8_t*>(
        buf.data() + static_cast<size_t>(c) * n);
  }

  // Output capacity: resampled sample count + slack for swr's buffer.
  const int out_max =
      static_cast<int>(static_cast<int64_t>(n) * _audio_sample_rate
                       / in_rate)
      + 1024;
  std::vector<std::vector<float>> out(
      static_cast<size_t>(_audio_channels),
      std::vector<float>(static_cast<size_t>(out_max)));
  std::vector<uint8_t*> outp(static_cast<size_t>(_audio_channels));
  for (int c = 0; c < _audio_channels; ++c) {
    outp[static_cast<size_t>(c)] =
        reinterpret_cast<uint8_t*>(out[static_cast<size_t>(c)].data());
  }

  const int n_out = _libs->swresample().api.convert(
      _swr, outp.data(), out_max, src.data(), n);
  if (n_out < 0) {
    session()->warn(fmt(
        "hls-broadcast('{}'): swr_convert failed ({})",
        this->id(), n_out));
    return;
  }
  if (n_out > 0) { afifo_push_(outp.data(), n_out); }
}

bool
HlsBroadcastStage::audio_stall_silence_due_(RuntimeContext& ctx)
    const noexcept
{
  // Only audio-only, realtime, no-TC streams self-clock silence: a
  // video+audio stream is already pumped at video cadence, ts-paced
  // audio has its own clock, and non-realtime muxing has no wall clock
  // to chase. Audio is always iport 1 (iport 0 is video). "Stall" = the
  // producer is wired but idle right now (no backlog) yet not at EOS;
  // once it closes, process() ends the stage instead of napping.
  return _want_audio && !_want_video && _realtime
      && _audio_started && !_audio_ts_mode
      && _aenc && _astream
      && ctx.num_iports() >= 2
      && ctx.backlog(1) == 0
      && !ctx.eos(1)
      && !ctx.stop_requested();
}

void
HlsBroadcastStage::prime_audio_silence_(RuntimeContext& ctx)
{
  // Audio-only, realtime, timestamp-less, not already started, and there is
  // genuinely no audio waiting yet (a producer that already emitted starts on
  // its real samples; priming is only to fill the gap BEFORE the first beat).
  if (!_prime_silence || _audio_started || _audio_ts_mode
      || !_realtime || _want_video || !_want_audio
      || ctx.num_iports() < 2 || ctx.backlog(1) > 0 || ctx.eos(1)) {
    return;
  }
  if (!maybe_open_muxer_() || !_aenc || !_astream) { return; }
  // Start the clock now (pts 0) without the jitter wait -- there is no real
  // audio to cushion; pump_audio_ synthesizes silence up to the wall clock and
  // real samples later resume from the advanced pts (gapless).
  _audio_started = true;
  _audio_epoch   = std::chrono::steady_clock::now();
  _audio_pts     = 0;
  session()->info(fmt(
      "hls-broadcast('{}'): audio-only live at startup -- broadcasting silence "
      "until the first PCM", this->id()));
  pump_audio_(/*flush=*/false);
}

void
HlsBroadcastStage::pump_audio_(bool flush)
{
  if (!_aenc || !_astream) { return; }
  const auto& cdc = _libs->avcodec().api;

  if (!_audio_started) {
    // Timestamp-less audio waits for an initial jitter cushion; ts-paced
    // audio starts immediately (its own clock keeps it smooth).
    int need = 0;
    if (!_audio_ts_mode) {
      need = static_cast<int>(_audio_buffer_seconds * _audio_sample_rate);
    }
    if (!flush && afifo_filled_() < need) { return; }
    _audio_started = true;
    _audio_epoch   = std::chrono::steady_clock::now();
    const int64_t anchor_us =
        _audio_ts_mode ? _audio_anchor_us
                       : (_want_video ? _video_media_us : 0);
    _audio_pts = _libs->avutil().api.rescale_q(
        anchor_us, AVRational{1, 1'000'000},
        AVRational{1, _audio_sample_rate});
    if (_audio_pts < 0) { _audio_pts = 0; }
    if (!_audio_ts_mode && _audio_buffer_seconds > 0.0) {
      session()->info(fmt(
          "hls-broadcast('{}'): audio jitter buffer filled "
          "(~{}s), starting at pts {}",
          this->id(), _audio_buffer_seconds, _audio_pts));
    }
  }

  // Realtime PACING: release real audio at ~the media-clock rate (plus the
  // jitter cushion), holding the rest in the FIFO. A bursty producer --
  // streaming TTS arrives FASTER than realtime -- would otherwise dump the
  // whole utterance at once, racing the audio pts seconds ahead of the clock;
  // when it stops the stream freezes while the clock catches up and a live
  // player skips the fast-delivered audio. Capping the drain leaves the excess
  // buffered, played out at realtime by the self-clocked cadence (audio-only:
  // the process() silence pump; video-locked: each video frame advances
  // _video_media_us and re-pumps). The pace clock matches the silence-fill's
  // target below -- the video media clock when video is present (keeps A/V
  // locked), else the wall clock since the cushion released. ts-paced audio
  // (its own clock), non-realtime muxing and the final flush drain in full.
  const bool pace = !flush && _audio_started && !_audio_ts_mode && _realtime;
  int64_t pace_limit = INT64_MAX;
  if (pace) {
    const int64_t target_us =
        _want_video
            ? _video_media_us
            : std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - _audio_epoch).count();
    const int64_t cushion =
        static_cast<int64_t>(_audio_buffer_seconds * _audio_sample_rate);
    pace_limit = _libs->avutil().api.rescale_q(
                     target_us, AVRational{1, 1'000'000},
                     AVRational{1, _audio_sample_rate})
                 + cushion;
  }

  while (afifo_filled_() >= _aenc_frame_size && _audio_pts < pace_limit) {
    afifo_pull_frame_(_aframe->data);
    _aframe->pts  = _audio_pts;
    _audio_pts   += _aenc_frame_size;
    int rc = cdc.send_frame(_aenc, _aframe);
    if (rc < 0) {
      session()->warn(fmt(
          "hls-broadcast('{}'): audio send_frame: {}",
          this->id(), av_err_(rc)));
      break;
    }
    drain_audio_packets_();
  }

  // No-TC audio must never pause the stream past the one-time startup
  // cushion: once the FIFO has been drained to empty, keep the audio
  // clock advancing to the current media time by emitting silence rather
  // than stalling. Real samples that arrive later simply resume from the
  // advanced pts, so the track stays gapless. The silence target is the
  // video media clock when video is present (keeps A/V locked), else the
  // wall clock since the cushion was released. ts-paced audio keeps its
  // own clock and non-realtime muxing has no wall clock to chase, so
  // both skip the fill.
  if (!flush && _audio_started && !_audio_ts_mode && _realtime) {
    const int64_t target_us =
        _want_video
            ? _video_media_us
            : std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - _audio_epoch).count();
    const int64_t target_pts = _libs->avutil().api.rescale_q(
        target_us, AVRational{1, 1'000'000},
        AVRational{1, _audio_sample_rate});
    while (_audio_pts + _aenc_frame_size <= target_pts) {
      for (int c = 0; c < _audio_channels; ++c) {
        std::memset(_aframe->data[c], 0,
                    static_cast<size_t>(_aenc_frame_size) * sizeof(float));
      }
      _aframe->pts = _audio_pts;
      _audio_pts  += _aenc_frame_size;
      if (cdc.send_frame(_aenc, _aframe) < 0) { break; }
      drain_audio_packets_();
    }
  }

  if (flush) {
    const int real = afifo_pull_padded_(_aframe->data);
    if (real > 0) {
      _aframe->pts = _audio_pts;
      _audio_pts  += _aenc_frame_size;
      if (cdc.send_frame(_aenc, _aframe) >= 0) {
        drain_audio_packets_();
      }
    }
    cdc.send_frame(_aenc, nullptr);   // EOF -> flush encoder
    drain_audio_packets_();
  }
}

int
HlsBroadcastStage::drain_audio_packets_()
{
  while (true) {
    int rc = _libs->avcodec().api.receive_packet(_aenc, _apkt);
    if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { return 0; }
    if (rc < 0) {
      session()->warn(fmt(
          "hls-broadcast('{}'): audio receive_packet: {}",
          this->id(), av_err_(rc)));
      return -1;
    }
    _apkt->stream_index = _astream->index;
    _libs->avcodec().api.packet_rescale_ts(_apkt, _aenc->time_base,
                                           _astream->time_base);
    rc = _libs->avformat().api.interleaved_write_frame(_ofctx, _apkt);
    _libs->avcodec().api.packet_unref(_apkt);
    if (rc < 0) {
      session()->warn(fmt(
          "hls-broadcast('{}'): audio interleaved_write_frame: {}",
          this->id(), av_err_(rc)));
      return -1;
    }
  }
}

void
HlsBroadcastStage::handle_audio_(const TensorBeat& tb)
{
  if (!_want_audio) { return; }
  if (!ensure_audio_encoder_()) {
    _want_audio = false;
    return;
  }
  if (!_audio_seen) {
    _audio_seen = true;
    if (tb.sideband.is_object()) {
      auto sb = tb.sideband.as_object();
      if (sb.contains("timestamp_us")) {
        const uint64_t ts = sb.at("timestamp_us").as_uint(0);
        _audio_ts_mode = true;
        note_epoch_(ts);
        _audio_anchor_us = static_cast<int64_t>(ts)
                         - static_cast<int64_t>(_av_epoch_us);
        if (_audio_anchor_us < 0) { _audio_anchor_us = 0; }
      }
    }
  }
  push_audio_samples_(tb);
  // Open audio-only output here; when video is also wanted the muxer
  // opens on the first video frame instead and the buffered samples
  // pump then.
  maybe_open_muxer_();
  pump_audio_(false);
}

void
HlsBroadcastStage::close_output_()
{
  if (!_ofctx) { return; }
  if (_header_written && !_trailer_written) {
    int rc = _libs->avformat().api.write_trailer(_ofctx);
    if (rc < 0) {
      session()->warn(fmt(
          "hls-broadcast('{}'): write_trailer: {}",
          this->id(), av_err_(rc)));
    }
    _trailer_written = true;
  }
  _libs->avformat().api.free_context(_ofctx);
  _ofctx   = nullptr;
  _vstream = nullptr;
  _astream = nullptr;
}

void
HlsBroadcastStage::teardown_()
{
  // Flush + close muxer first so the trailer is written before we
  // drop the encoders. write_trailer invokes our io_close on every
  // outstanding AVIOContext, draining _open_sinks naturally.
  if (_ofctx && _header_written && !_trailer_written) {
    if (_enc) {
      _libs->avcodec().api.send_frame(_enc, nullptr);
      drain_packets_();
    }
    if (_aenc && _astream) {
      // Release any samples still buffering (jitter cushion / partial
      // frame) then flush the audio encoder.
      pump_audio_(/*flush=*/true);
    }
  }
  close_output_();

  // Defensive: if the muxer bailed before write_trailer (e.g. a
  // failed write_header), any half-opened AVIOContexts are still
  // tracked here. Free them — we discard the buffered bytes since
  // the muxer never finished the file.
  for (auto& kv : _open_sinks) {
    AVIOContext*   pb    = kv.first;
    unsigned char* iobuf = pb ? pb->buffer : nullptr;
    AVIOContext*   tmp   = pb;
    if (tmp)   { _libs->avformat().api.avio_context_free(&tmp); }
    if (iobuf) { _libs->avutil().api.freep(&iobuf); }
  }
  _open_sinks.clear();

  // Stop the server before letting `_blobs` go out of scope; the
  // server holds a raw pointer into the blob store.
  if (_http) {
    _http->stop();
    _http.reset();
  }
  if (_pkt) {
    _libs->avcodec().api.packet_free(&_pkt);
  }
  if (_enc) {
    _libs->avcodec().api.free_context(&_enc);
  }
  if (_sws) {
    _libs->swscale().api.free_context(_sws);
    _sws = nullptr;
  }
  if (_gbrp_scratch) {
    _libs->avutil().api.frame_free(&_gbrp_scratch);
  }
  // Audio resources.
  if (_swr)    { _libs->swresample().api.free(&_swr); }
  if (_aframe) { _libs->avutil().api.frame_free(&_aframe); }
  if (_apkt)   { _libs->avcodec().api.packet_free(&_apkt); }
  if (_aenc)   { _libs->avcodec().api.free_context(&_aenc); }
  _afifo.clear();
}

Job
HlsBroadcastStage::process(RuntimeContext& ctx)
{
  const unsigned n = ctx.num_iports();

  // Gather the still-open iports (unread backlog, or not yet at EOS).
  std::vector<unsigned> open;
  open.reserve(n);
  for (unsigned p = 0; p < n; ++p) {
    if (ctx.backlog(p) > 0 || !ctx.eos(p)) { open.push_back(p); }
  }
  if (open.empty()) {
    // Every input is drained + closed -> we're done. drain() flushes.
    ctx.signal_done();
    co_return;
  }

  // Resolve roles up-front (idempotent) so an audio-only stream can go live
  // with a silent track at startup -- before the producer's first PCM beat --
  // and viewers that attach early hear silence instead of a 404. The stall
  // path below then keeps the silence advancing until real audio arrives.
  resolve_roles_(ctx);
  prime_audio_silence_(ctx);

  // Audio-only realtime stall: the PCM producer is wired but idle, so
  // read_any() below would block indefinitely and freeze the audio
  // track. Instead nap ~one frame on the pool timer (without pinning a
  // worker) and pump silence up to the wall clock, then return so the
  // driver re-enters -- a self-clocked silence source that keeps the
  // live edge advancing until real PCM resumes or the input closes. Real
  // beats arriving during the nap are drained on the next iteration (the
  // added latency is at most one frame, ~21ms at 48 kHz).
  if (audio_stall_silence_due_(ctx)) {
    pump_audio_(/*flush=*/false);
    const double frame_secs =
        static_cast<double>(_aenc_frame_size) / _audio_sample_rate;
    const auto chunk =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(frame_secs));
    ThreadPool* pool = session() ? session()->thread_pool() : nullptr;
    if (pool) {
      co_await TimerAwaiter{pool, chunk};
    } else {
      // No pool (e.g. a unit context): fall back to a blocking nap.
      std::this_thread::sleep_for(chunk);
    }
    co_return;
  }

  // Wait until at least one open port is readable, then drain whatever
  // is ready on each. read_any handles the two clock domains (video is
  // faster than audio) without blocking on the slower one.
  co_await ctx.read_any(open);

  for (unsigned p : open) {
    const std::uint32_t avail = ctx.backlog(p);
    for (std::uint32_t i = 0; i < avail; ++i) {
      auto beat = co_await ctx.read(p);
      if (!beat) { break; }   // this port hit EOS mid-drain
      const auto* tb =
          dynamic_cast<const TensorBeatPayload*>(beat.get());
      if (!tb) {
        session()->warn(fmt(
            "hls-broadcast('{}'): iport {} beat is not a TensorBeat "
            "-- dropping", this->id(), p));
        continue;
      }
      if (!_roles_resolved) { resolve_roles_(ctx); }
      // Strict positional roles: iport 0 carries video, iport 1 carries
      // audio. A beat whose rank doesn't match its port's role is
      // dropped (e.g. PCM audio wrongly wired to the video port 0).
      const bool is_video = (tb->shape.size() == 3 && tb->shape[0] == 3);
      const bool is_audio = (tb->shape.size() == 1
                             || tb->shape.size() == 2);
      if (p == 0) {
        if (is_video) {
          ++_frames_in;
          handle_video_(*tb);
        } else {
          session()->warn(fmt(
              "hls-broadcast('{}'): iport 0 is the VIDEO port but got a "
              "rank-{} beat (audio belongs on iport 1) -- dropping",
              this->id(), tb->shape.size()));
        }
      } else if (p == 1) {
        if (is_audio) {
          handle_audio_(*tb);
        } else {
          session()->warn(fmt(
              "hls-broadcast('{}'): iport 1 is the AUDIO port but got a "
              "rank-{} beat -- dropping", this->id(), tb->shape.size()));
        }
      }
    }
  }

  // If video was expected (>=2 inputs) but its port closed without ever
  // delivering a frame, fall back to an audio-only stream so a live
  // audio input still broadcasts.
  if (_want_video && !_enc && _want_audio && n >= 1 && ctx.eos(0)) {
    session()->info(fmt(
        "hls-broadcast('{}'): video input closed with no frames; "
        "broadcasting audio only", this->id()));
    _want_video = false;
    maybe_open_muxer_();
    pump_audio_(false);
  }

  if (_fatal) { ctx.signal_done(); }
}

Job
HlsBroadcastStage::drain(RuntimeContext& /*ctx*/)
{
  teardown_();
  co_return;
}

VPIPE_REGISTER_STAGE(HlsBroadcastStage)
VPIPE_REGISTER_SPEC(HlsBroadcastStage, kSpec)

}
