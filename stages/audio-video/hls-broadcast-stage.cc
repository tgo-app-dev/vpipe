#include "stages/audio-video/hls-broadcast-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <utility>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
}

using namespace std;

namespace vpipe {

namespace {

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
  _fps_num           = static_cast<int>(attr_int("fps_num"));
  _fps_den           = static_cast<int>(attr_int("fps_den"));
  _input_normalized  = attr_bool("input_normalized");
  _realtime          = attr_bool("realtime");
  _log_input_stats_every =
      static_cast<int>(attr_int("log_input_stats_every"));
  if (_log_input_stats_every < 0) { _log_input_stats_every = 0; }
  _serve_http        = attr_bool("serve_http");
  _bind_address      = attr_str("bind_address");
  _http_port         = static_cast<int>(attr_int("port"));

  if (_fps_num <= 0)            { _fps_num = 30; }
  if (_fps_den <= 0)            { _fps_den = 1; }
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
   .doc = "frame-rate numerator", .def_int = 30},
  {.key = "fps_den", .type = ConfigType::Int,
   .doc = "frame-rate denominator", .def_int = 1},
  {.key = "input_normalized", .type = ConfigType::Bool,
   .doc = "F32 input in [0,1] vs [0,255]", .def_bool = true},
  {.key = "realtime", .type = ConfigType::Bool,
   .doc = "PTS tracks wall-clock + forced keyframes", .def_bool = true},
  {.key = "log_input_stats_every", .type = ConfigType::Int,
   .doc = "log min/mean/max every Nth frame; 0 = off", .def_int = 0},
  {.key = "serve_http", .type = ConfigType::Bool,
   .doc = "run in-process static HTTP server", .def_bool = true},
  {.key = "bind_address", .type = ConfigType::String,
   .doc = "HTTP server bind address", .def_str = "0.0.0.0"},
  {.key = "port", .type = ConfigType::Int,
   .doc = "HTTP server port, [0,65535]", .def_int = 8080},
};
const PortSpec kIports[] = {
  {.name = "frames", .doc = "RGB TensorBeat [3,H,W] (F32 or U8); stable "
                            "resolution for the stage's lifetime",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "hls-broadcast",
  .doc       = "Sink: encodes incoming RGB frames (VideoToolbox/libx264) "
               "into a rolling in-memory HLS playlist + segments and "
               "serves them over an in-process HTTP server. 0 oports.",
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

  AVStream* st = _libs->avformat().api.new_stream(fctx, nullptr);
  if (!st) {
    _libs->avformat().api.free_context(fctx);
    session()->error(fmt(
        "hls-broadcast('{}'): avformat_new_stream failed",
        this->id()));
    return false;
  }
  rc = _libs->avcodec().api.parameters_from_context(st->codecpar,
                                                    _enc);
  if (rc < 0) {
    _libs->avformat().api.free_context(fctx);
    session()->error(fmt(
        "hls-broadcast('{}'): parameters_from_context: {}",
        this->id(), av_err_(rc)));
    return false;
  }
  st->time_base = AVRational{_fps_den, _fps_num};

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
  _vstream        = st;
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
  auto now = std::chrono::steady_clock::now();
  int64_t new_pts = 0;
  bool    force_kf = false;
  if (_realtime) {
    if (!_clock_started) {
      _epoch         = now;
      _last_kf_time  = now;
      _clock_started = true;
      force_kf = true;            // first frame must be IDR
    }
    int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(
        now - _epoch).count();
    new_pts = _libs->avutil().api.rescale_q(
        us, AVRational{1, 1'000'000}, _enc->time_base);
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
}

void
HlsBroadcastStage::teardown_()
{
  // Flush + close muxer first so the trailer is written before we
  // drop the encoder. write_trailer invokes our io_close on every
  // outstanding AVIOContext, draining _open_sinks naturally.
  if (_enc && _ofctx && _header_written && !_trailer_written) {
    _libs->avcodec().api.send_frame(_enc, nullptr);
    drain_packets_();
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
}

Job
HlsBroadcastStage::process(RuntimeContext& ctx)
{
  auto in_opt = co_await ctx.read(0);
  if (!in_opt) {
    ctx.signal_done();
    co_return;
  }
  const auto* tin = dynamic_cast<const TensorBeatPayload*>(in_opt.get());
  if (!tin || tin->shape.size() != 3 || tin->shape[0] != 3) {
    session()->warn(fmt(
        "hls-broadcast('{}'): iport0 not a [3,H,W] TensorBeat — "
        "dropping beat", this->id()));
    co_return;
  }
  const int H = static_cast<int>(tin->shape[1]);
  const int W = static_cast<int>(tin->shape[2]);

  if (!_enc) {
    if (!ensure_encoder_(H, W)) {
      ctx.signal_done();
      co_return;
    }
    if (!ensure_sws_(H, W)) {
      ctx.signal_done();
      co_return;
    }
    if (!open_output_()) {
      ctx.signal_done();
      co_return;
    }
    if (_serve_http) {
      _http = std::make_unique<StaticFileServer>(
          session(), /*doc_root*/ "", _bind_address, _http_port,
          &_blobs);
      if (!_http->start()) {
        session()->warn(fmt(
            "hls-broadcast('{}'): static HTTP server failed to "
            "start on {}:{}; the in-memory blob registry is still "
            "being populated and is accessible via the stage's "
            "blobs() inspector for tests.",
            this->id(), _bind_address, _http_port));
        _http.reset();
      }
    }
  } else if (H != _enc_h || W != _enc_w) {
    session()->warn(fmt(
        "hls-broadcast('{}'): resolution changed mid-stream "
        "({}x{} -> {}x{}); dropping frame",
        this->id(), _enc_w, _enc_h, W, H));
    co_return;
  }

  // Diagnostic: log min/mean/max of the input tensor periodically.
  // A frame whose mean is essentially 0 and whose range is < a few
  // percent is the smoking gun for "all-black playback" — it means
  // the data is already black before we touch it, so the bug is
  // upstream of this stage.
  if (_log_input_stats_every > 0) {
    if ((_frames_in % static_cast<uint64_t>(_log_input_stats_every))
        == 0) {
      double mn = 0.0, mx = 0.0, sum = 0.0;
      size_t n   = 0;
      bool   any = false;
      auto accum = [&](double v) {
        if (!any) { mn = mx = v; any = true; }
        else      { if (v < mn) { mn = v; } if (v > mx) { mx = v; } }
        sum += v;
      };
      if (tin->dtype == TensorBeat::DType::F32) {
        AlignedVector<float> tmp;
        const float* data = nullptr;
        if (tin->is_contiguous()) {
          data = tin->as_f32();
          n    = tin->element_count();
        } else {
          tmp  = tin->materialize_contiguous_as<float>();
          data = tmp.data();
          n    = tmp.size();
        }
        for (size_t i = 0; i < n; ++i) {
          accum(static_cast<double>(data[i]));
        }
      } else if (tin->dtype == TensorBeat::DType::U8) {
        AlignedVector<uint8_t> tmp;
        const uint8_t* data = nullptr;
        if (tin->is_contiguous()) {
          data = tin->as_u8();
          n    = tin->element_count();
        } else {
          tmp  = tin->materialize_contiguous();
          data = tmp.data();
          n    = tmp.size();
        }
        for (size_t i = 0; i < n; ++i) {
          accum(static_cast<double>(data[i]));
        }
      }
      session()->info(fmt(
          "hls-broadcast('{}'): input frame #{} dtype={} shape=[3,{},{}] "
          "elements={} min={:.4f} mean={:.4f} max={:.4f}",
          this->id(), _frames_in, tin->dtype_name(), H, W, n,
          mn, n > 0 ? sum / static_cast<double>(n) : 0.0, mx));
    }
  }
  ++_frames_in;

  AVFrame* yuv = tensor_to_yuv_(*tin);
  if (yuv) {
    send_and_mux_(yuv);
    _libs->avutil().api.frame_free(&yuv);
  }
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
