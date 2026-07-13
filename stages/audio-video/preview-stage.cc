#include "stages/audio-video/preview-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/thread-pool.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <coroutine>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

using namespace std;

namespace vpipe {

namespace {

// Default resolution for the pre-input black frame; the stream adopts the
// input's native size on the first real frame.
constexpr int kDefaultW = 640;
constexpr int kDefaultH = 360;

// GBRP plane indices: G=0, B=1, R=2. RGB channel c maps here.
const int kGbrpPlaneForChannel[3] = { 2, 0, 1 };

uint8_t
clamp_byte_(float v)
{
  if (v <= 0.0f) { return 0; }
  if (v >= 255.0f) { return 255; }
  return static_cast<uint8_t>(lroundf(v));
}

string
hex_byte_(uint8_t b)
{
  static const char* d = "0123456789abcdef";
  string s;
  s.push_back(d[(b >> 4) & 0x0f]);
  s.push_back(d[b & 0x0f]);
  return s;
}

// Re-schedules the awaiting coroutine on the pool after `delay` instead of
// blocking a worker with a sleep (the ChronoStage / hls-broadcast pattern).
// Drives the preview's self-clocked cadence without pinning a worker.
struct TimerAwaiter {
  ThreadPool*                         pool;
  std::chrono::steady_clock::duration delay;

  bool
  await_ready() const noexcept
  {
    return pool == nullptr
        || delay <= std::chrono::steady_clock::duration::zero();
  }
  void
  await_suspend(std::coroutine_handle<> h) const noexcept
  {
    ThreadPool* p = pool;
    auto        d = delay;
    p->schedule_after(d, h);
  }
  void await_resume() const noexcept {}
};

constexpr ConfigKey kAttrs[] = {
  {.key = "bitrate", .type = ConfigType::Int,
   .doc = "target bps, min 64000", .def_int = 2'000'000},
  {.key = "input_normalized", .type = ConfigType::Bool,
   .doc = "F32 input in [0,1] vs [0,255]", .def_bool = true},
  {.key = "title", .type = ConfigType::String,
   .doc = "optional label shown in the web-ui preview picker; empty = "
          "the stage id", .def_str = ""},
};
const PortSpec kIports[] = {
  {.name = "frames", .doc = "video RGB TensorBeat [3,H,W] (F32 or U8); the "
                            "first frame sets the native resolution",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-frames", .clock_group = 0},
  {.name = "audio", .doc = "OPTIONAL audio PCM TensorBeat: F32 rank-1 [n] "
                           "(mono) or rank-2 [channels, n]. "
                           "sideband.sample_rate honoured.",
   .type = &typeid(TensorBeatPayload),
   .tags = "pcm-samples", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "preview",
  .doc       = "Sink: self-clocked live video preview (default 25 fps; "
               "black before input, sample/repeat after, adopts input "
               "fps) encoded as fragmented-MP4 H.264 and streamed to the "
               "web-ui over WebSocket -> Media Source Extensions (plays "
               "over plain-HTTP LAN, no HTTPS needed). Optional PCM audio "
               "passthrough. 0 oports.",
  .display_name = "Preview",
  .category  = StageCategory::Network,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};

}  // namespace

PreviewStage::PreviewStage(const SessionContextIntf* s,
                           string                    id,
                           vector<InEdge>            iports,
                           FlexData                  config)
  : TypedStage<PreviewStage>(s, std::move(id), std::move(iports),
                             std::move(config))
  , _libs(s->ffmpeg_libraries())
{
  _bitrate          = attr_int("bitrate");
  _input_normalized = attr_bool("input_normalized");
  _title            = attr_str("title");
  if (_bitrate < 64'000) { _bitrate = 64'000; }

  allocate_oports(spec().oports.size());   // sink: 0 oports
  _channel = std::make_shared<PreviewChannel>();

  session()->info(fmt(
      "preview('{}'): fMP4/WebSocket live preview, bitrate={}, "
      "cadence={}fps default", this->id(), _bitrate, cadence_fps()));
}

PreviewStage::~PreviewStage()
{
  teardown_();
}

const StageSpec&
PreviewStage::spec() const noexcept
{
  return kSpec;
}

string
PreviewStage::av_err_(int rc) const
{
  char buf[256];
  _libs->avutil().api.strerror(rc, buf, sizeof buf);
  return string(buf);
}

int
PreviewStage::write_cb_(void* opaque, const uint8_t* buf, int size)
{
  auto* self = static_cast<PreviewStage*>(opaque);
  if (size > 0) {
    self->_mux_buf.insert(self->_mux_buf.end(), buf, buf + size);
  }
  return size;
}

void
PreviewStage::resolve_roles_(RuntimeContext& ctx)
{
  if (_roles_resolved) { return; }
  _want_video = ctx.iport_connected(0);
  _want_audio = ctx.num_iports() >= 2 && ctx.iport_connected(1);
  _roles_resolved = true;
  session()->info(fmt(
      "preview('{}'): roles video={} audio={}",
      this->id(), _want_video, _want_audio));
}

std::chrono::steady_clock::duration
PreviewStage::cadence_period_() const
{
  const double num = _cadence_num > 0 ? _cadence_num : 25;
  const double den = _cadence_den > 0 ? _cadence_den : 1;
  int64_t ns = static_cast<int64_t>(llround(1e9 * den / num));
  if (ns < 1'000'000)     { ns = 1'000'000; }        // cap 1000 fps
  if (ns > 1'000'000'000) { ns = 1'000'000'000; }    // floor 1 fps
  return std::chrono::nanoseconds(ns);
}

void
PreviewStage::parse_extradata_()
{
  _codec_string = "avc1.42e01f";   // constrained-baseline 3.1 fallback
  const uint8_t* e = _enc ? _enc->extradata : nullptr;
  const int      n = _enc ? _enc->extradata_size : 0;
  if (!e || n < 4) { return; }

  if (e[0] == 1 && n >= 4) {
    // avcC: version, profile, compat, level.
    _codec_string = "avc1." + hex_byte_(e[1]) + hex_byte_(e[2])
                            + hex_byte_(e[3]);
  } else {
    // Annex-B extradata: find the SPS NAL (type 7) for profile/level.
    for (int i = 0; i + 4 < n; ++i) {
      const bool sc4 =
          e[i] == 0 && e[i + 1] == 0 && e[i + 2] == 0 && e[i + 3] == 1;
      const bool sc3 = e[i] == 0 && e[i + 1] == 0 && e[i + 2] == 1;
      const int hdr = sc4 ? i + 4 : (sc3 ? i + 3 : -1);
      if (hdr < 0) { continue; }
      if (hdr + 3 < n && (e[hdr] & 0x1f) == 7) {
        _codec_string = "avc1." + hex_byte_(e[hdr + 1])
                                + hex_byte_(e[hdr + 2])
                                + hex_byte_(e[hdr + 3]);
        break;
      }
    }
  }
}

void
PreviewStage::emit_config_()
{
  string j = "{";
  bool first = true;
  if (_video_cfg_ready) {
    j += "\"video\":{\"codec\":\"" + _codec_string
       + "\",\"width\":" + to_string(_out_w)
       + ",\"height\":" + to_string(_out_h) + "}";
    first = false;
  }
  if (_audio_cfg_ready) {
    if (!first) { j += ","; }
    j += "\"audio\":{\"sampleRate\":" + to_string(_audio_rate)
       + ",\"channels\":" + to_string(_audio_ch) + "}";
  }
  j += "}";
  _channel->set_config(std::move(j), _video_cfg_ready, _audio_cfg_ready,
                       _out_w, _out_h);
}

bool
PreviewStage::build_pipeline_(int W, int H)
{
  teardown_media_();   // free any prior encoder/muxer (re-init path)
  _out_w = W;
  _out_h = H;

  const AVCodec* codec =
      _libs->avcodec().api.find_encoder_by_name("h264_videotoolbox");
  if (!codec) {
    session()->error(fmt(
        "preview('{}'): h264_videotoolbox encoder not available",
        this->id()));
    return false;
  }
  _enc = _libs->avcodec().api.alloc_context3(codec);
  if (!_enc) { return false; }
  _enc->width        = W;
  _enc->height       = H;
  _enc->pix_fmt      = AV_PIX_FMT_YUV420P;
  _enc->time_base    = AVRational{1, 90000};
  _enc->framerate    = AVRational{_cadence_num, _cadence_den};
  _enc->bit_rate     = _bitrate;
  _enc->gop_size     = 600;   // keyframes are forced at fragment boundaries
  _enc->max_b_frames = 0;
  _enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

  AVDictionary* opts = nullptr;
  _libs->avutil().api.dict_set(&opts, "realtime",   "1", 0);
  _libs->avutil().api.dict_set(&opts, "allow_sw",   "1", 0);
  _libs->avutil().api.dict_set(&opts, "prio_speed", "1", 0);
  int rc = _libs->avcodec().api.open2(_enc, codec, &opts);
  _libs->avutil().api.dict_free(&opts);
  if (rc < 0) {
    session()->error(fmt(
        "preview('{}'): avcodec_open2: {}", this->id(), av_err_(rc)));
    return false;
  }
  _pkt = _libs->avcodec().api.packet_alloc();
  if (!_pkt) { return false; }
  parse_extradata_();

  // Output frame (black to start).
  _frame = _libs->avutil().api.frame_alloc();
  if (!_frame) { return false; }
  _frame->format = AV_PIX_FMT_YUV420P;
  _frame->width  = W;
  _frame->height = H;
  rc = _libs->avutil().api.frame_get_buffer(_frame, 32);
  if (rc < 0) {
    session()->error(fmt(
        "preview('{}'): frame_get_buffer: {}", this->id(), av_err_(rc)));
    return false;
  }
  fill_black_();

  // Fragmented-MP4 muxer writing through our AVIO capture callback.
  rc = _libs->avformat().api.alloc_output_context2(
      &_mux, nullptr, "mp4", nullptr);
  if (rc < 0 || !_mux) {
    session()->error(fmt(
        "preview('{}'): alloc_output_context2(mp4): {}",
        this->id(), av_err_(rc)));
    return false;
  }
  _avio_buf = static_cast<uint8_t*>(_libs->avutil().api.malloc(4096));
  if (!_avio_buf) { return false; }
  _avio = _libs->avformat().api.avio_alloc_context(
      _avio_buf, 4096, /*write*/ 1, this, nullptr, &write_cb_, nullptr);
  if (!_avio) { return false; }
  _mux->pb = _avio;
  _mux->flags |= AVFMT_FLAG_CUSTOM_IO;

  _vstream = _libs->avformat().api.new_stream(_mux, nullptr);
  if (!_vstream) { return false; }
  rc = _libs->avcodec().api.parameters_from_context(_vstream->codecpar, _enc);
  if (rc < 0) { return false; }
  _vstream->time_base = AVRational{1, 90000};

  AVDictionary* mopts = nullptr;
  _libs->avutil().api.dict_set(
      &mopts, "movflags",
      "empty_moov+default_base_moof+frag_custom", 0);
  _mux_buf.clear();
  rc = _libs->avformat().api.write_header(_mux, &mopts);
  _libs->avutil().api.dict_free(&mopts);
  if (rc < 0) {
    session()->error(fmt(
        "preview('{}'): write_header(mp4): {}", this->id(), av_err_(rc)));
    return false;
  }
  _libs->avformat().api.avio_flush(_mux->pb);
  _init.assign(_mux_buf.begin(), _mux_buf.end());   // the fMP4 init segment
  _mux_buf.clear();
  if (_channel) { _channel->set_init(_init.data(), _init.size()); }

  _first_encoded     = true;
  _force_next_key    = false;
  _frames_since_flush = 0;
  _enc_pts           = 0;
  _frame_dur_90k = std::max<std::int64_t>(1, llround(
      90000.0 * _cadence_den / _cadence_num));
  _frag_len = std::max(1, static_cast<int>(llround(
      static_cast<double>(_cadence_num) / _cadence_den * 0.5)));

  _video_cfg_ready = true;
  emit_config_();
  return true;
}

void
PreviewStage::teardown_media_()
{
  if (_mux) {
    _libs->avformat().api.free_context(_mux);
    _mux = nullptr;
    _vstream = nullptr;
  }
  if (_avio) {
    _libs->avutil().api.freep(&_avio->buffer);
    _libs->avformat().api.avio_context_free(&_avio);
    _avio = nullptr;
    _avio_buf = nullptr;
  }
  if (_pkt)   { _libs->avcodec().api.packet_free(&_pkt); }
  if (_enc)   { _libs->avcodec().api.free_context(&_enc); }
  if (_sws)   { _libs->swscale().api.free_context(_sws); _sws = nullptr; }
  if (_gbrp)  { _libs->avutil().api.frame_free(&_gbrp); }
  if (_frame) { _libs->avutil().api.frame_free(&_frame); }
  _mux_buf.clear();
}

void
PreviewStage::fill_black_()
{
  if (!_frame) { return; }
  const int H = _out_h;
  std::memset(_frame->data[0], 16,
              static_cast<size_t>(_frame->linesize[0]) * H);
  std::memset(_frame->data[1], 128,
              static_cast<size_t>(_frame->linesize[1]) * ((H + 1) / 2));
  std::memset(_frame->data[2], 128,
              static_cast<size_t>(_frame->linesize[2]) * ((H + 1) / 2));
}

bool
PreviewStage::ensure_sws_(int W, int H)
{
  if (_sws && _gbrp && _gbrp->width == W && _gbrp->height == H) {
    return true;
  }
  if (_sws)  { _libs->swscale().api.free_context(_sws); _sws = nullptr; }
  if (_gbrp) { _libs->avutil().api.frame_free(&_gbrp); }

  _sws = _libs->swscale().api.get_context(
      W, H, AV_PIX_FMT_GBRP, W, H, AV_PIX_FMT_YUV420P,
      SWS_BICUBIC, nullptr, nullptr, nullptr);
  if (!_sws) { return false; }
  _gbrp = _libs->avutil().api.frame_alloc();
  if (!_gbrp) { return false; }
  _gbrp->format = AV_PIX_FMT_GBRP;
  _gbrp->width  = W;
  _gbrp->height = H;
  if (_libs->avutil().api.frame_get_buffer(_gbrp, 32) < 0) {
    _libs->avutil().api.frame_free(&_gbrp);
    return false;
  }
  return true;
}

void
PreviewStage::convert_to_frame_(const TensorBeat& tb)
{
  const int    W = _out_w;
  const int    H = _out_h;
  const size_t expected = static_cast<size_t>(3) * H * W;
  if (!ensure_sws_(W, H)) { _fatal = true; return; }

  if (tb.dtype == TensorBeat::DType::U8) {
    const uint8_t*         src = nullptr;
    AlignedVector<uint8_t> tmp;
    if (tb.is_contiguous() && tb.byte_size() == expected) {
      src = tb.as_u8();
    } else {
      tmp = tb.materialize_contiguous();
      src = tmp.data();
    }
    for (int c = 0; c < 3; ++c) {
      const int      plane = kGbrpPlaneForChannel[c];
      uint8_t*       dst   = _gbrp->data[plane];
      const int      ls    = _gbrp->linesize[plane];
      const uint8_t* src_c = src + static_cast<size_t>(c) * H * W;
      for (int y = 0; y < H; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * ls,
                    src_c + static_cast<size_t>(y) * W,
                    static_cast<size_t>(W));
      }
    }
  } else if (tb.dtype == TensorBeat::DType::F32) {
    const float*         src = nullptr;
    AlignedVector<float> tmp;
    if (tb.is_contiguous()
        && tb.byte_size() == expected * sizeof(float)) {
      src = tb.as_f32();
    } else {
      tmp = tb.materialize_contiguous_as<float>();
      src = tmp.data();
    }
    const float scale = _input_normalized ? 255.0f : 1.0f;
    for (int c = 0; c < 3; ++c) {
      const int    plane = kGbrpPlaneForChannel[c];
      uint8_t*     dst   = _gbrp->data[plane];
      const int    ls    = _gbrp->linesize[plane];
      const float* src_c = src + static_cast<size_t>(c) * H * W;
      for (int y = 0; y < H; ++y) {
        uint8_t*     drow = dst + static_cast<size_t>(y) * ls;
        const float* srow = src_c + static_cast<size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
          drow[x] = clamp_byte_(srow[x] * scale);
        }
      }
    }
  } else {
    return;   // unsupported dtype: keep the previous frame
  }

  _libs->swscale().api.scale(
      _sws, _gbrp->data, _gbrp->linesize, 0, H,
      _frame->data, _frame->linesize);
}

void
PreviewStage::adopt_fps_(const TensorBeat& tb)
{
  if (!tb.sideband.is_object()) { return; }
  auto sb = tb.sideband.as_object();
  if (!sb.contains("fps_num")) { return; }
  const int num = static_cast<int>(sb.at("fps_num").as_uint(0));
  const int den = sb.contains("fps_den")
                      ? static_cast<int>(sb.at("fps_den").as_uint(0)) : 1;
  if (num > 0 && den > 0 && (num != _cadence_num || den != _cadence_den)) {
    _cadence_num = num;
    _cadence_den = den;
    _frag_len = std::max(1, static_cast<int>(llround(
        static_cast<double>(num) / den * 0.5)));
    _frame_dur_90k = std::max<std::int64_t>(1, llround(
        90000.0 * den / num));
  }
}

void
PreviewStage::handle_video_frame_(const TensorBeat& tb)
{
  if (_fatal) { return; }
  const int W = static_cast<int>(tb.shape[2]);
  const int H = static_cast<int>(tb.shape[1]);
  if (W <= 0 || H <= 0) { return; }

  adopt_fps_(tb);
  if (!_have_frame) {
    // First real frame: adopt its native resolution (re-init the fMP4
    // stream if it differs from the pre-input black size).
    if (W != _out_w || H != _out_h) {
      if (!build_pipeline_(W, H)) { _fatal = true; return; }
    }
    _have_frame = true;
  } else if (W != _out_w || H != _out_h) {
    return;   // resolution is fixed after adoption; drop mismatches
  }
  convert_to_frame_(tb);
}

void
PreviewStage::handle_audio_(const TensorBeat& tb)
{
  if (_fatal || tb.dtype != TensorBeat::DType::F32) { return; }
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
  if (in_ch <= 0 || n <= 0) { return; }

  int rate = 0;
  if (tb.sideband.is_object()) {
    auto sb = tb.sideband.as_object();
    if (sb.contains("sample_rate")) {
      rate = static_cast<int>(sb.at("sample_rate").as_int(0));
    }
  }
  if (rate <= 0) { rate = 48000; }

  if (!_audio_seen) {
    _audio_rate = rate;
    _audio_ch   = in_ch;
    _audio_seen = true;
    _audio_cfg_ready = true;
    emit_config_();
  }
  if (in_ch != _audio_ch) { return; }   // stable

  auto buf = tb.materialize_contiguous_as<float>();
  if (static_cast<int>(buf.size()) < in_ch * n) { return; }
  std::vector<const float*> planes(static_cast<size_t>(in_ch));
  for (int c = 0; c < in_ch; ++c) {
    planes[c] = buf.data() + static_cast<size_t>(c) * n;
  }
  _channel->push_audio(planes.data(), in_ch, n);
}

void
PreviewStage::encode_tick_()
{
  if (!_enc || !_mux || !_frame || !_pkt) { return; }

  const bool force_key = _first_encoded || _force_next_key;
  _first_encoded  = false;
  _force_next_key = false;
  // Constant-cadence timestamps: the stage emits exactly one frame per tick
  // at the cadence rate, so a uniform PTS + explicit per-frame duration
  // keeps the muxer's DTS strictly monotonic and self-describing. (Deriving
  // PTS from the wall clock jittered enough that libav's mp4 muxer logged
  // "pts has no value" / negative-duration warnings.)
  _frame->pts       = _enc_pts;
  _frame->duration  = _frame_dur_90k;
  _frame->pict_type = force_key ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_NONE;
  _enc_pts += _frame_dur_90k;

  int rc = _libs->avcodec().api.send_frame(_enc, _frame);
  if (rc < 0) {
    session()->warn(fmt(
        "preview('{}'): send_frame: {}", this->id(), av_err_(rc)));
    return;
  }
  while (true) {
    rc = _libs->avcodec().api.receive_packet(_enc, _pkt);
    if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { break; }
    if (rc < 0) { break; }
    _pkt->stream_index = _vstream->index;
    if (_pkt->duration <= 0) { _pkt->duration = _frame_dur_90k; }
    _libs->avcodec().api.packet_rescale_ts(
        _pkt, AVRational{1, 90000}, _vstream->time_base);
    _libs->avformat().api.write_frame(_mux, _pkt);
    _libs->avcodec().api.packet_unref(_pkt);
  }

  if (++_frames_since_flush >= _frag_len) {
    // frag_custom: a NULL packet to av_write_frame flushes the pending
    // fragment (moof + mdat) through the AVIO capture callback.
    _libs->avformat().api.write_frame(_mux, nullptr);
    _libs->avformat().api.avio_flush(_mux->pb);
    if (!_mux_buf.empty() && _channel) {
      _channel->push_fragment(_mux_buf.data(), _mux_buf.size());
      _mux_buf.clear();
    }
    _frames_since_flush = 0;
    _force_next_key = true;   // next fragment starts on a keyframe
  }
}

Job
PreviewStage::process(RuntimeContext& ctx)
{
  resolve_roles_(ctx);

  if (!_clock_started) {
    if (!build_pipeline_(kDefaultW, kDefaultH)) { _fatal = true; }
    _clock_started = true;
    _next_tick = std::chrono::steady_clock::now();
  }

  // Drain inputs without blocking: the latest video frame (sampling), and
  // every audio chunk. Only the newest video frame is kept -- an unread
  // backlog is still consumed so the producer isn't back-pressured.
  if (!_fatal && _want_video) {
    const uint32_t avail = ctx.backlog(0);
    for (uint32_t i = 0; i < avail; ++i) {
      auto beat = co_await ctx.read(0);
      if (!beat) { break; }
      const auto* tb = dynamic_cast<const TensorBeatPayload*>(beat.get());
      if (tb && tb->shape.size() == 3 && tb->shape[0] == 3) {
        ++_frames_in;
        handle_video_frame_(*tb);
      }
    }
  }
  if (!_fatal && _want_audio) {
    const uint32_t avail = ctx.backlog(1);
    for (uint32_t i = 0; i < avail; ++i) {
      auto beat = co_await ctx.read(1);
      if (!beat) { break; }
      const auto* tb = dynamic_cast<const TensorBeatPayload*>(beat.get());
      if (tb && (tb->shape.size() == 1 || tb->shape.size() == 2)) {
        handle_audio_(*tb);
      }
    }
  }

  if (_fatal || ctx.stop_requested()) {
    teardown_();
    ctx.signal_done();
    co_return;
  }

  encode_tick_();

  // Self-clock to the next cadence tick.
  ThreadPool* pool = session() ? session()->thread_pool() : nullptr;
  const auto now = std::chrono::steady_clock::now();
  _next_tick += cadence_period_();
  auto dt = _next_tick - now;
  if (dt < std::chrono::steady_clock::duration::zero()) {
    dt = std::chrono::steady_clock::duration::zero();
    _next_tick = now;    // fell behind: reset the phase, don't burst-catch-up
  }
  if (pool) {
    co_await TimerAwaiter{pool, dt};
  } else {
    std::this_thread::sleep_for(dt);
  }
  co_return;
}

Job
PreviewStage::drain(RuntimeContext&)
{
  teardown_();
  co_return;
}

void
PreviewStage::teardown_()
{
  if (_torn) { return; }
  _torn = true;

  if (_enc && _mux && _pkt) {
    // Flush the encoder, then the final fragment + trailer, so nothing
    // buffered is lost before the stream ends.
    _libs->avcodec().api.send_frame(_enc, nullptr);
    while (true) {
      int rc = _libs->avcodec().api.receive_packet(_enc, _pkt);
      if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) { break; }
      if (rc < 0) { break; }
      _pkt->stream_index = _vstream->index;
      _libs->avcodec().api.packet_rescale_ts(
          _pkt, AVRational{1, 90000}, _vstream->time_base);
      _libs->avformat().api.write_frame(_mux, _pkt);
      _libs->avcodec().api.packet_unref(_pkt);
    }
    _libs->avformat().api.write_frame(_mux, nullptr);
    _libs->avformat().api.avio_flush(_mux->pb);
    if (!_mux_buf.empty() && _channel) {
      _channel->push_fragment(_mux_buf.data(), _mux_buf.size());
      _mux_buf.clear();
    }
    // Finalize the muxer for a clean shutdown, but DON'T forward the
    // trailer bytes (an mfra index) -- that is not a media fragment MSE
    // should append, so the last thing a viewer gets stays a moof+mdat.
    _libs->avformat().api.write_trailer(_mux);
    _libs->avformat().api.avio_flush(_mux->pb);
    _mux_buf.clear();
  }

  if (_channel) { _channel->close(); }
  teardown_media_();
}

VPIPE_REGISTER_STAGE(PreviewStage)
VPIPE_REGISTER_SPEC(PreviewStage, kSpec)

}
