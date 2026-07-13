#include "stages/audio-video/rtsp-capture-stage.h"
#include "common/beat-payload-intf.h"
#include "common/credential-cipher.h"
#include "common/credential-store.h"
#include "common/encoded-segment.h"
#include "common/ffmpeg-libraries.h"
#include "common/ffmpeg-log-tap.h"
#include "common/flex-data.h"
#include "common/host-identity.h"
#include "common/job.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/onvif-refresh.h"
#include "common/oport-policy.h"
#include "common/segment-writer.h"
#include "common/thread-pool.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"
#include "stages/trigger-beat.h"

#include <atomic>
#include <chrono>
#include <coroutine>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <utility>
#include <vector>
#include <algorithm>

namespace vpipe {

namespace {

constexpr const char* kHexDigits = "0123456789ABCDEF";

// RFC 3986 unreserved set (plus '~'); everything else gets %-encoded.
bool
url_unreserved(unsigned char c)
{
  return (c >= 'A' && c <= 'Z')
      || (c >= 'a' && c <= 'z')
      || (c >= '0' && c <= '9')
      || c == '-' || c == '_' || c == '.' || c == '~';
}

std::string
url_escape(std::string_view in)
{
  std::string out;
  out.reserve(in.size());
  for (unsigned char c : in) {
    if (url_unreserved(c)) {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('%');
      out.push_back(kHexDigits[(c >> 4) & 0xF]);
      out.push_back(kHexDigits[c & 0xF]);
    }
  }
  return out;
}

// Splice user:pass@ into an existing rtsp://host[:port]/path URL.
std::string
embed_credentials(std::string_view url,
                  std::string_view user,
                  std::string_view password)
{
  constexpr std::string_view kP = "rtsp://";
  if (url.size() < kP.size() || url.substr(0, kP.size()) != kP) {
    return std::string(url);
  }
  std::string out;
  out.reserve(url.size() + user.size() + password.size() + 4);
  out.append(kP);
  out.append(url_escape(user));
  out.push_back(':');
  out.append(url_escape(password));
  out.push_back('@');
  out.append(url.substr(kP.size()));
  return out;
}

void
stop_aware_sleep(RuntimeContext&            ctx,
                 std::chrono::milliseconds  total)
{
  using namespace std::chrono;
  auto deadline = steady_clock::now() + total;
  constexpr auto kChunk = milliseconds(50);
  while (true) {
    if (ctx.stop_requested()) {
      return;
    }
    auto now = steady_clock::now();
    if (now >= deadline) {
      return;
    }
    auto remaining = deadline - now;
    std::this_thread::sleep_for(remaining < kChunk ? remaining
                                                   : kChunk);
  }
}

// Opaque passed to AVIOInterruptCB. FFmpeg's blocking I/O code polls
// the callback periodically and aborts when it returns non-zero,
// which lets a session stop or the watchdog punch out of recv()/read()
// promptly instead of waiting for stimeout to elapse. `ctx` is the
// owning stage's RuntimeContext (stable for the whole run), so the
// callback reads the stop flag straight off it -- no relay thread is
// needed to mirror it into an atomic. `force_reconnect` is owned by
// capture_loop_ and stays valid for every AVFormatContext built in the
// connect loop.
struct InterruptCtx {
  RuntimeContext*    ctx             = nullptr;
  std::atomic<bool>* force_reconnect = nullptr;
};

int
interrupt_cb(void* opaque) noexcept
{
  auto* ic = static_cast<InterruptCtx*>(opaque);
  if (!ic) { return 0; }
  if (ic->ctx && ic->ctx->stop_requested()) {
    return 1;
  }
  if (ic->force_reconnect
      && ic->force_reconnect->load(std::memory_order_acquire)) {
    return 1;
  }
  return 0;
}

bool
ensure_directory_exists(std::string_view path,
                        const SessionContextIntf* session)
{
  std::string p(path);
  struct stat st {};
  if (::stat(p.c_str(), &st) == 0) {
    return (st.st_mode & S_IFDIR) != 0;
  }
  if (::mkdir(p.c_str(), 0755) == 0) {
    return true;
  }
  session->warn(fmt(
      "rtsp-capture: output_dir '{}' does not exist and could not "
      "be created", path));
  return false;
}

// 8-byte big-endian encoding of microseconds-since-epoch. Used as
// LMDB key so cursors iterate in time order regardless of platform
// endianness.
std::string
be64_us_key(std::chrono::system_clock::time_point t)
{
  uint64_t us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          t.time_since_epoch()).count());
  std::string out;
  out.resize(8);
  for (int i = 0; i < 8; ++i) {
    out[7 - i] = static_cast<char>(us & 0xff);
    us >>= 8;
  }
  return out;
}

bool
rtp_predicate(int level, std::string_view tag, std::string_view msg)
{
  if (level > AV_LOG_ERROR) {
    // Warnings can flap during healthy operation; bound the trigger
    // to ERROR-or-worse so the reconnect cycle stays calm.
    return false;
  }
  // Match by tag for the network-protocol classes that actually
  // surface a stream-level failure.
  bool tag_match = (tag == "rtsp")
                || (tag == "rtp")
                || (tag == "udp");
  if (!tag_match) {
    // Some FFmpeg builds emit RTP errors under the demuxer class
    // ("mov" / "matroska" type names). Fall back to substring.
    bool body_match =
        msg.find("RTP") != std::string_view::npos
        || msg.find("max delay reached") != std::string_view::npos
        || msg.find("missed") != std::string_view::npos;
    if (!body_match) {
      return false;
    }
  }
  // Even within rtsp/rtp/udp tags, harmless oddities exist. Require
  // either a specific phrase or a level of FATAL/PANIC.
  if (level <= AV_LOG_FATAL) {
    return true;
  }
  return msg.find("RTP") != std::string_view::npos
      || msg.find("missed") != std::string_view::npos
      || msg.find("timeout") != std::string_view::npos
      || msg.find("max delay reached") != std::string_view::npos
      || msg.find("decode error") != std::string_view::npos;
}

}

}

namespace vpipe {

RtspCaptureStage::RtspCaptureStage(const SessionContextIntf* s,
                                   std::string               id,
                                   std::vector<InEdge>       iports,
                                   FlexData                  config)
  : TypedStage<RtspCaptureStage>(s, std::move(id), std::move(iports),
                                 std::move(config))
{
  // Validation is deferred to launch (see Stage::fail_config). Route
  // parsing through an empty object when the config isn't one, so the
  // stage still constructs (with defaults) for any input.
  // Attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  const FlexData& cfg = this->config();
  if (!cfg.is_object()) {
    fail_config(fmt(
        "RtspCaptureStage('{}'): config must be an object",
        this->id()));
  }

  _camera_name = attr_str("camera_name");
  if (_camera_name.empty()) {
    fail_config(fmt(
        "RtspCaptureStage('{}'): config.camera_name is required",
        this->id()));
  }

  _output_dir = attr_path("output_dir", /*for_write=*/true);
  if (_output_dir.empty()) {
    fail_config(fmt(
        "RtspCaptureStage('{}'): config.output_dir is required",
        this->id()));
  }

  uint64_t seg = attr_uint("segment_seconds");
  if (seg == 0) { seg = 60; }
  _segment_seconds = std::chrono::seconds(seg);

  _cameras_db       = attr_str("cameras_db");
  _videos_db_suffix = attr_str("videos_db_suffix");
  _rtsp_transport   = attr_str("rtsp_transport");
  _stimeout_us = static_cast<unsigned>(attr_uint("stimeout_us"));
  _connect_timeout_ms =
      static_cast<unsigned>(attr_uint("connect_timeout_ms"));
  _reconnect_delay_ms =
      static_cast<unsigned>(attr_uint("reconnect_delay_ms"));
  _rediscover_timeout_ms =
      static_cast<unsigned>(attr_uint("rediscover_timeout_ms"));
  _video_bitrate = attr_uint("transcode_video_bitrate");
  _audio_bitrate = attr_uint("transcode_audio_bitrate");
  _oport_depth = static_cast<unsigned>(attr_uint("oport_depth"));
  if (_oport_depth == 0) { _oport_depth = 300; }

  // 2 oports: 0 = per-AU h.264 video, 1 = per-packet AAC audio.
  // DropOldest so a slow downstream consumer cannot stall the live
  // capture read loop -- losing the oldest queued AU is better than
  // missing RTP packets.
  allocate_oports(spec().oports.size());
  set_oport_policy(0, {_oport_depth, OverrunPolicy::DropOldest});
  set_oport_policy(1, {_oport_depth, OverrunPolicy::DropOldest});
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "camera_name", .type = ConfigType::String, .required = true,
   .doc = "camera key in the cameras sub-db", .suggest_db = "cameras"},
  {.key = "output_dir", .type = ConfigType::String, .required = true,
   .doc = "directory for finalized MP4 segments",
   .is_path = true, .path_write = true, .path_kind = "dir"},
  {.key = "segment_seconds", .type = ConfigType::Uint,
   .doc = "target segment length in seconds", .def_uint = 60},
  {.key = "cameras_db", .type = ConfigType::String,
   .doc = "LMDB sub-db holding camera records", .def_str = "cameras"},
  {.key = "videos_db_suffix", .type = ConfigType::String,
   .doc = "suffix for the per-camera videos sub-db",
   .def_str = "-videos"},
  {.key = "rtsp_transport", .type = ConfigType::String,
   .doc = "FFmpeg rtsp_transport (tcp/udp)", .def_str = "tcp"},
  {.key = "stimeout_us", .type = ConfigType::Uint,
   .doc = "FFmpeg socket I/O timeout (us)", .def_uint = 5'000'000},
  {.key = "connect_timeout_ms", .type = ConfigType::Uint,
   .doc = "connection timeout (ms)", .def_uint = 10000},
  {.key = "reconnect_delay_ms", .type = ConfigType::Uint,
   .doc = "backoff before reconnect (ms)", .def_uint = 2000},
  {.key = "rediscover_timeout_ms", .type = ConfigType::Uint,
   .doc = "ONVIF rediscovery timeout (ms)", .def_uint = 4000},
  {.key = "transcode_video_bitrate", .type = ConfigType::Uint,
   .doc = "H.264 encode bitrate when transcoding",
   .def_uint = 4'000'000},
  {.key = "transcode_audio_bitrate", .type = ConfigType::Uint,
   .doc = "AAC encode bitrate when transcoding", .def_uint = 128000},
  {.key = "oport_depth", .type = ConfigType::Uint,
   .doc = "output ring depth (DropOldest)", .def_uint = 300},
};
// Optional watchdog tick on iport 0 (clock 0); both encoded-segment
// oports live in the capture clock domain (clock 1).
const PortSpec kIports[] = {
  {.name = "watchdog", .doc = "optional tick stream (e.g. chrono); a "
                              "silent camera between ticks forces reconnect",
   .type = &typeid(TriggerPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "video", .doc = "EncodedSegment per H.264 access unit (AVCC)",
   .type = &typeid(EncodedSegmentPayload),
   .tags = "video-encoder-segments", .clock_group = 1},
  {.name = "audio", .doc = "EncodedSegment per AAC packet (ADTS-free)",
   .type = &typeid(EncodedSegmentPayload),
   .tags = "audio-encoder-segments", .clock_group = 1},
};
const StageSpec kSpec = {
  .type_name = "rtsp-capture",
  .doc       = "Long-running source: connects to a registered camera's "
               "RTSP stream (FFmpeg), writes IDR-aligned MP4 segments, "
               "and emits per-AU H.264 + AAC EncodedSegments. Apple-only.",
  .display_name = "RTSP Capture",
  .category  = StageCategory::Network,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
RtspCaptureStage::spec() const noexcept
{
  return kSpec;
}

namespace {

// Small planar-float FIFO sitting between swresample's output and the
// AAC encoder's input. The encoder requires exactly frame_size samples
// per send_frame call (the native FFmpeg AAC encoder doesn't have
// AV_CODEC_CAP_VARIABLE_FRAME_SIZE), but decoded RTSP audio frames are
// almost always a different size -- e.g. G.711 frames are 160 samples,
// AAC is 1024. The FIFO accumulates samples until a full encoder
// frame is available, then hands them out.
struct AudioPlanarFloatFifo {
  int                              channels = 0;
  int                              frame_size = 0;
  std::vector<std::vector<float>>  planes;

  void reset(int nb_channels, int nb_samples_per_frame)
  {
    channels   = nb_channels;
    frame_size = nb_samples_per_frame;
    planes.assign(static_cast<size_t>(channels), {});
  }

  int filled() const
  {
    return planes.empty()
         ? 0
         : static_cast<int>(planes.front().size());
  }

  // Append nb_samples planar floats from src[0..channels-1]. Caller
  // must guarantee src has `channels` valid planes.
  void push_planar(const uint8_t* const* src, int nb_samples)
  {
    for (int ch = 0; ch < channels; ++ch) {
      auto* f = reinterpret_cast<const float*>(src[ch]);
      planes[static_cast<size_t>(ch)]
          .insert(planes[static_cast<size_t>(ch)].end(),
                  f, f + nb_samples);
    }
  }

  // Drain one full frame_size-sample frame into dst[0..channels-1].
  // Returns true on success, false if the FIFO doesn't hold enough.
  bool pull_frame(uint8_t* const* dst)
  {
    if (filled() < frame_size) {
      return false;
    }
    for (int ch = 0; ch < channels; ++ch) {
      auto& src = planes[static_cast<size_t>(ch)];
      std::memcpy(dst[ch], src.data(),
                  static_cast<size_t>(frame_size) * sizeof(float));
      src.erase(src.begin(), src.begin() + frame_size);
    }
    return true;
  }
};

// Encapsulates the FFmpeg state for a single open input. Cleaned up
// in the dtor so the connection loop can rely on RAII unwinding.
struct CaptureSession {
  const FFmpegLibraries*    libs    = nullptr;
  const SessionContextIntf* session = nullptr;

  AVFormatContext* ictx = nullptr;
  AVDictionary*    opts = nullptr;

  int v_in_idx = -1;
  int a_in_idx = -1;

  // Transcode-path codec contexts. Null in stream-copy mode.
  AVCodecContext* v_dec = nullptr;
  AVCodecContext* v_enc = nullptr;
  AVCodecContext* a_dec = nullptr;
  AVCodecContext* a_enc = nullptr;

  // Scratch reusable frames + packets.
  AVPacket* in_pkt   = nullptr;
  AVPacket* out_pkt  = nullptr;
  AVFrame*  v_frame  = nullptr;
  AVFrame*  a_frame  = nullptr;

  // Audio transcode helpers (engaged only when the decoder's PCM
  // output is incompatible with the AAC encoder's expected input).
  SwrContext* swr             = nullptr;
  AVFrame*    a_swr_out_frame = nullptr;  // scratch swr output
  AVFrame*    a_enc_in_frame  = nullptr;  // exactly enc->frame_size
  AudioPlanarFloatFifo a_fifo{};
  int64_t     a_next_pts      = 0;        // running PTS in enc time_base

  bool stream_copy_v = true;
  bool stream_copy_a = true;

  // Codec-parameter sources for SegmentWriter output streams:
  // either the input AVStream's codecpar (stream-copy) or the
  // encoder context's codecpar (transcode).
  AVCodecParameters* v_out_par = nullptr;
  AVCodecParameters* a_out_par = nullptr;
  bool               v_out_par_owned = false;
  bool               a_out_par_owned = false;

  ~CaptureSession()
  {
    if (!libs) { return; }
    const auto& fmt_api  = libs->avformat().api;
    const auto& cdc_api  = libs->avcodec().api;
    const auto& util_api = libs->avutil().api;
    const auto& swr_api  = libs->swresample().api;

    if (swr)             { swr_api.free(&swr); }
    if (a_swr_out_frame) { util_api.frame_free(&a_swr_out_frame); }
    if (a_enc_in_frame)  { util_api.frame_free(&a_enc_in_frame); }
    if (v_dec) { cdc_api.free_context(&v_dec); }
    if (v_enc) { cdc_api.free_context(&v_enc); }
    if (a_dec) { cdc_api.free_context(&a_dec); }
    if (a_enc) { cdc_api.free_context(&a_enc); }
    if (in_pkt)  { cdc_api.packet_free(&in_pkt); }
    if (out_pkt) { cdc_api.packet_free(&out_pkt); }
    if (v_frame) { util_api.frame_free(&v_frame); }
    if (a_frame) { util_api.frame_free(&a_frame); }
    if (v_out_par && v_out_par_owned) {
      cdc_api.parameters_free(&v_out_par);
    }
    if (a_out_par && a_out_par_owned) {
      cdc_api.parameters_free(&a_out_par);
    }
    if (opts) { util_api.dict_free(&opts); }
    if (ictx) { fmt_api.close_input(&ictx); }
  }
};

// Open RTSP input + decide stream-copy vs transcode + open
// transcoders. Returns true on success, leaving `cs` in a state
// ready for the packet loop. `ic` (if non-null) is wired into the
// AVFormatContext's interrupt_callback so blocking FFmpeg I/O
// honours stop_requested / force_reconnect; its storage must
// outlive the returned cs.ictx (typically a local in process()).
bool
prepare_capture(CaptureSession&     cs,
                std::string_view    url,
                std::string_view    rtsp_transport,
                unsigned            stimeout_us,
                uint64_t            video_bitrate,
                uint64_t            audio_bitrate,
                std::chrono::seconds segment_seconds,
                InterruptCtx*       ic)
{
  const auto& fmt_api  = cs.libs->avformat().api;
  const auto& cdc_api  = cs.libs->avcodec().api;
  const auto& util_api = cs.libs->avutil().api;

  // Pre-allocate the format context so we can install the
  // interrupt_callback BEFORE open_input -- otherwise a hung
  // initial connect can't be aborted by stop_requested.
  cs.ictx = fmt_api.alloc_context();
  if (!cs.ictx) {
    cs.session->warn(fmt(
        "rtsp-capture: avformat_alloc_context failed"));
    return false;
  }
  if (ic) {
    cs.ictx->interrupt_callback.callback = &interrupt_cb;
    cs.ictx->interrupt_callback.opaque   = ic;
  }

  util_api.dict_set(&cs.opts, "rtsp_transport",
                    std::string(rtsp_transport).c_str(), 0);
  {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%u", stimeout_us);
    util_api.dict_set(&cs.opts, "stimeout", buf, 0);
  }

  std::string url_s(url);
  int rc = fmt_api.open_input(&cs.ictx, url_s.c_str(),
                              nullptr, &cs.opts);
  if (rc < 0) {
    cs.session->warn(fmt(
        "rtsp-capture: avformat_open_input failed ({})", rc));
    return false;
  }
  rc = fmt_api.find_stream_info(cs.ictx, nullptr);
  if (rc < 0) {
    cs.session->warn(fmt(
        "rtsp-capture: find_stream_info failed ({})", rc));
    return false;
  }

  for (unsigned i = 0; i < cs.ictx->nb_streams; ++i) {
    auto* st = cs.ictx->streams[i];
    if (cs.v_in_idx < 0
        && st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      cs.v_in_idx = static_cast<int>(i);
    } else if (cs.a_in_idx < 0
        && st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      cs.a_in_idx = static_cast<int>(i);
    }
  }
  if (cs.v_in_idx < 0) {
    cs.session->warn(fmt(
        "rtsp-capture: no video stream in input"));
    return false;
  }

  cs.stream_copy_v =
      (cs.ictx->streams[cs.v_in_idx]->codecpar->codec_id
       == AV_CODEC_ID_H264);
  cs.stream_copy_a =
      (cs.a_in_idx < 0)
      || (cs.ictx->streams[cs.a_in_idx]->codecpar->codec_id
          == AV_CODEC_ID_AAC);

  // -- Video transcode setup (if not stream-copy) ---
  if (!cs.stream_copy_v) {
    auto* v_st = cs.ictx->streams[cs.v_in_idx];
    const AVCodec* dec = cdc_api.find_decoder(v_st->codecpar->codec_id);
    const AVCodec* enc = cdc_api.find_encoder(AV_CODEC_ID_H264);
    if (!dec || !enc) {
      cs.session->warn(fmt(
          "rtsp-capture: missing H.264 decoder/encoder for "
          "transcoding"));
      return false;
    }
    cs.v_dec = cdc_api.alloc_context3(dec);
    cs.v_enc = cdc_api.alloc_context3(enc);
    if (!cs.v_dec || !cs.v_enc) { return false; }
    cdc_api.parameters_to_context(cs.v_dec, v_st->codecpar);
    cs.v_dec->time_base = v_st->time_base;
    if (cdc_api.open2(cs.v_dec, dec, nullptr) < 0) {
      cs.session->warn(fmt(
          "rtsp-capture: video decoder open failed"));
      return false;
    }
    cs.v_enc->width      = cs.v_dec->width;
    cs.v_enc->height     = cs.v_dec->height;
    cs.v_enc->pix_fmt    = AV_PIX_FMT_YUV420P;
    cs.v_enc->bit_rate   = static_cast<int64_t>(video_bitrate);
    cs.v_enc->time_base  = AVRational{1, 90000};
    cs.v_enc->framerate  = cs.v_dec->framerate.num
                              ? cs.v_dec->framerate
                              : AVRational{30, 1};
    // GOP = roughly target seconds * fps -- still gated by
    // pict_type forcing at exact rollover time.
    int fps_num = cs.v_enc->framerate.num
                  ? cs.v_enc->framerate.num : 30;
    int fps_den = cs.v_enc->framerate.den
                  ? cs.v_enc->framerate.den : 1;
    cs.v_enc->gop_size =
        std::max(1, static_cast<int>(
                       segment_seconds.count() * fps_num
                       / fps_den));
    if (cs.ictx->oformat
        || (cs.v_enc->codec
            && cs.v_enc->codec->id == AV_CODEC_ID_H264))
    {
      cs.v_enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    if (cdc_api.open2(cs.v_enc, enc, nullptr) < 0) {
      cs.session->warn(fmt(
          "rtsp-capture: video encoder open failed"));
      return false;
    }
    cs.v_out_par = cdc_api.parameters_alloc();
    if (!cs.v_out_par) { return false; }
    cs.v_out_par_owned = true;
    cdc_api.parameters_from_context(cs.v_out_par, cs.v_enc);
  } else {
    cs.v_out_par = cs.ictx->streams[cs.v_in_idx]->codecpar;
  }

  // -- Audio transcode setup (if any audio stream) ---
  if (cs.a_in_idx >= 0 && !cs.stream_copy_a) {
    auto* a_st = cs.ictx->streams[cs.a_in_idx];
    const AVCodec* dec = cdc_api.find_decoder(a_st->codecpar->codec_id);
    const AVCodec* enc = cdc_api.find_encoder(AV_CODEC_ID_AAC);
    if (!dec || !enc) {
      cs.session->warn(fmt(
          "rtsp-capture: missing AAC encoder for audio transcode"));
      return false;
    }
    cs.a_dec = cdc_api.alloc_context3(dec);
    cs.a_enc = cdc_api.alloc_context3(enc);
    if (!cs.a_dec || !cs.a_enc) { return false; }
    cdc_api.parameters_to_context(cs.a_dec, a_st->codecpar);
    if (cdc_api.open2(cs.a_dec, dec, nullptr) < 0) {
      cs.session->warn(fmt(
          "rtsp-capture: audio decoder open failed"));
      return false;
    }
    cs.a_enc->sample_rate = cs.a_dec->sample_rate;
    cs.a_enc->ch_layout   = cs.a_dec->ch_layout;
    cs.a_enc->sample_fmt  = AV_SAMPLE_FMT_FLTP;
    cs.a_enc->bit_rate    = static_cast<int64_t>(audio_bitrate);
    cs.a_enc->time_base   = AVRational{1, cs.a_enc->sample_rate};
    if (cdc_api.open2(cs.a_enc, enc, nullptr) < 0) {
      cs.session->warn(fmt(
          "rtsp-capture: audio encoder open failed -- the input "
          "ch_layout / sample_rate may not be supported by AAC"));
      return false;
    }

    // -- Decide whether swresample is required. The AAC encoder
    // always wants planar float at its configured rate/layout. If
    // any of those differ from what the decoder emits we route the
    // PCM through swr and log the reason so it's obvious in field
    // logs why the resampler is running.
    bool fmt_mismatch  =
        cs.a_dec->sample_fmt != cs.a_enc->sample_fmt;
    bool rate_mismatch =
        cs.a_dec->sample_rate != cs.a_enc->sample_rate;
    bool ch_mismatch   =
        cs.a_dec->ch_layout.nb_channels
            != cs.a_enc->ch_layout.nb_channels;

    if (fmt_mismatch || rate_mismatch || ch_mismatch) {
      std::string reasons;
      if (fmt_mismatch) {
        const char* in_n =
            util_api.get_sample_fmt_name(cs.a_dec->sample_fmt);
        const char* out_n =
            util_api.get_sample_fmt_name(cs.a_enc->sample_fmt);
        reasons += std::format("sample_fmt {}->{}",
                               in_n  ? in_n  : "?",
                               out_n ? out_n : "?");
      }
      if (rate_mismatch) {
        if (!reasons.empty()) { reasons += ", "; }
        reasons += std::format("sample_rate {}->{}",
                               cs.a_dec->sample_rate,
                               cs.a_enc->sample_rate);
      }
      if (ch_mismatch) {
        if (!reasons.empty()) { reasons += ", "; }
        reasons += std::format("channels {}->{}",
                               cs.a_dec->ch_layout.nb_channels,
                               cs.a_enc->ch_layout.nb_channels);
      }
      cs.session->info(fmt(
          "rtsp-capture: enabling swresample for audio ({})",
          reasons));

      const auto& swr_api = cs.libs->swresample().api;
      int rc = swr_api.alloc_set_opts2(
          &cs.swr,
          &cs.a_enc->ch_layout, cs.a_enc->sample_fmt,
          cs.a_enc->sample_rate,
          &cs.a_dec->ch_layout, cs.a_dec->sample_fmt,
          cs.a_dec->sample_rate,
          0, nullptr);
      if (rc < 0 || !cs.swr) {
        cs.session->warn(fmt(
            "rtsp-capture: swr_alloc_set_opts2 failed ({})", rc));
        return false;
      }
      rc = swr_api.init(cs.swr);
      if (rc < 0) {
        cs.session->warn(fmt(
            "rtsp-capture: swr_init failed ({})", rc));
        return false;
      }
    }

    // Allocate scratch frames the audio transcode loop reuses.
    // a_enc_in_frame is sized to exactly the encoder's frame_size
    // (the FIFO chunks input to that size); a_swr_out_frame is a
    // generous scratch buffer for one swr_convert call.
    int enc_frame_size =
        cs.a_enc->frame_size > 0 ? cs.a_enc->frame_size : 1024;
    cs.a_enc_in_frame = util_api.frame_alloc();
    if (!cs.a_enc_in_frame) { return false; }
    cs.a_enc_in_frame->nb_samples  = enc_frame_size;
    cs.a_enc_in_frame->format      = cs.a_enc->sample_fmt;
    cs.a_enc_in_frame->sample_rate = cs.a_enc->sample_rate;
    cs.a_enc_in_frame->ch_layout   = cs.a_enc->ch_layout;
    if (util_api.frame_get_buffer(cs.a_enc_in_frame, 0) < 0) {
      cs.session->warn(fmt(
          "rtsp-capture: frame_get_buffer for AAC input failed"));
      return false;
    }

    if (cs.swr) {
      constexpr int kSwrScratchSamples = 8192;
      cs.a_swr_out_frame = util_api.frame_alloc();
      if (!cs.a_swr_out_frame) { return false; }
      cs.a_swr_out_frame->nb_samples  = kSwrScratchSamples;
      cs.a_swr_out_frame->format      = cs.a_enc->sample_fmt;
      cs.a_swr_out_frame->sample_rate = cs.a_enc->sample_rate;
      cs.a_swr_out_frame->ch_layout   = cs.a_enc->ch_layout;
      if (util_api.frame_get_buffer(cs.a_swr_out_frame, 0) < 0) {
        cs.session->warn(fmt(
            "rtsp-capture: frame_get_buffer for swr scratch failed"));
        return false;
      }
    }

    cs.a_fifo.reset(cs.a_enc->ch_layout.nb_channels, enc_frame_size);
    cs.a_next_pts = 0;

    cs.a_out_par = cdc_api.parameters_alloc();
    if (!cs.a_out_par) { return false; }
    cs.a_out_par_owned = true;
    cdc_api.parameters_from_context(cs.a_out_par, cs.a_enc);
  } else if (cs.a_in_idx >= 0) {
    cs.a_out_par = cs.ictx->streams[cs.a_in_idx]->codecpar;
  }

  cs.in_pkt  = cdc_api.packet_alloc();
  cs.out_pkt = cdc_api.packet_alloc();
  if (cs.v_dec || cs.a_dec) {
    cs.v_frame = util_api.frame_alloc();
    cs.a_frame = util_api.frame_alloc();
  }
  return cs.in_pkt && cs.out_pkt
         && (!cs.v_dec || cs.v_frame)
         && (!cs.a_dec || cs.a_frame);
}

void
index_segment(LmdbEnv&            env,
              std::string_view    videos_db,
              const std::string&  camera_name,
              const SegmentInfo&  seg,
              const SessionContextIntf* session)
{
  if (seg.path.empty()) {
    return;
  }
  auto v = FlexData::make_object();
  v.as_object().insert_or_assign("path",
      FlexData::make_string(seg.path));
  v.as_object().insert_or_assign("camera_name",
      FlexData::make_string(camera_name));
  uint64_t start_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          seg.start_utc.time_since_epoch()).count());
  uint64_t end_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          seg.end_utc.time_since_epoch()).count());
  v.as_object().insert_or_assign("start_utc_us",
      FlexData::make_uint(start_us));
  v.as_object().insert_or_assign("end_utc_us",
      FlexData::make_uint(end_us));
  v.as_object().insert_or_assign("duration_us",
      FlexData::make_uint(
          static_cast<uint64_t>(seg.duration_us)));
  v.as_object().insert_or_assign("container",
      FlexData::make_string("mp4"));
  v.as_object().insert_or_assign("video_codec",
      FlexData::make_string("h264"));
  v.as_object().insert_or_assign("audio_codec",
      FlexData::make_string(seg.has_audio ? "aac" : ""));
  v.as_object().insert_or_assign("stream_copy_v",
      FlexData::make_bool(seg.stream_copy_video));
  v.as_object().insert_or_assign("stream_copy_a",
      FlexData::make_bool(seg.stream_copy_audio));

  std::string key   = be64_us_key(seg.start_utc);
  std::string bytes = v.to_binary();
  try {
    LmdbDb  db(env, videos_db);
    LmdbTxn txn(env, LmdbTxn::Mode::ReadWrite);
    db.put(txn, key, bytes);
    txn.commit();
  } catch (...) {
    if (session) {
      session->warn(fmt(
          "rtsp-capture: failed to index segment '{}'",
          seg.path));
    }
  }
}

// Per-AU live publisher. Lives alongside the SegmentWriter for the
// duration of one connected capture session.
//
// Video oport (0): one EncodedSegment Beat per H.264 access unit
// (i.e. per encoded frame). FFmpeg's RTP H.264 depacketizer
// reassembles FU-A/STAP-A so `av_read_frame` already returns one
// AU per packet; the transcode path's encoder likewise emits one
// AU per `avcodec_receive_packet`. Each Beat's `data` field
// carries exactly that one AU's bytes verbatim (annex-B framing
// for both stream-copy RTSP h.264 and transcode-with-GLOBAL_HEADER
// paths).
//
// Audio oport (1): one EncodedSegment Beat per audio packet (one
// AAC frame for stream-copy AAC inputs, one encoded AAC frame for
// the transcode path). Audio is no longer batched with video — the
// natural per-packet PTS lets a downstream consumer pair audio and
// video by wall-clock or by PTS as needed, with frame-level
// granularity.
//
// Latency: per-AU emission cuts the upstream→consumer delay from
// ~1 GOP (segment_seconds at low end, target_duration at high end)
// to a single frame interval (~33 ms at 30 fps). The
// SegmentWriter still rolls files on IDR boundaries on disk; only
// the live Beat path is now frame-grained.
//
// `db_key` on every Beat is the 8-byte BE microseconds-since-epoch
// LMDB key of the on-disk segment the AU belongs to (the same key
// the capture stage writes into the `<camera>-videos` sub-db). The
// downstream consumer can fetch path / start_utc / end_utc /
// codec details by looking that key up in LMDB — the per-AU Beat
// does not duplicate the parent-segment row.
//
// Lifetime: construct after prepare_capture() so codec params /
// extradata can be cached from cs.v_out_par / cs.a_out_par. Call
// note_new_segment() once after writer.open() and again on every
// rollover. Feed packets via feed_video / feed_audio in the order
// they arrive. No flush is needed at EOS / before reconnect —
// every feed_* call has already emitted its Beat.
//
// Decoder startup caveat: a downstream decoder that joins the
// pipeline mid-stream and receives the first Beat as a P-frame AU
// will reject it until the next IDR arrives (gop_size frames at
// worst). That's an inherent property of any per-AU live feed and
// matches what GOP-level emission gave us when the consumer
// joined just after an IDR.
struct Publisher {
  bool                              video_enabled = false;
  bool                              audio_enabled = false;

  // Cached codec params for the EncodedSegment payload.
  unsigned                          v_codec_id    = 0;
  unsigned                          a_codec_id    = 0;
  unsigned                          v_width       = 0;
  unsigned                          v_height      = 0;
  unsigned                          v_fps_num     = 0;
  unsigned                          v_fps_den     = 0;
  unsigned                          a_sample_rate = 0;
  unsigned                          a_channels    = 0;
  std::vector<uint8_t>              v_extradata;
  std::vector<uint8_t>              a_extradata;
  std::string                       camera_name;

  // Parent-segment context, refreshed on every rollover.
  std::string                                   seg_db_key;
  std::string                                   seg_path;
  std::chrono::system_clock::time_point         seg_start_utc{};

  void note_new_segment(std::chrono::system_clock::time_point start_utc,
                        const std::string& path)
  {
    seg_start_utc = start_utc;
    seg_db_key    = be64_us_key(start_utc);
    seg_path      = path;
  }

  // Emit one video Beat for this AU on oport 0. `is_key` is
  // informational only (the Beat carries the AU bytes verbatim; the
  // decoder sniffs NAL types itself). Caller must hand us the bytes
  // BEFORE SegmentWriter::write_packet (which consumes pkt->data).
  // Runs on the capture thread, so it pushes through the non-coroutine
  // write_sync path; a closed oport (teardown) just drops the AU and
  // the capture loop exits via ctx.stop_requested() on its next check.
  void feed_video(RuntimeContext& ctx,
                  const uint8_t*  data,
                  size_t          size,
                  bool            is_key)
  {
    (void)is_key;
    if (!video_enabled || size == 0) { return; }
    const auto now = std::chrono::system_clock::now();
    EncodedSegment es;
    es.kind        = EncodedSegment::Kind::Video;
    es.db_key      = seg_db_key;
    es.camera_name = camera_name;
    es.path        = seg_path;
    es.start_utc   = now;
    es.end_utc     = now;
    es.duration_us = 0;
    es.codec_id    = v_codec_id;
    es.width       = v_width;
    es.height      = v_height;
    es.fps_num     = v_fps_num;
    es.fps_den     = v_fps_den;
    es.extradata   = v_extradata;
    es.data.assign(data, data + size);
    ctx.write_sync(0,
        make_payload<EncodedSegmentPayload>(std::move(es)));
  }

  // Emit one audio Beat for this audio packet on oport 1. Same
  // non-coroutine write_sync path as feed_video.
  void feed_audio(RuntimeContext& ctx,
                  const uint8_t*  data,
                  size_t          size)
  {
    if (!audio_enabled || size == 0) { return; }
    const auto now = std::chrono::system_clock::now();
    EncodedSegment es;
    es.kind        = EncodedSegment::Kind::Audio;
    es.db_key      = seg_db_key;
    es.camera_name = camera_name;
    es.path        = seg_path;
    es.start_utc   = now;
    es.end_utc     = now;
    es.duration_us = 0;
    es.codec_id    = a_codec_id;
    es.sample_rate = a_sample_rate;
    es.channels    = a_channels;
    es.extradata   = a_extradata;
    es.data.assign(data, data + size);
    ctx.write_sync(1,
        make_payload<EncodedSegmentPayload>(std::move(es)));
  }
};

}

namespace {

// Hand-off between RtspCaptureStage::process() and the capture thread
// it spawns. process() co_awaits awaiter(); the capture thread calls
// signal() when capture_loop_ returns. signal() resumes the coroutine
// on the session ThreadPool (or, if process() has not suspended yet,
// marks `finished` so the awaiter resumes inline). This lets process()
// free its worker thread while the capture thread runs FFmpeg, instead
// of pinning a pool worker for the camera's whole lifetime.
struct CaptureThreadDone {
  std::mutex              mu;
  std::coroutine_handle<> waiter{};
  bool                    finished = false;
  ThreadPool*             pool     = nullptr;

  void signal()
  {
    std::coroutine_handle<> h{};
    {
      std::lock_guard<std::mutex> lk(mu);
      finished = true;
      h        = waiter;
      waiter   = {};
    }
    if (h && pool) {
      pool->schedule(h);
    }
  }

  struct Awaiter {
    CaptureThreadDone* st;

    bool await_ready()
    {
      std::lock_guard<std::mutex> lk(st->mu);
      return st->finished;
    }

    bool await_suspend(std::coroutine_handle<> h)
    {
      std::lock_guard<std::mutex> lk(st->mu);
      if (st->finished) {
        return false;          // already finished: resume inline
      }
      st->waiter = h;
      return true;
    }

    void await_resume() const noexcept {}
  };

  Awaiter awaiter() noexcept { return Awaiter{this}; }
};

}

Job
RtspCaptureStage::process(RuntimeContext& ctx)
{
  LmdbEnv* env = session()->lmdb_env();
  if (!env) {
    session()->error(fmt(
        "RtspCaptureStage('{}'): session lmdb_env() unavailable",
        this->id()));
  }

  if (!ensure_directory_exists(_output_dir, session())) {
    session()->error(fmt(
        "RtspCaptureStage('{}'): output_dir '{}' is not usable",
        this->id(), _output_dir));
  }

  const FFmpegLibraries* libs = session()->ffmpeg_libraries();
  if (!libs || !libs->valid()) {
    session()->error(fmt(
        "RtspCaptureStage('{}'): FFmpeg libraries unavailable",
        this->id()));
  }

  // A capture stage does long-lived blocking FFmpeg I/O. Running that
  // directly in the process() coroutine would pin a session worker for
  // the camera's entire lifetime, so the number of simultaneous
  // capture stages would be capped at the worker-pool size. Instead we
  // run the whole capture on a thread this stage owns; process() then
  // suspends (freeing its worker) and is resumed by the capture thread
  // when it exits on stop.
  CaptureThreadDone done;
  done.pool = session()->thread_pool();
  std::thread cap_thread([this, &ctx, &done] {
    try {
      capture_loop_(ctx);
    } catch (...) {
      // A std::thread must never let an exception escape (it would
      // std::terminate the process). Oport writes use the non-throwing
      // write_sync path; this is only a backstop for the surrounding
      // FFmpeg / LMDB calls.
    }
    done.signal();
  });

  co_await done.awaiter();
  if (cap_thread.joinable()) {
    cap_thread.join();
  }

  ctx.signal_done();
  co_return;
}

void
RtspCaptureStage::capture_loop_(RuntimeContext& ctx)
{
  using namespace std::chrono;

  LmdbEnv*               env  = session()->lmdb_env();
  const FFmpegLibraries* libs = session()->ffmpeg_libraries();
  if (!env || !libs || !libs->valid()) {
    return;          // process() validated these; defensive only.
  }
  libs->avformat().api.network_init();

  std::string videos_db = _camera_name + _videos_db_suffix;

  // Watchdog-abort plumbing. `force_reconnect` is owned here so the
  // InterruptCtx pointer stays valid for every AVFormatContext we
  // create inside the connect loop. The watchdog thread reads
  // `frames_count` (updated from the packet loop) at each chrono tick
  // and sets `force_reconnect` if no frames arrived in the last
  // interval. The session stop flag is read straight off the
  // RuntimeContext by interrupt_cb, so no relay thread is needed.
  std::atomic<bool>     force_reconnect{false};
  std::atomic<uint64_t> frames_count{0};
  InterruptCtx          ic;
  ic.ctx             = &ctx;
  ic.force_reconnect = &force_reconnect;

  // Watchdog thread: when iport 0 is wired, consume each chrono
  // tick non-blockingly and arm `force_reconnect` if no RTSP
  // packets arrived since the previous tick. The non-blocking
  // peek+release pattern is safe to drive from this (non-coroutine)
  // thread -- it never registers a waiter, only takes the
  // buffer's mutex briefly.
  const bool watchdog_enabled = ctx.num_iports() >= 1;
  std::atomic<bool> watchdog_exit{false};
  std::thread watchdog_thread;
  if (watchdog_enabled) {
    watchdog_thread = std::thread([&] {
      uint64_t last_count = frames_count.load(std::memory_order_acquire);
      while (!watchdog_exit.load(std::memory_order_acquire)
             && !ctx.stop_requested()) {
        auto aw = ctx.peek(0, 0);
        if (aw.await_ready()) {
          const auto* p = aw.await_resume();
          if (!p) {
            // Watchdog source closed (chrono finished or pipeline
            // tearing down). Exit; no further triggers.
            return;
          }
          ctx.release_read(0, 1);
          uint64_t now_count =
              frames_count.load(std::memory_order_acquire);
          if (now_count == last_count) {
            session()->warn(fmt(
                "RtspCaptureStage('{}'): watchdog: no RTSP packets "
                "since previous tick -- forcing reconnect",
                this->id()));
            force_reconnect.store(true, std::memory_order_release);
          }
          last_count = now_count;
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
      }
    });
  }

  // RAII join guard for the watchdog thread. capture_loop_ runs on a
  // thread this stage owns, but a stray exception from an FFmpeg /
  // LMDB call would otherwise destroy the still-joinable std::thread
  // local during unwind and std::terminate the process. The guard
  // signals exit and joins on any scope exit (success or unwind).
  struct JoinGuard {
    std::thread&       t;
    std::atomic<bool>& exit_flag;
    ~JoinGuard() noexcept {
      exit_flag.store(true, std::memory_order_release);
      try {
        if (t.joinable()) { t.join(); }
      } catch (...) {
        // join() can throw system_error if the thread state is bad;
        // a destructor must not propagate, and we'd rather leak the
        // OS handle than terminate the process here.
      }
    }
  };
  JoinGuard watchdog_guard{watchdog_thread, watchdog_exit};

  // Outer reconnect loop. Each pass: load record, decrypt, open
  // input, segment until the stream errors out or the user stops.
  bool tried_refresh_this_cycle = false;
  while (!ctx.stop_requested()) {
    auto opt_rec = load_camera(*env, _cameras_db, _camera_name);
    if (!opt_rec) {
      session()->error(fmt(
          "RtspCaptureStage('{}'): camera '{}' not in '{}' db",
          this->id(), _camera_name, _cameras_db));
    }
    CameraRecord rec = *opt_rec;

    // Decrypt password.
    auto uuid_b = host_uuid_bytes();
    auto kdf    = build_kdf_input(
        std::span<const unsigned char, 16>(uuid_b),
        os_user_id());
    auto key    = derive_key(kdf);
    std::span<const unsigned char, 32> key_sp(key);
    auto pw_opt = open(key_sp, rec.password_blob, {});
    std::memset(key.data(), 0, key.size());
    std::memset(kdf.data(), 0, kdf.size());
    if (!pw_opt) {
      session()->error(fmt(
          "RtspCaptureStage('{}'): password decrypt failed for "
          "'{}'", this->id(), _camera_name));
    }
    std::string pw = std::move(*pw_opt);

    std::string url = embed_credentials(rec.rtsp_uri,
                                        rec.username, pw);

    session()->info(fmt(
        "RtspCaptureStage('{}'): connecting to {} ...",
        this->id(), rec.rtsp_uri));

    CaptureSession cs;
    cs.libs    = libs;
    cs.session = session();
    // Fresh connection => clear any leftover watchdog trip. Frame
    // counter is left alone; the watchdog tracks deltas, so an
    // immediate post-reconnect tick won't falsely re-fire as long
    // as at least one packet arrives between ticks.
    force_reconnect.store(false, std::memory_order_release);
    bool ok = prepare_capture(cs, url, _rtsp_transport,
                              _stimeout_us, _video_bitrate,
                              _audio_bitrate, _segment_seconds,
                              &ic);
    if (!ok) {
      // Connection failed. If supports_onvif and we haven't tried
      // refresh this cycle yet, run discovery and update the record.
      if (rec.supports_onvif && !tried_refresh_this_cycle) {
        tried_refresh_this_cycle = true;
        session()->info(fmt(
            "RtspCaptureStage('{}'): connection failed; running "
            "ONVIF refresh ...", this->id()));
        bool refreshed = refresh_rtsp_uri(
            rec, pw, milliseconds(_rediscover_timeout_ms),
            session());
        // Wipe plaintext password from memory ASAP.
        std::memset(pw.data(), 0, pw.size());
        if (refreshed) {
          save_camera(*env, _cameras_db, rec, true);
          continue;  // immediate retry with new URI
        }
      } else {
        std::memset(pw.data(), 0, pw.size());
      }
      // Fallback: back off and retry the same URI.
      stop_aware_sleep(ctx, milliseconds(_reconnect_delay_ms));
      tried_refresh_this_cycle = false;
      continue;
    }
    // Successful open resets the refresh budget.
    tried_refresh_this_cycle = false;
    std::memset(pw.data(), 0, pw.size());

    // Install RTP-error tap.
    std::atomic<bool> rtp_error{false};
    LogTapHandle tap = install_log_tap(
        [&rtp_error](int lvl, std::string_view tag,
                     std::string_view msg) {
          if (rtp_predicate(lvl, tag, msg)) {
            rtp_error.store(true, std::memory_order_release);
          }
        });

    // Set up SegmentWriter.
    SegmentSpec spec{_output_dir, _camera_name, _segment_seconds};
    SegmentWriter writer(libs, session(), std::move(spec),
                         cs.stream_copy_v, cs.stream_copy_a);
    std::vector<OutputStreamSpec> streams;
    OutputStreamSpec vspec;
    vspec.codecpar = cs.v_out_par;
    vspec.is_video = true;
    vspec.input_time_base =
        cs.stream_copy_v
        ? cs.ictx->streams[cs.v_in_idx]->time_base
        : cs.v_enc->time_base;
    streams.push_back(vspec);
    if (cs.a_in_idx >= 0 && cs.a_out_par) {
      OutputStreamSpec aspec;
      aspec.codecpar = cs.a_out_par;
      aspec.is_video = false;
      aspec.input_time_base =
          cs.stream_copy_a
          ? cs.ictx->streams[cs.a_in_idx]->time_base
          : cs.a_enc->time_base;
      streams.push_back(aspec);
    }
    if (!writer.open(streams)) {
      remove_log_tap(tap);
      stop_aware_sleep(ctx, milliseconds(_reconnect_delay_ms));
      continue;
    }

    // Per-AU live publisher: each source packet becomes one Beat on
    // oport 0 (video) / oport 1 (audio). Bound to the writer's
    // current segment so every Beat tags its parent's LMDB key for
    // downstream lookups in `<camera>-videos`.
    Publisher pub;
    pub.video_enabled = ctx.has_consumers(0);
    pub.audio_enabled = ctx.has_consumers(1) && cs.a_in_idx >= 0;
    pub.camera_name   = _camera_name;
    if (cs.v_out_par) {
      pub.v_codec_id = static_cast<unsigned>(cs.v_out_par->codec_id);
      pub.v_width    = static_cast<unsigned>(cs.v_out_par->width);
      pub.v_height   = static_cast<unsigned>(cs.v_out_par->height);
      // Original source cadence: the input stream's avg_frame_rate,
      // falling back to r_frame_rate (many RTSP cameras leave avg unset
      // until enough frames arrive). 0/0 stays when neither is known.
      AVRational fr{0, 1};
      if (cs.v_in_idx >= 0) {
        const AVStream* v_in = cs.ictx->streams[cs.v_in_idx];
        fr = v_in->avg_frame_rate;
        if (fr.num <= 0 || fr.den <= 0) { fr = v_in->r_frame_rate; }
      }
      if (fr.num > 0 && fr.den > 0) {
        pub.v_fps_num = static_cast<unsigned>(fr.num);
        pub.v_fps_den = static_cast<unsigned>(fr.den);
      }
      if (cs.v_out_par->extradata && cs.v_out_par->extradata_size > 0) {
        pub.v_extradata.assign(
            cs.v_out_par->extradata,
            cs.v_out_par->extradata + cs.v_out_par->extradata_size);
      }
    }
    if (cs.a_out_par) {
      pub.a_codec_id = static_cast<unsigned>(cs.a_out_par->codec_id);
      pub.a_sample_rate =
          static_cast<unsigned>(cs.a_out_par->sample_rate);
      pub.a_channels =
          static_cast<unsigned>(cs.a_out_par->ch_layout.nb_channels);
      if (cs.a_out_par->extradata && cs.a_out_par->extradata_size > 0) {
        pub.a_extradata.assign(
            cs.a_out_par->extradata,
            cs.a_out_par->extradata + cs.a_out_par->extradata_size);
      }
    }
    pub.note_new_segment(writer.file_open_utc(),
                         writer.current_path());

    // Packet loop.
    const auto& fmt_api  = libs->avformat().api;
    const auto& cdc_api  = libs->avcodec().api;
    const auto& util_api = libs->avutil().api;
    int v_audio_idx = (cs.a_in_idx >= 0) ? 1 : -1;

    while (!ctx.stop_requested()
           && !rtp_error.load(std::memory_order_acquire)
           && !force_reconnect.load(std::memory_order_acquire))
    {
      int rc = fmt_api.read_frame(cs.ictx, cs.in_pkt);
      if (rc < 0) {
        if (rc != AVERROR_EOF) {
          session()->warn(fmt(
              "RtspCaptureStage('{}'): read_frame returned {}",
              this->id(), rc));
        }
        break;
      }
      // Liveness signal for the watchdog thread.
      frames_count.fetch_add(1, std::memory_order_release);

      if (cs.in_pkt->stream_index == cs.v_in_idx) {
        if (cs.stream_copy_v) {
          bool is_key =
              (cs.in_pkt->flags & AV_PKT_FLAG_KEY) != 0;
          // Snapshot bytes BEFORE write_packet -- interleaved_write_frame
          // takes ownership of the packet's buffer and unrefs it.
          pub.feed_video(ctx,
              cs.in_pkt->data,
              static_cast<size_t>(cs.in_pkt->size),
              is_key);
          auto info = writer.write_packet(cs.in_pkt, 0, is_key);
          if (info && !info->path.empty()) {
            index_segment(*env, videos_db, _camera_name,
                          *info, session());
            pub.note_new_segment(writer.file_open_utc(),
                                 writer.current_path());
          }
        } else {
          rc = cdc_api.send_packet(cs.v_dec, cs.in_pkt);
          if (rc < 0 && rc != AVERROR(EAGAIN)) {
            cdc_api.packet_unref(cs.in_pkt);
            continue;
          }
          while (true) {
            rc = cdc_api.receive_frame(cs.v_dec, cs.v_frame);
            if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
              break;
            }
            if (rc < 0) { break; }
            if (writer.wants_keyframe()) {
              cs.v_frame->pict_type = AV_PICTURE_TYPE_I;
            } else {
              cs.v_frame->pict_type = AV_PICTURE_TYPE_NONE;
            }
            rc = cdc_api.send_frame(cs.v_enc, cs.v_frame);
            util_api.frame_unref(cs.v_frame);
            if (rc < 0) { break; }
            while (true) {
              rc = cdc_api.receive_packet(cs.v_enc, cs.out_pkt);
              if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
                break;
              }
              if (rc < 0) { break; }
              bool is_key =
                  (cs.out_pkt->flags & AV_PKT_FLAG_KEY) != 0;
              pub.feed_video(ctx,
                  cs.out_pkt->data,
                  static_cast<size_t>(cs.out_pkt->size),
                  is_key);
              auto info = writer.write_packet(cs.out_pkt, 0,
                                              is_key);
              if (info && !info->path.empty()) {
                index_segment(*env, videos_db, _camera_name,
                              *info, session());
                pub.note_new_segment(writer.file_open_utc(),
                                     writer.current_path());
              }
            }
          }
          cdc_api.packet_unref(cs.in_pkt);
        }
      } else if (cs.in_pkt->stream_index == cs.a_in_idx
                 && v_audio_idx >= 0) {
        if (cs.stream_copy_a) {
          pub.feed_audio(ctx,
              cs.in_pkt->data,
              static_cast<size_t>(cs.in_pkt->size));
          writer.write_packet(cs.in_pkt, v_audio_idx, false);
        } else {
          rc = cdc_api.send_packet(cs.a_dec, cs.in_pkt);
          if (rc < 0 && rc != AVERROR(EAGAIN)) {
            cdc_api.packet_unref(cs.in_pkt);
            continue;
          }
          const auto& swr_api = libs->swresample().api;
          while (true) {
            rc = cdc_api.receive_frame(cs.a_dec, cs.a_frame);
            if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
              break;
            }
            if (rc < 0) { break; }

            // Convert if swr is engaged, otherwise the decoded
            // samples already match the encoder's expected layout
            // and can be pushed into the FIFO as-is.
            if (cs.swr) {
              int n_out = swr_api.convert(
                  cs.swr,
                  cs.a_swr_out_frame->data,
                  cs.a_swr_out_frame->nb_samples,
                  const_cast<const uint8_t**>(cs.a_frame->extended_data),
                  cs.a_frame->nb_samples);
              if (n_out > 0) {
                cs.a_fifo.push_planar(
                    cs.a_swr_out_frame->data, n_out);
              }
            } else {
              cs.a_fifo.push_planar(
                  cs.a_frame->extended_data,
                  cs.a_frame->nb_samples);
            }
            util_api.frame_unref(cs.a_frame);

            // Drain full encoder frames from the FIFO.
            while (cs.a_fifo.pull_frame(cs.a_enc_in_frame->data)) {
              cs.a_enc_in_frame->pts = cs.a_next_pts;
              cs.a_next_pts += cs.a_enc_in_frame->nb_samples;
              rc = cdc_api.send_frame(cs.a_enc, cs.a_enc_in_frame);
              if (rc < 0) { break; }
              while (true) {
                rc = cdc_api.receive_packet(cs.a_enc, cs.out_pkt);
                if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
                  break;
                }
                if (rc < 0) { break; }
                pub.feed_audio(ctx,
                    cs.out_pkt->data,
                    static_cast<size_t>(cs.out_pkt->size));
                writer.write_packet(cs.out_pkt, v_audio_idx, false);
              }
            }
          }
          cdc_api.packet_unref(cs.in_pkt);
        }
      } else {
        cdc_api.packet_unref(cs.in_pkt);
      }
    }

    remove_log_tap(tap);
    // Per-AU mode: every packet has already been emitted by
    // feed_video / feed_audio. Nothing to flush here.
    SegmentInfo final_seg = writer.close();
    if (!final_seg.path.empty()) {
      index_segment(*env, videos_db, _camera_name, final_seg,
                    session());
    }

    if (rtp_error.load(std::memory_order_acquire)) {
      session()->warn(fmt(
          "RtspCaptureStage('{}'): RTP error -- reconnecting "
          "after {} ms",
          this->id(), _reconnect_delay_ms));
    }
    if (!ctx.stop_requested()) {
      stop_aware_sleep(ctx, milliseconds(_reconnect_delay_ms));
    }
  }

  // The watchdog thread is torn down by watchdog_guard on scope exit.
  // process() joins this capture thread and calls signal_done() once
  // we return.
}

VPIPE_REGISTER_STAGE(RtspCaptureStage)
VPIPE_REGISTER_SPEC(RtspCaptureStage, kSpec)

}
