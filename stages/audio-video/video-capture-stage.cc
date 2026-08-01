#include "stages/audio-video/video-capture-stage.h"
#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/oport-policy.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

// AVIOInterruptCB opaque: poll a `stop` flag so libavformat can punch out of
// blocking reads inside the avfoundation demuxer.
struct InterruptCtx {
  std::atomic<bool>* stop_requested = nullptr;
};

int
interrupt_cb_(void* opaque) noexcept
{
  auto* ic = static_cast<InterruptCtx*>(opaque);
  if (!ic) { return 0; }
  if (ic->stop_requested
      && ic->stop_requested->load(std::memory_order_acquire)) {
    return 1;
  }
  return 0;
}

void
stop_aware_sleep_(RuntimeContext& ctx, std::chrono::milliseconds total)
{
  using namespace std::chrono;
  auto deadline = steady_clock::now() + total;
  constexpr auto kChunk = milliseconds(50);
  while (true) {
    if (ctx.stop_requested()) { return; }
    auto now = steady_clock::now();
    if (now >= deadline) { return; }
    auto remaining = deadline - now;
    std::this_thread::sleep_for(remaining < kChunk ? remaining : kChunk);
  }
}

string
lower_(string_view s)
{
  string o;
  o.reserve(s.size());
  for (char c : s) {
    o.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(c))));
  }
  return o;
}

}  // namespace

VideoCaptureStage::VideoCaptureStage(const SessionContextIntf* s,
                                     string                    id,
                                     vector<InEdge>            iports,
                                     FlexData                  config)
  : TypedStage<VideoCaptureStage>(s, std::move(id), std::move(iports),
                                  std::move(config))
{
  // Validation is deferred to launch (see Stage::fail_config).
  const FlexData& cfg = this->config();
  if (!cfg.is_object()) {
    fail_config(fmt(
        "VideoCaptureStage('{}'): config must be an object", this->id()));
  }
  FlexData empty_obj = FlexData::make_object();
  auto root = (cfg.is_object() ? cfg : empty_obj).as_object();

  if (root.contains("device_id")) {
    FlexData v = root.at("device_id");
    if (v.is_uint() || v.is_int()) {
      int64_t id_v = v.as_int(-1);
      if (id_v < 0) {
        fail_config(fmt(
            "VideoCaptureStage('{}'): device_id must be >= 0", this->id()));
      } else {
        _has_device_id = true;
        _device_id     = static_cast<uint64_t>(id_v);
      }
    } else {
      fail_config(fmt(
          "VideoCaptureStage('{}'): device_id must be an integer",
          this->id()));
    }
  }
  if (root.contains("device_name")) {
    _device_name = string(root.at("device_name").as_string(""));
  }
  if (!_has_device_id && _device_name.empty()) {
    fail_config(fmt(
        "VideoCaptureStage('{}'): exactly one of device_id or device_name "
        "must be set", this->id()));
  } else if (_has_device_id && !_device_name.empty()) {
    fail_config(fmt(
        "VideoCaptureStage('{}'): device_id and device_name are mutually "
        "exclusive", this->id()));
  }

  // Scalar attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _req_width          = static_cast<unsigned>(attr_uint("width"));
  _req_height         = static_cast<unsigned>(attr_uint("height"));
  _req_framerate      = attr_real("framerate");
  _pixel_format       = string(attr_str("pixel_format"));
  _camera_name        = string(attr_str("camera_name"));
  _reconnect_delay_ms = static_cast<unsigned>(attr_uint("reconnect_delay_ms"));
  _oport_depth        = static_cast<unsigned>(attr_uint("oport_depth"));
  if (_oport_depth == 0) { _oport_depth = 1; }

  // avfoundation's `video_size` is a single "WxH" string, so a half-specified
  // resolution has no meaning -- reject it rather than silently guessing the
  // other axis from the device default.
  if ((_req_width > 0) != (_req_height > 0)) {
    fail_config(fmt(
        "VideoCaptureStage('{}'): width and height must be set together "
        "(got {}x{})", this->id(), _req_width, _req_height));
  }
  if (_req_framerate < 0.0) {
    fail_config(fmt(
        "VideoCaptureStage('{}'): framerate must be >= 0 (got {})",
        this->id(), _req_framerate));
  }

  const string dt = lower_(attr_str("output_dtype"));
  if (dt.empty() || dt == "u8") {
    _output_dtype = TensorBeat::DType::U8;
  } else if (dt == "f32") {
    _output_dtype = TensorBeat::DType::F32;
  } else {
    fail_config(fmt(
        "VideoCaptureStage('{}'): output_dtype must be \"u8\" or \"f32\" "
        "(got '{}')", this->id(), dt));
  }

  allocate_oports(spec().oports.size());
  // DropOldest so a slow downstream consumer cannot stall live capture
  // (mirrors audio-capture / rtsp-capture).
  set_oport_policy(0, {_oport_depth, OverrunPolicy::DropOldest});
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "device_id", .type = ConfigType::Uint,
   .doc = "avfoundation VIDEO device index (mutually exclusive with "
          "device_name; video indices are numbered separately from audio)"},
  {.key = "device_name", .type = ConfigType::String,
   .doc = "avfoundation video device name; case-insensitive substring match "
          "(mutually exclusive with device_id)"},
  {.key = "width", .type = ConfigType::Uint,
   .doc = "requested capture width; set together with height "
          "(avfoundation video_size). 0 = device default",
   .def_uint = 0},
  {.key = "height", .type = ConfigType::Uint,
   .doc = "requested capture height; set together with width "
          "(avfoundation video_size). 0 = device default",
   .def_uint = 0},
  {.key = "framerate", .type = ConfigType::Real,
   .doc = "requested frames per second (avfoundation framerate). "
          "0 = device default",
   .def_real = 0.0},
  {.key = "pixel_format", .type = ConfigType::String,
   .doc = "requested capture pixel format (avfoundation pixel_format), e.g. "
          "uyvy422 / nv12 / bgr0; empty = device default. Output is RGB "
          "either way",
   .def_str = ""},
  {.key = "output_dtype", .type = ConfigType::String,
   .doc = "emitted element type: \"u8\" (default) or \"f32\" (normalized "
          "to [0,1])",
   .def_str = "u8"},
  {.key = "camera_name", .type = ConfigType::String,
   .doc = "label copied into each beat's sideband so multi-camera graphs can "
          "tell sources apart",
   .def_str = ""},
  {.key = "reconnect_delay_ms", .type = ConfigType::Uint,
   .doc = "backoff before reopen on error (ms)", .def_uint = 2000},
  {.key = "oport_depth", .type = ConfigType::Uint,
   .doc = "output ring depth (DropOldest)", .def_uint = 8},
};
const PortSpec kOports[] = {
  {.name = "frames", .doc = "planar RGB TensorBeat [3,H,W] (U8 or F32), one "
                            "per captured frame -- the same payload "
                            "video-to-rgb emits",
   .type = &typeid(TensorBeatPayload), .tags = "rgb-frames",
   .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "video-capture",
  .doc       = "Source: captures a camera via FFmpeg avfoundation and emits "
               "one planar RGB TensorBeat per frame. Apple-only. 0 iports.",
  .display_name = "Video Capture",
  .category  = StageCategory::Visual,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
VideoCaptureStage::spec() const noexcept
{
  return kSpec;
}

int
VideoCaptureStage::probe_device_index_by_name_()
{
  // Same one-shot probe the audio twin uses: spawn ffmpeg and parse the
  // device listing it writes to stderr. Returns -1 if ffmpeg is missing.
  FILE* p = ::popen(
      "ffmpeg -hide_banner -f avfoundation -list_devices true -i '' 2>&1",
      "r");
  if (!p) { return -1; }
  string out;
  char buf[512];
  while (std::fgets(buf, sizeof(buf), p)) { out.append(buf); }
  ::pclose(p);

  // ffmpeg prints the VIDEO block first, then the audio one:
  //   [AVFoundation indev @ 0x...] AVFoundation video devices:
  //   [AVFoundation indev @ 0x...] [0] MacBook Air Camera
  //   [AVFoundation indev @ 0x...] [1] MacBook Air Desk View Camera
  //   [AVFoundation indev @ 0x...] AVFoundation audio devices:
  //   [AVFoundation indev @ 0x...] [0] MacBook Air Microphone
  // Scanning must STOP at the audio header: video and audio indices are
  // separate namespaces, so matching a microphone's name here would hand
  // back an index into the wrong device list.
  const auto vb = out.find("video devices:");
  if (vb == string::npos) { return -1; }
  auto end = out.find("audio devices:", vb);
  if (end == string::npos) { end = out.size(); }

  const string lc_target = lower_(_device_name);

  size_t pos = out.find('\n', vb);
  if (pos == string::npos) { return -1; }
  ++pos;
  while (pos < end) {
    auto eol = out.find('\n', pos);
    if (eol == string::npos || eol > end) { eol = end; }
    const string line = out.substr(pos, eol - pos);
    pos = eol + 1;
    if (line.find("AVFoundation") == string::npos) { continue; }
    // "[AVFoundation indev @ 0x..] [0] Name" -> the SECOND bracket pair.
    const auto first_rb = line.find(']');
    if (first_rb == string::npos) { continue; }
    const auto lb = line.find('[', first_rb + 1);
    const auto rb = (lb == string::npos)
        ? string::npos : line.find(']', lb + 1);
    if (lb == string::npos || rb == string::npos) { continue; }
    const string idx_s = line.substr(lb + 1, rb - lb - 1);
    string name = line.substr(rb + 1);
    while (!name.empty() && (name.front() == ' ' || name.front() == '\t')) {
      name.erase(name.begin());
    }
    while (!name.empty()
           && (name.back() == '\r' || name.back() == '\n'
               || name.back() == ' ' || name.back() == '\t')) {
      name.pop_back();
    }
    if (lower_(name).find(lc_target) != string::npos) {
      try { return std::stoi(idx_s); }
      catch (...) { return -1; }
    }
  }
  return -1;
}

Job
VideoCaptureStage::process(RuntimeContext& ctx)
{
  using namespace std::chrono;

  const FFmpegLibraries* libs = session()->ffmpeg_libraries();
  if (!libs || !libs->valid()) {
    session()->error(fmt(
        "VideoCaptureStage('{}'): FFmpeg libraries unavailable", this->id()));
  }
  if (!libs->avdevice().valid()) {
    session()->error(fmt(
        "VideoCaptureStage('{}'): libavdevice not loaded -- install it "
        "(Homebrew ffmpeg ships it as libavdevice.dylib)", this->id()));
  }
  libs->avdevice().api.register_all();

  const AVInputFormat* ifmt =
      libs->avformat().api.find_input_format("avfoundation");
  if (!ifmt) {
    session()->error(fmt(
        "VideoCaptureStage('{}'): av_find_input_format(\"avfoundation\") "
        "returned null", this->id()));
  }

  int resolved_index = -1;
  if (_has_device_id) {
    resolved_index = static_cast<int>(_device_id);
  } else {
    resolved_index = probe_device_index_by_name_();
    if (resolved_index < 0) {
      session()->error(fmt(
          "VideoCaptureStage('{}'): no avfoundation VIDEO device matched "
          "'{}' (ffmpeg -f avfoundation -list_devices true shows the "
          "available indices)", this->id(), _device_name));
    }
    session()->info(fmt(
        "VideoCaptureStage('{}'): resolved device_name '{}' to avfoundation "
        "video index {}", this->id(), _device_name, resolved_index));
  }

  // avfoundation's URL grammar is "[VIDEO]:[AUDIO]". Video-only capture puts
  // the index BEFORE the colon -- the mirror image of audio-capture's ":N".
  const string url = std::to_string(resolved_index) + ":";

  // Stop relay -- mirrors ctx.stop_requested() into a stable atomic the
  // InterruptCtx can poll from FFmpeg's C callback.
  std::atomic<bool> stop_flag{false};
  std::atomic<bool> relay_exit{false};
  std::thread stop_relay([&] {
    while (!relay_exit.load(std::memory_order_acquire)) {
      if (ctx.stop_requested()) {
        stop_flag.store(true, std::memory_order_release);
      }
      std::this_thread::sleep_for(milliseconds(50));
    }
    if (ctx.stop_requested()) {
      stop_flag.store(true, std::memory_order_release);
    }
  });
  struct JoinGuard {
    std::thread&       t;
    std::atomic<bool>& exit_flag;
    ~JoinGuard() noexcept {
      exit_flag.store(true, std::memory_order_release);
      try { if (t.joinable()) { t.join(); } } catch (...) {}
    }
  };
  JoinGuard relay_guard{stop_relay, relay_exit};

  InterruptCtx ic;
  ic.stop_requested = &stop_flag;

  const auto& fmt_api  = libs->avformat().api;
  const auto& cdc_api  = libs->avcodec().api;
  const auto& util_api = libs->avutil().api;
  const auto& sws_api  = libs->swscale().api;

  AVPacket* pkt   = cdc_api.packet_alloc();
  AVFrame*  frame = util_api.frame_alloc();
  AVFrame*  gbrp  = util_api.frame_alloc();
  if (!pkt || !frame || !gbrp) {
    session()->error(fmt(
        "VideoCaptureStage('{}'): packet/frame alloc failed", this->id()));
  }
  // Freed on every exit path (including the co_return inside the loop).
  struct AvGuard {
    const FFmpegLibraries* libs;
    AVPacket** pkt; AVFrame** frame; AVFrame** gbrp;
    ~AvGuard() noexcept {
      libs->avcodec().api.packet_free(pkt);
      libs->avutil().api.frame_free(frame);
      libs->avutil().api.frame_free(gbrp);
    }
  };
  AvGuard av_guard{libs, &pkt, &frame, &gbrp};

  // Outer reconnect loop. Each pass: open the device, drain frames until
  // error or stop, then close.
  while (!ctx.stop_requested()) {
    AVFormatContext* ictx = fmt_api.alloc_context();
    if (!ictx) {
      session()->warn(fmt(
          "VideoCaptureStage('{}'): avformat_alloc_context failed",
          this->id()));
      stop_aware_sleep_(ctx, milliseconds(_reconnect_delay_ms));
      continue;
    }
    ictx->interrupt_callback.callback = &interrupt_cb_;
    ictx->interrupt_callback.opaque   = &ic;

    // The avfoundation knobs, passed through verbatim.
    AVDictionary* opts = nullptr;
    if (_req_width > 0 && _req_height > 0) {
      char b[64];
      std::snprintf(b, sizeof(b), "%ux%u", _req_width, _req_height);
      util_api.dict_set(&opts, "video_size", b, 0);
    }
    if (_req_framerate > 0.0) {
      char b[64];
      std::snprintf(b, sizeof(b), "%g", _req_framerate);
      util_api.dict_set(&opts, "framerate", b, 0);
    }
    if (!_pixel_format.empty()) {
      util_api.dict_set(&opts, "pixel_format", _pixel_format.c_str(), 0);
    }

    int rc = fmt_api.open_input(&ictx, url.c_str(),
        const_cast<AVInputFormat*>(ifmt), &opts);
    if (opts) { util_api.dict_free(&opts); }
    if (rc < 0) {
      char ebuf[256] = {0};
      util_api.strerror(rc, ebuf, sizeof(ebuf));
      // avfoundation rejects an unsupported size/rate combination and logs
      // the legal modes itself; say so rather than only showing the errno.
      session()->warn(fmt(
          "VideoCaptureStage('{}'): open_input failed ({}: {}); if width/"
          "height/framerate are set, avfoundation lists the modes it "
          "supports in its own log above. Retrying in {} ms",
          this->id(), rc, ebuf, _reconnect_delay_ms));
      stop_aware_sleep_(ctx, milliseconds(_reconnect_delay_ms));
      continue;
    }

    rc = fmt_api.find_stream_info(ictx, nullptr);
    if (rc < 0) {
      session()->warn(fmt(
          "VideoCaptureStage('{}'): find_stream_info failed ({}); reopening",
          this->id(), rc));
      fmt_api.close_input(&ictx);
      stop_aware_sleep_(ctx, milliseconds(_reconnect_delay_ms));
      continue;
    }

    int v_idx = -1;
    for (unsigned i = 0; i < ictx->nb_streams; ++i) {
      if (ictx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        v_idx = static_cast<int>(i);
        break;
      }
    }
    if (v_idx < 0) {
      session()->error(fmt(
          "VideoCaptureStage('{}'): no video stream on device {}",
          this->id(), url));
      fmt_api.close_input(&ictx);
      co_return;
    }

    auto* v_st  = ictx->streams[v_idx];
    auto* v_par = v_st->codecpar;
    _input_width    = static_cast<unsigned>(v_par->width);
    _input_height   = static_cast<unsigned>(v_par->height);
    _input_codec_id = static_cast<unsigned>(v_par->codec_id);
    // Negotiated cadence, forwarded on every beat's sideband so a sink can
    // adopt the camera's own rate (same field video-to-rgb propagates).
    AVRational fr = v_st->avg_frame_rate;
    if (fr.num <= 0 || fr.den <= 0) { fr = v_st->r_frame_rate; }
    _fps_num = (fr.num > 0 && fr.den > 0) ? static_cast<unsigned>(fr.num) : 0;
    _fps_den = (fr.num > 0 && fr.den > 0) ? static_cast<unsigned>(fr.den) : 0;

    // avfoundation hands over RAWVIDEO; the decoder is what turns the packet
    // into an AVFrame carrying a pixel format swscale can consume.
    const AVCodec* dec = cdc_api.find_decoder(v_par->codec_id);
    AVCodecContext* dctx = dec ? cdc_api.alloc_context3(dec) : nullptr;
    if (!dec || !dctx
        || cdc_api.parameters_to_context(dctx, v_par) < 0
        || cdc_api.open2(dctx, dec, nullptr) < 0) {
      session()->error(fmt(
          "VideoCaptureStage('{}'): no usable decoder for codec_id={} on "
          "device {}", this->id(), static_cast<int>(v_par->codec_id), url));
      if (dctx) { cdc_api.free_context(&dctx); }
      fmt_api.close_input(&ictx);
      co_return;
    }

    session()->info(fmt(
        "VideoCaptureStage('{}'): capturing device='{}' codec_id={} {}x{} "
        "fps={}/{} -> RGB {}",
        this->id(), url, static_cast<int>(_input_codec_id),
        _input_width, _input_height, _fps_num, _fps_den,
        _output_dtype == TensorBeat::DType::U8 ? "u8" : "f32"));

    // swscale: whatever the camera gives -> planar RGB (GBRP), same size.
    // Rebuilt per open so a device that comes back at another size/format is
    // handled; get_cached_context reuses it when the parameters match.
    SwsContext* sws = nullptr;

    while (!ctx.stop_requested()) {
      cdc_api.packet_unref(pkt);
      int read_rc = fmt_api.read_frame(ictx, pkt);
      if (read_rc == AVERROR(EAGAIN)) {
        // No frame ready yet -- the steady-state gap between frames, not an
        // error. Sleep well under a frame interval (30 fps = 33 ms).
        std::this_thread::sleep_for(milliseconds(2));
        continue;
      }
      if (read_rc < 0) {
        if (!ctx.stop_requested()) {
          char ebuf[256] = {0};
          util_api.strerror(read_rc, ebuf, sizeof(ebuf));
          session()->warn(fmt(
              "VideoCaptureStage('{}'): read_frame failed ({}: {}); "
              "reopening device", this->id(), read_rc, ebuf));
        }
        break;
      }
      if (pkt->stream_index != v_idx || pkt->size <= 0) { continue; }

      if (cdc_api.send_packet(dctx, pkt) < 0) { continue; }
      while (cdc_api.receive_frame(dctx, frame) == 0) {
        const auto now = system_clock::now();
        const int w = frame->width, h = frame->height;
        if (w <= 0 || h <= 0) { util_api.frame_unref(frame); continue; }

        sws = sws_api.get_cached_context(
            sws, w, h, static_cast<AVPixelFormat>(frame->format),
            w, h, AV_PIX_FMT_GBRP, SWS_BILINEAR,
            nullptr, nullptr, nullptr);
        if (!sws) {
          session()->warn(fmt(
              "VideoCaptureStage('{}'): sws_getCachedContext failed for "
              "{}x{} fmt={}", this->id(), w, h, frame->format));
          util_api.frame_unref(frame);
          continue;
        }
        // (Re)allocate the GBRP staging frame when the geometry changes.
        if (gbrp->width != w || gbrp->height != h
            || gbrp->format != AV_PIX_FMT_GBRP) {
          util_api.frame_unref(gbrp);
          gbrp->width  = w;
          gbrp->height = h;
          gbrp->format = AV_PIX_FMT_GBRP;
          if (util_api.frame_get_buffer(gbrp, 0) < 0) {
            session()->warn(fmt(
                "VideoCaptureStage('{}'): av_frame_get_buffer failed for "
                "{}x{} GBRP", this->id(), w, h));
            util_api.frame_unref(frame);
            continue;
          }
        }
        sws_api.scale(sws, frame->data, frame->linesize, 0, h,
                      gbrp->data, gbrp->linesize);

        TensorBeat tb;
        tb.dtype          = _output_dtype;
        tb.shape          = {3, h, w};
        tb.storage_offset = 0;
        const size_t esz = tb.element_byte_size();
        // One uniform per-row pitch that fits every plane's linesize; when it
        // equals w the beat is plain contiguous (no strides).
        int P = gbrp->linesize[0];
        if (gbrp->linesize[1] > P) { P = gbrp->linesize[1]; }
        if (gbrp->linesize[2] > P) { P = gbrp->linesize[2]; }
        const size_t row_stride = static_cast<size_t>(P == w ? w : P);
        if (P == w) {
          tb.data.assign(static_cast<size_t>(3) * h * w * esz, 0);
        } else {
          tb.strides = {static_cast<int64_t>(h) * P, P, 1};
          tb.data.assign(static_cast<size_t>(3) * h * P * esz, 0);
        }

        // GBRP plane indices: G=0, B=1, R=2. TensorBeat wants R, G, B.
        const int src_plane_for_channel[3] = {2, 0, 1};
        if (_output_dtype == TensorBeat::DType::U8) {
          uint8_t* dst_base = tb.as_u8();
          for (int c = 0; c < 3; ++c) {
            const int      sp  = src_plane_for_channel[c];
            const uint8_t* src = gbrp->data[sp];
            const int      ss  = gbrp->linesize[sp];
            uint8_t* dst_plane = dst_base
                + static_cast<size_t>(c) * h * row_stride;
            for (int y = 0; y < h; ++y) {
              std::memcpy(dst_plane + static_cast<size_t>(y) * row_stride,
                          src + static_cast<size_t>(y) * ss,
                          static_cast<size_t>(w));
            }
          }
        } else {
          float* dst_base = tb.as_f32();
          for (int c = 0; c < 3; ++c) {
            const int      sp  = src_plane_for_channel[c];
            const uint8_t* src = gbrp->data[sp];
            const int      ss  = gbrp->linesize[sp];
            float* dst_plane = dst_base
                + static_cast<size_t>(c) * h * row_stride;
            for (int y = 0; y < h; ++y) {
              const uint8_t* src_row = src + static_cast<size_t>(y) * ss;
              float* dst_row = dst_plane + static_cast<size_t>(y) * row_stride;
              for (int x = 0; x < w; ++x) {
                dst_row[x] = static_cast<float>(src_row[x]) * (1.0f / 255.0f);
              }
            }
          }
        }

        FlexData sb = FlexData::make_object();
        sb.as_object().insert_or_assign("timestamp_us",
            FlexData::make_uint(static_cast<uint64_t>(
                duration_cast<microseconds>(
                    now.time_since_epoch()).count())));
        if (!_camera_name.empty()) {
          sb.as_object().insert_or_assign("camera_name",
              FlexData::make_string(_camera_name));
        }
        if (_fps_num > 0 && _fps_den > 0) {
          sb.as_object().insert_or_assign("fps_num",
              FlexData::make_uint(_fps_num));
          sb.as_object().insert_or_assign("fps_den",
              FlexData::make_uint(_fps_den));
        }
        tb.sideband = std::move(sb);

        util_api.frame_unref(frame);
        ++_frames_emitted;
        co_await ctx.write(0,
            make_payload<TensorBeatPayload>(std::move(tb)));
      }
    }

    if (sws) { sws_api.free_context(sws); }
    cdc_api.free_context(&dctx);
    fmt_api.close_input(&ictx);
    if (ctx.stop_requested()) { break; }
    stop_aware_sleep_(ctx, milliseconds(_reconnect_delay_ms));
  }

  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(VideoCaptureStage)
VPIPE_REGISTER_SPEC(VideoCaptureStage, kSpec)

}
