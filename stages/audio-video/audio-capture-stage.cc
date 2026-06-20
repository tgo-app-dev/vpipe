#include "stages/audio-video/audio-capture-stage.h"
#include "common/beat-payload-intf.h"
#include "common/encoded-segment.h"
#include "common/ffmpeg-libraries.h"
#include "common/flex-data.h"
#include "common/oport-policy.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <atomic>
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

// AVIOInterruptCB opaque: poll a `stop` flag so libavformat can
// punch out of blocking reads inside the avfoundation demuxer.
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

}  // namespace

AudioCaptureStage::AudioCaptureStage(const SessionContextIntf* s,
                                     string                    id,
                                     vector<InEdge>            iports,
                                     FlexData                  config)
  : TypedStage<AudioCaptureStage>(s, std::move(id), std::move(iports),
                                  std::move(config))
{
  // Validation is deferred to launch (see Stage::fail_config).
  const FlexData& cfg = this->config();
  if (!cfg.is_object()) {
    fail_config(fmt(
        "AudioCaptureStage('{}'): config must be an object",
        this->id()));
  }
  FlexData empty_obj = FlexData::make_object();
  auto root = (cfg.is_object() ? cfg : empty_obj).as_object();

  if (root.contains("device_id")) {
    FlexData v = root.at("device_id");
    if (v.is_uint() || v.is_int()) {
      int64_t id_v = v.as_int(-1);
      if (id_v < 0) {
        fail_config(fmt(
            "AudioCaptureStage('{}'): device_id must be >= 0",
            this->id()));
      } else {
        _has_device_id = true;
        _device_id     = static_cast<uint64_t>(id_v);
      }
    } else {
      fail_config(fmt(
          "AudioCaptureStage('{}'): device_id must be an integer",
          this->id()));
    }
  }
  if (root.contains("device_name")) {
    _device_name = string(root.at("device_name").as_string(""));
  }
  if (!_has_device_id && _device_name.empty()) {
    fail_config(fmt(
        "AudioCaptureStage('{}'): exactly one of device_id or "
        "device_name must be set", this->id()));
  } else if (_has_device_id && !_device_name.empty()) {
    fail_config(fmt(
        "AudioCaptureStage('{}'): device_id and device_name are "
        "mutually exclusive", this->id()));
  }

  // Scalar attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _req_sample_rate    = static_cast<unsigned>(attr_uint("sample_rate"));
  _req_channels       = static_cast<unsigned>(attr_uint("channels"));
  _reconnect_delay_ms =
      static_cast<unsigned>(attr_uint("reconnect_delay_ms"));
  _oport_depth        = static_cast<unsigned>(attr_uint("oport_depth"));
  if (_oport_depth == 0) { _oport_depth = 1; }

  allocate_oports(spec().oports.size());
  // DropOldest so a slow downstream consumer cannot stall live
  // audio capture (mirrors rtsp-capture's oport policy).
  set_oport_policy(0, {_oport_depth, OverrunPolicy::DropOldest});
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "device_id", .type = ConfigType::Uint,
   .doc = "avfoundation device index (mutually exclusive with "
          "device_name)"},
  {.key = "device_name", .type = ConfigType::String,
   .doc = "avfoundation device name; case-insensitive substring "
          "match (mutually exclusive with device_id)"},
  {.key = "sample_rate", .type = ConfigType::Uint,
   .doc = "requested input sample rate (0 = device default)",
   .def_uint = 0},
  {.key = "channels", .type = ConfigType::Uint,
   .doc = "requested input channel count (0 = device default)",
   .def_uint = 0},
  {.key = "reconnect_delay_ms", .type = ConfigType::Uint,
   .doc = "backoff before reopen on error (ms)",
   .def_uint = 2000},
  {.key = "oport_depth", .type = ConfigType::Uint,
   .doc = "output ring depth (DropOldest)", .def_uint = 256},
};
const PortSpec kOports[] = {
  {.name = "audio", .doc = "EncodedSegment per audio packet (raw PCM, "
                           "device-native rate); feeds audio-to-pcm",
   .type = &typeid(EncodedSegmentPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "audio-capture",
  .doc       = "Source: captures a microphone / line-in device via "
               "FFmpeg avfoundation and emits one EncodedSegment per "
               "audio packet. Apple-only. 0 iports.",
  .display_name = "Audio Capture",
  .category  = StageCategory::Audio,
  .iports    = {},
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
AudioCaptureStage::spec() const noexcept
{
  return kSpec;
}

int
AudioCaptureStage::probe_device_index_by_name_()
{
  // FFmpeg's avfoundation demuxer logs device listings to stderr
  // when called with `-list_devices true -i ""`. We don't need to
  // spawn a child here -- we can drive the same path via a
  // throw-away AVFormatContext with `list_devices=true` and capture
  // the log lines. The simpler approach used here: spawn
  // /usr/bin/ffmpeg (Homebrew or system) and parse its stderr. This
  // avoids re-routing the FFmpeg log callback for a one-time probe
  // and matches how a user would discover the device themselves.
  // If ffmpeg is missing on PATH, returns -1.
  FILE* p = ::popen(
      "ffmpeg -hide_banner -f avfoundation -list_devices true "
      "-i '' 2>&1", "r");
  if (!p) { return -1; }
  string out;
  char buf[512];
  while (std::fgets(buf, sizeof(buf), p)) {
    out.append(buf);
  }
  ::pclose(p);

  // Parse the audio-devices block. ffmpeg prints something like:
  //   [AVFoundation indev @ 0x...] AVFoundation audio devices:
  //   [AVFoundation indev @ 0x...] [0] Built-in Microphone
  //   [AVFoundation indev @ 0x...] [1] External USB Mic
  string needle_block = "audio devices:";
  auto b = out.find(needle_block);
  if (b == string::npos) { return -1; }

  string lc_target;
  lc_target.reserve(_device_name.size());
  for (char c : _device_name) {
    lc_target.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(c))));
  }

  size_t pos = b;
  while (true) {
    auto eol = out.find('\n', pos);
    string line = (eol == string::npos)
        ? out.substr(pos) : out.substr(pos, eol - pos);
    if (eol == string::npos) { break; }
    pos = eol + 1;

    // Stop on a blank line or a non-AVFoundation tag.
    if (line.empty() || line.find("AVFoundation") == string::npos) {
      // Continue scanning -- in some builds video and audio blocks
      // are interleaved by other log lines; we'll bail when we hit
      // another "indev" block that's clearly not audio. But here we
      // only stop on EOF for robustness.
      if (pos >= out.size()) { break; }
      continue;
    }
    auto lb = line.find('[', line.find(']') + 1);
    auto rb = (lb == string::npos)
        ? string::npos : line.find(']', lb + 1);
    if (lb == string::npos || rb == string::npos) {
      if (pos >= out.size()) { break; }
      continue;
    }
    string idx_s = line.substr(lb + 1, rb - lb - 1);
    string name  = line.substr(rb + 1);
    while (!name.empty()
           && (name.front() == ' ' || name.front() == '\t')) {
      name.erase(name.begin());
    }
    while (!name.empty()
           && (name.back() == '\r' || name.back() == '\n'
               || name.back() == ' ' || name.back() == '\t')) {
      name.pop_back();
    }

    string lc;
    lc.reserve(name.size());
    for (char c : name) {
      lc.push_back(static_cast<char>(
          std::tolower(static_cast<unsigned char>(c))));
    }
    if (lc.find(lc_target) != string::npos) {
      try { return std::stoi(idx_s); }
      catch (...) { return -1; }
    }

    if (pos >= out.size()) { break; }
  }
  return -1;
}

Job
AudioCaptureStage::process(RuntimeContext& ctx)
{
  using namespace std::chrono;

  const FFmpegLibraries* libs = session()->ffmpeg_libraries();
  if (!libs || !libs->valid()) {
    session()->error(fmt(
        "AudioCaptureStage('{}'): FFmpeg libraries unavailable",
        this->id()));
  }
  if (!libs->avdevice().valid()) {
    session()->error(fmt(
        "AudioCaptureStage('{}'): libavdevice not loaded -- install "
        "it (Homebrew ffmpeg ships it as libavdevice.dylib)",
        this->id()));
  }
  // Register avfoundation. Calling more than once is harmless.
  libs->avdevice().api.register_all();

  const AVInputFormat* ifmt =
      libs->avformat().api.find_input_format("avfoundation");
  if (!ifmt) {
    session()->error(fmt(
        "AudioCaptureStage('{}'): av_find_input_format(\"avfoundation\") "
        "returned null", this->id()));
  }

  // Resolve device_name to an index if needed. Done once per
  // process() invocation; if the user hot-swaps the device while the
  // stage runs, the next reconnect cycle (after an EOF) will repeat
  // the probe.
  int resolved_index = -1;
  if (_has_device_id) {
    resolved_index = static_cast<int>(_device_id);
  } else {
    resolved_index = probe_device_index_by_name_();
    if (resolved_index < 0) {
      session()->error(fmt(
          "AudioCaptureStage('{}'): no avfoundation audio device "
          "matched '{}' (ffmpeg -f avfoundation -list_devices true "
          "shows the available indices)",
          this->id(), _device_name));
    }
    session()->info(fmt(
        "AudioCaptureStage('{}'): resolved device_name '{}' to "
        "avfoundation index {}",
        this->id(), _device_name, resolved_index));
  }

  // avfoundation URL grammar is "[VIDEO]:[AUDIO]". For audio-only
  // capture, leave VIDEO blank and put the audio index after the
  // colon: e.g. ":0" selects the device at audio index 0.
  string url = ":" + std::to_string(resolved_index);

  // Stop relay -- mirrors ctx.stop_requested() into a stable atomic
  // the InterruptCtx can poll without going through the
  // RuntimeContext (which doesn't expose its stop atomic at a stable
  // address suitable for FFmpeg's C callback).
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

  const auto& fmt_api = libs->avformat().api;
  const auto& cdc_api = libs->avcodec().api;
  const auto& util_api = libs->avutil().api;

  // Pre-allocate a packet once; reused across all reads.
  AVPacket* pkt = cdc_api.packet_alloc();
  if (!pkt) {
    session()->error(fmt(
        "AudioCaptureStage('{}'): av_packet_alloc failed",
        this->id()));
  }

  // Outer reconnect loop. Each pass: open the device, drain packets
  // until error or stop, then close.
  while (!ctx.stop_requested()) {
    AVFormatContext* ictx = fmt_api.alloc_context();
    if (!ictx) {
      session()->warn(fmt(
          "AudioCaptureStage('{}'): avformat_alloc_context failed",
          this->id()));
      stop_aware_sleep_(ctx, milliseconds(_reconnect_delay_ms));
      continue;
    }
    ictx->interrupt_callback.callback = &interrupt_cb_;
    ictx->interrupt_callback.opaque   = &ic;

    AVDictionary* opts = nullptr;
    if (_req_sample_rate > 0) {
      char b[32];
      std::snprintf(b, sizeof(b), "%u", _req_sample_rate);
      util_api.dict_set(&opts, "sample_rate", b, 0);
    }
    if (_req_channels > 0) {
      char b[32];
      std::snprintf(b, sizeof(b), "%u", _req_channels);
      util_api.dict_set(&opts, "channels", b, 0);
    }

    int rc = fmt_api.open_input(&ictx, url.c_str(),
        const_cast<AVInputFormat*>(ifmt), &opts);
    if (opts) { util_api.dict_free(&opts); }
    if (rc < 0) {
      char ebuf[256] = {0};
      util_api.strerror(rc, ebuf, sizeof(ebuf));
      session()->warn(fmt(
          "AudioCaptureStage('{}'): open_input failed ({}: {}); "
          "retrying in {} ms",
          this->id(), rc, ebuf, _reconnect_delay_ms));
      // ictx is freed by avformat_open_input on error.
      stop_aware_sleep_(ctx, milliseconds(_reconnect_delay_ms));
      continue;
    }

    rc = fmt_api.find_stream_info(ictx, nullptr);
    if (rc < 0) {
      session()->warn(fmt(
          "AudioCaptureStage('{}'): find_stream_info failed ({}); "
          "reopening",
          this->id(), rc));
      fmt_api.close_input(&ictx);
      stop_aware_sleep_(ctx, milliseconds(_reconnect_delay_ms));
      continue;
    }

    int a_idx = -1;
    for (unsigned i = 0; i < ictx->nb_streams; ++i) {
      auto* st = ictx->streams[i];
      if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
        a_idx = static_cast<int>(i);
        break;
      }
    }
    if (a_idx < 0) {
      session()->error(fmt(
          "AudioCaptureStage('{}'): no audio stream on device {}",
          this->id(), url));
      fmt_api.close_input(&ictx);
      co_return;
    }

    auto* a_st  = ictx->streams[a_idx];
    auto* a_par = a_st->codecpar;
    _input_sample_rate = static_cast<unsigned>(a_par->sample_rate);
    _input_channels    = static_cast<unsigned>(a_par->ch_layout.nb_channels);
    _input_codec_id    = static_cast<unsigned>(a_par->codec_id);

    session()->info(fmt(
        "AudioCaptureStage('{}'): capturing device='{}' codec_id={} "
        "sample_rate={} channels={}",
        this->id(), url,
        static_cast<int>(_input_codec_id),
        _input_sample_rate, _input_channels));

    // Inner read loop -- emit one Beat per packet, timestamped at
    // wall-clock arrival.
    while (!ctx.stop_requested()) {
      cdc_api.packet_unref(pkt);
      int read_rc = fmt_api.read_frame(ictx, pkt);
      if (read_rc == AVERROR(EAGAIN)) {
        // avfoundation's audio reader returns EAGAIN whenever its
        // capture queue has no buffer ready yet -- this is the
        // steady-state "no audio in the last few ms" path, not an
        // error. Sleep briefly so we don't burn a core spinning
        // while the device fills the next buffer (~10 ms is well
        // below typical avfoundation buffer durations of 20-40 ms).
        std::this_thread::sleep_for(milliseconds(5));
        continue;
      }
      if (read_rc < 0) {
        // EOF, interrupt, or device error. Reopen unless the user
        // asked us to stop.
        if (!ctx.stop_requested()) {
          char ebuf[256] = {0};
          util_api.strerror(read_rc, ebuf, sizeof(ebuf));
          session()->warn(fmt(
              "AudioCaptureStage('{}'): read_frame failed ({}: {}); "
              "reopening device",
              this->id(), read_rc, ebuf));
        }
        break;
      }
      if (pkt->stream_index != a_idx || pkt->size <= 0) {
        continue;
      }

      // Timestamp: avfoundation's clock differs from wall clock; we
      // anchor at host wall-clock at packet arrival and derive
      // duration from the packet's PTS unit when available.
      auto now = system_clock::now();
      int64_t dur_us = 0;
      if (pkt->duration > 0 && a_st->time_base.den > 0) {
        // duration (in time_base units) * 1e6 * num / den
        dur_us = static_cast<int64_t>(
            (static_cast<__int128>(pkt->duration) * 1'000'000
             * a_st->time_base.num) / a_st->time_base.den);
      } else if (_input_sample_rate > 0) {
        // Fall back to bytes / (bytes_per_sample * channels * rate).
        // PCM_S16LE = 2 bytes/sample, F32LE = 4, etc. Without an
        // exact map we approximate using 16-bit -- it's only a
        // hint, downstream resamples to its own clock.
        int bps = 2;
        dur_us = static_cast<int64_t>(
            (static_cast<int64_t>(pkt->size) * 1'000'000)
            / (static_cast<int64_t>(bps)
               * std::max(1u, _input_channels)
               * static_cast<int64_t>(_input_sample_rate)));
      }

      EncodedSegment es;
      es.kind        = EncodedSegment::Kind::Audio;
      es.start_utc   = now;
      es.end_utc     = now + microseconds(dur_us);
      es.duration_us = dur_us;
      es.codec_id    = _input_codec_id;
      es.sample_rate = _input_sample_rate;
      es.channels    = _input_channels;
      es.data.assign(pkt->data, pkt->data + pkt->size);

      ++_packets_emitted;
      co_await ctx.write(0,
          make_payload<EncodedSegmentPayload>(std::move(es)));
    }

    fmt_api.close_input(&ictx);
    if (ctx.stop_requested()) { break; }
    // Brief pause before reopening so we don't busy-loop on a
    // persistently failing device.
    stop_aware_sleep_(ctx, milliseconds(_reconnect_delay_ms));
  }

  cdc_api.packet_free(&pkt);
  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(AudioCaptureStage)
VPIPE_REGISTER_SPEC(AudioCaptureStage, kSpec)

}
