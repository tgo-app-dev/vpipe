#include "common/ffmpeg-libraries.h"
#include "common/ffmpeg-log-tap.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <dlfcn.h>
#include <string>
#include <string_view>

extern "C" {
#include <libavutil/log.h>
}

using namespace std;

namespace vpipe {

namespace {

// Process-wide pointer to the SessionContextIntf currently chosen as
// the destination for FFmpeg log lines. FFmpegLibraries' constructor
// publishes itself here; its destructor CAS-clears. Multi-instance
// in the same process: latest construction wins, which covers the
// usual single-Session shape cleanly.
std::atomic<const SessionContextIntf*> g_ff_log_target{nullptr};

void
ffmpeg_log_cb_(void* avcl, int level, const char* fmt_str, va_list vl)
{
  // Drop everything below VERBOSE (i.e. drop DEBUG and TRACE).
  if (level > AV_LOG_VERBOSE) {
    return;
  }

  char buf[1024];
  va_list vl_copy;
  va_copy(vl_copy, vl);
  int n = std::vsnprintf(buf, sizeof(buf), fmt_str, vl_copy);
  va_end(vl_copy);
  if (n <= 0) {
    return;
  }
  if (static_cast<size_t>(n) >= sizeof(buf)) {
    n = static_cast<int>(sizeof(buf)) - 1;
  }
  // Trim trailing newline / CR -- our delegate already adds one.
  while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) {
    buf[--n] = '\0';
  }
  if (n == 0) {
    return;
  }

  const char* class_name = "ffmpeg";
  if (avcl) {
    AVClass* const* pcl = static_cast<AVClass* const*>(avcl);
    if (*pcl && (*pcl)->class_name) {
      class_name = (*pcl)->class_name;
    }
  }

  std::string_view msg_view(buf, static_cast<size_t>(n));

  const SessionContextIntf* s =
    g_ff_log_target.load(std::memory_order_acquire);
  if (s) {
    s->log_verbose(fmt("[ffmpeg:{}] {}", class_name, msg_view));
  }

  // Per-stage observers (e.g. rtsp-capture watching for RTP errors).
  // Dispatched after the session log call so the session sees the
  // raw message regardless of whether any tap reacts. Taps must not
  // log through the session -- that would re-enter this callback.
  ffmpeg_log_tap_dispatch(level, class_name, msg_view);
}

// Per-library candidate list: bare sonames first (so the OS dynamic
// loader handles search paths normally), then versioned filenames
// covering FFmpeg 4..7 ABIs, then common Homebrew / system absolute
// paths. The first candidate that dlopens wins.
//
// We probe with RTLD_NOW | RTLD_LOCAL (matches LibraryHandle's
// default) so a successful probe means the lib is fully usable. The
// transient handle is closed and the picked path is then handed to
// LibraryHandle, which dlopens it again -- the OS's reference count
// makes that a no-op in terms of actual loading.

constexpr const char* kAvUtilCandidates[] = {
  "libavutil.dylib",
  "libavutil.59.dylib",
  "libavutil.58.dylib",
  "libavutil.57.dylib",
  "libavutil.56.dylib",
  "/opt/homebrew/lib/libavutil.dylib",
  "/usr/local/lib/libavutil.dylib",
  "libavutil.so",
  "libavutil.so.59",
  "libavutil.so.58",
  "libavutil.so.57",
  "libavutil.so.56",
};

constexpr const char* kAvCodecCandidates[] = {
  "libavcodec.dylib",
  "libavcodec.61.dylib",
  "libavcodec.60.dylib",
  "libavcodec.59.dylib",
  "libavcodec.58.dylib",
  "/opt/homebrew/lib/libavcodec.dylib",
  "/usr/local/lib/libavcodec.dylib",
  "libavcodec.so",
  "libavcodec.so.61",
  "libavcodec.so.60",
  "libavcodec.so.59",
  "libavcodec.so.58",
};

constexpr const char* kAvFormatCandidates[] = {
  "libavformat.dylib",
  "libavformat.61.dylib",
  "libavformat.60.dylib",
  "libavformat.59.dylib",
  "libavformat.58.dylib",
  "/opt/homebrew/lib/libavformat.dylib",
  "/usr/local/lib/libavformat.dylib",
  "libavformat.so",
  "libavformat.so.61",
  "libavformat.so.60",
  "libavformat.so.59",
  "libavformat.so.58",
};

constexpr const char* kAvDeviceCandidates[] = {
  "libavdevice.dylib",
  "libavdevice.61.dylib",
  "libavdevice.60.dylib",
  "libavdevice.59.dylib",
  "libavdevice.58.dylib",
  "/opt/homebrew/lib/libavdevice.dylib",
  "/usr/local/lib/libavdevice.dylib",
  "libavdevice.so",
  "libavdevice.so.61",
  "libavdevice.so.60",
  "libavdevice.so.59",
  "libavdevice.so.58",
};

constexpr const char* kSwResampleCandidates[] = {
  "libswresample.dylib",
  "libswresample.5.dylib",
  "libswresample.4.dylib",
  "libswresample.3.dylib",
  "/opt/homebrew/lib/libswresample.dylib",
  "/usr/local/lib/libswresample.dylib",
  "libswresample.so",
  "libswresample.so.5",
  "libswresample.so.4",
  "libswresample.so.3",
};

constexpr const char* kSwScaleCandidates[] = {
  "libswscale.dylib",
  "libswscale.8.dylib",
  "libswscale.7.dylib",
  "libswscale.6.dylib",
  "libswscale.5.dylib",
  "/opt/homebrew/lib/libswscale.dylib",
  "/usr/local/lib/libswscale.dylib",
  "libswscale.so",
  "libswscale.so.8",
  "libswscale.so.7",
  "libswscale.so.6",
  "libswscale.so.5",
};

template <size_t N>
string
find_first_loadable_(const SessionContextIntf*  session,
                     const char* const          (&candidates)[N],
                     string_view                lib_name)
{
  for (const char* c : candidates) {
    void* h = ::dlopen(c, RTLD_NOW | RTLD_LOCAL);
    if (h) {
      ::dlclose(h);
      session->info(fmt("{}: using '{}'", lib_name, c));
      return string(c);
    }
  }
  // Hand the canonical bare soname back so LibraryHandle's failure
  // path produces a meaningful error message.
  return string(candidates[0]);
}

}

// ---------------------------------------------------------------------
// LibAvUtil
// ---------------------------------------------------------------------

LibAvUtil::LibAvUtil(const SessionContextIntf* s, LoadMode mode)
  : LibraryHandle(s, find_first_loadable_(s, kAvUtilCandidates,
                                          "libavutil"),
                  mode)
{
  if (!valid()) {
    return;
  }

#define VPIPE_RESOLVE(F, N)                                            \
  api.F = reinterpret_cast<decltype(api.F)>(this->require_symbol(N))

  VPIPE_RESOLVE(version,             "avutil_version");
  VPIPE_RESOLVE(log_set_level,       "av_log_set_level");
  VPIPE_RESOLVE(log_set_callback,    "av_log_set_callback");
  VPIPE_RESOLVE(strerror,            "av_strerror");
  VPIPE_RESOLVE(dict_set,            "av_dict_set");
  VPIPE_RESOLVE(dict_get,            "av_dict_get");
  VPIPE_RESOLVE(dict_free,           "av_dict_free");
  VPIPE_RESOLVE(frame_alloc,         "av_frame_alloc");
  VPIPE_RESOLVE(frame_free,          "av_frame_free");
  VPIPE_RESOLVE(frame_unref,         "av_frame_unref");
  VPIPE_RESOLVE(frame_get_buffer,    "av_frame_get_buffer");
  VPIPE_RESOLVE(rescale_q,           "av_rescale_q");
  VPIPE_RESOLVE(rescale_q_rnd,       "av_rescale_q_rnd");
  VPIPE_RESOLVE(get_sample_fmt_name, "av_get_sample_fmt_name");
  VPIPE_RESOLVE(malloc,              "av_malloc");
  VPIPE_RESOLVE(freep,               "av_freep");
  VPIPE_RESOLVE(buffer_ref,          "av_buffer_ref");
  VPIPE_RESOLVE(buffer_unref,        "av_buffer_unref");
  VPIPE_RESOLVE(hwdevice_ctx_create, "av_hwdevice_ctx_create");
  VPIPE_RESOLVE(hwframe_transfer_data, "av_hwframe_transfer_data");

#undef VPIPE_RESOLVE
}

unsigned
LibAvUtil::version_major() const noexcept
{
  return api.version ? (api.version() >> 16) & 0xff : 0;
}

unsigned
LibAvUtil::version_minor() const noexcept
{
  return api.version ? (api.version() >> 8) & 0xff : 0;
}

unsigned
LibAvUtil::version_micro() const noexcept
{
  return api.version ? api.version() & 0xff : 0;
}

// ---------------------------------------------------------------------
// LibAvCodec
// ---------------------------------------------------------------------

LibAvCodec::LibAvCodec(const SessionContextIntf* s, LoadMode mode)
  : LibraryHandle(s, find_first_loadable_(s, kAvCodecCandidates,
                                          "libavcodec"),
                  mode)
{
  if (!valid()) {
    return;
  }

#define VPIPE_RESOLVE(F, N)                                            \
  api.F = reinterpret_cast<decltype(api.F)>(this->require_symbol(N))

  VPIPE_RESOLVE(version,                  "avcodec_version");
  VPIPE_RESOLVE(find_decoder,             "avcodec_find_decoder");
  VPIPE_RESOLVE(find_encoder,             "avcodec_find_encoder");
  VPIPE_RESOLVE(find_encoder_by_name,     "avcodec_find_encoder_by_name");
  VPIPE_RESOLVE(alloc_context3,           "avcodec_alloc_context3");
  VPIPE_RESOLVE(free_context,             "avcodec_free_context");
  VPIPE_RESOLVE(parameters_alloc,         "avcodec_parameters_alloc");
  VPIPE_RESOLVE(parameters_free,          "avcodec_parameters_free");
  VPIPE_RESOLVE(parameters_to_context,    "avcodec_parameters_to_context");
  VPIPE_RESOLVE(parameters_from_context,  "avcodec_parameters_from_context");
  VPIPE_RESOLVE(parameters_copy,          "avcodec_parameters_copy");
  VPIPE_RESOLVE(open2,                    "avcodec_open2");
  VPIPE_RESOLVE(send_packet,              "avcodec_send_packet");
  VPIPE_RESOLVE(receive_frame,            "avcodec_receive_frame");
  VPIPE_RESOLVE(send_frame,               "avcodec_send_frame");
  VPIPE_RESOLVE(receive_packet,           "avcodec_receive_packet");
  VPIPE_RESOLVE(flush_buffers,            "avcodec_flush_buffers");
  VPIPE_RESOLVE(packet_alloc,             "av_packet_alloc");
  VPIPE_RESOLVE(packet_free,              "av_packet_free");
  VPIPE_RESOLVE(packet_unref,             "av_packet_unref");
  VPIPE_RESOLVE(packet_rescale_ts,        "av_packet_rescale_ts");
  VPIPE_RESOLVE(get_hw_config,            "avcodec_get_hw_config");

#undef VPIPE_RESOLVE
}

unsigned
LibAvCodec::version_major() const noexcept
{
  return api.version ? (api.version() >> 16) & 0xff : 0;
}

unsigned
LibAvCodec::version_minor() const noexcept
{
  return api.version ? (api.version() >> 8) & 0xff : 0;
}

unsigned
LibAvCodec::version_micro() const noexcept
{
  return api.version ? api.version() & 0xff : 0;
}

// ---------------------------------------------------------------------
// LibAvFormat
// ---------------------------------------------------------------------

LibAvFormat::LibAvFormat(const SessionContextIntf* s, LoadMode mode)
  : LibraryHandle(s, find_first_loadable_(s, kAvFormatCandidates,
                                          "libavformat"),
                  mode)
{
  if (!valid()) {
    return;
  }

#define VPIPE_RESOLVE(F, N)                                            \
  api.F = reinterpret_cast<decltype(api.F)>(this->require_symbol(N))

  VPIPE_RESOLVE(version,                  "avformat_version");
  VPIPE_RESOLVE(network_init,             "avformat_network_init");
  VPIPE_RESOLVE(network_deinit,           "avformat_network_deinit");
  VPIPE_RESOLVE(alloc_context,            "avformat_alloc_context");
  VPIPE_RESOLVE(free_context,             "avformat_free_context");
  VPIPE_RESOLVE(open_input,               "avformat_open_input");
  VPIPE_RESOLVE(close_input,              "avformat_close_input");
  VPIPE_RESOLVE(find_stream_info,         "avformat_find_stream_info");
  VPIPE_RESOLVE(find_input_format,        "av_find_input_format");
  VPIPE_RESOLVE(read_frame,               "av_read_frame");
  VPIPE_RESOLVE(alloc_output_context2,    "avformat_alloc_output_context2");
  VPIPE_RESOLVE(new_stream,               "avformat_new_stream");
  VPIPE_RESOLVE(write_header,             "avformat_write_header");
  VPIPE_RESOLVE(write_trailer,            "av_write_trailer");
  VPIPE_RESOLVE(interleaved_write_frame,  "av_interleaved_write_frame");
  VPIPE_RESOLVE(avio_open,                "avio_open");
  VPIPE_RESOLVE(avio_closep,              "avio_closep");
  VPIPE_RESOLVE(avio_alloc_context,       "avio_alloc_context");
  VPIPE_RESOLVE(avio_context_free,        "avio_context_free");
  VPIPE_RESOLVE(avio_flush,               "avio_flush");
  VPIPE_RESOLVE(dump_format,              "av_dump_format");

#undef VPIPE_RESOLVE
}

unsigned
LibAvFormat::version_major() const noexcept
{
  return api.version ? (api.version() >> 16) & 0xff : 0;
}

unsigned
LibAvFormat::version_minor() const noexcept
{
  return api.version ? (api.version() >> 8) & 0xff : 0;
}

unsigned
LibAvFormat::version_micro() const noexcept
{
  return api.version ? api.version() & 0xff : 0;
}

// ---------------------------------------------------------------------
// LibSwResample
// ---------------------------------------------------------------------

LibSwResample::LibSwResample(const SessionContextIntf* s, LoadMode mode)
  : LibraryHandle(s, find_first_loadable_(s, kSwResampleCandidates,
                                          "libswresample"),
                  mode)
{
  if (!valid()) {
    return;
  }

#define VPIPE_RESOLVE(F, N)                                            \
  api.F = reinterpret_cast<decltype(api.F)>(this->require_symbol(N))

  VPIPE_RESOLVE(version,         "swresample_version");
  VPIPE_RESOLVE(alloc,           "swr_alloc");
  VPIPE_RESOLVE(alloc_set_opts2, "swr_alloc_set_opts2");
  VPIPE_RESOLVE(init,            "swr_init");
  VPIPE_RESOLVE(free,            "swr_free");
  VPIPE_RESOLVE(convert,         "swr_convert");
  VPIPE_RESOLVE(get_delay,       "swr_get_delay");

#undef VPIPE_RESOLVE
}

unsigned
LibSwResample::version_major() const noexcept
{
  return api.version ? (api.version() >> 16) & 0xff : 0;
}

unsigned
LibSwResample::version_minor() const noexcept
{
  return api.version ? (api.version() >> 8) & 0xff : 0;
}

unsigned
LibSwResample::version_micro() const noexcept
{
  return api.version ? api.version() & 0xff : 0;
}

// ---------------------------------------------------------------------
// LibSwScale
// ---------------------------------------------------------------------

LibSwScale::LibSwScale(const SessionContextIntf* s, LoadMode mode)
  : LibraryHandle(s, find_first_loadable_(s, kSwScaleCandidates,
                                          "libswscale"),
                  mode)
{
  if (!valid()) {
    return;
  }

#define VPIPE_RESOLVE(F, N)                                            \
  api.F = reinterpret_cast<decltype(api.F)>(this->require_symbol(N))

  VPIPE_RESOLVE(version,            "swscale_version");
  VPIPE_RESOLVE(get_context,        "sws_getContext");
  VPIPE_RESOLVE(get_cached_context, "sws_getCachedContext");
  VPIPE_RESOLVE(scale,              "sws_scale");
  VPIPE_RESOLVE(free_context,       "sws_freeContext");

#undef VPIPE_RESOLVE
}

unsigned
LibSwScale::version_major() const noexcept
{
  return api.version ? (api.version() >> 16) & 0xff : 0;
}

unsigned
LibSwScale::version_minor() const noexcept
{
  return api.version ? (api.version() >> 8) & 0xff : 0;
}

unsigned
LibSwScale::version_micro() const noexcept
{
  return api.version ? api.version() & 0xff : 0;
}

// ---------------------------------------------------------------------
// LibAvDevice
// ---------------------------------------------------------------------

LibAvDevice::LibAvDevice(const SessionContextIntf* s, LoadMode mode)
  : LibraryHandle(s, find_first_loadable_(s, kAvDeviceCandidates,
                                          "libavdevice"),
                  mode)
{
  if (!valid()) {
    return;
  }

#define VPIPE_RESOLVE(F, N)                                            \
  api.F = reinterpret_cast<decltype(api.F)>(this->require_symbol(N))

  VPIPE_RESOLVE(version,      "avdevice_version");
  VPIPE_RESOLVE(register_all, "avdevice_register_all");

#undef VPIPE_RESOLVE
}

unsigned
LibAvDevice::version_major() const noexcept
{
  return api.version ? (api.version() >> 16) & 0xff : 0;
}

unsigned
LibAvDevice::version_minor() const noexcept
{
  return api.version ? (api.version() >> 8) & 0xff : 0;
}

unsigned
LibAvDevice::version_micro() const noexcept
{
  return api.version ? api.version() & 0xff : 0;
}

// ---------------------------------------------------------------------
// FFmpegLibraries
// ---------------------------------------------------------------------

FFmpegLibraries::FFmpegLibraries(const SessionContextIntf* s,
                                 LibraryHandle::LoadMode   mode)
  : SessionMember(s)
  , _avutil(s, mode)
  , _avcodec(s, mode)
  , _swresample(s, mode)
  , _swscale(s, mode)
  , _avformat(s, mode)
  // avdevice is always loaded Optional regardless of the outer mode --
  // it isn't required for the existing rtsp/file pipelines, and many
  // bundled FFmpeg installs ship without it.
  , _avdevice(s, LibraryHandle::LoadMode::Optional)
{
  // Funnel FFmpeg's own log emissions into session->log_verbose().
  // Process-wide effects: bumps the global log threshold to VERBOSE
  // (DEBUG / TRACE still drop in our callback) and installs the
  // routing trampoline. The current Session is published into the
  // global pointer; latest-constructed wins.
  if (_avutil.valid() && _avutil.api.log_set_callback) {
    _avutil.api.log_set_level(AV_LOG_VERBOSE);
    _avutil.api.log_set_callback(&ffmpeg_log_cb_);
    g_ff_log_target.store(s, std::memory_order_release);
  }
}

FFmpegLibraries::~FFmpegLibraries()
{
  // Only clear the global pointer if it still names our session;
  // another live FFmpegLibraries (potentially on a different
  // session) may have published itself afterwards.
  const SessionContextIntf* expected = session();
  g_ff_log_target.compare_exchange_strong(expected, nullptr,
                                          std::memory_order_acq_rel,
                                          std::memory_order_relaxed);
  // Leave the callback installed: it is a leaf-safe trampoline that
  // just drops messages when no target is published.
}

bool
FFmpegLibraries::valid() const noexcept
{
  return _avutil.valid()
      && _avcodec.valid()
      && _swresample.valid()
      && _swscale.valid()
      && _avformat.valid();
}

}
