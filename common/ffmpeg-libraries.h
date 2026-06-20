#ifndef FFMPEG_LIBRARIES_H
#define FFMPEG_LIBRARIES_H

#include "common/library-handle.h"
#include "common/session-member.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/buffer.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/log.h>
#include <libavutil/mathematics.h>
#include <libavutil/mem.h>
#include <libavutil/rational.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace vpipe {

// Per-library wrappers around LibraryHandle that pre-resolve a curated
// set of FFmpeg functions into a typed `api` struct. Each wrapper:
//   * probes a per-library list of candidate sonames and dlopens the
//     first one that loads;
//   * resolves the curated functions via require_symbol (Required mode
//     throws on a missing symbol, surfacing ABI mismatches at startup);
//   * exposes version_major/minor/micro derived from the AV_VERSION_INT
//     bit packing (api.version() gives the raw packed unsigned).
//
// Stages can call any function in `api` directly. Symbols outside the
// curated set are still reachable through LibraryHandle's
// require_symbol_as<T>(...) on the same instance.

class LibAvUtil final : public LibraryHandle {
public:
  explicit LibAvUtil(const SessionContextIntf* session,
                     LoadMode                  mode = LoadMode::Required);

  unsigned version_major() const noexcept;
  unsigned version_minor() const noexcept;
  unsigned version_micro() const noexcept;

  struct Api {
    decltype(&::avutil_version)         version;
    decltype(&::av_log_set_level)       log_set_level;
    decltype(&::av_log_set_callback)    log_set_callback;
    decltype(&::av_strerror)            strerror;
    decltype(&::av_dict_set)            dict_set;
    decltype(&::av_dict_get)            dict_get;
    decltype(&::av_dict_free)           dict_free;
    decltype(&::av_frame_alloc)         frame_alloc;
    decltype(&::av_frame_free)          frame_free;
    decltype(&::av_frame_unref)         frame_unref;
    decltype(&::av_frame_get_buffer)    frame_get_buffer;
    decltype(&::av_rescale_q)           rescale_q;
    decltype(&::av_rescale_q_rnd)       rescale_q_rnd;
    decltype(&::av_get_sample_fmt_name) get_sample_fmt_name;
    decltype(&::av_malloc)              malloc;
    decltype(&::av_freep)               freep;
    decltype(&::av_buffer_ref)          buffer_ref;
    decltype(&::av_buffer_unref)        buffer_unref;
    decltype(&::av_hwdevice_ctx_create) hwdevice_ctx_create;
    decltype(&::av_hwframe_transfer_data) hwframe_transfer_data;
  } api{};
};

class LibAvCodec final : public LibraryHandle {
public:
  explicit LibAvCodec(const SessionContextIntf* session,
                      LoadMode                  mode = LoadMode::Required);

  unsigned version_major() const noexcept;
  unsigned version_minor() const noexcept;
  unsigned version_micro() const noexcept;

  struct Api {
    decltype(&::avcodec_version)               version;
    decltype(&::avcodec_find_decoder)          find_decoder;
    decltype(&::avcodec_find_encoder)          find_encoder;
    decltype(&::avcodec_find_encoder_by_name)  find_encoder_by_name;
    decltype(&::avcodec_alloc_context3)        alloc_context3;
    decltype(&::avcodec_free_context)          free_context;
    decltype(&::avcodec_parameters_alloc)      parameters_alloc;
    decltype(&::avcodec_parameters_free)       parameters_free;
    decltype(&::avcodec_parameters_to_context) parameters_to_context;
    decltype(&::avcodec_parameters_from_context) parameters_from_context;
    decltype(&::avcodec_parameters_copy)       parameters_copy;
    decltype(&::avcodec_open2)                 open2;
    decltype(&::avcodec_send_packet)           send_packet;
    decltype(&::avcodec_receive_frame)         receive_frame;
    decltype(&::avcodec_send_frame)            send_frame;
    decltype(&::avcodec_receive_packet)        receive_packet;
    decltype(&::avcodec_flush_buffers)         flush_buffers;
    decltype(&::av_packet_alloc)               packet_alloc;
    decltype(&::av_packet_free)                packet_free;
    decltype(&::av_packet_unref)               packet_unref;
    decltype(&::av_packet_rescale_ts)          packet_rescale_ts;
    decltype(&::avcodec_get_hw_config)         get_hw_config;
  } api{};
};

class LibAvFormat final : public LibraryHandle {
public:
  explicit LibAvFormat(const SessionContextIntf* session,
                       LoadMode                  mode = LoadMode::Required);

  unsigned version_major() const noexcept;
  unsigned version_minor() const noexcept;
  unsigned version_micro() const noexcept;

  struct Api {
    decltype(&::avformat_version)               version;
    decltype(&::avformat_network_init)          network_init;
    decltype(&::avformat_network_deinit)        network_deinit;
    decltype(&::avformat_alloc_context)         alloc_context;
    decltype(&::avformat_free_context)          free_context;
    decltype(&::avformat_open_input)            open_input;
    decltype(&::avformat_close_input)           close_input;
    decltype(&::avformat_find_stream_info)      find_stream_info;
    decltype(&::av_find_input_format)           find_input_format;
    decltype(&::av_read_frame)                  read_frame;
    decltype(&::avformat_alloc_output_context2) alloc_output_context2;
    decltype(&::avformat_new_stream)            new_stream;
    decltype(&::avformat_write_header)          write_header;
    decltype(&::av_write_trailer)               write_trailer;
    decltype(&::av_interleaved_write_frame)     interleaved_write_frame;
    decltype(&::avio_open)                      avio_open;
    decltype(&::avio_closep)                    avio_closep;
    decltype(&::avio_alloc_context)             avio_alloc_context;
    decltype(&::avio_context_free)              avio_context_free;
    decltype(&::avio_flush)                     avio_flush;
    decltype(&::av_dump_format)                 dump_format;
  } api{};
};

class LibSwResample final : public LibraryHandle {
public:
  explicit LibSwResample(const SessionContextIntf* session,
                         LoadMode                  mode = LoadMode::Required);

  unsigned version_major() const noexcept;
  unsigned version_minor() const noexcept;
  unsigned version_micro() const noexcept;

  struct Api {
    decltype(&::swresample_version)    version;
    decltype(&::swr_alloc)             alloc;
    decltype(&::swr_alloc_set_opts2)   alloc_set_opts2;
    decltype(&::swr_init)              init;
    decltype(&::swr_free)              free;
    decltype(&::swr_convert)           convert;
    decltype(&::swr_get_delay)         get_delay;
  } api{};
};

class LibSwScale final : public LibraryHandle {
public:
  explicit LibSwScale(const SessionContextIntf* session,
                      LoadMode                  mode = LoadMode::Required);

  unsigned version_major() const noexcept;
  unsigned version_minor() const noexcept;
  unsigned version_micro() const noexcept;

  struct Api {
    decltype(&::swscale_version)      version;
    decltype(&::sws_getContext)       get_context;
    decltype(&::sws_getCachedContext) get_cached_context;
    decltype(&::sws_scale)            scale;
    decltype(&::sws_freeContext)      free_context;
  } api{};
};

// libavdevice -- supplies platform input device demuxers (e.g.
// "avfoundation" on macOS, "v4l2" / "alsa" on Linux). Loaded as
// Optional in the FFmpegLibraries composite so installations without
// libavdevice keep working; stages that need an input device should
// check `valid()` before calling api.
class LibAvDevice final : public LibraryHandle {
public:
  explicit LibAvDevice(const SessionContextIntf* session,
                       LoadMode                  mode = LoadMode::Optional);

  unsigned version_major() const noexcept;
  unsigned version_minor() const noexcept;
  unsigned version_micro() const noexcept;

  struct Api {
    decltype(&::avdevice_version)      version;
    decltype(&::avdevice_register_all) register_all;
  } api{};
};

// Composite owner: load all four libraries together. Loaded in
// dependency order (avutil first, then avcodec / swresample, then
// avformat). valid() == true iff every sub-handle loaded.
//
// Side effect: construction installs a process-wide log callback
// that funnels FFmpeg's own log emissions through the session's
// log_verbose() channel and bumps FFmpeg's global log level to
// AV_LOG_VERBOSE. The most recently constructed instance wins as
// the routing target; multi-Session apps in the same process see
// the latest session's verbose channel.
class FFmpegLibraries final : public SessionMember {
public:
  explicit FFmpegLibraries(const SessionContextIntf* session,
                           LibraryHandle::LoadMode mode
                             = LibraryHandle::LoadMode::Required);
  ~FFmpegLibraries() override;

  FFmpegLibraries(const FFmpegLibraries&)            = delete;
  FFmpegLibraries& operator=(const FFmpegLibraries&) = delete;

  bool valid() const noexcept;

  const LibAvUtil&     avutil()     const noexcept { return _avutil; }
  const LibAvCodec&    avcodec()    const noexcept { return _avcodec; }
  const LibAvFormat&   avformat()   const noexcept { return _avformat; }
  const LibSwResample& swresample() const noexcept { return _swresample; }
  const LibSwScale&    swscale()    const noexcept { return _swscale; }
  // Optional; check `valid()` on the returned handle.
  const LibAvDevice&   avdevice()   const noexcept { return _avdevice; }

private:
  // Declaration order matters: avutil is the lowest-level dependency
  // and gets constructed first; the others come after.
  LibAvUtil     _avutil;
  LibAvCodec    _avcodec;
  LibSwResample _swresample;
  LibSwScale    _swscale;
  LibAvFormat   _avformat;
  LibAvDevice   _avdevice;
};

}

#endif
