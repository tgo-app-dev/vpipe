#ifndef FFMPEG_LOG_TAP_H
#define FFMPEG_LOG_TAP_H

#include <cstdint>
#include <functional>
#include <string_view>

namespace vpipe {

// Per-message tap fired by FFmpegLibraries' global log callback after
// the routine session->log_verbose() dispatch. `level` follows
// FFmpeg's AV_LOG_* convention (PANIC=0 ... TRACE=56); `tag` is the
// AVClass class_name (e.g. "rtsp", "rtp", "h264"); `msg` is the
// already-formatted message body without the trailing newline.
//
// Tap callbacks MUST be short and non-blocking: they run on the
// thread that emitted the FFmpeg log, including FFmpeg's internal
// I/O threads. They must not call further FFmpeg APIs and must not
// log through the session (that would re-enter this dispatch).
using LogTap =
    std::function<void(int                level,
                       std::string_view   tag,
                       std::string_view   msg)>;

// Opaque handle returned by install_log_tap; pass to remove_log_tap
// to detach. Cheap to copy.
struct LogTapHandle {
  std::uint64_t id = 0;
  explicit operator bool() const noexcept { return id != 0; }
};

// Add a tap. Thread-safe; can be called concurrently with dispatch
// and with other install / remove calls. Returns a handle with a
// non-zero id on success.
LogTapHandle install_log_tap(LogTap tap);

// Remove a previously-installed tap. Idempotent; safe to call with
// a default-constructed (zero-id) handle. Synchronizes with dispatch:
// once it returns, the tap is neither executing on any thread nor
// reachable by a later dispatch, so the caller may immediately destroy
// whatever state the tap captured (a stack-local, `this`, ...).
void remove_log_tap(LogTapHandle h);

// Called by the FFmpegLibraries global log callback. Iterates the
// currently-installed taps and forwards the message to each. Runs the
// callbacks under a SHARED lock of the registry (install / remove take
// the exclusive lock), so dispatches from multiple FFmpeg threads run
// concurrently while remove_log_tap waits for any in-flight dispatch of
// the tap it removes. Because the callbacks run under that lock, a tap
// that blocks will stall a concurrent remove -- which is why taps MUST
// be short and non-blocking (see LogTap above).
void ffmpeg_log_tap_dispatch(int              level,
                             std::string_view tag,
                             std::string_view msg);

}

#endif
