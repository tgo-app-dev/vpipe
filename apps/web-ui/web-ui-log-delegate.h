#ifndef WEBUI_WEB_UI_LOG_DELEGATE_H
#define WEBUI_WEB_UI_LOG_DELEGATE_H

#include "interfaces/log-delegate-intf.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace vpipe::webui {

// LogDelegateIntf that diverts a Session's diagnostic log stream (the
// log_* channels: debug / verbose / normal / always) to the browser's
// "Session Log" view, instead of writing it to stdout. It is the log
// counterpart of WebUiDelegate (which handles the user-facing
// error/warn/info + getline channels): error/warn/info still flow to
// the User I/O view; this captures the leveled log() stream.
//
// Lines are kept in an in-memory ring the browser polls
// (GET /api/log/console?since=N). A live, atomic threshold filters at
// capture time: a message is retained iff
//     level == LogLevel::Always || level <= threshold
// (smaller LogLevel ints are more severe). Filtering at capture -- not
// render -- is what makes a threshold change affect only FUTURE
// messages: already-captured lines are never re-filtered, and messages
// dropped under a stricter threshold are simply never stored.
//
// Threading: log() runs on pool worker threads; the API accessors
// (console_since / clear_console / max_log / set_max_log) run on the
// HTTP server thread. The ring is guarded by one mutex; the threshold
// is a lock-free atomic so set_threshold() never blocks a worker.
class WebUiLogDelegate final : public vpipe::LogDelegateIntf {
public:
  explicit WebUiLogDelegate(
      vpipe::LogLevel threshold = vpipe::LogLevel::Normal);
  ~WebUiLogDelegate() override = default;

  // ---- LogDelegateIntf (worker threads) ----------------------------
  void log(vpipe::LogLevel, const vpipe::VpipeFormat&) override;

  void set_threshold(vpipe::LogLevel level) override
  {
    _threshold.store(level, std::memory_order_relaxed);
  }
  vpipe::LogLevel threshold() const noexcept override
  {
    return _threshold.load(std::memory_order_relaxed);
  }

  // ---- API surface (HTTP thread) -----------------------------------
  struct LogLine {
    std::uint64_t seq = 0;
    std::string   level;   // lowercase: "error"|..|"debug"|"always"
    std::string   text;
  };

  // Lines newer than `since` (seq > since), in seq order. `*latest_seq`
  // is the highest seq held (or, when the batch is capped by `limit`,
  // the highest seq returned so the next poll resumes after it).
  // `limit` bounds the response in entries; 0 means unlimited.
  std::vector<LogLine>
  console_since(std::uint64_t since, std::uint64_t* latest_seq,
                std::size_t limit = 0) const;

  // Drop all log history (the "Clear" button).
  void clear_console();

  // Ring cap (terminal-style scrollback bound). Lines beyond this count
  // are dropped from the front (oldest first). Default kDefaultMaxLog;
  // the Settings page lets the user raise/lower it. Trimming on set is
  // immediate.
  static constexpr std::size_t kDefaultMaxLog = 8192;
  static constexpr std::size_t kMinMaxLog     = 16;
  static constexpr std::size_t kMaxMaxLog     = 262'144;
  std::size_t max_log() const;
  void        set_max_log(std::size_t n);

private:
  // Drop the single oldest entry when over the cap. Caller holds _mu.
  void trim_();

  std::atomic<vpipe::LogLevel> _threshold;

  mutable std::mutex   _mu;
  std::deque<LogLine>  _log;
  std::uint64_t        _next_seq = 1;
  std::size_t          _max_log  = kDefaultMaxLog;
};

}

#endif
