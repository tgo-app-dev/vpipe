#ifndef LOG_DELEGATE_INTF_H
#define LOG_DELEGATE_INTF_H

#include <string_view>

namespace vpipe {

struct VpipeFormat;

// Severity / verbosity ordering. Smaller numeric value = more severe.
// Always is a sentinel: delegates that honour a threshold must emit
// regardless of the threshold when level == Always.
enum class LogLevel : int {
  Error   = 0,
  Warn    = 1,
  Info    = 2,
  Normal  = 3,
  Verbose = 4,
  Debug   = 5,
  Always  = 6,
};

// Uppercase short name: "ERROR", "WARN", "INFO", "NORMAL",
// "VERBOSE", "DEBUG", "ALWAYS". Returns "?" for out-of-range values.
const char* to_cstr(LogLevel) noexcept;

// Lowercase parsing: "error", "warn", "info", "normal", "verbose",
// "debug". "always" is also accepted. Empty / unknown -> def.
LogLevel parse_log_level(std::string_view name,
                         LogLevel         def = LogLevel::Normal) noexcept;

// Log destination interface. Implementations decide filtering,
// formatting, and where the message ends up. Implementations MUST be
// safe to call concurrently from multiple threads. Implementations
// MUST NOT throw out of log() -- a logging failure should not crash
// the caller. Implementations SHOULD only invoke the VpipeFormat
// callable when the message is going to be emitted (the callable can
// be expensive).
class LogDelegateIntf {
public:
  virtual ~LogDelegateIntf() = default;
  virtual void log(LogLevel, const VpipeFormat&) = 0;

  // Live threshold accessor / mutator. set_threshold() must be safe to
  // call concurrently with log(); concrete delegates back this with an
  // atomic so callers see a consistent snapshot.
  virtual void     set_threshold(LogLevel) = 0;
  virtual LogLevel threshold() const noexcept = 0;
};

}

#endif
