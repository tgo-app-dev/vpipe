#include "interfaces/log-delegate-intf.h"

namespace vpipe {

const char*
to_cstr(LogLevel l) noexcept
{
  switch (l) {
    case LogLevel::Error:   return "ERROR";
    case LogLevel::Warn:    return "WARN";
    case LogLevel::Info:    return "INFO";
    case LogLevel::Normal:  return "NORMAL";
    case LogLevel::Verbose: return "VERBOSE";
    case LogLevel::Debug:   return "DEBUG";
    case LogLevel::Always:  return "ALWAYS";
  }
  return "?";
}

LogLevel
parse_log_level(std::string_view name, LogLevel def) noexcept
{
  if (name == "error")   return LogLevel::Error;
  if (name == "warn")    return LogLevel::Warn;
  if (name == "info")    return LogLevel::Info;
  if (name == "normal")  return LogLevel::Normal;
  if (name == "verbose") return LogLevel::Verbose;
  if (name == "debug")   return LogLevel::Debug;
  if (name == "always")  return LogLevel::Always;
  return def;
}

}
