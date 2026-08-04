// log-sink-intf.h -- structured logging, on its own.
//
// Split out of SessionContextIntf because it is what almost everything
// actually wants: of ~900 calls made through the session context in
// this tree, roughly 90% are one of these seven. A helper that only
// reports (the FFmpeg library wrappers, a format probe, a validator)
// should say so by taking a LogSinkIntf& rather than a whole session,
// which also makes it constructible in a test without one.
//
// SessionContextIntf inherits this, so every existing caller keeps
// working unchanged.

#ifndef LOG_SINK_INTF_H
#define LOG_SINK_INTF_H

namespace vpipe {

struct VpipeFormat;

class LogSinkIntf {
public:
  virtual ~LogSinkIntf() = default;

  virtual void error(const VpipeFormat&) const = 0;
  virtual void warn(const VpipeFormat&) const = 0;
  virtual void info(const VpipeFormat&) const = 0;
  virtual void log_debug(const VpipeFormat&) const = 0;
  virtual void log_verbose(const VpipeFormat&) const = 0;
  virtual void log_normal(const VpipeFormat&) const = 0;
  virtual void log_always(const VpipeFormat&) const = 0;
};

}

#endif
