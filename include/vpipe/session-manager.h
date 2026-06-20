// vpipe::SessionManager -- process-wide singleton that owns every
// SessionIntf instance. Library users never construct a Session
// directly; they ask the manager for one.
//
// The manager is process-wide because vpipe relies on a number of
// per-process resources (the StageRegistry, dlopen'd FFmpeg
// libraries, the session ThreadPool) that are cheap to initialize
// once and expensive to fork. Multiple sessions share none of
// their pipelines or runtimes -- only those process-wide singletons.
//
// Threading: every method on SessionManager is internally
// synchronized and safe to call from any thread.

#ifndef SESSION_MANAGER_H
#define SESSION_MANAGER_H

#include "vpipe/session-intf.h"
#include <string_view>

namespace vpipe {

class SessionManager {
public:
  // The singleton accessor. The reference is valid for the lifetime
  // of the process; the destructor is protected so the manager
  // cannot be destroyed by user code.
  static SessionManager& get();

  // Create a new session. The config string is one of:
  //   * empty / all-whitespace : built-in defaults.
  //   * starts with `{` or `[` : inline JSON, parsed via FlexData.
  //   * otherwise              : filesystem path to a JSON or
  //                              binary-FlexData config file.
  //
  // The returned pointer is borrowed: the manager retains ownership
  // and the caller must not delete it. Pass it back to
  // `destroy_session` when no longer needed. Returns nullptr if the
  // config could not be parsed.
  //
  // The pointer is `const SessionIntf*` purely as a pointer-stability
  // hint. The underlying object is not actually const; mutating
  // methods on SessionIntf may be called on it.
  virtual const SessionIntf* create_session(std::string_view = "") = 0;

  // Destroy a session previously returned by `create_session`. Any
  // pipelines still launched on it are stopped first. After this
  // call every PipelineHandle / StageHandle that referenced the
  // session is invalid (handle methods report Status{1}). It is
  // safe to pass nullptr or a session that was already destroyed.
  virtual void destroy_session(const SessionIntf*) = 0;

  // Number of sessions currently alive in this process.
  virtual unsigned num_sessions() const = 0;

protected:
  virtual ~SessionManager() = default;
};

}

#endif
