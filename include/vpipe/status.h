// vpipe::Status -- numeric return value used by mutating SessionIntf
// operations.
//
// Status is intentionally a thin POD: a single `unsigned code` that
// is either zero (success) or non-zero (failure). It is not a rich
// error type -- detailed diagnostics go through the session's log
// delegate. The numeric meaning is stable across the public API:
//
//   0  Success.
//   1  Bad request: the call's arguments did not refer to a live
//      object (e.g. an unknown PipelineHandle, an empty / unparseable
//      config, a path that did not resolve to a file).
//   2  Operation failed at the runtime level. The pipeline graph or
//      the launch path rejected the request; the log delegate carries
//      the reason.
//   3  Rejected because of a precondition. The receiver is in a state
//      that does not allow the operation right now (e.g. log-config
//      mutators are blocked while any pipeline is launched).
//
// Use `if (status)` -- via the implicit unsigned-to-bool conversion
// of `code` -- only after explicitly comparing against the codes you
// care about; the API surfaces no boolean shortcut.

#ifndef STATUS_H
#define STATUS_H

namespace vpipe {

struct Status {
  unsigned code;
};

// Returns a short human-readable label for `s` (e.g. "ok", "bad
// request", "failed", "rejected"). Intended for log messages. The
// returned pointer is to a static string and remains valid for the
// process lifetime.
const char* to_str(Status s);

}

#endif
