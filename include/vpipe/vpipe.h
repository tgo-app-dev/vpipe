// vpipe -- public API umbrella header.
//
// Including this header pulls in every type a library user needs to
// reach the vpipe runtime:
//
//   * SessionManager  (vpipe/session-manager.h)
//   * SessionIntf     (vpipe/session-intf.h)
//   * PipelineHandle, StageHandle, StagePortHandle
//                     (vpipe/pipeline-handle.h)
//   * Status          (vpipe/status.h)
//
// Everything else (the implementation classes, the Stage type tree,
// internal helpers) lives under non-installed headers and is reached
// only by code linked into libvpipe itself.
//
// Linking: applications link against libvpipe (shared) and rely on
// the SessionManager singleton to hand out a SessionIntf. There is
// no header-only path -- the vtables for SessionManager / Session /
// the StageRegistry live in the shared library.

#ifndef VPIPE_H
#define VPIPE_H

#include "vpipe/session-manager.h"

namespace vpipe {

// Returns a NUL-terminated build identifier of the form
// `<git-hash>*<dirty-count>` (e.g. `d950473*4`). The string is
// generated at build time from the git working tree and is stable
// for the life of the process. Useful for logging the exact build
// behind a deployed binary.
const char* vpipe_version();

}

#endif
