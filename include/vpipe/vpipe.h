// vpipe -- public API umbrella header.
//
// Including this header pulls in every type a library user needs to
// reach the vpipe runtime:
//
//   * SessionManager  (vpipe/session-manager.h)
//   * SessionIntf     (vpipe/session-intf.h)
//   * PipelineHandle, StageHandle, StagePortHandle
//                     (vpipe/pipeline-handle.h)
//   * CommandHandle, DataBuffer, BufferLayout -- commands to a running
//     stage, with data in and out by reference
//                     (vpipe/stage-command.h)
//   * Status          (vpipe/status.h)
//   * the live progress reports a host draws its own bars from
//                     (SessionIntf::progress, as /api/io/progress)
//   * SystemMonitor   the machine's load and GPU thermal verdict, as the
//                     web UI's status bar and `vpipe --gpu-thermal` read
//                     them (vpipe/system-monitor.h; included on its own:
//                     it needs no session)
//
// Specs, configs and command arguments go in as JSON text, or -- for a
// host that builds them in code -- as FlexData documents
// (common/flex-data.h, from the SDK), with no text written or parsed.
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

#include "vpipe/export.h"

VPIPE_API_BEGIN

namespace vpipe {

// Returns a NUL-terminated build identifier of the form
// `<git-hash>*<dirty-count>` (e.g. `d950473*4`). The string is
// generated at build time from the git working tree and is stable
// for the life of the process. Useful for logging the exact build
// behind a deployed binary.
const char* vpipe_version();

// The same build identity, split so a caller can compose its own string
// (e.g. an EXIF Software tag) without parsing vpipe_version()'s
// "<version> (<hash>)" shape.
//   vpipe_version_number() -> "0.1"
//   vpipe_build_hash()     -> "d950473*4"  (the `*N` suffix counts dirty files)
const char* vpipe_version_number();
const char* vpipe_build_hash();

}

VPIPE_API_END

#endif
