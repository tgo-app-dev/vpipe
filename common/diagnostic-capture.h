#ifndef COMMON_DIAGNOSTIC_CAPTURE_H
#define COMMON_DIAGNOSTIC_CAPTURE_H

// diagnostic-capture.h -- what a refused operation reported, handed
// back to whoever asked for it.
//
// THE PROBLEM. Every refusal in the load path already knows exactly
// why: pipeline_from_spec names the stage and the field it objects
// to, the loader names the parse position, StageRegistry names the
// unknown type. All of it goes to session->warn(), and the caller is
// handed a null handle and nothing else. So the web-ui answers
// "failed to load 'x'" and the operator has to go and read a console
// panel to find out what was wrong with their file.
//
// WHY NOT AN OUT-PARAMETER. Those reasons come from dozens of warn()
// sites across the loader, the spec reader, the registry and the
// runtime, several of them behind a virtual. A `std::string* err`
// threaded through all of them is a large diff AND a standing
// invitation to add the next site without it. The reports already
// exist and already travel; what was missing is a way for a caller to
// LISTEN to its own operation.
//
// So: a scope. While one of these is alive, every error() and warn()
// the session reports ON THIS THREAD is recorded here as well as
// being reported as usual -- nothing is diverted, and a caller that
// never looks changes nothing. The caller runs the operation, checks
// the result exactly as before, and on failure has the reasons.
//
// THREAD-SCOPED, deliberately. A session serves many pipelines at
// once and its user-facing channel is shared; a capture that recorded
// every thread would attach an unrelated stage's warning to this
// operator's rejected file, which is worse than saying nothing. The
// parse / build / validate path is synchronous on the calling thread,
// and so is every launch refusal -- the runtime's own checks all
// precede the point where drivers are scheduled -- so thread scope
// covers exactly the operation and nothing else.
//
// WHAT IT THEREFORE DOES NOT CATCH: what a worker thread reports once
// the pipeline is running -- a stage whose initialize() threw, or one
// skipped for an invalid config. Those do not refuse anything. They
// are reported per stage, while the pipeline runs, and belong to the
// log rather than to a verdict.

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "common/vpipe-api.h"

// Exported for the in-tree apps, NOT the plugin contract (vpipe/export.h).
VPIPE_HOST_API_BEGIN

namespace vpipe {

class DiagnosticCapture {
public:
  DiagnosticCapture();
  ~DiagnosticCapture();

  DiagnosticCapture(const DiagnosticCapture&) = delete;
  DiagnosticCapture& operator=(const DiagnosticCapture&) = delete;

  // Nothing was reported while this scope was open. Worth reporting in
  // its own right: a refusal with no reason is a missing warn() at the
  // site that refused, and a caller that says so names the defect
  // instead of hiding it behind an empty string.
  bool empty() const noexcept { return _lines.empty(); }

  const std::vector<std::string>& lines() const noexcept { return _lines; }

  // The reasons as one block, newline-separated, in the order they
  // were reported. Empty when nothing was.
  std::string text() const;

  // True once the caps below bit, i.e. `lines()` is a prefix of what
  // was reported rather than all of it.
  bool truncated() const noexcept { return _truncated; }

  // Is anything listening on this thread? The session asks before
  // formatting, so a capture costs nothing anywhere else.
  static bool active() noexcept;

  // Record one already-formatted report against every capture open on
  // this thread. Nested scopes both see it: an inner capture answering
  // one sub-step must not blind the outer one that owns the verdict.
  static void note(std::string_view);

private:
  // A refusal is a handful of lines. These bound a pathological case
  // (a spec with a thousand bad stages, a launch that reports per
  // edge) so a capture cannot grow into the thing it is describing.
  static constexpr std::size_t kMaxLines = 32;
  static constexpr std::size_t kMaxBytes = 8192;

  void note_(std::string_view);

  std::vector<std::string> _lines;
  DiagnosticCapture*       _outer     = nullptr;
  std::size_t              _bytes     = 0;
  bool                     _truncated = false;
};

}

VPIPE_HOST_API_END

#endif
