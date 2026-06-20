// vpipe::PipelineHandle / StageHandle -- opaque handles handed out
// by SessionIntf to refer to live pipeline graphs and stages owned
// by a session.
//
// Both types are value-semantic, cheap to copy / pass by value,
// and outlive nothing -- they are essentially typed pointers into
// the session's owning maps. A handle becomes invalid when the
// session it came from is destroyed (see SessionManager) or the
// pipeline is unloaded; subsequent operations on it report
// Status{1} rather than crashing.
//
// Use `valid()` (or `if (handle)` via the explicit bool operator)
// to check whether a handle refers to anything; a null handle is
// returned from APIs that did not find a match (e.g.
// load_pipeline of a non-existent path).
//
// Library-internal code reaches the underlying impl through the
// HandleAccess gateway declared here and defined in
// pipeline/pipeline-handle-impl.h. Public-API users cannot reach
// the impl pointer.

#ifndef PIPELINE_HANDLE_H
#define PIPELINE_HANDLE_H

#include <string>
#include <vector>

namespace vpipe {

class PipelineHandleImpl;
class StageHandleImpl;

// Internal accessor that grants in-library code (Session, the
// PipelineRuntime, tests, the nanobind binding) the right to construct
// handles from impl pointers and to read the impl back out. Defined in
// pipeline/pipeline-handle-impl.h, intentionally not on the public
// include path -- users of the public API cannot reach the impl
// pointer without including a private header.
class HandleAccess;

class StageHandle final {
public:
  ~StageHandle() = default;

  unsigned num_oports() const;

  // JSON description of this stage's configuration schema: an array of
  // { "key", "type", "required", "doc"?, "default", "current" }
  // objects -- one per configuration key the stage understands, with
  // its declared type, default, and the value this instance was given.
  // Returns "[]" for a null handle or a stage that advertises no
  // schema. Lets a client discover a stage's configuration keys,
  // types, and current values without knowing the stage type.
  std::string config_schema_json() const;

  // True iff this handle refers to a live stage. A default-/null-
  // constructed handle, or one returned from an unimplemented stub,
  // is invalid.
  bool valid() const noexcept { return _impl != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

private:
  friend class HandleAccess;
  explicit StageHandle(StageHandleImpl* impl) noexcept : _impl(impl) {}

  StageHandleImpl* _impl;
};

struct StagePortHandle {
  StageHandle stage;
  unsigned    oport;
};

class PipelineHandle final {
public:
  ~PipelineHandle() = default;

  // Construct a stage of `type` (a name registered with the
  // StageRegistry, e.g. "chrono", "shell", "video-file-decoder")
  // and insert it into this pipeline.
  //
  //   id          stage-local id, used in log messages.
  //   iports      list of {upstream-stage, oport-index} pairs that
  //               feed this stage's input ports, in order. Empty
  //               for source stages.
  //   config_json optional JSON object describing the stage's
  //               configuration. Each stage type defines its own
  //               schema -- see STAGES.md. An empty string is
  //               treated as the empty object `{}`.
  //
  // Returns a null handle (`!valid()`) on any error: unknown type,
  // bad config JSON, an upstream port index out of range, or the
  // stage's own ctor rejecting the config. Errors are reported
  // through the session's log delegate.
  StageHandle insert_stage(std::string         type,
                           std::string         id,
                           std::vector<StagePortHandle> iports,
                           std::string         config_json = "");

  PipelineHandle insert_pipeline(std::string id);

  // Re-point one of a stage's input ports after construction, the
  // post-construction edge edit behind the web-ui composer's
  // click-to-connect. `stage_id` names the consumer; `iport` is its
  // input-port index. The new source is `src_id`'s output port
  // `src_oport`; an EMPTY `src_id` DISCONNECTS the input (null source).
  //
  // Only re-points an EXISTING input port (it cannot grow the input
  // list -- add a new input by reinserting the stage with one more
  // binding). Returns false on any error: unknown stage id, iport out
  // of range, src oport out of range, the pipeline is launched, or the
  // graph rejected the move (see Graph::move_iport_to). Errors are
  // reported through the session's log delegate.
  bool move_iport(std::string stage_id, unsigned iport,
                  std::string src_id, unsigned src_oport);

  // True iff this handle refers to a live pipeline.
  bool valid() const noexcept { return _impl != nullptr; }
  explicit operator bool() const noexcept { return valid(); }

private:
  friend class HandleAccess;
  explicit PipelineHandle(PipelineHandleImpl* impl) noexcept
    : _impl(impl) {}

  PipelineHandleImpl* _impl;
};

}

#endif
