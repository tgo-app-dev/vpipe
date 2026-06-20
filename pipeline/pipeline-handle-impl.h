#ifndef PIPELINE_HANDLE_IMPL_H
#define PIPELINE_HANDLE_IMPL_H

#include "vpipe/pipeline-handle.h"
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

class Pipeline;
class PipelineRuntime;
class SessionContextIntf;
class Stage;

// Concrete impl behind PipelineHandle. Owns the Pipeline (graph of
// Stages) and -- once launched -- the PipelineRuntime (thread-pool
// drivers and edge buffers). One Impl per pipeline created via
// Session::create_pipeline; Session also owns the Impl. Pool and
// default-edge-capacity are reached through `session`.
class PipelineHandleImpl final {
public:
  PipelineHandleImpl(std::unique_ptr<Pipeline>  pipeline,
                     const SessionContextIntf*  session);
  ~PipelineHandleImpl();

  PipelineHandleImpl(const PipelineHandleImpl&)            = delete;
  PipelineHandleImpl& operator=(const PipelineHandleImpl&) = delete;

  // Public API surfaced through PipelineHandle. insert_stage
  // resolves the StagePortHandle list to InEdges (validating each
  // upstream port index against the source stage's allocated
  // oport count), parses `config_json` via FlexData::from_json,
  // looks up `type` in the StageRegistry, and inserts the
  // constructed Stage into the underlying Pipeline. The returned
  // StageHandle wraps a StageHandleImpl owned by this impl --
  // valid for the pipeline's lifetime.
  StageHandle    insert_stage(std::string         type,
                              std::string         id,
                              std::vector<StagePortHandle> iports,
                              std::string         config_json = "");
  PipelineHandle insert_pipeline(std::string id);

  // Re-point stage `stage_id`'s input port `iport` to `src_id`'s
  // output `src_oport` (empty src_id => disconnect). See
  // PipelineHandle::move_iport.
  bool move_iport(const std::string& stage_id, unsigned iport,
                  const std::string& src_id, unsigned src_oport);

  Pipeline*       pipeline()       noexcept { return _pipeline.get(); }
  const Pipeline* pipeline() const noexcept { return _pipeline.get(); }

  // Live runtime once launched (nullptr otherwise). Exposed so the
  // web-ui can read per-edge buffer-utilization snapshots
  // (PipelineRuntime::edge_buffer_stats) while the pipeline runs.
  const PipelineRuntime* runtime() const noexcept { return _runtime.get(); }

  // Lifecycle. Forwards to PipelineRuntime; returns false if a
  // launch was rejected (e.g., already launched).
  bool launch();
  void pause();
  void stop();

  // Block until the runtime reports every driver finished, or
  // timeout_ms elapses. A negative timeout waits forever. Returns
  // true on completion, false on timeout. Returns true immediately
  // if the pipeline is not currently launched.
  //
  // Not safe to call concurrently with stop()/unload on the same
  // impl from another thread; see SessionIntf::wait_pipelines().
  bool wait_idle(int timeout_ms);

  bool launched() const noexcept { return _runtime != nullptr; }

  // Storage URL associated with this pipeline. Set by
  // Session::load_pipeline (the path it was loaded from) or by
  // Session::store_pipeline(handle, path). The no-arg
  // store_pipeline reuses this for its destination.
  const std::string& storage_path() const noexcept
  { return _storage_path; }
  void storage_path(std::string p)
  { _storage_path = std::move(p); }

private:
  std::unique_ptr<Pipeline>        _pipeline;
  std::unique_ptr<PipelineRuntime> _runtime;
  const SessionContextIntf*        _session;
  std::string                      _storage_path;

  // One StageHandleImpl per inserted stage, owned by the pipeline.
  // The vector is append-only; existing handles remain valid even
  // when more stages are inserted because we don't relocate the
  // impls themselves (each is heap-allocated through unique_ptr).
  std::vector<std::unique_ptr<StageHandleImpl>> _stage_handles;
};

class StageHandleImpl final {
public:
  explicit StageHandleImpl(Stage* s) noexcept : _stage(s) {}

  unsigned num_oports() const;

  Stage*       stage()       noexcept { return _stage; }
  const Stage* stage() const noexcept { return _stage; }

private:
  Stage* _stage;
};

// Friend gateway between the public Pipeline/StageHandle types and
// the internal *Impl objects they wrap. Forward-declared in the
// public header; in-library callers include this header to gain
// access. Keeping the impl pointer accessible only through these
// static methods means the public ABI never has to expose impl().
class HandleAccess final {
public:
  static PipelineHandle
  make_pipeline(PipelineHandleImpl* p) noexcept
  { return PipelineHandle(p); }

  static StageHandle
  make_stage(StageHandleImpl* s) noexcept
  { return StageHandle(s); }

  static PipelineHandleImpl*
  impl(const PipelineHandle& h) noexcept
  { return h._impl; }

  static StageHandleImpl*
  impl(const StageHandle& h) noexcept
  { return h._impl; }
};

}

#endif
