#include "pipeline/pipeline-handle-impl.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage.h"
#include <exception>
#include <memory>
#include <utility>

using namespace std;

namespace vpipe {

PipelineHandleImpl::PipelineHandleImpl
  (unique_ptr<Pipeline>      pipeline,
   const SessionContextIntf* session)
  : _pipeline(std::move(pipeline))
  , _runtime(nullptr)
  , _session(session)
{
}

PipelineHandleImpl::~PipelineHandleImpl()
{
  // Runtime dtor will stop & wait_idle if still running, then
  // destroy drivers / contexts / buffers in safe order. The
  // Pipeline (graph of stages) outlives the runtime: stage objects
  // are referenced by RuntimeContexts, so we must destroy the
  // runtime before the pipeline.
  _runtime.reset();
  _pipeline.reset();
}

StageHandle
PipelineHandleImpl::insert_stage(string                  type,
                                 string                  id,
                                 vector<StagePortHandle> iports,
                                 string                  config_json)
{
  if (!_pipeline) {
    return HandleAccess::make_stage(nullptr);
  }
  // Resolve every StagePortHandle to an InEdge against the live
  // Stage*. A null handle, an oport index >= the upstream stage's
  // oport count, or a stage that no longer belongs to this pipeline
  // is rejected; the call returns null.
  vector<InEdge> resolved;
  resolved.reserve(iports.size());
  for (size_t i = 0; i < iports.size(); ++i) {
    const StagePortHandle& sph = iports[i];
    StageHandleImpl* simpl = HandleAccess::impl(sph.stage);
    Stage* upstream = simpl ? simpl->stage() : nullptr;
    if (!upstream) {
      if (_session) {
        _session->warn(fmt(
          "PipelineHandle::insert_stage('{}'): iport {} references "
          "a null upstream stage", id, i));
      }
      return HandleAccess::make_stage(nullptr);
    }
    if (upstream->graph() != _pipeline.get()) {
      if (_session) {
        _session->warn(fmt(
          "PipelineHandle::insert_stage('{}'): iport {} references "
          "a stage from a different pipeline", id, i));
      }
      return HandleAccess::make_stage(nullptr);
    }
    if (sph.oport >= upstream->num_oports()) {
      if (_session) {
        _session->warn(fmt(
          "PipelineHandle::insert_stage('{}'): iport {} oport index "
          "{} out of range (upstream '{}' has {} oports)",
          id, i, sph.oport, upstream->id(),
          upstream->num_oports()));
      }
      return HandleAccess::make_stage(nullptr);
    }
    resolved.push_back(InEdge{upstream, sph.oport});
  }

  // Parse config. An empty / whitespace string is treated as the
  // empty object so callers don't have to write `{}` for stages
  // that take no required config.
  FlexData config = FlexData::make_object();
  bool has_payload = false;
  for (char c : config_json) {
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
      has_payload = true;
      break;
    }
  }
  if (has_payload) {
    try {
      config = FlexData::from_json(config_json);
    } catch (const exception& e) {
      if (_session) {
        _session->warn(fmt(
          "PipelineHandle::insert_stage('{}'): config JSON parse "
          "failed: {}", id, e.what()));
      }
      return HandleAccess::make_stage(nullptr);
    }
  }

  // Build the stage through the registry. The registry catches any
  // exception thrown by the stage's ctor (the stage already routed
  // its diagnostic through session->error, which logs at Error
  // level) and returns nullptr; the only thing left for us to do is
  // distinguish "unknown type" from "ctor failed", since both
  // surface here as nullptr -- skip that distinction. The session
  // log already names the cause.
  StagePtr s = StageRegistry::get().create(type, _session, id,
                                           std::move(resolved),
                                           std::move(config));
  if (!s) {
    return HandleAccess::make_stage(nullptr);
  }

  Stage* raw = _pipeline->insert_stage(std::move(s));
  if (!raw) {
    return HandleAccess::make_stage(nullptr);
  }
  auto himpl = make_unique<StageHandleImpl>(raw);
  StageHandleImpl* hraw = himpl.get();
  _stage_handles.push_back(std::move(himpl));
  return HandleAccess::make_stage(hraw);
}

PipelineHandle
PipelineHandleImpl::insert_pipeline(string)
{
  if (_session) {
    _session->warn(fmt(
      "PipelineHandle::insert_pipeline is not yet implemented"));
  }
  return HandleAccess::make_pipeline(nullptr);
}

namespace {

// Find a stage by id within a live pipeline graph (linear scan over
// the vertex map). Returns nullptr if no Stage carries that id.
Stage*
find_stage_(Pipeline* pl, const string& id)
{
  for (auto it = pl->begin(); it != pl->end(); ++it) {
    if (Stage* s = dynamic_cast<Stage*>(*it)) {
      if (s->id() == id) { return s; }
    }
  }
  return nullptr;
}

}  // namespace

bool
PipelineHandleImpl::move_iport(const string& stage_id, unsigned iport,
                               const string& src_id, unsigned src_oport)
{
  if (!_pipeline) { return false; }
  if (_runtime) {
    if (_session) {
      _session->warn(fmt(
        "PipelineHandle::move_iport: pipeline '{}' is launched; stop "
        "it before editing edges", _pipeline->id()));
    }
    return false;
  }

  Stage* dst = find_stage_(_pipeline.get(), stage_id);
  if (!dst) {
    if (_session) {
      _session->warn(fmt(
        "PipelineHandle::move_iport: no stage '{}'", stage_id));
    }
    return false;
  }
  if (iport >= dst->num_iports()) {
    if (_session) {
      _session->warn(fmt(
        "PipelineHandle::move_iport: stage '{}' has no iport {} "
        "(it has {})", stage_id, iport, dst->num_iports()));
    }
    return false;
  }

  // Empty src => disconnect (null source).
  InEdge new_src{nullptr, 0};
  if (!src_id.empty()) {
    Stage* src = find_stage_(_pipeline.get(), src_id);
    if (!src) {
      if (_session) {
        _session->warn(fmt(
          "PipelineHandle::move_iport: no source stage '{}'", src_id));
      }
      return false;
    }
    if (src_oport >= src->num_oports()) {
      if (_session) {
        _session->warn(fmt(
          "PipelineHandle::move_iport: source '{}' has no oport {} "
          "(it has {})", src_id, src_oport, src->num_oports()));
      }
      return false;
    }
    new_src = InEdge{src, src_oport};
  }

  if (!_pipeline->move_iport_to(dst, iport, new_src)) {
    if (_session) {
      _session->warn(fmt(
        "PipelineHandle::move_iport: graph rejected moving '{}' "
        "iport {}", stage_id, iport));
    }
    return false;
  }
  return true;
}

bool
PipelineHandleImpl::launch()
{
  if (_runtime) {
    _session->warn(fmt(
      "PipelineHandleImpl::launch: pipeline '{}' already launched",
      _pipeline->id()));
    return false;
  }
  _runtime = make_unique<PipelineRuntime>(_pipeline.get(), _session);
  if (!_runtime->launch()) {
    _runtime.reset();
    return false;
  }
  return true;
}

void
PipelineHandleImpl::pause()
{
  if (_runtime) {
    _runtime->pause();
  }
}

void
PipelineHandleImpl::stop()
{
  if (_runtime) {
    _runtime->stop();
    _runtime.reset();
  }
}

bool
PipelineHandleImpl::wait_idle(int timeout_ms)
{
  if (!_runtime) {
    // No active runtime -- treat as "already idle". Mirrors the
    // semantics that an un-launched (or already-stopped) pipeline
    // has no outstanding work to await.
    return true;
  }
  return _runtime->wait_idle(timeout_ms);
}

unsigned
StageHandleImpl::num_oports() const
{
  return _stage ? _stage->num_oports() : 0;
}

StageHandle
PipelineHandle::insert_stage(string                  type,
                             string                  id,
                             vector<StagePortHandle> i,
                             string                  config_json)
{
  if (!_impl) {
    return HandleAccess::make_stage(nullptr);
  }
  return _impl->insert_stage(std::move(type), std::move(id),
                             std::move(i),
                             std::move(config_json));
}

PipelineHandle
PipelineHandle::insert_pipeline(string id)
{
  if (!_impl) {
    return HandleAccess::make_pipeline(nullptr);
  }
  return _impl->insert_pipeline(std::move(id));
}

bool
PipelineHandle::move_iport(string stage_id, unsigned iport,
                           string src_id, unsigned src_oport)
{
  if (!_impl) { return false; }
  return _impl->move_iport(stage_id, iport, src_id, src_oport);
}

unsigned
StageHandle::num_oports() const
{
  return _impl ? _impl->num_oports() : 0;
}

string
StageHandle::config_schema_json() const
{
  const Stage* s = _impl ? _impl->stage() : nullptr;
  if (!s) {
    return "[]";
  }
  return s->config_schema().to_json();
}

}
