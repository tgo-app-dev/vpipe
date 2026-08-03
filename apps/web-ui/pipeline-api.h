// pipeline-api.h -- /api/pipelines/* and /api/stage-types : the
// Pipeline Manager's model of the session's graphs.
//
// Each managed pipeline keeps an editable C++ spec (the source of
// truth) plus a live materialized PipelineHandle. Editing is only
// permitted while the pipeline is stopped; a structural edit mutates
// the spec and rebuilds the live graph (create_pipeline + ordered
// insert_stage). Graph introspection reads the live Stage objects
// directly (type, ports, payload types, current config).
//
// This controller owns the pipeline state, so it is also the one that
// can answer questions ABOUT that state. Its peers ask it rather than
// keeping their own view: DatabaseApi has to know whether anything is
// running before it mutates a db, SystemApi enumerates live HLS stages,
// and the stage-provided view panels resolve a (pipeline, stage) pair
// through the UiViewHostIntf implemented here.

#ifndef WEBUI_PIPELINE_API_H
#define WEBUI_PIPELINE_API_H

#include "apps/web-ui/api-context.h"
#include "apps/web-ui/http-server.h"
#include "common/flex-data.h"
#include "interfaces/ui-view-intf.h"
#include "vpipe/pipeline-handle.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vpipe {
class Pipeline;
class Stage;
}

namespace vpipe::webui {

// Resolve a PipelineHandle to its live Pipeline graph. Shared with
// SystemApi, which walks live stages to report the HLS streams.
Pipeline* live_pipeline(const PipelineHandle& h);

class PipelineApi {
public:
  explicit PipelineApi(ApiContext& ctx) : _ctx(ctx) {}

  void register_routes(HttpServer& s);

  // ---- what peers may ask about pipeline state -------------------

  // True if any managed pipeline is not Stopped (running or paused).
  // CALLER HOLDS ApiContext::mu -- the answer is only meaningful while
  // it is held, which is exactly why DatabaseApi takes the lock across
  // its check and its write.
  bool any_active_locked() const;

  // The view host backing the stage-provided panels. Wired into the UI
  // delegate at startup so a stage can reach it through its session
  // (UiDelegateIntf::ui_view_host).
  UiViewHostIntf* view_host() noexcept { return &_view_host; }

  // Every live stage of `type_name` across the loaded pipelines, as
  // (pipeline id, stage id, stage) triples. Locks internally.
  std::vector<UiViewHostIntf::StageRef>
  find_stages(std::string_view type_name) const;

  // Resolve a (pipeline, stage) pair to a live Stage; null when the
  // pipeline is not materialized, is stopped, or has no such stage.
  // Locks internally.
  Stage* live_stage(std::string_view pipeline,
                    std::string_view stage) const;

  // Run `fn(pipeline_id, state_name, pipeline)` for every materialized
  // pipeline. CALLER HOLDS ApiContext::mu. Used by SystemApi's HLS
  // enumeration, which needs the graph as well as the id.
  void for_each_live_locked(
      const std::function<void(const std::string&, const char*,
                               Pipeline&)>& fn) const;

private:
  // One stage entry in a pipeline's editable spec.
  struct StageSpec {
    std::string                                  id;
    std::string                                  type;
    // positional iport bindings: iports[i] feeds this stage's iport i,
    // sourced from {src stage id, that stage's oport index}.
    std::vector<std::pair<std::string, unsigned>> iports;
    FlexData                                     config;  // object
  };

  enum class State { Stopped, Running, Paused };

  struct Pipe {
    std::string                 id;
    // Editable spec, kept roughly in insertion (dependency) order; an
    // edge rewire (h_connect_) can leave it non-topological, so
    // to_flex_spec_ re-sorts on save (the core loader needs sources
    // declared first).
    std::vector<StageSpec>      stages;
    std::optional<PipelineHandle> handle;        // live materialization
    State                       state = State::Stopped;
    std::string                 storage_path;
    // Auxiliary data objects round-tripped with the pipeline document
    // but opaque to the pipeline core -- a map of named JSON objects
    // (e.g. the composer's view arrangement under "composer").
    // Persisted under the spec's "aux" key on save, recovered on load.
    FlexData                    aux;
  };

  // ---- request handlers ------------------------------------------
  HttpResponse h_stage_types_(const HttpRequest&);
  HttpResponse h_list_pipelines_(const HttpRequest&);
  HttpResponse h_create_pipeline_(const HttpRequest&);
  HttpResponse h_rename_pipeline_(const HttpRequest&);
  HttpResponse h_load_pipeline_(const HttpRequest&);
  HttpResponse h_get_pipeline_(const HttpRequest&);
  HttpResponse h_save_pipeline_(const HttpRequest&);
  HttpResponse h_set_pipeline_aux_(const HttpRequest&);
  HttpResponse h_unload_pipeline_(const HttpRequest&);
  HttpResponse h_launch_pipeline_(const HttpRequest&);
  HttpResponse h_pause_pipeline_(const HttpRequest&);
  HttpResponse h_stop_pipeline_(const HttpRequest&);
  // Per-edge buffer-utilization snapshot of a running pipeline, for the
  // graph overlay. {id, state, edges:[{from,from_port,to,to_port,
  // backlog,capacity,dropped,closed}]}. edges is empty unless launched.
  HttpResponse h_buffer_status_(const HttpRequest&);
  HttpResponse h_insert_stage_(const HttpRequest&);
  HttpResponse h_remove_stage_(const HttpRequest&);
  HttpResponse h_rename_stage_(const HttpRequest&);
  HttpResponse h_duplicate_stage_(const HttpRequest&);
  // Edge editing for the composer. connect re-points an existing input
  // port (in place, via PipelineHandle::move_iport) or appends a new one
  // (re-materialize); disconnect drops an input. Both are refused unless
  // the pipeline is stopped and return {id,state,graph}. connect body:
  // {from, from_port, to, to_port?}; disconnect body: {to, to_port}.
  HttpResponse h_connect_(const HttpRequest&);
  HttpResponse h_disconnect_(const HttpRequest&);
  HttpResponse h_get_stage_config_(const HttpRequest&);
  HttpResponse h_set_stage_config_(const HttpRequest&);

  // ---- internals -------------------------------------------------
  Pipe* find_(const std::string& id);

  // Resolve a stage by id within a pipe's live graph (nullptr if the
  // pipe isn't materialized or carries no such stage). Caller holds mu.
  Stage* live_stage_(const Pipe& p, const std::string& id) const;

  // Auto-stop any managed pipeline whose stages have all signalled done
  // (its runtime drained on its own, with no pause/stop requested):
  // finalize it via stop_pipeline and flip the cached state to Stopped,
  // so a self-terminating pipeline shows as stopped without an explicit
  // stop request. Caller holds mu. Invoked from the status-read handlers
  // (list / get / buffer-status) the UI polls.
  void reap_completed_();

  // (Re)build the live pipeline from `p.stages`. On failure the live
  // handle is left unloaded and `err` is set. Caller holds mu.
  bool materialize_(Pipe& p, std::string& err);
  // Drop the live handle (unload from the session).
  void dematerialize_(Pipe& p);

  // FlexData views of a pipe.
  FlexData pipe_summary_(const Pipe& p) const;      // {id,state,...}
  FlexData graph_json_(const Pipe& p) const;        // {nodes,edges}
  FlexData to_flex_spec_(const Pipe& p) const;      // pipeline-spec doc
  // Merge each key of `incoming` (an object) into p.aux, creating it as
  // an object if needed; a null value erases that key. No-op if not an
  // object.
  void merge_aux_(Pipe& p, const FlexData& incoming);
  // Stage indices in dependency order (Kahn, stable wrt the input
  // order). The core pipeline_from_spec loader resolves iports in a
  // single pass, so a saved spec MUST declare each stage after its
  // sources; in-memory edits (h_connect_ rewiring an earlier stage onto
  // a later source) can perturb Pipe::stages, so to_flex_spec_ re-sorts
  // here rather than trusting it.
  static std::vector<std::size_t> topo_order_(
      const std::vector<StageSpec>& stages);

  static const char* state_name_(State s);

  // What a view's backend may ask of this app: which stages of a type
  // exist across the loaded pipelines, and which of them are live.
  class ViewHost final : public UiViewHostIntf {
  public:
    explicit ViewHost(PipelineApi& api) : _api(api) {}
    std::vector<StageRef>
    find_stages(std::string_view type_name) const override;
    Stage*
    live_stage(std::string_view pipeline,
               std::string_view stage) const override;
  private:
    PipelineApi& _api;
  };
  ViewHost _view_host{*this};

  ApiContext&                        _ctx;
  std::vector<std::unique_ptr<Pipe>> _pipes;
};

}

#endif
