#ifndef WEBUI_SESSION_API_H
#define WEBUI_SESSION_API_H

#include "apps/web-ui/http-server.h"
#include "apps/web-ui/startup-checks.h"
#include "common/flex-data.h"
#include "vpipe/pipeline-handle.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {
class SessionIntf;
class SessionContextIntf;
class Stage;
}

namespace vpipe::webui {

class WebUiDelegate;
class WebUiLogDelegate;
class SystemStatusPoller;

// REST controller over a single vpipe Session, backing the Pipeline
// Manager UI. Each managed pipeline keeps an editable C++ spec (the
// source of truth) plus a live materialized PipelineHandle. Editing
// is only permitted while the pipeline is stopped; a structural edit
// mutates the spec and rebuilds the live graph (create_pipeline +
// ordered insert_stage). Graph introspection reads the live Stage
// objects directly (type, ports, payload types, current config).
//
// All operations are serialized by an internal mutex.
class SessionApi {
public:
  // `ui`, when non-null, backs the /api/io/* routes (console + getline
  // rendezvous) -- the same delegate the session was given via
  // set_ui_delegate(). `log`, when non-null, backs the /api/log/*
  // routes (session log console + debug level) -- the delegate given
  // via set_log_delegate(). Either being null 404s its routes.
  explicit SessionApi(SessionIntf* session, WebUiDelegate* ui = nullptr,
                      WebUiLogDelegate* log = nullptr);
  // Out-of-line so pImpl members (SystemStatusPoller) can be
  // incomplete in this header.
  ~SessionApi();

  // Register every /api/* route on the given server.
  void register_routes(HttpServer& server);

  // Record the startup permission-check results (from
  // run_permission_checks) so the browser can display the same report
  // via GET /api/startup-checks. Called once at boot, after the probes
  // finish; serialized internally.
  void set_startup_checks(const std::vector<PermissionCheck>& checks);

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
    // Editable spec, kept roughly in insertion (dependency) order; an edge
    // rewire (h_connect_) can leave it non-topological, so to_flex_spec_
    // re-sorts on save (the core loader needs sources declared first).
    std::vector<StageSpec>      stages;
    std::optional<PipelineHandle> handle;        // live materialization
    State                       state = State::Stopped;
    std::string                 storage_path;
  };

  // ---- request handlers (return ready HttpResponses) -------------
  HttpResponse h_stage_types_(const HttpRequest&);
  HttpResponse h_list_pipelines_(const HttpRequest&);
  HttpResponse h_create_pipeline_(const HttpRequest&);
  HttpResponse h_rename_pipeline_(const HttpRequest&);
  HttpResponse h_load_pipeline_(const HttpRequest&);
  HttpResponse h_get_pipeline_(const HttpRequest&);
  HttpResponse h_save_pipeline_(const HttpRequest&);
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
  // Edge editing for the composer. connect re-points an existing
  // input port (in place, via PipelineHandle::move_iport) or appends a
  // new one (re-materialize); disconnect drops an input. Both are
  // refused unless the pipeline is stopped and return {id,state,graph}.
  // connect body: {from, from_port, to, to_port?}; disconnect body:
  // {to, to_port}.
  HttpResponse h_connect_(const HttpRequest&);
  HttpResponse h_disconnect_(const HttpRequest&);
  HttpResponse h_get_stage_config_(const HttpRequest&);
  HttpResponse h_set_stage_config_(const HttpRequest&);

  // User I/O console + interactive input (backed by _ui).
  HttpResponse h_io_console_(const HttpRequest&);
  HttpResponse h_io_pending_(const HttpRequest&);
  HttpResponse h_io_input_(const HttpRequest&);
  HttpResponse h_io_clear_(const HttpRequest&);
  // Console history cap (terminal-style scrollback bound). Returns
  // {max_console, min, max}; setter accepts {max_console: N}.
  HttpResponse h_io_limit_get_(const HttpRequest&);
  HttpResponse h_io_limit_set_(const HttpRequest&);

  // Session log console (backed by _log). h_log_console_ returns the
  // ring incrementally ({latest, lines:[{seq,level,text}]}); clear
  // drops history. Level get/set read & mutate the live capture
  // threshold ({level, levels:[...]}); the change affects only future
  // messages. Limit get/set is the log ring's scrollback cap
  // ({max_log, min, max}).
  HttpResponse h_log_console_(const HttpRequest&);
  HttpResponse h_log_clear_(const HttpRequest&);
  HttpResponse h_log_level_get_(const HttpRequest&);
  HttpResponse h_log_level_set_(const HttpRequest&);
  HttpResponse h_log_limit_get_(const HttpRequest&);
  HttpResponse h_log_limit_set_(const HttpRequest&);

  // Database browser. list/keys/value are read-only; delete-key and
  // drop mutate and are refused while any pipeline is non-stopped (a
  // running stage could be writing). list reports a `deletable` flag so
  // the view shows its delete controls only when mutation is allowed.
  HttpResponse h_db_list_(const HttpRequest&);
  HttpResponse h_db_keys_(const HttpRequest&);
  // Streaming value-filtered scan: writes NDJSON records (meta / row* /
  // done) incrementally as matches are found, so a large result set (up
  // to 64k rows) reaches the client progressively.
  void         h_db_scan_stream_(const HttpRequest&, ResponseStream&);
  HttpResponse h_db_value_(const HttpRequest&);
  HttpResponse h_db_delete_key_(const HttpRequest&);
  HttpResponse h_db_drop_(const HttpRequest&);

  // Installed (registered) models, enriched with catalogue metadata
  // (category / input-output modalities / parent linkage) for the web-ui
  // compatibility-aware model browser.
  HttpResponse h_models_installed_(const HttpRequest&);

  // System-level metrics for the bottom status bar (GPU util / memory
  // through IOKit + MLX). Always available; no auth state needed
  // beyond the existing /api/* gating.
  HttpResponse h_system_status_(const HttpRequest&);

  // Startup permission self-test results, set once at boot. The browser
  // fetches this when it connects and shows the report in a dialog.
  // Returns {ready, has_warnings, checks:[{name,status,detail,hints}]};
  // ready is false while the (blocking) probes are still running.
  HttpResponse h_startup_checks_(const HttpRequest&);

  // UI/message localization. GET returns {language, supported:[...]};
  // PUT {language} sets the session locale (normalized; 400 if
  // unsupported). The browser keeps its own UI string catalogue; this
  // shares the language so server-produced messages match the client.
  HttpResponse h_i18n_get_(const HttpRequest&);
  HttpResponse h_i18n_set_(const HttpRequest&);

  // Active HLS streams across every launched pipeline. Enumerates the
  // live "hls-broadcast" stages and reports each one's serving
  // coordinates so the User I/O workspace can embed a player.
  // {streams:[{pipeline,stage,state,playlist_name,port,bind_address}]}.
  HttpResponse h_hls_streams_(const HttpRequest&);

  // Active preview streams (live "preview" stages) for the User I/O
  // workspace's low-latency Preview view. Enumerates the live "preview"
  // stages and reports each one's coordinates + best-effort media hints:
  // {streams:[{pipeline,stage,state,title,video,audio,width,height}]}.
  HttpResponse h_preview_streams_(const HttpRequest&);
  // Long-lived preview WebSocket (single-origin transport; see
  // common/preview-channel.h for the message protocol). Resolves the live
  // "preview" stage named by the :pipeline/:stage path params, subscribes
  // to its PreviewChannel, and relays its messages (fMP4 init/fragments +
  // PCM) to the browser until the client disconnects or the stage stops.
  // Closes immediately when no such live stage exists.
  void h_preview_ws_(const HttpRequest&, WebSocket&);

  // Performance profiler: start/stop the session's per-worker event
  // capture and retrieve the captured timeline. start accepts optional
  // {max_events}; data returns the dump_profiling document (live while
  // capturing, else the snapshot taken at stop).
  HttpResponse h_profiler_start_(const HttpRequest&);
  HttpResponse h_profiler_stop_(const HttpRequest&);
  HttpResponse h_profiler_reset_(const HttpRequest&);
  HttpResponse h_profiler_status_(const HttpRequest&);
  HttpResponse h_profiler_data_(const HttpRequest&);

  // {cwd, files: [...]} -- top-level .vpipeline files in the
  // server's cwd, used to populate the Load-Pipeline dialog's
  // autocomplete.
  HttpResponse h_cwd_pipelines_(const HttpRequest&);
  // GET /api/fs/list?path= : one directory's entries for the file
  // open/save dialog, in the session's (possibly sandboxed) namespace.
  HttpResponse h_fs_list_(const HttpRequest&);

  // ---- internals -------------------------------------------------
  Pipe* find_(const std::string& id);

  // Resolve a stage by id within a pipe's live graph (nullptr if the
  // pipe isn't materialized or carries no such stage). Caller holds _mu.
  Stage* live_stage_(const Pipe& p, const std::string& id) const;

  // True if any managed pipeline is not in the Stopped state (running
  // or paused). Caller holds _mu. Gates the mutating database ops.
  bool any_pipeline_active_() const;

  // Auto-stop any managed pipeline whose stages have all signalled done
  // (its runtime drained on its own, with no pause/stop requested):
  // finalize it via stop_pipeline and flip the cached state to Stopped,
  // so a self-terminating pipeline shows as stopped without an explicit
  // stop request. Caller holds _mu. Invoked from the status-read
  // handlers (list / get / buffer-status) the UI polls.
  void reap_completed_();

  // Snapshot the live profiling buffers to a JSON string. There is no
  // in-memory dump API, so this dumps to a temp file (dump_profiling)
  // and reads it back. Empty + `err` set on failure. Caller holds _mu.
  std::string profiler_dump_json_(std::string& err);
  // {enabled, max_events_per_thread, has_data}. Caller holds _mu.
  FlexData profiler_status_doc_() const;

  // (Re)build the live pipeline from `p.stages`. On failure the live
  // handle is left unloaded and `err` is set. Caller holds _mu.
  bool materialize_(Pipe& p, std::string& err);
  // Drop the live handle (unload from the session).
  void dematerialize_(Pipe& p);

  // FlexData views of a pipe.
  FlexData pipe_summary_(const Pipe& p) const;     // {id,state,...}
  FlexData graph_json_(const Pipe& p) const;        // {nodes,edges}
  FlexData to_flex_spec_(const Pipe& p) const;      // pipeline-spec doc
  // Stage indices in dependency order (Kahn, stable wrt the input order).
  // The core pipeline_from_spec loader resolves iports in a single pass, so
  // a saved spec MUST declare each stage after its sources; in-memory edits
  // (h_connect_ rewiring an earlier stage onto a later source) can perturb
  // Pipe::stages, so to_flex_spec_ re-sorts here rather than trusting it.
  static std::vector<std::size_t> topo_order_(
      const std::vector<StageSpec>& stages);

  static const char* state_name_(State s);

  SessionIntf*                       _session;
  SessionContextIntf*                _sctx;   // same object, context facet
  WebUiDelegate*                     _ui;
  WebUiLogDelegate*                  _log;
  std::mutex                         _mu;
  // Serialises the DB-browser handlers: the HTTP server now serves
  // requests on per-connection threads, and LMDB's mdb_dbi_open must
  // not run from concurrent transactions. Lock order, where both are
  // held (delete/drop): _mu before _db_mu.
  std::mutex                         _db_mu;
  std::vector<std::unique_ptr<Pipe>> _pipes;
  // Stateful poller for the bottom status bar: owns the IOReport
  // "Energy Model" subscription used to derive ANE power. Created
  // lazily by SessionApi's ctor so a non-Apple build doesn't pay for
  // it (the class is Apple-only by virtue of its .cc file).
  std::unique_ptr<SystemStatusPoller> _status;
  // Last profiling capture (dump_profiling JSON), retained after stop
  // so the timeline survives disabling (which frees the buffers).
  std::string                        _profiler_snapshot;
  // Startup permission-check report (FlexData object), set once at boot
  // by set_startup_checks and served by /api/startup-checks. Null (not
  // an object) until the probes finish -> the endpoint reports ready:false.
  FlexData                           _startup_checks;
};

}

#endif
