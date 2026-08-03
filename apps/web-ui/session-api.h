#ifndef WEBUI_SESSION_API_H
#define WEBUI_SESSION_API_H

#include "apps/web-ui/api-context.h"
#include "apps/web-ui/database-api.h"
#include "apps/web-ui/file-api.h"
#include "apps/web-ui/http-server.h"
#include "apps/web-ui/io-api.h"
#include "apps/web-ui/log-api.h"
#include "apps/web-ui/pipeline-api.h"
#include "apps/web-ui/profiler-api.h"
#include "apps/web-ui/startup-checks.h"
#include "apps/web-ui/system-api.h"
#include "apps/web-ui/view-api.h"
#include "interfaces/ui-view-intf.h"

#include <vector>

namespace vpipe {
class SessionIntf;
}

namespace vpipe::webui {

class WebUiDelegate;
class WebUiLogDelegate;

// The /api surface over a single vpipe Session, backing the Pipeline
// Manager UI.
//
// This class used to hold every handler. It is now a composition root:
// it owns the shared ApiContext, constructs one controller per subject,
// and hands each of them the server to register on. Nothing is served
// from here except /api/health, which belongs to no subject.
//
// Where to look for a route: /api/pipelines and /api/stage-types ->
// PipelineApi; /api/db and /api/models/installed -> DatabaseApi;
// /api/fs and /api/cwd-pipelines -> FileApi; /api/io -> IoApi;
// /api/log -> LogApi; /api/profiler -> ProfilerApi; /api/system,
// /api/startup-checks, /api/i18n and /api/hls -> SystemApi;
// /api/ui/views -> ViewApi.
//
// All operations are still serialized by ONE mutex, which lives in the
// ApiContext -- see api-context.h for why that was not split up.
class SessionApi {
public:
  // `ui`, when non-null, backs the /api/io/* routes (console + getline
  // rendezvous) -- the same delegate the session was given via
  // set_ui_delegate(). `log`, when non-null, backs the /api/log/*
  // routes (session log console + debug level) -- the delegate given
  // via set_log_delegate(). Either being null 404s its routes.
  explicit SessionApi(SessionIntf* session, WebUiDelegate* ui = nullptr,
                      WebUiLogDelegate* log = nullptr);
  // Out-of-line so a controller's pImpl member (SystemApi's
  // SystemStatusPoller) can stay incomplete in this header.
  ~SessionApi();

  // Register every /api/* route on the given server.
  void register_routes(HttpServer& server);

  // Record the startup permission-check results (from
  // run_permission_checks) so the browser can display the same report
  // via GET /api/startup-checks. Called once at boot, after the probes
  // finish; serialized internally.
  void set_startup_checks(const std::vector<PermissionCheck>& checks);

  // The view host backing this app's stage-provided panels. Wired into
  // the UI delegate at startup so a stage can reach it through its
  // session (UiDelegateIntf::ui_view_host). Answered by PipelineApi,
  // which owns the pipeline state the panels resolve against.
  UiViewHostIntf* view_host() noexcept { return _pipelines.view_host(); }

private:
  // Session, delegates, server and THE shared lock. See api-context.h
  // for why one mutex still covers every controller.
  ApiContext                         _ctx;

  // ---- per-subject controllers -----------------------------------
  PipelineApi                        _pipelines{_ctx};
  DatabaseApi                        _db{_ctx, _pipelines};
  FileApi                            _files{_ctx};
  IoApi                              _io{_ctx};
  LogApi                             _logs{_ctx};
  ProfilerApi                        _profiler{_ctx};
  SystemApi                          _system{_ctx, _pipelines};
  ViewApi                            _views{_ctx, _pipelines};
};

}

#endif
