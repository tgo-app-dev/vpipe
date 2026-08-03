// system-api.h -- session-wide status endpoints that belong to no one
// subject: /api/health, /api/system/status, /api/startup-checks,
// /api/i18n and /api/hls/streams.
//
// The HLS enumeration is the one that reaches outside itself: it walks
// the live graphs through PipelineApi to report where each launched
// hls-broadcast stage is serving.

#ifndef WEBUI_SYSTEM_API_H
#define WEBUI_SYSTEM_API_H

#include "apps/web-ui/api-context.h"
#include "apps/web-ui/http-server.h"
#include "apps/web-ui/pipeline-api.h"
#include "apps/web-ui/startup-checks.h"
#include "common/flex-data.h"

#include <memory>
#include <vector>

namespace vpipe::webui {

class SystemStatusPoller;

class SystemApi {
public:
  SystemApi(ApiContext& ctx, PipelineApi& pipelines);
  // Out-of-line so SystemStatusPoller can be incomplete here.
  ~SystemApi();

  void register_routes(HttpServer& s);

  // Record the startup permission-check results (from
  // run_permission_checks) so the browser can display the same report.
  // Called once at boot, after the probes finish; serialized
  // internally.
  void set_startup_checks(const std::vector<PermissionCheck>& checks);

private:
  // System-level metrics for the bottom status bar (GPU util / memory
  // through IOKit). Always available; no auth state needed beyond the
  // existing /api/* gating.
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

  ApiContext&  _ctx;
  PipelineApi& _pipelines;
  // Stateful poller for the bottom status bar: owns the IOReport
  // "Energy Model" subscription used to derive ANE power. Created in
  // the ctor so a non-Apple build doesn't pay for it (the class is
  // Apple-only by virtue of its .cc file).
  std::unique_ptr<SystemStatusPoller> _status;
  // Startup permission-check report (FlexData object), set once at boot
  // and served by /api/startup-checks. Null (not an object) until the
  // probes finish -> the endpoint reports ready:false.
  FlexData     _startup_checks;
};

}

#endif
