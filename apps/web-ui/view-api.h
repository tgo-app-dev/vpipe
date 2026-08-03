// view-api.h -- /api/ui/views and the per-panel WebSocket channel.
//
// These are panels a STAGE ships with itself rather than the app
// shipping them (see ui/ui-view-registry.h). The app's whole role is to
// advertise what is registered, serve the module bytes (via
// serve_static_'s registry lookup), and FORWARD the FlexData protocol
// -- it never interprets a view's messages.
//
// The host a backend resolves (pipeline, stage) pairs against belongs
// to PipelineApi, which owns the graphs; this controller only hands it
// over.

#ifndef WEBUI_VIEW_API_H
#define WEBUI_VIEW_API_H

#include "apps/web-ui/api-context.h"
#include "apps/web-ui/http-server.h"
#include "apps/web-ui/pipeline-api.h"

namespace vpipe::webui {

class ViewApi {
public:
  ViewApi(ApiContext& ctx, PipelineApi& pipelines)
    : _ctx(ctx), _pipelines(pipelines) {}

  void register_routes(HttpServer& s);

private:
  // Every registered view, for the front end's panel registry:
  // {views:[{id,stage_type,module,styles,label_key,icon}]}.
  HttpResponse h_views_(const HttpRequest&);

  // Long-lived per-panel channel. Instantiates the :view backend, then
  // pumps in both directions on this one thread until the client goes
  // away: inbound client frames become UiViewBackendIntf::on_message,
  // outbound backend messages become text frames (JSON) and bulk
  // payloads become binary frames ([u32 header_len][header][payload]).
  void h_view_ws_(const HttpRequest&, WebSocket&);

  UiViewHostIntf* view_host() noexcept { return _pipelines.view_host(); }

  ApiContext&  _ctx;
  PipelineApi& _pipelines;
};

}

#endif
