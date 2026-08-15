// plugin-api.h -- the plugin panel's endpoints: what is available under
// the work directory's plugins/, what is loaded, and the two actions a
// session may take on them.
//
// GET  /api/plugins          discovered + loaded, with the root's path
// POST /api/plugins/load     {path}         -- dlopen + register
// POST /api/plugins/enabled  {name, on}     -- offer / withhold its stages
//
// THERE IS NO UNLOAD, and that is a property of the host rather than an
// omission here. PluginManager never dlclose's: StageRegistry holds raw
// factory pointers into the dylib with no removal API, the StageSpec*
// handed to /api/stage-types points at its static storage, live stage
// instances hold vtables there, and the metal / video-family /
// VAE-family / catalogue registrations all reference it. Unmapping that
// under a running process is a use-after-unmap, not a reclaim.
//
// `enabled` is the honest neighbour: the plugin stays loaded and mapped,
// and the composer stops offering its stages. The UI says so in those
// words rather than calling it an unload.

#ifndef WEBUI_PLUGIN_API_H
#define WEBUI_PLUGIN_API_H

#include "apps/web-ui/api-context.h"
#include "apps/web-ui/http-server.h"

namespace vpipe::webui {

class PluginApi {
public:
  explicit PluginApi(ApiContext& ctx) : _ctx(ctx) {}

  void register_routes(HttpServer& s);

private:
  HttpResponse h_list_(const HttpRequest& r);
  HttpResponse h_load_(const HttpRequest& r);
  HttpResponse h_enabled_(const HttpRequest& r);

  ApiContext& _ctx;
};

}  // namespace vpipe::webui

#endif
