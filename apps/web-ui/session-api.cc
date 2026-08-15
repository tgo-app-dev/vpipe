#include "apps/web-ui/session-api.h"

#include "interfaces/session-context-intf.h"
#include "vpipe/session-intf.h"

#include <random>
#include <string>

using namespace std;

namespace vpipe::webui {

SessionApi::SessionApi(SessionIntf* session, WebUiDelegate* ui,
                       WebUiLogDelegate* log)
{
  _ctx.session = session;
  _ctx.sctx    = dynamic_cast<SessionContextIntf*>(session);
  _ctx.ui      = ui;
  _ctx.log     = log;
}

// Out-of-line so the controllers' incomplete pImpl members (notably
// SystemApi's SystemStatusPoller) do not have to be complete in the
// header.
SessionApi::~SessionApi() = default;

void
SessionApi::set_startup_checks(const std::vector<PermissionCheck>& checks)
{
  _system.set_startup_checks(checks);
}

void
SessionApi::register_routes(HttpServer& s)
{
  _ctx.server = &s;

  // The one route with no subject of its own.
  s.route("GET", "/api/health",
          [](const HttpRequest&) {
            // Per-process instance token. The web-ui client polls
            // /api/health and hard-reloads a (localhost) page when this
            // changes -- i.e. when the server has been restarted -- so a
            // stale page refreshes itself against the new server.
            static const std::string inst = [] {
              static const char hex[] = "0123456789abcdef";
              std::random_device rd;
              std::uniform_int_distribution<int> d(0, 15);
              std::string t;
              t.reserve(16);
              for (int i = 0; i < 16; ++i) { t.push_back(hex[d(rd)]); }
              return t;
            }();
            return HttpResponse::json(
                200, "{\"ok\":true,\"instance\":\"" + inst + "\"}");
          });

  // Each controller registers its own subject. IoApi and LogApi
  // register nothing when the session was given no matching delegate,
  // so those paths 404 rather than answering for a console that does
  // not exist.
  _pipelines.register_routes(s);
  _db.register_routes(s);
  _files.register_routes(s);
  _io.register_routes(s);
  _logs.register_routes(s);
  _plugins.register_routes(s);
  _profiler.register_routes(s);
  _system.register_routes(s);
  _views.register_routes(s);
}

}
