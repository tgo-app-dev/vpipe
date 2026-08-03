// log-api.h -- /api/log/* : the session log console, backed by the
// session's WebUiLogDelegate.
//
// Registers nothing when the session was given no log delegate, so the
// routes 404 rather than answering for a log that does not exist.

#ifndef WEBUI_LOG_API_H
#define WEBUI_LOG_API_H

#include "apps/web-ui/api-context.h"
#include "apps/web-ui/http-server.h"

namespace vpipe::webui {

class LogApi {
public:
  explicit LogApi(ApiContext& ctx) : _ctx(ctx) {}

  void register_routes(HttpServer& s);

private:
  // Returns the ring incrementally ({latest, lines:[{seq,level,text}]});
  // clear drops history.
  HttpResponse h_console_(const HttpRequest&);
  HttpResponse h_clear_(const HttpRequest&);
  // Level get/set read & mutate the live capture threshold
  // ({level, levels:[...]}); the change affects only future messages.
  HttpResponse h_level_get_(const HttpRequest&);
  HttpResponse h_level_set_(const HttpRequest&);
  // The log ring's scrollback cap ({max_log, min, max}).
  HttpResponse h_limit_get_(const HttpRequest&);
  HttpResponse h_limit_set_(const HttpRequest&);

  ApiContext& _ctx;
};

}

#endif
