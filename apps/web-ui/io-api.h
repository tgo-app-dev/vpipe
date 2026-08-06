// io-api.h -- /api/io/* : the user-I/O console and the interactive
// getline rendezvous, backed by the session's WebUiDelegate.
//
// Registers nothing when the session was given no UI delegate, so the
// routes 404 rather than answering for a console that does not exist.

#ifndef WEBUI_IO_API_H
#define WEBUI_IO_API_H

#include "apps/web-ui/api-context.h"
#include "apps/web-ui/http-server.h"

namespace vpipe::webui {

class IoApi {
public:
  explicit IoApi(ApiContext& ctx) : _ctx(ctx) {}

  void register_routes(HttpServer& s);

private:
  HttpResponse h_console_(const HttpRequest&);
  HttpResponse h_progress_(const HttpRequest&);
  HttpResponse h_pending_(const HttpRequest&);
  HttpResponse h_input_(const HttpRequest&);
  HttpResponse h_clear_(const HttpRequest&);
  HttpResponse h_interrupt_(const HttpRequest&);
  // Console history cap (terminal-style scrollback bound). Returns
  // {max_console, min, max}; setter accepts {max_console: N}.
  HttpResponse h_limit_get_(const HttpRequest&);
  HttpResponse h_limit_set_(const HttpRequest&);

  ApiContext& _ctx;
};

}

#endif
