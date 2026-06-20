#ifndef WEBUI_HTTP_SERVER_H
#define WEBUI_HTTP_SERVER_H

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vpipe::webui {

// Parsed HTTP request handed to a route handler.
struct HttpRequest {
  std::string method;   // upper-case verb, e.g. "GET", "POST"
  std::string path;     // decoded path, no query, e.g. "/api/pipelines"
  std::string query;    // raw query string after '?', may be empty
  std::string body;     // request entity body (Content-Length bytes)
  // Header names are stored lower-cased.
  std::unordered_map<std::string, std::string> headers;
  // Path parameters captured from ':name' route segments.
  std::unordered_map<std::string, std::string> params;
};

struct HttpResponse {
  int                                              status = 200;
  std::string                                      content_type =
      "application/json";
  std::string                                      body;
  std::vector<std::pair<std::string, std::string>> extra_headers;

  static HttpResponse json(int status, std::string body);
  // status-only JSON {"ok":true} / {"error":"..."} convenience.
  static HttpResponse ok();
  static HttpResponse error(int status, const std::string& message);
};

// Minimal single-threaded-accept HTTP/1.1 server for the web-ui app.
// One accept loop; each connection is read, dispatched, answered, and
// closed (Connection: close, no keep-alive). Routes are matched by
// method + path-segment pattern with ':name' captures. Any unmatched
// GET/HEAD falls back to the doc_root: an existing file is served
// verbatim, otherwise index.html is returned so the single-page app
// can own client-side routing. Every response carries permissive CORS
// headers and OPTIONS is answered with a preflight 204.
//
// This is intentionally self-contained (not a SessionMember) so the
// app's HTTP layer stays decoupled from the vpipe session; it logs to
// stderr. Not built for hostile input or high concurrency -- it backs
// a local single-operator control UI.
class HttpServer {
public:
  using Handler = std::function<HttpResponse(const HttpRequest&)>;

  HttpServer(std::string bind_address, int port, std::string doc_root);
  ~HttpServer();

  HttpServer(const HttpServer&)            = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // Register a handler. `pattern` is an absolute path with optional
  // ':name' capture segments, e.g. "/api/pipelines/:id/launch".
  void route(std::string method, std::string pattern, Handler h);

  // First-cut auth: when set non-empty, requests to /api/* arriving
  // from a non-loopback peer must carry header `X-Auth-Key` equal to
  // this key (else 401). Loopback clients and static assets are never
  // gated, so the page can always load and prompt for the key. Empty
  // (default) disables the gate entirely.
  void set_auth_key(std::string key) { _auth_key = std::move(key); }

  // Bind + listen + spawn the accept thread. False on bind/listen
  // error (logged to stderr).
  bool start();
  void stop();                       // idempotent; dtor calls it
  int  bound_port() const noexcept { return _bound_port; }

private:
  struct Route {
    std::string              method;
    std::vector<std::string> segments;   // pattern split on '/'
    Handler                  handler;
  };

  void        accept_loop_();
  void        serve_one_(int client_fd, bool loopback);
  bool        dispatch_(const HttpRequest& req, HttpResponse& out) const;
  HttpResponse serve_static_(const HttpRequest& req) const;

  std::string        _bind_address;
  int                _port;
  int                _bound_port = 0;
  std::string        _doc_root;
  std::string        _auth_key;
  int                _listen_fd  = -1;
  std::atomic<bool>  _stop{ false };
  std::thread        _thread;
  // Each accepted connection is served on its own detached thread so a
  // slow handler (a multi-second pipeline-stop drain) can't starve the
  // others -- the status bar, console and DB views stay responsive.
  // _inflight counts live handler threads so stop() can wait for them.
  std::atomic<int>   _inflight{ 0 };
  std::vector<Route> _routes;
};

}

#endif
