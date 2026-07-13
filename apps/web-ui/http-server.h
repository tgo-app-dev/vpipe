#ifndef WEBUI_HTTP_SERVER_H
#define WEBUI_HTTP_SERVER_H

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <sys/types.h>
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

// Byte transport for one accepted connection: plaintext (a raw socket) or
// TLS (an OpenSSL session over the socket). All client I/O in the server
// goes through this, so the TLS path is transparent to the request/response
// code. The Conn does NOT own the underlying fd -- the accept loop closes
// it after the Conn is torn down.
class Conn {
public:
  virtual ~Conn() = default;
  // Read up to `n` bytes into `buf`. Returns >0 = bytes read, 0 = clean
  // EOF / peer closed, <0 = error.
  virtual ssize_t read(void* buf, std::size_t n) = 0;
  // Write all `len` bytes; returns false once the peer / link has gone
  // away (so a streaming producer can stop early).
  virtual bool write_all(const char* data, std::size_t len) = 0;
};

// Incremental (chunk-as-you-go) response writer for streaming routes.
// The body is delimited by connection close (HTTP/1.1 Connection: close,
// no Content-Length) -- fetch()'s ReadableStream surfaces the bytes to
// the client as they arrive. begin() sends the status line + headers
// once; write() appends body bytes and returns false once the peer has
// gone away, so a long-running producer can stop early.
class ResponseStream {
public:
  explicit ResponseStream(Conn& conn) : _conn(&conn) {}

  // Send the status line + headers (CORS, Connection: close, no
  // Content-Length). Idempotent -- the first call wins.
  void begin(int status = 200,
             const std::string& content_type = "application/x-ndjson");
  // Append body bytes (auto-begins with defaults if needed). Returns
  // false once the connection is gone.
  bool write(std::string_view bytes);
  bool alive() const noexcept { return _alive; }

private:
  Conn* _conn;
  bool  _begun = false;
  bool  _alive = true;
};

// A negotiated WebSocket connection over a Conn (plaintext or TLS). The
// server side only sends -- binary messages (fMP4 init/fragments + PCM for
// the Preview view) -- and detects the peer going away via write failure.
// Frames are unmasked (server->client) per RFC 6455.
class WebSocket {
public:
  explicit WebSocket(Conn& conn) : _conn(&conn) {}

  // Send the 101 handshake response for the client's Sec-WebSocket-Key.
  // Returns false if the write failed.
  bool handshake(const std::string& sec_key);

  // Send one binary / text message. Returns false once the peer is gone.
  bool send_binary(const std::uint8_t* data, std::size_t len);
  bool send_text(std::string_view s);
  bool alive() const noexcept { return _alive; }

private:
  bool send_frame_(std::uint8_t opcode, const std::uint8_t* data,
                   std::size_t len);
  Conn* _conn;
  bool  _alive = true;
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
class TlsContext;   // apps/web-ui/tls-context.h (only when built with TLS)

class HttpServer {
public:
  using Handler = std::function<HttpResponse(const HttpRequest&)>;
  // Streaming handler: owns the response, writing incrementally through
  // the ResponseStream (see above). Used for potentially large / slow
  // results (e.g. a value-filtered DB scan) that should reach the client
  // as they are produced rather than buffered whole.
  using StreamHandler =
      std::function<void(const HttpRequest&, ResponseStream&)>;
  // WebSocket handler: owns the connection after the handshake, sending
  // messages through the WebSocket until the peer disconnects.
  using WsHandler = std::function<void(const HttpRequest&, WebSocket&)>;

  HttpServer(std::string bind_address, int port, std::string doc_root);
  ~HttpServer();

  HttpServer(const HttpServer&)            = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // Register a handler. `pattern` is an absolute path with optional
  // ':name' capture segments, e.g. "/api/pipelines/:id/launch".
  void route(std::string method, std::string pattern, Handler h);
  // Register a streaming handler (same pattern syntax). Streaming routes
  // are matched before buffered routes and take over the connection.
  void route_stream(std::string method, std::string pattern,
                    StreamHandler h);
  // Register a WebSocket route (GET only; the `Upgrade: websocket` request
  // is matched by path). WS routes are dispatched before stream/buffered
  // routes. Because browsers cannot set request headers on a WebSocket,
  // the /api/* auth key is accepted as a `?key=` query parameter here.
  void route_ws(std::string pattern, WsHandler h);

  // First-cut auth: when set non-empty, requests to /api/* arriving
  // from a non-loopback peer must carry header `X-Auth-Key` equal to
  // this key (else 401). Loopback clients and static assets are never
  // gated, so the page can always load and prompt for the key. Empty
  // (default) disables the gate entirely.
  void set_auth_key(std::string key) { _auth_key = std::move(key); }

  // Enable HTTPS. Loads (or generates + caches) a self-signed certificate
  // under `cache_dir`, whose SubjectAltName covers localhost + `bind_hint`
  // so the browser reaches a matching secure origin. MUST be called before
  // start(). Returns false and sets *err on failure -- including a build
  // without OpenSSL, where TLS support is compiled out. When enabled every
  // accepted connection is TLS-wrapped; plaintext clients are dropped.
  //
  // Why it matters beyond encryption: browser WebCodecs (the low-latency
  // Preview view) is only exposed in a SECURE CONTEXT, i.e. HTTPS or
  // localhost -- so a LAN viewer needs this to use Preview at all.
  bool enable_tls(const std::string& cache_dir,
                  const std::string& bind_hint, std::string* err);
  bool tls_enabled() const noexcept;

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
  struct StreamRoute {
    std::string              method;
    std::vector<std::string> segments;
    StreamHandler            handler;
  };
  struct WsRoute {
    std::vector<std::string> segments;
    WsHandler                handler;
  };

  void        accept_loop_();
  void        serve_one_(Conn& conn, bool loopback);
  bool        dispatch_(const HttpRequest& req, HttpResponse& out) const;
  // Returns true (and writes the whole response through `conn`) if a
  // streaming route matched; false if none did.
  bool        dispatch_stream_(const HttpRequest& req, Conn& conn) const;
  // Returns true if the request is a WebSocket upgrade matching a ws route
  // (handshake done + handler run through `conn`); false otherwise.
  bool        dispatch_ws_(const HttpRequest& req, Conn& conn) const;
  HttpResponse serve_static_(const HttpRequest& req) const;

  std::string        _bind_address;
  int                _port;
  int                _bound_port = 0;
  std::string        _doc_root;
  std::string        _auth_key;
  // Non-null when HTTPS is enabled (enable_tls). Guarded so a build
  // without OpenSSL never instantiates the incomplete type.
#ifdef VPIPE_WEBUI_TLS
  std::unique_ptr<TlsContext> _tls;
#endif
  int                _listen_fd  = -1;
  std::atomic<bool>  _stop{ false };
  std::thread        _thread;
  // Each accepted connection is served on its own detached thread so a
  // slow handler (a multi-second pipeline-stop drain) can't starve the
  // others -- the status bar, console and DB views stay responsive.
  // _inflight counts live handler threads so stop() can wait for them.
  std::atomic<int>   _inflight{ 0 };
  std::vector<Route>       _routes;
  std::vector<StreamRoute> _stream_routes;
  std::vector<WsRoute>     _ws_routes;
};

}

#endif
