#include "apps/web-ui/http-server.h"
#include "apps/web-ui/embedded-assets.h"
#ifdef VPIPE_WEBUI_TLS
#include "apps/web-ui/tls-context.h"
#endif

#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <netinet/in.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

namespace vpipe::webui {

namespace {

string
lower_(string s)
{
  for (char& c : s) {
    c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

// Percent-decode a URL path component. Leaves malformed escapes as-is.
string
url_decode_(string_view s)
{
  string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size() && isxdigit((unsigned char)s[i + 1])
        && isxdigit((unsigned char)s[i + 2])) {
      auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') { return c - '0'; }
        if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
        return c - 'A' + 10;
      };
      out.push_back(static_cast<char>(hex(s[i + 1]) * 16 + hex(s[i + 2])));
      i += 2;
    } else if (s[i] == '+') {
      out.push_back(' ');
    } else {
      out.push_back(s[i]);
    }
  }
  return out;
}

vector<string>
split_segments_(string_view path)
{
  vector<string> out;
  size_t i = 0;
  while (i < path.size()) {
    if (path[i] == '/') { ++i; continue; }
    size_t j = path.find('/', i);
    if (j == string_view::npos) { j = path.size(); }
    out.emplace_back(path.substr(i, j - i));
    i = j;
  }
  return out;
}

const char*
reason_phrase_(int code)
{
  switch (code) {
    case 200: return "OK";
    case 201: return "Created";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 500: return "Internal Server Error";
    default:  return "OK";
  }
}

string
mime_for_(string_view path)
{
  auto ends = [&](string_view sfx) {
    return path.size() >= sfx.size()
        && path.compare(path.size() - sfx.size(), sfx.size(), sfx) == 0;
  };
  if (ends(".html") || ends(".htm")) { return "text/html; charset=utf-8"; }
  if (ends(".css"))  { return "text/css; charset=utf-8"; }
  if (ends(".js"))   { return "application/javascript; charset=utf-8"; }
  if (ends(".json")) { return "application/json"; }
  if (ends(".svg"))  { return "image/svg+xml"; }
  if (ends(".png"))  { return "image/png"; }
  if (ends(".ico"))  { return "image/x-icon"; }
  if (ends(".txt"))  { return "text/plain; charset=utf-8"; }
  return "application/octet-stream";
}

bool
send_all_(int fd, const char* data, size_t len)
{
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(fd, data + sent, len - sent, 0);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) { continue; }
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return true;
}

// Plaintext transport: raw socket I/O with EINTR retry. Writes are
// SIGPIPE-safe because accept_loop_ sets SO_NOSIGPIPE on the fd. Does NOT
// own the fd (the accept loop closes it).
class PlainConn : public Conn {
public:
  explicit PlainConn(int fd) : _fd(fd) {}
  ssize_t
  read(void* buf, size_t n) override
  {
    for (;;) {
      ssize_t r = ::recv(_fd, buf, n, 0);
      if (r < 0 && errno == EINTR) { continue; }
      return r;
    }
  }
  bool
  write_all(const char* data, size_t len) override
  {
    return send_all_(_fd, data, len);
  }
private:
  int _fd;
};

// SHA-1 (RFC 3174), for the WebSocket accept key. Small + dependency-free
// so the server stays portable (no CommonCrypto / OpenSSL needed here).
void
sha1_(const uint8_t* data, size_t len, uint8_t out[20])
{
  uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE,
           h3 = 0x10325476, h4 = 0xC3D2E1F0;
  vector<uint8_t> m(data, data + len);
  const uint64_t bitlen = static_cast<uint64_t>(len) * 8;
  m.push_back(0x80);
  while (m.size() % 64 != 56) { m.push_back(0); }
  for (int i = 7; i >= 0; --i) {
    m.push_back(static_cast<uint8_t>(bitlen >> (i * 8)));
  }
  auto rol = [](uint32_t x, int c) { return (x << c) | (x >> (32 - c)); };
  for (size_t off = 0; off < m.size(); off += 64) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(m[off + i * 4]) << 24)
           | (uint32_t(m[off + i * 4 + 1]) << 16)
           | (uint32_t(m[off + i * 4 + 2]) << 8)
           | uint32_t(m[off + i * 4 + 3]);
    }
    for (int i = 16; i < 80; ++i) {
      w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }
    uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
    for (int i = 0; i < 80; ++i) {
      uint32_t f, k;
      if (i < 20)      { f = (b & c) | (~b & d);          k = 0x5A827999; }
      else if (i < 40) { f = b ^ c ^ d;                   k = 0x6ED9EBA1; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
      else             { f = b ^ c ^ d;                   k = 0xCA62C1D6; }
      const uint32_t t = rol(a, 5) + f + e + k + w[i];
      e = d; d = c; c = rol(b, 30); b = a; a = t;
    }
    h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
  }
  const uint32_t hs[5] = { h0, h1, h2, h3, h4 };
  for (int i = 0; i < 5; ++i) {
    out[i * 4]     = (hs[i] >> 24) & 0xff;
    out[i * 4 + 1] = (hs[i] >> 16) & 0xff;
    out[i * 4 + 2] = (hs[i] >> 8) & 0xff;
    out[i * 4 + 3] = hs[i] & 0xff;
  }
}

string
base64_(const uint8_t* data, size_t len)
{
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                         "0123456789+/";
  string out;
  size_t i = 0;
  for (; i + 3 <= len; i += 3) {
    const uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8)
                     | uint32_t(data[i + 2]);
    out.push_back(t[(n >> 18) & 63]);
    out.push_back(t[(n >> 12) & 63]);
    out.push_back(t[(n >> 6) & 63]);
    out.push_back(t[n & 63]);
  }
  if (len - i == 1) {
    const uint32_t n = uint32_t(data[i]) << 16;
    out.push_back(t[(n >> 18) & 63]);
    out.push_back(t[(n >> 12) & 63]);
    out += "==";
  } else if (len - i == 2) {
    const uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8);
    out.push_back(t[(n >> 18) & 63]);
    out.push_back(t[(n >> 12) & 63]);
    out.push_back(t[(n >> 6) & 63]);
    out.push_back('=');
  }
  return out;
}

// Extract a query-string parameter value (url-decoded), or "" if absent.
string
query_param_(string_view query, string_view name)
{
  size_t i = 0;
  while (i < query.size()) {
    const size_t amp = query.find('&', i);
    const size_t end = (amp == string_view::npos) ? query.size() : amp;
    string_view kv = query.substr(i, end - i);
    const size_t eq = kv.find('=');
    const string_view k = (eq == string_view::npos) ? kv : kv.substr(0, eq);
    if (k == name) {
      return (eq == string_view::npos) ? string()
                                       : url_decode_(kv.substr(eq + 1));
    }
    if (amp == string_view::npos) { break; }
    i = amp + 1;
  }
  return {};
}

}  // namespace

void
ResponseStream::begin(int status, const string& content_type)
{
  if (_begun) { return; }
  _begun = true;
  ostringstream os;
  os << "HTTP/1.1 " << status << ' ' << reason_phrase_(status) << "\r\n";
  if (!content_type.empty()) {
    os << "Content-Type: " << content_type << "\r\n";
  }
  os << "Access-Control-Allow-Origin: *\r\n";
  os << "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n";
  os << "Access-Control-Allow-Headers: Content-Type, X-Auth-Key\r\n";
  // No Content-Length: the body is delimited by connection close. Disable
  // any intermediary buffering so rows reach the browser as they stream.
  os << "Cache-Control: no-store\r\n";
  os << "X-Accel-Buffering: no\r\n";
  os << "Connection: close\r\n\r\n";
  string head = os.str();
  if (!_conn->write_all(head.data(), head.size())) { _alive = false; }
}

bool
ResponseStream::write(string_view bytes)
{
  if (!_begun) { begin(); }
  if (!_alive || bytes.empty()) { return _alive; }
  if (!_conn->write_all(bytes.data(), bytes.size())) { _alive = false; }
  return _alive;
}

HttpResponse
HttpResponse::json(int status, string body)
{
  HttpResponse r;
  r.status       = status;
  r.content_type = "application/json";
  r.body         = std::move(body);
  return r;
}

HttpResponse
HttpResponse::ok()
{
  return json(200, "{\"ok\":true}");
}

HttpResponse
HttpResponse::error(int status, const string& message)
{
  // Minimal JSON string escaping for the message.
  string esc;
  for (char c : message) {
    if (c == '"' || c == '\\') { esc.push_back('\\'); esc.push_back(c); }
    else if (c == '\n')        { esc += "\\n"; }
    else if (c == '\r')        { /* drop */ }
    else                       { esc.push_back(c); }
  }
  return json(status, "{\"error\":\"" + esc + "\"}");
}

bool
WebSocket::handshake(const string& sec_key)
{
  static const char* kGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
  const string concat = sec_key + kGuid;
  uint8_t dig[20];
  sha1_(reinterpret_cast<const uint8_t*>(concat.data()), concat.size(), dig);
  const string accept = base64_(dig, 20);
  const string resp =
      "HTTP/1.1 101 Switching Protocols\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      "Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
  if (!_conn->write_all(resp.data(), resp.size())) { _alive = false; }
  return _alive;
}

bool
WebSocket::send_frame_(uint8_t opcode, const uint8_t* data, size_t len)
{
  if (!_alive) { return false; }
  uint8_t hdr[10];
  size_t  hl = 0;
  hdr[0] = static_cast<uint8_t>(0x80 | opcode);   // FIN + opcode
  if (len < 126) {
    hdr[1] = static_cast<uint8_t>(len);
    hl = 2;
  } else if (len <= 0xFFFF) {
    hdr[1] = 126;
    hdr[2] = static_cast<uint8_t>((len >> 8) & 0xff);
    hdr[3] = static_cast<uint8_t>(len & 0xff);
    hl = 4;
  } else {
    hdr[1] = 127;
    for (int i = 0; i < 8; ++i) {
      hdr[2 + i] =
          static_cast<uint8_t>((static_cast<uint64_t>(len) >> ((7 - i) * 8)));
    }
    hl = 10;
  }
  if (!_conn->write_all(reinterpret_cast<const char*>(hdr), hl)) {
    _alive = false;
    return false;
  }
  if (len > 0
      && !_conn->write_all(reinterpret_cast<const char*>(data), len)) {
    _alive = false;
    return false;
  }
  return true;
}

bool
WebSocket::send_binary(const uint8_t* data, size_t len)
{
  return send_frame_(0x2, data, len);
}

bool
WebSocket::send_text(string_view s)
{
  return send_frame_(0x1, reinterpret_cast<const uint8_t*>(s.data()),
                     s.size());
}

HttpServer::HttpServer(string bind_address, int port, string doc_root)
  : _bind_address(std::move(bind_address))
  , _port(port)
  , _doc_root(std::move(doc_root))
{
}

HttpServer::~HttpServer()
{
  stop();
}

void
HttpServer::route(string method, string pattern, Handler h)
{
  Route r;
  r.method   = std::move(method);
  r.segments = split_segments_(pattern);
  r.handler  = std::move(h);
  _routes.push_back(std::move(r));
}

void
HttpServer::route_stream(string method, string pattern, StreamHandler h)
{
  StreamRoute r;
  r.method   = std::move(method);
  r.segments = split_segments_(pattern);
  r.handler  = std::move(h);
  _stream_routes.push_back(std::move(r));
}

void
HttpServer::route_ws(string pattern, WsHandler h)
{
  WsRoute r;
  r.segments = split_segments_(pattern);
  r.handler  = std::move(h);
  _ws_routes.push_back(std::move(r));
}

bool
HttpServer::enable_tls(const std::string& cache_dir,
                       const std::string& bind_hint, std::string* err)
{
#ifdef VPIPE_WEBUI_TLS
  auto ctx = TlsContext::create(cache_dir, bind_hint, err);
  if (!ctx) { return false; }
  _tls = std::move(ctx);
  return true;
#else
  (void)cache_dir;
  (void)bind_hint;
  if (err) {
    *err = "web-ui was built without TLS support (OpenSSL not found at "
           "build time)";
  }
  return false;
#endif
}

bool
HttpServer::tls_enabled() const noexcept
{
#ifdef VPIPE_WEBUI_TLS
  return _tls != nullptr;
#else
  return false;
#endif
}

bool
HttpServer::start()
{
  if (_listen_fd >= 0) { return true; }
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    fprintf(stderr, "web-ui: socket() failed: %s\n", strerror(errno));
    return false;
  }
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port   = htons(static_cast<uint16_t>(_port));
  if (_bind_address.empty() || _bind_address == "0.0.0.0"
      || _bind_address == "*") {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else if (::inet_pton(AF_INET, _bind_address.c_str(),
                         &addr.sin_addr) != 1) {
    fprintf(stderr, "web-ui: invalid bind address '%s'\n",
            _bind_address.c_str());
    ::close(fd);
    return false;
  }
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0) {
    fprintf(stderr, "web-ui: bind(%s:%d) failed: %s\n",
            _bind_address.c_str(), _port, strerror(errno));
    ::close(fd);
    return false;
  }
  if (::listen(fd, 16) < 0) {
    fprintf(stderr, "web-ui: listen() failed: %s\n", strerror(errno));
    ::close(fd);
    return false;
  }
  sockaddr_in resolved{};
  socklen_t   rlen = sizeof resolved;
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&resolved),
                    &rlen) == 0) {
    _bound_port = ntohs(resolved.sin_port);
  } else {
    _bound_port = _port;
  }
  _listen_fd = fd;
  _stop.store(false);
  _thread = thread([this] { accept_loop_(); });
  return true;
}

void
HttpServer::stop()
{
  _stop.store(true);
  int fd = _listen_fd;
  if (fd >= 0) {
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    _listen_fd = -1;
  }
  if (_thread.joinable()) {
    _thread.join();
  }
  // Wait for in-flight connection handlers to finish (a pipeline stop
  // may still be draining on one) so the caller can safely tear down
  // the session afterwards. Bounded (~10 s) so a wedged handler can't
  // hang shutdown forever.
  for (int i = 0; i < 1000 && _inflight.load() > 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void
HttpServer::accept_loop_()
{
  while (!_stop.load()) {
    sockaddr_in peer{};
    socklen_t   plen = sizeof peer;
    int cfd = ::accept(_listen_fd, reinterpret_cast<sockaddr*>(&peer),
                       &plen);
    if (cfd < 0) {
      if (_stop.load()) { break; }
      if (errno == EINTR) { continue; }
      continue;
    }
    // 127.0.0.0/8 is loopback.
    bool loopback =
        (ntohl(peer.sin_addr.s_addr) & 0xFF000000u) == 0x7F000000u;
    // Never let a write to a peer that has gone away raise SIGPIPE and
    // kill the process -- matters most for streaming responses, where the
    // client may close (Cancel / navigate) mid-scan.
#ifdef SO_NOSIGPIPE
    {
      int on = 1;
      ::setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
    }
#endif
    // Serve on a detached thread so one slow request (e.g. a pipeline
    // stop that takes seconds to drain) doesn't block every other
    // request behind it. Handlers are already thread-safe: most take
    // SessionApi's mutex, and the status poller + UI delegate carry
    // their own; the DB browser handlers are serialised by SessionApi.
    _inflight.fetch_add(1, std::memory_order_relaxed);
    std::thread([this, cfd, loopback]() {
      std::unique_ptr<Conn> conn;
      bool drop = false;
#ifdef VPIPE_WEBUI_TLS
      if (_tls) {
        conn = _tls->wrap(cfd);   // blocking TLS handshake over the socket
        // A failed handshake (plaintext client, bad ClientHello) must NOT
        // fall back to plaintext -- drop the connection.
        if (!conn) { drop = true; }
      }
#endif
      if (!conn && !drop) {
        conn = std::make_unique<PlainConn>(cfd);
      }
      if (conn) { serve_one_(*conn, loopback); }
      conn.reset();               // TLS: SSL_shutdown before the fd closes
      ::close(cfd);
      _inflight.fetch_sub(1, std::memory_order_acq_rel);
    }).detach();
  }
}

void
HttpServer::serve_one_(Conn& conn, bool loopback)
{
  // Read until end-of-headers, then any declared body.
  string buf;
  char   chunk[8192];
  size_t header_end = string::npos;
  while (true) {
    ssize_t n = conn.read(chunk, sizeof chunk);
    if (n <= 0) {
      return;     // client closed / error before a full request
    }
    buf.append(chunk, static_cast<size_t>(n));
    header_end = buf.find("\r\n\r\n");
    if (header_end != string::npos) { break; }
    if (buf.size() > (1u << 20)) { return; }   // 1MB header cap
  }

  HttpRequest req;
  // Request line.
  size_t line_end = buf.find("\r\n");
  string request_line = buf.substr(0, line_end);
  {
    istringstream ls(request_line);
    string target;
    ls >> req.method >> target;
    size_t q = target.find('?');
    if (q == string::npos) {
      req.path = url_decode_(target);
    } else {
      req.path  = url_decode_(target.substr(0, q));
      req.query = target.substr(q + 1);
    }
  }
  // Headers.
  size_t pos = line_end + 2;
  while (pos < header_end) {
    size_t eol = buf.find("\r\n", pos);
    if (eol == string::npos || eol > header_end) { break; }
    string line = buf.substr(pos, eol - pos);
    pos = eol + 2;
    size_t colon = line.find(':');
    if (colon == string::npos) { continue; }
    string name = lower_(line.substr(0, colon));
    string val  = line.substr(colon + 1);
    size_t b = val.find_first_not_of(" \t");
    if (b != string::npos) { val = val.substr(b); } else { val.clear(); }
    req.headers[name] = val;
  }
  // Body.
  size_t content_length = 0;
  if (auto it = req.headers.find("content-length"); it != req.headers.end()) {
    content_length = strtoul(it->second.c_str(), nullptr, 10);
  }
  size_t body_start = header_end + 4;
  if (content_length > 0) {
    while (buf.size() - body_start < content_length) {
      ssize_t n = conn.read(chunk, sizeof chunk);
      if (n <= 0) { break; }
      buf.append(chunk, static_cast<size_t>(n));
      if (buf.size() > body_start + (16u << 20)) { break; }  // 16MB cap
    }
    req.body = buf.substr(body_start,
                          std::min(content_length, buf.size() - body_start));
  }

  // First-cut auth: gate /api/* for non-loopback peers on a shared key.
  // Static assets stay open so a remote browser can load the page and
  // prompt for the key.
  bool needs_auth = !_auth_key.empty() && !loopback
                    && req.method != "OPTIONS"
                    && req.path.rfind("/api/", 0) == 0;
  if (needs_auth) {
    bool ok = false;
    auto kit = req.headers.find("x-auth-key");
    if (kit != req.headers.end() && kit->second == _auth_key) { ok = true; }
    if (!ok) {
      // A WebSocket handshake can't carry custom headers, so accept the
      // key as a ?key= query parameter for upgrade requests.
      auto up = req.headers.find("upgrade");
      if (up != req.headers.end()
          && lower_(up->second).find("websocket") != string::npos
          && query_param_(req.query, "key") == _auth_key) {
        ok = true;
      }
    }
    needs_auth = !ok;
  }

  // Build the response.
  HttpResponse resp;
  if (needs_auth) {
    resp = HttpResponse::error(401, "access key required");
  } else if (req.method == "OPTIONS") {
    resp.status = 204;
    resp.content_type.clear();
  } else if (dispatch_ws_(req, conn)) {
    // A WebSocket route owns the connection after the handshake.
    return;
  } else if (dispatch_stream_(req, conn)) {
    // A streaming route owns the connection: it has already written the
    // full response (headers + incremental body). Nothing more to send.
    return;
  } else if (!dispatch_(req, resp)) {
    if (req.method == "GET" || req.method == "HEAD") {
      resp = serve_static_(req);
    } else {
      resp = HttpResponse::error(404, "no route for " + req.path);
    }
  }

  // Serialize with CORS + Connection: close.
  ostringstream os;
  os << "HTTP/1.1 " << resp.status << ' '
     << reason_phrase_(resp.status) << "\r\n";
  if (!resp.content_type.empty()) {
    os << "Content-Type: " << resp.content_type << "\r\n";
  }
  os << "Content-Length: " << resp.body.size() << "\r\n";
  os << "Access-Control-Allow-Origin: *\r\n";
  os << "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n";
  os << "Access-Control-Allow-Headers: Content-Type, X-Auth-Key\r\n";
  os << "Access-Control-Max-Age: 86400\r\n";
  for (auto& [k, v] : resp.extra_headers) {
    os << k << ": " << v << "\r\n";
  }
  os << "Connection: close\r\n\r\n";
  string head = os.str();
  conn.write_all(head.data(), head.size());
  if (req.method != "HEAD" && !resp.body.empty()) {
    conn.write_all(resp.body.data(), resp.body.size());
  }
}

bool
HttpServer::dispatch_(const HttpRequest& req, HttpResponse& out) const
{
  vector<string> segs = split_segments_(req.path);
  for (const Route& r : _routes) {
    if (r.method != req.method) { continue; }
    if (r.segments.size() != segs.size()) { continue; }
    bool match = true;
    unordered_map<string, string> params;
    for (size_t i = 0; i < segs.size(); ++i) {
      const string& p = r.segments[i];
      if (!p.empty() && p[0] == ':') {
        params[p.substr(1)] = segs[i];
      } else if (p != segs[i]) {
        match = false;
        break;
      }
    }
    if (!match) { continue; }
    HttpRequest local = req;
    local.params = std::move(params);
    out = r.handler(local);
    return true;
  }
  return false;
}

bool
HttpServer::dispatch_ws_(const HttpRequest& req, Conn& conn) const
{
  if (req.method != "GET") { return false; }
  auto up  = req.headers.find("upgrade");
  auto key = req.headers.find("sec-websocket-key");
  if (up == req.headers.end() || key == req.headers.end()) { return false; }
  if (lower_(up->second).find("websocket") == string::npos) { return false; }

  vector<string> segs = split_segments_(req.path);
  for (const WsRoute& r : _ws_routes) {
    if (r.segments.size() != segs.size()) { continue; }
    bool match = true;
    unordered_map<string, string> params;
    for (size_t i = 0; i < segs.size(); ++i) {
      const string& p = r.segments[i];
      if (!p.empty() && p[0] == ':') {
        params[p.substr(1)] = segs[i];
      } else if (p != segs[i]) {
        match = false;
        break;
      }
    }
    if (!match) { continue; }
    HttpRequest local = req;
    local.params = std::move(params);
    WebSocket ws(conn);
    if (ws.handshake(key->second)) {
      r.handler(local, ws);
    }
    return true;   // matched a ws route: this connection is a WebSocket
  }
  return false;
}

bool
HttpServer::dispatch_stream_(const HttpRequest& req, Conn& conn) const
{
  vector<string> segs = split_segments_(req.path);
  for (const StreamRoute& r : _stream_routes) {
    if (r.method != req.method) { continue; }
    if (r.segments.size() != segs.size()) { continue; }
    bool match = true;
    unordered_map<string, string> params;
    for (size_t i = 0; i < segs.size(); ++i) {
      const string& p = r.segments[i];
      if (!p.empty() && p[0] == ':') {
        params[p.substr(1)] = segs[i];
      } else if (p != segs[i]) {
        match = false;
        break;
      }
    }
    if (!match) { continue; }
    HttpRequest local = req;
    local.params = std::move(params);
    ResponseStream rs(conn);
    r.handler(local, rs);
    return true;
  }
  return false;
}

HttpResponse
HttpServer::serve_static_(const HttpRequest& req) const
{
  // Reject traversal; map "/" to index.html.
  if (req.path.find("..") != string::npos) {
    return HttpResponse::error(400, "bad path");
  }
  string rel = req.path;
  if (rel.empty() || rel == "/") { rel = "/index.html"; }

  // Filesystem doc-root (when configured). Tried first so a dev running
  // from the build tree can live-edit the web/ files.
  auto read_file = [&](const string& full, HttpResponse& r) -> bool {
    struct stat st{};
    if (::stat(full.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
      return false;
    }
    ifstream f(full, ios::binary);
    if (!f) { return false; }
    ostringstream ss;
    ss << f.rdbuf();
    r.status       = 200;
    r.body         = ss.str();
    r.content_type = mime_for_(full);
    return true;
  };

  // Assets embedded into the binary at build time. The fallback for a
  // packaged build with no doc-root on disk; the content type comes from
  // the request path (the bytes are not NUL-terminated).
  auto read_embedded = [&](const string& path, HttpResponse& r) -> bool {
    const EmbeddedAsset* a = find_embedded_asset(path);
    if (a == nullptr) { return false; }
    r.status       = 200;
    r.body.assign(reinterpret_cast<const char*>(a->data), a->size);
    r.content_type = mime_for_(path);
    return true;
  };

  HttpResponse r;
  if (!_doc_root.empty() && read_file(_doc_root + rel, r)) { return r; }
  if (read_embedded(rel, r)) { return r; }
  // SPA fallback: any unknown non-asset path returns index.html so the
  // client router can render it (only for paths that aren't /api/*).
  if (rel.rfind("/api/", 0) != 0) {
    if (!_doc_root.empty() && read_file(_doc_root + "/index.html", r)) {
      return r;
    }
    if (read_embedded("/index.html", r)) { return r; }
  }
  return HttpResponse::error(404, "not found: " + req.path);
}

}
