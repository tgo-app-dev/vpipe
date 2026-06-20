#include "common/static-file-server.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

// Strip any '?...' query and any '#...' fragment from a URL path.
string
clean_path_(string_view raw)
{
  size_t cut = raw.size();
  if (auto p = raw.find('?'); p != string_view::npos) {
    cut = p;
  }
  if (auto p = raw.find('#'); p != string_view::npos && p < cut) {
    cut = p;
  }
  return string(raw.substr(0, cut));
}

// Case-insensitive prefix match. Used for header-name lookup in the
// request bytes (HTTP header names are case-insensitive per RFC 7230).
bool
ci_starts_with_(string_view s, string_view pfx)
{
  if (s.size() < pfx.size()) { return false; }
  for (size_t i = 0; i < pfx.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(s[i]))
        != std::tolower(static_cast<unsigned char>(pfx[i]))) {
      return false;
    }
  }
  return true;
}

// Strip leading SP/HT (HTTP field-value optional whitespace).
string_view
ltrim_ows_(string_view v)
{
  while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) {
    v.remove_prefix(1);
  }
  return v;
}

// Locate a header in the request buffer. Returns the raw value
// (without trailing CRLF, with any leading OWS stripped), or empty
// string_view if the header is absent.
string_view
find_header_(string_view request, string_view name)
{
  size_t pos = 0;
  // Skip the request line (first CRLF).
  size_t first_eol = request.find("\r\n");
  if (first_eol == string_view::npos) { return {}; }
  pos = first_eol + 2;
  while (pos < request.size()) {
    size_t eol = request.find("\r\n", pos);
    if (eol == string_view::npos) { break; }
    string_view line = request.substr(pos, eol - pos);
    pos = eol + 2;
    if (line.empty()) { break; }  // end of headers
    // Header format: "Name: value"
    size_t colon = line.find(':');
    if (colon == string_view::npos) { continue; }
    if (colon != name.size()) { continue; }
    if (!ci_starts_with_(line.substr(0, colon), name)) { continue; }
    return ltrim_ows_(line.substr(colon + 1));
  }
  return {};
}

// Parse a decimal unsigned integer. Returns false on empty input or
// any non-digit.
bool
parse_uint_(string_view s, size_t* out)
{
  if (s.empty()) { return false; }
  size_t v = 0;
  for (char c : s) {
    if (c < '0' || c > '9') { return false; }
    v = v * 10 + static_cast<size_t>(c - '0');
  }
  *out = v;
  return true;
}

}

StaticFileServer::StaticFileServer(const SessionContextIntf* s,
                                   string                    doc_root,
                                   string                    bind_address,
                                   int                       port,
                                   const InMemoryBlobStore*  blob_store)
  : SessionMember(s)
  , _doc_root(std::move(doc_root))
  , _bind_address(std::move(bind_address))
  , _port(port)
  , _blob_store(blob_store)
{
}

void
InMemoryBlobStore::put(string          path,
                       vector<uint8_t> bytes,
                       string          content_type)
{
  Entry e;
  e.bytes        = std::make_shared<const vector<uint8_t>>(
                       std::move(bytes));
  e.content_type = std::move(content_type);
  std::lock_guard<std::mutex> lk(_mutex);
  _entries[std::move(path)] = std::move(e);
}

void
InMemoryBlobStore::erase(const string& path)
{
  std::lock_guard<std::mutex> lk(_mutex);
  _entries.erase(path);
}

std::optional<InMemoryBlobStore::Entry>
InMemoryBlobStore::get(const string& path) const
{
  std::lock_guard<std::mutex> lk(_mutex);
  auto it = _entries.find(path);
  if (it == _entries.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool
InMemoryBlobStore::contains(const string& path) const
{
  std::lock_guard<std::mutex> lk(_mutex);
  return _entries.count(path) != 0;
}

vector<string>
InMemoryBlobStore::paths() const
{
  std::lock_guard<std::mutex> lk(_mutex);
  vector<string> out;
  out.reserve(_entries.size());
  for (const auto& kv : _entries) {
    out.push_back(kv.first);
  }
  return out;
}

std::size_t
InMemoryBlobStore::size() const
{
  std::lock_guard<std::mutex> lk(_mutex);
  return _entries.size();
}

StaticFileServer::~StaticFileServer()
{
  stop();
}

bool
StaticFileServer::running() const noexcept
{
  return _listen_fd >= 0;
}

string
StaticFileServer::mime_type_for(string_view path)
{
  auto ends_with = [](string_view s, string_view sfx) {
    return s.size() >= sfx.size()
        && s.compare(s.size() - sfx.size(), sfx.size(), sfx) == 0;
  };
  if (ends_with(path, ".m3u8")) {
    return "application/vnd.apple.mpegurl";
  }
  if (ends_with(path, ".ts")) {
    return "video/mp2t";
  }
  if (ends_with(path, ".m4s") || ends_with(path, ".mp4")) {
    return "video/mp4";
  }
  if (ends_with(path, ".html") || ends_with(path, ".htm")) {
    return "text/html; charset=utf-8";
  }
  if (ends_with(path, ".css")) {
    return "text/css; charset=utf-8";
  }
  if (ends_with(path, ".js")) {
    return "application/javascript; charset=utf-8";
  }
  if (ends_with(path, ".json")) {
    return "application/json";
  }
  if (ends_with(path, ".txt")) {
    return "text/plain; charset=utf-8";
  }
  return "application/octet-stream";
}

bool
StaticFileServer::start()
{
  if (_listen_fd >= 0) {
    return true;
  }
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    session()->warn(fmt(
        "StaticFileServer: socket() failed: {}", strerror(errno)));
    return false;
  }

  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port   = htons(static_cast<uint16_t>(_port));
  if (_bind_address.empty()
      || _bind_address == "0.0.0.0"
      || _bind_address == "*") {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else if (::inet_pton(AF_INET, _bind_address.c_str(),
                         &addr.sin_addr) != 1) {
    session()->warn(fmt(
        "StaticFileServer: invalid bind_address '{}'", _bind_address));
    ::close(fd);
    return false;
  }

  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr),
             sizeof addr) < 0) {
    session()->warn(fmt(
        "StaticFileServer: bind({}:{}) failed: {}",
        _bind_address, _port, strerror(errno)));
    ::close(fd);
    return false;
  }
  if (::listen(fd, 16) < 0) {
    session()->warn(fmt(
        "StaticFileServer: listen() failed: {}", strerror(errno)));
    ::close(fd);
    return false;
  }

  // Resolve the effective port (in case _port was 0 for "any").
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
  _thread = thread([this]{ accept_loop_(); });
  return true;
}

void
StaticFileServer::stop()
{
  _stop.store(true);
  int fd = _listen_fd;
  if (fd >= 0) {
    // shutdown() + close() reliably unblocks accept() on macOS / Linux.
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    _listen_fd = -1;
  }
  if (_thread.joinable()) {
    _thread.join();
  }
}

void
StaticFileServer::accept_loop_()
{
  while (!_stop.load(memory_order_acquire)) {
    int fd = _listen_fd;
    if (fd < 0) { break; }

    sockaddr_in peer{};
    socklen_t   peer_len = sizeof peer;
    int client = ::accept(fd, reinterpret_cast<sockaddr*>(&peer),
                          &peer_len);
    if (client < 0) {
      if (_stop.load(memory_order_acquire)) { break; }
      if (errno == EINTR) { continue; }
      // EBADF / EINVAL after shutdown -- exit loop.
      break;
    }
    serve_one_(client);
    ::close(client);
  }
}

bool
StaticFileServer::send_all_(int fd, const char* data, size_t len) const
{
  while (len > 0) {
    ssize_t n = ::send(fd, data, len, 0);
    if (n < 0) {
      if (errno == EINTR) { continue; }
      return false;
    }
    data += n;
    len  -= static_cast<size_t>(n);
  }
  return true;
}

void
StaticFileServer::send_status_(int fd, int code,
                               string_view reason) const
{
  string body = string(reason) + "\n";
  string hdr  = fmt(
      "HTTP/1.1 {} {}\r\n"
      "Content-Type: text/plain; charset=utf-8\r\n"
      "Content-Length: {}\r\n"
      "Connection: close\r\n\r\n",
      code, reason, body.size())();
  send_all_(fd, hdr.data(), hdr.size());
  send_all_(fd, body.data(), body.size());
}

void
StaticFileServer::serve_one_(int client_fd)
{
  // Read the request: up to ~8 KiB, expect headers terminated by
  // CRLFCRLF. We parse the request line + a small set of headers
  // (Range, Access-Control-Request-Headers); unknown headers are
  // accepted-and-ignored.
  char  buf[8192];
  size_t total = 0;
  while (total + 1 < sizeof buf) {
    ssize_t n = ::recv(client_fd, buf + total,
                       sizeof(buf) - 1 - total, 0);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) { continue; }
      return;
    }
    total += static_cast<size_t>(n);
    buf[total] = 0;
    if (strstr(buf, "\r\n\r\n")) { break; }
  }
  if (total == 0) {
    return;
  }
  const string_view request(buf, total);

  // Parse "METHOD PATH HTTP/1.X\r\n".
  char method[16]{};
  char path_raw[2048]{};
  if (sscanf(buf, "%15s %2047s", method, path_raw) != 2) {
    send_status_(client_fd, 400, "Bad Request");
    return;
  }

  const bool is_head = strcmp(method, "HEAD") == 0;
  const bool is_get  = strcmp(method, "GET")  == 0;
  const bool is_opts = strcmp(method, "OPTIONS") == 0;

  // OPTIONS = CORS preflight. Reply 204 + permissive Allow-* headers
  // so browser players hitting the stream from a different origin
  // can complete the preflight and issue their actual GET.
  if (is_opts) {
    send_cors_preflight_(client_fd, request);
    return;
  }
  if (!is_head && !is_get) {
    send_status_(client_fd, 405, "Method Not Allowed");
    return;
  }

  string path = clean_path_(path_raw);
  if (path.empty() || path[0] != '/'
      || path.find("..") != string::npos) {
    send_status_(client_fd, 400, "Bad Request");
    return;
  }

  // In-memory blob registry takes precedence over the filesystem
  // fallback. The blob store snapshot is captured under the store's
  // own lock and released before any I/O, so a slow client never
  // blocks the HLS producer.
  if (_blob_store && serve_blob_(client_fd, path, is_head, request)) {
    return;
  }

  if (_doc_root.empty()) {
    send_status_(client_fd, 404, "Not Found");
    return;
  }
  serve_file_(client_fd, path, is_head, request);
}

bool
StaticFileServer::serve_blob_(int client_fd, const string& path,
                              bool is_head, string_view request)
{
  auto entry = _blob_store->get(path);
  if (!entry) {
    return false;
  }
  const auto& bytes  = entry->bytes;
  const string& mime = entry->content_type;
  const size_t total = bytes ? bytes->size() : 0;

  Range rng = resolve_range_(request, total);
  if (rng.unsatisfiable) {
    send_range_not_satisfiable_(client_fd, total);
    return true;
  }

  const size_t off = rng.first;
  const size_t len = total == 0 ? 0 : (rng.last - rng.first + 1);

  string hdr;
  if (rng.partial) {
    hdr = fmt(
        "HTTP/1.1 206 Partial Content\r\n"
        "Content-Type: {}\r\n"
        "Content-Length: {}\r\n"
        "Content-Range: bytes {}-{}/{}\r\n"
        "Accept-Ranges: bytes\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Expose-Headers: Content-Length, Content-Range, "
        "Accept-Ranges\r\n"
        "Connection: close\r\n\r\n",
        mime.empty() ? mime_type_for(path) : mime,
        static_cast<long long>(len),
        static_cast<long long>(rng.first),
        static_cast<long long>(rng.last),
        static_cast<long long>(total))();
  } else {
    hdr = fmt(
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: {}\r\n"
        "Content-Length: {}\r\n"
        "Accept-Ranges: bytes\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Expose-Headers: Content-Length, Content-Range, "
        "Accept-Ranges\r\n"
        "Connection: close\r\n\r\n",
        mime.empty() ? mime_type_for(path) : mime,
        static_cast<long long>(len))();
  }
  if (!send_all_(client_fd, hdr.data(), hdr.size())) {
    return true;
  }
  if (!is_head && len > 0) {
    send_all_(client_fd,
              reinterpret_cast<const char*>(bytes->data() + off),
              len);
  }
  return true;
}

void
StaticFileServer::serve_file_(int client_fd, const string& path,
                              bool is_head, string_view request)
{
  string full = _doc_root + path;

  // Stat the file (also tells us if it's a regular file vs dir).
  struct stat st{};
  if (::stat(full.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
    send_status_(client_fd, 404, "Not Found");
    return;
  }
  const size_t total = static_cast<size_t>(st.st_size);

  Range rng = resolve_range_(request, total);
  if (rng.unsatisfiable) {
    send_range_not_satisfiable_(client_fd, total);
    return;
  }

  int file_fd = ::open(full.c_str(), O_RDONLY);
  if (file_fd < 0) {
    send_status_(client_fd, 404, "Not Found");
    return;
  }

  const size_t off       = rng.first;
  const size_t want_len  = total == 0 ? 0 : (rng.last - rng.first + 1);
  const auto   mime      = mime_type_for(path);
  string hdr;
  if (rng.partial) {
    hdr = fmt(
        "HTTP/1.1 206 Partial Content\r\n"
        "Content-Type: {}\r\n"
        "Content-Length: {}\r\n"
        "Content-Range: bytes {}-{}/{}\r\n"
        "Accept-Ranges: bytes\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Expose-Headers: Content-Length, Content-Range, "
        "Accept-Ranges\r\n"
        "Connection: close\r\n\r\n",
        mime, static_cast<long long>(want_len),
        static_cast<long long>(rng.first),
        static_cast<long long>(rng.last),
        static_cast<long long>(total))();
  } else {
    hdr = fmt(
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: {}\r\n"
        "Content-Length: {}\r\n"
        "Accept-Ranges: bytes\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Expose-Headers: Content-Length, Content-Range, "
        "Accept-Ranges\r\n"
        "Connection: close\r\n\r\n",
        mime, static_cast<long long>(want_len))();
  }
  if (!send_all_(client_fd, hdr.data(), hdr.size())) {
    ::close(file_fd);
    return;
  }

  if (!is_head && want_len > 0) {
    if (off > 0
        && ::lseek(file_fd, static_cast<off_t>(off), SEEK_SET) < 0) {
      ::close(file_fd);
      return;
    }
    size_t remaining = want_len;
    char chunk[64 * 1024];
    while (remaining > 0) {
      const size_t to_read =
          remaining < sizeof chunk ? remaining : sizeof chunk;
      ssize_t r = ::read(file_fd, chunk, to_read);
      if (r < 0) {
        if (errno == EINTR) { continue; }
        break;
      }
      if (r == 0) { break; }
      if (!send_all_(client_fd, chunk, static_cast<size_t>(r))) {
        break;
      }
      remaining -= static_cast<size_t>(r);
    }
  }
  ::close(file_fd);
}

void
StaticFileServer::send_cors_preflight_(int fd,
                                       string_view request) const
{
  // Echo back whatever the client asked to send in
  // Access-Control-Request-Headers (so hls.js / XHR with custom
  // headers like Range or If-* sail through). Default to "*" if
  // the client didn't ask -- some browsers still send a preflight
  // without the request-headers field.
  string_view req_hdrs =
      find_header_(request, "access-control-request-headers");
  string allow_hdrs;
  if (req_hdrs.empty()) {
    allow_hdrs = "*";
  } else {
    allow_hdrs.assign(req_hdrs);
  }
  string hdr = fmt(
      "HTTP/1.1 204 No Content\r\n"
      "Access-Control-Allow-Origin: *\r\n"
      "Access-Control-Allow-Methods: GET, HEAD, OPTIONS\r\n"
      "Access-Control-Allow-Headers: {}\r\n"
      "Access-Control-Max-Age: 86400\r\n"
      "Content-Length: 0\r\n"
      "Connection: close\r\n\r\n",
      allow_hdrs)();
  send_all_(fd, hdr.data(), hdr.size());
}

void
StaticFileServer::send_range_not_satisfiable_(int fd,
                                              size_t body_len) const
{
  string body = "Requested range not satisfiable\n";
  string hdr  = fmt(
      "HTTP/1.1 416 Range Not Satisfiable\r\n"
      "Content-Type: text/plain; charset=utf-8\r\n"
      "Content-Length: {}\r\n"
      "Content-Range: bytes */{}\r\n"
      "Accept-Ranges: bytes\r\n"
      "Access-Control-Allow-Origin: *\r\n"
      "Connection: close\r\n\r\n",
      body.size(), static_cast<long long>(body_len))();
  send_all_(fd, hdr.data(), hdr.size());
  send_all_(fd, body.data(), body.size());
}

StaticFileServer::Range
StaticFileServer::resolve_range_(string_view request,
                                 size_t body_len) const
{
  Range r;
  // Default: serve full body, 200 OK. Even when body_len == 0 we
  // hand back first=0/last=0 with partial=false; the caller treats
  // (last - first + 1) as length, so callers must special-case the
  // empty-body branch (they do).
  r.first = 0;
  r.last  = body_len == 0 ? 0 : body_len - 1;

  string_view raw = find_header_(request, "range");
  if (raw.empty()) {
    return r;
  }
  // Expect "bytes=...".
  constexpr string_view kPfx = "bytes=";
  if (!ci_starts_with_(raw, kPfx)) {
    return r;  // unknown unit -- fall back to 200 OK full body
  }
  raw.remove_prefix(kPfx.size());
  // Multi-range ("bytes=0-99,200-299") is not implemented; degrade
  // to 200 OK with the full body rather than risk a half-baked 206.
  if (raw.find(',') != string_view::npos) {
    return r;
  }
  const size_t dash = raw.find('-');
  if (dash == string_view::npos) {
    return r;  // malformed
  }
  const string_view a = raw.substr(0, dash);
  const string_view b = raw.substr(dash + 1);

  if (a.empty()) {
    // Suffix form: "-N" => last N bytes.
    size_t n;
    if (!parse_uint_(b, &n) || n == 0) {
      // n==0 is a satisfiable suffix range of size zero per RFC,
      // but practically useless and most servers treat as 416.
      if (b.empty()) { return r; }
      r.unsatisfiable = true;
      return r;
    }
    if (body_len == 0) {
      r.unsatisfiable = true;
      return r;
    }
    if (n >= body_len) {
      r.first = 0;
      r.last  = body_len - 1;
    } else {
      r.first = body_len - n;
      r.last  = body_len - 1;
    }
    r.partial = (r.first != 0 || r.last != body_len - 1);
    return r;
  }

  size_t start;
  if (!parse_uint_(a, &start)) { return r; }
  if (body_len == 0 || start >= body_len) {
    r.unsatisfiable = true;
    return r;
  }
  size_t end;
  if (b.empty()) {
    end = body_len - 1;
  } else {
    if (!parse_uint_(b, &end)) { return r; }
    if (end < start) { return r; }   // malformed -- ignore Range
    if (end >= body_len) { end = body_len - 1; }
  }
  r.first   = start;
  r.last    = end;
  r.partial = (start != 0 || end != body_len - 1);
  return r;
}

}
