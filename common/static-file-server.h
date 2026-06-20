#ifndef COMMON_STATIC_FILE_SERVER_H
#define COMMON_STATIC_FILE_SERVER_H

#include "common/session-member.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vpipe {

// Thread-safe in-memory blob registry. Maps URL paths (e.g.
// "/stream.m3u8") to a content-typed byte buffer. Producers (e.g.
// the HLS muxer) call put()/erase() from one thread; the
// StaticFileServer's accept thread serves snapshots concurrently.
// Buffers are shared by shared_ptr so a slow client doesn't block
// a producer that wants to replace the entry.
class InMemoryBlobStore {
public:
  struct Entry {
    std::shared_ptr<const std::vector<std::uint8_t>> bytes;
    std::string                                      content_type;
  };

  void put(std::string          path,
           std::vector<std::uint8_t> bytes,
           std::string          content_type);

  // No-op if the path is not registered.
  void erase(const std::string& path);

  // Returns a stable snapshot of the entry, or nullopt if absent.
  // The returned bytes pointer remains valid even if the entry is
  // replaced or erased afterwards.
  std::optional<Entry> get(const std::string& path) const;

  bool contains(const std::string& path) const;

  // Snapshot of currently registered paths. For tests + diagnostics.
  std::vector<std::string> paths() const;

  std::size_t size() const;

private:
  mutable std::mutex                       _mutex;
  std::unordered_map<std::string, Entry>   _entries;
};

// Minimal HTTP/1.1 static-file server. One accept loop, served
// synchronously per connection (sequential clients; concurrent
// requests are funneled, which is fine for HLS-style fan-out where
// the bulk of cost is in the kernel sendfile path anyway).
//
// Two backing modes, both optional and combinable:
//   * In-memory blob registry (preferred for live HLS). When a
//     blob_store is supplied, request paths are looked up there
//     first.
//   * Filesystem doc_root. When doc_root is non-empty, request paths
//     not satisfied by the blob registry are resolved against the
//     directory. Set doc_root to "" to disable filesystem mode
//     entirely.
//
// Supports: GET, HEAD, OPTIONS, single-range Range: bytes=X-Y on
// in-memory blobs. Every 2xx response advertises Accept-Ranges:
// bytes. OPTIONS replies with a CORS preflight (204 + Allow-*
// headers + 24h Max-Age), so browser players hitting the stream
// from another origin via fetch()/XHR clear the preflight cleanly.
// Returns: 200, 204, 206, 400, 404, 405, 416, plus the preflight
// 204 for OPTIONS.
// Does NOT support: keep-alive, gzip, directory listings, multi-
// range responses (Range with commas falls back to 200 full body).
// Path safety: any request path containing ".." or not starting
// with "/" is rejected with 400. Symlinks inside doc_root are
// followed.
//
// Lifecycle:
//   server.start()  -> binds + listens; spawns the accept thread.
//   server.stop()   -> sets stop flag, shutdown()s the listen fd
//                       to unblock accept(), joins the thread.
// Either call is idempotent. The destructor calls stop(). The
// blob_store pointer, if supplied, must outlive the server.
class StaticFileServer : public SessionMember {
public:
  StaticFileServer(const SessionContextIntf* session,
                   std::string               doc_root,
                   std::string               bind_address,
                   int                       port,
                   const InMemoryBlobStore*  blob_store = nullptr);
  ~StaticFileServer() override;

  StaticFileServer(const StaticFileServer&)            = delete;
  StaticFileServer& operator=(const StaticFileServer&) = delete;

  // Bind + listen + spawn accept thread. Returns false on bind /
  // listen error (port in use, permission denied, etc.). Errors
  // are also logged via session()->warn.
  bool start();

  // Idempotent. Closes the listen fd, joins the accept thread.
  void stop();

  bool running()      const noexcept;
  int  bound_port()   const noexcept { return _bound_port; }
  const std::string& doc_root() const noexcept { return _doc_root; }

  // Public for unit testing.
  static std::string mime_type_for(std::string_view path);

private:
  // Parsed Range request resolved against a known body length.
  // `partial` => respond with 206 + Content-Range; `unsatisfiable`
  // => respond with 416; otherwise serve [first..last] inclusive
  // (which equals the full body when no Range header was sent).
  struct Range {
    std::size_t first         = 0;
    std::size_t last          = 0;  // inclusive
    bool        partial       = false;
    bool        unsatisfiable = false;
  };

  void  accept_loop_();
  void  serve_one_  (int client_fd);
  bool  serve_blob_ (int client_fd, const std::string& path,
                     bool is_head, std::string_view request);
  void  serve_file_ (int client_fd, const std::string& path,
                     bool is_head, std::string_view request);
  bool  send_all_   (int fd, const char* data, size_t len) const;
  void  send_status_(int fd, int code, std::string_view reason) const;
  void  send_cors_preflight_(int fd, std::string_view request) const;
  void  send_range_not_satisfiable_(int fd, std::size_t body_len) const;
  Range resolve_range_(std::string_view request,
                       std::size_t      body_len) const;

  std::string              _doc_root;
  std::string              _bind_address;
  int                      _port;
  int                      _bound_port = 0;       // resolved after bind()
  int                      _listen_fd  = -1;
  std::atomic<bool>        _stop{ false };
  std::thread              _thread;
  const InMemoryBlobStore* _blob_store = nullptr;
};

}

#endif
