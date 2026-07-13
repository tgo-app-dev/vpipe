#ifndef WEBUI_TLS_CONTEXT_H
#define WEBUI_TLS_CONTEXT_H

#include "apps/web-ui/http-server.h"   // for Conn

#include <memory>
#include <string>

namespace vpipe::webui {

// Server-side TLS via OpenSSL. Loads -- or, on first run, generates + caches
// -- a self-signed certificate/key under `cache_dir`, holds an SSL_CTX, and
// wraps accepted sockets in a TLS Conn. Only compiled when the build found
// OpenSSL (VPIPE_WEBUI_TLS); HttpServer::enable_tls degrades to a clear
// error otherwise, so nothing here needs to exist in a no-OpenSSL build.
//
// pImpl keeps the OpenSSL headers out of this header (http-server.cc only
// needs the Conn contract), so including it doesn't pull libssl into every
// web-ui translation unit.
class TlsContext {
public:
  // Build a context serving a self-signed cert whose SubjectAltName covers
  // localhost + 127.0.0.1 (+ `bind_hint`, when it is an address, so the URL
  // the browser uses matches the cert). The cert is cached per bind_hint so
  // browser trust persists across restarts; delete the cache dir to
  // regenerate. Returns nullptr and sets *err on failure.
  static std::unique_ptr<TlsContext>
  create(const std::string& cache_dir, const std::string& bind_hint,
         std::string* err);

  ~TlsContext();
  TlsContext(const TlsContext&)            = delete;
  TlsContext& operator=(const TlsContext&) = delete;

  // Perform the TLS handshake over an accepted socket fd (blocking). Returns
  // a Conn that does all subsequent I/O over TLS, or nullptr if the
  // handshake failed (the caller closes the fd either way). The Conn does
  // NOT own the fd.
  std::unique_ptr<Conn> wrap(int fd);

  const std::string& cert_path() const noexcept { return _cert_path; }

private:
  struct Impl;
  TlsContext(std::unique_ptr<Impl> impl, std::string cert_path);

  std::unique_ptr<Impl> _impl;
  std::string           _cert_path;
};

}

#endif
