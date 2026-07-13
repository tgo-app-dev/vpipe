#include "apps/web-ui/tls-context.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <sys/stat.h>

#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>

namespace vpipe::webui {

namespace {

std::string
ossl_err_()
{
  unsigned long e = ERR_get_error();
  if (e == 0) { return "unknown OpenSSL error"; }
  char buf[256];
  ERR_error_string_n(e, buf, sizeof buf);
  return buf;
}

void
set_err_(std::string* err, std::string msg)
{
  if (err) { *err = std::move(msg); }
}

// Filename-safe tag for a bind address (an IP or "any").
std::string
tag_for_(const std::string& bind_hint)
{
  std::string s = bind_hint.empty() ? std::string("any") : bind_hint;
  for (char& c : s) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
                 || (c >= '0' && c <= '9') || c == '.';
    if (!ok) { c = '_'; }
  }
  return s;
}

// SubjectAltName list covering localhost + the bind address (as IP when it
// parses as one, else DNS), so the browser reaches a name-matching origin.
std::string
san_for_(const std::string& bind_hint)
{
  std::string san = "DNS:localhost,IP:127.0.0.1";
  // Already covered by the base list above -> don't duplicate.
  if (!bind_hint.empty() && bind_hint != "0.0.0.0" && bind_hint != "*"
      && bind_hint != "127.0.0.1" && bind_hint != "localhost") {
    struct in_addr  a4;
    struct in6_addr a6;
    if (::inet_pton(AF_INET, bind_hint.c_str(), &a4) == 1
        || ::inet_pton(AF_INET6, bind_hint.c_str(), &a6) == 1) {
      san += ",IP:" + bind_hint;
    } else {
      san += ",DNS:" + bind_hint;
    }
  }
  return san;
}

bool
write_key_pem_(const std::string& path, EVP_PKEY* pkey, std::string* err)
{
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) { set_err_(err, "cannot open " + path + " for writing"); return false; }
  int r = PEM_write_PrivateKey(f, pkey, nullptr, nullptr, 0, nullptr, nullptr);
  std::fclose(f);
  ::chmod(path.c_str(), 0600);   // private key: owner-only
  if (r != 1) { set_err_(err, "PEM_write_PrivateKey: " + ossl_err_()); return false; }
  return true;
}

bool
write_cert_pem_(const std::string& path, X509* x, std::string* err)
{
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) { set_err_(err, "cannot open " + path + " for writing"); return false; }
  int r = PEM_write_X509(f, x);
  std::fclose(f);
  if (r != 1) { set_err_(err, "PEM_write_X509: " + ossl_err_()); return false; }
  return true;
}

// Generate a 10-year self-signed RSA-2048 certificate with `san` and write
// the cert + key as PEM.
bool
generate_self_signed_(const std::string& cert_path,
                      const std::string& key_path,
                      const std::string& san, std::string* err)
{
  EVP_PKEY* pkey = EVP_RSA_gen(2048);
  if (!pkey) { set_err_(err, "EVP_RSA_gen: " + ossl_err_()); return false; }

  X509* x = X509_new();
  if (!x) {
    EVP_PKEY_free(pkey);
    set_err_(err, "X509_new failed");
    return false;
  }

  bool ok = false;
  do {
    X509_set_version(x, 2);   // X.509 v3
    ASN1_INTEGER_set(X509_get_serialNumber(x), 1);
    X509_gmtime_adj(X509_getm_notBefore(x), 0);
    X509_gmtime_adj(X509_getm_notAfter(x),
                    static_cast<long>(3650) * 24 * 3600);
    X509_set_pubkey(x, pkey);

    X509_NAME* name = X509_get_subject_name(x);
    X509_NAME_add_entry_by_txt(
        name, "CN", MBSTRING_ASC,
        reinterpret_cast<const unsigned char*>("vpipe-web-ui"), -1, -1, 0);
    X509_set_issuer_name(x, name);   // self-signed: issuer == subject

    X509_EXTENSION* ext = X509V3_EXT_conf_nid(
        nullptr, nullptr, NID_subject_alt_name, san.c_str());
    if (ext) {
      X509_add_ext(x, ext, -1);
      X509_EXTENSION_free(ext);
    }

    if (X509_sign(x, pkey, EVP_sha256()) == 0) {
      set_err_(err, "X509_sign: " + ossl_err_());
      break;
    }
    if (!write_key_pem_(key_path, pkey, err)) { break; }
    if (!write_cert_pem_(cert_path, x, err))  { break; }
    ok = true;
  } while (false);

  X509_free(x);
  EVP_PKEY_free(pkey);
  return ok;
}

// Ensure a cert/key pair exists for this bind address, generating them on
// first use. `cache_dir` is created if missing.
bool
ensure_cert_(const std::string& cache_dir, const std::string& bind_hint,
             std::string& cert_path, std::string& key_path, std::string* err)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(cache_dir, ec);   // ignore ec: may already exist

  const std::string tag = tag_for_(bind_hint);
  cert_path = cache_dir + "/cert-" + tag + ".pem";
  key_path  = cache_dir + "/key-"  + tag + ".pem";

  if (fs::exists(cert_path) && fs::exists(key_path)) {
    return true;   // reuse the cached identity (browser trust persists)
  }
  return generate_self_signed_(cert_path, key_path, san_for_(bind_hint), err);
}

// OpenSSL 3 auto-initializes on first use; do it once explicitly so error
// strings are loaded for diagnostics.
void
ossl_init_once_()
{
  static std::once_flag flag;
  std::call_once(flag, [] {
    OPENSSL_init_ssl(OPENSSL_INIT_LOAD_SSL_STRINGS
                     | OPENSSL_INIT_LOAD_CRYPTO_STRINGS,
                     nullptr);
  });
}

// TLS transport over one accepted socket. Owns the SSL session (not the
// fd); the dtor sends close_notify before the accept loop closes the fd.
class TlsConn : public Conn {
public:
  explicit TlsConn(SSL* ssl) : _ssl(ssl) {}
  ~TlsConn() override
  {
    if (_ssl) {
      SSL_shutdown(_ssl);   // one-way close_notify is enough before ::close
      SSL_free(_ssl);
    }
  }

  ssize_t
  read(void* buf, size_t n) override
  {
    const int want = static_cast<int>(std::min<size_t>(n, 1u << 20));
    for (;;) {
      int r = SSL_read(_ssl, buf, want);
      if (r > 0) { return r; }
      const int e = SSL_get_error(_ssl, r);
      if (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE) { continue; }
      if (e == SSL_ERROR_ZERO_RETURN) { return 0; }   // clean close_notify
      // SYSCALL with r==0 is an unexpected EOF (peer vanished): treat as
      // close. Anything else is a hard error.
      return (e == SSL_ERROR_SYSCALL && r == 0) ? 0 : -1;
    }
  }

  bool
  write_all(const char* data, size_t len) override
  {
    size_t sent = 0;
    while (sent < len) {
      const int want =
          static_cast<int>(std::min<size_t>(len - sent, 1u << 20));
      int r = SSL_write(_ssl, data + sent, want);
      if (r > 0) { sent += static_cast<size_t>(r); continue; }
      const int e = SSL_get_error(_ssl, r);
      if (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE) { continue; }
      return false;
    }
    return true;
  }

private:
  SSL* _ssl;
};

}  // namespace

struct TlsContext::Impl {
  SSL_CTX* ctx = nullptr;
  ~Impl() { if (ctx) { SSL_CTX_free(ctx); } }
};

TlsContext::TlsContext(std::unique_ptr<Impl> impl, std::string cert_path)
  : _impl(std::move(impl))
  , _cert_path(std::move(cert_path))
{
}

TlsContext::~TlsContext() = default;

std::unique_ptr<TlsContext>
TlsContext::create(const std::string& cache_dir, const std::string& bind_hint,
                   std::string* err)
{
  ossl_init_once_();

  std::string cert_path;
  std::string key_path;
  if (!ensure_cert_(cache_dir, bind_hint, cert_path, key_path, err)) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->ctx = SSL_CTX_new(TLS_server_method());
  if (!impl->ctx) {
    set_err_(err, "SSL_CTX_new: " + ossl_err_());
    return nullptr;
  }
  SSL_CTX_set_min_proto_version(impl->ctx, TLS1_2_VERSION);
  SSL_CTX_set_mode(impl->ctx, SSL_MODE_AUTO_RETRY);

  if (SSL_CTX_use_certificate_file(impl->ctx, cert_path.c_str(),
                                   SSL_FILETYPE_PEM) != 1) {
    set_err_(err, "load certificate " + cert_path + ": " + ossl_err_());
    return nullptr;
  }
  if (SSL_CTX_use_PrivateKey_file(impl->ctx, key_path.c_str(),
                                  SSL_FILETYPE_PEM) != 1) {
    set_err_(err, "load private key " + key_path + ": " + ossl_err_());
    return nullptr;
  }
  if (SSL_CTX_check_private_key(impl->ctx) != 1) {
    set_err_(err, "certificate/key mismatch: " + ossl_err_());
    return nullptr;
  }

  return std::unique_ptr<TlsContext>(
      new TlsContext(std::move(impl), std::move(cert_path)));
}

std::unique_ptr<Conn>
TlsContext::wrap(int fd)
{
  SSL* ssl = SSL_new(_impl->ctx);
  if (!ssl) { return nullptr; }
  SSL_set_fd(ssl, fd);
  for (;;) {
    int r = SSL_accept(ssl);
    if (r == 1) { break; }   // handshake complete
    const int e = SSL_get_error(ssl, r);
    if (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE) { continue; }
    SSL_free(ssl);           // handshake failed (plaintext client, etc.)
    return nullptr;
  }
  return std::make_unique<TlsConn>(ssl);
}

}
