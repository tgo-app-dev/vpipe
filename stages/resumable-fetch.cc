#include "stages/resumable-fetch.h"

#include "stages/xet-fetch.h"

#include "common/vpipe-format.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

#include <CommonCrypto/CommonDigest.h>
#include <curl/curl.h>

using std::string;
using std::vector;
namespace fs = std::filesystem;

namespace vpipe {

namespace {

string
lower_(string s)
{
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

// ---- HTTP -------------------------------------------------------------

size_t
write_to_string_(char* p, size_t s, size_t n, void* u)
{
  static_cast<string*>(u)->append(p, s * n);
  return s * n;
}

// The same, but counting as it goes. A range on the Xet path arrives as
// one buffered GET, so nothing downstream can see it move until it is
// whole -- and the writer that would report it is meanwhile blocked on
// SOME OTHER range. This is the hook that lets the caller report a
// transfer in flight; libcurl calls it about every 16 KB.
struct CountingSink {
  string*                                   out      = nullptr;
  const std::function<void(std::uint64_t)>* on_bytes = nullptr;
  std::uint64_t                             got      = 0;
};

size_t
write_counting_(char* p, size_t s, size_t n, void* u)
{
  auto* c = static_cast<CountingSink*>(u);
  const size_t got = s * n;
  c->out->append(p, got);
  c->got += got;
  if (c->on_bytes) { (*c->on_bytes)(c->got); }
  return got;
}

// ---- download progress -------------------------------------------------

// Per-download progress state handed to the libcurl xferinfo callback.
struct ProgressCtx {
  vpipe::UiProgress* progress = nullptr;
  // Poll predicate: when set and it returns true, the xferinfo callback
  // aborts the transfer mid-flight (-> CURLE_ABORTED_BY_CALLBACK).
  const std::function<bool()>* cancel = nullptr;
  // Bytes already on disk when this attempt started. libcurl counts
  // only what THIS transfer moved, so a resumed shard would otherwise
  // draw a bar that restarts at zero two thirds of the way in.
  const curl_off_t* base = nullptr;
};

// libcurl CURLOPT_XFERINFOFUNCTION. Pushes the raw byte counts every
// time libcurl calls -- no percentage-change throttle, because the
// renderers coalesce on their own clocks (see UiProgressRegistry) and
// update() is a mutex plus a few field writes.
int
progress_cb_(void* p, curl_off_t dltotal, curl_off_t dlnow,
             curl_off_t /*ultotal*/, curl_off_t /*ulnow*/)
{
  auto* c = static_cast<ProgressCtx*>(p);
  if (c && c->cancel && (*c->cancel)()) { return 1; }   // abort transfer
  if (!c || !c->progress) {
    return 0;
  }
  const curl_off_t base = c->base ? *c->base : 0;
  // dltotal is 0 until the response headers land, which the report
  // shows as indeterminate rather than as 0%. On a resumed transfer it
  // is the REMAINING length, so the prefix counts on BOTH sides or the
  // bar reads "3.0 GB / 1.2 GB".
  const auto now   = static_cast<std::uint64_t>(
      (dlnow < 0 ? 0 : dlnow) + base);
  const auto total = dltotal <= 0
      ? std::uint64_t{0}
      : static_cast<std::uint64_t>(dltotal + base);
  // Before the headers land there is no total, so report just what has
  // arrived -- "1.2 GB / 0.0 B" would read as a broken denominator.
  c->progress->update(now, total,
                      total > 0 ? human_bytes(now) + " / "
                                    + human_bytes(total)
                                : human_bytes(now));
  return 0;
}

// ---- one libcurl perform -----------------------------------------------

// A transfer moving less than this counts as stalled rather than slow.
// Low enough that a genuinely bad link still finishes, high enough that
// a half-open socket dribbling keepalives cannot hold a fetch open
// forever.
constexpr long kStallBytesPerSec = 1024;

// How one perform is bounded, and where it picks up.
struct PerformOpts {
  bool verify_tls = true;
  // Hard deadline on the whole transfer. Right for the metadata calls,
  // WRONG for a file: a 20 GB shard at 2 MB/s takes three hours and is
  // healthy every second of it. 0 -> none.
  long timeout_s = 0;
  // Give up after this long below kStallBytesPerSec. This is what
  // bounds a file transfer instead: it fires on a connection that has
  // died, not on one that is merely slow. 0 -> none.
  long stall_s = 0;
  // "Range: bytes=<n>-". 0 -> ask for the whole object.
  curl_off_t resume_from = 0;
  // An explicit "<first>-<last>" range, when the caller wants a slice
  // rather than a tail. Takes precedence over resume_from.
  string range;
};

// Shared easy-handle perform. `wcb`/`wdata` sink the body. Fills
// `*http_status` with the response code. When `progress` is non-null the
// transfer runs with the xferinfo callback (it draws the bar when a stream
// is set and/or polls the cancel predicate). Returns the CURLcode.
CURLcode
curl_perform_(const string& url, const string& token, const PerformOpts& o,
              size_t (*wcb)(char*, size_t, size_t, void*), void* wdata,
              long* http_status, ProgressCtx* progress)
{
  CURL* c = curl_easy_init();
  if (!c) {
    return CURLE_FAILED_INIT;
  }
  struct curl_slist* hdrs = nullptr;
  hdrs = curl_slist_append(hdrs, "User-Agent: vpipe-model-fetch/1");
  string auth;
  if (!token.empty()) {
    auth = "Authorization: Bearer " + token;
    hdrs = curl_slist_append(hdrs, auth.c_str());
  }
  curl_easy_setopt(c, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);
  curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
  if (progress) {
    // Attach the xferinfo callback whenever a ProgressCtx is present -- it
    // carries the bar stream and/or the cancel predicate (the callback
    // no-ops the bar when no stream is set).
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(c, CURLOPT_XFERINFOFUNCTION, &progress_cb_);
    curl_easy_setopt(c, CURLOPT_XFERINFODATA, progress);
  } else {
    curl_easy_setopt(c, CURLOPT_NOPROGRESS, 1L);
  }
  curl_easy_setopt(c, CURLOPT_SSL_VERIFYPEER, o.verify_tls ? 1L : 0L);
  curl_easy_setopt(c, CURLOPT_SSL_VERIFYHOST, o.verify_tls ? 2L : 0L);
  if (o.timeout_s > 0) {
    curl_easy_setopt(c, CURLOPT_TIMEOUT, o.timeout_s);
  }
  if (o.stall_s > 0) {
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, kStallBytesPerSec);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, o.stall_s);
  }
  if (!o.range.empty()) {
    curl_easy_setopt(c, CURLOPT_RANGE, o.range.c_str());
  } else if (o.resume_from > 0) {
    curl_easy_setopt(c, CURLOPT_RESUME_FROM_LARGE, o.resume_from);
  }
  curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 30L);
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, wcb);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, wdata);
  CURLcode rc = curl_easy_perform(c);
  if (http_status) {
    *http_status = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, http_status);
  }
  curl_slist_free_all(hdrs);
  curl_easy_cleanup(c);
  return rc;
}

string
hex_(const unsigned char* d, std::size_t n)
{
  static const char kHex[] = "0123456789abcdef";
  string out;
  out.reserve(n * 2);
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(kHex[d[i] >> 4]);
    out.push_back(kHex[d[i] & 0x0f]);
  }
  return out;
}

// ---- resumable file download -------------------------------------------

// Sink for one download attempt. Plain: what makes a resume safe is
// libcurl refusing a ranged request the server did not answer with a
// Content-Range (CURLE_RANGE_ERROR), and the checksum at the end
// catching everything subtler -- including a 206 granted for some
// other window, which arrives looking exactly like a good one.
struct DownloadSink {
  std::ofstream ofs;
  fs::path      path;
};

size_t
write_to_sink_(char* p, size_t s, size_t n, void* u)
{
  auto* d = static_cast<DownloadSink*>(u);
  d->ofs.write(p, static_cast<std::streamsize>(s * n));
  return d->ofs.good() ? s * n : 0;
}

// Which failures are worth another attempt. A transport that dropped,
// stalled or half-delivered will usually get further next time; 401,
// 403, 404 and a cancelled transfer will not, and retrying those only
// delays the message the caller needs to see.
bool
retryable_(CURLcode rc, long status)
{
  if (rc == CURLE_ABORTED_BY_CALLBACK) {
    return false;                       // the caller asked us to stop
  }
  if (rc != CURLE_OK) {
    return true;                        // transport: timed out, reset...
  }
  return status == 408 || status == 429 || status >= 500;
}

// Waits between attempts. Bounded on purpose: the point is to ride out
// a blip, not to hold a pipeline open for an hour. The partial file
// survives either way, so a fetch that exhausts its attempts and is
// re-run later picks up where this one stopped.
constexpr long kBackoffS[] = {2, 5, 15, 30, 60};

// Sleep `secs`, waking early if the caller cancels. False -> cancelled.
bool
nap_(long secs, const std::function<bool()>* cancel)
{
  for (long i = 0; i < secs * 4; ++i) {
    if (cancel && (*cancel)()) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  return !(cancel && (*cancel)());
}

std::uint64_t
file_size_or_0_(const fs::path& p)
{
  std::error_code ec;
  const auto n = fs::file_size(p, ec);
  return ec ? 0 : static_cast<std::uint64_t>(n);
}

}

string
human_bytes(std::uint64_t n)
{
  const char* u[] = { "B", "KB", "MB", "GB", "TB" };
  double v = static_cast<double>(n);
  int i = 0;
  while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
  return fmt("{:.1f} {}", v, u[i])();
}


void
ensure_curl_global_init()
{
  static std::once_flag once;
  std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

bool
http_get_text(const string& url, const string& token, bool verify_tls,
              long timeout_s, string& out, long& status, string& err)
{
  out.clear();
  PerformOpts o;
  o.verify_tls = verify_tls;
  o.timeout_s  = timeout_s;
  CURLcode rc = curl_perform_(url, token, o, &write_to_string_, &out,
                              &status, nullptr);
  if (rc != CURLE_OK) {
    err = curl_easy_strerror(rc);
    return false;
  }
  if (status < 200 || status >= 300) {
    err = fmt("HTTP {}", status)();
    return false;
  }
  return true;
}

bool
http_get_range(const string& url, const string& token, bool verify_tls,
               long stall_s, std::uint64_t off, std::uint64_t last,
               string& out, long& status, string& err,
               const std::function<bool()>* cancel,
               const std::function<void(std::uint64_t)>* on_bytes)
{
  out.clear();
  ProgressCtx pctx;
  pctx.cancel = cancel;
  CountingSink sink;
  sink.out      = &out;
  sink.on_bytes = on_bytes;
  PerformOpts o;
  o.verify_tls = verify_tls;
  o.stall_s    = stall_s;
  // The signed URLs the CAS hands out pin the range in the signature,
  // so this has to be the exact range the manifest named -- an open
  // "bytes=N-" would be refused.
  o.range = fmt("{}-{}", off, last)();
  CURLcode rc = curl_perform_(url, token, o, &write_counting_, &sink,
                              &status, cancel ? &pctx : nullptr);
  if (rc != CURLE_OK) {
    err = curl_easy_strerror(rc);
    return false;
  }
  if (status < 200 || status >= 300) {
    err = fmt("HTTP {}", status)();
    return false;
  }
  return true;
}

// ---- integrity ---------------------------------------------------------

bool
file_digest_hex(const fs::path& p, bool git_blob, string& hex, string& err)
{
  std::error_code ec;
  const auto sz = fs::file_size(p, ec);
  if (ec) {
    err = fmt("cannot size '{}': {}", p.string(), ec.message())();
    return false;
  }
  std::ifstream in(p, std::ios::binary);
  if (!in) {
    err = fmt("cannot open '{}' to check it", p.string())();
    return false;
  }
  CC_SHA1_CTX   s1{};
  CC_SHA256_CTX s256{};
  if (git_blob) {
    CC_SHA1_Init(&s1);
    string hdr = fmt("blob {}", static_cast<std::uint64_t>(sz))();
    hdr.push_back('\0');
    CC_SHA1_Update(&s1, hdr.data(), static_cast<CC_LONG>(hdr.size()));
  } else {
    CC_SHA256_Init(&s256);
  }
  vector<char> buf(std::size_t{1} << 20);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize got = in.gcount();
    if (got <= 0) {
      break;
    }
    if (git_blob) {
      CC_SHA1_Update(&s1, buf.data(), static_cast<CC_LONG>(got));
    } else {
      CC_SHA256_Update(&s256, buf.data(), static_cast<CC_LONG>(got));
    }
  }
  if (in.bad()) {
    err = fmt("read error while checking '{}'", p.string())();
    return false;
  }
  if (git_blob) {
    unsigned char d[CC_SHA1_DIGEST_LENGTH];
    CC_SHA1_Final(d, &s1);
    hex = hex_(d, sizeof d);
  } else {
    unsigned char d[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(d, &s256);
    hex = hex_(d, sizeof d);
  }
  return true;
}

FileCheck
check_file_digest(const fs::path& p, const FileDigest& want, string& detail)
{
  std::error_code ec;
  const auto have = fs::file_size(p, ec);
  if (ec) {
    detail = fmt("cannot size '{}': {}", p.string(), ec.message())();
    return FileCheck::Unreadable;
  }
  if (want.size > 0 && static_cast<std::uint64_t>(have) != want.size) {
    detail = fmt("{} bytes on disk, {} published",
                 static_cast<std::uint64_t>(have), want.size)();
    return FileCheck::Mismatch;
  }
  // LFS names the object by its content hash, so prefer it; git's blob
  // id is what is left for the small files git stores itself.
  const bool    git_blob = want.sha256.empty();
  const string& published = git_blob ? want.git_oid : want.sha256;
  if (published.empty()) {
    detail = "the repo publishes no checksum for this file";
    return FileCheck::NotPublished;
  }
  const char* alg = git_blob ? "git-blob sha1" : "sha256";
  string got, err;
  if (!file_digest_hex(p, git_blob, got, err)) {
    detail = err;
    return FileCheck::Unreadable;
  }
  if (got != lower_(published)) {
    detail = fmt("{} is {}, the repo publishes {}", alg, got, published)();
    return FileCheck::Mismatch;
  }
  detail = fmt("{} {}", alg, got)();
  return FileCheck::Ok;
}

// ---- download ----------------------------------------------------------

bool
fetch_file(const SessionContextIntf* s, const FetchRequest& req,
           const FetchOpts& o, const fs::path& dest, long& status,
           string& err, UiProgress* progress,
           const std::function<bool()>* cancel, FileCheck* checked)
{
  const string&     url   = req.url;
  const string&     token = req.token;
  const FileDigest& want  = req.want;
  std::error_code ec;
  fs::create_directories(dest.parent_path(), ec);
  const fs::path part = fs::path(dest.string() + ".part");
  const string   name = dest.filename().string();
  const unsigned attempts = o.retries + 1;

  bool fresh = false;                  // next attempt must not resume
  status     = 0;
  err.clear();
  for (unsigned a = 0; a < attempts; ++a) {
    if (cancel && (*cancel)()) {
      err = "canceled";
      return false;
    }
    if (fresh) {
      fs::remove(part, ec);
      fresh = false;
    }
    curl_off_t have = static_cast<curl_off_t>(file_size_or_0_(part));
    // What makes resume safe is the check at the end: a part left by an
    // older revision of the object splices into a file that is the
    // right LENGTH and wrong in the middle, and only a checksum finds
    // that. With nothing published to check against there is no such
    // backstop, so start over instead -- the sources that publish
    // nothing here serve megabytes, not gigabytes.
    if (have > 0 && !o.verify) {
      fs::remove(part, ec);
      have = 0;
    } else if (have > 0 && want.sha256.empty() && want.git_oid.empty()) {
      fs::remove(part, ec);
      have = 0;
    }
    // Only a prefix of the object can be resumed onto. Longer than the
    // published size means the part belongs to something else; exactly
    // the published size means a fetch was killed between the last byte
    // and the rename, which is worth checking before paying for the
    // bytes a second time.
    bool transfer = true;
    if (want.size > 0 && have > static_cast<curl_off_t>(want.size)) {
      fs::remove(part, ec);
      have = 0;
    } else if (want.size > 0
               && have == static_cast<curl_off_t>(want.size)) {
      transfer = false;
    }
    // The content store, when the repo publishes a hash to rebuild
    // from: the file arrives as many ranges at once instead of one
    // stream, which is where the throughput actually is. It appends to
    // the same part in order, so a failure here is not wasted -- the
    // plain path below continues from whatever it managed.
    if (transfer && !req.xet.hash.empty() && o.xet_streams > 0) {
      string xerr;
      if (xet_fetch(s, req.xet, token, o, part, want.size, xerr, progress,
                    cancel)) {
        transfer = false;
      } else if (cancel && (*cancel)()) {
        err = "canceled";
        return false;
      } else if (s) {
        s->info(fmt("    {}: the content store could not serve this ({});"
                    " taking it as one stream", name, xerr));
      }
      have = static_cast<curl_off_t>(file_size_or_0_(part));
      if (want.size > 0 && have == static_cast<curl_off_t>(want.size)) {
        transfer = false;
      }
    }
    if (transfer) {
      if (have > 0 && s) {
        s->info(fmt("    {}: resuming at {}", name, human_bytes(
            static_cast<std::uint64_t>(have))));
      }
      DownloadSink sink;
      sink.path = part;
      sink.ofs.open(part, have > 0 ? (std::ios::binary | std::ios::app)
                                   : (std::ios::binary | std::ios::trunc));
      if (!sink.ofs) {
        err = fmt("cannot open '{}' for writing", part.string())();
        return false;
      }
      ProgressCtx pctx;
      pctx.progress = progress;
      pctx.cancel   = cancel;
      pctx.base     = &have;
      PerformOpts po;
      po.verify_tls  = o.verify_tls;
      po.stall_s     = o.stall_s;
      po.resume_from = have;
      status = 0;
      const CURLcode rc =
          curl_perform_(url, token, po, &write_to_sink_, &sink, &status,
                        (progress || cancel) ? &pctx : nullptr);
      sink.ofs.close();
      if (rc != CURLE_OK || status < 200 || status >= 300) {
        err = rc != CURLE_OK ? string(curl_easy_strerror(rc))
                             : fmt("HTTP {}", status)();
        // Two ways a resume is refused rather than merely interrupted
        // -- the range was not satisfiable, or the server does not do
        // partial content at all. Both mean the same thing: stop trying
        // to continue and take the whole file next time round.
        if ((rc == CURLE_OK && status == 416)
            || rc == CURLE_RANGE_ERROR) {
          fresh = true;
        } else if (!retryable_(rc, status)) {
          // Nothing more to try. The part stays: it is still a valid
          // prefix, and the next RUN of this fetch will continue it.
          return false;
        }
        if (a + 1 < attempts) {
          const long wait = kBackoffS[std::min<std::size_t>(
              a, (sizeof kBackoffS / sizeof *kBackoffS) - 1)];
          if (s) {
            s->info(fmt("    {}: interrupted at {} ({}); attempt {}/{} "
                        "in {}s", name,
                        human_bytes(file_size_or_0_(part)), err,
                        a + 2, attempts, wait));
          }
          if (!nap_(wait, cancel)) {
            err = "canceled";
            return false;
          }
        }
        continue;
      }
    }
    if (o.verify) {
      string detail;
      const FileCheck v = check_file_digest(part, want, detail);
      if (v == FileCheck::Mismatch || v == FileCheck::Unreadable) {
        err = fmt("integrity check failed -- {}", detail)();
        // The bytes are wrong, so resuming onto them would only carry
        // the damage forward. Start this one over.
        fs::remove(part, ec);
        fresh = true;
        if (a + 1 < attempts) {
          if (s) {
            s->warn(fmt("    {}: {}; re-fetching whole, attempt {}/{}",
                        name, err, a + 2, attempts));
          }
          continue;
        }
        return false;
      }
      if (checked) {
        *checked = v;
      }
      // Only for the files big enough to have earned a progress
      // report: a line per small file doubles the length of a 200-file
      // fetch to say nothing new, while on a shard that took twenty
      // minutes it is the confirmation that the bytes are good.
      if (v == FileCheck::Ok && s
          && file_size_or_0_(part) >= kBigFileBytes) {
        s->info(fmt("    {}: {}", name, detail));
      }
    }
    // Renamed last, and straight over any older copy: `dest` appearing
    // is the signal that the bytes are all there and have been checked,
    // so there must be no window where neither name exists.
    fs::rename(part, dest, ec);
    if (ec) {
      err = fmt("cannot move '{}' into place: {}", part.string(),
                ec.message())();
      return false;
    }
    return true;
  }
  return false;
}


}
