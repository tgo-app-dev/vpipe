#include "minitest.h"
#include "stages/resumable-fetch.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;
using namespace vpipe;
namespace fs = std::filesystem;

namespace {

// A deterministic blob, so a resumed download can be compared with the
// bytes it was supposed to produce rather than with its own hash.
string
blob_(size_t n)
{
  string out;
  out.resize(n);
  uint32_t x = 0x1234567u;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    out[i] = static_cast<char>((x >> 16) & 0xff);
  }
  return out;
}

string
read_file_(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary);
  return string((std::istreambuf_iterator<char>(in)),
                std::istreambuf_iterator<char>());
}

void
write_file_(const fs::path& p, const string& bytes)
{
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// A loopback HTTP server that serves one blob and can misbehave on
// purpose: cut a response short mid-body (a dropped connection), ignore
// the Range header (a CDN that does not do partial content), or answer
// with a chosen status. It records what it was asked for, so a test can
// assert that the SECOND request resumed rather than started over.
class BlobServer {
public:
  explicit BlobServer(string body) : _body(std::move(body)) {}
  ~BlobServer() { stop(); }

  bool start()
  {
    _fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_fd < 0) { return false; }
    int yes = 1;
    ::setsockopt(_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    if (::bind(_fd, reinterpret_cast<sockaddr*>(&addr),
               sizeof(addr)) < 0) {
      ::close(_fd); _fd = -1; return false;
    }
    socklen_t alen = sizeof(addr);
    if (::getsockname(_fd, reinterpret_cast<sockaddr*>(&addr),
                      &alen) < 0) {
      ::close(_fd); _fd = -1; return false;
    }
    _port = ntohs(addr.sin_port);
    if (::listen(_fd, 8) < 0) {
      ::close(_fd); _fd = -1; return false;
    }
    _t = std::thread([this] { run_(); });
    return true;
  }

  void stop()
  {
    if (_fd >= 0) {
      int fd = _fd;
      _fd = -1;
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
    }
    if (_t.joinable()) { _t.join(); }
  }

  string url() const
  {
    return "http://127.0.0.1:" + std::to_string(_port) + "/blob.bin";
  }

  // Send only this many body bytes on the next `times` responses, then
  // close the socket -- what a transfer that drops mid-file looks like.
  void cut_after(size_t bytes, int times)
  {
    _cut = bytes;
    _cut_left = times;
  }
  void ignore_range(bool on) { _ignore = on; }
  // Answer 206, but for a window this many bytes BEFORE the one asked
  // for -- a server that honours partial content and gets the offset
  // wrong, which is the case a status-code check cannot see.
  void skew_range(long bytes) { _skew = bytes; }
  void force_status(int st)  { _status = st; }

  int requests() const     { return _requests.load(); }
  long last_range() const  { return _last_range.load(); }
  bool saw_range() const   { return _saw_range.load(); }

private:
  void run_()
  {
    while (_fd >= 0) {
      sockaddr_in caddr{};
      socklen_t   clen = sizeof(caddr);
      int cfd = ::accept(_fd, reinterpret_cast<sockaddr*>(&caddr), &clen);
      if (cfd < 0) { return; }
      handle_(cfd);
      ::close(cfd);
    }
  }

  static void send_all_(int fd, const char* p, size_t n)
  {
    while (n > 0) {
      const ssize_t w = ::send(fd, p, n, 0);
      if (w <= 0) { return; }
      p += w;
      n -= static_cast<size_t>(w);
    }
  }

  void handle_(int cfd)
  {
    string head;
    char   chunk[2048];
    while (head.find("\r\n\r\n") == string::npos) {
      const ssize_t n = ::recv(cfd, chunk, sizeof(chunk), 0);
      if (n <= 0) { return; }
      head.append(chunk, chunk + static_cast<size_t>(n));
    }
    _requests.fetch_add(1);

    long start = 0;
    const auto r = head.find("Range: bytes=");
    if (r != string::npos && !_ignore.load()) {
      start = std::strtol(head.c_str() + r + 13, nullptr, 10);
    }
    _saw_range.store(r != string::npos);
    _last_range.store(r == string::npos ? -1 : start);
    if (start > 0) {
      start -= std::min<long>(_skew.load(), start);
    }

    const int forced = _status.load();
    if (forced != 0) {
      const string resp = "HTTP/1.1 " + std::to_string(forced)
                        + " Nope\r\nContent-Length: 0\r\n"
                          "Connection: close\r\n\r\n";
      send_all_(cfd, resp.data(), resp.size());
      return;
    }
    if (start < 0 || static_cast<size_t>(start) > _body.size()) {
      const string resp = "HTTP/1.1 416 Range Not Satisfiable\r\n"
                          "Content-Length: 0\r\nConnection: close\r\n\r\n";
      send_all_(cfd, resp.data(), resp.size());
      return;
    }

    const string part = _body.substr(static_cast<size_t>(start));
    string       resp;
    if (start > 0) {
      resp = "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes "
           + std::to_string(start) + "-"
           + std::to_string(_body.size() - 1) + "/"
           + std::to_string(_body.size()) + "\r\n";
    } else {
      resp = "HTTP/1.1 200 OK\r\n";
    }
    resp += "Accept-Ranges: bytes\r\nContent-Length: "
          + std::to_string(part.size()) + "\r\nConnection: close\r\n\r\n";
    send_all_(cfd, resp.data(), resp.size());

    size_t send_n = part.size();
    if (_cut_left.load() > 0) {
      _cut_left.fetch_sub(1);
      send_n = std::min(send_n, _cut.load());
    }
    send_all_(cfd, part.data(), send_n);
  }

  string              _body;
  int                 _fd   = -1;
  int                 _port = 0;
  std::thread         _t;
  std::atomic<size_t> _cut{0};
  std::atomic<int>    _cut_left{0};
  std::atomic<bool>   _ignore{false};
  std::atomic<long>   _skew{0};
  std::atomic<int>    _status{0};
  std::atomic<int>    _requests{0};
  std::atomic<long>   _last_range{-1};
  std::atomic<bool>   _saw_range{false};
};

// A scratch directory that cleans up after itself.
class TempDir {
public:
  TempDir()
  {
    _p = fs::temp_directory_path()
       / ("vpipe-fetch-" + std::to_string(::getpid()) + "-"
          + std::to_string(++_n));
    fs::create_directories(_p);
  }
  ~TempDir()
  {
    std::error_code ec;
    fs::remove_all(_p, ec);
  }
  const fs::path& path() const { return _p; }

private:
  fs::path           _p;
  static inline int  _n = 0;
};

FetchRequest
req_(const string& url, const FileDigest& want)
{
  FetchRequest r;
  r.url  = url;
  r.want = want;
  return r;
}

FetchOpts
opts_(unsigned retries)
{
  FetchOpts o;
  o.verify_tls = false;   // plain HTTP loopback
  o.stall_s    = 0;
  o.retries    = retries;
  o.verify     = true;
  return o;
}

}

// The digests are pinned against values produced OUTSIDE this code --
// `git hash-object` and `shasum -a 256` over the same six bytes. A
// checksum routine that agrees only with itself would pass every check
// it ever ran, which is the failure mode worth spending a test on.
TEST(resumable_fetch, digests_match_git_and_sha256) {
  TempDir dir;
  const fs::path f = dir.path() / "hello.txt";
  write_file_(f, "hello\n");

  string hex, err;
  ASSERT_TRUE(file_digest_hex(f, /*git_blob=*/true, hex, err));
  EXPECT_TRUE(hex == "ce013625030ba8dba906f756967f9e9ca394464a");
  ASSERT_TRUE(file_digest_hex(f, /*git_blob=*/false, hex, err));
  EXPECT_TRUE(hex
      == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03");
}

TEST(resumable_fetch, a_check_says_which_check_it_ran) {
  TempDir dir;
  const fs::path f = dir.path() / "hello.txt";
  write_file_(f, "hello\n");

  // LFS: the SHA-256 wins when both are published.
  FileDigest lfs;
  lfs.size    = 6;
  lfs.sha256  =
      "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03";
  lfs.git_oid = "not-even-a-hash";
  string detail;
  EXPECT_TRUE(check_file_digest(f, lfs, detail) == FileCheck::Ok);
  EXPECT_TRUE(detail.find("sha256") != string::npos);

  // A small file git stores itself carries only the blob id.
  FileDigest git;
  git.size    = 6;
  git.git_oid = "ce013625030ba8dba906f756967f9e9ca394464a";
  EXPECT_TRUE(check_file_digest(f, git, detail) == FileCheck::Ok);
  EXPECT_TRUE(detail.find("git-blob") != string::npos);

  // Nothing published is NOT a pass -- the caller has to be able to
  // tell "checked and fine" from "there was nothing to check".
  FileDigest none;
  none.size = 6;
  EXPECT_TRUE(check_file_digest(f, none, detail)
              == FileCheck::NotPublished);

  // Wrong bytes, and wrong length, are both mismatches.
  FileDigest bad = lfs;
  bad.sha256 = string(64, 'a');
  EXPECT_TRUE(check_file_digest(f, bad, detail) == FileCheck::Mismatch);
  FileDigest short_ = lfs;
  short_.size = 5;
  EXPECT_TRUE(check_file_digest(f, short_, detail) == FileCheck::Mismatch);
  EXPECT_TRUE(detail.find("bytes on disk") != string::npos);

  FileDigest missing = lfs;
  EXPECT_TRUE(check_file_digest(dir.path() / "nope.txt", missing, detail)
              == FileCheck::Unreadable);
}

// The whole point of the exercise: a transfer that dies two thirds of
// the way through a big file continues from there instead of paying for
// the first two thirds again.
TEST(resumable_fetch, a_dropped_transfer_resumes_where_it_stopped) {
  ensure_curl_global_init();
  const string body = blob_(200000);
  BlobServer   srv(body);
  ASSERT_TRUE(srv.start());
  TempDir dir;
  const fs::path dest = dir.path() / "sub" / "blob.bin";

  srv.cut_after(120000, 1);          // drop the first attempt mid-body

  FileDigest want;
  want.size = body.size();
  {
    const fs::path ref = dir.path() / "ref.bin";
    write_file_(ref, body);
    string err;
    ASSERT_TRUE(file_digest_hex(ref, false, want.sha256, err));
  }

  long   status = 0;
  string err;
  ASSERT_TRUE(fetch_file(nullptr, req_(srv.url(), want), opts_(2),
                         dest, status, err, nullptr));
  EXPECT_TRUE(read_file_(dest) == body);
  // Two requests, and the second asked to continue from exactly what
  // the first managed to write.
  EXPECT_TRUE(srv.requests() == 2);
  EXPECT_TRUE(srv.last_range() == 120000);
  // The partial file is gone and the real one is in place -- a name
  // that exists always means a complete, checked file.
  EXPECT_FALSE(fs::exists(fs::path(dest.string() + ".part")));
  srv.stop();
}

// A server that answers a ranged request with the whole object would
// otherwise have its body appended to the prefix already on disk,
// producing a file longer than the original that no checksum saves.
TEST(resumable_fetch, a_range_the_server_ignores_is_not_appended) {
  ensure_curl_global_init();
  const string body = blob_(150000);
  BlobServer   srv(body);
  ASSERT_TRUE(srv.start());
  TempDir dir;
  const fs::path dest = dir.path() / "blob.bin";

  // Leave a prefix behind, then refuse to honour the Range.
  write_file_(fs::path(dest.string() + ".part"), body.substr(0, 90000));
  srv.ignore_range(true);

  FileDigest want;
  want.size = body.size();
  {
    const fs::path ref = dir.path() / "ref.bin";
    write_file_(ref, body);
    string err;
    ASSERT_TRUE(file_digest_hex(ref, false, want.sha256, err));
  }

  long   status = 0;
  string err;
  ASSERT_TRUE(fetch_file(nullptr, req_(srv.url(), want), opts_(1),
                         dest, status, err, nullptr));
  EXPECT_TRUE(read_file_(dest) == body);
  // The first attempt asked to continue and was refused; the second
  // took the object whole rather than splicing onto the stale prefix.
  EXPECT_TRUE(srv.requests() == 2);
  EXPECT_FALSE(srv.saw_range());
  srv.stop();
}

// A 206 for a window we did not ask for is the case a response-code
// check cannot see: the status says "partial content", so the bytes
// would be appended to a prefix they do not continue, and the file ends
// up the right length with a hole in the middle.
TEST(resumable_fetch, a_206_for_the_wrong_window_is_not_spliced_on) {
  ensure_curl_global_init();
  const string body = blob_(120000);
  BlobServer   srv(body);
  ASSERT_TRUE(srv.start());
  TempDir dir;
  const fs::path dest = dir.path() / "blob.bin";
  write_file_(fs::path(dest.string() + ".part"), body.substr(0, 90000));
  srv.skew_range(10000);            // grants 80000 when asked for 90000

  FileDigest want;
  want.size = body.size();
  {
    const fs::path ref = dir.path() / "ref.bin";
    write_file_(ref, body);
    string err;
    ASSERT_TRUE(file_digest_hex(ref, false, want.sha256, err));
  }

  long   status = 0;
  string err;
  ASSERT_TRUE(fetch_file(nullptr, req_(srv.url(), want), opts_(2),
                         dest, status, err, nullptr));
  EXPECT_TRUE(read_file_(dest) == body);
  // The skewed attempt was refused, and the retry took it whole.
  EXPECT_TRUE(srv.requests() == 2);
  EXPECT_FALSE(srv.saw_range());
  srv.stop();
}

// A part left over from another revision has the right length and the
// wrong bytes -- the one case where resuming silently produces a file
// that is complete and wrong. The checksum is what catches it.
TEST(resumable_fetch, a_part_that_fails_its_checksum_is_refetched) {
  ensure_curl_global_init();
  const string body = blob_(50000);
  BlobServer   srv(body);
  ASSERT_TRUE(srv.start());
  TempDir dir;
  const fs::path dest = dir.path() / "blob.bin";
  write_file_(fs::path(dest.string() + ".part"), string(body.size(), 'x'));

  FileDigest want;
  want.size = body.size();
  {
    const fs::path ref = dir.path() / "ref.bin";
    write_file_(ref, body);
    string err;
    ASSERT_TRUE(file_digest_hex(ref, false, want.sha256, err));
  }

  long   status = 0;
  string err;
  ASSERT_TRUE(fetch_file(nullptr, req_(srv.url(), want), opts_(2),
                         dest, status, err, nullptr));
  EXPECT_TRUE(read_file_(dest) == body);
  // Taken whole rather than continued: the bad prefix was dropped.
  EXPECT_TRUE(srv.requests() == 1);
  EXPECT_TRUE(srv.last_range() == -1);
  srv.stop();
}

// Bytes that never match are a failure, not an infinite retry loop.
TEST(resumable_fetch, a_wrong_checksum_fails_after_its_attempts) {
  ensure_curl_global_init();
  const string body = blob_(20000);
  BlobServer   srv(body);
  ASSERT_TRUE(srv.start());
  TempDir dir;
  const fs::path dest = dir.path() / "blob.bin";

  FileDigest want;
  want.size   = body.size();
  want.sha256 = string(64, 'b');     // not this file, and never will be

  long   status = 0;
  string err;
  EXPECT_FALSE(fetch_file(nullptr, req_(srv.url(), want), opts_(1),
                         dest, status, err, nullptr));
  EXPECT_TRUE(err.find("integrity check failed") != string::npos);
  EXPECT_TRUE(srv.requests() == 2);          // the first plus one retry
  EXPECT_FALSE(fs::exists(dest));
  EXPECT_FALSE(fs::exists(fs::path(dest.string() + ".part")));
  srv.stop();
}

// A 404 is an answer, not a blip. Retrying it only delays the message.
TEST(resumable_fetch, a_404_is_not_retried) {
  ensure_curl_global_init();
  BlobServer srv(blob_(1000));
  ASSERT_TRUE(srv.start());
  srv.force_status(404);
  TempDir dir;

  long   status = 0;
  string err;
  EXPECT_FALSE(fetch_file(nullptr, req_(srv.url(), FileDigest{}),
                          opts_(3), dir.path() / "blob.bin", status, err,
                          nullptr));
  EXPECT_TRUE(status == 404);
  EXPECT_TRUE(srv.requests() == 1);
  srv.stop();
}

// A source that publishes no checksum still downloads; it just cannot
// be checked, which the caller is told rather than sold as a pass.
TEST(resumable_fetch, an_unchecked_file_still_downloads) {
  ensure_curl_global_init();
  const string body = blob_(4096);
  BlobServer   srv(body);
  ASSERT_TRUE(srv.start());
  TempDir dir;
  const fs::path dest = dir.path() / "blob.bin";
  // And it is NOT resumed: with no checksum coming, a stale part would
  // splice into a file of the right length that is wrong in the middle,
  // with nothing left to catch it.
  write_file_(fs::path(dest.string() + ".part"), string(1024, 'x'));

  long   status = 0;
  string err;
  ASSERT_TRUE(fetch_file(nullptr, req_(srv.url(), FileDigest{}), opts_(1),
                         dest, status, err, nullptr));
  EXPECT_TRUE(read_file_(dest) == body);
  EXPECT_FALSE(srv.saw_range());
  string detail;
  EXPECT_TRUE(check_file_digest(dest, FileDigest{}, detail)
              == FileCheck::NotPublished);
  srv.stop();
}

TEST(resumable_fetch, human_bytes_reads_at_every_scale) {
  EXPECT_TRUE(human_bytes(0) == "0.0 B");
  EXPECT_TRUE(human_bytes(1536) == "1.5 KB");
  EXPECT_TRUE(human_bytes(uint64_t{3} << 30) == "3.0 GB");
}
