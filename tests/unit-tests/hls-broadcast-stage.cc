#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "common/session.h"
#include "common/static-file-server.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/hls-broadcast-stage.h"

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <netinet/in.h>
#include <streambuf>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

class CerrSilencer {
public:
  CerrSilencer() : _saved(cerr.rdbuf()), _null() { cerr.rdbuf(&_null); }
  ~CerrSilencer() { cerr.rdbuf(_saved); }
private:
  struct NullBuf : public streambuf {
    int overflow(int c) override { return c; }
  };
  streambuf* _saved;
  NullBuf    _null;
};

class RepeatTensorSource : public TypedStage<RepeatTensorSource> {
public:
  static constexpr const char* kTypeName = "ut-hls-repeat-source";
  using TypedStage::TypedStage;

  TensorBeat tb;
  int        count       = 1;
  int        per_beat_us = 0;   // optional inter-beat pause

  Job process(RuntimeContext& ctx) override
  {
    if (_emitted >= count) {
      ctx.signal_done();
      co_return;
    }
    ++_emitted;
    if (per_beat_us > 0) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(per_beat_us));
    }
    co_await ctx.write(0, make_payload<TensorBeatPayload>(tb));
  }

private:
  int _emitted = 0;
};

class ClosedSource : public TypedStage<ClosedSource> {
public:
  static constexpr const char* kTypeName = "ut-hls-closed-source";
  using TypedStage::TypedStage;

  Job process(RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};

// Keeps its oport OPEN but emits NOTHING for `naps` iterations (each a short
// sleep) before closing -- a producer that is wired + live but has not yet
// produced any data. Lets a downstream audio-only hls-broadcast prime + self-
// clock silence during the idle window.
class IdleThenCloseSource : public TypedStage<IdleThenCloseSource> {
public:
  static constexpr const char* kTypeName = "ut-hls-idle-source";
  using TypedStage::TypedStage;

  int naps   = 30;
  int nap_ms = 50;

  Job process(RuntimeContext& ctx) override
  {
    if (_n >= naps) { ctx.signal_done(); co_return; }
    ++_n;
    std::this_thread::sleep_for(std::chrono::milliseconds(nap_ms));
    co_return;   // stay open, produce nothing
  }

private:
  int _n = 0;
};

TensorBeat
make_tensor_(int H, int W, float v = 0.5f)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = {3, H, W};
  tb.resize_contiguous(static_cast<size_t>(3) * H * W);
  float* p = tb.as_f32();
  for (size_t i = 0; i < static_cast<size_t>(3) * H * W; ++i) {
    p[i] = v;
  }
  return tb;
}

TensorBeat
make_u8_tensor_(int H, int W, uint8_t v = 128)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::U8;
  tb.shape = {3, H, W};
  tb.resize_contiguous(static_cast<size_t>(3) * H * W);
  std::memset(tb.as_u8(), v, static_cast<size_t>(3) * H * W);
  return tb;
}

// A mono PCM audio beat: rank-1 [n] F32 tone, with a sideband carrying
// sample_rate (+ optional timestamp_us). This is the shape audio-to-pcm
// / text-to-speech emit.
TensorBeat
make_pcm_tensor_(int n, int sample_rate,
                 bool with_ts = false, uint64_t ts_us = 0)
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = {static_cast<int64_t>(n)};
  tb.resize_contiguous(static_cast<size_t>(n));
  float* p = tb.as_f32();
  for (int i = 0; i < n; ++i) {
    // A quiet 440 Hz tone -- content doesn't matter, just be non-zero.
    p[i] = 0.05f * std::sin(2.0f * 3.14159265f * 440.0f
                            * static_cast<float>(i)
                            / static_cast<float>(sample_rate));
  }
  FlexData sb = FlexData::make_object();
  sb.as_object().insert("sample_rate", FlexData::make_int(sample_rate));
  if (with_ts) {
    sb.as_object().insert("timestamp_us", FlexData::make_uint(ts_us));
  }
  tb.sideband = std::move(sb);
  return tb;
}

string
unique_tmp_dir_(const char* tag)
{
  string p = "/tmp/vpipe-hls-";
  p += tag;
  p += "-";
  p += to_string(::getpid());
  p += "-";
  p += to_string(static_cast<long long>(
      chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(p);
  return p;
}

void
rm_dir_(const string& p)
{
  std::error_code ec;
  std::filesystem::remove_all(p, ec);
}

// Tiny HTTP request to a TCP socket. Returns the full raw response
// (headers + body) or empty on connection failure. `method` defaults
// to GET; `extra` is a string of zero or more additional header lines,
// each terminated by CRLF (e.g. "Range: bytes=0-9\r\n").
string
http_request_(const string& host, int port,
              const string& method, const string& path,
              const string& extra = "")
{
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) { return ""; }
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port   = htons(static_cast<uint16_t>(port));
  ::inet_pton(AF_INET, host.c_str(), &a.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0) {
    ::close(fd);
    return "";
  }
  string req = method + " " + path + " HTTP/1.1\r\nHost: "
             + host + "\r\nConnection: close\r\n" + extra + "\r\n";
  if (::send(fd, req.data(), req.size(), 0) < 0) {
    ::close(fd);
    return "";
  }
  string out;
  char   buf[4096];
  while (true) {
    ssize_t n = ::recv(fd, buf, sizeof buf, 0);
    if (n <= 0) { break; }
    out.append(buf, buf + n);
  }
  ::close(fd);
  return out;
}

string
http_get_(const string& host, int port, const string& path)
{
  return http_request_(host, port, "GET", path);
}

int
ts_segment_count_(const InMemoryBlobStore& blobs)
{
  int n = 0;
  for (const auto& p : blobs.paths()) {
    if (p.size() >= 3
        && p.compare(p.size() - 3, 3, ".ts") == 0) {
      ++n;
    }
  }
  return n;
}

}  // namespace

TEST(static_file_server, mime_types_for_hls_extensions)
{
  EXPECT_TRUE(StaticFileServer::mime_type_for("/foo.m3u8")
              == "application/vnd.apple.mpegurl");
  EXPECT_TRUE(StaticFileServer::mime_type_for("/seg001.ts")
              == "video/mp2t");
  EXPECT_TRUE(StaticFileServer::mime_type_for("/init.mp4")
              == "video/mp4");
  EXPECT_TRUE(StaticFileServer::mime_type_for("/unknown.xyz")
              == "application/octet-stream");
}

TEST(static_file_server, serves_existing_file_and_404s_missing)
{
  CerrSilencer hush;
  Session sess;

  string dir = unique_tmp_dir_("sfs");
  // Write a small text file.
  {
    std::ofstream f(dir + "/hello.txt");
    f << "hi there";
  }

  StaticFileServer srv(&sess, dir, "127.0.0.1", 0);  // 0 = any port
  EXPECT_TRUE(srv.start());
  int port = srv.bound_port();
  EXPECT_TRUE(port > 0);

  string resp = http_get_("127.0.0.1", port, "/hello.txt");
  EXPECT_TRUE(resp.find("HTTP/1.1 200") != string::npos);
  EXPECT_TRUE(resp.find("text/plain") != string::npos);
  EXPECT_TRUE(resp.find("hi there") != string::npos);

  string miss = http_get_("127.0.0.1", port, "/no-such.txt");
  EXPECT_TRUE(miss.find("HTTP/1.1 404") != string::npos);

  string trav = http_get_("127.0.0.1", port, "/../etc/passwd");
  EXPECT_TRUE(trav.find("HTTP/1.1 400") != string::npos);

  srv.stop();
  EXPECT_TRUE(!srv.running());
  rm_dir_(dir);
}

TEST(static_file_server, serves_registered_blob_and_404s_unregistered)
{
  // Pure in-memory mode: no doc_root, only a blob store.
  CerrSilencer hush;
  Session sess;

  InMemoryBlobStore blobs;
  vector<uint8_t> body{'h', 'e', 'l', 'l', 'o'};
  blobs.put("/greet.txt", std::move(body), "text/plain; charset=utf-8");

  StaticFileServer srv(&sess, /*doc_root*/ "", "127.0.0.1", 0, &blobs);
  EXPECT_TRUE(srv.start());
  int port = srv.bound_port();
  EXPECT_TRUE(port > 0);

  string resp = http_get_("127.0.0.1", port, "/greet.txt");
  EXPECT_TRUE(resp.find("HTTP/1.1 200") != string::npos);
  EXPECT_TRUE(resp.find("text/plain") != string::npos);
  EXPECT_TRUE(resp.find("hello") != string::npos);
  // Content-Length must reflect the body length precisely.
  EXPECT_TRUE(resp.find("Content-Length: 5") != string::npos);

  // Replacing the entry serves the new bytes (different length).
  vector<uint8_t> body2{'h', 'i'};
  blobs.put("/greet.txt", std::move(body2),
            "text/plain; charset=utf-8");
  string resp2 = http_get_("127.0.0.1", port, "/greet.txt");
  EXPECT_TRUE(resp2.find("HTTP/1.1 200") != string::npos);
  EXPECT_TRUE(resp2.find("Content-Length: 2") != string::npos);

  // Erasing reverts to 404.
  blobs.erase("/greet.txt");
  string gone = http_get_("127.0.0.1", port, "/greet.txt");
  EXPECT_TRUE(gone.find("HTTP/1.1 404") != string::npos);

  srv.stop();
}

TEST(static_file_server, options_returns_cors_preflight)
{
  // hls.js / fetch() cross-origin issues OPTIONS first. Server must
  // 204 it with permissive Allow-* headers; returning 405 (the old
  // behaviour) blocks the actual GET in the browser.
  CerrSilencer hush;
  Session sess;

  InMemoryBlobStore blobs;
  blobs.put("/x.m3u8", vector<uint8_t>{'a'},
            "application/vnd.apple.mpegurl");

  StaticFileServer srv(&sess, "", "127.0.0.1", 0, &blobs);
  EXPECT_TRUE(srv.start());
  int port = srv.bound_port();

  // Default preflight (no Access-Control-Request-Headers).
  string r1 = http_request_("127.0.0.1", port, "OPTIONS", "/x.m3u8",
                            "Origin: https://example.com\r\n"
                            "Access-Control-Request-Method: GET\r\n");
  EXPECT_TRUE(r1.find("HTTP/1.1 204") != string::npos);
  EXPECT_TRUE(r1.find("Access-Control-Allow-Origin: *")
              != string::npos);
  EXPECT_TRUE(r1.find("Access-Control-Allow-Methods: "
                      "GET, HEAD, OPTIONS") != string::npos);
  EXPECT_TRUE(r1.find("Access-Control-Max-Age:") != string::npos);
  // Default header policy is permissive.
  EXPECT_TRUE(r1.find("Access-Control-Allow-Headers: *")
              != string::npos);

  // Echoes back the requested header list when the client supplies
  // Access-Control-Request-Headers (some browsers send a specific
  // list rather than rely on the wildcard).
  string r2 = http_request_("127.0.0.1", port, "OPTIONS", "/x.m3u8",
                            "Origin: https://example.com\r\n"
                            "Access-Control-Request-Method: GET\r\n"
                            "Access-Control-Request-Headers: "
                            "Range, X-Custom\r\n");
  EXPECT_TRUE(r2.find("HTTP/1.1 204") != string::npos);
  EXPECT_TRUE(r2.find(
      "Access-Control-Allow-Headers: Range, X-Custom")
      != string::npos);

  srv.stop();
}

TEST(static_file_server, get_advertises_accept_ranges)
{
  // Every 2xx response must carry "Accept-Ranges: bytes" so clients
  // know seeking is supported. Source filters (incl. MPC-HC's LAV)
  // stall when this header is missing.
  CerrSilencer hush;
  Session sess;

  InMemoryBlobStore blobs;
  vector<uint8_t> body(64);
  for (size_t i = 0; i < body.size(); ++i) {
    body[i] = static_cast<uint8_t>(i);
  }
  blobs.put("/bin", std::move(body), "application/octet-stream");

  StaticFileServer srv(&sess, "", "127.0.0.1", 0, &blobs);
  EXPECT_TRUE(srv.start());
  int port = srv.bound_port();

  string r = http_get_("127.0.0.1", port, "/bin");
  EXPECT_TRUE(r.find("HTTP/1.1 200") != string::npos);
  EXPECT_TRUE(r.find("Accept-Ranges: bytes") != string::npos);
  EXPECT_TRUE(r.find("Content-Length: 64")   != string::npos);

  srv.stop();
}

TEST(static_file_server, range_serves_206_partial_content)
{
  // Range: bytes=10-19 against a 64-byte blob -> 206 + Content-Range
  // + a body of exactly 10 bytes equal to bytes 10..19 of the blob.
  CerrSilencer hush;
  Session sess;

  InMemoryBlobStore blobs;
  vector<uint8_t> body(64);
  for (size_t i = 0; i < body.size(); ++i) {
    body[i] = static_cast<uint8_t>(i);
  }
  blobs.put("/bin", std::move(body), "application/octet-stream");

  StaticFileServer srv(&sess, "", "127.0.0.1", 0, &blobs);
  EXPECT_TRUE(srv.start());
  int port = srv.bound_port();

  string r = http_request_("127.0.0.1", port, "GET", "/bin",
                           "Range: bytes=10-19\r\n");
  EXPECT_TRUE(r.find("HTTP/1.1 206") != string::npos);
  EXPECT_TRUE(r.find("Content-Range: bytes 10-19/64") != string::npos);
  EXPECT_TRUE(r.find("Content-Length: 10") != string::npos);

  // Verify the body bytes match [10..19].
  auto bend = r.find("\r\n\r\n");
  EXPECT_TRUE(bend != string::npos);
  string body_bytes = r.substr(bend + 4);
  EXPECT_TRUE(body_bytes.size() == 10u);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_TRUE(static_cast<uint8_t>(body_bytes[i])
                == static_cast<uint8_t>(10 + i));
  }

  // Open-ended form "bytes=50-" -> last 14 bytes (50..63).
  string r2 = http_request_("127.0.0.1", port, "GET", "/bin",
                            "Range: bytes=50-\r\n");
  EXPECT_TRUE(r2.find("HTTP/1.1 206") != string::npos);
  EXPECT_TRUE(r2.find("Content-Range: bytes 50-63/64") != string::npos);
  EXPECT_TRUE(r2.find("Content-Length: 14") != string::npos);

  // Suffix form "bytes=-8" -> last 8 bytes (56..63).
  string r3 = http_request_("127.0.0.1", port, "GET", "/bin",
                            "Range: bytes=-8\r\n");
  EXPECT_TRUE(r3.find("HTTP/1.1 206") != string::npos);
  EXPECT_TRUE(r3.find("Content-Range: bytes 56-63/64") != string::npos);
  EXPECT_TRUE(r3.find("Content-Length: 8") != string::npos);

  srv.stop();
}

TEST(static_file_server, range_416_when_unsatisfiable)
{
  // Asking for a byte that doesn't exist must yield 416 with the
  // canonical "Content-Range: bytes */<total>" body-size advertise.
  CerrSilencer hush;
  Session sess;

  InMemoryBlobStore blobs;
  blobs.put("/tiny", vector<uint8_t>{'a', 'b', 'c'}, "text/plain");

  StaticFileServer srv(&sess, "", "127.0.0.1", 0, &blobs);
  EXPECT_TRUE(srv.start());
  int port = srv.bound_port();

  string r = http_request_("127.0.0.1", port, "GET", "/tiny",
                           "Range: bytes=100-200\r\n");
  EXPECT_TRUE(r.find("HTTP/1.1 416") != string::npos);
  EXPECT_TRUE(r.find("Content-Range: bytes */3") != string::npos);

  srv.stop();
}

TEST(static_file_server, range_falls_back_to_200_on_malformed)
{
  // Multi-range, unknown unit, and malformed forms should NOT 416 --
  // the spec lets a server ignore a Range request and respond 200.
  // That keeps non-byte units (e.g. "bytes=2-1" reversed, or
  // "lines=0-9") from breaking otherwise-correct clients.
  CerrSilencer hush;
  Session sess;

  InMemoryBlobStore blobs;
  blobs.put("/bin", vector<uint8_t>(32, 'z'), "application/octet-stream");

  StaticFileServer srv(&sess, "", "127.0.0.1", 0, &blobs);
  EXPECT_TRUE(srv.start());
  int port = srv.bound_port();

  // Multi-range: ignored, fall back to 200.
  string r1 = http_request_("127.0.0.1", port, "GET", "/bin",
                            "Range: bytes=0-5,10-15\r\n");
  EXPECT_TRUE(r1.find("HTTP/1.1 200") != string::npos);
  EXPECT_TRUE(r1.find("Content-Length: 32") != string::npos);

  // Unknown unit: ignored, fall back to 200.
  string r2 = http_request_("127.0.0.1", port, "GET", "/bin",
                            "Range: lines=0-9\r\n");
  EXPECT_TRUE(r2.find("HTTP/1.1 200") != string::npos);

  // Reversed range (end < start): ignored, fall back to 200.
  string r3 = http_request_("127.0.0.1", port, "GET", "/bin",
                            "Range: bytes=20-10\r\n");
  EXPECT_TRUE(r3.find("HTTP/1.1 200") != string::npos);

  srv.stop();
}

TEST(static_file_server, range_works_on_filesystem_doc_root)
{
  // Verify the disk-backed path also serves 206. (Same logic but a
  // separate code path: serve_file_ vs serve_blob_.)
  CerrSilencer hush;
  Session sess;

  string dir = unique_tmp_dir_("sfs_range");
  {
    std::ofstream f(dir + "/data.bin", std::ios::binary);
    for (int i = 0; i < 64; ++i) {
      char c = static_cast<char>(i);
      f.write(&c, 1);
    }
  }

  StaticFileServer srv(&sess, dir, "127.0.0.1", 0);
  EXPECT_TRUE(srv.start());
  int port = srv.bound_port();

  string r = http_request_("127.0.0.1", port, "GET", "/data.bin",
                           "Range: bytes=16-31\r\n");
  EXPECT_TRUE(r.find("HTTP/1.1 206") != string::npos);
  EXPECT_TRUE(r.find("Content-Range: bytes 16-31/64") != string::npos);
  EXPECT_TRUE(r.find("Content-Length: 16") != string::npos);
  auto bend = r.find("\r\n\r\n");
  EXPECT_TRUE(bend != string::npos);
  string body_bytes = r.substr(bend + 4);
  EXPECT_TRUE(body_bytes.size() == 16u);
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_TRUE(static_cast<uint8_t>(body_bytes[i])
                == static_cast<uint8_t>(16 + i));
  }

  srv.stop();
  rm_dir_(dir);
}

TEST(hls_broadcast_stage, defaults_and_derived_url)
{
  CerrSilencer hush;
  Session sess;

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("port", FlexData::make_int(9090));
  cfg.as_object().insert("bind_address",
                         FlexData::make_string("0.0.0.0"));

  HlsBroadcastStage stage(&sess, "bc", vector<InEdge>{},
                          std::move(cfg));
  EXPECT_TRUE(stage.playlist_name() == "stream.m3u8");
  // 0.0.0.0 should be rendered as localhost in the user-facing URL.
  EXPECT_TRUE(stage.playlist_url()
              == "http://localhost:9090/stream.m3u8");
  EXPECT_TRUE(!stage.encoder_initialized());
  EXPECT_TRUE(!stage.server_running());
  EXPECT_TRUE(stage.blobs().size() == 0);
  // Defaults: segment_duration=2s, live_start_offset=0 (tag off).
  EXPECT_TRUE(stage.segment_duration()  == 2.0);
  EXPECT_TRUE(stage.live_start_offset() == 0.0);
}

namespace {

// Run one hls-broadcast stage over `count` copies of `frame` (the
// RepeatTensorSource deep-copies it, sideband included) and return the
// stage's resolved {fps_num, fps_den} after the run. The encode fps is
// resolved on the first frame, so it's populated regardless of whether
// the encoder itself (codec availability) succeeded.
std::pair<int, int>
run_hls_resolve_fps_(Session& sess, FlexData cfg,
                     const TensorBeat& frame, int count)
{
  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RepeatTensorSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->tb    = frame;
  src_u->count = count;
  src_u->allocate_oports(1);
  auto* src = static_cast<RepeatTensorSource*>(
      pl->insert_stage(std::move(src_u)));

  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(
      pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  rt.launch();
  rt.wait_idle();
  rt.stop();
  return { bc->fps_num(), bc->fps_den() };
}

// Base config for the fps tests: cheap software encoder, deterministic
// PTS, no HTTP server. Callers layer fps_* on top (or omit for auto).
FlexData
fps_test_base_cfg_()
{
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("codec", FlexData::make_string("libx264"));
  cfg.as_object().insert("preset", FlexData::make_string("ultrafast"));
  cfg.as_object().insert("realtime", FlexData::make_bool(false));
  cfg.as_object().insert("serve_http", FlexData::make_bool(false));
  return cfg;
}

// A frame carrying an fps_num/fps_den sideband, the shape video-to-rgb
// forwards from the capture stream.
TensorBeat
make_tensor_with_fps_(int H, int W, unsigned fps_num, unsigned fps_den)
{
  TensorBeat tb = make_tensor_(H, W, 0.5f);
  FlexData sb = FlexData::make_object();
  sb.as_object().insert("fps_num", FlexData::make_uint(fps_num));
  sb.as_object().insert("fps_den", FlexData::make_uint(fps_den));
  tb.sideband = std::move(sb);
  return tb;
}

}  // namespace

TEST(hls_broadcast_stage, adopts_fps_from_sideband_when_unset)
{
  // No fps in config -> the stage adopts the input frame's sideband
  // rate (23.976 fps as 24000/1001), not the 30/1 fallback.
  CerrSilencer hush;
  Session sess;
  auto fps = run_hls_resolve_fps_(
      sess, fps_test_base_cfg_(),
      make_tensor_with_fps_(32, 32, 24000, 1001), 3);
  EXPECT_TRUE(fps.first  == 24000);
  EXPECT_TRUE(fps.second == 1001);
}

TEST(hls_broadcast_stage, config_fps_overrides_sideband)
{
  // An explicit fps in config wins over whatever the sideband says.
  CerrSilencer hush;
  Session sess;
  FlexData cfg = fps_test_base_cfg_();
  cfg.as_object().insert("fps_num", FlexData::make_int(10));
  cfg.as_object().insert("fps_den", FlexData::make_int(1));
  auto fps = run_hls_resolve_fps_(
      sess, std::move(cfg),
      make_tensor_with_fps_(32, 32, 24000, 1001), 3);
  EXPECT_TRUE(fps.first  == 10);
  EXPECT_TRUE(fps.second == 1);
}

TEST(hls_broadcast_stage, fps_falls_back_to_30_without_sideband_or_config)
{
  // No config fps and no sideband fps -> the 30/1 last-resort default.
  CerrSilencer hush;
  Session sess;
  auto fps = run_hls_resolve_fps_(
      sess, fps_test_base_cfg_(),
      make_tensor_(32, 32, 0.5f), 3);
  EXPECT_TRUE(fps.first  == 30);
  EXPECT_TRUE(fps.second == 1);
}

TEST(hls_broadcast_stage, accepts_fractional_segment_duration)
{
  // Pre-per-AU latency work shipped segment_duration as int (so 0.5
  // truncated to 0 and the muxer rejected the spec). It is now a
  // real number; verify the config parses fractions and the
  // keyframe-pacing threshold uses the float value (any int-cast in
  // the path would round 0.5 down to 0 and force-keyframe every
  // frame -- catastrophic for bitrate).
  CerrSilencer hush;
  Session sess;

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("segment_duration",
                         FlexData::make_real(0.5));
  cfg.as_object().insert("live_start_offset",
                         FlexData::make_real(1.0));
  HlsBroadcastStage stage(&sess, "bc", vector<InEdge>{},
                          std::move(cfg));
  EXPECT_TRUE(stage.segment_duration()  == 0.5);
  EXPECT_TRUE(stage.live_start_offset() == 1.0);
}

TEST(hls_broadcast_stage, live_start_offset_injects_ext_x_start_tag)
{
  // The lag triage motivated #EXT-X-START injection: with
  // live_start_offset > 0, every published playlist must carry
  //   #EXT-X-START:TIME-OFFSET=-N,PRECISE=YES
  // right after the leading #EXTM3U line so compliant players
  // (Safari, hls.js, ffplay) anchor playback ~N seconds behind the
  // live edge instead of the default 3 × target_duration. With
  // live_start_offset=0 the tag must NOT appear.
  Session sess;

  auto run_and_get_playlist = [&](double offset, double seg_dur) {
    auto pl = make_unique<Pipeline>("p", &sess);
    auto src_u = make_unique<RepeatTensorSource>(
        &sess, "src", vector<InEdge>{}, FlexData::make_object());
    src_u->tb    = make_tensor_(32, 32, 0.5f);
    src_u->count = 90;
    src_u->allocate_oports(1);
    auto* src = static_cast<RepeatTensorSource*>(
        pl->insert_stage(std::move(src_u)));

    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("codec",
                           FlexData::make_string("libx264"));
    cfg.as_object().insert("preset",
                           FlexData::make_string("ultrafast"));
    cfg.as_object().insert("realtime", FlexData::make_bool(false));
    cfg.as_object().insert("segment_duration",
                           FlexData::make_real(seg_dur));
    cfg.as_object().insert("live_start_offset",
                           FlexData::make_real(offset));
    cfg.as_object().insert("fps_num", FlexData::make_int(30));
    cfg.as_object().insert("fps_den", FlexData::make_int(1));
    cfg.as_object().insert("gop_size", FlexData::make_int(30));
    cfg.as_object().insert("serve_http", FlexData::make_bool(false));

    auto bc_u = make_unique<HlsBroadcastStage>(
        &sess, "bc", vector<InEdge>{{src, 0}}, std::move(cfg));
    auto* bc = static_cast<HlsBroadcastStage*>(
        pl->insert_stage(std::move(bc_u)));

    PipelineRuntime rt(pl.get(), &sess);
    rt.launch();
    rt.wait_idle();
    rt.stop();

    auto entry = bc->blobs().get("/stream.m3u8");
    if (!entry || !entry->bytes) { return string{}; }
    return string(entry->bytes->begin(), entry->bytes->end());
  };

  // Case 1: offset=1.5, segment_duration=1 -> tag present, with the
  // exact value (we rely on TIME-OFFSET=-1.5 to match the user
  // request for ~1.5 s lag).
  {
    CerrSilencer hush;
    string body = run_and_get_playlist(1.5, 1.0);
    EXPECT_TRUE(!body.empty());
    EXPECT_TRUE(body.find("#EXTM3U") == 0);
    EXPECT_TRUE(body.find("#EXT-X-START:TIME-OFFSET=-1.5,PRECISE=YES")
                != string::npos);
    // Must sit between #EXTM3U and the first #EXTINF so spec-
    // compliant players parse it before any media segment.
    auto pos_start = body.find("#EXT-X-START");
    auto pos_inf   = body.find("#EXTINF");
    EXPECT_TRUE(pos_start != string::npos);
    EXPECT_TRUE(pos_inf   != string::npos);
    EXPECT_TRUE(pos_start < pos_inf);
  }

  // Case 2: offset=0 -> no tag (preserves stock HLS behaviour for
  // users who haven't opted in).
  {
    CerrSilencer hush;
    string body = run_and_get_playlist(0.0, 1.0);
    EXPECT_TRUE(!body.empty());
    EXPECT_TRUE(body.find("#EXTM3U") == 0);
    EXPECT_TRUE(body.find("#EXT-X-START") == string::npos);
  }
}

TEST(hls_broadcast_stage, closes_on_iport_eos_without_setup)
{
  // No frames ever arrive -> encoder, muxer, and HTTP server should
  // not be initialized; no blobs should be registered.
  CerrSilencer hush;
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<ClosedSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->allocate_oports(1);
  auto* src = static_cast<ClosedSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("serve_http", FlexData::make_bool(false));

  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(
      pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(!bc->encoder_initialized());
  EXPECT_TRUE(!bc->server_running());
  EXPECT_TRUE(bc->blobs().size() == 0);
}

TEST(hls_broadcast_stage, end_to_end_publishes_playlist_and_segments)
{
  // Feed enough frames that the HLS muxer flushes at least one
  // segment. With hls_time=1 and fps=30, ~30 frames is one segment.
  // We feed 90 frames at 32x32 (cheap) to be safe.
  // NOTE: HLS mandates H.264/HEVC, so we use libx264 here. If
  // libx264 isn't built into the host FFmpeg the test is skipped.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RepeatTensorSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->tb    = make_tensor_(32, 32, 0.5f);
  src_u->count = 90;
  src_u->allocate_oports(1);
  auto* src = static_cast<RepeatTensorSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("codec", FlexData::make_string("libx264"));
  cfg.as_object().insert("preset", FlexData::make_string("ultrafast"));
  // Deterministic PTS for the test: 90 frames in fps=30 -> 3s media,
  // hls_time=1s -> 3 segments. Wall-clock mode would collapse all
  // 90 frames into a near-zero media window because the test source
  // emits them back-to-back without delay.
  cfg.as_object().insert("realtime", FlexData::make_bool(false));
  cfg.as_object().insert("segment_duration", FlexData::make_int(1));
  cfg.as_object().insert("fps_num", FlexData::make_int(30));
  cfg.as_object().insert("fps_den", FlexData::make_int(1));
  cfg.as_object().insert("gop_size", FlexData::make_int(30));
  // serve_http=false: verify side effects via the in-memory blob
  // store directly. The HTTP server is exercised by the
  // `serves_registered_blob_and_404s_unregistered` test above.
  cfg.as_object().insert("serve_http", FlexData::make_bool(false));

  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(
      pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // Playlist must have been published into the blob registry.
  auto pl_entry = bc->blobs().get("/stream.m3u8");
  EXPECT_TRUE(pl_entry.has_value());
  EXPECT_TRUE(pl_entry->bytes && !pl_entry->bytes->empty());
  EXPECT_TRUE(pl_entry->content_type
              == "application/vnd.apple.mpegurl");
  // Content should look like an HLS playlist.
  const auto& pb = *pl_entry->bytes;
  string pl_head(pb.begin(),
                 pb.begin() + std::min<size_t>(pb.size(), 7));
  EXPECT_TRUE(pl_head.rfind("#EXTM3U", 0) == 0);

  // 90 frames at 30 fps = 3 seconds media, with hls_time=1 the
  // muxer should emit ~3 segments. With playlist_max_size=5 (the
  // default) none should have rotated out yet, so we expect ≥ 2.
  EXPECT_TRUE(ts_segment_count_(bc->blobs()) >= 2);

  // Server was disabled in config.
  EXPECT_TRUE(!bc->server_running());
}

TEST(hls_broadcast_stage, serves_registered_blobs_over_http)
{
  // Same setup as end_to_end, but with serve_http=true and a port=0
  // ephemeral bind. We launch, wait until at least one segment has
  // been registered, then HTTP-GET both the playlist and a
  // segment from the live server to confirm the wire path works.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RepeatTensorSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->tb    = make_tensor_(32, 32, 0.5f);
  // 60 frames at 30 fps = 2 seconds media (1 + spare segment) with
  // hls_time=1. The 20 ms inter-frame pause stretches the wall-clock
  // run to ~1.2 s, giving the test plenty of room to land HTTP
  // requests against the live server before drain tears it down.
  src_u->count       = 60;
  src_u->per_beat_us = 20'000;
  src_u->allocate_oports(1);
  auto* src = static_cast<RepeatTensorSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("codec", FlexData::make_string("libx264"));
  cfg.as_object().insert("preset", FlexData::make_string("ultrafast"));
  cfg.as_object().insert("realtime", FlexData::make_bool(false));
  cfg.as_object().insert("segment_duration", FlexData::make_int(1));
  cfg.as_object().insert("fps_num", FlexData::make_int(30));
  cfg.as_object().insert("fps_den", FlexData::make_int(1));
  cfg.as_object().insert("gop_size", FlexData::make_int(30));
  cfg.as_object().insert("bind_address",
                         FlexData::make_string("127.0.0.1"));
  cfg.as_object().insert("port", FlexData::make_int(0));

  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(
      pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());

  // Poll for "server is up AND at least one segment is in the
  // store". The runtime drains very fast for in-memory output, but
  // there's still a small race between the first segment closing
  // and the io_close callback returning. Cap at ~3 s.
  string resp_playlist;
  string ts_name;
  for (int attempt = 0; attempt < 150; ++attempt) {
    if (bc->server_running()
        && bc->blobs().contains("/stream.m3u8")
        && ts_segment_count_(bc->blobs()) >= 1) {
      // Snapshot the first segment name so we can GET it.
      for (const auto& p : bc->blobs().paths()) {
        if (p.size() >= 3
            && p.compare(p.size() - 3, 3, ".ts") == 0) {
          ts_name = p;
          break;
        }
      }
      resp_playlist = http_get_("127.0.0.1", bc->http_port(),
                                "/stream.m3u8");
      if (resp_playlist.find("HTTP/1.1 200") != string::npos) {
        break;
      }
    }
    std::this_thread::sleep_for(chrono::milliseconds(20));
  }

  EXPECT_TRUE(resp_playlist.find("HTTP/1.1 200") != string::npos);
  EXPECT_TRUE(resp_playlist.find("application/vnd.apple.mpegurl")
              != string::npos);
  EXPECT_TRUE(resp_playlist.find("#EXTM3U") != string::npos);

  if (!ts_name.empty()) {
    string resp_ts = http_get_("127.0.0.1", bc->http_port(), ts_name);
    EXPECT_TRUE(resp_ts.find("HTTP/1.1 200") != string::npos);
    EXPECT_TRUE(resp_ts.find("video/mp2t") != string::npos);
    // MPEG-TS sync byte 0x47 appears at the start of the body.
    auto body = resp_ts.find("\r\n\r\n");
    EXPECT_TRUE(body != string::npos);
    if (body != string::npos && body + 4 < resp_ts.size()) {
      EXPECT_TRUE(static_cast<unsigned char>(resp_ts[body + 4])
                  == 0x47);
    }
  }

  rt.wait_idle();
  rt.stop();
}

TEST(hls_broadcast_stage, accepts_u8_tensor_input) {
  // hls-broadcast must accept dtype=U8 TensorBeats (the fast path
  // from video-to-rgb's output_dtype="u8"). Pinned to libx264 so
  // the test runs on hosts without VideoToolbox.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RepeatTensorSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->tb    = make_u8_tensor_(32, 32, 128);
  src_u->count = 90;
  src_u->allocate_oports(1);
  auto* src = static_cast<RepeatTensorSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("codec", FlexData::make_string("libx264"));
  cfg.as_object().insert("preset", FlexData::make_string("ultrafast"));
  cfg.as_object().insert("realtime", FlexData::make_bool(false));
  cfg.as_object().insert("segment_duration", FlexData::make_int(1));
  cfg.as_object().insert("fps_num", FlexData::make_int(30));
  cfg.as_object().insert("fps_den", FlexData::make_int(1));
  cfg.as_object().insert("gop_size", FlexData::make_int(30));
  cfg.as_object().insert("serve_http", FlexData::make_bool(false));

  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(
      pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(bc->blobs().contains("/stream.m3u8"));
  EXPECT_TRUE(ts_segment_count_(bc->blobs()) >= 2);
}

#if defined(__APPLE__) && defined(__arm64__)
TEST(hls_broadcast_stage, videotoolbox_encoder_writes_mpegts) {
  // Apple-Silicon only: confirm the default codec
  // (h264_videotoolbox) drives the encoder + HLS muxer end-to-end
  // and emits an MPEG-TS segment whose first byte is the 0x47 sync
  // byte. We do NOT pin `codec` -- the test is the canary for the
  // changed default.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RepeatTensorSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->tb    = make_tensor_(64, 64, 0.5f);
  src_u->count = 90;
  src_u->allocate_oports(1);
  auto* src = static_cast<RepeatTensorSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  // Default codec (h264_videotoolbox) -- intentionally not pinned.
  cfg.as_object().insert("realtime", FlexData::make_bool(false));
  cfg.as_object().insert("segment_duration", FlexData::make_int(1));
  cfg.as_object().insert("fps_num", FlexData::make_int(30));
  cfg.as_object().insert("fps_den", FlexData::make_int(1));
  cfg.as_object().insert("gop_size", FlexData::make_int(30));
  cfg.as_object().insert("serve_http", FlexData::make_bool(false));

  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(
      pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(bc->blobs().contains("/stream.m3u8"));

  // Find a .ts entry and verify the MPEG-TS sync byte at offset 0.
  std::string ts_name;
  for (const auto& p : bc->blobs().paths()) {
    if (p.size() >= 3 && p.compare(p.size() - 3, 3, ".ts") == 0) {
      ts_name = p;
      break;
    }
  }
  EXPECT_TRUE(!ts_name.empty());
  if (!ts_name.empty()) {
    auto entry = bc->blobs().get(ts_name);
    EXPECT_TRUE(entry.has_value());
    EXPECT_TRUE(entry->bytes && !entry->bytes->empty());
    EXPECT_TRUE(static_cast<int>((*entry->bytes)[0]) == 0x47);
  }
}
#endif

namespace {
// Emits `count` copies of `base`, each stamped with an incrementing
// timestamp_us sideband (ts0 + i*ts_step_us). Used to exercise the
// A/V timestamp-sync path (which needs monotonically-advancing wall
// clocks that RepeatTensorSource's identical copies can't provide).
class RampTsSource : public TypedStage<RampTsSource> {
public:
  static constexpr const char* kTypeName = "ut-hls-ramp-ts-source";
  using TypedStage::TypedStage;

  TensorBeat base;
  int        count      = 1;
  uint64_t   ts0        = 0;
  uint64_t   ts_step_us = 0;

  Job process(RuntimeContext& ctx) override
  {
    if (_i >= count) {
      ctx.signal_done();
      co_return;
    }
    TensorBeat tb = base;   // deep copy (shape + sideband)
    if (!tb.sideband.is_object()) { tb.sideband = FlexData::make_object(); }
    tb.sideband.as_object().insert_or_assign(
        "timestamp_us",
        FlexData::make_uint(ts0 + static_cast<uint64_t>(_i) * ts_step_us));
    ++_i;
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
  }

private:
  int _i = 0;
};
}  // namespace

TEST(hls_broadcast_stage, audio_on_video_iport0_is_rejected) {
  // iport 0 is STRICTLY the video port. Wiring PCM audio to it (the
  // stage's single input) is a misconfiguration: the audio beats are
  // dropped, nothing is muxed, and no stream is produced. (Audio-only
  // must leave iport 0 disconnected and wire audio to iport 1.)
  CerrSilencer hush;
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto src_u = make_unique<RepeatTensorSource>(
      &sess, "src", vector<InEdge>{}, FlexData::make_object());
  src_u->tb    = make_pcm_tensor_(4800, 48000);   // 0.1 s mono @ 48k
  src_u->count = 40;
  src_u->allocate_oports(1);
  auto* src = static_cast<RepeatTensorSource*>(
      pl->insert_stage(std::move(src_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("realtime", FlexData::make_bool(false));
  cfg.as_object().insert("segment_duration", FlexData::make_int(1));
  cfg.as_object().insert("serve_http", FlexData::make_bool(false));

  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc", vector<InEdge>{{src, 0}}, std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(
      pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(!bc->audio_muxed());
  EXPECT_TRUE(!bc->video_muxed());
  EXPECT_TRUE(!bc->blobs().contains("/stream.m3u8"));
}

TEST(hls_broadcast_stage, audio_only_via_disconnected_video_iport) {
  // Positional roles: iport 0 = video, iport 1 = audio. Leaving iport 0
  // DISCONNECTED (InEdge{nullptr,0}) and wiring audio to iport 1 yields
  // an audio-only broadcast -- the runtime treats the unwired iport as
  // an immediate-EOS input, and resolve_roles_ reads its connectedness.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto a_u = make_unique<RepeatTensorSource>(
      &sess, "asrc", vector<InEdge>{}, FlexData::make_object());
  a_u->tb    = make_pcm_tensor_(4800, 48000);
  a_u->count = 40;                              // 4.0 s
  a_u->allocate_oports(1);
  auto* a = static_cast<RepeatTensorSource*>(
      pl->insert_stage(std::move(a_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("realtime", FlexData::make_bool(false));
  cfg.as_object().insert("segment_duration", FlexData::make_int(1));
  cfg.as_object().insert("serve_http", FlexData::make_bool(false));
  cfg.as_object().insert("audio_buffer_seconds", FlexData::make_real(0.0));

  // iport 0 (video) unwired; iport 1 (audio) = asrc.
  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc",
      vector<InEdge>{ InEdge{nullptr, 0}, InEdge{a, 0} },
      std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(
      pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(bc->audio_muxed());
  EXPECT_TRUE(!bc->video_muxed());
  EXPECT_TRUE(bc->blobs().contains("/stream.m3u8"));
  EXPECT_TRUE(ts_segment_count_(bc->blobs()) >= 2);
}

TEST(hls_broadcast_stage, realtime_audio_only_primes_silence_before_audio) {
  // Audio-only + realtime: the stream must go LIVE with a SILENT track at
  // startup, before the producer emits any PCM, so a viewer can attach early.
  // An idle producer (open but silent) is wired to the audio iport (iport 1),
  // video (iport 0) left unwired; over the ~1.7s it stays open the stage should
  // prime + self-clock silence -> a real playlist + segments with NO audio ever
  // sent. (prime_silence defaults on.)
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto a_u = make_unique<IdleThenCloseSource>(
      &sess, "idle", vector<InEdge>{}, FlexData::make_object());
  a_u->naps = 34;                               // ~1.7s open, zero beats
  a_u->nap_ms = 50;
  a_u->allocate_oports(1);
  auto* a = static_cast<IdleThenCloseSource*>(
      pl->insert_stage(std::move(a_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("realtime", FlexData::make_bool(true));
  cfg.as_object().insert("segment_duration", FlexData::make_int(1));
  cfg.as_object().insert("serve_http", FlexData::make_bool(false));
  cfg.as_object().insert("audio_buffer_seconds", FlexData::make_real(0.2));

  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc",
      vector<InEdge>{ InEdge{nullptr, 0}, InEdge{a, 0} },
      std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(bc->audio_muxed());     // silence broadcast with no real audio
  EXPECT_TRUE(!bc->video_muxed());
  EXPECT_TRUE(bc->blobs().contains("/stream.m3u8"));
  EXPECT_TRUE(ts_segment_count_(bc->blobs()) >= 1);
}

TEST(hls_broadcast_stage, realtime_audio_only_paces_burst_without_loss) {
  // REALTIME audio-only path (the streaming-TTS use case): a producer that
  // bursts audio FASTER than realtime must not be dumped ahead of the wall
  // clock -- pump_audio_ paces the drain and buffers the excess in the FIFO, so
  // the self-clocked cadence / teardown flush plays it all out (no loss). Here
  // a source emits 4 s of PCM as fast as it can with realtime=true; the whole
  // 4 s must still reach the muxer (segments), exercising the paced code path.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto a_u = make_unique<RepeatTensorSource>(
      &sess, "asrc", vector<InEdge>{}, FlexData::make_object());
  a_u->tb    = make_pcm_tensor_(4800, 48000);   // 0.1 s mono @ 48k
  a_u->count = 40;                              // 4.0 s, pushed in a burst
  a_u->allocate_oports(1);
  auto* a = static_cast<RepeatTensorSource*>(
      pl->insert_stage(std::move(a_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("realtime", FlexData::make_bool(true));
  cfg.as_object().insert("segment_duration", FlexData::make_int(1));
  cfg.as_object().insert("serve_http", FlexData::make_bool(false));
  cfg.as_object().insert("audio_buffer_seconds", FlexData::make_real(0.2));

  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc",
      vector<InEdge>{ InEdge{nullptr, 0}, InEdge{a, 0} },
      std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(
      pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(bc->audio_muxed());
  EXPECT_TRUE(!bc->video_muxed());
  EXPECT_TRUE(bc->blobs().contains("/stream.m3u8"));
  // All 4 s reached the output (the burst was paced + buffered, then flushed) --
  // not truncated to a live-edge sliver. ~4 one-second segments expected.
  EXPECT_TRUE(ts_segment_count_(bc->blobs()) >= 3);
}

TEST(hls_broadcast_stage, video_plus_audio_muxes_both_streams) {
  // Video on iport 0 + audio on iport 1 -> one HLS output carrying both
  // streams. libx264 so the test runs without VideoToolbox.
  Session sess;

  auto pl = make_unique<Pipeline>("p", &sess);
  auto v_u = make_unique<RepeatTensorSource>(
      &sess, "vsrc", vector<InEdge>{}, FlexData::make_object());
  v_u->tb    = make_tensor_(32, 32, 0.5f);
  v_u->count = 90;                              // 3.0 s @ 30 fps
  v_u->allocate_oports(1);
  auto* v = static_cast<RepeatTensorSource*>(
      pl->insert_stage(std::move(v_u)));

  auto a_u = make_unique<RepeatTensorSource>(
      &sess, "asrc", vector<InEdge>{}, FlexData::make_object());
  a_u->tb    = make_pcm_tensor_(4800, 48000);   // 0.1 s mono @ 48k
  a_u->count = 30;                              // 3.0 s total
  a_u->allocate_oports(1);
  auto* a = static_cast<RepeatTensorSource*>(
      pl->insert_stage(std::move(a_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("codec", FlexData::make_string("libx264"));
  cfg.as_object().insert("preset", FlexData::make_string("ultrafast"));
  cfg.as_object().insert("realtime", FlexData::make_bool(false));
  cfg.as_object().insert("segment_duration", FlexData::make_int(1));
  cfg.as_object().insert("fps_num", FlexData::make_int(30));
  cfg.as_object().insert("fps_den", FlexData::make_int(1));
  cfg.as_object().insert("gop_size", FlexData::make_int(30));
  cfg.as_object().insert("serve_http", FlexData::make_bool(false));
  cfg.as_object().insert("audio_buffer_seconds", FlexData::make_real(0.0));

  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc", vector<InEdge>{{v, 0}, {a, 0}}, std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(
      pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(bc->video_muxed());    // video stream
  EXPECT_TRUE(bc->audio_muxed());    // audio stream
  EXPECT_TRUE(bc->blobs().contains("/stream.m3u8"));
  EXPECT_TRUE(ts_segment_count_(bc->blobs()) >= 2);
}

TEST(hls_broadcast_stage, av_timestamp_sync_muxes_both) {
  // Both inputs carry monotonically-advancing timestamp_us, so the
  // stage derives PTS for video AND audio from one shared UTC epoch
  // (the ts-sync path, active in realtime mode). Verify it muxes both
  // streams and emits at least one segment.
  Session sess;

  const uint64_t t0 = 1'700'000'000'000ull;   // arbitrary UTC epoch (us)

  auto pl = make_unique<Pipeline>("p", &sess);
  auto v_u = make_unique<RampTsSource>(
      &sess, "vsrc", vector<InEdge>{}, FlexData::make_object());
  v_u->base       = make_tensor_(32, 32, 0.5f);
  v_u->count      = 90;
  v_u->ts0        = t0;
  v_u->ts_step_us = 33'333;                    // ~30 fps
  v_u->allocate_oports(1);
  auto* v = static_cast<RampTsSource*>(
      pl->insert_stage(std::move(v_u)));

  auto a_u = make_unique<RampTsSource>(
      &sess, "asrc", vector<InEdge>{}, FlexData::make_object());
  a_u->base       = make_pcm_tensor_(4800, 48000);
  a_u->count      = 30;
  a_u->ts0        = t0;
  a_u->ts_step_us = 100'000;                   // 0.1 s chunks
  a_u->allocate_oports(1);
  auto* a = static_cast<RampTsSource*>(
      pl->insert_stage(std::move(a_u)));

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("codec", FlexData::make_string("libx264"));
  cfg.as_object().insert("preset", FlexData::make_string("ultrafast"));
  cfg.as_object().insert("realtime", FlexData::make_bool(true));
  cfg.as_object().insert("segment_duration", FlexData::make_int(1));
  cfg.as_object().insert("fps_num", FlexData::make_int(30));
  cfg.as_object().insert("fps_den", FlexData::make_int(1));
  cfg.as_object().insert("gop_size", FlexData::make_int(30));
  cfg.as_object().insert("serve_http", FlexData::make_bool(false));

  auto bc_u = make_unique<HlsBroadcastStage>(
      &sess, "bc", vector<InEdge>{{v, 0}, {a, 0}}, std::move(cfg));
  auto* bc = static_cast<HlsBroadcastStage*>(
      pl->insert_stage(std::move(bc_u)));

  PipelineRuntime rt(pl.get(), &sess);
  EXPECT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(bc->video_muxed());
  EXPECT_TRUE(bc->audio_muxed());
  EXPECT_TRUE(bc->blobs().contains("/stream.m3u8"));
  EXPECT_TRUE(ts_segment_count_(bc->blobs()) >= 1);
}
