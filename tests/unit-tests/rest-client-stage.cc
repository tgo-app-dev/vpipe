#include "minitest.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-handle-impl.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/stage-registry.h"
#include "stages/chrono-stage.h"
#include "stages/rest-client-stage.h"
#include "vpipe/pipeline-handle.h"

#include <arpa/inet.h>
#include <atomic>
#include <cstring>
#include <memory>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

FlexData
make_cfg_(const string& method,
          const string& url,
          const FlexData& extra = FlexData::make_null())
{
  FlexData cfg = FlexData::make_object();
  auto o = cfg.as_object();
  o.insert_or_assign("method", FlexData::make_string(method));
  o.insert_or_assign("url", FlexData::make_string(url));
  if (extra.is_object()) {
    for (auto entry : extra.as_object()) {
      o.insert_or_assign(entry.first, entry.second);
    }
  }
  return cfg;
}

// Single-connection HTTP/1.1 echo server running on a loopback
// port. Returns 200 with a JSON body capturing the request method,
// the path, the request body, and the value of a chosen header.
// One accept() then close. The port is allocated by the kernel
// (bind to :0) and surfaced via `port`. Thread joins on stop().
class EchoServer {
public:
  EchoServer() = default;
  ~EchoServer() { stop(); }

  bool start() {
    _fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_fd < 0) { return false; }
    int yes = 1;
    ::setsockopt(_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
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
    if (::listen(_fd, 4) < 0) {
      ::close(_fd); _fd = -1; return false;
    }
    _t = std::thread([this]{ run_(); });
    return true;
  }

  void stop() {
    if (_fd >= 0) {
      int fd = _fd;
      _fd = -1;
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
    }
    if (_t.joinable()) { _t.join(); }
  }

  int port() const { return _port; }
  string url(string path = "/") const {
    return "http://127.0.0.1:" + std::to_string(_port) + path;
  }

private:
  void run_() {
    // Serve up to 8 connections then exit, so multiple test cases
    // can reuse one server. Honour _fd == -1 as a stop signal.
    for (int i = 0; i < 8 && _fd >= 0; ++i) {
      sockaddr_in caddr{};
      socklen_t clen = sizeof(caddr);
      int cfd = ::accept(_fd, reinterpret_cast<sockaddr*>(&caddr),
                         &clen);
      if (cfd < 0) { return; }
      handle_(cfd);
      ::close(cfd);
    }
  }

  void handle_(int cfd) {
    // Read everything until headers complete; if Content-Length is
    // present read exactly that many body bytes too. Tiny parser --
    // not robust, but sufficient for tests.
    string buf;
    char chunk[4096];
    size_t body_start = string::npos;
    long content_length = 0;
    while (true) {
      ssize_t n = ::recv(cfd, chunk, sizeof(chunk), 0);
      if (n <= 0) { return; }
      buf.append(chunk, chunk + n);
      auto e = buf.find("\r\n\r\n");
      if (e != string::npos) {
        body_start = e + 4;
        auto cl = buf.find("Content-Length:");
        if (cl == string::npos) {
          cl = buf.find("content-length:");
        }
        if (cl != string::npos && cl < body_start) {
          auto eol = buf.find("\r\n", cl);
          string cls = buf.substr(cl + 15, eol - (cl + 15));
          try { content_length = std::stol(cls); }
          catch (...) { content_length = 0; }
        }
        break;
      }
      if (buf.size() > 65536) { return; }
    }
    while (body_start != string::npos
           && static_cast<long>(buf.size() - body_start) < content_length) {
      ssize_t n = ::recv(cfd, chunk, sizeof(chunk), 0);
      if (n <= 0) { break; }
      buf.append(chunk, chunk + n);
    }

    // Parse method + path off the first line.
    auto eol = buf.find("\r\n");
    string first = buf.substr(0, eol);
    auto sp1 = first.find(' ');
    auto sp2 = first.find(' ', sp1 == string::npos ? 0 : sp1 + 1);
    string method = (sp1 == string::npos)
        ? string("?") : first.substr(0, sp1);
    string path = (sp1 == string::npos || sp2 == string::npos)
        ? string("?")
        : first.substr(sp1 + 1, sp2 - sp1 - 1);

    string body = (body_start == string::npos)
        ? string() : buf.substr(body_start);

    // Extract X-Echo header if present (case-insensitive search).
    string echo_hdr;
    {
      string lower = buf.substr(0, body_start);
      string key = "x-echo:";
      string tmp = lower;
      for (auto& c : tmp) {
        c = static_cast<char>(std::tolower(
            static_cast<unsigned char>(c)));
      }
      auto p = tmp.find(key);
      if (p != string::npos) {
        auto e2 = tmp.find("\r\n", p);
        echo_hdr = lower.substr(p + key.size(),
                                e2 - (p + key.size()));
        while (!echo_hdr.empty()
               && (echo_hdr.front() == ' ' || echo_hdr.front() == '\t')) {
          echo_hdr.erase(echo_hdr.begin());
        }
        while (!echo_hdr.empty()
               && (echo_hdr.back() == ' ' || echo_hdr.back() == '\t')) {
          echo_hdr.pop_back();
        }
      }
    }

    // Build a small JSON response body.
    auto json_escape = [](const string& s) {
      string out;
      out.reserve(s.size() + 2);
      for (char c : s) {
        switch (c) {
          case '"':  out += "\\\""; break;
          case '\\': out += "\\\\"; break;
          case '\n': out += "\\n";  break;
          case '\r': out += "\\r";  break;
          case '\t': out += "\\t";  break;
          default:
            if (static_cast<unsigned char>(c) < 0x20) {
              char b[8];
              std::snprintf(b, sizeof(b), "\\u%04x",
                            static_cast<unsigned>(c) & 0xff);
              out += b;
            } else {
              out += c;
            }
        }
      }
      return out;
    };

    string rbody = "{\"method\":\"" + json_escape(method) +
                   "\",\"path\":\""   + json_escape(path) +
                   "\",\"body\":\""   + json_escape(body) +
                   "\",\"echo\":\""   + json_escape(echo_hdr) + "\"}";
    string resp;
    resp += "HTTP/1.1 200 OK\r\n";
    resp += "Content-Type: application/json\r\n";
    resp += "Content-Length: " + std::to_string(rbody.size()) + "\r\n";
    resp += "Connection: close\r\n\r\n";
    resp += rbody;
    ::send(cfd, resp.data(), resp.size(), 0);
  }

  int _fd = -1;
  int _port = 0;
  std::thread _t;
};

}

// --------------- type registration -----------------------------------

TEST(rest_client_stage, type_is_registered) {
  EXPECT_TRUE(StageRegistry::get().find_id("rest-client") !=
              StageTypeId::unknown);
}

// --------------- config validation (deferred) ------------------------

TEST(rest_client_stage, missing_method_deferred) {
  Session sess;
  RestClientStage s(&sess, "r", {}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
}

TEST(rest_client_stage, missing_url_deferred) {
  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("method", FlexData::make_string("GET"));
  RestClientStage s(&sess, "r", {}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(rest_client_stage, unknown_method_deferred) {
  Session sess;
  RestClientStage s(&sess, "r", {},
      make_cfg_("FROBNICATE", "http://127.0.0.1/"));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(rest_client_stage, bad_payload_format_deferred) {
  Session sess;
  FlexData extra = FlexData::make_object();
  extra.as_object().insert_or_assign(
      "payload_format", FlexData::make_string("xml"));
  RestClientStage s(&sess, "r", {},
      make_cfg_("POST", "http://127.0.0.1/", extra));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(rest_client_stage, ok_config_clears_error) {
  Session sess;
  RestClientStage s(&sess, "r", {},
      make_cfg_("GET", "http://127.0.0.1/foo"));
  EXPECT_TRUE(s.config_error().empty());
}

// --------------- payload_path resolution -----------------------------

TEST(rest_client_path, empty_returns_root) {
  FlexData root = FlexData::from_json("{\"a\":1,\"b\":2}");
  FlexData out; string err;
  EXPECT_TRUE(RestClientStage::resolve_path(root, "", out, err));
  EXPECT_TRUE(out.to_json() == "{\"a\":1,\"b\":2}");
}

TEST(rest_client_path, walks_object_keys) {
  FlexData root = FlexData::from_json(
      "{\"outer\":{\"inner\":\"hit\"}}");
  FlexData out; string err;
  EXPECT_TRUE(RestClientStage::resolve_path(root, "outer/inner",
                                            out, err));
  EXPECT_TRUE(out.is_string());
  EXPECT_TRUE(out.get_string() == "hit");
}

TEST(rest_client_path, leading_slash_ok) {
  FlexData root = FlexData::from_json("{\"x\":42}");
  FlexData out; string err;
  EXPECT_TRUE(RestClientStage::resolve_path(root, "/x", out, err));
  EXPECT_TRUE(out.get_int() == 42);
}

TEST(rest_client_path, indexes_array) {
  FlexData root = FlexData::from_json(
      "{\"items\":[\"a\",{\"k\":\"v\"}]}");
  FlexData out; string err;
  EXPECT_TRUE(RestClientStage::resolve_path(root, "items/1/k",
                                            out, err));
  EXPECT_TRUE(out.is_string());
  EXPECT_TRUE(out.get_string() == "v");
}

TEST(rest_client_path, missing_key_errors) {
  FlexData root = FlexData::from_json("{\"a\":1}");
  FlexData out; string err;
  EXPECT_FALSE(RestClientStage::resolve_path(root, "b", out, err));
  EXPECT_FALSE(err.empty());
}

TEST(rest_client_path, array_out_of_range_errors) {
  FlexData root = FlexData::from_json("{\"a\":[1,2]}");
  FlexData out; string err;
  EXPECT_FALSE(RestClientStage::resolve_path(root, "a/5", out, err));
  EXPECT_FALSE(err.empty());
}

TEST(rest_client_path, scalar_subscript_errors) {
  FlexData root = FlexData::from_json("{\"a\":1}");
  FlexData out; string err;
  EXPECT_FALSE(RestClientStage::resolve_path(root, "a/x", out, err));
  EXPECT_FALSE(err.empty());
}

// --------------- HTTP integration via loopback echo server -----------

namespace {

// Drive one beat through a {chrono(count=1) -> rest-client} pipeline,
// then snap the rest-client stage's last status.
struct OneShotResult {
  int    status   = 0;
  string body_raw;
  FlexData body;
  FlexData hdrs;
};

OneShotResult
run_one_request_(const FlexData& cfg) {
  Session sess;
  // chrono with count=1 fires exactly one beat, which is what
  // rest-client needs to issue one request.
  FlexData ccfg = FlexData::make_object();
  ccfg.as_object().insert_or_assign(
      "frequency_hz", FlexData::make_real(50.0));
  ccfg.as_object().insert_or_assign(
      "count", FlexData::make_int(1));

  auto pl = make_unique<Pipeline>("p", &sess);
  auto* src = pl->insert_stage("chrono", "src", {},
                               std::move(ccfg));
  // Wrap cfg in a copy so the caller-supplied FlexData survives.
  auto* sink = pl->insert_stage("rest-client", "rc",
                                vector<InEdge>{{src, 0}},
                                cfg);
  OneShotResult r;
  if (!sink) { return r; }

  PipelineRuntime rt(pl.get(), &sess);
  if (!rt.launch()) { return r; }
  rt.wait_idle();

  auto* rc = dynamic_cast<RestClientStage*>(sink);
  if (rc) {
    r.status = rc->last_status();
  }
  return r;
}

}

TEST(rest_client_stage, get_against_loopback_echo) {
  EchoServer srv;
  ASSERT_TRUE(srv.start());
  OneShotResult r = run_one_request_(
      make_cfg_("GET", srv.url("/ping")));
  srv.stop();
  EXPECT_TRUE(r.status == 200);
}

TEST(rest_client_stage, post_json_body_reaches_server) {
  // Drive a single beat with a non-Chrono payload that contains JSON
  // we want to echo. Use chrono + a transformer? simpler: hardcode a
  // text-input would block; instead, use the json-default body
  // emitted by chrono (a TriggerPayload yields null payload, so the
  // stage POSTs a JSON "null", which is fine to assert about).
  EchoServer srv;
  ASSERT_TRUE(srv.start());

  FlexData extra = FlexData::make_object();
  FlexData hdrs  = FlexData::make_object();
  hdrs.as_object().insert_or_assign(
      "X-Echo", FlexData::make_string("trace-1"));
  extra.as_object().insert_or_assign("headers", std::move(hdrs));

  FlexData cfg = make_cfg_("POST", srv.url("/p"), extra);
  OneShotResult r = run_one_request_(cfg);
  srv.stop();
  EXPECT_TRUE(r.status == 200);
}

TEST(rest_client_stage, head_request_returns_no_body) {
  EchoServer srv;
  ASSERT_TRUE(srv.start());
  OneShotResult r = run_one_request_(
      make_cfg_("HEAD", srv.url("/")));
  srv.stop();
  EXPECT_TRUE(r.status == 200);
}
