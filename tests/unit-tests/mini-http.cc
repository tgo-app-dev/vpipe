#include "minitest.h"
#include "common/mini-http.h"

#include <exception>
#include <string>

using namespace vpipe;

TEST(mini_http, parse_status_and_headers) {
  std::string raw =
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: application/soap+xml; charset=utf-8\r\n"
      "Content-Length: 5\r\n"
      "Connection: close\r\n"
      "\r\n"
      "hello";
  auto r = http::parse(raw);
  EXPECT_TRUE(r.status == 200);
  EXPECT_TRUE(r.body == "hello");
  bool has_ct = false, has_cl = false;
  for (const auto& kv : r.headers) {
    if (kv.first == "content-type") { has_ct = true; }
    if (kv.first == "content-length") { has_cl = true; }
  }
  EXPECT_TRUE(has_ct);
  EXPECT_TRUE(has_cl);
}

TEST(mini_http, parse_401) {
  std::string raw =
      "HTTP/1.1 401 Unauthorized\r\n"
      "WWW-Authenticate: Digest realm=\"x\"\r\n"
      "Content-Length: 0\r\n"
      "\r\n";
  auto r = http::parse(raw);
  EXPECT_TRUE(r.status == 401);
  EXPECT_TRUE(r.body.empty());
}

TEST(mini_http, parse_rejects_chunked) {
  std::string raw =
      "HTTP/1.1 200 OK\r\n"
      "Transfer-Encoding: chunked\r\n"
      "\r\n";
  bool threw = false;
  try {
    auto r = http::parse(raw);
    (void)r;
  } catch (std::exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(mini_http, parse_lflf_fallback) {
  std::string raw =
      "HTTP/1.1 200 OK\n"
      "Content-Length: 2\n"
      "\n"
      "OK";
  auto r = http::parse(raw);
  EXPECT_TRUE(r.status == 200);
  EXPECT_TRUE(r.body == "OK");
}
