#ifndef MINI_HTTP_H
#define MINI_HTTP_H

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vpipe::http {

struct Response {
  int                                              status = 0;
  std::vector<std::pair<std::string, std::string>> headers;
  std::string                                      body;
};

// POST `body` to host:port/path. `host_port` is either "host" (port
// defaulted by the caller -- ONVIF cameras almost always answer on
// 80) or "host:port".
//
// Performs HTTP/1.1 with `Connection: close`. Requires the server
// to send `Content-Length` (or no body, with Connection: close);
// throws std::runtime_error if `Transfer-Encoding: chunked` is
// detected -- ONVIF cameras don't use it on the LAN.
Response
post(std::string_view host_port,
     std::string_view path,
     std::string_view content_type,
     std::string_view body,
     std::chrono::milliseconds timeout);

// Pure response parser used by post() and exposed for unit tests.
// `raw` is the literal bytes received from the server (headers +
// CRLFCRLF + body).
Response
parse(std::string_view raw);

}

#endif
