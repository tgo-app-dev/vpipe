#include "common/mini-http.h"
#include "common/host-net.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace vpipe::http {

namespace {

std::string
ascii_lower(std::string_view s)
{
  std::string out(s);
  for (auto& c : out) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return out;
}

std::string_view
trim(std::string_view s)
{
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) { ++a; }
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) { --b; }
  return s.substr(a, b - a);
}

// Split "host:port" into (host, port-or-defaulted).
std::pair<std::string, uint16_t>
split_host_port(std::string_view host_port, uint16_t default_port)
{
  auto colon = host_port.find(':');
  if (colon == std::string_view::npos) {
    return {std::string(host_port), default_port};
  }
  std::string host(host_port.substr(0, colon));
  uint16_t    port = default_port;
  auto        ps   = host_port.substr(colon + 1);
  unsigned    p    = 0;
  auto        r    = std::from_chars(ps.data(), ps.data() + ps.size(), p);
  if (r.ec == std::errc{} && p > 0 && p <= 65535) {
    port = static_cast<uint16_t>(p);
  }
  return {std::move(host), port};
}

}

Response
parse(std::string_view raw)
{
  Response r{};
  // Split headers / body at CRLFCRLF (fall back to LFLF for sloppy
  // servers; some Hikvision firmware does this).
  size_t sep = raw.find("\r\n\r\n");
  size_t sep_len = 4;
  if (sep == std::string_view::npos) {
    sep = raw.find("\n\n");
    sep_len = 2;
  }
  if (sep == std::string_view::npos) {
    return r;  // status 0
  }

  std::string_view head = raw.substr(0, sep);
  std::string_view body = raw.substr(sep + sep_len);

  // First line: "HTTP/1.1 <status> <reason>"
  size_t eol = head.find('\n');
  std::string_view status_line =
      (eol == std::string_view::npos) ? head : head.substr(0, eol);
  while (!status_line.empty() &&
         (status_line.back() == '\r' || status_line.back() == ' ')) {
    status_line.remove_suffix(1);
  }
  // Skip "HTTP/1.x "
  auto sp1 = status_line.find(' ');
  if (sp1 == std::string_view::npos) {
    return r;
  }
  std::string_view rest = status_line.substr(sp1 + 1);
  auto             sp2  = rest.find(' ');
  std::string_view code =
      (sp2 == std::string_view::npos) ? rest : rest.substr(0, sp2);
  unsigned status_u = 0;
  auto sc = std::from_chars(code.data(), code.data() + code.size(),
                            status_u);
  if (sc.ec != std::errc{}) {
    return r;
  }
  r.status = static_cast<int>(status_u);

  // Header lines.
  size_t pos = (eol == std::string_view::npos) ? head.size()
                                               : eol + 1;
  while (pos < head.size()) {
    size_t nl = head.find('\n', pos);
    std::string_view line = (nl == std::string_view::npos)
                            ? head.substr(pos)
                            : head.substr(pos, nl - pos);
    while (!line.empty() &&
           (line.back() == '\r' || line.back() == ' ')) {
      line.remove_suffix(1);
    }
    if (!line.empty()) {
      auto c = line.find(':');
      if (c != std::string_view::npos) {
        std::string k = ascii_lower(trim(line.substr(0, c)));
        std::string v(trim(line.substr(c + 1)));
        if (k == "transfer-encoding" && ascii_lower(v).find("chunked")
            != std::string::npos) {
          throw std::runtime_error(
              "mini-http: chunked transfer-encoding not supported");
        }
        r.headers.emplace_back(std::move(k), std::move(v));
      }
    }
    if (nl == std::string_view::npos) { break; }
    pos = nl + 1;
  }

  r.body.assign(body);
  return r;
}

Response
post(std::string_view host_port,
     std::string_view path,
     std::string_view content_type,
     std::string_view body,
     std::chrono::milliseconds timeout)
{
  auto [host, port] = split_host_port(host_port, 80);

  std::string req;
  req.reserve(256 + body.size());
  req.append("POST ");
  req.append(path);
  req.append(" HTTP/1.1\r\nHost: ");
  req.append(host_port);
  req.append("\r\nUser-Agent: vpipe/0.1\r\nAccept: */*\r\n"
             "Connection: close\r\nContent-Type: ");
  req.append(content_type);
  req.append("\r\nContent-Length: ");
  char clen[24];
  std::snprintf(clen, sizeof(clen), "%zu", body.size());
  req.append(clen);
  req.append("\r\n\r\n");
  req.append(body);

  std::string raw = netx::tcp_request(host, port, req, timeout);
  if (raw.empty()) {
    return Response{};
  }
  return parse(raw);
}

}
