#include "apps/web-ui/log-api.h"
#include "apps/web-ui/api-common.h"
#include "apps/web-ui/web-ui-log-delegate.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "vpipe/session-intf.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe::webui {

HttpResponse
LogApi::h_console_(const HttpRequest& req)
{
  if (!_ctx.log) {
    return HttpResponse::error(404, "session log not available");
  }
  uint64_t since = 0;
  string sv = query_param(req.query, "since");
  if (!sv.empty()) {
    since = strtoull(sv.c_str(), nullptr, 10);
  }
  std::size_t limit = 2048;
  string lv = query_param(req.query, "limit");
  if (!lv.empty()) {
    limit = static_cast<std::size_t>(strtoull(lv.c_str(), nullptr, 10));
  }
  uint64_t latest = 0;
  auto lines = _ctx.log->console_since(since, &latest, limit);

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("latest", FlexData::make_uint(latest));
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (const auto& l : lines) {
    FlexData e = FlexData::make_object();
    auto eo = e.as_object();
    eo.insert("seq", FlexData::make_uint(l.seq));
    eo.insert("level", fstr(l.level));
    eo.insert("text", fstr(l.text));
    a.push_back(std::move(e));
  }
  oo.insert("lines", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
LogApi::h_clear_(const HttpRequest&)
{
  if (!_ctx.log) {
    return HttpResponse::error(404, "session log not available");
  }
  _ctx.log->clear_console();
  return HttpResponse::ok();
}

namespace {
// Debug-level choices the UI dropdown offers (most severe first). The
// "always" sentinel is not a valid threshold and is omitted.
const char* const kLogLevels[] = {
    "error", "warn", "info", "normal", "verbose", "debug" };
}  // namespace

HttpResponse
LogApi::h_level_get_(const HttpRequest&)
{
  if (!_ctx.log) {
    return HttpResponse::error(404, "session log not available");
  }
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  string cur = to_cstr(_ctx.log->threshold());
  for (char& c : cur) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  oo.insert("level", fstr(cur));
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (const char* lv : kLogLevels) { a.push_back(fstr(lv)); }
  oo.insert("levels", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
LogApi::h_level_set_(const HttpRequest& req)
{
  if (!_ctx.log) {
    return HttpResponse::error(404, "session log not available");
  }
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "expected object {level: NAME}");
  }
  auto bo = body->as_object();
  if (!bo.contains("level")) {
    return HttpResponse::error(400, "missing 'level'");
  }
  string name = string(bo.at("level").as_string(""));
  // "always" is a sentinel, not a threshold -- reject it explicitly so
  // a stray value can't silently disable filtering.
  const LogLevel sentinel = LogLevel::Always;
  const LogLevel parsed = parse_log_level(name, sentinel);
  if (parsed == sentinel) {
    return HttpResponse::error(400, "unknown log level '" + name + "'");
  }
  // Set the threshold directly on the delegate (not via
  // Session::debug_level, which refuses while a pipeline is launched):
  // set_threshold is atomic and MT-safe, and only future messages are
  // affected -- already-captured lines stay.
  _ctx.log->set_threshold(parsed);
  FlexData o = FlexData::make_object();
  string cur = to_cstr(parsed);
  for (char& c : cur) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  o.as_object().insert("level", fstr(cur));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
LogApi::h_limit_get_(const HttpRequest&)
{
  if (!_ctx.log) {
    return HttpResponse::error(404, "session log not available");
  }
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("max_log",
      FlexData::make_uint(static_cast<uint64_t>(_ctx.log->max_log())));
  oo.insert("min", FlexData::make_uint(
      static_cast<uint64_t>(webui::WebUiLogDelegate::kMinMaxLog)));
  oo.insert("max", FlexData::make_uint(
      static_cast<uint64_t>(webui::WebUiLogDelegate::kMaxMaxLog)));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
LogApi::h_limit_set_(const HttpRequest& req)
{
  if (!_ctx.log) {
    return HttpResponse::error(404, "session log not available");
  }
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "expected object {max_log: N}");
  }
  auto bo = body->as_object();
  if (!bo.contains("max_log")) {
    return HttpResponse::error(400, "missing 'max_log'");
  }
  const uint64_t n = bo.at("max_log").as_uint(0);
  if (n == 0) {
    return HttpResponse::error(400, "'max_log' must be > 0");
  }
  _ctx.log->set_max_log(static_cast<size_t>(n));
  FlexData o = FlexData::make_object();
  o.as_object().insert("max_log",
      FlexData::make_uint(static_cast<uint64_t>(_ctx.log->max_log())));
  return HttpResponse::json(200, o.to_json());
}

void
LogApi::register_routes(HttpServer& s)
{
  // Registered only when a WebUiLogDelegate backs the session.
  if (!_ctx.log) { return; }
  s.route("GET", "/api/log/console",
          [this](const HttpRequest& r) { return h_console_(r); });
  s.route("POST", "/api/log/clear",
          [this](const HttpRequest& r) { return h_clear_(r); });
  s.route("GET", "/api/log/level",
          [this](const HttpRequest& r) { return h_level_get_(r); });
  s.route("PUT", "/api/log/level",
          [this](const HttpRequest& r) { return h_level_set_(r); });
  s.route("GET", "/api/log/limit",
          [this](const HttpRequest& r) { return h_limit_get_(r); });
  s.route("PUT", "/api/log/limit",
          [this](const HttpRequest& r) { return h_limit_set_(r); });
}

}
