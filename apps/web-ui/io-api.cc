#include "apps/web-ui/io-api.h"
#include "apps/web-ui/api-common.h"
#include "apps/web-ui/web-ui-delegate.h"

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
IoApi::h_console_(const HttpRequest& req)
{
  if (!_ctx.ui) { return HttpResponse::error(404, "user I/O not available"); }
  uint64_t since = 0;
  string sv = query_param(req.query, "since");
  if (!sv.empty()) {
    since = strtoull(sv.c_str(), nullptr, 10);
  }
  // Cap the per-response line count. The default keeps the very
  // first poll after a tab switch from delivering 8 k+ lines at once
  // (which then takes the browser seconds to render synchronously).
  // The client preserves `latest` between polls, so a long backlog
  // catches up over a handful of ticks instead of one giant pause.
  std::size_t limit = 2048;
  string lv = query_param(req.query, "limit");
  if (!lv.empty()) {
    limit = static_cast<std::size_t>(strtoull(lv.c_str(), nullptr, 10));
  }
  uint64_t latest = 0;
  auto lines = _ctx.ui->console_since(since, &latest, limit);

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
    eo.insert("open", FlexData::make_bool(l.open));
    a.push_back(std::move(e));
  }
  oo.insert("lines", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

// Every live progress report. Reads the registry on UiDelegateIntf
// directly -- progress is delegate-agnostic state, so WebUiDelegate has
// nothing to implement here; only the RENDERING is web-specific.
//
// `version` lets the client skip a re-render when nothing moved, the
// same trick the console footer uses to avoid repainting at 10 Hz
// against work that ticks once a second.
HttpResponse
IoApi::h_progress_(const HttpRequest&)
{
  if (!_ctx.ui) { return HttpResponse::error(404, "user I/O not available"); }
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("version", FlexData::make_uint(_ctx.ui->progress_version()));
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (const auto& it : _ctx.ui->progress_snapshot()) {
    FlexData e = FlexData::make_object();
    auto eo = e.as_object();
    eo.insert("id", FlexData::make_uint(it.id));
    eo.insert("desc", fstr(it.desc));
    eo.insert("done", FlexData::make_uint(it.done));
    // 0 total means INDETERMINATE; the client draws motion, not a fill.
    eo.insert("total", FlexData::make_uint(it.total));
    eo.insert("detail", fstr(it.detail));
    eo.insert("seq", FlexData::make_uint(it.seq));
    a.push_back(std::move(e));
  }
  oo.insert("items", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
IoApi::h_pending_(const HttpRequest&)
{
  if (!_ctx.ui) { return HttpResponse::error(404, "user I/O not available"); }
  uint64_t id = 0;
  string   prompt;
  bool     masked  = false;
  bool     media   = false;
  bool     pending = _ctx.ui->pending_input(&id, &prompt, &masked, &media);

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("pending", FlexData::make_bool(pending));
  if (pending) {
    oo.insert("id", FlexData::make_uint(id));
    oo.insert("prompt", fstr(prompt));
    oo.insert("masked", FlexData::make_bool(masked));
    // getmedialine request: the client offers attach/drop controls and
    // embeds attachments as base64 media-line markers in the answer.
    oo.insert("media", FlexData::make_bool(media));
  }
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
IoApi::h_input_(const HttpRequest& req)
{
  if (!_ctx.ui) { return HttpResponse::error(404, "user I/O not available"); }
  auto body = parse_json_body(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  auto bo = body->as_object();
  if (!bo.contains("id")) {
    return HttpResponse::error(400, "missing 'id'");
  }
  uint64_t id   = bo.at("id").as_uint(0);
  string   text = string(bo.contains("text")
                             ? bo.at("text").as_string("")
                             : string_view{});
  if (!_ctx.ui->submit_input(id, std::move(text))) {
    return HttpResponse::error(409, "no matching pending input request");
  }
  return HttpResponse::ok();
}

HttpResponse
IoApi::h_clear_(const HttpRequest&)
{
  if (!_ctx.ui) { return HttpResponse::error(404, "user I/O not available"); }
  _ctx.ui->clear_console();
  return HttpResponse::ok();
}

HttpResponse
IoApi::h_interrupt_(const HttpRequest&)
{
  if (!_ctx.ui) { return HttpResponse::error(404, "user I/O not available"); }
  // Fire every stage-registered interrupt handler (see
  // UiDelegateIntf::register_interrupt_handler). Always a 200: the
  // button is unconditional, so "nothing was running" is a normal
  // outcome, not an error -- `handled` reports how many stages
  // actually cut work short.
  const int handled = _ctx.ui->dispatch_interrupt();
  FlexData o  = FlexData::make_object();
  auto     oo = o.as_object();
  oo.insert("handled", FlexData::make_int(handled));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
IoApi::h_limit_get_(const HttpRequest&)
{
  if (!_ctx.ui) { return HttpResponse::error(404, "user I/O not available"); }
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("max_console",
      FlexData::make_uint(static_cast<uint64_t>(_ctx.ui->max_console())));
  oo.insert("min", FlexData::make_uint(
      static_cast<uint64_t>(webui::WebUiDelegate::kMinMaxConsole)));
  oo.insert("max", FlexData::make_uint(
      static_cast<uint64_t>(webui::WebUiDelegate::kMaxMaxConsole)));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
IoApi::h_limit_set_(const HttpRequest& req)
{
  if (!_ctx.ui) { return HttpResponse::error(404, "user I/O not available"); }
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "expected object {max_console: N}");
  }
  auto bo = body->as_object();
  if (!bo.contains("max_console")) {
    return HttpResponse::error(400, "missing 'max_console'");
  }
  const uint64_t n = bo.at("max_console").as_uint(0);
  if (n == 0) {
    return HttpResponse::error(400, "'max_console' must be > 0");
  }
  _ctx.ui->set_max_console(static_cast<size_t>(n));
  FlexData o = FlexData::make_object();
  o.as_object().insert("max_console",
      FlexData::make_uint(static_cast<uint64_t>(_ctx.ui->max_console())));
  return HttpResponse::json(200, o.to_json());
}

void
IoApi::register_routes(HttpServer& s)
{
  // Registered only when a WebUiDelegate backs the session (otherwise
  // the routes 404).
  if (!_ctx.ui) { return; }
  s.route("GET", "/api/io/console",
          [this](const HttpRequest& r) { return h_console_(r); });
  s.route("GET", "/api/io/progress",
          [this](const HttpRequest& r) { return h_progress_(r); });
  s.route("GET", "/api/io/pending",
          [this](const HttpRequest& r) { return h_pending_(r); });
  s.route("POST", "/api/io/input",
          [this](const HttpRequest& r) { return h_input_(r); });
  s.route("POST", "/api/io/clear",
          [this](const HttpRequest& r) { return h_clear_(r); });
  s.route("POST", "/api/io/interrupt",
          [this](const HttpRequest& r) { return h_interrupt_(r); });
  s.route("GET", "/api/io/limit",
          [this](const HttpRequest& r) { return h_limit_get_(r); });
  s.route("PUT", "/api/io/limit",
          [this](const HttpRequest& r) { return h_limit_set_(r); });
}

}
