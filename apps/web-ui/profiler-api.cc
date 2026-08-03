#include "apps/web-ui/profiler-api.h"
#include "apps/web-ui/api-common.h"

#include "common/temp-root.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "vpipe/session-intf.h"

#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <utility>

using namespace std;

namespace vpipe::webui {

std::string
ProfilerApi::dump_json_(std::string& err)
{
  namespace fs = std::filesystem;
  const fs::path dir = vpipe::temp_root();
  // One temp file per process; _mu serializes profiler ops so reuse is
  // safe. ".json" extension selects dump_profiling's pretty-JSON form.
  fs::path tmp = dir / ("vpipe-webui-profiler-"
                        + std::to_string(static_cast<long>(::getpid()))
                        + ".json");
  Status st = _ctx.session->dump_profiling(tmp.string());
  if (st.code != 0) {
    err = "dump_profiling failed";
    return {};
  }
  std::ifstream in(tmp.string(), std::ios::binary);
  if (!in) {
    err = "could not read profiling dump";
    return {};
  }
  std::string out((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
  in.close();
  std::error_code rmec;
  fs::remove(tmp, rmec);            // best-effort cleanup
  return out;
}

FlexData
ProfilerApi::status_doc_() const
{
  const bool en = _ctx.sctx && _ctx.sctx->profiling_enabled();
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("enabled", FlexData::make_bool(en));
  oo.insert("max_events_per_thread",
            FlexData::make_uint(
                _ctx.sctx ? _ctx.sctx->profiling_max_events_per_thread() : 0u));
  oo.insert("has_data",
            FlexData::make_bool(en || !_snapshot.empty()));
  return o;
}

HttpResponse
ProfilerApi::h_start_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  unsigned max_events = 65536;
  if (body->is_object() && body->as_object().contains("max_events")) {
    const long n =
        static_cast<long>(body->as_object().at("max_events").as_int(0));
    if (n > 0) {
      max_events = static_cast<unsigned>(std::min<long>(n, 1L << 24));
    }
  }
  lock_guard<mutex> lk(_ctx.mu);
  Status st = _ctx.session->enable_profiling(max_events);
  if (st.code != 0) {
    return HttpResponse::error(
        500, "enable_profiling failed (max_events must be > 0)");
  }
  _snapshot.clear();       // previous capture is now stale
  return HttpResponse::json(200, status_doc_().to_json());
}

HttpResponse
ProfilerApi::h_stop_(const HttpRequest&)
{
  lock_guard<mutex> lk(_ctx.mu);
  if (_ctx.sctx && _ctx.sctx->profiling_enabled()) {
    // Snapshot BEFORE disabling -- disable_profiling frees the buffers.
    std::string err;
    std::string js = dump_json_(err);
    if (!js.empty()) { _snapshot = std::move(js); }
    _ctx.session->disable_profiling();
  }
  return HttpResponse::json(200, status_doc_().to_json());
}

HttpResponse
ProfilerApi::h_reset_(const HttpRequest&)
{
  lock_guard<mutex> lk(_ctx.mu);
  // Drop the retained snapshot (the stopped-state data). If a capture
  // is live, re-arm it: enable_profiling reallocates the buffers, so the
  // accumulated events are cleared and the anchor restarts -- a clean
  // slate without having to Stop first.
  _snapshot.clear();
  if (_ctx.sctx && _ctx.sctx->profiling_enabled()) {
    _ctx.session->enable_profiling(
        _ctx.sctx->profiling_max_events_per_thread());
  }
  return HttpResponse::json(200, status_doc_().to_json());
}

HttpResponse
ProfilerApi::h_status_(const HttpRequest&)
{
  lock_guard<mutex> lk(_ctx.mu);
  return HttpResponse::json(200, status_doc_().to_json());
}

HttpResponse
ProfilerApi::h_data_(const HttpRequest&)
{
  lock_guard<mutex> lk(_ctx.mu);
  if (_ctx.sctx && _ctx.sctx->profiling_enabled()) {
    std::string err;
    std::string js = dump_json_(err);
    if (js.empty()) {
      return HttpResponse::error(500, err.empty() ? "dump failed" : err);
    }
    return HttpResponse::json(200, js);
  }
  if (!_snapshot.empty()) {
    return HttpResponse::json(200, _snapshot);
  }
  // Nothing captured yet: a well-formed empty document.
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("threads", FlexData::make_array());
  oo.insert("stages", FlexData::make_array());
  oo.insert("enabled", FlexData::make_bool(false));
  oo.insert("num_workers", FlexData::make_uint(0u));
  return HttpResponse::json(200, o.to_json());
}

void
ProfilerApi::register_routes(HttpServer& s)
{
  s.route("POST", "/api/profiler/start",
          [this](const HttpRequest& r) { return h_start_(r); });
  s.route("POST", "/api/profiler/stop",
          [this](const HttpRequest& r) { return h_stop_(r); });
  s.route("POST", "/api/profiler/reset",
          [this](const HttpRequest& r) { return h_reset_(r); });
  s.route("GET", "/api/profiler/status",
          [this](const HttpRequest& r) { return h_status_(r); });
  s.route("GET", "/api/profiler/data",
          [this](const HttpRequest& r) { return h_data_(r); });
}

}
