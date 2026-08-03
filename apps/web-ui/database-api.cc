#include "apps/web-ui/database-api.h"
#include "apps/web-ui/api-common.h"
#include "apps/web-ui/db-browser.h"
#include "apps/web-ui/model-browser.h"

#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe::webui {


HttpResponse
DatabaseApi::h_list_(const HttpRequest&)
{
  DbBrowser db(_ctx.sctx);
  string err;
  FlexData doc;
  {
    lock_guard<mutex> dlk(_db_mu);
    doc = db.list_databases(err);
  }
  if (!err.empty()) { return HttpResponse::error(500, err); }
  // Entries / databases may be dropped only while every pipeline is
  // stopped (a running stage could be writing). Surface that so the
  // view shows or hides its delete controls accordingly.
  bool deletable;
  {
    lock_guard<mutex> lk(_ctx.mu);
    deletable = !_pipelines.any_active_locked();
  }
  if (doc.is_object()) {
    doc.as_object().insert("deletable", FlexData::make_bool(deletable));
  }
  return HttpResponse::json(200, doc.to_json());
}

HttpResponse
DatabaseApi::h_delete_key_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  // Hold _ctx.mu across the check + write so a concurrent launch can't slip
  // in between (launch also takes _ctx.mu).
  lock_guard<mutex> lk(_ctx.mu);
  if (_pipelines.any_active_locked()) {
    return HttpResponse::error(
        409, "stop all pipelines before modifying the database");
  }
  DbBrowser db(_ctx.sctx);
  string err;
  FlexData doc;
  {
    lock_guard<mutex> dlk(_db_mu);
    doc = db.delete_key(*body, err);
  }
  if (!err.empty()) { return HttpResponse::error(400, err); }
  return HttpResponse::json(200, doc.to_json());
}

HttpResponse
DatabaseApi::h_drop_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  lock_guard<mutex> lk(_ctx.mu);
  if (_pipelines.any_active_locked()) {
    return HttpResponse::error(
        409, "stop all pipelines before modifying the database");
  }
  DbBrowser db(_ctx.sctx);
  string err;
  FlexData doc;
  {
    lock_guard<mutex> dlk(_db_mu);
    doc = db.drop_database(*body, err);
  }
  if (!err.empty()) { return HttpResponse::error(400, err); }
  return HttpResponse::json(200, doc.to_json());
}

HttpResponse
DatabaseApi::h_keys_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  DbBrowser db(_ctx.sctx);
  string err;
  FlexData doc;
  {
    lock_guard<mutex> dlk(_db_mu);
    doc = db.query_keys(*body, err);
  }
  if (!err.empty()) { return HttpResponse::error(400, err); }
  return HttpResponse::json(200, doc.to_json());
}

void
DatabaseApi::h_scan_stream_(const HttpRequest& req, ResponseStream& rs)
{
  rs.begin(200, "application/x-ndjson");
  // Coalesce records into a buffer flushed every ~32 KB (and once at the
  // end) so a 64k-row scan doesn't cost one send() per row while still
  // reaching the client in small, timely chunks.
  string buf;
  buf.reserve(1u << 16);
  bool alive = true;
  auto flush = [&]() -> bool {
    if (!buf.empty()) { alive = rs.write(buf); buf.clear(); }
    return alive;
  };
  auto send_line = [&](const string& line) -> bool {
    if (!alive) { return false; }
    buf += line;
    buf.push_back('\n');
    if (buf.size() >= (1u << 15)) { return flush(); }
    return alive;
  };
  auto emit_err = [&](const string& msg) {
    FlexData e = FlexData::make_object();
    auto eo = e.as_object();
    eo.insert("t", FlexData::make_string("error"));
    eo.insert("error", FlexData::make_string(msg));
    send_line(e.to_json());
    flush();
  };

  auto body = parse_json_body(req);
  if (!body) { emit_err("invalid JSON body"); return; }

  DbBrowser db(_ctx.sctx);
  string err;
  DbBrowser::ScanStats stats;
  {
    lock_guard<mutex> dlk(_db_mu);
    db.stream_scan(
        *body,
        [&](const FlexData& meta) { return send_line(meta.to_json()); },
        [&](const FlexData& row) { return send_line(row.to_json()); },
        stats, err);
  }
  if (!err.empty()) { emit_err(err); return; }

  FlexData d = FlexData::make_object();
  auto dd = d.as_object();
  dd.insert("t", FlexData::make_string("done"));
  dd.insert("total", FlexData::make_uint(static_cast<uint64_t>(stats.matched)));
  dd.insert("scanned",
            FlexData::make_uint(static_cast<uint64_t>(stats.scanned)));
  dd.insert("truncated", FlexData::make_bool(stats.truncated));
  dd.insert("aborted", FlexData::make_bool(stats.aborted));
  send_line(d.to_json());
  flush();
}


HttpResponse
DatabaseApi::h_value_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  DbBrowser db(_ctx.sctx);
  string err;
  FlexData doc;
  {
    lock_guard<mutex> dlk(_db_mu);
    doc = db.read_value(*body, err);
  }
  if (!err.empty()) { return HttpResponse::error(400, err); }
  return HttpResponse::json(200, doc.to_json());
}

HttpResponse
DatabaseApi::h_models_installed_(const HttpRequest&)
{
  string err;
  FlexData doc;
  {
    lock_guard<mutex> dlk(_db_mu);
    doc = list_installed_models(_ctx.sctx, err);
  }
  if (!err.empty()) { return HttpResponse::error(500, err); }
  return HttpResponse::json(200, doc.to_json());
}

void
DatabaseApi::register_routes(HttpServer& s)
{
  s.route("GET", "/api/db/list",
          [this](const HttpRequest& r) { return h_list_(r); });
  s.route("POST", "/api/db/keys",
          [this](const HttpRequest& r) { return h_keys_(r); });
  s.route_stream("POST", "/api/db/scan",
                 [this](const HttpRequest& r, ResponseStream& rs) {
                   h_scan_stream_(r, rs);
                 });
  s.route("POST", "/api/db/value",
          [this](const HttpRequest& r) { return h_value_(r); });
  s.route("POST", "/api/db/delete-key",
          [this](const HttpRequest& r) { return h_delete_key_(r); });
  s.route("POST", "/api/db/drop",
          [this](const HttpRequest& r) { return h_drop_(r); });
  s.route("GET", "/api/models/installed",
          [this](const HttpRequest& r) { return h_models_installed_(r); });
}

}
