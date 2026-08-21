#include "apps/web-ui/system-api.h"
#include "apps/web-ui/api-common.h"
#include "apps/web-ui/system-status.h"

#include "common/host-net.h"
#include "common/i18n.h"
#include "common/vpipe-format.h"
// NOT under VPIPE_BUILD_APPLE_SILICON. That macro is PRIVATE to the
// `vpipe` library target, so predicating on it here compiles the block
// away silently in this one -- which read as a working endpoint
// reporting zeros. services()->generative_model_manager() already
// answers nullptr where there is no manager, so the null check IS the
// portability check and there is nothing for an ifdef to add.
#include "generative-models/generative-model-manager.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "pipeline/pipeline.h"
#include "pipeline/stage.h"
#include "vpipe/session-intf.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe::webui {

SystemApi::SystemApi(ApiContext& ctx, PipelineApi& pipelines)
  : _ctx(ctx)
  , _pipelines(pipelines)
  , _status(std::make_unique<SystemStatusPoller>())
{
}

// Out-of-line so the unique_ptr<SystemStatusPoller> dtor can see the
// complete type (the pImpl-with-unique_ptr incomplete-type pitfall).
SystemApi::~SystemApi() = default;

void
SystemApi::set_startup_checks(const std::vector<PermissionCheck>& checks)
{
  FlexData doc = FlexData::make_object();
  auto o = doc.as_object();
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  bool any_warn = false;
  for (const auto& c : checks) {
    FlexData co = FlexData::make_object();
    auto x = co.as_object();
    x.insert("name", fstr(c.name));
    x.insert("status", fstr(c.status));
    x.insert("detail", fstr(c.detail));
    FlexData hints = FlexData::make_array();
    auto ha = hints.as_array();
    for (const auto& h : c.hints) { ha.push_back(fstr(h)); }
    x.insert("hints", std::move(hints));
    if (c.status == "warn") { any_warn = true; }
    a.push_back(std::move(co));
  }
  o.insert("checks", std::move(arr));
  o.insert("has_warnings", FlexData::make_bool(any_warn));
  o.insert("ready", FlexData::make_bool(true));
  lock_guard<mutex> lk(_ctx.mu);
  _startup_checks = std::move(doc);
}

HttpResponse
SystemApi::h_startup_checks_(const HttpRequest&)
{
  lock_guard<mutex> lk(_ctx.mu);
  if (_startup_checks.is_object()) {
    return HttpResponse::json(200, _startup_checks.to_json());
  }
  // The HTTP server is up before the (blocking) probes finish, so a fast
  // client can arrive first -- report not-ready so it retries.
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("ready", FlexData::make_bool(false));
  oo.insert("has_warnings", FlexData::make_bool(false));
  oo.insert("checks", FlexData::make_array());
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SystemApi::h_i18n_get_(const HttpRequest&)
{
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("language",
            fstr(_ctx.sctx ? _ctx.sctx->language()
                           : string(default_language())));
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (const auto& l : supported_languages()) { a.push_back(fstr(l)); }
  oo.insert("supported", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SystemApi::h_i18n_set_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  auto bo = body->as_object();
  string tag = bo.contains("language")
                   ? string(bo.at("language").as_string("")) : "";
  if (tag.empty()) {
    return HttpResponse::error(400, "missing 'language'");
  }
  Status s = _ctx.session->set_language(tag);
  if (s.code != 0) {
    return HttpResponse::error(400, "unsupported language '" + tag + "'");
  }
  return h_i18n_get_(req);
}

// Whether any pipeline is RUNNING, which is what decides if a shrink is
// allowed. Asked of PipelineApi rather than of the session, because
// SessionIntf does not publish it and this controller already holds the
// lock its _locked query wants.
bool
SystemApi::any_running_() const
{
  bool running = false;
  _pipelines.for_each_live_locked(
      [&](const std::string&, const char* state, Pipeline&) {
        if (std::string(state) != "stopped") { running = true; }
      });
  return running;
}

HttpResponse
SystemApi::h_wired_pool_get_(const HttpRequest&)
{
  lock_guard<mutex> lk(_ctx.mu);
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  const std::size_t mb =
      _ctx.session != nullptr ? _ctx.session->wired_pool_mb() : 0;
  std::size_t used = 0, devmax = 0;
  int pct = 0;
  if (_ctx.sctx != nullptr && _ctx.sctx->services() != nullptr) {
    if (auto* mgr = _ctx.sctx->services()->generative_model_manager()) {
      used   = mgr->wired_pool_used() >> 20;
      devmax = mgr->wired_pool_device_max() >> 20;
      pct    = mgr->wired_pool_pct();
    }
  }
  oo.insert("mb", FlexData::make_int((long long)mb));
  oo.insert("used_mb", FlexData::make_int((long long)used));
  // 0 when there is no device to ask. The browser shows it as the
  // ceiling a larger request is clamped to, and says nothing when it is
  // unknown rather than implying there is no cap.
  oo.insert("device_max_mb", FlexData::make_int((long long)devmax));
  oo.insert("pct", FlexData::make_int((long long)pct));
  oo.insert("running", FlexData::make_bool(any_running_()));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SystemApi::h_wired_pool_set_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "expected object {mb: N}");
  }
  auto bo = body->as_object();
  if (!bo.contains("mb")) { return HttpResponse::error(400, "missing 'mb'"); }
  const long long mb = bo.at("mb").as_int(-1);
  if (mb < 0) {
    return HttpResponse::error(400, "'mb' must be 0 or more (0 restores the "
                                    "wired_pool_pct share of RAM)");
  }
  if (_ctx.session == nullptr) {
    return HttpResponse::error(404, "session not available");
  }
  const Status st = _ctx.session->set_wired_pool_mb((std::size_t)mb);
  // 3 is the shrink-while-running refusal, and it is the one a browser
  // must be able to tell apart: nothing is wrong, the request simply has
  // to wait for the run to stop. 409 rather than 400 says exactly that.
  if (st.code == 3) {
    return HttpResponse::error(409,
        "a pipeline is running: the wired pool can be raised but not "
        "lowered, because giving wired bytes back means unwiring buffers a "
        "model is still reading. Stop the pipeline to lower it.");
  }
  if (st.code != 0) {
    return HttpResponse::error(400, "the wired pool is not available in "
                                    "this build");
  }
  return h_wired_pool_get_(req);
}

HttpResponse
SystemApi::h_system_status_(const HttpRequest&)
{
  FlexData o = _status->query();
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SystemApi::h_hls_streams_(const HttpRequest&)
{
  lock_guard<mutex> lk(_ctx.mu);
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();

  // Through PipelineApi rather than over a local pipe list: it owns the
  // graphs, and the lock taken above is the one its _locked query wants.
  _pipelines.for_each_live_locked(
      [&](const string& pipe_id, const char* pipe_state, Pipeline& plref) {
    // Only a launched pipeline has a live HTTP server behind the
    // stage; a stopped one serves nothing. Paused keeps serving the
    // last playlist, so include it too.
    if (string(pipe_state) == "stopped") { return; }
    Pipeline* pl = &plref;
    for (auto it = pl->begin(); it != pl->end(); ++it) {
      const Stage* s = dynamic_cast<const Stage*>(*it);
      if (!s || string(s->type_name()) != "hls-broadcast") { continue; }

      // config_params() resolves declared defaults, so a stage that
      // never set "port"/"serve_http" still reports the real values.
      const auto params = s->config_params();
      auto param = [&](const char* k) -> const ConfigParam* {
        for (const auto& pr : params) {
          if (pr.key == k) { return &pr; }
        }
        return nullptr;
      };
      auto get_str = [&](const char* k, const char* def) -> string {
        const ConfigParam* pr = param(k);
        return pr ? string(pr->current_value.as_string(def)) : def;
      };
      auto get_int = [&](const char* k, long def) -> long {
        const ConfigParam* pr = param(k);
        return pr ? static_cast<long>(pr->current_value.as_int(def)) : def;
      };
      auto get_bool = [&](const char* k, bool def) -> bool {
        const ConfigParam* pr = param(k);
        return pr ? pr->current_value.as_bool(def) : def;
      };

      if (!get_bool("serve_http", true)) { continue; }

      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert("pipeline", fstr(pipe_id));
      eo.insert("stage", fstr(s->id()));
      eo.insert("state", fstr(pipe_state));
      eo.insert("playlist_name",
                fstr(get_str("playlist_name", "stream.m3u8")));
      eo.insert("port",
                FlexData::make_uint(
                    static_cast<uint64_t>(get_int("port", 8080))));
      // Report the RESOLVED bind address so the browser embeds a
      // player URL that actually connects. An empty configured value
      // is the stage's "auto" default; mirror the stage's own
      // resolution (web-ui address, else en0's LAN IP, else 0.0.0.0)
      // -- both inputs are stable for the session's life, so this
      // matches what the live stage bound to.
      string bind = get_str("bind_address", "");
      if (bind.empty()) {
        bind = _ctx.sctx ? _ctx.sctx->web_ui_bind_address() : string();
        if (bind.empty()) {
          const string lan = netx::primary_ipv4();
          bind = lan.empty() ? string("0.0.0.0") : lan;
        }
      }
      eo.insert("bind_address", fstr(bind));
      // Whether this broadcast carries an audio track: iport 1 (the strict
      // audio role) is wired to a producer. The UI uses it to auto-unmute the
      // player when a viewer attaches (a video-only stream stays muted).
      const auto& ie = s->iport_edges();
      const bool has_audio = ie.size() > 1 && ie[1].v != nullptr;
      eo.insert("audio", FlexData::make_bool(has_audio));
      a.push_back(std::move(e));
    }
  });
  oo.insert("streams", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

void
SystemApi::register_routes(HttpServer& s)
{
  s.route("GET", "/api/system/status",
          [this](const HttpRequest& r) { return h_system_status_(r); });
  s.route("GET", "/api/startup-checks",
          [this](const HttpRequest& r) { return h_startup_checks_(r); });
  s.route("GET", "/api/i18n",
          [this](const HttpRequest& r) { return h_i18n_get_(r); });
  s.route("PUT", "/api/i18n",
          [this](const HttpRequest& r) { return h_i18n_set_(r); });
  s.route("GET", "/api/system/wired-pool",
          [this](const HttpRequest& r) { return h_wired_pool_get_(r); });
  s.route("PUT", "/api/system/wired-pool",
          [this](const HttpRequest& r) { return h_wired_pool_set_(r); });
  s.route("GET", "/api/hls/streams",
          [this](const HttpRequest& r) { return h_hls_streams_(r); });
}

}
