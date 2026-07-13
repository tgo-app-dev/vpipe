#include "apps/web-ui/session-api.h"
#include "apps/web-ui/web-ui-delegate.h"
#include "apps/web-ui/web-ui-log-delegate.h"
#include "apps/web-ui/db-browser.h"
#include "apps/web-ui/model-browser.h"
#include "apps/web-ui/system-status.h"

#include "interfaces/log-delegate-intf.h"
#include "interfaces/session-context-intf.h"
#include "common/graph.h"
#include "common/preview-channel.h"
#include "common/temp-root.h"
#include "common/i18n.h"
#include "common/vertex.h"
#include "common/vpipe-format.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline-handle-impl.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline-spec.h"
#include "pipeline/stage.h"
#include "pipeline/stage-registry.h"
#include "vpipe/session-intf.h"

#include "common/host-net.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cxxabi.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <random>
#include <system_error>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

#include <unistd.h>

using namespace std;

namespace vpipe::webui {

namespace {

// Demangled, human-readable name for a payload type_info. Null (an
// "untyped" port) renders as "any".
string
demangle_(const type_info* ti)
{
  if (!ti) { return "any"; }
  int   status = 0;
  char* d = abi::__cxa_demangle(ti->name(), nullptr, nullptr, &status);
  string out = (status == 0 && d) ? string(d) : string(ti->name());
  std::free(d);
  return out;
}

// Two ports agree if both are typed and equal, or either is untyped
// (the same rule PipelineRuntime applies at launch).
bool
compatible_(const type_info* a, const type_info* b)
{
  if (!a || !b) { return true; }
  return *a == *b;
}

bool
blank_(const string& s)
{
  for (char c : s) {
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') { return false; }
  }
  return true;
}

// Strip leading/trailing ASCII whitespace (for user-supplied ids).
string
trim_(const string& s)
{
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == string::npos) { return {}; }
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

optional<FlexData>
parse_json_body_(const HttpRequest& req)
{
  if (blank_(req.body)) { return FlexData::make_object(); }
  try {
    return FlexData::from_json(req.body);
  } catch (const exception&) {
    return nullopt;
  }
}

FlexData
fstr_(string_view s)
{
  return FlexData::make_string(s);
}

// Extract a single query-string parameter value (no URL-decoding
// beyond '+'); returns empty when absent. `query` is the raw string
// after '?', e.g. "since=42&foo=bar".
string
query_param_(const string& query, const string& key)
{
  size_t i = 0;
  while (i < query.size()) {
    size_t amp = query.find('&', i);
    size_t end = (amp == string::npos) ? query.size() : amp;
    size_t eq  = query.find('=', i);
    if (eq != string::npos && eq < end) {
      if (query.compare(i, eq - i, key) == 0) {
        return query.substr(eq + 1, end - eq - 1);
      }
    }
    if (amp == string::npos) { break; }
    i = amp + 1;
  }
  return {};
}

// Percent-decode a query-string value (e.g. "%2Fa%20b" -> "/a b"). The
// client sends paths via encodeURIComponent, which escapes '/', spaces
// and non-ASCII, so a path query param must be decoded before use.
// '+' is left literal (encodeURIComponent uses %20 for space, %2B for a
// real '+'); a malformed trailing '%' is passed through unchanged.
string
url_decode_(const string& s)
{
  auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
  };
  string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      const int hi = hex(s[i + 1]);
      const int lo = hex(s[i + 2]);
      if (hi >= 0 && lo >= 0) {
        out.push_back(static_cast<char>((hi << 4) | lo));
        i += 2;
        continue;
      }
    }
    out.push_back(s[i]);
  }
  return out;
}

// Resolve a PipelineHandle to its live Pipeline graph (internal).
Pipeline*
live_pipeline_(const PipelineHandle& h)
{
  PipelineHandleImpl* impl = HandleAccess::impl(h);
  return impl ? impl->pipeline() : nullptr;
}

}  // namespace

SessionApi::SessionApi(SessionIntf* session, WebUiDelegate* ui,
                       WebUiLogDelegate* log)
  : _session(session)
  , _sctx(dynamic_cast<SessionContextIntf*>(session))
  , _ui(ui)
  , _log(log)
  , _status(std::make_unique<SystemStatusPoller>())
{
}

// The out-of-line dtor lives in the .cc so the std::unique_ptr<Impl>
// dtor (and therefore SystemStatusPoller's destructor) can see the
// complete type, avoiding the pImpl-with-unique_ptr incomplete-type
// pitfall.
SessionApi::~SessionApi() = default;

const char*
SessionApi::state_name_(State s)
{
  switch (s) {
    case State::Stopped: return "stopped";
    case State::Running: return "running";
    case State::Paused:  return "paused";
  }
  return "stopped";
}

SessionApi::Pipe*
SessionApi::find_(const string& id)
{
  for (auto& up : _pipes) {
    if (up->id == id) { return up.get(); }
  }
  return nullptr;
}

bool
SessionApi::any_pipeline_active_() const   // caller holds _mu
{
  for (auto& up : _pipes) {
    if (up->state != State::Stopped) { return true; }
  }
  return false;
}

void
SessionApi::reap_completed_()   // caller holds _mu
{
  for (auto& up : _pipes) {
    Pipe& p = *up;
    if (p.state != State::Running || !p.handle || !p.handle->valid()) {
      continue;
    }
    PipelineHandleImpl* impl = HandleAccess::impl(*p.handle);
    const PipelineRuntime* rt = impl ? impl->runtime() : nullptr;
    if (!rt || !rt->self_completed()) { continue; }
    // Every stage signalled done and the pipeline drained on its own:
    // tear the runtime down (close buffers, clear stage running flags)
    // and mark it stopped. stop_pipeline does not block here -- every
    // driver has already reached co_return.
    _session->stop_pipeline(*p.handle);
    p.state = State::Stopped;
    if (_sctx) {
      _sctx->info(fmt("pipeline '{}' auto-stopped: all stages done",
                      p.id));
    }
  }
}

void
SessionApi::set_startup_checks(const std::vector<PermissionCheck>& checks)
{
  FlexData doc = FlexData::make_object();
  auto o = doc.as_object();
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  bool any_warn = false;
  for (const auto& c : checks) {
    FlexData co = FlexData::make_object();
    auto x = co.as_object();
    x.insert("name", fstr_(c.name));
    x.insert("status", fstr_(c.status));
    x.insert("detail", fstr_(c.detail));
    FlexData hints = FlexData::make_array();
    auto ha = hints.as_array();
    for (const auto& h : c.hints) { ha.push_back(fstr_(h)); }
    x.insert("hints", std::move(hints));
    if (c.status == "warn") { any_warn = true; }
    a.push_back(std::move(co));
  }
  o.insert("checks", std::move(arr));
  o.insert("has_warnings", FlexData::make_bool(any_warn));
  o.insert("ready", FlexData::make_bool(true));
  lock_guard<mutex> lk(_mu);
  _startup_checks = std::move(doc);
}

HttpResponse
SessionApi::h_startup_checks_(const HttpRequest&)
{
  lock_guard<mutex> lk(_mu);
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
SessionApi::h_i18n_get_(const HttpRequest&)
{
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("language",
            fstr_(_sctx ? _sctx->language() : string(default_language())));
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (const auto& l : supported_languages()) { a.push_back(fstr_(l)); }
  oo.insert("supported", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_i18n_set_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  auto bo = body->as_object();
  string tag = bo.contains("language")
                   ? string(bo.at("language").as_string("")) : "";
  if (tag.empty()) {
    return HttpResponse::error(400, "missing 'language'");
  }
  Status s = _session->set_language(tag);
  if (s.code != 0) {
    return HttpResponse::error(400, "unsupported language '" + tag + "'");
  }
  return h_i18n_get_(req);
}

void
SessionApi::dematerialize_(Pipe& p)
{
  if (p.handle && p.handle->valid()) {
    _session->unload_pipeline(*p.handle);
  }
  p.handle.reset();
}

bool
SessionApi::materialize_(Pipe& p, string& err)
{
  dematerialize_(p);

  PipelineHandle h = _session->create_pipeline(p.id);
  if (!h.valid()) {
    err = "create_pipeline('" + p.id + "') failed";
    return false;
  }

  // Insert stages in dependency order (a stage's iport sources must be
  // inserted first). We don't trust p.stages to be pre-sorted; emit
  // greedily until all are placed or no progress is possible.
  unordered_map<string, StageHandle> by_id;
  vector<bool>                       done(p.stages.size(), false);
  size_t                             placed = 0;
  while (placed < p.stages.size()) {
    bool progress = false;
    for (size_t i = 0; i < p.stages.size(); ++i) {
      if (done[i]) { continue; }
      const StageSpec& st = p.stages[i];
      bool ready = true;
      for (auto& [src, op] : st.iports) {
        // An empty src is a DISCONNECTED (optional) iport -- no
        // dependency to wait for.
        if (src.empty()) { continue; }
        if (by_id.find(src) == by_id.end()) { ready = false; break; }
      }
      if (!ready) { continue; }

      vector<StagePortHandle> iports;
      iports.reserve(st.iports.size());
      for (auto& [src, op] : st.iports) {
        // Empty src -> a null StageHandle keeps the positional slot
        // unwired (see PipelineHandle::insert_stage).
        iports.push_back(src.empty()
            ? StagePortHandle{ StageHandle{}, 0 }
            : StagePortHandle{ by_id.at(src), op });
      }
      string cfg = st.config.is_object() ? st.config.to_json()
                                         : string("{}");
      StageHandle sh = h.insert_stage(st.type, st.id, iports, cfg);
      if (!sh.valid()) {
        _session->unload_pipeline(h);
        err = "failed to construct stage '" + st.id + "' (type '"
              + st.type + "'); check its config";
        return false;
      }
      by_id.emplace(st.id, sh);
      done[i] = true;
      ++placed;
      progress = true;
    }
    if (!progress) {
      _session->unload_pipeline(h);
      err = "stage dependency cycle or missing source in pipeline '"
            + p.id + "'";
      return false;
    }
  }

  p.handle = h;
  p.state  = State::Stopped;
  return true;
}

FlexData
SessionApi::pipe_summary_(const Pipe& p) const
{
  FlexData o = FlexData::make_object();
  auto v = o.as_object();
  v.insert("id", fstr_(p.id));
  v.insert("state", fstr_(state_name_(p.state)));
  v.insert("stage_count", FlexData::make_uint(p.stages.size()));
  v.insert("storage_path", fstr_(p.storage_path));
  v.insert("materialized",
           FlexData::make_bool(p.handle && p.handle->valid()));
  return o;
}

Stage*
SessionApi::live_stage_(const Pipe& p, const string& id) const
{
  if (!(p.handle && p.handle->valid())) { return nullptr; }
  Pipeline* pl = live_pipeline_(*p.handle);
  if (!pl) { return nullptr; }
  for (auto it = pl->begin(); it != pl->end(); ++it) {
    if (Stage* s = dynamic_cast<Stage*>(*it)) {
      if (s->id() == id) { return s; }
    }
  }
  return nullptr;
}

FlexData
SessionApi::graph_json_(const Pipe& p) const
{
  FlexData g = FlexData::make_object();
  auto go = g.as_object();
  FlexData nodes = FlexData::make_array();
  FlexData edges = FlexData::make_array();
  auto na = nodes.as_array();
  auto ea = edges.as_array();

  Pipeline* pl = (p.handle && p.handle->valid())
                     ? live_pipeline_(*p.handle) : nullptr;
  if (pl) {
    unordered_map<const Vertex*, string> ids;
    for (auto it = pl->begin(); it != pl->end(); ++it) {
      const Vertex* v = *it;
      if (const Stage* s = dynamic_cast<const Stage*>(v)) {
        ids.emplace(v, s->id());
      }
    }
    for (auto it = pl->begin(); it != pl->end(); ++it) {
      const Vertex* v = *it;
      const Stage*  s = dynamic_cast<const Stage*>(v);
      if (!s) { continue; }

      FlexData node = FlexData::make_object();
      auto no = node.as_object();
      no.insert("id", fstr_(s->id()));
      no.insert("type", fstr_(s->type_name()));
      // Non-empty when the stage's config is incomplete/invalid; the
      // stage still exists in the graph but is skipped at launch until
      // the problem is fixed (validation is deferred to launch).
      no.insert("config_error", fstr_(s->config_error()));

      FlexData iports = FlexData::make_array();
      auto ia = iports.as_array();
      for (unsigned i = 0; i < s->num_iports(); ++i) {
        FlexData pe = FlexData::make_object();
        auto peo = pe.as_object();
        peo.insert("index", FlexData::make_uint(i));
        peo.insert("type", fstr_(demangle_(s->iport_payload_type(i))));
        const std::string_view itags = s->iport_payload_tags(i);
        if (!itags.empty()) {
          peo.insert("tags", FlexData::make_string(itags));
        }
        peo.insert("clock",
                   FlexData::make_uint(s->iport_clock_group(i)));
        ia.push_back(std::move(pe));
      }
      no.insert("iports", std::move(iports));

      FlexData oports = FlexData::make_array();
      auto oa = oports.as_array();
      for (unsigned i = 0; i < s->num_oports(); ++i) {
        FlexData pe = FlexData::make_object();
        auto peo = pe.as_object();
        peo.insert("index", FlexData::make_uint(i));
        peo.insert("type", fstr_(demangle_(s->oport_payload_type(i))));
        const std::string_view otags = s->oport_payload_tags(i);
        if (!otags.empty()) {
          peo.insert("tags", FlexData::make_string(otags));
        }
        peo.insert("clock",
                   FlexData::make_uint(s->oport_clock_group(i)));
        oa.push_back(std::move(pe));
      }
      no.insert("oports", std::move(oports));
      na.push_back(std::move(node));

      // Edges: this stage's iport i is fed by iport_edges()[i].
      const auto& ins = s->iport_edges();
      for (size_t i = 0; i < ins.size(); ++i) {
        const InEdge& e = ins[i];
        auto sit = ids.find(e.v);
        if (sit == ids.end()) { continue; }
        const Stage* src = dynamic_cast<const Stage*>(e.v);
        FlexData edge = FlexData::make_object();
        auto eo = edge.as_object();
        eo.insert("from", fstr_(sit->second));
        eo.insert("from_port", FlexData::make_uint(e.p));
        eo.insert("to", fstr_(s->id()));
        eo.insert("to_port", FlexData::make_uint(i));
        eo.insert("type",
                  fstr_(demangle_(src ? src->oport_payload_type(e.p)
                                      : nullptr)));
        ea.push_back(std::move(edge));
      }
    }
  }
  go.insert("nodes", std::move(nodes));
  go.insert("edges", std::move(edges));
  return g;
}

// Kahn's algorithm over the StageSpec list (edge: iport source -> consumer).
// Returns the stage indices in dependency order, stable wrt the input order
// (each pass places the ready stages in their existing relative order, so a
// already-topological list is returned unchanged and an out-of-place stage
// moves minimally). A cycle -- which the runtime rejects at launch but which
// we must not UB on -- leaves the cyclic stages, appended at the end in
// input order so none are dropped.
std::vector<std::size_t>
SessionApi::topo_order_(const std::vector<StageSpec>& stages)
{
  const std::size_t n = stages.size();
  std::unordered_map<std::string, std::size_t> idx;
  for (std::size_t i = 0; i < n; ++i) { idx[stages[i].id] = i; }
  std::vector<unsigned>             indeg(n, 0);
  std::vector<std::vector<std::size_t>> consumers(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (const auto& [src, op] : stages[i].iports) {
      auto it = idx.find(src);
      if (it == idx.end() || it->second == i) { continue; }  // ext / self
      ++indeg[i];
      consumers[it->second].push_back(i);
    }
  }
  std::vector<std::size_t> out;
  out.reserve(n);
  std::vector<bool> placed(n, false);
  bool progress = true;
  while (out.size() < n && progress) {
    progress = false;
    for (std::size_t i = 0; i < n; ++i) {
      if (placed[i] || indeg[i] != 0) { continue; }
      placed[i] = true;
      out.push_back(i);
      progress = true;
      for (std::size_t c : consumers[i]) {
        if (!placed[c] && indeg[c] > 0) { --indeg[c]; }
      }
    }
  }
  for (std::size_t i = 0; i < n; ++i) {        // cycle leftovers, in order
    if (!placed[i]) { out.push_back(i); }
  }
  return out;
}

FlexData
SessionApi::to_flex_spec_(const Pipe& p) const
{
  FlexData spec = FlexData::make_object();
  auto o = spec.as_object();
  o.insert("id", fstr_(p.id));
  FlexData stages = FlexData::make_array();
  auto sa = stages.as_array();
  // Emit in dependency order so the single-pass core loader can resolve every
  // iport (in-memory edits can leave p.stages non-topological -- see above).
  for (std::size_t si : topo_order_(p.stages)) {
    const StageSpec& st = p.stages[si];
    FlexData e = FlexData::make_object();
    auto eo = e.as_object();
    eo.insert("id", fstr_(st.id));
    eo.insert("type", fstr_(st.type));
    FlexData ip = FlexData::make_array();
    auto ia = ip.as_array();
    for (auto& [src, op] : st.iports) {
      FlexData pe = FlexData::make_object();
      auto peo = pe.as_object();
      peo.insert("src", fstr_(src));
      peo.insert("oport", FlexData::make_uint(op));
      ia.push_back(std::move(pe));
    }
    eo.insert("iports", std::move(ip));
    eo.insert("config",
              st.config.is_object() ? st.config : FlexData::make_object());
    sa.push_back(std::move(e));
  }
  o.insert("stages", std::move(stages));
  o.insert("subpipelines", FlexData::make_array());
  return spec;
}

// ===================================================================
// Route handlers
// ===================================================================

namespace {

// Serialize a PortSpec span to a FlexData array of
// {name, doc, type} objects (type demangled; "any" when untyped).
FlexData
ports_to_flex_(std::span<const PortSpec> ports)
{
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (const PortSpec& p : ports) {
    FlexData o = FlexData::make_object();
    auto oo = o.as_object();
    oo.insert("name", FlexData::make_string(p.name));
    oo.insert("doc",  FlexData::make_string(p.doc));
    oo.insert("type", FlexData::make_string(demangle_(p.type)));
    // Optional finer-grained payload tags (comma-separated, OR semantics);
    // the composer checks them on top of the beat type. Omitted when empty.
    if (!p.tags.empty()) {
      oo.insert("tags", FlexData::make_string(p.tags));
    }
    a.push_back(std::move(o));
  }
  return arr;
}

}  // namespace

HttpResponse
SessionApi::h_stage_types_(const HttpRequest&)
{
  auto all = StageRegistry::get().all();
  vector<string> names;
  names.reserve(all.size());
  for (auto& [id, name] : all) { names.push_back(name); }
  sort(names.begin(), names.end());

  // One object per stage type. A type that registered a StageSpec
  // (VPIPE_REGISTER_SPEC) carries its category / description / ports /
  // attr count; one that didn't falls back to a generic entry so the
  // toolbox still lists it.
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (auto& n : names) {
    FlexData o = FlexData::make_object();
    auto oo = o.as_object();
    oo.insert("type", fstr_(n));
    // Qualify: SessionApi has its own private `StageSpec` (the editable
    // spec entry), so the registry's vpipe::StageSpec needs the prefix.
    const vpipe::StageSpec* sp = StageRegistry::get().spec(n);
    if (sp) {
      oo.insert("category",
                FlexData::make_string(stage_category_name(sp->category)));
      oo.insert("doc", FlexData::make_string(sp->doc));
      oo.insert("display_name", FlexData::make_string(sp->display_name));
      oo.insert("iports", ports_to_flex_(sp->iports));
      oo.insert("oports", ports_to_flex_(sp->oports));
      oo.insert("attr_count",
                FlexData::make_uint(sp->attrs.size()));
      // Hidden stages stay in the list (so an already-present instance
      // still renders with its spec) but the composer omits them from
      // the add-a-stage toolbox.
      oo.insert("hidden", FlexData::make_bool(sp->hidden));
    } else {
      oo.insert("category", fstr_("generic"));
      oo.insert("doc", fstr_(""));
      oo.insert("display_name", fstr_(""));
      oo.insert("iports", FlexData::make_array());
      oo.insert("oports", FlexData::make_array());
      oo.insert("attr_count", FlexData::make_uint(0));
      oo.insert("hidden", FlexData::make_bool(false));
    }
    a.push_back(std::move(o));
  }
  return HttpResponse::json(200, arr.to_json());
}

HttpResponse
SessionApi::h_list_pipelines_(const HttpRequest&)
{
  lock_guard<mutex> lk(_mu);
  reap_completed_();   // surface self-terminated pipelines as stopped
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (auto& up : _pipes) { a.push_back(pipe_summary_(*up)); }
  return HttpResponse::json(200, arr.to_json());
}

HttpResponse
SessionApi::h_create_pipeline_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  string id = string(body->is_object()
                         ? body->as_object().contains("id")
                               ? body->as_object().at("id").as_string("")
                               : ""
                         : "");
  if (id.empty()) { return HttpResponse::error(400, "missing 'id'"); }

  lock_guard<mutex> lk(_mu);
  if (find_(id)) {
    return HttpResponse::error(409, "pipeline '" + id + "' already exists");
  }
  auto p = make_unique<Pipe>();
  p->id = id;
  string err;
  if (!materialize_(*p, err)) {
    return HttpResponse::error(500, err);
  }
  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr_(p->id));
  d.insert("state", fstr_(state_name_(p->state)));
  d.insert("storage_path", fstr_(p->storage_path));
  d.insert("graph", graph_json_(*p));
  _pipes.push_back(std::move(p));
  return HttpResponse::json(201, detail.to_json());
}

HttpResponse
SessionApi::h_rename_pipeline_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  lock_guard<mutex> lk(_mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (p->state != State::Stopped) {
    return HttpResponse::error(409, "stop the pipeline before renaming");
  }
  string to = trim_(string(body->as_object().contains("to")
                               ? body->as_object().at("to").as_string("")
                               : ""));
  if (to.empty()) {
    return HttpResponse::error(400, "missing new pipeline id 'to'");
  }
  const string old = p->id;
  if (to != old) {
    if (find_(to)) {
      return HttpResponse::error(
          409, "pipeline '" + to + "' already exists");
    }
    // Re-materialize under the new id (the live handle is keyed by id).
    p->id = to;
    string err;
    if (!materialize_(*p, err)) {
      p->id = old;                    // revert + restore the old handle
      string e2;
      materialize_(*p, e2);
      return HttpResponse::error(500, err);
    }
  }
  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr_(p->id));
  d.insert("state", fstr_(state_name_(p->state)));
  d.insert("storage_path", fstr_(p->storage_path));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
SessionApi::h_load_pipeline_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  string path = string(body->as_object().contains("path")
                           ? body->as_object().at("path").as_string("")
                           : "");
  if (path.empty()) { return HttpResponse::error(400, "missing 'path'"); }

  // Confine the read to the filesystem sandbox. The client speaks the
  // sandbox's virtual namespace; `path` stays the virtual path we
  // remember for display, `real` is the file actually opened.
  string cerr;
  std::filesystem::path real =
      _sctx ? _sctx->confine_path(path, /*for_write=*/false, &cerr)
            : std::filesystem::path(path);
  if (real.empty()) {
    return HttpResponse::error(
        400, "path '" + path + "' rejected: "
                 + (cerr.empty() ? "outside sandbox" : cerr));
  }

  lock_guard<mutex> lk(_mu);
  PipelineHandle h = _session->load_pipeline(real.string());
  if (!h.valid()) {
    return HttpResponse::error(400, "failed to load '" + path + "'");
  }
  Pipeline* pl = live_pipeline_(h);
  if (!pl) {
    _session->unload_pipeline(h);
    return HttpResponse::error(500, "loaded pipeline has no graph");
  }
  string id = pl->id();
  if (find_(id)) {
    _session->unload_pipeline(h);
    return HttpResponse::error(
        409, "a pipeline named '" + id + "' is already open");
  }

  auto p = make_unique<Pipe>();
  p->id           = id;
  p->handle       = h;
  p->state        = State::Stopped;
  p->storage_path = path;

  // Recover the editable spec from the live graph.
  FlexData spec = pipeline_to_spec(*pl);
  if (spec.is_object() && spec.as_object().contains("stages")) {
    FlexData stagesF = spec.as_object().at("stages");
    if (stagesF.is_array()) {
      auto sa = stagesF.as_array();
      for (size_t i = 0; i < sa.size(); ++i) {
        FlexData e = sa.at(i);
        if (!e.is_object()) { continue; }
        auto eo = e.as_object();
        StageSpec st;
        st.id   = string(eo.contains("id") ? eo.at("id").as_string("") : "");
        st.type = string(eo.contains("type")
                             ? eo.at("type").as_string("") : "");
        st.config = eo.contains("config") ? eo.at("config")
                                          : FlexData::make_object();
        if (eo.contains("iports")) {
          FlexData ipF = eo.at("iports");
          if (ipF.is_array()) {
            auto ia = ipF.as_array();
            for (size_t j = 0; j < ia.size(); ++j) {
              FlexData pe = ia.at(j);
              // A null element (or an object with empty/missing src) is
              // a DISCONNECTED iport that keeps its positional slot.
              if (pe.is_null()) {
                st.iports.emplace_back(string(), 0u);
                continue;
              }
              if (!pe.is_object()) { continue; }
              auto peo = pe.as_object();
              string src = string(peo.contains("src")
                                      ? peo.at("src").as_string("") : "");
              unsigned op = static_cast<unsigned>(
                  peo.contains("oport") ? peo.at("oport").as_uint(0) : 0);
              st.iports.emplace_back(std::move(src), op);
            }
          }
        }
        p->stages.push_back(std::move(st));
      }
    }
  }

  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr_(p->id));
  d.insert("state", fstr_(state_name_(p->state)));
  d.insert("storage_path", fstr_(p->storage_path));
  d.insert("graph", graph_json_(*p));
  _pipes.push_back(std::move(p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
SessionApi::h_get_pipeline_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_mu);
  reap_completed_();   // surface self-terminated pipelines as stopped
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr_(p->id));
  d.insert("state", fstr_(state_name_(p->state)));
  d.insert("storage_path", fstr_(p->storage_path));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
SessionApi::h_save_pipeline_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  lock_guard<mutex> lk(_mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }

  string path = p->storage_path;
  if (body->is_object() && body->as_object().contains("path")) {
    string bp = string(body->as_object().at("path").as_string(""));
    if (!bp.empty()) { path = bp; }
  }
  if (path.empty()) {
    return HttpResponse::error(400, "no path given and none remembered");
  }
  // Confine the write to the filesystem sandbox (the client works in the
  // sandbox's virtual namespace; `path` stays the virtual path we echo
  // and remember, `real` is where the bytes actually land).
  string cerr;
  std::filesystem::path real =
      _sctx ? _sctx->confine_path(path, /*for_write=*/true, &cerr)
            : std::filesystem::path(path);
  if (real.empty()) {
    return HttpResponse::error(
        400, "path '" + path + "' rejected: "
                 + (cerr.empty() ? "outside sandbox" : cerr));
  }
  FlexData spec = to_flex_spec_(*p);
  ofstream f(real, ios::binary);
  if (!f) {
    return HttpResponse::error(500, "cannot open '" + path + "' for write");
  }
  f << spec.to_json(/*pretty=*/true);
  if (!f) {
    return HttpResponse::error(500, "write to '" + path + "' failed");
  }
  p->storage_path = path;
  FlexData o = FlexData::make_object();
  o.as_object().insert("ok", FlexData::make_bool(true));
  o.as_object().insert("storage_path", fstr_(path));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_unload_pipeline_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_mu);
  auto it = req.params.find("id");
  if (it == req.params.end()) {
    return HttpResponse::error(400, "missing id");
  }
  for (size_t i = 0; i < _pipes.size(); ++i) {
    if (_pipes[i]->id == it->second) {
      dematerialize_(*_pipes[i]);
      _pipes.erase(_pipes.begin() + static_cast<long>(i));
      return HttpResponse::ok();
    }
  }
  return HttpResponse::error(404, "no such pipeline");
}

HttpResponse
SessionApi::h_launch_pipeline_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (!p->handle || !p->handle->valid()) {
    return HttpResponse::error(409, "pipeline is not materialized");
  }
  if (p->state != State::Stopped) {
    return HttpResponse::error(409, "pipeline is already running");
  }
  Status s = _session->launch_pipeline(*p->handle);
  if (s.code != 0) {
    return HttpResponse::error(500, "launch failed (status " +
                               to_string(s.code) + ")");
  }
  p->state = State::Running;
  return HttpResponse::json(200, pipe_summary_(*p).to_json());
}

HttpResponse
SessionApi::h_pause_pipeline_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (p->state != State::Running) {
    return HttpResponse::error(409, "pipeline is not running");
  }
  Status s = _session->pause_pipeline(*p->handle);
  if (s.code != 0) {
    return HttpResponse::error(500, "pause failed (status " +
                               to_string(s.code) + ")");
  }
  p->state = State::Paused;
  return HttpResponse::json(200, pipe_summary_(*p).to_json());
}

HttpResponse
SessionApi::h_stop_pipeline_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (p->state == State::Stopped) {
    return HttpResponse::json(200, pipe_summary_(*p).to_json());
  }
  Status s = _session->stop_pipeline(*p->handle);
  if (s.code != 0) {
    return HttpResponse::error(500, "stop failed (status " +
                               to_string(s.code) + ")");
  }
  p->state = State::Stopped;
  return HttpResponse::json(200, pipe_summary_(*p).to_json());
}

HttpResponse
SessionApi::h_buffer_status_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_mu);
  reap_completed_();   // surface self-terminated pipelines as stopped
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("id", fstr_(p->id));
  oo.insert("state", fstr_(state_name_(p->state)));
  FlexData edges = FlexData::make_array();
  auto ea = edges.as_array();
  // Only a launched pipeline has live edge buffers. A stopped/edited
  // pipeline simply reports an empty edge list (the overlay clears).
  if (p->handle && p->handle->valid()) {
    PipelineHandleImpl* impl = HandleAccess::impl(*p->handle);
    const PipelineRuntime* rt = impl ? impl->runtime() : nullptr;
    if (rt) {
      for (const auto& e : rt->edge_buffer_stats()) {
        FlexData edge = FlexData::make_object();
        auto x = edge.as_object();
        x.insert("from", fstr_(e.from_id));
        x.insert("from_port", FlexData::make_uint(e.from_port));
        x.insert("to", fstr_(e.to_id));
        x.insert("to_port", FlexData::make_uint(e.to_port));
        x.insert("backlog", FlexData::make_uint(e.backlog));
        x.insert("capacity", FlexData::make_uint(e.capacity));
        x.insert("dropped", FlexData::make_uint(e.dropped));
        x.insert("closed", FlexData::make_bool(e.closed));
        ea.push_back(std::move(edge));
      }
    }
  }
  oo.insert("edges", std::move(edges));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_insert_stage_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  auto bo = body->as_object();

  lock_guard<mutex> lk(_mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (p->state != State::Stopped) {
    return HttpResponse::error(409, "stop the pipeline before editing");
  }

  StageSpec st;
  st.id   = string(bo.contains("id") ? bo.at("id").as_string("") : "");
  st.type = string(bo.contains("type") ? bo.at("type").as_string("") : "");
  st.config = bo.contains("config") && bo.at("config").is_object()
                  ? bo.at("config") : FlexData::make_object();
  if (st.id.empty() || st.type.empty()) {
    return HttpResponse::error(400, "stage 'id' and 'type' are required");
  }
  if (find_if(p->stages.begin(), p->stages.end(),
              [&](const StageSpec& s) { return s.id == st.id; })
      != p->stages.end()) {
    return HttpResponse::error(409, "stage '" + st.id + "' already exists");
  }
  if (StageRegistry::get().find_id(st.type) == StageTypeId::unknown) {
    return HttpResponse::error(400, "unknown stage type '" + st.type + "'");
  }
  if (bo.contains("iports") && bo.at("iports").is_array()) {
    FlexData ipF = bo.at("iports");
    auto ia = ipF.as_array();
    for (size_t j = 0; j < ia.size(); ++j) {
      FlexData pe = ia.at(j);
      // A null element is a DISCONNECTED iport (positional gap).
      if (pe.is_null()) {
        st.iports.emplace_back(string(), 0u);
        continue;
      }
      if (!pe.is_object()) { continue; }
      auto peo = pe.as_object();
      string src = string(peo.contains("src")
                              ? peo.at("src").as_string("") : "");
      unsigned op = static_cast<unsigned>(
          peo.contains("oport") ? peo.at("oport").as_uint(0) : 0);
      // An empty src is a disconnected iport -- no source to resolve.
      if (src.empty()) {
        st.iports.emplace_back(string(), 0u);
        continue;
      }
      // Topological insert: every (real) source must already exist.
      if (find_if(p->stages.begin(), p->stages.end(),
                  [&](const StageSpec& s) { return s.id == src; })
          == p->stages.end()) {
        return HttpResponse::error(
            400, "iport source '" + src + "' does not exist yet "
                 "(insert sources first)");
      }
      st.iports.emplace_back(std::move(src), op);
    }
  }

  // Commit speculatively, then rebuild + validate Beat types.
  p->stages.push_back(st);
  string err;
  if (!materialize_(*p, err)) {
    p->stages.pop_back();
    string e2;
    materialize_(*p, e2);
    return HttpResponse::error(400, err);
  }
  // Beat-type agreement on the new stage's inputs.
  Stage* ns = nullptr;
  if (Pipeline* pl = live_pipeline_(*p->handle)) {
    for (auto vit = pl->begin(); vit != pl->end(); ++vit) {
      if (Stage* s = dynamic_cast<Stage*>(*vit)) {
        if (s->id() == st.id) { ns = s; break; }
      }
    }
  }
  if (ns) {
    const auto& ins = ns->iport_edges();
    for (size_t i = 0; i < ins.size(); ++i) {
      Stage* src = dynamic_cast<Stage*>(ins[i].v);
      const type_info* a = src ? src->oport_payload_type(ins[i].p) : nullptr;
      const type_info* b = ns->iport_payload_type(static_cast<unsigned>(i));
      if (!compatible_(a, b)) {
        p->stages.pop_back();
        string e2;
        materialize_(*p, e2);
        return HttpResponse::error(
            400, "Beat type mismatch on iport " + to_string(i) +
                 " of '" + st.id + "': " + demangle_(b) +
                 " cannot accept " + demangle_(a));
      }
      // Deeper check: payload tags (finer constraint on the beat type).
      const std::string_view atg =
          src ? src->oport_payload_tags(ins[i].p) : std::string_view{};
      const std::string_view btg =
          ns->iport_payload_tags(static_cast<unsigned>(i));
      if (!port_tags_compatible(atg, btg)) {
        p->stages.pop_back();
        string e2;
        materialize_(*p, e2);
        return HttpResponse::error(
            400, "Payload tag mismatch on iport " + to_string(i) +
                 " of '" + st.id + "': {" + string(btg) +
                 "} cannot accept {" + string(atg) + "}");
      }
    }
  }

  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr_(p->id));
  d.insert("state", fstr_(state_name_(p->state)));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(201, detail.to_json());
}

HttpResponse
SessionApi::h_remove_stage_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_mu);
  auto pit = req.params.find("id");
  auto sit = req.params.find("sid");
  Pipe* p = (pit != req.params.end()) ? find_(pit->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (sit == req.params.end()) {
    return HttpResponse::error(400, "missing stage id");
  }
  if (p->state != State::Stopped) {
    return HttpResponse::error(409, "stop the pipeline before editing");
  }
  const string& sid = sit->second;
  auto target = find_if(p->stages.begin(), p->stages.end(),
                        [&](const StageSpec& s) { return s.id == sid; });
  if (target == p->stages.end()) {
    return HttpResponse::error(404, "no such stage '" + sid + "'");
  }
  // Topological removal: refuse if any other stage consumes this one.
  for (const StageSpec& s : p->stages) {
    for (auto& [src, op] : s.iports) {
      if (src == sid) {
        return HttpResponse::error(
            409, "stage '" + sid + "' feeds '" + s.id +
                 "'; remove downstream stages first");
      }
    }
  }
  p->stages.erase(target);
  string err;
  if (!materialize_(*p, err)) {
    return HttpResponse::error(500, err);
  }
  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr_(p->id));
  d.insert("state", fstr_(state_name_(p->state)));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
SessionApi::h_rename_stage_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  lock_guard<mutex> lk(_mu);
  auto pit = req.params.find("id");
  auto sit = req.params.find("sid");
  Pipe* p = (pit != req.params.end()) ? find_(pit->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (sit == req.params.end()) {
    return HttpResponse::error(400, "missing stage id");
  }
  if (p->state != State::Stopped) {
    return HttpResponse::error(409, "stop the pipeline before editing");
  }
  const string& sid = sit->second;
  string to = trim_(string(body->as_object().contains("to")
                               ? body->as_object().at("to").as_string("")
                               : ""));
  if (to.empty()) {
    return HttpResponse::error(400, "missing new stage id 'to'");
  }
  auto target = find_if(p->stages.begin(), p->stages.end(),
                        [&](const StageSpec& s) { return s.id == sid; });
  if (target == p->stages.end()) {
    return HttpResponse::error(404, "no such stage '" + sid + "'");
  }
  if (to != sid) {
    if (find_if(p->stages.begin(), p->stages.end(),
                [&](const StageSpec& s) { return s.id == to; })
        != p->stages.end()) {
      return HttpResponse::error(
          409, "stage '" + to + "' already exists");
    }
    // Rename the stage AND rewrite every iport that sources from it, so
    // the edges follow. Then rebuild the live graph.
    target->id = to;
    for (StageSpec& s : p->stages) {
      for (auto& [src, op] : s.iports) {
        if (src == sid) { src = to; }
      }
    }
    string err;
    if (!materialize_(*p, err)) {
      target->id = sid;                 // revert both the id and the edges
      for (StageSpec& s : p->stages) {
        for (auto& [src, op] : s.iports) {
          if (src == to) { src = sid; }
        }
      }
      string e2;
      materialize_(*p, e2);
      return HttpResponse::error(500, err);
    }
  }
  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr_(p->id));
  d.insert("state", fstr_(state_name_(p->state)));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

// Duplicate a stage: deep-copy its type + config (its SETTINGS) under a
// fresh, non-colliding id -- the source id plus a numeric suffix ("-2",
// "-3", ...) as the composer's suggestId does. The copy carries NO input
// connections (a duplicate clones settings, not wiring), so the operator
// wires it where they want. Optional body {to} overrides the generated id
// (rejected if it collides). Stopped-only, like the other topology edits.
HttpResponse
SessionApi::h_duplicate_stage_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_mu);
  auto pit = req.params.find("id");
  auto sit = req.params.find("sid");
  Pipe* p = (pit != req.params.end()) ? find_(pit->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (sit == req.params.end()) {
    return HttpResponse::error(400, "missing stage id");
  }
  if (p->state != State::Stopped) {
    return HttpResponse::error(409, "stop the pipeline before editing");
  }
  const string& sid = sit->second;
  auto src = find_if(p->stages.begin(), p->stages.end(),
                     [&](const StageSpec& s) { return s.id == sid; });
  if (src == p->stages.end()) {
    return HttpResponse::error(404, "no such stage '" + sid + "'");
  }

  auto taken = [&](const string& cand) {
    return find_if(p->stages.begin(), p->stages.end(),
                   [&](const StageSpec& s) { return s.id == cand; })
           != p->stages.end();
  };

  // New id: an optional caller-supplied 'to', else source id + a numeric
  // suffix that does not collide.
  string new_id;
  if (auto body = parse_json_body_(req); body && body->is_object()
      && body->as_object().contains("to")) {
    new_id = trim_(string(body->as_object().at("to").as_string("")));
    if (!new_id.empty() && taken(new_id)) {
      return HttpResponse::error(409, "stage '" + new_id + "' already exists");
    }
  }
  if (new_id.empty()) {
    for (int i = 2; ; ++i) {
      string cand = sid + "-" + to_string(i);
      if (!taken(cand)) { new_id = std::move(cand); break; }
    }
  }

  // Build the copy fully from `src` BEFORE push_back (which may realloc the
  // vector and dangle the iterator). config copy is a deep clone (FlexData
  // has value semantics); iports are intentionally left empty.
  StageSpec dup;
  dup.id     = new_id;
  dup.type   = src->type;
  dup.config = src->config;
  p->stages.push_back(std::move(dup));

  string err;
  if (!materialize_(*p, err)) {
    p->stages.pop_back();
    string e2;
    materialize_(*p, e2);
    return HttpResponse::error(400, err);
  }

  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr_(p->id));
  d.insert("state", fstr_(state_name_(p->state)));
  d.insert("stage", fstr_(new_id));   // the freshly created copy's id
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(201, detail.to_json());
}

HttpResponse
SessionApi::h_connect_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  auto bo = body->as_object();

  lock_guard<mutex> lk(_mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (p->state != State::Stopped) {
    return HttpResponse::error(409, "stop the pipeline before editing");
  }

  string from = string(bo.contains("from") ? bo.at("from").as_string("")
                                           : "");
  string to   = string(bo.contains("to") ? bo.at("to").as_string("") : "");
  unsigned from_port = static_cast<unsigned>(
      bo.contains("from_port") ? bo.at("from_port").as_uint(0) : 0);
  if (from.empty() || to.empty()) {
    return HttpResponse::error(400, "'from' and 'to' are required");
  }

  auto to_it = find_if(p->stages.begin(), p->stages.end(),
                       [&](const StageSpec& s) { return s.id == to; });
  if (to_it == p->stages.end()) {
    return HttpResponse::error(404, "no such consumer stage '" + to + "'");
  }
  if (find_if(p->stages.begin(), p->stages.end(),
              [&](const StageSpec& s) { return s.id == from; })
      == p->stages.end()) {
    return HttpResponse::error(404, "no such source stage '" + from + "'");
  }

  // Resolve the live source stage to range-check from_port and to read
  // payload types for the agreement check below.
  Stage* lfrom = live_stage_(*p, from);
  if (lfrom && from_port >= lfrom->num_oports()) {
    return HttpResponse::error(
        400, "source '" + from + "' has no oport " + to_string(from_port)
             + " (it has " + to_string(lfrom->num_oports()) + ")");
  }

  const size_t n = to_it->iports.size();
  const bool has_tp = bo.contains("to_port");
  const unsigned tp = static_cast<unsigned>(
      has_tp ? bo.at("to_port").as_uint(0) : 0);
  // repoint = fill an EXISTING port index (including one currently
  // disconnected). Otherwise the connection targets `to_port`, padding
  // any positions up to it with disconnected iports (a gap), or -- with
  // no explicit to_port -- appends at the current end.
  const bool repoint = has_tp && tp < n;
  const unsigned target_port = has_tp ? tp : static_cast<unsigned>(n);

  // Beat-type agreement check up front. iport_payload_type/
  // oport_payload_type are pure functions of the port index (the
  // stage's static declaration), so this is valid even for an append's
  // not-yet-existing target_port -- and checking before any mutation
  // means there is nothing to roll back on a mismatch.
  if (Stage* lt = live_stage_(*p, to)) {
    const type_info* a = lfrom ? lfrom->oport_payload_type(from_port)
                               : nullptr;
    const type_info* b = lt->iport_payload_type(target_port);
    if (!compatible_(a, b)) {
      return HttpResponse::error(
          400, "Beat type mismatch on iport " + to_string(target_port) +
               " of '" + to + "': " + demangle_(b) +
               " cannot accept " + demangle_(a));
    }
    // Deeper check: payload tags (finer constraint on the beat type).
    const std::string_view atg =
        lfrom ? lfrom->oport_payload_tags(from_port) : std::string_view{};
    const std::string_view btg = lt->iport_payload_tags(target_port);
    if (!port_tags_compatible(atg, btg)) {
      return HttpResponse::error(
          400, "Payload tag mismatch on iport " + to_string(target_port) +
               " of '" + to + "': {" + string(btg) +
               "} cannot accept {" + string(atg) + "}");
    }
  }

  if (repoint) {
    // In-place re-point of an existing input via move_iport (no
    // rebuild). Keep the spec binding in lock-step so a save still
    // round-trips, and revert both on rejection.
    pair<string, unsigned> old = to_it->iports[tp];
    to_it->iports[tp] = { from, from_port };
    if (!p->handle || !p->handle->move_iport(to, tp, from, from_port)) {
      to_it->iports[tp] = old;
      string e2;
      materialize_(*p, e2);
      return HttpResponse::error(400, "could not connect '" + from +
          "' -> '" + to + "' iport " + to_string(tp));
    }
  } else {
    // Append a new input (or fill a gap at/after the current end) --
    // changes arity, so rebuild from the spec. Pad the positions
    // between the current end and target_port with disconnected iports.
    auto saved = to_it->iports;
    while (to_it->iports.size() < target_port) {
      to_it->iports.emplace_back(string(), 0u);
    }
    to_it->iports.emplace_back(from, from_port);
    string err;
    if (!materialize_(*p, err)) {
      to_it->iports = std::move(saved);
      string e2;
      materialize_(*p, e2);
      return HttpResponse::error(400, err);
    }
  }

  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr_(p->id));
  d.insert("state", fstr_(state_name_(p->state)));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
SessionApi::h_disconnect_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  auto bo = body->as_object();

  lock_guard<mutex> lk(_mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (p->state != State::Stopped) {
    return HttpResponse::error(409, "stop the pipeline before editing");
  }

  string to = string(bo.contains("to") ? bo.at("to").as_string("") : "");
  if (to.empty()) { return HttpResponse::error(400, "'to' is required"); }
  if (!bo.contains("to_port")) {
    return HttpResponse::error(400, "'to_port' is required");
  }
  unsigned tp = static_cast<unsigned>(bo.at("to_port").as_uint(0));

  auto to_it = find_if(p->stages.begin(), p->stages.end(),
                       [&](const StageSpec& s) { return s.id == to; });
  if (to_it == p->stages.end()) {
    return HttpResponse::error(404, "no such consumer stage '" + to + "'");
  }
  if (tp >= to_it->iports.size()) {
    return HttpResponse::error(
        400, "iport " + to_string(tp) + " out of range ('" + to +
             "' has " + to_string(to_it->iports.size()) + " input(s))");
  }

  // Disconnect turns the input into a GAP (empty source) IN PLACE,
  // preserving the positional index of every later input -- iports are
  // positional, so a disconnected port must keep its slot rather than
  // shift its neighbours. move_iport accepts an empty src_id (null
  // source), so no rebuild is needed.
  pair<string, unsigned> old = to_it->iports[tp];
  to_it->iports[tp] = { string(), 0u };
  if (!p->handle || !p->handle->move_iport(to, tp, "", 0)) {
    to_it->iports[tp] = old;
    string e2;
    materialize_(*p, e2);
    return HttpResponse::error(
        400, "could not disconnect iport " + to_string(tp) + " of '"
             + to + "'");
  }
  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr_(p->id));
  d.insert("state", fstr_(state_name_(p->state)));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
SessionApi::h_get_stage_config_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_mu);
  auto pit = req.params.find("id");
  auto sit = req.params.find("sid");
  Pipe* p = (pit != req.params.end()) ? find_(pit->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (sit == req.params.end()) {
    return HttpResponse::error(400, "missing stage id");
  }
  const Stage* s = nullptr;
  if (p->handle && p->handle->valid()) {
    if (Pipeline* pl = live_pipeline_(*p->handle)) {
      for (auto it = pl->begin(); it != pl->end(); ++it) {
        if (const Stage* cs = dynamic_cast<const Stage*>(*it)) {
          if (cs->id() == sit->second) { s = cs; break; }
        }
      }
    }
  }
  if (!s) { return HttpResponse::error(404, "no such stage"); }

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("id", fstr_(s->id()));
  oo.insert("type", fstr_(s->type_name()));
  oo.insert("editable", FlexData::make_bool(p->state == State::Stopped));
  oo.insert("schema", s->config_schema());
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_set_stage_config_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "config body must be a JSON object");
  }
  lock_guard<mutex> lk(_mu);
  auto pit = req.params.find("id");
  auto sit = req.params.find("sid");
  Pipe* p = (pit != req.params.end()) ? find_(pit->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (sit == req.params.end()) {
    return HttpResponse::error(400, "missing stage id");
  }
  if (p->state != State::Stopped) {
    return HttpResponse::error(409, "stop the pipeline before editing");
  }
  auto target = find_if(p->stages.begin(), p->stages.end(),
                        [&](const StageSpec& s) { return s.id == sit->second; });
  if (target == p->stages.end()) {
    return HttpResponse::error(404, "no such stage");
  }
  FlexData old = target->config;
  target->config = *body;
  string err;
  if (!materialize_(*p, err)) {
    target->config = old;
    string e2;
    materialize_(*p, e2);
    return HttpResponse::error(400, err);
  }
  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr_(p->id));
  d.insert("state", fstr_(state_name_(p->state)));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
SessionApi::h_io_console_(const HttpRequest& req)
{
  if (!_ui) { return HttpResponse::error(404, "user I/O not available"); }
  uint64_t since = 0;
  string sv = query_param_(req.query, "since");
  if (!sv.empty()) {
    since = strtoull(sv.c_str(), nullptr, 10);
  }
  // Cap the per-response line count. The default keeps the very
  // first poll after a tab switch from delivering 8 k+ lines at once
  // (which then takes the browser seconds to render synchronously).
  // The client preserves `latest` between polls, so a long backlog
  // catches up over a handful of ticks instead of one giant pause.
  std::size_t limit = 2048;
  string lv = query_param_(req.query, "limit");
  if (!lv.empty()) {
    limit = static_cast<std::size_t>(strtoull(lv.c_str(), nullptr, 10));
  }
  uint64_t latest = 0;
  auto lines = _ui->console_since(since, &latest, limit);

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("latest", FlexData::make_uint(latest));
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (const auto& l : lines) {
    FlexData e = FlexData::make_object();
    auto eo = e.as_object();
    eo.insert("seq", FlexData::make_uint(l.seq));
    eo.insert("level", fstr_(l.level));
    eo.insert("text", fstr_(l.text));
    eo.insert("open", FlexData::make_bool(l.open));
    a.push_back(std::move(e));
  }
  oo.insert("lines", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_io_pending_(const HttpRequest&)
{
  if (!_ui) { return HttpResponse::error(404, "user I/O not available"); }
  uint64_t id = 0;
  string   prompt;
  bool     masked  = false;
  bool     media   = false;
  bool     pending = _ui->pending_input(&id, &prompt, &masked, &media);

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("pending", FlexData::make_bool(pending));
  if (pending) {
    oo.insert("id", FlexData::make_uint(id));
    oo.insert("prompt", fstr_(prompt));
    oo.insert("masked", FlexData::make_bool(masked));
    // getmedialine request: the client offers attach/drop controls and
    // embeds attachments as base64 media-line markers in the answer.
    oo.insert("media", FlexData::make_bool(media));
  }
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_io_input_(const HttpRequest& req)
{
  if (!_ui) { return HttpResponse::error(404, "user I/O not available"); }
  auto body = parse_json_body_(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  auto bo = body->as_object();
  if (!bo.contains("id")) {
    return HttpResponse::error(400, "missing 'id'");
  }
  uint64_t id   = bo.at("id").as_uint(0);
  string   text = string(bo.contains("text")
                             ? bo.at("text").as_string("")
                             : string_view{});
  if (!_ui->submit_input(id, std::move(text))) {
    return HttpResponse::error(409, "no matching pending input request");
  }
  return HttpResponse::ok();
}

HttpResponse
SessionApi::h_io_clear_(const HttpRequest&)
{
  if (!_ui) { return HttpResponse::error(404, "user I/O not available"); }
  _ui->clear_console();
  return HttpResponse::ok();
}

HttpResponse
SessionApi::h_system_status_(const HttpRequest&)
{
  FlexData o = _status->query();
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_hls_streams_(const HttpRequest&)
{
  lock_guard<mutex> lk(_mu);
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();

  for (auto& up : _pipes) {
    const Pipe& p = *up;
    // Only a launched pipeline has a live HTTP server behind the
    // stage; a stopped one serves nothing. Paused keeps serving the
    // last playlist, so include it too.
    if (p.state == State::Stopped) { continue; }
    if (!p.handle || !p.handle->valid()) { continue; }
    Pipeline* pl = live_pipeline_(*p.handle);
    if (!pl) { continue; }
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
      eo.insert("pipeline", fstr_(p.id));
      eo.insert("stage", fstr_(s->id()));
      eo.insert("state", fstr_(state_name_(p.state)));
      eo.insert("playlist_name",
                fstr_(get_str("playlist_name", "stream.m3u8")));
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
        bind = _sctx ? _sctx->web_ui_bind_address() : string();
        if (bind.empty()) {
          const string lan = netx::primary_ipv4();
          bind = lan.empty() ? string("0.0.0.0") : lan;
        }
      }
      eo.insert("bind_address", fstr_(bind));
      // Whether this broadcast carries an audio track: iport 1 (the strict
      // audio role) is wired to a producer. The UI uses it to auto-unmute the
      // player when a viewer attaches (a video-only stream stays muted).
      const auto& ie = s->iport_edges();
      const bool has_audio = ie.size() > 1 && ie[1].v != nullptr;
      eo.insert("audio", FlexData::make_bool(has_audio));
      a.push_back(std::move(e));
    }
  }
  oo.insert("streams", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_preview_streams_(const HttpRequest&)
{
  lock_guard<mutex> lk(_mu);
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();

  for (auto& up : _pipes) {
    const Pipe& p = *up;
    // A launched (or paused) pipeline has live stages behind its channels;
    // a stopped one serves nothing.
    if (p.state == State::Stopped) { continue; }
    if (!p.handle || !p.handle->valid()) { continue; }
    Pipeline* pl = live_pipeline_(*p.handle);
    if (!pl) { continue; }
    for (auto it = pl->begin(); it != pl->end(); ++it) {
      const Stage* s = dynamic_cast<const Stage*>(*it);
      if (!s || string(s->type_name()) != "preview") { continue; }

      // The channel exposes best-effort media hints (populated once the
      // first frame flows); the authoritative config reaches the browser
      // in the stream's own config frame, so a not-yet-started stream is
      // still listed and playable.
      const PreviewSource* src = dynamic_cast<const PreviewSource*>(s);
      std::shared_ptr<PreviewChannel> ch =
          src ? src->preview_channel() : nullptr;

      // `title` config -> picker label; fall back to the stage id.
      string title;
      for (const auto& pr : s->config_params()) {
        if (pr.key == "title") {
          title = string(pr.current_value.as_string(""));
          break;
        }
      }
      if (title.empty()) { title = s->id(); }

      FlexData e = FlexData::make_object();
      auto eo = e.as_object();
      eo.insert("pipeline", fstr_(p.id));
      eo.insert("stage", fstr_(s->id()));
      eo.insert("state", fstr_(state_name_(p.state)));
      eo.insert("title", fstr_(title));
      eo.insert("video",
                FlexData::make_bool(ch ? ch->has_video() : false));
      eo.insert("audio",
                FlexData::make_bool(ch ? ch->has_audio() : false));
      eo.insert("width", FlexData::make_uint(
                    static_cast<uint64_t>(ch ? ch->width() : 0)));
      eo.insert("height", FlexData::make_uint(
                    static_cast<uint64_t>(ch ? ch->height() : 0)));
      a.push_back(std::move(e));
    }
  }
  oo.insert("streams", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

void
SessionApi::h_preview_ws_(const HttpRequest& req, WebSocket& ws)
{
  string pipeline;
  string stage;
  {
    auto pit = req.params.find("pipeline");
    auto sit = req.params.find("stage");
    if (pit != req.params.end()) { pipeline = pit->second; }
    if (sit != req.params.end()) { stage = sit->second; }
  }

  // Resolve the live stage's channel under _mu, then release it: the relay
  // loop must not hold the API lock for the connection's whole life. The
  // shared_ptr keeps the channel alive even if the pipeline is stopped
  // (the stage's teardown then close()s it, ending this WebSocket).
  std::shared_ptr<PreviewChannel> ch;
  {
    lock_guard<mutex> lk(_mu);
    Pipe* p = find_(pipeline);
    if (p && p->handle && p->handle->valid()) {
      Stage* s = live_stage_(*p, stage);
      if (s) {
        auto* src = dynamic_cast<PreviewSource*>(s);
        if (src) { ch = src->preview_channel(); }
      }
    }
  }
  if (!ch) {
    return;   // no such live stage: let the WebSocket close.
  }

  auto sub = ch->subscribe();
  for (;;) {
    auto blob = ch->wait_frame(sub, 1000);
    if (blob) {
      if (!ws.send_binary(blob->data(), blob->size())) {
        break;   // client gone (send failed)
      }
    } else if (ch->closed()) {
      break;     // stream ended (pipeline stopped)
    } else if (!ws.alive()) {
      break;     // client gone
    }
    // else: a 1s wait timed out with nothing queued -> re-check + retry.
  }
  ch->unsubscribe(sub);
}

// -------------------------------------------------------------------
// Profiler
// -------------------------------------------------------------------

std::string
SessionApi::profiler_dump_json_(std::string& err)
{
  namespace fs = std::filesystem;
  const fs::path dir = vpipe::temp_root();
  // One temp file per process; _mu serializes profiler ops so reuse is
  // safe. ".json" extension selects dump_profiling's pretty-JSON form.
  fs::path tmp = dir / ("vpipe-webui-profiler-"
                        + std::to_string(static_cast<long>(::getpid()))
                        + ".json");
  Status st = _session->dump_profiling(tmp.string());
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
SessionApi::profiler_status_doc_() const
{
  const bool en = _sctx && _sctx->profiling_enabled();
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("enabled", FlexData::make_bool(en));
  oo.insert("max_events_per_thread",
            FlexData::make_uint(
                _sctx ? _sctx->profiling_max_events_per_thread() : 0u));
  oo.insert("has_data",
            FlexData::make_bool(en || !_profiler_snapshot.empty()));
  return o;
}

HttpResponse
SessionApi::h_profiler_start_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  unsigned max_events = 65536;
  if (body->is_object() && body->as_object().contains("max_events")) {
    const long n =
        static_cast<long>(body->as_object().at("max_events").as_int(0));
    if (n > 0) {
      max_events = static_cast<unsigned>(std::min<long>(n, 1L << 24));
    }
  }
  lock_guard<mutex> lk(_mu);
  Status st = _session->enable_profiling(max_events);
  if (st.code != 0) {
    return HttpResponse::error(
        500, "enable_profiling failed (max_events must be > 0)");
  }
  _profiler_snapshot.clear();       // previous capture is now stale
  return HttpResponse::json(200, profiler_status_doc_().to_json());
}

HttpResponse
SessionApi::h_profiler_stop_(const HttpRequest&)
{
  lock_guard<mutex> lk(_mu);
  if (_sctx && _sctx->profiling_enabled()) {
    // Snapshot BEFORE disabling -- disable_profiling frees the buffers.
    std::string err;
    std::string js = profiler_dump_json_(err);
    if (!js.empty()) { _profiler_snapshot = std::move(js); }
    _session->disable_profiling();
  }
  return HttpResponse::json(200, profiler_status_doc_().to_json());
}

HttpResponse
SessionApi::h_profiler_reset_(const HttpRequest&)
{
  lock_guard<mutex> lk(_mu);
  // Drop the retained snapshot (the stopped-state data). If a capture
  // is live, re-arm it: enable_profiling reallocates the buffers, so the
  // accumulated events are cleared and the anchor restarts -- a clean
  // slate without having to Stop first.
  _profiler_snapshot.clear();
  if (_sctx && _sctx->profiling_enabled()) {
    _session->enable_profiling(_sctx->profiling_max_events_per_thread());
  }
  return HttpResponse::json(200, profiler_status_doc_().to_json());
}

HttpResponse
SessionApi::h_profiler_status_(const HttpRequest&)
{
  lock_guard<mutex> lk(_mu);
  return HttpResponse::json(200, profiler_status_doc_().to_json());
}

HttpResponse
SessionApi::h_profiler_data_(const HttpRequest&)
{
  lock_guard<mutex> lk(_mu);
  if (_sctx && _sctx->profiling_enabled()) {
    std::string err;
    std::string js = profiler_dump_json_(err);
    if (js.empty()) {
      return HttpResponse::error(500, err.empty() ? "dump failed" : err);
    }
    return HttpResponse::json(200, js);
  }
  if (!_profiler_snapshot.empty()) {
    return HttpResponse::json(200, _profiler_snapshot);
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

HttpResponse
SessionApi::h_cwd_pipelines_(const HttpRequest&)
{
  // Returns the server-process cwd plus the .vpipeline filenames at
  // its top level. The Load-Pipeline dialog uses this to populate a
  // <datalist>, giving the browser native autocomplete-as-you-type
  // over the available files. Non-recursive on purpose -- it's a
  // quick picker, not a file browser. A user pointing to a path in
  // another directory just types it directly.
  namespace fs = std::filesystem;
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  std::error_code ec;
  fs::path cwd = fs::current_path(ec);
  oo.insert("cwd", fstr_(ec ? string() : cwd.string()));
  FlexData arr = FlexData::make_array();
  auto av = arr.as_array();
  if (!ec) {
    vector<string> names;
    for (auto it = fs::directory_iterator(cwd, ec);
         !ec && it != fs::directory_iterator();
         it.increment(ec)) {
      std::error_code stat_ec;
      if (!it->is_regular_file(stat_ec) || stat_ec) { continue; }
      const auto name = it->path().filename().string();
      if (name.size() > 10
          && name.compare(name.size() - 10, 10, ".vpipeline") == 0) {
        names.push_back(name);
      }
    }
    std::sort(names.begin(), names.end());
    for (const auto& n : names) {
      av.push_back(FlexData::make_string(n));
    }
  }
  oo.insert("files", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_fs_list_(const HttpRequest& req)
{
  // Directory listing for the web-ui file open/save dialog. Operates in
  // the session's filesystem namespace: when the sandbox is active the
  // client sees a chroot-like virtual tree whose root is "/" (real host
  // paths never leave the server); otherwise the virtual path IS the
  // host path. Read-only, non-recursive: one directory per call.
  namespace fs = std::filesystem;
  const bool sb = _sctx && _sctx->fs_sandboxed();
  auto strip = [](string s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == '/' || s[i] == '\\')) { ++i; }
    return s.substr(i);
  };

  // The client encodes the path with encodeURIComponent (so '/', spaces
  // and unicode survive the query string) -- decode it before use.
  string   want = url_decode_(query_param_(req.query, "path"));
  string   vpath;      // clean virtual path echoed to the client
  fs::path real;       // real directory actually enumerated
  std::error_code ec;
  if (sb) {
    if (want.empty()) { want = "/"; }
    // Normalise inside the virtual space; lexically_normal collapses "."
    // and clamps ".." at the root so the browser cannot point above it.
    fs::path v =
        fs::path("/" + strip(want)).lexically_normal();
    vpath = v.generic_string();
    if (vpath.empty()) { vpath = "/"; }
    const string rel = strip(vpath);
    if (rel.empty()) {
      real = _sctx->fs_sandbox_root();
    } else {
      string cerr;
      real = _sctx->confine_path(vpath, /*for_write=*/false, &cerr);
      if (real.empty()) {
        return HttpResponse::error(
            400, "path '" + vpath + "' rejected: "
                     + (cerr.empty() ? "outside sandbox" : cerr));
      }
    }
  } else {
    fs::path v = want.empty() ? fs::current_path(ec) : fs::path(want);
    if (v.is_relative()) { v = fs::current_path(ec) / v; }
    fs::path canon = fs::weakly_canonical(v, ec);
    real  = (ec || canon.empty()) ? v.lexically_normal() : canon;
    vpath = real.generic_string();
  }

  std::error_code de;
  if (!fs::is_directory(real, de) || de) {
    return HttpResponse::error(400, "not a directory: " + vpath);
  }

  // Whitelisted pass-through prefixes ("mounts") the browser can jump
  // to, and whether `real` sits AT one -- if so, "up" returns to the
  // sandbox home rather than a re-rooted parent confine would reject.
  std::vector<fs::path> wl = _sctx ? _sctx->fs_whitelist()
                                   : std::vector<fs::path>{};
  bool at_wl_root = false;
  if (!wl.empty()) {
    std::error_code ce;
    fs::path rc = fs::weakly_canonical(real, ce);
    if (ce || rc.empty()) { rc = real.lexically_normal(); }
    for (const auto& w : wl) {
      if (rc == w) { at_wl_root = true; break; }
    }
  }

  // Parent (virtual). Empty string == "already at the root, no parent".
  string parent;
  if (strip(vpath).empty()) {
    parent = "";                       // sandbox root
  } else if (sb && at_wl_root) {
    parent = "/";                      // a granted mount -> up == home
  } else {
    fs::path pp = fs::path(vpath).parent_path();
    if (pp.empty() || pp == fs::path(vpath)) {
      parent = "";                     // native filesystem root
    } else {
      parent = pp.generic_string();
      if (sb && parent.empty()) { parent = "/"; }
    }
  }

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("sandboxed", FlexData::make_bool(sb));
  oo.insert("path", fstr_(vpath));
  oo.insert("parent", fstr_(parent));
  // Granted mounts (empty unless --white-list-path was given). Surfaced
  // ONLY at the sandbox ROOT: they are shortcuts to jump OUT of the sandbox
  // home, so repeating them inside every directory listing -- and inside a
  // mount itself -- is just noise.
  FlexData mounts = FlexData::make_array();
  if (sb && strip(vpath).empty()) {
    auto ma = mounts.as_array();
    for (const auto& w : wl) {
      FlexData mo = FlexData::make_object();
      auto x = mo.as_object();
      const std::string base = w.filename().string();
      x.insert("name", fstr_(base.empty() ? w.generic_string() : base));
      x.insert("path", fstr_(w.generic_string()));
      ma.push_back(std::move(mo));
    }
  }
  oo.insert("mounts", std::move(mounts));

  struct Ent { string name; bool dir; uint64_t size; };
  vector<Ent> ents;
  for (auto it = fs::directory_iterator(
           real, fs::directory_options::skip_permission_denied, ec);
       !ec && it != fs::directory_iterator(); it.increment(ec)) {
    std::error_code se;
    const bool isdir = it->is_directory(se);
    if (se) { continue; }
    Ent e;
    e.name = it->path().filename().string();
    if (e.name.empty()) { continue; }
    e.dir  = isdir;
    e.size = 0;
    if (!isdir) {
      std::error_code re;
      if (!it->is_regular_file(re) || re) { continue; }  // skip non-files
      std::error_code ze;
      const auto sz = fs::file_size(*it, ze);
      e.size = ze ? 0 : static_cast<uint64_t>(sz);
    }
    ents.push_back(std::move(e));
  }
  auto ci_less = [](const string& a, const string& b) {
    return std::lexicographical_compare(
        a.begin(), a.end(), b.begin(), b.end(),
        [](unsigned char x, unsigned char y) {
          return std::tolower(x) < std::tolower(y);
        });
  };
  std::sort(ents.begin(), ents.end(), [&](const Ent& a, const Ent& b) {
    if (a.dir != b.dir) { return a.dir; }   // directories first
    return ci_less(a.name, b.name);
  });

  FlexData arr = FlexData::make_array();
  auto av = arr.as_array();
  av.reserve(ents.size());
  for (const auto& e : ents) {
    FlexData eo = FlexData::make_object();
    auto x = eo.as_object();
    x.insert("name", fstr_(e.name));
    x.insert("dir",  FlexData::make_bool(e.dir));
    if (!e.dir) { x.insert("size", FlexData::make_uint(e.size)); }
    av.push_back(std::move(eo));
  }
  oo.insert("entries", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_io_limit_get_(const HttpRequest&)
{
  if (!_ui) { return HttpResponse::error(404, "user I/O not available"); }
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("max_console",
      FlexData::make_uint(static_cast<uint64_t>(_ui->max_console())));
  oo.insert("min", FlexData::make_uint(
      static_cast<uint64_t>(webui::WebUiDelegate::kMinMaxConsole)));
  oo.insert("max", FlexData::make_uint(
      static_cast<uint64_t>(webui::WebUiDelegate::kMaxMaxConsole)));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_io_limit_set_(const HttpRequest& req)
{
  if (!_ui) { return HttpResponse::error(404, "user I/O not available"); }
  auto body = parse_json_body_(req);
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
  _ui->set_max_console(static_cast<size_t>(n));
  FlexData o = FlexData::make_object();
  o.as_object().insert("max_console",
      FlexData::make_uint(static_cast<uint64_t>(_ui->max_console())));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_log_console_(const HttpRequest& req)
{
  if (!_log) { return HttpResponse::error(404, "session log not available"); }
  uint64_t since = 0;
  string sv = query_param_(req.query, "since");
  if (!sv.empty()) {
    since = strtoull(sv.c_str(), nullptr, 10);
  }
  std::size_t limit = 2048;
  string lv = query_param_(req.query, "limit");
  if (!lv.empty()) {
    limit = static_cast<std::size_t>(strtoull(lv.c_str(), nullptr, 10));
  }
  uint64_t latest = 0;
  auto lines = _log->console_since(since, &latest, limit);

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("latest", FlexData::make_uint(latest));
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (const auto& l : lines) {
    FlexData e = FlexData::make_object();
    auto eo = e.as_object();
    eo.insert("seq", FlexData::make_uint(l.seq));
    eo.insert("level", fstr_(l.level));
    eo.insert("text", fstr_(l.text));
    a.push_back(std::move(e));
  }
  oo.insert("lines", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_log_clear_(const HttpRequest&)
{
  if (!_log) { return HttpResponse::error(404, "session log not available"); }
  _log->clear_console();
  return HttpResponse::ok();
}

namespace {
// Debug-level choices the UI dropdown offers (most severe first). The
// "always" sentinel is not a valid threshold and is omitted.
const char* const kLogLevels[] = {
    "error", "warn", "info", "normal", "verbose", "debug" };
}  // namespace

HttpResponse
SessionApi::h_log_level_get_(const HttpRequest&)
{
  if (!_log) { return HttpResponse::error(404, "session log not available"); }
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  string cur = to_cstr(_log->threshold());
  for (char& c : cur) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  oo.insert("level", fstr_(cur));
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (const char* lv : kLogLevels) { a.push_back(fstr_(lv)); }
  oo.insert("levels", std::move(arr));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_log_level_set_(const HttpRequest& req)
{
  if (!_log) { return HttpResponse::error(404, "session log not available"); }
  auto body = parse_json_body_(req);
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
  _log->set_threshold(parsed);
  FlexData o = FlexData::make_object();
  string cur = to_cstr(parsed);
  for (char& c : cur) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  o.as_object().insert("level", fstr_(cur));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_log_limit_get_(const HttpRequest&)
{
  if (!_log) { return HttpResponse::error(404, "session log not available"); }
  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("max_log",
      FlexData::make_uint(static_cast<uint64_t>(_log->max_log())));
  oo.insert("min", FlexData::make_uint(
      static_cast<uint64_t>(webui::WebUiLogDelegate::kMinMaxLog)));
  oo.insert("max", FlexData::make_uint(
      static_cast<uint64_t>(webui::WebUiLogDelegate::kMaxMaxLog)));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_log_limit_set_(const HttpRequest& req)
{
  if (!_log) { return HttpResponse::error(404, "session log not available"); }
  auto body = parse_json_body_(req);
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
  _log->set_max_log(static_cast<size_t>(n));
  FlexData o = FlexData::make_object();
  o.as_object().insert("max_log",
      FlexData::make_uint(static_cast<uint64_t>(_log->max_log())));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
SessionApi::h_db_list_(const HttpRequest&)
{
  DbBrowser db(_sctx);
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
    lock_guard<mutex> lk(_mu);
    deletable = !any_pipeline_active_();
  }
  if (doc.is_object()) {
    doc.as_object().insert("deletable", FlexData::make_bool(deletable));
  }
  return HttpResponse::json(200, doc.to_json());
}

HttpResponse
SessionApi::h_db_delete_key_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  // Hold _mu across the check + write so a concurrent launch can't slip
  // in between (launch also takes _mu).
  lock_guard<mutex> lk(_mu);
  if (any_pipeline_active_()) {
    return HttpResponse::error(
        409, "stop all pipelines before modifying the database");
  }
  DbBrowser db(_sctx);
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
SessionApi::h_db_drop_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  lock_guard<mutex> lk(_mu);
  if (any_pipeline_active_()) {
    return HttpResponse::error(
        409, "stop all pipelines before modifying the database");
  }
  DbBrowser db(_sctx);
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
SessionApi::h_db_keys_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  DbBrowser db(_sctx);
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
SessionApi::h_db_scan_stream_(const HttpRequest& req, ResponseStream& rs)
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

  auto body = parse_json_body_(req);
  if (!body) { emit_err("invalid JSON body"); return; }

  DbBrowser db(_sctx);
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
SessionApi::h_models_installed_(const HttpRequest&)
{
  string err;
  FlexData doc;
  {
    lock_guard<mutex> dlk(_db_mu);
    doc = list_installed_models(_sctx, "models", err);
  }
  if (!err.empty()) { return HttpResponse::error(500, err); }
  return HttpResponse::json(200, doc.to_json());
}

HttpResponse
SessionApi::h_db_value_(const HttpRequest& req)
{
  auto body = parse_json_body_(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  DbBrowser db(_sctx);
  string err;
  FlexData doc;
  {
    lock_guard<mutex> dlk(_db_mu);
    doc = db.read_value(*body, err);
  }
  if (!err.empty()) { return HttpResponse::error(400, err); }
  return HttpResponse::json(200, doc.to_json());
}

void
SessionApi::register_routes(HttpServer& s)
{
  s.route("GET", "/api/health",
          [](const HttpRequest&) {
            // Per-process instance token. The web-ui client polls
            // /api/health and hard-reloads a (localhost) page when this
            // changes -- i.e. when the server has been restarted -- so a
            // stale page refreshes itself against the new server.
            static const std::string inst = [] {
              static const char hex[] = "0123456789abcdef";
              std::random_device rd;
              std::uniform_int_distribution<int> d(0, 15);
              std::string t;
              t.reserve(16);
              for (int i = 0; i < 16; ++i) { t.push_back(hex[d(rd)]); }
              return t;
            }();
            return HttpResponse::json(
                200, "{\"ok\":true,\"instance\":\"" + inst + "\"}");
          });
  s.route("GET", "/api/stage-types",
          [this](const HttpRequest& r) { return h_stage_types_(r); });
  s.route("GET", "/api/pipelines",
          [this](const HttpRequest& r) { return h_list_pipelines_(r); });
  s.route("POST", "/api/pipelines",
          [this](const HttpRequest& r) { return h_create_pipeline_(r); });
  s.route("POST", "/api/pipelines/:id/rename",
          [this](const HttpRequest& r) { return h_rename_pipeline_(r); });
  s.route("POST", "/api/pipelines/load",
          [this](const HttpRequest& r) { return h_load_pipeline_(r); });
  s.route("GET", "/api/pipelines/:id",
          [this](const HttpRequest& r) { return h_get_pipeline_(r); });
  s.route("POST", "/api/pipelines/:id/save",
          [this](const HttpRequest& r) { return h_save_pipeline_(r); });
  s.route("POST", "/api/pipelines/:id/unload",
          [this](const HttpRequest& r) { return h_unload_pipeline_(r); });
  s.route("POST", "/api/pipelines/:id/launch",
          [this](const HttpRequest& r) { return h_launch_pipeline_(r); });
  s.route("POST", "/api/pipelines/:id/pause",
          [this](const HttpRequest& r) { return h_pause_pipeline_(r); });
  s.route("POST", "/api/pipelines/:id/stop",
          [this](const HttpRequest& r) { return h_stop_pipeline_(r); });
  s.route("GET", "/api/pipelines/:id/buffer-status",
          [this](const HttpRequest& r) { return h_buffer_status_(r); });
  s.route("POST", "/api/pipelines/:id/stages",
          [this](const HttpRequest& r) { return h_insert_stage_(r); });
  s.route("DELETE", "/api/pipelines/:id/stages/:sid",
          [this](const HttpRequest& r) { return h_remove_stage_(r); });
  s.route("POST", "/api/pipelines/:id/stages/:sid/rename",
          [this](const HttpRequest& r) { return h_rename_stage_(r); });
  s.route("POST", "/api/pipelines/:id/stages/:sid/duplicate",
          [this](const HttpRequest& r) { return h_duplicate_stage_(r); });
  s.route("POST", "/api/pipelines/:id/connect",
          [this](const HttpRequest& r) { return h_connect_(r); });
  s.route("POST", "/api/pipelines/:id/disconnect",
          [this](const HttpRequest& r) { return h_disconnect_(r); });
  s.route("GET", "/api/pipelines/:id/stages/:sid/config",
          [this](const HttpRequest& r) { return h_get_stage_config_(r); });
  s.route("PUT", "/api/pipelines/:id/stages/:sid/config",
          [this](const HttpRequest& r) { return h_set_stage_config_(r); });

  // User I/O console + interactive getline rendezvous. Registered only
  // when a WebUiDelegate backs the session (otherwise the routes 404).
  if (_ui) {
    s.route("GET", "/api/io/console",
            [this](const HttpRequest& r) { return h_io_console_(r); });
    s.route("GET", "/api/io/pending",
            [this](const HttpRequest& r) { return h_io_pending_(r); });
    s.route("POST", "/api/io/input",
            [this](const HttpRequest& r) { return h_io_input_(r); });
    s.route("POST", "/api/io/clear",
            [this](const HttpRequest& r) { return h_io_clear_(r); });
    s.route("GET", "/api/io/limit",
            [this](const HttpRequest& r) { return h_io_limit_get_(r); });
    s.route("PUT", "/api/io/limit",
            [this](const HttpRequest& r) { return h_io_limit_set_(r); });
  }

  // Session log console + debug-level control. Registered only when a
  // WebUiLogDelegate backs the session (otherwise the routes 404).
  if (_log) {
    s.route("GET", "/api/log/console",
            [this](const HttpRequest& r) { return h_log_console_(r); });
    s.route("POST", "/api/log/clear",
            [this](const HttpRequest& r) { return h_log_clear_(r); });
    s.route("GET", "/api/log/level",
            [this](const HttpRequest& r) { return h_log_level_get_(r); });
    s.route("PUT", "/api/log/level",
            [this](const HttpRequest& r) { return h_log_level_set_(r); });
    s.route("GET", "/api/log/limit",
            [this](const HttpRequest& r) { return h_log_limit_get_(r); });
    s.route("PUT", "/api/log/limit",
            [this](const HttpRequest& r) { return h_log_limit_set_(r); });
  }

  // Read-only database browser (always available; reads the session's
  // LMDB env).
  s.route("GET", "/api/system/status",
          [this](const HttpRequest& r) { return h_system_status_(r); });
  s.route("GET", "/api/startup-checks",
          [this](const HttpRequest& r) { return h_startup_checks_(r); });
  s.route("GET", "/api/i18n",
          [this](const HttpRequest& r) { return h_i18n_get_(r); });
  s.route("PUT", "/api/i18n",
          [this](const HttpRequest& r) { return h_i18n_set_(r); });

  // Active HLS streams (live "hls-broadcast" stages) for the User I/O
  // workspace's HLS video view.
  s.route("GET", "/api/hls/streams",
          [this](const HttpRequest& r) { return h_hls_streams_(r); });

  // Low-latency preview: discovery (buffered) + the long-lived binary
  // stream (single origin, auth-gated like every /api/* route). The stream
  // route has more path segments than the discovery route, so the two
  // never collide.
  s.route("GET", "/api/preview/streams",
          [this](const HttpRequest& r) { return h_preview_streams_(r); });
  s.route_ws("/api/preview/:pipeline/:stage/ws",
          [this](const HttpRequest& r, WebSocket& ws) {
            h_preview_ws_(r, ws);
          });

  // Performance profiler capture control + timeline retrieval.
  s.route("POST", "/api/profiler/start",
          [this](const HttpRequest& r) { return h_profiler_start_(r); });
  s.route("POST", "/api/profiler/stop",
          [this](const HttpRequest& r) { return h_profiler_stop_(r); });
  s.route("POST", "/api/profiler/reset",
          [this](const HttpRequest& r) { return h_profiler_reset_(r); });
  s.route("GET", "/api/profiler/status",
          [this](const HttpRequest& r) { return h_profiler_status_(r); });
  s.route("GET", "/api/profiler/data",
          [this](const HttpRequest& r) { return h_profiler_data_(r); });

  s.route("GET", "/api/cwd-pipelines",
          [this](const HttpRequest& r) { return h_cwd_pipelines_(r); });

  // Directory browsing for the file open/save dialog (sandbox-aware).
  s.route("GET", "/api/fs/list",
          [this](const HttpRequest& r) { return h_fs_list_(r); });

  s.route("GET", "/api/db/list",
          [this](const HttpRequest& r) { return h_db_list_(r); });
  s.route("POST", "/api/db/keys",
          [this](const HttpRequest& r) { return h_db_keys_(r); });
  s.route_stream("POST", "/api/db/scan",
          [this](const HttpRequest& r, ResponseStream& rs) {
            h_db_scan_stream_(r, rs);
          });
  s.route("POST", "/api/db/value",
          [this](const HttpRequest& r) { return h_db_value_(r); });
  s.route("POST", "/api/db/delete-key",
          [this](const HttpRequest& r) { return h_db_delete_key_(r); });
  s.route("POST", "/api/db/drop",
          [this](const HttpRequest& r) { return h_db_drop_(r); });
  s.route("GET", "/api/models/installed",
          [this](const HttpRequest& r) { return h_models_installed_(r); });
}

}
