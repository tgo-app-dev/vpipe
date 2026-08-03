#include "apps/web-ui/pipeline-api.h"
#include "apps/web-ui/api-common.h"

#include "common/graph.h"
#include "common/i18n.h"
#include "common/vertex.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/pipeline.h"
#include "pipeline/pipeline-handle-impl.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline-spec.h"
#include "pipeline/stage.h"
#include "pipeline/stage-registry.h"
#include "vpipe/session-intf.h"

#include <algorithm>
#include <cxxabi.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <mutex>
#include <span>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

// Recover the top-level "aux" object from a saved pipeline file (JSON).
// The pipeline core loader discards it, so we re-read the file here. Returns
// a null FlexData when the file isn't JSON, carries no aux, or can't be read
// -- the pipeline still loads, just without auxiliary data.
FlexData
aux_from_file_(const std::filesystem::path& real)
{
  try {
    ifstream f(real, ios::binary);
    if (!f) { return FlexData(); }
    string contents((istreambuf_iterator<char>(f)),
                    istreambuf_iterator<char>());
    if (blank(contents)) { return FlexData(); }
    const auto b = contents.find_first_not_of(" \t\r\n");
    if (b == string::npos || contents[b] != '{') {
      return FlexData();   // binary-FlexData spec (no JSON aux to recover)
    }
    FlexData doc = FlexData::from_json(contents);
    if (doc.is_object()) {
      auto o = doc.as_object();
      if (o.contains("aux")) {
        FlexData a = o.at("aux");
        if (a.is_object()) { return a; }
      }
    }
  } catch (const exception&) { /* leave aux empty */ }
  return FlexData();
}

}  // namespace

// Declared in pipeline-api.h: SystemApi walks live stages too.
Pipeline*
live_pipeline(const PipelineHandle& h)
{
  PipelineHandleImpl* impl = HandleAccess::impl(h);
  return impl ? impl->pipeline() : nullptr;
}

const char*
PipelineApi::state_name_(State s)
{
  switch (s) {
    case State::Stopped: return "stopped";
    case State::Running: return "running";
    case State::Paused:  return "paused";
  }
  return "stopped";
}

PipelineApi::Pipe*
PipelineApi::find_(const string& id)
{
  for (auto& up : _pipes) {
    if (up->id == id) { return up.get(); }
  }
  return nullptr;
}

bool
PipelineApi::any_active_locked() const   // caller holds _ctx.mu
{
  for (auto& up : _pipes) {
    if (up->state != State::Stopped) { return true; }
  }
  return false;
}

void
PipelineApi::reap_completed_()   // caller holds _ctx.mu
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
    _ctx.session->stop_pipeline(*p.handle);
    p.state = State::Stopped;
    if (_ctx.sctx) {
      _ctx.sctx->info(fmt("pipeline '{}' auto-stopped: all stages done",
                      p.id));
    }
  }
}


void
PipelineApi::dematerialize_(Pipe& p)
{
  if (p.handle && p.handle->valid()) {
    _ctx.session->unload_pipeline(*p.handle);
  }
  p.handle.reset();
}

bool
PipelineApi::materialize_(Pipe& p, string& err)
{
  dematerialize_(p);

  PipelineHandle h = _ctx.session->create_pipeline(p.id);
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
        _ctx.session->unload_pipeline(h);
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
      _ctx.session->unload_pipeline(h);
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
PipelineApi::pipe_summary_(const Pipe& p) const
{
  FlexData o = FlexData::make_object();
  auto v = o.as_object();
  v.insert("id", fstr(p.id));
  v.insert("state", fstr(state_name_(p.state)));
  v.insert("stage_count", FlexData::make_uint(p.stages.size()));
  v.insert("storage_path", fstr(p.storage_path));
  v.insert("materialized",
           FlexData::make_bool(p.handle && p.handle->valid()));
  return o;
}

Stage*
PipelineApi::live_stage_(const Pipe& p, const string& id) const
{
  if (!(p.handle && p.handle->valid())) { return nullptr; }
  Pipeline* pl = live_pipeline(*p.handle);
  if (!pl) { return nullptr; }
  for (auto it = pl->begin(); it != pl->end(); ++it) {
    if (Stage* s = dynamic_cast<Stage*>(*it)) {
      if (s->id() == id) { return s; }
    }
  }
  return nullptr;
}

FlexData
PipelineApi::graph_json_(const Pipe& p) const
{
  FlexData g = FlexData::make_object();
  auto go = g.as_object();
  FlexData nodes = FlexData::make_array();
  FlexData edges = FlexData::make_array();
  auto na = nodes.as_array();
  auto ea = edges.as_array();

  Pipeline* pl = (p.handle && p.handle->valid())
                     ? live_pipeline(*p.handle) : nullptr;
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
      no.insert("id", fstr(s->id()));
      no.insert("type", fstr(s->type_name()));
      // Non-empty when the stage's config is incomplete/invalid; the
      // stage still exists in the graph but is skipped at launch until
      // the problem is fixed (validation is deferred to launch).
      no.insert("config_error", fstr(s->config_error()));

      FlexData iports = FlexData::make_array();
      auto ia = iports.as_array();
      for (unsigned i = 0; i < s->num_iports(); ++i) {
        FlexData pe = FlexData::make_object();
        auto peo = pe.as_object();
        peo.insert("index", FlexData::make_uint(i));
        peo.insert("type", fstr(demangle_(s->iport_payload_type(i))));
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
        peo.insert("type", fstr(demangle_(s->oport_payload_type(i))));
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
        eo.insert("from", fstr(sit->second));
        eo.insert("from_port", FlexData::make_uint(e.p));
        eo.insert("to", fstr(s->id()));
        eo.insert("to_port", FlexData::make_uint(i));
        eo.insert("type",
                  fstr(demangle_(src ? src->oport_payload_type(e.p)
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
PipelineApi::topo_order_(const std::vector<StageSpec>& stages)
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
PipelineApi::to_flex_spec_(const Pipe& p) const
{
  FlexData spec = FlexData::make_object();
  auto o = spec.as_object();
  o.insert("id", fstr(p.id));
  FlexData stages = FlexData::make_array();
  auto sa = stages.as_array();
  // Emit in dependency order so the single-pass core loader can resolve every
  // iport (in-memory edits can leave p.stages non-topological -- see above).
  for (std::size_t si : topo_order_(p.stages)) {
    const StageSpec& st = p.stages[si];
    FlexData e = FlexData::make_object();
    auto eo = e.as_object();
    eo.insert("id", fstr(st.id));
    eo.insert("type", fstr(st.type));
    FlexData ip = FlexData::make_array();
    auto ia = ip.as_array();
    for (auto& [src, op] : st.iports) {
      FlexData pe = FlexData::make_object();
      auto peo = pe.as_object();
      peo.insert("src", fstr(src));
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
  // Round-trip auxiliary data objects (opaque to the pipeline core).
  if (p.aux.is_object() && !p.aux.as_object().empty()) {
    o.insert("aux", p.aux);
  }
  return spec;
}

void
PipelineApi::merge_aux_(Pipe& p, const FlexData& incoming)
{
  if (!incoming.is_object()) { return; }
  if (!p.aux.is_object()) { p.aux = FlexData::make_object(); }
  auto dst = p.aux.as_object();
  for (const auto& [k, v] : incoming.as_object()) {
    if (v.is_null()) { dst.erase(k); }
    else             { dst.insert_or_assign(k, v); }
  }
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
PipelineApi::h_stage_types_(const HttpRequest&)
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
    oo.insert("type", fstr(n));
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
      oo.insert("category", fstr("generic"));
      oo.insert("doc", fstr(""));
      oo.insert("display_name", fstr(""));
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
PipelineApi::h_list_pipelines_(const HttpRequest&)
{
  lock_guard<mutex> lk(_ctx.mu);
  reap_completed_();   // surface self-terminated pipelines as stopped
  FlexData arr = FlexData::make_array();
  auto a = arr.as_array();
  for (auto& up : _pipes) { a.push_back(pipe_summary_(*up)); }
  return HttpResponse::json(200, arr.to_json());
}

HttpResponse
PipelineApi::h_create_pipeline_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  string id = string(body->is_object()
                         ? body->as_object().contains("id")
                               ? body->as_object().at("id").as_string("")
                               : ""
                         : "");
  if (id.empty()) { return HttpResponse::error(400, "missing 'id'"); }

  lock_guard<mutex> lk(_ctx.mu);
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
  d.insert("id", fstr(p->id));
  d.insert("state", fstr(state_name_(p->state)));
  d.insert("storage_path", fstr(p->storage_path));
  d.insert("graph", graph_json_(*p));
  _pipes.push_back(std::move(p));
  return HttpResponse::json(201, detail.to_json());
}

HttpResponse
PipelineApi::h_rename_pipeline_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  lock_guard<mutex> lk(_ctx.mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (p->state != State::Stopped) {
    return HttpResponse::error(409, "stop the pipeline before renaming");
  }
  string to = trim(string(body->as_object().contains("to")
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
  d.insert("id", fstr(p->id));
  d.insert("state", fstr(state_name_(p->state)));
  d.insert("storage_path", fstr(p->storage_path));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
PipelineApi::h_load_pipeline_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
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
      _ctx.sctx ? _ctx.sctx->confine_path(path, /*for_write=*/false, &cerr)
            : std::filesystem::path(path);
  if (real.empty()) {
    return HttpResponse::error(
        400, "path '" + path + "' rejected: "
                 + (cerr.empty() ? "outside sandbox" : cerr));
  }

  lock_guard<mutex> lk(_ctx.mu);
  PipelineHandle h = _ctx.session->load_pipeline(real.string());
  if (!h.valid()) {
    return HttpResponse::error(400, "failed to load '" + path + "'");
  }
  Pipeline* pl = live_pipeline(h);
  if (!pl) {
    _ctx.session->unload_pipeline(h);
    return HttpResponse::error(500, "loaded pipeline has no graph");
  }
  string id = pl->id();
  if (find_(id)) {
    _ctx.session->unload_pipeline(h);
    return HttpResponse::error(
        409, "a pipeline named '" + id + "' is already open");
  }

  auto p = make_unique<Pipe>();
  p->id           = id;
  p->handle       = h;
  p->state        = State::Stopped;
  p->storage_path = path;
  // Recover auxiliary data objects (composer arrangement, etc.) that the
  // pipeline core dropped -- re-read from the file we just loaded.
  p->aux          = aux_from_file_(real);

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
  d.insert("id", fstr(p->id));
  d.insert("state", fstr(state_name_(p->state)));
  d.insert("storage_path", fstr(p->storage_path));
  d.insert("graph", graph_json_(*p));
  if (p->aux.is_object() && !p->aux.as_object().empty()) {
    d.insert("aux", p->aux);
  }
  _pipes.push_back(std::move(p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
PipelineApi::h_get_pipeline_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_ctx.mu);
  reap_completed_();   // surface self-terminated pipelines as stopped
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  FlexData detail = FlexData::make_object();
  auto d = detail.as_object();
  d.insert("id", fstr(p->id));
  d.insert("state", fstr(state_name_(p->state)));
  d.insert("storage_path", fstr(p->storage_path));
  d.insert("graph", graph_json_(*p));
  if (p->aux.is_object() && !p->aux.as_object().empty()) {
    d.insert("aux", p->aux);
  }
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
PipelineApi::h_save_pipeline_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body) { return HttpResponse::error(400, "invalid JSON body"); }
  lock_guard<mutex> lk(_ctx.mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }

  // A save may carry updated auxiliary data objects to persist alongside.
  if (body->is_object() && body->as_object().contains("aux")) {
    merge_aux_(*p, body->as_object().at("aux"));
  }

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
      _ctx.sctx ? _ctx.sctx->confine_path(path, /*for_write=*/true, &cerr)
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
  o.as_object().insert("storage_path", fstr(path));
  return HttpResponse::json(200, o.to_json());
}

// PUT /api/pipelines/:id/aux -- attach/update auxiliary data objects (e.g.
// the composer view arrangement). The request body's keys are merged into
// the pipeline's aux map (null value = erase). If the pipeline already has a
// storage path, the change is written through to that file so it persists;
// otherwise it stays in the session until the pipeline is next saved.
HttpResponse
PipelineApi::h_set_pipeline_aux_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  lock_guard<mutex> lk(_ctx.mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }

  // A body of {"aux": {...}} or a bare {...} object are both accepted.
  const FlexData& incoming =
      body->as_object().contains("aux") ? body->as_object().at("aux")
                                        : *body;
  merge_aux_(*p, incoming);

  bool persisted = false;
  if (!p->storage_path.empty()) {
    string cerr;
    std::filesystem::path real =
        _ctx.sctx ? _ctx.sctx->confine_path(p->storage_path, /*for_write=*/true,
                                    &cerr)
              : std::filesystem::path(p->storage_path);
    if (!real.empty()) {
      FlexData spec = to_flex_spec_(*p);
      ofstream f(real, ios::binary);
      if (f) { f << spec.to_json(/*pretty=*/true); persisted = bool(f); }
    }
  }

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("ok", FlexData::make_bool(true));
  oo.insert("persisted", FlexData::make_bool(persisted));
  oo.insert("storage_path", fstr(p->storage_path));
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
PipelineApi::h_unload_pipeline_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_ctx.mu);
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
PipelineApi::h_launch_pipeline_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_ctx.mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (!p->handle || !p->handle->valid()) {
    return HttpResponse::error(409, "pipeline is not materialized");
  }
  if (p->state != State::Stopped) {
    return HttpResponse::error(409, "pipeline is already running");
  }
  Status s = _ctx.session->launch_pipeline(*p->handle);
  if (s.code != 0) {
    return HttpResponse::error(500, "launch failed (status " +
                               to_string(s.code) + ")");
  }
  p->state = State::Running;
  return HttpResponse::json(200, pipe_summary_(*p).to_json());
}

HttpResponse
PipelineApi::h_pause_pipeline_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_ctx.mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (p->state != State::Running) {
    return HttpResponse::error(409, "pipeline is not running");
  }
  Status s = _ctx.session->pause_pipeline(*p->handle);
  if (s.code != 0) {
    return HttpResponse::error(500, "pause failed (status " +
                               to_string(s.code) + ")");
  }
  p->state = State::Paused;
  return HttpResponse::json(200, pipe_summary_(*p).to_json());
}

HttpResponse
PipelineApi::h_stop_pipeline_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_ctx.mu);
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (p->state == State::Stopped) {
    return HttpResponse::json(200, pipe_summary_(*p).to_json());
  }
  Status s = _ctx.session->stop_pipeline(*p->handle);
  if (s.code != 0) {
    return HttpResponse::error(500, "stop failed (status " +
                               to_string(s.code) + ")");
  }
  p->state = State::Stopped;
  return HttpResponse::json(200, pipe_summary_(*p).to_json());
}

HttpResponse
PipelineApi::h_buffer_status_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_ctx.mu);
  reap_completed_();   // surface self-terminated pipelines as stopped
  auto it = req.params.find("id");
  Pipe* p = (it != req.params.end()) ? find_(it->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }

  FlexData o = FlexData::make_object();
  auto oo = o.as_object();
  oo.insert("id", fstr(p->id));
  oo.insert("state", fstr(state_name_(p->state)));
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
        x.insert("from", fstr(e.from_id));
        x.insert("from_port", FlexData::make_uint(e.from_port));
        x.insert("to", fstr(e.to_id));
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
PipelineApi::h_insert_stage_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  auto bo = body->as_object();

  lock_guard<mutex> lk(_ctx.mu);
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
  if (Pipeline* pl = live_pipeline(*p->handle)) {
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
  d.insert("id", fstr(p->id));
  d.insert("state", fstr(state_name_(p->state)));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(201, detail.to_json());
}

HttpResponse
PipelineApi::h_remove_stage_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_ctx.mu);
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
  d.insert("id", fstr(p->id));
  d.insert("state", fstr(state_name_(p->state)));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
PipelineApi::h_rename_stage_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  lock_guard<mutex> lk(_ctx.mu);
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
  string to = trim(string(body->as_object().contains("to")
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
  d.insert("id", fstr(p->id));
  d.insert("state", fstr(state_name_(p->state)));
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
PipelineApi::h_duplicate_stage_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_ctx.mu);
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
  if (auto body = parse_json_body(req); body && body->is_object()
      && body->as_object().contains("to")) {
    new_id = trim(string(body->as_object().at("to").as_string("")));
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
  d.insert("id", fstr(p->id));
  d.insert("state", fstr(state_name_(p->state)));
  d.insert("stage", fstr(new_id));   // the freshly created copy's id
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(201, detail.to_json());
}

HttpResponse
PipelineApi::h_connect_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  auto bo = body->as_object();

  lock_guard<mutex> lk(_ctx.mu);
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
  d.insert("id", fstr(p->id));
  d.insert("state", fstr(state_name_(p->state)));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
PipelineApi::h_disconnect_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "invalid JSON body");
  }
  auto bo = body->as_object();

  lock_guard<mutex> lk(_ctx.mu);
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
  d.insert("id", fstr(p->id));
  d.insert("state", fstr(state_name_(p->state)));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

HttpResponse
PipelineApi::h_get_stage_config_(const HttpRequest& req)
{
  lock_guard<mutex> lk(_ctx.mu);
  auto pit = req.params.find("id");
  auto sit = req.params.find("sid");
  Pipe* p = (pit != req.params.end()) ? find_(pit->second) : nullptr;
  if (!p) { return HttpResponse::error(404, "no such pipeline"); }
  if (sit == req.params.end()) {
    return HttpResponse::error(400, "missing stage id");
  }
  const Stage* s = nullptr;
  if (p->handle && p->handle->valid()) {
    if (Pipeline* pl = live_pipeline(*p->handle)) {
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
  oo.insert("id", fstr(s->id()));
  oo.insert("type", fstr(s->type_name()));
  oo.insert("editable", FlexData::make_bool(p->state == State::Stopped));
  oo.insert("schema", s->config_schema());
  return HttpResponse::json(200, o.to_json());
}

HttpResponse
PipelineApi::h_set_stage_config_(const HttpRequest& req)
{
  auto body = parse_json_body(req);
  if (!body || !body->is_object()) {
    return HttpResponse::error(400, "config body must be a JSON object");
  }
  lock_guard<mutex> lk(_ctx.mu);
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
                        [&](const StageSpec& s) {
                          return s.id == sit->second;
                        });
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
  d.insert("id", fstr(p->id));
  d.insert("state", fstr(state_name_(p->state)));
  d.insert("graph", graph_json_(*p));
  return HttpResponse::json(200, detail.to_json());
}

std::vector<UiViewHostIntf::StageRef>
PipelineApi::ViewHost::find_stages(std::string_view type_name) const
{
  std::vector<StageRef> out;
  lock_guard<mutex> lk(_api._ctx.mu);
  for (const auto& up : _api._pipes) {
    const Pipe& p = *up;
    // Read the editable SPEC, not the live graph: a view's picker must
    // list stages in pipelines that are merely loaded, so it can follow
    // one that has not been launched yet.
    const bool live = p.state != State::Stopped && p.handle
                      && p.handle->valid();
    for (const auto& st : p.stages) {
      if (st.type != type_name) { continue; }
      StageRef r;
      r.pipeline = p.id;
      r.stage    = st.id;
      r.state    = state_name_(p.state);
      r.live     = live;
      r.config   = st.config;
      out.push_back(std::move(r));
    }
  }
  return out;
}

Stage*
PipelineApi::ViewHost::live_stage(std::string_view pipeline,
                                 std::string_view stage) const
{
  lock_guard<mutex> lk(_api._ctx.mu);
  Pipe* p = _api.find_(string(pipeline));
  if (p == nullptr || !p->handle || !p->handle->valid()) { return nullptr; }
  if (p->state == State::Stopped) { return nullptr; }
  return _api.live_stage_(*p, string(stage));
}

void
PipelineApi::for_each_live_locked(
    const std::function<void(const std::string&, const char*,
                             Pipeline&)>& fn) const
{
  for (const auto& up : _pipes) {
    const Pipe& p = *up;
    if (!p.handle || !p.handle->valid()) { continue; }
    Pipeline* pl = live_pipeline(*p.handle);
    if (pl == nullptr) { continue; }
    fn(p.id, state_name_(p.state), *pl);
  }
}

void
PipelineApi::register_routes(HttpServer& s)
{
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
  s.route("PUT", "/api/pipelines/:id/aux",
          [this](const HttpRequest& r) { return h_set_pipeline_aux_(r); });
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
}

}
