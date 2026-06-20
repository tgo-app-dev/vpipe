#include "pipeline/pipeline-spec.h"

#include "common/graph.h"
#include "common/vertex.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/pipeline.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage.h"

#include <cstdint>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

// Pull a string out of a FlexData object slot, falling back on a
// default. Returns empty string if the field is absent or of the
// wrong kind.
string
read_string_(const FlexData::ConstObjectView& obj,
             string_view                      key,
             string_view                      def = {})
{
  if (!obj.contains(key)) {
    return string(def);
  }
  return string(obj.at(key).as_string(def));
}

// Topological order over the pipeline's stages. The deserializer
// requires upstream stages to be declared before their consumers
// (so iports can resolve in a single pass), but Pipeline iteration
// is over an unordered_map -- order isn't stable. Kahn's algorithm
// produces a stable, dependency-respecting order.
//
// Input: pipeline. Output: vector of stage pointers in declaration
// order. Stages that are part of a cycle (none today, since the
// runtime rejects intra-domain cycles at launch, but we don't want
// to UB on a malformed pipeline) are appended at the end so we
// don't drop them silently.
vector<const Stage*>
topo_sort_stages_(const Pipeline& pl)
{
  // Collect all stages first, plus a fanout adjacency map and
  // an in-degree counter. We work with const Stage* throughout.
  vector<const Stage*> all;
  unordered_map<const Stage*, unsigned> indeg;
  for (auto it = pl.begin(); it != pl.end(); ++it) {
    auto* s = dynamic_cast<const Stage*>(*it);
    if (!s) {
      continue;
    }
    all.push_back(s);
    indeg[s] = 0;
  }
  // Compute in-degrees (count of iport edges pointing at *another
  // stage in this pipeline*).
  unordered_set<const Stage*> in_pl(all.begin(), all.end());
  for (const Stage* s : all) {
    for (const InEdge& e : s->iport_edges()) {
      auto* up = dynamic_cast<const Stage*>(e.v);
      if (up && in_pl.count(up)) {
        ++indeg[s];
      }
    }
  }
  // Sort initial-zero-indegree stages by id so the output is
  // stable across runs. Same trick at every queue extraction.
  auto cmp_id = [](const Stage* a, const Stage* b) {
    return a->id() > b->id();  // priority_queue picks largest;
                               // invert so smallest id comes out first.
  };
  priority_queue<const Stage*, vector<const Stage*>,
                 decltype(cmp_id)> q(cmp_id);
  for (const Stage* s : all) {
    if (indeg[s] == 0) {
      q.push(s);
    }
  }
  vector<const Stage*> out;
  out.reserve(all.size());
  while (!q.empty()) {
    const Stage* s = q.top();
    q.pop();
    out.push_back(s);
    // For each downstream consumer of any of s's oports, decrement
    // its in-degree; push when it hits zero.
    for (unsigned p = 0; p < s->num_oports(); ++p) {
      for (const OutEdge& e : s->oport_edges(p)) {
        auto* dn = dynamic_cast<const Stage*>(e.v);
        if (!dn || !in_pl.count(dn)) {
          continue;
        }
        auto it = indeg.find(dn);
        if (it == indeg.end()) {
          continue;
        }
        if (it->second > 0) {
          --it->second;
          if (it->second == 0) {
            q.push(dn);
          }
        }
      }
    }
  }
  // Append anything still left (only happens if there's a cycle).
  if (out.size() != all.size()) {
    unordered_set<const Stage*> seen(out.begin(), out.end());
    for (const Stage* s : all) {
      if (!seen.count(s)) {
        out.push_back(s);
      }
    }
  }
  return out;
}

}

FlexData
pipeline_to_spec(const Pipeline& pl)
{
  FlexData out = FlexData::make_object();
  out.as_object().insert_or_assign(
      "id", FlexData::make_string(pl.id()));

  FlexData stages = FlexData::make_array();
  for (const Stage* s : topo_sort_stages_(pl)) {
    FlexData stg = FlexData::make_object();
    stg.as_object().insert_or_assign(
        "id",   FlexData::make_string(s->id()));
    stg.as_object().insert_or_assign(
        "type", FlexData::make_string(s->type_name()));

    FlexData iports = FlexData::make_array();
    for (const InEdge& e : s->iport_edges()) {
      FlexData entry = FlexData::make_object();
      entry.as_object().insert_or_assign(
          "src",
          FlexData::make_string(e.v ? e.v->id() : string()));
      entry.as_object().insert_or_assign(
          "oport",
          FlexData::make_int(static_cast<int64_t>(e.p)));
      iports.as_array().push_back(std::move(entry));
    }
    stg.as_object().insert_or_assign("iports", std::move(iports));

    // Deep-copy the stage's config tree (FlexData copy-ctor clones).
    stg.as_object().insert_or_assign(
        "config", FlexData(s->config()));
    stages.as_array().push_back(std::move(stg));
  }
  out.as_object().insert_or_assign("stages", std::move(stages));

  FlexData subs = FlexData::make_array();
  for (const auto& kv : pl.graphs()) {
    const Pipeline* child =
        dynamic_cast<const Pipeline*>(kv.second.get());
    if (!child) {
      continue;
    }
    subs.as_array().push_back(pipeline_to_spec(*child));
  }
  out.as_object().insert_or_assign("subpipelines", std::move(subs));

  return out;
}

unique_ptr<Pipeline>
pipeline_from_spec(const FlexData&           spec,
                   const SessionContextIntf* session)
{
  auto warn = [&](const VpipeFormat& f) {
    if (session) {
      session->warn(f);
    }
  };

  if (!spec.is_object()) {
    warn(fmt("pipeline_from_spec: top-level spec must be an object"));
    return nullptr;
  }
  auto root = spec.as_object();
  string id = read_string_(root, "id");
  if (id.empty()) {
    warn(fmt("pipeline_from_spec: missing or empty 'id'"));
    return nullptr;
  }

  auto pl = make_unique<Pipeline>(std::move(id), session);

  // ---- Stages, in declared order. -----------------------------
  unordered_map<string, Stage*> by_id;
  if (root.contains("stages")) {
    FlexData stages_val = root.at("stages");
    if (!stages_val.is_array()) {
      warn(fmt("pipeline_from_spec('{}'): 'stages' must be an array",
               pl->id()));
      return nullptr;
    }
    auto stages = stages_val.as_array();
    for (size_t i = 0, n = stages.size(); i < n; ++i) {
      FlexData s_val = stages.at(i);
      if (!s_val.is_object()) {
        warn(fmt(
            "pipeline_from_spec('{}'): stages[{}] must be an object",
            pl->id(), i));
        return nullptr;
      }
      auto s = s_val.as_object();

      string sid   = read_string_(s, "id");
      string stype = read_string_(s, "type");
      if (sid.empty()) {
        warn(fmt(
            "pipeline_from_spec('{}'): stages[{}] missing or empty "
            "'id'", pl->id(), i));
        return nullptr;
      }
      if (stype.empty()) {
        warn(fmt(
            "pipeline_from_spec('{}'): stage '{}' missing or empty "
            "'type'", pl->id(), sid));
        return nullptr;
      }
      if (by_id.count(sid)) {
        warn(fmt(
            "pipeline_from_spec('{}'): duplicate stage id '{}'",
            pl->id(), sid));
        return nullptr;
      }

      // Resolve iports against already-inserted upstream stages.
      vector<InEdge> iports;
      if (s.contains("iports")) {
        FlexData ip_val = s.at("iports");
        if (!ip_val.is_array()) {
          warn(fmt(
              "pipeline_from_spec('{}'): stage '{}' iports must be "
              "an array", pl->id(), sid));
          return nullptr;
        }
        auto ip = ip_val.as_array();
        iports.reserve(ip.size());
        for (size_t j = 0, m = ip.size(); j < m; ++j) {
          FlexData e_val = ip.at(j);
          if (!e_val.is_object()) {
            warn(fmt(
                "pipeline_from_spec('{}'): stage '{}' iports[{}] "
                "must be an object", pl->id(), sid, j));
            return nullptr;
          }
          auto e = e_val.as_object();
          string src = read_string_(e, "src");
          if (src.empty()) {
            warn(fmt(
                "pipeline_from_spec('{}'): stage '{}' iports[{}] "
                "missing 'src'", pl->id(), sid, j));
            return nullptr;
          }
          auto srcit = by_id.find(src);
          if (srcit == by_id.end()) {
            warn(fmt(
                "pipeline_from_spec('{}'): stage '{}' iports[{}] "
                "references unknown / forward-declared stage '{}'",
                pl->id(), sid, j, src));
            return nullptr;
          }
          int64_t oport = 0;
          if (e.contains("oport")) {
            oport = e.at("oport").as_int(0);
          }
          if (oport < 0) {
            warn(fmt(
                "pipeline_from_spec('{}'): stage '{}' iports[{}] "
                "oport must be >= 0 (got {})",
                pl->id(), sid, j, oport));
            return nullptr;
          }
          Stage* upstream = srcit->second;
          if (static_cast<unsigned>(oport) >=
              upstream->num_oports()) {
            warn(fmt(
                "pipeline_from_spec('{}'): stage '{}' iports[{}] "
                "oport {} out of range (upstream '{}' has {} "
                "oports)",
                pl->id(), sid, j, oport, src,
                upstream->num_oports()));
            return nullptr;
          }
          iports.push_back(InEdge{upstream,
                                  static_cast<unsigned>(oport)});
        }
      }

      FlexData cfg = FlexData::make_object();
      if (s.contains("config")) {
        cfg = s.at("config");
      }

      StagePtr stage = StageRegistry::get().create(
          stype, session, sid, std::move(iports), std::move(cfg));
      if (!stage) {
        // StageRegistry::create already logged its own warn via
        // session->warn; just bail here.
        return nullptr;
      }
      Stage* raw = pl->insert_stage(std::move(stage));
      if (!raw) {
        warn(fmt(
            "pipeline_from_spec('{}'): pipeline insert_stage('{}') "
            "failed", pl->id(), sid));
        return nullptr;
      }
      by_id.emplace(sid, raw);
    }
  }

  // ---- Sub-pipelines. -----------------------------------------
  if (root.contains("subpipelines")) {
    FlexData sub_val = root.at("subpipelines");
    if (!sub_val.is_array()) {
      warn(fmt(
          "pipeline_from_spec('{}'): 'subpipelines' must be an "
          "array", pl->id()));
      return nullptr;
    }
    auto subs = sub_val.as_array();
    for (size_t i = 0, n = subs.size(); i < n; ++i) {
      FlexData child_spec = subs.at(i);
      auto child = pipeline_from_spec(child_spec, session);
      if (!child) {
        // Inner failure already warned.
        return nullptr;
      }
      pl->insert_graph(std::move(child));
    }
  }

  return pl;
}

}
