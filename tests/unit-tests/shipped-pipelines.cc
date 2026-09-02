// THE SHIPPED EXAMPLE PIPELINES ARE STRUCTURALLY VALID.
//
// docs/pipelines/*.vpipeline are documentation: the docs link to them,
// users download them and run them verbatim, and NOTHING in the build
// reads them. So a stage type that was renamed, a config key that was
// spelled wrong, an iport index past the end of a stage's port list --
// none of it surfaces until someone launches the file, by which point
// the failure is a stranger's, not ours.
//
// model-catalog.cc already walks the `prepare-` files for one SEMANTIC
// property (a quantize stage names a model an earlier stage registers).
// This is the structural half, over EVERY shipped pipeline:
//
//   * it parses as JSON and has a `stages` array;
//   * every `type` is a registered stage type;
//   * every config key is declared in that type's spec, and its JSON
//     value matches the declared ConfigType;
//   * every required config key is present;
//   * every non-empty `iports[].src` names a stage in the same file,
//     and its `oport` is within that stage's declared oports;
//   * a stage is not wired more iports than its spec declares.
//
// The checks are deliberately the ones a reader cannot do by eye and a
// launch would catch too late. They do NOT run a graph: no model, no
// GPU, no network -- which is what lets this stay in the default suite.
//
// TWO THINGS IT CANNOT SEE, on purpose. A stage whose spec has no
// `attrs` table declares no keys, so its config is unconstrained and is
// skipped rather than reported (`Any`-typed keys are likewise accepted
// whatever their value). And an EMPTY `src` is a legitimately
// unconnected optional iport -- the shipped generative graphs are full
// of them, because port POSITION is the contract and a hole has to be
// spelled out rather than omitted.

#include "minitest.h"

#include "common/flex-data.h"
#include "pipeline/stage-config.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage-spec.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

// True iff `v` could have come from a JSON literal of type `t`.
//
// Deliberately PERMISSIVE where JSON itself is: an integer literal is a
// legal `Real` (nothing writes `24.0` for an fps by hand), and an
// unsigned key takes a non-negative int, since a JSON parser has no
// reason to hand back uint for `0`. Text is String's multi-line
// spelling and takes a string. `Any` accepts everything, which is what
// `references` needs -- one path or an array of them.
bool
value_fits(const FlexData& v, ConfigType t)
{
  switch (t) {
    case ConfigType::Any:    return true;
    case ConfigType::Bool:   return v.is_bool();
    case ConfigType::Int:    return v.is_int() || v.is_uint();
    case ConfigType::Uint:   return v.is_uint() || (v.is_int() &&
                                                    v.as_int(-1) >= 0);
    case ConfigType::Real:   return v.is_real() || v.is_int() || v.is_uint();
    case ConfigType::String:
    case ConfigType::Text:   return v.is_string();
    case ConfigType::Array:  return v.is_array();
    case ConfigType::Object: return v.is_object();
  }
  return false;
}

const ConfigKey*
find_key(const StageSpec& sp, string_view k)
{
  for (const ConfigKey& ck : sp.attrs) {
    if (ck.key == k) { return &ck; }
  }
  return nullptr;
}

// One stage as the checks below need it. `spec` is null for a type the
// registry does not know -- reported once, at the type check, and then
// skipped rather than cascading into a config complaint per key.
struct StageRec {
  string           id;
  string           type;
  const StageSpec* spec = nullptr;
  size_t           n_iports = 0;
};

string
read_file(const filesystem::path& p)
{
  ifstream in(p);
  string   txt, line;
  while (getline(in, line)) { txt += line + "\n"; }
  return txt;
}

}   // namespace

TEST(shipped_pipelines, every_example_is_structurally_valid)
{
#ifndef VPIPE_DOCS_PIPELINES_DIR
  printf("[pipelines] no docs dir defined; skipped\n");
#else
  namespace fs = filesystem;
  const fs::path dir(VPIPE_DOCS_PIPELINES_DIR);
  if (!fs::is_directory(dir)) {
    printf("[pipelines] %s is not a directory; skipped\n",
           dir.string().c_str());
    return;
  }
  const StageRegistry& reg = StageRegistry::get();

  // Sorted, so a failure names the same file on every machine.
  vector<fs::path> files;
  for (const fs::directory_entry& de : fs::directory_iterator(dir)) {
    if (de.path().extension() == ".vpipeline") { files.push_back(de.path()); }
  }
  sort(files.begin(), files.end());
  // An empty sweep is a pass that proves nothing -- the docs dir moving
  // would read as green. Fail instead.
  EXPECT_TRUE(!files.empty());

  int n_stages = 0, n_keys = 0, n_edges = 0;

  for (const fs::path& f : files) {
    const string fn = f.filename().string();
    // The whole document, bound to a local: as_object() is a VIEW into
    // it, so a temporary here dangles.
    FlexData doc = FlexData::from_json(read_file(f));
    const bool parsed = doc.is_object();
    if (!parsed) { printf("[pipelines] %s: not a JSON object\n", fn.c_str()); }
    EXPECT_TRUE(parsed);
    if (!parsed) { continue; }

    auto root = doc.as_object();
    const bool has_stages = root.contains("stages");
    if (!has_stages) { printf("[pipelines] %s: no `stages`\n", fn.c_str()); }
    EXPECT_TRUE(has_stages);
    if (!has_stages) { continue; }

    FlexData stages = root.at("stages");
    if (!stages.is_array()) {
      printf("[pipelines] %s: `stages` is not an array\n", fn.c_str());
      EXPECT_TRUE(false);
      continue;
    }
    auto stage_arr = stages.as_array();

    // ---- pass 1: identity, type, config -----------------------------
    //
    // Two passes because an edge names a stage that may be declared
    // LATER in the file, so the id -> record map has to be complete
    // before any src is resolved.
    vector<StageRec> recs;
    for (const FlexData& st : stage_arr) {
      if (!st.is_object()) {
        printf("[pipelines] %s: a stage is not an object\n", fn.c_str());
        EXPECT_TRUE(false);
        continue;
      }
      auto     o = st.as_object();
      StageRec r;
      r.id   = o.contains("id")   ? string(o.at("id").as_string(""))   : "";
      r.type = o.contains("type") ? string(o.at("type").as_string("")) : "";

      const bool named = !r.id.empty() && !r.type.empty();
      if (!named) {
        printf("[pipelines] %s: a stage is missing `id` or `type`\n",
               fn.c_str());
      }
      EXPECT_TRUE(named);

      const bool known = reg.find_id(r.type) != StageTypeId::unknown;
      if (!known) {
        printf("[pipelines] %s: stage '%s' has unregistered type '%s'\n",
               fn.c_str(), r.id.c_str(), r.type.c_str());
      }
      EXPECT_TRUE(known);
      r.spec = known ? reg.spec(r.type) : nullptr;

      if (o.contains("iports")) {
        FlexData ip = o.at("iports");
        if (ip.is_array()) { r.n_iports = ip.as_array().size(); }
      }

      // A spec with no attrs table declares no schema; its config is
      // unconstrained, so there is nothing to check against.
      if (r.spec != nullptr && !r.spec->attrs.empty() && o.contains("config")) {
        FlexData cfg = o.at("config");
        if (cfg.is_object()) {
          auto c = cfg.as_object();
          for (const auto& [k, v] : c) {
            const ConfigKey* ck = find_key(*r.spec, k);
            if (ck == nullptr) {
              printf("[pipelines] %s: stage '%s' (%s) sets unknown key "
                     "'%s'\n", fn.c_str(), r.id.c_str(), r.type.c_str(),
                     string(k).c_str());
              EXPECT_TRUE(false);
              continue;
            }
            const bool fits = value_fits(v, ck->type);
            if (!fits) {
              printf("[pipelines] %s: stage '%s' (%s) key '%s' wants %s\n",
                     fn.c_str(), r.id.c_str(), r.type.c_str(),
                     string(k).c_str(),
                     string(config_type_name(ck->type)).c_str());
            }
            EXPECT_TRUE(fits);
            ++n_keys;
          }
          // Required keys. A stage whose value arrives on an iport at
          // run time still has to spell the key here, so this is a
          // real property of the FILE and not of the run.
          for (const ConfigKey& rk : r.spec->attrs) {
            if (!rk.required) { continue; }
            const bool present = c.contains(rk.key);
            if (!present) {
              printf("[pipelines] %s: stage '%s' (%s) omits required key "
                     "'%s'\n", fn.c_str(), r.id.c_str(), r.type.c_str(),
                     string(rk.key).c_str());
            }
            EXPECT_TRUE(present);
          }
        }
      }

      // Wiring more iports than the spec declares is the silent one:
      // the extra edge is simply never read, so a graph built against
      // an older port layout looks connected and behaves as if the
      // stage were not wired at all.
      if (r.spec != nullptr && r.n_iports > r.spec->iports.size()) {
        printf("[pipelines] %s: stage '%s' (%s) wires %zu iports, spec "
               "declares %zu\n", fn.c_str(), r.id.c_str(), r.type.c_str(),
               r.n_iports, r.spec->iports.size());
        EXPECT_TRUE(false);
      }
      recs.push_back(std::move(r));
      ++n_stages;
    }

    // ---- pass 2: the edges ------------------------------------------
    size_t si = 0;
    for (const FlexData& st : stage_arr) {
      if (!st.is_object()) { continue; }
      auto o = st.as_object();
      if (si >= recs.size()) { break; }
      const StageRec& me = recs[si++];
      if (!o.contains("iports")) { continue; }
      FlexData ip = o.at("iports");
      if (!ip.is_array()) { continue; }
      auto   edges = ip.as_array();
      size_t port  = 0;
      for (const FlexData& e : edges) {
        const size_t p = port++;
        if (!e.is_object()) { continue; }
        auto         eo = e.as_object();
        const string src =
            eo.contains("src") ? string(eo.at("src").as_string("")) : "";
        // An empty src is an unconnected optional port, spelled out to
        // keep the ports after it at their declared positions.
        if (src.empty()) { continue; }

        const StageRec* from = nullptr;
        for (const StageRec& r : recs) {
          if (r.id == src) { from = &r; break; }
        }
        if (from == nullptr) {
          printf("[pipelines] %s: stage '%s' iport %zu names '%s', which is "
                 "not a stage in this file\n", fn.c_str(), me.id.c_str(), p,
                 src.c_str());
          EXPECT_TRUE(false);
          continue;
        }
        const long long op =
            eo.contains("oport") ? eo.at("oport").as_int(0) : 0;
        if (from->spec != nullptr && !from->spec->oports.empty()) {
          const bool in_range =
              op >= 0 && (size_t)op < from->spec->oports.size();
          if (!in_range) {
            printf("[pipelines] %s: stage '%s' iport %zu reads '%s' oport "
                   "%lld, which declares %zu\n", fn.c_str(), me.id.c_str(),
                   p, src.c_str(), op, from->spec->oports.size());
          }
          EXPECT_TRUE(in_range);
        }
        ++n_edges;
      }
    }
  }

  printf("[pipelines] %zu shipped pipelines: %d stages, %d config keys, "
         "%d edges checked\n", files.size(), n_stages, n_keys, n_edges);
#endif
}
