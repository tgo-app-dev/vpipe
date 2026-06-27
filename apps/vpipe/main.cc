// vpipe -- command-line entrance to libvpipe.
//
// Launch one or more pipelines, concurrently, from the shell. A launch is
// either a full spec (--launch) or a single ad-hoc stage wrapped in a
// one-stage pipeline (--launch-stage). Per-launch config overrides are
// supplied with --stage-cfg and apply to the launch that immediately
// precedes them.
//
// Usage:
//   vpipe [--config CFG] LAUNCH [--stage-cfg OVERRIDE]... [LAUNCH ...]
//
//   LAUNCH is one of:
//     --launch <spec>              spec is a path to a pipeline JSON/binary
//                                  file, or an inline JSON object string.
//     --launch-stage <stage-type>  build a temporary one-stage pipeline of
//                                  the named registered stage type.
//
//   --stage-cfg <override>  applies to the most recent LAUNCH:
//       * after --launch:        STAGE::KEY=VALUE  (STAGE is a stage id in
//                                the spec)
//       * after --launch-stage:  KEY=VALUE
//     May be repeated. VALUE is parsed as JSON when it can be (5 -> int,
//     5.0 -> real, true -> bool, [..]/{..} -> array/object); otherwise it
//     is taken literally as a string. Quote to force a string: KEY='"5"'.
//
//   --config CFG   session config forwarded to SessionManager: inline JSON
//                  ({...}) or a path to a JSON/binary config file.
//   --help, -h     print this help.
//
// Examples:
//   vpipe --launch pipeline.json
//   vpipe --launch '{"id":"p","stages":[{"id":"t","type":"chrono",
//         "config":{"count":3}}]}'
//   vpipe --launch pipeline.json --stage-cfg chat::hf_dir=/models/qwen
//   vpipe --launch-stage chrono --stage-cfg frequency_hz=2 --stage-cfg count=5
//   vpipe --launch-stage chrono --stage-cfg count=3 \
//         --launch-stage onvif-discovery     # two pipelines, concurrent
//
// The executable dynamically links libvpipe; pipeline output and
// diagnostics flow through the session's default stdout delegates.

#include "common/flex-data.h"
#include "vpipe/pipeline-handle.h"
#include "vpipe/session-intf.h"
#include "vpipe/session-manager.h"
#include "vpipe/status.h"

#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace vpipe;

namespace {

// Set from the signal handler; the wait loop polls it to stop cleanly.
std::atomic<bool> g_interrupted{false};

void
on_signal(int)
{
  g_interrupted.store(true);
}

const char* const kUsage =
  "vpipe -- command-line entrance to libvpipe.\n"
  "\n"
  "Usage:\n"
  "  vpipe [--config CFG] LAUNCH [--stage-cfg OVERRIDE]... [LAUNCH ...]\n"
  "\n"
  "Launches (repeat to run pipelines concurrently):\n"
  "  --launch <spec>             pipeline JSON/binary file path, or an\n"
  "                              inline JSON object string.\n"
  "  --launch-stage <type>       a temporary one-stage pipeline of the\n"
  "                              named registered stage type.\n"
  "\n"
  "Overrides (apply to the preceding launch):\n"
  "  --stage-cfg STAGE::KEY=VAL  after --launch: set stage STAGE's config\n"
  "                              key KEY.\n"
  "  --stage-cfg KEY=VAL         after --launch-stage: set the stage's\n"
  "                              config key KEY.\n"
  "  VALUE is parsed as JSON when possible (5->int, 5.0->real, true->bool,\n"
  "  [..]/{..}->array/object), else taken as a string. Quote for a string:\n"
  "  KEY='\"5\"'.\n"
  "\n"
  "Other:\n"
  "  --config CFG                session config (inline JSON or file path).\n"
  "  --help, -h                  print this help.\n";

int
arg_err(const char* msg)
{
  std::fprintf(stderr, "vpipe: %s\nTry 'vpipe --help'.\n", msg);
  return 2;
}

// True iff the first non-space character is '{' or '[' (an inline JSON
// document rather than a filesystem path).
bool
looks_like_json(std::string_view s)
{
  for (char c : s) {
    if (std::isspace(static_cast<unsigned char>(c))) { continue; }
    return c == '{' || c == '[';
  }
  return false;
}

bool
read_file(const std::string& path, std::string& out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { return false; }
  std::ostringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

// Split "KEY=VALUE" at the first '='. lhs must be non-empty.
bool
split_kv(const std::string& s, std::string& lhs, std::string& val)
{
  const auto eq = s.find('=');
  if (eq == std::string::npos) { return false; }
  lhs = s.substr(0, eq);
  val = s.substr(eq + 1);
  return !lhs.empty();
}

// Parse an override VALUE: JSON when it parses (so numbers/bools/arrays/
// objects keep their type), otherwise a literal string.
FlexData
parse_value(const std::string& v)
{
  try {
    return FlexData::from_json(v);
  } catch (const std::exception&) {
    return FlexData::make_string(v);
  }
}

// Set spec.stages[<id == stage_id>].config[key] = value. Returns false if
// the spec has no such stage. FlexData views hand back copies, so we edit
// copies and write them back (set / insert_or_assign).
bool
apply_override(FlexData& spec, const std::string& stage_id,
               std::string_view key, FlexData value)
{
  if (!spec.is_object()) { return false; }
  auto so = spec.as_object();
  auto sit = so.find("stages");
  if (sit == so.end()) { return false; }
  FlexData stages = (*sit).second;          // deep copy of the stages array
  if (!stages.is_array()) { return false; }
  auto arr = stages.as_array();
  const std::size_t n = arr.size();
  int found = -1;
  for (std::size_t i = 0; i < n; ++i) {
    FlexData st = arr.at(i);
    if (!st.is_object()) { continue; }
    auto sto = st.as_object();
    auto idit = sto.find("id");
    if (idit != sto.end() && (*idit).second.as_string() == stage_id) {
      found = static_cast<int>(i);
      break;
    }
  }
  if (found < 0) { return false; }
  FlexData st = arr.at(static_cast<std::size_t>(found));
  FlexData cfg;
  {
    auto sto = st.as_object();
    auto cit = sto.find("config");
    if (cit != sto.end() && (*cit).second.is_object()) {
      cfg = (*cit).second;
    } else {
      cfg = FlexData::make_object();
    }
  }
  cfg.as_object().insert_or_assign(key, std::move(value));
  st.as_object().insert_or_assign("config", std::move(cfg));
  arr.set(static_cast<std::size_t>(found), std::move(st));
  so.insert_or_assign("stages", std::move(stages));
  return true;
}

struct Launch {
  enum Kind { Full, Single };
  Kind                     kind;
  std::string              source;   // spec (Full) or stage type (Single)
  std::vector<std::string> cfgs;     // raw --stage-cfg tokens, in order
};

// Build a single-stage temporary pipeline. cfgs are KEY=VALUE (no '::').
std::optional<PipelineHandle>
build_single(SessionIntf* s, const Launch& L, int idx)
{
  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    for (const auto& ov : L.cfgs) {
      std::string key, val;
      if (!split_kv(ov, key, val)) {
        std::fprintf(stderr,
                     "vpipe: --stage-cfg '%s': expected KEY=VALUE\n",
                     ov.c_str());
        return std::nullopt;
      }
      o.insert_or_assign(key, parse_value(val));
    }
  }
  const std::string pid = "cli-" + L.source + "-" + std::to_string(idx);
  PipelineHandle pl = s->create_pipeline(pid);
  if (!pl) {
    std::fprintf(stderr, "vpipe: failed to create pipeline for stage '%s'\n",
                 L.source.c_str());
    return std::nullopt;
  }
  StageHandle st = pl.insert_stage(L.source, L.source, {}, cfg.to_json());
  if (!st) {
    std::fprintf(stderr,
                 "vpipe: failed to create stage '%s' (unknown type or bad "
                 "config)\n",
                 L.source.c_str());
    s->unload_pipeline(pl);
    return std::nullopt;
  }
  return pl;
}

// Build a full pipeline from a file path or inline JSON. With no overrides
// the source is handed straight to load_pipeline, which accepts a path or an
// inline JSON spec; with overrides it is parsed, edited, and handed back as
// an inline JSON spec.
std::optional<PipelineHandle>
build_full(SessionIntf* s, const Launch& L)
{
  // No edits: load_pipeline takes a path or an inline JSON spec directly.
  if (L.cfgs.empty()) {
    PipelineHandle h = s->load_pipeline(L.source);
    if (!h) { return std::nullopt; }   // loader already logged the cause
    return h;
  }

  // Overrides: parse into a FlexData spec we can edit.
  FlexData spec;
  try {
    if (looks_like_json(L.source)) {
      spec = FlexData::from_json(L.source);
    } else {
      std::string contents;
      if (!read_file(L.source, contents)) {
        std::fprintf(stderr, "vpipe: --launch '%s': cannot read file\n",
                     L.source.c_str());
        return std::nullopt;
      }
      spec = looks_like_json(contents) ? FlexData::from_json(contents)
                                       : FlexData::from_binary(contents);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vpipe: --launch: parse failed: %s\n", e.what());
    return std::nullopt;
  }

  for (const auto& ov : L.cfgs) {
    std::string lhs, val;
    if (!split_kv(ov, lhs, val)) {
      std::fprintf(stderr, "vpipe: --stage-cfg '%s': expected STAGE::KEY="
                   "VALUE\n", ov.c_str());
      return std::nullopt;
    }
    const auto pos = lhs.find("::");
    if (pos == std::string::npos) {
      std::fprintf(stderr,
                   "vpipe: --stage-cfg '%s': full-pipeline override needs "
                   "STAGE::KEY=VALUE\n",
                   ov.c_str());
      return std::nullopt;
    }
    const std::string stage = lhs.substr(0, pos);
    const std::string key = lhs.substr(pos + 2);
    if (!apply_override(spec, stage, key, parse_value(val))) {
      std::fprintf(stderr,
                   "vpipe: --stage-cfg '%s': no stage with id '%s' in spec\n",
                   ov.c_str(), stage.c_str());
      return std::nullopt;
    }
  }

  // Hand the edited spec back to the loader as an inline JSON document.
  PipelineHandle h = s->load_pipeline(spec.to_json());
  if (!h) { return std::nullopt; }   // loader already logged the cause
  return h;
}

int
run(int argc, char** argv)
{
  std::string           config;
  std::vector<Launch>   launches;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::fputs(kUsage, stdout);
      return 0;
    } else if (a == "--config") {
      if (++i >= argc) { return arg_err("--config needs a value"); }
      config = argv[i];
    } else if (a == "--launch") {
      if (++i >= argc) { return arg_err("--launch needs a spec"); }
      launches.push_back({Launch::Full, argv[i], {}});
    } else if (a == "--launch-stage") {
      if (++i >= argc) { return arg_err("--launch-stage needs a stage type"); }
      launches.push_back({Launch::Single, argv[i], {}});
    } else if (a == "--stage-cfg") {
      if (++i >= argc) { return arg_err("--stage-cfg needs KEY=VALUE"); }
      if (launches.empty()) {
        return arg_err("--stage-cfg must follow --launch/--launch-stage");
      }
      launches.back().cfgs.push_back(argv[i]);
    } else {
      std::fprintf(stderr, "vpipe: unknown argument '%s'\n", a.c_str());
      std::fprintf(stderr, "Try 'vpipe --help'.\n");
      return 2;
    }
  }

  if (launches.empty()) {
    std::fputs(kUsage, stderr);
    return 2;
  }

  SessionManager& mgr = SessionManager::get();
  const SessionIntf* csess = mgr.create_session(config);
  if (csess == nullptr) {
    std::fprintf(stderr, "vpipe: failed to create session (bad --config?)\n");
    return 1;
  }
  SessionIntf* s = const_cast<SessionIntf*>(csess);

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  std::vector<PipelineHandle> handles;
  int built = 0;
  int failed = 0;
  for (int idx = 0; idx < static_cast<int>(launches.size()); ++idx) {
    const Launch& L = launches[static_cast<std::size_t>(idx)];
    std::optional<PipelineHandle> h =
        (L.kind == Launch::Full) ? build_full(s, L)
                                 : build_single(s, L, idx);
    if (!h) {
      ++failed;
      continue;
    }
    const Status st = s->launch_pipeline(*h);
    if (st.code != 0) {
      std::fprintf(stderr, "vpipe: launch failed for '%s' (status %u)\n",
                   L.source.c_str(), st.code);
      s->unload_pipeline(*h);
      ++failed;
      continue;
    }
    std::fprintf(stderr, "vpipe: launched %s '%s'\n",
                 L.kind == Launch::Full ? "pipeline" : "stage",
                 L.source.c_str());
    handles.push_back(*h);
    ++built;
  }

  if (built == 0) {
    std::fprintf(stderr, "vpipe: no pipelines launched\n");
    mgr.destroy_session(s);
    return 1;
  }

  // Wait for every launched pipeline to drain, polling so a SIGINT/SIGTERM
  // can preempt long-running (or endless) pipelines with a clean stop.
  for (;;) {
    const Status w = s->wait_pipelines(250);
    if (w.code == 0) { break; }            // all reached idle
    if (g_interrupted.load()) {
      std::fprintf(stderr, "\nvpipe: interrupted -- stopping pipelines...\n");
      for (auto& h : handles) { s->stop_pipeline(h); }
      break;
    }
    // Otherwise Status{4} (timeout): keep waiting.
  }

  mgr.destroy_session(s);
  return failed > 0 ? 1 : 0;
}

}  // namespace

int
main(int argc, char** argv)
{
  return run(argc, argv);
}
