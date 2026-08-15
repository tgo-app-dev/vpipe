// vpipe -- command-line entrance to libvpipe.
//
// Launch one or more pipelines, concurrently, from the shell. A launch is
// either a full spec (--launch) or a single ad-hoc stage wrapped in a
// one-stage pipeline (--launch-stage). Per-launch config overrides are
// supplied with --stage-cfg and apply to the launch that immediately
// precedes them.
//
// Usage:
//   vpipe [--config CFG] [--memory-cap-mb N] LAUNCH ...
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
#include "common/session.h"
#include "plugin/plugin-manager.h"
#include "stages/model-catalog.h"
#include "ui/ui-view-registry.h"
#include "common/stdio-ui-delegate.h"
#include "vpipe/pipeline-handle.h"
#include "vpipe/vpipe.h"
#include "vpipe/session-intf.h"
#include "vpipe/session-manager.h"
#include "vpipe/status.h"

#if defined(VPIPE_BUILD_APPLE_SILICON)
#include "apple-silicon/metal-compute/metal-compute.h"
#include <sys/sysctl.h>
#endif

#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace vpipe;

namespace {

// Set from the signal handler; the wait loop polls it to stop cleanly.
// SIGTERM (and SIGINT when no stdio delegate is reachable) means "stop
// now"; SIGINT otherwise goes through the delegate's two-stroke policy,
// where the first stroke interrupts stage work and only the second
// quits. See StdioUiDelegate::note_sigint / poll_sigint.
std::atomic<bool> g_interrupted{false};

// Published before the signal handler is installed and cleared only
// after the wait loop is done, so the handler never touches a dead
// delegate.
std::atomic<StdioUiDelegate*> g_stdio_ui{nullptr};

void
on_signal(int sig)
{
  if (sig == SIGINT) {
    StdioUiDelegate* ui = g_stdio_ui.load(std::memory_order_acquire);
    if (ui != nullptr) {
      // Async-signal-safe: bumps an atomic, nothing else. The policy
      // runs later, from the wait loop's poll_sigint().
      ui->note_sigint();
      return;
    }
  }
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
  "  --memory-cap-mb N           ceiling on actively-resident model weights\n"
  "                              + KV. Over it the least-recently-used\n"
  "                              weights are PARKED (handed to the kernel as\n"
  "                              purgeable, reclaimed only under real memory\n"
  "                              pressure and taken back on next use), not\n"
  "                              refused. Same as session config\n"
  "                              memory_cap_mb; 0/unset = uncapped.\n"
  "  --plugin PATH               load a plugin .dylib at startup "
  "(repeatable).\n"
  "  --version                   print the version and build identity\n"
  "                              ('<major>.<minor> (<git-hash>*<dirty>)')\n"
  "                              and exit.\n"
  "  --gpu-info                  print the GPU/OS facts that decide whether\n"
  "                              the matrix-core (M5 NAX) kernels are used.\n"
  "  --list-views                print the stage types that have a web-UI\n"
  "                              panel, as JSON. A pipeline containing one\n"
  "                              needs someone at the browser to finish.\n"
  "  --list-models               print the built-in model catalogue as\n"
  "                              JSON and exit. For a front end that\n"
  "                              offers the same choices as the stages\n"
  "                              without keeping its own copy of them.\n"
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
  std::string              config;
  std::vector<Launch>      launches;
  std::vector<std::string> plugins;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::fputs(kUsage, stdout);
      return 0;
    } else if (a == "--version") {
      // The identity of the libvpipe this binary is linked against,
      // not a number baked in somewhere else -- so whatever reports it
      // (a bug report, the app's About pane) names the build that is
      // actually running.
      std::printf("%s\n", vpipe_version());
      return 0;
    } else if (a == "--gpu-info") {
      // What a bug report from someone else's Mac needs to carry. The
      // matrix-core path depends on BOTH the GPU family and the OS
      // version, and "it hangs on my M5" is unactionable without
      // knowing which of those was true.
#if !defined(VPIPE_BUILD_APPLE_SILICON)
      std::printf("{\"metal_valid\":false,"
                  "\"note\":\"not an Apple Silicon build\"}\n");
      return 0;
#else
      Session sess;
      const metal_compute::MetalCompute* mc = sess.metal_compute();
      char osbuf[64] = {0};
      size_t osn = sizeof(osbuf) - 1;
      ::sysctlbyname("kern.osproductversion", osbuf, &osn, nullptr, 0);
      char cpu[128] = {0};
      size_t cn = sizeof(cpu) - 1;
      ::sysctlbyname("machdep.cpu.brand_string", cpu, &cn, nullptr, 0);
      FlexData o  = FlexData::make_object();
      auto     oo = o.as_object();
      oo.insert_or_assign("cpu", FlexData::make_string(cpu));
      oo.insert_or_assign("macos", FlexData::make_string(osbuf));
      oo.insert_or_assign("metal_valid",
                          FlexData::make_bool(mc != nullptr && mc->valid()));
      oo.insert_or_assign(
        "matrix_cores",
        FlexData::make_bool(mc != nullptr && mc->supports_matrix_cores()));
      if (mc != nullptr) {
        // Why, not just what: each of these implies a different fix.
        auto g = mc->matrix_core_gate();
        FlexData why  = FlexData::make_object();
        auto     wo   = why.as_object();
        wo.insert_or_assign("device_family_apple10",
                            FlexData::make_bool(g.device_family));
        wo.insert_or_assign("macos_26_2_or_newer",
                            FlexData::make_bool(g.macos));
        wo.insert_or_assign("env_VPIPE_NO_MATRIX_CORES",
                            FlexData::make_bool(g.env_disabled));
        wo.insert_or_assign("env_VPIPE_FORCE_MATRIX_CORES",
                            FlexData::make_bool(g.env_forced));
        oo.insert_or_assign("matrix_cores_gate", std::move(why));
      }
      std::printf("%s\n", o.to_json().c_str());
      return 0;
#endif
    } else if (a == "--list-views") {
      // The stage types that have a GUI panel in the web UI. A front
      // end needs this to tell a pipeline that runs unattended from one
      // that will sit waiting for someone to draw a mask or pick a
      // region -- and it must come from the registry, because a list
      // copied into a GUI goes stale the moment a view is added.
      FlexData arr = FlexData::make_array();
      auto     out = arr.as_array();
      for (const UiViewSpec* v : UiViewRegistry::get().all()) {
        FlexData o = FlexData::make_object();
        auto     oo = o.as_object();
        oo.insert_or_assign("id", FlexData::make_string(std::string(v->id)));
        oo.insert_or_assign("stage_type",
                            FlexData::make_string(std::string(v->stage_type)));
        oo.insert_or_assign("label_key",
                            FlexData::make_string(std::string(v->label_key)));
        out.push_back(std::move(o));
      }
      std::printf("%s\n", arr.to_json().c_str());
      return 0;
    } else if (a == "--list-models") {
      // Before any SESSION is created: this must not pay for -- or be
      // able to fail because of -- LMDB or the model manager.
      //
      // Plugins are the exception, and they have to be. The catalogue is
      // the single source of what a front end may offer, and a plugin
      // can now contribute entries to it (register_catalog_entries), so
      // a listing that skipped plugins would tell a picker that a model
      // the very same command line just loaded does not exist. That is
      // the failure this flag exists to prevent, pointed the other way.
      //
      // Loading is still cheap and session-free: a dlopen and the
      // plugin's own register call, with a null session (every
      // registration path tolerates one -- it is used for logging).
      // argv is PRE-SCANNED because --plugin may appear after this flag
      // and this branch returns without finishing the parse.
      {
        std::vector<std::string> pre;
        if (const char* env = std::getenv("VPIPE_PLUGINS")) {
          std::stringstream ss(env);
          std::string one;
          while (std::getline(ss, one, ':')) {
            if (!one.empty()) { pre.push_back(one); }
          }
        }
        for (int j = 1; j < argc; ++j) {
          if (std::string(argv[j]) == "--plugin" && j + 1 < argc) {
            pre.push_back(argv[j + 1]);
          }
        }
        if (!pre.empty()) { PluginManager::get().load_all(nullptr, pre); }
      }
      FlexData arr = FlexData::make_array();
      auto     out = arr.as_array();
      for (const ModelCatalogEntry& e : model_catalog()) {
        out.push_back(catalog_entry_to_flex(e));
      }
      std::printf("%s\n", arr.to_json().c_str());
      return 0;
    } else if (a == "--config") {
      if (++i >= argc) { return arg_err("--config needs a value"); }
      config = argv[i];
    } else if (a == "--memory-cap-mb") {
      if (++i >= argc) { return arg_err("--memory-cap-mb needs a value"); }
      // Forwarded through the environment: the model manager is not
      // reachable from the public SessionIntf, and an env override is
      // this tree's convention for memory knobs (cf VPIPE_RAM_LIMIT_MB).
      ::setenv("VPIPE_MEMORY_CAP_MB", argv[i], 1);
    } else if (a == "--plugin") {
      if (++i >= argc) { return arg_err("--plugin needs a path"); }
      plugins.push_back(argv[i]);
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

  // --plugin PATH (repeatable): fold into VPIPE_PLUGINS (after any
  // pre-existing value) so the session's plugin-load hook picks them up
  // when it constructs. Config here is an opaque string (inline JSON or a
  // file path), so the env is the clean injection point.
  if (!plugins.empty()) {
    std::string joined;
    if (const char* ex = std::getenv("VPIPE_PLUGINS")) { joined = ex; }
    for (const std::string& p : plugins) {
      if (!joined.empty()) { joined += ':'; }
      joined += p;
    }
    ::setenv("VPIPE_PLUGINS", joined.c_str(), 1);
  }

  SessionManager& mgr = SessionManager::get();
  const SessionIntf* csess = mgr.create_session(config);
  if (csess == nullptr) {
    std::fprintf(stderr, "vpipe: failed to create session (bad --config?)\n");
    return 1;
  }
  SessionIntf* s = const_cast<SessionIntf*>(csess);

  // Install our own stdio UI delegate (identical behaviour to the one
  // the session builds by default) so the SIGINT handler has something
  // to hand a Ctrl-C to. The session owns it; we keep a borrowed
  // pointer for the duration of the wait loop below.
  StdioUiDelegate* ui = nullptr;
  if (Session* concrete = dynamic_cast<Session*>(s)) {
    auto owned = std::make_unique<StdioUiDelegate>();
    ui = owned.get();
    concrete->set_ui_delegate(std::move(owned));
    g_stdio_ui.store(ui, std::memory_order_release);
  }

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
    g_stdio_ui.store(nullptr, std::memory_order_release);
    mgr.destroy_session(s);
    return 1;
  }

  // Wait for every launched pipeline to drain, polling so a SIGINT/SIGTERM
  // can preempt long-running (or endless) pipelines with a clean stop.
  // A Ctrl-C is offered to the stages first (poll_sigint dispatches the
  // registered interrupt handlers); only an unconsumed or repeated one
  // reaches the stop path below.
  for (;;) {
    const Status w = s->wait_pipelines(250);
    if (w.code == 0) { break; }            // all reached idle
    const bool quit =
        g_interrupted.load() || (ui != nullptr && ui->poll_sigint());
    if (quit) {
      std::fprintf(stderr, "\nvpipe: interrupted -- stopping pipelines...\n");
      for (auto& h : handles) { s->stop_pipeline(h); }
      break;
    }
    // Otherwise Status{4} (timeout): keep waiting.
  }

  g_stdio_ui.store(nullptr, std::memory_order_release);
  mgr.destroy_session(s);
  return failed > 0 ? 1 : 0;
}

}  // namespace

int
main(int argc, char** argv)
{
  return run(argc, argv);
}
