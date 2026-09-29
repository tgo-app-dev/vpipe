#include "minitest.h"
#include "common/job.h"
#include "common/lmdb-env.h"
#include "common/session.h"
#include "common/stdout-log-delegate.h"
#include "common/vpipe-format.h"
#include "interfaces/log-delegate-intf.h"
#include "pipeline/pipeline-handle-impl.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/generative-model-manager.h"
#include <sys/sysctl.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <streambuf>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace std;

namespace {

// Recording delegate used by tests below to assert routing.
struct CapturingDelegate : public vpipe::LogDelegateIntf {
  struct Entry { vpipe::LogLevel level; string msg; };
  vector<Entry>   entries;
  vpipe::LogLevel thr = vpipe::LogLevel::Always;
  void log(vpipe::LogLevel l, const vpipe::VpipeFormat& f) override {
    entries.push_back({ l, f() });
  }
  void set_threshold(vpipe::LogLevel l) override { thr = l; }
  vpipe::LogLevel threshold() const noexcept override { return thr; }
};

// Recording UI delegate: error/warn/info route here (not the log
// delegate) after the UI-delegate split.
struct CapturingUiDelegate : public vpipe::UiDelegateIntf {
  struct Entry { string kind; string msg; };
  vector<Entry> entries;
  void error(const vpipe::VpipeFormat& f) override {
    entries.push_back({ "error", f() });
  }
  void warn(const vpipe::VpipeFormat& f) override {
    entries.push_back({ "warn", f() });
  }
  void info(const vpipe::VpipeFormat& f) override {
    entries.push_back({ "info", f() });
  }
  vpipe::UiInputStatus
  getline(const vpipe::VpipeFormat&, string&,
          const std::function<bool()>&) override {
    return vpipe::UiInputStatus::Eof;
  }
  std::unique_ptr<vpipe::UiTextStream> open_text_stream() override {
    return std::make_unique<vpipe::NullUiTextStream>();
  }
};

class CerrCapture {
public:
  CerrCapture() : _saved(cerr.rdbuf()) { cerr.rdbuf(_buf.rdbuf()); }
  ~CerrCapture() { cerr.rdbuf(_saved); }
private:
  streambuf*   _saved;
  stringstream _buf;
};

// Tmpdir helper used by lmdb-related tests below. Creates a fresh
// directory under the system temp root and removes it on dtor.
string
session_tmpdir_()
{
  auto base = filesystem::temp_directory_path() /
              "vpipe_session_lmdb_test_XXXXXX";
  string tmpl = base.string();
  if (!mkdtemp(tmpl.data())) {
    throw runtime_error("mkdtemp failed");
  }
  return tmpl;
}

struct SessionTempDir {
  string path;
  SessionTempDir() : path(session_tmpdir_()) {}
  ~SessionTempDir() {
    error_code ec;
    filesystem::remove_all(path, ec);
  }
};

// Zero-iport, zero-oport stage. Used by the log-override gate test:
// launching it through Session::launch_pipeline flips the
// any_pipeline_launched() check, no edge wiring needed.
class NoopSourceStage
  : public vpipe::TypedStage<NoopSourceStage> {
public:
  static constexpr const char* kTypeName = "ut-noop-source";
  using TypedStage::TypedStage;

  vpipe::Job
  process(vpipe::RuntimeContext& ctx) override
  {
    ctx.signal_done();
    co_return;
  }
};
VPIPE_REGISTER_STAGE(NoopSourceStage)

}

TEST(session, error) {
  // Default-constructed Session installs a StdoutLogDelegate. The
  // delegate writes to cerr before error() throws; capture cerr so
  // CI output stays clean.
  CerrCapture cap;
  vpipe::Session x;
  bool hit_error = false;
  try {
    x.error(vpipe::fmt("test {}.", "error"));
  } catch (exception&) {
    hit_error = true;
  }
  EXPECT_TRUE(hit_error);
}

TEST(session, log_methods_route_to_delegate) {
  // log_* (debug/verbose/normal/always) route to the log delegate;
  // error/warn/info route to the UI delegate after the split.
  auto cap = make_unique<CapturingDelegate>();
  CapturingDelegate* raw = cap.get();
  vpipe::Session s(std::move(cap));
  auto uicap = make_unique<CapturingUiDelegate>();
  CapturingUiDelegate* uiraw = uicap.get();
  s.set_ui_delegate(std::move(uicap));

  s.warn       (vpipe::fmt("a"));
  s.info       (vpipe::fmt("b"));
  s.log_debug  (vpipe::fmt("c"));
  s.log_verbose(vpipe::fmt("d"));
  s.log_normal (vpipe::fmt("e"));
  s.log_always (vpipe::fmt("f"));

  // Log delegate sees only the four log_* levels.
  EXPECT_TRUE(raw->entries.size() == 4);
  EXPECT_TRUE(raw->entries[0].level == vpipe::LogLevel::Debug);
  EXPECT_TRUE(raw->entries[1].level == vpipe::LogLevel::Verbose);
  EXPECT_TRUE(raw->entries[2].level == vpipe::LogLevel::Normal);
  EXPECT_TRUE(raw->entries[3].level == vpipe::LogLevel::Always);

  // UI delegate sees warn + info, in order.
  EXPECT_TRUE(uiraw->entries.size() == 2);
  EXPECT_TRUE(uiraw->entries[0].kind == "warn");
  EXPECT_TRUE(uiraw->entries[0].msg  == "a");
  EXPECT_TRUE(uiraw->entries[1].kind == "info");
  EXPECT_TRUE(uiraw->entries[1].msg  == "b");
}

TEST(session, error_routes_then_throws) {
  vpipe::Session s;
  auto uicap = make_unique<CapturingUiDelegate>();
  CapturingUiDelegate* uiraw = uicap.get();
  s.set_ui_delegate(std::move(uicap));
  bool threw = false;
  try {
    s.error(vpipe::fmt("boom-{}", 7));
  } catch (exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
  EXPECT_TRUE(uiraw->entries.size() == 1);
  EXPECT_TRUE(uiraw->entries[0].kind == "error");
  EXPECT_TRUE(uiraw->entries[0].msg  == "boom-7");
}

TEST(session, json_config_selects_log_level) {
  // Empty JSON object -> defaults; verify we don't crash and config
  // round-trips into Session::config().
  vpipe::Session s(R"({"log":{"level":"verbose"}})");
  EXPECT_TRUE(s.config().is_object());
  EXPECT_TRUE(s.config().as_object()
                  .at("log").as_object()
                  .at("level").get_string() == "verbose");
}

TEST(session, malformed_config_falls_back_softly) {
  // A bad config string should warn (to cerr) and continue, not
  // crash and not throw.
  CerrCapture cap;
  bool threw = false;
  try {
    vpipe::Session s("not-a-json-and-not-a-real-path");
    EXPECT_TRUE(s.config().is_object());
  } catch (...) {
    threw = true;
  }
  EXPECT_FALSE(threw);
}

TEST(session, log_overrides_blocked_while_launched) {
  // log_to_db with no db.path now opens an env at "." (CWD). To
  // keep the test's lmdb files out of the runner's working
  // directory we run inside a fresh tmpdir.
  SessionTempDir dir;
  error_code ec;
  filesystem::path saved_cwd = filesystem::current_path(ec);
  ASSERT_FALSE(static_cast<bool>(ec));
  filesystem::current_path(dir.path, ec);
  ASSERT_FALSE(static_cast<bool>(ec));

  vpipe::Session sess;

  // Idle: all four mutators succeed.
  EXPECT_TRUE(sess.debug_level(2u).code        == 0u);
  EXPECT_TRUE(sess.debug_level(string_view("warn")).code == 0u);
  EXPECT_TRUE(sess.log_to_stdout().code        == 0u);
  // log_to_db falls back to CWD; succeeds with Status{0}.
  EXPECT_TRUE(sess.log_to_db().code            == 0u);

  // Launch a 1-stage pipeline. The stage signals done immediately;
  // the runtime stays attached until stop_pipeline, which is what
  // any_pipeline_launched() observes.
  auto handle = sess.create_pipeline("p");
  auto* pl    = vpipe::HandleAccess::impl(handle)->pipeline();
  pl->insert_stage(std::make_unique<NoopSourceStage>(
      &sess, "src", std::vector<vpipe::InEdge>{}));
  EXPECT_TRUE(sess.launch_pipeline(handle).code == 0u);

  // Busy: every override is rejected with Status{3}.
  CerrCapture cap;  // suppress the warn-to-stderr output
  EXPECT_TRUE(sess.debug_level(2u).code        == 3u);
  EXPECT_TRUE(sess.debug_level(string_view("warn")).code == 3u);
  EXPECT_TRUE(sess.log_to_stdout().code        == 3u);
  EXPECT_TRUE(sess.log_to_db().code            == 3u);

  // Stop drops the runtime; mutators succeed again.
  EXPECT_TRUE(sess.stop_pipeline(handle).code  == 0u);
  EXPECT_TRUE(sess.debug_level(2u).code        == 0u);
  EXPECT_TRUE(sess.unload_pipeline(handle).code == 0u);

  // Restore CWD so the tmpdir's removal in ~SessionTempDir
  // succeeds and other tests don't inherit a deleted directory.
  filesystem::current_path(saved_cwd, ec);
}

// The wired pool is RAISABLE while a pipeline runs and not lowerable.
//
// The asymmetry is the whole point and it is not a policy preference:
// making a lower limit true means unwiring buffers a model is still
// reading, and a raise costs nothing because nothing has to be given
// back. So a shrink is Status{3} -- the same "busy" code the log
// mutators use -- and the limit is left exactly as it was.
TEST(session, wired_pool_shrinks_only_while_idle) {
  SessionTempDir dir;
  error_code ec;
  filesystem::path saved_cwd = filesystem::current_path(ec);
  ASSERT_FALSE(static_cast<bool>(ec));
  filesystem::current_path(dir.path, ec);

  vpipe::Session sess;
  if (sess.generative_model_manager() == nullptr) {
    filesystem::current_path(saved_cwd, ec);
    return;                       // no manager in this build
  }
  // A middling absolute figure, so there is room to move both ways.
  ASSERT_TRUE(sess.set_wired_pool_mb(4096).code == 0u);
  const size_t base = sess.wired_pool_mb();
  EXPECT_TRUE(base == 4096u);

  auto handle = sess.create_pipeline("wp");
  auto* pl    = vpipe::HandleAccess::impl(handle)->pipeline();
  pl->insert_stage(std::make_unique<NoopSourceStage>(
      &sess, "src", std::vector<vpipe::InEdge>{}));
  EXPECT_TRUE(sess.launch_pipeline(handle).code == 0u);

  {
    CerrCapture cap;              // the refusal warns to stderr
    EXPECT_TRUE(sess.set_wired_pool_mb(512).code == 3u);
  }
  // UNCHANGED, not partially applied: a rejected request must leave the
  // limit where it was, or a front end that retries compounds it.
  EXPECT_TRUE(sess.wired_pool_mb() == base);

  // Raising is accepted mid-run.
  EXPECT_TRUE(sess.set_wired_pool_mb(6144).code == 0u);
  EXPECT_TRUE(sess.wired_pool_mb() == 6144u);

  // ...and once stopped, lowering works again.
  EXPECT_TRUE(sess.stop_pipeline(handle).code == 0u);
  EXPECT_TRUE(sess.set_wired_pool_mb(512).code == 0u);
  EXPECT_TRUE(sess.wired_pool_mb() == 512u);
  EXPECT_TRUE(sess.unload_pipeline(handle).code == 0u);

  filesystem::current_path(saved_cwd, ec);
}

// The swap allowance moves in BOTH directions at any time, and the
// contrast with the wired pool above is the point of the test rather
// than an incidental difference.
//
// The pool refuses a shrink mid-run because its bytes are already
// mlock'd: making a lower limit true means unwiring buffers a model is
// still reading. The allowance reserves nothing -- it only widens the
// arithmetic of the next preflight -- so there is nothing to give back
// and no state a lower figure could invalidate. A Status{3} here would
// mean someone had copied the pool's rule without its reason.
TEST(session, swap_allowance_moves_both_ways_while_running) {
  SessionTempDir dir;
  error_code ec;
  filesystem::path saved_cwd = filesystem::current_path(ec);
  ASSERT_FALSE(static_cast<bool>(ec));
  filesystem::current_path(dir.path, ec);

  vpipe::Session sess;
  if (sess.generative_model_manager() == nullptr) {
    filesystem::current_path(saved_cwd, ec);
    return;                       // no manager in this build
  }
  // The SHIPPED default, asserted so a change to it is a deliberate
  // edit to this line rather than a silent drift.
  EXPECT_TRUE(sess.swap_allowance_mb() == 4096u);

  auto handle = sess.create_pipeline("swap");
  auto* pl    = vpipe::HandleAccess::impl(handle)->pipeline();
  pl->insert_stage(std::make_unique<NoopSourceStage>(
      &sess, "src", std::vector<vpipe::InEdge>{}));
  EXPECT_TRUE(sess.launch_pipeline(handle).code == 0u);

  // DOWN while running -- accepted, where the wired pool returns 3.
  EXPECT_TRUE(sess.set_swap_allowance_mb(1024).code == 0u);
  EXPECT_TRUE(sess.swap_allowance_mb() == 1024u);
  // ...and up again.
  EXPECT_TRUE(sess.set_swap_allowance_mb(8192).code == 0u);
  EXPECT_TRUE(sess.swap_allowance_mb() == 8192u);

  // 0 is a REAL setting -- size a forward against reclaimable RAM alone
  // -- and not a request for the default. A getter reporting 4096 here
  // would mean the sentinel had leaked into the value.
  EXPECT_TRUE(sess.set_swap_allowance_mb(0).code == 0u);
  EXPECT_TRUE(sess.swap_allowance_mb() == 0u);

  EXPECT_TRUE(sess.stop_pipeline(handle).code == 0u);
  EXPECT_TRUE(sess.unload_pipeline(handle).code == 0u);

  filesystem::current_path(saved_cwd, ec);
}

// THE CONFIG PATH, which the setter test above does not touch: a
// figure typed into a session config (or forwarded from --swap-
// allowance-mb through the environment) has to reach the manager at
// construction, and an UNSET key has to leave the shipped default
// alone rather than overwriting it with zero.
//
// That last case is the one worth a test. The parser uses a sentinel to
// tell "not configured" from an explicit 0, because 0 is a real setting
// meaning "size against reclaimable RAM alone" -- collapsing the two
// would silently disable the allowance for every session that never
// mentions it.
TEST(session, swap_allowance_comes_from_config) {
  {
    vpipe::Session sess(R"({"swap_allowance_mb": 1234})");
    if (sess.generative_model_manager() == nullptr) { return; }
    EXPECT_TRUE(sess.swap_allowance_mb() == 1234u);
  }
  // An explicit 0 is honoured, not read as "unset".
  {
    vpipe::Session sess(R"({"swap_allowance_mb": 0})");
    EXPECT_TRUE(sess.swap_allowance_mb() == 0u);
  }
  // No key at all -> the manager's own default stands.
  {
    vpipe::Session sess(R"({"log_level": "info"})");
    EXPECT_TRUE(sess.swap_allowance_mb() == 4096u);
  }
  // The environment overrides the config, the way the other memory
  // knobs do -- this is how both apps forward their flag.
  {
    ::setenv("VPIPE_SWAP_ALLOWANCE_MB", "2048", 1);
    vpipe::Session sess(R"({"swap_allowance_mb": 1234})");
    EXPECT_TRUE(sess.swap_allowance_mb() == 2048u);
    ::unsetenv("VPIPE_SWAP_ALLOWANCE_MB");
  }
}

#ifdef VPIPE_BUILD_APPLE_SILICON
// The supply side. Deliberately a WEAK assertion: this reads the live
// machine, so the only thing true on every box is that the figure is
// sane -- it cannot exceed physical RAM, and it must not come back as a
// wrapped-around huge number when the subtractions overshoot (which is
// the arithmetic bug worth catching, since every term is unsigned).
TEST(session, swappable_other_bytes_is_bounded) {
  const size_t spare =
      vpipe::genai::GenerativeModelManager::swappable_other_bytes();
  size_t ram = 0;
  uint64_t mem = 0; size_t len = sizeof(mem);
  if (::sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
    ram = (size_t)mem;
  }
  if (ram > 0) { EXPECT_TRUE(spare < ram); }
}
#endif

// Chdir into `dir` for the duration, then restore -- the db-path tests
// all decide from the CWD, and a test that left the process somewhere
// else would move the ones after it.
struct ScopedCwd {
  filesystem::path saved;
  bool             ok = false;
  explicit ScopedCwd(const filesystem::path& to) {
    error_code ec;
    saved = filesystem::current_path(ec);
    if (ec) { return; }
    filesystem::current_path(to, ec);
    ok = !ec;
  }
  ~ScopedCwd() {
    if (saved.empty()) { return; }
    error_code ec;
    filesystem::current_path(saved, ec);
  }
};

// Materialise data.mdb / lock.mdb in `dir` by opening an env there.
bool
make_db_(const filesystem::path& dir)
{
  vpipe::Session sess(R"({"db":{"path":")" + dir.string() + R"("}})");
  vpipe::LmdbEnv* env = sess.lmdb_env();
  error_code ec;
  return env != nullptr && env->valid() &&
         filesystem::exists(dir / "data.mdb", ec);
}

TEST(session, lmdb_env_defaults_to_cwd_without_db_path) {
  // No db.path in config => lmdb_env() opens "." (the process
  // CWD). To avoid sprinkling data.mdb / lock.mdb files into the
  // build directory we chdir into a fresh tmpdir, exercise the
  // accessor, then restore the original CWD before the tmpdir is
  // removed.
  //
  // TWO levels down, and the intermediate is left empty on purpose: the
  // default is "." only while the parent has no database of its own (see
  // the sandbox test below), and one level down would make that depend on
  // whatever happens to sit in the system temp directory.
  SessionTempDir dir;
  error_code ec;
  const filesystem::path cwd = filesystem::path(dir.path) / "empty" / "run";
  filesystem::create_directories(cwd, ec);
  ASSERT_FALSE(static_cast<bool>(ec));
  ScopedCwd chdir_(cwd);
  ASSERT_TRUE(chdir_.ok);
  if (!chdir_.ok) { return; }
  vpipe::Session sess;
  vpipe::LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);
  if (env) {
    EXPECT_TRUE(env->valid());
    EXPECT_TRUE(env->path() == ".");
  }
}

TEST(session, lmdb_env_adopts_the_database_beside_a_sandbox_cwd) {
  // The web-ui's layout: the database sits in its starting directory and
  // the pipelines it writes live in `sandbox` beneath it. A CLI run from
  // inside `sandbox` finds no database in the CWD and must adopt the one
  // beside it rather than create a second, empty one -- otherwise every
  // registered model misses.
  SessionTempDir dir;
  const filesystem::path parent(dir.path);
  const filesystem::path sandbox = parent / "sandbox";
  error_code ec;
  filesystem::create_directories(sandbox, ec);
  ASSERT_FALSE(static_cast<bool>(ec));
  ASSERT_TRUE(make_db_(parent));

  ScopedCwd chdir_(sandbox);
  ASSERT_TRUE(chdir_.ok);
  if (!chdir_.ok) { return; }
  vpipe::Session sess;
  vpipe::LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);
  if (env) {
    EXPECT_TRUE(env->valid());
    // Canonical on both sides: the adopted path comes from
    // current_path(), which resolves symlinks, while temp_directory_path()
    // does not -- on macOS that is /private/var against /var for the same
    // directory.
    error_code e1, e2;
    EXPECT_TRUE(filesystem::canonical(filesystem::path(string(env->path())),
                                      e1) ==
                filesystem::canonical(parent, e2));
    EXPECT_FALSE(static_cast<bool>(e1));
    EXPECT_FALSE(static_cast<bool>(e2));
  }
  // ...and nothing was created in the sandbox itself.
  EXPECT_FALSE(filesystem::exists(sandbox / "data.mdb", ec));
}

TEST(session, lmdb_env_keeps_a_database_that_is_in_the_cwd) {
  // The CWD's own database WINS over the parent's, so a nested working
  // directory that has been used before keeps using what it has. Both
  // directories hold one here, which is the only arrangement that can
  // tell the two rules apart.
  SessionTempDir dir;
  const filesystem::path parent(dir.path);
  const filesystem::path child = parent / "child";
  error_code ec;
  filesystem::create_directories(child, ec);
  ASSERT_FALSE(static_cast<bool>(ec));
  ASSERT_TRUE(make_db_(parent));
  ASSERT_TRUE(make_db_(child));

  ScopedCwd chdir_(child);
  ASSERT_TRUE(chdir_.ok);
  if (!chdir_.ok) { return; }
  vpipe::Session sess;
  vpipe::LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);
  if (env) { EXPECT_TRUE(env->path() == "."); }
}

TEST(session, lmdb_env_ignores_a_parent_holding_only_a_lock_file) {
  // A stray lock.mdb is not a database: adopting a directory on the
  // strength of one would point the run at an empty env that LMDB then
  // creates beside the lock. The DATA file is what says a database is
  // there.
  SessionTempDir dir;
  const filesystem::path parent(dir.path);
  const filesystem::path child = parent / "child";
  error_code ec;
  filesystem::create_directories(child, ec);
  ASSERT_FALSE(static_cast<bool>(ec));
  { ofstream(parent / "lock.mdb") << "not a database"; }

  ScopedCwd chdir_(child);
  ASSERT_TRUE(chdir_.ok);
  if (!chdir_.ok) { return; }
  vpipe::Session sess;
  vpipe::LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);
  if (env) { EXPECT_TRUE(env->path() == "."); }
}

TEST(session, configured_db_path_beats_the_sandbox_rule) {
  // An explicit db.path is used as given, whatever the CWD looks like.
  SessionTempDir dir;
  const filesystem::path parent(dir.path);
  const filesystem::path sandbox = parent / "sandbox";
  const filesystem::path elsewhere = parent / "elsewhere";
  error_code ec;
  filesystem::create_directories(sandbox, ec);
  filesystem::create_directories(elsewhere, ec);
  ASSERT_FALSE(static_cast<bool>(ec));
  ASSERT_TRUE(make_db_(parent));

  ScopedCwd chdir_(sandbox);
  ASSERT_TRUE(chdir_.ok);
  if (!chdir_.ok) { return; }
  vpipe::Session sess(R"({"db":{"path":")" + elsewhere.string() +
                      R"("}})");
  vpipe::LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);
  if (env) { EXPECT_TRUE(env->path() == elsewhere.string()); }
}

TEST(session, lmdb_env_lazily_opens_when_db_path_configured) {
  SessionTempDir dir;
  string cfg = R"({"db":{"path":")" + dir.path + R"("}})";
  vpipe::Session sess(cfg);
  vpipe::LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);
  // Second call returns the same env -- it's cached.
  EXPECT_TRUE(sess.lmdb_env() == env);
  EXPECT_TRUE(env->valid());
}

TEST(session, log_to_db_uses_session_env) {
  SessionTempDir dir;
  string cfg = R"({"db":{"path":")" + dir.path + R"("}})";
  vpipe::Session sess(cfg);
  // The bootstrap delegate is stdout (since log.delegate isn't
  // "db"); but log_to_db() should now succeed against the
  // session-shared env.
  EXPECT_TRUE(sess.log_to_db().code == 0u);
}

namespace {

// Build a chrono(count=3, ~50Hz) -> shell pipeline through the
// public PipelineHandle API. Returns the handle with the pipeline
// already populated; not yet launched.
vpipe::PipelineHandle
build_chrono_shell_handle_(vpipe::Session& sess,
                           const string&   pl_id,
                           const string&   shell_cmd)
{
  auto pl = sess.create_pipeline(pl_id);
  auto src = pl.insert_stage(
      "chrono", "src", {},
      R"({"frequency_hz": 50.0, "count": 3})");
  string cfg = R"({"command":")" + shell_cmd + R"("})";
  pl.insert_stage("shell", "sink",
                  vector<vpipe::StagePortHandle>{{src, 0u}}, cfg);
  return pl;
}

}

TEST(session, store_then_load_round_trips_json) {
  SessionTempDir dir;
  vpipe::Session sess;
  string out_path = dir.path + "/p.json";
  string tick     = dir.path + "/p.json.tick";

  auto h = build_chrono_shell_handle_(
      sess, "p", "echo tick >> " + tick);
  ASSERT_TRUE(sess.store_pipeline(h, out_path).code == 0u);
  // File must exist and start with `{`.
  {
    ifstream in(out_path);
    ASSERT_TRUE(in.good());
    char c = '\0';
    in >> c;
    EXPECT_TRUE(c == '{');
  }

  // Unload + load back; verify the resulting pipeline runs and
  // produces the same shell-driven side effect.
  ASSERT_TRUE(sess.unload_pipeline(h).code == 0u);
  auto h2 = sess.load_pipeline(out_path);
  ASSERT_TRUE(h2.valid());

  ASSERT_TRUE(sess.launch_pipeline(h2).code == 0u);
  this_thread::sleep_for(chrono::milliseconds(200));
  ASSERT_TRUE(sess.stop_pipeline(h2).code == 0u);

  ifstream in(tick);
  size_t lines = 0;
  string line;
  while (getline(in, line)) ++lines;
  EXPECT_TRUE(lines == 3u);
  ::unlink(tick.c_str());
}

TEST(session, store_then_load_round_trips_binary) {
  SessionTempDir dir;
  vpipe::Session sess;
  string out_path = dir.path + "/p.bin";
  string tick     = dir.path + "/p.bin.tick";

  auto h = build_chrono_shell_handle_(
      sess, "p", "echo tick >> " + tick);
  ASSERT_TRUE(sess.store_pipeline(h, out_path).code == 0u);
  // File must NOT start with `{` (binary FlexData).
  {
    ifstream in(out_path, ios::binary);
    ASSERT_TRUE(in.good());
    char c = '\0';
    in.read(&c, 1);
    EXPECT_TRUE(c != '{' && c != '[');
  }

  ASSERT_TRUE(sess.unload_pipeline(h).code == 0u);
  auto h2 = sess.load_pipeline(out_path);
  ASSERT_TRUE(h2.valid());

  ASSERT_TRUE(sess.launch_pipeline(h2).code == 0u);
  this_thread::sleep_for(chrono::milliseconds(200));
  ASSERT_TRUE(sess.stop_pipeline(h2).code == 0u);

  ifstream in(tick);
  size_t lines = 0;
  string line;
  while (getline(in, line)) ++lines;
  EXPECT_TRUE(lines == 3u);
  ::unlink(tick.c_str());
}

TEST(session, store_pipeline_no_path_returns_status_1) {
  vpipe::Session sess;
  auto h = build_chrono_shell_handle_(sess, "p", "true");
  // create_pipeline doesn't set a storage path; no-arg
  // store_pipeline must reject.
  CerrCapture cap;
  vpipe::Status s = sess.store_pipeline(h);
  EXPECT_TRUE(s.code == 1u);
}

TEST(session, store_pipeline_with_path_then_no_arg_works) {
  SessionTempDir dir;
  vpipe::Session sess;
  string out_path = dir.path + "/p.json";
  auto h = build_chrono_shell_handle_(sess, "p", "true");
  ASSERT_TRUE(sess.store_pipeline(h, out_path).code == 0u);
  // Now the no-arg form should write to the same path.
  ASSERT_TRUE(sess.store_pipeline(h).code == 0u);
}
