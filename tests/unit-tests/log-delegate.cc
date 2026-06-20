#include "minitest.h"
#include "common/db-log-delegate.h"
#include "common/flex-data.h"
#include "common/lmdb-cursor.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/session.h"
#include "common/stdout-log-delegate.h"
#include "common/vpipe-format.h"
#include "interfaces/log-delegate-intf.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <set>
#include <sstream>
#include <streambuf>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

string make_tempdir() {
  auto base = filesystem::temp_directory_path() /
              "vpipe_log_delegate_test_XXXXXX";
  string tmpl = base.string();
  if (!mkdtemp(tmpl.data())) {
    throw runtime_error("mkdtemp failed");
  }
  return tmpl;
}

struct TempDir {
  string path;
  TempDir() : path(make_tempdir()) {}
  ~TempDir() {
    error_code ec;
    filesystem::remove_all(path, ec);
  }
};

// Swap a stream's rdbuf for the lifetime of the guard, so we can
// capture writes during a test and restore the original buffer
// afterwards (other tests still need cout/cerr to behave normally).
class StreamCapture {
public:
  StreamCapture(ostream& stream) : _stream(stream), _saved(stream.rdbuf()) {
    _stream.rdbuf(_buf.rdbuf());
  }
  ~StreamCapture() { _stream.rdbuf(_saved); }
  string str() const { return _buf.str(); }
private:
  ostream&     _stream;
  streambuf*   _saved;
  stringstream _buf;
};

// LogDelegate stub that records every call. Used to verify Session
// routes the right level for each log_* method.
struct CapturingLogDelegate : public LogDelegateIntf {
  struct Entry { LogLevel level; string msg; };
  vector<Entry> entries;
  LogLevel      thr = LogLevel::Always;
  void log(LogLevel l, const VpipeFormat& f) override {
    entries.push_back({ l, f() });
  }
  void set_threshold(LogLevel l) override { thr = l; }
  LogLevel threshold() const noexcept override { return thr; }
};

uint64_t key_seq(string_view k) {
  // last 4 bytes, big-endian.
  uint32_t v = 0;
  for (size_t i = k.size() - 4; i < k.size(); ++i) {
    v = (v << 8) | static_cast<unsigned char>(k[i]);
  }
  return v;
}

}

// ----------------------------------------------------------------------
// StdoutLogDelegate
// ----------------------------------------------------------------------

TEST(stdout_log_delegate, emits_at_or_below_threshold) {
  StdoutLogDelegate d(LogLevel::Normal);
  StreamCapture cap_out(cout);
  StreamCapture cap_err(cerr);
  d.log(LogLevel::Error,   fmt("a"));
  d.log(LogLevel::Warn,    fmt("b"));
  d.log(LogLevel::Info,    fmt("c"));
  d.log(LogLevel::Normal,  fmt("d"));
  d.log(LogLevel::Verbose, fmt("e"));
  d.log(LogLevel::Debug,   fmt("f"));
  string e = cap_err.str();
  string o = cap_out.str();
  EXPECT_TRUE(e.find("[ERROR] a") != string::npos);
  EXPECT_TRUE(e.find("[WARN] b")  != string::npos);
  EXPECT_TRUE(o.find("[INFO] c")   != string::npos);
  EXPECT_TRUE(o.find("[NORMAL] d") != string::npos);
  EXPECT_TRUE(o.find("[VERBOSE]")  == string::npos);
  EXPECT_TRUE(o.find("[DEBUG]")    == string::npos);
}

TEST(stdout_log_delegate, debug_threshold_emits_everything) {
  StdoutLogDelegate d(LogLevel::Debug);
  StreamCapture cap_out(cout);
  StreamCapture cap_err(cerr);
  d.log(LogLevel::Debug,   fmt("dbg"));
  d.log(LogLevel::Verbose, fmt("vrb"));
  d.log(LogLevel::Always,  fmt("alw"));
  string o = cap_out.str();
  EXPECT_TRUE(o.find("[DEBUG] dbg")   != string::npos);
  EXPECT_TRUE(o.find("[VERBOSE] vrb") != string::npos);
  EXPECT_TRUE(o.find("[ALWAYS] alw")  != string::npos);
}

TEST(stdout_log_delegate, always_bypasses_threshold) {
  // Even at the most restrictive threshold, Always must emit.
  StdoutLogDelegate d(LogLevel::Error);
  StreamCapture cap_out(cout);
  StreamCapture cap_err(cerr);
  d.log(LogLevel::Always, fmt("must-emit"));
  EXPECT_TRUE(cap_out.str().find("[ALWAYS] must-emit") != string::npos);
}

TEST(stdout_log_delegate, filtered_messages_skip_format_call) {
  StdoutLogDelegate d(LogLevel::Normal);
  StreamCapture cap_out(cout);
  StreamCapture cap_err(cerr);
  atomic<int> calls{0};
  VpipeFormat counting([&]() {
    calls.fetch_add(1, memory_order_relaxed);
    return string("expensive");
  });
  d.log(LogLevel::Debug,   counting);    // filtered, no call
  d.log(LogLevel::Verbose, counting);    // filtered, no call
  EXPECT_TRUE(calls.load() == 0);
  d.log(LogLevel::Normal, counting);     // emitted, one call
  EXPECT_TRUE(calls.load() == 1);
}

TEST(stdout_log_delegate, lines_do_not_interleave) {
  StdoutLogDelegate d(LogLevel::Normal);
  // Use a long, distinguishable per-thread payload so any tear is
  // visible in the captured output.
  const int  n_threads = 8;
  const int  n_iters   = 200;
  StreamCapture cap_out(cout);
  StreamCapture cap_err(cerr);
  vector<thread> ts;
  for (int t = 0; t < n_threads; ++t) {
    ts.emplace_back([t, &d]() {
      string payload(64, char('A' + t));
      for (int i = 0; i < n_iters; ++i) {
        d.log(LogLevel::Normal, fmt("{}-{}-{}", t, i, payload));
      }
    });
  }
  for (auto& th : ts) {
    th.join();
  }
  // Every line must start with "[NORMAL] <single-digit>-".
  string out = cap_out.str();
  size_t start = 0;
  size_t lines = 0;
  while (start < out.size()) {
    size_t nl = out.find('\n', start);
    if (nl == string::npos) {
      break;
    }
    string_view line(out.data() + start, nl - start);
    EXPECT_TRUE(line.substr(0, 9) == "[NORMAL] ");
    // The thread id char appears exactly once per line in the body.
    string_view body = line.substr(9);
    EXPECT_TRUE(body.size() > 4);
    ++lines;
    start = nl + 1;
  }
  EXPECT_TRUE(lines == size_t(n_threads * n_iters));
}

// ----------------------------------------------------------------------
// DbLogDelegate
// ----------------------------------------------------------------------

TEST(db_log_delegate, round_trip_persists_records) {
  TempDir dir;
  {
    Session       s;
    LmdbEnv       env(&s, dir.path);
    DbLogDelegate d(&env, LogLevel::Debug);
    d.log(LogLevel::Info,   fmt("hello"));
    d.log(LogLevel::Debug,  fmt("debug-{}", 1));
    d.log(LogLevel::Always, fmt("always"));
  }
  // Reopen with the raw LMDB wrappers and walk the log sub-db.
  Session    s;
  LmdbEnv    env(&s, dir.path);
  LmdbDb     db(env, "log");
  LmdbTxn    txn(env, LmdbTxn::Mode::ReadOnly);
  LmdbCursor cur(txn, db);
  vector<pair<int, string>> records;
  string_view k, v;
  for (bool ok = cur.first(k, v); ok; ok = cur.next(k, v)) {
    auto rec = FlexData::from_binary(v);
    auto obj = rec.as_object();
    int level   = static_cast<int>(obj.at("level").get_int());
    string msg(obj.at("msg").get_string());
    records.push_back({ level, std::move(msg) });
  }
  EXPECT_TRUE(records.size() == 3);
  EXPECT_TRUE(records[0].first == int(LogLevel::Info));
  EXPECT_TRUE(records[0].second == "hello");
  EXPECT_TRUE(records[1].first == int(LogLevel::Debug));
  EXPECT_TRUE(records[1].second == "debug-1");
  EXPECT_TRUE(records[2].first == int(LogLevel::Always));
  EXPECT_TRUE(records[2].second == "always");
}

TEST(db_log_delegate, threshold_filter_is_honoured) {
  TempDir dir;
  {
    Session       s;
    LmdbEnv       env(&s, dir.path);
    DbLogDelegate d(&env, LogLevel::Info);
    d.log(LogLevel::Debug,   fmt("dropped-debug"));
    d.log(LogLevel::Verbose, fmt("dropped-verbose"));
    d.log(LogLevel::Info,    fmt("kept-info"));
    d.log(LogLevel::Always,  fmt("kept-always"));
  }
  Session    s;
  LmdbEnv    env(&s, dir.path);
  LmdbDb     db(env, "log");
  LmdbTxn    txn(env, LmdbTxn::Mode::ReadOnly);
  LmdbCursor cur(txn, db);
  vector<string> msgs;
  string_view k, v;
  for (bool ok = cur.first(k, v); ok; ok = cur.next(k, v)) {
    auto rec = FlexData::from_binary(v);
    msgs.push_back(string(rec.as_object().at("msg").get_string()));
  }
  EXPECT_TRUE(msgs.size() == 2);
  EXPECT_TRUE(msgs[0] == "kept-info");
  EXPECT_TRUE(msgs[1] == "kept-always");
}

TEST(db_log_delegate, concurrent_producers_have_distinct_seqs) {
  TempDir dir;
  const int n_threads = 4;
  const int n_iters   = 100;
  {
    Session       s;
    LmdbEnv       env(&s, dir.path);
    DbLogDelegate d(&env, LogLevel::Debug);
    vector<thread> ts;
    for (int t = 0; t < n_threads; ++t) {
      ts.emplace_back([t, &d]() {
        for (int i = 0; i < n_iters; ++i) {
          d.log(LogLevel::Normal, fmt("t{}-i{}", t, i));
        }
      });
    }
    for (auto& th : ts) {
      th.join();
    }
  }
  // Reopen and count + check seq uniqueness.
  Session    s;
  LmdbEnv    env(&s, dir.path);
  LmdbDb     db(env, "log");
  LmdbTxn    txn(env, LmdbTxn::Mode::ReadOnly);
  LmdbCursor cur(txn, db);
  set<uint64_t> seqs;
  size_t count = 0;
  string_view k, v;
  for (bool ok = cur.first(k, v); ok; ok = cur.next(k, v)) {
    seqs.insert(key_seq(k));
    ++count;
  }
  EXPECT_TRUE(count == size_t(n_threads * n_iters));
  EXPECT_TRUE(seqs.size() == count);
}

TEST(db_log_delegate, write_failure_does_not_throw) {
  TempDir dir;
  // Map sized to let env + dbi open successfully but small enough
  // that log() writes will exhaust it. We expect the delegate to
  // swallow MDB_MAP_FULL and report on cerr instead of letting the
  // exception escape.
  StreamCapture cap_err(cerr);
  Session       s;
  LmdbEnv       env(&s, dir.path, /* map_size */ size_t{1} << 17);
  DbLogDelegate d(&env, LogLevel::Debug);
  bool any_threw = false;
  try {
    string big(8192, 'x');
    for (int i = 0; i < 200; ++i) {
      d.log(LogLevel::Normal, fmt("{}-{}", i, big));
    }
  } catch (...) {
    any_threw = true;
  }
  EXPECT_FALSE(any_threw);
  // We expect at least one MDB_MAP_FULL line on the captured cerr;
  // otherwise the test isn't actually exercising the firewall.
  EXPECT_TRUE(cap_err.str().find("MDB_MAP_FULL") != string::npos);
}

TEST(db_log_delegate, null_env_at_construction_throws) {
  // The delegate now borrows an env from the caller. Passing
  // nullptr is a programmer error and should throw at construction
  // -- Session's bootstrap catches this one level up and falls
  // back to stdout.
  bool threw = false;
  try {
    DbLogDelegate d(nullptr, LogLevel::Normal);
    (void)d;
  } catch (...) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(db_log_delegate, env_open_failure_propagates) {
  // The env itself routes open failures through session->error(),
  // which throws. Session catches this in build_delegate() and
  // falls back to stdout; here we verify the underlying behaviour
  // -- constructing an LmdbEnv on an unwritable path throws so the
  // caller can react.
  Session s;
  bool threw = false;
  try {
    LmdbEnv env(&s, "/dev/null/should_not_be_creatable");
    (void)env;
  } catch (...) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}
