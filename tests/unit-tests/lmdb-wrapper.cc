#include "minitest.h"
#include "common/lmdb-cursor.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/session.h"
#include "common/vpipe-format.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

string make_tempdir() {
  auto base = filesystem::temp_directory_path() / "vpipe_lmdbw_test_XXXXXX";
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

// Encode a uint64 as a big-endian 8-byte string so lexicographic order
// matches numeric order. Used by cursor / threading tests as a stand-in
// for the timestamp-keyed log/clip use cases.
string be64(uint64_t v) {
  string s(8, '\0');
  for (int i = 7; i >= 0; --i) {
    s[i] = static_cast<char>(v & 0xff);
    v >>= 8;
  }
  return s;
}

} // namespace

TEST(lmdb_wrapper, env_open_succeeds) {
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path);
  EXPECT_TRUE(env.valid());
  EXPECT_TRUE(env.path() == dir.path);
  EXPECT_TRUE(env.raw() != nullptr);
  EXPECT_TRUE(static_cast<bool>(env));
}

TEST(lmdb_wrapper, env_open_bad_path_throws) {
  Session s;
  bool threw = false;
  try {
    // /dev/null is a file; create_directories beneath it must fail.
    LmdbEnv env(&s, "/dev/null/vpipe_should_fail");
    (void)env;
  } catch (exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

TEST(lmdb_wrapper, txn_commit_persists) {
  TempDir dir;
  {
    Session s;
    LmdbEnv env(&s, dir.path);
    LmdbDb  db(env, "main");
    LmdbTxn txn(env);
    db.put(txn, "k", "v");
    txn.commit();
  }
  // Reopen in a fresh scope.
  Session s;
  LmdbEnv env(&s, dir.path);
  LmdbDb  db(env, "main");
  LmdbTxn txn(env, LmdbTxn::Mode::ReadOnly);
  auto v = db.get(txn, "k");
  EXPECT_TRUE(v.has_value());
  EXPECT_TRUE(*v == "v");
}

TEST(lmdb_wrapper, txn_default_aborts) {
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path);
  LmdbDb  db(env, "main");
  {
    LmdbTxn txn(env);
    db.put(txn, "ephemeral", "should_vanish");
    EXPECT_TRUE(txn.active());
    // No commit -> dtor aborts.
  }
  LmdbTxn rtxn(env, LmdbTxn::Mode::ReadOnly);
  auto v = db.get(rtxn, "ephemeral");
  EXPECT_FALSE(v.has_value());
}

TEST(lmdb_wrapper, named_dbs_isolated) {
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path);
  LmdbDb  logs(env,  "logs");
  LmdbDb  clips(env, "clips");
  {
    LmdbTxn txn(env);
    logs .put(txn, "id", "log_value");
    clips.put(txn, "id", "clip_value");
    txn.commit();
  }
  LmdbTxn rtxn(env, LmdbTxn::Mode::ReadOnly);
  auto lv = logs .get(rtxn, "id");
  auto cv = clips.get(rtxn, "id");
  EXPECT_TRUE(lv.has_value() && *lv == "log_value");
  EXPECT_TRUE(cv.has_value() && *cv == "clip_value");
}

TEST(lmdb_wrapper, del_returns_false_for_missing) {
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path);
  LmdbDb  db(env, "main");
  LmdbTxn txn(env);
  db.put(txn, "present", "x");
  EXPECT_TRUE (db.del(txn, "present"));
  EXPECT_FALSE(db.del(txn, "absent"));
  txn.commit();
}

TEST(lmdb_wrapper, cursor_range_scan) {
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path);
  LmdbDb  db(env, "ts");
  {
    LmdbTxn txn(env);
    for (uint64_t t = 100; t < 200; t += 10) {
      db.put(txn, be64(t), "v");
    }
    txn.commit();
  }
  LmdbTxn rtxn(env, LmdbTxn::Mode::ReadOnly);
  LmdbCursor cur(rtxn, db);

  // Walk window [130, 170].
  string_view k, v;
  vector<uint64_t> seen;
  bool hit = cur.seek_at_or_after(be64(130), k, v);
  while (hit) {
    if (k.size() != 8) {
      break;
    }
    uint64_t t = 0;
    for (size_t i = 0; i < 8; ++i) {
      t = (t << 8) | static_cast<unsigned char>(k[i]);
    }
    if (t > 170) {
      break;
    }
    seen.push_back(t);
    hit = cur.next(k, v);
  }
  EXPECT_TRUE(seen.size() == 5);
  EXPECT_TRUE(seen.front() == 130);
  EXPECT_TRUE(seen.back()  == 170);
}

TEST(lmdb_wrapper, move_transfers_ownership) {
  TempDir dir;
  Session s;

  LmdbEnv a(&s, dir.path);
  EXPECT_TRUE(a.valid());
  LmdbEnv b(std::move(a));
  EXPECT_TRUE (b.valid());
  EXPECT_FALSE(a.valid());

  LmdbDb  db(b, "main");
  LmdbTxn t1(b);
  EXPECT_TRUE(t1.active());
  LmdbTxn t2(std::move(t1));
  EXPECT_TRUE (t2.active());
  EXPECT_FALSE(t1.active());
  db.put(t2, "k", "v");
  t2.commit();

  LmdbTxn rtxn(b, LmdbTxn::Mode::ReadOnly);
  LmdbCursor c1(rtxn, db);
  EXPECT_TRUE(c1.raw() != nullptr);
  LmdbCursor c2(std::move(c1));
  EXPECT_TRUE (c2.raw() != nullptr);
  EXPECT_TRUE(c1.raw() == nullptr);
}

TEST(lmdb_wrapper, concurrent_readers) {
  // Validates that MDB_NOTLS is in effect: many threads each hold their own
  // read txn against a shared LmdbEnv. Without NOTLS this would either
  // serialize or fail because each thread would clobber the same TLS slot.
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path, size_t{1} << 24, /*max_dbs=*/4,
              /*max_readers=*/64);
  LmdbDb  db(env, "shared");
  constexpr int kKeys = 128;
  {
    LmdbTxn txn(env);
    for (int i = 0; i < kKeys; ++i) {
      db.put(txn, "k_" + to_string(i), to_string(i));
    }
    txn.commit();
  }

  constexpr int    kThreads = 16;
  atomic<int>      ok{0};
  atomic<bool>     fail{false};
  vector<thread>   workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&]() {
      try {
        LmdbTxn rtxn(env, LmdbTxn::Mode::ReadOnly);
        for (int i = 0; i < kKeys; ++i) {
          auto v = db.get(rtxn, "k_" + to_string(i));
          if (!v.has_value() || *v != to_string(i)) {
            fail = true;
            return;
          }
        }
        ++ok;
      } catch (...) {
        fail = true;
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  EXPECT_FALSE(fail.load());
  EXPECT_TRUE(ok.load() == kThreads);
}

TEST(lmdb_wrapper, concurrent_writer_serializes) {
  // LMDB serializes writers per env; the wrapper should not deadlock and
  // should not lose updates when many threads contend.
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path, size_t{1} << 24);
  LmdbDb  db(env, "writes");

  constexpr int    kThreads      = 8;
  constexpr int    kPerThread    = 64;
  atomic<bool>     fail{false};
  vector<thread>   workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&, t]() {
      try {
        for (int i = 0; i < kPerThread; ++i) {
          LmdbTxn txn(env);
          string key = "t" + to_string(t) + "_" + to_string(i);
          db.put(txn, key, "v");
          txn.commit();
        }
      } catch (...) {
        fail = true;
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  EXPECT_FALSE(fail.load());

  // Verify every (t, i) landed.
  LmdbTxn rtxn(env, LmdbTxn::Mode::ReadOnly);
  int     found = 0;
  for (int t = 0; t < kThreads; ++t) {
    for (int i = 0; i < kPerThread; ++i) {
      string key = "t" + to_string(t) + "_" + to_string(i);
      auto v = db.get(rtxn, key);
      if (v.has_value() && *v == "v") {
        ++found;
      }
    }
  }
  EXPECT_TRUE(found == kThreads * kPerThread);
}

TEST(lmdb_wrapper, reader_during_writer) {
  // MVCC: a read txn holds its snapshot across a concurrent commit; only a
  // fresh read txn sees the new data.
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path);
  LmdbDb  db(env, "mvcc");
  {
    LmdbTxn txn(env);
    db.put(txn, "x", "v0");
    txn.commit();
  }

  LmdbTxn rtxn(env, LmdbTxn::Mode::ReadOnly);
  // Snapshot view BEFORE the writer commits.
  auto pre = db.get(rtxn, "x");
  EXPECT_TRUE(pre.has_value() && *pre == "v0");

  atomic<bool> writer_done{false};
  thread writer([&]() {
    LmdbTxn wtxn(env);
    db.put(wtxn, "x", "v1");
    wtxn.commit();
    writer_done = true;
  });
  writer.join();
  EXPECT_TRUE(writer_done.load());

  // Same read-txn still sees v0.
  auto mid = db.get(rtxn, "x");
  EXPECT_TRUE(mid.has_value() && *mid == "v0");
  rtxn.abort();

  // Fresh read-txn sees v1.
  LmdbTxn rtxn2(env, LmdbTxn::Mode::ReadOnly);
  auto post = db.get(rtxn2, "x");
  EXPECT_TRUE(post.has_value() && *post == "v1");
}
