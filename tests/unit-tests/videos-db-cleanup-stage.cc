#include "minitest.h"
#include "common/flex-data.h"
#include "common/lmdb-cursor.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/session.h"
#include "pipeline/stage-registry.h"
#include "stages/videos-db-cleanup-stage.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

string
make_tempdir_()
{
  auto base = filesystem::temp_directory_path()
              / "vpipe_videos_cleanup_test_XXXXXX";
  string tmpl = base.string();
  if (!mkdtemp(tmpl.data())) {
    throw runtime_error("mkdtemp failed");
  }
  return tmpl;
}

struct TempDir {
  string path;
  TempDir() : path(make_tempdir_()) {}
  ~TempDir()
  {
    error_code ec;
    filesystem::remove_all(path, ec);
  }
};

// 8-byte big-endian microseconds-since-epoch -- matches the key
// format rtsp-capture writes into `<camera>-videos`. We can build
// keys for arbitrary time points (past or future) to seed a sweep
// scenario.
string
be64_us_key_(chrono::system_clock::time_point t)
{
  uint64_t us = static_cast<uint64_t>(
      chrono::duration_cast<chrono::microseconds>(
          t.time_since_epoch()).count());
  string out;
  out.resize(8);
  for (int i = 0; i < 8; ++i) {
    out[7 - i] = static_cast<char>(us & 0xff);
    us >>= 8;
  }
  return out;
}

size_t
count_records_(LmdbEnv& env, string_view db_name)
{
  LmdbDb     db (env, db_name);
  LmdbTxn    txn(env, LmdbTxn::Mode::ReadOnly);
  LmdbCursor cur(txn, db);
  size_t n = 0;
  string_view k, v;
  if (cur.first(k, v)) {
    do { ++n; } while (cur.next(k, v));
  }
  return n;
}

FlexData
cfg_(string_view camera,
     uint64_t    retention_seconds   = 0,
     uint64_t    sweep_interval_secs = 0,
     string_view db_suffix           = "")
{
  FlexData c = FlexData::make_object();
  c.as_object().insert_or_assign(
      "camera_name", FlexData::make_string(camera));
  if (retention_seconds) {
    c.as_object().insert_or_assign(
        "retention_seconds", FlexData::make_uint(retention_seconds));
  }
  if (sweep_interval_secs) {
    c.as_object().insert_or_assign(
        "sweep_interval_seconds",
        FlexData::make_uint(sweep_interval_secs));
  }
  if (!db_suffix.empty()) {
    c.as_object().insert_or_assign(
        "videos_db_suffix", FlexData::make_string(db_suffix));
  }
  return c;
}

}

TEST(videos_db_cleanup_stage, type_is_registered)
{
  EXPECT_TRUE(StageRegistry::get().find_id("videos-db-cleanup") !=
              StageTypeId::unknown);
}

// Construction succeeds for any config; invalid config is recorded in
// config_error() and deferred to launch.
TEST(videos_db_cleanup_stage, missing_camera_name_deferred)
{
  Session sess;
  VideosDbCleanupStage s(&sess, "x", {}, FlexData::make_object());
  EXPECT_FALSE(s.config_error().empty());
}

TEST(videos_db_cleanup_stage, zero_retention_deferred)
{
  Session sess;
  FlexData c = FlexData::make_object();
  c.as_object().insert_or_assign(
      "camera_name", FlexData::make_string("cam"));
  c.as_object().insert_or_assign(
      "retention_seconds", FlexData::make_uint(0));
  VideosDbCleanupStage s(&sess, "x", {}, std::move(c));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(videos_db_cleanup_stage, defaults_set_when_not_overridden)
{
  Session sess;
  VideosDbCleanupStage s(&sess, "x", {}, cfg_("cam"));
  EXPECT_TRUE(s.camera_name()      == "cam");
  EXPECT_TRUE(s.videos_db_suffix() == "-videos");
  EXPECT_TRUE(s.retention()        == chrono::seconds(86400));
  EXPECT_TRUE(s.sweep_interval()   == chrono::seconds(3600));
}

TEST(videos_db_cleanup_stage, config_overrides_defaults)
{
  Session sess;
  VideosDbCleanupStage s(&sess, "x", {},
                         cfg_("cam2", /*retention*/ 30,
                              /*interval*/ 5, /*suffix*/ "-clips"));
  EXPECT_TRUE(s.camera_name()      == "cam2");
  EXPECT_TRUE(s.videos_db_suffix() == "-clips");
  EXPECT_TRUE(s.retention()        == chrono::seconds(30));
  EXPECT_TRUE(s.sweep_interval()   == chrono::seconds(5));
}

TEST(videos_db_cleanup_stage, sweep_deletes_only_expired_entries)
{
  // Seed `cam-videos` with four entries: 2 days old, 26h old, 12h
  // old, "now". With retention=86400 (1 day) the first two should
  // be deleted and the last two retained. Keys are 8-byte BE
  // microseconds-since-epoch so cursor order is time order.
  TempDir tdir;
  const string sess_cfg =
      R"({"db":{"path":")" + tdir.path + R"("}})";
  Session sess(sess_cfg);
  LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);

  const string db_name = "cam-videos";
  const auto now       = chrono::system_clock::now();
  const auto k_2d      = be64_us_key_(now - chrono::hours(48));
  const auto k_26h     = be64_us_key_(now - chrono::hours(26));
  const auto k_12h     = be64_us_key_(now - chrono::hours(12));
  const auto k_now     = be64_us_key_(now);

  {
    LmdbDb  db (*env, db_name);
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadWrite);
    db.put(txn, k_2d,  "v2d");
    db.put(txn, k_26h, "v26h");
    db.put(txn, k_12h, "v12h");
    db.put(txn, k_now, "vnow");
    txn.commit();
  }
  EXPECT_TRUE(count_records_(*env, db_name) == 4u);

  VideosDbCleanupStage s(&sess, "cleanup", {},
                         cfg_("cam", /*retention*/ 86400));
  const size_t deleted = s.sweep_once();
  EXPECT_TRUE(deleted == 2u);

  // Only the two younger entries remain, in time order.
  LmdbDb     db (*env, db_name);
  LmdbTxn    txn(*env, LmdbTxn::Mode::ReadOnly);
  LmdbCursor cur(txn, db);
  string_view k, v;
  vector<string> kept_keys;
  if (cur.first(k, v)) {
    do { kept_keys.emplace_back(k); } while (cur.next(k, v));
  }
  EXPECT_TRUE(kept_keys.size() == 2u);
  EXPECT_TRUE(kept_keys[0] == k_12h);
  EXPECT_TRUE(kept_keys[1] == k_now);
}

TEST(videos_db_cleanup_stage, sweep_on_empty_db_is_noop)
{
  // The sub-db gets created on first LmdbDb open inside sweep_once
  // but contains no rows; the sweep must succeed and return 0.
  TempDir tdir;
  const string sess_cfg =
      R"({"db":{"path":")" + tdir.path + R"("}})";
  Session sess(sess_cfg);
  ASSERT_TRUE(sess.lmdb_env() != nullptr);

  VideosDbCleanupStage s(&sess, "cleanup", {},
                         cfg_("nobody", /*retention*/ 60));
  EXPECT_TRUE(s.sweep_once() == 0u);
}

TEST(videos_db_cleanup_stage, sweep_respects_videos_db_suffix)
{
  // A non-default suffix routes the sweep to a different sub-db;
  // entries in the default `<camera>-videos` sub-db must NOT be
  // touched. We seed both to prove isolation.
  TempDir tdir;
  const string sess_cfg =
      R"({"db":{"path":")" + tdir.path + R"("}})";
  Session sess(sess_cfg);
  LmdbEnv* env = sess.lmdb_env();
  ASSERT_TRUE(env != nullptr);

  const auto now = chrono::system_clock::now();
  const auto k_old = be64_us_key_(now - chrono::hours(48));
  {
    LmdbDb  default_db(*env, "cam-videos");
    LmdbDb  custom_db (*env, "cam-clips");
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadWrite);
    default_db.put(txn, k_old, "default");
    custom_db .put(txn, k_old, "custom");
    txn.commit();
  }

  VideosDbCleanupStage s(&sess, "cleanup", {},
                         cfg_("cam", /*retention*/ 60,
                              /*interval*/ 0, /*suffix*/ "-clips"));
  EXPECT_TRUE(s.sweep_once() == 1u);
  EXPECT_TRUE(count_records_(*env, "cam-clips")  == 0u);
  EXPECT_TRUE(count_records_(*env, "cam-videos") == 1u);
}
