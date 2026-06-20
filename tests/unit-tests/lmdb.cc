#include "minitest.h"
#include "lmdb.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <unistd.h>

using namespace std;

namespace {

string make_tempdir() {
  auto base = filesystem::temp_directory_path() / "vpipe_lmdb_test_XXXXXX";
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

} // namespace

TEST(lmdb, put_get_roundtrip) {
  TempDir dir;

  MDB_env* env = nullptr;
  EXPECT_TRUE(mdb_env_create(&env) == MDB_SUCCESS);
  EXPECT_TRUE(mdb_env_set_mapsize(env, size_t{1} << 20) == MDB_SUCCESS);
  EXPECT_TRUE(mdb_env_open(env, dir.path.c_str(), 0, 0644) == MDB_SUCCESS);

  MDB_txn* txn = nullptr;
  EXPECT_TRUE(mdb_txn_begin(env, nullptr, 0, &txn) == MDB_SUCCESS);

  MDB_dbi dbi;
  EXPECT_TRUE(mdb_dbi_open(txn, nullptr, MDB_CREATE, &dbi) == MDB_SUCCESS);

  const char* key_str = "hello";
  const char* val_str = "world";
  MDB_val key{strlen(key_str), const_cast<char*>(key_str)};
  MDB_val val{strlen(val_str), const_cast<char*>(val_str)};
  EXPECT_TRUE(mdb_put(txn, dbi, &key, &val, 0) == MDB_SUCCESS);
  EXPECT_TRUE(mdb_txn_commit(txn) == MDB_SUCCESS);

  MDB_txn* rtxn = nullptr;
  EXPECT_TRUE(mdb_txn_begin(env, nullptr, MDB_RDONLY, &rtxn) == MDB_SUCCESS);
  MDB_val out{};
  EXPECT_TRUE(mdb_get(rtxn, dbi, &key, &out) == MDB_SUCCESS);
  EXPECT_TRUE(out.mv_size == strlen(val_str));
  EXPECT_TRUE(memcmp(out.mv_data, val_str, out.mv_size) == 0);
  mdb_txn_abort(rtxn);

  mdb_dbi_close(env, dbi);
  mdb_env_close(env);
}

TEST(lmdb, get_missing_returns_notfound) {
  TempDir dir;

  MDB_env* env = nullptr;
  EXPECT_TRUE(mdb_env_create(&env) == MDB_SUCCESS);
  EXPECT_TRUE(mdb_env_set_mapsize(env, size_t{1} << 20) == MDB_SUCCESS);
  EXPECT_TRUE(mdb_env_open(env, dir.path.c_str(), 0, 0644) == MDB_SUCCESS);

  MDB_txn* txn = nullptr;
  EXPECT_TRUE(mdb_txn_begin(env, nullptr, 0, &txn) == MDB_SUCCESS);
  MDB_dbi dbi;
  EXPECT_TRUE(mdb_dbi_open(txn, nullptr, MDB_CREATE, &dbi) == MDB_SUCCESS);

  const char* key_str = "absent";
  MDB_val key{strlen(key_str), const_cast<char*>(key_str)};
  MDB_val out{};
  EXPECT_TRUE(mdb_get(txn, dbi, &key, &out) == MDB_NOTFOUND);

  mdb_txn_abort(txn);
  mdb_dbi_close(env, dbi);
  mdb_env_close(env);
}
