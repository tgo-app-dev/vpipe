#include "minitest.h"
#include "common/credential-store.h"
#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/session.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

using namespace std;
using namespace vpipe;

namespace {

string
make_tempdir()
{
  auto base = filesystem::temp_directory_path()
              / "vpipe_credstore_test_XXXXXX";
  string tmpl = base.string();
  if (!mkdtemp(tmpl.data())) {
    throw runtime_error("mkdtemp failed");
  }
  return tmpl;
}

struct TempDir {
  string path;
  TempDir() : path(make_tempdir()) {}
  ~TempDir()
  {
    error_code ec;
    filesystem::remove_all(path, ec);
  }
};

}

TEST(credential_store, round_trip_minimal)
{
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path);

  CameraRecord in;
  in.name           = "frontdoor";
  in.rtsp_uri       = "rtsp://192.168.0.42:554/Streaming";
  in.username       = "admin";
  in.password_blob  = "\x01\x02\x03sealed-bytes";
  in.uuid           = "00112233-4455-6677-8899-aabbccddeeff";
  in.device_xaddr   = "http://192.168.0.42/onvif/device_service";
  in.supports_onvif = true;

  EXPECT_TRUE(save_camera(env, "cameras", in));

  auto out = load_camera(env, "cameras", "frontdoor");
  EXPECT_TRUE(out.has_value());
  EXPECT_TRUE(out->name           == in.name);
  EXPECT_TRUE(out->rtsp_uri       == in.rtsp_uri);
  EXPECT_TRUE(out->username       == in.username);
  EXPECT_TRUE(out->password_blob  == in.password_blob);
  EXPECT_TRUE(out->uuid           == in.uuid);
  EXPECT_TRUE(out->device_xaddr   == in.device_xaddr);
  EXPECT_TRUE(out->supports_onvif == in.supports_onvif);
}

TEST(credential_store, load_missing_returns_nullopt)
{
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path);

  // Ensure the named sub-db exists (LmdbDb constructor creates it),
  // but never write the row we look up.
  {
    LmdbDb db(env, "cameras");
    (void)db;
  }
  auto out = load_camera(env, "cameras", "absent");
  EXPECT_FALSE(out.has_value());
}

TEST(credential_store, overwrite_existing_guard)
{
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path);

  CameraRecord rec;
  rec.name     = "frontdoor";
  rec.rtsp_uri = "rtsp://old";
  EXPECT_TRUE(save_camera(env, "cameras", rec, /*overwrite*/true));

  rec.rtsp_uri = "rtsp://new";
  // overwrite_existing=false must short-circuit and leave the old
  // value in place.
  EXPECT_FALSE(save_camera(env, "cameras", rec, /*overwrite*/false));
  auto reread = load_camera(env, "cameras", "frontdoor");
  EXPECT_TRUE(reread.has_value());
  EXPECT_TRUE(reread->rtsp_uri == "rtsp://old");

  // overwrite_existing=true (the default) replaces.
  EXPECT_TRUE(save_camera(env, "cameras", rec, /*overwrite*/true));
  auto reread2 = load_camera(env, "cameras", "frontdoor");
  EXPECT_TRUE(reread2->rtsp_uri == "rtsp://new");
}

TEST(credential_store, backward_compat_old_record)
{
  // Synthesize a FlexData object missing the new fields, write it
  // raw, and confirm load_camera defaults the new fields safely.
  TempDir dir;
  Session s;
  LmdbEnv env(&s, dir.path);

  auto v = FlexData::make_object();
  v.as_object().insert_or_assign(
      "rtsp_uri", FlexData::make_string("rtsp://legacy/path"));
  v.as_object().insert_or_assign(
      "username", FlexData::make_string("legacy"));
  v.as_object().insert_or_assign(
      "password_blob", FlexData::make_string("legacy-blob"));
  std::string bytes = v.to_binary();
  {
    LmdbDb  db(env, "cameras");
    LmdbTxn txn(env);
    db.put(txn, "oldcam", bytes);
    txn.commit();
  }

  auto out = load_camera(env, "cameras", "oldcam");
  EXPECT_TRUE(out.has_value());
  EXPECT_TRUE(out->name           == "oldcam");
  EXPECT_TRUE(out->rtsp_uri       == "rtsp://legacy/path");
  EXPECT_TRUE(out->username       == "legacy");
  EXPECT_TRUE(out->password_blob  == "legacy-blob");
  EXPECT_TRUE(out->uuid           == "");
  EXPECT_TRUE(out->device_xaddr   == "");
  EXPECT_FALSE(out->supports_onvif);
}
