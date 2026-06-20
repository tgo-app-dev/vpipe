#include "common/credential-store.h"
#include "common/flex-data.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"

#include <string>

namespace vpipe {

namespace {

CameraRecord
from_flex(std::string_view name_key, const FlexData& fd)
{
  CameraRecord r;
  r.name = std::string(name_key);
  if (!fd.is_object()) {
    return r;
  }
  auto obj = fd.as_object();
  r.rtsp_uri       = std::string(
      obj.contains("rtsp_uri")
          ? obj.at("rtsp_uri").as_string("") : "");
  r.username       = std::string(
      obj.contains("username")
          ? obj.at("username").as_string("") : "");
  r.password_blob  = std::string(
      obj.contains("password_blob")
          ? obj.at("password_blob").as_string("") : "");
  r.uuid           = std::string(
      obj.contains("uuid")
          ? obj.at("uuid").as_string("") : "");
  r.device_xaddr   = std::string(
      obj.contains("device_xaddr")
          ? obj.at("device_xaddr").as_string("") : "");
  r.supports_onvif =
      obj.contains("supports_onvif")
          ? obj.at("supports_onvif").as_bool(false) : false;
  // Embedded "name" field, if present, takes precedence over the
  // db key only when the key was passed empty (e.g. a test that
  // constructs a CameraRecord from a bare FlexData without going
  // through the LMDB path).
  if (r.name.empty() && obj.contains("name")) {
    r.name = std::string(obj.at("name").as_string(""));
  }
  return r;
}

FlexData
to_flex(const CameraRecord& r)
{
  auto v = FlexData::make_object();
  v.as_object().insert_or_assign(
      "name",           FlexData::make_string(r.name));
  v.as_object().insert_or_assign(
      "rtsp_uri",       FlexData::make_string(r.rtsp_uri));
  v.as_object().insert_or_assign(
      "username",       FlexData::make_string(r.username));
  v.as_object().insert_or_assign(
      "password_blob",  FlexData::make_string(r.password_blob));
  v.as_object().insert_or_assign(
      "uuid",           FlexData::make_string(r.uuid));
  v.as_object().insert_or_assign(
      "device_xaddr",   FlexData::make_string(r.device_xaddr));
  v.as_object().insert_or_assign(
      "supports_onvif", FlexData::make_bool(r.supports_onvif));
  return v;
}

}

std::optional<CameraRecord>
load_camera(LmdbEnv&         env,
            std::string_view db_name,
            std::string_view name)
{
  LmdbDb  db(env, db_name);
  LmdbTxn txn(env, LmdbTxn::Mode::ReadOnly);
  auto    view = db.get(txn, name);
  if (!view) {
    return std::nullopt;
  }
  // Copy the bytes -- the view is only valid until txn ends.
  std::string bytes(*view);
  txn.abort();
  FlexData fd;
  try {
    fd = FlexData::from_binary(bytes);
  } catch (const FlexData::Error&) {
    return std::nullopt;
  }
  return from_flex(name, fd);
}

bool
save_camera(LmdbEnv&            env,
            std::string_view    db_name,
            const CameraRecord& rec,
            bool                overwrite_existing)
{
  LmdbDb  db(env, db_name);
  LmdbTxn txn(env, LmdbTxn::Mode::ReadWrite);

  if (!overwrite_existing) {
    auto existing = db.get(txn, rec.name);
    if (existing) {
      txn.abort();
      return false;
    }
  }
  FlexData    v     = to_flex(rec);
  std::string bytes = v.to_binary();
  db.put(txn, rec.name, bytes);
  txn.commit();
  return true;
}

}
