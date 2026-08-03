#ifndef CREDENTIAL_STORE_H
#define CREDENTIAL_STORE_H

#include <optional>
#include <string>
#include <string_view>

namespace vpipe {

class LmdbEnv;

// The one LMDB sub-db the IP-camera registry lives in: onvif-discovery
// writes it, rtsp-capture reads it. Fixed system-wide rather than a
// per-stage config key, so a discovery run and a recorder can never
// disagree about where a camera record lives. The leading underscores
// keep it out of the way of the user-named sub-dbs (the per-camera
// "<name>-videos" indexes, ...).
inline constexpr std::string_view kIpCamRegistryDb =
    "__vpipe_ip_cam_registry";

// One row from the IP-camera registry. Marshaled to / from a
// FlexData object in load_camera / save_camera so the wire format
// stays in one place. `password_blob` holds the AES-256-GCM output
// from credential-cipher::seal(); decryption is the caller's
// responsibility (the store doesn't depend on host-identity /
// credential-cipher so it builds on every platform).
//
// `supports_onvif` and `uuid` / `device_xaddr` exist so a downstream
// recorder can re-run WS-Discovery if DHCP moved the camera. Old
// records written by earlier versions of the discovery stage may not
// carry these fields; `load_camera` defaults them to safe values
// (empty strings, supports_onvif=false).
struct CameraRecord {
  std::string name;
  std::string rtsp_uri;
  std::string username;
  std::string password_blob;
  std::string uuid;
  std::string device_xaddr;
  bool        supports_onvif = false;
};

// Look up a camera by name. Returns nullopt if the row is absent
// (MDB_NOTFOUND) or the value fails to deserialize as a FlexData
// object.
std::optional<CameraRecord>
load_camera(LmdbEnv& env, std::string_view name);

// Persist `rec` under key `rec.name` in the registry. Throws (via the
// session's error path) on LMDB failure. `overwrite_existing=false`
// short-circuits with `false` if a row with the same key already
// exists; returns `true` after a successful write.
bool
save_camera(LmdbEnv&             env,
            const CameraRecord&  rec,
            bool                 overwrite_existing = true);

}

#endif
