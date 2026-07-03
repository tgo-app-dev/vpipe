#include "stages/onvif-discovery-stage.h"
#include "common/credential-cipher.h"
#include "common/credential-store.h"
#include "common/flex-data.h"
#include "common/host-identity.h"
#include "common/job.h"
#include "common/lmdb-env.h"
#include "common/mini-http.h"
#include "common/onvif-soap.h"
#include "common/vpipe-format.h"
#include "common/ws-discovery.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

// Parse "http://host[:port]/path" into (host_port, path).
bool
split_url(const std::string& url,
          std::string&       host_port_out,
          std::string&       path_out)
{
  constexpr const char kP[] = "http://";
  if (url.compare(0, 7, kP) != 0) {
    return false;
  }
  size_t slash = url.find('/', 7);
  if (slash == std::string::npos) {
    host_port_out = url.substr(7);
    path_out      = "/";
  } else {
    host_port_out = url.substr(7, slash - 7);
    path_out      = url.substr(slash);
  }
  return true;
}

// Trim ASCII whitespace at both ends in-place.
void
trim(std::string& s)
{
  auto not_space = [](unsigned char c) {
    return !std::isspace(c);
  };
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(),
          s.end());
}

// Allowed in a camera name. The name is used both as an LMDB key
// and as the prefix of a sibling sub-db ("<name>-videos") and as
// part of an mp4 file name, so we restrict to a portable set.
bool
valid_camera_name(std::string_view s)
{
  if (s.empty()) {
    return false;
  }
  for (char c : s) {
    bool ok =
        (c >= 'A' && c <= 'Z')
        || (c >= 'a' && c <= 'z')
        || (c >= '0' && c <= '9')
        || c == '.' || c == '_' || c == '-';
    if (!ok) {
      return false;
    }
  }
  return true;
}

}

OnvifDiscoveryStage::OnvifDiscoveryStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  config)
  : TypedStage<OnvifDiscoveryStage>(s, std::move(id), std::move(iports),
                                    std::move(config))
{
  // Config attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _probe_timeout_ms =
      static_cast<unsigned>(attr_uint("probe_timeout_ms"));
  _db_name            = attr_str("db_name");
  _overwrite_existing = attr_bool("overwrite_existing");
  _mask_password      = attr_bool("mask_password");

  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "probe_timeout_ms", .type = ConfigType::Uint,
   .doc = "WS-Discovery probe timeout in ms", .def_uint = 3000},
  {.key = "db_name", .type = ConfigType::String,
   .doc = "LMDB sub-db storing camera records", .def_str = "cameras"},
  {.key = "overwrite_existing", .type = ConfigType::Bool,
   .doc = "replace an already-registered camera", .def_bool = true},
  {.key = "mask_password", .type = ConfigType::Bool,
   .doc = "mask the password while typing (getpasswd: stdin echo off / "
          "masked web-ui field); false reads it visibly", .def_bool = true},
};
const StageSpec kSpec = {
  .type_name = "onvif-discovery",
  .doc       = "Interactive one-shot: WS-Discovery probe for ONVIF "
               "cameras, prompts for selection + credentials, then "
               "persists an encrypted camera record to LMDB. 0 in / 0 "
               "out (Apple-only).",
  .display_name = "ONVIF Discovery",
  .category  = StageCategory::Preparation,
  .iports    = {},
  .oports    = {},
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
OnvifDiscoveryStage::spec() const noexcept
{
  return kSpec;
}

Job
OnvifDiscoveryStage::process(RuntimeContext& ctx)
{
  using namespace std::chrono;

  // All interactive reads route through the session UI delegate, which
  // owns prompt display + cancellation (stdin by default; diverted to
  // the browser under the web-ui delegate). Non-Ok (Eof / Canceled)
  // ends the flow via session()->error, which throws.
  auto cancel = [&ctx] { return ctx.stop_requested(); };

  // -------- 1. Discover --------
  session()->info(fmt(
      "Discovering ONVIF cameras on the local network ({:.1f}s probe)...",
      _probe_timeout_ms / 1000.0));
  auto cams = wsd::discover(milliseconds(_probe_timeout_ms),
                            session());
  if (cams.empty()) {
    // Multicast can fail silently on Wi-Fi APs that drop IGMP, on
    // hosts with the wrong default route bound, and across docker
    // bridges. Fall back to a /24 unicast sweep: parallel TCP probe
    // to :80 prunes dead IPs in one select() window, then we issue a
    // no-auth GetSystemDateAndTime to each survivor.
    session()->info(fmt(
        "No multicast responses. Sweeping local /24 via unicast "
        "WS-Discovery ..."));
    cams = wsd::sweep_subnet_24(milliseconds(_probe_timeout_ms),
                                session());
  }
  if (cams.empty()) {
    session()->info(fmt("No ONVIF devices responded."));
    ctx.signal_done();
    co_return;
  }

  session()->info(fmt("Found {} device(s):", cams.size()));
  for (size_t i = 0; i < cams.size(); ++i) {
    const auto& c = cams[i];
    std::string types_joined;
    for (size_t j = 0; j < c.types.size(); ++j) {
      if (j) { types_joined += ' '; }
      types_joined += c.types[j];
    }
    session()->info(fmt(
        "  [{}]  {}  urn:uuid:{}{}{}", i,
        c.xaddr.empty() ? "?" : c.xaddr, c.uuid,
        types_joined.empty() ? "" : "  (",
        types_joined.empty() ? "" : (types_joined + ")")));
  }

  // -------- 2. Select + credentials --------
  std::string sel_line;
  if (session()->getline(
          fmt("Select camera [0-{}]: ", cams.size() - 1),
          sel_line, cancel) != UiInputStatus::Ok) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): input closed during selection",
        this->id()));
  }
  size_t sel = 0;
  try {
    sel = static_cast<size_t>(std::stoul(sel_line));
  } catch (...) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): invalid selection '{}'",
        this->id(), sel_line));
  }
  if (sel >= cams.size()) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): selection {} out of range",
        this->id(), sel));
  }
  const auto& cam = cams[sel];
  if (cam.xaddr.empty()) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): camera has no http XAddr",
        this->id()));
  }

  std::string user;
  if (session()->getline(fmt("Username: "), user, cancel)
      != UiInputStatus::Ok) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): input closed during username",
        this->id()));
  }

  std::string camera_name;
  for (;;) {
    if (session()->getline(
            fmt("Camera name (used as DB key, [A-Za-z0-9._-]): "),
            camera_name, cancel) != UiInputStatus::Ok) {
      session()->error(fmt(
          "OnvifDiscoveryStage('{}'): input closed during "
          "camera name", this->id()));
    }
    trim(camera_name);
    if (valid_camera_name(camera_name)) {
      break;
    }
    session()->info(fmt(
        "Invalid name. Use only letters, digits, '.', '_' or '-'."));
  }

  std::string pw;
  {
    // getpasswd masks the typed password (terminal echo off on the
    // stdio delegate; a masked input field in the web UI). When
    // mask_password is disabled, fall back to a visible getline.
    const UiInputStatus st = _mask_password
        ? session()->getpasswd(fmt("Password: "), pw, cancel)
        : session()->getline(fmt("Password: "), pw, cancel);
    if (st != UiInputStatus::Ok) {
      session()->error(fmt(
          "OnvifDiscoveryStage('{}'): input closed during password",
          this->id()));
    }
  }

  // -------- 3. Authenticate + GetStreamUri --------
  onvif::WsseAuth auth{user, pw};

  std::string dev_hostport, dev_path;
  if (!split_url(cam.xaddr, dev_hostport, dev_path)) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): unsupported XAddr '{}'",
        this->id(), cam.xaddr));
  }

  session()->info(fmt("Authenticating against {} ...", cam.xaddr));

  auto soap_post = [&](const std::string& url,
                       const std::string& body_inner,
                       const std::string& action_uri)
      -> http::Response
  {
    std::string hp, pa;
    split_url(url, hp, pa);
    std::string env = onvif::make_envelope(action_uri, body_inner,
                                           &auth);
    return http::post(hp, pa,
                      "application/soap+xml; charset=utf-8",
                      env, milliseconds(5000));
  };

  auto cap_resp = soap_post(
      cam.xaddr, onvif::body_get_capabilities(),
      "http://www.onvif.org/ver10/device/wsdl/GetCapabilities");
  if (cap_resp.status == 401
      || onvif::is_auth_fault(cap_resp.body))
  {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): credentials rejected (status={})",
        this->id(), cap_resp.status));
  }
  if (cap_resp.status != 200) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): GetCapabilities returned {}",
        this->id(), cap_resp.status));
  }
  auto media_xaddr = onvif::parse_media_xaddr(cap_resp.body);
  if (!media_xaddr) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): no Media XAddr in capabilities",
        this->id()));
  }

  auto prof_resp = soap_post(
      *media_xaddr, onvif::body_get_profiles(),
      "http://www.onvif.org/ver10/media/wsdl/GetProfiles");
  if (prof_resp.status != 200) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): GetProfiles returned {}",
        this->id(), prof_resp.status));
  }
  auto profs = onvif::parse_profiles(prof_resp.body);
  if (!profs || profs->empty()) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): no profiles available",
        this->id()));
  }
  const auto& chosen = (*profs)[0];
  session()->info(fmt("OK. Profile '{}' (token={})",
                      chosen.name, chosen.token));

  auto uri_resp = soap_post(
      *media_xaddr, onvif::body_get_stream_uri(chosen.token),
      "http://www.onvif.org/ver10/media/wsdl/GetStreamUri");
  if (uri_resp.status != 200) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): GetStreamUri returned {}",
        this->id(), uri_resp.status));
  }
  auto rtsp = onvif::parse_stream_uri(uri_resp.body);
  if (!rtsp) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): could not parse stream URI",
        this->id()));
  }
  session()->info(fmt("RTSP URI: {}", *rtsp));

  // -------- 4. Encrypt password + persist --------
  auto uuid = host_uuid_bytes();
  auto kdf  = build_kdf_input(std::span<const unsigned char, 16>(uuid),
                              os_user_id());
  auto key  = derive_key(kdf);
  std::string sealed = seal(std::span<const unsigned char, 32>(key),
                             pw, {});
  // Zero the derived key + KDF input so they don't linger on the
  // worker thread's stack frame copy.
  std::memset(key.data(), 0, key.size());
  std::memset(kdf.data(),  0, kdf.size());

  if (sealed.empty()) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): password encryption failed",
        this->id()));
  }

  LmdbEnv* env = session()->lmdb_env();
  if (!env) {
    session()->error(fmt(
        "OnvifDiscoveryStage('{}'): session lmdb_env() unavailable",
        this->id()));
  }

  CameraRecord rec;
  rec.name           = camera_name;
  rec.rtsp_uri       = *rtsp;
  rec.username       = user;
  rec.password_blob  = sealed;
  rec.uuid           = cam.uuid;
  rec.device_xaddr   = cam.xaddr;
  rec.supports_onvif = true;

  bool wrote = save_camera(*env, _db_name, rec, _overwrite_existing);
  if (!wrote) {
    session()->warn(fmt(
        "OnvifDiscoveryStage('{}'): camera '{}' already registered; "
        "set overwrite_existing=true to replace",
        this->id(), camera_name));
    ctx.signal_done();
    co_return;
  }

  session()->info(fmt(
      "OnvifDiscoveryStage('{}'): saved camera '{}' "
      "(uuid={}, rtsp={})",
      this->id(), camera_name, cam.uuid, *rtsp));

  ctx.signal_done();
  co_return;
}

VPIPE_REGISTER_STAGE(OnvifDiscoveryStage)
VPIPE_REGISTER_SPEC(OnvifDiscoveryStage, kSpec)

}
