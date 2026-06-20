#include "common/onvif-refresh.h"
#include "common/onvif-soap.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <chrono>
#include <optional>
#include <string>
#include <utility>

namespace vpipe {

namespace {

// Same shape as the discovery stage's split_url -- kept local so
// onvif-refresh has no dependency on the stage TU.
bool
split_http_url(std::string_view url,
               std::string&     host_port_out,
               std::string&     path_out)
{
  constexpr std::string_view kP = "http://";
  if (url.size() < kP.size() || url.substr(0, kP.size()) != kP) {
    return false;
  }
  auto rest = url.substr(kP.size());
  auto slash = rest.find('/');
  if (slash == std::string_view::npos) {
    host_port_out.assign(rest);
    path_out = "/";
  } else {
    host_port_out.assign(rest.substr(0, slash));
    path_out.assign(rest.substr(slash));
  }
  return true;
}

http::Response
do_post(const RefreshDeps&        deps,
        std::string_view          host_port,
        std::string_view          path,
        std::string_view          content_type,
        std::string_view          body,
        std::chrono::milliseconds timeout)
{
  if (deps.post) {
    return deps.post(host_port, path, content_type, body, timeout);
  }
  return http::post(host_port, path, content_type, body, timeout);
}

std::vector<wsd::DiscoveredCamera>
do_discover(const RefreshDeps&         deps,
            std::chrono::milliseconds  window,
            const SessionContextIntf*  log)
{
  if (deps.discover) {
    return deps.discover(window, log);
  }
  return wsd::discover(window, log);
}

// Try to fetch the RTSP URI from a known device XAddr. Returns the
// fresh URI on success, nullopt otherwise.
std::optional<std::string>
fetch_rtsp_uri(std::string_view          device_xaddr,
               std::string_view          user,
               std::string_view          password,
               std::chrono::milliseconds timeout,
               const RefreshDeps&        deps,
               const SessionContextIntf* session)
{
  std::string dev_hp, dev_pa;
  if (!split_http_url(device_xaddr, dev_hp, dev_pa)) {
    return std::nullopt;
  }
  onvif::WsseAuth auth{std::string(user), std::string(password)};

  auto soap_call = [&](std::string_view url,
                       std::string_view body_inner,
                       std::string_view action_uri)
      -> http::Response
  {
    std::string hp, pa;
    if (!split_http_url(url, hp, pa)) {
      return http::Response{};
    }
    std::string env = onvif::make_envelope(action_uri, body_inner,
                                           &auth);
    return do_post(deps, hp, pa,
                   "application/soap+xml; charset=utf-8",
                   env, timeout);
  };

  auto cap_resp = soap_call(
      device_xaddr, onvif::body_get_capabilities(),
      "http://www.onvif.org/ver10/device/wsdl/GetCapabilities");
  if (cap_resp.status != 200) {
    if (session) {
      session->log_debug(fmt(
          "onvif-refresh: GetCapabilities at '{}' returned {}",
          device_xaddr, cap_resp.status));
    }
    return std::nullopt;
  }
  if (onvif::is_auth_fault(cap_resp.body)) {
    if (session) {
      session->warn(fmt(
          "onvif-refresh: credentials rejected by '{}'",
          device_xaddr));
    }
    return std::nullopt;
  }
  auto media_xaddr = onvif::parse_media_xaddr(cap_resp.body);
  if (!media_xaddr) {
    return std::nullopt;
  }

  auto prof_resp = soap_call(
      *media_xaddr, onvif::body_get_profiles(),
      "http://www.onvif.org/ver10/media/wsdl/GetProfiles");
  if (prof_resp.status != 200) {
    return std::nullopt;
  }
  auto profs = onvif::parse_profiles(prof_resp.body);
  if (!profs || profs->empty()) {
    return std::nullopt;
  }

  auto uri_resp = soap_call(
      *media_xaddr,
      onvif::body_get_stream_uri((*profs)[0].token),
      "http://www.onvif.org/ver10/media/wsdl/GetStreamUri");
  if (uri_resp.status != 200) {
    return std::nullopt;
  }
  auto rtsp = onvif::parse_stream_uri(uri_resp.body);
  if (!rtsp || rtsp->empty()) {
    return std::nullopt;
  }
  return *rtsp;
}

}

bool
refresh_rtsp_uri(CameraRecord&             rec,
                 std::string_view          plaintext_password,
                 std::chrono::milliseconds timeout,
                 const SessionContextIntf* session,
                 const RefreshDeps&        deps)
{
  // -- Step 1: probe the cached XAddr first. --
  if (!rec.device_xaddr.empty()) {
    auto fresh = fetch_rtsp_uri(rec.device_xaddr, rec.username,
                                plaintext_password, timeout,
                                deps, session);
    if (fresh) {
      bool changed = (*fresh != rec.rtsp_uri);
      rec.rtsp_uri = *fresh;
      if (session) {
        session->info(fmt(
            "onvif-refresh: '{}' reached at cached XAddr '{}' "
            "(rtsp_uri {})",
            rec.name, rec.device_xaddr,
            changed ? "updated" : "unchanged"));
      }
      return true;
    }
  }

  // -- Step 2: rerun WS-Discovery and match by UUID. --
  if (rec.uuid.empty()) {
    if (session) {
      session->warn(fmt(
          "onvif-refresh: '{}' has no stored UUID; cannot "
          "rediscover after IP change", rec.name));
    }
    return false;
  }
  auto cams = do_discover(deps, timeout, session);
  const wsd::DiscoveredCamera* match = nullptr;
  for (const auto& c : cams) {
    if (c.uuid == rec.uuid && !c.xaddr.empty()) {
      match = &c;
      break;
    }
  }
  if (!match) {
    if (session) {
      session->warn(fmt(
          "onvif-refresh: '{}' (uuid={}) not found by WS-Discovery",
          rec.name, rec.uuid));
    }
    return false;
  }

  auto fresh = fetch_rtsp_uri(match->xaddr, rec.username,
                              plaintext_password, timeout,
                              deps, session);
  if (!fresh) {
    if (session) {
      session->warn(fmt(
          "onvif-refresh: '{}' found at '{}' but SOAP refresh "
          "failed", rec.name, match->xaddr));
    }
    return false;
  }
  std::string old_xaddr = rec.device_xaddr;
  rec.device_xaddr = match->xaddr;
  rec.rtsp_uri     = *fresh;
  if (session) {
    session->info(fmt(
        "onvif-refresh: '{}' moved from '{}' to '{}' "
        "(rtsp_uri now '{}')",
        rec.name, old_xaddr, match->xaddr, *fresh));
  }
  return true;
}

}
