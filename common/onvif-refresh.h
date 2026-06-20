#ifndef ONVIF_REFRESH_H
#define ONVIF_REFRESH_H

#include "common/credential-store.h"
#include "common/mini-http.h"
#include "common/ws-discovery.h"

#include <chrono>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

class SessionContextIntf;

// Injectable dependencies for refresh_rtsp_uri. Production callers
// leave them default-constructed; tests substitute stubs for the
// HTTP and WS-Discovery calls.
struct RefreshDeps {
  using PostFn =
      std::function<http::Response(
          std::string_view          host_port,
          std::string_view          path,
          std::string_view          content_type,
          std::string_view          body,
          std::chrono::milliseconds timeout)>;
  using DiscoverFn =
      std::function<std::vector<wsd::DiscoveredCamera>(
          std::chrono::milliseconds  window,
          const SessionContextIntf*  log_sink)>;

  PostFn     post     = nullptr;
  DiscoverFn discover = nullptr;
};

// Update `rec.rtsp_uri` and `rec.device_xaddr` after the camera's IP
// may have moved (DHCP renewal, manual config). Caller supplies the
// plaintext ONVIF password -- the cipher dependency stays on the
// caller, which keeps this helper testable without CommonCrypto.
//
// Two-step fallback:
//   1. Try the cached `rec.device_xaddr` with an authenticated
//      GetCapabilities/GetProfiles/GetStreamUri. If the camera is
//      still on its old IP, this is a tight loop with one
//      round-trip.
//   2. If step 1 fails (connection refused, timeout, auth fault),
//      run WS-Discovery and find the entry whose UUID matches
//      `rec.uuid`. Retry the SOAP flow against the new XAddr.
//
// Returns true iff `rec.rtsp_uri` was updated and is plausibly
// valid. On false, `rec` is left untouched (so the caller can
// preserve its prior value, log, and retry).
bool
refresh_rtsp_uri(CameraRecord&             rec,
                 std::string_view          plaintext_password,
                 std::chrono::milliseconds timeout,
                 const SessionContextIntf* session,
                 const RefreshDeps&        deps = {});

}

#endif
