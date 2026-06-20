#include "minitest.h"
#include "common/credential-store.h"
#include "common/mini-http.h"
#include "common/onvif-refresh.h"
#include "common/ws-discovery.h"

#include <chrono>
#include <string>
#include <string_view>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

constexpr const char kCapsResponse[] = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tt="http://www.onvif.org/ver10/schema"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
 <s:Body>
  <tds:GetCapabilitiesResponse>
   <tds:Capabilities>
    <tt:Media>
     <tt:XAddr>http://192.168.0.99/onvif/Media</tt:XAddr>
    </tt:Media>
   </tds:Capabilities>
  </tds:GetCapabilitiesResponse>
 </s:Body>
</s:Envelope>)xml";

constexpr const char kProfilesResponse[] = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tt="http://www.onvif.org/ver10/schema"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
 <s:Body>
  <trt:GetProfilesResponse>
   <trt:Profiles token="Profile_1">
    <tt:Name>MainStream</tt:Name>
   </trt:Profiles>
  </trt:GetProfilesResponse>
 </s:Body>
</s:Envelope>)xml";

constexpr const char kStreamUriResponse[] = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tt="http://www.onvif.org/ver10/schema"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
 <s:Body>
  <trt:GetStreamUriResponse>
   <trt:MediaUri>
    <tt:Uri>rtsp://192.168.0.99:554/profile1</tt:Uri>
   </trt:MediaUri>
  </trt:GetStreamUriResponse>
 </s:Body>
</s:Envelope>)xml";

http::Response
make_ok(string_view body)
{
  http::Response r;
  r.status = 200;
  r.body   = string(body);
  return r;
}

// Returns the body that corresponds to the SOAP action embedded in
// the outgoing request. Reusable across tests.
http::Response
canned_responder(string_view, string_view,
                 string_view, string_view body,
                 chrono::milliseconds)
{
  if (body.find("GetCapabilities") != string_view::npos) {
    return make_ok(kCapsResponse);
  }
  if (body.find("GetProfiles") != string_view::npos) {
    return make_ok(kProfilesResponse);
  }
  if (body.find("GetStreamUri") != string_view::npos) {
    return make_ok(kStreamUriResponse);
  }
  http::Response r;
  r.status = 400;
  return r;
}

http::Response
always_fail(string_view, string_view,
            string_view, string_view,
            chrono::milliseconds)
{
  http::Response r;
  r.status = 0;  // simulate connection failure
  return r;
}

}

TEST(onvif_refresh, cached_xaddr_path_returns_updated_uri)
{
  RefreshDeps deps;
  deps.post = canned_responder;
  // discover should NOT be called when the cached XAddr works.
  deps.discover =
      [](chrono::milliseconds, const SessionContextIntf*) {
        return vector<wsd::DiscoveredCamera>{};
      };

  CameraRecord rec;
  rec.name         = "frontdoor";
  rec.rtsp_uri     = "rtsp://192.168.0.42:554/old";
  rec.username     = "admin";
  rec.uuid         = "uuid-abcdef";
  rec.device_xaddr = "http://192.168.0.99/onvif/device_service";
  rec.supports_onvif = true;

  bool ok = refresh_rtsp_uri(rec, "pw",
                             chrono::milliseconds(1000),
                             /*session*/ nullptr, deps);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(rec.rtsp_uri == "rtsp://192.168.0.99:554/profile1");
  // device_xaddr stays at its cached value -- discovery did not run.
  EXPECT_TRUE(rec.device_xaddr
              == "http://192.168.0.99/onvif/device_service");
}

TEST(onvif_refresh, discovery_path_updates_both_uri_and_xaddr)
{
  RefreshDeps deps;
  // The post stub synthesizes responses tied to the queried host:
  // GetCapabilities for host X returns a Media XAddr at host X, so a
  // follow-up GetProfiles ends up at the same host. The cached
  // address 192.168.0.99 is "dead": any request there fails.
  deps.post = [](string_view host_port, string_view,
                 string_view, string_view body,
                 chrono::milliseconds) -> http::Response {
    if (host_port.find("192.168.0.99") != string_view::npos) {
      http::Response r;
      r.status = 0;
      return r;
    }
    if (body.find("GetCapabilities") != string_view::npos) {
      string s =
          R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tt="http://www.onvif.org/ver10/schema"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
 <s:Body>
  <tds:GetCapabilitiesResponse>
   <tds:Capabilities>
    <tt:Media>
     <tt:XAddr>http://)xml"
          + string(host_port) + "/onvif/Media</tt:XAddr>"
          + R"xml(
    </tt:Media>
   </tds:Capabilities>
  </tds:GetCapabilitiesResponse>
 </s:Body>
</s:Envelope>)xml";
      return make_ok(s);
    }
    if (body.find("GetStreamUri") != string_view::npos) {
      string s =
          R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tt="http://www.onvif.org/ver10/schema"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
 <s:Body>
  <trt:GetStreamUriResponse>
   <trt:MediaUri>
    <tt:Uri>rtsp://)xml"
          + string(host_port) + ":554/profile1</tt:Uri>"
          + R"xml(
   </trt:MediaUri>
  </trt:GetStreamUriResponse>
 </s:Body>
</s:Envelope>)xml";
      return make_ok(s);
    }
    return make_ok(kProfilesResponse);
  };
  deps.discover =
      [](chrono::milliseconds, const SessionContextIntf*) {
        vector<wsd::DiscoveredCamera> v;
        wsd::DiscoveredCamera c;
        c.uuid  = "uuid-match";
        c.xaddr = "http://192.168.0.123/onvif/device_service";
        v.push_back(c);
        return v;
      };

  CameraRecord rec;
  rec.name         = "frontdoor";
  rec.rtsp_uri     = "rtsp://192.168.0.99:554/old";
  rec.username     = "admin";
  rec.uuid         = "uuid-match";
  rec.device_xaddr = "http://192.168.0.99/onvif/device_service";
  rec.supports_onvif = true;

  bool ok = refresh_rtsp_uri(rec, "pw",
                             chrono::milliseconds(1000),
                             /*session*/ nullptr, deps);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(rec.device_xaddr
              == "http://192.168.0.123/onvif/device_service");
  EXPECT_TRUE(rec.rtsp_uri
              == "rtsp://192.168.0.123:554/profile1");
}

TEST(onvif_refresh, no_match_in_discovery_leaves_record_alone)
{
  RefreshDeps deps;
  deps.post = always_fail;
  deps.discover =
      [](chrono::milliseconds, const SessionContextIntf*) {
        vector<wsd::DiscoveredCamera> v;
        wsd::DiscoveredCamera c;
        c.uuid  = "some-other-camera";
        c.xaddr = "http://192.168.0.7/onvif/device_service";
        v.push_back(c);
        return v;
      };

  CameraRecord rec;
  rec.name         = "frontdoor";
  rec.rtsp_uri     = "rtsp://192.168.0.99:554/old";
  rec.username     = "admin";
  rec.uuid         = "uuid-mine";
  rec.device_xaddr = "http://192.168.0.99/onvif/device_service";

  CameraRecord before = rec;
  bool ok = refresh_rtsp_uri(rec, "pw",
                             chrono::milliseconds(1000),
                             /*session*/ nullptr, deps);
  EXPECT_FALSE(ok);
  EXPECT_TRUE(rec.rtsp_uri     == before.rtsp_uri);
  EXPECT_TRUE(rec.device_xaddr == before.device_xaddr);
}

TEST(onvif_refresh, missing_uuid_blocks_discovery_fallback)
{
  RefreshDeps deps;
  deps.post = always_fail;
  bool discover_called = false;
  deps.discover =
      [&](chrono::milliseconds, const SessionContextIntf*) {
        discover_called = true;
        return vector<wsd::DiscoveredCamera>{};
      };

  CameraRecord rec;
  rec.name         = "anonymous";
  rec.rtsp_uri     = "rtsp://10.0.0.1/old";
  rec.username     = "admin";
  rec.uuid         = "";  // missing!
  rec.device_xaddr = "http://10.0.0.1/onvif/device_service";

  bool ok = refresh_rtsp_uri(rec, "pw",
                             chrono::milliseconds(500),
                             nullptr, deps);
  EXPECT_FALSE(ok);
  EXPECT_FALSE(discover_called);
}
