#include "minitest.h"
#include "common/ws-discovery.h"

#include <pugixml.hpp>
#include <string>

using namespace vpipe;

TEST(ws_discovery, probe_envelope_well_formed) {
  auto env = wsd::make_probe_envelope("11112222-3333-4444-5555-666677778888");
  pugi::xml_document doc;
  auto pr = doc.load_buffer(env.data(), env.size());
  EXPECT_TRUE(static_cast<bool>(pr));
  EXPECT_TRUE(env.find("d:Probe") != std::string::npos);
  EXPECT_TRUE(env.find("NetworkVideoTransmitter") != std::string::npos);
  EXPECT_TRUE(env.find("urn:uuid:11112222-3333-4444-5555-666677778888")
              != std::string::npos);
}

static constexpr const char kMatchFixture[] = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">
 <s:Header>
  <a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/ProbeMatches</a:Action>
 </s:Header>
 <s:Body>
  <d:ProbeMatches>
   <d:ProbeMatch>
    <a:EndpointReference>
     <a:Address>urn:uuid:9aeae3e8-0001-0002-0003-deadbeef0001</a:Address>
    </a:EndpointReference>
    <d:Types>dn:NetworkVideoTransmitter tds:Device</d:Types>
    <d:Scopes>onvif://www.onvif.org/name/AXIS onvif://www.onvif.org/Profile/Streaming</d:Scopes>
    <d:XAddrs>http://192.168.1.42/onvif/device_service https://192.168.1.42/onvif/device_service</d:XAddrs>
   </d:ProbeMatch>
   <d:ProbeMatch>
    <a:EndpointReference>
     <a:Address>urn:uuid:b5e1aaaa-0000-0000-0000-feedface0002</a:Address>
    </a:EndpointReference>
    <d:Types>dn:NetworkVideoTransmitter</d:Types>
    <d:Scopes>onvif://www.onvif.org/name/Hikvision</d:Scopes>
    <d:XAddrs>http://192.168.1.43/onvif/device_service</d:XAddrs>
   </d:ProbeMatch>
  </d:ProbeMatches>
 </s:Body>
</s:Envelope>)xml";

TEST(ws_discovery, parse_two_matches) {
  auto cams = wsd::parse_probe_match(kMatchFixture);
  EXPECT_TRUE(cams.size() == 2);
}

TEST(ws_discovery, parse_strips_urn_uuid_prefix) {
  auto cams = wsd::parse_probe_match(kMatchFixture);
  EXPECT_TRUE(cams.size() == 2);
  for (const auto& c : cams) {
    EXPECT_TRUE(c.uuid.find("urn:uuid:") == std::string::npos);
  }
  bool found1 = false;
  for (const auto& c : cams) {
    if (c.uuid == "9aeae3e8-0001-0002-0003-deadbeef0001") {
      found1 = true;
    }
  }
  EXPECT_TRUE(found1);
}

TEST(ws_discovery, parse_picks_first_http_xaddr) {
  auto cams = wsd::parse_probe_match(kMatchFixture);
  for (const auto& c : cams) {
    if (c.uuid == "9aeae3e8-0001-0002-0003-deadbeef0001") {
      EXPECT_TRUE(c.xaddr == "http://192.168.1.42/onvif/device_service");
    }
  }
}

TEST(ws_discovery, parse_types_and_scopes_split) {
  auto cams = wsd::parse_probe_match(kMatchFixture);
  for (const auto& c : cams) {
    if (c.uuid == "9aeae3e8-0001-0002-0003-deadbeef0001") {
      EXPECT_TRUE(c.types.size() == 2);
      EXPECT_TRUE(c.scopes.size() == 2);
    }
  }
}

TEST(ws_discovery, parse_empty_when_no_matches) {
  static constexpr const char kEmpty[] =
      R"xml(<?xml version="1.0"?><s:Envelope xmlns:s="x"><s:Body/></s:Envelope>)xml";
  auto cams = wsd::parse_probe_match(kEmpty);
  EXPECT_TRUE(cams.empty());
}
