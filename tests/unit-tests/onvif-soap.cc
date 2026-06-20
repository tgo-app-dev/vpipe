#include "minitest.h"
#include "common/onvif-soap.h"

#include <string>

using namespace vpipe;

TEST(onvif_soap, base64_known_vectors) {
  EXPECT_TRUE(onvif::base64_encode("") == "");
  EXPECT_TRUE(onvif::base64_encode("f")    == "Zg==");
  EXPECT_TRUE(onvif::base64_encode("fo")   == "Zm8=");
  EXPECT_TRUE(onvif::base64_encode("foo")  == "Zm9v");
  EXPECT_TRUE(onvif::base64_encode("foobar") == "Zm9vYmFy");
}

TEST(onvif_soap, make_wsse_security_well_formed) {
  onvif::WsseAuth a{"admin", "pw"};
  auto s = onvif::make_wsse_security(a);
  EXPECT_TRUE(s.find("<wsse:Security") != std::string::npos);
  EXPECT_TRUE(s.find("<wsse:UsernameToken>") != std::string::npos);
  EXPECT_TRUE(s.find("admin") != std::string::npos);
  EXPECT_TRUE(s.find("PasswordDigest") != std::string::npos);
  EXPECT_TRUE(s.find("<wsse:Nonce") != std::string::npos);
  EXPECT_TRUE(s.find("<wsu:Created>") != std::string::npos);
  // The plaintext password must NOT appear in the envelope -- only
  // its digest. Sanity check.
  EXPECT_TRUE(s.find(">pw<") == std::string::npos);
}

TEST(onvif_soap, make_envelope_contains_wsse_when_provided) {
  onvif::WsseAuth a{"admin", "pw"};
  auto env = onvif::make_envelope(
      "http://www.onvif.org/ver10/device/wsdl/GetCapabilities",
      onvif::body_get_capabilities(), &a);
  EXPECT_TRUE(env.find("<wsse:Security") != std::string::npos);
  EXPECT_TRUE(env.find("GetCapabilities") != std::string::npos);
}

TEST(onvif_soap, make_envelope_omits_wsse_when_null) {
  auto env = onvif::make_envelope("", onvif::body_get_profiles(),
                                  nullptr);
  EXPECT_TRUE(env.find("<wsse:Security") == std::string::npos);
  EXPECT_TRUE(env.find("GetProfiles") != std::string::npos);
}

static constexpr const char kCapsFixture[] = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tt="http://www.onvif.org/ver10/schema"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
 <s:Body>
  <tds:GetCapabilitiesResponse>
   <tds:Capabilities>
    <tt:Media>
     <tt:XAddr>http://192.168.1.42/onvif/Media</tt:XAddr>
    </tt:Media>
    <tt:Device>
     <tt:XAddr>http://192.168.1.42/onvif/Device</tt:XAddr>
    </tt:Device>
   </tds:Capabilities>
  </tds:GetCapabilitiesResponse>
 </s:Body>
</s:Envelope>)xml";

TEST(onvif_soap, parse_media_xaddr_happy) {
  auto u = onvif::parse_media_xaddr(kCapsFixture);
  EXPECT_TRUE(u.has_value());
  EXPECT_TRUE(*u == "http://192.168.1.42/onvif/Media");
}

static constexpr const char kProfilesFixture[] = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tt="http://www.onvif.org/ver10/schema"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
 <s:Body>
  <trt:GetProfilesResponse>
   <trt:Profiles token="Profile_1" fixed="true">
    <tt:Name>MainStream</tt:Name>
   </trt:Profiles>
   <trt:Profiles token="Profile_2" fixed="true">
    <tt:Name>SubStream</tt:Name>
   </trt:Profiles>
  </trt:GetProfilesResponse>
 </s:Body>
</s:Envelope>)xml";

TEST(onvif_soap, parse_profiles_two) {
  auto pr = onvif::parse_profiles(kProfilesFixture);
  EXPECT_TRUE(pr.has_value());
  EXPECT_TRUE(pr && pr->size() == 2);
  EXPECT_TRUE((*pr)[0].token == "Profile_1");
  EXPECT_TRUE((*pr)[0].name  == "MainStream");
  EXPECT_TRUE((*pr)[1].token == "Profile_2");
}

static constexpr const char kStreamUriFixture[] = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tt="http://www.onvif.org/ver10/schema"
            xmlns:trt="http://www.onvif.org/ver10/media/wsdl">
 <s:Body>
  <trt:GetStreamUriResponse>
   <trt:MediaUri>
    <tt:Uri>rtsp://192.168.1.42:554/Streaming/Channels/101</tt:Uri>
    <tt:Timeout>PT60S</tt:Timeout>
   </trt:MediaUri>
  </trt:GetStreamUriResponse>
 </s:Body>
</s:Envelope>)xml";

TEST(onvif_soap, parse_stream_uri_happy) {
  auto u = onvif::parse_stream_uri(kStreamUriFixture);
  EXPECT_TRUE(u.has_value());
  EXPECT_TRUE(*u == "rtsp://192.168.1.42:554/Streaming/Channels/101");
}

static constexpr const char kAuthFaultFixture[] = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">
 <s:Body>
  <s:Fault>
   <s:Code>
    <s:Value>s:Sender</s:Value>
    <s:Subcode>
     <s:Value>ter:NotAuthorized</s:Value>
    </s:Subcode>
   </s:Code>
   <s:Reason><s:Text xml:lang="en">Sender not authorized</s:Text></s:Reason>
  </s:Fault>
 </s:Body>
</s:Envelope>)xml";

TEST(onvif_soap, is_auth_fault_detects_NotAuthorized) {
  EXPECT_TRUE(onvif::is_auth_fault(kAuthFaultFixture));
}

TEST(onvif_soap, is_auth_fault_false_for_caps) {
  EXPECT_FALSE(onvif::is_auth_fault(kCapsFixture));
}

static constexpr const char kEndpointRefFixture[] = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
 <s:Body>
  <tds:GetEndpointReferenceResponse>
   <tds:GUID>urn:uuid:11112222-3333-4444-5555-666677778888</tds:GUID>
  </tds:GetEndpointReferenceResponse>
 </s:Body>
</s:Envelope>)xml";

TEST(onvif_soap, parse_endpoint_reference_strips_urn_uuid) {
  auto u = onvif::parse_endpoint_reference(kEndpointRefFixture);
  EXPECT_TRUE(u.has_value());
  EXPECT_TRUE(u && *u == "11112222-3333-4444-5555-666677778888");
}

static constexpr const char kDateTimeFixture[] = R"xml(<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
            xmlns:tt="http://www.onvif.org/ver10/schema"
            xmlns:tds="http://www.onvif.org/ver10/device/wsdl">
 <s:Body>
  <tds:GetSystemDateAndTimeResponse>
   <tds:SystemDateAndTime>
    <tt:DateTimeType>Manual</tt:DateTimeType>
    <tt:DaylightSavings>false</tt:DaylightSavings>
   </tds:SystemDateAndTime>
  </tds:GetSystemDateAndTimeResponse>
 </s:Body>
</s:Envelope>)xml";

TEST(onvif_soap, looks_like_onvif_datetime_response_true) {
  EXPECT_TRUE(onvif::looks_like_onvif_datetime_response(kDateTimeFixture));
}

TEST(onvif_soap, looks_like_onvif_datetime_response_false_for_caps) {
  EXPECT_FALSE(onvif::looks_like_onvif_datetime_response(kCapsFixture));
}
