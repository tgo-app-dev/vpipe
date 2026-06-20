#ifndef ONVIF_SOAP_H
#define ONVIF_SOAP_H

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe::onvif {

struct WsseAuth {
  std::string user;
  std::string password;
};

// Build the inner <wsse:Security> XML for a UsernameToken digest.
// Uses a fresh 16-byte random nonce and the current UTC time. Pure
// of side effects beyond the RNG read.
std::string
make_wsse_security(const WsseAuth& auth);

// Wrap `body_inner_xml` in a complete SOAP 1.2 envelope. `body_inner`
// is the literal Body content (no <s:Body> wrapper). If `wsse` is
// non-null, a <wsse:Security> header is inserted.
std::string
make_envelope(std::string_view action,
              std::string_view body_inner,
              const WsseAuth*  wsse);

// Body templates for the three ONVIF calls we need.
std::string
body_get_capabilities();

std::string
body_get_profiles();

std::string
body_get_stream_uri(std::string_view profile_token);

// No-auth probes used by the /24 subnet-sweep fallback.
//   GetSystemDateAndTime — universally supported on ONVIF cameras;
//                          its response identifies a host as ONVIF.
//   GetEndpointReference  — returns the device's stable UUID (the
//                           same value WS-Discovery would expose).
std::string
body_get_system_date_and_time();

std::string
body_get_endpoint_reference();

// Parse "<tds:GUID>urn:uuid:...</tds:GUID>" out of a
// GetEndpointReferenceResponse. Returns the bare UUID (urn:uuid:
// prefix stripped) on success.
std::optional<std::string>
parse_endpoint_reference(std::string_view xml);

// True iff `xml` is a valid GetSystemDateAndTimeResponse -- used to
// distinguish ONVIF cameras from random HTTP servers during the
// subnet sweep.
bool
looks_like_onvif_datetime_response(std::string_view xml);

// Response parsers. Each returns nullopt when the input is not a
// recognised success envelope. All parsers use local-name predicates
// rather than namespace prefixes -- vendors disagree on prefixes.
std::optional<std::string>
parse_media_xaddr(std::string_view xml);

struct Profile {
  std::string token;
  std::string name;
};
std::optional<std::vector<Profile>>
parse_profiles(std::string_view xml);

std::optional<std::string>
parse_stream_uri(std::string_view xml);

// True iff `xml` is a SOAP Fault with a NotAuthorized /
// FailedAuthentication subcode -- i.e. credentials were rejected.
bool
is_auth_fault(std::string_view xml);

// Base64 of arbitrary bytes. Exposed so tests and ws-discovery can
// reuse the same encoder.
std::string
base64_encode(std::string_view bytes);

// SHA1 of arbitrary bytes, returned as a raw 20-byte string. Exposed
// for the WSSE digest test.
std::string
sha1_raw(std::string_view bytes);

}

#endif
