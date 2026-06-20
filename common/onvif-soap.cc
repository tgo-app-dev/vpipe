#include "common/onvif-soap.h"

#include <pugixml.hpp>

#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#include <CommonCrypto/CommonRandom.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace vpipe::onvif {

namespace {

bool
fill_random(unsigned char* buf, size_t n)
{
#if defined(__APPLE__)
  if (CCRandomGenerateBytes(buf, n) == kCCSuccess) {
    return true;
  }
#endif
  int fd = ::open("/dev/urandom", O_RDONLY);
  if (fd < 0) { return false; }
  ssize_t got = ::read(fd, buf, n);
  ::close(fd);
  return got == static_cast<ssize_t>(n);
}

constexpr char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// ISO-8601 UTC timestamp: "YYYY-MM-DDTHH:MM:SS.000Z"
std::string
iso_utc_now()
{
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[40];
  std::snprintf(buf, sizeof(buf),
                "%04d-%02d-%02dT%02d:%02d:%02d.000Z",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec);
  return buf;
}

// Locate the first descendant whose local-name matches `name`.
// pugixml supports XPath but no namespace-blind matcher out of the
// box, so we hand-walk.
pugi::xml_node
find_local(const pugi::xml_node& root, const char* name)
{
  for (pugi::xml_node n : root.children()) {
    const char* tag = n.name();
    const char* colon = std::strchr(tag, ':');
    const char* local = colon ? colon + 1 : tag;
    if (std::strcmp(local, name) == 0) {
      return n;
    }
    pugi::xml_node deeper = find_local(n, name);
    if (deeper) {
      return deeper;
    }
  }
  return pugi::xml_node{};
}

// Collect all descendants whose local-name matches `name`.
void
find_all_local(const pugi::xml_node& root,
               const char*           name,
               std::vector<pugi::xml_node>& out)
{
  for (pugi::xml_node n : root.children()) {
    const char* tag = n.name();
    const char* colon = std::strchr(tag, ':');
    const char* local = colon ? colon + 1 : tag;
    if (std::strcmp(local, name) == 0) {
      out.push_back(n);
    }
    find_all_local(n, name, out);
  }
}

std::string
node_text(const pugi::xml_node& n)
{
  return n.text().get();
}

}

std::string
base64_encode(std::string_view bytes)
{
  const unsigned char* in =
      reinterpret_cast<const unsigned char*>(bytes.data());
  size_t      n = bytes.size();
  std::string out;
  out.reserve(((n + 2) / 3) * 4);
  size_t i = 0;
  while (i + 3 <= n) {
    uint32_t v = (uint32_t(in[i]) << 16) |
                 (uint32_t(in[i + 1]) << 8) |
                  uint32_t(in[i + 2]);
    out.push_back(kB64[(v >> 18) & 0x3f]);
    out.push_back(kB64[(v >> 12) & 0x3f]);
    out.push_back(kB64[(v >>  6) & 0x3f]);
    out.push_back(kB64[ v        & 0x3f]);
    i += 3;
  }
  if (i < n) {
    uint32_t v = uint32_t(in[i]) << 16;
    if (i + 1 < n) { v |= uint32_t(in[i + 1]) << 8; }
    out.push_back(kB64[(v >> 18) & 0x3f]);
    out.push_back(kB64[(v >> 12) & 0x3f]);
    if (i + 1 < n) {
      out.push_back(kB64[(v >> 6) & 0x3f]);
    } else {
      out.push_back('=');
    }
    out.push_back('=');
  }
  return out;
}

std::string
sha1_raw(std::string_view bytes)
{
#if defined(__APPLE__)
  unsigned char digest[CC_SHA1_DIGEST_LENGTH];
  CC_SHA1(bytes.data(),
          static_cast<CC_LONG>(bytes.size()),
          digest);
  return std::string(reinterpret_cast<const char*>(digest),
                     CC_SHA1_DIGEST_LENGTH);
#else
  (void)bytes;
  return {};
#endif
}

std::string
make_wsse_security(const WsseAuth& auth)
{
  unsigned char nonce[16];
  if (!fill_random(nonce, sizeof(nonce))) {
    std::memset(nonce, 0, sizeof(nonce));
  }
  std::string nonce_str(reinterpret_cast<const char*>(nonce),
                         sizeof(nonce));
  std::string created = iso_utc_now();

  std::string to_digest;
  to_digest.reserve(sizeof(nonce) + created.size() + auth.password.size());
  to_digest.append(nonce_str);
  to_digest.append(created);
  to_digest.append(auth.password);

  std::string digest = base64_encode(sha1_raw(to_digest));
  std::string b64nonce = base64_encode(nonce_str);

  std::string out;
  out.append(
      "<wsse:Security s:mustUnderstand=\"1\""
      " xmlns:wsse=\"http://docs.oasis-open.org/wss/2004/01/"
      "oasis-200401-wss-wssecurity-secext-1.0.xsd\""
      " xmlns:wsu=\"http://docs.oasis-open.org/wss/2004/01/"
      "oasis-200401-wss-wssecurity-utility-1.0.xsd\">"
      "<wsse:UsernameToken>"
      "<wsse:Username>");
  out.append(auth.user);
  out.append("</wsse:Username>"
             "<wsse:Password Type=\"http://docs.oasis-open.org/wss/"
             "2004/01/oasis-200401-wss-username-token-profile-1.0"
             "#PasswordDigest\">");
  out.append(digest);
  out.append("</wsse:Password>"
             "<wsse:Nonce EncodingType=\"http://docs.oasis-open.org/"
             "wss/2004/01/oasis-200401-wss-soap-message-security-1.0"
             "#Base64Binary\">");
  out.append(b64nonce);
  out.append("</wsse:Nonce>"
             "<wsu:Created>");
  out.append(created);
  out.append("</wsu:Created></wsse:UsernameToken></wsse:Security>");
  return out;
}

std::string
make_envelope(std::string_view action,
              std::string_view body_inner,
              const WsseAuth*  wsse)
{
  std::string out;
  out.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
             "<s:Envelope"
             " xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\""
             " xmlns:a=\"http://schemas.xmlsoap.org/ws/2004/08/"
             "addressing\""
             " xmlns:tds=\"http://www.onvif.org/ver10/device/wsdl\""
             " xmlns:trt=\"http://www.onvif.org/ver10/media/wsdl\""
             " xmlns:tt=\"http://www.onvif.org/ver10/schema\">");
  out.append("<s:Header>");
  if (!action.empty()) {
    out.append("<a:Action s:mustUnderstand=\"1\">");
    out.append(action);
    out.append("</a:Action>");
  }
  if (wsse) {
    out.append(make_wsse_security(*wsse));
  }
  out.append("</s:Header><s:Body>");
  out.append(body_inner);
  out.append("</s:Body></s:Envelope>");
  return out;
}

std::string
body_get_capabilities()
{
  return "<tds:GetCapabilities>"
         "<tds:Category>All</tds:Category>"
         "</tds:GetCapabilities>";
}

std::string
body_get_profiles()
{
  return "<trt:GetProfiles/>";
}

std::string
body_get_system_date_and_time()
{
  return "<tds:GetSystemDateAndTime/>";
}

std::string
body_get_endpoint_reference()
{
  return "<tds:GetEndpointReference/>";
}

std::optional<std::string>
parse_endpoint_reference(std::string_view xml)
{
  pugi::xml_document doc;
  if (!doc.load_buffer(xml.data(), xml.size())) {
    return std::nullopt;
  }
  // Response shape:
  //   <tds:GetEndpointReferenceResponse>
  //     <tds:GUID>urn:uuid:xxxxxxxx-...</tds:GUID>
  //   </tds:GetEndpointReferenceResponse>
  // Fallback: some firmwares wrap an EndpointReference/Address pair
  // instead of GUID; we look for either.
  pugi::xml_node g = find_local(doc, "GUID");
  if (g) {
    std::string s = node_text(g);
    if (s.compare(0, 9, "urn:uuid:") == 0) { s.erase(0, 9); }
    if (!s.empty()) { return s; }
  }
  pugi::xml_node er = find_local(doc, "EndpointReference");
  if (er) {
    pugi::xml_node addr = find_local(er, "Address");
    if (addr) {
      std::string s = node_text(addr);
      if (s.compare(0, 9, "urn:uuid:") == 0) { s.erase(0, 9); }
      if (!s.empty()) { return s; }
    }
  }
  return std::nullopt;
}

bool
looks_like_onvif_datetime_response(std::string_view xml)
{
  pugi::xml_document doc;
  if (!doc.load_buffer(xml.data(), xml.size())) {
    return false;
  }
  return static_cast<bool>(find_local(doc, "SystemDateAndTime"));
}

std::string
body_get_stream_uri(std::string_view profile_token)
{
  std::string out;
  out.append("<trt:GetStreamUri>"
             "<trt:StreamSetup>"
             "<tt:Stream>RTP-Unicast</tt:Stream>"
             "<tt:Transport><tt:Protocol>RTSP</tt:Protocol>"
             "</tt:Transport>"
             "</trt:StreamSetup>"
             "<trt:ProfileToken>");
  out.append(profile_token);
  out.append("</trt:ProfileToken></trt:GetStreamUri>");
  return out;
}

std::optional<std::string>
parse_media_xaddr(std::string_view xml)
{
  pugi::xml_document doc;
  if (!doc.load_buffer(xml.data(), xml.size())) {
    return std::nullopt;
  }
  // Find <Media>...<XAddr>...</XAddr>...</Media>
  pugi::xml_node media = find_local(doc, "Media");
  if (!media) {
    return std::nullopt;
  }
  pugi::xml_node x = find_local(media, "XAddr");
  if (!x) {
    return std::nullopt;
  }
  std::string url = node_text(x);
  if (url.empty()) {
    return std::nullopt;
  }
  return url;
}

std::optional<std::vector<Profile>>
parse_profiles(std::string_view xml)
{
  pugi::xml_document doc;
  if (!doc.load_buffer(xml.data(), xml.size())) {
    return std::nullopt;
  }
  std::vector<pugi::xml_node> profs;
  find_all_local(doc, "Profiles", profs);
  if (profs.empty()) {
    // ONVIF Media1 uses <trt:Profiles>; Media2 uses different
    // schema. We only support Media1 here.
    return std::nullopt;
  }
  std::vector<Profile> out;
  out.reserve(profs.size());
  for (const auto& p : profs) {
    Profile pf;
    pf.token = p.attribute("token").as_string();
    if (pf.token.empty()) {
      // Some firmwares put the token in a child <Token> element.
      pugi::xml_node tn = find_local(p, "Token");
      if (tn) { pf.token = node_text(tn); }
    }
    pugi::xml_node n = find_local(p, "Name");
    if (n) { pf.name = node_text(n); }
    if (!pf.token.empty()) {
      out.push_back(std::move(pf));
    }
  }
  if (out.empty()) {
    return std::nullopt;
  }
  return out;
}

std::optional<std::string>
parse_stream_uri(std::string_view xml)
{
  pugi::xml_document doc;
  if (!doc.load_buffer(xml.data(), xml.size())) {
    return std::nullopt;
  }
  pugi::xml_node mu = find_local(doc, "MediaUri");
  if (!mu) {
    return std::nullopt;
  }
  pugi::xml_node u = find_local(mu, "Uri");
  if (!u) {
    return std::nullopt;
  }
  std::string s = node_text(u);
  if (s.empty()) { return std::nullopt; }
  return s;
}

bool
is_auth_fault(std::string_view xml)
{
  pugi::xml_document doc;
  if (!doc.load_buffer(xml.data(), xml.size())) {
    return false;
  }
  pugi::xml_node fault = find_local(doc, "Fault");
  if (!fault) {
    return false;
  }
  // Collect every <Value>...</Value> under the Fault Code subtree
  // and search for known auth markers.
  std::vector<pugi::xml_node> vals;
  find_all_local(fault, "Value", vals);
  for (const auto& v : vals) {
    std::string t = node_text(v);
    if (t.find("NotAuthorized")        != std::string::npos
     || t.find("FailedAuthentication") != std::string::npos) {
      return true;
    }
  }
  // Also probe Subcode / Reason text for the same markers.
  std::vector<pugi::xml_node> texts;
  find_all_local(fault, "Text", texts);
  for (const auto& t : texts) {
    std::string s = node_text(t);
    if (s.find("Not Authorized")        != std::string::npos
     || s.find("authentication failed") != std::string::npos
     || s.find("NotAuthorized")         != std::string::npos
     || s.find("FailedAuthentication")  != std::string::npos) {
      return true;
    }
  }
  return false;
}

}
