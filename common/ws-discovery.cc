#include "common/ws-discovery.h"
#include "common/host-net.h"
#include "common/mini-http.h"
#include "common/onvif-soap.h"
#include "interfaces/session-context-intf.h"
#include "common/vpipe-format.h"

#include <pugixml.hpp>

#if defined(__APPLE__)
#include <CommonCrypto/CommonRandom.h>
#endif

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

#include <iostream>

namespace vpipe::wsd {

namespace {

constexpr const char* kProbeIp   = "239.255.255.250";
constexpr uint16_t    kProbePort = 3702;

bool
random_bytes(unsigned char* out, size_t n)
{
#if defined(__APPLE__)
  if (CCRandomGenerateBytes(out, n) == kCCSuccess) {
    return true;
  }
#endif
  int fd = ::open("/dev/urandom", O_RDONLY);
  if (fd < 0) { return false; }
  ssize_t got = ::read(fd, out, n);
  ::close(fd);
  return got == static_cast<ssize_t>(n);
}

std::string
random_uuid_v4()
{
  unsigned char b[16];
  if (!random_bytes(b, 16)) {
    std::memset(b, 0, 16);
  }
  // version 4, variant 1
  b[6] = static_cast<unsigned char>((b[6] & 0x0f) | 0x40);
  b[8] = static_cast<unsigned char>((b[8] & 0x3f) | 0x80);
  char buf[40];
  std::snprintf(buf, sizeof(buf),
      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      b[0],b[1],b[2],b[3], b[4],b[5], b[6],b[7],
      b[8],b[9], b[10],b[11],b[12],b[13],b[14],b[15]);
  return buf;
}

// Local-name walkers, copy-paste of the onvif-soap.cc style. We
// duplicate to keep the two TUs decoupled.
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
    if (deeper) { return deeper; }
  }
  return pugi::xml_node{};
}

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

std::vector<std::string>
split_whitespace(std::string_view s)
{
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() &&
           std::isspace(static_cast<unsigned char>(s[i]))) { ++i; }
    size_t start = i;
    while (i < s.size() &&
           !std::isspace(static_cast<unsigned char>(s[i]))) { ++i; }
    if (i > start) {
      out.emplace_back(s.substr(start, i - start));
    }
  }
  return out;
}

std::string
strip_urn_uuid(std::string_view s)
{
  constexpr const char kPrefix[] = "urn:uuid:";
  if (s.size() > 9
      && std::strncmp(s.data(), kPrefix, 9) == 0) {
    return std::string(s.substr(9));
  }
  return std::string(s);
}

}

std::string
make_probe_envelope(std::string_view message_id_uuid)
{
  std::string out;
  out.append(
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
      "<s:Envelope"
      " xmlns:s=\"http://www.w3.org/2003/05/soap-envelope\""
      " xmlns:a=\"http://schemas.xmlsoap.org/ws/2004/08/addressing\""
      " xmlns:d=\"http://schemas.xmlsoap.org/ws/2005/04/discovery\""
      " xmlns:dn=\"http://www.onvif.org/ver10/network/wsdl\">"
      "<s:Header>"
      "<a:Action s:mustUnderstand=\"1\">"
      "http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe"
      "</a:Action>"
      "<a:MessageID>urn:uuid:");
  out.append(message_id_uuid);
  out.append("</a:MessageID>"
             "<a:To s:mustUnderstand=\"1\">"
             "urn:schemas-xmlsoap-org:ws:2005:04:discovery"
             "</a:To>"
             "</s:Header>"
             "<s:Body>"
             "<d:Probe>"
             "<d:Types>dn:NetworkVideoTransmitter</d:Types>"
             "</d:Probe>"
             "</s:Body>"
             "</s:Envelope>");
  return out;
}

std::vector<DiscoveredCamera>
parse_probe_match(std::string_view xml)
{
  std::vector<DiscoveredCamera> out;
  pugi::xml_document doc;
  if (!doc.load_buffer(xml.data(), xml.size())) {
    return out;
  }
  std::vector<pugi::xml_node> matches;
  find_all_local(doc, "ProbeMatch", matches);
  for (const auto& m : matches) {
    DiscoveredCamera c;

    // EndpointReference/Address holds the UUID.
    pugi::xml_node er = find_local(m, "EndpointReference");
    if (er) {
      pugi::xml_node addr = find_local(er, "Address");
      if (addr) {
        c.uuid = strip_urn_uuid(addr.text().get());
      }
    }
    if (c.uuid.empty()) { continue; }

    pugi::xml_node ts = find_local(m, "Types");
    if (ts) {
      c.types = split_whitespace(ts.text().get());
    }
    pugi::xml_node ss = find_local(m, "Scopes");
    if (ss) {
      c.scopes = split_whitespace(ss.text().get());
    }
    pugi::xml_node xs = find_local(m, "XAddrs");
    if (xs) {
      for (const auto& u : split_whitespace(xs.text().get())) {
        if (u.compare(0, 7, "http://") == 0) {
          c.xaddr = u;
          break;
        }
      }
      if (c.xaddr.empty() && !split_whitespace(xs.text().get()).empty()) {
        c.xaddr = split_whitespace(xs.text().get()).front();
      }
    }
    out.push_back(std::move(c));
  }
  return out;
}

namespace {

// IPv4 dotted-quad parser: "a.b.c.d" -> 32-bit host-order value, or
// nullopt if malformed.
std::optional<uint32_t>
parse_dotted_quad(std::string_view s)
{
  unsigned a = 0, b = 0, c = 0, d = 0;
  char     buf[32];
  if (s.size() >= sizeof(buf)) { return std::nullopt; }
  std::memcpy(buf, s.data(), s.size());
  buf[s.size()] = '\0';
  int n = std::sscanf(buf, "%u.%u.%u.%u", &a, &b, &c, &d);
  if (n != 4 || a > 255 || b > 255 || c > 255 || d > 255) {
    return std::nullopt;
  }
  return (a << 24) | (b << 16) | (c << 8) | d;
}

std::string
format_dotted_quad(uint32_t ip)
{
  char buf[INET_ADDRSTRLEN];
  std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u",
                (ip >> 24) & 0xffu, (ip >> 16) & 0xffu,
                (ip >>  8) & 0xffu,  ip        & 0xffu);
  return buf;
}

}

std::vector<DiscoveredCamera>
sweep_subnet_24(std::chrono::milliseconds probe_timeout,
                const SessionContextIntf* log_sink)
{
  std::vector<DiscoveredCamera> out;

  auto locals = netx::local_ipv4_addresses();
  if (locals.empty()) {
    if (log_sink) {
      log_sink->warn(fmt(
          "ws-discovery sweep: no non-loopback IPv4 interfaces"));
    }
    return out;
  }

  // Walk each interface's /24 once; dedupe candidates across
  // interfaces that happen to share a subnet.
  std::vector<std::string> candidates;
  for (const auto& local : locals) {
    auto v = parse_dotted_quad(local);
    if (!v) { continue; }
    uint32_t base = *v & 0xffffff00u;
    for (unsigned i = 1; i <= 254; ++i) {
      uint32_t ip = base | i;
      if (ip == *v) { continue; }
      candidates.push_back(format_dotted_quad(ip));
    }
  }
  if (candidates.empty()) { return out; }

  // Best-effort ARP filtering to drop hosts that demonstrably don't
  // exist. We warm the kernel ARP cache with a tiny UDP knock to
  // each candidate, wait briefly, then read the resolved entries
  // via sysctl. If the cache read fails or comes back empty we
  // probe every candidate -- slower but never wrong.
  netx::warm_arp_cache(candidates);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  auto arp = netx::arp_cache_ipv4();
  std::unordered_set<std::string> arp_set(arp.begin(), arp.end());

  std::vector<std::string> targets;
  if (!arp_set.empty()) {
    targets.reserve(candidates.size());
    for (const auto& ip : candidates) {
      if (arp_set.count(ip)) { targets.push_back(ip); }
    }
  }
  if (targets.empty()) {
    targets = std::move(candidates);
  }
  if (log_sink) {
    log_sink->info(fmt(
        "ws-discovery sweep: probing {} target(s) via unicast "
        "WS-Discovery (arp-cache size {})",
        targets.size(), arp.size()));
  }

  // Unicast WS-Discovery: send the same Probe envelope used for
  // multicast to each candidate's UDP/3702. Responses come back as
  // ProbeMatch envelopes from the camera's own IP -- parsed by
  // the same `parse_probe_match` used for multicast. This matches
  // what ONVIF cameras do when reached directly: they speak the
  // discovery protocol whether the destination is multicast or
  // unicast.
  netx::UdpSocket sock;
  if (!sock.bind_ephemeral()) {
    if (log_sink) {
      log_sink->warn(fmt(
          "ws-discovery sweep: socket/bind failed"));
    }
    return out;
  }
  sock.set_recv_timeout(std::chrono::milliseconds(250));

  std::string msg_id = random_uuid_v4();
  std::string env    = make_probe_envelope(msg_id);
  for (const auto& ip : targets) {
    sock.send_to(env, ip.c_str(), kProbePort);
  }

  std::unordered_map<std::string, DiscoveredCamera> dedup;
  auto deadline = std::chrono::steady_clock::now() + probe_timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto pkt = sock.recv_one();
    if (!pkt) { continue; }
    for (auto& c : parse_probe_match(pkt->first)) {
      dedup.insert_or_assign(c.uuid, std::move(c));
    }
  }

  out.reserve(dedup.size());
  for (auto& kv : dedup) {
    out.push_back(std::move(kv.second));
  }
  std::sort(out.begin(), out.end(),
            [](const DiscoveredCamera& a, const DiscoveredCamera& b) {
              return a.uuid < b.uuid;
            });
  return out;
}

std::vector<DiscoveredCamera>
discover(std::chrono::milliseconds timeout,
         const SessionContextIntf* log_sink)
{
  std::unordered_map<std::string, DiscoveredCamera> dedup;

  netx::UdpSocket sock;
  if (!sock.bind_ephemeral()) {
    if (log_sink) {
      log_sink->warn(fmt("ws-discovery: socket/bind failed"));
    }
    return {};
  }
  sock.enable_multicast_loop();
  sock.set_multicast_ttl(4);
  sock.set_recv_timeout(std::chrono::milliseconds(250));

  std::string msg = random_uuid_v4();
  std::string env = make_probe_envelope(msg);
  if (!sock.send_to(env, kProbeIp, kProbePort)) {
    if (log_sink) {
      log_sink->warn(fmt(
          "ws-discovery: sendto({}:{}) failed",
          kProbeIp, kProbePort));
    }
    return {};
  }

  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto pkt = sock.recv_one();
    if (!pkt) { continue; }
    for (auto& c : parse_probe_match(pkt->first)) {
      dedup.insert_or_assign(c.uuid, std::move(c));
    }
  }

  std::vector<DiscoveredCamera> result;
  result.reserve(dedup.size());
  for (auto& kv : dedup) {
    result.push_back(std::move(kv.second));
  }
  std::sort(result.begin(), result.end(),
            [](const DiscoveredCamera& a, const DiscoveredCamera& b) {
              return a.uuid < b.uuid;
            });
  return result;
}

}
