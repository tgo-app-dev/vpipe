#ifndef WS_DISCOVERY_H
#define WS_DISCOVERY_H

#include <chrono>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
class SessionContextIntf;
}

namespace vpipe::wsd {

struct DiscoveredCamera {
  std::string              uuid;   // bare (no "urn:uuid:" prefix)
  std::string              xaddr;  // first http:// URL from XAddrs
  std::vector<std::string> scopes;
  std::vector<std::string> types;
};

// SOAP envelope for an ONVIF NetworkVideoTransmitter probe.
// `message_id` is the value to splice into <a:MessageID>urn:uuid:..</>.
std::string
make_probe_envelope(std::string_view message_id_uuid);

// Parse a ProbeMatches SOAP envelope. May contain multiple matches.
// Returns one entry per <ProbeMatch>; UUID is bare (urn:uuid: prefix
// stripped), XAddr is the first http:// URL among the
// space-separated XAddrs list.
std::vector<DiscoveredCamera>
parse_probe_match(std::string_view xml);

// End-to-end discovery: open a UDP multicast socket, send the
// probe, recv ProbeMatches until `timeout` elapses, dedupe by uuid.
// `log_sink` (may be nullptr) receives warnings about socket /
// option failures via warn(). Result is sorted by uuid for stable
// display.
std::vector<DiscoveredCamera>
discover(std::chrono::milliseconds  timeout,
         const SessionContextIntf*  log_sink);

// Fallback when multicast WS-Discovery turns up nothing -- common
// when a Wi-Fi AP drops IGMP, the kernel-selected interface isn't
// the camera LAN, or the host lives behind a bridge that doesn't
// forward multicast. Walks the /24 of every non-loopback IPv4
// interface and sends the same WS-Discovery Probe envelope as
// `discover()` to each candidate's UDP/3702 (unicast). ONVIF
// cameras answer unicast probes the same way they answer multicast,
// so the response is parsed by the same `parse_probe_match`. ARP
// cache is used to opportunistically prune obviously-dead IPs;
// when the cache is empty the probe is sent to all 254 hosts.
//
// `probe_timeout` is the wall-clock window for collecting responses
// after all probes have been sent.
std::vector<DiscoveredCamera>
sweep_subnet_24(std::chrono::milliseconds probe_timeout,
                const SessionContextIntf* log_sink);

}

#endif
