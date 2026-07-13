#ifndef HOST_NET_H
#define HOST_NET_H

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vpipe::netx {

// Thin RAII wrapper around an IPv4 UDP socket. Used by WS-Discovery
// for multicast probe + recv. Errors are reported by setting `_fd =
// -1` and returning false/empty optionals; the caller decides
// whether to escalate via session()->warn / error.
class UdpSocket {
public:
  UdpSocket() = default;
  UdpSocket(const UdpSocket&)            = delete;
  UdpSocket& operator=(const UdpSocket&) = delete;
  UdpSocket(UdpSocket&&) noexcept;
  UdpSocket& operator=(UdpSocket&&) noexcept;
  ~UdpSocket();

  // Allocate the socket and bind(INADDR_ANY:0). Returns false on
  // failure; on success valid() == true.
  bool bind_ephemeral();

  // setsockopt(IP_MULTICAST_LOOP=1).
  bool enable_multicast_loop();

  // setsockopt(IP_MULTICAST_TTL=ttl). Default-1 only crosses
  // link-local; 4 reaches most home/office LANs.
  bool set_multicast_ttl(int ttl);

  // setsockopt(SO_RCVTIMEO).
  bool set_recv_timeout(std::chrono::milliseconds d);

  // sendto(payload, ip, port). IPv4 dotted-quad ip string.
  bool send_to(std::string_view payload,
               const char*      ip,
               uint16_t         port);

  // Single recvfrom. Returns the datagram and the sender's
  // "ip:port" string on success; nullopt on timeout / error.
  std::optional<std::pair<std::string, std::string>>
  recv_one();

  bool valid() const noexcept { return _fd >= 0; }
  int  fd()    const noexcept { return _fd; }

private:
  int _fd = -1;
};

// Non-loopback IPv4 addresses bound to local interfaces. Returns one
// dotted-quad string per interface; useful for deriving the local
// /24 to sweep when multicast WS-Discovery comes up empty.
std::vector<std::string>
local_ipv4_addresses();

// Best-effort IPv4 address of the primary network interface: prefers
// en0 (the built-in LAN port on macOS), else the first non-loopback
// IPv4 that is up. Empty string if none is up -- callers fall back to
// 127.0.0.1 / 0.0.0.0. Unlike local_ipv4_addresses(), this applies the
// en0-first preference and returns a single "reachable-from-the-LAN"
// address, which is what an in-process HTTP server wants to bind to so
// other devices can connect out of the box.
std::string
primary_ipv4();

// Fire one tiny UDP datagram at each candidate (to the discard port
// 9). Errors are ignored; the only purpose is to make the kernel
// resolve the dest MAC, populating the ARP cache so a follow-up
// `arp_cache_ipv4()` call can identify live hosts. Best-effort.
void
warm_arp_cache(const std::vector<std::string>& ips);

// Snapshot of the kernel's IPv4 ARP cache (sysctl(NET_RT_FLAGS) on
// macOS). Returns only entries with a resolved hardware address --
// incomplete / pending entries are filtered out. May return empty if
// the cache lookup fails (we don't have privileges, the system lacks
// the interface, etc.); callers should treat empty as "ARP filtering
// unavailable, probe every candidate".
std::vector<std::string>
arp_cache_ipv4();

// Blocking TCP request: connect to host:port, write `request`, read
// response until the peer closes. Returns the raw response (headers
// + body, exactly as received). On any failure returns an empty
// string; the caller distinguishes from a legitimate empty-body
// response by checking the HTTP status from the parser.
//
// `timeout` is applied to connect + each read; the total wall time
// can be up to 2 * timeout in pathological cases (connect blocks
// then every read blocks). For LAN ONVIF that's fine.
std::string
tcp_request(std::string_view          host,
            uint16_t                  port,
            std::string_view          request,
            std::chrono::milliseconds timeout);

}

#endif
