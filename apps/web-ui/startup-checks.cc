#include "apps/web-ui/startup-checks.h"

#include "common/host-net.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace vpipe::webui {

namespace {

// Wait up to `total_ms` for `fd` to become ready (write-side when
// `write`, else read-side), polling in <=100 ms slices so a shutdown
// flag (`abort`) interrupts the wait promptly. Returns true if ready,
// false on timeout / abort / error.
bool
wait_fd_(int fd, bool write, int total_ms, const std::atomic<bool>* abort)
{
  int remaining = total_ms;
  while (remaining > 0) {
    if (abort != nullptr && abort->load()) { return false; }
    const int slice = remaining < 100 ? remaining : 100;
    fd_set set;
    FD_ZERO(&set);
    FD_SET(fd, &set);
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = slice * 1000;
    const int r = ::select(fd + 1, write ? nullptr : &set,
                           write ? &set : nullptr, nullptr, &tv);
    if (r > 0) { return true; }
    if (r < 0 && errno != EINTR) { return false; }
    remaining -= slice;
  }
  return false;
}

// ANSI colors, used only when stdout is an interactive terminal and the
// user hasn't opted out via NO_COLOR.
struct Palette {
  const char* reset = "";
  const char* dim   = "";
  const char* green = "";
  const char* yellow = "";
  const char* red   = "";
  const char* bold  = "";
};

Palette
palette_()
{
  Palette p;
  const bool tty = ::isatty(fileno(stdout)) != 0;
  const char* no_color = std::getenv("NO_COLOR");
  if (tty && (no_color == nullptr || *no_color == '\0')) {
    p.reset = "\033[0m";
    p.dim = "\033[2m";
    p.green = "\033[32m";
    p.yellow = "\033[33m";
    p.red = "\033[1;31m";
    p.bold = "\033[1m";
  }
  return p;
}

// Trim leading/trailing whitespace.
std::string
trim_(const std::string& s)
{
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) { return ""; }
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// IPv4 of the default-route gateway, or "" if it can't be determined.
// macOS/BSD: `route -n get default` prints "    gateway: 192.168.1.1".
// Linux: `ip route` prints "default via 192.168.1.1 ...".
std::string
default_gateway_()
{
#if defined(__APPLE__)
  FILE* p = ::popen("route -n get default 2>/dev/null", "r");
#else
  FILE* p = ::popen("ip route show default 2>/dev/null", "r");
#endif
  if (p == nullptr) { return ""; }
  std::string gw;
  char line[256];
  while (std::fgets(line, sizeof(line), p) != nullptr) {
    std::string s(line);
#if defined(__APPLE__)
    const auto pos = s.find("gateway:");
    if (pos != std::string::npos) {
      gw = trim_(s.substr(pos + 8));
      break;
    }
#else
    const auto pos = s.find("via ");
    if (pos != std::string::npos) {
      std::string rest = trim_(s.substr(pos + 4));
      gw = rest.substr(0, rest.find(' '));
      break;
    }
#endif
  }
  ::pclose(p);
  return gw;
}

// Probe whether we can reach the default gateway over TCP. Returns:
//    1  reachable (connected, or RST -- host responded)
//    0  blocked / unreachable (timeout / no route)
//   -1  could not test (no gateway found)
// NOTE: the gateway is NOT a reliable Local Network permission probe on
// macOS -- the OS special-cases the default route, so the gateway stays
// reachable even when access to *other* LAN hosts (cameras, RTSP, mDNS)
// is denied. We use this only to disambiguate "network down" from
// "local network gated" (see probe_mdns_).
int
probe_gateway_(std::string* gateway_out, const std::atomic<bool>* abort)
{
  const std::string gw = default_gateway_();
  if (gateway_out) { *gateway_out = gw; }
  if (gw.empty()) { return -1; }

  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(80);   // router admin / common LAN port
  if (::inet_pton(AF_INET, gw.c_str(), &addr.sin_addr) != 1) { return -1; }

  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) { return -1; }
  // Non-blocking connect + select so a blocked/dropped path times out
  // quickly instead of stalling startup.
  int flags = ::fcntl(fd, F_GETFL, 0);
  ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);

  int result = 0;   // assume blocked until proven reachable
  const int rc = ::connect(fd, reinterpret_cast<struct sockaddr*>(&addr),
                           sizeof(addr));
  if (rc == 0) {
    result = 1;   // immediate connect
  } else if (errno == ECONNREFUSED) {
    result = 1;   // host responded (port closed) -> reachable
  } else if (errno == EINPROGRESS) {
    if (wait_fd_(fd, /*write=*/true, 1500, abort)) {
      int err = 0;
      socklen_t len = sizeof(err);
      ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
      // 0 == connected; ECONNREFUSED == host reachable but port closed.
      result = (err == 0 || err == ECONNREFUSED) ? 1 : 0;
    } else {
      result = 0;   // timeout / abort -> treat as unreachable
    }
  } else {
    result = 0;   // EHOSTUNREACH / ENETUNREACH / EPERM / ...
  }
  ::close(fd);
  return result;
}

// Probe the LOCAL NETWORK permission for real by exercising the gated
// path the gateway test misses: an mDNS multicast query to
// 224.0.0.251:5353 out the primary interface, then a short wait for any
// response. macOS gates multicast + non-gateway LAN unicast behind the
// Local Network privacy permission; when it's denied the send is dropped
// (or fails) and no responder replies. Returns:
//    1  a response arrived  -> local-network access works
//    0  no response / send blocked -> likely denied (or no responders)
//   -1  could not run the probe (socket setup failed)
// The QU (unicast-response) bit asks responders to reply straight to our
// ephemeral port, so we don't have to join the group / share port 5353
// with the system mDNSResponder.
int
probe_mdns_(const std::atomic<bool>* abort)
{
  const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) { return -1; }

  // Send out the primary interface (en0) when we can identify it; TTL 1
  // keeps the query link-local.
  const std::string ip = primary_ipv4();
  if (!ip.empty()) {
    struct in_addr ifa;
    if (::inet_pton(AF_INET, ip.c_str(), &ifa) == 1) {
      ::setsockopt(fd, IPPROTO_IP, IP_MULTICAST_IF, &ifa, sizeof(ifa));
    }
  }
  unsigned char ttl = 1;
  ::setsockopt(fd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));

  // mDNS standard query, PTR "_services._dns-sd._udp.local" (the Bonjour
  // service-type meta-query -- anything offering any service answers),
  // QCLASS = IN with the unicast-response bit set.
  static const unsigned char query[] = {
    0x00, 0x00,                          // transaction id
    0x00, 0x00,                          // flags: standard query
    0x00, 0x01,                          // QDCOUNT = 1
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // AN/NS/AR = 0
    0x09, '_', 's', 'e', 'r', 'v', 'i', 'c', 'e', 's',
    0x07, '_', 'd', 'n', 's', '-', 's', 'd',
    0x04, '_', 'u', 'd', 'p',
    0x05, 'l', 'o', 'c', 'a', 'l',
    0x00,
    0x00, 0x0c,                          // QTYPE = PTR
    0x80, 0x01,                          // QCLASS = IN | unicast-response
  };

  struct sockaddr_in dst;
  std::memset(&dst, 0, sizeof(dst));
  dst.sin_family = AF_INET;
  dst.sin_port = htons(5353);
  ::inet_pton(AF_INET, "224.0.0.251", &dst.sin_addr);

  const ssize_t sent = ::sendto(fd, query, sizeof(query), 0,
                                reinterpret_cast<struct sockaddr*>(&dst),
                                sizeof(dst));
  if (sent < 0) {
    // EHOSTUNREACH / ENETUNREACH / EPERM here is the kernel refusing the
    // local-network send -> permission denied.
    ::close(fd);
    return 0;
  }

  int result = 0;
  if (wait_fd_(fd, /*write=*/false, 1200, abort)) {
    unsigned char buf[1500];
    const ssize_t got = ::recvfrom(fd, buf, sizeof(buf), 0, nullptr, nullptr);
    result = (got > 0) ? 1 : 0;
  }
  ::close(fd);
  return result;
}

// Full Disk Access probe: try to read a TCC-protected file. Reading any
// of these requires Full Disk Access; we conclude granted on the first
// successful open, denied on the first EACCES/EPERM, and unknown only if
// none of the candidates exist. Returns 1 granted, 0 denied, -1 unknown.
int
probe_full_disk_access_()
{
#if defined(__APPLE__)
  const char* home = std::getenv("HOME");
  if (home == nullptr || *home == '\0') { return -1; }
  const std::string h(home);
  const std::string candidates[] = {
    h + "/Library/Application Support/com.apple.TCC/TCC.db",
    h + "/Library/Messages/chat.db",
    h + "/Library/Safari/Bookmarks.plist",
  };
  for (const auto& path : candidates) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
      ::close(fd);
      return 1;   // readable -> FDA granted
    }
    if (errno == EACCES || errno == EPERM) { return 0; }   // exists, denied
    // ENOENT / other -> try the next candidate.
  }
  return -1;   // none present -> can't tell
#else
  return -1;
#endif
}

void
print_ok_(const Palette& pal, const char* label, const std::string& detail)
{
  std::printf("  %s[ ok ]%s %-18s %s%s%s\n", pal.green, pal.reset, label,
              pal.dim, detail.c_str(), pal.reset);
}

void
print_warn_(const Palette& pal, const char* label, const std::string& detail)
{
  std::printf("  %s[warn]%s %s%-18s%s %s\n", pal.red, pal.reset, pal.bold,
              label, pal.reset, detail.c_str());
}

void
print_hint_(const Palette& pal, const std::string& text)
{
  std::printf("         %s%s%s\n", pal.yellow, text.c_str(), pal.reset);
}

// Record a result into `out` AND print it, so the terminal and the web
// UI stay in sync. hint_() attaches to the most-recently recorded check.
void
ok_(const Palette& pal, std::vector<PermissionCheck>& out,
    const char* label, const std::string& detail)
{
  print_ok_(pal, label, detail);
  out.push_back(PermissionCheck{label, "ok", detail, {}});
}
void
warn_(const Palette& pal, std::vector<PermissionCheck>& out,
      const char* label, const std::string& detail)
{
  print_warn_(pal, label, detail);
  out.push_back(PermissionCheck{label, "warn", detail, {}});
}
void
hint_(const Palette& pal, std::vector<PermissionCheck>& out,
      const std::string& text)
{
  print_hint_(pal, text);
  if (!out.empty()) { out.back().hints.push_back(text); }
}

}  // namespace

std::string
primary_ipv4()
{
  // Single source of truth: the shared netx helper applies the same
  // en0-first preference. Kept as a thin webui:: alias so existing
  // callers (main.cc, the mDNS probe below) need no change.
  return vpipe::netx::primary_ipv4();
}

#if !defined(__APPLE__)
int
microphone_auth_status()
{
  return -1;   // no portable check off macOS
}
int
audio_input_device_count()
{
  return -1;   // no portable enumeration off macOS
}
#endif

std::vector<PermissionCheck>
run_permission_checks(const std::atomic<bool>* abort)
{
  std::vector<PermissionCheck> checks;
  const Palette pal = palette_();
  std::printf("\n%sStartup checks%s "
              "%s(permissions: System Settings > Privacy & Security)%s\n",
              pal.bold, pal.reset, pal.dim, pal.reset);

  // 2.1 Local-network access. The gateway alone is NOT a valid probe --
  // macOS special-cases the default route, so it stays reachable even
  // when access to other LAN hosts (cameras / RTSP / mDNS) is denied
  // (the exact symptom seen under tmux: gateway OK, 192.168.x.y:554
  // blocked). Exercise the real gated path with an mDNS multicast probe,
  // and use the gateway only to tell "network down" apart from "local
  // network gated".
  {
    const int mdns = probe_mdns_(abort);
    if (mdns == 1) {
      ok_(pal, checks, "Local network", "LAN multicast reachable");
    } else if (abort != nullptr && abort->load()) {
      // Shutting down mid-probe -- don't print a misleading verdict.
    } else if (probe_gateway_(nullptr, abort) == 1) {
      // Gateway reachable but the LAN multicast path is silent -> almost
      // certainly the Local Network permission (or a network with no
      // mDNS/Bonjour responders).
      warn_(pal, checks, "Local network",
            "gateway reachable, but LAN devices are not");
      hint_(pal, checks, "macOS gates local-network access, and the gateway is "
                         "EXEMPT -- so it");
      hint_(pal, checks, "can pass while LAN hosts (cameras / RTSP at "
                         "192.168.x.y:554, ONVIF");
      hint_(pal, checks, "discovery, mDNS) are blocked. Approve access in "
                         "System Settings >");
      hint_(pal, checks, "Privacy & Security > Local Network for your terminal "
                         "app. Note: under");
      hint_(pal, checks, "tmux/screen/ssh the CONTROLLING process is what's "
                         "gated -- launch from");
      hint_(pal, checks, "a granted terminal (or grant that helper). A quiet "
                         "network with no");
      hint_(pal, checks, "mDNS responders can also land here.");
    } else {
      warn_(pal, checks, "Local network", "no network route detected");
      hint_(pal, checks, "Neither the gateway nor LAN multicast responded -- "
                         "check connectivity.");
    }
  }

  // Bail out promptly if a shutdown was requested while the (only
  // blocking) network probe ran -- the rest is fast but pointless noise.
  if (abort != nullptr && abort->load()) {
    std::printf("\n");
    std::fflush(stdout);
    return checks;
  }

  // 2.2 Full Disk Access.
  {
    const int fda = probe_full_disk_access_();
    if (fda == 1) {
      ok_(pal, checks, "Full Disk Access", "granted");
    } else if (fda == -1) {
      // Either not macOS, or no protected file present to probe.
      ok_(pal, checks, "Full Disk Access", "not checked");
    } else {
      warn_(pal, checks, "Full Disk Access", "not granted");
      hint_(pal, checks, "Protected folders may be unreadable. Grant access in "
                         "System Settings >");
      hint_(pal, checks, "Privacy & Security > Full Disk Access (add your "
                         "terminal app), then restart.");
    }
  }

  // 2.3 Microphone.
  {
    const int mic = microphone_auth_status();
    if (mic == 1) {
      ok_(pal, checks, "Microphone", "authorized");
    } else if (mic == -1) {
      warn_(pal, checks, "Microphone", "permission not yet granted");
      hint_(pal, checks, "Audio capture will prompt or fail until approved in "
                         "System Settings >");
      hint_(pal, checks, "Privacy & Security > Microphone.");
    } else {
      warn_(pal, checks, "Microphone", "denied");
      hint_(pal, checks, "Audio capture will not work. Enable it in System "
                         "Settings >");
      hint_(pal, checks, "Privacy & Security > Microphone.");
    }
  }

  // Audio input device presence (hardware -- not a permission).
  {
    const int n = audio_input_device_count();
    if (n > 0) {
      ok_(pal, checks, "Audio input",
          std::to_string(n) + " input device(s) present");
    } else if (n == 0) {
      warn_(pal, checks, "Audio input", "no audio input device found");
      hint_(pal, checks, "Audio-capture stages will have nothing to record. "
                         "Connect a microphone");
      hint_(pal, checks, "or audio interface.");
    } else {
      ok_(pal, checks, "Audio input", "not checked");
    }
  }

  std::printf("\n");
  std::fflush(stdout);
  return checks;
}

}  // namespace vpipe::webui
