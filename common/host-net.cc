#include "common/host-net.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <net/route.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace vpipe::netx {

UdpSocket::UdpSocket(UdpSocket&& o) noexcept : _fd(o._fd)
{
  o._fd = -1;
}

UdpSocket&
UdpSocket::operator=(UdpSocket&& o) noexcept
{
  if (this != &o) {
    if (_fd >= 0) { ::close(_fd); }
    _fd = o._fd;
    o._fd = -1;
  }
  return *this;
}

UdpSocket::~UdpSocket()
{
  if (_fd >= 0) { ::close(_fd); }
}

bool
UdpSocket::bind_ephemeral()
{
  _fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (_fd < 0) {
    return false;
  }
  sockaddr_in addr{};
  addr.sin_family      = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port        = 0;
  if (::bind(_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(_fd);
    _fd = -1;
    return false;
  }
  return true;
}

bool
UdpSocket::enable_multicast_loop()
{
  unsigned char v = 1;
  return ::setsockopt(_fd, IPPROTO_IP, IP_MULTICAST_LOOP, &v, sizeof(v))
         == 0;
}

bool
UdpSocket::set_multicast_ttl(int ttl)
{
  unsigned char v = static_cast<unsigned char>(ttl);
  return ::setsockopt(_fd, IPPROTO_IP, IP_MULTICAST_TTL, &v, sizeof(v))
         == 0;
}

bool
UdpSocket::set_recv_timeout(std::chrono::milliseconds d)
{
  timeval tv{};
  tv.tv_sec  = static_cast<time_t>(d.count() / 1000);
  tv.tv_usec = static_cast<suseconds_t>((d.count() % 1000) * 1000);
  return ::setsockopt(_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
}

bool
UdpSocket::send_to(std::string_view payload,
                   const char*      ip,
                   uint16_t         port)
{
  sockaddr_in dst{};
  dst.sin_family = AF_INET;
  dst.sin_port   = htons(port);
  if (::inet_pton(AF_INET, ip, &dst.sin_addr) != 1) {
    return false;
  }
  ssize_t n = ::sendto(_fd,
                       payload.data(), payload.size(), 0,
                       reinterpret_cast<sockaddr*>(&dst), sizeof(dst));
  return n == static_cast<ssize_t>(payload.size());
}

std::optional<std::pair<std::string, std::string>>
UdpSocket::recv_one()
{
  // 64 KiB is the IPv4 datagram ceiling; ONVIF probe matches are
  // typically a few hundred bytes.
  char        buf[65536];
  sockaddr_in src{};
  socklen_t   src_len = sizeof(src);
  ssize_t     n = ::recvfrom(_fd, buf, sizeof(buf), 0,
                             reinterpret_cast<sockaddr*>(&src),
                             &src_len);
  if (n <= 0) {
    return std::nullopt;
  }
  char ipbuf[INET_ADDRSTRLEN] = {0};
  ::inet_ntop(AF_INET, &src.sin_addr, ipbuf, sizeof(ipbuf));
  char addr_str[INET_ADDRSTRLEN + 8];
  std::snprintf(addr_str, sizeof(addr_str), "%s:%u",
                ipbuf, static_cast<unsigned>(ntohs(src.sin_port)));
  return std::make_pair(std::string(buf, static_cast<size_t>(n)),
                        std::string(addr_str));
}

namespace {

bool
write_all(int fd, const char* data, size_t len)
{
  while (len > 0) {
    ssize_t n = ::send(fd, data, len, 0);
    if (n < 0) {
      if (errno == EINTR) { continue; }
      return false;
    }
    data += n;
    len  -= static_cast<size_t>(n);
  }
  return true;
}

}

std::vector<std::string>
local_ipv4_addresses()
{
  std::vector<std::string> out;
  ifaddrs* ifa = nullptr;
  if (::getifaddrs(&ifa) != 0 || !ifa) {
    return out;
  }
  for (ifaddrs* p = ifa; p != nullptr; p = p->ifa_next) {
    if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) {
      continue;
    }
    if ((p->ifa_flags & IFF_LOOPBACK) || !(p->ifa_flags & IFF_UP)) {
      continue;
    }
    auto* sin = reinterpret_cast<sockaddr_in*>(p->ifa_addr);
    char  buf[INET_ADDRSTRLEN] = {0};
    ::inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf));
    if (buf[0] != 0) {
      out.emplace_back(buf);
    }
  }
  ::freeifaddrs(ifa);
  return out;
}

void
warm_arp_cache(const std::vector<std::string>& ips)
{
  if (ips.empty()) { return; }
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) { return; }
  // Non-blocking so a single bad ICMP-unreachable reply can't stall
  // the loop.
  int flags = ::fcntl(fd, F_GETFL, 0);
  ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  for (const auto& ip : ips) {
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port   = htons(9);  // discard port; we don't care
    if (::inet_pton(AF_INET, ip.c_str(), &dst.sin_addr) != 1) {
      continue;
    }
    char b = 0;
    // Errors are intentionally ignored. We only need the kernel to
    // emit an ARP request; whether the datagram is dropped at the
    // discard port doesn't matter.
    (void)::sendto(fd, &b, 1, 0,
                   reinterpret_cast<sockaddr*>(&dst), sizeof(dst));
  }
  ::close(fd);
}

std::vector<std::string>
arp_cache_ipv4()
{
  std::vector<std::string> out;
  // sysctl(NET_RT_FLAGS, RTF_LLINFO) returns every routing-table
  // entry that has a link-layer address attached -- i.e. the ARP
  // table. macOS supports this with no special privileges.
  int    mib[] = { CTL_NET, PF_ROUTE, 0, AF_INET,
                   NET_RT_FLAGS, RTF_LLINFO };
  size_t needed = 0;
  if (::sysctl(mib, 6, nullptr, &needed, nullptr, 0) < 0
      || needed == 0) {
    return out;
  }
  std::vector<char> buf(needed);
  if (::sysctl(mib, 6, buf.data(), &needed, nullptr, 0) < 0) {
    return out;
  }
  const char* end = buf.data() + needed;
  for (char* next = buf.data(); next < end; ) {
    auto* rtm = reinterpret_cast<rt_msghdr*>(next);
    if (rtm->rtm_msglen == 0
        || next + rtm->rtm_msglen > end) {
      break;
    }
    next += rtm->rtm_msglen;
    auto* sin = reinterpret_cast<sockaddr_in*>(rtm + 1);
    if (sin->sin_family != AF_INET) { continue; }
    auto* sdl = reinterpret_cast<sockaddr_dl*>(
        reinterpret_cast<char*>(sin) + sin->sin_len);
    if (sdl->sdl_alen == 0) {
      // Pending / incomplete entry -- ARP request sent but no reply
      // yet. Skip; a future read may pick it up.
      continue;
    }
    char ipbuf[INET_ADDRSTRLEN] = {0};
    ::inet_ntop(AF_INET, &sin->sin_addr, ipbuf, sizeof(ipbuf));
    if (ipbuf[0] != 0) {
      out.emplace_back(ipbuf);
    }
  }
  return out;
}

std::string
tcp_request(std::string_view          host,
            uint16_t                  port,
            std::string_view          request,
            std::chrono::milliseconds timeout)
{
  // Resolve host.
  addrinfo  hints{};
  hints.ai_family   = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  std::string host_str(host);
  char port_str[8];
  std::snprintf(port_str, sizeof(port_str), "%u",
                static_cast<unsigned>(port));

  addrinfo* res = nullptr;
  if (::getaddrinfo(host_str.c_str(), port_str, &hints, &res) != 0) {
    return {};
  }

  int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd < 0) {
    ::freeaddrinfo(res);
    return {};
  }

  // Non-blocking connect with select() for the timeout, then put
  // back to blocking for read/write (simpler than non-blocking I/O
  // for our small request/response sizes).
  int flags = ::fcntl(fd, F_GETFL, 0);
  ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);

  int rc = ::connect(fd, res->ai_addr, res->ai_addrlen);
  if (rc < 0 && errno != EINPROGRESS) {
    ::close(fd);
    ::freeaddrinfo(res);
    return {};
  }
  if (rc < 0) {
    fd_set wset;
    FD_ZERO(&wset);
    FD_SET(fd, &wset);
    timeval tv{};
    tv.tv_sec  = static_cast<time_t>(timeout.count() / 1000);
    tv.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
    int sr = ::select(fd + 1, nullptr, &wset, nullptr, &tv);
    if (sr <= 0) {
      ::close(fd);
      ::freeaddrinfo(res);
      return {};
    }
    int err = 0;
    socklen_t elen = sizeof(err);
    ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &elen);
    if (err != 0) {
      ::close(fd);
      ::freeaddrinfo(res);
      return {};
    }
  }
  ::freeaddrinfo(res);
  ::fcntl(fd, F_SETFL, flags);  // back to blocking

  // Apply read timeout via SO_RCVTIMEO.
  timeval tv{};
  tv.tv_sec  = static_cast<time_t>(timeout.count() / 1000);
  tv.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  if (!write_all(fd, request.data(), request.size())) {
    ::close(fd);
    return {};
  }

  std::string out;
  out.reserve(4096);
  char buf[4096];
  for (;;) {
    ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n > 0) {
      out.append(buf, static_cast<size_t>(n));
      continue;
    }
    if (n == 0) {
      break;  // peer closed
    }
    if (errno == EINTR) { continue; }
    break;
  }
  ::close(fd);
  return out;
}

}
