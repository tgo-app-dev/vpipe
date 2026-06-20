#include "common/host-identity.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <unistd.h>

namespace vpipe {

std::array<unsigned char, 16>
host_uuid_bytes()
{
  std::array<unsigned char, 16> out{};
#if defined(__APPLE__)
  // 5-second deadline. gethostuuid is normally instantaneous, but
  // can stall on XPC under locked-down macOS configurations.
  struct timespec wait{};
  wait.tv_sec  = 5;
  wait.tv_nsec = 0;
  uuid_t u{};
  if (gethostuuid(u, &wait) == 0) {
    std::memcpy(out.data(), u, 16);
  } else {
    std::fprintf(stderr,
        "vpipe: gethostuuid() failed; cipher key falls back to "
        "all-zero machine UUID\n");
  }
#else
  // Non-APPLE platforms are not supported by the at-rest cipher in
  // the first cut. Returning all-zero here matches the documented
  // contract; the credential-cipher module is not compiled on these
  // builds, so this code path is only reached if callers use the
  // header for unrelated purposes.
#endif
  return out;
}

uint32_t
os_user_id()
{
  return static_cast<uint32_t>(::getuid());
}

std::string
build_kdf_input(std::span<const unsigned char, 16> uuid, uint32_t uid)
{
  char dec[16];
  int n = std::snprintf(dec, sizeof(dec), "%u",
                        static_cast<unsigned>(uid));
  std::string out;
  out.reserve(16u + 1u + static_cast<size_t>(n));
  out.append(reinterpret_cast<const char*>(uuid.data()), 16);
  out.push_back(':');
  out.append(dec, static_cast<size_t>(n));
  return out;
}

}
