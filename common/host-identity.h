#ifndef HOST_IDENTITY_H
#define HOST_IDENTITY_H

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <cstddef>

namespace vpipe {

// 16-byte host machine UUID. On APPLE returns the value from
// gethostuuid() with a 5-second deadline; on failure returns all-zero
// bytes and emits a warning to stderr (callers may still proceed --
// the at-rest cipher key just becomes weaker, not invalid).
std::array<unsigned char, 16>
host_uuid_bytes();

// POSIX getuid() as uint32_t. Stable for the lifetime of the process.
uint32_t
os_user_id();

// Canonical input buffer for the KDF used by credential-cipher:
//   uuid(16) || ':' || decimal-ascii(uid)
//
// Exposed so tests can assert the exact byte layout and so callers
// can derive matching keys without re-implementing the encoding.
std::string
build_kdf_input(std::span<const unsigned char, 16> uuid, uint32_t uid);

}

#endif
