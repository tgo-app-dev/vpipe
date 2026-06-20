#include "minitest.h"
#include "common/host-identity.h"

#include <array>
#include <cstring>
#include <span>
#include <unistd.h>

using namespace vpipe;

TEST(host_identity, os_user_id_matches_getuid) {
  EXPECT_TRUE(os_user_id() == static_cast<uint32_t>(::getuid()));
}

TEST(host_identity, host_uuid_bytes_is_16_bytes) {
  auto u = host_uuid_bytes();
  EXPECT_TRUE(u.size() == 16);
  // On a real macOS host this is non-zero; if it happens to be all
  // zero (gethostuuid failed) we still want the array shape correct
  // for downstream KDF.
  (void)u;
}

TEST(host_identity, build_kdf_input_encoding) {
  std::array<unsigned char, 16> uuid{
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f };
  auto out = build_kdf_input(
      std::span<const unsigned char, 16>(uuid), 501u);
  // expected = uuid(16) || ':' || "501"
  EXPECT_TRUE(out.size() == 16u + 1u + 3u);
  EXPECT_TRUE(std::memcmp(out.data(), uuid.data(), 16) == 0);
  EXPECT_TRUE(out[16] == ':');
  EXPECT_TRUE(out.compare(17, 3, "501") == 0);
}
