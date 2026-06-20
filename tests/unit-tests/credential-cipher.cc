#include "minitest.h"
#include "common/credential-cipher.h"
#include "common/host-identity.h"

#include <array>
#include <cstring>
#include <span>
#include <string>

using namespace vpipe;

namespace {

std::array<unsigned char, 32>
fixed_key()
{
  // Derive a deterministic 32B key from a fixed input. Tests don't
  // depend on the bytes -- only on round-trip + tampering behavior.
  return derive_key(std::string("vpipe.test.fixed.input.v1"));
}

}

TEST(credential_cipher, derive_key_is_deterministic) {
  auto k1 = fixed_key();
  auto k2 = fixed_key();
  EXPECT_TRUE(std::memcmp(k1.data(), k2.data(), 32) == 0);
  // ...and not all-zero (HMAC of a real input).
  bool any_nz = false;
  for (auto b : k1) { if (b) { any_nz = true; break; } }
  EXPECT_TRUE(any_nz);
}

TEST(credential_cipher, aes_cbc_hmac_round_trip) {
  auto key = fixed_key();
  std::string pt = "hunter2 the quick brown fox jumps over 13 lazy dogs";
  auto sealed = seal(std::span<const unsigned char, 32>(key), pt);
  EXPECT_TRUE(!sealed.empty());
  auto opened = open(std::span<const unsigned char, 32>(key), sealed);
  EXPECT_TRUE(opened.has_value());
  EXPECT_TRUE(*opened == pt);
}

TEST(credential_cipher, empty_plaintext_round_trips) {
  auto key = fixed_key();
  auto sealed = seal(std::span<const unsigned char, 32>(key), "");
  EXPECT_TRUE(!sealed.empty());
  auto opened = open(std::span<const unsigned char, 32>(key), sealed);
  EXPECT_TRUE(opened.has_value());
  EXPECT_TRUE(opened->empty());
}

TEST(credential_cipher, tampering_rejected) {
  auto key = fixed_key();
  auto sealed = seal(std::span<const unsigned char, 32>(key),
                     "secret-string");
  EXPECT_TRUE(sealed.size() > 16);
  // Flip one byte in the ciphertext region.
  sealed[20] = static_cast<char>(sealed[20] ^ 0x01);
  auto opened = open(std::span<const unsigned char, 32>(key), sealed);
  EXPECT_FALSE(opened.has_value());
}

TEST(credential_cipher, wrong_key_rejected) {
  auto k1 = fixed_key();
  auto k2 = derive_key("a different input");
  auto sealed = seal(std::span<const unsigned char, 32>(k1),
                     "secret-string");
  auto opened = open(std::span<const unsigned char, 32>(k2), sealed);
  EXPECT_FALSE(opened.has_value());
}

TEST(credential_cipher, aad_must_match) {
  auto key = fixed_key();
  auto sealed = seal(std::span<const unsigned char, 32>(key),
                     "secret", "context-A");
  auto with_a = open(std::span<const unsigned char, 32>(key),
                     sealed, "context-A");
  auto with_b = open(std::span<const unsigned char, 32>(key),
                     sealed, "context-B");
  EXPECT_TRUE(with_a.has_value());
  EXPECT_TRUE(with_a && *with_a == "secret");
  EXPECT_FALSE(with_b.has_value());
}
