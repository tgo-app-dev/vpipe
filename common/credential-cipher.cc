#include "common/credential-cipher.h"

#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonDigest.h>
#include <CommonCrypto/CommonHMAC.h>
#include <CommonCrypto/CommonRandom.h>

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace vpipe {

namespace {

// Fixed salt + info strings. Changing either invalidates every blob
// previously sealed -- treat as part of the on-disk format version.
constexpr char kSalt[] = "vpipe.cameras.salt.v1";

constexpr size_t kHashLen = 32;
constexpr size_t kKeyLen  = 32;   // first 32 bytes of HKDF output
constexpr size_t kMacLen  = 32;   // last 32 bytes of HKDF output
constexpr size_t kIvLen   = 16;
constexpr size_t kTagLen  = 32;   // HMAC-SHA256

// HKDF-SHA256 with variable output length L. We only use L = 64 here
// (32B AES key + 32B HMAC key) which requires two HKDF-Expand blocks.
void
hkdf_sha256(const unsigned char* ikm,  size_t ikm_len,
            const unsigned char* salt, size_t salt_len,
            const unsigned char* info, size_t info_len,
            unsigned char*       out,  size_t out_len)
{
  unsigned char prk[kHashLen];
  CCHmac(kCCHmacAlgSHA256, salt, salt_len, ikm, ikm_len, prk);

  unsigned char t_prev[kHashLen];
  size_t        produced = 0;
  unsigned char block_idx = 0;
  while (produced < out_len) {
    ++block_idx;
    // Buf = T(i-1) || info || block_idx
    unsigned char buf[kHashLen + 256];
    size_t bp = 0;
    if (block_idx > 1) {
      std::memcpy(buf, t_prev, kHashLen);
      bp += kHashLen;
    }
    if (info_len > sizeof(buf) - bp - 1) { info_len = 0; }
    std::memcpy(buf + bp, info, info_len);
    bp += info_len;
    buf[bp++] = block_idx;

    CCHmac(kCCHmacAlgSHA256, prk, kHashLen, buf, bp, t_prev);

    size_t take = (out_len - produced < kHashLen)
                      ? (out_len - produced) : kHashLen;
    std::memcpy(out + produced, t_prev, take);
    produced += take;
  }
}

bool
fill_random(unsigned char* buf, size_t n)
{
  if (CCRandomGenerateBytes(buf, n) == kCCSuccess) {
    return true;
  }
  int fd = ::open("/dev/urandom", O_RDONLY);
  if (fd < 0) { return false; }
  ssize_t got = ::read(fd, buf, n);
  ::close(fd);
  return got == static_cast<ssize_t>(n);
}

}

std::array<unsigned char, 32>
derive_key(std::string_view kdf_input)
{
  // We expose a 32-byte API key, but internally we expand 64 bytes
  // (AES + HMAC). Re-derived from `kdf_input` in seal/open so the
  // header API stays narrow.
  std::array<unsigned char, 32> key{};
  // Use HKDF-Extract output directly here so this function's return
  // is well-defined as "the 32-byte master key bound to kdf_input".
  unsigned char prk[kHashLen];
  CCHmac(kCCHmacAlgSHA256,
         kSalt, sizeof(kSalt) - 1,
         reinterpret_cast<const unsigned char*>(kdf_input.data()),
         kdf_input.size(),
         prk);
  std::memcpy(key.data(), prk, kHashLen);
  std::memset(prk, 0, sizeof(prk));
  return key;
}

namespace {

// Expand a 32B master key to (32B AES, 32B HMAC) using HKDF-Expand
// with a fixed info that includes the AAD length (binding the AAD
// into the keystream is overkill but free).
void
expand_subkeys(std::span<const unsigned char, 32> master,
               std::string_view                   aad,
               unsigned char                      aes_key[kKeyLen],
               unsigned char                      mac_key[kMacLen])
{
  // info = "vpipe.cameras.aes256cbc.v1" || aad
  static constexpr char kInfoTag[] = "vpipe.cameras.aes256cbc.v1";
  unsigned char info[sizeof(kInfoTag) - 1 + 256];
  size_t        info_len = 0;
  std::memcpy(info, kInfoTag, sizeof(kInfoTag) - 1);
  info_len += sizeof(kInfoTag) - 1;
  size_t take_aad = aad.size() <= 256 ? aad.size() : 256;
  std::memcpy(info + info_len, aad.data(), take_aad);
  info_len += take_aad;

  unsigned char okm[kKeyLen + kMacLen];
  hkdf_sha256(master.data(), master.size(),
              reinterpret_cast<const unsigned char*>(kSalt),
              sizeof(kSalt) - 1,
              info, info_len,
              okm, sizeof(okm));
  std::memcpy(aes_key, okm,            kKeyLen);
  std::memcpy(mac_key, okm + kKeyLen,  kMacLen);
  std::memset(okm, 0, sizeof(okm));
}

}

std::string
seal(std::span<const unsigned char, 32> key,
     std::string_view                   plaintext,
     std::string_view                   aad)
{
  unsigned char aes_key[kKeyLen];
  unsigned char mac_key[kMacLen];
  expand_subkeys(key, aad, aes_key, mac_key);

  unsigned char iv[kIvLen];
  if (!fill_random(iv, kIvLen)) {
    std::memset(aes_key, 0, sizeof(aes_key));
    std::memset(mac_key, 0, sizeof(mac_key));
    return {};
  }

  // AES-CBC with PKCS#7 padding. The ciphertext buffer must be at
  // least pt + 16 bytes.
  size_t ct_capacity = plaintext.size() + kCCBlockSizeAES128;
  std::string ct;
  ct.resize(ct_capacity);
  size_t          out_len = 0;
  CCCryptorStatus rc      = CCCrypt(
      kCCEncrypt, kCCAlgorithmAES, kCCOptionPKCS7Padding,
      aes_key, kKeyLen,
      iv,
      plaintext.empty() ? nullptr : plaintext.data(),
      plaintext.size(),
      ct.data(), ct.size(),
      &out_len);
  std::memset(aes_key, 0, sizeof(aes_key));
  if (rc != kCCSuccess) {
    std::memset(mac_key, 0, sizeof(mac_key));
    return {};
  }
  ct.resize(out_len);

  // tag = HMAC-SHA256(mac_key, iv || ct)
  unsigned char tag[kTagLen];
  CCHmacContext hctx;
  CCHmacInit(&hctx, kCCHmacAlgSHA256, mac_key, kMacLen);
  CCHmacUpdate(&hctx, iv, kIvLen);
  CCHmacUpdate(&hctx, ct.data(), ct.size());
  CCHmacFinal(&hctx, tag);
  std::memset(mac_key, 0, sizeof(mac_key));

  std::string out;
  out.reserve(kIvLen + ct.size() + kTagLen);
  out.append(reinterpret_cast<const char*>(iv), kIvLen);
  out.append(ct);
  out.append(reinterpret_cast<const char*>(tag), kTagLen);
  return out;
}

std::optional<std::string>
open(std::span<const unsigned char, 32> key,
     std::string_view                   sealed,
     std::string_view                   aad)
{
  if (sealed.size() < kIvLen + kTagLen) {
    return std::nullopt;
  }
  const size_t ct_len = sealed.size() - kIvLen - kTagLen;
  const unsigned char* p =
      reinterpret_cast<const unsigned char*>(sealed.data());
  const unsigned char* iv  = p;
  const unsigned char* ct  = p + kIvLen;
  const unsigned char* tag = p + kIvLen + ct_len;

  unsigned char aes_key[kKeyLen];
  unsigned char mac_key[kMacLen];
  expand_subkeys(key, aad, aes_key, mac_key);

  // Verify HMAC first; constant-time compare to avoid timing side
  // channels on the tag bytes.
  unsigned char expected[kTagLen];
  CCHmacContext hctx;
  CCHmacInit(&hctx, kCCHmacAlgSHA256, mac_key, kMacLen);
  CCHmacUpdate(&hctx, iv, kIvLen);
  CCHmacUpdate(&hctx, ct, ct_len);
  CCHmacFinal(&hctx, expected);
  std::memset(mac_key, 0, sizeof(mac_key));

  unsigned char diff = 0;
  for (size_t i = 0; i < kTagLen; ++i) {
    diff = static_cast<unsigned char>(diff | (expected[i] ^ tag[i]));
  }
  if (diff != 0) {
    std::memset(aes_key, 0, sizeof(aes_key));
    return std::nullopt;
  }

  std::string pt;
  pt.resize(ct_len);
  size_t          out_len = 0;
  CCCryptorStatus rc      = CCCrypt(
      kCCDecrypt, kCCAlgorithmAES, kCCOptionPKCS7Padding,
      aes_key, kKeyLen,
      iv,
      ct, ct_len,
      pt.data(), pt.size(),
      &out_len);
  std::memset(aes_key, 0, sizeof(aes_key));
  if (rc != kCCSuccess) {
    return std::nullopt;
  }
  pt.resize(out_len);
  return pt;
}

}
