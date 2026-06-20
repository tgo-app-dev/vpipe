#ifndef CREDENTIAL_CIPHER_H
#define CREDENTIAL_CIPHER_H

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace vpipe {

// Symmetric at-rest cipher for ONVIF passwords and any other small
// secrets the session needs to persist. Keys are derived from
// host-identity inputs (machine UUID + OS uid); the construction is
// HKDF-SHA256 + AES-256-GCM. Nonces are random per seal() call;
// sealed-output layout is nonce(12) || ciphertext || tag(16).
//
// All symbols are APPLE-only in the first cut: the impl uses
// CommonCrypto + Security. The header itself is platform-agnostic so
// declarations remain valid for clients reasoning about the
// interface in non-APPLE TUs (which won't link if they call the
// functions).

// HKDF-SHA256 with fixed salt + info, single 32-byte output block.
// Suitable for deriving an AES-256 key from build_kdf_input(...).
std::array<unsigned char, 32>
derive_key(std::string_view kdf_input);

// AES-256-GCM seal. Returns nonce(12) || ciphertext || tag(16).
// AAD is bound into the GCM tag but not stored.
std::string
seal(std::span<const unsigned char, 32> key,
     std::string_view                   plaintext,
     std::string_view                   aad = {});

// AES-256-GCM open. Returns nullopt on tag mismatch (constant time
// inside CommonCrypto). AAD must match what was passed to seal().
std::optional<std::string>
open(std::span<const unsigned char, 32> key,
     std::string_view                   sealed,
     std::string_view                   aad = {});

}

#endif
