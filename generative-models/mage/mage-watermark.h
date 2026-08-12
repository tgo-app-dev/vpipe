#ifndef GENERATIVE_MODELS_MAGE_MAGE_WATERMARK_H
#define GENERATIVE_MODELS_MAGE_MAGE_WATERMARK_H

// Gaussian-Shading provenance watermark for Mage-Flow
// (mage_flow/models/modules/mage_latent.py).
//
// The watermark lives in the INITIAL NOISE, not in the pixels: each latent
// entry is forced into the upper or lower half-plane of the normal
// distribution according to a key-seeded bit, then given a uniformly random
// magnitude within that half. The result is still exactly ~N(0,1), so it costs
// nothing in image quality, and it survives the denoise because the detector
// inverts the flow ODE back to the noise and reads the signs.
//
// BIT-COMPATIBILITY WITH MICROSOFT'S DETECTOR is the whole point, and it
// pins down exactly one thing: the per-entry XOR pad and message-index map,
// which `_pad_and_pos` draws from numpy's `default_rng(key)`. Those must match
// numpy bit for bit, so this file reproduces numpy's SeedSequence entropy mix,
// PCG64 (XSL-RR 128/64) and the Lemire bounded-integer path that
// `Generator.integers` uses. It is verified against a numpy dump in
// tests/unit-tests/mage-watermark.cc.
//
// What does NOT need reproducing: the magnitudes. `decode_bits` only looks at
// `z > 0`, and the sign is decided entirely by the target half-bit -- for
// half=1 the argument (1+u)/2 always exceeds 0.5, for half=0 it never does.
// So the reference's torch RNG is irrelevant to detection and we use vpipe's
// own generator for the magnitudes (still seed-reproducible), and the
// inverse-normal-CDF accuracy only shapes the magnitude distribution.

#include "common/flex-data.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace mage_wm {

// The published defaults (mage_latent.py).
inline constexpr const char* kDefaultPayload = "MageFlow";
inline constexpr const char* kDefaultKey     = "20260720";
inline constexpr int         kMsgBits        = 256;

// The reference's key precedence (mage_latent.resolve_gs_key): an explicit
// key, else $MAGEFLOW_GS_KEY, else the file named by $MAGEFLOW_GS_KEY_FILE or
// ~/.mageflow/gs_key, else the published default. Returned verbatim (still a
// string) so key_to_entropy can classify it as integer vs passphrase.
std::string resolve_key(const std::string& explicit_key);

// What a CALLER chooses about the watermark, parsed from the
// `mage-flow-model-config` beat. It lives with the watermark rather than
// in the driving stage because it is entirely this family's: no other
// model here puts provenance in its initial noise.
//
// `enabled` is a POSITIVE field over a negative config key
// (`no_watermark`), on purpose. The reference applies the watermark
// unconditionally and it is distribution-preserving, so the safe state
// is on and the config should have to say otherwise -- but code reading
// `if (!no_watermark)` at every use is how a double negative eventually
// gets read wrong.
struct Params {
  bool        enabled = true;
  // Empty means resolve_key()'s precedence chain decides. Held unresolved
  // so the environment is read when the noise is built, not when the
  // config beat happens to arrive.
  std::string key;

  static Params from_flex(const FlexData& fd, std::string* err = nullptr);
};

// numpy SeedSequence entropy for `key`: a pure (optionally signed) decimal
// string is the integer itself, taken as |n|; anything else is a passphrase
// hashed with SHA-256 into a 256-bit integer. Either way the result is the
// little-endian uint32 word array numpy's _int_to_uint32_array produces.
std::vector<std::uint32_t> key_to_entropy(const std::string& key);

// The key-seeded per-entry XOR pad (0/1) and message index (0..kMsgBits-1),
// i.e. `_pad_and_pos(n, key)`. Exposed for the compatibility test.
void pad_and_pos(std::size_t n, const std::string& key,
                 std::vector<std::uint8_t>& pad,
                 std::vector<std::uint16_t>& pos);

// `_payload_to_bits(payload)`: SHA-256("<payload>:<counter>") expanded
// LSB-first per byte, truncated to kMsgBits.
std::vector<std::uint8_t> payload_bits(const std::string& payload);

// Watermarked initial noise for a [C, H, W] latent, returned TOKEN-MAJOR
// ([H*W, C]) -- the layout the DiT consumes -- while the watermark itself is
// laid out channel-first, which is the order the detector reshapes into.
// `seed` drives only the magnitudes. Empty on a bad shape.
std::vector<float> encode_noise(int C, int H, int W, const std::string& key,
                                std::uint64_t seed,
                                const std::string& payload = kDefaultPayload);

// Detector side (for tests + a future verify tool): fraction of entries whose
// sign matches the expected half. ~1.0 on freshly encoded noise, ~0.5 on
// unwatermarked noise. `z` is channel-first [C*H*W].
double raw_accuracy(const std::vector<float>& z, const std::string& key,
                    const std::string& payload = kDefaultPayload);

}  // namespace mage_wm
}  // namespace genai
}  // namespace vpipe

#endif
