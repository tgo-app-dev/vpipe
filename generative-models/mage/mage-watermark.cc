#include "generative-models/mage/mage-watermark.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>

namespace vpipe {
namespace genai {
namespace mage_wm {

namespace {

// ---------------------------------------------------------------------------
// numpy SeedSequence (numpy/random/bit_generator.pyx)
// ---------------------------------------------------------------------------
constexpr std::uint32_t kXShift    = 16;
constexpr std::uint32_t kPoolSize  = 4;
constexpr std::uint32_t kInitA     = 0x43b0d7e5u;
constexpr std::uint32_t kMultA     = 0x931e8875u;
constexpr std::uint32_t kInitB     = 0x8b51f9ddu;
constexpr std::uint32_t kMultB     = 0x58f38dedu;
constexpr std::uint32_t kMixMultL  = 0xca01f9ddu;
constexpr std::uint32_t kMixMultR  = 0x4973f715u;

class SeedSequence {
 public:
  explicit SeedSequence(const std::vector<std::uint32_t>& entropy)
  {
    // mix_entropy: hash each pool slot, then cross-mix every ordered pair,
    // then fold in any entropy beyond the pool.
    std::uint32_t hc = kInitA;
    auto hash = [&](std::uint32_t v) {
      v ^= hc;
      hc *= kMultA;
      v *= hc;
      v ^= v >> kXShift;
      return v;
    };
    auto mix = [](std::uint32_t x, std::uint32_t y) {
      std::uint32_t r = kMixMultL * x - kMixMultR * y;
      r ^= r >> kXShift;
      return r;
    };
    _pool.assign(kPoolSize, 0);
    for (std::uint32_t i = 0; i < kPoolSize; ++i) {
      _pool[i] = hash(i < entropy.size() ? entropy[i] : 0u);
    }
    for (std::uint32_t s = 0; s < kPoolSize; ++s) {
      for (std::uint32_t d = 0; d < kPoolSize; ++d) {
        if (s != d) { _pool[d] = mix(_pool[d], hash(_pool[s])); }
      }
    }
    for (std::size_t s = kPoolSize; s < entropy.size(); ++s) {
      for (std::uint32_t d = 0; d < kPoolSize; ++d) {
        _pool[d] = mix(_pool[d], hash(entropy[s]));
      }
    }
  }

  std::vector<std::uint32_t> generate_state(std::size_t n) const
  {
    std::vector<std::uint32_t> out(n);
    std::uint32_t hc = kInitB;
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t v = _pool[i % kPoolSize];
      v ^= hc;
      hc *= kMultB;
      v *= hc;
      v ^= v >> kXShift;
      out[i] = v;
    }
    return out;
  }

 private:
  std::vector<std::uint32_t> _pool;
};

// ---------------------------------------------------------------------------
// PCG64 XSL-RR 128/64, seeded the way numpy's PCG64 bit generator is
// ---------------------------------------------------------------------------
class Pcg64 {
 public:
  explicit Pcg64(const SeedSequence& ss)
  {
    // numpy takes 4 uint64 of state; generate_state(4, uint64) is 8 uint32
    // words paired little-endian (lo | hi << 32).
    const std::vector<std::uint32_t> w = ss.generate_state(8);
    auto u64 = [&](std::size_t i) {
      return (std::uint64_t)w[2 * i] | ((std::uint64_t)w[2 * i + 1] << 32);
    };
    const u128 initstate = ((u128)u64(0) << 64) | u64(1);
    const u128 initseq   = ((u128)u64(2) << 64) | u64(3);
    _state = 0;
    _inc = (initseq << 1) | 1;
    step();
    _state += initstate;
    step();
  }

  std::uint64_t next_u64()
  {
    step();
    // XSL-RR: fold the 128-bit state to 64 and rotate by the top 6 bits.
    const std::uint64_t xored =
        (std::uint64_t)((_state >> 64) ^ (_state & ~(u128)0));
    const unsigned rot = (unsigned)(_state >> 122);
    return rot ? ((xored >> rot) | (xored << (64 - rot))) : xored;
  }

  // numpy buffers a 64-bit draw as two 32-bit ones, low half first.
  std::uint32_t next_u32()
  {
    if (_has_u32) { _has_u32 = false; return _u32; }
    const std::uint64_t n = next_u64();
    _has_u32 = true;
    _u32 = (std::uint32_t)(n >> 32);
    return (std::uint32_t)(n & 0xffffffffu);
  }

 private:
  using u128 = __uint128_t;
  static constexpr u128 kMult =
      (((u128)0x2360ED051FC65DA4ull) << 64) | 0x4385DF649FCCF645ull;
  void step() { _state = _state * kMult + _inc; }

  u128 _state = 0, _inc = 0;
  bool _has_u32 = false;
  std::uint32_t _u32 = 0;
};

// numpy Generator.integers(0, bound) for bound-1 <= UINT32_MAX: Lemire's
// bounded generation over next_uint32 (use_masked=False).
std::uint32_t
lemire_bounded(Pcg64& g, std::uint32_t rng_max)   // inclusive max
{
  if (rng_max == 0) { return 0; }
  const std::uint32_t rng_excl = rng_max + 1;
  std::uint64_t m = (std::uint64_t)g.next_u32() * rng_excl;
  std::uint32_t leftover = (std::uint32_t)(m & 0xffffffffu);
  if (leftover < rng_excl) {
    const std::uint32_t threshold =
        (std::uint32_t)((0xffffffffu - rng_max) % rng_excl);
    while (leftover < threshold) {
      m = (std::uint64_t)g.next_u32() * rng_excl;
      leftover = (std::uint32_t)(m & 0xffffffffu);
    }
  }
  return (std::uint32_t)(m >> 32);
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
void
sha256(const std::string& in, unsigned char out[CC_SHA256_DIGEST_LENGTH])
{
  CC_SHA256(in.data(), (CC_LONG)in.size(), out);
}

// numpy _int_to_uint32_array: little-endian 32-bit words, [0] for zero.
std::vector<std::uint32_t>
words_from_decimal(const std::string& digits)
{
  std::vector<std::uint32_t> w;   // little-endian base 2^32
  for (const char c : digits) {
    if (c < '0' || c > '9') { continue; }
    std::uint64_t carry = (std::uint64_t)(c - '0');
    for (std::size_t i = 0; i < w.size(); ++i) {
      const std::uint64_t cur = (std::uint64_t)w[i] * 10ull + carry;
      w[i] = (std::uint32_t)(cur & 0xffffffffu);
      carry = cur >> 32;
    }
    while (carry != 0) {
      w.push_back((std::uint32_t)(carry & 0xffffffffu));
      carry >>= 32;
    }
  }
  while (w.size() > 1 && w.back() == 0) { w.pop_back(); }
  if (w.empty()) { w.push_back(0); }
  return w;
}

bool
is_decimal(const std::string& s)
{
  std::size_t i = (!s.empty() && s[0] == '-') ? 1 : 0;
  if (i >= s.size()) { return false; }
  for (; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') { return false; }
  }
  return true;
}

// Acklam's inverse normal CDF (|rel err| < 1.2e-9) -- only the magnitude
// distribution rides on this; the SIGN comes from the half-bit.
double
ndtri(double p)
{
  static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                              -2.759285104469687e+02, 1.383577518672690e+02,
                              -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                              -1.556989798598866e+02, 6.680131188771972e+01,
                              -1.328068155288572e+01};
  static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                              -2.400758277161838e+00, -2.549732539343734e+00,
                              4.374664141464968e+00, 2.938163982698783e+00};
  static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                              2.445134137142996e+00, 3.754408661907416e+00};
  const double pl = 0.02425;
  if (p < pl) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
           / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  if (p > 1.0 - pl) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
           / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  const double q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
         * q
         / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

}  // namespace

std::string
resolve_key(const std::string& explicit_key)
{
  if (!explicit_key.empty()) { return explicit_key; }
  if (const char* e = std::getenv("MAGEFLOW_GS_KEY")) {
    if (*e != '\0') { return e; }
  }
  std::string kf;
  if (const char* f = std::getenv("MAGEFLOW_GS_KEY_FILE")) { kf = f; }
  if (kf.empty()) {
    if (const char* home = std::getenv("HOME")) {
      kf = std::string(home) + "/.mageflow/gs_key";
    }
  }
  if (!kf.empty()) {
    std::ifstream in(kf);
    if (in) {
      std::string line;
      std::getline(in, line);
      const auto ns = line.find_first_not_of(" \t\n\r");
      const auto ne = line.find_last_not_of(" \t\n\r");
      if (ns != std::string::npos) { return line.substr(ns, ne - ns + 1); }
    }
  }
  return kDefaultKey;
}

std::vector<std::uint32_t>
key_to_entropy(const std::string& key)
{
  std::string s = key;
  // trim
  const auto ns = s.find_first_not_of(" \t\n\r");
  const auto ne = s.find_last_not_of(" \t\n\r");
  s = (ns == std::string::npos) ? "" : s.substr(ns, ne - ns + 1);
  if (s.empty()) { s = kDefaultKey; }
  if (is_decimal(s)) {
    return words_from_decimal(s[0] == '-' ? s.substr(1) : s);   // abs()
  }
  // Passphrase: SHA-256 digest read BIG-endian as one 256-bit integer, then
  // split into little-endian 32-bit words.
  unsigned char dg[CC_SHA256_DIGEST_LENGTH];
  sha256(s, dg);
  std::vector<std::uint32_t> w(8);
  for (int i = 0; i < 8; ++i) {
    // word i (little-endian) is the i-th least significant 4 bytes, i.e. the
    // digest read from the tail.
    const int off = CC_SHA256_DIGEST_LENGTH - 4 * (i + 1);
    w[(std::size_t)i] = ((std::uint32_t)dg[off] << 24) |
                        ((std::uint32_t)dg[off + 1] << 16) |
                        ((std::uint32_t)dg[off + 2] << 8) |
                        (std::uint32_t)dg[off + 3];
  }
  while (w.size() > 1 && w.back() == 0) { w.pop_back(); }
  return w;
}

std::vector<std::uint8_t>
payload_bits(const std::string& payload)
{
  std::vector<std::uint8_t> out;
  out.reserve((std::size_t)kMsgBits);
  int counter = 0;
  while ((int)out.size() < kMsgBits) {
    unsigned char dg[CC_SHA256_DIGEST_LENGTH];
    sha256(payload + ":" + std::to_string(counter), dg);
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
      for (int k = 0; k < 8; ++k) {   // LSB-first within each byte
        out.push_back((std::uint8_t)((dg[i] >> k) & 1));
      }
    }
    ++counter;
  }
  out.resize((std::size_t)kMsgBits);
  return out;
}

void
pad_and_pos(std::size_t n, const std::string& key,
            std::vector<std::uint8_t>& pad, std::vector<std::uint16_t>& pos)
{
  // ONE generator, drawn in the reference's order: the whole pad array first,
  // then the whole pos array.
  Pcg64 g{SeedSequence(key_to_entropy(key))};
  pad.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    pad[i] = (std::uint8_t)lemire_bounded(g, 1);
  }
  pos.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    pos[i] = (std::uint16_t)lemire_bounded(g, (std::uint32_t)(kMsgBits - 1));
  }
}

std::vector<float>
encode_noise(int C, int H, int W, const std::string& key, std::uint64_t seed,
             const std::string& payload)
{
  if (C <= 0 || H <= 0 || W <= 0) { return {}; }
  const std::size_t hw = (std::size_t)H * W;
  const std::size_t n = (std::size_t)C * hw;
  const std::vector<std::uint8_t> msg = payload_bits(payload);
  std::vector<std::uint8_t> pad;
  std::vector<std::uint16_t> pos;
  pad_and_pos(n, key, pad, pos);

  // Magnitudes: vpipe's own generator (the reference's torch RNG cannot change
  // any SIGN, so reproducing it would buy nothing -- see the header).
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> ud(0.0, 1.0);

  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double half = (double)(msg[pos[i]] ^ pad[i]);
    double arg = (half + ud(rng)) * 0.5;
    arg = std::min(std::max(arg, 1e-6), 1.0 - 1e-6);
    const float z = (float)ndtri(arg);
    // channel-first index i = c*hw + t  ->  token-major [t, C]
    const std::size_t c = i / hw, t = i % hw;
    out[t * (std::size_t)C + c] = z;
  }
  return out;
}

double
raw_accuracy(const std::vector<float>& z, const std::string& key,
             const std::string& payload)
{
  if (z.empty()) { return 0.0; }
  const std::vector<std::uint8_t> msg = payload_bits(payload);
  std::vector<std::uint8_t> pad;
  std::vector<std::uint16_t> pos;
  pad_and_pos(z.size(), key, pad, pos);
  std::size_t hits = 0;
  for (std::size_t i = 0; i < z.size(); ++i) {
    const std::uint8_t observed = z[i] > 0.0f ? 1u : 0u;
    if (observed == (std::uint8_t)(msg[pos[i]] ^ pad[i])) { ++hits; }
  }
  return (double)hits / (double)z.size();
}

}  // namespace mage_wm
}  // namespace genai
}  // namespace vpipe
