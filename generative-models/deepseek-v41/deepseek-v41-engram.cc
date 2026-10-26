#include "generative-models/deepseek-v41/deepseek-v41-engram.h"

#include "common/flex-data.h"
#include "generative-models/deepseek-v41/deepseek-v41-config.h"

#include <CoreFoundation/CoreFoundation.h>

#include <format>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>

namespace vpipe::genai {

namespace {

// ---- numpy's PCG64 and Generator.integers ------------------------------

using u128 = unsigned __int128;

// numpy.random.SeedSequence(seed).generate_state(n, uint32) for an integer
// seed and no spawn key (bit_generator.pyx: _coerce_to_uint32_array,
// mix_entropy, generate_state; pool size 4).
std::vector<std::uint32_t>
seed_sequence_state(std::uint64_t seed, int n_words)
{
  constexpr std::uint32_t kInitA = 0x43b0d7e5u, kMultA = 0x931e8875u;
  constexpr std::uint32_t kInitB = 0x8b51f9ddu, kMultB = 0x58f38dedu;
  constexpr std::uint32_t kMixL = 0xca01f9ddu, kMixR = 0x4973f715u;
  constexpr int kShift = 16;
  std::vector<std::uint32_t> entropy;
  do {
    entropy.push_back((std::uint32_t)(seed & 0xffffffffu));
    seed >>= 32;
  } while (seed != 0);
  std::uint32_t hash_const = kInitA;
  auto hashmix = [&](std::uint32_t v) {
    v ^= hash_const;
    hash_const *= kMultA;
    v *= hash_const;
    v ^= v >> kShift;
    return v;
  };
  auto mix = [](std::uint32_t x, std::uint32_t y) {
    std::uint32_t r = kMixL * x - kMixR * y;
    r ^= r >> kShift;
    return r;
  };
  std::uint32_t pool[4];
  for (int i = 0; i < 4; ++i) {
    pool[i] = hashmix(i < (int)entropy.size() ? entropy[(std::size_t)i] : 0);
  }
  for (int s = 0; s < 4; ++s) {
    for (int d = 0; d < 4; ++d) {
      if (s != d) { pool[d] = mix(pool[d], hashmix(pool[s])); }
    }
  }
  for (std::size_t s = 4; s < entropy.size(); ++s) {
    for (int d = 0; d < 4; ++d) {
      pool[d] = mix(pool[d], hashmix(entropy[s]));
    }
  }
  std::vector<std::uint32_t> out((std::size_t)n_words);
  std::uint32_t hb = kInitB;
  for (int i = 0; i < n_words; ++i) {
    std::uint32_t v = pool[i % 4];
    v ^= hb;
    hb *= kMultB;
    v *= hb;
    v ^= v >> kShift;
    out[(std::size_t)i] = v;
  }
  return out;
}

struct Pcg64 {
  u128 state = 0, inc = 0;
  static constexpr u128 kMult =
      ((u128)2549297995355413924ull << 64) | 4865540595714422341ull;
  void step() { state = state * kMult + inc; }
  std::uint64_t next()
  {
    step();
    const std::uint64_t x = (std::uint64_t)(state >> 64) ^
                            (std::uint64_t)state;
    const unsigned r = (unsigned)(state >> 122);
    return (x >> r) | (x << ((64 - r) & 63));
  }
  // numpy: PCG64(SeedSequence(seed)): generate_state(4, uint64) as
  // (seed hi, seed lo, inc hi, inc lo), then pcg64_srandom_r.
  explicit Pcg64(std::uint64_t seed)
  {
    const std::vector<std::uint32_t> w = seed_sequence_state(seed, 8);
    std::uint64_t v[4];
    for (int i = 0; i < 4; ++i) {
      v[i] = (std::uint64_t)w[(std::size_t)(2 * i)] |
             ((std::uint64_t)w[(std::size_t)(2 * i + 1)] << 32);
    }
    const u128 init = ((u128)v[0] << 64) | v[1];
    const u128 seq = ((u128)v[2] << 64) | v[3];
    state = 0;
    inc = (seq << 1) | 1;
    step();
    state += init;
    step();
  }
};

// ---- primes ----------------------------------------------------------

std::uint64_t
mulmod_(std::uint64_t a, std::uint64_t b, std::uint64_t m)
{
  return (std::uint64_t)((u128)a * b % m);
}

bool
is_prime_(std::uint64_t n)
{
  if (n < 2) { return false; }
  for (std::uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
                          23ull, 29ull, 31ull, 37ull}) {
    if (n % p == 0) { return n == p; }
  }
  std::uint64_t d = n - 1;
  int s = 0;
  while ((d & 1) == 0) { d >>= 1; ++s; }
  for (std::uint64_t a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull,
                          23ull, 29ull, 31ull, 37ull}) {
    std::uint64_t x = 1, b = a % n, e = d;
    while (e != 0) {
      if (e & 1) { x = mulmod_(x, b, n); }
      b = mulmod_(b, b, n);
      e >>= 1;
    }
    if (x == 1 || x == n - 1) { continue; }
    bool comp = true;
    for (int r = 1; r < s; ++r) {
      x = mulmod_(x, x, n);
      if (x == n - 1) { comp = false; break; }
    }
    if (comp) { return false; }
  }
  return true;
}

// ---- text ------------------------------------------------------------

void
utf8_append_(std::string& s, std::uint32_t cp)
{
  if (cp < 0x80) {
    s.push_back((char)cp);
  } else if (cp < 0x800) {
    s.push_back((char)(0xC0 | (cp >> 6)));
    s.push_back((char)(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    s.push_back((char)(0xE0 | (cp >> 12)));
    s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    s.push_back((char)(0xF0 | (cp >> 18)));
    s.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back((char)(0x80 | (cp & 0x3F)));
  }
}

// Strict UTF-8 decode (what Rust's from_utf8 accepts); false if invalid.
bool
utf8_decode_(const std::string& s, std::vector<std::uint32_t>* out)
{
  out->clear();
  const auto* p = reinterpret_cast<const unsigned char*>(s.data());
  const std::size_t n = s.size();
  for (std::size_t i = 0; i < n;) {
    const unsigned c = p[i];
    std::uint32_t cp;
    int len;
    if (c < 0x80) { cp = c; len = 1; }
    else if (c >= 0xC2 && c <= 0xDF) { cp = c & 0x1F; len = 2; }
    else if (c >= 0xE0 && c <= 0xEF) { cp = c & 0x0F; len = 3; }
    else if (c >= 0xF0 && c <= 0xF4) { cp = c & 0x07; len = 4; }
    else { return false; }
    if (i + (std::size_t)len > n) { return false; }
    for (int k = 1; k < len; ++k) {
      const unsigned d = p[i + (std::size_t)k];
      if ((d & 0xC0) != 0x80) { return false; }
      cp = (cp << 6) | (d & 0x3F);
    }
    if ((len == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) ||
        (len == 4 && (cp < 0x10000 || cp > 0x10FFFF))) {
      return false;
    }
    out->push_back(cp);
    i += (std::size_t)len;
  }
  return true;
}

CFMutableStringRef
cf_from_cps_(const std::vector<std::uint32_t>& cps)
{
  std::vector<UniChar> u;
  u.reserve(cps.size() * 2);
  for (std::uint32_t cp : cps) {
    if (cp >= 0x10000) {
      cp -= 0x10000;
      u.push_back((UniChar)(0xD800 + (cp >> 10)));
      u.push_back((UniChar)(0xDC00 + (cp & 0x3FF)));
    } else {
      u.push_back((UniChar)cp);
    }
  }
  CFMutableStringRef m = CFStringCreateMutable(kCFAllocatorDefault, 0);
  CFStringAppendCharacters(m, u.data(), (CFIndex)u.size());
  return m;
}

std::vector<std::uint32_t>
cf_to_cps_(CFStringRef s)
{
  const CFIndex n = CFStringGetLength(s);
  std::vector<UniChar> u((std::size_t)n);
  CFStringGetCharacters(s, CFRangeMake(0, n), u.data());
  std::vector<std::uint32_t> out;
  out.reserve(u.size());
  for (std::size_t i = 0; i < u.size(); ++i) {
    std::uint32_t c = u[i];
    if (c >= 0xD800 && c <= 0xDBFF && i + 1 < u.size() &&
        u[i + 1] >= 0xDC00 && u[i + 1] <= 0xDFFF) {
      c = 0x10000 + ((c - 0xD800) << 10) + (u[i + 1] - 0xDC00);
      ++i;
    }
    out.push_back(c);
  }
  return out;
}

// One code point lowercased with no context (Rust's char::to_lowercase,
// which is what the training normalizer applies char by char).
void
lower_one_(std::uint32_t cp, std::vector<std::uint32_t>* out)
{
  if (cp < 0x80) {
    out->push_back(cp >= 'A' && cp <= 'Z' ? cp + 32 : cp);
    return;
  }
  CFMutableStringRef m = cf_from_cps_({cp});
  CFStringLowercase(m, nullptr);
  for (std::uint32_t c : cf_to_cps_(m)) { out->push_back(c); }
  CFRelease(m);
}

bool
is_ws_uni_(std::uint32_t cp)
{
  if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) { return true; }
  if (cp < 0x80) { return false; }
  static const CFCharacterSetRef ws =
      CFCharacterSetGetPredefined(kCFCharacterSetWhitespaceAndNewline);
  return CFCharacterSetIsLongCharacterMember(ws, (UTF32Char)cp);
}

// GPT-2's bytes_to_unicode, inverted: the byte a byte-level char stands
// for, or -1.
int
byte_of_(std::uint32_t cp)
{
  static const std::vector<int> inv = [] {
    std::vector<int> v(512, -1);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      const bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                        (b >= 0xAE && b <= 0xFF);
      v[(std::size_t)(keep ? b : 256 + n++)] = b;
    }
    return v;
  }();
  return cp < inv.size() ? inv[cp] : -1;
}

}  // namespace

std::vector<std::int64_t>
DeepseekV41Engram::numpy_integers(std::uint64_t seed, std::int64_t high,
                                  int n)
{
  // Generator.integers(0, high, n, int64): _rand_int64 with rng = high - 1
  // and the 64-bit Lemire draw (random_bounded_uint64, rng > 2^32 - 1).
  Pcg64 g(seed);
  const std::uint64_t rng = (std::uint64_t)high - 1;
  std::vector<std::int64_t> out;
  for (int i = 0; i < n; ++i) {
    if (rng == 0) { out.push_back(0); continue; }
    const std::uint64_t excl = rng + 1;
    u128 m = (u128)g.next() * excl;
    std::uint64_t left = (std::uint64_t)m;
    if (left < excl) {
      const std::uint64_t thr = (std::numeric_limits<std::uint64_t>::max() -
                                 rng) % excl;
      while (left < thr) {
        m = (u128)g.next() * excl;
        left = (std::uint64_t)m;
      }
    }
    out.push_back((std::int64_t)(m >> 64));
  }
  return out;
}

std::string
DeepseekV41Engram::normalize(const std::string& utf8)
{
  std::vector<std::uint32_t> cps;
  if (!utf8_decode_(utf8, &cps)) { return utf8; }
  // NFKC, then NFD.
  CFMutableStringRef m = cf_from_cps_(cps);
  CFStringNormalize(m, kCFStringNormalizationFormKC);
  CFStringNormalize(m, kCFStringNormalizationFormD);
  std::vector<std::uint32_t> s = cf_to_cps_(m);
  CFRelease(m);
  // StripAccents: drop every mark (General_Category M).
  static const CFCharacterSetRef marks =
      CFCharacterSetGetPredefined(kCFCharacterSetNonBase);
  std::vector<std::uint32_t> t;
  for (std::uint32_t c : s) {
    if (c >= 0x80 && CFCharacterSetIsLongCharacterMember(marks, c)) {
      continue;
    }
    lower_one_(c, &t);
  }
  // [ \t\r\n]+ -> " "
  std::vector<std::uint32_t> w;
  for (std::size_t i = 0; i < t.size(); ++i) {
    const std::uint32_t c = t[i];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      w.push_back(' ');
      while (i + 1 < t.size() && (t[i + 1] == ' ' || t[i + 1] == '\t' ||
                                  t[i + 1] == '\r' || t[i + 1] == '\n')) {
        ++i;
      }
    } else {
      w.push_back(c);
    }
  }
  // A lone space survives the strip (the sentinel's round trip).
  if (w.size() == 1 && w[0] == ' ') { return " "; }
  std::size_t a = 0, b = w.size();
  while (a < b && is_ws_uni_(w[a])) { ++a; }
  while (b > a && is_ws_uni_(w[b - 1])) { --b; }
  std::string out;
  for (std::size_t i = a; i < b; ++i) { utf8_append_(out, w[i]); }
  return out;
}

bool
DeepseekV41Engram::init(const DeepseekV41Config& cfg,
                        const std::string& tokenizer_json, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "DeepSeek-V4.1 engram: " + std::move(m); }
    return false;
  };
  if (cfg.engram_layers.empty()) { return true; }
  _ngram = cfg.engram_ngram;
  _heads = cfg.engram_heads;
  _cols = cfg.engram_cols();

  // ---- the compressed token map --------------------------------------
  std::ifstream in(tokenizer_json);
  if (!in) { return fail("cannot read " + tokenizer_json); }
  std::stringstream ss;
  ss << in.rdbuf();
  FlexData fd;
  try {
    fd = FlexData::from_json(ss.str());
  } catch (const std::exception& e) {
    return fail(std::string("tokenizer.json: ") + e.what());
  }
  auto root = fd.as_object();
  if (!root.contains("model")) { return fail("tokenizer.json has no model"); }
  const FlexData mfd = root.at("model");
  auto model = mfd.as_object();
  if (!model.contains("vocab") || !model.at("vocab").is_object()) {
    return fail("tokenizer.json has no vocab");
  }
  std::vector<std::string> text, raw;
  std::vector<char> special;
  {
    const FlexData vfd = model.at("vocab");
    auto vocab = vfd.as_object();
    std::int32_t n = 0;
    std::vector<std::pair<std::int32_t, std::string>> items;
    for (auto [k, v] : vocab) {
      const std::int32_t id = (std::int32_t)v.as_int(-1);
      if (id < 0) { continue; }
      items.emplace_back(id, std::string(k));
      n = std::max(n, id + 1);
    }
    if (root.contains("added_tokens") && root.at("added_tokens").is_array()) {
      const FlexData afd = root.at("added_tokens");
      for (auto a : afd.as_array()) {
        auto ao = a.as_object();
        const std::int32_t id = (std::int32_t)ao.at("id").as_int(-1);
        if (id >= 0) { n = std::max(n, id + 1); }
      }
    }
    raw.assign((std::size_t)n, std::string());
    text.assign((std::size_t)n, std::string());
    special.assign((std::size_t)n, 0);
    for (auto& [id, s] : items) { raw[(std::size_t)id] = std::move(s); }
    if (root.contains("added_tokens") && root.at("added_tokens").is_array()) {
      const FlexData afd = root.at("added_tokens");
      for (auto a : afd.as_array()) {
        auto ao = a.as_object();
        const std::int32_t id = (std::int32_t)ao.at("id").as_int(-1);
        if (id < 0) { continue; }
        const FlexData cfd = ao.at("content");
        raw[(std::size_t)id] = std::string(cfd.as_string());
        special[(std::size_t)id] = 1;
      }
    }
  }
  const std::size_t V = raw.size();
  if ((int)V < cfg.vocab) {
    // The model's vocabulary may pad the tokenizer's; padded ids never
    // occur and map to 0, as the release's list does past len(tokenizer).
  }
  _map.assign(std::max<std::size_t>(V, (std::size_t)cfg.vocab), 0);
  std::unordered_map<std::string, std::int32_t> key_id;
  std::vector<std::uint32_t> cps;
  for (std::size_t id = 0; id < V; ++id) {
    std::string decoded;
    bool bad = false;
    if (special[id]) {
      decoded = raw[id];
    } else {
      if (!utf8_decode_(raw[id], &cps)) { bad = true; }
      for (std::uint32_t c : cps) {
        const int b = byte_of_(c);
        if (b < 0) { bad = true; break; }
        decoded.push_back((char)b);
      }
    }
    std::string key;
    std::vector<std::uint32_t> chk;
    if (bad || !utf8_decode_(decoded, &chk) ||
        decoded.find("\xEF\xBF\xBD") != std::string::npos) {
      key = raw[id];
    } else {
      key = normalize(decoded);
      if (key.empty()) { key = decoded; }
    }
    auto it = key_id.find(key);
    if (it == key_id.end()) {
      it = key_id.emplace(std::move(key), (std::int32_t)key_id.size()).first;
    }
    _map[id] = it->second;
  }
  _cvocab = (int)key_id.size();
  if (_cvocab != cfg.engram_cvocab) {
    return fail(std::format("the tokenizer compresses to {} ids, the "
                            "checkpoint's tables were hashed over {}",
                            _cvocab, cfg.engram_cvocab));
  }
  if (cfg.engram_pad < 0 || (std::size_t)cfg.engram_pad >= V) {
    return fail("engram pad token outside the vocabulary");
  }
  _pad = _map[(std::size_t)cfg.engram_pad];

  // ---- the primes, in order, never reused -----------------------------
  std::set<std::uint64_t> seen;
  _primes.clear();
  _offsets.clear();
  for (std::size_t l = 0; l < cfg.engram_layers.size(); ++l) {
    std::vector<std::int64_t> flat;
    for (int g = 0; g < _ngram - 1; ++g) {
      std::uint64_t cur = (std::uint64_t)cfg.engram_vocab - 1;
      for (int h = 0; h < _heads; ++h) {
        std::uint64_t c = cur + 1;
        while (!is_prime_(c) || seen.count(c) != 0) { ++c; }
        seen.insert(c);
        cur = c;
        flat.push_back((std::int64_t)c);
      }
    }
    std::vector<std::int64_t> off(flat.size(), 0);
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < flat.size(); ++i) {
      off[i] = sum;
      sum += flat[i];
    }
    if (sum != cfg.engram_rows[l]) {
      return fail(std::format("layer {}'s primes sum to {} rows, its table "
                              "has {}", cfg.engram_layers[l], sum,
                              cfg.engram_rows[l]));
    }
    _primes.push_back(std::move(flat));
    _offsets.push_back(std::move(off));
  }

  // ---- the multipliers -------------------------------------------------
  const std::int64_t max_long = std::numeric_limits<std::int64_t>::max();
  const std::int64_t bound =
      std::max<std::int64_t>(1, (max_long / _cvocab) / 2);
  _mult.clear();
  for (int L : cfg.engram_layers) {
    std::vector<std::int64_t> v =
        numpy_integers((std::uint64_t)(10007 * (std::int64_t)L), bound,
                       _ngram);
    for (auto& x : v) { x = x * 2 + 1; }
    _mult.push_back(std::move(v));
  }
  return true;
}

std::int32_t
DeepseekV41Engram::compress(std::int32_t token) const
{
  if (token < 0 || (std::size_t)token >= _map.size()) { return 0; }
  return _map[(std::size_t)token];
}

void
DeepseekV41Engram::hash(std::span<const std::int32_t> hist,
                        std::int64_t* out) const
{
  const int pos = (int)hist.size() - 1;
  std::int64_t tok[16];
  bool blocked = false;
  for (int s = 0; s < _ngram; ++s) {
    const int p = pos - s;
    const std::int32_t src = p >= 0 ? hist[(std::size_t)p] : kDead;
    blocked = blocked || p < 0 || src == kDead;
    tok[s] = blocked ? _pad : src;
  }
  for (std::size_t l = 0; l < _mult.size(); ++l) {
    const auto& mul = _mult[l];
    std::int64_t rolling = tok[0] * mul[0];
    for (int i = 1; i < _ngram; ++i) {
      rolling ^= tok[i] * mul[(std::size_t)i];
      for (int h = 0; h < _heads; ++h) {
        const std::size_t c = (std::size_t)((i - 1) * _heads + h);
        out[l * (std::size_t)_cols + c] =
            rolling % _primes[l][c] + _offsets[l][c];
      }
    }
  }
}

}  // namespace vpipe::genai
