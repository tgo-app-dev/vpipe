#include "qr-code.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace vpipe::webui {

namespace {

using std::size_t;
using std::string;
using std::uint8_t;
using std::vector;

// ---- per-version tables (error-correction level L, versions 1-6) -----
//
// data_cw   total data codewords
// ec_cw     error-correction codewords PER BLOCK
// blocks    number of RS blocks
//
// Only version 6 uses more than one block here, and it splits evenly
// (136/2), so the general "some blocks carry one codeword more than
// others" case never arises and the interleave below stays a plain
// column-major walk. The divisibility is re-checked at encode time.
struct VersionInfo {
  int data_cw;
  int ec_cw;
  int blocks;
};

constexpr std::array<VersionInfo, 6> kVersions = {{
    {  19,  7, 1 },   // 1-L  21x21   25 alphanumeric characters
    {  34, 10, 1 },   // 2-L  25x25   47   -- the target for a LAN URL
    {  55, 15, 1 },   // 3-L  29x29   77
    {  80, 20, 1 },   // 4-L  33x33  114
    { 108, 26, 1 },   // 5-L  37x37  154
    { 136, 18, 2 },   // 6-L  41x41  195
}};

// Sole alignment-pattern centre coordinate for versions 2-6 (versions in
// this range have exactly one, at the intersection that misses all three
// finder patterns). Index 0 is version 1, which has none.
constexpr std::array<int, 6> kAlignPos = { 0, 18, 22, 26, 30, 34 };

int
version_size(int version)
{
  return 21 + 4 * (version - 1);
}

// ---- bit buffer -------------------------------------------------------
struct Bits {
  vector<bool> v;

  void
  push(std::uint32_t value, int len)
  {
    for (int i = len - 1; i >= 0; --i) {
      v.push_back(((value >> i) & 1u) != 0);
    }
  }
};

// ---- mode selection ---------------------------------------------------
// The alphanumeric set: 0-9 A-Z and nine punctuation marks. Note the
// absence of lower case -- which is why the caller upper-cases the
// scheme and host of the URL it encodes (both are case-insensitive) to
// stay in this mode and out of the 8-bits-per-character one.
int
alnum_value(char c)
{
  if (c >= '0' && c <= '9') { return c - '0'; }
  if (c >= 'A' && c <= 'Z') { return c - 'A' + 10; }
  switch (c) {
    case ' ': return 36;
    case '$': return 37;
    case '%': return 38;
    case '*': return 39;
    case '+': return 40;
    case '-': return 41;
    case '.': return 42;
    case '/': return 43;
    case ':': return 44;
    default:  return -1;
  }
}

bool
is_alnum_text(std::string_view s)
{
  for (char c : s) {
    if (alnum_value(c) < 0) { return false; }
  }
  return true;
}

// Data bits for `s`, excluding the mode indicator and character count.
size_t
payload_bits(std::string_view s, bool alnum)
{
  if (!alnum) { return s.size() * 8; }
  return (s.size() / 2) * 11 + ((s.size() % 2) ? 6 : 0);
}

// ---- GF(256) ----------------------------------------------------------
uint8_t
gf_mul(uint8_t x, uint8_t y)
{
  int z = 0;
  for (int i = 7; i >= 0; --i) {
    z = (z << 1) ^ ((z >> 7) * 0x11D);      // reduce by the QR primitive
    z ^= ((y >> i) & 1) * x;
  }
  return static_cast<uint8_t>(z);
}

// Generator polynomial of the given degree, as coefficients excluding
// the leading 1.
vector<uint8_t>
rs_divisor(int degree)
{
  vector<uint8_t> result(static_cast<size_t>(degree), 0);
  result[result.size() - 1] = 1;
  uint8_t root = 1;
  for (int i = 0; i < degree; ++i) {
    for (size_t j = 0; j < result.size(); ++j) {
      result[j] = gf_mul(result[j], root);
      if (j + 1 < result.size()) { result[j] ^= result[j + 1]; }
    }
    root = gf_mul(root, 0x02);
  }
  return result;
}

vector<uint8_t>
rs_remainder(const uint8_t* data, size_t n, const vector<uint8_t>& div)
{
  vector<uint8_t> result(div.size(), 0);
  for (size_t k = 0; k < n; ++k) {
    const uint8_t factor = data[k] ^ result[0];
    result.erase(result.begin());
    result.push_back(0);
    for (size_t i = 0; i < result.size(); ++i) {
      result[i] ^= gf_mul(div[i], factor);
    }
  }
  return result;
}

// ---- symbol construction ----------------------------------------------
struct Canvas {
  int             size = 0;
  vector<uint8_t> mod;      // 1 = dark
  vector<uint8_t> fixed;    // 1 = function module (never masked)

  explicit Canvas(int s)
      : size(s), mod(static_cast<size_t>(s) * s, 0),
        fixed(static_cast<size_t>(s) * s, 0)
  {}

  size_t
  idx(int x, int y) const
  {
    return static_cast<size_t>(y) * size + x;
  }
  void
  set_fn(int x, int y, bool dark)
  {
    if (x < 0 || y < 0 || x >= size || y >= size) { return; }
    mod[idx(x, y)]   = dark ? 1 : 0;
    fixed[idx(x, y)] = 1;
  }
};

void
draw_finder(Canvas& c, int cx, int cy)
{
  // The 7x7 eye plus its one-module separator, drawn as a distance field
  // so the ring structure falls out of the arithmetic.
  for (int dy = -4; dy <= 4; ++dy) {
    for (int dx = -4; dx <= 4; ++dx) {
      const int dist = std::max(std::abs(dx), std::abs(dy));
      c.set_fn(cx + dx, cy + dy, dist != 2 && dist != 4);
    }
  }
}

void
draw_function_patterns(Canvas& c, int version)
{
  const int n = c.size;
  // Timing patterns first; the finders overwrite their ends.
  for (int i = 0; i < n; ++i) {
    c.set_fn(6, i, i % 2 == 0);
    c.set_fn(i, 6, i % 2 == 0);
  }
  draw_finder(c, 3, 3);
  draw_finder(c, n - 4, 3);
  draw_finder(c, 3, n - 4);

  if (version >= 2) {
    const int p = kAlignPos[static_cast<size_t>(version) - 1];
    for (int dy = -2; dy <= 2; ++dy) {
      for (int dx = -2; dx <= 2; ++dx) {
        c.set_fn(p + dx, p + dy,
                 std::max(std::abs(dx), std::abs(dy)) != 1);
      }
    }
  }

  // Reserve the format-information modules (their real values are
  // written once the mask is chosen) and the always-dark module.
  for (int i = 0; i <= 8; ++i) {
    c.set_fn(8, i, false);
    c.set_fn(i, 8, false);
  }
  for (int i = 0; i < 8; ++i) {
    c.set_fn(n - 1 - i, 8, false);
    c.set_fn(8, n - 1 - i, false);
  }
  c.set_fn(8, n - 8, true);
}

// 15-bit format information: 2 bits EC level + 3 bits mask,
// BCH(15,5)-protected and XOR-masked so it is never all-zero.
void
draw_format(Canvas& c, int mask)
{
  // The level bits are NOT in level order: L=01, M=00, Q=11, H=10.
  const int data = (0b01 << 3) | mask;       // level L
  int rem = data << 10;
  for (int i = 14; i >= 10; --i) {
    if ((rem >> i) & 1) { rem ^= 0x537 << (i - 10); }
  }
  const int bits = ((data << 10) | rem) ^ 0x5412;
  const auto bit = [bits](int i) { return ((bits >> i) & 1) != 0; };
  const int n = c.size;

  for (int i = 0; i <= 5; ++i) { c.set_fn(8, i, bit(i)); }
  c.set_fn(8, 7, bit(6));
  c.set_fn(8, 8, bit(7));
  c.set_fn(7, 8, bit(8));
  for (int i = 9; i < 15; ++i) { c.set_fn(14 - i, 8, bit(i)); }

  for (int i = 0; i < 8; ++i) { c.set_fn(n - 1 - i, 8, bit(i)); }
  for (int i = 8; i < 15; ++i) { c.set_fn(8, n - 15 + i, bit(i)); }
  c.set_fn(8, n - 8, true);
}

// The zigzag: two-module-wide columns walked right to left, alternating
// upward and downward, skipping the vertical timing column.
void
draw_data(Canvas& c, const vector<uint8_t>& data)
{
  size_t bit = 0;
  const size_t total = data.size() * 8;
  for (int right = c.size - 1; right >= 1; right -= 2) {
    if (right == 6) { right = 5; }
    for (int vert = 0; vert < c.size; ++vert) {
      for (int j = 0; j < 2; ++j) {
        const int  x       = right - j;
        const bool upward  = ((right + 1) & 2) == 0;
        const int  y       = upward ? c.size - 1 - vert : vert;
        if (c.fixed[c.idx(x, y)] || bit >= total) { continue; }
        const uint8_t byte = data[bit >> 3];
        c.mod[c.idx(x, y)] = (byte >> (7 - (bit & 7))) & 1;
        ++bit;
      }
    }
  }
}

bool
mask_bit(int mask, int x, int y)
{
  switch (mask) {
    case 0:  return (x + y) % 2 == 0;
    case 1:  return y % 2 == 0;
    case 2:  return x % 3 == 0;
    case 3:  return (x + y) % 3 == 0;
    case 4:  return (y / 2 + x / 3) % 2 == 0;
    case 5:  return x * y % 2 + x * y % 3 == 0;
    case 6:  return (x * y % 2 + x * y % 3) % 2 == 0;
    default: return ((x + y) % 2 + x * y % 3) % 2 == 0;
  }
}

void
apply_mask(Canvas& c, int mask)
{
  for (int y = 0; y < c.size; ++y) {
    for (int x = 0; x < c.size; ++x) {
      if (c.fixed[c.idx(x, y)]) { continue; }
      c.mod[c.idx(x, y)] ^= mask_bit(mask, x, y) ? 1 : 0;
    }
  }
}

// The four penalty rules from the spec: runs of five or more, 2x2
// blocks, finder-like 1:1:3:1:1 patterns, and overall dark/light
// balance. Lower is better; the encoder tries all eight masks.
int
penalty(const Canvas& c)
{
  const int n = c.size;
  int score = 0;
  const auto dark = [&](int x, int y) { return c.mod[c.idx(x, y)] != 0; };

  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      // Rule 1, both directions.
      for (int dir = 0; dir < 2; ++dir) {
        const int dx = dir == 0 ? 1 : 0;
        const int dy = dir == 0 ? 0 : 1;
        if ((dir == 0 && x != 0) || (dir == 1 && y != 0)) {
          if (dark(x - dx, y - dy) == dark(x, y)) { continue; }
        }
        int run = 0;
        int cx = x, cy = y;
        while (cx < n && cy < n && dark(cx, cy) == dark(x, y)) {
          ++run;
          cx += dx;
          cy += dy;
        }
        if (run >= 5) { score += 3 + (run - 5); }
      }
      // Rule 2.
      if (x + 1 < n && y + 1 < n) {
        const bool a = dark(x, y);
        if (a == dark(x + 1, y) && a == dark(x, y + 1)
            && a == dark(x + 1, y + 1)) {
          score += 3;
        }
      }
      // Rule 3: 1:1:3:1:1 with four light modules on one side.
      for (int dir = 0; dir < 2; ++dir) {
        const int dx = dir == 0 ? 1 : 0;
        const int dy = dir == 0 ? 0 : 1;
        const int len = 11;
        if (x + dx * (len - 1) >= n || y + dy * (len - 1) >= n) { continue; }
        int pat = 0;
        for (int k = 0; k < len; ++k) {
          pat = (pat << 1) | (dark(x + dx * k, y + dy * k) ? 1 : 0);
        }
        if (pat == 0b10111010000 || pat == 0b00001011101) { score += 40; }
      }
    }
  }
  int darks = 0;
  for (uint8_t m : c.mod) { darks += m ? 1 : 0; }
  const int total = n * n;
  const int pct   = (darks * 100 + total / 2) / total;
  score += std::abs(pct - 50) / 5 * 10;
  return score;
}

}  // namespace

QrCode
qr_encode(std::string_view text)
{
  const bool alnum = is_alnum_text(text);
  const int  mode  = alnum ? 0b0010 : 0b0100;
  // Versions 1-9 all use a 9-bit alphanumeric / 8-bit byte count field,
  // so the header size does not depend on which version is picked.
  const int  count_bits = alnum ? 9 : 8;
  const size_t need = 4 + static_cast<size_t>(count_bits)
                    + payload_bits(text, alnum);

  int version = 0;
  for (int v = 1; v <= 6; ++v) {
    const VersionInfo& vi = kVersions[static_cast<size_t>(v) - 1];
    if (need <= static_cast<size_t>(vi.data_cw) * 8) { version = v; break; }
  }
  if (version == 0) { return {}; }
  const VersionInfo& vi = kVersions[static_cast<size_t>(version) - 1];
  if (vi.data_cw % vi.blocks != 0) { return {}; }   // see kVersions

  // ---- data codewords -------------------------------------------------
  Bits bits;
  bits.push(static_cast<std::uint32_t>(mode), 4);
  bits.push(static_cast<std::uint32_t>(text.size()), count_bits);
  if (alnum) {
    for (size_t i = 0; i + 1 < text.size(); i += 2) {
      const int pair = alnum_value(text[i]) * 45 + alnum_value(text[i + 1]);
      bits.push(static_cast<std::uint32_t>(pair), 11);
    }
    if (text.size() % 2) {
      bits.push(static_cast<std::uint32_t>(alnum_value(text.back())), 6);
    }
  } else {
    for (char ch : text) {
      bits.push(static_cast<std::uint8_t>(ch), 8);
    }
  }
  const size_t capacity = static_cast<size_t>(vi.data_cw) * 8;
  // Terminator (up to four zeroes), then pad to a codeword boundary,
  // then the two alternating pad codewords until the version is full.
  for (int i = 0; i < 4 && bits.v.size() < capacity; ++i) {
    bits.v.push_back(false);
  }
  while (bits.v.size() % 8 != 0) { bits.v.push_back(false); }
  vector<uint8_t> data;
  data.reserve(static_cast<size_t>(vi.data_cw));
  for (size_t i = 0; i < bits.v.size(); i += 8) {
    uint8_t b = 0;
    for (int k = 0; k < 8; ++k) {
      b = static_cast<uint8_t>((b << 1) | (bits.v[i + k] ? 1 : 0));
    }
    data.push_back(b);
  }
  for (uint8_t pad = 0xEC; data.size() < static_cast<size_t>(vi.data_cw);
       pad = static_cast<uint8_t>(pad == 0xEC ? 0x11 : 0xEC)) {
    data.push_back(pad);
  }

  // ---- error correction + interleave ----------------------------------
  const int per_block = vi.data_cw / vi.blocks;
  const vector<uint8_t> div = rs_divisor(vi.ec_cw);
  vector<vector<uint8_t>> ec;
  ec.reserve(static_cast<size_t>(vi.blocks));
  for (int b = 0; b < vi.blocks; ++b) {
    ec.push_back(rs_remainder(data.data() + b * per_block,
                              static_cast<size_t>(per_block), div));
  }
  vector<uint8_t> final_cw;
  final_cw.reserve(static_cast<size_t>(vi.data_cw)
                   + static_cast<size_t>(vi.ec_cw) * vi.blocks);
  for (int i = 0; i < per_block; ++i) {
    for (int b = 0; b < vi.blocks; ++b) {
      final_cw.push_back(data[static_cast<size_t>(b) * per_block + i]);
    }
  }
  for (int i = 0; i < vi.ec_cw; ++i) {
    for (int b = 0; b < vi.blocks; ++b) {
      final_cw.push_back(ec[static_cast<size_t>(b)][static_cast<size_t>(i)]);
    }
  }

  // ---- draw, then pick the mask that scores best ----------------------
  Canvas base(version_size(version));
  draw_function_patterns(base, version);
  draw_data(base, final_cw);

  Canvas best(0);
  int best_score = 0;
  for (int m = 0; m < 8; ++m) {
    Canvas c = base;
    apply_mask(c, m);
    draw_format(c, m);
    const int s = penalty(c);
    if (best.size == 0 || s < best_score) {
      best = std::move(c);
      best_score = s;
    }
  }

  QrCode out;
  out.size    = best.size;
  out.version = version;
  out.modules = std::move(best.mod);
  return out;
}

std::string
qr_terminal(const QrCode& qr, bool color)
{
  if (qr.size <= 0) { return {}; }
  const int quiet = 4;                       // required by the spec
  const int n     = qr.size + 2 * quiet;
  const auto dark = [&](int x, int y) {
    return qr.dark(x - quiet, y - quiet);
  };

  string out;
  out.reserve(static_cast<size_t>(n) * n);
  // Two module rows per line: the upper half-block's FOREGROUND paints
  // the top row and its BACKGROUND the bottom one.
  for (int y = 0; y < n; y += 2) {
    for (int x = 0; x < n; ++x) {
      const bool top = dark(x, y);
      const bool bot = (y + 1 < n) && dark(x, y + 1);
      if (!color) {
        // No palette to rely on: full block for dark, space for light.
        // On a dark terminal this reads as an inverted symbol, which
        // many readers still accept -- but --show-qr is best used where
        // colour is available.
        out += top ? (bot ? "█" : "▀")
                   : (bot ? "▄" : " ");
        continue;
      }
      out += top ? "\033[30m" : "\033[37m";   // fg: top module
      out += bot ? "\033[40m" : "\033[47m";   // bg: bottom module
      out += "▀";
    }
    if (color) { out += "\033[0m"; }
    out += '\n';
  }
  return out;
}

}  // namespace vpipe::webui
