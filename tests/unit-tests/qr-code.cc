// QR encoder tests (apps/web-ui/qr-code.cc), the symbol vpipe-web-ui
// prints for --show-qr.
//
// The golden grids below were produced by this encoder and then VERIFIED
// BY DECODING them with CoreImage's CIDetector -- so what they pin is a
// symbol a real reader accepted, not merely what the code did on the day
// it was written. That is the only reason a golden is worth anything
// here: nothing else in the tree can read a QR code back.
//
// The tables this exercises (Reed-Solomon parameters per version, the
// format-information bits, the alphanumeric packing) are exact values
// where a single wrong digit still produces a plausible-looking square
// that no phone can read. Hence goldens rather than shape assertions.

#include "minitest.h"
#include "apps/web-ui/qr-code.h"

#include <string>
#include <vector>

using namespace vpipe::webui;

namespace {

// Compare a symbol against a row-per-string picture ('1' = dark).
bool
matches(const QrCode& qr, const std::vector<std::string>& want)
{
  if (qr.size != static_cast<int>(want.size())) { return false; }
  for (int y = 0; y < qr.size; ++y) {
    if (static_cast<int>(want[static_cast<std::size_t>(y)].size())
        != qr.size) {
      return false;
    }
    for (int x = 0; x < qr.size; ++x) {
      const bool dark = want[static_cast<std::size_t>(y)]
                            [static_cast<std::size_t>(x)] == '1';
      if (qr.dark(x, y) != dark) { return false; }
    }
  }
  return true;
}

std::string
repeat(char c, int n)
{
  return std::string(static_cast<std::size_t>(n), c);
}

}  // namespace

// A URL of exactly the shape --show-qr prints: upper-cased scheme and
// host (which is what keeps it in alphanumeric mode) and a 14-character
// key. The size is the point of the exercise -- 25x25.
TEST(qr_code, lan_url_is_version_2) {
  const QrCode qr = qr_encode("HTTP://192.168.1.42:9876/UU9RPMB392XF5R");
  ASSERT_TRUE(qr.version == 2);
  ASSERT_TRUE(qr.size == 25);
  const std::vector<std::string> want = {
      "1111111011101011001111111",
      "1000001011100101101000001",
      "1011101010110110001011101",
      "1011101011010100101011101",
      "1011101000000010001011101",
      "1000001010011110001000001",
      "1111111000101010101111111",
      "0000000001001110100000000",
      "1100110000011001000101111",
      "1000000101101100111011000",
      "0011101000011101011000100",
      "0101100010100010001100011",
      "0100011010110000000101100",
      "1101110101001111110010100",
      "0011111001110001110000111",
      "0001010110000111000110000",
      "1110101101011001111111101",
      "0000000010111100100011110",
      "1111111001100011101011111",
      "1000001010101101100011001",
      "1011101010100110111110001",
      "1011101000010011010100011",
      "1011101000100001101110110",
      "1000001011110011000111011",
      "1111111011001111000011001",
  };
  EXPECT_TRUE(matches(qr, want));
}

// Lower case is outside the alphanumeric set, so this takes the 8-bit
// byte path -- a different mode indicator, count field and packing.
TEST(qr_code, byte_mode_lower_case) {
  const QrCode qr = qr_encode("http://a.b/c");
  ASSERT_TRUE(qr.version == 1);
  ASSERT_TRUE(qr.size == 21);
  const std::vector<std::string> want = {
      "111111101111101111111",
      "100000100101001000001",
      "101110100110001011101",
      "101110100111001011101",
      "101110100010101011101",
      "100000100001101000001",
      "111111100010101111111",
      "000000001111100000000",
      "110110000110101000001",
      "011000001110101001110",
      "001110101000011001111",
      "000010000011000100100",
      "110001100010010000010",
      "000000001010100011110",
      "111111100110111110100",
      "100000100101011101101",
      "101110101111100010010",
      "101110101011000001100",
      "101110100100110111111",
      "100000101011011010111",
      "111111101101011101000",
  };
  EXPECT_TRUE(matches(qr, want));
}

// The alphanumeric capacity of each version at level L. Getting a
// Reed-Solomon row wrong shows up here first: the version below its
// boundary must fit exactly, and one character more must roll over.
TEST(qr_code, version_capacity_boundaries) {
  const int cap[6]  = { 25, 47, 77, 114, 154, 195 };
  const int size[6] = { 21, 25, 29,  33,  37,  41 };
  for (int v = 0; v < 6; ++v) {
    const QrCode at = qr_encode(repeat('A', cap[v]));
    EXPECT_TRUE(at.version == v + 1);
    EXPECT_TRUE(at.size == size[v]);
    if (v + 1 < 6) {
      const QrCode over = qr_encode(repeat('A', cap[v] + 1));
      EXPECT_TRUE(over.version == v + 2);
    }
  }
  // Past version 6 the encoder declines rather than emitting a symbol
  // no reader could accept; the caller prints the plain URL instead.
  const QrCode too_big = qr_encode(repeat('A', 196));
  EXPECT_TRUE(too_big.version == 0);
  EXPECT_TRUE(too_big.size == 0);
}

// Byte mode costs 8 bits per character against alphanumeric's 5.5, which
// is exactly why --show-qr upper-cases the scheme and host.
TEST(qr_code, upper_case_url_is_smaller) {
  const std::string lower = "http://192.168.1.42:9876/uu9rpmb392xf5r";
  std::string upper = lower;
  for (char& c : upper) {
    if (c >= 'a' && c <= 'z') { c = static_cast<char>(c - 'a' + 'A'); }
  }
  const QrCode lo = qr_encode(lower);
  const QrCode up = qr_encode(upper);
  EXPECT_TRUE(lo.version > 0 && up.version > 0);
  EXPECT_TRUE(up.size < lo.size);
  EXPECT_TRUE(up.size == 25);
}

// The rendering contract: two module rows per line, and a light quiet
// zone. A dark quiet zone is the classic reason a terminal QR does not
// scan, so it is worth pinning.
TEST(qr_code, terminal_rendering_shape) {
  const QrCode qr = qr_encode("HTTP://192.168.1.42:9876/UU9RPMB392XF5R");
  ASSERT_TRUE(qr.size == 25);
  const std::string plain = qr_terminal(qr, /*color=*/false);
  ASSERT_TRUE(!plain.empty());

  std::vector<std::string> lines;
  for (std::size_t i = 0, j = 0; i <= plain.size(); ++i) {
    if (i == plain.size() || plain[i] == '\n') {
      if (i > j) { lines.push_back(plain.substr(j, i - j)); }
      j = i + 1;
    }
  }
  // 25 modules + a 4-module quiet zone each side = 33 rows -> 17 lines.
  EXPECT_TRUE(lines.size() == 17);
  // The first two lines are quiet zone: nothing but spaces.
  EXPECT_TRUE(lines[0].find_first_not_of(' ') == std::string::npos);
  EXPECT_TRUE(lines[1].find_first_not_of(' ') == std::string::npos);
  // The coloured form carries an SGR reset on every line, so a symbol
  // never bleeds its background into the rest of the console.
  const std::string colored = qr_terminal(qr, /*color=*/true);
  EXPECT_TRUE(colored.find("\033[0m\n") != std::string::npos);
  EXPECT_TRUE(colored.find("\033[47m") != std::string::npos);   // light bg
  EXPECT_TRUE(colored.find("\033[30m") != std::string::npos);   // dark fg
}
