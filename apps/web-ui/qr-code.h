#pragma once

// A small QR encoder, just big enough for "point a phone at the console".
//
// Scope is deliberately narrow: error-correction level L, versions 1-6
// (21x21 up to 41x41), alphanumeric or byte mode chosen automatically.
// That covers a LAN URL with room to spare and keeps the tables short --
// version 7 and up would additionally need the 18-bit version-information
// blocks, and no URL this prints needs them.
//
// Level L (7% recovery) rather than M (15%): a symbol drawn on a screen
// is the easy case for a reader -- crisp edges, uniform illumination, no
// print or fold damage -- and error correction is there for the hard one.
// The 25 characters L buys back are what keep a real LAN URL inside
// version 2, a 25x25 symbol. At M the same URL needs version 3 (29x29):
// a third more area on the console to guard against damage a terminal
// cannot inflict.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe::webui {

// A finished symbol: `size` x `size` modules, row-major, non-zero = dark.
struct QrCode {
  int                       size    = 0;   // 21 + 4 * (version - 1)
  int                       version = 0;   // 0 when encoding failed
  std::vector<std::uint8_t> modules;

  bool
  dark(int x, int y) const
  {
    if (x < 0 || y < 0 || x >= size || y >= size) { return false; }
    return modules[static_cast<std::size_t>(y) * size + x] != 0;
  }
};

// Encode `text` at the smallest version that fits. Returns a QrCode with
// version 0 if it does not fit in version 6 -- callers print the plain
// URL in that case rather than a symbol nothing can read.
QrCode qr_encode(std::string_view text);

// The symbol as lines of text, ready to write to a terminal.
//
// Two module rows per line via the upper-half-block glyph, so the result
// is about as tall as it is wide in a cell grid that is itself about 1:2.
// With `color`, each glyph carries an explicit black/white foreground and
// background, which is what makes the symbol scannable on a dark terminal
// -- the quiet zone has to be LIGHT, and "no color" on a dark theme gives
// an inverted symbol that only some readers accept.
//
// The required 4-module quiet zone is included.
std::string qr_terminal(const QrCode& qr, bool color);

}  // namespace vpipe::webui
