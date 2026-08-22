#include "minitest.h"
#include "stages/xet-fetch.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <compression.h>

using namespace std;
using namespace vpipe;

namespace {

// Split a stream the way the content store's writer does before it
// compresses: byte 0,4,8... first, then 1,5,9..., and so on, the four
// planes stored back to back. Written out longhand here rather than as
// "the inverse of the function under test", so the test pins the
// CONVENTION and not merely a round trip.
string
bg4_split_(const string& in)
{
  string out;
  out.reserve(in.size());
  for (size_t plane = 0; plane < 4; ++plane) {
    for (size_t i = plane; i < in.size(); i += 4) {
      out.push_back(in[i]);
    }
  }
  return out;
}

// Wrap bytes in the LZ4 frame the store writes: magic, FLG with block
// independence and nothing else, BD, header checksum, then one
// length-prefixed block and an end mark.
string
lz4_frame_(const string& raw)
{
  vector<uint8_t> blk(raw.size() + 1024);
  const size_t n = compression_encode_buffer(
      blk.data(), blk.size(),
      reinterpret_cast<const uint8_t*>(raw.data()), raw.size(), nullptr,
      COMPRESSION_LZ4_RAW);
  string out;
  const bool stored = (n == 0 || n >= raw.size());
  const uint32_t len = static_cast<uint32_t>(stored ? raw.size() : n);
  const uint32_t bs  = stored ? (len | 0x80000000u) : len;
  out.append("\x04\x22\x4d\x18", 4);
  out.push_back(static_cast<char>(0x60));   // FLG: v1, independent blocks
  out.push_back(static_cast<char>(0x50));   // BD:  256 KB blocks
  out.push_back(static_cast<char>(0x00));   // header checksum (unchecked)
  out.append(reinterpret_cast<const char*>(&bs), 4);
  if (stored) {
    out.append(raw);
  } else {
    out.append(reinterpret_cast<const char*>(blk.data()), n);
  }
  out.append(4, '\0');                      // end mark
  return out;
}

// An 8-byte chunk header: version, 24-bit compressed length, scheme,
// 24-bit uncompressed length.
string
chunk_(unsigned scheme, const string& payload, size_t ulen)
{
  string h(8, '\0');
  h[0] = 0;
  h[1] = static_cast<char>(payload.size() & 0xff);
  h[2] = static_cast<char>((payload.size() >> 8) & 0xff);
  h[3] = static_cast<char>((payload.size() >> 16) & 0xff);
  h[4] = static_cast<char>(scheme);
  h[5] = static_cast<char>(ulen & 0xff);
  h[6] = static_cast<char>((ulen >> 8) & 0xff);
  h[7] = static_cast<char>((ulen >> 16) & 0xff);
  return h + payload;
}

// Something bf16-shaped: a low byte that varies and a high byte that
// barely does, which is the whole reason the store separates the planes.
string
weights_(size_t n)
{
  string s;
  s.reserve(n);
  uint32_t x = 12345;
  for (size_t i = 0; i < n; i += 2) {
    x = x * 1103515245u + 12345u;
    s.push_back(static_cast<char>(x >> 24));         // mantissa-ish
    s.push_back(static_cast<char>(0x3f | (i & 1)));  // exponent-ish
  }
  s.resize(n);
  return s;
}

const unsigned char*
u_(const string& s)
{
  return reinterpret_cast<const unsigned char*>(s.data());
}

}

TEST(xet_fetch, bg4_regroup_undoes_the_documented_split) {
  // Hand-checked: eight bytes split into four planes of two.
  const string in = "\x01\x02\x03\x04\x05\x06\x07\x08"s;
  const string split = bg4_split_(in);
  EXPECT_TRUE(split == "\x01\x05\x02\x06\x03\x07\x04\x08"s);
  string buf = split, scratch;
  xet_bg4_regroup(&buf[0], buf.size(), scratch);
  EXPECT_TRUE(buf == in);

  // A length that is not a multiple of four: the remainder goes to the
  // leading planes, so plane 0 is one longer than plane 3.
  const string odd = "abcdefghij"s;                 // 10 = 4*2 + 2
  const string os  = bg4_split_(odd);
  EXPECT_TRUE(os == "aeibfjcgdh"s);
  string ob = os;
  xet_bg4_regroup(&ob[0], ob.size(), scratch);
  EXPECT_TRUE(ob == odd);
}

TEST(xet_fetch, decode_chunk_reads_every_scheme) {
  const string raw = weights_(9000);
  string out, err;

  // Scheme 0: the store kept the bytes as they are.
  out.clear();
  size_t used = xet_decode_chunk(u_(chunk_(0, raw, raw.size())),
                                 8 + raw.size(), out, err);
  EXPECT_TRUE(used == 8 + raw.size());
  EXPECT_TRUE(out == raw);

  // Scheme 1: an LZ4 frame around them.
  const string f1 = lz4_frame_(raw);
  const string c1 = chunk_(1, f1, raw.size());
  out.clear();
  used = xet_decode_chunk(u_(c1), c1.size(), out, err);
  EXPECT_TRUE(used == c1.size());
  EXPECT_TRUE(out == raw);

  // Scheme 2: the same, with the byte planes separated first. This is
  // what every bf16 checkpoint arrives as.
  const string f2 = lz4_frame_(bg4_split_(raw));
  const string c2 = chunk_(2, f2, raw.size());
  out.clear();
  used = xet_decode_chunk(u_(c2), c2.size(), out, err);
  EXPECT_TRUE(used == c2.size());
  EXPECT_TRUE(out == raw);

  // And it is worth the trouble: separating the planes is what gives
  // LZ4 anything to find in float weights.
  EXPECT_TRUE(f2.size() < f1.size());

  // Decoding appends, so a run of chunks rebuilds a run of bytes.
  out.clear();
  xet_decode_chunk(u_(c2), c2.size(), out, err);
  xet_decode_chunk(u_(c1), c1.size(), out, err);
  EXPECT_TRUE(out == raw + raw);
}

TEST(xet_fetch, decode_chunk_refuses_what_it_cannot_trust) {
  const string raw = weights_(512);
  string out, err;

  // A header that is not all there.
  EXPECT_TRUE(xet_decode_chunk(u_(string(4, '\0')), 4, out, err) == 0);

  // A payload the buffer does not hold: a range that arrived short must
  // not be read past, and must not be treated as a complete chunk.
  const string c = chunk_(0, raw, raw.size());
  EXPECT_TRUE(xet_decode_chunk(u_(c), c.size() - 10, out, err) == 0);

  // A scheme from a future version of the store.
  const string cx = chunk_(7, raw, raw.size());
  EXPECT_TRUE(xet_decode_chunk(u_(cx), cx.size(), out, err) == 0);
  EXPECT_TRUE(err.find("scheme") != string::npos);

  // Stored uncompressed, but the two lengths disagree -- one of them is
  // wrong and there is no way to tell which.
  const string cb = chunk_(0, raw, raw.size() - 1);
  EXPECT_TRUE(xet_decode_chunk(u_(cb), cb.size(), out, err) == 0);

  // A frame whose contents do not add up to the length the header
  // promised. Accepting it would put a short slice into the file and
  // shift everything after it.
  const string cs = chunk_(1, lz4_frame_(raw), raw.size() + 64);
  EXPECT_TRUE(xet_decode_chunk(u_(cs), cs.size(), out, err) == 0);

  // Not a frame at all.
  const string cg = chunk_(1, string(64, 'x'), raw.size());
  EXPECT_TRUE(xet_decode_chunk(u_(cg), cg.size(), out, err) == 0);
}
