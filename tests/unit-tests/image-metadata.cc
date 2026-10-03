// common/image-metadata: reading and writing the EXIF block that rides
// alongside image pixels.
//
// Self-contained: the containers here are hand-built minimal JPEG / PNG byte
// streams and the EXIF is a hand-built TIFF block, so nothing depends on a
// sample photo being present. The full round trip through the real stages on
// a real camera JPEG is exercised by load-image / save-image.

#include "minitest.h"

#include "common/flex-data.h"
#include "common/image-metadata.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vpipe;
using namespace vpipe::imgmeta;

namespace {

// A minimal little-endian TIFF/EXIF block: header + one IFD holding
// Make="VPIPE" (ASCII, 6 bytes incl. NUL -> stored out-of-line) and
// Orientation=1 (SHORT, inline).
std::vector<std::uint8_t>
make_tiff_()
{
  std::vector<std::uint8_t> t;
  auto u16 = [&](std::uint16_t v) {
    t.push_back((std::uint8_t)(v & 0xFF)); t.push_back((std::uint8_t)(v >> 8));
  };
  auto u32 = [&](std::uint32_t v) {
    t.push_back((std::uint8_t)(v & 0xFF));
    t.push_back((std::uint8_t)((v >> 8) & 0xFF));
    t.push_back((std::uint8_t)((v >> 16) & 0xFF));
    t.push_back((std::uint8_t)((v >> 24) & 0xFF));
  };
  t.push_back('I'); t.push_back('I');
  u16(42);
  u32(8);              // IFD0 at offset 8
  u16(2);              // two entries
  // Make (0x010F), ASCII, count 6, value at offset 8+2+24+4 = 38
  u16(0x010F); u16(2); u32(6); u32(38);
  // Orientation (0x0112), SHORT, count 1, inline value 1
  u16(0x0112); u16(3); u32(1); u16(1); u16(0);
  u32(0);              // next-IFD = none
  const char* mk = "VPIPE";
  t.insert(t.end(), mk, mk + 6);   // includes the NUL
  return t;
}

// SOI + a stub APP0 + EOI: enough structure for the segment walker.
std::vector<std::uint8_t>
make_jpeg_()
{
  return {0xFF, 0xD8,
          0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00,   // APP0, length 4
          0xFF, 0xD9};
}

// Signature + IHDR + IEND. Chunk CRCs are not validated by the writer, so
// zeros are fine here; the writer computes a real CRC for what it inserts.
std::vector<std::uint8_t>
make_png_()
{
  std::vector<std::uint8_t> p =
      {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  auto chunk = [&](const char* type, int len) {
    p.push_back((std::uint8_t)(len >> 24));
    p.push_back((std::uint8_t)(len >> 16));
    p.push_back((std::uint8_t)(len >> 8));
    p.push_back((std::uint8_t)len);
    p.insert(p.end(), type, type + 4);
    for (int i = 0; i < len; ++i) { p.push_back(0); }
    for (int i = 0; i < 4; ++i) { p.push_back(0); }   // CRC
  };
  chunk("IHDR", 13);
  chunk("IEND", 0);
  return p;
}

std::size_t
count_bytes_(const std::vector<std::uint8_t>& hay, const char* needle,
             std::size_t n)
{
  std::size_t hits = 0;
  if (hay.size() < n) { return 0; }
  for (std::size_t i = 0; i + n <= hay.size(); ++i) {
    if (std::memcmp(hay.data() + i, needle, n) == 0) { ++hits; }
  }
  return hits;
}

}  // namespace

TEST(image_metadata, parses_a_hand_built_tiff) {
  const auto tiff = make_tiff_();
  FlexData d = parse_exif(tiff);
  ASSERT_TRUE(d.is_object());
  auto o = d.as_object();
  ASSERT_TRUE(o.contains("Make"));
  EXPECT_TRUE(std::string(o.at("Make").as_string("")) == "VPIPE");
  ASSERT_TRUE(o.contains("Orientation"));
  EXPECT_TRUE(o.at("Orientation").as_int(-1) == 1);
}

TEST(image_metadata, rejects_non_tiff) {
  std::vector<std::uint8_t> junk = {1, 2, 3, 4, 5, 6, 7, 8};
  EXPECT_TRUE(parse_exif(junk).is_null());
  EXPECT_TRUE(parse_exif(std::vector<std::uint8_t>{}).is_null());
}

TEST(image_metadata, jpeg_round_trip) {
  const auto tiff = make_tiff_();
  auto jpeg = make_jpeg_();
  ASSERT_TRUE(jpeg_set_exif(jpeg, tiff));
  const auto back = find_exif_blob(jpeg);
  ASSERT_TRUE(back.size() == tiff.size());
  EXPECT_TRUE(std::memcmp(back.data(), tiff.data(), tiff.size()) == 0);
  // The APP1 must sit immediately after SOI.
  EXPECT_TRUE(jpeg[2] == 0xFF && jpeg[3] == 0xE1);
}

TEST(image_metadata, png_round_trip) {
  const auto tiff = make_tiff_();
  auto png = make_png_();
  ASSERT_TRUE(png_set_exif(png, tiff));
  const auto back = find_exif_blob(png);
  ASSERT_TRUE(back.size() == tiff.size());
  EXPECT_TRUE(std::memcmp(back.data(), tiff.data(), tiff.size()) == 0);
  // eXIf must follow IHDR: sig(8) + IHDR(12+13) = 33.
  EXPECT_TRUE(std::memcmp(png.data() + 33 + 4, "eXIf", 4) == 0);
}

// Re-saving an already-annotated file must REPLACE, not append -- two Exif
// blocks in one file is malformed and readers disagree about which wins.
TEST(image_metadata, second_write_replaces_rather_than_appends) {
  const auto tiff = make_tiff_();
  auto jpeg = make_jpeg_();
  ASSERT_TRUE(jpeg_set_exif(jpeg, tiff));
  const std::size_t after_one = jpeg.size();
  ASSERT_TRUE(jpeg_set_exif(jpeg, tiff));
  EXPECT_TRUE(jpeg.size() == after_one);
  EXPECT_TRUE(count_bytes_(jpeg, "Exif", 4) == 1);

  auto png = make_png_();
  ASSERT_TRUE(png_set_exif(png, tiff));
  const std::size_t png_one = png.size();
  ASSERT_TRUE(png_set_exif(png, tiff));
  EXPECT_TRUE(png.size() == png_one);
  EXPECT_TRUE(count_bytes_(png, "eXIf", 4) == 1);
}

TEST(image_metadata, no_exif_reads_empty) {
  EXPECT_TRUE(find_exif_blob(make_jpeg_()).empty());
  EXPECT_TRUE(find_exif_blob(make_png_()).empty());
  EXPECT_TRUE(find_exif_blob(std::vector<std::uint8_t>{}).empty());
  // Not a container this module scans.
  std::vector<std::uint8_t> bmp = {'B', 'M', 0, 0, 0, 0, 0, 0};
  EXPECT_TRUE(find_exif_blob(bmp).empty());
}

TEST(image_metadata, writers_refuse_wrong_container) {
  const auto tiff = make_tiff_();
  auto png = make_png_();
  auto jpeg = make_jpeg_();
  EXPECT_FALSE(jpeg_set_exif(png, tiff));    // PNG bytes, JPEG writer
  EXPECT_FALSE(png_set_exif(jpeg, tiff));    // JPEG bytes, PNG writer
  // An empty EXIF block is a no-op failure, not a corrupt write.
  EXPECT_FALSE(jpeg_set_exif(jpeg, std::vector<std::uint8_t>{}));
}

// An APP1 payload is length-limited to 65533 bytes; an oversized block must
// be refused rather than silently truncated into a corrupt segment.
TEST(image_metadata, jpeg_refuses_oversized_exif) {
  auto jpeg = make_jpeg_();
  std::vector<std::uint8_t> big(70000, 0);
  big[0] = 'I'; big[1] = 'I';
  EXPECT_FALSE(jpeg_set_exif(jpeg, big));
  EXPECT_TRUE(jpeg.size() == make_jpeg_().size());   // untouched
}

TEST(image_metadata, format_support_matches_the_writers) {
  EXPECT_TRUE(format_supports_exif("jpeg"));
  EXPECT_TRUE(format_supports_exif("jpg"));
  EXPECT_TRUE(format_supports_exif("png"));
  EXPECT_FALSE(format_supports_exif("bmp"));
  EXPECT_FALSE(format_supports_exif("webp"));
  EXPECT_TRUE(format_supports_exif("tiff"));
  EXPECT_TRUE(format_supports_exif("tif"));
}

namespace {

// A minimal but STRUCTURALLY REAL TIFF file: header, IFD0 with the pixel
// tags a decoder needs (including an out-of-line StripOffsets array), a
// Software tag, and the pixel bytes those offsets point at. Built in either
// byte order so the relocation path can be exercised across a mismatch.
std::vector<std::uint8_t>
make_tiff_file_(bool le)
{
  std::vector<std::uint8_t> f;
  auto u16 = [&](std::uint16_t v) {
    if (le) {
      f.push_back((std::uint8_t)v);
      f.push_back((std::uint8_t)(v >> 8));
    } else {
      f.push_back((std::uint8_t)(v >> 8));
      f.push_back((std::uint8_t)v);
    }
  };
  auto u32 = [&](std::uint32_t v) {
    if (le) {
      f.push_back((std::uint8_t)v);         f.push_back((std::uint8_t)(v >> 8));
      f.push_back((std::uint8_t)(v >> 16));
      f.push_back((std::uint8_t)(v >> 24));
    } else {
      f.push_back((std::uint8_t)(v >> 24));
      f.push_back((std::uint8_t)(v >> 16));
      f.push_back((std::uint8_t)(v >> 8));  f.push_back((std::uint8_t)v);
    }
  };
  f.push_back(le ? 'I' : 'M'); f.push_back(le ? 'I' : 'M');
  u16(42);
  u32(8);                                  // IFD0 immediately after header
  const std::uint16_t n = 4;
  u16(n);
  // 8 + 2 + 4*12 + 4 = 62 -> values start there.
  const std::uint32_t vbase = 62;
  u16(0x0100); u16(4); u32(1); u32(4);                 // ImageWidth = 4
  u16(0x0101); u16(4); u32(1); u32(4);                 // ImageLength = 4
  u16(0x0111); u16(4); u32(2); u32(vbase);             // StripOffsets[2]
  u16(0x0131); u16(2); u32(5); u32(vbase + 8);         // Software "orig"
  u32(0);                                              // no next IFD
  // Values: two strip offsets pointing at the pixel bytes, then "orig\0".
  u32(vbase + 14); u32(vbase + 16);
  const char* sw = "orig";
  f.insert(f.end(), sw, sw + 5);
  f.push_back(0);                                      // pad to even
  f.push_back(0xAA); f.push_back(0xBB);                // strip 0
  f.push_back(0xCC); f.push_back(0xDD);                // strip 1
  return f;
}

}  // namespace

// A TIFF has no separate EXIF block, so reading one must SYNTHESIZE a
// standalone block -- carrying the descriptive tags but none of the
// pixel-layout tags, which would be meaningless on another image.
TEST(image_metadata, tiff_source_synthesizes_a_block) {
  const auto file = make_tiff_file_(true);
  const auto blob = find_exif_blob(file);
  ASSERT_TRUE(!blob.empty());
  FlexData d = parse_exif(blob);
  ASSERT_TRUE(d.is_object());
  auto o = d.as_object();
  EXPECT_TRUE(std::string(o.at("Software").as_string("")) == "orig");
  // Pixel-layout tags must NOT ride along.
  EXPECT_FALSE(o.contains("ImageWidth"));
  EXPECT_FALSE(o.contains("Tag0x0111"));      // StripOffsets
}

TEST(image_metadata, tiff_merge_adds_tags_and_keeps_the_image_intact) {
  auto file = make_tiff_file_(true);
  const auto before = file;
  ASSERT_TRUE(tiff_set_exif(file, make_tiff_()));

  // Nothing already in the file may move: the merged IFD is APPENDED and only
  // the header's 4-byte first-IFD offset is rewritten. Everything the strips
  // point at must be byte-identical.
  ASSERT_TRUE(file.size() > before.size());
  EXPECT_TRUE(std::memcmp(file.data() + 8, before.data() + 8,
                          before.size() - 8) == 0);

  FlexData d = parse_exif(file);   // a TIFF file IS a TIFF block
  ASSERT_TRUE(d.is_object());
  auto o = d.as_object();
  EXPECT_TRUE(std::string(o.at("Make").as_string("")) == "VPIPE");
  EXPECT_TRUE(o.at("Orientation").as_int(-1) == 1);
  // The file's own tags survive the merge.
  EXPECT_TRUE(std::string(o.at("Software").as_string("")) == "orig");
  EXPECT_TRUE(o.at("ImageWidth").as_int(-1) == 4);
}

// A little-endian EXIF block merged into a BIG-endian TIFF: every relocated
// value has to be byte-swapped on the way, or the tags read as garbage.
TEST(image_metadata, tiff_merge_converts_byte_order) {
  auto file = make_tiff_file_(false);          // MM target
  ASSERT_TRUE(file[0] == 'M');
  ASSERT_TRUE(tiff_set_exif(file, make_tiff_()));   // II source
  FlexData d = parse_exif(file);
  ASSERT_TRUE(d.is_object());
  auto o = d.as_object();
  EXPECT_TRUE(std::string(o.at("Make").as_string("")) == "VPIPE");
  EXPECT_TRUE(o.at("Orientation").as_int(-1) == 1);
  EXPECT_TRUE(o.at("ImageWidth").as_int(-1) == 4);
  EXPECT_TRUE(std::string(o.at("Software").as_string("")) == "orig");
}

// The file's own value wins on a conflict -- importing the source's Software
// over the encoder's would misreport what wrote the file.
TEST(image_metadata, tiff_merge_does_not_overwrite_existing_tags) {
  auto file = make_tiff_file_(true);
  // Source block whose only tag collides with one the file already has.
  auto src = make_tiff_();
  ASSERT_TRUE(tiff_set_exif(file, src));
  FlexData d = parse_exif(file);   // as_object() is a VIEW: keep the owner
  auto o = d.as_object();
  EXPECT_TRUE(std::string(o.at("Software").as_string("")) == "orig");
}

TEST(image_metadata, tiff_refuses_bigtiff_and_non_tiff) {
  auto big = make_tiff_file_(true);
  big[2] = 43; big[3] = 0;                 // version 43 = BigTIFF
  EXPECT_FALSE(tiff_set_exif(big, make_tiff_()));
  auto jpeg = make_jpeg_();
  EXPECT_FALSE(tiff_set_exif(jpeg, make_tiff_()));
  EXPECT_TRUE(find_exif_blob(big).empty());
}

TEST(image_metadata, software_tag_on_an_empty_base) {
  const auto blk = exif_set_software({}, "Vpipe 0.1 abc*1 with a/model");
  ASSERT_TRUE(!blk.empty());
  FlexData d = parse_exif(blk);
  ASSERT_TRUE(d.is_object());
  auto o = d.as_object();
  EXPECT_TRUE(std::string(o.at("Software").as_string(""))
              == "Vpipe 0.1 abc*1 with a/model");
}

// Setting Software must not cost any other tag -- a camera block keeps
// everything it had, with only that one field rewritten.
TEST(image_metadata, software_tag_preserves_the_other_tags) {
  const auto base = make_tiff_();          // Make=VPIPE, Orientation=1
  const auto blk = exif_set_software(base, "written-by-vpipe");
  ASSERT_TRUE(!blk.empty());
  FlexData d = parse_exif(blk);
  ASSERT_TRUE(d.is_object());
  auto o = d.as_object();
  EXPECT_TRUE(std::string(o.at("Make").as_string("")) == "VPIPE");
  EXPECT_TRUE(o.at("Orientation").as_int(-1) == 1);
  EXPECT_TRUE(std::string(o.at("Software").as_string(""))
              == "written-by-vpipe");
}

// Replacing, not appending: a block that already has Software ends up with
// exactly one.
TEST(image_metadata, software_tag_replaces_an_existing_one) {
  const auto once  = exif_set_software({}, "first");
  const auto twice = exif_set_software(once, "second");
  FlexData d = parse_exif(twice);
  ASSERT_TRUE(d.is_object());
  auto o = d.as_object();
  EXPECT_TRUE(std::string(o.at("Software").as_string("")) == "second");
  // One IFD entry per tag: the block must not have grown a duplicate. Compare
  // the IFD entry COUNT (byte size is a bad proxy -- values are padded to even
  // offsets, so a one-character-longer string need not add one byte).
  auto entry_count = [](const std::vector<std::uint8_t>& b) {
    return (unsigned)b[8] | ((unsigned)b[9] << 8);   // little-endian count
  };
  EXPECT_TRUE(entry_count(once) == 1u);
  EXPECT_TRUE(entry_count(twice) == 1u);
}

TEST(image_metadata, empty_software_is_a_no_op) {
  const auto base = make_tiff_();
  const auto blk = exif_set_software(base, "");
  ASSERT_TRUE(blk.size() == base.size());
  EXPECT_TRUE(std::memcmp(blk.data(), base.data(), base.size()) == 0);
}

// A developed camera RAW comes out upright, so the EXIF carried with it
// is set to Orientation 1 -- in place, every other tag untouched, in
// either byte order.
TEST(image_metadata, orientation_is_set_in_place) {
  auto t = make_tiff_();
  const std::size_t size = t.size();
  EXPECT_TRUE(exif_set_orientation(t, 8));
  EXPECT_TRUE(t.size() == size);
  FlexData p = parse_exif(t);
  ASSERT_TRUE(p.is_object());
  if (!p.is_object()) { return; }
  EXPECT_TRUE(p.as_object().at("Orientation").as_uint() == 8);
  EXPECT_TRUE(std::string(p.as_object().at("Make").as_string()) == "VPIPE");
  EXPECT_TRUE(exif_set_orientation(t, 1));
  EXPECT_TRUE(parse_exif(t).as_object().at("Orientation").as_uint() == 1);

  // Big-endian: one IFD holding only Orientation = 6.
  std::vector<std::uint8_t> be = {
    'M', 'M', 0, 42, 0, 0, 0, 8,
    0, 1,                                  // one entry
    0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, 6, 0, 0,
    0, 0, 0, 0,                            // no next IFD
  };
  EXPECT_TRUE(exif_set_orientation(be, 1));
  EXPECT_TRUE(parse_exif(be).as_object().at("Orientation").as_uint() == 1);

  // Nothing to set: no Orientation entry, or not a TIFF block.
  std::vector<std::uint8_t> none = {
    'I', 'I', 42, 0, 8, 0, 0, 0,
    0, 0,                                  // no entries
    0, 0, 0, 0,
  };
  EXPECT_FALSE(exif_set_orientation(none, 1));
  std::vector<std::uint8_t> junk = {1, 2, 3};
  EXPECT_FALSE(exif_set_orientation(junk, 1));
}

// The Interoperability IFD (0xA005, inside the Exif IFD) is a pointer
// like the Exif and GPS ones, and must be relocated with the block.
// Copied as a plain LONG it kept its old offset -- into whatever the new
// block held there -- which a Canon CR2's EXIF did on its way into a
// JPEG, and which most camera JPEGs (they carry one) did through the
// Software stamp.
namespace {

// IFD0 {Make "VPIPE", ExifIFD} -> Exif {ISO 100, InteropIFD} -> Interop
// {InteroperabilityIndex "R98"}, little-endian. The Interop IFD sits
// BEFORE the Exif IFD, so an offset kept from here lands elsewhere in any
// re-laid-out block:
//    0  header            8  IFD0 (2 entries)    38  "VPIPE\0"
//   44  Interop IFD (1)  62  pad                 64  Exif IFD (2)   94 end
std::vector<std::uint8_t>
make_interop_tiff_()
{
  std::vector<std::uint8_t> t(94, 0);
  auto put16 = [&](std::size_t o, std::uint16_t v) {
    t[o] = (std::uint8_t)v; t[o + 1] = (std::uint8_t)(v >> 8);
  };
  auto put32 = [&](std::size_t o, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) { t[o + i] = (std::uint8_t)(v >> (8 * i)); }
  };
  auto entry = [&](std::size_t o, std::uint16_t tag, std::uint16_t type,
                   std::uint32_t count, std::uint32_t value) {
    put16(o, tag); put16(o + 2, type); put32(o + 4, count);
    put32(o + 8, value);
  };
  t[0] = 'I'; t[1] = 'I'; put16(2, 42); put32(4, 8);
  put16(8, 2);
  entry(10, 0x010F, 2, 6, 38);              // Make -> "VPIPE\0"
  entry(22, 0x8769, 4, 1, 64);              // ExifIFD
  put32(34, 0);
  std::memcpy(t.data() + 38, "VPIPE", 6);
  put16(44, 1);
  entry(46, 0x0001, 2, 4, 0);               // InteroperabilityIndex,
  std::memcpy(t.data() + 54, "R98", 4);     // "R98\0" inline
  put32(58, 0);
  put16(64, 2);
  entry(66, 0x8827, 3, 1, 100);             // ISOSpeedRatings
  entry(78, 0xA005, 4, 1, 44);              // InteropIFD
  put32(90, 0);
  return t;
}

// Follow IFD0 -> Exif -> Interop in a block; the InteroperabilityIndex
// string, or empty when a pointer leads nowhere sensible.
std::string
interop_index_(const std::vector<std::uint8_t>& t)
{
  auto u16 = [&](std::size_t o) -> std::uint32_t {
    return o + 2 <= t.size() ? (std::uint32_t)(t[o] | (t[o + 1] << 8)) : 0;
  };
  auto u32 = [&](std::size_t o) -> std::uint32_t {
    if (o + 4 > t.size()) { return 0; }
    return (std::uint32_t)t[o] | ((std::uint32_t)t[o + 1] << 8) |
           ((std::uint32_t)t[o + 2] << 16) | ((std::uint32_t)t[o + 3] << 24);
  };
  auto find = [&](std::size_t ifd, std::uint16_t tag) -> std::size_t {
    const std::uint32_t n = u16(ifd);
    if (n == 0 || ifd + 2 + 12 * n > t.size()) { return 0; }
    for (std::uint32_t i = 0; i < n; ++i) {
      if (u16(ifd + 2 + 12 * i) == tag) { return ifd + 2 + 12 * i; }
    }
    return 0;
  };
  const std::size_t ex = find(u32(4), 0x8769);
  if (ex == 0) { return {}; }
  const std::size_t io = find(u32(ex + 8), 0xA005);
  if (io == 0) { return {}; }
  const std::size_t idx = find(u32(io + 8), 0x0001);
  if (idx == 0 || u32(idx + 4) != 4) { return {}; }
  return std::string(reinterpret_cast<const char*>(t.data() + idx + 8));
}

}  // namespace

TEST(image_metadata, interop_ifd_is_relocated) {
  const auto src = make_interop_tiff_();
  EXPECT_TRUE(interop_index_(src) == "R98");   // the fixture is sound
  const auto out = exif_set_software(src, "vpipe test");
  EXPECT_TRUE(out.size() != src.size());       // re-laid out, not copied
  std::printf("[image_metadata] interop index after relocation: '%s'\n",
              interop_index_(out).c_str());
  EXPECT_TRUE(interop_index_(out) == "R98");
  EXPECT_TRUE(std::string(parse_exif(out).as_object().at("Software")
                              .as_string("")) == "vpipe test");
}
