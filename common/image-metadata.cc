#include "common/image-metadata.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace vpipe {
namespace imgmeta {

namespace {

constexpr std::uint8_t kPngSig[8] =
    {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};

bool
is_jpeg_(std::span<const std::uint8_t> f)
{
  return f.size() >= 2 && f[0] == 0xFF && f[1] == 0xD8;
}

bool
is_png_(std::span<const std::uint8_t> f)
{
  return f.size() >= 8 && std::memcmp(f.data(), kPngSig, 8) == 0;
}

std::uint32_t
be32_(const std::uint8_t* p)
{
  return (std::uint32_t)p[0] << 24 | (std::uint32_t)p[1] << 16
       | (std::uint32_t)p[2] << 8  | (std::uint32_t)p[3];
}

void
put_be32_(std::vector<std::uint8_t>& v, std::uint32_t x)
{
  v.push_back((std::uint8_t)(x >> 24));
  v.push_back((std::uint8_t)(x >> 16));
  v.push_back((std::uint8_t)(x >> 8));
  v.push_back((std::uint8_t)x);
}

// PNG chunk CRC-32 (IEEE 802.3, the same polynomial zlib uses). Implemented
// here so the module needs no zlib link just to stamp one chunk.
std::uint32_t
crc32_(std::span<const std::uint8_t> data)
{
  static std::uint32_t table[256];
  static bool built = false;
  if (!built) {
    for (std::uint32_t n = 0; n < 256; ++n) {
      std::uint32_t c = n;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      table[n] = c;
    }
    built = true;
  }
  std::uint32_t c = 0xFFFFFFFFu;
  for (std::uint8_t b : data) {
    c = table[(c ^ b) & 0xFF] ^ (c >> 8);
  }
  return c ^ 0xFFFFFFFFu;
}

// ---- TIFF/EXIF reader ---------------------------------------------------

struct Tiff {
  std::span<const std::uint8_t> d;
  bool little = true;

  std::uint16_t u16(std::size_t o) const
  {
    if (o + 2 > d.size()) { return 0; }
    return little ? (std::uint16_t)(d[o] | (d[o + 1] << 8))
                  : (std::uint16_t)((d[o] << 8) | d[o + 1]);
  }
  std::uint32_t u32(std::size_t o) const
  {
    if (o + 4 > d.size()) { return 0; }
    return little
        ? ((std::uint32_t)d[o] | ((std::uint32_t)d[o + 1] << 8)
           | ((std::uint32_t)d[o + 2] << 16) | ((std::uint32_t)d[o + 3] << 24))
        : (((std::uint32_t)d[o] << 24) | ((std::uint32_t)d[o + 1] << 16)
           | ((std::uint32_t)d[o + 2] << 8) | (std::uint32_t)d[o + 3]);
  }
};

// Byte width of each TIFF field type, indexed by type code (0 = unknown).
constexpr std::size_t kTypeSize[13] =
    {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8};

struct TagName { std::uint16_t tag; const char* name; };

// The tags worth naming. Anything absent still appears, as "Tag0xNNNN" --
// the reader never drops a field it cannot name.
constexpr TagName kIfdTags[] = {
  {0x010E, "ImageDescription"},   {0x010F, "Make"},
  {0x0110, "Model"},              {0x0112, "Orientation"},
  {0x011A, "XResolution"},        {0x011B, "YResolution"},
  {0x0128, "ResolutionUnit"},     {0x0131, "Software"},
  {0x0132, "DateTime"},           {0x013B, "Artist"},
  {0x8298, "Copyright"},          {0x0100, "ImageWidth"},
  {0x0101, "ImageLength"},        {0x9003, "DateTimeOriginal"},
  {0x9004, "DateTimeDigitized"},  {0x829A, "ExposureTime"},
  {0x829D, "FNumber"},            {0x8827, "ISOSpeedRatings"},
  {0x8822, "ExposureProgram"},    {0x9201, "ShutterSpeedValue"},
  {0x9202, "ApertureValue"},      {0x9204, "ExposureBiasValue"},
  {0x9207, "MeteringMode"},       {0x9209, "Flash"},
  {0x920A, "FocalLength"},        {0xA402, "ExposureMode"},
  {0xA403, "WhiteBalance"},       {0xA405, "FocalLengthIn35mmFilm"},
  {0xA406, "SceneCaptureType"},   {0xA002, "PixelXDimension"},
  {0xA003, "PixelYDimension"},    {0x9286, "UserComment"},
  {0x0001, "GPSLatitudeRef"},     {0x0002, "GPSLatitude"},
  {0x0003, "GPSLongitudeRef"},    {0x0004, "GPSLongitude"},
  {0x0005, "GPSAltitudeRef"},     {0x0006, "GPSAltitude"},
  {0x001D, "GPSDateStamp"},
};

const char*
tag_name_(std::uint16_t tag, bool gps)
{
  // GPS tag numbers overlap IFD0's, so only consult the GPS entries when
  // walking the GPS IFD (and vice versa) -- otherwise GPSLatitudeRef would
  // shadow ImageWidth.
  for (const TagName& t : kIfdTags) {
    const bool is_gps_tag = (t.tag <= 0x001D);
    if (is_gps_tag != gps) { continue; }
    if (t.tag == tag) { return t.name; }
  }
  return nullptr;
}

FlexData
read_value_(const Tiff& t, std::uint16_t type, std::uint32_t count,
            std::size_t value_off)
{
  const std::size_t esz = (type < 13) ? kTypeSize[type] : 0;
  if (esz == 0 || count == 0) { return FlexData::make_null(); }
  const std::size_t total = esz * (std::size_t)count;
  if (value_off + total > t.d.size()) { return FlexData::make_null(); }

  // ASCII (2) and UNDEFINED (7) read as text; UNDEFINED is where UserComment
  // and maker blobs live, so keep only its printable prefix rather than
  // spraying control bytes into the output.
  if (type == 2 || type == 7) {
    std::string s;
    s.reserve(total);
    for (std::size_t i = 0; i < total; ++i) {
      const char c = (char)t.d[value_off + i];
      if (c == '\0') { break; }
      if ((unsigned char)c >= 0x20 || c == '\t') { s.push_back(c); }
    }
    while (!s.empty() && s.back() == ' ') { s.pop_back(); }
    return FlexData::make_string(s);
  }

  auto one = [&](std::size_t i) -> FlexData {
    const std::size_t o = value_off + i * esz;
    switch (type) {
      case 1: return FlexData::make_uint(t.d[o]);
      case 3: return FlexData::make_uint(t.u16(o));
      case 4: return FlexData::make_uint(t.u32(o));
      case 9: return FlexData::make_int((std::int32_t)t.u32(o));
      case 5: case 10: {
        const std::uint32_t n = t.u32(o), d = t.u32(o + 4);
        if (d == 0) { return FlexData::make_real(0.0); }
        const double num = (type == 5) ? (double)n : (double)(std::int32_t)n;
        const double den = (type == 5) ? (double)d : (double)(std::int32_t)d;
        return FlexData::make_real(den == 0.0 ? 0.0 : num / den);
      }
      default: return FlexData::make_null();
    }
  };
  if (count == 1) { return one(0); }
  FlexData arr = FlexData::make_array();
  for (std::uint32_t i = 0; i < count && i < 64; ++i) {
    arr.as_array().push_back(one(i));
  }
  return arr;
}

// Walk one IFD, inserting its fields into `out`. `sub_exif` / `sub_gps`
// receive the offsets of the Exif / GPS sub-IFDs when this IFD points at
// them. Bounded by `depth` so a malformed chain cannot loop forever.
void
walk_ifd_(const Tiff& t, std::size_t off, bool gps, FlexData& out,
          std::uint32_t* sub_exif, std::uint32_t* sub_gps)
{
  if (off + 2 > t.d.size()) { return; }
  const std::uint16_t n = t.u16(off);
  // 12 bytes per entry; a count that runs past the block is corrupt.
  if (off + 2 + (std::size_t)n * 12 > t.d.size()) { return; }
  for (std::uint16_t i = 0; i < n; ++i) {
    const std::size_t e = off + 2 + (std::size_t)i * 12;
    const std::uint16_t tag   = t.u16(e);
    const std::uint16_t type  = t.u16(e + 2);
    const std::uint32_t count = t.u32(e + 4);
    if (tag == 0x8769 && sub_exif) { *sub_exif = t.u32(e + 8); continue; }
    if (tag == 0x8825 && sub_gps)  { *sub_gps  = t.u32(e + 8); continue; }

    const std::size_t esz = (type < 13) ? kTypeSize[type] : 0;
    if (esz == 0) { continue; }
    const std::size_t total = esz * (std::size_t)count;
    // <= 4 bytes live inline in the entry; longer values are at an offset
    // from the start of the TIFF block.
    const std::size_t vo = (total <= 4) ? (e + 8) : t.u32(e + 8);
    FlexData v = read_value_(t, type, count, vo);
    if (v.is_null()) { continue; }
    const char* nm = tag_name_(tag, gps);
    char fallback[16];
    if (nm == nullptr) {
      std::snprintf(fallback, sizeof(fallback), "Tag0x%04X", tag);
      nm = fallback;
    }
    out.as_object().insert_or_assign(nm, std::move(v));
  }
}

// ---- TIFF IFD relocation ------------------------------------------------
//
// Shared by both TIFF directions: synthesizing a standalone EXIF block FROM a
// TIFF file, and merging a block INTO one. Both need the same thing -- copy
// an IFD's entries into another buffer, moving every out-of-line value and
// rewriting its offset, converting byte order on the way.

// Tags that describe how THIS file stores its pixels. They are meaningful
// only for the image they came from: copied onto another image they range
// from wrong (Compression) to fatal (StripOffsets points the reader at bytes
// that are not there). Never copied in either direction.
bool
is_structural_tag_(std::uint16_t t)
{
  switch (t) {
    case 0x00FE: case 0x00FF:            // NewSubfileType / SubfileType
    case 0x0100: case 0x0101:            // ImageWidth / ImageLength
    case 0x0102: case 0x0103:            // BitsPerSample / Compression
    case 0x0106: case 0x0107:            // Photometric / Thresholding
    case 0x0111: case 0x0115: case 0x0116: case 0x0117:  // strips
    case 0x011C:                         // PlanarConfiguration
    case 0x0140: case 0x0142: case 0x0143:               // colormap / tiles
    case 0x0144: case 0x0145:
    case 0x014A:                         // SubIFDs
    case 0x0201: case 0x0202:            // JPEGInterchangeFormat(+Length)
      return true;
    default:
      return false;
  }
}

void
put16_(std::vector<std::uint8_t>& d, std::uint16_t v, bool le)
{
  if (le) { d.push_back((std::uint8_t)v); d.push_back((std::uint8_t)(v >> 8)); }
  else    { d.push_back((std::uint8_t)(v >> 8)); d.push_back((std::uint8_t)v); }
}

void
put32_(std::vector<std::uint8_t>& d, std::uint32_t v, bool le)
{
  if (le) {
    d.push_back((std::uint8_t)v);         d.push_back((std::uint8_t)(v >> 8));
    d.push_back((std::uint8_t)(v >> 16)); d.push_back((std::uint8_t)(v >> 24));
  } else {
    d.push_back((std::uint8_t)(v >> 24)); d.push_back((std::uint8_t)(v >> 16));
    d.push_back((std::uint8_t)(v >> 8));  d.push_back((std::uint8_t)v);
  }
}

// One entry's worth of value bytes, converted from src order to dst order.
// Byte-typed data (BYTE / ASCII / UNDEFINED / SBYTE) is order-agnostic and
// copies verbatim; wider types are swapped element by element, and RATIONALs
// as their two independent LONGs.
void
append_value_(const Tiff& src, std::size_t off, std::uint16_t type,
              std::uint32_t count, std::vector<std::uint8_t>& dst, bool dst_le)
{
  const std::size_t esz = (type < 13) ? kTypeSize[type] : 0;
  if (esz == 0) { return; }
  if (type == 1 || type == 2 || type == 6 || type == 7) {
    for (std::uint32_t i = 0; i < count; ++i) {
      dst.push_back(off + i < src.d.size() ? src.d[off + i] : 0);
    }
    return;
  }
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::size_t o = off + (std::size_t)i * esz;
    switch (type) {
      case 3: case 8:  put16_(dst, src.u16(o), dst_le); break;
      case 4: case 9: case 11: put32_(dst, src.u32(o), dst_le); break;
      case 5: case 10:
        put32_(dst, src.u32(o), dst_le);
        put32_(dst, src.u32(o + 4), dst_le);
        break;
      case 12:
        put32_(dst, src.u32(o), dst_le);
        put32_(dst, src.u32(o + 4), dst_le);
        break;
      default: break;
    }
  }
}

struct OutEntry {
  std::uint16_t tag = 0;
  std::uint16_t type = 0;
  std::uint32_t count = 0;
  std::uint8_t  inline_bytes[4] = {0, 0, 0, 0};
  std::uint32_t offset = 0;     // used when !is_inline
  bool          is_inline = false;
};

// Copy `src`'s IFD at `src_off` into `dst`, appending relocated values (and
// recursively any Exif / GPS sub-IFD). Fills `out`; returns false on a
// malformed IFD. `depth` bounds a cyclic chain.
bool
collect_ifd_(const Tiff& src, std::size_t src_off,
             std::vector<std::uint8_t>& dst, bool dst_le,
             bool skip_structural, std::vector<OutEntry>& out, int depth)
{
  if (depth > 4 || src_off + 2 > src.d.size()) { return false; }
  const std::uint16_t n = src.u16(src_off);
  if (src_off + 2 + (std::size_t)n * 12 > src.d.size()) { return false; }

  for (std::uint16_t i = 0; i < n; ++i) {
    const std::size_t e = src_off + 2 + (std::size_t)i * 12;
    const std::uint16_t tag   = src.u16(e);
    const std::uint16_t type  = src.u16(e + 2);
    const std::uint32_t count = src.u32(e + 4);
    const std::size_t   esz   = (type < 13) ? kTypeSize[type] : 0;
    if (esz == 0 || count == 0) { continue; }
    if (skip_structural && is_structural_tag_(tag)) { continue; }

    OutEntry oe;
    oe.tag = tag; oe.type = type; oe.count = count;

    if (tag == 0x8769 || tag == 0x8825) {
      // Sub-IFD: copy it wholesale (never structural-filtered -- an Exif IFD
      // has no pixel-layout tags) and point at its new home.
      std::vector<OutEntry> sub;
      if (!collect_ifd_(src, src.u32(e + 8), dst, dst_le,
                        /*skip_structural=*/false, sub, depth + 1)) {
        continue;
      }
      if (dst.size() & 1u) { dst.push_back(0); }
      const std::uint32_t sub_off = (std::uint32_t)dst.size();
      put16_(dst, (std::uint16_t)sub.size(), dst_le);
      for (const OutEntry& s : sub) {
        put16_(dst, s.tag, dst_le);
        put16_(dst, s.type, dst_le);
        put32_(dst, s.count, dst_le);
        if (s.is_inline) {
          dst.insert(dst.end(), s.inline_bytes, s.inline_bytes + 4);
        } else {
          put32_(dst, s.offset, dst_le);
        }
      }
      put32_(dst, 0, dst_le);              // sub-IFDs terminate the chain
      oe.type = 4; oe.count = 1;
      oe.offset = sub_off;
      oe.is_inline = false;
      // The pointer VALUE is 4 bytes, so it lives inline in the entry.
      std::vector<std::uint8_t> tmp;
      put32_(tmp, sub_off, dst_le);
      std::memcpy(oe.inline_bytes, tmp.data(), 4);
      oe.is_inline = true;
      out.push_back(oe);
      continue;
    }

    const std::size_t total = esz * (std::size_t)count;
    const std::size_t vo = (total <= 4) ? (e + 8) : src.u32(e + 8);
    if (vo + total > src.d.size()) { continue; }   // corrupt -> drop the tag
    if (total <= 4) {
      // Inline: re-emit through the same converter so the 4 bytes land in the
      // destination's order.
      std::vector<std::uint8_t> tmp;
      append_value_(src, vo, type, count, tmp, dst_le);
      tmp.resize(4, 0);
      std::memcpy(oe.inline_bytes, tmp.data(), 4);
      oe.is_inline = true;
    } else {
      if (dst.size() & 1u) { dst.push_back(0); }   // keep values word-aligned
      oe.offset = (std::uint32_t)dst.size();
      oe.is_inline = false;
      append_value_(src, vo, type, count, dst, dst_le);
    }
    out.push_back(oe);
  }
  return true;
}

void
write_entry_(std::vector<std::uint8_t>& dst, const OutEntry& e, bool le)
{
  put16_(dst, e.tag, le);
  put16_(dst, e.type, le);
  put32_(dst, e.count, le);
  if (e.is_inline) {
    dst.insert(dst.end(), e.inline_bytes, e.inline_bytes + 4);
  } else {
    put32_(dst, e.offset, le);
  }
}

bool
tiff_header_(std::span<const std::uint8_t> f, bool* le, std::uint32_t* ifd0)
{
  if (f.size() < 8) { return false; }
  if (f[0] == 'I' && f[1] == 'I')      { *le = true; }
  else if (f[0] == 'M' && f[1] == 'M') { *le = false; }
  else { return false; }
  Tiff t{f, *le};
  if (t.u16(2) != 42) { return false; }   // 43 = BigTIFF: a different format
  *ifd0 = t.u32(4);
  return *ifd0 >= 8 && *ifd0 < f.size();
}

}  // namespace

std::vector<std::uint8_t>
find_exif_blob(std::span<const std::uint8_t> f)
{
  static const std::uint8_t kExifId[6] = {'E', 'x', 'i', 'f', 0, 0};
  if (is_jpeg_(f)) {
    // Walk the marker segments from just past SOI. Every marker is
    // 0xFF <code>; APPn / most others carry a 2-byte big-endian length that
    // INCLUDES those two bytes. Stop at SOS (0xDA) -- entropy-coded data
    // follows and is not marker-scannable.
    std::size_t p = 2;
    while (p + 4 <= f.size()) {
      if (f[p] != 0xFF) { break; }
      const std::uint8_t m = f[p + 1];
      if (m == 0xD8 || (m >= 0xD0 && m <= 0xD9)) { p += 2; continue; }
      if (m == 0xDA) { break; }
      const std::size_t len = ((std::size_t)f[p + 2] << 8) | f[p + 3];
      if (len < 2 || p + 2 + len > f.size()) { break; }
      if (m == 0xE1 && len >= 2 + 6
          && std::memcmp(f.data() + p + 4, kExifId, 6) == 0) {
        const std::size_t start = p + 4 + 6;
        const std::size_t n     = len - 2 - 6;
        return std::vector<std::uint8_t>(f.begin() + (long)start,
                                         f.begin() + (long)(start + n));
      }
      p += 2 + len;
    }
    return {};
  }
  if (is_png_(f)) {
    std::size_t p = 8;
    while (p + 8 <= f.size()) {
      const std::uint32_t len = be32_(f.data() + p);
      if (p + 12 + (std::size_t)len > f.size()) { break; }
      const char* type = (const char*)f.data() + p + 4;
      if (std::memcmp(type, "eXIf", 4) == 0) {
        return std::vector<std::uint8_t>(
            f.begin() + (long)(p + 8), f.begin() + (long)(p + 8 + len));
      }
      if (std::memcmp(type, "IEND", 4) == 0) { break; }
      p += 12 + (std::size_t)len;   // len + type + data + crc
    }
    return {};
  }
  {
    // TIFF: there is no separate block to lift out -- the file's own IFD0 IS
    // the metadata. Synthesize a compact standalone block from it, dropping
    // the pixel-layout tags (which describe THIS file and would be wrong
    // attached to another image) and relocating everything else.
    bool le = true;
    std::uint32_t ifd0 = 0;
    if (!tiff_header_(f, &le, &ifd0)) { return {}; }
    const Tiff src{f, le};
    std::vector<std::uint8_t> out;
    out.push_back(le ? 'I' : 'M');
    out.push_back(le ? 'I' : 'M');
    put16_(out, 42, le);
    put32_(out, 8, le);                   // IFD0 goes right after the header
    // Values are appended first, so reserve the IFD's own span: it is
    // 2 + 12*n + 4 bytes and must start at offset 8. Collect entries into a
    // scratch buffer, then lay the block out for real.
    std::vector<OutEntry> ents;
    std::vector<std::uint8_t> scratch(8, 0);   // mirrors `out`'s header size
    if (!collect_ifd_(src, ifd0, scratch, le, /*skip_structural=*/true,
                      ents, 0) || ents.empty()) {
      return {};
    }
    // Re-run with values placed after the IFD, now that its size is known.
    const std::size_t ifd_bytes = 2 + ents.size() * 12 + 4;
    std::vector<std::uint8_t> body(8 + ifd_bytes, 0);
    ents.clear();
    if (!collect_ifd_(src, ifd0, body, le, /*skip_structural=*/true,
                      ents, 0) || ents.empty()) {
      return {};
    }
    if (2 + ents.size() * 12 + 4 != ifd_bytes) { return {}; }   // shape moved
    out.resize(8 + ifd_bytes, 0);
    out.insert(out.end(), body.begin() + (long)(8 + ifd_bytes), body.end());
    std::vector<std::uint8_t> ifd;
    put16_(ifd, (std::uint16_t)ents.size(), le);
    for (const OutEntry& e : ents) { write_entry_(ifd, e, le); }
    put32_(ifd, 0, le);
    std::memcpy(out.data() + 8, ifd.data(), ifd.size());
    return out;
  }
}

std::vector<std::uint8_t>
read_exif_blob(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { return {}; }
  // EXIF lives near the head of the file in both containers; a bounded read
  // keeps this cheap on a 50 MB raw-ish JPEG. 1 MiB comfortably covers an
  // APP1 (max 64 KiB) plus any preceding segments.
  std::vector<std::uint8_t> head(1u << 20);
  in.read(reinterpret_cast<char*>(head.data()), (long)head.size());
  head.resize((std::size_t)in.gcount());
  return find_exif_blob(head);
}

FlexData
parse_exif(std::span<const std::uint8_t> tiff)
{
  if (tiff.size() < 8) { return FlexData::make_null(); }
  Tiff t{tiff, true};
  if (tiff[0] == 'I' && tiff[1] == 'I')      { t.little = true; }
  else if (tiff[0] == 'M' && tiff[1] == 'M') { t.little = false; }
  else { return FlexData::make_null(); }
  if (t.u16(2) != 42) { return FlexData::make_null(); }

  FlexData out = FlexData::make_object();
  std::uint32_t sub_exif = 0, sub_gps = 0;
  walk_ifd_(t, t.u32(4), /*gps=*/false, out, &sub_exif, &sub_gps);
  if (sub_exif > 0 && sub_exif < tiff.size()) {
    walk_ifd_(t, sub_exif, /*gps=*/false, out, nullptr, nullptr);
  }
  if (sub_gps > 0 && sub_gps < tiff.size()) {
    walk_ifd_(t, sub_gps, /*gps=*/true, out, nullptr, nullptr);
  }
  return out;
}

bool
jpeg_set_exif(std::vector<std::uint8_t>& jpeg,
              std::span<const std::uint8_t> tiff)
{
  if (!is_jpeg_(jpeg) || tiff.empty()) { return false; }
  // APP1 length field is 16-bit and counts itself + "Exif\0\0" + payload.
  if (tiff.size() + 8 > 0xFFFF) { return false; }

  // Drop an existing APP1 Exif first so a re-save cannot accumulate two.
  {
    static const std::uint8_t kExifId[6] = {'E', 'x', 'i', 'f', 0, 0};
    std::size_t p = 2;
    while (p + 4 <= jpeg.size()) {
      if (jpeg[p] != 0xFF) { break; }
      const std::uint8_t m = jpeg[p + 1];
      if (m == 0xD8 || (m >= 0xD0 && m <= 0xD9)) { p += 2; continue; }
      if (m == 0xDA) { break; }
      const std::size_t len = ((std::size_t)jpeg[p + 2] << 8) | jpeg[p + 3];
      if (len < 2 || p + 2 + len > jpeg.size()) { break; }
      if (m == 0xE1 && len >= 2 + 6
          && std::memcmp(jpeg.data() + p + 4, kExifId, 6) == 0) {
        jpeg.erase(jpeg.begin() + (long)p,
                   jpeg.begin() + (long)(p + 2 + len));
        break;
      }
      p += 2 + len;
    }
  }

  const std::size_t seg_len = 2 + 6 + tiff.size();   // len field + id + data
  std::vector<std::uint8_t> seg;
  seg.reserve(2 + seg_len);
  seg.push_back(0xFF); seg.push_back(0xE1);
  seg.push_back((std::uint8_t)(seg_len >> 8));
  seg.push_back((std::uint8_t)(seg_len & 0xFF));
  const char* id = "Exif";
  seg.insert(seg.end(), id, id + 4);
  seg.push_back(0); seg.push_back(0);
  seg.insert(seg.end(), tiff.begin(), tiff.end());

  jpeg.insert(jpeg.begin() + 2, seg.begin(), seg.end());
  return true;
}

bool
png_set_exif(std::vector<std::uint8_t>& png,
             std::span<const std::uint8_t> tiff)
{
  if (!is_png_(png) || tiff.empty()) { return false; }

  // Remove any existing eXIf, and find the end of IHDR to insert after.
  std::size_t insert_at = 0;
  {
    std::size_t p = 8;
    while (p + 12 <= png.size()) {
      const std::uint32_t len = be32_(png.data() + p);
      if (p + 12 + (std::size_t)len > png.size()) { break; }
      const char* type = (const char*)png.data() + p + 4;
      const std::size_t whole = 12 + (std::size_t)len;
      if (std::memcmp(type, "IHDR", 4) == 0) { insert_at = p + whole; }
      if (std::memcmp(type, "eXIf", 4) == 0) {
        png.erase(png.begin() + (long)p, png.begin() + (long)(p + whole));
        continue;                       // re-read at the same offset
      }
      if (std::memcmp(type, "IEND", 4) == 0) { break; }
      p += whole;
    }
  }
  if (insert_at == 0) { return false; }   // no IHDR -> not a usable PNG

  std::vector<std::uint8_t> chunk;
  chunk.reserve(12 + tiff.size());
  put_be32_(chunk, (std::uint32_t)tiff.size());
  const char* type = "eXIf";
  chunk.insert(chunk.end(), type, type + 4);
  chunk.insert(chunk.end(), tiff.begin(), tiff.end());
  // CRC covers the type + data, not the length.
  const std::uint32_t crc = crc32_(
      std::span<const std::uint8_t>(chunk.data() + 4, 4 + tiff.size()));
  put_be32_(chunk, crc);

  png.insert(png.begin() + (long)insert_at, chunk.begin(), chunk.end());
  return true;
}

std::vector<std::uint8_t>
exif_set_software(std::span<const std::uint8_t> base,
                  std::string_view              software)
{
  if (software.empty()) {
    return std::vector<std::uint8_t>(base.begin(), base.end());
  }
  // Follow the source block's byte order when there is one, so a big-endian
  // camera block stays big-endian rather than being silently rewritten.
  bool le = true;
  std::uint32_t base_ifd = 0;
  const bool have_base = tiff_header_(base, &le, &base_ifd);
  const Tiff src{base, le};

  // The tag's value: ASCII, NUL-terminated (count includes the NUL).
  std::vector<std::uint8_t> sw(software.begin(), software.end());
  sw.push_back(0);

  // Two passes: the first learns how many entries survive (and therefore how
  // big the IFD is), the second re-collects with the value area starting
  // after it, so every recorded offset is final.
  for (int pass = 0; pass < 2; ++pass) {
    std::vector<OutEntry> ents;
    std::vector<std::uint8_t> values;
    static thread_local std::size_t s_ifd_bytes = 0;
    const std::size_t vbase = (pass == 0) ? 0 : 8 + s_ifd_bytes;
    values.resize(vbase, 0);
    if (have_base) {
      collect_ifd_(src, base_ifd, values, le, /*skip_structural=*/false,
                   ents, 0);
      ents.erase(std::remove_if(ents.begin(), ents.end(),
                                [](const OutEntry& e) {
                                  return e.tag == 0x0131;
                                }),
                 ents.end());
    }
    OutEntry se;
    se.tag = 0x0131; se.type = 2; se.count = (std::uint32_t)sw.size();
    if (sw.size() <= 4) {
      se.is_inline = true;
      std::memcpy(se.inline_bytes, sw.data(), sw.size());
    } else {
      if (values.size() & 1u) { values.push_back(0); }
      se.offset = (std::uint32_t)values.size();
      se.is_inline = false;
      values.insert(values.end(), sw.begin(), sw.end());
    }
    ents.push_back(se);
    std::stable_sort(ents.begin(), ents.end(),
                     [](const OutEntry& a, const OutEntry& b) {
                       return a.tag < b.tag;
                     });
    const std::size_t ifd_bytes = 2 + ents.size() * 12 + 4;
    if (pass == 0) { s_ifd_bytes = ifd_bytes; continue; }
    if (ifd_bytes != s_ifd_bytes) { return {}; }   // shape moved: bail

    std::vector<std::uint8_t> out;
    out.push_back(le ? 'I' : 'M');
    out.push_back(le ? 'I' : 'M');
    put16_(out, 42, le);
    put32_(out, 8, le);
    put16_(out, (std::uint16_t)ents.size(), le);
    for (const OutEntry& e : ents) { write_entry_(out, e, le); }
    put32_(out, 0, le);
    out.insert(out.end(), values.begin() + (long)vbase, values.end());
    return out;
  }
  return {};
}

bool
tiff_set_exif(std::vector<std::uint8_t>& file,
              std::span<const std::uint8_t> tiff)
{
  bool dst_le = true;
  std::uint32_t ifd0 = 0;
  if (!tiff_header_(file, &dst_le, &ifd0) || tiff.empty()) { return false; }

  bool src_le = true;
  std::uint32_t src_ifd0 = 0;
  if (!tiff_header_(tiff, &src_le, &src_ifd0)) { return false; }

  const Tiff dst_t{file, dst_le};
  const std::uint16_t n_old = dst_t.u16(ifd0);
  if (ifd0 + 2 + (std::size_t)n_old * 12 + 4 > file.size()) { return false; }

  // Whatever the file already says about itself wins: only tags it does NOT
  // carry are imported.
  std::vector<std::uint16_t> have;
  have.reserve(n_old);
  for (std::uint16_t i = 0; i < n_old; ++i) {
    have.push_back(dst_t.u16(ifd0 + 2 + (std::size_t)i * 12));
  }
  const std::uint32_t next_ifd = dst_t.u32(ifd0 + 2 + (std::size_t)n_old * 12);

  // Collect the importable source entries, appending their relocated values
  // to the END of the file (nothing already there moves, so every existing
  // offset stays valid).
  const Tiff src_t{tiff, src_le};
  std::vector<OutEntry> add;
  if (!collect_ifd_(src_t, src_ifd0, file, dst_le,
                    /*skip_structural=*/true, add, 0)) {
    return false;
  }
  add.erase(std::remove_if(add.begin(), add.end(),
                           [&](const OutEntry& e) {
                             return std::find(have.begin(), have.end(), e.tag)
                                 != have.end();
                           }),
            add.end());
  if (add.empty()) { return true; }   // nothing to add: the file is fine

  // Merge: the file's own 12-byte records verbatim (their offsets still
  // point at unmoved bytes) plus the imported ones, in ascending tag order,
  // which TIFF requires.
  struct Rec {
    std::uint16_t   tag;
    const std::uint8_t* raw;   // the file's own 12-byte record
    const OutEntry* oe;        // ...or an imported entry
  };
  std::vector<Rec> recs;
  recs.reserve((std::size_t)n_old + add.size());
  for (std::uint16_t i = 0; i < n_old; ++i) {
    const std::uint8_t* r = file.data() + ifd0 + 2 + (std::size_t)i * 12;
    recs.push_back({dst_t.u16(ifd0 + 2 + (std::size_t)i * 12), r, nullptr});
  }
  for (const OutEntry& e : add) { recs.push_back({e.tag, nullptr, &e}); }
  std::stable_sort(recs.begin(), recs.end(),
                   [](const Rec& a, const Rec& b) { return a.tag < b.tag; });

  // `file` may reallocate while the new IFD is appended, so copy the old
  // entry bytes out FIRST -- Rec::raw points into it.
  std::vector<std::uint8_t> raw_copy(
      file.begin() + (long)(ifd0 + 2),
      file.begin() + (long)(ifd0 + 2 + (std::size_t)n_old * 12));
  for (Rec& r : recs) {
    if (r.raw != nullptr) {
      r.raw = raw_copy.data() + (r.raw - (file.data() + ifd0 + 2));
    }
  }

  if (file.size() & 1u) { file.push_back(0); }
  const std::uint32_t new_ifd = (std::uint32_t)file.size();
  put16_(file, (std::uint16_t)recs.size(), dst_le);
  for (const Rec& r : recs) {
    if (r.raw != nullptr) { file.insert(file.end(), r.raw, r.raw + 12); }
    else                  { write_entry_(file, *r.oe, dst_le); }
  }
  put32_(file, next_ifd, dst_le);   // keep any further pages of a multi-page

  // Repoint the header. This is the only byte of the original file touched.
  std::vector<std::uint8_t> off;
  put32_(off, new_ifd, dst_le);
  std::memcpy(file.data() + 4, off.data(), 4);
  return true;
}

bool
format_supports_exif(std::string_view format)
{
  return format == "jpeg" || format == "jpg" || format == "png"
      || format == "tiff" || format == "tif";
}

}  // namespace imgmeta
}  // namespace vpipe
