#ifndef VPIPE_COMMON_IMAGE_METADATA_H
#define VPIPE_COMMON_IMAGE_METADATA_H

#include "common/flex-data.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
namespace imgmeta {

// Reading and writing the metadata that rides ALONGSIDE image pixels --
// today EXIF, the block a camera writes describing how the shot was taken.
//
// WHY THIS EXISTS RATHER THAN USING FFMPEG. FFmpeg surfaces essentially none
// of it for stills: on a real Sony JPEG that plainly contains an APP1 "Exif"
// block, ffprobe reports zero format tags and zero stream tags, exposing only
// an ICC profile and a display matrix as frame side data. So the block is
// read straight out of the file bytes and written straight into them.
//
// The EXIF payload itself is a TIFF structure (header + IFD chain). It is
// carried unchanged in both containers this module writes:
//   * JPEG -- an APP1 segment whose data is "Exif\0\0" followed by the TIFF
//             block.
//   * PNG  -- an `eXIf` chunk (PNG spec, 2017) whose data IS the TIFF block,
//             with no prefix.
// Keeping the block verbatim is what makes a load -> save round trip
// LOSSLESS: every tag survives, including maker notes and tags this module's
// reader does not name.

// Extract the raw EXIF TIFF block from an image file. Returns empty when the
// file has none, is unreadable, or is a container this module cannot scan.
// Recognizes JPEG (APP1 "Exif\0\0"), PNG (`eXIf` chunk) and TIFF. A TIFF file
// carries no separate block -- its own IFD0 IS the metadata -- so for TIFF a
// compact standalone block is SYNTHESIZED from it, holding the descriptive
// tags and the Exif/GPS sub-IFDs but none of the tags that describe that
// file's pixel layout (which would be meaningless attached to another image).
std::vector<std::uint8_t> read_exif_blob(const std::string& path);

// Same, from bytes already in memory.
std::vector<std::uint8_t> find_exif_blob(std::span<const std::uint8_t> file);

// Parse a raw TIFF/EXIF block into a flat FlexData object of
// {tag_name: value}. Walks IFD0, the Exif sub-IFD (tag 0x8769) and the GPS
// sub-IFD (0x8825). Known tags get their spec names; anything else is named
// "Tag0xNNNN" so nothing is silently dropped. Values are strings for ASCII,
// uint/int for integral types, real for rationals; multi-value tags become
// arrays. Returns a Null FlexData when `tiff` is not a valid TIFF block.
FlexData parse_exif(std::span<const std::uint8_t> tiff);

// Insert `tiff` into an in-memory JPEG as an APP1 "Exif\0\0" segment, placed
// immediately after SOI. Any APP1 Exif segment already present is REPLACED so
// the file never ends up with two. False when `jpeg` is not a JPEG or `tiff`
// is too large for a segment (65533 bytes of payload).
bool jpeg_set_exif(std::vector<std::uint8_t>&     jpeg,
                   std::span<const std::uint8_t>  tiff);

// Insert `tiff` into an in-memory PNG as an `eXIf` chunk, placed immediately
// after IHDR. Any existing `eXIf` chunk is replaced. False when `png` is not
// a PNG.
bool png_set_exif(std::vector<std::uint8_t>&    png,
                  std::span<const std::uint8_t> tiff);

// Merge `tiff`'s tags into an in-memory TIFF file's IFD0.
//
// TIFF is NOT like the other two. A TIFF file already IS a TIFF structure, so
// EXIF cannot ride along as an opaque segment -- it has to be merged into the
// file's own IFD, and the file's IFD is what points at its pixel data. So:
//   * a MERGED copy of IFD0 is appended at the end of the file and the
//     header's first-IFD offset is repointed at it. Nothing already in the
//     file moves, so every existing StripOffsets / StripByteCounts entry stays
//     valid. The original IFD0 is left in place as dead bytes.
//   * tags the file ALREADY has are never overwritten, and the tags that
//     describe the file's own pixels (dimensions, strips, compression, ...)
//     are never copied from the source at all. Importing the source's
//     StripOffsets over the target's would point the reader at nothing.
//   * values are relocated with byte-order conversion, so a little-endian
//     EXIF block merges correctly into a big-endian TIFF; the Exif (0x8769)
//     and GPS (0x8825) sub-IFDs are copied recursively.
// False when `file` is not a classic TIFF (BigTIFF, version 43, is refused --
// it is a different format with 64-bit offsets).
bool tiff_set_exif(std::vector<std::uint8_t>&    file,
                   std::span<const std::uint8_t> tiff);

// Return an EXIF block equal to `base` but with the Software tag (0x0131)
// set to `software`. `base` may be empty, in which case a minimal block
// holding only that tag is built. Every other tag in `base` is carried over
// unchanged (values relocated), so this records WHO wrote the file without
// disturbing what the camera recorded. Empty `software` returns `base`.
std::vector<std::uint8_t>
exif_set_software(std::span<const std::uint8_t> base,
                  std::string_view              software);

// True when this module can write EXIF into the named container ("jpeg" /
// "jpg" / "png"). Lets a caller warn accurately instead of failing silently
// on a format it cannot annotate.
bool format_supports_exif(std::string_view format);

}  // namespace imgmeta
}  // namespace vpipe

#endif
