#ifndef VPIPE_APPLE_MEDIA_COLOR_TAGS_H
#define VPIPE_APPLE_MEDIA_COLOR_TAGS_H

#include "common/flex-data.h"

#include <vector>

namespace vpipe::apple_media {

// What a picture's samples mean -- the sideband keys of common/
// beat-keys.h ("what a picture's samples MEAN") as one value, so a
// reader fills them once and a writer reads them once. CICP (ITU-T
// H.273) code points; 2 = unspecified.
struct ColorTags {
  int  primaries = 2;
  int  transfer = 2;
  int  matrix = 2;
  bool full_range = true;
  bool premultiplied = false;  // alpha mode of a 4-channel picture
  int  source_bits = 0;        // 0 = not known
  std::vector<double> mastering;      // 10 numbers (ST 2086), or empty
  std::vector<double> content_light;  // [max_cll, max_fall], or empty

  // A picture with no tags at all meant sRGB before the keys existed.
  static ColorTags srgb() { return {1, 13, 0}; }

  bool is_hdr() const { return transfer == 16 || transfer == 18; }
  bool is_linear() const { return transfer == 8; }

  // Writes every key it knows into `sideband` (an object; made one if
  // it is not).
  void write(FlexData& sideband) const;
  // Reads the keys; absent primaries AND transfer = sRGB (the old
  // meaning of an untagged picture).
  static ColorTags read(const FlexData& sideband);
};

}  // namespace vpipe::apple_media

#endif
