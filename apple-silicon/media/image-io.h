#ifndef VPIPE_APPLE_MEDIA_IMAGE_IO_H
#define VPIPE_APPLE_MEDIA_IMAGE_IO_H

#include "apple-silicon/media/color-tags.h"
#include "apple-silicon/tensor-beat.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe::apple_media {

// Stills through ImageIO: what FFmpeg's image path cannot keep -- more
// than 8 bits (16-bit PNG/TIFF), float (OpenEXR), alpha as stored, the
// colour space the file declares -- and camera RAW, which is DEVELOPED
// (Core Image's RAW engine) rather than decoded.

// A still's header, read without decoding it. For a RAW, width and
// height are the developed picture's: upright, as read_still returns it.
struct StillInfo {
  int         width = 0;
  int         height = 0;
  int         bits = 8;          // per component
  bool        is_float = false;  // half or float samples (EXR, float TIFF)
  bool        has_alpha = false;
  bool        raw = false;       // camera RAW (CR2, NEF, ARW, DNG, ...)
  int         orientation = 1;   // EXIF 1..8, as the file declares it
  std::string uti;               // ImageIO's type, "com.ilm.openexr-image"
};
bool probe_still(const std::string& path, StillInfo* out, std::string* err);

// The file is one ImageIO keeps and FFmpeg would flatten: deeper than
// 8 bits, float, a container FFmpeg reads poorly (OpenEXR, HEIC), or a
// camera RAW (which FFmpeg cannot develop at all).
bool prefers_imageio(const StillInfo&);

// How a camera RAW is developed. There is no colour space to read "as
// declared" -- the development decides it -- so the two looks name one:
//
//   Rendered  the camera's own look, as a viewer shows the file: its
//             tone curve, baseline exposure, noise reduction, highlight
//             recovery. Display-referred sRGB, 0..1 (what a model was
//             trained on, and what an sRGB export of the photo is).
//   Linear    scene-linear: no global or local tone curve, and the
//             highlight headroom the sensor captured kept above 1 (up to
//             about 3 on a 5D Mark IV CR2). Linear BT.2020, so no camera
//             colour is clipped; for F16 or F32 -- in 8 bits it bands.
//
// `exposure` is in EV, on top of the camera's baseline. White balance
// is as shot. Either way the picture comes out UPRIGHT: the file's EXIF
// orientation is applied, so a caller carrying its EXIF on must say
// Orientation 1.
struct RawOptions {
  enum class Look { Rendered, Linear };
  Look   look = Look::Rendered;
  double exposure = 0;
};

// Decode the still at `path` into a planar picture of `dtype` (U8 or
// F16), its colour as the file declares it -- no conversion unless the
// file's profile is not one CICP can name (an arbitrary ICC profile),
// when it is converted to sRGB (extended sRGB for F16, so nothing out
// of gamut is clipped). `keep_alpha`: 4 channels, straight alpha --
// opaque when the file has none, so the shape is the caller's choice
// (as load-image's `alpha: keep`); else 3. The EXIF orientation is NOT
// applied, as with the FFmpeg path -- except for a camera RAW, which is
// developed per `raw` and comes out upright (see RawOptions). F32 is
// accepted too: F16's samples, widened.
std::unique_ptr<TensorBeatPayload>
read_still(const std::string& path, TensorBeat::DType dtype,
           bool keep_alpha, ColorTags* tags, std::string* err,
           const RawOptions& raw = {});

// Formats encode_still writes, and their ImageIO types.
//   png, tiff  8 or 16 bits per component
//   exr        half float, scene-linear: a picture in another transfer
//              is linearized, alpha premultiplied as OpenEXR expects.
//              ImageIO stores Rec.709 primaries: wider primaries are
//              converted (linear, unclamped), so the colours survive
//              and read back tagged BT.709 linear
//   jpeg, heic 8 bits, lossy
bool still_format_known(std::string_view format);

struct StillWriteOptions {
  std::string format = "png";
  int         bits = 8;         // png / tiff: 8 or 16
  double      quality = 0.9;    // jpeg / heic: 0..1
};

// Encode a planar picture (U8 or F16 [3|4,H,W], tagged `tags`) into
// file bytes in memory, with the colour space embedded, so a caller can
// splice metadata before writing.
bool encode_still(const TensorBeat& pic, const ColorTags& tags,
                  const StillWriteOptions& opt, std::vector<std::uint8_t>* out,
                  std::string* err);

}  // namespace vpipe::apple_media

#endif
