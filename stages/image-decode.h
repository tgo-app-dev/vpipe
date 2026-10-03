#ifndef VPIPE_STAGES_IMAGE_DECODE_H
#define VPIPE_STAGES_IMAGE_DECODE_H

// One still image off disk (or a URL FFmpeg can open) into a planar U8
// [C, H, W] TensorBeat, channels R, G, B (, A). load-image decodes with it
// as it is; training-dataset also has it fit the picture to a bucket.

#include "apple-silicon/tensor-beat.h"
#include "common/ffmpeg-libraries.h"

#include <memory>
#include <string>

namespace vpipe {

struct ImageDecodeOptions {
  // Four channels instead of three. swscale fills an opaque alpha plane
  // for a source that has none, so this is a statement about the BEAT's
  // shape, not about the file.
  bool keep_alpha = false;
  // A target size: the picture is scaled to COVER out_w x out_h, keeping
  // its aspect, and center-cropped to it. 0 keeps the decoded size.
  int out_w = 0;
  int out_h = 0;
  // Mirror left to right (a training augmentation).
  bool flip = false;
};

// Null, with the reason in *err, when the file cannot be decoded.
std::unique_ptr<TensorBeatPayload>
decode_image_file(const FFmpegLibraries* libs, const std::string& url,
                  const ImageDecodeOptions& opt, std::string* err);

// The decoded size alone, without scaling: false when the file does not
// open. Cheap next to a decode only in that nothing is converted.
bool probe_image_size(const FFmpegLibraries* libs, const std::string& url,
                      int* w, int* h, std::string* err);

}  // namespace vpipe

#endif
