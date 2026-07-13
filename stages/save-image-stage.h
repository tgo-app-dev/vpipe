#ifndef VPIPE_STAGES_SAVE_IMAGE_STAGE_H
#define VPIPE_STAGES_SAVE_IMAGE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

class FFmpegLibraries;
class BeatPayloadIntf;

// Save-image sink: writes each planar U8 RGB TensorBeat [3, H, W]
// (channel order R, G, B, 0..255 -- the format load-image / vae-decode
// emit) to an image file via FFmpeg's still-image encoders. The inverse
// of load-image.
//
//   iport0  TensorBeatPayload, U8 [3, H, W] planar RGB. Non-matching
//           beats are warned + dropped. EOS ends the stage.
//   (no oports -- it is a sink.)
//
// Config (FlexData object on the 4th constructor parameter):
//   path        (string, required) -- output file path. May contain a
//                 printf integer conversion (e.g. "out/frame-%04d.png"):
//                 it is formatted with the 0-based beat index so a stream
//                 of images lands in distinct files. Without a conversion
//                 the first image writes to `path` verbatim and each later
//                 image gets a "-NNNNNN" suffix before the extension.
//   format      (string, default from the path extension, else "png") --
//                 png | jpeg (jpg) | webp | bmp | tiff.
//   quality     (int, default 90) -- lossy codecs (jpeg, lossy webp),
//                 1..100, higher is better.
//   compression (int, default 6) -- PNG zlib level 0..9, higher is
//                 smaller + slower (PNG is lossless, so this is size, not
//                 fidelity).
//   lossless    (bool, default false) -- webp lossless mode.
class SaveImageStage final : public TypedStage<SaveImageStage> {
public:
  static constexpr const char* kTypeName = "save-image";

  SaveImageStage(const SessionContextIntf* session,
                  std::string               id,
                  std::vector<InEdge>       iports,
                  FlexData                  config);
  ~SaveImageStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& path()           const noexcept { return _path; }
  std::uint64_t      images_written() const noexcept { return _written; }

private:
  // Encode the planar U8 RGB payload to `out_path`. Returns true on a
  // fully-written file; logs + returns false otherwise.
  bool encode_(const BeatPayloadIntf& beat, const std::string& out_path);

  // Materialise the output path for the `index`-th beat (printf token or
  // suffix rule -- see the class doc).
  std::string resolve_path_(std::uint64_t index) const;

  std::string av_err_(int rc) const;

  std::string _path;
  std::string _format;          // resolved lower-case codec key
  int         _quality     = 90;
  int         _compression = 6;
  bool        _lossless    = false;

  const FFmpegLibraries* _libs = nullptr;
  std::uint64_t          _seen    = 0;   // beats consumed (index source)
  std::uint64_t          _written = 0;   // files successfully written
};

}

#endif
