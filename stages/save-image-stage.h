#ifndef VPIPE_STAGES_SAVE_IMAGE_STAGE_H
#define VPIPE_STAGES_SAVE_IMAGE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/output-path.h"

#include <cstdint>
#include <span>
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
//   path        (string, required) -- output file path, a template: see
//                 stages/output-path.h. A printf integer conversion
//                 ("out/frame-%04d.png") takes the 0-based index, "%t"
//                 takes the local time, and without a conversion the
//                 first image writes `path` verbatim while each later
//                 one gets a "-NNNNNN" suffix before the extension.
//   no_overwrite (bool, default false) -- never write over a file that
//                 is already there: the sequence number starts past
//                 every name on disk instead of at 0. Off by default
//                 because the usual graph WANTS the same name rewritten
//                 -- it is the file somebody has open in a viewer.
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
  void reset_run_state() override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& path()           const noexcept { return _path; }
  std::uint64_t      images_written() const noexcept { return _written; }

private:
  // Encode the planar U8 RGB payload to `out_path`. Returns true on a
  // fully-written file; logs + returns false otherwise. When `exif_tiff` is
  // non-empty it is spliced into the encoded container (JPEG APP1 / PNG
  // eXIf) before the bytes hit disk; a splice failure warns but still writes
  // the image, since the pixels matter more than the metadata.
  bool encode_(const BeatPayloadIntf& beat, const std::string& out_path,
               std::span<const std::uint8_t> exif_tiff = {});

  std::string av_err_(int rc) const;

  std::string _path;
  std::string _format;          // resolved lower-case codec key
  int         _quality     = 90;
  int         _compression = 6;
  bool        _lossless    = false;

  bool            _no_overwrite = false;
  outpath::Tokens _tok;         // which template tokens `path` carries

  const FFmpegLibraries* _libs = nullptr;
  // The filename index lives in the picker now (it was `_seen`), so the
  // numbering rule and the no-overwrite scan cannot drift apart.
  outpath::SeqPicker     _seq;
  std::uint64_t          _written = 0;   // files successfully written
};

}

#endif
