#ifndef VPIPE_STAGES_LOAD_IMAGE_STAGE_H
#define VPIPE_STAGES_LOAD_IMAGE_STAGE_H

#include "common/beat-payload-intf.h"
#include "common/ffmpeg-libraries.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Source-ish stage: 0 or 1 inputs, 1 output. Reads still images from
// local files or network URLs via FFmpeg, decodes the first video
// frame of each input, converts to planar RGB, and emits one
// TensorBeat on out-port 0 per image with shape [3, H, W] and dtype
// U8 (raw 0..255 bytes, channel order R, G, B).
//
// Pacing: when an iport is wired (typically to a `chrono` stage),
// each upstream beat triggers exactly one decode+emit. With no iport
// the stage emits every URL back-to-back and signals done.
//
// Configuration (FlexData object on the 4th constructor parameter):
//   url   (string or array<string>, required) -- one path or a list
//                          of paths/URLs. Local paths, "file://",
//                          "http(s)://", "rtsp://", etc. are all
//                          handled by FFmpeg's demuxer autodetection.
class LoadImageStage final : public TypedStage<LoadImageStage> {
public:
  static constexpr const char* kTypeName = "load-image";

  LoadImageStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);
  ~LoadImageStage() override;

  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessor.
  const std::vector<std::string>& urls() const noexcept { return _urls; }

private:
  std::unique_ptr<BeatPayloadIntf>
  decode_url_(const std::string& url) const;

  // The oport-1 beat: {url, width, height} plus, when the source carries
  // EXIF, a parsed `exif` object AND `exif_tiff_b64` -- the untouched TIFF
  // block, base64'd. The parsed view is for reading and routing; the raw
  // block is what makes a load -> save round trip lossless, since it also
  // carries maker notes and tags the reader does not name. `image` is the
  // decoded beat, read only for its dimensions.
  std::unique_ptr<BeatPayloadIntf>
  build_metadata_(const std::string& url, const BeatPayloadIntf& image) const;

  std::vector<std::string> _urls;
  // `alpha: keep` emits four channels. DEFAULT DROP, because the beat
  // every other consumer here reads is [3,H,W] and turning a
  // transparent PNG into a 4-channel beat by surprise would break the
  // graph that was working yesterday. See the config doc.
  bool _keep_alpha = false;
  // `dtype`: "u8" (the default; what every consumer reads), "f16"
  // (deeper than 8 bits survive), or "auto" (f16 for a file deeper than
  // 8 bits or float). `decoder`: "auto" | "ffmpeg" | "imageio" -- auto
  // reads with ImageIO what FFmpeg would flatten (16-bit, float, EXR,
  // HEIC) and whatever is asked for as F16; FFmpeg otherwise.
  std::string _dtype = "u8";
  std::string _decoder = "auto";
  // A camera RAW is DEVELOPED (ImageIO / Core Image): `raw` "rendered"
  // (the camera's look, sRGB) or "linear" (scene-linear BT.2020), at
  // `raw_exposure` EV. It always reads through ImageIO, upright.
  bool        _raw_linear = false;
  double      _raw_exposure = 0;
  std::size_t              _next = 0;
  // The `images` list has gone out this run.
  bool                     _list_emitted = false;

  const FFmpegLibraries*   _libs = nullptr;
};

}

#endif
