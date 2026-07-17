#ifndef VPIPE_STAGES_IMAGE_RESAMPLE_STAGE_H
#define VPIPE_STAGES_IMAGE_RESAMPLE_STAGE_H

#include "apple-silicon/tensor-storage.h"
#include "common/flex-data.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

namespace metal_compute { class MetalCompute; }

// Apple-Silicon image-resample stage. Takes planar RGB TensorBeats
// (rgb-frames, [3,H,W], u8 or f32) and emits the same, resampled to the
// configured width x height. The iport and oport share one clock domain
// (1:1). u8 frames go through the (generalised) letterbox GPU kernel; a
// CPU bilinear fallback covers f32 and no-metal builds.
//
//   iport0  planar RGB TensorBeat [3,H,W] (u8 or f32), tag rgb-frames.
//   oport0  planar RGB TensorBeat [3,height,width] (same dtype), rgb-frames.
//
// `fit` picks how a mismatched input/output aspect ratio is handled:
//   pad      match the long side, centre, pad the rest with `pad_color`.
//   crop     match the short side, fill, centre-crop.
//   stretch  fill the output, change the aspect ratio.
//   manual   sample from (src_x, src_y) at `scale`, placed at the output
//            origin, `pad_color` where the source runs out.
// `algorithm` selects the interpolation (bilinear only for now).
class ImageResampleStage final : public TypedStage<ImageResampleStage> {
public:
  static constexpr const char* kTypeName = "image-resample";

  ImageResampleStage(const SessionContextIntf* session,
                     std::string               id,
                     std::vector<InEdge>       iports,
                     FlexData                  config);
  ~ImageResampleStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test / introspection accessors.
  int out_width()  const noexcept { return _out_w; }
  int out_height() const noexcept { return _out_h; }
  int fit_mode()   const noexcept { return _mode; }

private:
  int          _out_w{}, _out_h{};
  int          _mode{};            // 0 pad, 1 crop, 2 stretch, 3 manual
  int          _src_x{}, _src_y{}; // manual source origin
  double       _scale{};           // manual resample ratio
  std::uint8_t _pad_r{}, _pad_g{}, _pad_b{};

  metal_compute::MetalCompute* _mc = nullptr;
  // Reused src-upload staging buffer (only for a CpuCached u8 input on the
  // GPU path); re-allocated when the input dims change.
  std::unique_ptr<ExternalStorageHandle> _src_stage;
  int _stage_in_w = 0, _stage_in_h = 0;

  // CPU bilinear fallback (f32, or no metal). Handles u8 and f32; matches
  // the GPU kernel's geometry exactly for u8.
  void cpu_resample_(const std::uint8_t* src, int in_w, int in_h,
                     std::uint8_t* dst, bool is_f32) const;
};

}

#endif
