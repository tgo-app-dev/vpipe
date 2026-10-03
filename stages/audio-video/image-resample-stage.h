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

// Apple-Silicon image-resample stage. Takes planar RGB or RGBA
// TensorBeats (rgb-frames, [3|4,H,W], u8, f16 or f32) and emits the same,
// resampled to the configured width x height. The iport and oport share
// one clock domain (1:1). f16 and f32 frames resample on the GPU in their
// own element type (resample_planar_float: every filter, fit and channel
// count); u8 RGB through the letterbox / Lanczos kernels; a CPU path
// implements everything, for u8 RGBA and bicubic, no-metal builds and
// `gpu: false`.
//
//   iport0  planar TensorBeat [3|4,H,W] (u8, f16 or f32), tag rgb-frames.
//   oport0  planar TensorBeat [3|4,height,width] (same dtype), rgb-frames.
//
// `fit` picks how a mismatched input/output aspect ratio is handled:
//   pad      match the long side, centre, pad the rest with `pad_color`.
//   crop     match the short side, fill, centre-crop.
//   stretch  fill the output, change the aspect ratio.
//   manual   sample from (src_x, src_y) at `scale`, placed at the output
//            origin, `pad_color` where the source runs out.
// `algorithm` selects the interpolation: lanczos (default), bilinear,
// bicubic.
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

  // Test / introspection accessors. width()/height() return the
  // configured target; 0 means "infer from the source aspect ratio".
  int out_width()  const noexcept { return _out_w; }
  int out_height() const noexcept { return _out_h; }
  int fit_mode()   const noexcept { return _mode; }

  // Resolve the configured target against a concrete source size: a
  // dimension left at 0 (or negative) is inferred from the other so the
  // source aspect ratio is preserved. Both configured -> used verbatim.
  void resolve_out_dims_(int in_w, int in_h,
                         int* out_w, int* out_h) const noexcept;

private:
  int          _out_w{}, _out_h{};  // 0 on either axis = infer from AR
  int          _mode{};            // 0 pad, 1 crop, 2 stretch, 3 manual
  int          _alg{1};   // 0 bilinear, 1 lanczos (default), 2 bicubic
  int          _src_x{}, _src_y{}; // manual source origin
  double       _scale{};           // manual resample ratio
  std::uint8_t _pad_r{}, _pad_g{}, _pad_b{};
  bool         _gpu{true};         // false pins the CPU path

  metal_compute::MetalCompute* _mc = nullptr;
  // Reused src-upload staging buffer (a CpuCached input on a GPU path);
  // re-allocated when a frame needs more bytes than it holds.
  std::unique_ptr<ExternalStorageHandle> _src_stage;
  std::size_t _stage_bytes = 0;

  // CPU bilinear path. Handles u8 and f32 (f16 arrives widened); matches
  // the GPU kernels' geometry exactly. `out_w`/`out_h` are the
  // resolved (aspect-inferred) output dimensions.
  // `C` is the plane count: 3 for RGB, 4 for RGBA. An RGBA source is
  // handed here PREMULTIPLIED and unpremultiplied by the caller -- see
  // process().
  void cpu_resample_(const std::uint8_t* src, int in_w, int in_h,
                     int out_w, int out_h,
                     std::uint8_t* dst, bool is_f32, int C = 3) const;

  // CPU separable-kernel path (u8 bicubic, u8 RGBA, no metal, or
  // `gpu: false`). `cubic` picks Pillow's BICUBIC over Lanczos-3; both
  // match PIL and the GPU kernels' geometry and tables.
  void cpu_lanczos_(const std::uint8_t* src, int in_w, int in_h,
                    int out_w, int out_h,
                    std::uint8_t* dst, bool is_f32, bool cubic,
                    int C = 3) const;
};

}

#endif
