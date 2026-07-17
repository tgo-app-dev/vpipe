#ifndef VPIPE_APPLE_SILICON_COREML_YOLO_DETECTION_STAGE_H
#define VPIPE_APPLE_SILICON_COREML_YOLO_DETECTION_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

namespace metal_compute { class MetalCompute; }
class CoreMLLoadedModel;

// Runs a YOLOv6-shape CoreML detection model on a TensorBeat of an
// RGB frame and emits a FlexData detection record per frame.
//
// Pipeline shape:
//   video-to-rgb --> yolo-detection
//   iport 0: Beat<TensorBeat>  shape [3, H, W], float32 in [0, 1]
//   oport 0: Beat<FlexData>    object with `frame_width`, `frame_height`,
//                              and `detections` (array of objects with
//                              class_id, class_name, score, x1, y1,
//                              x2, y2 in original-frame pixel coords).
//
// Model contract:
//   * Input feature may be either:
//      - MLMultiArray (TensorType), shape [1, 3, S, S] or [3, S, S],
//        float32 in [0, 1]; OR
//      - Image (CVPixelBuffer / ImageType), pixel format
//        kCVPixelFormatType_32BGRA at S × S. Auto-detected from the
//        model's input feature description at initialize time; no
//        config knob required. For ImageType the stage builds a
//        BGRA8888 CVPixelBuffer from the u8 letterboxed RGB and
//        hands it to CoreML directly.
//   * Output feature is an MLMultiArray, shape [1, N, 4 + num_classes]
//     or [N, 4 + num_classes]. The 4 bbox values are
//     (cx, cy, w, h) in model-input pixels; the `num_classes` values
//     are sigmoid-activated class probabilities in [0, 1] (YOLOv6
//     deploy mode).
//
// Configuration (FlexData object on the 4th constructor parameter):
//   model_path            (string, required)
//   input_feature_name    (string, default "image")
//   output_feature_name   (string, default "predictions")
//   num_classes           (int,    optional)          -- auto-detected
//                         from the model's output last dim (4+num_classes)
//                         when absent or 0.
//   confidence_threshold  (real,   default 0.25)
//   iou_threshold         (real,   default 0.45)
//   max_detections        (int,    default 300)
//   class_names           (array<string>, optional)   -- empty array
//                         disables class_name in the output. When
//                         supplied alongside an auto-detected
//                         num_classes the array's length must match.
//   compute_units         (int,    default 2)         -- ComputeUnitsAll
//   uses_cpu_only         (bool,   default false)
//   oport_capacity        (int,    default 4)
//
// Input geometry is driven by the CoreML model:
//   * Fixed-shape models (square or rectangular): the model declares
//     its input pixel dims; the stage aspect-ratio-preserving center-
//     letterboxes incoming frames to that size. The fast path uses a
//     single Metal compute kernel that does bilinear letterbox +
//     u8→f32 normalisation in one dispatch (Shared MTL::Buffer in,
//     Shared MTL::Buffer out).
//   * Flexible-shape models (input shape declared with placeholder
//     dims): the stage feeds the source's native H × W directly with
//     no resample. F32 sources are bound zero-copy; u8 sources are
//     normalised on the CPU.
//   * Fast skip path: when the incoming TensorBeat already matches the
//     model's expected size *and* dtype (f32 planar RGB for TensorType
//     models) the resample stage is skipped entirely and the source
//     buffer is bound directly into the MLMultiArray.
class YoloDetectionStage final
  : public TypedStage<YoloDetectionStage>
{
public:
  static constexpr const char* kTypeName = "yolo-detection";

  YoloDetectionStage(const SessionContextIntf* session,
                     std::string               id,
                     std::vector<InEdge>       iports,
                     FlexData                  config);

  ~YoloDetectionStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

private:
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  std::string              _model_path;
  std::string              _models_db;
  std::string              _input_feature_name;
  std::string              _output_feature_name;
  // Model-driven; populated in initialize(). `_input_width` and
  // `_input_height` are the model's expected input pixel dims. They
  // are equal for square YOLO exports (the common case) and may
  // differ for rectangular exports. When the model declares a
  // flexible input shape, `_input_size_flexible` is set and both
  // dims are 0 — the stage then forwards frames at their native
  // dimensions with no letterboxing.
  int                      _input_width          = 0;
  int                      _input_height         = 0;
  bool                     _input_size_flexible  = false;
  // Optional num_classes; auto-detected from the model output's last
  // dim when 0. `_num_classes_user_set` tracks whether the user
  // supplied a value so we know whether to validate class_names size
  // at construction time vs initialize() time.
  int                      _num_classes         = 0;
  bool                     _num_classes_user_set = false;
  double                   _confidence_threshold{};
  double                   _iou_threshold{};
  int                      _max_detections{};
  std::vector<std::string> _class_names;
  int                      _compute_units{};
  bool                     _uses_cpu_only{};
  unsigned                 _oport_capacity{};

  std::shared_ptr<CoreMLLoadedModel> _loaded;

  // Session-shared metal-compute device captured at initialize() so
  // the hot path doesn't have to walk session() per call. nullptr on
  // non-Apple builds / when the device failed to come up. The Metal
  // letterbox fast path is gated on (_mc && Shared input storage).
  metal_compute::MetalCompute*       _mc = nullptr;
  bool                               _metal_letterbox_logged = false;
  bool                               _metal_bgra_logged      = false;

  // One-shot diagnostic: dump output shape/strides/dtype + the
  // first anchor row the first time we successfully run inference.
  // Cheap to set, invaluable for triaging "wrong scores / wrong
  // bboxes" bugs where the output tensor is laid out unexpectedly.
  bool                               _logged_output_spec = false;

  // Image-input plumbing (set at initialize() if the model declares
  // ImageType for its input feature). _image_pixel_format mirrors the
  // CVPixelFormatType the model expects; v1 supports
  // kCVPixelFormatType_32BGRA only.
  bool                               _image_input        = false;
  uint32_t                           _image_pixel_format = 0;
};

}

#endif
