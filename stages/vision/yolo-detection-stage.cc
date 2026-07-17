#include "stages/vision/yolo-detection-stage.h"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "apple-silicon/metal-compute/image-ops.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "apple-silicon/tensor-storage.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/oport-policy.h"
#include "common/perf-event.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#include <CoreVideo/CVPixelBuffer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

namespace {

// (CoreML feature marshaling + introspection now live in
// CoreMLLoadedModel::predict / input_descs; this stage handles only the
// GPU/CPU letterbox and the neutral CoreMLPredict* structs. CoreVideo is
// still used for the BGRA pixel-format constant.)

struct LetterboxParams {
  float scale = 1.0f;          // model_input = orig * scale
  int   pad_x = 0;             // pad on the left in model space
  int   pad_y = 0;             // pad on the top in model space
};

// Letterbox-resize a CHW float32 [0,1] frame at (W, H) into a CHW
// float32 [3, Sh, Sw] buffer using bilinear sampling. The aspect
// ratio of the source is preserved; the padding region is filled
// with the canonical YOLO grey (114/255). Returns the scale + pad
// applied so the inverse transform can map detections back.
LetterboxParams
letterbox_resize_(const float* src,    // [3, H, W]
                  int          W, int H,
                  int          Sw, int Sh,
                  float*       dst)    // [3, Sh, Sw]
{
  LetterboxParams p;
  if (W <= 0 || H <= 0 || Sw <= 0 || Sh <= 0) {
    return p;
  }
  const float sx = static_cast<float>(Sw) / static_cast<float>(W);
  const float sy = static_cast<float>(Sh) / static_cast<float>(H);
  p.scale = std::min(sx, sy);
  const int new_w = static_cast<int>(std::round(p.scale * W));
  const int new_h = static_cast<int>(std::round(p.scale * H));
  p.pad_x = (Sw - new_w) / 2;
  p.pad_y = (Sh - new_h) / 2;

  constexpr float kPad = 114.0f / 255.0f;
  std::fill(dst, dst + static_cast<size_t>(3) * Sw * Sh, kPad);

  const float inv = 1.0f / p.scale;
  for (int c = 0; c < 3; ++c) {
    const float* src_c = src + static_cast<size_t>(c) * H * W;
    float*       dst_c = dst + static_cast<size_t>(c) * Sw * Sh;
    for (int yo = 0; yo < new_h; ++yo) {
      const float syf = (yo + 0.5f) * inv - 0.5f;
      int iy0 = static_cast<int>(std::floor(syf));
      int iy1 = iy0 + 1;
      float fy = syf - iy0;
      iy0 = std::clamp(iy0, 0, H - 1);
      iy1 = std::clamp(iy1, 0, H - 1);
      float* dst_row = dst_c + (yo + p.pad_y) * static_cast<size_t>(Sw)
                            + p.pad_x;
      for (int xo = 0; xo < new_w; ++xo) {
        const float sxf = (xo + 0.5f) * inv - 0.5f;
        int ix0 = static_cast<int>(std::floor(sxf));
        int ix1 = ix0 + 1;
        float fx = sxf - ix0;
        ix0 = std::clamp(ix0, 0, W - 1);
        ix1 = std::clamp(ix1, 0, W - 1);
        const float v00 = src_c[iy0 * W + ix0];
        const float v01 = src_c[iy0 * W + ix1];
        const float v10 = src_c[iy1 * W + ix0];
        const float v11 = src_c[iy1 * W + ix1];
        const float v0  = v00 + fx * (v01 - v00);
        const float v1  = v10 + fx * (v11 - v10);
        dst_row[xo]     = v0 + fy * (v1 - v0);
      }
    }
  }
  return p;
}

// Bilinear-letterbox u8 RGB planar [3, H, W] directly into a
// BGRA8888 interleaved buffer of Sw × Sh. Used for the ct.ImageType
// fast path when the source TensorBeat is already u8 -- skips the
// u8→f32 and f32→u8 round-trips of the generic pipeline. Padding
// is the canonical YOLO grey (114) with alpha=255. Returns scale +
// pad so the inverse transform can map detections back to the
// original-frame coordinates.
LetterboxParams
letterbox_u8_to_bgra_(const uint8_t* src,    // [3, H, W] planar RGB
                      int            W, int H,
                      int            Sw, int Sh,
                      uint8_t*       dst,    // BGRA interleaved
                      size_t         bpr)
{
  LetterboxParams p;
  if (W <= 0 || H <= 0 || Sw <= 0 || Sh <= 0) {
    return p;
  }
  const float sx = static_cast<float>(Sw) / static_cast<float>(W);
  const float sy = static_cast<float>(Sh) / static_cast<float>(H);
  p.scale = std::min(sx, sy);
  const int new_w = static_cast<int>(std::round(p.scale * W));
  const int new_h = static_cast<int>(std::round(p.scale * H));
  p.pad_x = (Sw - new_w) / 2;
  p.pad_y = (Sh - new_h) / 2;

  // Pre-fill the whole buffer with the BGRA pad pixel
  // (114, 114, 114, 255). On little-endian (all Apple Silicon) a
  // single uint32_t store of 0xFF767676 lays down bytes in B, G, R, A
  // order, so one std::fill_n per row writes 4 bytes per iteration
  // and the compiler lowers it to a tight NEON store.
  constexpr uint8_t  kPad   = 114;
  constexpr uint32_t kPadPx =
      (static_cast<uint32_t>(0xFFu) << 24)
    | (static_cast<uint32_t>(kPad)  << 16)
    | (static_cast<uint32_t>(kPad)  <<  8)
    |  static_cast<uint32_t>(kPad);
  for (int y = 0; y < Sh; ++y) {
    auto* p32 = reinterpret_cast<uint32_t*>(dst + y * bpr);
    std::fill_n(p32, Sw, kPadPx);
  }

  // Bilinear-sample each (R, G, B) plane at the same (sy, sx) and
  // write BGRA into the (pad_x, pad_y, new_w, new_h) window. The
  // promotion to float for the 4-tap blend keeps the result bit-
  // identical to the f32 path within ±1 LSB.
  const float    inv     = 1.0f / p.scale;
  const size_t   plane   = static_cast<size_t>(H) * W;
  const uint8_t* r_plane = src + 0 * plane;
  const uint8_t* g_plane = src + 1 * plane;
  const uint8_t* b_plane = src + 2 * plane;
  for (int yo = 0; yo < new_h; ++yo) {
    const float syf = (yo + 0.5f) * inv - 0.5f;
    int   iy0 = static_cast<int>(std::floor(syf));
    int   iy1 = iy0 + 1;
    const float fy = syf - iy0;
    iy0 = std::clamp(iy0, 0, H - 1);
    iy1 = std::clamp(iy1, 0, H - 1);
    uint8_t* dst_row = dst + (yo + p.pad_y) * bpr + p.pad_x * 4;
    for (int xo = 0; xo < new_w; ++xo) {
      const float sxf = (xo + 0.5f) * inv - 0.5f;
      int   ix0 = static_cast<int>(std::floor(sxf));
      int   ix1 = ix0 + 1;
      const float fx = sxf - ix0;
      ix0 = std::clamp(ix0, 0, W - 1);
      ix1 = std::clamp(ix1, 0, W - 1);
      auto sample_ = [&](const uint8_t* pl) -> uint8_t {
        const float v00 = pl[iy0 * W + ix0];
        const float v01 = pl[iy0 * W + ix1];
        const float v10 = pl[iy1 * W + ix0];
        const float v11 = pl[iy1 * W + ix1];
        const float v0  = v00 + fx * (v01 - v00);
        const float v1  = v10 + fx * (v11 - v10);
        const float v   = v0 + fy * (v1 - v0);
        return static_cast<uint8_t>(v + 0.5f);
      };
      const uint8_t r = sample_(r_plane);
      const uint8_t g = sample_(g_plane);
      const uint8_t b = sample_(b_plane);
      dst_row[xo * 4 + 0] = b;
      dst_row[xo * 4 + 1] = g;
      dst_row[xo * 4 + 2] = r;
      dst_row[xo * 4 + 3] = 0xFF;
    }
  }
  return p;
}

struct Det {
  float x1, y1, x2, y2;   // in model-input coords
  float score;
  int   cls;
};

float
iou_(const Det& a, const Det& b)
{
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float iw  = std::max(0.0f, xx2 - xx1);
  const float ih  = std::max(0.0f, yy2 - yy1);
  const float inter = iw * ih;
  const float aa  = (a.x2 - a.x1) * (a.y2 - a.y1);
  const float ab  = (b.x2 - b.x1) * (b.y2 - b.y1);
  const float uni = aa + ab - inter;
  return uni > 0.0f ? (inter / uni) : 0.0f;
}

// Decode predictions for each anchor. Two layouts are supported,
// auto-selected by the caller from `stride`:
//   stride == 4 + nc      -- YOLOv8/anchor-free: (cx, cy, w, h, c_0..c_{nc-1}).
//                            Final score = max(c_i).
//   stride == 4 + 1 + nc  -- YOLOv5/v6/v7: (cx, cy, w, h, obj, c_0..c_{nc-1}).
//                            Final score = obj * max(c_i).
// Scores are assumed already sigmoid-activated (deploy-mode head).
// Drops anchors where the final score < threshold.
void
decode_predictions_(const float* preds, int N, int stride, int nc,
                    bool has_objectness, float thresh,
                    vector<Det>* out)
{
  out->reserve(static_cast<size_t>(N) / 64);
  const int cls_off = has_objectness ? 5 : 4;
  for (int i = 0; i < N; ++i) {
    const float* row = preds + static_cast<size_t>(i) * stride;
    const float obj = has_objectness ? row[4] : 1.0f;
    if (obj < thresh) {
      // Cheap early-out: even with class==1.0 the product would
      // miss threshold.
      continue;
    }
    float best = -1.0f;
    int   best_c = -1;
    for (int c = 0; c < nc; ++c) {
      const float s = row[cls_off + c];
      if (s > best) {
        best = s;
        best_c = c;
      }
    }
    const float score = obj * best;
    if (score < thresh || best_c < 0) {
      continue;
    }
    const float cx = row[0];
    const float cy = row[1];
    const float w  = row[2];
    const float h  = row[3];
    Det d;
    d.x1 = cx - 0.5f * w;
    d.y1 = cy - 0.5f * h;
    d.x2 = cx + 0.5f * w;
    d.y2 = cy + 0.5f * h;
    d.score = score;
    d.cls   = best_c;
    out->push_back(d);
  }
}

// Greedy NMS per class. dets is consumed (sorted in place); returns
// the kept indices into the (newly sorted) dets vector. Capped at
// `max_keep`.
vector<size_t>
nms_(vector<Det>& dets, float iou_thresh, size_t max_keep)
{
  std::sort(dets.begin(), dets.end(),
            [](const Det& a, const Det& b) { return a.score > b.score; });
  vector<bool>   suppressed(dets.size(), false);
  vector<size_t> kept;
  kept.reserve(std::min(dets.size(), max_keep));
  for (size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    kept.push_back(i);
    if (kept.size() >= max_keep) {
      break;
    }
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (suppressed[j] || dets[j].cls != dets[i].cls) {
        continue;
      }
      if (iou_(dets[i], dets[j]) > iou_thresh) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

}

YoloDetectionStage::YoloDetectionStage(const SessionContextIntf* s,
                                       string                    id,
                                       vector<InEdge>            iports,
                                       FlexData                  config)
  : TypedStage<YoloDetectionStage>(s, std::move(id),
                                   std::move(iports),
                                   std::move(config))
{
  // Scalar attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _model_path           = attr_str("model_path");
  _models_db            = attr_str("models_db");
  if (_models_db.empty()) {
    _models_db = "models";
  }
  _input_feature_name   = attr_str("input_feature_name");
  _output_feature_name  = attr_str("output_feature_name");
  _confidence_threshold = attr_real("confidence_threshold");
  _iou_threshold        = attr_real("iou_threshold");
  _max_detections       = static_cast<int>(attr_int("max_detections"));
  _compute_units        = static_cast<int>(attr_int("compute_units"));
  _uses_cpu_only        = attr_bool("uses_cpu_only");
  _oport_capacity       = static_cast<unsigned>(attr_uint("oport_capacity"));

  // num_classes (custom: honoured only when > 0, tracks user-set) and
  // class_names (composite array) are read from the config directly.
  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("num_classes")) {
      const int nc =
          static_cast<int>(root.at("num_classes").as_int(0));
      if (nc > 0) {
        _num_classes          = nc;
        _num_classes_user_set = true;
      }
    }
    if (root.contains("class_names")) {
      auto arr = root.at("class_names");
      if (arr.is_array()) {
        auto av = arr.as_array();
        for (size_t i = 0, n = av.size(); i < n; ++i) {
          _class_names.emplace_back(av.at(i).as_string(""));
        }
      }
    }
  }
  // Validation is deferred to launch (see Stage::fail_config): a stage
  // must construct for any config so a graph can be built/edited first.
  if (_model_path.empty()) {
    fail_config(fmt(
        "YoloDetectionStage('{}'): model_path is required",
        this->id()));
  }
  // Only validate class_names size eagerly when num_classes was set
  // by the user. When num_classes is auto-detected, the check moves
  // to initialize() once the model output description is read.
  if (_num_classes_user_set
      && !_class_names.empty()
      && static_cast<int>(_class_names.size()) != _num_classes) {
    fail_config(fmt(
        "YoloDetectionStage('{}'): class_names size {} does not match "
        "num_classes {}",
        this->id(), _class_names.size(), _num_classes));
  }
  if (_oport_capacity == 0) {
    _oport_capacity = 4;
  }
  allocate_oports(spec().oports.size());
  set_oport_policy(0, {_oport_capacity, OverrunPolicy::Backpressure});
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "model_path", .type = ConfigType::String, .required = true,
   .doc = "YOLO CoreML model: a models-DB key (registered by model-fetch) "
          "or an mlpackage path; a DB key wins over a same-named path",
   .suggest_db = "models", .suggest_db_type = "yolo"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db model-fetch registers into", .def_str = "models"},
  {.key = "input_feature_name", .type = ConfigType::String,
   .doc = "model input feature name; unset (default) auto-matches the "
          "model's sole input"},
  {.key = "output_feature_name", .type = ConfigType::String,
   .doc = "model output feature name; unset (default) auto-matches the "
          "model's sole output"},
  {.key = "num_classes", .type = ConfigType::Int,
   .doc = "class count; auto-detected when absent/0", .def_int = 0},
  {.key = "confidence_threshold", .type = ConfigType::Real,
   .doc = "drop detections below this score", .def_real = 0.25},
  {.key = "iou_threshold", .type = ConfigType::Real,
   .doc = "per-class NMS IoU threshold", .def_real = 0.45},
  {.key = "max_detections", .type = ConfigType::Int,
   .doc = "max detections kept after NMS", .def_int = 300},
  {.key = "class_names", .type = ConfigType::Array,
   .doc = "label names; length must match num_classes"},
  {.key = "compute_units", .type = ConfigType::Int,
   .doc = "0=CPUOnly 1=CPU+GPU 2=All 3=CPU+ANE", .def_int = 2},
  {.key = "uses_cpu_only", .type = ConfigType::Bool,
   .doc = "inference-time override of compute_units", .def_bool = false},
  {.key = "oport_capacity", .type = ConfigType::Uint,
   .doc = "output ring capacity", .def_uint = 4},
};
const PortSpec kIports[] = {
  {.name = "frames", .doc = "RGB TensorBeat [3,H,W], f32 in [0,1]",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "detections",
   .doc = "FlexData {frame_width,frame_height,detections[{class_id,"
          "class_name,score,x1,y1,x2,y2}]}",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "yolo-detection",
  .doc       = "Runs a YOLOv6-shape CoreML detector on each RGB frame "
               "(letterbox + NMS) and emits a FlexData detection record "
               "per frame.",
  .display_name = "YOLO Detector",
  .category  = StageCategory::Vision,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
YoloDetectionStage::spec() const noexcept
{
  return kSpec;
}

YoloDetectionStage::~YoloDetectionStage()
{
  // _loaded drops via shared_ptr; predict() owns all CoreML lifetime.
}

Job
YoloDetectionStage::initialize(RuntimeContext& /*ctx*/)
{
  CoreMLModelManager* mgr =
      session() ? session()->coreml_model_manager() : nullptr;
  if (!mgr) {
    session()->error(fmt(
        "YoloDetectionStage('{}'): session has no CoreML model "
        "manager (build without VPIPE_BUILD_APPLE_SILICON?)",
        this->id()));
  }
  // A models-DB key (e.g. a model-fetch'd yolo model) resolves to its
  // unpacked .mlpackage; a plain path passes through unchanged.
  _model_path = resolve_model_dir(session(), _models_db, _model_path);
  _loaded = mgr->load(_model_path, _compute_units);
  if (!_loaded) {
    session()->error(fmt(
        "YoloDetectionStage('{}'): model load failed for '{}'",
        this->id(), _model_path));
  }

  // Cache the session-shared metal-compute device so the hot path can
  // branch on a Shared input without walking session() every call.
  // nullptr is fine; the gate below also checks _mc->valid().
  _mc = session()->metal_compute();

  // Read the model's input + output feature descriptors (from the
  // manager's cached introspection) to drive the rest of the stage:
  //   - TensorType (MLMultiArray) vs ImageType (CVPixelBuffer)
  //   - the model's expected input pixel size (or "flexible")
  //   - num_classes (when not user-supplied) from the output's last
  //     dim, assuming the YOLOv6-deploy contract (4 + num_classes)
  const auto& in_descs = _loaded->input_descs();

  // Auto-match the input feature name when unset: a typical detector has
  // exactly one input, so the name is unambiguous. With any other count
  // the user must name one explicitly.
  if (_input_feature_name.empty()) {
    if (in_descs.size() == 1) {
      _input_feature_name = in_descs.begin()->first;
    } else {
      session()->error(fmt(
          "YoloDetectionStage('{}'): input_feature_name is unset and "
          "the model does not have exactly one input ({} found); set "
          "input_feature_name to disambiguate.",
          this->id(), static_cast<int>(in_descs.size())));
    }
  }
  auto in_it = in_descs.find(_input_feature_name);
  if (in_it == in_descs.end()) {
    session()->error(fmt(
        "YoloDetectionStage('{}'): input '{}' not found on the model "
        "(or is neither a TensorType nor a supported ImageType)",
        this->id(), _input_feature_name));
  } else if (in_it->second.kind == CoreMLFeatureKind::MultiArray) {
    // TensorType path: model takes [1, 3, H, W] or [3, H, W] f32. When
    // any spatial dim is <= 0 the model declares a flexible input -- we
    // feed the source's native dims through unchanged. Both square and
    // rectangular fixed inputs are supported.
    const vector<int64_t>& ms = in_it->second.shape;
    int64_t mh = -1, mw = -1;
    if (ms.size() == 4 && ms[0] == 1 && ms[1] == 3) {
      mh = ms[2]; mw = ms[3];
    } else if (ms.size() == 3 && ms[0] == 3) {
      mh = ms[1]; mw = ms[2];
    }
    if (mh > 0 && mw > 0) {
      _input_width         = static_cast<int>(mw);
      _input_height        = static_cast<int>(mh);
      _input_size_flexible = false;
    } else {
      _input_size_flexible = true;
      _input_width         = 0;
      _input_height        = 0;
    }
  } else {
    // ImageType path: model takes a CVPixelBuffer (built inside
    // predict()). We only support fixed-size BGRA8888; reject other
    // formats/flexible sizes with a clear message.
    _image_input        = true;
    _image_pixel_format = in_it->second.pixel_format;
    const int mw = in_it->second.image_width;
    const int mh = in_it->second.image_height;
    if (mw == 0 || mh == 0) {
      session()->error(fmt(
          "YoloDetectionStage('{}'): ImageType inputs with flexible "
          "pixel dimensions are not supported. Re-export the model with "
          "a fixed image size, or use a TensorType input.",
          this->id()));
    } else {
      _input_width         = mw;
      _input_height        = mh;
      _input_size_flexible = false;
    }
    if (_image_pixel_format != kCVPixelFormatType_32BGRA) {
      session()->error(fmt(
          "YoloDetectionStage('{}'): image-type model declared "
          "pixel-format=0x{:x} but only kCVPixelFormatType_32BGRA "
          "(0x{:x}) is supported. Re-export the model with "
          "ct.ImageType(color_layout=ct.colorlayout.BGR) or use a "
          "TensorType input.",
          this->id(), _image_pixel_format,
          static_cast<uint32_t>(kCVPixelFormatType_32BGRA)));
    }
  }

  // Auto-match the output feature name when unset: a typical detector
  // has exactly one output, so the name is unambiguous. With any other
  // count the user must name one explicitly.
  if (_output_feature_name.empty()) {
    const auto& od = _loaded->output_descs();
    if (od.size() == 1) {
      _output_feature_name = od.begin()->first;
    } else {
      session()->error(fmt(
          "YoloDetectionStage('{}'): output_feature_name is unset and "
          "the model does not have exactly one output ({} found); set "
          "output_feature_name to disambiguate.",
          this->id(), static_cast<int>(od.size())));
    }
  }

  // Auto-detect num_classes when the user didn't supply it. The YOLO
  // output's last dim is either `4 + nc` (YOLOv8 / anchor-free heads)
  // or `5 + nc` (YOLOv5/v6/v7, where row[4] is an objectness score).
  // Without extra info we can't disambiguate the two for arbitrary
  // class counts, so the precedence is:
  //   1. class_names — nc = class_names.size() (most authoritative).
  //   2. COCO heuristic — if (last_dim - 5 == 80) OR (last_dim - 4
  //      == 80), assume nc = 80; the runtime stride check below picks
  //      the right objectness branch.
  //   3. Anchor-free fallback (last_dim - 4) when nothing else fits;
  //      this is correct for YOLOv8-style heads.
  if (!_num_classes_user_set) {
    int64_t last_dim = -1;
    const auto& outs = _loaded->output_descs();
    auto it = outs.find(_output_feature_name);
    if (it != outs.end() && it->second.fixed
        && !it->second.shape.empty()) {
      last_dim = it->second.shape.back();
    }
    if (!_class_names.empty()) {
      _num_classes = static_cast<int>(_class_names.size());
    } else if (last_dim >= 84
               && (last_dim - 4 == 80 || last_dim - 5 == 80)) {
      _num_classes = 80;        // standard COCO export
    } else if (last_dim > 4) {
      _num_classes = static_cast<int>(last_dim - 4);
    }
    if (_num_classes <= 0) {
      session()->error(fmt(
          "YoloDetectionStage('{}'): num_classes was not supplied and "
          "could not be auto-detected from the model's '{}' output "
          "(missing or non-fixed shape). Set num_classes in the stage "
          "config, or provide class_names.",
          this->id(), _output_feature_name));
    }
    // Sanity-check the derived nc against the output last-dim: it
    // must equal either 4 + nc (no objectness) or 5 + nc (with). If
    // neither matches, the auto-detection is wrong and the user
    // needs to specify explicitly.
    if (last_dim > 0
        && last_dim != 4 + _num_classes
        && last_dim != 5 + _num_classes) {
      session()->error(fmt(
          "YoloDetectionStage('{}'): auto-detected num_classes={} but "
          "the model's '{}' output last-dim {} matches neither "
          "{} (4+nc, anchor-free) nor {} (5+nc, with-objectness). "
          "Set num_classes explicitly in the stage config.",
          this->id(), _num_classes, _output_feature_name, last_dim,
          4 + _num_classes, 5 + _num_classes));
    }
    if (!_class_names.empty()
        && static_cast<int>(_class_names.size()) != _num_classes) {
      session()->error(fmt(
          "YoloDetectionStage('{}'): class_names size {} does not "
          "match the derived num_classes {} (from output last-dim)",
          this->id(), _class_names.size(), _num_classes));
    }
  }

  if (_input_size_flexible) {
    session()->info(fmt(
        "YoloDetectionStage('{}'): model declares flexible input "
        "shape; frames will be fed at native dims (num_classes={})",
        this->id(), _num_classes));
  } else {
    session()->info(fmt(
        "YoloDetectionStage('{}'): model input {}x{} ({}), "
        "num_classes={}",
        this->id(), _input_width, _input_height,
        _image_input ? "ImageType BGRA8888" : "TensorType f32",
        _num_classes));
  }
  co_return;
}

Job
YoloDetectionStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  if (!in) {
    ctx.signal_done();
    co_return;
  }
  const TensorBeatPayload* tin =
      dynamic_cast<const TensorBeatPayload*>(in.get());
  if (!tin || tin->shape.size() != 3 || tin->shape[0] != 3) {
    session()->warn(fmt(
        "YoloDetectionStage('{}'): ignoring beat (expected TensorBeat "
        "with shape [3, H, W])", this->id()));
    co_return;
  }

  const int H = static_cast<int>(tin->shape[1]);
  const int W = static_cast<int>(tin->shape[2]);
  // Model-expected input dims. For fixed-shape models, Sw=_input_width
  // and Sh=_input_height (equal for square exports, may differ for
  // rectangular exports). For flexible-shape models we forward the
  // source's native dims.
  const int Sw = _input_size_flexible ? W : _input_width;
  const int Sh = _input_size_flexible ? H : _input_height;
  const bool dims_match = (W == Sw && H == Sh);

  // Fast path: u8 source + ImageType BGRA model. Skip the u8→f32
  // and f32→u8 round-trips entirely; the letterbox + RGB→BGRA pack
  // happens in a single pass directly into the CVPixelBuffer below.
  const bool fast_u8_image =
      _image_input
      && _image_pixel_format == kCVPixelFormatType_32BGRA
      && tin->dtype == TensorBeat::DType::U8;

  // Metal BGRA fast path: when the u8 source lives in a Shared
  // MTL::Buffer and the model takes ImageType BGRA, dispatch the
  // GPU letterbox+RGB→BGRA pack kernel and skip the CPU loop.
  const bool fast_u8_image_metal =
      fast_u8_image
      && tin->storage_class() == TensorStorageClass::Shared
      && tin->external
      && _mc && _mc->valid()
      && tin->is_contiguous();

  // Skip-resample fast path: TensorType model, incoming dims already
  // match the model's expectation, and dtype is f32 — bind the source
  // buffer directly into the MLMultiArray. No resample, no copy.
  // Strided f32 inputs still take this path after a one-shot
  // materialize_contiguous_as<float>().
  const bool skip_resample =
      !_image_input
      && dims_match
      && tin->dtype == TensorBeat::DType::F32;

  LetterboxParams lb;

  // Metal letterbox fast path: when the input lives in a Shared
  // MTLBuffer (because video-to-rgb produced it via the NV12->RGB
  // Metal path) and we're going to the MLMultiArray TensorType
  // branch, dispatch the Metal letterbox kernel directly on the
  // input MTL::Buffer and produce the F32 model-input bytes in a
  // freshly-allocated Shared MTL::Buffer. CoreML / ANE consume the
  // bytes via MLMultiArray initWithDataPointer on the same address.
  // The kernel does bilinear-letterbox + u8→f32 normalisation in a
  // single dispatch — when dims already match the model the kernel
  // still runs (it degenerates to a pure format conversion with
  // scale=1 and pad=0), giving us a one-shot u8→f32 pass on the GPU.
  // The kernel handles both square and rectangular outputs; flexible-
  // shape models are excluded because the runtime decision tree
  // routes them through the skip path / CPU u8→f32 instead.
  std::unique_ptr<ExternalStorageHandle> model_in_shared;
  bool metal_letterbox_used = false;
  if (!fast_u8_image
      && !skip_resample
      && !_input_size_flexible
      && tin->dtype == TensorBeat::DType::U8
      && tin->storage_class() == TensorStorageClass::Shared
      && tin->external
      && _mc && _mc->valid()
      && tin->is_contiguous()) {
    const size_t need_dst =
        static_cast<size_t>(3) * Sw * Sh * sizeof(float);
    auto dst =
        metal_compute::make_shared_storage(*_mc, need_dst, session());
    if (dst) {
      float ls = 0.0f;
      int   px = 0, py = 0;
      const bool ok = metal_compute::letterbox_planar_u8_to_f32_chw(
          *_mc, *tin->external, W, H, *dst, Sw, Sh, &ls, &px, &py,
          session());
      if (ok) {
        lb.scale = ls;
        lb.pad_x = px;
        lb.pad_y = py;
        model_in_shared = std::move(dst);
        metal_letterbox_used = true;
        if (!_metal_letterbox_logged) {
          session()->info(fmt(
              "YoloDetectionStage('{}'): using metal letterbox "
              "fast path ({}x{} u8 shared -> {}x{} f32 shared)",
              this->id(), W, H, Sw, Sh));
          _metal_letterbox_logged = true;
        }
      }
    }
  }

  // Source RGB -- materialize contiguous if strided. The fast path
  // wants a u8 [3, H, W] pointer; the skip path wants an f32 [3, H,
  // W] pointer that will be bound zero-copy; the generic CPU path
  // builds an f32 letterboxed [3, Sh, Sw] buffer.
  AlignedVector<uint8_t> u8_src_buf;
  AlignedVector<float>   src_buf_f32;
  AlignedVector<uint8_t> src_buf_u8;
  const uint8_t*  u8_src    = nullptr;
  const float*    src       = nullptr;
  const float*    skip_ptr  = nullptr;
  vector<float>   model_in;

  if (metal_letterbox_used) {
    // No CPU-side preparation needed; model_in_shared owns the
    // letterboxed bytes the model will consume.
  } else if (fast_u8_image_metal) {
    // GPU BGRA fast path: no CPU-side preparation; the kernel reads
    // tin->external directly and writes BGRA bytes into the
    // CVPixelBuffer below.
  } else if (skip_resample) {
    if (tin->is_contiguous()) {
      skip_ptr = tin->as_f32();
    } else {
      src_buf_f32 = tin->materialize_contiguous_as<float>();
      skip_ptr    = src_buf_f32.data();
    }
  } else if (fast_u8_image) {
    if (tin->is_contiguous()) {
      u8_src = tin->as_u8();
    } else {
      u8_src_buf = tin->materialize_contiguous();
      u8_src     = u8_src_buf.data();
    }
  } else if (tin->dtype == TensorBeat::DType::F32) {
    if (tin->is_contiguous()) {
      src = tin->as_f32();
    } else {
      src_buf_f32 = tin->materialize_contiguous_as<float>();
      src         = src_buf_f32.data();
    }
  } else if (tin->dtype == TensorBeat::DType::U8) {
    const uint8_t* u8_ptr = nullptr;
    if (tin->is_contiguous()) {
      u8_ptr = tin->as_u8();
    } else {
      src_buf_u8 = tin->materialize_contiguous();
      u8_ptr     = src_buf_u8.data();
    }
    const size_t n = static_cast<size_t>(3) * H * W;
    src_buf_f32.assign(n, 0.0f);
    constexpr float kInv255 = 1.0f / 255.0f;
    for (size_t i = 0; i < n; ++i) {
      src_buf_f32[i] = static_cast<float>(u8_ptr[i]) * kInv255;
    }
    src = src_buf_f32.data();
  } else {
    session()->warn(fmt(
        "YoloDetectionStage('{}'): unsupported input dtype '{}' "
        "(expected f32 or u8) — dropping beat",
        this->id(), tin->dtype_name()));
    co_return;
  }

  // Build the f32 model-sized buffer when no fast path captured it.
  // The TensorType + dims_match case can bind the normalised source
  // directly; everything else letterboxes into a fresh buffer.
  if (!fast_u8_image && !metal_letterbox_used && !skip_resample) {
    if (!_image_input && dims_match) {
      // u8 source has already been normalised into src_buf_f32 by
      // the dtype branch above; bind that buffer directly.
      skip_ptr = src;
    } else {
      model_in.assign(static_cast<size_t>(3) * Sw * Sh, 0.0f);
      lb = letterbox_resize_(src, W, H, Sw, Sh, model_in.data());
    }
  }

  // Run inference through the native predict() API. The letterbox above
  // produced either f32 model-input bytes (TensorType) or the source we
  // pack into BGRA (ImageType); predict() owns all CoreML / CoreVideo
  // marshaling, the per-model serialization, and the f16/f64 -> f32
  // output decode (incl. non-contiguous strides).
  vector<int64_t> out_shape;
  vector<float>   out_data;
  {
    CoreMLPredictInput cin;
    cin.name = _input_feature_name;

    // Input feature bytes -- must outlive the predict() call below.
    std::unique_ptr<ExternalStorageHandle> bgra_shared;
    AlignedVector<uint8_t>                  bgra_host;

    if (_image_input) {
      // Produce Sw×Sh BGRA8888 bytes (tight rows); predict() stages them
      // into a CVPixelBuffer of the model's fixed size.
      const size_t row_bytes  = static_cast<size_t>(Sw) * 4u;
      const size_t bgra_bytes = row_bytes * static_cast<size_t>(Sh);
      const uint8_t* bgra_ptr = nullptr;
      if (fast_u8_image_metal) {
        // GPU letterbox + RGB->BGRA into a Shared buffer (no host loop).
        bgra_shared = metal_compute::make_shared_storage(
            *_mc, bgra_bytes, session());
        float ls = 0.0f;
        int   px = 0, py = 0;
        const bool packed_ok = bgra_shared
            && metal_compute::letterbox_planar_u8_to_bgra_u8(
                   *_mc, *tin->external, W, H, *bgra_shared, Sw, Sh,
                   &ls, &px, &py, session());
        if (!packed_ok) {
          session()->warn(fmt(
              "YoloDetectionStage('{}'): metal BGRA letterbox dispatch "
              "failed; dropping beat", this->id()));
          co_return;
        }
        lb.scale = ls;
        lb.pad_x = px;
        lb.pad_y = py;
        bgra_ptr = static_cast<const uint8_t*>(bgra_shared->contents);
        if (!_metal_bgra_logged) {
          session()->info(fmt(
              "YoloDetectionStage('{}'): using metal BGRA letterbox fast "
              "path ({}x{} u8 shared -> {}x{} BGRA8888)",
              this->id(), W, H, Sw, Sh));
          _metal_bgra_logged = true;
        }
      } else {
        bgra_host.assign(bgra_bytes, 0);
        if (fast_u8_image) {
          // Single-pass bilinear letterbox + RGB->BGRA pack (CPU) from
          // the u8 source.
          lb = letterbox_u8_to_bgra_(u8_src, W, H, Sw, Sh,
                                     bgra_host.data(), row_bytes);
        } else {
          // Pack the f32 model_in [0,1] letterboxed buffer into BGRA8888
          // with clamp + ×255 + round.
          const size_t plane = static_cast<size_t>(Sw) * Sh;
          const float* r_plane = model_in.data() + 0 * plane;
          const float* g_plane = model_in.data() + 1 * plane;
          const float* b_plane = model_in.data() + 2 * plane;
          for (int y = 0; y < Sh; ++y) {
            uint8_t* row =
                bgra_host.data() + static_cast<size_t>(y) * row_bytes;
            const float* rr = r_plane + static_cast<size_t>(y) * Sw;
            const float* gg = g_plane + static_cast<size_t>(y) * Sw;
            const float* bb = b_plane + static_cast<size_t>(y) * Sw;
            for (int x = 0; x < Sw; ++x) {
              const float fr = std::clamp(rr[x], 0.0f, 1.0f);
              const float fg = std::clamp(gg[x], 0.0f, 1.0f);
              const float fb = std::clamp(bb[x], 0.0f, 1.0f);
              row[x * 4 + 0] = static_cast<uint8_t>(fb * 255.0f + 0.5f);
              row[x * 4 + 1] = static_cast<uint8_t>(fg * 255.0f + 0.5f);
              row[x * 4 + 2] = static_cast<uint8_t>(fr * 255.0f + 0.5f);
              row[x * 4 + 3] = 0xFF;
            }
          }
        }
        bgra_ptr = bgra_host.data();
      }
      cin.image           = bgra_ptr;
      cin.image_width     = Sw;
      cin.image_height    = Sh;
      cin.image_row_bytes = row_bytes;
    } else {
      // TensorType: bind the f32 letterboxed buffer zero-copy as
      // [1, 3, Sh, Sw]. Sh == Sw for fixed-square models, and equal to
      // the source's native (H, W) for flexible-shape models.
      const int64_t Sw64 = static_cast<int64_t>(Sw);
      const int64_t Sh64 = static_cast<int64_t>(Sh);
      const float* in_ptr;
      if (metal_letterbox_used) {
        in_ptr =
            reinterpret_cast<const float*>(model_in_shared->contents);
      } else if (skip_ptr) {
        in_ptr = skip_ptr;
      } else {
        in_ptr = model_in.data();
      }
      cin.data    = in_ptr;
      cin.dtype   = CoreMLDType::F32;
      cin.shape   = { 1, 3, Sh64, Sw64 };
      cin.strides = { static_cast<int64_t>(3) * Sh64 * Sw64,
                      Sh64 * Sw64, Sw64, 1 };
    }

    CoreMLPredictOutput cout;
    cout.name = _output_feature_name;
    cout.want = CoreMLDType::F32;   // f16/f64 outputs decoded to f32

    const CoreMLPredictInput cins[1]  = { std::move(cin) };
    CoreMLPredictOutput      couts[1] = { std::move(cout) };

    // ANE timeline: the CoreML detection is the Apple-Neural-Engine job.
    record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin);
    const bool ok = _loaded->predict(cins, couts, _uses_cpu_only);
    record_perf_event_aux(kPerfLaneANE, kPerfAnePredictBegin + 1u);
    if (!ok) {
      session()->warn(fmt(
          "YoloDetectionStage('{}'): prediction failed", this->id()));
      co_return;
    }

    out_shape = couts[0].shape;
    const float* of = static_cast<const float*>(couts[0].data);
    if (!of) {
      session()->warn(fmt(
          "YoloDetectionStage('{}'): output '{}' missing",
          this->id(), _output_feature_name));
      co_return;
    }
    size_t n = 1;
    for (auto d : out_shape) {
      n *= static_cast<size_t>(d);
    }
    out_data.assign(of, of + n);

    if (!_logged_output_spec) {
      _logged_output_spec = true;
      string shape_s, samples_s;
      for (size_t i = 0; i < out_shape.size(); ++i) {
        if (i) { shape_s += ","; }
        shape_s += to_string(out_shape[i]);
      }
      const size_t kSample = std::min<size_t>(n, 16);
      for (size_t i = 0; i < kSample; ++i) {
        if (i) { samples_s += " "; }
        char b[32];
        snprintf(b, sizeof b, "%.4f", out_data[i]);
        samples_s += b;
      }
      session()->info(fmt(
          "YoloDetectionStage('{}'): output '{}' shape=[{}] first16=[{}]",
          this->id(), _output_feature_name, shape_s, samples_s));
    }
  }

  // Interpret the output. Accept [1, N, 4+nc] or [N, 4+nc].
  int N = 0;
  int stride = 0;
  if (out_shape.size() == 3 && out_shape[0] == 1) {
    N      = static_cast<int>(out_shape[1]);
    stride = static_cast<int>(out_shape[2]);
  } else if (out_shape.size() == 2) {
    N      = static_cast<int>(out_shape[0]);
    stride = static_cast<int>(out_shape[1]);
  } else {
    session()->warn(fmt(
        "YoloDetectionStage('{}'): unexpected predictions shape "
        "(rank {}); expected [1, N, 4+nc] or [N, 4+nc]",
        this->id(), out_shape.size()));
    co_return;
  }
  bool has_objectness;
  if (stride == 4 + _num_classes) {
    has_objectness = false;          // YOLOv8 / anchor-free head
  } else if (stride == 5 + _num_classes) {
    has_objectness = true;           // YOLOv5/v6/v7 head with objectness
  } else {
    session()->warn(fmt(
        "YoloDetectionStage('{}'): predictions last-dim {} matches "
        "neither 4+num_classes({}) nor 5+num_classes({})",
        this->id(), stride, 4 + _num_classes, 5 + _num_classes));
    co_return;
  }

  // Decode + threshold.
  vector<Det> dets;
  decode_predictions_(out_data.data(), N, stride, _num_classes,
                      has_objectness,
                      static_cast<float>(_confidence_threshold),
                      &dets);

  // NMS (per-class).
  const float iou_t = static_cast<float>(_iou_threshold);
  const size_t max_keep = static_cast<size_t>(_max_detections);
  vector<size_t> kept = nms_(dets, iou_t, max_keep);

  // Build FlexData output.
  FlexData out_fd = FlexData::make_object();
  auto root = out_fd.as_object();
  root.insert("frame_width",  FlexData::make_int(W));
  root.insert("frame_height", FlexData::make_int(H));

  FlexData dets_arr = FlexData::make_array();
  auto dets_view = dets_arr.as_array();
  dets_view.reserve(kept.size());
  for (size_t idx : kept) {
    const Det& d = dets[idx];
    // Inverse letterbox: model coords -> original frame coords.
    const float ox1 = (d.x1 - lb.pad_x) / lb.scale;
    const float oy1 = (d.y1 - lb.pad_y) / lb.scale;
    const float ox2 = (d.x2 - lb.pad_x) / lb.scale;
    const float oy2 = (d.y2 - lb.pad_y) / lb.scale;
    const float cx1 = std::clamp(ox1, 0.0f, static_cast<float>(W));
    const float cy1 = std::clamp(oy1, 0.0f, static_cast<float>(H));
    const float cx2 = std::clamp(ox2, 0.0f, static_cast<float>(W));
    const float cy2 = std::clamp(oy2, 0.0f, static_cast<float>(H));

    FlexData entry = FlexData::make_object();
    auto e = entry.as_object();
    e.insert("class_id", FlexData::make_int(d.cls));
    if (!_class_names.empty()
        && d.cls >= 0
        && d.cls < static_cast<int>(_class_names.size())) {
      e.insert("class_name",
               FlexData::make_string(_class_names[d.cls]));
    }
    e.insert("score", FlexData::make_real(d.score));
    e.insert("x1",    FlexData::make_real(cx1));
    e.insert("y1",    FlexData::make_real(cy1));
    e.insert("x2",    FlexData::make_real(cx2));
    e.insert("y2",    FlexData::make_real(cy2));
    dets_view.push_back(std::move(entry));
  }
  root.insert("detections", std::move(dets_arr));

  co_await ctx.write(0,
      make_payload<FlexDataPayload>(std::move(out_fd)));
}

VPIPE_REGISTER_STAGE(YoloDetectionStage)
VPIPE_REGISTER_SPEC(YoloDetectionStage, kSpec)

}
