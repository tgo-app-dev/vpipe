#include "generative-models/shared/coreml-vision-encoder.h"

#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/coreml/coreml-cpp/CoreML.hpp"
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <CoreVideo/CVPixelBuffer.h>
#include <Foundation/Foundation.hpp>

#include <algorithm>
#include <cmath>
#endif

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>


namespace vpipe::genai {

#ifdef VPIPE_BUILD_APPLE_SILICON

namespace {

// Helper: build an NS::String from a UTF-8 C string with the same
// life-cycle semantics yolo-detection-stage uses (autoreleased).
NS::String*
ns_str_(const std::string& s)
{
  return NS::String::string(s.c_str(), NS::StringEncoding::UTF8StringEncoding);
}

// Helper: walk a CoreML shape NS::Array<NS::Number*> into int64 dims.
std::vector<std::int64_t>
read_shape_(const NS::Array* a)
{
  std::vector<std::int64_t> out;
  if (!a) { return out; }
  const NS::UInteger n = a->count();
  out.reserve(n);
  for (NS::UInteger i = 0; i < n; ++i) {
    auto* num = a->object<NS::Number>(i);
    out.push_back(num ? num->longLongValue()
                      : static_cast<std::int64_t>(0));
  }
  return out;
}

// Helper template: copy CoreML output of either f16/f32/f64 dtype
// into a contiguous f32 vector.
template <class Src>
void
copy_cast_to_f32_(const void* src, std::size_t n, std::vector<float>* out)
{
  const Src* p = static_cast<const Src*>(src);
  out->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    (*out)[i] = static_cast<float>(p[i]);
  }
}

// Letterbox a planar [3, in_h, in_w] u8 RGB image (already resident in a
// metal-compute SharedBuffer `src`) into a BGRA CVPixelBuffer
// (aspect-preserving, 114/255 grey pad) via the
// `letterbox_planar_u8_to_bgra_u8` kernel. Zero-copy on the input: `src`
// is bound straight into the kernel -- no host staging. Returns false on
// any failure (the CVPixelBuffer is left unlocked).
bool
mc_letterbox_bgra_buf_(metal_compute::MetalCompute&        mc,
                       const metal_compute::SharedBuffer&  src,
                       int                                 in_w,
                       int                                 in_h,
                       CVPixelBufferRef                    pb,
                       const SessionContextIntf*           session)
{
  namespace mcn = vpipe::metal_compute;
  if (!pb || in_w <= 0 || in_h <= 0) { return false; }
  const int out_w = static_cast<int>(CVPixelBufferGetWidth(pb));
  const int out_h = static_cast<int>(CVPixelBufferGetHeight(pb));
  if (out_w <= 0 || out_h <= 0) { return false; }

  const std::size_t src_bytes =
      static_cast<std::size_t>(3) * in_w * in_h;
  if (src.empty() || src.byte_size() < src_bytes) { return false; }

  // Letterbox geometry (identical to the metal-runtime path).
  const float sx = static_cast<float>(out_w) / static_cast<float>(in_w);
  const float sy = static_cast<float>(out_h) / static_cast<float>(in_h);
  const float scale = sx < sy ? sx : sy;
  const int   new_w = static_cast<int>(std::round(scale * in_w));
  const int   new_h = static_cast<int>(std::round(scale * in_h));
  const int   pad_x = (out_w - new_w) / 2;
  const int   pad_y = (out_h - new_h) / 2;
  const float inv   = 1.0f / scale;

  CVReturn lk = CVPixelBufferLockBaseAddress(pb, 0);
  if (lk != kCVReturnSuccess) { return false; }
  auto* base =
      static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(pb));
  const std::size_t bpr = CVPixelBufferGetBytesPerRow(pb);
  if (!base || bpr < static_cast<std::size_t>(out_w) * 4u) {
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return false;
  }
  const std::size_t total = bpr * static_cast<std::size_t>(out_h);
  mcn::SharedBuffer dst = mc.make_shared_buffer(total);
  if (dst.empty()) {
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return false;
  }

  std::uint32_t params[4]  = {
      static_cast<std::uint32_t>(in_w),  static_cast<std::uint32_t>(in_h),
      static_cast<std::uint32_t>(out_w), static_cast<std::uint32_t>(out_h) };
  std::uint32_t params2[4] = {
      static_cast<std::uint32_t>(new_w), static_cast<std::uint32_t>(new_h),
      static_cast<std::uint32_t>(pad_x), static_cast<std::uint32_t>(pad_y) };
  float         params3[4] = { inv, 0.0f, 0.0f, 0.0f };
  std::uint32_t dst_bpr    = static_cast<std::uint32_t>(bpr);

  bool ok = false;
  mcn::ComputeLibrary lib =
      mc.load_library("letterbox_planar_u8_to_bgra_u8");
  mcn::ComputeFunction fn = lib.valid()
      ? lib.function("letterbox_planar_u8_to_bgra_u8")
      : mcn::ComputeFunction();
  if (fn.valid()) {
    mcn::CommandStream stream = mc.make_command_stream();
    {
      mcn::ComputeEncoder enc = stream.begin_compute();
      if (enc.valid()) {
        enc.set_function(fn);
        enc.set_buffer(0, src, 0);
        enc.set_buffer(1, dst, 0);
        enc.set_constant_bytes(2, params,   sizeof(params));
        enc.set_constant_bytes(3, params2,  sizeof(params2));
        enc.set_constant_bytes(4, params3,  sizeof(params3));
        enc.set_constant_bytes(5, &dst_bpr, sizeof(dst_bpr));
        const unsigned tew  = fn.thread_execution_width();
        const unsigned maxt = fn.max_total_threads_per_threadgroup();
        const unsigned tgh  = (tew == 0) ? 1u
                                         : std::max(1u, maxt / tew);
        enc.dispatch(
            mcn::LaunchDims{ static_cast<unsigned>(out_w),
                             static_cast<unsigned>(out_h), 1u },
            mcn::LaunchDims{ tew == 0 ? 1u : tew, tgh, 1u });
        ok = true;
      }
    }
    if (ok) {
      mcn::CommandStream::Fence cb = stream.commit();
      cb.wait();
      ok = cb.completed();
    }
  }
  if (ok) {
    std::memcpy(base, dst.contents(), total);
  } else if (session) {
    session->warn(fmt(
        "CoreMLVisionEncoder: metal-compute letterbox dispatch failed "
        "({}x{} -> {}x{} BGRA)", in_w, in_h, out_w, out_h));
  }
  CVPixelBufferUnlockBaseAddress(pb, 0);
  return ok;
}

// Host-input wrapper: stage planar [3,in_h,in_w] u8 RGB into a Shared
// buffer, then letterbox. Used when the source frame isn't already a
// metal-compute (Shared/UMA) buffer.
bool
mc_letterbox_bgra_(metal_compute::MetalCompute& mc,
                   const std::uint8_t*          rgb,
                   int                          in_w,
                   int                          in_h,
                   CVPixelBufferRef             pb,
                   const SessionContextIntf*    session)
{
  if (in_w <= 0 || in_h <= 0) { return false; }
  const std::size_t src_bytes =
      static_cast<std::size_t>(3) * in_w * in_h;
  metal_compute::SharedBuffer src = mc.make_shared_buffer(src_bytes);
  if (src.empty()) { return false; }
  std::memcpy(src.contents(), rgb, src_bytes);
  return mc_letterbox_bgra_buf_(mc, src, in_w, in_h, pb, session);
}

}

struct CoreMLVisionEncoder::Impl {
  const SessionContextIntf*               session    = nullptr;
  // metal-compute device for the letterbox preprocessing (both paths).
  metal_compute::MetalCompute*            mc         = nullptr;
  std::shared_ptr<CoreMLLoadedModel>      loaded;

  // Cached input feature contract. `input_names` holds the model's
  // image input(s): {"image"} for a single-frame tower, or
  // {"image0","image1"} for the temporal-pair video export. is_pair
  // mirrors the latter. `input_name` is the first input (dims +
  // pixel-format constraints are read from it; both pair inputs share
  // the same shape).
  std::vector<std::string>                input_names = {"image"};
  std::string                             input_name  = "image";
  bool                                    is_pair     = false;
  int                                     input_w     = 0;
  int                                     input_h     = 0;
  std::uint32_t                           pixel_fmt   = 0;

  // Cached output feature contract.
  std::string                             output_name = "image_features";
  int                                     n_tokens    = 0;
  int                                     hidden      = 0;

  // Grid dims fed into the Result / EncodedImage.
  int                                     grid_h      = 0;
  int                                     grid_w      = 0;


  // One-shot logging flags.
  bool                                    logged_output = false;
};

#else   // !VPIPE_BUILD_APPLE_SILICON

struct CoreMLVisionEncoder::Impl {};

#endif

CoreMLVisionEncoder::CoreMLVisionEncoder()
  : _p(std::make_unique<Impl>())
{
}

CoreMLVisionEncoder::~CoreMLVisionEncoder() = default;

int
CoreMLVisionEncoder::model_input_width() const noexcept
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  return _p->input_w;
#else
  return 0;
#endif
}

int
CoreMLVisionEncoder::model_input_height() const noexcept
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  return _p->input_h;
#else
  return 0;
#endif
}

int
CoreMLVisionEncoder::output_n_tokens() const noexcept
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  return _p->n_tokens;
#else
  return 0;
#endif
}

int
CoreMLVisionEncoder::output_hidden() const noexcept
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  return _p->hidden;
#else
  return 0;
#endif
}

std::unique_ptr<CoreMLVisionEncoder>
CoreMLVisionEncoder::create(const LoadSpec&             spec,
                            MlxRuntime*                 runtime,
                            const SessionContextIntf*   session)
{
#ifndef VPIPE_BUILD_APPLE_SILICON
  (void)spec; (void)runtime;
  if (session) {
    session->warn(fmt(
        "CoreMLVisionEncoder: build was compiled without "
        "VPIPE_BUILD_APPLE_SILICON; CoreML vision tower is "
        "unavailable"));
  }
  return nullptr;
#else
  if (!session) {
    return nullptr;
  }
  auto* coreml_mgr = session->coreml_model_manager();
  if (!coreml_mgr) {
    session->warn(fmt(
        "CoreMLVisionEncoder: no CoreMLModelManager on this session"));
    return nullptr;
  }
  auto* mc_dev = session->metal_compute();
  if (mc_dev == nullptr || !mc_dev->valid()) {
    session->warn(fmt(
        "CoreMLVisionEncoder: no metal-compute device on this session "
        "(the GPU letterbox kernel requires metal-compute)"));
    return nullptr;
  }
  auto loaded = coreml_mgr->load(spec.mlpackage_path, spec.compute_units);
  if (!loaded) {
    session->warn(fmt(
        "CoreMLVisionEncoder: failed to load CoreML model from '{}'",
        spec.mlpackage_path));
    return nullptr;
  }

  std::unique_ptr<CoreMLVisionEncoder> out(new CoreMLVisionEncoder());
  auto& impl = *out->_p;
  impl.session       = session;
  impl.mc            = mc_dev;
  impl.loaded        = loaded;
  (void)runtime;

  // Pull the input + output contracts off the model description so
  // we can preflight the letterbox dims and the output shape at
  // load time (matches the YOLO ImageType path's discovery pass).
  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto* desc = loaded->model()->modelDescription();
  if (!desc) {
    pool->release();
    session->warn(fmt(
        "CoreMLVisionEncoder: model description unavailable for '{}'",
        spec.mlpackage_path));
    return nullptr;
  }

  // Discover the image input layout: a single "image" input (the
  // still-frame tower), or "image0"+"image1" (the temporal-pair video
  // export that merges two frames per token in-model). Both pair
  // inputs share the same shape, so dims/pixel-format are read from
  // the first.
  {
    auto* in_dict = desc->inputDescriptionsByName();
    auto image_constraint = [&](const char* nm) {
      auto* fd = in_dict
          ? in_dict->object<CML::FeatureDescription>(ns_str_(nm))
          : nullptr;
      return fd ? fd->imageConstraint() : nullptr;
    };
    auto* c_single = image_constraint("image");
    decltype(c_single) im_constraint = nullptr;
    if (c_single) {
      impl.input_names = {"image"};
      impl.is_pair     = false;
      im_constraint    = c_single;
    } else {
      auto* c0 = image_constraint("image0");
      auto* c1 = image_constraint("image1");
      if (c0 && c1) {
        impl.input_names = {"image0", "image1"};
        impl.is_pair     = true;
        im_constraint    = c0;
      }
    }
    if (!im_constraint) {
      pool->release();
      session->warn(fmt(
          "CoreMLVisionEncoder: model '{}' has no recognised image "
          "input. Expected a single 'image' ImageType, or "
          "'image0'+'image1' for the temporal-pair video export. "
          "Re-export with ct.ImageType(color_layout=ct.colorlayout."
          "BGR).",
          spec.mlpackage_path));
      return nullptr;
    }
    impl.input_name = impl.input_names.front();
    const NS::UInteger mw = im_constraint->pixelsWide();
    const NS::UInteger mh = im_constraint->pixelsHigh();
    if (mw == 0 || mh == 0) {
      pool->release();
      session->warn(fmt(
          "CoreMLVisionEncoder: model declares flexible pixel "
          "dimensions; re-export with a fixed image size."));
      return nullptr;
    }
    impl.input_w   = static_cast<int>(mw);
    impl.input_h   = static_cast<int>(mh);
    impl.pixel_fmt = im_constraint->pixelFormatType();
    if (impl.pixel_fmt != kCVPixelFormatType_32BGRA) {
      pool->release();
      session->warn(fmt(
          "CoreMLVisionEncoder: model expects pixel format 0x{:x} "
          "but only kCVPixelFormatType_32BGRA (0x{:x}) is supported.",
          impl.pixel_fmt,
          static_cast<std::uint32_t>(kCVPixelFormatType_32BGRA)));
      return nullptr;
    }
  }

  // Output: prefer the cached descriptor map from the model manager
  // when it has exactly one entry; otherwise discover the single
  // output by name.
  const auto& outs = loaded->output_descs();
  if (outs.size() == 1) {
    const auto& entry = *outs.begin();
    impl.output_name = entry.first;
    const auto& sh = entry.second.shape;
    // Accept [n_tokens, hidden] or [1, n_tokens, hidden].
    if (sh.size() == 2) {
      impl.n_tokens = static_cast<int>(sh[0]);
      impl.hidden   = static_cast<int>(sh[1]);
    } else if (sh.size() == 3 && sh[0] == 1) {
      impl.n_tokens = static_cast<int>(sh[1]);
      impl.hidden   = static_cast<int>(sh[2]);
    } else if (sh.size() == 1) {
      // Flat — caller will need to reshape; bail with a clear
      // diagnostic since we can't infer hidden.
      pool->release();
      session->warn(fmt(
          "CoreMLVisionEncoder: output '{}' has rank-1 shape; "
          "expected [n_tokens, hidden] or [1, n_tokens, hidden].",
          impl.output_name));
      return nullptr;
    } else {
      pool->release();
      session->warn(fmt(
          "CoreMLVisionEncoder: output '{}' has unexpected rank {}; "
          "expected 2 ([n_tokens, hidden]) or 3 ([1, n_tokens, "
          "hidden]).",
          impl.output_name, sh.size()));
      return nullptr;
    }
  }
  pool->release();

  // Grid dims: prefer caller override, else infer from input pixels.
  if (spec.grid_h > 0 && spec.grid_w > 0) {
    impl.grid_h = spec.grid_h;
    impl.grid_w = spec.grid_w;
  } else {
    const int factor = std::max(1, spec.patch_size)
                       * std::max(1, spec.spatial_merge_size);
    impl.grid_h = impl.input_h / factor;
    impl.grid_w = impl.input_w / factor;
  }
  // If we still don't have n_tokens (output descs were empty),
  // fall back to grid_h * grid_w.
  if (impl.n_tokens == 0 && impl.grid_h > 0 && impl.grid_w > 0) {
    impl.n_tokens = impl.grid_h * impl.grid_w;
  }
  // Sanity check: grid * grid should match the output token count.
  if (impl.grid_h * impl.grid_w != impl.n_tokens) {
    session->warn(fmt(
        "CoreMLVisionEncoder: grid {}x{} = {} tokens does not match "
        "model output n_tokens={}; using model output count and "
        "treating the grid as a 1xN strip for mRoPE.",
        impl.grid_h, impl.grid_w, impl.grid_h * impl.grid_w,
        impl.n_tokens));
    impl.grid_h = 1;
    impl.grid_w = impl.n_tokens;
  }

  session->info(fmt(
      "CoreMLVisionEncoder: loaded '{}' -> in={}x{} BGRA ({}), "
      "out='{}' [{}, {}] grid {}x{}",
      spec.mlpackage_path,
      impl.input_w, impl.input_h,
      impl.is_pair ? "temporal-pair image0+image1"
                   : "single image",
      impl.output_name, impl.n_tokens, impl.hidden,
      impl.grid_h, impl.grid_w));

  out->_valid = true;
  return out;
#endif
}

#ifdef VPIPE_BUILD_APPLE_SILICON

std::optional<CoreMLVisionEncoder::Result>
CoreMLVisionEncoder::encode_frames_host_(
    const std::vector<const std::uint8_t*>& frames, int H, int W,
    const metal_compute::SharedBuffer* src_buf)
{
  // src_buf (when set) is a single Shared/UMA [3,H,W] u8 frame bound
  // straight into the letterbox kernel (zero-copy input); otherwise the
  // host `frames` are staged. Exactly one source must be present.
  const int n_src = src_buf ? 1 : static_cast<int>(frames.size());
  if (!_valid || n_src == 0 || H <= 0 || W <= 0) {
    return std::nullopt;
  }
  PerfAuxScope _perf(_p->session, kPerfLaneLLM, kGvidLlmVision,
                     kPerfLlmVisionBegin, static_cast<std::uint64_t>(n_src));
  if (!src_buf) {
    for (const std::uint8_t* f : frames) {
      if (!f) { return std::nullopt; }
    }
  }
  auto& impl = *_p;
  const int Sw = impl.input_w;
  const int Sh = impl.input_h;

  // 1-3. Letterbox each source frame into its own BGRA CVPixelBuffer
  // (GPU bilinear resample + aspect-preserving pad + RGB->BGRA pack).
  // CoreML retains its own reference during prediction; release_pbs()
  // drops ours on every exit path.
  std::vector<CVPixelBufferRef> pbs;
  pbs.reserve(n_src);
  auto release_pbs = [&pbs]() {
    for (CVPixelBufferRef p : pbs) {
      if (p) { CFRelease(p); }
    }
    pbs.clear();
  };
  for (int i = 0; i < n_src; ++i) {
    CVPixelBufferRef pb = nullptr;
    CVReturn cv_rc = CVPixelBufferCreate(
        kCFAllocatorDefault,
        static_cast<std::size_t>(Sw),
        static_cast<std::size_t>(Sh),
        impl.pixel_fmt,
        nullptr,
        &pb);
    if (cv_rc != kCVReturnSuccess || !pb) {
      if (impl.session) {
        impl.session->warn(fmt(
            "CoreMLVisionEncoder: CVPixelBufferCreate ({}x{}) rc={}",
            Sw, Sh, static_cast<int>(cv_rc)));
      }
      release_pbs();
      return std::nullopt;
    }
    pbs.push_back(pb);   // owned now; release_pbs() frees it

    // GPU letterbox + RGB->BGRA pack into the pixel buffer via
    // metal-compute (no MetalRuntime, no MLX). Zero-copy input when a
    // Shared buffer was supplied; else stage the host frame.
    const bool lb_ok = impl.mc &&
        (src_buf
            ? mc_letterbox_bgra_buf_(*impl.mc, *src_buf, W, H, pb,
                                     impl.session)
            : mc_letterbox_bgra_(*impl.mc, frames[i], W, H, pb,
                                 impl.session));
    if (!lb_ok) {
      release_pbs();
      return std::nullopt;
    }
  }

  // 4. Run CoreML predict. Serialised on the loaded model's
  // predict_mutex so concurrent encodes through the same model
  // queue cleanly (CoreML serialises internally anyway; this gives
  // a cheap observable mutex on the caller side).
  // Native-f16 output buffer (GPU/UMA). Filled from the CoreML
  // MultiArray below: fp16 output is memcpy'd straight in (zero cast),
  // f32/f64 cast to f16. No host-f32 round-trip.
  metal_compute::SharedBuffer emb16;
  {
    std::lock_guard<std::mutex> lk(impl.loaded->predict_mutex());
    auto* pool = NS::AutoreleasePool::alloc()->init();

    NS::Error* err = nullptr;
    // Bind each model image input to a pixel buffer. A single-input
    // model uses pbs[0]; the two-input video model binds image0/image1
    // to pbs[0]/pbs[1] (or replicates pbs[0] when only one frame was
    // supplied -- e.g. a still-image encode() through the pair model).
    const std::size_t nin = impl.input_names.size();
    std::vector<const NS::Object*> objs(nin);
    std::vector<const NS::Object*> keys(nin);
    bool fv_ok = true;
    for (std::size_t i = 0; i < nin; ++i) {
      CVPixelBufferRef use = (i < pbs.size()) ? pbs[i] : pbs.back();
      CML::FeatureValue* fv =
          CML::FeatureValue::featureValueWithPixelBuffer(use);
      if (!fv) { fv_ok = false; break; }
      objs[i] = fv;
      keys[i] = ns_str_(impl.input_names[i]);
    }
    if (!fv_ok) {
      pool->release();
      release_pbs();
      if (impl.session) {
        impl.session->warn(fmt(
            "CoreMLVisionEncoder: featureValueWithPixelBuffer "
            "returned null"));
      }
      return std::nullopt;
    }
    NS::Dictionary* dict = NS::Dictionary::dictionary(
        objs.data(), keys.data(), nin);
    auto* dfp = CML::DictionaryFeatureProvider::alloc()
                    ->initWithDictionary(dict, &err);
    if (err || !dfp) {
      release_pbs();
      pool->release();
      if (impl.session) {
        impl.session->warn(fmt(
            "CoreMLVisionEncoder: feature provider init failed"));
      }
      return std::nullopt;
    }
    // ANE timeline: the vision tower's CoreML inference is an Apple-
    // Neural-Engine job, so it also shows on the ANE lane (named
    // "vision-tower") -- nested inside the LLM-lane vision-tower block,
    // which spans the whole encode (letterbox + predict + gather).
    auto* result = [&] {
      PerfAuxScope _ane(impl.session, kPerfLaneANE, kGvidLlmVision,
                        kPerfAnePredictBegin);
      return impl.loaded->model()->predictionFromFeatures(dfp, &err);
    }();
    if (err || !result) {
      std::string desc = "(no NSError details)";
      if (err && err->localizedDescription()
          && err->localizedDescription()->utf8String()) {
        desc = err->localizedDescription()->utf8String();
      }
      dfp->release();
      release_pbs();
      pool->release();
      if (impl.session) {
        impl.session->warn(fmt(
            "CoreMLVisionEncoder: predictionFromFeatures failed: {}",
            desc));
      }
      return std::nullopt;
    }

    auto* out_key = ns_str_(impl.output_name);
    auto* val     = result->featureValueForName(out_key);
    auto* arr     = val ? val->multiArrayValue() : nullptr;
    if (!arr) {
      dfp->release();
      release_pbs();
      pool->release();
      if (impl.session) {
        impl.session->warn(fmt(
            "CoreMLVisionEncoder: output '{}' missing or not a "
            "MultiArray", impl.output_name));
      }
      return std::nullopt;
    }

    const std::vector<std::int64_t> out_shape =
        read_shape_(arr->shape());
    const std::vector<std::int64_t> out_strides =
        read_shape_(arr->strides());
    std::size_t n = 1;
    for (auto d : out_shape) {
      n *= static_cast<std::size_t>(d);
    }
    const std::size_t want = static_cast<std::size_t>(impl.n_tokens)
                             * static_cast<std::size_t>(impl.hidden);
    if (n != want) {
      dfp->release();
      release_pbs();
      pool->release();
      if (impl.session) {
        impl.session->warn(fmt(
            "CoreMLVisionEncoder: output element count {} != "
            "expected {} ({} tokens * {} hidden); dropping image",
            n, want, impl.n_tokens, impl.hidden));
      }
      return std::nullopt;
    }

    // Check row-major contiguity; if non-contiguous, fall back to a
    // strided gather (mirrors yolo's defensive output path).
    std::vector<std::int64_t> expected_strides(out_shape.size(), 1);
    {
      std::int64_t running = 1;
      for (std::size_t d = out_shape.size(); d-- > 0;) {
        expected_strides[d] = running;
        running *= out_shape[d];
      }
    }
    const bool contig =
        out_strides.size() == out_shape.size()
        && out_strides == expected_strides;

    const CML::MultiArrayDataType dt = arr->dataType();
    const void* src = arr->dataPointer();
    if (dt != CML::MultiArrayDataTypeFloat32
        && dt != CML::MultiArrayDataTypeFloat16
        && dt != CML::MultiArrayDataTypeDouble) {
      dfp->release();
      release_pbs();
      pool->release();
      if (impl.session) {
        impl.session->warn(fmt(
            "CoreMLVisionEncoder: unsupported output dtype {}; "
            "expected Float32/Float16/Double",
            static_cast<long long>(dt)));
      }
      return std::nullopt;
    }
    // Write the [n_tokens, hidden] embeddings as f16 into a SharedBuffer.
    emb16 = impl.mc ? impl.mc->make_shared_buffer(n * 2) : metal_compute::SharedBuffer();
    if (emb16.empty()) {
      dfp->release();
      release_pbs();
      pool->release();
      if (impl.session) {
        impl.session->warn(fmt(
            "CoreMLVisionEncoder: make_shared_buffer({} f16) failed", n));
      }
      return std::nullopt;
    }
    auto* dp = static_cast<_Float16*>(emb16.contents());
    if (contig) {
      if (dt == CML::MultiArrayDataTypeFloat16) {
        std::memcpy(dp, src, n * 2);   // fp16 -> f16, zero cast
      } else if (dt == CML::MultiArrayDataTypeFloat32) {
        const auto* p = static_cast<const float*>(src);
        for (std::size_t i = 0; i < n; ++i) { dp[i] = (_Float16)p[i]; }
      } else {
        const auto* p = static_cast<const double*>(src);
        for (std::size_t i = 0; i < n; ++i) { dp[i] = (_Float16)p[i]; }
      }
    } else {
      auto gather = [&](std::size_t flat) -> float {
        std::size_t off = 0;
        std::size_t rem = flat;
        for (std::size_t d = out_shape.size(); d-- > 0;) {
          const std::size_t s = static_cast<std::size_t>(out_shape[d]);
          const std::size_t idx = (s == 0) ? 0 : (rem % s);
          if (s != 0) { rem /= s; }
          off += idx * static_cast<std::size_t>(out_strides[d]);
        }
        if (dt == CML::MultiArrayDataTypeFloat32) {
          return static_cast<const float*>(src)[off];
        } else if (dt == CML::MultiArrayDataTypeFloat16) {
          return static_cast<float>(static_cast<const __fp16*>(src)[off]);
        }
        return static_cast<float>(static_cast<const double*>(src)[off]);
      };
      for (std::size_t i = 0; i < n; ++i) {
        dp[i] = (_Float16)gather(i);
      }
    }

    if (!impl.logged_output) {
      impl.logged_output = true;
      const char* dt_s =
          dt == CML::MultiArrayDataTypeFloat32 ? "Float32" :
          dt == CML::MultiArrayDataTypeFloat16 ? "Float16" : "Double";
      if (impl.session) {
        impl.session->info(fmt(
            "CoreMLVisionEncoder: output '{}' [{}, {}] dtype={} "
            "contig={}",
            impl.output_name, impl.n_tokens, impl.hidden, dt_s,
            contig ? "true" : "false"));
      }
    }

    dfp->release();
    pool->release();
  }
  release_pbs();

  // 5. Return the native-f16 GPU embeddings. The MLX path's
  // encode()/encode_pair() wrap these in an mc::array.
  Result result;
  result.embeddings = std::move(emb16);
  result.n_tokens   = impl.n_tokens;
  result.out_hidden = impl.hidden;
  result.grid_h     = impl.grid_h;
  result.grid_w     = impl.grid_w;
  return result;
}

CoreMLVisionEncoder::Result
CoreMLVisionEncoder::encode_host(const std::uint8_t* rgb, int H, int W)
{
  auto r = encode_frames_host_({rgb}, H, W);
  return r ? std::move(*r) : Result{};
}

CoreMLVisionEncoder::Result
CoreMLVisionEncoder::encode_host(const metal_compute::SharedBuffer& src,
                                 int H, int W)
{
  // Zero-copy input: `src` is a Shared/UMA [3,H,W] u8 frame bound
  // straight into the letterbox kernel (no host staging).
  auto r = encode_frames_host_({}, H, W, &src);
  return r ? std::move(*r) : Result{};
}

CoreMLVisionEncoder::Result
CoreMLVisionEncoder::encode_pair_host(const std::uint8_t* rgbA,
                                     const std::uint8_t* rgbB,
                                     int H, int W)
{
  if (!_valid || !_p->is_pair) {
    return Result{};
  }
  auto r = encode_frames_host_({rgbA, rgbB}, H, W);
  return r ? std::move(*r) : Result{};
}

bool
CoreMLVisionEncoder::supports_temporal_pair() const noexcept
{
  return _valid && _p->is_pair;
}


#else   // !VPIPE_BUILD_APPLE_SILICON

CoreMLVisionEncoder::Result
CoreMLVisionEncoder::encode_host(const std::uint8_t*, int, int)
{
  return Result{};
}

CoreMLVisionEncoder::Result
CoreMLVisionEncoder::encode_pair_host(const std::uint8_t*,
                                     const std::uint8_t*, int, int)
{
  return Result{};
}

bool
CoreMLVisionEncoder::supports_temporal_pair() const noexcept
{
  return false;
}

std::optional<CoreMLVisionEncoder::Result>
CoreMLVisionEncoder::encode_frames_host_(
    const std::vector<const std::uint8_t*>&, int, int)
{
  return std::nullopt;
}

#endif

}
