#include "generative-models/shared/coreml-vision-encoder.h"

#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/coreml/coreml-model-manager.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

// CoreVideo only for the BGRA pixel-format constant used when validating
// an ImageType model; predict() owns all CVPixelBuffer construction.
#include <CoreVideo/CVPixelBuffer.h>

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

// Letterbox a planar [3, in_h, in_w] u8 RGB image (already resident in a
// metal-compute SharedBuffer `src`) into a BGRA8888 byte buffer laid out
// tightly (out_w*4 bytes/row) in a freshly-allocated Shared/UMA buffer,
// aspect-preserving with 114/255 grey pad, via the
// `letterbox_planar_u8_to_bgra_u8` kernel. The bytes are handed to
// CoreMLLoadedModel::predict() as an image input (predict owns the
// CVPixelBuffer). Empty SharedBuffer on failure.
metal_compute::SharedBuffer
mc_letterbox_bgra_buf_(metal_compute::MetalCompute&        mc,
                       const metal_compute::SharedBuffer&  src,
                       int in_w, int in_h, int out_w, int out_h,
                       const SessionContextIntf*           session)
{
  namespace mcn = vpipe::metal_compute;
  if (in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) { return {}; }
  const std::size_t src_bytes =
      static_cast<std::size_t>(3) * in_w * in_h;
  if (src.empty() || src.byte_size() < src_bytes) { return {}; }

  // Letterbox geometry (identical to the f16 path).
  const float sx = static_cast<float>(out_w) / static_cast<float>(in_w);
  const float sy = static_cast<float>(out_h) / static_cast<float>(in_h);
  const float scale = sx < sy ? sx : sy;
  const int   new_w = static_cast<int>(std::round(scale * in_w));
  const int   new_h = static_cast<int>(std::round(scale * in_h));
  const int   pad_x = (out_w - new_w) / 2;
  const int   pad_y = (out_h - new_h) / 2;
  const float inv   = 1.0f / scale;

  const std::size_t bpr   = static_cast<std::size_t>(out_w) * 4u;  // tight
  const std::size_t total = bpr * static_cast<std::size_t>(out_h);
  mcn::SharedBuffer dst = mc.make_shared_buffer(total);
  if (dst.empty()) { return {}; }

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
  if (!ok) {
    if (session) {
      session->warn(fmt(
          "CoreMLVisionEncoder: metal-compute BGRA letterbox dispatch "
          "failed ({}x{} -> {}x{})", in_w, in_h, out_w, out_h));
    }
    return {};
  }
  return dst;
}

// Host-input wrapper: stage planar [3,in_h,in_w] u8 RGB into a Shared
// buffer, then letterbox. Used when the source frame isn't already a
// metal-compute (Shared/UMA) buffer.
metal_compute::SharedBuffer
mc_letterbox_bgra_(metal_compute::MetalCompute& mc,
                   const std::uint8_t*          rgb,
                   int in_w, int in_h, int out_w, int out_h,
                   const SessionContextIntf*    session)
{
  if (in_w <= 0 || in_h <= 0) { return {}; }
  const std::size_t src_bytes =
      static_cast<std::size_t>(3) * in_w * in_h;
  metal_compute::SharedBuffer src = mc.make_shared_buffer(src_bytes);
  if (src.empty()) { return {}; }
  std::memcpy(src.contents(), rgb, src_bytes);
  return mc_letterbox_bgra_buf_(mc, src, in_w, in_h, out_w, out_h, session);
}

// Letterbox a planar [3,in_h,in_w] u8 RGB image (resident in a Shared/UMA
// buffer `src`) into an f16 RGB tensor in a freshly-allocated Shared/UMA
// buffer, laid out NCHW (chw) or NHWC, via letterbox_planar_u8_to_rgb_f16.
// Zero-copy: the returned buffer is wrapped DIRECTLY as a CoreML MLMultiArray
// (no CVPixelBuffer, no lock, no host memcpy). Empty SharedBuffer on failure.
metal_compute::SharedBuffer
mc_letterbox_rgb_f16_buf_(metal_compute::MetalCompute&        mc,
                          const metal_compute::SharedBuffer&  src,
                          int in_w, int in_h, int out_w, int out_h,
                          bool chw, const SessionContextIntf* session)
{
  namespace mcn = vpipe::metal_compute;
  if (in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0) { return {}; }
  const std::size_t src_bytes =
      static_cast<std::size_t>(3) * in_w * in_h;
  if (src.empty() || src.byte_size() < src_bytes) { return {}; }

  // Letterbox geometry (identical to the BGRA path).
  const float sx = static_cast<float>(out_w) / static_cast<float>(in_w);
  const float sy = static_cast<float>(out_h) / static_cast<float>(in_h);
  const float scale = sx < sy ? sx : sy;
  const int   new_w = static_cast<int>(std::round(scale * in_w));
  const int   new_h = static_cast<int>(std::round(scale * in_h));
  const int   pad_x = (out_w - new_w) / 2;
  const int   pad_y = (out_h - new_h) / 2;
  const float inv   = 1.0f / scale;

  const std::size_t dst_elems =
      static_cast<std::size_t>(3) * out_w * out_h;
  mcn::SharedBuffer dst = mc.make_shared_buffer(dst_elems * 2);   // f16
  if (dst.empty()) { return {}; }

  std::uint32_t params[4]  = {
      static_cast<std::uint32_t>(in_w),  static_cast<std::uint32_t>(in_h),
      static_cast<std::uint32_t>(out_w), static_cast<std::uint32_t>(out_h) };
  std::uint32_t params2[4] = {
      static_cast<std::uint32_t>(new_w), static_cast<std::uint32_t>(new_h),
      static_cast<std::uint32_t>(pad_x), static_cast<std::uint32_t>(pad_y) };
  float         params3[4] = { inv, 0.0f, 0.0f, 0.0f };
  std::uint32_t chw_u      = chw ? 1u : 0u;

  bool ok = false;
  mcn::ComputeLibrary lib =
      mc.load_library("letterbox_planar_u8_to_rgb_f16");
  mcn::ComputeFunction fn = lib.valid()
      ? lib.function("letterbox_planar_u8_to_rgb_f16")
      : mcn::ComputeFunction();
  if (fn.valid()) {
    mcn::CommandStream stream = mc.make_command_stream();
    {
      mcn::ComputeEncoder enc = stream.begin_compute();
      if (enc.valid()) {
        enc.set_function(fn);
        enc.set_buffer(0, src, 0);
        enc.set_buffer(1, dst, 0);
        enc.set_constant_bytes(2, params,  sizeof(params));
        enc.set_constant_bytes(3, params2, sizeof(params2));
        enc.set_constant_bytes(4, params3, sizeof(params3));
        enc.set_constant_bytes(5, &chw_u, sizeof(chw_u));
        const unsigned tew  = fn.thread_execution_width();
        const unsigned maxt = fn.max_total_threads_per_threadgroup();
        const unsigned tgh  = (tew == 0) ? 1u : std::max(1u, maxt / tew);
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
  if (!ok) {
    if (session) {
      session->warn(fmt(
          "CoreMLVisionEncoder: metal-compute rgb-f16 letterbox dispatch "
          "failed ({}x{} -> {}x{})", in_w, in_h, out_w, out_h));
    }
    return {};
  }
  return dst;
}

// Host-input wrapper: stage planar [3,in_h,in_w] u8 RGB into a Shared
// buffer, then letterbox to f16. Used when the source frame isn't already a
// metal-compute (Shared/UMA) buffer.
metal_compute::SharedBuffer
mc_letterbox_rgb_f16_(metal_compute::MetalCompute& mc,
                      const std::uint8_t* rgb, int in_w, int in_h,
                      int out_w, int out_h, bool chw,
                      const SessionContextIntf* session)
{
  if (in_w <= 0 || in_h <= 0) { return {}; }
  const std::size_t src_bytes =
      static_cast<std::size_t>(3) * in_w * in_h;
  metal_compute::SharedBuffer src = mc.make_shared_buffer(src_bytes);
  if (src.empty()) { return {}; }
  std::memcpy(src.contents(), rgb, src_bytes);
  return mc_letterbox_rgb_f16_buf_(mc, src, in_w, in_h, out_w, out_h, chw,
                                   session);
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

  // MultiArray-input path (zero-copy f16 RGB): set when the model's image
  // input(s) are MLMultiArray (not ImageType). The frame is letterboxed
  // straight into an f16 RGB tensor bound AS the MLMultiArray -- no
  // CVPixelBuffer, no host copy. marray_chw selects NCHW vs NHWC;
  // marray_shape is the exact shape used to wrap the f16 buffer.
  bool                                    input_is_marray = false;
  bool                                    marray_chw      = true;
  std::vector<std::int64_t>               marray_shape;

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

bool
CoreMLVisionEncoder::input_is_multiarray() const noexcept
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  return _p->input_is_marray;
#else
  return false;
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

  // Pull the input + output contracts off the model's cached
  // descriptors (input_descs / output_descs) so we can preflight the
  // letterbox dims and the output shape at load time. No CoreML types
  // here -- the manager already introspected the model.
  {
    const auto& in_descs = loaded->input_descs();
    auto find_in = [&](const char* nm) -> const CoreMLInputDesc* {
      auto it = in_descs.find(nm);
      return (it == in_descs.end()) ? nullptr : &it->second;
    };

    // Discover the image input layout: a single "image" input (the
    // still-frame tower), or "image0"+"image1" (the temporal-pair video
    // export that merges two frames per token in-model). Both pair
    // inputs share the same shape. Each may be an ImageType (BGRA path)
    // OR an MLMultiArray (zero-copy f16 RGB path).
    const bool have_single = find_in("image") != nullptr;
    const bool have_pair   = find_in("image0") != nullptr
                          && find_in("image1") != nullptr;
    if (have_single) {
      impl.input_names = {"image"};
      impl.is_pair     = false;
    } else if (have_pair) {
      impl.input_names = {"image0", "image1"};
      impl.is_pair     = true;
    } else {
      session->warn(fmt(
          "CoreMLVisionEncoder: model '{}' has no recognised image "
          "input. Expected a single 'image' (or 'image0'+'image1') "
          "ImageType or MLMultiArray input.",
          spec.mlpackage_path));
      return nullptr;
    }
    impl.input_name = impl.input_names.front();
    const CoreMLInputDesc* d = find_in(impl.input_name.c_str());

    if (d && d->kind == CoreMLFeatureKind::Image) {
      // ---- ImageType path: BGRA (predict() builds the CVPixelBuffer) --
      if (d->image_width == 0 || d->image_height == 0) {
        session->warn(fmt(
            "CoreMLVisionEncoder: model declares flexible pixel "
            "dimensions; re-export with a fixed image size."));
        return nullptr;
      }
      impl.input_w   = d->image_width;
      impl.input_h   = d->image_height;
      impl.pixel_fmt = d->pixel_format;
      if (impl.pixel_fmt != kCVPixelFormatType_32BGRA) {
        session->warn(fmt(
            "CoreMLVisionEncoder: model expects pixel format 0x{:x} "
            "but only kCVPixelFormatType_32BGRA (0x{:x}) is supported.",
            impl.pixel_fmt,
            static_cast<std::uint32_t>(kCVPixelFormatType_32BGRA)));
        return nullptr;
      }
    } else if (d && d->kind == CoreMLFeatureKind::MultiArray) {
      // ---- MultiArray path: zero-copy f16 RGB tensor ----
      if (d->dtype != CoreMLDType::F16) {
        session->warn(fmt(
            "CoreMLVisionEncoder: MLMultiArray input '{}' is not "
            "Float16; re-export the image input as float16 for the "
            "zero-copy path.", impl.input_name));
        return nullptr;
      }
      // Layout from the shape: [.,3,H,W] -> NCHW, [.,H,W,3] -> NHWC.
      const std::vector<std::int64_t>& sh = d->shape;
      int Hh = 0, Ww = 0; bool chw = true, ok_shape = false;
      if (sh.size() == 4) {
        if (sh[1] == 3) { chw = true;  Hh = (int)sh[2]; Ww = (int)sh[3];
                          ok_shape = true; }
        else if (sh[3] == 3) { chw = false; Hh = (int)sh[1]; Ww = (int)sh[2];
                          ok_shape = true; }
      } else if (sh.size() == 3) {
        if (sh[0] == 3) { chw = true;  Hh = (int)sh[1]; Ww = (int)sh[2];
                          ok_shape = true; }
        else if (sh[2] == 3) { chw = false; Hh = (int)sh[0]; Ww = (int)sh[1];
                          ok_shape = true; }
      }
      if (!ok_shape || Hh <= 0 || Ww <= 0) {
        session->warn(fmt(
            "CoreMLVisionEncoder: MLMultiArray input '{}' shape is not a "
            "fixed [..,3,H,W] (NCHW) or [..,H,W,3] (NHWC) RGB tensor.",
            impl.input_name));
        return nullptr;
      }
      impl.input_is_marray = true;
      impl.marray_chw      = chw;
      impl.marray_shape    = sh;
      impl.input_w         = Ww;
      impl.input_h         = Hh;
    } else {
      session->warn(fmt(
          "CoreMLVisionEncoder: input '{}' is neither an ImageType nor "
          "an MLMultiArray.", impl.input_name));
      return nullptr;
    }
  }

  // Output: use the cached descriptor map from the model manager.
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
      // Flat — caller would need to reshape; bail since we can't infer
      // hidden.
      session->warn(fmt(
          "CoreMLVisionEncoder: output '{}' has rank-1 shape; "
          "expected [n_tokens, hidden] or [1, n_tokens, hidden].",
          impl.output_name));
      return nullptr;
    } else {
      session->warn(fmt(
          "CoreMLVisionEncoder: output '{}' has unexpected rank {}; "
          "expected 2 ([n_tokens, hidden]) or 3 ([1, n_tokens, "
          "hidden]).",
          impl.output_name, sh.size()));
      return nullptr;
    }
  }

  // Grid dims: prefer caller override, else infer from input pixels.
  const bool grid_inferred = !(spec.grid_h > 0 && spec.grid_w > 0);
  if (!grid_inferred) {
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
  // Sanity check: grid * grid should match the output token count. A
  // mismatch on an INFERRED grid is expected and benign for pooled
  // soft-token exports (e.g. Gemma-4: the ViT pools to a fixed token
  // count that is not a patch grid) -- the model output count is
  // authoritative and a 1xN strip drives plain 1-D RoPE. Only an
  // explicit, wrong caller grid warrants a warning.
  if (impl.grid_h * impl.grid_w != impl.n_tokens) {
    auto msg = [&] {
      return fmt(
          "CoreMLVisionEncoder: {}x{} patch grid ({} tokens) != model "
          "output n_tokens={}; using the model count as a 1xN strip "
          "(1-D RoPE).",
          impl.grid_h, impl.grid_w, impl.grid_h * impl.grid_w,
          impl.n_tokens);
    };
    if (grid_inferred) { session->info(msg()); }
    else               { session->warn(msg()); }
    impl.grid_h = 1;
    impl.grid_w = impl.n_tokens;
  }

  session->info(fmt(
      "CoreMLVisionEncoder: loaded '{}' -> in={}x{} {} ({}), "
      "out='{}' [{}, {}] grid {}x{}",
      spec.mlpackage_path,
      impl.input_w, impl.input_h,
      impl.input_is_marray
          ? (impl.marray_chw ? "RGB-f16 MLMultiArray NCHW"
                             : "RGB-f16 MLMultiArray NHWC")
          : "BGRA CVPixelBuffer",
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

  // 1-3. Letterbox each source frame into the model's input form (the
  // backings are kept alive across predict()):
  //   * ImageType  -> BGRA8888 bytes in a Shared buffer; predict() stages
  //                   them into a CVPixelBuffer of the model's fixed size.
  //   * MultiArray -> an f16 RGB tensor in a Shared/UMA buffer, bound
  //                   zero-copy as the MLMultiArray input.
  std::vector<metal_compute::SharedBuffer> in_bufs;
  in_bufs.reserve(n_src);
  for (int i = 0; i < n_src; ++i) {
    if (!impl.mc) { return std::nullopt; }
    metal_compute::SharedBuffer dst;
    if (impl.input_is_marray) {
      dst = src_buf
          ? mc_letterbox_rgb_f16_buf_(*impl.mc, *src_buf, W, H, Sw, Sh,
                                      impl.marray_chw, impl.session)
          : mc_letterbox_rgb_f16_(*impl.mc, frames[i], W, H, Sw, Sh,
                                  impl.marray_chw, impl.session);
    } else {
      dst = src_buf
          ? mc_letterbox_bgra_buf_(*impl.mc, *src_buf, W, H, Sw, Sh,
                                   impl.session)
          : mc_letterbox_bgra_(*impl.mc, frames[i], W, H, Sw, Sh,
                               impl.session);
    }
    if (dst.empty()) { return std::nullopt; }
    in_bufs.push_back(std::move(dst));
  }

  // 4. Build the predict inputs. A single-input model uses index 0; the
  // two-input video model binds image0/image1 to [0]/[1] (replicating
  // frame 0 when only one frame was supplied -- a still encode() through
  // the pair model).
  const std::size_t nin = impl.input_names.size();
  std::vector<CoreMLPredictInput> ins(nin);
  for (std::size_t i = 0; i < nin; ++i) {
    const metal_compute::SharedBuffer& buf =
        (i < in_bufs.size()) ? in_bufs[i] : in_bufs.back();
    ins[i].name = impl.input_names[i];
    if (impl.input_is_marray) {
      ins[i].data  = buf.contents();
      ins[i].dtype = CoreMLDType::F16;
      ins[i].shape = impl.marray_shape;
    } else {
      ins[i].image =
          static_cast<const std::uint8_t*>(buf.contents());
      ins[i].image_width     = Sw;
      ins[i].image_height    = Sh;
      ins[i].image_row_bytes = static_cast<std::size_t>(Sw) * 4u;
    }
  }

  // Output: native f16 [n_tokens, hidden] into a Shared/UMA buffer.
  // Passing it as a zero-copy backing lets CoreML write f16 directly
  // when the model output is fixed-shape f16; otherwise predict() casts
  // the result (f32/f64) into it. No host-f32 round-trip either way.
  const std::size_t n = static_cast<std::size_t>(impl.n_tokens)
                        * static_cast<std::size_t>(impl.hidden);
  metal_compute::SharedBuffer emb16 =
      impl.mc ? impl.mc->make_shared_buffer(n * 2)
              : metal_compute::SharedBuffer();
  if (emb16.empty()) {
    if (impl.session) {
      impl.session->warn(fmt(
          "CoreMLVisionEncoder: make_shared_buffer({} f16) failed", n));
    }
    return std::nullopt;
  }

  CoreMLPredictOutput out;
  out.name          = impl.output_name;
  out.want          = CoreMLDType::F16;
  out.backing       = emb16.contents();
  out.backing_elems = n;
  std::vector<CoreMLPredictOutput> outs;
  outs.push_back(std::move(out));

  bool ok;
  {
    // ANE timeline: the vision tower's CoreML inference is an Apple-
    // Neural-Engine job (nested inside the LLM-lane vision-tower block,
    // which spans the whole encode: letterbox + predict + gather).
    PerfAuxScope _ane(impl.session, kPerfLaneANE, kGvidLlmVision,
                      kPerfAnePredictBegin);
    ok = impl.loaded->predict(ins, outs);
  }
  if (!ok) {
    if (impl.session) {
      impl.session->warn(fmt(
          "CoreMLVisionEncoder: prediction failed"));
    }
    return std::nullopt;
  }
  // predict() fills outs[0].shape from the result; sanity-check the
  // element count against the [n_tokens, hidden] contract.
  std::size_t got = 1;
  for (auto d : outs[0].shape) { got *= static_cast<std::size_t>(d); }
  if (got != n) {
    if (impl.session) {
      impl.session->warn(fmt(
          "CoreMLVisionEncoder: output element count {} != expected {} "
          "({} tokens * {} hidden); dropping image",
          got, n, impl.n_tokens, impl.hidden));
    }
    return std::nullopt;
  }

  if (!impl.logged_output) {
    impl.logged_output = true;
    if (impl.session) {
      impl.session->info(fmt(
          "CoreMLVisionEncoder: output '{}' [{}, {}]",
          impl.output_name, impl.n_tokens, impl.hidden));
    }
  }

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
