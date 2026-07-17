// CoreMLModelManager tests.
//
// The unconditional tests verify cache-miss behaviour and the
// session->coreml_model_manager() wiring. End-to-end load /
// reference-count / serialisation tests need a real model and are
// gated on:
//
//     VPIPE_TEST_COREML_MODEL=/path/to/tiny.mlmodelc
//     VPIPE_TEST_COREML_INPUT=name_of_input_feature
//     VPIPE_TEST_COREML_OUTPUT=name_of_output_feature
//
// When the model env var is unset, those tests pass trivially.

#include "minitest.h"

#include "apple-silicon/coreml/coreml-model-manager.h"
#ifdef VPIPE_BUILD_APPLE_SILICON
// The host vision-encoder API (encode_host / encode_pair_host) is built
// in both the MLX and no-MLX configs, so include it whenever the Apple
// backend is on. The MLX runtime header is MLX-only.
#include "generative-models/shared/coreml-vision-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#endif
#include "apple-silicon/tensor-beat.h"
#include "common/session.h"
#include "common/vpipe-format.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace std;

namespace {

const char* env_or_null_(const char* name) {
  const char* v = std::getenv(name);
  return (v && *v) ? v : nullptr;
}

}

TEST(coreml_model_manager, exposed_via_session_context_intf) {
  vpipe::Session sess;
  // The session lazily constructs the manager on first call; on
  // Apple builds with VPIPE_BUILD_APPLE_SILICON the result is a
  // valid pointer.
#ifdef VPIPE_BUILD_APPLE_SILICON
  EXPECT_TRUE(sess.coreml_model_manager() != nullptr);
  // Repeated calls return the same instance.
  EXPECT_TRUE(sess.coreml_model_manager() ==
              sess.coreml_model_manager());
  EXPECT_TRUE(sess.coreml_model_manager()->cached_count() == 0u);
#else
  EXPECT_TRUE(sess.coreml_model_manager() == nullptr);
#endif
}

TEST(coreml_model_manager, load_missing_path_returns_null) {
  vpipe::Session sess;
  auto* mgr = sess.coreml_model_manager();
#ifdef VPIPE_BUILD_APPLE_SILICON
  ASSERT_TRUE(mgr != nullptr);
  auto sp = mgr->load("/dev/null/does-not-exist.mlmodelc",
                      /* compute_units */ 2);
  EXPECT_TRUE(sp == nullptr);
  // Cache stays empty -- failed loads don't populate.
  EXPECT_TRUE(mgr->cached_count() == 0u);
#else
  (void)mgr;
#endif
}

#ifdef VPIPE_BUILD_APPLE_SILICON

TEST(coreml_model_manager, shares_load_for_same_key) {
  const char* model = env_or_null_("VPIPE_TEST_COREML_MODEL");
  if (!model) {
    return;
  }
  vpipe::Session sess;
  auto* mgr = sess.coreml_model_manager();
  ASSERT_TRUE(mgr != nullptr);

  auto a = mgr->load(model, /* compute_units */ 2);
  ASSERT_TRUE(a != nullptr);
  auto b = mgr->load(model, /* compute_units */ 2);
  ASSERT_TRUE(b != nullptr);

  // Same underlying CoreMLLoadedModel.
  EXPECT_TRUE(a.get() == b.get());
  EXPECT_TRUE(mgr->cached_count() == 1u);
}

TEST(coreml_model_manager, different_compute_units_load_separately) {
  const char* model = env_or_null_("VPIPE_TEST_COREML_MODEL");
  if (!model) {
    return;
  }
  vpipe::Session sess;
  auto* mgr = sess.coreml_model_manager();
  ASSERT_TRUE(mgr != nullptr);

  auto a = mgr->load(model, /* compute_units */ 0);  // CPUOnly
  auto b = mgr->load(model, /* compute_units */ 2);  // All
  ASSERT_TRUE(a != nullptr);
  ASSERT_TRUE(b != nullptr);
  EXPECT_TRUE(a.get() != b.get());
  EXPECT_TRUE(mgr->cached_count() == 2u);
}

TEST(coreml_model_manager, weak_ptr_releases_when_last_consumer_drops) {
  const char* model = env_or_null_("VPIPE_TEST_COREML_MODEL");
  if (!model) {
    return;
  }
  vpipe::Session sess;
  auto* mgr = sess.coreml_model_manager();
  ASSERT_TRUE(mgr != nullptr);

  {
    auto a = mgr->load(model, 2);
    auto b = mgr->load(model, 2);
    EXPECT_TRUE(mgr->cached_count() == 1u);
    (void)a; (void)b;
  }
  // Both shared_ptrs gone; weak_ptr in cache is expired.
  EXPECT_TRUE(mgr->cached_count() == 0u);

  // A fresh load reloads the model.
  auto c = mgr->load(model, 2);
  ASSERT_TRUE(c != nullptr);
  EXPECT_TRUE(mgr->cached_count() == 1u);
}

TEST(coreml_model_manager, serializes_concurrent_predicts_on_same_model) {
  // predict() serializes per model internally (the per-model mutex is no
  // longer exposed). Drive predict() concurrently from two threads with a
  // zero-filled input derived from the model's sole fixed input; assert
  // every call succeeds. A data race in the shared model / prediction
  // options would crash or corrupt rather than return cleanly.
  const char* model = env_or_null_("VPIPE_TEST_COREML_MODEL");
  if (!model) {
    return;
  }
  vpipe::Session sess;
  auto* mgr = sess.coreml_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  auto loaded = mgr->load(model, 2);
  ASSERT_TRUE(loaded != nullptr);

  // Need exactly one input and at least one output to build a call.
  const auto& in_descs  = loaded->input_descs();
  const auto& out_descs = loaded->output_descs();
  if (in_descs.size() != 1 || out_descs.empty()) {
    return;
  }
  const std::string    in_name  = in_descs.begin()->first;
  const vpipe::CoreMLInputDesc in_desc = in_descs.begin()->second;
  const std::string    out_name = out_descs.begin()->first;

  // Zero-filled input bytes (shared read-only across both threads).
  std::vector<std::uint8_t> in_bytes;
  const bool is_image = in_desc.kind == vpipe::CoreMLFeatureKind::Image;
  if (is_image) {
    if (in_desc.image_width <= 0 || in_desc.image_height <= 0) { return; }
    in_bytes.assign(static_cast<std::size_t>(in_desc.image_width)
                        * in_desc.image_height * 4u, 0);
  } else {
    if (!in_desc.fixed) { return; }
    std::size_t n = 1;
    for (auto d : in_desc.shape) { n *= static_cast<std::size_t>(d); }
    in_bytes.assign(n * vpipe::coreml_dtype_size(in_desc.dtype), 0);
  }

  atomic<bool> all_ok{true};
  auto worker = [&]() {
    for (int i = 0; i < 20; ++i) {
      vpipe::CoreMLPredictInput cin;
      cin.name = in_name;
      if (is_image) {
        cin.image        = in_bytes.data();
        cin.image_width  = in_desc.image_width;
        cin.image_height = in_desc.image_height;
      } else {
        cin.data  = in_bytes.data();
        cin.dtype = in_desc.dtype;
        cin.shape = in_desc.shape;
      }
      vpipe::CoreMLPredictOutput cout;
      cout.name = out_name;
      cout.want = vpipe::CoreMLDType::F32;
      const vpipe::CoreMLPredictInput cins[1]  = { std::move(cin) };
      vpipe::CoreMLPredictOutput      couts[1] = { std::move(cout) };
      if (!loaded->predict(cins, couts)) { all_ok.store(false); }
    }
  };

  thread t1(worker);
  thread t2(worker);
  t1.join();
  t2.join();

  EXPECT_TRUE(all_ok.load());
}

// Two-input temporal-pair CoreML vision tower (image0+image1). Env-
// gated on VPIPE_TEST_COREML_VISION_VIDEO pointing at the
// ..._video_w8.mlpackage. Confirms the loader detects the pair layout
// and that a 2-frame prediction returns a valid token grid.
// (CoreMLVisionEncoder returns EncodedImage with mlx::core::array, so
// this test is MLX-only.)

// Host (MLX-free) temporal-pair path -- the one the no-MLX metal LM
// stack (realtime-vqa / visual-qa) actually drives. Same env gate as the
// MLX test above. Confirms encode_pair_host merges two DISTINCT frames
// into one token grid and that the fused result differs from a single
// replicated frame (so the second input genuinely participates).
TEST(coreml_vision_encoder, temporal_pair_video_model_host) {
  const char* model = env_or_null_("VPIPE_TEST_COREML_VISION_VIDEO");
  if (!model) { return; }
  vpipe::Session sess;

  vpipe::genai::CoreMLVisionEncoder::LoadSpec spec;
  spec.mlpackage_path = model;
  spec.compute_units  = 2;
  // runtime is MLX-only; the host path passes nullptr.
  auto enc = vpipe::genai::CoreMLVisionEncoder::create(spec, nullptr, &sess);
  EXPECT_TRUE(enc != nullptr);
  if (!enc) { return; }
  EXPECT_TRUE(enc->implemented());
  EXPECT_TRUE(enc->supports_temporal_pair());

  const int W = enc->model_input_width();
  const int H = enc->model_input_height();
  EXPECT_TRUE(W > 0 && H > 0);
  if (W <= 0 || H <= 0) { return; }

  std::vector<std::uint8_t> fA(static_cast<size_t>(3) * H * W);
  std::vector<std::uint8_t> fB(static_cast<size_t>(3) * H * W);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const size_t i = (static_cast<size_t>(c) * H + y) * W + x;
        fA[i] = static_cast<std::uint8_t>((x + y + 32 * c) & 0xFF);
        fB[i] = static_cast<std::uint8_t>((2 * x - y + 64 * c) & 0xFF);
      }
    }
  }

  auto pair   = enc->encode_pair_host(fA.data(), fB.data(), H, W);
  auto single = enc->encode_host(fA.data(), H, W);
  EXPECT_TRUE(pair.n_tokens == enc->output_n_tokens());
  EXPECT_TRUE(pair.n_tokens > 0);
  EXPECT_TRUE(pair.grid_h * pair.grid_w == pair.n_tokens);
  EXPECT_FALSE(pair.embeddings.empty());
  EXPECT_TRUE(single.n_tokens == pair.n_tokens);

  // A real 2-frame fusion must NOT byte-match the single-frame encode of
  // frame A alone -- otherwise image1 was ignored.
  bool differs =
      pair.embeddings.byte_size() != single.embeddings.byte_size();
  if (!differs && !pair.embeddings.empty() && !single.embeddings.empty()) {
    differs = std::memcmp(pair.embeddings.contents(),
                          single.embeddings.contents(),
                          pair.embeddings.byte_size()) != 0;
  }
  EXPECT_TRUE(differs);

  sess.info(vpipe::fmt(
      "coreml_vision_encoder.temporal_pair_host: in={}x{} out_tokens={} "
      "grid {}x{} differs={}",
      W, H, pair.n_tokens, pair.grid_h, pair.grid_w, differs));
}

// Zero-copy f16 RGB MLMultiArray input path. Gated on
// VPIPE_TEST_COREML_MARRAY_MODEL -> a CoreML vision model whose image input
// is a Float16 MLMultiArray ([1,3,H,W] NCHW or [1,H,W,3] NHWC). The test
// fixture MarrayMeanRGB_f16io outputs the per-channel MEAN as its tokens, so
// feeding a SOLID colour at the model's exact dims (no letterbox pad) proves
// the resampled tensor reaches the model with the right RGB channel order +
// values through the zero-copy MLMultiArray bind (an R<->B swap would flip
// the returned channels).
TEST(coreml_vision_encoder, marray_zero_copy_rgb_f16) {
  const char* model = env_or_null_("VPIPE_TEST_COREML_MARRAY_MODEL");
  if (!model) { return; }
  vpipe::Session sess;
  if (sess.metal_compute() == nullptr || !sess.metal_compute()->valid()) {
    return;
  }
  vpipe::genai::CoreMLVisionEncoder::LoadSpec spec;
  spec.mlpackage_path = model;
  spec.compute_units  = 2;
  auto enc = vpipe::genai::CoreMLVisionEncoder::create(spec, nullptr, &sess);
  ASSERT_TRUE(enc != nullptr);
  ASSERT_TRUE(enc->implemented());
  // Must take the MLMultiArray (zero-copy) path, not the BGRA pixel buffer.
  EXPECT_TRUE(enc->input_is_multiarray());

  const int W = enc->model_input_width();
  const int H = enc->model_input_height();
  ASSERT_TRUE(W > 0 && H > 0);

  // Solid colour at the model's exact dims (no pad): R=200, G=100, B=50
  // planar [3,H,W]. The mean-RGB fixture returns these as its tokens.
  const int R = 200, G = 100, B = 50;
  std::vector<std::uint8_t> frame(static_cast<size_t>(3) * H * W);
  const size_t plane = static_cast<size_t>(H) * W;
  for (size_t i = 0; i < plane; ++i) {
    frame[0 * plane + i] = static_cast<std::uint8_t>(R);
    frame[1 * plane + i] = static_cast<std::uint8_t>(G);
    frame[2 * plane + i] = static_cast<std::uint8_t>(B);
  }
  auto r = enc->encode_host(frame.data(), H, W);
  ASSERT_TRUE(r.n_tokens > 0);
  ASSERT_TRUE(!r.embeddings.empty());
  ASSERT_TRUE(r.out_hidden >= 3);
  const _Float16* e =
      static_cast<const _Float16*>(r.embeddings.contents());
  const float got_r = static_cast<float>(e[0]);
  const float got_g = static_cast<float>(e[1]);
  const float got_b = static_cast<float>(e[2]);
  std::printf("[marray_rgb_f16] in=%dx%d tokens=%d hidden=%d -> "
              "R=%.1f G=%.1f B=%.1f\n",
              W, H, r.n_tokens, r.out_hidden, got_r, got_g, got_b);
  // RGB order + values must survive the zero-copy f16 bind (f16 is exact for
  // these small integers; allow a tiny slack).
  EXPECT_TRUE(std::fabs(got_r - static_cast<float>(R)) < 1.0f);
  EXPECT_TRUE(std::fabs(got_g - static_cast<float>(G)) < 1.0f);
  EXPECT_TRUE(std::fabs(got_b - static_cast<float>(B)) < 1.0f);
}

#endif  // VPIPE_BUILD_APPLE_SILICON
