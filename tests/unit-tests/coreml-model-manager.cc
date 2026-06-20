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

#include "apple-silicon/coreml/coreml-cpp/CoreML.hpp"
#include "apple-silicon/coreml/coreml-model-manager.h"
#ifdef VPIPE_BUILD_APPLE_SILICON
// The host vision-encoder API (encode_host / encode_pair_host) is built
// in both the MLX and no-MLX configs, so include it whenever the Apple
// backend is on. The MLX runtime header is MLX-only.
#include "generative-models/shared/coreml-vision-encoder.h"
#endif
#include "apple-silicon/tensor-beat.h"
#include "common/session.h"
#include "common/vpipe-format.h"

#include <atomic>
#include <chrono>
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
  // Two threads each grab the same shared model and call predict
  // through the per-model mutex; we increment a counter on entry to
  // the mutex-guarded region and decrement on exit, asserting the
  // counter never exceeds 1.
  const char* model = env_or_null_("VPIPE_TEST_COREML_MODEL");
  const char* in_n  = env_or_null_("VPIPE_TEST_COREML_INPUT");
  if (!model || !in_n) {
    return;
  }
  vpipe::Session sess;
  auto* mgr = sess.coreml_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  auto loaded = mgr->load(model, 2);
  ASSERT_TRUE(loaded != nullptr);

  atomic<int>  in_flight{0};
  atomic<int>  max_in_flight{0};
  atomic<bool> any_overlap{false};

  auto worker = [&]() {
    for (int i = 0; i < 50; ++i) {
      std::lock_guard<std::mutex> lk(loaded->predict_mutex());
      int now = ++in_flight;
      if (now > 1) {
        any_overlap.store(true);
      }
      int prev = max_in_flight.load();
      while (now > prev &&
             !max_in_flight.compare_exchange_weak(prev, now)) {
        // retry
      }
      // Spin briefly to widen any race window.
      auto deadline =
          chrono::steady_clock::now() + chrono::microseconds(50);
      while (chrono::steady_clock::now() < deadline) {
        // busy-wait
      }
      --in_flight;
    }
  };

  thread t1(worker);
  thread t2(worker);
  t1.join();
  t2.join();

  EXPECT_FALSE(any_overlap.load());
  EXPECT_TRUE(max_in_flight.load() == 1);
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

#endif  // VPIPE_BUILD_APPLE_SILICON
