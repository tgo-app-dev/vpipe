#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/visual-qa-stage.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

class CerrSilencer {
public:
  CerrSilencer() : _saved(cerr.rdbuf()), _null() { cerr.rdbuf(&_null); }
  ~CerrSilencer() { cerr.rdbuf(_saved); }
private:
  struct NullBuf : public streambuf {
    int overflow(int c) override { return c; }
  };
  streambuf* _saved;
  NullBuf    _null;
};

FlexData
basic_cfg_()
{
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir",
                         FlexData::make_string("/tmp/vqa-fake-model"));
  cfg.as_object().insert("questions",
                         FlexData::make_string("What is this?"));
  return cfg;
}

}

TEST(visual_qa_stage, config_defaults) {
  Session sess;
  CerrSilencer hush;
  VisualQaStage s(&sess, "vqa", vector<InEdge>{}, basic_cfg_());
  EXPECT_TRUE(s.hf_dir() == "/tmp/vqa-fake-model");
  EXPECT_TRUE(s.num_images() == 1);
  EXPECT_TRUE(s.max_new_tokens() == 1024);
  EXPECT_TRUE(s.questions().size() == 1);
  EXPECT_TRUE(s.questions()[0] == "What is this?");
  EXPECT_TRUE(s.num_oports() == 0);
}

TEST(visual_qa_stage, config_questions_array) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p","questions":["q1","q2","q3"]})");
  VisualQaStage s(&sess, "vqa", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.questions().size() == 3);
  EXPECT_TRUE(s.questions()[0] == "q1");
  EXPECT_TRUE(s.questions()[1] == "q2");
  EXPECT_TRUE(s.questions()[2] == "q3");
}

TEST(visual_qa_stage, config_num_images) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p","questions":"q","num_images":4})");
  VisualQaStage s(&sess, "vqa", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.num_images() == 4);
}

// Construction must succeed for any config (so a graph can be built/
// edited before required fields are supplied); the missing/empty/bad
// field is recorded in config_error() and deferred to launch.
TEST(visual_qa_stage, missing_hf_dir_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(R"({"questions":"q"})");
  VisualQaStage s(&sess, "vqa", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(visual_qa_stage, missing_questions_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(R"({"hf_dir":"/p"})");
  VisualQaStage s(&sess, "vqa", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(visual_qa_stage, empty_questions_array_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p","questions":[]})");
  VisualQaStage s(&sess, "vqa", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(visual_qa_stage, bad_num_images_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p","questions":"q","num_images":0})");
  VisualQaStage s(&sess, "vqa", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(visual_qa_stage, type_is_registered) {
  EXPECT_TRUE(string_view(VisualQaStage::kTypeName) == "visual-qa");
}

// ---- End-to-end runtime smoke (env-gated; real VLM) ----------------
//
// Targets the image-VQA path end-to-end on a real VLM. visual-qa is a
// sink (answers go to session()->info), so the assertion is that the
// multimodal round runs to completion without crashing; answer quality is
// covered by the token-exact encoder/prefill tests.
#if defined(VPIPE_BUILD_APPLE_SILICON)

#include "apple-silicon/metal-compute/image-ops.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-storage.h"

namespace {

// Emits one synthetic u8 [3,H,W] RGB image, then signals done. The frame
// is allocated as a Shared/UMA metal-compute buffer (like video-to-rgb's
// output) so the stage exercises the zero-copy input path (Shared
// TensorBeat -> encoder buffer bind); falls back to CpuCached when
// metal-compute is unavailable.
class OneImageSource : public TypedStage<OneImageSource> {
public:
  static constexpr const char* kTypeName = "ut-vqa-one-image-source";
  using TypedStage::TypedStage;

  int H = 128, W = 128;

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    const size_t nbytes = static_cast<size_t>(3) * H * W;
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::U8;
    tb.shape = { 3, H, W };
    uint8_t* p = nullptr;
    auto* mc = session()->metal_compute();
    if (mc && mc->valid()) {
      auto h = metal_compute::make_shared_storage(*mc, nbytes, session());
      if (h) { p = h->contents; tb.external = std::move(h); }
    }
    if (!tb.external) {
      tb.resize_contiguous(nbytes);
      p = tb.as_u8();
    }
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          p[(static_cast<size_t>(c) * H + y) * W + x] =
              static_cast<uint8_t>((x + 2 * y + 40 * c) & 0xFF);
        }
      }
    }
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
  }

private:
  bool _sent = false;
};

// Emits N synthetic u8 [3,H,W] RGB frames (varying per frame), then signals
// done. Drives the stage's video mode (with config.video.enabled +
// num_images == N): per-frame encode at the video budget, video-prompt
// render, multimodal prefill, decode.
class FourFrameVideoSource : public TypedStage<FourFrameVideoSource> {
public:
  static constexpr const char* kTypeName = "ut-vqa-four-frame-video-source";
  using TypedStage::TypedStage;

  int H = 128, W = 128, N = 4;

  Job process(RuntimeContext& ctx) override
  {
    if (_sent >= N) { ctx.signal_done(); co_return; }
    const int idx = _sent++;
    const size_t nbytes = static_cast<size_t>(3) * H * W;
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::U8;
    tb.shape = { 3, H, W };
    uint8_t* p = nullptr;
    auto* mc = session()->metal_compute();
    if (mc && mc->valid()) {
      auto h = metal_compute::make_shared_storage(*mc, nbytes, session());
      if (h) { p = h->contents; tb.external = std::move(h); }
    }
    if (!tb.external) {
      tb.resize_contiguous(nbytes);
      p = tb.as_u8();
    }
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          p[(static_cast<size_t>(c) * H + y) * W + x] =
              static_cast<uint8_t>((x + 2 * y + 40 * c + 17 * idx) & 0xFF);
        }
      }
    }
    co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
  }

private:
  int _sent = 0;
};

}  // namespace

TEST(visual_qa_stage, metal_image_smoke) {
  const char* path = std::getenv("VPIPE_METAL_VQA_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  Session  sess;
  Pipeline pl("vqa-smoke", &sess);

  auto src = make_unique<OneImageSource>(
      &sess, "image", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  auto* imgsrc = static_cast<OneImageSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(24));
    o.insert("max_pages", FlexData::make_int(16));
    o.insert("num_images", FlexData::make_int(1));
    o.insert("questions",
             FlexData::make_string("What color dominates this image?"));
  }
  auto vqa = make_unique<VisualQaStage>(
      &sess, "vqa", vector<InEdge>{ { imgsrc, 0 } }, std::move(cfg));
  auto* vqa_stage = static_cast<VisualQaStage*>(
      pl.insert_stage(std::move(vqa)));
  (void)vqa_stage;

  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(launched);
  rt.wait_idle();
  rt.stop();
  // Reaching here means the metal image-VQA round (vision encode +
  // mROPE multimodal prefill + per-question decode) ran end-to-end
  // without crashing. The answer is surfaced via session()->info.
  EXPECT_TRUE(true);
}

// CoreML vision tower on the no-MLX metal path: same as above but the
// vision forward runs through a pre-converted CoreML mlpackage
// (metal-compute letterbox preproc + host-f32 splice). Env:
// VPIPE_METAL_VQA_SMOKE_MODEL (the VLM dir) +
// VPIPE_METAL_VQA_COREML_PATH (the .mlpackage / .mlmodelc).
TEST(visual_qa_stage, metal_coreml_image_smoke) {
  const char* path  = std::getenv("VPIPE_METAL_VQA_SMOKE_MODEL");
  const char* cmlp  = std::getenv("VPIPE_METAL_VQA_COREML_PATH");
  if (!path || !*path || !cmlp || !*cmlp) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  Session  sess;
  Pipeline pl("vqa-coreml-smoke", &sess);

  auto src = make_unique<OneImageSource>(
      &sess, "image", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  auto* imgsrc = static_cast<OneImageSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("coreml_vision_path", FlexData::make_string(cmlp));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(24));
    o.insert("max_pages", FlexData::make_int(16));
    o.insert("num_images", FlexData::make_int(1));
    o.insert("questions",
             FlexData::make_string("What color dominates this image?"));
  }
  auto vqa = make_unique<VisualQaStage>(
      &sess, "vqa", vector<InEdge>{ { imgsrc, 0 } }, std::move(cfg));
  (void)pl.insert_stage(std::move(vqa));

  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(launched);
  rt.wait_idle();
  rt.stop();
  // Reaching here means the CoreML vision tower (metal-compute letterbox
  // -> CoreML predict -> host-f32 splice) drove a full metal image-VQA
  // round without crashing.
  EXPECT_TRUE(true);
}

// Metal-backend VIDEO smoke: 4 frames through the stage's video mode (the
// no-MLX metal path treats them as an image sequence; Gemma-4 frames use
// the video soft-token budget). Verifies the video plumbing runs end-to-
// end without crashing.
TEST(visual_qa_stage, metal_video_smoke) {
  const char* path = std::getenv("VPIPE_METAL_VQA_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  Session  sess;
  Pipeline pl("vqa-metal-video-smoke", &sess);

  auto src = make_unique<FourFrameVideoSource>(
      &sess, "video", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  auto* vidsrc = static_cast<FourFrameVideoSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(16));
    o.insert("max_pages", FlexData::make_int(16));
    o.insert("num_images", FlexData::make_int(4));
    o.insert("questions",
             FlexData::make_string("What is in this video?"));
    FlexData vid = FlexData::make_object();
    vid.as_object().insert("enabled", FlexData::make_bool(true));
    vid.as_object().insert("fps", FlexData::make_real(2.0));
    o.insert("video", std::move(vid));
  }
  auto vqa = make_unique<VisualQaStage>(
      &sess, "vqa", vector<InEdge>{ { vidsrc, 0 } }, std::move(cfg));
  (void)pl.insert_stage(std::move(vqa));

  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(launched);
  rt.wait_idle();
  rt.stop();
  EXPECT_TRUE(true);
}


#endif  // VPIPE_BUILD_APPLE_SILICON
