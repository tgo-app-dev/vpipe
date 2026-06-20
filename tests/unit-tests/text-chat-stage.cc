// text-chat-stage.cc -- tests for the interactive text-chat stage. The
// config/registration cases are MLX-free and run in BOTH the MLX and
// no-MLX builds (the stage builds on the VPIPE_BUILD_APPLE_SILICON axis;
// generation runs on the metal-compute backend when MLX is off).
//
// The end-to-end runtime smoke is env-gated on VPIPE_METAL_LM_SMOKE_MODEL
// (a metal-supported text decoder, same var as metal_lm_smoke.text_decode)
// and compiled only in the no-MLX build: it drives one user turn through a
// real mini-pipeline and asserts a non-empty assistant reply beat.

#include "minitest.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "common/vertex.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/text-chat-stage.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <streambuf>
#include <string>
#include <string_view>
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
                         FlexData::make_string("/tmp/chat-fake-model"));
  return cfg;
}

}  // namespace

// ---- Config / construction (no model; both builds) -----------------

TEST(text_chat_stage, type_is_registered) {
  EXPECT_TRUE(string_view(TextChatStage::kTypeName) == "text-chat");
}

TEST(text_chat_stage, config_defaults) {
  Session sess;
  CerrSilencer hush;
  TextChatStage s(&sess, "chat", vector<InEdge>{}, basic_cfg_());
  EXPECT_TRUE(s.hf_dir() == "/tmp/chat-fake-model");
  EXPECT_TRUE(s.max_new_tokens() == 1024);
  EXPECT_TRUE(s.num_oports() == 1);
}

TEST(text_chat_stage, missing_hf_dir_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(R"({"max_new_tokens":32})");
  TextChatStage s(&sess, "chat", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

TEST(text_chat_stage, bad_max_new_tokens_deferred) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/p","max_new_tokens":0})");
  TextChatStage s(&sess, "chat", vector<InEdge>{}, std::move(cfg));
  EXPECT_FALSE(s.config_error().empty());
}

// ---- End-to-end runtime smoke (env-gated; real text model) ---------
//
// Targets the metal/no-MLX generation path, which only exists in the
// no-MLX build (the MLX build drives the stage through the MLX path).
#if defined(VPIPE_BUILD_APPLE_SILICON)

namespace {

// Emits one FlexData user-message string, then signals done.
class OnePromptSource : public TypedStage<OnePromptSource> {
public:
  static constexpr const char* kTypeName = "ut-chat-prompt-source";
  using TypedStage::TypedStage;

  std::string prompt = "What is the capital of France? Answer in one word.";

  Job process(RuntimeContext& ctx) override
  {
    if (_sent) { ctx.signal_done(); co_return; }
    _sent = true;
    co_await ctx.write(0,
        make_payload<FlexDataPayload>(FlexData::make_string(prompt)));
  }

private:
  bool _sent = false;
};

// Collects every per-turn FlexData beat the stage emits on oport0.
class TurnSink : public TypedStage<TurnSink> {
public:
  static constexpr const char* kTypeName = "ut-chat-turn-sink";
  using TypedStage::TypedStage;

  std::vector<FlexData> take()
  {
    lock_guard<mutex> g(_mu);
    return _collected;
  }

  Job process(RuntimeContext& ctx) override
  {
    auto in = co_await ctx.read(0);
    if (!in) { ctx.signal_done(); co_return; }
    const auto* p = dynamic_cast<const FlexDataPayload*>(in.get());
    if (p) {
      lock_guard<mutex> g(_mu);
      _collected.push_back(p->data);
    }
  }

private:
  mutex                 _mu;
  std::vector<FlexData> _collected;
};

}  // namespace

// The flattened `mtp` flag wires into _mtp_enabled (metal path only).
// Default on; explicit false disables the MTP speculative-decode fast path.
// MTP is token-exact, so this is a perf-only switch. Pure config parse --
// no model.
TEST(text_chat_stage, mtp_flag_parsed) {
  Session sess;
  CerrSilencer hush;
  {
    TextChatStage s(&sess, "chat", vector<InEdge>{}, basic_cfg_());
    EXPECT_TRUE(s.mtp_enabled());   // kSpec default = true
  }
  {
    FlexData cfg = FlexData::from_json(R"({"hf_dir":"/p","mtp":false})");
    TextChatStage s(&sess, "chat", vector<InEdge>{}, std::move(cfg));
    EXPECT_FALSE(s.mtp_enabled());
  }
}

TEST(text_chat_stage, metal_chat_smoke) {
  const char* path = std::getenv("VPIPE_METAL_LM_SMOKE_MODEL");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  Session  sess;
  Pipeline pl("chat-smoke", &sess);

  auto src = make_unique<OnePromptSource>(
      &sess, "prompt", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  auto* promptsrc = static_cast<OnePromptSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(24));
    o.insert("max_pages", FlexData::make_int(8));
  }
  auto ch = make_unique<TextChatStage>(
      &sess, "chat", vector<InEdge>{ { promptsrc, 0 } }, std::move(cfg));
  auto* chat = static_cast<TextChatStage*>(
      pl.insert_stage(std::move(ch)));

  auto sk = make_unique<TurnSink>(
      &sess, "sink", vector<InEdge>{ { chat, 0 } },
      FlexData::make_object());
  auto* sink = static_cast<TurnSink*>(pl.insert_stage(std::move(sk)));

  PipelineRuntime rt(&pl, &sess);
  const bool launched = rt.launch();
  ::unsetenv("VPIPE_LLM_BACKEND");
  ASSERT_TRUE(launched);
  rt.wait_idle();
  rt.stop();

  auto turns = sink->take();
  ASSERT_TRUE(!turns.empty());
  bool found = false;
  for (const auto& fd : turns) {
    if (!fd.is_object()) { continue; }
    auto root = fd.as_object();
    const std::string text = root.contains("text")
        ? std::string(root.at("text").as_string("")) : std::string();
    const std::int64_t ctx_pos = root.contains("ctx_pos")
        ? root.at("ctx_pos").as_int(0) : 0;
    // ctx_pos is the MLX-free seq_len() counter; the metal backend tracks
    // K/V per context id internally and reports 0 here, so the generation
    // signal is the non-empty reply text, not ctx_pos.
    std::printf("[text_chat_stage.metal_smoke] ctx_pos=%lld text='%s'\n",
                static_cast<long long>(ctx_pos), text.c_str());
    if (!text.empty()) { found = true; }
  }
  EXPECT_TRUE(found);
}

#endif  // VPIPE_BUILD_APPLE_SILICON
