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
#include "common/vpipe-format.h"
#include "interfaces/ui-delegate-intf.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/text-chat-stage.h"

#include "common/media-line.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <streambuf>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <unistd.h>   // ::access (seatbelt availability check)

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
  // MCP tools are opt-in: off unless enable_tools is set.
  EXPECT_FALSE(s.tools_enabled());
}

TEST(text_chat_stage, enable_tools_config) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/tmp/chat-fake-model","enable_tools":true})");
  TextChatStage s(&sess, "chat", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.tools_enabled());
  EXPECT_FALSE(s.python_tool_enabled());
}

TEST(text_chat_stage, enable_python_tool_config) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/tmp/chat-fake-model","enable_python_tool":true})");
  TextChatStage s(&sess, "chat", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.python_tool_enabled());
  // Independent of enable_tools (which stays default-off here).
  EXPECT_FALSE(s.tools_enabled());
}

TEST(text_chat_stage, enable_file_tools_config) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/tmp/chat-fake-model","enable_file_tools":true})");
  TextChatStage s(&sess, "chat", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.file_tools_enabled());
  EXPECT_FALSE(s.tools_enabled());
  EXPECT_FALSE(s.python_tool_enabled());
}

TEST(text_chat_stage, enable_web_tools_config) {
  Session sess;
  CerrSilencer hush;
  FlexData cfg = FlexData::from_json(
      R"({"hf_dir":"/tmp/chat-fake-model","enable_web_tools":true})");
  TextChatStage s(&sess, "chat", vector<InEdge>{}, std::move(cfg));
  EXPECT_TRUE(s.config_error().empty());
  EXPECT_TRUE(s.web_tools_enabled());
  EXPECT_FALSE(s.tools_enabled());
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
    // ctx_pos comes from the owns_kv exec's context_seq_len (the metal
    // backends track K/V internally); after a full turn it must be a
    // real position: prompt + generated tokens + assistant close.
    std::printf("[text_chat_stage.metal_smoke] ctx_pos=%lld text='%s'\n",
                static_cast<long long>(ctx_pos), text.c_str());
    if (!text.empty()) { found = true; }
    EXPECT_TRUE(ctx_pos > 0);
  }
  EXPECT_TRUE(found);
}

// ---- MCP tool calling (env-gated; a Qwen/ChatML checkpoint) ---------
//
// Drives one "what time is it?" turn with enable_tools=true against a
// real ChatML model and asserts the tool loop actually fired: the stage
// logs a `tool call 'get_current_time'` line when it dispatches the
// model-emitted <tool_call>, and the final answer beat is non-empty.

namespace {

// UI delegate that records every info/warn/error message into shared
// storage (Session routes info/warn/error through the UI delegate) so a
// test can assert a specific line was emitted. Streams are no-ops.
class CapturingUiDelegate : public UiDelegateIntf {
public:
  CapturingUiDelegate(std::shared_ptr<std::mutex>               mu,
                      std::shared_ptr<std::vector<std::string>> lines)
    : _mu(std::move(mu)), _lines(std::move(lines)) {}

  void error(const VpipeFormat& f) override { record_(f); }
  void warn (const VpipeFormat& f) override { record_(f); }
  void info (const VpipeFormat& f) override { record_(f); }

  UiInputStatus
  getline(const VpipeFormat&, std::string&,
          const std::function<bool()>&) override
  {
    return UiInputStatus::Eof;
  }

  std::unique_ptr<UiTextStream> open_text_stream() override
  {
    return std::make_unique<NullUiTextStream>();
  }

private:
  void record_(const VpipeFormat& f)
  {
    std::string s = f();
    lock_guard<mutex> g(*_mu);
    _lines->push_back(std::move(s));
  }
  std::shared_ptr<std::mutex>               _mu;
  std::shared_ptr<std::vector<std::string>> _lines;
};

}  // namespace

TEST(text_chat_stage, metal_chat_tool_call_time) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  auto mu    = std::make_shared<std::mutex>();
  auto lines = std::make_shared<std::vector<std::string>>();
  Session sess;
  sess.set_ui_delegate(std::make_unique<CapturingUiDelegate>(mu, lines));
  Pipeline pl("chat-tool-smoke", &sess);

  auto src = make_unique<OnePromptSource>(
      &sess, "prompt", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  src->prompt = "What is the current date and time? Use your tools.";
  auto* promptsrc = static_cast<OnePromptSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(256));
    o.insert("max_pages", FlexData::make_int(16));
    o.insert("enable_tools", FlexData::make_bool(true));
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

  // The final answer beat must be non-empty.
  auto turns = sink->take();
  ASSERT_TRUE(!turns.empty());
  std::string final_text;
  for (const auto& fd : turns) {
    if (!fd.is_object()) { continue; }
    auto root = fd.as_object();
    if (root.contains("text")) {
      final_text = std::string(root.at("text").as_string(""));
    }
  }
  std::printf("[text_chat_stage.tool_call] final='%s'\n",
              final_text.c_str());
  EXPECT_TRUE(!final_text.empty());

  // The stage logged a get_current_time dispatch => the tool loop fired.
  bool tool_fired = false;
  {
    lock_guard<mutex> g(*mu);
    for (const auto& l : *lines) {
      if (l.find("tool call 'get_current_time'") != string::npos) {
        tool_fired = true;
        break;
      }
    }
  }
  EXPECT_TRUE(tool_fired);
}

// Same end-to-end tool-call smoke, but against a Gemma-4 checkpoint --
// which uses the bespoke `<|tool>`/`<|tool_call>`/`<|tool_response>` DSL
// (not Hermes JSON). Exercises the full Gemma runtime path: tools system
// turn -> decode -> model emits `<|tool_call>call:get_current_time{}`
// and stops on `<|tool_response>` (the assistant-close commit is skipped
// so the turn stays open) -> parse -> dispatch -> render `response:...`
// -> continue decoding the final answer in the SAME turn. Gated on the
// e4b test model (VPIPE_GEMMA4_TEST_MODEL_PATH).
TEST(text_chat_stage, metal_chat_tool_call_time_gemma) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  auto mu    = std::make_shared<std::mutex>();
  auto lines = std::make_shared<std::vector<std::string>>();
  Session sess;
  sess.set_ui_delegate(std::make_unique<CapturingUiDelegate>(mu, lines));
  Pipeline pl("chat-tool-smoke-gemma", &sess);

  auto src = make_unique<OnePromptSource>(
      &sess, "prompt", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  src->prompt = "What is the current date and time? Use your tools.";
  auto* promptsrc = static_cast<OnePromptSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(256));
    o.insert("max_pages", FlexData::make_int(16));
    o.insert("enable_tools", FlexData::make_bool(true));
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

  // The stage must NOT have disabled tools at init (the bug this fixes):
  // no "no tool-calling support" warning may appear.
  {
    lock_guard<mutex> g(*mu);
    for (const auto& l : *lines) {
      EXPECT_TRUE(l.find("no tool-calling support") == string::npos);
    }
  }

  auto turns = sink->take();
  ASSERT_TRUE(!turns.empty());
  std::string final_text;
  for (const auto& fd : turns) {
    if (!fd.is_object()) { continue; }
    auto root = fd.as_object();
    if (root.contains("text")) {
      final_text = std::string(root.at("text").as_string(""));
    }
  }
  std::printf("[text_chat_stage.tool_call_gemma] final='%s'\n",
              final_text.c_str());
  EXPECT_TRUE(!final_text.empty());

  // The stage logged a get_current_time dispatch => the Gemma tool loop
  // parsed the `<|tool_call>` DSL and ran the tool.
  bool tool_fired = false;
  {
    lock_guard<mutex> g(*mu);
    for (const auto& l : *lines) {
      if (l.find("tool call 'get_current_time'") != string::npos) {
        tool_fired = true;
        break;
      }
    }
  }
  EXPECT_TRUE(tool_fired);
}

TEST(text_chat_stage, metal_chat_python_tool_compute) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) {
    return;
  }
  if (::access("/usr/bin/sandbox-exec", X_OK) != 0) {
    return;   // no seatbelt -> the run_python tool can't run here
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  auto mu    = std::make_shared<std::mutex>();
  auto lines = std::make_shared<std::vector<std::string>>();
  Session sess;
  sess.set_ui_delegate(std::make_unique<CapturingUiDelegate>(mu, lines));
  Pipeline pl("chat-python-smoke", &sess);

  auto src = make_unique<OnePromptSource>(
      &sess, "prompt", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  src->prompt = "Use the run_python tool to compute 12345 * 67890 and "
                "tell me the exact integer result.";
  auto* promptsrc = static_cast<OnePromptSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(512));
    o.insert("max_pages", FlexData::make_int(24));
    o.insert("enable_python_tool", FlexData::make_bool(true));
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

  std::string final_text;
  for (const auto& fd : sink->take()) {
    if (!fd.is_object()) { continue; }
    auto root = fd.as_object();
    if (root.contains("text")) {
      final_text = std::string(root.at("text").as_string(""));
    }
  }
  std::printf("[text_chat_stage.python_tool] final='%s'\n",
              final_text.c_str());

  bool tool_fired = false;
  {
    lock_guard<mutex> g(*mu);
    for (const auto& l : *lines) {
      if (l.find("tool call 'run_python'") != string::npos) {
        tool_fired = true;
        break;
      }
    }
  }
  EXPECT_TRUE(tool_fired);
  // 12345 * 67890 == 838102050 -- the sandboxed python computed it and
  // the model relayed it.
  EXPECT_TRUE(final_text.find("838102050") != string::npos);
}

TEST(text_chat_stage, metal_chat_file_tools_write_read) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) {
    return;
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  // Use an explicit workspace dir so the test can inspect the file the
  // model writes.
  namespace fs = std::filesystem;
  fs::path ws = fs::temp_directory_path() / "vpipe-ut-chat-ws";
  std::error_code ec;
  fs::remove_all(ws, ec);
  fs::create_directories(ws, ec);

  auto mu    = std::make_shared<std::mutex>();
  auto lines = std::make_shared<std::vector<std::string>>();
  Session sess;
  sess.set_ui_delegate(std::make_unique<CapturingUiDelegate>(mu, lines));
  Pipeline pl("chat-file-smoke", &sess);

  auto src = make_unique<OnePromptSource>(
      &sess, "prompt", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  src->prompt = "Use the write_file tool to save the exact text "
                "'vpipe-was-here' to a file named marker.txt, then use "
                "read_file to read it back and tell me its contents.";
  auto* promptsrc = static_cast<OnePromptSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(512));
    o.insert("max_pages", FlexData::make_int(24));
    o.insert("enable_file_tools", FlexData::make_bool(true));
    o.insert("file_sandbox_dir", FlexData::make_string(ws.string()));
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

  std::string final_text;
  for (const auto& fd : sink->take()) {
    if (!fd.is_object()) { continue; }
    auto root = fd.as_object();
    if (root.contains("text")) {
      final_text = std::string(root.at("text").as_string(""));
    }
  }
  std::printf("[text_chat_stage.file_tools] final='%s'\n",
              final_text.c_str());

  bool wrote = false, read = false;
  {
    lock_guard<mutex> g(*mu);
    for (const auto& l : *lines) {
      if (l.find("tool call 'write_file'") != string::npos) { wrote = true; }
      if (l.find("tool call 'read_file'")  != string::npos) { read = true; }
    }
  }
  EXPECT_TRUE(wrote);
  EXPECT_TRUE(read);
  // The file the model wrote must actually exist in the workspace with
  // the requested content.
  std::ifstream f(ws / "marker.txt");
  std::string body((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  EXPECT_TRUE(body.find("vpipe-was-here") != string::npos);
  fs::remove_all(ws, ec);
}

TEST(text_chat_stage, metal_chat_web_tool_scrape) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  const char* net  = std::getenv("VPIPE_MCP_NET_TEST");
  if (!path || !*path || !net || !*net) {
    return;   // needs a model AND explicit network opt-in
  }
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);

  auto mu    = std::make_shared<std::mutex>();
  auto lines = std::make_shared<std::vector<std::string>>();
  Session sess;
  sess.set_ui_delegate(std::make_unique<CapturingUiDelegate>(mu, lines));
  Pipeline pl("chat-web-smoke", &sess);

  auto src = make_unique<OnePromptSource>(
      &sess, "prompt", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  src->prompt = "Use the scrape_page tool on https://example.com/ and "
                "tell me the main heading or title of the page.";
  auto* promptsrc = static_cast<OnePromptSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(path));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(512));
    o.insert("max_pages", FlexData::make_int(24));
    o.insert("enable_web_tools", FlexData::make_bool(true));
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

  std::string final_text;
  for (const auto& fd : sink->take()) {
    if (!fd.is_object()) { continue; }
    auto root = fd.as_object();
    if (root.contains("text")) {
      final_text = std::string(root.at("text").as_string(""));
    }
  }
  std::printf("[text_chat_stage.web_tool] final='%s'\n",
              final_text.c_str());

  bool fired = false;
  {
    lock_guard<mutex> g(*mu);
    for (const auto& l : *lines) {
      if (l.find("tool call 'scrape_page'") != string::npos
          || l.find("tool call 'fetch_url'") != string::npos) {
        fired = true;
        break;
      }
    }
  }
  EXPECT_TRUE(fired);
  // example.com's page is titled/headed "Example Domain".
  EXPECT_TRUE(final_text.find("Example Domain") != string::npos);
}

// ---- Media-line turns (env-gated; the VL checkpoint) ----------------

namespace {

// Run one prompt through a fresh {source -> text-chat -> sink} mini-
// pipeline against `model_path` and return the collected turn texts.
std::vector<std::string>
run_chat_turn_(const std::string& model_path, const std::string& prompt)
{
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session  sess;
  Pipeline pl("chat-media-smoke", &sess);

  auto src = make_unique<OnePromptSource>(
      &sess, "prompt", vector<InEdge>{}, FlexData::make_object());
  src->allocate_oports(1);
  src->prompt = prompt;
  auto* promptsrc = static_cast<OnePromptSource*>(
      pl.insert_stage(std::move(src)));

  FlexData cfg = FlexData::make_object();
  {
    auto o = cfg.as_object();
    o.insert("hf_dir", FlexData::make_string(model_path));
    o.insert("compute_dtype", FlexData::make_string("f16"));
    o.insert("max_new_tokens", FlexData::make_int(96));
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
  if (!launched) {
    return {};
  }
  rt.wait_idle();
  rt.stop();

  std::vector<std::string> texts;
  for (const auto& fd : sink->take()) {
    if (!fd.is_object()) { continue; }
    auto root = fd.as_object();
    if (root.contains("text")) {
      texts.emplace_back(root.at("text").as_string(""));
    }
  }
  return texts;
}

std::string
lower_(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Write a solid-red 128x128 binary PPM under the temp dir.
std::string
write_red_ppm_()
{
  const auto path = std::filesystem::temp_directory_path()
      / "vpipe-ut-chat-red.ppm";
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f << "P6\n128 128\n255\n";
  for (int i = 0; i < 128 * 128; ++i) {
    f.put(static_cast<char>(0xff));
    f.put(static_cast<char>(0x00));
    f.put(static_cast<char>(0x00));
  }
  return path.string();
}

}  // namespace

// An image attached via a media-line fs marker reaches the model: the
// marker is parsed before the tokenizer, the file decodes through
// FFmpeg, the LM's own vision tower encodes it, and the embeddings
// splice into the prefill at the marker position. A solid-red image +
// "what color" must put "red" somewhere in the reply (the VL family
// default keeps thinking on; the reasoning text counts too).
TEST(text_chat_stage, metal_chat_media_image_turn) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) {
    return;
  }
  const std::string img = write_red_ppm_();
  const std::string prompt =
      "Look at this image: "
      + vpipe::media_line::make_fs_marker(
            vpipe::media_line::Modality::Image, img)
      + " What is the dominant color of the image? "
        "Answer in one word.";
  auto texts = run_chat_turn_(path, prompt);
  ASSERT_TRUE(!texts.empty());
  bool has_red = false;
  for (const auto& t : texts) {
    std::printf("[text_chat_stage.media_image] text='%s'\n", t.c_str());
    if (lower_(t).find("red") != std::string::npos) { has_red = true; }
  }
  EXPECT_TRUE(has_red);
}

// An attachment whose modality the model does not provide (audio on
// Qwen3.5-VL: no audio tower, no audio_pad token) is warned about and
// DROPPED, and the surrounding text still runs as a plain turn.
TEST(text_chat_stage, metal_chat_media_unsupported_drops) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) {
    return;
  }
  // The capability check rejects the modality before touching the
  // path, so a nonexistent file is fine here.
  const std::string prompt =
      "What is the capital of France? "
      + vpipe::media_line::make_fs_marker(
            vpipe::media_line::Modality::Audio, "/nonexistent/clip.wav")
      + " Answer in one word.";
  auto texts = run_chat_turn_(path, prompt);
  ASSERT_TRUE(!texts.empty());
  bool has_paris = false;
  for (const auto& t : texts) {
    std::printf("[text_chat_stage.media_drop] text='%s'\n", t.c_str());
    if (lower_(t).find("paris") != std::string::npos) {
      has_paris = true;
    }
  }
  EXPECT_TRUE(has_paris);
}

namespace {

// Write a 1 s, 16 kHz, s16 mono WAV carrying a 440 Hz sine.
std::string
write_sine_wav_()
{
  const auto path = std::filesystem::temp_directory_path()
      / "vpipe-ut-chat-sine.wav";
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  constexpr int      kRate    = 16000;
  constexpr int      kSamples = kRate;
  constexpr uint32_t kDataLen = kSamples * 2;
  auto u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) {
      f.put(static_cast<char>((v >> (8 * i)) & 0xff));
    }
  };
  auto u16 = [&](uint16_t v) {
    f.put(static_cast<char>(v & 0xff));
    f.put(static_cast<char>(v >> 8));
  };
  f << "RIFF"; u32(36 + kDataLen); f << "WAVE";
  f << "fmt "; u32(16); u16(1); u16(1); u32(kRate);
  u32(kRate * 2); u16(2); u16(16);
  f << "data"; u32(kDataLen);
  for (int i = 0; i < kSamples; ++i) {
    const double v = 0.5 * std::sin(2.0 * 3.14159265358979 * 440.0
                                    * i / kRate);
    u16(static_cast<uint16_t>(static_cast<int16_t>(v * 32767.0)));
  }
  return path.string();
}

}  // namespace

// The Qwen3.5-VL family default is thinking-ON: the template opens the
// reasoning block in the prompt, the stage emits the unified
// thinking-START marker into the stream, and the model's generated
// `</think>` is rewritten to the unified END marker by the
// detokenizer. The turn text must carry the START marker (and, when
// the model closed its reasoning within budget, the END marker) --
// never the family-specific `</think>` literal.
TEST(text_chat_stage, metal_chat_thinking_markers_unified) {
  const char* path = std::getenv("VPIPE_QWEN35_TEST_MODEL_PATH");
  if (!path || !*path) {
    return;
  }
  auto texts = run_chat_turn_(
      path, "What is the capital of France? Answer in one word.");
  ASSERT_TRUE(!texts.empty());
  bool has_start = false, has_literal = false;
  for (const auto& t : texts) {
    if (t.find(vpipe::media_line::kThinkStart) != std::string::npos) {
      has_start = true;
    }
    if (t.find("</think>") != std::string::npos) { has_literal = true; }
  }
  EXPECT_TRUE(has_start);
  EXPECT_TRUE(!has_literal);
}

// Gemma-4 e4b is the tree's image+audio chat family: one turn mixing
// BOTH modalities must render (boi/pad/eoi + boa/pad/eoa in-line) and
// splice through the gemma towers. The signal here is mechanical --
// the turn prefills with both blocks and decodes a non-empty reply
// mentioning the image's color; the audio content assert stays loose
// (a bare sine has no canonical name).
TEST(text_chat_stage, metal_chat_media_gemma_image_audio_turn) {
  const char* path = std::getenv("VPIPE_GEMMA4_TEST_MODEL_PATH");
  if (!path || !*path) {
    return;
  }
  const std::string img = write_red_ppm_();
  const std::string wav = write_sine_wav_();
  const std::string prompt =
      "Here is a picture: "
      + vpipe::media_line::make_fs_marker(
            vpipe::media_line::Modality::Image, img)
      + " and a sound clip: "
      + vpipe::media_line::make_fs_marker(
            vpipe::media_line::Modality::Audio, wav)
      + " What is the dominant color of the picture?";
  auto texts = run_chat_turn_(path, prompt);
  ASSERT_TRUE(!texts.empty());
  bool has_red = false;
  for (const auto& t : texts) {
    std::printf("[text_chat_stage.media_gemma] text='%s'\n", t.c_str());
    if (lower_(t).find("red") != std::string::npos) { has_red = true; }
  }
  EXPECT_TRUE(has_red);
}

#endif  // VPIPE_BUILD_APPLE_SILICON
