// DeepSeek-V4.1-Flash (deepseek-ai/DeepSeek-V4.1-Flash) port tests, against
// the release's own inference code (tools/dump_deepseek_v41_golden.py: its
// model.py and engram.py imported as they are, the tilelang kernels
// restated in torch, run in fp32 -- the truth -- and in bf16, whose distance
// from fp32 is the bar).
//
// Env:
//   VPIPE_DSV41_SMALL_GOLDEN  a dir written by `dump_deepseek_v41_golden.py
//                             small`: the synthetic checkpoint (model/, the
//                             release's layout and tokenizer) and its golden.
//   VPIPE_DSV41_TEST_MODEL_PATH + VPIPE_DSV41_GOLDEN
//                             the release and a `real` golden.
// Each test skips without its variables.
//
// GEOMETRY of the synthetic model: an 80-token prompt over a 16-position
// window, so every query past 16 reads the ring and the chunk; ratio-2
// layers with 40 compressed positions and an index top-k of 16, so the
// indexer really chooses; a candidate source keeping 4 blocks of 8, so the
// reindexing layer after it is restricted; both engram layers. The decode
// steps run one row through every one of those paths after the prefill.

#include "minitest.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-bag.h"
#include "common/flex-data.h"
#include "common/media-decode.h"
#include "common/session.h"
#include "generative-models/deepseek-v41/deepseek-v41-config.h"
#include "generative-models/deepseek-v41/deepseek-v41-engram.h"
#include "generative-models/deepseek-v41/metal-deepseek-v41-model.h"
#include "generative-models/deepseek-v41/metal-deepseek-v41-model-exec.h"
#include "generative-models/deepseek-v41/metal-deepseek-v41-vision.h"
#include "generative-models/chat-template.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/lm-memory-limit.h"
#include "generative-models/lm-run-memory.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/moe-expert-cache.h"
#include "generative-models/sampler.h"
#include "generative-models/speculative-decode.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "pipeline/resource-claim-delivery.h"
#include "pipeline/resource-plan.h"
#include "stages/model-memory-host.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <format>
#include <fstream>
#include <sstream>
#include <string>
#include <list>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
namespace fs = std::filesystem;

namespace {

// The process's footprint (what Activity Monitor reports as its memory).
std::size_t
footprint_()
{
  task_vm_info_data_t i;
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&i, &n) !=
      KERN_SUCCESS) {
    return 0;
  }
  return (std::size_t)i.phys_footprint;
}

std::string
env_(const char* k)
{
  const char* v = std::getenv(k);
  return (v != nullptr && *v != '\0') ? std::string(v) : std::string();
}

template <typename T>
std::vector<T>
read_raw_(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary);
  std::vector<T> out;
  if (!in) { return out; }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  out.resize((std::size_t)n / sizeof(T));
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

FlexData
read_json_(const fs::path& p)
{
  std::ifstream in(p);
  if (!in) { return FlexData(); }
  return FlexData::from_json(in);
}

double
rel_l2_(const float* a, const float* b, std::size_t n)
{
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

std::vector<std::int32_t>
ints_(const FlexData& meta, const char* key)
{
  std::vector<std::int32_t> v;
  auto o = meta.as_object();
  if (!o.contains(key)) { return v; }
  const FlexData a = o.at(key);
  for (auto e : a.as_array()) { v.push_back((std::int32_t)e.as_int()); }
  return v;
}

int
argmax_(const float* x, int n)
{
  int b = 0;
  for (int i = 1; i < n; ++i) {
    if (x[i] > x[b]) { b = i; }
  }
  return b;
}

}  // namespace

// The engram hashing as the release computes it: the normalized token map
// (every one of the 129280 entries), the PCG64-drawn multipliers, the
// primes, and the rows each prompt position reads.
TEST(deepseek_v41, engram_matches_reference)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  DeepseekV41Config cfg;
  std::string err;
  ASSERT_TRUE(DeepseekV41Config::from_dir(md.string(), &cfg, &err));
  if (!err.empty()) { std::printf("[dsv41] %s\n", err.c_str()); }
  DeepseekV41Engram eg;
  ASSERT_TRUE(eg.init(cfg, (md / "tokenizer.json").string(), &err));
  if (!err.empty()) { std::printf("[dsv41] %s\n", err.c_str()); return; }
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> tm =
      read_raw_<std::int32_t>(fs::path(gd) / "token_map.i32");
  ASSERT_TRUE(!tm.empty());
  int diff = 0;
  for (std::size_t i = 0; i < tm.size() && i < eg.token_map().size(); ++i) {
    if (tm[i] != eg.token_map()[i]) {
      if (diff < 10) {
        std::printf("[dsv41] token %zu: compressed %d, reference %d\n", i,
                    eg.token_map()[i], tm[i]);
      }
      ++diff;
    }
  }
  std::printf("[dsv41] token map: %d of %zu differ; %d compressed ids\n",
              diff, tm.size(), eg.compressed_vocab());
  EXPECT_TRUE(diff == 0);
  // multipliers and primes
  {
    auto mo = meta.as_object();
    const FlexData mfd = mo.at("engram_multipliers");
    int l = 0;
    for (auto row : mfd.as_array()) {
      int j = 0;
      for (auto v : row.as_array()) {
        EXPECT_TRUE(eg.multipliers()[(std::size_t)l][(std::size_t)j] ==
                    v.as_int());
        ++j;
      }
      ++l;
    }
    EXPECT_TRUE(l == eg.layers());
    const FlexData pfd = mo.at("engram_primes");
    l = 0;
    for (auto layer : pfd.as_array()) {
      int c = 0;
      for (auto per : layer.as_array()) {
        for (auto v : per.as_array()) {
          EXPECT_TRUE(eg.primes()[(std::size_t)l][(std::size_t)c] ==
                      v.as_int());
          ++c;
        }
      }
      ++l;
    }
  }
  // the rows of every prompt position
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  const std::vector<std::int32_t> ref =
      read_raw_<std::int32_t>(fs::path(gd) / "engram_hash.i32");
  const int cols = eg.cols(), nl = eg.layers();
  ASSERT_TRUE(ref.size() == ids.size() * (std::size_t)(nl * cols));
  std::vector<std::int32_t> hist;
  std::vector<std::int64_t> out((std::size_t)(nl * cols));
  int bad = 0;
  for (std::size_t t = 0; t < ids.size(); ++t) {
    hist.push_back(eg.compress(ids[t]));
    eg.hash(hist, out.data());
    for (int k = 0; k < nl * cols; ++k) {
      if (out[(std::size_t)k] != ref[t * (std::size_t)(nl * cols) + k]) {
        ++bad;
      }
    }
  }
  std::printf("[dsv41] engram rows: %d of %zu differ\n", bad, ref.size());
  EXPECT_TRUE(bad == 0);
}

// The release's tokenizer against Hugging Face's: DeepSeek's pre-tokenizer
// is three Isolated splits in sequence (numbers by threes, CJK runs, then
// a punctuation / symbol aware split), a scanner of its own here.
TEST(deepseek_v41, tokenizer_matches_reference)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  auto tok = Tokenizer::from_huggingface_json((md / "tokenizer.json").string(),
                                              nullptr);
  ASSERT_TRUE(tok != nullptr);
  if (tok == nullptr) { return; }
  const FlexData cases = read_json_(fs::path(gd) / "tokenizer_cases.json");
  int n = 0, bad = 0;
  for (auto c : cases.as_array()) {
    auto o = c.as_object();
    const FlexData tfd = o.at("text");
    const std::string text(tfd.as_string());
    std::vector<std::int32_t> want;
    const FlexData ifd = o.at("ids");
    for (auto v : ifd.as_array()) { want.push_back((std::int32_t)v.as_int()); }
    const std::vector<std::int32_t> got = tok->encode(text);
    ++n;
    if (got != want) {
      ++bad;
      std::printf("[dsv41] tokenizer: \"%s\": %zu ids, reference %zu\n",
                  text.c_str(), got.size(), want.size());
    }
    // and back
    if (tok->decode(got) != text) {
      std::printf("[dsv41] tokenizer: \"%s\" does not round-trip\n",
                  text.c_str());
      ++bad;
    }
  }
  std::printf("[dsv41] tokenizer: %d of %d cases differ\n", bad, n);
  EXPECT_TRUE(bad == 0);
}

// The chat template: a turn as the release's encoding.py renders it (its
// test cases are text; here the ids of the same pieces).
TEST(deepseek_v41, chat_template_renders_the_reference_turns)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  auto tok = Tokenizer::from_huggingface_json((md / "tokenizer.json").string(),
                                              nullptr);
  ASSERT_TRUE(tok != nullptr);
  if (tok == nullptr) { return; }
  auto id = [&](const char* s) { return tok->special_token_id(s); };
  const std::int32_t bos = id("<\uff5cbegin\u2581of\u2581sentence\uff5c>");
  const std::int32_t user = id("<\uff5cUser\uff5c>");
  const std::int32_t asst = id("<\uff5cAssistant\uff5c>");
  const std::int32_t sys = id("<\uff5cSystem\uff5c>");
  const std::int32_t think = id("<think>"), unthink = id("</think>");
  const std::int32_t eos = id("<\uff5cend\u2581of\u2581sentence\uff5c>");
  ASSERT_TRUE(bos == 0 && eos == 1 && user >= 0 && asst >= 0 && sys >= 0 &&
              think >= 0 && unthink >= 0);
  auto text = [&](const char* s) { return tok->encode(s); };
  auto cat = [](std::vector<std::vector<std::int32_t>> parts) {
    std::vector<std::int32_t> v;
    for (auto& p : parts) { v.insert(v.end(), p.begin(), p.end()); }
    return v;
  };
  // chat mode: <bos><User>What is 2+2?<Assistant></think>
  {
    auto t = make_chat_template("DeepseekV41ForCausalLM", *tok);
    ASSERT_TRUE(t != nullptr);
    std::vector<std::int32_t> got;
    t->render_user_turn("What is 2+2?", true, &got);
    EXPECT_TRUE(got == cat({{bos, user}, text("What is 2+2?"),
                            {asst, unthink}}));
    EXPECT_TRUE(t->is_stop_token(eos) && t->assistant_close_token_id() == eos);
    EXPECT_TRUE(!t->assistant_prompt_opens_thinking());
  }
  // thinking mode with a system message, then a second turn
  {
    auto t = make_chat_template("DeepseekV41ForCausalLM", *tok, false, "high");
    std::vector<std::int32_t> got;
    ASSERT_TRUE(t->render_system_message("You are a helpful assistant.", {},
                                         true, &got));
    ASSERT_TRUE(t->render_message("user", "What is 2+2?", false, &got));
    ASSERT_TRUE(t->render_generation_prompt(&got));
    const std::vector<std::int32_t> want = cat(
        {{bos, sys},
         text("Reasoning Effort: 75 (range 1-100, the higher the value, the "
              "more thorough the reasoning)\n\nYou are a helpful "
              "assistant."),
         {user}, text("What is 2+2?"), {asst, think}});
    EXPECT_TRUE(got == want);
    EXPECT_TRUE(t->assistant_prompt_opens_thinking());
    EXPECT_TRUE(t->sanitize_output("Simple.</think>4") == "4");
  }
}

#define DSML_ "\uff5cDSML\uff5c"

// A conversation's tools, its call and the call's result, as the
// release's encoding.py renders its first test case in chat mode
// (encoding/tests/test_input_1.json through encode_messages(...,
// thinking_mode="chat"): the prompt below) -- its special and added
// tokens single ids, as HF's tokenizer splits them; the tools as the
// chat stage hands them over (a `{"type": "function", ...}` line each,
// FlexData's compact JSON) -- and the call read back from what the
// model wrote.
TEST(deepseek_v41, chat_template_renders_the_reference_tools)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  auto tok = Tokenizer::from_huggingface_json((md / "tokenizer.json").string(),
                                              nullptr);
  ASSERT_TRUE(tok != nullptr);
  if (tok == nullptr) { return; }
  ASSERT_TRUE(tok->special_token_id(DSML_) >= 0);
  // the reference's text as ids: the marks single, the runs between them
  // encoded whole
  auto ids = [&](std::string_view s) {
    const char* const marks[] = {
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
        "<\uff5cend\u2581of\u2581sentence\uff5c>", "<\uff5cUser\uff5c>",
        "<\uff5cAssistant\uff5c>", "<\uff5cSystem\uff5c>", DSML_, "<think>",
        "</think>"};
    std::vector<std::int32_t> v;
    std::size_t pos = 0;
    auto run = [&](std::string_view t) {
      const auto e = tok->encode(std::string(t));
      v.insert(v.end(), e.begin(), e.end());
    };
    while (pos < s.size()) {
      std::size_t at = std::string_view::npos;
      const char* hit = nullptr;
      for (const char* m : marks) {
        const std::size_t p = s.find(m, pos);
        if (p < at) {
          at = p;
          hit = m;
        }
      }
      if (hit == nullptr) {
        run(s.substr(pos));
        break;
      }
      if (at > pos) { run(s.substr(pos, at - pos)); }
      v.push_back(tok->special_token_id(hit));
      pos = at + std::strlen(hit);
    }
    return v;
  };
  const std::string want =
      "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cSystem\uff5c>You are "
      "a helpful assistant.\n\n## Tools\n\nYou have access to a set of "
      "tools to help answer the user's question. You can invoke tools by "
      "writing a \"<" DSML_ " calls>\" block like the following:\n\n<" DSML_
      " calls>\n<" DSML_ " invoke name=\"$TOOL_NAME\">\n<" DSML_ " parameter "
      "name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</"
      DSML_ " parameter>\n...\n</" DSML_ " invoke>\n<" DSML_ " invoke "
      "name=\"$TOOL_NAME2\">\n...\n</" DSML_ " invoke>\n</" DSML_ " calls>"
      "\n\nString parameters should be specified as is and set "
      "`string=\"true\"`. For all other types (numbers, booleans, arrays, "
      "objects), pass the value in JSON format and set `string=\"false\"`."
      "\n\nIf thinking_mode is enabled (triggered by <think>), you MUST "
      "output your complete reasoning inside <think>...</think> BEFORE any "
      "tool calls or final response.\n\nOtherwise, output directly after "
      "</think> with tool calls or final response.\n\n### Available Tool "
      "Schemas\n\n{\"name\": \"get_weather\", \"description\": \"Get the "
      "weather for a specific location\", \"parameters\": {\"type\": "
      "\"object\", \"properties\": {\"location\": {\"type\": \"string\", "
      "\"description\": \"The city name\"}, \"unit\": {\"type\": "
      "\"string\", \"enum\": [\"celsius\", \"fahrenheit\"], "
      "\"description\": \"Temperature unit\"}}, \"required\": "
      "[\"location\"]}}\n{\"name\": \"search\", \"description\": \"Search "
      "the web for information\", \"parameters\": {\"type\": \"object\", "
      "\"properties\": {\"query\": {\"type\": \"string\", \"description\": "
      "\"Search query\"}, \"num_results\": {\"type\": \"integer\", "
      "\"description\": \"Number of results to return\"}}, \"required\": "
      "[\"query\"]}}\n\nYou MUST strictly follow the above defined tool "
      "name and parameter schemas to invoke tool calls.\n<\uff5cUser\uff5c>"
      "What's the weather like in Beijing?<\uff5cAssistant\uff5c></think>"
      "\n\n<" DSML_ " calls>\n<" DSML_ " invoke name=\"get_weather\">\n<"
      DSML_ " parameter name=\"location\" string=\"true\">Beijing</" DSML_
      " parameter>\n<" DSML_ " parameter name=\"unit\" string=\"true\">"
      "celsius</" DSML_ " parameter>\n</" DSML_ " invoke>\n</" DSML_
      " calls><\uff5cend\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>"
      "<tool_result>{\"temperature\": 22, \"condition\": \"sunny\", "
      "\"humidity\": 45}</tool_result><\uff5cAssistant\uff5c></think>The "
      "weather in Beijing is currently sunny with a temperature of "
      "22\u00b0C and 45% humidity.<\uff5cend\u2581of\u2581sentence\uff5c>";
  const std::string tools =
      "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
      "\"description\":\"Get the weather for a specific location\","
      "\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":"
      "{\"type\":\"string\",\"description\":\"The city name\"},\"unit\":"
      "{\"type\":\"string\",\"enum\":[\"celsius\",\"fahrenheit\"],"
      "\"description\":\"Temperature unit\"}},\"required\":"
      "[\"location\"]}}}\n{\"type\":\"function\",\"function\":{\"name\":"
      "\"search\",\"description\":\"Search the web for information\","
      "\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":"
      "{\"type\":\"string\",\"description\":\"Search query\"},"
      "\"num_results\":{\"type\":\"integer\",\"description\":\"Number of "
      "results to return\"}},\"required\":[\"query\"]}}}";
  const std::string call =
      "\n\n<" DSML_ " calls>\n<" DSML_ " invoke name=\"get_weather\">\n<"
      DSML_ " parameter name=\"location\" string=\"true\">Beijing</" DSML_
      " parameter>\n<" DSML_ " parameter name=\"unit\" string=\"true\">"
      "celsius</" DSML_ " parameter>\n</" DSML_ " invoke>\n</" DSML_
      " calls>";
  const std::string result =
      "{\"temperature\": 22, \"condition\": \"sunny\", \"humidity\": 45}";
  auto t = make_chat_template("DeepseekV41ForCausalLM", *tok);
  ASSERT_TRUE(t != nullptr);
  std::vector<std::int32_t> got;
  ASSERT_TRUE(t->render_system_message("You are a helpful assistant.",
                                       tools, true, &got));
  ASSERT_TRUE(t->render_message(
      "user", "What's the weather like in Beijing?", false, &got));
  ASSERT_TRUE(t->render_message("assistant", call, false, &got));
  ASSERT_TRUE(t->render_tool_results_message(
      std::span<const std::string>(&result, 1), &got));
  ASSERT_TRUE(t->render_message(
      "assistant",
      "The weather in Beijing is currently sunny with a temperature of "
      "22\u00b0C and 45% humidity.",
      false, &got));
  const std::vector<std::int32_t> ref = ids(want);
  std::size_t same = 0;
  while (same < got.size() && same < ref.size() && got[same] == ref[same]) {
    ++same;
  }
  std::printf("[dsv41] tools: %zu ids, the reference %zu, alike to %zu\n",
              got.size(), ref.size(), same);
  EXPECT_TRUE(got == ref);

  // THINKING mode, the case's own. With tools every earlier turn keeps
  // its reasoning (encoding.py renders with drop_thinking off) -- each
  // turn as the stage decoded it, `{reasoning}</think>{answer}` -- and
  // without them these two keep theirs too: both follow the last user
  // message. The text is the chat mode's with the effort and the
  // reasoning in (checked against encoding.py's thinking output).
  const std::string effort =
      "Reasoning Effort: 75 (range 1-100, the higher the value, the more "
      "thorough the reasoning)\n\n";
  const std::string rc1 =
      "The user wants the weather in Beijing. I should call get_weather.";
  const std::string rc2 =
      "Got the weather data. Let me format a nice response.";
  const std::string answer =
      "The weather in Beijing is currently sunny with a temperature of "
      "22°C and 45% humidity.";
  const std::string sys = "<｜System｜>";
  const std::string asst = "<｜Assistant｜>";
  auto first_to = [](std::string* s, const std::string& a,
                     const std::string& b) {
    const std::size_t p = s->find(a);
    if (p != std::string::npos) { s->replace(p, a.size(), b); }
  };
  std::string want_think = want;
  first_to(&want_think, sys, sys + effort);
  first_to(&want_think, asst + "</think>", asst + "<think>" + rc1 + "</think>");
  first_to(&want_think, asst + "</think>", asst + "<think>" + rc2 + "</think>");
  std::string want_bare = want_think;   // no tools: their section goes
  {
    const std::size_t a = want_bare.find("\n\n## Tools");
    const std::string end = "to invoke tool calls.\n";
    const std::size_t b = want_bare.find(end);
    ASSERT_TRUE(a != std::string::npos && b != std::string::npos);
    if (a != std::string::npos && b != std::string::npos) {
      want_bare.erase(a, b + end.size() - a);
    }
  }
  auto tt = make_chat_template("DeepseekV41ForCausalLM", *tok,
                               /*disable_thinking=*/false);
  ASSERT_TRUE(tt != nullptr);
  auto thinking = [&](std::string_view tools_json) {
    std::vector<std::int32_t> v;
    tt->render_system_message("You are a helpful assistant.", tools_json,
                              true, &v);
    tt->render_message("user", "What's the weather like in Beijing?", false,
                       &v);
    tt->render_assistant_message(rc1 + "</think>" + call, tools_json, true,
                                 false, &v);
    tt->render_tool_results_message(
        std::span<const std::string>(&result, 1), &v);
    tt->render_assistant_message(rc2 + "</think>" + answer, tools_json,
                                 true, false, &v);
    return v;
  };
  EXPECT_TRUE(thinking(tools) == ids(want_think));
  EXPECT_TRUE(thinking("") == ids(want_bare));

  // A turn BEFORE the last user message: its reasoning dropped without
  // tools, kept with them; chat mode drops it either way. encoding.py's
  // renderings of system / "Hi" / that turn / "Bye".
  const std::string bos = "<｜begin▁of▁sentence｜>";
  const std::string usr = "<｜User｜>";
  const std::string eos = "<｜end▁of▁sentence｜>";
  const std::string greet = "Greet back.</think>Hello!";
  auto earlier = [&](const ChatTemplate& t, std::string_view tools_json) {
    std::vector<std::int32_t> v;
    t.render_system_message("You are a helpful assistant.", tools_json,
                            true, &v);
    t.render_message("user", "Hi", false, &v);
    t.render_assistant_message(greet, tools_json, false, false, &v);
    t.render_user_turn("Bye", false, &v);
    return v;
  };
  const std::string head = sys + "You are a helpful assistant.";
  EXPECT_TRUE(earlier(*tt, "") ==
              ids(bos + sys + effort + "You are a helpful assistant." + usr +
                  "Hi" + asst + "</think>Hello!" + eos + usr + "Bye" + asst +
                  "<think>"));
  EXPECT_TRUE(earlier(*tt, tools) ==
              ids(want_think.substr(0, want_think.find(usr)) + usr + "Hi" +
                  asst + "<think>Greet back.</think>Hello!" + eos + usr +
                  "Bye" + asst + "<think>"));
  EXPECT_TRUE(earlier(*t, "") ==
              ids(bos + head + usr + "Hi" + asst + "</think>Hello!" + eos +
                  usr + "Bye" + asst + "</think>"));
  // render_message's assistant turn is the turn with nothing to frame it
  {
    std::vector<std::int32_t> a, b;
    tt->render_message("assistant", greet, false, &a);
    tt->render_assistant_message(greet, {}, false, false, &b);
    EXPECT_TRUE(a == b);
  }

  // the call read back; and leniently, a call outside a calls block, left
  // open, its number as JSON
  auto calls = t->parse_tool_calls(call);
  ASSERT_TRUE(calls.size() == 1);
  if (calls.size() == 1) {
    EXPECT_TRUE(calls[0].name == "get_weather");
    const FlexData a = FlexData::from_json(calls[0].arguments_json);
    EXPECT_TRUE(a.as_object().at("location").as_string() == "Beijing");
    EXPECT_TRUE(a.as_object().at("unit").as_string() == "celsius");
  }
  calls = t->parse_tool_calls(
      "Looking.\n<" DSML_ " invoke name=\"search\">\n<" DSML_ " parameter "
      "name=\"query\" string=\"true\">foxes in 5 words</" DSML_
      " parameter>\n<" DSML_ " parameter name=\"num_results\" "
      "string=\"false\">5</" DSML_ " parameter>\n");
  ASSERT_TRUE(calls.size() == 1);
  if (calls.size() == 1) {
    EXPECT_TRUE(calls[0].name == "search");
    const FlexData a = FlexData::from_json(calls[0].arguments_json);
    EXPECT_TRUE(a.as_object().at("query").as_string() == "foxes in 5 words");
    EXPECT_TRUE(a.as_object().at("num_results").as_int() == 5);
  }
  EXPECT_TRUE(t->parse_tool_calls("No call at all.").empty());
  EXPECT_TRUE(t->tool_call_open_marker() == "<" DSML_ " calls>");
  EXPECT_TRUE(t->tool_call_close_marker() == "</" DSML_ " calls>");
}

#undef DSML_

// The synthetic model through the prefill (every row's logits) and then
// decode steps, against the fp32 reference, held to the reference's own
// bf16 distance.
TEST(deepseek_v41, small_model_matches_reference)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  const std::vector<std::int32_t> dec = ints_(meta, "decode_ids");
  const std::vector<float> l32 =
      read_raw_<float>(fs::path(gd) / "logits_f32.f32");
  const std::vector<float> d32 =
      read_raw_<float>(fs::path(gd) / "dlogits_f32.f32");
  auto mo = meta.as_object();
  const double bar = mo.at("bf16_vs_f32_logits").as_real();
  const double dbar = mo.at("bf16_vs_f32_dlogits").as_real();
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  std::string err;
  for (const bool bf16 : {true, false}) {
    MetalDeepseekV41Model::Options opt;
    opt.use_bf16 = bf16;
    auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                         mc, opt, &err);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) {
      std::printf("[dsv41] %s\n", err.c_str());
      return;
    }
    const int V = m->config().vocab;
    std::vector<float> lg;
    ASSERT_TRUE(m->forward(1, ids, &lg, /*all_rows=*/true, &err));
    if (!err.empty()) { std::printf("[dsv41] %s\n", err.c_str()); return; }
    // the prefill's attention on the matrix cores where there are some
    EXPECT_TRUE((m->nax_attentions() > 0) == mc->supports_matrix_cores());
    ASSERT_TRUE(lg.size() == l32.size());
    const double rel = rel_l2_(lg.data(), l32.data(), lg.size());
    int agree = 0;
    const int T = (int)ids.size();
    for (int t = 0; t < T; ++t) {
      agree += argmax_(lg.data() + (std::size_t)t * V, V) ==
               argmax_(l32.data() + (std::size_t)t * V, V);
    }
    std::vector<float> dl;
    for (std::int32_t t : dec) {
      std::vector<float> one;
      ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>{t}, &one, false,
                             &err));
      dl.insert(dl.end(), one.begin(), one.end());
    }
    const double drel =
        dl.size() == d32.size() ? rel_l2_(dl.data(), d32.data(), dl.size())
                                : 1e9;
    std::printf("[dsv41] %s: prefill logits %.5f from fp32 (ref bf16 %.5f), "
                "argmax %d/%d; decode %.5f (ref bf16 %.5f)\n",
                bf16 ? "bf16" : "f16", rel, bar, agree, T, drel, dbar);
    // The prefill on fp32's expert picks (force_routes; the golden's bf16r
    // the reference's bf16 on them): a bf16 run flips some near-tie pick
    // fp32 makes, and one flipped pick puts its row O(0.1) off wholesale
    // (MEASURED, the oq synthetic on the M5 Ultra: matmul2d's sums flip
    // row 43's at layer 0, steel's do not -- that row 0.153, the rest
    // 0.005 either way), so the free run's bf16 is one draw.
    double frel = -1.0, rbar = 0.0;
    const fs::path rf32 = fs::path(gd) / "route_f32.i32";
    if (bf16 && mo.contains("bf16r_vs_f32_logits") && fs::exists(rf32)) {
      rbar = mo.at("bf16r_vs_f32_logits").as_real();
      const auto p32 = read_raw_<std::int32_t>(rf32);
      const int NL = m->config().n_layers, KK = m->config().top_k;
      std::vector<std::vector<std::int32_t>> fr((std::size_t)NL);
      for (int L = 0; L < NL; ++L) {
        const std::size_t o = (std::size_t)L * T * KK;
        fr[(std::size_t)L].assign(p32.begin() + (std::ptrdiff_t)o,
                                  p32.begin() +
                                      (std::ptrdiff_t)(o + T * KK));
      }
      m->force_routes(fr);
      std::vector<float> fl;
      ASSERT_TRUE(m->forward(2, ids, &fl, /*all_rows=*/true, &err));
      frel = fl.size() == l32.size()
                 ? rel_l2_(fl.data(), l32.data(), fl.size()) : 1e9;
      std::printf("[dsv41] bf16 on fp32's picks: prefill %.5f from fp32 "
                  "(ref bf16 %.5f)\n", frel, rbar);
      EXPECT_TRUE(frel < 1.5 * rbar);
    }
    // f16 is the tight check: its rounding is 8x finer than bf16's, so an
    // implementation that matches the reference's arithmetic lands inside
    // the reference's own bf16 distance (0.002-0.031 measured; a misread
    // tensor O(1)) -- but not always far inside: a near-tie of the
    // indexer's top-k or a router flips a row wholesale in f16 too (the
    // M5's matrix-core prefill flipped row 64 at layer 2, putting one
    // synthetic at 0.031 where the M4 reads 0.016). bf16 is held to that
    // distance, the decode looser: six steps after a prefill are one draw
    // of a chaotic sum, and that draw put the reference's own bf16 0.034
    // from fp32 on one synthetic golden and 0.106 on another.
    if (bf16) {
      // the free run at the reference's bf16, or -- a pick flipped -- the
      // same arithmetic on fp32's picks (held above)
      EXPECT_TRUE(rel < 1.5 * bar || (frel >= 0.0 && frel < 1.5 * rbar));
      EXPECT_TRUE(drel < 3.0 * dbar);
    } else {
      EXPECT_TRUE(rel < bar);
      EXPECT_TRUE(drel < std::max(bar, dbar));
    }
    const auto& st = m->expert_cache()->stats();
    std::printf("[dsv41] expert cache: %llu picks, %.1f%% hits\n",
                (unsigned long long)st.picks, 100.0 * st.hit_rate());
  }
}

// Debug: each layer's residual against the reference's (f16, the closest
// arithmetic), to localize a departure. VPIPE_DSV41_LAYERS=1.
// The routed experts as grouped GEMMs (pairs sorted by expert into 32-row
// tiles) against the pair kernels: both routes engage where they are asked
// to, and the residual after the first layer -- the same input to both,
// so the experts' summation order is the only difference -- agrees to the
// compute dtype's rounding (a misread tile or a lost row is O(1)). The
// logits are reported: past a few layers a near-tie pick of the router or
// the indexer may flip, and the stack amplifies it.
TEST(deepseek_v41, grouped_experts_match_the_pairs)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  for (bool bf16 : {true, false}) {
    // 0: never grouped; 1: grouped on the f32 simdgroup GEMMs; 2: grouped
    // on the matrix cores where the GPU has them (else as 1)
    std::vector<float> lg[3], x0[3];
    std::uint64_t steps[3] = {0, 0, 0};
    long nax[3] = {0, 0, 0};
    for (int g = 0; g < 3; ++g) {
      ::setenv("VPIPE_DSV41_MOE_GROUPED_MIN", g == 0 ? "0" : "0.001", 1);
      ::setenv("VPIPE_DSV41_MOE_NAX", g == 2 ? "1" : "0", 1);
      std::string err;
      MetalDeepseekV41Model::Options opt;
      opt.use_bf16 = bf16;
      auto m = MetalDeepseekV41Model::load(
          WeightSet::open(md.string(), nullptr), mc, opt, &err);
      ASSERT_TRUE(m != nullptr);
      if (m == nullptr) { std::printf("[dsv41] %s\n", err.c_str()); break; }
      m->capture_layer(0);
      ASSERT_TRUE(m->forward(1, ids, &lg[g], /*all_rows=*/true, &err));
      steps[g] = m->grouped_moe_steps();
      nax[g] = m->nax_moe_gemms();
      x0[g] = m->captured();
    }
    ::unsetenv("VPIPE_DSV41_MOE_GROUPED_MIN");
    ::unsetenv("VPIPE_DSV41_MOE_NAX");
    bool sized = true;
    for (int g = 1; g < 3; ++g) {
      sized = sized && !x0[0].empty() && x0[0].size() == x0[g].size() &&
              lg[0].size() == lg[g].size();
    }
    ASSERT_TRUE(sized);
    if (!sized) { return; }
    double rx[3] = {0, 0, 0}, rl[3] = {0, 0, 0};
    for (int g = 1; g < 3; ++g) {
      rx[g] = rel_l2_(x0[g].data(), x0[0].data(), x0[0].size());
      rl[g] = rel_l2_(lg[g].data(), lg[0].data(), lg[0].size());
    }
    std::printf("[dsv41] %s: grouped %llu layer-steps, pairs %llu; layer 0 "
                "%.2e apart, logits %.5f; on the matrix cores (%ld GEMMs) "
                "%.2e / %.5f\n", bf16 ? "bf16" : "f16",
                (unsigned long long)steps[1], (unsigned long long)steps[0],
                rx[1], rl[1], nax[2], rx[2], rl[2]);
    EXPECT_TRUE(steps[0] == 0 && steps[1] > 0 && steps[2] > 0);
    EXPECT_TRUE(nax[1] == 0);
    // the twins engage wherever there are matrix cores (the golden's
    // widths are multiples of 64)
    EXPECT_TRUE((nax[2] > 0) == mc->supports_matrix_cores());
    EXPECT_TRUE(rx[1] < (bf16 ? 2e-3 : 3e-4));
    // staged in the element type: a weight's dequant rounded once more
    EXPECT_TRUE(rx[2] < (bf16 ? 6e-3 : 1e-3));
  }
}

// CED prefill (the decoder half on the prompt's last window): the encoder
// is the whole stack's to the bit -- the residual copies after the last
// encoder layer, every row -- and the decoder lands near it: the last
// row's logits and a greedy continuation's, against the whole-stack
// prefill. Engaged once, on a prompt longer than the window, and never
// where every row's logits are asked for.
TEST(deepseek_v41, ced_prefill_keeps_the_encoder_exact)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  ::unsetenv("VPIPE_DSV41_CED");
  for (bool bf16 : {true, false}) {
    std::vector<float> enc[2], last[2], dec[2], draft[2];
    std::vector<std::int32_t> dtok[2];
    std::vector<std::int32_t> toks[2], mine[2];
    long pieces[2] = {0, 0};
    int mid = -1, win = 0, vocab = 0;
    for (int g = 0; g < 2; ++g) {
      std::string err;
      MetalDeepseekV41Model::Options opt;
      opt.use_bf16 = bf16;
      opt.ced = g == 1;
      opt.dspark = true;          // its taps are decoder layers here
      auto m = MetalDeepseekV41Model::load(
          WeightSet::open(md.string(), nullptr), mc, opt, &err);
      ASSERT_TRUE(m != nullptr);
      if (m == nullptr) { std::printf("[dsv41] %s\n", err.c_str()); return; }
      mid = m->ced_mid();
      win = m->config().window;
      vocab = m->config().vocab;
      ASSERT_TRUE(mid > 0);
      if (mid <= 0) { return; }
      ASSERT_TRUE((int)ids.size() > m->config().window);
      m->capture_layer(mid - 1);
      ASSERT_TRUE(m->forward(1, ids, &last[g], /*all_rows=*/false, &err));
      enc[g] = m->captured();
      m->capture_layer(-1);
      pieces[g] = m->ced_pieces();
      // greedy steps on from the prompt, the CED model fed the whole
      // stack's tokens (a near tie may pick otherwise)
      const int V = m->config().vocab;
      auto pick = [&](const std::vector<float>& lg) {
        return argmax_(lg.data(), V);
      };
      // the token fed at step k: this model's pick, the whole stack's
      // for the CED model
      auto feed = [&](std::int32_t t) {
        if (g == 1) { t = toks[0][toks[1].size()]; }
        toks[g].push_back(t);
        return t;
      };
      std::int32_t t = pick(last[g]);
      // a DSpark draft from the prompt's taps, the same token both ways
      if (m->has_dspark()) {
        std::vector<float> dc;
        const std::int32_t t1 = g == 0 ? t : toks[0][0];
        ASSERT_TRUE(m->dspark_draft(1, t1, &dtok[g], &dc, &draft[g], &err));
      }
      for (int k = 0; k < 6; ++k) {
        t = feed(t);
        std::vector<float> lg;
        ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>{t}, &lg, false,
                               &err));
        dec[g].insert(dec[g].end(), lg.begin(), lg.end());
        mine[g].push_back(argmax_(lg.data(), V));
        t = pick(lg);
      }
      // a second turn in the same context: the decoder's rings now hold
      // the first turn's rows, which its windows must not read
      std::vector<float> lg2;
      ASSERT_TRUE(m->forward(1, ids, &lg2, /*all_rows=*/false, &err));
      dec[g].insert(dec[g].end(), lg2.begin(), lg2.end());
      mine[g].push_back(argmax_(lg2.data(), V));
      t = pick(lg2);
      for (int k = 0; k < 3; ++k) {
        t = feed(t);
        std::vector<float> lg;
        ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>{t}, &lg, false,
                               &err));
        dec[g].insert(dec[g].end(), lg.begin(), lg.end());
        mine[g].push_back(argmax_(lg.data(), V));
        t = pick(lg);
      }
      pieces[g] = m->ced_pieces();
      // every row's logits: never CED
      std::vector<float> all;
      ASSERT_TRUE(m->forward(2, ids, &all, /*all_rows=*/true, &err));
      EXPECT_TRUE(m->ced_pieces() == pieces[g]);
    }
    ASSERT_TRUE(!enc[0].empty() && enc[0].size() == enc[1].size() &&
                last[0].size() == last[1].size() &&
                dec[0].size() == dec[1].size());
    if (enc[0].empty() || enc[0].size() != enc[1].size() ||
        last[0].size() != last[1].size() || dec[0].size() != dec[1].size()) {
      return;
    }
    const bool same = std::memcmp(enc[0].data(), enc[1].data(),
                                  enc[0].size() * 4) == 0;
    const double rl = rel_l2_(last[1].data(), last[0].data(), last[0].size());
    const double rd = rel_l2_(dec[1].data(), dec[0].data(), dec[0].size());
    // The draft's rows up to the first draft token the two pick apart: a
    // row's logits take the Markov bias of the token before it, so past a
    // split near tie the rows read other inputs (MEASURED on the M4,
    // small3-oqn, a tail past the window: bf16 rows 0.017 0.011 then
    // ~1.0 after its second token went the other way; f16 0.008 each)
    std::size_t nr = dtok[0].size() + 1;
    for (std::size_t k = 0; k < dtok[0].size() && k < dtok[1].size(); ++k) {
      if (dtok[0][k] != dtok[1][k]) {
        nr = k + 1;
        break;
      }
    }
    const std::size_t V = (std::size_t)std::max(vocab, 1);
    nr = std::min(nr, draft[0].size() / V);
    const double rs = draft[0].size() == draft[1].size() && nr > 0
                          ? rel_l2_(draft[1].data(), draft[0].data(), nr * V)
                          : -1.0;
    // each step's own pick, fed the same tokens
    int agree = 0;
    for (std::size_t k = 0; k < mine[0].size(); ++k) {
      agree += mine[0][k] == mine[1][k];
    }
    std::printf("[dsv41] %s CED (decoder from layer %d on the last %d of %zu "
                "rows): encoder %s; last row's logits %.5f from the whole "
                "stack's, 6 decode steps + a second turn and 3 more %.5f, "
                "%d/%zu greedy picks the same (fed the same tokens); a "
                "DSpark draft's logits %.5f (its first %zu rows: to the "
                "first draft token picked apart)\n",
                bf16 ? "bf16" : "f16", mid, win + 8, ids.size(),
                same ? "BIT-IDENTICAL" : "DIFFERENT", rl, rd, agree,
                mine[0].size(), rs, nr);
    EXPECT_TRUE(pieces[0] == 0 && pieces[1] == 2);
    EXPECT_TRUE(same);
    EXPECT_TRUE(rl > 0.0);
    EXPECT_TRUE(rl < 0.05 && rd < 0.05);
    EXPECT_TRUE(rs >= 0.0 && rs < 0.05);
  }
}

// The CED rule (DeepInfer's KEEP_TOKENS >= window + backtracking): a CED
// prefill runs its decoder on the window plus the rows a rollback may take
// back, so a rollback into the prompt's last rows finds every decoder
// window it reads computed. A prompt taken back 8 rows and its last 8 run
// again (a few rows' kernels) against the prompt as it was, then 3 decode
// steps on each: the whole stack's model gives the kernels' own distance,
// and the CED model must keep to it (with only the window computed, the
// re-run rows read 8 decoder slots nothing wrote).
TEST(deepseek_v41, a_rollback_into_a_ced_prompt_reads_a_whole_window)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  const std::vector<std::int32_t> dec = ints_(meta, "decode_ids");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr || dec.size() < 3) { return; }
  ::unsetenv("VPIPE_DSV41_CED");
  const int P = (int)ids.size(), B = 8;
  for (bool bf16 : {true, false}) {
    double d[2] = {-1.0, -1.0};
    std::vector<double> row[2];
    long pieces[2] = {0, 0};
    for (int g = 0; g < 2; ++g) {
      std::string err;
      MetalDeepseekV41Model::Options opt;
      opt.use_bf16 = bf16;
      opt.ced = g == 1;
      auto m = MetalDeepseekV41Model::load(
          WeightSet::open(md.string(), nullptr), mc, opt, &err);
      ASSERT_TRUE(m != nullptr);
      if (m == nullptr) { std::printf("[dsv41] %s\n", err.c_str()); return; }
      ASSERT_TRUE(P > m->config().window + B + 1);
      std::vector<float> a, b, lg;
      ASSERT_TRUE(m->forward(1, ids, &lg, false, &err));
      a = lg;
      // the prompt's last rows' expert picks, for the rows run again (a
      // few rows' kernels against a prefill's: a near tie in a pick would
      // read as a distance here)
      const int KK = m->config().top_k;
      std::vector<std::vector<std::int32_t>> fr;
      for (const auto& rt : m->routes()) {
        fr.emplace_back(rt.end() - (std::ptrdiff_t)(B * KK), rt.end());
      }
      ASSERT_TRUE(m->branch(1, 2));
      // too far back is refused; 8 back is not
      EXPECT_TRUE(!m->truncate(1, P - B - 1));
      ASSERT_TRUE(m->truncate(1, P - B));
      m->force_routes(fr);
      ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>(ids.end() - B,
                                                          ids.end()),
                             &lg, false, &err));
      b = lg;
      for (int k = 0; k < 3; ++k) {
        const std::vector<std::int32_t> t{dec[(std::size_t)k]};
        ASSERT_TRUE(m->forward(2, t, &lg, false, &err));
        a.insert(a.end(), lg.begin(), lg.end());
        ASSERT_TRUE(m->forward(1, t, &lg, false, &err));
        b.insert(b.end(), lg.begin(), lg.end());
      }
      pieces[g] = m->ced_pieces();
      ASSERT_TRUE(a.size() == b.size() && !a.empty());
      if (a.size() != b.size() || a.empty()) { return; }
      d[g] = rel_l2_(b.data(), a.data(), a.size());
      const std::size_t V = (std::size_t)m->config().vocab;
      for (std::size_t r = 0; r * V < a.size(); ++r) {
        row[g].push_back(rel_l2_(b.data() + r * V, a.data() + r * V, V));
      }
    }
    // per row (the last row run again, then each step), CED against the
    // whole stack: a near tie split by a few rows' kernels against a
    // prefill's (an index pick nothing forces) moves one row of either
    // arm; reading 8 decoder slots nothing wrote moves them all (MEASURED
    // with the decoder on the window alone, f16: every row 2.1-4.1x)
    std::string line;
    int over = 0;
    for (std::size_t r = 0; r < row[0].size() && r < row[1].size(); ++r) {
      line += std::format(" {:.5f}/{:.5f}", row[1][r], row[0][r]);
      over += row[1][r] > std::max(2.0 * row[0][r], 2e-3);
    }
    std::printf("[dsv41] %s: a prompt taken back %d rows and run again, "
                "then 3 decode steps, from the prompt as it was, CED / "
                "whole stack:%s (overall %.5f / %.5f)\n",
                bf16 ? "bf16" : "f16", B, line.c_str(), d[1], d[0]);
    EXPECT_TRUE(pieces[0] == 0 && pieces[1] == 1);
    EXPECT_TRUE(d[0] >= 0.0 && d[1] >= 0.0 && row[1].size() == 4);
    // f16 the tight check; bf16's own near ties cover both arms
    EXPECT_TRUE(bf16 ? d[1] <= 1.5 * d[0] + 0.01 : over <= 1);
  }
}

// An oMLX pack's config carries no quantization_config but its own record
// of every quantized module: read where it kept the release's numbers
// (MXFP4 experts, MXFP8 projections, 4-bit affine n-gram tables) and
// refused, by module, where it requantized anything else.
TEST(deepseek_v41, omlx_config_is_read_and_checked)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  std::ifstream in(fs::path(gd) / "model" / "config.json");
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  if (text.find("omlx_deepseek_v41") == std::string::npos) {
    return;   // a release-layout synthetic
  }
  DeepseekV41Config c;
  std::string err;
  ASSERT_TRUE(DeepseekV41Config::from_json_text(text, &c, &err));
  EXPECT_TRUE(c.fp8 && c.fp4_experts);
  auto refused = [&](const std::string& from, const std::string& to) {
    std::string t2 = text;
    const std::size_t at = t2.find(from);
    if (at == std::string::npos) { return false; }
    t2.replace(at, from.size(), to);
    DeepseekV41Config c2;
    std::string e2;
    const bool ok = DeepseekV41Config::from_json_text(t2, &c2, &e2);
    std::printf("[dsv41] %s -> %s: %s\n", from.c_str(), to.c_str(),
                ok ? "accepted" : e2.c_str());
    return !ok;
  };
  EXPECT_TRUE(refused("\"mode\": \"mxfp4\"", "\"mode\": \"affine\""));
  EXPECT_TRUE(refused("\"bits\": 8", "\"bits\": 6"));
  EXPECT_TRUE(refused("\"group_size\": 32", "\"group_size\": 64"));
  // a pack below 4 bits (the oq synthetic): affine modules read, at widths
  // and groups the kernels have
  const bool oq = text.find("\"quantize_input\"") != std::string::npos;
  EXPECT_TRUE(c.affine == oq);
  if (oq) {
    EXPECT_TRUE(refused("\"bits\": 2", "\"bits\": 7"));
    EXPECT_TRUE(refused("\"group_size\": 64", "\"group_size\": 16"));
  }
}

// A decode step is the prefill's next row: the prompt prefilled and its
// continuation decoded a token at a time against the same tokens prefilled
// whole, every continuation row's logits (and the reference's fp32 decode
// beside both), in either dtype.
TEST(deepseek_v41, decode_steps_are_prefill_rows)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  const std::vector<std::int32_t> dec = ints_(meta, "decode_ids");
  const std::vector<float> d32 =
      read_raw_<float>(fs::path(gd) / "dlogits_f32.f32");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr || dec.empty()) { return; }
  for (bool bf16 : {true, false}) {
    std::string err;
    MetalDeepseekV41Model::Options opt;
    opt.use_bf16 = bf16;
    auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                         mc, opt, &err);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    const int V = m->config().vocab;
    std::vector<float> lg, dl;
    ASSERT_TRUE(m->forward(1, ids, &lg, false, &err));
    for (std::int32_t t : dec) {
      ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>{t}, &lg, false,
                             &err));
      dl.insert(dl.end(), lg.begin(), lg.end());
    }
    std::vector<std::int32_t> all = ids;
    all.insert(all.end(), dec.begin(), dec.end());
    std::vector<float> pf;
    ASSERT_TRUE(m->forward(2, all, &pf, /*all_rows=*/true, &err));
    const std::size_t T = ids.size();
    std::string line;
    double worst = 0.0;
    for (std::size_t i = 0; i < dec.size(); ++i) {
      const float* d = dl.data() + i * V;
      const float* p = pf.data() + (T + i) * V;
      const double dp = rel_l2_(d, p, (std::size_t)V);
      worst = std::max(worst, dp);
      line += std::format(" {:.4f}", dp);
      if (d32.size() == dec.size() * (std::size_t)V) {
        line += std::format("({:.4f}/{:.4f})",
                            rel_l2_(d, d32.data() + i * V, (std::size_t)V),
                            rel_l2_(p, d32.data() + i * V, (std::size_t)V));
      }
    }
    std::printf("[dsv41] %s: decode vs prefill rows (decode/prefill vs "
                "fp32):%s\n", bf16 ? "bf16" : "f16", line.c_str());
  }
}

// The parallel top-k picks what the serial one picks, to the index: rows
// with ties at the cut (a few values repeated), with -inf entries, with
// fewer finite than K, at the indexer's sizes.
TEST(deepseek_v41, parallel_topk_picks_the_same)
{
  using metal_compute::CommandStream;
  using metal_compute::ComputeEncoder;
  using metal_compute::ComputeFunction;
  using metal_compute::SharedBuffer;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("deepseek_v41_bf16");
  ComputeFunction f0 = lib.function("dsv41_topk");
  ComputeFunction f1 = lib.function("dsv41_topk_par");
  ASSERT_TRUE(f0.valid() && f1.valid());
  if (!f0.valid() || !f1.valid()) { return; }
  int checked = 0, differ = 0;
  for (auto [C, K] : std::vector<std::pair<int, int>>{
           {1000, 512}, {4096, 512}, {16384, 2048}, {65536, 512},
           {131073, 512}}) {
    for (int kind = 0; kind < 4; ++kind) {
      const int n = 2;
      SharedBuffer S = mc->make_shared_buffer((std::size_t)n * C * 4);
      auto* v = static_cast<float*>(S.contents());
      std::uint32_t x = 7u + (std::uint32_t)(C * 13 + kind);
      for (int i = 0; i < n * C; ++i) {
        x = x * 1664525u + 1013904223u;
        float f = ((float)(x >> 8) / 16777216.0f) * 4.0f - 1.0f;
        if (kind == 1) { f = (float)((x >> 8) % 37) * 0.25f; }   // ties
        if (kind == 2 && (x >> 7) % 3 != 0) { f = -INFINITY; }   // masked
        if (kind == 3 && i % C > K / 2) { f = -INFINITY; }       // few
        v[i] = f;
      }
      SharedBuffer out[2] = {mc->make_shared_buffer((std::size_t)n * K * 4),
                             mc->make_shared_buffer((std::size_t)n * K * 4)};
      for (int k = 0; k < 2; ++k) {
        CommandStream st = mc->make_command_stream();
        ComputeEncoder enc = st.begin_compute();
        enc.set_function(k == 0 ? f0 : f1);
        enc.set_buffer(0, S);
        enc.set_buffer(1, out[k]);
        enc.set_constant(2, C);
        enc.set_constant(3, K);
        enc.dispatch({(unsigned)(1024 * n), 1, 1}, {1024, 1, 1});
        enc.end();
        std::string why;
        ASSERT_TRUE(st.commit().wait_ok(&why));
      }
      ++checked;
      differ += std::memcmp(out[0].contents(), out[1].contents(),
                            (std::size_t)n * K * 4) != 0;
    }
  }
  std::printf("[dsv41] parallel top-k: %d of %d cases differ\n", differ,
              checked);
  EXPECT_TRUE(differ == 0);
}

// The collapse over several threadgroups a row (dsv41_hc_collapse) against
// the one-group kernels (dsv41_hc_pre_norm, dsv41_hc_post_pre(_moe)), the
// collapse alone, after a branch's post, after the MoE's parts: the
// copies a post writes and an unnormed collapse BIT-IDENTICAL, a normed one
// within an ulp (its sum of squares is added in another order) -- 40 runs
// a shape on fresh data, 1..8 rows and 300, so a group that read another's
// unfinished work would show as a gross error -- and every row's count back
// at zero after.
TEST(deepseek_v41, collapse_kernels_agree)
{
  using metal_compute::CommandStream;
  using metal_compute::ComputeEncoder;
  using metal_compute::ComputeFunction;
  using metal_compute::SharedBuffer;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("deepseek_v41_bf16");
  ComputeFunction fco = lib.function("dsv41_hc_collapse");
  ComputeFunction fpre = lib.function("dsv41_hc_pre_norm");
  ComputeFunction fpp = lib.function("dsv41_hc_post_pre");
  ComputeFunction fppm = lib.function("dsv41_hc_post_pre_moe");
  ASSERT_TRUE(fco.valid() && fpre.valid() && fpp.valid() && fppm.valid());
  if (!fco.valid() || !fpre.valid() || !fpp.valid() || !fppm.valid()) {
    return;
  }
  std::uint32_t x = 12345u;
  auto rnd = [&x] {
    x = x * 1664525u + 1013904223u;
    return ((float)(x >> 8) / 16777216.0f) * 2.0f - 1.0f;
  };
  auto bf = [](float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  };
  auto fl = [](std::uint16_t b) {
    const std::uint32_t u = (std::uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  auto bfbuf = [&](std::size_t n, float scale) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) { p[i] = bf(rnd() * scale); }
    return b;
  };
  auto f32buf = [&](std::size_t n, float lo, float hi) {
    SharedBuffer b = mc->make_shared_buffer(n * 4);
    auto* p = static_cast<float*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      p[i] = lo + (hi - lo) * 0.5f * (rnd() + 1.0f);
    }
    return b;
  };
  // a hidden size of whole groups, and the release's (5120: five of 1024)
  const int K = 6, tpg = 1024;
  for (int H : {4096, 5120}) {
  const int G = (H + tpg - 1) / tpg;
  int runs = 0, x_bad = 0, y_bad = 0, y_ulp = 0, cnt_bad = 0;
  double worst = 0.0;
  for (int n : {1, 2, 5, 8, 300}) {
    SharedBuffer ps = mc->make_shared_buffer((std::size_t)n * G * 4);
    SharedBuffer cnt = mc->make_shared_buffer((std::size_t)n * 4);
    std::memset(cnt.contents(), 0, (std::size_t)n * 4);
    for (int it = 0; it < 40; ++it) {
      const int mode = it % 4;    // 0 alone, 1 alone no norm, 2 post, 3 moe
      const int kk = mode < 2 ? -1 : mode == 2 ? 0 : K;
      const bool has_w = mode != 1;
      SharedBuffer X0 = bfbuf((std::size_t)n * 4 * H, 2.0f);
      SharedBuffer X1 = mc->make_shared_buffer(X0.byte_size());
      std::memcpy(X1.contents(), X0.contents(), X0.byte_size());
      SharedBuffer a = bfbuf((std::size_t)n * (kk > 0 ? K : 1) * H, 1.0f);
      SharedBuffer sh = bfbuf((std::size_t)n * H, 1.0f);
      SharedBuffer w = bfbuf((std::size_t)H, 1.5f);
      SharedBuffer pre = f32buf((std::size_t)n * 4, 0.1f, 1.0f);
      SharedBuffer post = f32buf((std::size_t)n * 4, 0.0f, 2.0f);
      SharedBuffer comb = f32buf((std::size_t)n * 16, 0.0f, 0.5f);
      SharedBuffer y0 = mc->make_shared_buffer((std::size_t)n * H * 2);
      SharedBuffer y1 = mc->make_shared_buffer((std::size_t)n * H * 2);
      CommandStream st = mc->make_command_stream();
      ComputeEncoder enc = st.begin_compute();
      // the one-group kernel
      if (kk < 0) {
        enc.set_function(fpre);
        enc.set_buffer(0, X0);
        enc.set_buffer(1, pre);
        enc.set_buffer(2, w);
        enc.set_buffer(3, y0);
        enc.set_constant(4, H);
        enc.set_constant(5, 1e-6f);
        enc.set_constant(6, has_w ? 1 : 0);
      } else {
        enc.set_function(kk > 0 ? fppm : fpp);
        enc.set_buffer(0, a);
        enc.set_buffer(1, X0);
        enc.set_buffer(2, pre);
        enc.set_buffer(3, post);
        enc.set_buffer(4, comb);
        enc.set_buffer(5, w);
        enc.set_buffer(6, y0);
        enc.set_constant(7, H);
        enc.set_constant(8, 1e-6f);
        if (kk > 0) {
          enc.set_buffer(9, sh);
          enc.set_constant(10, K);
        }
      }
      enc.dispatch({(unsigned)(1024 * n), 1, 1}, {1024, 1, 1});
      // the groups' kernel
      enc.set_function(fco);
      enc.set_buffer(0, a);
      enc.set_buffer(1, X1);
      enc.set_buffer(2, pre);
      enc.set_buffer(3, post);
      enc.set_buffer(4, comb);
      enc.set_buffer(5, w);
      enc.set_buffer(6, y1);
      enc.set_constant(7, H);
      enc.set_constant(8, 1e-6f);
      enc.set_buffer(9, sh);
      enc.set_constant(10, kk);
      enc.set_constant(11, has_w ? 1 : 0);
      enc.set_buffer(12, ps);
      enc.set_buffer(13, cnt);
      enc.dispatch({(unsigned)(tpg * G), (unsigned)n, 1},
                   {(unsigned)tpg, 1, 1});
      enc.end();
      std::string why;
      ASSERT_TRUE(st.commit().wait_ok(&why));
      ++runs;
      x_bad += std::memcmp(X0.contents(), X1.contents(), X0.byte_size()) !=
               0;
      const auto* p0 = static_cast<const std::uint16_t*>(y0.contents());
      const auto* p1 = static_cast<const std::uint16_t*>(y1.contents());
      bool bad = false;
      for (std::size_t i = 0; i < (std::size_t)n * H; ++i) {
        if (p0[i] == p1[i]) { continue; }
        ++y_ulp;
        const double d = std::fabs((double)fl(p0[i]) - (double)fl(p1[i]));
        const double tol = std::fabs((double)fl(p0[i])) / 64.0 + 1e-30;
        worst = std::max(worst, d / std::max(std::fabs((double)fl(p0[i])),
                                             1e-30));
        if (!has_w || d > tol) { bad = true; }
      }
      y_bad += bad;
      const auto* c = static_cast<const std::uint32_t*>(cnt.contents());
      for (int r = 0; r < n; ++r) { cnt_bad += c[r] != 0; }
    }
  }
  std::printf("[dsv41] collapse, H %d: %d runs, copies differ in %d, rows "
              "off in %d (%d values an ulp or so apart, worst %.2e), counts "
              "left %d\n", H, runs, x_bad, y_bad, y_ulp, worst, cnt_bad);
  EXPECT_TRUE(x_bad == 0);
  EXPECT_TRUE(y_bad == 0);
  EXPECT_TRUE(cnt_bad == 0);
  }
}

// The two-phase top-k (dsv41_topk_chunk, then dsv41_topk_merge) picks what
// the one-threadgroup one picks, to the index: random rows, ties at the
// cut, -inf entries, fewer finite than K, every value equal, the top values
// all in one chunk; a partial last chunk; the merge's candidates both
// gathered (under 3072 above its bound) and read where they lie -- each
// path counted from the chunks' own outputs.
TEST(deepseek_v41, two_phase_topk_picks_the_same)
{
  using metal_compute::CommandStream;
  using metal_compute::ComputeEncoder;
  using metal_compute::ComputeFunction;
  using metal_compute::SharedBuffer;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("deepseek_v41_bf16");
  ComputeFunction fp = lib.function("dsv41_topk_par");
  ComputeFunction fc = lib.function("dsv41_topk_chunk");
  ComputeFunction fm = lib.function("dsv41_topk_merge");
  ASSERT_TRUE(fp.valid() && fc.valid() && fm.valid());
  if (!fp.valid() || !fc.valid() || !fm.valid()) { return; }
  const unsigned NT = 256;
  auto key = [](float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return (b & 0x80000000u) != 0 ? ~b : (b | 0x80000000u);
  };
  const std::uint32_t fin = key(-INFINITY) + 1u;
  constexpr int CH = 4096;
  int checked = 0, differ = 0, gathered = 0, in_place = 0;
  for (auto [C, K] : std::vector<std::pair<int, int>>{
           {16384, 512}, {16384, 2048}, {20000, 512}, {65536, 512},
           {131073, 512}, {131072, 2048}}) {
    const int G = (C + CH - 1) / CH;
    const int R = std::clamp(3 * K / 2 / G, 1, std::min(K, 3072 / G));
    for (int kind = 0; kind < 6; ++kind) {
      const int n = 2;
      SharedBuffer S = mc->make_shared_buffer((std::size_t)n * C * 4);
      auto* v = static_cast<float*>(S.contents());
      std::uint32_t x = 5u + (std::uint32_t)(C * 7 + kind * 3 + K);
      for (int i = 0; i < n * C; ++i) {
        x = x * 1664525u + 1013904223u;
        float f = ((float)(x >> 8) / 16777216.0f) * 4.0f - 1.0f;
        const int t = i % C;
        if (kind == 1) { f = (float)((x >> 8) % 37) * 0.25f; }   // ties
        if (kind == 2 && (x >> 7) % 3 != 0) { f = -INFINITY; }   // masked
        if (kind == 3 && t > K / 2) { f = -INFINITY; }           // few
        if (kind == 4) { f = 0.5f; }                             // equal
        if (kind == 5) {                                         // one chunk
          f = t >= C / 2 && t < C / 2 + K + 100 ? f + 9.0f : f - 9.0f;
        }
        v[i] = f;
      }
      SharedBuffer o1 = mc->make_shared_buffer((std::size_t)n * K * 4);
      SharedBuffer o2 = mc->make_shared_buffer((std::size_t)n * K * 4);
      SharedBuffer ck = mc->make_shared_buffer((std::size_t)n * G * K * 4);
      SharedBuffer cp = mc->make_shared_buffer((std::size_t)n * G * K * 4);
      SharedBuffer tk = mc->make_shared_buffer((std::size_t)n * G * 4);
      SharedBuffer tr = mc->make_shared_buffer((std::size_t)n * G * R * 4);
      CommandStream st = mc->make_command_stream();
      ComputeEncoder enc = st.begin_compute();
      enc.set_function(fp);
      enc.set_buffer(0, S);
      enc.set_buffer(1, o1);
      enc.set_constant(2, C);
      enc.set_constant(3, K);
      enc.dispatch({(unsigned)(1024 * n), 1, 1}, {1024, 1, 1});
      enc.set_function(fc);
      enc.set_buffer(0, S);
      enc.set_buffer(1, ck);
      enc.set_buffer(2, cp);
      enc.set_buffer(3, tk);
      enc.set_buffer(4, tr);
      enc.set_constant(5, C);
      enc.set_constant(6, K);
      enc.set_constant(7, R);
      enc.dispatch({NT * (unsigned)G, (unsigned)n, 1}, {NT, 1, 1});
      enc.set_function(fm);
      enc.set_buffer(0, ck);
      enc.set_buffer(1, cp);
      enc.set_buffer(2, tk);
      enc.set_buffer(3, tr);
      enc.set_buffer(4, o2);
      enc.set_constant(5, C);
      enc.set_constant(6, K);
      enc.set_constant(7, R);
      enc.dispatch({NT * (unsigned)n, 1, 1}, {NT, 1, 1});
      enc.end();
      std::string why;
      ASSERT_TRUE(st.commit().wait_ok(&why));
      ++checked;
      differ += std::memcmp(o1.contents(), o2.contents(),
                            (std::size_t)n * K * 4) != 0;
      // which way the merge went, from what the chunks wrote
      const auto* pk = static_cast<const std::uint32_t*>(ck.contents());
      const auto* pt = static_cast<const std::uint32_t*>(tk.contents());
      const auto* pr = static_cast<const std::uint32_t*>(tr.contents());
      for (int i = 0; i < n; ++i) {
        std::uint32_t lo = fin;
        for (int g = 0; g < G; ++g) { lo = std::max(lo, pt[i * G + g]); }
        std::vector<std::uint32_t> u;
        for (int j = 0; j < G * R; ++j) {
          if (pr[(std::size_t)i * G * R + j] >= fin) {
            u.push_back(pr[(std::size_t)i * G * R + j]);
          }
        }
        if ((int)u.size() > K) {
          std::nth_element(u.begin(), u.begin() + (K - 1), u.end(),
                           std::greater<>());
          lo = std::max(lo, u[(std::size_t)K - 1]);
        }
        int total = 0;
        for (int j = 0; j < G * K; ++j) {
          total += pk[(std::size_t)i * G * K + j] >= lo;
        }
        (total <= 3072 ? gathered : in_place) += 1;
      }
    }
  }
  std::printf("[dsv41] two-phase top-k: %d of %d cases differ; rows "
              "gathered %d, read in place %d\n", differ, checked, gathered,
              in_place);
  EXPECT_TRUE(differ == 0);
  EXPECT_TRUE(gathered > 0 && in_place > 0);
}

// The indexer's scores on simdgroup matrices against the scalar kernel's:
// the same values to f32 rounding (only the order of each dot's sum moves),
// -inf at the same positions -- a query's visibility edge inside a tile,
// a candidate source's dropped blocks, the last partial tile.
TEST(deepseek_v41, index_scores_on_matrices_match)
{
  using metal_compute::CommandStream;
  using metal_compute::ComputeEncoder;
  using metal_compute::ComputeFunction;
  using metal_compute::SharedBuffer;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("deepseek_v41_bf16");
  ComputeFunction f0 = lib.function("dsv41_index_score");
  ComputeFunction f1 = lib.function("dsv41_index_score_mma");
  ASSERT_TRUE(f0.valid() && f1.valid());
  if (!f0.valid() || !f1.valid()) { return; }
  // and the matrix cores' (4 rows a key tile), where there are some
  ComputeFunction f2;
  if (mc->supports_matrix_cores()) {
    f2 = mc->load_library("deepseek_v41_nax_bf16")
             .function("dsv41_index_score_nax");
    ASSERT_TRUE(f2.valid());
  }
  const int nk = f2.valid() ? 3 : 2;
  for (const int IH : {32, 16}) {
  const int ID = 128, n = 7, C = 4100, blk = 8;
  const int nblk = (C + blk - 1) / blk;
  auto bf = [&](std::size_t count, std::uint32_t seed) {
    SharedBuffer b = mc->make_shared_buffer(count * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    std::uint32_t x = seed;
    for (std::size_t i = 0; i < count; ++i) {
      x = x * 1664525u + 1013904223u;
      const float v = ((float)(x >> 8) / 16777216.0f - 0.5f) * 2.0f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      p[i] = (std::uint16_t)(u >> 16);
    }
    return b;
  };
  SharedBuffer q = bf((std::size_t)n * IH * ID, 1), keys =
      bf((std::size_t)C * ID, 2), w = bf((std::size_t)n * IH, 3);
  SharedBuffer cand = mc->make_shared_buffer((std::size_t)n * nblk);
  auto* cp = static_cast<std::uint8_t*>(cand.contents());
  for (int i = 0; i < n * nblk; ++i) { cp[i] = (i * 7 + i / 5) % 3 == 0; }
  double worst = 0.0, worst2 = 0.0;
  int inf_differ = 0;
  int tail_hit = 0;
  for (int use_cand = 0; use_cand < 2; ++use_cand) {
    // a tail past the last row that nothing may write
    const std::size_t NS = (std::size_t)n * C + 64;
    SharedBuffer S[3] = {mc->make_shared_buffer(NS * 4),
                         mc->make_shared_buffer(NS * 4),
                         mc->make_shared_buffer(NS * 4)};
    for (SharedBuffer& b : S) {
      auto* p = static_cast<float*>(b.contents());
      for (std::size_t k = (std::size_t)n * C; k < NS; ++k) { p[k] = 7.0f; }
    }
    for (int k = 0; k < nk; ++k) {
      CommandStream st = mc->make_command_stream();
      ComputeEncoder enc = st.begin_compute();
      enc.set_function(k == 0 ? f0 : k == 1 ? f1 : f2);
      enc.set_buffer(0, q);
      enc.set_buffer(1, keys);
      enc.set_buffer(2, w);
      enc.set_buffer(3, cand);
      enc.set_buffer(4, S[k]);
      enc.set_constant(5, IH);
      enc.set_constant(6, C);
      // the rows see 4095 .. 4100 of the 4100 positions (ratio 2): the
      // edge inside the last tile, partial and live in every row
      enc.set_constant(7, 8189);
      enc.set_constant(8, 2);
      enc.set_constant(9, 0.01f);
      enc.set_constant(10, use_cand);
      enc.set_constant(11, blk);
      enc.set_constant(12, nblk);
      if (k == 0) {
        enc.dispatch({(unsigned)(256 * ((C + 255) / 256)), (unsigned)n, 1},
                     {256, 1, 1});
      } else if (k == 2) {
        enc.set_constant(13, n);
        enc.dispatch({(unsigned)(256 * ((C + 63) / 64)),
                      (unsigned)((n + 3) / 4), 1}, {256, 1, 1});
      } else {
        enc.dispatch({(unsigned)(128 * ((C + 63) / 64)), (unsigned)n, 1},
                     {128, 1, 1});
      }
      enc.end();
      std::string why;
      ASSERT_TRUE(st.commit().wait_ok(&why));
    }
    const auto* a = static_cast<const float*>(S[0].contents());
    const auto* b = static_cast<const float*>(S[1].contents());
    const auto* c2 = static_cast<const float*>(S[2].contents());
    double mx = 0.0;
    for (int k = 0; k < n * C; ++k) {
      if (std::isfinite(a[k])) { mx = std::max(mx, (double)std::fabs(a[k])); }
    }
    for (std::size_t k = (std::size_t)n * C; k < NS; ++k) {
      tail_hit += (a[k] != 7.0f) + (b[k] != 7.0f) +
                  (nk > 2 && c2[k] != 7.0f);
    }
    for (int k = 0; k < n * C; ++k) {
      if (std::isfinite(a[k]) != std::isfinite(b[k])) {
        ++inf_differ;
      } else if (std::isfinite(a[k])) {
        worst = std::max(worst, std::fabs((double)a[k] - b[k]) / mx);
      }
      if (nk < 3) { continue; }
      if (std::isfinite(a[k]) != std::isfinite(c2[k])) {
        ++inf_differ;
      } else if (std::isfinite(a[k])) {
        worst2 = std::max(worst2, std::fabs((double)a[k] - c2[k]) / mx);
      }
    }
  }
  std::printf("[dsv41] index scores, %d heads: simdgroup matrices %.2e of the "
              "largest apart, matrix cores %.2e%s, %d -inf differ, %d "
              "written past the rows\n", IH, worst, worst2,
              nk > 2 ? "" : " (none here)", inf_differ, tail_hit);
  EXPECT_TRUE(inf_differ == 0);
  EXPECT_TRUE(tail_hit == 0);
  EXPECT_TRUE(worst < 1e-5 && worst2 < 1e-5);
  }
  if (env_("VPIPE_DSV41_KERNEL_BENCH").empty() || !f2.valid()) { return; }
  // 2048 rows deep in a long context: 32K positions at ratio 2 (64K
  // tokens), every one visible
  {
    const int IH = 32, ID = 128, n = 2048, C = 32768, blk = 8;
    const int nblk = (C + blk - 1) / blk;
    auto bf = [&](std::size_t count, std::uint32_t seed) {
      SharedBuffer b = mc->make_shared_buffer(count * 2);
      auto* p = static_cast<std::uint16_t*>(b.contents());
      std::uint32_t x = seed;
      for (std::size_t i = 0; i < count; ++i) {
        x = x * 1664525u + 1013904223u;
        const float v = ((float)(x >> 8) / 16777216.0f - 0.5f) * 2.0f;
        std::uint32_t u;
        std::memcpy(&u, &v, 4);
        p[i] = (std::uint16_t)(u >> 16);
      }
      return b;
    };
    SharedBuffer q = bf((std::size_t)n * IH * ID, 5), keys =
        bf((std::size_t)C * ID, 6), w = bf((std::size_t)n * IH, 7);
    SharedBuffer cand = mc->make_shared_buffer((std::size_t)n * nblk);
    SharedBuffer S = mc->make_shared_buffer((std::size_t)n * C * 4);
    const double flop = 2.0 * n * (double)C * IH * ID;
    for (int round = 0; round < 3; ++round) {
      double ms[2];
      for (int k = 0; k < 2; ++k) {
        CommandStream st = mc->make_command_stream();
        ComputeEncoder enc = st.begin_compute();
        for (int rep = 0; rep < 3; ++rep) {
          enc.set_function(k == 0 ? f1 : f2);
          enc.set_buffer(0, q);
          enc.set_buffer(1, keys);
          enc.set_buffer(2, w);
          enc.set_buffer(3, cand);
          enc.set_buffer(4, S);
          enc.set_constant(5, IH);
          enc.set_constant(6, C);
          enc.set_constant(7, 2 * C + 1);
          enc.set_constant(8, 2);
          enc.set_constant(9, 0.01f);
          enc.set_constant(10, 0);
          enc.set_constant(11, blk);
          enc.set_constant(12, nblk);
          if (k == 1) {
            enc.set_constant(13, n);
            enc.dispatch({(unsigned)(256 * ((C + 63) / 64)),
                          (unsigned)((n + 3) / 4), 1}, {256, 1, 1});
          } else {
            enc.dispatch({(unsigned)(128 * ((C + 63) / 64)), (unsigned)n, 1},
                         {128, 1, 1});
          }
        }
        enc.end();
        auto fe = st.commit();
        std::string why;
        ASSERT_TRUE(fe.wait_ok(&why));
        ms[k] = fe.gpu_times().gpu_s / 3.0 * 1e3;
      }
      std::printf("[dsv41] index scores, 2048 rows x 32768 positions: "
                  "simdgroup matrices %.1f ms (%.1f TFLOP/s), matrix cores "
                  "%.1f ms (%.1f TFLOP/s)\n", ms[0], flop / ms[0] / 1e9,
                  ms[1], flop / ms[1] / 1e9);
    }
  }
}

// The prefill attention on the matrix cores (dsv41_attn_nax: a query's
// heads as the products' rows) against a simdgroup a head (dsv41_attn), on
// random latents at the release's shape: 64 heads of 512, a window of 128
// over the ring and this chunk, 512 picks (some -1, some past what a row
// sees), the sinks, the rotary tail turned back. Every shape the model
// sends it: a window alone, compressed at ratio 2 and 1, a CED decoder's
// tail (the window from its start), 8 heads (padded to 32). Matrix cores
// only. VPIPE_DSV41_KERNEL_BENCH=1 times both at 2048 rows.
TEST(deepseek_v41, attention_on_matrices_matches)
{
  using metal_compute::CommandStream;
  using metal_compute::ComputeEncoder;
  using metal_compute::ComputeFunction;
  using metal_compute::SharedBuffer;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  if (!mc->supports_matrix_cores()) {
    std::printf("[dsv41] attention on matrices: no matrix cores, skipped\n");
    return;
  }
  for (const bool bfe : {true, false}) {
  const std::string sx = bfe ? "_bf16" : "";
  auto lib = mc->load_library("deepseek_v41" + sx);
  auto lnx = mc->load_library("deepseek_v41_nax" + sx);
  ComputeFunction f0 = lib.function("dsv41_attn");
  ComputeFunction f1 = lnx.function("dsv41_attn_nax");
  ASSERT_TRUE(f0.valid() && f1.valid());
  if (!f0.valid() || !f1.valid()) { return; }
  constexpr int D = 512, rd = 64, W = 128;
  std::uint32_t x = 777u;
  auto rnd = [&x] {
    x = x * 1664525u + 1013904223u;
    return ((float)(x >> 8) / 16777216.0f) * 2.0f - 1.0f;
  };
  auto bf16 = [&](std::size_t count, float scale) {
    SharedBuffer b = mc->make_shared_buffer(count * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t k = 0; k < count; ++k) {
      const float v = rnd() * scale;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      if (bfe) {
        p[k] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
      } else {
        const _Float16 hv = (_Float16)v;
        std::memcpy(&p[k], &hv, 2);
      }
    }
    return b;
  };
  auto fl = [bfe](std::uint16_t b) {
    if (!bfe) {
      _Float16 hv;
      std::memcpy(&hv, &b, 2);
      return (float)hv;
    }
    const std::uint32_t u = (std::uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  struct Case {
    const char* name;
    int NH, n, p0, KC, ratio, lo;
  };
  const Case cases[] = {
      {"window alone", 64, 37, 200, 0, 1, 0},
      {"ratio 2, 512 picks", 64, 45, 1500, 512, 2, 0},
      {"ratio 1, 512 picks", 64, 40, 3000, 512, 1, 0},
      {"a CED tail", 64, 50, 3000, 512, 1, 3000},
      {"8 heads", 8, 33, 900, 512, 2, 0},
  };
  const int R = W + 8;
  bool all_ok = true;
  auto run = [&](const ComputeFunction& f, bool nax, const Case& cs,
                 const SharedBuffer& q, const SharedBuffer& kvn,
                 const SharedBuffer& ring, const SharedBuffer& comp,
                 const SharedBuffer& idx, const SharedBuffer& sink,
                 const SharedBuffer& tab, const SharedBuffer& o, int reps) {
    CommandStream st = mc->make_command_stream();
    ComputeEncoder enc = st.begin_compute();
    for (int r = 0; r < reps; ++r) {
      enc.set_function(f);
      enc.set_buffer(0, q);
      enc.set_buffer(1, kvn);
      enc.set_buffer(2, ring);
      enc.set_buffer(3, comp);
      enc.set_buffer(4, idx);
      enc.set_buffer(5, sink);
      enc.set_buffer(6, o);
      enc.set_buffer(7, tab);
      enc.set_constant(8, cs.n);
      enc.set_constant(9, cs.NH);
      enc.set_constant(10, W);
      enc.set_constant(11, cs.p0);
      enc.set_constant(12, cs.KC);
      enc.set_constant(13, cs.ratio);
      enc.set_constant(14, (float)(1.0 / std::sqrt((double)D)));
      enc.set_constant(15, std::min(8, cs.NH));
      enc.set_constant(16, rd);
      enc.set_constant(17, R);
      enc.set_constant(18, cs.lo);
      if (nax) { enc.set_constant(19, 1); }
      const int hpt = nax ? 32 : 8;
      enc.dispatch({(unsigned)(256 * ((cs.NH + hpt - 1) / hpt)),
                    (unsigned)cs.n, 1}, {256, 1, 1});
    }
    enc.end();
    std::string why;
    auto fe = st.commit();
    const bool ok = fe.wait_ok(&why);
    EXPECT_TRUE(ok);
    return fe.gpu_times().gpu_s;
  };
  auto make = [&](const Case& cs, SharedBuffer* bufs) {
    const int end = cs.p0 + cs.n;
    const int C = std::max(1, (end + 1) / std::max(cs.ratio, 1) + 1);
    bufs[0] = bf16((std::size_t)cs.n * cs.NH * D, 1.0f);       // q
    bufs[1] = bf16((std::size_t)cs.n * D, 1.0f);               // kvn
    bufs[2] = bf16((std::size_t)R * D, 1.0f);                  // ring
    bufs[3] = bf16((std::size_t)C * D, 1.0f);                  // comp
    bufs[4] = mc->make_shared_buffer(
        (std::size_t)std::max(cs.KC, 1) * cs.n * 4);           // idx
    auto* ip = static_cast<std::int32_t*>(bufs[4].contents());
    for (int k = 0; k < cs.KC * cs.n; ++k) {
      const float u = 0.5f * (rnd() + 1.0f);
      ip[k] = u < 0.05f ? -1 : (int)(u * (float)(C + 8)) - 4;
    }
    bufs[5] = mc->make_shared_buffer((std::size_t)cs.NH * 4);  // sink
    auto* sp = static_cast<float*>(bufs[5].contents());
    for (int h = 0; h < cs.NH; ++h) { sp[h] = rnd(); }
    const std::size_t nt = (std::size_t)(end + 1) * (rd / 2);
    bufs[6] = mc->make_shared_buffer(nt * 8);                  // tab
    auto* tp = static_cast<float*>(bufs[6].contents());
    for (std::size_t k = 0; k < nt; ++k) {
      const float a = 3.14159265f * rnd();
      tp[2 * k] = std::cos(a);
      tp[2 * k + 1] = std::sin(a);
    }
    bufs[7] = mc->make_shared_buffer((std::size_t)cs.n * cs.NH * D * 2);
    bufs[8] = mc->make_shared_buffer((std::size_t)cs.n * cs.NH * D * 2);
  };
  // both kernels against f64 on a case's first rows: the keys each sees,
  // the softmax with its sink, the tail turned back
  auto exact = [&](const Case& cs, SharedBuffer* b, int rows,
                   double* e0, double* e1) {
    const auto* q = static_cast<const std::uint16_t*>(b[0].contents());
    const auto* kv = static_cast<const std::uint16_t*>(b[1].contents());
    const auto* rg = static_cast<const std::uint16_t*>(b[2].contents());
    const auto* cp = static_cast<const std::uint16_t*>(b[3].contents());
    const auto* ip = static_cast<const std::int32_t*>(b[4].contents());
    const auto* sp = static_cast<const float*>(b[5].contents());
    const auto* tp = static_cast<const float*>(b[6].contents());
    const auto* o0 = static_cast<const std::uint16_t*>(b[7].contents());
    const auto* o1 = static_cast<const std::uint16_t*>(b[8].contents());
    const int G = std::min(8, cs.NH), hg = cs.NH / G;
    double n0 = 0.0, n1 = 0.0, den = 0.0;
    std::vector<double> o(D);
    for (int i = 0; i < rows; ++i) {
      const int pos = cs.p0 + i;
      const int wlo = std::max(std::max(0, cs.lo), pos - W + 1);
      const int vis = (pos + 1) / std::max(cs.ratio, 1);
      std::vector<const std::uint16_t*> keys;
      for (int p = wlo; p <= pos; ++p) {
        keys.push_back(p >= cs.p0 ? kv + (std::size_t)(p - cs.p0) * D
                                  : rg + (std::size_t)(p % R) * D);
      }
      for (int j = 0; j < cs.KC; ++j) {
        const int t = ip[(std::size_t)i * cs.KC + j];
        if (t >= 0 && t < vis) { keys.push_back(cp + (std::size_t)t * D); }
      }
      for (int h = 0; h < cs.NH; ++h) {
        const std::uint16_t* qr = q + ((std::size_t)i * cs.NH + h) * D;
        std::vector<double> sc(keys.size());
        double m = sp[h];
        for (std::size_t j = 0; j < keys.size(); ++j) {
          double d = 0.0;
          for (int e = 0; e < D; ++e) {
            d += (double)fl(qr[e]) * (double)fl(keys[j][e]);
          }
          sc[j] = d / std::sqrt((double)D);
          m = std::max(m, sc[j]);
        }
        double l = std::exp((double)sp[h] - m);
        std::fill(o.begin(), o.end(), 0.0);
        for (std::size_t j = 0; j < keys.size(); ++j) {
          const double w = std::exp(sc[j] - m);
          l += w;
          for (int e = 0; e < D; ++e) { o[e] += w * (double)fl(keys[j][e]); }
        }
        for (int e = 0; e < D; ++e) { o[e] /= l; }
        for (int e = D - rd; e < D; e += 2) {
          const std::size_t k = (std::size_t)pos * (rd / 2) +
                                (std::size_t)(e - (D - rd)) / 2;
          const double c = tp[2 * k], sn = tp[2 * k + 1];
          const double re = o[e], im = o[e + 1];
          o[e] = re * c + im * sn;
          o[e + 1] = -re * sn + im * c;
        }
        const std::size_t at = ((std::size_t)(h / hg) * cs.n + i) *
                                   (std::size_t)(hg * D) +
                               (std::size_t)(h % hg) * D;
        for (int e = 0; e < D; ++e) {
          n0 += std::pow((double)fl(o0[at + e]) - o[e], 2);
          n1 += std::pow((double)fl(o1[at + e]) - o[e], 2);
          den += o[e] * o[e];
        }
      }
    }
    *e0 = std::sqrt(n0 / den);
    *e1 = std::sqrt(n1 / den);
  };
  for (const Case& cs : cases) {
    SharedBuffer b[9];
    make(cs, b);
    run(f0, false, cs, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1);
    run(f1, true, cs, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[8], 1);
    const auto* o0 = static_cast<const std::uint16_t*>(b[7].contents());
    const auto* o1 = static_cast<const std::uint16_t*>(b[8].contents());
    const std::size_t N = (std::size_t)cs.n * cs.NH * D;
    double num = 0.0, den = 0.0, worst = 0.0;
    int nonfinite = 0;
    for (std::size_t k = 0; k < N; ++k) {
      const double a = fl(o0[k]), c = fl(o1[k]);
      if (!std::isfinite(c)) { ++nonfinite; continue; }
      num += (a - c) * (a - c);
      den += a * a;
      worst = std::max(worst, std::fabs(a - c));
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    double e0 = 0.0, e1 = 0.0;
    exact(cs, b, 4, &e0, &e1);
    const bool ok = nonfinite == 0 && rel < 4e-3 && e1 <= 1.25 * e0 + 1e-4;
    all_ok = all_ok && ok;
    std::printf("[dsv41] attention on matrices, %s, %s: rel-L2 %.2e from a "
                "simdgroup a head, worst %.2e, %d non-finite; from f64 "
                "(4 rows) %.2e against its %.2e\n", bfe ? "bf16" : "f16",
                cs.name, rel, worst, nonfinite, e1, e0);
  }
  EXPECT_TRUE(all_ok);
  if (env_("VPIPE_DSV41_KERNEL_BENCH").empty() || !bfe) { continue; }
  // a CED piece's encoder layer: 2048 rows deep in a long context
  const Case big = {"bench", 64, 2048, 16384, 512, 2, 0};
  SharedBuffer b[9];
  make(big, b);
  const double flop = 4.0 * big.NH * (W + big.KC) * D * (double)big.n;
  for (int round = 0; round < 3; ++round) {
    const double t0 =
        run(f0, false, big, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            4) / 4.0;
    const double t1 =
        run(f1, true, big, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[8],
            4) / 4.0;
    std::printf("[dsv41] attention, 2048 rows x 64 heads x 640 keys: a "
                "simdgroup a head %.2f ms (%.1f TFLOP/s), matrices %.2f ms "
                "(%.1f TFLOP/s)\n", t0 * 1e3, flop / t0 / 1e12, t1 * 1e3,
                flop / t1 / 1e12);
  }
  }
}

// A layer past the candidate source scoring only its kept blocks'
// positions (dsv41_index_score_kept, compacted, then dsv41_index_unpack)
// picks what scoring every position under the blocks' mask picks, to the
// index: blocks picked ascending with -1 pads, a row whose kept positions
// are fewer than K, a visibility edge inside the last block, slots a
// partial last tile (800), nothing written past them.
TEST(deepseek_v41, kept_positions_pick_the_same)
{
  using metal_compute::CommandStream;
  using metal_compute::ComputeEncoder;
  using metal_compute::ComputeFunction;
  using metal_compute::SharedBuffer;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("deepseek_v41_bf16");
  ComputeFunction fm = lib.function("dsv41_index_score_mma");
  ComputeFunction fk = lib.function("dsv41_index_score_kept");
  ComputeFunction fu = lib.function("dsv41_index_unpack");
  ComputeFunction ft = lib.function("dsv41_topk_par");
  ASSERT_TRUE(fm.valid() && fk.valid() && fu.valid() && ft.valid());
  if (!fm.valid() || !fk.valid() || !fu.valid() || !ft.valid()) { return; }
  const int IH = 32, ID = 128, n = 3, C = 4100, blk = 8, KB = 100;
  const int K = 512, nblk = (C + blk - 1) / blk, p0 = 4000;
  auto bf = [&](std::size_t count, std::uint32_t seed) {
    SharedBuffer b = mc->make_shared_buffer(count * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    std::uint32_t x = seed;
    for (std::size_t i = 0; i < count; ++i) {
      x = x * 1664525u + 1013904223u;
      const float v = ((float)(x >> 8) / 16777216.0f - 0.5f) * 2.0f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      p[i] = (std::uint16_t)(u >> 16);
    }
    return b;
  };
  SharedBuffer q = bf((std::size_t)n * IH * ID, 11), keys =
      bf((std::size_t)C * ID, 12), w = bf((std::size_t)n * IH, 13);
  // each row's blocks ascending, the last visible one among them; row 0
  // keeps 40 (313 positions, under K), the last row's last tile partial
  SharedBuffer bpick = mc->make_shared_buffer((std::size_t)n * KB * 4);
  SharedBuffer cand = mc->make_shared_buffer((std::size_t)n * nblk);
  auto* bp = static_cast<std::int32_t*>(bpick.contents());
  auto* cp = static_cast<std::uint8_t*>(cand.contents());
  std::memset(cp, 0, (std::size_t)n * nblk);
  for (int i = 0; i < n; ++i) {
    const int want = i == 0 ? 40 : KB;
    const int last = (p0 + i) / blk;
    std::vector<int> bs;
    std::uint32_t x = 99u + (std::uint32_t)i;
    while ((int)bs.size() < want - 1) {
      x = x * 1664525u + 1013904223u;
      const int b = (int)((x >> 8) % (std::uint32_t)last);
      if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); }
    }
    bs.push_back(last);
    std::sort(bs.begin(), bs.end());
    for (int k = 0; k < KB; ++k) {
      bp[(std::size_t)i * KB + k] = k < (int)bs.size() ? bs[(std::size_t)k]
                                                       : -1;
    }
    for (int b : bs) { cp[(std::size_t)i * nblk + b] = 1; }
  }
  const float ws = 0.01f;
  const std::size_t NS = (std::size_t)n * C + 64;
  SharedBuffer S = mc->make_shared_buffer(NS * 4);
  SharedBuffer idx[2] = {mc->make_shared_buffer((std::size_t)n * K * 4),
                         mc->make_shared_buffer((std::size_t)n * K * 4)};
  int tail_hit = 0;
  for (int arm = 0; arm < 2; ++arm) {
    const int R = arm == 0 ? C : KB * blk;
    auto* sp = static_cast<float*>(S.contents());
    for (std::size_t k = (std::size_t)n * R; k < NS; ++k) { sp[k] = 7.0f; }
    CommandStream st = mc->make_command_stream();
    ComputeEncoder enc = st.begin_compute();
    enc.set_function(arm == 0 ? fm : fk);
    enc.set_buffer(0, q);
    enc.set_buffer(1, keys);
    enc.set_buffer(2, w);
    enc.set_buffer(3, arm == 0 ? cand : bpick);
    enc.set_buffer(4, S);
    enc.set_constant(5, IH);
    enc.set_constant(6, C);
    enc.set_constant(7, p0);
    enc.set_constant(8, 1);
    enc.set_constant(9, ws);
    if (arm == 0) {
      enc.set_constant(10, 1);
      enc.set_constant(11, blk);
      enc.set_constant(12, nblk);
    } else {
      enc.set_constant(10, KB);
      enc.set_constant(11, blk);
    }
    enc.dispatch({(unsigned)(128 * ((R + 63) / 64)), (unsigned)n, 1},
                 {128, 1, 1});
    enc.set_function(ft);
    enc.set_buffer(0, S);
    enc.set_buffer(1, idx[arm]);
    enc.set_constant(2, R);
    enc.set_constant(3, K);
    enc.dispatch({(unsigned)(1024 * n), 1, 1}, {1024, 1, 1});
    if (arm == 1) {
      enc.set_function(fu);
      enc.set_buffer(0, idx[arm]);
      enc.set_buffer(1, bpick);
      enc.set_constant(2, K);
      enc.set_constant(3, KB);
      enc.set_constant(4, blk);
      enc.dispatch({(unsigned)K, (unsigned)n, 1}, {64, 1, 1});
    }
    enc.end();
    std::string why;
    ASSERT_TRUE(st.commit().wait_ok(&why));
    for (std::size_t k = (std::size_t)n * R; k < NS; ++k) {
      tail_hit += sp[k] != 7.0f;
    }
  }
  const auto* a = static_cast<const std::int32_t*>(idx[0].contents());
  const auto* b = static_cast<const std::int32_t*>(idx[1].contents());
  int differ = 0, picked = 0;
  for (int k = 0; k < n * K; ++k) {
    differ += a[k] != b[k];
    picked += a[k] >= 0;
  }
  std::printf("[dsv41] kept positions: %d of %d picks differ (%d picked), "
              "%d written past the slots\n", differ, n * K, picked,
              tail_hit);
  EXPECT_TRUE(differ == 0);
  EXPECT_TRUE(tail_hit == 0);
  EXPECT_TRUE(picked == 2 * K + 40 * blk - (blk - 1 - p0 % blk));
}

// The indexer's faster paths change nothing, to the bit: a layer past the
// candidate source scoring only the blocks it kept (VPIPE_DSV41_INDEX_KEPT)
// and a few rows' top-k in two phases (VPIPE_DSV41_TOPK_TWO, taken here at
// every width: on its own it waits for 32K positions), against every
// position scored under the mask on one threadgroup -- a prefill, decode
// steps and a verify of 6. The small model's 80-token prompt clears the
// kept path's floor (more blocks than the source keeps: 10 blocks of 8
// positions against its 4).
TEST(deepseek_v41, indexer_paths_change_nothing)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  const std::vector<std::int32_t> dec = ints_(meta, "decode_ids");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr || dec.size() < 6) { return; }
  // the last arm's scores 8 rows at a time (a prefill's 10 chunks)
  const char* arms[4][2] = {{"0", "0"}, {"1", "0"}, {"1", "2"}, {"1", "0"}};
  std::vector<float> out[4];
  std::uint64_t kept[4] = {0, 0, 0, 0}, two[4] = {0, 0, 0, 0};
  long chunks[4] = {0, 0, 0, 0};
  // (the f32 simdgroup scorer: the matrix cores' sums its heads in
  // another order -- below, against these)
  ::setenv("VPIPE_DSV41_INDEX_NAX", "0", 1);
  for (int arm = 0; arm < 4; ++arm) {
    ::setenv("VPIPE_DSV41_INDEX_KEPT", arms[arm][0], 1);
    ::setenv("VPIPE_DSV41_TOPK_TWO", arms[arm][1], 1);
    std::string err;
    MetalDeepseekV41Model::Options opt;
    opt.score_rows = arm == 3 ? 8 : 0;
    auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                         mc, opt, &err);
    ::unsetenv("VPIPE_DSV41_INDEX_KEPT");
    ::unsetenv("VPIPE_DSV41_TOPK_TWO");
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    std::vector<float> lg;
    ASSERT_TRUE(m->forward(1, ids, &lg, /*all_rows=*/true, &err));
    out[arm].insert(out[arm].end(), lg.begin(), lg.end());
    for (std::size_t i = 0; i < 4; ++i) {
      ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>{dec[i]}, &lg, false,
                             &err));
      out[arm].insert(out[arm].end(), lg.begin(), lg.end());
    }
    const std::vector<std::int32_t> v6(dec.begin(), dec.begin() + 6);
    ASSERT_TRUE(m->forward(1, v6, &lg, /*all_rows=*/true, &err));
    out[arm].insert(out[arm].end(), lg.begin(), lg.end());
    kept[arm] = m->index_kept_passes();
    two[arm] = m->topk_two_passes();
    chunks[arm] = m->score_chunks();
  }
  for (int arm = 1; arm < 4; ++arm) {
    const bool same = out[0].size() == out[arm].size() &&
                      std::memcmp(out[0].data(), out[arm].data(),
                                  out[0].size() * 4) == 0;
    std::printf("[dsv41] kept %s, two-phase %s: %llu kept passes, %llu "
                "two-phase, %ld more score chunks, %zu logits %s\n",
                arms[arm][0], arms[arm][1], (unsigned long long)kept[arm],
                (unsigned long long)two[arm], chunks[arm], out[0].size(),
                same ? "identical" : "DIFFER");
    EXPECT_TRUE(same);
  }
  EXPECT_TRUE(kept[0] == 0 && kept[1] > 0 && kept[2] > 0);
  EXPECT_TRUE(two[0] == 0 && two[1] == 0 && two[2] > 0);
  EXPECT_TRUE(chunks[0] == 0 && chunks[1] == 0 && chunks[3] >= 9);
  ::unsetenv("VPIPE_DSV41_INDEX_NAX");
  // the matrix cores' dense scores: the same arithmetic, other sums, so a
  // near tie in a top-k may go the other way -- held to the logits
  if (!mc->supports_matrix_cores()) { return; }
  std::string err;
  MetalDeepseekV41Model::Options opt;
  auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                       mc, opt, &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  std::vector<float> nx, lg;
  ASSERT_TRUE(m->forward(1, ids, &lg, /*all_rows=*/true, &err));
  nx.insert(nx.end(), lg.begin(), lg.end());
  const long passes = m->nax_index_passes();
  const double d = nx.size() <= out[1].size()
                       ? rel_l2_(nx.data(), out[1].data(), nx.size())
                       : 1e9;
  std::printf("[dsv41] the indexer's scores on the matrix cores: %ld "
              "passes, the prefill's logits %.2e from the f32 simdgroup "
              "scorer's\n", passes, d);
  EXPECT_TRUE(passes > 0);
  EXPECT_TRUE(d < 0.05);
}

// What only reorders or merges the same arithmetic changes nothing, to
// the bit: a decode or verify step's independent dispatches in concurrent
// groups (VPIPE_DSV41_CONCURRENT=1) and a sublayer's post taken with the
// next collapse (VPIPE_DSV41_HC_FUSE), each and both, against the serial
// encoder running them apart -- a prefill, decode steps and a verify of 6.
TEST(deepseek_v41, concurrent_groups_change_nothing)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  const std::vector<std::int32_t> dec = ints_(meta, "decode_ids");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr || dec.size() < 6) { return; }
  const char* arms[4][2] = {{"0", "0"}, {"1", "0"}, {"0", "1"}, {"1", "1"}};
  std::vector<float> out[4];
  for (int arm = 0; arm < 4; ++arm) {
    ::setenv("VPIPE_DSV41_CONCURRENT", arms[arm][0], 1);
    ::setenv("VPIPE_DSV41_HC_FUSE", arms[arm][1], 1);
    std::string err;
    MetalDeepseekV41Model::Options opt;
    auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                         mc, opt, &err);
    ::unsetenv("VPIPE_DSV41_CONCURRENT");
    ::unsetenv("VPIPE_DSV41_HC_FUSE");
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    std::vector<float> lg;
    ASSERT_TRUE(m->forward(1, ids, &lg, /*all_rows=*/true, &err));
    out[arm].insert(out[arm].end(), lg.begin(), lg.end());
    for (std::size_t i = 0; i < 4; ++i) {
      ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>{dec[i]}, &lg, false,
                             &err));
      out[arm].insert(out[arm].end(), lg.begin(), lg.end());
    }
    const std::vector<std::int32_t> v6(dec.begin(), dec.begin() + 6);
    ASSERT_TRUE(m->forward(1, v6, &lg, /*all_rows=*/true, &err));
    out[arm].insert(out[arm].end(), lg.begin(), lg.end());
  }
  for (int arm = 1; arm < 4; ++arm) {
    const bool same = out[0].size() == out[arm].size() &&
                      std::memcmp(out[0].data(), out[arm].data(),
                                  out[0].size() * 4) == 0;
    std::printf("[dsv41] concurrent %s, fused post %s: %zu logits %s\n",
                arms[arm][0], arms[arm][1], out[0].size(),
                same ? "identical" : "DIFFER");
    EXPECT_TRUE(same);
  }
}

// An MLX pack's wo_a is the release's FP8 widened to bf16; the model puts
// it back into codes and scales (exactly: every product IS the bf16 value)
// on every block, and the logits stay where the bf16 route's are -- the
// two differ by their sums' order alone. The release layout keeps its own
// codes. Checked on the fp32 reference too, the bar the bf16 route meets.
TEST(deepseek_v41, an_mlx_wo_a_goes_back_to_its_codes)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  const std::vector<float> l32 =
      read_raw_<float>(fs::path(gd) / "logits_f32.f32");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  std::vector<float> lg[2];
  int codes[2] = {0, 0};
  bool mlx = false;
  for (int arm = 0; arm < 2; ++arm) {
    if (arm == 1) { ::setenv("VPIPE_DSV41_WO_A_BF16", "1", 1); }
    std::string err;
    MetalDeepseekV41Model::Options opt;
    auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                         mc, opt, &err);
    ::unsetenv("VPIPE_DSV41_WO_A_BF16");
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    mlx = m->mlx_layout();
    codes[arm] = m->wo_a_codes();
    ASSERT_TRUE(m->forward(1, ids, &lg[arm], /*all_rows=*/true, &err));
  }
  const std::size_t blocks = (std::size_t)codes[0];
  std::printf("[dsv41] wo_a back to codes on %zu blocks (%s layout); "
              "codes vs bf16 %.5f", blocks, mlx ? "MLX" : "release",
              rel_l2_(lg[0].data(), lg[1].data(), lg[0].size()));
  if (l32.size() == lg[0].size()) {
    std::printf(", vs fp32 %.4f / %.4f",
                rel_l2_(lg[0].data(), l32.data(), l32.size()),
                rel_l2_(lg[1].data(), l32.data(), l32.size()));
  }
  std::printf("\n");
  EXPECT_TRUE(codes[1] == 0);
  if (!mlx) {
    EXPECT_TRUE(codes[0] == 0);
    return;
  }
  EXPECT_TRUE(blocks > 0);
  EXPECT_TRUE(rel_l2_(lg[0].data(), lg[1].data(), lg[0].size()) < 1e-2);
  if (l32.size() == lg[0].size()) {
    EXPECT_TRUE(rel_l2_(lg[0].data(), l32.data(), l32.size()) <=
                1.05 * rel_l2_(lg[1].data(), l32.data(), l32.size()) + 1e-3);
  }
}

// The expert cache is sized against the box less its PEERS: a launch that
// declared this checkpoint (as its stage does, at the size on disk) must
// not count it against itself. On a box made small enough that the cache
// is needed, the session's answer equals the session-free one.
TEST(deepseek_v41, the_cache_does_not_count_its_own_claim)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const std::string md = (fs::path(gd) / "model").string();
  std::size_t ex = 0, rest = 0, vis = 0, disk = 0;
  ASSERT_TRUE(MetalDeepseekV41ModelExec::expert_bytes(md, &ex, &rest, &vis,
                                                      &disk));
  DeepseekV41Config cfg;
  std::string err;
  ASSERT_TRUE(DeepseekV41Config::from_dir(md, &cfg, &err));
  const std::size_t run = MetalDeepseekV41Model::run_bytes(cfg, 2048, 4096);
  const std::size_t held = rest + vis +
                           MetalDeepseekV41Vision::encode_scratch_bytes(cfg);
  // the 8 GB headroom, the rest, the run, and half the experts
  const std::size_t ram = ((std::size_t)8 << 30) + held + run + ex / 2;
  ::setenv("VPIPE_RAM_LIMIT_MB", std::to_string(ram >> 20).c_str(), 1);
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  const std::size_t alone =
      MetalDeepseekV41ModelExec::lm_expert_cache_bytes(nullptr, md, run);
  mgr->declare_weights(md, disk);
  // and its run, as an LM stage's plan books it (lm_claims)
  mgr->declare_scratch(lm_run_label(md), run, std::string());
  const std::size_t declared =
      MetalDeepseekV41ModelExec::lm_expert_cache_bytes(&sess, md, run);
  // a peer's arena still counts
  const std::size_t peer = alone / 4;
  mgr->declare_scratch("peer", peer, std::string());
  const std::size_t beside =
      MetalDeepseekV41ModelExec::lm_expert_cache_bytes(&sess, md, run);
  ::unsetenv("VPIPE_RAM_LIMIT_MB");
  std::printf("[dsv41] cache alone %zu bytes, with its own claims %zu, "
              "beside a peer's arena %zu (experts %zu)\n", alone, declared,
              beside, ex);
  EXPECT_TRUE(alone > ex / 4 && alone < ex);
  EXPECT_TRUE(declared == alone);
  EXPECT_TRUE(beside + peer == alone);
}

// The host's limit on what the model holds (load_spec::kMemoryLimitMb):
// the cache what it leaves after the run and the rest of the model --
// none needed where the limit holds every expert, a layer's floor (the
// cache's own) where it holds nothing -- and DSpark only where every
// expert is resident within it. The scope the manager opens around a
// load carries the figure, nested scopes restoring the outer.
TEST(deepseek_v41, the_cache_is_sized_within_the_hosts_limit)
{
  {
    EXPECT_TRUE(LmMemoryLimit::current() == 0);
    LmMemoryLimit a((std::size_t)64 << 30);
    EXPECT_TRUE(LmMemoryLimit::current() == (std::size_t)64 << 30);
    {
      LmMemoryLimit b(0);
      EXPECT_TRUE(LmMemoryLimit::current() == 0);
    }
    EXPECT_TRUE(LmMemoryLimit::current() == (std::size_t)64 << 30);
  }
  EXPECT_TRUE(LmMemoryLimit::current() == 0);
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const std::string md = (fs::path(gd) / "model").string();
  std::size_t ex = 0, rest = 0, vis = 0, disk = 0;
  ASSERT_TRUE(MetalDeepseekV41ModelExec::expert_bytes(md, &ex, &rest, &vis,
                                                      &disk));
  DeepseekV41Config cfg;
  std::string err;
  ASSERT_TRUE(DeepseekV41Config::from_dir(md, &cfg, &err));
  const std::size_t run = MetalDeepseekV41Model::run_bytes(cfg, 2048, 4096);
  const std::size_t held = rest + vis +
                           MetalDeepseekV41Vision::encode_scratch_bytes(cfg);
  auto cache = [&](std::size_t limit) {
    return MetalDeepseekV41ModelExec::lm_expert_cache_bytes(
        nullptr, md, run, nullptr, false, limit);
  };
  std::printf("[dsv41] limit: experts %zu, the rest %zu, run %zu; no limit "
              "%zu, half %zu\n", ex, held, run, cache(0),
              cache(run + held + ex / 2));
  // this box holds the small model whole
  EXPECT_TRUE(cache(0) == 0);
  EXPECT_TRUE(cache(run + held + ex) == 0);
  EXPECT_TRUE(cache(run + held + ex / 2) == ex / 2);
  EXPECT_TRUE(cache(run + held) == 1);
  EXPECT_TRUE(cache(1) == 1);
  if (MetalDeepseekV41Model::dspark_bytes(md) > 0) {
    ::unsetenv("VPIPE_DSV41_EXPERT_CACHE_MB");
    ::unsetenv("VPIPE_DSV41_DSPARK");
    EXPECT_TRUE(MetalDeepseekV41ModelExec::dspark_pays(nullptr, md, run));
    EXPECT_TRUE(!MetalDeepseekV41ModelExec::dspark_pays(
        nullptr, md, run, nullptr, run + held + ex / 2));
  }
}

// The host's limit reaches the PLAN, which is what its peers size
// against: an LM stage's claim books the checkpoint at what its cache,
// sized within the limit, leaves it holding -- the limit less the run and
// the vision tower's encode, never under the floor -- and the weights
// planner honours it as the runtime delivers the claim. The cache sized
// beside that plan leaves out exactly its own share: beside a peer's
// weights, on a box that binds before the limit does, it is the
// session-free figure on that box less the peer (subtracting the size on
// disk instead -- the old rule -- takes some of the peer's room too).
TEST(deepseek_v41, a_hosts_limit_reaches_the_plan)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const std::string md = (fs::path(gd) / "model").string();
  std::size_t ex = 0, rest = 0, vis = 0, disk = 0;
  ASSERT_TRUE(MetalDeepseekV41ModelExec::expert_bytes(md, &ex, &rest, &vis,
                                                      &disk));
  DeepseekV41Config cfg;
  std::string err;
  ASSERT_TRUE(DeepseekV41Config::from_dir(md, &cfg, &err));
  const std::size_t vs =
      vis > 0 ? MetalDeepseekV41Vision::encode_scratch_bytes(cfg) : 0;
  Session sess;
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mgr != nullptr);
  if (mgr == nullptr) { return; }
  LmRunShape shape;
  shape.kv_tokens = 4096;
  shape.max_pages = 8;
  const std::size_t run = lm_run_bytes(&sess, md, shape);
  // a limit that leaves the cache three quarters of the experts
  const std::size_t held = rest + vis + ex * 3 / 4;
  const std::size_t limit = run + vs + held;
  const auto claims =
      model_memory::lm_claims(&sess, md, "chat", shape, -1, 0, limit);
  ASSERT_TRUE(!claims.empty() && claims[0].key == md);
  if (claims.empty()) { return; }
  const ResourceClaim& w = claims[0];
  const std::size_t preload = model_memory::dir_weights_bytes(md);
  std::printf("[dsv41] a %zu MB limit: held %zu MB (floor %zu, on disk "
              "%zu)\n", limit >> 20, held >> 20, w.floor_bytes >> 20,
              preload >> 20);
  EXPECT_TRUE(bag::integer(w.extra, model_memory::kClaimHeldBytes, 0) ==
              (std::int64_t)held);
  EXPECT_TRUE(w.floor_bytes > 0 && w.floor_bytes < held);
  ASSERT_TRUE(held < preload);
  EXPECT_TRUE(bag::integer(model_memory::lm_claims(&sess, md, "chat", shape,
                                                   -1, 0)[0].extra,
                           model_memory::kClaimHeldBytes, 0) == 0);
  // delivered as the runtime does, with and without the claim in view
  auto deliver = [&](Session& s, bool in_view) {
    auto& reg = ResourcePlannerRegistry::get();
    for (std::string_view k :
         {model_memory::kWeightsKind, model_memory::kScratchKind}) {
      if (ResourcePlanner* p = reg.find(k)) { p->begin_plan(&s); }
    }
    for (const ResourceClaim& c : claims) {
      ResourcePlanner* p = reg.find(c.kind);
      if (p == nullptr) { continue; }
      std::optional<DeliveringResourceClaim> d;
      if (in_view) { d.emplace(c); }
      p->claim(&s, c.key, c.phase, c.last_phase, c.floor_bytes);
    }
  };
  {
    Session blind;
    deliver(blind, false);
    EXPECT_TRUE(blind.generative_model_manager()->planned_weight_bytes(md) ==
                preload);
  }
  deliver(sess, true);
  EXPECT_TRUE(mgr->planned_weight_bytes(md) == held);
  EXPECT_TRUE(mgr->phase_footprint(std::string()) == held);
  EXPECT_TRUE(mgr->phase_footprint_floor(std::string()) == w.floor_bytes);
  // beside a peer's weights, the box binding at half the experts
  const std::size_t peer = (std::size_t)2 << 30;
  mgr->declare_weights((fs::path(gd) / "a-peer").string(), peer);
  const std::size_t box =
      (((std::size_t)8 << 30) + run + rest + vis + vs + ex / 2) >> 20;
  ::setenv("VPIPE_RAM_LIMIT_MB", std::to_string(box + (peer >> 20)).c_str(),
           1);
  const std::size_t got = MetalDeepseekV41ModelExec::lm_expert_cache_bytes(
      &sess, md, run, nullptr, false, limit);
  ::setenv("VPIPE_RAM_LIMIT_MB", std::to_string(box).c_str(), 1);
  const std::size_t alone = MetalDeepseekV41ModelExec::lm_expert_cache_bytes(
      nullptr, md, run, nullptr, false, limit);
  ::unsetenv("VPIPE_RAM_LIMIT_MB");
  std::printf("[dsv41] beside the plan and a %zu MB peer: a %zu-byte "
              "cache; alone on the box less the peer %zu\n", peer >> 20,
              got, alone);
  EXPECT_TRUE(alone > 1 && alone < ex * 3 / 4);
  EXPECT_TRUE(got == alone);
}

// DSpark loads where it pays: only where every expert is resident (a
// checked draft on a cached box reads its own missed experts, and loses)
// -- and the override wins both ways (cache sizes pinned by
// VPIPE_DSV41_EXPERT_CACHE_MB, 0 = resident).
TEST(deepseek_v41, dspark_loads_where_it_pays)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const std::string md = (fs::path(gd) / "model").string();
  const std::size_t head = MetalDeepseekV41Model::dspark_bytes(md);
  std::printf("[dsv41] DSpark head %zu bytes\n", head);
  ASSERT_TRUE(head > 0);
  if (head == 0) { return; }
  auto pays = [&](const char* cache_mb, const char* force) {
    if (cache_mb != nullptr) {
      ::setenv("VPIPE_DSV41_EXPERT_CACHE_MB", cache_mb, 1);
    } else {
      ::unsetenv("VPIPE_DSV41_EXPERT_CACHE_MB");
    }
    if (force != nullptr) {
      ::setenv("VPIPE_DSV41_DSPARK", force, 1);
    } else {
      ::unsetenv("VPIPE_DSV41_DSPARK");
    }
    return MetalDeepseekV41ModelExec::dspark_pays(nullptr, md, 0);
  };
  // a cache of any size, however large against the head, keeps it off
  const std::string big = std::to_string(((head * 64) >> 20) + 2);
  EXPECT_TRUE(!pays("1", nullptr));
  EXPECT_TRUE(!pays(big.c_str(), nullptr));
  EXPECT_TRUE(pays("0", nullptr));               // every expert resident
  EXPECT_TRUE(pays(big.c_str(), "1"));
  EXPECT_TRUE(!pays("0", "0"));
  ::unsetenv("VPIPE_DSV41_EXPERT_CACHE_MB");
  ::unsetenv("VPIPE_DSV41_DSPARK");
}

// How often a prompt's engram rows were read by the text before it (a
// study for a row cache of the model's own): each file of
// VPIPE_DSV41_ENGRAM_TEXTS (colon-separated) a user turn in turn; for each,
// its distinct rows per slot, and how many of them earlier turns read --
// against every row seen, and LRU caches of 1M / 4M / 16M rows.
TEST(deepseek_v41, engram_reuse)
{
  const std::string md = env_("VPIPE_DSV41_TEST_MODEL_PATH");
  const std::string files = env_("VPIPE_DSV41_ENGRAM_TEXTS");
  if (md.empty() || files.empty()) { return; }
  std::string err;
  DeepseekV41Config cfg;
  ASSERT_TRUE(DeepseekV41Config::from_dir(md, &cfg, &err));
  DeepseekV41Engram eg;
  ASSERT_TRUE(eg.init(cfg, (fs::path(md) / "tokenizer.json").string(), &err));
  auto tok = Tokenizer::from_huggingface_json(
      (fs::path(md) / "tokenizer.json").string(), nullptr);
  ASSERT_TRUE(tok != nullptr);
  auto tpl = make_chat_template("DeepseekV41ForCausalLM", *tok);
  const int cols = cfg.engram_cols(), NS = (int)cfg.engram_layers.size();
  struct Lru {
    std::size_t cap;
    std::list<std::uint64_t> order;
    std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator> at;
    bool touch(std::uint64_t k) {
      auto it = at.find(k);
      if (it != at.end()) {
        order.splice(order.begin(), order, it->second);
        return true;
      }
      order.push_front(k);
      at[k] = order.begin();
      if (order.size() > cap) {
        at.erase(order.back());
        order.pop_back();
      }
      return false;
    }
  };
  Lru lru[3] = {{1u << 20, {}, {}}, {4u << 20, {}, {}}, {16u << 20, {}, {}}};
  std::unordered_set<std::uint64_t> seen;
  std::stringstream fs_(files);
  std::string f;
  while (std::getline(fs_, f, ':')) {
    std::ifstream in(f);
    std::stringstream ss;
    ss << in.rdbuf();
    std::vector<std::int32_t> ids;
    tpl->render_user_turn(ss.str(), true, &ids);
    std::vector<std::int32_t> hist;
    std::vector<std::int64_t> r((std::size_t)NS * cols);
    std::unordered_set<std::uint64_t> mine;
    std::size_t lookups = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      hist.push_back(eg.compress(ids[i]));
      eg.hash(hist, r.data());
      for (int sl = 0; sl < NS; ++sl) {
        for (int c = 0; c < cols; ++c) {
          mine.insert((std::uint64_t)sl << 40 |
                      (std::uint64_t)r[(std::size_t)sl * cols + c]);
          ++lookups;
        }
      }
    }
    std::size_t hit = 0, hl[3] = {0, 0, 0};
    for (std::uint64_t k : mine) {
      hit += seen.count(k);
      for (Lru& l : lru) { hl[&l - lru] += l.touch(k); }
    }
    for (std::uint64_t k : mine) { seen.insert(k); }
    std::printf("[dsv41] engram reuse, %s: %zu tokens, %zu lookups, %zu "
                "distinct (%.0f%%); of those read before %.1f%% (LRU 1M "
                "%.1f%%, 4M %.1f%%, 16M %.1f%%)\n",
                fs::path(f).filename().string().c_str(), ids.size(),
                lookups, mine.size(), 100.0 * mine.size() / lookups,
                100.0 * hit / mine.size(), 100.0 * hl[0] / mine.size(),
                100.0 * hl[1] / mine.size(), 100.0 * hl[2] / mine.size());
  }
}

// How an engram layer's table rows come off the drive (a study;
// VPIPE_DSV41_ENGRAM_BENCH=1 with the release or a pack): 8192 positions'
// rows -- of the bench prompt file's text, and of random tokens (rows no
// run has read) -- read as the model reads them (codes, scales and biases
// of a lookup apart, cached pread) and other ways: with the cache's read-
// ahead off, uncached, each distinct row once, more threads.
TEST(deepseek_v41, engram_reads)
{
  const std::string md = env_("VPIPE_DSV41_TEST_MODEL_PATH");
  if (md.empty() || env_("VPIPE_DSV41_ENGRAM_BENCH").empty()) { return; }
  std::string err;
  DeepseekV41Config cfg;
  ASSERT_TRUE(DeepseekV41Config::from_dir(md, &cfg, &err));
  DeepseekV41Engram eg;
  ASSERT_TRUE(eg.init(cfg, (fs::path(md) / "tokenizer.json").string(), &err));
  if (!err.empty()) { std::printf("[dsv41] %s\n", err.c_str()); return; }
  const int cols = cfg.engram_cols(), NS = (int)cfg.engram_layers.size();
  const int N = 8192;
  // the positions' rows
  auto rows_of = [&](const std::vector<std::int32_t>& ids) {
    std::vector<std::int32_t> hist;
    for (std::int32_t t : ids) { hist.push_back(eg.compress(t)); }
    std::vector<std::int64_t> r((std::size_t)N * NS * cols);
    for (int i = 0; i < N; ++i) {
      eg.hash(std::span<const std::int32_t>(hist.data(), (std::size_t)i + 1),
              r.data() + (std::size_t)i * NS * cols);
    }
    return r;
  };
  std::vector<std::int32_t> text_ids, rand_ids;
  if (const std::string f = env_("VPIPE_DSV41_BENCH_PROMPT_FILE"); !f.empty()) {
    std::ifstream in(f);
    std::stringstream ss;
    ss << in.rdbuf();
    auto tok = Tokenizer::from_huggingface_json(
        (fs::path(md) / "tokenizer.json").string(), nullptr);
    ASSERT_TRUE(tok != nullptr);
    auto tpl = make_chat_template("DeepseekV41ForCausalLM", *tok);
    tpl->render_user_turn(ss.str(), true, &text_ids);
  }
  std::uint64_t x = (std::uint64_t)std::chrono::steady_clock::now()
                        .time_since_epoch().count();
  for (int i = 0; i < N; ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    rand_ids.push_back((std::int32_t)((x >> 33) % 128000));
  }
  // the first layer's table: where its three tensors lie
  struct Where { int fd = -1; std::uint64_t at = 0, row = 0; };
  auto locate = [&](const std::string& name) {
    Where w;
    std::ifstream in(fs::path(md) / "model.safetensors.index.json");
    const FlexData idx = FlexData::from_json(in);
    const std::string file = std::string(
        idx.as_object().at("weight_map").as_object().at(name).as_string(""));
    const std::string path = (fs::path(md) / file).string();
    std::ifstream sf(path, std::ios::binary);
    std::uint64_t hn = 0;
    sf.read(reinterpret_cast<char*>(&hn), 8);
    std::string hj(hn, '\0');
    sf.read(hj.data(), (std::streamsize)hn);
    std::istringstream hs(hj);
    const FlexData h = FlexData::from_json(hs);
    const FlexData t = h.as_object().at(name);
    auto to = t.as_object();
    const FlexData offs = to.at("data_offsets");
    const FlexData shape = to.at("shape");
    auto oa = offs.as_array();
    auto sa = shape.as_array();
    const std::uint64_t a = (std::uint64_t)oa[0].as_int();
    const std::uint64_t b = (std::uint64_t)oa[1].as_int();
    w.at = 8 + hn + a;
    w.row = (b - a) / (std::uint64_t)sa[0].as_int();
    w.fd = ::open(path.c_str(), O_RDONLY);
    return w;
  };
  const std::string L = std::to_string(cfg.engram_layers[0]);
  const std::string base = "language_model.layers." + L + ".engram.embed.";
  Where W[3] = {locate(base + "weight"), locate(base + "scales"),
                locate(base + "biases")};
  ASSERT_TRUE(W[0].fd >= 0 && W[1].fd >= 0 && W[2].fd >= 0);
  std::printf("[dsv41] engram table %s: rows of %llu + %llu + %llu bytes\n",
              base.c_str(), (unsigned long long)W[0].row,
              (unsigned long long)W[1].row, (unsigned long long)W[2].row);
  const long page = ::sysconf(_SC_PAGESIZE);
  // read slot 0's lookups of `r`, each distinct row once if `dedup`
  auto run = [&](const std::vector<std::int64_t>& r, bool dedup, int threads,
                 bool direct, int parts = 3, bool sorted = false,
                 long blk = 0) {
    if (blk == 0) { blk = page; }
    std::vector<std::int64_t> want;
    for (int i = 0; i < N; ++i) {
      for (int c = 0; c < cols; ++c) {
        want.push_back(r[(std::size_t)i * NS * cols + c]);
      }
    }
    if (dedup || sorted) { std::sort(want.begin(), want.end()); }
    if (dedup) {
      want.erase(std::unique(want.begin(), want.end()), want.end());
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<std::size_t> next{0};
    std::atomic<long> bad{0};
    auto work = [&] {
      std::vector<std::uint8_t> st((std::size_t)(4 * page));
      auto* buf = reinterpret_cast<std::uint8_t*>(
          ((std::uintptr_t)st.data() + page - 1) / page * page);
      std::uint8_t tmp[512];
      for (;;) {
        const std::size_t k = next.fetch_add(64);
        if (k >= want.size()) { break; }
        for (std::size_t j = k; j < std::min(want.size(), k + 64); ++j) {
          for (int q = 0; q < parts; ++q) {
            const Where& w = W[q];
            const std::uint64_t at = w.at + (std::uint64_t)want[j] * w.row;
            ssize_t got;
            if (direct) {
              const std::uint64_t a0 = at / blk * blk;
              const std::uint64_t a1 = (at + w.row + blk - 1) / blk * blk;
              got = ::pread(w.fd, buf, (std::size_t)(a1 - a0), (off_t)a0);
              got = got >= (ssize_t)(at + w.row - a0) ? (ssize_t)w.row : -1;
            } else {
              got = ::pread(w.fd, tmp, (std::size_t)w.row, (off_t)at);
            }
            if (got != (ssize_t)w.row) { ++bad; }
          }
        }
      }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) { pool.emplace_back(work); }
    for (auto& t : pool) { t.join(); }
    const double s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();
    EXPECT_TRUE(bad == 0);
    return std::make_pair(s, want.size());
  };
  auto mode = [&](int fdflag, int v) {
    for (const Where& w : W) { (void)::fcntl(w.fd, fdflag, v); }
  };
  if (!text_ids.empty()) {
    const std::vector<std::int32_t> tid(
        text_ids.begin(),
        text_ids.begin() + std::min<std::size_t>(text_ids.size(), N));
    if ((int)tid.size() == N) {
      const auto r = rows_of(tid);
      std::vector<std::int64_t> one;
      for (int i = 0; i < N; ++i) {
        for (int c = 0; c < cols; ++c) {
          one.push_back(r[(std::size_t)i * NS * cols + c]);
        }
      }
      std::vector<std::int64_t> u = one;
      std::sort(u.begin(), u.end());
      u.erase(std::unique(u.begin(), u.end()), u.end());
      std::printf("[dsv41] engram, the prompt's first %d positions: %zu "
                  "lookups, %zu distinct rows (%.0f%%)\n", N, one.size(),
                  u.size(), 100.0 * u.size() / one.size());
      mode(F_RDAHEAD, 1);
      const auto c1 = run(r, false, 24, false);
      const auto c2 = run(r, false, 24, false);
      const auto c3 = run(r, true, 24, false);
      const auto c4 = run(r, false, 24, false, 3, true);
      std::printf("[dsv41] engram, the prompt: cached reads %.2f s, again "
                  "%.2f s, each distinct row once %.2f s, all of them "
                  "sorted %.2f s\n", c1.first, c2.first, c3.first,
                  c4.first);
    }
  }
  // rows nothing has read: random tokens, a fresh sequence each way
  struct Arm { const char* name; bool dedup; int threads; int how;
               int parts; long blk; };
  const Arm arms[] = {
      {"cached, as the model reads (24 threads)", false, 24, 0, 3, 0},
      {"uncached, 16 KB pages", false, 24, 2, 3, 0},
      {"uncached, 4 KB blocks", false, 24, 2, 3, 4096},
      {"uncached, 4 KB blocks, 64 threads", false, 64, 2, 3, 4096},
      {"uncached, 4 KB blocks, the codes alone", false, 24, 2, 1, 4096},
      {"uncached, 4 KB blocks, the codes alone, 64 threads", false, 64, 2,
       1, 4096},
  };
  for (int round = 0; round < 2; ++round) {
    for (const Arm& a : arms) {
      for (int i = 0; i < N; ++i) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        rand_ids[(std::size_t)i] = (std::int32_t)((x >> 33) % 128000);
      }
      const auto r = rows_of(rand_ids);
      mode(F_NOCACHE, a.how == 2 ? 1 : 0);
      mode(F_RDAHEAD, a.how == 0 ? 1 : 0);
      const auto t = run(r, a.dedup, a.threads, a.how == 2, a.parts, false,
                         a.blk);
      std::printf("[dsv41] engram, %d random positions' rows (%zu reads of "
                  "3), %s: %.2f s (%.0f lookups/s)\n", N, t.second, a.name,
                  t.first, t.second / t.first);
    }
  }
  // the cache: one random set read twice back to back
  {
    for (int i = 0; i < N; ++i) {
      x = x * 6364136223846793005ull + 1442695040888963407ull;
      rand_ids[(std::size_t)i] = (std::int32_t)((x >> 33) % 128000);
    }
    const auto r = rows_of(rand_ids);
    mode(F_NOCACHE, 0);
    mode(F_RDAHEAD, 0);
    const auto t1 = run(r, true, 24, false);
    const auto t2 = run(r, true, 24, false);
    std::printf("[dsv41] engram, a random set's distinct rows, sorted, "
                "cached: %.2f s, the same again %.2f s\n", t1.first,
                t2.first);
  }
  // a read's own cost: one row, read 590K times by 24 threads
  {
    std::vector<std::int64_t> one((std::size_t)N * NS * cols, 12345);
    const auto t = run(one, false, 24, false);
    std::printf("[dsv41] engram, one row 590K times (24 threads): %.2f s\n",
                t.first);
  }
  // the tables mapped: a row a copy, first touch a fault
  {
    std::vector<const std::uint8_t*> map(3);
    std::vector<std::size_t> len(3);
    for (int q = 0; q < 3; ++q) {
      struct stat sb;
      ::fstat(W[q].fd, &sb);
      len[(std::size_t)q] = (std::size_t)sb.st_size;
      void* m = ::mmap(nullptr, len[(std::size_t)q], PROT_READ, MAP_SHARED,
                       W[q].fd, 0);
      map[(std::size_t)q] = m == MAP_FAILED ? nullptr
                                            : static_cast<std::uint8_t*>(m);
    }
    ASSERT_TRUE(map[0] != nullptr && map[1] != nullptr && map[2] != nullptr);
    for (int pass = 0; pass < 3; ++pass) {
      if (pass != 1) {
        for (int i = 0; i < N; ++i) {
          x = x * 6364136223846793005ull + 1442695040888963407ull;
          rand_ids[(std::size_t)i] = (std::int32_t)((x >> 33) % 128000);
        }
      }
      const auto r = rows_of(rand_ids);
      std::vector<std::int64_t> want;
      for (int i = 0; i < N; ++i) {
        for (int c = 0; c < cols; ++c) {
          want.push_back(r[(std::size_t)i * NS * cols + c]);
        }
      }
      const auto t0 = std::chrono::steady_clock::now();
      std::atomic<std::size_t> next{0};
      std::atomic<std::uint64_t> sum{0};
      auto work = [&] {
        std::uint64_t acc = 0;
        std::uint8_t tmp[512];
        for (;;) {
          const std::size_t k = next.fetch_add(64);
          if (k >= want.size()) { break; }
          for (std::size_t j = k; j < std::min(want.size(), k + 64); ++j) {
            for (int q = 0; q < 3; ++q) {
              const std::uint8_t* src = map[(std::size_t)q] + W[q].at +
                                        (std::uint64_t)want[j] * W[q].row;
              std::memcpy(tmp, src, (std::size_t)W[q].row);
              acc += tmp[0];
            }
          }
        }
        sum += acc;
      };
      std::vector<std::thread> pool;
      for (int t = 0; t < 24; ++t) { pool.emplace_back(work); }
      for (auto& t : pool) { t.join(); }
      const double sec = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
      std::printf("[dsv41] engram, mapped, %s: %.2f s (%.0f lookups/s)\n",
                  pass == 1 ? "the same rows again" : "fresh random rows",
                  sec, want.size() / sec);
    }
    for (int q = 0; q < 3; ++q) {
      ::munmap(const_cast<std::uint8_t*>(map[(std::size_t)q]),
               len[(std::size_t)q]);
    }
  }
  mode(F_NOCACHE, 0);
  mode(F_RDAHEAD, 1);
  for (const Where& w : W) { ::close(w.fd); }
}


// The decode's rate on the real model (VPIPE_DSV41_TEST_MODEL_PATH +
// VPIPE_DSV41_BENCH=<tokens>): a chat prompt, then greedy tokens through
// the exec, reported a block of 16 at a time with the expert cache's hit
// rate -- the rate a conversation settles to as the cache learns its
// experts. VPIPE_DSV41_BENCH_PROMPT=<tokens> makes the prompt that long
// (a prefill rate); VPIPE_DSV41_BENCH_SPARK=1 decodes speculatively with
// the DSpark head, VPIPE_DSV41_BENCH_BLOCK=<rows> its verify's rows,
// VPIPE_DSV41_BENCH_TEMP=<t> sampling it. VPIPE_DSV41_BENCH_TEXTS=<file>
// runs a prompt a line instead, one after the other, and totals them;
// VPIPE_DSV41_BENCH_THINK=<effort> renders thinking mode;
// VPIPE_DSV41_BENCH_PROMPT_FILE=<file> sends a whole file as the prompt.
TEST(deepseek_v41, release_decode_rate)
{
  const std::string md = env_("VPIPE_DSV41_TEST_MODEL_PATH");
  const int N = std::atoi(env_("VPIPE_DSV41_BENCH").c_str());
  if (md.empty() || N <= 0) { return; }
  const bool spark = !env_("VPIPE_DSV41_BENCH_SPARK").empty();
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  std::string err;
  DeepseekV41Config cfg;
  ASSERT_TRUE(DeepseekV41Config::from_dir(md, &cfg, &err));
  MetalDeepseekV41Model::Options opt;
  opt.dspark = spark;
  opt.expert_cache_bytes = MetalDeepseekV41ModelExec::lm_expert_cache_bytes(
      nullptr, md,
      MetalDeepseekV41Model::run_bytes(cfg, opt.max_rows, 8192, spark), mc,
      spark);
  using Clk = std::chrono::steady_clock;
  auto sec = [](auto a, auto b) {
    return std::chrono::duration<double>(b - a).count();
  };
  const auto tl = Clk::now();
  MetalDeepseekV41ModelExec ex(md, mc, opt, &sess);
  ASSERT_TRUE(ex.valid());
  if (!ex.valid()) { return; }
  MetalDeepseekV41Model* m = ex.model();
  std::printf("[dsv41] bench: loaded in %.1f s, holds %zu MB, cache %zu MB%s"
              ", footprint %zu MB\n", sec(tl, Clk::now()),
              m->resident_bytes() >> 20, m->expert_cache()->bytes() >> 20,
              spark ? ", DSpark loaded" : "", footprint_() >> 20);
  // VPIPE_DSV41_BENCH_PRELOAD=1: every expert read before the first
  // prompt (a steady state, not the cache's warm-up)
  if (!env_("VPIPE_DSV41_BENCH_PRELOAD").empty()) {
    const auto tp = Clk::now();
    std::string perr;
    const int got = m->preload_experts(&perr);
    const double s = sec(tp, Clk::now());
    const auto& pst = m->expert_cache()->stats();
    std::printf("[dsv41] bench: preloaded %d experts (%.1f GB) in %.1f s, "
                "%.1f GB/s%s%s\n", got, pst.bytes_read / 1e9, s,
                pst.bytes_read / 1e9 / std::max(s, 1e-9),
                perr.empty() ? "" : ": ", perr.c_str());
    m->expert_cache_mut()->reset_stats();
  }
  auto tok = Tokenizer::from_huggingface_json(
      (fs::path(md) / "tokenizer.json").string(), nullptr);
  ASSERT_TRUE(tok != nullptr);
  if (tok == nullptr) { return; }
  // VPIPE_DSV41_BENCH_THINK=<effort>: thinking mode at that effort
  const std::string think = env_("VPIPE_DSV41_BENCH_THINK");
  auto tpl = think.empty()
                 ? make_chat_template("DeepseekV41ForCausalLM", *tok)
                 : make_chat_template("DeepseekV41ForCausalLM", *tok,
                                      false, think);
  ASSERT_TRUE(tpl != nullptr);
  if (tpl == nullptr) { return; }
  std::vector<std::string> texts{
      "Write a short story about a lighthouse keeper."};
  const int P = std::atoi(env_("VPIPE_DSV41_BENCH_PROMPT").c_str());
  if (P > 0) {
    // a long prompt: a passage repeated, then the request
    const std::string para =
        "The keeper climbed the spiral stair each evening at dusk, lit the "
        "great lamp, and wrote in the log the state of the sea, the wind "
        "and the ships that passed in the night. ";
    const int per = std::max<int>(1, (int)tok->encode(para).size());
    std::string body;
    for (int k = 0; k * per < P; ++k) { body += para; }
    texts = {body + "\n\nSummarize the passage above in one sentence."};
  }
  // VPIPE_DSV41_BENCH_TEXTS=<file>: a prompt a line, each in a context of
  // its own, one after the other (the cache stays warm between them)
  if (const std::string f = env_("VPIPE_DSV41_BENCH_TEXTS"); !f.empty()) {
    std::ifstream in(f);
    texts.clear();
    for (std::string line; std::getline(in, line);) {
      if (!line.empty()) { texts.push_back(line); }
    }
    ASSERT_TRUE(!texts.empty());
  }
  // VPIPE_DSV41_BENCH_PROMPT_FILE=<file>: the whole file one prompt
  if (const std::string f = env_("VPIPE_DSV41_BENCH_PROMPT_FILE");
      !f.empty()) {
    std::ifstream in(f);
    std::stringstream ss;
    ss << in.rdbuf();
    // VPIPE_DSV41_BENCH_REPEAT=<n>: n times (the later ones warm)
    const int rep =
        std::max(1, std::atoi(env_("VPIPE_DSV41_BENCH_REPEAT").c_str()));
    texts.assign((std::size_t)rep, ss.str());
    ASSERT_TRUE(!texts[0].empty());
  }
  long all_tokens = 0, all_rounds = 0, all_kept = 0, all_drafted = 0;
  double all_s = 0.0;
  for (std::size_t q = 0; q < texts.size(); ++q) {
    std::vector<std::int32_t> ids;
    tpl->render_user_turn(texts[q], true, &ids);
    const ContextId ctx{1};
    const auto t0 = Clk::now();
    std::int32_t t = ex.prefill(ctx, ids);
    ASSERT_TRUE(t >= 0);
    if (t < 0) { return; }
    const auto t1 = Clk::now();
    std::printf("[dsv41] bench: prompt %zu: %zu tokens prefilled in %.1f s "
                "(%.1f tok/s)\n", q + 1, ids.size(), sec(t0, t1),
                (double)ids.size() / sec(t0, t1));
    std::vector<std::int32_t> out{t};
    auto tb = Clk::now();
    auto st0 = m->expert_cache()->stats();
    int last = 0;
    // with VPIPE_MC_GPU_TIME=1, the GPU's busy time a token too
    double busy0 = 0.0;
    unsigned long long nb0 = 0;
    metal_compute::CommandStream::gpu_time_stats(&busy0, &nb0);
    auto report = [&](int upto) {
      const auto now = Clk::now();
      const auto& st = m->expert_cache()->stats();
      const double h = st.picks > st0.picks
                           ? (double)(st.hits - st0.hits) /
                                 (double)(st.picks - st0.picks)
                           : 0.0;
      double busy1 = 0.0;
      unsigned long long nb1 = 0;
      metal_compute::CommandStream::gpu_time_stats(&busy1, &nb1);
      const int nt = std::max(upto - last, 1);
      const std::string gpu =
          nb1 > nb0 ? std::format("; GPU busy {:.2f} ms a token in {:.1f} "
                                  "buffers", (busy1 - busy0) / nt,
                                  (double)(nb1 - nb0) / nt)
                    : std::string();
      std::printf("[dsv41] bench: tokens %d-%d %.2f tok/s, hits %.0f%%, "
                  "%.1f GB read, waited %.1f s; ahead: width %d, %llu read, "
                  "%llu picked%s\n", last + 1, upto,
                  (double)(upto - last) / sec(tb, now), 100.0 * h,
                  (double)(st.bytes_read - st0.bytes_read) / 1e9,
                  (st.read_ms + st.wait_ms - st0.read_ms - st0.wait_ms) / 1e3,
                  m->expert_cache()->ahead_width(),
                  (unsigned long long)(st.ahead_read - st0.ahead_read),
                  (unsigned long long)(st.ahead_hits - st0.ahead_hits),
                  gpu.c_str());
      tb = now;
      st0 = st;
      last = upto;
      busy0 = busy1;
      nb0 = nb1;
    };
    const auto td = Clk::now();
    if (spark) {
      auto* sx = static_cast<SpecDecodeExt*>(
          ex.query_extension(spec_decode::kExtensionId));
      ASSERT_TRUE(sx != nullptr);
      if (sx == nullptr) { return; }
      std::vector<std::int32_t> got;
      std::function<bool(std::span<const std::int32_t>)> on =
          [&](std::span<const std::int32_t> k) {
            got.insert(got.end(), k.begin(), k.end());
            if ((int)got.size() - last >= 16) { report((int)got.size()); }
            return true;
          };
      std::function<bool(std::int32_t)> stop = [&](std::int32_t x) {
        return x == cfg.eos;
      };
      SpecDecodeRequest rq;
      rq.ctx = ctx;
      rq.first_token = t;
      rq.max_tokens = N;
      rq.on_tokens = &on;
      rq.is_stop = &stop;
      rq.block = std::atoi(env_("VPIPE_DSV41_BENCH_BLOCK").c_str());
      // VPIPE_DSV41_BENCH_TEMP=<t>: sampled (top-p 0.95), as the chat runs
      if (const std::string tv = env_("VPIPE_DSV41_BENCH_TEMP"); !tv.empty()) {
        rq.sampler.greedy = false;
        rq.sampler.temperature = (float)std::atof(tv.c_str());
        rq.sampler.top_p = 0.95f;
        rq.sampler.seed = 1;
        rq.prime = ids;
      }
      SpecDecodeResult res;
      ASSERT_TRUE(sx->generate(rq, &res));
      out = got;
      const double s = sec(td, Clk::now());
      std::printf("[dsv41] bench: DSpark %d tokens in %ld rounds (%.2f a "
                  "round), %ld of %ld drafts kept, %.2f tok/s overall\n",
                  res.produced, res.rounds,
                  (double)res.produced / std::max(1L, res.rounds),
                  res.accepted, res.drafted, res.produced / s);
      all_tokens += res.produced;
      all_rounds += res.rounds;
      all_kept += res.accepted;
      all_drafted += res.drafted;
      all_s += s;
    } else {
      // VPIPE_DSV41_BENCH_NO_EOS=1: past an end of sequence too (timing)
      const bool no_eos = !env_("VPIPE_DSV41_BENCH_NO_EOS").empty();
      for (int i = 1; i < N && (no_eos || t != cfg.eos); ++i) {
        t = ex.decode_one(ctx, t);
        ASSERT_TRUE(t >= 0);
        out.push_back(t);
        if (i % 16 == 0) { report(i); }
      }
      const double s = sec(td, Clk::now());
      std::printf("[dsv41] bench: %zu tokens, %.2f tok/s overall\n",
                  out.size(), out.size() / s);
      all_tokens += (long)out.size();
      all_s += s;
      // VPIPE_MC_GPU_TIME=2: one more token's command buffers, each
      // against the one before: the GPU's busy time, and its idle between
      if (env_("VPIPE_MC_GPU_TIME") == "2" && t != cfg.eos) {
        (void)metal_compute::CommandStream::take_gpu_spans();
        const double h0 = metal_compute::CommandStream::host_seconds();
        ASSERT_TRUE(ex.decode_one(ctx, t) >= 0);
        const double h1 = metal_compute::CommandStream::host_seconds();
        const auto sp = metal_compute::CommandStream::take_gpu_spans();
        double busy = 0.0, idle = 0.0;
        for (std::size_t i = 0; i < sp.size(); ++i) {
          busy += sp[i].end - sp[i].start;
          if (i > 0) { idle += std::max(0.0, sp[i].start - sp[i - 1].end); }
        }
        std::printf("[dsv41] bench: a token's %zu buffers: %.2f ms on the "
                    "host, GPU busy %.2f, idle between %.2f, before the "
                    "first %.2f, after the last %.2f\n", sp.size(),
                    (h1 - h0) * 1e3, busy * 1e3, idle * 1e3,
                    sp.empty() ? 0.0 : (sp[0].start - h0) * 1e3,
                    sp.empty() ? 0.0 : (h1 - sp.back().end) * 1e3);
        for (std::size_t i = 0; i < sp.size(); ++i) {
          std::printf("[dsv41] span %zu: start +%.3f ms, GPU %.3f, gap "
                      "after %.3f\n", i, (sp[i].start - h0) * 1e3,
                      (sp[i].end - sp[i].start) * 1e3,
                      i + 1 < sp.size() ? (sp[i + 1].start - sp[i].end) * 1e3
                                        : 0.0);
        }
      }
    }
    std::printf("[dsv41] bench: reply %zu: \"%s\"\n", q + 1,
                tok->decode(out).c_str());
    ex.release_context(ctx);
  }
  if (texts.size() > 1) {
    std::printf("[dsv41] bench: %zu prompts, %ld tokens in %.1f s, %.2f "
                "tok/s", texts.size(), all_tokens, all_s,
                all_tokens / std::max(1e-9, all_s));
    if (spark) {
      std::printf("; %.2f a round, %ld of %ld drafts kept",
                  (double)all_tokens / std::max(1L, all_rounds), all_kept,
                  all_drafted);
    }
    std::printf("\n");
  }
  if (spark) {
    // what a round cost by its rows, and the drafts' confidence against
    // what was kept
    const auto& ds = ex.draft_stats();
    if (ds.drafts > 0) {
      std::printf("[dsv41] bench: draft %.1f ms a round\n",
                  ds.draft_ms / (double)ds.drafts);
    }
    for (std::size_t r = 1; r < ds.verifies.size(); ++r) {
      if (ds.verifies[r] == 0) { continue; }
      std::printf("[dsv41] bench: verify of %zu rows %.1f ms (%lld)\n", r,
                  ds.verify_ms[r] / (double)ds.verifies[r],
                  (long long)ds.verifies[r]);
    }
    for (std::size_t i = 0; i * 10 < ds.reached.size(); ++i) {
      std::string line;
      for (std::size_t b = 0; b < 10; ++b) {
        const std::int64_t n = ds.reached[i * 10 + b];
        if (n == 0) { continue; }
        line += std::format(" {}0%: {}/{}", b, ds.kept[i * 10 + b], n);
      }
      std::printf("[dsv41] bench: draft %zu kept by confidence:%s\n", i,
                  line.c_str());
    }
  }
  std::printf("[dsv41] bench: footprint %zu MB\n", footprint_() >> 20);
  // VPIPE_DSV41_AHEAD_STUDY=K: what could have been read ahead
  {
    std::istringstream in(m->ahead_study());
    for (std::string line; std::getline(in, line);) {
      std::printf("[dsv41] study: %s\n", line.c_str());
    }
  }
}

// Reading ahead moves which slot holds an expert, never a value: the same
// tokens through a cache too small for every expert, read-ahead pinned at
// top-k and off, give the same logits to the bit -- decode steps and a
// verify's few rows -- and the read-ahead really ran (experts read ahead,
// some of them picked; the verify read ahead too).
TEST(deepseek_v41, read_ahead_is_exact)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  const std::vector<std::int32_t> dec = ints_(meta, "decode_ids");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  std::size_t ex = 0, rest = 0;
  ASSERT_TRUE(MetalDeepseekV41ModelExec::expert_bytes(md.string(), &ex,
                                                      &rest));
  std::vector<float> out[2];
  std::uint64_t ahead_read = 0, ahead_hits = 0, ahead_named = 0;
  for (int a = 0; a < 2; ++a) {
    ::setenv("VPIPE_MOE_EXPERT_AHEAD", a == 0 ? "0" : "8", 1);
    std::string err;
    MetalDeepseekV41Model::Options opt;
    opt.expert_cache_bytes = ex / 3;
    auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                         mc, opt, &err);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { break; }
    std::vector<float> lg;
    ASSERT_TRUE(m->forward(1, ids, &lg, false, &err));
    out[a] = lg;
    // greedy, so the second arm feeds the same tokens only if it agrees
    for (std::size_t i = 0; i < 2 * dec.size(); ++i) {
      const std::int32_t t = argmax_(lg.data(), m->config().vocab);
      ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>{t}, &lg, false,
                             &err));
      out[a].insert(out[a].end(), lg.begin(), lg.end());
    }
    // a verify's few rows read ahead for each row
    const std::uint64_t before = m->expert_cache()->stats().ahead_read;
    const std::vector<std::int32_t> few(dec.begin(),
                                        dec.begin() + std::min<std::size_t>(
                                                          dec.size(), 5));
    ASSERT_TRUE(m->forward(1, few, &lg, true, &err));
    out[a].insert(out[a].end(), lg.begin(), lg.end());
    if (a == 1) {
      EXPECT_TRUE(few.size() < 2 ||
                  m->expert_cache()->stats().ahead_read > before);
    }
    if (a == 1) {
      const auto& st = m->expert_cache()->stats();
      ahead_read = st.ahead_read;
      ahead_hits = st.ahead_hits;
      ahead_named = st.ahead_named;
      std::printf("[dsv41] read ahead: %llu read, %llu picked of %llu "
                  "named; hits %.1f%%\n", (unsigned long long)ahead_read,
                  (unsigned long long)ahead_hits,
                  (unsigned long long)ahead_named, 100.0 * st.hit_rate());
    }
  }
  ::unsetenv("VPIPE_MOE_EXPERT_AHEAD");
  ASSERT_TRUE(out[0].size() == out[1].size() && !out[0].empty());
  EXPECT_TRUE(std::memcmp(out[0].data(), out[1].data(),
                          out[0].size() * sizeof(float)) == 0);
  EXPECT_TRUE(ahead_read > 0 && ahead_hits > 0);
}

// A LONG PREFILL'S ENGRAM ROWS, read ahead: each piece's while the piece
// before it runs, into the set that piece frees (one set beside _erow).
// Exact -- the same pieces forwarded one call at a time read their own
// rows, and the two contexts' logits agree to the bit -- and what the
// prefill allocates is what run_bytes books for it: the peak device
// allocation, sampled from a thread. (MUTATION: a fresh set for every
// read-ahead, as before, holds two beside _erow -- 12 MB more than
// booked at these 512-row pieces.)
TEST(deepseek_v41, a_long_prefill_reads_ahead_in_what_it_books)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> base = ints_(meta, "ids");
  ASSERT_TRUE(!base.empty());
  if (base.empty()) { return; }
  // four pieces (three read ahead, two of them into a freed set)
  const int R = 512, P = 4 * R;
  std::vector<std::int32_t> ids;
  for (int i = 0; i < P; ++i) {
    ids.push_back(base[(std::size_t)i % base.size()]);
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  std::string err;
  MetalDeepseekV41Model::Options opt;
  opt.max_rows = R;
  auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                       mc, opt, &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { std::printf("[dsv41] %s\n", err.c_str()); return; }
  ASSERT_TRUE(!m->config().engram_layers.empty());
  // the prefill in one call, its peak allocation sampled
  const std::size_t alloc0 = mc->memory_budget().allocated;
  std::atomic<std::size_t> peak{alloc0};
  std::atomic<bool> done{false};
  std::thread sampler([&] {
    while (!done.load()) {
      const std::size_t a = mc->memory_budget().allocated;
      if (a > peak.load()) { peak.store(a); }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  });
  std::vector<float> one;
  const bool ok = m->forward(1, ids, &one, false, &err);
  done.store(true);
  sampler.join();
  ASSERT_TRUE(ok);
  if (!ok) { std::printf("[dsv41] %s\n", err.c_str()); return; }
  const std::size_t grew =
      std::max(peak.load(), mc->memory_budget().allocated) - alloc0;
  const long ahead = m->engram_read_ahead();
  // the same pieces, a call each
  std::vector<float> each;
  for (int i = 0; i < P; i += R) {
    const bool lastp = i + R == P;
    ASSERT_TRUE(m->forward(2, std::span<const std::int32_t>(ids).subspan(
                                  (std::size_t)i, (std::size_t)R),
                           lastp ? &each : nullptr, false, &err));
  }
  std::printf("[dsv41] a %d-row prefill in %d-row pieces: %ld read ahead; "
              "logits %s the pieces' a call each\n", P, R, ahead,
              one == each ? "match" : "DIFFER from");
  EXPECT_TRUE(ahead == P / R - 1);
  EXPECT_TRUE(m->engram_read_ahead() == ahead);
  ASSERT_TRUE(!one.empty() && one.size() == each.size());
  EXPECT_TRUE(std::memcmp(one.data(), each.data(),
                          one.size() * sizeof(float)) == 0);
  const std::size_t booked =
      MetalDeepseekV41Model::run_bytes(m->config(), R, P, false);
  std::printf("[dsv41] the prefill's peak allocation %.1f MB; run_bytes "
              "books %.1f MB (%.3fx)\n", grew / 1048576.0,
              booked / 1048576.0, grew > 0 ? (double)booked / grew : 0.0);
  // The sampled peak can only under-read the true one; 2 MB covers the
  // per-forward parameter blocks the formula leaves out.
  EXPECT_TRUE(grew > 0);
  EXPECT_TRUE(booked + (std::size_t(2) << 20) >= grew);
  EXPECT_TRUE((double)booked <= 1.25 * (double)grew + (8 << 20));
}

// A speculative verify's rows ARE decode rows: from one prefilled
// context, k tokens forwarded at once (all rows' logits) and the same k
// decoded one at a time give the same logits to the bit, for every k a
// verify takes -- what makes a speculative decode token-exact.
TEST(deepseek_v41, verify_rows_are_decode_rows)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  std::vector<std::int32_t> dec = ints_(meta, "decode_ids");
  while (dec.size() < 7) { dec.push_back(dec[dec.size() % 3]); }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  for (bool bf16 : {true, false}) {
    std::string err;
    MetalDeepseekV41Model::Options opt;
    opt.use_bf16 = bf16;
    auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                         mc, opt, &err);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    std::vector<float> lg;
    ASSERT_TRUE(m->forward(1, ids, &lg, false, &err));
    std::string line;
    bool all_same = true;
    for (int k = 2; k <= 7; ++k) {
      ASSERT_TRUE(m->branch(1, 10 + k) && m->branch(1, 20 + k));
      std::vector<float> one, rows;
      for (int i = 0; i < k; ++i) {
        ASSERT_TRUE(m->forward(10 + k, std::vector<std::int32_t>{dec[i]},
                               &lg, false, &err));
        one.insert(one.end(), lg.begin(), lg.end());
      }
      ASSERT_TRUE(m->forward(20 + k,
                             std::vector<std::int32_t>(dec.begin(),
                                                       dec.begin() + k),
                             &rows, /*all_rows=*/true, &err));
      const bool same = rows.size() == one.size() &&
                        std::memcmp(rows.data(), one.data(),
                                    one.size() * sizeof(float)) == 0;
      all_same = all_same && same;
      line += std::format(" k={}:{}", k,
                          same ? "same"
                               : std::format("{:.2e}",
                                             rows.size() == one.size()
                                                 ? rel_l2_(rows.data(),
                                                           one.data(),
                                                           one.size())
                                                 : 1.0));
      m->release(10 + k);
      m->release(20 + k);
    }
    std::printf("[dsv41] %s: verify rows vs decode rows:%s\n",
                bf16 ? "bf16" : "f16", line.c_str());
    EXPECT_TRUE(all_same);
  }
}

// A rollback forgets a verify's rejected rows exactly: forward six rows,
// keep a + 1 of them (truncate), decode two more; a context that decoded
// only the kept tokens gives those two steps' logits to the bit -- the
// window ring, the compressor's open group, its latents and the n-gram
// history all as if the rejected rows had never run. Every a, both dtypes.
// Where the handshake runs (an M5 Ultra; elsewhere this skips): a turn the
// GPU gave up on -- here abandoned on purpose -- costs a redo of the
// forward with commits, from the state the forward found (a ratio-2
// group's open row included), and nothing else: a decode step and a
// verify of 6 dropped mid-stack, then a rollback and more steps, all equal
// to the run with the handshake off, to the bit. Both prompt parities.
TEST(deepseek_v41, a_turn_given_up_is_redone)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  std::vector<std::int32_t> dec = ints_(meta, "decode_ids");
  while (dec.size() < 8) { dec.push_back(dec[dec.size() % 3]); }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  if (genai::MetalDeepseekV41Model::handshake_works(mc) == 0) {
    std::printf("[dsv41] no handshake on this GPU; nothing to redo\n");
    return;
  }
  std::vector<float> out[2];
  int redos = 0;
  for (int arm = 0; arm < 2; ++arm) {
    if (arm == 0) { ::setenv("VPIPE_DSV41_HANDSHAKE", "0", 1); }
    std::string err;
    MetalDeepseekV41Model::Options opt;
    auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                         mc, opt, &err);
    ::unsetenv("VPIPE_DSV41_HANDSHAKE");
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    ASSERT_TRUE(m->handshake() == (arm == 1));
    const int NL = m->config().n_layers;
    std::vector<float> lg;
    for (int T : {(int)ids.size(), (int)ids.size() - 1}) {
      const MetalDeepseekV41Model::CtxId cx = 10 + T % 2;
      const std::vector<std::int32_t> pr(ids.begin(), ids.begin() + T);
      ASSERT_TRUE(m->forward(cx, pr, &lg, false, &err));
      for (int i = 0; i < 2; ++i) {
        if (arm == 1 && i == 1) { m->hs_drop_next(NL / 2); }
        ASSERT_TRUE(m->forward(cx, std::vector<std::int32_t>{dec[i]}, &lg,
                               false, &err));
        out[arm].insert(out[arm].end(), lg.begin(), lg.end());
      }
      const std::vector<std::int32_t> v(dec.begin() + 2, dec.begin() + 8);
      if (arm == 1) { m->hs_drop_next(NL - 1); }
      ASSERT_TRUE(m->forward(cx, v, &lg, /*all_rows=*/true, &err));
      out[arm].insert(out[arm].end(), lg.begin(), lg.end());
      ASSERT_TRUE(m->truncate(cx, T + 4));
      for (int i = 4; i < 7; ++i) {
        ASSERT_TRUE(m->forward(cx, std::vector<std::int32_t>{dec[i]}, &lg,
                               false, &err));
        out[arm].insert(out[arm].end(), lg.begin(), lg.end());
      }
    }
    if (arm == 1) { redos = m->hs_redos(); }
  }
  const bool same = out[0].size() == out[1].size() &&
                    std::memcmp(out[0].data(), out[1].data(),
                                out[0].size() * 4) == 0;
  std::printf("[dsv41] handshake with %d forwards redone: %zu logits %s the "
              "run without it\n", redos, out[0].size(),
              same ? "equal" : "DIFFER from");
  EXPECT_TRUE(redos == 4);
  EXPECT_TRUE(same);
}

TEST(deepseek_v41, rollback_forgets_rejected_rows)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  std::vector<std::int32_t> dec = ints_(meta, "decode_ids");
  while (dec.size() < 8) { dec.push_back(dec[dec.size() % 3]); }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  for (bool bf16 : {true, false}) {
    std::string err;
    MetalDeepseekV41Model::Options opt;
    opt.use_bf16 = bf16;
    auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                         mc, opt, &err);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    std::vector<float> lg;
    // an odd prompt length: a ratio-2 group left open across the verify
    for (int T : {(int)ids.size(), (int)ids.size() - 1}) {
      const std::vector<std::int32_t> pr(ids.begin(), ids.begin() + T);
      ASSERT_TRUE(m->forward(1, pr, &lg, false, &err));
      std::string line;
      bool all_same = true;
      for (int a = 0; a <= 5; ++a) {
        ASSERT_TRUE(m->branch(1, 30) && m->branch(1, 40));
        // the verify: a garbage tail after the kept rows
        std::vector<std::int32_t> v(dec.begin(), dec.begin() + a + 1);
        while (v.size() < 6) { v.push_back(dec[7] + (int)v.size()); }
        std::vector<float> rows;
        ASSERT_TRUE(m->forward(30, v, &rows, true, &err));
        ASSERT_TRUE(m->truncate(30, T + a + 1));
        std::vector<float> x, y;
        for (int i = a + 1; i < a + 3; ++i) {
          ASSERT_TRUE(m->forward(30, std::vector<std::int32_t>{dec[i]}, &lg,
                                 false, &err));
          x.insert(x.end(), lg.begin(), lg.end());
        }
        for (int i = 0; i < a + 3; ++i) {
          ASSERT_TRUE(m->forward(40, std::vector<std::int32_t>{dec[i]}, &lg,
                                 false, &err));
          if (i > a) { y.insert(y.end(), lg.begin(), lg.end()); }
        }
        const bool same = x.size() == y.size() &&
                          std::memcmp(x.data(), y.data(),
                                      x.size() * sizeof(float)) == 0;
        all_same = all_same && same;
        line += std::format(" a={}:{}", a, same ? "same" : "DIFFERS");
        m->release(30);
        m->release(40);
      }
      std::printf("[dsv41] %s, prompt %d: rollback:%s\n",
                  bf16 ? "bf16" : "f16", T, line.c_str());
      EXPECT_TRUE(all_same);
      m->release(1);
    }
  }
}

// DSpark, the draft head, against the reference's forward_spec: after the
// prompt and after each teacher-forced decode step, a draft from that
// position -- the block's logits (the Markov bias added), its tokens and
// the confidence. A draft follows the main model's hidden whole, so a
// bf16 run's drafts carry every near-tie pick that flipped upstream (one
// flip put a synthetic's free drafts at 0.36); bf16 is held on fp32's
// picks (force_routes) to the reference's bf16 on those picks (bf16r),
// f16 -- the tight check -- free, to the reference's free bf16.
TEST(deepseek_v41, dspark_matches_reference)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  auto mo = meta.as_object();
  if (!mo.contains("bf16_vs_f32_spark_logits")) { return; }
  const double bar = mo.at("bf16_vs_f32_spark_logits").as_real();
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  const std::vector<std::int32_t> dec = ints_(meta, "decode_ids");
  const auto l32 = read_raw_<float>(fs::path(gd) / "spark_logits_f32.f32");
  const auto lr = read_raw_<float>(fs::path(gd) / "spark_logits_bf16r.f32");
  const auto i32 = read_raw_<std::int32_t>(fs::path(gd) /
                                           "spark_ids_f32.i32");
  const auto c32 = read_raw_<float>(fs::path(gd) / "spark_conf_f32.f32");
  const auto p32 = read_raw_<std::int32_t>(fs::path(gd) / "route_f32.i32");
  const auto q32 = read_raw_<std::int32_t>(fs::path(gd) / "droute_f32.i32");
  // each draft's stage picks [D, stages, B * k] (goldens that record them)
  const auto s32 =
      read_raw_<std::int32_t>(fs::path(gd) / "spark_route_f32.i32");
  const double rbar = lr.size() == l32.size()
                          ? rel_l2_(lr.data(), l32.data(), l32.size()) : 0.0;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  for (bool bf16 : {true, false}) {
    std::string err;
    MetalDeepseekV41Model::Options opt;
    opt.use_bf16 = bf16;
    opt.dspark = true;
    // the reference prefills the whole stack, and the drafts read the
    // decoder's taps: CED (the decoder on the prompt's last window) off
    opt.ced = false;
    auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                         mc, opt, &err);
    ASSERT_TRUE(m != nullptr && m->has_dspark());
    if (m == nullptr || !m->has_dspark()) {
      std::printf("[dsv41] %s\n", err.c_str());
      return;
    }
    const int B = m->config().dspark_block, NL = m->config().n_layers;
    const int KK = m->config().top_k, T = (int)ids.size();
    // fp32's picks of a forward: [layer][rows * k]
    auto picks = [&](const std::vector<std::int32_t>& r, std::size_t step,
                     int rows) {
      std::vector<std::vector<std::int32_t>> out((std::size_t)NL);
      for (int L = 0; L < NL; ++L) {
        const std::size_t o =
            (step * NL + (std::size_t)L) * (std::size_t)rows * KK;
        out[(std::size_t)L].assign(r.begin() + (std::ptrdiff_t)o,
                                   r.begin() + (std::ptrdiff_t)(o + rows *
                                                                KK));
      }
      return out;
    };
    for (bool forced : {false, true}) {
      if (forced && (!bf16 || rbar == 0.0)) { continue; }
      const MetalDeepseekV41Model::CtxId cx = forced ? 2 : 1;
      std::vector<float> lg, all, cf, allc;
      std::vector<std::int32_t> dr, alld;
      if (forced) { m->force_routes(picks(p32, 0, T)); }
      ASSERT_TRUE(m->forward(cx, ids, &lg, false, &err));
      for (std::size_t i = 0; i < dec.size(); ++i) {
        if (i > 0) {
          if (forced) { m->force_routes(picks(q32, i - 1, 1)); }
          ASSERT_TRUE(m->forward(cx, std::vector<std::int32_t>{dec[i - 1]},
                                 &lg, false, &err));
        }
        const std::size_t NS = (std::size_t)m->config().n_mtp;
        const std::size_t per =
            NS * (std::size_t)(B * m->config().dspark_top_k);
        if (forced && s32.size() == dec.size() * per) {
          // the reference's own stages route on fp32's picks too
          std::vector<std::vector<std::int32_t>> sp(NS);
          for (std::size_t k = 0; k < NS; ++k) {
            const auto at = s32.begin() + (std::ptrdiff_t)(i * per + k * per /
                                                           NS);
            sp[k].assign(at, at + (std::ptrdiff_t)(per / NS));
          }
          m->force_draft_routes(std::move(sp));
        }
        ASSERT_TRUE(m->dspark_draft(cx, dec[i], &dr, &cf, &lg, &err));
        all.insert(all.end(), lg.begin(), lg.end());
        alld.insert(alld.end(), dr.begin(), dr.end());
        allc.insert(allc.end(), cf.begin(), cf.end());
      }
      ASSERT_TRUE(all.size() == l32.size());
      if (all.size() != l32.size()) { return; }
      const double rel = rel_l2_(all.data(), l32.data(), all.size());
      // each draft's own distance: the indexer's top-k is a discrete pick
      // no forcing covers, and a near tie there moves a draft wholesale
      std::vector<double> per_draft;
      {
        const std::size_t per = all.size() / dec.size();
        std::string line;
        for (std::size_t d = 0; d < dec.size(); ++d) {
          per_draft.push_back(rel_l2_(all.data() + d * per,
                                      l32.data() + d * per, per));
          line += std::format(" {:.4f}", per_draft.back());
        }
        std::printf("[dsv41] %s DSpark%s: drafts%s\n", bf16 ? "bf16" : "f16",
                    forced ? " on fp32's picks" : "", line.c_str());
      }
      int same = 0, tot = 0;
      double cmax = 0.0;
      for (std::size_t d = 0; d < dec.size(); ++d) {
        for (int k = 0; k < B; ++k) {
          same += alld[d * B + k] == i32[d * (B + 1) + 1 + k];
          ++tot;
          cmax = std::max(cmax, (double)std::abs(allc[d * B + k] -
                                                 c32[d * B + k]));
        }
      }
      std::printf("[dsv41] %s DSpark%s: draft logits %.5f from fp32 (ref "
                  "bf16 %.5f), draft tokens %d/%d as fp32's, confidence off "
                  "by %.4f at most\n", bf16 ? "bf16" : "f16",
                  forced ? " on fp32's picks" : "", rel,
                  forced ? rbar : bar, same, tot, cmax);
      if (forced) {
        // the drafts' median at the reference's bar; one that jumped (a
        // near tie in a pick nothing forces) is reported, not held to it
        std::vector<double> q = per_draft;
        std::nth_element(q.begin(), q.begin() + (std::ptrdiff_t)q.size() / 2,
                         q.end());
        const double med = q[q.size() / 2];
        int jumped = 0;
        for (double v : per_draft) { jumped += v > 5.0 * rbar; }
        std::printf("[dsv41] bf16 DSpark on fp32's picks: median draft %.5f "
                    "(bar %.5f), %d of %zu jumped\n", med,
                    1.5 * rbar + 1e-3, jumped, per_draft.size());
        EXPECT_TRUE(med < 1.5 * rbar + 1e-3);
        EXPECT_TRUE(2 * jumped < (int)per_draft.size());
      } else if (!bf16) {
        EXPECT_TRUE(rel < bar);
      }
    }
  }
}

// A speculative decode with the DSpark head is the serial decode, token
// for token: greedy with the head's own drafts (noise on random weights:
// mostly rejected), greedy with known-right drafts carrying one planted
// error a round (every kept length 0..block happens, so every rollback),
// and both sampled (temperature, top-p, a seed) against the host
// sampler's serial loop drawing once a token -- the measured draft count
// (the default), the whole block and two drafts pinned
// (VPIPE_DSV41_DRAFTS).
TEST(deepseek_v41, dspark_decode_is_token_exact)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const std::string md = (fs::path(gd) / "model").string();
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  MetalDeepseekV41Model::Options opt;
  opt.dspark = true;
  MetalDeepseekV41ModelExec ex(md, mc, opt, &sess);
  ASSERT_TRUE(ex.valid() && ex.supports_mtp());
  if (!ex.valid() || !ex.supports_mtp()) { return; }
  auto* sx = static_cast<SpecDecodeExt*>(
      ex.query_extension(spec_decode::kExtensionId));
  ASSERT_TRUE(sx != nullptr && sx->drafter() == spec_decode::kDrafterMtp);
  const int N = 40;
  const int V = ex.model()->config().vocab;
  const std::string whole = std::to_string(ex.model()->config().dspark_block);
  const char* modes[] = {whole.c_str(), "2", nullptr};
  // the draft (DSpark's block, or chained a pass a draft) by the depth
  // (measured, or omlx's acceptance-only rule): every pair the serial
  // decode's tokens
  for (int mm = 0; mm < 4; ++mm) {
  const bool chain = (mm & 1) != 0, by_acc = (mm & 2) != 0;
  ex.set_mtp_modes(chain, by_acc);
  for (int arm = 0; arm < 12; ++arm) {
    const char* mode = modes[arm / 4];
    if (mode == nullptr) {
      ::unsetenv("VPIPE_DSV41_DRAFTS");
    } else {
      ::setenv("VPIPE_DSV41_DRAFTS", mode, 1);
    }
    const bool sampled = arm % 4 >= 2;
    const bool planted = arm % 2 == 1;
    GpuSamplerParams gp;
    SamplerParams sp;
    if (sampled) {
      gp.greedy = false;
      gp.temperature = sp.temperature = 0.8f;
      gp.top_p = sp.top_p = 0.9f;
      gp.seed = sp.seed = 1234;
    } else {
      sp.temperature = 0.0f;
    }
    // serial: prefill, then a token at a time
    const ContextId a{(std::uint32_t)(100 + mm * 20 + arm)};
    const ContextId b{(std::uint32_t)(200 + mm * 20 + arm)};
    const std::int32_t first = ex.prefill(a, ids);
    ASSERT_TRUE(first >= 0 && ex.branch_context(a, b));
    Sampler smp(sp);
    smp.prime(ids);
    smp.prime(std::span<const std::int32_t>(&first, 1));
    std::vector<std::int32_t> serial{first};
    while ((int)serial.size() < N) {
      const std::int32_t am = ex.decode_one(b, serial.back());
      ASSERT_TRUE(am >= 0);
      if (am < 0) { return; }
      const auto& lh = ex.last_logits_host();
      serial.push_back(sampled ? smp.sample(std::span<const float>(lh))
                               : am);
    }
    // speculative from the same prefill
    std::vector<std::int32_t> spec;
    int round = 0;
    if (planted) {
      ex.set_draft_override([&](std::int32_t, std::vector<std::int32_t>* d) {
        const std::size_t m = spec.size() - 1;   // the pending token's index
        for (std::size_t k = 0; k < d->size(); ++k) {
          (*d)[k] = m + 1 + k < serial.size() ? serial[m + 1 + k] : 0;
        }
        const std::size_t bad = (std::size_t)((round++ * 7) % 6);
        if (bad < d->size()) { (*d)[bad] = ((*d)[bad] + 1) % V; }
      });
    } else {
      ex.set_draft_override(nullptr);
    }
    std::function<bool(std::span<const std::int32_t>)> on =
        [&](std::span<const std::int32_t> t) {
          spec.insert(spec.end(), t.begin(), t.end());
          return true;
        };
    SpecDecodeRequest rq;
    rq.ctx = a;
    rq.first_token = first;
    rq.max_tokens = N;
    rq.sampler = gp;
    rq.prime = ids;
    rq.on_tokens = &on;
    SpecDecodeResult res;
    ASSERT_TRUE(sx->generate(rq, &res));
    std::printf("[dsv41] %s %s, %s, drafts %s: %zu tokens in %ld rounds, "
                "%ld of %ld drafts kept; %s the serial decode\n",
                chain ? "chained" : "block",
                by_acc ? "by acceptance" : "measured",
                sampled ? (planted ? "sampled, planted drafts"
                                   : "sampled, DSpark's drafts")
                        : (planted ? "greedy, planted drafts"
                                   : "greedy, DSpark's drafts"),
                mode == nullptr ? "chosen" : mode, spec.size(), res.rounds,
                res.accepted, res.drafted,
                spec == serial ? "matches" : "DIFFERS from");
    EXPECT_TRUE(spec == serial);
    EXPECT_TRUE(res.produced == N);
    // the context holds every kept token but the last, as the serial one
    EXPECT_TRUE(ex.context_seq_len(a) == ex.context_seq_len(b));
    // the whole block verified keeps most of a planted round
    if (planted && mode == whole.c_str()) {
      EXPECT_TRUE(res.accepted > res.rounds);
    }
    ex.release_context(a);
    ex.release_context(b);
  }
  }
  ::unsetenv("VPIPE_DSV41_DRAFTS");
  ex.set_draft_override(nullptr);
  ex.set_mtp_modes(false, false);
}

// The decode's kernels at the release's shapes, each dispatched `reps`
// times in one command buffer (VPIPE_DSV41_KERNEL_BENCH=1): the time a
// dispatch takes and the bytes it reads a second -- what a decode step's
// GPU time is made of. Random weights; nothing is checked.
TEST(deepseek_v41, decode_kernel_rates)
{
  using metal_compute::CommandStream;
  using metal_compute::ComputeEncoder;
  using metal_compute::ComputeFunction;
  using metal_compute::ComputeLibrary;
  using metal_compute::SharedBuffer;
  if (env_("VPIPE_DSV41_KERNEL_BENCH").empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("deepseek_v41_bf16");
  ASSERT_TRUE(lib.valid());
  auto buf = [&](std::size_t bytes, std::uint8_t fill_mod) {
    SharedBuffer b = mc->make_shared_buffer(bytes);
    auto* p = static_cast<std::uint8_t*>(b.contents());
    for (std::size_t i = 0; i < bytes; ++i) {
      p[i] = (std::uint8_t)((i * 2654435761u >> 13) % fill_mod);
    }
    return b;
  };
  const int H = 5120, I = 2304, V = 129280, reps = 50;
  // bf16 activations: small values (byte pattern of 0x3c.. high bytes)
  auto acts = [&](std::size_t n) {
    SharedBuffer b = mc->make_shared_buffer(n * 2);
    auto* p = static_cast<std::uint16_t*>(b.contents());
    for (std::size_t i = 0; i < n; ++i) {
      p[i] = (std::uint16_t)(0x3c00u + (i * 97u) % 200u);
    }
    return b;
  };
  auto time = [&](const char* what, double bytes, auto&& encode) {
    CommandStream st = mc->make_command_stream();
    ComputeEncoder enc = st.begin_compute();
    encode(enc);   // warm
    enc.end();
    std::string why;
    ASSERT_TRUE(st.commit().wait_ok(&why));
    ComputeEncoder e2 = st.begin_compute();
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) { encode(e2); }
    e2.end();
    ASSERT_TRUE(st.commit().wait_ok(&why));
    const double us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - t0)
                          .count() / reps;
    std::printf("[dsv41] kernel %-34s %8.1f us  %7.1f GB/s\n", what, us,
                bytes / us / 1e3);
  };
  // FP8 projections, one row, split along K as the model splits them (4
  // ways where a matrix has under 512 blocks of 8 rows and K >= 1024);
  // `G` batches on grid.z (wo_a's 8 groups)
  ComputeFunction g8 = lib.function("dsv41_gemv_fp8");
  ComputeFunction g8k = lib.function("dsv41_gemv_fp8_k4");
  ComputeFunction r8 = lib.function("dsv41_gemv_fp8_r8");
  ComputeFunction r8k = lib.function("dsv41_gemv_fp8_r8_k4");
  for (auto [N, K, G, nm] :
       std::vector<std::tuple<int, int, int, const char*>>{
           {32768, 1280, 1, "fp8 gemv wq_b 32768x1280"},
           {1280, 5120, 1, "fp8 gemv wq_a 1280x5120"},
           {512, 5120, 1, "fp8 gemv wkv 512x5120"},
           {1024, 4096, 8, "fp8 gemv wo_a 8 x 1024x4096"},
           {5120, 8192, 1, "fp8 gemv wo_b 5120x8192"},
           {4608, 5120, 1, "fp8 gemv shared w1|w3 4608x5120"},
           {5120, 2304, 1, "fp8 gemv shared w2 5120x2304"},
           {8192, 1280, 1, "fp8 gemv index wq_b 8192x1280"}}) {
    const std::size_t nk = (std::size_t)N * K * G;
    SharedBuffer w = buf(nk, 120), sc = buf(nk / 32, 4);
    auto* sp = static_cast<std::uint8_t*>(sc.contents());
    for (std::size_t i = 0; i < nk / 32; ++i) { sp[i] = 120; }
    SharedBuffer x = acts((std::size_t)K * G), y = acts((std::size_t)N * G);
    const int ks = (N + 7) / 8 < 512 && K >= 1024 ? 4 : 1;
    for (int map = 0; map < 2; ++map) {
      const std::string label = std::string(nm) + (map ? " r8" : "");
      time(label.c_str(), (double)nk * 33 / 32, [&](ComputeEncoder& enc) {
        enc.set_function(map ? (ks == 4 ? r8k : r8) : (ks == 4 ? g8k : g8));
        enc.set_buffer(0, w);
        enc.set_buffer(1, sc);
        enc.set_buffer(2, x);
        enc.set_buffer(3, y);
        enc.set_constant(4, K);
        enc.set_constant(5, N);
        enc.set_constant(6, 1);
        enc.set_constant(7, (long)K);
        enc.set_constant(8, (long)N * K);
        enc.set_constant(9, (long)N);
        enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2 * ks), (unsigned)G},
                     {32, (unsigned)(2 * ks), 1});
      });
    }
  }
  // GEMVs over bf16 weights (f32 or bf16 out), as the model splits them,
  // across the row and 8 lanes a row
  for (auto [N, K, G, f32, nm] :
       std::vector<std::tuple<int, int, int, bool, const char*>>{
           {384, 5120, 1, true, "bf16 gemv router 384x5120"},
           {1024, 5120, 1, true, "bf16 gemv compressor 1024x5120"},
           {64, 5120, 1, false, "bf16 gemv index wp 64x5120"},
           {1024, 4096, 8, false, "bf16 gemv wo_a 8 x 1024x4096"},
           {129280, 5120, 1, true, "bf16 gemv head 129280x5120"}}) {
    const std::size_t nk = (std::size_t)N * K * G;
    SharedBuffer w = acts(nk), x = acts((std::size_t)K * G);
    SharedBuffer y = mc->make_shared_buffer((std::size_t)N * G * 4);
    const int ks = (N + 7) / 8 < 512 && K >= 1024 ? 4 : 1;
    for (int map = 0; map < 2; ++map) {
      const std::string fn =
          std::string(f32 ? "dsv41_gemv_f32" : "dsv41_gemv_one") +
          (map ? "_r8" : "") + (ks == 4 ? "_k4" : "");
      ComputeFunction gf = lib.function(fn);
      ASSERT_TRUE(gf.valid());
      const std::string label = std::string(nm) + (map ? " r8" : "");
      time(label.c_str(), (double)nk * 2, [&](ComputeEncoder& enc) {
        enc.set_function(gf);
        enc.set_buffer(0, x);
        enc.set_buffer(1, w);
        enc.set_buffer(2, y);
        enc.set_constant(3, K);
        enc.set_constant(4, N);
        enc.set_constant(5, 1);
        enc.set_constant(6, (long)K);
        enc.set_constant(7, (long)N * K);
        enc.set_constant(8, (long)N);
        enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2 * ks), (unsigned)G},
                     {32, (unsigned)(2 * ks), 1});
      });
    }
  }
  // a dense bf16 projection's one row (the MLX pack's wo_a, a group)
  {
    ComputeFunction g1 = lib.function("dsv41_gemv_one");
    const int N = 1024, K = 4096;
    SharedBuffer w = acts((std::size_t)N * K), x = acts(K), y = acts(N);
    time("bf16 gemv wo_a group 1024x4096", (double)N * K * 2,
         [&](ComputeEncoder& enc) {
           enc.set_function(g1);
           enc.set_buffer(0, x);
           enc.set_buffer(1, w);
           enc.set_buffer(2, y);
           enc.set_constant(3, K);
           enc.set_constant(4, N);
           enc.set_constant(5, 1);
           enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
         });
  }
  // the head, f32 out
  {
    ComputeFunction gf = lib.function("dsv41_gemv_f32");
    SharedBuffer w = acts((std::size_t)V * H), x = acts(H);
    SharedBuffer y = mc->make_shared_buffer((std::size_t)V * 4);
    time("f32 gemv head 129280x5120", (double)V * H * 2,
         [&](ComputeEncoder& enc) {
           enc.set_function(gf);
           enc.set_buffer(0, x);
           enc.set_buffer(1, w);
           enc.set_buffer(2, y);
           enc.set_constant(3, H);
           enc.set_constant(4, V);
           enc.set_constant(5, 1);
           enc.dispatch({32, (unsigned)(((V + 7) / 8) * 2), 1}, {32, 2, 1});
         });
  }
  // routed experts, 6 pairs of one row over 8 slots
  {
    const int E = 8, np = 6;
    ComputeFunction sw = lib.function("dsv41_moe_swiglu_fp4");
    ComputeFunction dn = lib.function("dsv41_moe_down_fp4");
    SharedBuffer w1 = buf((std::size_t)E * I * H / 2, 255);
    SharedBuffer s1 = buf((std::size_t)E * I * H / 32, 1);
    SharedBuffer w3 = buf((std::size_t)E * I * H / 2, 255);
    SharedBuffer s3 = buf((std::size_t)E * I * H / 32, 1);
    SharedBuffer w2 = buf((std::size_t)E * H * I / 2, 255);
    SharedBuffer s2 = buf((std::size_t)E * H * I / 32, 1);
    for (SharedBuffer* b : {&s1, &s3, &s2}) {
      auto* p = static_cast<std::uint8_t*>(b->contents());
      for (std::size_t i = 0; i < b->byte_size(); ++i) { p[i] = 120; }
    }
    SharedBuffer x = acts(H), act = acts((std::size_t)np * I);
    SharedBuffer part = acts((std::size_t)np * H);
    SharedBuffer slot = mc->make_shared_buffer(np * 4);
    SharedBuffer wt = mc->make_shared_buffer(np * 4);
    for (int p = 0; p < np; ++p) {
      static_cast<std::int32_t*>(slot.contents())[p] = p;
      static_cast<float*>(wt.contents())[p] = 0.25f;
    }
    const double eb = (double)I * H / 2 * 33 / 16;
    time("fp4 experts gate|up, 6 pairs", np * 2 * eb / 3 * 1.5,
         [&](ComputeEncoder& enc) {
           enc.set_function(sw);
           enc.set_buffer(0, w1);
           enc.set_buffer(1, s1);
           enc.set_buffer(2, w3);
           enc.set_buffer(3, s3);
           enc.set_buffer(4, x);
           enc.set_buffer(5, act);
           enc.set_buffer(6, slot);
           enc.set_buffer(7, wt);
           enc.set_constant(8, H);
           enc.set_constant(9, I);
           enc.set_constant(10, np);
           enc.set_constant(11, 10.0f);
           enc.dispatch({32, (unsigned)((I / 4) * 2), (unsigned)np},
                        {32, 2, 1});
         });
    time("fp4 experts down, 6 pairs", (double)np * H * I / 2 * 33 / 32,
         [&](ComputeEncoder& enc) {
           enc.set_function(dn);
           enc.set_buffer(0, w2);
           enc.set_buffer(1, s2);
           enc.set_buffer(2, act);
           enc.set_buffer(3, part);
           enc.set_buffer(4, slot);
           enc.set_constant(5, I);
           enc.set_constant(6, H);
           enc.dispatch({32, (unsigned)((H / 8) * 2), (unsigned)np},
                        {32, 2, 1});
         });
  }
  // What a dependent dispatch costs by itself: a decode step is ~900 of
  // them in a row. 16 words filled, one after another (the encoder's
  // serial order), and 32 at a time with no order between them.
  {
    ComputeFunction fill = lib.function("dsv41_fill32");
    SharedBuffer b = mc->make_shared_buffer(64 * 4);
    time("an empty dispatch, serial", 64.0, [&](ComputeEncoder& enc) {
      enc.set_function(fill);
      enc.set_buffer(0, b);
      enc.set_constant(1, 1);
      enc.set_constant(2, 16);
      enc.dispatch({16, 1, 1}, {16, 1, 1});
    });
    // a chain of 32 dependent ones three ways: the serial encoder's own
    // ordering, and a concurrent encoder ordered by a barrier (over every
    // buffer, or over the one written)
    for (int how = 0; how < 3; ++how) {
      CommandStream st = mc->make_command_stream();
      auto chain = [&](ComputeEncoder& enc) {
        for (int i = 0; i < 32; ++i) {
          enc.set_function(fill);
          enc.set_buffer(0, b);
          enc.set_constant(1, i);
          enc.set_constant(2, 16);
          enc.dispatch({16, 1, 1}, {16, 1, 1});
          if (how == 1) { enc.memory_barrier(); }
          if (how == 2) { enc.memory_barrier_buffer(b); }
        }
      };
      const auto dt = how == 0 ? metal_compute::DispatchType::Serial
                               : metal_compute::DispatchType::Concurrent;
      {
        ComputeEncoder w = st.begin_compute(dt);
        chain(w);
        w.end();
        std::string why;
        ASSERT_TRUE(st.commit().wait_ok(&why));
      }
      ComputeEncoder e2 = st.begin_compute(dt);
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < 20; ++r) { chain(e2); }
      e2.end();
      std::string why;
      ASSERT_TRUE(st.commit().wait_ok(&why));
      const double us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - t0).count() /
                        (20 * 32);
      static const char* kHow[3] = {"serial encoder", "concurrent + barrier",
                                    "concurrent + buffer barrier"};
      std::printf("[dsv41] kernel a dependent empty dispatch, %-26s %6.2f "
                  "us\n", kHow[how], us);
    }
    time("32 empty dispatches, one concurrent group", 64.0 * 32,
         [&](ComputeEncoder& enc) {
           auto cs = enc.concurrent_scope();
           for (int i = 0; i < 32; ++i) {
             enc.set_function(fill);
             enc.set_buffer(0, b, (std::size_t)(i % 4) * 64);
             enc.set_constant(1, i);
             enc.set_constant(2, 16);
             enc.dispatch({16, 1, 1}, {16, 1, 1});
           }
         });
  }
  // A layer's small steps at one row, as the model dispatches them: the
  // hyper-connection mixes (twice a layer), the norms and rotations, the
  // window ring, the route, the shared expert's SwiGLU, the combine.
  {
    const int HC4 = 4 * H, MIX = 24, QL = 1280, D = 512, NH = 64, rd = 64;
    SharedBuffer X = acts((std::size_t)HC4), a = acts(H), y = acts(H);
    SharedBuffer w = acts(H);
    auto f32buf = [&](std::size_t n, float v) {
      SharedBuffer b = mc->make_shared_buffer(n * 4);
      auto* p = static_cast<float*>(b.contents());
      for (std::size_t i = 0; i < n; ++i) { p[i] = v; }
      return b;
    };
    SharedBuffer fnT = f32buf((std::size_t)HC4 * MIX, 0.001f);
    SharedBuffer part = f32buf((std::size_t)128 * (MIX + 1), 0.0f);
    SharedBuffer hs = f32buf(3, 1.0f), hb = f32buf(MIX, 0.0f);
    SharedBuffer pre = f32buf(4, 0.25f), post = f32buf(4, 0.5f);
    SharedBuffer comb = f32buf(16, 0.25f);
    ComputeFunction fpart = lib.function("dsv41_hc_mix_part");
    ComputeFunction fmix = lib.function("dsv41_hc_mix");
    ComputeFunction fpp = lib.function("dsv41_hc_post_pre");
    ComputeFunction frms = lib.function("dsv41_rms_norm");
    ComputeFunction frope = lib.function("dsv41_rope");
    ComputeFunction fring = lib.function("dsv41_ring_write");
    ComputeFunction froute = lib.function("dsv41_route");
    ComputeFunction fsw = lib.function("dsv41_swiglu_clamp");
    ComputeFunction fcomb = lib.function("dsv41_moe_combine");
    for (int S : {16, 64}) {
      time(std::format("hc mix part, {} slices", S).c_str(),
           (double)HC4 * MIX * 4, [&](ComputeEncoder& enc) {
             enc.set_function(fpart);
             enc.set_buffer(0, X);
             enc.set_buffer(1, fnT);
             enc.set_buffer(2, part);
             enc.set_constant(3, HC4);
             enc.set_constant(4, S);
             enc.dispatch({(unsigned)(256 * S), 1, 1}, {256, 1, 1});
           });
    }
    time("hc mix (Sinkhorn)", 0.0, [&](ComputeEncoder& enc) {
      enc.set_function(fmix);
      enc.set_buffer(0, part);
      enc.set_buffer(1, hs);
      enc.set_buffer(2, hb);
      enc.set_buffer(3, pre);
      enc.set_buffer(4, post);
      enc.set_buffer(5, comb);
      enc.set_constant(6, HC4);
      enc.set_constant(7, 1e-6f);
      enc.set_constant(8, 1e-6f);
      enc.set_constant(9, 20);
      enc.set_constant(10, 16);
      enc.set_constant(11, 1);
      enc.dispatch({32, 1, 1}, {32, 1, 1});
    });
    time("hc post + collapse + norm", (double)HC4 * 2 * 2,
         [&](ComputeEncoder& enc) {
           enc.set_function(fpp);
           enc.set_buffer(0, a);
           enc.set_buffer(1, X);
           enc.set_buffer(2, pre);
           enc.set_buffer(3, post);
           enc.set_buffer(4, comb);
           enc.set_buffer(5, w);
           enc.set_buffer(6, y);
           enc.set_constant(7, H);
           enc.set_constant(8, 1e-6f);
           enc.dispatch({1024, 1, 1}, {1024, 1, 1});
         });
    {
      // the same over a group a 1024 (dsv41_hc_collapse): the post of a
      // branch, of the MoE's 6 parts + the shared expert's, none
      ComputeFunction fco = lib.function("dsv41_hc_collapse");
      const int G = (H + 1023) / 1024, KP = 6;
      SharedBuffer prt = acts((std::size_t)KP * H), shx = acts(H);
      SharedBuffer ps = f32buf((std::size_t)G, 0.0f);
      SharedBuffer cnt = mc->make_shared_buffer(16);
      std::memset(cnt.contents(), 0, 16);
      for (int kk : {0, KP, -1}) {
        time(kk == 0    ? "hc post + collapse + norm, groups"
             : kk > 0   ? "hc moe post + collapse + norm, groups"
                        : "hc collapse + norm, groups",
             (double)HC4 * 2 * (kk >= 0 ? 2 : 1),
             [&](ComputeEncoder& enc) {
               enc.set_function(fco);
               enc.set_buffer(0, kk > 0 ? prt : a);
               enc.set_buffer(1, X);
               enc.set_buffer(2, pre);
               enc.set_buffer(3, post);
               enc.set_buffer(4, comb);
               enc.set_buffer(5, w);
               enc.set_buffer(6, y);
               enc.set_constant(7, H);
               enc.set_constant(8, 1e-6f);
               enc.set_buffer(9, shx);
               enc.set_constant(10, kk);
               enc.set_constant(11, 1);
               enc.set_buffer(12, ps);
               enc.set_buffer(13, cnt);
               enc.dispatch({(unsigned)(1024 * G), 1, 1}, {1024, 1, 1});
             });
      }
    }
    for (int C : {QL, D}) {
      SharedBuffer xv = acts(C), wv = acts(C), yv = acts(C);
      time(std::format("rms norm {}", C).c_str(), (double)C * 6,
           [&](ComputeEncoder& enc) {
             enc.set_function(frms);
             enc.set_buffer(0, xv);
             enc.set_buffer(1, wv);
             enc.set_buffer(2, yv);
             enc.set_constant(3, C);
             enc.set_constant(4, C);
             enc.set_constant(5, C);
             enc.set_constant(6, 1e-6f);
             enc.set_constant(7, 0);
             enc.dispatch({256, 1, 1}, {256, 1, 1});
           });
    }
    SharedBuffer tab = f32buf((std::size_t)4096 * rd, 0.5f);
    for (int heads : {NH, 1}) {
      SharedBuffer q = acts((std::size_t)heads * D);
      time(std::format("rope, {} heads", heads).c_str(),
           (double)heads * rd * 4, [&](ComputeEncoder& enc) {
             enc.set_function(frope);
             enc.set_buffer(0, q);
             enc.set_buffer(1, tab);
             enc.set_constant(2, heads * D);
             enc.set_constant(3, D);
             enc.set_constant(4, D);
             enc.set_constant(5, rd);
             enc.set_constant(6, 100);
             enc.set_constant(7, 1);
             enc.set_constant(8, 0);
             enc.dispatch({(unsigned)(rd / 2), (unsigned)heads, 1},
                          {(unsigned)(rd / 2), 1, 1});
           });
    }
    {
      SharedBuffer kv = acts(D), ring = acts((std::size_t)136 * D);
      time("ring write", (double)D * 4, [&](ComputeEncoder& enc) {
        enc.set_function(fring);
        enc.set_buffer(0, kv);
        enc.set_buffer(1, ring);
        enc.set_constant(2, 1);
        enc.set_constant(3, D);
        enc.set_constant(4, 136);
        enc.set_constant(5, 100);
        enc.dispatch({(unsigned)D, 1, 1}, {256, 1, 1});
      });
    }
    {
      const int E = 384, K = 6;
      SharedBuffer sc = f32buf(E, 0.1f), bias = f32buf(E, 0.0f);
      SharedBuffer vl = mc->make_shared_buffer(16);
      std::memset(vl.contents(), 0, 16);
      SharedBuffer eid = mc->make_shared_buffer(K * 4), wt = f32buf(K, 0);
      time("route, top 6 of 384", (double)E * 4, [&](ComputeEncoder& enc) {
        enc.set_function(froute);
        enc.set_buffer(0, sc);
        enc.set_buffer(1, bias);
        enc.set_buffer(2, bias);
        enc.set_buffer(3, vl);
        enc.set_buffer(4, eid);
        enc.set_buffer(5, wt);
        enc.set_constant(6, E);
        enc.set_constant(7, K);
        enc.set_constant(8, 1.5f);
        enc.set_constant(9, 1);
        enc.set_constant(10, 0);
        enc.dispatch({256, 1, 1}, {256, 1, 1});
      });
      SharedBuffer gu = acts((std::size_t)2 * I), sa = acts(I);
      time("shared SwiGLU", (double)I * 6, [&](ComputeEncoder& enc) {
        enc.set_function(fsw);
        enc.set_buffer(0, gu);
        enc.set_buffer(1, sa);
        enc.set_constant(2, I);
        enc.set_constant(3, 10.0f);
        enc.dispatch({(unsigned)I, 1, 1}, {256, 1, 1});
      });
      SharedBuffer prt = acts((std::size_t)K * H), sh = acts(H);
      time("combine 6 + shared", (double)(K + 2) * H * 2,
           [&](ComputeEncoder& enc) {
             enc.set_function(fcomb);
             enc.set_buffer(0, prt);
             enc.set_buffer(1, sh);
             enc.set_buffer(2, y);
             enc.set_constant(3, H);
             enc.set_constant(4, K);
             enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
           });
    }
  }
  // An oQ pack's affine codes (oQ2e: projections at 8 bits, experts at
  // 2; oQ3e: projections at 4-8, experts at 3), a bf16 scale and bias per
  // 64: the projections' one-row GEMVs, and the experts' halves.
  {
    const int gs = 64;
    for (int bits : {8, 6, 4}) {
      const std::string b = std::to_string(bits);
      ComputeFunction g1 = lib.function("dsv41_gemv_q" + b);
      ComputeFunction g1k = lib.function("dsv41_gemv_q" + b + "_k4");
      ASSERT_TRUE(g1.valid() && g1k.valid());
      for (auto [N, K, nm] :
           std::vector<std::tuple<int, int, const char*>>{
               {32768, 1280, "wq_b 32768x1280"},
               {1280, 5120, "wq_a 1280x5120"},
               {512, 5120, "wkv 512x5120"},
               {5120, 8192, "wo_b 5120x8192"},
               {4608, 5120, "shared w1|w3 4608x5120"},
               {5120, 2304, "shared w2 5120x2304"},
               {4096, 1280, "index wq_b 4096x1280"}}) {
        const std::size_t nk = (std::size_t)N * K;
        const std::size_t cb = nk * bits / 8, sb = nk / gs * 2;
        SharedBuffer w = buf(cb, 255), sc = acts(nk / gs), bi = acts(nk / gs);
        SharedBuffer x = acts(K), y = acts(N);
        const int ks = (N + 7) / 8 < 512 && K >= 1024 ? 4 : 1;
        const std::string label = "q" + b + " gemv " + nm;
        time(label.c_str(), (double)(cb + 2 * sb), [&](ComputeEncoder& enc) {
          enc.set_function(ks == 4 ? g1k : g1);
          enc.set_buffer(0, w);
          enc.set_buffer(1, sc);
          enc.set_buffer(2, bi);
          enc.set_buffer(3, x);
          enc.set_buffer(4, y);
          enc.set_constant(5, K);
          enc.set_constant(6, N);
          enc.set_constant(7, 1);
          enc.set_constant(8, gs);
          enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2 * ks), 1},
                       {32, (unsigned)(2 * ks), 1});
        });
      }
    }
    // the experts: 6 pairs of one row over 8 slots
    const int E = 8, np = 6;
    SharedBuffer x = acts(H), act = acts((std::size_t)np * I);
    SharedBuffer part = acts((std::size_t)np * H);
    SharedBuffer slot = mc->make_shared_buffer(np * 4);
    SharedBuffer wt = mc->make_shared_buffer(np * 4);
    for (int p = 0; p < np; ++p) {
      static_cast<std::int32_t*>(slot.contents())[p] = p;
      static_cast<float*>(wt.contents())[p] = 0.25f;
    }
    for (int bits : {2, 3, 4}) {
      const std::string b = std::to_string(bits);
      ComputeFunction sw = lib.function("dsv41_moe_swiglu_q" + b);
      ComputeFunction dn = lib.function("dsv41_moe_down_q" + b);
      ASSERT_TRUE(sw.valid() && dn.valid());
      const std::size_t nk = (std::size_t)I * H;
      const long cw = (long)(nk * bits / 8), cs = (long)(nk / gs);
      SharedBuffer w1 = buf((std::size_t)E * cw, 255);
      SharedBuffer w3 = buf((std::size_t)E * cw, 255);
      SharedBuffer w2 = buf((std::size_t)E * cw, 255);
      const std::size_t ns = (std::size_t)E * cs;
      SharedBuffer s1 = acts(ns), b1 = acts(ns), s3 = acts(ns);
      SharedBuffer b3 = acts(ns), s2 = acts(ns), b2 = acts(ns);
      const double half = (double)np * (cw + 4.0 * cs);
      time(("q" + b + " experts gate|up, 6 pairs").c_str(), 2 * half,
           [&](ComputeEncoder& enc) {
             enc.set_function(sw);
             enc.set_buffer(0, w1);
             enc.set_buffer(1, s1);
             enc.set_buffer(2, b1);
             enc.set_buffer(3, w3);
             enc.set_buffer(4, s3);
             enc.set_buffer(5, b3);
             enc.set_buffer(6, x);
             enc.set_buffer(7, act);
             enc.set_buffer(8, slot);
             enc.set_buffer(9, wt);
             enc.set_constant(10, H);
             enc.set_constant(11, I);
             enc.set_constant(12, np);
             enc.set_constant(13, 10.0f);
             enc.set_constant(14, gs);
             enc.set_constant(15, cw);
             enc.set_constant(16, cs);
             enc.dispatch({32, (unsigned)((I / 8) * 2), (unsigned)np},
                          {32, 2, 1});
           });
      time(("q" + b + " experts down, 6 pairs").c_str(), half,
           [&](ComputeEncoder& enc) {
             enc.set_function(dn);
             enc.set_buffer(0, w2);
             enc.set_buffer(1, s2);
             enc.set_buffer(2, b2);
             enc.set_buffer(3, act);
             enc.set_buffer(4, part);
             enc.set_buffer(5, slot);
             enc.set_constant(6, I);
             enc.set_constant(7, H);
             enc.set_constant(8, gs);
             enc.set_constant(9, cw);
             enc.set_constant(10, cs);
             enc.dispatch({32, (unsigned)((H / 8) * 2), (unsigned)np},
                          {32, 2, 1});
           });
    }
  }
  // a compressed layer's attention, one row: 64 heads over a 128-key
  // window and 512 picks, a simdgroup a head against split by keys
  {
    const int NH = 64, D = 512, W = 128, KC = 512, R = 136, C = 4096;
    const int CH = (W + KC + 15) / 16, rd = 64, pos = 5000;
    SharedBuffer q = acts((std::size_t)NH * D), kvn = acts(D);
    SharedBuffer ring = acts((std::size_t)R * D);
    SharedBuffer comp = acts((std::size_t)C * D);
    SharedBuffer idx = mc->make_shared_buffer((std::size_t)KC * 4);
    for (int j = 0; j < KC; ++j) {
      static_cast<std::int32_t*>(idx.contents())[j] = (j * 37) % (pos / 2);
    }
    SharedBuffer sink = mc->make_shared_buffer((std::size_t)NH * 4);
    std::memset(sink.contents(), 0, (std::size_t)NH * 4);
    SharedBuffer o = acts((std::size_t)NH * D);
    SharedBuffer tab = mc->make_shared_buffer((std::size_t)(pos + 1) * rd * 4);
    std::memset(tab.contents(), 0, tab.byte_size());
    SharedBuffer part = mc->make_shared_buffer((std::size_t)NH * CH * D * 4);
    SharedBuffer ml = mc->make_shared_buffer((std::size_t)NH * CH * 8);
    ComputeFunction fa = lib.function("dsv41_attn");
    ComputeFunction fs = lib.function("dsv41_attn_split");
    ComputeFunction fm = lib.function("dsv41_attn_merge");
    const double kb = (double)(W + KC) * D * 2;
    time("attention, a simdgroup a head", kb, [&](ComputeEncoder& enc) {
      enc.set_function(fa);
      enc.set_buffer(0, q);
      enc.set_buffer(1, kvn);
      enc.set_buffer(2, ring);
      enc.set_buffer(3, comp);
      enc.set_buffer(4, idx);
      enc.set_buffer(5, sink);
      enc.set_buffer(6, o);
      enc.set_buffer(7, tab);
      enc.set_constant(8, 1);
      enc.set_constant(9, NH);
      enc.set_constant(10, W);
      enc.set_constant(11, pos);
      enc.set_constant(12, KC);
      enc.set_constant(13, 2);
      enc.set_constant(14, 0.044f);
      enc.set_constant(15, 8);
      enc.set_constant(16, rd);
      enc.set_constant(17, R);
      enc.set_constant(18, 0);
      enc.dispatch({256 * 8, 1, 1}, {256, 1, 1});
    });
    time("attention, split by keys + merge", kb, [&](ComputeEncoder& enc) {
      enc.set_function(fs);
      enc.set_buffer(0, q);
      enc.set_buffer(1, kvn);
      enc.set_buffer(2, ring);
      enc.set_buffer(3, comp);
      enc.set_buffer(4, idx);
      enc.set_buffer(5, part);
      enc.set_buffer(6, ml);
      enc.set_constant(7, NH);
      enc.set_constant(8, W);
      enc.set_constant(9, pos);
      enc.set_constant(10, KC);
      enc.set_constant(11, 2);
      enc.set_constant(12, 0.044f);
      enc.set_constant(13, R);
      enc.set_constant(14, CH);
      enc.set_buffer(15, tab);
      enc.set_constant(16, 0);
      enc.dispatch({512 * 4, (unsigned)CH, 1}, {512, 1, 1});
      enc.set_function(fm);
      enc.set_buffer(0, part);
      enc.set_buffer(1, ml);
      enc.set_buffer(2, sink);
      enc.set_buffer(3, o);
      enc.set_buffer(4, tab);
      enc.set_constant(5, 1);
      enc.set_constant(6, NH);
      enc.set_constant(7, pos);
      enc.set_constant(8, CH);
      enc.set_constant(9, 8);
      enc.set_constant(10, rd);
      enc.dispatch({256 * (unsigned)NH, 1, 1}, {256, 1, 1});
    });
  }
  // the indexer at depth, one row: scores over C compressed positions
  // (all, or a candidate source's 16K), the top 512 of them
  {
    ComputeFunction fsc = lib.function("dsv41_index_score");
    ComputeFunction fsm = lib.function("dsv41_index_score_mma");
    ComputeFunction ftk = lib.function("dsv41_topk");
    ComputeFunction ftp = lib.function("dsv41_topk_par");
    ComputeFunction fsk = lib.function("dsv41_index_score_kept");
    ComputeFunction ftc = lib.function("dsv41_topk_chunk");
    ComputeFunction ftm = lib.function("dsv41_topk_merge");
    ComputeFunction fun = lib.function("dsv41_index_unpack");
    const int IH = 32, ID = 128;
    for (int C : {16384, 65536, 131072, 262144}) {
      SharedBuffer q = acts((std::size_t)IH * ID), w = acts(IH);
      SharedBuffer keys = acts((std::size_t)C * ID);
      SharedBuffer S = mc->make_shared_buffer((std::size_t)C * 4);
      SharedBuffer out = mc->make_shared_buffer(512 * 4);
      const int blk = 8, nblk = (C + blk - 1) / blk;
      SharedBuffer cand = mc->make_shared_buffer((std::size_t)nblk);
      auto* cp = static_cast<std::uint8_t*>(cand.contents());
      for (int b = 0; b < nblk; ++b) { cp[b] = b < 2048 ? 1 : 0; }
      for (int use_cand = 0; use_cand < 2; ++use_cand) {
        const std::string nm = std::format("index score C={}{}", C,
                                           use_cand ? " (16K kept)" : "");
        for (int mm = 0; mm < 2; ++mm) {
        const std::string nm2 = nm + (mm ? " mma" : "");
        time(nm2.c_str(), (double)C * ID * 2, [&](ComputeEncoder& enc) {
          enc.set_function(mm ? fsm : fsc);
          enc.set_buffer(0, q);
          enc.set_buffer(1, keys);
          enc.set_buffer(2, w);
          enc.set_buffer(3, cand);
          enc.set_buffer(4, S);
          enc.set_constant(5, IH);
          enc.set_constant(6, C);
          enc.set_constant(7, 2 * C);
          enc.set_constant(8, 2);
          enc.set_constant(9, 0.01f);
          enc.set_constant(10, use_cand);
          enc.set_constant(11, blk);
          enc.set_constant(12, nblk);
          if (mm) {
            enc.dispatch({(unsigned)(128 * ((C + 63) / 64)), 1, 1},
                         {128, 1, 1});
          } else {
            enc.dispatch({(unsigned)(256 * ((C + 255) / 256)), 1, 1},
                         {256, 1, 1});
          }
        });
        }
      }
      for (int par = 0; par < 2; ++par) {
      const std::string nm = std::format("top 512 of C={}{}", C,
                                         par ? " parallel" : "");
      time(nm.c_str(), (double)C * 4, [&](ComputeEncoder& enc) {
        enc.set_function(par ? ftp : ftk);
        enc.set_buffer(0, S);
        enc.set_buffer(1, out);
        enc.set_constant(2, C);
        enc.set_constant(3, 512);
        enc.dispatch({1024, 1, 1}, {1024, 1, 1});
      });
      }
      // the same across the cores: chunks, then the merge
      for (int K : {512, 2048}) {
        const int G = (C + 4095) / 4096;
        const int R = std::clamp(3 * K / 2 / G, 1, std::min(K, 3072 / G));
        SharedBuffer ck = mc->make_shared_buffer((std::size_t)G * K * 4);
        SharedBuffer cpos = mc->make_shared_buffer((std::size_t)G * K * 4);
        SharedBuffer tk = mc->make_shared_buffer((std::size_t)G * 4);
        SharedBuffer tr = mc->make_shared_buffer((std::size_t)G * R * 4);
        SharedBuffer o2 = mc->make_shared_buffer((std::size_t)K * 4);
        const unsigned NT = 256;
        const std::string nm = std::format("top {} of C={} two-phase", K,
                                           C);
        time(nm.c_str(), (double)C * 4, [&](ComputeEncoder& enc) {
          enc.set_function(ftc);
          enc.set_buffer(0, S);
          enc.set_buffer(1, ck);
          enc.set_buffer(2, cpos);
          enc.set_buffer(3, tk);
          enc.set_buffer(4, tr);
          enc.set_constant(5, C);
          enc.set_constant(6, K);
          enc.set_constant(7, R);
          enc.dispatch({NT * (unsigned)G, 1, 1}, {NT, 1, 1});
          enc.set_function(ftm);
          enc.set_buffer(0, ck);
          enc.set_buffer(1, cpos);
          enc.set_buffer(2, tk);
          enc.set_buffer(3, tr);
          enc.set_buffer(4, o2);
          enc.set_constant(5, C);
          enc.set_constant(6, K);
          enc.set_constant(7, R);
          enc.dispatch({NT, 1, 1}, {NT, 1, 1});
        });
        for (int ph = 0; ph < 2; ++ph) {
          const std::string n1 = std::format("top {} of C={} {} alone", K,
                                             C, ph ? "merge" : "chunks");
          time(n1.c_str(), (double)C * 4, [&](ComputeEncoder& enc) {
            if (ph == 0) {
              enc.set_function(ftc);
              enc.set_buffer(0, S);
              enc.set_buffer(1, ck);
              enc.set_buffer(2, cpos);
              enc.set_buffer(3, tk);
              enc.set_buffer(4, tr);
              enc.set_constant(5, C);
              enc.set_constant(6, K);
              enc.set_constant(7, R);
              enc.dispatch({NT * (unsigned)G, 1, 1}, {NT, 1, 1});
            } else {
              enc.set_function(ftm);
              enc.set_buffer(0, ck);
              enc.set_buffer(1, cpos);
              enc.set_buffer(2, tk);
              enc.set_buffer(3, tr);
              enc.set_buffer(4, o2);
              enc.set_constant(5, C);
              enc.set_constant(6, K);
              enc.set_constant(7, R);
              enc.dispatch({NT, 1, 1}, {NT, 1, 1});
            }
          });
        }
        if (K == 2048) {
          const std::string n1 = std::format("top 2048 of C={} parallel", C);
          time(n1.c_str(), (double)C * 4, [&](ComputeEncoder& enc) {
            enc.set_function(ftp);
            enc.set_buffer(0, S);
            enc.set_buffer(1, o2);
            enc.set_constant(2, C);
            enc.set_constant(3, 2048);
            enc.dispatch({1024, 1, 1}, {1024, 1, 1});
          });
        }
      }
      // a layer past the candidate source, whole: every position scored
      // under the 2048 blocks' mask and the top 512 of all C, against the
      // blocks' positions alone, their top 512 and the unpack
      const int KB = 2048, R = KB * blk;
      if (nblk <= KB) { continue; }
      SharedBuffer bpick = mc->make_shared_buffer((std::size_t)KB * 4);
      auto* bp = static_cast<std::int32_t*>(bpick.contents());
      for (int k = 0; k < KB; ++k) { bp[k] = k; }
      for (int kept = 0; kept < 2; ++kept) {
        const std::string nm = std::format(
            "past the source C={}{}", C, kept ? " kept" : " masked");
        time(nm.c_str(), (double)(kept ? R : C) * ID * 2,
             [&](ComputeEncoder& enc) {
          enc.set_function(kept ? fsk : fsm);
          enc.set_buffer(0, q);
          enc.set_buffer(1, keys);
          enc.set_buffer(2, w);
          enc.set_buffer(3, kept ? bpick : cand);
          enc.set_buffer(4, S);
          enc.set_constant(5, IH);
          enc.set_constant(6, C);
          enc.set_constant(7, 2 * C);
          enc.set_constant(8, 2);
          enc.set_constant(9, 0.01f);
          if (kept) {
            enc.set_constant(10, KB);
            enc.set_constant(11, blk);
          } else {
            enc.set_constant(10, 1);
            enc.set_constant(11, blk);
            enc.set_constant(12, nblk);
          }
          enc.dispatch({(unsigned)(128 * (((kept ? R : C) + 63) / 64)), 1,
                        1}, {128, 1, 1});
          enc.set_function(ftp);
          enc.set_buffer(0, S);
          enc.set_buffer(1, out);
          enc.set_constant(2, kept ? R : C);
          enc.set_constant(3, 512);
          enc.dispatch({1024, 1, 1}, {1024, 1, 1});
          if (kept) {
            enc.set_function(fun);
            enc.set_buffer(0, out);
            enc.set_buffer(1, bpick);
            enc.set_constant(2, 512);
            enc.set_constant(3, KB);
            enc.set_constant(4, blk);
            enc.dispatch({512, 1, 1}, {64, 1, 1});
          }
        });
      }
    }
  }
  // the small per-layer kernels, one row
  {
    const int HC4 = 4 * H, E = 384, K6 = 6, S = 16;
    SharedBuffer X = acts((std::size_t)HC4);
    SharedBuffer fnT = mc->make_shared_buffer((std::size_t)HC4 * 24 * 4);
    SharedBuffer part = mc->make_shared_buffer((std::size_t)S * 25 * 4);
    SharedBuffer f3 = mc->make_shared_buffer(64), f24 = mc->make_shared_buffer(
        24 * 4);
    SharedBuffer pre = mc->make_shared_buffer(64), post =
        mc->make_shared_buffer(64), comb = mc->make_shared_buffer(128);
    std::memset(fnT.contents(), 0, fnT.byte_size());
    std::memset(f3.contents(), 0, 64);
    std::memset(f24.contents(), 0, 96);
    ComputeFunction hp = lib.function("dsv41_hc_mix_part");
    ComputeFunction hf = lib.function("dsv41_hc_mix");
    time("hc mix (16 slices + finisher)", (double)HC4 * 24 * 4,
         [&](ComputeEncoder& enc) {
           enc.set_function(hp);
           enc.set_buffer(0, X);
           enc.set_buffer(1, fnT);
           enc.set_buffer(2, part);
           enc.set_constant(3, HC4);
           enc.set_constant(4, S);
           enc.dispatch({256 * (unsigned)S, 1, 1}, {256, 1, 1});
           enc.set_function(hf);
           enc.set_buffer(0, part);
           enc.set_buffer(1, f3);
           enc.set_buffer(2, f24);
           enc.set_buffer(3, pre);
           enc.set_buffer(4, post);
           enc.set_buffer(5, comb);
           enc.set_constant(6, HC4);
           enc.set_constant(7, 1e-6f);
           enc.set_constant(8, 1e-6f);
           enc.set_constant(9, 20);
           enc.set_constant(10, S);
           enc.set_constant(11, 1);
           enc.dispatch({32, 1, 1}, {32, 1, 1});
         });
    time("hc mix: the slices alone", (double)HC4 * 24 * 4,
         [&](ComputeEncoder& enc) {
           enc.set_function(hp);
           enc.set_buffer(0, X);
           enc.set_buffer(1, fnT);
           enc.set_buffer(2, part);
           enc.set_constant(3, HC4);
           enc.set_constant(4, S);
           enc.dispatch({256 * (unsigned)S, 1, 1}, {256, 1, 1});
         });
    time("hc mix: the finisher alone", 0.0, [&](ComputeEncoder& enc) {
      enc.set_function(hf);
      enc.set_buffer(0, part);
      enc.set_buffer(1, f3);
      enc.set_buffer(2, f24);
      enc.set_buffer(3, pre);
      enc.set_buffer(4, post);
      enc.set_buffer(5, comb);
      enc.set_constant(6, HC4);
      enc.set_constant(7, 1e-6f);
      enc.set_constant(8, 1e-6f);
      enc.set_constant(9, 20);
      enc.set_constant(10, S);
      enc.set_constant(11, 1);
      enc.dispatch({32, 1, 1}, {32, 1, 1});
    });
    ComputeFunction hpn = lib.function("dsv41_hc_pre_norm");
    SharedBuffer nw = acts(H), y = acts(H);
    time("hc pre + rms norm", (double)HC4 * 2, [&](ComputeEncoder& enc) {
      enc.set_function(hpn);
      enc.set_buffer(0, X);
      enc.set_buffer(1, pre);
      enc.set_buffer(2, nw);
      enc.set_buffer(3, y);
      enc.set_constant(4, H);
      enc.set_constant(5, 1e-6f);
      enc.set_constant(6, 1);
      enc.dispatch({1024, 1, 1}, {1024, 1, 1});
    });
    ComputeFunction hpo = lib.function("dsv41_hc_post");
    time("hc post", (double)HC4 * 4, [&](ComputeEncoder& enc) {
      enc.set_function(hpo);
      enc.set_buffer(0, y);
      enc.set_buffer(1, X);
      enc.set_buffer(2, post);
      enc.set_buffer(3, comb);
      enc.set_constant(4, H);
      enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
    });
    ComputeFunction rt = lib.function("dsv41_route");
    SharedBuffer sc = mc->make_shared_buffer(E * 4), bias =
        mc->make_shared_buffer(E * 4), vl = mc->make_shared_buffer(16);
    SharedBuffer eid = mc->make_shared_buffer(K6 * 4), wt =
        mc->make_shared_buffer(K6 * 4);
    for (int e = 0; e < E; ++e) {
      static_cast<float*>(sc.contents())[e] = (float)((e * 37) % 101) * 0.01f;
      static_cast<float*>(bias.contents())[e] = 0.0f;
    }
    time("router top-6 of 384", (double)E * 8, [&](ComputeEncoder& enc) {
      enc.set_function(rt);
      enc.set_buffer(0, sc);
      enc.set_buffer(1, bias);
      enc.set_buffer(2, bias);
      enc.set_buffer(3, vl);
      enc.set_buffer(4, eid);
      enc.set_buffer(5, wt);
      enc.set_constant(6, E);
      enc.set_constant(7, K6);
      enc.set_constant(8, 1.5f);
      enc.set_constant(9, 1);
      enc.set_constant(10, 0);
      enc.dispatch({256, 1, 1}, {256, 1, 1});
    });
    ComputeFunction fl = lib.function("dsv41_fill32");
    time("an empty dispatch", 4.0, [&](ComputeEncoder& enc) {
      enc.set_function(fl);
      enc.set_buffer(0, eid);
      enc.set_constant(1, 0);
      enc.set_constant(2, 1);
      enc.dispatch({1, 1, 1}, {32, 1, 1});
    });
    // a commit and wait: what a layer's sync costs the GPU's clock
    {
      CommandStream st = mc->make_command_stream();
      std::string why;
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < reps; ++r) {
        ComputeEncoder enc = st.begin_compute();
        enc.set_function(fl);
        enc.set_buffer(0, eid);
        enc.set_constant(1, 0);
        enc.set_constant(2, 1);
        enc.dispatch({1, 1, 1}, {32, 1, 1});
        enc.end();
        ASSERT_TRUE(st.commit().wait_ok(&why));
      }
      std::printf("[dsv41] kernel %-34s %8.1f us\n",
                  "a commit and wait (one dispatch)",
                  std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - t0)
                          .count() / reps);
    }
    // a handoff inside one committed buffer: the GPU signals, the host
    // (spinning) answers, the GPU waits for the answer -- what a layer's
    // resolve would cost without a commit
    {
      CommandStream st = mc->make_command_stream();
      metal_compute::Event ev = mc->make_event();
      ASSERT_TRUE(ev.valid());
      const std::uint64_t v0 = ev.signaled_value();
      for (int r = 0; r < reps; ++r) {
        ComputeEncoder enc = st.begin_compute();
        enc.set_function(fl);
        enc.set_buffer(0, eid);
        enc.set_constant(1, 0);
        enc.set_constant(2, 1);
        enc.dispatch({1, 1, 1}, {32, 1, 1});
        enc.end();
        st.encode_signal(ev, v0 + 2 * r + 1);
        st.encode_wait(ev, v0 + 2 * r + 2);
      }
      const auto t0 = std::chrono::steady_clock::now();
      auto f = st.commit();
      for (int r = 0; r < reps; ++r) {
        while (ev.signaled_value() < v0 + 2 * r + 1) {}
        ev.set_signaled_value(v0 + 2 * r + 2);
      }
      std::string why;
      ASSERT_TRUE(f.wait_ok(&why));
      std::printf("[dsv41] kernel %-34s %8.1f us\n",
                  "an event handoff (one dispatch)",
                  std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - t0)
                          .count() / reps);
    }
    // the same through shared memory: a one-thread kernel posts and spins,
    // the host answers -- no driver in the loop, if the two see each
    // other's stores while the buffer runs
    {
      ComputeFunction hs = lib.function("dsv41_handshake");
      ASSERT_TRUE(hs.valid());
      SharedBuffer fl3 = mc->make_shared_buffer(64);
      std::memset(fl3.contents(), 0, 64);
      auto* w = static_cast<std::uint32_t*>(fl3.contents());
      CommandStream st = mc->make_command_stream();
      ComputeEncoder enc = st.begin_compute();
      for (int r = 0; r < reps; ++r) {
        enc.set_function(fl);
        enc.set_buffer(0, eid);
        enc.set_constant(1, 0);
        enc.set_constant(2, 1);
        enc.dispatch({1, 1, 1}, {32, 1, 1});
        enc.set_function(hs);
        enc.set_buffer(0, fl3);
        enc.set_constant(1, (unsigned)(r + 1));
        enc.set_constant(2, (unsigned)(r + 1));
        enc.set_constant(3, 1u << 20);
        enc.dispatch({1, 1, 1}, {32, 1, 1});
      }
      enc.end();
      const auto t0 = std::chrono::steady_clock::now();
      auto f = st.commit();
      int late = 0;
      for (int r = 0; r < reps; ++r) {
        const auto ts = std::chrono::steady_clock::now();
        while (std::atomic_ref<std::uint32_t>(w[0]).load(
                   std::memory_order_acquire) < (std::uint32_t)(r + 1)) {
          if (std::chrono::steady_clock::now() - ts >
              std::chrono::seconds(2)) {
            ++late;
            break;
          }
        }
        std::atomic_ref<std::uint32_t>(w[1]).store(
            (std::uint32_t)(r + 1), std::memory_order_release);
      }
      std::string why;
      ASSERT_TRUE(f.wait_ok(&why));
      std::printf("[dsv41] kernel %-34s %8.1f us  (%u gave up, %d unseen)\n",
                  "a shared-memory handshake",
                  std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - t0)
                          .count() / reps,
                  w[2], late);
      // the visibility alone: the rounds inside one dispatch
      ComputeFunction hr = lib.function("dsv41_handshake_rounds");
      ASSERT_TRUE(hr.valid());
      std::memset(fl3.contents(), 0, 64);
      ComputeEncoder e3 = st.begin_compute();
      e3.set_function(hr);
      e3.set_buffer(0, fl3);
      e3.set_constant(1, (unsigned)reps);
      e3.set_constant(2, 1u << 20);
      e3.dispatch({1, 1, 1}, {32, 1, 1});
      e3.end();
      auto f3 = st.commit();
      // a bounded wait: a GPU that never sees the host gives up instead
      auto posted = [&](std::uint32_t r) {
        const auto ts = std::chrono::steady_clock::now();
        while (std::atomic_ref<std::uint32_t>(w[0]).load(
                   std::memory_order_acquire) < r) {
          if (std::chrono::steady_clock::now() - ts >
              std::chrono::seconds(2)) {
            return false;
          }
        }
        return true;
      };
      bool seen = posted(1u);
      const auto t1 = std::chrono::steady_clock::now();
      std::atomic_ref<std::uint32_t>(w[1]).store(1u,
                                                 std::memory_order_release);
      for (int r = 2; r <= reps && seen; ++r) {
        seen = posted((std::uint32_t)r);
        std::atomic_ref<std::uint32_t>(w[1]).store(
            (std::uint32_t)r, std::memory_order_release);
      }
      std::atomic_ref<std::uint32_t>(w[1]).store(~0u,
                                                 std::memory_order_release);
      const double rt = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - t1)
                            .count() / (reps - 1);
      ASSERT_TRUE(f3.wait_ok(&why));
      std::printf("[dsv41] kernel %-34s %8.2f us  (%u gave up)\n",
                  "a handshake round, no dispatch", rt, w[2]);
      // a dispatch a round, nothing between, serial and concurrent: where
      // the dispatch version's time goes (answer -> the next post)
      for (int conc = 0; conc < 2; ++conc) {
        std::memset(fl3.contents(), 0, 64);
        ComputeEncoder e4 = st.begin_compute(
            conc ? metal_compute::DispatchType::Concurrent
                 : metal_compute::DispatchType::Serial);
        for (int r = 0; r < reps; ++r) {
          e4.set_function(hs);
          e4.set_buffer(0, fl3);
          e4.set_constant(1, (unsigned)(r + 1));
          e4.set_constant(2, (unsigned)(r + 1));
          e4.set_constant(3, 1u << 20);
          e4.dispatch({1, 1, 1}, {32, 1, 1});
          if (conc) { e4.memory_barrier(); }
        }
        e4.end();
        auto f4 = st.commit();
        double turn = 0;
        auto ta = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
          if (!posted((std::uint32_t)(r + 1))) { break; }
          const auto now = std::chrono::steady_clock::now();
          if (r > 0) {
            turn += std::chrono::duration<double, std::micro>(now - ta)
                        .count();
          }
          ta = std::chrono::steady_clock::now();
          std::atomic_ref<std::uint32_t>(w[1]).store(
              (std::uint32_t)(r + 1), std::memory_order_release);
        }
        std::atomic_ref<std::uint32_t>(w[1]).store(
            ~0u, std::memory_order_release);
        ASSERT_TRUE(f4.wait_ok(&why));
        std::printf("[dsv41] kernel %-34s %8.2f us  (%u gave up)\n",
                    conc ? "answer -> next post, concurrent"
                         : "answer -> next post, serial",
                    turn / (reps - 1), w[2]);
      }
    }
  }
}

// A word the GPU writes while this host reads it, as it is now.
static std::uint32_t
load_now_(const std::uint32_t& x)
{
  return std::atomic_ref<std::uint32_t>(const_cast<std::uint32_t&>(x))
      .load(std::memory_order_acquire);
}

// dsv41_handshake_io's checksum of a mailbox's `n` words for round `seq`.
static std::uint32_t
hs_sum_(const std::uint32_t* w, std::uint32_t n, std::uint32_t seq)
{
  std::uint32_t s = 0;
  for (std::uint32_t k = 0; k < n; ++k) {
    const std::uint32_t v = load_now_(w[k]);
    s += v * (2u * k + 1u) + (v >> 16) * 0x9E3779B9u;
  }
  return s + seq * 2654435761u;
}

// What a running dispatch sees of the host's stores (VPIPE_DSV41_HS_PROBE=1):
// the answer word on the posted word's cache line or its own, read by a
// load or an atomic add of 0, with the host writing beside it as it waits
// (a heartbeat on the posted word's line) or not.
TEST(deepseek_v41, handshake_visibility_probe)
{
  using metal_compute::CommandStream;
  using metal_compute::ComputeEncoder;
  using metal_compute::ComputeFunction;
  using metal_compute::SharedBuffer;
  if (env_("VPIPE_DSV41_HS_PROBE").empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  auto lib = mc->load_library("deepseek_v41_bf16");
  ComputeFunction fp = lib.function("dsv41_handshake_probe");
  ASSERT_TRUE(fp.valid());
  SharedBuffer f = mc->make_shared_buffer(1024);
  auto* w = static_cast<std::uint32_t*>(f.contents());
  auto word = [w](int k) { return std::atomic_ref<std::uint32_t>(w[k]); };
  CommandStream st = mc->make_command_stream();
  const std::uint32_t rounds = 50;
  for (std::uint32_t mode = 0; mode < 2; ++mode) {
    for (std::uint32_t at : {1u, 64u}) {
      for (int beat = 0; beat < 2; ++beat) {
        std::memset(f.contents(), 0, 1024);
        ComputeEncoder enc = st.begin_compute();
        enc.set_function(fp);
        enc.set_buffer(0, f);
        enc.set_constant(1, rounds);
        enc.set_constant(2, 1u << 20);
        enc.set_constant(3, at);
        enc.set_constant(4, mode);
        enc.dispatch({1, 1, 1}, {32, 1, 1});
        enc.end();
        auto fence = st.commit();
        std::uint32_t seen = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::uint32_t r = 1; r <= rounds; ++r) {
          const auto ts = std::chrono::steady_clock::now();
          bool ok = true;
          while (word(0).load(std::memory_order_acquire) < r) {
            if (beat) { word(3).fetch_add(1, std::memory_order_relaxed); }
            if (word(2).load(std::memory_order_relaxed) != 0 ||
                std::chrono::steady_clock::now() - ts >
                    std::chrono::seconds(2)) {
              ok = false;
              break;
            }
          }
          if (!ok) { break; }
          seen = r;
          word((int)at).store(r, std::memory_order_release);
        }
        word((int)at).store(~0u, std::memory_order_release);
        std::string why;
        ASSERT_TRUE(fence.wait_ok(&why));
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0)
                              .count() / rounds;
        std::printf("[dsv41] probe: %s, answer %s, host %s: %u of %u rounds "
                    "seen, %u gave up, %.2f us a round\n",
                    mode == 0 ? "load" : "add 0",
                    at == 1 ? "on the post's line" : "on its own line",
                    beat ? "beating beside" : "quiet", seen, rounds, w[2],
                    us);
      }
    }
  }
}

// The decode's per-layer turn of the host (the expert cache's resolve)
// rides a handshake inside a running command buffer: the GPU's words reach
// the host through the mailbox, and the host's answer reaches the
// dispatches after it, every round. Plain stores both ways are counted
// too, for the record; nothing relies on them. The rounds go a buffer of
// 16 at a time, each answered before the next is encoded: a host that
// encodes far ahead of its answers can block on the queue's buffer limit
// while the GPU waits for it.
TEST(deepseek_v41, a_mid_buffer_handshake_carries_data)
{
  using metal_compute::CommandStream;
  using metal_compute::ComputeEncoder;
  using metal_compute::ComputeFunction;
  using metal_compute::ComputeLibrary;
  using metal_compute::SharedBuffer;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  ComputeLibrary lib = mc->load_library("deepseek_v41_bf16");
  ComputeFunction fw = lib.function("dsv41_probe_write");
  ComputeFunction fh = lib.function("dsv41_handshake_io");
  ComputeFunction fc = lib.function("dsv41_probe_check");
  ASSERT_TRUE(fw.valid() && fh.valid() && fc.valid());
  if (!fw.valid() || !fh.valid() || !fc.valid()) { return; }
  const std::string rv = env_("VPIPE_DSV41_HS_ROUNDS");
  const unsigned rounds = rv.empty() ? 2000u : (unsigned)std::stoi(rv);
  const unsigned NO = 3168, NI = 600, chunk = 16;
  auto words = [&](unsigned n) {
    SharedBuffer b = mc->make_shared_buffer((std::size_t)n * 4);
    std::memset(b.contents(), 0, (std::size_t)n * 4);
    return b;
  };
  SharedBuffer f = words(16), src = words(NO), bo = words(NO + 1);
  SharedBuffer bi = words(NI + 1), dst = words(NI), raw = words(NI);
  auto* fw32 = static_cast<std::uint32_t*>(f.contents());
  auto* box = static_cast<const std::uint32_t*>(bo.contents());
  auto* plain = static_cast<const std::uint32_t*>(src.contents());
  auto* in = static_cast<std::uint32_t*>(bi.contents());
  auto* rw = static_cast<std::uint32_t*>(raw.contents());
  auto word = [&](int k) { return std::atomic_ref<std::uint32_t>(fw32[k]); };
  unsigned long bad_box = 0, bad_plain = 0, bad_sums = 0;
  bool lost = false;
  CommandStream st = mc->make_command_stream();
  std::string why;
  {
    // the model's probe first: a GPU that does not see the host's stores
    // while a buffer runs (an M4 Pro: only once they leave the CPU's
    // caches) keeps the handshake off, and is not checked here
    const auto m = genai::MetalDeepseekV41Model::handshake_works(mc);
    if (!m) {
      std::printf("[dsv41] handshake: this GPU does not see host stores "
                  "mid-buffer; not checked here\n");
      return;
    }
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (unsigned r0 = 1; r0 <= rounds && !lost; r0 += chunk) {
    const unsigned r1 = std::min(rounds, r0 + chunk - 1);
    ComputeEncoder enc = st.begin_compute();
    for (unsigned r = r0; r <= r1; ++r) {
      enc.set_function(fw);
      enc.set_buffer(0, src);
      enc.set_constant(1, r);
      enc.set_constant(2, NO);
      enc.dispatch({NO, 1, 1}, {256, 1, 1});
      enc.set_function(fh);
      enc.set_buffer(0, f);
      enc.set_buffer(1, src);
      enc.set_buffer(2, src);
      enc.set_buffer(3, src);
      enc.set_buffer(4, bo);
      enc.set_buffer(5, bi);
      enc.set_buffer(6, dst);
      enc.set_buffer(7, dst);
      enc.set_buffer(8, dst);
      const std::uint32_t nv[4] = {NO, 0, 0, r};
      const std::uint32_t mv[4] = {NI, 0, 0, 1u << 24};
      enc.set_constant_bytes(9, nv, sizeof nv);
      enc.set_constant_bytes(10, mv, sizeof mv);
      enc.dispatch({256, 1, 1}, {256, 1, 1});
      enc.set_function(fc);
      enc.set_buffer(0, dst);
      enc.set_buffer(1, raw);
      enc.set_buffer(2, f);
      enc.set_constant(3, r);
      enc.set_constant(4, NI);
      enc.dispatch({NI, 1, 1}, {256, 1, 1});
    }
    enc.end();
    auto fence = st.commit();
    for (unsigned r = r0; r <= r1 && !lost; ++r) {
      const auto ts = std::chrono::steady_clock::now();
      while (word(0).load(std::memory_order_acquire) < r) {
        word(3).fetch_add(1, std::memory_order_relaxed);   // heartbeat
        if (word(2).load(std::memory_order_relaxed) != 0 ||
            std::chrono::steady_clock::now() - ts > std::chrono::seconds(2)) {
          lost = true;
          break;
        }
      }
      if (lost) { break; }
      // the mailbox once its checksum holds (as the model reads it)
      const auto tc = std::chrono::steady_clock::now();
      while (hs_sum_(box, NO, r) != load_now_(box[NO])) {
        if (std::chrono::steady_clock::now() - tc > std::chrono::seconds(1)) {
          ++bad_sums;
          break;
        }
      }
      for (unsigned k = 0; k < NO; ++k) {
        bad_box += box[k] != r * 4096u + k;
        bad_plain += plain[k] != r * 4096u + k;
      }
      for (unsigned k = 0; k < NI; ++k) {
        in[k] = r * 7u + k;
        rw[k] = r * 7u + k;
      }
      in[NI] = hs_sum_(in, NI, r);
      word(1).store(r, std::memory_order_release);
    }
    if (lost) {   // let the buffer finish: every later wait passes
      word(1).store(~0u, std::memory_order_release);
    }
    ASSERT_TRUE(fence.wait_ok(&why));
  }
  const double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - t0).count();
  std::printf("[dsv41] handshake: %u rounds, %.1f us a round; mailbox "
              "%lu / %u bad out (%lu sums), %u bad in; plain %lu bad out, "
              "%u bad in; %u gave up%s\n",
              rounds, us / rounds, bad_box, rounds * NO, bad_sums, fw32[5],
              bad_plain, fw32[6], fw32[2], lost ? " (LOST)" : "");
  ASSERT_TRUE(!lost);
  ASSERT_TRUE(fw32[2] == 0);
  ASSERT_TRUE(bad_box == 0 && bad_sums == 0);
  ASSERT_TRUE(fw32[5] == 0);
}

TEST(deepseek_v41, small_model_layers)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty() || env_("VPIPE_DSV41_LAYERS").empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  std::string err;
  MetalDeepseekV41Model::Options opt;
  opt.use_bf16 = env_("VPIPE_DSV41_BF16") == "1";
  auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                       mc, opt, &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { std::printf("%s\n", err.c_str()); return; }
  const std::vector<std::int32_t> r32 =
      read_raw_<std::int32_t>(fs::path(gd) / "route_f32.i32");
  const int T = (int)ids.size(), K = m->config().top_k;
  for (int L = 0; L < m->config().n_layers; ++L) {
    m->capture_layer(L);
    ASSERT_TRUE(m->forward(100 + L, ids, nullptr, false, &err));
    const std::vector<float> ref = read_raw_<float>(
        fs::path(gd) / std::format("resid_L{}_f32.f32", L));
    const auto& got = m->captured();
    const double rel = got.size() == ref.size()
                           ? rel_l2_(got.data(), ref.data(), got.size())
                           : -1.0;
    // routes
    int same = 0;
    const auto& rt = m->routes()[(std::size_t)L];
    for (int t = 0; t < T; ++t) {
      std::vector<int> a(rt.begin() + t * K, rt.begin() + (t + 1) * K);
      std::vector<int> b(r32.begin() + ((std::size_t)L * T + t) * K,
                         r32.begin() + ((std::size_t)L * T + t + 1) * K);
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      same += a == b;
    }
    // worst row
    double worst = 0;
    int wrow = -1;
    const std::size_t R = got.size() / T;
    for (int t = 0; t < T && got.size() == ref.size(); ++t) {
      const double r = rel_l2_(got.data() + t * R, ref.data() + t * R, R);
      if (r > worst) { worst = r; wrow = t; }
    }
    // the reference's own bf16 run, the bar
    const std::vector<float> r16 = read_raw_<float>(
        fs::path(gd) / std::format("resid_L{}_bf16.f32", L));
    const double rb = r16.size() == ref.size()
                          ? rel_l2_(r16.data(), ref.data(), ref.size())
                          : -1.0;
    std::printf("[dsv41] layer %d: resid %.5f (ref bf16 %.5f; worst row %d "
                "%.5f), routes %d/%d\n", L, rel, rb, wrow, worst, same, T);
    if (env_("VPIPE_DSV41_ROWS") == std::to_string(L)) {
      for (int t = 0; t < T; ++t) {
        std::printf("  row %d %.5f\n", t,
                    rel_l2_(got.data() + t * R, ref.data() + t * R, R));
      }
    }
  }
}

// Through the manager, the way text-chat loads it: the architecture picks
// the exec, the chat template renders the turn, and the greedy decode over
// LoadedLanguageModel's token-id path is the model's own (a branch of the
// context continues as the context does).
TEST(deepseek_v41, loads_through_the_manager)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  Session sess;
  auto* mc = sess.metal_compute();
  auto* mgr = sess.generative_model_manager();
  ASSERT_TRUE(mc != nullptr && mgr != nullptr);
  genai::LoadSpec spec;
  spec.hf_dir = md.string();
  spec.compute_dtype = "bf16";
  spec.page_tokens = 512;
  spec.max_pages = 16;
  auto lm = mgr->load(spec);
  ASSERT_TRUE(lm != nullptr && lm->valid());
  if (lm == nullptr || !lm->valid()) { return; }
  const ChatTemplate* tpl = lm->chat_template();
  ASSERT_TRUE(tpl != nullptr);
  if (tpl == nullptr) { return; }
  std::vector<std::int32_t> ids;
  tpl->render_user_turn("Name three colours of the rainbow.", true, &ids);
  auto ctx = lm->make_context();
  std::int32_t t = lm->prefill(ctx, ids);
  std::vector<std::int32_t> got;
  for (int i = 0; i < 8 && t >= 0; ++i) {
    got.push_back(t);
    t = lm->next_token(ctx);
  }
  ASSERT_TRUE(got.size() == 8);
  // the model alone, greedy
  std::string err;
  MetalDeepseekV41Model::Options opt;
  auto m = MetalDeepseekV41Model::load(WeightSet::open(md.string(), nullptr),
                                       mc, opt, &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { return; }
  std::vector<std::int32_t> want;
  std::vector<float> lg;
  ASSERT_TRUE(m->forward(7, ids, &lg, false, &err));
  const int V = m->config().vocab;
  for (int i = 0; i < 8; ++i) {
    const std::int32_t a = argmax_(lg.data(), V);
    want.push_back(a);
    ASSERT_TRUE(m->forward(7, std::vector<std::int32_t>{a}, &lg, false,
                           &err));
  }
  std::printf("[dsv41] manager decode %s the model's\n",
              got == want ? "matches" : "DIFFERS from");
  EXPECT_TRUE(got == want);
  // what the plan books for the model is what it holds (no cache: every
  // expert resident)
  std::size_t ex = 0, rest = 0;
  ASSERT_TRUE(MetalDeepseekV41ModelExec::expert_bytes(md.string(), &ex,
                                                      &rest));
  const double held = (double)m->resident_bytes();
  // An MLX pack's bf16 wo_a is booked at its size on disk and held as the
  // release's FP8 codes where it is them (known, and the safe side: a pack
  // that changed wo_a keeps it bf16); that term apart, the two agree.
  const DeepseekV41Config& mc_ = m->config();
  const double wn = (double)mc_.o_groups * mc_.o_lora;
  const double wk = (double)mc_.n_heads * mc_.head_dim / mc_.o_groups;
  const double wo_a_over = m->wo_a_codes() * (wn * wk * 2 - wn * wk * 33 / 32);
  std::printf("[dsv41] held %.1f MB, booked %.1f MB (experts %.1f, wo_a "
              "booked over its codes %.1f)\n", held / 1048576.0,
              (double)(rest + ex) / 1048576.0, (double)ex / 1048576.0,
              wo_a_over / 1048576.0);
  EXPECT_TRUE(std::abs(held - ((double)(rest + ex) - wo_a_over)) <
              0.02 * held);
  const double run = (double)m->run_held();
  const double booked = (double)MetalDeepseekV41Model::run_bytes(
      m->config(), (int)ids.size(), (int)ids.size() + 8);
  std::printf("[dsv41] run state held %.2f MB, booked %.2f MB\n",
              run / 1048576.0, booked / 1048576.0);
  EXPECT_TRUE(run <= booked && booked <= 1.1 * run + (16 << 20));
  // a branch continues as the original does
  auto br = lm->branch(ctx);
  ASSERT_TRUE(br.valid());
  const std::int32_t a = lm->next_token(ctx);
  const std::int32_t b = lm->next_token(br);
  EXPECT_TRUE(a == b);
  ::unsetenv("VPIPE_LLM_BACKEND");
}

// The release's config and engram layout: the primes each layer draws must
// sum to its table's rows (384006168 / 384016682), and the multipliers are
// numpy's (engram.compute_hash_multipliers((1, 14), 4, 99092)).
TEST(deepseek_v41, release_config_and_engram_layout)
{
  const std::string md = env_("VPIPE_DSV41_TEST_MODEL_PATH");
  if (md.empty()) { return; }
  DeepseekV41Config cfg;
  std::string err;
  ASSERT_TRUE(DeepseekV41Config::from_dir(md, &cfg, &err));
  if (!err.empty()) { std::printf("[dsv41] %s\n", err.c_str()); return; }
  EXPECT_TRUE(cfg.n_layers == 40 && cfg.dim == 5120 && cfg.n_experts == 384);
  // the attention layout the README describes
  EXPECT_TRUE(cfg.plan[0].ratio == 0 && cfg.plan[2].kv_source &&
              cfg.plan[3].kv_from == 2 && cfg.plan[20].ratio == 1 &&
              cfg.plan[20].candidate_source && cfg.plan[24].index_source &&
              !cfg.plan[24].owns_index_keys && cfg.plan[24].keys_from == 20 &&
              cfg.plan[24].uses_candidates && cfg.plan[39].index_from == 36);
  DeepseekV41Engram eg;
  ASSERT_TRUE(eg.init(cfg, (fs::path(md) / "tokenizer.json").string(), &err));
  if (!err.empty()) { std::printf("[dsv41] %s\n", err.c_str()); }
  const std::vector<std::vector<std::int64_t>> want = {
      {76632096046245, 4839876093313, 35959672319349, 73987337458391},
      {67716810739261, 51510806800915, 30921347202721, 82619226485591}};
  EXPECT_TRUE(eg.multipliers() == want);
  // what a forward books beside the weights, by context
  for (int ctx : {4096, 32768, 131072, 1 << 20}) {
    std::printf("[dsv41] run state at %d positions: %.0f MB\n", ctx,
                (double)MetalDeepseekV41Model::run_bytes(cfg, 2048, ctx) /
                    1048576.0);
  }
  // and the expert cache this box gets beside it at the shipped graph's
  // pool (65536 positions, 8192-row pieces), with and without the DSpark
  // head -- and how much more a forward could book before that changes
  Session sess;
  const MetalCompute* mc = sess.metal_compute();
  for (bool spark : {false, true}) {
    const std::size_t run =
        MetalDeepseekV41Model::run_bytes(cfg, 8192, 65536, spark);
    std::string line;
    for (int gb : {0, 1, 4, 16}) {
      const std::size_t cache = MetalDeepseekV41ModelExec::
          lm_expert_cache_bytes(nullptr, md, run + ((std::size_t)gb << 30),
                                mc, spark);
      line += std::format(" +{} GB: {}", gb,
                          cache == 0 ? std::string("resident")
                                     : std::format("{} MB", cache >> 20));
    }
    std::printf("[dsv41] %s: run %.0f MB; the cache%s\n",
                spark ? "with DSpark" : "plain", run / 1048576.0,
                line.c_str());
  }
}

// The release against the reference's own forward on its weights (a
// `real` golden): every prefill row's logits and the teacher-forced decode
// steps, held to the reference's own bf16 distance; the greedy picks. The
// routed experts come through the cache the engine sizes for this box.
TEST(deepseek_v41, release_matches_reference)
{
  const std::string md = env_("VPIPE_DSV41_TEST_MODEL_PATH");
  const std::string gd = env_("VPIPE_DSV41_GOLDEN");
  if (md.empty() || gd.empty()) { return; }
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  const std::vector<std::int32_t> ids = ints_(meta, "ids");
  const std::vector<std::int32_t> dec = ints_(meta, "decode_ids");
  const std::vector<float> l32 =
      read_raw_<float>(fs::path(gd) / "logits_f32.f32");
  const std::vector<float> d32 =
      read_raw_<float>(fs::path(gd) / "dlogits_f32.f32");
  auto mo = meta.as_object();
  const bool has_bar = mo.contains("bf16_vs_f32_logits");
  const double bar = has_bar ? mo.at("bf16_vs_f32_logits").as_real() : 0.05;
  const double dbar =
      has_bar && mo.contains("bf16_vs_f32_dlogits")
          ? mo.at("bf16_vs_f32_dlogits").as_real() : 0.05;
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  if (mc == nullptr) { return; }
  std::string err;
  MetalDeepseekV41Model::Options opt;
  DeepseekV41Config cfg;
  ASSERT_TRUE(DeepseekV41Config::from_dir(md, &cfg, &err));
  // a golden with DSpark's drafts: the head loads, and its drafts are
  // checked in the pass on fp32's picks below
  const fs::path sp32 = fs::path(gd) / "spark_logits_f32.f32";
  const fs::path spr = fs::path(gd) / "spark_logits_bf16r.f32";
  opt.dspark = fs::exists(sp32) && fs::exists(spr) &&
               env_("VPIPE_DSV41_NO_DSPARK").empty();
  opt.expert_cache_bytes =
      MetalDeepseekV41ModelExec::lm_expert_cache_bytes(
          nullptr, md,
          MetalDeepseekV41Model::run_bytes(cfg, opt.max_rows, 4096,
                                           opt.dspark),
          mc, opt.dspark);
  std::printf("[dsv41] expert cache %zu MB\n", opt.expert_cache_bytes >> 20);
  const auto t0 = std::chrono::steady_clock::now();
  auto m = MetalDeepseekV41Model::load(WeightSet::open(md, nullptr), mc, opt,
                                       &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { std::printf("[dsv41] %s\n", err.c_str()); return; }
  const auto t1 = std::chrono::steady_clock::now();
  std::printf("[dsv41] %s layout, holds %zu MB (expert cache %zu MB)\n",
              m->mlx_layout() ? "MLX" : "release",
              m->resident_bytes() >> 20,
              m->expert_cache()->bytes() >> 20);
  const int V = m->config().vocab;
  std::vector<float> lg;
  ASSERT_TRUE(m->forward(1, ids, &lg, /*all_rows=*/true, &err));
  if (!err.empty()) { std::printf("[dsv41] %s\n", err.c_str()); return; }
  const auto t2 = std::chrono::steady_clock::now();
  ASSERT_TRUE(lg.size() == l32.size());
  if (lg.size() != l32.size()) { return; }
  const double rel = rel_l2_(lg.data(), l32.data(), lg.size());
  const int T = (int)ids.size();
  int agree = 0;
  for (int t = 0; t < T; ++t) {
    agree += argmax_(lg.data() + (std::size_t)t * V, V) ==
             argmax_(l32.data() + (std::size_t)t * V, V);
  }
  std::vector<float> dl;
  std::vector<std::vector<std::vector<std::int32_t>>> droutes;
  for (std::int32_t t : dec) {
    std::vector<float> one;
    ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>{t}, &one, false,
                           &err));
    dl.insert(dl.end(), one.begin(), one.end());
    droutes.push_back(m->routes());
  }
  const auto t3 = std::chrono::steady_clock::now();
  const double drel = dl.size() == d32.size()
                          ? rel_l2_(dl.data(), d32.data(), dl.size()) : 1e9;
  int dagree = 0;
  for (std::size_t i = 0; i < dec.size() && dl.size() == d32.size(); ++i) {
    dagree += argmax_(dl.data() + i * V, V) == argmax_(d32.data() + i * V, V);
  }
  auto s = [](auto a, auto b) {
    return std::chrono::duration<double>(b - a).count();
  };
  std::printf("[dsv41] release: prefill %d rows %.5f from fp32 (ref bf16 "
              "%.5f), argmax %d/%d; decode %.5f (ref bf16 %.5f), argmax "
              "%d/%zu; load %.1f s, prefill %.1f s, decode %.1f s a token\n",
              T, rel, bar, agree, T, drel, dbar, dagree, dec.size(),
              s(t0, t1), s(t1, t2), s(t2, t3) / std::max<std::size_t>(
                                          1, dec.size()));
  // each step against fp32, beside the reference's own bf16 run
  const std::vector<float> d16 =
      read_raw_<float>(fs::path(gd) / "dlogits_bf16.f32");
  if (dl.size() == d32.size()) {
    std::string line;
    for (std::size_t i = 0; i < dec.size(); ++i) {
      const std::size_t o = i * (std::size_t)V;
      line += std::format(
          " {:.4f}", rel_l2_(dl.data() + o, d32.data() + o, (std::size_t)V));
      if (d16.size() == d32.size()) {
        line += std::format(
            "/{:.4f}",
            rel_l2_(d16.data() + o, d32.data() + o, (std::size_t)V));
      }
    }
    std::printf("[dsv41] decode steps (ours/ref bf16):%s\n", line.c_str());
  }
  // which layers' expert picks differ from fp32's at each decode step, and
  // how close fp32's last pick came (its 6th biased score less its 7th):
  // a flip at a near tie is the arithmetic's rounding, one at a wide
  // margin is not
  const fs::path rf = fs::path(gd) / "droute_f32.i32";
  if (fs::exists(rf)) {
    const auto r32 = read_raw_<std::int32_t>(rf);
    const auto r16 = read_raw_<std::int32_t>(fs::path(gd) / "droute_bf16.i32");
    const auto m32 = read_raw_<float>(fs::path(gd) / "dmargin_f32.f32");
    const int NL = m->config().n_layers, KK = m->config().top_k;
    auto set_of = [&](const std::int32_t* p) {
      std::vector<std::int32_t> v(p, p + KK);
      std::sort(v.begin(), v.end());
      return v;
    };
    for (std::size_t i = 0; i < droutes.size(); ++i) {
      std::string line;
      int ref_flips = 0;
      for (int L = 0; L < NL; ++L) {
        const std::size_t o = (i * NL + L) * (std::size_t)KK;
        if (o + KK > r32.size() || droutes[i][(std::size_t)L].size() <
                                       (std::size_t)KK) {
          continue;
        }
        const auto want = set_of(r32.data() + o);
        if (r16.size() == r32.size() && set_of(r16.data() + o) != want) {
          ++ref_flips;
        }
        if (set_of(droutes[i][(std::size_t)L].data()) != want) {
          const std::size_t mo = (i * NL + L) * (std::size_t)(KK + 1);
          line += std::format(" L{}(margin {:.2e})", L,
                              m32[mo + KK - 1] - m32[mo + KK]);
        }
      }
      std::printf("[dsv41] decode step %zu: picks off fp32 at%s; the "
                  "reference's bf16 off at %d layers\n", i,
                  line.empty() ? " none" : line.c_str(), ref_flips);
    }
  }
  const auto& st = m->expert_cache()->stats();
  std::printf("[dsv41] expert cache: %llu picks, %.1f%% hits, %.1f GB read\n",
              (unsigned long long)st.picks, 100.0 * st.hit_rate(),
              (double)st.bytes_read / 1e9);
  // The free run's distance is one draw of near ties: a bf16 run flips
  // some expert picks fp32 makes by a hair (the reference's own bf16 flips
  // 4-12 of 40 layers a step), and a flip in the last layers lands on the
  // logits whole. The prefill's many rows average the draws out; the six
  // decode steps do not, so they are held loosely here and tightly below.
  EXPECT_TRUE(rel < 1.5 * bar);
  EXPECT_TRUE(drel < 2.5 * dbar);
  // every decisive pick (fp32's top two at least 0.5 apart) the same
  auto gap = [&](const float* x) {
    float a = -INFINITY, b = -INFINITY;
    for (int v = 0; v < V; ++v) {
      if (x[v] > a) { b = a; a = x[v]; } else if (x[v] > b) { b = x[v]; }
    }
    return a - b;
  };
  int decisive = 0, held = 0;
  for (int t = 0; t < T; ++t) {
    const float* r = l32.data() + (std::size_t)t * V;
    if (gap(r) < 0.5f) { continue; }
    ++decisive;
    held += argmax_(lg.data() + (std::size_t)t * V, V) == argmax_(r, V);
  }
  for (std::size_t i = 0; i < dec.size() && dl.size() == d32.size(); ++i) {
    const float* r = d32.data() + i * V;
    if (gap(r) < 0.5f) { continue; }
    ++decisive;
    held += argmax_(dl.data() + i * V, V) == argmax_(r, V);
  }
  std::printf("[dsv41] decisive picks %d of %d held\n", held, decisive);
  EXPECT_TRUE(held == decisive);
  // The arithmetic alone: both sides take fp32's expert picks (the
  // golden's bf16r run, and force_routes here), each weighing them by its
  // own scores, against the reference's bf16 on those picks.
  const fs::path rf32 = fs::path(gd) / "route_f32.i32";
  if (!mo.contains("bf16r_vs_f32_logits") || !fs::exists(rf32) ||
      !fs::exists(fs::path(gd) / "droute_f32.i32")) {
    return;
  }
  const double rbar = mo.at("bf16r_vs_f32_logits").as_real();
  const double rdbar = mo.at("bf16r_vs_f32_dlogits").as_real();
  const auto p32 = read_raw_<std::int32_t>(rf32);
  const auto q32 = read_raw_<std::int32_t>(fs::path(gd) / "droute_f32.i32");
  const int NL = m->config().n_layers, KK = m->config().top_k;
  auto picks = [&](const std::vector<std::int32_t>& r, std::size_t step,
                   int rows) {
    std::vector<std::vector<std::int32_t>> out((std::size_t)NL);
    for (int L = 0; L < NL; ++L) {
      const std::size_t o =
          (step * NL + (std::size_t)L) * (std::size_t)rows * KK;
      out[(std::size_t)L].assign(r.begin() + (std::ptrdiff_t)o,
                                 r.begin() + (std::ptrdiff_t)(o + rows * KK));
    }
    return out;
  };
  std::vector<float> fl, fd, sl, sc;
  std::vector<std::int32_t> sd;
  auto draft = [&](std::int32_t tok) {
    if (!m->has_dspark()) { return; }
    std::vector<float> lg1, cf;
    std::vector<std::int32_t> d;
    ASSERT_TRUE(m->dspark_draft(2, tok, &d, &cf, &lg1, &err));
    sl.insert(sl.end(), lg1.begin(), lg1.end());
    sd.insert(sd.end(), d.begin(), d.end());
  };
  m->force_routes(picks(p32, 0, T));
  ASSERT_TRUE(m->forward(2, ids, &fl, /*all_rows=*/true, &err));
  draft(dec[0]);
  for (std::size_t i = 0; i < dec.size(); ++i) {
    std::vector<float> one;
    m->force_routes(picks(q32, i, 1));
    ASSERT_TRUE(m->forward(2, std::vector<std::int32_t>{dec[i]}, &one, false,
                           &err));
    fd.insert(fd.end(), one.begin(), one.end());
    if (i + 1 < dec.size()) { draft(dec[i + 1]); }
  }
  ASSERT_TRUE(fl.size() == l32.size() && fd.size() == d32.size());
  if (fl.size() != l32.size() || fd.size() != d32.size()) { return; }
  const double frel = rel_l2_(fl.data(), l32.data(), fl.size());
  const double fdrel = rel_l2_(fd.data(), d32.data(), fd.size());
  std::printf("[dsv41] on fp32's picks: prefill %.5f from fp32 (ref bf16 "
              "%.5f), decode %.5f (ref bf16 %.5f)\n", frel, rbar, fdrel,
              rdbar);
  EXPECT_TRUE(frel < 1.5 * rbar);
  EXPECT_TRUE(fdrel < 1.5 * rdbar);
  // DSpark's drafts from the same positions, against the reference's on
  // fp32's picks (the draft follows the main model's hidden whole: free
  // runs differ by every pick that flipped upstream)
  if (m->has_dspark()) {
    const auto s32 = read_raw_<float>(sp32);
    const auto sr = read_raw_<float>(spr);
    const auto i32 = read_raw_<std::int32_t>(fs::path(gd) /
                                             "spark_ids_f32.i32");
    ASSERT_TRUE(sl.size() == s32.size() && sr.size() == s32.size());
    if (sl.size() != s32.size() || sr.size() != s32.size()) { return; }
    const double drel = rel_l2_(sl.data(), s32.data(), s32.size());
    const double dbar2 = rel_l2_(sr.data(), s32.data(), s32.size());
    const int B = m->config().dspark_block;
    int same = 0;
    for (std::size_t k = 0; k < sd.size(); ++k) {
      same += sd[k] == i32[(k / B) * (B + 1) + 1 + k % B];
    }
    std::printf("[dsv41] DSpark on fp32's picks: drafts %.5f from fp32 (ref "
                "bf16 %.5f), %d/%zu draft tokens as fp32's\n", drel, dbar2,
                same, sd.size());
    // Each draft's (ours / the reference's bf16), and the MEDIAN held: a
    // draft's own routes are not forced, so one near tie flipped upstream
    // of a draft jumps it alone -- the reference's bf16 has one -- and the
    // overall figure follows whichever draft that is (a collapse's sum of
    // squares added in another order moved it 0.0029 -> 0.0113, draft 4
    // alone, the other five as before).
    const std::size_t nd = sd.size() / (std::size_t)B;
    ASSERT_TRUE(nd > 0 && s32.size() % nd == 0);
    if (nd == 0 || s32.size() % nd != 0) { return; }
    const std::size_t per = s32.size() / nd;
    std::vector<double> ours, refs;
    std::printf("[dsv41] DSpark on fp32's picks, each draft:");
    for (std::size_t k = 0; k < nd; ++k) {
      ours.push_back(rel_l2_(sl.data() + k * per, s32.data() + k * per, per));
      refs.push_back(rel_l2_(sr.data() + k * per, s32.data() + k * per, per));
      std::printf(" %.4f/%.4f", ours.back(), refs.back());
    }
    auto median = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    const double mo = median(ours), mr = median(refs);
    int jumped = 0;
    for (double v : ours) { jumped += v > 4.0 * mo; }
    std::printf("\n[dsv41] DSpark on fp32's picks: median draft %.5f (ref "
                "bf16 %.5f), %d of %zu jumped\n", mo, mr, jumped, nd);
    EXPECT_TRUE(mo < 1.5 * mr + 1e-3);
    EXPECT_TRUE(2 * jumped <= (int)nd);
  }
}

// The vision tower: the picture's patches as the processor makes them
// (Pillow's bicubic contain + gray pad, normalized, rounded), and the span's
// rows (ViT, aligner, the learned markers) against the fp32 reference.
TEST(deepseek_v41, vision_matches_reference)
{
  const std::string gd = env_("VPIPE_DSV41_SMALL_GOLDEN");
  if (gd.empty()) { return; }
  const fs::path md = fs::path(gd) / "model";
  const FlexData meta = read_json_(fs::path(gd) / "meta.json");
  auto mo = meta.as_object();
  if (!mo.contains("vl_grid")) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  ASSERT_TRUE(mc != nullptr);
  std::string err;
  auto img = decode_image_file(sess.ffmpeg_libraries(),
                               (fs::path(gd) / "vl_image.png").string(),
                               &err);
  ASSERT_TRUE(img.has_value());
  if (!img) { std::printf("[dsv41] %s\n", err.c_str()); return; }
  DeepseekV41Config cfg;
  ASSERT_TRUE(DeepseekV41Config::from_dir(md.string(), &cfg, &err));
  MetalDeepseekV41Vision::Plan pl;
  const std::vector<float> px = MetalDeepseekV41Vision::patches(
      cfg, img->rgb.data(), img->height, img->width, &pl, true);
  const std::vector<float> rpx =
      read_raw_<float>(fs::path(gd) / "vl_patches.f32");
  const std::vector<std::int32_t> grid = ints_(meta, "vl_grid");
  EXPECT_TRUE(pl.vit_h == grid[0] && pl.vit_w == grid[1]);
  ASSERT_TRUE(px.size() == rpx.size());
  if (px.size() != rpx.size()) { return; }
  std::size_t off = 0;
  for (std::size_t i = 0; i < px.size(); ++i) { off += px[i] != rpx[i]; }
  std::printf("[dsv41] vision: grid %dx%d, %zu of %zu patch values "
              "differ\n", pl.vit_h, pl.vit_w, off, px.size());
  for (const auto& [w, h] : std::vector<std::pair<int, int>>{
           {128, 128}, {1024, 1024}, {4000, 4000}, {4000, 1000}}) {
    const auto q = MetalDeepseekV41Vision::plan(cfg, w, h);
    std::printf("[dsv41] a %dx%d picture: %dx%d cells, %d tokens\n", w, h,
                q.llm_w, q.llm_h, q.span());
  }
  EXPECT_TRUE(off == 0);
  const double bar = mo.at("bf16_vs_f32_vl_span").as_real();
  const std::vector<float> s32 =
      read_raw_<float>(fs::path(gd) / "vl_span_f32.f32");
  for (const bool bf16 : {true, false}) {
    auto v = MetalDeepseekV41Vision::load(WeightSet::open(md.string(), nullptr),
                                          mc, cfg, bf16, &err);
    ASSERT_TRUE(v != nullptr);
    if (v != nullptr && bf16) {
      // the tower holds what the plan books for it
      std::size_t ex = 0, rest = 0, vis = 0;
      ASSERT_TRUE(MetalDeepseekV41ModelExec::expert_bytes(md.string(), &ex,
                                                          &rest, &vis));
      std::printf("[dsv41] vision tower holds %zu bytes, booked %zu\n",
                  v->resident_bytes(), vis);
      EXPECT_TRUE(v->resident_bytes() == vis);
    }
    if (v == nullptr) { std::printf("[dsv41] %s\n", err.c_str()); return; }
    const auto r = v->encode_patches(rpx, pl);
    ASSERT_TRUE(r.rows.size() == s32.size());
    if (r.rows.size() != s32.size()) { return; }
    const double rel = rel_l2_(r.rows.data(), s32.data(), s32.size());
    std::printf("[dsv41] vision %s: span rows %.5f from fp32 (ref bf16 "
                "%.5f), %d tokens\n", bf16 ? "bf16" : "f16", rel, bar,
                r.n_tokens);
    EXPECT_TRUE(rel < 1.5 * bar);
    // the image turn: the span spliced at its positions
    const std::vector<std::int32_t> ids = ints_(meta, "vl_ids");
    const int start = (int)mo.at("vl_span_start").as_int();
    MetalDeepseekV41Model::Options opt;
    opt.use_bf16 = bf16;
    auto m = MetalDeepseekV41Model::load(
        WeightSet::open(md.string(), nullptr), mc, opt, &err);
    ASSERT_TRUE(m != nullptr);
    if (m == nullptr) { return; }
    MetalDeepseekV41Model::Media media;
    for (int k = 0; k < r.n_tokens; ++k) {
      media.positions.push_back(start + k);
    }
    media.rows = r.rows.data();
    std::vector<float> lg;
    ASSERT_TRUE(m->forward(1, ids, &lg, true, &err, &media));
    const std::vector<float> l32 =
        read_raw_<float>(fs::path(gd) / "vl_logits_f32.f32");
    const double lbar = mo.at("bf16_vs_f32_vl_logits").as_real();
    const double lrel = lg.size() == l32.size()
                            ? rel_l2_(lg.data(), l32.data(), lg.size()) : 1e9;
    std::printf("[dsv41] image turn %s: logits %.5f from fp32 (ref bf16 "
                "%.5f), %zu rows\n", bf16 ? "bf16" : "f16", lrel, lbar,
                ids.size());
    EXPECT_TRUE(lrel < 1.5 * lbar);
  }
}

// Timing at the release's widths: `dump_deepseek_v41_golden.py small --size
// wide` (four layers, sixteen experts, everything resident). A prefill and
// decode steps, the per-forward split printed with VPIPE_DSV41_PROFILE.
// VPIPE_DSV41_WIDE = that dir.
TEST(deepseek_v41, wide_model_timing)
{
  const std::string gd = env_("VPIPE_DSV41_WIDE");
  if (gd.empty()) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  std::string err;
  MetalDeepseekV41Model::Options opt;
  opt.dspark = env_("VPIPE_DSV41_WIDE_NO_SPARK").empty();
  auto m = MetalDeepseekV41Model::load(
      WeightSet::open((fs::path(gd) / "model").string(), nullptr), mc, opt,
      &err);
  ASSERT_TRUE(m != nullptr);
  if (m == nullptr) { std::printf("[dsv41] %s\n", err.c_str()); return; }
  const int P = std::atoi(env_("VPIPE_DSV41_WIDE_ROWS").empty()
                              ? "512" : env_("VPIPE_DSV41_WIDE_ROWS").c_str());
  std::vector<std::int32_t> ids((std::size_t)P);
  for (int i = 0; i < P; ++i) {
    ids[(std::size_t)i] = 1000 + (i * 7919) % 90000;
  }
  std::vector<float> lg;
  auto t0 = std::chrono::steady_clock::now();
  ASSERT_TRUE(m->forward(1, ids, &lg, false, &err));
  auto t1 = std::chrono::steady_clock::now();
  const int V = m->config().vocab;
  std::int32_t t = argmax_(lg.data(), V);
  const std::string ws = env_("VPIPE_DSV41_WIDE_STEPS");
  const int steps = ws.empty() ? 16 : std::atoi(ws.c_str());
  double busy0 = 0;
  unsigned long long nb0 = 0;
  metal_compute::CommandStream::gpu_time_stats(&busy0, &nb0);
  for (int i = 0; i < steps; ++i) {
    ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>{t}, &lg, false,
                           &err));
    t = argmax_(lg.data(), V);
  }
  auto t2 = std::chrono::steady_clock::now();
  auto sec = [](auto a, auto b) {
    return std::chrono::duration<double>(b - a).count();
  };
  std::printf("[dsv41] wide (%d layers): prefill %d rows %.2f s, decode "
              "%.1f ms a token\n", m->config().n_layers, P, sec(t0, t1),
              sec(t1, t2) / steps * 1000);
  {
    double busy1 = 0;
    unsigned long long nb1 = 0;
    metal_compute::CommandStream::gpu_time_stats(&busy1, &nb1);
    if (nb1 > nb0) {
      std::printf("[dsv41] wide: the GPU busy %.1f ms a token in %.1f "
                  "command buffers (VPIPE_MC_GPU_TIME)\n",
                  (busy1 - busy0) / steps, (double)(nb1 - nb0) / steps);
    }
  }
  // VPIPE_MC_GPU_TIME=2: one more token's buffers, each from its commit:
  // when the GPU started and ended it, and the host's gap to the next
  if (env_("VPIPE_MC_GPU_TIME") == "2") {
    (void)metal_compute::CommandStream::take_gpu_spans();
    ASSERT_TRUE(m->forward(1, std::vector<std::int32_t>{t}, &lg, false,
                           &err));
    const auto sp = metal_compute::CommandStream::take_gpu_spans();
    for (std::size_t i = 0; i < sp.size(); ++i) {
      const auto& x = sp[i];
      std::printf("[dsv41] span %zu: commit +%.3f ms, start +%.3f, end "
                  "+%.3f (GPU %.3f), next commit %+.3f\n", i, 0.0,
                  (x.start - x.commit) * 1e3, (x.end - x.commit) * 1e3,
                  (x.end - x.start) * 1e3,
                  i + 1 < sp.size() ? (sp[i + 1].commit - x.end) * 1e3 : 0.0);
    }
  }
  // a speculative round's pieces: a verify of 1 + 5 rows (weights read
  // once for all its rows), and a draft (with the head loaded)
  {
    const int reps = 8;
    std::vector<std::int32_t> v6(6);
    for (int i = 0; i < 6; ++i) { v6[(std::size_t)i] = 1000 + 37 * i; }
    std::vector<float> rows;
    const auto a0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
      ASSERT_TRUE(m->branch(1, 50));
      ASSERT_TRUE(m->forward(50, v6, &rows, true, &err));
      m->release(50);
    }
    const auto a1 = std::chrono::steady_clock::now();
    std::printf("[dsv41] wide: a 6-row verify %.1f ms (a decode step %.1f)"
                "\n", sec(a0, a1) / reps * 1000, sec(t1, t2) / steps * 1000);
    if (m->has_dspark()) {
      std::vector<std::int32_t> dr;
      std::vector<float> cf;
      const auto b0 = std::chrono::steady_clock::now();
      for (int r = 0; r < reps; ++r) {
        ASSERT_TRUE(m->dspark_draft(1, t, &dr, &cf, nullptr, &err));
      }
      const auto b1 = std::chrono::steady_clock::now();
      std::printf("[dsv41] wide: a DSpark draft (%zu stage, %d rows) %.1f "
                  "ms\n", (std::size_t)m->config().n_mtp,
                  m->config().dspark_block, sec(b0, b1) / reps * 1000);
    }
  }
  // what the plan books for a forward is what it holds
  const double held = (double)m->run_held();
  const double booked = (double)MetalDeepseekV41Model::run_bytes(
      m->config(), P, P + steps, m->has_dspark());
  std::printf("[dsv41] run state held %.1f MB, booked %.1f MB\n",
              held / 1048576.0, booked / 1048576.0);
  EXPECT_TRUE(held <= booked && booked <= 1.1 * held + (16 << 20));
}
