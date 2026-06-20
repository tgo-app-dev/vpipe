#include "stages/realtime-vqa-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/i18n.h"
#include "common/lmdb-cursor.h"
#include "common/lmdb-db.h"
#include "common/lmdb-env.h"
#include "common/lmdb-txn.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"
#include "stages/trigger-beat.h"
// realtime-vqa-stage.cc is compiled only under VPIPE_BUILD_APPLE_SILICON
// (see stages/CMakeLists.txt), so the TensorBeat payload (used to sense PCM
// audio beats on iport2) is always available here.
#include "apple-silicon/tensor-beat.h"
// GPU VLM-input resampler (both builds): metal-compute letterbox kernel +
// reusable Shared buffers. Available whenever Apple-Silicon is, MLX or not.
#include "apple-silicon/metal-compute/image-ops.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-storage.h"
#include <cstring>


// No-MLX metal path: MLX-free headers driving generation on the
// metal-compute backend. chat-template / loaded-language-model / sampler /
// metal-qwen-vision come in via the stage header.
#if defined(VPIPE_BUILD_APPLE_SILICON)
#include "generative-models/gemma4/gemma4-unified-embedder.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/tokenizer.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/tensor-beat-bridge.h"
#include "apple-silicon/tensor-beat.h"
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

constexpr int kMinIdleTicks = 2;

// Debug: when VPIPE_VQA_DUMP_FRAMES=<dir> is set, write the EXACT planar
// [3,H,W] u8 RGB buffer the VLM encoder is about to consume to a P6 PPM
// (<dir>/<who>_<ts>.ppm). Lets a live run capture the real encoder input so
// a frame-DATA bug (e.g. a video-to-rgb NV12 stride/crop/colourspace error)
// can be told apart from an LM/encoder bug -- the LM path itself is verified
// correct against ffmpeg-extracted frames in llm-gemma4-model-exec.cc.
void
maybe_dump_frame_(const std::uint8_t* rgb, int H, int W, const char* who,
                  std::uint64_t ts)
{
  const char* dir = std::getenv("VPIPE_VQA_DUMP_FRAMES");
  if (!dir || !*dir || !rgb || H <= 0 || W <= 0) { return; }
  char path[1200];
  std::snprintf(path, sizeof path, "%s/%s_%020llu.ppm", dir, who,
                static_cast<unsigned long long>(ts));
  std::FILE* f = std::fopen(path, "wb");
  if (!f) { return; }
  std::fprintf(f, "P6\n%d %d\n255\n", W, H);
  const std::size_t plane = static_cast<std::size_t>(H) * W;
  std::vector<std::uint8_t> row(static_cast<std::size_t>(W) * 3);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const std::size_t p = static_cast<std::size_t>(y) * W + x;
      row[(std::size_t)x * 3 + 0] = rgb[0 * plane + p];
      row[(std::size_t)x * 3 + 1] = rgb[1 * plane + p];
      row[(std::size_t)x * 3 + 2] = rgb[2 * plane + p];
    }
    std::fwrite(row.data(), 1, row.size(), f);
  }
  std::fclose(f);
}

// ===================================================================
// Prompt strings -- ALL model-facing instruction text lives in this one
// region so it is easy to find and tweak. Add new prompts here.
//
// The prompts are LOCALIZED. Each is provided per supported UI locale
// (en-us / zh-cn / zh-tw) and selected at scene time by the stage's
// effective language -- the "language" config key, else the session
// language (see RealtimeVqaStage::effective_language_). These are MODEL-
// facing prompts, not UI chrome, so the catalogue lives here next to the
// stage rather than in common/i18n.* (documented as UI-chrome only).
// en-us is the fallback; add a locale by extending VqaPrompts and the
// kPrompts* table below, then map it in vqa_prompts_for_.
// ===================================================================

// ---- en-us (the fallback locale) ----

// The "describe this scene" instruction, shared by both prompt builders so
// the MLX and metal paths can't drift. (They were already identical; this
// makes the single source of truth explicit.)
constexpr const char* kDescribePrompt =
    "Describe what is happening in this scene. General requirements:\n"
    "- Focus on movements and changes over time\n"
    "- Note direction of movement when visible\n"
    "- Describe gestures and interactions between subjects\n"
    "- Avoid describing the background, environment or clothing details\n"
    "- Avoid assumptions or subjective interpretations\n"
    "- Be specific and concise\n"
    "- Take the summary before the scene as a reference. Do not repeat "
    "previous descriptions unless the actions described are part of a "
    "continuous, unified sequence in the current scene.\n";

// Gemma-4 e4b variant of the describe instruction. Gemma clams up under
// the heavy negative-constraint list above -- the "Avoid ... / Be concise
// / Do not repeat" bullets suppress it into a terse one-liner, or (on weak
// / ambiguous frames) an outright "you have not provided the actual visual
// frames" refusal -- while Qwen3.5-VL is robust to them. Verified in
// tests/unit-tests/llm-gemma4-model-exec.cc (realtime_vqa_prompt_diag): the
// heavy prompt yields a terse non-answer; this positive-framed variant
// keeps the SAME intent (movements / gestures / interactions, motion
// direction, prev-scene continuation) and produces rich, grounded
// descriptions without refusing.
[[maybe_unused]] constexpr const char* kDescribePromptGemma =
    "Briefly describe what is happening in this video in 2-3 sentences. "
    "Focus on what the people and animals are doing -- their movements, "
    "gestures, and interactions -- and note the direction of any motion. "
    "A short summary of the moments just before these frames may be "
    "provided as background; build on it and describe what is new.\n";

// Audio interpret prompt: m_interpret_audio_ asks the LM in isolation (the
// audio is the only non-text content there, so it is salient) for a short
// sound phrase added to the per-scene sound timeline.
// (Used by the no-MLX metal interpret path only -> maybe_unused for MLX.)
[[maybe_unused]] constexpr const char* kAudioInterpretPrompt =
    "Listen to the audio for this scene. In a few words, what sound is "
    "heard? If it is speech, tell the language, and content you heard.";

// Default per-question preamble (config key "question_preamble"): prepended
// to every per-question user turn to steer the answer format. An empty
// config value disables it.
constexpr const char* kQuestionPreambleDefault =
    "Please answer the question with a clear yes/no/unknown first, then a "
    "concise explanation only if necessary.";

// build_preamble_ fragments: the per-scene user-turn preamble (local
// date/time anchor + optional prior-scene recap + audio sound timeline).
constexpr const char* kPreambleDateTimePrefix =
    "The current local date and time is: ";
constexpr const char* kPreamblePrevScenePrefix =
    "Here's the summary of the scene before the first frame. "
    "Please take the summary only as a reference. Do not repeat it unless "
    "the actions described are part of a continuous, unified sequence "
    "connected to the current scene: ";
constexpr const char* kPreambleAudioHeader =
    "Sounds detected during this scene (timestamps are aligned with the "
    "video frame markers below):\n";
constexpr const char* kPreambleNoAudio =
    "No audible sound has been tagged confidently during this scene.\n";

// ---- zh-cn (Simplified Chinese) ----

constexpr const char* kDescribePromptZhCn =
    "请描述这个视频中正在发生的事情。总体要求：\n"
    "- 关注人物、动物、车辆的动作和变化\n"
    "- 尽可能注明对象的移动方向\n"
    "- 描述对象之间的手势和互动\n"
    "- 不要描述背景、环境或衣着细节\n"
    "- 不要做出假设或主观解读\n"
    "- 要具体而简洁。\n";
constexpr const char* kAudioInterpretPromptZhCn =
    "下面是这个场景的音频。请用几个词说明听到了什么声音？如果是语音，"
    "请说明语言以及你听到的内容。";
constexpr const char* kQuestionPreambleZhCn =
    "请先明确地用“是/否/不知道”回答问题，再在必要时给出简短的解释。";
constexpr const char* kPreambleDateTimePrefixZhCn =
    "当前本地时间是：";
constexpr const char* kPreamblePrevScenePrefixZhCn =
    "以下是本段视频之前场景的摘要。请只作为回答问题的参考。"
    "除非所描述的动作是当前场景中出现的动作的一部分，否则不要重复这个描述：";
constexpr const char* kPreambleAudioHeaderZhCn =
    "在本场景中检测到了以下声音（时间戳与下方的视频帧标记对齐）：\n";
constexpr const char* kPreambleNoAudioZhCn =
    "在本场景中未能识别出任何明显的的声音。\n";

// ---- zh-tw (Traditional Chinese) ----

constexpr const char* kDescribePromptZhTw =
    "請描述這個場景中正在發生的事情。總體要求：\n"
    "- 關注人物、動物、車輛的動作和變化\n"
    "- 盡可能註明對象的移動方向\n"
    "- 描述對象之間的手勢和互動\n"
    "- 不要描述背景、環境或衣著細節\n"
    "- 不要做出假設或主觀解讀\n"
    "- 要具體而簡潔。\n";
constexpr const char* kAudioInterpretPromptZhTw =
    "下面是這個場景的音訊。請用幾個詞說明聽到了什麼聲音？如果是語音，"
    "請說明語言以及你聽到的內容。";
constexpr const char* kQuestionPreambleZhTw =
    "請先明確地用「是/否/不知道」回答問題，再在必要時再給出簡短的解釋。";
constexpr const char* kPreambleDateTimePrefixZhTw =
    "目前本地時間為：";
constexpr const char* kPreamblePrevScenePrefixZhTw =
    "以下是第一幀之前場景的摘要。請僅作爲回答問題的參考。"
    "除非其描述的動作是當前場景中出現的動作的一部分，否則不要重複此摘要：";
constexpr const char* kPreambleAudioHeaderZhTw =
    "在本場景中偵測到了以下聲音（時間戳與下方的視訊幀標記對齊）：\n";
constexpr const char* kPreambleNoAudioZhTw =
    "在本場景中未能辨識出任何明確的聲音。\n";

// One localized prompt set. The fields mirror the en-us constants above;
// every shipped locale populates all of them. The "<X.Y seconds>" audio
// markers and the wall-clock datetime stay locale-neutral (they must line
// up with the chat template's video frame markers), so they are not part
// of this table.
struct VqaPrompts {
  const char* describe;
  const char* audio_interpret;
  const char* question_preamble;
  const char* preamble_datetime_prefix;
  const char* preamble_prevscene_prefix;
  const char* preamble_audio_header;
  const char* preamble_no_audio;
};

constexpr VqaPrompts kPromptsEn = {
    kDescribePrompt,          kAudioInterpretPrompt,
    kQuestionPreambleDefault, kPreambleDateTimePrefix,
    kPreamblePrevScenePrefix, kPreambleAudioHeader,
    kPreambleNoAudio};
constexpr VqaPrompts kPromptsZhCn = {
    kDescribePromptZhCn,          kAudioInterpretPromptZhCn,
    kQuestionPreambleZhCn,        kPreambleDateTimePrefixZhCn,
    kPreamblePrevScenePrefixZhCn, kPreambleAudioHeaderZhCn,
    kPreambleNoAudioZhCn};
constexpr VqaPrompts kPromptsZhTw = {
    kDescribePromptZhTw,          kAudioInterpretPromptZhTw,
    kQuestionPreambleZhTw,        kPreambleDateTimePrefixZhTw,
    kPreamblePrevScenePrefixZhTw, kPreambleAudioHeaderZhTw,
    kPreambleNoAudioZhTw};

// Select the prompt set for a normalized locale tag (en-us / zh-cn /
// zh-tw). Any other / unsupported tag uses en-us (the fallback).
const VqaPrompts&
vqa_prompts_for_(std::string_view norm_lang)
{
  if (norm_lang == "zh-cn") { return kPromptsZhCn; }
  if (norm_lang == "zh-tw") { return kPromptsZhTw; }
  return kPromptsEn;
}

// Pick the describe instruction for the model family. Reverted: Gemma now
// uses the SAME shared/Qwen prompt as every other family. The Gemma-
// specific variant (kDescribePromptGemma) was introduced to dodge "you
// have not provided the visual frames" refusals + terse meta-answers --
// but those were the bf16 multimodal splice bug (garbage image embeddings),
// now fixed. Flip back to kDescribePromptGemma for "gemma" if the shared
// prompt regresses quality (the variant is en-us only -- localize it too
// if it is reinstated).
const char*
describe_prompt_for_(const VqaPrompts& p, const genai::ChatTemplate* tpl)
{
  (void)tpl;
  return p.describe;
}

// Top (highest-score) class label from an audio-tagging FlexData object.
// The audio-tagging stage emits "tags" already sorted by descending
// score, so element 0 is the dominant class. Returns empty on a
// missing/malformed payload. Shared by the (path-agnostic) consume_audio_.
std::string
audio_top_label_(const FlexData& fd)
{
  if (!fd.is_object()) {
    return {};
  }
  auto root = fd.as_object();
  if (!root.contains("tags")) {
    return {};
  }
  FlexData tags = root.at("tags");
  if (!tags.is_array()) {
    return {};
  }
  auto arr = tags.as_array();
  if (arr.empty()) {
    return {};
  }
  FlexData t0 = arr.at(0);
  if (!t0.is_object()) {
    return {};
  }
  auto to = t0.as_object();
  if (!to.contains("label")) {
    return {};
  }
  return std::string(to.at("label").as_string(""));
}


// Format a `timestamp_us` (microseconds since UTC epoch, the
// convention TensorBeat.sideband.timestamp_us follows) as a local-
// time `YYYY-MM-DD HH:MM:SS.uuuuuu` string. ts_us == 0 (the
// "no-timestamp" sentinel callers use in this file) returns "(none)".
// Used by both the MLX and metal scene logging, so NOT gated on MLX.
std::string
ts_human_(std::uint64_t ts_us)
{
  if (ts_us == 0) {
    return "(none)";
  }
  const std::time_t sec =
      static_cast<std::time_t>(ts_us / 1000000ull);
  const unsigned long long us = ts_us % 1000000ull;
  std::tm tm{};
  ::localtime_r(&sec, &tm);
  char date_buf[32];
  std::strftime(date_buf, sizeof(date_buf),
                "%Y-%m-%d %H:%M:%S", &tm);
  char out[64];
  std::snprintf(out, sizeof(out), "%s.%06llu", date_buf, us);
  return out;
}

// Format ts_us into a natural-language local-time string for the LLM
// prompt preamble, e.g. "Tuesday, 2026-05-26 07:48:04". When ts_us is
// 0 (the "no-timestamp" sentinel used when a scene has no sideband
// timestamp) we fall back to the wall-clock now() so the model still
// sees a sensible "current time" anchor.
std::string
local_prompt_time_(std::uint64_t ts_us)
{
  std::time_t sec;
  if (ts_us == 0) {
    sec = std::time(nullptr);
  } else {
    sec = static_cast<std::time_t>(ts_us / 1000000ull);
  }
  std::tm tm{};
  ::localtime_r(&sec, &tm);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%A, %Y-%m-%d %H:%M:%S", &tm);
  return buf;
}

// 8-byte big-endian microseconds-since-epoch LMDB key. Matches the
// rtsp-capture `<camera>-videos` key encoding so cursor iteration is
// time-ordered regardless of host byte order. Shared by both paths'
// LMDB persistence.
std::string
be64_us_key_(std::uint64_t us)
{
  std::string out(8, '\0');
  for (int i = 0; i < 8; ++i) {
    out[7 - i] = static_cast<char>(us & 0xff);
    us >>= 8;
  }
  return out;
}

// True iff `stored` (a parsed video-questions row) matches `current` in
// both count and order. Anything missing / malformed -> mismatch (we'll
// re-stamp). Shared by sync_questions_record_.
bool
questions_match_(const FlexData& stored, const std::vector<std::string>& current)
{
  if (!stored.is_object()) { return false; }
  auto root = stored.as_object();
  if (!root.contains("questions")) { return false; }
  const FlexData& q = root.at("questions");
  if (!q.is_array()) { return false; }
  auto arr = q.as_array();
  if (arr.size() != current.size()) { return false; }
  for (std::size_t i = 0; i < current.size(); ++i) {
    const FlexData& e = arr.at(i);
    if (!e.is_string()) { return false; }
    if (std::string_view(current[i]) != e.as_string("")) { return false; }
  }
  return true;
}

}  // namespace

RealtimeVqaStage::RealtimeVqaStage(const SessionContextIntf* s,
                                   std::string               id,
                                   std::vector<InEdge>       iports,
                                   FlexData                  config)
  : TypedStage<RealtimeVqaStage>(s, std::move(id), std::move(iports),
                                 std::move(config))
{
  // Side-effecting setup runs for every config: a stage must
  // construct successfully for any config so a graph can be built/
  // edited before required fields are supplied. Config problems are
  // recorded via fail_config (first message wins) and deferred to
  // launch.
  allocate_oports(spec().oports.size());

  // Scalar attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default. Clamps repair out-of-range
  // overrides.
  _hf_dir               = attr_str("hf_dir");
  _models_db            = attr_str("models_db");
  if (_models_db.empty()) {
    _models_db = "models";
  }
  _coreml_vision_path   = attr_str("coreml_vision_path");
  _compute_dtype        = attr_str("compute_dtype");
  // UI/prompt locale for the built-in scene prompts. An explicit
  // "language" config pins them (normalized to a supported tag; an
  // unsupported tag normalizes to "" and falls back to inherit); left
  // unset, effective_language_() follows the session language at scene
  // time. Resolved before question_preamble (which uses it for its
  // locale default).
  _language             = normalize_language(attr_str("language"));
  _page_tokens          = static_cast<int>(attr_int("page_tokens"));
  _max_new_tokens       = static_cast<int>(attr_int("max_new_tokens"));
  _vlm_in_w             = static_cast<int>(attr_int("vlm_input_width"));
  _vlm_in_h             = static_cast<int>(attr_int("vlm_input_height"));
  _vlm_max_soft_tokens  =
      static_cast<int>(attr_int("vlm_max_soft_tokens"));
  _max_frame_gap_ms     = static_cast<int>(attr_int("max_frame_gap_ms"));
  _idle_ticks_to_end    = static_cast<int>(attr_int("idle_ticks_to_end"));
  if (_idle_ticks_to_end < kMinIdleTicks) {
    _idle_ticks_to_end = kMinIdleTicks;
  }
  _max_frames_per_scene =
      static_cast<int>(attr_int("max_frames_per_scene"));
  if (_max_frames_per_scene < 1) { _max_frames_per_scene = 1; }
  _catch_up_drop        = static_cast<int>(attr_int("catch_up_drop"));
  if (_catch_up_drop < 0) { _catch_up_drop = 0; }
  _batched_decode       = attr_bool("batched_decode");
  _pipelined_decode     = attr_bool("pipelined_decode");
  _prev_scene_recap     = attr_bool("prev_scene_recap");
  _video_fps            = static_cast<float>(attr_real("video_fps"));
  if (_video_fps <= 0.0f) { _video_fps = 1.0f; }

  // Per-question preamble: an explicit config value wins (including ""
  // to disable it); left unset, use the locale default for the effective
  // language (NOT the English ConfigKey default, which is for the schema
  // / UI only). Overridden below when the config carries the key.
  _question_preamble =
      vqa_prompts_for_(effective_language_()).question_preamble;

  // Tri-state / composite attributes (disable_thinking + the
  // string-or-array questions) have no flat ConfigKey form and are read
  // from the config directly.
  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("question_preamble")) {
      _question_preamble = attr_str("question_preamble");
    }
    if (root.contains("disable_thinking")) {
      _disable_thinking =
          root.at("disable_thinking").as_bool(false);
    }
    if (root.contains("questions")) {
      const FlexData& q = root.at("questions");
      if (q.is_string()) {
        _questions.emplace_back(q.as_string(""));
      } else if (q.is_array()) {
        auto a = q.as_array();
        _questions.reserve(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
          if (a.at(i).is_string()) {
            _questions.emplace_back(a.at(i).as_string(""));
          }
        }
      }
    }
  }

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Sampler knobs are flat `sampler_*` attributes; each defaults to the
  // SamplerParams default (kSpec.attrs), so leaving them all unset gives
  // greedy/argmax decoding. attr_* resolves config-else-default.
  _sampler_params.temperature =
      static_cast<float>(attr_real("sampler_temperature"));
  _sampler_params.top_k  = static_cast<int>(attr_int("sampler_top_k"));
  _sampler_params.top_p  = static_cast<float>(attr_real("sampler_top_p"));
  _sampler_params.min_p  = static_cast<float>(attr_real("sampler_min_p"));
  _sampler_params.repetition_penalty =
      static_cast<float>(attr_real("sampler_repetition_penalty"));
  _sampler_params.presence_penalty =
      static_cast<float>(attr_real("sampler_presence_penalty"));
  _sampler_params.seed =
      static_cast<std::uint64_t>(attr_uint("sampler_seed"));
#endif

  if (_hf_dir.empty()) {
    fail_config(fmt(
        "RealtimeVqaStage('{}'): config.hf_dir is required (non-empty "
        "string)", this->id()));
  }
  if (_questions.empty()) {
    fail_config(fmt(
        "RealtimeVqaStage('{}'): config.questions must be a non-empty "
        "string or array of strings", this->id()));
  }
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "VLM model: a models-DB key (registered by model-fetch) or an "
          "HF-style model dir; a DB key wins over a same-named path",
   .suggest_db = "models"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db model-fetch registers into", .def_str = "models"},
  {.key = "coreml_vision_path", .type = ConfigType::String,
   .doc = "pre-converted CoreML vision tower: a models-DB key (registered "
          "by model-fetch) or an mlpackage path; a DB key wins over a "
          "same-named path",
   .def_str = "", .suggest_db = "models",
   .suggest_db_type = "qwen3.5-vision-encoder"},
  {.key = "compute_dtype", .type = ConfigType::String,
   .doc = "bf16 | f16 | f32", .def_str = "bf16"},
  {.key = "language", .type = ConfigType::String,
   .doc = "IETF UI/prompt locale for the built-in scene prompts "
          "(en-us | zh-cn | zh-tw); empty inherits the session language",
   .def_str = ""},
  {.key = "page_tokens", .type = ConfigType::Int,
   .doc = "ContextManager K/V page size", .def_int = 1024},
  {.key = "max_new_tokens", .type = ConfigType::Int,
   .doc = "per-question generation budget", .def_int = 1024},
  {.key = "vlm_input_width", .type = ConfigType::Int,
   .doc = "fixed VLM input width; GPU letterbox-resample to this (with "
          "vlm_input_height) when the tower is GPU (not CoreML). 0 = "
          "native", .def_int = 0},
  {.key = "vlm_input_height", .type = ConfigType::Int,
   .doc = "fixed VLM input height; see vlm_input_width. 0 = native",
   .def_int = 0},
  {.key = "vlm_max_soft_tokens", .type = ConfigType::Int,
   .doc = "per-frame vision soft-token budget. 0 = encoder still/default "
          "image budget (best detail). A positive value caps tokens/frame "
          "to trade detail for prefill speed. Mainly affects Gemma-4 (its "
          "dense-video budget ~64 tok/frame was too coarse for wide camera "
          "frames). E.g. 64 restores the old coarse Gemma video budget.",
   .def_int = 0},
  {.key = "max_frame_gap_ms", .type = ConfigType::Int,
   .doc = "inter-frame gap (ms) that closes a scene", .def_int = 10000},
  {.key = "idle_ticks_to_end", .type = ConfigType::Int,
   .doc = "idle ticks with no frame that close a scene (min 2)",
   .def_int = 2},
  {.key = "max_frames_per_scene", .type = ConfigType::Int,
   .doc = "safety cap; closes scene early when reached", .def_int = 16},
  {.key = "catch_up_drop", .type = ConfigType::Int,
   .doc = "frames to skip per tick when backlog exceeds it; 0=off",
   .def_int = 256},
  {.key = "batched_decode", .type = ConfigType::Bool,
   .doc = "batch question branches that share the prefix",
   .def_bool = true},
  {.key = "pipelined_decode", .type = ConfigType::Bool,
   .doc = "GPU-resident pipelined SINGLE-branch decode (metal): overlaps "
          "host/GPU per token. Batched (multi-question) decode ignores this "
          "and always uses the shrinking path (the pipelined batched path "
          "is constant-N and wastes work on staggered answers).",
   .def_bool = true},
  {.key = "prev_scene_recap", .type = ConfigType::Bool,
   .doc = "carry prior scene description into the next describe, but "
          "only across temporally-continuous scenes; false disables it",
   .def_bool = true},
  {.key = "video_fps", .type = ConfigType::Real,
   .doc = "fallback marker cadence when timestamp_us absent",
   .def_real = 1.0},
  {.key = "disable_thinking", .type = ConfigType::Bool,
   .doc = "override chat-template thinking default", .def_bool = true},
  {.key = "questions", .type = ConfigType::Any, .required = true,
   .doc = "string or array<string> asked per scene"},
  {.key = "question_preamble", .type = ConfigType::String,
   .doc = "instruction prepended to every per-question turn (answer-format "
          "steer); empty disables",
   .def_str = kQuestionPreambleDefault},
  // Flat sampler knobs; all unset -> greedy/argmax decoding.
  {.key = "sampler_temperature", .type = ConfigType::Real,
   .doc = "softmax temperature; <= 0 forces argmax", .def_real = 0.2},
  {.key = "sampler_top_k", .type = ConfigType::Int,
   .doc = "keep top-k logits; 0 = disabled", .def_int = 10},
  {.key = "sampler_top_p", .type = ConfigType::Real,
   .doc = "nucleus top-p; >= 1 = disabled", .def_real = 1.0},
  {.key = "sampler_min_p", .type = ConfigType::Real,
   .doc = "min-p floor; <= 0 = disabled", .def_real = 0.0},
  {.key = "sampler_repetition_penalty", .type = ConfigType::Real,
   .doc = "repetition penalty; 1.0 = disabled", .def_real = 1.0},
  {.key = "sampler_presence_penalty", .type = ConfigType::Real,
   .doc = "presence penalty; 0.0 = disabled", .def_real = 0.0},
  {.key = "sampler_seed", .type = ConfigType::Uint,
   .doc = "RNG seed; 0 = non-deterministic", .def_uint = 0},
};
// iport0 frames + iport1 trigger + oport0 scene answers share the video
// clock (group 0); iport2 (optional audio tags OR PCM, hence untyped)
// is drained non-blocking. Clock groups match the pre-spec behavior
// (all 0).
const PortSpec kIports[] = {
  {.name = "frames", .doc = "planar u8 RGB TensorBeat [3,H,W]; sideband "
                            "timestamp_us drives scene boundaries",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "trigger", .doc = "periodic TriggerBeat; two idle ticks close "
                             "a scene",
   .type = &typeid(TriggerPayload), .clock_group = 0},
  {.name = "audio", .doc = "optional FlexData audio-tags OR PCM "
                           "TensorBeat (untyped); timestamp-gated drain",
   .type = nullptr, .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "scene", .doc = "FlexData per closed scene: questions + answers "
                           "+ frame/timestamp metadata",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "realtime-vqa",
  .doc       = "Real-time scene VQA: accumulates video frames into scenes "
               "(gap/idle-tick boundaries), then prefills once and decodes "
               "the configured questions per scene (batched), emitting a "
               "FlexData answer bundle.",
  .display_name = "Realtime VQA",
  .category  = StageCategory::Text,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
RealtimeVqaStage::spec() const noexcept
{
  return kSpec;
}

RealtimeVqaStage::~RealtimeVqaStage() = default;

#ifdef VPIPE_BUILD_APPLE_SILICON
const std::uint8_t*
RealtimeVqaStage::resample_frame_(const std::uint8_t* rgb, int H, int W,
                                  int* out_h, int* out_w)
{
  *out_h = H;
  *out_w = W;
  // Disabled (unset), or the frame is already at the target resolution.
  if (_vlm_in_w <= 0 || _vlm_in_h <= 0) { return rgb; }
  if (W == _vlm_in_w && H == _vlm_in_h) { return rgb; }

  namespace mcn = metal_compute;
  auto* mc = session() ? session()->metal_compute() : nullptr;
  if (!mc || !mc->valid()) {
    if (!_resample_warned) {
      session()->warn(fmt(
          "RealtimeVqaStage('{}'): metal-compute unavailable; VLM-input "
          "resample to {}x{} skipped (using native frames)",
          this->id(), _vlm_in_w, _vlm_in_h));
      _resample_warned = true;
    }
    return rgb;
  }
  const std::size_t src_bytes = static_cast<std::size_t>(3) * H * W;
  const std::size_t dst_bytes =
      static_cast<std::size_t>(3) * _vlm_in_w * _vlm_in_h;
  if (!_resample_src || _resample_src->byte_size < src_bytes) {
    _resample_src = mcn::make_shared_storage(*mc, src_bytes, session());
  }
  if (!_resample_dst || _resample_dst->byte_size < dst_bytes) {
    _resample_dst = mcn::make_shared_storage(*mc, dst_bytes, session());
  }
  if (!_resample_src || !_resample_dst
      || !_resample_src->contents || !_resample_dst->contents) {
    if (!_resample_warned) {
      session()->warn(fmt(
          "RealtimeVqaStage('{}'): VLM-input resample buffer alloc "
          "failed; using native frames", this->id()));
      _resample_warned = true;
    }
    return rgb;
  }
  std::memcpy(_resample_src->contents, rgb, src_bytes);
  // Letterbox (aspect-preserving) resample -> planar u8 [3,h,w]; the
  // dispatch commits + waits, so the dst host bytes are valid on return.
  if (!mcn::letterbox_planar_u8_to_u8_chw(
          *mc, *_resample_src, W, H, *_resample_dst,
          _vlm_in_w, _vlm_in_h, session())) {
    if (!_resample_warned) {
      session()->warn(fmt(
          "RealtimeVqaStage('{}'): VLM-input resample kernel failed; "
          "using native frames", this->id()));
      _resample_warned = true;
    }
    return rgb;
  }
  *out_h = _vlm_in_h;
  *out_w = _vlm_in_w;
  return _resample_dst->contents;
}
#endif

Job
RealtimeVqaStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  auto* mgr = session() ? session()->generative_model_manager() : nullptr;
  if (!mgr) {
    session()->error(fmt(
        "RealtimeVqaStage('{}'): no GenerativeModelManager", this->id()));
    co_return;
  }
  genai::LoadSpec spec;
  spec.hf_dir        = resolve_model_dir(session(), _models_db, _hf_dir);
  spec.compute_dtype = _compute_dtype;
  spec.page_tokens   = _page_tokens;  session()->info(fmt(
      "RealtimeVqaStage('{}'): [metal/no-MLX] loading model from '{}' "
      "(dtype={}, questions={})",
      this->id(), _hf_dir, _compute_dtype, _questions.size()));
  _lm = mgr->load(spec);
  if (!_lm || !_lm->valid()) {
    session()->error(fmt(
        "RealtimeVqaStage('{}'): [metal] failed to load model from '{}'",
        this->id(), _hf_dir));
    _lm.reset();
    co_return;
  }
  _chat_tpl = genai::make_chat_template(
      _lm->config().architecture, _lm->tokenizer(), _disable_thinking);
  if (!_chat_tpl) {
    session()->warn(fmt(
        "RealtimeVqaStage('{}'): [metal] no chat template for '{}'; "
        "scenes will close without answers",
        this->id(), _lm->config().architecture));
  }
  _mvis = _lm->metal_vision_encoder();
  _mgvis = _lm->metal_gemma4_vision_encoder();
  _mgaud = _lm->metal_gemma4_audio_encoder();
  _mguni = _lm->gemma4_unified_embedder();
  if (_mguni && !_mguni->has_vision()) { _mguni = nullptr; }
  if (!_mvis && !_mgvis && !_mguni) {
    session()->warn(fmt(
        "RealtimeVqaStage('{}'): [metal] no metal vision tower "
        "(model has no vision config?); frames will be dropped",
        this->id()));
  }
  // The Gemma ViT resizes the frame INTERNALLY to its soft-token budget
  // (vlm_max_soft_tokens), so a fixed vlm_input_width/height that is
  // SMALLER than that budget grid just pre-downscales the frame -> less
  // detail at no encode-cost saving (the grid, hence cost, is set by the
  // budget regardless of input size). vlm_input is meant for fixed-input
  // towers (CoreML). For Gemma, prefer native frames (unset vlm_input) or
  // set it at/above the budget grid. One-time hint, not a hard error.
  if (_mgvis && _vlm_in_w > 0 && _vlm_in_h > 0) {
    session()->warn(fmt(
        "RealtimeVqaStage('{}'): vlm_input_width/height ({}x{}) pre-"
        "downscales frames, but the Gemma ViT resizes internally to its "
        "vlm_max_soft_tokens budget -- a small vlm_input only loses "
        "detail (no cost saving). Consider unsetting vlm_input (use "
        "native frames) for best Gemma quality.",
        this->id(), _vlm_in_w, _vlm_in_h));
  }
  // Optional CoreML vision tower override (host-f32 + metal-compute
  // letterbox preproc). When it loads it takes priority over _mvis.
  if (!_coreml_vision_path.empty()) {
    genai::CoreMLVisionEncoder::LoadSpec cm;
    cm.mlpackage_path =
        resolve_model_dir(session(), _models_db, _coreml_vision_path);
    cm.compute_units  = 2;   // ComputeUnitsAll (CPU+GPU+ANE)
    cm.patch_size     = _lm->config().vision.patch_size;
    cm.spatial_merge_size = _lm->config().vision.spatial_merge_size;
    auto enc = genai::CoreMLVisionEncoder::create(
        cm, /*runtime=*/nullptr, session());
    if (enc && enc->implemented()) {
      session()->info(fmt(
          "RealtimeVqaStage('{}'): [metal] using CoreML vision tower at "
          "'{}'", this->id(), _coreml_vision_path));
      _m_coreml = std::move(enc);
    } else {
      session()->warn(fmt(
          "RealtimeVqaStage('{}'): [metal] CoreML vision tower load from "
          "'{}' failed; falling back to the metal MLX-free tower",
          this->id(), _coreml_vision_path));
    }
  }
  session()->info(fmt(
      "RealtimeVqaStage('{}'): [metal/no-MLX] model ready ({} layers, "
      "vocab={})", this->id(), _lm->config().n_layers,
      _lm->config().vocab_size));
  co_return;
}

// ===================================================================
// Path-agnostic helpers (compiled in BOTH builds). See the header note:
// these touch only scalars / strings / FlexData / LMDB / the chat
// template, so the MLX and metal paths share ONE copy. This is what
// keeps the two paths from drifting (e.g. the LMDB persistence the metal
// path previously lacked).
// ===================================================================

FlexData
RealtimeVqaStage::build_answer_(std::uint64_t scene_idx, std::size_t n_frames,
                                std::uint64_t first_ts_us,
                                std::uint64_t last_ts_us, bool has_ts,
                                const std::string& scene_description,
                                const std::vector<std::string>& answers) const
{
  FlexData out = FlexData::make_object();
  auto root = out.as_object();
  root.insert("scene_index", FlexData::make_uint(scene_idx));
  root.insert("n_frames",
              FlexData::make_uint(static_cast<std::uint64_t>(n_frames)));
  if (has_ts) {
    root.insert("first_ts_us", FlexData::make_uint(first_ts_us));
    root.insert("last_ts_us",  FlexData::make_uint(last_ts_us));
  }
  root.insert("scene_description", FlexData::make_string(scene_description));
  FlexData qs = FlexData::make_array();
  {
    auto v = qs.as_array();
    v.reserve(_questions.size());
    for (const auto& q : _questions) { v.push_back(FlexData::make_string(q)); }
  }
  root.insert("questions", std::move(qs));
  FlexData as = FlexData::make_array();
  {
    auto v = as.as_array();
    v.reserve(answers.size());
    for (const auto& a : answers) { v.push_back(FlexData::make_string(a)); }
  }
  root.insert("answers", std::move(as));
  return out;
}

std::string
RealtimeVqaStage::effective_language_() const
{
  // An explicit (already-normalized) config language pins the prompts;
  // otherwise follow the session's current language, normalized, falling
  // back to the default locale.
  if (!_language.empty()) {
    return _language;
  }
  std::string s =
      session() ? normalize_language(session()->language()) : std::string();
  return s.empty() ? std::string(default_language()) : s;
}

std::string
RealtimeVqaStage::build_preamble_(
    std::uint64_t first_ts_us, const std::string& prev_desc,
    const std::vector<std::pair<std::uint64_t, std::string>>& audio,
    std::uint64_t base_ts_us, bool audio_wired) const
{
  const VqaPrompts& p = vqa_prompts_for_(effective_language_());
  // Always lead with the current local date+time so the model has a
  // wall-clock anchor for time-of-day reasoning.
  std::string pre = p.preamble_datetime_prefix;
  pre += local_prompt_time_(first_ts_us);
  pre += ".\n\n";
  if (!prev_desc.empty()) {
    pre += p.preamble_prevscene_prefix;
    pre += prev_desc;
    pre += "\n\n";
  }
  // Audio sound timeline: same `<X.Y seconds>` markers + scene-relative
  // base as the video frames so the model can correlate sight + sound.
  if (!audio.empty()) {
    pre += p.preamble_audio_header;
    for (const auto& ev : audio) {
      const double t_s = ev.first >= base_ts_us
          ? static_cast<double>(ev.first - base_ts_us) * 1e-6
          : 0.0;
      char buf[48];
      std::snprintf(buf, sizeof(buf), "<%.1f seconds> ", t_s);
      pre += buf;
      pre += ev.second;
      pre += "\n";
    }
    pre += "\n";
  } else if (audio_wired) {
    pre += p.preamble_no_audio;
  }
  return pre;
}

std::string
RealtimeVqaStage::prev_recap_(const std::string& prev_desc,
                              std::uint64_t      prev_last_ts_us,
                              std::uint64_t      this_first_ts_us) const
{
  if (!_prev_scene_recap || prev_desc.empty()) {
    return {};
  }
  // No reliable timestamps on either side -> can't establish continuity;
  // treat as a fresh scene and drop the recap (the safe default that
  // prevents the stale-description echo lock).
  if (prev_last_ts_us == 0 || this_first_ts_us == 0
      || this_first_ts_us < prev_last_ts_us) {
    return {};
  }
  const std::uint64_t gap_us = this_first_ts_us - prev_last_ts_us;
  const std::uint64_t thresh_us =
      static_cast<std::uint64_t>(_max_frame_gap_ms) * 1000ull;
  // Continuous == the inter-scene gap is below the same threshold that
  // splits scenes. A scene that ended on a real gap / idle stretch is a
  // new event and gets no recap.
  return (gap_us < thresh_us) ? prev_desc : std::string{};
}

Job
RealtimeVqaStage::consume_audio_(
    RuntimeContext& ctx, std::uint64_t boundary_ts_us, bool scene_active,
    std::uint64_t scene_first_ts_us,
    std::vector<std::pair<std::uint64_t, std::string>>& scene_audio)
{
  if (ctx.num_iports() < 3 || _audio_eos) {
    co_return;
  }
  // Peek the backlog in arrival (== timestamp) order; release every entry
  // strictly before the boundary (relieving the upstream backpressure),
  // leave the rest queued.
  const std::uint32_t avail = ctx.backlog(2);
  std::uint32_t consumed = 0;
  for (std::uint32_t i = 0; i < avail; ++i) {
    const BeatPayloadIntf* p = co_await ctx.peek(2, i);
    if (!p) {
      _audio_eos = true;   // iport2 closed; audio is auxiliary, don't end
      break;
    }
    // Type-sense the audio iport: an audio-tagging stage emits FlexData
    // (top-label text -> sound timeline), a pcm stage emits a PCM
    // TensorBeat (mono f32 samples -> accumulate for soft-token encoding
    // at scene close, when the LM has an audio encoder).
    if (const auto* fdp = dynamic_cast<const FlexDataPayload*>(p)) {
      if (fdp->data.is_object()) {
        auto root = fdp->data.as_object();
        const std::uint64_t ts = root.contains("timestamp_us")
            ? root.at("timestamp_us").as_uint(0)
            : 0;
        if (ts >= boundary_ts_us) {
          break;   // ordered stream: everything later is too
        }
        std::string label = audio_top_label_(fdp->data);
        if (!label.empty() && label != "Silence"
            && scene_active && ts >= scene_first_ts_us) {
          // Collapse consecutive identical labels so a sound spanning
          // several overlapping windows is one prompt line.
          if (scene_audio.empty() || scene_audio.back().second != label) {
            scene_audio.emplace_back(ts, std::move(label));
          }
        }
      }
    } else if (const auto* tbp = dynamic_cast<const TensorBeatPayload*>(p)) {
      std::uint64_t ts = 0;
      bool ts_present = false;
      if (tbp->sideband.is_object()) {
        auto sb = tbp->sideband.as_object();
        if (sb.contains("timestamp_us")) {
          ts = sb.at("timestamp_us").as_uint(0);
          ts_present = true;
        }
        if (sb.contains("sample_rate")) {
          const int sr = static_cast<int>(sb.at("sample_rate").as_uint(0));
          if (sr > 0) { _scene_pcm_sr = sr; }
        }
      }
      if (ts_present && ts >= boundary_ts_us) {
        break;   // ordered stream
      }
      // mono PCM: shape [N] or [1, N], dtype F32.
      std::size_t n = 0;
      if (tbp->dtype == TensorBeat::DType::F32) {
        if (tbp->shape.size() == 1) {
          n = static_cast<std::size_t>(tbp->shape[0]);
        } else if (tbp->shape.size() == 2 && tbp->shape[0] == 1) {
          n = static_cast<std::size_t>(tbp->shape[1]);
        }
      }
      if (n > 0 && scene_active
          && (!ts_present || ts >= scene_first_ts_us)) {
        if (_scene_pcm.empty() && ts_present) {
          _scene_pcm_first_ts_us = ts;
        }
        if (tbp->is_contiguous()) {
          const float* s = tbp->as_f32();
          _scene_pcm.insert(_scene_pcm.end(), s, s + n);
        } else {
          const AlignedVector<float> v =
              tbp->materialize_contiguous_as<float>();
          _scene_pcm.insert(_scene_pcm.end(), v.data(), v.data() + n);
        }
      }
    }
    consumed = i + 1;
  }
  if (consumed > 0) {
    ctx.release_read(2, consumed);
    _audio_beats_released += consumed;
  }
  co_return;
}

void
RealtimeVqaStage::sync_questions_record_(const std::string& camera_name,
                                         std::uint64_t      ts_us)
{
  if (_questions_checked_cameras.count(camera_name) > 0) {
    return;
  }
  // Mark checked up-front so a failure doesn't re-enter the loop every
  // frame; we log warns and move on.
  _questions_checked_cameras.insert(camera_name);

  LmdbEnv* env = session() ? session()->lmdb_env() : nullptr;
  if (!env) {
    session()->warn(fmt(
        "RealtimeVqaStage('{}'): session lmdb_env() unavailable; "
        "skipping <{}>-video-questions sync",
        this->id(), camera_name));
    return;
  }
  const std::string db_name = camera_name + "-video-questions";
  try {
    LmdbDb db(*env, db_name);
    bool need_write = true;
    {
      LmdbTxn rtxn(*env, LmdbTxn::Mode::ReadOnly);
      LmdbCursor cur(rtxn, db);
      std::string_view k, v;
      if (cur.last(k, v)) {
        FlexData stored = FlexData::from_binary(v);
        if (questions_match_(stored, _questions)) { need_write = false; }
      }
    }
    if (!need_write) { return; }
    FlexData rec = FlexData::make_object();
    {
      auto root = rec.as_object();
      FlexData qs = FlexData::make_array();
      {
        auto v = qs.as_array();
        v.reserve(_questions.size());
        for (const auto& q : _questions) {
          v.push_back(FlexData::make_string(q));
        }
      }
      root.insert("questions", std::move(qs));
    }
    const std::string key   = be64_us_key_(ts_us);
    const std::string bytes = rec.to_binary();
    // Reuse the LmdbDb above: a second LmdbDb here would open a fresh boot
    // RW txn and deadlock against `wtxn` (LMDB serialises writers).
    LmdbTxn wtxn(*env, LmdbTxn::Mode::ReadWrite);
    db.put(wtxn, key, bytes);
    wtxn.commit();
    session()->info(fmt(
        "RealtimeVqaStage('{}'): wrote new questions epoch to '{}' "
        "(ts_us={}, {} question{})",
        this->id(), db_name, ts_us, _questions.size(),
        _questions.size() == 1 ? "" : "s"));
  } catch (const std::exception& e) {
    session()->warn(fmt(
        "RealtimeVqaStage('{}'): <{}> sync failed: {}",
        this->id(), db_name, e.what()));
  } catch (...) {
    session()->warn(fmt(
        "RealtimeVqaStage('{}'): <{}> sync failed (unknown error)",
        this->id(), db_name));
  }
}

void
RealtimeVqaStage::log_scene_qa_(const std::string&              camera_name,
                                std::uint64_t                   start_utc_us,
                                std::uint64_t                   end_utc_us,
                                const std::string&              description,
                                const std::vector<std::string>& answers)
{
  LmdbEnv* env = session() ? session()->lmdb_env() : nullptr;
  if (!env) {
    session()->warn(fmt(
        "RealtimeVqaStage('{}'): session lmdb_env() unavailable; "
        "skipping <{}>-video-qa write",
        this->id(), camera_name));
    return;
  }
  const std::string db_name = camera_name + "-video-qa";
  try {
    FlexData rec = FlexData::make_object();
    {
      auto root = rec.as_object();
      root.insert("start_utc_us", FlexData::make_uint(start_utc_us));
      root.insert("end_utc_us",   FlexData::make_uint(end_utc_us));
      root.insert("description",  FlexData::make_string(description));
      FlexData as = FlexData::make_array();
      {
        auto v = as.as_array();
        v.reserve(answers.size());
        for (const auto& a : answers) { v.push_back(FlexData::make_string(a)); }
      }
      root.insert("answers", std::move(as));
    }
    const std::string key   = be64_us_key_(start_utc_us);
    const std::string bytes = rec.to_binary();
    LmdbDb  db(*env, db_name);
    LmdbTxn txn(*env, LmdbTxn::Mode::ReadWrite);
    db.put(txn, key, bytes);
    txn.commit();
  } catch (const std::exception& e) {
    session()->warn(fmt(
        "RealtimeVqaStage('{}'): <{}> write failed: {}",
        this->id(), db_name, e.what()));
  } catch (...) {
    session()->warn(fmt(
        "RealtimeVqaStage('{}'): <{}> write failed (unknown error)",
        this->id(), db_name));
  }
}


// ===================================================================
// No-MLX metal path. Self-contained; compiled only for the metal-only
// build. Single-frame vision encode + mROPE multimodal prefill + simple
// per-question decode (greedy or host-sampled). No temporal-pair /
// batched / DeepStack / audio / LMDB (those stay on the MLX path).
// ===================================================================
#if defined(VPIPE_BUILD_APPLE_SILICON)

void
RealtimeVqaStage::m_reset_scene_()
{
  _m_imgs.clear();
  _m_frame_ts_us.clear();
  _m_first_ts_us = 0;
  _m_last_ts_us  = 0;
  _m_has_ts      = false;
  _m_last_recv_ts_us = 0;
  _m_pending_bytes.clear();
  _m_pending_h     = 0;
  _m_pending_w     = 0;
  _m_pending_ts_us = 0;
  _m_has_pending   = false;
  _scene_audio.clear();
  _scene_pcm.clear();
  _scene_pcm_first_ts_us = 0;
  // NB: _scene_camera_name is sticky -- NOT cleared here. The camera
  // doesn't change, and the camera-name sideband is intermittent per
  // frame, so we keep the last real name across scenes.
  // NB: _last_image_ts_us and _audio_eos persist across scenes (the audio
  // cursor / boundary are monotonic, not per-scene).
}

bool
RealtimeVqaStage::m_encode_frame_(const std::uint8_t* rgb, int H, int W,
                                  std::uint64_t ts_us, bool ts_present,
                                  const metal_compute::SharedBuffer* src_buf)
{
  if (!_lm || (!_mvis && !_m_coreml && !_mgvis && !_mguni)) {
    if (!_m_vision_warned) {
      session()->warn(fmt(
          "RealtimeVqaStage('{}'): [metal] vision tower unavailable; "
          "frame dropped", this->id()));
      _m_vision_warned = true;
    }
    return false;
  }
  std::uint64_t recv_ts_us = ts_us;
  if (!ts_present) {
    recv_ts_us = m_scene_active_()
        ? _m_last_recv_ts_us + static_cast<std::uint64_t>(
              1.0e6 / static_cast<double>(_video_fps))
        : 0;
  }
  _m_last_recv_ts_us = recv_ts_us;
  if (ts_present) {
    if (!_m_has_ts) { _m_first_ts_us = ts_us; _m_has_ts = true; }
    _m_last_ts_us = ts_us;
  }
  using clock = std::chrono::steady_clock;

  // Temporal-pair fusion (CoreML video export: image0+image1 merge two
  // DISTINCT consecutive frames into one token grid -- temporal_patch_
  // size = 2). Mirrors the MLX path: buffer the first frame, encode the
  // pair when its partner arrives, marker ts = the pair's average time.
  // This halves the image-token count vs encoding each frame on its own
  // (which also overflowed the LM context for longer scenes).
  if (_m_coreml && _m_coreml->supports_temporal_pair()) {
    if (_m_has_pending && _m_pending_h == H && _m_pending_w == W) {
      const auto t0 = clock::now();
      auto cr = _m_coreml->encode_pair_host(
          _m_pending_bytes.data(), rgb, H, W);
      const double enc_s =
          std::chrono::duration<double>(clock::now() - t0).count();
      _m_has_pending = false;
      _m_pending_bytes.clear();
      MImg m;
      m.embeddings = std::move(cr.embeddings);
      m.n_tokens   = cr.n_tokens;
      m.mh         = cr.grid_h;
      m.mw         = cr.grid_w;
      return m_append_mimg_(std::move(m),
                            (_m_pending_ts_us + recv_ts_us) / 2,
                            ts_us, ts_present, "pair", W, H, enc_s);
    }
    // No partner yet (or a mid-scene resolution change): flush any stale
    // pending frame, then hold this one for the next pair.
    if (_m_has_pending) { m_flush_pending_(); }
    _m_pending_bytes.assign(rgb,
        rgb + static_cast<std::size_t>(3) * H * W);
    _m_pending_h     = H;
    _m_pending_w     = W;
    _m_pending_ts_us = recv_ts_us;
    _m_has_pending   = true;
    return true;
  }

  // Single-frame path: single-image CoreML tower or the metal MLX-free
  // tower. Each frame is encoded on its own.
  const auto t0 = clock::now();
  MImg m;
  // Effective vision-tower input resolution, logged so the operator sees the
  // size the tower actually received: post-resample for the GPU towers
  // (vlm_input_width/height), native for CoreML (which letterboxes
  // internally). Defaults to the native frame; the GPU branch overwrites it.
  int eff_w = W, eff_h = H;
  if (_m_coreml) {
    // CoreML Result already reports the POST-merger token grid. Bind a
    // Shared/UMA frame buffer straight into the letterbox kernel
    // (zero-copy input); else stage the host bytes. (The CoreML tower
    // letterboxes internally, so no separate VLM-input resample here.)
    auto cr = src_buf ? _m_coreml->encode_host(*src_buf, H, W)
                      : _m_coreml->encode_host(rgb, H, W);
    m.embeddings = std::move(cr.embeddings);
    m.n_tokens   = cr.n_tokens;
    m.mh         = cr.grid_h;
    m.mw         = cr.grid_w;
  } else {
    // GPU metal towers: optionally letterbox-resample the frame to the
    // configured fixed VLM input resolution before encoding.
    int uh = H, uw = W;
    const std::uint8_t* urgb = resample_frame_(rgb, H, W, &uh, &uw);
    eff_w = uw;
    eff_h = uh;
    // Debug capture of the EXACT pixels handed to the GPU vision tower.
    maybe_dump_frame_(urgb, uh, uw, "vqa-enc", ts_us);
    if (_mgvis) {
      // Gemma-4 e4b metal ViT: native-f16 SharedBuffer rows, POST-merger
      // grid. Budget = config cap when set, else -1 = the still/default
      // image budget (~256 tok). The dense-video budget (~64) downsampled
      // wide camera frames too coarsely -> vague / hallucinated scenes.
      auto r = _mgvis->encode(
          urgb, uh, uw,
          _vlm_max_soft_tokens > 0 ? _vlm_max_soft_tokens : -1);
      m.embeddings = std::move(r.embeddings);
      m.n_tokens   = r.n_tokens;
      m.mh         = r.grid_h;
      m.mw         = r.grid_w;
    } else if (_mguni) {
      // Gemma-4-12B "unified": encoder-less shallow embedder, host-f32
      // rows routed through the TokenRef embeddings_host splice.
      if (auto r = _mguni->encode_image(urgb, uh, uw)) {
        m.n_tokens        = r->n_tokens;
        m.mh              = r->grid_h;
        m.mw              = r->grid_w;
        m.embeddings_host = std::move(r->rows);
      }
    } else {
      // Metal MLX-free tower reports PRE-merger patches -> divide by S.
      const int S = std::max(1, _mvis->config().spatial_merge);
      auto r = _mvis->encode(urgb, uh, uw);
      m.embeddings = std::move(r.embeddings);
      m.n_tokens   = r.n_tokens;
      m.mh         = r.grid_h / S;
      m.mw         = r.grid_w / S;
    }
  }
  const double enc_s =
      std::chrono::duration<double>(clock::now() - t0).count();
  return m_append_mimg_(std::move(m), recv_ts_us, ts_us, ts_present,
                        _m_coreml ? "coreml"
                                  : (_mgvis ? "metal-gemma4"
                                            : (_mguni ? "gemma4-unified"
                                                      : "metal-tower")),
                        eff_w, eff_h, enc_s);
}

bool
RealtimeVqaStage::m_append_mimg_(MImg&& m, std::uint64_t marker_ts_us,
                                 std::uint64_t real_ts_us, bool ts_present,
                                 const char* how, int W, int H, double enc_s)
{
  if (m.n_tokens <= 0 ||
      (m.embeddings.empty() && m.embeddings_host.empty())) {
    session()->warn(fmt(
        "RealtimeVqaStage('{}'): vision encode {}x{} returned null",
        this->id(), W, H));
    return false;
  }
  // Same scene-token line as the MLX path: position within the scene
  // (relative to the first frame's marker) plus the raw timestamp tag.
  const std::uint64_t base = _m_frame_ts_us.empty()
      ? marker_ts_us : _m_frame_ts_us.front();
  const double in_scene_s =
      static_cast<double>((base <= marker_ts_us)
                              ? (marker_ts_us - base) : 0) * 1e-6;
  session()->log_normal(fmt(
      "RealtimeVqaStage('{}'): scene-token {} ({}) {}x{} -> {} tok in "
      "{:.3f} s (in_scene={:.3f}s, real_ts_us={} [{}]{})",
      this->id(), (int)_m_imgs.size(), how, W, H, m.n_tokens, enc_s,
      in_scene_s, real_ts_us, ts_human_(real_ts_us),
      ts_present ? "" : " (synth)"));
  _m_imgs.push_back(std::move(m));
  _m_frame_ts_us.push_back(marker_ts_us);
  return true;
}

void
RealtimeVqaStage::m_flush_pending_()
{
  if (!_m_has_pending) { return; }
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  // Replicate the lone tail frame across the temporal patch (reference
  // last-frame padding for an odd frame count): encode_pair_host(f, f)
  // equals a single-frame tile on the two-input video model. Only
  // reachable when _m_coreml supports the pair layout (else _m_has_
  // pending is never set).
  MImg m;
  if (_m_coreml) {
    auto cr = _m_coreml->encode_pair_host(_m_pending_bytes.data(),
                                          _m_pending_bytes.data(),
                                          _m_pending_h, _m_pending_w);
    m.embeddings = std::move(cr.embeddings);
    m.n_tokens   = cr.n_tokens;
    m.mh         = cr.grid_h;
    m.mw         = cr.grid_w;
  }
  const double enc_s =
      std::chrono::duration<double>(clock::now() - t0).count();
  (void)m_append_mimg_(std::move(m), _m_pending_ts_us, _m_pending_ts_us,
                       /*ts_present=*/false, "pair-tail",
                       _m_pending_w, _m_pending_h, enc_s);
  _m_has_pending = false;
  _m_pending_bytes.clear();
}

std::string
RealtimeVqaStage::m_decode_(genai::LoadedLanguageModel::Context& ctx,
                            const genai::SamplerParams& sp)
{
  using clock = std::chrono::steady_clock;
  const auto t_start = clock::now();
  const auto* tpl = _chat_tpl.get();
  auto& tok = _lm->tokenizer();
  auto sd = tok.make_stream_decoder();
  std::string out;
  int produced = 0;
  std::int32_t cur = ctx.last_predicted_id();   // prefill's first token
  const bool argmax = genai::Sampler(sp).is_argmax();

  // MTP speculative head fast path (mtp.safetensors, Qwen3.5-OptiQ): the drafter
  // lets the verifier accept multiple tokens per forward, token-exact vs the
  // pdecode loop (greedy) or decode_pipelined (sampling -- the verify samples
  // each position). This is the SINGLE-decode path (scene description, audio
  // interpretation); the batched question decode (m_decode_batched_) never uses
  // it. rope is anchored on ctx (post-image/audio mROPE or sequential) exactly
  // as pdecode_begin reads it. Greedy OR penalty-free sampling -- the verify
  // applies no repetition/presence penalty, so a penalised sampler stays on the
  // loops below (which do apply it).
  const std::span<const std::int32_t> no_prompt;
  const bool mtp_ok = cur >= 0 && _lm->mtp_available()
      && (argmax || (sp.repetition_penalty == 1.0f
                     && sp.presence_penalty == 0.0f));
  if (mtp_ok) {
    auto is_stop = [tpl](std::int32_t id) {
      return tpl && tpl->is_stop_token(id);
    };
    auto on_toks =
        [&](std::span<const std::int32_t> toks) -> bool {
          for (std::int32_t id : toks) {
            out += tok.step(sd, id);
            ++produced;
          }
          return true;
        };
    _lm->mtp_generate(ctx, cur, _max_new_tokens, sp, is_stop, on_toks);
  } else if (_pipelined_decode
      && _lm->pdecode_begin(ctx, cur, no_prompt, sp, _max_new_tokens)) {
    // Pipelined decode, mirroring text-chat-stage's proven ordering: each
    // step DRAINS (pdecode_next) the in-flight forward BEFORE it REFILLS
    // (pdecode_commit), so a depth-1 pdecode ring (Qwen3.5) never
    // overflows. Run-ahead keeps a SECOND forward in flight only on
    // depth>=2 backends (e.g. gemma-e4b); on depth-1 the speculative
    // commit is a harmless no-op (ring already full). The earlier
    // commit-before-drain order broke HERE: the run-ahead pre-commit
    // filled the depth-1 ring, so the loop's first commit failed and the
    // scene description was truncated to the single prefill token ("The")
    // -- while the per-question branches (m_decode_batched_) were
    // unaffected. Verified by metal_pipeline_smoke on Qwen.
    const bool runahead = _lm->pdecode_supports_runahead();
    bool committed = (cur >= 0 && !(tpl && tpl->is_stop_token(cur)))
        ? _lm->pdecode_commit(ctx) : false;
    if (runahead && committed && _max_new_tokens > 1) {
      _lm->pdecode_commit(ctx);   // speculative 2nd forward (depth>=2 only)
    }
    while (produced < _max_new_tokens) {
      if (tpl && tpl->is_stop_token(cur)) { break; }
      out += tok.step(sd, cur);
      ++produced;
      if (produced >= _max_new_tokens || !committed) { break; }
      cur = _lm->pdecode_next(ctx);              // drain
      if (cur < 0) { break; }
      const bool cont = (produced + 1 < _max_new_tokens)
          && !(tpl && tpl->is_stop_token(cur));
      committed = cont ? _lm->pdecode_commit(ctx) : false;   // refill
    }
    _lm->pdecode_end(ctx);
  } else {
    genai::Sampler sampler(sp);
    if (cur >= 0) {
      sampler.prime(std::span<const std::int32_t>(&cur, 1));
    }
    while (cur >= 0 && produced < _max_new_tokens) {
      if (tpl && tpl->is_stop_token(cur)) { break; }
      out += tok.step(sd, cur);
      ++produced;
      if (produced >= _max_new_tokens) { break; }
      const std::int32_t am = _lm->next_token(ctx, cur);
      if (am < 0) { break; }
      if (sampler.is_argmax()) {
        cur = am;
      } else {
        const auto& logits = _lm->last_logits_host();
        if (logits.empty()) { cur = am; }
        else {
          cur = sampler.sample(
              std::span<const float>(logits.data(), logits.size()));
        }
      }
    }
  }
  const double dec_s =
      std::chrono::duration<double>(clock::now() - t_start).count();
  session()->info(fmt(
      "RealtimeVqaStage('{}'): decode_single {} tok in {:.3f} s "
      "({:.1f} tok/s){}",
      this->id(), produced, dec_s,
      dec_s > 0.0 ? static_cast<double>(produced) / dec_s : 0.0,
      argmax ? "" : " [sampled]"));
  return out;
}

std::string
RealtimeVqaStage::m_interpret_audio_(const std::vector<float>& audio_rows,
                                     int n_audio_tokens)
{
  const auto* tpl = _chat_tpl.get();
  if (!tpl || n_audio_tokens <= 0 || audio_rows.empty()) { return {}; }
  const std::int32_t apad = tpl->audio_pad_token_id();
  if (apad < 0) { return {}; }

  // [bos] user: <audio block> {prompt} -> model. Audio is the only
  // non-text content, so it dominates attention -> reliable, unlike when
  // buried behind the video prefix.
  std::vector<std::int32_t> ids;
  tpl->render_user_turn_audio(
      vqa_prompts_for_(effective_language_()).audio_interpret,
      n_audio_tokens, /*is_first_turn=*/true, &ids);
  std::vector<genai::TokenRef> refs;
  refs.reserve(ids.size());
  int aoff = 0;
  for (std::int32_t id : ids) {
    genai::TokenRef r;
    if (id == apad && aoff < n_audio_tokens) {
      r.kind               = genai::TokenRef::Kind::AudioTokens;
      r.embeddings_host    = &audio_rows;
      r.audio_token_offset = aoff++;
    } else {
      r.kind    = genai::TokenRef::Kind::Text;
      r.text_id = id;
    }
    refs.push_back(r);
  }

  auto ctx = _lm->make_context();
  if (!ctx.valid()) { return {}; }
  if (_lm->prefill_multimodal_metal(
          ctx, std::span<const genai::TokenRef>(refs),
          std::span<const std::pair<int, int>>{}) < 0) {
    return {};
  }
  // Greedy so the tag is stable; the prompt keeps it short and it stops at
  // the turn end.
  genai::SamplerParams sp;
  std::string phrase = m_decode_(ctx, sp);
  // Trim leading/trailing whitespace + a trailing period for a clean tag.
  std::size_t b = phrase.find_first_not_of(" \t\r\n");
  std::size_t e = phrase.find_last_not_of(" \t\r\n.");
  if (b == std::string::npos) { return {}; }
  return phrase.substr(b, e - b + 1);
}

std::vector<std::string>
RealtimeVqaStage::m_decode_batched_(
    std::vector<genai::LoadedLanguageModel::Context>& children)
{
  // ALWAYS the synchronous SHRINKING path for batched decode: it drops a
  // branch as it finishes and re-selects the matmul kernel by the current
  // active count via qmm_auto_ (so the tail runs qmv, not the wide batched
  // GEMV). The GPU-resident pipelined path overlaps host/GPU but is
  // CONSTANT-N (the overlap needs GPU-resident tokens, so it can't cheaply
  // shrink), which wastes work -- both weights AND per-branch attention --
  // once answers finish at staggered lengths. Multi-question answers are
  // staggered by nature, so pipelining is never used for batched decode.
  // (`pipelined_decode` governs SINGLE decode only -- see m_decode_.)
  return m_decode_batched_sync_(children);
}

std::vector<std::string>
RealtimeVqaStage::m_decode_batched_sync_(
    std::vector<genai::LoadedLanguageModel::Context>& children)
{
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  const auto* tpl = _chat_tpl.get();
  auto& tok = _lm->tokenizer();
  const int vocab = _lm->config().vocab_size;
  const std::size_t N = children.size();

  // Per-branch decode state. `cur` is the token to emit + feed as the next
  // step's input; it begins at the prefill's predicted token.
  struct BState {
    genai::Sampler                  sampler;
    genai::Tokenizer::StreamDecoder sd;
    std::string                   out;
    std::int32_t                  cur = -1;
    int                           produced = 0;
    bool                          done = false;
  };
  std::vector<BState> st;
  st.reserve(N);
  for (std::size_t k = 0; k < N; ++k) {
    genai::SamplerParams p = _sampler_params;
    if (p.seed != 0) { p.seed += static_cast<std::uint64_t>(k); }
    BState b{genai::Sampler(p), tok.make_stream_decoder(), {},
             children[k].last_predicted_id(), 0, false};
    if (b.cur >= 0) {
      b.sampler.prime(std::span<const std::int32_t>(&b.cur, 1));
    }
    st.push_back(std::move(b));
  }

  std::vector<float> logits;
  int total_steps = 0;
  while (true) {
    // Emit each still-active branch's current token, then collect the active
    // set for the next batched forward.
    std::vector<genai::LoadedLanguageModel::Context*> active_ctx;
    std::vector<std::int32_t>                       active_tok;
    std::vector<std::size_t>                        active_idx;
    for (std::size_t k = 0; k < N; ++k) {
      BState& b = st[k];
      if (b.done) { continue; }
      if (b.cur < 0 || (tpl && tpl->is_stop_token(b.cur))
          || b.produced >= _max_new_tokens) {
        b.done = true;
        continue;
      }
      b.out += tok.step(b.sd, b.cur);
      ++b.produced;
      if (b.produced >= _max_new_tokens) { b.done = true; continue; }
      active_ctx.push_back(&children[k]);
      active_tok.push_back(b.cur);
      active_idx.push_back(k);
    }
    if (active_ctx.empty()) { break; }
    if (!_lm->m_batched_decode_step(
            std::span<genai::LoadedLanguageModel::Context*>(
                active_ctx.data(), active_ctx.size()),
            std::span<const std::int32_t>(active_tok.data(),
                                          active_tok.size()),
            logits)) {
      // Should not happen (gated on m_batched_decode_supported upfront +
      // deterministic); truncate the active branches rather than spin.
      session()->warn(fmt(
          "RealtimeVqaStage('{}'): m_batched_decode_step failed at step "
          "{} ({} active); truncating", this->id(), total_steps,
          active_ctx.size()));
      break;
    }
    ++total_steps;
    for (std::size_t a = 0; a < active_idx.size(); ++a) {
      BState& b = st[active_idx[a]];
      const float* row = logits.data() + a * static_cast<std::size_t>(vocab);
      b.cur = b.sampler.sample(std::span<const float>(
          row, static_cast<std::size_t>(vocab)));
    }
  }

  std::vector<std::string> answers(N);
  int total_tok = 0;
  for (std::size_t k = 0; k < N; ++k) {
    total_tok += st[k].produced;
    answers[k] = std::move(st[k].out);
  }
  const double dec_s = std::chrono::duration<double>(clock::now() - t0).count();
  session()->info(fmt(
      "RealtimeVqaStage('{}'): m_decode_batched {} branches, {} steps, {} "
      "tok in {:.3f} s ({:.1f} tok/s)",
      this->id(), N, total_steps, total_tok, dec_s,
      dec_s > 0.0 ? static_cast<double>(total_tok) / dec_s : 0.0));
  return answers;
}

std::vector<std::string>
RealtimeVqaStage::m_decode_batched_pipelined_(
    std::vector<genai::LoadedLanguageModel::Context>& children)
{
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  const auto* tpl = _chat_tpl.get();
  auto& tok = _lm->tokenizer();
  const int N = static_cast<int>(children.size());

  // Per-branch host bookkeeping. The sampled tokens come from the GPU
  // (m_bdecode_next); we only stream-decode + stop-check here. CONSTANT-N:
  // every step's forward carries all N branches (the weight read is
  // amortised across N, so a finished branch costs only its cheap
  // per-branch attention); a finished branch is just not collected.
  std::vector<genai::Tokenizer::StreamDecoder> sd;
  std::vector<std::string>                   out((std::size_t)N);
  std::vector<std::int32_t>                  cur((std::size_t)N);
  std::vector<int>                           produced((std::size_t)N, 0);
  std::vector<char>                          done((std::size_t)N, 0);
  sd.reserve((std::size_t)N);
  std::vector<genai::LoadedLanguageModel::Context*> ptrs((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    sd.push_back(tok.make_stream_decoder());
    cur[(std::size_t)i] = children[(std::size_t)i].last_predicted_id();
    ptrs[(std::size_t)i] = &children[(std::size_t)i];
  }

  std::vector<std::string> answers((std::size_t)N);
  if (!_lm->m_bdecode_begin(
          std::span<genai::LoadedLanguageModel::Context*>(ptrs.data(),
                                                        ptrs.size()),
          std::span<const std::int32_t>(cur.data(), cur.size()),
          _sampler_params, _max_new_tokens)) {
    // Shouldn't happen (gated on m_bdecode_supported); fall back to sync.
    return m_decode_batched_sync_(children);
  }

  int total_steps = 0, total_tok = 0;
  std::vector<std::int32_t> toks;
  while (true) {
    // Emit each still-active branch's current token (stop-check first).
    bool any_active = false;
    for (int i = 0; i < N; ++i) {
      if (done[(std::size_t)i]) { continue; }
      const std::int32_t c = cur[(std::size_t)i];
      if (c < 0 || (tpl && tpl->is_stop_token(c))
          || produced[(std::size_t)i] >= _max_new_tokens) {
        done[(std::size_t)i] = 1;
        continue;
      }
      out[(std::size_t)i] += tok.step(sd[(std::size_t)i], c);
      ++produced[(std::size_t)i];
      if (produced[(std::size_t)i] >= _max_new_tokens) {
        done[(std::size_t)i] = 1;
        continue;
      }
      any_active = true;
    }
    if (!any_active) { break; }
    // One GPU-pipelined step over ALL N branches (commit overlaps the host
    // emit above with the GPU's forward); next() returns the N sampled ids.
    if (!_lm->m_bdecode_commit() || !_lm->m_bdecode_next(toks)
        || static_cast<int>(toks.size()) != N) {
      break;
    }
    ++total_steps;
    for (int i = 0; i < N; ++i) { cur[(std::size_t)i] = toks[(std::size_t)i]; }
  }
  _lm->m_bdecode_end();

  for (int i = 0; i < N; ++i) {
    total_tok += produced[(std::size_t)i];
    answers[(std::size_t)i] = std::move(out[(std::size_t)i]);
  }
  const double dec_s = std::chrono::duration<double>(clock::now() - t0).count();
  session()->info(fmt(
      "RealtimeVqaStage('{}'): m_decode_batched [pipelined] {} branches, {} "
      "steps, {} tok in {:.3f} s ({:.1f} tok/s)",
      this->id(), N, total_steps, total_tok, dec_s,
      dec_s > 0.0 ? static_cast<double>(total_tok) / dec_s : 0.0));
  return answers;
}

Job
RealtimeVqaStage::m_close_scene_(RuntimeContext& ctx)
{
  // Merge any buffered odd-tail frame (temporal-pair path) before we
  // count frames -- a scene that ended on an unpaired frame still emits
  // its last grid.
  if (_m_has_pending) { m_flush_pending_(); }
  const std::size_t n_frames = _m_imgs.size();
  if (n_frames == 0) { m_reset_scene_(); co_return; }
  // Final audio sweep: pull any classifications up to the last frame's
  // timestamp into this scene's sound timeline before rendering.
  co_await consume_audio_(ctx, _last_image_ts_us, m_scene_active_(),
                          _m_first_ts_us, _scene_audio);
  const std::uint64_t scene_idx   = _next_scene_idx++;
  const std::uint64_t first_ts_us = _m_first_ts_us;
  const std::uint64_t last_ts_us  = _m_last_ts_us;
  std::vector<std::string> answers(_questions.size());
  std::string scene_description;

  const auto* tpl = _chat_tpl.get();
  auto emit = [&]() -> Job {
    FlexData out = build_answer_(scene_idx, n_frames, first_ts_us, last_ts_us,
                                 _m_has_ts, scene_description, answers);
    // Persist the QA bundle to <camera>-video-qa (shared with the MLX
    // path). Skipped without a real timestamp -- no time-code key.
    // Don't persist a failed/empty scene: a prefill that returned -1
    // leaves no description and no answers, and writing it just litters
    // the db with blank rows (the empty <camera>-video-qa entries).
    bool has_content = !scene_description.empty();
    for (const auto& a : answers) {
      if (!a.empty()) { has_content = true; break; }
    }
    if (_m_has_ts && has_content) {
      // Fall back to "unknown" when the camera-name sideband never
      // arrived (mirrors the MLX path).
      const std::string cam = _scene_camera_name.empty()
          ? std::string("unknown") : _scene_camera_name;
      log_scene_qa_(cam, first_ts_us, last_ts_us,
                    scene_description, answers);
    }
    _m_prev_desc        = scene_description;
    _m_prev_last_ts_us  = _m_last_ts_us;   // before reset clears it
    m_reset_scene_();
    co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(out)));
    ++_scenes_closed;
  };

  if (!tpl || tpl->image_pad_token_id() < 0) {
    if (!_m_tpl_warned) {
      session()->warn(fmt(
          "RealtimeVqaStage('{}'): [metal] no chat template / image-pad; "
          "emitting empty answers", this->id()));
      _m_tpl_warned = true;
    }
    co_await emit();
    co_return;
  }
  // Always video: splice on the family video placeholder (== image_pad
  // for Qwen3-VL; distinct for Gemma-4).
  const std::int32_t video_pad = tpl->video_pad_token_id();

  std::vector<float> frame_ts_s;
  std::vector<int>   image_token_counts;
  std::vector<std::pair<int, int>> image_grids;
  frame_ts_s.reserve(n_frames);
  image_token_counts.reserve(n_frames);
  image_grids.reserve(n_frames);
  const std::uint64_t base_ts = _m_frame_ts_us.empty()
      ? 0 : _m_frame_ts_us.front();
  for (std::size_t i = 0; i < n_frames; ++i) {
    const std::uint64_t t = (i < _m_frame_ts_us.size())
        ? _m_frame_ts_us[i] : base_ts;
    frame_ts_s.push_back(
        static_cast<float>(static_cast<double>(t - base_ts) * 1e-6));
    image_token_counts.push_back(_m_imgs[i].n_tokens);
    // mh/mw are already POST-merger (normalised per encoder at encode).
    image_grids.emplace_back(_m_imgs[i].mh, _m_imgs[i].mw);
  }

  // PCM audio on iport2 (a pcm stage) + an LM audio encoder: encode the
  // scene's audio to soft tokens, then INTERPRET it to a short text phrase
  // (m_interpret_audio_ below) and add that to the sound TEXT timeline.
  // Splicing the raw audio tokens inline with the (large) video prefix did
  // NOT work: ~50 audio tokens are drowned by 768-4096 visual tokens, so
  // the audio question refused ("no sound / not mentioned in the scene
  // description") even though the tokens were present. Interpreting the
  // audio in ISOLATION (where it is salient) into text -- the Gemma
  // analogue of Qwen's CED tag timeline -- makes it readable by the
  // describe step AND every branched question. Two encoders feed it: the
  // 12B "unified" embedder (_mguni) or the e4b USM Conformer (_mgaud).
  std::vector<float> audio_rows;
  int n_audio_tokens = 0;
  if (!_scene_pcm.empty()) {
    if (_mguni && _mguni->has_audio()) {
      if (auto a =
              _mguni->encode_audio(_scene_pcm.data(), _scene_pcm.size())) {
        audio_rows = std::move(a->rows);
        n_audio_tokens = a->n_tokens;
      }
    } else if (_mgaud) {
      auto r = _mgaud->encode(_scene_pcm.data(), _scene_pcm.size(),
                              _scene_pcm_sr);
      audio_rows = std::move(r.embeddings);
      n_audio_tokens = r.n_tokens;
    }
  }
  if (n_audio_tokens > 0) {
    const double dur = _scene_pcm_sr > 0
        ? static_cast<double>(_scene_pcm.size()) / _scene_pcm_sr : 0.0;
    // Interpret the audio in isolation -> short phrase, then add it to the
    // scene's sound timeline (text). The audio question + describe read it
    // from the salient preamble instead of from drowned soft tokens.
    std::string phrase = m_interpret_audio_(audio_rows, n_audio_tokens);
    if (!phrase.empty()) {
      _scene_audio.emplace_back(_scene_pcm_first_ts_us, phrase);
    }
    _audio_tokens_spliced += static_cast<std::uint64_t>(n_audio_tokens);
    session()->info(fmt(
        "RealtimeVqaStage('{}'): scene {} audio {:.2f}s ({} samples @ "
        "{} Hz) -> {} soft tokens ({}) -> heard: \"{}\"", this->id(),
        scene_idx, dur, _scene_pcm.size(), _scene_pcm_sr, n_audio_tokens,
        _mguni ? "unified" : "e4b-usm",
        phrase.empty() ? "(none)" : phrase));
  }

  // User-turn preamble (date + prior-scene recap + sound text timeline),
  // shared with the MLX path via build_preamble_ so they can't drift. The
  // timeline now carries the interpreted audio phrase (above) as well as
  // any FlexData sound tags.
  // Recap only across temporally-continuous scenes (see prev_recap_).
  const std::string prev_recap = prev_recap_(
      _m_prev_desc, _m_prev_last_ts_us, _m_first_ts_us);
  const std::string pre = build_preamble_(
      _m_first_ts_us, prev_recap, _scene_audio, base_ts,
      /*audio_wired=*/ctx.num_iports() >= 3);
  const std::string describe_prompt =
      describe_prompt_for_(vqa_prompts_for_(effective_language_()), tpl);

  // Build the prompt token stream (frames + describe) -> TokenRefs. Audio
  // is text in the preamble, so there is no inline audio soft-token block.
  std::vector<std::int32_t> describe_ids;
  bool have_split = tpl->render_video_prefix(
      std::span<const float>(frame_ts_s),
      std::span<const int>(image_token_counts),
      /*is_first_turn=*/true, std::string_view(pre), &describe_ids);
  if (have_split) {
    if (!tpl->render_vlm_completion(describe_prompt, &describe_ids)) {
      describe_ids.clear();
      have_split = false;
    }
  }
  if (!have_split) {
    // Fallback (no prefix/completion split): one-shot video user turn.
    describe_ids.clear();
    tpl->render_user_turn_video(
        describe_prompt, std::span<const float>(frame_ts_s),
        std::span<const int>(image_token_counts),
        /*is_first_turn=*/true, std::string_view(pre), &describe_ids);
  }

  // ids -> refs: video_pad runs reference the per-frame host embeddings.
  // (Audio is text in the preamble now, not an inline soft-token block.)
  auto build_refs = [&](std::span<const std::int32_t> ids) {
    std::vector<genai::TokenRef> refs;
    refs.reserve(ids.size());
    std::size_t img_idx = 0;
    int img_off = 0;
    for (std::int32_t id : ids) {
      genai::TokenRef r;
      if (id == video_pad && img_idx < _m_imgs.size()) {
        r.kind = genai::TokenRef::Kind::ImageTokens;
        // gemma4_unified: host-f32 rows; else native-f16 SharedBuffer.
        if (!_m_imgs[img_idx].embeddings_host.empty()) {
          r.embeddings_host = &_m_imgs[img_idx].embeddings_host;
        } else {
          r.embeddings_buf = &_m_imgs[img_idx].embeddings;
        }
        r.image_token_offset = img_off++;
        if (img_off >= _m_imgs[img_idx].n_tokens) { ++img_idx; img_off = 0; }
      } else {
        r.kind = genai::TokenRef::Kind::Text;
        r.text_id = id;
      }
      refs.push_back(r);
    }
    return refs;
  };
  auto describe_refs = build_refs(std::span<const std::int32_t>(describe_ids));

  // Detach the reusable branch pool from the PREVIOUS scene's base before
  // building this scene's base: each pooled child still shares last
  // scene's base-context KV pages by refcount (its rebranch -- which would
  // release them -- doesn't run until after this prefill), so without this
  // the previous base's pages stay pinned and the (large, multi-frame)
  // describe-prefix prefill below can exhaust the KV page pool and return
  // -1. Detaching keeps each child's reserved SSM/conv buffers.
  for (auto& c : _branch_pool) {
    if (c.valid()) { _lm->m_detach_branch(c); }
  }

  auto base_ctx = _lm->make_context();
  if (!base_ctx.valid()) { co_await emit(); co_return; }
  const auto t_prefix_start = std::chrono::steady_clock::now();
  const std::int32_t prefix_pred = _lm->prefill_multimodal_metal(
      base_ctx, std::span<const genai::TokenRef>(describe_refs),
      std::span<const std::pair<int, int>>(image_grids));
  const double prefix_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_prefix_start).count();
  session()->info(fmt(
      "RealtimeVqaStage('{}'): scene {} describe-prefix {} tok in "
      "{:.3f} s ({:.1f} tok/s), {} frame{}, prev_summary={}",
      this->id(), scene_idx,
      static_cast<int>(describe_refs.size()), prefix_s,
      prefix_s > 0.0
          ? static_cast<double>(describe_refs.size()) / prefix_s
          : 0.0,
      n_frames, n_frames == 1 ? "" : "s",
      prev_recap.empty() ? "no" : "yes"));
  if (prefix_pred < 0) {
    session()->warn(fmt(
        "RealtimeVqaStage('{}'): scene {} describe-prefix prefill "
        "returned -1 ({} prefix tok exceeded the LM K/V budget); reduce "
        "the scene size (max_frames_per_scene / catch_up_drop) or lower "
        "vlm_input_width/height to shrink the per-frame token count",
        this->id(), scene_idx,
        static_cast<int>(describe_refs.size())));
    co_await emit();
    co_return;
  }
  scene_description = m_decode_(base_ctx, _sampler_params);
  const std::int32_t assistant_close = tpl->assistant_close_token_id();
  if (assistant_close >= 0) { (void)_lm->next_token(base_ctx, assistant_close); }

  // Per-question fanout: branch + prefill every question (each ends at its
  // own seq_len), then decode. When the metal backend supports batched
  // decode and there are >= 2 questions, run them BATCHED -- the weight-bound
  // projection / MLP / lm_head matmuls are read once per step for the whole
  // active set (decode is DRAM-bound on weights) while RoPE + attention run
  // per branch. Otherwise fall back to serial per-question decode.
  //
  // Branch contexts come from a reusable pool: reserve once (sized to the
  // question count), then rebranch the SAME contexts off this scene's base
  // every scene -- no per-scene allocation/free of the branch KV pages or
  // SSM/conv buffers. Questions beyond the reservation (or when the pool is
  // unsupported) fall back to a fresh per-scene branch().
  constexpr std::size_t kNoPool = static_cast<std::size_t>(-1);
  const std::size_t NQ = _questions.size();
  if (_branch_pool.size() < NQ && _lm->m_reserve_branches_supported()) {
    auto more = _lm->m_reserve_branches(
        static_cast<int>(NQ - _branch_pool.size()), _max_new_tokens + 256);
    for (auto& c : more) { _branch_pool.push_back(std::move(c)); }
  }

  std::vector<genai::LoadedLanguageModel::Context> children;
  std::vector<std::size_t>                       qidx;      // child -> question
  std::vector<std::size_t>                       pool_idx;  // child -> pool slot
  children.reserve(NQ);
  qidx.reserve(NQ);
  pool_idx.reserve(NQ);
  for (std::size_t i = 0; i < NQ; ++i) {
    genai::LoadedLanguageModel::Context child;
    std::size_t pslot = kNoPool;
    if (i < _branch_pool.size() && _lm->m_rebranch(_branch_pool[i], base_ctx)) {
      child = std::move(_branch_pool[i]);    // borrow the pooled context
      pslot = i;
    } else {
      child = _lm->branch(base_ctx);         // overflow / pool unsupported
    }
    if (!child.valid()) {
      if (pslot != kNoPool) { _branch_pool[pslot] = std::move(child); }
      continue;
    }
    std::vector<std::int32_t> q_ids;
    tpl->render_user_turn(question_prompt_(i), /*is_first_turn=*/false, &q_ids);
    if (_lm->prefill(child, std::span<const std::int32_t>(q_ids)) < 0) {
      if (pslot != kNoPool) { _branch_pool[pslot] = std::move(child); }
      continue;
    }
    children.push_back(std::move(child));
    qidx.push_back(i);
    pool_idx.push_back(pslot);
  }

  if (_batched_decode && children.size() >= 2
      && _lm->m_batched_decode_supported()) {
    auto a = m_decode_batched_(children);
    for (std::size_t k = 0; k < children.size() && k < a.size(); ++k) {
      answers[qidx[k]] = std::move(a[k]);
    }
  } else {
    for (std::size_t k = 0; k < children.size(); ++k) {
      genai::SamplerParams p = _sampler_params;
      if (p.seed != 0) { p.seed += static_cast<std::uint64_t>(qidx[k]); }
      answers[qidx[k]] = m_decode_(children[k], p);
    }
  }

  // Return the pooled contexts so their KV/state storage is retained for the
  // next scene; fresh (overflow) children are released at scope end.
  for (std::size_t k = 0; k < children.size(); ++k) {
    if (pool_idx[k] != kNoPool) {
      _branch_pool[pool_idx[k]] = std::move(children[k]);
    }
  }

  // Debug summary -- same shape as the MLX path so logs read identically
  // across backends.
  session()->info(fmt(
      "RealtimeVqaStage('{}'): scene {} closed: {} frame{}, "
      "ts=[{} ({}), {} ({})] us",
      this->id(), scene_idx, n_frames, n_frames == 1 ? "" : "s",
      first_ts_us, ts_human_(first_ts_us),
      last_ts_us,  ts_human_(last_ts_us)));
  session()->info(fmt(
      "RealtimeVqaStage('{}'): scene {} description = {}",
      this->id(), scene_idx, scene_description));
  for (std::size_t i = 0; i < _questions.size(); ++i) {
    session()->info(fmt(
        "RealtimeVqaStage('{}'): scene {} Q[{}] = {}",
        this->id(), scene_idx, i, _questions[i]));
    session()->info(fmt(
        "RealtimeVqaStage('{}'): scene {} A[{}] = {}",
        this->id(), scene_idx, i, answers[i]));
  }
  co_await emit();
}

Job
RealtimeVqaStage::m_process_(RuntimeContext& ctx)
{
  // Wake on a video frame (iport0) OR a chrono tick (iport1), whichever
  // lands first -- so each frame is encoded into tokens the moment it
  // arrives instead of once per tick. Encoding only pauses while
  // m_close_scene_'s prefill/decode runs (synchronous in this
  // coroutine); frames that pile up during it drain on the next wake.
  //
  // Wait only on ports with a live producer: a closed port is reported
  // perpetually "readable" by read_any (so its drained EOS doesn't
  // strand the wait), which would spin if we kept waiting on it. eos()
  // is true only once a port is BOTH closed and drained, so dropping it
  // never abandons buffered beats. When both are at EOS, end (flushing
  // any in-progress scene) instead of looping forever.
  std::vector<unsigned> wait_ports;
  if (!ctx.eos(0)) { wait_ports.push_back(0); }
  if (!ctx.eos(1)) { wait_ports.push_back(1); }
  if (wait_ports.empty()) {
    if (m_scene_active_()) { co_await m_close_scene_(ctx); }
    ctx.signal_done();
    co_return;
  }
  co_await ctx.read_any(std::move(wait_ports));

  // 1) Encode every frame currently available (ASAP).
  const std::uint32_t bl = ctx.backlog(0);
  for (std::uint32_t i = 0; i < bl; ++i) {
    auto p = co_await ctx.read(0);
    if (!p) {
      if (m_scene_active_()) { co_await m_close_scene_(ctx); }
      ctx.signal_done();
      co_return;
    }
    const auto* tbp = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (!tbp || tbp->shape.size() != 3 || tbp->shape[0] != 3
        || tbp->dtype != TensorBeat::DType::U8) {
      continue;
    }
    const int H = static_cast<int>(tbp->shape[1]);
    const int W = static_cast<int>(tbp->shape[2]);
    std::uint64_t ts_us = 0;
    bool ts_present = false;
    if (tbp->sideband.is_object()) {
      auto sb = tbp->sideband.as_object();
      if (sb.contains("timestamp_us")) {
        ts_us = sb.at("timestamp_us").as_uint(0);
        ts_present = true;
      }
    }
    // Monotonic latest real image timestamp -- the boundary for draining
    // iport2 audio ("prior to its image data"). Tracked across scenes so
    // the audio cursor advances with the video even between scenes.
    if (ts_present && ts_us > _last_image_ts_us) {
      _last_image_ts_us = ts_us;
    }
    // Latch camera_name (shared with the MLX path) + sync the questions
    // epoch on the scene's first frame -- so the metal path persists to
    // the same <camera>-video-questions / <camera>-video-qa sub-dbs.
    std::string camera_name = "unknown";
    if (tbp->sideband.is_object()) {
      auto sb = tbp->sideband.as_object();
      if (sb.contains("camera_name")) {
        camera_name = std::string(sb.at("camera_name").as_string("unknown"));
      }
    }
    // Sticky: latch the camera name from ANY frame that carries a real
    // one -- the sideband is intermittent across frames, so a scene
    // whose first frame happens to miss it would otherwise log to
    // "unknown-video-qa" ("on and off"). Kept across scenes (the camera
    // doesn't change); not cleared by m_reset_scene_.
    if (camera_name != "unknown" && !camera_name.empty()) {
      _scene_camera_name = camera_name;
    }
    if (!m_scene_active_()) {
      sync_questions_record_(
          _scene_camera_name.empty() ? "unknown" : _scene_camera_name,
          ts_us);
    }
    if (ts_present && _m_has_ts && m_scene_active_()
        && ts_us > _m_last_ts_us) {
      const std::uint64_t gap_ms = (ts_us - _m_last_ts_us) / 1000;
      if (gap_ms >= static_cast<std::uint64_t>(_max_frame_gap_ms)) {
        co_await m_close_scene_(ctx);
      }
    }
    // Zero-copy input: a contiguous beat (incl. a Shared/UMA frame from
    // video-to-rgb's metal-compute buffer) is read straight through its
    // storage pointer -- no materialize host copy. Strided beats fall
    // back to a one-shot contiguous materialize.
    AlignedVector<std::uint8_t> scratch;
    const std::uint8_t* rgb;
    if (tbp->is_contiguous()) {
      rgb = tbp->as_u8();
    } else {
      scratch = tbp->materialize_contiguous();
      rgb = scratch.data();
    }
    // CoreML path: when the frame is a Shared/UMA metal-compute buffer,
    // bridge it (zero-copy) and bind it straight into the letterbox
    // kernel -- no host staging into the encoder. The metal MLX-free
    // tower reads `rgb` host-side (UMA), so it needs no buffer.
    metal_compute::SharedBuffer fsb;
    const metal_compute::SharedBuffer* src_buf = nullptr;
    if (_m_coreml && tbp->is_contiguous()
        && tbp->storage_class() == TensorStorageClass::Shared) {
      auto* mc = session()->metal_compute();
      if (mc) {
        fsb = metal_compute::from_tensor_beat(*mc, *tbp);
        if (!fsb.empty()) { src_buf = &fsb; }
      }
    }
    (void)m_encode_frame_(rgb, H, W, ts_us, ts_present, src_buf);
    _had_frame_this_tick = true;
    if (static_cast<int>(_m_imgs.size()) >= _max_frames_per_scene) {
      co_await m_close_scene_(ctx);
    }
  }
  // 2) Drain audio-tagging classifications up to the latest received
  //    frame. Non-blocking + timestamp-gated; releases the consumed
  //    beats so iport2 (and the upstream audio path back to rtsp-
  //    capture) doesn't back up. No-op when iport2 is unwired.
  co_await consume_audio_(ctx, _last_image_ts_us, m_scene_active_(),
                          _m_first_ts_us, _scene_audio);

  // 3) Process every chrono tick currently available: idle-heartbeat
  //    accounting. A tick that finds no frame arrived since the
  //    previous tick (_had_frame_this_tick) bumps the idle counter;
  //    crossing idle_ticks_to_end closes the in-progress scene. The
  //    flag is cleared per tick so the count tracks "ticks since the
  //    last frame" exactly as the old one-tick-per-process loop did.
  const std::uint32_t nt = ctx.backlog(1);
  for (std::uint32_t i = 0; i < nt; ++i) {
    auto tick = co_await ctx.read(1);
    if (!tick) {
      if (m_scene_active_()) { co_await m_close_scene_(ctx); }
      ctx.signal_done();
      co_return;
    }
    if (_had_frame_this_tick) {
      _consec_idle_ticks = 0;
    } else {
      ++_consec_idle_ticks;
      // Only close on idle once the scene actually has video content: at
      // least one encoded token grid (=> at least one video token). With
      // temporal-pair fusion the first frame is held pending and _m_imgs
      // gets its first entry only when the second frame arrives, so this
      // also means "at least 2 frames received". Never close a scene
      // that has no video tokens just because idle_ticks_to_end elapsed.
      if (_consec_idle_ticks >= _idle_ticks_to_end && !_m_imgs.empty()) {
        co_await m_close_scene_(ctx);
        _consec_idle_ticks = 0;
      }
    }
    _had_frame_this_tick = false;
  }
  // Termination is handled at the top of the next call (both ports at
  // EOS -> flush + signal_done), so a tick/frame source closing mid-
  // drain still lets the remaining buffered beats drain first.
}

#endif  // metal-only

Job
RealtimeVqaStage::process(RuntimeContext& ctx)
{
  // Driver: wake on a video frame (iport0) OR a chrono tick (iport1),
  // whichever lands first, so each frame is encoded into tokens the
  // moment it arrives rather than once per tick. We still detect scene
  // boundaries inline (time-gap on frame arrival, idle-tick count when
  // no frame arrived since the last tick). Encoding only pauses while a
  // close_scene_ prefill/decode runs (synchronous in this coroutine);
  // frames that pile up during it drain on the next wake.
  if (ctx.num_iports() < 2) {
    session()->error(fmt(
        "RealtimeVqaStage('{}'): iport1 (chrono trigger) must be "
        "wired; got {} iports", this->id(), ctx.num_iports()));
    ctx.signal_done();
    co_return;
  }

#if defined(VPIPE_BUILD_APPLE_SILICON)
  // Metal/no-MLX path: single-frame encode + mROPE multimodal prefill +
  // simple per-question decode. Fully replaces the MLX driver below.
  co_await m_process_(ctx);
  co_return;
#endif

  // Wait only on ports with a live producer: a closed port is reported
  // perpetually "readable" by read_any, which would spin if we kept
  // waiting on it. eos() is true only once a port is BOTH closed and
  // drained, so dropping it never abandons buffered beats. Both at EOS
  // => end (flush any in-progress scene) rather than loop forever.
  std::vector<unsigned> wait_ports;
  if (!ctx.eos(0)) { wait_ports.push_back(0); }
  if (!ctx.eos(1)) { wait_ports.push_back(1); }
  if (wait_ports.empty()) {
    ctx.signal_done();
    co_return;
  }
  co_await ctx.read_any(std::move(wait_ports));


  // Catch-up: when the consumer (this stage) falls behind the
  // producer by more than `_catch_up_drop` frames, skip the oldest
  // `_catch_up_drop` frames via release_read so we land closer to
  // the producer's current write pointer. Skipped frames do NOT go
  // through encode_frame_ -- they are dropped silently. The check
  // is gated on _catch_up_drop > 0 (default 0 = disabled, the
  // user-requested no-catch-up behaviour).
  //
  // BETWEEN-SCENES ONLY: catch-up never fires mid-scene -- a scene
  // that has already accumulated frames must not get a gap in its
  // timeline from silent drops. We only let catch-up run when the
  // in-progress scene buffer is empty (i.e. we've either just closed
  // the previous scene or haven't started a new one yet). After the
  // catch-up skip, the frames that DO get drained in this tick are
  // the start of a fresh scene.

  // Encode every frame currently available (ASAP). read() short-
  // circuits when backlog > 0 (await_ready returns true), so no suspend
  // on the drain loop. _had_frame_this_tick is NOT reset here -- it
  // persists across wakes and is cleared per chrono tick below, so the
  // idle count tracks "ticks since the last frame".
  const std::uint32_t bl = ctx.backlog(0);
  for (std::uint32_t i = 0; i < bl; ++i) {
    auto p = co_await ctx.read(0);
    if (!p) {
      ctx.signal_done();
      co_return;
    }
    const auto* tbp = dynamic_cast<const TensorBeatPayload*>(p.get());
    if (!tbp
        || tbp->shape.size() != 3
        || tbp->shape[0] != 3
        || tbp->dtype != TensorBeat::DType::U8) {
      session()->warn(fmt(
          "RealtimeVqaStage('{}'): iport0 expected u8 [3,H,W] "
          "TensorBeat; dropping beat", this->id()));
      continue;
    }
    const int H = static_cast<int>(tbp->shape[1]);
    const int W = static_cast<int>(tbp->shape[2]);

    std::uint64_t ts_us = 0;
    bool ts_present = false;
    std::string camera_name = "unknown";
    if (tbp->sideband.is_object()) {
      auto sb = tbp->sideband.as_object();
      if (sb.contains("timestamp_us")) {
        ts_us = sb.at("timestamp_us").as_uint(0);
        ts_present = true;
      }
      if (sb.contains("camera_name")) {
        // sb.at(...) returns a FlexData TEMPORARY; .as_string()'s view
        // points into its storage. Construct the std::string in the
        // same full-expression so the bytes are copied out before the
        // temporary dies -- otherwise camera_name ends up with junk
        // and the LMDB sub-db name is garbled.
        std::string cn(sb.at("camera_name").as_string(""));
        if (!cn.empty()) {
          camera_name = std::move(cn);
        }
      }
    }

    (void)H; (void)W; (void)ts_us; (void)ts_present;
    (void)camera_name;
  }


  // Process every chrono tick currently available: idle-tick
  // accounting. A tick that finds no frame arrived since the previous
  // tick (_had_frame_this_tick) bumps the counter; the threshold closes
  // the in-progress scene. The flag is cleared per tick so the count
  // tracks "ticks since the last frame" exactly as the old one-tick-
  // per-process loop did.
  const std::uint32_t nt = ctx.backlog(1);
  for (std::uint32_t i = 0; i < nt; ++i) {
    auto tick = co_await ctx.read(1);
    if (!tick) {
      // Trigger source closed. Flush any in-progress scene, then done.
      ctx.signal_done();
      co_return;
    }
    if (_had_frame_this_tick) {
      _consec_idle_ticks = 0;
    } else {
      ++_consec_idle_ticks;
    }
    _had_frame_this_tick = false;
  }
  // Termination is handled at the top of the next call (both ports at
  // EOS -> flush + signal_done), so a source closing mid-drain still
  // lets the remaining buffered beats drain first.
}

Job
RealtimeVqaStage::drain(RuntimeContext& ctx)
{
#if defined(VPIPE_BUILD_APPLE_SILICON)
  if (m_scene_active_()) {
    co_await m_close_scene_(ctx);
  }
#else
  (void)ctx;
#endif
  co_return;
}

VPIPE_REGISTER_STAGE(RealtimeVqaStage)
VPIPE_REGISTER_SPEC(RealtimeVqaStage, kSpec)

}
