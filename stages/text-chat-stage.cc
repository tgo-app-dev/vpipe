#include "stages/text-chat-stage.h"

#include "common/beat-payload-intf.h"
#include "common/temp-root.h"
#include "common/flex-data.h"
#include "common/media-decode.h"
#include "common/media-line.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/chat-template.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/gemma4/gemma4-unified-embedder.h"
#include "generative-models/gemma4/metal-gemma4-audio.h"
#include "generative-models/gemma4/metal-gemma4-vision.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/qwen3/metal-audio-encoder.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/shared/mcp/python-sandbox.h"
#include "generative-models/shared/mcp/shell-tool.h"
#include "generative-models/token-muxer.h"
#include "generative-models/tokenizer.h"
#endif
#ifdef VPIPE_BUILD_APPLE_SILICON
#include <cstdlib>   // setenv (metal backend selection)
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <span>
#include <vector>

#include <cstdlib>   // mkdtemp
#include <unistd.h>

using namespace std;

namespace vpipe {

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

// One encoded media attachment of a media-line turn, in chunk order.
// Vision towers hand back native-f16 GPU rows (`embeddings`); the
// gemma4_unified embedder and every audio tower hand back host-f32
// rows (`embeddings_host`). TokenRef borrows whichever is set.
struct MediaItem {
  genai::MediaChunk::Kind     kind = genai::MediaChunk::Kind::Image;
  metal_compute::SharedBuffer embeddings;        // native f16
  std::vector<float>          embeddings_host;   // host f32
  int n_tokens = 0;
  int mh       = 0;   // post-merger grid (images; mROPE)
  int mw       = 0;
};

// Encode one decoded RGB image through whichever vision tower the LM
// carries. Tower priority mirrors visual-qa (gemma-4 e4b tower, then
// the encoder-less 12B unified embedder, then the Qwen metal tower).
optional<MediaItem>
encode_image_item_(genai::MetalQwenVisionEncoder*   mvis,
                   genai::MetalGemma4VisionEncoder* mgvis,
                   genai::Gemma4UnifiedEmbedder*    mguni,
                   const uint8_t*                   rgb,
                   int                              H,
                   int                              W)
{
  MediaItem m;
  m.kind = genai::MediaChunk::Kind::Image;
  if (mgvis) {
    auto r       = mgvis->encode(rgb, H, W);
    m.embeddings = std::move(r.embeddings);
    m.n_tokens   = r.n_tokens;
    m.mh         = r.grid_h;
    m.mw         = r.grid_w;
  } else if (mguni && mguni->has_vision()) {
    auto r = mguni->encode_image(rgb, H, W);
    if (!r) {
      return nullopt;
    }
    m.n_tokens        = r->n_tokens;
    m.mh              = r->grid_h;
    m.mw              = r->grid_w;
    m.embeddings_host = std::move(r->rows);
  } else if (mvis) {
    // The Qwen metal tower reports the PRE-merger patch grid.
    const int S  = std::max(1, mvis->config().spatial_merge);
    auto      r  = mvis->encode(rgb, H, W);
    m.embeddings = std::move(r.embeddings);
    m.n_tokens   = r.n_tokens;
    m.mh         = r.grid_h / S;
    m.mw         = r.grid_w / S;
  } else {
    return nullopt;
  }
  if (m.n_tokens <= 0
      || (m.embeddings.empty() && m.embeddings_host.empty())) {
    return nullopt;
  }
  return m;
}

// Encode one decoded mono-f32 PCM clip through whichever audio tower
// the LM carries. All towers expect 16 kHz.
optional<MediaItem>
encode_audio_item_(genai::MetalAudioEncoder*       ma,
                   genai::MetalGemma4AudioEncoder* mga,
                   genai::Gemma4UnifiedEmbedder*   mguni,
                   const float*                    pcm,
                   size_t                          n_samples,
                   int                             sample_rate)
{
  MediaItem m;
  m.kind = genai::MediaChunk::Kind::Audio;
  if (mga) {
    auto r            = mga->encode(pcm, n_samples, sample_rate);
    m.embeddings_host = std::move(r.embeddings);
    m.n_tokens        = r.n_tokens;
  } else if (mguni && mguni->has_audio()) {
    auto r = mguni->encode_audio(pcm, n_samples);
    if (!r) {
      return nullopt;
    }
    m.embeddings_host = std::move(r->rows);
    m.n_tokens        = r->n_tokens;
  } else if (ma) {
    auto r            = ma->encode(pcm, n_samples, sample_rate);
    m.embeddings_host = std::move(r.embeddings);
    m.n_tokens        = r.n_tokens;
  } else {
    return nullopt;
  }
  if (m.n_tokens <= 0 || m.embeddings_host.empty()) {
    return nullopt;
  }
  return m;
}

}  // namespace
#endif

TextChatStage::TextChatStage(const SessionContextIntf* s,
                             string                    id,
                             vector<InEdge>            iports,
                             FlexData                  config)
  : TypedStage<TextChatStage>(s, std::move(id), std::move(iports),
                              std::move(config))
{
  // Construction must succeed for any config (see Stage::fail_config):
  // a stage must construct so a graph can be built/edited before
  // required fields are supplied. Config problems are recorded via
  // fail_config (first message wins) and deferred to launch.
  // Scalar attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _hf_dir        = attr_str("hf_dir");
  _models_db     = attr_str("models_db");
  if (_models_db.empty()) {
    _models_db = "models";
  }
  _compute_dtype = attr_str("compute_dtype");
  _page_tokens   = static_cast<int>(attr_int("page_tokens"));
  {
    int64_t v = attr_int("max_pages");
    if (v < 1) {
      fail_config(fmt(
          "TextChatStage('{}'): max_pages must be >= 1 (got {})",
          this->id(), v));
    }
    _max_pages = static_cast<uint32_t>(v);
  }
  {
    int64_t v = attr_int("max_new_tokens");
    if (v < 1) {
      fail_config(fmt(
          "TextChatStage('{}'): max_new_tokens must be >= 1 "
          "(got {})", this->id(), v));
    }
    _max_new_tokens = static_cast<int>(v);
  }
  _enable_tools = attr_bool("enable_tools");
  _enable_python_tool = attr_bool("enable_python_tool");
  _enable_file_tools = attr_bool("enable_file_tools");
  _enable_shell_tool = attr_bool("enable_shell_tool");
  _file_sandbox_dir = attr_str("file_sandbox_dir");
  _enable_web_tools = attr_bool("enable_web_tools");
  _web_allow_private = attr_bool("web_allow_private");

  // disable_thinking is tri-state (unset = family default), with no flat
  // ConfigKey form, so it's read from the config object directly.
  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("disable_thinking")) {
      _disable_thinking = root.at("disable_thinking").as_bool(false);
    }
  }
#ifdef VPIPE_BUILD_APPLE_SILICON
  // Flat `sampler_*` knobs (mirrors realtime-vqa). Each defaults to the
  // SamplerParams default in kSpec.attrs, so leaving them all unset gives
  // greedy/argmax decoding (text-chat's historical default). attr_*
  // resolves config-else-default.
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
  // MTP speculative-decode head (metal path only): engaged when the loaded
  // model carries one (Qwen3.5-OptiQ / GGUF NextN). Token-exact vs the
  // standard decode path, so this is a perf-only switch; default on, set
  // false to force the pdecode loop.
  _mtp_enabled = attr_bool("mtp");
  _i8_prefill = attr_bool("i8_prefill");
  // Prefix-seed the MTP drafter's KV (decode- vs prefill-throughput tradeoff).
  // Default on for chat; applied to the LM at launch (only when MTP is used).
  _mtp_prefix_seed = attr_bool("mtp_prefix_seed");
#endif

  if (_hf_dir.empty()) {
    fail_config(fmt(
        "TextChatStage('{}'): config.hf_dir is required (non-empty "
        "string)", this->id()));
  }

  // Always allocate the per-turn FlexData oport. If the user wires
  // nothing downstream, the runtime allocates an OportBuffer with no
  // cursors and writes are silently dropped, so existing pipelines
  // that treated this as a sink continue to work unchanged.
  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "model: a models-DB key (registered by model-fetch) or an "
          "HF-style model dir; a DB key wins over a same-named path",
   .suggest_db = "models",
   // text-chat is a text->text LM: the model must accept text and emit
   // text (so the browser hides ASR/image/audio-out models).
   .need_inputs = "text", .need_outputs = "text"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db model-fetch registers into", .def_str = "models"},
  {.key = "compute_dtype", .type = ConfigType::String,
   .doc = "bf16 | f16 | f32", .def_str = "bf16"},
  {.key = "page_tokens", .type = ConfigType::Int,
   .doc = "ContextManager K/V page size", .def_int = 512},
  {.key = "max_pages", .type = ConfigType::Int,
   .doc = "per-LM page pool capacity (>= 1)", .def_int = 64},
  {.key = "max_new_tokens", .type = ConfigType::Int,
   .doc = "per-turn generation budget (>= 1)", .def_int = 1024},
  {.key = "disable_thinking", .type = ConfigType::Bool,
   .doc = "override chat-template thinking default", .def_bool = false},
  {.key = "enable_tools", .type = ConfigType::Bool,
   .doc = "enable MCP tool calling: advertise the built-in tools "
          "(get_current_time) and run any <tool_call> the model emits, "
          "feeding results back for another decode round (ChatML/Qwen "
          "families only)", .def_bool = false},
  {.key = "enable_python_tool", .type = ConfigType::Bool,
   .doc = "add a sandboxed `run_python` tool (seatbelt: no network, "
          "ephemeral scratch fs, home reads blocked, cpu/time limits). "
          "Independent opt-in; either tool flag activates the tool loop. "
          "Executes model-written code -- enable deliberately",
   .def_bool = false},
  {.key = "enable_file_tools", .type = ConfigType::Bool,
   .doc = "add confined read_file / write_file / list_files tools rooted "
          "at file_sandbox_dir (paths cannot escape the root). Any tool "
          "flag activates the tool loop", .def_bool = false},
  {.key = "file_sandbox_dir", .type = ConfigType::String,
   .doc = "workspace root for the file tools; empty => an ephemeral "
          "per-stage temp dir created at launch and removed at teardown",
   .def_str = ""},
  {.key = "enable_shell_tool", .type = ConfigType::Bool,
   .doc = "add a sandboxed `run_shell` tool (seatbelt: no network, writes "
          "confined to the file_sandbox_dir workspace, home reads blocked, "
          "cpu/time limits). Shares the workspace with the file tools. "
          "Runs model-written commands -- enable deliberately",
   .def_bool = false},
  {.key = "enable_web_tools", .type = ConfigType::Bool,
   .doc = "add fetch_url / scrape_page tools (http/https GET; SSRF-"
          "guarded: private/localhost targets refused). Any tool flag "
          "activates the tool loop", .def_bool = false},
  {.key = "web_allow_private", .type = ConfigType::Bool,
   .doc = "allow the web tools to reach private/localhost addresses "
          "(disables the SSRF guard) -- for trusted local use only",
   .def_bool = false},
  // Flat sampler knobs; all at SamplerParams defaults -> greedy/argmax.
  {.key = "sampler_temperature", .type = ConfigType::Real,
   .doc = "softmax temperature; <= 0 forces argmax", .def_real = 1.0},
  {.key = "sampler_top_k", .type = ConfigType::Int,
   .doc = "keep top-k logits; 0 = disabled", .def_int = 0},
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
  {.key = "mtp", .type = ConfigType::Bool,
   .doc = "use the MTP speculative-decode head when the model carries one "
          "(token-exact; perf only); false forces the standard decode path",
   .def_bool = true},
  {.key = "mtp_prefix_seed", .type = ConfigType::Bool,
   .doc = "seed the MTP drafter's KV with the prompt at decode start: higher "
          "draft acceptance / decode throughput for a small extra prefill "
          "cost (token-exact; perf only). Default on (chat is decode-bound); "
          "set false to favor prefill throughput. No effect unless mtp is on.",
   .def_bool = true},
  {.key = "i8_prefill", .type = ConfigType::Bool,
   .doc = "accelerated mode (LOSSY): dynamic-int8 prefill GEMMs, ~2x their "
          "f16 rate on matrix-core GPUs at int8 quality (prefill is NOT "
          "token-exact with this on; decode untouched; IGNORED without NAX matmul2d -- matrix-core GPU + kernels). Default false; env "
          "VPIPE_I8_GEMM overrides.",
   .def_bool = false},
};
const PortSpec kIports[] = {
  {.name = "user",
   .doc = "FlexData string: the user's turn text; may embed image/"
          "audio attachments as media-line markers (fs path or "
          "base64), spliced into the prefill via the model's own "
          "vision/audio towers",
   .type = &typeid(FlexDataPayload),
   .tags = "text", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "assistant",
   .doc = "FlexData {text,prefill_ms,decode_ms,ctx_pos} per turn "
          "(downstream optional)",
   .type = &typeid(FlexDataPayload),
   .tags = "text", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "text-chat",
  .doc       = "Conversational LM stage: appends each user turn to a "
               "persistent K/V chat context, streams the assistant reply "
               "to the UI, and emits a FlexData turn record. /clear "
               "resets. Media-line markers in a turn attach images/audio "
               "(decoded via FFmpeg, encoded by the model's own towers); "
               "unsupported modalities warn and drop.",
  .display_name = "Chat",
  .category  = StageCategory::Text,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
TextChatStage::spec() const noexcept
{
  return kSpec;
}

TextChatStage::~TextChatStage()
{
  // Remove the ephemeral file-tool workspace we created at launch (a
  // caller-supplied file_sandbox_dir is left untouched).
  if (_file_sandbox_ephemeral && !_file_sandbox_root.empty()) {
    std::error_code ec;
    std::filesystem::remove_all(_file_sandbox_root, ec);
  }
}

void
TextChatStage::setup_file_sandbox_()
{
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::path root;
  if (!_file_sandbox_dir.empty()) {
    // Caller-supplied workspace: create it if missing, keep it on
    // teardown.
    root = _file_sandbox_dir;
    fs::create_directories(root, ec);
    if (ec) {
      session()->warn(fmt(
          "TextChatStage('{}'): file_sandbox_dir '{}' could not be "
          "created ({}); file/shell tools disabled",
          this->id(), _file_sandbox_dir, ec.message()));
      _enable_file_tools = false;
      _enable_shell_tool = false;
      return;
    }
    _file_sandbox_ephemeral = false;
  } else {
    // Ephemeral per-stage workspace under the app temp root (CWD-local).
    const fs::path base = vpipe::temp_root();
    std::string tmpl = (base / "vpipe-chat-ws-XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (!::mkdtemp(buf.data())) {
      session()->warn(fmt(
          "TextChatStage('{}'): could not create an ephemeral file "
          "workspace; file/shell tools disabled", this->id()));
      _enable_file_tools = false;
      _enable_shell_tool = false;
      return;
    }
    root = buf.data();
    _file_sandbox_ephemeral = true;
  }
  _file_sandbox = std::make_shared<FileSandbox>(root);
  _file_sandbox_root = _file_sandbox->root().string();
}

Job
TextChatStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
#ifdef VPIPE_BUILD_APPLE_SILICON
  // No-MLX build: the LM subsystem runs on the metal-compute backend.
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  // All backend work (load, dequantize, forward pass, free) is driven
  // through the LM manager itself (MLX: Session::mlx_runtime() worker;
  // metal: inline on the metal-compute device), so we don't pin any GPU
  // stream here.
  auto* mgr = session() ? session()->generative_model_manager()
                        : nullptr;
  if (!mgr) {
    session()->error(fmt(
        "TextChatStage('{}'): no GenerativeModelManager (is this an "
        "apple-silicon build?)", this->id()));
    co_return;
  }
  genai::LoadSpec spec;
  spec.hf_dir        = resolve_model_dir(session(), _models_db, _hf_dir);
  spec.compute_dtype = _compute_dtype;
  spec.page_tokens   = _page_tokens;
  spec.max_pages     = _max_pages;
  session()->info(fmt(
      "TextChatStage('{}'): loading model from '{}' "
      "(dtype={}, page_tokens={}, max_pages={})",
      this->id(), _hf_dir, _compute_dtype,
      _page_tokens, _max_pages));
  _lm = mgr->load(spec);
  if (!_lm || !_lm->valid()) {
    session()->error(fmt(
        "TextChatStage('{}'): failed to load model from '{}'",
        this->id(), _hf_dir));
    _lm.reset();
    co_return;
  }
  // Apply the MTP prefix-seed preference before any prefill (no-op unless the
  // model carries a metal MTP head). Chat is decode-bound, so default on.
  if (_mtp_enabled) { _lm->set_mtp_prefix_seed(_mtp_prefix_seed); }
  // Accelerated mode (LOSSY, opt-in): dynamic-int8 prefill GEMMs. No-op on
  // backends without the route; env VPIPE_I8_GEMM overrides either way.
  if (_i8_prefill) { _lm->set_i8_prefill(true); }
  // Cache whichever media-encoder towers this checkpoint carries; the
  // media-line turn path (attachment markers in the user line) uses
  // them to splice image/audio embeddings into the prefill. All null
  // on a text-only checkpoint -- media turns then warn and drop.
  _mvis     = _lm->metal_vision_encoder();
  _mgvis    = _lm->metal_gemma4_vision_encoder();
  _mguni    = _lm->gemma4_unified_embedder();
  _m_audio  = _lm->metal_audio_encoder();
  _mg_audio = _lm->metal_gemma4_audio_encoder();
  // Acquire the persistent chat context here so every beat reuses the
  // same K/V cache; the conversation history becomes the model's
  // memory. Reset via `/clear` in process().
  _chat_ctx = _lm->make_context();
  if (!_chat_ctx.valid()) {
    session()->error(fmt(
        "TextChatStage('{}'): failed to acquire chat context",
        this->id()));
    _lm.reset();
    co_return;
  }
  // Build stage-local chat template so the disable_thinking override
  // is applied without mutating the LM's shared default template.
  // When the config flag is unset the factory uses the family
  // default (Qwen3-VL: thinking-ON, Qwen3 text-only: OFF, others: n/a).
  _chat_tpl = genai::make_chat_template(
      _lm->config().architecture, _lm->tokenizer(),
      _disable_thinking);
  if (!_chat_tpl) {
    session()->warn(fmt(
        "TextChatStage('{}'): no chat template registered for "
        "architecture '{}'; falling back to the LM's default (which "
        "is also null in this case)",
        this->id(), _lm->config().architecture));
  }
  // MCP tools: seed the registry when any tool is enabled and the model's
  // chat template renders the tool-calling scaffold (ChatML/Qwen
  // families). A template without tool support (Llama-3, Gemma-4) warns
  // and leaves tools off so the plain chat path is unchanged. The runtime
  // tool gate in process() is `!_tools.empty()`, so both flags feed one
  // registry.
  if (_enable_tools || _enable_python_tool || _enable_file_tools
      || _enable_shell_tool || _enable_web_tools) {
    if (_chat_tpl && _chat_tpl->supports_tools()) {
      if (_enable_tools) {
        _tools = make_builtin_tool_registry();   // get_current_time
      }
      if (_enable_python_tool) {
        _tools.add(make_python_tool());           // sandboxed run_python
      }
      // The file tools and the shell tool share one workspace, so set it
      // up once when either is on, then register each enabled surface.
      if (_enable_file_tools || _enable_shell_tool) {
        setup_file_sandbox_();
      }
      if (_enable_file_tools && _file_sandbox) {   // read/write/list_files
        add_file_tools(_tools, _file_sandbox);
      }
      if (_enable_shell_tool && _file_sandbox) {   // sandboxed run_shell
        ShellToolOptions so;
        so.workspace = _file_sandbox->root();
        if (session()) {
          for (const auto& w : session()->fs_whitelist()) {
            so.extra_writable.push_back(w);
          }
        }
        _tools.add(make_shell_tool(so));
      }
      if (_enable_web_tools) {                     // fetch_url / scrape_page
        UrlFetchOptions wo;
        wo.allow_private = _web_allow_private;
        _tools.add(make_fetch_url_tool(wo));
        _tools.add(make_scrape_page_tool(wo));
      }
      session()->info(fmt(
          "TextChatStage('{}'): MCP tools enabled ({} tool(s){}{}{}{})",
          this->id(), static_cast<int>(_tools.size()),
          _enable_python_tool ? ", incl. sandboxed python" : "",
          _enable_shell_tool ? ", incl. sandboxed shell" : "",
          (_enable_file_tools || _enable_shell_tool) && _file_sandbox
              ? fmt(", workspace {}", _file_sandbox_root)()
              : std::string(),
          _enable_web_tools ? ", incl. web fetch" : ""));
    } else {
      session()->warn(fmt(
          "TextChatStage('{}'): a tool flag is set but the model's chat "
          "template ('{}') has no tool-calling support; tools disabled",
          this->id(),
          _chat_tpl ? std::string(_chat_tpl->family_name())
                    : std::string("<none>")));
      _enable_tools = false;
      _enable_python_tool = false;
      _enable_file_tools = false;
      _enable_shell_tool = false;
      _enable_web_tools = false;
    }
  }
  _seeded = false;
  session()->info(fmt(
      "TextChatStage('{}'): model ready ({} layers, vocab={}{})",
      this->id(), _lm->config().n_layers,
      _lm->config().vocab_size,
      _disable_thinking.has_value()
          ? (*_disable_thinking ? ", thinking=off" : ", thinking=on")
          : ""));
#else
  session()->error(fmt(
      "TextChatStage('{}'): this build was compiled without "
      "VPIPE_BUILD_APPLE_SILICON; the LLM subsystem is unavailable",
      this->id()));
#endif
  co_return;
}

Job
TextChatStage::process(RuntimeContext& ctx)
{
  auto t = co_await ctx.read(0);
  if (!t) {
    ctx.signal_done();
    co_return;
  }

  // Every input beat must produce exactly one output beat, otherwise
  // a feedback-driven trigger downstream (feedback-rx + feedback-tx)
  // never sees a new beat and the loop deadlocks. All early-exit
  // paths below (no model, wrong payload type, empty text, /clear,
  // prefill failure) emit a minimal FlexData beat carrying the
  // current context position and an empty `text` field.
  auto build_turn_payload =
      [](const std::string& text,
         int64_t prefill_ms,
         int64_t decode_ms,
         int64_t ctx_pos) {
        FlexData fd = FlexData::make_object();
        auto root = fd.as_object();
        root.insert("text", FlexData::make_string(text));
        root.insert("prefill_ms", FlexData::make_int(prefill_ms));
        root.insert("decode_ms", FlexData::make_int(decode_ms));
        root.insert("ctx_pos", FlexData::make_int(ctx_pos));
        return make_payload<FlexDataPayload>(std::move(fd));
      };

  auto current_ctx_pos = [this]() -> int64_t {
#ifdef VPIPE_BUILD_APPLE_SILICON
    if (_lm && _chat_ctx.valid()) {
      return static_cast<int64_t>(_chat_ctx.seq_len());
    }
#endif
    return 0;
  };

#ifdef VPIPE_BUILD_APPLE_SILICON
  if (!_lm) {
    session()->warn(fmt(
        "TextChatStage('{}'): no model loaded (initialize failed?); "
        "emitting empty turn beat", this->id()));
    co_await ctx.write(0,
        build_turn_payload("", 0, 0, current_ctx_pos()));
    co_return;
  }
#endif
  // We accept any payload that carries a FlexData string. Everything
  // else gets a single warn and we emit an empty turn beat -- the
  // stage is forgiving so a misconnected pipeline still drains its
  // input AND keeps any feedback loop pumping.
  const auto* fdp = dynamic_cast<const FlexDataPayload*>(t.get());
  if (!fdp) {
    session()->warn(fmt(
        "TextChatStage('{}'): expected FlexDataPayload on in-port "
        "0, got {}", this->id(), t->describe()));
    co_await ctx.write(0,
        build_turn_payload("", 0, 0, current_ctx_pos()));
    co_return;
  }
  if (!fdp->data.is_string()) {
    session()->warn(fmt(
        "TextChatStage('{}'): FlexData payload is not a string "
        "(type-discriminator-mismatch)", this->id()));
    co_await ctx.write(0,
        build_turn_payload("", 0, 0, current_ctx_pos()));
    co_return;
  }
  const string user_text = string(fdp->data.as_string(""));
  if (user_text.empty()) {
    co_await ctx.write(0,
        build_turn_payload("", 0, 0, current_ctx_pos()));
    co_return;
  }

#ifdef VPIPE_BUILD_APPLE_SILICON
  // `/clear` resets the chat context's K/V cache. _seeded flips back
  // to false so families that need a session-start token (Llama-3
  // BOS) re-emit it on the next turn; ChatML treats _seeded as a
  // no-op anyway. We still emit a turn beat so any feedback loop
  // sees the cleared ctx_pos and the next iteration fires.
  if (user_text == "/clear") {
    _chat_ctx = _lm->make_context();
    _seeded = false;
    session()->info(fmt("[chat cleared]"));
    co_await ctx.write(0,
        build_turn_payload("", 0, 0, current_ctx_pos()));
    co_return;
  }

  // Build the chat turn appended to the existing context, delegating
  // the model-family-specific token layout (Llama-3 headers vs
  // ChatML, BOS-once vs no-BOS, optional reasoning prefix, etc.) to
  // the LM's ChatTemplate. The first turn passes is_first_turn=true
  // so families that need a session-start token (Llama-3 BOS) emit
  // it exactly once; subsequent turns skip it because the K/V cache
  // already begins with it. `/clear` flips _seeded back to false to
  // re-seed.
  const auto& tok = _lm->tokenizer();
  const genai::ChatTemplate* tpl = _chat_tpl.get();
  if (!tpl) {
    session()->warn(fmt(
        "TextChatStage('{}'): model architecture has no registered "
        "chat template; emitting empty turn beat", this->id()));
    co_await ctx.write(0,
        build_turn_payload("", 0, 0, current_ctx_pos()));
    co_return;
  }
  // ---- media-line markers: processed BEFORE the tokenizer ----------
  // A marker-bearing line decomposes into ordered chunks (text runs +
  // attachments). Each attachment is decoded with FFmpeg to the
  // encoder's native input (RGB u8 [3,H,W] / mono f32 @16 kHz), run
  // through the model's OWN vision/audio tower, and spliced into the
  // prefill at its marker position. A modality the loaded model does
  // not provide -- and any item that fails to decode/encode -- warns
  // and is dropped, keeping the surrounding text.
  vector<genai::MediaChunk> chunks;
  vector<MediaItem>         items;   // encoded media, in chunk order
  if (media_line::has_media_marker(user_text)) {
    vector<string> perrs;
    auto segs = media_line::parse(user_text, &perrs);
    for (const auto& e : perrs) {
      session()->warn(fmt("TextChatStage('{}'): {}", this->id(), e));
    }
    const bool image_ok = tpl->image_pad_token_id() >= 0
        && (_mvis || _mgvis || (_mguni && _mguni->has_vision()));
    const bool audio_ok = tpl->audio_pad_token_id() >= 0
        && (_m_audio || _mg_audio || (_mguni && _mguni->has_audio()));
    const FFmpegLibraries* libs = session()->ffmpeg_libraries();
    for (auto& seg : segs) {
      if (seg.kind == media_line::Segment::Kind::Text) {
        genai::MediaChunk c;
        c.kind = genai::MediaChunk::Kind::Text;
        c.text = std::move(seg.text);
        chunks.push_back(std::move(c));
        continue;
      }
      const bool  is_image =
          seg.modality == media_line::Modality::Image;
      const char* mod_name = is_image ? "image" : "audio";
      if (is_image ? !image_ok : !audio_ok) {
        session()->warn(fmt(
            "TextChatStage('{}'): the loaded model ('{}') does not "
            "support {} input; dropping the attachment",
            this->id(), _lm->config().architecture, mod_name));
        continue;
      }
      const bool from_path =
          seg.kind == media_line::Segment::Kind::FsPath;
      string             derr;
      optional<MediaItem> item;
      if (is_image) {
        auto img = from_path
            ? decode_image_file(libs, seg.text, &derr)
            : decode_image_bytes(
                  libs,
                  span<const uint8_t>(seg.bytes.data(),
                                      seg.bytes.size()),
                  &derr);
        if (img) {
          item = encode_image_item_(_mvis, _mgvis, _mguni,
                                    img->rgb.data(), img->height,
                                    img->width);
          if (!item) { derr = "vision encode failed"; }
        }
      } else {
        constexpr int kAudioRate = 16000;   // every tower's input rate
        auto au = from_path
            ? decode_audio_file(libs, seg.text, kAudioRate, &derr)
            : decode_audio_bytes(
                  libs,
                  span<const uint8_t>(seg.bytes.data(),
                                      seg.bytes.size()),
                  kAudioRate, &derr);
        if (au) {
          item = encode_audio_item_(_m_audio, _mg_audio, _mguni,
                                    au->pcm.data(), au->pcm.size(),
                                    kAudioRate);
          if (!item) { derr = "audio encode failed"; }
        }
      }
      if (!item) {
        session()->warn(fmt(
            "TextChatStage('{}'): dropping {} attachment{}{}: {}",
            this->id(), mod_name, from_path ? " " : "",
            from_path ? seg.text : string(), derr));
        continue;
      }
      genai::MediaChunk c;
      c.kind = is_image ? genai::MediaChunk::Kind::Image
                        : genai::MediaChunk::Kind::Audio;
      c.n_tokens = item->n_tokens;
      chunks.push_back(std::move(c));
      items.push_back(std::move(*item));
    }
  } else {
    genai::MediaChunk c;
    c.kind = genai::MediaChunk::Kind::Text;
    c.text = user_text;
    chunks.push_back(std::move(c));
  }

  // MCP tools are advertised in a leading system turn (below). On the
  // FIRST turn that system turn carries the session-start token (Gemma's
  // <bos>), so the user turn that follows must NOT emit it again -- pass
  // is_first_turn=false to the user render when a tools system turn will
  // precede it. A no-op for ChatML/Qwen (no session-start token).
  const bool tools_active = !_tools.empty() && tpl->supports_tools();
  const bool tools_seed   = tools_active && !_seeded;
  const bool user_is_first = !_seeded && !tools_seed;

  vector<int32_t> ids;
  bool have_media = !items.empty();
  if (have_media
      && !tpl->render_user_turn_media(
             span<const genai::MediaChunk>(chunks.data(),
                                           chunks.size()),
             /*is_first_turn=*/user_is_first, &ids)) {
    session()->warn(fmt(
        "TextChatStage('{}'): chat template '{}' cannot render the "
        "mixed media turn; dropping {} attachment(s)",
        this->id(), tpl->family_name(),
        static_cast<int>(items.size())));
    have_media = false;
    items.clear();
    ids.clear();
  }
  if (!have_media && ids.empty()) {
    // Text-only turn (no markers, or every attachment dropped).
    string text_only;
    for (const auto& c : chunks) {
      if (c.kind == genai::MediaChunk::Kind::Text) {
        text_only += c.text;
      }
    }
    if (text_only.empty()) {
      // The line was attachments only and none survived.
      co_await ctx.write(0,
          build_turn_payload("", 0, 0, current_ctx_pos()));
      co_return;
    }
    tpl->render_user_turn(text_only, /*is_first_turn=*/user_is_first, &ids);
  }

  // ---- MCP tools: advertise them in a leading system turn ----------
  // When tools are enabled (and the family renders the tool scaffold),
  // the very first prompt in a fresh context is prefixed with a system
  // turn declaring the callable tools. Subsequent turns skip it -- the
  // declaration already lives in the K/V cache. After decode we scan
  // the reply for tool-call blocks, run them, and feed the results
  // back for another round (see the tool loop below).
  if (tools_seed) {
    vector<int32_t> sys;
    if (tpl->render_tools_system_turn(_tools.tools_json(),
                                      /*is_first_turn=*/true, &sys)
        && !sys.empty()) {
      ids.insert(ids.begin(), sys.begin(), sys.end());
    }
  }

  using clock = std::chrono::steady_clock;

  // Termination reason for a decode round. Surfaced after the round to
  // explain non-natural stops (budget, decode error, page exhaustion,
  // pipeline stop) as a single warn line.
  enum class StopReason {
    StopToken,        // model emitted EOS/EOT -- the normal path
    MaxNewTokens,     // hit the per-turn token budget
    DecodeError,      // step_pipelined / materialize_lazy returned -1
    PipelineStopped,  // ctx.stop_requested() while decoding
  };
  // Accumulators across the (possibly several) decode rounds a single
  // user beat runs: one base round plus one per tool-call round.
  double     prefill_s_total    = 0.0;
  double     decode_s_total     = 0.0;
  int        prefill_n_total    = 0;
  int        decode_calls_total = 0;
  StopReason reason             = StopReason::StopToken;

  // Stop check + assistant-turn close token come from the same
  // ChatTemplate. The template encapsulates the family-specific
  // stop set (Llama-3: eot/eom/end_of_text; ChatML: im_end/endoftext)
  // so this stage stays family-agnostic.
  auto is_stop =
      [tpl](int32_t id) {
        return tpl->is_stop_token(id);
      };
  const int32_t assistant_close = tpl->assistant_close_token_id();

  // The token the last decode round stopped on (or the last emitted
  // token on a budget cutoff). Lets the tool loop tell a turn-CLOSING
  // stop (<end_of_turn>/eos) from a turn-CONTINUING one (Gemma's
  // <|tool_response>) so the assistant-close commit and the tool-results
  // injection stay in the right order.
  int32_t last_stop_token = -1;

  // Decode one assistant turn. `next` is the token prefill predicted for
  // this round (its logits are still in last_logits_host() when a
  // sampled path re-samples below); `prompt_ids` are the round's full
  // prompt ids (sampler priming + pdecode). Streams the reply to the UI,
  // commits the assistant-close token to the K/V cache, accumulates
  // decode timing/token counts, sets `reason`, and returns the decoded
  // text. Called once for the user turn, then once per tool-call round.
  auto decode_turn =
      [&](int32_t next, std::span<const int32_t> prompt_ids)
          -> std::string {
    const auto t_decode_start = clock::now();
    // Stream the assistant's tokens to the user as they are produced via
    // the UI delegate's live text stream (stdout by default; the web-ui
    // console under the web delegate). We also accumulate the full text
    // so the per-turn FlexData beat on out-port 0 carries it.
    auto out_stream = session()->open_text_stream();
    std::string assistant_text;
    auto emit_chunk =
        [&assistant_text, &out_stream](const std::string& chunk) {
          if (chunk.empty()) {
            return;
          }
          assistant_text += chunk;
          out_stream->write(chunk);
        };

    auto sd = tok.make_stream_decoder();
    // Thinking-ON templates (Qwen3/Qwen3-VL) open the reasoning block in
    // the PROMPT (`<think>\n` in the assistant extras), so its start
    // token never streams; emit the unified thinking-start marker
    // ourselves so front ends can fold the reasoning. The CLOSE token is
    // generated by the model and the detokenizer rewrites it to
    // media_line::kThinkEnd (as it does both of Gemma's channel tokens).
    if (tpl->assistant_prompt_opens_thinking()) {
      emit_chunk(string(media_line::kThinkStart));
    }
    // Per-turn sampler — fresh seen-token set each turn so the
    // repetition / presence penalty rolls over cleanly between
    // conversations. Prime with the prompt ids so the very first
    // sampled token already penalises tokens the user just typed.
    genai::Sampler sampler(_sampler_params);
    sampler.prime(prompt_ids);
    const bool sampled_path = !sampler.is_argmax();
    // For sampled decode, re-sample from the prefill's logits (the
    // forward pass that just ran computed them; last_logits() returns
    // them on the host). The prefill stashed argmax as the next id;
    // we replace that with our sampled choice. For argmax decode we
    // keep `next` as the prefill's argmax.
    if (sampled_path && !is_stop(next)) {
      const auto& pre = _lm->last_logits_host();
      if (!pre.empty()) {
        next = sampler.sample(
            std::span<const float>(pre.data(), pre.size()));
      }
    }
    if (next >= 0 && !is_stop(next)) {
      emit_chunk(tok.step(sd, next));
      sampler.prime(std::span<const int32_t>(&next, 1));
    }
    reason = StopReason::StopToken;
    if (_max_new_tokens <= 1 && !is_stop(next)) {
      // Degenerate budget: prefill produced one token and we're done.
      reason = StopReason::MaxNewTokens;
    }
    int decode_calls = 0;
    // Metal/no-MLX MTP fast path: the bundled MTP speculative head
    // (mtp.safetensors, present on Qwen3.5-OptiQ) drafts ahead so the verifier
    // accepts multiple tokens per forward -- token-exact vs the pdecode loop
    // (greedy) or decode_pipelined (sampling, the verify samples each position).
    // The prefill's first token was already streamed above (and primes the
    // pipeline); on_tokens drops MTP's echo of it (out_ids[0] == next) and emits
    // the rest, stopping as soon as a stop token is seen. Greedy OR penalty-free
    // sampling -- the verify applies no repetition/presence penalty, so a
    // penalised sampler stays on the loops below (which apply it).
    const bool mtp_no_penalty =
        (_sampler_params.repetition_penalty == 1.0f
         && _sampler_params.presence_penalty == 0.0f);
    if (next >= 0 && !is_stop(next) && _mtp_enabled && _lm->mtp_available()
        && (!sampled_path || mtp_no_penalty)) {
      bool first_echo = true;
      bool stopped = false;
      int  produced = 0;
      auto on_toks =
          [&](std::span<const int32_t> toks) -> bool {
            for (int32_t id : toks) {
              if (first_echo) { first_echo = false; continue; }
              emit_chunk(tok.step(sd, id));
              ++decode_calls;
            }
            return !ctx.stop_requested();
          };
      _lm->mtp_generate(_chat_ctx, next, _max_new_tokens, _sampler_params,
                        is_stop, on_toks, &produced, &stopped);
      if (stopped) {
        reason = StopReason::StopToken;
      } else if (ctx.stop_requested()) {
        reason = StopReason::PipelineStopped;
      } else {
        reason = StopReason::MaxNewTokens;
      }
    } else {
    // Metal/no-MLX: GPU-resident pipelined decode -- the token chain stays
    // on the GPU (in-stream embed + on-device argmax/sampling, no host logit
    // pull), and committing the NEXT token's forward BEFORE detokenize+emit
    // overlaps the GPU with the host's per-token work. Stop-token / abort /
    // budget semantics are preserved: a step (which appends its input
    // token's KV) is committed only after that token is confirmed not-stop.
    // Falls back to the synchronous next_token loop if the backend can't
    // pipeline (e.g. a Llama metal model without the primitive).
    const bool pipelined = _lm->pdecode_begin(
        _chat_ctx, next, prompt_ids,
        _sampler_params, _max_new_tokens);
    // Run-ahead: when the backend rolls back speculative KV (pdecode_end), keep
    // a SECOND forward in flight so the GPU never idles on the CPU command-
    // buffer encode (it runs the next forward while the host detokenizes AND
    // while the CPU encodes the one after). A stop token's speculative forward
    // is rolled back by pdecode_end. Paged backends without rollback stay
    // strictly one-in-flight (commit only after not-stop).
    const bool runahead = pipelined && _lm->pdecode_supports_runahead();
    if (pipelined) {
      // Kick off the forward for token 1 (input = the first token `next`,
      // already confirmed not-stop above) so the GPU is busy immediately.
      bool committed = (next >= 0 && !is_stop(next) && !ctx.stop_requested())
          ? _lm->pdecode_commit(_chat_ctx) : false;
      // Speculative second forward (input = token 1, not yet read) -> two in
      // flight. Discarded by pdecode_end's rollback if token 1 is a stop.
      if (runahead && committed && _max_new_tokens > 1
          && !ctx.stop_requested()) {
        _lm->pdecode_commit(_chat_ctx);
      }
      for (int i = 1; i < _max_new_tokens; ++i) {
        if (ctx.stop_requested()) { reason = StopReason::PipelineStopped; break; }
        if (is_stop(next)) { reason = StopReason::StopToken; break; }
        if (!committed) { reason = StopReason::DecodeError; break; }
        next = _lm->pdecode_next(_chat_ctx);
        if (next < 0) { reason = StopReason::DecodeError; break; }
        ++decode_calls;
        // Commit token i+1's forward (input = `next`) BEFORE emit so it runs
        // on the GPU while the host detokenizes + streams token i. Only when
        // `next` is not a stop token (else its KV must not be appended).
        const bool cont = (i + 1 < _max_new_tokens) && !is_stop(next)
            && !ctx.stop_requested();
        committed = cont ? _lm->pdecode_commit(_chat_ctx) : false;
        if (!is_stop(next)) {
          emit_chunk(tok.step(sd, next));
        } else {
          reason = StopReason::StopToken;
        }
        if (i + 1 == _max_new_tokens && !is_stop(next)) {
          reason = StopReason::MaxNewTokens;
        }
      }
      _lm->pdecode_end(_chat_ctx);
    } else
    // Fallback: synchronous next_token(forced) advances the K/V and returns
    // the position's argmax; sampled decode draws from the host logits.
    for (int i = 1; i < _max_new_tokens; ++i) {
      if (ctx.stop_requested()) {
        reason = StopReason::PipelineStopped;
        break;
      }
      if (is_stop(next)) {
        reason = StopReason::StopToken;
        break;
      }
      const int32_t am = _lm->next_token(_chat_ctx, next);
      if (am < 0) {
        reason = StopReason::DecodeError;
        break;
      }
      ++decode_calls;
      if (sampled_path) {
        const auto& lh = _lm->last_logits_host();
        next = lh.empty()
            ? am
            : sampler.sample(
                  std::span<const float>(lh.data(), lh.size()));
      } else {
        next = am;
      }
      if (next < 0) {
        reason = StopReason::DecodeError;
        break;
      }
      if (!is_stop(next)) {
        emit_chunk(tok.step(sd, next));
      } else {
        reason = StopReason::StopToken;
      }
      if (i + 1 == _max_new_tokens && !is_stop(next)) {
        reason = StopReason::MaxNewTokens;
      }
    }
    }
    const auto t_decode_end = clock::now();
    // Finalize the streamed reply (terminates the line on stdio / closes
    // the console entry in the web UI). Then commit the assistant's
    // end-of-turn marker to the K/V cache so the *next* turn sees a
    // clean turn boundary. Without this, the next prefill would tack the
    // new header straight onto the assistant's last content token and
    // confuse the model on long conversations. The EOT commit is
    // bookkeeping, so it isn't included in the decode tok/s figure.
    out_stream->end();
    // Commit the assistant-close token only for a genuine turn close. A
    // turn-CONTINUING stop (Gemma's <|tool_response>, emitted after a
    // tool call to request the result) leaves the turn OPEN so the
    // tool-results turn resumes it in place; committing <end_of_turn>
    // here would break the single-turn tool exchange.
    last_stop_token = next;
    if (assistant_close >= 0 && !tpl->stop_token_continues_turn(next)) {
      (void)_lm->next_token(_chat_ctx, assistant_close);
    }
    decode_calls_total += decode_calls;
    decode_s_total += std::chrono::duration<double>(
        t_decode_end - t_decode_start).count();
    return assistant_text;
  };

  // Plain-text prefill of a round's prompt ids, timed into the
  // accumulators. Returns the predicted next token (-1 on failure).
  // Round 1 with media uses prefill_multimodal_metal inline instead.
  auto prefill_plain =
      [&](std::span<const int32_t> pids) -> int32_t {
    const auto t0 = clock::now();
    const int32_t n = _lm->prefill(_chat_ctx, pids);
    const auto t1 = clock::now();
    prefill_s_total += std::chrono::duration<double>(t1 - t0).count();
    prefill_n_total += static_cast<int>(pids.size());
    return n;
  };

  // ---- round 1: prefill (media or plain) + decode ------------------
  int32_t next;
  if (have_media) {
    const auto t_prefill_start = clock::now();
    // ids -> TokenRefs: each pad id takes the next row of its item
    // (items consumed in chunk order per modality); everything else
    // is a plain text id. One (grid_h, grid_w) per IMAGE item feeds
    // mROPE; audio items contribute no grid (1-D RoPE positions).
    const int32_t image_pad = tpl->image_pad_token_id();
    const int32_t audio_pad = tpl->audio_pad_token_id();
    vector<MediaItem*>       imgs;
    vector<MediaItem*>       auds;
    vector<pair<int, int>>   image_grids;
    for (auto& it : items) {
      if (it.kind == genai::MediaChunk::Kind::Image) {
        imgs.push_back(&it);
        image_grids.emplace_back(it.mh, it.mw);
      } else {
        auds.push_back(&it);
      }
    }
    vector<genai::TokenRef> refs;
    refs.reserve(ids.size());
    size_t ii = 0, ai = 0;
    int    io = 0, ao = 0;
    for (int32_t id : ids) {
      genai::TokenRef r;
      if (image_pad >= 0 && id == image_pad && ii < imgs.size()) {
        r.kind = genai::TokenRef::Kind::ImageTokens;
        if (!imgs[ii]->embeddings_host.empty()) {
          r.embeddings_host = &imgs[ii]->embeddings_host;
        } else {
          r.embeddings_buf = &imgs[ii]->embeddings;
        }
        r.image_token_offset = io++;
        if (io >= imgs[ii]->n_tokens) { ++ii; io = 0; }
      } else if (audio_pad >= 0 && id == audio_pad
                 && ai < auds.size()) {
        r.kind             = genai::TokenRef::Kind::AudioTokens;
        r.embeddings_host  = &auds[ai]->embeddings_host;
        r.audio_token_offset = ao++;
        if (ao >= auds[ai]->n_tokens) { ++ai; ao = 0; }
      } else {
        r.kind    = genai::TokenRef::Kind::Text;
        r.text_id = id;
      }
      refs.push_back(r);
    }
    next = _lm->prefill_multimodal_metal(
        _chat_ctx, span<const genai::TokenRef>(refs),
        span<const pair<int, int>>(image_grids));
    const auto t_prefill_end = clock::now();
    prefill_s_total += std::chrono::duration<double>(
        t_prefill_end - t_prefill_start).count();
    prefill_n_total += static_cast<int>(ids.size());
  } else {
    next = prefill_plain(ids);
  }
  if (next < 0) {
    session()->warn(fmt(
        "TextChatStage('{}'): prefill failed (returned -1) after "
        "{} input tokens at ctx_pos={}. Most likely the K/V page "
        "pool (max_pages={}, page_tokens={}) is full -- the running "
        "conversation has outgrown the configured budget. Use "
        "`/clear` to reset the context or relaunch the pipeline "
        "with a larger max_pages. Emitting empty turn beat.",
        this->id(),
        static_cast<int>(ids.size()),
        _chat_ctx.seq_len(),
        _max_pages, _page_tokens));
    co_await ctx.write(0,
        build_turn_payload("", 0, 0, current_ctx_pos()));
    co_return;
  }
  _seeded = true;
  std::string assistant_text = decode_turn(next, ids);

  // ---- tool-call rounds (MCP) --------------------------------------
  // Scan the reply for <tool_call> blocks; run each tool locally, feed
  // the results back as a tool-response turn, and decode again. Repeat
  // until the model answers without calling a tool, or we hit the round
  // cap (a guard against a model that loops on tool calls). The FlexData
  // `text` beat carries the FINAL round's answer; every round streams.
  if (tools_active) {
    constexpr int kMaxToolRounds = 8;
    // Triage helpers: count the family's tool-call open markers, and cap
    // a string for logging (tool code / results can be large). The marker
    // is family-specific ("<tool_call>" for ChatML/Qwen, "<|tool_call>"
    // for Gemma-4) so it comes from the template.
    const string_view call_marker = tpl->tool_call_open_marker();
    auto count_markers = [&call_marker](const string& s) {
      int n = 0;
      for (size_t p = s.find(call_marker); p != string::npos;
           p = s.find(call_marker, p + call_marker.size())) {
        ++n;
      }
      return n;
    };
    auto excerpt = [](const string& s, size_t cap) {
      return s.size() <= cap ? s
                             : s.substr(0, cap) + "...[+"
                                   + std::to_string(s.size() - cap) + "B]";
    };
    int tool_round = 0;
    while (true) {
      const int markers = count_markers(assistant_text);
      auto calls = tpl->parse_tool_calls(assistant_text);
      // One trace per scan so a tool turn is always followed in the debug
      // log -- both when it fires and when it silently produces nothing.
      session()->log_debug(fmt(
          "TextChatStage('{}'): tool scan (round {}): {} tool-call "
          "marker(s), {} parsed call(s), reply {} chars",
          this->id(), tool_round, markers,
          static_cast<int>(calls.size()),
          static_cast<int>(assistant_text.size())));
      if (calls.empty()) {
        if (markers > 0) {
          // The model TRIED to call a tool but nothing parsed -- almost
          // always a malformed call (e.g. unescaped quotes inside a
          // `code` argument). This is exactly why such a turn "ends" with
          // no tool result: the raw block is returned verbatim. Make it
          // visible (warn) and dump the offending payload for triage.
          session()->warn(fmt(
              "TextChatStage('{}'): the model emitted {} tool-call "
              "block(s) that did not parse; no tool ran and the raw block "
              "is returned. Enable debug logging to see the payload (a "
              "common cause is malformed arguments).",
              this->id(), markers));
          const size_t a = assistant_text.find(call_marker);
          const string blk = (a == string::npos)
              ? assistant_text : assistant_text.substr(a);
          session()->log_debug(fmt(
              "TextChatStage('{}'): unparsed tool-call payload: {}",
              this->id(), excerpt(blk, 2000)));
        }
        break;
      }
      if (++tool_round > kMaxToolRounds) {
        session()->warn(fmt(
            "TextChatStage('{}'): reached the tool-call round cap ({}) "
            "in one turn; not running further tool calls.",
            this->id(), kMaxToolRounds));
        break;
      }
      vector<string> results;
      vector<string> tool_names;
      results.reserve(calls.size());
      tool_names.reserve(calls.size());
      for (const auto& c : calls) {
        session()->info(fmt(
            "TextChatStage('{}'): tool call '{}' args {}",
            this->id(), c.name, excerpt(c.arguments_json, 512)));
        string res = _tools.dispatch(c);
        session()->log_debug(fmt(
            "TextChatStage('{}'): tool '{}' returned {} chars: {}",
            this->id(), c.name, static_cast<int>(res.size()),
            excerpt(res, 1000)));
        results.push_back(std::move(res));
        tool_names.push_back(c.name);
      }
      vector<int32_t> tr_ids;
      if (!tpl->render_tool_results_turn(
              span<const string>(tool_names.data(), tool_names.size()),
              span<const string>(results.data(), results.size()),
              &tr_ids)
          || tr_ids.empty()) {
        session()->log_debug(fmt(
            "TextChatStage('{}'): tool-results turn rendered no tokens "
            "(template '{}'); ending the tool loop",
            this->id(), tpl->family_name()));
        break;
      }
      const int32_t tn = prefill_plain(tr_ids);
      if (tn < 0) {
        session()->warn(fmt(
            "TextChatStage('{}'): prefill of the tool-response turn "
            "failed (returned -1) at ctx_pos={}; assistant turn "
            "truncated.", this->id(), _chat_ctx.seq_len()));
        reason = StopReason::DecodeError;
        break;
      }
      session()->log_debug(fmt(
          "TextChatStage('{}'): fed {} tool result(s) back as {} tokens; "
          "decoding round {}", this->id(),
          static_cast<int>(results.size()),
          static_cast<int>(tr_ids.size()), tool_round));
      assistant_text = decode_turn(tn, tr_ids);
    }
  }

  // If the tool loop exited with the assistant turn still OPEN in the
  // K/V cache -- the last decode stopped on a turn-continuing token
  // (Gemma's <|tool_response>) but we stopped feeding tools (no
  // parseable call, the round cap, or a prefill error) -- close it now
  // so the next user turn starts on a clean boundary. A no-op for
  // families whose stop tokens all close the turn, and for turns that
  // ended on a genuine stop (assistant_close already committed).
  if (assistant_close >= 0
      && tpl->stop_token_continues_turn(last_stop_token)) {
    (void)_lm->next_token(_chat_ctx, assistant_close);
  }

  const double prefill_s    = prefill_s_total;
  const double decode_s     = decode_s_total;
  const int    prefill_n    = prefill_n_total;
  const int    decode_calls = decode_calls_total;
  const double prefill_tps =
      prefill_s > 0.0 ? prefill_n / prefill_s : 0.0;
  const double decode_tps =
      decode_s > 0.0 ? decode_calls / decode_s : 0.0;
  // ctx_pos is the context's K/V length AFTER this turn (incl. the
  // just-committed assistant close token) -- how full the
  // max_pages*page_tokens budget is.
  session()->info(fmt(
      "TextChatStage('{}'): prefill {} tok in {:.3f} s = {:.1f} tok/s, "
      "decode {} tok in {:.3f} s = {:.1f} tok/s, ctx_pos {}",
      this->id(),
      prefill_n, prefill_s, prefill_tps,
      decode_calls, decode_s, decode_tps,
      static_cast<std::int64_t>(_chat_ctx.seq_len())));

  // Tell the user *why* the turn ended when it wasn't a natural
  // model-emitted stop token. Silent on the happy path so well-
  // behaved turns stay quiet in the log.
  switch (reason) {
  case StopReason::StopToken:
    break;
  case StopReason::MaxNewTokens:
    session()->warn(fmt(
        "TextChatStage('{}'): hit max_new_tokens budget ({}) before "
        "the model emitted a stop token; assistant turn truncated. "
        "Increase max_new_tokens in the stage config to allow longer "
        "responses.",
        this->id(), _max_new_tokens));
    break;
  case StopReason::DecodeError:
    session()->warn(fmt(
        "TextChatStage('{}'): decode failed mid-generation at "
        "ctx_pos={} after {} of up to {} new tokens. Most likely the "
        "K/V page pool (max_pages={}, page_tokens={}) ran out -- the "
        "conversation has outgrown the configured budget. Use "
        "`/clear` to reset the context or relaunch the pipeline "
        "with a larger max_pages / page_tokens. Assistant turn "
        "truncated.",
        this->id(), _chat_ctx.seq_len(), decode_calls,
        _max_new_tokens, _max_pages, _page_tokens));
    break;
  case StopReason::PipelineStopped:
    session()->warn(fmt(
        "TextChatStage('{}'): pipeline stop requested mid-generation "
        "after {} new tokens; assistant turn truncated.",
        this->id(), decode_calls));
    break;
  }

  // Emit the per-turn FlexData object on out-port 0. When nothing is
  // wired downstream the runtime drops this write; when a feedback
  // pair (or any other consumer) is wired, it observes one beat per
  // assistant turn carrying the response plus timing/position
  // metadata.
  FlexData fd = FlexData::make_object();
  {
    auto root = fd.as_object();
    root.insert("text",
        FlexData::make_string(assistant_text));
    root.insert("prefill_ms",
        FlexData::make_int(
            static_cast<int64_t>(prefill_s * 1000.0)));
    root.insert("decode_ms",
        FlexData::make_int(
            static_cast<int64_t>(decode_s * 1000.0)));
    root.insert("ctx_pos",
        FlexData::make_int(
            static_cast<int64_t>(_chat_ctx.seq_len())));
  }
  co_await ctx.write(0,
      make_payload<FlexDataPayload>(std::move(fd)));
#else
  (void)user_text;
  // No apple-silicon LLM in this build: still emit a beat so feedback
  // loops don't hang. ctx_pos is 0 because there is no LM / no K/V ctx.
  co_await ctx.write(0,
      build_turn_payload("", 0, 0, 0));
#endif
  co_return;
}

VPIPE_REGISTER_STAGE(TextChatStage)
VPIPE_REGISTER_SPEC(TextChatStage, kSpec)

}
