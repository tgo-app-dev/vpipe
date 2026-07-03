#include "stages/text-chat-stage.h"

#include "common/beat-payload-intf.h"
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
#include "generative-models/token-muxer.h"
#include "generative-models/tokenizer.h"
#endif
#ifdef VPIPE_BUILD_APPLE_SILICON
#include <cstdlib>   // setenv (metal backend selection)
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <span>
#include <vector>

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
   .suggest_db = "models"},
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
};
const PortSpec kIports[] = {
  {.name = "user",
   .doc = "FlexData string: the user's turn text; may embed image/"
          "audio attachments as media-line markers (fs path or "
          "base64), spliced into the prefill via the model's own "
          "vision/audio towers",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "assistant",
   .doc = "FlexData {text,prefill_ms,decode_ms,ctx_pos} per turn "
          "(downstream optional)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
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

TextChatStage::~TextChatStage() = default;

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

  vector<int32_t> ids;
  bool have_media = !items.empty();
  if (have_media
      && !tpl->render_user_turn_media(
             span<const genai::MediaChunk>(chunks.data(),
                                           chunks.size()),
             /*is_first_turn=*/!_seeded, &ids)) {
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
    tpl->render_user_turn(text_only, /*is_first_turn=*/!_seeded, &ids);
  }

  using clock = std::chrono::steady_clock;
  const auto t_prefill_start = clock::now();
  int32_t next;
  if (have_media) {
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
  } else {
    next = _lm->prefill(_chat_ctx, ids);
  }
  const auto t_prefill_end = clock::now();
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

  // Stop check + assistant-turn close token come from the same
  // ChatTemplate. The template encapsulates the family-specific
  // stop set (Llama-3: eot/eom/end_of_text; ChatML: im_end/endoftext)
  // so this stage stays family-agnostic.
  auto is_stop =
      [tpl](int32_t id) {
        return tpl->is_stop_token(id);
      };
  const int32_t assistant_close = tpl->assistant_close_token_id();

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
  sampler.prime(std::span<const int32_t>(ids.data(), ids.size()));
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
  // Termination reason for this turn. Used after the decode loop to
  // surface non-natural stops (max budget, decode error, page
  // exhaustion, pipeline stop) to the user as a single warn line.
  enum class StopReason {
    StopToken,        // model emitted EOS/EOT -- the normal path
    MaxNewTokens,     // hit the per-turn token budget
    DecodeError,      // step_pipelined / materialize_lazy returned -1
    PipelineStopped,  // ctx.stop_requested() while decoding
  };
  StopReason reason = StopReason::StopToken;
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
      _chat_ctx, next,
      std::span<const int32_t>(ids.data(), ids.size()),
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
  // end-of-turn marker to the K/V cache so the *next* user turn sees a
  // clean turn boundary. Without this, the next prefill would tack the
  // new user header straight onto the assistant's last content token
  // and confuse the model on long conversations. The EOT commit is
  // bookkeeping, so it isn't included in the decode tok/s figure
  // logged below.
  out_stream->end();
  if (assistant_close >= 0) {
    (void)_lm->next_token(_chat_ctx, assistant_close);
  }

  const double prefill_s = std::chrono::duration<double>(
      t_prefill_end - t_prefill_start).count();
  const double decode_s = std::chrono::duration<double>(
      t_decode_end - t_prefill_end).count();
  const int prefill_n = static_cast<int>(ids.size());
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
