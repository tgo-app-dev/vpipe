#include "stages/visual-qa-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/media-line.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"


// No-MLX metal path: MLX-free headers driving image-VQA on the
// metal-compute backend (chat-template / loaded-language-model / sampler /
// metal-qwen-vision come in via the stage header).
#if defined(VPIPE_BUILD_APPLE_SILICON)
#include "generative-models/gemma4/gemma4-unified-embedder.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/tokenizer.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/tensor-beat-bridge.h"
#include "apple-silicon/tensor-beat.h"
#include <cstdlib>   // setenv (metal backend selection)
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

VisualQaStage::VisualQaStage(const SessionContextIntf* s,
                             string                    id,
                             vector<InEdge>            iports,
                             FlexData                  config)
  : TypedStage<VisualQaStage>(s, std::move(id), std::move(iports),
                              std::move(config))
{
  // Construction must succeed for any config (see Stage::fail_config):
  // a stage must construct so a graph can be built/edited before
  // required fields are supplied. Config problems are recorded via
  // fail_config (first message wins, ordered by importance) and
  // deferred to launch.
  // Nested video.fps default (1.0) has no flat ConfigKey form; seed it
  // here (the single source) and override from the video.* sub-object.
  _video_fps = 1.0f;

  // Scalar attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _hf_dir              = attr_str("hf_dir");
  _models_db           = attr_str("models_db");
  if (_models_db.empty()) {
    _models_db = "models";
  }
  _compute_dtype       = attr_str("compute_dtype");
  _page_tokens         = static_cast<int>(attr_int("page_tokens"));
  _coreml_vision_path  = attr_str("coreml_vision_path");
  _coreml_compute_units =
      static_cast<int>(attr_int("coreml_compute_units"));
  {
    int64_t v = attr_int("max_pages");
    if (v < 1) {
      fail_config(fmt(
          "VisualQaStage('{}'): max_pages must be >= 1 (got {})",
          this->id(), v));
    }
    _max_pages = static_cast<uint32_t>(v);
  }
  {
    int64_t v = attr_int("max_new_tokens");
    if (v < 1) {
      fail_config(fmt(
          "VisualQaStage('{}'): max_new_tokens must be >= 1 "
          "(got {})", this->id(), v));
    }
    _max_new_tokens = static_cast<int>(v);
  }
  {
    int64_t v = attr_int("num_images");
    if (v < 1) {
      fail_config(fmt(
          "VisualQaStage('{}'): num_images must be >= 1 (got {})",
          this->id(), v));
    }
    _num_images = static_cast<int>(v);
  }
  {
    int64_t v = attr_int("pause_ms_between_rounds");
    if (v < 0) { v = 0; }
    _pause_ms_between_rounds = static_cast<int>(v);
  }
  _batched_decode          = attr_bool("batched_decode");
  _i8_prefill              = attr_bool("i8_prefill");
  _pre_image_prompt        = attr_str("pre_image_prompt");
  _post_image_prompt       = attr_str("post_image_prompt");
  _decode_after_post_image = attr_bool("decode_after_post_image");
  if (_decode_after_post_image && _post_image_prompt.empty()) {
    session()->warn(fmt(
        "VisualQaStage('{}'): decode_after_post_image=true with empty "
        "post_image_prompt -- the model will be asked to produce a "
        "reply with no post-image instruction; consider setting a "
        "non-empty post_image_prompt", this->id()));
  }

  // Tri-state / composite attributes (disable_thinking, sampler, the
  // video sub-object, and the string-or-array questions) have no flat
  // ConfigKey form and are read from the config directly.
  const FlexData& cfg = this->config();
  if (cfg.is_object()) {
    auto root = cfg.as_object();
    if (root.contains("disable_thinking")) {
      _disable_thinking = root.at("disable_thinking").as_bool(false);
    }
#ifdef VPIPE_BUILD_APPLE_SILICON
    if (root.contains("sampler")) {
      _sampler_params = genai::parse_sampler_config(root.at("sampler"));
    }
#endif

    if (root.contains("video")) {
      FlexData v = root.at("video");
      if (v.is_object()) {
        auto vobj = v.as_object();
        if (vobj.contains("enabled")) {
          _video_enabled = vobj.at("enabled").as_bool(false);
        }
        if (vobj.contains("fps")) {
          double f = vobj.at("fps").as_real(1.0);
          if (!(f > 0.0)) {
            fail_config(fmt(
                "VisualQaStage('{}'): config.video.fps must be > 0 "
                "(got {})", this->id(), f));
          }
          _video_fps = static_cast<float>(f);
        }
      } else {
        fail_config(fmt(
            "VisualQaStage('{}'): config.video must be an object",
            this->id()));
      }
    }

    if (root.contains("questions")) {
      FlexData q = root.at("questions");
      if (q.is_string()) {
        _questions.emplace_back(string(q.as_string("")));
      } else if (q.is_array()) {
        for (FlexData e : q.as_array()) {
          if (!e.is_string()) {
            fail_config(fmt(
                "VisualQaStage('{}'): config.questions array entries "
                "must be strings", this->id()));
          }
          _questions.emplace_back(string(e.as_string("")));
        }
      } else {
        fail_config(fmt(
            "VisualQaStage('{}'): config.questions must be a string or "
            "array of strings", this->id()));
      }
    }
  }

  // Required-field checks, ordered by importance (fail_config keeps
  // only the first message).
  if (_hf_dir.empty()) {
    fail_config(fmt(
        "VisualQaStage('{}'): config.hf_dir is required (non-empty "
        "string)", this->id()));
  }
  if (_questions.empty()) {
    fail_config(fmt(
        "VisualQaStage('{}'): config.questions is required (non-empty "
        "string or array of strings)", this->id()));
  }
  for (const auto& qstr : _questions) {
    if (qstr.empty()) {
      fail_config(fmt(
          "VisualQaStage('{}'): config.questions contains an empty "
          "entry", this->id()));
    }
  }
  allocate_oports(spec().oports.size());   // sink: 0 oports
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
   .doc = "pre-converted CoreML vision tower path", .def_str = ""},
  {.key = "coreml_compute_units", .type = ConfigType::Int,
   .doc = "CoreML compute units (0=CPU,1=+GPU,2=All,3=+ANE)",
   .def_int = 2},
  {.key = "compute_dtype", .type = ConfigType::String,
   .doc = "bf16 | f16 | f32", .def_str = "bf16"},
  {.key = "page_tokens", .type = ConfigType::Int,
   .doc = "ContextManager K/V page size", .def_int = 512},
  {.key = "max_pages", .type = ConfigType::Int,
   .doc = "per-LM page pool capacity (>= 1)", .def_int = 64},
  {.key = "max_new_tokens", .type = ConfigType::Int,
   .doc = "per-question generation budget (>= 1)", .def_int = 1024},
  {.key = "num_images", .type = ConfigType::Int,
   .doc = "images/frames per Q&A round (>= 1)", .def_int = 1},
  {.key = "pause_ms_between_rounds", .type = ConfigType::Int,
   .doc = "sleep ms after each Q&A round", .def_int = 0},
  {.key = "batched_decode", .type = ConfigType::Bool,
   .doc = "batch question branches that share the prefix",
   .def_bool = true},
  {.key = "i8_prefill", .type = ConfigType::Bool,
   .doc = "accelerated mode (LOSSY): dynamic-int8 prefill GEMMs, ~2x their "
          "f16 rate on matrix-core GPUs at int8 quality (prefill is NOT "
          "token-exact with this on; IGNORED without NAX matmul2d -- matrix-core GPU + kernels). Default false; VPIPE_I8_GEMM "
          "overrides.",
   .def_bool = false},
  {.key = "pre_image_prompt", .type = ConfigType::String,
   .doc = "user-turn text before the image block", .def_str = ""},
  {.key = "post_image_prompt", .type = ConfigType::String,
   .doc = "user-turn text after the image block", .def_str = ""},
  {.key = "decode_after_post_image", .type = ConfigType::Bool,
   .doc = "decode a reply after post-image, then branch",
   .def_bool = false},
  {.key = "disable_thinking", .type = ConfigType::Bool,
   .doc = "override chat-template thinking default", .def_bool = false},
  {.key = "sampler", .type = ConfigType::Object,
   .doc = "decode sampler knobs (temperature/top_k/top_p/...)"},
  {.key = "video", .type = ConfigType::Object,
   .doc = "video mode: { enabled:bool, fps:real }"},
  {.key = "questions", .type = ConfigType::Any, .required = true,
   .doc = "string or array<string> asked per round"},
};
const PortSpec kIports[] = {
  {.name = "images", .doc = "RGB TensorBeat [3,H,W]; num_images per Q&A "
                            "round (or frames in video mode)",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "visual-qa",
  .doc       = "Sink: loads a vision-language model, encodes incoming "
               "image/video frames, and asks the configured questions per "
               "round, streaming answers to the UI. 0 oports.",
  .display_name = "Visual Q&A",
  .category  = StageCategory::Vision,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
VisualQaStage::spec() const noexcept
{
  return kSpec;
}

VisualQaStage::~VisualQaStage() = default;

Job
VisualQaStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
#if   defined(VPIPE_BUILD_APPLE_SILICON)
  // No-MLX build: load + run the VLM on the metal-compute backend.
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  auto* mgr = session() ? session()->generative_model_manager() : nullptr;
  if (!mgr) {
    session()->error(fmt(
        "VisualQaStage('{}'): no GenerativeModelManager", this->id()));
    co_return;
  }
  genai::LoadSpec spec;
  spec.hf_dir        = resolve_model_dir(session(), _models_db, _hf_dir);
  spec.compute_dtype = _compute_dtype;
  spec.page_tokens   = _page_tokens;
  spec.max_pages     = _max_pages;
  session()->info(fmt(
      "VisualQaStage('{}'): [metal/no-MLX] loading model from '{}' "
      "(dtype={}, num_images={}, questions={})",
      this->id(), _hf_dir, _compute_dtype, _num_images,
      _questions.size()));
  _lm = mgr->load(spec);
  if (!_lm || !_lm->valid()) {
    session()->error(fmt(
        "VisualQaStage('{}'): [metal] failed to load model from '{}'",
        this->id(), _hf_dir));
    _lm.reset();
    co_return;
  }
  // Accelerated mode (LOSSY, opt-in): dynamic-int8 prefill GEMMs. No-op
  // on backends without the route; env VPIPE_I8_GEMM overrides.
  if (_i8_prefill) { _lm->set_i8_prefill(true); }
  _chat_tpl = genai::make_chat_template(
      _lm->config().architecture, _lm->tokenizer(), _disable_thinking);
  if (!_chat_tpl) {
    session()->warn(fmt(
        "VisualQaStage('{}'): [metal] no chat template for '{}'; "
        "rounds will drop", this->id(), _lm->config().architecture));
  }
  _mvis  = _lm->metal_vision_encoder();
  _mgvis = _lm->metal_gemma4_vision_encoder();
  _mguni = _lm->gemma4_unified_embedder();
  if (_mguni && !_mguni->has_vision()) { _mguni = nullptr; }
  if (!_mvis && !_mgvis && !_mguni) {
    session()->warn(fmt(
        "VisualQaStage('{}'): [metal] no metal vision tower (model has "
        "no vision config?); every round will drop", this->id()));
  }
  // Optional CoreML vision tower override (host-f32 + metal-compute
  // letterbox preproc). When it loads it takes priority over _mvis.
  if (!_coreml_vision_path.empty()) {
    genai::CoreMLVisionEncoder::LoadSpec cm;
    cm.mlpackage_path = _coreml_vision_path;
    cm.compute_units  = _coreml_compute_units;
    cm.patch_size     = _lm->config().vision.patch_size;
    cm.spatial_merge_size = _lm->config().vision.spatial_merge_size;
    auto enc = genai::CoreMLVisionEncoder::create(
        cm, /*runtime=*/nullptr, session());
    if (enc && enc->implemented()) {
      session()->info(fmt(
          "VisualQaStage('{}'): [metal] using CoreML vision tower at "
          "'{}'", this->id(), _coreml_vision_path));
      _m_coreml = std::move(enc);
    } else {
      session()->warn(fmt(
          "VisualQaStage('{}'): [metal] CoreML vision tower load from "
          "'{}' failed; falling back to the metal MLX-free tower",
          this->id(), _coreml_vision_path));
    }
  }
  session()->info(fmt(
      "VisualQaStage('{}'): [metal/no-MLX] model ready ({} layers, "
      "vocab={}, vision={})", this->id(), _lm->config().n_layers,
      _lm->config().vocab_size,
      _m_coreml ? "coreml"
                : (_mvis ? "metal-tower"
                         : (_mgvis ? "metal-gemma4"
                                   : (_mguni ? "gemma4-unified" : "none")))));
#else
  session()->error(fmt(
      "VisualQaStage('{}'): this build was compiled without "
      "VPIPE_BUILD_APPLE_SILICON; the LLM subsystem is unavailable",
      this->id()));
#endif
  co_return;
}

Job
VisualQaStage::process(RuntimeContext& ctx)
{
  // Acquire one image beat. Using ctx.read() (not peek) means the
  // OportBuffer slot is moved out and the producer's backpressure
  // releases the moment we move past this image -- exactly the
  // early-release-while-encoding semantics the spec calls for. The
  // raw TensorBeat bytes live in `p` only until the unique_ptr goes
  // out of scope at the end of this process() iteration; the
  // encoded embeddings (which we DO keep) live in _round_images
  // instead.
  auto p = co_await ctx.read(0);
  if (!p) {
    ctx.signal_done();
    co_return;
  }

#if   defined(VPIPE_BUILD_APPLE_SILICON)
  if (!_lm) { co_return; }
  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(p.get());
  if (!tbp || tbp->shape.size() != 3 || tbp->shape[0] != 3
      || tbp->dtype != TensorBeat::DType::U8) {
    session()->warn(fmt(
        "VisualQaStage('{}'): [metal] expected TensorBeat [3,H,W] U8; "
        "dropping beat", this->id()));
    co_return;
  }
  const int H = static_cast<int>(tbp->shape[1]);
  const int W = static_cast<int>(tbp->shape[2]);
  if (_m_coreml || _mvis || _mgvis || _mguni) {
    if (_video_enabled && !_m_video_warned) {
      session()->warn(fmt(
          "VisualQaStage('{}'): [metal] video 2:1 temporal merge is not "
          "supported on the no-MLX path; encoding each frame as a "
          "standalone image (no per-frame timestamps). Use realtime-vqa "
          "for streaming video QA.", this->id()));
      _m_video_warned = true;
    }
    // Zero-copy input: a contiguous beat (incl. a Shared/UMA frame) is
    // read straight through its storage pointer; strided beats fall back
    // to a one-shot materialize.
    AlignedVector<std::uint8_t> scratch;
    const std::uint8_t* rgb;
    if (tbp->is_contiguous()) {
      rgb = tbp->as_u8();
    } else {
      scratch = tbp->materialize_contiguous();
      rgb = scratch.data();
    }
    const auto t_ve = std::chrono::steady_clock::now();
    MImg m;
    if (_m_coreml) {
      // CoreML Result already reports the POST-merger token grid. When
      // the frame is a Shared/UMA metal-compute buffer, bind it straight
      // into the letterbox kernel (zero-copy input via from_tensor_beat);
      // else stage the host bytes.
      auto* mc = session()->metal_compute();
      genai::CoreMLVisionEncoder::Result cr;
      if (mc && tbp->is_contiguous()
          && tbp->storage_class() == TensorStorageClass::Shared) {
        auto sb = metal_compute::from_tensor_beat(*mc, *tbp);
        cr = _m_coreml->encode_host(sb, H, W);
      } else {
        cr = _m_coreml->encode_host(rgb, H, W);
      }
      m.embeddings = std::move(cr.embeddings);
      m.n_tokens   = cr.n_tokens;
      m.mh         = cr.grid_h;
      m.mw         = cr.grid_w;
    } else if (_mgvis) {
      // Gemma-4 metal tower reports the POST-merger grid directly. In
      // video mode use the smaller per-frame budget (this metal path
      // still treats frames as a standalone image sequence -- see the
      // warning above -- but the token budget should match video).
      auto r = _mgvis->encode(
          rgb, H, W,
          _video_enabled ? _mgvis->video_soft_token_budget() : -1);
      m.embeddings = std::move(r.embeddings);
      m.n_tokens   = r.n_tokens;
      m.mh         = r.grid_h;
      m.mw         = r.grid_w;
    } else if (_mguni) {
      // Gemma-4-12B "unified": encoder-less shallow embedder. Host-f32 rows
      // ([n_tokens, H]) routed through the TokenRef embeddings_host splice.
      if (auto r = _mguni->encode_image(rgb, H, W)) {
        m.n_tokens       = r->n_tokens;
        m.mh             = r->grid_h;
        m.mw             = r->grid_w;
        m.embeddings_host = std::move(r->rows);
      }
    } else {
      // Metal MLX-free tower reports PRE-merger patches -> divide by S.
      const int S = std::max(1, _mvis->config().spatial_merge);
      auto r = _mvis->encode(rgb, H, W);
      m.embeddings = std::move(r.embeddings);
      m.n_tokens   = r.n_tokens;
      m.mh         = r.grid_h / S;
      m.mw         = r.grid_w / S;
    }
    const double ve_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_ve).count();
    session()->info(fmt(
        "VisualQaStage('{}'): [metal] vision encode ({}) {}x{} -> {} tok "
        "in {:.3f} s", this->id(),
        _m_coreml ? "coreml"
                  : (_mgvis ? "metal-gemma4"
                            : (_mguni ? "gemma4-unified" : "metal-tower")),
        W, H, m.n_tokens, ve_s));
    if (m.n_tokens > 0 &&
        (!m.embeddings.empty() || !m.embeddings_host.empty())) {
      _m_imgs.push_back(std::move(m));
    }
  } else if (!_m_vision_warned) {
    session()->warn(fmt(
        "VisualQaStage('{}'): [metal] vision tower unavailable; round "
        "will advance but won't run inference", this->id()));
    _m_vision_warned = true;
  }
#endif

  // Advance the round counter regardless of whether the encode
  // produced real embeddings. The spec wants us to "move the read
  // pointer forward to encourage early release" even on a partial
  // round -- ctx.read() already does that -- and to close rounds
  // by image-count so a follow-up image-set that finally finds a
  // working encoder will still align.
  ++_seen_this_round;
  if (_seen_this_round >= _num_images) {
#if   defined(VPIPE_BUILD_APPLE_SILICON)
    m_run_round_();
    _m_imgs.clear();
#endif
    _seen_this_round = 0;
    if (_pause_ms_between_rounds > 0) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(_pause_ms_between_rounds));
    }
  }
}

Job
VisualQaStage::drain(RuntimeContext& ctx)
{
  (void)ctx;
#if   defined(VPIPE_BUILD_APPLE_SILICON)
  if (_lm && !_m_imgs.empty()) {
    session()->info(fmt(
        "VisualQaStage('{}'): [metal] drain with partial round of {} "
        "image{} (requested {}); running questions over what we have",
        this->id(), _m_imgs.size(), _m_imgs.size() == 1 ? "" : "s",
        _num_images));
    m_run_round_();
    _m_imgs.clear();
  }
#endif
  co_return;
}


// ===================================================================
// No-MLX metal path. Image-mode Q&A round: per-frame metal embeddings
// spliced via mROPE multimodal prefill, then per-question decode.
// ===================================================================
#if defined(VPIPE_BUILD_APPLE_SILICON)

std::string
VisualQaStage::m_decode_(genai::LoadedLanguageModel::Context& ctx,
                         const genai::SamplerParams& sp)
{
  const auto* tpl = _chat_tpl.get();
  auto& tok = _lm->tokenizer();
  auto sd = tok.make_stream_decoder();
  auto out_stream = session()->open_text_stream();
  std::string out;
  int produced = 0;
  std::int32_t cur = ctx.last_predicted_id();   // prefill's first token

  // Thinking-ON templates open the reasoning block in the PROMPT
  // (`<think>\n` in the assistant extras), so its start token never
  // streams; emit the unified thinking-start marker so front ends can
  // fold the reasoning (the model-generated close token is rewritten
  // by the detokenizer).
  if (tpl && tpl->assistant_prompt_opens_thinking()) {
    const std::string m(media_line::kThinkStart);
    out += m;
    out_stream->write(m);
  }

  // MTP speculative head fast path (mtp.safetensors, Qwen3.5-OptiQ): the drafter
  // lets the verifier accept multiple tokens per forward, token-exact vs the
  // pdecode loop (greedy) or decode_pipelined (sampling -- the verify samples
  // each position). rope is anchored on ctx (post-image mROPE after a
  // multimodal prefill, or sequential after the per-question text prefill)
  // exactly as pdecode_begin reads it. Greedy OR penalty-free sampling -- the
  // verify applies no repetition/presence penalty, so a penalised sampler stays
  // on the loops below.
  const bool mtp_ok = cur >= 0 && _lm->mtp_available()
      && (genai::Sampler(sp).is_argmax()
          || (sp.repetition_penalty == 1.0f && sp.presence_penalty == 0.0f));
  if (mtp_ok) {
    auto is_stop = [tpl](std::int32_t id) {
      return tpl && tpl->is_stop_token(id);
    };
    auto on_toks =
        [&](std::span<const std::int32_t> toks) -> bool {
          for (std::int32_t id : toks) {
            const std::string piece = tok.step(sd, id);
            out += piece;
            out_stream->write(piece);
            ++produced;
          }
          return true;
        };
    _lm->mtp_generate(ctx, cur, _max_new_tokens, sp, is_stop, on_toks);
    out_stream->end();
    return out;
  }

  // GPU-resident pipelined decode: on-device embed + argmax/sampling (no
  // host logit pull) over the post-image mROPE positions. pdecode_commit
  // is the next_token equivalent -- it appends the (non-stop) input
  // token's KV and computes the next. No text prompt, so the penalty
  // seen-set is primed from the first token only. Falls back to the
  // synchronous next_token loop when the backend can't pipeline.
  const std::span<const std::int32_t> no_prompt;
  if (_lm->pdecode_begin(ctx, cur, no_prompt, sp, _max_new_tokens)) {
    // Run-ahead pipeline (matches text-chat / audio-transcribe): prime a
    // second forward so each token's encode overlaps the in-flight GPU step,
    // and refill BEFORE detokenizing so the GPU runs the next forward while the
    // host streams the current token. The old commit-then-next loop kept the
    // pipeline at depth 1, exposing the per-token encode bubble (~+2%/tok).
    const bool first_stop = tpl && tpl->is_stop_token(cur);
    bool committed = !first_stop ? _lm->pdecode_commit(ctx) : false;
    if (_lm->pdecode_supports_runahead() && committed &&
        _max_new_tokens > 1) {
      _lm->pdecode_commit(ctx);                   // speculative 2nd forward
    }
    if (!first_stop) {
      const std::string piece = tok.step(sd, cur);
      out += piece;
      out_stream->write(piece);
      ++produced;
    }
    for (int i = 1; i < _max_new_tokens && committed && !first_stop; ++i) {
      cur = _lm->pdecode_next(ctx);
      if (cur < 0) { break; }
      const bool stop = tpl && tpl->is_stop_token(cur);
      const bool cont = (i + 1 < _max_new_tokens) && !stop;
      committed = cont ? _lm->pdecode_commit(ctx) : false;   // refill ahead
      if (stop) { break; }
      const std::string piece = tok.step(sd, cur);
      out += piece;
      out_stream->write(piece);
      ++produced;
    }
    _lm->pdecode_end(ctx);
    out_stream->end();
    return out;
  }

  genai::Sampler sampler(sp);
  if (cur >= 0) {
    sampler.prime(std::span<const std::int32_t>(&cur, 1));
  }
  while (cur >= 0 && produced < _max_new_tokens) {
    if (tpl && tpl->is_stop_token(cur)) { break; }
    const std::string piece = tok.step(sd, cur);
    out += piece;
    out_stream->write(piece);
    ++produced;
    if (produced >= _max_new_tokens) { break; }
    const std::int32_t am = _lm->next_token(ctx, cur);
    if (am < 0) { break; }
    if (sampler.is_argmax()) {
      cur = am;
    } else {
      const auto& logits = _lm->last_logits_host();
      cur = logits.empty()
          ? am
          : sampler.sample(
                std::span<const float>(logits.data(), logits.size()));
    }
  }
  out_stream->end();
  return out;
}

void
VisualQaStage::m_run_round_()
{
  if (_m_imgs.empty()) { return; }
  const auto* tpl = _chat_tpl.get();
  if (!tpl || tpl->image_pad_token_id() < 0) {
    if (!_m_tpl_warned) {
      session()->warn(fmt(
          "VisualQaStage('{}'): [metal] no chat template / image-pad "
          "token; cannot run question", this->id()));
      _m_tpl_warned = true;
    }
    return;
  }
  const std::int32_t image_pad = tpl->image_pad_token_id();

  std::vector<int> image_token_counts;
  std::vector<std::pair<int, int>> image_grids;
  image_token_counts.reserve(_m_imgs.size());
  image_grids.reserve(_m_imgs.size());
  for (const auto& r : _m_imgs) {
    image_token_counts.push_back(r.n_tokens);
    // mh/mw are already POST-merger (normalised per encoder at encode).
    image_grids.emplace_back(r.mh, r.mw);
  }

  std::vector<std::int32_t> prefix_ids;
  if (!tpl->render_vlm_prefix_ex(image_token_counts,
                                 /*is_first_turn=*/true,
                                 _pre_image_prompt, _post_image_prompt,
                                 /*close_turn=*/_decode_after_post_image,
                                 &prefix_ids)) {
    session()->warn(fmt(
        "VisualQaStage('{}'): [metal] chat template '{}' does not "
        "support the requested prefix rendering", this->id(),
        tpl->family_name()));
    return;
  }

  // ids -> refs: image_pad runs reference per-frame host embeddings.
  auto build_refs = [&](std::span<const std::int32_t> ids) {
    std::vector<genai::TokenRef> refs;
    refs.reserve(ids.size());
    std::size_t img_idx = 0;
    int img_off = 0;
    for (std::int32_t id : ids) {
      genai::TokenRef r;
      if (id == image_pad && img_idx < _m_imgs.size()) {
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
  auto prefix_refs = build_refs(std::span<const std::int32_t>(prefix_ids));

  auto base_ctx = _lm->make_context();
  if (!base_ctx.valid()) {
    session()->warn(fmt(
        "VisualQaStage('{}'): [metal] failed to acquire base context",
        this->id()));
    return;
  }
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  const std::int32_t prefix_pred = _lm->prefill_multimodal_metal(
      base_ctx, std::span<const genai::TokenRef>(prefix_refs),
      std::span<const std::pair<int, int>>(image_grids));
  const double prefix_s =
      std::chrono::duration<double>(clock::now() - t0).count();
  if (prefix_pred < 0) {
    session()->warn(fmt(
        "VisualQaStage('{}'): [metal] vision-prefix prefill returned -1",
        this->id()));
    return;
  }
  session()->info(fmt(
      "VisualQaStage('{}'): [metal] shared vision prefix {} tok in "
      "{:.3f} s ({} image{})", this->id(),
      static_cast<int>(prefix_refs.size()), prefix_s, _m_imgs.size(),
      _m_imgs.size() == 1 ? "" : "s"));

  // Optional pre-question decode (decode_after_post_image): the prefix
  // already opened the assistant turn; decode a reply, then commit the
  // assistant_close so each question branch sees a closed turn.
  if (_decode_after_post_image) {
    const std::string reply = m_decode_(base_ctx, _sampler_params);
    session()->info(fmt("[pre-question reply] {}", reply));
    const std::int32_t ac = tpl->assistant_close_token_id();
    if (ac >= 0) { (void)_lm->next_token(base_ctx, ac); }
  }

  // Per-question fanout (serial branch + prefill + decode).
  for (std::size_t i = 0; i < _questions.size(); ++i) {
    auto child = _lm->branch(base_ctx);
    if (!child.valid()) { continue; }
    std::vector<std::int32_t> q_ids;
    if (_decode_after_post_image) {
      tpl->render_user_turn(_questions[i], /*is_first_turn=*/false, &q_ids);
    } else if (!tpl->render_vlm_completion(_questions[i], &q_ids)) {
      continue;
    }
    if (_lm->prefill(child, std::span<const std::int32_t>(q_ids)) < 0) {
      continue;
    }
    genai::SamplerParams p = _sampler_params;
    if (p.seed != 0) { p.seed += static_cast<std::uint64_t>(i); }
    session()->info(fmt(
        "VisualQaStage('{}'): [metal] --- Q{}: {}",
        this->id(), i + 1, _questions[i]));
    const std::string ans = m_decode_(child, p);
    session()->info(fmt("{}", ans));
  }
}
#endif

VPIPE_REGISTER_STAGE(VisualQaStage)
VPIPE_REGISTER_SPEC(VisualQaStage, kSpec)

}
