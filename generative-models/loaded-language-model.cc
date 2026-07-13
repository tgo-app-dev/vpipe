#include "generative-models/loaded-language-model.h"

#include "generative-models/chat-template.h"
#include "generative-models/qwen3/metal-audio-encoder.h"
#include "generative-models/gemma4/metal-gemma-model-exec.h"
#include "generative-models/llama3/metal-llama-model-exec.h"
#include "generative-models/qwen3/metal-qwen-model-exec.h"
#include "generative-models/gemma4/metal-gemma4-audio.h"
#include "generative-models/gemma4/gemma4-unified-embedder.h"
#include "generative-models/gemma4/metal-gemma4-vision.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/model-exec.h"
#include "generative-models/model-exec-registry.h"
#include "generative-models/sampler.h"
#include "generative-models/token-muxer.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <random>


#include <cstdlib>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe::genai {

struct LoadedLanguageModel::Impl {
  // Order matters: ModelExec and TokenMuxer reference into
  // _weights (the embedding table and per-layer projections).
  // ContextManager is consumed by ModelExec and shares the
  // session pointer.
  LoadedWeights                     weights;
  unique_ptr<Tokenizer>             tokenizer;
  unique_ptr<ChatTemplate>          chat_tpl;
  unique_ptr<ContextManager>        ctx_mgr;
  // Polymorphic via the ModelExec base. Dispatched on
  // weights.config.architecture in the ctor:
  //   "LlamaForCausalLM"               -> LlamaModelExec
  //   "Qwen3_5ForConditionalGeneration"-> Qwen3_5ModelExec
  // Lazy-decode and profiling paths (forward_chunk_lazy,
  // set_profile, ...) live on LlamaModelExec only; the entry points
  // below downcast and fall back to the sync path when the concrete
  // type doesn't support them.
  unique_ptr<ModelExec>             exec;
  // Metal-compute vision tower (Qwen3-VL, no MLX). Built on the metal
  // backend when the model has a vision config; encode() returns host
  // f32 image embeddings for prefill_multimodal_metal().
  unique_ptr<MetalQwenVisionEncoder> metal_vision;
  // Metal-compute audio tower (Qwen3-ASR, no MLX). Built on the metal
  // backend when the model has an audio config; encode() returns host
  // f32 audio embeddings for prefill_multimodal_metal(). MLX-free.
  unique_ptr<MetalAudioEncoder>      metal_audio;
  // Metal-compute Gemma-4 vision / audio towers (no MLX). Built on the
  // metal backend for the Gemma-4 family; the vision Result is a native-
  // f16 SharedBuffer + the audio Result host f32, both spliced via the
  // owns_kv metal multimodal path. MLX-free.
  unique_ptr<MetalGemma4VisionEncoder> metal_gemma4_vision;
  unique_ptr<MetalGemma4AudioEncoder>  metal_gemma4_audio;
  // Gemma-4-12B "unified" (gemma4_unified): encoder-less shallow multimodal
  // embedder (vision + audio in one object), weights from the mmproj GGUF.
  unique_ptr<Gemma4UnifiedEmbedder>    gemma4_unified;
  // The MLX executor every public entry point routes through.
  // Captured at construction time so we don't have to chase the
  // session lazy-init path on every call. May be nullptr -- see
  // ctor doc.
  MlxRuntime*                       runtime = nullptr;
  const SessionContextIntf*         session = nullptr;
  bool                              valid   = false;
  // Selected via VPIPE_LLM_BACKEND=metal: route prefill/next_token
  // through the metal-compute exec's token-id path instead of the
  // MLX muxer + forward_chunk path.
  bool                              metal_backend = false;
  // metal compute dtype is bf16 (else f16). Gates the native-f16
  // zero-copy multimodal splice (only valid when the model is f16).
  bool                              metal_bf16    = false;
  // Tokens the model must NEVER predict in text output, baked in at load
  // and always applied (Gemma-4 multimodal end markers <image|>/<audio|>;
  // the llama.cpp reference masks exactly these to -inf). set_suppressed_
  // tokens() merges any stage-provided ids ON TOP of this base so a stage
  // override never drops it.
  std::vector<std::int32_t>         base_suppress;
  // Active branch count of the in-flight m_bdecode pipelined batch
  // (CONSTANT-N). Captured at m_bdecode_begin so each m_bdecode_next can
  // tag its decode perf event with the per-step token count -- otherwise
  // the profiler reads 0 and the pipelined batch shows no tok/s.
  int                               bdecode_n     = 0;

  // ---- Wired-residency state -------------------------------------
  //
  // We DON'T set wired_limit at load (per-buffer commits during
  // bind ballooned load time across back-to-back processes). We
  // DON'T set it once per per generation call either (~11 sets +
  // restores for prefill + 10 decodes is too churny across
  // processes, AND lowered prefill tok/s ~10%). Instead we scope
  // wiring to a user-held WiredScope; while at least one scope
  // is alive, wired_limit is set; when the last scope exits, we
  // synchronize and restore wired_limit to 0. This matches mlx-
  // vlm's pattern where wired_limit is scoped to the whole
  // generate() call (prefill + N decodes).
  //
  // wired_limit_max is the GPU's reported safe cap; captured
  // once at ctor.
  std::size_t                       wired_limit_max = 0;
  // Refcount of active WiredScope holders. The dispatch_ machinery
  // serializes generation calls on a single MlxRuntime thread, so
  // the count itself is touched only from that thread and doesn't
  // need atomics. (If a future caller drives MLX from multiple
  // threads, switch to std::atomic<int>.)
  int                               wired_refcount = 0;
  std::size_t                       wired_prev_limit = 0;

};

namespace {

// Helper: run `f` on the MLX runtime when one is bound to the LM,
// else inline on the caller's thread. Inlines the void/non-void
// dispatch so callers can write `dispatch_(_impl->runtime, [&]{...})`
// without worrying about which path applies.
template <typename F>
auto dispatch_(MlxRuntime* rt, F&& f) -> std::invoke_result_t<F&>
{
  (void)rt;   // no MLX runtime in metal-only builds; always inline
  return f();
}


}

namespace {


// Map the parsed HF config to the metal Llama exec's Config (text
// path; llama3 rope-scaling fields default to a no-op when absent).
MetalLlamaModel::Config
make_metal_cfg_(const ModelConfig& c)
{
  MetalLlamaModel::Config m;
  m.n_layers      = c.n_layers;
  m.hidden        = c.hidden;
  m.n_heads       = c.n_heads;
  m.n_kv_heads    = c.n_kv_heads;
  m.head_dim      = c.head_dim;
  m.ffn_inner     = c.ffn_inner;
  m.vocab         = c.vocab_size;
  m.rope_theta    = c.rope_theta;
  m.rms_eps       = c.rms_eps;
  m.rope_factor   = c.rope_scaling.factor;
  m.rope_low_freq = c.rope_scaling.low_freq_factor;
  m.rope_high_freq = c.rope_scaling.high_freq_factor;
  m.rope_orig_ctx =
      static_cast<float>(c.rope_scaling.original_max_position_embeddings);
  m.max_seq       = 2048;
  return m;
}

// Map the parsed HF config to the metal Qwen exec's Config. Covers both
// the hybrid Qwen3.5(-VL) family (GDN + full-attn, 4-bit, gated q_proj,
// "language_model." weight root) and the dense Qwen3-ASR text decoder
// (all full-attn, 8-bit, non-gated, full RoPE, root weights) -- the
// differences are all derived from ModelConfig. The derivation lives on
// MetalQwenModel (config_from) so the tests share the exact mapping;
// this thin alias keeps the call sites below readable.
MetalQwenModel::Config
make_metal_qwen_cfg_(const ModelConfig& c)
{
  // MOSS-TTS-Local-v1.5 ("MossTTSLocalModel"): the backbone is a dense Qwen3
  // text LM stored under the "transformer." prefix, with a text head tied to
  // transformer.embed_tokens (text_lm_head == embed_tokens, byte-identical).
  // For text eval (perplexity) we build that backbone WITH its tied head so
  // it emits full [vocab] text logits -- unlike the TTS stage's
  // v15_backbone_cfg_, which sets backbone_only (no logits) for synthesis.
  // Values mirror v15_backbone_cfg_ / arch-detect.cc, head enabled.
  if (c.architecture == "MossTTSLocalModel") {
    MetalQwenModel::Config m;
    m.n_layers   = c.n_layers;
    m.hidden     = c.hidden;
    m.n_heads    = c.n_heads;
    m.n_kv_heads = c.n_kv_heads;
    m.head_dim   = c.head_dim > 0 ? c.head_dim
                 : (c.n_heads > 0 ? c.hidden / c.n_heads : 0);
    m.ffn_inner  = c.ffn_inner;
    m.vocab      = c.vocab_size;
    m.rope_theta = c.rope_theta;
    m.rms_eps    = c.rms_eps;
    m.rotary_dim = m.head_dim;
    m.full_attn_interval = 1;       // every layer is full attention
    // Honor a quantization block: a model-quantize'd MOSS has an AFFINE
    // backbone (mixed 4/8 or uniform) -- only its embed/tied-head stay bf16
    // (handled in MetalQwenModel's load). quant_bits drives the fused-width
    // kernel selection (w8g64 for uniform-8, else w4g64). The raw-HF bf16
    // source (no quant block) keeps the dense path.
    if (c.quantization.bits == 4 || c.quantization.bits == 8) {
      m.dense      = false;
      m.quant_bits = c.quantization.bits;
    } else {
      m.dense      = true;          // raw-HF bf16 (no affine quant)
    }
    // MOSS's Qwen3 backbone uses STANDARD RMSNorm (weight*h, ones-init), not
    // the zero-centered (1+weight) convention -- so the dense raw-HF +1 fold
    // must be DISABLED here or the logits are corrupted.
    m.zero_centered_norm = false;
    m.tie_embeddings   = true;      // tied text head -> [vocab] logits
    m.backbone_only    = false;     // MUST emit the lm_head (logits)
    m.attn_output_gate = false;     // plain Qwen3 attention (no output gate)
    m.use_bf16         = true;      // ctor overrides per compute_dtype
    m.weight_prefix    = "transformer.";
    m.model_seg        = "";
    return m;
  }
  // MOSS-TTS-Realtime ("MossTTSRealtime"): the backbone is a dense Qwen3-1.7B
  // text LM under "language_model.". It has no trained text head, but its input
  // text embedding (embed_tokens[0]) is byte-identical to
  // language_model.embed_tokens, so a head TIED to that emits full [vocab]
  // logits -- enough for A/B divergence (e.g. 8-bit vs bf16 quant error).
  // Absolute perplexity is meaningless (the model does not predict text). The
  // shape comes from language_config (parsed into the dense fields).
  if (c.architecture == "MossTTSRealtime") {
    MetalQwenModel::Config m;
    m.n_layers   = c.n_layers;
    m.hidden     = c.hidden;
    m.n_heads    = c.n_heads;
    m.n_kv_heads = c.n_kv_heads;
    m.head_dim   = c.head_dim > 0 ? c.head_dim
                 : (c.n_heads > 0 ? c.hidden / c.n_heads : 0);
    m.ffn_inner  = c.ffn_inner;
    m.vocab      = c.vocab_size;
    m.rope_theta = c.rope_theta;
    m.rms_eps    = c.rms_eps;
    m.rotary_dim = m.head_dim;
    m.full_attn_interval = 1;       // every layer is full attention
    if (c.quantization.bits == 4 || c.quantization.bits == 8) {
      m.dense      = false;         // model-quantize'd 8-bit affine backbone
      m.quant_bits = c.quantization.bits;
    } else {
      m.dense      = true;          // raw-HF bf16 (unquantized)
    }
    m.zero_centered_norm = false;   // plain Qwen3 std RMSNorm (no +1 fold)
    m.tie_embeddings   = true;      // tied text head -> [vocab] logits
    m.backbone_only    = false;     // MUST emit the lm_head (logits)
    m.attn_output_gate = false;     // plain Qwen3 attention (no output gate)
    m.use_bf16         = true;      // ctor overrides per compute_dtype
    m.weight_prefix    = "language_model.";
    m.model_seg        = "";
    return m;
  }
  return MetalQwenModel::config_from(c);
}

// Map the parsed vision_config to the metal vision tower's Config
// (Qwen3-VL ViT). MLX-free. Derivation lives on MetalQwenVisionEncoder
// (config_from) so the tests share the exact mapping.
MetalQwenVisionEncoder::Config
make_metal_vision_cfg_(const ModelConfig& c)
{
  return MetalQwenVisionEncoder::config_from(c);
}

// Map the parsed audio_config to the metal audio tower's Config
// (Qwen3-ASR Whisper-style encoder). MLX-free. Mirrors the mapping in
// make_metal_audio_encoder().
MetalAudioEncoder::Config
make_metal_audio_cfg_(const ModelConfig& c)
{
  MetalAudioEncoder::Config m;
  const auto& a = c.audio;
  m.n_mel          = a.num_mel_bins;
  m.d_model        = a.d_model;
  m.n_layers       = a.encoder_layers;
  m.n_heads        = a.encoder_attention_heads;
  m.ffn            = a.encoder_ffn_dim;
  m.output_dim     = a.output_dim;
  m.conv_hidden    = a.downsample_hidden_size;
  m.n_window       = a.n_window;
  m.n_window_infer = a.n_window_infer;
  m.sample_rate    = a.sample_rate;
  return m;
}

}

LoadedLanguageModel::LoadedLanguageModel(
    LoadedWeights              weights,
    unique_ptr<Tokenizer>      tokenizer,
    ComputeDtype               compute_dtype_in,
    int                        page_tokens,
    uint32_t                   max_pages,
    MlxRuntime*                runtime,
    const SessionContextIntf*  session,
    const std::string&         model_dir)
  : _impl(make_unique<Impl>())
{
  _impl->weights   = std::move(weights);
  _impl->tokenizer = std::move(tokenizer);
  _impl->runtime   = runtime;
  _impl->session   = session;

  if (!_impl->tokenizer) {
    if (session) {
      session->warn(fmt(
          "LoadedLanguageModel: tokenizer is null; cannot proceed"));
    }
    return;
  }

  // VPIPE_LOAD_PROFILE=1 logs per-phase wall timing during construction.
  using Clock = std::chrono::steady_clock;
  const bool profile_load =
      std::getenv("VPIPE_LOAD_PROFILE") != nullptr;
  auto t_phase = Clock::now();
  auto phase_log = [&](const char* name) {
    if (!profile_load || !session) {
      t_phase = Clock::now();
      return;
    }
    auto now = Clock::now();
    auto ms = std::chrono::duration<double, std::milli>(
        now - t_phase).count();
    session->info(fmt("[load-profile] {}: {:.1f} ms", name, ms));
    t_phase = now;
  };

  // Chat template: pure-CPU rendering helper, built off the runtime
  // worker because it doesn't touch any mlx::core::array state.
  // make_chat_template returns nullptr for unknown architectures;
  // raw prefill still works, only the lm->chat_template() accessor
  // returns nullptr in that case.
  _impl->chat_tpl =
      make_chat_template(_impl->weights.config.architecture,
                         *_impl->tokenizer);
  phase_log("chat_tpl");
  if (!_impl->chat_tpl && session) {
    session->warn(fmt(
        "LoadedLanguageModel: no chat template for architecture "
        "'{}'; chat_template() will return nullptr",
        _impl->weights.config.architecture));
  }


  // Build ContextManager + LlamaModelExec + TokenMuxer through the
  // MLX runtime (when one is bound) so the K/V page pool, the
  // dequantized embedding table, and the per-layer projection
  // bindings all live in arrays whose (Stream, encoder) TLS map is
  // owned by the same thread that later evaluates them. When no
  // runtime is bound (direct-construction tests), build inline and
  // the caller's thread owns everything end-to-end.
  bool ok = dispatch_(_impl->runtime, [&] {
    // Pin the model's working set as wired memory so the OS can't
    // evict weight pages under cache pressure. DEFERRED until
    // AFTER bind_weights_ + warmup_forward: setting wired_limit
    // upfront makes every per-buffer malloc during bind trigger
    // an individual MTLResidencySet::commit() to the Metal
    // driver. With ~320 per-tensor allocations during Qwen3.5's
    // weight bind, that's 320 GPU-driver round-trips per process;
    // across back-to-back processes the driver state accumulates
    // and load time balloons (42s -> 63s -> 68s observed). mlx-
    // vlm sidesteps this by scoping wired_limit to its generate()
    // call -- bind runs with wired_limit = 0 (no commits), then
    // generate sets the limit which resizes the residency set
    // with a SINGLE commit that adds all already-allocated
    // buffers in one go. We do the same: bind first, set
    // wired_limit second.
    metal_compute::MetalCompute* mc_be =
        session ? session->metal_compute() : nullptr;
    const char* be_env = std::getenv("VPIPE_LLM_BACKEND");
    const std::string& arch_be = _impl->weights.config.architecture;
    const bool metal_llama = arch_be == "LlamaForCausalLM";
    const bool metal_qwen = arch_be == "Qwen3_5ForConditionalGeneration"
        || arch_be == "Qwen3_5MoeForConditionalGeneration"
        || arch_be == "Qwen3ASRForConditionalGeneration"
        // MOSS-TTS-Local-v1.5: TTS model with a dense Qwen3 TEXT backbone
        // (under "transformer.", tied text head). Loaded as a Qwen text LM
        // so it emits [vocab] text logits for perplexity / text eval.
        || arch_be == "MossTTSLocalModel"
        // MOSS-TTS-Realtime: Qwen3-1.7B backbone under "language_model." with a
        // tied text head (embed_tokens). Loaded as a Qwen text LM so it emits
        // [vocab] logits. It has NO real text head (predicts audio), so
        // absolute perplexity is meaningless -- use the A/B divergence mode.
        || arch_be == "MossTTSRealtime";
    const bool metal_gemma = arch_be == "Gemma4ForConditionalGeneration"
        || arch_be == "Gemma4UnifiedForConditionalGeneration";
    // A plugin may register a new arch -> ModelExec factory. Consulted
    // BEFORE the built-in cascade below so a plugin family is selectable
    // by its config.json `architecture` string.
    const bool metal_plugin = ModelExecRegistry::get().contains(arch_be);
    // Select the metal backend when explicitly requested
    // (VPIPE_LLM_BACKEND=metal). In a build WITHOUT MLX the metal path is
    // the ONLY LM backend, so default to it when the env var is unset --
    // otherwise a plain text-chat load fails with "set VPIPE_LLM_BACKEND
    // =metal" even though there is no other backend to pick. An explicit
    // non-metal value still opts out (and then fails below, since there
    // is no MLX path to fall back to in this build).
    const bool backend_wants_metal =
        be_env == nullptr || std::string(be_env) == "metal";
    const bool use_metal = backend_wants_metal && mc_be != nullptr &&
        mc_be->valid() && !model_dir.empty() &&
        (metal_llama || metal_qwen || metal_gemma || metal_plugin);
    if (use_metal) {
      // Bookkeeping-only ctx_mgr (the metal exec owns its own KV).
      ContextManager::Spec s;
      s.page_tokens = page_tokens;
      s.max_pages   = max_pages;
      s.n_layers    = 0;
      _impl->ctx_mgr = make_unique<ContextManager>(s, session);
      if (metal_plugin) {
        // A plugin-registered arch builds its own ModelExec (which owns
        // its KV) from the raw config + runtime knobs.
        ModelExecCreateArgs a{
            model_dir, _impl->weights.config, mc_be, session,
            (std::uint32_t)page_tokens, (std::uint32_t)max_pages,
            compute_dtype_in == ComputeDtype::BF16};
        _impl->exec = ModelExecRegistry::get().create(arch_be, a);
        _impl->metal_bf16 = (compute_dtype_in == ComputeDtype::BF16);
      } else if (metal_gemma) {
        MetalGemmaModel::Config mcfg =
            MetalGemmaModel::config_from(_impl->weights.config);
        mcfg.page_tokens = page_tokens;
        // Gemma's metal KV is contiguous + PREALLOCATED (no lazy growth),
        // so the page cap directly sizes a multi-GB eager allocation PER
        // context AND per question branch (e4b: ~8 GB/context at the
        // generic 4096-page LoadSpec default -> a multi-branch realtime-vqa
        // run OOMs a 16 GB box). The callers stopped passing a Gemma-sized
        // max_pages, so bound the eager preallocation to a model-reasonable
        // default here. A smaller caller request is still honored.
        // TODO: replace with on-demand contiguous-KV growth (like the paged
        // Qwen/Llama backend) and drop this cap.
        // 512*128 = 65536 tok. Bounds the eager per-context (+per-branch) KV
        // preallocation -- raising it risks OOM on 16 GB boxes under multi-
        // branch realtime-vqa, so the DEFAULT stays 128. VPIPE_GEMMA_MAX_PAGES
        // overrides it (e.g. 144 for a 64k prefill+decode, 256 for the model's
        // native 128k) -- intended for the 64 GB box / long-context benches.
        std::uint32_t kGemmaMaxPages = 128;
        if (const char* e = std::getenv("VPIPE_GEMMA_MAX_PAGES")) {
          const int v = std::atoi(e);
          if (v >= 1 && v <= 4096) { kGemmaMaxPages = (std::uint32_t)v; }
        }
        const std::uint32_t gemma_pages =
            max_pages > 0 ? std::min(max_pages, kGemmaMaxPages)
                          : kGemmaMaxPages;
        mcfg.max_pages   = (int)gemma_pages;
        // Size the contiguous KV to the (capped) budget (page_tokens *
        // max_pages) rather than config_from's modest default: that is what
        // actually caps the conversation length, so without it a chat
        // silently truncates at the 2048 default. Clamp to 1M tokens as a
        // final hard bound on preallocation.
        {
          const std::int64_t budget =
              (std::int64_t)page_tokens * (std::int64_t)gemma_pages;
          mcfg.max_seq = (int)std::min<std::int64_t>(
              budget > 0 ? budget : mcfg.max_seq, 1 << 20);
        }
        mcfg.use_bf16    = (compute_dtype_in == ComputeDtype::BF16);
        _impl->metal_bf16 = mcfg.use_bf16;
        _impl->exec = make_unique<MetalGemmaModelExec>(
            model_dir, mc_be, mcfg, session);
      } else if (metal_qwen) {
        MetalQwenModel::Config mcfg =
            make_metal_qwen_cfg_(_impl->weights.config);
        mcfg.page_tokens = page_tokens;
        mcfg.max_pages   = (int)max_pages;
        mcfg.use_bf16    = (compute_dtype_in == ComputeDtype::BF16);
        _impl->metal_bf16 = mcfg.use_bf16;
        _impl->exec = make_unique<MetalQwenModelExec>(
            model_dir, mc_be, mcfg, session);
      } else {
        MetalLlamaModel::Config mcfg = make_metal_cfg_(_impl->weights.config);
        mcfg.page_tokens = page_tokens;
        mcfg.max_pages   = (int)max_pages;
        _impl->exec = make_unique<MetalLlamaModelExec>(
            model_dir, mc_be, mcfg, session);
      }
      if (!_impl->exec || !_impl->exec->valid()) {
        return false;
      }
      _impl->metal_backend = true;
      phase_log("metal_model_exec (load + bind, no MLX)");
      // Gemma-4 text decoders must NEVER emit the multimodal STRUCTURAL
      // control tokens -- the end-of-image / end-of-audio markers
      // (<image|>/<audio|>). The QAT-4bit 12B intermittently assigns them
      // the TOP logit in visually-themed text (e.g. "sketching the memory
      // <image|>topology"), and the llama.cpp reference masks exactly
      // these two (attr CONTROL) to -inf. Bake the same mask in as a
      // PERMANENT base so EVERY consumer (text-chat, which never calls
      // set_suppressed_tokens) gets it; stage-set suppressions merge on
      // top (see set_suppressed_tokens). No-op for families/tokenizers
      // without these tokens (special_token_id -> -1).
      if (metal_gemma && _impl->tokenizer) {
        for (const char* t : {"<image|>", "<audio|>"}) {
          const std::int32_t id = _impl->tokenizer->special_token_id(t);
          if (id >= 0) { _impl->base_suppress.push_back(id); }
        }
        if (!_impl->base_suppress.empty()) {
          _impl->exec->set_suppressed_tokens(_impl->base_suppress);
        }
      }
      // Metal vision tower (Qwen3-VL): host-f32 image embeddings for the
      // metal multimodal splice. No MLX in the forward.
      if (_impl->weights.config.vision.present && metal_qwen) {
        _impl->metal_vision = MetalQwenVisionEncoder::load(
            model_dir, mc_be,
            make_metal_vision_cfg_(_impl->weights.config));
        if (_impl->metal_vision) {
          _impl->metal_vision->set_session(session);
          phase_log("metal_vision_encoder (load + bind, no MLX)");
        }
      }
      // Gemma-4-12B "unified" (gemma4_unified): the encoder-LESS shallow
      // multimodal embedder (vision + audio in one object) from the mmproj
      // GGUF. Replaces the e4b ViT/Conformer (those are gated !unified).
      const bool g4_unified = _impl->weights.config.vision.unified ||
                              _impl->weights.config.audio.unified;
      if (metal_gemma && g4_unified) {
        const bool from_st = _impl->weights.config.vision.unified_st ||
                             _impl->weights.config.audio.unified_st;
        if (from_st) {
          // Raw safetensors 12B: adaptor weights live in model.safetensors.
          _impl->gemma4_unified =
              Gemma4UnifiedEmbedder::load_safetensors(model_dir, mc_be);
        } else {
          const std::string& mmp =
              !_impl->weights.config.vision.mmproj_path.empty()
                  ? _impl->weights.config.vision.mmproj_path
                  : _impl->weights.config.audio.mmproj_path;
          _impl->gemma4_unified = Gemma4UnifiedEmbedder::load(mmp);
        }
        if (_impl->gemma4_unified) {
          _impl->gemma4_unified->set_session(session);
          phase_log("gemma4_unified_embedder (load + bind, no MLX)");
        }
      }
      // Metal vision tower (Gemma-4 e4b): native-f16 image embeddings for the
      // owns_kv metal multimodal splice. No MLX in the forward.
      if (_impl->weights.config.vision.present && metal_gemma && !g4_unified) {
        _impl->metal_gemma4_vision = MetalGemma4VisionEncoder::load(
            model_dir, mc_be,
            MetalGemma4VisionEncoder::config_from(_impl->weights.config));
        if (_impl->metal_gemma4_vision) {
          _impl->metal_gemma4_vision->set_session(session);
          phase_log("metal_gemma4_vision_encoder (load + bind, no MLX)");
        }
      }
      // Metal audio tower (Gemma-4 e4b): host-f32 audio embeddings for the
      // owns_kv metal multimodal splice. No MLX in the forward.
      if (_impl->weights.config.audio.present && metal_gemma && !g4_unified) {
        _impl->metal_gemma4_audio = MetalGemma4AudioEncoder::load(
            model_dir, mc_be,
            MetalGemma4AudioEncoder::config_from(_impl->weights.config));
        if (_impl->metal_gemma4_audio) {
          _impl->metal_gemma4_audio->set_session(session);
          phase_log("metal_gemma4_audio_encoder (load + bind, no MLX)");
        }
      }
      // Metal audio tower (Qwen3-ASR): host-f32 audio embeddings for the
      // metal multimodal splice. No MLX in the forward.
      if (_impl->weights.config.audio.present &&
          arch_be == "Qwen3ASRForConditionalGeneration") {
        _impl->metal_audio = MetalAudioEncoder::load(
            model_dir, mc_be,
            make_metal_audio_cfg_(_impl->weights.config));
        if (_impl->metal_audio) {
          phase_log("metal_audio_encoder (load + bind, no MLX)");
        }
      }
    } else {
      // No MLX in this build: the metal backend is the only LM path, so
      // a non-metal load (VPIPE_LLM_BACKEND != metal, or unsupported
      // arch) can't be served.
      if (session) {
        session->warn(fmt(
            "LoadedLanguageModel: built without MLX; set "
            "VPIPE_LLM_BACKEND=metal and use a metal-supported model "
            "(arch '{}')", _impl->weights.config.architecture));
      }
      return false;
    }
    // Warmup: walk every quantized projection (q/k/v/o + gate/up/down)
    // and every norm through Metal so the kernels page in the
    // safetensors data instead of leaving it as mmap'd references.
    // Without this, the FIRST real prefill eats the full disk-read
    // cost -- tens of seconds for an 8B 4-bit model from a network
    // mount -- after the caller has already logged "model ready",
    // making load() a lie.
    //
    // We run TWO dummy forwards back-to-back on a throwaway context:
    //   1. T=8 prefill: triggers JIT for the multi-token matmul
    //      kernels (tile-shaped gemm), the rope/mrope path, and the
    //      last-layer trim (which slices T->1 then runs the MLP's
    //      gemv-shape kernel for one layer).
    //   2. T=1 decode: triggers JIT for the single-token gemv kernels
    //      across ALL layers and the decode-mode lazy-graph variant
    //      (emit_evals=false, no per-layer eval).
    // Without (1), the first real prefill JIT-compiles the gemm
    // matmul kernels at runtime, costing 100-300 ms before the
    // first response token. Without (2), the first decode step
    // JITs the decode kernels. Doing both at load time shifts that
    // cost to pipeline initialize() so steady-state requests are
    // hot from the first token.
    // MLX-path warmup (muxer + forward_chunk). The metal backend has
    // no muxer and JITs/loads inside MetalLlamaModel::load, so skip.

    // Token-id warmup for own-KV execs (metal backends + Gemma-4), which the
    // muxer warmup above skips. PSOs are already built at load (.function()
    // compiles them), but the FIRST real forward still eats the GPU
    // clock-ramp + first-command-buffer cost -- on metal Gemma-4 a 299-token
    // prefill measured ~958 ms cold vs ~540 ms warm. Paying it here with a
    // tiny dummy prefill + decode shifts that ~0.4 s off the user's first
    // turn. An 8-token warmup fully closes the gap (the cold cost is
    // GPU-clock/first-CB, NOT per-size buffer wiring), so keep it cheap.
    // Release both the ctx_mgr bookkeeping and the exec's own KV (mirrors
    // ~Context) so the warmup context doesn't leak.
    if (_impl->exec
        && (_impl->metal_backend || _impl->exec->owns_kv())) {
      auto warmup_ctx = _impl->ctx_mgr->acquire_root();
      if (warmup_ctx.valid()) {
        const std::int32_t warmup_ids[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const std::int32_t t = _impl->exec->prefill(
            warmup_ctx, std::span<const std::int32_t>(warmup_ids, 8));
        if (t >= 0) { (void)_impl->exec->decode_one(warmup_ctx, t); }
        _impl->ctx_mgr->release(warmup_ctx);
        _impl->exec->release_context(warmup_ctx);
        phase_log("warmup T=8 prefill+decode (own-KV: GPU clock + first CB)");
      }
    }

    // Cache the recommended wired-memory cap so the per-call
    // WiredScope can use it without re-querying device_info on
    // every generation. We DO NOT set the limit here; that
    // happens scoped per generation call (the mlx-vlm pattern)
    // so that:
    //   1. Bind-time per-tensor allocations don't trigger
    //      MTLResidencySet::commit() each — the residency set
    //      stays empty during load, so we sidestep the per-
    //      buffer driver round-trip cost that was making load
    //      time balloon across back-to-back processes.
    //   2. The wired set is empty at LM-not-actively-generating
    //      moments (including process exit), so we don't leave
    //      Metal with a "requested residency" for ~10GB of
    //      buffers when our process terminates.
    //   3. VPIPE_LLM_WIRED_LIMIT=0 still disables wiring entirely
    //      (the WiredScope no-ops); useful for A/B.
    // Disk-bound first pass is done; subsequent forward passes hit
    // the OS page cache and don't need per-layer eval to stay under
    // the GPU watchdog. Drop the per-layer sync so each user-visible
    // forward runs as one (or a few) Metal command buffers --
    // halves the decode wall time on an 8B model. Same idea for
    // ContextManager's per-write_kv eval: the next forward pass's
    // SDPA forces correct ordering through the lazy graph, so
    // making each slice_update synchronous is pure overhead.
    //
    // CAVEAT for hybrid (Qwen3.5 Gated DeltaNet) execs: the SSM
    // recurrence builds a much deeper per-layer lazy graph than a
    // pure Llama forward, so the 32-layer-wide combined graph
    // trips the Metal watchdog when the final argmax tries to
    // submit everything as one command buffer. Keep per-layer
    // eval ON for Qwen3.5 (and any future hybrid models flagged
    // by ModelConfig::is_linear_layer); the per-layer fence is
    // cheap relative to the SSM step cost.
    const bool is_hybrid =
        !_impl->weights.config.is_linear_layer.empty();
    if (!is_hybrid) {
      _impl->exec->set_eval_per_layer(false);
    }
    _impl->ctx_mgr->set_eval_per_write(false);
    return true;
  });
  if (!ok) {
    if (session) {
      session->warn(fmt(
          "LoadedLanguageModel: model exec failed to bind weights "
          "(architecture='{}'); model unusable",
          _impl->weights.config.architecture));
    }
    return;
  }
  _impl->valid = true;
}

LoadedLanguageModel::~LoadedLanguageModel()
{
  if (!_impl) { return; }
  // Drain destruction onto the MlxRuntime worker thread.
  //
  // Impl owns weights (the per-tensor mlx::core::array map),
  // exec (Llama/Qwen3.5 forward state with per-layer projection
  // arrays), muxer (embedding table refs / quantized lookup
  // arrays), and ctx_mgr (K/V pool arrays + per-context SSM state
  // arrays). Every one of these arrays was created on the runtime
  // worker thread, so its underlying Metal buffer's residency
  // bookkeeping (MTLCommandEncoder / MTLCommandBuffer TLS map)
  // belongs to that thread.
  //
  // Destroying the arrays on a DIFFERENT thread (the test/
  // application thread that drops the last shared_ptr) drops them
  // back into the allocator pool via a non-worker TLS view; the
  // documented MLX failure mode is the command-buffer-strand we
  // were chasing -- per the comment in mlx-runtime.h, "an array
  // created on one thread and later accessed from another either
  // throws or STRANDS THE GPU COMMAND BUFFER PAST THE WATCHDOG
  // DEADLINE." When that happens the process dies with an
  // uncaught METAL exception, and the next process inherits the
  // bad driver state (we observed back-to-back R1..R8 timeouts
  // across separate test invocations of both Qwen3.5 AND Llama).
  //
  // The fix is to make ~LoadedLanguageModel deterministically
  // tear Impl down on the same thread that built it. Synchronize
  // first so no pending GPU work references the buffers we're
  // about to release.
  // Metal-only build: no MLX runtime / arrays to drain onto a worker.
  _impl.reset();
}

// ---- WiredScope (refcount-based wired_limit RAII) -----------------

LoadedLanguageModel::WiredScope::~WiredScope()
{
  if (!_lm || !_lm->_impl) { return; }
  _lm = nullptr;
}

LoadedLanguageModel::WiredScope::WiredScope(WiredScope&& o) noexcept
  : _lm(o._lm)
{
  o._lm = nullptr;
}

LoadedLanguageModel::WiredScope&
LoadedLanguageModel::WiredScope::operator=(WiredScope&& o) noexcept
{
  if (this != &o) {
    // Release any state this scope was holding first.
    this->~WiredScope();
    _lm = o._lm;
    o._lm = nullptr;
  }
  return *this;
}

LoadedLanguageModel::WiredScope
LoadedLanguageModel::wired_scope()
{
  WiredScope ws;
  if (!valid() || !_impl) { return ws; }
  return ws;
}

bool
LoadedLanguageModel::valid() const noexcept
{
  return _impl && _impl->valid;
}

const ModelConfig&
LoadedLanguageModel::config() const noexcept
{
  return _impl->weights.config;
}

const Tokenizer&
LoadedLanguageModel::tokenizer() const noexcept
{
  return *_impl->tokenizer;
}

const ChatTemplate*
LoadedLanguageModel::chat_template() const noexcept
{
  return _impl ? _impl->chat_tpl.get() : nullptr;
}

VisionEncoder*
LoadedLanguageModel::vision_encoder() const noexcept
{
  return nullptr;   // MLX-only towers; metal encoders wired separately
}

AudioEncoder*
LoadedLanguageModel::audio_encoder() const noexcept
{
  return nullptr;
}

MetalQwenVisionEncoder*
LoadedLanguageModel::metal_vision_encoder() const noexcept
{
  return _impl ? _impl->metal_vision.get() : nullptr;
}

MetalAudioEncoder*
LoadedLanguageModel::metal_audio_encoder() const noexcept
{
  return _impl ? _impl->metal_audio.get() : nullptr;
}

MetalGemma4VisionEncoder*
LoadedLanguageModel::metal_gemma4_vision_encoder() const noexcept
{
  return _impl ? _impl->metal_gemma4_vision.get() : nullptr;
}

MetalGemma4AudioEncoder*
LoadedLanguageModel::metal_gemma4_audio_encoder() const noexcept
{
  return _impl ? _impl->metal_gemma4_audio.get() : nullptr;
}

Gemma4UnifiedEmbedder*
LoadedLanguageModel::gemma4_unified_embedder() const noexcept
{
  return _impl ? _impl->gemma4_unified.get() : nullptr;
}

const std::vector<float>&
LoadedLanguageModel::last_logits_host() const noexcept
{
  static const std::vector<float> kEmpty;
  return (_impl && _impl->exec) ? _impl->exec->last_logits_host() : kEmpty;
}

// ---- Context RAII ---------------------------------------------------

LoadedLanguageModel::Context::~Context()
{
  if (!_lm || !_lm->_impl || !_lm->_impl->ctx_mgr || !_id.valid()) {
    return;
  }
  // ctx_mgr->release touches the K/V page pool's mlx::core::array
  // values (clears the slot mask and, on a partial-tail CoW, may
  // free a page). When a runtime is bound, run it on that runtime
  // so ref-count drops happen with the same TLS encoder map that
  // built the buffers; otherwise (direct-construction tests) just
  // run inline.
  //
  // Destructors are implicitly noexcept(true): any exception that
  // escapes here during stack unwinding (e.g. an unwinding triggered
  // by ctx.write throwing "write to closed buffer" during pipeline
  // teardown) calls std::terminate(). dispatch_ can throw -- bad_alloc
  // from the inbox lambda, future_error if the worker had been torn
  // down, or a Metal command-buffer-execution-failed re-thrown from
  // a synchronize() that the runtime worker happened to run. Swallow
  // everything so clean SIGINT teardown is never derailed by a
  // bookkeeping release call.
  try {
    auto* impl = _lm->_impl.get();
    auto id    = _id;
    dispatch_(_lm->_impl->runtime, [impl, id] {
      impl->ctx_mgr->release(id);
      // Own-KV execs (metal backends + Gemma-4) hold the real KV keyed
      // by ContextId (the ctx_mgr above is bookkeeping-only,
      // n_layers=0). Release the exec's KV too, or it leaks on every
      // context.
      if (impl->exec && (impl->metal_backend || impl->exec->owns_kv())) {
        impl->exec->release_context(id);
      }
    });
  } catch (...) {
    // Intentional: destructor must not propagate during unwinding.
    // The K/V pages this context owned will be reclaimed when the
    // page pool itself is torn down with the LM.
  }
}

LoadedLanguageModel::Context::Context(Context&& o) noexcept
  : _lm(o._lm)
  , _id(o._id)
  , _last_predicted(o._last_predicted)
  , _rope_next_position(o._rope_next_position)
{
  o._lm                  = nullptr;
  o._id                  = {};
  o._last_predicted      = -1;
  o._rope_next_position  = -1;
}

LoadedLanguageModel::Context&
LoadedLanguageModel::Context::operator=(Context&& o) noexcept
{
  if (this != &o) {
    if (_lm && _lm->_impl && _lm->_impl->ctx_mgr && _id.valid()) {
      // Same swallow-all-exceptions reasoning as ~Context: this op
      // is declared noexcept, so any throw from dispatch_ is a
      // terminate(). Callers (e.g. ctx vector resize during scene
      // close after the output buffer was closed) must never abort
      // because of a bookkeeping release.
      try {
        auto* impl = _lm->_impl.get();
        auto id    = _id;
        dispatch_(_lm->_impl->runtime, [impl, id] {
          impl->ctx_mgr->release(id);
          if (impl->exec
              && (impl->metal_backend || impl->exec->owns_kv())) {
            impl->exec->release_context(id);
          }
        });
      } catch (...) {
      }
    }
    _lm                    = o._lm;
    _id                    = o._id;
    _last_predicted        = o._last_predicted;
    _rope_next_position    = o._rope_next_position;
    o._lm                  = nullptr;
    o._id                  = {};
    o._last_predicted      = -1;
    o._rope_next_position  = -1;
  }
  return *this;
}

int
LoadedLanguageModel::Context::seq_len() const
{
  if (!_lm || !_lm->_impl) { return 0; }
  // owns_kv() execs (the metal backends) track K/V internally and the
  // LM-side ctx_mgr is bookkeeping-only (n_layers=0, length stays 0)
  // -- ask the exec first. -1 = "not tracked here" falls through to
  // the ContextManager path.
  if (_lm->_impl->exec && _lm->_impl->exec->owns_kv()) {
    const int n = _lm->_impl->exec->context_seq_len(_id);
    if (n >= 0) { return n; }
  }
  if (!_lm->_impl->ctx_mgr) { return 0; }
  return _lm->_impl->ctx_mgr->seq_len_of(_id);
}

// ---- LoadedLanguageModel generation API -----------------------------
//
// Every generation entry point routes its MLX-touching body through
// MlxRuntime so the actual work runs on the dedicated MLX thread.
// The per-LM mutex is gone: serialisation is now handled by the
// runtime's single-thread inbox (which is also strictly serial across
// all LMs in the session). The callers continue to use the API
// synchronously -- run() blocks until the worker finishes.

LoadedLanguageModel::Context
LoadedLanguageModel::make_context()
{
  Context c;
  if (!valid()) { return c; }
  c._lm = this;
  c._id = dispatch_(_impl->runtime, [this] {
    return _impl->ctx_mgr->acquire_root();
  });
  return c;
}

LoadedLanguageModel::Context
LoadedLanguageModel::branch(const Context& parent)
{
  Context c;
  if (!valid() || !parent.valid()) { return c; }
  c._id = dispatch_(_impl->runtime, [this, &parent] {
    return _impl->ctx_mgr->branch(parent._id);
  });
  if (!c._id.valid()) { return c; }
  if (_impl->metal_backend
      || (_impl->exec && _impl->exec->owns_kv())) {
    const bool ok = dispatch_(_impl->runtime, [this, &parent, &c] {
      return _impl->exec->branch_context(parent._id, c._id);
    });
    if (!ok) { return Context{}; }
  }
  c._lm = this;
  c._last_predicted     = parent._last_predicted;
  c._rope_next_position = parent._rope_next_position;
  return c;
}

vector<LoadedLanguageModel::Context>
LoadedLanguageModel::branch(const Context& parent, int n_branches)
{
  vector<Context> out;
  if (!valid() || !parent.valid() || n_branches < 1) {
    return out;
  }
  auto ids = dispatch_(_impl->runtime, [this, &parent, n_branches] {
    return _impl->ctx_mgr->branch(parent._id, n_branches);
  });
  out.reserve(ids.size());
  for (auto id : ids) {
    Context c;
    if (!id.valid()) { continue; }
    if (_impl->metal_backend
        || (_impl->exec && _impl->exec->owns_kv())) {
      const bool ok = dispatch_(_impl->runtime, [this, &parent, id] {
        return _impl->exec->branch_context(parent._id, id);
      });
      if (!ok) { continue; }
    }
    c._lm                 = this;
    c._id                 = id;
    c._last_predicted     = parent._last_predicted;
    c._rope_next_position = parent._rope_next_position;
    out.push_back(std::move(c));
  }
  return out;
}

int32_t
LoadedLanguageModel::prefill(Context&                       ctx,
                             span<const int32_t>            tokens)
{
  if (!valid() || !ctx.valid() || tokens.empty()) { return -1; }
  PerfAuxScope _perf(_impl->session, kPerfLaneLLM, kGvidLlmPrefill,
                     kPerfLlmPrefillBegin, tokens.size());
  if (_impl->metal_backend
      || (_impl->exec && _impl->exec->owns_kv())) {
    // Token-id path: own-KV execs (metal backends + Gemma-4) embed +
    // cache internally. Gemma-4 needs the raw ids for its per-layer-
    // input embeddings, so it MUST take this path (not muxer +
    // forward_chunk).
    int32_t r = dispatch_(_impl->runtime, [this, &ctx, tokens]() -> int32_t {
      return _impl->exec->prefill(ctx._id, tokens);
    });
    if (r >= 0) { ctx._last_predicted = r; }
    return r;
  }
  return -1;   // metal-only build reaches here only if not metal_backend
}

int32_t
LoadedLanguageModel::prefill_multimodal(Context&             ctx,
                                        span<const TokenRef> refs)
{
  // No MLX: metal backend only -- route to the host-splice path (audio
  // / text; empty grids => 1-D RoPE).
  return prefill_multimodal_metal(ctx, refs, {});
}

int32_t
LoadedLanguageModel::prefill_multimodal_metal(
    Context&                                  ctx,
    span<const TokenRef>                      refs,
    span<const std::pair<int, int>>           image_grids)
{
  if (!valid() || !ctx.valid() || refs.empty()) { return -1; }
  PerfAuxScope _perf(_impl->session, kPerfLaneLLM, kGvidLlmPrefill,
                     kPerfLlmPrefillBegin, refs.size());
  if (!_impl->metal_backend) {
    if (_impl->session) {
      _impl->session->warn(fmt(
          "LoadedLanguageModel::prefill_multimodal_metal: requires the "
          "metal backend (VPIPE_LLM_BACKEND=metal)"));
    }
    return -1;
  }

  // owns_kv metal Gemma: PLE needs the raw ids + embed_scale lives inside
  // the forward, so the embedding-splice path below (prefill_embeddings)
  // can't serve it. Build (ids, mm rows, positions) from refs and drive
  // the metal exec's token-id multimodal prefill (the forward splices the
  // rows + zeros the PLE ids). DeepStack/mROPE grids are ignored (Gemma
  // is soft-token + 1-D RoPE). Branches on Kind so AudioTokens reuse it.
  if (auto* gx =
          dynamic_cast<MetalGemmaModelExec*>(_impl->exec.get())) {
    const int Hh = _impl->weights.config.hidden;
    std::vector<std::int32_t> ids;
    ids.reserve(refs.size());
    std::vector<int>   positions;
    std::vector<float> mm_rows;
    bool ok = true;
    for (std::size_t i = 0; i < refs.size(); ++i) {
      const auto& r = refs[i];
      const bool is_img = r.kind == TokenRef::Kind::ImageTokens;
      const bool is_aud = r.kind == TokenRef::Kind::AudioTokens;
      if (!is_img && !is_aud) { ids.push_back(r.text_id); continue; }
      ids.push_back(0);
      positions.push_back((int)i);
      const int off = is_img ? r.image_token_offset : r.audio_token_offset;
      if (r.embeddings_buf != nullptr) {
        const auto* sb = r.embeddings_buf;
        const std::size_t o = (std::size_t)off * Hh;
        if ((o + Hh) * 2 > sb->byte_size()) { ok = false; break; }
        const auto* src =
            static_cast<const _Float16*>(sb->contents()) + o;
        for (int d = 0; d < Hh; ++d) { mm_rows.push_back((float)src[d]); }
      } else if (r.embeddings_host != nullptr) {
        const auto& hv = *r.embeddings_host;
        const std::size_t o = (std::size_t)off * Hh;
        if (o + Hh > hv.size()) { ok = false; break; }
        for (int d = 0; d < Hh; ++d) { mm_rows.push_back(hv[o + d]); }
      } else {
        ok = false;
        break;
      }
    }
    if (!ok) { return -1; }
    const int n_mm = (int)positions.size();
    if (n_mm == 0) {
      int32_t pr = dispatch_(_impl->runtime,
          [&]() -> int32_t { return gx->prefill(ctx._id, ids); });
      if (pr >= 0) { ctx._last_predicted = pr; }
      return pr;
    }
    int32_t pred = dispatch_(_impl->runtime, [&]() -> int32_t {
      return gx->prefill_multimodal_ids(ctx._id, ids, mm_rows, n_mm,
                                        positions);
    });
    if (pred >= 0) { ctx._last_predicted = pred; }
    return pred;
  }

  const int H = _impl->weights.config.hidden;
  const int n = (int)refs.size();

  // 3-axis mROPE positions when image grids are supplied (VQA); else
  // plain 1-D RoPE (audio / text).
  std::vector<int32_t> pos_t, pos_h, pos_w;
  int rope_next = -1;
  const bool have_grids = !image_grids.empty();
  if (have_grids) {
    if (!build_mrope_position_ids(refs, image_grids, &pos_t, &pos_h, &pos_w,
                                  &rope_next)) {
      return -1;
    }
  }

  // ---- Native-f16 zero-copy fast path ------------------------------
  // When the model computes in f16 and every image/audio row is backed
  // by an f16 SharedBuffer (TokenRef::embeddings_buf), assemble the
  // [n,H] stream directly in f16: text rows from the metal embed gather
  // (f16), image rows memcpy'd straight from the encoder's f16 buffer
  // (no host f32 round-trip, no cast). The image tokens dominate, so the
  // f16-only copy is the big win.
  auto* mc_dev = _impl->session ? _impl->session->metal_compute() : nullptr;
  if (mc_dev != nullptr && !_impl->metal_bf16) {
    bool all_buf = true;
    for (int i = 0; i < n; ++i) {
      if (refs[i].kind != TokenRef::Kind::Text
          && refs[i].embeddings_buf == nullptr) { all_buf = false; break; }
    }
    if (all_buf) {
      const std::size_t rowb = (std::size_t)H * 2;   // f16 row bytes
      metal_compute::SharedBuffer x =
          mc_dev->make_shared_buffer((std::size_t)n * rowb);
      std::vector<std::int32_t> tids2;
      for (int i = 0; i < n; ++i) {
        if (refs[i].kind == TokenRef::Kind::Text) {
          tids2.push_back(refs[i].text_id);
        }
      }
      metal_compute::SharedBuffer trows_buf;
      if (!tids2.empty()) {
        trows_buf = _impl->exec->embed_text_buf(
            std::span<const std::int32_t>(tids2.data(), tids2.size()));
      }
      bool ok = !x.empty()
          && (tids2.empty()
              || (!trows_buf.empty()
                  && trows_buf.byte_size() >= tids2.size() * rowb));
      if (ok) {
        auto* xb = static_cast<std::uint8_t*>(x.contents());
        const auto* tb = static_cast<const std::uint8_t*>(
            tids2.empty() ? nullptr : trows_buf.contents());
        std::size_t tk = 0;
        for (int i = 0; i < n; ++i) {
          std::uint8_t* dst = xb + (std::size_t)i * rowb;
          if (refs[i].kind == TokenRef::Kind::Text) {
            std::memcpy(dst, tb + tk * rowb, rowb);
            ++tk;
          } else {
            const auto* sb = refs[i].embeddings_buf;
            const int off = (refs[i].kind == TokenRef::Kind::ImageTokens)
                ? refs[i].image_token_offset : refs[i].audio_token_offset;
            const std::size_t o = (std::size_t)off * rowb;
            if (o + rowb > sb->byte_size()) { ok = false; break; }
            std::memcpy(dst,
                static_cast<const std::uint8_t*>(sb->contents()) + o, rowb);
          }
        }
      }
      if (ok) {
        int32_t pred = -1;
        if (have_grids) {
          std::vector<int32_t> pos(3 * (std::size_t)n);
          for (int t = 0; t < n; ++t) {
            pos[t] = pos_t[t];
            pos[n + t] = pos_h[t];
            pos[2 * n + t] = pos_w[t];
          }
          pred = _impl->exec->prefill_embeddings_mrope_buf(
              ctx._id, std::move(x), pos, n);
          if (pred >= 0) {
            ctx._last_predicted = pred;
            ctx._rope_next_position = rope_next;
          }
        } else {
          pred = _impl->exec->prefill_embeddings_buf(
              ctx._id, std::move(x), n);
          if (pred >= 0) { ctx._last_predicted = pred; }
        }
        return pred;
      }
      // else: fall through to the host-f32 path below.
    }
  }

  // Assemble the [n, H] host-f32 stream: text rows from the metal embed
  // table, image/audio rows from the encoder's f16 SharedBuffer
  // (embeddings_buf, read back to f32) or host f32 buffer
  // (embeddings_host, e.g. audio). Used for bf16 models and any case the
  // f16 fast path above didn't take.
  std::vector<float> emb((std::size_t)n * H, 0.0f);
  std::vector<int32_t> tids;
  std::vector<int> tpos;
  for (int i = 0; i < n; ++i) {
    if (refs[i].kind == TokenRef::Kind::Text) {
      tids.push_back(refs[i].text_id);
      tpos.push_back(i);
    }
  }
  std::vector<float> trows;
  if (!tids.empty()) {
    trows = _impl->exec->embed_text_rows(
        std::span<const std::int32_t>(tids.data(), tids.size()));
    if ((int)trows.size() < (int)tids.size() * H) { return -1; }
  }
  std::size_t tk = 0;
  for (int i = 0; i < n; ++i) {
    float* dst = &emb[(std::size_t)i * H];
    if (refs[i].kind == TokenRef::Kind::Text) {
      std::memcpy(dst, &trows[tk * H], (std::size_t)H * sizeof(float));
      ++tk;
    } else {
      const int off = (refs[i].kind == TokenRef::Kind::ImageTokens)
          ? refs[i].image_token_offset : refs[i].audio_token_offset;
      if (refs[i].embeddings_buf != nullptr) {
        // f16 SharedBuffer (vision) -> f32 row.
        const auto* sb = refs[i].embeddings_buf;
        const std::size_t o = (std::size_t)off * H * 2;
        if (o + (std::size_t)H * 2 > sb->byte_size()) { return -1; }
        const auto* p = reinterpret_cast<const _Float16*>(
            static_cast<const std::uint8_t*>(sb->contents()) + o);
        for (int k = 0; k < H; ++k) { dst[k] = (float)p[k]; }
      } else {
        const std::vector<float>* src = refs[i].embeddings_host;
        if (src == nullptr) { return -1; }
        const std::size_t o = (std::size_t)off * H;
        if (o + (std::size_t)H > src->size()) { return -1; }
        std::memcpy(dst, src->data() + o, (std::size_t)H * sizeof(float));
      }
    }
  }

  int32_t pred = -1;
  if (have_grids) {
    std::vector<int32_t> pos(3 * (std::size_t)n);
    for (int t = 0; t < n; ++t) {
      pos[t] = pos_t[t];
      pos[n + t] = pos_h[t];
      pos[2 * n + t] = pos_w[t];
    }
    pred = _impl->exec->prefill_embeddings_mrope(ctx._id, emb, pos, n);
    if (pred >= 0) {
      ctx._last_predicted = pred;
      ctx._rope_next_position = rope_next;
    }
  } else {
    pred = _impl->exec->prefill_embeddings(ctx._id, emb, n);
    if (pred >= 0) { ctx._last_predicted = pred; }
  }
  return pred;
}


int32_t
LoadedLanguageModel::next_token(Context& ctx)
{
  if (!valid() || !ctx.valid()) { return -1; }
  if (ctx._last_predicted < 0) {
    if (_impl->session) {
      _impl->session->warn(fmt(
          "LoadedLanguageModel::next_token: no prior prediction "
          "in the context (call prefill first or pass a forced id)"));
    }
    return -1;
  }
  return next_token(ctx, ctx._last_predicted);
}

int32_t
LoadedLanguageModel::next_token(Context& ctx, int32_t forced)
{
  if (!valid() || !ctx.valid()) { return -1; }
  PerfAuxScope _perf(_impl->session, kPerfLaneLLM, kGvidLlmDecode,
                     kPerfLlmDecodeBegin, 1);
  if (_impl->metal_backend
      || (_impl->exec && _impl->exec->owns_kv())) {
    // After a multimodal (mROPE) prefill, ctx._rope_next_position holds
    // the rotary position for the next token (the KV-slot position is
    // wrong post-image); advance it per step. Plain text decode uses the
    // KV-slot position (rope override < 0).
    const int rope_pos = ctx._rope_next_position;
    int32_t r = dispatch_(_impl->runtime,
        [this, &ctx, forced, rope_pos]() -> int32_t {
          return rope_pos >= 0
              ? _impl->exec->decode_one_at(ctx._id, forced, rope_pos)
              : _impl->exec->decode_one(ctx._id, forced);
        });
    if (r >= 0) {
      ctx._last_predicted = r;
      if (ctx._rope_next_position >= 0) { ++ctx._rope_next_position; }
    }
    return r;
  }
  (void)forced;
  return -1;   // metal-only build serves decode via the metal branch above
}

int32_t
LoadedLanguageModel::next_token_greedy(Context& ctx)
{
  if (!valid() || !ctx.valid()) { return -1; }
  if (ctx._last_predicted < 0) { return next_token(ctx); }
  return next_token_greedy(ctx, ctx._last_predicted);
}

int32_t
LoadedLanguageModel::next_token_greedy(Context& ctx, int32_t forced)
{
  if (!valid() || !ctx.valid()) { return -1; }
  // Metal backend only: route to the exec's greedy fast path (on-GPU
  // embed + argmax, no host logit pull). Mirrors the next_token() metal
  // bookkeeping. The MLX backend has no logit-free decode -> next_token().
  if (_impl->metal_backend) {
    const int rope_pos = ctx._rope_next_position;
    int32_t r = dispatch_(_impl->runtime,
        [this, &ctx, forced, rope_pos]() -> int32_t {
          return rope_pos >= 0
              ? _impl->exec->decode_one_greedy_at(ctx._id, forced, rope_pos)
              : _impl->exec->decode_one_greedy(ctx._id, forced);
        });
    if (r >= 0) {
      ctx._last_predicted = r;
      if (ctx._rope_next_position >= 0) { ++ctx._rope_next_position; }
    }
    return r;
  }
  return next_token(ctx, forced);
}

namespace {
// Translate the host SamplerParams to the GPU sampler config. `greedy`
// reuses the host Sampler's argmax-equivalence test so the metal path
// takes the GPU argmax kernel for exactly the configs the host would.
GpuSamplerParams gpu_sampler_params_(const SamplerParams& p)
{
  GpuSamplerParams g;
  g.greedy             = Sampler(p).is_argmax();
  g.temperature        = p.temperature;
  g.top_k              = p.top_k;
  g.top_p              = p.top_p;
  g.min_p              = p.min_p;
  g.repetition_penalty = p.repetition_penalty;
  g.presence_penalty   = p.presence_penalty;
  g.seed               = p.seed;
  // seed 0 = "non-deterministic" for the host Sampler (it seeds from
  // random_device). The GPU derives per-step seeds from this base, so a
  // 0 base would make every turn identical -- mint a random nonzero base
  // to preserve the host's behaviour.
  if (g.seed == 0 && !g.greedy) {
    std::random_device rd;
    g.seed = ((std::uint64_t)rd() << 32) ^ (std::uint64_t)rd() ^ 0x1ull;
  }
  return g;
}
}  // namespace

bool
LoadedLanguageModel::pdecode_begin(Context& ctx, int32_t first_token,
                                   span<const int32_t> prompt,
                                   const SamplerParams& params, int max_tokens)
{
  if (!valid() || !ctx.valid() || !_impl->metal_backend) { return false; }
  // pdecode_begin processes the prompt (prefill) + primes the pipeline;
  // the prompt prefill dominates, so it shows on the LLM lane as
  // text-prefill (per-token decode is recorded in pdecode_next).
  PerfAuxScope _perf(_impl->session, kPerfLaneLLM, kGvidLlmPrefill,
                     kPerfLlmPrefillBegin, prompt.size());
  const GpuSamplerParams sp = gpu_sampler_params_(params);
  const int rope_first = ctx._rope_next_position;
  bool ok = dispatch_(_impl->runtime, [&]() -> bool {
    return _impl->exec->pdecode_begin(ctx._id, first_token, prompt, sp,
                                      max_tokens, rope_first);
  });
  if (ok) { ctx._last_predicted = first_token; }
  return ok;
}

bool
LoadedLanguageModel::pdecode_commit(Context& ctx)
{
  if (!valid() || !ctx.valid() || !_impl->metal_backend) { return false; }
  bool ok = dispatch_(_impl->runtime, [&]() -> bool {
    return _impl->exec->pdecode_commit(ctx._id);
  });
  // Mirror next_token's mROPE bookkeeping: each committed step appends one
  // token's KV and advances the rotary position by one.
  if (ok && ctx._rope_next_position >= 0) { ++ctx._rope_next_position; }
  return ok;
}

int32_t
LoadedLanguageModel::pdecode_next(Context& ctx)
{
  if (!valid() || !ctx.valid() || !_impl->metal_backend) { return -1; }
  PerfAuxScope _perf(_impl->session, kPerfLaneLLM, kGvidLlmDecode,
                     kPerfLlmDecodeBegin, 1);
  int32_t r = dispatch_(_impl->runtime, [&]() -> int32_t {
    return _impl->exec->pdecode_next(ctx._id);
  });
  if (r >= 0) { ctx._last_predicted = r; }
  return r;
}

void
LoadedLanguageModel::pdecode_end(Context& ctx)
{
  if (!valid() || !ctx.valid() || !_impl->metal_backend) { return; }
  dispatch_(_impl->runtime, [&]() -> int {
    _impl->exec->pdecode_end(ctx._id);
    return 0;
  });
}

bool
LoadedLanguageModel::pdecode_supports_runahead() const
{
  return valid() && _impl->metal_backend && _impl->exec
      && _impl->exec->pdecode_supports_runahead();
}

bool
LoadedLanguageModel::m_batched_decode_supported() const
{
  return valid() && _impl->metal_backend && _impl->exec
      && _impl->exec->supports_batched_decode();
}

bool
LoadedLanguageModel::m_batched_decode_step(
    std::span<Context*>           ctxs,
    std::span<const std::int32_t> in_tokens,
    std::vector<float>&           out_logits)
{
  if (!valid() || !_impl->metal_backend || !_impl->exec) { return false; }
  const int N = static_cast<int>(ctxs.size());
  if (N == 0 || static_cast<int>(in_tokens.size()) != N) { return false; }
  // One batched step decodes N branches in parallel; record it as a
  // single text-decode block with value = branch count.
  PerfAuxScope _perf(_impl->session, kPerfLaneLLM, kGvidLlmDecode,
                     kPerfLlmDecodeBegin, static_cast<std::uint64_t>(N));
  std::vector<ContextId>    cids(static_cast<std::size_t>(N));
  std::vector<std::int32_t> rope_pos(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    if (ctxs[static_cast<std::size_t>(i)] == nullptr
        || !ctxs[static_cast<std::size_t>(i)]->valid()) {
      return false;
    }
    cids[static_cast<std::size_t>(i)] = ctxs[static_cast<std::size_t>(i)]->_id;
    // -1 => sequential text decode (rope == KV slot); >= 0 => mROPE-advanced
    // position after a multimodal prefill (mirrors pdecode_begin's
    // rope_first = ctx._rope_next_position).
    rope_pos[static_cast<std::size_t>(i)] =
        ctxs[static_cast<std::size_t>(i)]->_rope_next_position;
  }
  const bool ok = dispatch_(_impl->runtime, [&]() -> bool {
    return _impl->exec->batched_decode_logits(
        std::span<const ContextId>(cids.data(), cids.size()),
        in_tokens,
        std::span<const std::int32_t>(rope_pos.data(), rope_pos.size()),
        out_logits);
  });
  if (!ok) { return false; }
  // Advance each branch's bookkeeping: the appended token is now its last
  // predicted input, and a post-image branch's mROPE position steps by 1.
  for (int i = 0; i < N; ++i) {
    ctxs[static_cast<std::size_t>(i)]->_last_predicted =
        in_tokens[static_cast<std::size_t>(i)];
    if (ctxs[static_cast<std::size_t>(i)]->_rope_next_position >= 0) {
      ctxs[static_cast<std::size_t>(i)]->_rope_next_position += 1;
    }
  }
  return true;
}

bool
LoadedLanguageModel::m_bdecode_supported() const
{
  return valid() && _impl->metal_backend && _impl->exec
      && _impl->exec->supports_batched_pipelined_decode();
}

bool
LoadedLanguageModel::m_bdecode_begin(std::span<Context*> ctxs,
                                     std::span<const std::int32_t> first_tokens,
                                     const SamplerParams& params, int max_tokens)
{
  if (!valid() || !_impl->metal_backend || !_impl->exec) { return false; }
  const int N = static_cast<int>(ctxs.size());
  if (N == 0 || static_cast<int>(first_tokens.size()) != N) { return false; }
  std::vector<ContextId>    cids(static_cast<std::size_t>(N));
  std::vector<std::int32_t> rope_pos(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    Context* c = ctxs[static_cast<std::size_t>(i)];
    if (c == nullptr || !c->valid()) { return false; }
    cids[static_cast<std::size_t>(i)] = c->_id;
    // -1 => sequential text decode; >= 0 => mROPE-advanced post-image
    // anchor (mirrors pdecode_begin's rope_first).
    rope_pos[static_cast<std::size_t>(i)] = c->_rope_next_position;
  }
  const GpuSamplerParams sp = gpu_sampler_params_(params);
  const bool ok = dispatch_(_impl->runtime, [&]() -> bool {
    return _impl->exec->bdecode_begin(
        std::span<const ContextId>(cids.data(), cids.size()), first_tokens, sp,
        max_tokens, std::span<const std::int32_t>(rope_pos.data(),
                                                  rope_pos.size()));
  });
  if (ok) {
    for (int i = 0; i < N; ++i) {
      ctxs[static_cast<std::size_t>(i)]->_last_predicted =
          first_tokens[static_cast<std::size_t>(i)];
    }
    // Constant-N pipelined batch: remember the branch count so each
    // m_bdecode_next can tag its decode perf block with the per-step
    // token count (= N) for the profiler's tok/s.
    _impl->bdecode_n = N;
  }
  return ok;
}

bool
LoadedLanguageModel::m_bdecode_supports_runahead() const
{
  return valid() && _impl->metal_backend && _impl->exec
      && _impl->exec->bdecode_supports_runahead();
}

bool
LoadedLanguageModel::m_bdecode_commit()
{
  if (!valid() || !_impl->metal_backend || !_impl->exec) { return false; }
  return dispatch_(_impl->runtime,
                   [&]() -> bool { return _impl->exec->bdecode_commit(); });
}

bool
LoadedLanguageModel::m_bdecode_next(std::vector<std::int32_t>& out_tokens)
{
  if (!valid() || !_impl->metal_backend || !_impl->exec) { return false; }
  // One pipelined step decodes all N branches in parallel; tag the decode
  // block with the branch count (value on begin + end) so the profiler
  // derives tok/s, matching m_batched_decode_step's value = N.
  PerfAuxScope _perf(_impl->session, kPerfLaneLLM, kGvidLlmDecode,
                     kPerfLlmDecodeBegin,
                     static_cast<std::uint64_t>(_impl->bdecode_n));
  return dispatch_(_impl->runtime, [&]() -> bool {
    return _impl->exec->bdecode_next(out_tokens);
  });
}

void
LoadedLanguageModel::m_bdecode_end()
{
  if (!valid() || !_impl->metal_backend || !_impl->exec) { return; }
  dispatch_(_impl->runtime, [&]() -> int {
    _impl->exec->bdecode_end();
    return 0;
  });
}

bool
LoadedLanguageModel::m_reserve_branches_supported() const
{
  return valid() && _impl->metal_backend && _impl->exec
      && _impl->exec->supports_branch_pool();
}

vector<LoadedLanguageModel::Context>
LoadedLanguageModel::m_reserve_branches(int n, int max_tokens)
{
  vector<Context> out;
  if (!m_reserve_branches_supported() || n < 1) { return out; }
  out.reserve(static_cast<std::size_t>(n));
  for (int b = 0; b < n; ++b) {
    // A lightweight bookkeeping id (the metal exec owns the real KV); the
    // reserve below allocates the exec-private pooled slot keyed by it.
    ContextId id = dispatch_(_impl->runtime,
                             [this] { return _impl->ctx_mgr->acquire_root(); });
    if (!id.valid()) { break; }
    const bool ok = dispatch_(_impl->runtime, [this, id, max_tokens] {
      return _impl->exec->reserve_branch_context(id, max_tokens);
    });
    if (!ok) {
      dispatch_(_impl->runtime,
                [this, id] { _impl->ctx_mgr->release(id); return 0; });
      break;
    }
    Context c;
    c._lm = this;
    c._id = id;
    out.push_back(std::move(c));
  }
  return out;
}

bool
LoadedLanguageModel::m_rebranch(Context& child, const Context& parent)
{
  if (!m_reserve_branches_supported() || !child.valid() || !parent.valid()) {
    return false;
  }
  const bool ok = dispatch_(_impl->runtime, [this, &child, &parent] {
    return _impl->exec->rebranch_context(parent._id, child._id);
  });
  if (!ok) { return false; }
  child._last_predicted     = parent._last_predicted;
  child._rope_next_position = parent._rope_next_position;
  return true;
}

bool
LoadedLanguageModel::m_detach_branch(Context& child)
{
  if (!m_reserve_branches_supported() || !child.valid()) { return false; }
  const bool ok = dispatch_(_impl->runtime, [this, &child] {
    return _impl->exec->detach_branch_context(child._id);
  });
  if (ok) {
    child._last_predicted     = -1;
    child._rope_next_position = -1;
  }
  return ok;
}

bool
LoadedLanguageModel::mtp_available() const
{
  return valid() && _impl->exec && _impl->exec->supports_mtp();
}

void
LoadedLanguageModel::set_mtp_prefix_seed(bool on)
{
  if (_impl && _impl->exec) { _impl->exec->set_mtp_prefix_seed(on); }
}

void
LoadedLanguageModel::set_suppressed_tokens(std::span<const std::int32_t> ids)
{
  if (!_impl || !_impl->exec) { return; }
  // Merge the caller's ids ON TOP of the permanent base (Gemma multimodal
  // end markers) so a stage-set suppression never drops the base mask.
  if (_impl->base_suppress.empty()) {
    _impl->exec->set_suppressed_tokens(ids);
    return;
  }
  std::vector<std::int32_t> merged = _impl->base_suppress;
  for (std::int32_t id : ids) {
    if (std::find(merged.begin(), merged.end(), id) == merged.end()) {
      merged.push_back(id);
    }
  }
  _impl->exec->set_suppressed_tokens(
      std::span<const std::int32_t>(merged.data(), merged.size()));
}

std::span<const std::int32_t>
LoadedLanguageModel::base_suppressed_tokens() const noexcept
{
  if (!_impl) { return {}; }
  return std::span<const std::int32_t>(_impl->base_suppress.data(),
                                       _impl->base_suppress.size());
}

void
LoadedLanguageModel::set_i8_prefill(bool on)
{
  if (_impl && _impl->exec) { _impl->exec->set_i8_gemm(on); }
}

bool
LoadedLanguageModel::mtp_generate(
    Context& ctx, int32_t first_token, int max_tokens,
    const SamplerParams&                                 params,
    const std::function<bool(int32_t)>&                  is_stop,
    const std::function<bool(span<const int32_t>)>&      on_tokens,
    int* produced, bool* hit_stop)
{
  if (produced) { *produced = 0; }
  if (hit_stop) { *hit_stop = false; }
  if (!valid() || !ctx.valid() || !_impl->exec || max_tokens <= 0) {
    return false;
  }
  // rope_first = the first decoded token's rotary position. -1 (sequential
  // text) or the mROPE-advanced position a prior multimodal prefill stored on
  // the context (mirrors pdecode_begin's rope_first = ctx._rope_next_position).
  const int rope_first = ctx._rope_next_position;
  // The verify's sampler: greedy => argmax accept; non-greedy => speculative
  // sampling (per-position GPU sample). Same SamplerParams->GpuSamplerParams
  // path the pdecode loop uses (seed 0 -> a random base so turns differ).
  const GpuSamplerParams sp = gpu_sampler_params_(params);
  PerfAuxScope _perf(_impl->session, kPerfLaneLLM, kGvidLlmDecode,
                     kPerfLlmDecodeBegin, 1);
  int  prod   = 0;
  bool hit    = false;
  // Track the last produced token so the context's bookkeeping reflects the
  // final decode state (the verify appends each kept token's KV internally).
  int32_t last_tok = first_token;
  auto on_wrapped =
      [&last_tok, &on_tokens](span<const int32_t> toks) -> bool {
        if (!toks.empty()) { last_tok = toks.back(); }
        return on_tokens ? on_tokens(toks) : true;
      };
  const bool ok = dispatch_(_impl->runtime, [&]() -> bool {
    return _impl->exec->mtp_generate(ctx._id, first_token, max_tokens,
                                     rope_first, sp, is_stop, on_wrapped,
                                     &prod, &hit);
  });
  // One mtp_generate produces the WHOLE decode (many tokens accepted across the
  // spec rounds), but the perf scope opened with a placeholder count of 1 --
  // retag its end event with the real token count so the profiler reports decode
  // tok/s, not ~1/decode. (next_token tags 1/call; m_bdecode_next tags N.)
  _perf.set_value(static_cast<std::uint64_t>(prod));
  if (!ok) { return false; }
  if (prod > 0) {
    ctx._last_predicted = last_tok;
    // Each appended token advances the mROPE position by one (no-op when the
    // context is on the sequential text path, _rope_next_position < 0).
    if (ctx._rope_next_position >= 0) { ctx._rope_next_position += prod; }
  }
  if (produced) { *produced = prod; }
  if (hit_stop) { *hit_stop = hit; }
  return true;
}


// The profile API is only meaningful for LlamaModelExec (the
// per-stage tick instrumentation lives on the concrete class). For
// other architectures the toggles are no-ops and profile_totals
// returns zeros. The interface stays public so callers don't need
// to know which exec is loaded.
void
LoadedLanguageModel::set_profile(bool b)
{
  if (!_impl) { return; }
  (void)b;   // per-stage profiling is MLX-exec-only
}

LoadedLanguageModel::ProfileTotals
LoadedLanguageModel::profile_totals() const
{
  if (!_impl) { return ProfileTotals{}; }
  return ProfileTotals{};
}

void
LoadedLanguageModel::profile_reset()
{
  if (!_impl) { return; }
}

bool
build_mrope_position_ids(
    span<const TokenRef>                     refs,
    span<const std::pair<int, int>>          image_grids,
    std::vector<int32_t>*                    out_pos_t,
    std::vector<int32_t>*                    out_pos_h,
    std::vector<int32_t>*                    out_pos_w,
    int*                                     out_rope_next_position)
{
  if (!out_pos_t || !out_pos_h || !out_pos_w) { return false; }
  const int N = static_cast<int>(refs.size());
  out_pos_t->assign(N, 0);
  out_pos_h->assign(N, 0);
  out_pos_w->assign(N, 0);

  int cur = 0;
  std::size_t img_idx = 0;
  int img_off = 0;       // offset within current image
  int img_base = 0;      // cur at start of current image
  int img_local_max = 0; // running max of (t, h, w) intra-image
  bool in_image = false;

  auto close_image_if_open = [&]() {
    if (in_image) {
      cur = img_base + img_local_max + 1;
      in_image = false;
    }
  };

  for (int i = 0; i < N; ++i) {
    if (refs[i].kind == TokenRef::Kind::Text) {
      close_image_if_open();
      (*out_pos_t)[i] = cur;
      (*out_pos_h)[i] = cur;
      (*out_pos_w)[i] = cur;
      ++cur;
    } else {
      // ImageTokens. If the previous image just finished (img_off
      // reached its size) we close it here so the new image gets a
      // fresh base. This handles the back-to-back-image case where no
      // text ref sits between two images.
      if (in_image && img_off == 0) {
        // Already opened this iteration (no-op).
      }
      if (!in_image) {
        img_base      = cur;
        img_off       = 0;
        img_local_max = 0;
        in_image      = true;
      }
      if (img_idx >= image_grids.size()) {
        return false;
      }
      const int mh = image_grids[img_idx].first;
      const int mw = image_grids[img_idx].second;
      if (mh <= 0 || mw <= 0) {
        return false;
      }
      const int t_idx = 0;
      const int h_idx = img_off / mw;
      const int w_idx = img_off % mw;
      (*out_pos_t)[i] = img_base + t_idx;
      (*out_pos_h)[i] = img_base + h_idx;
      (*out_pos_w)[i] = img_base + w_idx;
      const int local_max = std::max({t_idx, h_idx, w_idx});
      if (local_max > img_local_max) {
        img_local_max = local_max;
      }
      ++img_off;
      if (img_off >= mh * mw) {
        ++img_idx;
        // Close this image immediately so a subsequent ImageTokens
        // ref opens a fresh base for the next image. A subsequent
        // Text ref will also re-close (no-op).
        close_image_if_open();
      }
    }
  }
  close_image_if_open();
  // Confirm we consumed exactly the supplied image_grids.
  if (img_idx != image_grids.size()) {
    return false;
  }
  if (out_rope_next_position) {
    *out_rope_next_position = cur;
  }
  return true;
}

}
