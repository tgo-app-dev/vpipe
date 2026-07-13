#include "stages/text-to-speech-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/ffmpeg-libraries.h"
#include "generative-models/moss/metal-moss-tts-model.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/moss/metal-moss-codec-v2.h"
#include "generative-models/moss/metal-moss-v15-model.h"
#include "generative-models/moss/moss-v15-processor.h"
#include "generative-models/moss/metal-moss-rt-model.h"
#include "generative-models/moss/moss-rt-processor.h"
#include "generative-models/tokenizer.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <utility>
#endif

#include <cstdint>
#include <string>
#include <vector>

using namespace std;

namespace vpipe {

TextToSpeechStage::TextToSpeechStage(const SessionContextIntf* s,
                                     string                    id,
                                     vector<InEdge>            iports,
                                     FlexData                  config)
  : TypedStage<TextToSpeechStage>(s, std::move(id), std::move(iports),
                                  std::move(config))
{
  // Construction must succeed for any config (see Stage::fail_config):
  // a stage must construct so a graph can be built/edited before
  // required fields are supplied. Config problems are recorded via
  // fail_config (first message wins) and deferred to launch. Scalar
  // attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  _hf_dir    = attr_str("hf_dir");
  _codec_dir = attr_str("codec_dir");
  _models_db = attr_str("models_db");
  _codec_int8 = (attr_str("codec_quant") == "int8");
  _instruction = attr_str("instruction");
  _language    = attr_str("language");
  if (_models_db.empty()) {
    _models_db = "models";
  }
  {
    int64_t v = attr_int("max_new_tokens");
    if (v < 1) {
      fail_config(fmt(
          "TextToSpeechStage('{}'): max_new_tokens must be >= 1 "
          "(got {})", this->id(), v));
    }
    _max_new_tokens = static_cast<int>(v);
  }
  {
    int64_t v = attr_int("stream_chunk_frames");
    _stream_chunk = v > 0 ? static_cast<int>(v) : 0;
  }
  _interrupt_on_new_text = attr_bool("interrupt_on_new_text");
  {
    int64_t v = attr_int("max_frames");
    if (v < 1) {
      fail_config(fmt(
          "TextToSpeechStage('{}'): max_frames must be >= 1 (got {})",
          this->id(), v));
    }
    _max_frames = static_cast<int>(v);
  }
  _audio_temp  = attr_real("audio_temperature");
  _audio_top_p = attr_real("audio_top_p");
  _audio_top_k = static_cast<int>(attr_int("audio_top_k"));
  _audio_rep   = attr_real("audio_repetition_penalty");
  _text_temp   = attr_real("text_temperature");
  _text_top_p  = attr_real("text_top_p");
  _text_top_k  = static_cast<int>(attr_int("text_top_k"));
  _text_rep    = attr_real("text_repetition_penalty");
  _sampler_seed = static_cast<std::uint64_t>(attr_uint("sampler_seed"));
  _voice_lock        = attr_bool("voice_lock");
  _voice_ref_seconds = attr_real("voice_ref_seconds");
  // A 2nd (PCM reference) iport means the codec needs its encode path, so a
  // reference voice can be analysed for cloning. iport_edges() reflects the
  // edges this stage was constructed with (the graph is frozen post-launch).
  _with_encoder = this->iport_edges().size() >= 2;

  if (_hf_dir.empty()) {
    fail_config(fmt(
        "TextToSpeechStage('{}'): config.hf_dir is required (non-empty "
        "string -- the MOSS-TTS LM directory)", this->id()));
  }
  if (_codec_dir.empty()) {
    fail_config(fmt(
        "TextToSpeechStage('{}'): config.codec_dir is required (non-empty "
        "string -- the MOSS-Audio-Tokenizer directory)", this->id()));
  }

  // Always allocate the PCM oport. When nothing is wired downstream the
  // runtime allocates an OportBuffer with no cursors and writes are
  // silently dropped, so a sink-style pipeline still works.
  allocate_oports(spec().oports.size());
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "MOSS-TTS LM model: a models-DB key (registered by model-fetch / "
          "model-quantize) or an HF-style model dir; a DB key wins over a "
          "same-named path. 8B (moss-tts), v1.5 (moss-tts-local), or realtime "
          "(moss-tts-realtime).",
   .suggest_db = "models",
   .suggest_db_type = "moss-tts,moss-tts-local,moss-tts-realtime"},
  {.key = "codec_dir", .type = ConfigType::String, .required = true,
   .doc = "MOSS-Audio-Tokenizer (codec) model: a models-DB key or a "
          "filesystem path; a DB key wins over a same-named path. Match the "
          "LM variant: moss-codec (8B) or moss-codec-v2 (v1.5).",
   .suggest_db = "models", .suggest_db_type = "moss-codec,moss-codec-v2"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db model-fetch registers into", .def_str = "models"},
  {.key = "codec_quant", .type = ConfigType::String,
   .doc = "codec weight precision: \"int8\" stores the codec's transformer "
          "GEMM weights as int8 group-32 affine (~half the resident codec "
          "footprint, small audio-quality cost); default/empty = f16",
   .def_str = ""},
  {.key = "max_new_tokens", .type = ConfigType::Int,
   .doc = "8B-only: per-beat delay-pattern generation budget (>= 1)",
   .def_int = 1024},
  // Streaming: emit PCM incrementally as the LM generates instead of decoding
  // the whole utterance at the end. The codec streams its decode with a
  // windowed K/V cache, so a chunk of `stream_chunk_frames` codec frames is
  // decoded + emitted every that-many frames (first audio out ~one chunk after
  // audio starts). 0 => legacy one-shot (single PCM beat per text beat).
  {.key = "stream_chunk_frames", .type = ConfigType::Int,
   .doc = "emit PCM every N generated codec frames (near-realtime streaming); "
          "0 = one-shot (single beat). Each frame ~= 80 ms of audio.",
   .def_int = 16},
  // Barge-in: when a NEW text beat arrives on the text iport while the current
  // utterance is still generating, cut the current one short (emit the audio
  // produced so far, then stop) and move on to the new text, rather than
  // finishing the whole current utterance first. Default on (interactive TTS
  // wants the latest text to win); set false to always generate each text beat
  // to completion (queued FIFO).
  {.key = "interrupt_on_new_text", .type = ConfigType::Bool,
   .doc = "abort the in-flight utterance (after flushing what it produced) as "
          "soon as new text arrives, then speak the new text; false = finish "
          "each utterance fully before the next",
   .def_bool = true},
  // v1.5-only (model_type moss_tts_local): per-beat frame budget + prompt
  // fields. Ignored for the 8B variant.
  {.key = "max_frames", .type = ConfigType::Int,
   .doc = "v1.5-only: per-beat frame budget (each ~= 80 ms @ 48 kHz); >= 1",
   .def_int = 1000},
  {.key = "instruction", .type = ConfigType::String,
   .doc = "v1.5-only: optional style instruction (prompt field)",
   .def_str = "None"},
  {.key = "language", .type = ConfigType::String,
   .doc = "v1.5-only: optional language tag (prompt field)",
   .def_str = "None"},
  // Sampling (flattened, separate audio + text channels). Defaults are the
  // MossTTSDelay-8B recommendations; greedy decoding (temperature <= 0)
  // degenerates (silent loops to max_new_tokens), so sampling is the default.
  {.key = "audio_temperature", .type = ConfigType::Real,
   .doc = "audio-code softmax temperature; <= 0 forces greedy", .def_real = 1.7},
  {.key = "audio_top_p", .type = ConfigType::Real,
   .doc = "audio-code nucleus top-p (1.0 = off)", .def_real = 0.8},
  {.key = "audio_top_k", .type = ConfigType::Int,
   .doc = "audio-code top-k (0 = off)", .def_int = 25},
  {.key = "audio_repetition_penalty", .type = ConfigType::Real,
   .doc = "audio-code repetition penalty (1.0 = off)", .def_real = 1.0},
  {.key = "text_temperature", .type = ConfigType::Real,
   .doc = "free-text-token temperature; <= 0 forces greedy. Default greedy: "
          "vpipe GENERATES the text channel (re-emits the transcript), so it "
          "must follow it to reach audio_end; sampling it over-generates. The "
          "reference's text temp 1.5 is for its teacher-forced text path.",
   .def_real = 0.0},
  {.key = "text_top_p", .type = ConfigType::Real,
   .doc = "free-text-token nucleus top-p (1.0 = off)", .def_real = 1.0},
  {.key = "text_top_k", .type = ConfigType::Int,
   .doc = "free-text-token top-k (0 = off)", .def_int = 50},
  {.key = "text_repetition_penalty", .type = ConfigType::Real,
   .doc = "free-text-token repetition penalty (1.0 = off)", .def_real = 1.0},
  {.key = "sampler_seed", .type = ConfigType::Uint,
   .doc = "audio-sampling RNG seed = the 'voice lock'. Nonzero (default) makes "
          "a given text deterministic -> the SAME voice every run (per-script: "
          "different texts may still differ; for a cross-text identity use "
          "voice_lock or voice cloning via the audio iport). 0 = fresh voice.",
   .def_uint = 0x6d6f7373},   // 'moss'
  {.key = "voice_lock", .type = ConfigType::Bool,
   .doc = "design-once: cache the FIRST generated voice and reuse it as the "
          "clone reference for every later beat, so the timbre stays the same "
          "across different texts (the first voice is picked by sampler_seed). "
          "A reference on the audio iport overrides it. Needs no audio input.",
   .def_bool = false},
  {.key = "voice_ref_seconds", .type = ConfigType::Real,
   .doc = "max seconds of reference audio kept for cloning (longer = more "
          "prompt cost per beat); applies to both the iport reference and "
          "voice_lock. <= 0 keeps the whole clip.",
   .def_real = 12.0},
};
const PortSpec kIports[] = {
  {.name = "text",
   .doc = "FlexData string (or object with a \"text\" key): the text to "
          "synthesize",
   .type = &typeid(FlexDataPayload),
   .tags = "text", .clock_group = 0},
  {.name = "audio-ref",
   .doc = "OPTIONAL mono f32 PCM TensorBeat (any rate; sideband.sample_rate "
          "honoured): a reference voice to clone. Latest beat is sticky "
          "(applies to all later utterances until superseded). Its own clock "
          "group -- it arrives independently of the text stream / PCM output, "
          "not rate-locked to them.",
   .type = &typeid(TensorBeatPayload), .clock_group = 1},
};
const PortSpec kOports[] = {
  {.name = "pcm",
   .doc = "TensorBeat f32 PCM (sideband.sample_rate set); downstream "
          "optional. 8B variant: rank-1 [n_samples] mono @ 24 kHz. v1.5 "
          "variant: rank-2 [2, n_samples] stereo @ 48 kHz.",
   .type = &typeid(TensorBeatPayload),
   .tags = "pcm-samples", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "text-to-speech",
  .doc       = "Text-to-speech (MOSS-TTS, metal): synthesizes each input "
               "text beat into a PCM waveform and emits it as a TensorBeat. "
               "The variant is chosen from the LM dir's config.json "
               "model_type: \"moss_tts\" (8B delay-pattern -> 24 kHz mono) "
               "or \"moss_tts_local\" (v1.5 depth decoder -> 48 kHz stereo).",
  .display_name = "Speak",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
TextToSpeechStage::spec() const noexcept
{
  return kSpec;
}

TextToSpeechStage::~TextToSpeechStage() = default;

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

// The LM dir's config.json top-level "model_type" ("moss_tts_local" => v1.5,
// else the 8B path). Empty on a missing/unparsable config.
std::string
read_model_type_(const std::string& dir)
{
  std::ifstream in(dir + "/config.json");
  if (!in) { return {}; }
  try {
    FlexData root = FlexData::from_json(in);
    if (root.is_object()) {
      auto o = root.as_object();
      if (o.contains("model_type")) {
        return std::string(o.at("model_type").as_string(""));
      }
    }
  } catch (...) {}
  return {};
}

// Build the v1.5 backbone MetalQwenModel::Config from the quantized LM dir's
// config.json: the affine bit-width from top-level quantization.bits (default
// 8), the Qwen3 dims from the nested qwen3_config (defaults = the known v1.5
// shape). transformer.* naming, dense full-attn, bf16 compute.
genai::MetalQwenModel::Config
v15_backbone_cfg_(const std::string& dir)
{
  genai::MetalQwenModel::Config c;
  c.n_layers = 36; c.hidden = 2560; c.n_heads = 32; c.n_kv_heads = 8;
  c.head_dim = 128; c.ffn_inner = 9728; c.vocab = 151936;
  c.rope_theta = 1.0e6f; c.rms_eps = 1e-6f; c.rotary_dim = 128;
  c.full_attn_interval = 1; c.tie_embeddings = false; c.use_bf16 = true;
  c.quant_bits = 8; c.dense = true; c.attn_output_gate = false;
  c.backbone_only = true; c.weight_prefix = "transformer."; c.model_seg = "";
  c.max_seq = 2048; c.page_tokens = 256;
  std::ifstream in(dir + "/config.json");
  if (in) {
    try {
      FlexData root = FlexData::from_json(in);
      if (root.is_object()) {
        auto o = root.as_object();
        if (o.contains("quantization")) {
          FlexData q = o.at("quantization");
          if (q.is_object() && q.as_object().contains("bits")) {
            c.quant_bits = (int)q.as_object().at("bits").as_int(c.quant_bits);
          }
        }
        if (o.contains("qwen3_config")) {
          FlexData qc = o.at("qwen3_config");
          if (qc.is_object()) {
            auto q = qc.as_object();
            auto gi = [&](const char* k, int d) {
              return q.contains(k) ? (int)q.at(k).as_int(d) : d;
            };
            c.n_layers   = gi("num_hidden_layers", c.n_layers);
            c.hidden     = gi("hidden_size", c.hidden);
            c.n_heads    = gi("num_attention_heads", c.n_heads);
            c.n_kv_heads = gi("num_key_value_heads", c.n_kv_heads);
            c.head_dim   = gi("head_dim", c.head_dim);
            c.ffn_inner  = gi("intermediate_size", c.ffn_inner);
            c.vocab      = gi("vocab_size", c.vocab);
            if (q.contains("rope_theta")) {
              c.rope_theta = (float)q.at("rope_theta").as_real(c.rope_theta);
            }
            if (q.contains("rms_norm_eps")) {
              c.rms_eps = (float)q.at("rms_norm_eps").as_real(c.rms_eps);
            }
            c.rotary_dim = c.head_dim;
          }
        }
      }
    } catch (...) {}
  }
  return c;
}

// Build a MetalQwenModel::Config for a MOSS-TTS-Realtime sub-transformer. Both
// the backbone (28 layers) and the depth/local decoder (4 layers) are PLAIN
// dense Qwen3 (RMSNorm, GQA 16/8, head_dim 128, SwiGLU 6144, rope 1e6), reused
// via MetalQwenModel in backbone_only mode, 8-bit affine. `prefix`/`seg` select
// the weight subtree ("language_model."/"model." vs "local_transformer."/
// "model."); `n_layers`/`max_seq` differ (backbone streams the prompt; the
// local decoder is 16 positions/frame).
genai::MetalQwenModel::Config
rt_qwen_cfg_(const std::string& dir, const std::string& prefix,
             const std::string& seg, int n_layers, int max_seq, int page_tokens)
{
  genai::MetalQwenModel::Config c;
  c.n_layers = n_layers; c.hidden = 2048; c.n_heads = 16; c.n_kv_heads = 8;
  c.head_dim = 128; c.ffn_inner = 6144; c.vocab = 151936;
  c.rope_theta = 1.0e6f; c.rms_eps = 1e-6f; c.rotary_dim = 128;
  c.full_attn_interval = 1; c.tie_embeddings = false; c.use_bf16 = true;
  c.quant_bits = 8; c.dense = true; c.attn_output_gate = false;
  // Plain Qwen3 standard RMSNorm (no +1). Only consulted on the dense (raw
  // bf16) path -- MetalQwenModel auto-detects raw vs 8-bit affine from the dir,
  // so ONE config loads either an unquantized bf16 checkpoint OR a
  // model-quantize'd 8-bit one. (No effect on the quantized path.)
  c.zero_centered_norm = false;
  c.backbone_only = true; c.weight_prefix = prefix; c.model_seg = seg;
  c.max_seq = max_seq; c.page_tokens = page_tokens;
  std::ifstream in(dir + "/config.json");
  if (in) {
    try {
      FlexData root = FlexData::from_json(in);
      if (root.is_object()) {
        auto o = root.as_object();
        if (o.contains("quantization")) {
          FlexData q = o.at("quantization");
          if (q.is_object() && q.as_object().contains("bits")) {
            c.quant_bits = (int)q.as_object().at("bits").as_int(c.quant_bits);
          }
        }
        // language_config carries the backbone shape (num_hidden_layers etc).
        if (seg == "model." && prefix == "language_model." &&
            o.contains("language_config")) {
          FlexData lc = o.at("language_config");
          if (lc.is_object()) {
            auto q = lc.as_object();
            auto gi = [&](const char* k, int d) {
              return q.contains(k) ? (int)q.at(k).as_int(d) : d;
            };
            c.n_layers   = gi("num_hidden_layers", c.n_layers);
            c.hidden     = gi("hidden_size", c.hidden);
            c.n_heads    = gi("num_attention_heads", c.n_heads);
            c.n_kv_heads = gi("num_key_value_heads", c.n_kv_heads);
            c.head_dim   = gi("head_dim", c.head_dim);
            c.ffn_inner  = gi("intermediate_size", c.ffn_inner);
            c.vocab      = gi("vocab_size", c.vocab);
            if (q.contains("rope_theta")) {
              c.rope_theta = (float)q.at("rope_theta").as_real(c.rope_theta);
            }
            c.rotary_dim = c.head_dim;
          }
        }
      }
    } catch (...) {}
  }
  return c;
}

}  // namespace
#endif  // VPIPE_BUILD_APPLE_SILICON

Job
TextToSpeechStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_hf_dir.empty() || _codec_dir.empty()) {
    co_return;  // ctor already recorded the config error.
  }
  auto* mc = session() ? session()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "TextToSpeechStage('{}'): no metal-compute backend on this "
        "session (Metal unavailable, or the build lacks "
        "VPIPE_BUILD_APPLE_SILICON); the stage is inert", this->id()));
    co_return;
  }
  const std::string lm_dir =
      resolve_model_dir(session(), _models_db, _hf_dir);
  const std::string codec_dir =
      resolve_model_dir(session(), _models_db, _codec_dir);

  // Pick the variant from the LM dir's config.json. "moss_tts_realtime" => the
  // realtime streaming path (Qwen3-1.7B backbone + depth decoder -> 24 kHz
  // mono); "moss_tts_local" => the v1.5 depth-decoder path (48 kHz stereo);
  // anything else => the 8B delay-pattern path (24 kHz mono).
  const std::string _mt = read_model_type_(lm_dir);
  if (_mt == "moss_tts_realtime") {
    session()->info(fmt(
        "TextToSpeechStage('{}'): loading MOSS-TTS-Realtime LM from '{}'",
        this->id(), _hf_dir));
    genai::MetalMossRtModel::Config cfg;
    cfg.backbone = rt_qwen_cfg_(lm_dir, "language_model.", "model.", 28, 2048,
                                256);
    cfg.local    = rt_qwen_cfg_(lm_dir, "local_transformer.", "model.", 4, 32,
                                32);
    _lm_rt = genai::MetalMossRtModel::load(lm_dir, mc, cfg);
    if (!_lm_rt) {
      session()->error(fmt(
          "TextToSpeechStage('{}'): failed to load MOSS-TTS-Realtime LM from "
          "'{}'; inert. Point hf_dir at the unquantized bf16 checkpoint (runs "
          "as-is) OR a model-quantize'd 8-bit dir (~2x faster).",
          this->id(), _hf_dir));
      co_return;
    }
    session()->info(fmt(
        "TextToSpeechStage('{}'): loading MOSS-Audio-Tokenizer codec from "
        "'{}'{}", this->id(), _codec_dir,
        _with_encoder ? " (with encoder: voice cloning enabled)" : ""));
    _codec = genai::MetalMossCodec::load(codec_dir, mc, _codec_int8,
                                         _with_encoder);
    if (!_codec || !_codec->valid()) {
      session()->error(fmt(
          "TextToSpeechStage('{}'): failed to load MOSS codec from '{}'; inert",
          this->id(), _codec_dir));
      _lm_rt.reset();
      co_return;
    }
    // Route the codec's audio-codec-lane profiler events to this session (the
    // MetalMossRtModel already took mc->session() at load). Without this the
    // codec-decode events never fire.
    _codec->set_session(session());
    _tokenizer = genai::Tokenizer::from_huggingface_json(
        lm_dir + "/tokenizer.json", session());
    if (!_tokenizer) {
      session()->error(fmt(
          "TextToSpeechStage('{}'): no tokenizer.json in '{}'; inert",
          this->id(), lm_dir));
      _lm_rt.reset(); _codec.reset();
      co_return;
    }
    session()->info(fmt(
        "TextToSpeechStage('{}'): ready (realtime, 24 kHz mono, "
        "n_vq={}, max_frames={})", this->id(), _lm_rt->config().n_vq,
        _max_frames));
    co_return;
  }
  if (_mt == "moss_tts_local") {
    session()->info(fmt(
        "TextToSpeechStage('{}'): loading MOSS-TTS-Local-v1.5 LM from '{}'",
        this->id(), _hf_dir));
    genai::MetalMossV15Model::Config cfg;
    cfg.backbone = v15_backbone_cfg_(lm_dir);
    _lm_v15 = genai::MetalMossV15Model::load(lm_dir, mc, cfg);
    if (!_lm_v15) {
      session()->error(fmt(
          "TextToSpeechStage('{}'): failed to load v1.5 LM from '{}' "
          "(quantize the bf16 source with model-quantize first); inert",
          this->id(), _hf_dir));
      co_return;
    }
    session()->info(fmt(
        "TextToSpeechStage('{}'): loading codec-v2 from '{}'{}", this->id(),
        _codec_dir,
        _with_encoder ? " (with encoder: voice cloning enabled)" : ""));
    _codec_v15 = genai::MetalMossCodecV2::load(codec_dir, mc, _with_encoder);
    if (!_codec_v15 || !_codec_v15->valid()) {
      session()->error(fmt(
          "TextToSpeechStage('{}'): failed to load codec-v2 from '{}'; inert",
          this->id(), _codec_dir));
      _lm_v15.reset();
      co_return;
    }
    _tokenizer = genai::Tokenizer::from_huggingface_json(
        lm_dir + "/tokenizer.json", session());
    if (!_tokenizer) {
      session()->error(fmt(
          "TextToSpeechStage('{}'): no tokenizer.json in '{}'; inert",
          this->id(), lm_dir));
      _lm_v15.reset(); _codec_v15.reset();
      co_return;
    }
    session()->info(fmt(
        "TextToSpeechStage('{}'): ready (v1.5, 48 kHz stereo, max_frames={})",
        this->id(), _max_frames));
    co_return;
  }

  using clock = std::chrono::steady_clock;
  auto ms = [](clock::duration d) {
    return static_cast<int>(
        std::chrono::duration<double, std::milli>(d).count());
  };

  session()->info(fmt(
      "TextToSpeechStage('{}'): loading MOSS-TTS LM from '{}'",
      this->id(), _hf_dir));
  const auto t_lm0 = clock::now();
  _lm = genai::MetalMossTtsModel::load(lm_dir, mc);
  if (!_lm || !_lm->valid()) {
    session()->error(fmt(
        "TextToSpeechStage('{}'): failed to load MOSS-TTS LM from '{}'; "
        "the stage is inert", this->id(), _hf_dir));
    _lm.reset();
    co_return;
  }
  session()->info(fmt(
      "TextToSpeechStage('{}'): MOSS-TTS LM loaded in {} ms",
      this->id(), ms(clock::now() - t_lm0)));

  session()->info(fmt(
      "TextToSpeechStage('{}'): loading MOSS-Audio-Tokenizer codec from "
      "'{}'{}", this->id(), _codec_dir,
      _with_encoder ? " (with encoder: voice cloning enabled)" : ""));
  const auto t_cc0 = clock::now();
  _codec = genai::MetalMossCodec::load(codec_dir, mc, _codec_int8,
                                       _with_encoder);
  if (!_codec || !_codec->valid()) {
    session()->error(fmt(
        "TextToSpeechStage('{}'): failed to load MOSS codec from '{}'; "
        "the stage is inert", this->id(), _codec_dir));
    _lm.reset();
    _codec.reset();
    co_return;
  }
  session()->info(fmt(
      "TextToSpeechStage('{}'): MOSS codec loaded in {} ms",
      this->id(), ms(clock::now() - t_cc0)));

  // Tokenizer for the LM prompt. The MOSS user_inst prompt encodes the
  // <|im_start|> / <|im_end|> markers as single special-token ids.
  _tokenizer = genai::Tokenizer::from_huggingface_json(
      lm_dir + "/tokenizer.json", session());
  if (!_tokenizer) {
    session()->error(fmt(
        "TextToSpeechStage('{}'): failed to load tokenizer from "
        "'{}/tokenizer.json'; the stage is inert",
        this->id(), lm_dir));
    _lm.reset();
    _codec.reset();
    co_return;
  }

  // Route profiling events (text-prefill / text-decode / audio-codec) onto
  // the session's LLM perf lane.
  _lm->set_session(session());
  _codec->set_session(session());

  // Cold-start warmup: the FIRST forward pass pays the Metal pipeline-state
  // compilation + first-touch weight residency (tens of seconds on the 8B LM)
  // with the GPU mostly idle. Run a tiny throwaway LM generation + codec decode
  // HERE, at load, so that cost lands during stage init -- the first real
  // synthesis then runs warm. Disable with VPIPE_TTS_NO_WARMUP for A/B.
  if (std::getenv("VPIPE_TTS_NO_WARMUP") == nullptr) {
    const auto t_w0 = clock::now();
    const int n_vq = _lm->config().n_vq;
    const int pad  = _lm->config().audio_pad_code;
    // A few rows: channel 0 a valid text id, audio channels at pad. The exact
    // ids are irrelevant -- this only compiles kernels + makes weights
    // resident; the generated output is discarded.
    std::vector<std::vector<std::int32_t>> wprompt(
        4, std::vector<std::int32_t>(static_cast<std::size_t>(1 + n_vq), pad));
    wprompt[0][0] = _lm->config().im_start;
    wprompt[1][0] = _lm->config().pad_token;
    wprompt[2][0] = _lm->config().pad_token;
    wprompt[3][0] = _lm->config().pad_token;
    (void)_lm->generate_delay_greedy(wprompt, 8);
    // Codec: a few zero-code frames warm the RVQ decode + 4 transformer stages.
    std::vector<std::vector<std::int32_t>> wcodes(
        4, std::vector<std::int32_t>(static_cast<std::size_t>(n_vq), 0));
    (void)_codec->decode(wcodes, nullptr);
    session()->info(fmt(
        "TextToSpeechStage('{}'): warmup done in {} ms (cold pipeline-state "
        "+ weight residency paid at load; first synthesis runs warm)",
        this->id(), ms(clock::now() - t_w0)));
  }

  session()->info(fmt(
      "TextToSpeechStage('{}'): ready (n_vq={}, sample_rate={}, "
      "max_new_tokens={})",
      this->id(), _lm->config().n_vq, _codec->sample_rate(),
      _max_new_tokens));
#else
  session()->error(fmt(
      "TextToSpeechStage('{}'): this build was compiled without "
      "VPIPE_BUILD_APPLE_SILICON; the MOSS-TTS subsystem is unavailable",
      this->id()));
#endif
  co_return;
}

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

// Minimal text normalization (full unicode \p{L} regex normalization is
// a TODO -- std::regex can't do \p{L}). Replace CR/CRLF with a space,
// drop ASCII control chars except spaces, collapse whitespace runs to a
// single space, and trim leading/trailing whitespace.
std::string
normalize_text_(const std::string& in)
{
  std::string spaced;
  spaced.reserve(in.size());
  for (char c : in) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (c == '\r' || c == '\n') {
      spaced.push_back(' ');
    } else if (uc < 0x20 || uc == 0x7f) {
      // Drop ASCII control chars (keep nothing); spaces are >= 0x20.
      continue;
    } else {
      spaced.push_back(c);
    }
  }
  // Collapse whitespace runs + trim.
  std::string out;
  out.reserve(spaced.size());
  bool in_ws = false;
  for (char c : spaced) {
    const bool is_ws =
        std::isspace(static_cast<unsigned char>(c)) != 0;
    if (is_ws) {
      in_ws = true;
      continue;
    }
    if (in_ws && !out.empty()) {
      out.push_back(' ');
    }
    in_ws = false;
    out.push_back(c);
  }
  return out;
}

// Render the MOSS user_inst prompt EXACTLY (the reference processor
// format), substituting `text` for {TEXT}. There is a trailing newline
// after "assistant".
std::string
render_moss_prompt_(const std::string& text)
{
  std::string p;
  p += "<|im_start|>user\n";
  p += "<user_inst>\n";
  p += "- Reference(s):\n";
  p += "None\n";
  p += "- Instruction:\n";
  p += "None\n";
  p += "- Tokens:\n";
  p += "None\n";
  p += "- Quality:\n";
  p += "None\n";
  p += "- Sound Event:\n";
  p += "None\n";
  p += "- Ambient Sound:\n";
  p += "None\n";
  p += "- Language:\n";
  p += "None\n";
  p += "- Text:\n";
  p += text;
  p += "\n";
  p += "</user_inst><|im_end|>\n";
  p += "<|im_start|>assistant\n";
  return p;
}

// Encode the rendered prompt to ids, treating the <|im_start|> / <|im_end|>
// markers as SINGLE special-token ids. vpipe's Tokenizer::encode() does NOT
// recognise special tokens in input text (it would byte-level-BPE the literal
// "<|im_start|>" into ~6 tokens), so we split on the special-token literals,
// BPE the text between, and inject special_token_id() for each marker.
std::vector<std::int32_t>
encode_moss_prompt_(const genai::Tokenizer& tok, const std::string& s)
{
  static const char* const kSpecials[] = {"<|im_start|>", "<|im_end|>"};
  std::vector<std::int32_t> ids;
  std::size_t pos = 0;
  while (pos < s.size()) {
    std::size_t best = std::string::npos;
    std::size_t best_len = 0;
    std::int32_t best_id = -1;
    for (const char* sp : kSpecials) {
      const std::size_t f = s.find(sp, pos);
      if (f != std::string::npos &&
          (best == std::string::npos || f < best)) {
        best = f;
        best_len = std::char_traits<char>::length(sp);
        best_id = tok.special_token_id(sp);
      }
    }
    const std::size_t end = (best == std::string::npos) ? s.size() : best;
    if (end > pos) {
      const std::vector<std::int32_t> seg = tok.encode(s.substr(pos, end - pos));
      ids.insert(ids.end(), seg.begin(), seg.end());
    }
    if (best == std::string::npos) { break; }
    if (best_id >= 0) { ids.push_back(best_id); }
    pos = best + best_len;
  }
  return ids;
}

// Build the MOSS prompt GRID [seq][1 + n_vq]: channel 0 = text/control id,
// channels 1..n_vq = audio codes (audio_pad_code where inactive). When `ref`
// is non-null/non-empty, splice it into the - Reference(s): section as a
// single-speaker USER audio block, exactly as the MOSS processor does: the
// channel-0 span is audio_start, then audio_user_slot for every (delayed) row,
// then audio_end; channels 1..n_vq carry the reference codes with the delay
// pattern applied (delayed[r][cb] = ref[r-cb][cb], else pad). Without a
// reference this is the plain (- Reference(s): None) prompt.
std::vector<std::vector<std::int32_t>>
build_moss_grid_(const genai::Tokenizer&                       tok,
                 const std::string&                            text,
                 const genai::MetalMossTtsModel::Config&       cfg,
                 const std::vector<std::vector<std::int32_t>>* ref)
{
  const int n_vq  = cfg.n_vq;
  const int pad   = cfg.audio_pad_code;
  const int nchan = 1 + n_vq;
  std::vector<std::vector<std::int32_t>> grid;
  auto text_row = [&](std::int32_t id) {
    std::vector<std::int32_t> r(static_cast<std::size_t>(nchan), pad);
    r[0] = id;
    return r;
  };
  auto push_text = [&](const std::string& s) {
    for (std::int32_t id : encode_moss_prompt_(tok, s)) {
      grid.push_back(text_row(id));
    }
  };

  if (ref == nullptr || ref->empty()) {
    push_text(render_moss_prompt_(text));
    return grid;
  }

  // Reference (S1) audio block, then the remaining user_inst sections + text.
  push_text("<|im_start|>user\n<user_inst>\n- Reference(s):\n[S1]:\n");
  grid.push_back(text_row(cfg.audio_start));
  const int T = static_cast<int>(ref->size());
  for (int r = 0; r < T + n_vq - 1; ++r) {           // apply_delay_pattern
    std::vector<std::int32_t> g(static_cast<std::size_t>(nchan), pad);
    g[0] = cfg.audio_user_slot;
    for (int cb = 0; cb < n_vq; ++cb) {
      const int src = r - cb;                          // delayed[r][cb]
      if (src >= 0 && src < T) {
        int v = (*ref)[static_cast<std::size_t>(src)][static_cast<std::size_t>(cb)];
        if (v < 0) { v = 0; }
        if (v >= pad) { v = pad - 1; }
        g[static_cast<std::size_t>(1 + cb)] = v;
      }
    }
    grid.push_back(std::move(g));
  }
  grid.push_back(text_row(cfg.audio_end));
  push_text("\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n"
            "- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n"
            "- Text:\n" + text + "\n</user_inst><|im_end|>\n"
            "<|im_start|>assistant\n");
  return grid;
}

// Resample a mono f32 clip from in_sr to out_sr via FFmpeg swresample (proper
// anti-aliased conversion). Returns empty on failure (FFmpeg unavailable, bad
// rate); an in_sr == out_sr clip is copied through. One-shot (whole clip +
// flush).
std::vector<float>
resample_pcm_(const SessionContextIntf* session, const float* in,
              std::size_t n_in, int in_sr, int out_sr)
{
  if (in == nullptr || n_in == 0 || in_sr <= 0 || out_sr <= 0) { return {}; }
  if (in_sr == out_sr) { return std::vector<float>(in, in + n_in); }
  const FFmpegLibraries* libs = session ? session->ffmpeg_libraries() : nullptr;
  if (libs == nullptr) { return {}; }
  const auto& swr = libs->swresample().api;
  AVChannelLayout mono_in  = AV_CHANNEL_LAYOUT_MONO;
  AVChannelLayout mono_out = AV_CHANNEL_LAYOUT_MONO;
  SwrContext* sw = nullptr;
  if (swr.alloc_set_opts2(&sw, &mono_out, AV_SAMPLE_FMT_FLT, out_sr,
                          &mono_in, AV_SAMPLE_FMT_FLT, in_sr, 0, nullptr) < 0
      || sw == nullptr) {
    return {};
  }
  if (swr.init(sw) < 0) { swr.free(&sw); return {}; }
  const std::int64_t max_out =
      static_cast<std::int64_t>(n_in) * out_sr / in_sr + 64;
  std::vector<float> out(static_cast<std::size_t>(max_out));
  const std::uint8_t* in_ptr = reinterpret_cast<const std::uint8_t*>(in);
  float* dst = out.data();
  int n_out = swr.convert(sw, reinterpret_cast<std::uint8_t**>(&dst),
                          static_cast<int>(max_out), &in_ptr,
                          static_cast<int>(n_in));
  if (n_out >= 0 && n_out < max_out) {            // flush the resampler tail
    float* dst2 = out.data() + n_out;
    const int n_flush = swr.convert(
        sw, reinterpret_cast<std::uint8_t**>(&dst2),
        static_cast<int>(max_out - n_out), nullptr, 0);
    if (n_flush > 0) { n_out += n_flush; }
  }
  swr.free(&sw);
  if (n_out <= 0) { return {}; }
  out.resize(static_cast<std::size_t>(n_out));
  return out;
}

// Build a `out_sr` channel-major STEREO reference wave ([ch0 N | ch1 N], the
// layout MetalMossCodecV2::encode wants) from a PCM TensorBeat. Accepts rank-1
// mono [N] or rank-2 channel-major [C, N]; resamples each channel to out_sr
// (sideband.sample_rate honoured, default out_sr); a mono clip is duplicated to
// both channels. Caps to `max_seconds` (<= 0 = whole clip). Empty on failure.
std::vector<float>
build_stereo_ref_wave_(const SessionContextIntf* session,
                       const TensorBeatPayload& tbp, int out_sr,
                       double max_seconds)
{
  int in_sr = out_sr;
  if (tbp.sideband.is_object()) {
    auto sb = tbp.sideband.as_object();
    if (sb.contains("sample_rate")) {
      in_sr = static_cast<int>(sb.at("sample_rate").as_int(in_sr));
    }
  }
  int in_ch = 1;
  std::int64_t in_n = static_cast<std::int64_t>(tbp.element_count());
  if (tbp.shape.size() == 2) {
    in_ch = static_cast<int>(tbp.shape[0]);
    in_n  = tbp.shape[1];
  } else if (tbp.shape.size() == 1) {
    in_ch = 1;
    in_n  = tbp.shape[0];
  }
  if (in_ch < 1 || in_n < 1) { return {}; }
  const float* base = tbp.as_f32();
  auto ch_resampled = [&](int c) {
    return resample_pcm_(session, base + static_cast<std::int64_t>(c) * in_n,
                         static_cast<std::size_t>(in_n), in_sr, out_sr);
  };
  std::vector<float> l = ch_resampled(0);
  if (l.empty()) { return {}; }
  std::vector<float> r = (in_ch >= 2) ? ch_resampled(1) : l;
  std::size_t keep = std::min(l.size(), r.size());  // channels equal-length
  if (max_seconds > 0.0) {
    const std::size_t cap =
        static_cast<std::size_t>(max_seconds * out_sr);
    if (cap >= 1 && keep > cap) { keep = cap; }
  }
  std::vector<float> flat;
  flat.reserve(keep * 2);
  flat.insert(flat.end(), l.begin(), l.begin() + keep);
  flat.insert(flat.end(), r.begin(), r.begin() + keep);
  return flat;
}

// A reference-voice PCM TensorBeat -> a mono f32 clip resampled to `out_sr`
// (sideband.sample_rate honoured; multi-channel averaged to mono), capped to
// `max_seconds` (<= 0 = whole clip). For the realtime / 8B 24 kHz mono codec
// encode path. Empty on failure.
std::vector<float>
build_mono_ref_wave_(const SessionContextIntf* session,
                     const TensorBeatPayload& tbp, int out_sr,
                     double max_seconds)
{
  int in_sr = out_sr;
  if (tbp.sideband.is_object()) {
    auto sb = tbp.sideband.as_object();
    if (sb.contains("sample_rate")) {
      in_sr = static_cast<int>(sb.at("sample_rate").as_int(in_sr));
    }
  }
  int in_ch = 1;
  std::int64_t in_n = static_cast<std::int64_t>(tbp.element_count());
  if (tbp.shape.size() == 2) { in_ch = static_cast<int>(tbp.shape[0]);
                               in_n = tbp.shape[1]; }
  else if (tbp.shape.size() == 1) { in_ch = 1; in_n = tbp.shape[0]; }
  if (in_ch < 1 || in_n < 1) { return {}; }
  const float* base = tbp.as_f32();
  std::vector<float> mono((std::size_t)in_n);
  if (in_ch == 1) {
    std::memcpy(mono.data(), base, (std::size_t)in_n * sizeof(float));
  } else {
    for (std::int64_t i = 0; i < in_n; ++i) {
      float s = 0.0f;
      for (int c = 0; c < in_ch; ++c) { s += base[(std::int64_t)c * in_n + i]; }
      mono[(std::size_t)i] = s / (float)in_ch;
    }
  }
  std::vector<float> out =
      resample_pcm_(session, mono.data(), (std::size_t)in_n, in_sr, out_sr);
  if (max_seconds > 0.0) {
    const std::size_t cap = static_cast<std::size_t>(max_seconds * out_sr);
    if (cap >= 1 && out.size() > cap) { out.resize(cap); }
  }
  return out;
}

}  // namespace
#endif  // VPIPE_BUILD_APPLE_SILICON

Job
TextToSpeechStage::process(RuntimeContext& ctx)
{
  auto t = co_await ctx.read(0);
  if (!t) {
    ctx.signal_done();
    co_return;
  }

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Barge-in predicate: true once a NEW text beat is waiting on the text iport
  // (iport 0) while we're mid-utterance. When interrupt_on_new_text is set, the
  // generators poll this and stop ASAP (after flushing the audio produced so
  // far); process() then re-runs and serves the newer text. The audio-ref iport
  // (iport 1) is deliberately excluded -- it is sticky + its own clock domain.
  auto new_text_pending = [&]() {
    return _interrupt_on_new_text && ctx.backlog(0) > 0;
  };

  // ---- realtime variant (Qwen3-1.7B backbone + depth decoder -> 24 kHz mono)
  if (_lm_rt) {
    if (!_codec || !_tokenizer) {
      session()->warn(fmt(
          "TextToSpeechStage('{}'): realtime models not loaded; dropping beat",
          this->id()));
      co_return;
    }
    const auto* rtfdp = dynamic_cast<const FlexDataPayload*>(t.get());
    if (rtfdp == nullptr) {
      session()->warn(fmt(
          "TextToSpeechStage('{}'): expected FlexDataPayload on in-port 0, "
          "got {}; dropping beat", this->id(), t->describe()));
      co_return;
    }
    std::string rtraw(rtfdp->data.as_string(""));
    if (rtraw.empty() && rtfdp->data.is_object()) {
      auto obj = rtfdp->data.as_object();
      if (obj.contains("text")) { rtraw = std::string(obj.at("text").as_string("")); }
    }
    const std::string rt_text = normalize_text_(rtraw);
    if (rt_text.empty()) {
      session()->warn(fmt(
          "TextToSpeechStage('{}'): empty text; dropping beat", this->id()));
      co_return;
    }
    using rtclock = std::chrono::steady_clock;
    const auto rt0 = rtclock::now();

    // Drain reference-audio beats on iport1 (voice cloning). The latest sets
    // the cloned voice for this + subsequent text beats (sticky). Reference
    // PCM is resampled to 24 kHz mono, codec-encoded, and the FIRST n_vq (16)
    // codebooks kept (the realtime model's codebook count). backlog() keeps it
    // non-blocking.
    if (_with_encoder && _codec->has_encoder() && ctx.num_iports() >= 2) {
      while (ctx.backlog(1) > 0) {
        auto rp = co_await ctx.read(1);
        if (!rp) { break; }
        const auto* tbp = dynamic_cast<const TensorBeatPayload*>(rp.get());
        if (tbp == nullptr || tbp->dtype != TensorBeat::DType::F32) {
          session()->warn(fmt(
              "TextToSpeechStage('{}'): reference iport expects an f32 PCM "
              "TensorBeat, got {}; ignoring", this->id(), rp->describe()));
          continue;
        }
        std::vector<float> ref_wave = build_mono_ref_wave_(
            session(), *tbp, _codec->sample_rate(), _voice_ref_seconds);
        if (ref_wave.empty()) { continue; }
        auto rc32 = _codec->encode(ref_wave);   // [Tref][32]
        if (rc32.empty()) { continue; }
        const int nvq = _lm_rt->config().n_vq;
        std::vector<std::vector<std::int32_t>> rc;
        rc.reserve(rc32.size());
        for (auto& fr : rc32) {
          const int m = std::min<int>(nvq, (int)fr.size());
          rc.emplace_back(fr.begin(), fr.begin() + m);   // keep first n_vq
        }
        _ref_codes = std::move(rc);
        _ref_set   = true;
        session()->info(fmt(
            "TextToSpeechStage('{}'): cloned reference voice -> {} codec frames "
            "(24 kHz)", this->id(), static_cast<int>(_ref_codes.size())));
      }
    }

    genai::MossRtPromptIds pids;
    auto prompt_grid = _ref_set
        ? genai::moss_rt_build_clone_grid(*_tokenizer, _ref_codes, pids)
        : genai::moss_rt_build_prompt_grid(*_tokenizer, pids);
    std::vector<std::int32_t> text_ids = _tokenizer->encode(rt_text);
    // Audio codes MUST be sampled (the reference defaults: temp 0.8 / top_p
    // 0.6 / top_k 30 / rep_pen 1.1); greedy degenerates into silence.
    genai::MossSampling rt_sp;
    rt_sp.temperature        = 0.8f;
    rt_sp.top_k              = 30;
    rt_sp.top_p              = 0.6f;
    rt_sp.repetition_penalty = 1.1f;
    const int rt_sr  = _lm_rt->sampling_rate();
    const int rt_nvq = _lm_rt->config().n_vq;   // active codebooks (16)
    const int max_f  = _max_frames > 0 ? _max_frames : 1000;
    // Build a mono PCM TensorBeat [1, samples] from a wav vector.
    auto make_pcm_beat = [&](const std::vector<float>& wav) {
      TensorBeat tb;
      tb.dtype = TensorBeat::DType::F32;
      tb.shape = { 1, static_cast<std::int64_t>(wav.size()) };
      tb.resize_contiguous(wav.size());
      std::memcpy(tb.as_f32(), wav.data(), wav.size() * sizeof(float));
      tb.sideband = FlexData::make_object();
      tb.sideband.as_object().insert("sample_rate", FlexData::make_int(rt_sr));
      return tb;
    };

    std::vector<std::vector<int>> frames;
    std::int64_t total_samps = 0;
    if (_stream_chunk > 0) {
      // STREAMING: decode + emit PCM every _stream_chunk frames via the codec's
      // windowed-KV streaming decode (first audio out ~one chunk after start).
      // Reuse the cached ring state across beats (see the v1.5 path); the guard
      // also re-allocates if the active-codebook count changes.
      if (!_stream_v1 || _stream_v1->max_chunk != _stream_chunk
          || _stream_v1->n_active != rt_nvq) {
        _stream_v1 = _codec->decode_stream_begin(_stream_chunk, rt_nvq);
      } else {
        _stream_v1->reset();
      }
      std::vector<std::vector<std::int32_t>> buf;
      bool open = (_stream_v1 != nullptr);
      auto flush = [&]() {
        if (!open || buf.empty()) { return; }
        std::vector<float> pcm = _codec->decode_stream_chunk(*_stream_v1, buf);
        buf.clear();
        if (pcm.empty()) { return; }
        total_samps += static_cast<std::int64_t>(pcm.size());
        open = ctx.write_sync(
            0, make_payload<TensorBeatPayload>(make_pcm_beat(pcm)));
        if (open) { ++_clips_emitted; }
      };
      auto on_frame = [&](const std::vector<int>& codes) -> bool {
        buf.emplace_back(codes.begin(), codes.end());
        if (static_cast<int>(buf.size()) >= _stream_chunk) { flush(); }
        return open && !new_text_pending();   // barge-in: stop on new text
      };
      frames = _lm_rt->generate(prompt_grid, text_ids, max_f, rt_sp,
                                _sampler_seed, on_frame);
      flush();
    } else {
      frames = _lm_rt->generate(
          prompt_grid, text_ids, max_f, rt_sp, _sampler_seed,
          [&](const std::vector<int>&) { return !new_text_pending(); });
    }
    const auto rt_gen = rtclock::now();
    if (frames.empty()) {
      session()->warn(fmt(
          "TextToSpeechStage('{}'): realtime generated 0 frames; emitting "
          "nothing", this->id()));
      co_return;
    }
    // voice_lock (design-once): cache the FIRST generated voice as the clone
    // reference for later beats (a fixed sampler_seed picks it deterministically).
    // An external iport reference overrides it (_ref_set already true).
    if (_voice_lock && !_ref_set) {
      std::vector<std::vector<std::int32_t>> rc;
      rc.reserve(frames.size());
      for (const auto& f : frames) { rc.emplace_back(f.begin(), f.end()); }
      if (_voice_ref_seconds > 0.0) {
        const std::size_t cap = static_cast<std::size_t>(
            _voice_ref_seconds * 12.5);   // 12.5 Hz frame rate
        if (cap >= 1 && rc.size() > cap) { rc.resize(cap); }
      }
      _ref_codes = std::move(rc);
      _ref_set   = true;
      session()->info(fmt(
          "TextToSpeechStage('{}'): voice_lock engaged -- {} frames cached as "
          "the realtime reference", this->id(),
          static_cast<int>(_ref_codes.size())));
    }
    if (_stream_chunk <= 0) {
      // ONE-SHOT: decode the whole utterance, emit a single PCM beat.
      std::vector<std::vector<std::int32_t>> rt_codes;
      rt_codes.reserve(frames.size());
      for (const auto& f : frames) {
        rt_codes.emplace_back(f.begin(), f.end());
      }
      std::vector<float> rt_wav = _codec->decode(rt_codes, nullptr, rt_nvq);
      if (rt_wav.empty()) {
        session()->warn(fmt(
            "TextToSpeechStage('{}'): codec produced 0 samples", this->id()));
        co_return;
      }
      total_samps = static_cast<std::int64_t>(rt_wav.size());
      ++_clips_emitted;
      co_await ctx.write(
          0, make_payload<TensorBeatPayload>(make_pcm_beat(rt_wav)));
    }
    auto rtms = [](rtclock::duration d) {
      return static_cast<int>(
          std::chrono::duration<double, std::milli>(d).count());
    };
    std::string rt_mode = _stream_chunk > 0
        ? fmt("streamed {}-frame chunks", _stream_chunk)()
        : std::string("one-shot decode");
    session()->info(fmt(
        "TextToSpeechStage('{}'): realtime {} chars -> {} frames -> {} PCM = "
        "{:.2f}s @ {} Hz ({} ms gen, {})",
        this->id(), static_cast<int>(rt_text.size()),
        static_cast<int>(frames.size()), static_cast<int>(total_samps),
        total_samps / static_cast<double>(rt_sr), rt_sr,
        rtms(rt_gen - rt0), rt_mode));
    co_return;
  }

  // ---- v1.5 variant (depth decoder -> codec-v2 -> 48 kHz stereo) ----
  if (_lm_v15) {
    if (!_codec_v15 || !_tokenizer) {
      session()->warn(fmt(
          "TextToSpeechStage('{}'): v1.5 models not loaded; dropping beat",
          this->id()));
      co_return;
    }
    const auto* v15fdp = dynamic_cast<const FlexDataPayload*>(t.get());
    if (v15fdp == nullptr) {
      session()->warn(fmt(
          "TextToSpeechStage('{}'): expected FlexDataPayload on in-port 0, "
          "got {}; dropping beat", this->id(), t->describe()));
      co_return;
    }
    std::string raw(v15fdp->data.as_string(""));
    if (raw.empty() && v15fdp->data.is_object()) {
      auto obj = v15fdp->data.as_object();
      if (obj.contains("text")) {
        raw = std::string(obj.at("text").as_string(""));
      }
    }
    const std::string v15_text = normalize_text_(raw);
    if (v15_text.empty()) {
      session()->warn(fmt(
          "TextToSpeechStage('{}'): empty text; dropping beat", this->id()));
      co_return;
    }
    using v15clock = std::chrono::steady_clock;
    const auto vt0 = v15clock::now();

    // Drain reference-audio beats on iport1 (voice cloning). The latest sets
    // the cloned voice for this and subsequent text beats (sticky). v1.5
    // expects 48 kHz stereo; mono / other rates are resampled + channel-
    // duplicated, capped to voice_ref_seconds. backlog() keeps it non-blocking.
    if (_with_encoder && _codec_v15->has_encoder() && ctx.num_iports() >= 2) {
      while (ctx.backlog(1) > 0) {
        auto rp = co_await ctx.read(1);
        if (!rp) { break; }
        const auto* tbp = dynamic_cast<const TensorBeatPayload*>(rp.get());
        if (tbp == nullptr || tbp->dtype != TensorBeat::DType::F32) {
          session()->warn(fmt(
              "TextToSpeechStage('{}'): reference iport expects an f32 PCM "
              "TensorBeat, got {}; ignoring", this->id(), rp->describe()));
          continue;
        }
        std::vector<float> ref_wave = build_stereo_ref_wave_(
            session(), *tbp, _codec_v15->sample_rate(), _voice_ref_seconds);
        if (ref_wave.empty()) {
          session()->warn(fmt(
              "TextToSpeechStage('{}'): reference PCM produced no usable "
              "samples; ignoring", this->id()));
          continue;
        }
        auto rc = _codec_v15->encode(ref_wave);
        if (rc.empty()) {
          session()->warn(fmt(
              "TextToSpeechStage('{}'): codec-v2 encode of the reference "
              "produced 0 frames; ignoring", this->id()));
          continue;
        }
        _ref_codes = std::move(rc);
        _ref_set   = true;
        session()->info(fmt(
            "TextToSpeechStage('{}'): cloned reference voice -> {} codec "
            "frames (48 kHz stereo)", this->id(),
            static_cast<int>(_ref_codes.size())));
      }
    }

    genai::MossV15PromptIds pids;
    pids.instruction = _instruction;
    pids.language    = _language;
    // With a clone reference (iport or voice_lock cache) build the cloning
    // grid -- ref codes spliced into - Reference(s):; else the plain TTS grid.
    auto grid = _ref_set
        ? genai::moss_v15_build_clone_grid(*_tokenizer, _ref_codes, v15_text,
                                           pids)
        : genai::moss_v15_build_tts_grid(*_tokenizer, v15_text, pids);
    if (grid.empty()) { co_return; }
    // Audio RVQ codes MUST be sampled (defaults temp 1.7 / top_p 0.8 /
    // top_k 25); greedy degenerates into silence after the first sentence.
    genai::MossSampling v15_audio_sp;
    v15_audio_sp.temperature        = static_cast<float>(_audio_temp);
    v15_audio_sp.top_k              = _audio_top_k;
    v15_audio_sp.top_p              = static_cast<float>(_audio_top_p);
    v15_audio_sp.repetition_penalty = static_cast<float>(_audio_rep);
    const int v15_sr = _codec_v15->sample_rate();
    const int v15_ch = _codec_v15->channels();
    // Build a PCM TensorBeat (channel-major [ch, samples]) from a wav vector.
    auto make_pcm_beat = [&](const std::vector<float>& wav) {
      const std::int64_t per =
          static_cast<std::int64_t>(wav.size()) / (v15_ch > 0 ? v15_ch : 1);
      TensorBeat tb;
      tb.dtype = TensorBeat::DType::F32;
      tb.shape = { static_cast<std::int64_t>(v15_ch), per };
      tb.resize_contiguous(wav.size());
      std::memcpy(tb.as_f32(), wav.data(), wav.size() * sizeof(float));
      tb.sideband = FlexData::make_object();
      tb.sideband.as_object().insert("sample_rate", FlexData::make_int(v15_sr));
      return tb;
    };

    std::vector<std::vector<int>> frames;
    std::int64_t total_samps = 0;   // samples/channel emitted this beat
    if (_stream_chunk > 0) {
      // STREAMING: decode + emit PCM every _stream_chunk generated frames via
      // the codec's windowed-KV streaming decode, so the first audio leaves
      // ~one chunk after audio starts instead of after the whole utterance.
      // The generate callback is synchronous (can't co_await), so chunks go
      // out with write_sync; open=false (downstream closed) stops generation.
      // Reuse the cached ring state across beats; allocate only on first use
      // (or if the chunk size changed), else just re-arm the positions.
      if (!_stream_v15 || _stream_v15->max_chunk != _stream_chunk) {
        _stream_v15 = _codec_v15->decode_stream_begin(_stream_chunk);
      } else {
        _stream_v15->reset();
      }
      std::vector<std::vector<std::int32_t>> buf;
      bool open = (_stream_v15 != nullptr);
      auto flush = [&]() {
        if (!open || buf.empty()) { return; }
        std::vector<float> pcm =
            _codec_v15->decode_stream_chunk(*_stream_v15, buf);
        buf.clear();
        if (pcm.empty()) { return; }
        total_samps += static_cast<std::int64_t>(pcm.size())
                       / (v15_ch > 0 ? v15_ch : 1);
        open = ctx.write_sync(
            0, make_payload<TensorBeatPayload>(make_pcm_beat(pcm)));
        if (open) { ++_clips_emitted; }
      };
      auto on_frame = [&](const std::vector<int>& codes) -> bool {
        buf.emplace_back(codes.begin(), codes.end());
        if (static_cast<int>(buf.size()) >= _stream_chunk) { flush(); }
        return open && !new_text_pending();   // barge-in: stop on new text
      };
      frames = _lm_v15->generate(grid, _max_frames, v15_audio_sp,
                                 _sampler_seed, on_frame);
      flush();   // the final partial (< _stream_chunk) chunk
    } else {
      frames = _lm_v15->generate(
          grid, _max_frames, v15_audio_sp, _sampler_seed,
          [&](const std::vector<int>&) { return !new_text_pending(); });
    }
    const auto vt_gen = v15clock::now();
    if (frames.empty()) {
      session()->warn(fmt(
          "TextToSpeechStage('{}'): v1.5 generated 0 frames; emitting nothing",
          this->id()));
      co_return;
    }
    // voice_lock (design-once): cache the first generated voice as the clone
    // reference for later beats. An iport reference overrides it (_ref_set).
    if (_voice_lock && !_ref_set) {
      std::vector<std::vector<std::int32_t>> rc;
      rc.reserve(frames.size());
      for (const auto& f : frames) { rc.emplace_back(f.begin(), f.end()); }
      if (_voice_ref_seconds > 0.0) {
        const std::size_t cap = static_cast<std::size_t>(
            _voice_ref_seconds * _codec_v15->sample_rate() / 3840.0);
        if (cap >= 1 && rc.size() > cap) { rc.resize(cap); }
      }
      _ref_codes = std::move(rc);
      _ref_set   = true;
      session()->info(fmt(
          "TextToSpeechStage('{}'): voice_lock engaged -- {} frames cached as "
          "the v1.5 reference for subsequent beats", this->id(),
          static_cast<int>(_ref_codes.size())));
    }
    if (_stream_chunk <= 0) {
      // ONE-SHOT: decode the whole utterance and emit a single PCM beat.
      std::vector<std::vector<std::int32_t>> v15_codes;
      v15_codes.reserve(frames.size());
      for (const auto& f : frames) {
        v15_codes.emplace_back(f.begin(), f.end());
      }
      std::vector<float> v15_wav = _codec_v15->decode(v15_codes);
      if (v15_wav.empty()) {
        session()->warn(fmt(
            "TextToSpeechStage('{}'): codec-v2 produced 0 samples",
            this->id()));
        co_return;
      }
      total_samps = static_cast<std::int64_t>(v15_wav.size())
                    / (v15_ch > 0 ? v15_ch : 1);
      ++_clips_emitted;
      co_await ctx.write(
          0, make_payload<TensorBeatPayload>(make_pcm_beat(v15_wav)));
    }
    auto v15ms = [](v15clock::duration d) {
      return static_cast<int>(
          std::chrono::duration<double, std::milli>(d).count());
    };
    std::string mode = _stream_chunk > 0
        ? fmt("streamed {}-frame chunks", _stream_chunk)()
        : std::string("one-shot decode");
    session()->info(fmt(
        "TextToSpeechStage('{}'): v1.5 {} chars -> {} prompt rows{} -> {} "
        "frames -> {}x{} PCM = {:.2f}s @ {} Hz ({} ms gen, {})",
        this->id(), static_cast<int>(v15_text.size()),
        static_cast<int>(grid.size()), _ref_set ? " (cloned)" : "",
        static_cast<int>(frames.size()),
        v15_ch, static_cast<int>(total_samps),
        total_samps / static_cast<double>(v15_sr), v15_sr,
        v15ms(vt_gen - vt0), mode));
    co_return;
  }

  if (!_lm || !_codec || !_tokenizer) {
    session()->warn(fmt(
        "TextToSpeechStage('{}'): models not loaded (initialize "
        "failed?); dropping beat", this->id()));
    co_return;
  }

  // Accept a FlexData string or a FlexData object with a "text" key.
  const auto* fdp = dynamic_cast<const FlexDataPayload*>(t.get());
  if (!fdp) {
    session()->warn(fmt(
        "TextToSpeechStage('{}'): expected FlexDataPayload on in-port 0, "
        "got {}; dropping beat", this->id(), t->describe()));
    co_return;
  }
  std::string in_text;
  if (fdp->data.is_string()) {
    in_text = std::string(fdp->data.as_string(""));
  } else if (fdp->data.is_object()) {
    auto obj = fdp->data.as_object();
    if (obj.contains("text") && obj.at("text").is_string()) {
      in_text = std::string(obj.at("text").as_string(""));
    } else {
      session()->warn(fmt(
          "TextToSpeechStage('{}'): FlexData object has no string "
          "\"text\" key; dropping beat", this->id()));
      co_return;
    }
  } else {
    session()->warn(fmt(
        "TextToSpeechStage('{}'): FlexData payload is neither a string "
        "nor an object with a \"text\" key; dropping beat", this->id()));
    co_return;
  }

  const std::string text = normalize_text_(in_text);
  if (text.empty()) {
    session()->warn(fmt(
        "TextToSpeechStage('{}'): empty text after normalization; "
        "emitting nothing", this->id()));
    co_return;
  }

  using clock = std::chrono::steady_clock;
  const auto t_start = clock::now();

  const int n_vq = _lm->config().n_vq;             // 32
  const int pad  = _lm->config().audio_pad_code;   // 1024

  // 0. Drain any reference-audio beats on iport1 (voice cloning). The latest
  // sets the cloned voice for this and subsequent text beats (sticky). Each
  // is resampled to the codec rate, encoded to RVQ codes, and capped to
  // voice_ref_seconds. backlog() keeps the read non-blocking.
  if (_with_encoder && ctx.num_iports() >= 2) {
    while (ctx.backlog(1) > 0) {
      auto rp = co_await ctx.read(1);
      if (!rp) { break; }
      const auto* tbp = dynamic_cast<const TensorBeatPayload*>(rp.get());
      if (tbp == nullptr || tbp->dtype != TensorBeat::DType::F32) {
        session()->warn(fmt(
            "TextToSpeechStage('{}'): reference iport expects an f32 PCM "
            "TensorBeat, got {}; ignoring", this->id(), rp->describe()));
        continue;
      }
      int rsr = _codec->sample_rate();
      if (tbp->sideband.is_object()) {
        auto sb = tbp->sideband.as_object();
        if (sb.contains("sample_rate")) {
          rsr = static_cast<int>(sb.at("sample_rate").as_int(rsr));
        }
      }
      const std::size_t n_in =
          static_cast<std::size_t>(tbp->element_count());
      std::vector<float> ref_pcm = resample_pcm_(
          session(), tbp->as_f32(), n_in, rsr, _codec->sample_rate());
      if (ref_pcm.empty()) {
        session()->warn(fmt(
            "TextToSpeechStage('{}'): reference resample {} Hz -> {} Hz "
            "produced no samples; ignoring", this->id(), rsr,
            _codec->sample_rate()));
        continue;
      }
      auto rc = _codec->encode(ref_pcm);
      if (rc.empty()) {
        session()->warn(fmt(
            "TextToSpeechStage('{}'): codec encode of the reference "
            "produced 0 frames; ignoring", this->id()));
        continue;
      }
      if (_voice_ref_seconds > 0.0) {              // cap reference length
        const std::size_t cap = static_cast<std::size_t>(
            _voice_ref_seconds * _codec->sample_rate() / 1920.0);
        if (cap >= 1 && rc.size() > cap) { rc.resize(cap); }
      }
      _ref_codes = std::move(rc);
      _ref_set   = true;
      session()->info(fmt(
          "TextToSpeechStage('{}'): cloned reference voice from {} samples "
          "@ {} Hz -> {} codec frames", this->id(), n_in, rsr,
          static_cast<int>(_ref_codes.size())));
    }
  }

  // 1. Build the prompt grid [seq][1 + n_vq]: channel 0 = text/control id,
  // channels 1..n_vq = audio codes. A clone reference (from iport1 above, or
  // the voice_lock cache) splices into - Reference(s):; else None.
  std::vector<std::vector<std::int32_t>> prompt = build_moss_grid_(
      *_tokenizer, text, _lm->config(), _ref_set ? &_ref_codes : nullptr);
  if (prompt.empty()) {
    session()->warn(fmt(
        "TextToSpeechStage('{}'): empty prompt grid for the text; "
        "dropping beat", this->id()));
    co_return;
  }

  // 2. Sampled delay-pattern generation -> [G][1 + n_vq]. Separate audio +
  // text sampling (config; defaults = MossTTSDelay-8B recommendation). Greedy
  // (temperature <= 0) degenerates into silent loops, so sampling is default.
  genai::MossSampling audio_sp;
  audio_sp.temperature        = static_cast<float>(_audio_temp);
  audio_sp.top_k              = _audio_top_k;
  audio_sp.top_p              = static_cast<float>(_audio_top_p);
  audio_sp.repetition_penalty = static_cast<float>(_audio_rep);
  genai::MossSampling text_sp;
  text_sp.temperature        = static_cast<float>(_text_temp);
  text_sp.top_k              = _text_top_k;
  text_sp.top_p              = static_cast<float>(_text_top_p);
  text_sp.repetition_penalty = static_cast<float>(_text_rep);
  // NOTE: the 8B delay-pattern path is not yet streamed -- de-delaying a real
  // frame needs its code channels from the next n_vq delayed rows, so streaming
  // it requires a rolling de-delay (a self-contained follow-up). It always
  // decodes one-shot; flag it once so stream_chunk_frames isn't silently ignored.
  if (_stream_chunk > 0) {
    session()->info(fmt(
        "TextToSpeechStage('{}'): stream_chunk_frames set but the 8B "
        "delay-pattern variant decodes one-shot (streaming N/A here)",
        this->id()));
  }
  auto gen = _lm->generate_delay(
      prompt, _max_new_tokens, audio_sp, text_sp, _sampler_seed,
      [&]() { return new_text_pending(); });   // barge-in: stop on new text
  const auto t_gen = clock::now();
  if (gen.empty()) {
    session()->warn(fmt(
        "TextToSpeechStage('{}'): generate_delay_greedy produced 0 rows; "
        "emitting nothing", this->id()));
    co_return;
  }

  // 3. De-delay + drop all-pad frames -> codes [T][n_vq]. tokens[t][cb]
  // = gen[cb + t][1 + cb]; a frame whose every codebook is pad is
  // dropped. Matches the reference _decode_generated_audio pipeline.
  const int Gg      = static_cast<int>(gen.size());
  const int out_len = Gg - n_vq + 1;
  std::vector<std::vector<std::int32_t>> codes;
  for (int row = 0; row < out_len; ++row) {
    std::vector<std::int32_t> r(static_cast<std::size_t>(n_vq), 0);
    bool all_pad = true;
    for (int cb = 0; cb < n_vq; ++cb) {
      int v = gen[static_cast<std::size_t>(cb + row)]
                 [static_cast<std::size_t>(1 + cb)];
      if (v != pad) { all_pad = false; }
      if (v < 0 || v >= pad) { v = pad - 1; }  // clamp pad/OOB to valid
      r[static_cast<std::size_t>(cb)] = v;
    }
    if (!all_pad) { codes.push_back(std::move(r)); }
  }
  if (codes.empty()) {
    session()->warn(fmt(
        "TextToSpeechStage('{}'): no non-pad audio frames after de-delay "
        "({} generated rows); emitting nothing", this->id(), Gg));
    co_return;
  }

  // 3b. voice_lock (design-once): cache the FIRST generated voice and reuse it
  // as the clone reference for later beats, so the timbre stays consistent
  // across texts. An external iport reference (above) takes precedence (it
  // sets _ref_set), so this only fires until a voice is locked.
  if (_voice_lock && !_ref_set) {
    _ref_codes = codes;                            // copy (codes is decoded next)
    if (_voice_ref_seconds > 0.0) {
      const std::size_t cap = static_cast<std::size_t>(
          _voice_ref_seconds * _codec->sample_rate() / 1920.0);
      if (cap >= 1 && _ref_codes.size() > cap) { _ref_codes.resize(cap); }
    }
    _ref_set = true;
    session()->info(fmt(
        "TextToSpeechStage('{}'): voice_lock engaged -- {} frames cached as "
        "the reference for subsequent beats", this->id(),
        static_cast<int>(_ref_codes.size())));
  }

  // 4. Codec decode -> [T*1920] f32 PCM @ sample_rate.
  std::vector<float> wave = _codec->decode(codes, nullptr);
  const int sr = _codec->sample_rate();
  const auto t_decode = clock::now();
  if (wave.empty()) {
    session()->warn(fmt(
        "TextToSpeechStage('{}'): codec decode produced 0 PCM samples; "
        "emitting nothing", this->id()));
    co_return;
  }

  // 5. Emit the PCM as a rank-1 [n_samples] f32 TensorBeat, with the
  // sample rate in the sideband object.
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = { static_cast<std::int64_t>(wave.size()) };
  tb.resize_contiguous(wave.size());
  std::memcpy(tb.as_f32(), wave.data(), wave.size() * sizeof(float));
  tb.sideband = FlexData::make_object();
  tb.sideband.as_object().insert("sample_rate", FlexData::make_int(sr));

  double peak = 0.0;
  for (float s : wave) {
    const double a = s < 0.0f ? -static_cast<double>(s)
                              :  static_cast<double>(s);
    if (a > peak) { peak = a; }
  }
  auto ms = [](clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  session()->info(fmt(
      "TextToSpeechStage('{}'): {} chars -> {} prompt rows{} -> {} gen rows "
      "-> {} frames -> {} PCM samples = {:.2f}s @ {} Hz, peak={:.3f} "
      "({} ms gen + {} ms decode)",
      this->id(),
      static_cast<int>(text.size()),
      static_cast<int>(prompt.size()),
      _ref_set ? " (cloned)" : "", Gg,
      static_cast<int>(codes.size()), wave.size(),
      wave.size() / static_cast<double>(sr), sr, peak,
      static_cast<int>(ms(t_gen - t_start)),
      static_cast<int>(ms(t_decode - t_gen))));

  ++_clips_emitted;
  co_await ctx.write(0, make_payload<TensorBeatPayload>(std::move(tb)));
#else
  (void)t;
  // No apple-silicon MOSS-TTS in this build: emit nothing.
  co_return;
#endif
  co_return;
}

VPIPE_REGISTER_STAGE(TextToSpeechStage)
VPIPE_REGISTER_SPEC(TextToSpeechStage, kSpec)

}
