#include "stages/text-to-speech-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"
#include "stages/sampler-spec.h"
#include "pipeline/memory-plan.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "common/ffmpeg-libraries.h"
#include "generative-models/moss/metal-moss-tts-model.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/weight-set.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/moss/metal-moss-codec-v2.h"
#include "generative-models/moss/metal-moss-v15-model.h"
#include "generative-models/moss/moss-v15-processor.h"
#include "generative-models/moss/metal-moss-rt-model.h"
#include "generative-models/moss/moss-rt-processor.h"
#include "generative-models/moss/moss-delay-processor.h"
#include "generative-models/moss/moss-text-normalizer.h"
#include "generative-models/tokenizer.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <utility>
#endif

#include <cstdint>
#include <string>
#include <vector>

using namespace std;

namespace vpipe {

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

// MOSS's own recommended AUDIO-code sampling. Unlike every other channel in
// the LLM stages, this one does NOT fall back to greedy when its sampler
// iport is unwired: MOSS's audio head degenerates into silent loops under
// argmax, so an unprogrammed audio channel keeps the model's defaults (the
// MossTTSDelay-8B recommendation, shared with v1.5). A wired sampler-select
// replaces this wholesale.
//
// The seed is the historical 'moss' constant: nonzero makes a given text
// deterministic, so the same script yields the same voice every run.
genai::SamplerParams
moss_audio_default_sampler()
{
  genai::SamplerParams p;
  p.temperature        = 1.7f;
  p.top_k              = 25;
  p.top_p              = 0.8f;
  p.repetition_penalty = 1.0f;
  p.seed               = 0x6d6f7373;   // 'moss'
  return p;
}

// The realtime variant (moss_tts_realtime) has its OWN recommendation --
// temp 0.8 / top_k 30 / top_p 0.6 / rep 1.1 -- distinct from the 8B/v1.5
// numbers above. Applied only when iport2 is unwired.
genai::SamplerParams
moss_rt_audio_default_sampler()
{
  genai::SamplerParams p;
  p.temperature        = 0.8f;
  p.top_k              = 30;
  p.top_p              = 0.6f;
  p.repetition_penalty = 1.1f;
  p.seed               = 0x6d6f7373;   // 'moss'
  return p;
}

// SamplerParams -> MossSampling (see warn_unsupported_knobs_ for the two
// knobs that do not survive the narrowing).
genai::MossSampling
moss_sampling_of(const genai::SamplerParams& p)
{
  genai::MossSampling m;
  m.temperature        = p.temperature;
  m.top_k              = p.top_k;
  m.top_p              = p.top_p;
  m.repetition_penalty = p.repetition_penalty;
  return m;
}

// MossSampling carries no min_p / presence penalty, so a sampler-select spec
// that sets either cannot be honoured here. Say so once, on latch, rather
// than dropping the knob silently.
void
warn_unsupported_knobs_(const SessionContextIntf*   session,
                        const std::string&          who,
                        const genai::SamplerParams& p)
{
  if (p.min_p > 0.0f) {
    session->warn(fmt("{}: min_p {} is not supported by the MOSS sampler "
                      "(temperature / top_k / top_p / repetition_penalty "
                      "only); ignoring it", who, p.min_p));
  }
  if (p.presence_penalty != 0.0f) {
    session->warn(fmt("{}: presence_penalty {} is not supported by the MOSS "
                      "sampler (temperature / top_k / top_p / "
                      "repetition_penalty only); ignoring it", who,
                      p.presence_penalty));
  }
}

}  // namespace
#endif

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
  _codec_int8 = (attr_str("codec_quant") == "int8");
  _instruction   = attr_str("instruction");
  _language      = attr_str("language");
  _quality       = attr_str("quality");
  _sound_event   = attr_str("sound_event");
  _ambient_sound = attr_str("ambient_sound");
  _wait_reference = attr_bool("wait_for_reference");
  {
    int64_t v = attr_int("duration_tokens");
    if (v < 0) {
      fail_config(fmt(
          "TextToSpeechStage('{}'): duration_tokens must be >= 0 (0 = let "
          "the model decide; got {})", this->id(), v));
    }
    _duration_tokens = v > 0 ? static_cast<int>(v) : 0;
  }
  _lm_quant = attr_str("lm_quant");
  if (_lm_quant.empty()) { _lm_quant = "w8"; }
  if (_lm_quant != "w8" && _lm_quant != "bf16") {
    fail_config(fmt(
        "TextToSpeechStage('{}'): lm_quant must be \"w8\" or \"bf16\" "
        "(got \"{}\")", this->id(), _lm_quant));
  }
  _lm_quant_bits = _lm_quant == "bf16" ? 0 : 8;
  _i8_gemm = attr_bool("i8_gemm");
  _text_normalizer = attr_str("text_normalizer");
  if (_text_normalizer.empty()) { _text_normalizer = "auto"; }
  if (_text_normalizer != "auto" && _text_normalizer != "robust" &&
      _text_normalizer != "basic") {
    fail_config(fmt(
        "TextToSpeechStage('{}'): text_normalizer must be \"auto\", "
        "\"robust\" or \"basic\" (got \"{}\")", this->id(),
        _text_normalizer));
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
  // Sampling is programmed by two `sampler-select` stages on the OPTIONAL
  // iports 2 (audio codes) and 3 (free text), latched on the first beat.
  // Unwired: the TEXT channel falls back to greedy (the SamplerParams
  // default) like every other LLM stage, but the AUDIO channel keeps MOSS's
  // own recommendation -- greedy audio degenerates into silence.
#ifdef VPIPE_BUILD_APPLE_SILICON
  _audio_sp = moss_audio_default_sampler();
#endif
  _voice_lock        = attr_bool("voice_lock");
  _voice_ref_seconds = attr_real("voice_ref_seconds");
  // A wired PCM-reference iport1 means the codec needs its encode path, so a
  // reference voice can be analysed for cloning. iport_edges() reflects the
  // edges this stage was constructed with (the graph is frozen post-launch);
  // check the SLOT, not the count -- the sampler iports 2/3 sit after it, so
  // a graph that wires only a sampler must not pay for the encoder.
  _with_encoder = this->iport_edges().size() >= 2
      && this->iport_edges()[1].v != nullptr;

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
          "same-named path. Delay-pattern MOSS-TTS 8B / MOSS-TTS-v1.5 "
          "(moss-tts; bf16 as published, or model-quantize'd), Local-v1.5 "
          "(moss-tts-local), or realtime (moss-tts-realtime).",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "moss-tts,moss-tts-local,moss-tts-realtime"},
  {.key = "codec_dir", .type = ConfigType::String, .required = true,
   .doc = "MOSS-Audio-Tokenizer (codec) model: a models-DB key or a "
          "filesystem path; a DB key wins over a same-named path. Match the "
          "LM variant: moss-codec (MOSS-TTS 8B / v1.5, realtime) or "
          "moss-codec-v2 (Local-v1.5).",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "moss-codec,moss-codec-v2"},
  {.key = "codec_quant", .type = ConfigType::String,
   .doc = "codec weight precision: \"int8\" stores the codec's transformer "
          "GEMM weights as int8 group-32 affine (~half the resident codec "
          "footprint, small audio-quality cost); default/empty = f16",
   .def_str = ""},
  {.key = "max_new_tokens", .type = ConfigType::Int,
   .doc = "delay-pattern variants (MOSS-TTS 8B, MOSS-TTS-v1.5) only: "
          "per-beat generation budget in rows (>= 1; 12.5 rows ~= 1 s of "
          "audio, plus 32 rows of delay tail)",
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
  // The depth-decoder variants' frame budget (Local-v1.5, realtime).
  {.key = "max_frames", .type = ConfigType::Int,
   .doc = "Local-v1.5 / realtime only: per-beat frame budget (each ~= "
          "80 ms); >= 1",
   .def_int = 1000},
  // Prompt fields (the processor's <user_inst> block). "None" is how the
  // reference renders an unset field.
  {.key = "instruction", .type = ConfigType::String,
   .doc = "optional style instruction (prompt field; MOSS-TTS / v1.5 and "
          "Local-v1.5). \"None\" = unset",
   .def_str = "None"},
  {.key = "language", .type = ConfigType::String,
   .doc = "language of the text, by English name: \"English\", "
          "\"Chinese\", \"French\", ... (prompt field; MOSS-TTS / v1.5 and "
          "Local-v1.5). MOSS-TTS-v1.5 is markedly better on almost every "
          "language when it is set. \"None\" = unset",
   .def_str = "None"},
  {.key = "quality", .type = ConfigType::String,
   .doc = "delay-pattern variants only: a whole-utterance quality / style "
          "hint (prompt field), e.g. \"Studio recording\", \"Telephone "
          "call quality\". \"None\" = unset",
   .def_str = "None"},
  {.key = "sound_event", .type = ConfigType::String,
   .doc = "delay-pattern variants only: a whole-utterance sound-event hint "
          "(prompt field), e.g. \"Laughter\", \"Sigh\", \"Breathing\" -- "
          "the whole utterance is conditioned on it, not one point in it. "
          "\"None\" = unset",
   .def_str = "None"},
  {.key = "ambient_sound", .type = ConfigType::String,
   .doc = "delay-pattern variants only: a whole-utterance ambience hint "
          "(prompt field), e.g. \"Rain\", \"Office room tone\". "
          "Experimental on the base model: weak, or a longer tail. "
          "\"None\" = unset",
   .def_str = "None"},
  {.key = "wait_for_reference", .type = ConfigType::Bool,
   .doc = "with the audio-ref iport wired: the FIRST utterance waits for "
          "a reference beat (or that port's end) before it is spoken. "
          "Off, the port is drained without waiting, so a reference that "
          "arrives after the first text (one decoded from a file, behind "
          "a graph that emits the text at launch) clones nothing for it",
   .def_bool = false},
  {.key = "duration_tokens", .type = ConfigType::Int,
   .doc = "delay-pattern variants only: the expected length of the speech "
          "in audio tokens (12.5 tokens ~= 1 s), the prompt's duration "
          "control. 0 = let the model decide",
   .def_int = 0},
  {.key = "lm_quant", .type = ConfigType::String,
   .doc = "how an UNQUANTIZED LM checkpoint's backbone is held: \"w8\" "
          "(default) quantizes it in memory as it loads, to affine 8-bit "
          "group-64 -- ~53% of the bf16 bytes at near-bf16 accuracy, and "
          "the faster decode since every token reads every weight; "
          "\"bf16\" keeps it as published. On the delay-pattern model "
          "\"w8\" also holds the text embedding and text head at w8 "
          "(their slot-token rows stay exact). A model-quantize'd or MLX "
          "8-bit LM loads as it sits either way. When even that does not "
          "fit the box, the backbone streams its layers (decided before "
          "loading, from the memory plan) and keeps as many resident as "
          "the machine allows; an 8-bit pack streams at half the bytes "
          "per layer of a bf16 checkpoint, so on a small box point hf_dir "
          "at one (model-quantize, bits 8)",
   .def_str = "w8"},
  {.key = "i8_gemm", .type = ConfigType::Bool,
   .doc = "accelerated mode (LOSSY): the LM backbone's prefill GEMMs as "
          "dynamic-int8 matmul2d, ~1e-2 per GEMM. MATRIX CORES ONLY -- an "
          "M5 path; any other box stays dense. It declines GEMMs under "
          "1024 rows, so only a long prompt (long-form text, or text plus "
          "a long cloned voice) reaches it; generation itself is one row a "
          "frame and never does. Env VPIPE_I8_GEMM overrides",
   .def_bool = false},
  {.key = "text_normalizer", .type = ConfigType::String,
   .doc = "text clean-up before the prompt is built: \"robust\" = "
          "MOSS-TTS-v1.5's own normalizer (line breaks become sentence "
          "breaks, CJK/Latin spacing, brackets, arrows, repeated "
          "punctuation; URLs, e-mails and file names protected), "
          "\"basic\" = control characters dropped and whitespace "
          "collapsed, \"auto\" (default) = robust exactly when the LM dir "
          "ships v1.5's normalizer script, as its processor does",
   .def_str = "auto"},
  {.key = "voice_lock", .type = ConfigType::Bool,
   .doc = "design-once: cache the FIRST generated voice and reuse it as the "
          "clone reference for every later beat, so the timbre stays the same "
          "across different texts (the first voice is picked by the audio "
          "sampler's seed). A reference on the audio iport overrides it. "
          "Needs no audio input.",
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
          "synthesize. An object may also carry \"end_of_response\" (bool); "
          "barge-in only cuts the current utterance on a new beat when the "
          "current one is end_of_response (true, or absent -> defaults true). "
          "A mid-reply chunk (false, from text-chat's stream oport) is always "
          "spoken in full",
   .type = &typeid(FlexDataPayload),
   .tags = "text", .clock_group = 0},
  {.name = "audio-ref",
   .doc = "OPTIONAL mono f32 PCM TensorBeat (any rate; sideband.sample_rate "
          "honoured): a reference voice to clone. Latest beat is sticky "
          "(applies to all later utterances until superseded). Its own clock "
          "group -- it arrives independently of the text stream / PCM output, "
          "not rate-locked to them.",
   .type = &typeid(TensorBeatPayload), .clock_group = 1},
  {.name = "audio-sampler",
   .doc = "OPTIONAL token-sampler spec FlexData (sampler-select) for the "
          "AUDIO-code channel; its seed is also the voice seed. Latched on "
          "the first beat. Unwired keeps MOSS's own recommended sampling "
          "(NOT greedy -- greedy audio degenerates into silent loops)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "text-sampler",
   .doc = "OPTIONAL token-sampler spec FlexData (sampler-select) for the "
          "free-TEXT channel. Latched on the first beat; unwired = greedy, "
          "which is the right default here (vpipe generates the text channel "
          "and it must follow the transcript to reach audio_end)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "pcm",
   .doc = "TensorBeat f32 PCM (sideband.sample_rate set); downstream "
          "optional. Delay-pattern variants: rank-1 [n_samples] mono @ "
          "24 kHz. Local-v1.5: rank-2 [2, n_samples] stereo @ 48 kHz. "
          "Realtime: [1, n_samples] mono @ 24 kHz.",
   .type = &typeid(TensorBeatPayload),
   .tags = "pcm-samples", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "text-to-speech",
  .doc       = "Text-to-speech (MOSS-TTS, metal): synthesizes each input "
               "text beat into a PCM waveform and emits it as a TensorBeat. "
               "The variant is chosen from the LM dir's config.json "
               "model_type: \"moss_tts_delay\" / \"moss_tts\" (MOSS-TTS 8B "
               "and MOSS-TTS-v1.5, delay pattern -> 24 kHz mono), "
               "\"moss_tts_local\" (Local-v1.5 depth decoder -> 48 kHz "
               "stereo) or \"moss_tts_realtime\" (24 kHz mono).",
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
  // Plain Qwen3 RMSNorm: consulted when the checkpoint is raw bf16, where
  // the Qwen3.5 default would fold +1 into every norm.
  c.zero_centered_norm = false;
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

// What one of the three MOSS LM variants will hold, from its checkpoint
// alone -- the only source available while a graph is being planned.
struct LmPlan {
  std::string variant;       // "delay" | "local" | "realtime"
  std::string layer_prefix;  // the streamable backbone stack
  int n_layers = 0, n_kv = 8, head_dim = 128;
  genai::MetalMossTtsModel::MemoryPlan mem;
};

LmPlan
lm_plan_(const std::string& dir, int quant_bits)
{
  LmPlan p;
  const std::string mt = read_model_type_(dir);
  std::string sub = "language_config";
  if (mt == "moss_tts_local") {
    p.variant = "local";
    p.layer_prefix = "transformer.layers.";
    p.n_layers = 36;
    sub = "qwen3_config";
  } else if (mt == "moss_tts_realtime") {
    p.variant = "realtime";
    p.layer_prefix = "language_model.model.layers.";
    p.n_layers = 28;
  } else {
    p.variant = "delay";
    p.layer_prefix = "language_model.layers.";
    p.n_layers = 36;
  }
  std::ifstream in(dir + "/config.json");
  if (in) {
    try {
      FlexData root = FlexData::from_json(in);
      if (root.is_object() && root.as_object().contains(sub)) {
        const FlexData lc = root.as_object().at(sub);
        if (lc.is_object()) {
          const auto o = lc.as_object();
          auto gi = [&](const char* k, int d) {
            return o.contains(k) ? (int)o.at(k).as_int(d) : d;
          };
          p.n_layers = gi("num_hidden_layers", p.n_layers);
          p.n_kv     = gi("num_key_value_heads", p.n_kv);
          p.head_dim = gi("head_dim", p.head_dim);
        }
      }
    } catch (...) {}
  }
  // The delay model also holds its text embedding and text head at w8
  // when the backbone is (MossTtsLoadOptions::trunk_w8).
  p.mem = genai::MetalMossTtsModel::plan_memory(
      dir, p.layer_prefix, p.n_layers, quant_bits, p.variant == "delay");
  return p;
}

}  // namespace
#endif  // VPIPE_BUILD_APPLE_SILICON

#ifdef VPIPE_BUILD_APPLE_SILICON
void
TextToSpeechStage::release_models_()
{
  // Streaming state first: both StreamStates hold buffers minted by
  // their codec, so they must go before the codec they came from.
  _stream_v15.reset();
  _stream_v1.reset();
  _lm_rt.reset();
  _lm_v15.reset();
  _codec_v15.reset();
  _lm.reset();
  _codec.reset();
  _tokenizer.reset();
  // Hand the checkpoints to the manager's removable pool now that nothing
  // borrows them: a relaunch over the same models then finds the cached
  // trunk (embedding tables, heads) instead of re-reading it, and under
  // pressure the pages simply go.
  if (auto* mgr = session() != nullptr
                      ? session()->services()->generative_model_manager()
                      : nullptr) {
    if (!_lm_dir_resolved.empty()) { mgr->pool_weights(_lm_dir_resolved); }
    if (!_codec_dir_resolved.empty()) {
      mgr->pool_weights(_codec_dir_resolved);
    }
  }
  // Per-run conversation state that belongs to the weights just freed.
  _ref_codes.clear();
  _ref_set = false;
}
#endif

std::string
TextToSpeechStage::scratch_label_() const
{
  // Per STAGE: two TTS stages in one graph each allocate their own.
  return "text-to-speech:" + this->id();
}

// What the stage allocates to run, beyond weights: the LM's K/V at the
// configured budget (plus a generous prompt -- a long text, or a cloned
// voice spliced into it), and the codec's decode working set plus the PCM
// of a whole utterance. Bounded from configuration, corrected per beat.
std::size_t
TextToSpeechStage::scratch_bytes_(const std::string& lm_dir,
                                  std::size_t* kv_part) const
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  const LmPlan p = lm_plan_(lm_dir, _lm_quant_bits);
  const bool delay = p.variant == "delay";
  const std::size_t budget =
      (std::size_t)(delay ? _max_new_tokens : _max_frames);
  const std::size_t prompt =
      512 + (std::size_t)(std::max(0.0, _voice_ref_seconds) * 12.5) + 32;
  std::size_t tokens = budget + prompt;
  tokens = (tokens + 255) / 256 * 256;            // K/V pages hold 256
  const std::size_t kv = tokens * (std::size_t)p.n_layers * p.n_kv *
                         p.head_dim * 2 /*K,V*/ * 2 /*bf16*/;
  // PCM: 1920 samples per 80 ms frame at 24 kHz mono (3840 x 2 channels
  // at 48 kHz for Local-v1.5), f32, for the one-shot decode's utterance.
  const std::size_t pcm = budget * 1920 * 4 * (delay ? 1 : 2);
  if (kv_part != nullptr) { *kv_part = kv; }
  return kv + pcm + ((std::size_t)256 << 20);
#else
  (void)lm_dir;
  if (kv_part != nullptr) { *kv_part = 0; }
  return 0;
#endif
}

StageMemory
TextToSpeechStage::declare_memory() const
{
  StageMemory m;
#ifdef VPIPE_BUILD_APPLE_SILICON
  // BOTH holdings, each NAMED by its checkpoint so a second stage over
  // either one is not billed twice -- and each at what it will HOLD, not
  // what it weighs on disk: the LM backbone built (w8 in memory by
  // default) with its streaming floor, the codec at f16 without the
  // encoder it only loads for cloning.
  std::string lm_dir;
  if (!_hf_dir.empty()) {
    lm_dir = resolve_model_dir(session(), _hf_dir);
    const LmPlan p = lm_plan_(lm_dir, _lm_quant_bits);
    m.hold(lm_dir, p.mem.preload > 0 ? p.mem.preload
                                     : model_memory::dir_weights_bytes(lm_dir),
           p.mem.raw ? p.mem.floor : 0,
           /*releases=*/false, /*reclaimable=*/false);
    if (!_codec_dir.empty()) {
      const std::string cd = resolve_model_dir(session(), _codec_dir);
      const std::size_t held =
          p.variant == "local"
              ? genai::MetalMossCodecV2::plan_bytes(cd, _with_encoder)
              : genai::MetalMossCodec::plan_bytes(cd, _with_encoder);
      m.hold(cd, held > 0 ? held : model_memory::dir_weights_bytes(cd));
    }
    m.scratch = scratch_bytes_(lm_dir);
    // One PCM chunk leaves per stream_chunk_frames frames (one utterance
    // in one-shot mode).
    const std::size_t frames =
        _stream_chunk > 0 ? (std::size_t)_stream_chunk
                          : (std::size_t)(p.variant == "delay"
                                              ? _max_new_tokens
                                              : _max_frames);
    m.outputs = {frames * 1920 * 4 * (p.variant == "local" ? 2u : 1u)};
  }
#endif
  return m;
}

std::vector<ResourceClaim>
TextToSpeechStage::declare_resources() const
{
  std::vector<ResourceClaim> out;
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_hf_dir.empty()) { return out; }
  const std::string lm_dir = resolve_model_dir(session(), _hf_dir);
  const LmPlan p = lm_plan_(lm_dir, _lm_quant_bits);
  // STREAMABLE when the backbone is raw: its floor -- the trunk plus a
  // slot pair -- is a promise the loader keeps (stream_decode), and the
  // claim is the only ledger the refusal check reads. A pack holds as it
  // sits, so it claims plainly.
  if (p.mem.raw && p.mem.floor > 0) {
    out.push_back(model_memory::weight_claim_streamable(lm_dir, p.mem.floor));
  } else {
    for (auto& c : model_memory::weight_claims({lm_dir})) {
      out.push_back(std::move(c));
    }
  }
  if (!_codec_dir.empty()) {
    for (auto& c : model_memory::weight_claims(
             {resolve_model_dir(session(), _codec_dir)})) {
      out.push_back(std::move(c));
    }
  }
  // Held for the whole run: a TTS stage serves every beat it is given.
  for (auto& c : model_memory::scratch_claims(scratch_label_(),
                                              scratch_bytes_(lm_dir), {})) {
    out.push_back(std::move(c));
  }
#endif
  return out;
}

// The plan was declared from the checkpoints; now the models say what they
// actually hold. Both ledgers, every time -- after load, and after every
// utterance, since residency grows a streaming backbone as it decodes.
void
TextToSpeechStage::revise_holdings_()
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_lm_dir_resolved.empty()) { return; }
  std::size_t lm_held = 0, codec_held = 0, kv = 0;
  if (_lm) {
    lm_held = _lm->resident_bytes();
    kv = _lm->kv_bytes();
  } else if (_lm_v15 && _lm_v15->backbone()) {
    lm_held = _lm_v15->backbone()->resident_bytes() + _lm_trunk;
    if (auto* cm = _lm_v15->backbone()->context_manager()) {
      kv = cm->resident_bytes();
    }
  } else if (_lm_rt && _lm_rt->backbone()) {
    lm_held = _lm_rt->backbone()->resident_bytes() + _lm_trunk;
    if (auto* cm = _lm_rt->backbone()->context_manager()) {
      kv = cm->resident_bytes();
    }
  }
  if (_codec) { codec_held = _codec->resident_bytes(); }
  if (_codec_v15) { codec_held = _codec_v15->resident_bytes(); }
  if (lm_held == 0) { return; }
  // Int8 engagement since the last revision: an enabled tier is not an
  // engaged one (it declines every GEMM under 1024 rows), so say which.
  {
    std::uint64_t i8 = 0;
    if (_lm) { i8 = _lm->i8_gemm_count(); }
    else if (_lm_v15 && _lm_v15->backbone()) {
      i8 = _lm_v15->backbone()->i8_gemm_count();
    } else if (_lm_rt && _lm_rt->backbone()) {
      i8 = _lm_rt->backbone()->i8_gemm_count();
    }
    if (i8 > _i8_seen) {
      session()->info(fmt(
          "TextToSpeechStage('{}'): int8 tier took {} prefill GEMMs",
          this->id(), i8 - _i8_seen));
    }
    _i8_seen = i8;
  }
  // A streaming backbone's kept set moves as it decodes: say where it got
  // to, once per change, so a slow utterance reads as disk or as compute.
  if (_lm_streams) {
    int kept = 0, layers = 0;
    if (_lm) {
      kept = _lm->resident_layer_count();
      layers = _lm->n_layers();
    } else if (_lm_v15 && _lm_v15->backbone()) {
      kept = _lm_v15->backbone()->resident_layer_count();
      layers = _lm_v15->backbone()->config().n_layers;
    } else if (_lm_rt && _lm_rt->backbone()) {
      kept = _lm_rt->backbone()->resident_layer_count();
      layers = _lm_rt->backbone()->config().n_layers;
    }
    if (kept != _kept_seen) {
      session()->info(fmt(
          "TextToSpeechStage('{}'): streaming backbone now keeps {}/{} "
          "layers resident (LM holds {} MB)", this->id(), kept, layers,
          lm_held >> 20));
      _kept_seen = kept;
    }
  }
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_declaration(_lm_dir_resolved, lm_held);
    if (codec_held > 0 && !_codec_dir_resolved.empty()) {
      mgr->revise_declaration(_codec_dir_resolved, codec_held);
    }
    // The scratch claim's K/V term was a bound from config; now it is
    // what the pools grew to. The codec / PCM allowance stays as bounded.
    std::size_t kv_bound = 0;
    const std::size_t bound = scratch_bytes_(_lm_dir_resolved, &kv_bound);
    if (kv > 0) {
      mgr->revise_scratch(scratch_label_(), bound - kv_bound + kv);
    }
  }
  StageMemory m = declare_memory();
  for (StageHolding& h : m.holdings) {
    if (h.source == _lm_dir_resolved) {
      correct_loaded_holding(h, lm_held, _lm_floor);
    } else if (h.source == _codec_dir_resolved && codec_held > 0) {
      correct_loaded_holding(h, codec_held);
    }
  }
  revise_memory(m);
#endif
}

Job
TextToSpeechStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_hf_dir.empty() || _codec_dir.empty()) {
    co_return;  // ctor already recorded the config error.
  }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "TextToSpeechStage('{}'): no metal-compute backend on this "
        "session (Metal unavailable, or the build lacks "
        "VPIPE_BUILD_APPLE_SILICON); the stage is inert", this->id()));
    co_return;
  }
  // Release anything a PREVIOUS run left resident before loading again.
  // A stopped pipeline keeps its stages alive (PipelineRuntime::stop
  // drains the drivers; only unload/re-materialize destroys the Stage),
  // so a plain Stop-then-Start re-enters initialize() with the last
  // run's weights still held. Without this the loads below would build
  // a complete second copy -- the assignment only frees the old pointer
  // AFTER the new model is fully constructed -- and peak briefly at 2x.
  // Measured on MOSS-TTS v1.5 + codec v2: 10.2 GB resident, 18.5 GB
  // peak across a relaunch, which is what makes a restart fail on a box
  // sized for one copy. Unlike the LM-manager-backed stages (whose
  // reload hits a shared cache and returns the same instance) and the
  // diffusion stages (guarded by _load_attempted), nothing here dedupes
  // the second load, so the release has to be explicit.
  release_models_();
  // Per-launch reset, same reason: the sampler-select sources upstream
  // re-emit on every launch, so a stage that never clears these keeps
  // the previous run's latch and leaves the new beats unread.
  _audio_sp_read = false;
  _audio_sp_set  = false;
  _text_sp_read  = false;

  const std::string lm_dir =
      resolve_model_dir(session(), _hf_dir);
  const std::string codec_dir =
      resolve_model_dir(session(), _codec_dir);
  // "auto" follows the checkpoint's own processor: MOSS-TTS-v1.5 ships
  // (and runs) its normalizer script, every other MOSS checkpoint does not.
  _robust_text = _text_normalizer == "robust" ||
                 (_text_normalizer == "auto" &&
                  genai::moss_tts_has_text_normalizer(lm_dir));

  using clock = std::chrono::steady_clock;
  auto ms = [](clock::duration d) {
    return static_cast<int>(
        std::chrono::duration<double, std::milli>(d).count());
  };
  auto mb = [](std::size_t b) { return b >> 20; };
  _lm_dir_resolved    = lm_dir;
  _codec_dir_resolved = codec_dir;
  _i8_seen            = 0;
  _kept_seen          = -1;
  auto* mgr = session()->services()->generative_model_manager();

  // The variant comes from the LM dir's config.json: "moss_tts_realtime"
  // (Qwen3-1.7B backbone + depth decoder -> 24 kHz mono), "moss_tts_local"
  // (Local-v1.5 depth decoder -> codec-v2, 48 kHz stereo), anything else
  // the delay pattern (MOSS-TTS 8B / MOSS-TTS-v1.5 -> 24 kHz mono).
  const LmPlan lmp = lm_plan_(lm_dir, _lm_quant_bits);
  const bool realtime = lmp.variant == "realtime";
  const bool local    = lmp.variant == "local";

  // 1. THE CODEC FIRST, and its declaration corrected at once: the LM's
  // streaming decision below sizes itself against everything this graph
  // holds, and the codec's F32 checkpoint is ~4x what the codec keeps.
  session()->info(fmt(
      "TextToSpeechStage('{}'): loading {} from '{}'{}", this->id(),
      local ? "codec-v2" : "MOSS-Audio-Tokenizer codec", _codec_dir,
      _with_encoder ? " (with encoder: voice cloning enabled)" : ""));
  const auto t_cc0 = clock::now();
  std::size_t codec_held = 0;
  if (local) {
    _codec_v15 = genai::MetalMossCodecV2::load(
        genai::open_weight_set(codec_dir, session()), mc, _with_encoder);
    if (!_codec_v15 || !_codec_v15->valid()) {
      session()->error(fmt(
          "TextToSpeechStage('{}'): failed to load codec-v2 from '{}'; inert",
          this->id(), _codec_dir));
      _codec_v15.reset();
      co_return;
    }
    codec_held = _codec_v15->resident_bytes();
  } else {
    _codec = genai::MetalMossCodec::load(
        genai::open_weight_set(codec_dir, session()), mc, _codec_int8,
        _with_encoder);
    if (!_codec || !_codec->valid()) {
      session()->error(fmt(
          "TextToSpeechStage('{}'): failed to load MOSS codec from '{}'; "
          "the stage is inert", this->id(), _codec_dir));
      _codec.reset();
      co_return;
    }
    // Route the codec's audio-codec-lane profiler events to this session.
    _codec->set_session(session());
    codec_held = _codec->resident_bytes();
  }
  if (mgr != nullptr && codec_held > 0) {
    mgr->revise_declaration(codec_dir, codec_held);
  }
  session()->info(fmt(
      "TextToSpeechStage('{}'): codec loaded in {} ms, holding {} MB "
      "({} MB on disk)", this->id(), ms(clock::now() - t_cc0),
      mb(codec_held), mb(model_memory::dir_weights_bytes(codec_dir))));

  // 2. THE LM'S MEMORY DECISIONS (docs/MODEL-MEMORY.md). An autoregressive
  // backbone re-reads every weight per token, so whatever of it the box
  // cannot keep in RAM is paid per token: it must never page. Streaming is
  // an argument to the backbone's load -- irreversible -- so it is taken
  // HERE, by plan_streaming with the wider headroom, against the backbone
  // as it will be BUILT: the bytes in-memory quantization sheds are its
  // `retires`, exactly as an AdaLN bake's are. An affine w8 pack streams
  // as it sits, at half a bf16 source's bytes per read; any other pack
  // (4-bit, mixed) holds as it sits.
  _lm_floor   = lmp.mem.raw ? lmp.mem.floor : 0;
  _lm_trunk   = lmp.mem.trunk;
  _lm_streams = false;
  const int qbits = lmp.mem.raw ? _lm_quant_bits : 0;
  if (lmp.mem.raw && lmp.mem.floor > 0) {
    const auto sp = model_memory::plan_streaming(
        session(), lm_dir, codec_dir, model_memory::kStreamHeadroom,
        lmp.mem.retires, std::string_view{});
    _lm_streams = sp.stream;
    session()->info(fmt(
        "TextToSpeechStage('{}'): LM backbone {}{} -> {} MB held ({} MB "
        "on disk); {} GB with everything else resident + {} GB headroom "
        "vs {} GB RAM -> {}", this->id(),
        lmp.mem.pack  ? "is an affine w8 pack"
        : qbits == 8 ? "quantized in memory to w8"
                     : "kept bf16",
        qbits == 8 && lmp.variant == "delay" ? ", text embedding + head w8"
                                             : "",
        mb(lmp.mem.preload), mb(lmp.mem.disk), sp.footprint >> 30,
        model_memory::kStreamHeadroom >> 30, model_memory::phys_ram() >> 30,
        _lm_streams ? fmt("STREAM its layers (floor {} MB; residency grows "
                          "the kept set into what the box can hold)",
                          mb(lmp.mem.floor))()
                    : std::string("resident")));
  } else {
    session()->info(fmt(
        "TextToSpeechStage('{}'): LM is a quantized pack, held as it sits "
        "({} MB)", this->id(), mb(lmp.mem.disk)));
  }

  // 3. The LM.
  session()->info(fmt("TextToSpeechStage('{}'): loading {} LM from '{}'",
                      this->id(),
                      realtime ? "MOSS-TTS-Realtime"
                      : local  ? "MOSS-TTS-Local-v1.5"
                               : "MOSS-TTS",
                      _hf_dir));
  const auto t_lm0 = clock::now();
  genai::MetalQwenModel* backbone = nullptr;
  if (realtime) {
    genai::MetalMossRtModel::Config cfg;
    cfg.backbone = rt_qwen_cfg_(lm_dir, "language_model.", "model.", 28, 2048,
                                256);
    cfg.backbone.load_quant_bits = qbits;
    cfg.backbone.stream_decode   = _lm_streams;
    cfg.local    = rt_qwen_cfg_(lm_dir, "local_transformer.", "model.", 4, 32,
                                32);
    _lm_rt = genai::MetalMossRtModel::load(
        genai::open_weight_set(lm_dir, session()), mc, cfg);
    if (_lm_rt) { backbone = _lm_rt->backbone(); }
  } else if (local) {
    genai::MetalMossV15Model::Config cfg;
    cfg.backbone = v15_backbone_cfg_(lm_dir);
    cfg.backbone.load_quant_bits = qbits;
    cfg.backbone.stream_decode   = _lm_streams;
    _lm_v15 = genai::MetalMossV15Model::load(
        genai::open_weight_set(lm_dir, session()), mc, cfg);
    if (_lm_v15) { backbone = _lm_v15->backbone(); }
  } else {
    genai::MossTtsLoadOptions opts;
    opts.quant_bits = qbits;
    opts.stream     = _lm_streams;
    _lm = genai::MetalMossTtsModel::load(
        genai::open_weight_set(lm_dir, session()), mc, opts);
    if (_lm && !_lm->valid()) { _lm.reset(); }
  }
  if (!_lm && !_lm_v15 && !_lm_rt) {
    session()->error(fmt(
        "TextToSpeechStage('{}'): failed to load the MOSS LM from '{}'; the "
        "stage is inert", this->id(), _hf_dir));
    release_models_();
    co_return;
  }
  // The realtime / Local wrappers load their own tables after the backbone;
  // wire trunk-then-layers now (the delay model does this inside load()).
  if (backbone != nullptr) { backbone->wire_resident(); }
  // The int8 prefill tier, when asked for. set_i8_gemm(false) still
  // consults VPIPE_I8_GEMM, so the env override reaches every variant.
  bool i8_on = false;
  if (_lm) {
    _lm->set_i8_gemm(_i8_gemm);
    i8_on = _lm->i8_gemm_enabled();
  } else if (backbone != nullptr) {
    backbone->set_i8_gemm(_i8_gemm);
    i8_on = backbone->i8_gemm_enabled();
  }
  if (_i8_gemm && !i8_on) {
    session()->info(fmt(
        "TextToSpeechStage('{}'): i8_gemm asked for, but this GPU has no "
        "matrix cores (or the int8 kernels did not load); the backbone "
        "prefill stays dense", this->id()));
  }

  _tokenizer = genai::Tokenizer::from_huggingface_json(
      lm_dir + "/tokenizer.json", session());
  if (!_tokenizer) {
    session()->error(fmt(
        "TextToSpeechStage('{}'): no tokenizer.json in '{}'; inert",
        this->id(), lm_dir));
    release_models_();
    co_return;
  }

  // 4. Residency, for a streaming backbone. The reserve is what must stay
  // clear for what runs after a forward and has not allocated yet -- the
  // codec's decode of a chunk -- and is SET (growth stays off until it
  // is); the schedule is one forward per generated frame.
  if (_lm_streams) {
    const std::size_t reserve = (std::size_t)256 << 20;
    const int forwards = local || realtime ? _max_frames : _max_new_tokens;
    if (_lm) {
      _lm->set_residency_reserve(reserve);
      _lm->set_residency_schedule(forwards);
    } else if (backbone != nullptr) {
      backbone->set_residency_reserve(reserve);
      backbone->set_residency_schedule(forwards);
    }
  }

  if (_lm) {
    // Route profiling events (text-prefill / text-decode) onto the LLM lane.
    _lm->set_session(session());
    // Cold-start warmup: the FIRST forward pass pays the Metal
    // pipeline-state compilation + first-touch weight residency with the GPU
    // mostly idle. Run a tiny throwaway LM generation + codec decode HERE,
    // at load, so that cost lands during stage init -- the first real
    // synthesis then runs warm. Disable with VPIPE_TTS_NO_WARMUP for A/B.
    if (std::getenv("VPIPE_TTS_NO_WARMUP") == nullptr) {
      const auto t_w0 = clock::now();
      const int n_vq = _lm->config().n_vq;
      const int pad  = _lm->config().audio_pad_code;
      // A few rows: channel 0 a valid text id, audio channels at pad. The
      // exact ids are irrelevant -- this only compiles kernels + makes
      // weights resident; the generated output is discarded.
      std::vector<std::vector<std::int32_t>> wprompt(
          4, std::vector<std::int32_t>(static_cast<std::size_t>(1 + n_vq),
                                       pad));
      wprompt[0][0] = _lm->config().im_start;
      wprompt[1][0] = _lm->config().pad_token;
      wprompt[2][0] = _lm->config().pad_token;
      wprompt[3][0] = _lm->config().pad_token;
      (void)_lm->generate_delay_greedy(wprompt, 8);
      // Codec: a few zero-code frames warm the RVQ decode + transformers.
      std::vector<std::vector<std::int32_t>> wcodes(
          4, std::vector<std::int32_t>(static_cast<std::size_t>(n_vq), 0));
      (void)_codec->decode(wcodes, nullptr);
      session()->info(fmt(
          "TextToSpeechStage('{}'): warmup done in {} ms (cold "
          "pipeline-state + weight residency paid at load; first synthesis "
          "runs warm)", this->id(), ms(clock::now() - t_w0)));
    }
  }

  // 5. The plan was declared from the checkpoints; correct both ledgers
  // with what the models actually hold.
  revise_holdings_();
  std::size_t lm_held = 0;
  int kept = 0, layers = 0;
  if (_lm) {
    lm_held = _lm->resident_bytes();
    kept = _lm->resident_layer_count();
    layers = _lm->n_layers();
  } else if (backbone != nullptr) {
    lm_held = (std::size_t)backbone->resident_bytes() + _lm_trunk;
    kept = backbone->resident_layer_count();
    layers = backbone->config().n_layers;
  }
  session()->info(fmt(
      "TextToSpeechStage('{}'): ready ({} -- LM loaded in {} ms, holding {} "
      "MB{}{}; codec {} MB; language={}, text normalizer={})", this->id(),
      realtime ? "realtime, 24 kHz mono"
      : local  ? "Local-v1.5, 48 kHz stereo"
               : "delay pattern, 24 kHz mono",
      ms(clock::now() - t_lm0), mb(lm_held),
      qbits == 8 ? ", backbone w8 in memory" : "",
      _lm_streams ? fmt(", streaming: {}/{} layers resident so far", kept,
                        layers)()
                  : std::string(),
      mb(codec_held), _language,
      _robust_text ? "robust (MOSS-TTS-v1.5)" : "basic"));
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

// The BASIC text clean-up (text_normalizer "basic"; every checkpoint but
// MOSS-TTS-v1.5 under "auto"). Replace CR/CRLF with a space, drop ASCII
// control chars except spaces, collapse whitespace runs to a single
// space, and trim leading/trailing whitespace. The robust one is
// genai::moss_tts_normalize_text.
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

// The text clean-up `text_normalizer` resolved to: MOSS-TTS-v1.5's own
// normalizer, or the basic one above.
std::string
clean_text_(const std::string& raw, bool robust)
{
  return robust ? genai::moss_tts_normalize_text(raw) : normalize_text_(raw);
}

// The delay-pattern prompt ids, from the loaded model's config.json.
genai::MossDelayPromptIds
delay_prompt_ids_(const genai::MetalMossTtsModel::Config& c)
{
  genai::MossDelayPromptIds ids;
  ids.im_start        = c.im_start;
  ids.im_end          = c.im_end;
  ids.audio_start     = c.audio_start;
  ids.audio_end       = c.audio_end;
  ids.audio_user_slot = c.audio_user_slot;
  ids.n_vq            = c.n_vq;
  ids.audio_pad       = c.audio_pad_code;
  return ids;
}

// The MOSS processor's loudness_normalize, applied to a clone reference
// before it is encoded: scale toward -20 dBFS RMS, by at most +-3 dB.
void
loudness_normalize_(std::vector<float>& wav)
{
  if (wav.empty()) { return; }
  double sq = 0.0;
  for (float v : wav) { sq += static_cast<double>(v) * v; }
  const double dbfs = 10.0 * std::log10(sq / wav.size() + 1e-9);
  const double gain = std::clamp(-20.0 - dbfs, -3.0, 3.0);
  const float factor = static_cast<float>(std::pow(10.0, gain / 20.0));
  for (float& v : wav) { v *= factor; }
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
  const FFmpegLibraries* libs = session ? session->services()->ffmpeg_libraries() : nullptr;
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
  // wait_for_reference: the first utterance waits until the reference
  // voice is readable (or its port ended) -- peeked, not taken: each
  // variant's drain below reads it as it always does.
  if (_wait_reference && !_ref_waited && _with_encoder &&
      ctx.num_iports() >= 2 && ctx.iport_connected(1)) {
    _ref_waited = true;
    (void)co_await ctx.peek(1, 0);
  }

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Latch the two token-sampler specs off the OPTIONAL sampler iports 2
  // (audio codes) and 3 (free text), once each: a `sampler-select` source
  // emits a single beat at launch, which we cache and reuse for every later
  // utterance. Read AFTER the text beat so declared-but-unwired ports never
  // stall the first utterance. Unwired audio keeps MOSS's own sampling (set
  // in the ctor); unwired text stays greedy.
  if (!_audio_sp_read && ctx.num_iports() >= 3 && ctx.iport_connected(2)) {
    auto sb = co_await ctx.read(2);
    _audio_sp_read = true;      // beat consumed either way; never re-read
    const std::string who =
        "TextToSpeechStage('" + this->id() + "') audio";
    bool ok = false;
    const genai::SamplerParams p =
        sampler_params_from_beat(sb.get(), session(), who, &ok);
    // Only a GOOD beat replaces MOSS's own audio sampling. A rejected one
    // must not fall back to the helper's greedy return here -- greedy audio
    // degenerates into silence, which is the whole reason this channel has a
    // non-greedy default.
    if (ok) {
      _audio_sp = p;
      _audio_sp_set = true;
      warn_unsupported_knobs_(session(), who, _audio_sp);
    }
  }
  if (!_text_sp_read && ctx.num_iports() >= 4 && ctx.iport_connected(3)) {
    auto sb = co_await ctx.read(3);
    _text_sp_read = true;
    const std::string who = "TextToSpeechStage('" + this->id() + "') text";
    bool ok = false;
    const genai::SamplerParams p =
        sampler_params_from_beat(sb.get(), session(), who, &ok);
    if (ok) {
      _text_sp = p;
      warn_unsupported_knobs_(session(), who, _text_sp);
    }
  }

  // End-of-response gating for barge-in. A streaming text producer
  // (text-chat's `stream` oport) splits ONE reply into several beats and
  // marks only the LAST with end_of_response=true; the earlier chunks
  // carry end_of_response=false. Those non-final chunks are the SAME
  // reply continuing, so a newer beat waiting behind one is NOT a new
  // request -- it must be spoken in sequence, never cut. Only when the
  // beat we are speaking is end_of_response (a complete reply -- or a
  // legacy beat with no such field, which defaults true) does a newer
  // beat mean "new request": barge-in. So the cut is armed only for EOR
  // beats. A plain-string beat has no field and stays end_of_response, so
  // non-streaming producers keep the original barge-in behaviour.
  bool cur_is_eor = true;
  if (const auto* fdp0 = dynamic_cast<const FlexDataPayload*>(t.get())) {
    if (fdp0->data.is_object()) {
      auto obj = fdp0->data.as_object();
      if (obj.contains("end_of_response")) {
        cur_is_eor = obj.at("end_of_response").as_bool(true);
      }
    }
  }

  // Barge-in predicate: true once a NEW text beat is waiting on the text iport
  // (iport 0) while we're mid-utterance AND the current beat is the last of its
  // reply (cur_is_eor). When interrupt_on_new_text is set, the generators poll
  // this and stop ASAP (after flushing the audio produced so far); process()
  // then re-runs and serves the newer text. A mid-reply chunk (cur_is_eor
  // false) is finished in full so the reply plays continuously. The audio-ref
  // iport (iport 1) is deliberately excluded -- it is sticky + its own clock
  // domain.
  auto new_text_pending = [&]() {
    return _interrupt_on_new_text && cur_is_eor && ctx.backlog(0) > 0;
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
    const std::string rt_text = clean_text_(rtraw, _robust_text);
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
    // Audio codes MUST be sampled -- greedy degenerates into silence -- so an
    // unwired audio sampler iport keeps the realtime variant's OWN reference
    // defaults (temp 0.8 / top_p 0.6 / top_k 30 / rep_pen 1.1), which differ
    // from the 8B/v1.5 numbers. A wired sampler-select replaces them.
    const genai::SamplerParams rt_params =
        _audio_sp_set ? _audio_sp : moss_rt_audio_default_sampler();
    const genai::MossSampling rt_sp = moss_sampling_of(rt_params);
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
                                rt_params.seed, on_frame);
      flush();
    } else {
      frames = _lm_rt->generate(
          prompt_grid, text_ids, max_f, rt_sp, rt_params.seed,
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
    // Residency may have grown the backbone during the utterance.
    revise_holdings_();
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
    const std::string v15_text = clean_text_(raw, _robust_text);
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
    // Audio RVQ codes MUST be sampled (MOSS defaults temp 1.7 / top_p 0.8 /
    // top_k 25); greedy degenerates into silence after the first sentence.
    const genai::MossSampling v15_audio_sp = moss_sampling_of(_audio_sp);
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
                                 _audio_sp.seed, on_frame);
      flush();   // the final partial (< _stream_chunk) chunk
    } else {
      frames = _lm_v15->generate(
          grid, _max_frames, v15_audio_sp, _audio_sp.seed,
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
    // Residency may have grown the backbone during the utterance.
    revise_holdings_();
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

  const std::string text = clean_text_(in_text, _robust_text);
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
  // sets the cloned voice for this and subsequent text beats (sticky). Like
  // the reference processor: mixed down to mono, resampled to the codec
  // rate, loudness-normalized (-20 dBFS, +-3 dB), then encoded; capped to
  // voice_ref_seconds first. backlog() keeps the read non-blocking.
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
      std::vector<float> ref_pcm = build_mono_ref_wave_(
          session(), *tbp, _codec->sample_rate(), _voice_ref_seconds);
      if (ref_pcm.empty()) {
        session()->warn(fmt(
            "TextToSpeechStage('{}'): reference PCM produced no usable "
            "samples; ignoring", this->id()));
        continue;
      }
      loudness_normalize_(ref_pcm);
      auto rc = _codec->encode(ref_pcm);
      if (rc.empty()) {
        session()->warn(fmt(
            "TextToSpeechStage('{}'): codec encode of the reference "
            "produced 0 frames; ignoring", this->id()));
        continue;
      }
      _ref_codes = std::move(rc);
      _ref_set   = true;
      session()->info(fmt(
          "TextToSpeechStage('{}'): cloned reference voice -> {} codec "
          "frames", this->id(), static_cast<int>(_ref_codes.size())));
    }
  }

  // 1. Build the prompt grid [seq][1 + n_vq]: channel 0 = text/control id,
  // channels 1..n_vq = audio codes. A clone reference (from iport1 above, or
  // the voice_lock cache) splices into - Reference(s):; else None.
  genai::MossDelayUserFields fields;
  fields.instruction   = _instruction;
  fields.language      = _language;
  fields.quality       = _quality;
  fields.sound_event   = _sound_event;
  fields.ambient_sound = _ambient_sound;
  if (_duration_tokens > 0) {
    fields.tokens = std::to_string(_duration_tokens);
  }
  std::vector<std::vector<std::int32_t>> prompt = genai::moss_delay_build_grid(
      *_tokenizer, text, delay_prompt_ids_(_lm->config()), fields,
      _ref_set ? &_ref_codes : nullptr);
  if (prompt.empty()) {
    session()->warn(fmt(
        "TextToSpeechStage('{}'): empty prompt grid for the text; "
        "dropping beat", this->id()));
    co_return;
  }

  // 2. Sampled delay-pattern generation, de-delayed into codec frames as the
  // rows arrive. Separate audio + text sampling, each from its own sampler
  // iport. Greedy audio (temperature <= 0) degenerates into silent loops, so
  // the audio channel falls back to the MossTTSDelay-8B recommendation
  // rather than to argmax; the text channel does fall back to greedy (it
  // must follow the transcript).
  const genai::MossSampling audio_sp = moss_sampling_of(_audio_sp);
  const genai::MossSampling text_sp  = moss_sampling_of(_text_sp);
  const int sr = _codec->sample_rate();
  auto make_pcm_beat = [&](const std::vector<float>& wav) {
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = { static_cast<std::int64_t>(wav.size()) };
    tb.resize_contiguous(wav.size());
    std::memcpy(tb.as_f32(), wav.data(), wav.size() * sizeof(float));
    tb.sideband = FlexData::make_object();
    tb.sideband.as_object().insert("sample_rate", FlexData::make_int(sr));
    return tb;
  };

  // Frame f's codebook cb is generated in row f + cb, so a frame is whole
  // n_vq - 1 rows after its first code: the de-delay completes frames on
  // the fly, the codec streams them out in stream_chunk_frames chunks
  // (bit-identical to decoding the whole utterance at once), and the first
  // audio leaves ~n_vq + one chunk of rows after speech starts.
  genai::MossDelayDedelay dedelay(n_vq, pad);
  std::vector<std::vector<std::int32_t>> codes;   // every frame, in order
  std::vector<std::int32_t> frame;
  std::int64_t total_samps = 0;
  double peak = 0.0;
  bool open = true;
  std::size_t streamed = 0;                       // frames sent to the codec
  auto note_peak = [&](const std::vector<float>& wav) {
    for (float v : wav) {
      peak = std::max(peak, static_cast<double>(std::fabs(v)));
    }
  };
  auto flush = [&]() {
    if (!open || streamed >= codes.size()) { return; }
    std::vector<std::vector<std::int32_t>> chunk(
        codes.begin() + static_cast<std::ptrdiff_t>(streamed), codes.end());
    streamed = codes.size();
    std::vector<float> pcm = _codec->decode_stream_chunk(*_stream_v1, chunk);
    if (pcm.empty()) { return; }
    total_samps += static_cast<std::int64_t>(pcm.size());
    note_peak(pcm);
    open = ctx.write_sync(
        0, make_payload<TensorBeatPayload>(make_pcm_beat(pcm)));
    if (open) { ++_clips_emitted; }
  };
  const bool streaming = _stream_chunk > 0;
  if (streaming) {
    // Reuse the cached ring state across beats; reallocate only when the
    // chunk size or the active-codebook count changed.
    if (!_stream_v1 || _stream_v1->max_chunk != _stream_chunk ||
        _stream_v1->n_active != 0) {
      _stream_v1 = _codec->decode_stream_begin(_stream_chunk);
    } else {
      _stream_v1->reset();
    }
    open = (_stream_v1 != nullptr);
  }
  // Progress: the speech so far, in seconds of it -- a fraction of the
  // duration asked for, when one was (12.5 frames a second).
  UiProgress bar = session()->open_progress("speech");
  const double fps = sr / 1920.0;
  auto gen = _lm->generate_delay(
      prompt, _max_new_tokens, audio_sp, text_sp, _audio_sp.seed,
      [&](const std::vector<std::int32_t>& row) {
        if (dedelay.push(row, frame)) {
          codes.push_back(frame);
          const auto n = static_cast<std::uint64_t>(codes.size());
          bar.update(_duration_tokens > 0 ? std::min<std::uint64_t>(
                                                n, _duration_tokens)
                                          : 0,
                     static_cast<std::uint64_t>(_duration_tokens),
                     fmt("speaking: {:.1f} s",
                         static_cast<double>(n) / fps)());
          if (streaming &&
              static_cast<int>(codes.size() - streamed) >= _stream_chunk) {
            flush();
          }
        }
        // Barge-in: stop on new text; also stop once downstream closed,
        // or the graph is asked to stop.
        return (!streaming || open) && !new_text_pending() &&
               !ctx.stop_requested();
      });
  if (ctx.stop_requested()) { co_return; }
  if (streaming) { flush(); }   // the final partial chunk
  const auto t_gen = clock::now();
  const int Gg = static_cast<int>(gen.size());
  if (codes.empty()) {
    session()->warn(fmt(
        "TextToSpeechStage('{}'): no non-pad audio frames after de-delay "
        "({} generated rows); emitting nothing", this->id(), Gg));
    co_return;
  }

  // 3. voice_lock (design-once): cache the FIRST generated voice and reuse it
  // as the clone reference for later beats, so the timbre stays consistent
  // across texts. An external iport reference (above) takes precedence (it
  // sets _ref_set), so this only fires until a voice is locked.
  if (_voice_lock && !_ref_set) {
    _ref_codes = codes;
    if (_voice_ref_seconds > 0.0) {
      const std::size_t cap = static_cast<std::size_t>(
          _voice_ref_seconds * sr / 1920.0);
      if (cap >= 1 && _ref_codes.size() > cap) { _ref_codes.resize(cap); }
    }
    _ref_set = true;
    session()->info(fmt(
        "TextToSpeechStage('{}'): voice_lock engaged -- {} frames cached as "
        "the reference for subsequent beats", this->id(),
        static_cast<int>(_ref_codes.size())));
  }

  // 4. One-shot: decode the whole utterance and emit a single PCM beat.
  if (!streaming) {
    bar.update(0, 0, "decoding");
    std::vector<float> wave = _codec->decode(codes, nullptr);
    if (wave.empty()) {
      session()->warn(fmt(
          "TextToSpeechStage('{}'): codec decode produced 0 PCM samples; "
          "emitting nothing", this->id()));
      co_return;
    }
    total_samps = static_cast<std::int64_t>(wave.size());
    note_peak(wave);
    ++_clips_emitted;
    co_await ctx.write(0, make_payload<TensorBeatPayload>(make_pcm_beat(wave)));
  }
  const auto t_done = clock::now();

  auto ms = [](clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  const std::string mode = streaming
      ? fmt("streamed {}-frame chunks", _stream_chunk)()
      : fmt("{} ms one-shot decode", static_cast<int>(ms(t_done - t_gen)))();
  session()->info(fmt(
      "TextToSpeechStage('{}'): {} chars -> {} prompt rows{} -> {} gen rows "
      "-> {} frames -> {} PCM samples = {:.2f}s @ {} Hz, peak={:.3f} "
      "({} ms gen, {})",
      this->id(),
      static_cast<int>(text.size()),
      static_cast<int>(prompt.size()),
      _ref_set ? " (cloned)" : "", Gg,
      static_cast<int>(codes.size()), total_samps,
      total_samps / static_cast<double>(sr), sr, peak,
      static_cast<int>(ms(t_gen - t_start)), mode));
  // Residency may have grown the backbone during the utterance, and K/V
  // is now what the pools grew to.
  revise_holdings_();
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
