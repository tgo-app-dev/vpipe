#include "stages/text-to-speech-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/tensor-beat.h"
#include "generative-models/moss/metal-moss-tts-model.h"
#include "generative-models/moss/metal-moss-codec.h"
#include "generative-models/tokenizer.h"
#include <cctype>
#include <chrono>
#include <cstring>
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
   .doc = "MOSS-TTS LM model: a models-DB key (registered by model-fetch) "
          "or an HF-style model dir; a DB key wins over a same-named path",
   .suggest_db = "models", .suggest_db_type = "moss-tts"},
  {.key = "codec_dir", .type = ConfigType::String, .required = true,
   .doc = "MOSS-Audio-Tokenizer (codec) model: a models-DB key or a "
          "filesystem path; a DB key wins over a same-named path",
   .suggest_db = "models", .suggest_db_type = "moss-codec"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db model-fetch registers into", .def_str = "models"},
  {.key = "max_new_tokens", .type = ConfigType::Int,
   .doc = "per-beat delay-pattern generation budget (>= 1)",
   .def_int = 1024},
};
const PortSpec kIports[] = {
  {.name = "text",
   .doc = "FlexData string (or object with a \"text\" key): the text to "
          "synthesize",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "pcm",
   .doc = "TensorBeat f32 [n_samples] mono PCM @ 24 kHz "
          "(sideband.sample_rate set); downstream optional",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "text-to-speech",
  .doc       = "Text-to-speech (MOSS-TTS, metal): synthesizes each input "
               "text beat into a 24 kHz mono PCM waveform (delay-pattern "
               "code generation + MOSS Audio Tokenizer decode) and emits "
               "it as a TensorBeat.",
  .display_name = "Speak",
  .category  = StageCategory::Audio,
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

  session()->info(fmt(
      "TextToSpeechStage('{}'): loading MOSS-TTS LM from '{}'",
      this->id(), _hf_dir));
  _lm = genai::MetalMossTtsModel::load(lm_dir, mc);
  if (!_lm || !_lm->valid()) {
    session()->error(fmt(
        "TextToSpeechStage('{}'): failed to load MOSS-TTS LM from '{}'; "
        "the stage is inert", this->id(), _hf_dir));
    _lm.reset();
    co_return;
  }

  session()->info(fmt(
      "TextToSpeechStage('{}'): loading MOSS-Audio-Tokenizer codec from "
      "'{}'", this->id(), _codec_dir));
  _codec = genai::MetalMossCodec::load(codec_dir, mc);
  if (!_codec || !_codec->valid()) {
    session()->error(fmt(
        "TextToSpeechStage('{}'): failed to load MOSS codec from '{}'; "
        "the stage is inert", this->id(), _codec_dir));
    _lm.reset();
    _codec.reset();
    co_return;
  }

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

  // 1. Build the prompt grid [seq][1 + n_vq]: channel 0 = text id,
  // channels 1..n_vq = audio_pad_code (== no reference audio).
  const int n_vq      = _lm->config().n_vq;             // 32
  const int pad       = _lm->config().audio_pad_code;   // 1024
  const int n_chan    = 1 + n_vq;                       // 33
  const std::string prompt_str = render_moss_prompt_(text);
  std::vector<std::int32_t> ids = encode_moss_prompt_(*_tokenizer, prompt_str);
  if (ids.empty()) {
    session()->warn(fmt(
        "TextToSpeechStage('{}'): tokenizer produced 0 ids for the "
        "prompt; dropping beat", this->id()));
    co_return;
  }
  std::vector<std::vector<std::int32_t>> prompt(
      ids.size(), std::vector<std::int32_t>(
                      static_cast<std::size_t>(n_chan), pad));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    prompt[i][0] = ids[i];   // channels 1..n_vq stay at audio_pad_code
  }

  // 2. Greedy delay-pattern generation -> [G][1 + n_vq].
  auto gen = _lm->generate_delay_greedy(prompt, _max_new_tokens);
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
      "TextToSpeechStage('{}'): {} chars -> {} prompt ids -> {} gen rows "
      "-> {} frames -> {} PCM samples = {:.2f}s @ {} Hz, peak={:.3f} "
      "({} ms gen + {} ms decode)",
      this->id(),
      static_cast<int>(text.size()),
      static_cast<int>(ids.size()), Gg,
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
