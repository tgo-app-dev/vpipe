#include "stages/audio-transcribe-stage.h"

#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/chat-template.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/loaded-language-model.h"
#include "generative-models/sampler.h"
#include "generative-models/tokenizer.h"
#include "apple-silicon/tensor-beat.h"
#include <algorithm>
#include <cstdlib>   // setenv (metal backend selection)
#endif
#if defined(VPIPE_BUILD_APPLE_SILICON)
#include "generative-models/qwen3/metal-audio-encoder.h"
#endif

#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace vpipe {

AudioTranscribeStage::AudioTranscribeStage(
    const SessionContextIntf* s,
    std::string               id,
    std::vector<InEdge>       iports,
    FlexData                  config)
  : TypedStage<AudioTranscribeStage>(s, std::move(id), std::move(iports),
                                     std::move(config))
{
  // Streaming mode is selected purely from the wired graph: when the
  // pipeline spec gives us two iports (PCM + segment markers) we
  // operate in streaming mode for the life of the stage. With one
  // iport (the legacy default) every beat is a complete clip.
  _streaming = this->iport_edges().size() >= 2;
  // Construction must succeed for any config (see Stage::fail_config):
  // a stage must construct so a graph can be built/edited before
  // required fields are supplied. Config problems are recorded via
  // fail_config (first message wins) and deferred to launch.
  // Attribute defaults live in kSpec.attrs; attr_* resolves the
  // configured value else that default.
  allocate_oports(spec().oports.size());   // sink: 0 oports

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
          "AudioTranscribeStage('{}'): max_pages must be >= 1 (got {})",
          this->id(), v));
    }
    _max_pages = static_cast<uint32_t>(v);
  }
  {
    int64_t v = attr_int("max_new_tokens");
    if (v < 1) {
      fail_config(fmt(
          "AudioTranscribeStage('{}'): max_new_tokens must be >= 1 "
          "(got {})", this->id(), v));
    }
    _max_new_tokens = static_cast<int>(v);
  }
  {
    int64_t v = attr_int("sample_rate");
    if (v < 1) {
      fail_config(fmt(
          "AudioTranscribeStage('{}'): sample_rate must be >= 1 (got {})",
          this->id(), v));
    }
    _sample_rate = static_cast<int>(v);
  }
  _language_hint = attr_str("language_hint");
  {
    double v = attr_real("pcm_buffer_s");
    if (v <= 0.0 || v > 300.0) {
      fail_config(fmt(
          "AudioTranscribeStage('{}'): pcm_buffer_s {} outside (0, 300]",
          this->id(), v));
    }
    _pcm_buffer_s = v;
  }
  _late_marker_skip = attr_bool("late_marker_skip");
#ifdef VPIPE_BUILD_APPLE_SILICON
  // Optional "sampler" sub-object (composite; no flat ConfigKey form) --
  // same schema as visual-qa / realtime-vqa / text-chat. Empty / absent
  // => greedy argmax (Sampler::is_argmax() returns true). When present,
  // every decode step host-pulls the logits and runs the configured
  // temperature / top_k / top_p / min_p / repetition_penalty /
  // presence_penalty / seed combo via Sampler::sample().
  {
    const FlexData& cfg = this->config();
    if (cfg.is_object()) {
      auto root = cfg.as_object();
      if (root.contains("sampler")) {
        _sampler_params = genai::parse_sampler_config(root.at("sampler"));
      }
    }
  }
#endif

  if (_hf_dir.empty()) {
    fail_config(fmt(
        "AudioTranscribeStage('{}'): config.hf_dir is required "
        "(non-empty string)", this->id()));
  }
  if (_streaming) {
    // Reserve enough room for the rolling PCM buffer up-front; the
    // ingest path will only realloc when a beat exceeds the cap.
    _pcm_buf.reserve(
        static_cast<std::size_t>(_pcm_buffer_s * _sample_rate + 1.0));
  }
}

namespace {
constexpr ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "ASR-LM model: a models-DB key (the huggingface.co path "
          "registered by model-fetch) or a filesystem path. A DB key "
          "wins over a same-named path.",
   .suggest_db = "models"},
  {.key = "models_db", .type = ConfigType::String,
   .doc = "LMDB sub-db model-fetch registers into", .def_str = "models"},
  {.key = "compute_dtype", .type = ConfigType::String,
   .doc = "bf16 | f16 | f32", .def_str = "f16"},
  {.key = "page_tokens", .type = ConfigType::Int,
   .doc = "ContextManager K/V page size", .def_int = 512},
  {.key = "max_pages", .type = ConfigType::Int,
   .doc = "per-LM page pool capacity (>= 1)", .def_int = 16},
  {.key = "max_new_tokens", .type = ConfigType::Int,
   .doc = "per-clip generation budget (>= 1)", .def_int = 256},
  {.key = "sample_rate", .type = ConfigType::Int,
   .doc = "fallback Hz when sideband.sample_rate absent (>= 1)",
   .def_int = 16000},
  {.key = "language_hint", .type = ConfigType::String,
   .doc = "non-empty pre-emits 'language X' so only transcript gens",
   .def_str = ""},
  {.key = "pcm_buffer_s", .type = ConfigType::Real,
   .doc = "streaming rolling PCM buffer length s, (0,300]",
   .def_real = 30.0},
  {.key = "late_marker_skip", .type = ConfigType::Bool,
   .doc = "drop segment markers whose start is older than the buffer",
   .def_bool = true},
  {.key = "sampler", .type = ConfigType::Object,
   .doc = "decode sampler knobs (temperature/top_k/top_p/...)"},
};
const PortSpec kIports[] = {
  {.name = "audio", .doc = "mono F32 PCM TensorBeat [N] or [1,N]; one clip "
                           "per beat in block mode, a continuous PCM stream "
                           "in streaming mode; sideband.sample_rate honoured",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "segments", .doc = "OPTIONAL FlexData utterance markers "
                              "{start_us,end_us,index,is_partial} (e.g. from "
                              "audio-segment); connecting it switches the "
                              "stage to streaming mode (slice the rolling "
                              "PCM buffer per marker)",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "transcript", .doc = "FlexData {text, start_us, end_us} per "
                                "transcribed clip (start_us/end_us present in "
                                "streaming mode only). Optional -- unconnected "
                                "is fine (the transcript still logs).",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "audio-transcribe",
  .doc       = "Transcribes each incoming PCM clip with a Qwen3-ASR language "
               "model (encoder + greedy/sampled decode), logs the transcript "
               "via the UI delegate, and emits it as FlexData {text[, "
               "start_us, end_us]} on oport 0.",
  .display_name = "Transcribe",
  .category  = StageCategory::Audio,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};
}  // namespace

const StageSpec&
AudioTranscribeStage::spec() const noexcept
{
  return kSpec;
}

AudioTranscribeStage::~AudioTranscribeStage() = default;

std::string
AudioTranscribeStage::resolve_model_dir(const std::string& ref) const
{
  return vpipe::resolve_model_dir(session(), _models_db, ref);
}

Job
AudioTranscribeStage::initialize(RuntimeContext& /*ctx*/)
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_hf_dir.empty()) {
    co_return;  // ctor already errored.
  }
  // No-MLX build: transcription runs on the metal-compute backend.
  ::setenv("VPIPE_LLM_BACKEND", "metal", 1);
  auto* mgr = session()->generative_model_manager();
  if (!mgr) {
    session()->error(fmt(
        "AudioTranscribeStage('{}'): no GenerativeModelManager on this session "
        "(build was compiled without VPIPE_BUILD_APPLE_SILICON)",
        this->id()));
    co_return;
  }
  genai::LoadSpec spec;
  spec.hf_dir        = resolve_model_dir(_hf_dir);
  spec.compute_dtype = _compute_dtype;
  spec.page_tokens   = _page_tokens;
  spec.max_pages     = _max_pages;
  _lm = mgr->load(spec);
  if (!_lm || !_lm->valid()) {
    session()->error(fmt(
        "AudioTranscribeStage('{}'): LM load failed for '{}'",
        this->id(), _hf_dir));
    _lm.reset();
    co_return;
  }
  _m_audio = _lm->metal_audio_encoder();
  if (!_m_audio) {
    session()->error(fmt(
        "AudioTranscribeStage('{}'): [metal] architecture '{}' has no "
        "metal audio encoder (loaded a non-audio model, or the metal "
        "backend isn't active)", this->id(),
        _lm->config().architecture));
    _lm.reset();
    co_return;
  }
  const int native_sr = _m_audio->config().sample_rate;
  // Shared: the ASR chat template must expose an audio_pad token.
  if (!_lm->chat_template()
      || _lm->chat_template()->audio_pad_token_id() < 0) {
    session()->error(fmt(
        "AudioTranscribeStage('{}'): LM '{}' chat template has no "
        "audio_pad_token_id; cannot render ASR prompts",
        this->id(), _lm->config().architecture));
    _lm.reset();
    co_return;
  }
  session()->info(fmt(
      "AudioTranscribeStage('{}'): model ready ({} layers, vocab={}, "
      "audio_sr={}{}{}{})",
      this->id(), _lm->config().n_layers, _lm->config().vocab_size,
      native_sr,
      _language_hint.empty() ? "" : ", language=" + _language_hint,
      _streaming ? ", streaming" : ", block-mode",
      _streaming
          ? ", pcm_buffer_s=" + std::to_string(_pcm_buffer_s)
          : std::string()));
#else  // !VPIPE_BUILD_APPLE_SILICON
  session()->error(fmt(
      "AudioTranscribeStage('{}'): this build was compiled without "
      "VPIPE_BUILD_APPLE_SILICON; the LLM subsystem is unavailable",
      this->id()));
#endif
  co_return;
}

Job
AudioTranscribeStage::process(RuntimeContext& ctx)
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  const bool encoder_ready = _lm && _m_audio;
  if (!encoder_ready) {
    if (!_encoder_unavailable_warned) {
      session()->warn(fmt(
          "AudioTranscribeStage('{}'): LM / encoder unavailable; dropping "
          "all incoming audio beats", this->id()));
      _encoder_unavailable_warned = true;
    }
    // Drain whatever we get so upstream backpressure clears. In
    // streaming mode that's iport0 PCM + iport1 segment markers.
    std::vector<unsigned> wait_ports;
    if (!ctx.eos(0)) { wait_ports.push_back(0); }
    if (_streaming && !ctx.eos(1)) { wait_ports.push_back(1); }
    if (wait_ports.empty()) { ctx.signal_done(); co_return; }
    co_await ctx.read_any(std::move(wait_ports));
    for (unsigned port : { 0u, 1u }) {
      if (port == 1u && !_streaming) { break; }
      const std::uint32_t n = ctx.backlog(port);
      for (std::uint32_t i = 0; i < n; ++i) {
        auto p = co_await ctx.read(port);
        if (!p) { break; }
      }
    }
    co_return;
  }

  if (_streaming) {
    // Wait until either iport has something (or closes). A closed
    // iport is reported perpetually "readable" by read_any so it
    // doesn't strand the wait; eos() is true only once it's both
    // closed AND drained, so dropping it from the wait set never
    // abandons buffered beats.
    std::vector<unsigned> wait_ports;
    if (!ctx.eos(0)) { wait_ports.push_back(0); }
    if (!ctx.eos(1)) { wait_ports.push_back(1); }
    if (wait_ports.empty()) {
      ctx.signal_done();
      co_return;
    }
    co_await ctx.read_any(std::move(wait_ports));

    // 1) Ingest every PCM beat currently available. We always pull
    // PCM before markers so a marker arriving in the same wake sees a
    // buffer that's at least as up-to-date as the marker's claim.
    const std::uint32_t n_pcm = ctx.backlog(0);
    for (std::uint32_t i = 0; i < n_pcm; ++i) {
      auto pp = co_await ctx.read(0);
      if (!pp) {
        // iport0 EOS: don't end -- iport1 may still deliver markers
        // for already-buffered audio. The outer wait_ports check
        // handles "both at EOS -> done".
        break;
      }
      const auto* tbp = dynamic_cast<const TensorBeatPayload*>(pp.get());
      if (!tbp) {
        session()->warn(fmt(
            "AudioTranscribeStage('{}'): expected TensorBeatPayload on "
            "iport 0, got {}; dropping beat",
            this->id(), pp->describe()));
        continue;
      }
      ingest_pcm_beat_(*tbp);
    }

    // 2) Drain segment markers and slice the buffer for each.
    const std::uint32_t n_seg = ctx.backlog(1);
    for (std::uint32_t i = 0; i < n_seg; ++i) {
      auto sp = co_await ctx.read(1);
      if (!sp) { break; }
      const auto* fp = dynamic_cast<const FlexDataPayload*>(sp.get());
      if (!fp || !fp->data.is_object()) {
        session()->warn(fmt(
            "AudioTranscribeStage('{}'): expected a FlexData segment "
            "marker on iport 1, got {}; dropping beat",
            this->id(), sp->describe()));
        continue;
      }
      const FlexData& fd = fp->data;
      auto obj = fd.as_object();
      if (!obj.contains("start_us") || !obj.contains("end_us")) {
        session()->warn(fmt(
            "AudioTranscribeStage('{}'): segment marker missing "
            "start_us/end_us; dropping beat", this->id()));
        continue;
      }
      ++_segments_seen;
      slice_and_transcribe_(obj.at("start_us").as_uint(0),
                            obj.at("end_us").as_uint(0));
    }
    for (auto& fd : _out_pending) {
      co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(fd)));
    }
    _out_pending.clear();
    co_return;
  }

  // ---- Block mode (single iport) -------------------------------------
  auto p = co_await ctx.read(0);
  if (!p) {
    ctx.signal_done();
    co_return;
  }
  const bool trace =
      std::getenv("VPIPE_AUDIO_TRANSCRIBE_TRACE") != nullptr;
  if (trace) {
    session()->log_debug(fmt(
        "AudioTranscribeStage('{}'): process() entered; got beat: {}",
        this->id(), p->describe()));
  }
  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(p.get());
  if (!tbp) {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): expected TensorBeatPayload on in-port "
        "0, got {}; dropping beat", this->id(), p->describe()));
    co_return;
  }
  if (tbp->dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): expected TensorBeat dtype=F32 (mono "
        "PCM samples), got {}; dropping beat",
        this->id(), TensorBeat::name_of(tbp->dtype)));
    co_return;
  }
  // Accept shape [n_samples] or [1, n_samples].
  std::size_t n_samples = 0;
  if (tbp->shape.size() == 1) {
    n_samples = static_cast<std::size_t>(tbp->shape[0]);
  } else if (tbp->shape.size() == 2 && tbp->shape[0] == 1) {
    n_samples = static_cast<std::size_t>(tbp->shape[1]);
  } else {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): expected TensorBeat shape [N] or "
        "[1,N], got rank={} dim0={}; dropping beat",
        this->id(),
        static_cast<int>(tbp->shape.size()),
        tbp->shape.empty() ? -1 : static_cast<int>(tbp->shape[0])));
    co_return;
  }
  if (n_samples == 0) {
    co_return;
  }

  int sr = _sample_rate;
  if (tbp->sideband.is_object()) {
    auto sb = tbp->sideband.as_object();
    if (sb.contains("sample_rate")) {
      sr = static_cast<int>(sb.at("sample_rate").as_int(_sample_rate));
    }
  }
  AlignedVector<std::uint8_t> scratch;
  const float* pcm;
  if (tbp->is_contiguous()) {
    pcm = tbp->as_f32();
  } else {
    scratch = tbp->materialize_contiguous();
    if (scratch.size() < n_samples * sizeof(float)) {
      session()->warn(fmt(
          "AudioTranscribeStage('{}'): materialized buffer too small "
          "({} bytes vs {} samples * 4); dropping beat",
          this->id(), scratch.size(), n_samples));
      co_return;
    }
    pcm = reinterpret_cast<const float*>(scratch.data());
  }
  _cur_has_ts = false;   // block mode: one clip per beat, no timestamp span
  transcribe_pcm_(pcm, n_samples, sr);
  ++_clips_processed;
  for (auto& fd : _out_pending) {
    co_await ctx.write(0, make_payload<FlexDataPayload>(std::move(fd)));
  }
  _out_pending.clear();
#else
  (void)ctx;
  co_return;
#endif
  co_return;
}

Job
AudioTranscribeStage::drain(RuntimeContext& /*ctx*/)
{
  co_return;
}


#if defined(VPIPE_BUILD_APPLE_SILICON)

bool
AudioTranscribeStage::m_transcribe_one_(const float* pcm,
                                        std::size_t  n_samples,
                                        int          sample_rate)
{
  if (!_lm || !_m_audio) { return false; }
  using Clock = std::chrono::steady_clock;
  const auto t_start = Clock::now();

  // 1. Encode the audio (metal-compute, host f32). The metal encoder
  // has no session handle, so time it from the stage on the audio lane.
  auto r = [&] {
    PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmAudio,
                       kPerfLlmAudioBegin,
                       static_cast<std::uint64_t>(n_samples));
    return _m_audio->encode(pcm, n_samples, sample_rate);
  }();
  if (r.n_tokens <= 0 || r.embeddings.empty()) {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): [metal] audio encoder produced 0 "
        "tokens for {:.2f} s clip @ {} Hz; dropping", this->id(),
        static_cast<double>(n_samples) / std::max(1, sample_rate),
        sample_rate));
    return false;
  }
  const int n_audio_tokens = r.n_tokens;
  const auto t_enc_end = Clock::now();

  // 2. Render the ASR prompt; weave AudioTokens refs into the stream.
  auto* tpl = _lm->chat_template();
  const std::int32_t audio_pad = tpl->audio_pad_token_id();
  std::vector<std::int32_t> ids;
  if (!tpl->render_asr_prompt(/*system_prompt=*/"", n_audio_tokens,
                              _language_hint, &ids)) {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): [metal] chat template refused to "
        "render the ASR prompt; dropping clip", this->id()));
    return false;
  }
  std::vector<genai::TokenRef> refs;
  refs.reserve(ids.size());
  int audio_off = 0;
  for (std::int32_t id : ids) {
    genai::TokenRef ref;
    if (id == audio_pad) {
      ref.kind               = genai::TokenRef::Kind::AudioTokens;
      ref.embeddings_host    = &r.embeddings;
      ref.audio_token_offset = audio_off++;
    } else {
      ref.kind    = genai::TokenRef::Kind::Text;
      ref.text_id = id;
    }
    refs.push_back(ref);
  }

  // 3. Multimodal prefill (empty image_grids => plain 1-D RoPE).
  auto lm_ctx = _lm->make_context();
  if (!lm_ctx.valid()) {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): [metal] failed to create LM context",
        this->id()));
    return false;
  }
  std::int32_t tok = _lm->prefill_multimodal_metal(
      lm_ctx, std::span<const genai::TokenRef>(refs),
      std::span<const std::pair<int, int>>());
  if (tok < 0) {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): [metal] prefill_multimodal_metal "
        "failed", this->id()));
    return false;
  }
  const auto t_prefill_end = Clock::now();

  // 4. Decode: greedy argmax, or host-sampled when a sampler knob is set.
  genai::Sampler sampler(_sampler_params);
  if (!sampler.is_argmax()) {
    sampler.prime(std::span<const std::int32_t>(&tok, 1));
  }
  std::vector<std::int32_t> gen = { tok };
  int n_decoded = 1;
  // GPU-resident pipelined decode: on-device embed + argmax/sampling, with
  // token i+1's forward committed before detokenize so the GPU overlaps the
  // host. No text prompt here, so the penalty seen-set is primed from the
  // first token only (matching the host sampler.prime above). Falls back to
  // the synchronous loop when the backend can't pipeline.
  const std::span<const std::int32_t> no_prompt;
  const bool pipelined =
      _lm->pdecode_begin(lm_ctx, tok, no_prompt, _sampler_params,
                         _max_new_tokens);
  if (pipelined) {
    bool committed = !tpl->is_stop_token(tok)
        ? _lm->pdecode_commit(lm_ctx) : false;
    // Run-ahead: second speculative forward (rollback-capable backends only)
    // so two stay in flight, hiding the per-token encode bubble; pdecode_end
    // discards any speculative tail past a stop token.
    const bool runahead = _lm->pdecode_supports_runahead();
    if (runahead && committed && _max_new_tokens > 1) {
      _lm->pdecode_commit(lm_ctx);
    }
    for (int i = 1; i < _max_new_tokens; ++i) {
      if (tpl->is_stop_token(tok)) { break; }
      if (!committed) { break; }
      const std::int32_t next = _lm->pdecode_next(lm_ctx);
      if (next < 0) { break; }
      const bool cont = (i + 1 < _max_new_tokens) && !tpl->is_stop_token(next);
      committed = cont ? _lm->pdecode_commit(lm_ctx) : false;
      if (tpl->is_stop_token(next)) { break; }
      gen.push_back(next);
      ++n_decoded;
      tok = next;
    }
    _lm->pdecode_end(lm_ctx);
  } else
  for (int i = 1; i < _max_new_tokens; ++i) {
    if (tpl->is_stop_token(tok)) { break; }
    std::int32_t next;
    if (sampler.is_argmax()) {
      // Greedy: fold embed + argmax into the decode command buffer and
      // skip the per-token host logit pull (last_logits_host() unused).
      next = _lm->next_token_greedy(lm_ctx, tok);
    } else {
      const std::int32_t am = _lm->next_token(lm_ctx, tok);
      if (am < 0) { break; }
      const auto& lh = _lm->last_logits_host();
      next = lh.empty()
          ? am
          : sampler.sample(std::span<const float>(lh.data(), lh.size()));
    }
    if (next < 0) { break; }
    if (tpl->is_stop_token(next)) { break; }
    gen.push_back(next);
    ++n_decoded;
    tok = next;
  }
  while (!gen.empty() && tpl->is_stop_token(gen.back())) {
    gen.pop_back();
  }
  const auto text = _lm->tokenizer().decode(
      std::span<const std::int32_t>(gen.data(), gen.size()));
  const auto t_decode_end = Clock::now();

  auto ms = [](Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  session()->info(fmt("[{}] {}", this->id(), text));
  // Queue the transcript for oport 0 (process() drains it). {text[, start_us,
  // end_us]} -- the span is present only in streaming mode.
  {
    FlexData out = FlexData::make_object();
    auto o = out.as_object();
    o.insert("text", FlexData::make_string(text));
    if (_cur_has_ts) {
      o.insert("start_us", FlexData::make_uint(_cur_start_us));
      o.insert("end_us", FlexData::make_uint(_cur_end_us));
    }
    _out_pending.push_back(std::move(out));
  }
  session()->log_verbose(fmt(
      "AudioTranscribeStage('{}'): [metal] {:.2f} s clip -> {} audio "
      "tokens, {} decoded ({} ms encode + {} ms prefill + {} ms "
      "decode){}", this->id(),
      static_cast<double>(n_samples) / std::max(1, sample_rate),
      n_audio_tokens, n_decoded,
      static_cast<int>(ms(t_enc_end - t_start)),
      static_cast<int>(ms(t_prefill_end - t_enc_end)),
      static_cast<int>(ms(t_decode_end - t_prefill_end)),
      sampler.is_argmax() ? "" : " [sampled]"));
  return true;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

#ifdef VPIPE_BUILD_APPLE_SILICON

bool
AudioTranscribeStage::transcribe_pcm_(const float* pcm,
                                      std::size_t  n_samples,
                                      int          sample_rate)
{
  return m_transcribe_one_(pcm, n_samples, sample_rate);
}

bool
AudioTranscribeStage::ingest_pcm_beat_(const TensorBeatPayload& tbp)
{
  if (tbp.dtype != TensorBeat::DType::F32) {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): streaming iport0 expected F32 PCM, "
        "got {}; dropping beat",
        this->id(), TensorBeat::name_of(tbp.dtype)));
    return false;
  }
  std::size_t n = 0;
  if (tbp.shape.size() == 1) {
    n = static_cast<std::size_t>(tbp.shape[0]);
  } else if (tbp.shape.size() == 2 && tbp.shape[0] == 1) {
    n = static_cast<std::size_t>(tbp.shape[1]);
  } else {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): streaming iport0 expected [N] or "
        "[1,N], got rank={}; dropping beat",
        this->id(), static_cast<int>(tbp.shape.size())));
    return false;
  }
  if (n == 0) { return true; }

  int beat_sr = _sample_rate;
  std::uint64_t beat_ts_us = 0;
  bool beat_has_ts = false;
  if (tbp.sideband.is_object()) {
    auto sb = tbp.sideband.as_object();
    if (sb.contains("sample_rate")) {
      beat_sr = static_cast<int>(
          sb.at("sample_rate").as_int(_sample_rate));
    }
    if (sb.contains("timestamp_us")) {
      beat_ts_us = sb.at("timestamp_us").as_uint(0);
      beat_has_ts = true;
    }
  }
  if (beat_sr != _sample_rate) {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): streaming iport0 sample_rate {} != "
        "expected {}; dropping beat",
        this->id(), beat_sr, _sample_rate));
    return false;
  }

  // First beat: latch the base timestamp.
  if (!_pcm_have_ts) {
    _pcm_base_us = beat_has_ts ? beat_ts_us : 0;
    _pcm_have_ts = beat_has_ts;
    _pcm_buf.clear();
  } else if (beat_has_ts) {
    // Discontinuity check. The expected timestamp of the FIRST sample
    // in this beat is `_pcm_base_us + size * 1e6 / sr`. A drift of
    // more than half a beat's worth of samples (16 ms @ 16 kHz for a
    // 32-ms frame) is treated as a discontinuity: drop the buffered
    // tail and rebase. This catches device pauses / clock jumps; small
    // jitter is absorbed silently.
    const std::uint64_t expect_us =
        _pcm_base_us
        + static_cast<std::uint64_t>(_pcm_buf.size()) * 1'000'000ULL
              / static_cast<std::uint64_t>(_sample_rate);
    const std::uint64_t jitter_us =
        static_cast<std::uint64_t>(n) * 1'000'000ULL
        / (2ULL * static_cast<std::uint64_t>(_sample_rate));
    const std::uint64_t hi = expect_us + jitter_us;
    const std::uint64_t lo = expect_us > jitter_us
                                 ? expect_us - jitter_us : 0;
    if (beat_ts_us < lo || beat_ts_us > hi) {
      session()->warn(fmt(
          "AudioTranscribeStage('{}'): streaming PCM discontinuity "
          "(expected ~{} got {}); rebasing buffer",
          this->id(), expect_us, beat_ts_us));
      _pcm_buf.clear();
      _pcm_base_us = beat_ts_us;
    }
  }

  AlignedVector<float> tmp = tbp.materialize_contiguous_as<float>();
  _pcm_buf.insert(_pcm_buf.end(), tmp.data(), tmp.data() + n);

  // Prune to fit pcm_buffer_s. Drop from the FRONT and advance
  // _pcm_base_us by the number of samples dropped.
  const std::size_t cap =
      static_cast<std::size_t>(_pcm_buffer_s * _sample_rate);
  if (cap > 0 && _pcm_buf.size() > cap) {
    const std::size_t drop = _pcm_buf.size() - cap;
    _pcm_buf.erase(_pcm_buf.begin(),
                   _pcm_buf.begin() + static_cast<long>(drop));
    _pcm_base_us += static_cast<std::uint64_t>(drop) * 1'000'000ULL
                    / static_cast<std::uint64_t>(_sample_rate);
  }
  return true;
}

bool
AudioTranscribeStage::slice_and_transcribe_(std::uint64_t start_us,
                                            std::uint64_t end_us)
{
  if (end_us <= start_us) {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): segment marker has empty range "
        "[{}..{}]; skipping", this->id(), start_us, end_us));
    return false;
  }
  if (!_pcm_have_ts) {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): segment marker arrived before any "
        "timestamped PCM beat; cannot slice -- check that audio-to-pcm "
        "is emitting sideband.timestamp_us"));
    return false;
  }
  const std::uint64_t buf_end_us =
      _pcm_base_us
      + static_cast<std::uint64_t>(_pcm_buf.size()) * 1'000'000ULL
            / static_cast<std::uint64_t>(_sample_rate);
  if (start_us < _pcm_base_us) {
    if (_late_marker_skip) {
      ++_segments_dropped_late;
      session()->warn(fmt(
          "AudioTranscribeStage('{}'): late marker [{}..{}] (buf base "
          "{}); skipping",
          this->id(), start_us, end_us, _pcm_base_us));
      return false;
    }
    start_us = _pcm_base_us;
  }
  if (start_us >= buf_end_us) {
    ++_segments_dropped_late;
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): marker [{}..{}] starts past "
        "buffer tail {}; skipping",
        this->id(), start_us, end_us, buf_end_us));
    return false;
  }
  const std::uint64_t clamped_end = std::min(end_us, buf_end_us);
  const std::uint64_t off_us = start_us - _pcm_base_us;
  const std::size_t off_samples = static_cast<std::size_t>(
      off_us * static_cast<std::uint64_t>(_sample_rate) / 1'000'000ULL);
  const std::uint64_t dur_us = clamped_end - start_us;
  const std::size_t n_samples = static_cast<std::size_t>(
      dur_us * static_cast<std::uint64_t>(_sample_rate) / 1'000'000ULL);
  if (n_samples == 0 || off_samples >= _pcm_buf.size()
      || off_samples + n_samples > _pcm_buf.size()) {
    session()->warn(fmt(
        "AudioTranscribeStage('{}'): marker [{}..{}] slice out of "
        "buffer (off={} n={} buf={}); skipping",
        this->id(), start_us, end_us, off_samples, n_samples,
        _pcm_buf.size()));
    return false;
  }
  session()->log_debug(fmt(
      "AudioTranscribeStage('{}'): transcribing segment [{}..{}] = "
      "{:.2f} s",
      this->id(), start_us, clamped_end, dur_us / 1.0e6));
  // Carry the (clamped) segment span onto the transcript emitted on oport 0.
  _cur_start_us = start_us;
  _cur_end_us   = clamped_end;
  _cur_has_ts   = true;
  const bool ok = transcribe_pcm_(_pcm_buf.data() + off_samples,
                                  n_samples, _sample_rate);
  if (ok) { ++_clips_processed; }
  return ok;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(AudioTranscribeStage)
VPIPE_REGISTER_SPEC(AudioTranscribeStage, kSpec)

}
