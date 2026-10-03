#include "stages/generate-audio-stage.h"

#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-memory.h"
#include "stages/model-provenance.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "generative-models/shared/metal-sol-attention.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "generative-models/yue2/metal-yue2-model.h"
#include "generative-models/yue2/yue2-protocol.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <utility>

namespace vpipe {

namespace {

// The KV of one token: 28 layers x (K, V) x 8 heads x 128 x bf16.
constexpr std::size_t kKvBytesPerToken = 28u * 2u * 8u * 128u * 2u;
// What a typical request's prefix costs in tokens, for plan-time sizing
// only (a full score plus long lyrics; the real figure is known at run).
constexpr int kPlanPrefixTokens = 3000;

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "the audio generator: a models-DB key or a model dir. Today "
          "YuE2 (m-a-p/YuE2-3B). The VAE is NOT here -- wire "
          "audio_latent to audio-vae-decode with m-a-p/YuE2-Vae",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "yue2"},
  {.key = "style", .type = ConfigType::String,
   .doc = "style prompt -- genre, mood, instruments, vocal (the [Tags] "
          "block). A prompt beat's \"style\" overrides it",
   .def_str = ""},
  {.key = "lyrics", .type = ConfigType::Text,
   .doc = "lyrics, sectioned with [Verse] / [Chorus] / [Bridge] / "
          "[Intro] / [Outro] headers. A prompt beat overrides it",
   .def_str = ""},
  {.key = "abc", .type = ConfigType::Text,
   .doc = "OPTIONAL score in ABC notation for a cover or an edit (needs "
          "cot melody or full). Empty: the model writes its own",
   .def_str = ""},
  {.key = "cot", .type = ConfigType::String,
   .doc = "planning: \"full\" (melody + chords, the default), \"melody\" "
          "(melody only -- recommended for covers), \"off\" (no score)",
   .def_str = "full"},
  {.key = "seed", .type = ConfigType::Int,
   .doc = "seed for the token draws and the flow-matching noise",
   .def_int = 831001},
  {.key = "cfg_scale", .type = ConfigType::Real,
   .doc = "text guidance of the song phase; < 0 keeps the protocol's "
          "(1.0 for full/melody, 1.01 for off). Values above 1 double "
          "the song phase's decode cost",
   .def_real = -1.0},
  {.key = "ode_steps", .type = ConfigType::Int,
   .doc = "flow-matching midpoint steps (two model passes each); the "
          "release protocol is 32",
   .def_int = 32},
  {.key = "max_seconds", .type = ConfigType::Real,
   .doc = "cap on the song's length in seconds; 0 = the protocol's "
          "(9000 tokens, 6 minutes). The model ends the song itself "
          "before that",
   .def_real = 0.0},
  {.key = "abc_temperature", .type = ConfigType::Real,
   .doc = "score sampling temperature", .def_real = 0.7},
  {.key = "abc_top_p", .type = ConfigType::Real,
   .doc = "score nucleus mass", .def_real = 0.9},
  {.key = "abc_top_k", .type = ConfigType::Int,
   .doc = "score top-k", .def_int = 30},
  {.key = "abc_repetition_penalty", .type = ConfigType::Real,
   .doc = "score repetition penalty over the last 100 tokens",
   .def_real = 1.005},
  {.key = "song_temperature", .type = ConfigType::Real,
   .doc = "song-token sampling temperature", .def_real = 1.0},
  {.key = "song_top_p", .type = ConfigType::Real,
   .doc = "song-token nucleus mass", .def_real = 0.95},
  {.key = "song_top_k", .type = ConfigType::Int,
   .doc = "song-token top-k", .def_int = 100},
  {.key = "song_repetition_penalty", .type = ConfigType::Real,
   .doc = "song-token repetition penalty over the last 50 tokens",
   .def_real = 1.2},
  // The flow matching's acceleration tiers -- the same spellings as the
  // image and video stages, because they are one vocabulary.
  {.key = "i8_gemm", .type = ConfigType::Bool,
   .doc = "accelerated mode (LOSSY): the flow matching's big GEMMs as "
          "dynamic-int8 matmul2d, ~1e-2 per GEMM. MATRIX CORES ONLY -- an "
          "M5 path; any other box stays dense. Env VPIPE_I8_GEMM overrides",
   .def_bool = false},
  {.key = "ane_ffn", .type = ConfigType::Bool,
   .doc = "accelerated feed-forward (fp16): the flow-matching pass's "
          "SwiGLU rows split between the Apple Neural Engine and the GPU, "
          "run concurrently -- exact up to the ANE's fp16 (MEASURED "
          "0.0206 from the reference's fp32 against the GPU's 0.0191). "
          "For the M4 series, which has no GPU matrix cores. One module "
          "of fixed size whatever the song's length; the resource plan "
          "decides whether it fits",
   .def_bool = false},
  {.key = "ane_rows", .type = ConfigType::Real,
   .doc = "share of the feed-forward's rows given to the ANE; 0 (default) "
          "balances from the two engines' measured times",
   .def_real = 0.0},
  {.key = "ane_layers", .type = ConfigType::Int,
   .doc = "cap on how many of the 28 layers split; 0 (default) = all",
   .def_int = 0},
  {.key = "sage_attn", .type = ConfigType::Bool,
   .doc = "accelerated attention (LOSSY): the flow matching's QK^T in "
          "INT8, K smoothed by its mean. MATRIX CORES ONLY -- an M5 "
          "instruction; any other box says so once and stays dense",
   .def_bool = false},
  {.key = "sage_dense_layers", .type = ConfigType::Int,
   .doc = "leading layers left in the tensor dtype by sage_attn",
   .def_int = 0},
  {.key = "sol_attn", .type = ConfigType::Bool,
   .doc = "accelerated attention (LOSSY): Sol-Attn routes whole key "
          "blocks of the flow matching's attention over [text prefix ; "
          "song tokens ; latents], attending the ones a proxy score keeps "
          "and standing in for the rest with centroids. The text prefix "
          "-- instruction, style, lyrics, score -- is always attended "
          "exactly",
   .def_bool = false},
  {.key = "sol_tau", .type = ConfigType::Real,
   .doc = "Sol-Attn threshold in standard deviations of a query block's "
          "own proxy-score distribution; HIGHER keeps fewer blocks. 0 is "
          "not dense: below-average blocks are still dropped",
   .def_real = 0.0},
  {.key = "sol_dense_layers", .type = ConfigType::Int,
   .doc = "leading layers left dense by sol_attn",
   .def_int = 1},
  {.key = "sol_key_block", .type = ConfigType::Int,
   .doc = "keys per Sol centroid: 32 or 64; 0 (default) takes 64",
   .def_int = 0},
  {.key = "sol_local_radius", .type = ConfigType::Int,
   .doc = "key blocks either side of a query's own that are always exact",
   .def_int = 1},
};
const PortSpec kIports[] = {
  {.name = "prompt",
   .doc = "OPTIONAL FlexData: a string (the lyrics) or an object with any "
          "of style / tags, lyrics, abc, cot, seed, cfg_scale; absent keys "
          "fall back to config. One song per beat. Unwired, the stage "
          "writes one song from its config",
   .type = &typeid(FlexDataPayload),
   .tags = "text", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "audio_latent",
   .doc = "f32 latent audio [1, 64, frames] at 25 latents/s "
          "(latents_per_second in the sideband); decode with "
          "audio-vae-decode",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
  {.name = "score",
   .doc = "FlexData string: the song's ABC score (planned, or the one "
          "supplied). Not written for cot=off",
   .type = &typeid(FlexDataPayload),
   .tags = "text", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "generate-audio",
  .doc       = "Generates a song from a style prompt and lyrics (YuE2, "
               "metal): plans an ABC score, writes the song's semantic "
               "tokens, then flow-matches them to latent audio for "
               "audio-vae-decode.",
  .display_name = "Generate Audio",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

GenerateAudioStage::GenerateAudioStage(const SessionContextIntf* s,
                                       std::string id,
                                       std::vector<InEdge> iports,
                                       FlexData config)
  : TypedStage<GenerateAudioStage>(s, std::move(id), std::move(iports),
                                   std::move(config))
{
  _hf_dir = attr_str("hf_dir");
  _style = attr_str("style");
  _lyrics = attr_str("lyrics");
  _abc = attr_str("abc");
  _cot = attr_str("cot");
  _seed = attr_int("seed");
  _cfg_scale = attr_real("cfg_scale");
  _ode_steps = (int)attr_int("ode_steps");
  _max_seconds = attr_real("max_seconds");
  _abc_temperature = attr_real("abc_temperature");
  _abc_top_p = attr_real("abc_top_p");
  _abc_top_k = (int)attr_int("abc_top_k");
  _abc_penalty = attr_real("abc_repetition_penalty");
  _song_temperature = attr_real("song_temperature");
  _song_top_p = attr_real("song_top_p");
  _song_top_k = (int)attr_int("song_top_k");
  _song_penalty = attr_real("song_repetition_penalty");
  if (_hf_dir.empty()) {
    fail_config(fmt("GenerateAudioStage('{}'): config.hf_dir is required",
                    this->id()));
  }
  if (_cot != "full" && _cot != "melody" && _cot != "off") {
    fail_config(fmt("GenerateAudioStage('{}'): cot must be full, melody or "
                    "off (got '{}')", this->id(), _cot));
  }
  if (_ode_steps < 1) {
    fail_config(fmt("GenerateAudioStage('{}'): ode_steps must be >= 1",
                    this->id()));
  }
  // The tiers, SETTLED: an out-of-range value is the one it falls back
  // to, said once, so the model never re-checks.
  _i8_gemm = attr_bool("i8_gemm");
  _ane_ffn = attr_bool("ane_ffn");
  _ane_rows = std::clamp(attr_real("ane_rows"), 0.0, 1.0);
  _ane_layers = (int)std::max<std::int64_t>(0, attr_int("ane_layers"));
  _sage_attn = attr_bool("sage_attn");
  _sage_dense_layers =
      (int)std::max<std::int64_t>(0, attr_int("sage_dense_layers"));
  _sol_attn = attr_bool("sol_attn");
  _sol_tau = attr_real("sol_tau");
  if (!std::isfinite(_sol_tau)) { _sol_tau = 0.0; }
  _sol_dense_layers =
      (int)std::max<std::int64_t>(0, attr_int("sol_dense_layers"));
  _sol_local_radius =
      (int)std::max<std::int64_t>(-1, attr_int("sol_local_radius"));
  {
    const std::int64_t kb = attr_int("sol_key_block");
    _sol_key_block = (kb == 32 || kb == 64) ? (int)kb : 0;
    if (kb != 0 && _sol_key_block == 0) {
      session()->warn(fmt("GenerateAudioStage('{}'): sol_key_block {} is "
                          "not 32 or 64; using 64", this->id(), kb));
    }
  }
  allocate_oports(spec().oports.size());
}

GenerateAudioStage::~GenerateAudioStage() = default;

const StageSpec&
GenerateAudioStage::spec() const noexcept
{
  return kSpec;
}

std::string
GenerateAudioStage::ane_claim_label_() const
{
  return "generate-audio:" + this->id();
}

std::size_t
GenerateAudioStage::scratch_bytes_() const
{
  const int frames = _max_seconds > 0.0
                         ? (int)std::ceil(_max_seconds * 25.0)
                         : 9000;
  const std::size_t ctx = (std::size_t)std::min(24576,
                                                kPlanPrefixTokens + frames);
  // The decode context's paged KV (the pool keeps its high-water mark),
  // the flow matching's contiguous copy of it plus its own rows, and the
  // NAR activations: ~45 KB a latent row.
  std::size_t b = ctx * kKvBytesPerToken +
                  (ctx + (std::size_t)frames) * kKvBytesPerToken +
                  (std::size_t)(frames + 2) * 45 * 1024;
#ifdef VPIPE_BUILD_APPLE_SILICON
  // Sol's routing scratch, which this stage lends it nothing for: the
  // queries are the latent rows, the keys everything before them too.
  if (_sol_attn) {
    auto* mc = session() ? session()->services()->metal_compute() : nullptr;
    b += genai::MetalSolAttention::scratch_bytes(
        16, 8, (int)(ctx + (std::size_t)frames), 128,
        _sol_key_block > 0 ? _sol_key_block : genai::sol::kBlock,
        mc != nullptr && mc->supports_matrix_cores());
  }
#endif
  return b;
}

std::vector<ResourceClaim>
GenerateAudioStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  std::vector<ResourceClaim> c = model_memory::weight_claims(
      {resolve_model_dir(session(), _hf_dir)});
  auto s = model_memory::scratch_claims("generate-audio", scratch_bytes_(),
                                        model_memory::kPhaseDenoise);
  c.insert(c.end(), s.begin(), s.end());
#ifdef VPIPE_BUILD_APPLE_SILICON
  // The ANE module, in UNITS: CoreML holds those bytes and no ledger in
  // this process can see them. Granted or not at initialize().
  if (_ane_ffn) {
    auto a = model_memory::coreml_claims(
        ane_claim_label_(), genai::MetalYue2Model::ane_runtime_bytes(), 1,
        model_memory::kPhaseDenoise);
    c.insert(c.end(), a.begin(), a.end());
  }
#endif
  return c;
}

StageMemory
GenerateAudioStage::declare_memory() const
{
  StageMemory m;
  if (_hf_dir.empty()) { return m; }
  const std::string d = resolve_model_dir(session(), _hf_dir);
  m.hold(d, model_memory::dir_weights_bytes(d));
  return m;
}

Job
GenerateAudioStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  _one_shot_done = false;
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_hf_dir.empty()) { co_return; }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "GenerateAudioStage('{}'): no metal-compute backend; inert",
        this->id()));
    co_return;
  }
  // A relaunch re-enters here with the last run's model held: drop it
  // before loading, or the peak is two copies.
  _model.reset();
  _tok.reset();
  const std::string dir = resolve_model_dir(session(), _hf_dir);
  genai::MetalYue2Model::Config cfg;
  std::string err;
  if (!genai::MetalYue2Model::config_from_dir(dir, &cfg, &err)) {
    session()->error(fmt(
        "GenerateAudioStage('{}'): '{}' is not an audio generator this "
        "stage knows ({}); inert", this->id(), dir, err));
    co_return;
  }
  _tok = genai::Tokenizer::from_tiktoken(
      (std::filesystem::path(dir) / "qwen.tiktoken").string(),
      genai::yue2::tiktoken_specials(), session());
  if (!_tok) {
    session()->error(fmt(
        "GenerateAudioStage('{}'): '{}' has no usable qwen.tiktoken; inert",
        this->id(), dir));
    co_return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  _model = genai::MetalYue2Model::load(
      genai::open_weight_set(dir, session()), mc, cfg, &err);
  if (!_model) {
    session()->error(fmt("GenerateAudioStage('{}'): YuE2 did not load: {}; "
                         "inert", this->id(), err));
    co_return;
  }
  {
    genai::MetalYue2Model::Accel a;
    a.i8_gemm    = _i8_gemm;
    a.ane_ffn    = _ane_ffn;
    a.ane_rows   = (float)_ane_rows;
    a.ane_layers = _ane_layers;
    a.sage.enabled      = _sage_attn;
    a.sage.dense_layers = _sage_dense_layers;
    a.sol.enabled      = _sol_attn;
    a.sol.tau          = (float)_sol_tau;
    a.sol.dense_layers = _sol_dense_layers;
    a.sol.key_block    = _sol_key_block;
    a.sol.local_radius = _sol_local_radius;
    // WHETHER the ANE runs is the plan's grant, read here where it exists.
    if (a.ane_ffn) {
      const std::size_t bytes = genai::MetalYue2Model::ane_runtime_bytes();
      if (model_memory::coreml_grant(session(), ane_claim_label_(), bytes,
                                     1) <= 0) {
        session()->info(fmt(
            "GenerateAudioStage('{}'): ane_ffn was requested but the plan "
            "left no room for its module ({} MB); keeping the GPU",
            this->id(), bytes >> 20));
        a.ane_ffn = false;
      }
    }
    if (!_model->set_accel(a, &err)) {
      session()->error(fmt("GenerateAudioStage('{}'): {}; inert",
                           this->id(), err));
      _model.reset();
      co_return;
    }
  }
  session()->info(fmt(
      "GenerateAudioStage('{}'): YuE2 loaded in {:.1f} s ({:.2f} GB; NAR "
      "on {} GEMMs, {} attention)", this->id(),
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count(),
      (double)_model->resident_bytes() / 1e9,
      _model->uses_mma() ? "matmul2d" : "steel",
      _model->uses_attn_nax() ? "NAX" : "steel"));
#endif
  co_return;
}

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

// A prompt beat over the config defaults: a bare string is the lyrics.
void
apply_prompt_(const FlexData& fd, genai::yue2::Request* r,
              std::string* cot)
{
  if (fd.is_string()) {
    r->lyrics = std::string(fd.as_string());
    return;
  }
  if (!fd.is_object()) { return; }
  auto o = fd.as_object();
  auto str = [&](const char* k, std::string* dst) {
    if (o.contains(k) && o.at(k).is_string()) {
      *dst = std::string(o.at(k).as_string());
    }
  };
  str("tags", &r->style);
  str("style", &r->style);
  str("lyrics", &r->lyrics);
  str("abc", &r->abc);
  str("cot", cot);
  if (o.contains("seed")) { r->seed = (std::uint64_t)o.at("seed").as_int(); }
  if (o.contains("cfg_scale")) {
    r->cfg_scale = (float)o.at("cfg_scale").as_real(-1.0);
  }
}

}  // namespace
#endif

Job
GenerateAudioStage::process(RuntimeContext& ctx)
{
#ifndef VPIPE_BUILD_APPLE_SILICON
  ctx.signal_done();
  co_return;
#else
  namespace yue2 = genai::yue2;
  using clk = std::chrono::steady_clock;
  yue2::Request req;
  req.style = _style;
  req.lyrics = _lyrics;
  req.abc = _abc;
  req.seed = (std::uint64_t)_seed;
  req.cfg_scale = (float)_cfg_scale;
  std::string cot = _cot;

  const bool wired = ctx.num_iports() > 0 && ctx.iport_connected(0);
  if (wired) {
    auto in = co_await ctx.read(0);
    if (!in) {
      ctx.signal_done();
      co_return;
    }
    if (const auto* p = dynamic_cast<const FlexDataPayload*>(in.get())) {
      apply_prompt_(p->data, &req, &cot);
    }
  } else {
    // Unwired: one song from the config, then the stage is done.
    if (_one_shot_done) {
      ctx.signal_done();
      co_return;
    }
    _one_shot_done = true;
  }
  auto finish_one_shot = [&]() {
    if (!wired) { ctx.signal_done(); }
  };
  if (!_model || !_tok) {
    session()->warn(fmt("GenerateAudioStage('{}'): no model; skipping",
                        this->id()));
    finish_one_shot();
    co_return;
  }
  if (!yue2::parse_cot(cot, &req.cot)) {
    session()->warn(fmt("GenerateAudioStage('{}'): cot '{}' is not full, "
                        "melody or off; skipping", this->id(), cot));
    finish_one_shot();
    co_return;
  }
  std::string err;
  if (!req.valid(&err)) {
    session()->warn(fmt("GenerateAudioStage('{}'): {}; skipping",
                        this->id(), err));
    finish_one_shot();
    co_return;
  }
  if (req.style.empty() && req.lyrics.empty()) {
    session()->warn(fmt("GenerateAudioStage('{}'): neither style nor lyrics "
                        "given; skipping", this->id()));
    finish_one_shot();
    co_return;
  }

  const int ctx_len = _model->config().context;
  auto clamp_budget = [&](yue2::Sampling* s, std::size_t prefix) {
    const int room = ctx_len - (int)prefix;
    s->max_tokens = std::max(1, std::min(s->max_tokens, room));
    s->min_tokens = std::min(s->min_tokens, s->max_tokens);
  };
  UiProgress bar = session()->open_progress("song");
  auto stop = [&]() { return ctx.stop_requested(); };

  // ---- 1. the score -----------------------------------------------------
  std::vector<std::int32_t> abc_ids;
  std::string score;
  if (req.cot != yue2::Cot::kOff) {
    if (!req.abc.empty()) {
      abc_ids = _tok->encode(yue2::nfc(req.abc));
      score = req.abc;
    } else {
      const std::vector<std::int32_t> pre =
          yue2::token_prefix(req, *_tok, nullptr);
      yue2::Sampling s = yue2::Sampling::abc_default();
      s.temperature = (float)_abc_temperature;
      s.top_p = (float)_abc_top_p;
      s.top_k = _abc_top_k;
      s.repetition_penalty = (float)_abc_penalty;
      clamp_budget(&s, pre.size());
      genai::MetalYue2Model::DecodeResult res;
      const bool ok = _model->decode(
          pre, yue2::Phase::kAbc, s, req.seed, nullptr, 1.0f, false,
          [&](std::int32_t, int k) {
            bar.update(0, 0, fmt("planning score: {} tokens", k + 1)());
            return !stop();
          },
          &res, &err);
      if (!ok) {
        session()->warn(fmt("GenerateAudioStage('{}'): planning failed: {}",
                            this->id(), err));
        finish_one_shot();
        co_return;
      }
      if (res.stopped) { co_return; }
      abc_ids = std::move(res.ids);
      score = _tok->decode(abc_ids);
      session()->info(fmt(
          "GenerateAudioStage('{}'): score planned -- {} tokens in {:.1f} s "
          "({:.1f} tok/s){}", this->id(), abc_ids.size(), res.seconds,
          res.seconds > 0 ? (double)abc_ids.size() / res.seconds : 0.0,
          res.truncated ? ", TRUNCATED at the budget" : ""));
    }
  }

  // ---- 2. the song's semantic tokens -----------------------------------
  const std::vector<std::int32_t> prefix = yue2::token_prefix(
      req, *_tok, req.cot == yue2::Cot::kOff ? nullptr : &abc_ids);
  const float guide = req.guidance();
  std::vector<std::int32_t> negative;
  if (guide != 1.0f) {
    negative = yue2::negative_prefix(
        req, *_tok, req.cot == yue2::Cot::kOff ? nullptr : &abc_ids);
  }
  yue2::Sampling ss = yue2::Sampling::semantic_default();
  ss.temperature = (float)_song_temperature;
  ss.top_p = (float)_song_top_p;
  ss.top_k = _song_top_k;
  ss.repetition_penalty = (float)_song_penalty;
  if (_max_seconds > 0.0) {
    ss.max_tokens = std::max(
        1, std::min(ss.max_tokens,
                    (int)std::ceil(_max_seconds * yue2::kLatentsPerSecond)));
  }
  clamp_budget(&ss, std::max(prefix.size(), negative.size()));
  genai::MetalYue2Model::DecodeResult song;
  {
    const bool ok = _model->decode(
        prefix, yue2::Phase::kSemantic, ss, req.seed,
        negative.empty() ? nullptr : &negative, guide,
        req.cot == yue2::Cot::kOff,
        [&](std::int32_t, int k) {
          bar.update(0, 0, fmt("writing the song: {:.1f} s",
                               (double)(k + 1) /
                                   yue2::kLatentsPerSecond)());
          return !stop();
        },
        &song, &err);
    if (!ok) {
      session()->warn(fmt("GenerateAudioStage('{}'): the song phase "
                          "failed: {}", this->id(), err));
      finish_one_shot();
      co_return;
    }
    if (song.stopped) { co_return; }
  }
  std::vector<std::int32_t> codes;
  codes.reserve(song.ids.size());
  for (std::int32_t id : song.ids) {
    codes.push_back(id - yue2::kCodecOffset);
  }
  if (codes.empty()) {
    session()->warn(fmt("GenerateAudioStage('{}'): the model ended the song "
                        "before its first frame; nothing to emit",
                        this->id()));
    finish_one_shot();
    co_return;
  }
  session()->info(fmt(
      "GenerateAudioStage('{}'): song written -- {} tokens ({:.1f} s of "
      "audio) in {:.1f} s ({:.1f} tok/s){}", this->id(), codes.size(),
      (double)codes.size() / yue2::kLatentsPerSecond, song.seconds,
      song.seconds > 0 ? (double)codes.size() / song.seconds : 0.0,
      song.truncated ? ", TRUNCATED at the budget" : ""));

  // ---- 3. latents --------------------------------------------------------
  std::vector<float> lat;
  const auto t_nar = clk::now();
  {
    const bool ok = _model->synthesize(
        prefix, codes, req.seed, _ode_steps, &lat,
        [&](int done, int total) {
          bar.update((std::uint64_t)done, (std::uint64_t)total,
                     "synthesizing");
          return !stop();
        },
        &err);
    if (!ok) {
      if (!err.empty()) {
        session()->warn(fmt("GenerateAudioStage('{}'): flow matching "
                            "failed: {}", this->id(), err));
      }
      finish_one_shot();
      co_return;
    }
  }
  const double nar_s =
      std::chrono::duration<double>(clk::now() - t_nar).count();
  {
    // What the tiers DID, not what was asked: a knob that silently did
    // nothing and one that did something look the same in a wall clock.
    const auto& st = _model->nar_stats();
    std::string tiers;
    if (st.ane_rows > 0) {
      tiers += fmt(", ANE {:.0f}% of feed-forward rows",
                   100.0 * (double)st.ane_rows / (double)st.ffn_rows)();
    }
    if (st.sol_layers > 0) {
      tiers += fmt(", Sol kept {:.1f}% of key blocks",
                   100.0 * st.sol_kept())();
    }
    if (st.sage_layers > 0) {
      tiers += fmt(", Sage on {} attention calls", st.sage_layers)();
    }
    if (st.i8_gemms > 0) {
      tiers += fmt(", {} int8 GEMMs", st.i8_gemms)();
    }
    session()->info(fmt(
        "GenerateAudioStage('{}'): latents in {:.1f} s ({} steps, {:.0f} ms "
        "an evaluation{})", this->id(), nar_s, _ode_steps,
        st.evals > 0 ? st.ms / st.evals : 0.0, tiers));
  }
  bar.finish();

  // [frames, 64] -> the beat's channel-major [1, 64, frames].
  const int Z = yue2::kLatentDim;
  const int T = (int)(lat.size() / (std::size_t)Z);
  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  out->shape = {1, (std::int64_t)Z, (std::int64_t)T};
  out->resize_contiguous(lat.size());
  float* dst = out->as_f32();
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < Z; ++c) {
      dst[(std::size_t)c * T + t] = lat[(std::size_t)t * Z + c];
    }
  }
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign(
      sideband::kLatentsPerSecond,
      FlexData::make_real((double)yue2::kLatentsPerSecond));
  out->sideband = std::move(sb);
  provenance::set_model_name(out->sideband, _hf_dir);
  co_await ctx.write(0, std::move(out));
  if (req.cot != yue2::Cot::kOff && ctx.num_oports() > 1) {
    _last_score = score;
    co_await ctx.write(1, make_payload<FlexDataPayload>(
                              FlexData::make_string(score)));
  }
  ++_songs;
  finish_one_shot();
  co_return;
#endif
}

VPIPE_REGISTER_STAGE(GenerateAudioStage)
VPIPE_REGISTER_SPEC(GenerateAudioStage, kSpec)

}  // namespace vpipe
