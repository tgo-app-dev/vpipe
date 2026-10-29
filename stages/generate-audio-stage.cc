#include "stages/generate-audio-stage.h"

#include "common/beat-keys.h"
#include "common/beat-payload-intf.h"
#include "common/flex-bag.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "pipeline/spec-extra.h"
#include "stages/model-memory-host.h"
#include "stages/model-memory.h"
#include "stages/model-provenance.h"
#include "stages/model-registry.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/tensor-beat.h"
#include "generative-models/shared/metal-sol-attention.h"
#include "generative-models/minimax-music3/metal-music3-ar.h"
#include "generative-models/minimax-music3/metal-music3-dit.h"
#include "generative-models/minimax-music3/music3-protocol.h"
#include "generative-models/tokenizer.h"
#include "generative-models/weight-set.h"
#include "generative-models/yue2/metal-oobleck-decoder.h"
#include "generative-models/yue2/metal-yue2-model.h"
#include "generative-models/yue2/yue2-protocol.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

namespace vpipe {

namespace {

// The longest prefix the plan books for, in tokens: style and lyrics, plus
// a score at the protocol's planning budget unless cot=off (no score).
// Plan-time only; a song whose real prefix is longer revises the claims
// before it runs.
constexpr SpecExtra kI8TierChoices[] = {
  {spec_key::kChoices, "accurate,fast,token,token_plain"},
};
constexpr int kPlanTextTokens  = 1024;
constexpr int kPlanScoreTokens = 4096;

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = true,
   .doc = "the audio generator: a models-DB key or a model dir -- YuE2 "
          "(m-a-p/YuE2-3B) or MiniMax-Music3 (MiniMaxAI/MiniMax-Music3). "
          "The VAE is NOT here -- wire audio_latent to audio-vae-decode "
          "with m-a-p/YuE2-Vae, or with the same MiniMax-Music3 directory "
          "(its vocoder/)",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "yue2,minimax-music3"},
  {.key = "style", .type = ConfigType::String,
   .doc = "style prompt -- genre, mood, instruments, vocal (YuE2's [Tags] "
          "block; MiniMax-Music3's music description, REQUIRED there, "
          "best as a structured caption: global metadata, vocal details, "
          "arrangement). A prompt beat's \"style\" (or \"caption\") "
          "overrides it",
   .def_str = ""},
  {.key = "lyrics", .type = ConfigType::Text,
   .doc = "lyrics, sectioned with [Verse] / [Chorus] / [Bridge] / "
          "[Intro] / [Outro] headers (MiniMax-Music3: each header on its "
          "own line -- text after one on its line is dropped -- and the "
          "lyrics are required). A prompt beat overrides it",
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
          "(YuE2 1.0 for full/melody, 1.01 for off -- values above 1 "
          "double the song phase's decode cost; MiniMax-Music3 1.5, its "
          "codes are always guided)",
   .def_real = -1.0},
  {.key = "flow_cfg_scale", .type = ConfigType::Real,
   .doc = "MiniMax-Music3: guidance of the flow matching against an "
          "all-zero condition; < 0 keeps the protocol's 1.7",
   .def_real = -1.0},
  {.key = "ode_steps", .type = ConfigType::Int,
   .doc = "flow-matching steps; 0 (default) keeps the model's protocol: "
          "YuE2 32 midpoint steps (two model passes each), MiniMax-Music3 "
          "30 Euler steps (one guided pair of passes each, per 8 s "
          "window)",
   .def_int = 0},
  {.key = "max_seconds", .type = ConfigType::Real,
   .doc = "cap on the song's length in seconds; 0 = the protocol's "
          "(9000 frames, 6 minutes; MiniMax-Music3 also within its 10240-"
          "token context). The model ends the song itself before that",
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
  {.key = "lm_quant", .type = ConfigType::String,
   .doc = "how the AR stack -- the half that writes the song's tokens -- "
          "is held: \"w8\" (default) quantizes its projections in memory "
          "as it loads, to affine 8-bit group-64; \"bf16\" keeps it as "
          "published. YuE2: 1.26 GB less, a faster song phase, as close to "
          "the reference's fp32 run as bf16. MiniMax-Music3 (its LM and "
          "depth decoder both): 8.4 against 15.5 GB and ~58 against ~150 "
          "ms a frame on the M4 Pro, 1.2-1.3x the reference's own bf16 "
          "distance from fp32. The flow-matching stack stays as it is "
          "either way (see i8_gemm_tier)",
   .def_str = "w8"},
  // The flow matching's acceleration tiers -- the same spellings as the
  // image and video stages, because they are one vocabulary.
  {.key = "i8_gemm", .type = ConfigType::Bool,
   .doc = "accelerated mode (LOSSY): the flow matching's big GEMMs as "
          "dynamic-int8 matmul2d, ~1e-2 per GEMM. MATRIX CORES ONLY -- an "
          "M5 path; any other box stays dense. Env VPIPE_I8_GEMM overrides",
   .def_bool = false},
  {.key = "i8_gemm_tier", .type = ConfigType::String, .required = false,
   .doc = "which int8 GEMM i8_gemm runs, MiniMax-Music3 only (YuE2 runs "
          "accurate): 'accurate' (default; the flow-matching transformer "
          "kept 16-bit, its weights quantized per call at groups of 64: "
          "~1.1x the 16-bit flow matching on a base M5), 'fast' (its block "
          "weights HELD as rotated int8 at groups of 512: ~1.4x), 'token' "
          "(int8 per weight row x per token after a 256-wide Hadamard "
          "rotation: ~1.7x, the song still nearer the reference's fp32 "
          "than the reference's own bf16) or 'token_plain' (the same "
          "unrotated: no faster, and several times further). The last "
          "three hold the transformer in 2.5 GB where 16-bit takes 4.9, "
          "converted once as it loads from whatever the checkpoint stores "
          "(F32, F16 or BF16). Env VPIPE_I8_TIER overrides",
   .def_str = "accurate", .extra = kI8TierChoices},
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
          "of style / tags / caption, lyrics, abc, cot, seed, cfg_scale; "
          "absent keys fall back to config. One song per beat. Unwired, "
          "the stage writes one song from its config",
   .type = &typeid(FlexDataPayload),
   .tags = "text", .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "audio_latent",
   .doc = "f32 latent audio (latents_per_second in the sideband): YuE2 "
          "[1, 64, frames] at 25 a second, MiniMax-Music3 [1, 128, "
          "latents] at 44100 / 512; decode with audio-vae-decode",
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
  .doc       = "Generates a song from a style prompt and lyrics (YuE2 or "
               "MiniMax-Music3, metal): writes the song's tokens -- YuE2 "
               "after planning an ABC score -- then flow-matches them to "
               "latent audio for audio-vae-decode.",
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
  : TypedStage<GenerateAudioStage>(s, std::move(id), iports,
                                   std::move(config)),
    _prompt_wired(!iports.empty())
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
  _flow_cfg = attr_real("flow_cfg_scale");
  if (_hf_dir.empty()) {
    fail_config(fmt("GenerateAudioStage('{}'): config.hf_dir is required",
                    this->id()));
  }
  if (_cot != "full" && _cot != "melody" && _cot != "off") {
    fail_config(fmt("GenerateAudioStage('{}'): cot must be full, melody or "
                    "off (got '{}')", this->id(), _cot));
  }
  if (_ode_steps < 0) {
    fail_config(fmt("GenerateAudioStage('{}'): ode_steps must be >= 0",
                    this->id()));
  }
  {
    std::string q = attr_str("lm_quant");
    if (q.empty()) { q = "w8"; }
    if (q != "w8" && q != "bf16") {
      fail_config(fmt("GenerateAudioStage('{}'): lm_quant must be \"w8\" "
                      "or \"bf16\" (got \"{}\")", this->id(), q));
    }
    _lm_quant_bits = q == "bf16" ? 0 : 8;
  }
  // The tiers, SETTLED: an out-of-range value is the one it falls back
  // to, said once, so the model never re-checks.
  _i8_gemm = attr_bool("i8_gemm");
  {
    const std::string t = attr_str("i8_gemm_tier");
    _i8_tier = t == "token_plain" ? 3 : t == "token" ? 2 : t == "fast" ? 1
                                                                       : 0;
    if (!t.empty() && t != "accurate" && _i8_tier == 0) {
      fail_config(fmt("GenerateAudioStage('{}'): i8_gemm_tier must be "
                      "'accurate', 'fast', 'token' or 'token_plain' (got "
                      "'{}')", this->id(), t));
    }
  }
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

std::string
GenerateAudioStage::scratch_label_() const
{
  return "generate-audio:" + this->id();
}

GenerateAudioStage::Family
GenerateAudioStage::family() const
{
  if (_hf_dir.empty()) { return Family::kNone; }
  const std::filesystem::path root(resolve_model_dir(session(), _hf_dir));
  std::ifstream in(root / "config.json");
  if (!in) { return Family::kNone; }
  try {
    const FlexData fd = FlexData::from_json(in);
    if (!fd.is_object()) { return Family::kNone; }
    auto o = fd.as_object();
    const std::string t = o.contains("model_type") &&
                                  o.at("model_type").is_string()
                              ? std::string(o.at("model_type").as_string())
                              : std::string();
    if (t == "minimax_music3") { return Family::kMusic3; }
    if (t == "yue2") { return Family::kYue2; }
  } catch (const std::exception&) {
  }
  return Family::kNone;
}

std::vector<std::string>
GenerateAudioStage::music3_dirs_() const
{
  const std::filesystem::path root(resolve_model_dir(session(), _hf_dir));
  std::vector<std::string> v;
  for (const char* sub : {"language_model", "rvq_depth_decoder",
                          "condition_encoder", "transformer"}) {
    v.push_back((root / sub).string());
  }
  return v;
}

int
GenerateAudioStage::music3_dit_tier() const
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  return genai::MetalMusic3Dit::effective_i8_tier(
      _i8_gemm ? _i8_tier : -1, mc != nullptr && mc->supports_matrix_cores());
#else
  return -1;
#endif
}

bool
GenerateAudioStage::music3_swaps() const
{
  if (family() != Family::kMusic3) { return false; }
  if (const char* e = std::getenv("VPIPE_MUSIC3_SWAP")) {
    return std::atoi(e) != 0;
  }
#ifdef VPIPE_BUILD_APPLE_SILICON
  const std::size_t ram = model_memory::phys_ram();
  if (ram == 0) { return false; }
  // The four parts as they will be held, a song's working set at the
  // plan's bounds (both halves'), and the revisable cushion. Total RAM,
  // not free memory: every pass must answer alike.
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const SongBytes b = song_bytes_(-1, -1, false);
  const std::size_t need =
      genai::MetalMusic3Ar::lm_weight_bytes(root, _lm_quant_bits) +
      genai::MetalMusic3Ar::depth_weight_bytes(root, _lm_quant_bits) +
      genai::MetalMusic3Dit::condition_weight_bytes(root) +
      genai::MetalMusic3Dit::transformer_weight_bytes(root,
                                                      music3_dit_tier()) +
      b.held + b.flow + b.handoff + model_memory::kHeadroom;
  return ram < need;
#else
  return false;
#endif
}

int
GenerateAudioStage::plan_frames_() const
{
  // The protocol's song budget (9000 tokens, one latent frame each), or
  // max_seconds when that is shorter.
  int frames = 9000;
#ifdef VPIPE_BUILD_APPLE_SILICON
  frames = family() == Family::kMusic3
               ? genai::music3::kMaxFrames
               : genai::yue2::Sampling::semantic_default().max_tokens;
#endif
  if (_max_seconds > 0.0) {
    frames = std::min(frames, family() == Family::kMusic3
                                  ? std::max(1, (int)(_max_seconds * 25.0))
                                  : (int)std::ceil(_max_seconds * 25.0));
  }
  return frames;
}

GenerateAudioStage::SongBytes
GenerateAudioStage::song_bytes_(int prefix, int frames, bool guided) const
{
  SongBytes b;
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_hf_dir.empty()) { return b; }
  if (family() == Family::kMusic3) {
    // MiniMax-Music3: the AR half's two contexts and the hidden-state
    // handoff, and the DiT's window, all kept once allocated; the decode
    // downstream at its vocoder's config, of the song's stitched latents.
    //
    // At plan time the bounds are the protocol's: a prompt of up to 5000
    // tokens (the prefill's activations), and as many frames as the
    // shortest prompt leaves in the context (the handoff) -- the K/V is
    // the context's whole either way.
    const std::string root = resolve_model_dir(session(), _hf_dir);
    genai::MetalMusic3Ar::Config ac;
    (void)genai::MetalMusic3Ar::config_from_dir(root, &ac, nullptr);
    ac.lm_quant_bits = _lm_quant_bits;
    if (frames < 0) {
      frames = std::min(plan_frames_(), ac.context - 4);
    }
    if (prefix < 0) {
      // A prompt beat can bring up to the protocol's 5000 tokens; the
      // config's own caption and lyrics are known now, and a byte-level
      // BPE never yields more tokens than the text has bytes.
      prefix = genai::music3::kMaxPromptTokens;
      if (!_prompt_wired) {
        prefix = std::min<int>(
            prefix, (int)genai::music3::prompt_text(_style, _lyrics).size());
      }
    }
    frames = std::max(frames, 1);
    genai::MetalMusic3Dit::Config dc;
    (void)genai::MetalMusic3Dit::config_from_dir(root, &dc, nullptr);
    auto* mc = session() ? session()->services()->metal_compute() : nullptr;
    b.held = genai::MetalMusic3Ar::run_bytes(
        ac, prefix, frames, mc != nullptr && mc->supports_matrix_cores());
    b.flow = genai::MetalMusic3Dit::window_scratch_bytes(dc,
                                                         music3_dit_tier());
    b.handoff = genai::MetalMusic3Ar::handoff_bytes(ac, frames);
    genai::MetalOobleckDecoder::Config vc;
    (void)genai::MetalOobleckDecoder::config_from_dir(
        (std::filesystem::path(root) / "vocoder").string(), &vc, nullptr);
    genai::MetalOobleckDecoder::decode_cost(
        vc, genai::MetalOobleckDecoder::kDefaultCoreFrames,
        genai::music3::stitched_latents(frames), &b.pcm, &b.decode);
    (void)guided;
    return b;
  }
  // The checkpoint's own dims (defaults when there is no readable config:
  // a plan has to say something before the model exists).
  genai::MetalYue2Model::Config cfg;
  (void)genai::MetalYue2Model::config_from_dir(
      resolve_model_dir(session(), _hf_dir), &cfg);
  if (frames < 0) { frames = plan_frames_(); }
  if (prefix < 0) {
    prefix = kPlanTextTokens + (_cot == "off" ? 0 : kPlanScoreTokens);
  }
  // The song phase decodes a negative context beside the prompt whenever
  // guidance is not 1: what the config asks for, or what a beat did.
  const bool cfg_guided = _cfg_scale >= 0.0 ? _cfg_scale != 1.0
                                            : _cot == "off";
  const auto r = genai::MetalYue2Model::run_bytes(cfg, prefix, frames,
                                                  guided || cfg_guided);
  b.held = r.held;
  b.transient = r.transient;
  // Sol's routing scratch is the model's for the rest of the run too:
  // the queries are the latent rows, the keys everything before them.
  if (_sol_attn) {
    auto* mc = session() ? session()->services()->metal_compute() : nullptr;
    const int keys = std::min(cfg.context, prefix + 2 * frames + 3);
    b.held += genai::MetalSolAttention::scratch_bytes(
        cfg.n_heads, cfg.n_kv_heads, keys, cfg.head_dim,
        _sol_key_block > 0 ? _sol_key_block : genai::sol::kBlock,
        mc != nullptr && mc->supports_matrix_cores());
  }
  // The decode downstream, at YuE2-Vae's released config and the tile a
  // loaded decoder uses: the arena is the tile's, the PCM the song's.
  genai::MetalOobleckDecoder::decode_cost(
      genai::MetalOobleckDecoder::Config{},
      genai::MetalOobleckDecoder::kDefaultCoreFrames, frames, &b.pcm,
      &b.decode);
#else
  (void)prefix;
  (void)frames;
  (void)guided;
#endif
  return b;
}

void
GenerateAudioStage::revise_song_bytes_(const SongBytes& b)
{
  auto* mgr = session() ? session()->services()->generative_model_manager()
                        : nullptr;
  if (mgr == nullptr) { return; }
  const std::string l = scratch_label_();
  auto up = [&](std::size_t want, std::size_t& booked, const char* sfx) {
    if (want <= booked) { return; }
    booked = want;
    mgr->revise_scratch(l + sfx, want);
  };
  up(b.held, _booked.held, "/held");
  up(b.transient, _booked.transient, "/ode");
  up(b.flow, _booked.flow, "/flow");
  up(b.handoff, _booked.handoff, "/handoff");
  up(b.decode, _booked.decode, "/audio-decode");
  up(b.pcm, _booked.pcm, "/pcm");
}

std::size_t
GenerateAudioStage::held_scratch_bytes() const
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (family() == Family::kMusic3) {
    return (_m3_ar ? _m3_ar->held_scratch_bytes() : 0) +
           (_m3_dit ? _m3_dit->scratch_bytes() : 0);
  }
  return _model ? _model->held_scratch_bytes() : 0;
#else
  return 0;
#endif
}

std::vector<ResourceClaim>
GenerateAudioStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  // MiniMax-Music3 holds four parts of its repository; the vocoder is
  // audio-vae-decode's, and anything else there (the original layout's
  // files) loads nowhere.
  const bool m3 = family() == Family::kMusic3;
  // MiniMax-Music3's halves taking turns: each in its own phase -- a rule
  // of this stage's own figures and the box (music3_swaps), so it is
  // certain before anything loads and the same whatever the graph holds.
  const bool swap = m3 && music3_swaps();
  std::vector<ResourceClaim> c;
  if (swap) {
    const auto d = music3_dirs_();
    c = model_memory::weight_claims_in_phase({d[0], d[1]},
                                             model_memory::kPhaseCondition);
    for (ResourceClaim& r : model_memory::weight_claims_in_phase(
             {d[2], d[3]}, model_memory::kPhaseDenoise)) {
      c.push_back(std::move(r));
    }
  } else {
    c = model_memory::weight_claims(
        m3 ? music3_dirs_()
           : std::vector<std::string>{resolve_model_dir(session(), _hf_dir)});
  }
#ifdef VPIPE_BUILD_APPLE_SILICON
  // ... each booked at what it HOLDS, not at its size on disk (28.3 GB of
  // which 13.3 is held at lm_quant w8): the LM's layers built w8 and only
  // two windows of its 200000-row tables, the F32 DiT and condition
  // encoder narrowed to f16 -- or the DiT's blocks in their int8 layout.
  // The planner never books one above the checkpoint.
  if (m3 && c.size() == 4) {
    const std::string root = resolve_model_dir(session(), _hf_dir);
    const std::size_t held[4] = {
        genai::MetalMusic3Ar::lm_weight_bytes(root, _lm_quant_bits),
        genai::MetalMusic3Ar::depth_weight_bytes(root, _lm_quant_bits),
        genai::MetalMusic3Dit::condition_weight_bytes(root),
        genai::MetalMusic3Dit::transformer_weight_bytes(root,
                                                        music3_dit_tier())};
    for (int i = 0; i < 4; ++i) {
      if (held[i] > 0) {
        bag::set_integer(c[(std::size_t)i].extra,
                         model_memory::kClaimHeldBytes,
                         (std::int64_t)held[i]);
      }
    }
  }
#endif
  const SongBytes b = song_bytes_(-1, -1, false);
  const std::string l = scratch_label_();
  auto add = [&](std::vector<ResourceClaim> v) {
    c.insert(c.end(), v.begin(), v.end());
  };
  // The KV pool and the NAR scratch are allocated by the first song and
  // kept: UNPHASED, alive through the audio decode and whatever follows --
  // unless MiniMax-Music3's halves take turns, when each half's working
  // set goes with it and the hidden states pass from one to the other.
  add(model_memory::scratch_claims(
      l + "/held", b.held,
      swap ? model_memory::kPhaseCondition : std::string_view{}));
  add(model_memory::scratch_claims(l + "/ode", b.transient,
                                   model_memory::kPhaseDenoise));
  if (m3) {
    add(model_memory::scratch_claims(
        l + "/flow", b.flow,
        swap ? model_memory::kPhaseDenoise : std::string_view{}));
    add(model_memory::payload_claims(l + "/handoff", b.handoff,
                                     model_memory::kPhaseCondition,
                                     model_memory::kPhaseDenoise));
  }
  add(model_memory::scratch_claims(l + "/audio-decode", b.decode,
                                   model_memory::kPhaseDecodeAudio));
  add(model_memory::payload_claims(l + "/pcm", b.pcm,
                                   model_memory::kPhaseDecodeAudio,
                                   model_memory::kPhaseDecodeAudio));
#ifdef VPIPE_BUILD_APPLE_SILICON
  // The ANE module, in UNITS: CoreML holds those bytes and no ledger in
  // this process can see them. Granted or not at initialize().
  if (_ane_ffn && family() == Family::kYue2) {
    add(model_memory::coreml_claims(
        ane_claim_label_(), genai::MetalYue2Model::ane_runtime_bytes(), 1,
        model_memory::kPhaseDenoise));
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
  // What the model will HOLD: with lm_quant w8 the AR stack at w8. The
  // claims ledger sizes from disk and is corrected once it has loaded.
  std::size_t held = 0;
  std::size_t latent_ch = 64;
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (family() == Family::kMusic3) {
    const auto dirs = music3_dirs_();
    // What each part holds, as declare_resources books it, and whether the
    // stage lets go of it (the halves taking turns).
    const bool swap = music3_swaps();
    m.hold(dirs[0], genai::MetalMusic3Ar::lm_weight_bytes(d, _lm_quant_bits),
           0, swap);
    m.hold(dirs[1],
           genai::MetalMusic3Ar::depth_weight_bytes(d, _lm_quant_bits), 0,
           swap);
    m.hold(dirs[2], genai::MetalMusic3Dit::condition_weight_bytes(d), 0,
           swap);
    m.hold(dirs[3],
           genai::MetalMusic3Dit::transformer_weight_bytes(
               d, music3_dit_tier()), 0, swap);
    latent_ch = genai::music3::kLatentChannels;
    const SongBytes b = song_bytes_(-1, -1, false);
    // Taking turns, one half's working set at a time.
    m.scratch = (swap ? std::max(b.held, b.flow) : b.held + b.flow) +
                b.handoff;
    m.outputs = {(std::size_t)genai::music3::stitched_latents(plan_frames_()) *
                 latent_ch * 4};
    return m;
  }
  held = genai::MetalYue2Model::weight_bytes(d, _lm_quant_bits);
#endif
  (void)latent_ch;
  m.hold(d, held > 0 ? held : model_memory::dir_weights_bytes(d));
  // The report ledger's view of what declare_resources() claims: a song's
  // working set while it runs, and the latent it hands downstream (f32
  // [1, 64, frames]). The VAE decode's terms are claimed here too, but
  // they are that stage's to report.
  const SongBytes b = song_bytes_(-1, -1, false);
  m.scratch = b.held + b.transient;
  m.outputs = {(std::size_t)plan_frames_() * 64 * 4};
  return m;
}

Job
GenerateAudioStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  _one_shot_done = false;
  // What declare_resources() booked for this launch; a beat revises up.
  _booked = song_bytes_(-1, -1, false);
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
  _m3_ar.reset();
  _m3_dit.reset();
  const std::string dir = resolve_model_dir(session(), _hf_dir);
  if (family() == Family::kMusic3) {
    initialize_music3_(dir);
    co_return;
  }
  genai::MetalYue2Model::Config cfg;
  std::string err;
  if (!genai::MetalYue2Model::config_from_dir(dir, &cfg, &err)) {
    session()->error(fmt(
        "GenerateAudioStage('{}'): '{}' is not an audio generator this "
        "stage knows ({}); inert", this->id(), dir, err));
    co_return;
  }
  cfg.ar_quant_bits = _lm_quant_bits;
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
  // The claims ledger booked the checkpoint's disk size; correct it to
  // what the model holds (the AR stack at w8 holds 1.26 GB less).
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->revise_declaration(dir, _model->resident_bytes());
  }
  session()->info(fmt(
      "GenerateAudioStage('{}'): YuE2 loaded in {:.1f} s ({:.2f} GB; AR "
      "{}; NAR on {} GEMMs, {} attention)", this->id(),
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count(),
      (double)_model->resident_bytes() / 1e9,
      _model->ar_quantized() ? "w8 in memory" : "bf16",
      _model->uses_mma() ? "matmul2d" : "steel",
      _model->uses_attn_nax() ? "NAX" : "steel"));
#endif
  co_return;
}

#ifdef VPIPE_BUILD_APPLE_SILICON
void
GenerateAudioStage::initialize_music3_(const std::string& root)
{
  namespace fs = std::filesystem;
  genai::MetalMusic3Ar::Config ac;
  genai::MetalMusic3Dit::Config dc;
  std::string err;
  if (!genai::MetalMusic3Ar::config_from_dir(root, &ac, &err) ||
      !genai::MetalMusic3Dit::config_from_dir(root, &dc, &err)) {
    session()->error(fmt("GenerateAudioStage('{}'): '{}' is not a "
                         "MiniMax-Music3 layout this stage reads ({}); "
                         "inert", this->id(), root, err));
    return;
  }
  const std::string tj =
      (fs::path(root) / "tokenizer" / "tokenizer.json").string();
  _tok = genai::Tokenizer::from_huggingface_json(tj, session());
  if (!_tok || !genai::music3::read_added_tokens(tj, &_m3_added, &err)) {
    session()->error(fmt("GenerateAudioStage('{}'): '{}' has no usable "
                         "tokenizer/tokenizer.json ({}); inert", this->id(),
                         root, err));
    _tok.reset();
    return;
  }
  _m3_root = root;
  _m3_swap = music3_swaps();
  // The flow-matching tiers ane_ffn / sage_attn / sol_attn are YuE2's;
  // say once that they do nothing here rather than read like they worked.
  if (_ane_ffn || _sage_attn || _sol_attn) {
    session()->info(fmt("GenerateAudioStage('{}'): ane_ffn / sage_attn / "
                        "sol_attn are not wired for MiniMax-Music3 -- its "
                        "flow matching runs without them", this->id()));
  }
  if (_i8_gemm && music3_dit_tier() < 0) {
    session()->info(fmt("GenerateAudioStage('{}'): i8_gemm needs the GPU's "
                        "matrix cores; MiniMax-Music3's flow matching runs "
                        "16-bit here", this->id()));
  }
  if (_m3_swap) {
    session()->info(fmt(
        "GenerateAudioStage('{}'): MiniMax-Music3's two halves take turns "
        "on this {:.0f} GB box: the language model and depth decoder load "
        "for a song's codes and are let go before the flow matching loads, "
        "and the other way round for the next song", this->id(),
        (double)model_memory::phys_ram() / (double)(1ull << 30)));
  }
  // Taking turns, the first song starts with the AR half alone.
  if (!load_music3_ar_() || (!_m3_swap && !load_music3_flow_())) {
    _m3_ar.reset();
    _m3_dit.reset();
    _tok.reset();
  }
}

bool
GenerateAudioStage::load_music3_ar_()
{
  if (_m3_ar) { return true; }
  auto* mc = session()->services()->metal_compute();
  genai::MetalMusic3Ar::Config ac;
  std::string err;
  if (!genai::MetalMusic3Ar::config_from_dir(_m3_root, &ac, &err)) {
    session()->error(fmt("GenerateAudioStage('{}'): MiniMax-Music3's "
                         "language model: {}", this->id(), err));
    return false;
  }
  ac.lm_quant_bits = _lm_quant_bits;
  ac.depth_quant_bits = _lm_quant_bits;
  const auto dirs = music3_dirs_();
  const auto t0 = std::chrono::steady_clock::now();
  _m3_ar = genai::MetalMusic3Ar::load(
      genai::open_weight_set(dirs[0], session()),
      genai::open_weight_set(dirs[1], session()), mc, ac, &err);
  if (!_m3_ar) {
    session()->error(fmt("GenerateAudioStage('{}'): MiniMax-Music3's "
                         "language model did not load: {}", this->id(),
                         err));
    return false;
  }
  // The claims booked what each part holds; what it does hold, now.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    const std::size_t depth = _m3_ar->depth_bytes();
    mgr->revise_declaration(dirs[0], _m3_ar->resident_bytes() - depth);
    mgr->revise_declaration(dirs[1], depth);
  }
  session()->info(fmt(
      "GenerateAudioStage('{}'): MiniMax-Music3's language model loaded in "
      "{:.1f} s ({:.2f} GB; LM {}, depth decoder {})", this->id(),
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count(),
      (double)_m3_ar->resident_bytes() / 1e9,
      _m3_ar->lm_quantized() ? "w8 in memory" : "bf16",
      _m3_ar->depth_quantized() ? "w8 in memory" : "bf16"));
  return true;
}

bool
GenerateAudioStage::load_music3_flow_()
{
  if (_m3_dit) { return true; }
  auto* mc = session()->services()->metal_compute();
  genai::MetalMusic3Dit::Config dc;
  std::string err;
  if (!genai::MetalMusic3Dit::config_from_dir(_m3_root, &dc, &err)) {
    session()->error(fmt("GenerateAudioStage('{}'): MiniMax-Music3's flow "
                         "matching: {}", this->id(), err));
    return false;
  }
  dc.i8_tier = _i8_gemm ? _i8_tier : -1;
  const auto dirs = music3_dirs_();
  const auto t0 = std::chrono::steady_clock::now();
  _m3_dit = genai::MetalMusic3Dit::load(
      genai::open_weight_set(dirs[3], session()),
      genai::open_weight_set(dirs[2], session()), mc, dc, &err);
  if (!_m3_dit) {
    session()->error(fmt("GenerateAudioStage('{}'): MiniMax-Music3's flow "
                         "matching did not load: {}", this->id(), err));
    return false;
  }
  if (auto* mgr = session()->services()->generative_model_manager()) {
    const std::size_t cond = _m3_dit->condition_bytes();
    mgr->revise_declaration(dirs[2], cond);
    mgr->revise_declaration(dirs[3], _m3_dit->resident_bytes() - cond);
  }
  const int t = _m3_dit->i8_tier();
  session()->info(fmt(
      "GenerateAudioStage('{}'): MiniMax-Music3's flow matching loaded in "
      "{:.1f} s ({:.2f} GB; DiT {} on {} GEMMs, {} attention)", this->id(),
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count(),
      (double)_m3_dit->resident_bytes() / 1e9,
      t < 0    ? std::string("f16")
      : t == 0 ? std::string("f16, int8 GEMMs (accurate)")
               : fmt("held as int8 ({})",
                     genai::MetalMusic3Dit::i8_tier_name(t))(),
      _m3_dit->uses_mma() ? "matmul2d" : "steel",
      _m3_dit->uses_attn_nax() ? "NAX" : "steel"));
  return true;
}

void
GenerateAudioStage::release_music3_dirs_(int first, int last)
{
  // The phase claims' other half: the bytes really given back (a pooled
  // set keeps a DiT's derived entries resident -- they cannot be parked),
  // reported, and SETTLED, since macOS returns freed Metal memory after
  // the free and the next half's load should not land on top of it.
  auto* mgr = session()->services()->generative_model_manager();
  const auto dirs = music3_dirs_();
  for (int i = first; i <= last; ++i) {
    if (mgr != nullptr) {
      mgr->drop_weights(dirs[(std::size_t)i]);
      mgr->note_phase_released(dirs[(std::size_t)i]);
    }
  }
  if (auto* mc = session()->services()->metal_compute()) {
    (void)mc->memory_budget_settled();
  }
}

void
GenerateAudioStage::release_music3_ar_()
{
  if (!_m3_ar) { return; }
  _m3_ar.reset();
  release_music3_dirs_(0, 1);
}

void
GenerateAudioStage::release_music3_flow_()
{
  if (!_m3_dit) { return; }
  _m3_dit.reset();
  release_music3_dirs_(2, 3);
}

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
  str("caption", &r->style);
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
  if (family() == Family::kMusic3) {
    namespace m3 = genai::music3;
    // Taking turns, a song starts with the AR half alone: the last song's
    // flow half let go (it already was, unless that song failed) and this
    // one's language model loaded again.
    if (_tok && _m3_swap) {
      release_music3_flow_();
      (void)load_music3_ar_();
    }
    if (!_tok || !_m3_ar || (!_m3_swap && !_m3_dit)) {
      session()->warn(fmt("GenerateAudioStage('{}'): no model; skipping",
                          this->id()));
      finish_one_shot();
      co_return;
    }
    if (req.style.empty() || req.lyrics.empty()) {
      session()->warn(fmt("GenerateAudioStage('{}'): MiniMax-Music3 needs "
                          "both a caption (style) and lyrics; skipping",
                          this->id()));
      finish_one_shot();
      co_return;
    }
    const std::vector<std::int32_t> ids = m3::encode_text(
        *_tok, _m3_added, m3::prompt_text(req.style, req.lyrics));
    if ((int)ids.size() > m3::kMaxPromptTokens) {
      session()->warn(fmt("GenerateAudioStage('{}'): the prompt is {} "
                          "tokens; MiniMax-Music3 takes at most {}; "
                          "skipping", this->id(), ids.size(),
                          m3::kMaxPromptTokens));
      finish_one_shot();
      co_return;
    }
    // int(seconds * 25), as the reference converts audio_duration.
    int max_frames = m3::kMaxFrames;
    if (_max_seconds > 0.0) {
      max_frames = std::max(
          1, std::min(max_frames,
                      (int)(_max_seconds * m3::kFramesPerSecond)));
    }
    revise_song_bytes_(song_bytes_((int)ids.size(), max_frames, false));
    UiProgress bar = session()->open_progress("song");
    auto stop = [&]() { return ctx.stop_requested(); };
    std::string err;
    genai::MetalMusic3Ar::Song song;
    const float ar_cfg = req.cfg_scale >= 0.0f ? req.cfg_scale : m3::kArCfg;
    const bool ok = _m3_ar->generate(
        ids, max_frames, req.seed,
        [&](int f) {
          bar.update(0, 0, fmt("writing the song: {:.1f} s",
                               (double)f / m3::kFramesPerSecond)());
          return !stop();
        },
        &song, &err, ar_cfg);
    // The codes are written: taking turns, the language model goes before
    // the flow matching loads.
    if (_m3_swap) { release_music3_ar_(); }
    if (!ok) {
      session()->warn(fmt("GenerateAudioStage('{}'): the song phase "
                          "failed: {}", this->id(), err));
      finish_one_shot();
      co_return;
    }
    if (song.stopped) { co_return; }
    if (song.frames == 0) {
      session()->warn(fmt("GenerateAudioStage('{}'): the model ended the "
                          "song before its first frame; nothing to emit",
                          this->id()));
      finish_one_shot();
      co_return;
    }
    const double ar_s = song.seconds - song.prefill_seconds;
    session()->info(fmt(
        "GenerateAudioStage('{}'): song written -- {} frames ({:.1f} s of "
        "audio{}) in {:.1f} s ({:.1f} frames/s, prompt {} tokens in "
        "{:.1f} s)", this->id(), song.frames,
        (double)song.frames / m3::kFramesPerSecond,
        song.ended ? ", the model ended it" : ", at the cap", song.seconds,
        ar_s > 0 ? (double)song.frames / ar_s : 0.0, ids.size(),
        song.prefill_seconds));
    const int steps = _ode_steps > 0 ? _ode_steps : m3::kDitSteps;
    const float cfg = _flow_cfg >= 0.0 ? (float)_flow_cfg : m3::kDitCfg;
    if (_m3_swap && !load_music3_flow_()) {
      finish_one_shot();
      co_return;
    }
    std::vector<float> lat;
    const auto t_nar = std::chrono::steady_clock::now();
    const bool ok2 = _m3_dit->synthesize(
        song.hiddens.data(), song.frames, req.seed, steps, cfg, &lat,
        [&](int done, int total) {
          bar.update((std::uint64_t)done, (std::uint64_t)total,
                     "synthesizing");
          return !stop();
        },
        &err);
    if (!ok2) {
      if (!err.empty()) {
        session()->warn(fmt("GenerateAudioStage('{}'): flow matching "
                            "failed: {}", this->id(), err));
      }
      finish_one_shot();
      co_return;
    }
    const auto& st = _m3_dit->stats();
    session()->info(fmt(
        "GenerateAudioStage('{}'): latents in {:.1f} s ({} windows x {} "
        "steps, {:.0f} ms an evaluation of both guidance rows)", this->id(),
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      t_nar).count(),
        st.windows, steps, st.evals > 0 ? st.ms / st.evals : 0.0));
    // Taking turns, the flow half goes before the decode downstream.
    if (_m3_swap) { release_music3_flow_(); }
    bar.finish();
    const int Z = m3::kLatentChannels;
    const int T = (int)(lat.size() / (std::size_t)Z);
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::F32;
    out->shape = {1, (std::int64_t)Z, (std::int64_t)T};
    out->resize_contiguous(lat.size());
    std::memcpy(out->as_f32(), lat.data(), lat.size() * sizeof(float));
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign(
        sideband::kLatentsPerSecond,
        FlexData::make_real(m3::kLatentsPerSecond));
    out->sideband = std::move(sb);
    provenance::set_model_name(out->sideband, _hf_dir);
    co_await ctx.write(0, std::move(out));
    ++_songs;
    finish_one_shot();
    co_return;
  }
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
  // This song's bound, now that its prefix, budget and guidance are known:
  // the claims rise to it before anything is allocated for it.
  revise_song_bytes_(song_bytes_(
      (int)std::max(prefix.size(), negative.size()), ss.max_tokens,
      guide != 1.0f));
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
        prefix, codes, req.seed, _ode_steps > 0 ? _ode_steps : 32, &lat,
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
    // And what the song left held beside the weights against what the
    // claims book for it: a figure above the booking is a plan that
    // under-reads, and this line is where it shows.
    session()->info(fmt(
        "GenerateAudioStage('{}'): latents in {:.1f} s ({} steps, {:.0f} ms "
        "an evaluation{}); {} MB held beside the weights, {} MB booked",
        this->id(), nar_s, _ode_steps > 0 ? _ode_steps : 32,
        st.evals > 0 ? st.ms / st.evals : 0.0, tiers,
        _model->held_scratch_bytes() >> 20, _booked.held >> 20));
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
