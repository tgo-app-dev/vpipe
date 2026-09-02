#include "stages/generate-video-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-config-source.h"
#include "stages/model-registry.h"
#include "stages/denoise-progress.h"
#include "stages/model-provenance.h"

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"
#include "generative-models/wan/metal-wan-vae.h"
#include "generative-models/weight-set.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

namespace {

// How many MiniMax-H3 runtime-LoRA slots this stage carries beats for.
// Must not exceed MetalMiniMaxH3Transformer::kMaxLoraSlots, which is the
// authority; it is repeated rather than referenced because this file's
// slot bookkeeping lives outside the Apple-Silicon guard and the
// transformer header does not exist on the other side of it. A stage
// that named more than the DiT has slots would have its extras warned
// about and dropped there, which is a message rather than a bug.
constexpr int kH3LoraSlots = 2;

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "video model root. The layout is the resident family's own -- an "
          "upstream tree of per-component directories, or a repack that "
          "names them differently -- and each registered family claims the "
          "roots it recognises, so this takes the checkpoint root and the "
          "family that claims it decides. OPTIONAL: a model-select source "
          "on the model iport overrides it",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "wan-i2v,wan-t2v,minimax-h3-fl2va,minimax-h3-ref2va",
   .model_channel = "diffusion-model"},
  {.key = "height", .type = ConfigType::Int, .required = false,
   .doc = "video height in pixels. ROUNDED UP to the resident family's VAE "
          "stride times its DiT patch, which differs per family, so any "
          "positive value is accepted and the stage logs what it used",
   .def_int = 480},
  {.key = "width", .type = ConfigType::Int, .required = false,
   .doc = "video width in pixels; same multiple as height", .def_int = 832},
  {.key = "frames", .type = ConfigType::Int, .required = false,
   .doc = "video frames. Rounded UP to the nearest count the resident "
          "model's VAE can chunk, which differs per family, so any positive "
          "number is accepted here", .def_int = 81},
  {.key = "fps", .type = ConfigType::Real, .required = false,
   .doc = "frame rate stamped on the emitted latent, for the decoder and the "
          "encoder downstream of it. This stage does not resample: the value "
          "describes the clip the family generated, so a family conditioned "
          "at another rate wants it set",
   .def_real = 24.0},
  {.key = "steps", .type = ConfigType::Int, .required = false,
   .doc = "denoising steps", .def_int = 40},
  {.key = "seed", .type = ConfigType::Int, .required = false,
   .doc = "initial-noise RNG seed (default 0)"},
  {.key = "i8_gemm", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated mode (LOSSY): dynamic-int8 GEMMs for the DiT's big "
          "block matmuls instead of bf16, at int8 quality. Taken only by a "
          "family whose forward materializes a dense weight for the matrix "
          "cores; a steel-only family IGNORES it, as does any box without "
          "NAX matmul2d. It also self-gates on ROWS "
          "(>= 1024), so a short clip keeps the bf16 tiles either way, and "
          "on how much a projection would have to be zero-padded to reach a "
          "whole int8 chunk (refused past 10% extra MACs). "
          "MEASURED on minimax-h3, 2 blocks at production geometry: 2.02-2.11 "
          "s/step -> 1.52-1.55 s (1.35x), with the step's velocity rms moving "
          "0.11%. Default false; env VPIPE_I8_GEMM overrides",
   .def_bool = false},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "drop the resident model's weights after each clip and reload on "
          "the next one. \"auto\" (default) decides from physical RAM vs the "
          "pipeline's weight bytes; \"always\" / \"never\" force it",
   .def_str = "auto"},
};
const PortSpec kIports[] = {
  {.name = "conditioning",
   .doc = "text hidden states from a diffusion-conditioner: bf16 [text_seq, "
          "dim] at the resident family's own encoder width. The conditioner "
          "resolves the same checkpoint this stage does, so the width "
          "matches by construction rather than by configuration",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
  {.name = "neg_conditioning",
   .doc = "OPTIONAL negative conditioning (the conditioner's oport1) for "
          "classifier-free guidance. A family that is NOT guidance-distilled "
          "needs it: without one its guidance is forced to 1. A "
          "guidance-distilled family runs a single forward per step and "
          "ignores it",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
  {.name = "model", .doc = "OPTIONAL shared model reference from a model-select "
                           "source; overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "sampler",
   .doc = "OPTIONAL sampler spec FlexData (diffusion-sampler-select). Read "
          "by a family that samples from a spec, in place of the sampler it "
          "ships; inert for a family that carries its own sampler and its "
          "own schedule, which is shaped through that family's model-config "
          "source instead",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "scheduler",
   .doc = "OPTIONAL scheduler spec FlexData (scheduler-select). Inert for the "
          "same families the sampler port is inert for",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "ref_latent0",
   .doc = "OPTIONAL image-to-video conditioning latent from vae-encode, in "
          "the resident family's own latent geometry. Which SHAPE depends on "
          "how the family conditions: a clip-shaped tensor (the conditioning "
          "image followed by blank frames) or a single FIRST-frame keyframe "
          "anchor. Present => image-to-video",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "ref_latent1",
   .doc = "OPTIONAL LAST-frame keyframe anchor from a second vae-encode, "
          "for a family that anchors on keyframes rather than on one "
          "clip-shaped latent. With ref_latent0 this is the first-and-last "
          "mode",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "ref_video_rows",
   .doc = "OPTIONAL reference VIDEO rows from a video-ref-encoder: f32 "
          "[rows, 96], for a family with a reference-conditioned partition. "
          "Already packed into DiT rows and concatenated in reference "
          "order, because every reference is encoded at a resolution of its "
          "own and rows are the only shape they share. Wiring it is what "
          "puts this stage in its reference-conditioned mode; the geometry "
          "the rows belong to rides on the conditioning beat's sideband",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "ref_audio_rows",
   .doc = "OPTIONAL reference AUDIO rows from the same video-ref-encoder: "
          "f32 [rows, 32], channel-major within a reference. Rides along at "
          "a clean timestep and is never denoised",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
  {.name = "model_config",
   .doc = "OPTIONAL model-specific parameters from the resident family's "
          "own config source -- each family ships one, carrying whatever it "
          "alone has to say (guidance and expert boundaries, sigma shifts, "
          "condition timesteps, audio duration). Passed to that family's "
          "GenerationParams UNREAD, so this stage carries none of the knobs. "
          "Unwired, the family runs its documented defaults",
   .type = &typeid(FlexDataPayload),
   .tags = "model-config", .clock_group = 0},
  {.name = "audio_conditioning",
   .doc = "OPTIONAL second-modality conditioning for a family that "
          "generates a soundtrack from its own projection of the caption: "
          "an AUDIO context beside the video one on iport0, at a width of "
          "its own. A separate port because it is a different WIDTH from a "
          "different projection, so packing both into one beat would make "
          "the split a convention rather than a type. Unwired, such a "
          "family generates VIDEO ONLY and says so; it must not invent a "
          "zero context, which would be conditioning that says silence "
          "rather than nothing. Ignored by a family that takes no second "
          "context",
   .type = &typeid(TensorBeatPayload),
   .tags = "conditioning", .clock_group = 0},
};
[[maybe_unused]] constexpr unsigned kModelPort   = 2;
[[maybe_unused]] constexpr unsigned kSamplerPort = 3;
[[maybe_unused]] constexpr unsigned kSchedPort   = 4;
[[maybe_unused]] constexpr unsigned kRefPort     = 5;
[[maybe_unused]] constexpr unsigned kRefPort1    = 6;
[[maybe_unused]] constexpr unsigned kRefVideoRowsPort = 7;
[[maybe_unused]] constexpr unsigned kRefAudioRowsPort = 8;
[[maybe_unused]] constexpr unsigned kModelCfgPort     = 9;
[[maybe_unused]] constexpr unsigned kAudioCondPort    = 10;

// The keys that MOVED to the per-family config stages. Named here so a
// pipeline written against the old union says what to do instead of
// silently running the defaults -- an unknown key is not an error
// anywhere in this runtime, which is exactly what makes a quiet removal
// the wrong way to do this.
struct MovedKey { const char* key; const char* to; };
constexpr MovedKey kMovedKeys[] = {
  {"guidance_scale",     "wan2-model-config"},
  {"guidance_scale_2",   "wan2-model-config"},
  {"boundary_ratio",     "wan2-model-config"},
  {"video_shift",        "minimax-h3-model-config"},
  {"audio_shift",        "minimax-h3-model-config"},
  {"condition_timestep", "minimax-h3-model-config"},
  {"audio_seconds",      "minimax-h3-model-config"},
};

const PortSpec kOports[] = {
  {.name = "latent",
   .doc = "f32 latent VIDEO [z, T, H/r, W/r] (whitened), sideband "
          "{fps, frames}. z/r are the resident family's VAE geometry",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
  {.name = "audio_latent",
   .doc = "f32 latent AUDIO [audio_channels, audio_latents] from a family "
          "that generates a soundtrack -- the families that do not never "
          "write it",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "generate-video",
  .doc       = "Video DiT denoiser: conditioning (+ optional keyframe or "
               "reference latents) -> the resident family's sampler over its "
               "transformer -> a latent VIDEO, and a latent SOUNDTRACK from "
               "the families that generate one, on the metal-compute backend. "
               "Feed vae-decode (and audio-vae-decode).",
  .display_name = "Generate Video",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

GenerateVideoStage::GenerateVideoStage(const SessionContextIntf* s,
                                       std::string               id,
                                       std::vector<InEdge>       iports,
                                       FlexData                  config)
  : TypedStage<GenerateVideoStage>(s, std::move(id), std::move(iports),
                                   std::move(config))
{
  _hf_dir = attr_str("hf_dir");
  _height = (int)attr_int("height");
  _width  = (int)attr_int("width");
  _frames = (int)attr_int("frames");
  _fps    = attr_real("fps");
  _steps  = (int)attr_int("steps");
  _i8_gemm = attr_bool("i8_gemm");
  _seed   = (std::uint64_t)attr_int("seed");
  // The family-specific keys are gone from this stage; a pipeline still
  // carrying one gets told where it went. Warning rather than failing
  // because the key is now inert, not wrong -- the graph runs, it just
  // runs on defaults, and that is precisely the outcome that needs to be
  // said out loud.
  {
    auto o = this->config().as_object();
    for (const auto& mk : kMovedKeys) {
      if (o.contains(mk.key)) {
        session()->warn(fmt(
            "GenerateVideoStage('{}'): config key '{}' moved to the '{}' "
            "stage and is IGNORED here; wire one to the model_config iport "
            "or the family runs its defaults", this->id(), mk.key, mk.to));
      }
    }
  }
  if (_height <= 0) { _height = 480; }
  if (_width  <= 0) { _width  = 832; }
  if (_frames <= 0) { _frames = 81; }
  if (_steps  <= 0) { _steps  = 40; }
  if (!(_fps > 0.0)) { _fps = 24.0; }
  if (_height <= 0) { _height = 480; }
  if (_width  <= 0) { _width  = 832; }
  // The frame SIZE is not validated here either, and for exactly the
  // reason the frame COUNT is not (see below): the legal grid is the
  // family's VAE stride times its DiT patch, which differs per family,
  // and the family is not known until the checkpoint is read.
  // resolve_config_ rounds UP to the resident family's grid.
  //
  // This used to refuse anything that was not a multiple of 16, which was
  // both too strict (a caller should not have to know) and too loose: on
  // a coarser grid it passed sizes that cannot be patched at all, e.g.
  // 1360, whose latent width at stride 16 is 85 -- odd, and so not
  // divisible by the patch.
  // The frame count is NOT validated here, deliberately. A count the VAE
  // cannot chunk has no latent representation -- but the rule is PER
  // FAMILY, and the shapes are unrelated to each other: one family
  // compresses in 4-frame chunks after a 1-frame first chunk (4k+1),
  // another takes 17-frame clips keeping 5 latents (17n+5). The family
  // is only known once the checkpoint is read, which happens after this
  // constructor, so resolve_config_ rounds UP to the resident family's
  // rule instead and any positive count is accepted.
  //
  // Rounding rather than rejecting because such rules share almost no
  // legal counts -- 81 is legal under 4k+1 and impossible under 17n+5;
  // 56 is the reverse -- so a graph carrying one family's number could
  // not change checkpoints without being re-authored, which is exactly
  // what being family-generic is supposed to buy.
#ifdef VPIPE_BUILD_APPLE_SILICON
  {
    bool bad = false;
    _unload_cfg = model_memory::parse_unload_policy(
        attr_str("unload_when_idle"), &bad);
    if (bad) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): unload_when_idle '{}' is not "
          "auto|always|never; using auto", this->id(),
          attr_str("unload_when_idle")));
    }
  }
  // UniPC multistep on the checkpoint's flow schedule: the spec this
  // stage starts from for a family that samples from one. A family
  // carrying its own sampler ignores it, and a sampler-select source on
  // the sampler iport overrides it.
  _sampler_spec.method = "unipc";
  _sampler_spec.order = 2;
  _sampler_spec.solver_bh2 = true;
  _scheduler_spec.type = "unipc_flow";
  _scheduler_spec.steps = _steps;
  _scheduler_spec.shift = 3.0;
  _scheduler_spec.num_train = 1000;
#endif
  allocate_oports(spec().oports.size());
}

GenerateVideoStage::~GenerateVideoStage() = default;

const StageSpec&
GenerateVideoStage::spec() const noexcept
{
  return kSpec;
}

int
GenerateVideoStage::latent_frames() const noexcept
{
  return _frames > 0 ? 1 + (_frames - 1) / 4 : 0;
}

void
GenerateVideoStage::align_frames_(int aligned)
{
  if (aligned <= 0 || aligned == _frames) { return; }
  // SAY SO. The clip that comes back is longer than the one that was
  // asked for, its duration at `fps` is longer to match, and a graph
  // downstream that assumed the configured count would otherwise find out
  // by way of a shape it did not expect. One line naming both numbers is
  // cheaper than that, and it is info rather than a warning because
  // rounding is the designed behaviour, not a problem being tolerated.
  session()->info(fmt(
      "GenerateVideoStage('{}'): frames {} -> {}, the nearest count at or "
      "above it that the {} VAE can chunk", this->id(), _frames, aligned,
      _family));
  _frames = aligned;
}

// Round the frame SIZE up to what the resident family can tile, same
// contract as align_frames_ and for the same reason: the rule is the
// VAE's spatial stride times the DiT's patch, both of which are
// properties of the checkpoint.
void
GenerateVideoStage::align_size_(int gh, int gw)
{
  if (gh <= 0 || gw <= 0) { return; }
  const int h = ((_height + gh - 1) / gh) * gh;
  const int w = ((_width  + gw - 1) / gw) * gw;
  if (h == _height && w == _width) { return; }
  // SAY SO, for the reason align_frames_ does: the clip that comes back
  // is a different shape from the one that was asked for, and a graph
  // downstream would otherwise discover that as a surprise.
  session()->info(fmt(
      "GenerateVideoStage('{}'): {}x{} -> {}x{}, the nearest size at or "
      "above it that the {} VAE and DiT patch can tile", this->id(),
      _width, _height, w, h, _family));
  _height = h;
  _width  = w;
}

void
GenerateVideoStage::apply_constant(unsigned iport, const FlexData& beat)
{
  // Pre-launch twin of the runtime latch in process(): the same
  // beat and the same parse, early enough that declare_resources()
  // sees the model. Bookkeeping only -- nothing loads here; the
  // pipeline is not assembled yet (see Stage::apply_constant).
  if (iport != kModelPort) { return; }
  apply_model_select_beat(beat, _hf_dir);
}

// Geometry as the family will actually produce it, for the plan-time
// arena estimate. False when the family's rounding rule could not be
// determined -- the caller must NOT then treat the geometry as a bound.
//
// Every video model constrains its latent shape, so the clip that comes
// back is rounded UP from what config asked for, and the accounting has
// to use the rounded numbers or it describes a clip nobody will make.
// MEASURED on MiniMax-H3: config 9 frames, produced 22, so the
// unrounded estimate was 2.4x short -- and an arena that under-declares
// reads as room that is not there.
//
// The same rules resolve_config_ applies, from the same sources. Each
// arm is a cheap query by construction: a registered family answers
// without loading, and the built-in arms read one JSON (or, for a
// Comfy-Org repack, the safetensors __metadata__ that stands in for it).
//
// RETURNING FALSE MATTERS AS MUCH AS THE ROUNDING. A source none of
// these arms recognises -- a GGUF DiT, say, which has no config.json and
// no metadata envelope -- would otherwise fall through with the
// unrounded numbers, which is not a conservative bound but an
// optimistic one, and it is exactly the bug this function exists to fix
// returning silently.
bool
GenerateVideoStage::planned_geometry_(const std::string& root, int* out_w,
                                      int* out_h, int* out_frames) const
{
  int w = _width, h = _height, frames = _frames;
  bool known = false;
#ifdef VPIPE_BUILD_APPLE_SILICON
  auto round_up = [](int v, int g) {
    return g > 0 ? ((v + g - 1) / g) * g : v;
  };
  if (genai::VideoModelFamily* f =
          genai::VideoModelRegistry::get().claim_for(
              session(), root, resolve_model(session(), _hf_dir).model_type)) {
    const int af = f->align_frames(root, frames);
    if (af > 0) { frames = af; }
    int gh = 0, gw = 0;
    f->size_grid(root, &gh, &gw);
    h = round_up(h, gh);
    w = round_up(w, gw);
    known = true;
  } else {
    // PROBE, do not ask the record. The models DB commonly says nothing
    // -- the shipped H3 graph logs "model_type '-', so probed the
    // checkpoint" -- so a family test keyed on model_type never fires
    // and the rounding silently does not happen. config_from_json is
    // the same probe resolve_config_ uses and refuses a checkpoint that
    // is not its own, so a success is the detection and the patch sizes
    // at once. It reads a diffusers config.json OR a Comfy-Org repack's
    // __metadata__, which is why no separate repack arm is needed.
    namespace fs = std::filesystem;
    std::string err;
    genai::MetalMiniMaxH3Transformer::Config h3;
    genai::MetalWanTransformer::Config wan;
    const std::string part =
        genai::MetalMiniMaxH3Transformer::partition_of_model_type(
            resolve_model(session(), _hf_dir).model_type);
    if (genai::MetalMiniMaxH3Transformer::config_from_json(root, h3, &err,
                                                           part)) {
      // 17-frame chunks keeping 5 latents each, so only 17n+5 has a
      // latent form -- 9 becomes 22, which is the whole of the 2.4x.
      frames = genai::minimax_h3::align_num_frames(frames, 17, 5);
      h = round_up(h, 16 * h3.patch_h);      // 16x VAE times the patch
      w = round_up(w, 16 * h3.patch_w);
      known = true;
    } else if (genai::MetalWanTransformer::config_from_json(
                   (fs::path(root) / "transformer").string(), wan, &err)) {
      frames = genai::MetalWanVae::align_num_frames(frames);
      h = round_up(h, 8 * wan.patch_h);      // wan's VAE is 8x spatial
      w = round_up(w, 8 * wan.patch_w);
      known = true;
    }
  }
#else
  (void)root;
#endif
  if (out_w != nullptr)      { *out_w = w; }
  if (out_h != nullptr)      { *out_h = h; }
  if (out_frames != nullptr) { *out_frames = frames; }
  return known;
}

std::vector<ResourceClaim>
GenerateVideoStage::declare_resources() const
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_hf_dir.empty()) { return {}; }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  // A plugin family declares its own, for the same reason the built-ins
  // do: this runs BEFORE any driver starts, and a checkpoint that
  // declares nothing is invisible to every peer sizing itself against
  // it. Asked of the registry rather than of `_plugin_family`, because
  // this is const and runs before resolve_config_ has latched anything.
  // The decode ARENA this stage's output implies, declared here because
  // nothing downstream can: vae-decode sizes from whatever latent
  // arrives, so at planning time it cannot name a number. For a clip the
  // dominant transient is the OUTPUT, and it grows linearly with length
  // -- 9 bytes per output pixel across the decode's bf16 frames and the
  // planar-U8 clip buffered behind them.
  //
  // A BOUND. The true figure needs the pixel-frame count the VAE
  // expands the latent into, which is a property of the loaded model, so
  // vae-decode revises this on every beat. Declaring the config geometry
  // first is what gives peers something to size against before the first
  // clip exists -- and what gives the revise something to correct.
  // A bound only when the geometry is known to be the produced one.
  // Otherwise the MARKER: an unrounded estimate is not a conservative
  // bound, it is an optimistic one, and a presence marker that the
  // first beat replaces is the honest answer to "how big will this be".
  int aw = _width, ah = _height, af = _frames;
  std::size_t arena_bytes = model_memory::kUnknownArena;
  if (planned_geometry_(root, &aw, &ah, &af)) {
    arena_bytes = model_memory::video_decode_scratch_bytes(aw, ah, af);
  } else {
    session()->log_debug(fmt(
        "GenerateVideoStage('{}'): no rounding rule for '{}', so the decode "
        "arena is declared as present-but-unsized and corrected on the "
        "first clip", this->id(), root));
  }
  // No geometry to size from: a presence marker, so the per-beat figure
  // has an entry to revise. See model_memory::kUnknownArena.
  if (arena_bytes == 0) { arena_bytes = model_memory::kUnknownArena; }
  std::vector<ResourceClaim> arena = model_memory::scratch_claims(
      "vae-decode", arena_bytes, model_memory::kPhaseDecode);
  // The LATENT this stage emits, alive from the denoise that writes it
  // through the last decode that reads it.
  //
  // It is the clearest case of a term no weights-and-scratch accounting
  // can see: nothing HOLDS it -- it is a payload in flight between
  // stages -- and yet on a constrained box it is resident across three
  // phases and is frequently larger than the VAE decoding it. Sized from
  // the same rounded geometry the arena uses, so the two cannot describe
  // different clips.
  // Whose formulas size the beat-shaped terms. A registered family's
  // latent and soundtrack are ITS shape -- channel count, spatial and
  // temporal compression all differ -- so a built-in's formula over a
  // plugin's geometry is a confident number for the wrong model. Asked
  // once, here, and used for both the payload sizes below and the
  // holdings in declare_memory().
  genai::VideoModelFamily* fam = genai::VideoModelRegistry::get().claim_for(
      session(), root, resolve_model(session(), _hf_dir).model_type);

  if (arena_bytes != model_memory::kUnknownArena && aw > 0 && ah > 0 &&
      af > 0) {
    // 0 from a family that declines to size its own latent, and the
    // claim is then skipped rather than filled in from a built-in.
    const std::size_t latent =
        fam != nullptr
            ? fam->latent_bytes(root, aw, ah, af)
            : genai::MetalMiniMaxH3VideoVae::latent_bytes(aw, ah, af);
    if (latent > 0) {
      for (auto& c : model_memory::payload_claims(
               fmt("{}-latent", this->id())(), latent,
               model_memory::kPhaseDenoise, model_memory::kPhaseDecode)) {
        arena.push_back(std::move(c));
      }
    }
    // The SOUNDTRACK, on the same principle: its length comes from this
    // stage's frames/fps, so the audio decode downstream cannot size
    // itself and this is the only place that can. Plan-time seconds are
    // the clip's own length -- what `audio_seconds: 0` means -- and the
    // model-config beat that could shorten it does not arrive until
    // after the barrier, so the runtime revises rather than the plan
    // guessing.
    std::size_t alat = 0, pcm = 0, aarena = 0;
    bool has_audio = false;
    if (_fps > 0.0) {
      if (fam != nullptr) {
        has_audio = fam->audio_cost(root, af, _fps, &alat, &pcm, &aarena);
      } else {
        const int alf =
            genai::MetalMiniMaxH3AudioVae::latent_frames_for_seconds(
                (double)af / _fps);
        genai::MetalMiniMaxH3AudioVae::decode_cost(alf, &pcm, &aarena);
        has_audio = true;
      }
    }
    if (has_audio) {
      for (auto& c : model_memory::scratch_claims(
               fmt("{}-audio-decode", this->id())(), aarena,
               model_memory::kPhaseDecodeAudio)) {
        arena.push_back(std::move(c));
      }
      // Alive until the mux reads it, which is after the video decode.
      for (auto& c : model_memory::payload_claims(
               fmt("{}-pcm", this->id())(), pcm,
               model_memory::kPhaseDecodeAudio, model_memory::kPhaseDecode)) {
        arena.push_back(std::move(c));
      }
    }
  }

  if (fam != nullptr) {
    std::vector<ResourceClaim> out = fam->declare_resources(root);
    for (auto& c : arena) { out.push_back(std::move(c)); }
    return out;
  }
  // ONE expert, not both. The stage holds exactly one at a time, so
  // claiming the pair would size every peer against a peak that never
  // occurs and push them all into streaming for nothing.
  //
  // Through the DiT resolver, because this runs BEFORE resolve_config_ and
  // so before the family is known -- and a repack spells its DiT
  // `diffusion_models/`, not `transformer/`. Claiming a path that does not
  // exist is not a harmless miss: this is the declaration every peer sizes
  // itself against, so a 24 GB DiT reported as 0 bytes is exactly the
  // silent under-count the resource-planning phase exists to prevent.
  // Falls back to the diffusers spelling when nothing resolves.
  // Declared BEFORE anything loads, so the partition can only come from
  // the reference itself -- resolve_model() reads the record without
  // touching the weights.
  const std::string part =
      genai::MetalMiniMaxH3Transformer::partition_of_model_type(
          resolve_model(session(), _hf_dir).model_type);
  std::string dit =
      genai::MetalMiniMaxH3Transformer::resolve_dit_dir(root, part);
  if (dit == root) { dit = (fs::path(root) / "transformer").string(); }
  // BOTH numbers: what this DiT weighs, and the floor it can be reduced
  // to by streaming its blocks. A graph that does not fit the first and
  // does fit the second is not one to refuse -- it is one to stream.
  std::vector<ResourceClaim> out{model_memory::weight_claim_streamable(
      dit, genai::MetalMiniMaxH3Transformer::streaming_floor_bytes(dit))};
  for (auto& c : arena) { out.push_back(std::move(c)); }
  return out;
#else
  return {};
#endif
}

StageMemory
GenerateVideoStage::declare_memory() const
{
  StageMemory m;
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_hf_dir.empty()) { return m; }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);

  // A registered family answers for itself, in both ledgers. Without
  // this arm the built-in resolvers below run over a plugin's root and
  // produce not a gap but a WRONG holding -- an H3-shaped directory
  // guess, sized by H3's formulas, named as the source the pooling path
  // will later look for. A hole is recoverable; a confident wrong number
  // is what a peer sizes an irreversible decision against.
  genai::VideoModelFamily* fam = genai::VideoModelRegistry::get().claim_for(
      session(), root, resolve_model(session(), _hf_dir).model_type);

  if (fam != nullptr) {
    m.holdings = fam->declare_holdings(root);
    // The POLICY is the stage's, not the family's: `unload_when_idle` is
    // this stage's config key, and a family cannot see it. The family
    // said what it holds and how small it can get; this says what
    // happens to it when idle.
    for (StageHolding& h : m.holdings) {
      h.releases    = _unload_cfg == model_memory::UnloadPolicy::kDestroy;
      h.reclaimable = _unload_cfg == model_memory::UnloadPolicy::kAuto;
    }
    // The first holding names the checkpoint the pooling path releases.
    // Same reason as the built-in below: this method runs for every
    // policy where decide_resources() returns early unless it is
    // `destroy`.
    if (!m.holdings.empty()) { _dit_dir_declared_ = m.holdings.front().source; }
    int fw = _width, fh = _height, ff = _frames;
    if (!planned_geometry_(root, &fw, &fh, &ff)) { return m; }
    m.outputs.resize(2, 0);
    m.outputs[0] = fam->latent_bytes(root, fw, fh, ff);
    std::size_t alat = 0, apcm = 0, aarena = 0;
    if (_fps > 0.0 &&
        fam->audio_cost(root, ff, _fps, &alat, &apcm, &aarena)) {
      m.outputs[1] = alat;
    }
    return m;
  }

  const std::string part =
      genai::MetalMiniMaxH3Transformer::partition_of_model_type(
          resolve_model(session(), _hf_dir).model_type);
  std::string dit =
      genai::MetalMiniMaxH3Transformer::resolve_dit_dir(root, part);
  if (dit == root) { dit = (fs::path(root) / "transformer").string(); }
  // NAMED by its directory, so two stages over one checkpoint are one
  // set of weights in the plan rather than two.
  // Remembered HERE as well as in decide_resources, because this method
  // runs for every policy and that one returns early unless the policy
  // is `destroy` -- so under `auto` the pooling below had an empty
  // string and did nothing, silently.
  _dit_dir_declared_ = dit;
  m.hold(dit, model_memory::dir_weights_bytes(dit),
         genai::MetalMiniMaxH3Transformer::streaming_floor_bytes(dit),
         // `destroy` only, the same condition the phase claim carries:
         // `auto` resolves against a figure this would itself change.
         _unload_cfg == model_memory::UnloadPolicy::kDestroy,
         // ...and `auto` is the RECLAIMABLE case -- held while there is
         // room, given back under pressure, found still there or
         // reloaded when next used.
         _unload_cfg == model_memory::UnloadPolicy::kAuto);

  int aw = _width, ah = _height, af = _frames;
  if (!planned_geometry_(root, &aw, &ah, &af)) { return m; }
  // oport 0 is the video latent, oport 1 the audio latent. Sized from
  // the same rounded geometry the arena uses, so the two cannot
  // describe different clips.
  m.outputs.resize(2, 0);
  m.outputs[0] = genai::MetalMiniMaxH3VideoVae::latent_bytes(aw, ah, af);
  if (_fps > 0.0) {
    const int alf = genai::MetalMiniMaxH3AudioVae::latent_frames_for_seconds(
        (double)af / _fps);
    std::size_t pcm = 0, aarena = 0;
    genai::MetalMiniMaxH3AudioVae::decode_cost(alf, &pcm, &aarena);
    (void)aarena;
    // The audio LATENT, not the PCM: what leaves this stage is what the
    // audio VAE will decode, and it is the decode that produces samples.
    m.outputs[1] = (std::size_t)alf * 2 * 32 * 2;    // [stereo, 32, frames]
  }
#endif
  return m;
}

std::vector<ResourceClaim>
GenerateVideoStage::decide_resources() const
{
#ifdef VPIPE_BUILD_APPLE_SILICON
  // ONLY when the release is certain from config. `destroy` qualifies;
  // `auto` does not, because it resolves against a figure this claim
  // would itself change -- the stage would see the room its own phase
  // invented and conclude it need not drop the DiT after all.
  if (_hf_dir.empty() ||
      _unload_cfg != model_memory::UnloadPolicy::kDestroy) {
    return {};
  }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const std::string part =
      genai::MetalMiniMaxH3Transformer::partition_of_model_type(
          resolve_model(session(), _hf_dir).model_type);
  std::string dit =
      genai::MetalMiniMaxH3Transformer::resolve_dit_dir(root, part);
  if (dit == root) { dit = (fs::path(root) / "transformer").string(); }
  // Remembered, so the release below names exactly what was claimed --
  // the resolver can answer differently once the model has loaded, and
  // a release under a different name reads as a promise never kept.
  _dit_dir_declared_ = dit;
  return model_memory::weight_claims_in_phase({dit},
                                              model_memory::kPhaseDenoise);
#else
  return {};
#endif
}

void
GenerateVideoStage::reset_run_state()
{
  _emitted = 0;
  _model_latched = false;
  _sampler_latched = false;
  _scheduler_latched = false;
  _cfg_latched = false;
  _model_cfg = FlexData{};
}

#ifdef VPIPE_BUILD_APPLE_SILICON

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

void
GenerateVideoStage::resolve_config_()
{
  if (_have_cfg || _hf_dir.empty()) { return; }
  namespace fs = std::filesystem;
  // The whole record, not just the path: `model_type` is what tells the
  // two MiniMax-H3 partitions apart when they share a directory.
  _resolved = resolve_model(session(), _hf_dir);
  _root = _resolved.dir;
  const std::string t1 = (fs::path(_root) / "transformer").string();
  // ---- plugin families first ----------------------------------------
  // Asked BEFORE the two built-ins, mirroring how LoadedLanguageModel
  // consults ModelExecRegistry before its own arch if-chain: a
  // checkpoint an out-of-tree family claims is that family's, and the
  // built-in probes below never see it. On the common graph the
  // registry is empty and this costs one null check.
  //
  // A family's `claims` is required to be sure of itself (see
  // video-model-registry.h) precisely because it is asked first --
  // claiming someone else's checkpoint here would shadow a working
  // built-in path.
  _plugin_family = genai::VideoModelRegistry::get().claim_for(
      session(), _root, _resolved.model_type);
  if (_plugin_family != nullptr) {
    _family = std::string(_plugin_family->tag());
    align_frames_(_plugin_family->align_frames(_root, _frames));
    {
      int gh = 0, gw = 0;
      _plugin_family->size_grid(_root, &gh, &gw);
      align_size_(gh, gw);
    }
    _two_experts = false;
    _have_cfg    = true;
    // The family is only now known, so this is the first moment a config
    // beat that already arrived can be handed to it.
    apply_model_config_();
    session()->info(fmt(
        "GenerateVideoStage('{}'): plugin family '{}' at {}x{}x{} frames, "
        "{} steps (checkpoint {})", this->id(), _family, _width, _height,
        _frames, _steps, _root));
    return;
  }
  // The DiT's own `_class_name` picks the family. Reading it here rather
  // than taking it from config keeps a graph from having to be rewired
  // to change checkpoints -- the same stage, ports and keys serve both.
  //
  // H3 is asked FIRST, and through its own reader rather than by opening
  // `transformer/config.json`, because that spelling is a diffusers one
  // and this family also ships as a Comfy-Org repack -- one .safetensors
  // under `diffusion_models/` with the config in its `__metadata__`, or a
  // directory checkpoint there once model-quantize has processed it.
  // config_from_json resolves all three and REFUSES a checkpoint that is
  // not H3, which is what makes trying it first safe: a success is the
  // detection and the config at once, and a failure costs one JSON parse.
  // (Reading `transformer/config.json` directly is what left every stage
  // in this graph inert on a repack root.)
  std::string h3err;
  // What the models DB says this reference IS, which the DIRECTORY may
  // not be able to say: two records can share one local_path so that a
  // repo holding both partitions downloads once, and then only the
  // record distinguishes them.
  const std::string want_partition =
      genai::MetalMiniMaxH3Transformer::partition_of_model_type(
          _resolved.model_type);
  if (genai::MetalMiniMaxH3Transformer::config_from_json(_root, _h3_cfg,
                                                         &h3err,
                                                         want_partition)) {
    _family = "minimax-h3";
  }
  if (_family == "minimax-h3") {
    // WHICH partition. The two ship byte-identical DiT configs, so the
    // detection above cannot tell them apart -- and running a Ref2VA
    // checkpoint through the t2va path is the worst failure this stage
    // has: it loads, it runs at full cost, and it generates video
    // conditioned on nothing at all. Refuse instead.
    // WHICH partition. The two ship byte-identical DiT configs, so the
    // detection above cannot tell them apart, and running Ref2VA
    // weights through the t2va path is the worst failure this stage
    // has: it loads, it runs at full cost, and it generates video
    // conditioned on nothing at all. What closes that is the check in
    // initialize() -- a Ref2VA checkpoint with no reference rows wired
    // is refused there, where iport connectivity is known.
    _h3_partition =
        genai::MetalMiniMaxH3Transformer::partition_of(_root, want_partition);
    // Which partition, and on whose authority. Worth a line: when a
    // directory holds both, "probed" and "the models DB" can disagree,
    // and the DB is the one that knows.
    session()->info(fmt(
        "GenerateVideoStage('{}'): MiniMax-H3 partition '{}' (models DB "
        "says model_type '{}', so {})", this->id(),
        _h3_partition.empty() ? "unknown" : _h3_partition,
        _resolved.model_type.empty() ? "-" : _resolved.model_type,
        want_partition.empty() ? "probed the checkpoint"
                               : "taken from the record"));
    // Now that the family is known, round the frame count UP to H3's own
    // rule (see the constructor): the video VAE takes 17-frame clips and
    // keeps 5 latents from each, so only 17n+5 has a latent form.
    align_frames_(genai::minimax_h3::align_num_frames(_frames, 17, 5));
    // ...and the same for the frame SIZE. H3's video VAE is 16x spatial
    // and the DiT patches 2x on top, so the legal grid is 32 -- one
    // factor coarser than the multiple-of-16 a caller naturally reaches
    // for. A width like 1360 IS a multiple of 16 and gives an odd latent
    // (85), which the packer cannot patch and used to refuse deep inside
    // build_packed_sequence, reporting a latent geometry rather than the
    // pixel size that produced it.
    align_size_(16 * _h3_cfg.patch_h, 16 * _h3_cfg.patch_w);
    // _h3_cfg is already filled -- the detection above IS the read.
    _two_experts = false;   // one stack; nothing to switch at
    _have_cfg    = true;
    // The family is only now known, so this is the first moment a config
    // beat that already arrived can be parsed by anything.
    apply_model_config_();
    // NOTE: the idle-unload policy is NOT decided here. It needs the
    // streaming verdict, which only exists after the load -- see
    // resolve_unload_policy_h3_().
    session()->info(fmt(
        "GenerateVideoStage('{}'): MiniMax-H3 (video+audio) at {}x{}x{} "
        "frames, {} steps, shifts {:.1f}/{:.1f}", this->id(), _width, _height,
        _frames, _steps, _h3_params.video_shift, _h3_params.audio_shift));
    return;
  }
  std::string cerr;
  if (!genai::MetalWanTransformer::config_from_json(t1, _cfg, &cerr)) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): {}; inert", this->id(), cerr));
    return;
  }
  align_frames_(genai::MetalWanVae::align_num_frames(_frames));
  // Wan's VAE is 8x spatial against H3's 16, so its grid is 16.
  align_size_(8 * _cfg.patch_h, 8 * _cfg.patch_w);
  _two_experts = fs::exists(fs::path(_root) / "transformer_2" / "config.json");
  _have_cfg = true;
  apply_model_config_();   // as above: the family is only now known
  if (!_unload_resolved) {
    _unload_resolved = true;
    const std::vector<std::string> peers = {
        (fs::path(_root) / "text_encoder").string(),
        (fs::path(_root) / "vae").string()};
    switch (_unload_cfg) {
      case model_memory::UnloadPolicy::kAlways: _unload_idle = true;  break;
      case model_memory::UnloadPolicy::kNever:  _unload_idle = false; break;
      default:
        _unload_idle = model_memory::bounded(session(), peers,
                                             model_memory::kHeadroom);
        break;
    }
  }
  session()->info(fmt(
      "GenerateVideoStage('{}'): Wan {} at {}x{}x{} frames, {} steps, "
      "in_channels {}{}", this->id(),
      _two_experts ? "A14B (two experts)" : "single-expert",
      _width, _height, _frames, _steps, _cfg.in_channels,
      _two_experts ? fmt(", boundary {:.2f}", _wan_params.boundary_ratio)()
                   : std::string()));
}

// Both directions of one question: what did the caller ask for, and what
// does the checkpoint say. It runs on every change to either because a
// config beat and a model reference arrive on different ports and either
// can be first -- parsing at only one of those moments loses whichever
// came earlier.
void
GenerateVideoStage::apply_model_config_()
{
  if (!_have_cfg) { return; }   // the family is not known yet
  // Report a config wired to the wrong family rather than applying it.
  // The two share no keys, so nothing would be read and the graph would
  // run on defaults while showing a config stage that says otherwise.
  const std::string want = model_config::family_of(_model_cfg);
  if (!want.empty() && want != _family) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): the model_config beat is for the '{}' "
        "family but the resident checkpoint is '{}'; IGNORING it and using "
        "the defaults. Wire the config stage that matches the checkpoint",
        this->id(), want, _family));
    return;
  }
  // A plugin family parses its own beat, at generate time, from the
  // request -- which is the whole point of passing it down UNREAD: a
  // knob the family adds later needs no change here. So there is
  // nothing to do but keep `_model_cfg`, which the caller already did.
  // The family match above still ran, so a config wired to the wrong
  // checkpoint is still reported rather than handed over.
  if (_plugin_family != nullptr) { return; }
  std::string perr;
  if (_family == "minimax-h3") {
    _h3_params =
        genai::MetalMiniMaxH3Transformer::GenerationParams::from_flex(
            _model_cfg, &perr);
    // The LoRA is read here but consumed at LOAD, so a beat that names
    // one after the DiT is already built cannot take effect -- say so
    // rather than run the un-adapted model while the graph claims
    // otherwise. Wiring the port defers the load, so this only bites a
    // trigger-driven config that CHANGES the adapter mid-run.
    //
    // `_have_cfg` says the FAMILY is known, not that a beat has arrived:
    // this runs the moment the checkpoint is identified, which is before
    // the first config beat is read. So `_model_cfg` is routinely a
    // default FlexData here, and as_object() on one THROWS -- which is
    // what took the whole stage down rather than leaving it inert.
    // family_of() and from_flex() above already tolerate it; this has to
    // as well.
    // Both adapter slots, read the same way. Slot 0 is the plain `lora`
    // keys and slot 1 the `lora2` ones, so a graph that names one
    // adapter reads exactly as it always did and a second is additive.
    // Which is which is the CALLER's order, not a role the model knows:
    // the usual pairing is a few-step Turbo distillation and a style or
    // identity adapter, but nothing here depends on that.
    static const char* const kKeys[kH3LoraSlots][3] = {
      {"lora",  "lora_scale",  "lora_qkv_layout"},
      {"lora2", "lora2_scale", "lora2_qkv_layout"},
    };
    _h3_lora.resize(kH3LoraSlots);
    for (int i = 0; i < kH3LoraSlots; ++i) {
      H3LoraSlot want = _h3_lora[(std::size_t)i];   // nothing said: keep
      if (_model_cfg.is_object()) {
        const auto o = _model_cfg.as_object();
        // An object beat WITHOUT the key turns that slot OFF: the config
        // source emits the keys only when set, so their absence is the
        // graph saying it wants no adapter there.
        want.path = o.contains(kKeys[i][0])
            ? std::string(o.at(kKeys[i][0]).as_string("")) : "";
        want.scale = o.contains(kKeys[i][1])
            ? o.at(kKeys[i][1]).as_real(1.0) : 1.0;
        if (o.contains(kKeys[i][2])) {
          want.qkv = std::string(o.at(kKeys[i][2]).as_string(""));
        }
      }
      H3LoraSlot& have = _h3_lora[(std::size_t)i];
      // The two halves of "which adapter, how strong" have different
      // lifetimes and this is where that shows. The PATH is load-time --
      // an adapted mlp.fc1 rules out the fused-SwiGLU kernel, which is
      // chosen before the blocks load -- so changing it under a built
      // DiT is refused rather than half-applied. The STRENGTH is not: it
      // rides the accumulating GEMM as a per-forward constant, so a
      // trigger-driven config can sweep it across beats for the cost of
      // a setter -- PER SLOT, which is the point of having two: the
      // distillation stays where it was trained while the style adapter
      // is dialled.
      if (_h3_dit != nullptr && want.path != have.path) {
        session()->warn(fmt(
            "GenerateVideoStage('{}'): the DiT is already built, so the "
            "model_config beat's `{}` is IGNORED -- it is a load-time "
            "argument. `{}` still applies",
            this->id(), kKeys[i][0], kKeys[i][1]));
      } else {
        have.path = want.path;
        have.qkv  = want.qkv;
      }
      if (want.scale != have.scale) {
        have.scale = want.scale;
        if (_h3_dit != nullptr && _h3_dit->lora_modules(i) > 0) {
          _h3_dit->set_lora_scale(i, (float)want.scale);
          session()->log_debug(fmt(
              "GenerateVideoStage('{}'): runtime LoRA slot {} strength "
              "-> {:.3f}", this->id(), i, want.scale));
        }
      }
    }
  } else {
    _wan_params =
        genai::MetalWanTransformer::GenerationParams::from_flex(_model_cfg,
                                                                &perr);
    // The checkpoint's own boundary, UNLESS the graph asked for one. A
    // stock A14B states it in model_index.json and every graph that never
    // mentions the key must keep getting it; a graph that does mention it
    // means it.
    if (!_wan_params.boundary_ratio_set) {
      _wan_params.boundary_ratio =
          genai::MetalWanTransformer::boundary_ratio_from_index(
              _root, _wan_params.boundary_ratio);
    }
    // One expert has nothing to switch to, whatever anyone asked for.
    if (!_two_experts) { _wan_params.boundary_ratio = 0.0; }
  }
  if (!perr.empty() && !_model_cfg.is_null()) {
    session()->warn(fmt("GenerateVideoStage('{}'): model_config: {}",
                        this->id(), perr));
  }
}

// Assemble the request from the beats process() already read and hand it
// to the family. Everything here is BORROWED -- the beats outlive the
// call -- which is what keeps a 22B generation from copying its own
// conditioning.
bool
GenerateVideoStage::run_plugin_family_(RuntimeContext& ctx,
                                       const void* cond, int cond_rows,
                                       int cond_dim,
                                       const FlexData* cond_sideband,
                                       const void* neg, int neg_rows,
                                       const TensorBeatPayload* ref,
                                       const TensorBeatPayload* ref_last,
                                       const TensorBeatPayload* ref_video_rows,
                                       const TensorBeatPayload* ref_audio_rows,
                                       const TensorBeatPayload* audio_cond,
                                       genai::VideoGenResult* out)
{
  if (!_plugin_gen) { return false; }
  genai::VideoGenRequest req;
  req.height = _height;
  req.width  = _width;
  req.frames = _frames;
  req.fps    = _fps;
  req.steps  = _steps;
  req.seed   = _seed;

  req.cond          = cond;
  req.cond_rows     = cond_rows;
  req.cond_dim      = cond_dim;
  req.cond_sideband = cond_sideband;
  req.neg           = neg;
  req.neg_rows      = neg_rows;
  if (audio_cond != nullptr && audio_cond->shape.size() == 2) {
    req.audio_cond      = audio_cond->data.data();
    req.audio_cond_rows = (int)audio_cond->shape[0];
    req.audio_cond_dim  = (int)audio_cond->shape[1];
  }

  // Shapes go across as the encoder wrote them. Normalising here would
  // mean this stage deciding what a family's conditioning latent looks
  // like, which is exactly the knowledge that does not belong to it.
  auto take = [](const TensorBeatPayload* t, const float** p,
                 std::vector<int>* shape) {
    if (t == nullptr) { return; }
    *p = t->as_f32();
    shape->assign(t->shape.begin(), t->shape.end());
  };
  take(ref, &req.ref, &req.ref_shape);
  take(ref_last, &req.ref_last, &req.ref_last_shape);
  if (ref_video_rows != nullptr && ref_video_rows->shape.size() == 2) {
    req.ref_video_rows   = ref_video_rows->as_f32();
    req.n_ref_video_rows = (int)ref_video_rows->shape[0];
    req.ref_video_dim    = (int)ref_video_rows->shape[1];
  }
  if (ref_audio_rows != nullptr && ref_audio_rows->shape.size() == 2) {
    req.ref_audio_rows   = ref_audio_rows->as_f32();
    req.n_ref_audio_rows = (int)ref_audio_rows->shape[0];
    req.ref_audio_dim    = (int)ref_audio_rows->shape[1];
  }
  // UNREAD, on purpose: see the note in apply_model_config_.
  req.model_config = _model_cfg.is_object() ? &_model_cfg : nullptr;

  // THE SAME BAR THE BUILT-IN FAMILIES GET. This branch used to wire
  // `progress` as a bare stop check and open no bar at all, so a plugin
  // family reported NOTHING for the whole generation -- and a 22B DiT
  // streaming its weights on a bounded box is minutes per step, which is
  // indistinguishable from a hang. It is also the one thing that makes a
  // slow run diagnosable, so the absence cost more than the report.
  //
  // Block-granular where the family offers it, exactly as MiniMax-H3 is
  // driven below: a step-granular bar on a stack this size sits still for
  // the entire time anything is happening.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, _steps, /*forwards_per_step=*/1);
  auto block_fn = prog.block_fn();
  // No end_forward() here: the host cannot see where a plugin's forward
  // returns (the callback fires on block ENTRY, so `done` never reaches
  // `total`). end_step re-syncs the bar to the exact boundary anyway, so
  // the cost is the bar trailing by one block within a step.
  req.block_progress = [&block_fn, &ctx](int done, int total) {
    block_fn(done, total);
    return !ctx.stop_requested();
  };
  // A 22B DiT spends minutes per generation, so a Stop that only lands
  // between generations is a Stop that appears to hang. The family calls
  // this between steps; one that never does simply cannot be
  // interrupted, and that is its own choice to answer for.
  req.progress = [&prog, &ctx](int step, int total) {
    // `total` is the count the family's SCHEDULER settled on, not the
    // configured `steps`; adopting it is what makes the bar finish at
    // 100%. `step` is 1-based on entry, end_step takes the 0-based index
    // it just finished.
    prog.set_steps(total);
    prog.end_step(step - 1);
    return !ctx.stop_requested();
  };

  try {
    const bool ok = _plugin_gen->generate(req, out);
    bar.finish();
    return ok;
  } catch (const std::exception& e) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): family '{}' threw during generation: {}",
        this->id(), _family, e.what()));
    return false;
  } catch (...) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): family '{}' threw a non-standard "
        "exception during generation", this->id(), _family));
    return false;
  }
}

bool
GenerateVideoStage::ensure_expert_(int which)
{
  // A plugin family owns its own residency, so there is no expert to
  // switch: build the generator once and keep it. The streaming verdict
  // is computed here rather than inside the family so every video model
  // in the process answers the memory question the same way -- the
  // family may still overrule it, but it starts from the graph's view.
  if (_plugin_family != nullptr) {
    resolve_config_();
    if (!_have_cfg) { return false; }
    if (_plugin_gen) { return true; }
    auto* pmc = session()->services()->metal_compute();
    if (pmc == nullptr) { return false; }
    genai::VideoModelCreateArgs args;
    args.root       = _root;
    args.model_type = _resolved.model_type;
    args.metal      = pmc;
    args.session    = session();
    args.prefer_streaming =
        !model_memory::bounded(session(), {_root},
                               model_memory::kStreamHeadroom);
    // The config beat, for the keys that decide WHAT to load. Available
    // here because iport9 is read before iport0 and this load is driven
    // by iport0's conditioning -- so a config source that emitted once at
    // launch has already landed. Null when nothing was wired.
    args.model_config = _model_cfg.is_object() ? &_model_cfg : nullptr;
    // The clip this graph intends to make, through the family's own
    // rounding, so a load-time decision that scales with the beat has the
    // right order of magnitude instead of a constant. Left at 0 when the
    // family states no rule -- which a family must read as "unknown",
    // not as "empty".
    {
      int gw = _width, gh = _height, gf = _frames;
      if (planned_geometry_(_root, &gw, &gh, &gf)) {
        args.width = gw; args.height = gh; args.frames = gf;
      }
    }
    try {
      _plugin_gen = _plugin_family->load(args);
    } catch (const std::exception& e) {
      session()->error(fmt(
          "GenerateVideoStage('{}'): family '{}' threw loading '{}': {}; "
          "inert", this->id(), _family, _root, e.what()));
      return false;
    }
    if (!_plugin_gen) {
      session()->error(fmt(
          "GenerateVideoStage('{}'): family '{}' could not load '{}'; inert",
          this->id(), _family, _root));
      return false;
    }
    session()->info(fmt(
        "GenerateVideoStage('{}'): '{}' loaded -- latent {} channels at 1/{} "
        "spatial{}", this->id(), _family, _plugin_gen->latent_channels(),
        _plugin_gen->spatial_compression(),
        args.prefer_streaming ? ", box is tight (streaming advised)" : ""));
    // CORRECT THE PLAN, now that the family has decided what it keeps.
    //
    // The plan-time holding was the checkpoint's size on disk, which is
    // what a stage can know from configuration; a family that streamed
    // its blocks now holds a small fraction of it. The family cannot
    // revise this itself -- revise_memory is a Stage method and a family
    // is not a stage -- so the correction goes through the number it
    // already owes the host: resident_bytes(), which docs/MODEL-MEMORY.md
    // requires of any family whose weights a WeightSet cannot see.
    //
    // Both columns, because a streaming family has no larger form left
    // to grow back into within this launch. Skipped at 0, which is the
    // documented "cannot answer cheaply" -- overwriting a real plan-time
    // figure with a decline would report the checkpoint as free.
    // ONE holding only. resident_bytes() is the generator's TOTAL, so a
    // family that declared several cannot have it apportioned between
    // them -- and splitting it by a guess would replace figures that
    // were at least individually honest. Left alone in that case, which
    // means the correction is available to any family that wants it by
    // declaring the checkpoint it streams as one entry.
    const std::uint64_t held = _plugin_gen->resident_bytes();
    StageMemory m = declare_memory();
    if (held > 0 && m.holdings.size() == 1 &&
        m.holdings[0].preload != (std::size_t)held) {
      m.holdings[0].preload = (std::size_t)held;
      m.holdings[0].floor   = (std::size_t)held;
      revise_memory(m);
      session()->log_debug(fmt(
          "GenerateVideoStage('{}'): '{}' holds {} MB after loading; the "
          "plan is corrected from what the checkpoint weighs",
          this->id(), _family, (std::size_t)held >> 20));
    }
    return true;
  }
  if (_family == "minimax-h3") {
    resolve_config_();
    if (!_have_cfg) { return false; }
    if (_h3_dit) { return true; }
    auto* h3mc = session()->services()->metal_compute();
    if (h3mc == nullptr) { return false; }
    // The 33B DiT is the largest here: ~33 GB at w8, which does NOT fit
    // resident beside the 10 GB video VAE on a 64 GB box -- and it is the
    // SEQUENCE that decides, since the model's own default is 124 frames
    // (37 latent frames, ~19k rows at 960x544). Preloading it works only
    // for the short, low-resolution sequences that are outside the
    // model's supported range anyway, so this asks the same question the
    // other DiT families ask rather than assuming the stack fits.
    namespace fs = std::filesystem;
    const std::string dit_dir =
        genai::MetalMiniMaxH3Transformer::resolve_dit_dir(_root,
                                                          _h3_partition);
    // Through the encoder's own resolver, not by spelling a sibling of
    // the DiT: on a Comfy-Org repack the DiT's parent IS the root and the
    // encoder lives under `text_encoders/`, so building the path by hand
    // produced a directory that does not exist and silently dropped the
    // encoder out of the streaming decision.
    std::string enc_dir =
        genai::MiniMaxH3TextEncoder::resolve_encoder_dir(_root);
    if (enc_dir == _root || !fs::exists(enc_dir)) { enc_dir.clear(); }
    // What the AdaLN bake will retire, before anything loads.
    //
    // The bake runs moments after this and drops every adaln_proj -- 39%
    // of a bf16 checkpoint. Those weights are read once and never
    // coexist with anything, so counting them here decides an
    // IRREVERSIBLE question against a model that is about to shed them.
    //
    // Subtracted only when the bake is CERTAIN: it refuses a schedule
    // whose tables would blow its budget, and a refusal leaves the
    // projections resident on a model that has already declined to
    // stream -- the one direction that thrashes. `kTimestepsUpperBound`
    // is the same worst case the scratch sizing below uses, so the row
    // count fed to the check is an upper bound rather than a guess.
    std::size_t dit_retires = 0;
    {
      constexpr int kRowsPerStep = 4;      // == kTimestepsUpperBound below
      const int max_rows = _steps > 0 ? _steps * kRowsPerStep : 0;
      if (genai::MetalMiniMaxH3Transformer::adaln_bake_certain(_h3_cfg,
                                                              max_rows)) {
        dit_retires =
            genai::MetalMiniMaxH3Transformer::adaln_retired_bytes(dit_dir);
      }
    }
    const auto plan = model_memory::plan_streaming(
        session(), dit_dir, enc_dir, model_memory::kStreamHeadroom,
        dit_retires);
    bool stream_blocks = plan.stream;
    if (const char* e = std::getenv("VPIPE_H3_STREAM")) {
      stream_blocks = (std::atoi(e) != 0);
    }
    // The pinned prefix is retired -- see
    // MetalMiniMaxH3Transformer::load. BlockResidency supersedes it:
    // admission against the live budget instead of a fraction of total
    // RAM computed before the run.
    // At info, not debug: this is the irreversible decision, and when a
    // run turns out slow it is the first number anyone needs.
    session()->info(fmt(
        "GenerateVideoStage('{}'): MiniMax-H3 footprint {} GB (others {} GB, "
        "less {} GB the AdaLN bake retires) + {} GB headroom -> {}",
        this->id(), plan.footprint >> 30, plan.others >> 30,
        plan.retires >> 30, model_memory::kStreamHeadroom >> 30,
        stream_blocks ? "STREAM blocks" : "PRELOAD"));
    _h3_cfg.i8_gemm = _i8_gemm;
    // The runtime LoRA, when the model_config beat named one. It is a
    // LOAD-time argument and not a per-step knob: an adapted mlp.fc1
    // rules out the fused-SwiGLU kernel, which is decided before the
    // blocks are built. Wiring the config port therefore defers this
    // load to the first beat, exactly as generate-image does for
    // FLUX.2's klein_kv.
    //
    // Independent of the int8 route above: that one picks how the BASE
    // projection multiplies and applies on the matrix-core path only,
    // while the adapter is a bf16 side GEMM either way.
    // The config names a MODEL (a registry key), a directory or a file;
    // this is the one place that turns any of those into the single
    // .safetensors the loader opens. A key beats a directory scan
    // because both Turbo checkpoints of a repo land side by side, and
    // only the record says which one the key meant.
    // One LoraSpec per named slot, in the order the graph named them --
    // which is the order the DiT binds and the order `lora_scale(i)`
    // addresses afterwards. A slot whose file cannot be resolved is
    // DROPPED rather than left as a hole, so the surviving adapters keep
    // consecutive indices and a strength beat still reaches them.
    std::vector<genai::MetalMiniMaxH3Transformer::LoraSpec> loras;
    using QL = genai::MetalMiniMaxH3Transformer::LoraSpec::QkvLayout;
    _h3_lora.resize(kH3LoraSlots);
    for (int i = 0; i < kH3LoraSlots; ++i) {
      const H3LoraSlot& sl = _h3_lora[(std::size_t)i];
      if (sl.path.empty()) { continue; }
      std::string lerr;
      genai::MetalMiniMaxH3Transformer::LoraSpec spec;
      spec.path = resolve_adapter_file(session(), sl.path, &lerr);
      if (spec.path.empty()) {
        session()->warn(fmt(
            "GenerateVideoStage('{}'): {} -- generating WITHOUT that "
            "adapter", this->id(), lerr));
        continue;
      }
      spec.scale = (float)sl.scale;
      // LOAD-time like the path: the rows are permuted once, at bind.
      if (sl.qkv == "per_head") { spec.qkv_layout = QL::kPerHead; }
      else if (sl.qkv == "flat") { spec.qkv_layout = QL::kFlat; }
      else if (!sl.qkv.empty() && sl.qkv != "auto") {
        session()->warn(fmt(
            "GenerateVideoStage('{}'): lora_qkv_layout '{}' is not one of "
            "auto / flat / per_head; using auto", this->id(), sl.qkv));
      }
      loras.push_back(std::move(spec));
    }
    // NOTE for the pinned prefix: stream_pin_count now measures the
    // trunk but still assumes 1 GB of activation scratch, and this model
    // holds ~4 GB at its own sizes. It is not passed here because the
    // figure needs `seq`, which comes out of the packing and does not
    // exist until a beat -- and H3 already covers that where it can be
    // known: preflight_h3_scratch_ computes the real number at the first
    // beat and parks, or refuses, if the prefix turned out too generous.
    _h3_dit = genai::MetalMiniMaxH3Transformer::load(
        genai::open_weight_set(dit_dir, session()), h3mc, _h3_cfg,
        stream_blocks, loras);
    resolve_unload_policy_h3_(stream_blocks);
    if (_h3_dit && stream_blocks) {
      // A streamed DiT holds ~one block, not the checkpoint, so the
      // load-time claim would keep sizing every peer against 33 GB. What
      // the SET holds is the right number: the retained top-level
      // tensors are cached there, the streamed blocks deliberately are
      // not.
      auto* mgr = session()->services()->generative_model_manager();
      auto ws = mgr != nullptr ? mgr->weight_set(dit_dir) : nullptr;
      if (ws) {
        const std::size_t held = ws->stats().bytes;
        mgr->revise_declaration(dit_dir, held);
        // The same correction on the topological plan. What the plan
        // could only bound -- the checkpoint's size on disk -- the model
        // now knows exactly, because it has finished deciding what to
        // keep. Both floors are set to what is actually held: a
        // streaming DiT has no larger form left to grow back into
        // within this launch.
        {
          StageMemory m = declare_memory();
          for (StageHolding& h : m.holdings) {
            if (h.source != dit_dir) { continue; }
            h.preload = held;
            h.floor = held;
          }
          revise_memory(m);
        }
        session()->log_debug(fmt(
            "GenerateVideoStage('{}'): streaming DiT keeps {} MB resident, "
            "revised down from {} MB on disk", this->id(), held >> 20,
            model_memory::dir_weights_bytes(dit_dir) >> 20));
      }
    }
    // Say how many blocks are PINNED, the way the conditioner says it for
    // WHICH OF THE TWO IT ACTUALLY DID.
    //
    // This said "streams its N blocks" unconditionally, including on a
    // preload run that had just been told PRELOAD three lines earlier --
    // and the residency probe printed by the AdaLN bake reads as a
    // streaming figure whatever the model does, so a preloaded run
    // looked like a streaming one that had pinned a fraction of its
    // stack. Two lines that contradict the verdict above them are worse
    // than no line at all.
    if (_h3_dit) {
      session()->info(stream_blocks
          ? fmt("GenerateVideoStage('{}'): MiniMax-H3 DiT streams its {} "
                "blocks; the resident set grows into free RAM as the "
                "denoise runs and is given back when the box needs it",
                this->id(), _h3_cfg.n_layers)
          : fmt("GenerateVideoStage('{}'): MiniMax-H3 DiT is PRELOADED -- "
                "all {} blocks resident, no per-step reads and nothing "
                "for block residency to grow", this->id(),
                _h3_cfg.n_layers));
    }
    if (!_h3_dit) {
      session()->error(fmt(
          "GenerateVideoStage('{}'): MiniMax-H3 DiT load failed from '{}'",
          this->id(), _root));
      return false;
    }
    return true;
  }
  if (_dit && _expert == which) { return true; }
  resolve_config_();
  if (!_have_cfg) { return false; }
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr) { return false; }
  namespace fs = std::filesystem;
  const std::string dir =
      (fs::path(_root) / (which == 0 ? "transformer" : "transformer_2"))
          .string();
  // Drop first, THEN load. At bf16 an expert is ~28 GB, so holding both
  // across the swap is not a transient peak, it is an out-of-memory kill.
  if (_dit) {
    _dit.reset();
    _expert = -1;
  }
  session()->info(fmt(
      "GenerateVideoStage('{}'): loading the {}-noise expert from '{}'",
      this->id(), which == 0 ? "high" : "low", dir));
  std::shared_ptr<genai::WeightSet> ws = genai::open_weight_set(dir, session());
  if (!ws) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): no readable checkpoint under '{}'",
        this->id(), dir));
    return false;
  }
  _dit = genai::MetalWanTransformer::load(ws, mc, _cfg);
  if (!_dit) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): failed to load the DiT from '{}'",
        this->id(), dir));
    return false;
  }
  _expert = which;
  return true;
}

Job
GenerateVideoStage::initialize(RuntimeContext& ctx)
{
  // Defer everything when a model-select source feeds the model iport (its
  // beat only arrives after the init barrier). The expert itself is loaded
  // lazily at the first denoise step regardless -- it is 28 GB, and a graph
  // that never receives a prompt should not pay for it.
  if (!(ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort))) {
    resolve_config_();
  }
  // A Ref2VA checkpoint with no reference rows wired would load, run at
  // full 33B cost, and generate video conditioned on nothing -- the two
  // partitions ship byte-identical DiT configs, so nothing downstream
  // would notice.
  //
  // WARNED here, ENFORCED in process(). fail_config() cannot be used at
  // this point: the runtime reads config_error() BEFORE launch and a
  // stage that sets it during initialize() is never re-checked, so the
  // refusal would be dead code. This is the first moment iport
  // connectivity is known, which is why the notice lives here and the
  // per-request refusal -- the one that actually stops a wrong
  // generation -- lives where the beat arrives.
  if (_h3_partition == "ref2va" &&
      !(ctx.num_iports() > kRefVideoRowsPort &&
        ctx.iport_connected(kRefVideoRowsPort))) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): '{}' is MiniMax-H3's Ref2VA partition, "
        "which conditions on a list of reference images, clips and "
        "soundtracks, but iport{} (ref_video_rows) is unwired -- every "
        "request will be refused. Wire a `video-ref-encoder` to iport{} "
        "and iport{}, or use the FL2VA checkpoint for text-to-video and "
        "first/last-frame work.",
        this->id(), _root, kRefVideoRowsPort, kRefVideoRowsPort,
        kRefAudioRowsPort));
  }
  co_return;
}

namespace h3 = genai::minimax_h3;

// Should the DiT be dropped after each clip? Decided AFTER the load, and
// that timing is the whole point.
//
// `auto` asks model_memory::bounded(peers), i.e. "does RAM cover the
// PEERS' weights plus headroom". Asked before those peers have loaded it
// measures a footprint of ~0 and always answers "roomy" -- and for this
// stage the peers are the text encoder and the video VAE, both of which
// are now loaded lazily by OTHER stages, so at any moment this stage
// could ask, the honest answer is zero. The rule was never wrong; it was
// being handed nothing.
//
// So the streaming verdict decides instead, and it is a better signal
// anyway: plan_streaming has ALREADY compared this checkpoint against the
// box and concluded the stack does not fit resident. A model that has to
// stream its own blocks is not one to keep alive across a decode that
// needs the working set -- which on MiniMax-H3 is exactly what happened:
// the DiT stayed resident and a 251 MB video decode was refused with zero
// headroom and 5.3 GB physically free.
void
GenerateVideoStage::resolve_unload_policy_h3_(bool streamed)
{
  if (_unload_resolved) { return; }
  _unload_resolved = true;
  std::vector<std::string> peers;
  for (const std::string& p :
       {genai::MiniMaxH3TextEncoder::resolve_encoder_dir(_root),
        genai::MetalMiniMaxH3VideoVae::resolve_vae_dir(_root)}) {
    if (!p.empty() && p != _root) { peers.push_back(p); }
  }
  switch (_unload_cfg) {
    case model_memory::UnloadPolicy::kAlways: _unload_idle = true;  break;
    case model_memory::UnloadPolicy::kNever:  _unload_idle = false; break;
    default:
      _unload_idle = streamed ||
                     model_memory::bounded(session(), peers,
                                           model_memory::kHeadroom);
      break;
  }
  if (_unload_idle) {
    session()->info(fmt(
        "GenerateVideoStage('{}'): memory-bounded -- the DiT is dropped "
        "after each clip so the VAE decode has working set, and reloads "
        "on the next prompt{}", this->id(),
        streamed ? " (its blocks stream, so the box is already tight)" : ""));
  }
}

// Is there room for this forward's SCRATCH, and if not, can we make it?
//
// The image stages never needed this: a 1024x1024 DiT's activations are
// tens of MB. A video forward's are not -- MiniMax-H3's scratch is ~200 KB
// per row, so the 9382-row production layout wants ~1.9 GB and a 19k-row
// one ~3.9 GB, none of which the weight accounting can see (it is
// allocated per forward, not per checkpoint).
//
// Getting this wrong on a 16 GB box is not a failed allocation. Two
// panic logs on this box record the end state -- "no checkins from
// watchdogd in 94 seconds", free 14 MB, file cache ~0, 4-5 GB wired --
// where the VM has nothing left to reclaim, userspace stops being
// scheduled and the kernel watchdog fires. So refusing loudly is the
// SAFE outcome here, and proceeding hopefully is the dangerous one.
//
// The scratch is what makes that reachable, and NOT because a Metal
// SharedBuffer is unpageable. It is not: MEASURED on the M4 Pro, a 5.9
// GB resident set of them read 100% in-core when written and 0% one
// forward later, every sampled page carrying MINCORE_PAGED_OUT, with
// 9.9 GB of this process compressed. SharedBuffer::set_wired() is what
// would pin them, and the block-residency path is its only caller.
bool
GenerateVideoStage::preflight_h3_scratch_(int seq, int text_rows)
{
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr || seq <= 0) { return true; }
  auto mb = mc->memory_budget();
  if (mb.recommended == 0) { return true; }   // budget query unavailable

  // The distinct-timestep count is decided inside denoise(), but its
  // buffers are sub-MB at any plausible value (mod is n_t x adaln_out),
  // so an upper bound costs nothing and keeps this callable before the
  // schedule exists.
  constexpr int kTimestepsUpperBound = 4;
  const bool dq = _h3_dit && _h3_dit->uses_matrix_cores();
  const std::size_t need = genai::MetalMiniMaxH3Transformer::scratch_bytes(
      _h3_cfg, seq, text_rows, kTimestepsUpperBound, dq);

  // Tell the DiT how much room to leave clear when it decides whether to
  // keep a streamed block resident. Its growth must not eat the scratch
  // the very next forward needs -- that would trade a disk read for an
  // allocation failure. A further margin on top keeps room for the peers
  // that run after the denoise (the VAE decode above all).
  if (_h3_dit) {
    _h3_dit->set_residency_reserve(need + (1ull << 30));
  }

  // Both budgets, for the reason generate-image gives: fits() is our Metal
  // working set and misses other processes' resident memory; fits_physical
  // is host-wide reclaimable RAM and catches it.
  if (mb.fits(need) && mb.fits_physical(need)) { return true; }

  // Make room. Parking hands weight pages to the kernel as purgeable --
  // reversible, and free unless something actually takes them.
  std::size_t parked = 0;
  if (auto* gm = session()->services()->generative_model_manager()) {
    parked = gm->reclaim_at_least(need);
  }
  mb = mc->memory_budget();
  if (mb.fits(need) && mb.fits_physical(need)) {
    session()->info(fmt(
        "GenerateVideoStage('{}'): parked ~{} MB to fit the {}-row forward's "
        "~{} MB of scratch", this->id(), parked >> 20, seq, need >> 20));
    return true;
  }
  session()->error(fmt(
      "GenerateVideoStage('{}'): not enough memory for a {}-row forward -- "
      "it needs ~{} MB of scratch and there is ~{} MB of GPU working set / "
      "~{} MB reclaimable{}. Refusing rather than thrashing: wired Metal "
      "buffers cannot be paged out, so overcommitting here takes the whole "
      "machine down rather than failing this stage. Use a smaller "
      "height/width/frames, or free another model first",
      this->id(), seq, need >> 20, mb.headroom >> 20,
      mb.available_physical >> 20,
      parked > 0 ? fmt(" after parking ~{} MB", parked >> 20)()
                 : std::string()));
  return false;
}

// Read a `ref2va` plan off the conditioning beat's sideband.
//
// The plan and the rows arrive on different beats and this is where
// they are made to agree: the sideband says how many latent frames and
// cells each reference encoded to, the row beats say how many rows were
// packed, and the two are derived from the same encode. A disagreement
// means the beats are from different requests, which is worth naming
// here rather than discovering as a shape error 50 layers down.
bool
GenerateVideoStage::parse_h3_references_(const FlexData& sideband,
                                         const TensorBeatPayload* video_rows,
                                         const TensorBeatPayload* audio_rows,
                                         H3References* out) const
{
  if (!sideband.is_object()) { return true; }
  // as_object()/as_array() return VIEWS into their owner, so every value
  // taken out of one is bound to a local before it is read.
  const auto so = sideband.as_object();
  if (!so.contains("references")) { return true; }
  const FlexData refs = so.at("references");
  if (!refs.is_array()) { return true; }
  const auto ra = refs.as_array();
  if (ra.size() == 0) { return true; }

  out->refs.clear();
  out->refs.reserve(ra.size());
  for (std::size_t i = 0; i < ra.size(); ++i) {
    const FlexData e = ra[i];
    if (!e.is_object()) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): reference {} of the conditioning plan "
          "is not an object; skipping", this->id(), i + 1));
      return false;
    }
    const auto eo = e.as_object();
    auto num = [&](const char* k) -> int {
      return eo.contains(k) ? (int)eo.at(k).as_int() : 0;
    };
    h3::Reference r;
    const FlexData kd = eo.contains("kind") ? eo.at("kind") : FlexData{};
    const std::string kind = kd.is_string() ? std::string(kd.as_string()) : "";
    r.kind = kind == "video"   ? h3::Reference::Kind::kVideo
             : kind == "audio" ? h3::Reference::Kind::kAudio
                               : h3::Reference::Kind::kImage;
    r.num_latent_frames = num("latent_frames");
    r.latent_height     = num("latent_height");
    r.latent_width      = num("latent_width");
    r.num_audio_latents = num("audio_latents");
    out->refs.push_back(r);
  }

  if (so.contains("token_tags")) {
    const FlexData t = so.at("token_tags");
    if (t.is_array()) {
      const auto ta = t.as_array();
      out->text_tags.reserve(ta.size());
      for (std::size_t i = 0; i < ta.size(); ++i) {
        out->text_tags.push_back((int)ta[i].as_int());
      }
    }
  }

  auto rows_of = [](const TensorBeatPayload* t, int want_elems,
                    const float** data) -> int {
    *data = nullptr;
    if (t == nullptr || t->shape.size() != 2) { return 0; }
    if ((int)t->shape[1] != want_elems) { return -1; }
    if (t->shape[0] > 0) { *data = t->as_f32(); }
    return (int)t->shape[0];
  };
  const int PE = _h3_cfg.video_patch_elems();
  const int AC = _h3_cfg.audio_channels;
  const int vr = rows_of(video_rows, PE, &out->video_rows);
  const int ar = rows_of(audio_rows, AC, &out->audio_rows);
  if (vr < 0 || ar < 0) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): the reference rows are not [n, {}] / "
        "[n, {}]; skipping", this->id(), PE, AC));
    return false;
  }
  out->n_video_rows = vr;
  out->n_audio_rows = ar;
  return true;
}

// The minimax-h3 denoise: one packed sequence carrying both modalities,
// so this produces two latents where the Wan path produces one.
bool
GenerateVideoStage::run_h3_(const void* cond, int text_rows, const float* ref,
                            int ref_frames, const H3References* r2v,
                            std::vector<float>* video_out,
                            std::vector<int>* video_shape,
                            std::vector<float>* audio_out,
                            std::vector<int>* audio_shape)
{
  auto* mc = session()->services()->metal_compute();
  if (mc == nullptr || !_h3_dit) { return false; }
  const auto& c = _h3_cfg;

  // Latent geometry. The video VAE compresses 16x spatially and 4x in
  // time with the causal round-UP the encoder does, so the frame count
  // is not a plain division.
  const int lh = _height / 16, lw = _width / 16;
  // The video VAE encodes in 17-frame clips that yield 5 latent frames
  // each and then drops 3, so a request has to be snapped UP to a length
  // the chunking can actually produce. These two are the released
  // checkpoint's `vae_clip_length` / tokens-per-chunk; a checkpoint that
  // changed them would want them read from video_vae/config.json rather
  // than assumed here.
  constexpr int kFramesPerChunk = 17, kLatentsPerChunk = 5;
  const int aligned =
      h3::align_num_frames(_frames, kFramesPerChunk, kLatentsPerChunk);
  const int lt =
      h3::video_latent_num_frames(aligned, kFramesPerChunk, kLatentsPerChunk);
  if (lh <= 0 || lw <= 0 || lt <= 0) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): {}x{}x{} does not give a usable "
        "MiniMax-H3 latent grid", this->id(), _width, _height, _frames));
    return false;
  }
  // Audio latents are counted from the VIDEO frames and fps, so the two
  // modalities stay the same duration by construction -- unless the
  // config asked for a soundtrack of its own length. The rule is the
  // model's, so the model layer states it.
  const int alat = _h3_params.audio_latents(aligned, _fps);

  h3::PackedLayout L;
  // Text rows are tagged 1 EXCEPT the rows of a vision block, which
  // MiniMax-H3 tags 0 (video). A `t2va` / `fl2va` prompt has no vision
  // block so its tags are uniform; a `ref2va` presentation's are not,
  // and the conditioner is the only thing that knows where the blocks
  // landed -- which is why they travel with the conditioning.
  std::vector<int> tags((std::size_t)text_rows, h3::kTextTag);
  if (r2v != nullptr && (int)r2v->text_tags.size() == text_rows) {
    tags = r2v->text_tags;
  }
  // One latent frame per anchor. No ref => text-to-video-and-audio; one
  // => a first-frame anchor; two => first AND last, which is what the
  // FL2VA partition is named for.
  std::vector<h3::Anchor> anchors;
  if (r2v == nullptr && ref != nullptr && ref_frames > 0) {
    anchors.push_back(h3::Anchor::kFirst);
    if (ref_frames >= 2) { anchors.push_back(h3::Anchor::kLast); }
  }
  // h3::kAudioChannels is the STEREO count (2) -- how many soundtrack
  // channels the audio rows are packed channel-major over. It is NOT
  // `c.audio_channels`, which is the 32-wide audio LATENT that each of
  // those rows carries. Passing the latter here packs 16x the audio rows
  // the model was trained with; every one of them is a real row that the
  // 33B forward pays for, and the result still decodes, just as a
  // soundtrack whose two "channels" are slices of one 32-way split.
  const bool packed =
      r2v != nullptr
          ? h3::build_ref2va_packed_sequence(tags, r2v->refs, lt, lh, lw,
                                             alat, c.patch_h, c.patch_w,
                                             h3::kAudioChannels, &L)
          : h3::build_packed_sequence(tags, lt, lh, lw, alat, c.patch_h,
                                      c.patch_w, h3::kAudioChannels, anchors,
                                      &L);
  if (!packed) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): could not pack a {}x{}x{} latent with {} "
        "audio latents{}", this->id(), lt, lh, lw, alat,
        r2v != nullptr ? " and the request's references" : ""));
    return false;
  }
  // The sequence length is known now and nothing large has been allocated
  // yet, which is the only moment a preflight is worth anything.
  if (!preflight_h3_scratch_(L.seq_len, text_rows)) { return false; }

  const int PE = c.video_patch_elems();
  const int AC = c.audio_channels;
  const int vrows = (int)L.video_indices.size();
  // Both buffers are sized by the INDEX vectors, not by the generated
  // counts: a `ref2va` request leads each modality with reference rows
  // that the denoise head still writes through, and `num_audio_rows` is
  // the generated tail alone.
  const int arows = (int)L.audio_indices.size();
  std::vector<float> vid((std::size_t)vrows * PE);
  std::vector<float> aud((std::size_t)arows * AC);
  {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : vid) { v = nd(rng); }
    for (auto& v : aud) { v = nd(rng); }
  }
  // Diagnostic: take the INITIAL NOISE from files instead of the RNG.
  // A reference implementation draws from torch's generator, not from
  // mt19937_64, so two runs of the same prompt are different SAMPLES and
  // can only be compared by eye. Feeding both the same noise makes the
  // comparison elementwise. Files are raw f32 already in ROW layout:
  // video [num_video_indices, PE], audio [num_audio_rows, AC].
  {
    auto slurp = [&](const char* env, std::vector<float>& dst,
                     const char* what) {
      const char* p = std::getenv(env);
      if (p == nullptr) { return; }
      std::FILE* f = std::fopen(p, "rb");
      if (f == nullptr) {
        session()->warn(fmt("GenerateVideoStage('{}'): cannot open {} noise "
                            "'{}'", this->id(), what, p));
        return;
      }
      const std::size_t n = std::fread(dst.data(), 4, dst.size(), f);
      std::fclose(f);
      if (n != dst.size()) {
        session()->warn(fmt("GenerateVideoStage('{}'): {} noise '{}' has {} "
                            "floats, expected {}", this->id(), what, p, n,
                            dst.size()));
        return;
      }
      session()->info(fmt("GenerateVideoStage('{}'): loaded {} initial noise "
                          "({} floats) from {}", this->id(), what, n, p));
    };
    slurp("VPIPE_H3_NOISE_VID", vid, "video");
    slurp("VPIPE_H3_NOISE_AUD", aud, "audio");
  }

  // The `ref2va` references are ALREADY packed into rows -- the encoder
  // did it, where each reference's own latent geometry was known -- so
  // they are copied straight into the leading conditioning rows of both
  // buffers. The denoise loop never writes those rows, so what lands
  // here is what the model reads at every step.
  //
  // The counts are checked rather than trusted: the canvas the layout
  // was built from is this stage's config and the one the references
  // were encoded at is the encoder's, so a graph whose two `frames`
  // disagree lands here with a real mismatch. Left alone it would
  // surface as a shape error 50 layers down.
  if (r2v != nullptr) {
    if (r2v->n_video_rows != L.num_condition_video_rows ||
        r2v->n_audio_rows != L.num_condition_audio_rows) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): the layout reserves {} reference video "
          "and {} reference audio rows but the encoder packed {} and {}. The "
          "video-ref-encoder's `frames` and this stage's do not agree; "
          "skipping", this->id(), L.num_condition_video_rows,
          L.num_condition_audio_rows, r2v->n_video_rows, r2v->n_audio_rows));
      return false;
    }
    if (r2v->video_rows != nullptr && r2v->n_video_rows > 0) {
      std::memcpy(vid.data(), r2v->video_rows,
                  (std::size_t)r2v->n_video_rows * PE * sizeof(float));
    }
    if (r2v->audio_rows != nullptr && r2v->n_audio_rows > 0) {
      std::memcpy(aud.data(), r2v->audio_rows,
                  (std::size_t)r2v->n_audio_rows * AC * sizeof(float));
    }
    // Diagnostic: blank the reference rows, leaving the layout, the
    // presentation and the noise byte-identical. "Do the reference
    // LATENTS reach the picture" cannot be answered by changing the
    // reference FILE -- that moves the conditioning tokens and the row
    // counts with it -- so the only clean ablation is the values alone.
    // `video` / `audio` / `all`.
    if (const char* ab = std::getenv("VPIPE_H3_REF_ABLATE")) {
      const std::string what(ab);
      const bool all = what == "1" || what == "all";
      if ((all || what == "video") && r2v->n_video_rows > 0) {
        std::fill(vid.begin(),
                  vid.begin() + (std::size_t)r2v->n_video_rows * PE, 0.0f);
      }
      if ((all || what == "audio") && r2v->n_audio_rows > 0) {
        std::fill(aud.begin(),
                  aud.begin() + (std::size_t)r2v->n_audio_rows * AC, 0.0f);
      }
      session()->info(fmt(
          "GenerateVideoStage('{}'): ABLATION -- reference {} rows zeroed",
          this->id(), what));
    }
  }

  // Patchify the anchors into the CONDITIONING rows, which lead the
  // video block one whole latent frame per anchor and in the same cell
  // order the generated frames use. The denoise loop never writes these
  // rows, so what is put here is what the model sees at every step.
  if (!anchors.empty()) {
    const int gh0 = lh / c.patch_h, gw0 = lw / c.patch_w;
    const int rows_per_frame = gh0 * gw0;
    const std::size_t rplane = (std::size_t)lh * lw;
    const int ZCr = c.video_channels;
    for (std::size_t k = 0; k < anchors.size(); ++k) {
      for (int cell = 0; cell < rows_per_frame; ++cell) {
        float* row = vid.data() +
                     ((std::size_t)k * rows_per_frame + cell) * PE;
        const int by = (cell / gw0) * c.patch_h;
        const int bx = (cell % gw0) * c.patch_w;
        for (int ch = 0; ch < ZCr; ++ch) {
          for (int y = 0; y < c.patch_h; ++y) {
            for (int x = 0; x < c.patch_w; ++x) {
              row[((std::size_t)ch * c.patch_h + y) * c.patch_w + x] =
                  ref[((std::size_t)ch * ref_frames + (int)k) * rplane +
                      (std::size_t)(by + y) * lw + (bx + x)];
            }
          }
        }
      }
    }
  }

  // Diagnostic: write the text conditioning the DiT is about to read to
  // a raw f32 file, so the REFERENCE pipeline can be driven with the
  // exact same rows. It is the one DiT input with no golden of its own
  // (the encoder goldens stop at tap 3), and feeding it to the reference
  // separates "our conditioning is wrong" from "everything downstream
  // is wrong" without needing a 32B reference encoder to fit in RAM.
  if (const char* dp = std::getenv("VPIPE_H3_COND_DUMP")) {
    std::FILE* f = std::fopen(dp, "wb");
    if (f != nullptr) {
      const auto* d = reinterpret_cast<const std::uint16_t*>(cond);
      const std::size_t n = (std::size_t)text_rows * c.text_dim;
      std::vector<float> v(n);
      for (std::size_t k = 0; k < n; ++k) {
        const std::uint32_t u = (std::uint32_t)d[k] << 16;
        std::memcpy(&v[k], &u, 4);
      }
      std::fwrite(v.data(), 4, n, f);
      std::fclose(f);
      session()->info(fmt("GenerateVideoStage('{}'): dumped {} x {} text "
                          "conditioning rows to {}", this->id(), text_rows,
                          c.text_dim, dp));
    }
  }
  metal_compute::SharedBuffer tb =
      mc->make_shared_buffer((std::size_t)text_rows * c.text_dim * 2);
  if (tb.empty()) { return false; }
  std::memcpy(tb.contents(), cond,
              (std::size_t)text_rows * c.text_dim * 2);

  genai::DenoiseRequest req;
  req.dit    = _h3_dit.get();
  req.layout = &L;
  req.text   = &tb;
  req.video  = vid.data();
  req.audio  = aud.data();
  req.num_steps   = _steps;
  // The shifts and both condition levels in one move, so a knob added to
  // GenerationParams reaches the loop without this call site changing.
  req.set_params(_h3_params);
  UiProgress bar = session()->open_progress("denoise");
  // Block-granular, like the image DiTs. A step here is one forward of a
  // 33B stack over a ~19k-row sequence -- tens of seconds at the model's
  // own geometry -- so a step-granular bar sits still for the entire time
  // anything is happening. H3 is guidance-distilled, so exactly ONE
  // forward per step.
  DenoiseProgress prog(&bar, _steps, /*forwards_per_step=*/1);
  ScopedBlockProgress<genai::MetalMiniMaxH3Transformer> hook(_h3_dit.get(),
                                                             prog);
  req.progress = [&](int step, int total) {
    // `total` is the count the SCHEDULER settled on, which is not
    // `_steps`: that is a sigma grid including the terminal zero, and the
    // shift can collapse duplicates on top. Adopting it is what makes the
    // bar finish at 100% instead of at (steps-1)/steps.
    prog.set_steps(total);
    // `step` is 1-based on entry here; end_step takes the 0-based index
    // it just finished, and re-syncs the bar to the exact boundary.
    prog.end_step(step - 1);
    return true;
  };
  std::string derr;
  const bool ok = genai::denoise(req, &derr);
  bar.finish();
  if (!ok) {
    session()->warn(fmt("GenerateVideoStage('{}'): {}", this->id(), derr));
    return false;
  }

  // Unpatchify the GENERATED video rows back to a [z, T, lh, lw] grid.
  // Each row is one (1, patch_h, patch_w) cell of z_channels, and the
  // conditioning rows lead the block, so the generated ones start at
  // num_condition_rows.
  const int ZC = c.video_channels;
  const int ph = c.patch_h, pw = c.patch_w;
  const int gh = lh / ph, gw = lw / pw;
  video_out->assign((std::size_t)ZC * lt * lh * lw, 0.0f);
  *video_shape = {ZC, lt, lh, lw};
  const std::size_t plane = (std::size_t)lh * lw;
  for (int r = 0; r < L.num_video_rows; ++r) {
    const float* row = vid.data() +
                       ((std::size_t)L.num_condition_rows + r) * PE;
    const int cell = r % (gh * gw);
    const int t    = r / (gh * gw);
    const int by   = (cell / gw) * ph, bx = (cell % gw) * pw;
    for (int ch = 0; ch < ZC; ++ch) {
      for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
          (*video_out)[(std::size_t)ch * lt * plane + (std::size_t)t * plane +
                       (std::size_t)(by + y) * lw + (bx + x)] =
              row[((std::size_t)ch * ph + y) * pw + x];
        }
      }
    }
  }
  // Diagnostic: the FINAL video latent, [z, T, lh, lw] f32, so it can be
  // compared cell-by-cell against the reference's. Spatial coherence is
  // a property of the LATENT, so this settles "is the tile grid already
  // in the latent" without involving the VAE at all.
  if (const char* lp = std::getenv("VPIPE_H3_LATENT_DUMP")) {
    std::FILE* f = std::fopen(lp, "wb");
    if (f != nullptr) {
      std::fwrite(video_out->data(), 4, video_out->size(), f);
      std::fclose(f);
      session()->info(fmt("GenerateVideoStage('{}'): dumped latent "
                          "[{}, {}, {}, {}] to {}", this->id(), ZC, lt, lh,
                          lw, lp));
    }
  }

  // The audio rows come out of the packed sequence as [stereo * alat, AC]
  // -- channel-major in time, each row a 32-wide latent. Transpose to the
  // [stereo, AC, alat] the audio VAE decodes, so the packed-sequence
  // layout stops at this stage's boundary the same way the video
  // unpatchify above does.
  //
  // The GENERATED rows are the tail: a `ref2va` request leads the buffer
  // with one clean block per reference soundtrack, which is conditioning
  // and not part of what was generated.
  const int acond = L.num_condition_audio_rows;
  audio_out->assign((std::size_t)h3::kAudioChannels * AC * alat, 0.0f);
  for (int ch = 0; ch < h3::kAudioChannels; ++ch) {
    for (int i = 0; i < alat; ++i) {
      const float* row =
          aud.data() + (std::size_t)(acond + ch * alat + i) * AC;
      for (int k = 0; k < AC; ++k) {
        (*audio_out)[((std::size_t)ch * AC + k) * alat + i] = row[k];
      }
    }
  }
  *audio_shape = {h3::kAudioChannels, AC, alat};
  return true;
}

void
GenerateVideoStage::tag_model_(TensorBeat& tb) const
{
  // `_hf_dir` is the reference the USER named (a registry key like
  // "local/MiniMax-H3-FL2VA-8bit", or a path), which is the meaningful
  // identity of the generator -- not the directory it resolved to.
  provenance::set_model_name(tb.sideband, _hf_dir);
}

Job
GenerateVideoStage::process(RuntimeContext& ctx)
{
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      if (apply_model_select_beat(mfd->data, _hf_dir)) { resolve_config_(); }
    }
  }
  if (!_sampler_latched && ctx.num_iports() > kSamplerPort &&
      ctx.iport_connected(kSamplerPort)) {
    auto sb = co_await ctx.read(kSamplerPort);
    _sampler_latched = true;
    if (const auto* sfd =
            sb ? dynamic_cast<const FlexDataPayload*>(sb.get()) : nullptr) {
      std::string serr;
      _sampler_spec = genai::FlowSamplerSpec::from_flex(sfd->data, &serr);
      if (!serr.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): sampler spec: {}",
                            this->id(), serr));
      }
    }
  }
  if (!_scheduler_latched && ctx.num_iports() > kSchedPort &&
      ctx.iport_connected(kSchedPort)) {
    auto cb = co_await ctx.read(kSchedPort);
    _scheduler_latched = true;
    if (const auto* cfd =
            cb ? dynamic_cast<const FlexDataPayload*>(cb.get()) : nullptr) {
      std::string cerr;
      _scheduler_spec = genai::FlowSchedulerSpec::from_flex(cfd->data, &cerr);
      if (!cerr.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): scheduler spec: {}",
                            this->id(), cerr));
      }
    }
  }

  // The model config. Latched like the sampler and scheduler, but
  // RE-READ whenever another beat is waiting: a config source with no
  // trigger emits once for the whole run, while one driven by a trigger
  // emits per request. Blocking on the first beat and polling after
  // serves both -- the first request waits for the parameters it was
  // wired to use, and later ones pick up a change without waiting for
  // one that may never come.
  if (ctx.num_iports() > kModelCfgPort && ctx.iport_connected(kModelCfgPort) &&
      (!_cfg_latched || ctx.backlog(kModelCfgPort) > 0)) {
    auto gb = co_await ctx.read(kModelCfgPort);
    _cfg_latched = true;
    if (const auto* gfd =
            gb ? dynamic_cast<const FlexDataPayload*>(gb.get()) : nullptr) {
      _model_cfg = gfd->data;
      // Only if the family is already known; otherwise resolve_config_
      // applies it the moment it is.
      apply_model_config_();
    }
  }

  auto in = co_await ctx.read(0);
  if (!in) { ctx.signal_done(); co_return; }
  const auto* cond = dynamic_cast<const TensorBeatPayload*>(in.get());
  if (cond == nullptr || cond->shape.size() != 2) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): expected a [text_seq, {}] conditioning "
        "TensorBeat, got {}; skipping", this->id(), _cfg.text_dim,
        in->describe()));
    co_return;
  }
  // The negative is emitted BEFORE the positive by the conditioner, so by
  // the time the positive lands its pair is already queued -- a non-blocking
  // poll rather than a read that could deadlock when there is no negative.
  std::unique_ptr<BeatPayloadIntf> negb;
  if (ctx.num_iports() > 1 && ctx.iport_connected(1) && ctx.backlog(1) > 0) {
    negb = co_await ctx.read(1);
  }
  const auto* neg =
      negb ? dynamic_cast<const TensorBeatPayload*>(negb.get()) : nullptr;
  std::unique_ptr<BeatPayloadIntf> refb;
  if (ctx.num_iports() > kRefPort && ctx.iport_connected(kRefPort)) {
    refb = co_await ctx.read(kRefPort);
  }
  const auto* ref =
      refb ? dynamic_cast<const TensorBeatPayload*>(refb.get()) : nullptr;
  std::unique_ptr<BeatPayloadIntf> refb1;
  if (ctx.num_iports() > kRefPort1 && ctx.iport_connected(kRefPort1)) {
    refb1 = co_await ctx.read(kRefPort1);
  }
  const auto* ref1 =
      refb1 ? dynamic_cast<const TensorBeatPayload*>(refb1.get()) : nullptr;
  // The `ref2va` reference rows. Read unconditionally when wired: a
  // video-ref-encoder emits BOTH every request, with 0 rows when a
  // modality is absent, so a poll would be a race where a read is not.
  std::unique_ptr<BeatPayloadIntf> rvb, rab;
  if (ctx.num_iports() > kRefVideoRowsPort &&
      ctx.iport_connected(kRefVideoRowsPort)) {
    rvb = co_await ctx.read(kRefVideoRowsPort);
  }
  if (ctx.num_iports() > kRefAudioRowsPort &&
      ctx.iport_connected(kRefAudioRowsPort)) {
    rab = co_await ctx.read(kRefAudioRowsPort);
  }
  std::unique_ptr<BeatPayloadIntf> acb;
  if (ctx.num_iports() > kAudioCondPort &&
      ctx.iport_connected(kAudioCondPort)) {
    acb = co_await ctx.read(kAudioCondPort);
  }
  const auto* act =
      acb ? dynamic_cast<const TensorBeatPayload*>(acb.get()) : nullptr;
  const auto* rvt =
      rvb ? dynamic_cast<const TensorBeatPayload*>(rvb.get()) : nullptr;
  const auto* rat =
      rab ? dynamic_cast<const TensorBeatPayload*>(rab.get()) : nullptr;

  if (!ensure_expert_(0)) {
    session()->warn(fmt(
        "GenerateVideoStage('{}'): no DiT; skipping", this->id()));
    co_return;
  }

  // ---- a plugin family: it owns the whole generation ------------------
  if (_plugin_family != nullptr && _plugin_gen) {
    genai::VideoGenResult res;
    if (!run_plugin_family_(ctx, cond->data.data(), (int)cond->shape[0],
                            (int)cond->shape[1], &cond->sideband,
                            neg ? neg->data.data() : nullptr,
                            neg ? (int)neg->shape[0] : 0,
                            ref, ref1, rvt, rat, act, &res)) {
      if (ctx.stop_requested()) {
        session()->info(fmt(
            "GenerateVideoStage('{}'): stopped mid-denoise; dropping the "
            "partial clip", this->id()));
      }
      co_return;                       // the family already said why
    }
    if (res.video.empty() || res.video_shape.size() != 4) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): family '{}' returned success but no "
          "4-D video latent; nothing to publish", this->id(), _family));
      co_return;
    }
    if (_unload_idle) { _plugin_gen->release_idle(); }

    auto vout = std::make_unique<TensorBeatPayload>();
    vout->dtype = TensorBeat::DType::F32;
    vout->shape = {res.video_shape[0], res.video_shape[1],
                   res.video_shape[2], res.video_shape[3]};
    vout->resize_contiguous(res.video.size());
    std::memcpy(vout->as_f32(), res.video.data(),
                res.video.size() * sizeof(float));
    {
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign("fps", FlexData::make_real(_fps));
      sb.as_object().insert_or_assign(
          "frames", FlexData::make_int((std::int64_t)_frames));
      vout->sideband = std::move(sb);
    }
    ++_emitted;
    session()->info(fmt(
        "GenerateVideoStage('{}'): emitted '{}' latents #{} -- video "
        "[{}, {}, {}, {}]{}", this->id(), _family, _emitted,
        res.video_shape[0], res.video_shape[1], res.video_shape[2],
        res.video_shape[3],
        res.audio.empty() ? std::string() : std::string(" + audio")));
    tag_model_(*vout);
    co_await ctx.write(0, std::move(vout));

    // As for the built-in audio family: only when the port is wired. A
    // graph that wants video from an audio-video model should not have
    // to sink a beat it will not read.
    if (ctx.num_oports() > 1 && !res.audio.empty() &&
        res.audio_shape.size() == 3) {
      auto aout = std::make_unique<TensorBeatPayload>();
      aout->dtype = TensorBeat::DType::F32;
      aout->shape = {res.audio_shape[0], res.audio_shape[1],
                     res.audio_shape[2]};
      aout->resize_contiguous(res.audio.size());
      std::memcpy(aout->as_f32(), res.audio.data(),
                  res.audio.size() * sizeof(float));
      if (res.latents_per_second > 0.0) {
        FlexData sb = FlexData::make_object();
        sb.as_object().insert_or_assign(
            "latents_per_second",
            FlexData::make_real(res.latents_per_second));
        aout->sideband = std::move(sb);
      }
      tag_model_(*aout);
      co_await ctx.write(1, std::move(aout));
    }
    co_return;
  }

  // ---- minimax-h3: one packed sequence, two latents out ---------------
  if (_family == "minimax-h3") {
    if ((int)cond->shape[1] != _h3_cfg.text_dim) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): MiniMax-H3 wants [text_seq, {}] "
          "conditioning, got [{}, {}]; skipping", this->id(),
          _h3_cfg.text_dim, cond->shape[0], cond->shape[1]));
      co_return;
    }
    if (neg != nullptr) {
      // Not an error -- a graph wired for Wan should still run -- but
      // silently dropping it would misrepresent what ran.
      session()->log_debug(fmt(
          "GenerateVideoStage('{}'): MiniMax-H3 is guidance-distilled; the "
          "negative conditioning on iport1 is ignored", this->id()));
    }
    std::vector<float> vlat, alat_out;
    std::vector<int>   vshape, ashape;
    // ---- ref2va: the request's plan, off the conditioning sideband ---
    // The geometry travels WITH the conditioning rather than on its own
    // port because the two are one request: pairing a conditioning with
    // another request's layout packs cleanly and then fails 50 layers
    // deep.
    H3References r2v;
    bool have_r2v = false;
    if (!parse_h3_references_(cond->sideband, rvt, rat, &r2v)) {
      co_return;   // already warned
    }
    have_r2v = !r2v.refs.empty();
    if (_h3_partition == "ref2va" && !have_r2v) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): a Ref2VA checkpoint is resident but the "
          "conditioning carries no references; a Ref2VA forward without them "
          "generates video conditioned on nothing. Skipping", this->id()));
      co_return;
    }
    if (have_r2v && _h3_partition == "fl2va") {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): the conditioning carries references but "
          "'{}' is the FL2VA partition, which has no reference blocks; "
          "skipping", this->id(), _root));
      co_return;
    }
    // The keyframe anchor arrives on the SAME port Wan's i2v latent
    // does, from a vae-encode over the keyframe image -- so a graph
    // changes checkpoints without being rewired.
    const int lh0 = _height / 16, lw0 = _width / 16;
    auto anchor_ok = [&](const TensorBeatPayload* t) {
      return t != nullptr && t->shape.size() == 4 &&
             (int)t->shape[0] == _h3_cfg.video_channels &&
             (int)t->shape[2] == lh0 && (int)t->shape[3] == lw0;
    };
    const float* refp = nullptr;
    int ref_frames = 0;
    if (ref != nullptr) {
      if (anchor_ok(ref)) {
        refp = ref->as_f32();
        ref_frames = (int)ref->shape[1];
      } else {
        session()->warn(fmt(
            "GenerateVideoStage('{}'): keyframe latent does not match "
            "[{}, n, {}, {}]; generating without an anchor", this->id(),
            _h3_cfg.video_channels, lh0, lw0));
      }
    }
    // The LAST-frame anchor arrives as its own beat, because the stage
    // that makes one encodes ONE image -- so first-and-last is two
    // vae-encodes, not one that somehow emits two latent frames. They
    // are concatenated here, on the frame axis, into the [z, k, lh, lw]
    // block run_h3_ patchifies (anchor k -> conditioning frame k).
    std::vector<float> stacked;
    if (ref1 != nullptr) {
      if (refp == nullptr) {
        // Dropping it silently would run a plain t2v while the graph
        // says otherwise. L2V is not a partition this model was
        // trained for, so the honest move is to name what was ignored.
        session()->warn(fmt(
            "GenerateVideoStage('{}'): a LAST-frame anchor on iport{} needs "
            "a FIRST-frame anchor on iport{} too; ignoring it", this->id(),
            kRefPort1, kRefPort));
      } else if (!anchor_ok(ref1)) {
        session()->warn(fmt(
            "GenerateVideoStage('{}'): last-frame keyframe latent does not "
            "match [{}, n, {}, {}]; generating with the first frame only",
            this->id(), _h3_cfg.video_channels, lh0, lw0));
      } else {
        const int ZC = _h3_cfg.video_channels;
        const int n1 = (int)ref1->shape[1];
        const std::size_t plane = (std::size_t)lh0 * lw0;
        const int total = ref_frames + n1;
        stacked.resize((std::size_t)ZC * total * plane);
        const float* a = refp;
        const float* b = ref1->as_f32();
        for (int c = 0; c < ZC; ++c) {
          float* dst = stacked.data() + (std::size_t)c * total * plane;
          std::memcpy(dst, a + (std::size_t)c * ref_frames * plane,
                      (std::size_t)ref_frames * plane * sizeof(float));
          std::memcpy(dst + (std::size_t)ref_frames * plane,
                      b + (std::size_t)c * n1 * plane,
                      (std::size_t)n1 * plane * sizeof(float));
        }
        refp = stacked.data();
        ref_frames = total;
      }
    }
    // Cooperative stop, polled per BLOCK inside the DiT. A step is one
    // pass over 50 blocks of a 33B stack, so stopping only between steps
    // leaves a stop request hanging for tens of seconds; this makes it
    // land in about a block. Cleared afterwards so a stale `ctx` is never
    // captured past this generation -- the lambda holds a reference.
    auto stopping = [&ctx]() { return ctx.stop_requested(); };
    if (_h3_dit) { _h3_dit->set_stream_stop(stopping); }
    const bool ok_h3 =
        run_h3_(cond->data.data(), (int)cond->shape[0], refp, ref_frames,
                have_r2v ? &r2v : nullptr,
                &vlat, &vshape, &alat_out, &ashape);
    if (_h3_dit) { _h3_dit->set_stream_stop({}); }
    // What the adaptive residency actually reached. This is the number
    // that says whether the machine got used: with 0 pinned at load, every
    // resident block here was earned at runtime out of free headroom, and
    // the count is what a "why was this slow" question needs first.
    if (_h3_dit && _h3_dit->streaming()) {
      session()->info(fmt(
          "GenerateVideoStage('{}'): DiT residency ended at {} of {} blocks "
          "({} MB), {} pinned at load", this->id(),
          _h3_dit->resident_block_count() + _h3_dit->pinned_blocks(),
          _h3_cfg.n_layers,
          (_h3_dit->resident_block_bytes()) >> 20,
          _h3_dit->pinned_blocks()));
    }
    if (!ok_h3) {
      if (ctx.stop_requested()) {
        session()->info(fmt(
            "GenerateVideoStage('{}'): stopped mid-denoise; dropping the "
            "partial clip", this->id()));
      }
      co_return;
    }
    if (_unload_idle) {
      // SETTLED with the manager, which owns the checkpoint: the reset
      // below only ends this stage's borrow, and the call asks for it to
      // be parked now if nobody else is borrowing it. The order against
      // the reset used to be the whole thing -- the manager's reference
      // was weak, so pooling after it was a silent no-op -- and is now
      // free either way.
      //
      // `auto` means REMOVABLE, not gone: the checkpoint stays alive and
      // purgeable so a relaunch pays no reload, and the kernel may take
      // the pages meanwhile. `destroy` is the caller asking for the bytes
      // back now, so it keeps dropping them.
      //
      // Settling REFUSES a set the AdaLN bake specialised, which is this
      // model on every schedule -- it is dropped rather than kept, so
      // the reload is real and visible either way.
      _h3_dit.reset();
      if (auto* mgr = session()->services()->generative_model_manager()) {
        if (_unload_cfg == model_memory::UnloadPolicy::kDestroy) {
          mgr->drop_weights(_dit_dir_declared_);
        } else {
          mgr->pool_weights(_dit_dir_declared_);
        }
      }
      // The other half of the denoise-phase claim. Peers sized against
      // the promise that these bytes are gone by the time the decodes
      // run; this is where it is kept, and saying so is what stops the
      // manager reporting it broken at the end of the launch.
      if (auto* mgr = session()->services()->generative_model_manager()) {
        mgr->note_phase_released(_dit_dir_declared_);
      }
    }

    auto vout = std::make_unique<TensorBeatPayload>();
    vout->dtype = TensorBeat::DType::F32;
    vout->shape = {vshape[0], vshape[1], vshape[2], vshape[3]};
    vout->resize_contiguous(vlat.size());
    std::memcpy(vout->as_f32(), vlat.data(), vlat.size() * sizeof(float));
    {
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign("fps", FlexData::make_real(_fps));
      sb.as_object().insert_or_assign(
          "frames", FlexData::make_int((std::int64_t)_frames));
      vout->sideband = std::move(sb);
    }
    ++_emitted;
    session()->info(fmt(
        "GenerateVideoStage('{}'): emitted MiniMax-H3 latents #{} -- video "
        "[{}, {}, {}, {}] + audio [{}, {}, {}]", this->id(), _emitted,
        vshape[0], vshape[1], vshape[2], vshape[3], ashape[0], ashape[1],
        ashape[2]));
    tag_model_(*vout);
    co_await ctx.write(0, std::move(vout));

    // The audio half. Written only when the port is connected: the audio
    // VAE is a separate decode stage, and a graph that only wants video
    // should not have to wire a sink for a beat it will not read.
    if (ctx.num_oports() > 1 && !alat_out.empty()) {
      auto aout = std::make_unique<TensorBeatPayload>();
      aout->dtype = TensorBeat::DType::F32;
      aout->shape = {ashape[0], ashape[1], ashape[2]};
      aout->resize_contiguous(alat_out.size());
      std::memcpy(aout->as_f32(), alat_out.data(),
                  alat_out.size() * sizeof(float));
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign(
          "latents_per_second",
          FlexData::make_int((std::int64_t)h3::kAudioLatentsPerSecond));
      aout->sideband = std::move(sb);
      tag_model_(*aout);
      co_await ctx.write(1, std::move(aout));
    }
    co_return;
  }

  if (ref1 != nullptr) {
    // Wan's i2v conditioning is ONE clip-shaped tensor that already spans
    // every frame, so there is no second anchor to take. Debug rather than
    // warn: a graph built for h3 should still run here.
    session()->log_debug(fmt(
        "GenerateVideoStage('{}'): iport{} (last-frame anchor) is minimax-h3 "
        "only; ignoring it", this->id(), kRefPort1));
  }

  auto* mc = session()->services()->metal_compute();
  const int ZC = 16;                       // the VAE's latent channels
  const int T  = latent_frames();
  const int h8 = _height / 8, w8 = _width / 8;
  const int text_seq = (int)cond->shape[0];
  const std::size_t nlat = (std::size_t)ZC * T * h8 * w8;
  const std::size_t plane = (std::size_t)h8 * w8;

  // ---- image-to-video conditioning channels -------------------------
  // in_channels 36 = 16 noise + 4 first-frame mask + 16 image latent. The
  // mask is 4 channels rather than 1 because the VAE's 4x temporal
  // compression folds four video frames into one latent frame: latent
  // frame 0 covers video frame 0 alone (the conditioning image) plus three
  // repeats of it, and every later latent frame covers four blank ones.
  const bool i2v = _cfg.in_channels > ZC;
  std::vector<float> cond_ch;              // [20, T, h8, w8] = mask + image
  if (i2v) {
    if (ref == nullptr || ref->shape.size() != 4 ||
        (int)ref->shape[0] != ZC || (int)ref->shape[1] != T ||
        (int)ref->shape[2] != h8 || (int)ref->shape[3] != w8) {
      session()->warn(fmt(
          "GenerateVideoStage('{}'): this checkpoint has in_channels {}, so it "
          "is image-to-video and needs a [{}, {}, {}, {}] conditioning latent "
          "on ref_latent0 (from a vae-encode over the conditioning image); "
          "skipping", this->id(), _cfg.in_channels, ZC, T, h8, w8));
      co_return;
    }
    cond_ch.assign((std::size_t)20 * T * plane, 0.0f);
    // Mask channel c of latent frame t is 1 exactly where the video frame
    // it stands for is the conditioning image: (t=0, c=0..3) covers video
    // frame 0 four times over, and nothing else does.
    for (int c = 0; c < 4; ++c) {
      float* m = cond_ch.data() + ((std::size_t)c * T) * plane;
      for (std::size_t i = 0; i < plane; ++i) { m[i] = 1.0f; }
    }
    std::memcpy(cond_ch.data() + (std::size_t)4 * T * plane, ref->as_f32(),
                nlat * sizeof(float));
  }

  // ---- schedule + initial noise --------------------------------------
  genai::FlowSchedulerSpec sched = _scheduler_spec;
  if (sched.steps <= 0) { sched.steps = _steps; }
  genai::FlowSampler sampler(_sampler_spec, sched);
  sampler.reset();
  const std::vector<double>& sigmas = sampler.sigmas();

  std::vector<float> x(nlat);
  {
    std::mt19937_64 rng(_seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) { v = nd(rng); }
  }

  // Text conditioning is projected by the EXPERT's own text_embedder, so it
  // is re-projected on a swap rather than computed once for the clip.
  SharedBuffer tpos, tneg;
  int expert_for_text = -1;
  auto upload_bf16 = [&](const TensorBeatPayload& b) {
    const std::size_t n = (std::size_t)b.shape[0] * b.shape[1];
    SharedBuffer s = mc->make_shared_buffer(n * 2);
    if (!s.empty()) { std::memcpy(s.contents(), b.as_bf16(), n * 2); }
    return s;
  };
  const SharedBuffer cond_raw = upload_bf16(*cond);
  const SharedBuffer neg_raw = neg != nullptr ? upload_bf16(*neg)
                                              : SharedBuffer{};
  const bool cfg_on = neg != nullptr;

  // The DiT input buffer, rebuilt per evaluation from the sampler's
  // candidate latent plus the constant conditioning channels.
  SharedBuffer dit_in =
      mc->make_shared_buffer((std::size_t)_cfg.in_channels * T * plane * 2);
  if (dit_in.empty()) {
    session()->warn(fmt("GenerateVideoStage('{}'): latent allocation failed",
                        this->id()));
    co_return;
  }

  bool failed = false;
  // Block-granular denoise progress, as the image DiTs report it. A 14B
  // step at 720p is seconds of silence, and this loop previously reported
  // nothing at all -- only a debug log.
  //
  // Guidance runs the DiT a second time per step, so the bar is sized for
  // two forwards when a negative prompt is present. When an expert's
  // guidance is exactly 1.0 that second forward is skipped and the step
  // fills only part way; end_step re-syncs at the boundary, so it is a
  // pacing detail rather than a wrong number.
  UiProgress bar = session()->open_progress("denoise");
  DenoiseProgress prog(&bar, sched.steps, cfg_on ? 2 : 1);
  ScopedBlockProgress<genai::MetalWanTransformer> hook(_dit.get(), prog);
  auto denoise = [&](const std::vector<float>& xin,
                     double sigma) -> std::vector<float> {
    std::vector<float> out(nlat, 0.0f);
    if (failed) { return out; }
    // Two experts, one boundary, one crossing: the schedule descends, so
    // this swaps at most once per clip.
    const int want = _wan_params.expert_for(sigma);
    if (!ensure_expert_(want)) { failed = true; return out; }
    // Crossing the boundary rebuilds _dit, which takes the old hook with
    // it; re-arm on whatever is loaded now. No-op when nothing swapped.
    hook.rearm(_dit.get());
    if (_expert != expert_for_text) {
      std::string terr;
      tpos = _dit->encode_text(cond_raw, text_seq, &terr);
      tneg = cfg_on ? _dit->encode_text(neg_raw, (int)neg->shape[0], &terr)
                    : SharedBuffer{};
      if (tpos.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): text projection: {}",
                            this->id(), terr));
        failed = true;
        return out;
      }
      expert_for_text = _expert;
    }
    auto* dp = static_cast<std::uint16_t*>(dit_in.contents());
    auto put = [&](std::size_t i, float v) {
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      dp[i] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    };
    for (std::size_t i = 0; i < nlat; ++i) { put(i, xin[i]); }
    for (std::size_t i = 0; i < cond_ch.size(); ++i) {
      put(nlat + i, cond_ch[i]);
    }
    // The scheduler's timestep is sigma on its native 0..num_train scale,
    // which is what the DiT's sinusoidal embedding was trained against.
    const float ts = (float)(sigma * (double)sched.num_train);
    const double g = _wan_params.guidance_for(_expert);
    std::string derr;
    SharedBuffer vp =
        _dit->forward(dit_in, T, h8, w8, tpos, text_seq, ts, &derr);
    if (vp.empty()) {
      session()->warn(fmt("GenerateVideoStage('{}'): DiT forward: {}",
                          this->id(), derr));
      failed = true;
      return out;
    }
    auto bf16 = [](std::uint16_t b) {
      const std::uint32_t u = (std::uint32_t)b << 16;
      float f;
      std::memcpy(&f, &u, 4);
      return f;
    };
    const auto* pp = static_cast<const std::uint16_t*>(vp.contents());
    if (cfg_on && g != 1.0) {
      SharedBuffer vn = _dit->forward(dit_in, T, h8, w8, tneg,
                                      (int)neg->shape[0], ts, &derr);
      if (vn.empty()) {
        session()->warn(fmt("GenerateVideoStage('{}'): negative forward: {}",
                            this->id(), derr));
        failed = true;
        return out;
      }
      const auto* np = static_cast<const std::uint16_t*>(vn.contents());
      for (std::size_t i = 0; i < nlat; ++i) {
        const float u = bf16(np[i]);
        out[i] = u + (float)g * (bf16(pp[i]) - u);
      }
    } else {
      for (std::size_t i = 0; i < nlat; ++i) { out[i] = bf16(pp[i]); }
    }
    return out;
  };

  {
    PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmDit, kPerfLlmDitBegin,
                       (std::uint64_t)T * (h8 / 2) * (w8 / 2));
    for (int i = 0; i < sched.steps && !failed; ++i) {
      session()->log_debug(fmt(
          "GenerateVideoStage('{}'): step {}/{} sigma {:.4f} ({}-noise "
          "expert)", this->id(), i + 1, sched.steps, sigmas[(std::size_t)i],
          _wan_params.expert_for(sigmas[(std::size_t)i]) == 1 ? "low"
                                                             : "high"));
      sampler.step(i, x, denoise);
      prog.end_step(i);
    }
  }
  bar.finish();
  if (failed) { co_return; }

  if (_unload_idle) {
    _dit.reset();
    _expert = -1;
    expert_for_text = -1;
  }

  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::F32;
  out->shape = {ZC, T, h8, w8};
  out->resize_contiguous(nlat);
  std::memcpy(out->as_f32(), x.data(), nlat * sizeof(float));
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign("fps", FlexData::make_real(_fps));
  sb.as_object().insert_or_assign("frames",
                                  FlexData::make_int((std::int64_t)_frames));
  out->sideband = std::move(sb);
  ++_emitted;
  session()->info(fmt(
      "GenerateVideoStage('{}'): emitted latent video #{} [{}, {}, {}, {}] "
      "({} frames at {}x{}, {:.3f} fps)", this->id(), _emitted, ZC, T, h8, w8,
      _frames, _width, _height, _fps));
  tag_model_(*out);
  co_await ctx.write(0, std::move(out));
}

#else   // !VPIPE_BUILD_APPLE_SILICON

Job
GenerateVideoStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (session()) {
    session()->error(fmt(
        "GenerateVideoStage('{}'): built without VPIPE_BUILD_APPLE_SILICON; "
        "the metal Wan DiT is unavailable, the stage is inert", this->id()));
  }
  co_return;
}

Job
GenerateVideoStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  (void)in;
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(GenerateVideoStage)
VPIPE_REGISTER_SPEC(GenerateVideoStage, kSpec)

}  // namespace vpipe
