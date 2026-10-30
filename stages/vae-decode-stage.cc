#include "generative-models/flux3/metal-flux3-transformer.h"
#include "generative-models/shared/accel-settings.h"
#include "stages/vae-decode-stage.h"
#include "common/host-report.h"
#include "common/flex-bag.h"
#include "common/beat-keys.h"

#include <cstring>

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-detect.h"
#include "stages/model-memory-host.h"
#include "stages/model-provenance.h"
#include "stages/model-registry.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {

VaeDecodeStage::VaeDecodeStage(const SessionContextIntf* s,
                               std::string               id,
                               std::vector<InEdge>       iports,
                               FlexData                  config)
  : TypedStage<VaeDecodeStage>(s, std::move(id), std::move(iports),
                               std::move(config))
{
  // Deferred validation (see Stage::fail_config): construct for any config so
  // a graph can be built/edited before hf_dir is supplied. hf_dir is OPTIONAL
  // now -- a model-select source on the model iport can supply it instead --
  // so the "no model at all" case is reported at initialize()/process() time
  // (when iport connectivity is known), not here.
  _hf_dir    = attr_str("hf_dir");
  _keep_loaded = attr_real("keep_loaded");
  _vae_weights = attr_str("vae_dir");
  _fast_decoder = attr_str("fast_decoder");
  // THE ACCELERATION BAG, built once and handed to a registered family
  // whole. Written whether the tier is on or off, and with the SETTLED
  // value, so no family repeats this stage's reading or reaches a
  // different answer than the log line did. The built-in codecs never
  // see it: they are compiled with this stage and take their kernels
  // from their own configs.
  _accel = FlexData::make_object();
  genai::accel::set_flag(&_accel, genai::accel::kI8Gemm,
                         attr_bool("i8_gemm"));
  {
    const std::string t = attr_str("i8_gemm_tier");
    const bool known = t == genai::accel::kI8TierFast ||
                       t == genai::accel::kI8TierToken ||
                       t == genai::accel::kI8TierTokenPlain;
    if (!t.empty() && !known && t != genai::accel::kI8TierAccurate) {
      fail_config(fmt("VaeDecodeStage('{}'): i8_gemm_tier must be 'accurate', "
                      "'fast', 'token' or 'token_plain' (got '{}')",
                      this->id(), t));
    }
    genai::accel::set_text(&_accel, genai::accel::kI8GemmTier,
                           known ? t
                                 : std::string(genai::accel::kI8TierAccurate));
  }
  genai::accel::set_flag(&_accel, genai::accel::kAneFfn, attr_bool("ane_ffn"));
  {
    double rows = attr_real("ane_rows");
    if (!(rows >= 0.0)) { rows = 0.0; }
    if (rows > 1.0) { rows = 1.0; }
    genai::accel::set_real(&_accel, genai::accel::kAneRows, rows);
    const long long layers = attr_int("ane_layers");
    genai::accel::set_integer(&_accel, genai::accel::kAneLayers,
                              layers > 0 ? layers : 0);
  }
#ifdef VPIPE_BUILD_APPLE_SILICON
  _fps = attr_real("fps");
  {
    const std::string dt = attr_str("dtype");
    if (dt == "f16") {
      _out_dtype = TensorBeat::DType::F16;
    } else if (dt != "u8") {
      fail_config(fmt("VaeDecodeStage('{}'): dtype must be u8 or f16",
                      this->id()));
    }
  }
  if (!(_fps > 0.0)) { _fps = 24.0; }
  {
    bool bad = false;
    _unload_cfg = model_memory::parse_unload_policy(
        attr_str("unload_when_idle"), &bad);
    if (bad) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): unload_when_idle '{}' is not auto|always|never; "
          "using auto", this->id(), attr_str("unload_when_idle")));
    }
  }
#endif
  allocate_oports(spec().oports.size());
}

VaeDecodeStage::~VaeDecodeStage()
{
  // Kept warm (`keep_loaded`) where it was kept resident between beats:
  // a stage that unloads after each beat does so because the box is
  // tight.
  if ((_vae || _flux2_vae) && _keep_loaded > 0 && !_unload_idle &&
      !_vae_warm_key.empty() && session() != nullptr) {
    if (auto* mgr = session()->services()->generative_model_manager()) {
      session()->info(fmt("VaeDecodeStage('{}'): the VAE kept warm for "
                          "{:.0f} s", this->id(), _keep_loaded));
      if (_vae) {
        mgr->keep_model_warm(kKrea2VaeWarmKind, vae_dir_for_release_(),
                             _vae_warm_key, std::move(_vae), 0,
                             _keep_loaded);
      } else {
        mgr->keep_model_warm(kFlux2VaeWarmKind, vae_dir_for_release_(),
                             _vae_warm_key, std::move(_flux2_vae), 0,
                             _keep_loaded);
      }
    }
  }
}

namespace {
// The model iport (a model-select source) overrides hf_dir. Appended after
// the primary `latent` input, so it is iport1. (Referenced only from the
// Apple-gated code below; marked maybe_unused for the inert non-Apple build.)
[[maybe_unused]] constexpr unsigned kModelPort = 1;
// The wavelet colour fix's source pictures, read in lockstep with the
// latents that announce them (sideband::kColorRefFrames).
[[maybe_unused]] constexpr unsigned kColorRefPort = 2;

// MiniMax-H3's video VAE works in IMAGENET-normalized pixel space, so
// its decoder's output has to be un-normalized before it means [0, 1].
// These are the reference's own constants, not a preprocessing choice.
[[maybe_unused]] constexpr float kImagenetMean[3] = {0.485f, 0.456f,
                                                     0.406f};
[[maybe_unused]] constexpr float kImagenetStd[3]  = {0.229f, 0.224f,
                                                     0.225f};

// What THIS stage implements: a two-way decision (destroy or keep, else
// resolve from the box). "park" would land in the `auto` branch and mean
// something other than what it says, so it is not offered. The legacy
// "always" / "never" stay accepted and show as unlisted.
constexpr SpecExtra kUnloadChoices[] = {
  {spec_key::kChoices, "auto,destroy,keep"},
};
constexpr SpecExtra kI8TierChoices[] = {
  {spec_key::kChoices, "accurate,fast,token,token_plain"},
};
constexpr SpecExtra kDtypeChoices[] = {
  {spec_key::kChoices, "u8,f16"},
};
const ConfigKey kAttrs[] = {
  {.key = "dtype", .type = ConfigType::String, .required = false,
   .doc = "u8 | f16: the pictures this emits. u8 (the default) quantizes "
          "the decoder's samples to 8 bits, what most consumers read. f16 "
          "keeps them, 0..1 -- for a 16-bit PNG/TIFF, OpenEXR or a 10-bit "
          "video (save-image, avf-save-video), nothing is lost here",
   .def_str = "u8", .extra = kDtypeChoices},
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "the model root whose VAE this decodes; the resident family "
          "locates the VAE within it (conventionally <hf_dir>/vae). "
          "OPTIONAL: a model-select source on the model iport overrides it. "
          "May name a STANDALONE VAE (krea2-vae) rather than a whole "
          "model: that is how a latent is decoded through a VAE other "
          "than the one its own model ships. Give the VAE its own "
          "model-select source, or set this key directly, so the DiT "
          "stages keep pointing at the model",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "krea2,krea2-vae,flux2,flux3,fibo2,qwen-image,"
                      "qwen-image-edit,qwen-image-21,"
       "boogu-image,boogu-image-edit,"
       "wan-t2v,wan-i2v,minimax-h3-fl2va,minimax-h3-ref2va,"
       "minimax-h3-image-vae,vosr",
   .model_channel = "diffusion-model"},
  {.key = "vae_dir", .type = ConfigType::String, .required = false,
   .doc = "the VAE's WEIGHTS, when they are not hf_dir's own: a "
          "directory, a registry key, or a single file -- a Draw Things "
          ".ckpt (Qwen-Image-2.1's VAE, MiniMax-H3's video VAE), read under "
          "the release's names. The config (latent statistics, geometry) "
          "still comes from hf_dir's VAE",
   .is_path = true, .path_filter = "*.safetensors,*.ckpt"},
  {.key = "fast_decoder", .type = ConfigType::String, .required = false,
   .doc = "MiniMax-H3 only: decode through the FAST decoder Open-H3-"
          "Regenerate-2K ships (vae_fast/, from NVIDIA's StreamRender-H3) "
          "instead of the released one -- a registry key, its directory "
          "or the .safetensors. LOSSY: the release puts it ~30 dB from "
          "the released decoder. A 24-layer ViT over windows of 5 latent "
          "frames x 16 x 16 tokens with no tiles and no overlap: MEASURED "
          "88 s against 318 for a 2560x1440, 39-frame clip on an M4 "
          "Pro. Holds 3.2 GB (bf16) where the released decoder holds "
          "4.6, but its working set grows with the clip -- it keeps every "
          "token between layers: 12.2 GB for 124 frames at 2K, besides "
          "the clip it emits. The plan learns that at the first clip; "
          "before it, it counts the released decoder's working set. "
          "Takes i8_gemm / i8_gemm_tier (matrix cores) and "
          "ane_ffn like the released one; the latent statistics still "
          "come from hf_dir's VAE. A single latent frame (a picture) is "
          "not its to decode",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "minimax-h3-fast-vae",
   .is_path = true, .path_filter = "*.safetensors"},
  {.key = "fps", .type = ConfigType::Real, .required = false,
   .doc = "frame rate stamped on each decoded VIDEO frame's sideband when "
          "the latent does not carry one. A latent from generate-video does "
          "carry it, so this is the fallback for one read from elsewhere. "
          "Ignored when the latent is an image",
   .def_real = 24.0},
  {.key = "unload_when_idle", .type = ConfigType::String, .required = false,
   .doc = "drop the VAE weights after each beat and reload on the next one. "
          "This stage is idle for the whole denoise, so on a memory-bounded box "
          "releasing it (weights AND the decode working set) is what lets a "
          "large DiT run in the same machine. \"auto\" (default) decides from "
          "physical RAM vs the pipeline's weight bytes; \"destroy\" / "
          "\"keep\" force it (legacy spellings: \"always\" / \"never\"). "
          "\"park\" is NOT implemented here and resolves as \"auto\": "
          "parking is the manager's to do and needs this stage's borrow "
          "to end first -- only diffusion-conditioner does that today",
   .def_str = "auto", .extra = kUnloadChoices},
  {.key = "i8_gemm", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated mode (LOSSY): dynamic-int8 GEMMs for the codec's big "
          "matmuls. OFF BY DEFAULT AND FOR A QUALITY REASON, not because it "
          "is unimplemented: a codec is the last thing between a latent and "
          "the pixels a person looks at, so an approximation in it has "
          "nowhere to be absorbed -- where the same trade inside a DiT is "
          "followed by dozens of blocks and a decode. It is settable so the "
          "trade can be MEASURED. Read by the Qwen-Image codec (Krea-2's, "
          "Qwen-Image-2.1's: its 3x3 convs on the int8 hardware "
          "convolution, an activation channel at a step of its own, and "
          "its materialized mid-block attention's two GEMMs -- MEASURED "
          "1.15x / 1.26x a 1024^2 decode at 'accurate' / 'fast', 52-60 / "
          "50-57 dB from the f16 picture), by the 2D AutoencoderKL (FLUX.2's "
          "and FLUX.1's: FLUX.2, FLUX.3, Z-Image, MOSS-Image, Boogu, Fibo "
          "2.0 -- its 3x3 convs the same way), by MiniMax-H3's fast decoder "
          "(fast_decoder: its block projections -- MEASURED 1.34x / 1.94x a "
          "768P decode at 'accurate' / 'token' on a base M5) and by a "
          "REGISTERED family; the other built-in codecs ignore it. Matrix "
          "cores only. Env VPIPE_I8_GEMM overrides",
   .def_bool = false},
  {.key = "i8_gemm_tier", .type = ConfigType::String, .required = false,
   .doc = "which int8 i8_gemm runs, in generate-image's words: 'accurate' "
          "(default) takes only the resnet convs -- their inputs are "
          "RMS-normed -- and runs the attention's GEMMs at groups of 64; "
          "'fast', 'token' and 'token_plain' take every 3x3 the route "
          "fits, the residual stream's too, and run the GEMMs at that "
          "tier (a depth its layout cannot take stays accurate). The "
          "Qwen-Image codec and the 2D AutoencoderKL (convs only: its mid "
          "attention runs a flash kernel); MiniMax-H3's fast decoder holds "
          "its block projections in the int8 layout from 'fast' up; others "
          "read 'accurate'",
   .def_str = "accurate", .extra = kI8TierChoices},
  {.key = "keep_loaded", .type = ConfigType::Real, .required = false,
   .doc = "seconds to keep the VAE BUILT after this stage ends (its graph "
          "unloaded), where it is kept resident: the next graph asking for "
          "the same checkpoint and settings is handed it -- no read, no "
          "int8 convs made again. Released sooner when a launch that does "
          "not use it needs the room. 0 (the default) lets it go with the "
          "graph. The Qwen-Image / Krea-2 VAE and FLUX.2's AutoencoderKL",
   .def_real = 0.0},
  {.key = "ane_ffn", .type = ConfigType::Bool, .required = false,
   .doc = "accelerated decode (LOSSY, fp16): split the decoder blocks' "
          "feed-forward between the GPU and the Apple Neural Engine by rows, "
          "concurrently. MiniMax-H3's video VAE only, released or fast: its "
          "decoder is a ViT whose feed-forward is ~70% of the decode, and "
          "attention stays on the GPU. One shared runtime-weight module "
          "(~0.3 GB), claimed from the resource plan and staged per tile "
          "(the fast decoder: per layer) per block. MEASURED on an M4 Pro, "
          "the fast decoder at 2560x1440 x 39 frames: 59 s against 88. "
          "Other families ignore it",
   .def_bool = false},
  {.key = "ane_rows", .type = ConfigType::Real, .required = false,
   .doc = "the ANE's share of each feed-forward's rows, rounded to whole "
          "chunks; 0 (default) balances from the two engines' measured times",
   .def_real = 0.0},
  {.key = "ane_layers", .type = ConfigType::Int, .required = false,
   .doc = "cap on how many decoder blocks use the ANE feed-forward; 0 "
          "(default) means all. Not a memory setting -- the blocks share one "
          "module -- so set it only to cover fewer blocks on purpose",
   .def_int = 0},
};
const PortSpec kIports[] = {
  {.name = "latent",
   .doc = "f32 latent, unpacked and whitened: [z, H/r, W/r] for an image, "
          "[z, T, H/r, W/r] for a clip. z and the spatial stride r are the "
          "resident family's VAE geometry",
   .type = &typeid(TensorBeatPayload),
   .tags = "latent", .clock_group = 0},
  {.name = "model", .doc = "OPTIONAL shared model reference from a model-select "
                           "source; overrides the hf_dir config",
   .type = &typeid(FlexDataPayload), .clock_group = 0},
  {.name = "color_reference",
   .doc = "OPTIONAL source pictures, planar [n, 3, H, W] (U8, F16 or F32): "
          "one beat for each that a latent's sideband announces -- "
          "`color_ref_frames` for the WAVELET colour fix, then "
          "`decoder_cond_frames` for a TINY decoder that reads the source "
          "beside the latent (FlashVSR's, which then decodes in place of "
          "the VAE) -- and read only for those. flashvsr-src-encoder's "
          "`color_reference` output",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};
const PortSpec kOports[] = {
  {.name = "image",
   .doc = "decoded planar U8 TensorBeat [C, H, W] -- RGB, or RGBA when "
          "the VAE decodes four channels (Qwen-Image-2.1). "
          "A VIDEO latent emits "
          "one beat PER FRAME rather than a clip-shaped tensor, each "
          "carrying {frame, frames, fps} in its sideband, so a consumer "
          "knows where in the clip it sits and how many follow",
   .type = &typeid(TensorBeatPayload),
   // The DECLARED group is the image answer; the real one is resolved
   // per checkpoint in oport_clock_group(), because a video family
   // emits F of these per latent and an image family one. See that
   // override -- the analyzer reads it, not this.
   .tags = "rgb-frames", .clock_group = 0},
  {.name = "clip",
   .doc = "OPTIONAL: the same pixels as ONE beat, planar U8 RGB "
          "[frames, 3, H, W] with {frames, fps} on the sideband -- the "
          "shape temporal-stack builds, so it is what a "
          "reference-conditioned model takes. Written only for a VIDEO "
          "latent and only when a consumer is wired, because building it "
          "costs a copy of the whole clip. oport0 is unchanged, so a "
          "graph that wants frames needs no unpacking",
   .type = &typeid(TensorBeatPayload),
   .tags = "rgb-clip", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "vae-decode",
  .doc       = "Decodes a VAE latent into planar U8 RGB on the metal-compute "
               "backend -- the second half of the generate/decode split, for "
               "images and for video both. An IMAGE latent produces one "
               "beat; a VIDEO latent produces ONE BEAT PER FRAME, so the "
               "downstream is the per-frame machinery that already exists "
               "(save-image, rgb-to-video -> save-video, a preview), and "
               "optionally the whole clip in one beat on oport1. Which "
               "decoder runs is the resident family's, and a family "
               "registered by a plugin decodes its own.",
  .display_name = "VAE Decode",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = kOports,
  .attrs     = kAttrs,
};

}  // namespace

const StageSpec&
VaeDecodeStage::spec() const noexcept
{
  return kSpec;
}

void
VaeDecodeStage::apply_constant(unsigned iport, const FlexData& beat)
{
  // Pre-launch twin of the runtime latch in process(): the same
  // beat and the same parse, early enough that declare_resources()
  // sees the model. Bookkeeping only -- nothing loads here; the
  // pipeline is not assembled yet (see Stage::apply_constant).
  if (iport != kModelPort) { return; }
  apply_model_select_beat(beat, _hf_dir);
}

#ifdef VPIPE_BUILD_APPLE_SILICON

void
VaeDecodeStage::reset_run_state()
{
  // If a previous run left the weights UNLOADED (the idle-unload
  // policy drops them between beats), let this launch load them again:
  // ensure_loaded_'s once-only guard is per-Stage, not per-launch, so
  // without this the stage stays inert for the whole run. When the
  // weights are still held we deliberately leave the guard set --
  // reloading on top of a resident copy is exactly what doubles peak
  // memory.
  // EVERY decoder, not three of them. _wan_vae and _h3_vae used to be
  // missing here, so a relaunch with a resident wan or H3 VAE saw all
  // the tested pointers null, cleared the guard, and let the next
  // ensure_loaded_() build a second decoder ON TOP of the live one --
  // momentarily 2x peak, which for H3's 10.4 GB is 20.8 GB.
  if (!_vae && !_flux2_vae && !_wan_vae && !_h3_vae && !_h3_fast &&
      !_plugin_dec) {
    _load_attempted = false;
    _unloaded       = false;
  }

  // Per-launch reset: the stage survives a stop/relaunch, and the
  // select sources upstream re-emit on every launch. Without this the
  // re-emitted beat is never latched and this stage keeps the previous
  // run's selection.
  _model_latched = false;
  // A stream belongs to the run that opened it.
  _wan_stream.reset();
  _tiny_stream.reset();
  if (!_tiny) { _tiny_tried = false; }
  _held.clear();
  _ref_port_noted = false;
  _h3_cut_said = false;

  // Both are measurements of the PREVIOUS run's beats. The ratchet in
  // particular: a 2048^2 clip in run 1 must not keep this stage
  // unloading through a run 2 of thumbnails, and the manager has
  // dropped the declaration those numbers refined anyway.
  _arena_decided = 0;
  _arena_stated  = 0;
  _idle_peers.clear();
}

unsigned
VaeDecodeStage::oport_clock_group(unsigned p) const noexcept
{
  // The clip is one beat per latent whatever the family, so it always
  // shares the latent's clock. That is the port a feedback loop closes
  // through, and it is true by construction rather than by probe.
  if (p != 0) { return 0; }

#ifdef VPIPE_BUILD_APPLE_SILICON
  if (_oport0_group < 0) {
    // The IMAGE answer is the fallback, not a guess dressed as one: it
    // is what this stage reported for every checkpoint before it could
    // tell them apart, so a probe that comes up empty changes nothing
    // rather than splitting a domain nobody asked to have split.
    _oport0_group = 0;
    try {
      if (!_hf_dir.empty()) {
        const std::string root = resolve_model_dir(session(), _hf_dir);
        const std::string fam  = builtin_vae_family(resolve_vae_dir(root));
        // The two families whose oport0 emits F beats for one latent.
        // A registered out-of-tree family is not classified here: it
        // decides per LATENT (rank 4 is a clip), so there is nothing to
        // read off the checkpoint, and it keeps the old answer.
        if (fam == "wan" || fam == "minimax-h3") { _oport0_group = 1; }
      }
    } catch (...) {
      // noexcept, and a checkpoint that cannot be probed must not take
      // the launch down over a clock label.
    }
  }
  return (unsigned)_oport0_group;
#else
  return 0;
#endif
}

std::vector<ResourceClaim>
VaeDecodeStage::declare_resources() const
{
  if (_hf_dir.empty()) { return {}; }
  namespace fs = std::filesystem;
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const std::string vae_dir = resolve_vae_dir(root);
  // The registry, not `_vae_family`: this runs BEFORE ensure_loaded_ has
  // latched anything, and it is const.
  if (genai::VaeModelFamily* f = genai::VaeModelRegistry::get().claim_for(
          session(), root, vae_dir, resolve_model(session(), _hf_dir).model_type)) {
    return f->declare_resources(root, vae_dir);
  }
  // PHASED to the decode. A VAE is loaded when a latent arrives and
  // dropped after -- it never coexists with the encoder or the DiT on a
  // box where that matters -- so counting it as persistent puts its
  // whole size into the conditioning and denoise columns, where it is
  // not. It declares no FLOOR because it has no streaming form: a VAE
  // holds what it weighs.
  // Through the same resolver the release uses, so the claim and its
  // release cannot name different things -- and so a repack does not
  // claim its whole repository.
  //
  // ONLY WHEN THE LOAD WAITS FOR A LATENT, which it does with a source on
  // the model iport (every shipped graph; see initialize). Named by
  // hf_dir alone the stage loads at initialize and holds the VAE from
  // there, so that claim is unphased: a phase it does not keep would
  // leave the condition and the denoise free of a VAE that is there --
  // and a peer that lets the same checkpoint go after the condition
  // (video-ref-encoder) would then have it counted in neither.
  const std::string vdir = vae_dir_for_release_();
  const bool waits = iport_edges().size() > kModelPort &&
                     iport_edges()[kModelPort].v != nullptr;
  std::vector<ResourceClaim> out =
      waits ? model_memory::weight_claims_in_phase(
                  {vdir}, model_memory::kPhaseDecode)
            : model_memory::weight_claims({vdir});
#ifdef VPIPE_BUILD_APPLE_SILICON
  // MiniMax-H3's fast decoder: its f32 file held as bf16 (or the block
  // projections' int8 layout), booked at that rather than at its size on
  // disk; its ANE module the fast decoder's.
  const std::string fast = fast_decoder_file_();
  if (!fast.empty()) {
    const std::size_t held =
        genai::MetalMiniMaxH3FastVae::weight_bytes(fast, fast_i8_tier_());
    for (ResourceClaim& c : out) {
      if (held > 0) {
        bag::set_integer(c.extra, model_memory::kClaimHeldBytes,
                         (std::int64_t)held);
      }
    }
    if (genai::accel::flag(&_accel, genai::accel::kAneFfn)) {
      genai::MetalMiniMaxH3FastVae::Config fc;
      for (auto& c : model_memory::coreml_claims(
               ane_claim_label_(),
               genai::MetalMiniMaxH3FastVae::ane_runtime_bytes(fc), 1,
               model_memory::kPhaseDecode)) {
        out.push_back(std::move(c));
      }
    }
    return out;
  }
  // MiniMax-H3's released VAE: at what the decode holds -- its decoder
  // half, bf16 (4.6 GB) -- not the F32 file's 9.9 GB with the encoder.
  if (builtin_vae_family(resolve_vae_dir(root)) == "minimax-h3") {
    const std::size_t held = genai::MetalMiniMaxH3VideoVae::held_bytes(
        vdir, genai::MetalMiniMaxH3VideoVae::Half::kDecoder);
    for (ResourceClaim& c : out) {
      if (held > 0 && c.key == vdir) {
        bag::set_integer(c.extra, model_memory::kClaimHeldBytes,
                         (std::int64_t)held);
      }
    }
  }
  // ANE RESIDENCY: one unit in the decode phase, for MiniMax-H3's VAE --
  // the only codec with the tier. CoreML holds the module's bytes where no
  // other ledger can see them. See ensure_loaded_ for the grant.
  // Gated on the SAME family probe ensure_loaded_ dispatches on: the H3
  // config reader parses other families' config.json too, and a claim for
  // a module nothing builds is a lie the plan would size peers against.
  if (genai::accel::flag(&_accel, genai::accel::kAneFfn) &&
      builtin_vae_family(resolve_vae_dir(root)) == "minimax-h3") {
    genai::MetalMiniMaxH3VideoVae::Config hc;
    std::string herr;
    if (genai::MetalMiniMaxH3VideoVae::config_from_json(
            genai::MetalMiniMaxH3VideoVae::resolve_vae_dir(root), hc,
            &herr)) {
      for (auto& c : model_memory::coreml_claims(
               ane_claim_label_(),
               genai::MetalMiniMaxH3VideoVae::ane_runtime_bytes(hc), 1,
               model_memory::kPhaseDecode)) {
        out.push_back(std::move(c));
      }
    }
  }
#endif
  return out;
}

std::string
VaeDecodeStage::ane_claim_label_() const
{
  return "ane-ffn/" + std::string(this->id());
}

Job
VaeDecodeStage::initialize(RuntimeContext& ctx)
{
  // Defer the VAE load when a model-select source feeds the model iport (its
  // beat only arrives after the init barrier, in process()). Otherwise load
  // now from the config hf_dir, as before.
  const bool model_from_iport =
      ctx.num_iports() > kModelPort && ctx.iport_connected(kModelPort);
  if (!model_from_iport) { ensure_loaded_(); }
  co_return;
}

// Loading chatter: info on the first load, debug on an idle-unload reload (a
// bounded box reloads per beat, and one line per frame is noise).
void
VaeDecodeStage::load_note_(const VpipeFormat& msg) const
{
  if (_quiet_reload) { session()->log_debug(msg); }
  else               { session()->info(msg); }
}

// Publish the arena this beat actually needs, and re-decide the idle
// policy against it.
//
// Two things are going on, and they want the same thing.
//
// The LEDGER wants the truth: `bytes` is exact for this beat, where the
// plan-time claim was a bound from config geometry -- and for video it
// is a loose one, since the pixel frame count is the VAE's expansion of
// the latent and only the loaded model knows it. Peers reading
// scratch_footprint() should see this.
//
// The DECISION follows it in BOTH directions. An earlier version only
// tightened, on the theory that a stable answer avoids churn, and that
// was wrong for the case this mechanism is shared with: a run of image
// edits at mixed sizes has one large frame and several small ones, and
// a one-way rule makes every small frame after the large one pay a
// reload it did not need. Nothing about the decision is on the critical
// path -- it is taken after a decode completes and decides only whether
// to hold the weights until the next beat. resolve_idle_unload carries
// the band that keeps a boundary-sitting arena from flipping every beat.
// The image half of the same publication, keyed on pixel size.
//
// Resolves the checkpoint itself because the decode branches do not
// carry it -- one registry lookup per beat, against a decode measured
// in hundreds of milliseconds.
void
VaeDecodeStage::publish_image_arena_(int px_w, int px_h)
{
  if (px_w <= 0 || px_h <= 0 || _hf_dir.empty()) { return; }
  revise_decode_arena_(model_memory::vae_decode_scratch_bytes(
      resolve_model_dir(session(), _hf_dir), px_w, px_h));
}

void
VaeDecodeStage::revise_decode_arena_(std::size_t bytes)
{
  if (bytes == 0 || session() == nullptr ||
      session()->services() == nullptr) {
    return;
  }
  auto* mgr = session()->services()->generative_model_manager();
  if (mgr != nullptr) { mgr->revise_scratch("vae-decode", bytes); }
  // And on the topological plan, where this stage's arena is its OWN
  // scratch rather than a claim generate-video made on its behalf. That
  // indirection exists in the claim model because a phase name has to be
  // chosen by someone who knows the geometry; here the plan asks the
  // stage that allocates, and it answers once it has a beat to answer
  // from.
  {
    StageMemory m = declare_memory();
    m.scratch = bytes;
    revise_memory(m);
  }

  if (bytes != _arena_stated) {
    session()->log_debug(fmt(
        "VaeDecodeStage('{}'): decode arena {} MB -> {} MB (this beat's real "
        "geometry; the plan could only bound it)", this->id(),
        _arena_stated >> 20, bytes >> 20));
    _arena_stated = bytes;
  }

  // Only `auto` is ours to revisit: an explicit keep/always is the
  // user's answer and a measurement does not overrule it.
  if (_unload_cfg != model_memory::UnloadPolicy::kAuto) { return; }
  if (bytes == _arena_decided) { return; }
  _arena_decided = bytes;

  const std::size_t ram = model_memory::phys_ram();
  const std::size_t fp  = model_memory::weight_footprint(session(),
                                                         _idle_peers);
  // The ANE module sits beside the arena for the whole decode, so the
  // question is whether BOTH fit next to the peers.
  const bool want = model_memory::resolve_idle_unload(
      ram, fp, bytes + _ane_bytes, _unload_idle);
  if (want == _unload_idle) { return; }
  _unload_idle = want;
  session()->log_debug(fmt(
      "VaeDecodeStage('{}'): this beat needs {} MB beside {} MB of peers "
      "on a {} MB box -> {} from now on", this->id(), bytes >> 20,
      fp >> 20, ram >> 20,
      _unload_idle ? "UNLOAD after each beat" : "keep resident"));
}

StageMemory
VaeDecodeStage::declare_memory() const
{
  StageMemory m;
  if (_hf_dir.empty()) { return m; }
  const std::string vae_dir = vae_dir_for_release_();
#ifdef VPIPE_BUILD_APPLE_SILICON
  // A registered family answers for itself. Its holding is already
  // named by the path IT owns, which on a single-file pack is the file
  // rather than the root.
  const std::string root = resolve_model_dir(session(), _hf_dir);
  if (genai::VaeModelFamily* f = genai::VaeModelRegistry::get().claim_for(
          session(), root, resolve_vae_dir(root),
          resolve_model(session(), _hf_dir).model_type)) {
    m.holdings = f->declare_holdings(root,
                                     genai::VaeModelFamily::kRoleVideo);
    if (!m.holdings.empty()) {
      // The POLICY is this stage's: a decode's weights go at its idle
      // point, which is what `releases` means here.
      for (StageHolding& h : m.holdings) { h.releases = true; }
      return m;
    }
  }
#endif
  // A VAE has no streaming form: it holds what it weighs, so `floor`
  // stays 0 and is read as `preload`. The fast decoder holds its f32 file
  // narrowed.
  std::size_t held = 0;
#ifdef VPIPE_BUILD_APPLE_SILICON
  if (!fast_decoder_file_().empty()) {
    held = genai::MetalMiniMaxH3FastVae::weight_bytes(vae_dir,
                                                      fast_i8_tier_());
  } else if (builtin_vae_family(resolve_vae_dir(root)) == "minimax-h3") {
    held = genai::MetalMiniMaxH3VideoVae::held_bytes(
        vae_dir, genai::MetalMiniMaxH3VideoVae::Half::kDecoder);
  }
#endif
  m.hold(vae_dir, held > 0 ? held : model_memory::dir_weights_bytes(vae_dir),
         0, /*releases=*/true);
  // The arena and the frames it produces are sized by the clip's
  // geometry, which arrives on a beat -- generate-video declares both,
  // because it is the stage that knows the geometry at plan time.
  return m;
}

std::string
VaeDecodeStage::vae_weights_() const
{
  if (_vae_weights.empty()) { return {}; }
  return resolve_model_dir(session(), _vae_weights);
}

#ifdef VPIPE_BUILD_APPLE_SILICON
namespace {

// The settled bag (the ctor's) as the codecs take it: the Qwen-Image
// one, and the 2D AutoencoderKL through the same two fields.
genai::MetalKrea2Vae::Accel
krea2_accel_(const FlexData& bag)
{
  namespace ac = genai::accel;
  const std::string t = ac::text(&bag, ac::kI8GemmTier);
  return {ac::flag(&bag, ac::kI8Gemm),
          t == ac::kI8TierTokenPlain ? 3
          : t == ac::kI8TierToken    ? 2
          : t == ac::kI8TierFast     ? 1 : 0};
}

}  // namespace
#endif

std::string
VaeDecodeStage::fast_decoder_file_() const
{
  if (_fast_decoder.empty() || _hf_dir.empty()) { return {}; }
#ifdef VPIPE_BUILD_APPLE_SILICON
  const std::string root = resolve_model_dir(session(), _hf_dir);
  if (builtin_vae_family(resolve_vae_dir(root)) != "minimax-h3") {
    return {};
  }
  return genai::MetalMiniMaxH3FastVae::resolve_weights(
      resolve_model_dir(session(), _fast_decoder));
#else
  return {};
#endif
}

#ifdef VPIPE_BUILD_APPLE_SILICON
int
VaeDecodeStage::fast_i8_tier_() const
{
  const auto ka = krea2_accel_(_accel);
  auto* mc = session() != nullptr && session()->services() != nullptr
                 ? session()->services()->metal_compute()
                 : nullptr;
  return genai::MetalMiniMaxH3FastVae::effective_i8_tier(
      ka.i8_gemm ? ka.i8_tier : -1,
      mc != nullptr && mc->supports_matrix_cores());
}
#endif

std::string
VaeDecodeStage::vae_dir_for_release_() const
{
  if (_hf_dir.empty()) { return {}; }
  // MiniMax-H3's fast decoder, when named, is the only checkpoint this
  // stage then reads: the released VAE gives its config alone.
  if (const std::string f = fast_decoder_file_(); !f.empty()) { return f; }
  // Weights named apart from the model (`vae_dir`) are what loads, so
  // they are what is claimed and released.
  if (const std::string w = vae_weights_(); !w.empty()) { return w; }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const std::string generic = resolve_vae_dir(root);
  // resolve_vae_dir() returns `root` UNCHANGED for a pack with no
  // `vae/config.json`, which is every Comfy-Org repack -- and naming the
  // root here would claim, and later release, the whole repository. On
  // the MiniMax-H3 repack that is 117 GB of DiT and text encoder
  // attributed to a 5 GB VAE.
  //
  // So ask the family that CAN find its own single file. Only when the
  // generic answer degenerated, so a diffusers layout keeps taking the
  // path it always did.
  //
  // Through vae_claim_dir: MiniMax-H3's released VAE is claimed where
  // its weights sit (`video_vae/source/`), as video-ref-encoder claims
  // it. Claimed here as `video_vae`, the two NESTED, and the plan kept
  // video-ref-encoder's condition alone, with this stage's decode lost.
  if (generic != root) { return vae_claim_dir(generic); }
#ifdef VPIPE_BUILD_APPLE_SILICON
  // A registered family first, for exactly the same reason the built-in
  // arm below exists: it is the only thing that knows where its own
  // weights sit in a pack with no vae/config.json. Without it every
  // release, pool and phase-release this stage performs names the ROOT,
  // and a plugin VAE credits itself with the DiT beside it.
  if (genai::VaeModelFamily* f = genai::VaeModelRegistry::get().claim_for(
          session(), root, generic, resolve_model(session(), _hf_dir)
                                        .model_type)) {
    const std::string p =
        f->vae_path(root, genai::VaeModelFamily::kRoleVideo);
    if (!p.empty()) { return p; }
  }
  const std::string h3 = genai::MetalMiniMaxH3VideoVae::resolve_vae_dir(root);
  if (h3 != root) { return h3; }
#endif
  // A STANDALONE VAE degenerates the same way and for the same reason,
  // but the answer is one level further down: the checkpoint is not a
  // directory at all, it is the single freely-named safetensors inside
  // one. This MUST be the same resolver ensure_loaded_ opens the
  // WeightSet through -- a claim, a phase release and a pool_weights
  // that name the directory while the set is keyed on the file are three
  // no-ops the manager cannot report, because it has never heard of the
  // name being released.
  return resolve_vae_weights_path(generic);
}

void
VaeDecodeStage::unload_vae_()
{
  // The phase claim's other half. declare_resources() promised these
  // bytes are gone before the peers that sized against them run; this is
  // where that promise is kept, and reporting it is what stops the
  // manager warning at the end of the launch that it was not.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    mgr->note_phase_released(vae_dir_for_release_());
  }

  // _wan_vae was in neither the guard nor the resets, so the idle policy
  // never unloaded a Wan VAE at all: with only _wan_vae held every
  // tested pointer was null and this returned early.
  if (!_vae && !_flux2_vae && !_wan_vae && !_h3_vae && !_h3_fast &&
      !_plugin_dec && !_tiny) {
    return;
  }
  // SETTLED when the policy is `auto`. The manager owns the checkpoint,
  // so dropping these models only ends this stage's borrow; what the
  // call does is ask for it to be settled now -- parked if nobody else
  // is borrowing it, left alone if someone is. The order against the
  // resets below no longer matters (it did when the manager's reference
  // was weak and there was nothing left to pool afterwards).
  _vae.reset();
  _flux2_vae.reset();
  _wan_vae.reset();
  _h3_vae.reset();
  _h3_fast.reset();
  _plugin_dec.reset();
  // The tiny decoder's MemBlock memories are its scratch: an open stream
  // keeps it, and it goes with the stream's last beat.
  if (_tiny && !_tiny_stream) {
    _tiny.reset();
    _tiny_tried = false;
    if (auto* mgr = session()->services()->generative_model_manager()) {
      mgr->pool_weights(_tiny_path);
    }
  }
  // SETTLE IT NOW that the borrow has ended. `destroy` is the caller
  // asking for the bytes back, which parking does not do; anything else
  // keeps the checkpoint for the next clip, purgeable, so a peer that
  // genuinely needs the room takes it through reclaim_at_least() and one
  // that does not leaves the next decode nothing to reload.
  if (auto* mgr = session()->services()->generative_model_manager()) {
    const std::string vdir = vae_dir_for_release_();
    if (_unload_cfg == model_memory::UnloadPolicy::kDestroy) {
      mgr->drop_weights(vdir);
    } else {
      mgr->pool_weights(vdir);
    }
  }
  _unloaded = true;
  _quiet_reload = true;
  session()->log_debug(fmt("VaeDecodeStage('{}'): VAE decoder unloaded (idle)",
                           this->id()));
}

void
VaeDecodeStage::reload_vae_()
{
  if (!_unloaded) { return; }
  _unloaded = false;
  _load_attempted = false;      // let ensure_loaded_ run its body again
  ensure_loaded_();
}

bool
VaeDecodeStage::ensure_wan_vae_()
{
  if (_wan_vae) { return true; }
  if (!_wan_deferred) { return false; }
  _wan_deferred = false;
  auto* mc = session()->services()->metal_compute();
  const std::string root = resolve_model_dir(session(), _hf_dir);
  const std::string vae_dir = resolve_vae_dir(root);
  genai::MetalWanVae::Config wcfg;
  std::string werr;
  if (!genai::MetalWanVae::config_from_json(vae_dir, wcfg, &werr) &&
      !genai::MetalWanVae::config_for_native_checkpoint(vae_dir, wcfg)) {
    session()->error(fmt(
        "VaeDecodeStage('{}'): {}; inert", this->id(), werr));
    return false;
  }
  std::shared_ptr<genai::WeightSet> ws =
      genai::open_weight_set(resolve_vae_weights_path(vae_dir), session());
  if (mc == nullptr || !ws) {
    session()->error(fmt(
        "VaeDecodeStage('{}'): no readable checkpoint under '{}'; inert",
        this->id(), vae_dir));
    return false;
  }
  load_note_(fmt("VaeDecodeStage('{}'): loading Wan video VAE from '{}'",
                 this->id(), vae_dir));
  // Decode-only: the encoder half is the vae-encode stage's business.
  _wan_vae = genai::MetalWanVae::load(ws, mc, wcfg, /*with_encoder=*/false);
  if (!_wan_vae) {
    session()->error(fmt(
        "VaeDecodeStage('{}'): failed to load the Wan VAE from '{}'; inert",
        this->id(), vae_dir));
  }
  return _wan_vae != nullptr;
}

bool
VaeDecodeStage::ensure_tiny_()
{
  if (_tiny) { return true; }
  if (_tiny_tried) { return false; }
  _tiny_tried = true;
  auto* mc = session()->services()->metal_compute();
  const std::string root = resolve_model_dir(session(), _hf_dir);
  genai::FlashVsrLayout layout;
  if (!genai::resolve_flashvsr_layout(root, &layout) ||
      layout.tiny_decoder.empty()) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): the latents ask for the model's tiny decoder "
        "and '{}' has no {} beside it; decoding with the VAE", this->id(),
        root, genai::FlashVsrTinyDecoder::kFile));
    return false;
  }
  std::string err;
  _tiny_path = layout.tiny_decoder;
  if (mc != nullptr) {
    _tiny = genai::FlashVsrTinyDecoder::load(
        genai::open_weight_set(_tiny_path, session()), mc, &err);
  }
  if (!_tiny) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): the tiny decoder at '{}' did not load ({}); "
        "decoding with the VAE", this->id(), _tiny_path,
        err.empty() ? "no metal-compute backend" : err));
    return false;
  }
  load_note_(fmt(
      "VaeDecodeStage('{}'): loading FlashVSR's tiny decoder from '{}' -- "
      "a TAEHV decoder conditioned on the source pictures", this->id(),
      _tiny_path));
  return true;
}


void
VaeDecodeStage::ensure_loaded_()
{
  if (_load_attempted) { return; }   // idempotent: load at most once
  _load_attempted = true;
  if (_hf_dir.empty()) {
    session()->error(fmt(
        "VaeDecodeStage('{}'): no model -- set config.hf_dir or wire a "
        "model-select source to the model iport; inert", this->id()));
    return;
  }
  auto* mc = session() ? session()->services()->metal_compute() : nullptr;
  if (mc == nullptr) {
    session()->error(fmt(
        "VaeDecodeStage('{}'): no metal-compute backend on this session; "
        "the stage is inert", this->id()));
    return;
  }
  const std::string root = resolve_model_dir(session(), _hf_dir);
  // FLUX.3 in BFL's own layout keeps its VAE as ae.safetensors at the
  // root: say that, rather than reporting the root's config as a VAE
  // config with no latent statistics.
  if (genai::flux3::is_bfl_root_layout(root)) {
    session()->error(fmt("VaeDecodeStage('{}'): {}; inert", this->id(),
                         genai::flux3::bfl_root_layout_message(root)));
    return;
  }
  // An out-of-tree family gets asked FIRST, and it has to happen here --
  // before the unload verdict and before open_weight_set below. Both of
  // those assume the diffusers layout: resolve_vae_dir returns `root`
  // unchanged for a pack with no `vae/config.json`, vae_family_ then
  // falls through to "krea2", and open_weight_set(root) would index
  // whatever else lives in that tree (for LTX-2.5, 39 GB of DiT).
  const std::string probe_vae_dir = resolve_vae_dir(root);
  if (!_family_probed) {
    _family_probed = true;
    _vae_family = genai::VaeModelRegistry::get().claim_for(
        session(), root, probe_vae_dir, resolve_model(session(), _hf_dir).model_type);
  }
  // Idle-unload decision (auto: the DiT + text encoder of this same pipeline are
  // resident during a run, so size the box against THEIR weights -- the VAE's
  // own are small, it is the decode working set that has to fit beside them).
  if (!_unload_resolved) {
    _unload_resolved = true;
    // The built-in guess is the DIFFUSERS spelling, which sums to ZERO on
    // a Comfy-style pack -- and a zero peer footprint reads as "roomy,
    // keep the VAE resident", which beside a 39 GB DiT is the exact wrong
    // answer. A registered family names its own peers instead.
    const std::vector<std::string> peers =
        (_vae_family != nullptr)
            ? _vae_family->idle_peers(root)
            : std::vector<std::string>{
                  (std::filesystem::path(root) / "transformer").string(),
                  (std::filesystem::path(root) / "text_encoder").string(),
                  (std::filesystem::path(root) / "mllm").string()};
    _idle_peers = peers;
    const std::size_t fp = model_memory::weight_footprint(session(), peers);
    // The decode ARENA, if the stage that has the geometry declared one.
    //
    // This is the quantity the decision was always about and could not
    // name: the VAE's own weights are small, and what has to fit beside
    // the peers is the working set. Until it was declared the only
    // stand-in was kHeadroom -- a flat 6 GB that is wildly wrong in both
    // directions, being ~3x the FLUX.2 arena at 1024^2 and less than
    // half of it at 2K.
    //
    // 0 when nothing declared one (no generate-image in the graph, an
    // unreadable VAE config), and then this falls back to exactly the
    // old behaviour rather than to a guess of its own.
    const std::size_t arena =
        model_memory::scratch_footprint(session(), model_memory::kPhaseDecode);
    const std::size_t need = arena > 0 ? arena : model_memory::kHeadroom;
    switch (_unload_cfg) {
      case model_memory::UnloadPolicy::kAlways: _unload_idle = true;  break;
      case model_memory::UnloadPolicy::kNever:  _unload_idle = false; break;
      default: {
        const std::size_t ram = model_memory::phys_ram();
        _unload_idle = ram != 0 && ram < fp + need;
        break;
      }
    }
    session()->log_debug(fmt(
        "VaeDecodeStage('{}'): peer footprint {} MB + {} MB {} vs {} MB "
        "RAM, unload_when_idle={} -> {}", this->id(), fp >> 20, need >> 20,
        arena > 0 ? "declared decode arena" : "headroom (no arena declared)",
        model_memory::phys_ram() >> 20,
        model_memory::unload_policy_name(_unload_cfg),
        _unload_idle ? "UNLOAD after each beat" : "keep resident"));
  }
  namespace fs = std::filesystem;
  // `vae/` for every diffusers checkpoint, `video_vae/source/` for
  // MiniMax-H3. Shared with vae-encode so the two halves of one model
  // can never resolve to different directories.
  const std::string vae_dir = probe_vae_dir;

  if (_vae_family != nullptr) {
    // `_family` carries the tag so the log lines read like a built-in's,
    // but every dispatch is guarded on the POINTER -- a family that
    // tagged itself "wan" still cannot fall into the wan branch.
    _family = std::string(_vae_family->tag());
    genai::VaeModelCreateArgs args;
    args.root       = root;
    args.vae_dir    = vae_dir;
    args.model_type = resolve_model(session(), _hf_dir).model_type;
    args.metal      = mc;
    args.session    = session();
    try {
      _plugin_dec = _vae_family->load_decoder(args);
    } catch (const std::exception& e) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): VAE family '{}' threw loading '{}': {}; "
          "inert", this->id(), _family, root, e.what()));
      return;
    } catch (...) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): VAE family '{}' threw a non-standard "
          "exception loading '{}'; inert", this->id(), _family, root));
      return;
    }
    if (!_plugin_dec) {
      // The family already said why through args.session.
      return;
    }
    load_note_(fmt(
        "VaeDecodeStage('{}'): VAE family '{}' loaded -- {} latent channels, "
        "1/{} spatial, {}x temporal ({:.1f} MB)", this->id(), _family,
        _plugin_dec->latent_channels(), _plugin_dec->spatial_compression(),
        _plugin_dec->temporal_compression(),
        (double)_plugin_dec->resident_bytes() / (1024.0 * 1024.0)));
    return;                        // NOT the built-in chain below
  }

  std::string unknown_class;
  _family = builtin_vae_family(vae_dir, &unknown_class);
  if (_family.empty()) {
    // Refused BY NAME. Read as the default it would have been opened as
    // a Qwen-Image VAE and failed on whatever that model checks first.
    session()->error(fmt(
        "VaeDecodeStage('{}'): '{}' names the VAE class '{}', which nothing "
        "built in reads ({}) and no registered family claims; the stage "
        "is inert", this->id(), vae_dir, unknown_class,
        builtin_vae_classes()));
    return;
  }
  // One shared, reference-counted view of this checkpoint. The peer VAE
  // stage in the same graph (encode beside decode) names the same
  // directory and gets the SAME set, so the two halves of the model are
  // loaded once between them instead of once each -- and two pipelines
  // running the same model share it too.
  // Falls back to a private set when the session has no manager; see
  // open_weight_set().
  // MiniMax-H3 keeps the WEIGHTS one level below the config that named
  // the family, so the set is opened on the resolved leaf. Every other
  // family's resolver returns what it was handed -- and a STANDALONE VAE
  // is one freely-named safetensors in a directory nothing globs, so
  // that one is opened on the FILE. See resolve_vae_weights_path.
  const std::string ws_override = vae_weights_();
  const std::string ws_dir =
      !ws_override.empty() ? ws_override
      : (_family == "minimax-h3")
          ? genai::MetalMiniMaxH3VideoVae::resolve_vae_dir(vae_dir)
          : resolve_vae_weights_path(vae_dir);
  if (!ws_override.empty()) {
    load_note_(fmt("VaeDecodeStage('{}'): the VAE's weights from '{}', its "
                   "config from '{}'", this->id(), ws_override, vae_dir));
  }
  std::shared_ptr<genai::WeightSet> ws =
      genai::open_weight_set(ws_dir, session());
  if (!ws) {
    session()->error(fmt(
        "{}('{}'): no readable checkpoint under '{}'; inert",
        "VaeDecodeStage", this->id(), vae_dir));
    return;
  }
  if (_family == "flux2") {
    // FLUX.2 AutoencoderKLFlux2: read the geometry from config.json; the latent
    // whitening (patch + BatchNorm) is internal, so no latents_mean/std here.
    genai::MetalFlux2Vae::Config fcfg;
    std::ifstream in(fs::path(vae_dir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto obj = fd.as_object();
        if (obj.contains("latent_channels")) {
          fcfg.latent_channels =
              (int)obj.at("latent_channels").as_int(fcfg.latent_channels);
        }
        if (obj.contains("norm_num_groups")) {
          fcfg.norm_groups =
              (int)obj.at("norm_num_groups").as_int(fcfg.norm_groups);
        }
        if (obj.contains("block_out_channels")) {
          FlexData bo = obj.at("block_out_channels");
          auto sp = bo.as_real_span();
          for (int i = 0; i < 4 && i < (int)sp.size(); ++i) {
            fcfg.block_out[i] = (int)sp[(std::size_t)i];
          }
        }
        if (obj.contains("layers_per_block")) {
          fcfg.layers_per_block =
              (int)obj.at("layers_per_block").as_int(fcfg.layers_per_block);
        }
      }
    }
    _vae_warm_key = fmt("flux2|{}|{}|{} {} {} {} {} {} {} {}", vae_dir,
                        ws_dir, fcfg.patch, fcfg.latent_channels,
                        fcfg.norm_groups, fcfg.block_out[0],
                        fcfg.block_out[1], fcfg.block_out[2],
                        fcfg.block_out[3], fcfg.layers_per_block)();
    // i8_gemm, settled in the ctor: flags on the model, no weights of its
    // own (the int8 weight is folded per call), so a warm model takes
    // this graph's setting as a fresh one does.
    auto settle_accel = [&] {
      const auto ka = krea2_accel_(_accel);
      _flux2_vae->set_accel({ka.i8_gemm, ka.i8_tier});
      if (const std::string note = _flux2_vae->accel_note(); !note.empty()) {
        session()->log_normal(
            fmt("VaeDecodeStage('{}'): {}", this->id(), note));
      }
    };
    if (_keep_loaded > 0 && !_unload_idle) {
      if (auto* mgr = session()->services()->generative_model_manager()) {
        if (auto warm = mgr->take_warm<genai::MetalFlux2Vae>(
                kFlux2VaeWarmKind, vae_dir_for_release_(), _vae_warm_key)) {
          _flux2_vae = std::move(warm);
          load_note_(fmt("VaeDecodeStage('{}'): the AutoencoderKL kept "
                         "warm is taken back", this->id()));
          settle_accel();
          return;
        }
      }
    }
    load_note_(fmt(
        "VaeDecodeStage('{}'): loading 2D AutoencoderKL ({}) from '{}'",
        this->id(), fcfg.patch > 1 ? "FLUX.2, patchified" : "plain", vae_dir));
    _flux2_vae = genai::MetalFlux2Vae::load(ws, mc, fcfg);
    if (!_flux2_vae) {
      session()->error(fmt(
          "VaeDecodeStage('{}'): failed to load the AutoencoderKL from '{}'; "
          "inert",
          this->id(), vae_dir));
      return;
    }
    settle_accel();
    return;
  }

  if (_family == "minimax-h3" && !_fast_decoder.empty()) {
    // THE FAST DECODER: its own weights, the released VAE's config (the
    // latent statistics) and nothing else of it.
    const std::string file = fast_decoder_file_();
    std::string ferr;
    if (file.empty()) {
      genai::MetalMiniMaxH3FastVae::resolve_weights(
          resolve_model_dir(session(), _fast_decoder), &ferr);
      session()->error(fmt("VaeDecodeStage('{}'): fast_decoder '{}': {}; "
                           "inert", this->id(), _fast_decoder, ferr));
      return;
    }
    std::shared_ptr<genai::WeightSet> fws =
        genai::open_weight_set(file, session());
    genai::MetalMiniMaxH3FastVae::Config fcfg;
    if (!fws ||
        !genai::MetalMiniMaxH3FastVae::config_from(*fws, vae_dir, &fcfg,
                                                   &ferr)) {
      session()->error(fmt(
          "VaeDecodeStage('{}'): '{}' is not MiniMax-H3's fast decoder "
          "({}); inert", this->id(), file,
          ferr.empty() ? std::string("unreadable") : ferr));
      return;
    }
    fcfg.i8_tier = fast_i8_tier_();
    if (genai::accel::flag(&_accel, genai::accel::kAneFfn)) {
      const std::size_t bytes =
          genai::MetalMiniMaxH3FastVae::ane_runtime_bytes(fcfg);
      if (model_memory::coreml_grant(session(), ane_claim_label_(), bytes,
                                     1) > 0) {
        fcfg.ane_ffn    = true;
        fcfg.ane_rows   =
            (float)genai::accel::real(&_accel, genai::accel::kAneRows, 0.0);
        fcfg.ane_layers = (int)genai::accel::integer(
            &_accel, genai::accel::kAneLayers, 0);
        _ane_bytes = bytes;
      } else {
        session()->info(fmt(
            "VaeDecodeStage('{}'): the ANE feed-forward was requested but "
            "the plan left no room for its module ({} MB); keeping the GPU",
            this->id(), bytes >> 20));
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    _h3_fast = genai::MetalMiniMaxH3FastVae::load(fws, mc, fcfg, &ferr);
    if (!_h3_fast) {
      session()->error(fmt(
          "VaeDecodeStage('{}'): failed to load MiniMax-H3's fast decoder "
          "from '{}' ({}); inert", this->id(), file, ferr));
      return;
    }
    const int tier = _h3_fast->i8_tier();
    load_note_(fmt(
        "VaeDecodeStage('{}'): MiniMax-H3's FAST decoder (lossy) from '{}' "
        "in {:.1f} s -- {} layers, {} MB held; GEMMs {}{}, attention {}{}",
        this->id(), file,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count(),
        fcfg.n_layers, _h3_fast->weight_bytes() >> 20,
        _h3_fast->matrix_cores() ? "matmul2d" : "steel",
        tier < 0 ? std::string()
                 : fmt(" + int8 {}", tier == 0   ? "accurate"
                                     : tier == 1 ? "fast"
                                     : tier == 2 ? "token"
                                                 : "token_plain")(),
        _h3_fast->nax_attention() ? "matrix-core flash" : "steel flash",
        fcfg.ane_ffn ? ", the ANE taking a share of each feed-forward"
                     : ""));
    return;
  }
  if (_family == "minimax-h3") {
    genai::MetalMiniMaxH3VideoVae::Config h3cfg;
    std::string h3err;
    if (!genai::MetalMiniMaxH3VideoVae::config_from_json(vae_dir, h3cfg,
                                                         &h3err)) {
      session()->error(fmt(
          "VaeDecodeStage('{}'): {}; inert", this->id(), h3err));
      return;
    }
    // The ANE tier runs only on the plan's grant: its module is one fixed
    // unit, and without the grant the config stays off.
    if (genai::accel::flag(&_accel, genai::accel::kAneFfn)) {
      const std::size_t bytes =
          genai::MetalMiniMaxH3VideoVae::ane_runtime_bytes(h3cfg);
      if (model_memory::coreml_grant(session(), ane_claim_label_(), bytes,
                                     1) > 0) {
        h3cfg.ane_ffn    = true;
        h3cfg.ane_rows   =
            (float)genai::accel::real(&_accel, genai::accel::kAneRows, 0.0);
        h3cfg.ane_layers = (int)genai::accel::integer(
            &_accel, genai::accel::kAneLayers, 0);
        _ane_bytes = bytes;
      } else {
        session()->info(fmt(
            "VaeDecodeStage('{}'): the ANE feed-forward was requested but "
            "the plan left no room for its module ({} MB); keeping the GPU",
            this->id(), bytes >> 20));
      }
    }
    load_note_(fmt("VaeDecodeStage('{}'): loading the MiniMax-H3 video VAE "
                   "from '{}'", this->id(), vae_dir));
    _h3_vae = ws_override.empty()
                  ? genai::MetalMiniMaxH3VideoVae::load(vae_dir, mc, h3cfg)
                  : genai::MetalMiniMaxH3VideoVae::load(ws, mc, h3cfg);
    if (!_h3_vae) {
      session()->error(fmt(
          "VaeDecodeStage('{}'): failed to load the MiniMax-H3 video VAE "
          "from '{}'; inert", this->id(), vae_dir));
    }
    return;
  }
  if (_family == "wan") {
    // Deferred while the latents ask for the tiny decoder (FlashVSR):
    // loaded by the first one that does not.
    _wan_deferred = true;
    if (!_want_tiny) { ensure_wan_vae_(); }
    return;
  }

  genai::MetalKrea2Vae::Config cfg;   // Qwen-Image VAE defaults
  // Read the per-channel latent statistics (and z_dim / base_dim) from the
  // vae config.json; the un-whiten needs latents_mean / latents_std.
  //
  // NOT NECESSARILY THIS DIRECTORY'S. A standalone VAE ships weights and
  // nothing else, and when the model it serves is already installed, its
  // config is the authority and is already on disk -- so borrow it
  // rather than ask for a copy. resolve_vae_config_path owns that
  // choice; the checkpoint's own file still wins when it has one.
  const std::string cfg_path = resolve_vae_config_path(
      session(), _hf_dir, vae_dir, "AutoencoderKLQwenImage");
  if (!cfg_path.empty() &&
      cfg_path != (fs::path(vae_dir) / "config.json").string()) {
    // Said out loud, every time. Which sixteen numbers un-whitened a
    // latent is not something to infer silently: borrowed from the
    // wrong parent it is a colour grade nobody can trace back.
    load_note_(fmt(
        "VaeDecodeStage('{}'): '{}' carries no config.json; taking the "
        "latent statistics from '{}'", this->id(), vae_dir, cfg_path));
  }
  const bool have_config = !cfg_path.empty();
  {
    std::ifstream in(cfg_path);
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        // One reader for both VAE stages. It also picks up the keys
        // AutoencoderKLQwenImage21 needs -- dim_mult (whose LENGTH is
        // the level count), decoder_base_dim, is_residual and the
        // in/out channel counts -- which a per-stage copy of the
        // four-key version would have been missing on one side.
        genai::MetalKrea2Vae::config_from_json(fd, &cfg);
      }
    }
  }
  if ((int)cfg.latents_mean.size() != cfg.z_dim ||
      (int)cfg.latents_std.size() != cfg.z_dim) {
    // TWO DIFFERENT FAULTS, and the old message named neither. It
    // reported "0/0 of z_dim 16", which is what an ABSENT config.json
    // produces -- and a reader with no config at all looks for the
    // statistics inside a file that is not there. A STANDALONE VAE
    // checkpoint is exactly the case where the file can be missing on
    // its own (the weights are one safetensors and the config is
    // fetched beside them), so say which of the two happened and name
    // the path, because the fix for the first is to put a file there.
    if (!have_config) {
      session()->error(fmt(
          "VaeDecodeStage('{}'): no config.json at '{}' and no installed "
          "parent model to take one from, so the per-channel "
          "latents_mean / latents_std that un-whiten a latent are unknown "
          "and the stage is inert. Either register the model this VAE "
          "serves (a Krea-2 or Qwen-Image checkpoint, whose vae/config."
          "json is then used in place), or copy that file to "
          "'{}/config.json'", this->id(), vae_dir, vae_dir));
      return;
    }
    session()->error(fmt(
        "VaeDecodeStage('{}'): the config at '{}' carries {}/{} of the "
        "{} latents_mean / latents_std entries the un-whiten needs; the "
        "stage is inert", this->id(), cfg_path,
        cfg.latents_mean.size(), cfg.latents_std.size(), cfg.z_dim));
    return;
  }

  // What was built, as a warm hold's key: the weights, the config read,
  // the int8 settled (set_accel below).
  _vae_warm_key = fmt("{}|{}|{}", vae_dir, ws_dir, _accel.to_json())();
  if (_keep_loaded > 0 && !_unload_idle) {
    if (auto* mgr = session()->services()->generative_model_manager()) {
      if (auto warm = mgr->take_warm<genai::MetalKrea2Vae>(
              kKrea2VaeWarmKind, vae_dir_for_release_(), _vae_warm_key)) {
        _vae = std::move(warm);
        load_note_(fmt("VaeDecodeStage('{}'): the Qwen-Image VAE kept "
                       "warm is taken back", this->id()));
        return;
      }
    }
  }
  load_note_(fmt(
      "VaeDecodeStage('{}'): loading Qwen-Image VAE from '{}'", this->id(),
      vae_dir));
  _vae = genai::MetalKrea2Vae::load(ws, mc, cfg);
  if (!_vae) {
    session()->error(fmt(
        "VaeDecodeStage('{}'): failed to load the VAE from '{}'; inert",
        this->id(), vae_dir));
    return;
  }
  // i8_gemm, settled in the ctor. After the load: what it can take
  // depends on the attention member the load measured.
  _vae->set_accel(krea2_accel_(_accel));
  if (const std::string note = _vae->accel_note(); !note.empty()) {
    session()->log_normal(fmt("VaeDecodeStage('{}'): {}", this->id(), note));
  }
  session()->log_debug(fmt(
      "VaeDecodeStage('{}'): VAE ready from '{}' (z_dim {}, base_dim {}, "
      "num_res_blocks {})", this->id(), vae_dir, cfg.z_dim, cfg.base_dim,
      cfg.num_res_blocks));
}


namespace {

// Carry the generating model forward. generate-image stamps `model_name` onto
// the latent's sideband; the decoded image inherits it so a downstream
// save-image can record what produced the pixels. Anything else already on
// the latent's sideband is left behind -- it describes the LATENT, not the
// image -- and nothing is invented when the producer sent no name.
void
forward_model_name_(const TensorBeatPayload& src, TensorBeat& dst)
{
  provenance::carry_model_name(src.sideband, dst.sideband);
}

// BFL's single-reference edit generates on the reference's padded canvas
// and keeps only the picture: sideband::kCropBox on the latent (from
// image-resample `pad_to_grid`, through vae-encode and generate-image)
// says where it sits. Cut the decoded planar [C, H, W] canvas to it. A box
// that is not a rectangle inside the picture is left alone. True if cut.
// The same box as a rectangle of a W x H picture, for a caller that cuts
// as it converts. False (the whole picture) when there is no box or it is
// not a rectangle strictly inside the picture.
bool
crop_rect_(const TensorBeatPayload& lat, int W, int H, int* x0, int* y0,
           int* cw, int* ch)
{
  *x0 = 0; *y0 = 0; *cw = W; *ch = H;
  if (!lat.sideband.is_object()) { return false; }
  FlexData sb = lat.sideband;            // as_object() is a view: keep it
  auto so = sb.as_object();
  if (!so.contains(sideband::kCropBox)) { return false; }
  FlexData box = so.at(sideband::kCropBox);
  if (!box.is_array() || box.as_array().size() != 4) { return false; }
  int b[4];
  for (int i = 0; i < 4; ++i) {
    b[i] = (int)box.as_array()[(std::size_t)i].as_int(-1);
  }
  if (b[0] < 0 || b[1] < 0 || b[2] > W || b[3] > H || b[2] <= b[0] ||
      b[3] <= b[1] || (b[2] - b[0] == W && b[3] - b[1] == H)) {
    return false;
  }
  *x0 = b[0]; *y0 = b[1]; *cw = b[2] - b[0]; *ch = b[3] - b[1];
  return true;
}

bool
crop_to_box_(const TensorBeatPayload& lat,
             std::unique_ptr<TensorBeatPayload>& out)
{
  if (!lat.sideband.is_object() || out == nullptr ||
      out->shape.size() != 3) {
    return false;
  }
  FlexData sb = lat.sideband;            // as_object() is a view: keep it
  auto so = sb.as_object();
  if (!so.contains(sideband::kCropBox)) { return false; }
  FlexData box = so.at(sideband::kCropBox);
  if (!box.is_array() || box.as_array().size() != 4) { return false; }
  int b[4];
  for (int i = 0; i < 4; ++i) {
    b[i] = (int)box.as_array()[(std::size_t)i].as_int(-1);
  }
  const int C = (int)out->shape[0], H = (int)out->shape[1];
  const int W = (int)out->shape[2];
  if (b[0] < 0 || b[1] < 0 || b[2] > W || b[3] > H || b[2] <= b[0] ||
      b[3] <= b[1] || (b[2] - b[0] == W && b[3] - b[1] == H)) {
    return false;
  }
  const int cw = b[2] - b[0], ch = b[3] - b[1];
  const std::size_t esz = TensorBeat::byte_size_of(out->dtype);
  const auto src = out->materialize_contiguous();
  auto cut = std::make_unique<TensorBeatPayload>();
  cut->dtype = out->dtype;
  cut->shape = {C, ch, cw};
  cut->sideband = out->sideband;
  cut->resize_contiguous((std::size_t)C * ch * cw);
  for (int c = 0; c < C; ++c) {
    for (int y = 0; y < ch; ++y) {
      std::memcpy(cut->bytes_() + (((std::size_t)c * ch + y) * cw) * esz,
                  src.data() + (((std::size_t)c * H + b[1] + y) * W + b[0])
                                   * esz,
                  (std::size_t)cw * esz);
    }
  }
  out = std::move(cut);
  return true;
}

// One decoded sample, already mapped to 0..1, into a picture of either
// dtype: U8 rounded, F16 as it is -- both clamped to 0..1 (NaN to 0), as
// the reference's PIL conversion clamps.
inline void
put_unit_(TensorBeatPayload& out, std::size_t i, float unit)
{
  if (!(unit > 0.0f)) { unit = 0.0f; }
  if (unit > 1.0f) { unit = 1.0f; }
  if (out.dtype == TensorBeat::DType::F16) {
    reinterpret_cast<_Float16*>(out.bytes_())[i] = (_Float16)unit;
  } else {
    out.bytes_()[i] = (std::uint8_t)std::lround(unit * 255.0f);
  }
}

}  // namespace

#ifdef VPIPE_BUILD_APPLE_SILICON

std::unique_ptr<TensorBeatPayload>
VaeDecodeStage::held_beat_(const HeldFrame& h) const
{
  if (h.unit.size() != (std::size_t)3 * h.H * h.W || h.H <= 0) {
    return nullptr;
  }
  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = _out_dtype;
  out->shape = {3, h.H, h.W};
  out->resize_contiguous(h.unit.size());
  for (std::size_t i = 0; i < h.unit.size(); ++i) {
    put_unit_(*out, i, h.unit[i]);
  }
  out->sideband = h.sb;
  return out;
}

std::unique_ptr<TensorBeatPayload>
VaeDecodeStage::begin_clip_(RuntimeContext& ctx, int F, int H, int W) const
{
  // Nothing wired, nothing built. Returning null rather than a flag
  // means no caller can hold a flag that disagrees with a pointer.
  if (!(ctx.num_oports() > 1 && ctx.has_consumers(1))) { return nullptr; }
  if (F <= 0 || H <= 0 || W <= 0) { return nullptr; }
  auto clip = std::make_unique<TensorBeatPayload>();
  clip->dtype = _out_dtype;  // the frames' own
  clip->shape = {(std::int64_t)F, 3, (std::int64_t)H, (std::int64_t)W};
  clip->resize_contiguous((std::size_t)F * 3 * (std::size_t)H * W);
  return clip;
}

void
VaeDecodeStage::add_to_clip_(TensorBeatPayload* clip, int f,
                             const TensorBeatPayload& frame)
{
  if (clip == nullptr) { return; }
  const std::size_t per = frame.byte_size();
  if (per == 0) { return; }
  const std::size_t off = (std::size_t)f * per;
  // Bounds-checked rather than trusted: F comes from the decoder and
  // the frame geometry from the family, and a clip built from a
  // disagreement between them would be a heap overwrite, not a bad
  // picture.
  if (off + per > clip->byte_size()) { return; }
  std::memcpy(clip->bytes_() + off, frame.bytes_(), per);
}

void
VaeDecodeStage::finish_clip_(TensorBeatPayload* clip, double fps,
                             const TensorBeatPayload& src) const
{
  if (clip == nullptr) { return; }
  // The same keys temporal-stack puts on a stacked video group, so a
  // consumer cannot tell where its clip was built. `frames` is read off
  // the leading extent rather than being a second opinion about it.
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign(
      sideband::kFrames,
      FlexData::make_int(clip->shape.empty() ? 0 : clip->shape[0]));
  sb.as_object().insert_or_assign(sideband::kFps, FlexData::make_real(fps));
  clip->sideband = std::move(sb);
  forward_model_name_(src, *clip);
}

#endif

Job
VaeDecodeStage::process(RuntimeContext& ctx)
{
  // Latch the shared model (iport1) once -- a model-select source overrides the
  // hf_dir config -- then lazily load the VAE before the first decode.
  if (!_model_latched && ctx.num_iports() > kModelPort &&
      ctx.iport_connected(kModelPort)) {
    auto mb = co_await ctx.read(kModelPort);
    _model_latched = true;
    if (const auto* mfd =
            mb ? dynamic_cast<const FlexDataPayload*>(mb.get()) : nullptr) {
      // Record WHICH model, but do not load it yet. The model-select beat
      // arrives at startup, and loading here would hold the VAE resident
      // through everything upstream of the first latent -- which for a
      // video graph is the entire denoise. That is not a rounding error:
      // MiniMax-H3's video VAE is 10.4 GB, and holding it through a 33B
      // denoise on a 16 GB box left the DiT with ZERO Metal working-set
      // headroom and its forward was refused. Nothing needs a VAE before
      // there is a latent to decode, so the load waits for one.
      apply_model_select_beat(mfd->data, _hf_dir);
    }
  }
  auto in = co_await ctx.read(0);
  if (!in) {                                    // upstream EOS -> close oport
    // Frames held for a cross-fade that never came go out as they are.
    for (const HeldFrame& h : _held) {
      if (auto b = held_beat_(h)) {
        ++_images_emitted;
        co_await ctx.write(0, std::move(b));
      }
    }
    _held.clear();
    ctx.signal_done();
    co_return;
  }
  // A latent its model's TINY decoder decodes needs no VAE weights: read
  // its ask before anything loads, so they wait for a latent that does.
  _want_tiny = false;
  if (const auto* lt = dynamic_cast<const TensorBeatPayload*>(in.get());
      lt != nullptr && lt->sideband.is_object()) {
    FlexData sb = lt->sideband;          // as_object() is a view: keep it
    auto o = sb.as_object();
    _want_tiny = o.contains(sideband::kDecoder) &&
                 o.at(sideband::kDecoder).as_string() == "tiny";
  }
  // NOW there is work for it. First latent => first load (the model-select
  // branch above deliberately only recorded the directory); a later one =>
  // reload if a previous beat dropped the VAE to leave the denoise room.
  ensure_loaded_();
  if (_unloaded) { reload_vae_(); }
  const auto* tbp = dynamic_cast<const TensorBeatPayload*>(in.get());
  // A video latent carries a time axis, so the wan family arrives 4-D
  // ([z, T, H/8, W/8]) where every image VAE here is 3-D.
  if (tbp == nullptr || tbp->dtype != TensorBeat::DType::F32 ||
      (tbp->shape.size() != 3 && tbp->shape.size() != 4)) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): expected an f32 [z,H/8,W/8] latent TensorBeat "
        "(or [z,T,H/8,W/8] for a video VAE), got {}; skipping", this->id(),
        in->describe()));
    co_return;
  }
  // ---- Content-policy refusal ------------------------------------------
  // A prompt a family's content screen refused never reaches a DiT, so
  // there is no latent to decode -- the upstream beat is a marker carrying
  // only the intended image size. Paint the reference's refusal image (a
  // plain white frame: no text, no category, nothing that would tell a
  // prober WHICH rule they hit) and emit it in the generated image's place.
  // No VAE needed, so this runs even when the VAE never loaded.
  if (tbp->sideband.is_object()) {
    FlexData sb = tbp->sideband;          // as_object() is a view: keep it
    auto o = sb.as_object();
    if (o.contains(sideband::kContentBlocked)
        && o.at(sideband::kContentBlocked).as_bool(false)) {
      const int H = o.contains(sideband::kRefusalHeight)
                        ? (int)o.at(sideband::kRefusalHeight).as_int(0) : 0;
      const int W = o.contains(sideband::kRefusalWidth)
                        ? (int)o.at(sideband::kRefusalWidth).as_int(0) : 0;
      if (H <= 0 || W <= 0) {
        session()->warn(fmt(
            "VaeDecodeStage('{}'): refusal beat carries no image size; "
            "dropping", this->id()));
        co_return;
      }
      auto out = std::make_unique<TensorBeatPayload>();
      out->dtype = _out_dtype;
      out->shape = {3, H, W};
      const std::size_t n = (std::size_t)3 * H * W;
      out->resize_contiguous(n);
      for (std::size_t i = 0; i < n; ++i) { put_unit_(*out, i, 1.0f); }
      ++_images_emitted;
      session()->info(fmt(
          "VaeDecodeStage('{}'): content-policy refusal -> blank {}x{} image",
          this->id(), W, H));
      forward_model_name_(*tbp, *out);
      if (_unload_idle) { unload_vae_(); }
      co_await ctx.write(0, std::move(out));
      co_return;
    }
  }

  auto* mc = session()->services()->metal_compute();

  // ---- 2D AutoencoderKL (FLUX.2 / the plain FLUX.1 VAE): input
  // [dit_channels, H/px, W/px] with px = 8*patch; decode directly (the
  // patch + whitening is internal), no per-channel un-whiten. ----
  if (_family == "flux2") {
    if (!_flux2_vae) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): FLUX.2 VAE not loaded; skipping", this->id()));
      co_return;
    }
    const int Cdit = (int)tbp->shape[0];
    const int h16 = (int)tbp->shape[1];
    const int w16 = (int)tbp->shape[2];
    if (Cdit != _flux2_vae->config().dit_channels() || h16 <= 0 || w16 <= 0) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): latent [{}, {}, {}] does not match "
          "dit_channels {}; skipping", this->id(), Cdit, h16, w16,
          _flux2_vae->config().dit_channels()));
      co_return;
    }
    // Pixels per latent cell: the conv trunk is 8x, times the VAE's patch
    // factor (2 on AutoencoderKLFlux2, 1 on the plain AutoencoderKL).
    const int px = 8 * _flux2_vae->config().patch;
    // Publish the arena from the geometry actually in hand. The image
    // branches need this as much as the video ones: an EDIT graph takes
    // its size from the reference image, so generate-image had no
    // config geometry to declare from -- and Qwen-Image-Edit has no
    // free_*_dit_for_decode_ at all, so nothing upstream states it
    // either. Same estimator the plan used, at the real pixel size.
    publish_image_arena_(w16 * px, h16 * px);
    const std::size_t nz = (std::size_t)Cdit * h16 * w16;
    metal_compute::SharedBuffer z = mc->make_shared_buffer(nz * 2);
    { auto* d = static_cast<_Float16*>(z.contents());
      const float* s = tbp->as_f32();
      for (std::size_t i = 0; i < nz; ++i) { d[i] = (_Float16)s[i]; } }
    std::string derr;
    metal_compute::SharedBuffer rgb;
    {   // LLM-lane perf event: one VAE decode per image (value = pixel count).
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae,
                         kPerfLlmVaeBegin,
                         (std::uint64_t)(h16 * px) * (w16 * px));
      rgb = _flux2_vae->decode(z, h16, w16, &derr);
    }
    if (rgb.empty()) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): AutoencoderKL decode failed ({}); skipping",
          this->id(), derr.empty() ? "unknown error" : derr));
      co_return;
    }
    const int H = h16 * px, W = w16 * px;
    const std::size_t n = (std::size_t)3 * H * W;
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = _out_dtype;
    out->shape = {3, H, W};
    out->resize_contiguous(n);
    const auto* rp = static_cast<const _Float16*>(rgb.contents());
    for (std::size_t i = 0; i < n; ++i) {
      put_unit_(*out, i, ((float)rp[i] + 1.0f) * 0.5f);
    }
    ++_images_emitted;
    session()->log_debug(fmt(
        "VaeDecodeStage('{}'): AutoencoderKL decoded + emitted image #{} "
        "[3, {}, {}]",
        this->id(), _images_emitted, H, W));
    if (crop_to_box_(*tbp, out)) {
      session()->info(fmt(
          "VaeDecodeStage('{}'): cut the {}x{} canvas back to the "
          "reference's {}x{}", this->id(), W, H, out->shape[2],
          out->shape[1]));
    }
    forward_model_name_(*tbp, *out);
    if (_unload_idle) { unload_vae_(); }
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Wan video VAE: input [z_dim, T, h8, w8]; decode to
  // ---- MiniMax-H3: a ViT decoder over a 16x-compressed latent --------
  if (_family == "minimax-h3") {
    if (!_h3_vae && !_h3_fast) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): MiniMax-H3 VAE not loaded; skipping",
          this->id()));
      co_return;
    }
    if (tbp->shape.size() != 4) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): the MiniMax-H3 VAE decodes a VIDEO latent "
          "[z, T, H/16, W/16]; got a {}-D tensor. Skipping", this->id(),
          tbp->shape.size()));
      co_return;
    }
    // The geometry from whichever decoder is loaded: the fast one takes
    // the released VAE's.
    const int vz = _h3_fast ? _h3_fast->config().z_channels
                            : _h3_vae->config().z_channels;
    const int vp = _h3_fast ? _h3_fast->config().patch
                            : _h3_vae->config().patch;
    const int vpt = _h3_fast ? _h3_fast->config().patch_t
                             : _h3_vae->config().patch_t;
    const int Cz = (int)tbp->shape[0];
    const int LT = (int)tbp->shape[1];
    const int lh = (int)tbp->shape[2];
    const int lw = (int)tbp->shape[3];
    if (Cz != vz || LT <= 0 || lh <= 0 || lw <= 0) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): latent [{}, {}, {}, {}] does not match "
          "z_channels {}; skipping", this->id(), Cz, LT, lh, lw,
          vz));
      co_return;
    }

    // ONE latent frame is a PICTURE, and it is the one length the
    // chunked video decoder cannot read: `token_drop` spends the first
    // chunk priming the temporal window, so `decode_video` needs 7
    // latent frames and refuses 1 -- while the encoder turns a still
    // into exactly 1. Only a checkpoint whose decoder was fine-tuned for
    // it can decode this, and it declares that itself.
    const bool one_frame = (LT == 1);
    if (one_frame && _h3_fast) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): this is a single-frame latent (a picture), "
          "and the fast decoder decodes clips; unset fast_decoder for it. "
          "Skipping", this->id()));
      co_return;
    }
    if (one_frame && !_h3_vae->can_decode_image()) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): this is a single-frame latent and the "
          "loaded MiniMax-H3 VAE does not decode one -- its metadata "
          "declares no 'h3_t1_direct'. Point hf_dir at an image-trained H3 "
          "VAE (Mamad8/MiniMax-H3-Image-VAE); skipping", this->id()));
      co_return;
    }

    double fps = _fps;
    if (tbp->sideband.is_object()) {
      FlexData sb = tbp->sideband;        // as_object() is a view: keep it
      auto o = sb.as_object();
      if (o.contains(sideband::kFps) &&
          o.at(sideband::kFps).as_real(0.0) > 0.0) {
        fps = o.at(sideband::kFps).as_real(fps);
      }
    }

    // Un-whiten on the way in. The DiT generates in NORMALIZED latent
    // space, so decoding the beat as-is would feed the VAE a latent with
    // the wrong scale AND the wrong per-channel offset -- which decodes
    // to a plausible but washed-out clip rather than to obvious noise.
    const std::size_t nz = (std::size_t)Cz * LT * lh * lw;
    const std::size_t vox = (std::size_t)LT * lh * lw;
    // The fast decoder takes the whitened f32 latent as it is: it folds
    // the un-whitening into its embedding.
    metal_compute::SharedBuffer z;
    if (!_h3_fast) {
      z = mc->make_shared_buffer(nz * 2);
      if (z.empty()) { co_return; }
      const auto& vc = _h3_vae->config();
      auto* d = static_cast<std::uint16_t*>(z.contents());
      const float* s2 = tbp->as_f32();
      const bool whiten = (int)vc.latents_mean.size() == Cz &&
                          (int)vc.latents_std.size() == Cz;
      for (int c = 0; c < Cz; ++c) {
        const float mu = whiten ? vc.latents_mean[(std::size_t)c] : 0.0f;
        const float sd = whiten ? vc.latents_std[(std::size_t)c] : 1.0f;
        for (std::size_t i = 0; i < vox; ++i) {
          const std::size_t k = (std::size_t)c * vox + i;
          const float v = s2[k] * sd + mu;
          std::uint32_t u;
          std::memcpy(&u, &v, 4);
          d[k] = (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
        }
      }
    }

    // Preflight the decode's OUTPUT before allocating it, and park to
    // make room if it does not fit. A video decode's output is not the
    // few MB an image decode's is: [3, frames, H, W] at bf16 is 175 MB
    // for a 2.3 s 960x544 clip and grows linearly with length and area,
    // and this stage then buffers the whole clip again as planar U8
    // because ctx.write is a coroutine and the VAE's frame sink is a
    // plain callback.
    //
    // SCOPE, stated rather than implied: this counts the two buffers
    // whose size is known EXACTLY here. It does NOT include the VAE's
    // per-chunk internal working set, which is real and unmeasured on
    // this path -- the FLUX.2 decode OOM was exactly a peak estimate
    // that missed the internal term, so this is deliberately called a
    // floor and not a peak. Measure decode_video's high-water mark and
    // fold it in before treating a pass here as a guarantee.
    {
      // `decoded_frames` answers 0 for a single latent frame -- that is
      // the refusal, not a size -- so the image path states its own 1
      // rather than preflighting an allocation of nothing.
      const std::size_t out_frames =
          one_frame ? 1u
          : _h3_fast ? (std::size_t)_h3_fast->decoded_frames(LT)
                     : (std::size_t)_h3_vae->decoded_frames(LT);
      const std::size_t px = (std::size_t)(lh * vp) *
                             (std::size_t)(lw * vp) * out_frames;
      // The fast decoder's working set grows with the clip -- its f32
      // stream holds every latent cell -- and is exact, its output
      // included.
      // The released decoder's clip (bf16) and the chunk it works in
      // beside it -- its tiles, their stitch, its ViT scratch.
      const std::size_t need =
          (_h3_fast ? _h3_fast->scratch_bytes(LT, lh, lw)
           : one_frame
               ? px * 3 * 2
               : genai::MetalMiniMaxH3VideoVae::decode_video_bytes(
                     _h3_vae->config(), LT, lh, lw))
          + px * 3;                    // the stage's planar-U8 clip
      // PHYSICAL pressure is the gate; the Metal working set is only
      // advisory here, and conflating the two refused work that fits.
      //
      // This check necessarily runs AFTER the VAE has loaded -- the frame
      // count comes from the model -- so `headroom` already has the VAE's
      // own 10.4 GB subtracted from it. Testing fits() there asks "is
      // there room for the output BESIDES everything, including the thing
      // I just legitimately loaded to do this work", and the answer on a
      // 16 GB box is no: it refused a 251 MB decode with 5.1 GB of
      // physically reclaimable RAM sitting free. That is a false negative
      // that blocks a working pipeline.
      //
      // What actually took the machine down was PHYSICAL exhaustion --
      // free 14 MB, file cache drained to 2.7 MB, pageout unable to make
      // progress -- not a soft working-set overrun. recommendedMaxWorking
      // SetSize is advisory on UMA, and the documented GPU OOM (code 8)
      // that fits() exists for was a 12-16 GB single command buffer, three
      // orders off a quarter-gigabyte output. So: refuse on physical, warn
      // on working set, and let a small allocation past the soft limit.
      // Publish it. `need` is a function of the BEAT -- it scales with
      // out_frames -- so it is the one figure the plan could not have
      // held: a config-derived estimate has no frame term and cannot
      // acquire one. Revising here is what makes the ledger describe
      // this clip rather than a guess about clips in general, and it is
      // the same declare-then-revise the streaming DiTs use for weights.
      revise_decode_arena_(need);

      auto* mcb = session()->services()->metal_compute();
      auto mb = mcb ? mcb->memory_budget() : metal_compute::MetalCompute::
                                                 MemoryBudget{};
      if (mb.recommended != 0 && !mb.fits_physical(need)) {
        std::size_t parked = 0;
        if (auto* gm = session()->services()->generative_model_manager()) {
          parked = gm->reclaim_at_least(need);
        }
        mb = mcb->memory_budget();
        if (mb.fits_physical(need)) {
          session()->info(fmt(
              "VaeDecodeStage('{}'): parked ~{} MB to fit a {}-frame "
              "{}x{} decode (~{} MB of output)", this->id(), parked >> 20,
              out_frames, lw * vp, lh * vp, need >> 20));
        } else {
          // For the host to act on (common/host-report.h).
          FlexData d = FlexData::make_object();
          auto o = d.as_object();
          auto u = [](std::size_t b) { return FlexData::make_uint(b); };
          o.insert("stage", FlexData::make_string(this->id()));
          o.insert("step", FlexData::make_string("decode"));
          o.insert("frames", u(out_frames));
          o.insert("width", u((std::size_t)(lw * vp)));
          o.insert("height", u((std::size_t)(lh * vp)));
          o.insert("need", u(need));
          FlexData parts = FlexData::make_array();
          for (const auto& [name, b] :
               {std::pair<const char*, std::size_t>{"decoded_frames",
                                                    px * 3 * 2},
                {"frames_u8", px * 3}}) {
            FlexData e = FlexData::make_object();
            e.as_object().insert("name", FlexData::make_string(name));
            e.as_object().insert("bytes", u(b));
            parts.as_array().push_back(std::move(e));
          }
          o.insert("parts", std::move(parts));
          FlexData gates = FlexData::make_array();
          {
            FlexData e = FlexData::make_object();
            auto eo = e.as_object();
            eo.insert("name", FlexData::make_string("reclaimable_ram"));
            eo.insert("need", u(need));
            eo.insert("have", u(mb.available_physical));
            eo.insert("ok", FlexData::make_bool(false));
            gates.as_array().push_back(std::move(e));
          }
          o.insert("gates", std::move(gates));
          o.insert("parked", u(parked));
          host_report(session(), "memory", std::move(d));
          session()->error(fmt(
              "VaeDecodeStage('{}'): not enough memory to decode {} frames "
              "at {}x{} -- the output alone is ~{} MB and only ~{} MB is "
              "reclaimable{}. Refusing rather than thrashing: wired Metal "
              "buffers cannot be paged out, so overcommitting takes the "
              "machine down rather than failing this stage",
              this->id(), out_frames, lw * vp, lh * vp,
              need >> 20, mb.available_physical >> 20,
              parked > 0 ? fmt(" after parking ~{} MB", parked >> 20)()
                         : std::string()));
          co_return;
        }
      } else if (mb.recommended != 0 && !mb.fits(need)) {
        session()->log_debug(fmt(
            "VaeDecodeStage('{}'): {}-frame decode needs ~{} MB with ~{} MB "
            "of Metal working set left but ~{} MB physically reclaimable -- "
            "proceeding; the working-set figure is advisory on UMA",
            this->id(), out_frames, need >> 20, mb.headroom >> 20,
            mb.available_physical >> 20));
      }
    }

    int F = 0;
    std::string derr;
    metal_compute::SharedBuffer rgb;
    {
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae,
                         kPerfLlmVaeBegin,
                         (std::uint64_t)LT * vpt *
                             (std::uint64_t)(lh * vp) *
                             (std::uint64_t)(lw * vp));
      // A minute of a 2.4B ViT with nothing else to look at. The tile is
      // the only unit the decode passes through often enough to be worth
      // reporting, and it is what the bar counts.
      UiProgress bar = session()->open_progress("vae decode");
      if (_h3_fast) {
        // A layer at a time over the whole clip; a stop lands between
        // layers (each commits and waits).
        _h3_fast->set_progress([&bar](int done, int total) {
          bar.update((std::uint64_t)done, (std::uint64_t)total);
        });
        _h3_fast->set_stop([&ctx]() { return ctx.stop_requested(); });
        rgb = _h3_fast->decode(tbp->as_f32(), LT, lh, lw, &derr);
        F = rgb.empty() ? 0 : _h3_fast->decoded_frames(LT);
        _h3_fast->set_progress(nullptr);
        _h3_fast->set_stop(nullptr);
      } else {
        _h3_vae->set_tile_progress([&bar](int done, int total) {
          bar.update((std::uint64_t)(done < 0 ? 0 : done),
                     (std::uint64_t)(total < 0 ? 0 : total));
        });
        // The two agree on memory: `decode_video` emits channel-first
        // [3, F, H, W] and `decode_image` [3, H, W], which at F == 1 is
        // the same buffer. So only the call differs and every line below
        // -- the u8 quantize, the per-frame beats, the diagnostics -- is
        // the video path's, unchanged.
        if (one_frame) {
          rgb = _h3_vae->decode_image(z, lh, lw, &derr);
          F = rgb.empty() ? 0 : 1;
        } else {
          rgb = _h3_vae->decode_video(z, LT, lh, lw, &F, &derr);
        }
        // Cleared before the bar it captures leaves scope -- the VAE
        // outlives this call and would otherwise hold a dangling reference
        // into the next decode.
        _h3_vae->set_tile_progress(nullptr);
      }
      bar.finish();
      // Set the module booking to what is HELD after this decode: the
      // module's bytes when the tier armed, 0 when it declined. Both ways,
      // every clip, so the ledger follows a reload that arms after one that
      // did not.
      const bool tried = _h3_fast ? _h3_fast->ane_attempted()
                                  : _h3_vae->ane_attempted();
      const bool armed = _h3_fast ? _h3_fast->ane_armed()
                                  : _h3_vae->ane_armed();
      if (tried) {
        if (!armed) { _ane_bytes = 0; }
        if (auto* mgr = session()->services()->generative_model_manager()) {
          mgr->revise_scratch("coreml:" + ane_claim_label_(), _ane_bytes);
        }
      }
    }
    if (rgb.empty() || F <= 0) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): MiniMax-H3 video decode failed ({}); "
          "skipping", this->id(), derr.empty() ? "unknown error" : derr));
      co_return;
    }
    const int H = lh * vp, W = lw * vp;
    const auto* rp = static_cast<const std::uint16_t*>(rgb.contents());
    const std::size_t plane = (std::size_t)H * W;
    // Diagnostic: the decoder's own pixels, before the u8 quantize and
    // before anything downstream can reshape them. The unit tests cover
    // decode_video() on an ENCODER latent; the un-whiten that runs ahead
    // of it here is covered by nothing, so this is the only way to see
    // which side of the stage a bad frame came from.
    if (const char* rd = std::getenv("VPIPE_H3_RGB_DUMP")) {
      std::ofstream f(rd, std::ios::binary);
      for (std::size_t i = 0; i < (std::size_t)3 * F * plane; ++i) {
        const std::uint32_t u = (std::uint32_t)rp[i] << 16;
        float x;
        std::memcpy(&x, &u, 4);
        f.write(reinterpret_cast<const char*>(&x), 4);
      }
      session()->info(fmt("VaeDecodeStage('{}'): dumped rgb [3, {}, {}, {}] "
                          "to {}", this->id(), F, H, W, rd));
    }
    if (_unload_idle) { unload_vae_(); }
    // The clip, when a consumer asked for one. Filled frame by frame
    // from the beats below rather than from a second pass over `rgb`,
    // so the conversion lives in exactly one place and the two ports
    // cannot drift apart in what they show.
    // NOT for a picture. oport1's contract is a CLIP, and this port's
    // documentation already draws the line: emitting [1, 3, H, W] for a
    // still would invent a one-frame clip where MiniMax-H3 itself makes
    // a real distinction between the two -- a reference image advances
    // its rotary clock by one integer slot, a one-frame video does not.
    // How many of the decoded frames are real (sideband::kValidFrames: a
    // refine of a 121-frame draft decodes 124), and the rectangle the
    // picture is cut to (sideband::kCropBox: a 1080-row target computes
    // at 1088). Both absent on a generation, which emits every frame
    // whole as it always has.
    int Fo = F;
    if (tbp->sideband.is_object()) {
      FlexData vsb = tbp->sideband;       // as_object() is a view: keep it
      auto vo = vsb.as_object();
      if (vo.contains(sideband::kValidFrames)) {
        const auto v = vo.at(sideband::kValidFrames).as_int(-1);
        if (v > 0 && v < F) { Fo = (int)v; }
      }
    }
    int cx0 = 0, cy0 = 0, cw = W, chh = H;
    const bool cropped = crop_rect_(*tbp, W, H, &cx0, &cy0, &cw, &chh);
    if ((Fo != F || cropped) && !_h3_cut_said) {
      _h3_cut_said = true;
      session()->info(fmt(
          "VaeDecodeStage('{}'): keeping {} of {} decoded frames{}",
          this->id(), Fo, F,
          cropped ? fmt(", cut to {}x{} at ({}, {}) of {}x{}", cw, chh, cx0,
                        cy0, W, H)()
                  : std::string()));
    }
    const std::size_t oplane = (std::size_t)cw * chh;
    auto clip = one_frame ? nullptr : begin_clip_(ctx, Fo, chh, cw);
    for (int f = 0; f < Fo; ++f) {
      auto out = std::make_unique<TensorBeatPayload>();
      out->dtype = _out_dtype;
      out->shape = {3, chh, cw};
      out->resize_contiguous(3 * oplane);
      for (int c = 0; c < 3; ++c) {
        const std::uint16_t* src = rp + ((std::size_t)c * F + f) * plane;
        const std::size_t base = (std::size_t)c * oplane;
        // The ViT decoder emits IMAGENET-NORMALIZED pixels, not [-1, 1]:
        // the encoder's first act is `(x + 1)/2` then `(. - mean)/std`,
        // and the reference decode undoes exactly that before clamping.
        // Treating its output as [-1, 1] leaves it ~1/std = 4.4x too
        // wide, so the u8 conversion clips most of the frame -- and
        // because each 16x16 token block clips against its own local
        // statistics, the result is a BLOCK GRID rather than an
        // obviously blown-out image.
        const float pm = kImagenetMean[c], ps = kImagenetStd[c];
        for (int y = 0; y < chh; ++y) {
          const std::uint16_t* srow =
              src + (std::size_t)(cy0 + y) * W + (std::size_t)cx0;
          const std::size_t drow = base + (std::size_t)y * cw;
          for (int x0 = 0; x0 < cw; ++x0) {
            const std::uint32_t u = (std::uint32_t)srow[x0] << 16;
            float x;
            std::memcpy(&x, &u, 4);
            put_unit_(*out, drow + (std::size_t)x0, x * ps + pm);  // [0, 1]
          }
        }
      }
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign(sideband::kFrame,
                                      FlexData::make_int((std::int64_t)f));
      sb.as_object().insert_or_assign(sideband::kFrames,
                                      FlexData::make_int((std::int64_t)Fo));
      sb.as_object().insert_or_assign(sideband::kFps, FlexData::make_real(fps));
      out->sideband = std::move(sb);
      forward_model_name_(*tbp, *out);
      add_to_clip_(clip.get(), f, *out);
      ++_images_emitted;
      co_await ctx.write(0, std::move(out));
    }
    if (clip) {
      finish_clip_(clip.get(), fps, *tbp);
      auto payload = std::move(clip);
      co_await ctx.write(1, std::move(payload));
    }
    session()->log_debug(fmt(
        "VaeDecodeStage('{}'): decoded [{}, {}, {}, {}] -> {} frames "
        "[3, {}, {}] @ {:.3f} fps", this->id(), Cz, LT, lh, lw, F, H, W, fps));
    co_return;
  }

  // ---- an out-of-tree family (VaeModelRegistry) ------------------------
  //
  // Guarded on the POINTER, and placed before every built-in branch: a
  // family that tagged itself like a built-in still cannot fall through
  // into one. The family owns un-whitening, tiling and colour space; the
  // U8 quantisation, the per-frame beat and the sideband stay here so
  // there is exactly one place pixels are made.
  if (_vae_family != nullptr) {
    if (!_plugin_dec) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): VAE family '{}' has no decoder loaded; "
          "skipping", this->id(), _family));
      co_return;
    }
    const bool is_video = tbp->shape.size() == 4;
    const int Cz = (int)tbp->shape[0];
    const int T  = is_video ? (int)tbp->shape[1] : 1;
    if (Cz != _plugin_dec->latent_channels()) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): the latent has {} channels but family '{}' "
          "wants {}; skipping", this->id(), Cz, _family,
          _plugin_dec->latent_channels()));
      co_return;
    }
    // The producer knows the clip's rate; the config is the fallback for
    // a latent that arrived from somewhere else (a file, a test).
    double fps = _fps;
    if (tbp->sideband.is_object()) {
      FlexData sb = tbp->sideband;        // as_object() is a view: keep it
      auto o = sb.as_object();
      if (o.contains(sideband::kFps) &&
          o.at(sideband::kFps).as_real(0.0) > 0.0) {
        fps = o.at(sideband::kFps).as_real(fps);
      }
    }
    // The bar the family's `progress` feeds. Declared BEFORE the request
    // so it outlives the callback that captures it.
    //
    // OPENED LAZILY, on the first report. `progress` is optional -- a
    // family may never call it -- and a bar opened up front would then
    // sit at zero for the whole decode, which reads worse than the
    // silence it replaced. A default-constructed UiProgress is inert, so
    // a family that stays quiet costs nothing and shows nothing.
    UiProgress bar;
    bool bar_open = false;
    genai::VaeDecodeRequest req;
    req.accel = &_accel;
    // ALWAYS INSTALLED, even though nothing is named yet: a
    // family that had to guard every lookup would eventually
    // forget one, and calling an empty std::function throws.
    req.input =
        [](std::string_view, genai::NamedTensor*) { return false; };

    req.latent = tbp->as_f32();
    req.shape.reserve(tbp->shape.size());
    for (std::int64_t d : tbp->shape) { req.shape.push_back((int)d); }
    req.fps      = fps;
    req.sideband = &tbp->sideband;
    // BOTH of the things this callback is for. It was wired as a
    // cancellation predicate alone, with `done` and `total` discarded --
    // so a registered family had no way to publish progress at all, and
    // the only "vae decode" bar in the tree was the built-in MiniMax-H3
    // one driven by a method the plugin interface does not have. A VAE
    // decode is tens of seconds to a minute at production geometry; that
    // is exactly the wait a bar exists for, and there was no reason it
    // should have depended on which side of the registry the model sat.
    req.progress = [&ctx, &bar, &bar_open, this](int done, int total) {
      if (!bar_open) {
        bar_open = true;
        bar = session()->open_progress("vae decode");
      }
      bar.update(done < 0 ? 0 : (std::uint64_t)done,
                 total < 0 ? 0 : (std::uint64_t)total);
      return !ctx.stop_requested();
    };

    // The frames are collected and written AFTERWARDS for the reason the
    // wan branch gives above: ctx.write is a coroutine and the sink is a
    // plain callback. What is held is the clip as planar U8, a third of
    // the f32 the family handed over.
    std::vector<std::unique_ptr<TensorBeatPayload>> frames;
    int seen = 0;
    auto sink = [&](const genai::VaeFrameChunk& c) -> bool {
      if (c.rgb == nullptr || c.n <= 0 || c.height <= 0 || c.width <= 0) {
        return false;
      }
      const std::size_t plane = (std::size_t)c.height * c.width;
      for (int k = 0; k < c.n; ++k) {
        auto out = std::make_unique<TensorBeatPayload>();
        out->dtype = _out_dtype;
        out->shape = {3, c.height, c.width};
        out->resize_contiguous((std::size_t)3 * plane);
        for (int ch = 0; ch < 3; ++ch) {
          // A family with fewer than 3 channels comes out greyscale
          // rather than read past the end of its own chunk.
          const int sc = (ch < c.channels) ? ch : (c.channels - 1);
          const float* src =
              c.rgb + ((std::size_t)sc * c.n + k) * plane;
          const std::size_t base = (std::size_t)ch * plane;
          for (std::size_t i = 0; i < plane; ++i) {
            put_unit_(*out, base + i, (src[i] + 1.0f) * 0.5f);
          }
        }
        // Video stamps {frame, frames, fps}; an image VAE stamps nothing,
        // which is the split the built-in branches already draw.
        if (is_video) {
          FlexData sb = FlexData::make_object();
          sb.as_object().insert_or_assign(
              sideband::kFrame,
              FlexData::make_int((std::int64_t)(c.frame0 + k)));
          sb.as_object().insert_or_assign(
              sideband::kFrames,
              FlexData::make_int((std::int64_t)c.frames_total));
          sb.as_object().insert_or_assign(sideband::kFps,
                                          FlexData::make_real(fps));
          out->sideband = std::move(sb);
        } else {
          // A registered family's picture is cut back like a built-in
          // one's (sideband::kCropBox): the key promises it of
          // vae-decode, whichever VAE ran -- the family sees the box in
          // req.sideband but has no reason to know what it means.
          crop_to_box_(*tbp, out);
        }
        forward_model_name_(*tbp, *out);
        frames.push_back(std::move(out));
        ++seen;
      }
      return !ctx.stop_requested();
    };
    const int want = is_video ? _plugin_dec->decoded_frames(T) : 1;
    // Same publication the built-in video branch makes, and needed more
    // here: a registered family's frame expansion is entirely its own
    // (LTX-2.5 compresses time by 8), so the plan's config-derived bound
    // is the furthest from the truth exactly where it cannot be checked.
    {
      const int sc = _plugin_dec->spatial_compression();
      const std::size_t px =
          (std::size_t)((int)tbp->shape[tbp->shape.size() - 2] * sc) *
          (std::size_t)((int)tbp->shape[tbp->shape.size() - 1] * sc) *
          (std::size_t)(want > 0 ? want : 1);
      revise_decode_arena_(px * 3 * 2 + px * 3);
    }
    std::string derr;
    bool ok = false;
    try {
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae,
                         kPerfLlmVaeBegin, (std::uint64_t)want);
      ok = _plugin_dec->decode(req, sink, &derr);
      // Closed here rather than at scope exit: what follows is the frame
      // write-out, and a bar left open across it reports the decode as
      // still running when it is not.
      bar.finish();
    } catch (const std::exception& e) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): VAE family '{}' threw decoding: {}; "
          "skipping", this->id(), _family, e.what()));
      co_return;
    } catch (...) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): VAE family '{}' threw a non-standard "
          "exception decoding; skipping", this->id(), _family));
      co_return;
    }
    if (!ok) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): decode failed ({}); skipping", this->id(),
          derr.empty() ? "unknown error" : derr));
      co_return;
    }
    session()->log_debug(fmt(
        "VaeDecodeStage('{}'): family '{}' decoded {} -> {} frames",
        this->id(), _family, tbp->describe(), seen));
    if (_unload_idle) { unload_vae_(); }
    // Only a VIDEO latent gets a clip: emitting [1, 3, H, W] for a still
    // would invent a one-frame CLIP, which is a different request from a
    // picture to every consumer that reads references.
    auto clip = (is_video && !frames.empty() && frames[0] != nullptr
                 && frames[0]->shape.size() == 3)
                    ? begin_clip_(ctx, (int)frames.size(),
                                  (int)frames[0]->shape[1],
                                  (int)frames[0]->shape[2])
                    : nullptr;
    int clip_at = 0;
    for (auto& f : frames) {
      add_to_clip_(clip.get(), clip_at++, *f);
      ++_images_emitted;
      co_await ctx.write(0, std::move(f));
    }
    if (clip) {
      finish_clip_(clip.get(), fps, *tbp);
      auto payload = std::move(clip);
      co_await ctx.write(1, std::move(payload));
    }
    co_return;
  }

  // F = 1 + 4*(T-1) RGB frames, emitted ONE BEAT PER FRAME so the whole
  // downstream (save-image, rgb-to-video -> save-video, a preview) is the
  // per-frame machinery that already exists. ----
  if (_family == "wan") {
    if (tbp->shape.size() != 4) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): the Wan VAE decodes a VIDEO latent "
          "[z, T, H/8, W/8]; got a {}-D tensor. Skipping", this->id(),
          tbp->shape.size()));
      co_return;
    }
    const int Cz = (int)tbp->shape[0];
    const int T  = (int)tbp->shape[1];
    const int h8 = (int)tbp->shape[2];
    const int w8 = (int)tbp->shape[3];
    // The producer knows the clip's rate; the config is the fallback for a
    // latent that arrived from somewhere else (a file, a test).
    double fps = _fps;
    // A STREAM's beat: whether it continues the last one, whether it ends
    // the stream, how many of its frames are real, and each frame's colour
    // reference. All absent on a beat that stands alone.
    bool in_stream = false, continues = false, final_beat = false;
    long long valid = -1;
    int blend_head = 0, blend_tail = 0;
    std::vector<std::array<float, 6>> color_ref;
    // A source fed several times: runs of decoded frames merged back into
    // one, after `merge_lead` that pass alone. And the wavelet colour
    // fix's source pictures, announced here and read below.
    std::vector<float> merge_w;
    int merge_lead = 0, ref_frames = 0, ref_levels = 5;
    // The decoder the latent asks for, and the tiny one's source pictures.
    int cond_frames = 0;
    if (tbp->sideband.is_object()) {
      FlexData sb = tbp->sideband;        // as_object() is a view: keep it
      auto o = sb.as_object();
      if (o.contains(sideband::kFps) &&
          o.at(sideband::kFps).as_real(0.0) > 0.0) {
        fps = o.at(sideband::kFps).as_real(fps);
      }
      if (o.contains(sideband::kStreamFinal)) {
        in_stream = true;
        final_beat = o.at(sideband::kStreamFinal).as_bool(false);
        continues = o.contains(sideband::kStreamContinues) &&
                    o.at(sideband::kStreamContinues).as_bool(false);
      }
      if (o.contains(sideband::kValidFrames)) {
        valid = o.at(sideband::kValidFrames).as_int(-1);
      }
      if (o.contains(sideband::kBlendHead)) {
        blend_head = (int)o.at(sideband::kBlendHead).as_int(0);
      }
      if (o.contains(sideband::kBlendTail)) {
        blend_tail = (int)o.at(sideband::kBlendTail).as_int(0);
      }
      if (o.contains(sideband::kMergeWeights)) {
        FlexData mw = o.at(sideband::kMergeWeights);   // keep the owner
        if (mw.is_array()) {
          for (const FlexData& x : mw.as_array()) {
            merge_w.push_back((float)x.as_real(0.0));
          }
        }
      }
      if (o.contains(sideband::kMergeLead)) {
        merge_lead = (int)o.at(sideband::kMergeLead).as_int(0);
      }
      if (o.contains(sideband::kColorRefFrames)) {
        ref_frames = (int)o.at(sideband::kColorRefFrames).as_int(0);
      }
      if (o.contains(sideband::kColorRefLevels)) {
        ref_levels = std::clamp(
            (int)o.at(sideband::kColorRefLevels).as_int(5), 1, 12);
      }
      if (o.contains(sideband::kDecoderCondFrames)) {
        cond_frames = (int)o.at(sideband::kDecoderCondFrames).as_int(0);
      }
      if (o.contains(sideband::kColorRef)) {
        FlexData refs = o.at(sideband::kColorRef);   // keep the view's owner
        if (refs.is_array()) {
          for (const FlexData& one : refs.as_array()) {
            if (!one.is_array() || one.as_array().size() != 6) {
              color_ref.clear();
              break;
            }
            std::array<float, 6> v{};
            std::size_t k = 0;
            for (const FlexData& x : one.as_array()) {
              v[k++] = (float)x.as_real(0.0);
            }
            color_ref.push_back(v);
          }
        }
      }
    }
    // THE PICTURES this latent announces, read NOW, before anything below
    // can skip it: they come in lockstep with the latents, so a beat left
    // unread pairs with the next latent -- and the producer waits on this
    // input. The colour fix's first, then the tiny decoder's.
    const bool pics_wired = ctx.num_iports() > kColorRefPort &&
                            ctx.iport_connected(kColorRefPort);
    std::unique_ptr<BeatPayloadIntf> ref_in, cond_in;
    if (pics_wired && ref_frames > 0) {
      ref_in = co_await ctx.read(kColorRefPort);
    }
    if (pics_wired && cond_frames > 0) {
      cond_in = co_await ctx.read(kColorRefPort);
    }
    if (!pics_wired && (ref_frames > 0 || cond_frames > 0) &&
        !_ref_port_noted) {
      _ref_port_noted = true;
      session()->warn(fmt(
          "VaeDecodeStage('{}'): latents announce source pictures (for {}), "
          "but nothing is wired to the color_reference input (iport {}); "
          "restoring without them", this->id(),
          ref_frames > 0 ? "the wavelet colour fix" : "the tiny decoder",
          kColorRefPort));
    }
    // THE DECODER: the model's tiny one where the latent asks and its
    // pictures came, else the VAE.
    const auto* cond_tb =
        cond_in ? dynamic_cast<const TensorBeatPayload*>(cond_in.get())
                : nullptr;
    bool tiny = _want_tiny && cond_tb != nullptr && ensure_tiny_();
    if (tiny && (cond_tb->dtype != TensorBeat::DType::U8 &&
                 cond_tb->dtype != TensorBeat::DType::F16 &&
                 cond_tb->dtype != TensorBeat::DType::F32)) {
      tiny = false;
    }
    if (tiny && (!cond_tb->strides.empty() || cond_tb->shape.size() != 4 ||
                 cond_tb->shape[0] != cond_frames || cond_frames <= 0 ||
                 cond_tb->shape[1] != 3 || cond_tb->shape[2] != h8 * 8 ||
                 cond_tb->shape[3] != w8 * 8)) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): the tiny decoder's pictures are {}, not {} "
          "planar [3, {}, {}] pictures; decoding this latent with the VAE",
          this->id(), cond_in->describe(), cond_frames, h8 * 8, w8 * 8));
      tiny = false;
    }
    if (_want_tiny && !tiny && cond_in == nullptr && pics_wired) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): the latent asks for the tiny decoder and "
          "announces no pictures for it; decoding with the VAE", this->id()));
    }
    if (!tiny && !ensure_wan_vae_()) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): Wan VAE not loaded; skipping", this->id()));
      co_return;
    }
    const int z_dim = tiny ? 16 : _wan_vae->config().z_dim;
    if (Cz != z_dim || T <= 0 || h8 <= 0 || w8 <= 0) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): latent [{}, {}, {}, {}] does not match z_dim "
          "{}; skipping", this->id(), Cz, T, h8, w8, z_dim));
      co_return;
    }
    // THE DECODE CONTINUES a stream's carries from the beat before, so a
    // cut between latents is no cut in the frames. A beat that opens a
    // stream starts fresh carries; one that continues a stream this stage
    // holds none for (it did not see the opening, or a decode failed)
    // decodes cold, which is the seam a stream exists to remove -- said.
    // A beat that opens and ends its stream at once -- a clip -- needs no
    // carries kept, and decoding it without them leaves it free to tile.
    const auto cold_note = [&] {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): this latent continues a stream whose "
          "earlier beats this stage holds no decode carries for; decoding "
          "it cold, which shows as a change where it begins", this->id()));
    };
    if (tiny) {
      // The tiny decoder's memories, carried while the source is one
      // stream; a beat that stands alone opens (and ends) its own.
      _wan_stream.reset();
      if (!in_stream || !continues || !_tiny_stream) {
        if (in_stream && continues) { cold_note(); }
        _tiny_stream =
            std::make_unique<genai::FlashVsrTinyDecoder::Stream>();
      }
    } else if (!in_stream || (!continues && final_beat)) {
      _tiny_stream.reset();
      _wan_stream.reset();
    } else if (!continues || !_wan_stream || _wan_stream->frames == 0) {
      _tiny_stream.reset();
      if (continues) { cold_note(); }
      _wan_stream = std::make_unique<genai::MetalWanVae::DecodeStream>();
    }
    long long base = tiny         ? _tiny_stream->frames
                     : _wan_stream ? _wan_stream->frames
                                   : 0;
    // A continued decode's first latent frame is four frames, not one.
    const int F_dec = base > 0 ? 4 * T : genai::MetalWanVae::video_frames(T);
    const int F = (valid >= 0 && valid < F_dec) ? (int)valid : F_dec;
    const int H = h8 * 8, W = w8 * 8;
    // THE MERGE, for a source its producer fed several times: after the
    // lead, each run of r decoded frames is one output frame. Everything
    // below that counts frames -- the colour references, the cross-fade,
    // the frame index on the beats -- counts the merged ones.
    const int r = merge_w.size() > 1 ? (int)merge_w.size() : 1;
    const int lead = r > 1 ? std::clamp(merge_lead, 0, F) : 0;
    int F_out = F;
    if (r > 1) {
      if ((F - lead) % r != 0) {
        session()->warn(fmt(
            "VaeDecodeStage('{}'): {} decoded frames after a lead of {} are "
            "not whole runs of {} to merge; the last {} are dropped",
            this->id(), F - lead, lead, r, (F - lead) % r));
      }
      F_out = lead + (F - lead) / r;
    }
    // THE WAVELET COLOUR FIX's source pictures, read above in lockstep
    // with the latent.
    const TensorBeatPayload* ref_tb = nullptr;
    if (ref_frames > 0 && ref_in) {
      ref_tb = dynamic_cast<const TensorBeatPayload*>(ref_in.get());
      const bool ok_dtype =
          ref_tb != nullptr && (ref_tb->dtype == TensorBeat::DType::U8 ||
                                ref_tb->dtype == TensorBeat::DType::F16 ||
                                ref_tb->dtype == TensorBeat::DType::F32);
      if (!ok_dtype || !ref_tb->strides.empty() ||
          ref_tb->shape.size() != 4 ||
          ref_tb->shape[0] != F_out || ref_tb->shape[1] != 3 ||
          ref_tb->shape[2] != H || ref_tb->shape[3] != W) {
        session()->warn(fmt(
            "VaeDecodeStage('{}'): the colour reference is {}, not {} "
            "planar [3, {}, {}] pictures; restoring this latent without "
            "the wavelet colour fix", this->id(), ref_in->describe(), F_out,
            H, W));
        ref_tb = nullptr;
      }
      if (ref_tb != nullptr && !_wavelet) {
        _wavelet = std::make_unique<WaveletColorFix>(mc);
      }
      if (ref_tb != nullptr && !_wavelet->valid()) {
        session()->warn(fmt(
            "VaeDecodeStage('{}'): the wavelet colour fix's kernel is "
            "unavailable; restoring without it", this->id()));
        ref_tb = nullptr;
      }
    }
    // THE CROSS-FADE between overlapping clips. Frames held from the beat
    // before that this one does not blend go out first, as they are; a
    // blend can take no more than were held, nor more than this beat has.
    std::vector<std::unique_ptr<TensorBeatPayload>> frames;
    frames.reserve((std::size_t)F_out);
    blend_head = std::clamp(blend_head, 0, F_out);
    if (blend_head > (int)_held.size()) {
      if (!_held.empty() || blend_head > 0) {
        session()->warn(fmt(
            "VaeDecodeStage('{}'): this latent cross-fades its first {} "
            "frame(s) into the last beat's, which held {}; blending {}",
            this->id(), blend_head, _held.size(), _held.size()));
      }
      blend_head = (int)_held.size();
    }
    if (blend_head == 0 || !_held.empty()) {
      for (std::size_t i = (std::size_t)blend_head; i < _held.size(); ++i) {
        if (auto b = held_beat_(_held[i])) { frames.push_back(std::move(b)); }
      }
      if (blend_head == 0) { _held.clear(); }
    }
    blend_tail = std::clamp(blend_tail, 0, F_out - blend_head);
    std::vector<HeldFrame> new_held;

    // The VAE's input: f16, un-whitened. The tiny decoder reads the
    // latent as the denoiser left it.
    metal_compute::SharedBuffer zw;
    if (!tiny) {
      const std::size_t nz = (std::size_t)Cz * T * h8 * w8;
      metal_compute::SharedBuffer z = mc->make_shared_buffer(nz * 2);
      { auto* d = static_cast<_Float16*>(z.contents());
        const float* s = tbp->as_f32();
        for (std::size_t i = 0; i < nz; ++i) { d[i] = (_Float16)s[i]; } }
      zw = _wan_vae->unwhiten(z, T, h8, w8);
      if (zw.empty()) {
        session()->warn(fmt(
            "VaeDecodeStage('{}'): un-whiten failed; skipping", this->id()));
        co_return;
      }
    }

    // The VAE streams its output a chunk at a time so a long clip never
    // exists in f16 all at once. The beat boundary cannot stream with it:
    // ctx.write is a coroutine and the sink is a plain callback, so the
    // frames are converted to U8 in the sink -- where the f16 chunk is
    // still alive and is then dropped -- and written out afterwards. What
    // is held is the clip as planar U8 RGB, which is a THIRD of the f16 it
    // replaces (81 frames of 480x832 = 97 MB), not the f16 working set the
    // chunking exists to bound.
    std::vector<float> unit((std::size_t)3 * H * W);
    const std::size_t hw = (std::size_t)H * W;
    const std::size_t per = 3 * hw;
    // One OUTPUT frame, in [-1, 1]: its colour fix, the cross-fade, and out
    // -- or held back for the next beat's blend.
    std::vector<float> refbuf;
    auto finish = [&](int o, std::vector<float>& x) {
      const std::array<float, 6>* ref =
          o < (int)color_ref.size() ? &color_ref[(std::size_t)o] : nullptr;
      if (ref != nullptr && hw > 1) {
        // THE REFERENCE'S COLOUR FIX (v1.1): each channel's mean and
        // spread moved to the source frame's (AdaIN, the unbiased variance
        // plus 1e-5), then clamped to the decode's range.
        for (int c = 0; c < 3; ++c) {
          float* p = x.data() + (std::size_t)c * hw;
          double sum = 0.0, sq = 0.0;
          for (std::size_t i = 0; i < hw; ++i) {
            sum += (double)p[i];
            sq += (double)p[i] * (double)p[i];
          }
          const double mu = sum / (double)hw;
          const double var =
              std::max(0.0, (sq - (double)hw * mu * mu) / (double)(hw - 1));
          const double sd = std::sqrt(var + 1e-5);
          const float scale = (float)((*ref)[3 + c] / sd);
          const float shift = (float)((*ref)[c] - mu * (*ref)[3 + c] / sd);
          for (std::size_t i = 0; i < hw; ++i) {
            p[i] = std::clamp(p[i] * scale + shift, -1.0f, 1.0f);
          }
        }
      } else if (ref_tb != nullptr && o < F_out) {
        // THE WAVELET COLOUR FIX (FlashVSR+): the frame's low frequencies
        // from its source picture, its detail its own.
        refbuf.resize(per);
        const std::size_t at = (std::size_t)o * per;
        if (ref_tb->dtype == TensorBeat::DType::U8) {
          const std::uint8_t* rp8 = ref_tb->as_u8() + at;
          for (std::size_t i = 0; i < per; ++i) {
            refbuf[i] = (float)rp8[i] / 127.5f - 1.0f;
          }
        } else if (ref_tb->dtype == TensorBeat::DType::F16) {
          const auto* rp16 =
              reinterpret_cast<const _Float16*>(ref_tb->as_f16()) + at;
          for (std::size_t i = 0; i < per; ++i) {
            refbuf[i] = 2.0f * (float)rp16[i] - 1.0f;
          }
        } else {
          const float* rp32 = ref_tb->as_f32() + at;
          for (std::size_t i = 0; i < per; ++i) {
            refbuf[i] = 2.0f * rp32[i] - 1.0f;
          }
        }
        std::string werr;
        if (!_wavelet->apply(x.data(), refbuf.data(), H, W, ref_levels,
                             &werr)) {
          session()->warn(fmt(
              "VaeDecodeStage('{}'): wavelet colour fix failed ({}); this "
              "frame goes out without it", this->id(), werr));
        }
      }
      auto out = std::make_unique<TensorBeatPayload>();
      out->dtype = _out_dtype;
      out->shape = {3, H, W};
      out->resize_contiguous(per);
      for (std::size_t i = 0; i < per; ++i) { unit[i] = (x[i] + 1.0f) * 0.5f; }
      // A frame the last beat also restored: a raised cosine from its
      // copy to this one, so neither clip's start or end is seen alone.
      if (o < blend_head) {
        const HeldFrame& h = _held[(std::size_t)o];
        const float w = (float)(0.5 - 0.5 * std::cos(
            M_PI * (double)(o + 1) / (double)(blend_head + 1)));
        for (std::size_t i = 0; i < unit.size(); ++i) {
          unit[i] = (1.0f - w) * h.unit[i] + w * unit[i];
        }
      }
      for (std::size_t i = 0; i < unit.size(); ++i) {
        put_unit_(*out, i, unit[i]);
      }
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign(sideband::kFrame,
                                      FlexData::make_int((std::int64_t)o));
      sb.as_object().insert_or_assign(
          sideband::kFrames, FlexData::make_int((std::int64_t)F_out));
      sb.as_object().insert_or_assign(sideband::kFps,
                                      FlexData::make_real(fps));
      out->sideband = std::move(sb);
      forward_model_name_(*tbp, *out);
      // The next beat repeats this one's last frames: hold them for it.
      if (o >= F_out - blend_tail) {
        HeldFrame h;
        h.unit = unit;
        h.H = H;
        h.W = W;
        h.sb = out->sideband;
        new_held.push_back(std::move(h));
        return;
      }
      frames.push_back(std::move(out));
    };
    // The decoded frame in [-1, 1], and the run being merged.
    std::vector<float> xf(per), acc;
    if (r > 1) { acc.assign(per, 0.0f); }
    // Decoded frame `local` of this beat, in `xf`: into its run, or
    // finished as it is.
    auto take = [&](long long local) {
      if (r == 1 || local < lead) {
        finish((int)local, xf);
        return;
      }
      // A RUN: the weighted sum of its r decoded frames, in f32 as the
      // reference sums it. Runs never straddle a beat (the producer cuts
      // on whole runs), but they do straddle the decode's chunks, which
      // is why the sum lives out here.
      const int slot = (int)((local - lead) % r);
      const float w = merge_w[(std::size_t)slot];
      for (std::size_t i = 0; i < per; ++i) {
        acc[i] = (slot == 0 ? 0.0f : acc[i]) + w * xf[i];
      }
      if (slot == r - 1) {
        finish(lead + (int)((local - lead) / r), acc);
      }
    };
    auto sink = [&](const metal_compute::SharedBuffer& rgb, int frame0,
                    int n) -> bool {
      const auto* rp = static_cast<const _Float16*>(rgb.contents());
      for (int k = 0; k < n; ++k) {
        // This beat's own index of the frame: a stream's decode counts
        // from the stream's start.
        const long long local = (long long)frame0 + k - base;
        // The stream's padding, past its last real frame.
        if (local < 0 || local >= F) { continue; }
        // The VAE hands back [3, n, H, W] (channel-first over the chunk),
        // so frame k is a stride into each channel plane, not a contiguous
        // block.
        for (int c = 0; c < 3; ++c) {
          const _Float16* src = rp + ((std::size_t)c * n + k) * hw;
          float* d = xf.data() + (std::size_t)c * hw;
          for (std::size_t i = 0; i < hw; ++i) { d[i] = (float)src[i]; }
        }
        take(local);
      }
      return true;
    };
    // THE TINY DECODER's conditioning: decoded frame d's source picture,
    // by the merge's own rule (the lead alone, then one per run of r), the
    // last standing in for the padding past the source's end.
    const auto picture = [&](long long d) {
      genai::FlashVsrTinyDecoder::Picture p;
      const long long m = cond_tb != nullptr ? cond_tb->shape[0] : 0;
      long long i = (r == 1 || d < lead) ? d : lead + (d - lead) / r;
      i = std::clamp<long long>(i, 0, m - 1);
      const std::size_t at = (std::size_t)i * per;
      if (cond_tb->dtype == TensorBeat::DType::U8) {
        p.u8 = cond_tb->as_u8() + at;
      } else if (cond_tb->dtype == TensorBeat::DType::F16) {
        p.f16 = reinterpret_cast<const _Float16*>(cond_tb->as_f16()) + at;
      } else {
        p.f32 = cond_tb->as_f32() + at;
      }
      return p;
    };
    const auto tiny_sink = [&](long long d, float* chw) {
      if (d < 0 || d >= F) { return true; }   // past the real frames
      std::memcpy(xf.data(), chw, per * sizeof(float));
      take(d);
      return true;
    };
    {
      // The clip, and what the decode holds to make it: for the VAE
      // carries for the whole clip and a full-resolution chunk pool,
      // several times the clip at 1920x1152; for the tiny decoder its
      // scratch, a commit's input and outputs and one frame a layer.
      const std::size_t px =
          (std::size_t)F * (std::size_t)H * (std::size_t)W;
      std::size_t scratch = 0;
      if (tiny) {
        genai::FlashVsrTinyDecoder::plan(_tiny_path, T, h8, w8, nullptr,
                                         &scratch);
        revise_decode_arena_(px * 3 + scratch);
      } else {
        revise_decode_arena_(px * 3 * 2 + px * 3 +
                             _wan_vae->decode_peak_bytes(h8, w8));
      }
    }
    std::string derr;
    bool ok = false;
    {
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae,
                         kPerfLlmVaeBegin,
                         (std::uint64_t)F * (std::uint64_t)H * W);
      if (tiny) {
        ok = _tiny->decode(tbp->as_f32(), T, h8, w8, picture, tiny_sink,
                           _tiny_stream.get(), &derr);
        if (!ok) { _tiny_stream.reset(); }
      } else {
        ok = _wan_vae->decode(zw, T, h8, w8, sink, &derr,
                              _wan_stream.get());
      }
      // A stream's decode that could not run whole-plane loses its
      // carries either way; decoding this beat cold, which may tile, at
      // least keeps its frames.
      if (!tiny && !ok && _wan_stream && frames.empty()) {
        session()->warn(fmt(
            "VaeDecodeStage('{}'): the stream's decode failed ({}); decoding "
            "this beat cold, so its first frames will not match the last "
            "beat's", this->id(), derr));
        _wan_stream.reset();
        derr.clear();
        base = 0;
        ok = _wan_vae->decode(zw, T, h8, w8, sink, &derr);
      }
    }
    if (in_stream && final_beat) { _wan_stream.reset(); }
    // The tiny decoder's memories go with the stream's last beat, and a
    // beat that stands alone was its own stream.
    if (!in_stream || final_beat) { _tiny_stream.reset(); }
    if (ok) { _held = std::move(new_held); }
    if (!ok) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): video decode failed ({}); skipping",
          this->id(), derr.empty() ? "unknown error" : derr));
      co_return;
    }
    session()->log_debug(fmt(
        "VaeDecodeStage('{}'): decoded [{}, {}, {}, {}] -> {} frames "
        "[3, {}, {}] @ {:.3f} fps{}", this->id(), Cz, T, h8, w8,
        frames.size(), H, W, fps, tiny ? " (tiny decoder)" : ""));
    if (_unload_idle) { unload_vae_(); }
    auto clip = (!frames.empty() && frames[0] != nullptr
                 && frames[0]->shape.size() == 3)
                    ? begin_clip_(ctx, (int)frames.size(),
                                  (int)frames[0]->shape[1],
                                  (int)frames[0]->shape[2])
                    : nullptr;
    int clip_at = 0;
    for (auto& f : frames) {
      add_to_clip_(clip.get(), clip_at++, *f);
      ++_images_emitted;
      co_await ctx.write(0, std::move(f));
    }
    if (clip) {
      finish_clip_(clip.get(), fps, *tbp);
      auto payload = std::move(clip);
      co_await ctx.write(1, std::move(payload));
    }
    co_return;
  }


  if (!_vae) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): VAE not loaded; skipping", this->id()));
    co_return;
  }
  const int Cz = (int)tbp->shape[0];
  const int h8 = (int)tbp->shape[1];
  const int w8 = (int)tbp->shape[2];
  // Pixels per latent cell. 8 for every checkpoint this path served
  // until Qwen-Image-2.1, whose five levels make it 16 -- so it is asked
  // rather than assumed. The name h8/w8 is kept because it is what the
  // rest of this function and the model's own API still call them.
  const int px = _vae->config().spatial_factor();
  session()->log_debug(fmt(
      "VaeDecodeStage('{}'): beat received, latent [{}, {}, {}] -> image "
      "[{}, {}]", this->id(), Cz, h8, w8, h8 * px, w8 * px));
  if (Cz != _vae->config().z_dim || h8 <= 0 || w8 <= 0) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): latent [{}, {}, {}] does not match z_dim {}; "
        "skipping", this->id(), Cz, h8, w8, _vae->config().z_dim));
    co_return;
  }

  // f32 latent -> f16 channel-first buffer -> un-whiten -> decode.
  const std::size_t nz = (std::size_t)Cz * h8 * w8;
  metal_compute::SharedBuffer z = mc->make_shared_buffer(nz * 2);
  { auto* d = static_cast<_Float16*>(z.contents());
    const float* s = tbp->as_f32();
    for (std::size_t i = 0; i < nz; ++i) { d[i] = (_Float16)s[i]; } }

    // Publish the arena from the geometry actually in hand. The image
    // branches need this as much as the video ones: an EDIT graph takes
    // its size from the reference image, so generate-image had no
    // config geometry to declare from -- and Qwen-Image-Edit has no
    // free_*_dit_for_decode_ at all, so nothing upstream states it
    // either. Same estimator the plan used, at the real pixel size.
  publish_image_arena_(w8 * px, h8 * px);
  metal_compute::SharedBuffer zw = _vae->unwhiten(z, h8, w8);
  if (zw.empty()) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): un-whiten failed; skipping", this->id()));
    co_return;
  }
  std::string derr;
  metal_compute::SharedBuffer rgb;
  {
    PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae, kPerfLlmVaeBegin,
                       (std::uint64_t)(h8 * px) * (w8 * px));
    rgb = _vae->decode(zw, h8, w8, &derr);
  }
  if (rgb.empty()) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): decode failed ({}); skipping", this->id(),
        derr.empty() ? "unknown error" : derr));
    co_return;
  }

  // f16 [C,H,W] in [-1,1] -> planar (x+1)/2: U8 rounded and clamped, or
  // F16 clamped to 0..1 (`dtype`).
  const int H = h8 * px, W = w8 * px;
  // THE CHANNEL COUNT IS THE VAE'S, not three. Qwen-Image-2.1 decodes
  // FOUR -- RGBA is its headline feature -- and this used to truncate
  // to three and say so, which was honest but threw the alpha away.
  // A consumer that cannot take four (a video muxer, a vision tower)
  // composites or refuses on its own; that decision belongs where the
  // pixels are used, not here where they are produced.
  const int OC = _vae->config().out_channels;
  if (OC != 3 && OC != 4) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): this VAE decodes {} channels, and an image "
        "beat is RGB or RGBA; dropping beat", this->id(), OC));
    co_return;
  }
  const std::size_t n = (std::size_t)OC * H * W;
  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = _out_dtype;
  out->shape = {OC, H, W};
  out->resize_contiguous(n);
  const auto* rp = static_cast<const _Float16*>(rgb.contents());
  for (std::size_t i = 0; i < n; ++i) {
    put_unit_(*out, i, ((float)rp[i] + 1.0f) * 0.5f);
  }
  ++_images_emitted;
  session()->log_debug(fmt(
      "VaeDecodeStage('{}'): decoded + emitted image #{} planar {} {} "
      "[{}, {}, {}]", this->id(), _images_emitted,
      _out_dtype == TensorBeat::DType::F16 ? "F16" : "U8",
      OC == 4 ? "RGBA" : "RGB", OC, H, W));
  if (crop_to_box_(*tbp, out)) {
    session()->info(fmt(
        "VaeDecodeStage('{}'): cut the {}x{} canvas back to the "
        "reference's {}x{}", this->id(), W, H, out->shape[2],
        out->shape[1]));
  }
  forward_model_name_(*tbp, *out);
  if (_unload_idle) { unload_vae_(); }
  co_await ctx.write(0, std::move(out));
}

#else   // !VPIPE_BUILD_APPLE_SILICON

Job
VaeDecodeStage::initialize(RuntimeContext& ctx)
{
  (void)ctx;
  if (session()) {
    session()->error(fmt(
        "VaeDecodeStage('{}'): built without VPIPE_BUILD_APPLE_SILICON; the "
        "metal VAE is unavailable, the stage is inert", this->id()));
  }
  co_return;
}

Job
VaeDecodeStage::process(RuntimeContext& ctx)
{
  auto in = co_await ctx.read(0);
  (void)in;
  ctx.signal_done();
  co_return;
}

#endif  // VPIPE_BUILD_APPLE_SILICON

VPIPE_REGISTER_STAGE(VaeDecodeStage)
VPIPE_REGISTER_SPEC(VaeDecodeStage, kSpec)

}
