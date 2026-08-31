#include "stages/vae-decode-stage.h"

#include <cstring>

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/shared/comfy-checkpoint.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-detect.h"
#include "stages/model-provenance.h"
#include "stages/model-registry.h"

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
#ifdef VPIPE_BUILD_APPLE_SILICON
  _fps = attr_real("fps");
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

VaeDecodeStage::~VaeDecodeStage() = default;

namespace {
// The model iport (a model-select source) overrides hf_dir. Appended after
// the primary `latent` input, so it is iport1. (Referenced only from the
// Apple-gated code below; marked maybe_unused for the inert non-Apple build.)
[[maybe_unused]] constexpr unsigned kModelPort = 1;

// MiniMax-H3's video VAE works in IMAGENET-normalized pixel space, so
// its decoder's output has to be un-normalized before it means [0, 1].
// These are the reference's own constants, not a preprocessing choice.
[[maybe_unused]] constexpr float kImagenetMean[3] = {0.485f, 0.456f,
                                                     0.406f};
[[maybe_unused]] constexpr float kImagenetStd[3]  = {0.229f, 0.224f,
                                                     0.225f};

const ConfigKey kAttrs[] = {
  {.key = "hf_dir", .type = ConfigType::String, .required = false,
   .doc = "the model root whose VAE this decodes; the resident family "
          "locates the VAE within it (conventionally <hf_dir>/vae). "
          "OPTIONAL: a model-select source on the model iport overrides it",
   .suggest_db = kModelRegistryDb,
   .suggest_db_type = "krea2,flux2,qwen-image-edit,mage-flow,mage-flow-edit,"
       "boogu-image,boogu-image-edit,"
       "wan-t2v,wan-i2v,minimax-h3-fl2va,minimax-h3-ref2va",
   .model_channel = "diffusion-model"},
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
          "physical RAM vs the pipeline's weight bytes; \"always\" / "
          "\"never\" force it",
   .def_str = "auto"},
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
};
const PortSpec kOports[] = {
  {.name = "image",
   .doc = "decoded planar U8 RGB TensorBeat [3, H, W]. A VIDEO latent emits "
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

#ifdef VPIPE_BUILD_APPLE_SILICON
// Detect the VAE family from the vae config.json `_class_name`:
// "AutoencoderKLFlux2" -> "flux2"; "MageVAE" -> "mage"; anything else
// (AutoencoderKLQwenImage) -> "krea2".
std::string
vae_family_(const std::string& vae_dir)
{
  namespace fs = std::filesystem;
  // A Comfy-Org repack ships its VAEs as bare .safetensors under `vae/`
  // with no config.json at all, so the `_class_name` read below finds
  // nothing and falls through to the "krea2" default -- which then opens
  // that directory as a Qwen-Image VAE and reports "no readable
  // checkpoint". The architecture is in the file's `__metadata__` key
  // instead, and probing for it costs one header read.
  if (!genai::comfy::resolve_component(vae_dir, "vae", "minimax_h3_video_vae",
                                       {"video_vae"}).empty()) {
    return "minimax-h3";
  }
  std::ifstream in(fs::path(vae_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto obj = fd.as_object();
      if (obj.contains("_class_name")) {
        const std::string cls(obj.at("_class_name").as_string(""));
        if (cls == "AutoencoderKLFlux2") { return "flux2"; }
        // The PLAIN diffusers AutoencoderKL (the FLUX.1 VAE Boogu-Image uses)
        // runs on the SAME code path -- MetalFlux2Vae is config-driven, and at
        // patch 1 with a scalar shift/scale whitening it IS a plain
        // AutoencoderKL. Same family string, so the branches below are shared.
        if (cls == "AutoencoderKL") { return "flux2"; }
        if (cls == "MageVAE") { return "mage"; }
        // The VIDEO VAE. Same tensor names as the Qwen-Image one
        // (it IS the general form of it), so the class name is the
        // only thing that tells them apart -- and getting it wrong
        // would decode a video latent as a single still.
        if (cls == "AutoencoderKLWan") { return "wan"; }
        // MiniMax-H3's video VAE. Its own class because the DECODER is a
        // 36-layer ViT -- one token per latent voxel, each projecting a
        // whole 4x16x16 pixel block -- not an upsampling conv stack.
        if (cls == "MiniMaxH3VideoVAE") { return "minimax-h3"; }
      }
    }
  }
  return "krea2";
}

// Mage-Flow's MageVAE geometry from vae/config.json (latent_channels 128,
// downsample_factor 16). Everything else -- the DiCo trunk width, the CoD
// decoder, the per-pixel MLP head -- is fixed by the checkpoint, not
// configurable, so it stays on MetalMageVae::Config's defaults.
genai::MetalMageVae::Config
mage_vae_config_(const std::string& vae_dir)
{
  genai::MetalMageVae::Config c;
  std::ifstream in(std::filesystem::path(vae_dir) / "config.json");
  if (in) {
    FlexData fd = FlexData::from_json(in);
    if (fd.is_object()) {
      auto o = fd.as_object();
      if (o.contains("latent_channels")) {
        c.latent_channels =
            (int)o.at("latent_channels").as_int(c.latent_channels);
      }
      if (o.contains("downsample_factor")) {
        c.patch = (int)o.at("downsample_factor").as_int(c.patch);
      }
    }
  }
  return c;
}
#endif
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
  if (!_vae && !_flux2_vae && !_mage_vae && !_wan_vae && !_h3_vae &&
      !_plugin_dec) {
    _load_attempted = false;
    _unloaded       = false;
  }

  // Per-launch reset: the stage survives a stop/relaunch, and the
  // select sources upstream re-emit on every launch. Without this the
  // re-emitted beat is never latched and this stage keeps the previous
  // run's selection.
  _model_latched = false;

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
        const std::string fam  = vae_family_(resolve_vae_dir(root));
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
  return model_memory::weight_claims_in_phase({vae_dir_for_release_()},
                                              model_memory::kPhaseDecode);
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
  const bool want =
      model_memory::resolve_idle_unload(ram, fp, bytes, _unload_idle);
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
                                     genai::VaeModelFamily::Role::kVideo);
    if (!m.holdings.empty()) {
      // The POLICY is this stage's: a decode's weights go at its idle
      // point, which is what `releases` means here.
      for (StageHolding& h : m.holdings) { h.releases = true; }
      return m;
    }
  }
#endif
  // A VAE has no streaming form: it holds what it weighs, so `floor`
  // stays 0 and is read as `preload`.
  m.hold(vae_dir, model_memory::dir_weights_bytes(vae_dir), 0,
         /*releases=*/true);
  // The arena and the frames it produces are sized by the clip's
  // geometry, which arrives on a beat -- generate-video declares both,
  // because it is the stage that knows the geometry at plan time.
  return m;
}

std::string
VaeDecodeStage::vae_dir_for_release_() const
{
  if (_hf_dir.empty()) { return {}; }
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
  if (generic != root) { return generic; }
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
        f->vae_path(root, genai::VaeModelFamily::Role::kVideo);
    if (!p.empty()) { return p; }
  }
  const std::string h3 = genai::MetalMiniMaxH3VideoVae::resolve_vae_dir(root);
  if (h3 != root) { return h3; }
#endif
  return generic;
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
  if (!_vae && !_flux2_vae && !_mage_vae && !_wan_vae && !_h3_vae &&
      !_plugin_dec) {
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
  _mage_vae.reset();
  _wan_vae.reset();
  _h3_vae.reset();
  _plugin_dec.reset();
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

  _family = vae_family_(vae_dir);
  // One shared, reference-counted view of this checkpoint. The peer VAE
  // stage in the same graph (encode beside decode) names the same
  // directory and gets the SAME set, so the two halves of the model are
  // loaded once between them instead of once each -- and two pipelines
  // running the same model share it too.
  // Falls back to a private set when the session has no manager; see
  // open_weight_set().
  // MiniMax-H3 keeps the WEIGHTS one level below the config that named
  // the family, so the set is opened on the resolved leaf. Every other
  // family's resolver returns what it was handed.
  const std::string ws_dir =
      (_family == "minimax-h3")
          ? genai::MetalMiniMaxH3VideoVae::resolve_vae_dir(vae_dir)
          : vae_dir;
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
    load_note_(fmt(
        "VaeDecodeStage('{}'): loading 2D AutoencoderKL ({}) from '{}'",
        this->id(), fcfg.patch > 1 ? "FLUX.2, patchified" : "plain", vae_dir));
    _flux2_vae = genai::MetalFlux2Vae::load(ws, mc, fcfg);
    if (!_flux2_vae) {
      session()->error(fmt(
          "VaeDecodeStage('{}'): failed to load the AutoencoderKL from '{}'; "
          "inert",
          this->id(), vae_dir));
    }
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
    load_note_(fmt("VaeDecodeStage('{}'): loading the MiniMax-H3 video VAE "
                   "from '{}'", this->id(), vae_dir));
    _h3_vae = genai::MetalMiniMaxH3VideoVae::load(vae_dir, mc, h3cfg);
    if (!_h3_vae) {
      session()->error(fmt(
          "VaeDecodeStage('{}'): failed to load the MiniMax-H3 video VAE "
          "from '{}'; inert", this->id(), vae_dir));
    }
    return;
  }
  if (_family == "wan") {
    genai::MetalWanVae::Config wcfg;
    std::string werr;
    if (!genai::MetalWanVae::config_from_json(vae_dir, wcfg, &werr)) {
      session()->error(fmt(
          "VaeDecodeStage('{}'): {}; inert", this->id(), werr));
      return;
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
    return;
  }

  if (_family == "mage") {
    load_note_(fmt("VaeDecodeStage('{}'): loading MageVAE from '{}'",
                        this->id(), vae_dir));
    // Decode-only here: the encoder half (student.dconv_encoder.*) is the
    // vae-encode stage's business, and skipping it saves ~67M params.
    _mage_vae = genai::MetalMageVae::load(ws, mc,
                                          mage_vae_config_(vae_dir),
                                          /*with_encoder=*/false);
    if (!_mage_vae) {
      session()->error(fmt(
          "VaeDecodeStage('{}'): failed to load the MageVAE from '{}'; inert",
          this->id(), vae_dir));
    }
    return;
  }

  genai::MetalKrea2Vae::Config cfg;   // Qwen-Image VAE defaults
  // Read the per-channel latent statistics (and z_dim / base_dim) from the
  // vae config.json; the un-whiten needs latents_mean / latents_std.
  {
    std::ifstream in(fs::path(vae_dir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto obj = fd.as_object();
        if (obj.contains("z_dim")) {
          cfg.z_dim = (int)obj.at("z_dim").as_int(cfg.z_dim);
        }
        if (obj.contains("base_dim")) {
          cfg.base_dim = (int)obj.at("base_dim").as_int(cfg.base_dim);
        }
        if (obj.contains("num_res_blocks")) {
          cfg.num_res_blocks =
              (int)obj.at("num_res_blocks").as_int(cfg.num_res_blocks);
        }
        if (obj.contains("latents_mean")) {
          FlexData lm = obj.at("latents_mean");
          for (auto v : lm.as_real_span()) { cfg.latents_mean.push_back((float)v); }
        }
        if (obj.contains("latents_std")) {
          FlexData ls = obj.at("latents_std");
          for (auto v : ls.as_real_span()) { cfg.latents_std.push_back((float)v); }
        }
      }
    }
  }
  if ((int)cfg.latents_mean.size() != cfg.z_dim ||
      (int)cfg.latents_std.size() != cfg.z_dim) {
    session()->error(fmt(
        "VaeDecodeStage('{}'): vae config.json is missing latents_mean/"
        "latents_std ({}/{} of z_dim {}); the stage is inert", this->id(),
        cfg.latents_mean.size(), cfg.latents_std.size(), cfg.z_dim));
    return;
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

}  // namespace

#ifdef VPIPE_BUILD_APPLE_SILICON

std::unique_ptr<TensorBeatPayload>
VaeDecodeStage::begin_clip_(RuntimeContext& ctx, int F, int H, int W) const
{
  // Nothing wired, nothing built. Returning null rather than a flag
  // means no caller can hold a flag that disagrees with a pointer.
  if (!(ctx.num_oports() > 1 && ctx.has_consumers(1))) { return nullptr; }
  if (F <= 0 || H <= 0 || W <= 0) { return nullptr; }
  auto clip = std::make_unique<TensorBeatPayload>();
  clip->dtype = TensorBeat::DType::U8;
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
  std::memcpy(clip->as_u8() + off, frame.bytes_(), per);
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
      "frames",
      FlexData::make_int(clip->shape.empty() ? 0 : clip->shape[0]));
  sb.as_object().insert_or_assign("fps", FlexData::make_real(fps));
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
  if (!in) { ctx.signal_done(); co_return; }   // upstream EOS -> close oport
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
  // A prompt the Mage-Flow content screen refused never reaches a DiT, so
  // there is no latent to decode -- the upstream beat is a marker carrying
  // only the intended image size. Paint the reference's refusal image (a
  // plain white frame: no text, no category, nothing that would tell a
  // prober WHICH rule they hit) and emit it in the generated image's place.
  // No VAE needed, so this runs even when the VAE never loaded.
  if (tbp->sideband.is_object()) {
    FlexData sb = tbp->sideband;          // as_object() is a view: keep it
    auto o = sb.as_object();
    if (o.contains("content_blocked")
        && o.at("content_blocked").as_bool(false)) {
      const int H = o.contains("refusal_height")
                        ? (int)o.at("refusal_height").as_int(0) : 0;
      const int W = o.contains("refusal_width")
                        ? (int)o.at("refusal_width").as_int(0) : 0;
      if (H <= 0 || W <= 0) {
        session()->warn(fmt(
            "VaeDecodeStage('{}'): refusal beat carries no image size; "
            "dropping", this->id()));
        co_return;
      }
      auto out = std::make_unique<TensorBeatPayload>();
      out->dtype = TensorBeat::DType::U8;
      out->shape = {3, H, W};
      const std::size_t n = (std::size_t)3 * H * W;
      out->resize_contiguous(n);
      std::memset(out->as_u8(), 0xff, n);      // white
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
    out->dtype = TensorBeat::DType::U8;
    out->shape = {3, H, W};
    out->resize_contiguous(n);
    const auto* rp = static_cast<const _Float16*>(rgb.contents());
    std::uint8_t* op = out->as_u8();
    for (std::size_t i = 0; i < n; ++i) {
      float v = ((float)rp[i] + 1.0f) * 0.5f * 255.0f;
      v = std::round(v);
      if (v < 0.0f) { v = 0.0f; }
      if (v > 255.0f) { v = 255.0f; }
      op[i] = (std::uint8_t)v;
    }
    ++_images_emitted;
    session()->log_debug(fmt(
        "VaeDecodeStage('{}'): AutoencoderKL decoded + emitted image #{} "
        "[3, {}, {}]",
        this->id(), _images_emitted, H, W));
    forward_model_name_(*tbp, *out);
    if (_unload_idle) { unload_vae_(); }
    co_await ctx.write(0, std::move(out));
    co_return;
  }

  // ---- Mage-Flow MageVAE: input [128, H/16, W/16]; decode straight to RGB in
  // [-1,1] (no per-channel un-whiten -- MageVAE has no latents_mean/std). ----
  // ---- Wan video VAE: input [z_dim, T, h8, w8]; decode to
  // ---- MiniMax-H3: a ViT decoder over a 16x-compressed latent --------
  if (_family == "minimax-h3") {
    if (!_h3_vae) {
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
    const auto& vc = _h3_vae->config();
    const int Cz = (int)tbp->shape[0];
    const int LT = (int)tbp->shape[1];
    const int lh = (int)tbp->shape[2];
    const int lw = (int)tbp->shape[3];
    if (Cz != vc.z_channels || LT <= 0 || lh <= 0 || lw <= 0) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): latent [{}, {}, {}, {}] does not match "
          "z_channels {}; skipping", this->id(), Cz, LT, lh, lw,
          vc.z_channels));
      co_return;
    }
    double fps = _fps;
    if (tbp->sideband.is_object()) {
      FlexData sb = tbp->sideband;        // as_object() is a view: keep it
      auto o = sb.as_object();
      if (o.contains("fps") && o.at("fps").as_real(0.0) > 0.0) {
        fps = o.at("fps").as_real(fps);
      }
    }

    // Un-whiten on the way in. The DiT generates in NORMALIZED latent
    // space, so decoding the beat as-is would feed the VAE a latent with
    // the wrong scale AND the wrong per-channel offset -- which decodes
    // to a plausible but washed-out clip rather than to obvious noise.
    const std::size_t nz = (std::size_t)Cz * LT * lh * lw;
    const std::size_t vox = (std::size_t)LT * lh * lw;
    metal_compute::SharedBuffer z = mc->make_shared_buffer(nz * 2);
    if (z.empty()) { co_return; }
    {
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
      const std::size_t out_frames =
          (std::size_t)_h3_vae->decoded_frames(LT);
      const std::size_t px = (std::size_t)(lh * vc.patch) *
                             (std::size_t)(lw * vc.patch) * out_frames;
      const std::size_t need = px * 3 * 2      // decode output, bf16
                             + px * 3;         // the stage's planar-U8 clip
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
              out_frames, lw * vc.patch, lh * vc.patch, need >> 20));
        } else {
          session()->error(fmt(
              "VaeDecodeStage('{}'): not enough memory to decode {} frames "
              "at {}x{} -- the output alone is ~{} MB and only ~{} MB is "
              "reclaimable{}. Refusing rather than thrashing: wired Metal "
              "buffers cannot be paged out, so overcommitting takes the "
              "machine down rather than failing this stage",
              this->id(), out_frames, lw * vc.patch, lh * vc.patch,
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
                         (std::uint64_t)LT * vc.patch_t *
                             (std::uint64_t)(lh * vc.patch) *
                             (std::uint64_t)(lw * vc.patch));
      // A minute of a 2.4B ViT with nothing else to look at. The tile is
      // the only unit the decode passes through often enough to be worth
      // reporting, and it is what the bar counts.
      UiProgress bar = session()->open_progress("vae decode");
      _h3_vae->set_tile_progress([&bar](int done, int total) {
        bar.update((std::uint64_t)(done < 0 ? 0 : done),
                   (std::uint64_t)(total < 0 ? 0 : total));
      });
      rgb = _h3_vae->decode_video(z, LT, lh, lw, &F, &derr);
      // Cleared before the bar it captures leaves scope -- the VAE
      // outlives this call and would otherwise hold a dangling reference
      // into the next decode.
      _h3_vae->set_tile_progress(nullptr);
      bar.finish();
    }
    if (rgb.empty() || F <= 0) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): MiniMax-H3 video decode failed ({}); "
          "skipping", this->id(), derr.empty() ? "unknown error" : derr));
      co_return;
    }
    const int H = lh * vc.patch, W = lw * vc.patch;
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
    auto clip = begin_clip_(ctx, F, H, W);
    for (int f = 0; f < F; ++f) {
      auto out = std::make_unique<TensorBeatPayload>();
      out->dtype = TensorBeat::DType::U8;
      out->shape = {3, H, W};
      out->resize_contiguous(3 * plane);
      std::uint8_t* op = out->as_u8();
      for (int c = 0; c < 3; ++c) {
        const std::uint16_t* src = rp + ((std::size_t)c * F + f) * plane;
        std::uint8_t* dst = op + (std::size_t)c * plane;
        // The ViT decoder emits IMAGENET-NORMALIZED pixels, not [-1, 1]:
        // the encoder's first act is `(x + 1)/2` then `(. - mean)/std`,
        // and the reference decode undoes exactly that before clamping.
        // Treating its output as [-1, 1] leaves it ~1/std = 4.4x too
        // wide, so the u8 conversion clips most of the frame -- and
        // because each 16x16 token block clips against its own local
        // statistics, the result is a BLOCK GRID rather than an
        // obviously blown-out image.
        const float pm = kImagenetMean[c], ps = kImagenetStd[c];
        for (std::size_t i = 0; i < plane; ++i) {
          const std::uint32_t u = (std::uint32_t)src[i] << 16;
          float x;
          std::memcpy(&x, &u, 4);
          float unit = x * ps + pm;             // -> [0, 1]
          if (unit < 0.0f) { unit = 0.0f; }
          if (unit > 1.0f) { unit = 1.0f; }
          float v = std::round(unit * 255.0f);
          if (v < 0.0f) { v = 0.0f; }
          if (v > 255.0f) { v = 255.0f; }
          dst[i] = (std::uint8_t)v;
        }
      }
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign("frame",
                                      FlexData::make_int((std::int64_t)f));
      sb.as_object().insert_or_assign("frames",
                                      FlexData::make_int((std::int64_t)F));
      sb.as_object().insert_or_assign("fps", FlexData::make_real(fps));
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
      if (o.contains("fps") && o.at("fps").as_real(0.0) > 0.0) {
        fps = o.at("fps").as_real(fps);
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
        out->dtype = TensorBeat::DType::U8;
        out->shape = {3, c.height, c.width};
        out->resize_contiguous((std::size_t)3 * plane);
        std::uint8_t* op = out->as_u8();
        for (int ch = 0; ch < 3; ++ch) {
          // A family with fewer than 3 channels comes out greyscale
          // rather than read past the end of its own chunk.
          const int sc = (ch < c.channels) ? ch : (c.channels - 1);
          const float* src =
              c.rgb + ((std::size_t)sc * c.n + k) * plane;
          std::uint8_t* dst = op + (std::size_t)ch * plane;
          for (std::size_t i = 0; i < plane; ++i) {
            float v = std::round((src[i] + 1.0f) * 0.5f * 255.0f);
            if (!(v > 0.0f)) { v = 0.0f; }        // also catches NaN
            if (v > 255.0f) { v = 255.0f; }
            dst[i] = (std::uint8_t)v;
          }
        }
        // Video stamps {frame, frames, fps}; an image VAE stamps nothing,
        // which is the split the built-in branches already draw.
        if (is_video) {
          FlexData sb = FlexData::make_object();
          sb.as_object().insert_or_assign(
              "frame", FlexData::make_int((std::int64_t)(c.frame0 + k)));
          sb.as_object().insert_or_assign(
              "frames", FlexData::make_int((std::int64_t)c.frames_total));
          sb.as_object().insert_or_assign("fps", FlexData::make_real(fps));
          out->sideband = std::move(sb);
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
    if (!_wan_vae) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): Wan VAE not loaded; skipping", this->id()));
      co_return;
    }
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
    if (Cz != _wan_vae->config().z_dim || T <= 0 || h8 <= 0 || w8 <= 0) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): latent [{}, {}, {}, {}] does not match z_dim "
          "{}; skipping", this->id(), Cz, T, h8, w8,
          _wan_vae->config().z_dim));
      co_return;
    }
    // The producer knows the clip's rate; the config is the fallback for a
    // latent that arrived from somewhere else (a file, a test).
    double fps = _fps;
    if (tbp->sideband.is_object()) {
      FlexData sb = tbp->sideband;        // as_object() is a view: keep it
      auto o = sb.as_object();
      if (o.contains("fps") && o.at("fps").as_real(0.0) > 0.0) {
        fps = o.at("fps").as_real(fps);
      }
    }
    const int F = genai::MetalWanVae::video_frames(T);
    const int H = h8 * 8, W = w8 * 8;

    const std::size_t nz = (std::size_t)Cz * T * h8 * w8;
    metal_compute::SharedBuffer z = mc->make_shared_buffer(nz * 2);
    { auto* d = static_cast<_Float16*>(z.contents());
      const float* s = tbp->as_f32();
      for (std::size_t i = 0; i < nz; ++i) { d[i] = (_Float16)s[i]; } }
    metal_compute::SharedBuffer zw = _wan_vae->unwhiten(z, T, h8, w8);
    if (zw.empty()) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): un-whiten failed; skipping", this->id()));
      co_return;
    }

    // The VAE streams its output a chunk at a time so a long clip never
    // exists in f16 all at once. The beat boundary cannot stream with it:
    // ctx.write is a coroutine and the sink is a plain callback, so the
    // frames are converted to U8 in the sink -- where the f16 chunk is
    // still alive and is then dropped -- and written out afterwards. What
    // is held is the clip as planar U8 RGB, which is a THIRD of the f16 it
    // replaces (81 frames of 480x832 = 97 MB), not the f16 working set the
    // chunking exists to bound.
    std::vector<std::unique_ptr<TensorBeatPayload>> frames;
    frames.reserve((std::size_t)F);
    auto sink = [&](const metal_compute::SharedBuffer& rgb, int frame0,
                    int n) -> bool {
      const auto* rp = static_cast<const _Float16*>(rgb.contents());
      const std::size_t per = (std::size_t)3 * H * W;
      for (int k = 0; k < n; ++k) {
        auto out = std::make_unique<TensorBeatPayload>();
        out->dtype = TensorBeat::DType::U8;
        out->shape = {3, H, W};
        out->resize_contiguous(per);
        std::uint8_t* op = out->as_u8();
        // The VAE hands back [3, n, H, W] (channel-first over the chunk),
        // so frame k is a stride into each channel plane, not a contiguous
        // block.
        for (int c = 0; c < 3; ++c) {
          const _Float16* src =
              rp + ((std::size_t)c * n + k) * (std::size_t)H * W;
          std::uint8_t* dst = op + (std::size_t)c * H * W;
          for (std::size_t i = 0; i < (std::size_t)H * W; ++i) {
            float v = std::round(((float)src[i] + 1.0f) * 0.5f * 255.0f);
            if (v < 0.0f) { v = 0.0f; }
            if (v > 255.0f) { v = 255.0f; }
            dst[i] = (std::uint8_t)v;
          }
        }
        FlexData sb = FlexData::make_object();
        sb.as_object().insert_or_assign("frame",
                                        FlexData::make_int((std::int64_t)(frame0 + k)));
        sb.as_object().insert_or_assign("frames", FlexData::make_int((std::int64_t)F));
        sb.as_object().insert_or_assign("fps", FlexData::make_real(fps));
        out->sideband = std::move(sb);
        forward_model_name_(*tbp, *out);
        frames.push_back(std::move(out));
      }
      return true;
    };
    {
      const std::size_t px =
          (std::size_t)F * (std::size_t)H * (std::size_t)W;
      revise_decode_arena_(px * 3 * 2 + px * 3);
    }
    std::string derr;
    bool ok = false;
    {
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae,
                         kPerfLlmVaeBegin,
                         (std::uint64_t)F * (std::uint64_t)H * W);
      ok = _wan_vae->decode(zw, T, h8, w8, sink, &derr);
    }
    if (!ok) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): video decode failed ({}); skipping",
          this->id(), derr.empty() ? "unknown error" : derr));
      co_return;
    }
    session()->log_debug(fmt(
        "VaeDecodeStage('{}'): decoded [{}, {}, {}, {}] -> {} frames "
        "[3, {}, {}] @ {:.3f} fps", this->id(), Cz, T, h8, w8,
        frames.size(), H, W, fps));
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


  if (_family == "mage") {
    if (!_mage_vae) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): MageVAE not loaded; skipping", this->id()));
      co_return;
    }
    const int Cz = (int)tbp->shape[0];
    const int h = (int)tbp->shape[1], w = (int)tbp->shape[2];
    const int P = _mage_vae->config().patch;
    if (Cz != _mage_vae->config().latent_channels || h <= 0 || w <= 0) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): latent [{}, {}, {}] does not match "
          "latent_channels {}; skipping", this->id(), Cz, h, w,
          _mage_vae->config().latent_channels));
      co_return;
    }
    const std::size_t nz = (std::size_t)Cz * h * w;
    metal_compute::SharedBuffer z = mc->make_shared_buffer(nz * 2);
    if (z.empty()) { co_return; }
    { auto* d = static_cast<_Float16*>(z.contents());
      const float* s = tbp->as_f32();
      for (std::size_t i = 0; i < nz; ++i) { d[i] = (_Float16)s[i]; } }
    std::string derr;
    metal_compute::SharedBuffer rgb;
    {
      PerfAuxScope _perf(session(), kPerfLaneLLM, kGvidLlmVae,
                         kPerfLlmVaeBegin, (std::uint64_t)(h * P) * (w * P));
      rgb = _mage_vae->decode(z, h, w, &derr);
    }
    if (rgb.empty()) {
      session()->warn(fmt(
          "VaeDecodeStage('{}'): MageVAE decode failed ({}); skipping",
          this->id(), derr.empty() ? "unknown error" : derr));
      co_return;
    }
    const int H = h * P, W = w * P;
    const std::size_t n = (std::size_t)3 * H * W;
    auto out = std::make_unique<TensorBeatPayload>();
    out->dtype = TensorBeat::DType::U8;
    out->shape = {3, H, W};
    out->resize_contiguous(n);
    const auto* rp = static_cast<const _Float16*>(rgb.contents());
    std::uint8_t* op = out->as_u8();
    for (std::size_t i = 0; i < n; ++i) {
      // MetalMageVae::decode does NOT clamp (the reference clamps at the PIL
      // conversion, which is this).
      float v = std::round(((float)rp[i] + 1.0f) * 0.5f * 255.0f);
      if (v < 0.0f) { v = 0.0f; }
      if (v > 255.0f) { v = 255.0f; }
      op[i] = (std::uint8_t)v;
    }
    ++_images_emitted;
    session()->log_debug(fmt(
        "VaeDecodeStage('{}'): MageVAE decoded + emitted image #{} [3, {}, {}]",
        this->id(), _images_emitted, H, W));
    forward_model_name_(*tbp, *out);
    if (_unload_idle) { unload_vae_(); }
    co_await ctx.write(0, std::move(out));
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
  session()->log_debug(fmt(
      "VaeDecodeStage('{}'): beat received, latent [{}, {}, {}] -> image "
      "[{}, {}]", this->id(), Cz, h8, w8, h8 * 8, w8 * 8));
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
  publish_image_arena_(w8 * 8, h8 * 8);
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
                       (std::uint64_t)(h8 * 8) * (w8 * 8));
    rgb = _vae->decode(zw, h8, w8, &derr);
  }
  if (rgb.empty()) {
    session()->warn(fmt(
        "VaeDecodeStage('{}'): decode failed ({}); skipping", this->id(),
        derr.empty() ? "unknown error" : derr));
    co_return;
  }

  // f16 [3,H,W] in [-1,1] -> planar U8 RGB (x+1)/2*255, rounded + clamped.
  const int H = h8 * 8, W = w8 * 8;
  const std::size_t n = (std::size_t)3 * H * W;
  auto out = std::make_unique<TensorBeatPayload>();
  out->dtype = TensorBeat::DType::U8;
  out->shape = {3, H, W};
  out->resize_contiguous(n);
  const auto* rp = static_cast<const _Float16*>(rgb.contents());
  std::uint8_t* op = out->as_u8();
  for (std::size_t i = 0; i < n; ++i) {
    float v = ((float)rp[i] + 1.0f) * 0.5f * 255.0f;
    v = std::round(v);
    if (v < 0.0f) { v = 0.0f; }
    if (v > 255.0f) { v = 255.0f; }
    op[i] = (std::uint8_t)v;
  }
  ++_images_emitted;
  session()->log_debug(fmt(
      "VaeDecodeStage('{}'): decoded + emitted image #{} planar U8 RGB "
      "[3, {}, {}]", this->id(), _images_emitted, H, W));
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
