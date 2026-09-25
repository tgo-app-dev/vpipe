#ifndef VPIPE_GENERATIVE_MODELS_IMAGE_MODEL_REGISTRY_H
#define VPIPE_GENERATIVE_MODELS_IMAGE_MODEL_REGISTRY_H

#include "common/flex-data.h"
#include "generative-models/gen-input.h"
#include "pipeline/memory-plan.h"
#include "pipeline/resource-plan.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
class SessionContextIntf;
namespace metal_compute { class MetalCompute; }
}

namespace vpipe::genai {

// An out-of-tree IMAGE model family, for `generate-image`.
//
// The counterpart to video-model-registry.h, drawn at the same place and
// for the same reason -- and this file is deliberately its mirror rather
// than a second design, so a family that has shipped one can read the
// other without re-learning anything.
//
// WHY IT IS NOT A VIRTUAL BASE OVER THE DiTs. generate-image holds five
// built-in denoisers side by side and dispatches on a `_class_name`
// string, because their PER-STEP forwards do not agree: Krea-2 takes a
// packed [img_seq, z*4] latent with one timestep, Qwen-Image-Edit runs a
// dual stream, FLUX.2 appends multi-reference tokens with their own RoPE
// band, Boogu is a NextDiT. An interface wide enough for all five
// describes none of them.
//
// So the seam is one level up, at the GENERATION. Every image family
// answers the same question -- given conditioning, geometry, a step
// count and a seed, produce a latent -- and a family owns its whole
// denoise loop below that: its scheduler, its guidance rule, its
// patchify, its residency policy. The stage owns the ports, the beats,
// the geometry bookkeeping and the sampler/scheduler SELECTIONS, which
// it hands down unparsed.
//
// IT EMITS A LATENT, NOT PIXELS, and that is the cleanest part of this
// boundary: `generate-image`'s oport0 is an f32 latent that `vae-decode`
// un-whitens and decodes. So a family that also registers a VAE family
// (see vae-model-registry.h) needs no stage of its own for either half.
//
// The registry is consulted BEFORE the built-in flux2 / krea2 /
// qwen-image-edit / boogu-image / mage-flow dispatch, mirroring how
// VideoModelRegistry sits in front of generate-video's two built-ins and
// how ModelExecRegistry sits in front of LoadedLanguageModel's arch
// if-chain. The built-in families are unchanged and unregistered: this
// ADDS a path, it does not reroute the existing ones. Converting them
// would be a large refactor of a stage whose family branching reaches
// into load, unload, resource declaration, LoRA slots and the
// model-config re-parse -- and it would buy the built-ins nothing.

// One generation's inputs. Every pointer is BORROWED from a beat that
// outlives the call; nothing here is owned.
struct ImageGenRequest {
  // ---- geometry, after the stage applied the family's size rule -----
  int           height = 0;      // pixels, already through size_grid()
  int           width  = 0;
  int           steps  = 0;
  std::uint64_t seed   = 0;
  // Classifier-free guidance. 1 means no CFG and the stage will not have
  // read a negative beat; above 1 with `neg` present is the second pass.
  double        guidance_scale = 1.0;

  // ---- conditioning (iport0), and its negative (iport1) -------------
  //
  // FAMILY-SHAPED AND FAMILY-TYPED, exactly as it arrives from the
  // diffusion-conditioner: the stage does not interpret it. `rows` is
  // the token count and `dim` the encoder width; `elem_size` is 2 for
  // f16/bf16 and 4 for f32, and `is_bf16` separates the two 2-byte
  // cases. A family reads what its own conditioner emitted, which it
  // also wrote -- pairing a conditioner and a DiT from different
  // families is a graph error the stage cannot catch and does not try.
  const void* cond      = nullptr;
  int         cond_rows = 0;
  int         cond_dim  = 0;
  int         cond_elem_size = 2;
  bool        cond_is_bf16   = false;
  // Null when the graph wired no negative, or when guidance_scale is 1.
  const void* neg      = nullptr;
  int         neg_rows = 0;

  // Anything the conditioner sent alongside the tokens -- grid shapes,
  // token counts per reference, a family's own bookkeeping. Null when
  // the conditioner sent none.
  const FlexData* cond_sideband = nullptr;

  // ---- reference latents (iport5 / iport6) --------------------------
  //
  // Channel-first f32, the format vae-encode emits. What they MEAN is
  // the family's: FLUX.2 embeds each as multi-reference conditioning
  // with its own RoPE band, Krea-2 uses the first as an img2img init
  // mixed in at the strength-selected sigma. A family that has neither
  // ignores them, and a graph that wired neither leaves both null.
  const float*     ref_latent0 = nullptr;
  std::vector<int> ref0_shape;
  const float*     ref_latent1 = nullptr;
  std::vector<int> ref1_shape;

  // ---- the family's own knobs (iport2 model_config), UNPARSED -------
  // Passed down exactly as the config source emitted it, so a knob a
  // family adds later needs no change in generate-image. Null / non-
  // object means the graph wired no config source and the family runs
  // its documented defaults.
  const FlexData* model_config = nullptr;

  // Sampler / scheduler selections (iport3 / iport4) when the graph made
  // one, else null -- a family reads what it understands and ignores the
  // rest, the way the built-ins do.
  const FlexData* sampler_spec   = nullptr;
  const FlexData* scheduler_spec = nullptr;

  // Returns false to ABORT the generation (the pipeline is stopping).
  //
  // Call it at the END of each step with a ONE-BASED index -- `step` is
  // the step that just finished and `total` the count the family's own
  // scheduler settled on, which need not be the configured `steps`. A
  // family that never calls this cannot be interrupted and reports
  // nothing, which on a 12B DiT is a Stop that takes minutes and a bar
  // that never moves.
  std::function<bool(int step, int total)> progress;

  // The same, per BLOCK of the transformer stack, and it exists for
  // resolution: one step of a 12B stack that may be streaming its
  // weights leaves a step-granular bar still for the whole time
  // anything is happening. `done` is blocks entered and `total` the
  // depth; false ABORTS, so a family that calls it gets cancellation
  // inside a step as well as between steps.
  std::function<bool(int done, int total)> block_progress;

  // Each denoise step's latent, LIVE, for a graph watching the picture
  // form (generate-image's oport1). Optional in both directions: null
  // when nothing downstream is listening, and a family that cannot
  // produce an intermediate simply never calls it. Same shape and
  // meaning as ImageGenResult::latent.
  std::function<void(int step, int total, const float* latent,
                     const std::vector<int>& shape)> step_latent;

  // THE CROSS-FAMILY ACCELERATION SETTINGS, one pointer that stays one
  // pointer. See generative-models/shared/accel-settings.h for the keys,
  // the readers, and why this is a bag rather than fields: typed fields
  // put the host's vocabulary into the plugin ABI, so adding a tier
  // invalidated every family's binary including the ones implementing
  // none of them.
  //
  // Owned by the stage and valid for the call. Null is legitimate and
  // reads as every default.
  const FlexData* accel = nullptr;

  // ---- the growth seam ----------------------------------------------
  //
  // NEW INPUTS GO HERE, NOT IN A NEW FIELD. See gen-input.h for the
  // argument; in short, every typed field above was a layout change and
  // an invalidated binary for families that do not use it.
  //
  // `input` looks up anything with a SHAPE by name; `extras` carries
  // scalars the host wants to state and no family has to read. Both are
  // legitimately absent -- a null `input` and a null `extras` are what a
  // graph that wired neither produces, and a family must work then.
  vpipe::genai::NamedInputFn input;
  const FlexData*            extras = nullptr;

  // Named OUTPUTS, mid-generation -- the input seam's mirror, and where
  // the next thing a family hands back goes. See gen-input.h; the first
  // name is kOutputPreviewX0 (a live preview's clean estimate, shaped
  // like ImageGenResult::latent). Always installed by the host.
  vpipe::genai::OutputWantedFn output_wanted;
  vpipe::genai::NamedOutputFn  output;
};

// What a generation produced: the latent `vae-decode` will un-whiten.
struct ImageGenResult {
  std::vector<float> latent;   // f32, row-major over `shape`
  std::vector<int>   shape;    // channel-first [z, H/r, W/r]
  // WHAT THE FAMILY WANTS SAID BACK, for anything this struct has no
  // field for -- realized sparsity, a chosen variant, a per-step
  // statistic. The stage forwards it onto the beat's sideband and reads
  // none of it, so a family adding one costs nobody a rebuild.
  FlexData sideband;
};

// ONE resident checkpoint of a family. Built by ImageModelFamily::load()
// and held by the stage for as long as that checkpoint is the model.
class ImageGenerator {
public:
  virtual ~ImageGenerator() = default;

  // The latent geometry this checkpoint emits, for the log line the
  // stage writes before it spends minutes generating. The SHAPE the
  // stage publishes is the one `generate` returns, never these -- a
  // family that mispredicts its own geometry should produce a confusing
  // log, not a mislabelled beat.
  virtual int latent_channels() const = 0;
  virtual int spatial_compression() const = 0;

  // Run one generation. False means it warned and produced nothing --
  // the stage then writes no beat, which is what every refusal in this
  // graph does. Never throws across this boundary; a family that lets an
  // exception out takes the host down with it.
  virtual bool generate(const ImageGenRequest& req, ImageGenResult* out) = 0;

  // Drop what can be dropped between requests. Called when the stage's
  // idle-unload policy resolved to `always`, or when a peer needs the
  // room. A family with nothing to release does nothing.
  virtual void release_idle() {}

  // Bytes this generator is currently holding, for the model manager's
  // session-wide view. 0 when the family cannot answer cheaply -- but a
  // family whose weights are invisible to a WeightSet SHOULD answer, or
  // the graph sizes itself against a checkpoint it cannot see. See
  // docs/MODEL-MEMORY.md, "Declarations".
  virtual std::uint64_t resident_bytes() const { return 0; }
};

// Everything a family's `load` receives. Mirrors VideoModelCreateArgs.
struct ImageModelCreateArgs {
  std::string                  root;        // the resolved checkpoint dir
  std::string                  model_type;  // the models-DB record's hint,
                                            // empty when there is none
  metal_compute::MetalCompute* metal   = nullptr;
  const SessionContextIntf*    session = nullptr;
  // The stage's residency verdict, so a family need not re-derive it:
  // true means "the box is tight, stream your blocks". A family is free
  // to disagree -- it knows its own sizes -- but it should say so.
  bool                         prefer_streaming = false;

  // The `model_config` beat, for the keys a family must answer BEFORE it
  // has a model -- null when no config stage is wired.
  //
  // ImageGenRequest carries this too, and that is the one to read for
  // anything per-generation. This copy exists for the strictly smaller
  // set of keys that decide WHAT TO LOAD, which the request sees too
  // late: by then the checkpoint is chosen and, for a streamed DiT,
  // irrevocably so. A root holding more than one DiT -- a quantized pack
  // beside a bf16 one -- is the case that needs it.
  //
  // Borrowed, and only for the duration of the call: the stage owns the
  // FlexData. A family that wants it later must copy it.
  const FlexData*              model_config = nullptr;

  // THE CROSS-FAMILY ACCELERATION SETTINGS, as the graph asked for them,
  // and the same object ImageGenRequest carries. A family that decides
  // at LOAD -- building a kernel variant, sizing int8 scratch -- reads
  // them here; one that decides per generation reads them there.
  //
  // The host applies NONE of them on a family's behalf; nothing outside
  // a family touches its forward. They are here because every DiT in
  // this tree, in-tree or not, runs the same steel flash kernel and the
  // same GEMM kernels, so a plugin family that wants them has the same
  // decisions to make and no way to learn what the graph asked without
  // being told.
  //
  // accel-settings.h gives the keys and the readers; each tier's own
  // header gives a `config_from_flex` returning its typed Config, so a
  // family that wants one writes
  //
  //   const sage::Config c = sage::config_from_flex(args.accel);
  //
  // and a family that wants none includes nothing at all.
  const FlexData*              accel = nullptr;

  // The picture the graph INTENDS to make, already through the family's
  // own size_grid. 0 when the stage could not settle it.
  //
  // For families that must size something at LOAD against a term that
  // scales with the BEAT -- a pinned block prefix against the activation
  // scratch, above all. It is the PLAN, not a promise: the first request
  // may differ, and a family that cares must re-check.
  int width = 0, height = 0;

  // ---- what the graph pointed at, and the growth seam ---------------
  //
  // The DiT the graph NAMED, resolved, when it named one -- the
  // `dit_dir` config, which points a run at a quantized or fused DiT
  // beside the checkpoint rather than the one inside it. EMPTY means
  // the graph named none and the family finds its own weights under
  // `root`, which is what every family did before this existed.
  //
  // It is a typed field rather than an `extras` key because it is not
  // an optional capability: it is the same resolved path the built-in
  // families have always been handed, and a family that ignores it
  // silently loads weights the operator did not ask for.
  //
  // NOTE WHAT IT DOES NOT REACH. The planning calls -- declare_resources,
  // declare_holdings, latent_bytes -- are asked about the ROOT, and are
  // asked before this struct exists. So a graph that names a smaller
  // quantized DiT is sized against the checkpoint's own, which
  // over-declares. That is the safe direction: a streaming model revises
  // down once it knows what it holds, where under-declaring is a box
  // admitted to a graph it cannot run.
  std::string dit_dir;

  // NEW ARGUMENTS GO IN HERE, NOT IN A NEW FIELD. See gen-input.h.
  // Null is what a host with nothing to add produces.
  const FlexData* extras = nullptr;
};

// An image model FAMILY: process-wide, stateless, one per architecture.
class ImageModelFamily {
public:
  virtual ~ImageModelFamily() = default;

  // The family tag, as it appears in a model-config beat's
  // `model_family` key and in this stage's logs. Must match what the
  // family's ModelConfigSourceStage stamps, or every config wired to it
  // is reported as a mismatch and dropped.
  virtual std::string_view tag() const noexcept = 0;

  // Does `root` hold a checkpoint of this family? `model_type` is what
  // the models DB recorded, which the DIRECTORY may not be able to say.
  // Must be CHEAP -- it runs for every family on every model resolve --
  // and must be SURE: claiming a checkpoint that is not yours loads at
  // full cost and computes nonsense, which is the worst failure this
  // stage has.
  virtual bool claims(const std::string& root,
                      const std::string& model_type) const = 0;

  // The size grid this family can tile, as (height, width) multiples --
  // the VAE's spatial stride times the DiT's patch. The stage rounds the
  // requested size UP to it, so a caller asks for the picture it wants
  // rather than one derived from the checkpoint's compression ratios.
  //
  // Asked of the FAMILY so it can be answered before a 12B DiT loads,
  // and the stage REPORTS any change rather than refusing: a graph
  // should not have to be re-authored to change families.
  virtual void size_grid(const std::string& /*root*/, int* gh,
                         int* gw) const
  {
    if (gh != nullptr) { *gh = 16; }
    if (gw != nullptr) { *gw = 16; }
  }

  // The VAE's spatial compression, for the one thing the stage must
  // compute before the model exists: the output size INFERRED from a
  // reference latent when the graph set neither width nor height. 8 for
  // Krea-2 and Qwen-Image-Edit, 16 for FLUX.2 and Mage-Flow. 0 means
  // "cannot say", and the stage then falls back to its default size
  // instead of guessing a family's ratio.
  virtual int latent_scale(const std::string& /*root*/) const { return 0; }

  // What loading this checkpoint will take, for the planning phase that
  // runs before any stage initialises. Weight claims (see
  // stages/model-memory.h) are the usual answer; empty means the family
  // declines to declare, which makes it invisible to peers sizing
  // themselves -- see docs/MODEL-MEMORY.md.
  virtual std::vector<ResourceClaim>
  declare_resources(const std::string& /*root*/) const { return {}; }

  // The same checkpoints again, in the TOPOLOGICAL plan's terms rather
  // than the phase vocabulary's. Two ledgers, and a family has to answer
  // both -- see docs/MODEL-MEMORY.md, "Which ledger do I use?".
  //
  // Answer with `source` naming the checkpoint, `preload` what it costs
  // resident and `floor` the least it can be held at while still being
  // held. Leave `releases` and `reclaimable` alone: whether idle weights
  // are dropped is the STAGE's `unload_when_idle` policy, and the stage
  // stamps it onto whatever comes back from here.
  //
  // Empty -- the default -- is a hole in the plan, and a hole reads as
  // room that is not there.
  virtual std::vector<StageHolding>
  declare_holdings(const std::string& /*root*/) const { return {}; }

  // Bytes of the LATENT this family emits at this geometry, which the
  // stage declares as a payload alive from the denoise that writes it to
  // the decode that reads it. Asked of the family because the shape is
  // the family's: channel count and spatial compression both differ, and
  // a host substituting a built-in's formula would report a confident
  // number for the wrong model. 0 -- the default -- means "cannot say",
  // and the stage then declares nothing rather than something plausible.
  virtual std::size_t latent_bytes(const std::string& /*root*/,
                                   int /*width*/, int /*height*/) const
  {
    return 0;
  }

  // Build the generator. Null (after warning through `args.session`)
  // leaves the stage inert rather than taking the pipeline down.
  virtual std::unique_ptr<ImageGenerator>
  load(const ImageModelCreateArgs& args) = 0;
};

// Process-wide family set. Same singleton discipline as StageRegistry
// and VideoModelRegistry: a plugin MUST link the host libvpipe shared so
// it registers into THIS instance rather than forking a second one.
class ImageModelRegistry {
public:
  static ImageModelRegistry& get() noexcept;

  // Takes ownership. First-wins on `tag()`: a second family claiming a
  // tag already present is ignored and returns false, so two plugins
  // shipping the same model cannot silently shadow each other.
  bool add(std::unique_ptr<ImageModelFamily> f);

  // The first registered family that claims `root`, or null. Order is
  // registration order, which for plugins is load order -- so a
  // deliberately narrow `claims` is a family's own responsibility.
  // Never throws: a family whose `claims` throws is skipped and warned.
  ImageModelFamily* claim_for(const SessionContextIntf* session,
                              const std::string&        root,
                              const std::string&        model_type) const;

  ImageModelFamily* find(std::string_view tag) const noexcept;

  std::vector<ImageModelFamily*> all() const;

private:
  ImageModelRegistry() = default;

  mutable std::mutex                             _mu;
  std::vector<std::unique_ptr<ImageModelFamily>> _families;
};

}

#endif
