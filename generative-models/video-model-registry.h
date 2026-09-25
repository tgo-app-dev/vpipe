#ifndef VPIPE_GENERATIVE_MODELS_VIDEO_MODEL_REGISTRY_H
#define VPIPE_GENERATIVE_MODELS_VIDEO_MODEL_REGISTRY_H

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

// An out-of-tree VIDEO model family, for `generate-video`.
//
// WHY THIS IS NOT A VIRTUAL BASE OVER THE DiTs. generate-video holds its
// two built-in denoisers side by side rather than behind one base, and
// the reason is written down in its header: Wan takes a 4-D latent with
// one timestep, MiniMax-H3 takes a packed sequence with per-row
// timesteps, and an interface wide enough for both describes neither.
// That argument is about the PER-STEP forward, and it still holds -- so
// this interface is not drawn there.
//
// It is drawn one level up, at the GENERATION. Every video family
// answers the same question -- given conditioning, geometry, a step
// count and a seed, produce a latent video and (for the families that
// have one) a latent soundtrack -- and the built-in H3 branch already
// has exactly that shape internally (`run_h3_`). A family owns its whole
// denoise loop: its scheduler, its guidance rule, its residency policy,
// its patchify/unpatchify. The stage owns the ports, the beats and the
// geometry bookkeeping. Nothing about a step is in the contract, so a
// family whose step looks like nothing else here still fits.
//
// The registry is consulted BEFORE the built-in wan / minimax-h3
// dispatch, mirroring how LoadedLanguageModel consults ModelExecRegistry
// before its built-in arch if-chain (see model-exec-registry.h). The
// built-in families are unchanged and unregistered: this adds a path,
// it does not reroute the existing ones.

// One generation's inputs. Every pointer is BORROWED from a beat that
// outlives the call; nothing here is owned.
struct VideoGenRequest {
  // ---- geometry, after the stage applied the family's frame rule ----
  int           height = 0;      // pixels, multiple of 16
  int           width  = 0;
  int           frames = 0;      // already aligned by align_frames()
  double        fps    = 16.0;
  int           steps  = 0;
  std::uint64_t seed   = 0;

  // ---- conditioning (iport0 / iport1) -------------------------------
  // Raw bf16 rows as the diffusion-conditioner emitted them, plus the
  // beat's sideband -- which is where a family that splits one request
  // over several beats (H3's ref2va geometry) puts the plan.
  const void*     cond      = nullptr;   // bf16 [cond_rows, cond_dim]
  int             cond_rows = 0;
  int             cond_dim  = 0;
  const FlexData* cond_sideband = nullptr;
  // OPTIONAL negative conditioning for classifier-free guidance. Null on
  // a guidance-distilled family, or when the graph wired none -- a
  // family that needs one and did not get it must say so and refuse,
  // not silently generate unguided.
  const void*     neg       = nullptr;
  int             neg_rows  = 0;

  // ---- OPTIONAL second-modality conditioning (iport10) --------------
  // An AUDIO cross-attention context, for the families that generate a
  // soundtrack from its own projection of the caption rather than from
  // the video context. Null when the graph wired none, which for such a
  // family means video-only -- and a family that wants one must SAY so
  // and fall back, not invent a zero context, because a zero context is
  // conditioning that says something (silence) rather than nothing.
  //
  // Separate from `cond` because it is a different WIDTH from a
  // different projection: LTX-2.5 is 4096 video / 2048 audio out of the
  // same hidden states. Packing both into one beat would make the split
  // a convention rather than a type.
  const void*     audio_cond      = nullptr;   // bf16 [rows, dim]
  int             audio_cond_rows = 0;
  int             audio_cond_dim  = 0;

  // ---- optional conditioning latents (iport5 / iport6) --------------
  // f32, as a vae-encode emitted them (already whitened). `ref` is the
  // first-frame / image-to-video latent, `ref_last` the closing anchor.
  // Shapes are the encoder's, NOT normalised here: a family that wants
  // [z, T, h, w] and got something else knows its own rule.
  const float*            ref = nullptr;
  std::vector<int>        ref_shape;
  const float*            ref_last = nullptr;
  std::vector<int>        ref_last_shape;

  // ---- optional reference rows (iport7 / iport8) --------------------
  const float*     ref_video_rows   = nullptr;
  int              n_ref_video_rows = 0;
  int              ref_video_dim    = 0;
  const float*     ref_audio_rows   = nullptr;
  int              n_ref_audio_rows = 0;
  int              ref_audio_dim    = 0;

  // ---- the family's own knobs (iport9), UNPARSED --------------------
  // Passed down exactly as the config source emitted it, so a knob added
  // later needs no change in generate-video. Null / non-object means the
  // graph wired no config source and the family runs its documented
  // defaults. See stages/model-config-source.h.
  const FlexData* model_config = nullptr;

  // Sampler / scheduler selections (iport3 / iport4) when the graph made
  // one, else empty -- a family reads what it understands and ignores
  // the rest, the way the built-ins do.
  const FlexData* sampler_spec   = nullptr;
  const FlexData* scheduler_spec = nullptr;

  // Returns false to ABORT the generation (the pipeline is stopping).
  //
  // Call it at the END of each step with a ONE-BASED index, which is what
  // the stage's progress bar re-syncs on -- `step` is the step that just
  // finished, and `total` is the count the family's own scheduler settled
  // on, which need not be the configured `steps`. A family that never
  // calls this cannot be interrupted and reports nothing, which on a 22B
  // model is a Stop that takes minutes and a bar that never moves.
  std::function<bool(int step, int total)> progress;

  // The same, per BLOCK of the transformer stack. Optional, and the
  // reason it exists is resolution: a step here is one forward of a 22B
  // stack that may be streaming its weights, so a step-granular bar sits
  // still for the entire time anything is happening. A stack of 48
  // blocks gives ~50x the resolution for a callback that costs a
  // compare. The built-in DiTs report at this granularity through
  // set_block_progress; this is the same thing reachable from a plugin.
  //
  // `done` is blocks entered so far and `total` the stack depth. False
  // ABORTS, so a family that calls this also gets cancellation inside a
  // step rather than only between steps.
  std::function<bool(int done, int total)> block_progress;

  // The same acceleration settings VideoModelCreateArgs carries, repeated
  // per generation. A family that decides at LOAD (building a kernel
  // variant, sizing int8 scratch) reads them there; one that decides per
  // forward reads them here, and the two are the same object -- the
  // stage fills both from one config.
  //
  // ONE POINTER, and it stays one pointer. See
  // generative-models/shared/accel-settings.h for the keys, the readers
  // and why this is a bag: typed fields here put the host's vocabulary
  // into the plugin ABI, so adding a tier invalidated every family's
  // binary including the ones that implement none of them.
  //
  // Owned by the stage and valid for the call. Null is legitimate and
  // reads as every default.
  const FlexData* accel = nullptr;

  // ---- the growth seam ----------------------------------------------
  //
  // NEW INPUTS GO HERE, NOT IN A NEW FIELD. See gen-input.h for the
  // argument, and read this struct's own typed fields as the evidence:
  // `cond`, then `neg`, then `audio_cond`, then `ref`, then `ref_last`,
  // then the two reference-row triples. Each arrived with a capability,
  // each moved this layout, and each invalidated every family binary
  // including the ones that ignore it.
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
  // like VideoGenResult::video). Always installed by the host.
  vpipe::genai::OutputWantedFn output_wanted;
  vpipe::genai::NamedOutputFn  output;
};

// What a generation produced. A family that generates no audio simply
// leaves the audio fields empty -- an unconnected oport1 is not an error.
struct VideoGenResult {
  std::vector<float> video;        // f32, row-major over `video_shape`
  std::vector<int>   video_shape;  // [z, T, H/r, W/r]
  std::vector<float> audio;        // f32 [stereo, channels, latents]
  std::vector<int>   audio_shape;
  double latents_per_second = 0.0; // stamped on the audio beat
  // WHAT THE FAMILY WANTS SAID BACK, for anything this struct has no
  // field for -- realized sparsity, a chosen variant, a per-step
  // statistic. The stage forwards it onto the beat's sideband and reads
  // none of it, so a family adding one costs nobody a rebuild.
  FlexData sideband;
};

// ONE resident checkpoint of a family. Built by VideoModelFamily::load()
// and held by the stage for as long as that checkpoint is the model.
class VideoGenerator {
public:
  virtual ~VideoGenerator() = default;

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
  virtual bool generate(const VideoGenRequest& req, VideoGenResult* out) = 0;

  // Drop what can be dropped between requests. Called when the stage's
  // idle-unload policy resolved to `always`, or when a peer needs the
  // room. A family with nothing to release does nothing.
  virtual void release_idle() {}

  // Bytes this generator is currently holding, for the model manager's
  // session-wide view. 0 when the family cannot answer cheaply -- but a
  // family whose weights are invisible to a WeightSet (because it read
  // them uncached, like every LM does) SHOULD answer, or the graph sizes
  // itself against a checkpoint it cannot see. See
  // docs/MODEL-MEMORY.md, "Declarations".
  virtual std::uint64_t resident_bytes() const { return 0; }
};

// Everything a family's `load` receives. Mirrors ModelExecCreateArgs.
struct VideoModelCreateArgs {
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
  // VideoGenRequest carries this too, and that is the one to read for
  // anything per-generation. This copy exists for the strictly smaller
  // set of keys that decide WHAT TO LOAD, which the request sees too
  // late: by then the checkpoint is chosen and, for a streamed DiT,
  // irrevocably so.
  //
  // The case that forced it: a family whose root holds more than one DiT
  // (LTX-2.5 ships `distilled` and `dev`, and a quantized pack may sit
  // beside either). Its config stage declared a `variant` key, the beat
  // carried it, and the family could not see it at load -- so selecting
  // `dev` silently loaded `distilled`. A key that names what to load has
  // to arrive before the load.
  //
  // Borrowed, and only for the duration of the call: the stage owns the
  // FlexData. A family that wants it later must copy it.
  const FlexData*              model_config = nullptr;

  // THE CROSS-FAMILY ACCELERATION SETTINGS, as the graph asked for them.
  //
  // These are not built-in knobs the host applies on a family's behalf --
  // nothing outside the family touches its forward. They are here
  // because every DiT in this tree, in-tree or not, runs the same steel
  // flash kernel and the same GEMM kernels, so a plugin family that
  // wants them has the same three decisions to make and no way to learn
  // what the graph asked without being told.
  //
  // A family that implements none of this ignores them, which is what
  // the defaults mean. A family that implements SOME should say which
  // in its own log line: silence is indistinguishable from a knob that
  // did nothing, and the numbers get believed either way.
  //
  // Every tier here is DRIVEABLE from a plugin, and each one costs
  // exactly the header it needs and no more: `i8_gemm` through
  // I8GemmContext, `sage_attn` through MetalSageAttention, `sol_attn`
  // through MetalSolAttention -- which takes the head-major q/k/v an
  // unfused attention already has and needs nothing else from the family
  // but the head count, the row count and the scale.
  //
  // accel-settings.h gives the keys and the readers; each tier's own
  // header gives a `config_from_flex` that returns its typed Config, so
  // a family that wants one writes
  //
  //   const sol::Config c = sol::config_from_flex(args.accel);
  //
  // and a family that wants none includes nothing at all. That is the
  // point of the bag: the cost of a tier falls on its users.
  //
  // Sol's Config carries a SINK that is NOT in the bag and cannot be:
  // the rows the routing must read exactly are where a family's own
  // sequence stops being the modality being summarised, and only the
  // family knows that. H3 fills it from its packed layout; a per-stream
  // self-attention over one modality leaves it at zero.
  const FlexData*              accel = nullptr;

  // The clip the graph INTENDS to generate, already through the family's
  // own align_frames / size_grid. 0 when the stage could not settle it.
  //
  // For families that must size something at LOAD against a term that
  // scales with the BEAT -- a pinned block prefix against the activation
  // scratch, above all. Sizing that against a constant is wrong at every
  // geometry but one, and on a bounded box it is the difference between
  // a prefix that fits and one that evicts itself.
  //
  // It is the PLAN, not a promise: the first request may differ, and a
  // family that cares must re-check. What it buys is a load-time
  // decision made against the right order of magnitude instead of none.
  int width = 0, height = 0, frames = 0;

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

// A video model FAMILY: process-wide, stateless, one per architecture.
class VideoModelFamily {
public:
  virtual ~VideoModelFamily() = default;

  // The family tag, as it appears in a model-config beat's
  // `model_family` key and in this stage's logs ("ltx-2.5"). Must match
  // what the family's ModelConfigSourceStage stamps, or every config
  // wired to it is reported as a mismatch and dropped.
  virtual std::string_view tag() const noexcept = 0;

  // Does `root` hold a checkpoint of this family? `model_type` is what
  // the models DB recorded, which the DIRECTORY may not be able to say
  // (two records can share one local_path). Must be CHEAP -- it runs for
  // every family on every model resolve -- and must be SURE: claiming a
  // checkpoint that is not yours loads at full cost and computes
  // nonsense, which is the worst failure this stage has.
  virtual bool claims(const std::string& root,
                      const std::string& model_type) const = 0;

  // Round `frames` up to a count this checkpoint's VAE can actually
  // chunk (Wan's 4k+1, H3's 17n+5, LTX-2.5's 8k+1). Asked of the FAMILY
  // rather than the generator because the stage settles geometry when it
  // resolves the checkpoint, which is before anything is loaded -- and a
  // family that had to load 22B of weights to answer "how many frames"
  // would make every graph pay for the question. Returning `frames`
  // unchanged means any count is legal. The stage REPORTS the change;
  // rejecting instead would mean a graph could not change families
  // without being re-authored.
  virtual int align_frames(const std::string& /*root*/, int frames) const
  {
    return frames;
  }

  // The frame SIZE grid this family can tile, as (height, width)
  // multiples -- the VAE's spatial stride times the DiT's patch. The
  // stage rounds the requested size UP to it, so a caller asks for the
  // picture it wants rather than for one it had to derive from the
  // checkpoint's compression ratios.
  //
  // Same contract, and the same reasoning, as align_frames above: asked
  // of the FAMILY so it can be answered before 22B of weights load, and
  // reported rather than rejected so a graph can change families without
  // being re-authored. (0, 0) -- the default -- means any size is legal.
  virtual void size_grid(const std::string& /*root*/, int* gh, int* gw) const
  {
    if (gh != nullptr) { *gh = 0; }
    if (gw != nullptr) { *gw = 0; }
  }

  // What loading this checkpoint will take, for the planning phase that
  // runs before any stage initialises. Weight claims (see
  // stages/model-memory.h) are the usual answer; empty means the family
  // declines to declare, which makes it invisible to peers sizing
  // themselves -- see docs/MODEL-MEMORY.md.
  virtual std::vector<ResourceClaim>
  declare_resources(const std::string& /*root*/) const { return {}; }

  // The same checkpoints again, in the TOPOLOGICAL plan's terms rather
  // than the phase vocabulary's.
  //
  // Two ledgers, and a family has to answer both -- see
  // docs/MODEL-MEMORY.md, "Which ledger do I use?". This one is derived
  // from the graph's edges instead of from a phase name a stage
  // asserts, so it is the one that stays right in a graph nobody
  // anticipated.
  //
  // Answer with `source` naming the checkpoint (the directory or file
  // that will be opened -- the plan merges two stages holding one
  // checkpoint by that name), `preload` what it costs resident, and
  // `floor` the least it can be held at while still being held. Leave
  // `releases` and `reclaimable` alone: whether idle weights are
  // dropped is the STAGE's `unload_when_idle` policy, and the stage
  // stamps it onto whatever comes back from here.
  //
  // Empty -- the default -- means the family declines, and the stage
  // then has nothing to put in the plan for it. That is not the same as
  // costing nothing; it is a hole, and it reads as room that is not
  // there.
  virtual std::vector<StageHolding>
  declare_holdings(const std::string& /*root*/) const { return {}; }

  // Bytes of the LATENT this family emits for a clip of this geometry,
  // which the stage declares as a payload alive from the denoise that
  // writes it to the last decode that reads it.
  //
  // Asked of the family because the shape is the family's: channel
  // count, spatial compression and temporal compression all differ, and
  // a host that substituted a built-in's formula would report a
  // confident number for the wrong model. 0 -- the default -- means
  // "cannot say", and the stage then declares nothing rather than
  // something plausible.
  //
  // `frames` is the PIXEL frame count after align_frames, not the
  // latent count.
  virtual std::size_t latent_bytes(const std::string& /*root*/, int /*width*/,
                                   int /*height*/, int /*frames*/) const
  {
    return 0;
  }

  // Bytes a generation of this geometry ALLOCATES beyond the weights
  // declare_resources() claims -- activation scratch, per-clip caches --
  // which the stage declares as scratch in the denoise phase. 0, the
  // default, means "cannot say", and nothing is declared.
  //
  // For a family whose largest allocation is not its checkpoint.
  // FlashVSR's key/value window at 1920x1152 is three times its weights;
  // a plan that sees only the weights admits a graph the box cannot run,
  // and on a 16 GB M5 that is a kernel panic rather than a refusal.
  //
  // `model_config` is the family's own model-config beat when the graph
  // wired one as a constant, so a knob that changes the size is honoured;
  // null means the family's defaults. The geometry is already through
  // size_grid and align_frames.
  virtual std::size_t
  denoise_scratch_bytes(const std::string& /*root*/, int /*width*/,
                        int /*height*/, int /*frames*/,
                        const FlexData* /*model_config*/) const
  {
    return 0;
  }

  // The soundtrack terms, for a family that generates one alongside the
  // video. False -- the default -- means this family emits no audio, and
  // the stage declares none.
  //
  // `latent` is what leaves the stage on its audio oport, `pcm` what the
  // audio decode produces from it, and `arena` what that decode
  // allocates while running. They have three different lifetimes, which
  // is why they are three numbers: the latent spans denoise to
  // decode-audio, the PCM spans decode-audio to decode (the mux reads it
  // after the frames exist), and the arena is gone when the decode ends.
  virtual bool audio_cost(const std::string& /*root*/, int /*frames*/,
                          double /*fps*/, std::size_t* /*latent*/,
                          std::size_t* /*pcm*/, std::size_t* /*arena*/) const
  {
    return false;
  }

  // Build the generator. Null (after warning through `args.session`)
  // leaves the stage inert rather than taking the pipeline down.
  virtual std::unique_ptr<VideoGenerator>
  load(const VideoModelCreateArgs& args) = 0;
};

// Process-wide family set. Same singleton discipline as StageRegistry
// and ResourcePlannerRegistry: a plugin MUST link the host libvpipe
// shared so it registers into THIS instance rather than forking a
// second one.
class VideoModelRegistry {
public:
  static VideoModelRegistry& get() noexcept;

  // Takes ownership. First-wins on `tag()`: a second family claiming a
  // tag already present is ignored and returns false, so two plugins
  // shipping the same model cannot silently shadow each other.
  bool add(std::unique_ptr<VideoModelFamily> f);

  // The first registered family that claims `root`, or null. Order is
  // registration order, which for plugins is load order -- so a
  // deliberately narrow `claims` is a family's own responsibility.
  // Never throws: a family whose `claims` throws is skipped and warned.
  VideoModelFamily* claim_for(const SessionContextIntf* session,
                              const std::string&        root,
                              const std::string&        model_type) const;

  VideoModelFamily* find(std::string_view tag) const noexcept;

  std::vector<VideoModelFamily*> all() const;

private:
  VideoModelRegistry() = default;

  mutable std::mutex                             _mu;
  std::vector<std::unique_ptr<VideoModelFamily>> _families;
};

}

#endif
