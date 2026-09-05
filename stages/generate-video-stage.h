#ifndef VPIPE_STAGES_GENERATE_VIDEO_STAGE_H
#define VPIPE_STAGES_GENERATE_VIDEO_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The video denoisers (MetalWanTransformer, MetalMiniMaxH3Transformer) are
// from-scratch metal-compute modules on the VPIPE_BUILD_APPLE_SILICON axis.
// On non-Apple builds the stage is an inert stub.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/krea2/flow-sampler.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"
#include "generative-models/minimax-h3/minimax-h3-denoise.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/video-model-registry.h"
#include "generative-models/wan/metal-wan-transformer.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

struct TensorBeat;

// Video DiT denoiser: text (and optionally keyframe or reference latents)
// -> a latent VIDEO, plus a latent SOUNDTRACK from the families that
// generate one. The video counterpart of `generate-image`, and
// deliberately its mirror -- same conditioning input from a
// `diffusion-conditioner`, same optional sampler / scheduler / model
// sources, same "emit a latent and let vae-decode make pixels" split.
//
// Two things make it its own stage rather than a mode of generate-image.
//
// The first is the TIME AXIS: everything it touches is 4-D. The latent is
// [z, T, H/r, W/r] over a family-specific temporal rule, the DiT
// patchifies over (frame, row, column) with a 3-axis RoPE, and the emitted
// beat is 4-D where every image stage emits 3-D.
//
// The second is that a video model can be too large to hold at once. Wan
// 2.2's A14B is TWO 14B experts -- a high-noise one and a low-noise one --
// switched partway down the sigma schedule at `boundary_ratio`, each with
// its own guidance scale. At bf16 one expert is ~28 GB, so they cannot
// both be resident: this stage holds exactly one and swaps at the
// boundary. That swap is a load from disk, so it happens at most ONCE per
// generation (the schedule is monotonically decreasing, so the boundary is
// crossed once), and it is why the two guidance scales are separate config
// keys rather than one. MiniMax-H3 answers the same pressure differently
// -- one 33B stack whose blocks STREAM -- which is why the residency
// policy is per family rather than one rule here.
//
// FAMILIES. The stage is family-generic the way `generate-image` is: one
// stage, a `_family` tag read from the DiT's `_class_name`, and the UNION
// of what its families need on the ports and in the config. A key or a
// port that does not apply to the resident family is inert, not an
// error -- a graph should not have to be rewired to change checkpoints.
//
//   wan          Wan 2.1/2.2. TWO 14B experts on 2.2's A14B, switched at
//                `boundary_ratio`; CFG against a negative prompt.
//   minimax-h3   MiniMax-H3, both partitions. ONE 33B stack emitting
//                video AND audio from a single packed sequence.
//                Guidance-DISTILLED, so it has no negative pass at all,
//                and its two sigma schedules (video shift 12, audio
//                shift 3) advance in lockstep. FL2VA conditions on
//                keyframe anchors (iport5/6); REF2VA conditions on a
//                list of references encoded by a `video-ref-encoder`
//                (iport7/8). The two ship byte-identical DiT configs
//                and are told apart by the packaging, so a Ref2VA
//                checkpoint with no references wired is REFUSED rather
//                than run -- it would generate video conditioned on
//                nothing.
//
//   iport0  TensorBeatPayload conditioning from a diffusion-conditioner:
//           bf16 [text_seq, enc_hidden] -- umT5-XXL 4096 for wan, the
//           Qwen3-VL-32B layer-50 tap at 5120 for minimax-h3.
//   iport1  OPTIONAL negative conditioning (the conditioner's oport1) for
//           classifier-free guidance. Wan is NOT distilled -- without a
//           negative there is nothing to guide away from, so guidance is
//           forced to 1 and the second forward per step is skipped.
//           IGNORED by minimax-h3, which is guidance-distilled: there is
//           no unconditional pass to blend with, and running one would
//           double the cost of a 33B model for nothing.
//   iport2  OPTIONAL FlexDataPayload model reference (model-select);
//           overrides hf_dir.
//   iport3  OPTIONAL sampler spec (diffusion-sampler-select). Default is
//           UniPC multistep, which is what every Wan checkpoint ships.
//   iport4  OPTIONAL scheduler spec (scheduler-select). Default is the
//           checkpoint's flow schedule (shift 3.0).
//   iport5  OPTIONAL image-to-video conditioning latent from a vae-encode
//           stage. For wan: f32 [16, T, H/8, W/8], the VAE encoding of the
//           conditioning image followed by F-1 blank frames. For
//           minimax-h3: f32 [24, 1, H/16, W/16], the FIRST-frame keyframe
//           anchor. Present => image-to-video; absent => text-to-video.
//   iport6  OPTIONAL LAST-frame keyframe anchor, same format as iport5 and
//           from a second vae-encode over the closing image (minimax-h3;
//           IGNORED by wan, whose i2v latent is one clip-shaped tensor).
//           Wiring it turns first-frame i2v into the FL2VA partition this
//           checkpoint is named for -- generation interpolates between the
//           two stills. Only meaningful alongside iport5: a last frame with
//           no first frame is not a mode the model was trained for, so it
//           warns and is dropped rather than silently becoming an L2V.
//   iport7  OPTIONAL reference VIDEO rows from a `video-ref-encoder`
//           (minimax-h3 REF2VA): f32 [rows, 96], the image and video
//           references' latents already packed into DiT rows and
//           concatenated in reference order. Ragged by nature -- every
//           reference is encoded at a resolution of its own -- so rows
//           are the only shape they share. Wiring it is what makes this
//           stage a Ref2VA denoiser.
//   iport8  OPTIONAL reference AUDIO rows from the same encoder: f32
//           [rows, 32], channel-major within a reference. They ride at a
//           clean timestep and are never denoised.
//   iport10 OPTIONAL second-modality conditioning: LTX-2.5's 2048-wide
//           AUDIO context, beside the 4096-wide video one on iport0.
//           Unwired, such a family generates video only and says so.
//           Ignored by wan and minimax-h3.
//
//   iport9  OPTIONAL model-specific parameters from the resident family's
//           config source -- `wan2-model-config` (guidance, expert
//           boundary) or `minimax-h3-model-config` (sigma shifts,
//           condition timesteps, audio duration). One FlexData object
//           tagged `model-config`, passed to the family's own
//           GenerationParams::from_flex UNREAD, so this stage carries
//           none of the knobs and a family can add one without touching
//           it. Unwired, every family runs its documented defaults.
//
// WHY THE KNOBS LEFT THIS STAGE'S CONFIG. Serving several families from
// one stage made its config the UNION of what they need, where each key
// was inert on the family that was not resident -- and inert SILENTLY, so
// a `video_shift` set on a Wan graph did nothing and said nothing. Which
// keys apply is now a wiring decision: what is connected is what applies,
// the graph shows which family it was built for, and a mismatch is a
// warning instead of a no-op. What stays here is what every video model
// answers to -- geometry, length, steps, seed, residency.
//
// The GEOMETRY those rows belong to -- how many latent frames and cells
// each reference encoded to, and the per-row modality tags of the
// presentation -- rides on the CONDITIONING beat's sideband, not on a
// port of its own: it is one request, and pairing a conditioning with
// another request's layout packs cleanly and then fails 50 layers deep.
//
//   oport0  TensorBeatPayload f32 [z, T, H/r, W/r] latent video (whitened,
//           the boundary vae-decode reads), sideband {fps, frames}.
//           z/r are the family's VAE geometry: 16 and 8 for wan, 24 and
//           16 for minimax-h3.
//   oport1  TensorBeatPayload f32 [stereo, latent_channels, audio_latents]
//           latent AUDIO (2 x 32 x N for minimax-h3), sideband
//           {latents_per_second} -- what `audio-vae-decode` reads. Never
//           written by a family that does not generate audio, so a graph
//           that leaves it unconnected is not doing anything wrong.
//
// Config (FlexData object):
//   hf_dir           (string, OPTIONAL) -- the Wan model root
//   height / width   (int)  -- VIDEO pixels; both must be multiples of 16
//   frames           (int)  -- video frames. ROUNDED UP to the nearest
//                              count the resident family's VAE can chunk:
//                              4k+1 for wan (81, 121, ...), 17n+5 for
//                              minimax-h3 (22, 39, 56, ...). The two rules
//                              share almost no legal counts, so any
//                              positive number is accepted and adjusted
//                              rather than rejected -- otherwise a graph
//                              could not change families without being
//                              re-authored. The adjustment is logged.
//   fps              (real) -- stamped on the latent for the decoder
//   steps            (int)  -- denoising steps
//   seed             (int)  -- initial-noise RNG seed
//   unload_when_idle (string) -- auto|always|never
//
// Everything family-specific lives on iport9 instead; see the config
// stages named above.
class GenerateVideoStage final : public TypedStage<GenerateVideoStage> {
public:
  static constexpr const char* kTypeName = "generate-video";

  GenerateVideoStage(const SessionContextIntf* session,
                     std::string               id,
                     std::vector<InEdge>       iports,
                     FlexData                  config);
  ~GenerateVideoStage() override;

  Job initialize(RuntimeContext& ctx) override;

  // One 14B expert at bf16 is ~28 GB and this stage holds one at a time,
  // so what it declares is ONE expert -- declaring both would size the box
  // against a peak that never happens and push every peer into streaming.
  std::vector<ResourceClaim> declare_resources() const override;
  // The DiT is claimed for the DENOISE phase when this stage will
  // certainly drop it after each clip. Same two conditions the
  // conditioner's encoder claim satisfies (see
  // model_memory::weight_claims_in_phase): the release has to be certain
  // from CONFIG, and the decision to release must not itself consult a
  // phased figure.
  //
  // It matters more here than anywhere: a 63 GB DiT counted as
  // persistent appears in the conditioning and both decode columns as
  // well as its own, and those are the phases a constrained box is
  // usually tightest in.
  std::vector<ResourceClaim> decide_resources() const override;

  // The topological plan's view of this stage. Same facts as the claims
  // above, minus the phase: what the DiT weighs, what it weighs
  // streaming, whether it lets go, and how big the latents on its two
  // oports are. Who consumes those is the runtime's business.
  StageMemory declare_memory() const override;

  // Latch a `model-select` constant before the planning phase, so
  // the claim above is made against the model this graph will
  // actually run rather than against an empty hf_dir.
  void apply_constant(unsigned iport, const FlexData& beat) override;
  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // How many keyframe anchors a MiniMax-H3 request takes: 0 for
  // text-to-video, 1 for a first frame, 2 for first AND last. Sets
  // `*ignored` when a keyframe was wired and had to be dropped.
  //
  // Public and static for the reason VideoRefEncoderStage::
  // media_from_beat is: this is the part of the port contract that
  // fails SILENTLY when it is wrong -- a Ref2VA graph takes no anchors
  // at all, so a wired keyframe is read and goes nowhere -- and that
  // deserves a test which needs neither a 33B model nor a runtime.
  static int h3_anchor_count(bool is_ref2va, bool have_keyframe,
                             int ref_frames, bool* ignored);

  // Test-only accessors.
  const std::string& hf_dir()          const noexcept { return _hf_dir; }
  std::uint64_t      latents_emitted() const noexcept { return _emitted; }
  int                latent_frames()   const noexcept;

private:
  // Which DiT family the resident checkpoint is, from its `_class_name`
  // (see resolve_config_). "wan" keeps the historical behaviour, so a
  // config that predates the split still means what it did.
  std::string   _family = "wan";
  std::string   _hf_dir;
  // The VDN-H3 release root the DiT was BUILT with, or empty for the
  // stock attention. Latched from the model_config beat like a LoRA
  // path -- see the `linear_branch` key on `minimax-h3-model-config`,
  // which is where it is configured. Not a key on this stage: it is a
  // fact about one family and nothing else here reads it.
  std::string   _vdn_dir;

  // The branch the CONFIG asks for, straight out of the beat.
  //
  // Apart from `_vdn_dir` because they answer different questions at
  // different times: this one serves declare_resources(), which is
  // const and runs before any driver, so it reads the folded constant
  // rather than a value the runtime latch has not set yet.
  std::string vdn_dir_() const;
  int           _height = 480;
  int           _width  = 832;
  int           _frames = 81;
  double        _fps    = 16.0;
  int           _steps  = 40;
  // LOSSY dynamic-int8 block GEMMs (opt-in; minimax-h3 only)
  bool        _i8_gemm{};
  std::uint64_t _seed   = 0;
  std::uint64_t _emitted = 0;

  bool _model_latched     = false;
  bool _sampler_latched   = false;
  bool _scheduler_latched = false;
  bool _cfg_latched       = false;

  // The last model-config beat, held UNPARSED. Which family's parser
  // reads it is not known until the checkpoint resolves, and a config
  // beat can arrive before the model does.
  FlexData _model_cfg;
  // The runtime LoRA SLOTS for the MiniMax-H3 DiT, off the model_config
  // beat, in the order the DiT binds them. Two, so a few-step Turbo
  // distillation and a style or identity adapter can ride together --
  // see kMaxLoraSlots in the transformer. The PATH is load-time; the
  // STRENGTH is not, and each slot's moves on its own.
  struct H3LoraSlot {
    std::string path;
    double      scale = 1.0;
    // "auto" | "flat" | "per_head"; empty is auto. LOAD-time like the
    // path -- the rows are permuted once at bind, not per forward -- and
    // per SLOT because the row order is a property of the FILE.
    std::string qkv;
  };
  std::vector<H3LoraSlot> _h3_lora;

#ifdef VPIPE_BUILD_APPLE_SILICON
  genai::FlowSamplerSpec   _sampler_spec;
  genai::FlowSchedulerSpec _scheduler_spec;

  // The resident family's per-generation parameters, as ITS model layer
  // defines them. Both are held for the same reason the two DiTs are:
  // the families share no knobs, and one struct wide enough for both
  // would describe neither.
  genai::MetalWanTransformer::GenerationParams _wan_params;
  genai::MetalMiniMaxH3Transformer::GenerationParams _h3_params;
  // Re-parse `_model_cfg` into the family that is now resident, and
  // reconcile it with what the checkpoint itself says. Called whenever
  // either input changes -- a new config beat, or a model that just
  // resolved -- because a beat can arrive on either side first.
  void apply_model_config_();

  // Exactly one expert is resident. `_expert` is which one (0 = high
  // noise / `transformer`, 1 = low noise / `transformer_2`, -1 = none).
  std::unique_ptr<genai::MetalWanTransformer> _dit;
  int _expert = -1;
  genai::MetalWanTransformer::Config _cfg;

  // MiniMax-H3's single stack. Held beside the Wan expert rather than
  // behind a virtual base, matching how generate-image carries its five
  // DiTs: the two have different forward signatures (one packed
  // sequence with per-row timesteps against a 4-D latent) and an
  // interface wide enough for both would describe neither.
  std::unique_ptr<genai::MetalMiniMaxH3Transformer> _h3_dit;
  genai::MetalMiniMaxH3Transformer::Config _h3_cfg;

  // A family contributed by a PLUGIN, from genai::VideoModelRegistry.
  // Consulted BEFORE the two built-in families, the way
  // LoadedLanguageModel consults ModelExecRegistry before its built-in
  // arch dispatch -- so an out-of-tree checkpoint resolves to its own
  // family and the built-in path is not touched. Null on every graph
  // that runs a built-in checkpoint, which is the common case.
  //
  // `_plugin_family` is process-wide and borrowed (the registry owns it
  // and outlives every stage); `_plugin_gen` is THIS stage's resident
  // checkpoint. A family owns its whole denoise loop, so there is
  // nothing here that mirrors _wan_params / _h3_params: the model-config
  // beat goes down unread in the request. See video-model-registry.h.
  genai::VideoModelFamily*              _plugin_family = nullptr;
  std::unique_ptr<genai::VideoGenerator> _plugin_gen;
  // The plugin branch of process(), mirroring run_h3_: builds the
  // request from the beats already read and publishes what came back.
  // False means the family warned and produced nothing.
  bool run_plugin_family_(RuntimeContext& ctx,
                          const void* cond, int cond_rows, int cond_dim,
                          const FlexData* cond_sideband,
                          const void* neg, int neg_rows,
                          const class TensorBeatPayload* ref,
                          const class TensorBeatPayload* ref_last,
                          const class TensorBeatPayload* ref_video_rows,
                          const class TensorBeatPayload* ref_audio_rows,
                          const class TensorBeatPayload* audio_cond,
                          genai::VideoGenResult* out);
  // What the models DB said about `_hf_dir`. Held because the DIRECTORY
  // cannot always answer which model a reference meant -- see
  // ResolvedModel.
  ResolvedModel _resolved;
  bool _have_cfg    = false;
  bool _two_experts = false;
  // Said ONCE: a keyframe anchor wired on a Ref2VA graph. A continuous
  // graph makes one clip per beat, and the wiring cannot change between
  // them, so repeating it per request would bury every other line.
  bool _kf_on_ref2va_said = false;
  std::string _root;

  // Load expert `which`, dropping whichever is resident first. The drop
  // has to happen BEFORE the load, not after: 2 x 28 GB does not fit on
  // any machine this runs on, so an overlap is not a peak, it is an OOM.
  bool ensure_expert_(int which);
  // Adopt `aligned` as the frame count, reporting the change when there
  // is one. The family's own rule computes the value (MetalWanVae::
  // align_num_frames / minimax_h3::align_num_frames); this only records
  // it, so the two callers cannot differ on how it is announced.
  // Geometry as the family will actually produce it -- config rounded up
  // to the model's latent constraints. The arena estimate must use these
  // or it describes a clip nobody will make; see the definition.
  bool planned_geometry_(const std::string& root, int* out_w, int* out_h,
                         int* out_frames) const;
  void align_frames_(int aligned);
  // Round the frame size up to the resident family's tiling grid.
  void align_size_(int gh, int gw);
  void resolve_config_();

  // Stamp the generating model onto an output beat's sideband, video
  // and audio alike. The chain generate-video -> vae-decode ->
  // rgb-to-video -> save-video (and its audio twin) carries it through
  // to the container, so a saved clip records what produced it. Merges
  // into any sideband already there.
  void tag_model_(TensorBeat& tb) const;

  // "fl2va" / "ref2va" / empty. The two partitions are one architecture
  // and two TASKS, and nothing in the weights distinguishes them, so
  // this is read from the packaging -- see
  // MetalMiniMaxH3Transformer::partition_of.
  std::string _h3_partition;

  // The `ref2va` conditioning that arrives beside the prompt embeds,
  // from a `video-ref-encoder`. It is one request split over three
  // beats: the geometry and the per-row modality tags ride on the
  // conditioning's sideband, the latent rows on their own ports (they
  // are megabytes, and every reference is encoded at a resolution of
  // its own, so rows are the only shape they share).
  struct H3References {
    std::vector<genai::minimax_h3::Reference> refs;
    // Borrowed from the beats, which outlive the forward.
    const float* video_rows   = nullptr;
    int          n_video_rows = 0;
    const float* audio_rows   = nullptr;
    int          n_audio_rows = 0;
    // MiniMax-H3's per-row modality tag for the conditioning rows: text
    // is 1, but a vision block's rows are tagged 0 (video). The DiT
    // reads this, so it cannot be assumed uniform the way a text-only
    // prompt's can.
    std::vector<int> text_tags;
  };
  // The minimax-h3 branch of process(): builds the packed layout, runs
  // genai::denoise, and unpatchifies both modalities back to latents.
  // Returns false when it warned and produced nothing.
  // `cond` is the conditioning tensor's raw bf16 rows. Kept as plain
  // types so this header does not have to see the beat payloads.
  // `cond` is the conditioning tensor's raw bf16 rows. `ref` is the
  // OPTIONAL keyframe anchor latent from vae-encode, already whitened,
  // as f32 [z, ref_frames, lh, lw] -- one latent frame per anchor, so
  // 1 = first only and 2 = first AND last.
  // Whether this forward's scratch fits, parking LRU weights if not.
  // False => refuse the forward; see the note on its definition for why
  // refusing beats proceeding on a 16 GB box.
  // The idle-unload decision, made after the load so it can use the
  // streaming verdict. See the note on its definition.
  void resolve_unload_policy_h3_(bool streamed);
  bool preflight_h3_scratch_(int seq, int text_rows);
  // `r2v` is the `ref2va` request when there is one; null is the
  // `t2va` / `fl2va` path, where `ref` carries the keyframe anchors
  // instead. The two are mutually exclusive by construction -- they are
  // different checkpoints.
  // Read a `ref2va` plan off the conditioning beat's sideband and pair
  // it with the two reference-row beats. Leaves `out->refs` empty (and
  // returns true) when the conditioning carries no references at all --
  // that is a `t2va` / `fl2va` request, not a malformed one. False only
  // on a plan that IS present and does not add up, after warning.
  bool parse_h3_references_(const FlexData& sideband,
                            const class TensorBeatPayload* video_rows,
                            const class TensorBeatPayload* audio_rows,
                            H3References* out) const;

  bool run_h3_(const void* cond, int text_rows, const float* ref,
               int ref_frames, const H3References* r2v,
               std::vector<float>* video_out, std::vector<int>* video_shape,
               std::vector<float>* audio_out, std::vector<int>* audio_shape);

  model_memory::UnloadPolicy _unload_cfg = model_memory::UnloadPolicy::kAuto;
  // What decide_resources() named in its phase claim. Mutable because
  // that method is const by the Stage contract and this is a note to
  // self, not state the plan reads back.
  mutable std::string _dit_dir_declared_;
  bool _unload_idle     = false;
  bool _unload_resolved = false;
#endif
};

}  // namespace vpipe

#endif
