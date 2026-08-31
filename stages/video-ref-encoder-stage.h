#ifndef VPIPE_STAGES_VIDEO_REF_ENCODER_STAGE_H
#define VPIPE_STAGES_VIDEO_REF_ENCODER_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// MiniMax-H3's `ref2va` reference encoders are from-scratch, MLX-free
// metal-compute modules on the VPIPE_BUILD_APPLE_SILICON axis. On
// non-Apple builds the stage is an inert stub.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/minimax-h3/metal-minimax-h3-audio-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/minimax-h3/minimax-h3-reference-encoder.h"
#include "generative-models/minimax-h3/minimax-h3-text-encoder.h"
#include "generative-models/qwen3/metal-qwen-vision.h"
#include "generative-models/weight-set.h"
#include "stages/model-memory.h"
#include "stages/model-registry.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// VideoRefEncoderStage: a list of reference media -> the conditioning
// and the reference latent rows a reference-to-video-and-audio DiT
// denoises against.
//
// This is `diffusion-conditioner` for MiniMax-H3's `ref2va` partition,
// and it is a stage of its own for one structural reason: a `ref2va`
// request conditions on a VARIABLE list of up to twelve heterogeneous
// references, and a pipeline's ports are static. Twelve `load-image` /
// `load-video` / `vae-encode` chains cannot express "three clips and
// nine stills" without being re-authored per request, so the LIST is
// the input and this stage decodes it.
//
// That is also what the reference implementation does -- its references
// arrive as in-memory media carrying their own rates, decoded by the
// caller -- and the rates are the reason it matters. MiniMax-H3
// resamples every reference onto its own 24 fps and onto the audio
// VAE's sample rate, so a frame rate lost between a decoder stage and
// here is a request conditioned at the wrong speed, with nothing to
// raise about it. Decoding a clip and its soundtrack out of ONE open
// container is the only way the two stay in sync.
//
// Three models are resident while it runs, and each reads the same
// reference DIFFERENTLY:
//
//   * the Qwen3-VL-32B vision tower, at 2 fps and at its own
//     smart-resized canvas, whose tokens are spliced into the prompt
//     presentation ("<Picture 1>: ", "<Audio 2>: ", "<Video 1>: " plus
//     one timestamped block per merged frame pair);
//   * the video VAE, over the full 24 fps clip at MiniMax-H3's canvas,
//     whose latent rows the DiT holds pinned for every denoising step;
//   * the audio VAE, over the soundtrack, whose rows ride along clean.
//
// Feeding either visual model the other's pixels produces a
// well-shaped tensor and a wrong conditioning, which is why the
// pairing lives in one place (minimax-h3-reference-encoder.h) rather
// than being spread over the graph.
//
//   iport0  prompt       FlexDataPayload (string or {text: ...}),
//                        required. Used VERBATIM -- no chat template,
//                        no special tokens.
//   iport1  model        OPTIONAL FlexDataPayload model reference (a
//                        `model-select` source); overrides hf_dir so
//                        the encoder / DiT / vae stages of a graph
//                        share one model. When wired, the load is
//                        DEFERRED to the first process().
//   iport2  ref1         OPTIONAL TensorBeatPayload reference media,
//   ...                  ONE BEAT PER REFERENCE, numbered after the
//   iport7  ref6         `references` list. Kind from the tensor's
//                        RANK -- [N] or [channels, N] f32 audio,
//                        [3, H, W] u8 picture, [frames, 3, H, W] u8
//                        clip -- and the rates from the sideband
//                        (`fps`, `sr`), which are REQUIRED and never
//                        defaulted. Optional sideband: `short_edge`
//                        for this one reference, `attach` to override
//                        the stage's `attach_audio` for this beat. A
//                        WIRED port beats every request; an empty
//                        tensor is how it says "nothing this time".
//
// THE FILE REFERENCES ARE CONFIG, not a port. They are files to open,
// which is what a `references` config key is and what the composer's
// file browser fills in -- it multi-selects straight into the list. The
// list arrived as a beat first, and as a JSON STRING at that (no stage
// in this tree emits a structured FlexData literal, so it had to be
// authored as text somewhere and parsed here); that made the one input
// a user actually edits the one input they could not pick with a file
// dialog, and made a typo in a quoted path a runtime warning instead of
// something the browser could not produce.
//
// The tensor iports are the OTHER half of that argument rather than a
// reversal of it: a path cannot name a still that does not exist yet, a
// crop, or a clip that was never written to disk, and those are exactly
// the references a graph produces rather than picks. Six of them,
// because the request's shape has to be static for the numbering to be
// -- a port that could fall silent would renumber every reference after
// it, and the numbering IS the request.
//
//   oport0  conditioning TensorBeatPayload bf16 [n_tokens, 5120] -- the
//                        Qwen3-VL-32B layer-50 tap over the whole
//                        presentation, the same contract
//                        `diffusion-conditioner` emits, so
//                        generate-video's conditioning port takes
//                        either. Its SIDEBAND carries the plan: the
//                        per-row modality tags and the latent geometry
//                        of every reference, which is what the packed
//                        layout is built from.
//   oport1  ref_video    TensorBeatPayload f32 [rows, 96] -- the image
//                        and video references' latents, already packed
//                        into DiT rows and concatenated in reference
//                        order. Ragged geometry collapses here: every
//                        reference is encoded at a resolution of its
//                        own, so rows are the only shape they share.
//                        Emitted with 0 rows when no reference carries
//                        video, never withheld.
//   oport2  ref_audio    TensorBeatPayload f32 [rows, 32] -- the
//                        reference soundtracks, channel-major within a
//                        reference, in the same order. 0 rows when
//                        none.
//
// Config (FlexData object):
//   references       (string or array of strings, REQUIRED) -- the
//                              reference files IN THE ORDER THE MODEL
//                              SHOULD READ THEM. The order labels them
//                              in the presentation ("<Picture 1>",
//                              "<Video 2>", ...) and lays them out on
//                              the shared rotary clock, so a different
//                              order is a different request, not a
//                              different spelling of one.
//
//                              WHAT each file is -- image, video or
//                              audio -- is read from the FILE, not from
//                              its name or from a `kind` the caller
//                              states. See probe_media_file: an
//                              extension is a claim and the bytes are
//                              the fact, and a browser hands over
//                              whatever the user picked. A container
//                              with one frame is an IMAGE reference,
//                              not a one-frame clip: the two rules
//                              genuinely differ (2048 short edge and no
//                              area cap, against the target's canvas
//                              rule), so reading a still as video would
//                              encode it at a quarter the detail.
//
//                              A video reference conditions on its own
//                              soundtrack when it has one.
//   hf_dir           (string, OPTIONAL) -- the MiniMax-H3 model dir. A
//                                          model-select source on
//                                          iport1 overrides it.
//   frames           (int)  -- the GENERATED frame count, snapped up to
//                              `17n + 5`. MUST MATCH the generate-video
//                              stage's `frames`: it is the duration
//                              every reference is truncated to, and the
//                              layout the DiT builds is sized from the
//                              same number. A mismatch is a shape error
//                              50 layers deep.
//   reference_image_short_edge (int, default 2048) -- what an image
//                              reference's short edge is scaled to. It
//                              has NO area cap and never binds the
//                              generated canvas.
//   reference_video_short_edge (int, default 768) -- what a video
//                              reference's short edge is scaled to,
//                              under the area cap below. 0 means the
//                              CLIP'S OWN, which is how a graph declines
//                              an upscale: at 768 a 960x544 reference
//                              lands on 1344x768, 1.98x the pixels
//                              interpolated from the same information.
//                              It is charged for three times, not twice
//                              -- the VAE encode, the reference rows the
//                              DiT attends to on every step, AND the
//                              conditioner, because the vision tower is
//                              handed these same pixels and so gets
//                              fewer tokens for the clip. MEASURED on
//                              the two-reference example, 768 -> 0:
//                              13120 -> 7144 reference rows and
//                              3115 -> 2119 conditioning rows. Both are
//                              fidelity channels; compare, do not assume.
//   attach_audio     (array of int) -- reference iport numbers (1..6)
//                              whose AUDIO is the soundtrack of the
//                              reference before it rather than a
//                              reference of its own. POSITIONAL: it
//                              folds onto whichever reference precedes
//                              it, which may be one from the
//                              `references` LIST, since files are read
//                              first. A list and not a switch, because
//                              a request may carry both an attached
//                              soundtrack and a standalone piece of
//                              music. A beat's own `attach` sideband
//                              overrides it either way.
//   reference_image_max_pixels (int, default 0 = UNCAPPED) -- an area
//                              cap for image references. Uncapped is the
//                              released checkpoint's rule, where
//                              reference_image_short_edge is the only
//                              bound -- which it stops being the moment
//                              a short edge of 0 takes a picture at the
//                              size it arrived. A graph feeding raw
//                              pictures through the reference iports
//                              wants this set: an uncapped 4K still is
//                              ~220 VAE tiles and 8160 DiT rows.
//   reference_video_max_pixels (int, default 768*1344) -- the area cap
//                              that canvas is held under. Applied after
//                              the short edge, so it is what binds a
//                              reference LARGER than the canvas whatever
//                              the short edge says.
//   video_sample_fps (real, default 2.0) -- the rate the CONDITIONER
//                              reads a video reference at. Not the rate
//                              the VAE encodes it at, which is 24.
//   max_prompt_tokens (int, default 16384) -- the conditioner's
//                              sequence pool. A presentation is far
//                              longer than a text-only prompt: one
//                              2048-short-edge image is thousands of
//                              vision tokens, so the text path's 4096
//                              overflows on the first reference. It is
//                              KV for 50 layers of a 32B model (~200 KB
//                              a token), which is why it is a knob and
//                              not a global default.
//   unload_when_idle (string) -- auto|always|never
//
// Six TENSOR reference iports (2..7) sit after prompt and model, for the
// references a file list cannot name: a generated still, a cropped
// frame, a clip that was never written to disk. One beat per reference,
// numbered after the `references` list, kind taken from the tensor's
// RANK and rates from its sideband. A wired port must beat every
// request -- an empty tensor says "nothing this time" -- because a port
// that could fall silent would renumber every reference after it, and
// the numbering IS the request. See kRefPortDoc in the .cc.
//
// Their default short edge is 0: encode the picture at the size it
// arrived. A caller reaching for a port has already decided the size,
// and re-resolving it against a per-kind default would undo exactly
// that decision.
class VideoRefEncoderStage final : public TypedStage<VideoRefEncoderStage> {
public:
  static constexpr const char* kTypeName = "video-ref-encoder";

  VideoRefEncoderStage(const SessionContextIntf* session,
                       std::string               id,
                       std::vector<InEdge>       iports,
                       FlexData                  config);
  ~VideoRefEncoderStage() override;

  Job initialize(RuntimeContext& ctx) override;

  // The conditioner and both VAEs, plus the DiT this stage is paired
  // with -- the same dirs its own sizing uses, so the declaration and
  // the estimate cannot disagree.
  std::vector<ResourceClaim> declare_resources() const override;
  // See Stage::declare_memory.
  StageMemory declare_memory() const override;

  // Latch a `model-select` constant before the planning phase, so
  // the claim above is made against the model this graph will
  // actually run rather than against an empty hf_dir.
  void apply_constant(unsigned iport, const FlexData& beat) override;
  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir() const noexcept { return _hf_dir; }
  // The reference paths AS RESOLVED -- confined to the session file
  // sandbox at construction. Exposed because that resolution is the
  // difference between a path the composer's file browser produced
  // opening and not opening, and it is not observable any other way
  // without a model to load.
  const std::vector<std::string>& references() const noexcept
  {
    return _references;
  }
  std::uint64_t requests_encoded() const noexcept { return _emitted; }

#ifdef VPIPE_BUILD_APPLE_SILICON
  // One reference iport beat -> a MediaReference: kind from the tensor's
  // RANK, rates from the sideband. Public and static because it is the
  // part of the port contract that fails SILENTLY when it is wrong -- a
  // clip taken for a still, a soundtrack read at the wrong rate -- and
  // that deserves a test that does not need a model or a runtime. `n` is
  // the 1-based port number, for the message only. False with a reason
  // in `err`.
  static bool
  media_from_beat(const TensorBeatPayload& tb, int n,
                  genai::minimax_h3::MediaReference* out, std::string* err);
#endif

private:
  std::string   _hf_dir;
  std::string   _enc_dir;
  // The reference files, in read order, resolved from the config once.
  // Paths only: WHAT each one is comes from the file when it is opened,
  // not from anything held here.
  std::vector<std::string> _references;
  int           _frames = 121;
  int           _ref_short_edge = 2048;
  // The video-reference canvas. 0 short edge = the clip's own, which is
  // how a graph declines an upscale (see normalize_video_reference).
  int           _vid_short_edge = 768;
  int           _vid_max_pixels = 768 * 1344;
  // 0 = uncapped, the released checkpoint's image rule.
  int           _img_max_pixels = 0;
  // Reference iport numbers (1-based) whose audio is the SOUNDTRACK of
  // the reference before it. A beat's own `attach` sideband overrides.
  std::vector<int> _attach_audio;
  double        _video_sample_fps = 2.0;
  int           _max_prompt_tokens = 16384;
  std::uint64_t _emitted = 0;

  bool _model_latched  = false;
  bool _load_attempted = false;

#ifdef VPIPE_BUILD_APPLE_SILICON
  // The conditioner backbone (Qwen3-VL-32B tapped at layer 50) and its
  // vision tower. The tower is a separate load over the SAME
  // checkpoint -- 380M parameters next to the backbone's 32B -- because
  // a `ref2va` request runs it over every reference before it has a
  // prompt to encode.
  std::unique_ptr<genai::MiniMaxH3TextEncoder>   _enc;
  std::unique_ptr<genai::MetalQwenVisionEncoder> _vision;
  // Both VAEs, ENCODER halves only: each builds its encoder on the
  // first encode() and a graph that never decodes here never pays for
  // the video VAE's 2.4B-parameter ViT decoder.
  std::unique_ptr<genai::MetalMiniMaxH3VideoVae> _video_vae;
  std::unique_ptr<genai::MetalMiniMaxH3AudioVae> _audio_vae;

  genai::minimax_h3::ReferenceLimits _limits;

  // Resolve _hf_dir + load the four models (idempotent: the
  // _load_attempted guard runs the body at most once). Called from
  // initialize() (config model) or the first process() (model iport).
  void ensure_loaded_();
  bool load_models_(metal_compute::MetalCompute* mc);

  // DECODE the configured reference files, classifying each by content
  // and bringing its rates along. False (with a warned reason) on an
  // undecodable or unclassifiable file -- a reference that silently
  // dropped out would change the presentation's numbering and every
  // label after it, which is a quietly different request rather than a
  // failed one.
  bool decode_references_(
      std::vector<genai::minimax_h3::MediaReference>* out);

  // Warn about every reference the encoder had to reshape, and log the
  // ones it took as given. See the definition for why this is WARN.
  void report_fits_(const genai::minimax_h3::EncodedReferences& enc);

  // ---- idle unload -------------------------------------------------
  // The conditioner is the largest resident block in a `ref2va` graph
  // (~48 GB bf16 for the 50 tapped layers, ~26 GB at w8) and it is idle
  // for the whole denoise -- which on this model is minutes. Same rule
  // and same three settings as the other generative stages.
  model_memory::UnloadPolicy _unload_cfg = model_memory::UnloadPolicy::kAuto;
  bool _unload_idle     = false;
  bool _unload_resolved = false;
  bool _unloaded        = false;
  std::string _root;
  // "ref2va" / "fl2va" from the models DB, empty when it did not say.
  // A repo holding both partitions resolves by FILENAME otherwise, and
  // the filename prefers fl2va.
  std::string _partition;
  // The dirs the idle-unload heuristic weighs: this stage's own
  // components plus the DiT it is paired with. Filled at load.
  std::vector<std::string> _peer_dirs;

  void resolve_unload_policy_();
  void unload_models_();
  bool reload_models_();
#endif
};

}  // namespace vpipe

#endif
