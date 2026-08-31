#ifndef VPIPE_STAGES_VAE_DECODE_STAGE_H
#define VPIPE_STAGES_VAE_DECODE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The Qwen-Image VAE decoder (MetalKrea2Vae) is a from-scratch, MLX-free
// metal-compute module on the VPIPE_BUILD_APPLE_SILICON axis. On non-Apple
// builds the stage is an inert stub (the constructor errors through session()
// and every beat emits nothing).
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/tensor-beat.h"
#include "stages/model-memory.h"
#include "generative-models/krea2/metal-krea2-vae.h"
#include "generative-models/flux2/metal-flux2-vae.h"
#include "generative-models/mage/metal-mage-vae.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/vae-model-registry.h"
#include "generative-models/wan/metal-wan-vae.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// VAE decode stage: the second half of the `krea2` (Krea-2-Turbo) text-to-
// image split -- it turns a latent tensor into an RGB image. Runs the
// Qwen-Image VAE decoder (AutoencoderKLQwenImage) on the metal-compute
// backend.
//
//   iport0  TensorBeatPayload, an f32 latent [z_dim, H/8, W/8] (channel-
//           first, unpacked but NOT un-whitened -- the boundary the DiT stage
//           emits). The stage un-whitens per channel (latents*std + mean) then
//           decodes.
//
//   iport1  OPTIONAL FlexDataPayload model reference (from a `model-select`
//           source). Latched on the first beat; OVERRIDES the hf_dir config so
//           the conditioner / DiT / vae-encode / vae-decode of a graph can
//           share one model choice. When wired, the VAE load is DEFERRED to
//           the first process() (the beat only arrives after the init barrier).
//
//   oport0  TensorBeatPayload, planar U8 RGB [3, H, W] (channel order R,G,B,
//           0..255) -- the same format load-image emits, so it flows into the
//           image sinks. The decoded [-1,1] float is mapped (x+1)/2*255.
//
//   oport1  OPTIONAL TensorBeatPayload, the same pixels as ONE CLIP:
//           planar U8 RGB [frames, 3, H, W], one beat per VIDEO latent,
//           sideband {frames, fps}. The shape temporal-stack builds for
//           a clip reference, so it is what a reference-conditioned
//           model takes.
//
//           WHY A SECOND PORT AND NOT A MODE. oport0 is unchanged, so
//           every graph that decodes to frames -- save-image,
//           rgb-to-video -> save-video, a preview -- is untouched and
//           needs no unpacking stage to get back what it already had.
//           The two ports are the same pixels at two granularities, and
//           a graph wires whichever it wants (or both).
//
//           It is written ONLY for a video latent, and only when a
//           consumer is wired (ctx.has_consumers). An image latent has
//           no time axis, and emitting [1, 3, H, W] for one would
//           invent a one-frame CLIP where the model draws a real
//           distinction between that and a still.
//
//           BUILDING IT COSTS A COPY OF THE CLIP -- ~380 MB at
//           960x544x243 -- which is why it is conditional on the
//           wiring. It is not a new memory regime: the families that
//           decode video already hold the whole clip before they emit
//           anything (the wan and plugin paths collect their frames,
//           the MiniMax-H3 one decodes into a single bf16 buffer),
//           because ctx.write is a coroutine and a VAE frame sink is a
//           plain callback.
//
//           A CONSUMER THAT HOLDS THIS BEAT HOLDS THE WHOLE CLIP. Where
//           the point is one frame -- seeding the next generation from
//           the last frame of this one -- slice it down BEFORE anything
//           caches it, or a feedback register pins 380 MB for the
//           length of the next generation instead of 1.5 MB.
//
// Config (FlexData object on the 4th constructor parameter):
//   hf_dir     (string, OPTIONAL) -- the Krea-2-Turbo model directory; the VAE
//                                    weights are read from <hf_dir>/vae (or
//                                    hf_dir itself if it is already a vae dir).
//                                    A model-select source on iport1 overrides
//                                    it; required only when iport1 is unwired.
class VaeDecodeStage final : public TypedStage<VaeDecodeStage> {
public:
  static constexpr const char* kTypeName = "vae-decode";

  VaeDecodeStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);
  ~VaeDecodeStage() override;

  Job initialize(RuntimeContext& ctx) override;

  // The VAE this stage loads. Small next to the DiT and the encoder,
  // but it is resident during a decode and nothing else declares it.
  std::vector<ResourceClaim> declare_resources() const override;

  // Latch a `model-select` constant before the planning phase, so
  // the claim above is made against the model this graph will
  // actually run rather than against an empty hf_dir.
  void apply_constant(unsigned iport, const FlexData& beat) override;
  void reset_run_state() override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()          const noexcept { return _hf_dir; }
  std::uint64_t      images_emitted()  const noexcept { return _images_emitted; }

private:
  // The arena the idle policy was last decided against, and the peers it
  // was weighed with. Not a high-water mark: the policy follows the beat
  // down as well as up. See revise_decode_arena_.
  std::size_t              _arena_decided = 0;
  // The last arena PUT ON THE BOOKS, so a restatement is logged once
  // rather than on every beat of a fixed-size run.
  std::size_t              _arena_stated  = 0;
  std::vector<std::string> _idle_peers;

  // The VAE directory this stage's phase claim named, recomputed the
  // same way declare_resources() computes it -- one expression, so the
  // claim and its release cannot name different things.
  std::string vae_dir_for_release_() const;

#ifdef VPIPE_BUILD_APPLE_SILICON
  // ---- the clip oport (oport1) ----
  //
  // Allocate the [F, 3, H, W] clip, or return null when nothing is
  // wired to oport1 -- the null IS the "do not build it" answer, so
  // every caller is one `if (clip)` rather than a flag plus a pointer
  // that can disagree with it.
  std::unique_ptr<TensorBeatPayload>
  begin_clip_(RuntimeContext& ctx, int F, int H, int W) const;
  // Copy frame `f`'s [3, H, W] planar bytes into the clip. Silently
  // does nothing when `clip` is null, so a caller need not re-test.
  static void
  add_to_clip_(TensorBeatPayload* clip, int f,
               const TensorBeatPayload& frame);
  // Stamp {frames, fps} and the producer's model name, matching what
  // temporal-stack puts on a stacked video group.
  void
  finish_clip_(TensorBeatPayload* clip, double fps,
               const TensorBeatPayload& src) const;
#endif

  // See Stage::declare_memory.
  StageMemory declare_memory() const override;

public:
  // THE TWO OPORTS DO NOT ADVANCE AT THE SAME RATE, and which rate
  // oport0 runs at depends on the checkpoint.
  //
  //   * oport1, the clip, is ONE beat per latent by construction, so it
  //     always shares the latent iport's clock.
  //   * oport0 is one beat per latent for an IMAGE checkpoint and one
  //     PER FRAME for a video one. Those are different clocks, and only
  //     the family says which.
  //
  // It matters because a feedback pair must stay inside one clock
  // domain. Reporting the image answer for a video checkpoint -- which
  // is what this stage did before it could tell them apart -- admits a
  // loop that closes through the FRAME stream, where `feedback-rx`
  // caches whichever frame won the scheduling. The clip oport is the
  // one that can carry a loop, and now it is the one that says so.
  //
  // The family is probed from the checkpoint, so this answer needs
  // `_hf_dir` -- which for the graphs that matter arrives as a
  // model-select constant. That is why the runtime's clock-domain
  // analysis runs after constant folding; see Phase 3.7 in
  // pipeline-runtime.cc. With no checkpoint to probe it falls back to
  // the image answer, i.e. exactly what it reported before.
  unsigned oport_clock_group(unsigned p) const noexcept override;

private:
  // Resolved once, lazily, from the checkpoint: 0 = one beat per
  // latent, 1 = one per frame. -1 = not yet asked.
  mutable int _oport0_group = -1;

  std::string _hf_dir;
  // "krea2" (Qwen-Image VAE) | "flux2" (AutoencoderKLFlux2) |
  // "mage" (MageVAE) | "wan" (AutoencoderKLWan, the VIDEO one)
  std::string _family;
  std::uint64_t _images_emitted = 0;

  // Model iport bookkeeping (used only when a model-select source is wired):
  // latch the model beat once, and load the VAE at most once (lazily, since
  // the beat only arrives after the init barrier).
  bool _model_latched  = false;
  bool _load_attempted = false;

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Loaded lazily (exactly one per the detected family) by ensure_loaded_ --
  // from initialize() when the model comes from config, or from process() when
  // it arrives on the model iport. Left null on failure so the stage stays
  // inert (process() warns + emits nothing).
  std::unique_ptr<genai::MetalKrea2Vae> _vae;
  std::unique_ptr<genai::MetalFlux2Vae> _flux2_vae;
  std::unique_ptr<genai::MetalMageVae>  _mage_vae;
  std::unique_ptr<genai::MetalWanVae>   _wan_vae;
  // MiniMax-H3's video VAE. Its decoder is a ViT rather than an
  // upsampling conv stack, so it tiles STRUCTURALLY and returns the whole
  // clip at once instead of streaming chunks through a sink.
  std::unique_ptr<genai::MetalMiniMaxH3VideoVae> _h3_vae;

  // ---- an out-of-tree family (VaeModelRegistry) -------------------------
  // Consulted BEFORE the built-in `_class_name` chain, mirroring how
  // generate-video consults VideoModelRegistry before its built-in
  // dispatch. BORROWED: the registry is process-wide and outlives every
  // stage. `_family` is set from the family's tag() so the log lines read
  // like a built-in's, but every dispatch below is guarded on the
  // POINTER, never on the string.
  genai::VaeModelFamily*             _vae_family = nullptr;
  std::unique_ptr<genai::VaeDecoder> _plugin_dec;
  // The probe reads a checkpoint header, so it runs once rather than on
  // every idle-unload reload.
  bool _family_probed = false;

  // Frame rate stamped onto each emitted video frame's sideband when the
  // producer did not supply one. Only meaningful for the "wan" family.
  double _fps = 16.0;

  // Resolve _hf_dir + load the VAE (idempotent: the _load_attempted guard runs
  // the body at most once). No-op when _hf_dir is still empty.
  void ensure_loaded_();

  // ---- idle unload -------------------------------------------------------
  // The VAE weights are small next to a DiT, but on a memory-bounded box the
  // decode's own activation working set has to fit BESIDE whatever is still
  // resident, and this stage is idle for the whole denoise. `unload_when_idle`
  // (auto | always | never) drops the VAE after each beat and reloads it on the
  // next one; auto decides from physical RAM vs the pipeline's weight bytes,
  // the same rule the DiT and conditioner stages use.
  model_memory::UnloadPolicy _unload_cfg = model_memory::UnloadPolicy::kAuto;
  bool _unload_idle   = false;
  bool _unload_resolved = false;
  bool _unloaded      = false;
  bool _quiet_reload  = false;   // reload logs at debug, not info
  void load_note_(const VpipeFormat& msg) const;
  void unload_vae_();
  // Publish this beat's real arena and ratchet the idle policy.
  void revise_decode_arena_(std::size_t bytes);
  // The same, for an image decode, from its pixel size.
  void publish_image_arena_(int px_w, int px_h);
  void reload_vae_();
#endif
};

}

#endif
