#ifndef VPIPE_STAGES_VAE_ENCODE_STAGE_H
#define VPIPE_STAGES_VAE_ENCODE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The Qwen-Image VAE encoder (MetalKrea2Vae, with_encoder) is a from-scratch,
// MLX-free metal-compute module on the VPIPE_BUILD_APPLE_SILICON axis. On
// non-Apple builds the stage is an inert stub.
#ifdef VPIPE_BUILD_APPLE_SILICON
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

// VAE encode stage: the mirror of vae-decode -- it turns an RGB image into a
// latent. Runs the Qwen-Image VAE encoder (AutoencoderKLQwenImage) on the
// metal-compute backend, emitting the WHITENED latent that the generate-image
// stage's `latent` port consumes for img2img (and that vae-decode round-trips
// back to an image).
//
//   iport0  TensorBeatPayload, a U8 or f32 RGB image [3, H, W] (channel-first,
//           U8 0..255 or f32 [-1,1]) -- the format load-image / vae-decode
//           emit. Converted to [-1,1] and encoded to the posterior mode.
//
//   iport1  OPTIONAL FlexDataPayload model reference (from a `model-select`
//           source). Latched on the first beat; OVERRIDES the hf_dir config so
//           a graph's conditioner / DiT / vae-encode / vae-decode can share one
//           model choice. When wired, the VAE load is DEFERRED to the first
//           process() (the beat only arrives after the init barrier).
//
//   oport0  TensorBeatPayload, an f32 latent [z_dim, H/8, W/8] (channel-first,
//           unpacked, WHITENED -- (x-mean)/std) -- the same format the
//           generate-image stage emits, so it flows into the `latent` port
//           there (img2img init) or straight back into vae-decode. A video
//           family emits [z_dim, T, h, w]; the shape is the ENCODER's, not
//           one predicted here.
//
// OUT-OF-TREE FAMILIES are consulted first, through VaeModelRegistry --
// the same order and the same reason as vae-decode. The built-in
// `_class_name` chain below cannot be joined from a plugin, and its
// FALLBACK is "krea2", so a checkpoint it does not recognise is not
// refused but misread: a family whose latent is 32x at 128 channels
// would be encoded at 8x and 16, and the beat would look fine. A family
// that claims a checkpoint and supplies no encoder therefore leaves this
// stage INERT rather than falling through.
//
// Config (FlexData object on the 4th constructor parameter):
//   hf_dir     (string, OPTIONAL) -- the Krea-2-Turbo model directory; the VAE
//                                    weights are read from <hf_dir>/vae (or
//                                    hf_dir itself if it is already a vae dir).
//                                    A model-select source on iport1 overrides
//                                    it; required only when iport1 is unwired.
//   target_width, target_height (int, optional) -- when BOTH are set (and
//                                    multiples of 8), the input image is
//                                    letterbox-resized to this size before
//                                    encoding: the picture is scaled to fit
//                                    with its original aspect ratio, centered,
//                                    and the leftover top/bottom (or left/right)
//                                    is padded. Lets an arbitrarily-sized
//                                    reference feed a fixed-size text-to-image.
//   pad_color  (array [r,g,b] 0..255, or a name "black"/"white"/"gray";
//                                    default black) -- the letterbox pad color.
class VaeEncodeStage final : public TypedStage<VaeEncodeStage> {
public:
  static constexpr const char* kTypeName = "vae-encode";

  VaeEncodeStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);
  ~VaeEncodeStage() override;

  Job initialize(RuntimeContext& ctx) override;

  // The VAE this stage loads. Small next to the DiT and the encoder,
  // but it is resident during a decode and nothing else declares it.
  std::vector<ResourceClaim> declare_resources() const override;
  // See Stage::declare_memory.
  StageMemory declare_memory() const override;

  // The VAE directory this stage names, resolved ONE way so the claim,
  // the plan and the pool cannot name different things. See the twin in
  // VaeDecodeStage for why the generic resolver is not enough.
  std::string vae_dir_for_release_() const;

  // Latch a `model-select` constant before the planning phase, so
  // the claim above is made against the model this graph will
  // actually run rather than against an empty hf_dir.
  void apply_constant(unsigned iport, const FlexData& beat) override;
  void reset_run_state() override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()          const noexcept { return _hf_dir; }
  std::uint64_t      latents_emitted() const noexcept { return _latents_emitted; }

private:
  std::string _hf_dir;
  // Optional letterbox resize target (both > 0 => enabled). Multiples of 8.
  int _target_w = 0;
  int _target_h = 0;
  // Letterbox pad color, per channel in 0..255 (default black).
  int _pad_r = 0;
  int _pad_g = 0;
  int _pad_b = 0;
  std::string _family;   // "krea2" | "flux2" | "mage" | "wan"
  // Video frames the wan conditioning clip spans. Must match the
  // generate-video stage's `frames`: the conditioning latent is
  // the VAE encoding of image-then-blanks over exactly that many
  // frames, and the DiT concatenates it channel-wise onto a noise
  // latent of that shape.
  int _frames = 81;
  std::uint64_t _latents_emitted = 0;

  // Parse the `pad_color` config attr into _pad_r/_pad_g/_pad_b (0..255).
  // Accepts an [r,g,b] array, a name (black/white/gray|grey), or a single
  // gray level; leaves the black default on an unrecognized value.
  void parse_pad_color_();

  // Model iport bookkeeping (used only when a model-select source is wired):
  // latch the model beat once, and load the VAE at most once (lazily, since
  // the beat only arrives after the init barrier).
  bool _model_latched  = false;
  bool _load_attempted = false;

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Loaded lazily (one per the detected family) by ensure_loaded_ -- from
  // initialize() when the model comes from config, or from process() when it
  // arrives on the model iport. Left null on failure so the stage stays inert
  // (process() warns + emits nothing).
  std::unique_ptr<genai::MetalKrea2Vae> _vae;
  std::unique_ptr<genai::MetalFlux2Vae> _flux2_vae;
  std::unique_ptr<genai::MetalMageVae>  _mage_vae;
  std::unique_ptr<genai::MetalWanVae>   _wan_vae;
  // MiniMax-H3's video VAE, encoder half: a keyframe anchor for FL2VA.
  std::unique_ptr<genai::MetalMiniMaxH3VideoVae> _h3_vae;

  // An OUT-OF-TREE family (VaeModelRegistry), consulted BEFORE the
  // built-in `_class_name` chain -- the same order vae-decode uses. Null
  // for every built-in checkpoint, and then nothing here costs anything.
  genai::VaeModelFamily*             _vae_family = nullptr;
  std::unique_ptr<genai::VaeEncoder> _plugin_enc;

  // Resolve _hf_dir + load the VAE encoder (idempotent: the _load_attempted
  // guard runs the body at most once). No-op when _hf_dir is still empty.
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
  void reload_vae_();
#endif
};

}

#endif
