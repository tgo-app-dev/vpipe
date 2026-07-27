#ifndef VPIPE_STAGES_VAE_ENCODE_STAGE_H
#define VPIPE_STAGES_VAE_ENCODE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The Qwen-Image VAE encoder (MetalKrea2Vae, with_encoder) is a from-scratch,
// MLX-free metal-compute module on the VPIPE_BUILD_APPLE_SILICON axis. On
// non-Apple builds the stage is an inert stub.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/krea2/metal-krea2-vae.h"
#include "generative-models/flux2/metal-flux2-vae.h"
#include "generative-models/mage/metal-mage-vae.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// VAE encode stage: the mirror of vae-decode -- it turns an RGB image into a
// latent. Runs the Qwen-Image VAE encoder (AutoencoderKLQwenImage) on the
// metal-compute backend, emitting the WHITENED latent that the text-to-image
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
//           text-to-image stage emits, so it flows into the `latent` port
//           there (img2img init) or straight back into vae-decode.
//
// Config (FlexData object on the 4th constructor parameter):
//   hf_dir     (string, OPTIONAL) -- the Krea-2-Turbo model directory; the VAE
//                                    weights are read from <hf_dir>/vae (or
//                                    hf_dir itself if it is already a vae dir).
//                                    A model-select source on iport1 overrides
//                                    it; required only when iport1 is unwired.
//   models_db  (string, default "models") -- registry for resolve_model_dir.
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
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()          const noexcept { return _hf_dir; }
  std::uint64_t      latents_emitted() const noexcept { return _latents_emitted; }

private:
  std::string _hf_dir;
  std::string _models_db;
  // Optional letterbox resize target (both > 0 => enabled). Multiples of 8.
  int _target_w = 0;
  int _target_h = 0;
  // Letterbox pad color, per channel in 0..255 (default black).
  int _pad_r = 0;
  int _pad_g = 0;
  int _pad_b = 0;
  std::string _family;   // "krea2" | "flux2" | "mage"
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

  // Resolve _hf_dir + load the VAE encoder (idempotent: the _load_attempted
  // guard runs the body at most once). No-op when _hf_dir is still empty.
  void ensure_loaded_();
#endif
};

}

#endif
