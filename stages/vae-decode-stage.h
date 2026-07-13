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
#include "generative-models/krea2/metal-krea2-vae.h"
#include "generative-models/flux2/metal-flux2-vae.h"
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
//   oport0  TensorBeatPayload, planar U8 RGB [3, H, W] (channel order R,G,B,
//           0..255) -- the same format load-image emits, so it flows into the
//           image sinks. The decoded [-1,1] float is mapped (x+1)/2*255.
//
// Config (FlexData object on the 4th constructor parameter):
//   hf_dir     (string, required) -- the Krea-2-Turbo model directory; the VAE
//                                    weights are read from <hf_dir>/vae (or
//                                    hf_dir itself if it is already a vae dir).
//   models_db  (string, default "models") -- registry for resolve_model_dir.
class VaeDecodeStage final : public TypedStage<VaeDecodeStage> {
public:
  static constexpr const char* kTypeName = "vae-decode";

  VaeDecodeStage(const SessionContextIntf* session,
                 std::string               id,
                 std::vector<InEdge>       iports,
                 FlexData                  config);
  ~VaeDecodeStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()          const noexcept { return _hf_dir; }
  std::uint64_t      images_emitted()  const noexcept { return _images_emitted; }

private:
  std::string _hf_dir;
  std::string _models_db;
  std::string _family;   // "krea2" (Qwen-Image VAE) | "flux2" (AutoencoderKLFlux2)
  std::uint64_t _images_emitted = 0;

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Loaded lazily in initialize() (exactly one per the detected family); left
  // null on failure so the stage stays inert (process() warns + emits nothing).
  std::unique_ptr<genai::MetalKrea2Vae> _vae;
  std::unique_ptr<genai::MetalFlux2Vae> _flux2_vae;
#endif
};

}

#endif
