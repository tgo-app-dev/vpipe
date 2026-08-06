#ifndef VPIPE_STAGES_GENERATE_VIDEO_STAGE_H
#define VPIPE_STAGES_GENERATE_VIDEO_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The Wan denoiser (MetalWanTransformer) is a from-scratch, MLX-free
// metal-compute module on the VPIPE_BUILD_APPLE_SILICON axis. On non-Apple
// builds the stage is an inert stub.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/krea2/flow-sampler.h"
#include "generative-models/wan/metal-wan-transformer.h"
#include "stages/model-memory.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Video DiT denoiser: text (and optionally an image) -> a latent VIDEO,
// for the Wan 2.1/2.2 family. The video counterpart of `generate-image`,
// and deliberately its mirror -- same conditioning input from a
// `diffusion-conditioner`, same optional sampler / scheduler / model
// sources, same "emit a latent and let vae-decode make pixels" split.
//
// Two things make it its own stage rather than a mode of generate-image.
//
// The first is the TIME AXIS: everything it touches is 4-D. The latent is
// [z, T, H/8, W/8] with T = 1 + (F-1)/4, the DiT patchifies over (frame,
// row, column) with a 3-axis RoPE, and the emitted beat is 4-D where every
// image stage emits 3-D.
//
// The second is that Wan 2.2's A14B is TWO 14B experts -- a high-noise one
// and a low-noise one -- switched partway down the sigma schedule at
// `boundary_ratio`, each with its own guidance scale. At bf16 one expert
// is ~28 GB, so they cannot both be resident: this stage holds exactly one
// and swaps at the boundary. That swap is a load from disk, so it happens
// at most ONCE per generation (the schedule is monotonically decreasing,
// so the boundary is crossed once), and it is why the two guidance scales
// are separate config keys rather than one.
//
//   iport0  TensorBeatPayload conditioning from a diffusion-conditioner:
//           bf16 [text_seq, 4096] umT5-XXL hidden states.
//   iport1  OPTIONAL negative conditioning (the conditioner's oport1) for
//           classifier-free guidance. Wan is NOT distilled -- without a
//           negative there is nothing to guide away from, so guidance is
//           forced to 1 and the second forward per step is skipped.
//   iport2  OPTIONAL FlexDataPayload model reference (model-select);
//           overrides hf_dir.
//   iport3  OPTIONAL sampler spec (diffusion-sampler-select). Default is
//           UniPC multistep, which is what every Wan checkpoint ships.
//   iport4  OPTIONAL scheduler spec (scheduler-select). Default is the
//           checkpoint's flow schedule (shift 3.0).
//   iport5  OPTIONAL image-to-video conditioning latent from a vae-encode
//           stage: f32 [16, T, H/8, W/8], the VAE encoding of the
//           conditioning image followed by F-1 blank frames. Present =>
//           image-to-video; absent => text-to-video.
//
//   oport0  TensorBeatPayload f32 [16, T, H/8, W/8] latent video (whitened,
//           the boundary vae-decode reads), sideband {fps, frames}.
//
// Config (FlexData object):
//   hf_dir           (string, OPTIONAL) -- the Wan model root
//   height / width   (int)  -- VIDEO pixels; both must be multiples of 16
//   frames           (int)  -- video frames; F % 4 == 1 (81, 121, ...)
//   fps              (real) -- stamped on the latent for the decoder
//   steps            (int)  -- denoising steps
//   seed             (int)  -- initial-noise RNG seed
//   guidance_scale   (real) -- CFG for the HIGH-noise expert
//   guidance_scale_2 (real) -- CFG for the LOW-noise expert
//   boundary_ratio   (real) -- sigma below which the low-noise expert takes
//                              over; 0 => single expert
//   unload_when_idle (string) -- auto|always|never
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
  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()          const noexcept { return _hf_dir; }
  std::uint64_t      latents_emitted() const noexcept { return _emitted; }
  int                latent_frames()   const noexcept;

private:
  std::string   _hf_dir;
  int           _height = 480;
  int           _width  = 832;
  int           _frames = 81;
  double        _fps    = 16.0;
  int           _steps  = 40;
  std::uint64_t _seed   = 0;
  double        _guidance   = 3.5;
  double        _guidance_2 = 3.5;
  double        _boundary   = 0.9;
  std::uint64_t _emitted = 0;

  bool _model_latched     = false;
  bool _sampler_latched   = false;
  bool _scheduler_latched = false;

#ifdef VPIPE_BUILD_APPLE_SILICON
  genai::FlowSamplerSpec   _sampler_spec;
  genai::FlowSchedulerSpec _scheduler_spec;

  // Exactly one expert is resident. `_expert` is which one (0 = high
  // noise / `transformer`, 1 = low noise / `transformer_2`, -1 = none).
  std::unique_ptr<genai::MetalWanTransformer> _dit;
  int _expert = -1;
  genai::MetalWanTransformer::Config _cfg;
  bool _have_cfg    = false;
  bool _two_experts = false;
  std::string _root;

  // Load expert `which`, dropping whichever is resident first. The drop
  // has to happen BEFORE the load, not after: 2 x 28 GB does not fit on
  // any machine this runs on, so an overlap is not a peak, it is an OOM.
  bool ensure_expert_(int which);
  void resolve_config_();

  model_memory::UnloadPolicy _unload_cfg = model_memory::UnloadPolicy::kAuto;
  bool _unload_idle     = false;
  bool _unload_resolved = false;
#endif
};

}  // namespace vpipe

#endif
