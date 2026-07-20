#ifndef VPIPE_STAGES_TEXT_TO_IMAGE_STAGE_H
#define VPIPE_STAGES_TEXT_TO_IMAGE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The Krea-2-Turbo text-to-image stack (Qwen3-VL text encoder + Krea2
// MMDiT + FlowMatchEuler sampler) is a from-scratch, MLX-free metal-compute
// path on the VPIPE_BUILD_APPLE_SILICON axis. On non-Apple builds this stage
// is an inert stub.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/krea2/flow-sampler.h"
#include "generative-models/krea2/metal-krea2-transformer.h"
#include "generative-models/flux2/metal-flux2-transformer.h"
#include "generative-models/qwen-image/metal-qwen-image-transformer.h"
#endif

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Text-to-image (DiT) stage: the denoiser half of the diffusion split -- it
// consumes ready-made conditioning from a `diffusion-conditioner` stage (which
// owns the tokenizer + text encoder + vision tower) and runs the family DiT
// (Krea2 12B MMDiT / FLUX.2 / Qwen-Image-Edit dual-stream) + FlowMatchEuler
// sampler on the metal-compute backend, then emits the unpacked latent for the
// vae-decode stage. Pair it with a diffusion-conditioner on the SAME hf_dir.
//
//   iport0  TensorBeatPayload conditioning from the diffusion-conditioner stage
//           (family-shaped + typed: krea2 f16 [n,12,2560]; flux2 f16
//           [n,3*enc_hidden]; qwen-image-edit bf16 [n_real,3584]).
//
//   iport1  OPTIONAL TensorBeatPayload neg_conditioning (same shape/type, the
//           conditioner's oport1). Drives classifier-free guidance: when it is
//           present AND guidance_scale != 1, each denoise step runs the DiT a
//           second time on the negative conditioning and combines
//           v = v_neg + scale*(v_pos - v_neg). Read per generation when a beat is
//           available. Unwired (or scale 1) => no CFG (single DiT pass, the
//           token-exact turbo default).
//
//   iport2  OPTIONAL FlexDataPayload sampler spec (from a `sampler-select`
//           stage). Latched on the first beat and reused for every prompt;
//           selects the integrator (euler / heun / dpmpp_2m / dpmpp_sde) + its
//           knobs. Unwired => the config default (euler, the token-exact turbo).
//
//   iport3  OPTIONAL FlexDataPayload scheduler spec (from a `scheduler-select`
//           stage). Latched likewise; selects the sigma schedule (simple /
//           karras / exponential), steps and shift. Unwired => config default
//           (simple / `steps` / shift 1.15).
//
//   iport4  OPTIONAL TensorBeatPayload reference latent 0 -- an f32 latent
//   iport5  OPTIONAL TensorBeatPayload reference latent 1 (channel-first,
//           the format the vae-encode stage emits: FLUX.2 [dit_channels, H/16,
//           W/16], Krea-2 [z_dim, H/8, W/8]). Read once when a beat is available
//           and cached. FLUX.2 uses them as multi-reference conditioning: each
//           reference is embedded, appended to the image-token stream and given
//           its own RoPE T offset so the two references occupy distinct position
//           bands. Krea-2 has no multi-ref path, so ref latent 0 doubles as the
//           img2img init (mixed into the noise at the strength-selected sigma)
//           and ref latent 1 is ignored. Unwired => text-to-image from noise.
//
//   oport0  TensorBeatPayload, an f32 latent [z_dim, H/8, W/8] (channel-first,
//           unpacked, still whitened -- the vae-decode stage un-whitens).
//
// Config (FlexData object):
//   hf_dir     (string, required) -- the model dir; the transformer/ subfolder's
//                                    _class_name selects the DiT family. (The
//                                    text_encoder/ + tokenizer/ live here too but
//                                    are the diffusion-conditioner's concern.)
//   dit_dir    (string, optional) -- override for the DiT (transformer) dir,
//                                    e.g. a model-quantize'd 4/8-bit DiT; else
//                                    <hf_dir>/transformer.
//   height     (int, default 256) -- output height  (must be a multiple of 16).
//   width      (int, default 256) -- output width   (must be a multiple of 16).
//   steps      (int, default 8)   -- turbo sampler steps.
//   seed       (int, default 0)   -- initial-noise RNG seed.
//   guidance_scale (real, default 1) -- classifier-free guidance scale. 1
//                                    disables CFG (single DiT pass). >1 with a
//                                    negative prompt on iport1 runs a second
//                                    DiT pass per step and pushes the velocity
//                                    away from the negative (v_neg + scale*
//                                    (v_pos - v_neg)); ~2x DiT cost per step.
//   models_db  (string, default "models") -- registry for resolve_model_dir.
//   init_latents (string, optional) -- debug/repro: a raw f32 file of packed
//                                    initial latents [img_seq, z_dim*4] to use
//                                    instead of RNG noise (for golden anchoring).
//   adopt_latent_dims (bool, default false) -- when set (with strength>0 and a
//                                    latent wired on iport1), the output size is
//                                    taken from the incoming latent's H/W
//                                    (shape[1]*8 x shape[2]*8) instead of the
//                                    configured width/height, so an
//                                    arbitrarily-sized reference img2img "just
//                                    works" without matching width/height.
class TextToImageStage final : public TypedStage<TextToImageStage> {
public:
  static constexpr const char* kTypeName = "text-to-image";

  TextToImageStage(const SessionContextIntf* session,
                   std::string               id,
                   std::vector<InEdge>       iports,
                   FlexData                  config);
  ~TextToImageStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()        const noexcept { return _hf_dir; }
  std::uint64_t      latents_emitted() const noexcept { return _latents_emitted; }

private:
  std::string _hf_dir;
  std::string _dit_dir;
  std::string _models_db;
  std::string _init_latents;
  int         _height{};
  int         _width{};
  int         _steps{};
  double      _strength{};      // img2img: 0 => text-to-image (pure noise)
  double      _guidance_scale{};     // CFG scale; 1 => disabled (single pass)
  bool        _adopt_latent_dims{};  // take output size from the img2img latent
  bool        _i8_gemm{};            // LOSSY dynamic-int8 DiT GEMMs (opt-in)
  std::uint64_t _seed{};
  std::uint64_t _latents_emitted = 0;

  // "krea2" | "flux2" | "qwen-image-edit" (from the transformer _class_name).
  std::string _family = "krea2";

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Loaded lazily in initialize() (one DiT per the detected family); left null
  // on failure (stage stays inert). The text encoder + tokenizer + vision tower
  // live in the paired diffusion-conditioner stage, not here.
  std::unique_ptr<genai::MetalKrea2Transformer>   _dit;
  std::unique_ptr<genai::MetalFlux2Transformer>   _flux2_dit;
  std::unique_ptr<genai::MetalQwenImageTransformer> _qie_dit;
  // On a memory-bounded box (DiT + the conditioner's resident encoder can't fit
  // a large decode too), drop the DiT's per-forward scratch after each
  // generation so it doesn't crowd out the downstream vae-decode. Set from the
  // same RAM heuristic that decides DiT streaming.
  bool _release_scratch = false;

  // The active sampler (integrator) + scheduler (sigma schedule) specs. Seeded
  // from config (euler + simple / _steps / shift 1.15) and each overridden by
  // the first beat latched off the sampler / scheduler iports.
  genai::FlowSamplerSpec   _sampler_spec;
  genai::FlowSchedulerSpec _scheduler_spec;
  bool                     _sampler_latched   = false;
  bool                     _scheduler_latched = false;

  // A reference-image latent as it arrives on a ref-latent iport: the
  // vae-encode output, channel-first f32 [c, h, w] (FLUX.2 dit_channels @ H/16,
  // Krea-2 z_dim @ H/8). Cached across prompts once a beat is seen.
  struct RefLatent {
    std::vector<float> chw;
    int c = 0, h = 0, w = 0;
    bool empty() const { return chw.empty(); }
  };
  // Cached reference latents from iport4 / iport5 (read once when a beat is
  // available, reused for every later prompt like the negative prompt).
  RefLatent _ref[2];

  // The conditioning -> unpacked-latent forward: fuse the tapped text
  // conditioning through the DiT's text tower, sample the FlowMatchEuler turbo
  // steps and unpack. `cond` is the diffusion-conditioner's krea2 conditioning
  // (f16 [n_real, 12, EH]); `cond_neg` is the optional negative conditioning
  // (empty => no CFG). When `init_latent` is non-null (a whitened latent
  // [z, H/8, W/8], from the vae-encode stage) with strength>0 it is an img2img
  // run: scale_noise init, denoising only the tail steps. `init_packed` (when
  // non-null) overrides the packed init (repro / golden). `refs` are optional
  // reference-conditioning latents ([z_dim, h, w] channel-first) packed to DiT
  // tokens + given per-reference RoPE frame offsets. Returns the unpacked latent
  // [z, H/8, W/8] (whitened) or empty. When `emit_step` is set it is called with
  // the unpacked latent AFTER each sampler step (same [z, H/8, W/8] format) so
  // process() can stream it LIVE to the step_latent oport for a per-step preview.
  std::vector<float>
  generate_(const metal_compute::SharedBuffer& cond, int n_real,
            const metal_compute::SharedBuffer& cond_neg, int n_real_neg,
            int gen_h, int gen_w,
            const std::vector<float>* init_packed,
            const std::vector<float>* init_latent,
            const std::vector<RefLatent>& refs,
            const std::function<void(const std::vector<float>&)>& emit_step =
                {}) const;

  // FLUX.2 forward: `context` is the diffusion-conditioner's flux2 conditioning
  // (f16 [n, 3*enc_hidden] concatenated taps) -> FLUX DiT sampler loop (no
  // text-fusion tower, no CFG). `refs` are optional reference images
  // (patchify-packed to DiT tokens + given per-reference RoPE T offsets inside
  // the DiT). Returns the DiT-facing latent [dit_channels, H/16, W/16]
  // (channel-first) or empty.
  std::vector<float>
  generate_flux2_(const metal_compute::SharedBuffer& context, int n_real,
                  int gen_h, int gen_w,
                  const std::vector<RefLatent>& refs,
                  const std::function<void(const std::vector<float>&)>&
                      emit_step = {}) const;

  // Qwen-Image-Edit forward: `txt` is the diffusion-conditioner's qwen-image-edit
  // conditioning (bf16 [n_real, 3584] image-aware last-hidden, POST final-norm);
  // run the dual-stream MetalQwenImageTransformer over pure-noise packed latents
  // with the M2 dynamic-shift sampler and (when `txt_neg` is set + scale>1)
  // norm-preserving true-CFG. `refs` are reference latents (vae-encode output
  // [16, h, w]) packed 2x2 to DiT tokens + appended in their own RoPE frame
  // bands. Returns the unpacked, whitened latent [16, H/8, W/8] (channel-first)
  // or empty.
  std::vector<float>
  generate_qie_(const metal_compute::SharedBuffer& txt, int n_real,
                const metal_compute::SharedBuffer& txt_neg, int n_real_neg,
                int gen_h, int gen_w,
                const std::vector<float>* init_packed,
                const std::vector<RefLatent>& refs,
                const std::function<void(const std::vector<float>&)>&
                    emit_step = {}) const;
#endif
};

}

#endif
