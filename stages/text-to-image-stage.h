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
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/tokenizer.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Text-to-image stage: the first half of the `krea2` (Krea-2-Turbo) split --
// prompt -> latents. Runs the Qwen3-VL text encoder (a 12-layer tap), the
// Krea2 12B MMDiT denoiser and the 8-step turbo FlowMatchEuler sampler on the
// metal-compute backend, then emits the unpacked latent for the vae-decode
// stage.
//
//   iport0  FlexDataPayload carrying the prompt (a plain string OR an object
//           with a "text" key, like text-chat / text-to-speech).
//
//   iport1  OPTIONAL FlexDataPayload negative prompt (same string-or-{text:}
//           shape as iport0). Drives classifier-free guidance: when a negative
//           prompt is present AND guidance_scale != 1, each denoise step runs
//           the DiT a second time conditioned on the negative and combines
//           v = v_neg + scale*(v_pos - v_neg). Read once per prompt when a beat
//           is available and cached, so a single fixed negative is reused for
//           every generation. Unwired (or scale 1) => no CFG (single DiT pass,
//           the token-exact turbo default).
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
//   hf_dir     (string, required) -- the Krea-2-Turbo model dir (text_encoder/,
//                                    transformer/, tokenizer/ subfolders).
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
#ifdef VPIPE_BUILD_APPLE_SILICON
  // Tokenize a prompt into the compact encoder id sequence
  // [prefix | prompt | suffix] (the 34-token system prefix is dropped after
  // encoding). Empty if the tokenizer is not loaded. Exposed for tests.
  std::vector<std::int32_t> tokenize_prompt(const std::string& prompt) const;
#endif

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

  std::string _family = "krea2";   // "krea2" | "flux2" (transformer _class_name)

#ifdef VPIPE_BUILD_APPLE_SILICON
  // Loaded lazily in initialize() (one DiT per the detected family); left null
  // on failure (stage stays inert).
  std::unique_ptr<genai::MetalQwenModel>          _encoder;
  std::unique_ptr<genai::MetalKrea2Transformer>   _dit;
  std::unique_ptr<genai::MetalFlux2Transformer>   _flux2_dit;
  std::unique_ptr<genai::Tokenizer>               _tokenizer;
  metal_compute::SharedBuffer                     _embed;   // encoder embed table
  int _enc_hidden = 2560;
  // Text-encoder dir, retained so the encoder can be reloaded after a free.
  std::string _enc_dir;
  // Free the text encoder + embed table after each generation's conditioning is
  // built (the encoder is idle through denoise + VAE decode) and lazily reload
  // it for the next prompt, so its multi-GB footprint doesn't crowd out a large
  // decode on a memory-bounded box. Auto-on when the DiT is streamed
  // (memory-constrained); force with VPIPE_FLUX2_FREE_ENCODER.
  bool _free_encoder = false;

  // The active sampler (integrator) + scheduler (sigma schedule) specs. Seeded
  // from config (euler + simple / _steps / shift 1.15) and each overridden by
  // the first beat latched off the sampler / scheduler iports.
  genai::FlowSamplerSpec   _sampler_spec;
  genai::FlowSchedulerSpec _scheduler_spec;
  bool                     _sampler_latched   = false;
  bool                     _scheduler_latched = false;
  // Last negative prompt seen on iport1, cached across generations (a fixed
  // negative is read once and reused). Empty => no CFG conditioning.
  std::string              _negative_prompt;

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

  // (Re)load the text encoder + embed table from _enc_dir into _encoder/_embed
  // (also sets _enc_hidden). Returns false and logs on failure. Called by
  // initialize() and to lazily reload after a _free_encoder free. mc non-null.
  bool load_encoder_(metal_compute::MetalCompute* mc);

  // The full prompt -> unpacked-latent forward: tokenize, encode (12-layer
  // tap), fuse text, sample the FlowMatchEuler turbo steps and unpack. When
  // `init_latent` is non-null (a whitened latent [z, H/8, W/8], from the
  // vae-encode stage) with strength>0 it is an img2img run: scale_noise init,
  // denoising only the tail steps. `init_packed` (when non-null) overrides the
  // packed init (repro / golden). `refs` are optional reference-conditioning
  // latents ([z_dim, h, w] channel-first) packed to DiT tokens + given per-
  // reference RoPE frame offsets (Qwen-Image-Edit multi-reference; only steers
  // a reference-trained checkpoint). Returns the unpacked latent [z, H/8, W/8]
  // (whitened) or empty.
  std::vector<float>
  generate_(const std::string& prompt, const std::string& negative,
            int gen_h, int gen_w,
            const std::vector<float>* init_packed,
            const std::vector<float>* init_latent,
            const std::vector<RefLatent>& refs) const;

  // FLUX.2 forward: tokenize -> Qwen3 dense encoder {10,20,30}-layer tap concat
  // ([n, 7680]) -> FLUX DiT sampler loop (no text-fusion tower, no CFG). `refs`
  // are optional reference images (patchify-packed to DiT tokens + given per-
  // reference RoPE T offsets inside the DiT). Returns the DiT-facing latent
  // [dit_channels, H/16, W/16] (channel-first) or empty.
  std::vector<float>
  generate_flux2_(const std::string& prompt, int gen_h, int gen_w,
                  const std::vector<RefLatent>& refs) const;
#endif
};

}

#endif
