#ifndef VPIPE_STAGES_DIFFUSION_CONDITIONER_STAGE_H
#define VPIPE_STAGES_DIFFUSION_CONDITIONER_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The prompt/image -> conditioning half of the diffusion split. Owns the
// tokenizer, the text encoder, and (for image-aware families) the Qwen2.5-VL
// vision tower; emits the per-family conditioning tensor the text-to-image
// (DiT) stage consumes. This is the from-scratch, MLX-free metal-compute path
// on the VPIPE_BUILD_APPLE_SILICON axis; an inert stub off it.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/qwen-image/metal-qwen25-vision.h"
#include "generative-models/qwen3/metal-qwen-model.h"
#include "generative-models/tokenizer.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// DiffusionConditionerStage: prompt text (+ optional reference image for
// image-aware models) -> conditioning embeddings for a diffusion DiT.
//
// This is the encoder half pulled out of the text-to-image stage, so the raw
// reference IMAGE goes to the VLM here (semantic understanding) while the VAE
// LATENT of that same image goes to the DiT stage (spatial detail) -- two
// distinct consumers, no longer conflated on one stage. The conditioner and the
// text-to-image stage are a MATCHED PAIR keyed by the same `hf_dir`: the
// conditioning tensor's shape + semantics are family-specific.
//
//   iport0  prompt        FlexDataPayload (string or {text: ...}), required.
//   iport1  negative       OPTIONAL FlexDataPayload negative prompt (for the
//                          DiT's classifier-free guidance). Its conditioning is
//                          emitted on oport1.
//   iport2  ref_image      OPTIONAL planar U8 RGB TensorBeat [3,H,W] (load-image
//                          format). Image-aware families (Qwen-Image-Edit) run
//                          it through the Qwen2.5-VL vision tower to make the
//                          prompt embeds understand the source; text-only
//                          families (FLUX.2, Krea-2) ignore it. Latched once.
//
//   oport0  conditioning   TensorBeatPayload bf16, family-shaped:
//                            krea2  [n_real, 12, 2560] (12-tap; the DiT fuses)
//                            flux2  [n_real, 3*enc_hidden] (concatenated taps)
//                            qwen-image-edit [n_real, 3584] (last-hidden,
//                              image-aware, POST encoder final-norm)
//   oport1  neg_conditioning  same shape, emitted only when a negative is set.
//
// Config (FlexData object):
//   hf_dir     (string, required) -- the model dir (text_encoder/, transformer/,
//                                    tokenizer/); the transformer's _class_name
//                                    selects the family + encoder.
//   models_db  (string, default "models") -- registry for resolve_model_dir.
class DiffusionConditionerStage final
    : public TypedStage<DiffusionConditionerStage> {
public:
  static constexpr const char* kTypeName = "diffusion-conditioner";

  DiffusionConditionerStage(const SessionContextIntf* session,
                            std::string               id,
                            std::vector<InEdge>       iports,
                            FlexData                  config);
  ~DiffusionConditionerStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process   (RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  const std::string& hf_dir() const noexcept { return _hf_dir; }
  std::uint64_t conditionings_emitted() const noexcept { return _emitted; }

private:
  std::string _hf_dir;
  std::string _enc_dir;
  std::string _models_db;
  std::string _family = "krea2";   // krea2 | flux2 | qwen-image-edit
  std::uint64_t _emitted = 0;

#ifdef VPIPE_BUILD_APPLE_SILICON
  std::unique_ptr<genai::MetalQwenModel>          _encoder;
  std::unique_ptr<genai::Tokenizer>               _tokenizer;
  metal_compute::SharedBuffer                     _embed;      // encoder embeds
  mutable std::unique_ptr<genai::MetalQwen25Vision> _vision;   // QIE, lazy
  int _enc_hidden = 2560;

  // Cached negative prompt + raw reference image, latched once like the DiT
  // stage's negative / ref-latent inputs.
  std::string               _negative_prompt;
  bool                      _negative_latched = false;
  std::vector<std::uint8_t> _ref_rgb;
  int _ref_rgb_h = 0, _ref_rgb_w = 0;

  bool load_encoder_(metal_compute::MetalCompute* mc);

  // Encode `text` (+ the cached reference image, for image-aware families) into
  // the family conditioning tensor; returns the bf16 buffer + sets `n_real`.
  // `vtok`/`n_img` carry the (already-computed) vision tokens for QIE (empty
  // for text-only). Empty buffer on failure.
  metal_compute::SharedBuffer
  encode_(const std::string& text, const char* which, int& n_real,
          const metal_compute::SharedBuffer& vtok, int n_img) const;

  // Run the reference image through the vision tower (QIE only) -> [n_img, 3584]
  // vision tokens; sets n_img. Empty when no ref image / not image-aware.
  metal_compute::SharedBuffer vision_tokens_(metal_compute::MetalCompute* mc,
                                             int& n_img) const;
#endif
};

}  // namespace vpipe

#endif
