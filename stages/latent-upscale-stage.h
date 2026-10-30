#ifndef VPIPE_STAGES_LATENT_UPSCALE_STAGE_H
#define VPIPE_STAGES_LATENT_UPSCALE_STAGE_H

#include "common/flex-data.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

namespace genai {
class WeightSet;
class MetalMiniMaxH3LatentUpscaler;
}

// Enlarges a video LATENT without decoding it: MiniMax-H3's latent
// upscaler (LBH-123-AI's H3-Latent-Upscaler, a 3D-conv network), which
// Open-H3-Regenerate-2K runs between encoding a draft and refining it at
// the larger size -- a 768P latent (48 x 84 cells) to 2K (90 x 160).
//
//   iport0  latent  f32 [24, T, h, w], whitened, as vae-encode and
//                   generate-video emit it
//   iport1  model   OPTIONAL model-select reference; overrides hf_dir
//
//   oport0  latent  f32 [24, T, H, W], whitened, the input's sideband
//                   carried; when the requested size is not whole 32-px
//                   cells (1080 rows compute at 1088, 2160 at 2176) a
//                   centred `crop_box` that vae-decode cuts the picture
//                   back to
//
// The target is the upscaler node's own rule: `width` x `height` pixels
// rounded to `align` (Python's round), then to whole latent cells; or
// `scale` times the latent's own pixel size. The network is told the
// ratio (an embedding of scale - 1), so the size is not only a resize.
// Above `two_pass_above` rows it runs twice, to that height first, as
// the release does for 4K.
//
// Config:
//   hf_dir         the upscaler: a models-DB key, the Open-H3-Regenerate-
//                  2K release (its latent_upsampler/), the upscaler's own
//                  repo, or the .safetensors
//   width, height  target pixels (both, or `scale`)
//   scale          target as a factor of the input's pixel size
//   align          pixel alignment of the target, 32 (the node's)
//   two_pass_above rows above which the target is reached in two passes
//                  (1440; 0 never)
//   normalization  release | once -- see MetalMiniMaxH3LatentUpscaler
//   frames         planning only: the clip length the plan books the
//                  pass's activations at (the latent says the truth)
//   unload_when_idle auto | destroy | keep
class LatentUpscaleStage final : public TypedStage<LatentUpscaleStage> {
public:
  static constexpr const char* kTypeName = "latent-upscale";

  LatentUpscaleStage(const SessionContextIntf* session, std::string id,
                     std::vector<InEdge> iports, FlexData config);
  ~LatentUpscaleStage() override;

  Job initialize(RuntimeContext& ctx) override;
  Job process(RuntimeContext& ctx) override;
  void reset_run_state() override;
  void apply_constant(unsigned iport, const FlexData& beat) override;

  std::vector<ResourceClaim> declare_resources() const override;
  StageMemory declare_memory() const override;

  const StageSpec& spec() const noexcept override;

  // Test-only.
  std::int64_t latents_emitted() const noexcept { return _emitted; }

private:
  std::string weights_file_(std::string* err = nullptr) const;
  bool ensure_loaded_();
  void release_();
  // The output cells for an input of h x w cells, and the pixels asked
  // for (the crop box's size).
  bool target_(int h, int w, int* H, int* W, double* scale, int* want_w,
               int* want_h) const;

  std::string _hf_dir;
  int    _width = 0, _height = 0, _align = 32, _two_pass_above = 1440;
  int    _plan_frames = 124;
  double _scale = 0.0;
  bool   _norm_once = false;
  std::string _unload;
  // i8_gemm / i8_gemm_tier: -1 off, 0 accurate .. 3 token_plain; and what
  // it settles to on this box, for the plan.
  int _i8_tier = -1;
  int effective_i8_tier_() const;
  // A pass's activations at the planned geometry, for both ledgers; and
  // the real figure, published once a latent says what it is.
  std::string scratch_label_() const;
  std::size_t planned_scratch_() const;
  void revise_scratch_(std::size_t bytes);
  std::size_t _scratch_stated = 0;

  bool _model_latched = false;
  std::string _loaded_file;
  std::shared_ptr<genai::WeightSet> _ws;
  std::unique_ptr<genai::MetalMiniMaxH3LatentUpscaler> _model;
  std::int64_t _emitted = 0;
};

}  // namespace vpipe

#endif  // VPIPE_STAGES_LATENT_UPSCALE_STAGE_H
