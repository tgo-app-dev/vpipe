#ifndef VPIPE_STAGES_MAGE_FLOW_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_MAGE_FLOW_MODEL_CONFIG_STAGE_H

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source stage: the MAGE-FLOW-specific parameters, read by both halves of
// the graph.
//
// Two things are this family's alone. It puts a Gaussian-Shading
// PROVENANCE WATERMARK in the initial noise -- no other model here does
// -- and its reference pipeline grounds a conditioning image at its own
// resolution (`vl_cond_long_edge` 384, with processor bounds far above
// the Qwen defaults so a small reference is upscaled rather than
// patched as-is).
//
// Those two land in different stages: the watermark in `generate-image`,
// the grounding in `diffusion-conditioner`. They are still ONE stage,
// because they are one checkpoint's facts -- see the note on kConfigTag
// in stages/model-config-source.h. Wire this oport to both.
//
// Configuration (FlexData object):
//   no_watermark  (bool) -- DISABLE the provenance watermark. Negative-
//                           named so the safe default (on, as the
//                           reference applies it) needs no config; it is
//                           distribution-preserving, so it costs no
//                           image quality.
//   watermark_key (string) -- Gaussian-Shading key: an integer or a
//                           passphrase. Unset => $MAGEFLOW_GS_KEY, else
//                           $MAGEFLOW_GS_KEY_FILE / ~/.mageflow/gs_key,
//                           else the published default. The detector
//                           needs the SAME key.
//   vl_long_edge / vl_pixel_budget / vl_min_pixels / vl_max_pixels
//                          -- the grounded encode's preprocessing; see
//                           genai::GroundedEncodeParams. Omit them and
//                           the family's own numbers apply.
class MageFlowModelConfigStage final
  : public ModelConfigSourceStage<MageFlowModelConfigStage> {
public:
  static constexpr const char* kTypeName = "mage-flow-model-config";

  MageFlowModelConfigStage(const SessionContextIntf* session,
                           std::string               id,
                           std::vector<InEdge>       iports,
                           FlexData                  config);

  const StageSpec& spec() const noexcept override;

  FlexData resolved_config() const;

private:
  bool        _no_watermark = false;
  std::string _watermark_key;
  // The grounded-encode overrides are NOT held as members: they are
  // forwarded from the config only when actually set (see
  // model_config::copy_grounded_keys), because an unset one has to leave
  // the model layer's per-family number alone rather than replace it
  // with a default that merely looks configured.
};

}  // namespace vpipe

#endif
