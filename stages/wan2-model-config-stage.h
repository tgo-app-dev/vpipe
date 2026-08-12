#ifndef VPIPE_STAGES_WAN2_MODEL_CONFIG_STAGE_H
#define VPIPE_STAGES_WAN2_MODEL_CONFIG_STAGE_H

#include "stages/model-config-source.h"

#include <string>
#include <vector>

namespace vpipe {

// Source stage: the WAN-specific half of a `generate-video` request.
//
// Wan 2.1/2.2 needs three things no other video family here does, all of
// them consequences of how the checkpoints are built: it is NOT
// guidance-distilled, so it takes a classifier-free guidance scale; and
// A14B is TWO 14B experts switched partway down the sigma schedule, so
// the low-noise one takes a scale of its own and the crossing point is a
// number. A guidance-distilled single-stack family (MiniMax-H3) has no
// use for any of them.
//
// The beat is a FlexData object tagged `generate-video-config`; the
// consuming stage passes it to MetalWanTransformer::GenerationParams
// unread, so a knob added there needs no change in `generate-video`.
// See stages/model-config-source.h for the contract.
//
// With no trigger iport wired it emits once and signals done. With one
// wired, each inbound beat re-emits (any payload -- receipt is the
// signal), so guidance can change per clip in a continuously generating
// graph; EOS upstream ends the stage.
//
// Configuration (FlexData object):
//   guidance_scale   (real) -- CFG scale; on A14B, the HIGH-noise
//                              expert's. Needs a negative conditioning
//                              wired at the consumer, which without one
//                              forces guidance to 1 rather than paying
//                              for a second forward that cannot steer.
//   guidance_scale_2 (real) -- the LOW-noise expert's, on a two-expert
//                              checkpoint.
//   boundary_ratio   (real) -- the sigma at which the low-noise expert
//                              takes over. Set explicitly it OVERRIDES
//                              the checkpoint's own model_index.json;
//                              left out, the checkpoint decides.
class Wan2ModelConfigStage final
  : public ModelConfigSourceStage<Wan2ModelConfigStage> {
public:
  static constexpr const char* kTypeName = "wan2-model-config";

  Wan2ModelConfigStage(const SessionContextIntf* session,
                       std::string               id,
                       std::vector<InEdge>       iports,
                       FlexData                  config);

  const StageSpec& spec() const noexcept override;

  // The beat this stage emits. Public so a test can read it without
  // running a pipeline, and so the shape stays one expression.
  FlexData resolved_config() const;
  // The default log line would print the object; this one says where the
  // boundary CAME FROM, which is the part a reader cannot infer.
  void report_config(const FlexData& fd) const;

private:
  double _guidance   = 3.5;
  double _guidance_2 = 3.5;
  double _boundary   = 0.9;
  // Whether `boundary_ratio` was ASKED for. An absent key must not be
  // emitted at all: the consumer reads the checkpoint's model_index.json
  // for it, and a default sent as if it were a choice would override the
  // checkpoint on every graph that never mentioned the key.
  bool _boundary_set = false;
};

}  // namespace vpipe

#endif
