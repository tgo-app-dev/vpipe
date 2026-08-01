#ifndef VPIPE_STAGES_SAMPLER_SELECT_STAGE_H
#define VPIPE_STAGES_SAMPLER_SELECT_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Source stage: program the LLM TOKEN sampler and emit its spec as a FlexData
// beat, once, on oport0. The generative LLM stages (`text-chat`, `visual-qa`,
// `realtime-vqa`, `audio-transcribe`, `text-to-speech`) latch the spec off
// their optional sampler iport; an unwired port leaves the stage on GREEDY
// (argmax) decoding -- except text-to-speech's AUDIO channel, which falls
// back to MOSS's own sampling because greedy audio degenerates.
//
// Mirrors `diffusion-sampler-select` (which programs the *diffusion*
// integrator) but is a different thing entirely: this one carries the
// token-sampling knobs and its spec is tagged `"sampler":"token"` so a
// consumer can reject a mis-wired diffusion beat instead of silently reading
// zeros out of it.
//
// The spec is exactly what `genai::parse_sampler_config` reads:
//   { sampler:"token", temperature, top_k, top_p, min_p,
//     repetition_penalty, presence_penalty, seed }
//
// Every knob at its default reduces to argmax, so a bare `sampler-select` with
// no config decodes greedily -- same as not wiring one at all. Set at least
// `temperature` (or top_k / top_p / min_p) to actually sample.
//
// Config (FlexData object, all optional):
//   temperature        (real, default 1.0) -- softmax temperature; <= 0 forces
//                                             argmax.
//   top_k              (int,  default 0)   -- keep the top k logits; 0 = off.
//   top_p              (real, default 1.0) -- nucleus; >= 1 = off.
//   min_p              (real, default 0.0) -- drop below min_p * max_prob;
//                                             <= 0 = off.
//   repetition_penalty (real, default 1.0) -- 1.0 = off.
//   presence_penalty   (real, default 0.0) -- 0.0 = off.
//   seed               (uint, default 0)   -- 0 = fresh non-deterministic seed.
class SamplerSelectStage final : public TypedStage<SamplerSelectStage> {
public:
  static constexpr const char* kTypeName = "sampler-select";

  SamplerSelectStage(const SessionContextIntf* session,
                     std::string               id,
                     std::vector<InEdge>       iports,
                     FlexData                  config);
  ~SamplerSelectStage() override;

  void reset_run_state() override;
  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only: the resolved spec that would be emitted.
  FlexData resolved_spec() const;

private:
  double        _temperature = 1.0;
  std::int64_t  _top_k = 0;
  double        _top_p = 1.0;
  double        _min_p = 0.0;
  double        _repetition_penalty = 1.0;
  double        _presence_penalty = 0.0;
  std::uint64_t _seed = 0;
  std::uint64_t _emitted = 0;
};

}

#endif
