#ifndef VPIPE_STAGES_FLASHVSR_SRC_ENCODER_STAGE_H
#define VPIPE_STAGES_FLASHVSR_SRC_ENCODER_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

// The source projection is a from-scratch, MLX-free metal-compute module
// on the VPIPE_BUILD_APPLE_SILICON axis; an inert stub off it.
#ifdef VPIPE_BUILD_APPLE_SILICON
#include "generative-models/flashvsr/flashvsr-lq-proj.h"
#include "stages/model-memory.h"
#endif

#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// FlashVSR's source encoder: a low-quality CLIP in pixels -> the
// conditioning rows its denoiser adds into the residual stream.
//
// WHY ITS OWN STAGE, and not a branch of `diffusion-conditioner`. That
// stage's own warning says it: a family needing another encoder ships
// its own conditioner. This family needs one that shares nothing with
// the text path -- no prompt, no tokenizer, no negative, no guidance,
// and no text encoder at all, because its text context is a CONSTANT
// baked into the checkpoint. What it has instead is a learned causal
// convolution over pixels. Branching the shared conditioner on that
// would have cost it a pixel-shaped input port and a fourth family, to
// express something that is not a variant of what it does.
//
// So this stage is MODEL-SPECIFIC in the same sense the `*-model-config`
// stages are, and carries the same category: its applicability is
// decided by the checkpoint rather than by the graph. A
// `flashvsr-src-encoder` is meaningless anywhere else, which is exactly
// what lets tooling offer it only where it belongs.
//
// WHAT IT EMITS IS ORDINARY CONDITIONING. The projection's output is
// [tokens, hidden] bf16 -- the shape a conditioning beat already has --
// so it feeds `generate-video`'s existing conditioning port and NOTHING
// in that stage changed to accept it. That is the whole point of putting
// the projection here: the graph reads like every other generative
// graph, conditioner into generator.
//
//   iport0  source   TensorBeatPayload, planar U8 RGB [frames, 3, H, W]
//                    with `fps` on the sideband -- the shape
//                    `temporal-stack` builds. REQUIRED: this family
//                    conditions on its source and nothing else, so an
//                    unwired port is a refusal rather than a fallback.
//                    Read fresh every beat, never latched: a restoration
//                    graph hands a different clip each time.
//   iport1  model    OPTIONAL FlexDataPayload model reference (from a
//                    `model-select` source); OVERRIDES the hf_dir config
//                    so the encoder, the denoiser and the VAE of one
//                    graph share a single model choice. When wired, the
//                    load is DEFERRED to the first process() -- the beat
//                    only arrives after the init barrier.
//
//   oport0  conditioning  TensorBeatPayload bf16 [row_frames * tokens,
//                    hidden]: one row frame per FOUR source frames (the
//                    VAE's temporal compression, which the projection
//                    matches with two stride-2 causal convolutions) and
//                    one token per cell of the denoiser's own grid (the
//                    frame over 16). The sideband carries the geometry
//                    the rows belong to -- source width, height, frame
//                    count and row frame count -- because a row count
//                    alone cannot say which of those produced it.
//
// THE PROJECTION IS CAUSAL, which shapes how this stage drives it. It
// carries the last two input frames of each convolution across calls,
// and its OPENING call seeds those carries and produces no rows at all.
// So a clip is fed in the reference's own irregular pattern rather than
// one shot: seven slices to open, the first of them a single frame, then
// two per denoise chunk. One call over the whole clip returns nothing.
//
// Config (FlexData object):
//   hf_dir  (string, OPTIONAL) -- the FlashVSR model directory. A
//           model-select source on iport1 overrides it; required only
//           when iport1 is unwired.
class FlashVsrSrcEncoderStage final
    : public TypedStage<FlashVsrSrcEncoderStage> {
public:
  static constexpr const char* kTypeName = "flashvsr-src-encoder";

  FlashVsrSrcEncoderStage(const SessionContextIntf* session,
                          std::string               id,
                          std::vector<InEdge>       iports,
                          FlexData                  config);
  ~FlashVsrSrcEncoderStage() override;

  Job initialize(RuntimeContext& ctx) override;

  // The projection is ~0.6 GB and is resident while a clip is encoded.
  // It is the only model this stage holds, and nothing else declares it
  // -- the denoiser stopped owning it when the projection moved here, so
  // a missing declaration here is a real hole rather than a duplicate.
  std::vector<ResourceClaim> declare_resources() const override;
  // ...and on the memory plan, where it read 0 MB.
  StageMemory declare_memory() const override;

  // Latch a `model-select` constant before the planning phase, so the
  // claim above is made against the model this graph will actually run
  // rather than against an empty hf_dir.
  void apply_constant(unsigned iport, const FlexData& beat) override;
  void reset_run_state() override;
  Job  process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& hf_dir()   const noexcept { return _hf_dir; }
  std::uint64_t      emitted()  const noexcept { return _emitted; }

private:
#ifdef VPIPE_BUILD_APPLE_SILICON
  bool ensure_loaded_();

  std::unique_ptr<genai::FlashVsrLqProj> _proj;
#endif
  std::string   _hf_dir;
  bool          _model_latched = false;
  bool          _load_attempted = false;
  std::uint64_t _emitted = 0;
};

}  // namespace vpipe

#endif
