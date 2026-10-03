#ifndef VPIPE_STAGES_TENSOR_LIST_STAGE_H
#define VPIPE_STAGES_TENSOR_LIST_STAGE_H

#include "apple-silicon/tensor-beat.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {

// Several beats in, ONE list out: a general producer for the list ports.
//
// The multi-reference consumers take their pictures as ONE
// TensorListPayload -- diffusion-conditioner `ref_images`, vae-encode
// `images` (whose `latents` list feeds generate-image `ref_latents`) --
// so a graph can hold as many references as its model takes instead of
// the two fixed ports. Until this stage only load-image produced such a
// list, and only from files it reads itself: a picture that arrives as a
// BEAT -- resampled by image-resample, handed in by a host through
// external-input or a plugin source -- had no way into one.
//
// One beat from every wired iport makes one list, in IPORT order: item k
// is what came in on the k-th wired port, and an unwired slot is skipped
// rather than becoming an empty item. Order is the contract -- the
// conditioner labels pictures by position and the DiT pairs each latent
// with the picture at the same index -- so a list is all or nothing. A
// round cut short (one input ends before the others delivered) or
// holding something other than a tensor emits nothing, rather than a
// shorter list that pairs the wrong things.
//
// The items MOVE out of the beats read, so a Metal shared-buffer beat
// joins the list with no copy. Each item keeps its own sideband; the
// list's sideband is the `sideband` config, if any.
class TensorListStage final : public TypedStage<TensorListStage> {
public:
  static constexpr const char* kTypeName = "tensor-list";
  // Declared (typed) iports: ten, Qwen-Image-2.1's most references. The
  // ctor refuses a graph wiring past them.
  static constexpr unsigned kMaxItems = 10;

  TensorListStage(const SessionContextIntf* s, std::string id,
                  std::vector<InEdge> iports, FlexData config);

  Job process(RuntimeContext& ctx) override;
  Job initialize(RuntimeContext& ctx) override;
  const StageSpec& spec() const noexcept override;

  // Test-only.
  std::uint64_t lists_emitted() const noexcept { return _emitted; }

private:
  FlexData      _sideband_cfg;
  std::uint64_t _emitted = 0;
};

}  // namespace vpipe

#endif
