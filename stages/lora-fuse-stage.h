#ifndef VPIPE_STAGES_LORA_FUSE_STAGE_H
#define VPIPE_STAGES_LORA_FUSE_STAGE_H

#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe {

// Preparation stage: fuse a base model with a LoRA adapter, writing a new
// self-contained model with the LoRA baked into the weights (W + scale*dW)
// and registering it in the models DB.
//
// 1 optional trigger iport (any beat) + 1 FlexData "summary" oport, so it
// cascades with model-fetch / model-quantize into a setup recipe: when the
// trigger iport is wired the fusion waits for one beat before running, and on
// success it emits a summary beat (a `text` report field + the fused output's
// details) that both drives save-text and triggers the next recipe stage.
// Standalone (unwired) it still runs once at launch.
//
// The base is a safetensors model directory (for Krea-2-Turbo, the
// `transformer/` DiT -- the LoRA only adapts the DiT, and the fused DiT is used
// via the text-to-image stage's `dit_dir`, like a quantized DiT). The LoRA is a
// single .safetensors adapter. Two formats fuse: diffusers-named lora_A/B pairs
// and ai-toolkit / ComfyUI adapters (`diffusion_model.*` names, incl. LoKr
// lokr_w1/lokr_w2); keys map to base weights by name (leading-prefix strip +
// the ai-toolkit -> diffusers remap), and dW is B@A or kron(w1,w2).
//
// Config (FlexData object):
//   base_model  (string, required) -- base model dir (or a models-DB key).
//   lora        (string, required) -- LoRA .safetensors file, or a dir/key
//                                     containing exactly one .safetensors.
//   output_name (string, required) -- result name -> <cwd>/models/<name>
//                                     (registered), or an explicit "/.." path.
//   scale       (real, default 1.0) -- LoRA fusion strength (lora_scale *
//                                      alpha/rank; alpha=rank when unspecified).
//   models_db   (string, default "models") -- registry sub-db.
class LoraFuseStage final : public TypedStage<LoraFuseStage> {
public:
  static constexpr const char* kTypeName = "lora-fuse";

  LoraFuseStage(const SessionContextIntf* session,
                std::string               id,
                std::vector<InEdge>       iports,
                FlexData                  config);
  ~LoraFuseStage() override;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

  // Test-only accessors.
  const std::string& base_model() const noexcept { return _base_model; }
  double             scale()      const noexcept { return _scale; }

private:
  bool fuse_once(const std::function<bool()>& stop);
  void register_output_(const std::string& key, const std::string& dir);

  std::string _base_model;
  std::string _lora;
  std::string _output_name;
  std::string _models_db;
  double      _scale{};
  // Resolved output dir of the last fuse (for the summary beat).
  std::string _out_dir;
};

}

#endif
