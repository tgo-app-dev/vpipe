#ifndef GENERATIVE_MODELS_QWEN_IMAGE_QWEN_IMAGE21_LORA_TRAINER_H
#define GENERATIVE_MODELS_QWEN_IMAGE_QWEN_IMAGE21_LORA_TRAINER_H

// Qwen-Image-2.1 behind train::LoraTrainer: the model, its trainer, and
// the family's own knowledge -- how a conditioning's slot mask and a
// target grid become a joint layout, how a latent is packed, which
// schedule inference samples on, and how a preview is sampled.

#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"
#include "generative-models/qwen-image/qwen-image21-layout.h"
#include "generative-models/qwen-image/qwen-image21-trainer.h"
#include "generative-models/train/lora-trainer.h"

#include <memory>
#include <string>

namespace vpipe {
namespace genai {

class WeightSet;

class QwenImage21LoraTrainer final : public train::LoraTrainer {
 public:
  struct LoadArgs {
    std::shared_ptr<WeightSet> ws;         // the transformer's weights
    metal_compute::MetalCompute* mc = nullptr;
    MetalQwenImage21Transformer::Config cfg;   // dims + accel tiers
    bool stream_blocks = false;
  };

  // Whether `transformer_dir` holds this family's DiT.
  static bool claims(const std::string& transformer_dir);
  static std::unique_ptr<QwenImage21LoraTrainer> load(const LoadArgs& a,
                                                      std::string* err);

  const char* family() const noexcept override { return "qwen-image-21"; }
  std::vector<train::ModuleShape>
  modules(const std::string& targets) const override;
  int bind(train::LoraParams* params, std::string* err) override;
  metal_compute::SharedBuffer pack_latent(const float* chw, int c, int h,
                                          int w, std::string* err) override;
  int latent_channels() const noexcept override;
  bool step(const train::TrainSample& s, train::StepStats* out,
            std::string* err) override;
  bool eval_loss(const train::TrainSample& s, double* loss,
                 std::string* err) override;
  double schedule_mu(int grid_h, int grid_w) const noexcept override;
  bool preview(const train::PreviewRequest& r, std::vector<float>* chw,
               std::string* err) override;
  void set_stop(std::function<bool()> stop) override;
  void set_options(const FlexData& o) override;
  std::size_t arena_bytes(int grid_h, int grid_w, int text_rows, int rank,
                          bool keep_all) const override;
  std::uint64_t resident_bytes() const override;

  MetalQwenImage21Transformer* model() const noexcept { return _m.get(); }
  QwenImage21Trainer* trainer() const noexcept { return _t.get(); }

 private:
  QwenImage21LoraTrainer() = default;
  // The joint layout of a text-to-image sample: the conditioning's rows
  // (its slot mask, or all text), then a grid_h x grid_w target.
  bool layout_(const FlexData* sideband, int text_rows, int gh, int gw,
               qi21::Layout* lay, std::string* err) const;

  metal_compute::MetalCompute* _mc = nullptr;
  std::unique_ptr<MetalQwenImage21Transformer> _m;
  std::unique_ptr<QwenImage21Trainer> _t;
  std::function<bool()> _stop;
};

}  // namespace genai
}  // namespace vpipe

#endif
