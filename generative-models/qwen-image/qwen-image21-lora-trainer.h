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

#include <map>
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
    // A RANK (cfg.tp): the widest sample's joint rows, its exchange
    // buffers made at them from the first (set_tp_exchange_floor).
    int floor_rows = 0;
  };

  // Whether `transformer_dir` holds this family's DiT.
  static bool claims(const std::string& transformer_dir);
  static std::unique_ptr<QwenImage21LoraTrainer> load(const LoadArgs& a,
                                                      std::string* err);

  const char* family() const noexcept override { return "qwen-image-21"; }
  std::vector<train::ModuleShape>
  modules(const std::string& targets) const override;
  int bind(train::LoraParams* params, std::string* err) override;
  metal_compute::SharedBuffer pack_latent(const float* chw, int c, int t,
                                          int h, int w,
                                          std::string* err) override;
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
  std::size_t arena_bytes(int frames, int grid_h, int grid_w, int text_rows,
                          int rank, bool keep_all) const override;
  std::uint64_t resident_bytes() const override;

  // A RANK of a run split across hosts (the model's Config::tp).
  bool split() const override { return _m->config().tp.active(); }
  bool split_cuts(const std::vector<train::ModuleShape>& mods,
                  std::map<std::string, train::ModuleCut>* out,
                  std::string* err) const override
  {
    return _t->split_cuts(mods, out, err);
  }
  bool split_begin(std::shared_ptr<TpChannel> ch, std::uint64_t first_seq,
                   std::string* err) override
  {
    if (!_m->begin_tp_generation(std::move(ch), err, first_seq)) {
      return false;
    }
    _m->set_tp_meet(true);   // a rank's tuning waited out on the CPU
    return true;
  }
  void split_end() override { _m->end_tp_generation(); }
  std::size_t split_exchange_bytes() const override { return _split_bytes; }
  bool split_bulk(const void* mine, void* others, std::size_t bytes,
                  TpDType dtype, std::string* err) override
  {
    return _m->tp_bulk(mine, others, bytes, dtype, err);
  }

  MetalQwenImage21Transformer* model() const noexcept { return _m.get(); }
  QwenImage21Trainer* trainer() const noexcept { return _t.get(); }

 private:
  QwenImage21LoraTrainer() = default;
  // The joint layout of a text-to-image sample: the conditioning's rows
  // (its slot mask, or all text), then a grid_h x grid_w target.
  bool layout_(const FlexData* sideband, int text_rows, int gh, int gw,
               qi21::Layout* lay, std::string* err) const;

  metal_compute::MetalCompute* _mc = nullptr;
  std::size_t _split_bytes = 0;         // split_exchange_bytes
  std::unique_ptr<MetalQwenImage21Transformer> _m;
  std::unique_ptr<QwenImage21Trainer> _t;
  std::function<bool()> _stop;
};

}  // namespace genai
}  // namespace vpipe

#endif
