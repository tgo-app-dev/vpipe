#ifndef VPIPE_GENERATIVE_MODELS_TRAIN_LORA_PARAMS_H
#define VPIPE_GENERATIVE_MODELS_TRAIN_LORA_PARAMS_H

// The trainable half of a LoRA: every adapted projection's A [rank, k] and
// B [n, rank], held three ways.
//
//   masters   f32, what the optimizer updates;
//   grads     f32, what a backward accumulates into;
//   shadows   the storage type, what the FORWARD reads -- bound into the
//             model as ordinary lora::Factors, so training runs the same
//             runtime-LoRA path inference does.
//
// All of each kind lives in ONE flat arena (every A, then every B), and a
// module's factors are subviews of it: the optimizer is two dispatches,
// not two per module, and nothing is copied between the optimizer and the
// forward.
//
// A shadow is written with the runtime loader's own arithmetic --
// ELT(ELT(master) * alpha / rank) for A, ELT(master) for B -- so an adapter
// saved as bf16 and bound again by lora::Adapter reproduces exactly the
// factors the trainer last ran. That is the "one forward" contract, and a
// test holds it to the bit.
//
// The file is the diffusers / PEFT spelling:
//   transformer.<module>.lora_A.weight  [rank, k]
//   transformer.<module>.lora_B.weight  [n, rank]
// with alpha and rank in __metadata__, which every reader in this tree,
// diffusers and ComfyUI all take.

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/runtime-lora.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace train {

// One projection an adapter can target: its weight is [n, k].
struct ModuleShape {
  std::string name;   // the module path, e.g. "transformer_blocks.3.attn.to_q"
  int n = 0;
  int k = 0;
};

class LoraParams {
 public:
  struct Module {
    std::string name;
    int n = 0;
    int k = 0;
    std::size_t a_off = 0;   // elements into the A arenas
    std::size_t b_off = 0;   // elements into the B arenas
    lora::Factors live;      // shadow subviews, bound into the model
  };

  // Allocate every arena for `mods` at `rank`. Masters are zero and the
  // shadows unwritten until init_random(), load() or refresh.
  bool create(metal_compute::MetalCompute* mc,
              const std::vector<ModuleShape>& mods, int rank, float alpha,
              std::string* err);

  // A Kaiming-uniform (PEFT's default: bound 1 / sqrt(k)), B zero -- so
  // step 0 is exactly the base model. Writes the shadows on the CPU.
  void init_random(std::uint64_t seed);

  // Masters from an adapter file of matching modules and rank (any float
  // dtype). Modules the file lacks keep their current values; a module
  // of the wrong shape is refused. `loaded` counts the modules read.
  bool load(const std::string& path, int* loaded, std::string* err);

  // The adapter as bf16, from the masters (or from `ema`, an f32 arena of
  // the masters' layout), with `meta` added to alpha and rank.
  bool save(const std::string& path,
            const std::map<std::string, std::string>& meta,
            std::string* err,
            const metal_compute::SharedBuffer* ema_a = nullptr,
            const metal_compute::SharedBuffer* ema_b = nullptr) const;

  // Rewrite the shadows from the masters on the CPU (after load, resume,
  // or anything else that touched the masters outside the optimizer).
  void refresh_shadows_cpu();

  int rank() const noexcept { return _rank; }
  float alpha() const noexcept { return _alpha; }
  // What an A shadow is multiplied by: alpha / rank.
  float a_mul() const noexcept { return _alpha / (float)_rank; }
  std::size_t a_elems() const noexcept { return _na; }
  std::size_t b_elems() const noexcept { return _nb; }
  std::size_t param_count() const noexcept { return _na + _nb; }

  std::vector<Module>& modules() noexcept { return _mods; }
  const std::vector<Module>& modules() const noexcept { return _mods; }
  // By module path; null when not trained.
  const Module* find(const std::string& name) const;

  const metal_compute::SharedBuffer& a_master() const { return _a_m; }
  const metal_compute::SharedBuffer& b_master() const { return _b_m; }
  const metal_compute::SharedBuffer& a_grad() const { return _a_g; }
  const metal_compute::SharedBuffer& b_grad() const { return _b_g; }
  const metal_compute::SharedBuffer& a_shadow() const { return _a_s; }
  const metal_compute::SharedBuffer& b_shadow() const { return _b_s; }

  // Bytes the whole set holds (masters, grads, shadows), for the plan.
  static std::size_t bytes_for(std::size_t params) noexcept
  {
    return params * (4 + 4 + 2);
  }

 private:
  metal_compute::MetalCompute* _mc = nullptr;
  int _rank = 0;
  float _alpha = 0.0f;
  std::size_t _na = 0, _nb = 0;
  std::vector<Module> _mods;
  metal_compute::SharedBuffer _a_m, _b_m, _a_g, _b_g, _a_s, _b_s;
};

// ---- a single-file safetensors writer ------------------------------------

struct OutTensor {
  std::string name;
  std::string dtype;                  // "BF16", "F32", ...
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};

// Writes to `path` + ".tmp" and renames, so a reader never sees half a
// file; `meta` becomes __metadata__.
bool write_safetensors(const std::string& path,
                       const std::vector<OutTensor>& tensors,
                       const std::map<std::string, std::string>& meta,
                       std::string* err);

// Round-to-nearest-even, as every loader here does.
std::uint16_t f32_to_bf16(float f) noexcept;
float bf16_to_f32(std::uint16_t h) noexcept;

}  // namespace train
}  // namespace genai
}  // namespace vpipe

#endif
