#ifndef VPIPE_GENERATIVE_MODELS_HIDDEN_STATE_ENCODER_H
#define VPIPE_GENERATIVE_MODELS_HIDDEN_STATE_ENCODER_H

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {
class SessionContextIntf;
namespace metal_compute { class MetalCompute; }
}

namespace vpipe::genai {

// Reading a language model's PER-LAYER HIDDEN STATES, for the diffusion
// text encoders that condition on a stack of them rather than on the
// last one.
//
// WHY THIS IS AN INTERFACE AND NOT A METHOD. Three families in this tree
// already do it and each reached into a different model: Krea-2 and
// FLUX.2 tap Qwen3 at {10,20,30}, Qwen-Image-Edit takes the last hidden,
// and LTX-2.5 consumes ALL of them (49 for Gemma-4 12B) through a single
// 188160-wide projection. Each of those wrote its own call into its own
// model's header. A fourth would too, and an out-of-tree family cannot
// -- a plugin has no way to reach MetalGemmaModel or MetalQwenModel.
//
// So the tap is stated once, here, in the vocabulary the REFERENCES use
// (HuggingFace's `output_hidden_states`), and each model implements it.
//
// ---- THE INDEX CONVENTION, which is the part that bites ----
//
// `indices` are HuggingFace's, exactly:
//
//   0            the EMBEDDING output -- the input to layer 0
//   k (1..L-1)   the output of layer k-1, un-normed
//   L            the output of layer L-1 AFTER THE FINAL NORM
//
// so a model with L layers offers L+1 states. The last one is NOT the
// raw residual: HF appends inside the loop before each layer and once
// more after `self.norm`, so index L has been through it. A tap that
// returns the un-normed residual there is a plausible vector of the
// right shape that no reference produces -- and a consumer stacking all
// L+1 states (LTX-2.5) hits it every time.
struct HiddenTapRequest {
  std::vector<int> indices;   // HF indices, each in [0, n_layers]
  // Right-padded prompts: every query attends only to keys with
  // position < key_valid_len, matching HF's attention_mask over the
  // real prefix. 0 disables (the whole sequence is real).
  int key_valid_len = 0;
};

// [slot][pos][hidden] in the model's compute dtype, slot j being
// `request.indices[j]`. One buffer rather than N so a consumer that
// concatenates them (every one of them does) reads a single allocation.
struct HiddenTapResult {
  metal_compute::SharedBuffer data;
  int slots = 0;
  int tokens = 0;
  int hidden = 0;
  // The dtype the buffer holds, so a consumer converts once and
  // correctly rather than assuming. "bf16" or "f16".
  std::string dtype;

  bool valid() const { return !data.empty() && slots > 0 && tokens > 0; }
};

// A language model that can be asked for hidden states.
//
// Deliberately NARROW: it does not tokenize, sample, or manage KV
// beyond what one encode needs. A diffusion text encoder wants exactly
// this and nothing else, and keeping it this small is what lets a model
// implement it without exposing its whole surface.
class HiddenStateEncoder {
public:
  virtual ~HiddenStateEncoder() = default;

  virtual int n_layers() const noexcept = 0;
  virtual int hidden_dim() const noexcept = 0;

  // The highest index available PER TOKEN. Normally n_layers(), but a
  // model whose tail is computed for the last position only reports the
  // boundary instead -- Gemma-4's KV-shared layers do exactly that, and
  // asking past it is refused rather than answered with one row.
  //
  // A consumer of the FULL stack (LTX-2.5) must check this: a stack that
  // silently stops short is a projection fed the wrong rows.
  virtual int max_tap_index() const noexcept { return n_layers(); }

  // Encode `ids` and return the requested states. False (with `err`) on
  // a bad index or a backend failure; never throws.
  //
  // An index outside [0, n_layers()] is REFUSED rather than clamped: a
  // silently-clamped tap reads a different layer than the caller asked
  // for and nothing downstream can tell.
  virtual bool encode(const std::vector<std::int32_t>& ids,
                      const HiddenTapRequest&          req,
                      HiddenTapResult*                 out,
                      std::string*                     err) = 0;

  // Every index this encoder can actually serve, in order.
  HiddenTapRequest available_indices(int key_valid_len = 0) const
  {
    return all_indices(max_tap_index(), key_valid_len);
  }

  // Every index in [0, n_layers], in order -- what a consumer of the
  // full stack wants. Prefer available_indices() unless the caller has
  // already checked max_tap_index().
  static HiddenTapRequest all_indices(int n_layers, int key_valid_len = 0)
  {
    HiddenTapRequest r;
    r.key_valid_len = key_valid_len;
    r.indices.reserve((std::size_t)n_layers + 1);
    for (int i = 0; i <= n_layers; ++i) { r.indices.push_back(i); }
    return r;
  }
};

// Everything a factory receives to open one.
struct HiddenStateEncoderArgs {
  // The checkpoint: a HuggingFace directory, or a single comfy-style
  // .safetensors file.
  std::string                  dir;
  metal_compute::MetalCompute* metal   = nullptr;
  const SessionContextIntf*    session = nullptr;

  // The model config, when the caller already has it.
  //
  // WHY THIS EXISTS. A comfy-style pack carries its config in the
  // safetensors `__metadata__` and ships no config.json -- and it may
  // NEST it: LTX-2.5's text encoder is one file whose Gemma-4 config
  // sits under a `gemma_config` key next to LTX's own fields. Only the
  // caller knows how to unwrap that, so it hands the unwrapped object
  // in rather than the factory guessing at a layout it cannot know.
  //
  // Empty (a null FlexData) means "read it from `dir`", which is what a
  // plain HF checkpoint wants.
  FlexData                     config;

  // The architecture, when the caller knows it and the config does not
  // say. Only consulted by open(); create() is already told.
  std::string                  arch;

  // Stream the backbone's layers instead of holding them, for a caller
  // that only ever PREFILLS -- which every consumer of this interface
  // does, by construction: it reads a stack of hidden states out of one
  // forward and never decodes.
  //
  // A 12B text encoder at w8 is 15.3 GB resident and peaks at 16.14 GB
  // loading, which on a 16 GB box is the machine; streamed it is one
  // layer plus the pinned prefix. The cost is re-reading the stack per
  // prefill, which for one caption is a fraction of what the diffusion
  // model beside it spends per STEP.
  //
  // `pin_frac` sizes that prefix as a fraction of RAM; 0 is pure
  // streaming. Both are HINTS -- a backbone with no streaming support
  // ignores them and loads as it always did, because refusing here
  // would make a memory optimisation into a compatibility break.
  bool                         stream_layers = false;
  double                       pin_frac      = 0.0;

  // Where a factory reports WHY it could not open the checkpoint.
  //
  // Not redundant with `session`: an offline tool or a plugin test has
  // no session, and "the factory returned null" on its own names the
  // architecture but not the problem -- which is the difference between
  // a missing tensor and an unparseable config.
  std::string*                 err = nullptr;
};

using HiddenStateEncoderFactory =
    std::function<std::unique_ptr<HiddenStateEncoder>(
        const HiddenStateEncoderArgs&)>;

// Process-wide `architecture` -> factory map, the same shape and the
// same singleton discipline as ModelExecRegistry and
// VideoModelRegistry: a plugin links the host libvpipe shared and
// registers into THIS instance.
//
// Keyed on the checkpoint's `architecture` string so a caller that knows
// only a path can still open the right encoder -- which is what a
// conditioner serving several families needs.
class HiddenStateEncoderRegistry {
public:
  static HiddenStateEncoderRegistry& get() noexcept;

  // First-wins, like every other registry here.
  bool register_arch(std::string arch, HiddenStateEncoderFactory f);
  bool contains(std::string_view arch) const noexcept;
  std::vector<std::string> architectures() const;

  // Null when `arch` is unregistered, or the factory returned null /
  // threw (logged through args.session).
  std::unique_ptr<HiddenStateEncoder>
  create(std::string_view arch, const HiddenStateEncoderArgs& args) const;

  // Open whatever `dir` holds, by reading its architecture first. Null
  // when the checkpoint cannot be identified or its arch has no
  // factory -- and it NAMES what it found, because "no encoder" and
  // "an encoder this build cannot open" are different problems.
  std::unique_ptr<HiddenStateEncoder>
  open(const HiddenStateEncoderArgs& args, std::string* err) const;

private:
  HiddenStateEncoderRegistry() = default;

  mutable std::mutex _mu;
  std::vector<std::pair<std::string, HiddenStateEncoderFactory>> _map;
};

}

#endif
