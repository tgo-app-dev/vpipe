#ifndef VPIPE_GENERATIVE_MODELS_MODEL_EXEC_REGISTRY_H
#define VPIPE_GENERATIVE_MODELS_MODEL_EXEC_REGISTRY_H

#include "generative-models/model-exec.h"
#include "generative-models/model-loader.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace vpipe {
class SessionContextIntf;
namespace metal_compute { class MetalCompute; }
}

namespace vpipe::genai {

// Everything a plugin's ModelExec factory receives -- the inputs the
// built-in arch dispatch has in scope when it constructs a Metal*ModelExec
// (see LoadedLanguageModel). The factory builds its own model + returns a
// ModelExec that owns its KV (owns_kv()==true), so LoadedLanguageModel
// drives it through the token-id path.
struct ModelExecCreateArgs {
  std::string                  model_dir;    // resolved model directory
  const ModelConfig&           config;       // parsed config.json
  metal_compute::MetalCompute* metal;        // the GPU backend
  const SessionContextIntf*    session;      // for logging
  std::uint32_t                page_tokens;  // KV page sizing
  std::uint32_t                max_pages;
  bool                         use_bf16;     // compute dtype: bf16 vs f16
};

using ModelExecFactory =
    std::function<std::unique_ptr<ModelExec>(const ModelExecCreateArgs&)>;

// Process-wide `architecture` -> ModelExec factory map. LoadedLanguageModel
// consults it BEFORE its built-in arch if-chain, so a plugin can add a new
// LM family keyed by its config.json `architecture` string without editing
// the core. First-wins (like StageRegistry); a plugin registers into this
// singleton from its vpipe_plugin_register. The registry lives in libvpipe
// so a plugin that links libvpipe shares it.
class ModelExecRegistry {
public:
  static ModelExecRegistry& get() noexcept;

  // Register `factory` for `arch`. First registration wins: a later one
  // for the same arch is ignored and returns false. Thread-safe.
  bool register_arch(std::string arch, ModelExecFactory factory);

  bool contains(std::string_view arch) const noexcept;

  // Build the exec for `arch`. Returns nullptr if `arch` is not
  // registered, or if the factory returned null / threw (logged).
  std::unique_ptr<ModelExec> create(std::string_view           arch,
                                    const ModelExecCreateArgs& args) const;

private:
  ModelExecRegistry() = default;

  mutable std::mutex                                _mu;
  std::unordered_map<std::string, ModelExecFactory> _map;
};

}

#endif
