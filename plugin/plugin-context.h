#ifndef VPIPE_PLUGIN_CONTEXT_H
#define VPIPE_PLUGIN_CONTEXT_H

#include "pipeline/stage-registry.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vpipe {

class SessionContextIntf;

// The C++ registration facade handed to a plugin's vpipe_plugin_register.
// Thin by design: it forwards to the process-wide registries (StageRegistry
// now; the metal runtime-library store and the ModelExecRegistry in later
// phases). Borrowed -- valid only for the duration of the register call.
//
// All registration is FIRST-WINS (mirrors StageRegistry::register_type). A
// no-op registration (the name/arch is already taken) is logged as a
// warning through the session so collisions are visible.
class VpipePluginContext {
public:
  VpipePluginContext(const SessionContextIntf* session,
                     std::string_view          plugin_name);

  // ---- introspection -------------------------------------------------
  // The host's plugin ABI version (== VPIPE_PLUGIN_ABI_VERSION) and a
  // human libvpipe version string, so a plugin can feature-detect.
  std::uint32_t    abi_version()  const noexcept;
  std::string_view host_version() const noexcept;

  // ---- stages --------------------------------------------------------
  // Register a stage type T (a TypedStage<T> subclass with a static
  // kTypeName + a public ctor taking (session, id, iports, config)).
  // Optionally register its file-static StageSpec (a pointer with static
  // storage duration) so the web-ui composer can describe it.
  //
  // NOTE: a TypedStage<T> ALSO self-registers its factory at load time
  // (its vtable references type()->_type_id, whose init calls
  // register_type). register_type is first-wins, so this call is
  // idempotent w.r.t. the factory; its material effect is attaching the
  // StageSpec and logging. Registration is safe to call redundantly, so
  // a plugin need not also use the VPIPE_REGISTER_STAGE macro.
  template <class T>
  void register_stage(const StageSpec* spec = nullptr)
  {
    register_stage(T::kTypeName, &make_stage_<T>, spec);
  }

  // Non-template form: an explicit factory (+ optional spec) for a stage
  // that isn't expressed as a TypedStage<T> at registration time.
  void register_stage(std::string_view       type_name,
                      StageRegistry::Factory factory,
                      const StageSpec*       spec = nullptr);

  // ---- metal shaders -------------------------------------------------
  // Register a plugin's offline-compiled `.metallib` bytes under `name`,
  // so the plugin's stages/models resolve its kernels via
  // session()->metal_compute()->load_library(name).function(...). The
  // bytes are copied; the caller's buffer need not outlive the call.
  // First-wins: returns false (+ warns) if `name` is already registered
  // or metal-compute is unavailable (including on non-apple builds).
  bool register_metal_library(std::string_view name,
                              const void* bytes, std::size_t n);
  bool register_metal_library_file(std::string_view name,
                                   std::string_view path);

private:
  // A non-capturing factory (convertible to StageRegistry::Factory) that
  // constructs T from the canonical stage ctor arguments.
  template <class T>
  static StagePtr make_stage_(const SessionContextIntf* s,
                              std::string               id,
                              std::vector<InEdge>       iports,
                              FlexData                  config)
  {
    return std::make_unique<T>(s, std::move(id), std::move(iports),
                               std::move(config));
  }

  const SessionContextIntf* _session;
  std::string               _plugin;
};

}

#endif
