#ifndef VPIPE_PLUGIN_CONTEXT_H
#define VPIPE_PLUGIN_CONTEXT_H

#include "pipeline/stage-registry.h"
#include "interfaces/session-services-intf.h"
#include "stages/model-catalog.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vpipe {

class SessionContextIntf;

namespace genai { class VideoModelFamily; }
namespace genai { class VaeModelFamily; }
namespace genai { class QuantizableFamily; }

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
  // session()->services()->metal_compute()->load_library(name).function(...). The
  // bytes are copied; the caller's buffer need not outlive the call.
  // First-wins: returns false (+ warns) if `name` is already registered
  // or metal-compute is unavailable (including on non-apple builds).
  bool register_metal_library(std::string_view name,
                              const void* bytes, std::size_t n);
  bool register_metal_library_file(std::string_view name,
                                   std::string_view path);

  // ---- video model families ------------------------------------------
  // Contribute a VIDEO model family, so `generate-video` can run a
  // checkpoint this build has never heard of. The family owns its whole
  // denoise loop -- scheduler, guidance, residency, patchify -- and the
  // stage keeps the ports, the beats and the geometry. See
  // generative-models/video-model-registry.h for what a family answers.
  //
  // Takes ownership; the registry outlives every stage. First-wins on
  // the family's `tag()`: a second family for a tag already present is
  // refused (returns false + warns) rather than silently shadowing the
  // first, because two plugins shipping one model is a deployment
  // mistake worth seeing.
  bool register_video_family(std::unique_ptr<genai::VideoModelFamily> family);

  // ---- VAE families ----------------------------------------------------
  // Contribute a VAE DECODER, so the stock `vae-decode` stage can turn a
  // latent from an out-of-tree checkpoint into RGB frames. The family
  // owns its un-whitening, its tiling, its colour space and its
  // residency; the stage keeps the ports, the U8 quantisation, the
  // per-frame beat and the idle-unload policy.
  //
  // This is the COUNTERPART to register_video_family: a plugin adding a
  // video model registers BOTH -- one family generates the latent, the
  // other decodes it -- and needs a stage of its own for neither. See
  // generative-models/vae-model-registry.h.
  //
  // Takes ownership; the registry outlives every stage. First-wins on
  // `tag()`, and a tag colliding with a built-in family name
  // ("wan", "minimax-h3", "flux2", "mage", "krea2") is refused outright:
  // dispatch is pointer-guarded so it would still run the right code,
  // but every log line would read as a built-in.
  bool register_vae_family(std::unique_ptr<genai::VaeModelFamily> family);

  // ---- quantizable families --------------------------------------------
  // Contribute a QUANTIZE recipe, so `model-quantize` can package this
  // plugin's checkpoint the way it packages a built-in one: copy every
  // component, quantize the one `target` names, register the result as a
  // whole model. Without it an out-of-tree repo falls through to the
  // single-component path, which quantizes one FILE at a time and writes
  // beside it -- workable, but it cannot produce one directory holding a
  // quantized DiT, a quantized encoder and untouched VAEs.
  //
  // What a family supplies is DATA -- where each component lives and
  // which of its tensors are matrices. The assembly, the quantization
  // and the registration stay with the stage, so a family cannot get
  // wrong the parts that are not its business. See
  // generative-models/quantize-family-registry.h.
  //
  // Takes ownership; the registry outlives every stage. First-wins on
  // `tag()`, and a tag colliding with a built-in family name is refused
  // for the same reason it is on the two registries above.
  bool
  register_quantize_family(std::unique_ptr<genai::QuantizableFamily> family);

  // ---- model catalogue -------------------------------------------------
  // Contribute downloadable-model entries, so a family this plugin adds
  // is fetchable and browsable the way a built-in one is (the drill-down
  // menu, `model-fetch`, the web-ui model browser). Returns how many
  // entries were taken; duplicates of what the catalogue already lists
  // are dropped. See stages/model-catalog.h for the lifetime rule.
  std::size_t
  register_catalog_entries(std::vector<ModelCatalogEntry> entries);

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
