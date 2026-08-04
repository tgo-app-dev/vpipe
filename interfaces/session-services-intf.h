// session-services-intf.h -- the heavyweight subsystems a session owns.
//
// These used to hang off SessionContextIntf directly, which meant the
// lowest interface layer in the tree named six subsystems that sit
// ABOVE it -- CoreML, the generative-model manager, metal-compute,
// FFmpeg, LMDB. Every translation unit that only wanted to log dragged
// those names in with it, and a session context could not be described
// without describing the whole stack.
//
// They live here now, reached through SessionContextIntf::services().
// Code that needs a subsystem includes this header and says so;
// session-context-intf.h no longer mentions any of them.
//
// The accessors are DEFAULTED to nullptr rather than pure, and
// services() never returns null (contexts with no subsystems get a
// shared inert instance). That keeps the call shape identical to what
// it replaced: metal_compute() already answered nullptr on a non-Apple
// build, so `services()->metal_compute()` returning nullptr is the same
// answer through one more hop, not a new failure mode.

#ifndef SESSION_SERVICES_INTF_H
#define SESSION_SERVICES_INTF_H

#include "interfaces/service-req.h"

#include <string_view>

namespace vpipe {

class CoreMLModelManager;
class FFmpegLibraries;
class LmdbEnv;

namespace genai { class GenerativeModelManager; }
namespace metal_compute { class MetalCompute; }

// The stable id of a service, as a trait rather than a member, so a
// service can be an existing concrete class (LmdbEnv, MetalCompute)
// without being rewritten to carry one.
//
// The trailing "/N" is the SERVICE's own compatibility version, not the
// plugin ABI's. Bump it when the service's shape changes in a way a
// consumer would notice; a consumer compiled against "/1" then simply
// stops resolving rather than binding to something it misunderstands.
template <class T> struct ServiceId;

class SessionServicesIntf {
public:
  virtual ~SessionServicesIntf() = default;

  // Type-erased lookup: the service registered under `id`, or null when
  // this session provides none.
  //
  // Why a string key rather than one virtual accessor per service.
  // Plugins are loaded only when their reported ABI version EQUALS the
  // host's (plugin/plugin-abi.h), so every accessor added to this class
  // would change its vtable and force every plugin to rebuild. Keyed
  // lookup lets the service SET grow without the interface's shape
  // changing, which is the whole point for plugin development.
  //
  // Plugins CONSUME services; they do not register them. A plugin that
  // needs to publish a service to its own group is a separate path,
  // deliberately not this one -- it raises lifetime and id-ownership
  // questions worth deciding on purpose rather than by accident.
  virtual void* find_service(std::string_view /*id*/) const
  {
    return nullptr;
  }

  // Typed front door. Checked at the call site; the erasure is confined
  // to the ABI boundary.
  template <class T>
  T* service() const
  {
    return static_cast<T*>(find_service(ServiceId<T>::value));
  }

  // Lazily-materialized FFmpeg dlopen wrapper, shared across every
  // SessionMember in this session. The first caller pays the cost of
  // probing sonames and resolving the curated symbol table; every
  // subsequent caller gets the same instance back. Construction is
  // serialized internally; safe to call concurrently.
  //
  // The implementation may throw on first call if a Required-mode
  // load fails (e.g. no compatible FFmpeg on the system); subsequent
  // calls after a successful first construction never throw and
  // return the same non-null pointer. Contexts where FFmpeg is
  // intentionally unavailable (e.g. log delegates) return nullptr.
  const FFmpegLibraries* ffmpeg_libraries() const
  {
    return service<FFmpegLibraries>();
  }

  // Lazily-materialized session-shared LMDB environment. The path
  // and map size come from the session config's top-level `db.path`
  // and `db.map_size_mb`. When `db.path` is missing or empty the
  // env opens at "." -- i.e. the process CWD at first-open time.
  // Returns nullptr only if opening the env failed (the failure is
  // reported through the session's log delegate). Concrete sub-
  // databases (LmdbDb) live inside this single env -- both the log
  // delegate and any application stage that needs persistent KV
  // storage share it. Safe to call concurrently; the first call
  // serializes construction internally.
  LmdbEnv* lmdb_env() const { return service<LmdbEnv>(); }

  // Session-shared CoreML model cache. Callers use
  // `coreml_model_manager()->load(path, compute_units)` to obtain a
  // shared_ptr to a loaded model; duplicate requests share one
  // load. Returns nullptr on non-Apple builds (and on Apple if
  // VPIPE_BUILD_APPLE_SILICON was disabled at build time, since the
  // manager type is then a forward declaration with no
  // implementation). Safe to call concurrently.
  CoreMLModelManager* coreml_model_manager() const
  {
    return service<CoreMLModelManager>();
  }

  // Session-shared LLM manager (text + multi-modal). Lazily
  // constructed on first call. Backed by the metal-compute LM
  // subsystem, so the manager is only available on apple-silicon
  // builds; on any other build configuration this returns nullptr.
  // Safe to call concurrently. See `generative-models/` for the
  // managed types.
  genai::GenerativeModelManager* generative_model_manager() const
  {
    return service<genai::GenerativeModelManager>();
  }

  // Session-shared Metal compute kernel framework (CUDA-shape
  // surface: load_library / make buffer / encode dispatch on a
  // stream). Lazily constructed on first call. Returns nullptr on
  // non-Apple builds. Even on Apple-Silicon the returned pointer
  // may be valid()==false (Metal unavailable); callers must check
  // before dispatching. Safe to call concurrently. See
  // apple-silicon/metal-compute/metal-compute.h.
  metal_compute::MetalCompute* metal_compute() const
  {
    return service<metal_compute::MetalCompute>();
  }
};

// Ids for the five services the core session provides. Named here, in
// one place, so a consumer (including a plugin) resolves them by type
// and never spells a string at a call site.
template <> struct ServiceId<FFmpegLibraries> {
  static constexpr std::string_view value = "vpipe.ffmpeg/1";
};
template <> struct ServiceId<LmdbEnv> {
  static constexpr std::string_view value = "vpipe.lmdb/1";
};
template <> struct ServiceId<CoreMLModelManager> {
  static constexpr std::string_view value = "vpipe.coreml/1";
};
template <> struct ServiceId<genai::GenerativeModelManager> {
  static constexpr std::string_view value = "vpipe.genai/1";
};
template <> struct ServiceId<metal_compute::MetalCompute> {
  static constexpr std::string_view value = "vpipe.metal-compute/1";
};

// State a dependency on a service by TYPE, so a stage never spells an
// id string at a call site and a renamed/version-bumped service is a
// compile error rather than a requirement that silently stops matching.
template <class T>
constexpr ServiceReq require_service()
{
  return ServiceReq{ServiceId<T>::value, true};
}

template <class T>
constexpr ServiceReq optional_service()
{
  return ServiceReq{ServiceId<T>::value, false};
}

// The answer for a context that owns no subsystems -- every accessor
// nullptr. Shared and stateless, so SessionContextIntf::services() can
// hand it back by default instead of returning null and making all
// ~110 call sites test for it.
SessionServicesIntf* null_session_services() noexcept;

}

#endif
