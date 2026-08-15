#ifndef VPIPE_PLUGIN_MANAGER_H
#define VPIPE_PLUGIN_MANAGER_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace vpipe {

class SessionContextIntf;
class LibraryHandle;

// Process-wide loader for vpipe plugin .dylibs. It dlopen's each plugin
// once (dedup by canonical path), enforces the ABI-version handshake,
// logs the plugin's metadata (including its license), and calls
// vpipe_plugin_register so the plugin can add stages / shaders / models.
//
// Loaded handles are held for the WHOLE process lifetime and never
// dlclose'd: a plugin's registrations, factories, and metallib bytes must
// outlive every Session that might use them. Both apps destroy their
// Session before main() returns, so plugin-provided objects are torn down
// on the owning thread before any image terminator runs.
class PluginManager {
public:
  static PluginManager& get() noexcept;

  // True when a plugin reporting `plugin_abi` is loadable by this host.
  // Strict equality for now (backward compatibility is not yet a goal).
  static bool is_abi_compatible(std::uint32_t plugin_abi) noexcept;

  // Load one plugin from `path`. `session` is used for logging and as the
  // dlopen owner. Returns true on success or if the same canonical path
  // was already loaded (idempotent). Any failure -- bad path, ABI
  // mismatch, missing symbol, or a throwing register -- is logged as a
  // warning and returns false; this never throws.
  bool load(const SessionContextIntf* session, std::string_view path);

  // Load a batch (the session's `plugins:` config array / VPIPE_PLUGINS).
  void load_all(const SessionContextIntf*        session,
                const std::vector<std::string>&  paths);

  // Names of successfully-registered plugins (diagnostics / web-ui).
  std::vector<std::string> loaded() const;

  // What a loaded plugin advertised, plus this process's view of it.
  struct Record {
    std::string   path;        // canonical path it was loaded from
    std::string   name;        // VpipePluginInfo::name (or the path)
    std::string   version;
    std::string   vendor;
    std::string   license;
    std::string   description;
    bool          enabled = true;
  };
  std::vector<Record> records() const;

  // Is this canonical path already loaded? The question the web-ui's
  // plugin listing asks of each file it discovered.
  bool is_loaded_path(std::string_view path) const;

  // ---- enable / disable ----------------------------------------------
  //
  // A DISABLED plugin is still fully loaded and mapped; its stages are
  // simply withheld from the composer and refused for new instances.
  //
  // This is not a weaker unload -- it is the only safe thing in the
  // neighbourhood, and the distinction is worth stating plainly wherever
  // it surfaces. Handles are never dlclose'd (see the class comment):
  // StageRegistry holds raw factory pointers into the dylib and has no
  // removal API, the StageSpec* it hands the web-ui points at the
  // dylib's static storage, live stage instances hold vtables there, and
  // the metal-library / video-family / VAE-family / catalogue
  // registrations all reference it. Unmapping any of that under a
  // running process is a use-after-unmap, not a reclaim.
  //
  // So "turn this plugin off" means: stop offering it. The bytes stay.
  // A real unload needs deregistration across those registries plus a
  // liveness check, which is an architectural change rather than a
  // method.
  void set_enabled(std::string_view name, bool on);
  bool enabled(std::string_view name) const;

private:
  PluginManager() = default;

  mutable std::mutex                          _mu;
  std::unordered_set<std::string>             _loaded_paths;  // realpaths
  std::vector<std::unique_ptr<LibraryHandle>> _handles;       // kept alive
  std::vector<std::string>                    _names;
  std::vector<Record>                         _records;
};

}

#endif
