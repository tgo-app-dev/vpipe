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

private:
  PluginManager() = default;

  mutable std::mutex                          _mu;
  std::unordered_set<std::string>             _loaded_paths;  // realpaths
  std::vector<std::unique_ptr<LibraryHandle>> _handles;       // kept alive
  std::vector<std::string>                    _names;
};

}

#endif
