#include "plugin/plugin-manager.h"
#include "plugin/plugin-abi.h"
#include "plugin/plugin-context.h"
#include "pipeline/stage-registry.h"

#include "common/library-handle.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <filesystem>
#include <exception>
#include <utility>

namespace fs = std::filesystem;

namespace vpipe {

namespace {

// Resolve `path` to a canonical absolute string for dedup. Falls back
// through absolute -> raw so an unusual path still dedups on SOMETHING
// stable rather than failing the load.
std::string
canonical_(std::string_view path)
{
  std::error_code ec;
  fs::path p = fs::weakly_canonical(fs::path(path), ec);
  if (ec || p.empty()) {
    p = fs::absolute(fs::path(path), ec);
  }
  if (ec || p.empty()) {
    return std::string(path);
  }
  return p.string();
}

}  // namespace

PluginManager&
PluginManager::get() noexcept
{
  static PluginManager instance;
  return instance;
}

bool
PluginManager::is_abi_compatible(std::uint32_t plugin_abi) noexcept
{
  // Strict equality for now. If/when backward compatibility becomes a
  // goal, relax this to a major-version comparison.
  return plugin_abi == VPIPE_PLUGIN_ABI_VERSION;
}

bool
PluginManager::load(const SessionContextIntf* session, std::string_view path)
{
  const std::string canon = canonical_(path);

  std::lock_guard<std::mutex> lk(_mu);
  if (_loaded_paths.count(canon) != 0) {
    return true;                       // already loaded this process
  }

  // The stage-type id the registry is ABOUT to hand out. Everything
  // registered from here on belongs to this plugin -- captured BEFORE
  // the dlopen because a TypedStage<T> registers its own factory from a
  // static initialiser as the dylib maps, well before
  // vpipe_plugin_register is called. See StageRegistry::attribute_since.
  const StageTypeId first_type = StageRegistry::get().next_id();

  // dlopen (Optional: warns + valid()==false on failure, never throws).
  auto handle = std::make_unique<LibraryHandle>(
      session, canon, LibraryHandle::LoadMode::Optional);
  if (!handle->valid()) {
    return false;                      // LibraryHandle already warned
  }

  auto warn = [&](const VpipeFormat& f) {
    if (session != nullptr) { session->warn(f); }
  };

  // Resolve the three required entry points.
  auto abi_fn = reinterpret_cast<vpipe_plugin_abi_version_fn>(
      handle->get_symbol("vpipe_plugin_abi_version"));
  auto info_fn = reinterpret_cast<vpipe_plugin_info_fn>(
      handle->get_symbol("vpipe_plugin_info"));
  auto reg_fn = reinterpret_cast<vpipe_plugin_register_fn>(
      handle->get_symbol("vpipe_plugin_register"));
  if (abi_fn == nullptr || info_fn == nullptr || reg_fn == nullptr) {
    warn(fmt("plugin '{}': not a vpipe plugin (missing entry points "
             "vpipe_plugin_abi_version / _info / _register)", canon));
    // Keep the handle out of the loaded set so a later, corrected build
    // at the same path can be retried within another process -- but do
    // NOT dlclose here (benign to leak; avoids unload races).
    return false;
  }

  // ABI-version handshake.
  const std::uint32_t plugin_abi = abi_fn();
  if (!is_abi_compatible(plugin_abi)) {
    warn(fmt("plugin '{}': ABI version {} is incompatible with host {} "
             "(rebuild the plugin against this vpipe)",
             canon, plugin_abi, VPIPE_PLUGIN_ABI_VERSION));
    return false;
  }

  // Metadata (logged; central to commercial add-on provenance).
  const VpipePluginInfo* info = info_fn();
  std::string pname = canon;
  if (info != nullptr && info->name != nullptr) {
    pname = info->name;
  }
  if (info != nullptr) {
    if (session != nullptr) {
      session->info(fmt(
          "plugin loaded: {} v{} -- {} [license: {}] ({})",
          info->name        ? info->name        : "?",
          info->version     ? info->version     : "?",
          info->description ? info->description : "",
          info->license     ? info->license     : "unspecified",
          info->vendor      ? info->vendor      : "unknown vendor"));
    }
  } else {
    warn(fmt("plugin '{}': vpipe_plugin_info() returned null", canon));
  }

  // Register the plugin's extensions. A throwing register must not take
  // down the host: demote to a warning. Registrations that succeeded
  // before a throw stay in the registries; the handle is kept alive.
  VpipePluginContext ctx(session, pname);
  // Stamp provenance on EVERY exit path, including the throwing ones --
  // the comment below is explicit that partial registrations stay in the
  // registries, so a stage that made it in must still say where it came
  // from. A guard rather than three call sites, because the one that
  // gets forgotten is always the error path.
  struct AttributeOnExit {
    StageTypeId first;
    const std::string& name;
    ~AttributeOnExit()
    {
      StageRegistry::get().attribute_since(first, name);
    }
  } attribute_guard{first_type, pname};
  try {
    reg_fn(&ctx);
  } catch (const std::exception& e) {
    warn(fmt("plugin '{}': vpipe_plugin_register threw: {}", pname,
             e.what()));
    // Keep the dylib mapped (partial registrations may reference it);
    // record the path so we don't re-run its (partial) registration.
    _loaded_paths.insert(canon);
    _handles.push_back(std::move(handle));
    return false;
  } catch (...) {
    warn(fmt("plugin '{}': vpipe_plugin_register threw a non-standard "
             "exception", pname));
    _loaded_paths.insert(canon);
    _handles.push_back(std::move(handle));
    return false;
  }

  _loaded_paths.insert(canon);
  _handles.push_back(std::move(handle));
  _names.push_back(pname);
  {
    Record r;
    r.path    = canon;
    r.name    = pname;
    if (info != nullptr) {
      r.version     = info->version     ? info->version     : "";
      r.vendor      = info->vendor      ? info->vendor      : "";
      r.license     = info->license     ? info->license     : "";
      r.description = info->description ? info->description : "";
    }
    r.enabled = true;
    _records.push_back(std::move(r));
  }
  return true;
}

std::vector<PluginManager::Record>
PluginManager::records() const
{
  std::lock_guard<std::mutex> lk(_mu);
  return _records;
}

bool
PluginManager::is_loaded_path(std::string_view path) const
{
  // Through the SAME canonicaliser load() dedups with -- a listing that
  // answered "not loaded" for a path load() would treat as already
  // loaded would offer a button that does nothing.
  const std::string canon = canonical_(path);
  std::lock_guard<std::mutex> lk(_mu);
  return _loaded_paths.count(canon) != 0;
}

void
PluginManager::set_enabled(std::string_view name, bool on)
{
  std::lock_guard<std::mutex> lk(_mu);
  for (Record& r : _records) {
    if (r.name == name) { r.enabled = on; }
  }
}

bool
PluginManager::enabled(std::string_view name) const
{
  std::lock_guard<std::mutex> lk(_mu);
  for (const Record& r : _records) {
    if (r.name == name) { return r.enabled; }
  }
  // Not a plugin we loaded -- the built-ins, whose stages carry an empty
  // origin. Never withheld.
  return true;
}

void
PluginManager::load_all(const SessionContextIntf*       session,
                        const std::vector<std::string>& paths)
{
  for (const std::string& p : paths) {
    if (!p.empty()) {
      load(session, p);
    }
  }
}

std::vector<std::string>
PluginManager::loaded() const
{
  std::lock_guard<std::mutex> lk(_mu);
  return _names;
}

}
