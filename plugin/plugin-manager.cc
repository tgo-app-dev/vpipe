#include "plugin/plugin-manager.h"
#include "plugin/plugin-abi.h"
#include "plugin/plugin-context.h"
#include "pipeline/stage-registry.h"

#include "common/library-handle.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <libkern/OSByteOrder.h>
#include <mach-o/fat.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#endif

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

// ---- the version, read from the FILE -------------------------------------
//
// So a plugin outside the support window is refused before any of its code
// runs: dlopen executes its static initialisers, which may call into host
// code with layouts from another ABI. Only the `vpipe_plugin_abi` constant
// is read -- a plain 4-byte value in a const section, no relocation.
enum class FileAbi { Found, Absent, Unreadable };

#ifdef __APPLE__
FileAbi
read_file_abi_(const std::string& path, std::uint32_t* out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { return FileAbi::Unreadable; }
  auto at = [&](std::uint64_t off, void* dst, std::size_t n) {
    f.seekg(static_cast<std::streamoff>(off));
    f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
  };
#if defined(__arm64__) || defined(__aarch64__)
  const cpu_type_t want = CPU_TYPE_ARM64;
#else
  const cpu_type_t want = CPU_TYPE_X86_64;
#endif
  std::uint32_t magic = 0;
  if (!at(0, &magic, 4)) { return FileAbi::Unreadable; }
  std::uint64_t base = 0;
  if (magic == FAT_CIGAM || magic == FAT_CIGAM_64) {
    // Universal binary: headers are big-endian; take the host's slice.
    fat_header fh{};
    if (!at(0, &fh, sizeof fh)) { return FileAbi::Unreadable; }
    const std::uint32_t n = OSSwapBigToHostInt32(fh.nfat_arch);
    bool found = false;
    for (std::uint32_t i = 0; i < n && !found; ++i) {
      if (magic == FAT_CIGAM_64) {
        fat_arch_64 a{};
        if (!at(sizeof fh + i * sizeof a, &a, sizeof a)) { break; }
        if (static_cast<cpu_type_t>(OSSwapBigToHostInt32(a.cputype)) ==
            want) {
          base  = OSSwapBigToHostInt64(a.offset);
          found = true;
        }
      } else {
        fat_arch a{};
        if (!at(sizeof fh + i * sizeof a, &a, sizeof a)) { break; }
        if (static_cast<cpu_type_t>(OSSwapBigToHostInt32(a.cputype)) ==
            want) {
          base  = OSSwapBigToHostInt32(a.offset);
          found = true;
        }
      }
    }
    if (!found || !at(base, &magic, 4)) { return FileAbi::Unreadable; }
  }
  if (magic != MH_MAGIC_64) { return FileAbi::Unreadable; }
  mach_header_64 mh{};
  if (!at(base, &mh, sizeof mh)) { return FileAbi::Unreadable; }

  std::vector<section_64> sects;          // n_sect is 1-based, in order
  symtab_command          st{};
  bool                    have_st = false;
  std::uint64_t off = base + sizeof mh;
  for (std::uint32_t i = 0; i < mh.ncmds; ++i) {
    load_command lc{};
    if (!at(off, &lc, sizeof lc) || lc.cmdsize < sizeof lc) {
      return FileAbi::Unreadable;
    }
    if (lc.cmd == LC_SEGMENT_64) {
      segment_command_64 sc{};
      if (!at(off, &sc, sizeof sc)) { return FileAbi::Unreadable; }
      for (std::uint32_t j = 0; j < sc.nsects; ++j) {
        section_64 s{};
        if (!at(off + sizeof sc + j * sizeof s, &s, sizeof s)) {
          return FileAbi::Unreadable;
        }
        sects.push_back(s);
      }
    } else if (lc.cmd == LC_SYMTAB) {
      if (!at(off, &st, sizeof st)) { return FileAbi::Unreadable; }
      have_st = true;
    }
    off += lc.cmdsize;
  }
  if (!have_st) { return FileAbi::Unreadable; }

  std::vector<char>     strtab(st.strsize);
  std::vector<nlist_64> syms(st.nsyms);
  if (!at(base + st.stroff, strtab.data(), strtab.size()) ||
      !at(base + st.symoff, syms.data(), syms.size() * sizeof(nlist_64))) {
    return FileAbi::Unreadable;
  }
  static constexpr char kName[] = "_vpipe_plugin_abi";
  for (const nlist_64& n : syms) {
    if ((n.n_type & N_STAB) != 0 || (n.n_type & N_TYPE) != N_SECT) {
      continue;
    }
    const std::uint32_t sx = n.n_un.n_strx;
    if (sx >= strtab.size() || strtab.size() - sx < sizeof kName ||
        std::strncmp(strtab.data() + sx, kName, sizeof kName) != 0) {
      continue;
    }
    if (n.n_sect == 0 || n.n_sect > sects.size()) {
      return FileAbi::Unreadable;
    }
    const section_64& s = sects[n.n_sect - 1];
    if (n.n_value < s.addr || n.n_value + 4 > s.addr + s.size) {
      return FileAbi::Unreadable;
    }
    const std::uint64_t foff = base + s.offset + (n.n_value - s.addr);
    return at(foff, out, 4) ? FileAbi::Found : FileAbi::Unreadable;
  }
  return FileAbi::Absent;
}
#else
FileAbi
read_file_abi_(const std::string&, std::uint32_t*)
{
  // No pre-read off Apple: the version is checked after dlopen instead.
  return FileAbi::Unreadable;
}
#endif

}  // namespace

PluginManager&
PluginManager::get() noexcept
{
  // LEAKED, never destroyed -- which is what "never dlclose'd" has to
  // mean at exit too. A function-local static is destroyed in reverse
  // order of construction, and its `_handles` dlclose every plugin. Any
  // registry constructed BEFORE the first load (VideoModelRegistry is,
  // by a built-in family's static registration) is destroyed AFTER that,
  // and then deletes a plugin's family through a vtable in an unmapped
  // image: SIGSEGV in __cxa_finalize, on every clean exit of a process
  // that loaded a video-family plugin.
  static PluginManager* const instance = new PluginManager();
  return *instance;
}

bool
PluginManager::is_abi_compatible(std::uint32_t plugin_abi) noexcept
{
  // The support window: this version and the one before it, never below
  // the version that introduced the window.
  return plugin_abi >= VPIPE_PLUGIN_ABI_OLDEST &&
         plugin_abi <= VPIPE_PLUGIN_ABI_VERSION;
}

bool
PluginManager::load(const SessionContextIntf* session, std::string_view path)
{
  const std::string canon = canonical_(path);

  std::lock_guard<std::mutex> lk(_mu);
  if (_loaded_paths.count(canon) != 0) {
    return true;                       // already loaded this process
  }

  auto warn = [&](const VpipeFormat& f) {
    if (session != nullptr) { session->warn(f); }
  };
  auto window = [] {
    return VPIPE_PLUGIN_ABI_OLDEST == VPIPE_PLUGIN_ABI_VERSION
        ? fmt("{}", VPIPE_PLUGIN_ABI_VERSION)()
        : fmt("{}..{}", VPIPE_PLUGIN_ABI_OLDEST, VPIPE_PLUGIN_ABI_VERSION)();
  };

  // The version, from the FILE, before any of the plugin's code runs.
  std::uint32_t file_abi = 0;
  const FileAbi fa = read_file_abi_(canon, &file_abi);
  if (fa == FileAbi::Absent) {
    warn(fmt("plugin '{}': carries no `vpipe_plugin_abi`, so it was built "
             "for plugin ABI 7 or older; this host loads ABI {}. Rebuild "
             "it against this vpipe", canon, window()));
    return false;
  }
  if (fa == FileAbi::Found && !is_abi_compatible(file_abi)) {
    warn(fmt("plugin '{}': built for plugin ABI {}; this host loads ABI {}. "
             "Rebuild it against this vpipe", canon, file_abi, window()));
    return false;
  }

  // The stage-type id the registry is ABOUT to hand out. Everything
  // registered from here on belongs to this plugin -- captured BEFORE
  // the dlopen because a TypedStage<T> registers its own factory from a
  // static initialiser as the dylib maps, well before
  // vpipe_plugin_register is called. See StageRegistry::attribute_since.
  const StageTypeId first_type = StageRegistry::get().next_id();

  // And everything registered while it loads is QUARANTINED until the
  // handshake below passes -- withdrawn on every path that refuses it.
  StageRegistry::get().begin_quarantine();
  struct Quarantine {
    bool admit = false;
    ~Quarantine() { StageRegistry::get().end_quarantine(admit); }
  } quarantine;

  // dlopen (Optional: warns + valid()==false on failure, never throws).
  auto handle = std::make_unique<LibraryHandle>(
      session, canon, LibraryHandle::LoadMode::Optional);
  if (!handle->valid()) {
    return false;                      // LibraryHandle already warned
  }

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

  // ABI-version handshake: the function must agree with the constant it
  // was read from, which is what off-Apple builds check instead.
  const std::uint32_t plugin_abi = abi_fn();
  if (!is_abi_compatible(plugin_abi) ||
      (fa == FileAbi::Found && plugin_abi != file_abi)) {
    warn(fmt("plugin '{}': built for plugin ABI {}; this host loads ABI {}. "
             "Rebuild it against this vpipe", canon, plugin_abi, window()));
    return false;
  }

  // Metadata (logged; central to commercial add-on provenance).
  const VpipePluginInfo* info = info_fn();
  std::string pname = canon;
  if (info != nullptr && info->name != nullptr) {
    pname = info->name;
  }
  // The features it cannot run without -- refused by name, before it
  // registers anything, rather than failing on a missing symbol later.
  if (info != nullptr && info->schema_version >= 2 &&
      info->required_features != nullptr) {
    std::string missing;
    for (const char* const* f = info->required_features; *f != nullptr;
         ++f) {
      if (!host_has_feature(*f)) {
        if (!missing.empty()) { missing += ", "; }
        missing += *f;
      }
    }
    if (!missing.empty()) {
      warn(fmt("plugin '{}': requires feature(s) this host does not "
               "provide: {}", pname, missing));
      return false;
    }
  }
  quarantine.admit = true;

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
  VpipePluginContext ctx(session, pname, plugin_abi);
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
    r.abi     = plugin_abi;
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
