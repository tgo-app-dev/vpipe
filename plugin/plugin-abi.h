#ifndef VPIPE_PLUGIN_ABI_H
#define VPIPE_PLUGIN_ABI_H

// The vpipe plugin ABI: the stable, C-linkage contract every plugin
// .dylib must satisfy. A plugin is a C++ shared library that links the
// host libvpipe (so it observes the same registry singletons + RTTI) and
// exports the `extern "C"` entry points below. All the rich registration
// happens in C++ via VpipePluginContext (plugin-context.h); this header
// is deliberately C-clean so the version/discovery handshake never
// depends on C++ ABI details.
//
// See docs/PLUGINS.md, "Versioning", for the author-facing rules.
//
// ---- the version ----------------------------------------------------------
//
// ONE INTEGER, plus FEATURE FLAGS for what is added in between.
//
//   * The integer (VPIPE_PLUGIN_ABI_VERSION) changes for any change that an
//     already-built plugin could observe: a field in a struct the plugin
//     allocates, reads or is handed; a virtual in an interface it
//     implements or calls; the signature of an exported function; the
//     meaning of a contracted kernel. What that covers is the STABLE
//     SURFACE -- the installed SDK headers, exported with VPIPE_API --
//     and the layout and symbol snapshots under abi/ are what enforce
//     it: a change to the surface fails the build until the snapshot
//     is regenerated, which is where the number is decided. It no
//     longer rests on someone remembering; it did not, repeatedly.
//
//   * A FEATURE is an addition that breaks nothing already built: a new
//     function, a new registry, a new interface a plugin may opt into, a
//     new command or kernel in a contract. It is a string ("name/rev"),
//     listed by the host (vpipe::host_features()) and queried by a plugin
//     (VpipePluginContext::has_feature). A plugin that cannot run without
//     one names it in VpipePluginInfo::required_features, and a host without it
//     refuses the plugin at load, by name, instead of letting it fail on
//     a missing symbol later.
//
//   * THE SUPPORT WINDOW is the last TWO versions: a host at N loads
//     plugins built for N and N-1 (VPIPE_PLUGIN_ABI_OLDEST). Keeping N-1
//     working is the host's job -- the old layouts and entry points stay,
//     or are translated -- so a version change is rare, batched, and
//     never casual. Eight is the first version under this scheme, so
//     there is no seven to keep: its OLDEST is eight.
//
// The version is read from the FILE before the plugin is loaded (the
// `vpipe_plugin_abi` data symbol, see VPIPE_PLUGIN_DEFINE), so a plugin
// outside the window is refused without running any of its code -- and
// with a message that names both versions, not a missing-symbol error.
// Anything a plugin registers while it loads (a TypedStage registers its
// factory from a static initialiser) is held back until the handshake
// passes, and withdrawn if it does not.

#include <stddef.h>
#include <stdint.h>

// The ABI this SDK builds plugins for, and the oldest one a host built
// from it still loads.
#define VPIPE_PLUGIN_ABI_VERSION 8u
#define VPIPE_PLUGIN_ABI_OLDEST  8u

// Layout version of VpipePluginInfo, so the struct grows additively.
//   1  name, version, vendor, license, description
//   2  + required_features
#define VPIPE_PLUGIN_INFO_SCHEMA 2u

// ---- features -------------------------------------------------------------
//
// The features ABI 8 ships with. A later host may add more; it never
// withdraws one while the ABI that introduced it is inside its window.
//   stage-commands   docs/STAGE-COMMANDS.md
//   kernel-contract  apple-silicon/metal-compute/kernel-contract.h
//   named-outputs    generative-models/gen-input.h's input/output seams
//   family-profiles  VpipePluginContext::register_family_profile
//   accel-bag        generative-models/shared/accel-settings.h
#define VPIPE_FEATURE_STAGE_COMMANDS  "stage-commands/1"
#define VPIPE_FEATURE_KERNEL_CONTRACT "kernel-contract/1"
#define VPIPE_FEATURE_NAMED_OUTPUTS   "named-outputs/1"
#define VPIPE_FEATURE_FAMILY_PROFILES "family-profiles/1"
#define VPIPE_FEATURE_ACCEL_BAG       "accel-bag/1"

#ifdef __cplusplus
namespace vpipe { class VpipePluginContext; }
extern "C" {
#endif

// Static metadata a plugin advertises. All strings are BORROWED and must
// outlive the process (string literals / static storage). `vendor` and
// `license` matter for commercial add-ons -- the loader logs them.
typedef struct VpipePluginInfo {
  uint32_t    schema_version;   // == VPIPE_PLUGIN_INFO_SCHEMA
  const char* name;             // stable plugin id, e.g. "acme-codecs"
  const char* version;          // the plugin's own version, e.g. "1.2.0"
  const char* vendor;           // e.g. "Acme Inc."
  const char* license;          // e.g. "Commercial", "Apache-2.0"
  const char* description;      // one-line summary
  // schema 2: the features this plugin cannot run without, as a
  // null-terminated array of VPIPE_FEATURE_* strings; null = none.
  const char* const* required_features;
} VpipePluginInfo;

// The host resolves these by name (unmangled) from the plugin dylib. The
// typedefs describe the pointer types the loader casts dlsym results to;
// the plugin DEFINES them (via VPIPE_PLUGIN_DEFINE below).
//
//   const uint32_t         vpipe_plugin_abi;          (data: read from the
//                                                      file before loading)
//   uint32_t               vpipe_plugin_abi_version(void);
//   const VpipePluginInfo* vpipe_plugin_info(void);
//   void                   vpipe_plugin_register(VpipePluginContext*);
typedef uint32_t (*vpipe_plugin_abi_version_fn)(void);
typedef const VpipePluginInfo* (*vpipe_plugin_info_fn)(void);
#ifdef __cplusplus
typedef void (*vpipe_plugin_register_fn)(vpipe::VpipePluginContext*);
#else
typedef void (*vpipe_plugin_register_fn)(void*);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

// Convenience for the (C++) plugin author: emit the exported symbols.
// `INFO_PTR` is a pointer to a static VpipePluginInfo; `REGFN` is a
// `void(vpipe::VpipePluginContext*)`. Place at file scope:
//
//   static const char* const kRequires[] = {
//       VPIPE_FEATURE_KERNEL_CONTRACT, nullptr };
//   static const VpipePluginInfo kInfo = {
//       VPIPE_PLUGIN_INFO_SCHEMA, "acme-codecs", "1.0.0",
//       "Acme Inc.", "Commercial", "Acme audio codecs", kRequires };
//   static void acme_register(vpipe::VpipePluginContext* c) { ... }
//   VPIPE_PLUGIN_DEFINE(&kInfo, acme_register)
#ifdef __cplusplus
#define VPIPE_PLUGIN_DEFINE(INFO_PTR, REGFN)                             \
  extern "C" __attribute__((visibility("default"), used))               \
  const uint32_t vpipe_plugin_abi = VPIPE_PLUGIN_ABI_VERSION;            \
  extern "C" __attribute__((visibility("default")))                     \
  uint32_t vpipe_plugin_abi_version(void)                                \
  {                                                                      \
    return VPIPE_PLUGIN_ABI_VERSION;                                     \
  }                                                                      \
  extern "C" __attribute__((visibility("default")))                     \
  const ::VpipePluginInfo* vpipe_plugin_info(void)                       \
  {                                                                      \
    return (INFO_PTR);                                                   \
  }                                                                      \
  extern "C" __attribute__((visibility("default")))                     \
  void vpipe_plugin_register(::vpipe::VpipePluginContext* c)             \
  {                                                                      \
    (REGFN)(c);                                                          \
  }
#endif

#endif
