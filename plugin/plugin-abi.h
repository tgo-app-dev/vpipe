#ifndef VPIPE_PLUGIN_ABI_H
#define VPIPE_PLUGIN_ABI_H

// The vpipe plugin ABI: the stable, C-linkage contract every plugin
// .dylib must satisfy. A plugin is a C++ shared library that links the
// host libvpipe (so it observes the same registry singletons + RTTI) and
// exports the three `extern "C"` entry points below. All the rich
// registration happens in C++ via VpipePluginContext (plugin-context.h);
// this header is deliberately C-clean so the version/discovery handshake
// never depends on C++ ABI details.
//
// See docs/PLUGINS.md for the author-facing guide.

#include <stddef.h>
#include <stdint.h>

// Bumped on ANY incompatible change to the plugin contract. The host loads
// a plugin only when the plugin's reported version EQUALS this value
// (strict equality -- backward compatibility is not a goal yet).
// Independent of the dylib SOVERSION, which guards the underlying
// C++/singleton ABI.
//
// NO LEDGER OF PAST VERSIONS, deliberately. Strict equality gives the
// number no ordering meaning -- N does not mean "N-1 and more", it is an
// opaque cookie -- so a list of superseded versions describes nothing a
// plugin author can act on. It also rots: this comment previously
// documented 1 and 2 while the value read 4.
//
// WHAT COUNTS AS THE CONTRACT is the part that is not obvious, and it is
// wider than this file. A bump is required for any change to the three C
// entry points below, to the VpipePluginContext facade -- and to the
// INTERFACES A PLUGIN SUBCLASSES (VideoModelFamily, VaeModelFamily,
// ModelExec, Stage, ...). Adding a virtual to one of those changes no C
// symbol and no facade method, yet moves the vtable: a plugin built
// against the older header passes every check and then calls through the
// wrong slot. That has already happened once here.
#define VPIPE_PLUGIN_ABI_VERSION 2u

// Layout version of VpipePluginInfo, so the struct can grow additively
// without breaking the three-symbol contract.
#define VPIPE_PLUGIN_INFO_SCHEMA 1u

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
} VpipePluginInfo;

// The host resolves these three symbols by name (unmangled) from the
// plugin dylib. The typedefs describe the pointer types the loader casts
// dlsym results to; the plugin DEFINES the actual functions (usually via
// VPIPE_PLUGIN_DEFINE below).
//
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

// Convenience for the (C++) plugin author: emit the three exported
// symbols. `INFO_PTR` is a pointer to a static VpipePluginInfo; `REGFN` is
// a `void(vpipe::VpipePluginContext*)`. Place at file scope:
//
//   static const VpipePluginInfo kInfo = {
//       VPIPE_PLUGIN_INFO_SCHEMA, "acme-codecs", "1.0.0",
//       "Acme Inc.", "Commercial", "Acme audio codecs" };
//   static void acme_register(vpipe::VpipePluginContext* c) { ... }
//   VPIPE_PLUGIN_DEFINE(&kInfo, acme_register)
#ifdef __cplusplus
#define VPIPE_PLUGIN_DEFINE(INFO_PTR, REGFN)                             \
  extern "C" uint32_t vpipe_plugin_abi_version(void)                     \
  {                                                                      \
    return VPIPE_PLUGIN_ABI_VERSION;                                     \
  }                                                                      \
  extern "C" const ::VpipePluginInfo* vpipe_plugin_info(void)           \
  {                                                                      \
    return (INFO_PTR);                                                   \
  }                                                                      \
  extern "C" void vpipe_plugin_register(::vpipe::VpipePluginContext* c)  \
  {                                                                      \
    (REGFN)(c);                                                          \
  }
#endif

#endif
