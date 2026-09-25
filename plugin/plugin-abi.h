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
//
// It covers the STRUCTS those interfaces are handed, too -- and that is
// the half that was missed: `sol::Config` was added beside `sage` in
// VideoGenRequest without a bump, which is a layout change an old plugin
// would have passed every check and then misread. This value is the one
// thing standing between that and a wrong answer, so a field is never
// added to one of those structs without touching this line.
//
// ...AND WHAT DOES NOT COUNT. Adding a KEY to
// generative-models/shared/accel-settings.h is not a bump: the bag is
// one pointer whose layout does not move, its readers are header-only
// and compiled into the plugin, and a key an old plugin was never told
// about is one it never asks for. That is the whole reason those
// settings stopped being fields -- see the same header, and
// docs/PLUGINS.md, "Why a bag and not fields".
//
// THREE CARRIED TWO CHANGES, which was only legitimate because it
// never shipped. It was introduced for the acceleration bag and then
// reused for `register_image_family`, both on the same day, with nothing
// outside this tree ever having reported 3 -- so no binary exists that
// says 3 and means only the first of them. Reusing a number that HAS
// been released is the exact failure this line prevents: two different
// contracts answering to one cookie, and a plugin that passes the check
// and then misreads. The test is not "has anything been built against
// it", it is "has anything LEFT" -- check the number's introducing
// commit against the public remote before ever doing this again.
//
// FOUR CARRIES TWO CHANGES TOO, on the same terms. It was introduced for
// VideoModelFamily::denoise_scratch_bytes (a new virtual, 722924e) and
// then absorbed MetalCompute::MemoryBudget::self_graphics (a field added
// MID-struct, 5ecb0ad) -- a struct plugins get back BY VALUE from
// memory_budget(), so every field after it moved. Both landed before any
// plugin was rebuilt against 4, so no binary reports 4 and means only
// the first.
// FIVE: generative-models/shared/wired-pool.h's WiredPool became OPAQUE --
// one pointer, every method out of line -- after its layout had changed
// under a plugin that holds one as a member (SenseNova-U1.5). From here its
// layout and method signatures are FROZEN like ane::Tier's below; its
// open() options, info() fields, new methods and the policy itself grow
// without a bump.
// THE ANE TIER IS THE SAME SHAPE OF PROMISE, and was built that way so
// new work never costs a bump. generative-models/shared/ane-tier.h
// FREEZES ane::Binding, the signatures of ane::create/runtime_bytes and of
// ane::Tier's methods, and Tier::Plan's values -- a change to any of those
// is a bump. It does NOT freeze its vocabulary: a new kind (another
// activation, another projection, an attention block), a new spec or
// params key, a new binding name, a new info() field are all additions an
// old plugin never asks for. Tier itself is one pointer with out-of-line
// methods, so its layout is not compiled into anything.
// SIX: lora::Factors grew two fields (generative-models/shared/runtime-lora.h,
// f9e7451) and was NOT bumped for, which is what this line exists to prevent.
// `parts` and `group` were appended to an existing struct the host FILLS IN
// THROUGH A CALLER'S POINTER -- lora::Stack::bind(module, n, k, Factors* out)
// -- so a plugin compiled against the five-field version passes a smaller
// object and the host writes past its end. Not a misread: an overwrite of
// whatever the plugin had after it. The same commit did it again in
// shared/ane-ffn.h. The struct's own comment states the hazard exactly ("EVERY
// consumer of a Factors has to know this form -- a kernel that reads `b` as
// [n, rank] reads past a banded B") and the bump was still missed, which is
// the point: knowing a change is dangerous and remembering that danger is
// spelled as a number here are two different things.
#define VPIPE_PLUGIN_ABI_VERSION 6u

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
