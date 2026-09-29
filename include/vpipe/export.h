#ifndef VPIPE_EXPORT_H
#define VPIPE_EXPORT_H

// What libvpipe EXPORTS, and therefore what a plugin can link against.
//
// libvpipe is built HIDDEN BY DEFAULT (-fvisibility=hidden). A header
// whose declarations belong to the stable surface -- the installed SDK
// headers, which ARE the plugin contract (plugin/plugin-abi.h) -- wraps
// them in VPIPE_API_BEGIN / VPIPE_API_END, which give every declaration
// in between default visibility: its functions are exported, and its
// classes' vtables and typeinfo are too, so a plugin can derive from them
// and dynamic_cast across the boundary. Everything else stays inside the
// library, where it can change without any plugin noticing.
//
// VPIPE_HOST_API_BEGIN / _END do the same for the handful of INTERNAL
// headers the in-tree apps (the CLI, the web UI) link against. Those
// headers are not installed and are not part of the plugin contract; the
// separate spelling is so a reader can tell the two apart.
//
// A single declaration outside such a region can be exported with
// VPIPE_API.

#define VPIPE_API __attribute__((visibility("default")))

// Inside a class in a stable header: its LAYOUT is not part of the plugin
// ABI. For a host-owned object a plugin only ever reaches through a
// pointer or reference -- it never allocates one, holds one by value,
// derives from it or reads a member -- so the host may change its members
// freely; its virtuals and exported methods are still contract. The ABI
// snapshot (cmake/abi-snapshot.py) leaves such a layout out.
#define VPIPE_ABI_OPAQUE using vpipe_abi_opaque = void

#define VPIPE_API_BEGIN      _Pragma("GCC visibility push(default)")
#define VPIPE_API_END        _Pragma("GCC visibility pop")
#define VPIPE_HOST_API_BEGIN _Pragma("GCC visibility push(default)")
#define VPIPE_HOST_API_END   _Pragma("GCC visibility pop")

#endif
