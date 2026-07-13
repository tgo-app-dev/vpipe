#ifndef VPIPE_OS_SANDBOX_H
#define VPIPE_OS_SANDBOX_H

#include <filesystem>
#include <string>
#include <vector>

namespace vpipe {

// Request to confine the CURRENT process's filesystem WRITES at the OS
// level -- a kernel backstop layered UNDER the app-level PathSandbox
// (path-sandbox.h). Reads are left permissive: the app-level policy
// already confines stage reads, and frameworks / model loading / the GPU
// driver need broad read access, so a read-deny profile is fragile. This
// layer's job is to stop any write that slips past the app policy from
// landing outside the sandbox. Applied ONCE at startup; it cannot be
// relaxed afterward.
//
// INCOMPATIBILITY: a whole-process seatbelt CANNOT be nested. Once it is
// in force, a child that applies its own profile -- the `run_python` chat
// tool spawns `sandbox-exec` to isolate each call -- fails with
// "sandbox_apply: Operation not permitted". So this backstop is opt-in
// (web-ui --os-sandbox) and mutually exclusive with the per-call python
// sandbox; the default confinement is the app-level PathSandbox (stages)
// plus each python call's own per-call seatbelt.
struct OsSandboxSpec {
  // Directories the process may WRITE to (recursively), IN ADDITION to
  // the OS-required scratch / cache / device locations the implementation
  // always grants. Populate with: the app sandbox root, the model output
  // dir (<cwd>/models), the LMDB db.path, and the app temp root
  // (.vpipe-tmp, which holds the per-python-call scratch dirs).
  std::vector<std::filesystem::path> writable_roots;
};

enum class OsSandboxStatus {
  Applied,      // a kernel sandbox is now in force
  Unavailable,  // no OS sandbox on this platform (no-op; app policy stands)
  Failed,       // the OS refused to install it (see err); process continues
};

// Install an OS-level filesystem-write sandbox on the current process.
// NEVER aborts the process: on failure it returns Failed with a message so
// the caller can warn and continue (fail-open). The app-level PathSandbox
// stays enforced regardless of the result here.
//
// PORTABILITY: every OS-specific line lives in os-sandbox.cc behind this
// one entry point. On macOS it compiles a seatbelt (Sandbox) profile; a
// future Linux/CUDA port swaps the implementation for Landlock or seccomp
// with no change to callers.
OsSandboxStatus
apply_os_file_sandbox(const OsSandboxSpec& spec, std::string* err);

}  // namespace vpipe

#endif  // VPIPE_OS_SANDBOX_H
