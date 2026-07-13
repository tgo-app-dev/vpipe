#include "common/os-sandbox.h"

#include <cstdlib>
#include <system_error>

// ---- OS-specific implementation (the ONLY platform-dependent code) ----
//
// macOS: compile + apply a seatbelt (Sandbox) profile via the libsandbox
// entry point that `sandbox-exec -p` uses. A Linux/CUDA port replaces this
// #if branch with Landlock (preferred) or seccomp; the vpipe::
// apply_os_file_sandbox contract below is unchanged.
#if defined(__APPLE__)
extern "C" {
int  sandbox_init_with_parameters(const char* profile, uint64_t flags,
                                  const char* const parameters[],
                                  char** errorbuf);
void sandbox_free_error(char* errorbuf);
}
#endif

namespace vpipe {

namespace fs = std::filesystem;

namespace {

#if defined(__APPLE__)

// Escape a path for embedding in an SBPL double-quoted string literal.
std::string sbpl_quote_(const std::string& s)
{
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    if (c == '"' || c == '\\') { out.push_back('\\'); }
    out.push_back(c);
  }
  return out;
}

void allow_write_subpath_(std::string& p, const fs::path& dir)
{
  std::error_code ec;
  fs::path c = fs::weakly_canonical(dir, ec);
  const std::string s =
      (ec || c.empty()) ? dir.lexically_normal().string() : c.string();
  if (s.empty()) { return; }
  p += "(allow file-write* (subpath \"" + sbpl_quote_(s) + "\"))\n";
}

// Build the SBPL profile. Strategy: allow everything, deny writes across
// the whole tree, then re-allow writes to (a) the OS-required scratch /
// cache / device locations a metal + CoreML + ffmpeg process needs and
// (b) the caller's roots. Seatbelt is "last matching rule wins", so the
// specific allows override the broad deny; reads stay allowed throughout.
// Write-focused + permissive-read = defense-in-depth that keeps the GPU /
// framework paths working.
//
// NOTE: this profile needs on-device validation (launch, load a model,
// run a pipeline, watch Console.app for "Sandbox: ... deny file-write"
// and extend the always-allowed set). It is applied FAIL-OPEN.
std::string build_profile_(const OsSandboxSpec& spec)
{
  std::string p =
      "(version 1)\n"
      "(allow default)\n"
      "(deny file-write* (subpath \"/\"))\n"
      "(allow file-write* (subpath \"/private/var/folders\"))\n"
      "(allow file-write* (subpath \"/private/tmp\"))\n"
      "(allow file-write* (subpath \"/private/var/tmp\"))\n"
      "(allow file-write* (subpath \"/dev\"))\n";
  const char* home = std::getenv("HOME");
  if (home && *home) {
    const std::string h = sbpl_quote_(home);
    p += "(allow file-write* (subpath \"" + h + "/Library/Caches\"))\n";
    p += "(allow file-write* (subpath \"" + h
       + "/Library/Saved Application State\"))\n";
  }
  for (const auto& r : spec.writable_roots) { allow_write_subpath_(p, r); }
  return p;
}

#endif  // __APPLE__

}  // namespace

OsSandboxStatus
apply_os_file_sandbox(const OsSandboxSpec& spec, std::string* err)
{
#if defined(__APPLE__)
  const std::string profile = build_profile_(spec);
  char* eb = nullptr;
  const int rc =
      sandbox_init_with_parameters(profile.c_str(), 0, nullptr, &eb);
  if (rc != 0) {
    if (err) { *err = eb ? std::string(eb) : "sandbox_init failed"; }
    if (eb) { sandbox_free_error(eb); }
    return OsSandboxStatus::Failed;
  }
  return OsSandboxStatus::Applied;
#else
  (void)spec;
  if (err) {
    *err = "OS filesystem sandbox not implemented on this platform";
  }
  return OsSandboxStatus::Unavailable;
#endif
}

}  // namespace vpipe
