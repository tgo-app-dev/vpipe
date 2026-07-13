#ifndef VPIPE_PATH_SANDBOX_H
#define VPIPE_PATH_SANDBOX_H

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

// Portable, chroot-like confinement of a stage-supplied file path to a
// sandbox root. This is the OS-independent layer of the web-ui filesystem
// sandbox; an optional OS-level backstop (os-sandbox.h) enforces the same
// boundary in the kernel where available.
//
// When enabled, the root behaves as "/": a path is taken relative to the
// root (leading separators are stripped, so "/etc/passwd" resolves to
// "<root>/etc/passwd"), symlinks and ".." are resolved with
// weakly_canonical -- catching both traversal and symlink escapes -- and
// a result that would still leave the root is rejected. When disabled,
// confine() returns the path unchanged (native access, for
// --expose-native-file-system and non-web-ui hosts).
//
// WHITELIST (pass-through): an optional set of real host prefixes the
// operator explicitly grants (web-ui --white-list-path). An ABSOLUTE
// user path that resolves inside a whitelisted prefix is returned
// UNCHANGED (symlink-resolved, still verified to stay within the
// prefix), NOT re-rooted -- so a granted external directory is reachable
// by its real path. The whitelist is checked first, so it wins over the
// re-root for overlapping names. Relative paths never match (they stay
// confined).
//
// Model-manager file access is intentionally NOT routed through this
// policy; the model registry / loader / fetch / quantize paths stay
// exempt so models can live outside the sandbox.
class PathSandbox {
 public:
  PathSandbox() = default;                            // disabled
  explicit PathSandbox(std::filesystem::path root,
                       std::vector<std::filesystem::path> whitelist = {});

  bool enabled() const noexcept { return _enabled; }
  const std::filesystem::path& root() const noexcept { return _root; }
  // Canonicalized real prefixes granted pass-through access (may be
  // empty). Surfaced so a browser can offer them as reachable "mounts".
  const std::vector<std::filesystem::path>&
  whitelist() const noexcept { return _whitelist; }

  // Confine `user_path` to the sandbox. Disabled -> returns it verbatim.
  // Enabled -> a whitelisted absolute path passes through unchanged;
  // otherwise the re-rooted, symlink-resolved absolute path, or *err set
  // and {} returned when the path escapes the root. When `for_write` is
  // set the parent directory is created.
  std::filesystem::path confine(std::string_view user_path, bool for_write,
                                std::string* err = nullptr) const;

 private:
  bool                               _enabled = false;
  std::filesystem::path              _root;
  std::vector<std::filesystem::path> _whitelist;
};

// True when `s` is a network URL (a known scheme://, e.g. rtsp://,
// http(s)://, rtmp://, srt://, udp://, ...). Such targets are not local
// filesystem paths and are left unconfined by the sandbox. Bare/relative
// paths and file:// return false (they ARE local and get confined).
bool is_network_url(std::string_view s);

}  // namespace vpipe

#endif  // VPIPE_PATH_SANDBOX_H
