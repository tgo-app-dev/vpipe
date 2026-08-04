// fs-sandbox-intf.h -- the session's filesystem namespace policy.
//
// Defaulted to native pass-through, so a context with no sandbox
// inherits the role without implementing it. The web-ui file browser is
// the main consumer that needs more than confine_path: it presents a
// chroot-like virtual namespace and therefore has to ask whether the
// sandbox is on, where its root really is, and what is whitelisted.

#ifndef FS_SANDBOX_INTF_H
#define FS_SANDBOX_INTF_H

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace vpipe {

class FsSandboxIntf {
public:
  virtual ~FsSandboxIntf() = default;

  // Confine a stage-supplied LOCAL file path to the session's filesystem
  // sandbox. When the sandbox is disabled (the CLI, or the web-ui run
  // with --expose-native-file-system) the path is returned unchanged.
  // When enabled the path is re-rooted under the sandbox (chroot-like:
  // the root acts as "/"), symlink/".." escapes set *err and return an
  // empty path, and `for_write` creates the parent directory. Network
  // URLs are NOT paths -- callers should skip this for rtsp/http(s).
  // Model-manager file access intentionally does NOT go through here.
  virtual std::filesystem::path
  confine_path(std::string_view user_path, bool /*for_write*/,
               std::string* /*err*/ = nullptr) const
  {
    return std::filesystem::path(user_path);
  }

  // True when the filesystem sandbox is active (web-ui default). Lets a
  // client -- e.g. the web-ui file browser -- present a chroot-like
  // virtual namespace (root == "/") instead of native host paths.
  virtual bool fs_sandboxed() const { return false; }

  // The real directory the sandbox root maps to, or {} when not
  // sandboxed. Server-side only (never surfaced to a sandboxed client),
  // it lets a browser list the root itself -- confine_path("/") rejects
  // the empty relative path, so the root needs this direct handle.
  virtual std::filesystem::path fs_sandbox_root() const { return {}; }

  // Real host prefixes the sandbox grants pass-through access to
  // (web-ui --white-list-path), or empty. A browser can offer these as
  // reachable "mounts"; confine_path() accepts absolute paths inside
  // them unchanged.
  virtual std::vector<std::filesystem::path> fs_whitelist() const
  {
    return {};
  }
};

}

#endif
