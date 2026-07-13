#include "common/temp-root.h"

#include <cstdlib>
#include <system_error>

namespace vpipe {

namespace fs = std::filesystem;

namespace {

fs::path
resolve_temp_root_()
{
  std::error_code ec;
  const fs::path cwd  = fs::current_path(ec);
  const fs::path base = ec ? fs::path(".") : cwd;

  fs::path root;
  const char* env = std::getenv("VPIPE_TMPDIR");
  if (env && *env) {
    fs::path p(env);
    root = p.is_absolute() ? p : (base / p);
  } else {
    root = base / ".vpipe-tmp";
  }

  std::error_code mk;
  fs::create_directories(root, mk);
  if (mk) {
    // Could not create the CWD-local root -- fall back to the OS temp dir
    // so callers never get an unusable path.
    std::error_code te;
    const fs::path os = fs::temp_directory_path(te);
    return te ? fs::path(".") : os;
  }
  return root;
}

}  // namespace

const fs::path&
temp_root()
{
  // Resolved once at first use (capturing the start CWD), then cached so
  // the root is stable for the process lifetime. Thread-safe init.
  static const fs::path root = resolve_temp_root_();
  return root;
}

}  // namespace vpipe
