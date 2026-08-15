#include "common/plugins-root.h"

#include <cstdlib>
#include <system_error>

namespace vpipe {

namespace fs = std::filesystem;

namespace {

fs::path
resolve_plugins_root_()
{
  std::error_code ec;
  const fs::path cwd  = fs::current_path(ec);
  const fs::path base = ec ? fs::path(".") : cwd;

  const char* env = std::getenv("VPIPE_PLUGINS_DIR");
  if (env && *env) {
    fs::path p(env);
    return p.is_absolute() ? p : (base / p);
  }
  return base / "plugins";
}

}  // namespace

const fs::path&
plugins_root()
{
  // Resolved once at first use (capturing the start CWD), then cached, so
  // the answer cannot move under a caller that later chdir's. NOT
  // created -- see the header.
  static const fs::path root = resolve_plugins_root_();
  return root;
}

}  // namespace vpipe
