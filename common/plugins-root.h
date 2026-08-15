#ifndef VPIPE_PLUGINS_ROOT_H
#define VPIPE_PLUGINS_ROOT_H

#include <filesystem>

namespace vpipe {

// The conventional place to keep vpipe plugin dylibs: the work
// directory's `plugins/`. Resolved once (cached) as:
//   * $VPIPE_PLUGINS_DIR when set (a relative value is taken against the
//     start working directory), else
//   * "<start working directory>/plugins".
//
// "start working directory" is the process CWD captured the first time
// this or temp_root() is called, matching that helper exactly so the two
// roots cannot drift apart.
//
// TWO THINGS THIS DELIBERATELY DOES NOT DO, both for the same reason --
// a directory that is scanned must not become a directory that is
// trusted:
//
//   * it does NOT create the directory. temp_root() creates its own
//     because the process writes there; nothing writes here, and
//     materialising a `plugins/` folder in whatever directory someone
//     happened to run vpipe from is litter with no payoff. Callers
//     report the path and whether it exists.
//   * nothing here loads anything. Discovery is a listing, not an
//     instruction: dropping a .dylib into a scanned folder must not by
//     itself be enough to execute its code in this process. Loading is
//     an explicit act -- `--plugin PATH`, the `plugins:` config array, or
//     a deliberate action in the web-ui plugin panel.
const std::filesystem::path& plugins_root();

}  // namespace vpipe

#endif  // VPIPE_PLUGINS_ROOT_H
