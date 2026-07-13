#ifndef VPIPE_TEMP_ROOT_H
#define VPIPE_TEMP_ROOT_H

#include <filesystem>

namespace vpipe {

// The application's temporary-file root: a self-maintained directory the
// process owns, in place of the OS / app-sandbox temp dir. Resolved once
// (cached) as:
//   * $VPIPE_TMPDIR when set (a relative value is taken against the start
//     working directory), else
//   * "<start working directory>/.vpipe-tmp".
// The directory is created on first call. If it cannot be created the OS
// temp dir (std::filesystem::temp_directory_path) is returned as a last
// resort, so callers always get a usable path.
//
// "start working directory" is the process CWD captured the first time
// this is called (normally at startup, before any chdir), so the root
// stays stable even if the process later changes directory.
//
// Callers create their own uniquely-named files / subdirectories under
// this root (as they did under the OS temp dir) and remain responsible
// for removing them.
const std::filesystem::path& temp_root();

}  // namespace vpipe

#endif  // VPIPE_TEMP_ROOT_H
