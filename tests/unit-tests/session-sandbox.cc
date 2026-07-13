#include "minitest.h"
#include "common/session.h"

#include <filesystem>
#include <string>

using namespace std;
using namespace vpipe;
namespace fs = std::filesystem;

namespace {

string sandbox_cfg(const string& root) {
  return "{\"file_sandbox\":{\"enabled\":true,\"root\":\"" + root + "\"}}";
}

}  // namespace

// No config -> sandbox disabled -> paths pass through unchanged (the CLI
// and --expose-native-file-system behavior).
TEST(session_sandbox, disabled_by_default) {
  Session sess;
  string err;
  const auto p = sess.confine_path("/etc/passwd", false, &err);
  EXPECT_TRUE(err.empty());
  EXPECT_TRUE(p.string() == "/etc/passwd");
}

// Enabled -> an absolute path is re-rooted under the sandbox, not left
// pointing at the host location.
TEST(session_sandbox, absolute_rerooted_when_enabled) {
  const string root = (fs::temp_directory_path() / "vpipe_sess_sbx1").string();
  Session sess(sandbox_cfg(root));
  string err;
  const auto p = sess.confine_path("/etc/passwd", false, &err);
  EXPECT_TRUE(err.empty());
  EXPECT_TRUE(p.string() != "/etc/passwd");
  EXPECT_TRUE(p.filename() == "passwd");
  std::error_code ec; fs::remove_all(root, ec);
}

// Enabled -> a traversal that escapes the root is rejected.
TEST(session_sandbox, escape_rejected_when_enabled) {
  const string root = (fs::temp_directory_path() / "vpipe_sess_sbx2").string();
  Session sess(sandbox_cfg(root));
  string err;
  const auto p = sess.confine_path("../../../etc/passwd", false, &err);
  EXPECT_TRUE(p.empty());
  EXPECT_FALSE(err.empty());
  std::error_code ec; fs::remove_all(root, ec);
}
