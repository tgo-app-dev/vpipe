#include "minitest.h"
#include "common/path-sandbox.h"

#include <filesystem>
#include <string>
#include <system_error>
#include <unistd.h>

using namespace std;
using namespace vpipe;
namespace fs = std::filesystem;

namespace {

string make_root() {
  auto base = fs::temp_directory_path() / "vpipe_path_sandbox_XXXXXX";
  string tmpl = base.string();
  if (!mkdtemp(tmpl.data())) { throw runtime_error("mkdtemp failed"); }
  return tmpl;
}

struct TempRoot {
  string path;
  TempRoot() : path(make_root()) {}
  ~TempRoot() { std::error_code ec; fs::remove_all(path, ec); }
};

// True when `p` is non-empty and does not escape `root`.
bool within(const fs::path& p, const fs::path& root) {
  if (p.empty()) { return false; }
  const string rel = p.lexically_relative(root).generic_string();
  return rel != ".." && rel.rfind("../", 0) != 0;
}

}  // namespace

TEST(path_sandbox, disabled_passes_through) {
  PathSandbox sb;   // default -> disabled
  EXPECT_FALSE(sb.enabled());
  EXPECT_TRUE(sb.confine("/etc/passwd", false).string() == "/etc/passwd");
  EXPECT_TRUE(sb.confine("rel/x", true).string() == "rel/x");
}

TEST(path_sandbox, relative_lands_under_root) {
  TempRoot t; PathSandbox sb{fs::path(t.path)};
  EXPECT_TRUE(sb.enabled());
  string err;
  const auto p = sb.confine("sub/dir/file.txt", false, &err);
  EXPECT_TRUE(err.empty());
  EXPECT_TRUE(within(p, sb.root()));
  EXPECT_TRUE(p.filename() == "file.txt");
}

TEST(path_sandbox, absolute_is_rerooted) {
  TempRoot t; PathSandbox sb{fs::path(t.path)};
  const auto p = sb.confine("/etc/passwd", false);
  EXPECT_TRUE(within(p, sb.root()));
  EXPECT_TRUE(p.generic_string().rfind(sb.root().generic_string(), 0) == 0);
}

TEST(path_sandbox, dotdot_escape_rejected) {
  TempRoot t; PathSandbox sb{fs::path(t.path)};
  string err;
  const auto p = sb.confine("../../etc/passwd", false, &err);
  EXPECT_TRUE(p.empty());
  EXPECT_FALSE(err.empty());
}

TEST(path_sandbox, symlink_escape_rejected) {
  TempRoot t; PathSandbox sb{fs::path(t.path)};
  std::error_code ec;
  fs::create_symlink("/etc", sb.root() / "escape", ec);
  if (ec) { return; }   // platform without symlink support -> skip
  string err;
  const auto p = sb.confine("escape/passwd", false, &err);
  EXPECT_TRUE(p.empty());
  EXPECT_FALSE(err.empty());
}

TEST(path_sandbox, for_write_creates_parent) {
  TempRoot t; PathSandbox sb{fs::path(t.path)};
  const auto p = sb.confine("a/b/c/out.txt", true);
  EXPECT_TRUE(within(p, sb.root()));
  EXPECT_TRUE(fs::exists(p.parent_path()));
}

TEST(path_sandbox, empty_path_rejected_when_enabled) {
  TempRoot t; PathSandbox sb{fs::path(t.path)};
  string err;
  const auto p = sb.confine("", false, &err);
  EXPECT_TRUE(p.empty());
  EXPECT_FALSE(err.empty());
}

// ---- whitelist (--white-list-path) pass-through -------------------

TEST(path_sandbox, whitelist_absolute_passes_through) {
  TempRoot root; TempRoot ext;
  PathSandbox sb{fs::path(root.path), {fs::path(ext.path)}};
  EXPECT_TRUE(sb.whitelist().size() == 1u);
  EXPECT_TRUE(sb.whitelist()[0].is_absolute());
  string err;
  const auto p =
      sb.confine((fs::path(ext.path) / "data.bin").string(), false, &err);
  EXPECT_TRUE(err.empty());
  EXPECT_TRUE(within(p, sb.whitelist()[0]));  // real path, inside the grant
  EXPECT_FALSE(within(p, sb.root()));         // NOT re-rooted
  EXPECT_TRUE(p.filename() == "data.bin");
}

TEST(path_sandbox, whitelist_exact_root_passes_through) {
  TempRoot root; TempRoot ext;
  PathSandbox sb{fs::path(root.path), {fs::path(ext.path)}};
  const auto p = sb.confine(ext.path, false);
  EXPECT_TRUE(within(p, sb.whitelist()[0]));
  EXPECT_FALSE(within(p, sb.root()));
}

TEST(path_sandbox, non_whitelisted_absolute_still_rerooted) {
  TempRoot root; TempRoot ext;
  PathSandbox sb{fs::path(root.path), {fs::path(ext.path)}};
  const auto p = sb.confine("/etc/passwd", false);
  EXPECT_TRUE(within(p, sb.root()));      // confined, not passed through
  EXPECT_FALSE(within(p, sb.whitelist()[0]));
}

TEST(path_sandbox, whitelist_relative_never_matches) {
  TempRoot root; TempRoot ext;
  PathSandbox sb{fs::path(root.path), {fs::path(ext.path)}};
  const auto p = sb.confine("sub/x", false);  // relative -> stays confined
  EXPECT_TRUE(within(p, sb.root()));
}

TEST(path_sandbox, whitelist_for_write_creates_parent) {
  TempRoot root; TempRoot ext;
  PathSandbox sb{fs::path(root.path), {fs::path(ext.path)}};
  const auto p =
      sb.confine((fs::path(ext.path) / "deep/out.txt").string(), true);
  EXPECT_TRUE(within(p, sb.whitelist()[0]));
  EXPECT_TRUE(fs::exists(p.parent_path()));
}

TEST(path_sandbox, whitelist_symlink_out_not_passed_through) {
  TempRoot root; TempRoot ext;
  std::error_code ec;
  fs::create_symlink("/etc", fs::path(ext.path) / "out", ec);
  if (ec) { return; }   // no symlink support -> skip
  PathSandbox sb{fs::path(root.path), {fs::path(ext.path)}};
  // ext/out -> /etc, so ext/out/passwd resolves to /etc/passwd, OUTSIDE
  // the grant: it must NOT pass through (falls back to the re-root).
  const auto p = sb.confine((fs::path(ext.path) / "out/passwd").string(),
                            false);
  EXPECT_FALSE(within(p, sb.whitelist()[0]));
}
