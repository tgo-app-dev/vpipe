// file-sandbox.cc -- tests for the directory-confined file I/O surface
// and its read_file / write_file / list_files MCP tools. Pure std::
// filesystem, so these run in any build that includes the mcp library.

#include "minitest.h"

#include "common/flex-data.h"
#include "generative-models/shared/mcp/file-sandbox.h"
#include "generative-models/shared/mcp/mcp-tools.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <cstdlib>
#include <unistd.h>

using namespace std;
using namespace vpipe;

namespace fs = std::filesystem;

namespace {

// RAII temp dir (mkdtemp) that removes itself on scope exit.
struct TempDir {
  fs::path path;
  TempDir()
  {
    fs::path base = fs::temp_directory_path();
    string tmpl = (base / "vpipe-fsbx-test-XXXXXX").string();
    vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    path = ::mkdtemp(buf.data());
  }
  ~TempDir()
  {
    error_code ec;
    fs::remove_all(path, ec);
  }
};

}  // namespace

TEST(file_sandbox, write_then_read_round_trip)
{
  TempDir d;
  FileSandbox sb(d.path);
  string err;
  uint64_t written = 0;
  EXPECT_TRUE(sb.write_file("hello.txt", "hi there", false, &written, &err));
  EXPECT_TRUE(written == 8);
  string out;
  bool trunc = false;
  EXPECT_TRUE(sb.read_file("hello.txt", &out, &trunc, &err));
  EXPECT_TRUE(out == "hi there");
  EXPECT_FALSE(trunc);
}

TEST(file_sandbox, write_creates_nested_dirs)
{
  TempDir d;
  FileSandbox sb(d.path);
  string err;
  uint64_t w = 0;
  EXPECT_TRUE(sb.write_file("a/b/c.txt", "x", false, &w, &err));
  EXPECT_TRUE(fs::exists(d.path / "a" / "b" / "c.txt"));
}

TEST(file_sandbox, append_mode)
{
  TempDir d;
  FileSandbox sb(d.path);
  string err;
  uint64_t w = 0;
  EXPECT_TRUE(sb.write_file("log.txt", "one\n", false, &w, &err));
  EXPECT_TRUE(sb.write_file("log.txt", "two\n", true, &w, &err));
  string out;
  bool tr = false;
  EXPECT_TRUE(sb.read_file("log.txt", &out, &tr, &err));
  EXPECT_TRUE(out == "one\ntwo\n");
}

TEST(file_sandbox, traversal_is_rejected)
{
  TempDir d;
  FileSandbox sb(d.path);
  string err;
  string out;
  bool tr = false;
  // Reading and writing outside the root via `..` must fail.
  EXPECT_FALSE(sb.read_file("../escape.txt", &out, &tr, &err));
  uint64_t w = 0;
  EXPECT_FALSE(sb.write_file("../escape.txt", "x", false, &w, &err));
  EXPECT_FALSE(sb.write_file("a/../../escape.txt", "x", false, &w, &err));
  // The escape file must not have been created next to the root.
  EXPECT_FALSE(fs::exists(d.path.parent_path() / "escape.txt"));
}

TEST(file_sandbox, absolute_path_is_confined)
{
  TempDir d;
  FileSandbox sb(d.path);
  string err;
  uint64_t w = 0;
  // A leading "/" is stripped -> lands inside the sandbox, does NOT
  // touch the real /etc.
  EXPECT_TRUE(sb.write_file("/etc/passwd", "not real", false, &w, &err));
  EXPECT_TRUE(fs::exists(d.path / "etc" / "passwd"));
}

TEST(file_sandbox, symlink_escape_is_rejected)
{
  TempDir d;
  FileSandbox sb(d.path);
  // Plant a symlink inside the sandbox that points at the real /etc.
  error_code ec;
  fs::create_directory_symlink("/etc", d.path / "link", ec);
  if (ec) { return; }   // platform without symlink support
  string out, err;
  bool tr = false;
  // Reading through the symlink must be denied (it resolves outside).
  EXPECT_FALSE(sb.read_file("link/hosts", &out, &tr, &err));
}

TEST(file_sandbox, read_missing_and_directory_error)
{
  TempDir d;
  FileSandbox sb(d.path);
  string out, err;
  bool tr = false;
  EXPECT_FALSE(sb.read_file("nope.txt", &out, &tr, &err));
  uint64_t w = 0;
  EXPECT_TRUE(sb.write_file("dir/f.txt", "x", false, &w, &err));
  // Reading a directory path is an error.
  EXPECT_FALSE(sb.read_file("dir", &out, &tr, &err));
}

TEST(file_sandbox, binary_read_rejected)
{
  TempDir d;
  FileSandbox sb(d.path);
  // Write raw bytes with an embedded NUL directly (bypass write_file's
  // text intent) then confirm read_file refuses it.
  {
    ofstream f(d.path / "bin.dat", ios::binary);
    const char bytes[] = { 'a', '\0', 'b' };
    f.write(bytes, sizeof bytes);
  }
  string out, err;
  bool tr = false;
  EXPECT_FALSE(sb.read_file("bin.dat", &out, &tr, &err));
}

TEST(file_sandbox, read_truncates_at_cap)
{
  TempDir d;
  FileSandboxOptions o;
  o.max_read_bytes = 4;
  FileSandbox sb(d.path, o);
  string err;
  uint64_t w = 0;
  EXPECT_TRUE(sb.write_file("big.txt", "0123456789", false, &w, &err));
  string out;
  bool tr = false;
  EXPECT_TRUE(sb.read_file("big.txt", &out, &tr, &err));
  EXPECT_TRUE(out.size() == 4);
  EXPECT_TRUE(tr);
}

TEST(file_sandbox, write_over_cap_rejected)
{
  TempDir d;
  FileSandboxOptions o;
  o.max_write_bytes = 3;
  FileSandbox sb(d.path, o);
  string err;
  uint64_t w = 0;
  EXPECT_FALSE(sb.write_file("x.txt", "toolong", false, &w, &err));
}

TEST(file_sandbox, list_dir_reports_entries)
{
  TempDir d;
  FileSandbox sb(d.path);
  string err;
  uint64_t w = 0;
  sb.write_file("f1.txt", "aa", false, &w, &err);
  sb.write_file("sub/f2.txt", "b", false, &w, &err);
  vector<FileSandbox::Entry> ents;
  bool tr = false;
  EXPECT_TRUE(sb.list_dir(".", &ents, &tr, &err));
  bool saw_file = false, saw_dir = false;
  for (const auto& e : ents) {
    if (e.name == "f1.txt" && !e.is_dir && e.size == 2) { saw_file = true; }
    if (e.name == "sub" && e.is_dir) { saw_dir = true; }
  }
  EXPECT_TRUE(saw_file);
  EXPECT_TRUE(saw_dir);
}

TEST(file_sandbox, tools_round_trip_via_json)
{
  auto d  = std::make_shared<TempDir>();
  auto sb = std::make_shared<FileSandbox>(d->path);
  McpToolRegistry reg;
  add_file_tools(reg, sb);
  EXPECT_TRUE(reg.has("read_file"));
  EXPECT_TRUE(reg.has("write_file"));
  EXPECT_TRUE(reg.has("list_files"));

  McpToolCall wr{ "write_file",
      "{\"path\":\"note.txt\",\"content\":\"remember milk\"}" };
  FlexData wres = FlexData::from_json(reg.dispatch(wr));
  EXPECT_TRUE(wres.is_object() && wres.as_object().contains("ok"));

  McpToolCall rd{ "read_file", "{\"path\":\"note.txt\"}" };
  FlexData rres = FlexData::from_json(reg.dispatch(rd));
  EXPECT_TRUE(rres.is_object());
  EXPECT_TRUE(string(rres.as_object().at("content").as_string(""))
              == "remember milk");
}

TEST(file_sandbox, tool_traversal_returns_error)
{
  auto d  = std::make_shared<TempDir>();
  auto sb = std::make_shared<FileSandbox>(d->path);
  McpToolRegistry reg;
  add_file_tools(reg, sb);
  McpToolCall rd{ "read_file", "{\"path\":\"../../etc/passwd\"}" };
  FlexData res = FlexData::from_json(reg.dispatch(rd));
  EXPECT_TRUE(res.is_object() && res.as_object().contains("error"));
}
