#include "minitest.h"
#include "common/command-sandbox.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;
using namespace vpipe;

namespace {

namespace fs = std::filesystem;

// Collect lines a run emits, guarded so the drain thread and the (separate)
// stdin-pump thread don't race the vectors.
struct Captured {
  mutex          mu;
  vector<string> out;
  vector<string> err;
  void add_out(string_view s) { lock_guard<mutex> g(mu); out.emplace_back(s); }
  void add_err(string_view s) { lock_guard<mutex> g(mu); err.emplace_back(s); }
};

CommandSandboxIo
capture_io_(Captured& cap)
{
  CommandSandboxIo io;
  io.on_stdout_line = [&cap](string_view l) { cap.add_out(l); };
  io.on_stderr_line = [&cap](string_view l) { cap.add_err(l); };
  return io;
}

bool
has_line_(const vector<string>& v, const string& want)
{
  for (const auto& s : v) {
    if (s == want) { return true; }
  }
  return false;
}

// mkdtemp under /tmp; the seatbelt subpath rules canonicalize /tmp ->
// /private/tmp so a run confined here still matches at runtime.
string
make_tmpdir_(const char* tag)
{
  string tmpl = "/tmp/vpipe-cmdsbx-";
  tmpl += tag;
  tmpl += "-XXXXXX";
  vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  const char* p = ::mkdtemp(buf.data());
  return p ? string(p) : string();
}

bool
sandbox_available_()
{
#if defined(__APPLE__)
  return ::access("/usr/bin/sandbox-exec", X_OK) == 0;
#else
  return false;
#endif
}

}  // namespace

// ---- native (unsandboxed) I/O plumbing ----------------------------

TEST(command_sandbox, captures_stdout_and_stderr_lines) {
  Captured cap;
  CommandSandboxSpec spec;                 // enabled=false -> native
  CommandSandboxResult r = run_shell_command(
      "echo out1; echo out2; echo err1 1>&2", spec, capture_io_(cap));
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.exit_code == 0);
  EXPECT_TRUE(has_line_(cap.out, "out1"));
  EXPECT_TRUE(has_line_(cap.out, "out2"));
  EXPECT_TRUE(has_line_(cap.err, "err1"));
  EXPECT_FALSE(has_line_(cap.out, "err1"));
}

// A final line with no trailing newline is still delivered (flushed at EOF).
TEST(command_sandbox, flushes_unterminated_tail) {
  Captured cap;
  CommandSandboxSpec spec;
  CommandSandboxResult r = run_shell_command(
      "printf 'no-newline'", spec, capture_io_(cap));
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(has_line_(cap.out, "no-newline"));
}

TEST(command_sandbox, exit_code_propagates) {
  Captured cap;
  CommandSandboxSpec spec;
  CommandSandboxResult r = run_shell_command("exit 7", spec, capture_io_(cap));
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.exit_code == 7);
  EXPECT_FALSE(r.signaled);
}

TEST(command_sandbox, stdin_pumped_from_provider) {
  Captured cap;
  CommandSandboxSpec spec;
  CommandSandboxIo io = capture_io_(cap);
  // Two canned lines, then EOF -> `cat` echoes them and exits.
  auto remaining = make_shared<vector<string>>(
      vector<string>{"hello", "world"});
  auto idx = make_shared<size_t>(0);
  io.provide_stdin =
      [remaining, idx](string& line, const function<bool()>&) -> bool {
        if (*idx >= remaining->size()) { return false; }
        line = (*remaining)[(*idx)++];
        return true;
      };
  CommandSandboxResult r = run_shell_command("cat", spec, io);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.exit_code == 0);
  EXPECT_TRUE(has_line_(cap.out, "hello"));
  EXPECT_TRUE(has_line_(cap.out, "world"));
}

TEST(command_sandbox, timeout_kills_the_child) {
  Captured cap;
  CommandSandboxSpec spec;
  spec.timeout_ms = 300;
  CommandSandboxResult r = run_shell_command("sleep 30", spec,
                                             capture_io_(cap));
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.timed_out);
  EXPECT_TRUE(r.signaled);          // SIGKILLed, not a clean exit
}

TEST(command_sandbox, cancel_predicate_kills_the_child) {
  Captured cap;
  CommandSandboxSpec spec;
  CommandSandboxIo io = capture_io_(cap);
  io.should_cancel = []() { return true; };   // abort at once
  CommandSandboxResult r = run_shell_command("sleep 30", spec, io);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.canceled);
  EXPECT_TRUE(r.signaled);
}

// ---- seatbelt confinement (macOS) ---------------------------------

TEST(command_sandbox, sandbox_confines_writes_to_writable_roots) {
  if (!sandbox_available_()) { return; }     // vacuous skip off macOS
  const string inside  = make_tmpdir_("in");
  const string outside = make_tmpdir_("out");
  ASSERT_TRUE(!inside.empty() && !outside.empty());

  Captured cap;
  CommandSandboxSpec spec;
  spec.enabled = true;
  spec.writable_roots.push_back(inside);
  spec.cwd = inside;
  // A write into the workspace succeeds; a write outside is denied by the
  // kernel, so that file must never appear.
  string cmd = "echo yes > " + inside + "/inside.txt; "
               "echo no > " + outside + "/outside.txt";
  CommandSandboxResult r = run_shell_command(cmd, spec, capture_io_(cap));
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(fs::exists(inside + "/inside.txt"));
  EXPECT_FALSE(fs::exists(outside + "/outside.txt"));

  error_code ec;
  fs::remove_all(inside, ec);
  fs::remove_all(outside, ec);
}

TEST(command_sandbox, exec_whitelist_gates_programs) {
  if (!sandbox_available_()) { return; }
  const string work = make_tmpdir_("exec");
  ASSERT_TRUE(!work.empty());

  // Allow /bin/date only. Running it succeeds; running an unlisted program
  // (/bin/hostname) is refused by the kernel (non-zero exit, no stdout).
  {
    Captured cap;
    CommandSandboxSpec spec;
    spec.enabled = true;
    spec.writable_roots.push_back(work);
    spec.cwd = work;
    spec.exec_allow = {"/bin/date"};
    CommandSandboxResult r = run_shell_command("/bin/date +%Y", spec,
                                               capture_io_(cap));
    EXPECT_TRUE(r.ok);
    EXPECT_TRUE(r.exit_code == 0);
    EXPECT_TRUE(cap.out.size() >= 1);
  }
  {
    Captured cap;
    CommandSandboxSpec spec;
    spec.enabled = true;
    spec.writable_roots.push_back(work);
    spec.cwd = work;
    spec.exec_allow = {"/bin/date"};
    CommandSandboxResult r = run_shell_command("/bin/hostname", spec,
                                               capture_io_(cap));
    EXPECT_TRUE(r.ok);
    EXPECT_TRUE(r.exit_code != 0);       // exec of an unlisted binary refused
    EXPECT_TRUE(cap.out.empty());
  }

  error_code ec;
  fs::remove_all(work, ec);
}
