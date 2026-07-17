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

// Commands run under bash, not POSIX /bin/sh, so `echo -e` behaves like the
// user's terminal: it interprets the escapes and does NOT emit a literal
// "-e " (the macOS /bin/sh = bash-in-POSIX-mode echo quirk). Guards against a
// regression back to /bin/sh.
TEST(command_sandbox, echo_dash_e_interprets_like_a_terminal) {
  Captured cap;
  CommandSandboxSpec spec;                 // native: behavior is the shell's
  CommandSandboxResult r = run_shell_command(
      "echo -e '10\\n20\\n30'", spec, capture_io_(cap));
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.exit_code == 0);
  EXPECT_TRUE(has_line_(cap.out, "10"));   // NOT "-e 10"
  EXPECT_TRUE(has_line_(cap.out, "20"));
  EXPECT_TRUE(has_line_(cap.out, "30"));
  for (const auto& l : cap.out) {
    EXPECT_TRUE(l.find("-e") == string::npos);
  }
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

// $TMPDIR points at a subdir UNDER the workspace, so a temp file written
// there stays inside the launch CWD; and the system Darwin temp is NOT
// writable. (Nothing is written outside the CWD -- tools that ignore $TMPDIR
// and hardcode the Darwin temp, e.g. the macOS `mktemp` CLI, fail rather
// than escape.)
TEST(command_sandbox, temp_files_stay_under_workspace) {
  if (!sandbox_available_()) { return; }
  const string work = make_tmpdir_("tmp");
  ASSERT_TRUE(!work.empty());

  Captured cap;
  CommandSandboxSpec spec;
  spec.enabled = true;
  spec.writable_roots.push_back(work);
  spec.cwd = work;
  const string cmd =
      "echo ok > \"$TMPDIR/probe.txt\" && echo WROTE_TMP; "
      "D=$(getconf DARWIN_USER_TEMP_DIR); echo x > \"${D}vpipe_esc\" "
      "2>/dev/null && echo SYS_WROTE || echo SYS_DENIED";
  CommandSandboxResult r = run_shell_command(cmd, spec, capture_io_(cap));
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(has_line_(cap.out, "WROTE_TMP"));
  EXPECT_TRUE(fs::exists(work + "/.vpipe_tmp/probe.txt")); // temp under the CWD
  EXPECT_TRUE(has_line_(cap.out, "SYS_DENIED"));           // system temp denied
  EXPECT_FALSE(has_line_(cap.out, "SYS_WROTE"));

  error_code ec;
  fs::remove_all(work, ec);
}

// `cd` must work when the sandbox workspace lives UNDER the real $HOME (the
// persistent web-ui case, $CWD/sandbox). The secret-protecting read-deny is
// scoped to file CONTENTS, so navigation works while a $HOME secret's
// contents stay unreadable.
TEST(command_sandbox, cd_works_for_workspace_under_home) {
  if (!sandbox_available_()) { return; }
  const char* home = ::getenv("HOME");
  if (!home || !*home) { return; }

  const fs::path base = fs::path(home) / ".vpipe_cmdsbx_home_test";
  const fs::path ws   = base / "ws";
  error_code ec;
  fs::create_directories(ws / "sub", ec);
  ASSERT_TRUE(!ec);
  const fs::path secret = base / "secret.txt";   // under $HOME, NOT under ws
  { std::ofstream f(secret.string()); f << "TOPSECRET\n"; }

  Captured cap;
  CommandSandboxSpec spec;
  spec.enabled = true;
  spec.writable_roots.push_back(ws.string());
  spec.cwd = ws.string();
  // Navigate (relative + absolute, all under $HOME); write AND read back a
  // file in the workspace (the workspace read-allow must still win over the
  // $HOME deny); then prove the $HOME secret's CONTENTS stay unreadable.
  const string cmd =
      "cd . && cd sub && cd '" + ws.string() + "' && echo CD-OK; "
      "echo payload > wsfile.txt && cat wsfile.txt; "
      "cat '" + secret.string() + "' 2>/dev/null && echo LEAK || echo NOLEAK";
  CommandSandboxResult r = run_shell_command(cmd, spec, capture_io_(cap));
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(has_line_(cap.out, "CD-OK"));    // navigation works under $HOME
  EXPECT_TRUE(has_line_(cap.out, "payload"));  // workspace read-back works
  EXPECT_TRUE(has_line_(cap.out, "NOLEAK"));   // secret contents stay blocked
  EXPECT_FALSE(has_line_(cap.out, "LEAK"));

  fs::remove_all(base, ec);
}

// With allow_system_temp opted in, $TMPDIR points at the per-user system
// temp and a write to the Darwin temp (which the macOS `mktemp` CLI uses,
// ignoring $TMPDIR) is permitted.
TEST(command_sandbox, system_temp_allowed_when_opted_in) {
  if (!sandbox_available_()) { return; }
  const string work = make_tmpdir_("systemp");
  ASSERT_TRUE(!work.empty());

  Captured cap;
  CommandSandboxSpec spec;
  spec.enabled = true;
  spec.allow_system_temp = true;              // opt in
  spec.writable_roots.push_back(work);
  spec.cwd = work;
  const string cmd =
      "D=$(getconf DARWIN_USER_TEMP_DIR); echo ok > \"${D}vpipe_sys.$$\" && "
      "cat \"${D}vpipe_sys.$$\" && echo SYS_OK && rm -f \"${D}vpipe_sys.$$\"; "
      "case \"$TMPDIR\" in /var/folders/*|/private/var/folders/*) "
      "echo TMPDIR_SYSTEM;; esac";
  CommandSandboxResult r = run_shell_command(cmd, spec, capture_io_(cap));
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(has_line_(cap.out, "SYS_OK"));        // system temp writable
  EXPECT_TRUE(has_line_(cap.out, "TMPDIR_SYSTEM")); // $TMPDIR -> system temp

  error_code ec;
  fs::remove_all(work, ec);
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
