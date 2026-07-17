#include "common/command-sandbox.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__APPLE__) || defined(__unix__)
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

extern char** environ;

using namespace std;

namespace vpipe {

namespace fs = std::filesystem;

namespace {

#if defined(__APPLE__)
constexpr const char* kSandboxExec = "/usr/bin/sandbox-exec";
#endif

// The shell we exec `-c <command>` with. Prefer /bin/bash so interactive-
// shell idioms behave as they do in the user's terminal -- notably
// `echo -e`: macOS /bin/sh is bash in POSIX mode, where `echo` does NOT take
// `-e` as a flag and prints a literal "-e ". Fall back to /bin/sh where bash
// is absent. Resolved once. (`make` and friends still re-exec /bin/sh for
// recipes, so the exec whitelist allows both.)
const char*
shell_path_()
{
#if defined(__APPLE__) || defined(__unix__)
  static const char* const kShell =
      (::access("/bin/bash", X_OK) == 0) ? "/bin/bash" : "/bin/sh";
#else
  static const char* const kShell = "/bin/sh";
#endif
  return kShell;
}

// Canonical form (symlinks + /var -> /private/var + ".." collapsed), with a
// lexical-normal fallback when the target does not exist yet. Matches how
// PathSandbox / the python sandbox resolve roots so seatbelt subpath rules
// actually fire at runtime.
string
canon_(const fs::path& p)
{
  std::error_code ec;
  fs::path c = fs::weakly_canonical(p, ec);
  return (ec || c.empty()) ? p.lexically_normal().string() : c.string();
}

// Escape a path for embedding in a seatbelt string literal.
string
esc_(const string& s)
{
  string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

// Resolve one exec_allow entry to zero or more canonical absolute paths.
// An absolute path is used verbatim (when it exists); a bare name is looked
// up in a standard set of bin dirs. Non-existent entries contribute nothing
// (they simply can't be launched).
void
resolve_exec_(const string& name, vector<string>& out)
{
#if defined(__APPLE__) || defined(__unix__)
  auto add = [&](const string& p) {
    if (::access(p.c_str(), X_OK) == 0) {
      out.push_back(canon_(p));
    }
  };
  if (!name.empty() && name.front() == '/') {
    add(name);
    return;
  }
  static const char* kBinDirs[] = {
    "/usr/bin", "/bin", "/usr/local/bin", "/opt/homebrew/bin",
    "/usr/sbin", "/sbin",
  };
  for (const char* d : kBinDirs) {
    add(string(d) + "/" + name);
  }
#else
  (void)name;
  (void)out;
#endif
}

#if defined(__APPLE__)
// Compile the seatbelt profile that contains the command's side effects:
// deny writes outside `writable` (+ /dev, + `temp_roots` when granted), deny
// the real $HOME reads, deny network (unless allowed), and -- when
// `exec_allow` is non-empty -- deny process-exec of anything but the shell
// and the whitelisted programs.
string
build_profile_(const CommandSandboxSpec& spec, const string& home,
               const vector<string>& temp_roots)
{
  string p;
  p += "(version 1)\n";
  p += "(allow default)\n";
  if (!spec.allow_network) {
    p += "(deny network*)\n";
  }
  p += "(deny file-write*)\n";
  p += "(allow file-write*\n";
  for (const auto& w : spec.writable_roots) {
    p += "  (subpath \"" + esc_(canon_(w)) + "\")\n";
  }
  // Besides the writable roots, only /dev -- and, when allow_system_temp was
  // set, the per-user system temp (so tools that hardcode it, ignoring
  // $TMPDIR, work). With it OFF `temp_roots` is empty and the child's $TMPDIR
  // points at a subdir UNDER the workspace (see the env setup), so every temp
  // file stays inside the launch CWD; nothing is written outside it (bar the
  // caller's explicit whitelist grants).
  for (const auto& t : temp_roots) {
    p += "  (subpath \"" + esc_(t) + "\")\n";
  }
  p += "  (subpath \"/dev\"))\n";
  if (!home.empty()) {
    // Deny reads under the real $HOME so model code can't slurp secrets
    // (~/.ssh, ~/.aws, keychains) into the reply -- but then re-allow
    // file-read-METADATA. A blunt file-read* deny also blocks the stat/lookup
    // that `cd`, `getcwd`, and path resolution need, so any command that
    // changes directory breaks the instant the sandbox workspace lives under
    // $HOME (the persistent web-ui case, $CWD/sandbox). Re-allowing metadata
    // (stat/traversal only -- NOT file contents or directory listing) makes
    // navigation work while secrets stay unreadable. Contents re-open only
    // under the writable roots below. (Order matters: a same-specificity
    // file-read* allow on a writable root under $HOME must come AFTER this
    // deny to win; a more-specific file-read-data deny here would instead
    // beat that allow and break reads in a $HOME workspace.)
    p += "(deny file-read* (subpath \"" + esc_(home) + "\"))\n";
    p += "(allow file-read-metadata (subpath \"" + esc_(home) + "\"))\n";
    // Re-open FULL reads under the writable roots (a workspace may live under
    // $HOME, and the child must read/list what it just wrote there).
    for (const auto& w : spec.writable_roots) {
      p += "(allow file-read* (subpath \"" + esc_(canon_(w)) + "\"))\n";
    }
  }
  if (!spec.exec_allow.empty()) {
    vector<string> allowed;
    // The shell machinery must exec: we run the command with /bin/bash
    // (shell_path_), but `make` and other tools re-exec /bin/sh for their
    // recipes, so both have to be allowed or a whitelisted program never
    // gets a shell to launch it.
    resolve_exec_("/bin/sh", allowed);
    resolve_exec_("/bin/bash", allowed);
    for (const auto& n : spec.exec_allow) {
      resolve_exec_(n, allowed);
    }
    p += "(deny process-exec*)\n";
    p += "(allow process-exec*\n";
    for (const auto& a : allowed) {
      p += "  (literal \"" + esc_(a) + "\")\n";
    }
    p += ")\n";
  }
  return p;
}
#endif

// Split `data` into complete lines, invoking `emit` per line (newline
// stripped). `carry` holds the partial line across chunks. flush_tail()
// emits whatever remains (an unterminated final line) at EOF.
struct LineSplitter {
  string carry;
  void
  feed(const char* data, size_t n,
       const function<void(string_view)>& emit)
  {
    if (!emit) {
      return;
    }
    size_t start = 0;
    for (size_t i = 0; i < n; ++i) {
      if (data[i] == '\n') {
        carry.append(data + start, i - start);
        emit(carry);
        carry.clear();
        start = i + 1;
      }
    }
    carry.append(data + start, n - start);
  }
  void
  flush_tail(const function<void(string_view)>& emit)
  {
    if (emit && !carry.empty()) {
      emit(carry);
      carry.clear();
    }
  }
};

}  // namespace

std::vector<std::string>
system_temp_roots()
{
  std::vector<std::string> out;
#if defined(__APPLE__)
  auto add_confstr = [&out](int name) {
    const size_t n = ::confstr(name, nullptr, 0);
    if (n == 0) { return; }
    std::string buf(n, '\0');
    if (::confstr(name, buf.data(), n) == 0) { return; }
    buf.resize(::strlen(buf.c_str()));      // drop confstr's trailing NUL
    if (!buf.empty()) { out.push_back(canon_(buf)); }
  };
  add_confstr(_CS_DARWIN_USER_TEMP_DIR);
  add_confstr(_CS_DARWIN_USER_CACHE_DIR);
#endif
  return out;
}

CommandSandboxResult
run_shell_command(const std::string&        command,
                  const CommandSandboxSpec& spec,
                  const CommandSandboxIo&   io)
{
  CommandSandboxResult r;
#if !(defined(__APPLE__) || defined(__unix__))
  (void)command;
  (void)spec;
  (void)io;
  r.error = "run_shell_command requires a POSIX host";
  return r;
#else

  // TMPDIR handed to a sandboxed child: the per-user system temp when
  // allow_system_temp granted it (writable under the profile below);
  // otherwise empty and the env setup points $TMPDIR under the workspace.
  string sandbox_tmpdir;
#if defined(__APPLE__)
  string profile;
  if (spec.enabled) {
    if (::access(kSandboxExec, X_OK) != 0) {
      r.error = "sandbox-exec not found (macOS seatbelt unavailable)";
      return r;
    }
    string home;
    if (const char* h = ::getenv("HOME"); h && *h) {
      home = canon_(h);
    }
    const vector<string> temp_roots =
        spec.allow_system_temp ? system_temp_roots() : vector<string>{};
    profile = build_profile_(spec, home, temp_roots);
    if (!temp_roots.empty()) { sandbox_tmpdir = temp_roots.front(); }
  }
#else
  if (spec.enabled) {
    r.error = "command sandbox requires macOS (seatbelt)";
    return r;
  }
#endif

  // Resolve the child working directory (canonical, created for a
  // sandboxed run so the write-allow subpath and the cwd agree).
  string cwd;
  if (!spec.cwd.empty()) {
    if (spec.enabled) {
      std::error_code mk;
      fs::create_directories(spec.cwd, mk);
    }
    cwd = canon_(spec.cwd);
  }

  // argv (parent-owned strings; c_str() captured just before fork).
  const char* const shell = shell_path_();
  vector<string> argv_s;
#if defined(__APPLE__)
  if (spec.enabled) {
    argv_s = {kSandboxExec, "-p", profile, shell, "-c", command};
  } else {
    argv_s = {shell, "-c", command};
  }
#else
  argv_s = {shell, "-c", command};
#endif
  vector<char*> argv;
  argv.reserve(argv_s.size() + 1);
  for (auto& s : argv_s) {
    argv.push_back(const_cast<char*>(s.c_str()));
  }
  argv.push_back(nullptr);

  // Environment. A sandboxed run gets a minimal, secret-free env with
  // HOME/TMPDIR pointed at the workspace; a native run inherits the
  // parent's so ordinary shell usage behaves as expected.
  vector<string> env_s;
  vector<char*>  envp;
  char**         child_env = environ;
  if (spec.enabled) {
    const string ws = !cwd.empty()
        ? cwd
        : (spec.writable_roots.empty() ? string(".")
                                       : canon_(spec.writable_roots.front()));
    // $TMPDIR: the per-user system temp when allow_system_temp granted it;
    // otherwise a dedicated subdir UNDER the workspace (already a writable
    // root, and under the launch CWD), so every temp file a command creates
    // stays inside the CWD. The subdir is created here (parent side) before
    // the child runs; falls back to the workspace root if it can't be made.
    string tmpdir = sandbox_tmpdir;
    if (tmpdir.empty()) {
      tmpdir = ws;
      std::error_code te;
      const fs::path td = fs::path(ws) / ".vpipe_tmp";
      fs::create_directories(td, te);
      if (!te) { tmpdir = td.string(); }
    }
    env_s = {
      "PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
      "HOME=" + ws,
      "TMPDIR=" + tmpdir,
      "LC_ALL=en_US.UTF-8",
    };
    envp.reserve(env_s.size() + 1);
    for (auto& s : env_s) {
      envp.push_back(const_cast<char*>(s.c_str()));
    }
    envp.push_back(nullptr);
    child_env = envp.data();
  }

  // Pipes: stdout + stderr always; stdin only when a source is supplied.
  int out_pipe[2] = {-1, -1};
  int err_pipe[2] = {-1, -1};
  int in_pipe[2]  = {-1, -1};
  const bool want_stdin = static_cast<bool>(io.provide_stdin);
  if (::pipe(out_pipe) != 0 || ::pipe(err_pipe) != 0
      || (want_stdin && ::pipe(in_pipe) != 0)) {
    r.error = string("pipe failed: ") + std::strerror(errno);
    for (int fd : {out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1],
                   in_pipe[0], in_pipe[1]}) {
      if (fd >= 0) { ::close(fd); }
    }
    return r;
  }

  const rlim_t cpu = spec.cpu_seconds > 0
      ? static_cast<rlim_t>(spec.cpu_seconds) : 0;
  const rlim_t asb = spec.address_space_mb > 0
      ? static_cast<rlim_t>(spec.address_space_mb) * 1024 * 1024 : 0;
  const rlim_t fsz = spec.file_size_mb > 0
      ? static_cast<rlim_t>(spec.file_size_mb) * 1024 * 1024 : 0;
  const char* cwd_cstr = cwd.empty() ? nullptr : cwd.c_str();
  const char* argv0    = argv_s.front().c_str();

  const pid_t pid = ::fork();
  if (pid < 0) {
    r.error = string("fork failed: ") + std::strerror(errno);
    for (int fd : {out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1],
                   in_pipe[0], in_pipe[1]}) {
      if (fd >= 0) { ::close(fd); }
    }
    return r;
  }
  if (pid == 0) {
    // ---- child: async-signal-safe syscalls only ----
    ::setsid();                        // own process group => group-kill
    ::close(out_pipe[0]);
    ::close(err_pipe[0]);
    ::dup2(out_pipe[1], STDOUT_FILENO);
    ::dup2(err_pipe[1], STDERR_FILENO);
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);
    if (want_stdin) {
      ::close(in_pipe[1]);
      ::dup2(in_pipe[0], STDIN_FILENO);
      ::close(in_pipe[0]);
    } else {
      const int nul = ::open("/dev/null", O_RDONLY);
      if (nul >= 0) {
        ::dup2(nul, STDIN_FILENO);
        ::close(nul);
      }
    }
    if (cpu) { rlimit l{cpu, cpu + 2}; ::setrlimit(RLIMIT_CPU, &l); }
    if (fsz) { rlimit l{fsz, fsz};     ::setrlimit(RLIMIT_FSIZE, &l); }
#ifdef RLIMIT_AS
    if (asb) { rlimit l{asb, asb};     ::setrlimit(RLIMIT_AS, &l); }
#else
    (void)asb;
#endif
    // No RLIMIT_NPROC: it caps the real UID's processes system-wide, so on an
    // interactive session the limit is already blown and the child's first
    // fork() (e.g. to run an external program from the shell) would fail.
    { rlimit l{0, 0}; ::setrlimit(RLIMIT_CORE, &l); }
    if (cwd_cstr) { ::chdir(cwd_cstr); }
    ::execve(argv0, argv.data(), child_env);
    ::_exit(127);                      // exec failed
  }

  // ---- parent ----
  ::close(out_pipe[1]);
  ::close(err_pipe[1]);
  if (want_stdin) {
    ::close(in_pipe[0]);
  }

  std::atomic<bool> aborted{false};

  // stdin pump: source lines from io.provide_stdin on a private thread and
  // feed them to the child. `aborted` (set once the child is gone) folds
  // into the cancel predicate so a blocking source unblocks promptly.
  std::thread pump;
  if (want_stdin) {
    const int wfd = in_pipe[1];
    pump = std::thread([&io, &aborted, wfd]() {
      auto cancel = [&aborted, &io]() {
        return aborted.load(std::memory_order_acquire)
            || (io.should_cancel && io.should_cancel());
      };
      string line;
      for (;;) {
        if (cancel()) { break; }
        line.clear();
        if (!io.provide_stdin(line, cancel)) { break; }   // EOF
        line.push_back('\n');
        size_t off = 0;
        bool broken = false;
        while (off < line.size()) {
          const ssize_t w = ::write(wfd, line.data() + off,
                                    line.size() - off);
          if (w < 0) {
            if (errno == EINTR) { continue; }
            broken = true;      // EPIPE: child closed its stdin
            break;
          }
          off += static_cast<size_t>(w);
        }
        if (broken) { break; }
      }
      ::close(wfd);
    });
  }

  // Drain stdout + stderr, routing each completed line to its hook, under
  // an optional wall-clock deadline and cancel poll.
  using clock = std::chrono::steady_clock;
  const bool have_deadline = spec.timeout_ms > 0;
  const auto deadline = clock::now()
      + std::chrono::milliseconds(spec.timeout_ms > 0 ? spec.timeout_ms : 0);
  LineSplitter out_split, err_split;
  int  ofd = out_pipe[0];
  int  efd = err_pipe[0];
  bool killed = false;
  char buf[4096];
  while (ofd >= 0 || efd >= 0) {
    // Timeout / cancel checks.
    bool fire_kill = false;
    if (have_deadline && !killed && clock::now() >= deadline) {
      r.timed_out = true;
      fire_kill = true;
    }
    if (!killed && io.should_cancel && io.should_cancel()) {
      r.canceled = true;
      fire_kill = true;
    }
    if (fire_kill) {
      killed = true;
      aborted.store(true, std::memory_order_release);
      // Kill the leader directly AND the process group: at t=0 the child
      // may not have finished setsid(), so its group may not exist yet --
      // the direct pid kill always lands, the group kill sweeps any
      // grandchildren once the session is set up.
      ::kill(pid, SIGKILL);
      ::kill(-pid, SIGKILL);
    }

    fd_set rd;
    FD_ZERO(&rd);
    int maxfd = -1;
    if (ofd >= 0) { FD_SET(ofd, &rd); maxfd = ofd > maxfd ? ofd : maxfd; }
    if (efd >= 0) { FD_SET(efd, &rd); maxfd = efd > maxfd ? efd : maxfd; }
    timeval tv;
    tv.tv_sec  = 0;
    tv.tv_usec = 200 * 1000;           // 200ms tick to re-poll cancel/deadline
    const int sel = ::select(maxfd + 1, &rd, nullptr, nullptr, &tv);
    if (sel < 0) {
      if (errno == EINTR) { continue; }
      break;
    }
    if (sel == 0) { continue; }        // tick -> re-evaluate deadline/cancel
    if (ofd >= 0 && FD_ISSET(ofd, &rd)) {
      const ssize_t n = ::read(ofd, buf, sizeof buf);
      if (n <= 0) {
        out_split.flush_tail(io.on_stdout_line);
        ::close(ofd);
        ofd = -1;
      } else {
        out_split.feed(buf, static_cast<size_t>(n), io.on_stdout_line);
      }
    }
    if (efd >= 0 && FD_ISSET(efd, &rd)) {
      const ssize_t n = ::read(efd, buf, sizeof buf);
      if (n <= 0) {
        err_split.flush_tail(io.on_stderr_line);
        ::close(efd);
        efd = -1;
      } else {
        err_split.feed(buf, static_cast<size_t>(n), io.on_stderr_line);
      }
    }
  }
  if (ofd >= 0) { ::close(ofd); }
  if (efd >= 0) { ::close(efd); }

  int status = 0;
  ::waitpid(pid, &status, 0);

  // The child is reaped: release any stdin source still blocked in
  // provide_stdin, then join the pump.
  aborted.store(true, std::memory_order_release);
  if (pump.joinable()) {
    pump.join();
  }

  r.ok = true;
  if (WIFEXITED(status)) {
    r.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    r.signaled    = true;
    r.term_signal = WTERMSIG(status);
  }
  return r;
#endif
}

}  // namespace vpipe
