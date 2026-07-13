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
constexpr const char* kShell = "/bin/sh";

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
// deny writes outside `writable`, deny the real $HOME reads, deny network
// (unless allowed), and -- when `exec_allow` is non-empty -- deny
// process-exec of anything but the shell and the whitelisted programs.
string
build_profile_(const CommandSandboxSpec& spec, const string& home)
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
  p += "  (subpath \"/dev\"))\n";
  if (!home.empty()) {
    p += "(deny file-read* (subpath \"" + esc_(home) + "\"))\n";
    // Re-open reads under the writable roots (a workspace may live under
    // $HOME, and the child must read what it just wrote there).
    for (const auto& w : spec.writable_roots) {
      p += "(allow file-read* (subpath \"" + esc_(canon_(w)) + "\"))\n";
    }
  }
  if (!spec.exec_allow.empty()) {
    vector<string> allowed;
    // The shell machinery must exec: /bin/sh on macOS is a bash "variant"
    // that internally re-execs /bin/bash, so both have to be allowed or
    // even a whitelisted program never gets a shell to launch it.
    resolve_exec_(kShell, allowed);
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
    profile = build_profile_(spec, home);
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
  vector<string> argv_s;
#if defined(__APPLE__)
  if (spec.enabled) {
    argv_s = {kSandboxExec, "-p", profile, kShell, "-c", command};
  } else {
    argv_s = {kShell, "-c", command};
  }
#else
  argv_s = {kShell, "-c", command};
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
    env_s = {
      "PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
      "HOME=" + ws,
      "TMPDIR=" + ws,
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
  const rlim_t nproc = spec.max_procs > 0
      ? static_cast<rlim_t>(spec.max_procs) : 0;
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
#ifdef RLIMIT_NPROC
    if (nproc) { rlimit l{nproc, nproc}; ::setrlimit(RLIMIT_NPROC, &l); }
#else
    (void)nproc;
#endif
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
