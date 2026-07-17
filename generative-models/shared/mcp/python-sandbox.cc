#include "generative-models/shared/mcp/python-sandbox.h"

#include "common/command-sandbox.h"
#include "common/flex-data.h"
#include "common/temp-root.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <signal.h>
#include <sys/resource.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>
#endif

using namespace std;

namespace vpipe {

#if defined(__APPLE__)
namespace fs = std::filesystem;

namespace {

// Absolute python3, homebrew first (its framework lives under /opt, which
// stays readable under the allow-default profile). Empty if none found.
string
find_python3_()
{
  const char* cands[] = {
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3",
    "/usr/bin/python3",
  };
  for (const char* c : cands) {
    if (::access(c, X_OK) == 0) {
      return c;
    }
  }
  return {};
}

// Seatbelt profile: allow-default, but deny network, confine writes to
// the scratch dir (+ /dev), and deny reads of the invoking user's real
// home so model code can't slurp secrets into the reply.
string
build_profile_(const string& scratch_real, const string& home_real,
               const vector<string>& temp_roots)
{
  string p;
  p += "(version 1)\n";
  p += "(allow default)\n";
  p += "(deny network*)\n";
  p += "(deny file-write*)\n";
  p += "(allow file-write*\n";
  p += "  (subpath \"" + scratch_real + "\")\n";
  // The per-user system temp when allow_system_temp granted it (so code that
  // shells out to tools hardcoding it works); empty otherwise, keeping temp
  // inside the scratch dir under the CWD.
  for (const auto& t : temp_roots) {
    p += "  (subpath \"" + t + "\")\n";
  }
  p += "  (subpath \"/dev\"))\n";
  if (!home_real.empty()) {
    // Deny reads under the real $HOME so model code can't slurp secrets into
    // the reply -- but then re-allow file-read-METADATA. A blunt file-read*
    // deny also blocks the stat/lookup that os.getcwd() / os.chdir() / path
    // resolution need, so any code touching its own working directory breaks
    // when the scratch dir lives under $HOME (temp_root() defaults to
    // $CWD/.vpipe-tmp). Re-allowing metadata (stat/traversal only -- NOT file
    // contents or directory listing) keeps secrets unreadable while
    // navigation works; contents re-open only under the scratch dir below. A
    // more-specific file-read-data deny would instead beat the scratch
    // file-read* allow and break reads in the scratch itself.
    p += "(deny file-read* (subpath \"" + home_real + "\"))\n";
    p += "(allow file-read-metadata (subpath \"" + home_real + "\"))\n";
    p += "(allow file-read* (subpath \"" + scratch_real + "\"))\n";
  }
  return p;
}

}  // namespace
#endif

PythonSandboxResult
run_python_sandboxed(const string& code, const PythonSandboxOptions& opts)
{
  PythonSandboxResult r;
#if !defined(__APPLE__)
  (void)code;
  (void)opts;
  r.error = "python sandbox requires macOS (seatbelt)";
  return r;
#else
  static constexpr const char* kSandboxExec = "/usr/bin/sandbox-exec";
  if (::access(kSandboxExec, X_OK) != 0) {
    r.error = "sandbox-exec not found (macOS seatbelt unavailable)";
    return r;
  }
  const string py = find_python3_();
  if (py.empty()) {
    r.error = "python3 not found";
    return r;
  }

  // Ephemeral scratch dir under the app temp root (CWD-local). canonical()
  // below resolves any /var -> /private/var symlink so the seatbelt
  // subpath rule actually matches at runtime.
  std::error_code ec;
  const fs::path base = vpipe::temp_root();
  string tmpl = (base / "vpipe-pysbx-XXXXXX").string();
  vector<char> tbuf(tmpl.begin(), tmpl.end());
  tbuf.push_back('\0');
  if (!::mkdtemp(tbuf.data())) {
    r.error = string("mkdtemp failed: ") + std::strerror(errno);
    return r;
  }
  const fs::path scratch = tbuf.data();
  fs::path scratch_c = fs::canonical(scratch, ec);
  const string scratch_s = (ec ? scratch : scratch_c).string();

  string home_real;
  if (const char* h = ::getenv("HOME"); h && *h) {
    fs::path hc = fs::canonical(h, ec);
    home_real = ec ? string(h) : hc.string();
  }

  const vector<string> temp_roots =
      opts.allow_system_temp ? system_temp_roots() : vector<string>{};
  const string profile = build_profile_(scratch_s, home_real, temp_roots);
  // TMPDIR: the per-user system temp when granted (writable above), else the
  // ephemeral scratch under the CWD.
  const string tmpdir = temp_roots.empty() ? scratch_s : temp_roots.front();

  // argv: sandbox-exec -p <profile> python3 -I -B -c <code>. -I isolates
  // python (ignores env / user site); -B suppresses .pyc writes.
  vector<string> argv_s = {
    kSandboxExec, "-p", profile, py, "-I", "-B", "-c", code,
  };
  vector<char*> argv;
  argv.reserve(argv_s.size() + 1);
  for (auto& s : argv_s) {
    argv.push_back(const_cast<char*>(s.c_str()));
  }
  argv.push_back(nullptr);

  // Minimal env: HOME -> scratch (so `~` lands in the writable dir); TMPDIR
  // -> scratch by default, or the system temp when allow_system_temp; no
  // inherited secrets.
  const string env_home = "HOME=" + scratch_s;
  const string env_tmp  = "TMPDIR=" + tmpdir;
  vector<string> env_s = {
    "PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
    env_home, env_tmp,
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONNOUSERSITE=1",
    "LC_ALL=en_US.UTF-8",
  };
  vector<char*> envp;
  envp.reserve(env_s.size() + 1);
  for (auto& s : env_s) {
    envp.push_back(const_cast<char*>(s.c_str()));
  }
  envp.push_back(nullptr);

  int pipefd[2];
  if (::pipe(pipefd) != 0) {
    r.error = string("pipe failed: ") + std::strerror(errno);
    fs::remove_all(scratch, ec);
    return r;
  }

  // rlimits reduced to primitives so the child touches no libc++ state.
  const rlim_t cpu   = opts.cpu_seconds     > 0
      ? static_cast<rlim_t>(opts.cpu_seconds) : 0;
  const rlim_t asb   = opts.address_space_mb > 0
      ? static_cast<rlim_t>(opts.address_space_mb) * 1024 * 1024 : 0;
  const rlim_t fsz   = opts.file_size_mb    > 0
      ? static_cast<rlim_t>(opts.file_size_mb) * 1024 * 1024 : 0;
  const char* scratch_cstr = scratch_s.c_str();

  const pid_t pid = ::fork();
  if (pid < 0) {
    r.error = string("fork failed: ") + std::strerror(errno);
    ::close(pipefd[0]);
    ::close(pipefd[1]);
    fs::remove_all(scratch, ec);
    return r;
  }
  if (pid == 0) {
    // ---- child: async-signal-safe syscalls only ----
    ::setsid();                       // new process group => group-kill
    ::close(pipefd[0]);
    ::dup2(pipefd[1], STDOUT_FILENO);
    ::dup2(pipefd[1], STDERR_FILENO);
    ::close(pipefd[1]);
    const int nul = ::open("/dev/null", O_RDONLY);
    if (nul >= 0) {
      ::dup2(nul, STDIN_FILENO);
      ::close(nul);
    }
    if (cpu) { rlimit l{cpu, cpu + 2}; ::setrlimit(RLIMIT_CPU, &l); }
    if (fsz) { rlimit l{fsz, fsz};     ::setrlimit(RLIMIT_FSIZE, &l); }
#ifdef RLIMIT_AS
    if (asb) { rlimit l{asb, asb};     ::setrlimit(RLIMIT_AS, &l); }
#else
    (void)asb;
#endif
    // No RLIMIT_NPROC: it caps the real UID's processes system-wide, so on an
    // interactive session it is already exceeded and the child's first fork()
    // would fail with EAGAIN.
    { rlimit l{0, 0}; ::setrlimit(RLIMIT_CORE, &l); }
    ::chdir(scratch_cstr);            // best-effort cwd = scratch
    ::execve(kSandboxExec, argv.data(), envp.data());
    ::_exit(127);                     // exec failed
  }

  // ---- parent: drain stdout+stderr under a wall-clock deadline ----
  ::close(pipefd[1]);
  const int rfd = pipefd[0];
  using clock = std::chrono::steady_clock;
  const auto deadline =
      clock::now() + std::chrono::milliseconds(opts.timeout_ms);
  auto drain_deadline = deadline;   // extended after we kill
  bool killed = false;
  char buf[4096];
  for (;;) {
    const auto now = clock::now();
    if (!killed && now >= deadline) {
      r.timed_out = true;
      killed = true;
      ::kill(-pid, SIGKILL);        // kill the whole process group
      drain_deadline = now + std::chrono::milliseconds(2000);
    }
    if (killed && now >= drain_deadline) {
      break;
    }
    const auto until = killed ? drain_deadline : deadline;
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        until - now).count();
    if (ms < 0) { ms = 0; }
    fd_set rd;
    FD_ZERO(&rd);
    FD_SET(rfd, &rd);
    timeval tv;
    tv.tv_sec  = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    const int sel = ::select(rfd + 1, &rd, nullptr, nullptr, &tv);
    if (sel < 0) {
      if (errno == EINTR) { continue; }
      break;
    }
    if (sel == 0) { continue; }      // deadline tick -> re-evaluate
    const ssize_t n = ::read(rfd, buf, sizeof buf);
    if (n <= 0) { break; }           // EOF or error
    if (r.output.size() < opts.max_output_bytes) {
      const size_t room = opts.max_output_bytes - r.output.size();
      const size_t take = static_cast<size_t>(n) <= room
          ? static_cast<size_t>(n) : room;
      r.output.append(buf, take);
      if (static_cast<size_t>(n) > room) { r.output_truncated = true; }
    } else {
      r.output_truncated = true;     // keep draining, discard
    }
  }
  ::close(rfd);
  int status = 0;
  ::waitpid(pid, &status, 0);
  r.ok = true;
  if (WIFEXITED(status)) {
    r.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    r.signaled    = true;
    r.term_signal = WTERMSIG(status);
  }
  fs::remove_all(scratch, ec);
  return r;
#endif
}

namespace {

// Read the `code` argument as either a single string or an ARRAY of
// source lines joined with '\n'. The array form lets a model emit code
// line-by-line, which sidesteps the frequent failure of not escaping
// quotes inside one big JSON string. Returns "" when absent / wrong type.
string
extract_code_(const FlexData& args)
{
  if (!args.is_object()) {
    return {};
  }
  auto o = args.as_object();
  if (!o.contains("code")) {
    return {};
  }
  const FlexData code = o.at("code");
  if (code.is_string()) {
    return string(code.get_string());
  }
  if (code.is_array()) {
    string out;
    auto arr = code.as_array();
    for (size_t i = 0; i < arr.size(); ++i) {
      if (i) {
        out.push_back('\n');
      }
      const FlexData line = arr[i];
      out += line.is_string() ? string(line.get_string())
                              : string(line.as_string(""));
    }
    return out;
  }
  return {};
}

}  // namespace

McpTool
make_python_tool(const PythonSandboxOptions& opts)
{
  McpTool t;
  t.name = "run_python";
  t.description =
      "Execute a short Python 3 snippet in a locked-down sandbox (no "
      "network, ephemeral scratch filesystem, CPU/time limits) and return "
      "its stdout/stderr. Print any results you want to see.";
  t.parameters_json =
      "{\"type\":\"object\",\"properties\":{\"code\":{\"type\":"
      "[\"string\",\"array\"],\"items\":{\"type\":\"string\"},"
      "\"description\":\"Python 3 source to run: either a single string, "
      "or an array of source lines (joined with newlines). Prefer the "
      "array form for multi-line code to avoid quote-escaping mistakes. "
      "Print output to stdout.\"}},\"required\":[\"code\"]}";
  t.handler = [opts](const string& args_json) -> string {
    string code;
    try {
      code = extract_code_(FlexData::from_json(args_json));
    } catch (const std::exception&) {
      // fall through: empty code -> error result below
    }
    FlexData res = FlexData::make_object();
    auto ro = res.as_object();
    if (code.empty()) {
      ro.insert("error", FlexData::make_string(
          "missing 'code' argument (string or array of lines)"));
      return res.to_json(false);
    }
    const PythonSandboxResult pr = run_python_sandboxed(code, opts);
    if (!pr.ok && !pr.error.empty()) {
      ro.insert("error", FlexData::make_string(pr.error));
      return res.to_json(false);
    }
    ro.insert("stdout", FlexData::make_string(pr.output));
    ro.insert("exit_code", FlexData::make_int(pr.exit_code));
    if (pr.timed_out) {
      ro.insert("timed_out", FlexData::make_bool(true));
    }
    if (pr.signaled) {
      ro.insert("signal", FlexData::make_int(pr.term_signal));
    }
    if (pr.output_truncated) {
      ro.insert("truncated", FlexData::make_bool(true));
    }
    return res.to_json(false);
  };
  return t;
}

}
