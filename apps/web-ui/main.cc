// vpipe-web-ui -- serves a browser UI bound to a single vpipe Session.
//
// Usage:
//   vpipe-web-ui [--bind ADDR] [--port N] [--config CFG]
//                [--memory-cap-mb N] [--wired-pool-mb N]
//
//   --bind      interface to listen on (default: this machine's LAN
//               address, i.e. en0's IPv4, so the UI is reachable from
//               other devices; falls back to 127.0.0.1 if no interface
//               is up)
//   --port      TCP port (default 9876; 0 = pick any free port)
//   --config    session config string forwarded to create_session
//               (inline JSON, a path, or empty for defaults)
//   --memory-cap-mb  ceiling on actively-resident model weights + KV.
//   --wired-pool-mb  ceiling on memory made unswappable (mlock'd). Also
//                    editable at run time from the Settings panel.
//               Over it the least-recently-used weights are PARKED
//               (handed to the kernel as purgeable, reclaimed only
//               under real pressure, taken back on next use) rather
//               than loads refused. Same as session config
//               memory_cap_mb; 0/unset = uncapped.
//
// The UI assets are embedded in the binary and served from memory.
//
// One Session is created for the process; the Pipeline Manager (and
// later views) drive it through the /api/* REST surface.

#include "apps/web-ui/http-server.h"
#include "apps/web-ui/qr-code.h"
#include "apps/web-ui/session-api.h"
#include "apps/web-ui/startup-checks.h"
#include "apps/web-ui/web-ui-delegate.h"
#include "apps/web-ui/web-ui-log-delegate.h"

#include "common/session.h"
#include "common/session-config.h"
#include "common/os-sandbox.h"
#include "common/temp-root.h"
#include "vpipe/session-intf.h"
#include "vpipe/session-manager.h"

#include <memory>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <system_error>
#include <cstdio>
#include <csignal>
#include <mutex>
#include <random>
#include <string>
#include <vector>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

using namespace std;
using namespace vpipe;

namespace {

atomic<bool>            g_stop{ false };
condition_variable      g_cv;
mutex                   g_cv_mu;

void
on_signal_(int)
{
  g_stop.store(true);
  g_cv.notify_all();
}

string
arg_value_(int argc, char** argv, const string& flag, const string& def)
{
  for (int i = 1; i + 1 < argc; ++i) {
    if (flag == argv[i]) { return argv[i + 1]; }
  }
  return def;
}

bool
has_flag_(int argc, char** argv, const string& flag)
{
  for (int i = 1; i < argc; ++i) {
    if (flag == argv[i]) { return true; }
  }
  return false;
}

// Collect EVERY value of a repeatable "--flag VALUE" option, in order.
vector<string>
arg_values_(int argc, char** argv, const string& flag)
{
  vector<string> out;
  for (int i = 1; i + 1 < argc; ++i) {
    if (flag == argv[i]) { out.emplace_back(argv[i + 1]); ++i; }
  }
  return out;
}

// True when stdout is an interactive terminal and the user hasn't opted
// out via NO_COLOR -- gates all ANSI coloring in this banner.
bool
stdout_color_()
{
  const char* nc = std::getenv("NO_COLOR");
  return ::isatty(fileno(stdout)) != 0 && (nc == nullptr || *nc == '\0');
}

// Startup banner. Bold-cyan when stdout is a TTY (and NO_COLOR unset),
// plain otherwise.
void
print_banner_()
{
  const bool color = stdout_color_();
  const char* c = color ? "\033[1;36m" : "";
  const char* r = color ? "\033[0m" : "";
  std::printf("%s"
R"( _    ______  ________  ______
| |  / / __ \/  _/ __ \/ ____/
| | / / /_/ // // /_/ / __/
| |/ / ____// // ____/ /___
|___/_/   /___/_/   /_____/   )"
              "%s\n\n", c, r);
  std::fflush(stdout);
}

void
print_usage_(const char* prog)
{
  printf(
    "vpipe-web-ui -- serve a browser UI bound to one vpipe Session.\n"
    "\n"
    "Usage: %s [options]\n"
    "\n"
    "Options:\n"
    "  --bind ADDR    Interface to listen on. Default: this machine's\n"
    "                 LAN address (en0), so other devices can connect.\n"
    "                 Use 127.0.0.1 for this machine only, or 0.0.0.0\n"
    "                 for all interfaces.\n"
    "  --port N       TCP port to listen on (default 9876; 0 picks any\n"
    "                 free port).\n"
    "  --config CFG   Session config forwarded to create_session:\n"
    "                 inline JSON, a file path, or empty for defaults.\n"
    "  --expose-native-file-system\n"
    "                 Disable the default filesystem sandbox and give\n"
    "                 stages full host file access. Off by default: stage\n"
    "                 file I/O is confined to ./sandbox (models exempt).\n"
    "  --white-list-path DIR\n"
    "                 Grant pass-through access to a real host directory\n"
    "                 (outside ./sandbox), reachable by its real path and\n"
    "                 shown as a mount in the file dialog. Repeatable.\n"
    "                 Ignored with --expose-native-file-system.\n"
    "  --os-sandbox   Add a whole-process OS write-backstop (macOS\n"
    "                 seatbelt) under the app-level sandbox. OFF by\n"
    "                 default: it cannot be nested, so it DISABLES the\n"
    "                 run_python chat tool (which sandboxes each call\n"
    "                 itself). Enable only without the python tool.\n"
    "  --memory-cap-mb N\n"
    "                 Ceiling on actively-resident model weights + KV.\n"
    "                 Over it the least-recently-used weights are PARKED\n"
    "                 (purgeable), not refused. 0/unset = uncapped.\n"
    "  --wired-pool-mb N\n"
    "                 Ceiling on memory made UNSWAPPABLE (mlock'd):\n"
    "                 weights, a streaming model's resident blocks,\n"
    "                 activation scratch. Clamped to what the GPU can\n"
    "                 keep resident. Unset = the wired_pool_pct share of\n"
    "                 RAM (default 75 percent), 0 = no wiring. Also\n"
    "                 editable at run time from the Settings panel --\n"
    "                 raisable while a pipeline runs, not lowerable.\n"
    "  --plugin PATH  Load a plugin .dylib at startup (adds stages /\n"
    "                 shaders / models). Repeatable. See docs/PLUGINS.md.\n"
    "  --tls          Serve over HTTPS with a cached self-signed cert\n"
    "                 (~/.vpipe/webui-tls). Needed for the low-latency\n"
    "                 Preview view on LAN clients: browser WebCodecs only\n"
    "                 works in a secure context (HTTPS or localhost). The\n"
    "                 cert is self-signed, so accept the one-time browser\n"
    "                 warning. Requires an OpenSSL-enabled build.\n"
    "  --show-qr      Also print a QR code for a one-scan link that\n"
    "                 carries the access key, so a phone does not have to\n"
    "                 retype it. Scanning opens the UI already\n"
    "                 authenticated and drops the key from the address\n"
    "                 bar. The link is a long secret in a URL: it is only\n"
    "                 as private as the console showing it.\n"
    "  --emit-connection-json PATH\n"
    "                 Write {url, scheme, host, port, key, qr_link, pid}\n"
    "                 to PATH once listening, mode 0600, and remove it on\n"
    "                 exit. For a GUI launcher: it carries the bound port\n"
    "                 (which --port 0 only settles at startup) and the\n"
    "                 --show-qr link, neither of which is on stdout.\n"
    "  -h, --help     Show this help and exit.\n"
    "\n"
    "Remote access: connections from other computers must supply the\n"
    "8-character key printed at startup; localhost connects without one.\n",
    prog);
}

// 8-char access key from an unambiguous alphanumeric alphabet (no
// 0/O/1/l/I) so it's easy to read off the console and retype.
string
random_key_(size_t n = 8)
{
  static const char alpha[] =
      "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789";
  std::random_device rd;
  std::uniform_int_distribution<size_t> dist(0, sizeof(alpha) - 2);
  string out;
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) { out.push_back(alpha[dist(rd)]); }
  return out;
}

// The --show-qr link secret. UPPER CASE on purpose, from the alphabet a
// QR alphanumeric segment allows: that mode packs 5.5 bits per character
// where byte mode spends 8, which is the difference between a 25x25
// symbol and a larger one for a typical LAN URL. 14 characters of a
// 32-symbol alphabet is 70 bits -- far more than the 8-char console key,
// which is what a secret living in a URL should be.
string
random_qr_key_(size_t n = 14)
{
  static const char alpha[] = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  std::random_device rd;
  std::uniform_int_distribution<size_t> dist(0, sizeof(alpha) - 2);
  string out;
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) { out.push_back(alpha[dist(rd)]); }
  return out;
}

// --emit-connection-json: the details a GUI launcher needs to present
// this server -- the URL to open, the access key to show, the QR link
// when there is one -- written as JSON, atomically, mode 0600.
//
// A front end could scrape stdout instead, but two things are not there
// to scrape. The QR link is DELIBERATELY never printed (it carries the
// link secret in plain text, which is precisely what the symbol exists
// to keep out of scrollback and screen shares), and with --port 0 the
// bound port is not known until after start(). So this file is the one
// channel carrying all three -- and being mode 0600, it carries the key
// no more widely than the console already does.
//
// Written to a fresh O_EXCL temp and renamed: a reader started at the
// same time sees either no file or a complete one, never a half-written
// one, and the key never exists in a file another user could open, not
// even for the instant before a chmod.
//
// No JSON escaping: every value comes from a constrained alphabet -- a
// scheme literal, an IP or hostname from the interface list, an integer,
// and keys from the two generators above -- so none can contain a
// character that would need it.
bool
write_connection_json_(const string& path,
                       const string& scheme,
                       const string& host,
                       int           port,
                       const string& key,
                       const string& qr_link,
                       string*       err)
{
  const string tmp = path + ".tmp";
  ::unlink(tmp.c_str());
  const int fd =
      ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_TRUNC, 0600);
  if (fd < 0) {
    if (err) { *err = string("open: ") + strerror(errno); }
    return false;
  }

  string js = "{\n";
  js += "  \"url\": \"" + scheme + "://" + host + ":"
      + std::to_string(port) + "/\",\n";
  js += "  \"scheme\": \"" + scheme + "\",\n";
  js += "  \"host\": \"" + host + "\",\n";
  js += "  \"port\": " + std::to_string(port) + ",\n";
  js += "  \"key\": \"" + key + "\",\n";
  if (!qr_link.empty()) {
    js += "  \"qr_link\": \"" + qr_link + "\",\n";
  }
  js += "  \"pid\": " + std::to_string(static_cast<long>(::getpid()))
      + "\n";
  js += "}\n";

  bool   ok  = true;
  size_t off = 0;
  while (off < js.size()) {
    const ssize_t n = ::write(fd, js.data() + off, js.size() - off);
    if (n <= 0) {
      if (errno == EINTR) { continue; }
      if (err) { *err = string("write: ") + strerror(errno); }
      ok = false;
      break;
    }
    off += static_cast<size_t>(n);
  }
  ::close(fd);

  if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) {
    if (ok && err) { *err = string("rename: ") + strerror(errno); }
    ::unlink(tmp.c_str());
    return false;
  }
  return true;
}

}  // namespace

int
main(int argc, char** argv)
{
  if (has_flag_(argc, argv, "--help") || has_flag_(argc, argv, "-h")) {
    print_usage_(argv[0]);
    return 0;
  }

  print_banner_();

  // Default bind: this machine's LAN address (en0's IPv4) so the UI is
  // reachable from other devices out of the box; fall back to loopback
  // when no interface is up. An explicit --bind always wins.
  string bind_addr;
  if (has_flag_(argc, argv, "--bind")) {
    bind_addr = arg_value_(argc, argv, "--bind", "127.0.0.1");
  } else {
    string lan = webui::primary_ipv4();
    bind_addr = lan.empty() ? "127.0.0.1" : lan;
  }
  string port_str  = arg_value_(argc, argv, "--port", "9876");
  // Forwarded through the environment for the same reason as the CLI:
  // the model manager is not reachable from the public SessionIntf.
  const string cap = arg_value_(argc, argv, "--memory-cap-mb", "");
  if (!cap.empty()) { ::setenv("VPIPE_MEMORY_CAP_MB", cap.c_str(), 1); }
  // The wired pool's ceiling. Editable afterwards from the Settings
  // panel, which is why this is only the STARTING value -- see
  // SessionIntf::set_wired_pool_mb for what a running pipeline allows.
  const string wired = arg_value_(argc, argv, "--wired-pool-mb", "");
  if (!wired.empty()) { ::setenv("VPIPE_WIRED_POOL_MB", wired.c_str(), 1); }
  string cfg       = arg_value_(argc, argv, "--config", "");
  int    port      = static_cast<int>(strtol(port_str.c_str(), nullptr, 10));

  // Filesystem sandbox: by default the web-ui confines every stage's file
  // I/O to a "sandbox" directory under the starting CWD (model-manager
  // access stays exempt). --expose-native-file-system opts back into full
  // host access. The setting is folded into the session config under the
  // `file_sandbox` key, which the Session reads at construction.
  const bool expose_native =
      has_flag_(argc, argv, "--expose-native-file-system");
  FlexData cfg_obj;
  try {
    cfg_obj = cfg.empty() ? FlexData::make_object()
                          : parse_session_config(cfg);
  } catch (const std::exception& e) {
    fprintf(stderr, "web-ui: failed to parse --config: %s\n", e.what());
    return 1;
  }
  if (!cfg_obj.is_object()) {
    fprintf(stderr, "web-ui: --config must resolve to a JSON object\n");
    return 1;
  }
  if (!expose_native) {
    std::error_code ec;
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    const std::filesystem::path root =
        (ec ? std::filesystem::path(".") : cwd) / "sandbox";
    FlexData fs_obj = FlexData::make_object();
    fs_obj.as_object().insert_or_assign("enabled", FlexData::make_bool(true));
    fs_obj.as_object().insert_or_assign(
        "root", FlexData::make_string(root.string()));
    // --white-list-path DIR (repeatable): real host prefixes granted
    // pass-through access, reachable by their real path (and offered as
    // "mounts" in the file dialog). Everything else stays confined.
    const std::vector<std::string> wl =
        arg_values_(argc, argv, "--white-list-path");
    if (!wl.empty()) {
      FlexData wl_arr = FlexData::make_array();
      auto wa = wl_arr.as_array();
      for (const auto& p : wl) {
        wa.push_back(FlexData::make_string(p));
      }
      fs_obj.as_object().insert_or_assign("whitelist", std::move(wl_arr));
    }
    cfg_obj.as_object().insert_or_assign("file_sandbox", std::move(fs_obj));
  }

  // --plugin PATH (repeatable): plugin .dylibs to dlopen at session
  // startup, before any pipeline is built (in addition to any set via
  // the VPIPE_PLUGINS env). See docs/PLUGINS.md.
  {
    const std::vector<std::string> plugins =
        arg_values_(argc, argv, "--plugin");
    if (!plugins.empty()) {
      FlexData arr = FlexData::make_array();
      auto pa = arr.as_array();
      for (const auto& p : plugins) {
        pa.push_back(FlexData::make_string(p));
      }
      cfg_obj.as_object().insert_or_assign("plugins", std::move(arr));
    }
  }
  const string merged_cfg = cfg_obj.to_json();

  // One session for this process. create_session returns a borrowed
  // const pointer; we const_cast to call the mutating control methods
  // (the underlying object is not actually const -- see session-manager.h).
  const SessionIntf* csess =
      SessionManager::get().create_session(merged_cfg);
  if (!csess) {
    fprintf(stderr, "web-ui: failed to create session (bad --config?)\n");
    return 1;
  }
  SessionIntf* session = const_cast<SessionIntf*>(csess);

  // Divert the session's user-facing I/O (error/warn/info + getline)
  // to the browser's User I/O page. The concrete Session owns the
  // delegate; SessionApi keeps a borrowed pointer to back /api/io/*.
  // (SessionManager always hands back a Session; guard the cast.)
  webui::WebUiDelegate*    ui  = nullptr;
  webui::WebUiLogDelegate* log = nullptr;
  if (Session* concrete = dynamic_cast<Session*>(session)) {
    // Record the UI's bind interface so HTTP-hosting stages (e.g.
    // hls-broadcast) default to the same address the browser reaches
    // the UI on -- an HLS stream then shows up on the same host:port
    // family without extra config.
    concrete->set_web_ui_bind_address(bind_addr);
    auto d = std::make_unique<webui::WebUiDelegate>();
    ui = d.get();
    concrete->set_ui_delegate(std::move(d));
    // Divert the session's diagnostic log stream (log_* channels) to
    // the browser's Session Log view, the way the UI delegate diverts
    // error/warn/info. SessionApi keeps a borrowed pointer for the
    // /api/log/* routes.
    auto ld = std::make_unique<webui::WebUiLogDelegate>();
    log = ld.get();
    concrete->set_log_delegate(std::move(ld));
  }

  // OS-level filesystem-WRITE backstop (macOS seatbelt), OPT-IN via
  // --os-sandbox. It is NOT on by default because a whole-process seatbelt
  // cannot be nested: it breaks the per-call `sandbox-exec` that the
  // `run_python` chat tool uses to isolate each call (sandbox_apply ->
  // "Operation not permitted"). The default confinement is the app-level
  // sandbox for stages (always on here) plus each python call's own
  // per-call seatbelt. Enable this only for deployments that do not use
  // the in-process python tool. Fail-open.
  if (!expose_native && has_flag_(argc, argv, "--os-sandbox")) {
    fprintf(stderr, "web-ui: --os-sandbox: applying a whole-process "
                    "seatbelt; the run_python chat tool will be disabled "
                    "(nested sandboxes are not permitted).\n");
    std::error_code sec;
    const std::filesystem::path cwd = std::filesystem::current_path(sec);
    OsSandboxSpec spec;
    spec.writable_roots.push_back(sec ? std::filesystem::path(".") : cwd);
    spec.writable_roots.push_back(temp_root());
    std::string serr;
    switch (apply_os_file_sandbox(spec, &serr)) {
      case OsSandboxStatus::Applied:
        break;
      case OsSandboxStatus::Failed:
        fprintf(stderr, "web-ui: OS sandbox not applied: %s "
                        "(app-level sandbox still active)\n", serr.c_str());
        break;
      case OsSandboxStatus::Unavailable:
        fprintf(stderr, "web-ui: OS sandbox unavailable on this platform "
                        "(app-level sandbox still active)\n");
        break;
    }
  }

  // DECLARATION ORDER IS LOAD-BEARING. Handler threads run inside the
  // server and borrow the api (and, through its view host, the session's
  // pipelines), so the SERVER must be destroyed first -- its destructor
  // stops accepting and drains the handlers. Declaring the api first
  // makes it outlive them. Reversed, the api is torn down while handlers
  // are still running and the first one to take its mutex aborts the
  // process on shutdown.
  webui::SessionApi api(session, ui, log);
  webui::HttpServer server(bind_addr, port);
  string auth_key = random_key_();
  server.set_auth_key(auth_key);
  const bool show_qr = has_flag_(argc, argv, "--show-qr");
  string qr_key;
  if (show_qr) {
    qr_key = random_qr_key_();
    server.set_qr_key(qr_key);
  }
  api.register_routes(server);
  // Publish the view host through the UI delegate, so a stage reaching
  // for its session's delegate can tell a graphical front end is
  // attached and enumerate stages for its own panel.
  if (ui != nullptr) { ui->set_view_host(api.view_host()); }

  // HTTPS (opt-in): must be enabled before start(). A self-signed cert is
  // generated + cached under ~/.vpipe/webui-tls on first run. Required for
  // the Preview view on non-localhost clients (WebCodecs is secure-context
  // only). Fatal if requested but unavailable -- the operator asked for it.
  if (has_flag_(argc, argv, "--tls")) {
    const char* home = std::getenv("HOME");
    const string tls_dir =
        (home && *home ? string(home) : string(".")) + "/.vpipe/webui-tls";
    string terr;
    if (!server.enable_tls(tls_dir, bind_addr, &terr)) {
      fprintf(stderr, "web-ui: --tls failed: %s\n", terr.c_str());
      SessionManager::get().destroy_session(session);
      return 1;
    }
  }

  if (!server.start()) {
    fprintf(stderr, "web-ui: could not start HTTP server on %s:%d\n",
            bind_addr.c_str(), port);
    SessionManager::get().destroy_session(session);
    return 1;
  }

  signal(SIGINT, on_signal_);
  signal(SIGTERM, on_signal_);

  // Highlight the two things the operator needs to act on: the URL to
  // open and the access key to type. Bold-cyan URL, bold-green key.
  const bool color = stdout_color_();
  const char* url_c = color ? "\033[1;36m" : "";
  const char* key_c = color ? "\033[1;32m" : "";
  const char* rst   = color ? "\033[0m" : "";
  const char* scheme = server.tls_enabled() ? "https" : "http";
  printf("vpipe-web-ui listening on %s%s://%s:%d/%s\n",
         url_c, scheme, bind_addr.c_str(), server.bound_port(), rst);
  if (server.tls_enabled()) {
    printf("  (HTTPS with a self-signed cert -- accept the one-time browser "
           "warning; this is what enables the Preview view on LAN clients)\n");
  }
  printf("Remote access key: %s%s%s\n", key_c, auth_key.c_str(), rst);
  printf("  (localhost connects without a key; other computers must "
         "enter this key)\n");

  // --show-qr: the same access, as something a phone camera can take.
  // The scheme and host are UPPER-CASED for the symbol only -- both are
  // case-insensitive per RFC 3986, and staying inside the QR
  // alphanumeric set is what keeps a typical LAN URL down to a 25x25
  // symbol. The path is already upper case by construction.
  //
  // The URL itself is NOT printed. It carries the link secret in plain
  // text, so echoing it beside the symbol would put on the console --
  // and into any scrollback, screen share or terminal log -- exactly
  // what the QR exists to keep off it. The symbol is the only rendering.
  if (show_qr) {
    string host = bind_addr;
    for (char& ch : host) { ch = static_cast<char>(toupper(ch)); }
    string scheme_up = server.tls_enabled() ? "HTTPS" : "HTTP";
    const string url = scheme_up + "://" + host + ":"
                     + std::to_string(server.bound_port()) + "/" + qr_key;
    const webui::QrCode qr = webui::qr_encode(url);
    if (qr.version == 0) {
      // Cannot happen for a v4 address at this key length, and the
      // fallback is the key above rather than the link -- printing the
      // link would defeat the point of hiding it.
      printf("  (address too long for a QR symbol -- use the key above)\n");
    } else {
      printf("  (alternatively, scan the QR code to get authenticated)\n");
      printf("\n");
      fputs(webui::qr_terminal(qr, color).c_str(), stdout);
    }
    // Conditional, and only where the symbol would otherwise fail
    // silently: a scan that cannot reach the host, or one that lands on
    // a certificate warning, both look like "the QR code is broken".
    if (bind_addr == "127.0.0.1" || bind_addr == "localhost") {
      printf("  %sNOTE%s: bound to loopback, so only this machine can "
             "reach it. Use --bind <lan-ip> for a phone.\n", key_c, rst);
    }
    if (server.tls_enabled()) {
      printf("  NOTE: with --tls the certificate is self-signed, so the "
             "phone shows a warning to accept first.\n");
    }
  }
  fflush(stdout);

  // --emit-connection-json: publish before the startup checks, not
  // after. Those probe the network and can sit in a multi-second
  // timeout on an unreachable LAN, and a launcher polling for this file
  // is showing the user a spinner until it lands. The server is already
  // accepting connections at this point, so the details are true.
  const string conn_json =
      arg_value_(argc, argv, "--emit-connection-json", "");
  if (!conn_json.empty()) {
    const string qr_link =
        show_qr ? string(scheme) + "://" + bind_addr + ":"
                    + std::to_string(server.bound_port()) + "/" + qr_key
                : string();
    string werr;
    if (!write_connection_json_(conn_json, scheme, bind_addr,
                                server.bound_port(), auth_key, qr_link,
                                &werr)) {
      // Not fatal: the server is up and the console still shows the URL
      // and key, so a human can carry on. Only the launcher loses out.
      fprintf(stderr, "web-ui: --emit-connection-json %s: %s\n",
              conn_json.c_str(), werr.c_str());
    }
  }

  // Quick permission self-tests (local network / full disk / mic) -- print
  // colored, actionable warnings pointing at System Settings. The blocking
  // network probes watch g_stop so a Ctrl-C during them (e.g. when the LAN
  // is unreachable and they're mid-timeout) shuts down promptly. The
  // structured results are handed to the API so the browser shows the same
  // report in a dialog when a client connects.
  api.set_startup_checks(webui::run_permission_checks(&g_stop));

  printf("Press Ctrl-C to stop.\n");
  fflush(stdout);

  {
    unique_lock<mutex> lk(g_cv_mu);
    g_cv.wait(lk, [] { return g_stop.load(); });
  }

  printf("\nweb-ui: shutting down...\n");
  fflush(stdout);
  // Take the connection file down before the port closes, so a launcher
  // never reads a live-looking key for a server that has stopped
  // answering. Best-effort: a kill -9 leaves it behind, which is why it
  // carries the pid a reader can check.
  if (!conn_json.empty()) { ::unlink(conn_json.c_str()); }
  server.stop();
  SessionManager::get().destroy_session(session);
  return 0;
}
