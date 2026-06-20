#ifndef WEBUI_STARTUP_CHECKS_H
#define WEBUI_STARTUP_CHECKS_H

#include <atomic>
#include <string>
#include <vector>

namespace vpipe::webui {

// Best-effort IPv4 address of the primary network interface (en0 on
// macOS; otherwise the first non-loopback IPv4). Empty string if none
// is up -- the caller should fall back to 127.0.0.1.
std::string primary_ipv4();

// One startup permission/self-test result, the structured twin of a
// console line so the same information can be surfaced in the web UI.
struct PermissionCheck {
  std::string              name;     // e.g. "Local network"
  std::string              status;   // "ok" | "warn"
  std::string              detail;   // e.g. "LAN multicast reachable"
  std::vector<std::string> hints;    // actionable follow-up lines (warn)
};

// Run the startup permission self-tests and print colored, actionable
// warnings to the console pointing the user at System Settings > Privacy
// & Security. macOS-focused (Local Network / Full Disk Access /
// Microphone are macOS TCC permissions); on other platforms the checks
// that don't apply are skipped. Color is auto-disabled when stdout is
// not a TTY or NO_COLOR is set.
//
// `abort` (optional): when it flips true mid-run (e.g. a Ctrl-C shutdown
// flag), the blocking network probes bail within ~100 ms and the
// remaining checks are skipped, so shutdown stays responsive even when
// the LAN is unreachable.
//
// Returns the structured results (in console order) so a caller can mirror
// them in a UI; the console printing is unchanged.
std::vector<PermissionCheck>
run_permission_checks(const std::atomic<bool>* abort = nullptr);

// Microphone authorization status (platform-specific impl):
//    1  authorized
//    0  denied / restricted
//   -1  not-determined / unknown / unsupported platform
// macOS: AVCaptureDevice authorizationStatusForMediaType:audio
// (apps/web-ui/startup-checks-mac.mm).
int microphone_auth_status();

// Number of audio INPUT devices present on the system (hardware
// topology -- independent of microphone permission). Returns the count
// (>= 0), or -1 if it can't be determined / unsupported platform.
// macOS: CoreAudio HAL device enumeration (startup-checks-mac.mm).
int audio_input_device_count();

}  // namespace vpipe::webui

#endif
