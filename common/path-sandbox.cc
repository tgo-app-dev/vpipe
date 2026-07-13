#include "common/path-sandbox.h"

#include <array>
#include <cctype>
#include <string>
#include <system_error>

namespace vpipe {

namespace fs = std::filesystem;

namespace {

// Resolve a path to a comparable canonical form (symlinks + /var ->
// /private/var alias + ".." collapsed). Falls back to a lexical normal
// form when the target does not exist yet.
fs::path canon_(const fs::path& p)
{
  std::error_code ec;
  fs::path c = fs::weakly_canonical(p, ec);
  return (ec || c.empty()) ? p.lexically_normal() : c;
}

// True when `resolved` is `prefix` or lies inside it.
bool within_(const fs::path& resolved, const fs::path& prefix)
{
  const fs::path    rel = resolved.lexically_relative(prefix);
  const std::string s   = rel.generic_string();
  return !rel.empty() && s != ".." && s.rfind("../", 0) != 0;
}

}  // namespace

PathSandbox::PathSandbox(fs::path root, std::vector<fs::path> whitelist)
  : _enabled(true)
{
  // Canonicalize the root once so containment compares resolved paths
  // (resolves symlinks and the /var -> /private/var alias). Fall back to
  // a lexically-normal form when the dir does not exist yet.
  _root = canon_(root);
  // Canonicalize each granted prefix the same way. Empty entries are
  // dropped; the list is small (operator-supplied) so no de-dup needed.
  _whitelist.reserve(whitelist.size());
  for (auto& w : whitelist) {
    if (w.empty()) { continue; }
    _whitelist.push_back(canon_(w));
  }
}

fs::path
PathSandbox::confine(std::string_view user_path, bool for_write,
                     std::string* err) const
{
  if (!_enabled) {
    return fs::path(user_path);
  }
  // Whitelist pass-through (checked first, so it wins over the re-root):
  // an ABSOLUTE path that resolves inside a granted prefix is returned
  // unchanged. weakly_canonical resolves symlinks so the target must
  // genuinely stay within the prefix -- a symlink pointing out is caught.
  if (!_whitelist.empty()) {
    const fs::path up{std::string(user_path)};
    if (up.is_absolute()) {
      const fs::path resolved = canon_(up);
      for (const auto& w : _whitelist) {
        if (within_(resolved, w)) {
          if (for_write) {
            std::error_code mk;
            fs::create_directories(resolved.parent_path(), mk);
          }
          return resolved;
        }
      }
    }
  }
  // Treat every path as relative to the root: strip leading separators so
  // an absolute-looking "/etc/passwd" lands at "<root>/etc/passwd".
  std::string rel(user_path);
  while (!rel.empty() && (rel.front() == '/' || rel.front() == '\\')) {
    rel.erase(0, 1);
  }
  if (rel.empty()) {
    if (err) { *err = "empty path"; }
    return {};
  }
  // weakly_canonical resolves symlinks in the existing prefix AND
  // normalizes ".." across the whole path, so it catches both traversal
  // and symlink escapes in one shot. Fall back to a lexical normal form
  // when the target does not exist yet.
  std::error_code ec;
  const fs::path combined = _root / rel;
  fs::path resolved = fs::weakly_canonical(combined, ec);
  if (ec || resolved.empty()) {
    resolved = combined.lexically_normal();
  }
  const fs::path    relp = resolved.lexically_relative(_root);
  const std::string s    = relp.generic_string();
  if (relp.empty() || s == ".." || s.rfind("../", 0) == 0) {
    if (err) { *err = "path escapes the sandbox"; }
    return {};
  }
  if (for_write) {
    std::error_code mk;
    fs::create_directories(resolved.parent_path(), mk);
  }
  return resolved;
}

bool
is_network_url(std::string_view s)
{
  const auto pos = s.find("://");
  if (pos == std::string_view::npos || pos == 0) {
    return false;   // bare / relative path
  }
  std::string scheme(s.substr(0, pos));
  for (char& c : scheme) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  // Known network protocols ffmpeg accepts. file:// is deliberately
  // absent -- it is a local path and must be confined. Unknown schemes
  // fall through to false and are confined too (fail-closed).
  static constexpr std::array<std::string_view, 19> kNet = {
    "http",  "https", "rtsp", "rtsps", "rtmp", "rtmps", "rtp",
    "srt",   "udp",   "tcp",  "tls",   "mms",  "mmsh",  "mmst",
    "rist",  "ftp",   "ftps", "sftp",  "hls",
  };
  for (std::string_view n : kNet) {
    if (scheme == n) { return true; }
  }
  return false;
}

}  // namespace vpipe
