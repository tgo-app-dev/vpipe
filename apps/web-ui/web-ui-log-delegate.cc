#include "apps/web-ui/web-ui-log-delegate.h"
#include "common/vpipe-format.h"

#include <algorithm>
#include <cctype>
#include <utility>

using namespace std;

namespace vpipe::webui {

namespace {

// Lowercase short name for a level ("error".."debug", "always"), used
// as the per-line CSS class the browser colors on.
string
level_name_(vpipe::LogLevel level)
{
  string s = vpipe::to_cstr(level);   // "ERROR", "WARN", ...
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

}  // namespace

WebUiLogDelegate::WebUiLogDelegate(vpipe::LogLevel threshold)
  : _threshold(threshold)
{
}

void
WebUiLogDelegate::trim_()
{
  // Steady state: each push grows by one and this pops one, so the ring
  // stays at the cap. A cap reduction's bulk drain is handled in
  // set_max_log (these are short strings, so the inline loop there is
  // cheap and bounded by kMaxMaxLog).
  if (_log.size() > _max_log) {
    _log.pop_front();
  }
}

void
WebUiLogDelegate::log(vpipe::LogLevel level, const vpipe::VpipeFormat& f)
{
  // Filter at capture so a later threshold change affects only future
  // messages. Always is a sentinel: emit regardless of the threshold.
  const vpipe::LogLevel th = _threshold.load(std::memory_order_relaxed);
  if (level != vpipe::LogLevel::Always && level > th) {
    return;
  }
  string text;
  try {
    text = f();
  } catch (...) {
    text = "<formatter threw>";
  }
  lock_guard<mutex> lk(_mu);
  _log.push_back(LogLine{ _next_seq++, level_name_(level),
                          std::move(text) });
  trim_();
}

vector<WebUiLogDelegate::LogLine>
WebUiLogDelegate::console_since(uint64_t since, uint64_t* latest_seq,
                               size_t limit) const
{
  lock_guard<mutex> lk(_mu);
  vector<LogLine> out;
  if (limit > 0) {
    out.reserve(std::min(limit, _log.size()));
  }
  bool capped = false;
  for (const auto& l : _log) {
    if (l.seq > since) {
      if (limit > 0 && out.size() >= limit) {
        capped = true;
        break;
      }
      out.push_back(l);
    }
  }
  if (latest_seq) {
    // When capped, advance only as far as we drained so the next poll
    // resumes exactly here; otherwise expose the true tip.
    *latest_seq = (capped && !out.empty()) ? out.back().seq
                                           : (_next_seq - 1);
  }
  return out;
}

void
WebUiLogDelegate::clear_console()
{
  lock_guard<mutex> lk(_mu);
  _log.clear();
}

size_t
WebUiLogDelegate::max_log() const
{
  lock_guard<mutex> lk(_mu);
  return _max_log;
}

void
WebUiLogDelegate::set_max_log(size_t n)
{
  if (n < kMinMaxLog) { n = kMinMaxLog; }
  if (n > kMaxMaxLog) { n = kMaxMaxLog; }
  lock_guard<mutex> lk(_mu);
  _max_log = n;
  while (_log.size() > _max_log) {
    _log.pop_front();
  }
}

}
