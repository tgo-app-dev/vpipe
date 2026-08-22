#ifndef VPIPE_FORMAT_H
#define VPIPE_FORMAT_H

#include <cmath>
#include <format>
#include <functional>
#include <string>
#include <string_view>

namespace vpipe {

struct VpipeFormat {
  typedef std::function<std::string()> formatter_t;
  formatter_t _f;
  VpipeFormat(formatter_t f) : _f(std::move(f)) {};
  std::string operator()() const { return _f(); };
  VpipeFormat(VpipeFormat&&) = default;
  VpipeFormat(const VpipeFormat&) = default;
};

template <typename... Args>
VpipeFormat
fmt(std::string_view f, Args&&... args) {
  return VpipeFormat([=]()
                     { return std::vformat(f, std::make_format_args(args...));
                     });
}

// A duration a PERSON reads, from a second count that may be anything
// between one frame and one season. A pipeline runs for as long as it is
// left running, so the same figure has to serve "42.3 s" and "2 mo 14 d"
// without the caller choosing a scale it cannot know in advance.
//
// TWO UNITS, never three. The second one is the precision anyone acts
// on, and a third is noise at every scale -- "2 mo 14 d 6 h" answers no
// question that "2 mo 14 d" does not, while making the line harder to
// scan.
//
// The long units are FIXED, not calendar: a month is 30 days and a year
// is 12 of those months. A duration has no calendar to vary over -- a
// run that spans February is not shorter than one that spans March --
// and the three figures then stay consistent with each other, so
// "1 y 0 mo" is exactly 360 days rather than an amount that depends on
// when it started. Said out loud because a reader otherwise assumes
// whichever convention they carry.
//
// Sub-second durations are milliseconds and seconds carry one decimal,
// because that is the range where a ratio is the point. Anything not
// greater than zero -- including a NaN from an unstarted clock -- is
// "0 ms" rather than a negative age.
inline std::string
human_duration(double seconds)
{
  if (!(seconds > 0.0)) { return "0 ms"; }
  if (seconds < 1.0) {
    const double ms = seconds * 1000.0;
    // Below ten milliseconds an integer rounds a real span to "0 ms",
    // which reads as "it did not run" where the fact is "it was quick".
    return ms < 10.0 ? std::format("{:.2f} ms", ms)
                     : std::format("{} ms", (long long)std::llround(ms));
  }
  // Rounded ONCE, here, so the branch and the arithmetic agree: 59.96 s
  // becomes 60 and reads as "1 m 00 s", where deciding on the unrounded
  // value would print "60.0 s".
  const long long t = std::llround(seconds);
  constexpr long long kMin   = 60;
  constexpr long long kHour  = 60 * kMin;
  constexpr long long kDay   = 24 * kHour;
  constexpr long long kMonth = 30 * kDay;
  constexpr long long kYear  = 12 * kMonth;
  if (t < kMin)   { return std::format("{:.1f} s", seconds); }
  if (t < kHour)  { return std::format("{} m {:02} s", t / kMin, t % kMin); }
  if (t < kDay)   {
    return std::format("{} h {:02} m", t / kHour, (t % kHour) / kMin);
  }
  if (t < kMonth) {
    return std::format("{} d {} h", t / kDay, (t % kDay) / kHour);
  }
  if (t < kYear)  {
    return std::format("{} mo {} d", t / kMonth, (t % kMonth) / kDay);
  }
  return std::format("{} y {} mo", t / kYear, (t % kYear) / kMonth);
}

}

#endif

