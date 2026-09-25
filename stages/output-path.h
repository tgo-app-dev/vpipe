#ifndef VPIPE_STAGES_OUTPUT_PATH_H
#define VPIPE_STAGES_OUTPUT_PATH_H

#include <cstdint>
#include <ctime>
#include <string>
#include <string_view>

namespace vpipe::outpath {

// WHERE A SAVE STAGE'S NEXT FILE GOES, as one rule for save-image,
// save-video and save-audio. Those three had three different answers to
// the same question -- save-image formatted a printf conversion or
// appended "-%06u", save-audio appended "-%03llu" and knew no token at
// all, save-video simply overwrote its one file every launch -- and the
// third one was the bug report (issue #38): the operator had to retype
// the output name in the web-ui before every run.
//
// TOKENS IN A PATH TEMPLATE:
//
//   %%                a literal '%'.
//   %t                the local wall-clock time, "%Y%m%d-%H%M%S".
//   %t{FMT}           the same under an explicit strftime format. FMT
//                     must stay inside ONE path component: a '/' there
//                     would name a directory nothing has created, and
//                     would leave the confined prefix the stage already
//                     resolved (see Stage::attr_path), so it is refused
//                     at config time rather than failing at the open.
//   %d %i %u %x %X    the sequence number, with the usual printf flags
//                     and width ("%04d"). Every conversion in the
//                     template gets the SAME number.
//
// ANYTHING ELSE IS COPIED VERBATIM, including a stray '%'. This is a
// filename, not a format string, and the difference is not cosmetic:
// the old code handed the user's path straight to snprintf with one int
// argument, so a path holding "%s" read a pointer off the stack. There
// is no conversion here that consumes an argument we did not supply.
//
// A template that carries NO sequence conversion still gets one when it
// needs it: file 0 is written verbatim and each later file takes a
// "-NNN" suffix before the extension. `pad` is that suffix's width, and
// it is per stage only because each stage already shipped one -- 6 for
// save-image and save-video, 3 for save-audio -- and renaming the files
// somebody's scripts already glob is not this change's business.

// The default %t format. Sortable, filename-safe, no ':' (illegal on
// Windows, awkward everywhere else). SECOND resolution, which is why a
// stream of images wants a sequence number beside it or `no_overwrite`.
inline constexpr const char* kDefaultTimeFormat = "%Y%m%d-%H%M%S";

// The widest field width a conversion may ask for. A filename is not a
// report column, and an unbounded width is an invitation to allocate.
inline constexpr int kMaxWidth = 32;

// What a template carries, filled by check().
struct Tokens {
  bool sequence = false;   // at least one integer conversion
  bool time     = false;   // at least one %t
};

// Validate `tmpl`. Returns an empty string when it is usable, else the
// reason, phrased for fail_config(). `out` may be null.
std::string check(std::string_view tmpl, Tokens* out);

// Expand `tmpl` for sequence number `seq` at wall-clock `when`. `pad` is
// the fallback suffix width described above. An invalid template
// expands as best it can rather than throwing; check() is what refuses
// one, at config time, where the operator can still fix it.
std::string expand(std::string_view tmpl, std::uint64_t seq, int pad,
                   std::time_t when);

// A monotonic cursor over the sequence numbers a stage has used, plus
// the no-overwrite scan.
//
// MONOTONIC IS THE POINT. The scan advances the cursor and never rewinds
// it, so the cost of "skip what is on disk" is paid once per run rather
// than per file, and two files written in one run can never collide even
// if something deletes one of them underneath us. A deleted middle
// number IS reused -- the scan stops at the first free name -- but only
// while the cursor has not passed it yet.
class SeqPicker {
public:
  // `no_overwrite` turns on the scan; without it this is exactly the
  // numbering the three stages already had.
  void configure(bool no_overwrite) { _skip = no_overwrite; reset(); }

  // Per-launch reset. Called from reset_run_state(): the stage survives
  // a stop/relaunch, and without this the second launch would start at
  // the index the first one stopped at -- which for the default
  // (overwrite) mode means the file the operator is watching is never
  // rewritten.
  void reset() { _next = 0; }

  // The path for the next file, advancing the cursor. Returns "" when
  // no free name was found within kMaxProbe (the caller reports it);
  // that cannot happen unless `no_overwrite` is set.
  //
  // `tmpl` is passed per call rather than held, because save-audio
  // rewrites the extension from the format it actually encoded before
  // it knows the name.
  std::string take(std::string_view tmpl, int pad);

  // The next number this picker would use, without taking it.
  std::uint64_t peek() const { return _next; }

  bool skips_existing() const { return _skip; }

private:
  bool          _skip = false;
  std::uint64_t _next = 0;
};

// How far the no-overwrite scan will look before giving up. A directory
// with this many files is not a numbering problem any more.
inline constexpr std::uint64_t kMaxProbe = 1000000;

}   // namespace vpipe::outpath

#endif
