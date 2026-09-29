#ifndef VPIPE_STAGES_GENERATION_INPUT_H
#define VPIPE_STAGES_GENERATION_INPUT_H

// How the inputs of a GENERATING stage pair up when a run makes more
// than one output -- one per GENERATION. generate-image and the
// diffusion-conditioner in front of it follow the same rule, so a
// conditioning and the latent it is denoised against always come from
// the same beats:
//
//   * ONE BEAT, THEN THE END OF STREAM: that beat is BROADCAST, the input
//     of every generation of the run. A prompt, a scheduler, one source
//     picture for many prompts.
//   * MORE THAN ONE: CONSUMED, one beat per generation, in order. A
//     folder of pictures, a prompt per trigger.
//   * THE RUN ENDS when a consumed input ends, or when a generation
//     would have no new beat on any input -- which is what makes a run of
//     broadcasts exactly one generation long. Two consumed inputs of
//     different lengths therefore make as many generations as the
//     SHORTER one: an input that runs out is not stretched by repeating
//     its last beat, because a picture paired with the wrong prompt is a
//     wrong answer that nothing downstream can detect.
//
// The end of stream is the only thing that tells the first two apart,
// so the decision is taken at the SECOND generation, whose read returns
// either a beat or the end. A producer that sends one beat and stays
// open is consumed, not broadcast: the next generation waits for its
// next beat. `Mode::kLatch` is the answer for such a source -- it holds
// the first beat whatever follows.
//
// The helper only keeps the count and makes the call. The READ is the
// stage's, because it is a co_await and so has to be spelled in the
// stage's own coroutine:
//
//   GenerationRound round;
//   if (ctx.iport_connected(p) && _in.wants()) {
//     auto b = co_await ctx.read(p);
//     const auto took = _in.took(b != nullptr);
//     if (round.note(took)) { ctx.signal_done(); co_return; }
//     if (took == GenerationInput::Took::kFresh) { ...use b... }
//   }
//   ...every other input...
//   if (round.idle()) { ctx.signal_done(); co_return; }

#include <chrono>
#include <cstdint>
#include <string_view>

namespace vpipe {

class GenerationInput {
public:
  enum class Mode : std::uint8_t {
    kAuto,    // broadcast one beat, consume several
    kLatch,   // the first beat, for the whole run
    kFresh,   // a new beat EVERY generation: its end ends the run
  };
  enum class Took : std::uint8_t {
    kFresh,   // a new beat: use it
    kHold,    // the stream ended after one beat: keep using it
    kNone,    // the stream ended before sending anything
    kEnded,   // a consumed stream ended: the run is over
  };

  explicit GenerationInput(Mode m = Mode::kAuto) noexcept : _mode(m) {}

  void set_mode(Mode m) noexcept { _mode = m; }
  Mode mode() const noexcept { return _mode; }

  // Whether this generation reads the port at all. False once the input
  // is broadcasting (or has nothing), and after a latched first beat.
  bool
  wants() const noexcept
  {
    if (_eos) { return false; }
    return !(_mode == Mode::kLatch && _beats > 0);
  }

  // What the read returned -- a beat, or the end of stream -- and so
  // what this generation does with the input.
  Took
  took(bool beat) noexcept
  {
    if (beat) {
      ++_beats;
      return Took::kFresh;
    }
    _eos = true;
    if (_beats == 0) { return Took::kNone; }
    if (_mode == Mode::kAuto && _beats == 1) { return Took::kHold; }
    return Took::kEnded;
  }

  int beats() const noexcept { return _beats; }

  // A new run: the stage survives a relaunch, its producers re-emit.
  void
  reset() noexcept
  {
    _beats = 0;
    _eos   = false;
  }

private:
  Mode _mode;
  int  _beats = 0;
  bool _eos   = false;
};

// One generation's worth of reads.
struct GenerationRound {
  bool fresh = false;   // some input brought a new beat
  bool ended = false;   // a consumed input ran out

  // Fold one input's answer in. True when the run is over and the stage
  // should stop reading and end.
  bool
  note(GenerationInput::Took t) noexcept
  {
    if (t == GenerationInput::Took::kFresh) { fresh = true; }
    if (t == GenerationInput::Took::kEnded) { ended = true; }
    return ended;
  }

  // Nothing new arrived on any input: everything is broadcasting, so
  // this generation would repeat the last one.
  bool idle() const noexcept { return !fresh; }
};

// ---- the seed of each generation: `seed_sequence` ------------------------
//
// generate-image and generate-video take the same key with the same
// meaning, so the words and the arithmetic live here once.
enum class SeedSequence : std::uint8_t {
  kIncrement,   // seed, seed+1, seed+2, ... by generation number
  kRandomize,   // a new seed per generation, off the clock
  kKeep,        // seed, every time
};

// The config key's choices, in the order the editor offers them.
inline constexpr const char* kSeedSequenceChoices =
    "increment,randomize,keep";

// The config spelling; false (and `out` untouched) for anything else, so
// the stage can say what it did instead.
inline bool
parse_seed_sequence(std::string_view s, SeedSequence* out) noexcept
{
  if (s == "increment") { *out = SeedSequence::kIncrement; return true; }
  if (s == "randomize") { *out = SeedSequence::kRandomize; return true; }
  if (s == "keep")      { *out = SeedSequence::kKeep;      return true; }
  return false;
}

// The seed of generation `n` (from 0) of a run whose `seed` is `base`.
// The number is the generation's position in the run, counted before a
// refusal or a dropped beat can end it, so generation n gets the same
// seed however the ones before it went.
inline std::uint64_t
generation_seed(SeedSequence q, std::uint64_t base, std::uint64_t n) noexcept
{
  switch (q) {
  case SeedSequence::kKeep:
    return base;
  case SeedSequence::kRandomize: {
    // THE CLOCK, mixed (splitmix64) so two generations a nanosecond apart
    // differ in every bit, with the generation number folded in for a
    // clock too coarse to tell them apart. Kept under 2^53: the seed is
    // logged so it can be typed back into `seed`, and the web editor
    // holds a number as a double -- one above that would come back as a
    // DIFFERENT seed, silently.
    std::uint64_t z = (std::uint64_t)std::chrono::system_clock::now()
                          .time_since_epoch().count() +
                      n * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return z & ((1ull << 53) - 1);
  }
  case SeedSequence::kIncrement:
  default:
    return base + n;
  }
}

}  // namespace vpipe

#endif
