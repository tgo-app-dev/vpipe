#ifndef VPIPE_GENERATIVE_MODELS_YUE2_YUE2_PROTOCOL_H
#define VPIPE_GENERATIVE_MODELS_YUE2_YUE2_PROTOCOL_H

// The YuE2 request protocol ("yue2-native-v1"): the token ids the
// checkpoint reserves, how a request becomes the AR prefix, and the
// per-step token pick of each of its two decode phases.
//
// A song is THREE passes over one checkpoint:
//
//   1. ABC   the AR weights write a score in ABC notation (melody and,
//            for cot=full, chords) after the request text. Ordinary text
//            vocabulary only, ending at ABC_END. Skipped for cot=off and
//            whenever the caller supplies a score.
//   2. SEM   the same weights continue with one semantic codec token per
//            25 Hz audio frame (32768 codes at kCodecOffset), ending at
//            MUSIC_END. Its length IS the song's length.
//   3. NAR   a second set of layer weights turns those tokens into 64-wide
//            VAE latents by flow matching (MetalYue2Model::synthesize).
//
// Everything here is host-side and checkpoint-agnostic: the numbers are
// the protocol's, and the reference refuses a config that changes them.

#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vpipe::genai {
class Tokenizer;
}

namespace vpipe::genai::yue2 {

// ---- reserved ids -----------------------------------------------------
constexpr std::int32_t kEod         = 151643;   // <|endoftext|>, also BOS
constexpr std::int32_t kAbcStart    = 151847;
constexpr std::int32_t kAbcEnd      = 151848;
constexpr std::int32_t kMusicStart  = 151851;
constexpr std::int32_t kMusicEnd    = 151852;
constexpr std::int32_t kCodecOffset = 151853;
constexpr std::int32_t kCodecSize   = 32768;
constexpr std::int32_t kLatentStart = 184621;
constexpr std::int32_t kLatentEnd   = 184622;
constexpr std::int32_t kVocab       = 184704;
constexpr int          kContext     = 24576;
constexpr int          kLatentDim   = 64;
// 48 kHz over the VAE's 1920-sample hop: one semantic token, one latent
// frame, 40 ms of audio.
constexpr int          kLatentsPerSecond = 25;

// The tiktoken specials that follow the 151643 ordinary ranks, in id
// order: <|endoftext|> is kEod and <abc> / </abc> are kAbcStart / kAbcEnd.
// The ids past them (MUSIC_*, the codec, LATENT_*) exist only in the
// model's vocabulary, never in the text tokenizer's.
std::vector<std::string> tiktoken_specials();

// ---- the request ------------------------------------------------------
// Chain-of-thought planning: what the ABC pass writes before the music.
enum class Cot { kOff, kMelody, kFull };

bool        parse_cot(std::string_view s, Cot* out);
const char* cot_name(Cot c);
// The native instruction each mode was trained under.
std::string_view instruction(Cot c);

// One phase's sampling. The two defaults are the release protocol's.
struct Sampling {
  float temperature        = 1.0f;
  float top_p              = 0.95f;
  int   top_k              = 100;
  float repetition_penalty = 1.2f;
  int   penalty_window     = 50;    // the last N kept tokens, 1..100
  int   min_tokens         = 200;   // the end id is barred before this
  int   max_tokens         = 9000;

  static Sampling abc_default();       // .7 / .9 / 30 / 1.005 / 100 / 32 / 4096
  static Sampling semantic_default();  // 1 / .95 / 100 / 1.2 / 50 / 200 / 9000
  bool valid(std::string* err) const;
};

struct Request {
  std::string   style;          // the [Tags] block: genre, instruments, ...
  std::string   lyrics;         // sectioned lyrics ([Verse], [Chorus], ...)
  std::string   abc;            // a supplied score; empty => the model plans
  Cot           cot  = Cot::kFull;
  std::uint64_t seed = 831001;
  float         cfg_scale = -1.0f;   // < 0 => the protocol default

  // Text guidance of the SEMANTIC phase: 1.0 (none) for melody / full,
  // 1.01 for off, unless set. The ABC phase is never guided.
  float guidance() const;
  // The request text the prefix encodes, NFC-normalized as the
  // reference's tokenizer normalizes it.
  std::string text() const;
  bool valid(std::string* err) const;
};

// Unicode NFC. The reference normalizes every string it tokenizes; text
// typed on a Mac usually is NFC already, but a decomposed accent would
// otherwise tokenize as two pieces.
std::string nfc(std::string_view s);

// The AR prefix: [EOD] + text, then for cot=off [ABC_START, ABC_END,
// MUSIC_START]; otherwise [ABC_START] alone (the model writes the score)
// or, given `abc_ids`, [ABC_START] + abc + [ABC_END, MUSIC_START].
std::vector<std::int32_t> token_prefix(
    const Request& r, const Tokenizer& tok,
    const std::vector<std::int32_t>* abc_ids);
// The CFG branch: the bare instruction and, for a planned mode, the SAME
// score ids as the positive branch.
std::vector<std::int32_t> negative_prefix(
    const Request& r, const Tokenizer& tok,
    const std::vector<std::int32_t>* abc_ids);

// The original context chunks the flow matching runs over: [a, b) codec
// frame ranges, each short enough that prefix + frames + both latent
// borders fit the context twice over.
std::vector<std::pair<int, int>> chunk_ranges(int frames, int prefix_tokens,
                                              int context = kContext);

// ---- arithmetic the reference does in bf16 ------------------------------
float bf16_round(float x);
// Classifier-free guidance as the reference spells it on bf16 tensors:
// u + s * (c - u), rounding after every operation.
void cfg_combine(const float* cond, const float* uncond, float scale,
                 float* out, int n);

// ---- the per-step pick ------------------------------------------------
enum class Phase { kAbc, kSemantic };

// The allowed id window of a phase, as [lo, hi) plus the end id: the
// ordinary text ids and ABC_END for the score, the codec and MUSIC_END
// for the music. A head computed over [row0, row0 + rows) needs no more.
struct Window {
  std::int32_t row0 = 0;
  std::int32_t rows = 0;
  std::int32_t end  = 0;
};
Window phase_window(Phase p);

// Applies the reference's distribution() to one step's logits and picks:
// vocabulary mask, the end id barred before min_tokens, the WINDOWED
// repetition penalty over the last `penalty_window` kept ids, then argmax
// (temperature 0) or temperature -> top-k -> top-p -> a draw.
//
// `legacy` is cot=off's historical arithmetic: the penalty runs on bf16
// scores and top-p always keeps three. The draw uses this class's own
// generator, so a sampled run is reproducible for a seed but is NOT the
// reference's song for that seed -- greedy is what is token-exact.
class TokenPicker {
public:
  TokenPicker(Phase p, const Sampling& s, std::uint64_t seed, bool legacy);

  // `logits` holds vocabulary ids [w.row0, w.row0 + w.rows) for this
  // phase's window. Returns the picked id; the caller passes it back to
  // accept() unless it is the end id.
  std::int32_t pick(const float* logits, int step);
  void accept(std::int32_t id) { _history.push_back(id); }

  const Window& window() const { return _w; }
  const std::vector<std::int32_t>& history() const { return _history; }

private:
  Phase                     _phase;
  Sampling                  _s;
  bool                      _legacy;
  Window                    _w;
  std::mt19937_64           _rng;
  std::vector<std::int32_t> _history;
  std::vector<float>        _scores;   // reused per step
  std::vector<std::pair<float, std::int32_t>> _cand;
};

// ---- noise ------------------------------------------------------------
// torch.randn(n, generator=torch.Generator("cpu").manual_seed(seed)) for a
// float32 tensor of n >= 16 elements: mt19937 uniforms, then the
// sixteen-wide Box-Muller of ATen's normal_fill (a tail that is not a
// multiple of 16 redraws the last 16). Bit-identical to the reference, so
// the flow matching starts from the same noise for the same seed.
std::vector<float> torch_cpu_randn(std::uint64_t seed, std::size_t n);

}  // namespace vpipe::genai::yue2

#endif
