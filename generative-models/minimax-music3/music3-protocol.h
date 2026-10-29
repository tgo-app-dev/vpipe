#ifndef VPIPE_GENERATIVE_MODELS_MINIMAX_MUSIC3_MUSIC3_PROTOCOL_H
#define VPIPE_GENERATIVE_MODELS_MINIMAX_MUSIC3_MUSIC3_PROTOCOL_H

// The MiniMax-Music3 request protocol: the token ids its checkpoint
// reserves, how a caption and lyrics become the language model's prompt,
// the per-frame code picks, and the windows the flow matching runs over.
//
// A song is THREE stages (the diffusers modular pipeline's, which is the
// reference here; sglang-omni's implementation agrees with it):
//
//   1. AR       per 25 Hz frame, the Qwen3-8B LM picks the frame's
//               semantic code c0 (16384 codes, or <|audio_end|>) under
//               classifier-free guidance 1.5 -- the unconditional prompt
//               keeps the first id and the last two and has every other
//               replaced by <|audio_cfg|> -- then a 4-layer depth decoder
//               picks the residual codes c1..c7 (1024 each) under the same
//               guidance. The frame is fed back as the SUM of its eight
//               embeddings, scaled by 8^-0.5. The FIRST frame only
//               advances the state past <|audio_start|>: its codes are
//               fed back, its hidden states are not kept.
//   2. FLOW     the eight hidden states that picked each kept frame's
//               codes (32768 wide) are mixed into a 2048-wide condition at
//               44100 / 512 latents a second, and a 2.4B DiT solves 128-wide
//               latents from noise by 30 Euler steps under guidance 1.7
//               (unconditional = an all-zero condition), in 200-frame
//               windows on a 100-frame hop; each window's first 172
//               latents are pinned to the previous window's by a noise
//               blend.
//   3. DECODE   a DAC vocoder per window, the overlaps cropped. Because
//               the crop seam sits 86 latents inside the 172 both windows
//               share EXACTLY, and the vocoder sees ~11 latents either
//               side, decoding the concatenated crops is the reference's
//               waveform -- so the stage emits ONE latent per song
//               (stitched_latents) and the VAE decodes it as any other.
//
// Everything here is host-side and checkpoint-agnostic.

#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vpipe::genai {
class Tokenizer;
}

namespace vpipe::genai::music3 {

// ---- reserved ids -----------------------------------------------------
constexpr std::int32_t kImStart      = 151644;
constexpr std::int32_t kImEnd        = 151645;
constexpr std::int32_t kAudioCfg     = 151654;
constexpr std::int32_t kAudioStart   = 151669;
constexpr std::int32_t kAudioEnd     = 151670;
constexpr std::int32_t kCaptionStart = 151671;
constexpr std::int32_t kCaptionEnd   = 151672;
constexpr std::int32_t kLyricsStart  = 151673;
constexpr std::int32_t kLyricsEnd    = 151674;
constexpr std::int32_t kCodeOffset   = 151675;   // c0 code 0
constexpr int kSemanticCodes = 16384;
constexpr int kResidualCodes = 1024;
constexpr int kCodebooks     = 8;
// The c0 window of the head: <|audio_end|> first, then the 16384 codes --
// [kAudioEnd, kCodeOffset + kSemanticCodes), with the four ids between
// them never picked (masked).
constexpr std::int32_t kHeadRow0 = kAudioEnd;
constexpr int kHeadRows = kCodeOffset + kSemanticCodes - kAudioEnd;
constexpr int kSemanticBase = kCodeOffset - kAudioEnd;   // code 0's row

constexpr int kMaxPromptTokens = 5000;
constexpr int kMaxFrames       = 9000;
constexpr double kFramesPerSecond = 25.0;

// ---- the AR picks -------------------------------------------------------
constexpr float kArCfg  = 1.5f;
constexpr int   kArTopK = 50;

// ---- the flow matching ---------------------------------------------------
constexpr int   kLatentChannels = 128;
constexpr int   kConditionDim   = 2048;
constexpr int   kHop            = 512;      // samples a latent, 44.1 kHz
constexpr int   kSampleRate     = 44100;
constexpr int   kDitSteps       = 30;
constexpr float kDitCfg         = 1.7f;
constexpr int   kChunkFrames    = 200;
constexpr int   kChunkHop       = 100;
constexpr int   kOverlapLatents = 172;      // pinned to the previous window
constexpr int   kCropLeft       = 86;       // latents a later window drops
constexpr int   kCropRight      = 344 - 86; // ... an earlier one
// Latents a second, 44100 / 512.
constexpr double kLatentsPerSecond = (double)kSampleRate / kHop;

// ---- the prompt -----------------------------------------------------------
// The reference's _clean_caption: <|a b|> tags rewritten "a is b", the
// markdown forms of its input contract stripped (headings, bullets, bold,
// emphasis, rules, "- ", "• ", four-space indents), lines split as
// Python's str.splitlines splits them and blank lines collapsed. Python's
// regex semantics are emulated, Unicode whitespace included.
std::string clean_caption(std::string_view caption);
// The reference's _normalize_lyrics: on each line, consecutive LEADING
// tags are kept and the rest of the line dropped; "] " / " [" / " ^ "
// become line breaks; every [tag] is lower-cased (Unicode); "[start]\n"
// goes first.
std::string normalize_lyrics(std::string_view lyrics);
// <|im_start|><|caption_start|>caption<|caption_end|><|lyrics_start|>
// lyrics<|lyrics_end|><|im_end|><|audio_start|>, both cleaned.
std::string prompt_text(std::string_view caption, std::string_view lyrics);

// Python's str.lower() and str.isspace(), as the cleaners use them.
std::string unicode_lower(std::string_view s);
bool        py_isspace(char32_t c);

// A tokenizer.json's added tokens, as (content, id) -- what a Hugging
// Face tokenizer splits the text on before its model ever sees it.
using AddedTokens = std::vector<std::pair<std::string, std::int32_t>>;
// The "added_tokens" array of a tokenizer.json, without parsing the rest.
bool read_added_tokens(const std::string& tokenizer_json, AddedTokens* out,
                       std::string* err = nullptr);
// The text's ids as a Hugging Face tokenizer gives them: split on added
// tokens (leftmost, then longest), each run between them NFC-normalized
// (the checkpoint's normalizer) and BPE-encoded.
std::vector<std::int32_t> encode_text(const Tokenizer& tok,
                                      const AddedTokens& added,
                                      std::string_view text);
// The guidance branch: every id but the first and the last two replaced
// by <|audio_cfg|>.
std::vector<std::int32_t> unconditional_ids(
    const std::vector<std::int32_t>& ids);

// ---- windows ----------------------------------------------------------------
// Latents a window of `frames` hidden frames resamples to:
// max(1, int(frames * 44100 / 24000 * 960 / 512)), in double, in that
// order.
int latent_length(int frames);
// Each window's first frame: one window for a song of up to 200 frames,
// else range(0, frames - 100, 100).
std::vector<int> window_starts(int frames);
// torch's nearest interpolation from `in` to `out` positions (CPU
// arithmetic: a float scale in / out, floor of index * scale, clamped).
std::vector<int> nearest_indices(int in, int out);
// The window's latent range kept in the stitched song: [0, L) when it is
// the only one, [0, L - 258) the first, [86, L - 258) a middle, [86, L)
// the last.
std::pair<int, int> kept_latents(int window, int windows, int latents);
// The latents a song of `frames` hidden frames stitches to.
int stitched_latents(int frames);

// The Euler grid: N + 1 flow times, t[k] = 1 - float(linspace(1, 1/N, N)
// [k]) and t[N] = 1 (the scheduler's inverted sigmas). Step k evaluates
// at t[k] and moves by t[k + 1] - t[k].
std::vector<float> flow_times(int steps);

// ---- the picks ------------------------------------------------------------
// The reference's draws are torch.multinomial; these draw from the SAME
// distribution with a seeded generator of their own, so a song is
// reproducible for a seed but is not the reference's song for it.
class FramePicker {
public:
  explicit FramePicker(std::uint64_t seed, float cfg = kArCfg)
      : _rng(seed), _cfg(cfg) {}

  // c0 from the head's window logits ([kHeadRows], window order) of the
  // conditional and unconditional rows: guidance, restricted to the
  // conditional row's top 50, then a top-50 draw. Returns the window
  // index: 0 is <|audio_end|>, kSemanticBase + c the code c (the four
  // ids between are never picked).
  int pick_semantic(const float* cond, const float* uncond);
  // c_i (1..7) from one depth head's [1024] logits of both rows:
  // guidance, then a top-50 draw.
  int pick_residual(const float* cond, const float* uncond);

  // The guided, masked scores a pick draws from (-inf where barred), for
  // tests: the same arithmetic as the picks.
  static void semantic_scores(const float* cond, const float* uncond,
                              std::vector<float>* out, float cfg = kArCfg);
  static void residual_scores(const float* cond, const float* uncond,
                              int n, std::vector<float>* out,
                              float cfg = kArCfg);

private:
  int draw_(std::vector<float>& scores);

  std::mt19937_64 _rng;
  float _cfg = kArCfg;
  std::vector<float> _s;
  std::vector<float> _tmp;
};

}  // namespace vpipe::genai::music3

#endif
