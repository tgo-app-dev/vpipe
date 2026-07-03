#ifndef VPIPE_COMMON_MEDIA_LINE_H
#define VPIPE_COMMON_MEDIA_LINE_H

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Media-line marker protocol: a single line of user text that carries
// inline image/audio attachments as special text sequences, so the
// whole message still travels every text-only channel (stdin, the
// web-ui input POST, a FlexData string beat) unchanged.
//
// Two transports x two modalities:
//
//   filesystem path (typed by a local stdio user):
//     <|__vpipe_fs_im_start__|>/path/to/pic.jpg<|__vpipe_fs_im_end__|>
//     <|__vpipe_fs_au_start__|>/path/to/clip.m4a<|__vpipe_fs_au_end__|>
//
//   base64 payload (built by the web-ui from an attached/dropped file):
//     <|__vpipe_base64_im_start__|>LENGTH,DATA<|__vpipe_base64_im_end__|>
//     <|__vpipe_base64_au_start__|>LENGTH,DATA<|__vpipe_base64_au_end__|>
//   where LENGTH is the DECODED byte count in decimal and DATA is
//   standard-alphabet base64 ('=' padding optional, ASCII whitespace
//   tolerated). The length lets a parser sanity-check the payload and
//   a reader skip it without decoding.
//
// Producers: UiDelegateIntf::getmedialine implementations (the stdio
// user types fs markers by hand; the web-ui client emits base64
// markers). Consumer: text-chat parses the markers BEFORE the LLM
// tokenizer ever sees the line, decodes the media (FFmpeg) and splices
// encoder embeddings at the marker positions. to_display() renders a
// marker-bearing line for transcripts/echoes with each attachment
// compressed to a short glyph so megabytes of base64 never hit a
// console.
namespace vpipe::media_line {

enum class Modality : std::uint8_t { Image, Audio };

// One parsed run of a media line, in original line order.
struct Segment {
  enum class Kind : std::uint8_t {
    Text,    // a plain-text run (`text`)
    FsPath,  // a filesystem-path attachment (`text` = the path)
    Bytes,   // an inline base64 attachment (`bytes` = decoded payload)
  };
  Kind                      kind     = Kind::Text;
  Modality                  modality = Modality::Image;  // FsPath/Bytes
  std::string               text;
  std::vector<std::uint8_t> bytes;
};

// Marker constants. The "<|__vpipe_" prefix is shared by all of them
// (and is what has_media_marker / parse scan for).
inline constexpr std::string_view kMarkerPrefix = "<|__vpipe_";
inline constexpr std::string_view kFsImageStart = "<|__vpipe_fs_im_start__|>";
inline constexpr std::string_view kFsImageEnd   = "<|__vpipe_fs_im_end__|>";
inline constexpr std::string_view kFsAudioStart = "<|__vpipe_fs_au_start__|>";
inline constexpr std::string_view kFsAudioEnd   = "<|__vpipe_fs_au_end__|>";
inline constexpr std::string_view kB64ImageStart =
    "<|__vpipe_base64_im_start__|>";
inline constexpr std::string_view kB64ImageEnd =
    "<|__vpipe_base64_im_end__|>";
inline constexpr std::string_view kB64AudioStart =
    "<|__vpipe_base64_au_start__|>";
inline constexpr std::string_view kB64AudioEnd =
    "<|__vpipe_base64_au_end__|>";

// Unified THINKING markers (model OUTPUT, not attachments). The
// generative-model detokenizers rewrite each family's reasoning
// begin/end tokens (Qwen3 `<think>`/`</think>`, Gemma-4
// `<|channel>`/`<channel|>`) to these vpipe-internal strings while
// streaming, so every front end responds to ONE marker pair instead
// of per-family syntax: the web-ui folds/reveals the enclosed text
// behind its Thinking toggle; the stdio delegate renders them as
// readable `⟦think⟧`/`⟦/think⟧` tags. A stream may BEGIN inside a
// thinking block (Qwen's opening `<think>` lives in the prompt, so it
// is never streamed) -- chat stages consult
// ChatTemplate::assistant_prompt_opens_thinking() and emit kThinkStart
// themselves; an unterminated block runs to end-of-message.
inline constexpr std::string_view kThinkStart =
    "<|__vpipe_think_start__|>";
inline constexpr std::string_view kThinkEnd = "<|__vpipe_think_end__|>";

// Replace the thinking markers with plain readable tags
// (`⟦think⟧` / `⟦/think⟧`) for text-only surfaces (stdio).
std::string render_think_markers_plain(std::string_view text);

// Standard-alphabet base64. decode tolerates '=' padding and ASCII
// whitespace; returns nullopt on any other non-alphabet byte.
std::string base64_encode(std::span<const std::uint8_t> bytes);
std::optional<std::vector<std::uint8_t>>
base64_decode(std::string_view text);

// Fast check: does `line` contain at least one well-known start marker?
bool has_media_marker(std::string_view line);

// Split `line` into ordered text runs and media segments. Malformed
// markers (unterminated, unknown "<|__vpipe_..." sequence, bad base64,
// LENGTH mismatch, empty path) never abort the parse: a diagnostic is
// appended to *errors (when non-null) and the malformed attachment is
// DROPPED (its bytes are not leaked into the text), except an unknown
// marker prefix which is kept literally as text. A line with no
// markers parses to a single Text segment.
std::vector<Segment>
parse(std::string_view line, std::vector<std::string>* errors = nullptr);

// Build one marker string (the exact wire form parse() accepts).
std::string make_fs_marker(Modality m, std::string_view path);
std::string
make_base64_marker(Modality m, std::span<const std::uint8_t> bytes);

// Transcript-safe rendering: text runs verbatim, each attachment
// compressed to a short glyph -- fs markers to "⟦<glyph> <path>⟧",
// base64 markers to "⟦<glyph> <N> bytes⟧" (image glyph 🖼, audio 🔊).
// Malformed markers render as "⟦<glyph>?⟧". Never returns base64 data.
std::string to_display(std::string_view line);

}

#endif
