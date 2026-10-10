#ifndef VPIPE_GENERATIVE_MODELS_CONTENT_SCREEN_H
#define VPIPE_GENERATIVE_MODELS_CONTENT_SCREEN_H

// A FAMILY'S CONTENT SCREEN, run by `diffusion-conditioner` on every
// prompt before it is encoded. Feature VPIPE_FEATURE_CONTENT_SCREEN
// ("content-screen/1").
//
// Some checkpoints put a policy classifier ON their text encoder: the
// same weights that produce the conditioning are asked, in a chat turn,
// whether the request is allowed, and a refusal stops the generation
// before a DiT ever runs. The classifier is a property of the model, so
// it belongs to the family that knows the model -- its policy text, the
// turn it renders, how it reads the answer. What it RUNS ON belongs to
// the host: the encoder is the conditioner's, loaded once, and a second
// copy for the screen would double the largest thing the stage holds.
//
// So the seam is drawn at ONE GREEDY CHAT. The family implements
// ContentScreen and hands it out through its model family's
// `query_extension(kExtension)`; the conditioner lends it an
// EncoderChat bound to the encoder it already owns and to the picture
// under judgement, and reads back a Verdict.
//
// WHAT THE HOST GUARANTEES, and none of it is configurable:
//
//   * A family that provides a screen is screened on EVERY prompt. No
//     config key, beat or profile entry turns it off -- a graph that
//     could omit it would be a bypass, which is what the screen exists
//     to prevent. The encoder is loaded with its output head whenever a
//     screen is present, whatever the conditioning profile says, since
//     a screen that cannot generate can only refuse.
//   * EVERY PICTURE IS JUDGED. With references wired, the screen runs
//     once per reference, each pass bound to that reference's picture,
//     and the prompt is refused if ANY pass refuses -- so a picture in
//     a second slot cannot pass unscreened.
//   * FAIL-CLOSED where the host can see a failure: a screen that throws
//     refuses. A Verdict defaults to `blocked`, so a screen that returns
//     early without clearing the prompt refuses as well.
//   * A refused prompt emits the conditioner's refusal beat (sideband
//     `content_blocked`), which generate-image and vae-decode carry
//     through as a blank picture of the requested size. The log says
//     THAT a prompt was refused, never which category tripped: naming
//     it turns the gate into an oracle to probe against. The verdict's
//     detail goes to the debug log only.
//
// WHAT THE FAMILY OWNS: the policy text, the turn it renders, the
// token cap, what an empty prompt means, and how the answer is read --
// including what a failed generation means. The host never interprets
// generated text.

#include "common/flex-data.h"

#include <string>
#include <string_view>
#include <vector>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe::genai::screen {

// The id a family's ImageModelFamily (or VideoModelFamily) answers in
// `query_extension`, returning a ContentScreen*. The pointer must stay
// valid for as long as the family is registered.
inline constexpr std::string_view kExtension = "vpipe.content-screen/1";

// One piece of a chat turn. The turn is the parts in order; the host
// tokenizes each and concatenates the ids, so a special token is never
// scanned out of surrounding text -- a policy full of punctuation and
// quoting is exactly where a marker scanner goes silently wrong.
struct ChatPart {
  enum class Kind : int {
    // A special token, by its spelling ("<|im_start|>"). A spelling the
    // tokenizer does not know fails the generation.
    Special = 0,
    // Plain text, run through the tokenizer.
    Text = 1,
    // The picture bound to this pass, as the encoder's own vision rows:
    // one image-pad position per row, filled with the tower's output and
    // its deepstack features, at the stock multimodal positions. The
    // family writes whatever opens and closes a picture in its template
    // as Special parts around it. Fails the generation when no picture
    // is bound (ScreenInput::images == 0).
    Image = 2,
  };
  Kind        kind = Kind::Text;
  std::string text;   // Special and Text; ignored for Image
};

// One greedy generation.
struct ChatRequest {
  std::vector<ChatPart> parts;
  // The special that ends the generation, by spelling; not included in
  // the text returned. Empty runs to the cap.
  std::string stop;
  // A cap on generated tokens. 0 means the host's default (256).
  int max_new_tokens = 0;
  // Scalars for anything this struct has no field for, LENT for the call
  // (common/flex-bag.h). Null today.
  const FlexData* borrowed_extra = nullptr;
};

// What the host lends a screen: greedy decoding over the conditioner's
// own encoder, with the picture under judgement already bound. Valid
// only for the duration of ContentScreen::screen.
class EncoderChat {
public:
  virtual ~EncoderChat() = default;

  // Render `req.parts`, prefill, and decode greedily until `req.stop` or
  // the cap. The decoded text (stop token excluded) goes to `*text`.
  // False on any failure, with the reason in `*why`; `*text` is then
  // unspecified. Never throws.
  virtual bool generate(const ChatRequest& req, std::string* text,
                        std::string* why) = 0;

public:
  // EXTENSION POINT (plugin ABI; docs/PLUGINS.md, "Versioning").
  virtual void* query_extension(std::string_view id) noexcept
  {
    (void)id;
    return nullptr;
  }
};

// What one pass judges.
struct ScreenInput {
  // The prompt exactly as the graph sent it (untrimmed).
  std::string prompt;
  // Pictures bound to this pass: 0 for a text-only prompt, 1 when a
  // reference rides it (an edit). Several references are several passes.
  int images = 0;
  // Which reference this pass judges, of how many. 0 of 1 when text-only.
  int pass   = 0;
  int passes = 1;
  // LENT for the call. Null today.
  const FlexData* borrowed_extra = nullptr;
};

// The answer. Defaults to REFUSED: only an explicit clearance lets a
// prompt through.
struct Verdict {
  bool                     blocked = true;
  std::vector<std::string> categories;   // debug log only
  std::string              reason;       // debug log only
  std::string              raw;          // the classifier's own text
  // Anything this struct has no field for.
  FlexData extra;
};

// Implemented by the family. Called from the conditioner's worker, one
// pass at a time; a family shared by several conditioner stages is
// called from several threads, so keep no per-call state in members.
class ContentScreen {
public:
  virtual ~ContentScreen() = default;

  virtual Verdict screen(EncoderChat& chat, const ScreenInput& in) = 0;

public:
  // EXTENSION POINT (plugin ABI; docs/PLUGINS.md, "Versioning").
  virtual void* query_extension(std::string_view id) noexcept
  {
    (void)id;
    return nullptr;
  }
};

}  // namespace vpipe::genai::screen

VPIPE_API_END

#endif
