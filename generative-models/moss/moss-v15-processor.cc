#include "generative-models/moss/moss-v15-processor.h"

#include <algorithm>

#include "generative-models/tokenizer.h"

namespace vpipe::genai {

namespace {

// One channel-0 text/control row: [id, audio_pad, audio_pad, ...] (n_vq+1 wide).
std::vector<std::int32_t>
text_row_(const MossV15PromptIds& ids, std::int32_t id)
{
  std::vector<std::int32_t> r((std::size_t)(ids.n_vq + 1),
                              (std::int32_t)ids.audio_pad);
  r[0] = id;
  return r;
}

// The prompt-field tail shared by both builders: the after-Reference(s) fields
// block, then the target text, the closing </user_inst>, the assistant turn,
// and the trailing audio_start. Matches _user_prompt_after_reference_ids +
// text + _assistant_prompt_prefix_ids + [audio_start] in the HF processor
// (each fragment tokenized SEPARATELY, add_special_tokens=False; special
// markers injected as single ids).
void
push_after_reference_(const Tokenizer& tok, const std::string& text,
                      const MossV15PromptIds& ids,
                      std::vector<std::vector<std::int32_t>>& grid)
{
  auto push_text = [&](const std::string& s) {
    for (std::int32_t id : tok.encode(s)) { grid.push_back(text_row_(ids, id)); }
  };
  // _render_user_prompt_after_reference: defaults are "None"; instruction +
  // language are configurable prompt fields.
  const std::string after_ref =
      "\n- Instruction:\n" + ids.instruction +
      "\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone"
      "\n- Ambient Sound:\n"
      "None\n- Language:\n" + ids.language + "\n- Text:\n";
  push_text(after_ref);
  push_text(text);
  push_text("\n</user_inst>");
  grid.push_back(text_row_(ids, ids.im_end));
  push_text("\n");
  grid.push_back(text_row_(ids, ids.im_start));
  push_text("assistant\n");
  grid.push_back(text_row_(ids, ids.audio_start));
}

// The user-prompt prefix shared by both builders: [im_start] + "user\n" +
// "<user_inst>\n- Reference(s):\n" (matches _user_prompt_prefix_ids).
void
push_prefix_(const Tokenizer& tok, const MossV15PromptIds& ids,
             std::vector<std::vector<std::int32_t>>& grid)
{
  auto push_text = [&](const std::string& s) {
    for (std::int32_t id : tok.encode(s)) { grid.push_back(text_row_(ids, id)); }
  };
  grid.push_back(text_row_(ids, ids.im_start));
  push_text("user\n");
  push_text("<user_inst>\n- Reference(s):\n");
}

}  // namespace

std::vector<std::vector<std::int32_t>>
moss_v15_build_tts_grid(const Tokenizer& tok, const std::string& text,
                        const MossV15PromptIds& ids)
{
  // Plain (no-reference) generation: prefix, "None", then the field tail.
  std::vector<std::vector<std::int32_t>> grid;
  push_prefix_(tok, ids, grid);
  for (std::int32_t id : tok.encode("None")) {
    grid.push_back(text_row_(ids, id));
  }
  push_after_reference_(tok, text, ids, grid);
  return grid;
}

std::vector<std::vector<std::int32_t>>
moss_v15_build_clone_grid(
    const Tokenizer& tok,
    const std::vector<std::vector<std::int32_t>>& ref_codes,
    const std::string& text, const MossV15PromptIds& ids)
{
  if (ref_codes.empty()) {           // no reference => plain TTS grid
    return moss_v15_build_tts_grid(tok, text, ids);
  }
  std::vector<std::vector<std::int32_t>> grid;
  push_prefix_(tok, ids, grid);

  // Reference USER audio block (replaces the plain builder's "None"):
  // audio_start, then one row per reference frame (channel 0 = audio_user_slot,
  // channels 1..n_vq = the frame's RVQ codes; NO delay pattern -- the v1.5
  // local transformer consumes the codes straight), then audio_end.
  grid.push_back(text_row_(ids, ids.audio_start));
  for (const auto& frame : ref_codes) {
    std::vector<std::int32_t> r((std::size_t)(ids.n_vq + 1),
                                (std::int32_t)ids.audio_pad);
    r[0] = ids.audio_user_slot;
    const int m = std::min<int>(ids.n_vq, (int)frame.size());
    for (int cb = 0; cb < m; ++cb) {
      r[(std::size_t)(1 + cb)] = frame[(std::size_t)cb];
    }
    grid.push_back(std::move(r));
  }
  grid.push_back(text_row_(ids, ids.audio_end));

  push_after_reference_(tok, text, ids, grid);
  return grid;
}

}  // namespace vpipe::genai
