#include "generative-models/moss/moss-rt-processor.h"

#include <algorithm>

#include "generative-models/tokenizer.h"

namespace vpipe::genai {

namespace {

// Append `<|im_start|>system\n<system_text><|im_end|>\n` to `grid`.
void push_system_(const Tokenizer& tok, const MossRtPromptIds& ids,
                  std::vector<std::vector<std::int32_t>>& grid)
{
  auto row = [&](std::int32_t id) {
    std::vector<std::int32_t> r((std::size_t)(ids.n_vq + 1),
                                (std::int32_t)ids.audio_pad);
    r[0] = id;
    grid.push_back(std::move(r));
  };
  row(ids.im_start);
  for (std::int32_t id : tok.encode("system\n" + ids.system_text)) { row(id); }
  row(ids.im_end);
  for (std::int32_t id : tok.encode("\n")) { row(id); }
}

// Append `<|im_start|>assistant\n`.
void push_assistant_(const Tokenizer& tok, const MossRtPromptIds& ids,
                     std::vector<std::vector<std::int32_t>>& grid)
{
  auto row = [&](std::int32_t id) {
    std::vector<std::int32_t> r((std::size_t)(ids.n_vq + 1),
                                (std::int32_t)ids.audio_pad);
    r[0] = id;
    grid.push_back(std::move(r));
  };
  row(ids.im_start);
  for (std::int32_t id : tok.encode("assistant\n")) { row(id); }
}

}  // namespace

std::vector<std::vector<std::int32_t>>
moss_rt_build_prompt_grid(const Tokenizer& tok, const MossRtPromptIds& ids)
{
  std::vector<std::vector<std::int32_t>> grid;
  push_system_(tok, ids, grid);
  push_assistant_(tok, ids, grid);
  return grid;
}

std::vector<std::vector<std::int32_t>>
moss_rt_build_clone_grid(
    const Tokenizer& tok,
    const std::vector<std::vector<std::int32_t>>& ref_codes,
    const MossRtPromptIds& ids)
{
  if (ref_codes.empty()) { return moss_rt_build_prompt_grid(tok, ids); }
  std::vector<std::vector<std::int32_t>> grid;
  push_system_(tok, ids, grid);
  // Voice-clone context block: <|im_start|>context\n...timbre:{ref frames}
  // <|im_end|>\n. Each ref frame is a row with channel 0 = ref_audio_pad and
  // channels 1..n_vq = the reference codebooks.
  auto row = [&](std::int32_t id) {
    std::vector<std::int32_t> r((std::size_t)(ids.n_vq + 1),
                                (std::int32_t)ids.audio_pad);
    r[0] = id;
    grid.push_back(std::move(r));
  };
  row(ids.im_start);
  for (std::int32_t id : tok.encode(
           "context\nThe assistant section should be synthesized using the "
           "following voice timbre:")) {
    row(id);
  }
  for (const auto& frame : ref_codes) {
    std::vector<std::int32_t> r((std::size_t)(ids.n_vq + 1),
                                (std::int32_t)ids.audio_pad);
    r[0] = ids.ref_audio_pad;
    const int m = std::min<int>(ids.n_vq, (int)frame.size());
    for (int cb = 0; cb < m; ++cb) {
      r[(std::size_t)(1 + cb)] = frame[(std::size_t)cb];
    }
    grid.push_back(std::move(r));
  }
  row(ids.im_end);
  for (std::int32_t id : tok.encode("\n")) { row(id); }
  push_assistant_(tok, ids, grid);
  return grid;
}

}  // namespace vpipe::genai
