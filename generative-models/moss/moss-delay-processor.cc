#include "generative-models/moss/moss-delay-processor.h"

#include "generative-models/tokenizer.h"

#include <string_view>

namespace vpipe::genai {

namespace {

// Encode a rendered prompt fragment, treating <|im_start|> / <|im_end|>
// as SINGLE special-token ids. Tokenizer::encode() does not recognise
// special tokens in its input (it would byte-level-BPE the literal into
// ~6 tokens), so split on the literals, BPE what lies between, and inject
// special_token_id() for each marker -- what the HF tokenizer does with
// its added tokens.
std::vector<std::int32_t>
encode_with_markers_(const Tokenizer& tok, const std::string& s)
{
  static const char* const kSpecials[] = {"<|im_start|>", "<|im_end|>"};
  std::vector<std::int32_t> ids;
  std::size_t pos = 0;
  while (pos < s.size()) {
    std::size_t best = std::string::npos;
    std::size_t best_len = 0;
    std::int32_t best_id = -1;
    for (const char* sp : kSpecials) {
      const std::size_t f = s.find(sp, pos);
      if (f != std::string::npos &&
          (best == std::string::npos || f < best)) {
        best = f;
        best_len = std::char_traits<char>::length(sp);
        best_id = tok.special_token_id(sp);
      }
    }
    const std::size_t end = (best == std::string::npos) ? s.size() : best;
    if (end > pos) {
      const std::vector<std::int32_t> seg =
          tok.encode(s.substr(pos, end - pos));
      ids.insert(ids.end(), seg.begin(), seg.end());
    }
    if (best == std::string::npos) { break; }
    if (best_id >= 0) { ids.push_back(best_id); }
    pos = best + best_len;
  }
  return ids;
}

// Everything after {reference}: the remaining fields, the text, and the
// assistant turn the generation continues from.
std::string
after_reference_(const std::string& text, const MossDelayUserFields& f)
{
  return "\n- Instruction:\n" + f.instruction +
         "\n- Tokens:\n" + f.tokens +
         "\n- Quality:\n" + f.quality +
         "\n- Sound Event:\n" + f.sound_event +
         "\n- Ambient Sound:\n" + f.ambient_sound +
         "\n- Language:\n" + f.language +
         "\n- Text:\n" + text +
         "\n</user_inst><|im_end|>\n<|im_start|>assistant\n";
}

}  // namespace

std::vector<std::vector<std::int32_t>>
moss_delay_build_grid(const Tokenizer& tok, const std::string& text,
                      const MossDelayPromptIds& ids,
                      const MossDelayUserFields& fields,
                      const std::vector<std::vector<std::int32_t>>* ref)
{
  const int n_vq  = ids.n_vq;
  const int pad   = ids.audio_pad;
  const int nchan = 1 + n_vq;
  std::vector<std::vector<std::int32_t>> grid;
  auto text_row = [&](std::int32_t id) {
    std::vector<std::int32_t> r(static_cast<std::size_t>(nchan), pad);
    r[0] = id;
    return r;
  };
  auto push_text = [&](const std::string& s) {
    for (std::int32_t id : encode_with_markers_(tok, s)) {
      grid.push_back(text_row(id));
    }
  };

  const std::string head = "<|im_start|>user\n<user_inst>\n- Reference(s):\n";
  if (ref == nullptr || ref->empty()) {
    push_text(head + "None" + after_reference_(text, fields));
    return grid;
  }

  // One speaker's reference: "[S1]:\n", then the delay-patterned block.
  push_text(head + "[S1]:\n");
  grid.push_back(text_row(ids.audio_start));
  const int T = static_cast<int>(ref->size());
  for (int r = 0; r < T + n_vq - 1; ++r) {           // apply_delay_pattern
    std::vector<std::int32_t> g(static_cast<std::size_t>(nchan), pad);
    g[0] = ids.audio_user_slot;
    for (int cb = 0; cb < n_vq; ++cb) {
      const int src = r - cb;
      if (src < 0 || src >= T) { continue; }
      const auto& frame = (*ref)[static_cast<std::size_t>(src)];
      if (static_cast<int>(frame.size()) <= cb) { continue; }
      int v = frame[static_cast<std::size_t>(cb)];
      if (v < 0) { v = 0; }
      if (v >= pad) { v = pad - 1; }
      g[static_cast<std::size_t>(1 + cb)] = v;
    }
    grid.push_back(std::move(g));
  }
  grid.push_back(text_row(ids.audio_end));
  push_text(after_reference_(text, fields));
  return grid;
}

MossDelayDedelay::MossDelayDedelay(int n_vq, int audio_pad)
  : _n_vq(n_vq), _pad(audio_pad),
    _ring(static_cast<std::size_t>(n_vq),
          std::vector<std::int32_t>(static_cast<std::size_t>(n_vq), 0))
{
}

bool
MossDelayDedelay::push(const std::vector<std::int32_t>& row,
                       std::vector<std::int32_t>& frame)
{
  auto& slot = _ring[static_cast<std::size_t>(_rows % _n_vq)];
  for (int cb = 0; cb < _n_vq; ++cb) {
    const std::size_t c = static_cast<std::size_t>(1 + cb);
    slot[static_cast<std::size_t>(cb)] = c < row.size() ? row[c] : _pad;
  }
  ++_rows;
  if (_rows < _n_vq) { return false; }
  // Frame f = rows - n_vq; its codebook cb is in row f + cb.
  const long long f = _rows - _n_vq;
  frame.assign(static_cast<std::size_t>(_n_vq), 0);
  bool all_pad = true;
  for (int cb = 0; cb < _n_vq; ++cb) {
    const auto& src = _ring[static_cast<std::size_t>((f + cb) % _n_vq)];
    int v = src[static_cast<std::size_t>(cb)];
    if (v != _pad) { all_pad = false; }
    if (v < 0 || v >= _pad) { v = _pad - 1; }
    frame[static_cast<std::size_t>(cb)] = v;
  }
  return !all_pad;
}

}  // namespace vpipe::genai
