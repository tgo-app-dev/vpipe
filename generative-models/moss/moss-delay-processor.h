#ifndef VPIPE_GENAI_MOSS_MOSS_DELAY_PROCESSOR_H
#define VPIPE_GENAI_MOSS_MOSS_DELAY_PROCESSOR_H

// Host-side text->grid processor for the MOSS-TTS DELAY-PATTERN models
// (`MossTTSDelay`: MOSS-TTS 1.0 8B and MOSS-TTS-v1.5). Builds the
// generation prompt exactly as the HF processor does for one user turn:
//
//   <|im_start|>user\n<user_inst>\n- Reference(s):\n{reference}\n
//   - Instruction:\n{instruction}\n- Tokens:\n{tokens}\n
//   - Quality:\n{quality}\n- Sound Event:\n{sound_event}\n
//   - Ambient Sound:\n{ambient_sound}\n- Language:\n{language}\n
//   - Text:\n{text}\n</user_inst><|im_end|>\n<|im_start|>assistant\n
//
// laid into channel 0 of a [seq][1+n_vq] grid (channels 1..n_vq =
// audio_pad). A voice-clone reference replaces {reference} with
// "[S1]:\n" and an audio block: one audio_start row, then T + n_vq - 1
// audio_user_slot rows carrying the reference codes in the DELAY pattern
// (row r, codebook cb holds ref[r - cb][cb]), then one audio_end row.
//
// The text is spliced in as given: v1.5's normalizer
// (moss_tts_normalize_text) is the caller's to apply first, because 1.0's
// processor does not run it.

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe::genai {

class Tokenizer;

// Grid geometry + the control ids the prompt carries (config.json).
struct MossDelayPromptIds {
  int im_start        = 151644;
  int im_end          = 151645;
  int audio_start     = 151652;
  int audio_end       = 151653;
  int audio_user_slot = 151654;
  int n_vq            = 32;
  int audio_pad       = 1024;
};

// The <user_inst> fields. The reference renders an unset field as
// str(None), so "None" is the default for every one of them.
struct MossDelayUserFields {
  std::string instruction   = "None";
  std::string tokens        = "None";   // expected audio tokens, 12.5/s
  std::string quality       = "None";
  std::string sound_event   = "None";
  std::string ambient_sound = "None";
  std::string language      = "None";   // e.g. "English", "French"
};

// Build the generation grid for `text`. `ref` (when non-null and
// non-empty) is a voice-clone reference, RVQ codes [T][>= n_vq] in
// 0..audio_pad-1 (MetalMossCodec::encode); values outside are clamped.
std::vector<std::vector<std::int32_t>> moss_delay_build_grid(
    const Tokenizer& tok, const std::string& text,
    const MossDelayPromptIds& ids, const MossDelayUserFields& fields,
    const std::vector<std::vector<std::int32_t>>* ref = nullptr);

// Rolling de-delay of generated rows into audio frames. Frame f's
// codebook cb sits in row f + cb, so frame f completes when row
// f + n_vq - 1 arrives: feeding every row of a generation yields exactly
// the reference's apply_de_delay_pattern, minus its all-pad frames. The
// trailing n_vq - 1 rows complete nothing (their frames would read past
// the end), which is also what a batch de-delay drops.
//
// Pad / out-of-range codes inside a kept frame are clamped to
// audio_pad - 1, which keeps the codec in range.
class MossDelayDedelay {
public:
  MossDelayDedelay(int n_vq, int audio_pad);

  // Feed one generated row [1 + n_vq] (channel 0 = text). Returns true
  // and fills `frame` [n_vq] when a non-all-pad frame completes.
  bool push(const std::vector<std::int32_t>& row,
            std::vector<std::int32_t>& frame);

  void reset() { _rows = 0; }

private:
  int _n_vq;
  int _pad;
  long long _rows = 0;
  // Audio channels of the last n_vq rows, row r at slot r % n_vq.
  std::vector<std::vector<std::int32_t>> _ring;
};

}  // namespace vpipe::genai

#endif
