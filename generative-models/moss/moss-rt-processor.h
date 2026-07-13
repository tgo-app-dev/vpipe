#ifndef VPIPE_GENAI_MOSS_MOSS_RT_PROCESSOR_H
#define VPIPE_GENAI_MOSS_MOSS_RT_PROCESSOR_H

// Host-side prompt builder for MOSS-TTS-Realtime (moss_tts_realtime). Mirrors
// the reference `MossTTSRealtimeProcessor.make_ensemble` (inferencer.py): the
// system prompt + the `<|im_start|>assistant\n` opener laid into channel 0 of a
// [seq][1+n_vq] grid (channels 1..n_vq = the audio codebook pad 1024). vpipe's
// Tokenizer does NOT treat the <|im_*|> markers as special ids, so each plain
// text fragment is tokenized SEPARATELY and the special markers are injected as
// single ids (same approach as moss_v15_build_tts_grid). The realtime STEP loop
// (prefill first-N text + per-step next-text-token feedback) lives in
// MetalMossRtModel::generate; this only builds the fixed system/assistant
// prompt prefix.

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe::genai {

class Tokenizer;

struct MossRtPromptIds {
  int im_start        = 151644;
  int im_end          = 151645;
  int ref_audio_pad   = 151654;   // channel-0 marker for a reference-audio frame
  int n_vq            = 16;
  int audio_pad       = 1024;   // per-codebook pad (ch 1..n_vq of prompt rows)
  // The reference tts base system prompt (verbatim from inferencer.py).
  std::string system_text =
      "You are a highly expressive text-to-speech (TTS) engine developed by "
      "Mosi Intelligence. \nYou possess natural language understanding, "
      "emotional modeling, and multi-style speech generation capabilities, "
      "allowing you to generate the corresponding speech based on the text "
      "given in the assistant.";
};

// Build the fixed prompt prefix grid: `<|im_start|>system\n<system_text>
// <|im_end|>\n<|im_start|>assistant\n`, one row per token, channel 0 = the
// token/marker id, channels 1..n_vq = audio_pad. The target text is streamed in
// afterwards by MetalMossRtModel::generate (prefill + per-step feedback).
std::vector<std::vector<std::int32_t>> moss_rt_build_prompt_grid(
    const Tokenizer& tok, const MossRtPromptIds& ids);

// Voice-clone variant: the same prompt, plus a `<|im_start|>context\nThe
// assistant section should be synthesized using the following voice timbre:`
// block whose reference-audio frames carry `ref_codes` ([Tref][n_vq], each
// 0..1023 -- the first n_vq codebooks of MetalMossCodec::encode) in channels
// 1..n_vq, with channel 0 = ref_audio_pad (151654). Mirrors the reference
// MossTTSRealtimeProcessor.make_ensemble(prompt_audio_tokens). Empty ref_codes
// => falls back to the plain prompt grid.
std::vector<std::vector<std::int32_t>> moss_rt_build_clone_grid(
    const Tokenizer& tok,
    const std::vector<std::vector<std::int32_t>>& ref_codes,
    const MossRtPromptIds& ids);

}  // namespace vpipe::genai

#endif
