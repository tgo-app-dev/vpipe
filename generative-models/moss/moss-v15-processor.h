#ifndef VPIPE_GENAI_MOSS_MOSS_V15_PROCESSOR_H
#define VPIPE_GENAI_MOSS_MOSS_V15_PROCESSOR_H

// Host-side text->grid processor for MOSS-TTS-Local-v1.5. Builds the generation
// prompt EXACTLY as the HF processor's _build_generation_or_voice_clone_codes:
// special token ids inserted directly, each text fragment tokenized SEPARATELY
// then concatenated, laid into channel 0 of a [seq][1+n_vq] grid (channels
// 1..n_vq = audio_pad). Two builders: the plain text-only grid (audio_codes
// empty) and the voice-CLONE grid (reference RVQ codes spliced into the
// - Reference(s): section). The grid feeds MetalMossV15Model::generate.

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe::genai {

class Tokenizer;

struct MossV15PromptIds {
  int im_start    = 151644;
  int im_end      = 151645;
  int audio_start = 151669;
  int audio_end   = 151670;   // closes a reference audio block (voice clone)
  int audio_user_slot = 151654;  // channel-0 marker for each reference frame
  int n_vq        = 12;
  int audio_pad   = 1024;
  // Optional prompt fields (default "None"); language tags speech language.
  std::string instruction = "None";
  std::string language    = "None";
};

// Build the plain (no-reference) TTS generation grid for `text`:
// [seq][1+n_vq] int32, channel 0 = prompt token ids, channels 1..n_vq = pad.
std::vector<std::vector<std::int32_t>> moss_v15_build_tts_grid(
    const Tokenizer& tok, const std::string& text,
    const MossV15PromptIds& ids);

// Build the voice-CLONE generation grid for `text` conditioned on a reference
// clip's RVQ codes `ref_codes` (shape [Tref][n_vq], each entry 0..1023 -- the
// output of MetalMossCodecV2::encode). Mirrors the HF processor's
// _build_generation_or_voice_clone_codes (audio_codes_list non-empty): the
// reference codes are spliced into the - Reference(s): section as a single
// USER audio block -- one audio_start row, then Tref rows whose channel 0 is
// audio_user_slot and channels 1..n_vq carry ref_codes[t][cb] (NO per-codebook
// delay; the v1.5 local-transformer lays the codes straight), then one
// audio_end row -- replacing the plain builder's "None". The remaining prompt
// fields + target text + assistant turn (ending at audio_start) follow exactly
// as in the plain builder. Empty `ref_codes` => falls back to the plain grid.
std::vector<std::vector<std::int32_t>> moss_v15_build_clone_grid(
    const Tokenizer& tok,
    const std::vector<std::vector<std::int32_t>>& ref_codes,
    const std::string& text, const MossV15PromptIds& ids);

}  // namespace vpipe::genai

#endif
