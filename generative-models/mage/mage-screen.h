#ifndef VPIPE_GENERATIVE_MODELS_MAGE_MAGE_SCREEN_H
#define VPIPE_GENERATIVE_MODELS_MAGE_MAGE_SCREEN_H

#include <string>
#include <vector>

#ifdef VPIPE_BUILD_APPLE_SILICON
#include "apple-silicon/metal-compute/metal-compute.h"
#endif

namespace vpipe {

class SessionContextIntf;

namespace genai {

class MetalQwenModel;
class Tokenizer;

// Mage-Flow's MANDATORY content-policy screen.
//
// microsoft/Mage-Flow puts the policy classifier ON the text encoder
// (`TextEncoder.screen_text` / `screen_edit`) rather than in the pipeline, so
// it runs on the exact Qwen3-VL weights that produce the diffusion
// conditioning and is not a separable, toggleable pre-pass. vpipe keeps that
// property: `diffusion-conditioner` runs this on every mage-flow prompt, off
// the encoder it already owns, with no config key to turn it off. There is no
// "screening disabled" path -- an encoder that cannot classify cannot
// generate either (see the FAIL-CLOSED note below).
//
// FAIL-CLOSED. Every failure -- generation returned nothing, the output had
// no JSON in it, the model never loaded -- yields `violates = true`. A broken
// classifier must not become a bypass, so the default-constructed verdict
// BLOCKS and only an explicit `{"violates": false, ...}` from the model
// clears a prompt.
//
// The classification itself is a short greedy generation (a JSON verdict), a
// different shape of work from the single embedding forward that produces the
// conditioning -- so this is a second pass over the same weights, not one
// fused forward. The reference makes the same trade for the same reason.
struct MageScreenVerdict {
  bool                     violates = true;   // FAIL-CLOSED default
  std::vector<std::string> categories;
  std::string              reason;
  std::string              raw;     // the classifier's raw text (diagnostics)
};

// The two policy prompts, VERBATIM from the reference
// (mage_flow/models/modules/mage_text.py). They are the policy: reformatting
// or paraphrasing them changes what gets blocked, so they live in their own
// data-only translation unit and are never re-wrapped.
extern const char* const kMageFilterSystem;       // text-to-image
extern const char* const kMageFilterEditSystem;   // image-edit (multimodal)

// Pull the classifier's verdict out of its raw output: strip a ``` fence,
// take the first BALANCED top-level JSON object (brace-counting, string- and
// escape-aware), and read violates / categories / reason from it.
//
// Returns false when no verdict could be read -- the caller must then FAIL
// CLOSED. `out` is left untouched on failure, so a default-constructed
// MageScreenVerdict already carries the blocking answer.
bool mage_parse_verdict(const std::string& text, MageScreenVerdict* out);

#ifdef VPIPE_BUILD_APPLE_SILICON

// What to classify. `n_img == 0` selects the text (t2i) policy; a non-zero
// `n_img` selects the multimodal EDIT policy, which judges the source image
// as well as the instruction -- that is the whole point of the edit gate (an
// NSFW / copyrighted-character / real-public-figure source photo is blocked
// even under an innocuous "change the background").
struct MageScreenRequest {
  std::string prompt;

  // Qwen3-VL vision tokens for the source image: f16 [n_img, hidden], the
  // tower's own output. `img_mh`/`img_mw` are its post-merger grid (for the
  // 3-axis mROPE positions), `deepstack` the per-layer features to inject at
  // LM layers 0.. (borrowed pointers; the caller keeps ownership).
  const metal_compute::SharedBuffer* vision = nullptr;
  int                                n_img  = 0;
  int                                img_mh = 0;
  int                                img_mw = 0;
  std::vector<const metal_compute::SharedBuffer*> deepstack;

  // 0 => the reference caps: 160 (text) / 192 (edit).
  int max_new_tokens = 0;
};

// Run the classifier on `lm` (which MUST have been loaded with an lm_head --
// i.e. NOT backbone_only -- since this generates). Greedy, stopping at
// <|im_end|>. Never throws; every failure path returns a BLOCKING verdict.
MageScreenVerdict
mage_screen(MetalQwenModel&           lm,
            const Tokenizer&          tok,
            const MageScreenRequest&  req,
            const SessionContextIntf* session);

#endif  // VPIPE_BUILD_APPLE_SILICON

}  // namespace genai
}  // namespace vpipe

#endif
