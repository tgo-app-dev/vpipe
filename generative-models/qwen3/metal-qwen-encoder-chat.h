#ifndef VPIPE_GENERATIVE_MODELS_QWEN3_METAL_QWEN_ENCODER_CHAT_H
#define VPIPE_GENERATIVE_MODELS_QWEN3_METAL_QWEN_ENCODER_CHAT_H

// The host's EncoderChat (generative-models/content-screen.h) over a
// Qwen3-VL language model a conditioner already owns: render a turn from
// its parts, splice one bound picture's vision rows over its image-pad
// positions, prefill, and decode greedily. Host-internal -- a family
// reaches it only through the interface.

#include "generative-models/content-screen.h"

#include "apple-silicon/metal-compute/shared-buffer.h"

#include <string>
#include <vector>

namespace vpipe::genai {

class MetalQwenModel;
class Tokenizer;

// One picture, as the vision tower emitted it: f16 [rows, hidden], its
// post-merger grid (for the 3-axis positions) and the per-layer deepstack
// features to inject at LM layers 0... Pointers are BORROWED.
struct ChatPicture {
  const metal_compute::SharedBuffer*              vision = nullptr;
  int                                             rows = 0;
  int                                             grid_h = 0;
  int                                             grid_w = 0;
  std::vector<const metal_compute::SharedBuffer*> deepstack;
};

class QwenEncoderChat final : public screen::EncoderChat {
public:
  // `lm` MUST have its output head (not backbone-only). `picture` may be
  // null: a text-only pass, where an Image part fails the generation.
  QwenEncoderChat(MetalQwenModel& lm, const Tokenizer& tok,
                  const ChatPicture* picture)
    : _lm(lm), _tok(tok), _picture(picture)
  {}

  bool generate(const screen::ChatRequest& req, std::string* text,
                std::string* why) override;

  // The reason the last generation failed, or empty. The screen decides
  // what a failure MEANS; this is so the host can still say that one
  // happened, which a verdict alone does not.
  const std::string& last_error() const noexcept { return _last_error; }

private:
  MetalQwenModel&    _lm;
  const Tokenizer&   _tok;
  const ChatPicture* _picture;
  std::string        _last_error;
};

}  // namespace vpipe::genai

#endif
