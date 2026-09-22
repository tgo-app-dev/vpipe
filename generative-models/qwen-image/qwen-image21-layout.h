#ifndef GENERATIVE_MODELS_QWEN_IMAGE_QWEN_IMAGE21_LAYOUT_H
#define GENERATIVE_MODELS_QWEN_IMAGE_QWEN_IMAGE21_LAYOUT_H

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace qi21 {

// The joint text/image sequence Qwen-Image-2.1's DiT runs over, as pure
// integer arithmetic: no GPU, no weights, nothing to load.
//
// It is split out for the same reason minimax-h3-layout is. The layout
// IS the model's structure here -- the transformer itself only ever sees
// a per-row tag, a rotary coordinate and a modulation row -- and every
// part of it is a silent failure when wrong: a mis-centred rotary grid,
// an image block one token long, a prefix boundary off by one. Testing
// that against the reference needs no checkpoint, so it should not
// require one.
//
// WHAT THE SEQUENCE IS. The vision-language encoder has already placed
// the condition images in its own token stream, one `<|image_pad|>` slot
// per 2x2 group of latent tokens. The pipeline then APPENDS one slot per
// 2x2 group of the target image. So the encoder-side sequence is
//
//     [ text ... <img slots> ... text ... | <target slots> ]
//
// and the DiT's joint sequence is that with every image slot expanded
// FOUR-fold and the latents substituted in:
//
//     [ text | cond-image blocks | target-image block ]
//
// Text rows keep their encoder embedding; image rows take a latent.

// One image in the sequence, in LATENT tokens. `frame` is 1 for a still;
// it is kept because the rotary axis is (frame, height, width) and the
// reference's arithmetic multiplies through it.
struct ImgBlock {
  int frame = 1;
  int grid_h = 0;
  int grid_w = 0;
  int tokens() const { return frame * grid_h * grid_w; }
};

// A contiguous run of the joint sequence that attends as a unit.
//
// Attention is BLOCK-CAUSAL: `(q >= kv) or same_image_block`. Text is
// strictly causal; every image block is internally bidirectional. That
// decomposes into exactly this -- for an image block, one dense
// non-causal dispatch whose keys are [0, end); for the text runs, one
// causal dispatch. No mask buffer anywhere.
struct Segment {
  int start = 0;        // first joint row
  int end = 0;          // one past the last
  bool is_text = false; // a text run (causal) rather than an image block
  int block_id = -1;    // -1 for text; else the image's index in `blocks`
};

// Everything the forward needs about the sequence it is about to run.
struct Layout {
  int joint_len = 0;      // rows in the joint sequence
  int text_len = 0;       // of which are text
  int image_len = 0;      // of which are image (= sum of blocks' tokens)
  int prefix_len = 0;     // rows before the TARGET block, i.e. what the
                          // cross-step KV cache holds
  int target_len = 0;     // the target block's tokens

  // Per joint row.
  std::vector<std::uint8_t> is_image;   // 1 at an image row
  std::vector<std::uint8_t> is_target;  // 1 at a TARGET-image row; this is
                                        // also the modulation-row selector,
                                        // since under causal_condition
                                        // everything else modulates from t=0
  std::vector<std::int32_t> block_id;   // -1 at text, else the image index
  // Rotary coordinates, one per axis. SIGNED: an image block's height and
  // width are centred on zero, so these go negative and the angle must be
  // formed from the signed value.
  std::vector<std::int32_t> pos_t, pos_h, pos_w;
  // Which encoder-side rows are still VALID keys. Prompts are LEFT-padded
  // by the processor and the padding must never be attended to; it is
  // kept as a QUERY so no row is fully masked. Empty means every row is
  // valid.
  std::vector<std::uint8_t> key_valid;

  std::vector<Segment> segments;        // in order, covering [0, joint_len)

  // The image rows in order, as offsets into the packed latent the caller
  // supplies. Row `image_row[i]` of the joint sequence takes latent i.
  std::vector<std::int32_t> image_row;
  // The text rows in order, paired with the ENCODER row each one takes.
  // Not simply "the i-th text row takes encoder row i": an image slot
  // occupies an encoder row too, so the encoder index runs ahead. The
  // reference gets this for free by expanding the encoder sequence and
  // then overwriting the image positions; doing it by scatter instead
  // means saying which source row each destination takes.
  std::vector<std::int32_t> text_row, text_src;
};

// Build the layout.
//
//   slot        one entry per ENCODER-side token, 1 where the encoder
//               reserved an image slot -- INCLUDING the target slots the
//               pipeline appends. Each 1 becomes four joint rows.
//   valid       one entry per encoder-side token, 0 where it is padding.
//               May be empty, meaning all valid.
//   blocks      the images, CONDITION IMAGES FIRST AND THE TARGET LAST.
//               Their token counts must sum to four times the number of
//               set entries in `slot`.
//
// Returns false with a reason in `err` rather than throwing; a caller
// that got the bookkeeping wrong should see which invariant it broke,
// not a plausible picture.
bool build_layout(const std::vector<std::uint8_t>& slot,
                  const std::vector<std::uint8_t>& valid,
                  const std::vector<ImgBlock>& blocks, Layout* out,
                  std::string* err);

// ---- the conditioning prompt ----------------------------------------
//
// Here for the same reason the layout is: it is a FACT about the model
// that needs no checkpoint to state and no GPU to check, and getting it
// wrong is silent. Qwen-Image-2.1's system prompt is its own -- neither
// "Describe the image by detailing..." (Krea-2, Mage-Flow,
// Qwen-Image-2512) nor "Describe the key features of the input image"
// (Qwen-Image-Edit, Boogu) -- and a model conditioned on the wrong one
// loads, runs and produces a picture.
//
// Unlike most families here it uses ONE system prompt for both tasks,
// keeps the trailing generation turn, and labels its references
// `<image1>`, `<image2>` as literal text rather than "Picture N: ".
const char* system_prompt();

// The system turn alone -- the span the hidden states DROP. The
// reference derives that count by tokenizing the system turn rather
// than hardcoding it, so this returns the string to tokenize rather
// than a number.
std::string system_prefix();

// The full templated prompt. `n_ref` condition images render as
// `<imageN><|vision_start|><|image_pad|><|vision_end|>` blocks, space
// separated after the first, before the user text.
std::string prompt_template(int n_ref, const std::string& text);

// The rotary angle's integer position is signed, so the table cannot be
// a plain prefix. This is the reference's centred range for one axis:
// extent 4 -> [-2,-1,0,1], extent 5 -> [-3,-2,-1,0,1]. The extra row goes
// on the NEGATIVE side, which is not the obvious half of the two.
inline int centred_lo(int extent) { return -(extent - extent / 2); }

}  // namespace qi21
}  // namespace genai
}  // namespace vpipe

#endif
