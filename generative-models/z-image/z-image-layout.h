#ifndef GENERATIVE_MODELS_Z_IMAGE_Z_IMAGE_LAYOUT_H
#define GENERATIVE_MODELS_Z_IMAGE_Z_IMAGE_LAYOUT_H

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
namespace zimage {

// The joint image/caption sequence Z-Image's DiT runs over, as pure
// integer arithmetic: no GPU, no weights, nothing to load.
//
// Split out for the reason qwen-image21-layout and minimax-h3-layout
// are: the sequence IS the model's structure, every part of it is a
// silent failure when wrong, and checking it against the reference
// needs no checkpoint.
//
// WHAT THE SEQUENCE IS, and it is the opposite order from most of this
// tree:
//
//     [ image patches | padding | caption tokens | padding ]
//
// IMAGE FIRST. The reference builds `torch.cat([x, cap])` in its basic
// (text-to-image) mode, and the final layer reads the leading
// `img_tokens` rows straight back out. Putting the caption first
// produces a picture -- the wrong one.
//
// BOTH HALVES ARE PADDED TO A MULTIPLE OF 32, independently, and the
// padding is not inert: each padded row carries a LEARNED token
// (`x_pad_token` / `cap_pad_token`, one 3840-wide vector each) and
// participates in attention like any other row. There is no attention
// mask anywhere in this model at batch 1 -- attention is dense and
// bidirectional over the whole joint sequence.
//
// THE TWO HALVES PAD THEIR POSITIONS DIFFERENTLY, which is the detail
// that survives every test that does not look for it. Caption padding
// CONTINUES the caption's position run; image padding is pinned to the
// origin (0,0,0). That asymmetry is in the reference's `_pad_with_ids`:
// the caption asks for a position grid already sized to the padded
// length, so its pad rows fall inside the grid, while the image asks
// for a grid of its real extent and its pad rows take the separate
// origin row.

// The reference's SEQ_MULTI_OF.
inline constexpr int kSeqMultiple = 32;

// The caption is truncated here by the reference's tokenizer call
// (`max_length=512, truncation=True`).
inline constexpr int kMaxCaptionTokens = 512;

// Pixels per patch token: the VAE's 8x stride times the DiT's patch 2.
// Also the size grid -- the reference REFUSES a height or width that is
// not a multiple of it.
inline constexpr int kPixelsPerToken = 16;

// Everything the forward needs about the sequence it is about to run.
struct Layout {
  int grid_h = 0;      // patch rows    (latent H / patch)
  int grid_w = 0;      // patch columns (latent W / patch)
  int img_tokens = 0;  // grid_h * grid_w, the REAL image rows
  int img_total = 0;   // those plus padding, a multiple of kSeqMultiple
  int cap_len = 0;     // real caption tokens
  int cap_total = 0;   // those plus padding, a multiple of kSeqMultiple
  int joint_len = 0;   // img_total + cap_total

  // Rotary coordinates, one per axis, per joint row. Unsigned here --
  // unlike Qwen-Image-2.1 the grid is NOT centred on zero.
  std::vector<std::int32_t> pos_t, pos_h, pos_w;

  // Where each half starts in the joint sequence. The image is first,
  // so these are 0 and img_total; named anyway, because the refiner
  // stacks run over the halves BEFORE they are joined and both they
  // and the final slice read them.
  int img_off() const { return 0; }
  int cap_off() const { return img_total; }
  // Rows carrying the learned pad token, as the half-relative ranges
  // [img_tokens, img_total) and [cap_len, cap_total).
  int img_pad() const { return img_total - img_tokens; }
  int cap_pad() const { return cap_total - cap_len; }
};

// Build the layout. `grid_h`/`grid_w` are PATCH counts (latent extent
// divided by the DiT's patch size 2), `cap_len` the real caption tokens
// after templating and truncation.
//
// Returns false with a reason in `err` rather than throwing: a caller
// that got the bookkeeping wrong should see which invariant it broke,
// not a plausible picture.
bool build_layout(int grid_h, int grid_w, int cap_len, Layout* out,
                  std::string* err);

// ---- the conditioning prompt ----------------------------------------
//
// Here for the same reason the layout is: a FACT about the model that
// needs no checkpoint to state and no GPU to check.
//
// Z-Image has NO system prompt -- it is the only image family in this
// tree that does not. The reference renders Qwen3's own chat template
// over a single user message with a generation turn and thinking
// ENABLED, which for Qwen3 means no forced empty `<think>` block. So
// the whole of it is:
//
//     <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
//
// and NOTHING is dropped from the hidden states afterwards -- there is
// no system span to drop, and the reference keeps every unmasked row.
std::string prompt_template(const std::string& text);

// The conditioning tap: HuggingFace's `hidden_states[-2]`, the
// UN-NORMED output of the second-to-last decoder layer.
//
// Returned in the vocabulary of `MetalQwenModel::forward_embeddings_taps`,
// whose `tap_layers` entry L captures the residual AFTER layer L and so
// equals HF's `hidden_states[L+1]`. hidden_states[-2] is
// hidden_states[n_layers - 1], hence layer n_layers - 2.
//
// Stated as a function because two different off-by-ones live here and
// both return a plausible tensor of exactly the right shape: taking the
// LAST layer instead of the second-to-last, and applying the encoder's
// final norm to a tap that the reference leaves un-normed.
inline int tap_layer(int n_layers) { return n_layers - 2; }

// ---- patch packing --------------------------------------------------
//
// The DiT consumes the latent as one row per patch, and the ROW's own
// ordering is part of the model: the reference reshapes
// (C, F, H, W) -> (F_t, H_t, W_t, pF, pH, pW, C) and flattens the last
// four, so within a row the CHANNEL runs fastest and the patch offsets
// outside it. Getting that backwards is a permutation of 64 numbers
// that every shape check passes.
//
// `lat_h`/`lat_w` are LATENT extents and must divide by `patch`.
// `rows` is [ (lat_h/patch) * (lat_w/patch), patch*patch*channels ].
void pack_latents(const float* z, int channels, int lat_h, int lat_w,
                  int patch, float* rows);

// The inverse, which is the reference's unpatchify: rows back to a
// channel-first [channels, lat_h, lat_w] latent.
void unpack_latents(const float* rows, int channels, int lat_h, int lat_w,
                    int patch, float* z);

}  // namespace zimage
}  // namespace genai
}  // namespace vpipe

#endif
