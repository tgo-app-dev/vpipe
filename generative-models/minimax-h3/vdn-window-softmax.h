#ifndef GENERATIVE_MODELS_MINIMAX_H3_VDN_WINDOW_SOFTMAX_H
#define GENERATIVE_MODELS_MINIMAX_H3_VDN_WINDOW_SOFTMAX_H

// VDN-H3's softmax half: exact attention over a WINDOW of whole frames.
//
// (video, video) pairs are restricted to the query frame's chunk window;
// every pair touching a GLOBAL row -- the prompt, the soundtrack --
// stays dense in both directions, because those are few and the model
// reads them exactly. The linear branch then carries the complement, so
// the window's edge is a contract between the two halves rather than an
// approximation either of them makes alone.
//
// WHY A WINDOW OF CHUNKS AND NOT OF FRAMES. The video VAE encodes every
// 5 latent frames as one unit, so a frame that sees part of a
// neighbouring chunk sees a fragment of something never coded
// separably. Chunk alignment guarantees every frame a complete previous,
// current and next chunk; a centred frame window cannot express that at
// any radius, because it gives frames 5 and 9 different spans and one of
// them always straddles a boundary. See vdn-geometry.h.
//
// THE GATE IS PER HEAD, not per channel, and that is not a saving. A
// windowed softmax renormalises to 1 no matter how little of the
// sequence it saw, so this scales the branch back toward the share it
// actually captured -- a scalar property of a DISTRIBUTION. The linear
// branch's gate is per channel because it is routing a new pathway,
// which is a different question.
//
// This is the CPU reference: one exact softmax per query row over its
// allowed keys, O(seq^2). It exists to check a block-sparse kernel
// against, and to be the thing a port is wrong relative to.

#include "generative-models/minimax-h3/vdn-geometry.h"

#include <cstddef>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace vdn {

// q / k / v / out: [seq_len, heads, head_dim], already QK-normed and
// RoPE'd -- this function does neither. `scale` is the usual
// head_dim^-0.5.
//
// Rows the mask allows nothing at all would be a division by zero; that
// cannot happen for a well-formed layout (every row sees at least
// itself), and a row that somehow allowed nothing is left ZERO rather
// than NaN, because a NaN here propagates through the residual stream
// and the frame it came from is unrecoverable from the output.
void window_softmax(const float* q, const float* k, const float* v,
                    const WindowMask& mask, int heads, int head_dim,
                    float scale, float* out);

// gate = sigmoid(W x + b), one value per (token, head), applied to the
// attention output and flattened into the layout to_out wants:
// [seq_len, heads * head_dim].
void apply_softmax_gate(const float* attn, const float* x, int seq_len,
                        int heads, int head_dim, int hidden,
                        const float* gate_w, const float* gate_b,
                        float* out);

}  // namespace vdn
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
