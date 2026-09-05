// Copyright © 2024 Apple Inc.

#pragma once

///////////////////////////////////////////////////////////////////////////////
// Attn param classes
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct AttnParams {
  int B; ///< Batch Size
  int H; ///< Heads
  int D; ///< Head Dim

  int qL; ///< Query Sequence Length
  int kL; ///< Key Sequence Length

  int gqa_factor; ///< Group Query factor
  float scale; ///< Attention scale

  int NQ; ///< Number of query blocks
  int NK; ///< Number of key/value blocks

  int NQ_aligned; ///< Number of full query blocks
  int NK_aligned; ///< Number of full key/value blocks

  int qL_rem; ///< Remainder in last query block
  int kL_rem; ///< Remainder in last key/value block
  int qL_off; ///< Offset in query sequence start

  int64_t Q_strides[3]; ///< Query  strides (B, H, L, D = 1)
  int64_t K_strides[3]; ///< Key    strides (B, H, L, D = 1)
  int64_t V_strides[3]; ///< Value  strides (B, H, L, D = 1)
  int64_t O_strides[3]; ///< Output strides (B, H, L, D = 1)
};

struct AttnMaskParams {
  int64_t M_strides[3]; ///< Mask  strides (B, H, qL, kL = 1)
};

/* vpipe: the geometry a BLOCK-SPARSE run masks its span edges against.
 *
 * The block list the host supplies is rounded OUTWARD to whole key
 * blocks, so a block at a span's edge carries keys the mask forbids and
 * the kernel has to test them. The test is closed-form -- a row and a
 * column each map to a video frame, and the frame's window says yes or
 * no -- so nothing is materialised: this is four ints and one [lo, hi]
 * per frame against a [qL, kL] mask that would be 11 billion elements
 * at video geometry.
 */
struct AttnSpanParams {
  int video_start; ///< First video row of the packed sequence
  int tokens_per_frame;
  int num_frames;
  int anchors; ///< bit 0: anchor COLUMNS, bit 1: anchor ROWS
};

} // namespace steel
} // namespace mlx
