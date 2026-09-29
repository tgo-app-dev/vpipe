#ifndef GENERATIVE_MODELS_SHARED_SAGE_ATTENTION_H
#define GENERATIVE_MODELS_SHARED_SAGE_ATTENTION_H

#include "generative-models/shared/accel-settings.h"

// SageAttention: the QK^T product of a flash attention run in INT8.
//
// thu-ml/SageAttention (Zhang et al., arXiv:2410.02367). This is a
// from-scratch implementation of the published method against that
// reference; no code is copied.
//
// THE IDEA, and what separates it from Sol-Attn next door. Sol decides
// which key blocks to ATTEND; Sage changes how the ones it attends are
// COMPUTED. They are orthogonal, they compose, and neither reads the
// other -- Sol's exact half runs the same steel flash kernel the dense
// path does, so turning Sage on accelerates that half too.
//
// A flash attention's matrix work is two products per key block: QK^T
// and P*V. Sage quantizes the first pair of operands to int8 with ONE
// scale per block -- per query block for Q, per key block for K, which
// is the granularity the kernel already tiles at -- so dequantizing a
// score tile is a single scalar multiply. P*V is left in the tensor
// dtype: P is a probability, already well-conditioned, and quantizing
// it buys the same again for a much worse error.
//
// THE SMOOTHING IS NOT AN OPTIMISATION, IT IS WHAT MAKES IT WORK. K is
// quantized as K - mean(K) over TOKENS. A key channel's outlier is not
// variation between tokens but a large bias shared by all of them, and
// subtracting it spends the int8 range on the signal instead of on the
// bias. It is EXACT rather than approximate: subtracting a per-channel
// mean shifts every score in a row by the same <q, mean>, and softmax
// does not see a per-row shift, so nothing is added back afterwards.
// MEASURED (sage_attention.per_block_int8_needs_the_smoothing): with it
// the cosine against an f64 reference holds 0.99991 however large the
// bias; without it, it falls away as the bias grows.
//
// Q IS NOT SMOOTHED, and this is the asymmetry a reader expects to be a
// bug. Only the key side carries the shared bias the argument is about,
// and a shift of Q would NOT cancel -- it would move each score by
// <q_shift, k_j>, which varies along the row.
//
// WHAT IT COSTS AND WHAT IT BUYS. MEASURED on an M5, 8 heads x 20036
// rows x 128: the f16 kernel 156.6 ms against 130.6 for the int8 arm
// INCLUDING its prologue, so 1.20x, of which the prologue is 2.2 ms.
// The ceiling is 1.33x -- int8 is 2.00x on the fragment pipe and QK is
// half a flash kernel's matrix work -- so this is most of what was
// there to take. Accuracy at that geometry is cosine 0.99992 against a
// double-precision reference, where the f16 kernel itself is 0.99992.
//
// MATRIX CORES ONLY. The int8 fragment MMA is an M5 (NAX) instruction;
// the ALU steel kernel has no int8 branch and never will, because
// without matrix units there is nothing to win. A box without them
// reports unavailable and the caller runs dense -- see
// MetalSageAttention::available().

#include <cstddef>

// TOOLKIT: compiled into each plugin (libvpipe_toolkit.a), not exported
// by libvpipe -- see the vpipe_toolkit target and plugin/plugin-abi.h.

namespace vpipe {
namespace genai {
namespace sage {

// The function constant the steel flash kernel specialises on. A
// function built with it TRUE takes four extra operands at buffers
// 15..18 (q8, k8, q_scale, k_scale); one built with it false is the
// dense kernel, unchanged and not merely equivalent.
inline constexpr int kQkInt8Constant = 306;

struct Config {
  bool enabled = false;

  // Leading transformer blocks left in the tensor dtype.
  //
  // Zero by default, unlike Sol's one. The reason the two differ: Sol
  // DROPS keys, so an early block's less redundant residual stream is
  // an argument for leaving it alone, while Sage computes every key and
  // every query -- it only computes them in int8, with a per-block
  // scale and an exact smoothing. There is no published profile that
  // needs a dense prefix here, and offering one that defaults to on
  // would cost speed for a worry nothing has measured. It exists so a
  // caller who does measure something can act on it.
  int dense_layers = 0;

  // K - mean(K) before quantizing. On is the method; off is the A/B
  // that shows why (see the note above), and is not a supported way to
  // run a model.
  bool smooth_k = true;
};

// The same settings out of the acceleration BAG a generating stage hands
// every family (accel-settings.h). Header-only, so it compiles into the
// caller and an old plugin reads the keys its own host documented; see
// the note beside sol::config_from_flex for why that matters.
//
// `smooth_k` is NOT a key. It is the A/B that shows why the smoothing
// exists, not a supported way to run a model, so it stays a compile-time
// decision of whoever is measuring rather than something a graph can
// turn off.
inline Config
config_from_flex(const FlexData* bag)
{
  Config c;
  c.enabled = accel::flag(bag, accel::kSageAttn, c.enabled);
  c.dense_layers =
      (int)accel::integer(bag, accel::kSageDenseLayers, c.dense_layers);
  return c;
}

// The key mean is a reduction over the whole sequence, and doing it with
// one thread per channel is D * heads threads each walking every token --
// 1024 threads and 20036 dependent adds at video geometry, which
// MEASURED 4.33 ms of a 6.0 ms prologue. It is split over this many
// chunks of tokens and reduced in a second pass.
inline constexpr int kMeanChunks = 64;

// Scales are one per BLOCK of the attention kernel's own tiling, so the
// counts a caller must size against are the kernel's, not this file's.
inline int query_blocks(int tokens, int bq)
{
  return bq > 0 ? (tokens + bq - 1) / bq : 0;
}
inline int key_blocks(int tokens, int bk)
{
  return bk > 0 ? (tokens + bk - 1) / bk : 0;
}

}  // namespace sage
}  // namespace genai
}  // namespace vpipe

#endif  // GENERATIVE_MODELS_SHARED_SAGE_ATTENTION_H
