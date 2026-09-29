#ifndef GENERATIVE_MODELS_SHARED_SOL_ATTENTION_H
#define GENERATIVE_MODELS_SHARED_SOL_ATTENTION_H

#include "generative-models/shared/accel-settings.h"

// Sol-Attn: training-free sparse attention by on-the-fly routing.
//
// NVlabs/Sana, `techniques/sparse_backends/sol_attn` (Sol-Attn, Li et
// al. 2026, arXiv:2607.24027), Apache-2.0. This is a from-scratch
// implementation of the published method against that reference's
// Triton and MPS sources; no code is copied.
//
// THE IDEA. Attention is dominated by key blocks that contribute almost
// nothing. Sol-Attn decides which ones matter WHILE the online softmax
// runs, from a proxy it has to compute anyway, and then reuses that same
// proxy as the approximation for everything it skipped -- so there is no
// routing map to materialise and no second pass.
//
// Per KV block of 64 keys it precomputes two summaries:
//
//   kc[n] = mean_j k_j        the block's key centroid
//   vc[n] = SUM_j v_j         the block's value SUM, not its mean
//
// and per QUERY block a scalar threshold. Then, for each query block,
// one [64 x D] x [D x N] product gives a proxy score against EVERY key
// block at 1/64 of the dense cost. A block is EXACT when its proxy beats
// the threshold, and APPROXIMATE otherwise -- and an approximate block
// is folded into the same running softmax as if all 64 of its keys had
// the centroid's logit:
//
//   numerator   += exp(proxy_n) * vc[n]      (= sum_j exp(s) v_j)
//   denominator += exp(proxy_n) * len_n
//
// which is exactly why vc is a sum. Blocks the routing keeps are then
// attended properly, key by key.
//
// THE THRESHOLD IS A DISTRIBUTION, NOT A QUANTILE. For a query block
// with centroid qbar, the proxy score across key blocks has mean
// <qbar, mean_n kc[n]> and a variance this estimates from the spread of
// the centroids; the threshold is mean + tau * sd, in LOG2 units because
// that is what the softmax runs in. So `tau` is measured in standard
// deviations of a block's own score distribution -- which is what makes
// one number work across heads, layers and sequence lengths, where a
// fixed "keep k blocks" would not. Higher tau keeps fewer blocks.
//
// `kDiag` estimates that variance from the per-channel spread of the key
// centroids alone; `kExact` uses their full second-moment matrix, which
// is D x D per head and costs more to build. The published Stage-2
// profile uses kDiag.
//
// WHAT IS ALWAYS EXACT, and neither is a tuning knob:
//
//   * a band of `local_radius` blocks either side of the query's own.
//     Attention's near-diagonal is where a centroid is least
//     representative -- neighbouring keys are precisely the ones a query
//     distinguishes between -- so routing is not asked about them.
//   * an optional KV SINK, a contiguous token range every query attends
//     exactly. That is what a packed sequence needs: H3 puts text and
//     audio in one run with the video, and a prompt summarised by its
//     centroid is a prompt half-read.
//
// Sequence layout is HEAD-MAJOR [H, T, D] throughout, which is what this
// tree's own sdpa kernels use; the published reference is BTHD.

#include <cstddef>
#include <string>
#include <vector>

// TOOLKIT: compiled into each plugin (libvpipe_toolkit.a), not exported
// by libvpipe -- see the vpipe_toolkit target and plugin/plugin-abi.h.

namespace vpipe {
namespace genai {
namespace sol {

// The block size is not a tuning parameter: the summaries, the
// threshold statistics and the proxy's 1/64 cost ratio are all defined
// against it, and the published kernels are built for 64.
inline constexpr int kBlock = 64;

enum class Threshold { kDiag, kExact };

bool parse_threshold(const std::string& name, Threshold* out);
const char* threshold_name(Threshold t);

struct Config {
  bool      enabled      = false;
  float     tau          = 1.0f;
  Threshold thresh       = Threshold::kDiag;
  // Leading transformer blocks left DENSE. The published profile leaves
  // layer 0 dense: the first block is where the residual stream is
  // least redundant, and it is one layer of 50.
  int       dense_layers = 1;
  int       local_radius = 1;
  // An exact KV sink, in TOKENS: every query attends [sink_start,
  // sink_start + sink_tokens) exactly. Rounded OUTWARD to whole blocks,
  // because exactness is a property of the block the kernel visits.
  int       sink_start   = 0;
  int       sink_tokens  = 0;
  // The KEY block size. 0 takes kBlock (64), which is both the published
  // value and -- MEASURED on an M4 Pro at 56 heads x 20036 rows -- the
  // knee: 32 gives 3.18x over dense steel, 64 gives 4.16x, 128 gives
  // 3.91x, 256 gives 2.70x. Bigger blocks make the routing cheaper and
  // the centroid a worse stand-in, and the two cross here.
  //
  // Must be a multiple of steel's 16-key block, because the CSR the
  // exact half walks is in steel's units.
  int       key_block    = 0;
};

// The same settings out of the acceleration BAG a generating stage
// hands every family (accel-settings.h).
//
// HEADER-ONLY, so it compiles into the caller: a plugin built against an
// older host reads the keys that host documented and gets this file's
// defaults for anything it has never heard of -- which is the same thing
// it gets from a graph that asked for nothing. That is what lets the
// host add a knob here without invalidating a binary that does not use
// it.
//
// The defaults below are THIS file's, deliberately: they are the shipped
// values, and the stage writes settled values into the bag whenever the
// tier is on, so the two agree wherever it matters and the reader is
// still correct on a bag that carries nothing.
inline Config
config_from_flex(const FlexData* bag)
{
  Config c;
  c.enabled      = accel::flag(bag, accel::kSolAttn, c.enabled);
  c.tau          = (float)accel::real(bag, accel::kSolTau, (double)c.tau);
  c.dense_layers =
      (int)accel::integer(bag, accel::kSolDenseLayers, c.dense_layers);
  c.local_radius =
      (int)accel::integer(bag, accel::kSolLocalRadius, c.local_radius);
  c.key_block = (int)accel::integer(bag, accel::kSolKeyBlock, c.key_block);
  // The SINK is not in the bag and cannot be: it is where a family's own
  // sequence stops being the modality the routing is summarising, and
  // only the family knows that. H3 fills it from its packed layout; a
  // per-stream self-attention leaves it at zero.
  return c;
}

// What a forward actually routed, for a benchmark to report. Realized
// sparsity is a property of the DATA, not of tau alone, so it is
// measured rather than predicted.
struct Stats {
  long long exact_blocks  = 0;   // (query block, key block) pairs kept
  long long approx_blocks = 0;
  long long total_blocks  = 0;
  double exact_fraction() const
  {
    return total_blocks > 0 ? (double)exact_blocks / (double)total_blocks
                            : 0.0;
  }
};

inline int num_blocks(int tokens) { return (tokens + kBlock - 1) / kBlock; }

// Block summaries. `qc` is the per-QUERY-block centroid (mean), `kc` the
// per-key-block centroid, `vc` the per-key-block SUM. Each [H, N, D],
// fp32. q/k/v are head-major [H, T, D].
void summaries(const float* q, const float* k, const float* v, int heads,
               int tokens, int d, float* qc, float* kc, float* vc);

// Per (head, query block) routing threshold, in LOG2 space: [H, N].
void thresholds(const float* qc, const float* kc, int heads, int blocks,
                int d, float scale, float tau, Threshold t, float* out);

// The routed forward. q/k/v/out are head-major [H, T, D] fp32; kc/vc are
// what summaries() produced; `thr` what thresholds() produced.
//
// `stats` (optional) accumulates what was routed.
// ROUTING READS `qc`, NOT THE SCORE TILE. The published kernel routes
// on the column mean of the proxy scores over the query block's rows,
// which is <mean_i q_i, kc[n]> -- the same quantity as <qc, kc[n]>, and
// the one the threshold was built against. Taking it from `qc` directly
// is the same arithmetic in one dot product instead of 64, and it makes
// the routing DECISION identical between this reference and the kernel:
// a routing disagreement moves the output by far more than any
// precision difference, so it must not be something the two can drift
// apart on.
void forward(const float* q, const float* k, const float* v,
             const float* qc, const float* kc, const float* vc,
             const float* thr, int heads, int tokens, int d, float scale,
             const Config& cfg, float* out, Stats* stats = nullptr);

// Dense attention over the same layout, for the tests and the A/B.
void dense(const float* q, const float* k, const float* v, int heads,
           int tokens, int d, float scale, float* out);

}  // namespace sol
}  // namespace genai
}  // namespace vpipe

#endif  // GENERATIVE_MODELS_SHARED_SOL_ATTENTION_H
