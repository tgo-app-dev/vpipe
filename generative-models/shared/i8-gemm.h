#ifndef VPIPE_GENERATIVE_MODELS_SHARED_I8_GEMM_H
#define VPIPE_GENERATIVE_MODELS_SHARED_I8_GEMM_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/kernel-autotune.h"
#include "generative-models/shared/tuned-answers.h"

#include <cstddef>
#include <cstdlib>
#include <string_view>
#include <vector>

// TOOLKIT: compiled into each plugin (libvpipe_toolkit.a), not exported
// by libvpipe -- see the vpipe_toolkit target and plugin/plugin-abi.h.

namespace vpipe {
namespace genai {

// Dynamic-int8 accelerated GEMM ("accelerated mode" for large matmuls --
// DiT blocks, LLM prefill): the f16/bf16 activation is quantized ON THE FLY
// to i8 with per-(row, group) scales, the f16/bf16 weight (dense, or a
// dequant scratch) is quantized per-(out-channel, group) into a reusable
// i8 scratch, and the product runs on the matrix units' int8 pipe -- ~2x
// the matmul2d rate at qualifying shapes -- accumulating each group in f32
// with its own pair of scales and storing the element type back.
//
// THE GROUP IS 64 on both operands (quant_f16_i8_row_g64, or its rotated twin
// -- see below -- + gemm_i8i8_sc_f16_n64_g64rs). It was 512, and the reason it
// could not be finer was the kernel, not the quantization: the drained
// per-group kernel paid a threadgroup round-trip per group, which MEASURED g64
// at 6.4 TOP/s against g512's 19.3 -- below f16. The register-resident kernel
// keeps the partial in a cooperative tensor and stages each strip of scales
// once, and at that design g64 runs 17.4 / 16.9 TOP/s (shallow / deep K)
// against the same kernel's 21.3 / 19.3 at 512 -- so the finer group costs
// 12-18% of the coarser one's rate and buys the precision a 64-wide group has
// over a 512-wide one on every outlier channel (gemm_i8.group_size,
// gemm_i8.g64_register_acc). VPIPE_I8_GROUP=512 selects the old pair for an
// A/B, and 32 / 128 the same register design one step finer or coarser;
// anything else is 64.
//
// THE GROUPS OF 64 ARE ROTATED (set_g64_rotation; on by default): each is
// Walsh-Hadamard transformed on both operands before its scale is taken
// (quant_f16_i8_row_g64_r), so an outlier channel's energy spreads over its
// group instead of setting the group's scale. A group's dot product is the same
// in the rotated basis, so the GEMM is unchanged. MEASURED on the base M5
// (gemm_i8.g64_rotation_cost, Qwen-Image-2.1's block at 4096 rows, two runs):
// free -- the seven projections 0.9-1.0% FASTER per call, the quantizer within
// 0.01 ms either way. On real weights and activations captured from
// Qwen-Image-2.1 (672 projections) and MiniMax-H3 FL2VA (600), the output's
// rel-L2 from the exact product, median 6.27e-3 -> 5.04e-3 and 8.71e-3 ->
// 7.22e-3, worst 2.13e-2 -> 1.12e-2 and 1.45e-2 -> 1.19e-2: under the fast tier
// on both. The Qwen-Image-2.1 codec's attention GEMMs moved under 0.15 dB on
// two 1024^2 pictures, every tier; its P.V is nonetheless kept plain
// (gemm_unrotated -- the probabilities are all positive, the case below). THE
// TRADE: a group whose values share a large common offset -- a mean several
// times their spread -- reads WORSE rotated, since the transform gathers the
// mean into one coefficient (eight times the mean), which then sets the scale
// (gemm_i8.g64_quantizers_match_their_restatement: 2.5x at a 4 sigma offset).
// Real block projections do not look like that; a new caller whose operand
// might should measure. gemm() alone takes it: the signed and native routes
// read a pack's own codes, which are in the weight's own basis, and rotating
// those would round the weight twice (no gain measured: Qwen-Image-2.1 0.93x,
// MiniMax-H3 1.02x the signed route's median). VPIPE_I8_ACC_ROT=0 is the A/B.
//
// THE WHOLE SET ON ONE DESIGN, MEASURED on the M5 (gemm_i8.group_size
// at 4096x12288x4096 / 4096x4096x12288; gemm_i8.group_quality rel-L2
// vs f64 on gaussian data and on data with one outlier channel in 64 at
// 32x; the Krea-2 DiT forward against f16; VOSR's int8 tier, whose
// plain matrix-core tier is 303 ms):
//
//   group   TOP/s          gaussian  outliers  Krea-2   VOSR
//    32     14.0 / 13.6    7.6e-3    1.35e-2   5.6e-3   361 ms  4.40e-2
//    64     17.3 / 16.4    8.4e-3    1.86e-2   8.3e-3   297 ms  4.72e-2
//   128     19.3 / 18.5    9.2e-3    2.36e-2   7.5e-3   303 ms  5.81e-2
//   512     20.3 / 19.2   10.6e-3    3.38e-2  11.3e-3   273 ms  7.12e-2
//
// The depth is never the cost: a 32-deep matmul into the cooperative
// accumulator runs at the 512-deep rate (kaccr-32 19.6 against
// kaccr-512 20.2). The cost is the epilogue, once per group per output,
// so halving the group adds the same work again -- g32 sits 30% under
// its ceiling, g64 13%, g128 4%. What a finer group buys is on the
// outlier row, which is what real activations look like. 64 is the
// default because 32 turns VOSR's int8 tier SLOWER than its plain
// matrix-core tier and 128 gives back a third of the outlier-row
// precision for 12% of rate. (Krea-2's 128 < 64 is a single forward's
// rel-L2, not a trend; the unit rows are monotone.) The table predates
// the rotation below and was measured with the groups unrotated.
//
// The activation/weight/scale/output element type is chosen by `bf16`: the
// f16 caller loads the `dense_gemm_mma`/`affine_dequant` metallibs, the bf16
// caller (e.g. the FLUX.2 klein DiT, whose residual stream overflows f16) the
// `_bf16` twins -- the same kernels compiled with VPIPE_ELT=bfloat. The int8
// scratch is format-independent and the scale scratch is 2 bytes either way,
// so only the library selection changes. Reading a bf16 weight/activation
// through the f16 kernels (or vice versa) reinterprets the bits -> garbage;
// the caller MUST match `bf16` to its buffers' element type.
//
// LOSSY: int8 quantization, rel-L2 ~1e-2 per GEMM. Strictly OPT-IN and
// never part of a token-exact path. The per-call weight re-quant costs
// one extra pass over the weight (~1-2% of a qualifying GEMM), so the
// mode has NO persistent memory cost and composes with every checkpoint
// format that already produces an f16/bf16 weight for the matmul.
//
// The stage/model switch arrives via `want`; env VPIPE_I8_GEMM=0|1
// overrides either way (A/B), VPIPE_I8_GEMM_MIN_M tunes the M gate.
class I8GemmContext {
 public:
  I8GemmContext(metal_compute::MetalCompute* mc, bool want, bool bf16 = false);

  bool enabled() const { return _on; }

  // The row floor every route gates on: 1024 by default, the requant
  // route's crossover, where the weight is quantized per call. A caller
  // whose weights are HELD in the fast layout may lower it: that GEMM
  // reads half the f16 route's weight bytes at any row count, and the
  // dense route under the floor would have to dequantize the layout first
  // (MEASURED on the base M5, MiniMax-Music3's DiT at 696 rows: token 175
  // ms an evaluation on the int8 route, 376 dequantized, f16 318). Inline,
  // no symbol; VPIPE_I8_GEMM_MIN_M, when set, wins.
  void set_min_rows(int m)
  {
    if (std::getenv("VPIPE_I8_GEMM_MIN_M") == nullptr) {
      _min_m = m > 1 ? m : 1;
    }
  }
  int min_rows() const { return _min_m; }

  // Shape gate: the win regime is big-M compute-bound GEMMs (measured
  // crossover ~1k rows on M5). K must split into whole groups, OR the
  // padding quantizer must be available to make it so -- see kpad_() -- and
  // the padding must be CHEAP, which is the last clause.
  //
  // Padding is exact but not free: it is (KP-K)/K extra int8 MACs on every
  // call, and the int8 rate has to beat bf16 by more than that for the mode
  // to still pay. Capped at _max_pad_pct (10 by default).
  //
  // Where the cap bites depends on the group. At 512 the pad is at most
  // 511, so any K >= 5120 passes whatever its remainder and below that it
  // depends on K % 512 -- K=2816 pads 256 (9.1%, taken), K=1600 pads 448
  // (28%, refused). At 64 the pad is at most 63, which is under the cap
  // for every K this gate admits, so the clause is dormant there.
  bool accepts(int M, int N, int K) const
  {
    return accepts_(M, N, K, _g64_rot);
  }

  // Encode act-quant + weight-quant + the i8 GEMM (element type = the ctor's
  // `bf16`; x/w/y must all be that type):
  //   y[M,N] (elem offset ye) = x[M,K] (elem offset xe) @ w[N,K]^T (dense)
  // Returns false -- with nothing encoded -- when the shape does not
  // qualify or a scratch allocation fails (caller keeps its dense path).
  bool gemm(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& x, std::size_t xe,
            const metal_compute::SharedBuffer& w,
            const metal_compute::SharedBuffer& y, std::size_t ye,
            int M, int N, int K)
  {
    return gemm(enc, x, xe, w, metal_compute::SharedBuffer{}, y, ye, M, N,
                K);
  }

  // ...and the twin that adds a per-column bias. A model whose every
  // projection carries one (VOSR) could otherwise reach none of this:
  // the int8 path computes x @ w^T and had nowhere to put the b. An
  // empty `b` is the overload above, and is byte-identical to what
  // shipped -- the term is not computed, not added as zero.
  bool gemm(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& x, std::size_t xe,
            const metal_compute::SharedBuffer& w,
            const metal_compute::SharedBuffer& b,
            const metal_compute::SharedBuffer& y, std::size_t ye,
            int M, int N, int K);

  // ...and gemm() with its groups NEVER rotated, whatever the context's
  // setting and VPIPE_I8_ACC_ROT say: for an operand whose groups share a
  // large common offset, the case the rotation loses (the class note) --
  // attention probabilities, all positive (a P.V GEMM: MetalKrea2Vae's
  // materialized mid attention). Same shape gate as accepts().
  bool gemm_unrotated(metal_compute::ComputeEncoder& enc,
                      const metal_compute::SharedBuffer& x, std::size_t xe,
                      const metal_compute::SharedBuffer& w,
                      const metal_compute::SharedBuffer& y, std::size_t ye,
                      int M, int N, int K)
  {
    return gemm_(enc, x, xe, w, metal_compute::SharedBuffer{}, y, ye, M, N,
                 K, false);
  }

  // ---- NATIVE affine consumption -----------------------------------
  //
  // The checkpoint's own codes -- MLX affine, w = s*q + b, group 64, w4
  // nibbles or w8 bytes, scales and biases [N, K/64] in the element type
  // -- against the int8 activation, with no dequant scratch and no
  // requant: the matrix units read the codes as they lie (uint8 x uint8
  // or uint8 x uint4), and the scale and zero point are folded into the
  // f32 epilogue exactly (gemm_u8q_impl). Per call it costs the u8
  // activation quant and one pass over the codes for their group sums;
  // it saves the dequant pass, the requant pass, the N*K*2 dense scratch
  // and the second quantization of the weight.
  //
  // MEASURED on the M5 (gemm_i8.native_affine_rate, the whole route a
  // caller pays, dequant + act quant + requant + GEMM against act quant
  // + code sums + GEMM), time relative to the dequant + requant route:
  //
  //   w8  H3 fc2       1024 x 7168 x 14336   1.21x
  //   w8  square       4096 x 4096 x 4096    1.01x
  //   w4  FLUX.2 proj  4096 x 12288 x 4096   0.83x
  //   w4  deep K       4096 x 4096 x 12288   0.83x
  //
  // and f32 quality 6.1e-3 against the round trip's 7.1e-3 at 256 x 512
  // x 1024, both widths (its groups unrotated) -- the weight is quantized
  // once, not twice. The
  // w4 loss is the matrix pipe's: uint8 x uint4 runs 17.6 TOP/s where
  // int8 x int8 runs 22.2 (gemm_i8.native_affine_kernel_arms), and that
  // 20% is more than the two passes it saves in isolation. A signed
  // 4-bit code format (int8 x int4b, 18.8) measured 14.9 against the
  // requant kernel's 17.3 and would not close it either.
  //
  // IN A MODEL the w4 gap is much smaller than that microbench: the
  // round trip's two weight passes stream every weight twice more from
  // DRAM per call, which the one-weight-cache-warm microbench under-
  // prices. MEASURED, FLUX.2-klein-9B (w4) DiT step, i8 on:
  //
  //   768^2  (seq 2368)   requant 3007 ms   native 3014 ms   1.00x
  //   1024^2 (seq 4160)   requant 5325 ms   native 5537 ms   0.96x
  //
  // with the DiT's drift against bf16 3.71e-2 -> 3.40e-2, and no dequant
  // scratch (N*K*2) nor requant scratch (N*K) held for the run. MiniMax-
  // H3 (w8): the LoRA test's "int8 alone moves" 1.02e-2 -> 9.7e-3 at
  // 1.0-1.2x the route rate. So the DEFAULT is native for BOTH widths,
  // at the price of up to 4% on a 1024^2 w4 DiT step: VPIPE_I8_NATIVE =
  // 0 (off), 8 (w8 only, the fastest), 4 (w4 only), 1 or 12 (both, the
  // default).
  //
  // AND IN THE SIGNED LAYOUT, PER CALL, where signed_available(bits): the
  // codes are converted into a scratch (one pass, as the code sums above
  // were) and the signed GEMM runs instead of the unsigned one -- see
  // gemm_signed. The unsigned kernel remains for VPIPE_I8_SIGNED=0 and a
  // host without the signed kernels. Either way the shape gate is the
  // requant route's, 1024 rows: a caller of this route was promised that.
  //
  // Returns false with nothing encoded when the shape does not qualify
  // (as accepts(), and K % 64 == 0 -- a g64 checkpoint's K always is),
  // when bits is not 4 or 8, when the group is not 64, or when the
  // policy above excludes the width; the caller then takes its dequant
  // path and the requant gemm() above.
  bool gemm_affine(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& x, std::size_t xe,
                   const metal_compute::SharedBuffer& codes,
                   const metal_compute::SharedBuffer& scales,
                   const metal_compute::SharedBuffer& qbias, int bits,
                   const metal_compute::SharedBuffer& y, std::size_t ye,
                   int M, int N, int K)
  {
    return gemm_affine(enc, x, xe, codes, scales, qbias, bits,
                       metal_compute::SharedBuffer{}, y, ye, M, N, K);
  }
  bool gemm_affine(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& x, std::size_t xe,
                   const metal_compute::SharedBuffer& codes,
                   const metal_compute::SharedBuffer& scales,
                   const metal_compute::SharedBuffer& qbias, int bits,
                   const metal_compute::SharedBuffer& b,
                   const metal_compute::SharedBuffer& y, std::size_t ye,
                   int M, int N, int K);

  // Whether gemm_affine can run at all for this bit width.
  bool native_available(int bits) const
  {
    if (!_on || _group != 64 || !_fn_quant_u8.valid()) { return false; }
    if (bits == 8) {
      return (_native & 8) != 0 && _fn_u8q_w8.valid() && _fn_wsum8.valid();
    }
    if (bits == 4) {
      return (_native & 4) != 0 && _fn_u8q_w4.valid() && _fn_wsum4.valid();
    }
    return false;
  }
  // The native policy mask (8 | 4), and a way for a test to set it.
  int native_policy() const { return _native; }
  void set_native_policy(int mask) { _native = mask & 12; }

  // ---- the SIGNED-CODE layout ---------------------------------------
  //
  // The native route reads the checkpoint's codes as they lie and pays
  // for it twice: the matrix units have no int8 x uint8 form, so it runs
  // uint8 x uint8 (16% under int8 x int8 at a wide N on the M5 Pro), and
  // the activation's +128 has to come back out through two more post-
  // loop matmuls. Store the codes SIGNED instead -- q ^ 0x80 per byte
  // (w8), q ^ 0x8 per nibble (w4), which is lossless -- with the zero
  // point moved to b' = b + 128 s / 8 s, and the product is int8 x int8
  // with one post-loop matmul and no per-call pass over the weight. Only
  // b' is rounded, once, to the element type.
  //
  // MEASURED on the M5 Pro (gemm_i8.w8_against_bf16_routes: 4096 rows,
  // FLUX.2 / Z-Image / FLUX.3 projections, bf16, splits off), per GEMM
  // against the same weight held bf16 under i8 (requant + g64rs):
  //
  //   w8 dequant + requant   0.95-0.98x     rel-L2 vs f64  7.7e-3 (plain)
  //   w8 native              0.95-1.00x                    6.3e-3
  //   w8 signed              1.03-1.06x                    6.6e-3
  //
  // So a w8 pack is FASTER than bf16 under i8 only in this layout.
  //
  // A family converts each weight ONCE -- at load, inside the derived
  // buffer it keeps, or after each streamed refill, in place in its own
  // slot (encode_to_signed allows in == out) -- and from then on every
  // reader of that weight must understand the layout: gemm_signed here,
  // encode_dequant_signed for the dense matmul2d route below the i8
  // floor. The steel qmm and the qmv kernels do NOT read it; a weight
  // that needs either stays unsigned. VPIPE_I8_SIGNED = 0 (off), 8, 4,
  // or 1 / 12 (both, the default).
  bool signed_available(int bits) const
  {
    if (!_on || _group != 64 || !_fn_quant_s8.valid() ||
        !_fn_fold.valid()) {
      return false;
    }
    if (bits == 8) {
      return (_signed & 8) != 0 && _fn_s8_w8.valid() &&
             _fn_to_s8.valid() && _fn_dq_s8.valid();
    }
    if (bits == 4) {
      return (_signed & 4) != 0 && _fn_s8_w4.valid() &&
             _fn_to_s4.valid() && _fn_dq_s4.valid();
    }
    return false;
  }
  int signed_policy() const { return _signed; }
  void set_signed_policy(int mask) { _signed = mask & 12; }

  // y[M,N] = x[M,K] @ w^T for a weight in the signed layout: codes [N, K
  // bits/8], scales and zero points b' [N, K/64] in the element type.
  // GROUP 64 ON THE WEIGHT as well as the activation, and only the caller
  // knows the weight's: a group-32 pack's scales are twice as long and
  // would be read as these, so a caller with one must not call this (nor
  // gemm_affine) -- every family here gates on its _quant_group.
  // Returns false with nothing encoded when the shape does not qualify
  // (as gemm_affine) or the width is unavailable.
  bool gemm_signed(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& x, std::size_t xe,
                   const metal_compute::SharedBuffer& codes,
                   const metal_compute::SharedBuffer& scales,
                   const metal_compute::SharedBuffer& zb, int bits,
                   const metal_compute::SharedBuffer& y, std::size_t ye,
                   int M, int N, int K)
  {
    return gemm_signed(enc, x, xe, codes, scales, zb, bits,
                       metal_compute::SharedBuffer{}, y, ye, M, N, K);
  }
  bool gemm_signed(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& x, std::size_t xe,
                   const metal_compute::SharedBuffer& codes,
                   const metal_compute::SharedBuffer& scales,
                   const metal_compute::SharedBuffer& zb, int bits,
                   const metal_compute::SharedBuffer& b,
                   const metal_compute::SharedBuffer& y, std::size_t ye,
                   int M, int N, int K);

  // ---- THE FAST TIER ---------------------------------------------
  //
  // Two tiers of int8 GEMM. ACCURATE (the default) is everything above:
  // groups of 64 on both operands, rotated where gemm() quantizes both
  // (the class note). FAST runs symmetric int8 at a COARSE
  // group gc (512 by default) on both operands, so the per-group fold
  // runs once per gc instead of once per 64 and a group's contraction is
  // one matmul -- and it ROTATES each group with an orthonormal
  // Walsh-Hadamard transform first (both operands, so the product is
  // unchanged), which spreads an outlier channel over its group instead
  // of letting it set the group's scale. MEASURED (gemm_i8.coarse_tier,
  // 256x512x4096 against f64): with no outliers the fast tier is 1.30-
  // 1.45x the accurate tier's error; with outlier channels (1/64 x24,
  // 1/512 x64) it is 0.5-0.6x of it, where unrotated it was 1.8-2.7x --
  // against the accurate tier's groups UNROTATED. Rotated, the accurate
  // tier is under the fast one on real activations too (the class note).
  // MEASURED on Qwen-Image-2.1's cached step (forward_ab), against the
  // accurate tier with split-K off (the fast tier never splits, and the
  // split tuner's picks move the accurate tier 1-7% load to load): 1.041x
  // on the M5 Ultra at 1024^2, 1.156-1.183x on the base M5 at 512^2.
  // g1024 is SLOWER than g512 on both (0.974x / 1.098x) and further from
  // the accurate output: at g512 the fold is already nearly free, and 32
  // values a lane in the rotated quantizer cost more than half the folds
  // save. Pictures as close to the dense ones as the accurate tier's
  // (median PSNR 35 dB over two prompts x three seeds, none under 30).
  //
  // The weight is HELD in the fast layout: int8 codes [N, K] in the
  // rotated basis and scales [N, K/gc] in the element type, converted
  // once (encode_to_fast from a w8 g64 pack, encode_quant_fast from a
  // dense weight). No zero point, no split, no tuning.
  //
  // TOKEN is the same layout at its limit: the group is the WHOLE ROW
  // (gc == kTokenGroup) and nothing rotates -- int8 weights per output
  // channel and activations per token, each over all of K, which is how
  // Draw Things stores its "8-bit" (i8x) checkpoints. Nothing varies
  // along K, so a GEMM is one int32 accumulation with the two scales
  // applied at the store, and the activation quantizer is a plain amax a
  // row. Scales are f32: [N] for a weight, [M] for an activation. The
  // GEMM is a register-fragment kernel (gemm_i8pt_m128x64: each
  // simdgroup loads its own operands, K permuted inside each 64-deep
  // step so a lane's four fragments are one 16-byte load, the K loop
  // MLX's gemm_nax one).
  //   MEASURED (gemm_i8.per_token, quantizer + GEMM against the g512
  //   fast tier at Qwen-Image-2.1's shapes, 4096 rows): M5 Ultra 1.29x
  //   q/k/v/out, 1.35x gate/up, 1.62x down; base M5 (throttled, ratios
  //   only) 2.0 / 1.49 / 1.73x. The matmul2d tile kernels at the same
  //   layout reach only 1.0-1.4x.
  //   The price is accuracy: with no rotation an outlier channel sets
  //   its token's scale for all of K (gemm_i8.coarse_tier's study).
  //   ROTATED (set_token_rotation, VPIPE_I8_TOKEN_ROT): both operands
  //   Walsh-Hadamard transformed in blocks of 256 along K before their
  //   row scale is taken -- Draw Things' "Hadamard-256" i8x -- which
  //   costs only the quantizers (the GEMM is the same kernel) and needs
  //   K % 256 == 0. A rotated weight and a plain one are different
  //   bytes: layout_tag() names which a cache entry holds.
  enum class Tier { kAccurate = 0, kFast = 1, kToken = 2 };
  static constexpr int kTokenGroup = 1 << 30;
  // The caller's choice; VPIPE_I8_TIER (accurate | fast | token)
  // overrides it.
  void set_tier(Tier t) { _tier = _tier_env >= 0 ? (Tier)_tier_env : t; }
  Tier tier() const { return _tier; }
  // The token tier's rotation; VPIPE_I8_TOKEN_ROT (0 | 1) overrides it.
  void set_token_rotation(bool r) { _tok_rot = _tok_rot_env >= 0
                                                   ? _tok_rot_env != 0 : r; }
  bool token_rotated() const { return _tok_rot; }
  // THE ACCURATE TIER'S ROTATION (see the class note; on by default):
  // gemm()'s two quantizers Walsh-Hadamard transform each group of 64
  // before its scale is taken (quant_f16_i8_row_g64_r), the GEMM the
  // same kernel. Group 64 and gemm() alone -- the native and signed
  // routes read a pack's own codes, which no rotation reaches.
  // VPIPE_I8_ACC_ROT (0 | 1) overrides the caller. Returns whether it
  // rotates now: never where the kernels are missing (an older host) or
  // the group is not 64.
  bool set_g64_rotation(bool r);
  bool g64_rotated() const { return _g64_rot; }
  // The token tiers' GEMM kernel: its K loop -- MLX's blocked one (512
  // deep, a barrier each) or, double_buffered, one with no barrier that
  // loads the next step's operands under this step's matmuls -- and its
  // tile, 128x64 (4x2 simdgroups) or, wide_n, 64x128 (2x4). The context
  // DECIDES THIS ITSELF -- blocked 128x64, the fastest in the measured
  // forwards (Qwen-Image-2.1 1024^2 on the M5 Ultra: double-buffered
  // 0.965x at 128x64, 0.986x at 64x128; base M5 512^2 0.945x) -- and no
  // stage exposes it; this is for an A/B, and VPIPE_I8_TOKEN_KERNEL /
  // VPIPE_I8_TOKEN_TILE override it. Every choice computes the same
  // int32 sums and applies the scales the same way, so the output is
  // BIT-IDENTICAL, and the weight layout does not depend on it. Call
  // before the first GEMM.
  void set_token_kernel(bool double_buffered, bool wide_n);
  // The kernel set_token_kernel (or the env) settled on, for a log line:
  // "blocked 128x64", "double-buffered 64x128", ...
  std::string token_kernel() const;
  // What a fast-layout weight at group gc holds, for a cache key:
  // "fast<gc>", "tok" or "tokr" (rotated).
  std::string layout_tag(int gc) const
  {
    return gc == kTokenGroup ? (_tok_rot ? "tokr" : "tok")
                             : "fast" + std::to_string(gc);
  }
  // The group a K takes on the fast tier: the largest of 1024 / 512 /
  // 256 that is no more than the target (VPIPE_I8_FAST_GROUP, 512) and
  // divides K; on the token tier kTokenGroup where K % 64 == 0 (256
  // rotated) and K <= 16384 -- 0 where none does, or the tier is
  // accurate, or its kernels are missing. Such a weight stays on the
  // accurate tier.
  int fast_group(int K) const;
  // Bytes of a fast-layout weight's (or activation's) scales, [N, K]
  // at group gc: N * K/gc elements, or N f32 at kTokenGroup.
  static std::size_t fast_scale_bytes(int N, int K, int gc)
  {
    return gc == kTokenGroup ? (std::size_t)N * 4
                             : (std::size_t)N * (std::size_t)(K / gc) * 2;
  }
  // A PLAN-TIME bound on what one forward's context holds at M rows, for
  // a stage that has no context to ask (a DiT's forward_scratch_bytes):
  // the activation codes and g64 scales at the widest K the int8 GEMMs
  // read (`k_act`), one [n_w, k_w] weight's codes and scales -- the
  // accurate tier's per-call requant, a w4 pack's per-call conversion --
  // the tuner's [2048, n_w] output, and the split-K planes: the accurate
  // tier's deep-K GEMMs fill the default budget (512 MB), the coarse
  // tiers keep only the narrow model-level GEMMs on the requant route
  // (an [M, 128] at 16 splits at most). All grow-only, held until
  // release_scratch(). Inline: it adds no symbol.
  static std::size_t forward_bytes_bound(int M, int k_act, int n_w,
                                         int k_w, bool accurate) noexcept
  {
    const std::size_t m = M > 0 ? (std::size_t)M : 0;
    const std::size_t ka = k_act > 0 ? (std::size_t)k_act : 0;
    const std::size_t nw = n_w > 0 ? (std::size_t)n_w : 0;
    const std::size_t kw = k_w > 0 ? (std::size_t)k_w : 0;
    std::size_t n = m * ka + 2 * m * (ka / 64) * 2;
    n += nw * kw + nw * (kw / 64) * 2;
    n += (std::size_t)kTuneRows * nw * 2;
    const std::size_t budget = (std::size_t)512 << 20;
    const std::size_t narrow = (std::size_t)16 * ((m + 63) / 64 * 64) * 128 * 4;
    n += accurate ? budget : (narrow < budget ? narrow : budget);
    return n;
  }
  // Whether a GEMM takes the fast route: the int8 route's row floor, as
  // the accurate tier's, and a K fast_group admits.
  bool fast_accepts(int M, int N, int K) const
  {
    return _on && fast_group(K) > 0 && M >= _min_m && N >= 16 &&
           (_tier != Tier::kToken || (N % 4) == 0);
  }
  // y[M,N] = x[M,K] @ w^T for a weight in the fast layout at group gc.
  // False with nothing encoded where fast_accepts declines.
  bool gemm_fast(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& x, std::size_t xe,
                 const metal_compute::SharedBuffer& codes,
                 const metal_compute::SharedBuffer& scales, int gc,
                 const metal_compute::SharedBuffer& b,
                 const metal_compute::SharedBuffer& y, std::size_t ye,
                 int M, int N, int K);
  // THE ACTIVATION ONCE for several fast GEMMs over one input (q|k|v,
  // gate|up): quantize_fast fills the scratch, and each
  // gemm_fast_quantized after it reads that scratch rather than running
  // its own rotated quantizer -- three passes saved a block on a DiT.
  // The second declines (nothing encoded) unless M, K and gc are the
  // ones quantized.
  bool quantize_fast(metal_compute::ComputeEncoder& enc,
                     const metal_compute::SharedBuffer& x, std::size_t xe,
                     int M, int K, int gc);
  bool gemm_fast_quantized(metal_compute::ComputeEncoder& enc,
                           const metal_compute::SharedBuffer& codes,
                           const metal_compute::SharedBuffer& scales, int gc,
                           const metal_compute::SharedBuffer& b,
                           const metal_compute::SharedBuffer& y,
                           std::size_t ye, int M, int N, int K);
  // ComfyUI's int8 CONVROT weights on the TOKEN tier, run as stored: the
  // activation rotated as they were -- comfy_kitchen's normalized regular
  // Hadamard over each 256-block of K -- and quantized per row into the
  // scratch gemm_fast_quantized(gc = kTokenGroup) then reads, with the
  // weight's codes and its row scales. False, nothing encoded, where the
  // tier is not token, K % 256 != 0, or the kernel is missing (an older
  // host: convrot_available() false).
  bool quantize_token_convrot(metal_compute::ComputeEncoder& enc,
                              const metal_compute::SharedBuffer& x,
                              std::size_t xe, int M, int K);
  bool convrot_available() const { return _fn_tok_cr.valid(); }
  // ...the down projection of a SwiGLU from its two halves, as
  // gemm_signed_swiglu (the product rounded to the element type first).
  bool gemm_fast_swiglu(metal_compute::ComputeEncoder& enc,
                        const metal_compute::SharedBuffer& gate,
                        std::size_t ge,
                        const metal_compute::SharedBuffer& up,
                        std::size_t ue,
                        const metal_compute::SharedBuffer& codes,
                        const metal_compute::SharedBuffer& scales, int gc,
                        const metal_compute::SharedBuffer& y,
                        std::size_t ye, int M, int N, int K);
  // A g64 AFFINE pack (scales and biases [N, K/64]) into the fast
  // layout: each element dequantized, its group rotated and requantized.
  // At bits 8 the codes are u8 [N, K] and codes_out may BE codes_in (in
  // place). At bits 4 they are nibbles [N, K/2] (byte b holding k = 2b
  // low, 2b + 1 high) and codes_out [N, K] -- twice their size -- must
  // be a buffer of its own. out_scales is [N, K/gc]. False with nothing
  // encoded where fast_converts(bits, gc) is not.
  bool encode_to_fast(metal_compute::ComputeEncoder& enc,
                      const metal_compute::SharedBuffer& codes_in,
                      const metal_compute::SharedBuffer& scales,
                      const metal_compute::SharedBuffer& biases,
                      const metal_compute::SharedBuffer& codes_out,
                      const metal_compute::SharedBuffer& out_scales, int gc,
                      int N, int K, int bits = 8) const;
  // Whether a g64 pack at `bits` (4 or 8) converts into the layout at
  // group gc: the tier's kernels are there, and at w4 its converters
  // (newer than the tier: a host without them keeps a w4 weight on the
  // accurate tier).
  bool fast_converts(int bits, int gc) const;
  // A W4 PACK CONVERTED PER CALL. Its layout is twice its size, so a w4
  // weight on the fast or token tier is HELD as its pack and converted
  // (encode_to_fast, bits 4) into the context's weight scratch right
  // before each GEMM that reads it: fast_scratch_codes() /
  // fast_scratch_scales(), grown to the largest weight asked and valid
  // until the next call. Dispatches on one encoder run in order -- which
  // the activation scratch already relies on -- so each GEMM reads its
  // weight before the next conversion overwrites it. The same values as
  // a layout held, so every route over them agrees. MEASURED, a block's
  // six K = 3072 projections at 4096 rows on the M5 Ultra
  // (gemm_i8.a_conversion_overlaps_the_gemm): 0.96x the held layout's
  // GEMMs on the token tier, 0.99x fast -- and converting one GEMM AHEAD
  // in a concurrent scope bought 0.4% / 2%, so it is not done. False
  // with nothing encoded where encode_to_fast declines or the scratch
  // cannot be had.
  bool encode_to_fast_scratch(metal_compute::ComputeEncoder& enc,
                              const metal_compute::SharedBuffer& codes,
                              const metal_compute::SharedBuffer& scales,
                              const metal_compute::SharedBuffer& biases,
                              int gc, int N, int K);
  const metal_compute::SharedBuffer& fast_scratch_codes() const
  {
    return _wq;
  }
  const metal_compute::SharedBuffer& fast_scratch_scales() const
  {
    return _ws;
  }
  // What that scratch holds for an [N, K] weight at group gc: a plan's
  // figure for a model whose w4 weights take the tier.
  static std::size_t fast_scratch_bytes(int N, int K, int gc)
  {
    return (std::size_t)N * (std::size_t)K + fast_scale_bytes(N, K, gc);
  }
  // A DENSE weight [N, K] (the element type) into the fast layout.
  bool encode_quant_fast(metal_compute::ComputeEncoder& enc,
                         const metal_compute::SharedBuffer& w,
                         const metal_compute::SharedBuffer& codes_out,
                         const metal_compute::SharedBuffer& out_scales,
                         int gc, int N, int K) const;
  // A fast-layout weight back to dense [N, K] in the element type (the
  // rotation undone), for the routes that do not read the layout.
  bool encode_dequant_fast(metal_compute::ComputeEncoder& enc,
                           const metal_compute::SharedBuffer& codes,
                           const metal_compute::SharedBuffer& scales, int gc,
                           const metal_compute::SharedBuffer& out, int N,
                           int K) const;

  // A SwiGLU's DOWN PROJECTION from its two halves: gemm_signed over
  // silu(gate) * up without storing the product -- the quantizer reads
  // gate and up ([M, K] at element offsets ge / ue) and rounds their
  // product to the element type exactly as swiglu_v4_f16 stores it, so
  // the result is that kernel followed by gemm_signed, less one write and
  // one read of the [M, K] product. False with nothing encoded where
  // gemm_signed would decline or the fused quantizer is not there.
  bool gemm_signed_swiglu(metal_compute::ComputeEncoder& enc,
                          const metal_compute::SharedBuffer& gate,
                          std::size_t ge,
                          const metal_compute::SharedBuffer& up,
                          std::size_t ue,
                          const metal_compute::SharedBuffer& codes,
                          const metal_compute::SharedBuffer& scales,
                          const metal_compute::SharedBuffer& zb, int bits,
                          const metal_compute::SharedBuffer& y,
                          std::size_t ye, int M, int N, int K);

  // Whether a weight of this shape would take gemm_signed at `M` rows --
  // the gate a caller asks BEFORE choosing between it and a dense route.
  //
  // THE ROW FLOOR is the requant route's (1024) by default, so i8_gemm
  // runs the same GEMMs lossy whichever layout a weight is in -- what
  // changes is how fast and how close. VPIPE_I8_SIGNED_MIN_M lowers it,
  // and below the requant floor the signed GEMM is the FASTEST route a
  // w8 weight has: MEASURED on the M5 Pro (gemm_i8.w8_against_bf16_
  // routes, 3072 / 12288 x 3072 / 4096 / 9216) against its dequant +
  // matmul2d, 1.2-2.4x at 64-768 rows, and against bf16's own matmul2d
  // 1.05-3.7x; still 1.15-1.2x bf16 at 16 and 32 rows. But it is a
  // precision choice too: those rows are a model's TEXT stream, whose
  // error reaches every block. MEASURED at 16 rows: FLUX.2-klein-4B's
  // 1024^2 forward 1.5% faster for 3.43e-2 -> 3.57e-2 of distance from
  // the dense forward, and Z-Image's with a 64-token caption 3.84e-2 ->
  // 5.21e-2 (its caption refiner then runs int8), past what the route it
  // replaced cost (4.50e-2). So it is opt-in.
  bool signed_accepts(int M, int N, int K, int bits) const
  {
    const int floor = _signed_min_m > 0 ? _signed_min_m : _min_m;
    return signed_available(bits) && M >= floor && K >= 1024 && N >= 16 &&
           (K % 64) == 0;
  }

  // Encode the conversion of an affine weight [N, K] to the signed
  // layout. `codes_out` may be `codes_in` and `zb` may be `biases`: each
  // thread reads its own words and zero point before writing them.
  bool encode_to_signed(metal_compute::ComputeEncoder& enc,
                        const metal_compute::SharedBuffer& codes_in,
                        const metal_compute::SharedBuffer& scales,
                        const metal_compute::SharedBuffer& biases,
                        const metal_compute::SharedBuffer& codes_out,
                        const metal_compute::SharedBuffer& zb, int bits,
                        int N, int K) const;

  // Encode the dequant of a signed-layout weight into a dense [N, K]
  // element-type matrix -- the operand of a dense matmul2d GEMM, for the
  // rows gemm_signed declines.
  bool encode_dequant_signed(metal_compute::ComputeEncoder& enc,
                             const metal_compute::SharedBuffer& codes,
                             const metal_compute::SharedBuffer& scales,
                             const metal_compute::SharedBuffer& zb, int bits,
                             const metal_compute::SharedBuffer& out, int N,
                             int K) const;

  // Drop the grow-only act/weight scratches (they re-grow on demand at the
  // next gemm()), and the untuned shapes recorded since the last replay
  // with the weight references they hold. Call between generations on a
  // memory-bounded box so the idle scratch doesn't crowd out a large
  // downstream allocation.
  void release_scratch();

  // ---- split-K over the 512-groups ---------------------------------
  //
  // int8 and split-K answer DIFFERENT deficits, and a deep-K GEMM has
  // both. int8 doubles the matrix pipe's rate; a split fixes OCCUPANCY --
  // a single-op reduction over a deep K leaves only (M/64)*(N/64)
  // threadgroups each walking one long serial contraction, and the units
  // stall with nothing else in flight. The int8 pipe is not exempt: the
  // prototype measured the ff-down at K=12288 running 18.5 TOPS against
  // 22.5-23.6 at K=4096, the same cliff the f16 path has.
  //
  // It is nearly free to add here because the g512 kernel ALREADY walks K
  // as independent 512-groups, each scaled by its own pair before it joins
  // a float accumulator -- so a plane is a contiguous range of that loop
  // and the fold is the rest of the sum. Splitting by GROUP INDEX (not by
  // K) keeps every plane boundary on a 512 multiple for free.
  //
  // MEASURED on an M5 at H3's fc2 shape, N=7168 K=14336 (gemm_i8.splitk),
  // against this class's own single-op arm:
  //
  //     M     single   best split      gain
  //    256    9.15     15.22 (S=7)    1.66x
  //   1024   12.86     16.02 (S=2)    1.25x
  //   2560   12.99     15.96 (S=2)    1.23x
  //
  // NUMERICS: a reassociation and a weaker one than the dense f32-plane
  // split-K's, because the per-group summand is bit-IDENTICAL whichever
  // plane owns it -- only the order of the G float adds moves. Measured
  // 0.02-0.03% of outputs differing by at most ~1 f16 ulp, against the
  // dense path's own 0.48-0.60%.

  // Pick the split for this shape by MEASURING, and cache it per
  // (N, K, row bucket).
  //
  // CALL WITH NO ENCODER OPEN: it runs its own command streams. `run(enc)`
  // must encode the CALLER'S normal GEMM -- the same call the forward
  // makes -- so what is compared is by construction what will run; the
  // flags below steer it. Same contract as MmaSplitK::tune, deliberately:
  // a caller that already tunes one can tune the other beside it.
  //
  // NOTE the S that wins is NOT the one the dense chooser's kc heuristic
  // would pick, and it moves with M: at K=14336 that rule says S=7, where
  // S=2 wins once M >= 1024 and S=14 LOSES (0.95x). The base grid is
  // (N/64)*(M/64) threadgroups, so a wide N and a large M already fill the
  // machine; and the fold reads S*M*N*4 bytes, 411 MB at S=14/M=1024. So
  // this is measured rather than derived.
  //
  // `caller` names what `run` encodes (the family and its route: "h3-dit
  // mma128x256 w4g64"), which is the part of the question this context
  // cannot see. Given one, the answer is filed in the session and read
  // back by the next context to ask it (tuned-answers.h); without one it
  // is this context's alone, as before.
  template <class Run>
  int tune(metal_compute::MetalCompute* mc, int K, int N, int M, Run&& run,
           std::string_view caller = {})
  {
    if (!_on || mc == nullptr || !split_available()) { return 0; }
    const int key = m_key_(M);
    for (const Tuned& t : _tuned) {
      if (t.N == N && t.K == K && t.M == key) { return t.splits; }
    }
    KernelTuningCache* const answers = caller.empty() ? nullptr : _answers;
    const TuneKey skey = split_key_(caller, 0, N, K, M);
    int kept = 0;
    if (find_tuned(answers, skey, kept)) {
      _tuned.push_back(Tuned{N, K, key, kept});
      return kept;
    }
    const std::vector<int> cands = tune_candidates(M, N, K);
    if (cands.size() < 2) {
      _tuned.push_back(Tuned{N, K, key, 0});
      store_tuned(answers, skey, 0);
      return 0;
    }
    const int w = autotune_vote((int)cands.size(), /*rounds=*/3,
        /*reps_for_us=*/1,
        [&](int i) {
          bypass_split = (cands[(std::size_t)i] == 0);
          force_splits = bypass_split ? 0 : cands[(std::size_t)i];
          const double t = autotune_time(mc, 1, run);
          bypass_split = false;
          force_splits = 0;
          return t;
        });
    const int splits = cands[(std::size_t)w];
    _tuned.push_back(Tuned{N, K, key, splits});
    store_tuned(answers, skey, splits);
    return splits;
  }

  // DEFERRED TUNING, for a caller that has no convenient closure.
  //
  // tune() above wants the caller's own GEMM as a thunk, which suits a
  // family that already tunes something (H3 hands it the same
  // dispatch_row_bands_ its forward runs). The others do not have one to
  // hand: their weights may still be streaming when the shape is first
  // known, and the dense operand a quantized checkpoint contracts is a
  // dequant scratch that only exists INSIDE an encoder.
  //
  // So gemm() records a shape it has not tuned, and this replays it later
  // -- at any point with no encoder open -- using the quantized scratches
  // that call already left behind. Nothing is allocated but a destination,
  // nothing is read from the caller, and what is measured is exactly the
  // part that differs: the candidates share the two quantize passes, so
  // excluding them removes variance from the vote rather than hiding
  // anything.
  //
  // The scratch contents are the LAST shape's by then, which does not
  // matter -- an int8 GEMM's time does not depend on its data -- and the
  // buffers are grow-only, so a recorded shape is one they were already
  // sized for.
  //
  // Cheap and idempotent: returns immediately when nothing is pending.
  // Call it between forwards, or at the end of one.
  void tune_pending(metal_compute::MetalCompute* mc);

  // Whether the split kernels loaded at all (the fold lives in a different
  // metallib, so it can be absent independently).
  bool split_available() const
  {
    return _fn_gemm_sk.valid() && _fn_fold.valid() && _max_splits > 0;
  }

  // Every split this shape admits: G divisible for balanced planes, and a
  // row block that still fills the 64-row tile.
  std::vector<int> split_candidates(int M, int N, int K) const;

  // WHERE THE SPLIT IS FOR: the deep-K cliff. A single-op contraction
  // past it runs at about half rate (the occupancy deficit described
  // above), and the split is the cure. Short of it there is nothing to
  // cure, so an untuned shape there runs UNSPLIT, and the tuner weighs
  // only the single op against S=2 -- cheap, and two candidates leave a
  // one-sample vote little room to pick wrong.
  //
  // MEASURED on the M5 Pro, g64 requant, S=2's rate against the single
  // op's (MxNxK):
  //   YuE2 at 5347 rows, Nx2048 and 2048x6144       0.67-0.90x
  //   Qwen-Image-2.1 / Z-Image at 4096 rows, K 3840 / 4096   0.75-0.83x
  // and the native w4/w8 route at Krea-2 / FLUX.2 shapes, 4096 rows,
  // K 4096 / 6144: 0.86-0.92x at five of six (the sixth, 4096^3 w4,
  // disagreed between its two kernel modes). Over a YuE2 song that was
  // 1258 ms per evaluation against 1130 unsplit.
  //
  // AND PAST IT TOO, on the g64 register kernels: the cliff was the
  // drained g512 kernel's. MEASURED on the M5 Pro (gemm_i8.split_rule_
  // sweep: the requant route and the signed layout, 512-8192 rows, every
  // deep-K projection in tree -- K 9216 / 10240 / 12288 / 14336 / 16384),
  // the single op won at every shape from 1024 rows, the i8 floor, with
  // S=2 at 0.92-0.99x; the only splits that won were S=3 at 1024 rows x
  // 3072 x 9216 (1.07-1.10x) and S=2 at H3's fc2 at 1024 rows (1.02x).
  // So an UNTUNED shape runs unsplit at every K -- it was S=2 from 9216,
  // which cost 7-8% at 4096 rows (bf16 under i8, 3072-4096 x 9216-12288)
  // on every route the tuner never reached, the native one among them.
  // split_min_k() now only widens the TUNER's candidates (all balanced
  // splits from it, S=2 alone below). Default 9216; VPIPE_I8_SPLIT_MIN_K
  // moves it, and VPIPE_I8_UNTUNED_SPLIT=2 restores the old untuned S=2
  // from it (for an A/B).
  int split_min_k() const { return _split_min_k; }

  // What tune() and tune_pending() weigh for a shape: the single op, and
  // below split_min_k() S=2 alone, at or past it every split the shape
  // admits.
  std::vector<int> tune_candidates(int M, int N, int K) const;

  // Set by tune() around each candidate; see MmaSplitK for the pattern.
  bool bypass_split = false;
  int  force_splits = 0;

  // The width this shape would use right now. Test-only: a split is not
  // observable from the OUTPUT (it is a reassociation), so a test that
  // wants to know the tuner's answer took effect has to ask.
  int plan_for_test(int M, int N, int K) const { return plan_(M, N, K); }
  // Test-only: shapes recorded and not yet tuned, and whether a shape
  // has a tuned answer (a tuned 0 and the untuned default 0 are the same
  // plan, so the plan cannot say).
  int pending_for_test() const { return (int)_pending.size(); }
  // Test-only: shapes this context measured itself (tune_pending's
  // votes), against those it read from the session's answers.
  int measured_for_test() const { return _measured; }
  bool tuned_for_test(int M, int N, int K) const
  {
    const int key = m_key_(M);
    for (const Tuned& t : _tuned) {
      if (t.N == N && t.K == K && t.M == key) { return true; }
    }
    return false;
  }

  // The quantization group on both operands (64; 32, 128 or 512 under
  // the A/B).
  int group() const { return _group; }

 private:
  // accepts() and gemm() with the rotation decided per call: `rot`
  // takes the rotated quantizers, else the plain pair.
  bool accepts_(int M, int N, int K, bool rot) const
  {
    if (!_on || M < _min_m || K < 1024 || N < 16) { return false; }
    const int KP = kpad_(K);
    if (KP == K) { return true; }
    if (!(rot ? _fn_quant_r_pad : _fn_quant_pad).valid()) { return false; }
    return (KP - K) * 100 <= K * _max_pad_pct;
  }
  bool gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t xe,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer& b,
             const metal_compute::SharedBuffer& y, std::size_t ye, int M,
             int N, int K, bool rot);
  metal_compute::MetalCompute* _mc = nullptr;
  bool _on = false;
  bool _bf16 = false;
  int _group = 64;                   // VPIPE_I8_GROUP: 32|64|128|512
  int _native = 12;                  // VPIPE_I8_NATIVE: 8 | 4 mask
  int _signed = 12;                  // VPIPE_I8_SIGNED: 8 | 4 mask
  // The signed layout: its activation quantizer, conversion, dequant and
  // GEMM (+ split twins).
  metal_compute::ComputeFunction _fn_quant_s8, _fn_to_s8, _fn_to_s4,
      _fn_quant_swiglu_s8;
  // The fast tier's kernels, per group (index 0/1/2 = 256/512/1024,
  // 3 = the token tier's).
  struct FastFns {
    metal_compute::ComputeFunction quant, swiglu, to_fast, dequant, gemm;
    metal_compute::ComputeFunction to_fast4;   // from a w4 pack; optional
  };
  FastFns _fast[5];                  // [4]: the token tier, rotated
  int _tok_wm = 4, _tok_wn = 2;      // the token GEMM's simdgroup grid
  bool _tok_db = false;              // ...its double-buffered K loop
  bool _tok_kernel_env = false;      // VPIPE_I8_TOKEN_KERNEL / _TILE set
  bool _tok_rot = false;
  int _tok_rot_env = -1;             // VPIPE_I8_TOKEN_ROT; -1 = caller's
  bool _g64_rot = false;             // gemm() takes the _r quantizers
  int _g64_rot_env = -1;             // VPIPE_I8_ACC_ROT; -1 = caller's
  metal_compute::ComputeFunction _fn_quant_r, _fn_quant_r_pad;
  int tok_wm_(int gc) const { return gc == kTokenGroup ? _tok_wm : 0; }
  const FastFns* fast_fns_(int gc) const;
  Tier _tier = Tier::kAccurate;
  int _tier_env = -1;                // VPIPE_I8_TIER; -1 = the caller's
  int _fast_group = 512;             // VPIPE_I8_FAST_GROUP
  int _fast_bm = 128, _fast_bn = 64; // the coarse GEMM's tile
  // What quantize_fast last left in the scratch (0 = nothing usable).
  int _fq_m = 0, _fq_k = 0, _fq_gc = 0;
  metal_compute::ComputeFunction _fn_tok_cr;   // quant_i8_tok_cr
  metal_compute::ComputeFunction _fn_dq_s8, _fn_dq_s4;
  metal_compute::ComputeFunction _fn_s8_w8, _fn_s8_w4;
  metal_compute::ComputeFunction _fn_s8_w8_sk, _fn_s8_w4_sk;

  // gemm_affine's per-call conversion targets: codes and zero points.
  metal_compute::SharedBuffer _sc, _szb;
  metal_compute::ComputeFunction _fn_quant_u8, _fn_wsum8, _fn_wsum4;
  metal_compute::ComputeFunction _fn_u8q_w8, _fn_u8q_w4;
  metal_compute::ComputeFunction _fn_u8q_w8_sk, _fn_u8q_w4_sk;
  // Native-path scratches: u8 activations [M,K], their row-group sums
  // [M,G] i16, and the weight's group sums [N,G] i16 (the activation
  // scales share _as).
  metal_compute::SharedBuffer _xu, _xs, _wsum, _uf, _chi, _clo;
  int _cmode = 1;                    // VPIPE_I8_NATIVE_CMODE (0 | 1)
  metal_compute::ComputeFunction _fn_u8q_w8_cm, _fn_u8q_w4_cm;
  metal_compute::ComputeFunction _fn_u8q_w8_cm_sk, _fn_u8q_w4_cm_sk;

 public:
  // The form of the -128*Q correction: 0 = per group in integer, 1 =
  // folded into the post-loop matmul as an f16 hi/lo pair. 1 is the
  // default -- MEASURED never slower, 4-8% faster where the staged
  // column it drops was the last one. Test knob.
  void set_native_cmode(int m) { _cmode = m == 1 ? 1 : 0; }
  int native_cmode() const { return _cmode; }

 private:
  int _min_m = 1024;
  int _signed_min_m = 0;             // VPIPE_I8_SIGNED_MIN_M; 0 = _min_m
  int _max_pad_pct = 10;   // VPIPE_I8_MAX_PAD_PCT
  metal_compute::ComputeLibrary _lib_q, _lib_g;
  metal_compute::ComputeFunction _fn_quant, _fn_quant_pad, _fn_gemm;
  metal_compute::ComputeLibrary   _lib_elt;
  metal_compute::ComputeFunction  _fn_gemm_sk, _fn_fold;
  metal_compute::ComputeFunction  _fn_fold_bias;
  // Max f32 plane scratch, bytes (VPIPE_I8_SPLITK_MAX_MB, default 512).
  // A constraint on the SEARCH, not a veto after it: past the budget the
  // next-narrower split still runs, and past every one the single-op GEMM
  // does -- slower, allocation-free.
  std::size_t _plane_budget = 0;
  int _max_splits = 16;              // VPIPE_I8_SPLITK_MAX_S; 0 disables
  int _split_min_k = 9216;           // VPIPE_I8_SPLIT_MIN_K; see above
  int _untuned_split = 0;            // VPIPE_I8_UNTUNED_SPLIT (A/B)
  metal_compute::SharedBuffer _planes;

  struct Tuned { int N, K, M, splits; };
  std::vector<Tuned> _tuned;
  // The session's kept answers (tuned-answers.h), or null; and the
  // shapes already asked of it and not found, so a forward's hundred
  // calls to an untuned shape ask once.
  KernelTuningCache* _answers = nullptr;
  struct Asked { int N, K, M; };
  std::vector<Asked> _asked;
  int _measured = 0;
  // Shapes a route has run but nobody has tuned. Bounded, because a
  // caller sweeping sequence lengths must not grow it without limit. The
  // requant route replays over its own scratches; the native and signed
  // routes read the caller's weight, so the record holds references to
  // it until the replay (a streamed slot or a resident tensor -- either
  // outlives the forward that recorded it).
  enum class Route { kRequant, kNative, kSigned };
  struct Pending {
    int N, K, M;
    Route route = Route::kRequant;
    int bits = 8;
    metal_compute::SharedBuffer codes, scales, zb;
  };
  std::vector<Pending> _pending;
  // Record a shape for tune_pending unless it is settled or queued.
  void record_(Pending p);
  // Before a route plans a shape it has not settled: the session's
  // answer to it, when one is kept, settles it -- the first call then
  // runs the tuned split, and nothing is recorded to measure.
  void settle_(Route route, int bits, int N, int K, int M);
  // The session key of a split answer. `route` is a Route's name or a
  // tune() caller's; everything else that moves the vote is this
  // context's own state: the element type, the group, the split
  // limits, the native correction's form.
  TuneKey split_key_(std::string_view route, int bits, int N, int K,
                     int M) const
  {
    return TuneKey("i8-split/1")("route", route)("bits", bits)
        ("bf16", _bf16)("group", _group)("cmode", _cmode)
        ("max_s", _max_splits)("min_k", _split_min_k)
        ("plane_mb", (long long)(_plane_budget >> 20))
        ("N", N)("K", K)("M", m_key_(M));
  }
  metal_compute::SharedBuffer _tune_y;
  // Rows the deferred tuner MEASURES at, whatever the shape's real M. A
  // destination is M*N*2 bytes and a video forward's M is six figures --
  // 1.5 GB at H3's fc2 -- where the answer is already flat above one row
  // block (S=2 won at both 1024 and 2560 rows there). So it measures a
  // bounded stand-in and keys the answer to the real row bucket.
  static constexpr int kTuneRows = 2048;
  // What one timed sample of the deferred tuner should last, at least.
  static constexpr double kTuneSampleMs = 4.0;
  // Encode ONE candidate's GEMM (+ fold) over the existing scratches
  // (and, for the native and signed routes, the recorded weight).
  void encode_tuned_(metal_compute::ComputeEncoder& enc, const Pending& p,
                     int M, int KP, int S);
  // The native and signed GEMMs alone, single op or split in row blocks
  // -- shared by the routes and by the replay. `y` is the element-type
  // output (unused when split: the planes are folded into it after).
  void encode_native_gemm_(metal_compute::ComputeEncoder& enc,
                           const metal_compute::SharedBuffer& codes,
                           const metal_compute::SharedBuffer& scales,
                           const metal_compute::SharedBuffer& qbias,
                           int bits, const metal_compute::SharedBuffer& b,
                           const metal_compute::SharedBuffer& y,
                           std::size_t ye, int M, int N, int K, int S);
  void encode_signed_gemm_(metal_compute::ComputeEncoder& enc,
                           const metal_compute::SharedBuffer& codes,
                           const metal_compute::SharedBuffer& scales,
                           const metal_compute::SharedBuffer& zb, int bits,
                           const metal_compute::SharedBuffer& b,
                           const metal_compute::SharedBuffer& y,
                           std::size_t ye, int M, int N, int K, int S);
  // Row buckets, so a forward whose row count wobbles does not re-tune --
  // and so the answer is attached to the count it was measured at.
  static int m_key_(int M)
  {
    if (M <= 512) { return 512; }
    if (M <= 1024) { return 1024; }
    if (M <= 2048) { return 2048; }
    return 4096;
  }
  // Rows whose S planes fit the budget, floored at the 64-row tile.
  int rows_per_block_(int S, int N) const
  {
    const std::size_t per_row = (std::size_t)S * (std::size_t)N * 4u;
    if (per_row == 0) { return 0; }
    const std::size_t r = _plane_budget / per_row;
    return (int)((r / 64u) * 64u);
  }
  // ...and for a GEMM of M rows: no block wider than the GEMM (rounded
  // to the 64-row tile), which is the same single block below the
  // budget's rows -- the same dispatches -- but planes sized for the
  // rows there are rather than the budget. Sized by the budget alone, a
  // 64-wide proj_out tuned on the fast tier held 512 MB of planes all run.
  int rows_per_block_(int S, int N, int M) const
  {
    const int r = rows_per_block_(S, N);
    const int m = ((M > 0 ? M : 0) + 63) / 64 * 64;
    return r < m ? r : (m > 0 ? m : r);
  }
  // The split this call will use: the tuner's answer when there is one,
  // else the measured-safe default. See the note on `tune`.
  int plan_(int M, int N, int K) const;

  // K rounded up to a whole number of groups. The int8 GEMM contracts in
  // group-wide chunks, so a K that is not a multiple of the group has no
  // chunk to sit in; the quantizer zero-fills up to here instead, which is
  // exact -- the very group a caller padding its own buffers would hand
  // over; zeros add nothing to the dot product, rotated or not, since both
  // operands carry them in the same places -- and costs (kpad-K)/K extra
  // int8 MACs: 4.8% at H3's hidden 5376 under the 512 group, none at all
  // under 64.
  int kpad_(int K) const
  {
    return ((K + _group - 1) / _group) * _group;
  }
  // Grow-only scratches: xq[M,K] i8 + as[M,G] scales (activations),
  // wq[N,K] i8 + ws[N,G] scales (per-call weight re-quant, and a w4
  // pack's per-call conversion into the fast layout --
  // encode_to_fast_scratch). Scales are the element type (2 bytes either
  // way; the token tier's f32 per row), so the byte sizes are
  // format-agnostic.
  metal_compute::SharedBuffer _xq, _as, _wq, _ws;
};

}  // namespace genai
}  // namespace vpipe

#endif
