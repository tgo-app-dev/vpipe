#ifndef VPIPE_GENERATIVE_MODELS_SHARED_I8_GEMM_H
#define VPIPE_GENERATIVE_MODELS_SHARED_I8_GEMM_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/kernel-autotune.h"

#include <cstddef>
#include <cstdlib>
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
// THE GROUP IS 64 on both operands (quant_f16_i8_row_g64 +
// gemm_i8i8_sc_f16_n64_g64rs). It was 512, and the reason it could not be
// finer was the kernel, not the quantization: the drained per-group kernel
// paid a threadgroup round-trip per group, which MEASURED g64 at 6.4 TOP/s
// against g512's 19.3 -- below f16. The register-resident kernel keeps the
// partial in a cooperative tensor and stages each strip of scales once,
// and at that design g64 runs 17.4 / 16.9 TOP/s (shallow / deep K) against
// the same kernel's 21.3 / 19.3 at 512 -- so the finer group costs 12-18%
// of the coarser one's rate and buys the precision a 64-wide group has
// over a 512-wide one on every outlier channel (gemm_i8.group_size,
// gemm_i8.g64_register_acc). VPIPE_I8_GROUP=512 selects the old pair for
// an A/B, and 32 / 128 the same register design one step finer or
// coarser; anything else is 64.
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
// rel-L2, not a trend; the unit rows are monotone.)
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
    if (!_on || M < _min_m || K < 1024 || N < 16) { return false; }
    const int KP = kpad_(K);
    if (KP == K) { return true; }
    if (!_fn_quant_pad.valid()) { return false; }
    return (KP - K) * 100 <= K * _max_pad_pct;
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
  // x 1024, both widths -- the weight is quantized once, not twice. The
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

  // Drop the grow-only act/weight scratches (they re-grow on demand at the
  // next gemm()). Call between generations on a memory-bounded box so the
  // idle scratch doesn't crowd out a large downstream allocation.
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
  template <class Run>
  int tune(metal_compute::MetalCompute* mc, int K, int N, int M, Run&& run)
  {
    if (!_on || mc == nullptr || !split_available()) { return 0; }
    const int key = m_key_(M);
    for (const Tuned& t : _tuned) {
      if (t.N == N && t.K == K && t.M == key) { return t.splits; }
    }
    const std::vector<int> cands = tune_candidates(M, N, K);
    if (cands.size() < 2) {
      _tuned.push_back(Tuned{N, K, key, 0});
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
  // 1258 ms per evaluation against 1130 unsplit. Default 9216 (the cliff
  // sits between 9k and 10k); VPIPE_I8_SPLIT_MIN_K moves it, and 0
  // restores the old rule (every untuned shape at S=2, every width
  // tuned).
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

  // The quantization group on both operands (64; 32, 128 or 512 under
  // the A/B).
  int group() const { return _group; }

 private:
  metal_compute::MetalCompute* _mc = nullptr;
  bool _on = false;
  int _group = 64;                   // VPIPE_I8_GROUP: 32|64|128|512
  int _native = 12;                  // VPIPE_I8_NATIVE: 8 | 4 mask
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
  metal_compute::SharedBuffer _planes;

  struct Tuned { int N, K, M, splits; };
  std::vector<Tuned> _tuned;
  // Shapes gemm() has run but nobody has tuned. Bounded, because a caller
  // sweeping sequence lengths must not grow it without limit.
  std::vector<Tuned> _pending;
  metal_compute::SharedBuffer _tune_y;
  // Rows the deferred tuner MEASURES at, whatever the shape's real M. A
  // destination is M*N*2 bytes and a video forward's M is six figures --
  // 1.5 GB at H3's fc2 -- where the answer is already flat above one row
  // block (S=2 won at both 1024 and 2560 rows there). So it measures a
  // bounded stand-in and keys the answer to the real row bucket.
  static constexpr int kTuneRows = 2048;
  // What one timed sample of the deferred tuner should last, at least.
  static constexpr double kTuneSampleMs = 4.0;
  // Encode ONE candidate's GEMM (+ fold) over the existing scratches.
  void encode_tuned_(metal_compute::ComputeEncoder& enc, int M, int N,
                     int KP, int S);
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
  // The split this call will use: the tuner's answer when there is one,
  // else the measured-safe default. See the note on `tune`.
  int plan_(int M, int N, int K) const;

  // K rounded up to a whole number of groups. The int8 GEMM contracts in
  // group-wide chunks, so a K that is not a multiple of the group has no
  // chunk to sit in; the quantizer zero-fills up to here instead, which is
  // exact (zeros add nothing to the dot product and cannot move an absmax
  // scale) and costs (kpad-K)/K extra int8 MACs -- 4.8% at H3's hidden
  // 5376 under the 512 group, none at all under 64.
  int kpad_(int K) const
  {
    return ((K + _group - 1) / _group) * _group;
  }
  // Grow-only scratches: xq[M,K] i8 + as[M,G] scales (activations),
  // wq[N,K] i8 + ws[N,G] scales (per-call weight re-quant). Scales are the
  // element type (2 bytes either way), so the byte sizes are format-agnostic.
  metal_compute::SharedBuffer _xq, _as, _wq, _ws;
};

}  // namespace genai
}  // namespace vpipe

#endif
