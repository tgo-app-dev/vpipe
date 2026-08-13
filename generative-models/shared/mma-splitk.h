#ifndef VPIPE_GENERATIVE_MODELS_SHARED_MMA_SPLITK_H
#define VPIPE_GENERATIVE_MODELS_SHARED_MMA_SPLITK_H

// Split-K deep-reduction dense matmul2d for the LM prefill GEMMs (M5+).
//
// A single-op matmul2d reduction underutilizes the matrix units once K gets
// very deep: only N/BN threadgroups exist, each walking one long serial
// contraction, so the pipeline stalls with nothing else in flight. Every
// down_proj in this tree is in that regime -- it contracts over ffn, which
// is 2-4x the gate|up depth over a narrower N. The single-op kernel manages
// 4.4-6.8 TFLOP/s where the SAME weights contracted in the gate|up direction
// reach 10-12. A transpose control at identical FLOPs and identical weight
// bytes runs 10.9-11.6, which is what establishes this as contraction DEPTH
// rather than the narrow N (optiq_blocks.mlp_tile_probe).
//
// With the chunk-width chooser below, all four measured depths come back to
// 0.9-1.0 of their own gate|up rate (mlp_fuse.splitk_plan, M5):
//
//   K       N      single    planned            gain
//   12288   4096   8974      11134  S=6 kc2048  1.24x
//   14336   5376   6821      10697  S=7 kc2048  1.57x
//   17408   5120   4745      10363  S=8 kc2176  2.18x
//   21504   5376   4352      10444  S=12 kc1792 2.40x
//
// Splitting K into S chunks gives each chunk its own threadgroup plane
// (grid.z), multiplying the threadgroup count by S and shortening each
// reduction back into the fast regime. A fold then sums the planes.
//
// NUMERICS -- why this uses the f32-plane kernels and not the f16-plane ones
// the diffusion models use. Folding f16 planes costs one rounding per plane
// against the single-op path's one in total; measured at these shapes that
// perturbs 37.5% of outputs by up to ~5 ulp. That is fine for a rel-L2-
// verified DiT and is exactly the kind of drift that flips a near-tie argmax
// in a decoder held to greedy token-exact. With f32 planes + an f32 fold the
// output rounds to f16 exactly once, and the only remaining difference from
// the single-op path is where the f32 contraction was split -- a
// reassociation. That leaves 0.5-0.6% of outputs differing by a single f16
// ulp, smaller than the steel-vs-matmul2d difference this codebase already
// accepts between prefill paths.
//
// MEASURED at the widths the chooser now reaches (S=6..12, M5, cooled so the
// clock holds at ~1.3-1.5 GHz): 0.48-0.60% of outputs differ from the
// single-op result by at most one f16 ulp (max |d| 9.8e-4, max rel 1.9e-3)
// -- the same distribution the 4-way split shipped with. Splitting DEEPER
// did not move it, which is what the f32 planes buy: the fold rounds once
// whatever S is.
//
// Not verified here: the two OptiQ checkpoints this was built for do not fit
// in 16 GB, so their end-to-end greedy token-exact run against the single-op
// path is owed on the 64 GB box. VPIPE_LM_NO_SPLITK=1 is the A/B and the
// kill switch. Note the reach is WIDER than it was -- any depth >= 12288
// with an exact chunk now splits, where before only two hardcoded depths
// did -- so a model that never touched this path (Qwen3.5-9B, ffn 12288)
// now does.

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "generative-models/shared/kernel-autotune.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace vpipe::genai {

// Loaded kernels + the partial-plane scratch. One per model instance.
struct MmaSplitK {
  // THE SPLIT IS CHOSEN BY CHUNK WIDTH, not by a depth table.
  //
  // What a split is worth is set by how deep each chunk still is, and the
  // split COUNT is only how you get there. Measured on M5 (fanless, cooled
  // so the clock holds; `mlp_fuse.down_proj` prints the GPU clock beside
  // every number, because a throttled arm and a slow kernel are otherwise
  // the same reading):
  //
  //   K=17408, N=5120, M=4096   unsplit 4658 GF/s
  //                             kc4352 (S=4, what shipped)  5867
  //                             kc2176 (S=8)               10028
  //                             gate|up on the same weights 10660
  //
  // So the width that shipped recovered a quarter of the deficit and a
  // narrower chunk recovers nearly all of it. Hardcoding S=4 per depth also
  // meant every new checkpoint needed its own entry and got nothing until
  // it had one -- H3's DiT fc2 (14336) and Qwen3.5-9B's down (12288) both
  // ran unsplit for exactly that reason.
  //
  // The rule: walk the split counts whose chunk is EXACT, keep the ones with
  // a compiled kernel and a plane budget that fits, and take the chunk width
  // nearest kTargetKC. A depth with no usable candidate returns no plan and
  // the caller keeps its single-op dispatch, as before.
  //
  // M5 ONLY. On a non-matrix-core Mac the steel path is not sensitive to
  // contraction depth at all, and `enabled` is already false there.
  static constexpr int kTargetKC = 2048;
  static constexpr int kMaxS     = 16;
  // The split kernel's row tile. A row block below this cannot fill it.
  static constexpr int kRowTile  = 128;

  // Below this depth a split is not attempted. 12288 and 14336 (MiniMax-H3's
  // DiT fc2) are measured -- both gain ~2x; 9216, the next one down and the
  // only deep ffn on a checkpoint that fits in 16 GB, is not -- so the floor sits between them rather than
  // enabling a depth nothing has measured. VPIPE_LM_SPLITK_MIN_K lowers it
  // for that experiment.
  static constexpr int kMinSplitK = 12288;

  // ...and below this many ROWS. The deficit a split fixes is a large-M
  // phenomenon: at 128-256 rows the whole GEMM is small enough that the
  // single-op tile already runs 10-12 TFLOP/s and the split LOSES (0.77-1.11x
  // over four depths), while at 512 it wins on all four (1.09-1.68x) and at
  // 1024 by 1.5-2.1x. Without this floor a short chat turn pays it: a
  // 91-token prefill on Qwen3.5-9B measured 0.99 s split vs 0.38 s unsplit,
  // because the caller's matrix-core threshold is only 64 rows.
  static constexpr int kMinSplitM = 512;

  // Borrowed, not owned: ComputeLibrary is move-only and the caller holds it
  // as a member for the model's lifetime, which is also this struct's.
  metal_compute::ComputeLibrary* lib = nullptr;
  metal_compute::ComputeFunction fn_fold;
  metal_compute::SharedBuffer planes;
  bool enabled = false;
  // kc -> kernel, resolved on first use. A model asks for the same depth in
  // every layer, so this is looked up once per checkpoint and not per GEMM.
  // RESERVED at load: fn_for_kc hands out pointers INTO this vector, and a
  // reallocation would dangle every one of them. At most kMaxS-1 widths can
  // ever be probed (one per split count), so the reserve is exact.
  std::vector<std::pair<int, metal_compute::ComputeFunction>> by_kc;

  // Set by tune() while it measures one arm; plan() obeys them so the tuner
  // drives the CALLER'S OWN dispatch rather than a reimplementation of it.
  bool bypass = false;        // this call: no split, whatever plan() would say
  int  force_splits = 0;      // this call: exactly this split count

  // What tune() decided, per shape. `splits == 0` means it measured the
  // unsplit dispatch as the winner -- a real answer, not a missing entry,
  // which is why plan() must distinguish "tuned to none" from "untuned".
  struct Tuned { int N, K, M, splits; };
  std::vector<Tuned> tuned;

  void load(metal_compute::MetalCompute* mc,
            metal_compute::ComputeLibrary& lib_dense,
            metal_compute::ComputeLibrary& lib_elt)
  {
    if (mc == nullptr || !mc->supports_matrix_cores()) { return; }
    lib = &lib_dense;
    fn_fold = lib_elt.function("splitk_fold_f32_f16");
    enabled = fn_fold.valid() && lib_dense.valid();
    by_kc.reserve((std::size_t)kMaxS);
    if (const char* e = std::getenv("VPIPE_LM_NO_SPLITK")) {
      if (std::atoi(e) != 0) { enabled = false; }
    }
  }

  // How many rows of one split can be in flight inside the plane budget.
  // Planes are S f32 copies of [rows, N]; at S=12, N=5376 a full 9382-row
  // prefill would want 2.4 GB, which is not a thing to allocate on a 16 GB
  // box -- so encode() runs the split in row blocks of this size instead.
  static int rows_per_block(int S, int N)
  {
    const std::size_t per_row = (std::size_t)N * (std::size_t)S * 4;
    if (per_row == 0) { return 0; }
    const std::size_t rows = budget_bytes() / per_row;
    return (int)((rows / (std::size_t)kRowTile) * (std::size_t)kRowTile);
  }

  static int env_int_(const char* name, int def)
  {
    const char* e = std::getenv(name);
    if (e == nullptr) { return def; }
    const int v = std::atoi(e);
    return v > 0 ? v : def;
  }

  // The chunk kernel for `kc`, or nullptr. Names are formulaic, so which
  // widths exist is decided in dense_gemm_mma.metal and discovered here --
  // there is no second list to keep in sync.
  const metal_compute::ComputeFunction* fn_for_kc(int kc)
  {
    for (auto& e : by_kc) {
      if (e.first == kc) { return e.second.valid() ? &e.second : nullptr; }
    }
    if (lib == nullptr) { return nullptr; }
    metal_compute::ComputeFunction f = lib->function(
        "dense_gemm_mma_splitk32_n128x256_k" + std::to_string(kc) + "_f16");
    by_kc.emplace_back(kc, std::move(f));
    return by_kc.back().second.valid() ? &by_kc.back().second : nullptr;
  }

  struct Plan {
    const metal_compute::ComputeFunction* fn = nullptr;
    int splits = 0;
  };

  // Best split for this (K, N, M): exact chunk, kernel present, planes
  // within budget, width nearest the target. The budget matters as much as
  // the width -- planes are S f32 copies of the output, so a deeper split
  // that would be faster can be the one that does not fit, and then the next
  // best width is the right answer rather than no split at all.
  Plan plan(int K, int N, int M)
  {
    Plan best;
    if (!enabled || bypass) { return best; }
    if (force_splits > 0) {
      if (K % force_splits != 0) { return best; }
      const metal_compute::ComputeFunction* fn = fn_for_kc(K / force_splits);
      if (fn != nullptr) { best = Plan{fn, force_splits}; }
      return best;
    }
    // A tuned shape skips every heuristic below, including the floors: the
    // floors are guesses about where a split pays and the measurement is
    // not.
    for (const Tuned& t : tuned) {
      if (t.N == N && t.K == K && t.M == M) {
        if (t.splits <= 0) { return best; }
        const metal_compute::ComputeFunction* fn = fn_for_kc(K / t.splits);
        if (fn != nullptr) { best = Plan{fn, t.splits}; }
        return best;
      }
    }
    if (K < env_int_("VPIPE_LM_SPLITK_MIN_K", kMinSplitK)) { return best; }
    if (M < env_int_("VPIPE_LM_SPLITK_MIN_M", kMinSplitM)) { return best; }
    const int target = env_int_("VPIPE_LM_SPLITK_TARGET_KC", kTargetKC);
    const int forced = env_int_("VPIPE_LM_SPLITK_S", 0);
    int best_score = 0;
    for (int S = 2; S <= kMaxS; ++S) {
      if (K % S != 0) { continue; }
      if (forced > 0 && S != forced) { continue; }
      // Budget is a floor on the ROW BLOCK, not a veto on the split: encode()
      // walks M in blocks that fit, so a deep split costs more dispatches
      // rather than more memory. A block still has to fill the 128-row tile.
      if (rows_per_block(S, N) < kRowTile) { continue; }   // budget too tight
      const int kc = K / S;
      const metal_compute::ComputeFunction* fn = fn_for_kc(kc);
      if (fn == nullptr) { continue; }
      const int score = kc > target ? kc - target : target - kc;
      if (best.fn == nullptr || score < best_score) {
        best_score = score;
        best = Plan{fn, S};
      }
    }
    return best;
  }

  // Max partial-plane scratch, MB, via VPIPE_LM_SPLITK_MAX_MB.
  static std::size_t budget_bytes()
  {
    static const std::size_t b = []() {
      const char* e = std::getenv("VPIPE_LM_SPLITK_MAX_MB");
      const long v = (e != nullptr) ? std::atol(e) : 0;
      return (std::size_t)(v > 0 ? v : 512) * 1024u * 1024u;
    }();
    return b;
  }

  // Measure the unsplit dispatch against every split this shape admits, and
  // remember the winner for (N, K, M).
  //
  // `run(enc)` encodes the CALLER'S normal GEMM -- the same call the forward
  // makes, dequant and all -- and tune() toggles `bypass` / `force_splits`
  // around it. So what is compared is by construction what will run, and no
  // arm here is a reimplementation that could drift from the real path.
  //
  // Why measure at all, when plan() already has a rule: the rule is a
  // constant, and this decision has four inputs -- M, N*K, the tile, and the
  // GPU. MEASURED on M5: a split wins by 2.4x at (5376, 21504) and LOSES 7%
  // at (512, 13824), and the K floor that separates those two was fitted to
  // eight shapes on one machine. The width that a chunk should have is not
  // resolvable from N and K alone (the weight's cache residency decides it),
  // and a Pro/Max with a different cache would move the crossover without
  // moving anything this header can see.
  //
  // CALL WITH NO ENCODER OPEN: it runs its own command streams. Cost is
  // rounds x (1 + candidates) dispatches of one GEMM, once per shape, which
  // is why the caller tunes at the geometry it is about to run and caches.
  //
  // NOTE for a token-exact caller: a split changes the answer by ~1 ulp on
  // 0.5% of outputs, so a tuner free to choose differently on the next run
  // makes the output irreproducible. Tune once, then keep the result for the
  // process (which is what `tuned` does) -- do not re-tune per turn.
  template <class Run>
  int tune(metal_compute::MetalCompute* mc, int K, int N, int M, Run&& run)
  {
    if (!enabled || mc == nullptr) { return 0; }
    for (const Tuned& t : tuned) {
      if (t.N == N && t.K == K && t.M == M) { return t.splits; }
    }
    // Candidate 0 is "no split". The rest are every exact chunk with a
    // kernel and a row block that fits -- the same filter plan() applies,
    // minus the floors, which exist only to keep this search cheap.
    std::vector<int> cands{0};
    for (int S = 2; S <= kMaxS; ++S) {
      if (K % S != 0) { continue; }
      if (rows_per_block(S, N) < kRowTile) { continue; }
      if (fn_for_kc(K / S) == nullptr) { continue; }
      cands.push_back(S);
    }
    if (cands.size() < 2) {
      tuned.push_back(Tuned{N, K, M, 0});
      return 0;
    }
    const int w = autotune_vote((int)cands.size(), /*rounds=*/3,
        /*reps_for_us=*/1,
        [&](int i) {
          bypass = (cands[(std::size_t)i] == 0);
          force_splits = bypass ? 0 : cands[(std::size_t)i];
          const double t = autotune_time(mc, 1, run);
          bypass = false;
          force_splits = 0;
          return t;
        });
    const int splits = cands[(std::size_t)w];
    tuned.push_back(Tuned{N, K, M, splits});
    return splits;
  }

  // Encode y[M,N] = x[M,K] @ wdeq[N,K]^T through the split path, or return
  // false to let the caller run its single-op dispatch. `M` must already have
  // passed the caller's matrix-core row threshold.
  //
  // The plane scratch is S f32 copies of the output and so grows with M:
  // 352 MB at M=4096, S=4 on the 31B. These are the two models that already
  // peak ~39% above the reference on footprint, which is why plan() treats
  // the budget as a constraint on the SEARCH rather than as a veto after the
  // fact -- past it the next-best width still runs, and past every width the
  // single-op GEMM (slower, allocation-free) is the trade taken.
  // `x_off` / `y_off` are ELEMENT offsets into x and y, for a caller that
  // packs several matrices into one buffer (the video DiT scatters its
  // modalities into one sequence). The LM callers pass none.
  bool encode(metal_compute::MetalCompute* mc,
              metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& x,
              const metal_compute::SharedBuffer& wdeq,
              const metal_compute::SharedBuffer& y,
              int K, int N, int M,
              std::size_t x_off = 0, std::size_t y_off = 0)
  {
    const Plan p = plan(K, N, M);
    if (p.fn == nullptr) { return false; }
    const int block = std::min(M, rows_per_block(p.splits, N));
    if (block <= 0) { return false; }
    const std::size_t need =
        (std::size_t)block * (std::size_t)N * (std::size_t)p.splits * 4;
    if (planes.empty() || planes.byte_size() < need) {
      planes = mc->make_shared_buffer(need);
      if (planes.empty()) { return false; }   // fall back, do not fault
    }
    // One (split, fold) pair per row block. The blocks share the plane
    // scratch, and Metal's hazard tracking serialises them through it --
    // which is what keeps the scratch bounded no matter how tall M is.
    for (int r0 = 0; r0 < M; r0 += block) {
      const int rows = std::min(block, M - r0);
      const std::size_t plane = (std::size_t)rows * (std::size_t)N;
      enc.set_function(*p.fn);
      enc.set_buffer(0, x, (x_off + (std::size_t)r0 * (std::size_t)K) * 2);
      enc.set_buffer(1, wdeq);
      enc.set_buffer(2, planes);
      enc.set_constant(3, K);
      enc.set_constant(4, N);
      enc.set_constant(5, rows);
      enc.dispatch({(unsigned)(((N + 255) / 256) * 256),
                    (unsigned)((rows + 127) / 128), (unsigned)p.splits},
                   {256, 1, 1});
      // Fold: out = (elt)(sum_s planes[s]). Metal's hazard tracking orders
      // this after the split writes (both touch `planes`).
      enc.set_function(fn_fold);
      enc.set_buffer(0, planes);
      enc.set_buffer(1, y, (y_off + (std::size_t)r0 * (std::size_t)N) * 2);
      enc.set_constant(2, (int)plane);
      enc.set_constant(3, p.splits);
      enc.dispatch({(unsigned)plane, 1, 1}, {256, 1, 1});
    }
    return true;
  }
};

}  // namespace vpipe::genai

#endif
