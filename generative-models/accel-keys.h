#ifndef VPIPE_GENERATIVE_MODELS_ACCEL_KEYS_H
#define VPIPE_GENERATIVE_MODELS_ACCEL_KEYS_H

// The ACCELERATION BAG's keys: what a generating stage puts in the
// `accel` FlexData it hands every model family. The contract and the
// readers are in generative-models/shared/accel-settings.h (toolkit);
// the NAMES are here, in the stable SDK, because they are what the host
// promises every plugin in its support window -- added, never renamed,
// never repurposed, and absent meaning the shipped default.

#include <string_view>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {
namespace genai {
namespace accel {

// ---- the keys -------------------------------------------------------
//
// Spelled exactly as the graph spells them, because they ARE the graph's
// keys: `generate-video` documents them in its own config schema and
// copies the settled values here. One vocabulary, not two.

// int8 GEMMs in the block. Bool; default false.
inline constexpr std::string_view kI8Gemm = "i8_gemm";

// SageAttention -- the QK^T product in int8. Bool; default false.
inline constexpr std::string_view kSageAttn = "sage_attn";
// Leading blocks left in the tensor dtype. Int; default 0.
inline constexpr std::string_view kSageDenseLayers = "sage_dense_layers";

// Sol-Attn -- which key blocks are attended exactly. Bool; default
// false.
inline constexpr std::string_view kSolAttn = "sol_attn";
// The threshold, in standard deviations of the routing proxy's spread.
// Real; default 1.0.
inline constexpr std::string_view kSolTau = "sol_tau";
// The routing block, in keys. Int; 32 or 64, default 64 -- already
// corrected by the stage, so a value here is one the host accepted.
inline constexpr std::string_view kSolKeyBlock = "sol_key_block";
// Leading blocks left dense. Int; default 1.
inline constexpr std::string_view kSolDenseLayers = "sol_dense_layers";
// Key blocks either side of the query's own, kept exact. Int; default 1.
inline constexpr std::string_view kSolLocalRadius = "sol_local_radius";

// ---- MotionCache ----------------------------------------------------
//
// Step-level caching (shared/motion-cache.h): a denoiser forward whose
// output a cached residual predicts well enough is skipped. Unlike every
// tier above it saves whole FORWARDS rather than making one cheaper, so it
// composes with all of them. Implemented by the family's SAMPLING LOOP,
// not by its transformer.
//
// Bool; default false.
inline constexpr std::string_view kMotionCache = "motion_cache";
// Accumulated predicted relative change under which a step reuses.
// Real; default 0.15. Higher skips more.
inline constexpr std::string_view kMotionCacheThreshold =
    "motion_cache_threshold";
// Weight of frame-to-frame motion in every change measure. Real;
// default 1.0; 0 weighs every position equally.
inline constexpr std::string_view kMotionCacheStrength =
    "motion_cache_strength";
// Leading forwards always computed. Int; default 4.
inline constexpr std::string_view kMotionCacheWarmup = "motion_cache_warmup";
// Most reused forwards in a row. Int; default 2, at least 1.
inline constexpr std::string_view kMotionCacheMaxSkips =
    "motion_cache_max_skips";
// The window it is consulted in, as fractions of the sampling range (0
// noise, 1 clean). Real; defaults 0.15 and 0.95, start < end.
inline constexpr std::string_view kMotionCacheStart = "motion_cache_start";
inline constexpr std::string_view kMotionCacheEnd = "motion_cache_end";
// Latent stride of the change estimator. Int; default 8, at least 1.
inline constexpr std::string_view kMotionCacheSubsample =
    "motion_cache_subsample";

// ---- the ANE tier ---------------------------------------------------
//
// The block's feed-forward on the Apple Neural Engine. This is the one
// tier whose prize is not a better kernel but a SECOND ENGINE: on an M4
// the GPU has no int8 path worth having (integer MAD runs 2.0 TOP/s
// against 7.94 TFLOP/s f16) and its f16 GEMM is already at 84-92% of
// roofline, so the ANE is the only unused compute on the die. Measured
// there: ~14 TOPS on a tiled DiT-shaped FFN against the GPU's achieved
// 6.6-7.3, and the gap is wider on a base M4, which carries the same
// 16-core ANE against roughly half the GPU.
//
// TWO COSTS A FAMILY MUST WEIGH BEFORE SETTING THIS. The ANE runs fp16
// ONLY -- int8 weights buy footprint under its ~8-10 MB weight buffer,
// not arithmetic -- so a QUANTIZED block is dequantized to fp16 for it.
// And the module holds memory no other ledger sees: the families that
// implement this (shared/ane-ffn.h) run ONE runtime-weight module whose
// weights are staged per block, so the cost is one block's fp16 weights
// twice over plus the ANE's rows -- fixed, and claimed from the plan.
//
// Bool; default false.
inline constexpr std::string_view kAneFfn = "ane_ffn";

// The share of an FFN's ROWS given to the ANE, with the GPU taking the
// rest concurrently. Rows are independent in a feed-forward, so the
// split is exact -- no partial sums, no seam. Real in [0,1]; default
// 0.0, meaning "let the family choose", which it should do from the two
// engines' measured rates (~12.6 TOPS ANE against ~7 TFLOP/s GPU puts
// the balance near 0.64).
//
// 1.0 gives the whole FFN to the ANE and leaves the GPU idle for its
// duration, which is the right setting only when the GPU has other work
// to overlap. The crossing itself is free -- measured at -0.3 to -1.0 ms
// against a GPU-only control, i.e. the ANE phase overlaps the GPU's
// commit/wait rather than serialising behind it.
inline constexpr std::string_view kAneRows = "ane_rows";

// Directory of precompiled ANE module templates. String; default empty.
// UNUSED by a family whose feed-forward runs on a runtime-weight module
// (it emits its own graph); kept so the key a graph may carry keeps its
// meaning for any family that still stamps templates.
inline constexpr std::string_view kAneTemplates = "ane_templates";

// A CAP on how many of a model's blocks use the ANE feed-forward.
// Integer; default 0, meaning every eligible block. NOT a memory setting:
// the blocks share one runtime-weight module whose cost is fixed however
// many use it, and whether that module fits at all is the resource plan's
// decision. Set it only to cover fewer blocks on purpose.
inline constexpr std::string_view kAneLayers = "ane_layers";

// The fused q|k|v projection on the ANE too, as a SECOND module beside
// the feed-forward's, split the same way and sharing the one ANE worker.
// Bool; default false.
//
// REQUIRES kAneFfn, and not by accident: the plan books both modules as
// ONE CoreML unit, so they are granted together or not at all. A tier
// that could be granted alone would need a unit of its own, and nothing
// asks for that yet.
//
// It is a SEPARATE cost, which is the reason this is a key rather than
// something kAneFfn implies: the qkv module holds its own weight slots
// and staging, so a family that books only the feed-forward under-reports
// the moment a graph turns this on.
inline constexpr std::string_view kAneQkv = "ane_qkv";

}  // namespace accel
}  // namespace genai
}  // namespace vpipe

VPIPE_API_END

#endif
