#ifndef VPIPE_GENERATIVE_MODELS_SHARED_ACCEL_SETTINGS_H
#define VPIPE_GENERATIVE_MODELS_SHARED_ACCEL_SETTINGS_H

// The acceleration settings a generating stage hands EVERY model family,
// as an open bag rather than as struct fields.
//
// ---------------------------------------------------------------------
// WHY A BAG, when typed fields read better
//
// Because typed fields put the host's vocabulary into the plugin ABI.
// `VideoGenRequest` carried `sage::Config sage; bool i8_gemm;`, and
// adding `sol::Config sol` beside them changed the struct's layout --
// which means every out-of-tree family, including ones that implement
// none of these, has to be rebuilt before it can be loaded again. The
// cost is paid by plugins that do not use the feature, for a decision
// they were not part of, and it is paid EVERY time a tier is added.
//
// That is the wrong trade for a channel whose whole purpose is optional
// hints. So the structs carry one pointer that never changes shape, and
// the vocabulary lives HERE, in a header that ships with the host and is
// read by code compiled into the caller.
//
// ---------------------------------------------------------------------
// THE CONTRACT
//
//   * A KEY IS ADDED, NEVER RENAMED AND NEVER REPURPOSED. An old plugin
//     reading `sol_tau` must keep getting what `sol_tau` has always
//     meant. If a setting's meaning changes, it gets a new key and the
//     old one keeps its old meaning for as long as anything might ask.
//   * ABSENT MEANS THE SHIPPED DEFAULT, and a reader supplies it. A
//     family therefore behaves identically against a host that predates
//     a key and one that has it turned off.
//   * A FAMILY IGNORES WHAT IT DOES NOT KNOW. The host never requires
//     any of this to be implemented; every tier here is an optional
//     accelerator, and a family that implements none is correct.
//   * ...BUT IT SHOULD SAY WHICH, because a knob that silently did
//     nothing and one that did something are indistinguishable in a
//     wall clock. `tiers_on()` is here for that log line.
//   * THE VALUES ARE SETTLED, not raw. The stage validates and corrects
//     before filling the bag -- an out-of-range `sol_key_block` is
//     already the value it fell back to -- so a family can use what it
//     reads without re-checking, and two families cannot disagree about
//     what the graph asked for.
//
// ---------------------------------------------------------------------
// HEADER-ONLY ON PURPOSE
//
// Every reader below is `inline` and compiles into the CALLER. A plugin
// built against an older host carries that host's readers and cannot see
// a key it was never told about; one built against a newer host sees the
// new key without the host having changed shape. The only libvpipe
// symbols involved are FlexData's own, which is the type this boundary
// already speaks -- `model_config` is the same idea for settings that
// belong to ONE family, where this is the same idea for settings that
// belong to all of them.
//
// This makes the acceleration VOCABULARY soft. It does not make the
// whole plugin ABI soft: FlexData's own layout is still shared, as are
// the config structs of any tier a family chooses to drive. The
// difference is the rate -- FlexData changes rarely and deliberately,
// where this list grows whenever someone has an idea.

#include "common/flex-data.h"

#include <string>
#include <string_view>
#include <vector>

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

// ---- reading --------------------------------------------------------
//
// A null bag reads as every default, which is what a family gets from a
// host too old to send one -- and is the same thing it gets from a graph
// that asked for nothing.

inline bool
has(const FlexData* bag, std::string_view key)
{
  if (bag == nullptr || !bag->is_object()) { return false; }
  return bag->as_object().contains(key);
}

inline bool
flag(const FlexData* bag, std::string_view key, bool dflt = false)
{
  if (!has(bag, key)) { return dflt; }
  return bag->as_object().at(key).as_bool(dflt);
}

inline double
real(const FlexData* bag, std::string_view key, double dflt)
{
  if (!has(bag, key)) { return dflt; }
  return bag->as_object().at(key).as_real(dflt);
}

inline long long
integer(const FlexData* bag, std::string_view key, long long dflt)
{
  if (!has(bag, key)) { return dflt; }
  return (long long)bag->as_object().at(key).as_int(dflt);
}

inline std::string
text(const FlexData* bag, std::string_view key, const std::string& dflt = {})
{
  if (!has(bag, key)) { return dflt; }
  return std::string(bag->as_object().at(key).as_string(dflt));
}

// ---- writing --------------------------------------------------------
//
// For the STAGE, and for a test that wants to drive a family without
// one. Kept beside the readers so the two spellings of a key cannot
// drift apart -- they are the same constant.

inline void
set_flag(FlexData* bag, std::string_view key, bool v)
{
  if (bag == nullptr || !bag->is_object()) { return; }
  bag->as_object().insert_or_assign(key, FlexData::make_bool(v));
}

inline void
set_real(FlexData* bag, std::string_view key, double v)
{
  if (bag == nullptr || !bag->is_object()) { return; }
  bag->as_object().insert_or_assign(key, FlexData::make_real(v));
}

inline void
set_integer(FlexData* bag, std::string_view key, long long v)
{
  if (bag == nullptr || !bag->is_object()) { return; }
  bag->as_object().insert_or_assign(key, FlexData::make_int((int64_t)v));
}

inline void
set_text(FlexData* bag, std::string_view key, const std::string& v)
{
  if (bag == nullptr || !bag->is_object()) { return; }
  bag->as_object().insert_or_assign(key, FlexData::make_string(v));
}

// ---- what the graph turned on ---------------------------------------
//
// The ENABLE flag of every tier this reader knows about that the graph
// set true. A family logs the ones it does not implement:
//
//   for (auto t : accel::tiers_on(req.accel)) {
//     if (t != accel::kSolAttn) { warn(... " is not implemented here"); }
//   }
//
// It is deliberately not a list of every key present. What a user needs
// told is that a TIER they asked for is not going to happen; that
// `sol_tau` went unread by a family with no Sol is not news.
//
// A plugin built against an older host lists the tiers that host knew,
// which is exactly right: it cannot implement one it has never heard of,
// and cannot report on one either.
inline std::vector<std::string_view>
tiers_on(const FlexData* bag)
{
  std::vector<std::string_view> out;
  for (std::string_view k :
       {kI8Gemm, kSageAttn, kSolAttn, kAneFfn, kMotionCache}) {
    if (flag(bag, k)) { out.push_back(k); }
  }
  return out;
}

}  // namespace accel
}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_SHARED_ACCEL_SETTINGS_H
