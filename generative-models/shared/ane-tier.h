#ifndef VPIPE_GENERATIVE_MODELS_SHARED_ANE_TIER_H
#define VPIPE_GENERATIVE_MODELS_SHARED_ANE_TIER_H

// THE ANE TIER, AS A PLUGIN SEES IT: work a model splits between the GPU
// and the Apple Neural Engine, DESCRIBED rather than typed.
//
// Everything that varies between kinds of work -- which math, its widths,
// its activation, the tensors a block hands over, the numbers a block
// measured -- travels as FlexData keys and named buffer bindings. So a new
// activation, a new projection shape, or one day an attention block is a
// new KIND or a new KEY here, and a plugin built before it keeps loading:
// what it never asks for it never sees, and what it asks an older host for
// is refused by name, never misread.
//
// WHAT IS FROZEN -- changing any of it is a VPIPE_PLUGIN_ABI_VERSION bump:
//   * ane::Binding, {name, buffer}
//   * the signatures of ane::runtime_bytes and ane::create
//   * the signatures of ane::Tier's methods, and Tier::Plan's values
// WHAT GROWS FREELY -- never a bump:
//   * kinds, spec keys, binding names, params, timing and info() fields.
//     Added, never renamed; an absent key is its documented default.
//
// Tier is OPAQUE: its only member is a pointer and every method is out of
// line, so nothing about its layout is compiled into a caller. The
// vocabulary below is inline and compiled into the caller, which is what
// lets it grow on either side independently -- the accel bag's contract
// (accel-settings.h), for the same reasons.
//
// The in-tree families drive the machinery underneath (AneFeedForward)
// directly; that header is not installed, and is free to change.

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "common/vpipe-api.h"

VPIPE_API_BEGIN

namespace vpipe {
class SessionContextIntf;
namespace metal_compute { class MetalCompute; }
namespace genai {
namespace ane {

// ---- the vocabulary: the SPEC, create() / runtime_bytes() ------------

inline constexpr std::string_view kKind = "kind";
// A feed-forward over the TAIL rows of a [rows][hidden] plane, written to
// the same rows of an output plane of the same width.
inline constexpr std::string_view kKindFfn = "ffn";
// One projection, [rows][in_width] -> [rows][out_width], over the tail rows.
inline constexpr std::string_view kKindMatmul = "matmul";

// ffn: "swiglu" (the default: silu(x@gate) * (x@up), then down) or
// "gelu_tanh" (torch's approximate="tanh" GELU between up and down).
inline constexpr std::string_view kActivation = "activation";
inline constexpr std::string_view kActSwiglu = "swiglu";
inline constexpr std::string_view kActGeluTanh = "gelu_tanh";

inline constexpr std::string_view kHidden = "hidden";      // ffn
inline constexpr std::string_view kInner = "inner";        // ffn
inline constexpr std::string_view kInWidth = "in_width";   // matmul
inline constexpr std::string_view kOutWidth = "out_width"; // matmul
// ffn: every projection carries a bias (bind `<proj>.bias`; an unbound one
// stages as zeros).
inline constexpr std::string_view kBiases = "biases";

// Rows of the first forward, which sets the opening split. Required.
inline constexpr std::string_view kSeq = "seq";
// The ANE's share of the rows, 0..1. 0 (the default) balances from the two
// engines' measured rates, and turns the split off where the GPU is faster.
inline constexpr std::string_view kRows = "rows";
// Rows per ANE predict (default 2048, 256..16384) and the in-graph tile
// (default 1024).
inline constexpr std::string_view kChunk = "chunk";
inline constexpr std::string_view kTile = "tile";
inline constexpr std::string_view kProfile = "profile";    // a line per block
inline constexpr std::string_view kTag = "tag";            // the log prefix

// ---- stage(): PARAMS and BUFFER bindings -----------------------------
//
// The projections a kind stages, in its own names:
//   ffn swiglu   gate, up, down
//   ffn gelu     up, down
//   matmul       w
// Each projection `<p>` binds, as it has them:
//   <p>.w                       dense bf16, [out][in]
//   <p>.codes <p>.scales <p>.qbias   affine 4/8-bit, bf16 scales/biases
//   <p>.bias                    bf16 [out]
//   <p>.lora<i>.a <p>.lora<i>.b  adapter i (0, 1): A [rank][in], B [out][rank]
// and reads these params (defaults in brackets):
//   group                       the quantization group [0]
//   <p>.bits                    4 or 8; 0 means dense [0]
//   <p>.stride <p>.offset       source row r reads row r*stride+offset [1, 0]
//   <p>.lora<i>.rank            required with an adapter
//   <p>.lora<i>.scale           [1.0]
//   <p>.lora<i>.b_stride <p>.lora<i>.b_offset   [1, 0]
// A binding name the kind does not know is REFUSED: a typo would otherwise
// stage zeros and render.
inline constexpr std::string_view kGroup = "group";
inline constexpr std::string_view kProjGate = "gate";
inline constexpr std::string_view kProjUp = "up";
inline constexpr std::string_view kProjDown = "down";
inline constexpr std::string_view kProjW = "w";

// "<p>.<field>", for building binding names and param keys.
inline std::string
key(std::string_view proj, std::string_view field)
{
  std::string k(proj);
  k += '.';
  k += field;
  return k;
}

// ---- begin(): ARGS and IO bindings -----------------------------------
//
//   args  seq   rows in this forward (required)
//   io    in    [seq][in width], bf16
//         out   [seq][out width], bf16; may be `in` itself
inline constexpr std::string_view kIoIn = "in";
inline constexpr std::string_view kIoOut = "out";

// ---- finish() / note_probe(): TIMING ---------------------------------
//
//   gpu_ms    the GPU's part of the measured section
//   drain_ms  the wait before the split, when the caller had one
inline constexpr std::string_view kGpuMs = "gpu_ms";
inline constexpr std::string_view kDrainMs = "drain_ms";

// ---- info() ----------------------------------------------------------
//
//   kind, activation, chunk, rows_per_block, timing (bool), host_bytes,
//   error (the last refusal, empty when none), lost_row0, gpu_only
inline constexpr std::string_view kInfoRowsPerBlock = "rows_per_block";
inline constexpr std::string_view kInfoTiming = "timing";
inline constexpr std::string_view kInfoHostBytes = "host_bytes";
inline constexpr std::string_view kInfoError = "error";
// THE GPU FALLBACK. After a finish() that returned false, the first row
// the ANE did not write: rows [lost_row0, seq) still hold what they held
// before the split -- the input, where `in` was `out` -- and the caller
// computes them on its GPU path. Rows before it were written and are
// exact. -1 when nothing was lost; ABSENT on a host older than the key,
// where the caller has no range to trust and should fail the forward.
inline constexpr std::string_view kInfoLostRow0 = "lost_row0";
// True once a block has lost rows: plan_block() answers kGpu from then on,
// for the life of the tier.
inline constexpr std::string_view kInfoGpuOnly = "gpu_only";

// ---- the frozen surface ----------------------------------------------

// One named buffer. The buffer must outlive the call that takes it -- for
// stage(), until join_stage() returns.
struct Binding {
  std::string_view                   name;
  const metal_compute::SharedBuffer* buf = nullptr;
};

class Tier;

// What a Tier of this spec holds -- weight slots, staging and one chunk of
// host rows -- for a plan-time CoreML claim. 0 for a spec this host cannot
// build, which a caller should read as "claim nothing".
std::size_t runtime_bytes(const FlexData& spec);

// Build the module (compiled once per shape, then cached). Null, with the
// reason in `why`, for an unknown kind or key value, a spec missing what it
// needs, no ANE, or too few rows for one chunk -- and the caller keeps its
// GPU path.
std::unique_ptr<Tier> create(const SessionContextIntf* session,
                             metal_compute::MetalCompute* mc,
                             const FlexData& spec, std::string* why);

// The per-block protocol, the same for every kind:
//
//   plan = plan_block()                     kSplit / kProbe / kGpu
//   if (needs_barrier()) drain queued GPU work first
//   kSplit: stage(layer, params, weights)   runs on the ANE worker
//           ... encode and commit up to the split ...
//           join_stage(layer); wait for the GPU
//           rows = begin(args, io)          the ANE takes the tail rows
//           ... GPU computes rows [0, seq - rows) ...
//           finish(layer, timing)           joins; false = rows lost:
//                                           compute [info()[lost_row0],
//                                           seq) on the GPU
//   kProbe: same split point, GPU over every row, note_probe(timing)
//   kGpu:   nothing
class Tier {
  VPIPE_ABI_OPAQUE;   // host-owned: see vpipe/export.h
 public:
  enum class Plan : int { kGpu = 0, kSplit = 1, kProbe = 2 };

  ~Tier();
  Tier(const Tier&)            = delete;
  Tier& operator=(const Tier&) = delete;

  Plan plan_block();
  bool needs_barrier() const;

  // False when the bindings or params are refused before anything is
  // dispatched; the reason is info()["error"].
  bool stage(int layer, const FlexData& params,
             std::span<const Binding> buffers);
  bool join_stage(int layer);

  // Rows the ANE took (the tail of the sequence), 0 when none.
  int  begin(const FlexData& args, std::span<const Binding> io);
  bool finish(int layer, const FlexData& timing);
  void note_probe(const FlexData& timing);
  void join();

  FlexData info() const;

 private:
  struct Impl;
  explicit Tier(Impl* p) noexcept;
  Impl* _p = nullptr;

  friend std::unique_ptr<Tier> create(const SessionContextIntf* session,
                                      metal_compute::MetalCompute* mc,
                                      const FlexData& spec, std::string* why);
};

}  // namespace ane
}  // namespace genai
}  // namespace vpipe

VPIPE_API_END

#endif
