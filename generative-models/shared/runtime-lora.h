#ifndef VPIPE_GENERATIVE_MODELS_SHARED_RUNTIME_LORA_H
#define VPIPE_GENERATIVE_MODELS_SHARED_RUNTIME_LORA_H

#include "apple-silicon/metal-compute/metal-compute.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MetalLlamaWeights;

// Runtime LoRA: the parts of "attach an adapter to a DiT" that are the
// same for every family.
//
// RUNTIME rather than FUSED, and the distinction is not a preference.
// Fusing writes A*B into the base weights, which for a QUANTIZED base
// means rounding the sum back to the base's storage precision -- so for
// a few-step distillation, where the whole point is a small precise
// correction, runtime is the accurate path and fusing is the cheap one.
// It also makes the strength a knob rather than a rebuild.
//
// WHAT IS SHARED AND WHAT IS NOT. Everything about reading the FILE is
// shared: the two publisher spellings, the alpha/rank rescale, the bf16
// conversion, the shape check, and the counting that tells "this adapter
// is for another model" from "this module simply is not adapted". What
// is NOT shared is the module NAME LIST and the forward -- those are the
// model, and a family that pretended otherwise would bind an adapter to
// the wrong projections and produce plausible wrong pictures.
namespace lora {

// One projection's factors: A [rank, K] and B [N, rank], both bf16 and
// both in the checkpoint's own orientation, so one transposed-W GEMM
// serves them -- t = x A^T, then y += t B^T.
struct Factors {
  metal_compute::SharedBuffer a, b;
  int rank = 0;
  bool empty() const { return rank <= 0 || a.empty() || b.empty(); }
};

// An opened adapter file, queried per module.
//
// The STRENGTH is deliberately not here. A per-module `alpha` is a
// property of the FILE and folds into A once, at bind; the caller's
// strength is a property of the REQUEST and rides the accumulating GEMM
// as a constant, so it can be turned without rebuilding any weights.
// Folding the caller's strength in here would take that away.
class Adapter {
public:
  // Rewrites ONE of the FILE's module names into the MODEL's spelling,
  // or "" when that name is not in the convention it handles. See
  // shared/lora-names.h.
  //
  // A plain function pointer rather than a std::function because there
  // is exactly one of these per family, it is a pure map, and it must
  // stay cheap enough to run over every tensor name in the file.
  using Rename = std::string (*)(std::string);

  // Opens `path`. Returns null and sets `err` when it cannot be read.
  //
  // `rename` is the family's file-name convention map, or null. With
  // one installed the adapter indexes the file BOTH ways, so a module
  // is found under its own name or under the other publisher's -- which
  // is the difference between binding an ai-toolkit / ComfyUI adapter
  // and binding nothing at all while reporting success.
  static std::unique_ptr<Adapter> open(const std::string& path,
                                       metal_compute::MetalCompute* mc,
                                       std::string* err,
                                       Rename rename = nullptr);
  ~Adapter();

  // Bind one module. `n`/`k` are the BASE projection's dimensions, and a
  // file whose factors do not fit them is counted as skipped rather than
  // applied in part: a shape that does not fit means the adapter was
  // trained against a different model, and applying whichever modules
  // happen to match would be worse than applying none.
  //
  // A module the file simply does not carry is NOT a skip -- an adapter
  // is free to touch some projections and not others, and that is the
  // ordinary case rather than an error.
  bool bind(const std::string& module, int n, int k, Factors* out);

  // Does the file carry `module` at all, under any spelling this
  // adapter knows? For choosing BETWEEN layouts before binding either:
  // a publisher may ship one adapter decomposed two ways, and which one
  // is in hand is a question about names, not about shapes.
  bool has(const std::string& module) const;

  // Where a source row of B lands in the destination.
  //
  // B is [n, rank] and its rows are the projection's OUTPUT CHANNELS,
  // so an adapter trained against a model that ORDERS those channels
  // differently has right factors and a wrong row order. Two cases in
  // this tree, both real: diffusers' SwiGLU is [value; gate] where the
  // kernels here read [gate; value], and diffusers keeps to_q/to_k/to_v
  // separate where this model fuses them into one projection whose
  // columns a publisher may group flat or per head.
  //
  // `part` indexes the source module (0 for a single one).
  using RowMap = std::function<int(int part, int row)>;

  // Bind SEVERAL source modules into ONE fused projection, or one
  // module whose output channels are reordered.
  //
  // A = the parts' A stacked on the RANK axis; B = each part's rows
  // scattered to `rows(part, row)` in a zero-filled [n, sum(rank)].
  // Which is a block-diagonal B, and is exactly what the upstream
  // ComfyUI conversion of these adapters does -- VERIFIED against it
  // tensor-for-tensor, not inferred from its description.
  //
  // The zeros are the price and they are not small: a fused B is
  // `parts` times the size of the separate ones and two thirds of it is
  // zero. It buys a destination the accumulating GEMM can write
  // CONTIGUOUSLY, where three separate applications would each need a
  // column band of one wider matrix, which is a strided store no kernel
  // here has. The published fused adapters pay the same price on disk.
  //
  // Each part's own alpha/rank folds into ITS OWN slice of A before the
  // stack, so parts of different rank compose correctly and no global
  // rescale is needed. (Upstream instead multiplies the fused alpha by
  // the part count, which is the same arithmetic done once the ranks
  // are known to be equal.)
  //
  // Every part must be present and agree on `k`; a missing one makes
  // the whole projection unbound rather than partly bound, because half
  // a fused adapter is not a weaker adapter but a wrong one.
  bool bind_fused(const std::vector<std::string>& modules, int n, int k,
                  Factors* out, const RowMap& rows);

  // How many of the file's modules were reached through the rename
  // rather than by their own name. Reported so a log says WHICH
  // convention the adapter turned out to be in -- the two are otherwise
  // indistinguishable once binding has succeeded, and the answer is the
  // first thing worth knowing when one of them stops working.
  int renamed() const { return _renamed; }

  // The file-level `__metadata__` alpha, or 0 when it states none. A
  // peft/diffusers export puts its alpha here ONCE instead of writing a
  // per-module `.alpha` tensor, and it is the whole of the adapter's
  // strength -- read wrong, the factors are right and the result is
  // scaled by rank/alpha.
  float metadata_alpha() const { return _meta_alpha; }

  int  modules() const { return _modules; }   // bound successfully
  int  skipped() const { return _skipped; }   // present but wrong shape
  int  max_rank() const { return _max_rank; }

  // One line naming what was bound, for the model to log after it has
  // bound everything it means to.
  std::string summary(const std::string& path, float scale) const;

  // Does any tensor name contain `needle`? Header only -- no tensor is
  // read -- because the answer can be needed BEFORE there is a model to
  // bind to: an adapted pre-activation projection forbids a fused
  // activation kernel, and that decides how the blocks are BUILT.
  //
  // A file that cannot be opened answers false. The real error belongs
  // to the bind that follows; refusing a fusion over a file that turns
  // out to be unreadable would be a slowdown chosen for a reason that
  // never materialized.
  //
  // `rename` MUST be the same one the bind will use. The needle is a
  // name in the MODEL's spelling, and an ai-toolkit file spells it
  // differently -- so without the map here an adapter would bind its
  // pre-activation projections through the rename and then find the
  // fused kernel still in place, with the delta silently dropped on the
  // two biggest projections in the block. That is the exact failure
  // this predicate exists to prevent, arrived at by a different route.
  static bool file_touches(const std::string& path, const std::string& needle,
                           Rename rename = nullptr);

private:
  Adapter() = default;
  // Build `_by_renamed` from the file's tensor names. Called once at
  // open() when a rename is installed.
  void index_renames_();
  // Discover this file's factor-tensor spelling and its file-level
  // alpha. Called once at open().
  void index_suffix_();
  // The file's spelling of `<module>`'s A or B tensor.
  std::string factor_name_(const std::string& key, char ab) const;
  // Resolve `module` to the file's own key, or "" when absent.
  std::string resolve_(const std::string& module, bool* via_rename) const;
  // `key`'s rank and the alpha/rank multiplier that folds into A.
  bool factor_meta_(const std::string& key, int* rank, float* mul);
  // One factor as bf16, scaled by `m`.
  metal_compute::SharedBuffer take_(const std::string& name, float m);

  metal_compute::MetalCompute* _mc = nullptr;
  std::unique_ptr<MetalLlamaWeights> _w;
  Rename _rename = nullptr;
  // model spelling -> the file's own spelling, for the modules whose
  // names the rename actually changed. Empty when no rename is
  // installed, and when the file is already in the model's convention.
  std::map<std::string, std::string> _by_renamed;
  // What follows a module name in this file: ".lora_A.weight", or
  // peft's ".lora_A.<adapter>.weight". Discovered once rather than
  // guessed per lookup, so any adapter name works and not just
  // "default".
  std::string _suf_a = ".lora_A.weight";
  std::string _suf_b = ".lora_B.weight";
  float _meta_alpha = 0.0f;
  int _modules = 0, _skipped = 0, _max_rank = 0, _renamed = 0;
};

// Applies bound factors to a projection's output: t = x A^T, then
// y += scale * t B^T.
//
// SHARED for the same reason the reader is: this is arithmetic about a
// LoRA, not about a model. What differs per family is which projections
// carry factors, and that is the caller's list.
//
// THE STRENGTH RIDES ON EXACTLY ONE OF THE TWO GEMMS, and which one
// depends on the route. Steel's epilogue is a scalar expression
// (acc * s + y) so it can take it; matmul2d's accumulate is a mode with
// nowhere to put a coefficient, so the matrix-core pair scales its
// smaller intermediate instead. Same product either way -- what must
// never happen is BOTH, which is why the first call reports back whether
// it took the scale.
class Applier {
public:
  // Loads the tiles once. Safe to call on a box without matrix cores:
  // the steel pair is the fallback and is always used when a tile is
  // missing or `allow_mma` is false.
  bool init(metal_compute::MetalCompute* mc, bool allow_mma);

  // Ready to apply anything at all.
  bool valid() const { return _steel.valid() && _steel_acc.valid(); }

  // Grow the [M, rank] intermediate. Called once per forward with the
  // widest shape the pass will use, so `apply` never allocates.
  bool ensure_scratch(std::size_t elems);

  // One projection. A no-op at scale 0 -- which is a legitimate request
  // (an A/B against the un-adapted model) and the cheapest way to serve
  // it is not to encode the GEMMs at all, so "off" is exactly off rather
  // than a pair of roundings that happen to cancel.
  void apply(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t x_off,
             const Factors& f,
             const metal_compute::SharedBuffer& y, std::size_t y_off,
             int m, int n, int k, float scale, int mma_min_m);

private:
  const metal_compute::ComputeFunction* route_a_(int rank) const;
  const metal_compute::ComputeFunction* route_b_(int rank) const;

  metal_compute::MetalCompute* _mc = nullptr;
  metal_compute::ComputeLibrary _lib_gemm, _lib_mma;
  metal_compute::ComputeFunction _steel, _steel_acc;
  metal_compute::ComputeFunction _a64, _a128, _b128, _b256;
  metal_compute::SharedBuffer _scratch;
  bool _mma = false;
};

}  // namespace lora
}  // namespace genai
}  // namespace vpipe

#endif
