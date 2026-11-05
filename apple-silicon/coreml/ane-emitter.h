#ifndef VPIPE_APPLE_SILICON_COREML_ANE_EMITTER_H
#define VPIPE_APPLE_SILICON_COREML_ANE_EMITTER_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vpipe {

class CoreMLModelManager;

// Writes a compiled CoreML model (.mlmodelc) directly, with no
// coremltools and no Python.
//
// WHY. A compiled module's SHAPE is baked into its graph, so anything
// that ships prebuilt modules is keyed on the shapes it was built for
// -- a new resolution, a new width, a new model means a new artefact.
// Emitting the graph here removes that: any shape, at load, from the
// caller's own weights.
//
// THIS IS FEASIBLE ONLY BECAUSE THE THREE FILES ARE SIMPLE, and each
// was established by inspecting what coremltools produces:
//
//   model.mil           ASCII TEXT. Not a protobuf -- the graph is
//                       emitted as source, which is why this class is
//                       a few hundred lines rather than a schema.
//   coremldata.bin      a fixed 83-byte header, a small protobuf
//                       describing one input and one output, and a
//                       16-byte footer.
//   weights/weight.bin  a 64-byte file header, then per weight a
//                       64-byte entry header (magic 0xdeadbeef, byte
//                       size at +8, data offset at +16) and the data
//                       verbatim, row-major, no permutation. EVERY
//                       ENTRY HEADER STARTS ON A 64-BYTE BOUNDARY: after
//                       a slab whose size is not a multiple of 64 the
//                       next header is padded forward (coremltools
//                       writes 2146 bytes of data, then 30 of padding),
//                       so data offsets are always 64-aligned. The file
//                       is not padded after its last slab.
//
// `analytics/` is NOT required -- a module without it loads and
// predicts.
//
// A `BLOBFILE` offset in the .mil names the weight's ENTRY HEADER, not
// its data: entry 0 is referenced at 64 and its bytes start at 128.
// Getting that wrong reads the header as weights.
//
// NONE OF THIS IS DOCUMENTED BY APPLE. It is inferred from emitted
// artefacts and could change with any macOS release, which is why
// self_test() exists and why every failure path here must leave the
// caller free to use its GPU implementation instead. Emitting a model
// that does not load is recoverable; emitting one that loads and
// computes the wrong thing is not, so the self-test checks NUMBERS,
// not just that a load succeeded.
struct AneBlobSlab {
  const void* data  = nullptr;   // row-major, element order as-is
  std::size_t bytes = 0;
};

// The graphs this emits. Deliberately a short enumeration rather than a
// general IR: a general one would be a compiler, and two shapes cover
// what a transformer block's feed-forward needs.
struct AneGraphSpec {
  enum class Kind {
    Matmul,      // y[M,N] = x[M,K] @ W[K,N]
    SwiGluFfn,   // y[M,K] = (silu(x@Wg) * (x@Wu)) @ Wd, N == ffn inner
    GeluFfn,     // y[M,K] = gelu_tanh(x@Wu) @ Wd, N == ffn inner
    // MOSS-Image's: y = (4 tanh(g/4) sigmoid(g) * 25 tanh(u/25)) @ Wd,
    // g = x@Wg, u = x@Wu. The SwiGLU's three inputs (and biases); only the
    // activation differs.
    SituFfn,
  };

  Kind kind = Kind::Matmul;
  int  M = 0;              // rows
  int  K = 0;              // Matmul: reduction; the FFNs: hidden
  int  N = 0;              // Matmul: output; the FFNs: ffn inner
  // Tiling INSIDE the graph. The ANE's weight buffer is ~8-10 MB and
  // its rate falls off a cliff above that, so a big matmul is emitted
  // as tiles -- but in ONE graph, because per-call tiling from the host
  // measured 4.7-9x worse (small ops, and a predict floor per call).
  // 0 means "do not tile that axis".
  int  bk = 0;
  int  bn = 0;
  // RUNTIME weights: every matrix is a model INPUT (input_features())
  // rather than a constant, one whole [in,out] array per projection,
  // tiled by slicing INSIDE the graph at the same bk/bn. The graph then
  // holds no weights at all, so ONE compiled module serves any weights of
  // this shape -- and, having no weight file, it is cached on content, so
  // it compiles once per shape rather than once per block. MEASURED on a
  // Krea-2 feed-forward with IOSurface-bound inputs: 11.2 TOPS at 1024
  // tiles against 14.0 for the best constant form.
  bool runtime_weights = false;
  // With runtime_weights only: declare each matrix [out,in] -- the layout a
  // checkpoint stores -- and read it with transpose_y. The ANE computes it
  // ~9% slower (a Krea-2 feed-forward at 1024 tiles: 119.2 ms against
  // 109.2 for [in,out]), but the caller fills its inputs row for row:
  // MEASURED 6.7 ms against 81-116 ms to transpose one 193 MB matrix,
  // which is the difference between staging a block's weights under its
  // attention and waiting on them.
  bool weights_out_in = false;
  // With runtime_weights and SwiGluFfn only: three more inputs, bg [1,N],
  // bu [1,N] and bd [1,K], each added to its projection's output -- for a
  // feed-forward whose Linears carry biases.
  bool biases = false;
  // ROTATED INT8 (W8A8), with runtime_weights and weights_out_in, for
  // Matmul and SwiGluFfn without biases, every reduction width a multiple
  // of 256. Each weight is an int8 INPUT [out, in] whose rows were
  // Walsh-Hadamard rotated per 256-block of their K and quantized each to
  // its own step, beside an fp16 input [1, out] of 128 x those steps
  // (input_features names them); each projection's input is rotated by
  // the same transform INSIDE the graph and quantized per token there --
  // the GPU's `token` tier's arithmetic. x and the output stay fp16. The
  // graph uses the iOS 26 generation's opset (`ios19`), so it loads only
  // where CoreML takes int8 arrays (macOS 26 and later). MEASURED on
  // MiniMax-H3's feed-forward (5376 -> 14336, 2048 rows, 1024 tiles), M4
  // Pro: 14.3 TOP/s against the fp16 form's 9.6, and on real activations
  // the GPU token tier's error to two digits.
  bool int8 = false;
  // With int8: no rotation -- the weights quantized per row as stored and
  // each input per token as it is, the GPU's `token_plain` arithmetic.
  // The widths need not be multiples of 256 then.
  bool int8_plain = false;
  // With int8: GROUPED along K -- the GPU's `fast` tier's arithmetic --
  // at this target group (256, 512 or 1024; 0 = per token). Each
  // projection takes group_for(int8_group, its K): the largest of 1024 /
  // 512 / 256 no larger than the target that divides K. Each group of
  // an input is rotated whole and quantized at its own step, the
  // weights the same per (row, group) -- their steps the [K/g, out]
  // inputs -- and a group's product is scaled by both steps before the
  // groups are summed: one matmul a group. Rotated (not with
  // int8_plain).
  int  int8_group = 0;
  static int group_for(int target, int K) noexcept;

  bool valid() const;
};

class AneEmitter {
 public:
  // Byte size of each weight slab the graph expects, IN GRAPH ORDER --
  // which is the order emit() consumes them and the order they land in
  // the blob. For a tiled graph this is one entry per tile, n-tile
  // major and k-tile minor.
  static std::vector<std::size_t> weight_slabs(const AneGraphSpec& s);

  static std::size_t input_elems(const AneGraphSpec& s);
  static std::size_t output_elems(const AneGraphSpec& s);

  // Names the emitted model declares, so a caller can bind without
  // introspecting.
  static const char* input_name() noexcept { return "x"; }
  static std::string output_name(const AneGraphSpec& s);

  // One array in a model's interface: fp16, or int8 (CoreML's
  // MLMultiArrayDataTypeInt8, macOS 26 and later).
  struct Feature {
    std::string               name;
    std::vector<std::int64_t> shape;
    bool                      int8 = false;
  };

  // Where write_bundle() puts each slab's ENTRY HEADER, for slabs of
  // these byte sizes -- which is what a BLOBFILE offset must name. One
  // implementation of the 64-byte alignment rule, so a hand-written
  // graph and the blob can never disagree about it.
  static std::vector<std::size_t>
  blob_offsets(std::span<const std::size_t> bytes);

  // The `program(1.3)` preamble every emitted model.mil opens with.
  static const char* mil_preamble() noexcept;

  // The general form emit() is built on: a .mlmodelc from MIL source,
  // its interface, and the weight slabs its BLOBFILE constants name (in
  // blob order; empty for a graph with no constants, which then gets no
  // weights/ directory, as coremltools writes it). This is how a graph
  // with RUNTIME weights -- declared as extra function inputs rather
  // than constants -- is emitted. `dir` is replaced.
  static bool write_bundle(const std::string& dir, const std::string& mil,
                           std::span<const Feature> inputs,
                           std::span<const Feature> outputs,
                           std::span<const AneBlobSlab> weights,
                           std::string* err);

  // The model's inputs in declaration order: `x` [M,K], then for a
  // runtime-weight graph one matrix per projection, [in,out] -- Matmul
  // `w` [K,N]; SwiGluFfn `wg` [K,N], `wu` [K,N], `wd` [N,K] -- or each of
  // those transposed when weights_out_in. An int8 graph's matrices are
  // int8 and followed, in the same order, by one fp16 [1, out] row of
  // steps a projection: `w_s`, or `wg_s`, `wu_s`, `wd_s`.
  static std::vector<Feature> input_features(const AneGraphSpec& s);

  // Write a complete .mlmodelc into `dir` (created, replacing any
  // existing one). `weights` must match weight_slabs() exactly in count
  // and in every size -- a mismatch is refused rather than truncated,
  // because a short slab would be read as whatever followed it.
  static bool emit(const AneGraphSpec& s,
                   std::span<const AneBlobSlab> weights,
                   const std::string& dir, std::string* err);

  // Emit a small model, load it, run it, and check the RESULT against a
  // reference computed here. Cheap enough (64x64x64) to run at startup,
  // and the only thing that can tell a format that still loads from one
  // that still works.
  //
  // A consumer should call this once and disable the ANE path on false
  // rather than trusting emit() blind.
  static bool self_test(CoreMLModelManager& mgr, const std::string& scratch,
                        std::string* err);

 private:
  static std::string emit_mil_(const AneGraphSpec& s);
  static std::vector<unsigned char> emit_description_(
      std::span<const Feature> inputs, std::span<const Feature> outputs);
  static std::vector<unsigned char> emit_blob_(
      std::span<const AneBlobSlab> weights);
};

}  // namespace vpipe

#endif
