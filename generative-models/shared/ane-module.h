#ifndef VPIPE_GENERATIVE_MODELS_SHARED_ANE_MODULE_H
#define VPIPE_GENERATIVE_MODELS_SHARED_ANE_MODULE_H

#include "apple-silicon/coreml/ane-emitter.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstddef>
#include <functional>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vpipe {

class CoreMLLoadedModel;
class SessionContextIntf;

namespace genai {

// An ANE-resident compute module: a CoreML graph whose SHAPE is fixed
// at build time and whose WEIGHTS are stamped in from the caller's own
// buffers at model load.
//
// WHY THIS EXISTS. On an M4 the GPU has no int8 path at all (integer
// MAD runs 2.0 TOP/s against 7.94 TFLOP/s f16 -- a quarter rate), and
// the f16 GEMM path is already at 84-92% of its roofline, so there is
// no lever left on the GPU. The ANE is the only unused compute on the
// die, and on DiT-shaped work it delivers ~14 TOPS against the GPU's
// achieved 6.6-7.3 -- about 2x here and closer to 4x on a base M4,
// which carries the SAME 16-core ANE against roughly half the GPU.
//
// WHAT THE ANE WANTS (all measured, M4 Pro, fp16):
//
//   * a weight-dedicated on-chip buffer of ~8-10 MB. Per-op weights
//     under it run 16-17 TOPS; over it the rate falls off a cliff
//     (8.4 MB 16.8, 11.8 MB 12.2, 50 MB 4.2).
//   * activation streaming proportional to M*(K+N), so at a fixed K*N
//     the rate is SYMMETRIC in K and N and peaks at K == N.
//   * M >= ~2048, because the weight load amortises over rows. The
//     dataflow is weight-stationary: the buffer holds the WEIGHT, and
//     the cliff does not move when M changes.
//
// So a big matmul is emitted as an explicit (Bm,Bk,Bn) tiling inside
// ONE graph. Tiles over M and N are independent; the K loop
// accumulates, and accumulation is CHEAP -- squareness matters far
// more. A Krea-2 FFN tiled (4096,2048,2048) measures 14.2 TOPS against
// 2.7 monolithic, while a tiling contorted to avoid K-splits entirely
// (Bk=6144, Bn=512) manages only 4.3.
//
// KEEP A WHOLE BLOCK IN ONE MODULE. predict() carries ~1.1-2.3 ms of
// fixed cost, so granularity is set by amortising THAT: a whole FFN
// (825 GFLOP, ~50 ms) sits at 2-4% overhead and a whole block (1649
// GFLOP) at 1-2%. Per-tile dispatch is hopeless. The GPU<->ANE
// crossing itself is FREE -- measured at -0.3 to -1.0 ms against a
// GPU-only control with the same commit count, i.e. the ANE phase
// overlaps the GPU's commit/wait rather than serialising behind it --
// so a design does NOT have to avoid crossings.
//
// WHY WEIGHTS ARE STAMPED RATHER THAN PASSED. CoreML will accept the
// weight as a runtime INPUT and the op still lands on the ANE, but it
// then re-stages the weight per internal M-tile and costs ~37% (9-10
// TOPS against 14). A baked weight keeps the full rate. The price is
// an ANE compile at load, ~34 ms per MB -- which is recoverable: the
// compile runs in ANECompilerService.xpc at about one core, and
// parallelises ~2.3x across processes. Run it while the ANE is IDLE
// (GPU-only warmup steps) and it is free; running it UNDER ANE
// inference costs the inference 1.34x, so do not interleave.
//
// THE TEMPLATE. Building a CoreML graph needs coremltools, which is
// not a runtime dependency here -- so the graph is produced OFFLINE by
// tools/make_ane_module.py as a compiled .mlmodelc with dummy weights,
// and this class only stamps real bytes into it. That works because
// the weight blob is a plain index of raw row-major arrays: a 64-byte
// file header, then per weight a 64-byte entry header (magic
// 0xdeadbeef, byte size at +8, data offset at +16) followed by the
// data verbatim -- no permutation, no compression, no padding. The
// hardware's internal tiling is applied by the driver at load, which
// is also why the declared layout (NN vs NT) measures identically.
//
// That layout is UNDOCUMENTED. parse_weight_blob() therefore validates
// the magic, the entry count and the size arithmetic on every load and
// fails loudly rather than stamping bytes into a format that moved
// under a macOS update. A caller that gets `false` should fall back to
// its GPU path, not proceed.
struct AneWeight {
  const void* data  = nullptr;   // row-major, element order as-is
  std::size_t bytes = 0;
};

// AneWorker -- the threads an ANE dispatch runs on -- lives with the
// CoreML wrapper (apple-silicon/coreml/ane-worker.h) and is owned by
// the session's CoreMLModelManager, because it is a property of the
// DEVICE rather than of any model family: the vision towers, the
// inference stage and a generative family all dispatch to the same
// ANE and should share one thread. Reach it through
// `session->services()->coreml_model_manager()->ane_worker()`.

// One entry discovered in a compiled module's weight blob, in graph
// order. `offset` is absolute within the file.
struct AneWeightSlot {
  std::size_t offset = 0;
  std::size_t bytes  = 0;
};

// A logical [K,N] row-major fp16 weight that the TEMPLATE holds as
// tiles. A tiled graph carries one blob entry per emitted tile, so a
// caller holding one contiguous matrix has to hand over slabs -- and
// every ordering produces correctly SHAPED slabs, which is what makes
// getting it wrong silent. build_tiled() does the gather so no caller
// re-derives it.
//
// The order is the one tools/make_ane_module.py emits: n-tile major,
// k-tile minor (see tile_order() there, which prints the count). Bk /
// Bn must be the tile the template was generated with.
struct AneTiledWeight {
  const void* data = nullptr;   // [K,N] row-major fp16
  std::size_t K = 0, N = 0;     // logical shape
  std::size_t Bk = 0, Bn = 0;   // tile, as the template was built
};

// Read the weight index out of a compiled module's
// `weights/weight.bin`. Returns false (with `err` set) if the file is
// missing, the magic does not match, or the entries do not tile the
// file exactly -- i.e. whenever the assumed format no longer holds.
bool
parse_weight_blob(const std::string&          weight_bin_path,
                  std::vector<AneWeightSlot>* out,
                  std::string*                err);

// The IOSurface-backed fp16 buffers a RUNTIME-weight module reads its
// matrices from, one per declared weight input, shaped to match -- int8
// (8-bit pixel buffers) for a rotated int8 graph's matrices.
//
// Why pixel buffers rather than host memory: the module stages its
// weight inputs into wired memory either way (a fixed cost per MODEL, not
// per buffer or per weights), but an IOSurface-backed array is memory the
// Neural Engine can take as it is. MEASURED on a Krea-2 feed-forward:
// 117.5 ms against 129.3 for host buffers, the same wired total, and
// identical outputs.
//
// Refill in place per use -- the module re-reads the contents on every
// call, so the next block's weights in the same slots are what it
// computes with (verified bit-exact against a fresh buffer set).
class AneWeightSlots {
 public:
  static std::unique_ptr<AneWeightSlots> create(const AneGraphSpec& spec);
  ~AneWeightSlots();

  AneWeightSlots(const AneWeightSlots&)            = delete;
  AneWeightSlots& operator=(const AneWeightSlots&) = delete;

  std::size_t count() const noexcept { return _pb.size(); }
  int rows(std::size_t i) const noexcept { return _rows[i]; }
  int cols(std::size_t i) const noexcept { return _cols[i]; }
  // Whether slot i holds int8 elements (else fp16).
  bool int8(std::size_t i) const noexcept { return _i8[i]; }
  // Opaque CVPixelBufferRef for slot i.
  void* pixel_buffer(std::size_t i) const noexcept { return _pb[i]; }
  // Element bytes across every slot.
  std::size_t bytes() const noexcept;

  // Lock slot i and hand `fill` its base address and BYTES PER ROW,
  // which a pixel buffer may pad past cols*2, then unlock. False if the
  // lock failed.
  bool fill(std::size_t i,
            const std::function<void(void* base, std::size_t bytes_per_row)>&
                fill);

 private:
  AneWeightSlots() = default;
  std::vector<void*> _pb;
  std::vector<int>   _rows, _cols;
  std::vector<bool>  _i8;
};

class AneModule {
 public:
  // Copy `template_dir` (a compiled .mlmodelc) into `work_dir`, stamp
  // `weights` into its blob in graph order, and load it through the
  // session's CoreML model manager.
  //
  // `weights` must match the template's entry count and each entry's
  // byte size exactly; a mismatch is a build failure, never a silent
  // truncation. Returns nullptr on any failure, having logged why.
  //
  // The load performs the ANE compile, so this is SLOW (~34 ms per MB
  // of weight) and belongs on a load path, not in a forward pass.
  static std::unique_ptr<AneModule>
  build(const SessionContextIntf* session, const std::string& template_dir,
        const std::string& work_dir, std::span<const AneWeight> weights);

  // Build WITHOUT a template, by emitting the graph (AneEmitter).
  //
  // This is the form that does not need anything shipped or generated
  // ahead of time: the shape comes from `spec`, so a new resolution, a
  // new width or a new model needs no artefact and no Python. The
  // weights are the same LOGICAL matrices build_tiled() takes, gathered
  // into the tiles the emitted graph expects.
  //
  // A caller should ask emitter_usable() once first. Emission depends
  // on undocumented formats, and the honest failure mode is not a bad
  // load but a model that runs and is wrong.
  static std::unique_ptr<AneModule>
  build_emitted(const SessionContextIntf* session, const AneGraphSpec& spec,
                std::span<const AneTiledWeight> weights,
                const std::string& work_dir);

  // Whether emission produces a model that computes the right answer on
  // THIS machine and THIS macOS. Runs AneEmitter's self-test once per
  // process, caches the verdict, and logs the reason on failure.
  //
  // Cheap (a 64-cube matmul) and the whole point: a format that has
  // drifted still parses and still runs.
  static bool emitter_usable(const SessionContextIntf* session);

  // A RUNTIME-weight module (spec.runtime_weights): the graph holds no
  // weights, so it is built once per shape and fed any weights of that
  // shape through run(..., AneWeightSlots). Having no weight file, the
  // compiled program is cached on content -- the first build of a shape
  // pays the compile (~11 s for a Krea-2 feed-forward at 1024 tiles),
  // later builds load in tens of milliseconds.
  static std::unique_ptr<AneModule>
  build_runtime(const SessionContextIntf* session, const AneGraphSpec& spec,
                const std::string& work_dir);

  // As build(), but each entry is one LOGICAL matrix that the template
  // holds split into tiles; the gather into per-tile slabs happens
  // here, in the template's emission order. Entries are expanded in
  // the order given, so for an FFN pass W1 then W2.
  static std::unique_ptr<AneModule>
  build_tiled(const SessionContextIntf*        session,
              const std::string&               template_dir,
              const std::string&               work_dir,
              std::span<const AneTiledWeight>  weights);

  ~AneModule();

  AneModule(const AneModule&)            = delete;
  AneModule& operator=(const AneModule&) = delete;

  bool valid() const noexcept;

  // Whether the compiled bundle actually contains an ANE program. The
  // E5 bundle names the winning engine directly -- `main/main_ane/`
  // for the ANE against `main/main_bnns/` for the CPU path -- so a
  // module that silently fell back is detectable without running it.
  // False means the module WILL run, just not where it was meant to.
  bool ane_resident() const noexcept { return _ane_resident; }

  // Element counts the template declares, for the caller to size its
  // buffers against.
  std::size_t input_elems() const noexcept { return _in_elems; }
  std::size_t output_elems() const noexcept { return _out_elems; }

  // Run the module: `x` is read and `y` written IN PLACE, zero-copy,
  // straight out of Metal-shared memory. Both must be fp16 and at
  // least input_elems() / output_elems() elements.
  //
  // The caller is responsible for ordering: the GPU work producing `x`
  // must have completed (commit().wait()) before this is called, and
  // the GPU work consuming `y` enqueued after it returns. CoreML
  // offers no Metal-shared-event interop, so there is no way to
  // express the dependency to the driver instead.
  bool run(const metal_compute::SharedBuffer& x,
           const metal_compute::SharedBuffer& y)
  {
    return run(x, 0, y, 0);
  }

  // ...and the twin that takes ELEMENT offsets, which is what a row
  // split needs. Rows are independent in a feed-forward, so giving the
  // ANE a contiguous band of them is exact -- no partial sums, no seam.
  //
  // The GPU should take the HEAD and the ANE the TAIL: a caller's GEMM
  // helper typically writes at an output offset but reads its input
  // from zero, so head-to-GPU needs no change on that side, while this
  // path can offset both ends freely.
  bool run(const metal_compute::SharedBuffer& x, std::size_t xe,
           const metal_compute::SharedBuffer& y, std::size_t ye);

  // ...and for a runtime-weight module, with its matrices in `weights`.
  // The slots must match the module's weight inputs in count and shape.
  bool run(const metal_compute::SharedBuffer& x, std::size_t xe,
           const metal_compute::SharedBuffer& y, std::size_t ye,
           const AneWeightSlots& weights);

  // How many weight inputs a runtime-weight module takes; 0 otherwise.
  std::size_t weight_inputs() const noexcept { return _w_names.size(); }

  const std::string& path() const noexcept { return _path; }

 private:
  AneModule() = default;

  std::shared_ptr<CoreMLLoadedModel> _model;
  std::string                        _path;
  std::string                        _in_name;
  std::string                        _out_name;
  std::vector<std::int64_t>          _in_shape;
  std::vector<std::int64_t>          _out_shape;
  std::size_t                        _in_elems     = 0;
  std::size_t                        _out_elems    = 0;
  bool                               _ane_resident = false;
  // Runtime-weight modules only: the weight inputs, in order.
  std::vector<std::string>                _w_names;
  std::vector<std::vector<std::int64_t>>  _w_shapes;
};

}  // namespace genai
}  // namespace vpipe

#endif
