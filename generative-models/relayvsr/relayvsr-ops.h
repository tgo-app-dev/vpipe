#ifndef VPIPE_GENERATIVE_MODELS_RELAYVSR_RELAYVSR_OPS_H
#define VPIPE_GENERATIVE_MODELS_RELAYVSR_RELAYVSR_OPS_H

// INTERNAL to generative-models/relayvsr/: what the keyframe DiT and the
// decoder share -- the kernels of one element type, the GEMM route, the
// windowed attention's encoder and the steel cross-attention's. Not
// installed, not exported.

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class I8GemmContext;
class MetalSageAttention;
class WeightSet;

namespace fp8 {
class Expander;
struct Weight;
}  // namespace fp8

namespace relayvsr {

// Host mirror of the kernel's LocalWindowAttnParams
// (gpu-kernels/metal/attention/local_window_attn.metal) -- and of its
// matrix-core twin's LwaNaxParams (local_window_attn_nax.metal), the same
// fields in the same order.
struct LocalWindowAttnParams {
  int H, W;
  int Hq, Hkv;
  int planes;
  int tiles_x;
  int q_tok, q_head;
  int o_tok, o_head;
  int k_head, v_head;
  float scale;
  int k_tok[4];
  int v_tok[4];
  int win_h[4];
  int win_w[4];
  int shifted[4];
};

// One key plane of a windowed attention: its K and V (byte offsets into
// the buffers), token strides, window and rule.
struct WindowPlane {
  const metal_compute::SharedBuffer* k = nullptr;
  std::size_t k_off = 0;
  int k_tok = 0;
  const metal_compute::SharedBuffer* v = nullptr;
  std::size_t v_off = 0;
  int v_tok = 0;
  int win_h = 0, win_w = 0;
  bool shifted = false;
};

// ---- the int8 tiers ------------------------------------------------------
//
// Both models take the int8 GEMM tiers of the image DiTs (i8-gemm.h) on
// their block projections, matrix cores only: -1 off, 0 accurate (the
// element-type weight quantized per call, groups of 64, K >= 1024), 1 fast
// (held int8 at a coarse rotated group), 2 token (held int8 per output
// channel, K rotated in blocks of 256), 3 token_plain (the same unrotated).
// What a model keeps beside them -- its embedding, its head, the decoder's
// encoder convs -- stays in the element type: those write the pixels or
// read the frame.
//
// The tier a stage asked for (`asked`, -1 for none) as the engine runs it:
// -1 without matrix cores; VPIPE_I8_GEMM, VPIPE_I8_TIER and
// VPIPE_I8_TOKEN_ROT override, as everywhere.
int effective_i8_tier(int asked, bool matrix_cores);
// The group an [N, K] projection is HELD at on `tier` (fast or token), as
// I8GemmContext::fast_group picks it; 0 where the tier holds nothing or its
// layout does not take K -- that projection stays in the element type.
int held_group(int K, int tier);
// What such a projection holds: its codes and scales, or its element-type
// weight (2 bytes an element). `int8`: the checkpoint stores it as
// ComfyUI's int8 (see Proj), which every tier but fast keeps as stored --
// its codes and an f32 scale a row -- and fast converts into its layout.
std::uint64_t proj_bytes(int N, int K, int tier, bool int8 = false);
// Whether a ComfyUI int8 projection of depth K, quantized in ConvRot group
// `rot`, runs AS STORED on `tier`: the token tiers (2 and 3) take a 256
// group's codes as their own layout, the activation rotated alike.
// VPIPE_RELAYVSR_CONVROT_NATIVE=0 converts them instead (the A/B).
bool convrot_native(int K, int tier, int rot);

// The Neural Engine tier's chunk -- the rows a predict takes -- for a
// forward of `rows`: 2048 (where the ANE's rate saturates) from 4096 rows,
// else 1024, else 512, so an auto share that opens at half has a whole
// chunk to take. VPIPE_RELAYVSR_ANE_CHUNK (256 .. 16384) pins it.
int ane_chunk(int rows);

// One block projection as a model holds it: the element-type weight
// [N, K] and its bias, or -- on a tier that holds the layout -- its int8
// codes [N, K] and scales at group `gc` in place of the weight.
//
// ComfyUI's INT8 (a comfy_quant record, int8_tensorwise; Comfy-Org's
// RelayVSR repack) is KEPT AS STORED: `i8` the file's codes [N, K], `i8s`
// their f32 scale a row (N of them) and `rot` the ConvRot group they were
// quantized in (0 unrotated; shared/fp8-layout.h). Off the token tiers
// such a weight is widened to the element type right before each GEMM
// (fp8::Expander: the codes un-rotated exactly, times the scale, rounded
// once -- the host decode to the bit); on them (`cr`) it IS the held
// layout -- `codes` / `scales` alias `i8` / `i8s`, gc the token group --
// and the activation is rotated alike (I8GemmContext::
// quantize_token_convrot), so nothing is quantized twice. The fast tier
// converts it into its own layout at load, from the widened weight.
struct Proj {
  metal_compute::SharedBuffer w;
  metal_compute::SharedBuffer b;            // may be empty
  metal_compute::SharedBuffer codes, scales;
  int gc = 0;                               // 0: the element-type weight
  int n = 0, k = 0;
  metal_compute::SharedBuffer i8, i8s;      // ComfyUI int8, as stored
  int rot = 0;
  bool cr = false;                          // ...run as the token layout
  bool held() const noexcept { return gc != 0; }
  // ComfyUI int8 widened per GEMM.
  bool stored() const noexcept { return gc == 0 && !i8.empty(); }
  bool valid() const noexcept
  {
    return n > 0 && k > 0 &&
           (held() ? !codes.empty() && !scales.empty()
                   : stored() ? !i8s.empty() : !w.empty());
  }
  // The weight as held -- what proj_bytes predicts at the tier the rule
  // names -- and with its bias.
  std::uint64_t weight_bytes() const noexcept
  {
    return held()     ? codes.byte_size() + scales.byte_size()
           : stored() ? i8.byte_size() + i8s.byte_size()
                      : w.byte_size();
  }
  std::uint64_t bytes() const noexcept
  {
    return weight_bytes() + b.byte_size();
  }
};

// The kernels of ONE element type -- f16 for the decoder, bf16 for the
// keyframe DiT -- and the routes over them.
class Ops {
 public:
  Ops();
  ~Ops();
  Ops(const Ops&) = delete;
  Ops& operator=(const Ops&) = delete;

  // False with `err` when a kernel this port cannot run without is missing:
  // a kernel name the metallib does not carry resolves to an INVALID
  // function that dispatches nothing, and nothing above it can tell, so
  // every one this port needs is checked here.
  bool load(metal_compute::MetalCompute* mc, bool bf16, std::string* err);

  bool bf16() const noexcept { return _bf16; }
  // Whether this load runs its GEMMs on the matrix cores (M5 and later;
  // VPIPE_RELAYVSR_NO_MMA2=1 keeps them on the steel tiles).
  bool matrix_cores() const noexcept { return _mma; }
  metal_compute::MetalCompute* mc() const noexcept { return _mc; }

  // y[M, N] = x[M, K] W[N, K]^T (+ b). Byte offsets on x and y.
  void gemm(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& x, std::size_t x_off,
            const metal_compute::SharedBuffer& w,
            const metal_compute::SharedBuffer* b,
            const metal_compute::SharedBuffer& y, std::size_t y_off,
            int M, int N, int K) const;

  // A 3x3 conv over a channel-last [H * W, Cin] map into [OH * OW, Cout],
  // weights [Cout, 9 * Cin] (tap-major), through the shared gather conv:
  // `stride` 2 is torch's Conv2d(k3, s2, p1), `silu` fuses the activation.
  void conv3x3(metal_compute::ComputeEncoder& enc,
               const metal_compute::SharedBuffer& in,
               const metal_compute::SharedBuffer& w,
               const metal_compute::SharedBuffer& b,
               const metal_compute::SharedBuffer& out, int H, int W,
               int Cin, int Cout, int stride, bool silu) const;

  // ---- the int8 tier (see effective_i8_tier) --------------------------
  //
  // Settle it BEFORE the first proj() load: a tier that holds the layout
  // converts each projection as it loads. `tier` is effective_i8_tier's
  // answer; false (tier -1 here) where the context declines -- no matrix
  // cores, or its kernels missing.
  bool set_i8(int tier, bool autotune);
  // The tier this load runs: -1 off.
  int i8_tier() const noexcept { return _i8_tier; }

  // A block projection, `weight` [N, K] and its `bias` ("" for none): the
  // element-type weight through tensor_elt (under `key`), or, on a tier
  // that holds the layout and a K it takes, the layout -- converted once
  // into a derived entry of the set (the element-type weight read
  // uncached and dropped). `hold` false keeps the element type whatever
  // the tier (a weight the ANE stages). `q`, when the checkpoint stores
  // the weight as ComfyUI's int8 (fp8::resolve's answer): kept as stored
  // (see Proj) -- run as the token layout where convrot_native says so,
  // converted on the fast tier. `cache` false reads everything uncached
  // (a weight the load consumes). An invalid Proj on a failure, with
  // *err set when the encoding cannot be run.
  Proj load_proj(WeightSet& ws, const std::string& weight,
                 const std::string& bias, int N, int K,
                 const std::string& key, bool hold = true,
                 const fp8::Weight* q = nullptr, bool cache = true,
                 std::string* err = nullptr);
  // The weight a dense GEMM reads: `p.w`, or a stored int8 weight widened
  // into this load's scratch (encoded into `enc`; valid until the next
  // widening). Null for a held layout, or when the widening cannot run.
  const metal_compute::SharedBuffer*
  dense_weight(metal_compute::ComputeEncoder& enc, const Proj& p) const;
  // THE WIDENING'S SCRATCH is made ONCE, at the widest projection the
  // model keeps as stored (load_proj with `cache`): every widening of a
  // forward fits it, so it never grows under an encoder -- a grown copy
  // stays alive, retained by the command buffer, beside the one it
  // replaced until that buffer completes. A load's own widenings (the
  // keyframe model's cross K / V fold, `cache` false) take it at that size
  // too, or at their own where nothing is kept as stored.
  //   widens()       whether a forward widens (some projection is stored)
  //   drop_widened() give the scratch back, once the GPU work reading it
  //                  has finished -- after the fold, where widens() is
  //                  false: the plan books it only where a forward widens
  //                  (widen_bytes)
  bool widens() const noexcept { return _wdeq_max > 0; }
  void drop_widened() noexcept { _wdeq = metal_compute::SharedBuffer{}; }
  // TEST SEAMS: the scratch's bytes now, and its allocations since load.
  std::uint64_t widened_held() const noexcept { return _wdeq.byte_size(); }
  long widen_allocs() const noexcept { return _n_wdeq_allocs; }
  // What the per-GEMM widening holds: an element-type copy of the widest
  // ComfyUI int8 projection no tier holds (at `tier`) -- 0 for a
  // checkpoint without int8 (`int8` false).
  static std::uint64_t widen_bytes(int tier, bool int8,
                                   const std::vector<int>& n,
                                   const std::vector<int>& k);
  // ComfyUI int8 projections run AS STORED on the token layout since load,
  // and widened per GEMM (engagement checks).
  long convrot_gemms() const noexcept { return _n_cr; }
  long widened_gemms() const noexcept { return _n_widen; }

  // y[M, N] = x[M, K] W^T + b through the projection's route: a held
  // layout's tier GEMM, the accurate tier's per-call quantization where it
  // takes the shape, else the dense GEMM. Byte offsets on x and y.
  void proj(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& x, std::size_t x_off,
            const Proj& p, const metal_compute::SharedBuffer& y,
            std::size_t y_off, int M) const;
  // SEVERAL projections of ONE input (q, k and v): where every one holds
  // the layout at one group, the activation is quantized once and each
  // GEMM reads it; otherwise each takes proj()'s route. Byte offsets.
  struct Out {
    const Proj* p = nullptr;
    const metal_compute::SharedBuffer* y = nullptr;
    std::size_t y_off = 0;
  };
  void proj_shared(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& x, std::size_t x_off,
                   int M, const std::vector<Out>& outs) const;

  // At the top of a forward, with no encoder open: the accurate tier's
  // split-K tuning of the shapes it met (a no-op otherwise).
  void tune_pending();
  // GEMMs run on the int8 route / densely, since load (engagement checks).
  long i8_gemms() const noexcept { return _n_i8; }
  long dense_gemms() const noexcept { return _n_dense; }
  // What the int8 context holds at most in a forward of M rows over a
  // model's block projections (`projs`, each [n, k]) at `tier`. A
  // projection runs the int8 route where the tier holds its layout
  // (held_group) or, held or not, where the ACCURATE route takes its depth
  // (k >= 1024) -- on tier 0 every such one, on a holding tier any whose
  // depth its layout cannot take (the decoder's 1408-deep down on fast and
  // token). Held: the activation's codes and scales at the widest k of
  // them all. Grown only where some projection runs the accurate route:
  // the widest such weight's per-call codes, and -- where its tuner runs
  // (`autotune`) -- the tuner's [2048, n] output and the split-K planes,
  // two f32 [M, n] (M to the 64-row tile) at the widest such n, or
  // I8GemmContext's whole 512 MB budget where one is 9216 deep or more
  // (the tuner weighs every split there). 0 at tier -1.
  struct ProjShape {
    int n = 0, k = 0;
  };
  static std::uint64_t i8_scratch_bytes(int tier, bool autotune, int M,
                                        const std::vector<ProjShape>& projs);

  // Every query of the [H, W] grid over `planes` (1..4). On the matrix
  // cores (local_window_attn_nax) where this load has them, else on the
  // ALU kernel; `i8` there runs SageAttention's int8: 1 the QK (Q and K
  // per token, K smoothed by one mean across the planes), 2 the P*V too
  // (V per channel, one mean and scale across the planes) -- quantized
  // here, into the encoder, before the attention. Without matrix cores
  // `i8` is ignored: int8 has no ALU kernel to run on.
  void window_attention(metal_compute::ComputeEncoder& enc,
                        const metal_compute::SharedBuffer& q,
                        std::size_t q_off, int q_tok,
                        const std::vector<WindowPlane>& planes,
                        const metal_compute::SharedBuffer& out,
                        std::size_t o_off, int o_tok, int H, int W, int Hq,
                        int Hkv, int head_dim, int i8 = 0) const;
  // Whether window_attention runs on the matrix cores (VPIPE_RELAYVSR_NO_
  // NAX_ATTN=1 keeps the ALU kernel), and a test's way to choose.
  bool nax_attention() const noexcept { return _nax_attn; }
  void set_nax_attention(bool on) noexcept { _nax_attn = on && _nax_ok; }
  // TEST SEAM: with the prologue off, an int8 call reuses the operands the
  // last one quantized -- what a benchmark of the attention alone wants.
  void set_window_prologue(bool on) noexcept { _w8_prologue = on; }
  // ...and which matrix-core kernel: STAGED (the default: one window
  // union an 8 x 8 tile, through threadgroup memory) or DIRECT (each
  // simdgroup gathers its own); VPIPE_RELAYVSR_LWA_DIRECT=1 the A/B.
  void set_window_staged(bool on) noexcept { _lwa_staged = on; }
  // The most planes this model's calls attend -- the keyframe's ring, the
  // decoder's four. The int8 prologue sizes every plane's K and V for that
  // many at its first call, as window_int8_bytes books them, rather than
  // growing as the ring fills: each growth allocated the larger copy
  // beside the smaller, and macOS returns a freed buffer's pages only
  // later.
  void set_window_planes(int planes) noexcept
  {
    _w8_planes = std::clamp(planes, 1, 4);
  }
  // What its int8 prologue holds (grown to the largest call, held while
  // the model is): Q's codes and scales, every plane's K codes and
  // scales, K's mean, the statistics' chunks, and with `i8` 2 every
  // plane's V codes and V's mean and step. 0 at `i8` 0.
  static std::uint64_t window_int8_bytes(int planes, int Hq, int Hkv,
                                         int n, int head_dim, int i8);
  // TEST SEAMS: what the prologue holds now, and how many buffers it has
  // allocated since load (each grows once at a fixed geometry).
  std::uint64_t window_int8_held() const noexcept;
  long window_int8_allocs() const noexcept { return _w8_allocs; }

  // Dense attention of `n` token-major queries over `kn` token-major keys
  // (head dim 128, no mask): the keyframe DiT's cross-attention onto its
  // fixed context. On the matrix cores' flash kernel (attn_steel_nax)
  // where window_attention is, else the steel one; `i8` there runs
  // SageAttention's as window_attention's does -- 1 the int8 QK, 2 the
  // P*V too (MetalSageAttention's prologue, into the encoder before the
  // dispatch) -- from 4096 queries, where it pays. Without matrix cores
  // `i8` is ignored.
  void cross_attention(metal_compute::ComputeEncoder& enc,
                       const metal_compute::SharedBuffer& q, int n,
                       const metal_compute::SharedBuffer& k,
                       const metal_compute::SharedBuffer& v, int kn,
                       const metal_compute::SharedBuffer& out,
                       int heads, int i8 = 0) const;
  bool cross_on_matrix_cores() const noexcept { return _xnax; }
  // What its int8 prologue holds (MetalSageAttention's scratch, grown to
  // the largest call). 0 at `i8` 0, and where `n` takes no int8.
  static std::uint64_t cross_int8_bytes(int heads, int n, int kn, int i8);

  // The kernels, by name -- relayvsr.metal's and a few shared ones.
  metal_compute::ComputeFunction upload, lq_rows, rms_silu, patch_rows,
      ln_mod, ln_affine, gated_res, rope_wan, x0, cond_rows, rms_embed,
      rms_heads64, rope3d, swiglu, silu, frame_out, rms, resadd, gelu;

 private:
  metal_compute::MetalCompute* _mc = nullptr;
  bool _bf16 = false;
  bool _mma = false;
  bool _mma_bias = false;
  std::unique_ptr<I8GemmContext> _i8;
  int _i8_tier = -1;
  // The matrix-core window attention: [head dim 64 | 128][f16 | int8 QK |
  // int8 QK and P*V], and the int8 prologue's kernels.
  metal_compute::ComputeLibrary _lib_lwan;
  metal_compute::ComputeFunction _fn_lwan[2][2][3], _fn_lwa_qtok,
      _fn_lwa_stats, _fn_lwa_reduce, _fn_lwa_qv;
  bool _nax_ok = false, _nax_attn = false, _w8_prologue = true;
  bool _lwa_staged = true;
  int _w8_planes = 1;          // set_window_planes
  mutable long _w8_allocs = 0;
  // The prologue's scratch, grow-only.
  mutable metal_compute::SharedBuffer _w8_q, _w8_qs, _w8_k, _w8_ks, _w8_km,
      _w8_part, _w8_v, _w8_vm, _w8_vs;
  // A held layout widened to the element type, for a GEMM its tier
  // declines (none at this model's shapes: grown only if one does) -- and
  // every stored int8 weight's per-GEMM widening.
  mutable metal_compute::SharedBuffer _wdeq;
  mutable long _n_i8 = 0, _n_dense = 0, _n_cr = 0, _n_widen = 0;
  mutable long _n_wdeq_allocs = 0;
  std::size_t _wdeq_max = 0;   // the widest stored projection, bytes
  // The int8 widening (bf16 only), loaded at the first int8 weight.
  std::unique_ptr<fp8::Expander> _x8;
  bool widen_(metal_compute::ComputeEncoder& enc, const Proj& p) const;
  Proj load_int8_(WeightSet& ws, const std::string& weight, Proj p,
                  const std::string& key, bool hold, const fp8::Weight& q,
                  bool cache, std::string* err);
  metal_compute::ComputeLibrary _lib_rv, _lib_gemm, _lib_mma, _lib_elt,
      _lib_rms, _lib_lwa, _lib_attn;
  metal_compute::ComputeFunction _fn_gemm, _fn_gemm_bn64, _fn_gemm_mma,
      _fn_gemm_mma_wide, _fn_gemm_mma_b, _fn_gemm_mma_wide_b, _fn_bias_rows,
      _fn_conv_s2_silu, _fn_conv_s1_silu, _fn_conv_s2, _fn_lwa64, _fn_lwa128;
  // The steel cross-attention, by (queries aligned, keys aligned), and
  // its matrix-core twins by [f16 | int8 QK | int8 QK and P*V][aligned].
  metal_compute::ComputeFunction _fn_xattn[4];
  metal_compute::ComputeLibrary _lib_attn_nax;
  metal_compute::ComputeFunction _fn_xnax[3][4];
  bool _xnax = false;
  std::unique_ptr<MetalSageAttention> _sage_x;
};

// VPIPE_RELAYVSR_PROFILE=1: a keyframe's or a frame's GPU time by section.
// Every mark commits what was encoded since the last one, WAITS for it and
// reopens the encoder on the same stream, so the run is serialized (and an
// ANE split's overlap is gone): read the split, not the total.
class Profile {
 public:
  explicit Profile(const char* what);
  bool on() const noexcept { return _on; }
  // Everything encoded since the last mark (or the start), charged to `cat`.
  void mark(metal_compute::ComputeEncoder& enc, const char* cat);
  // Time spent on the host since the last mark (an ANE predict joined, a
  // staging job), charged to `cat` without a commit.
  void host(const char* cat);
  // One line -- each section's ms and marks -- at info on the session, and
  // reset.
  void report(metal_compute::MetalCompute* mc, const std::string& tail);

 private:
  struct Bucket {
    std::string name;
    double ms = 0.0;
    long marks = 0;
  };
  void add_(const char* cat);
  bool _on = false;
  std::string _what;
  std::vector<Bucket> _b;
  double _t = 0.0;
};

// ---- small host helpers -------------------------------------------------

std::uint16_t to_bf16(float f);
float from_bf16(std::uint16_t h);
std::uint16_t to_f16(float f);
float from_f16(std::uint16_t h);
// `v` rounded to the element type and back.
float round_elt(float v, bool bf16);

// A checkpoint tensor (F32, BF16 or F16) as f32, uncached. For SMALL
// tensors only: macOS keeps a freed large host block in the process's
// footprint, so a large tensor is read with read_raw and converted
// element by element into the buffer that keeps it.
bool read_f32(WeightSet& ws, metal_compute::MetalCompute* mc,
              const std::string& name, std::vector<float>* out,
              std::vector<std::int64_t>* shape = nullptr);

// A checkpoint tensor as the file stores it, read uncached into a Metal
// buffer (not the host heap), and its element i as f32.
struct RawTensor {
  metal_compute::SharedBuffer buf;
  std::vector<std::int64_t> shape;
  enum class Kind : int { kF32, kBF16, kF16 } kind = Kind::kF32;
  std::size_t n = 0;

  float at(std::size_t i) const
  {
    const void* p = buf.contents();
    switch (kind) {
      case Kind::kBF16: return from_bf16(((const std::uint16_t*)p)[i]);
      case Kind::kF16:  return from_f16(((const std::uint16_t*)p)[i]);
      default:          return ((const float*)p)[i];
    }
  }
};
bool read_raw(WeightSet& ws, metal_compute::MetalCompute* mc,
              const std::string& name, RawTensor* out);
// `t` in the element type, rounded once from its f32 value.
metal_compute::SharedBuffer convert_elt(metal_compute::MetalCompute* mc,
                                        const RawTensor& t, bool bf16);
// A 3x3 conv's weight [Cout, Cin, 3, 3] tap-major for the gather conv,
// [Cout, 9, Cin], in the element type. Empty when `t` is not a 3x3 conv.
metal_compute::SharedBuffer tap_major_elt(metal_compute::MetalCompute* mc,
                                          const RawTensor& t, bool bf16);
// A buffer of `v` in the element type.
metal_compute::SharedBuffer upload_elt(metal_compute::MetalCompute* mc,
                                       const std::vector<float>& v,
                                       bool bf16);
metal_compute::SharedBuffer upload_f32(metal_compute::MetalCompute* mc,
                                       const std::vector<float>& v);
// A checkpoint tensor in the element type, held by the model (a derived
// entry of the set under `key`, so two loads of one checkpoint share it).
metal_compute::SharedBuffer tensor_elt(WeightSet& ws,
                                       metal_compute::MetalCompute* mc,
                                       const std::string& name, bool bf16,
                                       const std::string& key);

}  // namespace relayvsr
}  // namespace genai
}  // namespace vpipe

#endif
