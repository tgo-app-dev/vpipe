#ifndef GENERATIVE_MODELS_FLASHVSR_FLASHVSR_LQ_PROJ_H
#define GENERATIVE_MODELS_FLASHVSR_FLASHVSR_LQ_PROJ_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;    // generative-models/weight-set.h
struct MmaSplitK;   // generative-models/shared/mma-splitk.h

// FlashVSR's source projection: the low-quality clip, in PIXELS, turned
// into rows the denoiser adds into its residual stream.
//
// This is the whole of how the source reaches the model. There is no VAE
// encode on the way in and no cross-attention to it; the denoiser's only
// other input is noise.
//
//   pixel unshuffle (1,16,16)   3 x H x W  ->  768 x H/16 x W/16
//   causal conv3d (4,3,3) s(2,1,1)          ->  2048, frames halved
//   RMS norm over channels, SiLU
//   causal conv3d (4,3,3) s(2,1,1)          ->  3072, frames halved
//   RMS norm over channels, SiLU
//   linear 3072 -> 1536                     ->  one row per token
//
// The two stride-2 temporal convolutions are why 4 source frames become
// 1 row frame, which is exactly the VAE's temporal compression -- so the
// rows land on the denoiser's own token grid with no resampling. The
// 16x16 spatial unshuffle does the same for the other two axes: the VAE
// is 8x and the patch is 2, so the token grid is the frame over 16.
//
// THE CONVOLUTIONS ARE CAUSAL AND CARRY STATE. Each keeps the last two
// INPUT frames it saw, so the next call's first output can reach back
// across the boundary rather than seeing a zero left edge. That is what
// makes a clip processed in pieces identical to the same clip processed
// whole, and it is the reason this class is stateful at all.
//
// THE ORDER OF THE CARRY IS THE TRAP. A chunk must be convolved against
// the PREVIOUS chunk's tail and only then overwrite it. The reference
// ships both spellings side by side -- its buffered variant stores the
// current tail before convolving, so every chunk pads with its own last
// two frames instead of its predecessor's -- and the checkpoint this
// class loads was trained with the causal one. The difference is silent:
// both produce rows of the right shape, and the wrong one degrades
// gently across a boundary rather than failing.
//
// THE FIRST CALL RETURNS NOTHING, and that is not an error. It prepends
// three copies of the opening frame, runs the first convolution to seed
// both carries, and produces no rows -- the second call is the first
// that can. A caller that treats an empty return as failure will refuse
// every clip.
class FlashVsrLqProj {
 public:
  struct Config {
    int in_channels  = 3;
    int shuffle_h    = 16;
    int shuffle_w    = 16;
    int hidden1      = 2048;
    int hidden2      = 3072;
    int out_dim      = 1536;   // the denoiser's residual width
    // How many blocks of the stack get their own output projection. The
    // module is built for up to one per block; the v1.1 checkpoint
    // ships ONE, so the source is injected at block 0 alone. Read from
    // the checkpoint rather than assumed, because the count is the
    // difference between conditioning one block and thirty.
    int n_out_layers = 1;
    // The temporal carry, in frames. A property of the (4,3,3) kernel
    // with its causal left padding, not a tuning choice.
    static constexpr int kCarry = 2;
  };

  // `prefix` is what the projection's tensor names carry in `ws`:
  // "lq_proj." in a converted directory, where they share one index with
  // the denoiser, and nothing in the published `LQ_proj_in.ckpt`, which
  // holds the projection alone.
  static std::unique_ptr<FlashVsrLqProj>
  load(std::shared_ptr<WeightSet>   ws,
       metal_compute::MetalCompute* mc,
       std::string*                 err = nullptr,
       const std::string&           prefix = "lq_proj.");

  ~FlashVsrLqProj();

  // FOR THE MEMORY PLAN, from the checkpoint's header alone (no load).
  //
  // What the projection's weights hold, bf16: the tensors under `prefix`
  // in the set at `source` -- which, in a converted directory, is the
  // denoiser's own set, so its file size would be the wrong answer.
  static std::size_t weight_bytes(const std::string& source,
                                  const std::string& prefix);
  // What projecting one clip of `frames` source frames at height x width
  // holds beyond the weights: the im2col band, a call's activations (four
  // source frames in, the reference's cadence), both convolutions'
  // carries, and the clip's rows -- kept until the beat is assembled, and
  // then copied into it. 741 MB at 1920x1152 x 25 against 748 MB of live
  // buffers measured.
  static std::size_t working_bytes(const std::string& source,
                                   const std::string& prefix, int height,
                                   int width, int frames);

  const Config& config() const noexcept { return _cfg; }

  // Drop the carries and return to the opening state. MUST be called
  // between clips: a second clip convolved against the first one's tail
  // is conditioned on the end of a video it has nothing to do with.
  void reset();

  // One piece of the source clip.
  //
  // `lq` is planar u8 RGB [n, 3, H, W] -- the beat's own layout, taken
  // as u8 so the scale into [-1,1] happens inside the unshuffle instead
  // of in a full-size f32 copy the caller would have to hold.
  //
  // Appends one row set per row frame produced to `out`, each
  // [tokens, out_dim] bf16 with tokens = n_rows * (H/16) * (W/16).
  // Produces NOTHING on the first call after reset(); see the class
  // comment. False means it warned through `err`.
  bool stream_forward(const std::uint8_t* lq, int n, int height, int width,
                      std::vector<metal_compute::SharedBuffer>* out,
                      int* out_row_frames, std::string* err = nullptr);

  std::uint64_t resident_bytes() const noexcept;

 private:
  FlashVsrLqProj() = default;

  struct Conv;
  struct Carry;
  struct Ctx;    // one forward's scratch pool and encoder

  std::unique_ptr<Conv> load_conv_(WeightSet& ws, const std::string& nm,
                                   std::string* err);
  metal_compute::SharedBuffer load_vec_(WeightSet& ws, const std::string& nm);
  metal_compute::SharedBuffer load_mat_(WeightSet& ws, const std::string& nm);

  // One output frame of a (4,3,3) causal convolution: gather its four
  // temporal taps through the replicate im2col, then one GEMM. `taps`
  // are already-resolved frame views, so replication at the sequence
  // start is a repeated pointer rather than a branch in here.
  void conv_frame_(Ctx& cx, const Conv& c,
                   const metal_compute::SharedBuffer* const taps[4],
                   const std::size_t tap_off[4],
                   const metal_compute::SharedBuffer& out,
                   std::size_t out_off, int OH, int OW);
  void gemm_bias_(Ctx& cx, const metal_compute::SharedBuffer& x,
                  const metal_compute::SharedBuffer& w,
                  const metal_compute::SharedBuffer& b,
                  const metal_compute::SharedBuffer& y, int M, int N, int K,
                  std::size_t y_off = 0, std::size_t x_off = 0);
  metal_compute::SharedBuffer&
  norm_silu_(Ctx& cx, const metal_compute::SharedBuffer& x, std::size_t rows,
             int C, const metal_compute::SharedBuffer& gamma);
  // Keep the last two frames of a sequence of `m`, whose frame j lives
  // at `frame_offs[j]` elements into `src`. Slots are addressed modulo
  // the running count, so a one-frame chunk beside a two-frame history
  // moves nothing.
  void save_carry_(Ctx& cx, Carry& carry,
                   const metal_compute::SharedBuffer& src,
                   const std::vector<std::size_t>& frame_offs,
                   std::size_t hw, int cin);

  Config                       _cfg{};
  metal_compute::MetalCompute* _mc = nullptr;
  std::shared_ptr<WeightSet>   _ws;

  metal_compute::ComputeLibrary  _lib_elt, _lib_gemm, _lib_rms;
  metal_compute::ComputeFunction _fn_unshuffle, _fn_im2col, _fn_silu,
                                 _fn_copy, _fn_rms, _fn_gemm;

  // MATRIX CORES, on by default where the GPU has them. Every GEMM here
  // is a convolution's column matrix, and the columns are DEEP: 36 taps
  // over the unshuffled channels is K = 27648 into conv1 and 73728 into
  // conv2, which is the regime a single-op contraction stalls in -- so
  // the deep ones go through the split-K planes. VPIPE_FVSR_NO_MMA2 is
  // the A/B, shared with the denoiser; VPIPE_LM_NO_SPLITK drops only the
  // split.
  bool _use_mma2 = false;
  metal_compute::ComputeLibrary  _lib_dense_mma;
  metal_compute::ComputeFunction _fn_gemm_mma, _fn_gemm_mma_wide,
                                 _fn_bias_rows;
  std::unique_ptr<MmaSplitK>     _splitk;

 public:
  // Whether the convolutions run on the matrix cores. A silent fallback
  // to the ALU kernel is several times slower and numerically the same
  // answer, so only a path-selection check can see it.
  bool uses_matrix_cores() const noexcept { return _use_mma2; }

 private:

  std::unique_ptr<Conv>  _conv1;
  std::unique_ptr<Conv>  _conv2;
  metal_compute::SharedBuffer _gamma1;   // [hidden1]
  metal_compute::SharedBuffer _gamma2;   // [hidden2]
  std::vector<metal_compute::SharedBuffer> _out_w;   // [out_dim, hidden2]
  std::vector<metal_compute::SharedBuffer> _out_b;   // [out_dim]

  std::unique_ptr<Carry> _carry1;
  std::unique_ptr<Carry> _carry2;
  bool _opened = false;   // false until the seeding call has run
};

}  // namespace genai
}  // namespace vpipe

#endif
