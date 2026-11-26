#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_METAL_MAGI2_REFINER_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_METAL_MAGI2_REFINER_H

// MAGI-2-preview's REFINER (refiner/, written from the release's
// inference/model/magi2_refiner.py and pipeline/refiner_data_proxy.py --
// read, not run): the second stage of its 1080p preset, a dense DiT that
// re-denoises the preview's latent, trilinearly upsampled, in a few steps.
//
// 30 layers at 4096 wide: RMSNorm (weight + 1), q|k|v with grouped-query
// attention (32 query heads over 8 key/value heads, 128 wide), q / k
// RMSNorm and the element-wise Fourier rope on dims [0, 96), a sigmoid
// gate per head, then RMSNorm and a swiglu7 MLP (int(4 x 4096 x 2/3) // 4
// x 4 = 10920); no hyper-connections, no sinks, NO TIME INPUT. Layers
// {0, 1, 28, 29} keep a weight per modality (video, audio, text).
//
// THE WINDOW. Video tokens are grouped into blocks of (8 frames, 4, 4)
// cells (edge blocks smaller) and a block attends the key blocks within a
// radius of (2, 2, 2) in the block grid, plus every text token; text
// attends everything. Here each block is laid out contiguously and
// PADDED to a multiple of 32 tokens, so every 32-row query tile lies in
// one block and attends one list of 16-key blocks -- the steel attention's
// block-sparse spans -- the padding keys excluded by an additive mask row.
//
// THE RESIDUAL is f32: the release rounds the embedding to bf16 and adds
// layer 0's attention in bf16, after which its MLP's f32 output promotes
// the stream to f32 for good.

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/dit-flops.h"
#include "generative-models/shared/sage-attention.h"
#include "generative-models/tensor-parallel.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class MetalSageAttention;
class TpOverlap;
class WeightSet;
namespace magi2 {
class Gemms;
struct Proj;
}  // namespace magi2

class MetalMagi2Refiner {
 public:
  struct Config {
    int hidden = 4096;
    int head_dim = 128;
    int n_q = 32, n_kv = 8;
    int video_c = 48, audio_c = 64, text_c = 5120;
    int mlp_inter = 10920;
    std::vector<int> layers;      // the checkpoint's indices, in order
    std::vector<char> mm;         // per modality (video, audio, text)
    int block[3] = {8, 4, 4};     // (frames, rows, columns) a block
    int radius[3] = {2, 2, 2};    // blocks each way
    float eps = 1e-6f;
    int mlp_rows = 16384;         // the MLP runs in row chunks of this
    // ACCELERATION (magi2-accel.h). The int8 tier the projections run, as
    // the engine settled it (magi2::effective_i8_tier: -1 off, 0 accurate,
    // 1 fast, 2 token, 3 token_plain) -- a holding tier converts them at
    // load -- and whether its split-K tuner runs; SageAttention on the
    // window attention (matrix cores only).
    int i8_tier = -1;
    bool autotune = true;
    sage::Config sage;
    // TENSOR PARALLEL (tensor-parallel.h): this model as one RANK of the
    // refiner split across hosts -- its heads are the query heads (n_heads
    // n_q, a share whole kv groups), its feed-forward rows the MLP's
    // (n_ffn mlp_inter): its q | k | v, gate and output projection by
    // heads, its up | gate and down by rows. The norms, embedders and
    // head whole on every rank. The attention's output projection and the
    // MLP run in row BANDS (mlp_rows), each band's partial sum exchanged
    // with the other ranks' while the GPU works on the next band.
    TpShare tp;
  };

  // A RANK's widths (Config::tp; the whole model's when inactive): query
  // heads [h0, h1), their kv heads [kv0, kv1), MLP rows [f0, f1).
  struct Share {
    int h0 = 0, h1 = 0, kv0 = 0, kv1 = 0, f0 = 0, f1 = 0;
    int heads() const noexcept { return h1 - h0; }
    int kv() const noexcept { return kv1 - kv0; }
    int rows() const noexcept { return f1 - f0; }
  };
  // False, with `err`, for a share this config cannot take.
  static bool share_of(const Config& c, Share* out, std::string* err);
  // The refiner's share at the FRACTIONS of another model's (the preview
  // DiT's, whose widths a stage splits by): every boundary scaled to this
  // model's widths, the heads' to whole kv groups -- so the ranks' shares
  // tile it as theirs tile the other.
  static TpShare share_at(const Config& c, const TpShare& other);

  // Shapes off the tensors; `subset` names the layers to run (empty: all).
  static bool config_from_weights(WeightSet& ws, Config* out,
                                  std::string* err,
                                  const std::vector<int>& subset = {});
  static std::unique_ptr<MetalMagi2Refiner>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, std::string* err);
  ~MetalMagi2Refiner();
  MetalMagi2Refiner(const MetalMagi2Refiner&) = delete;
  MetalMagi2Refiner& operator=(const MetalMagi2Refiner&) = delete;

  // A forward's input: the noisy video latent [video_c, T, H, W] and the
  // conditioner's rows [text_len, text_c], host f32.
  struct Sample {
    int T = 0, H = 0, W = 0;
    const float* video = nullptr;
    int text_len = 0;
    const float* text = nullptr;
    // The sampling step (0-based; -1 unknown): sage_pv8_from_step's clock.
    int step = -1;
  };
  // The velocity [video_c, T, H, W].
  bool forward(const Sample& s, std::vector<float>* vel, std::string* err);

  // Called with (layers done, layers) after each layer; false STOPS the
  // forward (it returns false with an empty error).
  void set_block_progress(std::function<bool(int, int)> fn)
  {
    _block_cb = std::move(fn);
  }

  const Config& config() const noexcept { return _cfg; }

  // ---- tensor parallel (Config::tp), as MetalMagi2Dit's ---------------
  // A clip's channel; `first_seq` the exchange count the clip has reached
  // (the preview DiT's, over the same channel before it).
  bool begin_tp_generation(std::shared_ptr<TpChannel> ch,
                           std::function<bool()> stop,
                           std::uint64_t first_seq, std::string* err);
  void end_tp_generation();
  std::uint64_t tp_exchanges() const noexcept { return _tp_exchanges; }
  // Whether the last forward ran its bands' exchanges under the next
  // band's work (VPIPE_MAGI2_TP_LAG=0: in line, the same arithmetic).
  bool tp_overlapped() const noexcept { return _tp_lagged; }

  // ---- the sequence (host; public for tests and the plan) ------------
  // Laid out for an attention kernel's query tile `bq` and key block `bk`
  // (32 / 16 on the steel kernel, 64 / 32 on the matrix-core one).
  struct Seq {
    int bq = 32, bk = 16;
    int nv = 0;                 // video tokens (T H W)
    int nv_pad = 0;             // the padded video span
    int n = 0;                  // the padded sequence
    // Per padded slot: a video token's raster index, nv + j for text
    // token j, or -1 for padding.
    std::vector<int> src;
    // Per query tile, its key blocks (CSR).
    std::vector<int> qb_off, qb_blocks;
    // The (query, key) pairs a head's window attention visits: each query
    // tile's rows (the last one's to the sequence's end) times its key
    // blocks' keys (likewise) -- the padding slots inside them included,
    // which the mask row zeroes but the kernel multiplies.
    long long pairs = 0;
  };
  static Seq sequence(const Config& c, int T, int H, int W, int text_len,
                      int bq = 32, int bk = 16);

  // ---- the plan --------------------------------------------------------
  // What a load of `c` holds, off the tensor headers: every tensor as
  // stored but the audio embedder and head it does not read, the video
  // head's norm folded into its f32 linear -- and on a tier that holds the
  // int8 layout (c.i8_tier), each projection it takes at that layout.
  static std::size_t weight_bytes(WeightSet& ws, const Config& c);
  std::size_t resident_bytes() const;
  void release_scratch();
  // What a forward of this geometry allocates -- with `matrix_cores`,
  // laid out for the matrix-core attention's wider query tile (the
  // larger of the two), and with c's acceleration: the int8 context's
  // grow-only scratch at c.i8_tier and SageAttention's prologue where it
  // runs (magi2::sage_runs).
  static std::size_t scratch_bytes(const Config& c, int T, int H, int W,
                                   int text_len, bool matrix_cores = true);
  // The M5 routes: matmul2d GEMMs, the matrix-core attention.
  // VPIPE_MAGI2_NO_MMA2 / VPIPE_MAGI2_NO_NAX_ATTN are the A/Bs.
  bool uses_mma() const noexcept { return _mma; }
  bool uses_nax_attention() const noexcept { return _nax; }
  // The acceleration this load runs: the int8 tier (-1 off) and its GEMMs
  // on the int8 route / dense since load; SageAttention, and its attention
  // calls with the int8 QK alone / with the P*V too since load.
  int i8_tier() const noexcept;
  long i8_gemms() const noexcept;
  long dense_gemms() const noexcept;
  bool sage_on() const noexcept { return _sage != nullptr; }
  long sage_calls() const noexcept { return _n_sage; }
  long sage_pv8_calls() const noexcept { return _n_sage_pv8; }

  // The GPU arithmetic forward() credits to the achieved-FLOP/s meter
  // (common/perf-flops.h), counted as it is encoded and credited as each
  // layer's command buffer retires: every projection and f32 linear as
  // 2MNK over the rows it runs (the padded sequence's), and the window
  // attention at Seq::pairs a head (with SageAttention, as the attention
  // it stands for).
  const GpuFlopsTally& gpu_flops() const noexcept { return _flops; }

 private:
  MetalMagi2Refiner() = default;
  struct Layer;
  struct Kernels;
  struct Scratch;
  bool load_(std::string* err);
  bool ensure_scratch_(const Seq& q, std::string* err);
  // SageAttention's object (its kernels and its prologue's scratch),
  // made again after release_scratch() gave the scratch back.
  bool load_sage_(std::string* err);

  std::shared_ptr<WeightSet> _ws;
  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::vector<Layer> _layers;
  std::vector<float> _bands;
  metal_compute::SharedBuffer _emb_w[2], _emb_b[2];   // video, text
  metal_compute::SharedBuffer _out_w;                  // video, folded
  std::unique_ptr<Kernels> _k;
  std::unique_ptr<magi2::Gemms> _g;
  std::unique_ptr<MetalSageAttention> _sage;
  std::unique_ptr<Scratch> _s;
  metal_compute::CommandStream _stream;
  GpuFlopsTally _flops;   // see gpu_flops()
  std::function<bool(int, int)> _block_cb;
  bool _mma = false, _nax = false;
  bool _sage_wanted = false;
  long _n_sage = 0, _n_sage_pv8 = 0;
  // This rank's widths, the clip's channel and its band exchanges: the
  // attention's partials in lanes 0 and 1, the MLP's in 2 and 3, a band
  // apiece by its parity.
  Share _sh;
  static constexpr int kTpLanes = 4;
  std::shared_ptr<TpChannel> _tp_ch;
  std::unique_ptr<TpOverlap> _tp_pipe;
  std::uint64_t _tp_seq = 0, _tp_exchanges = 0;
  bool _tp_lagged = false;
  metal_compute::SharedBuffer _tp_part[kTpLanes], _tp_recv[kTpLanes];
  bool tp_buffer_(metal_compute::SharedBuffer& b, std::size_t bytes);
};

}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_METAL_MAGI2_REFINER_H
