#ifndef VPIPE_GENERATIVE_MODELS_MAGI2_METAL_MAGI2_DIT_H
#define VPIPE_GENERATIVE_MODELS_MAGI2_METAL_MAGI2_DIT_H

// MAGI-2-preview's preview-stage DiT (sand-ai/MAGI-2-preview, `preview/`;
// Apache-2.0), written from the release's inference/model/magi2_preview.py
// and its data proxy -- read, not run.
//
// ONE packed sequence a forward: the video latent's tokens, the audio
// latent's, the prompt's text features, and for image-to-video an image's
// latent tokens behind a special token pooled from the prompt. Each token
// is embedded by its modality's f32 linear into 4 streams x hidden (the
// hyper-connections' residual), the first 64 channels then REPLACED by a
// sinusoid of the token's timestep (t * 1000; clean conditioning t = 0).
//
// 40 layers, each two sublayers wrapped in a hyper-connection over the 4
// streams (a norm over the whole row, a f32 projection to pre / post /
// res coefficients -- sigmoid, 2 sigmoid, Sinkhorn): attention (per-head
// sigmoid gate, q / k RMSNorm, an element-wise Fourier rope, one sink a
// head) and an MLP -- swiglu7 in the four "mm" layers, a multi-head MoE
// (12 heads of 256, 256 experts each, top 6) with shared and
// modality-specific experts in the rest. In the mm layers every norm and
// linear is per MODALITY (video, audio, text). A f32 head per output
// modality reads the whole 4-stream row.
//
// TOKEN ORDER. The release permutes tokens by modality for its grouped
// linears and back for attention; attention over one sample is
// permutation-invariant (no mask, a coordinate per token), so this lays
// the sequence out grouped once and never permutes:
//   [video | image tokens] [audio] [text | special tokens]
// -- three contiguous modality spans.

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
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
class WiredPool;
namespace magi2 {
class Gemms;
struct Proj;
}  // namespace magi2

class MetalMagi2Dit {
 public:
  struct Config {
    int hidden = 3072;
    int head_dim = 128;
    int n_q = 24, n_kv = 24;
    int streams = 4;
    int video_c = 48, audio_c = 64, text_c = 5120;
    int time_dim = 64;
    // The multi-head MoE: the row cut into moe_heads of hidden / moe_heads,
    // each routed to top_k of moe_experts (sigmoid scores, the EMA bias for
    // the choice, probs renormalized x route_scale).
    int moe_heads = 12, moe_experts = 256, top_k = 6;
    float route_scale = 4.9f;
    int expert_inter = 1280, shared_inter = 1280, ms_inter = 1280;
    int mlp_inter = 8192;
    // The layers this model runs, in order -- the checkpoint's indices
    // (all of them, or a subset a test names) -- and their kinds.
    std::vector<int> layers;
    std::vector<char> mm, moe;
    int sk_iters = 20;
    float sk_eps = 1e-12f;
    float eps = 1e-6f;
    // EXPERT RESIDENCY. A MoE layer's routed experts (W_gate, W_up, W_down:
    // 5.6 GB a layer at the release's shapes) are the model -- 202.5 of
    // its 212 GB. Everything else (~10 GB) is held; of the experts, the
    // first `resident_moe` MoE layers are held too -- read uncached, and
    // each admitted through the session's wired pool, past which it
    // streams -- and the rest STREAM through two slots, each read issued
    // under the previous layer's GPU work (pread, F_NOCACHE). -1: every
    // MoE layer the pool admits held.
    int resident_moe = -1;
    // ACCELERATION (magi2-accel.h). The int8 tier the block projections
    // run, as the engine settled it (magi2::effective_i8_tier: -1 off, 0
    // accurate, 1 fast, 2 token, 3 token_plain) -- a holding tier converts
    // them at load; the routed experts keep their own bf16 kernel -- and
    // whether its split-K tuner runs; SageAttention on the matrix-core
    // attention (its sinks kept).
    int i8_tier = -1;
    bool autotune = true;
    sage::Config sage;
    // TENSOR PARALLEL (tensor-parallel.h): this model as one RANK of a DiT
    // split across hosts. Its heads are the attention's (n_heads n_q), its
    // feed-forward rows count EXPERTS (n_ffn moe_experts): the rank holds
    // its query heads of every q | k | v, gate and sink and their columns
    // of the output projection; of each MoE head's routed experts the run
    // [ffn0, ffn1); and of the shared, modality and mm-layer MLPs the
    // rows at the same fraction (Share). Routing, norms, the hyper-
    // connections, the split / merge projections, embedders and heads
    // run whole on every rank. Each sublayer's output is a PARTIAL sum,
    // added to the other ranks' (begin_tp_generation's channel) before
    // its hyper-connection takes it: twice a layer. Inactive: the whole
    // model.
    TpShare tp;
  };

  // A RANK'S WIDTHS (Config::tp; the whole model when it is inactive):
  // query heads [h0, h1) and their kv heads [kv0, kv1); each MoE head's
  // experts [e0, e1); and the dense widths at the same fraction f of the
  // experts, cut at f * W / moe_experts -- the shared and modality-
  // specific experts' [sh0, sh1) and [ms0, ms1), the mm layers' MLP
  // [mlp0, mlp1). The ranks' shares tile every width.
  struct Share {
    int h0 = 0, h1 = 0, kv0 = 0, kv1 = 0;
    int e0 = 0, e1 = 0;
    int sh0 = 0, sh1 = 0, ms0 = 0, ms1 = 0, mlp0 = 0, mlp1 = 0;
    int heads() const noexcept { return h1 - h0; }
    int kv() const noexcept { return kv1 - kv0; }
    int experts() const noexcept { return e1 - e0; }
    int shared() const noexcept { return sh1 - sh0; }
    int ms() const noexcept { return ms1 - ms0; }
    int mlp() const noexcept { return mlp1 - mlp0; }
  };
  // False, with `err`, for a share this config cannot take: its widths
  // are not the model's (n_heads n_q, n_ffn moe_experts), a range is
  // empty, or a query-head range splits a kv group.
  static bool share_of(const Config& c, Share* out, std::string* err);

  // The routed experts of one MoE layer, in bytes (all three tensors).
  std::uint64_t expert_layer_bytes() const noexcept { return _expert_bytes; }
  // What the model holds: every weight buffer -- the trunk, the held
  // experts, the two expert slots -- the scratch apart.
  std::size_t resident_bytes() const;
  // Gives the forward's scratch back between generations.
  void release_scratch();

  // Called with (layers done, layers) as a forward's layers finish on the
  // GPU -- at each streamed layer's wait, and once at the end -- so a bar
  // moves inside a step; false STOPS the forward (forward() then returns
  // false with an empty error).
  void set_block_progress(std::function<bool(int done, int total)> fn)
  {
    _block_cb = std::move(fn);
  }

  // ---- the plan, before anything loads ------------------------------
  // The trunk as the load holds it (everything but the routed experts),
  // the block projections at the layout c.i8_tier holds them in.
  static std::size_t trunk_bytes(WeightSet& ws, const Config& c);
  // The forward's scratch for a batch of `batch` samples of n tokens,
  // each with `inputs` input elements (video, images, audio and text
  // rows) -- what ensure_scratch_ and the batch's own buffers allocate --
  // and with c's acceleration: the int8 context's grow-only scratch at
  // c.i8_tier and SageAttention's prologue where it runs on a box with
  // (`matrix_cores`) or without them.
  static std::size_t scratch_bytes(const Config& c, int n, int batch,
                                   int inputs, bool matrix_cores = true);
  // MoE layers held / streamed.
  int moe_held() const noexcept { return _moe_held; }
  int moe_streamed() const noexcept { return _moe_streamed; }
  // The M5 routes: matmul2d GEMMs, the matrix-core attention.
  // VPIPE_MAGI2_NO_MMA2 / VPIPE_MAGI2_NO_NAX_ATTN are the A/Bs.
  bool uses_mma() const noexcept;
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
  // The routed experts on the matrix cores (magi2_nax.metal; M5 and an
  // expert width of whole 64s; VPIPE_MAGI2_NO_NAX_MOE=1 the A/B), and its
  // dispatches since load.
  bool uses_nax_experts() const noexcept { return _nax_moe; }
  long nax_expert_dispatches() const noexcept { return _n_moe_nax; }
  // ...and without matrix cores the STAGED simdgroup kernel (the weights
  // through threadgroup memory; VPIPE_MAGI2_MOE_SG=1 the first one).
  bool uses_staged_experts() const noexcept { return _tg_moe; }

  // The GPU arithmetic forward() credits to the achieved-FLOP/s meter
  // (common/perf-flops.h), counted as it is encoded and credited as each
  // command buffer retires (a flush at every layer and before every
  // commit): every projection and f32 linear as 2MNK -- embedders,
  // hyper-connection dots, routers, heads included -- the attention at
  // n x n pairs a head (with SageAttention, as the dense attention it
  // stands for), and each routed expert pick as its three [256 x
  // expert_inter] products. A RANK computes only the picks of its own
  // experts, which the GPU sorts out; it counts them at its share of the
  // experts (e1 - e0 of moe_experts), what an even routing gives it.
  const GpuFlopsTally& gpu_flops() const noexcept { return _flops; }

  // Every shape off the tensors; the release's constants (top_k, the route
  // scale, Sinkhorn's iterations) are not in the file. `subset` names the
  // layers to run (empty: all, in order).
  static bool config_from_weights(WeightSet& ws, Config* out,
                                  std::string* err,
                                  const std::vector<int>& subset = {});

  static std::unique_ptr<MetalMagi2Dit>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, std::string* err);

  ~MetalMagi2Dit();
  MetalMagi2Dit(const MetalMagi2Dit&) = delete;
  MetalMagi2Dit& operator=(const MetalMagi2Dit&) = delete;

  // One image of image-to-video: its VAE latent [C, 1, h, w] and the
  // special token pooled from the prompt (text_c f32).
  struct Image {
    const float* latent = nullptr;
    int h = 0, w = 0;
    const float* special = nullptr;
  };

  // One forward's input, host f32. The text features are the conditioner's
  // [text_len, text_c]; `t` the step's normalized timestep (the scheduler's
  // integer timestep / 1000), every video and audio token's.
  struct Sample {
    int T = 0, H = 0, W = 0;
    const float* video = nullptr;    // [video_c, T, H, W]
    int audio_len = 0;
    const float* audio = nullptr;    // [audio_len, audio_c]
    int text_len = 0;
    const float* text = nullptr;     // [text_len, text_c]
    std::vector<Image> images;
    float t = 0.0f;
    // The sampling step (0-based; -1 unknown): sage_pv8_from_step's clock.
    int step = -1;
  };

  // TESTS: the MoE picks of layer `idx` given -- [tokens, heads, top_k]
  // head-local experts in this model's token order -- so a comparison
  // with a reference does not turn on a near tie. Cleared by an empty
  // vector.
  void force_routes(int idx, std::vector<int> picks);
  // TESTS: keep each MoE layer's picks of the next forwards ([tokens,
  // heads, top_k] head-local experts, this model's token order).
  void keep_routes(bool on) { _keep_routes = on; }
  std::vector<int> routes(int idx) const;

  // The velocity: video [video_c, T, H, W], audio [audio_len, audio_c].
  bool forward(const Sample& s, std::vector<float>* vel_video,
               std::vector<float>* vel_audio, std::string* err);
  // A BATCH -- a CFG step's conditional and unconditional passes -- run
  // layer by layer, every sample through a layer before the next one, so
  // a streamed layer's experts are read ONCE for the batch. Each sample
  // keeps its own residual stream; the rest of the scratch is shared.
  bool forward(const std::vector<const Sample*>& batch,
               std::vector<std::vector<float>>* vel_video,
               std::vector<std::vector<float>>* vel_audio, std::string* err);

  const Config& config() const noexcept { return _cfg; }

  // ---- tensor parallel (Config::tp) ------------------------------------
  // A RANK's link to the others for ONE generation -- a clip -- as
  // MiniMax H3's (tensor-parallel.h): opened before its first forward,
  // let go when it ends, however it ends; the exchange count new each
  // clip. `stop` is the run's, which the channel polls while it waits.
  bool begin_tp_generation(std::shared_ptr<TpChannel> ch,
                           std::function<bool()> stop, std::string* err);
  void end_tp_generation();
  // Exchanges since load, and whether the last forward overlapped a
  // sample's exchanges with the next sample's work (a batch of two or
  // more; VPIPE_MAGI2_TP_LAG=0 runs them in line, the same arithmetic).
  std::uint64_t tp_exchanges() const noexcept { return _tp_exchanges; }
  // This clip's exchanges so far (a later model on the same channel
  // counts on from it).
  std::uint64_t tp_seq() const noexcept { return _tp_seq; }
  bool tp_overlapped() const noexcept { return _tp_lagged; }
  // The rank's exchange buffers for a batch of `batch` samples of n tokens
  // (in scratch_bytes on a rank).
  static std::size_t tp_exchange_bytes(const Config& c, int n, int batch);

  // ---- the sequence (host; public for tests) ----------------------------
  //
  // The rope table of a sample in this model's token order: per token the
  // f32 sin and cos [head_dim * 3 / 8] the release's element-wise Fourier
  // embedding gives (coordinates (t, h, w), the size and reference size
  // per axis, h and w centred), and the token count of each span.
  struct Layout {
    int n = 0;                 // tokens
    int nv = 0;                // video span: generated + image tokens
    int na = 0;                // audio span
    int nt = 0;                // text span: prompt + special tokens
    std::vector<float> sin, cos;   // [n, R]
    std::vector<float> coords;     // [n, 9], the release's columns
  };
  static Layout layout(const Sample& s, const std::vector<float>& bands);
  // sinusoidal_embedding_1d(dim, t): (t * 1000) times the frequencies,
  // cos | sin, in f32 as torch does it.
  static std::vector<float> time_embedding(float t, int dim);

 private:
  MetalMagi2Dit() = default;
  struct Layer;
  struct Scratch;
  bool load_(std::string* err);
  bool ensure_scratch_(int n, std::string* err);
  bool load_sage_(std::string* err);

  std::shared_ptr<WeightSet> _ws;
  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::vector<Layer> _layers;
  std::unique_ptr<Scratch> _s;
  std::vector<float> _bands;
  metal_compute::SharedBuffer _emb_w[3], _emb_b[3];   // video audio text
  metal_compute::SharedBuffer _out_w[2];              // video audio, folded
  struct Kernels;
  std::unique_ptr<Kernels> _k;
  std::unique_ptr<magi2::Gemms> _g;
  std::unique_ptr<MetalSageAttention> _sage;
  bool _nax = false, _sage_wanted = false, _nax_moe = false;
  bool _tg_moe = false;
  long _n_sage = 0, _n_sage_pv8 = 0, _n_moe_nax = 0;
  metal_compute::CommandStream _stream;
  GpuFlopsTally _flops;   // see gpu_flops()
  // The streamed experts' two slots: [slot][W_gate, W_up, W_down].
  metal_compute::SharedBuffer _slot[2][3];
  std::vector<std::pair<int, metal_compute::SharedBuffer>> _forced;
  bool _keep_routes = false;
  std::vector<std::pair<int, std::vector<int>>> _kept;
  std::uint64_t _expert_bytes = 0;
  std::function<bool(int, int)> _block_cb;
  int _moe_held = 0, _moe_streamed = 0;
  // The held experts' wired pool, and whether a layer may still be
  // admitted (false from the first refusal on).
  std::unique_ptr<WiredPool> _wire;
  bool _admit = true;
  // This rank's widths (the whole model's when not split).
  Share _sh;
  // A rank's reads of its experts: of each of the three tensors, the
  // share's run of every head's experts, into `dst` (a layer's three).
  bool read_experts_(int idx, metal_compute::SharedBuffer dst[3]) const;
  // A tensor of every MoE head's experts, `whole` bytes, as the share
  // holds it.
  std::uint64_t expert_share_(std::uint64_t whole) const noexcept;
  // ...the clip's channel and its exchanges: a partial and the others'
  // sum a lane, one lane a sample of the batch (kTpLanes at most), kept
  // at one address until forgotten (TpChannel::forget).
  static constexpr int kTpLanes = 4;
  std::shared_ptr<TpChannel> _tp_ch;
  std::unique_ptr<TpOverlap> _tp_pipe;
  std::uint64_t _tp_seq = 0;
  std::uint64_t _tp_exchanges = 0;
  bool _tp_lagged = false;
  metal_compute::SharedBuffer _tp_part[kTpLanes], _tp_recv[kTpLanes];
  bool tp_buffer_(metal_compute::SharedBuffer& b, std::size_t bytes);
};

}  // namespace genai
}  // namespace vpipe

#endif  // VPIPE_GENERATIVE_MODELS_MAGI2_METAL_MAGI2_DIT_H
