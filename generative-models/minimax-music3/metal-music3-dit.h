#ifndef VPIPE_GENERATIVE_MODELS_MINIMAX_MUSIC3_METAL_MUSIC3_DIT_H
#define VPIPE_GENERATIVE_MODELS_MINIMAX_MUSIC3_METAL_MUSIC3_DIT_H

// MetalMusic3Dit -- MiniMax-Music3's synthesis half on metal-compute: the
// condition encoder that turns the language model's per-frame hidden
// states into a 2048-wide condition on the latent timeline, and the 2.4B
// flow-matching transformer that solves the song's 128-wide latents from
// it, window by window (music3-protocol.h has the windows and the blend).
//
// The transformer is Stable Audio's continuous transformer as the
// checkpoint carries it: the input is [latent ; zeros ; condition]
// (2304 channels) through a residual 1x1 conv and a projection to 2048,
// the time step's embedding is PREPENDED as one extra token and dropped
// after the 36 blocks, and a block is pre-LayerNorm attention (32 heads of
// 64, rotary on the first 32 channels of each head, positions 0.. with the
// time token at 0, no mask) and a GLU feed-forward ([value | gate],
// value * silu(gate), 8192 wide, both linears biased). Classifier-free
// guidance runs the conditional and the all-zero-condition rows as ONE
// batch of two.
//
// f16 activations, not bf16: the residual stream peaks near 10^4 -- on
// the TIME TOKEN, whose size follows the time step, not the song (the
// song's rows stay under ~1.5e3) -- which leaves f16 a 6x margin and buys
// 8x the mantissa. The weights are F32 on disk and narrow to f16 once.
//
// The Euler steps, the guidance mix and the overlap blend run on the host
// in f32, as the fp32 reference runs them.
//
// INT8 (Config::i8_tier; matrix cores only -- the int8 context declines
// elsewhere and the transformer stays f16): the 36 blocks' four
// projections run dynamic-int8 GEMMs. `accurate` keeps the f16 weights
// and quantizes them per call at groups of 64; `fast`, `token` and
// `token_plain` HOLD each projection in its int8 layout, converted once
// at load from whatever the checkpoint stores (F32, F16 or BF16), with no
// f16 copy kept -- the transformer then holds 2.4 GB where f16 holds 4.6.
// A GEMM under the int8 route's row floor (a song's short last window)
// dequantizes its weight into one f16 stand-in and runs the f16 route.
// The small projections around the stack and the condition encoder stay
// f16 at every tier.

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/dit-flops.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
class ComputeEncoder;
}

namespace vpipe::genai {

class I8GemmContext;
class WeightSet;

class MetalMusic3Dit {
public:
  struct Config {
    int in_channels  = 128;
    int cond_dim     = 2048;
    int layers       = 36;
    int heads        = 32;
    int head_dim     = 64;
    int ff_inner     = 8192;
    int rotary_dim   = 32;
    int fourier_dim  = 256;
    // The condition encoder.
    int cond_hidden  = 4096;
    int cond_layers  = 8;
    float ln_eps     = 1e-5f;
    // The blocks' int8 GEMMs: -1 off, 0 accurate, 1 fast, 2 token (rotated),
    // 3 token_plain -- generate-image's i8_gemm_tier, by number. Not read
    // from config.json.
    int i8_tier = -1;
    int dim() const { return heads * head_dim; }
    int concat() const { return 2 * in_channels + cond_dim; }
  };

  // The tier a load with `i8_tier` will run on a box with (or without)
  // matrix cores, as the int8 context decides it: -1 where it declines,
  // and VPIPE_I8_GEMM / VPIPE_I8_TIER over the config, as the context
  // reads them. 1..3 HOLD the block projections as int8.
  static int effective_i8_tier(int i8_tier, bool matrix_cores);

  // transformer/config.json and condition_encoder/config.json under the
  // model root. False with a reason if either is not Music3's.
  static bool config_from_dir(const std::string& root, Config* out,
                              std::string* err = nullptr);

  // `dit` is the transformer's set, `cond` the condition encoder's. The
  // model KEEPS both.
  static std::unique_ptr<MetalMusic3Dit> load(
      std::shared_ptr<WeightSet> dit, std::shared_ptr<WeightSet> cond,
      metal_compute::MetalCompute* mc, const Config& cfg,
      std::string* err = nullptr);
  ~MetalMusic3Dit();

  const Config& config() const { return _cfg; }

  // What `root`'s transformer/ and condition_encoder/ will hold, from
  // their tensor headers: every tensor at f16 but the time embedder's and
  // the layer mix's, which live on the host at f32 -- and, at an
  // effective tier (effective_i8_tier) that holds the int8 layout, the
  // block projections as its codes and scales. 0 when unreadable.
  static std::size_t transformer_weight_bytes(const std::string& root,
                                              int effective_tier = -1);
  static std::size_t condition_weight_bytes(const std::string& root);
  // What it holds, the host copies included.
  std::size_t resident_bytes() const;
  // ... of which the condition encoder's (condition_encoder/).
  std::size_t condition_bytes() const
  {
    return _cond_w.byte_size() + _cond_b.byte_size() + _mix.size() * 4;
  }
  // The working set of one window's solve (both guidance rows), for the
  // plan: allocated at the first window and kept -- with the int8
  // context's scratch at an effective tier, and the f16 stand-in a held
  // layout dequantizes into under the int8 floor.
  static std::size_t window_scratch_bytes(const Config& c,
                                          int effective_tier = -1);
  std::size_t scratch_bytes() const;

  // ---- the pieces, for tests --------------------------------------------
  // The condition of a window: `hiddens` is [frames, 8 * 4096] f16 (the
  // AR stage's handoff), the result [latent_length(frames), 2048] f32.
  bool condition(const _Float16* hiddens, int frames,
                 std::vector<float>* cond, std::string* err = nullptr);
  // One evaluation of both guidance rows at flow time `t`: `x` is the
  // state [L, 128] time-major f32, `cond` [L, 2048] f32 (the unconditional
  // row conditions on zeros). `vc` / `vu` come back [L, 128] f32.
  bool velocity(const std::vector<float>& x, float t,
                const std::vector<float>& cond, std::vector<float>* vc,
                std::vector<float>* vu, std::string* err = nullptr);

  // ---- a song -------------------------------------------------------------
  // Called once per transformer evaluation with (done, total) over the
  // song. Return false to stop.
  using ProgressFn = std::function<bool(int done, int total)>;

  // The song's stitched latents, CHANNEL-major [128, stitched_latents(
  // frames)], from `frames` rows of hidden states ([frames, 32768] f16):
  // each window solved by `steps` Euler steps under guidance `cfg`, from
  // noise torch.randn would draw for `seed` (one float32 draw per window,
  // in order, from one generator), unless `noise` gives each window's
  // [128, L] channel-major. Stopped early, it returns false with `err`
  // empty.
  bool synthesize(const _Float16* hiddens, int frames, std::uint64_t seed,
                  int steps, float cfg, std::vector<float>* latents,
                  const ProgressFn& progress, std::string* err = nullptr,
                  const std::vector<std::vector<float>>* noise = nullptr);

  // What the last synthesize() did.
  struct Stats {
    int    evals = 0;
    int    windows = 0;
    double ms = 0.0;           // wall time of the evaluations
  };
  const Stats& stats() const { return _stats; }

  // Whether the M5 matrix-core GEMM / attention routes were selected.
  bool uses_mma() const { return _use_mma; }
  bool uses_attn_nax() const { return _use_nax; }
  // The int8 tier the blocks run (-1 none), its name for a log line, and
  // the GEMMs it took since load: those on the int8 route, and those a
  // held layout ran dequantized under the floor.
  int i8_tier() const { return _i8_tier; }
  static const char* i8_tier_name(int tier);
  long long i8_gemms() const { return _i8_gemms; }
  long long i8_dequant_gemms() const { return _i8_deq_gemms; }

  // The GPU arithmetic credited to the achieved-FLOP/s meter.
  const GpuFlopsTally& gpu_flops() const noexcept { return _flops; }

private:
  MetalMusic3Dit() = default;

  // A block projection [n, k]: f16, or (a tier that holds it) its int8
  // layout -- codes [n, k] and scales at group gc -- and no f16.
  struct Proj {
    metal_compute::SharedBuffer w, codes, scales;
    int n = 0, k = 0, gc = 0;
    bool empty() const { return w.empty() && codes.empty(); }
    std::size_t bytes() const
    {
      return w.byte_size() + codes.byte_size() + scales.byte_size();
    }
  };
  struct Block {
    metal_compute::SharedBuffer ln1_w, ln1_b, ln2_w, ln2_b;
    Proj qkv, out;                                 // [6144, 2048], [2048^2]
    Proj ff_in;                                    // interleaved [gate,value]
    metal_compute::SharedBuffer ff_in_b;
    Proj ff_out;
    metal_compute::SharedBuffer ff_out_b;
  };

  bool init_(const std::shared_ptr<WeightSet>& dit,
             const std::shared_ptr<WeightSet>& cond,
             metal_compute::MetalCompute* mc, const Config& cfg,
             std::string* err);
  bool ensure_scratch_(int L);
  bool rope_table_(int L);
  // The time token for flow time t, f32 [dim] (host).
  void time_embedding_(float t, std::vector<float>* out) const;
  // A window's scratch, rope tables and condition columns (row 0 the
  // window's condition, row 1 zeros): once a window, not once a step.
  bool begin_window_(const std::vector<float>& cond, int L,
                     std::string* err);
  bool forward_(const std::vector<float>& x, float t, int L,
                std::vector<float>* vc, std::vector<float>* vu,
                std::string* err);
  // The f16 GEMM y = x w^T (+ bias): matmul2d or steel.
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t xe,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer* bias,
             const metal_compute::SharedBuffer& y, std::size_t ye,
             int M, int N, int K);
  // A block projection: its tier's int8 route, else the f16 one (a held
  // layout dequantized first). False when nothing could be encoded.
  bool gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t xe,
             const Proj& w, const metal_compute::SharedBuffer* bias,
             const metal_compute::SharedBuffer& y, std::size_t ye, int M);
  void dense_gemm_(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& x, std::size_t xe,
                   const metal_compute::SharedBuffer& w,
                   const metal_compute::SharedBuffer* bias,
                   const metal_compute::SharedBuffer& y, std::size_t ye,
                   int M, int N, int K);
  // The kept stream (VPIPE_MUSIC3_FRESH_STREAMS=1: a new one each call).
  metal_compute::CommandStream& stream_();

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::shared_ptr<WeightSet> _ws, _cond_ws;

  // Transformer weights (f16; the rest of the time embedder on the host).
  std::vector<Block> _blocks;
  metal_compute::SharedBuffer _pre, _proj_in, _proj_out, _post;
  std::vector<float> _fourier, _te1_w, _te1_b, _te2_w, _te2_b;
  // Condition encoder: the conv as a GEMM [2048, 3 * 4096] in (tap, cin)
  // order, its bias; the layer mix on the host.
  metal_compute::SharedBuffer _cond_w, _cond_b;
  std::vector<float> _mix;           // softmax(layer_weight_logits) * scale

  // Scratch for one window of L latents, both guidance rows: M = 2 (L + 1)
  // transformer rows.
  int _scratch_L = 0;
  metal_compute::SharedBuffer _in, _pre_o, _x, _n, _qkv, _att, _tmp, _ff,
      _glu, _po, _pq, _rcos, _rsin, _ap;
  metal_compute::SharedBuffer _c_in, _c_out;   // condition encoder
  int _cond_frames = 0;
  std::vector<float> _temb_t;        // the cached time embedding's t
  std::vector<std::vector<float>> _temb;

  Stats _stats;
  GpuFlopsTally _flops;

  metal_compute::ComputeLibrary _lib_gemm, _lib_mma, _lib_elt, _lib_rope,
      _lib_attn;
  metal_compute::ComputeFunction _fn_gemm, _fn_gemm_bm64, _fn_gemm_bm64bn64,
      _fn_mma, _fn_mma_deep, _fn_bias_rows, _fn_ln, _fn_add, _fn_glu,
      _fn_rope, _fn_attn;
  int _attn_L = 0;
  bool _use_mma = false;
  bool _use_nax = false;

  // The int8 GEMMs (null when off or declined), the tier they run, and
  // the f16 stand-in a held layout dequantizes into under the floor.
  std::unique_ptr<I8GemmContext> _i8;
  int _i8_tier = -1;
  metal_compute::SharedBuffer _wdeq;
  long long _i8_gemms = 0, _i8_deq_gemms = 0;
  metal_compute::CommandStream _stream;
};

}  // namespace vpipe::genai

#endif
