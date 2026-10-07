#ifndef VPIPE_GENERATIVE_MODELS_YUE2_METAL_YUE2_MODEL_H
#define VPIPE_GENERATIVE_MODELS_YUE2_METAL_YUE2_MODEL_H

// MetalYue2Model -- YuE2-3B (m-a-p) on metal-compute: the AR-NAR
// Mixture-of-Transformers that writes a song's score and semantic tokens,
// then its 64-wide VAE latents. The VAE itself is a separate checkpoint
// and a separate class (MetalOobleckDecoder).
//
// ONE CHECKPOINT, TWO WEIGHT SETS PER LAYER. Every decoder layer carries
// an AR path (self_attn, mlp, input_/post_attention_layernorm -- the
// Qwen3 names and shapes, a 1.7B-class dense stack) and a NAR path
// (nar_self_attn, nar_mlp, nar_input_/nar_pre_mlp_layernorm, same shapes).
// The reference's per-token mixing of the two is a TRAINING mechanism;
// inference runs each pass with one set:
//
//   AR   MetalQwenModel, backbone-only, owns decode and its paged KV. The
//        embedding table and lm_head are this class's: a phase only ever
//        picks from a window of the 184704-wide vocabulary (32769 rows
//        for the music), so the head GEMV covers that window alone --
//        a sixth of the full head's bytes per token.
//
//   NAR  flow matching over the latent rows only. The AR weights prefill
//        the chunk's tokens (prefix + codec + MUSIC_END) ONCE, causally;
//        each of the 64 velocity evaluations then runs the NAR weights
//        over [START, latents..., END], attending that frozen K/V plus its
//        own rows with NO mask. Positions continue the AR sequence.
//
// The arithmetic the reference does in bf16 OUTSIDE the network -- the
// timestep's sigmoid, its embedder, the midpoint ODE update -- is done
// on the host here with the same roundings, because it is a few hundred
// numbers per evaluation and getting it wrong is a quiet drift.

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/sage-attention.h"
#include "generative-models/shared/sol-attention.h"
#include "generative-models/yue2/yue2-protocol.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
class ComputeEncoder;
}

namespace vpipe::genai {

class AneFeedForward;
class I8GemmContext;
class MetalQwenModel;
class MetalSageAttention;
class MetalSolAttention;
class WeightSet;

class MetalYue2Model {
public:
  struct Config {
    int   n_layers   = 28;
    int   hidden     = 2048;
    int   n_heads    = 16;
    int   n_kv_heads = 8;
    int   head_dim   = 128;
    int   ffn        = 6144;
    int   vocab      = yue2::kVocab;
    float rope_theta = 1.0e6f;
    float rms_eps    = 1e-6f;
    int   latent_dim = yue2::kLatentDim;
    int   max_latent_frames = 24576;
    float timestep_shift    = 1.0f;
    int   context    = yue2::kContext;
    // How the AR stack is held. 8: its projections as affine w8 group-64,
    // built in memory from the bf16 checkpoint as each layer loads
    // (MetalQwenModel::Config::load_quant_bits) -- ~53% of their bytes,
    // and the song phase's decode reads every one per token. 0: bf16, as
    // published. The NAR stack, the embedding and the head stay bf16.
    // Not read from config.json: the caller's choice.
    int   ar_quant_bits = 0;
  };

  // config.json with model_type "yue2" (and latent_type "vae", the only
  // one the release supports). False with a reason otherwise.
  static bool config_from_dir(const std::string& dir, Config* out,
                              std::string* err = nullptr);
  // Cheap probe for the stage: is `dir` a YuE2 LM checkpoint?
  static bool is_yue2_dir(const std::string& dir);

  // The returned model KEEPS the set: both passes read their tensors
  // from it, and the AR backbone is a MetalQwenModel over the same set.
  static std::unique_ptr<MetalYue2Model> load(
      std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
      const Config& cfg, std::string* err = nullptr);
  // Private, unregistered set -- tools and tests without a session.
  static std::unique_ptr<MetalYue2Model> load(
      const std::string& dir, metal_compute::MetalCompute* mc,
      std::string* err = nullptr);
  ~MetalYue2Model();

  const Config& config() const { return _cfg; }
  MetalQwenModel* backbone() { return _lm.get(); }
  // Whether the AR stack was built w8 in memory (ar_quant_bits 8 on a
  // checkpoint whose AR stack the backbone can build).
  bool ar_quantized() const;
  // What a model loaded from `dir` will hold in weights, from the
  // checkpoint alone -- for a stage that books before anything loads.
  // `ar_quant_bits` as Config::ar_quant_bits; 0 when `dir` is unreadable.
  static std::size_t weight_bytes(const std::string& dir, int ar_quant_bits);

  // ---- AR ---------------------------------------------------------------
  struct DecodeResult {
    std::vector<std::int32_t> ids;   // kept ids, the end id excluded
    bool   truncated = false;        // max_tokens reached without the end
    bool   stopped   = false;        // the callback asked to stop
    double seconds   = 0.0;
    double prefill_seconds = 0.0;
  };
  // Called after every pick, the end id included. Return false to stop.
  using TokenFn = std::function<bool(std::int32_t id, int step)>;

  // One phase of decode from `prefix` on a fresh context. `negative`
  // (with `cfg` != 1) runs the guided second branch on its own context;
  // `legacy` is cot=off's bf16 penalty arithmetic. Ids come back as
  // vocabulary ids (codec ids still carry kCodecOffset).
  bool decode(const std::vector<std::int32_t>& prefix, yue2::Phase phase,
              const yue2::Sampling& s, std::uint64_t seed,
              const std::vector<std::int32_t>* negative, float cfg,
              bool legacy, const TokenFn& on_token, DecodeResult* out,
              std::string* err = nullptr);

  // Test hook: decode TEACHER-FORCED along `forced`. Each step's pick is
  // recorded in `picks`, but forced[k] is what is fed back and what the
  // penalty window sees -- so a near-tie broken the other way costs one
  // position rather than the rest of the run.
  bool decode_forced(const std::vector<std::int32_t>& prefix,
                     yue2::Phase phase, const yue2::Sampling& s,
                     const std::vector<std::int32_t>& forced,
                     std::vector<std::int32_t>* picks,
                     std::string* err = nullptr);

  // Test hook: the logits over the WHOLE vocabulary at the last position
  // of `prefix` (a fresh context, prefill only).
  bool prefill_logits(const std::vector<std::int32_t>& prefix,
                      std::vector<float>* logits, std::string* err = nullptr);

  // ---- NAR --------------------------------------------------------------
  // Called once per velocity evaluation with (done, total) over the
  // whole song. Return false to stop.
  using ProgressFn = std::function<bool(int done, int total)>;

  // The song's latents, [frames, latent_dim] f32 row-major, from the AR
  // prefix (as decoded, with or without the score) and the codec CODES
  // (kCodecOffset removed). Solved per original context chunk with
  // `steps` midpoint steps from torch-identical noise for `seed`, unless
  // `noise` ([frames, latent_dim]) is given.
  bool synthesize(const std::vector<std::int32_t>& prefix,
                  const std::vector<std::int32_t>& codec,
                  std::uint64_t seed, int steps,
                  std::vector<float>* latents, const ProgressFn& progress,
                  std::string* err = nullptr,
                  const std::vector<float>* noise = nullptr);

  // Test hook: ONE velocity evaluation for a chunk whose AR tokens are
  // `ar_tokens` (prefix + codec + MUSIC_END), at ODE state `state`
  // ([T, latent_dim]; rounded to bf16 as the reference does) and raw
  // timestep `raw_t` (the logit of t, before the sigmoid shift).
  bool velocity(const std::vector<std::int32_t>& ar_tokens,
                const std::vector<float>& state, double raw_t,
                std::vector<float>* v, std::string* err = nullptr);

  // ---- acceleration tiers of the flow matching ------------------------
  //
  // The NAR pass is ~70% of a song (MEASURED on the M4 Pro: 320 of 470 s
  // for 3:34), and every one of its 64 evaluations is a 28-layer DiT
  // forward over the latent rows -- the shape the shared tiers were built
  // for. All three are OFF unless asked for; Sage and Sol are LOSSY.
  //
  //   ane_ffn   the NAR SwiGLU's tail rows on the ANE, the head rows on
  //             the GPU, concurrently (shared/ane-ffn.h). For the M4
  //             series, which has no matrix cores; never judged on a
  //             fanless box. Exact up to the ANE's fp16.
  //   sage      the QK product of the attention in int8 (M5 matrix cores
  //             only; elsewhere it says so once and stays dense).
  //   sol       Sol-Attn routing over the [prefix ; codec ; latents] keys,
  //             with the TEXT PREFIX held exact as a sink. Both arms.
  //   i8_gemm   the big GEMMs as dynamic-int8 matmul2d (M5 only; LOSSY,
  //             ~1e-2 per GEMM). Declines below its row and depth gates.
  struct Accel {
    bool  i8_gemm    = false;
    bool  ane_ffn    = false;
    float ane_rows   = 0.0f;      // the ANE's share; <= 0 balances itself
    int   ane_layers = 0;         // cap on split layers; 0 = all 28
    sage::Config sage;
    sol::Config  sol;
  };
  // Loads what the tiers need (Sage and Sol kernels now, the ANE module at
  // the first forward, which is what sizes it). False only when a tier
  // that was asked for failed in a way that must not pass for a run with
  // it -- Sage's kernels not loading on a box that has the instruction.
  bool set_accel(const Accel& a, std::string* err = nullptr);
  const Accel& accel() const { return _accel; }
  bool sage_armed() const { return _sage != nullptr; }
  bool i8_armed() const { return _i8 != nullptr; }
  bool sol_armed() const { return _sol != nullptr; }
  bool ane_armed() const { return _ane != nullptr; }

  // What the CoreML module of the ANE tier holds, for the plan: fixed,
  // whatever the song's length.
  static std::size_t ane_runtime_bytes();

  // What the last synthesize() actually did, for the log line and for a
  // test that has to show a tier ENGAGED before it shows it accurate.
  struct NarStats {
    int    evals          = 0;
    double ms             = 0.0;      // wall time of the evaluations
    long long ane_rows    = 0;        // feed-forward rows the ANE ran
    long long ffn_rows    = 0;        // ... out of these
    int    sage_layers    = 0;        // attention calls that took int8 QK
    long long i8_gemms    = 0;        // GEMMs the int8 route took
    int    sol_layers     = 0;        // ... that Sol routed
    long long sol_exact   = 0;        // routed (query, key) blocks kept
    long long sol_total   = 0;
    double sol_kept() const
    {
      return sol_total > 0 ? (double)sol_exact / (double)sol_total : 0.0;
    }
  };
  const NarStats& nar_stats() const { return _stats; }
  void reset_nar_stats() { _stats = NarStats{}; }

  // What this model holds: its own tensors plus the backbone's.
  std::size_t resident_bytes() const;

  // What a song holds BESIDE the weights, for a stage that has to plan
  // before anything loads. `prefix` is the longest prompt prefix the song
  // phase sees (style, lyrics and score), `frames` its latent rows -- one
  // per song token -- and `guided` whether the song phase decodes a
  // negative context beside it (CFG).
  //
  //   held       allocated by a song and never given back: the AR pool,
  //              whose Metal pages grow by DOUBLING and keep their high-
  //              water capacity, and the NAR row scratch (grow-only).
  //              Alive for the rest of the run, whatever phase it is.
  //   transient  allocated and freed inside one song: the flow matching's
  //              head-major copy of the chunk's K/V, or the pool's old
  //              capacity while a doubling copies it, whichever is larger.
  struct RunBytes {
    std::size_t held = 0, transient = 0;
  };
  static RunBytes run_bytes(const Config& c, int prefix, int frames,
                            bool guided);
  // The held term as it stands NOW: the AR pool's capacity plus the NAR
  // row scratch. For checking run_bytes() against a real song.
  std::size_t held_scratch_bytes() const;

  // Whether the M5 matrix-core GEMM / attention routes were selected.
  bool uses_mma() const { return _use_mma; }
  bool uses_attn_nax() const { return _use_nax; }

private:
  MetalYue2Model() = default;

  struct NarLayer {
    metal_compute::SharedBuffer in_ln, mlp_ln;
    metal_compute::SharedBuffer q, k, v, o, q_norm, k_norm;
    metal_compute::SharedBuffer gate, up, down;
  };

  // One original context chunk, set up for its ODE: the AR prefix's K/V
  // per layer (token-major [L, kv_heads * head_dim], with the NAR rows'
  // own K/V written behind it by every evaluation) and the RoPE tables
  // of the NAR positions.
  // K/V are HEAD-MAJOR [kv_heads, ar_len + nar_rows, head_dim] -- what
  // Sol reads, and what steel and Sage take through their strides.
  struct Chunk {
    int ar_len = 0;
    int nar_rows = 0;                 // latent frames + START + END
    int frames = 0;
    int sink = 0;                     // text-prefix tokens Sol keeps exact
    std::vector<metal_compute::SharedBuffer> k, v;
    metal_compute::SharedBuffer cos, sin, params;
  };

  bool init_(const std::shared_ptr<WeightSet>& ws,
             metal_compute::MetalCompute* mc, const Config& cfg,
             std::string* err);

  // Embedding rows for `ids`, bf16 [n, hidden].
  metal_compute::SharedBuffer embed_(const std::vector<std::int32_t>& ids,
                                     std::size_t from, std::size_t n);
  // The head over a phase window, from a final-normed hidden, into
  // _logits; then widened to `out` (window rows, f32).
  void encode_head_(metal_compute::ComputeEncoder& enc,
                    const metal_compute::SharedBuffer& h,
                    const yue2::Window& w);
  void read_logits_(const yue2::Window& w, std::vector<float>* out) const;

  bool run_decode_(const std::vector<std::int32_t>& prefix,
                   yue2::Phase phase, const yue2::Sampling& s,
                   std::uint64_t seed,
                   const std::vector<std::int32_t>* negative, float cfg,
                   bool legacy, const TokenFn& on_token,
                   const std::vector<std::int32_t>* forced,
                   std::vector<std::int32_t>* picks, DecodeResult* out,
                   std::string* err);
  bool setup_chunk_(const std::vector<std::int32_t>& ar_tokens, int frames,
                    Chunk* c, std::string* err);
  bool eval_(Chunk& c, const std::vector<float>& state, double raw_t,
             std::vector<float>* v, std::string* err);
  bool ensure_scratch_(int rows);
  // Bytes ensure_scratch_ allocates per row; run_bytes() books them.
  static std::size_t nar_row_bytes_(const Config& c);
  void time_embedding_(double raw_t, std::vector<std::uint16_t>* out) const;
  bool ane_setup_(int seq);
  bool ane_eligible_(int L) const;
  void ane_stage_(int L);

  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t xe,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer* bias,
             const metal_compute::SharedBuffer& y, std::size_t ye,
             int M, int N, int K);
  const metal_compute::ComputeFunction* attn_fn_(int q_rows, int k_rows,
                                                 bool i8);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::shared_ptr<WeightSet> _ws;
  std::unique_ptr<MetalQwenModel> _lm;

  // AR, this class's half: embedding table, untied head, logit scratch.
  metal_compute::SharedBuffer _embed, _head, _logits, _d_emb;

  // NAR weights.
  std::vector<NarLayer> _nar;
  metal_compute::SharedBuffer _final_ln, _vae2llm_w, _vae2llm_b;
  metal_compute::SharedBuffer _llm2vae_w, _llm2vae_b, _pos;
  // The timestep embedder runs on the host; its weights widened once.
  std::vector<float> _te0_w, _te0_b, _te2_w, _te2_b;

  // NAR scratch, sized for the largest chunk seen. _n_q / _n_k / _n_vv are
  // the token-major projections; _n_qh and _n_aoh the head-major query
  // and attention output.
  int _scratch_rows = 0;
  metal_compute::SharedBuffer _n_in, _n_x, _n_h, _n_q, _n_k, _n_vv, _n_qh,
      _n_aoh, _n_ao, _n_o, _n_g, _n_u, _n_v, _n_t;

  // The tiers (set_accel).
  Accel _accel;
  std::unique_ptr<AneFeedForward> _ane;
  int _ane_tried_seq = 0;             // the seq the last create() was given
  std::unique_ptr<MetalSageAttention> _sage;
  std::unique_ptr<I8GemmContext> _i8;
  std::unique_ptr<MetalSolAttention> _sol;
  bool _sol_warned = false;
  NarStats _stats;

  // Kernels.
  metal_compute::ComputeLibrary _lib_gemm, _lib_mma, _lib_elt, _lib_rms,
      _lib_rope, _lib_attn, _lib_attn_nax;
  metal_compute::ComputeFunction _fn_gemv, _fn_gemm, _fn_gemm_bm64,
      _fn_gemm_bm64bn64, _fn_mma, _fn_mma_deep, _fn_rms, _fn_rms_heads,
      _fn_tr_rope, _fn_add, _fn_add4, _fn_bias_rows, _fn_swiglu,
      _fn_swiglu4, _fn_transpose, _fn_kv_gather;
  std::unordered_map<unsigned, metal_compute::ComputeFunction> _attn_fns;
  bool _use_mma = false;
  bool _use_nax = false;
};

}  // namespace vpipe::genai

#endif
