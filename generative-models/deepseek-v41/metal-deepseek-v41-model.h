#ifndef VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_METAL_DEEPSEEK_V41_MODEL_H
#define VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_METAL_DEEPSEEK_V41_MODEL_H

// MetalDeepseekV41Model -- DeepSeek-V4.1-Flash (model_type deepseek_v41)
// on metal-compute. The reference is the release's inference/model.py;
// tools/dump_deepseek_v41_golden.py runs it.
//
// THE BLOCK. The residual is hc = 4 copies of the hidden state
// (hyper-connections). Each sublayer reads a learned mix of them (the
// PREVIOUS sublayer's `pre`), and writes its output back as post[k] * out
// + sum_j comb[j][k] * copy j, the three coefficient sets derived from the
// stream itself and comb made doubly stochastic by Sinkhorn. Two layers
// add an ENGRAM lookup first: n-gram hashed rows of a ~98 GB table each,
// read BY ROW from the checkpoint (never resident), projected to a key a
// copy and a shared value and gated per copy.
//
// ATTENTION is latent and single-headed on the key side: one 512-wide
// latent per position is both key and value for all 64 query heads (the
// last 64 dims rotated), the query low-rank (wq_a, norm, wq_b), the output
// low-rank and grouped (wo_a per group of 8 heads, then wo_b), a sink per
// head in the softmax. A query attends the last `window` positions'
// latents of its own layer, plus -- in a compressed layer -- latents of
// `ratio` positions pooled into one, made by the layer's kv source, the
// positions picked by its index source's top-k (see deepseek-v41-config.h
// for which layer does what). Rotary: window-only layers at rope_theta,
// compressed layers at compress_rope_theta with YaRN.
//
// THE MoE: 384 routed experts (6 a token) + one shared, sqrt(softplus)
// scores, a selection bias, the training clamps on the SwiGLU. The routed
// experts are E2M1 (FP4) with an E8M0 scale per 32 and are run as stored;
// they always come through a MoeExpertCache (the released 289 GB of them
// fit no box), which a small model fills whole. The checkpoint's FP8
// projections (E4M3, an E8M0 scale per 32x32 block) are kept as codes.
// oMLX's packs below 4 bits (oQ2e, oQ2.5e, oQ3e) store the backbone's
// experts and FP8 projections as MLX's affine codes instead (2-8 bits, a
// bf16 scale and bias per 64, the width per module and layer): run as
// stored too, by kernels of their own (dsv41_*_q<bits>), the cache's slots
// as wide as the widest layer's expert.
//
// WHAT IS NOT HERE YET: the vision tower, the DSpark draft head, the CED
// prefill shortcut (decoder layers on the last rows only), and the
// deployment's own activation / KV quantizations, which the golden runs
// without (it is held to the stored weights' arithmetic).

#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/deepseek-v41/deepseek-v41-config.h"
#include "generative-models/deepseek-v41/deepseek-v41-engram.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vpipe::metal_compute {
class MetalCompute;
class ComputeEncoder;
}

namespace vpipe::genai {

class WeightSet;
class MoeExpertCache;
class SafetensorsRanges;

class MetalDeepseekV41Model {
public:
  struct Options {
    bool use_bf16 = true;
    // Bytes the routed experts' cache may take. 0: every expert resident
    // (the cache holds the whole model, filled as experts are first
    // picked).
    std::size_t expert_cache_bytes = 0;
    // Rows a forward runs at once; a longer prefill runs in pieces, each
    // piece's index scores made a chunk of rows at a time (under 1 GB).
    int max_rows = 8192;
    // The most rows a chunk of index scores takes (0: as the budget
    // allows; else a multiple of 8) -- the tests' way to reach many chunks
    // on a small model.
    int score_rows = 0;
    // Load the DSpark draft head (the checkpoint's mtp.* stages) for
    // speculative decoding: ~8 GB more, held resident.
    bool dspark = false;
    // A prefill piece longer than the window runs the decoder half on its
    // last window only (the README's causal encoder-decoder, "SWA bounded
    // replay"; see forward_piece_). VPIPE_DSV41_CED=0|1 overrides.
    bool ced = true;
  };

  // The FP8 projections are held as their codes unless
  // VPIPE_DSV41_DENSE_PROJ=1 (the A/B: widened to the compute dtype).
  static bool keeps_fp8_projections();
  static std::unique_ptr<MetalDeepseekV41Model> load(
      const std::shared_ptr<WeightSet>& ws, metal_compute::MetalCompute* mc,
      const Options& opt, std::string* err = nullptr);
  ~MetalDeepseekV41Model();

  const DeepseekV41Config& config() const { return _cfg; }
  // Whether the checkpoint is an MLX pack (oMLX's oQ layout) rather than
  // DeepSeek's release.
  bool mlx_layout() const { return _mlx; }
  // A tensor's name in a layout, from the release's spelling.
  static std::string tensor_name(const std::string& release, bool mlx);
  const DeepseekV41Engram& engram() const { return _engram; }

  using CtxId = std::uint64_t;
  // The positions a context holds (0 for one never used).
  int  seq_len(CtxId ctx) const;
  bool branch(CtxId parent, CtxId child);
  // Back to `pos` positions, no more than 8 behind the furthest the
  // context has reached: a speculative verify's rows past what it
  // accepted, or a prompt's last rows. False where it cannot.
  bool truncate(CtxId ctx, int pos);
  void release(CtxId ctx);

  // Rows that replace the embeddings of some of a forward's positions --
  // an image's span (MetalDeepseekV41Vision): `positions` index the ids
  // given, `rows` [positions.size(), dim] f32. Those positions take no
  // part in an n-gram (the engram gate is shut there and look-back stops
  // at them) and route with the image bias, as the reference's image
  // spans do.
  struct Media {
    std::vector<int> positions;
    const float* rows = nullptr;
  };

  // Run `ids` at the context's next positions (any length: pieces of
  // max_rows). `logits`: the last row's [V] f32, or every row's [n, V]
  // with all_rows. False with `err` set on failure (the context is then
  // left as it was before the call only up to the piece that failed).
  bool forward(CtxId ctx, std::span<const std::int32_t> ids,
               std::vector<float>* logits, bool all_rows,
               std::string* err = nullptr, const Media* media = nullptr);

  // What it holds: the resident weights, the contexts' state, the expert
  // cache.
  std::size_t resident_bytes() const;
  std::size_t kv_bytes() const;
  // What a forward of `rows` rows at a context of `ctx` positions holds
  // beside the weights: its scratch, one context's state, the rotary
  // tables and the last row's logits. run_held() is what is held now.
  // `spark`: with the DSpark draft head loaded.
  static std::size_t run_bytes(const DeepseekV41Config& c, int rows,
                               int ctx, bool spark = false);
  std::size_t run_held() const;
  // One context's state at `ctx` positions (the part of run_bytes each
  // further context adds).
  static std::size_t context_bytes(const DeepseekV41Config& c, int ctx);
  const MoeExpertCache* expert_cache() const { return _cache.get(); }
  MoeExpertCache* expert_cache_mut() { return _cache.get(); }
  // Read every routed expert into the cache now, layer by layer, until it
  // is full (benches: a steady state without the warm-up); how many.
  int preload_experts(std::string* err = nullptr);

  // ---- DSpark, the draft head --------------------------------------------
  // Loaded (Options::dspark), every forward commits its rows' main hidden
  // -- the target layers' inputs -- to each stage's window; a draft then
  // runs the block [token, noise...] (dspark_block rows, at the positions
  // after the context's) through the stages and the Markov-biased head.
  bool has_dspark() const { return !_stages.empty(); }
  // A draft from the context's last position: `token` is the next token,
  // decided and not yet run. `drafts` gets the block's dspark_block
  // tokens after it (greedy), `conf` the confidence of each block row (the
  // reference's confidence head), `logits` (optional) the block's [B, V]
  // logits, the Markov bias added.
  // `given` (a chained draft): the tokens drafted so far, rows 1.. of a
  // block of real tokens; the next draft (and its confidence, logits) is
  // the last row's alone.
  bool dspark_draft(CtxId ctx, std::int32_t token,
                    std::vector<std::int32_t>* drafts,
                    std::vector<float>* conf,
                    std::vector<float>* logits = nullptr,
                    std::string* err = nullptr,
                    const std::vector<std::int32_t>* given = nullptr);
  // The bytes the draft head holds when loaded, from the checkpoint.
  static std::size_t dspark_bytes(const std::string& model_dir);
  // Whether a running command buffer and this host see each other's
  // stores (what a step's per-layer handshake stands on; an M5 does, an M4
  // Pro does not reliably), as the loads a waiting GPU makes in a quarter
  // second: 0 when they do not. Probed once a process.
  static std::uint32_t handshake_works(metal_compute::MetalCompute* mc);
  // Whether this model's decode and verify steps take the handshake.
  bool handshake() const { return _hs_ok; }
  // Debug: the next handshake forward abandons layer `layer`'s turn (the
  // GPU then runs on stale slots, and the forward is redone with commits).
  void hs_drop_next(int layer) { _hs_drop = layer; }
  // Forwards redone after a turn the GPU gave up on.
  int hs_redos() const { return _hs_redos; }
  // Indexer passes that scored only a candidate source's kept blocks, and
  // top-k selections taken in two phases.
  std::uint64_t index_kept_passes() const { return _kept_passes; }
  std::uint64_t topk_two_passes() const { return _two_passes; }
  // Debug, VPIPE_DSV41_AHEAD_STUDY=K: how well a decode step's picks, and
  // the ones that held it up, are named ahead of time -- by the router of
  // layer L run on the input of layer L - k (k = 1..K), and by the step
  // before's own scores at L; a report of the run so far.
  std::string ahead_study() const;
  // The blocks (backbone layers and draft stages) whose bf16 wo_a (an MLX
  // pack's) was put back into the release's FP8 codes.
  int wo_a_codes() const { return _wo_a_codes; }
  // Layer-steps whose routed experts ran as grouped GEMMs (a prefill of
  // at least VPIPE_DSV41_MOE_GROUPED_MIN pairs an expert; 0 = never).
  std::uint64_t grouped_moe_steps() const { return _grouped_steps; }
  // the grouped GEMMs dispatched on the matrix cores (deepseek_v41_nax)
  long nax_moe_gemms() const { return _nax_gemms; }
  long nax_attentions() const { return _nax_attn; }
  long nax_index_passes() const { return _nax_index; }
  // prefill pieces whose decoder ran on the last window (CED), and the
  // first decoder layer (-1: the plan is not a causal encoder-decoder)
  long ced_pieces() const { return _ced_pieces; }
  // prefill pieces that ran on engram rows read while the piece before ran
  long engram_read_ahead() const { return _engram_ahead; }
  // Index-score chunks past each pass's first (score_rows_).
  long score_chunks() const { return _score_chunks_n; }
  int ced_mid() const { return _ced_mid; }

  // Debug: the 4-copy residual after layer `layer` of the last forward's
  // rows, [n, 4H] f32 (empty unless capture_layer() was set first).
  void capture_layer(int layer) { _capture = layer; }
  const std::vector<float>& captured() const { return _captured; }
  // Debug: the next forward (one piece) takes these expert picks, [L][n*k]
  // in each row's order, weighed by its own scores -- an A/B against a
  // reference's picks with no near-tie draw between them.
  void force_routes(std::vector<std::vector<std::int32_t>> r)
  {
    _force = std::move(r);
  }
  // Debug: the next draft's stages take these picks, [stage][B*k].
  void force_draft_routes(std::vector<std::vector<std::int32_t>> r)
  {
    _force_draft = std::move(r);
  }
  // Debug: each layer's routed expert ids of the last forward, [L][n*k].
  const std::vector<std::vector<std::int32_t>>& routes() const
  {
    return _routes;
  }

private:
  MetalDeepseekV41Model() = default;
  bool init_(const std::shared_ptr<WeightSet>& ws, std::string* err);

  // A projection: rows in the compute dtype; or FP8 E4M3 codes [N, K]
  // with an E8M0 scale per 32 of a row; or (an oQ pack below 4 bits) MLX's
  // affine codes, `bits` a value in a row's bit stream, with a bf16 scale
  // `s` and bias `b` per `gs` of a row. Codes are kept as the checkpoint
  // stores them (a decode reads them; a prefill widens them per call).
  struct Proj {
    metal_compute::SharedBuffer w, s, b;
    int N = 0, K = 0;
    bool fp8 = false;
    int bits = 0, gs = 0;          // affine (bits > 0)
    bool affine() const { return bits > 0; }
    std::size_t bytes() const
    {
      return w.byte_size() + s.byte_size() + b.byte_size();
    }
  };
  // A layer's routed experts: the release's FP4 (bits 0), or affine codes
  // -- w1 and w3 at `b13` bits, w2 at `b2`, a scale and bias per `gs`.
  struct ExpertFmt {
    int b13 = 0, b2 = 0, gs = 0;
  };
  struct Layer {
    // attention
    metal_compute::SharedBuffer attn_norm, ffn_norm, q_norm, kv_norm, sink;
    Proj wq_a, wq_b, wkv, wo_a, wo_b, i_wq_b;
    metal_compute::SharedBuffer c_w, c_norm;          // compressor
    metal_compute::SharedBuffer i_wp, i_wk, i_knorm;  // indexer
    // hyper-connections: fn transposed [4H, 24], scale [3], base [24]
    metal_compute::SharedBuffer ha_fn, ha_scale, ha_base;
    metal_compute::SharedBuffer hf_fn, hf_scale, hf_base;
    // MoE
    metal_compute::SharedBuffer gate_w, gate_b, gate_bvl;
    Proj sh_w13, sh_w2;
    // engram
    Proj e_wkv;
    metal_compute::SharedBuffer e_qk;
    std::string e_rows, e_scales, e_biases; // the table's tensor names
    ExpertFmt ex;
  };

  // A DSpark stage: a block (window-only attention, its own MoE) with its
  // experts resident in the expert cache's slot layout -- part k of expert
  // e at e * part_bytes[k] -- so the same kernels run them.
  struct Stage {
    Layer ly;
    metal_compute::SharedBuffer ex[6];     // w1, its scales, w3, .., w2, ..
  };
  struct LayerState {
    metal_compute::SharedBuffer ring;       // [W, D]
    metal_compute::SharedBuffer comp;       // kv source: [cap, D]
    metal_compute::SharedBuffer pend;       // ratio > 1: [ratio, 2D] f32
    metal_compute::SharedBuffer keys;       // index keys: [cap, 128]
    // ratio > 1: the last kKeepRows + ratio - 1 compressor inputs written,
    // [that, 2D] f32 at slot position % that, for truncate()
    metal_compute::SharedBuffer keep;
    int cap = 0;                            // compressed rows held
  };
  struct Ctx {
    int pos = 0;
    std::vector<LayerState> L;
    std::vector<std::int32_t> hist;         // compressed token ids
    // the first position truncate() may take the context back to (the
    // compressors' kept inputs reach back that far)
    int back_lo = 0;
    // DSpark: each stage's window of main latents, slot % (W + slack)
    std::vector<metal_compute::SharedBuffer> spark_ring;
  };


  // A prefill piece's engram rows, read on a worker while the piece before
  // it runs (forward()): its hashes over `hist` (the compressed history to
  // the piece's end), then each slot's table rows.
  struct EngramPrep {
    int pos = -1, n = 0;
    std::vector<metal_compute::SharedBuffer> rows;   // [slot]
    bool ok = false;
    std::string err;
  };
  // `after_reads` runs once this piece's own engram rows are read (on the
  // reading worker, or here when there are none to read).
  bool forward_piece_(Ctx& c, std::span<const std::int32_t> ids,
                      std::vector<float>* logits, bool all_rows,
                      std::string* err, const std::vector<int>& mpos,
                      const float* mrows, EngramPrep* prep = nullptr,
                      const std::function<void()>* after_reads = nullptr);
  bool ensure_ctx_(Ctx& c, int end_pos, std::string* err);
  bool ensure_scratch_(int n, int ctx_end);
  static std::vector<std::size_t> scratch_sizes_(const DeepseekV41Config& c,
                                                 int n, int cols,
                                                 bool spark);
  bool ensure_rope_(int positions);
  bool engram_rows_(int slot, const std::vector<std::int64_t>& rows, int n,
                    metal_compute::SharedBuffer& dst, std::string* err);
  bool engram_read_(const std::vector<std::int32_t>& hist,
                    std::size_t hist0, int n,
                    std::vector<metal_compute::SharedBuffer>& dst,
                    std::string* err);

  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer& x, std::size_t xoff,
             const metal_compute::SharedBuffer& y, std::size_t yoff, int K,
             int N, int M, std::size_t woff = 0);
  // y[M, nrows] = x @ P[row0 .. row0 + nrows)^T (nrows < 0: to the end).
  void gemm_p_(metal_compute::ComputeEncoder& enc, const Proj& p,
               const metal_compute::SharedBuffer& x, std::size_t xoff,
               const metal_compute::SharedBuffer& y, std::size_t yoff, int M,
               int row0 = 0, int nrows = -1);
  // Block.hc_mixes / hc_pre (then RMSNorm by `w`, none: the collapse
  // alone) / hc_post over a residual X [n, HC, H].
  void hc_mix_(metal_compute::ComputeEncoder& enc,
               const metal_compute::SharedBuffer& X, int n,
               const metal_compute::SharedBuffer& fn,
               const metal_compute::SharedBuffer& sc,
               const metal_compute::SharedBuffer& bs,
               metal_compute::SharedBuffer* mix);
  // hc_mix_'s two halves: the copies' slice sums, then the mix itself
  void hc_mix_part_(metal_compute::ComputeEncoder& enc,
                    const metal_compute::SharedBuffer& X, int n,
                    const metal_compute::SharedBuffer& fn);
  void hc_mix_finish_(metal_compute::ComputeEncoder& enc, int n,
                      const metal_compute::SharedBuffer& sc,
                      const metal_compute::SharedBuffer& bs,
                      metal_compute::SharedBuffer* mix);
  void force_picks_(const std::vector<std::int32_t>& f, int rows, int E,
                    int K);
  void hc_pre_(metal_compute::ComputeEncoder& enc,
               const metal_compute::SharedBuffer& X, int n,
               const metal_compute::SharedBuffer& pre,
               const metal_compute::SharedBuffer* w,
               const metal_compute::SharedBuffer& y);
  // the collapse, after the branch's post when post_mix is given (`a` the
  // branch, or with K > 0 the K routed parts and `sh`, summed)
  void hc_coll_(metal_compute::ComputeEncoder& enc,
                const metal_compute::SharedBuffer& X, int n,
                const metal_compute::SharedBuffer& pre,
                const metal_compute::SharedBuffer* w,
                const metal_compute::SharedBuffer& y,
                const metal_compute::SharedBuffer* a,
                const metal_compute::SharedBuffer* post_mix,
                const metal_compute::SharedBuffer* sh, int K);
  void hc_post_(metal_compute::ComputeEncoder& enc,
                const metal_compute::SharedBuffer& a,
                const metal_compute::SharedBuffer& X, int n,
                metal_compute::SharedBuffer* mix);
  void gemv_(metal_compute::ComputeEncoder& enc, int kind,
             const metal_compute::SharedBuffer& w, std::size_t woff,
             const metal_compute::SharedBuffer* s, std::size_t soff,
             const metal_compute::SharedBuffer& x, std::size_t xoff,
             const metal_compute::SharedBuffer& y, std::size_t yoff, int K,
             int N, int M, int batches = 1, std::int64_t xb = 0,
             std::int64_t wb = 0, std::int64_t yb = 0);
  void gemv_q_(metal_compute::ComputeEncoder& enc, const Proj& p, int row0,
               int N, const metal_compute::SharedBuffer& x, std::size_t xoff,
               const metal_compute::SharedBuffer& y, std::size_t yoff,
               int M);
  void gemm_groups_(metal_compute::ComputeEncoder& enc, const Proj& p,
                    const metal_compute::SharedBuffer& x,
                    const metal_compute::SharedBuffer& y, int M, int G,
                    int rows);
  void gemm_f32_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& w,
                 const metal_compute::SharedBuffer& x, std::size_t xoff,
                 const metal_compute::SharedBuffer& y, std::size_t yoff,
                 int K, int N, int M);
  void rms_(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& x, std::size_t xoff,
            const metal_compute::SharedBuffer& w,
            const metal_compute::SharedBuffer& y, std::size_t yoff, int C,
            int rows, int ldx, int ldy, bool x_f32);
  void rope_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t xoff,
             bool compressed, int rows, int heads, int ld_row, int ld_head,
             int D, int pos0, int step, bool inverse);
  void fill_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& b, std::int32_t v,
             std::size_t words);
  bool topk_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& s, int rows, int C, int K,
             const metal_compute::SharedBuffer& out, std::size_t ooff = 0);

  metal_compute::MetalCompute* _mc = nullptr;
  Options _opt;
  DeepseekV41Config _cfg;
  DeepseekV41Engram _engram;
  std::shared_ptr<WeightSet> _ws;
  std::unique_ptr<SafetensorsRanges> _rows_file;     // engram rows
  std::unique_ptr<SafetensorsRanges> _expert_file;   // routed experts
  std::unique_ptr<MoeExpertCache> _cache;
  std::vector<std::unique_ptr<Stage>> _stages;
  Proj _main_proj;
  metal_compute::SharedBuffer _main_norm, _spark_norm;
  metal_compute::SharedBuffer _markov_embed, _markov_head;
  std::vector<float> _conf_w;              // [H + markov rank]
  // DSpark scratch: the main hidden [n, targets * H], main_x [n, H], the
  // block's residual, tokens, logits, Markov rows and bias, collapse
  metal_compute::SharedBuffer _mh, _mx, _sX, _stok, _slog, _memb, _mbias;
  metal_compute::SharedBuffer _xpre, _hcpart;
  metal_compute::ComputeFunction _fn_hc_mean, _fn_spark_attn, _fn_gather_tok;
  metal_compute::ComputeFunction _fn_add_argmax, _fn_hc_part;
  bool ensure_spark_(int rows);
  metal_compute::CommandStream _stream;   // every forward's and draft's
  int _ahead_max = 0;                      // the widest read-ahead width
  metal_compute::SharedBuffer _ahead_sc;   // the next layer's raw scores
  // A few rows' experts by slot, {slot, count, pairs...} (10 ints a group)
  metal_compute::SharedBuffer _mgrp;
  int _rows_mm = 2;                       // pairs a group (2, 4, 8; 0 off)
  std::vector<std::int32_t> _ahead_rank;
  // A decode or verify step's per-layer resolve as a HANDSHAKE inside the
  // running command buffers, not a commit and wait (~3 us against ~280 on
  // an M5 Ultra): words {posted, answered, gave up, heartbeat}; mailboxes
  // out (picks, next-layer scores) and in (slots). Taken where the load's
  // probe saw both sides' stores; VPIPE_DSV41_HANDSHAKE=0 is the A/B.
  metal_compute::ComputeFunction _fn_hs;
  // the same with a decode step's route before it in its dispatch;
  // VPIPE_DSV41_HS_ROUTE=0 keeps the route its own dispatch
  metal_compute::ComputeFunction _fn_hs_route;
  bool _hs_route = true;
  metal_compute::SharedBuffer _hsf, _hsbo, _hsbi;
  bool _hs_ok = false;
  std::uint32_t _hs_spins = 1u << 20;   // loads a give-up waits (probed)
  bool _hs_redo = false;                  // a give-up's forward, synced
  // ahead_study()'s record: [L][k][E] scores of the routers run early, this
  // step's and the step before's own [L][E], picks and blockers [L]
  struct AheadStudy {
    int k = 0, skip = 0;
    metal_compute::SharedBuffer early;
    std::vector<float> cur, prev;
    bool have_prev = false;
    std::vector<std::vector<int>> picks, blocked;
    // what each predictor named for layer L ([L * (K + 1) + p], the K
    // early routers then the step before; best first) and whether each
    // was not held when it was named
    std::vector<std::vector<std::int32_t>> named;
    std::vector<std::vector<std::uint8_t>> fresh_at;
    // per predictor (the K early routers, the step before, then the
    // unions of L - 1's with L - 2's .. L - K's) and width
    std::vector<std::array<double, 4>> pick_hit, block_hit, fresh;
    std::vector<double> picks_n, blocked_n, blocked_at;
    double steps = 0.0, seen = 0.0, read_ms = 0.0, wait_ms = 0.0;
    double reads = 0.0;
  };
  AheadStudy _study;
  void study_name_(int L);
  void study_step_();
  int _hs_drop = -1;                      // hs_drop_next's layer
  int _hs_redos = 0;
  std::uint64_t _kept_passes = 0, _two_passes = 0;
  // The heartbeat a waiting GPU sees move while the host is there (a long
  // expert read included); it stops with the process, so a forward the
  // host left gives up after one wait.
  std::thread _hs_beat;
  std::mutex _hs_mu;
  std::condition_variable _hs_cv;
  bool _hs_beating = false, _hs_quit = false;
  void hs_beating_(bool on);
  bool _bf16 = true;
  bool _mma = false;
  bool _mlx = false;

  metal_compute::ComputeLibrary _lib, _lib_dense, _lib_mma;
  metal_compute::ComputeFunction _fn_embed, _fn_rows_hc, _fn_hc_mix;
  metal_compute::ComputeFunction _fn_hc_pre, _fn_hc_post, _fn_rms, _fn_rope;
  metal_compute::ComputeFunction _fn_hc_post_pre, _fn_hc_post_pre_moe;
  metal_compute::ComputeFunction _fn_kv_norm_rope;
  // the collapse over several threadgroups a row (dsv41_hc_collapse):
  // their sums of squares [rows, G] and a row's count of groups done,
  // zero between uses -- set back by the last group, and by a fill at the
  // next forward after one failed (_hccnt_dirty); VPIPE_DSV41_HC_MT=0
  // keeps the one-group kernels
  metal_compute::ComputeFunction _fn_hc_coll;
  metal_compute::SharedBuffer _hcps, _hccnt;
  bool _hc_mt = true;
  bool _hccnt_dirty = false;
  bool _ced = true;                       // Options::ced, VPIPE_DSV41_CED
  int _ced_mid = -1;
  long _ced_pieces = 0;
  long _engram_ahead = 0;
  long _score_chunks_n = 0;
  int _hc_slices = 16;                    // the mix's (VPIPE_DSV41_HC_SLICES)
  metal_compute::ComputeFunction _fn_ring, _fn_attn, _fn_ungroup;
  metal_compute::ComputeFunction _fn_attn_split, _fn_attn_merge;
  metal_compute::ComputeFunction _fn_pool, _fn_keep, _fn_iscore;
  metal_compute::ComputeFunction _fn_iscore_mma, _fn_topk_par;
  metal_compute::ComputeFunction _fn_iscore_kept, _fn_iunpack;
  metal_compute::ComputeFunction _fn_topk_chunk, _fn_topk_merge;
  metal_compute::ComputeFunction _fn_bscore, _fn_bmask, _fn_topk;
  metal_compute::ComputeFunction _fn_route, _fn_swiglu4, _fn_down4;
  metal_compute::ComputeFunction _fn_swiglu4r[3], _fn_down4r[3];
  metal_compute::ComputeFunction _fn_msort, _fn_gu4g, _fn_down4g;
  // deepseek_v41_nax, a prefill's kernels on the matrix cores: the grouped
  // expert GEMMs (weights and activations staged in the element type, f32
  // accumulators; VPIPE_DSV41_MOE_NAX=0 keeps the f32 simdgroup ones), the
  // attention and the indexer's scores (below).
  metal_compute::ComputeLibrary _lib_nax;
  metal_compute::ComputeFunction _fn_gu4n, _fn_down4n;
  bool _moe_nax = false;
  long _nax_gemms = 0;                    // dispatched (tests)
  // a prefill's attention, a query's heads as the products' rows
  // (dsv41_attn_nax; VPIPE_DSV41_ATTN_NAX=0)
  metal_compute::ComputeFunction _fn_attn_nax;
  bool _attn_nax = false, _attn_psplit = true;
  long _nax_attn = 0;                     // dispatched (tests)
  // the indexer's dense scores, 4 query rows a key tile
  // (dsv41_index_score_nax; VPIPE_DSV41_INDEX_NAX=0)
  metal_compute::ComputeFunction _fn_iscore_nax;
  bool _index_nax = false;
  long _nax_index = 0;                    // dispatched (tests)
  metal_compute::ComputeFunction _fn_combine, _fn_swiglu, _fn_egate;
  metal_compute::ComputeFunction _fn_gemm32, _fn_fill;
  metal_compute::ComputeFunction _fn_rows_at, _fn_dq8, _fn_copy32;
  // the few-row GEMVs: [f32 | element type | FP8][one row | rows][KS 1 | 4]
  metal_compute::ComputeFunction _fn_gv[3][2][2];
  // the same, 8 lanes a row (VPIPE_DSV41_GEMV_R8=0: off)
  metal_compute::ComputeFunction _fn_gvr[3][2][2];
  bool _gemv_r8 = true;
  metal_compute::ComputeFunction _fn_gemv, _fn_gemm, _fn_gemm64;
  metal_compute::ComputeFunction _fn_mma, _fn_mma_deep;
  // MLX's affine codes, by width (kQBits): the few-row GEMVs [rows][KS],
  // the widening, the experts' halves [one pair | rows 2, 4, 8] and their
  // grouped GEMMs -- loaded for the widths the checkpoint stores
  static constexpr int kQBits[6] = {2, 3, 4, 5, 6, 8};
  static int qidx_(int bits);
  struct QFns {
    metal_compute::ComputeFunction gv[2][2], dq, sw[4], dn[4], gu, dg;
    // the grouped GEMMs' matrix-core twins (deepseek_v41_nax), if loaded
    metal_compute::ComputeFunction gun, dgn;
    bool loaded = false;
  };
  QFns _qf[6];
  bool load_q_(int bits, std::string* err);
  // The affine expert pools' strides a slot (codes in bytes, scales and
  // biases in elements; w1 / w3, then w2): the widest layer's sizes.
  long _xcw13 = 0, _xcs13 = 0, _xcw2 = 0, _xcs2 = 0;
  bool _xaffine = false;

  metal_compute::SharedBuffer _embed, _head, _norm;
  std::vector<Layer> _layers;
  // Rotary tables (cos, sin) a position: window-only layers', compressed
  // layers' (YaRN).
  metal_compute::SharedBuffer _rope_plain, _rope_comp;
  int _rope_cap = 0;

  std::unordered_map<CtxId, Ctx> _ctx;

  // scratch: the most rows and compressed positions a forward has had
  int _rows = 0, _cols = 0;
  static int index_cols_(const DeepseekV41Config& c, int ctx_end);
  static int score_rows_(int n, int width);
  static int piece_rows_(const DeepseekV41Config& c, int max_rows, int p0,
                         int left);
  metal_compute::SharedBuffer _ids, _X, _xa, _qa, _qr, _q, _kvr, _kv;
  metal_compute::SharedBuffer _ks, _pooled, _kk, _qi, _wp, _S;
  metal_compute::SharedBuffer _B, _bpick, _cmask, _idx, _o, _ot, _og, _ao;
  metal_compute::SharedBuffer _sc, _eid, _wt, _slot, _gu, _sact, _shout;
  metal_compute::SharedBuffer _act, _part, _fo, _ekv, _logits;
  metal_compute::SharedBuffer _srow, _sdst, _swt, _tile2e;
  double _grouped_ppe = 4.0;              // pairs an expert, grouped from
  std::uint64_t _grouped_steps = 0;
  metal_compute::SharedBuffer _mrows, _mpos, _tmask, _imask;
  metal_compute::SharedBuffer _wdq;       // an FP8 projection widened
  metal_compute::SharedBuffer _mixA[3], _mixF[3], _pm[2];
  // a few rows' split attention: partials [n, NH, chunks, 512], (max, sum)
  metal_compute::SharedBuffer _apart, _aml;
  metal_compute::SharedBuffer _tkk, _tkp, _tkt, _tkr;   // two-phase top-k
  bool _split_attn = true;                // VPIPE_DSV41_SPLIT_ATTN=0: off
  bool _fp8_wo_a = true;                  // VPIPE_DSV41_WO_A_BF16=1: off
  bool _concurrent = true;                // VPIPE_DSV41_CONCURRENT=0: off
  bool _hc_fuse = true;                   // VPIPE_DSV41_HC_FUSE=0: off
  bool _index_mma = true;                 // VPIPE_DSV41_INDEX_MMA=0: off
  bool _index_kept = true;                // VPIPE_DSV41_INDEX_KEPT=0: off
  bool _topk_par = true;                  // VPIPE_DSV41_TOPK_PAR=0: off
  int _topk_two = 1;                      // VPIPE_DSV41_TOPK_TWO: 0 off
  int _wo_a_codes = 0;                    // layers whose wo_a went back
  std::vector<metal_compute::SharedBuffer> _erow;   // per engram slot

  int _capture = -1;
  std::vector<float> _captured;
  std::vector<std::vector<std::int32_t>> _routes, _force, _force_draft;
};

}  // namespace vpipe::genai

#endif  // VPIPE_GENERATIVE_MODELS_DEEPSEEK_V41_METAL_DEEPSEEK_V41_MODEL_H
