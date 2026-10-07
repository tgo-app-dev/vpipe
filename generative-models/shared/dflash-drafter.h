#ifndef VPIPE_GENERATIVE_MODELS_SHARED_DFLASH_DRAFTER_H
#define VPIPE_GENERATIVE_MODELS_SHARED_DFLASH_DRAFTER_H

// DFlash / DFlash 2 -- a block-diffusion DRAFTER for speculative decoding
// (z-lab, arXiv:2602.06036; Inco AI's DFlash 2, inco.ai/blog/dflash2).
//
// A drafter is a small Qwen3-shaped stack with no embedding and no output
// head of its own: it borrows both from the TARGET it drafts for, and it
// conditions on the target's hidden states at a handful of layers
// (dflash_config.target_layer_ids). One forward drafts a whole block:
//
//   context  the target's taps at every committed position, concatenated
//            across the tapped layers, -> fc -> hidden_norm: one row a
//            position, projected by each layer's k_proj / v_proj into
//            that layer's K/V cache (RoPE at the position's own index).
//   block    [last committed token, mask, mask, ...] at the next L
//            positions, embedded by the target, through the layers: each
//            block query attends to the cached context (a sliding window
//            on sliding layers) and to the block itself (non-causal, or
//            causal on DFlash v1's sliding layers).
//   drafts   the target's lm_head over block rows 1..L-1. DFlash takes
//            each row's argmax; DFlash 2 keeps each row's top-k and walks
//            them with its candidate-path selector, so the drafts form one
//            coherent path rather than L-1 independent guesses.
//
// DFlash 2 also wraps every attention and MLP sublayer in a grouped
// two-tap dynamic causal convolution along the block (the "suffix decay"
// fix). Both versions are read from config.json alone: the architecture
// name and dflash_config decide what is built.
//
// The drafter never decides a token: the target verifies every draft, so
// a wrong draft costs speed and never correctness. That is also why its
// weights may be held quantized (w8 / w4 in memory) -- a property of the
// acceptance rate, not of the output.
//
// THREADING / STATE: one context cache (per-layer K/V of the committed
// positions, each slot tagged with its absolute position). A caller
// serving several target contexts resets it when it switches; holes in
// the context (positions that were never ingested) cost acceptance only.

#include "apple-silicon/metal-compute/metal-compute.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vpipe::genai {

class WeightSet;

// dflash_config + the Qwen3 fields of a drafter's config.json.
struct DFlashConfig {
  int   hidden     = 0;
  int   n_layers   = 0;
  int   n_heads    = 0;
  int   n_kv_heads = 0;
  int   head_dim   = 0;
  int   ffn        = 0;
  int   vocab      = 0;
  float rms_eps    = 1e-6f;
  float rope_theta = 1.0e7f;
  // Tokens a draft forward covers: the anchor plus block_size - 1 drafts.
  int   block_size = 16;
  // Target layers whose OUTPUT (HF hidden_states[id + 1]) the drafter
  // conditions on, in fc's concatenation order.
  std::vector<int> target_layer_ids;
  int   num_target_layers = 0;
  int   mask_token_id = 0;
  // Per layer: the sliding window (0 = full attention) and whether the
  // block attends causally within itself.
  std::vector<int>  layer_window;
  std::vector<bool> layer_causal;
  // DFlash 2 (all 0 for DFlash v1).
  int   conv_kernel = 0;      // taps of the dynamic convolution
  int   conv_group  = 0;      // channels sharing one dynamic correction
  int   selector_rank = 0;
  int   selector_top_k = 0;
  // Rarely set; defaults reproduce the reference's.
  float input_embedding_scale = 1.0f;
  float output_multiplier     = 1.0f;
  float final_logit_softcap   = 0.0f;

  bool v2() const noexcept { return conv_kernel > 0 && selector_rank > 0; }
  const char* kind() const noexcept { return v2() ? "dflash2" : "dflash"; }
  int qd()  const noexcept { return n_heads * head_dim; }
  int kvd() const noexcept { return n_kv_heads * head_dim; }

  // Parse `dir`/config.json. Nullopt (with *err) when it is not a DFlash
  // drafter this implementation can run.
  static std::optional<DFlashConfig> from_dir(const std::string& dir,
                                              std::string* err = nullptr);
};

// True iff `dir` holds a DFlash / DFlash 2 drafter (its config.json names
// DFlashDraftModel or DFlash2DraftModel).
bool is_dflash_drafter_dir(const std::string& dir);

// What a drafter borrows from its target: the token embedding and the
// output head. Both encode into the caller's encoder, so a draft forward
// and the verify that consumes it share one command buffer.
class DFlashTarget {
public:
  virtual ~DFlashTarget() = default;
  virtual int hidden() const = 0;
  virtual int vocab() const = 0;
  // out[n, hidden] = embed(ids[0..n)), ids int32 at byte offset ids_off.
  virtual void encode_embed(metal_compute::ComputeEncoder&    enc,
                            const metal_compute::SharedBuffer& ids,
                            std::size_t                         ids_off,
                            const metal_compute::SharedBuffer& out,
                            int                                 n) = 0;
  // logits[n, vocab] = lm_head(x[n, hidden]), x at byte offset x_off.
  virtual void encode_head(metal_compute::ComputeEncoder&    enc,
                           const metal_compute::SharedBuffer& x,
                           std::size_t                         x_off,
                           const metal_compute::SharedBuffer& logits,
                           int                                 n) = 0;
};

class MetalDFlashDrafter {
public:
  struct Options {
    // Weight precision held in memory: 0 = the compute dtype as read
    // (dense), 8 / 4 = affine group-64, quantized at load.
    int  quant_bits  = 8;
    // The TARGET's activation dtype: bf16 (true) or f16. The drafter itself
    // always computes in bf16 -- its training dtype, and the only one its
    // residual stream fits: DFlash 2's for Qwen3.8-27B runs to tens of
    // thousands, past f16's range -- and converts where it meets an f16
    // target (taps in, embeddings in, hidden out to the head).
    bool target_bf16 = true;
    // Cache capacity of a FULL-attention layer, in positions (a sliding
    // layer holds its window). Past it the oldest positions are dropped.
    int  max_context = 32768;
  };

  ~MetalDFlashDrafter();
  MetalDFlashDrafter(const MetalDFlashDrafter&)            = delete;
  MetalDFlashDrafter& operator=(const MetalDFlashDrafter&) = delete;

  // Build from a drafter checkpoint the manager holds. Null (with *err)
  // when the checkpoint is not runnable here.
  static std::unique_ptr<MetalDFlashDrafter>
  load(std::shared_ptr<WeightSet>    ws,
       metal_compute::MetalCompute*  mc,
       const Options&                opt,
       std::string*                  err = nullptr);

  const DFlashConfig& config() const noexcept { return _cfg; }
  const Options& options() const noexcept { return _opt; }
  const std::string& dir() const noexcept;

  // Bytes the drafter holds: weights + the context caches.
  std::size_t resident_bytes() const noexcept;

  // Forget every cached context position.
  void reset();

  // Project `rows` context positions into every layer's cache. `taps`
  // holds the target's layer outputs LAYER-MAJOR: [n_taps][tap_rows][H]
  // in the TARGET's dtype (what its tap capture writes); rows
  // [row0, row0 + rows) of it are positions [pos0, pos0 + rows).
  void encode_ingest(metal_compute::ComputeEncoder&    enc,
                     const metal_compute::SharedBuffer& taps,
                     int tap_rows, int row0, int rows, int pos0);

  // Draft one block. `block_ids` holds L int32 ids -- the anchor (the
  // last committed token, not yet in the target's context) then L - 1
  // mask ids -- at positions pos0 .. pos0 + L - 1. The L - 1 drafts land
  // in out_ids[out_off ..) (int32), on the GPU.
  void encode_draft(metal_compute::ComputeEncoder&    enc,
                    DFlashTarget&                      target,
                    const metal_compute::SharedBuffer& block_ids,
                    int L, int pos0,
                    const metal_compute::SharedBuffer& out_ids,
                    int out_off);

  // A block's ids for `anchor` at block length L: [anchor, mask, ...].
  void fill_block_ids(std::int32_t* dst, std::int32_t anchor, int L) const;

  // Scratch for a test: the final-normed block hidden [L, H] of the last
  // encode_draft, bf16 (rows 1.. feed the head).
  const metal_compute::SharedBuffer& last_hidden() const noexcept;
  // Its top-k candidates of the last draft: values [L-1, k] f32, ids
  // [L-1, k] int32 (DFlash 2 only; empty otherwise).
  const metal_compute::SharedBuffer& last_cand_val() const noexcept;
  const metal_compute::SharedBuffer& last_cand_idx() const noexcept;
  // Bisecting aid: when on, encode_draft also copies the block residual
  // after every layer into debug_layers() [n_layers][L][H].
  void set_debug_layers(bool on);
  const metal_compute::SharedBuffer& debug_layers() const noexcept;

private:
  MetalDFlashDrafter() = default;
  struct Impl;
  std::unique_ptr<Impl> _impl;
  DFlashConfig _cfg;
  Options      _opt;
};

}  // namespace vpipe::genai

#endif
