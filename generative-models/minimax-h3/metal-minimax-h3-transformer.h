#ifndef GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_TRANSFORMER_H
#define GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_TRANSFORMER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "generative-models/shared/ane-ffn.h"
#include "generative-models/minimax-h3/metal-vdn-branch.h"
#include "generative-models/minimax-h3/minimax-h3-diffusers-layout.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/shared/fp8-expand.h"
#include "generative-models/shared/block-residency.h"
#include "generative-models/shared/wired-pool.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/mma-splitk.h"
#include "generative-models/shared/dit-block-progress.h"
#include "generative-models/shared/dit-flops.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/metal-sol-attention.h"
#include "generative-models/shared/runtime-lora.h"
#include "generative-models/shared/sage-attention.h"
#include "generative-models/shared/sol-attention.h"

namespace vpipe {
namespace genai {

class WeightSet;   // generative-models/weight-set.h
namespace minimax_h3 { class StreamKv; }   // minimax-h3-stream-kv.h

// The MiniMax-H3 omni denoiser (MiniMaxH3DiTModel): a 33B dense
// SINGLE-STREAM transformer that predicts video AND audio velocity from
// one packed sequence.
//
// It is the simplest topology of any DiT in this tree and the most
// unusual conditioning. Every block is: RMSNorm -> modulate -> full
// self-attention -> gated residual -> RMSNorm -> modulate -> SwiGLU ->
// gated residual. There is NO cross-attention anywhere, no per-modality
// block weights, and no second stream -- the text, the keyframe
// conditioning rows, the audio rows and the target video rows are all
// just rows of one sequence that attends to itself.
//
// What makes it a multimodal model is entirely at the edges:
//
//   * two input patch projections (video 24*1*2*2 = 96 -> 5376, audio
//     32 -> 5376) and a text projection 5120 -> 5376 followed by a
//     2-block token refiner;
//   * a per-ROW AdaLN. Each block projects the shared timestep embedding
//     2688 -> 96768 = 6 modulation vectors x 3 modalities, and every row
//     reads the entry at `timestep_index * 3 + tag`. One forward
//     therefore serves rows at DIFFERENT noise levels, which is how the
//     keyframes stay pinned at their noise-augmentation level while the
//     generated rows denoise;
//   * two output heads, video 5376 -> 96 and audio 5376 -> 32, both run
//     over every row with the rows of each modality selected after.
//
// Four details that a port gets wrong silently:
//
//   * the attention inner width is LARGER than the residual stream --
//     56 heads x 128 = 7168 against hidden 5376. Deriving hidden from
//     heads*head_dim, which is right for every other model here, gives a
//     model that loads and produces nonsense.
//   * fc1 is FUSED: one [2*ffn, hidden] matrix holding [gate; up], gate
//     first, and the block computes silu(gate)*up. The halves are not
//     labelled anywhere -- reading them the other way round is a model
//     that loads and generates smooth nonsense. diffusers implements the
//     OTHER order, [value; gate], and reconciles it at conversion time:
//     `scripts/convert_minimax_h3_to_diffusers.py` swaps the two halves
//     when it writes `ff.net.0.proj`. So a diffusers-FORMAT checkpoint
//     is value-first and the raw MiniMaxAI one is gate-first, and this
//     reads the raw layout.
//   * RoPE rotates only 96 of the 128 head channels, rotate-half, and
//     passes the other 32 through.
//   * q/k RMS is PER HEAD over head_dim (Wan's is across the whole
//     projection).
//
// SIZE. At bf16 the checkpoint is 66 GB, which does not fit any box here
// -- so unlike every other model in this tree, quantizing it is a
// precondition for running it at all rather than a memory optimization.
// About 13B of the 33B parameters are the per-block AdaLN projections;
// those depend only on the timestep, so a run that knows its schedule up
// front could bake every step's modulation and never keep them resident.
// That is not done yet (see the note on _mod in the .cc).
class MetalMiniMaxH3Transformer {
 public:
  // Rows one matmul2d dispatch may touch, over an N-wide destination
  // contracting a K-wide source.
  //
  // The mma tiles address through `dextents<int32_t, 2>`, so the tile
  // whose base crosses 2^31 BYTES stops storing -- silently, leaving
  // whatever was in the buffer. MEASURED at H3's fc1 shape
  // ([38222, 28672] bf16 = 2.19 GB, the 1344x768 x 124-frame sequence):
  // rows 37504..38221 came back exactly as they went in. Banding M
  // rebases each dispatch, which keeps every offset the kernel computes
  // small; below the line the band is the whole thing and nothing about
  // the encoding changes. VPIPE_H3_MMA_ROW_BAND forces a band, which is
  // how a small shape can be tested against its own unbanded self.
  //
  // BOTH extents, because the limit is a property of every operand and
  // not of the destination. Banding on N alone protects the widest
  // tensor a projection WRITES and leaves the one it READS unguarded --
  // which is invisible on three of H3's four projections and reachable
  // on the fourth: fc2 has the block's smallest N (5376, so its band is
  // 199680 and never bites) and its largest K (14336, so at 75136 rows
  // its source is 2.15 GB). MEASURED, minimax_h3_blocks.
  // tail_rows_match_a_small_reference at [76800, 14336] bf16: mma128
  // and mma128x256tn2 return 9633792 wrong elements, every row from
  // 75008 = ceil128(2^31 / 28672) to the end, against a small-buffer
  // reference. steel, the int8 arm and split-K are clean.
  static int mma_row_band(int N, int K);

  struct Config {
    int hidden        = 5376;
    int n_heads       = 56;
    int head_dim      = 128;
    int n_layers      = 50;
    int n_refiner     = 2;
    int ffn           = 14336;    // SwiGLU inner; fc1 is 2x this wide
    int video_channels = 24;
    int audio_channels = 32;      // audio LATENT channels, not stereo
    int patch_t       = 1;
    int patch_h       = 2;
    int patch_w       = 2;
    int text_dim      = 5120;
    int freq_dim      = 256;      // sinusoidal timestep width
    int time_hidden   = 5376;
    int time_dim      = 2688;     // AdaLN projection input
    // ComfyUI's PRUNED form (its "curve form", community DiTs saved
    // pruned): no timestep MLP. A `adaln_t_table` [curve_grid, time_dim]
    // (1025 x 8) holds points of the time-embedding curve, lerped at
    // t * (curve_grid - 1), and every AdaLN projection is [D, time_dim]
    // over that point, with no silu, in f32. 0 = the released form.
    int curve_grid    = 0;
    bool pruned() const { return curve_grid > 0; }
    // PRUNED only: a checkpoint that carries the RELEASED timestep MLP
    // (`time_embedder.*`; the graph's own partition), for what needs
    // the full embedding on a pruned DiT -- a full-model adapter's AdaLN
    // modules, ported onto the curve, and an adapted or flow-map
    // embedding, projected onto it (see CurveBasis). Empty: those are
    // skipped (AdaLN modules) or refused (a flow map), as before.
    std::string time_mlp_dir;
    int rope_freq_dim = 16;       // per AXIS; 3 axes, doubled -> rot 96
    // Only for a checkpoint with no `rope.inv_freq` tensor (the
    // diffusers layout), which the table is then computed from.
    float rope_theta  = 10000.0f;
    float norm_eps       = 1e-5f;
    float qk_norm_eps    = 1e-5f;
    float final_norm_eps = 1e-5f;

    // How the fused `attn.qkv_proj` groups its 3*inner() output rows.
    // MiniMaxAI's released checkpoint groups them PER HEAD --
    // [h0(q,k,v) | h1(q,k,v) | ...] -- while Comfy-Org's single-file
    // conversion reorders them flat, [all q | all k | all v].
    //
    // This is the one field that CANNOT be read off the weights: both
    // layouts carry the same tensor name and the same [3*inner, hidden]
    // shape, so reading the wrong one loads cleanly and scrambles
    // attention in every block -- which looks like a broken model, not
    // like a broken loader. It is therefore set from WHERE the
    // checkpoint came from: config_from_json() leaves it true for a
    // diffusers directory and clears it for a Comfy-Org component.
    bool qkv_per_head = true;

    // ACCELERATED MODE (lossy, opt-in): run the block GEMMs as dynamic
    // int8 instead of bf16 -- quantize the activation and the (dequanted)
    // weight per call and multiply in int8. Only the matrix-core route can
    // take it, since it is the only one that materializes a dense weight,
    // and I8GemmContext self-gates on rows (M >= 1024), so a short clip
    // silently keeps the bf16 tiles. All four projections qualify: hidden
    // 5376 is not a multiple of the int8 GEMM's 512-wide chunk, and the
    // context zero-pads the contraction to 5632 rather than refusing the
    // shape (exact -- mlp_fuse.i8_kpad pins it bit-identical to padding by
    // hand) at a cost of 4.8% extra int8 MACs, inside its 10% cap.
    // Set from generate-video's `i8_gemm`; VPIPE_I8_GEMM overrides.
    bool i8_gemm = false;
    // ...and which int8 GEMM (I8GemmContext::Tier, generate-video's
    // `i8_gemm_tier`): 0 accurate (the per-call route above), 1 fast,
    // 2 token, 3 token unrotated. The last three HOLD the 50 main
    // blocks' four projections in I8GemmContext's int8 layout (rotated
    // coarse groups, or one scale a row and a token), converted once
    // into the entry a kept block holds; a Draw Things i8x checkpoint's
    // codes ARE the unrotated token layout and are read as stored,
    // kept or streamed. Off under the ANE tier, whose stagers read the
    // checkpoint's own weights. VPIPE_I8_TIER / VPIPE_I8_TOKEN_ROT
    // override it.
    int i8_tier = 0;
    // A w4 pack's block projections on the fast / token tiers: HOLD
    // their int8 layout (true; twice their bytes, the fastest) or keep
    // the pack and convert it before each GEMM into one scratch (false,
    // the default; a streamed block always does). The engine's call --
    // generate-image sets it where the box has room.
    bool i8_hold_w4 = false;

    // Sol-Attn (LOSSY), from generate-video's `sol_attn*` keys. Off by
    // default. Attention here is dense self-attention over the packed
    // sequence, which is exactly the shape the method approximates; the
    // KV SINK is set from the layout rather than from config, because
    // the prompt and the soundtrack sit in that sequence and a prompt
    // read through a block centroid is a prompt half-read.
    //
    // MUTUALLY EXCLUSIVE WITH THE VDN LINEAR BRANCH, and not as a
    // policy: VDN replaces attention with a windowed softmax whose
    // complement a linear recurrence carries, so approximating the
    // softmax half would leave the two halves describing different key
    // sets and the partition -- the whole invariant of that design --
    // would stop holding.
    //
    // INDEPENDENT OF `i8_gemm` above, and settable with it. The two act
    // on different halves of a block: i8_gemm on the GEMMs, this on the
    // attention between them. Neither reads the other's state.
    sol::Config sol;
    // SageAttention. Independent of `sol` beside it and of `i8_gemm`:
    // Sol decides which key blocks are attended, Sage how the ones that
    // are get computed, and i8_gemm how the GEMMs around them do. None
    // of the three reads the others.
    sage::Config sage;

    // ---- the ANE tier (LOSSY, opt-in) -------------------------------
    //
    // The block feed-forward on the Apple Neural Engine beside the GPU:
    // the rows are split between the two engines and run concurrently,
    // which is exact because a feed-forward is row-independent. ONE
    // runtime-weight module serves every block (see shared/ane-ffn.h), so
    // the tier's memory is fixed -- ane_runtime_bytes() -- and it compiles
    // once per shape. fp16 on the ANE, which is safe for the feed-forward
    // alone: only its output re-enters the bf16 residual stream, and that
    // addition stays on the GPU. A 4/8-bit affine checkpoint is
    // dequantized into the module's inputs per block.
    //
    // `ane_rows` is the ANE's share of the rows, 0 auto-balancing from the
    // measured times; `ane_layers` caps how many blocks use it, 0 all.
    // Refiner blocks never do: they run over the text rows alone, far
    // under one chunk. A runtime LoRA on fc1 or fc2 is merged into the
    // ANE's staged weights per block; the GPU's rows apply it as usual.
    // Set from generate-video's `ane_*` keys once the plan has granted the
    // module; VPIPE_H3_ANE_CHUNK / VPIPE_H3_ANE_PROFILE for tuning.
    bool  ane_ffn    = false;
    float ane_rows   = 0.0f;
    int   ane_layers = 0;
    // The fused qkv projection split onto the ANE the same way, on its
    // own matmul tier (auto share, same on/off controller). Set from
    // generate-video's `ane_qkv` key, which REQUIRES `ane_ffn`: the two
    // modules are booked as one CoreML unit and granted together.
    // VPIPE_H3_ANE_QKV=1 also turns it on.
    bool  ane_qkv    = false;

    int inner() const { return n_heads * head_dim; }        // 7168
    int video_patch_elems() const
    {
      return video_channels * patch_t * patch_h * patch_w;  // 96
    }
    // Which TASK partition these weights are ("fl2va" / "ref2va", empty
    // when nothing said). Carried on the Config because it decides
    // WHICH FILE load() reads from a tree holding both -- a 66 GB
    // choice that no property of the bytes can settle.
    std::string partition;

    // 2 * 3 * rope_freq_dim: three axes share one inv_freq table and the
    // concatenated block is then concatenated with itself.
    int rope_rot() const { return 6 * rope_freq_dim; }      // 96
    int adaln_out() const { return 6 * hidden * minimax_h3::kModalityNum; }
  };

  // What a CALLER chooses per generation, as against Config, which is
  // what the checkpoint IS. It lives here rather than in the driving
  // stage because every field is a fact about this family -- two
  // schedules, two condition levels, a soundtrack -- and none of them
  // means anything to a video DiT that generates no audio.
  struct GenerationParams {
    // The two sigma schedules. Video and audio are stepped in lockstep
    // over the same step count but with DIFFERENT shifts: the released
    // checkpoint's are 12 and 3, and they are not interchangeable.
    double video_shift = 12.0;
    double audio_shift = 3.0;
    // The timestep the pinned CONDITION rows sit at. They are real
    // encoded frames, so 1.0 -- CLEAN in this model's convention, where
    // t = 1 - sigma. Exposed because it is the reference's
    // `condition_video_timestep` rather than a constant, and a
    // checkpoint trained with noise-augmented anchors would want it
    // lower.
    double condition_timestep = 1.0;
    // The same, for a `ref2va` reference SOUNDTRACK. Separate from the
    // video one because the two conditionings are independent: a
    // reference video's frames and its audio are encoded by different
    // VAEs and enter as different rows.
    double condition_audio_timestep = 1.0;
    // Soundtrack duration. 0 derives it from the video, which is what
    // keeps the two modalities the same length by construction; a
    // positive value is for a clip whose audio is deliberately not.
    double audio_seconds = 0.0;
    // A REFINE (generate-video's init_latent: Open-H3-Regenerate-2K):
    // the noise the clip is put back to, and the shift its steps follow
    // -- MiniMaxH3Scheduler::refine_sigmas. Negative start = the
    // release's own grid for the step count (0.909375 at 3 steps).
    double refine_start_sigma = -1.0;
    double refine_shift       = 6.0;

    // Parse the `minimax-h3-model-config` beat. Absent keys keep the
    // defaults above; `err` collects what was ignored, and is not fatal.
    static GenerationParams from_flex(const FlexData& fd,
                                      std::string* err = nullptr);

    // How many audio latents this request generates for `num_frames`
    // video frames at `fps` -- `audio_seconds` when one was asked for,
    // the video's own duration otherwise. The rounding rule is the
    // model's, so it belongs beside it rather than in each caller.
    int audio_latents(int num_frames, double fps) const;
  };

  // Read a Config out of the checkpoint's `transformer/config.json`.
  // False when the file is missing or is not a MiniMaxH3DiTModel config.
  // `partition` ("fl2va" / "ref2va", empty = probe) says WHICH of the two
  // task partitions is meant, for a tree that holds both. It is
  // authoritative: a caller who read it from the models DB knows
  // something the bytes do not say.
  static bool config_from_json(const std::string& dit_dir, Config& out,
                               std::string* err = nullptr,
                               const std::string& partition = {});

  // `dit_dir` may be the transformer directory itself, the partition
  // root that holds it (`.../FL2VA`), or the repository root -- the
  // catalogue registers the REPO, whose pipeline lives one level down in
  // a partition subdirectory, so a caller holding a registry path has
  // neither of the first two.
  // `partition` picks between the two DiT files when a repo holds both.
  // Empty keeps the historical preference (`fl2va`), which is the right
  // default only because every graph that predates Ref2VA meant it.
  static std::string resolve_dit_dir(const std::string& path,
                                     const std::string& partition = {});

  // Which TASK partition a checkpoint is: "fl2va", "ref2va", or empty
  // when nothing says.
  //
  // Worth a function of its own because the two are indistinguishable
  // from the weights: same 535 tensor names, same shapes, same embedded
  // config, byte for byte. Only the packaging says -- `model_index.json`
  // carries `_minimax_h3.partition` in a diffusers checkout, and a
  // Comfy-Org repack says it in the DiT's FILENAME and nowhere else. A
  // caller that guesses wrong loads and runs and conditions on nothing,
  // so "empty" is a real answer and not a failure.
  // The config.json key a DERIVED checkpoint (model-quantize's output)
  // records its partition under. Public so the producer and the reader
  // cannot drift apart on the spelling.
  static constexpr const char* kPartitionKey = "_minimax_h3_partition";

  // `hint` (from the models DB) WINS: two records can share one
  // directory precisely so that the tree cannot answer this, and a
  // caller holding the record knows what probing cannot find out.
  static std::string partition_of(const std::string& path,
                                  const std::string& hint = {});

  // "minimax-h3-fl2va" / "minimax-h3-ref2va" -> "fl2va" / "ref2va";
  // anything else -> empty. The models DB speaks in model_type and this
  // class in partitions, and one spelling of the mapping keeps the two
  // from drifting.
  static std::string partition_of_model_type(const std::string& model_type);

  // `stream_blocks` (memory-bounded): don't preload the 50 main blocks --
  // retain the weight set and load/free each block on demand inside
  // forward(), so ~one block is resident instead of the whole DiT. At w8
  // a block is ~0.65 GB against ~33 GB for the stack, and this is the
  // LARGEST of the DiTs here, so a 64 GB box cannot hold it resident
  // beside the 10 GB video VAE. The preloaded top-level weights (patch
  // projections, time MLP, refiner, norm_out heads) stay. Slower (a
  // commit+wait per block). Mirrors the Krea-2 / FLUX-2 / QIE path.
  //
  // THE PINNED PREFIX IS RETIRED. BlockResidency replaces it.
  //
  // It was a fraction of TOTAL ram decided before the run, so it could
  // not see the machine it landed on: not another process, not this
  // graph's peers, not the moment a peer let go. On a tight box that is
  // where it did most harm -- plan_streaming's own note records a 16 GB
  // run taken from no swap at all to 11 GB resident with 3 GB of swap
  // moving continuously, by a prefix sized exactly that way.
  //
  // What replaces it measures: admission against the live budget, a
  // probe read off the room actually free, doubling while the box stays
  // healthy, and a shed the moment the pages kept are found outside RAM.
  //
  // Prefer the WeightSet overload: the set is the manager's shared,
  // reference-counted view of the checkpoint. The dir overload opens a
  // PRIVATE set (tests, and callers with no session to ask).
  // Resolves the DiT through `cfg.partition`, so a directory holding
  // both partitions loads the one the caller asked for rather than the
  // one whose filename sorts first.
  // A LoRA adapter applied AT RUNTIME: every adapted projection computes
  // y = W x + scale * B (A x) instead of having the delta folded into W.
  //
  // Why not just fuse it into the weights. MEASURED on the MiniMax-H3
  // Turbo adapter against its bf16 base: the update is 2-4e-4 relative to
  // the weights while bf16's step is ~4e-3 relative, so `W + dW` rounds
  // straight back to `W` for 94% of elements and the merge keeps 46-78%
  // of the delta's norm. A 4-bit base is coarser again. Rank 64 against
  // K=5376 costs ~1.5% of the projection's FLOPs and is immune to the
  // base's storage precision -- so for a few-step distillation, where the
  // whole point is a small, precise correction, runtime is the accurate
  // path and fusing is the cheap one.
  //
  // Named at LOAD because it changes how the weights are BUILT: an
  // adapted `mlp.fc1` cannot take the fused-SwiGLU kernel, whose epilogue
  // writes silu(gate)*up straight out of the accumulator and leaves
  // nowhere for a pre-activation delta to land.
  struct LoraSpec {
    std::string path;          // a single .safetensors of lora_A/B pairs
    float       scale = 1.0f;  // folded into A at load; 1.0 = as trained

    // Which row order the adapter's FUSED `attn.qkv_proj` is in.
    //
    // The two publishers order that projection's 3*inner output rows
    // differently -- Comfy-Org's repack FLAT ([all q | all k | all v]),
    // every MiniMaxAI release PER HEAD -- and this tree re-orders
    // neither, pointing the attention at the right offsets instead. So
    // an adapter's delta has to be in the same order as the base it is
    // added to, and one built for the other publisher lands on the
    // wrong channels in every block.
    //
    // kAuto reads FLAT, because every H3 adapter published so far is:
    // lightx2v's ComfyUI copies measurably (their B is exactly
    // block-diagonal), larryvrh's Turbo by its repo's own
    // `base_model: Comfy-Org/MiniMax-H3`. On a per-head DiT the rows
    // are then permuted at bind. kPerHead is the escape hatch for an
    // adapter actually trained on MiniMaxAI's weights -- none is known,
    // and assuming one exists would silently break every adapter that
    // does.
    enum class QkvLayout { kAuto, kFlat, kPerHead };
    QkvLayout qkv_layout = QkvLayout::kAuto;
  };

  // How many adapters may ride the DiT at once: the shared Stack's
  // eight, which is what the stages' `lora` .. `lora8` keys name.
  //
  // The common request is a few-step Turbo distillation, which is a
  // correction to the WHOLE model and belongs at the strength it was
  // trained at, beside style or identity adapters, which are the things
  // actually being dialled. They stay SEPARATE slots rather than one
  // merged set of factors because their strengths move independently --
  // concatenating them on the rank axis is exact arithmetic (A = [s1 A1;
  // s2 A2], B = [B1 | B2]) and would be free in the forward, but it
  // folds every strength into A and so turns the live knob back into a
  // rebuild, for the adapters most likely to be swept. Each slot costs
  // its own pair of skinny GEMMs and its own rows of the [seq, rank]
  // scratch; the base tile absorbs one adapter's second GEMM.
  static constexpr int kMaxLoraSlots = lora::Stack::kMax;

  static std::unique_ptr<MetalMiniMaxH3Transformer>
  load(const std::string& dit_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  static std::unique_ptr<MetalMiniMaxH3Transformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false,
       const std::vector<LoraSpec>& loras = {});

  // How many projections carry a runtime adapter, summed over the slots
  // that bound (0 = none attached).
  int lora_modules() const;
  // An AdaLN curve branch's input (see CurveSet) for each timestep, as
  // the reference reads it: pos = t * (knots - 1), lo = floor(pos)
  // clamped to knots - 2, torch.lerp(table[lo], table[lo + 1], pos - lo)
  // -- its two-sided form, in f32. `table` is [knots, 8], `out` [t, 8].
  static void curve_coords(const std::vector<float>& table, int knots,
                           const std::vector<float>& t, float* out);
  // The same for ONE slot, so a caller can report which adapter took.
  int lora_modules(int slot) const;
  // Slots that bound. Indices below this are live; the rest are not.
  int lora_slots() const { return static_cast<int>(_lora.size()); }
  // Of a slot's projections, how many bound BANDED (split q/k/v with no
  // block-diagonal zeros; see lora::Factors::banded).
  int lora_banded(int slot) const;
  // Bytes every slot's GPU factors hold -- what attaching the adapters
  // costs in memory, which is not what they cost on disk.
  std::size_t lora_bytes() const;
  // The width of the forward's [rows, rank] adapter scratch, every
  // slot's widest projection together.
  int lora_scratch_rank() const { return _lora_rank_total; }
  // GEMMs run on ComfyUI's int8 ConvRot (or W4A8) codes as stored, on the
  // token tier (Linear::cr), since load -- engagement, for a test.
  std::uint64_t convrot_native_gemms() const { return _cr_gemms; }

  // What an adapter FILE will cost before anything loads -- the plan's
  // figure, off its header; lora_bytes() is the loaded model's. Every
  // factor is held bf16 whatever the file stores. `whole`: all of them
  // (a preloaded DiT). `streamed`: each main block's spilled to a file
  // and the largest block's set held for the block being run, beside
  // the rest (spill_lora_). `rank`: the widest the [rows, rank] scratch
  // takes for this file, a block's q, k and v together as the bind fuses
  // them. Main blocks at or past `n_layers` are not counted, as the bind
  // skips them (0 counts every one). qkv bound BANDED, the default;
  // VPIPE_H3_LORA_NO_BAND's block-diagonal B holds three of qkv's B.
  struct LoraFootprint {
    std::size_t whole    = 0;
    std::size_t streamed = 0;
    int         rank     = 0;
  };
  // `pruned`: the DiT's config when it is ComfyUI's pruned form. Its
  // AdaLN modules are then held as f32 deltas [D * 9] (bind_pruned_
  // adaln_) rather than as factors -- ComfyUI `.diff` / `.diff_b`
  // patches too, which a released DiT does not read -- and take no
  // [rows, rank] scratch; an adapter on the timestep MLP adds that MLP,
  // read to the host from `pruned->time_mlp_dir`.
  static LoraFootprint lora_footprint(const std::string& path,
                                      int n_layers = 0,
                                      const Config* pruned = nullptr);
  // Whether an adapter FILE moves a pruned DiT's timestep embedding (it
  // adapts the timestep MLP, or is a flow map), so the forward needs the
  // off-curve tables bake_adaln builds (off_curve_bytes).
  static bool moves_time_embedding(const std::string& path);
  // What a pruned DiT holds to run such an adapter for `max_rows`
  // schedule rows: the tables, one released AdaLN block read while they
  // are built, and the released MLP on the host.
  static std::size_t off_curve_bytes(const Config& c, int max_rows);

  // The adapter's strength, per FORWARD. Live because it can be: the
  // factors are the adapter's and the strength is the request's, so it
  // rides the accumulating GEMM as a constant rather than being folded
  // into A. Turning it costs nothing and needs no reload -- which is the
  // difference between a knob and a rebuild, and the reason to prefer
  // the runtime path when a strength has to be swept. 0 skips the two
  // GEMMs entirely, so "off" is exactly off.
  // Per SLOT, because two adapters are two requests: the distillation is
  // pinned where it was trained and the style one is the knob. An index
  // past what bound is ignored.
  void set_lora_scale(int slot, float s);
  float lora_scale(int slot) const;

  // A FLOW-MAP adapter: one that conditions every step on the INTERVAL
  // it integrates, (t, r), rather than on the point t. HyperFlow
  // (videorebirth/hyperflow, after AnyFlow's two-time conditioning) is
  // the one that exists: an 8-step self-distillation of this model.
  //
  // It is a LoRA plus exactly one architectural change, and that change
  // is at the timestep MLP. Beside the checkpoint's own embedder it
  // carries an ENDPOINT embedder -- a copy of the base one with its own
  // adapter -- that embeds r, and the two are blended
  //
  //     temb = emb(t) + gate * (emb_r(r) - emb(t))
  //
  // before the silu every AdaLN applies. Each distinct timestep of a
  // forward therefore needs its endpoint too (Step::endpoints); rows
  // that do not move -- conditioning anchors, a reference soundtrack --
  // carry r == t.
  //
  // WHAT THE FILE DECIDES. Its header names the raw sigma grid it was
  // distilled on (9 points, 8 forwards) and the shifts it was distilled
  // at; those are a property of the weights, the way a step count is
  // for any distillation, and denoise() takes them from here unless the
  // caller supplies a grid of its own.
  struct FlowMap {
    float gate = 0.0f;                 // `hyperflow_gate`
    std::vector<float> sigmas;         // raw (UNshifted) grid, 1 -> 0
    double video_shift = 0.0;          // trained shifts; 0 = unstated
    double audio_shift = 0.0;
    std::string version;               // `hyperflow_version`
    int slot = -1;                     // which adapter slot carries it
  };
  // The model evaluations a flow-map adapter FILE's grid drives, read
  // from its safetensors header alone -- 0 when the file is not one,
  // has no grid, or cannot be read. For a caller sizing a run BEFORE
  // the DiT exists, which is when the step count bounds the AdaLN
  // bake's table and a graph's `steps` would understate it.
  static int flow_map_steps(const std::string& adapter_file);
  // Null when no slot carries a flow-map adapter.
  const FlowMap* flow_map() const
  {
    return _flow.slot >= 0 ? &_flow : nullptr;
  }
  // Whether a forward NEEDS endpoints: a flow-map adapter is bound and
  // its slot's strength is not 0. At 0 the adapter is off in both of
  // its halves -- the LoRA and the blend move together, because one on
  // without the other is a state it was never trained in -- and the
  // model is exactly the base again.
  bool two_time_active() const;

  // A few-step adapter's SAMPLING CONTRACT, stated in a `manifest.json`
  // BESIDE its file rather than in it -- Pika H3 Flex's (Pika-Labs/
  // Pika-H3-Flex, `fl2va/` and `ref2va/`):
  //
  //   "trunk": "transformer" | "transformer_ref",
  //   "numerics_policy": "bf16_conditioning_v1" | "fp32_conditioning_v1",
  //   "sampling": {"knots": [1000, 750, 500, 250, 100, 40, 0],
  //                "video_shift": 12.0, "audio_shift": 3.0,
  //                "method": "renoise"}
  //
  // The knots over 1000 are a RAW grid each modality pushes through its
  // own shift, in double and then rounded to float32 (diffusers'
  // `shift_provided_sigmas`). "renoise" is the update the adapter was
  // trained for: each step takes the clean estimate x0 = x + sigma v and
  // puts it at the next sigma with FRESH noise, (1 - s') x0 + s' n --
  // not an Euler step, which the release says outright is not
  // equivalent. What the file decides, as a flow-map adapter's header
  // does: denoise() runs it unless the caller names a grid of its own.
  struct SamplingContract {
    std::string release;               // `release_id`
    std::string task;                  // "fl2va" / "ref2va"
    std::string trunk;                 // "transformer" / "transformer_ref"
    std::vector<double> sigmas;        // knots / 1000, raw, 1 -> 0
    double video_shift = 0.0;          // 0 = unstated
    double audio_shift = 0.0;
    bool   renoise = false;            // method "renoise"
    // `bf16_conditioning_v1`: the keyframe latents are rounded to bf16
    // before they are noised and packed.
    bool   bf16_conditioning = false;
    int    slot = -1;                  // which adapter slot states it
  };
  // The contract an adapter FILE states, read from the `manifest.json`
  // in its directory. False with `err` empty when there is none (no
  // manifest, or one without `sampling`); false with a reason when one
  // is there and does not parse -- a malformed contract is refused, not
  // read as "no contract", since the run would then be the generic
  // sampler the release warns against.
  static bool sampling_contract_of(const std::string& adapter_file,
                                   SamplingContract* out, std::string* err);
  // The partition a contract's `trunk` names ("fl2va" / "ref2va"), or
  // empty for a trunk this model does not know.
  static std::string contract_partition(const SamplingContract& c);
  // The contract of the slot that stated one while its strength is not
  // 0; null otherwise. At 0 the adapter is off, and so is its sampler.
  const SamplingContract* sampling_contract() const;

  // The timestep MLP's output BEFORE the silu, in f32, for each
  // `timesteps[i]` (and, when `endpoints` is given and a flow-map
  // adapter is live, blended with the endpoint embedding of
  // `endpoints[i]`). What the forward rounds to bf16 after its silu;
  // public so a test can hold it against the reference's own module.
  // PRUNED (Config::curve_grid), the point of `adaln_t_table` each
  // timestep lands on (curve_coords, t clamped to [0, 1]), which the
  // AdaLN projections read as it is: no MLP and no silu.
  bool time_embedding(const std::vector<float>& timesteps,
                      const std::vector<float>* endpoints,
                      std::vector<float>* out) const;

  // Where row `r` of part `pt` (0 = q, 1 = k, 2 = v) of a SPLIT adapter
  // lands in the fused [3 * inner] qkv output.
  //
  // Public and static because it is the one piece of the diffusers
  // conversion that has to agree with something else: the forward's own
  // Q_OFF / K_OFF / V_OFF and QKV_HSTRIDE, which describe the same
  // layout for the buffers it hands the attention. If those two ever
  // disagree the adapter lands on the wrong channels and nothing fails
  // -- see the test that pins them together.
  //
  //   flat     [ all q | all k | all v ]        -- Comfy-Org's repack
  //   per head [ q0 k0 v0 | q1 k1 v1 | ... ]    -- MiniMaxAI's releases
  static int qkv_fused_row(int pt, int r, int inner, int head_dim,
                           bool per_head)
  {
    if (!per_head) { return pt * inner + r; }
    const int h = r / head_dim, d = r % head_dim;
    return h * 3 * head_dim + pt * head_dim + d;
  }

  // Leading blocks pinned resident in streaming mode (0 = pure streaming,
  // or preloaded). For logging the RAM-for-speed decision.
  // Always 0 now; kept so callers that report it need not all change
  // in one commit. Remove once nothing asks.
  int pinned_blocks() const { return 0; }
  bool streaming() const { return _stream_blocks; }
  // Whether the main blocks' projections run in the int8 layout
  // (Config::i8_tier), and -- for a preloaded stack -- whether the first
  // one holds a Draw Things checkpoint's own codes. What a test asserts
  // before it compares anything.
  bool i8_layout() const { return _i8_layout; }
  bool i8x_native() const
  {
    return !_blocks.empty() && _blocks[0].qkv.i8x && _blocks[0].fc2.i8x;
  }
  // The token tiers' GEMM kernel it took ("blocked 128x64", ...; empty
  // without int8).
  std::string i8_token_kernel() const
  {
    return _i8 ? _i8->token_kernel() : std::string();
  }

  // Bytes the ANE feed-forward tier holds for a `seq`-row forward of this
  // shape; see AneFeedForward::runtime_bytes. Static because the PLAN asks
  // before anything loads.
  static std::size_t ane_runtime_bytes(const Config& c, int seq) noexcept;
  // Rows per ANE predict: VPIPE_H3_ANE_CHUNK or the default.
  static int ane_chunk_rows() noexcept;
  // Whether the tier tried to arm, and whether it did: a plan that booked
  // the module can release the booking when it did not.
  bool ane_attempted() const noexcept { return _ane_tried; }
  bool ane_armed() const noexcept { return _ane != nullptr; }
  // Whether a tier LOST rows on some block and has stayed on the GPU
  // since (AneFeedForward::gpu_only): the rows were recomputed on
  // the GPU, and no later block split.
  bool ane_gpu_only() const noexcept
  {
    return _ane != nullptr && _ane->gpu_only();
  }
  bool ane_qkv_gpu_only() const noexcept
  {
    return _ane_qkv != nullptr && _ane_qkv->gpu_only();
  }
  // Take the ANE tiers off for this model before a forward that cannot
  // afford them: modules released, config off, and the VPIPE_H3_ANE_QKV
  // override ignored. The stage calls this when a clip fits only without
  // the ANE's memory, rather than refusing the clip.
  void disable_ane() noexcept
  {
    _cfg.ane_ffn  = false;
    _cfg.ane_qkv  = false;
    _ane_disabled = true;
    _ane.reset();
    _ane_qkv.reset();
    _ane_tried     = true;
    _ane_qkv_tried = true;
  }

  ~MetalMiniMaxH3Transformer();   // out-of-line: _ws is fwd-declared

  // Everything one denoiser evaluation needs that is not a weight. The
  // caller owns the layout and the row buffers; this is a view.
  struct Step {
    // Patchified video rows [num_video_rows, 96] bf16, in the order of
    // `layout->video_indices` -- conditioning rows FIRST.
    const metal_compute::SharedBuffer* video = nullptr;
    // Audio rows [num_audio_rows, audio_channels] bf16, channel-major.
    const metal_compute::SharedBuffer* audio = nullptr;
    // Text conditioning [num_text_rows, text_dim] bf16, straight from
    // the encoder tap.
    const metal_compute::SharedBuffer* text = nullptr;
    const minimax_h3::PackedLayout* layout = nullptr;
    // The DISTINCT timesteps present in the sequence, sorted, in [0, 1]
    // and unscaled, plus each row's index into them. Both come from
    // minimax_h3::build_row_timesteps.
    const std::vector<float>* timesteps         = nullptr;
    const std::vector<int>*   row_timestep_index = nullptr;
    // Each distinct timestep's ENDPOINT, parallel to `timesteps` --
    // required while a flow-map adapter is live (see FlowMap) and
    // ignored otherwise. From minimax_h3::build_row_time_pairs, which
    // makes the distinct entries PAIRS: two rows at the same t heading
    // to different r are different entries.
    const std::vector<float>* endpoints         = nullptr;
    // Which entry of a baked schedule this evaluation is, or -1 for
    // "not baked" (then `timesteps` drives the AdaLN projection as
    // before). See bake_adaln.
    int schedule_index = -1;
    // The GENERATED video's latent grid. Only the hybrid VDN path reads
    // it -- its window is over whole frames and its short conv is a
    // (t, h, w) stencil, neither of which can be recovered from a row
    // count. Left at zero the hybrid is off for this forward, which is
    // what every non-VDN caller wants and gets by saying nothing.
    int video_grid_h = 0;
    int video_grid_w = 0;
    // Run this forward's attention DENSE even when Sol-Attn is on. The
    // routing is per forward and so is this: a sampler that wants its
    // early, structure-deciding steps exact (HyperFlow's published Sol
    // recipe keeps the first two of eight dense) says so step by step.
    bool dense_attention = false;
    // This forward's Sol-Attn ROUTING in place of the config's: its tau
    // and threshold form, where `sol_routing` is set. Per forward like
    // dense_attention -- a sampler that routes its steps differently
    // says so step by step. Neither changes what Sol holds: the
    // threshold's buffers are the call's own dead scratch.
    bool  sol_routing = false;
    float sol_tau     = 1.0f;
    bool  sol_exact   = false;
    // Sage's int8 P*V (sage::Config::pv8) off for this forward -- the
    // int8 QK alone -- as the loop's sage_pv8_from_step says.
    bool sage_pv8_off = false;
    // TaoMate-H3's STREAMING attention (minimax-h3-stream-kv.h). Set, and
    // every main block's attention splits in two:
    //   text rows   -> the text rows alone
    //   other rows  -> [text ; the cache's committed clean K/V ; this
    //                   forward's media rows]
    // non-causally, over a layout of [text | audio | video] rows. Sol,
    // VDN, Sage and the fused-attention spellings are not used for such
    // a forward -- each assumes one square self-attention.
    minimax_h3::StreamKv* stream_kv = nullptr;
    // ...and this is the chunk's CLEAN forward: after each block's
    // attention has read the cache, the block's cache is rewritten as the
    // rows that survive this commit plus this forward's media rows. The
    // output heads do not run; the Velocity comes back ok and empty.
    bool stream_commit = false;
  };

  struct Velocity {
    // [num_video_rows, 96] and [num_audio_rows, audio_channels] bf16, in
    // the same row order the inputs were given in.
    metal_compute::SharedBuffer video, audio;
    // Success, stated rather than read off a buffer: an AUDIO-ONLY
    // forward (no video rows) has no video velocity, and a streaming
    // commit has neither.
    bool ok = false;
    bool empty() const { return !ok; }
  };

  // One denoiser evaluation. Empty on failure, with a reason in `err`.
  Velocity forward(const Step& in, std::string* err = nullptr);

  // Size the activation scratch for at least `seq` rows and `n_text`
  // text rows, and KEEP it for any forward that fits, rather than
  // rebuilding on every change of length. A streaming run alternates
  // four chunk lengths and a short audio-only teacher; rebuilt each time,
  // that is several GB reallocated (and both sets briefly live) several
  // times a request. Every buffer in the scratch is addressed with the
  // forward's OWN length, so a larger one is a correct one. 0 restores
  // the exact-fit policy every other caller relies on.
  //
  // `qkv_elems`: what the q|k|v scratch must hold for a stream forward's
  // K/V staging (StreamKv::staging_bytes / 2), named up front so the
  // first forward -- the audio-only teacher -- already builds it that
  // wide. Raised later, it would REBUILD the scratch, and a rebuild holds
  // both sets at once.
  void set_scratch_floor(int seq, int n_text, std::size_t qkv_elems = 0)
  {
    _scratch_floor_seq  = seq > 0 ? seq : 0;
    _scratch_floor_text = n_text > 0 ? n_text : 0;
    _qkv_floor_elems    = _scratch_floor_seq > 0 ? qkv_elems : 0;
  }

  // Attach VDN-H3's linear branch, turning every main block's attention
  // into the hybrid: a chunk-windowed softmax over the frames near the
  // query, plus a bidirectional delta rule carrying everything the
  // window cannot see. The two PARTITION the keys, so this is not an
  // approximation bolted onto full attention -- running one half alone
  // is wrong in a way that looks like a plausible video.
  //
  // `dir` is the VDN release root (the one holding `config.json` and
  // `linear_branch/`). The branch SHARES this model's q/k/v projections
  // -- that is why one LoRA on attn.orig.to_{q,k,v} feeds both halves --
  // so it brings no projections of its own, and attaching it does not
  // change what this checkpoint means.
  //
  // Off until this is called, and a forward that leaves Step's grid at
  // zero still runs the stock stack.
  bool attach_linear_branch(const std::string& dir, std::string* err);
  bool has_linear_branch() const { return (bool)_vdn; }

  // Non-zero if any of the branch's per-frame I + A was not positive
  // definite on the last forward. READ IT: the solve returns a NaN
  // there, and a NaN in the residual stream makes the frame it came
  // from unrecoverable from the output rather than merely wrong.
  unsigned linear_solve_failures() const
  {
    return _vdn ? _vdn->solve_failures() : 0u;
  }

  // How many blocks ran the LINEAR half on the last forward, and whether
  // the window reached every frame.
  //
  // Reported because the two are the same fact seen twice and getting
  // them out of step is silent: a window that covers the clip makes the
  // hybrid the stock model plus the gate, and a linear branch left
  // running under it contributes the text state to every video row --
  // no crash, no NaN, just a term the reference does not have.
  unsigned linear_blocks_run() const { return _vdn_blocks_run; }

  // How much of the dense key-block grid the window actually visits:
  // the block-sparse kernel's whole reason for existing, and the number
  // that says whether it is worth taking at a given geometry. Both zero
  // until a windowed forward has built the list.
  std::size_t window_key_blocks() const { return _vdn_kb_sparse; }
  std::size_t dense_key_blocks() const { return _vdn_kb_dense; }
  bool linear_window_covers_all() const { return _vdn_full_cover; }
  // What the forwards have encoded for perf-flops, credited to the GPU
  // as each block retires (generative-models/shared/dit-flops.h).
  const GpuFlopsTally& gpu_flops() const noexcept { return _flops; }

  // What the attached branch's blocks hold, plus the two per-forward
  // buffers the hybrid adds. Reported apart from resident_block_bytes()
  // because it is a SECOND checkpoint -- 4.28 GB over 50 blocks -- and a
  // stage sizing this model has to count it.
  std::size_t linear_branch_bytes() const
  {
    if (!_vdn) { return 0; }
    return _vdn->resident_bytes() + _vdn_ro.byte_size()
           + _vdn_proj.byte_size();
  }

  // Precompute every step's modulation and drop the projections that
  // produce it. `schedule` is one entry per step: that step's distinct
  // timesteps, exactly as build_row_timesteps returns them, and a Step
  // then names its entry through `schedule_index`.
  //
  // This is THE memory decision on this model, and it is a lopsided one.
  // The 50 per-block AdaLN projections are 12.9 GB of the 23.5 GB 4-bit
  // checkpoint -- 55% of it -- and they exist to multiply the handful of
  // DISTINCT timesteps a sequence carries (three, typically). Measured,
  // they are 1.3% of a step's GPU time and 55% of the bytes it streams.
  // So the arithmetic is not worth removing and the residency is.
  //
  // Baked, each block keeps a [total_rows, adaln_out] bf16 table instead
  // of a [adaln_out, time_dim] projection: 2.5 MB against 258 MB at six
  // steps. The weights are then released and, in streaming mode, never
  // re-read -- which is the whole point, since a streaming run pays for
  // them on EVERY step.
  //
  // Two things this does not do. It does not fold a runtime LoRA's adaln
  // adapter in: those factors are rank 16 and ~3 MB a block, so keeping
  // them applied per step costs almost nothing and keeps `lora_scale` a
  // knob rather than a rebuild. And it refuses rather than degrades when
  // the table would be too large -- the size grows with the step count,
  // and at some point holding it is worse than recomputing.
  //
  // Idempotent per schedule; calling it with a different schedule
  // rebuilds. Requires the adaln weights to still be reachable, so it
  // must run before anything releases them.
  //
  // `endpoints`, when given, is the same shape as `schedule` and holds
  // each entry's endpoint -- required under a live flow-map adapter,
  // since the table IS the modulation and the modulation is a function
  // of (t, r).
  bool bake_adaln(const std::vector<std::vector<float>>& schedule,
                  std::string* err = nullptr,
                  const std::vector<std::vector<float>>* endpoints = nullptr);
  bool adaln_baked() const { return !_adaln_tab.empty(); }

  // Bytes the AdaLN bake will RETIRE, read from the checkpoint index
  // without loading anything -- the sum of every `adaln_proj` tensor.
  //
  // Static and knowable before any run, which is the point: the block-
  // streaming decision is irreversible and is taken at construction,
  // while the bake happens moments later and removes 39% of a bf16
  // checkpoint (24.3 of 61.7 GB). Sizing that decision against weights
  // this model is about to stop holding is how a box with room ends up
  // streaming.
  //
  // Returns 0 when the checkpoint cannot be opened, so an unreadable
  // index is simply not counted rather than guessed at.
  static std::size_t adaln_retired_bytes(const std::string& dit_dir);

  // The LEAST this DiT holds when it streams its blocks: everything that
  // is not a block, plus the two in-flight slots.
  //
  // Read from the checkpoint, before anything loads, because that is
  // when the planning phase asks. The resident set grows ON TOP of this
  // as free memory allows and is shed when it does not, so this is a
  // floor and not a prediction -- which is exactly what a "will this
  // graph fit" question wants.
  static std::size_t streaming_floor_bytes(const std::string& dit_dir);

  // Will the bake certainly happen for a schedule of at most
  // `max_rows` rows? The bake REFUSES a schedule whose tables would
  // exceed its budget, and a refusal leaves every projection resident --
  // so this answers with the WORST case, and a caller may subtract
  // adaln_retired_bytes() only when it is true.
  static bool adaln_bake_certain(const Config& cfg, int max_rows);
  // What the tables cost, for the memory report. 0 when not baked.
  std::size_t adaln_table_bytes() const;

  const Config& config() const { return _cfg; }

  // The backend this model runs on, so a driver holding only the model
  // can allocate its own row buffers without the caller threading a
  // second handle alongside it.
  metal_compute::MetalCompute* metal_compute() const { return _mc; }

  // ---- adaptive block residency (streaming mode) -----------------------
  //
  // Streaming keeps ~one block resident, which on a 16 GB box leaves most
  // of the machine idle: the 4-bit DiT streams in ~4.5 GB of traffic per
  // forward while 10+ GB sits free. This grows a RESIDENT SET into that
  // slack so later steps re-read less from disk.
  //
  // WHY THIS IS NOT A CACHE. The access pattern is a repeating sequential
  // scan -- blocks 0..N-1, then the next step scans 0..N-1 again. For a
  // cyclic scan with capacity C < N, LRU has a ZERO percent hit rate: the
  // block you evict to make room is always the one you need next time
  // round. Recency is exactly the wrong signal here. What works for a
  // looping scan is a FIXED subset held permanently, which gives C/N.
  // So this grows a resident set and then leaves it alone; it never
  // evicts in order to admit, and there is no recency bookkeeping at all.
  //
  // Anti-thrash, in the order the failure modes appear:
  //   * admission only spends genuinely FREE headroom, never another
  //     block's seat;
  //   * it keeps `reserve_bytes` clear for the next forward's scratch --
  //     admitting into the scratch's room would just push the scratch out;
  //   * hysteresis: admit above the high watermark, evict below the low
  //     one, so a budget hovering at the line does not oscillate;
  //   * a RATCHET: after any eviction the growth ceiling drops to just
  //     under what was resident when the pressure hit, so the set cannot
  //     climb straight back into it.
  //
  // `set_residency_reserve` is how a caller tells this how much room the
  // rest of the forward needs -- scratch_bytes() plus whatever the graph
  // wants left over. 0 disables growth entirely (pure streaming).
  void set_residency_reserve(std::size_t bytes) { _resid.set_reserve(bytes); }

  // Bytes currently held by promoted blocks (the adapter factors kept
  // with them included), and how many there are. Reported so a run can
  // say how much of the box it actually used.
  std::size_t resident_block_bytes() const { return _resid.bytes(); }
  int resident_block_count() const { return _resid.count(); }
  // The streaming slot pair, while it exists.
  std::size_t slot_bytes() const
  {
    return block_bytes_(_slot[0]) + block_bytes_(_slot[1]);
  }
  // What a STREAMING model holds beyond its weight set's cached entries
  // (the trunk): the promoted blocks, the slot pair and the AdaLN tables
  // the bake built -- not the adapter factors kept with promoted blocks,
  // which the adapter file's own claim books. The figure a per-clip
  // holding correction adds to the set's -- promoted blocks are this
  // model's own buffers, which the set never sees. 0 for a preloaded
  // model, whose blocks are the set's.
  std::size_t streamed_holding_bytes() const;
  // EVERY BLOCK KEPT: give the slot pair back -- two blocks a preload
  // never held. Call with no forward running; the next streamed block
  // (after an eviction) builds it again. Returns the bytes freed.
  std::size_t release_idle_slots();

  // Give back at least `bytes` of promoted blocks, for a peer that needs
  // the room (the VAE decode is the case this exists for). Returns what
  // was freed, and ratchets the growth ceiling down so the set does not
  // immediately reclaim it. Never touches the configured pinned prefix.
  std::size_t release_resident_blocks(std::size_t bytes);
  // Free the highest resident block; its bytes, or 0 when there is
  // nothing left.
  //
  std::size_t evict_tail_block_();


  // Pages of the resident set examined, and how many are still in RAM.
  void resident_pages_(std::size_t* examined, std::size_t* incore,
                       std::size_t* paged_out = nullptr) const;

  // Cooperative stop, polled once per BLOCK inside forward().
  //
  // The denoise loop already takes a per-STEP callback, but a step here is
  // one forward of a 50-block 33B stack over a ~9k-row packed sequence --
  // tens of seconds. Honouring a stop only between steps means a user who
  // presses stop waits about a minute for it. Polled per block it lands in
  // roughly one block, which is what "responsive" has to mean at this
  // scale. Set it around a generation and clear it afterwards, as the
  // image DiTs do; an empty function disables the check.
  void set_stream_stop(std::function<bool()> stop)
  {
    _stream_stop = std::move(stop);
  }

  // Bytes of per-forward SCRATCH this geometry needs, BEFORE anything is
  // allocated -- so a stage can decide whether the machine has room
  // instead of finding out by thrashing. At video sequence lengths this
  // is the model's largest live allocation (~200 KB per row: ~1.9 GB at
  // the 9382-row production layout, ~3.9 GB at 19k), which is why the
  // video stages preflight it and the image ones never needed to.
  //
  // `with_dequant` adds the matrix-core dequant scratch (the widest
  // projection, 2*ffn x hidden), which gemm_mma_ allocates lazily on a
  // quantized checkpoint with matrix cores. The caller knows whether that
  // applies before a model exists; uses_matrix_cores() answers it after.
  // `narrow_ff` drops the [seq, 2*ffn] fc1 intermediate to the width the
  // fused-SwiGLU path leaves it at. It defaults to the WIDE figure so a
  // caller that cannot know over-estimates: refusing a forward that would
  // have fit costs a smaller canvas, and under-estimating one that does
  // not costs the machine. ff_scratch_narrow() answers it once a model
  // exists.
  // `arena_floor` raises the attention arena to at least that many
  // bytes, which is how the VDN branch's scratch is paid for: it does
  // not get an allocation of its own, it is carved out of the arena
  // (see vdn_scratch_bytes and MetalVdnBranch::set_arena), so the
  // branch costs this model the amount by which its need exceeds an
  // arena the attention was going to hold anyway -- nothing at all at
  // both geometries measured.
  // `widen` floors the dequant scratch at what an FP8 / int8 / W4A8
  // checkpoint widens into it per GEMM (fp8_widen_bytes), on EVERY GPU
  // and including projections the four above do not cover (a released
  // AdaLN, 2,688 wide, at the bake).
  static std::size_t scratch_bytes(const Config& c, int seq, int n_text,
                                   int n_t, bool with_dequant,
                                   bool narrow_ff = false,
                                   std::size_t arena_floor = 0,
                                   int band = 0, std::size_t widen = 0);

  // The widest projection an FP8 / int8 / W4A8 checkpoint widens to bf16
  // ahead of its GEMM (into the dequant scratch), N x K x 2 -- the fused
  // q|k|v of a file that stores it split included; 0 for a checkpoint
  // with none. Off the headers (`dit_dir` as config_from_json takes it),
  // or, for a loaded model, off what it read.
  static std::size_t fp8_widen_bytes(const std::string& dit_dir,
                                     const Config& c);
  std::size_t fp8_widen_bytes() const { return _f8_widen; }

  // THE BANDED FORWARD. A long sequence's main blocks run a BAND of rows
  // at a time: every row's residual and q / k / v are kept -- attention
  // needs every key -- and one band of everything else (the normed rows,
  // the qkv projection's output, the attention output, the feed-
  // forward). 54 KB a row against 137 whole: at 2K x 121 frames (134k
  // rows) 7.7 GiB of activations against 17. Each row's arithmetic is
  // the whole forward's -- Sol, Sage and Sage's int8 P*V included: Sage
  // quantizes K and V once a layer and each band's Q, Sol attends each
  // band's queries over every key (a second Sol object for the last
  // band). The rows a band takes for `seq` under `c` -- BALANCED: every
  // band but the last that many, at most the configured band, the last
  // long enough to clear every route's row floor -- or 0 for the whole
  // sequence: under 2 bands' worth, and wherever a path needs the whole
  // sequence's buffers -- the ANE tiers (and, at the forward, streaming
  // attention, VDN's branch, the in-place q/k/v layout).
  // VPIPE_H3_BAND_ROWS=<rows> sets the band (0: never).
  static int band_rows(const Config& c, int seq);
  static constexpr int kBandRows = 8192;
  // The band the last forward ran its main blocks in (0: whole).
  int last_band_rows() const { return _last_band; }
  // ...and the one this model's forward takes at `seq` (band_rows under
  // the load's kernels and branches; a request's streaming attention
  // aside).
  int forward_band(int seq) const;

  // What the per-forward scratch buffers ACTUALLY hold right now, and
  // what the matrix-core dequant scratch holds, kept apart because they
  // are sized by different things -- the first by the geometry, the
  // second by the widest projection in the checkpoint. Both zero before
  // the first forward. They exist so scratch_bytes() can be CHECKED
  // against reality rather than trusted; see the test named on its
  // definition.
  std::size_t scratch_resident_bytes() const;
  std::size_t dequant_scratch_bytes() const;

  // What the VDN branch will ALSO want, for this geometry, or 0 when no
  // branch is attached.
  //
  // Kept apart from scratch_bytes() above because it is a SECOND
  // allocation with a different shape: that one is sized by the packed
  // sequence, this one mostly by [frames, heads, d, d] banks that grow
  // with the clip -- 1.0 GB at the docs pipeline's 37 latent frames and
  // 2.2 GB at 92. It is spent at the branch's first reserve(), which is
  // inside the first forward and therefore AFTER every planning
  // decision, so a caller that preflights the transformer's scratch and
  // not this one sizes the box against half of what it is about to use.
  // That was the gap: the DiT would keep blocks resident against a
  // budget that had not heard of the branch, and the machine swapped.
  //
  // SPLIT THE WAY THE MEMORY ACTUALLY FALLS OUT, which is why there is
  // an out-parameter. `*arena_floor` is the part the ATTENTION ARENA
  // absorbs -- the branch runs in the gap between the previous block's
  // FF and this block's attention, when nothing in that arena is live,
  // so it carves its scratch there and the arena is merely sized to the
  // wider of its two users. The RETURN VALUE is what is left over: the
  // branch's two host-written constants, plus the readout, which is the
  // one buffer that cannot share because it spans the attention it is
  // waiting to be added to.
  //
  // Pass the floor to scratch_bytes() / the forward and add the return
  // value on top; the sum is the same shape of answer as before, and
  // smaller by however much the arena covers.
  //
  // `grid_h`/`grid_w` are the PATCH grid, the same numbers Step carries.
  std::size_t vdn_scratch_bytes(const minimax_h3::PackedLayout& L,
                                int grid_h, int grid_w, int n_text,
                                std::size_t* arena_floor = nullptr) const;

  // What SOL-ATTN will also want at this sequence length, or 0 when it
  // is off. The same shape of question as vdn_scratch_bytes above and
  // for the same reason: it is a second allocation, spent at the first
  // routed block and therefore after every planning decision.
  //
  // MOST OF IT IS NOT AN ALLOCATION AT ALL. Sol's scratch lives for one
  // call, and with it on the fused qkv projection and the arena's last
  // attention window are both dead for exactly that stretch -- so what
  // this returns is what is LEFT after that loan, which at production
  // geometry is a few hundred bytes against 861 MB unlent. It is the
  // real carve and not a subtraction; see
  // MetalSolAttention::private_bytes.
  std::size_t sol_scratch_bytes(int seq) const;
  // The share of key blocks Sol-Attn's routing kept exact over the LAST
  // forward's routed layers: valid once forward() has returned, until
  // the next one starts (which reports and resets it). -1 where nothing
  // was routed.
  double sol_kept_fraction() const;
  // What the Sol objects hold of their own -- pinned, unlent scratch and
  // the prepared keys -- which is what sol_scratch_bytes() books.
  std::size_t sol_held_bytes() const;

  // ...and SageAttention's, which a preflight needs for the same reason
  // and did not have: its int8 twins of q and k are H * seq * head_dim
  // BYTES each, which at video geometry is a gigabyte and a half that
  // nothing sizing the box had heard of.
  std::size_t sage_scratch_bytes(int seq) const;

  // ...and the INT8 CONTEXT's, the same shape of question again: the
  // activation codes and scales of the widest GEMM at this sequence, the
  // per-call weight scratch (the accurate tier requantizes every weight
  // per call; a w4 pack on fast / token converts into it) and split-K's
  // planes -- grown inside the first forward and kept. A bound
  // (I8GemmContext::forward_bytes_bound), 0 when the context is off.
  std::size_t i8_scratch_bytes(int seq) const;

  // How many times the activation scratch has been ALLOCATED.
  //
  // One per geometry is right; more than one within a denoise is the
  // bug this counts: the set is built in full before the old one is
  // dropped, so a rebuild peaks at both at once -- 13 GB twice at video
  // geometry. See kTimestepSlots.
  // Timestep slots the activation scratch is SIZED for, whatever a
  // single forward presents. `n_t` moves within a denoise (1 on the
  // first step, 2 once the video and audio schedules diverge) and three
  // small buffers depend on it, so keying the scratch on the exact
  // value rebuilt every activation buffer in the model to grow a few
  // hundred KB. Must not exceed GenerateVideoStage's
  // kTimestepsUpperBound, which is what the preflight sizes the box
  // against.
  static constexpr int kTimestepSlots = 4;

  int scratch_rebuilds() const { return _scratch_rebuilds; }

  // Which kernel one block projection dispatches on.
  //
  // The steel arms are affine_qmm_steel tile heights, and run everywhere.
  // The mma arms dequantize the weight ONCE into a scratch and then run a
  // dense matmul2d tile on it -- the M5 matrix-core path, so they exist
  // only where supports_matrix_cores() holds. A dense (unquantized)
  // checkpoint skips the dequant and takes the same tiles.
  //
  // This is ONE list rather than a steel knob plus an mma knob because
  // the choice between them is not structural: at a short sequence the
  // dequant is not amortized and steel wins, at a long one matmul2d wins
  // by 2-3x, and where the crossover sits depends on (M, N, K) and on the
  // machine. Making it one enum is what lets the autotune below answer
  // the whole question by measurement instead of by threshold.
  enum class GemmRoute {
    kAuto = 0,        // tuned, or the fallback rule when tuning is off
    kSteelBm32,
    kSteelBm64,
    kSteelBm128,
    kMma128,          // dense_gemm_mma_t_n128_f16      (128x128)
    kMma128x256,      // dense_gemm_mma_t_n128x256_f16  (128x256)
    kMma128x256Tn2,   // dense_gemm_mma_t_n128x256_tn2_f16 (128x512 region)
  };

  // Cap on the quantized-GEMM tile height: 0 = BM32 only, 1 = +BM64,
  // 2 = +BM128. Same knob, and the same reason for it, as the Wan DiT:
  // the only way to compare two tiles without the thermal spread between
  // processes is to alternate them inside one.
  int  quant_bits() const { return _quant_bits; }
  int  quant_group() const { return _quant_group; }

  void set_qmm_tile(int cap);
  int qmm_tile() const { return _qmm_tile; }

  // Force every block projection onto one route, for the same reason
  // set_qmm_tile exists and with the same caveat: a cross-process A/B
  // cannot resolve two kernels on M5, where the GPU clock is gated by a
  // time-integrated SoC power budget and the run-to-run spread is wider
  // than the effect being measured. The arms have to alternate inside one
  // loaded model -- and at 33B there is no room for a second one anyway.
  //
  // Forcing a route DISABLES the autotune, exactly as set_qmm_tile does:
  // a tuned answer that overrode the forced route would run one kernel in
  // every arm while appearing to test several. kAuto restores tuning.
  // A route whose kernels never loaded (mma on M4) falls back rather than
  // failing, so a bench can ask for it unconditionally.
  void set_gemm_route(GemmRoute r);
  GemmRoute gemm_route() const { return _forced_route; }

  // Whether the M5 matrix-core GEMM path is available at all -- i.e. the
  // dequant + dense matmul2d kernels loaded. A path-SELECTION accessor,
  // like the Qwen fast-path guard: a silent fallback to steel is 2-3x
  // slower but numerically fine, so no correctness test can see it.
  bool uses_matrix_cores() const { return _use_mma2; }
  // Whether the dequant scratch (_w_deq) exists at all: the matrix-core
  // path widens quantized weights into it, and an FP8 checkpoint widens
  // every weight into it on EVERY machine. What scratch_bytes()'s
  // `with_dequant` has to be told.
  bool uses_dequant_scratch() const { return _use_mma2 || _f8.any(); }
  // True when every block runs the fused-SwiGLU FF, so the ff scratch is
  // the narrow one. Pair with scratch_bytes()'s `narrow_ff`.
  bool ff_scratch_narrow() const { return !_ff_needs_wide; }
  // Whether the DiT's flash attention is the matrix-core (NAX) kernel.
  bool uses_nax_attention() const { return _attn_nax; }
  // What the streamed-weight prefetch did on the last forward: reads
  // ISSUED and reads a block actually CONSUMED. A gap between them is
  // work paid for and discarded -- and on the fallback destination it
  // also leaves a block's worth of buffers live for the rest of the
  // pass, which is the opposite of what a memory-tight run wants. Zero
  // when nothing streamed.
  int prefetch_started() const { return _pf_started; }
  int prefetch_hits() const { return _pf_hit; }
  // Whether the last streamed forward ran with TWO reusable block slots.
  // False means single-slot (the budget refused the second, or
  // VPIPE_H3_SINGLE_SLOT asked for it) and is a distinct third path from
  // slots-off.
  bool slot_pair() const { return _slot_pair; }

  // The row count a tuned GEMM answer is filed under.
  //
  // The tuner measures at min(M, ceiling) rows, so every geometry at or
  // above the ceiling produces an IDENTICAL measurement -- same rows,
  // same shapes, same candidates. Keying the cache on the caller's own
  // row count therefore made a second video geometry pay again for an
  // answer that could not differ, which at the production shapes was
  // ~5 s of GPU per distinct sequence length. Below the ceiling the row
  // count really is the caller's and the key is exact.
  //
  // Public because it is the property worth pinning: if the store and
  // the lookup ever disagreed about this, every projection would fall
  // silently back to the untuned rule and only a timing would show it.
  int tune_row_ceiling() const { return _tune_ceiling; }
  int tune_row_key(int M) const { return std::min(M, _tune_ceiling); }
  // How many times the tuner has actually MEASURED, as against found its
  // answer already cached. Test seam for the line above.
  int qmm_tune_count() const { return _qmm_tune_count; }
  // Which candidate indices survive the GEMM tuner's warm pass, given
  // each one's warm time and how far behind the leader an arm may be and
  // still be timed properly.
  //
  // Pure, and public, because it is a RULE rather than a measurement and
  // the rule is what can silently go wrong: the two fastest must always
  // survive, the leader must never be dropped, and a warm pass that
  // failed to time must prune nothing. None of those is visible in a
  // timing. See the note in tune_qmm_ for why the threshold is loose.
  static std::vector<int> tune_prune_survivors(
      const std::vector<double>& warm, double keep);
  // Which of attention's four transposes are gone. The two halves are
  // INDEPENDENT and, measured, they are not the same trade -- see the
  // note on _fused_attn -- so they are separate bits rather than one
  // switch.
  //
  // A path-SELECTION accessor: every combination is numerically
  // identical, so only a timing can tell them apart. The setter is for
  // the in-process A/B; bits the build cannot honour are refused, and
  // fused_attention() then reports what actually runs.
  static constexpr int kFusedAttnOut = 1;   // store O row-major
  static constexpr int kFusedAttnQkv = 2;   // read q/k/v in place
  int fused_attention() const { return _fused_attn; }
  void set_fused_attention(int mask);

  // What the route autotune MEASURED for the sequence length last run,
  // e.g. "9382: qkv=mma128x256 o=bm64 fc1=mma128x256tn2 fc2=bm32". Empty
  // before the first forward, or when tuning is off. Reported so a perf
  // run can say which kernels produced its number -- a timing with no
  // record of what ran is not reproducible.
  std::string qmm_tuning() const;

  void set_block_progress(DitBlockProgressFn fn)
  {
    _block_progress = std::move(fn);
  }

 private:
  // Defined with the runtime-LoRA members below; declared here because
  // gemm_ and its dispatchers take one by pointer.
  // The shared factor pair (generative-models/shared/runtime-lora.h):
  // A [rank, K] and B [N, rank], both bf16 and both in the checkpoint's
  // own orientation. An ALIAS rather than a second definition -- the
  // reader of the adapter file and the forward that applies it have to
  // agree about the layout, and two structs that merely look alike agree
  // only until one of them moves.
  using LoraFactors = lora::Factors;

  // The adapters attached to ONE projection, in slot order. Shared with
  // every other family that takes one (generative-models/shared/
  // runtime-lora.h) -- an ALIAS rather than a second definition, for the
  // same reason LoraFactors is one.
  //
  // Built per call rather than cached per block: it is three words a
  // slot on the stack, and the alternative is five of these per block
  // holding a strength that a config beat can change between forwards.
  using LoraStack = lora::Stack;
  static_assert(kMaxLoraSlots <= lora::Stack::kMax,
                "a slot the shared Stack cannot hold would be dropped "
                "silently at the projection rather than at load");

  MetalMiniMaxH3Transformer() = default;

  // A Linear: either a dense bf16 [N, K] matrix or an MLX-affine
  // quantized triple, plus an optional bias. Which one is detected from
  // the presence of a `<name>.scales` sibling, so one checkpoint can be
  // dense and another quantized with no config to keep in sync.
  struct Linear {
    metal_compute::SharedBuffer w, b;
    metal_compute::SharedBuffer codes, scales, qbias;
    bool quantized = false;
    int  bits = 0;
    // fc1 only: the gate and up rows have been interleaved into the
    // [g0,u0,g1,u1,...] pairing `affine_qmm_swiglu`'s epilogue reads out
    // of one accumulator fragment. A block whose fc1 is NOT interleaved
    // takes the split path, so the two can coexist within one forward --
    // which is what a streamed block that has not been promoted yet is.
    bool gu_inter = false;
    // FP8 storage: `codes` holds the checkpoint's own [N, K] bytes,
    // widened to bf16 on the GPU right before each GEMM (fp8_dense_).
    // Deliberately NOT `quantized`: there are no group scales, and every
    // path that reads an affine triple must not be handed one. A SCALED
    // weight carries `f8_scale`, f32 -- one value, or one per row when
    // `f8_per_row` (shared/fp8-expand.h). ComfyUI's INT8 rides here too
    // (f8 kI8), `f8_rot` its ConvRot group, undone in the widening.
    fp8::Format f8 = fp8::Format::kNone;
    metal_compute::SharedBuffer f8_scale;
    bool f8_per_row = false;
    int f8_rot = 0;
    // W4A8 (f8 kW4A8): `codes` the [N, K / 2] nibbles, `f8_scale` the
    // row scale, these the E4M3 relative scales [N, K / f8_qgroup] and
    // the codebook (16 f32; empty: code - 8).
    metal_compute::SharedBuffer f8_srel, f8_cb;
    int f8_qgroup = 0;
    // I8GemmContext's int8 LAYOUT (Config::i8_tier) when gc > 0: `codes`
    // int8 [n, k] and `cscales` its scales at group gc (kTokenGroup: f32
    // [n]) -- or, `pc`, a w4 pack's codes / scales / qbias as stored,
    // converted per projection (gemm_). Deliberately NOT `quantized`,
    // which means the affine pack every other route reads.
    // `i8x`: Draw Things' own codes, refilled raw by name in a slot.
    int gc = 0;
    int n = 0, k = 0;
    metal_compute::SharedBuffer cscales;
    bool i8x = false;
    // ComfyUI's int8 CONVROT codes run as stored on the token tier (gc ==
    // kTokenGroup): `cr` their group (256), the activation rotated alike
    // (I8GemmContext::quantize_token_convrot); `cscales` the row scale.
    // `cr_w4`: a W4A8 weight's nibbles (`codes`, with f8_srel / f8_cb /
    // f8_qgroup), decoded to those int8 codes per GEMM.
    int cr = 0;
    bool cr_w4 = false;
    // The w4 pack kept as stored and converted per projection (gemm_);
    // false for a layout held (Config::i8_hold_w4), w4 or w8.
    bool pc = false;
    // A STREAMED block's layout, rebuilt after every refill out of what
    // the slot refills (refill_block_): `ds`, a dense weight -- `w` keeps
    // its bf16, quantized into codes / cscales; `ps`, a w8 pack -- its
    // codes / scales / qbias are refilled and the codes converted IN
    // PLACE. A promotion keeps the layout alone (keep_block_), as the
    // entry a preloaded block holds.
    bool ds = false;
    bool ps = false;
    bool is_fp8() const { return f8 != fp8::Format::kNone; }
    bool empty() const {
      return (quantized || is_fp8() || gc > 0) ? codes.empty() : w.empty();
    }
  };

  // Put `nm`'s tensors into `dst`, reusing its buffers when they already
  // fit and building them when they do not. Only the AdaLN bake wants
  // this: it is the one reader whose tensors are all the same shape.
  bool adaln_into_(const std::string& nm, Linear& dst);

  // The per-run startup report in forward() fires once. Everything
  // it measures -- scratch, wiring, the GEMM autotune -- is paid on
  // the first forward and never again.
  bool _prologue_logged = false;

  struct Block {
    metal_compute::SharedBuffer n1, n2;   // RMSNorm gammas
    Linear qkv, out;
    metal_compute::SharedBuffer qn, kn;   // per-head q/k RMS gammas [128]
    Linear fc1, fc2;
    Linear adaln;                         // only on the 50 main blocks
  };

  // Cached entries are the ones the model KEEPS; a streamed block is read
  // per forward and dropped, so caching it would silently undo streaming
  // and put the whole stack back in RAM.
  enum class Retain { Cached, Streamed };

  // Wire (mlock) or unwire every buffer of a resident block, returning
  // the bytes whose state actually changed.
  //
  // A kept block is written ONCE and thereafter read only by the GPU, so
  // it accrues no CPU reference bits and is the coldest anonymous memory
  // in the process. MEASURED on the M4 Pro 64 GB box: a 5.9 GB resident
  // set read 100% in-core at every admission and 0% -- paged_out on
  // every sampled page, 9.9 GB of self_compressed -- one forward later,
  // while 46 GB of clean file cache went untrimmed. The compressor
  // harvests cold anonymous pages opportunistically; being cold is the
  // whole of the problem, and wiring is the statement that these pages
  // are not spare.
  //
  // Unwiring is safe on a live buffer: set_wired(false) munlocks and
  // restores NonVolatile, so nothing becomes discardable. (It used to
  // flip the buffer to PURGEABLE VOLATILE, which made unwiring correct
  // only on the way to destruction -- that was a crash on mapped
  // subviews and is fixed in SharedBuffer.) What it still costs is the
  // protection itself, which the next allocation may not get back.
  std::size_t wire_block_(Block& b, bool on);

  // What ONE resident block weighs, from the checkpoint rather than from
  // a Block that may not exist yet -- the residency probe is sized
  // before the first block is admitted.
  //
  // AdaLN is excluded because the bake retires it: a resident block is
  // the post-bake shape, and counting the projections would make a block
  // read 1.6x its real cost and the probe correspondingly timid.
  std::size_t resident_block_bytes_() const;

  // Does the checkpoint still describe the Linear this SLOT was built
  // for? Sizes are the refill's own business -- it compares against the
  // destination and reports kFailed -- but quantized-ness and bit width
  // are not, and they are per-LINEAR facts a mixed 4/8 pack varies from
  // layer to layer.
  //
  // Getting this wrong is silent. A slot built quantized whose block is
  // dense would refill the dense weight into the codes buffer and still
  // dispatch the quantized kernel over it, which produces numbers rather
  // than an error.
  bool linear_matches_(WeightSet& ws, const std::string& nm,
                       const Linear& l) const;

  // ---- the streamed block's reusable destinations --------------------
  //
  // Two Blocks, allocated ONCE and refilled in place for the whole run,
  // the way h3.c's two BF16 slots work. What they replace is a block's
  // worth of SharedBuffers allocated and freed on every block of every
  // forward -- 16 tensors x 50 blocks x every step -- which cost both the
  // allocation itself and, far more, the demand-faulted memcpy out of the
  // shard mapping that filled them.
  //
  // Two and not more: the depth that matters is one outstanding read,
  // which is what the prefetch already issues. A deeper ring would buy
  // nothing here (the GPU still takes the blocks strictly in order) and
  // would cost a block of live memory per extra slot on a box that is
  // streaming precisely because it has none to spare.
  //
  // These do NOT change what is resident. A promoted block is COPIED out
  // of its slot (see clone_block_), because the slot has to stay put for
  // the next read -- so the residency set, its growth and its ratchet all
  // work exactly as before.
  Block _slot[2];
  // Which slot holds the block the forward is currently running. The
  // prefetch always targets the other one.
  int   _slot_cur = 0;
  // Whether the slots are allocated, and for WHICH tensor set: `baked`
  // decides whether a block carries its AdaLN projection, and a slot
  // allocated without it cannot be refilled with it. Re-allocated rather
  // than assumed when the answer changes.
  bool  _slots_ready = false;
  bool  _slots_adaln = false;
  // Whether the SECOND slot exists. Refused when the box will not take a
  // durable block-sized allocation, and the run is then single-slot: no
  // prefetch, but the read shape and the absence of per-block allocation
  // -- the larger half of the win -- are unaffected.
  bool  _slot_pair   = false;
  // Set when a refill could not serve (a dtype no raw read can place, a
  // size that disagrees with the checkpoint) and the forward fell back to
  // allocating per block. Sticky for the run and logged once: a fallback
  // that flapped would be a performance mystery rather than a fact.
  bool  _slots_off = false;
  // Whether the env kill-switch still has to be read. Once, on the first
  // streamed block, rather than per block.
  bool  _slots_first = true;

  // Allocate `dst` with the same shapes and flags as `src`, optionally
  // copying the bytes. Both uses are one function because a promotion and
  // a second slot differ only in whether the contents come along.
  bool clone_block_(const Block& src, Block& dst, bool copy_bytes) const;

  // Refill an already-allocated block IN PLACE from the checkpoint.
  // Returns false without having produced a usable block when any tensor
  // cannot be served by a raw read -- the caller must then rebuild it
  // with load_block_, not patch this one up.
  bool refill_block_(WeightSet& ws, const std::string& prefix, Block& b,
                     bool with_adaln);

  metal_compute::SharedBuffer weight_(WeightSet& ws, const std::string& nm,
                                      Retain r);
  Linear linear_(WeightSet& ws, const std::string& nm, bool bias, Retain r);
  // A main block's projection in I8GemmContext's int8 layout (see
  // Config::i8_tier), or an EMPTY Linear where it cannot be -- the caller
  // then takes linear_(). Native for a Draw Things i8x weight; converted
  // once into a derived entry otherwise, and then for a kept block only.
  Linear i8_linear_(WeightSet& ws, const std::string& nm, Retain r);
  // A layout weight widened into the dequant scratch for a shape the int8
  // GEMM declines, from its layout operands (`l`'s own, or the scratch a
  // w4 pack was converted into); `view` aliases the scratch, as
  // fp8_dense_'s does.
  const Linear* i8_dense_(metal_compute::ComputeEncoder& enc,
                          const Linear& l,
                          const metal_compute::SharedBuffer& codes,
                          const metal_compute::SharedBuffer& cscales,
                          Linear& view, int N, int K);
  // A linear the checkpoint stores in PIECES (the diffusers layout's
  // split qkv and swapped fc1 -- minimax-h3-diffusers-layout.h), built
  // whole: FP8 codes when every piece is FP8 of one format, else bf16.
  Linear assemble_(WeightSet& ws, const std::string& nm,
                   const std::vector<minimax_h3::Part>& parts, bool bias,
                   Retain r);
  // Put `nm` into the Linear `l` already holds, in place, when its
  // buffers fit -- FP8 and assembled linears; quantized and plain dense
  // ones keep refill_block_'s per-tensor path. False when any piece
  // cannot be placed, and the caller rebuilds the Linear.
  bool refill_linear_(WeightSet& ws, const std::string& nm, Linear& l);
  // The weight `l` holds as a dense bf16 operand for this GEMM: `l`
  // itself, or -- for FP8 -- a `view` aliasing _w_deq, which the
  // expansion has just been encoded into. Null (and _fp8_failed latched)
  // when an FP8 weight cannot be expanded.
  const Linear* fp8_dense_(metal_compute::ComputeEncoder& enc,
                           const Linear& l, Linear& view, int N, int K);
  // `main`: one of the 50 main blocks (the int8 layout may apply; the
  // refiner's text rows sit under its row floor). `with_adaln`: read the
  // AdaLN projection -- false for a refiner block and for a main block
  // once the bake has replaced it. Two questions: a baked main block is
  // still a main block.
  bool load_block_(WeightSet& ws, const std::string& prefix, Block& b,
                   bool main, bool with_adaln, Retain r);
  // A streamed layout weight's conversion (Linear::ds / ps), encoded on
  // `e`; true with nothing encoded for any other weight.
  bool encode_fast_(metal_compute::ComputeEncoder& e, Linear& l) const;
  // What a PROMOTION of `b` keeps: no dense source, no dead pack scales,
  // a w4 pack whose layout is held (Config::i8_hold_w4) at the layout's
  // size. Residency and the wired pool are asked for this.
  std::size_t kept_block_bytes_(const Block& b) const;
  // ...and makes the promoted block hold exactly that, whether it was
  // cloned out of a slot or moved in from the per-block fallback.
  // Idempotent.
  bool keep_block_(Block& b) const;
  // The refill's conversions, on a stream of the reader's (a fresh queue's
  // first commit costs 1-2 ms; four projections a block added up).
  mutable metal_compute::CommandStream _cvt_stream;

  // Offsets are in ELEMENTS, not bytes, and exist because every
  // modality writes into its own contiguous slice of the one packed
  // sequence -- the reference's index_copy scatters are destination
  // offsets here.
  // `lora`, when given, is every adapter attached to this projection.
  // Each one's first GEMM is encoded BEFORE the base one (the fused tile
  // consumes its output); ONE of their second GEMMs is folded into the
  // base tile where the route allows, and the rest are encoded
  // separately. The base weight is read once either way -- what a second
  // adapter costs is its own pair of skinny GEMMs, ~1.5% of the
  // projection at rank 64.
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t x_off,
             const Linear& l, const metal_compute::SharedBuffer& y,
             std::size_t y_off, int M, int N, int K,
             const LoraStack* lora = nullptr);

  // The [seq, rot/2] f32 cos/sin tables for a packed layout's (t, h, w)
  // grid. Rebuilt whenever the layout changes, which in a denoise loop
  // is once.
  void build_rope_(const minimax_h3::PackedLayout& L,
                   metal_compute::SharedBuffer& cos_out,
                   metal_compute::SharedBuffer& sin_out) const;

  // The timestep MLP, on the HOST in f32. Tiny (one row per distinct
  // timestep) and singled out by the reference as the one place a
  // precision drop biases every block identically at every step, which
  // then accumulates coherently along the trajectory instead of
  // averaging out. Returns silu(temb) already cast to bf16, which is the
  // dtype the AdaLN projections consume.
  // `endpoints` as in Step: consulted only while two_time_active(), and
  // then required.
  bool time_embed_(const std::vector<float>& timesteps,
                   const std::vector<float>* endpoints,
                   metal_compute::SharedBuffer& out) const;
  // One row of the MLP: sinusoid(t) -> proj_in -> silu -> proj_out, with
  // every live slot's host adapter on each projection. `endpoint` picks
  // the ENDPOINT embedder's adapters (the base weights are shared);
  // `base_only` takes none (the pruned curve's basis is fit to the base).
  void time_mlp_row_(double t, bool endpoint, float* out,
                     bool base_only = false) const;
  // The released MLP's embedding before the silu, every live adapter on
  // it and a flow map's blend: time_embedding() for a released DiT, and
  // what a pruned one projects (and bakes the rest of) when an adapter
  // moves it.
  bool mlp_embedding_(const std::vector<float>& timesteps,
                      const std::vector<float>* endpoints,
                      std::vector<float>* out) const;

  struct Scratch {
    int seq = -1, n_text = -1, n_t = -1;
    metal_compute::SharedBuffer rcos, rsin;
    metal_compute::SharedBuffer x, nm, qkv;
    // ONE allocation backing qh|kh|vh|oh|ob, with `ff` aliased over the
    // whole of it. The five attention buffers are dead by the time the
    // FF runs -- attention has finished and `ob` has been folded into
    // `nm` -- so the widest of the two uses is what has to be held, not
    // their sum. At video sequence lengths that is the difference
    // between 7.9 GB of scratch and 5.7 GB.
    //
    // `ff` cannot alias `qkv` instead, which is the obvious-looking
    // choice: the UNFUSED SwiGLU reads s.ff and writes s.qkv, so those
    // two are live at the same instant. (The fused path writes straight
    // into s.qkv and never touches s.ff at all -- that reuse is already
    // here, and it is why streaming, which forces the unfused path, is
    // what makes this scratch its widest.)
    metal_compute::SharedBuffer attn;
    metal_compute::SharedBuffer qh, kh, vh, oh, ob, ff, proj;  // in `attn`
    metal_compute::SharedBuffer txt, temb, mod, fmod;
    metal_compute::SharedBuffer adaln_idx, tstep_idx;
    metal_compute::SharedBuffer lora;   // [seq, max rank], when attached
    // The banded layout's band of queries gathered head-major for Sol
    // ([H, band, D], beside ob and oh); empty in the whole layout.
    metal_compute::SharedBuffer qb;
    // > 0: the BANDED layout (band_rows) -- x, the rope tables, the
    // timestep indices and qh | kh | vh are the whole sequence's; nm,
    // qkv, lora and ob | ff are one band's; proj (the final layer's, when
    // the blocks are done) lies over qh | kh.
    int band = 0;
  };

  // Choose the attention query tile for this model, ONCE, on the first
  // forward long enough for the wide tile to matter and dense enough to
  // use it (no spans / sage / Sol: those build block lists and int8
  // scales for one tile size). Times both tiles over a truncated window
  // -- the ranking is stable above the crossover, and probing at a real
  // 100k-row sequence would itself cost ~25 s.
  void tune_attn_tile_(int seq, bool dense);   // arms the probe; see above

  bool ensure_scratch_(int seq, int n_text, int n_t,
                       std::size_t arena_floor, int band = 0);
  Scratch _s;
  int     _scratch_rebuilds = 0;

  // ---- VDN-H3's hybrid attention -------------------------------------
  //
  // Null unless attach_linear_branch() succeeded. The branch owns its
  // own weight set: the linear half is a separate checkpoint from the
  // DiT and the manager has to see it as one.
  std::unique_ptr<minimax_h3::MetalVdnBranch> _vdn;
  minimax_h3::vdn::Config                     _vdn_cfg;
  // The readout, bf16 [video_rows, inner], and its projection into the
  // residual stream, bf16 [video_rows, hidden]. Both are per forward
  // and reused across the 50 blocks.
  //
  // `_vdn_proj` IS A WINDOW INTO THE ATTENTION ARENA, over `ob` -- the
  // attention output, which the out projection has already consumed by
  // the time the branch's own projection is encoded, and which `proj`
  // aliases for the final layer on the same argument. It fits by
  // construction: video rows are at most the sequence and hidden is
  // narrower than inner, so it is never wider than the window it sits
  // in.
  //
  // `_vdn_ro` IS NOT, and it is the one buffer in the whole path that
  // cannot be: it is written before qk_norm and read after the out
  // projection, so it is live across the attention -- which is exactly
  // when the arena belongs to somebody else.
  metal_compute::SharedBuffer _vdn_ro, _vdn_proj;
  // The window as CSR spans for sdpa_spans: group 0 is the global rows
  // (prompt and soundtrack, dense in both directions) and group 1+f is
  // video frame f. Rebuilt only when the geometry changes.
  metal_compute::SharedBuffer _vdn_span_off, _vdn_span_lo, _vdn_span_hi;
  // The same window for the BLOCK-SPARSE steel kernel: per query block,
  // the key blocks it visits, plus the four numbers and the per-frame
  // bounds the kernel masks a block's edges against. A second shape of
  // one fact -- see vdn::build_block_spans for why it is not merely a
  // coarser WindowSpans.
  metal_compute::SharedBuffer _vdn_qb_off, _vdn_qb_blocks;
  metal_compute::SharedBuffer _vdn_sp_params, _vdn_sp_bounds;
  int _vdn_bspan_seq = -1, _vdn_bspan_bq = 0, _vdn_bspan_bk = 0;
  std::size_t _vdn_kb_sparse = 0, _vdn_kb_dense = 0;
  // Held because sdpa_spans_f16 is resolved out of it LATER, when a
  // branch is attached, and a ComputeFunction does not keep its library
  // alive.
  metal_compute::ComputeLibrary   _lib_sdpa;
  metal_compute::ComputeFunction  _fn_sdpa_spans;
  // The block-sparse steel kernel, which is the SAME entry point as the
  // dense one specialised on has_spans -- so it is cached beside it and
  // rebuilt on the same terms.
  metal_compute::ComputeFunction  _fn_attn_spans;
  int _attn_spans_seq = -1;
  // And which KERNEL it is, because both have the specialisation now:
  // the block list, the param block and the dispatch grid are all built
  // for one tile size, and a cached 64x32 pipeline addressed as 32x16 is
  // a different window rather than a slower one.
  int _attn_spans_nax = -1;
  // Which tiling the cached dense pair was built for. The cached
  // functions are tagged with the sequence length alone -- so without
  // this a later forward would reuse a 64x32 pipeline while addressing
  // it as 32x16.
  int _attn_nax_built = -1;
  // Whether the cached pair was built for Sage. A THIRD tag beside the
  // sequence and the kernel, because turning Sage on changes the
  // function and nothing else about the geometry does -- so without it
  // an A/B in one process would measure the first arm twice.
  int _attn_sage_built = -1;
  int _attn_pv8_built = -1;   // ...and its int8 P*V (sage::Config::pv8)
  int _vdn_frames = 0, _vdn_tpf = 0, _vdn_vstart = 0, _vdn_seq = 0;
  // A window wide enough to reach every frame IS the original attention,
  // and then the linear half must NOT run: it carries the window's
  // COMPLEMENT, and the complement of everything is nothing. Left on it
  // adds the text state to every video row -- no crash, no NaN, a
  // contribution the reference does not make. The reference calls this
  // `full_cover` and takes the stock attention kernel for it too, so the
  // full-cover path is exactly the unhybridised model plus the gate.
  bool _vdn_full_cover = false;
  unsigned _vdn_blocks_run = 0;
  // The window per query frame, UNREBASED -- the anchor skip belongs to
  // the branch, which applies it, because whether frames 0 and F-1 leave
  // the linear input is a property of the configuration and not of the
  // caller. The same bounds build the softmax side's spans, which is
  // what makes the two halves a partition rather than two guesses.
  std::vector<minimax_h3::vdn::Bound> _vdn_bounds;
  // Sized for the geometry the span buffers describe.
  // `bq`/`bk` are the tiles of the kernel that will READ the block list,
  // which is why they are a parameter and not a constant: the ALU steel
  // kernel is 32x16 and the matrix-core one 64x32, and a list built for
  // one is a different window under the other.
  bool ensure_vdn_(const Step& in, const minimax_h3::PackedLayout& L,
                   int bq, int bk, std::string* err);

  // Every buffer of the scratch, so the wired pool can take them all
  // without a second list that drifts from the struct above.
  std::vector<metal_compute::SharedBuffer*> scratch_buffers_();
  // How much of `_s.qkv` Sage may carve from at this geometry, or 0.
  // ONE place, because the ESTIMATE and the LEND must not disagree
  // about it -- and the conditions are not obvious: Sol takes that
  // buffer whenever it is on, and the fused attention spellings read
  // q/k/v out of it in place, so it is spare in exactly one case.
  std::size_t sage_lend_bytes_(int seq) const;

  // Wire (or give back) the model's FIXED holdings -- the activation
  // scratch. Charged to the shared pool like everything else.
  std::size_t wire_fixed_(bool on);

  // The measured GEMM route for one projection shape at the tuned
  // sequence length. Keyed on (N, K) rather than on a name because that
  // is what gemm_ has in hand, and two projections that share a shape
  // share an answer.
  struct QmmTune { int N = 0, K = 0; GemmRoute route = GemmRoute::kSteelBm32; };
  // Tuned shapes for ONE sequence length. Kept per M rather than only
  // for the last one: a graph that alternates two geometries would
  // otherwise re-tune on every forward, and a tune is ~3 blocks of GEMM
  // -- cheap once per clip, ruinous once per step.
  struct QmmTuneSet { int m = -1; std::vector<QmmTune> shapes; };
  std::vector<QmmTuneSet> _qmm_tuned;
  std::string _qmm_tuning_desc;   // human-readable, for qmm_tuning()
  bool _qmm_manual = false;       // set_qmm_tile/route called: tuning off
  int  _qmm_tune_count = 0;       // measurements taken, not cache hits
  GemmRoute _forced_route = GemmRoute::kAuto;
  // The rows a GEMM's route is looked up at, when not its own: the
  // banded forward's main blocks take the WHOLE sequence's routes (0:
  // each GEMM its own M).
  int _route_rows = 0;
  void tune_qmm_(int M);
  // The route for one projection: the forced one, else the tuned one,
  // else the fallback rule.
  GemmRoute gemm_route_(int M, int N, int K) const;
  // True when `r` can actually run this shape here (kernels loaded, M
  // amortizes the tile, N non-degenerate). Both the tuner and the
  // dispatcher ask, so an unavailable route is never a candidate AND
  // never dispatched.
  bool route_ok_(GemmRoute r, int M, int N, int K) const;
  static const char* route_name_(GemmRoute r);
  // One projection through `route`, bias excluded (gemm_ adds it).
  void gemm_route_dispatch_(metal_compute::ComputeEncoder& enc,
                            const metal_compute::SharedBuffer& x,
                            std::size_t x_off, const Linear& l,
                            const metal_compute::SharedBuffer& y,
                            std::size_t y_off, int M, int N, int K,
                            GemmRoute route, const LoraFactors* lora,
                            bool* lora_folded, std::size_t lora_off = 0);
  // `route`, over rows banded by mma_row_band -- the base projection
  // only, which is what the autotune measures.
  void dispatch_row_bands_(metal_compute::ComputeEncoder& enc,
                           const metal_compute::SharedBuffer& x,
                           std::size_t x_off, const Linear& l,
                           const metal_compute::SharedBuffer& y,
                           std::size_t y_off, int M, int N, int K,
                           GemmRoute route);
  void qmm_dispatch_(metal_compute::ComputeEncoder& enc,
                     const metal_compute::SharedBuffer& x, std::size_t x_off,
                     const Linear& l, const metal_compute::SharedBuffer& y,
                     std::size_t y_off, int M, int N, int K, int bm);
  // Dequant-once into _w_deq (quantized) or the weight as-is (dense),
  // then one dense matmul2d tile. False when the route cannot run, which
  // leaves the caller on steel.
  // `lora_folded` is set when the adapter's second GEMM was absorbed
  // into the tile, which tells the caller not to encode it separately.
  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& x, std::size_t x_off,
                 const Linear& l, const metal_compute::SharedBuffer& y,
                 std::size_t y_off, int M, int N, int K, GemmRoute route,
                 const LoraFactors* lora = nullptr,
                 bool* lora_folded = nullptr, std::size_t lora_off = 0);

  metal_compute::MetalCompute* _mc = nullptr;
  Config _cfg;
  std::shared_ptr<WeightSet> _ws;
  DitBlockProgressFn _block_progress;
  // Polled once per block in forward(); empty = no stop check.
  std::function<bool()> _stream_stop;

  // The shared grow-into-free-RAM policy; see block-residency.h for why
  // this is a fixed resident set rather than a cache.
  BlockResidency _resid;
  static std::size_t block_bytes_(const Block& b);

  Linear _video_patch, _audio_patch, _cond_proj;
  Linear _time_in, _time_out;                 // read to the HOST, f32
  std::vector<float> _time_in_w, _time_in_b, _time_out_w, _time_out_b;
  std::vector<float> _inv_freq;               // [rope_freq_dim], f32
  std::vector<Block> _refiner;                // 2 blocks, no AdaLN
  metal_compute::SharedBuffer _refiner_final_norm;
  // Preloaded: all 50. Streaming: empty at load and filled by growth as
  // free memory allows; blocks not resident are read from the retained weight
  // set on demand in forward() and freed after use.
  std::vector<Block> _blocks;
  bool _stream_blocks = false;
  // Said once when a checkpoint needs per-tensor repair on the refill
  // path, so a pack that pays for it every forward is visible.
  bool        _refill_repaired = false;
  // This model's window onto the manager's wired pool (shared/wired-pool.h).
  WiredPool   _wire;
  metal_compute::SharedBuffer _final_norm;
  Linear _final_adaln, _video_out, _audio_out;

  int _quant_bits = 0;
  int _quant_group = 64;

  metal_compute::ComputeLibrary _lib_gemm, _lib_elt, _lib_rms, _lib_rope,
      _lib_attn, _lib_attn_nax, _lib_qmm, _lib_dense_mma, _lib_dequant;
  metal_compute::ComputeFunction _fn_qmm4, _fn_qmm8;
  metal_compute::ComputeFunction _fn_qmm4_bm64, _fn_qmm8_bm64;
  metal_compute::ComputeFunction _fn_qmm4_bm128, _fn_qmm8_bm128;
  int _qmm_tile = 0;

  // ---- fused SwiGLU feed-forward (steel qmm only) -----------------------
  // One GEMM whose register-local epilogue writes silu(gate)*up, in place
  // of fc1-into-[rows, 2*ffn] plus a separate elementwise pass. Costs a
  // load-time interleave of fc1's rows (see Linear::gu_inter) and buys
  // both the pass and the double-width intermediate.
  //
  // Steel only, ON PURPOSE. matmul2d cannot fold the epilogue, so on the
  // M5 path it would dequantize the interleaved weight, run the same
  // matmul, and then need a swiglu fold anyway -- the same traffic plus a
  // second copy of the widest weight. Krea-2 measured that as neutral and
  // defaults it off there; H3 does not offer it at all rather than ship
  // an untested arm on hardware this was not measured on.
  metal_compute::ComputeFunction _fn_qmm_swiglu4, _fn_qmm_swiglu8;
  metal_compute::ComputeFunction _fn_qmm_swiglu4_bm64, _fn_qmm_swiglu8_bm64;
  bool _fuse_ff = false;
  // The main blocks' projections take I8GemmContext's int8 layout
  // (Config::i8_tier); decided before any block loads.
  bool _i8_layout = false;

  // ---- the ANE tier ------------------------------------------------
  // Built LAZILY at the first forward, because the chunked module needs
  // the row count to set the split and that is not known until then. A
  // null module means every block keeps its GPU feed-forward, which is
  // what every failure falls back to.
  std::unique_ptr<AneFeedForward> _ane;
  bool _ane_tried       = false;
  bool _ane_skip_warned = false;   // one warning for ineligible blocks
  bool _sage_off_noted  = false;   // one note when Sage fell back to dense
  // The qkv tier: a matmul module of its own, split like the FF.
  std::unique_ptr<AneFeedForward> _ane_qkv;
  bool _ane_qkv_tried = false;
  bool _ane_disabled  = false;   // disable_ane() was called
  bool ane_qkv_setup_(int seq);
  bool ane_qkv_eligible_(int L, const Block& b);
  void ane_qkv_stage_(int L, const Block& b, const LoraStack& lora);
  bool ane_setup_(int seq);        // false == tier off
  // Whether block L's feed-forward can go to the ANE: under the cap, no
  // bias, fc1/fc2 dense bf16 or 4/8-bit affine, and not an adapted fc1 in
  // the interleaved layout (the adapter's rows are [gate; up]).
  bool ane_eligible_(int L, const Block& b, bool adapted);
  // Dispatch block L's gate/up/down into the module's slots and return at
  // once, so the conversion runs under the block's attention. fc1 is
  // [gate; up] halves, or interleaved pairs once fused (Linear::gu_inter).
  // The live adapters on fc1/fc2 are merged into the staged rows.
  void ane_stage_(int L, const Block& b, const LoraStack& fc1,
                  const LoraStack& fc2);

  // ---- runtime LoRA ------------------------------------------------
  // One projection's factors. A [rank, K] and B [N, rank], both bf16 and
  // both in the checkpoint's own orientation, so the same transposed-W
  // GEMM serves them: t = x A^T, then y += t B^T.
  struct BlockLora {
    LoraFactors qkv, out, fc1, fc2, adaln;
    bool any() const
    {
      return !qkv.empty() || !out.empty() || !fc1.empty() || !fc2.empty() ||
             !adaln.empty();
    }
  };
  // An AdaLN CURVE BRANCH set -- LMS Sharp's, as Open-H3-Regenerate-2K
  // ships it (`lora/sharp_curve_branches.safetensors`): per main block,
  // `<block>.adaln_proj.linear.curve_A` [r, 8] and `.curve_B` [adaln_out,
  // r], whose INPUT is not the projection's but an 8-D point c(t) read
  // linearly off a 1025 x 8 table at the timestep (the table ComfyUI's
  // pruned H3 replaces the AdaLN MLP with; `adaln_t_table`). The block's
  // modulation gains slot.scale * B A c(t) for every distinct timestep of
  // a forward. B A is formed once at bind, f32 [adaln_out, 8] -- 3.1 MB
  // a block where B is 24.8 -- and stays resident when the blocks stream.
  struct CurveSet {
    std::vector<metal_compute::SharedBuffer> wc;   // per main block
    std::vector<float> table;                      // [knots, 8]
    int knots    = 0;
    int branches = 0;
    int rank     = 0;
    bool empty() const { return branches == 0; }
  };
  // One attached adapter. Slots are independent all the way down: their
  // own factors, their own live strength, and DISJOINT regions of the
  // shared [rows, rank] scratch, so the second adapter's first GEMM does
  // not have to wait on the first adapter's second.
  struct LoraSlot {
    std::vector<BlockLora> blocks;    // per main block
    std::vector<BlockLora> refiner;   // per refiner block (no adaln)
    LoraFactors            final_adaln;  // final_layer.adaln_proj.linear
    // The timestep MLP runs on the host in f32, so its adapter does too:
    // proj_in / proj_out of the checkpoint's own embedder, and of a
    // flow-map adapter's ENDPOINT embedder (empty for any other file).
    lora::HostFactors      time_in, time_out;
    lora::HostFactors      end_in, end_out;
    // Set when the FILE carries AdaLN curve branches (see CurveSet).
    CurveSet curve;
    // Set when the FILE is a flow-map adapter; load() then promotes it
    // to `_flow` with this slot's index.
    bool    two_time = false;
    FlowMap flow;
    // Set when a manifest beside the FILE states a sampling contract;
    // load() then promotes it to `_contract` with this slot's index.
    bool             has_contract = false;
    SamplingContract contract;
    // PRUNED: what this adapter adds to the AdaLN, per main block and
    // the final layer, f32 [D * 8 + D] = (dW, db) at strength 1 --
    // ComfyUI `.diff` / `.diff_b` patches as the file stores them, and a
    // full-model adapter's AdaLN modules ported onto the curve basis.
    // Empty buffer: nothing for that projection.
    std::vector<metal_compute::SharedBuffer> pdelta;
    metal_compute::SharedBuffer              pdelta_final;
    int   modules  = 0;
    int   banded   = 0;
    int   max_rank = 0;
    float scale    = 1.0f;
    // Ranks of the slots BEFORE this one. The scratch is [seq, sum of
    // ranks], so this slot's region starts at `rows * rank_prefix` --
    // which needs the row count, and that is a forward-time number.
    int   rank_prefix = 0;
  };
  std::vector<LoraSlot> _lora;
  // The flow-map adapter's header facts; `slot` < 0 when there is none.
  FlowMap _flow;
  // The sampling contract an adapter's manifest states; `slot` < 0 when
  // there is none.
  SamplingContract _contract;
  // Summed max ranks: what the [seq, rank] scratch has to hold for all
  // of them at once, and the stride that turns a rank prefix into an
  // offset.
  int _lora_rank_total = 0;
  // Rows the scratch was last allocated for. Set with it, read by
  // lora_stack_ -- a stack built before any forward would have nowhere
  // to point, and there is no forward without the scratch.
  std::size_t _lora_scratch_rows = 0;

  // ---- adapter factors STREAMED WITH THEIR BLOCK ----------------------
  //
  // A streamed DiT keeps about one main block resident, but its adapters'
  // factors stayed resident for all 50: TaoMate-H3's rank-128 set (fused
  // q|k|v, out, fc1, fc2 and the AdaLN projection) is ~50 MB a block,
  // ~2.4 GB in all -- beside a block of ~1.3 GB (bf16) or ~0.65 GB (w8)
  // and a box that is streaming precisely because it has no room.
  //
  // So, once the slots are bound -- every conversion done: alpha folded,
  // q|k|v fused or banded, rows permuted -- a streamed DiT SPILLS each
  // main block's factors to an unlinked temp file and frees them, keeping
  // only their shape (rank, parts, group). A streamed block's iteration
  // reads its factors back into ONE set of buffers per slot, refilled in
  // place: the block's commit is waited for before the next block starts
  // (see the promotion below the block), so nothing still reads them.
  // A block PROMOTED to resident takes a resident copy of its factors,
  // and an evicted one gives it back. File-backed pages are the cheapest
  // memory the kernel can take back, unlike the anonymous GPU buffers
  // they replace. VPIPE_H3_LORA_RESIDENT=1 keeps them all resident.
  struct LoraSpillAt {
    std::uint64_t a_off = 0, a_bytes = 0, b_off = 0, b_bytes = 0;
  };
  struct BlockSpill { LoraSpillAt qkv, out, fc1, fc2, adaln; };
  // The five adapted projections of a main block, in one order for the
  // spill, the fill and the copies.
  static constexpr LoraFactors BlockLora::* kLoraMembers[] = {
    &BlockLora::qkv, &BlockLora::out, &BlockLora::fc1, &BlockLora::fc2,
    &BlockLora::adaln,
  };
  static constexpr LoraSpillAt BlockSpill::* kSpillMembers[] = {
    &BlockSpill::qkv, &BlockSpill::out, &BlockSpill::fc1, &BlockSpill::fc2,
    &BlockSpill::adaln,
  };
  struct SpillFile {
    int fd = -1;
    SpillFile() = default;
    SpillFile(const SpillFile&) = delete;
    SpillFile& operator=(const SpillFile&) = delete;
    ~SpillFile();
  };
  SpillFile _lora_spill_file;
  std::vector<std::vector<BlockSpill>> _lora_spill;   // [slot][main block]
  // The streamed block's factors, per slot; and which block they hold.
  std::vector<BlockLora> _lora_stream;
  int _lora_stream_layer = -1;
  // Spill every main block not resident now. False (nothing changed) when
  // the file cannot be made.
  bool spill_lora_();
  // Read block `layer`'s factors into _lora_stream.
  bool fill_lora_(int layer);
  // A promoted block's factors made resident from _lora_stream; an
  // evicted block's given back.
  void keep_lora_(int layer);
  void drop_lora_(int layer);
  // What keeping block `layer` keeps of its spilled factors (0 where
  // nothing spilled) -- booked with the block by residency, wired with
  // it, given back with it.
  std::size_t lora_layer_bytes_(int layer) const;
  std::size_t wire_lora_(int layer, bool on);
  // Factor bytes residency holds for kept blocks: inside _resid.bytes(),
  // and out of streamed_holding_bytes() -- they are the adapter's own
  // declaration's, which books every factor.
  std::size_t _lora_kept = 0;

  // The live adapters for one projection of one block, named by the
  // member that holds it. `refiner` picks which stack of blocks.
  LoraStack lora_stack_(bool refiner, int layer,
                        LoraFactors BlockLora::* which) const;
  // The same for final_layer.adaln_proj.linear, which is not in a block.
  LoraStack lora_stack_final_() const;
  metal_compute::ComputeFunction _fn_gemm_acc;
  // The matrix-core twins of that pair. `_a64`/`_a128` scale on store and
  // serve the rank-wide first GEMM; `_b128`/`_b256` seed their register
  // tile from y and serve the rank-deep second one. Bound only where the
  // dense mma tiles are, so a non-M5 build leaves them invalid and
  // lora_apply_ falls through to steel.
  metal_compute::ComputeFunction _fn_lora_a64, _fn_lora_a128;
  metal_compute::ComputeFunction _fn_lora_b128, _fn_lora_b256;
  // The base-projection tiles that also carry the adapter's second
  // factor, so `y = W x + B (A x)` is one store instead of three.
  metal_compute::ComputeFunction _fn_lora_fused128, _fn_lora_fused256;
  // BANDED twins of the second-factor kernels, for a split q/k/v adapter
  // fused WITHOUT a block-diagonal B (see lora::Factors::banded): each
  // output tile contracts over its own part's window of t. Steel always;
  // the matrix-core ones where the dense mma library is.
  metal_compute::ComputeFunction _fn_gemm_acc_band;
  metal_compute::ComputeFunction _fn_lora_b128_band, _fn_lora_b256_band;
  metal_compute::ComputeFunction _fn_lora_fused128_band,
                                 _fn_lora_fused256_band;
  // VPIPE_H3_LORA_NO_BAND: bind split q/k/v the old way, block-diagonal.
  // The A/B that says the banded form changes nothing but the zeros.
  bool _lora_band_off = false;
  // ---- baked AdaLN (see bake_adaln) ---------------------------------
  // Per block, [total_rows, adaln_out] bf16; plus the final layer's
  // [total_rows, 2*hidden]. Empty = not baked.
  std::vector<metal_compute::SharedBuffer> _adaln_tab;
  metal_compute::SharedBuffer _final_adaln_tab;
  // Per step: where its rows start in the tables, and how many. A step's
  // rows are CONTIGUOUS, which is what lets a block bind its slice
  // directly instead of gathering.
  std::vector<int> _adaln_row0, _adaln_nt;
  metal_compute::ComputeFunction _fn_copy;
  // copy_rect_f16: the streaming attention's K/V gathers.
  metal_compute::ComputeFunction _fn_copy_rect;

  bool _lora_mma_off = false;    // VPIPE_H3_NO_LORA_MMA
  bool _lora_fuse_off = false;   // VPIPE_H3_NO_LORA_FUSE
  // Deepest base contraction the fold is allowed on; see gemm_mma_ for
  // the cliff this is protecting against. VPIPE_H3_LORA_FUSE_MAX_K.
  int _lora_fuse_max_k = 8192;

  // Read the adapter and bind its factors to the modules above. Called
  // from load() once the blocks exist.
  bool bind_lora_(const LoraSpec& spec, LoraSlot& slot, std::string* err);
  // A file of AdaLN curve branches into slot.curve (true, with `*took`
  // set); `*took` false for a file that carries none.
  bool bind_curves_(const std::string& path, LoraSlot& slot, bool* took,
                    std::string* err);
  // Add every live curve slot's delta to block L's modulation [n_t,
  // adaln_out] in `mod`; false when none applies (nothing encoded).
  bool curve_live_(int L) const;
  void curve_apply_(metal_compute::ComputeEncoder& enc, int L,
                    const metal_compute::SharedBuffer& mod, int n_t);
  metal_compute::ComputeLibrary  _lib_regen;
  metal_compute::ComputeFunction _fn_curve_add;
  // The PRUNED form's AdaLN (Config::curve_grid): `adaln_t_table` on the
  // host in f32, every projection AS STORED in the trunk -- [D, 8] and
  // [D], `dtype` 0 F32 / 1 F16 / 2 BF16, widened exactly in the kernel
  // -- and this forward's points, [n_t][8] f32, fresh every forward as
  // `_curve_coords` is.
  struct PrunedAdaln {
    metal_compute::SharedBuffer w, b;
    int dtype = 0;
  };
  std::vector<float>       _adaln_t_table;
  std::vector<PrunedAdaln> _padaln;          // one a main block
  PrunedAdaln              _final_padaln;
  metal_compute::SharedBuffer    _pcoords;
  metal_compute::ComputeFunction _fn_pruned_mod;
  metal_compute::ComputeFunction _fn_pruned_patch;
  // The projection's modulation into `out` [n_t, D]; `L` the main block,
  // -1 the final layer, whose live slots' deltas are added to the
  // weight in f32 first (into `_padaln_patch`).
  // `res` (and `res_off`, bytes): the baked off-curve rows to add, when
  // an adapter moves the embedding (bake_adaln).
  void pruned_mod_(metal_compute::ComputeEncoder& enc,
                   const PrunedAdaln& p, int L,
                   const metal_compute::SharedBuffer& out, int n_t, int D,
                   const metal_compute::SharedBuffer* res = nullptr,
                   std::size_t res_off = 0);
  metal_compute::ComputeFunction _fn_pruned_res;
  // A native W4A8 weight's int8 codes for the GEMM running (gemm_), from
  // w4a8_to_i8; and that kernel.
  metal_compute::SharedBuffer    _cr_codes;
  // fp8_widen_bytes() for what this model read.
  std::size_t _f8_widen = 0;
  metal_compute::SharedBuffer convrot_row_scales_(
      const MetalLlamaWeights& src, const fp8::Weight& fw, int N);
  std::uint64_t                  _cr_gemms = 0;
  metal_compute::ComputeLibrary  _lib_dq;
  metal_compute::ComputeFunction _fn_w4a8_i8;
  metal_compute::SharedBuffer _padaln_patch;
  // The pruned curve's affine basis in the RELEASED embedding, fit at
  // the first need from Config::time_mlp_dir's MLP over the table:
  //   silu(temb(t)) ~ U c(t) + m       (U [mlp, 8], m [mlp])
  // `pinv` [8, mlp] = (U^T U)^-1 U^T takes an embedding back onto the
  // curve's coordinates. `residual` is the fit's relative residual.
  struct CurveBasis {
    std::vector<float> u, m, pinv;
    int    mlp = 0;
    double residual = 0.0;
    bool empty() const { return mlp == 0; }
  };
  CurveBasis _basis;
  // Read the released MLP and fit the basis; false + *err when there is
  // no time_mlp_dir, it cannot be read, or it does not span the table.
  bool ensure_curve_basis_(std::string* err);
  // The MLP's output width: the AdaLN input released, the basis's pruned.
  int mlp_dim_() const { return _cfg.pruned() ? _basis.mlp : _cfg.time_dim; }
  // Whether this forward's points come from the MLP projected onto the
  // basis (a live adapter on the timestep MLP, or a flow map) rather
  // than straight off the table.
  bool pruned_projects_() const;
  // A pruned adapter's AdaLN deltas: `.diff` / `.diff_b` read off the
  // file, then the full-model AdaLN modules `ad` binds ported through
  // the basis. Fills slot.pdelta / pdelta_final.
  bool bind_pruned_adaln_(lora::Adapter* ad, const std::string& path,
                          bool diffusers, LoraSlot& slot, std::string* err);
  // bake_adaln for a pruned DiT whose embedding an adapter moves: the
  // rows (`all`, and a flow map's endpoints) the curve cannot carry,
  // through the released AdaLN, into _adaln_tab / _final_adaln_tab.
  bool bake_pruned_off_curve_(const std::vector<float>& all,
                              const std::vector<float>* all_r,
                              std::string* err);
  // This forward's c(t), [curve slot][n_t][8] f32, filled after the
  // timestep embedding; `_curve_n_t` the n_t it holds.
  metal_compute::SharedBuffer _curve_coords;
  int _curve_n_t = 0;
  // The timestep-MLP half of a bind, both branches: the base embedder's
  // host factors for ANY adapter that carries them, then -- for a
  // flow-map adapter -- the endpoint embedder's and the header facts
  // (gate, grid, shifts). A file that is not one passes with only the
  // first; a file that is HALF of one -- the header without the
  // embedder, or the reverse -- is refused.
  bool bind_flow_map_(lora::Adapter& ad, const std::string& path,
                      LoraSlot& slot, std::string* err);
  // Which matrix-core tile serves each half of the adapter, or nullptr
  // for "stay on steel". Split in two because the two GEMMs are skinny in
  // different dimensions and do not flip together -- see the measured
  // tables at the definitions.
  const metal_compute::ComputeFunction* lora_route_a_(int rank) const;
  const metal_compute::ComputeFunction* lora_route_b_(int rank, int N) const;
  // The adapter as its two halves, because the second one has two homes.
  //
  // `lora_gemm_a_` writes t = [scale *] x A^T into the shared scratch and
  // returns whether it APPLIED the strength -- only the matrix-core tile
  // can, and the answer decides both where the scale goes and whether the
  // fused base tile is legal at all (that tile has no scale of its own).
  // `scale` is the SLOT's, passed rather than read off the object: with
  // two adapters live there is no single current strength.
  bool lora_gemm_a_(metal_compute::ComputeEncoder& enc,
                    const metal_compute::SharedBuffer& x, std::size_t x_off,
                    const LoraFactors& lf, int M, int K, float scale,
                    std::size_t lora_off = 0);
  void lora_gemm_b_(metal_compute::ComputeEncoder& enc, const LoraFactors& lf,
                    const metal_compute::SharedBuffer& y, std::size_t y_off,
                    int M, int N, bool scale_applied, float scale,
                    std::size_t lora_off = 0);
  // Every attached adapter, back to back. The standalone form, for the
  // modulation projections that do not go through gemm_.
  void lora_apply_(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& x, std::size_t x_off,
                   const LoraStack& st,
                   const metal_compute::SharedBuffer& y, std::size_t y_off,
                   int M, int N, int K);
  // Whether the ff scratch still has to hold a [seq, 2*ffn] fc1 output.
  // True unless EVERY block that can run is interleaved -- a streaming
  // stack can present an unfused block at any step, and one such block
  // writing into a buffer sized for the fused path is a heap overrun,
  // not a slow forward.
  bool _ff_needs_wide = true;

  // Rewrite `l` (an fc1) into the interleaved gate/up layout, in place.
  // False when the layout is not one this can express, and `l` is then
  // untouched and still correct on the split path.
  bool interleave_gu_(Linear& l) const;
  // The same layout for a block being loaded to KEEP, built through the
  // weight set so the source layout is never cached alongside it.
  Linear fc1_gu_(WeightSet& ws, const std::string& nm);

  // ---- M5 matrix cores: dequant-once + dense matmul2d -------------------
  metal_compute::ComputeFunction _fn_dense_mma, _fn_dense_mma_deep,
      _fn_dense_mma_tn2;
  metal_compute::ComputeFunction _fn_dequant4, _fn_dequant8;
  // The [N, K] bf16 dequant scratch, shared across a block's projections.
  // Sized by the WIDEST projection (fc1: 2*ffn x hidden = 308 MB at the
  // released config), grown on demand and kept -- regrowing it per block
  // would be an allocation inside the denoise loop.
  metal_compute::SharedBuffer _w_deq;
  // The checkpoint's FP8 weights (shared/fp8-layout.h) and the kernels
  // that widen them per GEMM. The latch reports an expansion that could
  // not run, which leaves a projection unwritten, as a failed forward.
  fp8::Layout   _f8;
  fp8::Expander _fp8x;
  bool          _fp8_failed = false;
  // Dynamic-int8 accelerated GEMMs (Config::i8_gemm). Null unless enabled
  // AND available; the DiT runs bf16, so it holds the bf16 i8 kernels --
  // the f16 ones would misread these buffers into NaN.
  std::unique_ptr<I8GemmContext> _i8;
  // Sol-Attn's kernels and scratch. Null unless Config::sol.enabled and
  // the kernels validated; see the attention dispatch.
  std::unique_ptr<MetalSolAttention> _sol;
  // perf-flops: GEMMs and attention counted as encoded, credited at
  // retire; Sol's exact half at the keep ratio it realizes.
  GpuFlopsTally _flops;
  SolFlops _sol_flops;
  // The int8 QK prologue and its operands. Null when the box has no
  // matrix cores or the config did not ask -- and every use is guarded,
  // because a box without them runs the same forward densely.
  std::unique_ptr<MetalSageAttention> _sage;
  // Deep-K split for fc2 (K = ffn = 14336). Its own gate|up direction runs
  // ~11 TFLOP/s where the single-op reduction over ffn manages ~5.3, and the
  // chooser closes that; see mma-splitk.h. Self-gates on K, on rows, and on
  // a chunk kernel existing, so every other projection here is untouched.
  MmaSplitK _splitk;
  bool _use_mma2 = false;
  // M below which the 128-row tile is not amortized and steel wins. Only
  // a FLOOR on what the autotune may pick; the tuner still measures above
  // it, because "the tile fits" and "the tile wins" are different claims.
  int _mma_min_m = 64;
  metal_compute::ComputeFunction _fn_gemm, _fn_rms, _fn_rms_heads, _fn_trope,
      _fn_modulate, _fn_gated, _fn_swiglu, _fn_transpose, _fn_residual,
      _fn_bias_add, _fn_sdpa;
  // Rope written back over the fused projection, for the path that never
  // transposes. See _fused_attn.
  metal_compute::ComputeFunction _fn_rope_ip;
  // The non-finite tripwire and its 8-word report. Runs in the stream, so
  // finding the op that mints a NaN costs a pass over the tensor rather
  // than a GPU drain -- and draining is what made this bug vanish.
  metal_compute::ComputeFunction _fn_nan_trip;
  metal_compute::SharedBuffer    _nan_report;
  metal_compute::SharedBuffer _attn_p_main, _attn_p_text;
  // Sage's int8 P*V on the VIDEO rows only, unless pv8_all_rows: the
  // prompt's and the soundtrack's query blocks on the int8 QK alone
  // (`_fn_attn_main_i8_qk`, Q per token as the prologue made it), the
  // rest from `_attn_band_nb` on, with params counting blocks from there
  // (`_attn_p_band`).
  metal_compute::ComputeFunction _fn_attn_main_i8_qk;
  metal_compute::SharedBuffer _attn_p_band;
  int _attn_band_nb = 0;
  // The BANDED forward's (band_rows): its q / k / v written a band at a
  // time into the whole-sequence head-major windows, and its attention a
  // band of queries over every key -- one parameter block and pipeline
  // for the full bands, one for the last; keyed by the sequence, the band
  // and the kernel.
  metal_compute::ComputeFunction _fn_trope_band;
  // [full band | last band][0 dense, 1 int8 QK (per-block Q scales),
  // 2 int8 QK (per-token Q: band 0's prompt and soundtrack rows under
  // the int8 P*V), 3 int8 QK and P*V]; band 0's parameters past those
  // rows (_attn_p_bandq_rest, as _attn_p_band is the whole sequence's).
  metal_compute::ComputeFunction _fn_attn_bandv[2][4];
  metal_compute::SharedBuffer _attn_p_bandq, _attn_p_bandq_tail;
  metal_compute::SharedBuffer _attn_p_bandq_rest;
  int _attn_bandq_seq = -1, _attn_bandq_rows = -1, _attn_bandq_nax = -1;
  int _attn_bandq_sage = -1, _attn_bandq_nb = 0;
  int _last_band = 0;
  // Sol's last band: ONE band size per Sol object, whose parameter block
  // and scratch are per object -- a narrower call on the same object
  // would rewrite the parameters the full bands' encoded dispatches
  // read, and rebuild the scratch they use.
  std::unique_ptr<MetalSolAttention> _sol_tail;
  // Sol's routed key blocks over the last forward's calls, for its
  // report (a banded layer makes several calls).
  long long _sol_routed_total = 0;
  // The Sol config this forward routes with: the load's, or the Step's
  // own tau and threshold (Step::sol_routing).
  sol::Config _sol_fwd;
  // The streaming attention's two parameter blocks (text over text, media
  // over [cache ; text ; media]) and its pipelines, one per alignment
  // pair -- qL and kL move every chunk, so they are built on demand and
  // kept: [nax][align_Q][align_K].
  metal_compute::SharedBuffer _sattn_p_text, _sattn_p_media;
  metal_compute::ComputeFunction _fn_sattn[2][2][2];
  // A floor under the activation scratch: kept rather than rebuilt while
  // a forward fits it. See set_scratch_floor().
  int _scratch_floor_seq = 0, _scratch_floor_text = 0;
  // Elements the q|k|v scratch must hold beyond its own [seq, 3 * inner]:
  // a stream forward's K/V staging lives in it (StreamKv::adopt_staging).
  std::size_t _qkv_floor_elems = 0;
  metal_compute::ComputeFunction _fn_attn_main, _fn_attn_text;
  // The int8-QK twin of _fn_attn_main. Two instantiations rather than
  // one, because `sage.dense_layers` leaves the leading blocks on the
  // f16 kernel and the two must coexist within one forward.
  metal_compute::ComputeFunction _fn_attn_main_i8;
  // THE WIDE QUERY TILE (BQ 128 = WM 8), and the probe that chooses it.
  // The NAX attention streams the whole K/V once per query tile, so a
  // wider tile halves that re-read -- which pays only once the K/V no
  // longer fits in cache, and only on a GPU that is memory-bound rather
  // than occupancy-bound there. MEASURED bd128 bf16, bit-identical
  // output: an M5 Pro gains 1.13x at 19k rows and 1.22x at 100k (holding
  // ~24 TFLOP/s where BQ 64 decays to 20); a 10-core M5 Air LOSES 5% at
  // 9.4k and gains 8-13% at 19k-38k; a 48q/12kv GQA shape gains nothing
  // on either, its K/V being 4x smaller. So it is probed, not assumed.
  // Rows at or above which the wide tile is worth probing and using. Both
  // machines measured crossed near 19k; the probe times the two tiles over
  // the same window, which is all the ranking needs.
  static constexpr int kAttnBqMinRows   = 16384;
  static constexpr int kAttnBqProbeRows = 16384;
  metal_compute::ComputeFunction _fn_attn_main_bq128;
  metal_compute::SharedBuffer _attn_p_main_bq128;
  int  _attn_bq        = 64;     // the tile in force (64 or 128)
  int  _attn_bq128_seq = -1;     // sequence the BQ 128 pair was built for
  int  _attn_wide_built = -1;    // whether the built pair is the wide one
  // THE REAL-WORKLOAD PROBE, on the ANE tier's pattern (plan_block /
  // note_probe): the streamed path already commits and waits per block, so a
  // block's wall time is there for the taking. Warm on the incumbent, then
  // alternate whole blocks between the two tiles and keep the median winner
  // -- the real shape at the real length, no synthetic dispatch and no
  // truncation. `_attn_tile_blocks` counts main blocks seen; the medians are
  // small rings, one per tile.
  static constexpr int kTileWarmBlocks  = 2;   // settle clocks/caches first
  static constexpr int kTileProbeBlocks = 6;   // 3 per tile, alternating
  static constexpr int kTileRing        = 3;
  int    _attn_tile_blocks = 0;
  int    _attn_tile_n[2]   = {0, 0};
  double _attn_tile_ms[2][kTileRing] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  bool   _attn_tile_latched = false;
  int    _attn_tile_probing = -1;  // tile this block is measuring, -1 = none

  // The tile the NEXT main block should run, and the block's measured wall
  // time coming back. `probing` says whether that time is a probe sample.
  int  plan_attn_tile_(bool wide_available, bool streaming);
  void note_attn_tile_(double block_ms);
  // The session key of the probe's verdict (tuned-answers.h), and the
  // sequence the probe running now is measuring.
  TuneKey attn_tile_key_(int seq) const;
  int     _attn_tile_seq = -1;
  // Rows the GEMM tuner measures at, and therefore the row count a tuned
  // answer is filed under -- see tune_row_key(). Fixed at load from
  // VPIPE_H3_TUNE_ROWS.
  int _tune_ceiling = 4096;
  int _pf_started = 0, _pf_hit = 0;
  int _attn_seq = -1, _attn_text = -1;
  // Which layout the cached AttnParams describe. Tracked apart from the
  // sequence lengths so set_fused_attention() re-fills the params
  // without rebuilding the pipeline states.
  int _attn_fused = -1;
  bool _steel_ok = false;
  // The matrix-core (NAX) flash attention. Same param block and function
  // constants as the ALU steel kernel; only the tile sizes differ, so the
  // two are interchangeable at the call site.
  bool _attn_nax = false;
  // Which of the four per-block [seq, inner] transposes attention no
  // longer needs. Steel takes per-axis STRIDES for Q, K, V and O, so the
  // head-major layout was never a requirement of the kernel -- only of
  // the buffers we happened to hand it.
  //
  // The two bits are NOT the same trade, which is the whole reason they
  // are separate. MEASURED on the M5 at 9382 rows, 2 blocks, the three
  // masks interleaved inside one process:
  //
  //   kFusedAttnOut  the transpose after attention becomes a strided
  //                  STORE. Free -- attention 482 ms either way -- and
  //                  it is one fewer dispatch. Shipped.
  //   kFusedAttnQkv  q/k/v become strided READS, and rope moves in
  //                  place. Prep 26.7 -> 21.2 ms, and attention
  //                  482 -> 505.8 ms: +23.8 to save 5.5, in every one of
  //                  three measurements with no overlap between the arms.
  //                  A net ~1% loss. NOT shipped.
  //
  // The asymmetry is K and V REUSE. Q is read once and O written once,
  // so a stride costs them nothing; K and V are re-read by every query
  // block. Head-major, one head's K is a contiguous 2.4 MB that stays
  // cached across the whole Q loop. In the fused projection the same
  // 2.4 MB is scattered one head_dim slice per 43 KB row, so each pass
  // walks 400 MB of address space to touch it. That is a property of
  // the layout, not of this GPU, and it is why the Qkv bit is kept as
  // an A/B arm rather than as a tuning knob anyone should expect to
  // flip.
  //
  // Worth knowing before attacking any of this again: at these shapes
  // the four transposes are ~1.1 GB of traffic against ~9.7 TFLOP of
  // arithmetic per block, and the DiT runs at ~10.6 TFLOP/s. The whole
  // prize was under half a percent of the step. It was measured, not
  // assumed, and the answer is that the transposes were never where the
  // time is.
  //
  // Steel only: the scalar sdpa fallback has no strides. The head-major
  // windows therefore stay allocated for it, so this saves traffic, not
  // bytes. VPIPE_H3_FUSED_ATTN=<mask> overrides the default.
  int _fused_attn = 0;
};

}  // namespace genai
}  // namespace vpipe

#endif
