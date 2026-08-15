#ifndef GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_TRANSFORMER_H
#define GENERATIVE_MODELS_MINIMAX_H3_METAL_MINIMAX_H3_TRANSFORMER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/flex-data.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"
#include "generative-models/shared/block-residency.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/mma-splitk.h"
#include "generative-models/shared/dit-block-progress.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;   // generative-models/weight-set.h

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
    int rope_freq_dim = 16;       // per AXIS; 3 axes, doubled -> rot 96
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
  // `pin_frac` (streaming only): when > 0, pin a LEADING prefix of blocks
  // resident so pinned + running stays within that fraction of physical
  // RAM. Pinned blocks are read once and reused by every forward; only
  // the tail streams. 0 => pure streaming.
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
  };

  static std::unique_ptr<MetalMiniMaxH3Transformer>
  load(const std::string& dit_dir, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false, double pin_frac = 0.0,
       const LoraSpec* lora = nullptr);

  static std::unique_ptr<MetalMiniMaxH3Transformer>
  load(std::shared_ptr<WeightSet> ws, metal_compute::MetalCompute* mc,
       const Config& cfg, bool stream_blocks = false, double pin_frac = 0.0,
       const LoraSpec* lora = nullptr);

  // How many projections carry a runtime adapter (0 = none attached).
  int lora_modules() const { return _lora_modules; }

  // The adapter's strength, per FORWARD. Live because it can be: the
  // factors are the adapter's and the strength is the request's, so it
  // rides the accumulating GEMM as a constant rather than being folded
  // into A. Turning it costs nothing and needs no reload -- which is the
  // difference between a knob and a rebuild, and the reason to prefer
  // the runtime path when a strength has to be swept. 0 skips the two
  // GEMMs entirely, so "off" is exactly off.
  void set_lora_scale(float s) { _lora_scale = s; }
  float lora_scale() const { return _lora_scale; }

  // Leading blocks pinned resident in streaming mode (0 = pure streaming,
  // or preloaded). For logging the RAM-for-speed decision.
  int pinned_blocks() const { return _pinned; }
  bool streaming() const { return _stream_blocks; }

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
    // Which entry of a baked schedule this evaluation is, or -1 for
    // "not baked" (then `timesteps` drives the AdaLN projection as
    // before). See bake_adaln.
    int schedule_index = -1;
  };

  struct Velocity {
    // [num_video_rows, 96] and [num_audio_rows, audio_channels] bf16, in
    // the same row order the inputs were given in.
    metal_compute::SharedBuffer video, audio;
    bool empty() const { return video.empty(); }
  };

  // One denoiser evaluation. Empty on failure, with a reason in `err`.
  Velocity forward(const Step& in, std::string* err = nullptr);

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
  bool bake_adaln(const std::vector<std::vector<float>>& schedule,
                  std::string* err = nullptr);
  bool adaln_baked() const { return !_adaln_tab.empty(); }
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

  // Bytes currently held by promoted blocks, and how many there are.
  // Reported so a run can say how much of the box it actually used.
  std::size_t resident_block_bytes() const { return _resid.bytes(); }
  int resident_block_count() const { return _resid.count(); }

  // Give back at least `bytes` of promoted blocks, for a peer that needs
  // the room (the VAE decode is the case this exists for). Returns what
  // was freed, and ratchets the growth ceiling down so the set does not
  // immediately reclaim it. Never touches the configured pinned prefix.
  std::size_t release_resident_blocks(std::size_t bytes);
  // Free the highest resident block; its bytes, or 0 when there is
  // nothing left. `allow_pinned` lets it take one out of the pinned
  // prefix (shrinking `_pinned` to match), which is only right when a
  // measurement says that prefix is no longer in RAM.
  std::size_t evict_tail_block_(bool allow_pinned = false);
  // Pages of the resident set examined, and how many are still in RAM.
  void resident_pages_(std::size_t* examined, std::size_t* incore) const;

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
  static std::size_t scratch_bytes(const Config& c, int seq, int n_text,
                                   int n_t, bool with_dequant,
                                   bool narrow_ff = false);

  // What the per-forward scratch buffers ACTUALLY hold right now, and
  // what the matrix-core dequant scratch holds, kept apart because they
  // are sized by different things -- the first by the geometry, the
  // second by the widest projection in the checkpoint. Both zero before
  // the first forward. They exist so scratch_bytes() can be CHECKED
  // against reality rather than trusted; see the test named on its
  // definition.
  std::size_t scratch_resident_bytes() const;
  std::size_t dequant_scratch_bytes() const;

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
  // True when every block runs the fused-SwiGLU FF, so the ff scratch is
  // the narrow one. Pair with scratch_bytes()'s `narrow_ff`.
  bool ff_scratch_narrow() const { return !_ff_needs_wide; }
  // Whether the DiT's flash attention is the matrix-core (NAX) kernel.
  bool uses_nax_attention() const { return _attn_nax; }

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
  struct LoraFactors;

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
    bool empty() const { return quantized ? codes.empty() : w.empty(); }
  };

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

  metal_compute::SharedBuffer weight_(WeightSet& ws, const std::string& nm,
                                      Retain r);
  Linear linear_(WeightSet& ws, const std::string& nm, bool bias, Retain r);
  bool load_block_(WeightSet& ws, const std::string& prefix, Block& b,
                   bool with_adaln, Retain r);

  // Offsets are in ELEMENTS, not bytes, and exist because every
  // modality writes into its own contiguous slice of the one packed
  // sequence -- the reference's index_copy scatters are destination
  // offsets here.
  // `lora`, when given, is applied to this projection: the adapter's
  // first GEMM is encoded BEFORE the base one (the fused tile consumes
  // its output), and its second is folded into the base tile where the
  // route allows and encoded separately where it does not.
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x, std::size_t x_off,
             const Linear& l, const metal_compute::SharedBuffer& y,
             std::size_t y_off, int M, int N, int K,
             const LoraFactors* lora = nullptr);

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
  bool time_embed_(const std::vector<float>& timesteps,
                   metal_compute::SharedBuffer& out) const;

  struct Scratch {
    int seq = -1, n_text = -1, n_t = -1;
    metal_compute::SharedBuffer rcos, rsin;
    metal_compute::SharedBuffer x, nm, qkv, qh, kh, vh, oh, ob, ff, proj;
    metal_compute::SharedBuffer txt, temb, mod, fmod;
    metal_compute::SharedBuffer adaln_idx, tstep_idx;
    metal_compute::SharedBuffer lora;   // [seq, max rank], when attached
  };
  bool ensure_scratch_(int seq, int n_text, int n_t);
  Scratch _s;

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
  GemmRoute _forced_route = GemmRoute::kAuto;
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
  // `route`, over rows banded by mma_row_band_ -- the base projection
  // only, which is what the autotune measures.
  void dispatch_row_bands_(metal_compute::ComputeEncoder& enc,
                           const metal_compute::SharedBuffer& x,
                           std::size_t x_off, const Linear& l,
                           const metal_compute::SharedBuffer& y,
                           std::size_t y_off, int M, int N, int K,
                           GemmRoute route);
  // Rows one matmul2d dispatch may write into an N-wide destination.
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
  int mma_row_band_(int N) const;
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
  // Preloaded: all 50. Streaming: only the pinned prefix (_pinned,
  // possibly 0); blocks L >= _pinned are read from the retained weight
  // set on demand in forward() and freed after use.
  std::vector<Block> _blocks;
  bool _stream_blocks = false;
  int  _pinned = 0;
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

  // ---- runtime LoRA ------------------------------------------------
  // One projection's factors. A [rank, K] and B [N, rank], both bf16 and
  // both in the checkpoint's own orientation, so the same transposed-W
  // GEMM serves them: t = x A^T, then y += t B^T.
  struct LoraFactors {
    metal_compute::SharedBuffer a, b;
    int rank = 0;
    bool empty() const { return rank <= 0 || a.empty() || b.empty(); }
  };
  struct BlockLora {
    LoraFactors qkv, out, fc1, fc2, adaln;
    bool any() const
    {
      return !qkv.empty() || !out.empty() || !fc1.empty() || !fc2.empty() ||
             !adaln.empty();
    }
  };
  std::vector<BlockLora> _lora_blocks;    // per main block
  std::vector<BlockLora> _lora_refiner;   // per refiner block (no adaln)
  LoraFactors            _lora_final;     // final_layer.adaln_proj.linear
  int                    _lora_modules = 0;
  float                  _lora_scale = 1.0f;
  int                    _lora_max_rank = 0;
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

  bool _lora_mma_off = false;    // VPIPE_H3_NO_LORA_MMA
  bool _lora_fuse_off = false;   // VPIPE_H3_NO_LORA_FUSE
  // Deepest base contraction the fold is allowed on; see gemm_mma_ for
  // the cliff this is protecting against. VPIPE_H3_LORA_FUSE_MAX_K.
  int _lora_fuse_max_k = 8192;

  // Read the adapter and bind its factors to the modules above. Called
  // from load() once the blocks exist.
  bool bind_lora_(const LoraSpec& spec, std::string* err);
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
  bool lora_gemm_a_(metal_compute::ComputeEncoder& enc,
                    const metal_compute::SharedBuffer& x, std::size_t x_off,
                    const LoraFactors& lf, int M, int K,
                    std::size_t lora_off = 0);
  void lora_gemm_b_(metal_compute::ComputeEncoder& enc, const LoraFactors& lf,
                    const metal_compute::SharedBuffer& y, std::size_t y_off,
                    int M, int N, bool scale_applied,
                    std::size_t lora_off = 0);
  // Both halves back to back. The standalone form, for the modulation
  // projections that do not go through gemm_.
  void lora_apply_(metal_compute::ComputeEncoder& enc,
                   const metal_compute::SharedBuffer& x, std::size_t x_off,
                   const LoraFactors& lf,
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
  // Dynamic-int8 accelerated GEMMs (Config::i8_gemm). Null unless enabled
  // AND available; the DiT runs bf16, so it holds the bf16 i8 kernels --
  // the f16 ones would misread these buffers into NaN.
  std::unique_ptr<I8GemmContext> _i8;
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
  metal_compute::SharedBuffer _attn_p_main, _attn_p_text;
  metal_compute::ComputeFunction _fn_attn_main, _fn_attn_text;
  int _attn_seq = -1, _attn_text = -1;
  bool _steel_ok = false;
  // The matrix-core (NAX) flash attention. Same param block and function
  // constants as the ALU steel kernel; only the tile sizes differ, so the
  // two are interchangeable at the call site.
  bool _attn_nax = false;
};

}  // namespace genai
}  // namespace vpipe

#endif
