#ifndef GENERATIVE_MODELS_FLASHVSR_METAL_FLASHVSR_TRANSFORMER_H
#define GENERATIVE_MODELS_FLASHVSR_METAL_FLASHVSR_TRANSFORMER_H

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/shared/dit-block-progress.h"
#include "generative-models/shared/sage-attention.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

class WeightSet;            // generative-models/weight-set.h
class I8GemmContext;        // generative-models/shared/i8-gemm.h
class MetalSageAttention;   // generative-models/shared/metal-sage-attention.h
struct MmaSplitK;           // generative-models/shared/mma-splitk.h

// FlashVSR-v1.1 and FlashVSR+ (Turbo, Base): the one-step, streaming 4x
// video super-resolution DiT. The two releases are this one transformer
// driven by different schedules (see Schedule); the comments below speak
// in v1.1's numbers where the schedule decides them.
//
// The denoiser is Wan 2.1-T2V-1.3B's transformer, tensor for tensor: 30
// blocks at 1536 wide, 12 heads x head_dim 128, an UNGATED 8960 feed-
// forward, patch (1,2,2), 16 latent channels in and out, 3-axis RoPE.
// Everything this class does that MetalWanTransformer does not comes
// from how FlashVSR drives that transformer, and there are four such
// things. They are why this is its own class rather than a mode of the
// Wan one, whose forward is a whole clip at one timestep with a live
// text conditioning and no cache at all.
//
//   * THE SOURCE IS THE CONDITIONING, and it is not text. A learned
//     causal projection reads the bicubic-upscaled low-quality clip in
//     PIXELS at the output resolution and emits rows on this model's own
//     token grid, which block 0 adds into the residual stream. Nothing
//     is VAE-encoded on the way in, and there is no text encoder
//     anywhere in the family.
//
//     THAT PROJECTION LIVES IN THE CONDITIONER (flashvsr-lq-proj.h),
//     not here, and the reason is the port contract: its output is
//     [tokens, hidden] bf16, which is exactly what a conditioning beat
//     is. Running it there means the graph reads like every other
//     generative graph -- conditioner into generator -- and the shared
//     generate-video stage needs neither a second, pixel-shaped input
//     port nor an optional conditioning port. What arrives here is
//     `Request::rows`.
//
//   * IT DENOISES IN CHUNKS AGAINST A KV CACHE. One forward covers a
//     chunk of latent frames, not the clip, and attends over its own
//     tokens plus a window of previous chunks held per block -- v1.1's
//     chunks are 2 latent frames after an opening one of 6 with no cache
//     behind it, FlashVSR+'s are 4 throughout. So the unit of work is
//     a chunk and the loop that walks them lives HERE,
//     for the reason VOSR's flow loop lives in its transformer: the
//     chunking, the projection's own causal cadence and the cache
//     window are one mechanism, and a caller driving chunks would have
//     to own all three.
//
//   * ATTENTION IS BLOCK-SPARSE OVER 3-D WINDOWS. Tokens are permuted
//     into (2,8,8) windows of 128, each window scored by its mean
//     against every cached window, and a query window reads the top-k
//     of those unioned with a local spatial band. See lcsa_() for why
//     the permutation is what makes this the tree's existing span path
//     rather than a new kernel.
//
//   * THE TEXT CONDITIONING IS A CONSTANT, and so is the timestep.
//     FlashVSR runs at a fixed t=1000 over one fixed 512-token context
//     that ships with the checkpoint, with guidance distilled away. So
//     the timestep embedding, its six modulation vectors, and EVERY
//     block's cross-attention K and V are folded once at load and never
//     recomputed. That removes two projections per block from the
//     forward and leaves cross-attention with nothing to do but read a
//     constant, which is also why there is no text encoder anywhere in
//     this family.
//
// WHAT THIS CLASS IS NOT. It runs a whole clip per call and returns the
// whole latent. The model is built to stream, and streaming it through
// the pipeline a chunk at a time needs a continuation the video family
// contract does not express today; that is deliberately a later step,
// and nothing here forecloses it -- the chunk loop is already the shape
// such a driver would call into.
class MetalFlashVsrTransformer {
 public:
  // HOW A RELEASE DRIVES THE STACK. FlashVSR-v1.1 and FlashVSR+ ship the
  // same Wan 2.1-T2V-1.3B transformer, tensor for tensor, and differ only
  // in how they stream it -- which no tensor says, so the family sets it
  // from the checkpoint's layout. Every row is silent when wrong: the
  // other release's rule produces a latent of the right shape.
  //
  //                    v1.1                     FlashVSR+
  //   chunks           6 latent frames, then 2  4, every chunk
  //   top-k            ratio * 768*1280 / (HW)  ratio (2.5), unscaled
  //   top-k group      one temporal window      the whole chunk
  //   cache trim       one window per chunk     down to the kept count
  //   empty query row  left empty               its first candidate
  //   RoPE angles      double                   f32 product, as the
  //                                             reference computes them
  //   hybrid layers    --                       Turbo: 25 of 30 attend a
  //                                             3x3 band, keep one window
  struct Schedule {
    int  open_latents  = 6;     // the opening chunk's latent frames
    int  chunk_latents = 2;     // ...every later chunk's
    // Temporal windows (two latent frames each) one top-k group spans.
    int  group_windows = 1;
    bool area_topk     = true;
    bool trim_to_keep  = false;
    bool first_candidate = false;
    bool f32_rope      = false;
    // Turbo's hybrid layers, ascending; empty for v1.1 and Base.
    std::vector<int> sliding_layers;
    int  sliding_local_range = 3;
    int  sliding_keep        = 1;

    static Schedule v1_1() { return Schedule{}; }
    static Schedule plus(bool turbo);
    bool is_plus() const noexcept { return !area_topk; }
    bool sliding(int layer) const noexcept;
    // Latent frames a request of `chunks` holds, from the opening chunk.
    int latents(int chunks) const noexcept
    {
      return chunks > 0 ? open_latents + chunk_latents * (chunks - 1) : 0;
    }
    // Source frames the projection reads through chunk `c` (from 0):
    // the opening frame alone, then four per latent frame.
    long long frames_through(int c) const noexcept
    {
      return 1 + 4LL * latents(c + 1);
    }
    // Restored frames, as the decode numbers them, that chunk `c` ends
    // at: four short of what it read, the look-ahead.
    long long restored_end(int c) const noexcept
    {
      return frames_through(c) - 4;
    }
    long long restored_begin(int c) const noexcept
    {
      return c == 0 ? 0 : restored_end(c - 1);
    }
  };

  struct Config {
    int   n_heads      = 12;
    int   head_dim     = 128;
    int   hidden       = 1536;   // n_heads * head_dim
    int   ffn          = 8960;
    int   n_layers     = 30;
    int   in_channels  = 16;
    int   out_channels = 16;
    int   text_dim     = 4096;   // umT5-XXL width, for the fixed context
    int   text_tokens  = 512;    // and its fixed length
    int   freq_dim     = 256;
    int   patch_t      = 1;
    int   patch_h      = 2;
    int   patch_w      = 2;
    float norm_eps     = 1e-6f;
    double rope_theta  = 10000.0;
    // The constant this checkpoint was distilled at. Not a knob: the
    // modulation vectors baked at load are only correct for this one.
    float timestep     = 1000.0f;
    // How the release streams the stack; see Schedule. No tensor says
    // it, so config_from_checkpoint() leaves v1.1's and the family sets
    // it from the layout.
    Schedule schedule;

    // ---- acceleration: opt-in, independent, decided at load ----------
    //
    // The matrix-core kernels are NOT here: like every sibling DiT they
    // are on wherever the GPU has matrix cores, with VPIPE_FVSR_NO_MMA2
    // and VPIPE_FVSR_NO_ATTN_NAX as the A/Bs. These two are LOSSY and so
    // are the graph's choice. Sage rides on the NAX flash entry and is
    // dropped, with a log line, where that entry is unavailable.
    bool         i8_gemm = false;
    // Whether the kernel tuners MEASURE (the accel key `autotune`,
    // accel-keys.h): false takes the choice believed fastest on this
    // GPU, timing nothing.
    bool autotune = true;
    sage::Config sage;

    // ---- residency: decided at load, by the box ----------------------
    //
    // STREAM THE BLOCKS: keep only what the load FOLDS (each block's
    // modulation and cross-attention key/value, ~95 MB for the stack)
    // and read every block's weights from the checkpoint when it runs,
    // into two reused slots with the next block's read under the current
    // block's GPU work. Block-outer, a block runs once per CLIP, so each
    // is read once per clip. 2.84 GB of bf16 weights become ~0.3 GB. The
    // checkpoint is F32, so a read lands in a scratch and is narrowed
    // into its slot. VPIPE_FVSR_NO_SLOTS is the per-block-allocation A/B.
    bool stream_blocks = false;

    // The 3-axis RoPE split, derived as Wan derives it: h = w =
    // 2*(head_dim/6), t takes the remainder. 44/42/42 at 128.
    int rope_h() const { return 2 * (head_dim / 6); }
    int rope_w() const { return 2 * (head_dim / 6); }
    int rope_t() const { return head_dim - 2 * (2 * (head_dim / 6)); }
  };

  // What a caller chooses per generation. Separate from Config, which
  // is what the checkpoint IS.
  struct Params {
    // LCSA. `sparse_ratio` is the reference's own knob, in its own
    // release's terms: v1.1 scales it by the frame area exactly as its
    // scripts do, so the same value means the same selectivity at every
    // resolution (2.0; 1.5 is faster), while FlashVSR+ takes it as is
    // (2.5) -- defaults() picks the release's. `local_range` is the
    // always-exact spatial band in window blocks, 9 for sharper detail
    // and 11 for stability; Turbo's hybrid layers keep their own 3.
    // `kv_ratio` is the retained cache in temporal windows.
    double sparse_ratio = 2.0;
    double kv_ratio     = 3.0;
    int    local_range  = 11;

    static Params defaults(const Schedule& s)
    {
      Params p;
      if (s.is_plus()) { p.sparse_ratio = 2.5; }
      return p;
    }
  };

  // Read the geometry out of the checkpoint's own tensor shapes.
  //
  // FlashVSR ships no transformer config: its config.json names the
  // family and nothing else. Every field above except the head count is
  // recoverable from a shape, and the head count follows from Wan's
  // invariant head_dim of 128 -- which holds across every Wan config,
  // 1536/12 and 5120/40 alike. Deriving beats a table keyed on a
  // checkpoint hash, which is what the reference does and what silently
  // mis-loads the next release.
  static bool config_from_checkpoint(const std::string& dit_dir,
                                     Config*            out,
                                     std::string*       err = nullptr);

  // `dit_dir` holds the denoiser, the source projection and the fixed
  // context; they are three files of one checkpoint rather than three
  // checkpoints, so they share one WeightSet.
  static std::unique_ptr<MetalFlashVsrTransformer>
  load(std::shared_ptr<WeightSet>   ws,
       metal_compute::MetalCompute* mc,
       const Config&                cfg,
       std::string*                 err = nullptr);

  // ...with the fixed context read from a SEPARATE set, under its own
  // name. As published, the context is a file of its own
  // (`posi_prompt.pth`, tensor `posi_prompt`, [1, tokens, width]); a
  // converted directory carries it in the denoiser's index as
  // `posi_context` [tokens, width]. Both shapes are accepted, and the
  // token count is read off the tensor.
  static std::unique_ptr<MetalFlashVsrTransformer>
  load(std::shared_ptr<WeightSet>   ws,
       std::shared_ptr<WeightSet>   ctx_ws,
       const std::string&           ctx_name,
       metal_compute::MetalCompute* mc,
       const Config&                cfg,
       std::string*                 err = nullptr);

  ~MetalFlashVsrTransformer();

  // Frame counts a schedule can chunk: the opening frame, then four
  // source frames per latent frame, in whole chunks -- 8k+1 for v1.1
  // (which needs 25 before anything is restored) and 16k+1 for
  // FlashVSR+. A clip restores all but its last four frames, the
  // look-ahead. Rounding is UP for the reason every align rule here
  // rounds up: down silently delivers less video than was asked for.
  static int  align_frames(const Schedule& s, int frames);
  // Latent frames a clip yields. NOT 2 per chunk for v1.1: the opening
  // chunk emits six, so it is 2*chunks + 4.
  static int  latent_frames(const Schedule& s, int aligned_frames);
  static int  chunks_of(const Schedule& s, int aligned_frames);
  static int  restored_frames(const Schedule& s, int aligned_frames);
  // ...the same three for v1.1's schedule.
  static int  align_frames(int frames);
  static int  latent_frames(int aligned_frames);
  static int  restored_frames(int aligned_frames);
  // Output width and height must both be multiples of 128: the window
  // partition needs the token grid to divide by 8, and the grid is the
  // frame over 16.
  static constexpr int kSizeGrid = 128;

  // One clip.
  struct Request {
    // The source, ALREADY PROJECTED into rows by the conditioner:
    // bf16 [row_frames * tokens, hidden], one row frame per four source
    // frames and one token per cell of the denoiser's own grid.
    //
    // The projection lives in the conditioner and not here, which is
    // what lets the conditioning port carry exactly what it says it
    // carries -- rows at the resident family's own width -- instead of
    // this stage growing a second, pixel-shaped input. See
    // flashvsr-lq-proj.h.
    const void*         rows       = nullptr;
    int                 row_frames = 0;
    int                 frames  = 0;   // already through align_frames
    int                 height  = 0;   // already through the size grid
    int                 width   = 0;
    std::uint64_t       seed    = 0;
    // False ABORTS, per chunk and per block. A clip is many chunks and
    // a chunk is 30 blocks, so the block hook is what keeps a bar
    // moving and a Stop responsive inside one chunk.
    std::function<bool(int chunk, int total)> progress;
    std::function<bool(int done,  int total)> block_progress;
    // THE INITIAL NOISE, when a caller has one to pin. Two runtimes
    // cannot agree on a pseudo-random field, so a comparison against the
    // reference has to be handed the reference's draw; a real run leaves
    // this null and the seed decides. f32 [out_channels, T, H/8, W/8].
    const float* init_noise = nullptr;
    Params params;

    // ONE SOURCE OVER SEVERAL REQUESTS, restored as the reference
    // restores it: in one pass. `chunk0` is the stream's index of this
    // request's first chunk. 0 opens the stream -- the opening chunk is
    // this request's first, and `frames` (or `chunks`) says how many
    // follow. Above 0 the request CONTINUES the one before it: its rows
    // hold its own chunks only (Schedule::chunk_latents row frames each),
    // `chunks` says how many, and every block attends the key/value
    // window the previous request left. `keep_window` is what leaves one: each block's
    // window survives the call, for a request starting at the chunk this
    // one ended before.
    //
    // The result is the latent one request over the whole source would
    // have produced, bit for bit -- the window keeps its ring slots, the
    // noise is drawn per latent frame of the STREAM, and a continuation
    // takes the seed, geometry and params the stream opened with (a
    // mismatch is refused, not reconciled). Without this, every request
    // opens cold, which is a visible change of texture at every seam.
    int  chunk0 = 0;
    int  chunks = 0;           // 0 = from `frames` (an opening request)
    bool keep_window = false;
  };

  // Returns the whitened latent [16, T, H/8, W/8] that vae-decode takes,
  // T being the request's latent frames (Schedule::latents for an opening
  // one). False means it warned through `err` and produced nothing.
  bool generate(const Request& req, std::vector<float>* latent,
                std::vector<int>* shape, std::string* err = nullptr);

  // Bytes held: the weights, plus the per-block KV window, which GROWS
  // with the output size and is the larger term above roughly 512x512.
  // A caller that reports only the weights under-declares by more than
  // the weights at 768x1408.
  std::uint64_t resident_bytes() const noexcept;

  // The temporal windows layer `layer` keeps after a chunk: the ratio's
  // (Turbo's hybrid layers keep one whatever it says).
  static int keep_windows(const Config& cfg, int layer, double kv_ratio);
  // The ring that layer runs in: what it keeps and one chunk more, and
  // never fewer than the OPENING chunk writes before anything is
  // trimmed -- three windows for v1.1, two for FlashVSR+.
  static int layer_ring(const Config& cfg, int layer, double kv_ratio);
  // The largest ring of any layer, which is what one buffer serving the
  // stack is sized by.
  static int kv_windows(const Config& cfg, double kv_ratio);
  // ...for a clip of `frames`: a clip of ONE chunk never holds more than
  // that chunk's windows, whatever the ratio.
  static int ring_windows(const Config& cfg, int frames, double kv_ratio);
  // The same two for v1.1's schedule.
  static int kv_windows(double kv_ratio);
  static int ring_windows(int frames, double kv_ratio);

  // THE ORDER generate() runs the stack in, and the one memory decision
  // in this class. Chunk-outer (every block of a chunk, then the next
  // chunk) holds a key/value window PER BLOCK for the whole clip;
  // block-outer (every chunk of a block, then the next block) holds ONE,
  // and the residual stream and RoPE tables of every chunk instead. The
  // two compute the same function bit for bit. Block-outer is taken
  // whenever it holds less -- at 1920x1152, for any clip under ~1400
  // frames -- and VPIPE_FVSR_CHUNK_OUTER=1 forces the other (A/B).
  static bool blocks_outer(const Config& cfg, int height, int width,
                           int frames, const Params& params);

  // ONE block's key/value window at this geometry: a K and a V over
  // `windows` temporal windows, bf16. 425 MB at 1920x1152 and 4 windows.
  static std::uint64_t kv_ring_bytes(const Config& cfg, int height,
                                     int width, int windows);

  // What a STREAM holds between requests (Request::keep_window): every
  // block's retained windows, the ring's capacity less the slot the next
  // chunk writes. 9.6 GB at 1920x1152 and kv_ratio 3 -- the price of a
  // seamless stream, and why a box that cannot hold it restores clips.
  static std::uint64_t stream_window_bytes(const Config& cfg, int height,
                                           int width, const Params& params);
  // The chunk a held window continues into, or -1 when none is held.
  int  window_next_chunk() const noexcept;
  // Let a held window go (the stream ended, or will not continue here).
  void drop_window() noexcept;

  // What one clip's key/value windows ALLOCATE: one ring, or one per
  // block, as blocks_outer() decides. Was the largest term a denoise
  // holds above ~512 px -- 12.7 GB at 1920x1152 and kv_ratio 3, chunk-
  // outer -- and is 425 MB block-outer.
  static std::uint64_t kv_window_bytes(const Config& cfg, int height,
                                       int width, int frames,
                                       const Params& params);

  // What a STREAMING load holds of the weights (Config::stream_blocks):
  // what the load folds, the slot pair and the narrowing scratch. ~0.3 GB
  // for the 1.3B, against 2.84 GB resident.
  static std::uint64_t streaming_floor_bytes(const Config& cfg);

  // Everything a generate() of this geometry allocates beyond the
  // weights: the windows above, the opening chunk's activation scratch
  // (the largest chunk: three windows), the routing and span buffers, the
  // residual streams and RoPE tables the order keeps, and the clip's
  // noise and latent. For the memory plan, so it is sized for the ALU
  // entry's finer span list, which is the larger of the two; the opt-in
  // int8 and Sage scratches are not in it.
  static std::uint64_t denoise_scratch_bytes(const Config& cfg, int height,
                                             int width, int frames,
                                             const Params& params);

  const Config& config() const noexcept { return _cfg; }

  // WHICH PATHS THIS LOAD SELECTED. Every one of them falls back to the
  // ALU kernel -- or to the f16 product -- without changing the shape of
  // the answer, so a silent fallback is invisible to a numerical test and
  // costs a multiple of the run time. These are for a path-selection
  // check, and for the one log line that says what a graph actually got.
  bool uses_matrix_cores() const noexcept { return _use_mma2; }
  bool uses_attn_nax() const noexcept { return _use_attn_nax; }
  bool uses_i8_gemm() const noexcept;
  bool uses_sage() const noexcept;
  // Whether this load streams its blocks (Config::stream_blocks), and
  // how many block reads the last generate() found already prefetched.
  bool streams_blocks() const noexcept;
  int prefetch_hits() const noexcept;
  std::string accel_summary() const;

  // Test-only views of what the load FOLDED. They exist because these
  // are constants computed once and then never recomputed, so a mistake
  // in them is invisible for the rest of the run: it shows up as a
  // slightly wrong picture, never as a failure.
  const std::vector<float>& t_vec_f()    const noexcept { return _t_vec_f; }
  const std::vector<float>& t_mod_f()    const noexcept { return _t_mod_f; }
  const std::vector<float>& head_mod_f() const noexcept { return _head_mod_f; }
  // The cross-attention key and value of one block, un-bf16'd.
  bool cross_kv(int block, std::vector<float>* k,
                std::vector<float>* v) const;
  // Run the patch embedding alone over one latent chunk, channel-first
  // f32 [in_channels, T, h8, w8], and return the tokens it produces.
  // A seam rather than a step of a longer call because the PACK ORDER
  // it encodes is silent when wrong -- and is deliberately not the
  // mirror of the un-patchify at the other end of the stack.
  bool patchify_tokens(const float* latent, int T, int h8, int w8,
                       std::vector<float>* tokens, std::string* err);
  // One block's SELF-ATTENTION over one chunk's tokens, cacheless. The
  // seam where every mechanism this model adds meets at once.
  bool block_self_attention(int block, const float* x_tokens, int f, int h,
                            int w, int t_off, const Params& params,
                            std::vector<float>* out, std::string* err);
  // The routing decision for one chunk, computed on the GPU and on the
  // host, with the number of flags that disagree. See the definition:
  // they cannot be bit-identical, and the count is the point.
  bool route_compare(const float* x_tokens, int f, int h, int w, int t_off,
                     const Params& params, std::size_t* n_flags,
                     std::size_t* n_diff, std::string* err);
  // The block-level routing decision the production path makes for block
  // `bi` over one cacheless chunk: [H, Nq, Nk] bytes, 1 = attended -- the
  // reference's own block mask, comparable entry for entry.
  bool route_blocks(int bi, const float* x_tokens, int f, int h, int w,
                    int t_off, const Params& params,
                    std::vector<std::uint8_t>* keep, std::string* err);
  // One chunk through the FIRST `n_blocks` blocks, returning the residual
  // stream. The seam that lets a failure be attributed to a block rather
  // than to the stack.
  bool run_chunk_blocks(int n_blocks, const float* tokens,
                        const float* lq_rows, int f, int h, int w, int t_off,
                        const Params& params, std::vector<float>* out,
                        std::string* err);

 private:
  MetalFlashVsrTransformer() = default;

  // Every layer's own ring at this geometry, summed: what the chunk-
  // outer order allocates.
  static std::uint64_t kv_rings_bytes_(const Config& cfg, int height,
                                       int width, int frames,
                                       const Params& params);

  struct Block;
  struct Stream;   // the per-block KV window, live for one generate()
  struct Streaming;   // the slot pair a streaming load reads blocks into
  struct WindowCarry;   // every block's window between stream requests

  // Copy block `bi`'s retained windows between the ring `cache_idx` and
  // the carry: INTO the ring before a continuing request runs the block,
  // OUT of it after a request that keeps its window. Slots are kept, so
  // the attention reads the keys in the order one request would have.
  bool carry_window_(Stream& st, int cache_idx, int bi, bool into_ring,
                     std::string* err);

  bool fold_constants_(WeightSet& w, std::string* err);
  // The three-axis RoPE tables for one chunk, built in WINDOW-PERMUTED
  // order and at an ABSOLUTE temporal offset -- see the definition for
  // why both of those matter.
  void build_rope_(int f_len, int h, int w, int t_off,
                   metal_compute::SharedBuffer& cos_out,
                   metal_compute::SharedBuffer& sin_out) const;
  bool load_blocks_(WeightSet& w, std::string* err);
  // The routing decision, on the host from GPU-reduced block means.
  std::vector<std::uint8_t>
  // `first_win`, when not -1, is the ring slot of the oldest window in
  // the key order: a row the top-k left empty keeps its first candidate
  // there (FlashVSR+).
  route_(const std::vector<float>& qm, const std::vector<float>& km, int Nq,
         int Nk, int sq, int nh_blk, int nw_blk, int topk, int local_range,
         int kL, int bq, int first_win = -1) const;
  int _seq_for_route = 0;   // the query length route_ is expanding for
  // The routing's knobs, as this release spells them: the top-k at a
  // geometry, layer `bi`'s band, and the top-k groups of an f-frame
  // chunk.
  int topk_(int S, int h, int w, const Params& prm) const;
  int local_range_(int bi, const Params& prm) const;
  int route_groups_(int f) const;
  void build_band_(int nh_blk, int nw_blk, int local_range,
                   metal_compute::SharedBuffer& out) const;
  void route_gpu_(metal_compute::ComputeEncoder& enc,
                  const metal_compute::SharedBuffer& qm,
                  const metal_compute::SharedBuffer& km,
                  const metal_compute::SharedBuffer& band,
                  const metal_compute::SharedBuffer& p,
                  const metal_compute::SharedBuffer& thr,
                  const metal_compute::SharedBuffer& flags, int Nq, int Nk,
                  int sq, int S, int topk, int kL, int NQ, int bq);

  Config                       _cfg{};
  metal_compute::MetalCompute* _mc = nullptr;
  std::shared_ptr<WeightSet>   _ws;
  std::shared_ptr<WeightSet>   _ctx_ws;    // == _ws when converted
  std::string                  _ctx_name = "posi_context";
  std::vector<Block>           _blocks;
  std::unique_ptr<Streaming>   _streaming;   // set when stream_blocks
  std::unique_ptr<WindowCarry> _carry;       // set while a stream is open

  // ---- folded at load, because the timestep and the context are both
  // constants of this checkpoint. See the class comment.
  metal_compute::ComputeLibrary  _lib_elt, _lib_gemm, _lib_rms, _lib_rope,
                                 _lib_attn;
  metal_compute::ComputeFunction _fn_pack, _fn_unpack, _fn_window,
                                 _fn_gemm, _fn_rms, _fn_ln_mod, _fn_ln_aff,
                                 _fn_transpose, _fn_blockmean, _fn_gelu,
                                 _fn_gated, _fn_resadd, _fn_trope, _fn_copy,
                                 _fn_sdpa, _fn_gemm_bm64, _fn_gemm_bn64;
  int _gemm_tile = 1;
  bool _dense_attn = false;   // measurement only; see the .cc
  metal_compute::ComputeLibrary  _lib_sdpa, _lib_route;
  metal_compute::ComputeFunction _fn_rt_scores, _fn_rt_softmax,
                                 _fn_rt_thresh, _fn_rt_flags, _fn_rt_keep,
                                 _fn_sp_off, _fn_sp_emit, _fn_rt_fallback;
  bool _use_spans = true;
  // Query tiles routed to no key block during the last generate(); see
  // forward_chunk_.
  std::uint64_t _empty_tiles = 0;

 public:
  std::uint64_t empty_query_tiles() const noexcept { return _empty_tiles; }

 private:
  // The span list, from the block-level keep map. `first_win` >= 0 also
  // gives a row the top-k left empty its first candidate (FlashVSR+),
  // in the window at that ring slot, inside `band`.
  void build_spans_(metal_compute::ComputeEncoder& enc, Stream& st,
                    const metal_compute::SharedBuffer& band, int Nq, int Nk,
                    int sq, int NQ, int bq, int first_win);
  // Size the clip's state: `n_caches` key/value rings, each layer's for
  // the chunks after the first too when `full` (a stream, or a clip of
  // more than one chunk), and every per-chunk buffer for the largest
  // chunk, `f_max` latent frames.
  bool begin_clip_(Stream& st, int h, int w, int f_max, bool full,
                   int n_caches, const Params& prm, std::string* err);
  // The RoPE tables of a chunk, in `st.rope[slot]`, rebuilt only when the
  // slot held another chunk's.
  bool rope_(Stream& st, int slot, int f, int t_off);
  // Block `bi` over one chunk's residual stream, in place, against key/
  // value window `cache_idx`. The unit both orders are built from.
  // `wb` is the block's WEIGHTS -- a slot when streaming, null for the
  // resident block; what the load folded is always read from _blocks.
  bool block_(Stream& st, int bi, const Block* wb, int cache_idx, int f,
              int t_off, int rope_slot, metal_compute::SharedBuffer& x,
              const metal_compute::SharedBuffer* lq, const Params& prm,
              std::string* err);
  // The weights block `bi` runs on: null when resident, a slot when
  // streaming -- with block `next` read ahead into the other slot.
  bool acquire_block_(int bi, int next, const Block** out,
                      std::string* err);
  void configure_streaming_();
  bool head_(Stream& st, const metal_compute::SharedBuffer& x, int seq,
             metal_compute::SharedBuffer* noise_out, std::string* err);
  void report_prof_(Stream& st, const char* what, int n_blocks);
  // The block-outer clip loop; see blocks_outer().
  // The two orders over one request's `chunks`: `T` latent frames from
  // stream frame `g0`, chunk `ci`'s size and stream offset from the two
  // functions. generate() has already settled all of that.
  bool generate_blocks_outer_(const Request& req,
                              const std::vector<float>& noise, Stream& st,
                              int chunks, int T, int g0,
                              const std::function<int(int)>& chunk_f,
                              const std::function<int(int)>& chunk_t,
                              std::vector<float>* latent, std::string* err);
  bool generate_chunks_outer_(const Request& req,
                              const std::vector<float>& noise, Stream& st,
                              int chunks, int T, int g0,
                              const std::function<int(int)>& chunk_f,
                              const std::function<int(int)>& chunk_t,
                              std::vector<float>* latent, std::string* err);
  bool ensure_kernels_(std::string* err);

  // ---- matrix cores and the two opt-in tiers, decided at load --------
  //
  // THE TILES FOLLOW THE KERNEL. The NAX flash entry tiles queries by 64
  // and keys by 32 where the ALU one tiles 32 and 16, and everything this
  // model sizes in attention tiles -- the routing flags, the CSR offsets,
  // the span list's key-block unit -- has to be derived from the entry
  // that will read it. 128 divides all four, so the conversion from
  // routing windows stays exact either way.
  int attn_bq_() const noexcept { return _use_attn_nax ? 64 : 32; }
  int attn_bk_() const noexcept { return _use_attn_nax ? 32 : 16; }
  metal_compute::ComputeFunction
  attn_fn_(const metal_compute::FunctionConstants& fc);
  // The matmul2d route for one GEMM, bias included; false with nothing
  // encoded when the shape does not qualify.
  bool gemm_mma_(metal_compute::ComputeEncoder& enc,
                 const metal_compute::SharedBuffer& x,
                 const metal_compute::SharedBuffer& w,
                 const metal_compute::SharedBuffer& b,
                 const metal_compute::SharedBuffer& y, int M, int N, int K);
  bool _use_mma2 = false, _use_attn_nax = false;
  metal_compute::ComputeLibrary  _lib_dense_mma, _lib_attn_nax;
  metal_compute::ComputeFunction _fn_gemm_mma, _fn_gemm_mma_wide,
                                 _fn_bias_rows;
  std::unique_ptr<MmaSplitK>     _splitk;
  std::unique_ptr<I8GemmContext> _i8;
  // TWO Sage objects, one per attention SHAPE. A Sage scratch is rebuilt
  // whenever the geometry it was sized for changes, and within one block
  // the self-attention (kL keys) and the cross-attention (the 512-token
  // context) alternate -- so a shared object would reallocate its int8
  // operands twice per block, sixty times a chunk.
  std::unique_ptr<MetalSageAttention> _sage, _sage_x;
  // ...and the same two for the OPENING chunk's shape, which block-outer
  // runs inside every block beside the steady one.
  std::unique_ptr<MetalSageAttention> _sage_open, _sage_x_open;

  struct FwdCtxFwd;   // unused placeholder; see the .cc's FwdCtx
  void gemm_(metal_compute::ComputeEncoder& enc,
             const metal_compute::SharedBuffer& x,
             const metal_compute::SharedBuffer& w,
             const metal_compute::SharedBuffer& b,
             const metal_compute::SharedBuffer& y, int M, int N, int K);
  void rms_(metal_compute::ComputeEncoder& enc,
            const metal_compute::SharedBuffer& x,
            const metal_compute::SharedBuffer& g,
            const metal_compute::SharedBuffer& y, int rows, int C);
  void permute_(metal_compute::ComputeEncoder& enc,
                const metal_compute::SharedBuffer& x,
                const metal_compute::SharedBuffer& y, int f, int h, int w,
                int D, bool inverse);
  void trope_(metal_compute::ComputeEncoder& enc,
              const metal_compute::SharedBuffer& x,
              const metal_compute::SharedBuffer& y,
              const metal_compute::SharedBuffer& rcos,
              const metal_compute::SharedBuffer& rsin, int NH, int seq,
              int HD);
  void blockmean_(metal_compute::ComputeEncoder& enc,
                  const metal_compute::SharedBuffer& x,
                  const metal_compute::SharedBuffer& y, int HD, int seq,
                  int blk, int NH, int nb);
  void append_cache_(metal_compute::ComputeEncoder& enc,
                     const metal_compute::SharedBuffer& src,
                     const metal_compute::SharedBuffer& dst, int NH, int seq,
                     int HD, std::size_t cap_tok, int held, int src_tok0,
                     int n_tok);
  // One chunk through the whole stack, then the head: the chunk-outer
  // order, and the test seams' path.
  bool forward_chunk_(Stream& st, int f, int h, int w, int t_off,
                      int chunk_idx,
                      const metal_compute::SharedBuffer& x_in,
                      const metal_compute::SharedBuffer* lq,
                      const Params& params,
                      metal_compute::SharedBuffer* noise_out,
                      const Request& req, int max_blocks,
                      std::vector<float>* x_out, std::string* err);
  // Trim the ring `cache_idx`, which layer `bi` is running in. Takes the
  // Stream and an index rather than the Cache itself: the nested type is
  // not complete here.
  void trim_cache_(Stream& st, int cache_idx, int bi, const Params& params);
  bool unpatchify_(const metal_compute::SharedBuffer& tokens, int T, int h8,
                   int w8, std::vector<float>* out, std::string* err);

  // The patch embedding, flattened to [hidden, in_channels*patch_vol] --
  // which is already its memory order -- and the output head.
  metal_compute::SharedBuffer  _patch_w, _patch_b;
  metal_compute::SharedBuffer  _head_w, _head_b;

  metal_compute::SharedBuffer  _t_vec;    // [hidden]
  metal_compute::SharedBuffer  _t_mod;    // [6, hidden]
  metal_compute::SharedBuffer  _head_mod; // [2, hidden]
  // The same three in f32, kept because the per-block modulation is
  // folded against them and because a test can then check the fold
  // rather than the tensor it was folded into.
  std::vector<float>           _t_vec_f, _t_mod_f, _head_mod_f;
  // The conditioning after text_embedding, which every block's cross
  // key and value are projected from. Held only for the load.
  std::vector<float>           _ctx_f;
};

}  // namespace genai
}  // namespace vpipe

#endif
