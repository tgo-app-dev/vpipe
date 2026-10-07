#ifndef GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_TAOMATE_H
#define GENERATIVE_MODELS_MINIMAX_H3_MINIMAX_H3_TAOMATE_H

#include "apple-silicon/metal-compute/shared-buffer.h"
#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {
class MetalMiniMaxH3Transformer;
}  // namespace genai
}  // namespace vpipe

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace taomate {

// TaoMate-H3 (Alibaba TaoLive AIGC): streaming text-to-audio-video on
// MiniMax-H3's FL2VA weights with a 3-step LoRA.
//
// The adapter is NOT a drop-in few-step LoRA, and running it as one is a
// different method from the one it was distilled for. What upstream's
// runtime does, request by request (one request = 5 seconds):
//
//   1. AUDIO comes from the BASE model. A 9-forward, audio-only T2A run
//      (no video rows at all; the second request on is anchored on the
//      previous request's last second of clean audio, as frozen
//      reference rows) keeps its states 3, 6 and 9.
//   2. VIDEO comes from the LoRA, in four CHUNKS (2, 2, 2, 1 groups of
//      17 frames; the first request also owns the 5-frame prefix). Each
//      chunk runs THREE denoising steps -- states 0 -> 16 -> 33 -> 49 of
//      a 50-point shifted schedule -- and after step k its audio rows
//      are OVERWRITTEN with the teacher's state 3(k+1), so the published
//      audio is the base model's exactly. The video's attention:
//        text rows      -> text only
//        current media  -> [text ; cached clean media ; current media]
//   3. A fourth, CLEAN forward per chunk (t = 1, renormalised video +
//      clean audio) writes that chunk's post-RoPE keys and values into
//      the cache. The cache keeps chunk 0's VIDEO for the whole run (the
//      sink) and the two most recent chunks whole; every 12 requests its
//      audio rows are dropped.
//
// Everything on this page is host arithmetic -- the plan, the packed
// layouts with their positions on the run's global timeline, the
// schedule, the noise, the renormalisation -- and each is a port of the
// upstream function named beside it, held to its numbers by the
// `minimax_h3_taomate.*` tests.

// ---- the validated contract (upstream config.DirectPolicy) ----------
//
// Not knobs. Upstream: "changing one of them creates a different
// inference method and must be evaluated separately."
constexpr int kRequestFrames        = 124;   // native frames, request 0
constexpr int kPrefixFrames         = 5;
constexpr int kPrefixLatents        = 2;
constexpr int kGroupFrames          = 17;
constexpr int kGroupLatents         = 5;
constexpr int kChunkGroups[4]       = {2, 2, 2, 1};
constexpr int kRequestVideoLatents  = 37;    // a whole native request
constexpr int kRequestAudioLatents  = 207;   // per channel
constexpr int kGridPoints           = 50;
constexpr int kStateIndices[4]      = {0, 16, 33, 49};
constexpr int kTeacherGridPoints    = 10;    // 9 forwards
constexpr int kTeacherStates[3]     = {3, 6, 9};
constexpr int kRolloverLatents      = 40;    // 1 s of the previous audio
constexpr int kAudioKvResetRequests = 12;
constexpr int kTransportVideoLatents = 2;    // a continuation's VAE prefix
constexpr std::uint64_t kVideoNoiseStride = 1000003;
constexpr double kVideoShift = 12.0;
constexpr double kAudioShift = 3.0;
constexpr int kVideoFps = 24;
constexpr int kAudioLatentRate = 40;

// ---- the plan (upstream streaming/geometry.py) ----------------------

struct Phase {
  int index = 0;
  int group_count = 0;
  int frame_start = 0, frame_stop = 0;
  int video_latent_start = 0, video_latent_stop = 0;
  int audio_latent_start = 0, audio_latent_stop = 0;   // per channel

  int video_latents() const { return video_latent_stop - video_latent_start; }
  int audio_latents() const { return audio_latent_stop - audio_latent_start; }
};

struct Plan {
  int native_frames = 0;
  std::vector<Phase> phases;

  int video_latents() const
  {
    return phases.empty() ? 0 : phases.back().video_latent_stop;
  }
  int audio_latents() const
  {
    return phases.empty() ? 0 : phases.back().audio_latent_stop;
  }
};

// round_half_even(frame * 40 / 24): an audio boundary on the global frame
// timeline.
int audio_latent_boundary(std::int64_t frame);

// Request 0 is direct_5s_plan(): 124 frames, 37 video latents, 207 audio
// latents. Every later one is canonical_continuation_plan(): no 5-frame
// prefix -- 7 groups, 119 frames, 35 latents -- and its audio boundaries
// rounded on the GLOBAL frame timeline, so it is 198 or 199 latents.
Plan request_plan(int request_index);

// ---- positions ------------------------------------------------------
//
// The rotary time of global video latent `n`, in audio-latent units
// (1/40 s), as an exact multiple of 1/3: 5 * sum_{k<n} w[k % 5] with
// w = (1, 4, 4, 4, 4). Kept as an integer numerator so a position is
// rounded ONCE, the way upstream's Fraction arithmetic rounds it.
std::int64_t video_time_thirds(std::int64_t latent_index);

// One chunk's packed layout on the run's global timeline
// (build_streaming_phase_packed_layout): [text | audio | video], the
// prompt right-aligned against the request's media origin.
//
//   media_origin         text length of REQUEST 0, fixed for the run
//   video_latent_offset  global latents before this request
//   audio_latent_offset  global audio latents (per channel) before it
bool build_chunk_layout(const std::vector<int>& text_tags, const Phase& phase,
                        int latent_h, int latent_w, int media_origin,
                        int video_latent_offset, int audio_latent_offset,
                        PackedLayout* out);

// The teacher's audio-only layout: [text | reference audio | target
// audio], each audio block channel-major. `ref_latents` 0 is request 0
// (minimax_h3_audio_only_packed_sequence: audio from t = text length);
// otherwise the reference rows are the previous request's last
// `ref_latents` clean latents at `ref_time_start`, and the target starts
// at `target_time_start` (..._frozen_prefix_packed_sequence). The
// reference rows are CONDITION audio rows (num_condition_audio_rows),
// which is what pins them at t = 1. The spatial grid -- the width
// extremes the two channels sit at -- is the video canvas's, though no
// video row is present.
bool build_teacher_layout(const std::vector<int>& text_tags,
                          int ref_latents, int audio_latents, int latent_h,
                          int latent_w, int ref_time_start,
                          int target_time_start, PackedLayout* out);

// ---- schedule (upstream denoise_schedule.py) ------------------------
//
// The chunk's RAW grid: linspace(1, 0, 50) at kStateIndices, for
// MiniMaxH3Scheduler::set_sigmas under each modality's shift. The
// teacher's is set_timesteps(kTeacherGridPoints).
std::vector<float> chunk_raw_grid();

// ---- noise (upstream pipeline.generate / teacher._official_audio_noise)

// (request_seed + request_index * 1000003) mod 2^63: each request's
// video draws from its own substream, so a prompt boundary cannot replay
// the previous request's noise.
std::uint64_t video_noise_seed(std::uint64_t request_seed, int request_index);

// torch.randn(1, 24, T, H, W) from `seed`, patchified (1, 2, 2) into
// [T * H/2 * W/2, 96] rows, element (c * 2 + y) * 2 + x.
std::vector<float> video_noise_rows(std::uint64_t seed, int channels,
                                    int latent_t, int latent_h, int latent_w);

// The OFFICIAL audio noise: the full request's video draw is made and
// DISCARDED, then torch.randn(2 * 207, 32) -- channel-major rows. Its
// identity depends on the canvas through the discarded draw.
std::vector<float> audio_noise_rows(std::uint64_t seed, int video_channels,
                                    int latent_h, int latent_w,
                                    int audio_channels);

// ---- renormalisation (H3StreamingRuntime._renorm_clean_video_rows) --
//
// Every chunk's clean video is affinely matched, per channel, to the
// FIRST chunk's statistics: mean and population std over the chunk's
// rows, std clamped to 1e-6. The first call records the anchor and
// leaves its rows alone.
class VideoRenorm {
public:
  void apply(float* rows, int num_rows, int channels);
  bool anchored() const { return !_mean.empty(); }
  void reset() { _mean.clear(); _std.clear(); }

private:
  std::vector<float> _mean, _std;
};

// Rows [start, stop) of each channel of a channel-major [channels * T,
// C] audio block, as a new channel-major block.
std::vector<float> audio_slice(const float* rows, int total_latents,
                               int start, int stop, int channels, int width);

// ---- recognising the adapter ----------------------------------------
//
// A TaoMate adapter, from its safetensors HEADER alone: the bare
// lowercase `lora_a` spelling on the fused qkv of both the main stack and
// the token refiner, and no AdaLN factors. Nothing in the file names the
// method -- its metadata says only `weight_source` / `optimizer_step` --
// so the inventory is the signature. A generate-video run whose adapter
// slot holds one runs this method instead of the ordinary denoise.
bool is_adapter(const std::string& safetensors_path);

// ---- the driver -----------------------------------------------------

// One 5-second request: its prompt and its seed.
struct Request {
  // [n_text, text_dim] bf16 -- the conditioner's tap, as any H3 forward
  // takes it -- and its rows' tags (text 1 for a plain prompt).
  const metal_compute::SharedBuffer* text = nullptr;
  std::vector<int> text_tags;
  // The request's VIDEO noise seed before the per-request stride
  // (upstream: the prompt file's per-request seed, else the run seed).
  std::uint64_t seed = 0;
};

struct RunConfig {
  int latent_h = 0;              // pixels / 16
  int latent_w = 0;
  // The run seed: the AUDIO teacher draws request r's noise from
  // audio_seed + r (upstream --audio-seed-base, which is the run seed).
  std::uint64_t audio_seed = 8301;
  // The adapter's slot on the DiT, and its strength for the video. The
  // teacher runs the base model -- this slot at scale 0 -- whatever any
  // OTHER slot is set to.
  int   lora_slot  = 0;
  float lora_scale = 1.0f;
  // Progress in forwards (done, total); return false to stop.
  std::function<bool(int done, int total)> progress;
  // One line per milestone of the run, for the host's log.
  std::function<void(const std::string&)> log;
};

struct RunResult {
  // The whole published timeline, transport prefixes already removed:
  // 37 + 35 (R - 1) video latents of [lh/2 * lw/2] patchified rows of 96,
  // and the audio as channel-major [2 * audio_latents, 32] rows -- the
  // shapes H3's own denoise hands back, so the caller unpatchifies and
  // transposes them the same way.
  std::vector<float> video_rows;
  int video_latents = 0;
  std::vector<float> audio_rows;
  int audio_latents = 0;
  int forwards = 0;              // teacher + chunk + commit
  bool stopped = false;
  std::size_t kv_bytes = 0;      // what the clean-KV cache held
};

// Rows the clean-KV cache must hold for this canvas over a run of any
// length (the first chunk whole plus the two largest chunks), and the
// longest chunk sequence for a prompt of `max_text` rows.
int kv_capacity_rows(int latent_h, int latent_w);
int max_chunk_seq(int latent_h, int latent_w, int max_text);

// What a run holds on the GPU besides the model: the clean-KV cache and
// its staging, for `layers` blocks of `heads` x `head_dim`.
std::size_t kv_cache_bytes(int latent_h, int latent_w, int max_text,
                           int layers, int heads, int head_dim);

// Run the method over `requests` on `dit`, whose slot `cfg.lora_slot`
// holds the TaoMate adapter. False with a reason on failure; a STOP
// returns false with `out->stopped` set and no reason.
bool run(MetalMiniMaxH3Transformer* dit, const RunConfig& cfg,
         const std::vector<Request>& requests, RunResult* out,
         std::string* err);

}  // namespace taomate
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe

#endif
