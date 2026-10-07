#include "generative-models/minimax-h3/minimax-h3-taomate.h"

#include "common/vpipe-format.h"
#include "generative-models/minimax-h3/metal-minimax-h3-transformer.h"
#include "generative-models/minimax-h3/minimax-h3-scheduler.h"
#include "generative-models/minimax-h3/minimax-h3-stream-kv.h"
#include "generative-models/shared/runtime-lora.h"
#include "generative-models/shared/torch-randn.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

namespace vpipe {
namespace genai {
namespace minimax_h3 {
namespace taomate {

int
audio_latent_boundary(std::int64_t frame)
{
  // Fraction(frame * 40, 24), rounded half to even -- Python's round().
  const std::int64_t num = frame * kAudioLatentRate;
  const std::int64_t den = kVideoFps;
  const std::int64_t q = num / den;
  const std::int64_t r = num % den;
  const std::int64_t doubled = 2 * r;
  if (doubled < den) { return (int)q; }
  if (doubled > den) { return (int)(q + 1); }
  return (int)(q + (q & 1));
}

Plan
request_plan(int request_index)
{
  Plan p;
  if (request_index <= 0) {
    // direct_5s_plan: the first chunk also owns the 5-frame, 2-latent
    // prefix every standalone H3 clip starts with.
    p.native_frames = kRequestFrames;
    int frame_stop = 0, video_stop = 0;
    for (int i = 0; i < 4; ++i) {
      Phase ph;
      ph.index = i;
      ph.group_count = kChunkGroups[i];
      ph.frame_start = frame_stop;
      ph.video_latent_start = video_stop;
      frame_stop += kChunkGroups[i] * kGroupFrames;
      video_stop += kChunkGroups[i] * kGroupLatents;
      if (i == 0) {
        frame_stop += kPrefixFrames;
        video_stop += kPrefixLatents;
      }
      ph.frame_stop = frame_stop;
      ph.video_latent_stop = video_stop;
      ph.audio_latent_start = audio_latent_boundary(ph.frame_start);
      ph.audio_latent_stop = audio_latent_boundary(ph.frame_stop);
      p.phases.push_back(ph);
    }
    return p;
  }
  // canonical_continuation_plan: a continuation does not regenerate the
  // prefix, it advances seven native groups, and its audio is cut on the
  // GLOBAL frame timeline -- 198 or 199 latents, not a repeated 207.
  const int steady = kRequestFrames - kPrefixFrames;
  const std::int64_t gfs =
      (std::int64_t)kRequestFrames + (std::int64_t)(request_index - 1) * steady;
  const int gas = audio_latent_boundary(gfs);
  p.native_frames = steady;
  int frame_stop = 0, video_stop = 0;
  for (int i = 0; i < 4; ++i) {
    Phase ph;
    ph.index = i;
    ph.group_count = kChunkGroups[i];
    ph.frame_start = frame_stop;
    ph.video_latent_start = video_stop;
    frame_stop += kChunkGroups[i] * kGroupFrames;
    video_stop += kChunkGroups[i] * kGroupLatents;
    ph.frame_stop = frame_stop;
    ph.video_latent_stop = video_stop;
    ph.audio_latent_start = audio_latent_boundary(gfs + ph.frame_start) - gas;
    ph.audio_latent_stop = audio_latent_boundary(gfs + ph.frame_stop) - gas;
    p.phases.push_back(ph);
  }
  return p;
}

std::int64_t
video_time_thirds(std::int64_t n)
{
  if (n <= 0) { return 0; }
  const std::int64_t full = n / 5, rem = n % 5;
  // One full group of five latents spans 1 + 4 + 4 + 4 + 4 = 17 frames.
  const std::int64_t s = full * 17 + (rem > 0 ? 1 + 4 * (rem - 1) : 0);
  return 5 * s;
}

bool
build_chunk_layout(const std::vector<int>& text_tags, const Phase& phase,
                   int latent_h, int latent_w, int media_origin,
                   int video_latent_offset, int audio_latent_offset,
                   PackedLayout* out)
{
  if (out == nullptr || phase.video_latents() <= 0 ||
      phase.audio_latents() <= 0) {
    return false;
  }
  PackedLayout L;
  if (!build_packed_sequence(text_tags, phase.video_latents(), latent_h,
                             latent_w, phase.audio_latents(), 2, 2,
                             kAudioChannels, {}, &L)) {
    return false;
  }
  auto t_of = [&](int row) -> double& {
    return L.position_ids[(std::size_t)row * 3];
  };
  const int n_text = L.num_text_rows;
  // The prompt sits right-aligned against the REQUEST's media start:
  // float(origin + video_time(offset) - text_len), rounded once, then
  // + i in double.
  const double prompt_start =
      (double)(3 * (std::int64_t)(media_origin - n_text) +
               video_time_thirds(video_latent_offset)) / 3.0;
  for (int i = 0; i < n_text; ++i) { t_of(i) = prompt_start + (double)i; }
  // Video: float(origin + video_time(global latent)), each rounded once.
  const int frame_rows = (latent_h / 2) * (latent_w / 2);
  for (int f = 0; f < phase.video_latents(); ++f) {
    const std::int64_t g =
        (std::int64_t)video_latent_offset + phase.video_latent_start + f;
    const double t =
        (double)(3 * (std::int64_t)media_origin + video_time_thirds(g)) / 3.0;
    for (int r = 0; r < frame_rows; ++r) {
      t_of(L.video_start + f * frame_rows + r) = t;
    }
  }
  // Audio: origin + global latent index, both channels.
  const int na = phase.audio_latents();
  for (int c = 0; c < kAudioChannels; ++c) {
    for (int j = 0; j < na; ++j) {
      t_of(L.audio_start + c * na + j) =
          (double)media_origin + (double)audio_latent_offset +
          (double)(phase.audio_latent_start + j);
    }
  }
  *out = std::move(L);
  return true;
}

bool
build_teacher_layout(const std::vector<int>& text_tags, int ref_latents,
                     int audio_latents, int latent_h, int latent_w,
                     int ref_time_start, int target_time_start,
                     PackedLayout* out)
{
  if (out == nullptr || audio_latents <= 0 || ref_latents < 0 ||
      latent_h <= 0 || latent_w <= 0 || latent_h % 2 != 0 ||
      latent_w % 2 != 0 || text_tags.empty()) {
    return false;
  }
  const int n_text = (int)text_tags.size();
  const int ref_rows = ref_latents * kAudioChannels;
  const int tgt_rows = audio_latents * kAudioChannels;
  PackedLayout L;
  L.seq_len = n_text + ref_rows + tgt_rows;
  L.num_text_rows = n_text;
  L.condition_start = n_text;
  L.num_condition_rows = 0;
  L.audio_start = n_text;
  L.num_audio_rows = tgt_rows;   // GENERATED rows only
  L.video_start = L.seq_len;
  L.num_video_rows = 0;
  L.position_ids.assign((std::size_t)L.seq_len * 3, 0.0);
  L.token_tags.assign((std::size_t)L.seq_len, kAudioTag);
  auto pos = [&](int row, int axis) -> double& {
    return L.position_ids[(std::size_t)row * 3 + (std::size_t)axis];
  };
  for (int i = 0; i < n_text; ++i) {
    pos(i, 0) = (double)i;
    L.token_tags[(std::size_t)i] = text_tags[(std::size_t)i];
  }
  const double sqrt_area = std::sqrt((double)latent_h * (double)latent_w);
  const std::vector<double> w_grid =
      spatial_position_grid(latent_w, 2, sqrt_area);
  if (w_grid.empty()) { return false; }
  // Request 0 starts its audio at the prompt's end, like any H3 clip.
  const int tgt_t0 = ref_latents > 0 ? target_time_start : n_text;
  auto block = [&](int row0, int count, int t0) {
    for (int c = 0; c < kAudioChannels; ++c) {
      for (int j = 0; j < count; ++j) {
        const int row = row0 + c * count + j;
        pos(row, 0) = (double)t0 + (double)j;
        pos(row, 2) = c == 0 ? w_grid.front() : w_grid.back();
      }
    }
  };
  if (ref_latents > 0) { block(n_text, ref_latents, ref_time_start); }
  block(n_text + ref_rows, audio_latents, tgt_t0);
  for (int i = 0; i < ref_rows + tgt_rows; ++i) {
    L.audio_indices.push_back(n_text + i);
  }
  L.audio_runs.push_back({n_text, ref_rows + tgt_rows});
  L.num_condition_video_rows = 0;
  L.num_condition_audio_rows = ref_rows;
  L.ref2va = false;
  *out = std::move(L);
  return true;
}

std::vector<float>
chunk_raw_grid()
{
  const std::vector<float> lin =
      MiniMaxH3Scheduler::linspace_grid(kGridPoints);
  std::vector<float> g;
  for (int i : kStateIndices) { g.push_back(lin[(std::size_t)i]); }
  return g;
}

std::uint64_t
video_noise_seed(std::uint64_t request_seed, int request_index)
{
  const unsigned __int128 v =
      (unsigned __int128)request_seed +
      (unsigned __int128)request_index * kVideoNoiseStride;
  return (std::uint64_t)(v % ((unsigned __int128)1 << 63));
}

namespace {

std::vector<float>
patchify_(const std::vector<float>& raw, int channels, int lt, int lh,
          int lw)
{
  const int ph = lh / 2, pw = lw / 2;
  const int elems = channels * 4;
  std::vector<float> rows((std::size_t)lt * ph * pw * elems);
  for (int c = 0; c < channels; ++c) {
    for (int t = 0; t < lt; ++t) {
      for (int y = 0; y < lh; ++y) {
        for (int x = 0; x < lw; ++x) {
          const std::size_t src =
              (((std::size_t)c * lt + t) * lh + y) * lw + x;
          const std::size_t row =
              ((std::size_t)t * ph + y / 2) * pw + x / 2;
          const int e = (c * 2 + (y & 1)) * 2 + (x & 1);
          rows[row * elems + e] = raw[src];
        }
      }
    }
  }
  return rows;
}

}  // namespace

std::vector<float>
video_noise_rows(std::uint64_t seed, int channels, int latent_t,
                 int latent_h, int latent_w)
{
  TorchCpuRandn g(seed);
  const std::vector<float> raw =
      g.randn((std::size_t)channels * latent_t * latent_h * latent_w);
  if (raw.empty()) { return {}; }
  return patchify_(raw, channels, latent_t, latent_h, latent_w);
}

std::vector<float>
audio_noise_rows(std::uint64_t seed, int video_channels, int latent_h,
                 int latent_w, int audio_channels)
{
  TorchCpuRandn g(seed);
  // The full-request video draw, discarded: advancing this exact
  // generator is part of the audio's identity.
  (void)g.randn((std::size_t)video_channels * kRequestVideoLatents *
                latent_h * latent_w);
  return g.randn((std::size_t)kAudioChannels * kRequestAudioLatents *
                 audio_channels);
}

void
VideoRenorm::apply(float* rows, int num_rows, int channels)
{
  if (rows == nullptr || num_rows <= 0 || channels <= 0) { return; }
  std::vector<double> sum((std::size_t)channels, 0.0);
  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < channels; ++c) {
      sum[(std::size_t)c] += rows[(std::size_t)r * channels + c];
    }
  }
  std::vector<float> mean((std::size_t)channels), sd((std::size_t)channels);
  for (int c = 0; c < channels; ++c) {
    mean[(std::size_t)c] = (float)(sum[(std::size_t)c] / num_rows);
  }
  std::vector<double> var((std::size_t)channels, 0.0);
  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < channels; ++c) {
      const double d =
          (double)rows[(std::size_t)r * channels + c] - mean[(std::size_t)c];
      var[(std::size_t)c] += d * d;
    }
  }
  for (int c = 0; c < channels; ++c) {
    sd[(std::size_t)c] = std::max(
        (float)std::sqrt(var[(std::size_t)c] / num_rows), 1e-6f);
  }
  if (_mean.empty()) {
    _mean = std::move(mean);
    _std = std::move(sd);
    return;
  }
  // (x - mean) / std * anchor_std + anchor_mean, in that order.
  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < channels; ++c) {
      float& x = rows[(std::size_t)r * channels + c];
      x = (x - mean[(std::size_t)c]) / sd[(std::size_t)c] *
              _std[(std::size_t)c] +
          _mean[(std::size_t)c];
    }
  }
}

std::vector<float>
audio_slice(const float* rows, int total_latents, int start, int stop,
            int channels, int width)
{
  std::vector<float> out;
  if (rows == nullptr || start < 0 || stop > total_latents || stop <= start) {
    return out;
  }
  const int n = stop - start;
  out.resize((std::size_t)channels * n * width);
  for (int c = 0; c < channels; ++c) {
    std::copy(rows + ((std::size_t)c * total_latents + start) * width,
              rows + ((std::size_t)c * total_latents + stop) * width,
              out.data() + (std::size_t)c * n * width);
  }
  return out;
}


bool
is_adapter(const std::string& path)
{
  using lora::Adapter;
  return Adapter::file_touches(path, "blocks.0.attn.qkv_proj.lora_a") &&
         Adapter::file_touches(path,
                               "token_refiner.blocks.0.attn.qkv_proj.lora_a") &&
         !Adapter::file_touches(path, "adaln");
}

// ---- the driver -----------------------------------------------------

namespace {

int
chunk_rows_(const Phase& p, int frame_rows)
{
  return p.video_latents() * frame_rows + kAudioChannels * p.audio_latents();
}

// The largest chunk any request can have -- or, with `skip_first`, any
// chunk but the run's very first (which also owns the 5-frame prefix).
// Continuations differ only in how their audio rounds, so a dozen
// requests cover every case.
int
max_chunk_rows_(int frame_rows, bool skip_first = false)
{
  int m = 0;
  for (int r = 0; r < 14; ++r) {
    for (const Phase& p : request_plan(r).phases) {
      if (skip_first && r == 0 && p.index == 0) { continue; }
      m = std::max(m, chunk_rows_(p, frame_rows));
    }
  }
  return m;
}

std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0) {
    return (std::uint16_t)((u >> 16) | 0x40u);
  }
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
bf16_to_f32_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

void
upload_(const std::vector<float>& src, metal_compute::SharedBuffer& dst)
{
  auto* d = static_cast<std::uint16_t*>(dst.contents());
  for (std::size_t i = 0; i < src.size(); ++i) { d[i] = f32_to_bf16_(src[i]); }
}

// sorted unique of a forward's row timesteps -- what build_row_timesteps
// returns -- known without a layout, so the whole run's AdaLN schedule
// can be baked before the first forward.
std::vector<float>
uniq_(std::initializer_list<float> v)
{
  std::vector<float> u(v);
  std::sort(u.begin(), u.end());
  u.erase(std::unique(u.begin(), u.end()), u.end());
  return u;
}

}  // namespace

int
kv_capacity_rows(int latent_h, int latent_w)
{
  // What the retention can ever hold: the first chunk whole beside one
  // other (before the first trim), or its VIDEO beside two others (every
  // trim after). The first chunk is the largest -- it owns the prefix --
  // so counting it three times over-books by a chunk's worth of GB.
  const int fr = (latent_h / 2) * (latent_w / 2);
  const Plan first = request_plan(0);
  const Phase& p0 = first.phases[0];
  const int other = max_chunk_rows_(fr, true);
  return std::max(chunk_rows_(p0, fr) + other,
                  p0.video_latents() * fr + 2 * other);
}

int
max_chunk_seq(int latent_h, int latent_w, int max_text)
{
  return max_text + max_chunk_rows_((latent_h / 2) * (latent_w / 2));
}

StreamKv::Storage
kv_storage(int latent_h, int latent_w, bool disk, int bits)
{
  StreamKv::Storage st;
  st.disk = disk;
  st.bits = disk ? bits : 16;
  if (disk) {
    const int fr = (latent_h / 2) * (latent_w / 2);
    const Plan first = request_plan(0);
    st.slot_rows = chunk_rows_(first.phases[0], fr) +
                   2 * max_chunk_rows_(fr, true);
    st.new_rows = max_chunk_rows_(fr);
  }
  return st;
}

std::size_t
kv_cache_bytes(int latent_h, int latent_w, int max_text, int layers,
               int heads, int head_dim, const StreamKv::Storage& st)
{
  // The staging lives in the DiT's q|k|v scratch, which the preflight
  // counts already ([seq, 3 * inner]); only what it needs beyond that is
  // the cache's.
  const int seq = max_chunk_seq(latent_h, latent_w, max_text);
  const std::size_t all =
      StreamKv::bytes_for(layers, heads, head_dim,
                          kv_capacity_rows(latent_h, latent_w), seq, st);
  const std::size_t qkv = (std::size_t)seq * 3 * heads * head_dim * 2;
  const std::size_t staging =
      2 * (std::size_t)heads * head_dim * 2 *
      (std::size_t)(kv_capacity_rows(latent_h, latent_w) + seq);
  return all - std::min(staging, qkv);
}

bool
run(MetalMiniMaxH3Transformer* dit, const RunConfig& cfg,
    const std::vector<Request>& requests, RunResult* out, std::string* err)
{
  using metal_compute::SharedBuffer;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (dit == nullptr || out == nullptr || requests.empty()) {
    return fail("taomate: nothing to run");
  }
  *out = RunResult{};
  const MetalMiniMaxH3Transformer::Config& c = dit->config();
  metal_compute::MetalCompute* mc = dit->metal_compute();
  const int lh = cfg.latent_h, lw = cfg.latent_w;
  if (mc == nullptr || lh <= 0 || lw <= 0 || lh % 2 != 0 || lw % 2 != 0) {
    return fail("taomate: bad latent canvas");
  }
  if (cfg.lora_slot < 0 || cfg.lora_slot >= dit->lora_slots() ||
      dit->lora_modules(cfg.lora_slot) <= 0) {
    return fail("taomate: the DiT has no TaoMate adapter in its slot");
  }
  const int R = (int)requests.size();
  const int fr = (lh / 2) * (lw / 2);
  const int PE = c.video_patch_elems();
  const int AC = c.audio_channels;
  int max_text = 0;
  for (const Request& q : requests) {
    if (q.text == nullptr || q.text_tags.empty()) {
      return fail("taomate: a request without a prompt");
    }
    max_text = std::max(max_text, (int)q.text_tags.size());
  }
  auto log = [&](const std::string& m) { if (cfg.log) { cfg.log(m); } };

  // ---- the schedules ------------------------------------------------
  MiniMaxH3Scheduler tv(kVideoShift), ta(kAudioShift);
  MiniMaxH3Scheduler cv(kVideoShift), ca(kAudioShift);
  if (!tv.set_timesteps(kTeacherGridPoints) ||
      !ta.set_timesteps(kTeacherGridPoints) ||
      !cv.set_sigmas(chunk_raw_grid()) || !ca.set_sigmas(chunk_raw_grid())) {
    return fail("taomate: schedule construction failed");
  }
  const int t_steps = (int)tv.timesteps().size();          // 9
  const int c_steps = (int)cv.timesteps().size();          // 3
  if (t_steps != kTeacherGridPoints - 1 ||
      (int)ta.timesteps().size() != t_steps ||
      c_steps != 3 || (int)ca.timesteps().size() != c_steps) {
    return fail("taomate: the schedules lost a point");
  }
  // Every forward this run makes, as a baked AdaLN schedule: the
  // teacher's steps without and with frozen reference rows (t = 1), the
  // chunk's three steps, and the commit. Text rows ride the VIDEO clock.
  std::vector<std::vector<float>> sched;
  for (int i = 0; i < t_steps; ++i) {
    sched.push_back(uniq_({tv.timesteps()[(std::size_t)i],
                           ta.timesteps()[(std::size_t)i]}));
  }
  for (int i = 0; i < t_steps; ++i) {
    sched.push_back(uniq_({tv.timesteps()[(std::size_t)i],
                           ta.timesteps()[(std::size_t)i], 1.0f}));
  }
  for (int k = 0; k < c_steps; ++k) {
    sched.push_back(uniq_({cv.timesteps()[(std::size_t)k],
                           ca.timesteps()[(std::size_t)k]}));
  }
  sched.push_back({1.0f});
  const int kTeach0 = 0, kTeachR = t_steps, kChunk = 2 * t_steps;
  const int kCommit = 2 * t_steps + c_steps;
  bool baked = false;
  {
    std::string berr;
    if (std::getenv("VPIPE_H3_NO_ADALN_BAKE") == nullptr) {
      baked = dit->bake_adaln(sched, &berr);
    } else {
      berr = "disabled by VPIPE_H3_NO_ADALN_BAKE";
    }
    if (!baked) {
      if (dit->adaln_baked()) {
        return fail("taomate: the DiT is baked for another schedule and "
                    "re-baking for this one failed: " + berr);
      }
      log("taomate: AdaLN not baked (" + berr + "); projections per step");
    }
  }

  const int total_forwards =
      R * t_steps + R * 4 * (c_steps + 1);
  int done = 0;
  auto tick = [&]() {
    ++done;
    ++out->forwards;
    if (cfg.progress && !cfg.progress(done, total_forwards)) {
      out->stopped = true;
      return false;
    }
    return true;
  };

  // The scratch for the longest forward of the run, kept for all of them
  // -- its q|k|v part wide enough to hold the K/V staging too, which
  // lives there (StreamKv::adopt_staging) instead of beside it.
  {
    const std::size_t row = (std::size_t)c.n_heads * c.head_dim * 2;
    const std::size_t staging =
        2 * row * (std::size_t)(kv_capacity_rows(lh, lw) +
                                max_chunk_seq(lh, lw, max_text));
    dit->set_scratch_floor(max_chunk_seq(lh, lw, max_text), max_text,
                           staging / 2);
  }
  struct ScratchFloorReset {
    MetalMiniMaxH3Transformer* d;
    ~ScratchFloorReset() { d->set_scratch_floor(0, 0); }
  } floor_reset{dit};
  const float scale0 = dit->lora_scale(cfg.lora_slot);
  struct ScaleReset {
    MetalMiniMaxH3Transformer* d;
    int slot;
    float s;
    ~ScaleReset() { d->set_lora_scale(slot, s); }
  } scale_reset{dit, cfg.lora_slot, scale0};

  // One forward. `vrows` may be empty (the teacher); the commit wants no
  // velocity back. The rows go up through two buffers kept for the run --
  // grown, never reallocated per forward (each is read by the forward
  // that is waited for before the next upload).
  SharedBuffer vb_keep, ab_keep;
  auto upload = [&](const std::vector<float>& rows, SharedBuffer& keep,
                    SharedBuffer* view) -> bool {
    const std::size_t n = rows.size() * 2;
    if (keep.byte_size() < n) {
      keep = SharedBuffer{};
      keep = mc->make_shared_buffer(n);
      if (keep.empty()) { return false; }
    }
    *view = keep.subview(0, n);
    upload_(rows, *view);
    return !view->empty();
  };
  auto forward = [&](const PackedLayout& L, const Request& q,
                     const std::vector<float>& vrows,
                     const std::vector<float>& arows, float t_video,
                     float t_audio, int sched_index, StreamKv* kv,
                     bool commit, std::vector<float>* vel_video,
                     std::vector<float>* vel_audio) -> bool {
    SharedBuffer vb, ab;
    if (!vrows.empty() && !upload(vrows, vb_keep, &vb)) {
      return fail("taomate: video upload allocation");
    }
    if (!arows.empty() && !upload(arows, ab_keep, &ab)) {
      return fail("taomate: audio upload allocation");
    }
    std::vector<float> uniq;
    std::vector<int> ri;
    build_row_timesteps(L, t_video, t_audio, 1.0f, &uniq, &ri, 1.0f);
    MetalMiniMaxH3Transformer::Step st;
    st.video = vrows.empty() ? nullptr : &vb;
    st.audio = arows.empty() ? nullptr : &ab;
    st.text = q.text;
    st.layout = &L;
    st.timesteps = &uniq;
    st.row_timestep_index = &ri;
    st.schedule_index = baked ? sched_index : -1;
    st.stream_kv = kv;
    st.stream_commit = commit;
    std::string ferr;
    MetalMiniMaxH3Transformer::Velocity v = dit->forward(st, &ferr);
    if (kv != nullptr && kv->on_disk()) {
      const StreamKv::IoStats io = kv->io_stats();
      out->kv_read += io.read;
      out->kv_written += io.written;
    }
    if (v.empty()) {
      if (ferr == "stopped") {
        out->stopped = true;
        if (err != nullptr) { err->clear(); }
        return false;
      }
      return fail("taomate: forward: " + ferr);
    }
    if (vel_video != nullptr) {
      vel_video->resize(vrows.size());
      const auto* g = static_cast<const std::uint16_t*>(v.video.contents());
      for (std::size_t k = 0; k < vel_video->size(); ++k) {
        (*vel_video)[k] = bf16_to_f32_(g[k]);
      }
    }
    if (vel_audio != nullptr) {
      vel_audio->resize(arows.size());
      const auto* g = static_cast<const std::uint16_t*>(v.audio.contents());
      for (std::size_t k = 0; k < vel_audio->size(); ++k) {
        (*vel_audio)[k] = bf16_to_f32_(g[k]);
      }
    }
    return true;
  };

  // The clean-K/V cache, FIRST: it is the run's largest allocation after
  // the model, and failing to make it should cost nothing -- not nine
  // teacher forwards a request.
  StreamKv kv;
  {
    std::string kerr;
    // Its staging is not made here: the forwards carve it out of the
    // DiT's q|k|v scratch (or make it apart, when they cannot).
    const StreamKv::Storage st =
        kv_storage(lh, lw, cfg.kv_disk, cfg.kv_bits);
    if (!kv.init(mc, c.n_layers, c.n_heads, c.head_dim,
                 kv_capacity_rows(lh, lw), st, &kerr)) {
      return fail("taomate: " + (kerr.empty()
                                     ? std::string("kv staging allocation")
                                     : kerr));
    }
    out->kv_bytes = kv.bytes();
    if (kv.on_disk()) {
      log(fmt("taomate: clean-KV cache {} rows x {} layers on disk, {}, "
              "read a layer ahead through two {} MB slots", kv.capacity(),
              c.n_layers, kv.bits() == 8 ? "8-bit (groups of 64)" : "bf16",
              (kv.bytes() / 2) >> 20)());
    } else {
      log(fmt("taomate: clean-KV cache {} rows x {} layers, {} MB",
              kv.capacity(), c.n_layers, kv.bytes() >> 20)());
    }
  }

  // ---- 1. the audio teacher: the BASE model, every request ----------
  dit->set_lora_scale(cfg.lora_slot, 0.0f);
  // [request][state] -> channel-major [2 * active, 32].
  std::vector<std::array<std::vector<float>, 3>> milestones((std::size_t)R);
  std::vector<int> active_audio((std::size_t)R, 0);
  std::vector<std::vector<float>> official_audio((std::size_t)R);
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < R; ++r) {
    const Request& q = requests[(std::size_t)r];
    const Plan plan = request_plan(r);
    const int act = plan.audio_latents();
    active_audio[(std::size_t)r] = act;
    const int prefix = kRequestAudioLatents - act;
    official_audio[(std::size_t)r] =
        audio_noise_rows(cfg.audio_seed + (std::uint64_t)r, c.video_channels,
                         lh, lw, AC);
    if (official_audio[(std::size_t)r].empty()) {
      return fail("taomate: audio noise");
    }
    std::vector<float> target =
        audio_slice(official_audio[(std::size_t)r].data(),
                    kRequestAudioLatents, prefix, kRequestAudioLatents,
                    kAudioChannels, AC);
    const int n_text = (int)q.text_tags.size();
    int ref = 0, ref_t0 = 0, tgt_t0 = 0;
    std::vector<float> rows;
    if (r > 0) {
      const int prev = active_audio[(std::size_t)r - 1];
      ref = kRolloverLatents;
      ref_t0 = n_text + prev - kRolloverLatents;
      tgt_t0 = n_text + prev;
      rows = audio_slice(milestones[(std::size_t)r - 1][2].data(), prev,
                         prev - kRolloverLatents, prev, kAudioChannels, AC);
    }
    const std::size_t ref_elems = rows.size();
    rows.insert(rows.end(), target.begin(), target.end());
    PackedLayout L;
    if (!build_teacher_layout(q.text_tags, ref, act, lh, lw, ref_t0, tgt_t0,
                              &L)) {
      return fail("taomate: teacher layout");
    }
    std::vector<float> vel;
    for (int i = 0; i < t_steps; ++i) {
      if (!forward(L, q, {}, rows, tv.timesteps()[(std::size_t)i],
                   ta.timesteps()[(std::size_t)i],
                   (r == 0 ? kTeach0 : kTeachR) + i, nullptr, false, nullptr,
                   &vel)) {
        return false;
      }
      // Only the target rows step; the reference rows lead the buffer
      // and stay clean.
      if (!ta.step(vel.data() + ref_elems, i, rows.data() + ref_elems,
                   rows.size() - ref_elems)) {
        return fail("taomate: teacher step");
      }
      for (int k = 0; k < 3; ++k) {
        if (i + 1 == kTeacherStates[k]) {
          milestones[(std::size_t)r][(std::size_t)k].assign(
              rows.begin() + (std::ptrdiff_t)ref_elems, rows.end());
        }
      }
      if (!tick()) { return false; }
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  log(fmt("taomate: audio teacher, {} request(s) x {} forwards: {:.1f} s",
          R, t_steps,
          std::chrono::duration<double>(t1 - t0).count())());

  // ---- 2. the video: the adapter, chunk by chunk --------------------
  dit->set_lora_scale(cfg.lora_slot, cfg.lora_scale);
  const int media_origin = (int)requests[0].text_tags.size();
  int video_off = 0, audio_off = 0;
  VideoRenorm renorm;
  std::vector<std::vector<float>> audio_ch((std::size_t)kAudioChannels);
  for (int r = 0; r < R; ++r) {
    const Request& q = requests[(std::size_t)r];
    const Plan plan = request_plan(r);
    const int act = active_audio[(std::size_t)r];
    if (r > 0 && r % kAudioKvResetRequests == 0) {
      const int dropped = kv.drop_audio();
      log(fmt("taomate: request {}: audio KV reset ({} rows)", r,
              dropped)());
    }
    std::vector<float> vnoise =
        video_noise_rows(video_noise_seed(q.seed, r), c.video_channels,
                         kRequestVideoLatents, lh, lw);
    if (vnoise.empty()) { return fail("taomate: video noise"); }
    // A continuation does not regenerate the prefix: its first two
    // latents' noise is not used.
    const int vskip = kRequestVideoLatents - plan.video_latents();
    const std::size_t v0 = (std::size_t)vskip * fr * PE;
    for (const Phase& ph : plan.phases) {
      const auto tc0 = std::chrono::steady_clock::now();
      const std::size_t rd0 = out->kv_read, wr0 = out->kv_written;
      const int cache_rows = kv.rows();
      PackedLayout L;
      if (!build_chunk_layout(q.text_tags, ph, lh, lw, media_origin,
                              video_off, audio_off, &L)) {
        return fail("taomate: chunk layout");
      }
      const std::size_t row_elems = (std::size_t)fr * PE;
      std::vector<float> vrows(
          vnoise.begin() +
              (std::ptrdiff_t)(v0 + ph.video_latent_start * row_elems),
          vnoise.begin() +
              (std::ptrdiff_t)(v0 + ph.video_latent_stop * row_elems));
      const int prefix = kRequestAudioLatents - act;
      // The chunk's audio starts as the official noise -- the same rows
      // the teacher started from -- and after step k is the teacher's
      // state 3(k + 1).
      std::vector<float> arows = audio_slice(
          official_audio[(std::size_t)r].data(), kRequestAudioLatents,
          prefix + ph.audio_latent_start, prefix + ph.audio_latent_stop,
          kAudioChannels, AC);
      std::vector<float> vel;
      for (int k = 0; k < c_steps; ++k) {
        if (!forward(L, q, vrows, arows, cv.timesteps()[(std::size_t)k],
                     ca.timesteps()[(std::size_t)k], kChunk + k, &kv, false,
                     &vel, nullptr)) {
          return false;
        }
        if (!cv.step(vel.data(), k, vrows.data(), vrows.size())) {
          return fail("taomate: chunk step");
        }
        arows = audio_slice(milestones[(std::size_t)r][(std::size_t)k].data(),
                            act, ph.audio_latent_start, ph.audio_latent_stop,
                            kAudioChannels, AC);
        if (!tick()) { return false; }
      }
      // Clean: matched to the first chunk's statistics, then committed.
      renorm.apply(vrows.data(), ph.video_latents() * fr, PE);
      if (!forward(L, q, vrows, arows, 1.0f, 1.0f, kCommit, &kv, true,
                   nullptr, nullptr)) {
        return false;
      }
      out->kv_disk_peak = std::max(out->kv_disk_peak, kv.disk_bytes());
      log(fmt("taomate: request {} chunk {}: {} rows over a cache of {}, "
              "{} steps + the commit in {:.1f} s{}", r + 1, ph.index + 1,
              L.num_video_rows + L.num_audio_rows, cache_rows, c_steps,
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() - tc0).count(),
              kv.on_disk()
                  ? fmt(" (the cache: {} MB read, {} MB written)",
                        (out->kv_read - rd0) >> 20,
                        (out->kv_written - wr0) >> 20)()
                  : std::string())());
      if (!tick()) { return false; }
      out->video_rows.insert(out->video_rows.end(), vrows.begin(),
                             vrows.end());
    }
    // The request's audio is the teacher's clean latent, per channel.
    const std::vector<float>& clean = milestones[(std::size_t)r][2];
    for (int ch = 0; ch < kAudioChannels; ++ch) {
      audio_ch[(std::size_t)ch].insert(
          audio_ch[(std::size_t)ch].end(),
          clean.begin() + (std::ptrdiff_t)((std::size_t)ch * act * AC),
          clean.begin() + (std::ptrdiff_t)((std::size_t)(ch + 1) * act * AC));
    }
    video_off += plan.video_latents();
    audio_off += plan.audio_latents();
    log(fmt("taomate: request {} of {}: {} video + {} audio latents, cache "
            "{} rows ({} audio)", r + 1, R, plan.video_latents(), act,
            kv.rows(), kv.audio_rows())());
  }
  out->video_latents = video_off;
  out->audio_latents = audio_off;
  for (int ch = 0; ch < kAudioChannels; ++ch) {
    out->audio_rows.insert(out->audio_rows.end(),
                           audio_ch[(std::size_t)ch].begin(),
                           audio_ch[(std::size_t)ch].end());
  }
  const auto t2 = std::chrono::steady_clock::now();
  log(fmt("taomate: video, {} request(s) x 4 chunks x {} forwards: {:.1f} s",
          R, c_steps + 1,
          std::chrono::duration<double>(t2 - t1).count())());
  if (kv.on_disk()) {
    log(fmt("taomate: the cache's files peaked at {} MB; the forwards read "
            "{} MB back, the commits wrote {} MB", out->kv_disk_peak >> 20,
            out->kv_read >> 20, out->kv_written >> 20)());
  }
  return true;
}

}  // namespace taomate
}  // namespace minimax_h3
}  // namespace genai
}  // namespace vpipe
