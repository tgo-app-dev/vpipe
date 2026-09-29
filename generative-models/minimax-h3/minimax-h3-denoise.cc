#include "generative-models/minimax-h3/minimax-h3-denoise.h"

#include "generative-models/minimax-h3/minimax-h3-scheduler.h"
#include "generative-models/shared/motion-cache.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::SharedBuffer;

namespace {

inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

inline float
bf16_to_f32_(std::uint16_t b)
{
  const std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

bool
motion_cache_gathers(const MetalMiniMaxH3Transformer::Config& cfg,
                     int grid_h, int grid_w, int video_rows,
                     int audio_rows, int subsample,
                     MotionCacheGathers* out, std::string* why)
{
  auto no = [&](std::string m) {
    if (why != nullptr) { *why = std::move(m); }
    return false;
  };
  const int ph = cfg.patch_h, pw = cfg.patch_w;
  const int PE = cfg.video_patch_elems();
  const int cells = grid_h * grid_w;
  const int f = std::max(1, subsample);
  if (grid_h <= 0 || grid_w <= 0) {
    return no("the request carries no video grid");
  }
  if (video_rows <= 0 || video_rows % cells != 0) {
    return no(fmt("{} generated rows are not whole {}x{} frames",
                  video_rows, grid_h, grid_w)());
  }
  if (cfg.patch_t != 1) { return no("a temporal patch is not supported"); }
  if ((std::uint64_t)video_rows * (std::uint64_t)PE > 0xffffffffull) {
    return no("the video state is past 32-bit offsets");
  }
  const int lt = video_rows / cells;
  const int H = grid_h * ph, W = grid_w * pw;
  out->channels = cfg.video_channels;
  out->frames   = lt;
  out->plane    = ((H + f - 1) / f) * ((W + f - 1) / f);
  out->video.clear();
  out->video.reserve((std::size_t)out->channels * lt * out->plane);
  for (int c = 0; c < cfg.video_channels; ++c) {
    for (int t = 0; t < lt; ++t) {
      for (int y = 0; y < H; y += f) {
        for (int x = 0; x < W; x += f) {
          const std::size_t row = (std::size_t)t * cells +
                                  (std::size_t)(y / ph) * grid_w +
                                  (std::size_t)(x / pw);
          const std::size_t el =
              ((std::size_t)c * ph + (std::size_t)(y % ph)) * pw +
              (std::size_t)(x % pw);
          out->video.push_back((std::uint32_t)(row * PE + el));
        }
      }
    }
  }
  out->audio.clear();
  const int AC = cfg.audio_channels;
  if (audio_rows > 0 && audio_rows % minimax_h3::kAudioChannels == 0) {
    const int alat = audio_rows / minimax_h3::kAudioChannels;
    for (int ch = 0; ch < minimax_h3::kAudioChannels; ++ch) {
      for (int t = 0; t < alat; t += f) {
        for (int k = 0; k < AC; ++k) {
          out->audio.push_back(
              (std::uint32_t)(((std::size_t)ch * alat + t) * AC + k));
        }
      }
    }
  }
  return true;
}

bool
denoise(const DenoiseRequest& req, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (req.dit == nullptr || req.layout == nullptr || req.text == nullptr ||
      req.video == nullptr || req.audio == nullptr) {
    return fail("denoise: incomplete request");
  }
  const minimax_h3::PackedLayout& L = *req.layout;
  const auto& cfg = req.dit->config();
  const int PE   = cfg.video_patch_elems();
  const int AC   = cfg.audio_channels;
  // Conditioning rows lead each modality's buffer and are never
  // stepped. Video has always had them (a keyframe anchor); AUDIO has
  // them only in `ref2va`, where a reference soundtrack rides through
  // at its own fixed level -- so the audio slice is offset the same way
  // the video slice always was, rather than starting at row 0.
  const int ncond = L.num_condition_video_rows;
  const int nvid  = L.num_video_rows;         // generated rows only
  const int ncaud = L.num_condition_audio_rows;
  const int naud  = L.num_audio_rows;         // generated rows only
  const int vrows = (int)L.video_indices.size();
  const int arows = (int)L.audio_indices.size();
  if (vrows != ncond + nvid) {
    return fail("denoise: layout video_indices does not match "
                "condition + generated rows");
  }
  if (arows != ncaud + naud) {
    return fail("denoise: layout audio_indices does not match "
                "reference + generated rows");
  }
  if (nvid <= 0) { return fail("denoise: nothing to generate"); }

  MiniMaxH3Scheduler sv(req.video_shift);
  MiniMaxH3Scheduler sa(req.audio_shift);
  // `res_multistep` -- the second-order sampler the reference template
  // ships -- against the first-order Euler this loop has always used.
  // Under measurement rather than default-on: it changes BOTH modalities'
  // output, and the video at 4 steps is already good.
  const bool res_ms = std::getenv("VPIPE_H3_RES_MULTISTEP") != nullptr;
  std::vector<float> vprev, aprev;
  // A FLOW-MAP adapter conditions each step on where it lands, so the
  // loop hands the DiT (t, r) pairs rather than t alone -- and runs the
  // grid the adapter was distilled on unless the caller named another.
  const bool two_time = req.dit->two_time_active();
  const MetalMiniMaxH3Transformer::FlowMap* fm = req.dit->flow_map();
  const std::vector<float>* grid =
      !req.sigma_grid.empty() ? &req.sigma_grid
      : (two_time && fm != nullptr && !fm->sigmas.empty()) ? &fm->sigmas
                                                           : nullptr;
  if (grid != nullptr) {
    if (!sv.set_sigmas(*grid) || !sa.set_sigmas(*grid)) {
      return fail(fmt("denoise: the {} sigma grid ({} points) is not a "
                      "strictly decreasing 1 -> 0 grid under shifts "
                      "{:.4g}/{:.4g}",
                      grid == &req.sigma_grid ? "requested" : "adapter's",
                      grid->size(), req.video_shift, req.audio_shift)());
    }
  } else if (req.num_steps < 1 ||
             !sv.set_timesteps(req.num_steps + 1) ||
             !sa.set_timesteps(req.num_steps + 1)) {
    // `num_steps` counts FORWARDS, and the scheduler takes grid POINTS:
    // N steps is the N + 1 points of linspace(1, 0, N + 1), the terminal
    // zero included. See DenoiseRequest::num_steps for why the two differ.
    return fail("denoise: bad step count " + std::to_string(req.num_steps));
  }
  // The shift collapses duplicate sigmas, and it does so at whichever
  // grid points ITS shift happens to collide -- so the two schedules can
  // come back with different lengths from the same request. Step i of
  // one has to pair with step i of the other, so run the shorter.
  const int steps = std::min(sv.num_inference_steps(),
                             sa.num_inference_steps());
  if (steps <= 0) { return fail("denoise: empty schedule"); }

  metal_compute::MetalCompute* mc = req.dit->metal_compute();
  if (mc == nullptr) { return fail("denoise: no metal-compute"); }
  SharedBuffer vb = mc->make_shared_buffer((std::size_t)vrows * PE * 2);
  // Sized by the TOTAL audio rows, not the generated ones: a `ref2va`
  // request hands in a reference block per soundtrack ahead of them,
  // and the head writes every row it is given.
  SharedBuffer ab = mc->make_shared_buffer((std::size_t)arows * AC * 2);
  if (vb.empty() || (naud > 0 && ab.empty())) {
    return fail("denoise: activation allocation failed");
  }

  // Per-step trace, opt-in: it costs two extra host passes over the
  // velocity and one line of log per step.
  const SessionContextIntf* sess =
      mc != nullptr ? mc->session() : nullptr;
  const bool prof =
      sess != nullptr && std::getenv("VPIPE_H3_DENOISE_PROFILE") != nullptr;

  std::vector<float> uniq, uniq_r;
  std::vector<int>   row_idx;
  // The one place a step's row plan is built, so the bake below and the
  // forward in the loop cannot disagree about it.
  auto plan = [&](int i, std::vector<float>* u, std::vector<float>* ur,
                  std::vector<int>* ri) {
    const float tv = sv.timesteps()[(std::size_t)i];
    const float ta = sa.timesteps()[(std::size_t)i];
    if (two_time) {
      minimax_h3::build_row_time_pairs(
          L, tv, sv.endpoints()[(std::size_t)i], ta,
          sa.endpoints()[(std::size_t)i], req.condition_timestep, u, ur, ri,
          req.condition_audio_timestep);
    } else {
      minimax_h3::build_row_timesteps(L, tv, ta, req.condition_timestep, u,
                                      ri, req.condition_audio_timestep);
      ur->clear();
    }
  };

  // Precompute every step's modulation before the loop, which is the one
  // place that can: the two schedules are fully built by now and the
  // condition timesteps are constants, so every timestep this run will
  // ever ask for is known. That lets the DiT drop the 50 AdaLN
  // projections -- 55% of the 4-bit checkpoint -- and a streaming run
  // stops re-reading them on every step.
  //
  // Advisory. A refusal (a schedule long enough that the tables cost
  // more than they save) is logged and the loop runs exactly as before,
  // because this is a memory optimization and not a correctness step.
  bool baked = false;
  {
    std::vector<std::vector<float>> sched, sched_r;
    sched.reserve((std::size_t)steps);
    for (int i = 0; i < steps; ++i) {
      std::vector<float> u, ur;
      std::vector<int>   ri;
      plan(i, &u, &ur, &ri);
      sched.push_back(std::move(u));
      sched_r.push_back(std::move(ur));
    }
    std::string berr;
    // The A/B. Baking trades a per-step read of 55% of the checkpoint for
    // one up-front read, which is unambiguously less I/O -- but on a
    // passively cooled box less I/O means the GPU stops waiting and
    // starts drawing power, and the clock it loses to that can cost more
    // than the reads. Whether it is a win is therefore a property of the
    // MACHINE, and this switch is how to find out on a given one.
    if (std::getenv("VPIPE_H3_NO_ADALN_BAKE") != nullptr) {
      berr = "disabled by VPIPE_H3_NO_ADALN_BAKE";
      baked = false;
    } else {
      baked = req.dit->bake_adaln(sched, &berr,
                                  two_time ? &sched_r : nullptr);
    }
    if (!baked && sess != nullptr) {
      sess->log_normal(fmt("denoise: AdaLN not baked ({}); running the "
                           "projections per step", berr));
    }
  }

  // MotionCache: a step whose forward a cached residual predicts well
  // enough runs no forward at all. The window is a fraction of the
  // sampling range mapped through the VIDEO shift -- the schedule the
  // reference's sampler carries, the audio being mapped onto it -- and
  // the estimator reads the generated rows only, through gathers that
  // undo this loop's patch packing.
  const motion_cache::Config mcfg =
      motion_cache::config_from_flex(req.accel);
  motion_cache::Cache mcache;
  if (mcfg.enabled) {
    std::string why;
    MotionCacheGathers mg;
    if (!mcache.begin(
            req.accel,
            motion_cache::percent_to_sigma(mcfg.start, req.video_shift),
            motion_cache::percent_to_sigma(mcfg.end, req.video_shift),
            /*x0_sign=*/1.0f)) {
      why = fmt("the window {:.2f}-{:.2f} is empty", mcfg.start,
                mcfg.end)();
    } else if (!motion_cache_gathers(cfg, req.video_grid_h,
                                     req.video_grid_w, nvid, naud,
                                     mcache.subsample(), &mg, &why)) {
      mcache = motion_cache::Cache();
    }
    if (why.empty()) {
      mcache.set_video(mg.video.data(), mg.channels, mg.frames, mg.plane,
                       (std::size_t)nvid * PE);
      if (!mg.audio.empty()) {
        mcache.set_audio(mg.audio.data(), mg.audio.size(),
                         (std::size_t)naud * AC);
      }
      if (sess != nullptr) {
        sess->info(fmt("h3-denoise: motion_cache on -- {}",
                       mcache.describe()));
      }
    } else if (sess != nullptr) {
      sess->warn(fmt("h3-denoise: motion_cache is OFF for this clip: {}",
                     why));
    }
  }
  // Only a computed step fills these; a reused one is filled by the
  // cache. Held across steps so a reuse has somewhere to land.
  std::vector<float> vel_v((std::size_t)nvid * PE);
  std::vector<float> vel_a((std::size_t)naud * AC);
  // Whether the cached residual covers the audio -- a computed step
  // whose head wrote none leaves nothing to reuse there.
  bool cached_audio = false;
  int forwards = 0, reused_steps = 0, steps_run = 0;

  // A preview's clean estimate, filled only on a step that renders one.
  std::vector<float> preview_x0;
  for (int i = 0; i < steps; ++i) {
    // Every row's timestep, this step. The conditioning rows keep their
    // own value while the generated video and audio rows walk their two
    // schedules -- that is what one forward over rows at DIFFERENT noise
    // levels means, and it is the whole reason the AdaLN table is
    // indexed per row.
    // t = 1 - sigma, t = 1 meaning CLEAN. The transformer consumes it
    // unscaled and recovers sigma itself; feeding a pure-noise latent
    // t = 0 would tell the model it is already done.
    plan(i, &uniq, &uniq_r, &row_idx);
    // Which ROLE each distinct entry serves, first and last step: the
    // one line that shows the two schedules and the pinned references
    // actually landed on the rows they were meant for, in a layout that
    // interleaves them. Roles, not tags -- a vision block's prompt rows
    // are tagged VIDEO and would otherwise read as generated video.
    if (prof && (i == 0 || i + 1 == steps)) {
      enum { kText, kRefVid, kGenVid, kRefAud, kGenAud, kRoles };
      std::vector<int> role((std::size_t)L.seq_len, kText);
      for (int k = 0; k < vrows; ++k) {
        role[(std::size_t)L.video_indices[(std::size_t)k]] =
            k < ncond ? kRefVid : kGenVid;
      }
      for (int k = 0; k < arows; ++k) {
        role[(std::size_t)L.audio_indices[(std::size_t)k]] =
            k < ncaud ? kRefAud : kGenAud;
      }
      std::vector<std::array<int, kRoles>> cnt(uniq.size(),
                                               std::array<int, kRoles>{});
      for (std::size_t k = 0; k < row_idx.size(); ++k) {
        ++cnt[(std::size_t)row_idx[k]][(std::size_t)role[k]];
      }
      std::string s;
      for (std::size_t e = 0; e < uniq.size(); ++e) {
        const auto& c = cnt[e];
        s += fmt(" | t {:.6f}", uniq[e])();
        if (two_time) { s += fmt(" r {:.6f}", uniq_r[e])(); }
        s += fmt(": text {} refvid {} genvid {} refaud {} genaud {}",
                 c[kText], c[kRefVid], c[kGenVid], c[kRefAud], c[kGenAud])();
      }
      sess->info(fmt("h3-denoise step {} row plan ({} entries){}", i + 1,
                     uniq.size(), s));
    }

    float* vx = req.video + (std::size_t)ncond * PE;
    float* ax = req.audio + (std::size_t)ncaud * AC;
    const double sigma_v = sv.sigmas()[(std::size_t)i];
    // Before anything is uploaded: a reused step never touches the GPU.
    const bool reuse =
        mcache.armed() && mcache.decide(sigma_v, vx, naud > 0 ? ax : nullptr);
    bool have_aud = false;
    const auto t_fwd0 = std::chrono::steady_clock::now();
    if (reuse) {
      mcache.predict(vx, vel_v.data(), cached_audio ? ax : nullptr,
                     cached_audio ? vel_a.data() : nullptr);
      have_aud = cached_audio;
      ++reused_steps;
    } else {
      {
        auto* d = static_cast<std::uint16_t*>(vb.contents());
        for (std::size_t k = 0; k < (std::size_t)vrows * PE; ++k) {
          d[k] = f32_to_bf16_(req.video[k]);
        }
      }
      if (arows > 0) {
        // Every audio row, reference blocks included -- they are what the
        // generated rows attend to. Uploading only the generated count
        // would leave the tail of the buffer whatever the allocation had
        // in it, which is how a reference request turns into NaN.
        auto* d = static_cast<std::uint16_t*>(ab.contents());
        for (std::size_t k = 0; k < (std::size_t)arows * AC; ++k) {
          d[k] = f32_to_bf16_(req.audio[k]);
        }
      }

      MetalMiniMaxH3Transformer::Step st;
      st.video  = &vb;
      st.audio  = &ab;
      st.text   = req.text;
      st.layout = &L;
      st.timesteps          = &uniq;
      st.row_timestep_index = &row_idx;
      st.endpoints          = two_time ? &uniq_r : nullptr;
      st.schedule_index     = baked ? i : -1;
      st.dense_attention    = i < req.sol_dense_steps;
      st.video_grid_h       = req.video_grid_h;
      st.video_grid_w       = req.video_grid_w;
      std::string ferr;
      MetalMiniMaxH3Transformer::Velocity v = req.dit->forward(st, &ferr);
      if (v.empty()) {
        return fail("denoise: step " + std::to_string(i) + ": " +
                    (ferr.empty() ? std::string("forward failed") : ferr));
      }
      ++forwards;
      // The GENERATED tails only. The head writes every row it was given,
      // conditioning included, and those are never stepped.
      {
        const auto* g =
            static_cast<const std::uint16_t*>(v.video.contents()) +
            (std::size_t)ncond * PE;
        for (std::size_t k = 0; k < vel_v.size(); ++k) {
          vel_v[k] = bf16_to_f32_(g[k]);
        }
      }
      if (naud > 0 && !v.audio.empty()) {
        const auto* g =
            static_cast<const std::uint16_t*>(v.audio.contents()) +
            (std::size_t)ncaud * AC;
        for (std::size_t k = 0; k < vel_a.size(); ++k) {
          vel_a[k] = bf16_to_f32_(g[k]);
        }
        have_aud = true;
      }
      // With the state the forward ran ON -- the scheduler has not
      // stepped it yet.
      if (mcache.armed()) {
        mcache.update(sigma_v, vx, vel_v.data(), have_aud ? ax : nullptr,
                      have_aud ? vel_a.data() : nullptr);
        cached_audio = have_aud;
      }
    }
    const auto t_fwd1 = std::chrono::steady_clock::now();

    // Step ONLY the generated tail. The conditioning rows lead the video
    // block precisely so this is a contiguous slice -- writing them
    // would dissolve the anchors the model is being asked to interpolate
    // between, and it would do it gradually enough to look like a weak
    // conditioning rather than a bug.
    double vrms = 0.0, arms = 0.0, xrms = 0.0, vxcorr = 0.0;
    double xarms = 0.0, axcorr = 0.0;
    {
      std::vector<float>& vel = vel_v;
      for (std::size_t k = 0; k < vel.size(); ++k) {
        vrms += (double)vel[k] * (double)vel[k];
      }
      vrms = vel.empty() ? 0.0 : std::sqrt(vrms / (double)vel.size());
      // Correlation of the predicted velocity with the latent it was
      // predicted FROM, taken BEFORE the scheduler overwrites it. A
      // network that has learned nothing about this input still emits a
      // scaled copy of it, and that reads as a plausible |v| with a
      // corr near -1: x0 = x + sigma*v then collapses to a rescaling of
      // the input noise, which is prompt-independent noise out.
      {
        double sxy = 0.0, sxx = 0.0;
        for (std::size_t k = 0; k < vel.size(); ++k) {
          sxy += (double)vel[k] * (double)vx[k];
          sxx += (double)vx[k] * (double)vx[k];
        }
        const double den =
            std::sqrt(sxx) * std::sqrt((double)vel.size()) * vrms;
        vxcorr = den > 0.0 ? sxy / den : 0.0;
      }
      // The clean estimate, taken before the step overwrites the state it
      // is estimated from.
      if (req.preview && req.preview_due && req.preview_due(i + 1, steps)) {
        preview_x0.resize(vel.size());
        if (!sv.estimate_x0(vel.data(), i, vx, preview_x0.data(),
                            vel.size())) {
          preview_x0.clear();
        }
      }
      const bool vok =
          res_ms ? sv.step_res(vel.data(), i, vx, vel.size(), &vprev)
                 : sv.step(vel.data(), i, vx, vel.size());
      if (!vok) {
        return fail("denoise: video step " + std::to_string(i) + " failed");
      }
      for (std::size_t k = 0; k < vel.size(); ++k) {
        xrms += (double)vx[k] * (double)vx[k];
      }
      xrms = vel.empty() ? 0.0 : std::sqrt(xrms / (double)vel.size());
    }
    if (have_aud) {
      // Only the generated tail is stepped, so both the velocity and the
      // state are read from `ncaud` on.
      std::vector<float>& vel = vel_a;
      for (std::size_t k = 0; k < vel.size(); ++k) {
        arms += (double)vel[k] * (double)vel[k];
      }
      arms = vel.empty() ? 0.0 : std::sqrt(arms / (double)vel.size());
      // The SAME correlation the video branch takes, for the same
      // reason. Without it the audio trace is |v| alone, and |v| ~ 1 is
      // exactly what both a working branch and a dead one produce -- a
      // network echoing a scaled copy of its input reads as a healthy
      // magnitude and lands at corr near -1, whereupon x0 = x + sigma*v
      // is just a rescaling of the input noise. Video had this from the
      // start; audio going without is why "video fine, audio noise"
      // could be traced this far without the trace saying anything.
      {
        double sxy = 0.0, sxx = 0.0;
        for (std::size_t k = 0; k < vel.size(); ++k) {
          sxy += (double)vel[k] * (double)ax[k];
          sxx += (double)ax[k] * (double)ax[k];
        }
        const double den =
            std::sqrt(sxx) * std::sqrt((double)vel.size()) * arms;
        axcorr = den > 0.0 ? sxy / den : 0.0;
      }
      const bool aok =
          res_ms ? sa.step_res(vel.data(), i, ax, vel.size(), &aprev)
                 : sa.step(vel.data(), i, ax, vel.size());
      if (!aok) {
        return fail("denoise: audio step " + std::to_string(i) + " failed");
      }
      for (std::size_t k = 0; k < vel.size(); ++k) {
        xarms += (double)ax[k] * (double)ax[k];
      }
      xarms = vel.empty() ? 0.0 : std::sqrt(xarms / (double)vel.size());
    }
    ++steps_run;

    // Per-step trace. Timing alone cannot separate "the GPU work never
    // launched" from "it launched and returned garbage" -- a kernel that
    // did not run leaves its output buffer untouched, which reads as a
    // fast step AND a velocity that is zero or frozen. So record both,
    // plus the latent's own RMS, which is what should walk from ~1
    // (noise) down as the trajectory converges.
    if (prof) {
      const auto t_end = std::chrono::steady_clock::now();
      const double fwd_ms =
          std::chrono::duration<double, std::milli>(t_fwd1 - t_fwd0).count();
      const double tot_ms =
          std::chrono::duration<double, std::milli>(t_end - t_fwd0).count();
      // What MotionCache made of the step, when it was asked: the score
      // is the predicted relative change of the output, and the sum is
      // what the threshold compares.
      std::string mc;
      if (mcache.armed()) {
        mc = reuse ? "  REUSED" : "  computed";
        if (mcache.last_score() >= 0.0) {
          mc += fmt(" (score {:.5f}, accumulated {:.5f})",
                    mcache.last_score(), mcache.accumulated())();
        }
      }
      sess->info(fmt(
          "h3-denoise step {:2}/{}  sigma {:.4f} -> {:.4f}  fwd {:7.1f} ms  "
          "total {:7.1f} ms  |v_vid| {:.5f}  |x_vid| {:.5f} "
          "corr_vid {:+.5f}  |  |v_aud| {:.5f}  |x_aud| {:.5f} "
          "corr_aud {:+.5f}{}",
          i + 1, steps, sv.sigmas()[(std::size_t)i],
          sv.sigmas()[(std::size_t)i + 1], fwd_ms, tot_ms, vrms, xrms,
          vxcorr, arms, xarms, axcorr, mc));
    }

    if (!preview_x0.empty()) {
      req.preview(i + 1, steps, preview_x0.data());
      preview_x0.clear();
    }
    if (req.progress && !req.progress(i + 1, steps)) { break; }
  }
  if (req.report != nullptr) {
    req.report->steps    = steps_run;
    req.report->forwards = forwards;
    req.report->reused   = reused_steps;
    req.report->motion_cache = mcache.armed();
  }
  // The one line that says whether the setting did anything. The ratio
  // is of FORWARDS, the dominant cost of a step, not of wall time.
  if (mcache.armed() && sess != nullptr) {
    sess->info(fmt("h3-denoise: motion_cache reused {} of {} steps -- {} "
                   "forwards ran, {:.2f}x fewer", reused_steps, steps_run,
                   forwards,
                   forwards > 0 ? (double)steps_run / forwards : 1.0));
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
