#include "generative-models/minimax-h3/minimax-h3-denoise.h"

#include "generative-models/minimax-h3/minimax-h3-scheduler.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
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
  if (!sv.set_timesteps(req.num_steps) || !sa.set_timesteps(req.num_steps)) {
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

  std::vector<float> uniq;
  std::vector<int>   row_idx;

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
    std::vector<std::vector<float>> sched;
    sched.reserve((std::size_t)steps);
    for (int i = 0; i < steps; ++i) {
      std::vector<float> u;
      std::vector<int>   ri;
      minimax_h3::build_row_timesteps(
          L, sv.timesteps()[(std::size_t)i], sa.timesteps()[(std::size_t)i],
          req.condition_timestep, &u, &ri, req.condition_audio_timestep);
      sched.push_back(std::move(u));
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
      baked = req.dit->bake_adaln(sched, &berr);
    }
    if (!baked && sess != nullptr) {
      sess->log_normal(fmt("denoise: AdaLN not baked ({}); running the "
                           "projections per step", berr));
    }
  }

  for (int i = 0; i < steps; ++i) {
    // Every row's timestep, this step. The conditioning rows keep their
    // own value while the generated video and audio rows walk their two
    // schedules -- that is what one forward over rows at DIFFERENT noise
    // levels means, and it is the whole reason the AdaLN table is
    // indexed per row.
    // t = 1 - sigma, t = 1 meaning CLEAN. The transformer consumes it
    // unscaled and recovers sigma itself; feeding a pure-noise latent
    // t = 0 would tell the model it is already done.
    const float tv = sv.timesteps()[(std::size_t)i];
    const float ta = sa.timesteps()[(std::size_t)i];
    const float tc = req.condition_timestep;
    minimax_h3::build_row_timesteps(L, tv, ta, tc, &uniq, &row_idx,
                                    req.condition_audio_timestep);

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
    st.schedule_index     = baked ? i : -1;
    st.video_grid_h       = req.video_grid_h;
    st.video_grid_w       = req.video_grid_w;
    std::string ferr;
    const auto t_fwd0 = std::chrono::steady_clock::now();
    MetalMiniMaxH3Transformer::Velocity v = req.dit->forward(st, &ferr);
    const auto t_fwd1 = std::chrono::steady_clock::now();
    if (v.empty()) {
      return fail("denoise: step " + std::to_string(i) + ": " +
                  (ferr.empty() ? std::string("forward failed") : ferr));
    }

    // Step ONLY the generated tail. The conditioning rows lead the video
    // block precisely so this is a contiguous slice -- writing them
    // would dissolve the anchors the model is being asked to interpolate
    // between, and it would do it gradually enough to look like a weak
    // conditioning rather than a bug.
    double vrms = 0.0, arms = 0.0, xrms = 0.0, vxcorr = 0.0;
    double xarms = 0.0, axcorr = 0.0;
    {
      const auto* g = static_cast<const std::uint16_t*>(v.video.contents());
      std::vector<float> vel((std::size_t)nvid * PE);
      for (std::size_t k = 0; k < vel.size(); ++k) {
        vel[k] = bf16_to_f32_(g[(std::size_t)ncond * PE + k]);
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
        const float* xb = req.video + (std::size_t)ncond * PE;
        double sxy = 0.0, sxx = 0.0;
        for (std::size_t k = 0; k < vel.size(); ++k) {
          sxy += (double)vel[k] * (double)xb[k];
          sxx += (double)xb[k] * (double)xb[k];
        }
        const double den = std::sqrt(sxx) * std::sqrt((double)vel.size()) * vrms;
        vxcorr = den > 0.0 ? sxy / den : 0.0;
      }
      float* vx = req.video + (std::size_t)ncond * PE;
      const bool vok =
          res_ms ? sv.step_res(vel.data(), i, vx, vel.size(), &vprev)
                 : sv.step(vel.data(), i, vx, vel.size());
      if (!vok) {
        return fail("denoise: video step " + std::to_string(i) + " failed");
      }
      const float* x = req.video + (std::size_t)ncond * PE;
      for (std::size_t k = 0; k < vel.size(); ++k) {
        xrms += (double)x[k] * (double)x[k];
      }
      xrms = vel.empty() ? 0.0 : std::sqrt(xrms / (double)vel.size());
    }
    if (naud > 0 && !v.audio.empty()) {
      // The head writes every audio row it was given, reference rows
      // included; only the generated tail is stepped, so both the
      // velocity and the state are read from `ncaud` on.
      const auto* g = static_cast<const std::uint16_t*>(v.audio.contents()) +
                      (std::size_t)ncaud * AC;
      float* ax = req.audio + (std::size_t)ncaud * AC;
      std::vector<float> vel((std::size_t)naud * AC);
      for (std::size_t k = 0; k < vel.size(); ++k) {
        vel[k] = bf16_to_f32_(g[k]);
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
      sess->info(fmt(
          "h3-denoise step {:2}/{}  sigma {:.4f} -> {:.4f}  fwd {:7.1f} ms  "
          "total {:7.1f} ms  |v_vid| {:.5f}  |x_vid| {:.5f} "
          "corr_vid {:+.5f}  |  |v_aud| {:.5f}  |x_aud| {:.5f} "
          "corr_aud {:+.5f}",
          i + 1, steps, sv.sigmas()[(std::size_t)i],
          sv.sigmas()[(std::size_t)i + 1], fwd_ms, tot_ms, vrms, xrms,
          vxcorr, arms, xarms, axcorr));
    }

    if (req.progress && !req.progress(i + 1, steps)) { break; }
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
