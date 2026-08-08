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
  const int ncond = L.num_condition_rows;
  const int nvid  = L.num_video_rows;         // generated rows only
  const int naud  = L.num_audio_rows;
  const int vrows = (int)L.video_indices.size();
  if (vrows != ncond + nvid) {
    return fail("denoise: layout video_indices does not match "
                "condition + generated rows");
  }
  if (nvid <= 0) { return fail("denoise: nothing to generate"); }

  MiniMaxH3Scheduler sv(req.video_shift);
  MiniMaxH3Scheduler sa(req.audio_shift);
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
  SharedBuffer ab = mc->make_shared_buffer((std::size_t)naud * AC * 2);
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
    minimax_h3::build_row_timesteps(L, tv, ta, tc, &uniq, &row_idx);

    {
      auto* d = static_cast<std::uint16_t*>(vb.contents());
      for (std::size_t k = 0; k < (std::size_t)vrows * PE; ++k) {
        d[k] = f32_to_bf16_(req.video[k]);
      }
    }
    if (naud > 0) {
      auto* d = static_cast<std::uint16_t*>(ab.contents());
      for (std::size_t k = 0; k < (std::size_t)naud * AC; ++k) {
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
      if (!sv.step(vel.data(), i, req.video + (std::size_t)ncond * PE,
                   vel.size())) {
        return fail("denoise: video step " + std::to_string(i) + " failed");
      }
      const float* x = req.video + (std::size_t)ncond * PE;
      for (std::size_t k = 0; k < vel.size(); ++k) {
        xrms += (double)x[k] * (double)x[k];
      }
      xrms = vel.empty() ? 0.0 : std::sqrt(xrms / (double)vel.size());
    }
    if (naud > 0 && !v.audio.empty()) {
      const auto* g = static_cast<const std::uint16_t*>(v.audio.contents());
      std::vector<float> vel((std::size_t)naud * AC);
      for (std::size_t k = 0; k < vel.size(); ++k) {
        vel[k] = bf16_to_f32_(g[k]);
        arms += (double)vel[k] * (double)vel[k];
      }
      arms = vel.empty() ? 0.0 : std::sqrt(arms / (double)vel.size());
      if (!sa.step(vel.data(), i, req.audio, vel.size())) {
        return fail("denoise: audio step " + std::to_string(i) + " failed");
      }
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
          "total {:7.1f} ms  |v_vid| {:.5f}  |v_aud| {:.5f}  |x_vid| {:.5f}  "
          "corr(v,x) {:+.5f}",
          i + 1, steps, sv.sigmas()[(std::size_t)i],
          sv.sigmas()[(std::size_t)i + 1], fwd_ms, tot_ms, vrms, arms, xrms,
          vxcorr));
    }

    if (req.progress && !req.progress(i + 1, steps)) { break; }
  }
  return true;
}

}  // namespace genai
}  // namespace vpipe
