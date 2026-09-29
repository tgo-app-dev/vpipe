#include "generative-models/shared/motion-cache.h"

#include "common/vpipe-format.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vpipe {
namespace genai {
namespace motion_cache {

// A port of the reference's MiniMaxH3MotionCacheState, with the tensors
// replaced by gathers over host state. The arithmetic follows it line for
// line -- including the parts that look odd, e.g. that a reused step does
// not update the cache, so every score while skipping is measured against
// the last COMPUTED input and the accumulated sum counts that drift once
// per skipped step. Those are the reference's semantics, and a threshold
// tuned against it only transfers if they are kept.
struct Cache::Impl {
  Config cfg;
  bool   armed = false;
  double start_sigma = 1.0;
  double end_sigma   = 0.0;
  float  sign        = 1.0f;

  // Gathers and their grids.
  std::vector<std::uint32_t> vg, ag;
  int C = 0, T = 0, P = 0;
  std::size_t nv = 0, na = 0;

  // Counters.
  int    step = 0, calls = 0, skipped = 0, consec = 0;
  double accum = 0.0, last_score = -1.0;

  // The last COMPUTED forward, subsampled.
  bool have_prev = false;
  std::vector<float> prev_vx, prev_ax, prev_vo, prev_ao;
  // Per (t, position) motion weight, mean 1.
  std::vector<float> w;
  bool   have_vrate = false, have_arate = false;
  double vrate = 0.0, arate = 0.0, vnorm = 1e-6, anorm = 1e-6;
  // The full residual, v + sign * x, generated rows.
  bool have_resid = false;
  std::vector<float> rv, ra;

  void
  reset_run()
  {
    step = calls = skipped = consec = 0;
    accum = 0.0;
    last_score = -1.0;
    have_prev = have_vrate = have_arate = have_resid = false;
    vrate = arate = 0.0;
    vnorm = anorm = 1e-6;
    prev_vx.clear(); prev_ax.clear(); prev_vo.clear(); prev_ao.clear();
    w.clear(); rv.clear(); ra.clear();
  }

  static void
  gather(const std::vector<std::uint32_t>& g, const float* src,
         std::vector<float>* out)
  {
    out->resize(g.size());
    for (std::size_t k = 0; k < g.size(); ++k) { (*out)[k] = src[g[k]]; }
  }

  // mean(|a - b| * w) over [C][T*P], w broadcast over channels. An
  // empty `wt` weighs every element 1 -- the audio stream, and a video
  // with no weights yet.
  double
  mean_change(const std::vector<float>& a, const std::vector<float>& b,
              const std::vector<float>* wt) const
  {
    if (a.empty() || a.size() != b.size()) { return 0.0; }
    double s = 0.0;
    if (wt != nullptr && !wt->empty()) {
      const std::size_t tp = wt->size();
      for (std::size_t k = 0; k < a.size(); ++k) {
        s += std::fabs((double)a[k] - (double)b[k]) * (*wt)[k % tp];
      }
    } else {
      for (std::size_t k = 0; k < a.size(); ++k) {
        s += std::fabs((double)a[k] - (double)b[k]);
      }
    }
    return s / (double)a.size();
  }

  double
  mean_norm(const std::vector<float>& a, const std::vector<float>* wt) const
  {
    if (a.empty()) { return 1e-6; }
    double s = 0.0;
    if (wt != nullptr && !wt->empty()) {
      const std::size_t tp = wt->size();
      for (std::size_t k = 0; k < a.size(); ++k) {
        s += std::fabs((double)a[k]) * (*wt)[k % tp];
      }
    } else {
      for (float x : a) { s += std::fabs((double)x); }
    }
    return std::max(s / (double)a.size(), 1e-6);
  }

  // The reference's build_motion_weights over the subsampled clean
  // estimate d, [C][T][P]: per position, the channel-mean |d_t - d_t-1|
  // (frame 0 borrows frame 1's), normalised to mean 1, then
  // 1 + strength * that, renormalised to mean 1.
  void
  motion_weights(const std::vector<float>& d, std::vector<float>* out) const
  {
    const std::size_t tp = (std::size_t)T * P;
    out->assign(tp, 1.0f);
    if (T < 2 || cfg.motion_strength == 0.0 || P <= 0 || C <= 0) { return; }
    std::vector<double> fd(tp, 0.0);
    for (int c = 0; c < C; ++c) {
      const float* dc = d.data() + (std::size_t)c * tp;
      for (int t = 1; t < T; ++t) {
        for (int p = 0; p < P; ++p) {
          fd[(std::size_t)t * P + p] +=
              std::fabs((double)dc[(std::size_t)t * P + p] -
                        (double)dc[(std::size_t)(t - 1) * P + p]);
        }
      }
    }
    for (std::size_t k = (std::size_t)P; k < tp; ++k) { fd[k] /= C; }
    for (int p = 0; p < P; ++p) { fd[(std::size_t)p] = fd[(std::size_t)P + p]; }
    double mean = 0.0;
    for (double x : fd) { mean += x; }
    mean = std::max(mean / (double)tp, 1e-6);
    double wmean = 0.0;
    for (std::size_t k = 0; k < tp; ++k) {
      fd[k] = 1.0 + cfg.motion_strength * (fd[k] / mean);
      wmean += fd[k];
    }
    wmean = std::max(wmean / (double)tp, 1e-6);
    for (std::size_t k = 0; k < tp; ++k) {
      (*out)[k] = (float)(fd[k] / wmean);
    }
  }

  bool
  active(double sigma) const
  {
    return end_sigma < sigma && sigma <= start_sigma;
  }
};

Cache::Cache() : _impl(std::make_unique<Impl>()) {}
Cache::~Cache() = default;
Cache::Cache(Cache&&) noexcept = default;
Cache& Cache::operator=(Cache&&) noexcept = default;

bool
Cache::begin(const FlexData* accel, double start_sigma, double end_sigma,
             float x0_sign)
{
  Impl& m = *_impl;
  m.reset_run();
  m.vg.clear(); m.ag.clear();
  m.C = m.T = m.P = 0;
  m.nv = m.na = 0;
  m.cfg = config_from_flex(accel);
  // Settled by the stage already; clamped again because a bag can come
  // from anywhere, and every one of these has a value that makes the
  // arithmetic below meaningless rather than merely aggressive.
  m.cfg.threshold = std::max(0.0, m.cfg.threshold);
  m.cfg.motion_strength = std::max(0.0, m.cfg.motion_strength);
  m.cfg.warmup_steps = std::max(0, m.cfg.warmup_steps);
  m.cfg.max_skips = std::max(1, m.cfg.max_skips);
  m.cfg.subsample = std::max(1, m.cfg.subsample);
  m.start_sigma = start_sigma;
  m.end_sigma   = end_sigma;
  m.sign        = x0_sign < 0.0f ? -1.0f : 1.0f;
  m.armed = m.cfg.enabled && m.cfg.start < m.cfg.end;
  return m.armed;
}

bool
Cache::armed() const
{
  return _impl->armed;
}

int
Cache::subsample() const
{
  return _impl->cfg.subsample;
}

void
Cache::set_video(const std::uint32_t* gather, int channels, int frames,
                 int plane, std::size_t video_elems)
{
  Impl& m = *_impl;
  const std::size_t n = (std::size_t)std::max(0, channels) *
                        (std::size_t)std::max(0, frames) *
                        (std::size_t)std::max(0, plane);
  m.vg.assign(gather, gather + (gather != nullptr ? n : 0));
  m.C = channels;
  m.T = frames;
  m.P = plane;
  m.nv = video_elems;
}

void
Cache::set_audio(const std::uint32_t* gather, std::size_t count,
                 std::size_t audio_elems)
{
  Impl& m = *_impl;
  m.ag.assign(gather, gather + (gather != nullptr ? count : 0));
  m.na = audio_elems;
}

bool
Cache::decide(double sigma, const float* video_x, const float* audio_x)
{
  Impl& m = *_impl;
  m.last_score = -1.0;
  if (!m.armed) { return false; }
  ++m.step;
  ++m.calls;
  if (m.step <= m.cfg.warmup_steps || !m.active(sigma)) {
    m.accum  = 0.0;
    m.consec = 0;
    return false;
  }
  if (!m.have_resid || m.vg.empty()) { return false; }
  // No gain yet: two forwards have to have run before one can be
  // predicted. A model without an audio stream has no audio gain to
  // wait for.
  const bool audio = !m.ag.empty() && audio_x != nullptr;
  if (!m.have_vrate || (audio && !m.have_arate)) { return false; }

  std::vector<float> vx, ax;
  Impl::gather(m.vg, video_x, &vx);
  double score =
      m.vrate * m.mean_change(vx, m.prev_vx, &m.w) / m.vnorm;
  if (audio) {
    Impl::gather(m.ag, audio_x, &ax);
    score = std::max(
        score, m.arate * m.mean_change(ax, m.prev_ax, nullptr) / m.anorm);
  }
  m.last_score = score;
  m.accum += score;
  const bool reuse = m.accum < m.cfg.threshold && m.consec < m.cfg.max_skips;
  if (reuse) {
    ++m.consec;
    ++m.skipped;
  } else {
    m.accum  = 0.0;
    m.consec = 0;
  }
  return reuse;
}

void
Cache::predict(const float* video_x, float* video_v, const float* audio_x,
               float* audio_v) const
{
  const Impl& m = *_impl;
  // v' = (v + s x) - s x': the reference's out' = x' + (out - x) with
  // out = -s v.
  for (std::size_t k = 0; k < m.nv && k < m.rv.size(); ++k) {
    video_v[k] = m.rv[k] - m.sign * video_x[k];
  }
  if (audio_x != nullptr && audio_v != nullptr) {
    for (std::size_t k = 0; k < m.na && k < m.ra.size(); ++k) {
      audio_v[k] = m.ra[k] - m.sign * audio_x[k];
    }
  }
}

void
Cache::update(double sigma, const float* video_x, const float* video_v,
              const float* audio_x, const float* audio_v)
{
  Impl& m = *_impl;
  if (!m.armed || m.vg.empty()) { return; }
  std::vector<float> vx, vo, ax, ao;
  Impl::gather(m.vg, video_x, &vx);
  Impl::gather(m.vg, video_v, &vo);
  const bool audio = !m.ag.empty() && audio_x != nullptr &&
                     audio_v != nullptr;
  if (audio) {
    Impl::gather(m.ag, audio_x, &ax);
    Impl::gather(m.ag, audio_v, &ao);
  }
  // The clean estimate the motion is read from: x0 = x + s sigma v.
  std::vector<float> d(vx.size());
  const float ss = m.sign * (float)sigma;
  for (std::size_t k = 0; k < d.size(); ++k) { d[k] = vx[k] + ss * vo[k]; }
  std::vector<float> w;
  m.motion_weights(d, &w);

  if (m.have_prev) {
    const double vin  = m.mean_change(vx, m.prev_vx, &w);
    const double vout = m.mean_change(vo, m.prev_vo, &w);
    if (vin > 1e-6) {
      m.vrate = vout / vin;
      m.have_vrate = true;
    }
    if (audio) {
      const double ain  = m.mean_change(ax, m.prev_ax, nullptr);
      const double aout = m.mean_change(ao, m.prev_ao, nullptr);
      if (ain > 1e-6) {
        m.arate = aout / ain;
        m.have_arate = true;
      }
    }
  }
  m.vnorm = m.mean_norm(vo, &w);
  m.anorm = audio ? m.mean_norm(ao, nullptr) : 1e-6;
  m.prev_vx = std::move(vx);
  m.prev_vo = std::move(vo);
  m.prev_ax = std::move(ax);
  m.prev_ao = std::move(ao);
  m.w = std::move(w);
  m.have_prev = true;

  m.rv.resize(m.nv);
  for (std::size_t k = 0; k < m.nv; ++k) {
    m.rv[k] = video_v[k] + m.sign * video_x[k];
  }
  m.ra.resize(audio ? m.na : 0);
  for (std::size_t k = 0; k < m.ra.size(); ++k) {
    m.ra[k] = audio_v[k] + m.sign * audio_x[k];
  }
  m.have_resid = true;
}

int
Cache::calls() const
{
  return _impl->calls;
}

int
Cache::skipped() const
{
  return _impl->skipped;
}

double
Cache::last_score() const
{
  return _impl->last_score;
}

double
Cache::accumulated() const
{
  return _impl->accum;
}

std::string
Cache::describe() const
{
  const Impl& m = *_impl;
  return fmt("threshold {:.3g}, motion strength {:.3g}, warm-up {}, at "
             "most {} in a row, window {:.2f}-{:.2f} (sigma {:.4f} -> "
             "{:.4f}), subsample {}", m.cfg.threshold,
             m.cfg.motion_strength, m.cfg.warmup_steps, m.cfg.max_skips,
             m.cfg.start, m.cfg.end, m.start_sigma, m.end_sigma,
             m.cfg.subsample)();
}

}  // namespace motion_cache
}  // namespace genai
}  // namespace vpipe
