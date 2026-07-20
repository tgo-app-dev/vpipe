#include "generative-models/krea2/flow-sampler.h"

#include <algorithm>
#include <cmath>

namespace vpipe {
namespace genai {

namespace {

// Flow-matching time-shift of a base sigma by mu. exponential:
// sigma' = e^mu/(e^mu + (1/sigma - 1)); linear: mu*sigma/(1 + (mu-1)*sigma).
double
time_shift(double sigma, double mu, bool exponential)
{
  if (exponential) {
    const double emu = std::exp(mu);
    return emu / (emu + (1.0 / sigma - 1.0));
  }
  return mu * sigma / (1.0 + (mu - 1.0) * sigma);
}

// k-diffusion ancestral split: how much of the sigma_from->sigma_to move is
// deterministic (sigma_down) vs re-noised (sigma_up), scaled by eta.
void
ancestral(double sf, double st, double eta, double& sd, double& su)
{
  if (eta <= 0.0 || sf <= 0.0) { sd = st; su = 0.0; return; }
  const double up =
      eta * std::sqrt((st * st * (sf * sf - st * st)) / (sf * sf));
  su = std::min(st, up);
  sd = std::sqrt(std::max(0.0, st * st - su * su));
}

std::uint64_t
xorshift64(std::uint64_t& s)
{
  s ^= s << 13; s ^= s >> 7; s ^= s << 17;
  return s;
}

}  // namespace

// ---- FlowSchedulerSpec -------------------------------------------------

FlexData
FlowSchedulerSpec::to_flex() const
{
  FlexData fd = FlexData::make_object();
  auto o = fd.as_object();
  o.insert_or_assign("scheduler", FlexData::make_string("flow_match"));
  o.insert_or_assign("type", FlexData::make_string(type));
  o.insert_or_assign("steps", FlexData::make_int(steps));
  o.insert_or_assign("shift", FlexData::make_real(shift));
  o.insert_or_assign("shift_type", FlexData::make_string(shift_type));
  o.insert_or_assign("rho", FlexData::make_real(rho));
  if (dynamic_shift) {
    o.insert_or_assign("dynamic_shift", FlexData::make_bool(true));
    o.insert_or_assign("base_shift", FlexData::make_real(base_shift));
    o.insert_or_assign("max_shift", FlexData::make_real(max_shift));
    o.insert_or_assign("shift_terminal", FlexData::make_real(shift_terminal));
    o.insert_or_assign("base_seq", FlexData::make_int(base_seq));
    o.insert_or_assign("max_seq", FlexData::make_int(max_seq));
    o.insert_or_assign("num_train", FlexData::make_int(num_train));
  }
  return fd;
}

FlowSchedulerSpec
FlowSchedulerSpec::from_flex(const FlexData& fd, std::string* err)
{
  FlowSchedulerSpec s;
  if (!fd.is_object()) { return s; }
  auto o = fd.as_object();
  if (o.contains("type")) {
    s.type = std::string(o.at("type").as_string(s.type.c_str()));
  }
  if (o.contains("steps")) { s.steps = (int)o.at("steps").as_int(s.steps); }
  if (o.contains("shift")) { s.shift = o.at("shift").as_real(s.shift); }
  if (o.contains("shift_type")) {
    s.shift_type =
        std::string(o.at("shift_type").as_string(s.shift_type.c_str()));
  }
  if (o.contains("rho")) { s.rho = o.at("rho").as_real(s.rho); }
  if (o.contains("dynamic_shift")) {
    s.dynamic_shift = o.at("dynamic_shift").as_bool(false);
  }
  if (o.contains("base_shift")) {
    s.base_shift = o.at("base_shift").as_real(s.base_shift);
  }
  if (o.contains("max_shift")) {
    s.max_shift = o.at("max_shift").as_real(s.max_shift);
  }
  if (o.contains("shift_terminal")) {
    s.shift_terminal = o.at("shift_terminal").as_real(s.shift_terminal);
  }
  if (o.contains("base_seq")) {
    s.base_seq = (int)o.at("base_seq").as_int(s.base_seq);
  }
  if (o.contains("max_seq")) {
    s.max_seq = (int)o.at("max_seq").as_int(s.max_seq);
  }
  if (o.contains("num_train")) {
    s.num_train = (int)o.at("num_train").as_int(s.num_train);
  }
  if (s.steps < 1) { s.steps = 1; }
  if (s.type != "simple" && s.type != "karras" && s.type != "exponential") {
    if (err != nullptr) {
      *err = "unknown scheduler type '" + s.type + "'; using 'simple'";
    }
    s.type = "simple";
  }
  if (s.shift_type != "exponential" && s.shift_type != "linear") {
    s.shift_type = "exponential";
  }
  if (s.rho <= 0.0) { s.rho = 7.0; }
  return s;
}

std::vector<double>
FlowSchedulerSpec::sigmas() const
{
  return sigmas(img_seq_len);
}

std::vector<double>
FlowSchedulerSpec::sigmas(int img_seq_len_override) const
{
  const int S = steps < 1 ? 1 : steps;
  const bool expo = shift_type != "linear";

  // ---- FlowMatchEuler dynamic shifting (Qwen-Image / SD3) --------------
  // Reproduces diffusers FlowMatchEulerDiscreteScheduler.set_timesteps with
  // use_dynamic_shifting as the QwenImageEditPlus pipeline calls it: the base
  // sigmas the pipeline passes are linspace(1, 1/steps, steps) (NOT the
  // scheduler's internal 1/num_train default); mu = calculate_shift(
  // image_seq_len); each base sigma time-shifted by mu; then
  // stretch_shift_to_terminal so the last nonzero sigma == shift_terminal.
  // (num_train only maps sigma->t as t=sigma*num_train downstream; vpipe feeds
  // the DiT `timestep` = sigma directly, so it does not enter the schedule.)
  if (dynamic_shift) {
    // calculate_shift(image_seq_len, base_seq, max_seq, base_shift, max_shift)
    const double denom = (double)(max_seq - base_seq);
    const double m = denom != 0.0 ? (max_shift - base_shift) / denom : 0.0;
    const double b = base_shift - m * (double)base_seq;
    const double mu = (double)img_seq_len_override * m + b;

    std::vector<double> sig((std::size_t)S + 1);
    for (int i = 0; i < S; ++i) {
      const double r = (S == 1) ? 0.0 : (double)i / (double)(S - 1);
      // base = linspace(1.0, 1/S, S) -- the pipeline's explicit sigmas arg.
      const double base = 1.0 + r * ((1.0 / (double)S) - 1.0);
      sig[(std::size_t)i] = time_shift(base, mu, expo);
    }
    // stretch_shift_to_terminal: 1 - (1 - sig) / ((1 - sig[S-1])/(1 - term)).
    if (shift_terminal > 0.0 && S >= 1) {
      const double last_omz = 1.0 - sig[(std::size_t)S - 1];
      const double scale = last_omz / (1.0 - shift_terminal);
      if (scale != 0.0) {
        for (int i = 0; i < S; ++i) {
          sig[(std::size_t)i] = 1.0 - (1.0 - sig[(std::size_t)i]) / scale;
        }
      }
    }
    sig[(std::size_t)S] = 0.0;   // terminal
    return sig;
  }

  const double smax = 1.0, smin = 1.0 / (double)S;

  std::vector<double> base((std::size_t)S);
  if (type == "karras") {
    const double mi = std::pow(smin, 1.0 / rho), ma = std::pow(smax, 1.0 / rho);
    for (int i = 0; i < S; ++i) {
      const double r = (S == 1) ? 0.0 : (double)i / (double)(S - 1);
      base[(std::size_t)i] = std::pow(ma + r * (mi - ma), rho);
    }
  } else if (type == "exponential") {
    const double lma = std::log(smax), lmi = std::log(smin);
    for (int i = 0; i < S; ++i) {
      const double r = (S == 1) ? 0.0 : (double)i / (double)(S - 1);
      base[(std::size_t)i] = std::exp(lma + r * (lmi - lma));
    }
  } else {   // simple: linspace(1, 1/S, S)
    for (int i = 0; i < S; ++i) {
      base[(std::size_t)i] =
          (S == 1) ? 1.0
                   : 1.0 + (double)i * ((1.0 / (double)S) - 1.0) / (double)(S - 1);
    }
  }

  std::vector<double> sig((std::size_t)S + 1);
  for (int i = 0; i < S; ++i) {
    sig[(std::size_t)i] = time_shift(base[(std::size_t)i], shift, expo);
  }
  sig[(std::size_t)S] = 0.0;   // terminal
  return sig;
}

// ---- FlowSamplerSpec ---------------------------------------------------

std::string
FlowSamplerSpec::canon_method(const std::string& m, bool* ok)
{
  if (ok != nullptr) { *ok = true; }
  if (m == "euler" || m == "heun" || m == "dpmpp_2m" || m == "dpmpp_sde") {
    return m;
  }
  if (m == "dpm++_2m" || m == "dpmpp2m") { return "dpmpp_2m"; }
  if (m == "dpm++_sde" || m == "dpmppsde") { return "dpmpp_sde"; }
  if (ok != nullptr) { *ok = false; }
  return "euler";
}

FlexData
FlowSamplerSpec::to_flex() const
{
  FlexData fd = FlexData::make_object();
  auto o = fd.as_object();
  o.insert_or_assign("sampler", FlexData::make_string("flow_match"));
  o.insert_or_assign("method", FlexData::make_string(method));
  o.insert_or_assign("eta", FlexData::make_real(eta));
  o.insert_or_assign("s_noise", FlexData::make_real(s_noise));
  o.insert_or_assign("seed", FlexData::make_int((std::int64_t)seed));
  return fd;
}

FlowSamplerSpec
FlowSamplerSpec::from_flex(const FlexData& fd, std::string* err)
{
  FlowSamplerSpec s;
  if (!fd.is_object()) { return s; }
  auto o = fd.as_object();
  if (o.contains("method")) {
    bool ok = true;
    s.method =
        canon_method(std::string(o.at("method").as_string("euler")), &ok);
    if (!ok && err != nullptr) {
      *err = "unknown sampler method; using 'euler'";
    }
  }
  if (o.contains("eta")) { s.eta = o.at("eta").as_real(s.eta); }
  if (o.contains("s_noise")) { s.s_noise = o.at("s_noise").as_real(s.s_noise); }
  if (o.contains("seed")) {
    s.seed = (std::uint64_t)o.at("seed").as_int((std::int64_t)s.seed);
  }
  return s;
}

// ---- FlowSampler -------------------------------------------------------

FlowSampler::FlowSampler(FlowSamplerSpec sampler, FlowSchedulerSpec scheduler)
  : _sampler(std::move(sampler)), _scheduler(std::move(scheduler))
{
  _sigmas = _scheduler.sigmas();
  reset();
}

void
FlowSampler::reset()
{
  _old_denoised.clear();
  _have_prev = false;
  _t_prev = 0.0;
  _rng = _sampler.seed != 0 ? _sampler.seed : 0x9E3779B97F4A7C15ULL;
}

std::vector<float>
FlowSampler::velocity_(const std::vector<float>& x, double sigma,
                       const DenoiseFn& denoise) const
{
  return denoise(x, sigma);
}

std::vector<float>
FlowSampler::denoised_(const std::vector<float>& x, double sigma,
                       const DenoiseFn& denoise) const
{
  std::vector<float> v = denoise(x, sigma);
  if (v.size() != x.size()) { return {}; }
  std::vector<float> d(x.size());
  for (std::size_t k = 0; k < x.size(); ++k) {
    d[k] = x[k] - (float)sigma * v[k];
  }
  return d;
}

void
FlowSampler::gaussian_(std::vector<float>& out)
{
  // xorshift64 uniforms -> Box-Muller normals (deterministic given seed).
  for (std::size_t k = 0; k < out.size(); k += 2) {
    double u1 = (double)(xorshift64(_rng) >> 11) * (1.0 / 9007199254740992.0);
    const double u2 =
        (double)(xorshift64(_rng) >> 11) * (1.0 / 9007199254740992.0);
    if (u1 < 1e-12) { u1 = 1e-12; }
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double a = 2.0 * M_PI * u2;
    out[k] = (float)(r * std::cos(a));
    if (k + 1 < out.size()) { out[k + 1] = (float)(r * std::sin(a)); }
  }
}

void
FlowSampler::step(int i, std::vector<float>& x, const DenoiseFn& denoise)
{
  const double si = _sigmas[(std::size_t)i];
  const double sn = _sigmas[(std::size_t)i + 1];
  const std::string& m = _sampler.method;

  if (m == "heun") {
    const std::vector<float> v1 = velocity_(x, si, denoise);
    if (v1.size() != x.size()) { return; }
    const double dt = sn - si;
    if (sn <= 0.0) {
      for (std::size_t k = 0; k < x.size(); ++k) { x[k] += (float)(dt * v1[k]); }
      return;
    }
    std::vector<float> x2(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
      x2[k] = x[k] + (float)(dt * v1[k]);
    }
    const std::vector<float> v2 = velocity_(x2, sn, denoise);
    if (v2.size() == x.size()) {
      for (std::size_t k = 0; k < x.size(); ++k) {
        x[k] += (float)(dt * 0.5 * ((double)v1[k] + (double)v2[k]));
      }
    } else {
      for (std::size_t k = 0; k < x.size(); ++k) { x[k] += (float)(dt * v1[k]); }
    }
    return;
  }

  if (m == "dpmpp_2m") {
    const std::vector<float> den = denoised_(x, si, denoise);
    if (den.size() != x.size()) { return; }
    if (sn <= 0.0) {
      x = den;                       // terminal: land on the x0 prediction
    } else {
      const double t = -std::log(si), t_next = -std::log(sn);
      const double h = t_next - t, eh = std::exp(-h);
      if (!_have_prev) {
        for (std::size_t k = 0; k < x.size(); ++k) {
          x[k] = (float)(eh * x[k] - (eh - 1.0) * den[k]);
        }
      } else {
        const double r = (t - _t_prev) / h;   // h_last / h
        const double a = 1.0 + 1.0 / (2.0 * r), b = 1.0 / (2.0 * r);
        for (std::size_t k = 0; k < x.size(); ++k) {
          const double dd = a * den[k] - b * _old_denoised[k];
          x[k] = (float)(eh * x[k] - (eh - 1.0) * dd);
        }
      }
      _t_prev = t;
    }
    _old_denoised = den;
    _have_prev = true;
    return;
  }

  if (m == "dpmpp_sde") {
    const std::vector<float> den = denoised_(x, si, denoise);
    if (den.size() != x.size()) { return; }
    if (sn <= 0.0) {
      x = den;                       // terminal Euler == land on x0
      return;
    }
    const double t = -std::log(si), t_next = -std::log(sn);
    const double r = 0.5, h = t_next - t, s = t + h * r;
    const double fac = 1.0 / (2.0 * r);      // = 1
    const double sigma_s = std::exp(-s);     // = sqrt(si*sn)

    double sd1, su1; ancestral(si, sigma_s, _sampler.eta, sd1, su1);
    const double a1 = sd1 / si;
    std::vector<float> x2(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
      x2[k] = (float)(a1 * x[k] + (1.0 - a1) * den[k]);
    }
    if (su1 > 0.0) {
      std::vector<float> nz(x.size());
      gaussian_(nz);
      for (std::size_t k = 0; k < x.size(); ++k) {
        x2[k] += (float)(nz[k] * _sampler.s_noise * su1);
      }
    }
    const std::vector<float> den2 = denoised_(x2, sigma_s, denoise);
    if (den2.size() != x.size()) { return; }

    double sd2, su2; ancestral(si, sn, _sampler.eta, sd2, su2);
    const double a2 = sd2 / si;
    for (std::size_t k = 0; k < x.size(); ++k) {
      const double dd = (1.0 - fac) * den[k] + fac * den2[k];
      x[k] = (float)(a2 * x[k] + (1.0 - a2) * dd);
    }
    if (su2 > 0.0) {
      std::vector<float> nz2(x.size());
      gaussian_(nz2);
      for (std::size_t k = 0; k < x.size(); ++k) {
        x[k] += (float)(nz2[k] * _sampler.s_noise * su2);
      }
    }
    return;
  }

  // euler (default).
  const std::vector<float> v = velocity_(x, si, denoise);
  if (v.size() != x.size()) { return; }
  const double dt = sn - si;
  for (std::size_t k = 0; k < x.size(); ++k) { x[k] += (float)(dt * v[k]); }
}

}  // namespace genai
}  // namespace vpipe
