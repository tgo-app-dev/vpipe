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
  if (type == "boogu_v1") {
    o.insert_or_assign("base_shift", FlexData::make_real(base_shift));
    o.insert_or_assign("max_shift", FlexData::make_real(max_shift));
    o.insert_or_assign("base_seq", FlexData::make_int(base_seq));
    o.insert_or_assign("max_seq", FlexData::make_int(max_seq));
    o.insert_or_assign("seq_len", FlexData::make_int(seq_len));
  }
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
  if (o.contains("seq_len")) {
    s.seq_len = (int)o.at("seq_len").as_int(s.seq_len);
  }
  if (s.steps < 1) { s.steps = 1; }
  if (s.type == "boogu_v1") { return s; }
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

  // ---- Boogu-Image v1 time shifting (ASCENDING sigma) ------------------
  // Reproduces the checkpoint's FlowMatchEulerDiscreteScheduler.set_timesteps:
  // base = linspace(0, 1, S+1)[:-1]; mu = lin(seq_len) over the
  // (base_seq, base_shift) -> (max_seq, max_shift) line; each base t pushed
  // through the logistic shift; then a terminal 1.0 appended (the reference
  // concatenates torch.ones(1) so the last Euler step lands on clean).
  // NOTE the inverted convention: 0 is noise and 1 is clean here, so the
  // returned array INCREASES. Callers must integrate x += (t[i+1]-t[i]) * v.
  if (type == "boogu_v1") {
    const int L = seq_len > 0 ? seq_len : img_seq_len_override;
    const double denom = (double)(max_seq - base_seq);
    const double m = denom != 0.0 ? (max_shift - base_shift) / denom : 0.0;
    const double b = base_shift - m * (double)base_seq;
    const double mu = (double)L * m + b;
    const double num = std::exp(mu);
    std::vector<double> sig((std::size_t)S + 1);
    for (int i = 0; i < S; ++i) {
      const double t = (double)i / (double)S;      // linspace(0,1,S+1)[:-1]
      double t1 = 1.0 - t;                          // the reference flips...
      const double e = 1e-8;
      if (t1 < e) { t1 = e; } else if (t1 > 1.0 - e) { t1 = 1.0 - e; }
      const double y = num / (num + (1.0 / t1 - 1.0));   // sigma exponent 1
      sig[(std::size_t)i] = 1.0 - y;                // ...and flips back
    }
    sig[(std::size_t)S] = 1.0;                      // terminal: fully clean
    return sig;
  }

  // ---- UniPC flow sigmas (Wan) -----------------------------------------
  // diffusers UniPCMultistepScheduler.set_timesteps under use_flow_sigmas,
  // which is what every Wan video checkpoint ships. Two details separate it
  // from the dynamic_shift branch below and both matter:
  //   * the base grid is linspace(1, 1/num_train, S+1)[:-1] -- S+1 points
  //     with the last dropped, NOT linspace(1, 1/num_train, S). The two
  //     agree only at the first point.
  //   * the shift is the scheduler's own static curve
  //     s' = shift*s / (1 + (shift-1)*s), not the exponential/linear pair
  //     `shift_type` selects, so shift_type does not apply here.
  // sigma[0] is nudged off an exact 1 because the first UniPC update takes
  // log(alpha) = log(1 - sigma), which is -inf at 1.
  if (type == "unipc_flow") {
    std::vector<double> sig((std::size_t)S + 1);
    const double lo = 1.0 / (double)(num_train > 0 ? num_train : 1000);
    for (int i = 0; i < S; ++i) {
      const double r = (double)i / (double)S;          // [:-1] of S+1 points
      const double base = 1.0 + r * (lo - 1.0);
      sig[(std::size_t)i] = shift * base / (1.0 + (shift - 1.0) * base);
    }
    if (S > 0 && std::fabs(sig[0] - 1.0) < 1e-6) { sig[0] -= 1e-6; }
    sig[(std::size_t)S] = 0.0;                          // final_sigmas zero
    return sig;
  }

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
  if (m == "euler" || m == "heun" || m == "dpmpp_2m" || m == "dpmpp_sde" ||
      m == "dmd" || m == "unipc") {
    return m;
  }
  if (m == "unipc_multistep" || m == "uni_pc") { return "unipc"; }
  if (m == "dpm++_2m" || m == "dpmpp2m") { return "dpmpp_2m"; }
  if (m == "dpm++_sde" || m == "dpmppsde") { return "dpmpp_sde"; }
  if (m == "dmd_student" || m == "turbo") { return "dmd"; }
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
  if (method == "dmd") {
    o.insert_or_assign("conditioning_sigma",
                       FlexData::make_real(conditioning_sigma));
  }
  if (method == "unipc") {
    o.insert_or_assign("order", FlexData::make_int(order));
    o.insert_or_assign("solver_type",
                       FlexData::make_string(solver_bh2 ? "bh2" : "bh1"));
  }
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
  if (o.contains("conditioning_sigma")) {
    s.conditioning_sigma =
        o.at("conditioning_sigma").as_real(s.conditioning_sigma);
  }
  if (o.contains("order")) {
    s.order = (int)o.at("order").as_int(s.order);
    if (s.order < 1) { s.order = 1; }
  }
  if (o.contains("solver_type")) {
    s.solver_bh2 = std::string(o.at("solver_type").as_string("bh2")) != "bh1";
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
  _uni_m.clear();
  _uni_sigma.clear();
  _uni_last_sample.clear();
  _uni_have_last = false;
  _uni_order = 0;
}

// One UniPC update, predictor or corrector -- they differ only in which
// sigma pair they span and whether the newest x0 difference joins the sum,
// so the reference's two near-identical routines are one here.
//
// Under flow sigmas alpha = 1 - sigma and sigma_t = sigma, so
// lambda = log(alpha) - log(sigma) and h = lambda_to - lambda_from. With
// predict_x0 the update is
//     x_t = (sigma_to/sigma_from)*x - alpha_to*expm1(hh)*m0
//           - alpha_to*B(h)*sum_k rho_k * D1_k
// where hh = -h, B(h) = expm1(hh) for bh2 (h for bh1), and the rho are the
// solution of R rho = b with R_ij = rks_j^(i-1). The reference special-cases
// the two orders this solver ever reaches (predictor order 2 -> rho = 0.5,
// corrector order 1 -> rho = 0.5); the general solve below reproduces those
// and keeps a higher order available.
void
FlowSampler::unipc_update_(std::vector<float>& out, const std::vector<float>& x,
                           const std::vector<float>& m0, double sigma_from,
                           double sigma_to, int order, bool corrector,
                           const std::vector<float>* d1_t) const
{
  const std::size_t n = x.size();
  out.assign(n, 0.0f);
  auto lam = [](double s) {
    return std::log(1.0 - s) - std::log(s);
  };
  const double alpha_to = 1.0 - sigma_to;
  const double h  = lam(sigma_to) - lam(sigma_from);
  const double hh = -h;                       // predict_x0
  const double h_phi_1 = std::expm1(hh);
  const double B_h = _sampler.solver_bh2 ? std::expm1(hh) : hh;

  // The multistep history's relative step ratios and x0 differences. The
  // newest entry (_uni_m.back()) IS m0, so the loop walks backwards from
  // the one before it; the corrector's history starts one step further
  // back because its interval already consumed the newest sigma.
  std::vector<double> rks;
  std::vector<const std::vector<float>*> d1s;
  const int have = (int)_uni_m.size();
  for (int i = 1; i < order; ++i) {
    // model_output_list[-(i+1)] in the reference -- the SAME expression for
    // predictor and corrector. What differs is only when the list was last
    // appended to: the corrector runs before this step's output joins it,
    // the predictor after. Each entry's own sigma is carried alongside it,
    // so no index arithmetic on the schedule is needed either way.
    const int idx = have - 1 - i;
    if (idx < 0) { break; }
    const double rk = (lam(_uni_sigma[(std::size_t)idx]) - lam(sigma_from)) / h;
    if (rk == 0.0) { break; }
    rks.push_back(rk);
    d1s.push_back(&_uni_m[(std::size_t)idx]);
  }
  rks.push_back(1.0);

  // R rho = b, with b_i = h_phi_k * i! / B_h stepped as the reference does.
  // The reference does NOT solve the system in its two commonest cases --
  // it substitutes 0.5 outright (predictor at order 2, corrector at order
  // 1). Those are not what the general solve returns, so reproducing them
  // is not an optimization: solving instead is simply a different sampler.
  const int K = corrector ? (int)rks.size() : (int)rks.size() - 1;
  std::vector<double> rho;
  const bool hardcoded_half =
      (corrector && order == 1) || (!corrector && order == 2);
  if (hardcoded_half) {
    rho.assign(1, 0.5);
  } else if (K > 0) {
    std::vector<double> R((std::size_t)K * K), b((std::size_t)K);
    double h_phi_k = h_phi_1 / hh - 1.0;
    double factorial_i = 1.0;
    for (int i = 1; i <= K; ++i) {
      for (int j = 0; j < K; ++j) {
        R[(std::size_t)(i - 1) * K + j] = std::pow(rks[(std::size_t)j], i - 1);
      }
      b[(std::size_t)(i - 1)] = h_phi_k * factorial_i / B_h;
      factorial_i *= (double)(i + 1);
      h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
    }
    // Gaussian elimination with partial pivoting; K is 1 or 2 in practice.
    rho.assign((std::size_t)K, 0.0);
    for (int c = 0; c < K; ++c) {
      int piv = c;
      for (int r = c + 1; r < K; ++r) {
        if (std::fabs(R[(std::size_t)r * K + c]) >
            std::fabs(R[(std::size_t)piv * K + c])) {
          piv = r;
        }
      }
      if (piv != c) {
        for (int j = 0; j < K; ++j) {
          std::swap(R[(std::size_t)c * K + j], R[(std::size_t)piv * K + j]);
        }
        std::swap(b[(std::size_t)c], b[(std::size_t)piv]);
      }
      const double d = R[(std::size_t)c * K + c];
      if (d == 0.0) { continue; }
      for (int r = c + 1; r < K; ++r) {
        const double f = R[(std::size_t)r * K + c] / d;
        for (int j = c; j < K; ++j) {
          R[(std::size_t)r * K + j] -= f * R[(std::size_t)c * K + j];
        }
        b[(std::size_t)r] -= f * b[(std::size_t)c];
      }
    }
    for (int r = K - 1; r >= 0; --r) {
      double acc = b[(std::size_t)r];
      for (int j = r + 1; j < K; ++j) {
        acc -= R[(std::size_t)r * K + j] * rho[(std::size_t)j];
      }
      const double d = R[(std::size_t)r * K + r];
      rho[(std::size_t)r] = (d != 0.0) ? acc / d : 0.0;
    }
  }

  const double sr = sigma_to / sigma_from;
  for (std::size_t k = 0; k < n; ++k) {
    double res = 0.0;
    for (std::size_t j = 0; j < d1s.size(); ++j) {
      // D1_j = (m_j - m0) / rk_j, as the reference forms it.
      const double d1 =
          ((double)(*d1s[j])[k] - (double)m0[k]) / rks[j];
      res += rho[j] * d1;
    }
    if (corrector && d1_t != nullptr && !rho.empty()) {
      res += rho.back() * ((double)(*d1_t)[k] - (double)m0[k]);
    }
    out[k] = (float)(sr * (double)x[k] - alpha_to * h_phi_1 * (double)m0[k]
                     - alpha_to * B_h * res);
  }
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

  if (m == "unipc") {
    // One model evaluation per step, used twice: first to CORRECT where the
    // previous step actually landed, then to predict the next point. That
    // is what buys second-order accuracy without a second DiT pass.
    std::vector<float> m0 = denoised_(x, si, denoise);
    if (m0.size() != x.size()) { return; }

    const int solver_order = _sampler.order < 1 ? 1 : _sampler.order;
    // Corrector: re-derive step i-1's landing point now that the model has
    // been evaluated AT it. Skipped on the first step, where there is no
    // previous interval to correct.
    if (i > 0 && _uni_have_last && !_uni_m.empty()) {
      std::vector<float> corrected;
      unipc_update_(corrected, _uni_last_sample, _uni_m.back(),
                    _sigmas[(std::size_t)i - 1], si, _uni_order,
                    /*corrector=*/true, &m0);
      if (corrected.size() == x.size()) {
        x = std::move(corrected);
        // The corrected sample changes what the model output MEANS at this
        // sigma only through x0 = x - sigma*v, and the reference reuses the
        // uncorrected conversion here too -- so m0 stands.
      }
    }

    _uni_m.push_back(m0);
    _uni_sigma.push_back(si);
    if ((int)_uni_m.size() > solver_order + 1) {
      _uni_m.erase(_uni_m.begin());
      _uni_sigma.erase(_uni_sigma.begin());
    }

    // lower_order_final + the multistep warmup: the order can be no higher
    // than the history is deep, and it winds back down as the schedule runs
    // out of steps to look ahead to.
    const int remaining = (int)_sigmas.size() - 1 - i;
    int this_order = std::min(solver_order, remaining);
    this_order = std::min(this_order, _uni_order + 1);
    if (this_order < 1) { this_order = 1; }
    _uni_order = this_order;

    _uni_last_sample = x;
    _uni_have_last = true;

    if (sn <= 0.0) {
      // The terminal sigma is 0, where the update reduces to landing on the
      // x0 prediction (sigma_to/sigma_from = 0 and alpha_to = 1).
      x = _uni_m.back();
      return;
    }
    std::vector<float> next;
    unipc_update_(next, x, _uni_m.back(), si, sn, this_order,
                  /*corrector=*/false, nullptr);
    if (next.size() == x.size()) { x = std::move(next); }
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
