#include "generative-models/train/optimizer.h"

#include "apple-silicon/metal-compute/command-stream.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>

namespace vpipe {
namespace genai {
namespace train {

using metal_compute::SharedBuffer;

namespace {

// Adafactor's second-moment decay: beta2 at step t is 1 - t^kAfDecay.
constexpr double kAfDecay = -0.8;

namespace k {
constexpr const char* kKind = "kind";
constexpr const char* kKindValue = "optimizer";
constexpr const char* kOptimizer = "optimizer";
constexpr const char* kLr = "learning_rate";
constexpr const char* kBeta1 = "beta1";
constexpr const char* kBeta2 = "beta2";
constexpr const char* kEps = "eps";
constexpr const char* kWeightDecay = "weight_decay";
constexpr const char* kSchedule = "lr_schedule";
constexpr const char* kWarmup = "warmup_steps";
constexpr const char* kMinLr = "min_lr_ratio";
constexpr const char* kRestarts = "restarts";
constexpr const char* kGradClip = "grad_clip";
constexpr const char* kEma = "ema";
constexpr const char* kDCoef = "prodigy_d_coef";
constexpr const char* kD0 = "prodigy_d0";
constexpr const char* kGrowth = "prodigy_growth";
constexpr const char* kBiasCorr = "bias_correction";
constexpr const char* kLoraPlus = "loraplus_lr_ratio";
}  // namespace k

bool
in_list_(const std::string& v, const char* list)
{
  std::string l = list;
  std::size_t p = 0;
  while (p <= l.size()) {
    const std::size_t c = l.find(',', p);
    const std::string item =
        l.substr(p, c == std::string::npos ? std::string::npos : c - p);
    if (item == v) { return true; }
    if (c == std::string::npos) { break; }
    p = c + 1;
  }
  return false;
}

SharedBuffer
zeros_(metal_compute::MetalCompute* mc, std::size_t bytes)
{
  SharedBuffer b = mc->make_shared_buffer(std::max<std::size_t>(bytes, 4));
  if (!b.empty()) { std::memset(b.contents(), 0, b.byte_size()); }
  return b;
}

}  // namespace

double
OptimizerSpec::effective_lr() const noexcept
{
  if (lr > 0.0) { return lr; }
  if (optimizer == "prodigy") { return 1.0; }
  if (optimizer == "lion") { return 3e-5; }
  if (optimizer == "sgd") { return 1e-2; }
  return 1e-4;
}

int
OptimizerSpec::effective_warmup(int total) const noexcept
{
  if (warmup_steps >= 0) { return std::min(warmup_steps, std::max(total, 0)); }
  return std::max(0, (int)std::lround(0.02 * total));
}

FlexData
OptimizerSpec::to_flex() const
{
  FlexData fd = FlexData::make_object();
  auto o = fd.as_object();
  o.insert_or_assign(k::kKind, FlexData::make_string(k::kKindValue));
  o.insert_or_assign(k::kOptimizer, FlexData::make_string(optimizer));
  o.insert_or_assign(k::kLr, FlexData::make_real(lr));
  o.insert_or_assign(k::kBeta1, FlexData::make_real(beta1));
  o.insert_or_assign(k::kBeta2, FlexData::make_real(beta2));
  o.insert_or_assign(k::kEps, FlexData::make_real(eps));
  o.insert_or_assign(k::kWeightDecay, FlexData::make_real(weight_decay));
  o.insert_or_assign(k::kSchedule, FlexData::make_string(lr_schedule));
  o.insert_or_assign(k::kWarmup, FlexData::make_int(warmup_steps));
  o.insert_or_assign(k::kMinLr, FlexData::make_real(min_lr_ratio));
  o.insert_or_assign(k::kRestarts, FlexData::make_int(restarts));
  o.insert_or_assign(k::kGradClip, FlexData::make_real(grad_clip));
  o.insert_or_assign(k::kEma, FlexData::make_real(ema));
  o.insert_or_assign(k::kDCoef, FlexData::make_real(prodigy_d_coef));
  o.insert_or_assign(k::kD0, FlexData::make_real(prodigy_d0));
  o.insert_or_assign(k::kGrowth, FlexData::make_real(prodigy_growth));
  o.insert_or_assign(k::kBiasCorr, FlexData::make_bool(bias_correction));
  o.insert_or_assign(k::kLoraPlus, FlexData::make_real(lora_plus_ratio));
  return fd;
}

bool
OptimizerSpec::is_optimizer_spec(const FlexData& f)
{
  if (!f.is_object()) { return false; }
  auto o = f.as_object();
  return o.contains(k::kKind) &&
         std::string(o.at(k::kKind).as_string("")) == k::kKindValue;
}

bool
OptimizerSpec::from_flex(const FlexData& f, OptimizerSpec* out,
                         std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (!f.is_object()) { return fail("an optimizer spec is an object"); }
  OptimizerSpec s;
  auto o = f.as_object();
  auto real = [&](const char* key, double* v) {
    if (o.contains(key)) { *v = o.at(key).as_real(*v); }
  };
  if (o.contains(k::kOptimizer)) {
    s.optimizer = std::string(o.at(k::kOptimizer).as_string("adamw"));
  }
  if (!in_list_(s.optimizer, kOptimizers)) {
    return fail("unknown optimizer '" + s.optimizer + "' (" +
                std::string(kOptimizers) + ")");
  }
  real(k::kLr, &s.lr);
  real(k::kBeta1, &s.beta1);
  real(k::kBeta2, &s.beta2);
  real(k::kEps, &s.eps);
  real(k::kWeightDecay, &s.weight_decay);
  if (o.contains(k::kSchedule)) {
    s.lr_schedule = std::string(o.at(k::kSchedule).as_string("auto"));
  }
  if (!in_list_(s.lr_schedule, kSchedules)) {
    return fail("unknown lr_schedule '" + s.lr_schedule + "' (" +
                std::string(kSchedules) + ")");
  }
  if (o.contains(k::kWarmup)) {
    s.warmup_steps = (int)o.at(k::kWarmup).as_int(s.warmup_steps);
  }
  real(k::kMinLr, &s.min_lr_ratio);
  if (o.contains(k::kRestarts)) {
    s.restarts = (int)o.at(k::kRestarts).as_int(s.restarts);
  }
  real(k::kGradClip, &s.grad_clip);
  real(k::kEma, &s.ema);
  real(k::kDCoef, &s.prodigy_d_coef);
  real(k::kD0, &s.prodigy_d0);
  real(k::kGrowth, &s.prodigy_growth);
  if (o.contains(k::kBiasCorr)) {
    s.bias_correction = o.at(k::kBiasCorr).as_bool(s.bias_correction);
  }
  real(k::kLoraPlus, &s.lora_plus_ratio);
  if (s.lr < 0.0 || s.beta1 < 0.0 || s.beta1 >= 1.0 || s.beta2 < 0.0 ||
      s.beta2 >= 1.0 || s.eps <= 0.0 || s.weight_decay < 0.0 ||
      s.min_lr_ratio < 0.0 || s.min_lr_ratio > 1.0 || s.restarts < 1 ||
      s.grad_clip < 0.0 || s.ema < 0.0 || s.ema >= 1.0 ||
      s.prodigy_d0 <= 0.0 || s.prodigy_d_coef <= 0.0 ||
      s.prodigy_growth < 1.0 || s.lora_plus_ratio <= 0.0) {
    return fail("an optimizer value is out of range");
  }
  *out = s;
  return true;
}

std::size_t
Optimizer::state_bytes_per_param(const OptimizerSpec& s, int rank) noexcept
{
  std::size_t b = 0;
  if (s.optimizer == "adamw") { b = 8; }
  // Two code bytes, and the block scales (8 bytes a 256) rounded up.
  else if (s.optimizer == "adamw8bit") { b = 3; }
  // A row and a column vector a factor: (rows + cols) * 4 bytes over
  // rows * cols parameters, one of them the rank -- under 8 / rank bytes
  // a parameter (the plan rounds it up: 1 from rank 8).
  else if (s.optimizer == "adafactor") {
    const int r = std::max(rank, 1);
    b = (std::size_t)((8 + r - 1) / r);
  }
  else if (s.optimizer == "prodigy") { b = 16; }   // m, v, s, x0
  else { b = 4; }                                    // lion, sgd: m
  if (s.ema > 0.0) { b += 4; }
  return b;
}

bool
Optimizer::init(const TrainOps* ops, LoraParams* params,
                const OptimizerSpec& spec, int total_steps, std::string* err)
{
  _ops = ops;
  _p = params;
  _spec = spec;
  _total = std::max(total_steps, 1);
  _warmup = spec.effective_warmup(_total);
  _t = 0;
  _d = spec.prodigy_d0;
  _d_max = spec.prodigy_d0;
  _d_num = 0.0;
  if (ops == nullptr || !ops->valid() || params == nullptr) {
    if (err != nullptr) { *err = "optimizer: no ops or parameters"; }
    return false;
  }
  metal_compute::MetalCompute* mc = ops->mc();
  const std::size_t na = params->a_elems(), nb = params->b_elems();
  if (spec.optimizer == "adamw8bit") {
    // Codes of 0 and zero scales: both moments start at zero.
    const std::vector<float> codes = TrainOps::adamw8_codes();
    _codes = zeros_(mc, codes.size() * 4);
    if (!_codes.empty()) {
      std::memcpy(_codes.contents(), codes.data(), codes.size() * 4);
    }
    // The code of 0 is the same index in either map's table; find it.
    std::uint8_t zs = 0, zu = 0;
    for (int i = 0; i < 256; ++i) {
      if (codes[(std::size_t)i] == 0.0f) { zs = (std::uint8_t)i; }
      if (codes[(std::size_t)(256 + i)] == 0.0f) { zu = (std::uint8_t)i; }
    }
    auto fill = [&](std::size_t n, std::uint8_t v) {
      SharedBuffer b = zeros_(mc, n);
      if (!b.empty()) { std::memset(b.contents(), v, b.byte_size()); }
      return b;
    };
    const std::size_t B = TrainOps::kOpt8Block;
    _qm_a = fill(na, zs);
    _qm_b = fill(nb, zs);
    _qv_a = fill(na, zu);
    _qv_b = fill(nb, zu);
    _am_a = zeros_(mc, ((na + B - 1) / B) * 2 * 4);
    _am_b = zeros_(mc, ((nb + B - 1) / B) * 2 * 4);
    if (_codes.empty() || _qm_a.empty() || _qm_b.empty() || _qv_a.empty() ||
        _qv_b.empty() || _am_a.empty() || _am_b.empty()) {
      if (err != nullptr) { *err = "optimizer: state allocation failed"; }
      return false;
    }
  }
  if (spec.optimizer == "adafactor" && !build_adafactor_(err)) {
    return false;
  }
  // Lion's and SGD's momentum, AdamW's and Prodigy's first moment.
  if (spec.optimizer != "adamw8bit" && spec.optimizer != "adafactor") {
    _m_a = zeros_(mc, na * 4);
    _m_b = zeros_(mc, nb * 4);
  }
  if (spec.optimizer == "adamw" || spec.optimizer == "prodigy") {
    _v_a = zeros_(mc, na * 4);
    _v_b = zeros_(mc, nb * 4);
  }
  if (spec.optimizer == "prodigy") {
    _s_a = zeros_(mc, na * 4);
    _s_b = zeros_(mc, nb * 4);
    _x0_a = zeros_(mc, na * 4);
    _x0_b = zeros_(mc, nb * 4);
    _num = zeros_(mc, std::max(na, nb) * 4);
    std::memcpy(_x0_a.contents(), params->a_master().contents(), na * 4);
    std::memcpy(_x0_b.contents(), params->b_master().contents(), nb * 4);
  }
  if (spec.ema > 0.0) {
    _ema_a = zeros_(mc, na * 4);
    _ema_b = zeros_(mc, nb * 4);
    std::memcpy(_ema_a.contents(), params->a_master().contents(), na * 4);
    std::memcpy(_ema_b.contents(), params->b_master().contents(), nb * 4);
  }
  const int groups = TrainOps::reduce_groups(std::max(na, nb));
  _part = zeros_(mc, (std::size_t)groups * 2 * 4);
  if ((spec.optimizer != "adamw8bit" && spec.optimizer != "adafactor" &&
       (_m_a.empty() || _m_b.empty())) ||
      _part.empty()) {
    if (err != nullptr) { *err = "optimizer: state allocation failed"; }
    return false;
  }
  return true;
}

bool
Optimizer::build_adafactor_(std::string* err)
{
  metal_compute::MetalCompute* mc = _ops->mc();
  const int rank = _p->rank();
  // A: each module's [rank, k]. B: its [n, rank] rows -- where a fused
  // group's rows interleave (b_rows), the group's span as one matrix.
  std::vector<TrainOps::AfMat> ma, mb;
  for (const LoraParams::Module& m : _p->modules()) {
    TrainOps::AfMat a;
    a.off = (std::uint32_t)m.a_off;
    a.rows = (std::uint32_t)rank;
    a.cols = (std::uint32_t)m.k;
    ma.push_back(a);
  }
  std::vector<std::pair<std::size_t, std::size_t>> spans;   // [row0, row1)
  for (const LoraParams::Module& m : _p->modules()) {
    std::size_t r0 = m.b_off / (std::size_t)rank, r1 = r0 + (std::size_t)m.n;
    if (!m.b_rows.empty()) {
      r0 = (std::size_t)*std::min_element(m.b_rows.begin(), m.b_rows.end());
      r1 = (std::size_t)*std::max_element(m.b_rows.begin(), m.b_rows.end()) +
           1;
    }
    spans.push_back({r0, r1});
  }
  std::sort(spans.begin(), spans.end());
  std::vector<std::pair<std::size_t, std::size_t>> merged;
  for (const auto& sp : spans) {
    if (!merged.empty() && sp.first < merged.back().second) {
      merged.back().second = std::max(merged.back().second, sp.second);
    } else {
      merged.push_back(sp);
    }
  }
  std::size_t covered = 0;
  for (const auto& sp : merged) {
    TrainOps::AfMat b;
    b.off = (std::uint32_t)(sp.first * (std::size_t)rank);
    b.rows = (std::uint32_t)(sp.second - sp.first);
    b.cols = (std::uint32_t)rank;
    covered += sp.second - sp.first;
    mb.push_back(b);
  }
  if (covered * (std::size_t)rank != _p->b_elems()) {
    if (err != nullptr) {
      *err = "optimizer: adafactor could not tile the B factors' rows";
    }
    return false;
  }
  auto build = [&](std::vector<TrainOps::AfMat>& v, AfArena* A) {
    std::uint32_t cur = 0, rc = 0, cc = 0;
    for (TrainOps::AfMat& m : v) {
      m.row_off = cur;
      cur += m.rows;
      m.col_off = cur;
      cur += m.cols;
      m.rcum = rc;
      m.ccum = cc;
      rc += m.rows;
      cc += m.cols;
    }
    A->nmat = (int)v.size();
    A->rows = (int)rc;
    A->cols = (int)cc;
    A->mats = zeros_(mc, v.size() * sizeof(TrainOps::AfMat));
    A->state = zeros_(mc, (std::size_t)cur * 4);
    A->usq = zeros_(mc, v.size() * 4);
    if (A->mats.empty() || A->state.empty() || A->usq.empty()) {
      return false;
    }
    std::memcpy(A->mats.contents(), v.data(),
                v.size() * sizeof(TrainOps::AfMat));
    return true;
  };
  if (!build(ma, &_af_a) || !build(mb, &_af_b)) {
    if (err != nullptr) { *err = "optimizer: state allocation failed"; }
    return false;
  }
  return true;
}

double
Optimizer::lr_at(int t) const noexcept
{
  const double lr = _spec.effective_lr();
  if (t < _warmup) { return lr * (double)(t + 1) / (double)_warmup; }
  const double span = std::max(1, _total - _warmup);
  const double p = std::clamp((double)(t - _warmup) / span, 0.0, 1.0);
  const double lo = _spec.min_lr_ratio;
  const std::string& sch = _spec.lr_schedule;
  if (sch == "linear") { return lr * (1.0 - p * (1.0 - lo)); }
  if (sch == "cosine" || sch == "cosine_restarts") {
    double q = p;
    if (sch == "cosine_restarts" && _spec.restarts > 1) {
      q = std::fmod(p * _spec.restarts, 1.0);
    }
    return lr * (lo + (1.0 - lo) * 0.5 * (1.0 + std::cos(M_PI * q)));
  }
  return lr;
}

bool
Optimizer::step(int accum, double* grad_norm, double* lr_used,
                std::string* err)
{
  metal_compute::MetalCompute* mc = _ops->mc();
  auto run = [&](const std::function<void(metal_compute::ComputeEncoder&)>&
                     fn) {
    metal_compute::CommandStream st = mc->make_command_stream();
    metal_compute::ComputeEncoder enc = st.begin_compute();
    fn(enc);
    enc.end();
    std::string e;
    if (!st.commit().wait_ok(&e)) {
      if (err != nullptr) { *err = "optimizer: " + e; }
      return false;
    }
    return true;
  };
  const std::size_t na = _p->a_elems(), nb = _p->b_elems();
  const int ga = TrainOps::reduce_groups(na);
  const int gb = TrainOps::reduce_groups(nb);

  // The global norm of the (summed) gradient, folded in index order.
  if (!run([&](metal_compute::ComputeEncoder& enc) {
        _ops->sqnorm_partial(enc, _p->a_grad(), 0, na, _part, 0);
        _ops->sqnorm_partial(enc, _p->b_grad(), 0, nb, _part,
                             (std::size_t)ga);
      })) {
    return false;
  }
  double sq = 0.0;
  const auto* pp = static_cast<const float*>(_part.contents());
  for (int i = 0; i < ga + gb; ++i) { sq += (double)pp[i]; }
  const double inv_acc = 1.0 / (double)std::max(accum, 1);
  const double gnorm = std::sqrt(sq) * inv_acc;
  if (!std::isfinite(gnorm)) {
    if (err != nullptr) { *err = "optimizer: the gradient is not finite"; }
    return false;
  }
  double scale = inv_acc;
  if (_spec.grad_clip > 0.0 && gnorm > _spec.grad_clip) {
    scale *= _spec.grad_clip / gnorm;
  }
  const double lr = lr_at(_t);
  if (grad_norm != nullptr) { *grad_norm = gnorm; }
  if (lr_used != nullptr) { *lr_used = lr; }

  OptParams o;
  o.lr = (float)lr;
  o.beta1 = (float)_spec.beta1;
  o.beta2 = (float)_spec.beta2;
  o.eps = (float)_spec.eps;
  o.weight_decay = (float)_spec.weight_decay;
  o.grad_scale = (float)scale;
  const int t1 = _t + 1;
  const std::string& opt = _spec.optimizer;

  if (opt == "prodigy") {
    const double b3 = std::sqrt(_spec.beta2);
    double bc = 1.0;
    if (_spec.bias_correction) {
      bc = std::sqrt(1.0 - std::pow(_spec.beta2, t1)) /
           (1.0 - std::pow(_spec.beta1, t1));
    }
    const double dlr = _d * lr * bc;
    o.beta3 = (float)b3;
    o.d = (float)_d;
    o.d0 = (float)_spec.prodigy_d0;
    o.dlr = (float)dlr;
    double num = 0.0;
    double den = 0.0;
    // Pass 1, per arena: moments and s with the current d, and the
    // numerator's elements, folded on the host.
    for (int arena = 0; arena < 2; ++arena) {
      const bool isa = arena == 0;
      const std::size_t n = isa ? na : nb;
      o.n = (int)n;
      if (!run([&](metal_compute::ComputeEncoder& enc) {
            _ops->prodigy_accum(enc, isa ? _p->a_master() : _p->b_master(),
                                isa ? _p->a_grad() : _p->b_grad(),
                                isa ? _x0_a : _x0_b, isa ? _m_a : _m_b,
                                isa ? _v_a : _v_b, isa ? _s_a : _s_b, _num,
                                o);
            _ops->abs_partial(enc, isa ? _s_a : _s_b, 0, n, _part, 0);
          })) {
        return false;
      }
      const auto* nu = static_cast<const float*>(_num.contents());
      for (std::size_t i = 0; i < n; ++i) { num += (double)nu[i]; }
      const int g = TrainOps::reduce_groups(n);
      for (int i = 0; i < g; ++i) { den += (double)pp[i]; }
    }
    _d_num = b3 * _d_num + num;
    if (den > 0.0) {
      const double d_hat = _spec.prodigy_d_coef * _d_num / den;
      double d = _d;
      if (d == _spec.prodigy_d0) { d = std::max(d, d_hat); }
      _d_max = std::max(_d_max, d_hat);
      d = std::min(_d_max, d * _spec.prodigy_growth);
      _d = std::max(d, _spec.prodigy_d0);
    }
    o.d = (float)_d;   // the new d rides eps; the step keeps the old dlr
    if (!run([&](metal_compute::ComputeEncoder& enc) {
          o.n = (int)na;
          o.mul = _p->a_mul();
          _ops->prodigy_step(enc, _p->a_master(), _p->a_grad(), _m_a, _v_a,
                             _p->a_shadow(), o);
          o.n = (int)nb;
          o.mul = 1.0f;
          // LoRA+: B's step that many times A's.
          o.dlr = (float)(dlr * _spec.lora_plus_ratio);
          _ops->prodigy_step(enc, _p->b_master(), _p->b_grad(), _m_b, _v_b,
                             _p->b_shadow(), o);
          if (!_ema_a.empty()) {
            _ops->ema(enc, _p->a_master(), _ema_a, (float)_spec.ema, na);
            _ops->ema(enc, _p->b_master(), _ema_b, (float)_spec.ema, nb);
          }
        })) {
      return false;
    }
    ++_t;
    return true;
  }

  if (opt == "adafactor") {
    TrainOps::AfParams af;
    af.grad_scale = (float)scale;
    af.beta2t = (float)(1.0 - std::pow((double)t1, kAfDecay));
    af.eps1 = 1e-30f;
    af.weight_decay = (float)_spec.weight_decay;
    af.clip = 1.0f;
    if (!run([&](metal_compute::ComputeEncoder& enc) {
          for (int arena = 0; arena < 2; ++arena) {
            const bool isa = arena == 0;
            const AfArena& A = isa ? _af_a : _af_b;
            af.lr = (float)(isa ? lr : lr * _spec.lora_plus_ratio);
            af.mul = isa ? _p->a_mul() : 1.0f;
            af.nmat = A.nmat;
            const SharedBuffer& p = isa ? _p->a_master() : _p->b_master();
            const SharedBuffer& g = isa ? _p->a_grad() : _p->b_grad();
            af.total = A.rows;
            _ops->adafactor_stats(enc, g, A.state, A.mats, af, true);
            af.total = A.cols;
            _ops->adafactor_stats(enc, g, A.state, A.mats, af, false);
            _ops->adafactor_update(enc, p, g, A.state, A.mats, A.usq,
                                   isa ? _p->a_shadow() : _p->b_shadow(), af,
                                   0);
            _ops->adafactor_update(enc, p, g, A.state, A.mats, A.usq,
                                   isa ? _p->a_shadow() : _p->b_shadow(), af,
                                   1);
          }
          if (!_ema_a.empty()) {
            _ops->ema(enc, _p->a_master(), _ema_a, (float)_spec.ema, na);
            _ops->ema(enc, _p->b_master(), _ema_b, (float)_spec.ema, nb);
          }
        })) {
      return false;
    }
    ++_t;
    return true;
  }

  o.bc1 = (float)(1.0 - std::pow(_spec.beta1, t1));
  o.bc2 = (float)(1.0 - std::pow(_spec.beta2, t1));
  if (!run([&](metal_compute::ComputeEncoder& enc) {
        for (int arena = 0; arena < 2; ++arena) {
          const bool isa = arena == 0;
          o.n = (int)(isa ? na : nb);
          o.mul = isa ? _p->a_mul() : 1.0f;
          // LoRA+: B learns lora_plus_ratio times as fast as A.
          o.lr = (float)(isa ? lr : lr * _spec.lora_plus_ratio);
          const SharedBuffer& p = isa ? _p->a_master() : _p->b_master();
          const SharedBuffer& g = isa ? _p->a_grad() : _p->b_grad();
          const SharedBuffer& s = isa ? _p->a_shadow() : _p->b_shadow();
          const SharedBuffer& m = isa ? _m_a : _m_b;
          if (opt == "adamw8bit") {
            _ops->adamw8(enc, p, g, isa ? _qm_a : _qm_b, isa ? _qv_a : _qv_b,
                         isa ? _am_a : _am_b, s, _codes, o);
          } else if (opt == "adamw") {
            _ops->adamw(enc, p, g, m, isa ? _v_a : _v_b, s, o);
          } else if (opt == "lion") {
            _ops->lion(enc, p, g, m, s, o);
          } else {
            _ops->sgd(enc, p, g, m, s, o);
          }
        }
        if (!_ema_a.empty()) {
          _ops->ema(enc, _p->a_master(), _ema_a, (float)_spec.ema, na);
          _ops->ema(enc, _p->b_master(), _ema_b, (float)_spec.ema, nb);
        }
      })) {
    return false;
  }
  ++_t;
  return true;
}

// ---- state -----------------------------------------------------------

namespace {

constexpr std::uint32_t kStateMagic = 0x4f54504fu;   // "OPTO"
// 2: the 8-bit moments and Adafactor's row / column states after version
// 1's arenas (a version-1 file still loads, for every optimizer but those
// two).
constexpr std::uint32_t kStateVersion = 2;

}  // namespace

bool
Optimizer::save_state(const std::string& path, std::string* err) const
{
  const std::string tmp = path + ".tmp";
  std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
  if (!o) {
    if (err != nullptr) { *err = "cannot write " + tmp; }
    return false;
  }
  auto put = [&](const void* p, std::size_t n) {
    o.write(static_cast<const char*>(p), (std::streamsize)n);
  };
  const std::uint64_t na = _p->a_elems(), nb = _p->b_elems();
  put(&kStateMagic, 4);
  put(&kStateVersion, 4);
  put(&na, 8);
  put(&nb, 8);
  const std::int32_t t = _t;
  put(&t, 4);
  put(&_d, 8);
  put(&_d_max, 8);
  put(&_d_num, 8);
  auto arena = [&](const SharedBuffer& b, std::uint64_t n) {
    const std::uint8_t present = b.empty() ? 0 : 1;
    put(&present, 1);
    if (present != 0) { put(b.contents(), n * 4); }
  };
  arena(_p->a_master(), na);
  arena(_p->b_master(), nb);
  arena(_m_a, na); arena(_m_b, nb);
  arena(_v_a, na); arena(_v_b, nb);
  arena(_s_a, na); arena(_s_b, nb);
  arena(_x0_a, na); arena(_x0_b, nb);
  arena(_ema_a, na); arena(_ema_b, nb);
  // The 8-bit moments, by their byte sizes.
  auto raw = [&](const SharedBuffer& b) {
    const std::uint64_t n = b.empty() ? 0 : b.byte_size();
    put(&n, 8);
    if (n > 0) { put(b.contents(), n); }
  };
  raw(_qm_a); raw(_qm_b); raw(_qv_a); raw(_qv_b); raw(_am_a); raw(_am_b);
  raw(_af_a.state); raw(_af_b.state);
  o.close();
  if (!o) {
    if (err != nullptr) { *err = "short write to " + tmp; }
    return false;
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    if (err != nullptr) { *err = "rename: " + ec.message(); }
    return false;
  }
  return true;
}

bool
Optimizer::load_state(const std::string& path, std::string* err)
{
  std::ifstream in(path, std::ios::binary);
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  if (!in) { return fail("cannot open " + path); }
  auto get = [&](void* p, std::size_t n) {
    in.read(static_cast<char*>(p), (std::streamsize)n);
    return (bool)in;
  };
  std::uint32_t magic = 0, ver = 0;
  std::uint64_t na = 0, nb = 0;
  std::int32_t t = 0;
  if (!get(&magic, 4) || !get(&ver, 4) || magic != kStateMagic ||
      ver < 1 || ver > kStateVersion) {
    return fail("not an optimizer state file: " + path);
  }
  if (!get(&na, 8) || !get(&nb, 8) || na != _p->a_elems() ||
      nb != _p->b_elems()) {
    return fail("the optimizer state is for a different adapter shape");
  }
  if (!get(&t, 4) || !get(&_d, 8) || !get(&_d_max, 8) || !get(&_d_num, 8)) {
    return fail("truncated optimizer state");
  }
  auto arena = [&](const SharedBuffer& b, std::uint64_t n) {
    std::uint8_t present = 0;
    if (!get(&present, 1)) { return false; }
    if (present == 0) { return b.empty(); }
    if (b.empty()) { return false; }
    return get(b.contents(), n * 4);
  };
  if (!arena(_p->a_master(), na) || !arena(_p->b_master(), nb) ||
      !arena(_m_a, na) || !arena(_m_b, nb) || !arena(_v_a, na) ||
      !arena(_v_b, nb) || !arena(_s_a, na) || !arena(_s_b, nb) ||
      !arena(_x0_a, na) || !arena(_x0_b, nb) || !arena(_ema_a, na) ||
      !arena(_ema_b, nb)) {
    return fail("the optimizer state does not match this optimizer");
  }
  auto raw = [&](const SharedBuffer& b) {
    std::uint64_t n = 0;
    if (!get(&n, 8)) { return false; }
    const std::uint64_t want = b.empty() ? 0 : b.byte_size();
    if (n != want) { return false; }
    return n == 0 || get(b.contents(), n);
  };
  if (ver >= 2) {
    if (!raw(_qm_a) || !raw(_qm_b) || !raw(_qv_a) || !raw(_qv_b) ||
        !raw(_am_a) || !raw(_am_b) || !raw(_af_a.state) ||
        !raw(_af_b.state)) {
      return fail("the optimizer state does not match this optimizer");
    }
  } else if (!_qm_a.empty() || !_af_a.state.empty()) {
    return fail("the optimizer state predates 8-bit AdamW and Adafactor");
  }
  _t = t;
  _p->refresh_shadows_cpu();
  return true;
}

}  // namespace train
}  // namespace genai
}  // namespace vpipe
