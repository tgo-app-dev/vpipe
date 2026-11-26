#include "generative-models/magi2/magi2-accel.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/dit-flops.h"
#include "generative-models/shared/accel-settings.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/tp-cut.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace vpipe {
namespace genai {
namespace magi2 {

using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

bool
env_on_(const char* name)
{
  const char* e = std::getenv(name);
  return e != nullptr && e[0] != '\0' && e[0] != '0';
}

}  // namespace

// ---- the int8 tiers --------------------------------------------------------

int
asked_i8_tier(const FlexData* accel)
{
  if (!accel::flag(accel, accel::kI8Gemm, false)) { return -1; }
  const std::string t = accel::text(accel, accel::kI8GemmTier);
  return t == accel::kI8TierTokenPlain ? 3
         : t == accel::kI8TierToken    ? 2
         : t == accel::kI8TierFast     ? 1
                                       : 0;
}

int
effective_i8_tier(int asked, bool matrix_cores)
{
  bool on = asked >= 0;
  if (const char* e = std::getenv("VPIPE_I8_GEMM")) { on = std::atoi(e) != 0; }
  if (!on || !matrix_cores) { return -1; }
  int t = std::clamp(asked, 0, 3);
  if (const char* e = std::getenv("VPIPE_I8_TIER")) {
    if (std::strcmp(e, "token_plain") == 0 || std::strcmp(e, "3") == 0) {
      t = 3;
    } else if (std::strcmp(e, "token") == 0 || std::strcmp(e, "2") == 0) {
      t = 2;
    } else if (std::strcmp(e, "fast") == 0 || std::strcmp(e, "1") == 0) {
      t = 1;
    } else if (std::strcmp(e, "accurate") == 0 || std::strcmp(e, "0") == 0) {
      t = 0;
    }
  }
  if (t < 2) { return t; }
  bool rot = t == 2;
  if (const char* e = std::getenv("VPIPE_I8_TOKEN_ROT")) {
    rot = std::atoi(e) != 0;
  }
  return rot ? 2 : 3;
}

const char*
tier_name(int tier)
{
  switch (tier) {
    case 0: return "accurate";
    case 1: return "fast";
    case 2: return "token";
    case 3: return "token_plain";
    default: return "off";
  }
}

int
held_group(int K, int tier)
{
  if (K <= 0) { return 0; }
  if (tier == 1) {
    int target = 512;
    if (const char* e = std::getenv("VPIPE_I8_FAST_GROUP")) {
      target = std::atoi(e);
    }
    for (const int g : {1024, 512, 256}) {
      if (g <= target && K % g == 0) { return g; }
    }
    return 0;
  }
  if (tier == 2 || tier == 3) {
    return K % (tier == 2 ? 256 : 64) == 0 && K <= 16384
               ? I8GemmContext::kTokenGroup
               : 0;
  }
  return 0;
}

std::uint64_t
proj_bytes(int N, int K, int tier)
{
  const std::uint64_t nk = (std::uint64_t)N * (std::uint64_t)K;
  const int gc = held_group(K, tier);
  return gc > 0 ? nk + I8GemmContext::fast_scale_bytes(N, K, gc) : nk * 2u;
}

// ---- SageAttention ---------------------------------------------------------

bool
sage_runs(const sage::Config& cfg, bool matrix_cores)
{
  return cfg.enabled && matrix_cores &&
         std::getenv("VPIPE_NO_SAGE_ATTN") == nullptr &&
         !env_on_("VPIPE_MAGI2_NO_NAX_ATTN");
}

bool
sage_pv8_at(const sage::Config& cfg, int step)
{
  if (!cfg.pv8) { return false; }
  return step < 0 || step >= std::max(0, cfg.pv8_from_step);
}

// ---- the GEMM routes -------------------------------------------------------

Gemms::Gemms() = default;
Gemms::~Gemms() = default;

bool
Gemms::load(MetalCompute* mc, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  _mc = mc;
  _lib_gemm = mc->load_library("dense_gemm_bf16");
  _fn_dgemm = _lib_gemm.function("dense_gemm_t_bm64_f16");
  _fn_dgemm_bn64 = _lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  if (!_fn_dgemm.valid() || !_fn_dgemm_bn64.valid()) {
    return fail("the steel GEMM kernels did not load");
  }
  // THE M5 ROUTE. It takes over only when every entry it needs loads.
  if (mc->supports_matrix_cores() && !env_on_("VPIPE_MAGI2_NO_MMA2")) {
    _lib_mma = mc->load_library("dense_gemm_mma_bf16");
    _fn_mma_n128 = _lib_mma.function("dense_gemm_mma_t_n128_f16");
    _fn_mma_deep = _lib_mma.function("dense_gemm_mma_t_n128x256_f16");
    _fn_mma_tn2 = _lib_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");
    _mma = _fn_mma_n128.valid() && _fn_mma_deep.valid();
  }
  return true;
}

bool
Gemms::set_i8(int tier, bool autotune)
{
  _i8.reset();
  _i8_tier = -1;
  if (tier < 0 || _mc == nullptr) { return false; }
  auto i8 = std::make_unique<I8GemmContext>(_mc, true, /*bf16=*/true);
  i8->set_tier(tier >= 2   ? I8GemmContext::Tier::kToken
               : tier == 1 ? I8GemmContext::Tier::kFast
                           : I8GemmContext::Tier::kAccurate);
  i8->set_token_rotation(tier == 2);
  i8->set_autotune(autotune);
  if (!i8->enabled()) { return false; }
  _i8 = std::move(i8);
  _i8_tier = _i8->tier() == I8GemmContext::Tier::kAccurate ? 0
             : _i8->tier() == I8GemmContext::Tier::kFast   ? 1
             : _i8->token_rotated()                        ? 2
                                                           : 3;
  // A held layout takes its tier's GEMM at any row count -- the audio and
  // text spans are a few hundred rows -- rather than being widened back.
  if (_i8_tier >= 1) { _i8->set_min_rows(1); }
  return true;
}

bool
Gemms::load_stacked(WeightSet& ws, const std::string& name, int m, int N,
                    int K, const std::string& key, std::vector<Proj>* out,
                    std::string* err)
{
  auto fail = [&](const std::string& msg) {
    if (err != nullptr) { *err = fmt("{}: {}", name, msg)(); }
    return false;
  };
  out->clear();
  const auto* info = ws.src().info(name);
  if (info == nullptr) { return fail("missing"); }
  if (info->dtype != "BF16") { return fail("not bf16"); }
  const std::size_t nk = (std::size_t)N * (std::size_t)K;
  if ((std::size_t)info->nbytes != (std::size_t)m * nk * 2) {
    return fail(fmt("{} bytes where [{} x {}, {}] bf16 takes {}",
                    info->nbytes, m, N, K, (std::size_t)m * nk * 2)());
  }
  const int gc = _i8 && _i8_tier >= 1 ? _i8->fast_group(K) : 0;
  if (gc == 0) {
    SharedBuffer w = ws.tensor(name, _mc, WeightSet::Residency::Copied);
    if (w.empty()) { return fail("did not read"); }
    for (int i = 0; i < m; ++i) {
      Proj p;
      p.n = N;
      p.k = K;
      p.w = w.subview((std::size_t)i * nk * 2, nk * 2);
      out->push_back(std::move(p));
    }
    return true;
  }
  // THE HELD LAYOUT, rows of the whole stack at once: its scales are [rows,
  // K / gc] (or [rows] at the token group), so modality i's slab is rows
  // [i N, (i + 1) N) of both.
  const std::size_t sb = I8GemmContext::fast_scale_bytes(N, K, gc);
  const std::size_t codes = (std::size_t)m * nk;
  MetalCompute* mc = _mc;
  I8GemmContext* i8 = _i8.get();
  SharedBuffer both = ws.derived(
      key + "bf16/" + _i8->layout_tag(gc) + "/" + name,
      [&]() -> SharedBuffer {
        // The bf16 weight, read uncached and consumed here.
        SharedBuffer f = ws.read(name, mc, WeightSet::Residency::Copied);
        SharedBuffer o = mc->make_shared_buffer(codes + (std::size_t)m * sb);
        if (f.empty() || o.empty()) { return {}; }
        metal_compute::CommandStream st = mc->make_command_stream();
        bool ok = false;
        {
          ComputeEncoder e = st.begin_compute();
          ok = i8->encode_quant_fast(e, f, o.subview(0, codes),
                                     o.subview(codes, (std::size_t)m * sb),
                                     gc, m * N, K);
        }
        return ok && st.commit().wait_ok(nullptr) ? std::move(o)
                                                  : SharedBuffer{};
      });
  if (both.empty()) { return fail("the int8 layout did not build"); }
  for (int i = 0; i < m; ++i) {
    Proj p;
    p.n = N;
    p.k = K;
    p.gc = gc;
    p.codes = both.subview((std::size_t)i * nk, nk);
    p.scales = both.subview(codes + (std::size_t)i * sb, sb);
    out->push_back(std::move(p));
  }
  return true;
}

bool
Gemms::load_cut(WeightSet& ws, const std::string& name, int m, int N, int K,
                const std::vector<std::pair<int, int>>& rows, int c0, int c1,
                std::vector<Proj>* out, std::string* err)
{
  auto fail = [&](const std::string& msg) {
    if (err != nullptr) { *err = fmt("{}: {}", name, msg)(); }
    return false;
  };
  out->clear();
  const auto* info = ws.src().info(name);
  if (info == nullptr) { return fail("missing"); }
  if (info->dtype != "BF16") { return fail("not bf16"); }
  if ((std::size_t)info->nbytes != (std::size_t)m * N * K * 2) {
    return fail(fmt("{} bytes where [{} x {}, {}] bf16 takes {}",
                    info->nbytes, m, N, K, (std::size_t)m * N * K * 2)());
  }
  if (c1 <= 0) { c1 = K; }
  int nl = 0;
  for (const auto& [a, b] : rows) {
    if (a < 0 || b > N || a >= b) { return fail("a share outside its rows"); }
    nl += b - a;
  }
  if (nl <= 0 || c0 < 0 || c1 > K || c0 >= c1) {
    return fail("an empty share");
  }
  const int kl = c1 - c0;
  // Every slab's rows, in order, as rows of the stacked tensor.
  std::vector<std::pair<int, int>> all;
  for (int i = 0; i < m; ++i) {
    for (const auto& [a, b] : rows) { all.emplace_back(i * N + a, i * N + b); }
  }
  SharedBuffer cut;
  if (c0 == 0 && c1 == K) {
    cut = tp_read_rows(_mc, ws.src(), name, "BF16", m * N, all);
  }
  if (cut.empty()) {
    SharedBuffer f = ws.read(name, _mc, WeightSet::Residency::Copied);
    if (f.empty()) { return fail("did not read"); }
    cut = tp_cut(_mc, f, m * N, all, (std::size_t)c0 * 2,
                 (std::size_t)c1 * 2);
  }
  if (cut.empty()) { return fail("the share did not cut"); }
  const std::size_t nk = (std::size_t)nl * (std::size_t)kl;
  const int gc = _i8 && _i8_tier >= 1 ? _i8->fast_group(kl) : 0;
  if (gc == 0) {
    for (int i = 0; i < m; ++i) {
      Proj p;
      p.n = nl;
      p.k = kl;
      p.w = cut.subview((std::size_t)i * nk * 2, nk * 2);
      out->push_back(std::move(p));
    }
    return true;
  }
  // The share's layout, converted from the cut (consumed here).
  const std::size_t sb = I8GemmContext::fast_scale_bytes(nl, kl, gc);
  const std::size_t codes = (std::size_t)m * nk;
  SharedBuffer o = _mc->make_shared_buffer(codes + (std::size_t)m * sb);
  if (o.empty()) { return fail("no room for the share's int8 layout"); }
  metal_compute::CommandStream st = _mc->make_command_stream();
  bool ok = false;
  {
    ComputeEncoder e = st.begin_compute();
    ok = _i8->encode_quant_fast(e, cut, o.subview(0, codes),
                                o.subview(codes, (std::size_t)m * sb), gc,
                                m * nl, kl);
  }
  if (!ok || !st.commit().wait_ok(nullptr)) {
    return fail("the share's int8 layout did not build");
  }
  for (int i = 0; i < m; ++i) {
    Proj p;
    p.n = nl;
    p.k = kl;
    p.gc = gc;
    p.codes = o.subview((std::size_t)i * nk, nk);
    p.scales = o.subview(codes + (std::size_t)i * sb, sb);
    out->push_back(std::move(p));
  }
  return true;
}

int
Gemms::band_rows(int N, int K)
{
  return std::max(128, (int)(((std::size_t)1 << 30) /
                             (std::size_t)std::max(1, std::max(N, K))) /
                           128 * 128);
}

void
Gemms::dense(ComputeEncoder& enc, const SharedBuffer& x, std::size_t x_off,
             const SharedBuffer& w, std::size_t w_off, const SharedBuffer& y,
             std::size_t y_off, int M, int N, int K) const
{
  if (M <= 0 || N <= 0) { return; }
  if (_flops != nullptr) { _flops->gemm(M, N, K); }
  dense_(enc, x, x_off, w, w_off, y, y_off, M, N, K);
}

void
Gemms::dense_(ComputeEncoder& enc, const SharedBuffer& x, std::size_t x_off,
              const SharedBuffer& w, std::size_t w_off, const SharedBuffer& y,
              std::size_t y_off, int M, int N, int K) const
{
  if (M <= 0 || N <= 0) { return; }
  const int band = band_rows(N, K);
  for (int r0 = 0; r0 < M; r0 += band) {
    const int rn = std::min(band, M - r0);
    const std::size_t xo = x_off + (std::size_t)r0 * K * 2;
    const std::size_t yo = y_off + (std::size_t)r0 * N * 2;
    if (_mma && rn >= 64) {
      int RN = 256;
      const ComputeFunction* fn = &_fn_mma_deep;
      if (K < 6144) {
        fn = &_fn_mma_n128;
        RN = 128;
      } else if (K < 12288 && _fn_mma_tn2.valid()) {
        fn = &_fn_mma_tn2;
        RN = 512;
      }
      enc.set_function(*fn);
      enc.set_buffer(0, x, xo);
      enc.set_buffer(1, w, w_off);
      enc.set_buffer(2, w, w_off);
      enc.set_buffer(3, y, yo);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, rn);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                    (unsigned)((rn + 127) / 128), 1},
                   {256, 1, 1});
      continue;
    }
    const bool bn64 = N >= 512;
    enc.set_function(bn64 ? _fn_dgemm_bn64 : _fn_dgemm);
    enc.set_buffer(0, x, xo);
    enc.set_buffer(1, w, w_off);
    enc.set_buffer(2, w, w_off);
    enc.set_buffer(3, y, yo);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, rn);
    enc.set_constant(7, 0);
    const unsigned gx =
        (unsigned)(((N + (bn64 ? 63 : 31)) / (bn64 ? 64 : 32)) * 32);
    enc.dispatch({gx, (unsigned)(((rn + 63) / 64) * 2), 2}, {32, 2, 2});
  }
}

void
Gemms::proj(ComputeEncoder& enc, const SharedBuffer& x, std::size_t x_off,
            const Proj& p, const SharedBuffer& y, std::size_t y_off,
            int M) const
{
  if (M <= 0) { return; }
  // Counted once, as asked: every route below does this arithmetic.
  if (_flops != nullptr) { _flops->gemm(M, p.n, p.k); }
  if (!_i8) {
    ++_n_dense;
    dense_(enc, x, x_off, p.w, 0, y, y_off, M, p.n, p.k);
    return;
  }
  const SharedBuffer none;
  const int band = band_rows(p.n, p.k);
  for (int r0 = 0; r0 < M; r0 += band) {
    const int rn = std::min(band, M - r0);
    const std::size_t xo = x_off + (std::size_t)r0 * p.k * 2;
    const std::size_t yo = y_off + (std::size_t)r0 * p.n * 2;
    if (p.held()) {
      if (_i8->gemm_fast(enc, x, xo / 2, p.codes, p.scales, p.gc, none, y,
                         yo / 2, rn, p.n, p.k)) {
        ++_n_i8;
        continue;
      }
      // Declined (the tier's floor): widened back to bf16.
      const std::size_t need = (std::size_t)p.n * p.k * 2;
      if (_wdeq.byte_size() < need) { _wdeq = _mc->make_shared_buffer(need); }
      if (_wdeq.empty() ||
          !_i8->encode_dequant_fast(enc, p.codes, p.scales, p.gc, _wdeq, p.n,
                                    p.k)) {
        return;
      }
      ++_n_dense;
      dense_(enc, x, xo, _wdeq, 0, y, yo, rn, p.n, p.k);
      continue;
    }
    // The bf16 weight: the int8 requant route where the context takes the
    // shape (K >= 1024 and the rows, on any tier -- a depth a held layout
    // cannot take stays accurate), else dense.
    if (_i8->gemm(enc, x, xo / 2, p.w, y, yo / 2, rn, p.n, p.k)) {
      ++_n_i8;
      continue;
    }
    ++_n_dense;
    dense_(enc, x, xo, p.w, 0, y, yo, rn, p.n, p.k);
  }
}

void
Gemms::tune_pending()
{
  if (_i8) { _i8->tune_pending(_mc); }
}

void
Gemms::release_scratch()
{
  if (_i8) { _i8->release_scratch(); }
  _wdeq = SharedBuffer{};
}

namespace {

// What the int8 context holds for a forward's projections at `tier`: each
// of its buffers at the largest shape that reaches it (reserve()).
struct I8Plan {
  std::uint64_t xq = 0, as = 0, wq = 0, ws = 0;   // its four buffers
  std::uint64_t nw = 0, mw = 0;                   // the accurate GEMMs'
  bool deep = false;
  // The (rows, k) and (n, k) reserve() is asked for, per buffer.
  std::vector<std::pair<int, int>> act, w;
};

I8Plan
i8_plan_(int tier, int M, const std::vector<Gemms::Shape>& projs)
{
  I8Plan p;
  if (tier < 0 || M <= 0) { return p; }
  for (const Gemms::Shape& s : projs) {
    const bool held = held_group(s.k, tier) != 0;
    const bool acc = !held && s.k >= 1024;
    if (!held && !acc) { continue; }          // the dense GEMM
    const int m = s.rows > 0 ? std::min(s.rows, M) : M;
    const int rows = std::min(m, Gemms::band_rows(s.n, s.k));
    p.xq = std::max<std::uint64_t>(
        p.xq, I8GemmContext::act_codes_bytes(rows, s.k));
    p.as = std::max<std::uint64_t>(
        p.as, I8GemmContext::act_scale_bytes(rows, s.k));
    p.act.emplace_back(rows, s.k);
    if (!acc) { continue; }
    p.wq = std::max<std::uint64_t>(
        p.wq, I8GemmContext::act_codes_bytes(s.n, s.k));
    p.ws = std::max<std::uint64_t>(
        p.ws, I8GemmContext::act_scale_bytes(s.n, s.k));
    p.w.emplace_back(s.n, s.k);
    p.nw = std::max<std::uint64_t>(p.nw, (std::uint64_t)s.n);
    p.mw = std::max<std::uint64_t>(p.mw, (std::uint64_t)rows);
    p.deep = p.deep || s.k >= 9216;
  }
  return p;
}

// A buffer's bytes as the device holds it: whole 16 KB pages.
std::uint64_t
paged_(std::uint64_t b)
{
  return (b + 16383) / 16384 * 16384;
}

}  // namespace

bool
Gemms::reserve(int M, const std::vector<Shape>& projs)
{
  if (!_i8) { return true; }
  const I8Plan p = i8_plan_(_i8_tier, M, projs);
  bool ok = true;
  for (const auto& a : p.act) {
    ok = ok && _i8->reserve(a.first, a.second, 0, 0);
  }
  for (const auto& w : p.w) {
    ok = ok && _i8->reserve(0, 0, w.first, w.second);
  }
  return ok;
}

std::uint64_t
Gemms::i8_scratch_bytes(int tier, bool autotune, int M,
                        const std::vector<Shape>& projs)
{
  const I8Plan p = i8_plan_(tier, M, projs);
  std::uint64_t b = paged_(p.xq) + paged_(p.as) + paged_(p.wq) +
                    paged_(p.ws);
  if (p.wq == 0 || !autotune) { return b; }
  // Where the accurate route runs and its tuner with it: the tuner's
  // output rows and the split-K planes.
  b += paged_(2048u * p.nw * 2u);       // the tuner's rows (its kTuneRows)
  const std::uint64_t budget = (std::uint64_t)512 << 20;
  const std::uint64_t m64 = (p.mw + 63) / 64 * 64;
  const std::uint64_t two = 2u * m64 * p.nw * 4u;
  b += p.deep ? budget : std::min(two, budget);    // the split planes
  return b;
}

}  // namespace magi2
}  // namespace genai
}  // namespace vpipe
