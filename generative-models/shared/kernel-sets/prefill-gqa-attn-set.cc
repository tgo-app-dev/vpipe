#include "generative-models/shared/kernel-sets/prefill-gqa-attn-set.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace vpipe::genai {

namespace {
// n-regime lower bounds + the n each is probed at (around the steel/flash
// crossover ~2k). The long regime is probed at a modest n (the O(n^2) probe +
// its qt[Hq,n,D] buffer stay cheap; the steel>=flash verdict holds above it).
const int kRegimeLo[]  = {0, 1536, 3072, 6144};
const int kRegimeN[]   = {768, 2048, 3072, 6144};
// The VERY LONG regime exists for the NAX members: MLX's head-dim split pulls
// ahead of the plain kernel only with length -- MEASURED on the M5 Pro at
// Qwen3.5-4B, a 5.65 / 5.66 ms tie at 3072 but 1.28x vs 1.24x (8k) and 1.53x vs
// 1.45x (16k) end to end -- so a probe capped at 3072 picked the wrong one; at
// kernel level they separate by 4096. It is probed only when a NAX member
// exists, and only among the long-n members; elsewhere it inherits the long
// regime's choice unprobed and allocates nothing for it.
constexpr int kVeryLong = 3;
const char* kName[7] = {"scalar", "qtile", "flash", "mma", "steel", "nax",
                        "nax-split"};

// The steel attention parameter block (mlx steel/attn/params.h AttnParams),
// which the NAX kernels read at buffer 4.
struct NaxAttnParams {
  int B, H, D, qL, kL, gqa_factor;
  float scale;
  int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

constexpr int kNaxBQ = 64;
constexpr int kNaxBK = 32;
}  // namespace

bool
PrefillGqaAttnSet::load(metal_compute::ComputeLibrary& lib_sdpa,
                        metal_compute::ComputeLibrary* lib_attn,
                        metal_compute::ComputeLibrary* lib_mma, bool use_mma,
                        metal_compute::ComputeLibrary* lib_nax, bool bf16,
                        int head_dim) {
  if (head_dim != 256 && head_dim != 128) { return false; }
  _head_dim = head_dim;
  const bool d256 = head_dim == 256;
  _fn[kScalar] = lib_sdpa.function("sdpa_paged_causal_f16");
  _fn[kQtile] = lib_sdpa.function("sdpa_paged_qtile_f16");
  _fn[kFlash] = lib_sdpa.function("sdpa_paged_flash_f16");
  if (lib_mma != nullptr && lib_mma->valid()) {
    _fn[kMma] = lib_mma->function(d256 ? "sdpa_mma_f16" : "sdpa_mma_d128_f16");
  }
  if (lib_attn != nullptr && lib_attn->valid()) {
    _fn[kSteel] = lib_attn->function(d256 ? "attn_steel_paged_bd256"
                                          : "attn_steel_paged_bd128");
  }
  _have[kScalar] = _fn[kScalar].valid();
  _have[kQtile] = d256 && _fn[kQtile].valid();   // a 256-wide kernel only
  _have[kFlash] = _fn[kFlash].valid();
  _have[kMma] = use_mma && _fn[kMma].valid();
  _have[kSteel] = _fn[kSteel].valid();
  // The NAX members: matrix-core GPUs only (use_mma), and only when the pool
  // gather and both kernels are present. A probe pipeline per kernel proves
  // the entry points exist before the set offers them.
  _lib_nax = (lib_nax != nullptr && lib_nax->valid()) ? lib_nax : nullptr;
  _nax_bf16 = bf16;
  _fn_gather = lib_sdpa.function("kv_gather_paged_f16");
  if (use_mma && _lib_nax != nullptr && _fn_gather.valid()) {
    _have[kNax] = nax_fn(false, true, true).valid();
    _have[kNaxSplit] = nax_fn(true, true, true).valid();
  }
  return _have[kFlash] || _have[kQtile];        // need a usable non-scalar member
}

int
PrefillGqaAttnSet::regime_of(int n) const {
  int r = 0;
  for (int i = 1; i < kRegimes; ++i) {
    if (n >= kRegimeLo[i]) { r = i; }
  }
  return r;
}

int
PrefillGqaAttnSet::min_n() const {
  return _head_dim == 256 ? 0 : kRegimeLo[1];
}

const char*
PrefillGqaAttnSet::kernel_name(int n) const {
  return kName[_member[regime_of(n)]];
}

bool
PrefillGqaAttnSet::has(std::string_view member) const {
  for (int m = 0; m < kMembers; ++m) {
    if (member == kName[m]) { return _have[m]; }
  }
  return false;
}

const metal_compute::ComputeFunction&
PrefillGqaAttnSet::nax_fn(bool split, bool align_q, bool align_k) const {
  metal_compute::ComputeFunction& f =
      _nax_fn[split ? 1 : 0][align_q ? 1 : 0][align_k ? 1 : 0];
  if (!f.valid() && _lib_nax != nullptr) {
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, align_q).set_bool(201, align_k)
        .set_bool(300, false).set_bool(301, true).set_bool(302, false);
    // The head-dim split exists at bd256 only; at 128 it is not offered.
    if (split && _head_dim != 256) { return f; }
    std::string name = split ? "attn_steel_nax_dsplit_h_bd256"
                             : "attn_steel_nax_h_bd" + std::to_string(_head_dim);
    if (_nax_bf16) { name += "_bf16"; }
    f = _lib_nax->function(name, fc);
  }
  return f;
}

void
PrefillGqaAttnSet::encode_nax(metal_compute::ComputeEncoder& enc,
                              const Attn& a, bool split) const {
  const int D = _dims.D, Hq = _dims.Hq, Hkv = _dims.Hkv;
  const int n = a.n;
  const int kL = a.q_offset + a.n;
  // Strided K/V: the whole prefix plus this chunk, gathered from the pool.
  const std::size_t need = (std::size_t)Hkv * (std::size_t)kL * D * 2;
  if (_mc != nullptr && (_kc.byte_size() < need || _vc.byte_size() < need)) {
    _kc = _mc->make_shared_buffer(need);
    _vc = _mc->make_shared_buffer(need);
  }
  if (_kc.byte_size() < need || _vc.byte_size() < need) { return; }
  enc.set_function(_fn_gather);
  enc.set_buffer(0, *a.kpool);
  enc.set_buffer(1, *a.vpool);
  enc.set_buffer(2, _kc);
  enc.set_buffer(3, _vc);
  enc.set_constant(4, Hkv);
  enc.set_constant(5, D);
  enc.set_constant(6, a.page_tokens);
  enc.set_constant(7, kL);
  enc.set_buffer(8, *a.page_table);
  enc.dispatch({(unsigned)a.page_tokens, (unsigned)Hkv, (unsigned)a.n_pages},
               {32, 1, 1});

  NaxAttnParams p{};
  p.B = 1; p.H = Hq; p.D = D;
  p.qL = n; p.kL = kL;
  p.gqa_factor = Hq / Hkv;
  p.scale = a.scale;
  p.NQ = (n + kNaxBQ - 1) / kNaxBQ;
  p.NK = (kL + kNaxBK - 1) / kNaxBK;
  p.NQ_aligned = n / kNaxBQ;
  p.NK_aligned = kL / kNaxBK;
  p.qL_rem = n - p.NQ_aligned * kNaxBQ;
  p.kL_rem = kL - p.NK_aligned * kNaxBK;
  p.qL_off = kL - n;
  p.Q_strides[0] = (std::int64_t)Hq * n * D;
  p.Q_strides[1] = (std::int64_t)n * D;
  p.Q_strides[2] = D;
  p.K_strides[0] = (std::int64_t)Hkv * kL * D;
  p.K_strides[1] = (std::int64_t)kL * D;
  p.K_strides[2] = D;
  for (int i = 0; i < 3; ++i) {
    p.V_strides[i] = p.K_strides[i];
    p.O_strides[i] = p.Q_strides[i];
  }
  enc.set_function(nax_fn(split, n % kNaxBQ == 0, kL % kNaxBK == 0));
  enc.set_buffer(0, *a.qt);
  enc.set_buffer(1, _kc);
  enc.set_buffer(2, _vc);
  enc.set_buffer(3, *a.out);
  enc.set_constant(4, p);
  const unsigned wn = split ? 2u : 1u;
  enc.dispatch({32u * (unsigned)p.NQ, 4u * (unsigned)Hq, wn}, {32u, 4u, wn});
}

void
PrefillGqaAttnSet::encode_member(metal_compute::ComputeEncoder& enc,
                                 const Attn& a, int member) const {
  if (member == kNax || member == kNaxSplit) {
    encode_nax(enc, a, member == kNaxSplit);
    return;
  }
  const int D = _dims.D, Hq = _dims.Hq, Hkv = _dims.Hkv;
  enc.set_function(_fn[member]);
  enc.set_buffer(0, *a.qt);
  enc.set_buffer(1, *a.kpool);
  enc.set_buffer(2, *a.vpool);
  enc.set_buffer(3, *a.out);
  enc.set_constant(4, a.scale);
  enc.set_constant(5, D);
  enc.set_constant(6, Hq);
  enc.set_constant(7, Hkv);
  enc.set_constant(8, a.n);
  enc.set_constant(9, a.q_offset);
  enc.set_constant(10, a.page_tokens);
  enc.set_constant(11, a.n_pages);
  enc.set_buffer(12, *a.page_table);
  const unsigned n = (unsigned)a.n;
  const unsigned Hqu = (unsigned)Hq;
  switch (member) {
    case kSteel:
      enc.dispatch({32 * ((n + 31) / 32), 4 * Hqu, 1}, {32, 4, 1});
      break;
    case kMma:
      enc.dispatch({128, Hqu, (n + 15) / 16}, {128, 1, 1});
      break;
    case kFlash:
      enc.dispatch({256, Hqu, (n + 7) / 8}, {256, 1, 1});
      break;
    case kQtile:
      enc.dispatch({512, Hqu, (n + 15) / 16}, {512, 1, 1});
      break;
    default:  // kScalar
      enc.dispatch({32, Hqu, n}, {32, 1, 1});
      break;
  }
}

void
PrefillGqaAttnSet::dispatch(metal_compute::ComputeEncoder& enc,
                            const Attn& a) const {
  if (!_ready) { return; }
  encode_member(enc, a, _member[regime_of(a.n)]);
}

bool
PrefillGqaAttnSet::dispatch_member(metal_compute::ComputeEncoder& enc,
                                   const Attn& a,
                                   std::string_view member) const {
  if (!_ready) { return false; }
  for (int m = 0; m < kMembers; ++m) {
    if (member == kName[m] && _have[m]) {
      encode_member(enc, a, m);
      return true;
    }
  }
  return false;
}

void
PrefillGqaAttnSet::prepare(metal_compute::MetalCompute* mc, Dims dims,
                           TuningReport& rep) {
  _dims = dims;
  _mc = mc;
  for (int i = 0; i < kRegimes; ++i) {
    _member[i] = _have[kFlash] ? kFlash : kQtile;
  }
  const int D = dims.D, Hq = dims.Hq, Hkv = dims.Hkv;
  if (mc == nullptr || D != _head_dim || Hkv <= 0 || Hq % Hkv != 0) {
    return;
  }
  if (!_have[kFlash] && !_have[kQtile]) { return; }
  _ready = true;

  // VPIPE_QWEN_STEEL_ATTN=0 keeps steel out (A/B); explicit STEEL_MIN or
  // PREFILL_AUTOTUNE=0 skips the probe (keep the heuristic defaults below).
  const bool steel_ok = !(std::getenv("VPIPE_QWEN_STEEL_ATTN")
      && std::atoi(std::getenv("VPIPE_QWEN_STEEL_ATTN")) == 0);
  for (int i = 1; i < kRegimes; ++i) {          // default mid/long -> steel
    if (steel_ok && _have[kSteel]) { _member[i] = kSteel; }
    else if (_have[kMma]) { _member[i] = kMma; }
  }
  // A pinned member wins over both the defaults and the probe.
  auto pin = [&]() {
    const char* e = std::getenv("VPIPE_QWEN_PREFILL_ATTN");
    if (e == nullptr || *e == '\0') { return false; }
    for (int m = 0; m < kMembers; ++m) {
      if (std::string_view(e) == kName[m] && _have[m]) {
        for (int i = 0; i < kRegimes; ++i) { _member[i] = m; }
        rep.add("prefill-attn", 0.0, std::string("pinned ") + kName[m]);
        return true;
      }
    }
    // SAID, not a silent fallback: an A/B that pins a member this model or
    // GPU does not have would otherwise measure the tuner's pick under the
    // pinned member's name.
    std::fprintf(stderr,
                 "[prefill-attn] VPIPE_QWEN_PREFILL_ATTN=%s is not available "
                 "here; tuning instead\n", e);
    rep.add("prefill-attn", 0.0, std::string("pin unavailable: ") + e);
    return false;
  };
  if (pin()) { return; }
  if (std::getenv("VPIPE_QWEN_STEEL_MIN")) { return; }
  if (const char* e = std::getenv("VPIPE_QWEN_PREFILL_AUTOTUNE")) {
    if (std::atoi(e) == 0) { return; }
  }
  const bool log = std::getenv("VPIPE_QWEN_AUTOTUNE_LOG") != nullptr;
  const int page_tokens = 512;
  const float scale = 1.0f / std::sqrt((float)D);

  // Probe buffers sized by the regimes actually probed: qt/out are
  // Hq * n * D, ~250 MB each at the very-long probe, which a GPU without the
  // NAX members never runs.
  const bool probe_very_long = _have[kNax] || _have[kNaxSplit];
  int max_n = 0;
  for (int i = 0; i < kRegimes; ++i) {
    if (i == kVeryLong && !probe_very_long) { continue; }
    if (kRegimeN[i] > max_n) { max_n = kRegimeN[i]; }
  }
  const int max_pages = (max_n + page_tokens - 1) / page_tokens;
  const std::size_t qb = (std::size_t)Hq * max_n * D * 2;
  const std::size_t kvb = (std::size_t)max_pages * Hkv * page_tokens * D * 2;
  metal_compute::SharedBuffer qt = mc->make_shared_buffer(qb);
  metal_compute::SharedBuffer kp = mc->make_shared_buffer(kvb);
  metal_compute::SharedBuffer vp = mc->make_shared_buffer(kvb);
  metal_compute::SharedBuffer out = mc->make_shared_buffer(qb);
  metal_compute::SharedBuffer pgt =
      mc->make_shared_buffer((std::size_t)max_pages * 3 * 4);
  if (qt.empty() || kp.empty() || vp.empty() || out.empty() || pgt.empty()) {
    return;
  }
  int* const pt = static_cast<int*>(pgt.contents());

  Attn a;
  a.qt = &qt;
  a.kpool = &kp;
  a.vpool = &vp;
  a.out = &out;
  a.page_table = &pgt;
  a.q_offset = 0;
  a.scale = scale;
  a.page_tokens = page_tokens;

  const auto t_all = std::chrono::steady_clock::now();
  std::string detail;
  for (int ri = 0; ri < kRegimes; ++ri) {
    const int n = kRegimeN[ri];
    a.n = n;
    a.n_pages = (n + page_tokens - 1) / page_tokens;
    for (int i = 0; i < a.n_pages; ++i) {
      pt[i * 3 + 0] = i;
      pt[i * 3 + 1] = page_tokens;
      pt[i * 3 + 2] = i * page_tokens;
    }
    // Candidates: short regime -> {qtile, flash}; mid/long -> {flash, steel,
    // mma}; the NAX members in every regime (their probe includes the pool
    // gather they need). Only the present members.
    const bool nax = _have[kNax] || _have[kNaxSplit];
    if (ri < regime_of(min_n())) { continue; }   // never dispatched
    if (ri == kVeryLong && !nax) {
      _member[ri] = _member[ri - 1];
      continue;
    }
    std::vector<int> cand;
    if (ri == 0) {
      if (_have[kQtile]) { cand.push_back(kQtile); }
      if (_have[kFlash]) { cand.push_back(kFlash); }
    } else {
      if (_have[kFlash] && ri != kVeryLong) { cand.push_back(kFlash); }
      if (steel_ok && _have[kSteel]) { cand.push_back(kSteel); }
      if (_have[kMma]) { cand.push_back(kMma); }
    }
    if (_have[kNax]) { cand.push_back(kNax); }
    if (_have[kNaxSplit]) { cand.push_back(kNaxSplit); }
    if (cand.size() < 2) { continue; }           // nothing to choose
    // O(n^2) prefill attention -> few reps suffice (the per-dispatch signal is
    // large); keep the probe wall-time bounded.
    const int reps = (n <= 1024) ? 3 : 1;        // O(n^2) -> 1 rep at large n
    auto bench = [&](int i) -> double {
      const int member = cand[(std::size_t)i];
      return autotune_time(mc, reps, [&](metal_compute::ComputeEncoder& enc) {
        encode_member(enc, a, member);
      });
    };
    std::vector<double> us;
    const int w = autotune_vote((int)cand.size(), 2, reps, bench, &us);
    _member[ri] = cand[(std::size_t)w];
    if (!detail.empty()) { detail += "/"; }
    detail += std::string(kName[_member[ri]]) + "@" + std::to_string(n);
    if (log) {
      std::string line = "[qwen] prefill-attn regime " + std::to_string(ri)
          + " @n=" + std::to_string(n) + ":";
      for (std::size_t i = 0; i < cand.size(); ++i) {
        char b[48];
        std::snprintf(b, sizeof(b), " %s %.0fus", kName[cand[i]], us[i]);
        line += b;
      }
      line += " -> " + std::string(kName[_member[ri]]);
      std::fprintf(stderr, "%s\n", line.c_str());
    }
  }
  const double ms = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_all).count() * 1e3;
  rep.add("prefill-attn", ms, detail);
  if (log) {
    std::fprintf(stderr, "[qwen] prefill-attn tuning %.0fms -> %s\n", ms,
                 detail.c_str());
  }
}

}  // namespace vpipe::genai
