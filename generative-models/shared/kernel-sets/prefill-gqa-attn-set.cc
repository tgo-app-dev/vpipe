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
const int kRegimeLo[]  = {0, 1536, 3072};
const int kRegimeN[]   = {768, 2048, 3072};   // long probe capped (O(n^2) cost)
const char* kName[5] = {"scalar", "qtile", "flash", "mma", "steel"};
}  // namespace

bool
PrefillGqaAttnSet::load(metal_compute::ComputeLibrary& lib_sdpa,
                        metal_compute::ComputeLibrary* lib_attn,
                        metal_compute::ComputeLibrary* lib_mma, bool use_mma) {
  _fn[kScalar] = lib_sdpa.function("sdpa_paged_causal_f16");
  _fn[kQtile] = lib_sdpa.function("sdpa_paged_qtile_f16");
  _fn[kFlash] = lib_sdpa.function("sdpa_paged_flash_f16");
  if (lib_mma != nullptr && lib_mma->valid()) {
    _fn[kMma] = lib_mma->function("sdpa_mma_f16");
  }
  if (lib_attn != nullptr && lib_attn->valid()) {
    _fn[kSteel] = lib_attn->function("attn_steel_paged_bd256");
  }
  _have[kScalar] = _fn[kScalar].valid();
  _have[kQtile] = _fn[kQtile].valid();
  _have[kFlash] = _fn[kFlash].valid();
  _have[kMma] = use_mma && _fn[kMma].valid();
  _have[kSteel] = _fn[kSteel].valid();
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

const char*
PrefillGqaAttnSet::kernel_name(int n) const {
  return kName[_member[regime_of(n)]];
}

void
PrefillGqaAttnSet::encode_member(metal_compute::ComputeEncoder& enc,
                                 const Attn& a, int member) const {
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

void
PrefillGqaAttnSet::prepare(metal_compute::MetalCompute* mc, Dims dims,
                           TuningReport& rep) {
  _dims = dims;
  for (int i = 0; i < kRegimes; ++i) {
    _member[i] = _have[kFlash] ? kFlash : kQtile;
  }
  const int D = dims.D, Hq = dims.Hq, Hkv = dims.Hkv;
  if (mc == nullptr || D != 256 || Hkv <= 0 || Hq % Hkv != 0) { return; }
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
  if (std::getenv("VPIPE_QWEN_STEEL_MIN")) { return; }
  if (const char* e = std::getenv("VPIPE_QWEN_PREFILL_AUTOTUNE")) {
    if (std::atoi(e) == 0) { return; }
  }
  const bool log = std::getenv("VPIPE_QWEN_AUTOTUNE_LOG") != nullptr;
  const int page_tokens = 512;
  const float scale = 1.0f / std::sqrt((float)D);

  int max_n = 0;
  for (int i = 0; i < kRegimes; ++i) {
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
    // Candidates: short regime -> {scalar?, qtile, flash}; mid/long -> {flash,
    // steel, mma}. Only the present members.
    std::vector<int> cand;
    if (ri == 0) {
      if (_have[kQtile]) { cand.push_back(kQtile); }
      if (_have[kFlash]) { cand.push_back(kFlash); }
    } else {
      if (_have[kFlash]) { cand.push_back(kFlash); }
      if (steel_ok && _have[kSteel]) { cand.push_back(kSteel); }
      if (_have[kMma]) { cand.push_back(kMma); }
    }
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
