#include "generative-models/shared/kernel-sets/decode-gqa-attn-set.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/shared/tuned-answers.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace vpipe::genai {

namespace {
// Regime lower bounds (KV length) + the ctx each is probed at (a pos inside the
// regime; vec2's split flips 128->256 at 8k). kRegimes mirrors the header.
const int kRegimeLo[]  = {0, 8192, 24576};
const int kRegimeCtx[] = {4096, 16384, 32768};
int vec2_split(int pos) { return pos <= 8192 ? 128 : 256; }
int kbl_split(int pos) {
  const int s = (pos + 127) / 128;
  return s < 16 ? 16 : (s > 256 ? 256 : s);
}
}  // namespace

bool
DecodeGqaAttnSet::load(metal_compute::ComputeLibrary& lib, bool use_bf16) {
  _use_bf16 = use_bf16;
  _vec2 = lib.function("sdpa_paged_gqa_vec2_f16");
  _merge2 = lib.function("sdpa_gqa_merge2_f16");
  _kbl = lib.function("sdpa_paged_gqa_kbl_f16");
  _merge = lib.function("sdpa_gqa_merge_f16");
  // Usable if at least one member + its merge resolved.
  return (_vec2.valid() && _merge2.valid()) || (_kbl.valid() && _merge.valid());
}

int
DecodeGqaAttnSet::regime_of(int pos) const {
  int r = 0;
  for (int i = 1; i < kRegimes; ++i) {
    if (pos >= kRegimeLo[i]) { r = i; }
  }
  return r;
}

int
DecodeGqaAttnSet::split_for(int pos) const {
  static const int kEnv = []() {
    const char* e = std::getenv("VPIPE_QWEN_GQA_VEC2");
    return e ? (std::atoi(e) != 0 ? 1 : 0) : -1;
  }();
  if (kEnv >= 0) { return kEnv != 0 ? vec2_split(pos) : kbl_split(pos); }
  return _split[regime_of(pos)];
}

bool
DecodeGqaAttnSet::uses_vec2(int pos) const {
  static const int kEnv = []() {
    const char* e = std::getenv("VPIPE_QWEN_GQA_VEC2");
    return e ? (std::atoi(e) != 0 ? 1 : 0) : -1;
  }();
  return (kEnv >= 0) ? (kEnv != 0) : (_kernel[regime_of(pos)] != 0);
}

const char*
DecodeGqaAttnSet::kernel_name(int pos) const {
  return uses_vec2(pos) ? "vec2" : "kbl";
}

void
DecodeGqaAttnSet::encode_member(metal_compute::ComputeEncoder& enc,
                                const Attn& a, bool vec2, int split, int rows,
                                const metal_compute::SharedBuffer* o,
                                const metal_compute::SharedBuffer* mb,
                                const metal_compute::SharedBuffer* lb) const {
  const int D = _dims.D, Hq = _dims.Hq, Hkv = _dims.Hkv, G = Hq / Hkv;
  const metal_compute::SharedBuffer& oacc = o != nullptr ? *o : _oacc;
  const metal_compute::SharedBuffer& mm = mb != nullptr ? *mb : _m;
  const metal_compute::SharedBuffer& ll = lb != nullptr ? *lb : _l;
  const int heads = Hq * rows;   // the merges' heads: row r's at r * Hq
  // Pass 1: per-(kvhead, split) partials into the set's scratch.
  enc.set_function(vec2 ? _vec2 : _kbl);
  enc.set_buffer(0, *a.q, a.q_off);
  enc.set_buffer(1, *a.kpool);
  enc.set_buffer(2, *a.vpool);
  enc.set_buffer(3, oacc);
  enc.set_buffer(4, mm);
  enc.set_buffer(5, ll);
  enc.set_constant(6, a.scale);
  enc.set_constant(7, D);
  enc.set_constant(8, Hq);
  enc.set_constant(9, Hkv);
  enc.set_constant(10, a.pos);
  enc.set_constant(11, a.page_tokens);
  enc.set_constant(12, a.n_pages);
  enc.set_buffer(13, *a.page_table, a.pgt_off);
  enc.set_constant(14, split);
  enc.dispatch({32, (unsigned)Hq, (unsigned)(split * rows)},
               {32, (unsigned)G, 1});
  // Pass 2: merge the splits -> out. vec2 uses the coalesced merge2 (sums=l
  // before maxs=m, 1 TG/qhead); kbl uses the one-thread-per-d merge.
  if (vec2) {
    enc.set_function(_merge2);
    enc.set_buffer(0, oacc);
    enc.set_buffer(1, ll);
    enc.set_buffer(2, mm);
    enc.set_buffer(3, *a.out, a.out_off);
    enc.set_constant(4, D);
    enc.set_constant(5, split);
    enc.dispatch({1024, (unsigned)heads, 1}, {1024, 1, 1});
  } else {
    enc.set_function(_merge);
    enc.set_buffer(0, oacc);
    enc.set_buffer(1, mm);
    enc.set_buffer(2, ll);
    enc.set_buffer(3, *a.out, a.out_off);
    enc.set_constant(4, D);
    enc.set_constant(5, split);
    enc.set_constant(6, heads);
    enc.dispatch({(unsigned)(heads * D), 1, 1}, {256, 1, 1});
  }
}

void
DecodeGqaAttnSet::dispatch_rows(metal_compute::ComputeEncoder& enc,
                                const Attn& a, int rows,
                                const metal_compute::SharedBuffer& o,
                                const metal_compute::SharedBuffer& m,
                                const metal_compute::SharedBuffer& l) const {
  if (!_ready || rows <= 0) { return; }
  static const int kEnv = []() {
    const char* e = std::getenv("VPIPE_QWEN_GQA_VEC2");
    return e ? (std::atoi(e) != 0 ? 1 : 0) : -1;
  }();
  const int r = regime_of(a.pos);
  const bool vec2 = (kEnv >= 0) ? (kEnv != 0) : (_kernel[r] != 0);
  const int split = (kEnv >= 0) ? (vec2 ? vec2_split(a.pos) : kbl_split(a.pos))
                                : _split[r];
  encode_member(enc, a, vec2, split, rows, &o, &m, &l);
}

void
DecodeGqaAttnSet::dispatch(metal_compute::ComputeEncoder& enc,
                           const Attn& a) const {
  if (!_ready) { return; }
  static const int kEnv = []() {
    const char* e = std::getenv("VPIPE_QWEN_GQA_VEC2");
    return e ? (std::atoi(e) != 0 ? 1 : 0) : -1;
  }();
  const int r = regime_of(a.pos);
  const bool vec2 = (kEnv >= 0) ? (kEnv != 0) : (_kernel[r] != 0);
  const int split = (kEnv >= 0) ? (vec2 ? vec2_split(a.pos) : kbl_split(a.pos))
                                : _split[r];
  encode_member(enc, a, vec2, split);
}

void
DecodeGqaAttnSet::prepare(metal_compute::MetalCompute* mc, Dims dims,
                          TuningReport& rep, bool measure) {
  _dims = dims;
  for (int i = 0; i < kRegimes; ++i) {
    _kernel[i] = 1;
    _split[i] = vec2_split(kRegimeCtx[i] - 1);
  }
  const int D = dims.D, Hq = dims.Hq, Hkv = dims.Hkv;
  if (mc == nullptr || D != 256 || Hkv <= 0 || Hq % Hkv != 0) { return; }
  if (!_vec2.valid() || !_merge2.valid() || !_kbl.valid() || !_merge.valid()) {
    return;
  }
  // Set-owned partial scratch O[Hq,256,D] + m/l[Hq,256] (max split).
  const std::size_t sp = 256;
  _oacc = mc->make_shared_buffer((std::size_t)Hq * sp * D * sizeof(float));
  _m = mc->make_shared_buffer((std::size_t)Hq * sp * sizeof(float));
  _l = mc->make_shared_buffer((std::size_t)Hq * sp * sizeof(float));
  if (_oacc.empty() || _m.empty() || _l.empty()) { return; }
  _ready = true;

  // Skip the probe under an explicit override / disabled autotune.
  if (!measure) {
    rep.add("decode-attn", 0.0, "untuned vec2");
    return;
  }
  if (std::getenv("VPIPE_QWEN_GQA_VEC2")) { return; }
  if (const char* e = std::getenv("VPIPE_QWEN_GQA_AUTOTUNE")) {
    if (std::atoi(e) == 0) { return; }
  }
  const bool log = std::getenv("VPIPE_QWEN_AUTOTUNE_LOG") != nullptr;
  // What this session measured for these dims before -- the helper is
  // loaded again whenever something else needed its memory
  // (tuned-answers.h). Kernel and split per regime.
  KernelTuningCache* const answers = tuning_cache(mc);
  const TuneKey key = TuneKey("decode-gqa-attn/1")("D", D)("Hq", Hq)
      ("Hkv", Hkv)("bf16", _use_bf16);
  std::vector<int> kept;
  if (find_tuned(answers, key, kept) && kept.size() == 2 * kRegimes) {
    std::string detail;
    for (int ri = 0; ri < kRegimes; ++ri) {
      _kernel[ri] = kept[(std::size_t)(2 * ri)] != 0 ? 1 : 0;
      _split[ri] = kept[(std::size_t)(2 * ri + 1)];
      if (!detail.empty()) { detail += "/"; }
      detail += std::string(_kernel[ri] ? "vec2@" : "kbl@")
              + std::to_string(_split[ri]);
    }
    rep.add("decode-attn", 0.0, detail + ", the session's");
    return;
  }
  const int page_tokens = 512;
  const float scale = 1.0f / std::sqrt((float)D);

  int max_ctx = 0;
  for (int i = 0; i < kRegimes; ++i) {
    if (kRegimeCtx[i] > max_ctx) { max_ctx = kRegimeCtx[i]; }
  }
  const int max_pages = (max_ctx + page_tokens - 1) / page_tokens;
  const std::size_t kvb = (std::size_t)max_pages * Hkv * page_tokens * D * 2;
  metal_compute::SharedBuffer q = mc->make_shared_buffer((std::size_t)Hq * D * 2);
  metal_compute::SharedBuffer kp = mc->make_shared_buffer(kvb);
  metal_compute::SharedBuffer vp = mc->make_shared_buffer(kvb);
  metal_compute::SharedBuffer out = mc->make_shared_buffer((std::size_t)Hq * D * 2);
  metal_compute::SharedBuffer pgt =
      mc->make_shared_buffer((std::size_t)max_pages * 3 * 4);
  if (q.empty() || kp.empty() || vp.empty() || out.empty() || pgt.empty()) {
    return;
  }
  int* const pt = static_cast<int*>(pgt.contents());

  int n_pages = 0, pos = 0;
  Attn a;
  a.q = &q;
  a.kpool = &kp;
  a.vpool = &vp;
  a.out = &out;
  a.page_table = &pgt;
  a.scale = scale;

  struct Cand { bool vec2; int split; };
  const auto t_all = std::chrono::steady_clock::now();
  std::string detail;
  for (int ri = 0; ri < kRegimes; ++ri) {
    const int ctx = kRegimeCtx[ri];
    n_pages = (ctx + page_tokens - 1) / page_tokens;
    pos = ctx - 1;
    a.pos = pos;
    a.page_tokens = page_tokens;
    a.n_pages = n_pages;
    int reps = 32 * 4096 / ctx;
    if (reps < 6) { reps = 6; }
    if (reps > 32) { reps = 32; }
    for (int i = 0; i < n_pages; ++i) {
      pt[i * 3 + 0] = i;
      pt[i * 3 + 1] = page_tokens;
      pt[i * 3 + 2] = i * page_tokens;
    }
    const int vs = vec2_split(pos);
    const int ks = kbl_split(pos);
    int ks2 = (ks * 2 <= 256) ? ks * 2 : (ks > 32 ? ks / 2 : ks);
    if (ks2 == ks) { ks2 = (ks < 256) ? 256 : 128; }
    const Cand cand[4] = {{true, vs}, {true, vs == 128 ? 256 : 128},
                          {false, ks}, {false, ks2}};
    auto bench = [&](int i) -> double {
      const Cand cd = cand[i];
      return autotune_time(mc, reps, [&](metal_compute::ComputeEncoder& enc) {
        encode_member(enc, a, cd.vec2, cd.split);
      });
    };
    std::vector<double> us;
    const int w = autotune_vote(4, 3, reps, bench, &us);
    _kernel[ri] = cand[w].vec2 ? 1 : 0;
    _split[ri] = cand[w].split;
    if (!detail.empty()) { detail += "/"; }
    detail += std::string(cand[w].vec2 ? "vec2@" : "kbl@")
            + std::to_string(cand[w].split);
    if (log) {
      std::fprintf(stderr,
          "[qwen] decode-attn regime %d @ctx=%d: vec2@%d %.0fus  vec2@%d %.0fus"
          "  kbl@%d %.0fus  kbl@%d %.0fus  -> %s@%d\n",
          ri, ctx, cand[0].split, us[0], cand[1].split, us[1], cand[2].split,
          us[2], cand[3].split, us[3], cand[w].vec2 ? "vec2" : "kbl",
          cand[w].split);
    }
  }
  const double ms = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t_all).count() * 1e3;
  rep.add("decode-attn", ms, detail);
  std::vector<int> answer;
  for (int ri = 0; ri < kRegimes; ++ri) {
    answer.push_back(_kernel[ri]);
    answer.push_back(_split[ri]);
  }
  store_tuned(answers, key, answer);
  if (log) {
    std::fprintf(stderr, "[qwen] decode-attn tuning %.0fms -> %s\n", ms,
                 detail.c_str());
  }
}

}  // namespace vpipe::genai
