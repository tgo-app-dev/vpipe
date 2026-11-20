#include "generative-models/relayvsr/relayvsr-ops.h"

#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/fp8-expand.h"
#include "generative-models/shared/fp8-layout.h"
#include "generative-models/shared/fp8.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vpipe {
namespace genai {
namespace relayvsr {

using metal_compute::ComputeEncoder;
using metal_compute::ComputeFunction;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// The steel flash kernel's params (attn_steel.metal; the same struct the
// DiTs mirror).
struct SteelAttnParams {
  int B, H, D;
  int qL, kL;
  int gqa_factor;
  float scale;
  int NQ, NK;
  int NQ_aligned, NK_aligned;
  int qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

}  // namespace

std::uint16_t
to_bf16(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) { return (std::uint16_t)0x7fc0; }
  const std::uint32_t r = (u >> 16) & 1u;
  u += 0x7fffu + r;
  return (std::uint16_t)(u >> 16);
}

float
from_bf16(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::uint16_t
to_f16(float f)
{
  const _Float16 h = (_Float16)f;
  std::uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

float
from_f16(std::uint16_t u)
{
  _Float16 h;
  std::memcpy(&h, &u, 2);
  return (float)h;
}

float
round_elt(float v, bool bf16)
{
  return bf16 ? from_bf16(to_bf16(v)) : from_f16(to_f16(v));
}

bool
read_f32(WeightSet& ws, MetalCompute* mc, const std::string& name,
         std::vector<float>* out, std::vector<std::int64_t>* shape)
{
  const auto* info = ws.src().info(name);
  if (info == nullptr) { return false; }
  SharedBuffer b = ws.read(name, mc, WeightSet::Residency::Copied);
  if (b.empty()) { return false; }
  std::size_t n = 1;
  for (const auto d : info->shape) { n *= (std::size_t)d; }
  out->assign(n, 0.0f);
  const void* p = b.contents();
  if (info->dtype == "F32") {
    std::memcpy(out->data(), p, n * 4);
  } else if (info->dtype == "BF16") {
    const auto* s = (const std::uint16_t*)p;
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_bf16(s[i]); }
  } else if (info->dtype == "F16") {
    const auto* s = (const std::uint16_t*)p;
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = from_f16(s[i]); }
  } else {
    return false;
  }
  if (shape != nullptr) { *shape = info->shape; }
  return true;
}

bool
read_raw(WeightSet& ws, MetalCompute* mc, const std::string& name,
         RawTensor* out)
{
  const auto* info = ws.src().info(name);
  if (info == nullptr) { return false; }
  RawTensor t;
  if (info->dtype == "F32") {
    t.kind = RawTensor::Kind::kF32;
  } else if (info->dtype == "BF16") {
    t.kind = RawTensor::Kind::kBF16;
  } else if (info->dtype == "F16") {
    t.kind = RawTensor::Kind::kF16;
  } else {
    return false;
  }
  t.buf = ws.read(name, mc, WeightSet::Residency::Copied);
  if (t.buf.empty()) { return false; }
  t.shape = info->shape;
  t.n = 1;
  for (const auto d : info->shape) { t.n *= (std::size_t)d; }
  *out = std::move(t);
  return true;
}

SharedBuffer
convert_elt(MetalCompute* mc, const RawTensor& t, bool bf16)
{
  SharedBuffer b =
      mc->make_shared_buffer(std::max<std::size_t>(t.n, 1) * 2);
  if (b.empty()) { return b; }
  auto* d = (std::uint16_t*)b.contents();
  for (std::size_t i = 0; i < t.n; ++i) {
    const float v = t.at(i);
    d[i] = bf16 ? to_bf16(v) : to_f16(v);
  }
  return b;
}

SharedBuffer
tap_major_elt(MetalCompute* mc, const RawTensor& t, bool bf16)
{
  if (t.shape.size() != 4 || t.shape[2] != 3 || t.shape[3] != 3) {
    return SharedBuffer{};
  }
  const std::int64_t co = t.shape[0], ci = t.shape[1];
  SharedBuffer b = mc->make_shared_buffer(std::max<std::size_t>(t.n, 1) * 2);
  if (b.empty()) { return b; }
  auto* d = (std::uint16_t*)b.contents();
  for (std::int64_t a = 0; a < co; ++a) {
    for (std::int64_t c = 0; c < ci; ++c) {
      for (int tap = 0; tap < 9; ++tap) {
        const float v = t.at((std::size_t)((a * ci + c) * 9 + tap));
        d[(std::size_t)((a * 9 + tap) * ci + c)] =
            bf16 ? to_bf16(v) : to_f16(v);
      }
    }
  }
  return b;
}

SharedBuffer
upload_elt(MetalCompute* mc, const std::vector<float>& v, bool bf16)
{
  SharedBuffer b = mc->make_shared_buffer(std::max<std::size_t>(v.size(), 1)
                                          * 2);
  if (b.empty()) { return b; }
  auto* d = (std::uint16_t*)b.contents();
  for (std::size_t i = 0; i < v.size(); ++i) {
    d[i] = bf16 ? to_bf16(v[i]) : to_f16(v[i]);
  }
  return b;
}

SharedBuffer
upload_f32(MetalCompute* mc, const std::vector<float>& v)
{
  SharedBuffer b = mc->make_shared_buffer(std::max<std::size_t>(v.size(), 1)
                                          * 4);
  if (!b.empty() && !v.empty()) {
    std::memcpy(b.contents(), v.data(), v.size() * 4);
  }
  return b;
}

SharedBuffer
tensor_elt(WeightSet& ws, MetalCompute* mc, const std::string& name,
           bool bf16, const std::string& key)
{
  const auto* info = ws.src().info(name);
  if (info == nullptr) { return {}; }
  // Already the element type: the checkpoint's own bytes, cached.
  if (info->dtype == (bf16 ? "BF16" : "F16")) {
    return ws.tensor(name, mc, WeightSet::Residency::Copied);
  }
  return ws.derived(key + (bf16 ? "bf16/" : "f16/") + name, [&]() {
    RawTensor r;
    if (!read_raw(ws, mc, name, &r)) { return SharedBuffer{}; }
    return convert_elt(mc, r, bf16);
  });
}

namespace {

double
now_ms_()
{
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

Profile::Profile(const char* what) : _what(what)
{
  const char* e = std::getenv("VPIPE_RELAYVSR_PROFILE");
  _on = e != nullptr && *e != '\0' && std::atoi(e) != 0;
  _t = now_ms_();
}

void
Profile::add_(const char* cat)
{
  const double t = now_ms_();
  const double ms = t - _t;
  _t = t;
  for (Bucket& b : _b) {
    if (b.name == cat) {
      b.ms += ms;
      ++b.marks;
      return;
    }
  }
  _b.push_back(Bucket{cat, ms, 1});
}

void
Profile::mark(ComputeEncoder& enc, const char* cat)
{
  if (!_on) { return; }
  (void)metal_compute::CommandStream::sync_and_reopen(enc);
  add_(cat);
}

void
Profile::host(const char* cat)
{
  if (_on) { add_(cat); }
}

void
Profile::report(MetalCompute* mc, const std::string& tail)
{
  if (!_on) { return; }
  double total = 0.0;
  for (const Bucket& b : _b) { total += b.ms; }
  std::string line = fmt("RelayVSR profile, {}: {:.1f} ms", _what, total)();
  for (const Bucket& b : _b) {
    line += fmt(" | {} {:.1f} ({:.0f}%, {})", b.name, b.ms,
                total > 0.0 ? 100.0 * b.ms / total : 0.0, b.marks)();
  }
  if (!tail.empty()) { line += " | " + tail; }
  if (mc != nullptr && mc->session() != nullptr) {
    mc->session()->info(fmt("{}", line));
  } else {
    std::fprintf(stderr, "%s\n", line.c_str());
  }
  _b.clear();
  _t = now_ms_();
}

// ---- the int8 tiers --------------------------------------------------------

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
proj_bytes(int N, int K, int tier, bool int8)
{
  const std::uint64_t nk = (std::uint64_t)N * (std::uint64_t)K;
  const int gc = held_group(K, tier);
  if (int8) {
    // As stored (codes and an f32 scale a row) -- which is also the token
    // layout's size -- but for the fast tier's own layout.
    return tier == 1 && gc > 0
               ? nk + I8GemmContext::fast_scale_bytes(N, K, gc)
               : nk + (std::uint64_t)N * 4u;
  }
  return gc > 0 ? nk + I8GemmContext::fast_scale_bytes(N, K, gc) : nk * 2u;
}

bool
convrot_native(int K, int tier, int rot)
{
  if (const char* e = std::getenv("VPIPE_RELAYVSR_CONVROT_NATIVE")) {
    if (std::atoi(e) == 0) { return false; }
  }
  return (tier == 2 || tier == 3) && rot == 256 && K % 256 == 0 &&
         held_group(K, tier) == I8GemmContext::kTokenGroup;
}

int
ane_chunk(int rows)
{
  if (const char* e = std::getenv("VPIPE_RELAYVSR_ANE_CHUNK")) {
    const int c = std::atoi(e);
    if (c >= 256 && c <= 16384) { return c; }
  }
  return rows >= 4096 ? 2048 : rows >= 2048 ? 1024 : 512;
}

Ops::Ops() = default;
Ops::~Ops() = default;

bool
Ops::set_i8(int tier, bool autotune)
{
  _i8.reset();
  _i8_tier = -1;
  if (tier < 0 || _mc == nullptr) { return false; }
  auto i8 = std::make_unique<I8GemmContext>(_mc, true, _bf16);
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
  // A held layout takes its tier's GEMM at any row count -- a small
  // frame's keyframe has under the requant route's 1024 -- rather than
  // being widened back to the element type.
  if (_i8_tier >= 1) { _i8->set_min_rows(1); }
  return true;
}

Proj
Ops::load_proj(WeightSet& ws, const std::string& weight,
               const std::string& bias, int N, int K, const std::string& key,
               bool hold, const fp8::Weight* q, bool cache, std::string* err)
{
  Proj p;
  p.n = N;
  p.k = K;
  if (!bias.empty()) {
    if (cache) {
      p.b = tensor_elt(ws, _mc, bias, _bf16, key);
    } else {
      RawTensor r;
      if (read_raw(ws, _mc, bias, &r)) { p.b = convert_elt(_mc, r, _bf16); }
    }
    if (p.b.empty()) { return Proj{}; }
  }
  if (q != nullptr) {
    return load_int8_(ws, weight, std::move(p), key, hold, *q, cache, err);
  }
  const int gc = hold && _i8 && _i8_tier >= 1 ? _i8->fast_group(K) : 0;
  if (gc == 0) {
    if (cache) {
      p.w = tensor_elt(ws, _mc, weight, _bf16, key);
    } else {
      const auto* info = ws.src().info(weight);
      if (info != nullptr && info->dtype == (_bf16 ? "BF16" : "F16")) {
        p.w = ws.read(weight, _mc, WeightSet::Residency::Copied);
      } else {
        RawTensor r;
        if (read_raw(ws, _mc, weight, &r)) {
          p.w = convert_elt(_mc, r, _bf16);
        }
      }
    }
    if (p.w.empty() || p.w.byte_size() < (std::size_t)N * K * 2) {
      return Proj{};
    }
    return p;
  }
  const std::size_t nk = (std::size_t)N * (std::size_t)K;
  const std::size_t sb = I8GemmContext::fast_scale_bytes(N, K, gc);
  MetalCompute* mc = _mc;
  SharedBuffer both = ws.derived(
      key + (_bf16 ? "bf16/" : "f16/") + _i8->layout_tag(gc) + "/" + weight,
      [&]() -> SharedBuffer {
        // The element-type weight, read uncached and consumed here.
        const auto* info = ws.src().info(weight);
        if (info == nullptr) { return {}; }
        SharedBuffer f;
        if (info->dtype == (_bf16 ? "BF16" : "F16")) {
          f = ws.read(weight, mc, WeightSet::Residency::Copied);
        } else {
          RawTensor r;
          if (read_raw(ws, mc, weight, &r)) { f = convert_elt(mc, r, _bf16); }
        }
        SharedBuffer out = mc->make_shared_buffer(nk + sb);
        if (f.empty() || f.byte_size() < nk * 2 || out.empty()) { return {}; }
        metal_compute::CommandStream st = mc->make_command_stream();
        bool ok = false;
        {
          ComputeEncoder e = st.begin_compute();
          ok = _i8->encode_quant_fast(e, f, out.subview(0, nk),
                                      out.subview(nk, sb), gc, N, K);
        }
        return ok && st.commit().wait_ok(nullptr) ? std::move(out)
                                                  : SharedBuffer{};
      });
  if (both.empty()) { return Proj{}; }
  p.gc = gc;
  p.codes = both.subview(0, nk);
  p.scales = both.subview(nk, sb);
  return p;
}

Proj
Ops::load_int8_(WeightSet& ws, const std::string& weight, Proj p,
                const std::string& key, bool hold, const fp8::Weight& q,
                bool cache, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return Proj{};
  };
  const int N = p.n, K = p.k;
  if (q.format != fp8::Format::kI8 || !_bf16 ||
      (q.rot_group != 0 && K % q.rot_group != 0)) {
    return fail(fmt("relayvsr: '{}' is an 8-bit weight this model does not "
                    "run (only ComfyUI's int8, in the bf16 model)", weight)());
  }
  if (!_x8) {
    auto x = std::make_unique<fp8::Expander>();
    if (!x->load(_mc) || !x->int8_valid()) {
      return fail("relayvsr: the int8 widening kernels are missing (an older "
                  "host): ComfyUI's int8 checkpoint cannot run here");
    }
    _x8 = std::move(x);
  }
  // Its scale, one f32 a row (a tensor-wide one repeated).
  auto scales = [&]() -> SharedBuffer {
    std::vector<float> v;
    std::string se;
    if (!fp8::read_scale(ws.src(), q, (std::size_t)N, &v, &se) || v.empty()) {
      return {};
    }
    if (v.size() == 1) { v.assign((std::size_t)N, v[0]); }
    if (v.size() != (std::size_t)N) { return {}; }
    return upload_f32(_mc, v);
  };
  // Read uncached: only the routes that keep the codes as stored keep it
  // (a derived entry, below); the fast tier's conversion consumes it.
  SharedBuffer sc = scales();
  if (sc.empty()) { return fail(fmt("relayvsr: '{}' has no scale", weight)()); }
  auto kept = [&]() -> SharedBuffer {
    if (!cache) { return std::move(sc); }
    return ws.derived(key + "i8-scale/" + weight,
                      [&]() -> SharedBuffer { return std::move(sc); });
  };
  const std::size_t nk = (std::size_t)N * (std::size_t)K;
  // THE TOKEN TIERS run the codes as they are.
  if (hold && _i8 && _i8->convrot_available() &&
      convrot_native(K, _i8_tier, q.rot_group) &&
      _i8->fast_group(K) == I8GemmContext::kTokenGroup) {
    p.i8 = ws.tensor(weight, _mc, WeightSet::Residency::Copied);
    if (p.i8.empty() || p.i8.byte_size() < nk) { return Proj{}; }
    p.rot = q.rot_group;
    p.cr = true;
    p.gc = I8GemmContext::kTokenGroup;
    p.i8s = kept();
    if (p.i8s.empty()) { return Proj{}; }
    p.codes = p.i8.subview(0, p.i8.byte_size());
    p.scales = p.i8s.subview(0, p.i8s.byte_size());
    return p;
  }
  // A TIER THAT HOLDS ITS OWN LAYOUT (fast; token where the codes are not
  // its own): the weight widened on the GPU, then quantized into the
  // layout -- once, into a derived entry; the codes read uncached.
  const int gc = hold && _i8 && _i8_tier >= 1 ? _i8->fast_group(K) : 0;
  if (gc != 0) {
    const std::size_t sb = I8GemmContext::fast_scale_bytes(N, K, gc);
    MetalCompute* mc = _mc;
    SharedBuffer both = ws.derived(
        key + "bf16-i8/" + _i8->layout_tag(gc) + "/" + weight,
        [&]() -> SharedBuffer {
          SharedBuffer codes = ws.read(weight, mc,
                                       WeightSet::Residency::Copied);
          SharedBuffer f = mc->make_shared_buffer(nk * 2);
          SharedBuffer out = mc->make_shared_buffer(nk + sb);
          if (codes.byte_size() < nk || f.empty() || out.empty()) {
            return {};
          }
          fp8::Codes c;
          c.codes = &codes;
          c.format = fp8::Format::kI8;
          c.scale = &sc;
          c.per_row = true;
          c.rot_group = q.rot_group;
          metal_compute::CommandStream st = mc->make_command_stream();
          bool ok = false;
          {
            ComputeEncoder e = st.begin_compute();
            ok = _x8->encode(e, c, f, (std::size_t)N, (std::size_t)K) &&
                 _i8->encode_quant_fast(e, f, out.subview(0, nk),
                                        out.subview(nk, sb), gc, N, K);
          }
          return ok && st.commit().wait_ok(nullptr) ? std::move(out)
                                                    : SharedBuffer{};
        });
    if (both.empty()) { return Proj{}; }
    p.gc = gc;
    p.codes = both.subview(0, nk);
    p.scales = both.subview(nk, sb);
    return p;
  }
  // AS STORED, widened right before each GEMM -- into one scratch, sized
  // at the widest such projection a forward will widen (widen_).
  p.i8 = cache ? ws.tensor(weight, _mc, WeightSet::Residency::Copied)
               : ws.read(weight, _mc, WeightSet::Residency::Copied);
  if (p.i8.empty() || p.i8.byte_size() < nk) { return Proj{}; }
  p.i8s = kept();
  if (p.i8s.empty()) { return Proj{}; }
  p.rot = q.rot_group;
  if (cache) { _wdeq_max = std::max(_wdeq_max, nk * 2); }
  return p;
}

bool
Ops::widen_(ComputeEncoder& enc, const Proj& p) const
{
  if (!_x8 || p.i8.empty() || p.i8s.empty()) { return false; }
  const std::size_t need = (std::size_t)p.n * (std::size_t)p.k * 2;
  if (_wdeq.byte_size() < need) {
    _wdeq = _mc->make_shared_buffer(std::max(need, _wdeq_max));
    if (_wdeq.empty()) { return false; }
    ++_n_wdeq_allocs;
  }
  fp8::Codes c;
  c.codes = &p.i8;
  c.format = fp8::Format::kI8;
  c.scale = &p.i8s;
  c.per_row = true;
  c.rot_group = p.rot;
  if (!_x8->encode(enc, c, _wdeq, (std::size_t)p.n, (std::size_t)p.k)) {
    return false;
  }
  ++_n_widen;
  return true;
}

const SharedBuffer*
Ops::dense_weight(ComputeEncoder& enc, const Proj& p) const
{
  if (p.held()) { return nullptr; }
  if (p.stored()) { return widen_(enc, p) ? &_wdeq : nullptr; }
  return &p.w;
}

std::uint64_t
Ops::widen_bytes(int tier, bool int8, const std::vector<int>& n,
                 const std::vector<int>& k)
{
  if (!int8) { return 0; }
  std::uint64_t b = 0;
  for (std::size_t i = 0; i < n.size() && i < k.size(); ++i) {
    // A projection no tier holds is widened per GEMM.
    if (held_group(k[i], tier) == 0 || tier < 1) {
      b = std::max<std::uint64_t>(b, (std::uint64_t)n[i] * k[i] * 2u);
    }
  }
  return b;
}

void
Ops::proj(ComputeEncoder& enc, const SharedBuffer& x, std::size_t x_off,
          const Proj& p, const SharedBuffer& y, std::size_t y_off,
          int M) const
{
  if (M <= 0) { return; }
  const std::size_t xe = x_off / 2, ye = y_off / 2;
  if (p.held()) {
    // ComfyUI's ConvRot codes as the token layout: the activation rotated
    // and quantized alike, then the tier's GEMM over the codes as stored.
    if (p.cr ? _i8->quantize_token_convrot(enc, x, xe, M, p.k) &&
                   _i8->gemm_fast_quantized(enc, p.codes, p.scales, p.gc,
                                            p.b, y, ye, M, p.n, p.k)
             : _i8->gemm_fast(enc, x, xe, p.codes, p.scales, p.gc, p.b, y,
                              ye, M, p.n, p.k)) {
      ++_n_i8;
      if (p.cr) { ++_n_cr; }
      return;
    }
    // Declined (the tier's row floor): widened back to the element type.
    if (p.cr) {
      if (!widen_(enc, p)) { return; }
    } else {
      const std::size_t need = (std::size_t)p.n * p.k * 2;
      if (_wdeq.byte_size() < need) { _wdeq = _mc->make_shared_buffer(need); }
      if (_wdeq.empty() ||
          !_i8->encode_dequant_fast(enc, p.codes, p.scales, p.gc, _wdeq, p.n,
                                    p.k)) {
        return;
      }
    }
    ++_n_dense;
    gemm(enc, x, x_off, _wdeq, p.b.empty() ? nullptr : &p.b, y, y_off, M,
         p.n, p.k);
    return;
  }
  // A STORED int8 weight: widened, then whatever an element-type weight
  // takes -- the accurate tier's per-call quantization, else dense.
  const SharedBuffer* w = &p.w;
  if (p.stored()) {
    if (!widen_(enc, p)) { return; }
    w = &_wdeq;
  }
  // The element-type weight: the int8 requant route where the context takes
  // the shape (K >= 1024, on any tier -- a depth a held layout cannot take
  // stays accurate), else the dense GEMM.
  if (_i8 && _i8->gemm(enc, x, xe, *w, p.b, y, ye, M, p.n, p.k)) {
    ++_n_i8;
    return;
  }
  ++_n_dense;
  gemm(enc, x, x_off, *w, p.b.empty() ? nullptr : &p.b, y, y_off, M, p.n,
       p.k);
}

void
Ops::proj_shared(ComputeEncoder& enc, const SharedBuffer& x,
                 std::size_t x_off, int M, const std::vector<Out>& outs) const
{
  bool shared = _i8 != nullptr && outs.size() > 1 && M > 0;
  const int gc = shared ? outs[0].p->gc : 0;
  const int K = shared ? outs[0].p->k : 0;
  const bool cr = shared && outs[0].p->cr;
  for (const Out& o : outs) {
    shared = shared && o.p->held() && o.p->gc == gc && o.p->k == K &&
             o.p->cr == cr;
  }
  // One quantizer for all: the tier's, or -- ComfyUI's ConvRot codes --
  // the rotated token quantizer they were made for.
  if (shared && (cr ? _i8->quantize_token_convrot(enc, x, x_off / 2, M, K)
                    : _i8->quantize_fast(enc, x, x_off / 2, M, K, gc))) {
    for (const Out& o : outs) {
      if (_i8->gemm_fast_quantized(enc, o.p->codes, o.p->scales, gc, o.p->b,
                                   *o.y, o.y_off / 2, M, o.p->n, K)) {
        ++_n_i8;
        if (cr) { ++_n_cr; }
      } else {
        // Its own route; it quantizes the same rows the same way.
        proj(enc, x, x_off, *o.p, *o.y, o.y_off, M);
      }
    }
    return;
  }
  for (const Out& o : outs) { proj(enc, x, x_off, *o.p, *o.y, o.y_off, M); }
}

void
Ops::tune_pending()
{
  if (_i8) { _i8->tune_pending(_mc); }
}

namespace {

// The cross-attention takes SageAttention's int8 from this many queries:
// its prologue over the 512-token context is a fixed ~0.3 ms a call that
// the int8 product earns back only past ~3.3K rows (MEASURED, M5 Max, a
// keyframe's 30 calls: 2040 rows 13.3 ms dense against 16.4 int8 QK and
// P*V; 8160 rows 51.7 against 38.7). VPIPE_RELAYVSR_XSAGE_MIN_ROWS the A/B,
// read per call (a test lowers it to reach the route at its sizes).
int
cross_sage_min_rows()
{
  const char* e = std::getenv("VPIPE_RELAYVSR_XSAGE_MIN_ROWS");
  return e != nullptr ? std::max(0, std::atoi(e)) : 4096;
}

// The window attention's int8 prologue: its statistics' chunk count, and
// the planes' description the prologue kernels read (LwaPlanes).
constexpr int kStatChunks = 64;

struct LwaPlanes {
  int planes;
  int n;
  int heads;
  int D;
  int head_stride;
  int tok[4];
};

}  // namespace

std::uint64_t
Ops::window_int8_bytes(int planes, int Hq, int Hkv, int n, int head_dim,
                       int i8)
{
  if (i8 <= 0) { return 0; }
  const std::uint64_t N = (std::uint64_t)n, D = (std::uint64_t)head_dim;
  const std::uint64_t P = (std::uint64_t)planes;
  const std::uint64_t C = (std::uint64_t)Hkv * D;
  std::uint64_t b = (std::uint64_t)Hq * N * (D + 4);       // Q codes, scales
  b += P * Hkv * N * (D + 4);                               // K codes, scales
  b += 2 * C * 4 + (std::uint64_t)kStatChunks * C * 8;      // centre, chunks
  if (i8 >= 2) { b += P * Hkv * N * D + C * 4; }            // V, its centre
  return b;
}

std::uint64_t
Ops::window_int8_held() const noexcept
{
  std::uint64_t b = 0;
  for (const SharedBuffer* x : {&_w8_q, &_w8_qs, &_w8_k, &_w8_ks, &_w8_km,
                                &_w8_part, &_w8_v, &_w8_vm, &_w8_vs}) {
    b += x->byte_size();
  }
  return b;
}

std::uint64_t
Ops::i8_scratch_bytes(int tier, bool autotune, int M,
                      const std::vector<ProjShape>& projs)
{
  if (tier < 0 || M <= 0) { return 0; }
  std::uint64_t ka = 0, w = 0, nw = 0;
  bool deep = false;
  for (const ProjShape& p : projs) {
    const bool held = held_group(p.k, tier) != 0;
    const bool acc = !held && p.k >= 1024;
    if (!held && !acc) { continue; }          // the dense GEMM
    ka = std::max<std::uint64_t>(ka, (std::uint64_t)p.k);
    if (!acc) { continue; }
    w = std::max<std::uint64_t>(w, (std::uint64_t)p.n * p.k);
    nw = std::max<std::uint64_t>(nw, (std::uint64_t)p.n);
    deep = deep || p.k >= 9216;
  }
  const std::uint64_t m = (std::uint64_t)M;
  // The activation: int8 codes and their scales (the accurate tier's g64
  // pairs; a coarse group's or a row's are fewer).
  std::uint64_t b = m * ka + 2u * m * (ka / 64) * 2u;
  if (w == 0) { return b; }
  b += w + (w / 64) * 2u;                       // a weight, per call
  if (autotune) {
    b += 2048u * nw * 2u;              // the tuner's rows (its kTuneRows)
    const std::uint64_t budget = (std::uint64_t)512 << 20;
    const std::uint64_t m64 = (m + 63) / 64 * 64;
    const std::uint64_t two = 2u * m64 * nw * 4u;
    b += deep ? budget : std::min(two, budget);    // the split planes
  }
  return b;
}

bool
Ops::load(MetalCompute* mc, bool bf16, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return false;
  };
  _mc = mc;
  _bf16 = bf16;
  const std::string sfx = bf16 ? "_bf16" : "";
  _lib_rv   = mc->load_library("relayvsr" + sfx);
  _lib_gemm = mc->load_library("dense_gemm" + sfx);
  _lib_elt  = mc->load_library("llm_elementwise" + sfx);
  _lib_rms  = mc->load_library("rms_norm" + sfx);
  _lib_lwa  = mc->load_library("local_window_attn");
  _lib_attn = mc->load_library("attn_steel");
  struct Need {
    ComputeFunction* fn;
    const metal_compute::ComputeLibrary* lib;
    const char* name;
  };
  const Need need[] = {
      {&upload, &_lib_rv, "rvsr_upload"},
      {&lq_rows, &_lib_rv, "rvsr_lq_rows"},
      {&rms_silu, &_lib_rv, "rvsr_rms_silu"},
      {&patch_rows, &_lib_rv, "rvsr_patch_rows"},
      {&ln_mod, &_lib_rv, "rvsr_ln_mod"},
      {&ln_affine, &_lib_rv, "rvsr_ln_affine"},
      {&gated_res, &_lib_rv, "rvsr_gated_res"},
      {&rope_wan, &_lib_rv, "rvsr_rope_wan"},
      {&x0, &_lib_rv, "rvsr_x0"},
      {&cond_rows, &_lib_rv, "rvsr_cond_rows"},
      {&rms_embed, &_lib_rv, "rvsr_rms_embed"},
      {&rms_heads64, &_lib_rv, "rvsr_rms_heads64"},
      {&rope3d, &_lib_rv, "rvsr_rope3d"},
      {&swiglu, &_lib_rv, "rvsr_swiglu"},
      {&silu, &_lib_rv, "rvsr_silu"},
      {&frame_out, &_lib_rv, "rvsr_frame_out"},
      {&rms, &_lib_rms, "rms_norm_fast_f16"},
      {&resadd, &_lib_elt, "residual_add_f16"},
      {&gelu, &_lib_elt, "gelu_tanh_ff_f16"},
      {&_fn_gemm, &_lib_gemm, "dense_gemm_t_bm64_f16"},
      {&_fn_gemm_bn64, &_lib_gemm, "dense_gemm_t_bm64bn64_f16"},
      {&_fn_conv_s2_silu, &_lib_gemm, "conv3x3_gemm_s2p1_bn64_silu_f16"},
      {&_fn_conv_s1_silu, &_lib_gemm, "conv3x3_gemm_s1_bn64_silu_f16"},
      {&_fn_conv_s2, &_lib_gemm, "conv3x3_gemm_s2p1_bn128_f16"},
      {&_fn_lwa64, &_lib_lwa,
       bf16 ? "local_window_attn_bd64_bf16" : "local_window_attn_bd64_f16"},
      {&_fn_lwa128, &_lib_lwa,
       bf16 ? "local_window_attn_bd128_bf16" : "local_window_attn_bd128_f16"},
  };
  for (const Need& n : need) {
    *n.fn = n.lib->function(n.name);
    if (!n.fn->valid()) {
      return fail(fmt("RelayVSR: kernel '{}' did not load", n.name)());
    }
  }
  for (int a = 0; a < 4; ++a) {
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (a & 1) != 0).set_bool(201, (a & 2) != 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false);
    _fn_xattn[a] = _lib_attn.function(
        bf16 ? "attn_steel_h_bd128_bf16" : "attn_steel_h_bd128", fc);
    if (!_fn_xattn[a].valid()) {
      return fail("RelayVSR: the steel cross-attention did not load");
    }
  }
  _mma = false;
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_RELAYVSR_NO_MMA2") == nullptr) {
    _lib_mma = mc->load_library("dense_gemm_mma" + sfx);
    _fn_gemm_mma = _lib_mma.function("dense_gemm_mma_t_n128_f16");
    _fn_gemm_mma_wide = _lib_mma.function("dense_gemm_mma_t_n128x256_f16");
    _fn_bias_rows = _lib_elt.function("bias_add_rows_f16");
    _fn_gemm_mma_b = _lib_mma.function("dense_gemm_mma_t_n128_bias_f16");
    _fn_gemm_mma_wide_b =
        _lib_mma.function("dense_gemm_mma_t_n128x256_bias_f16");
    _mma = _fn_gemm_mma.valid() && _fn_gemm_mma_wide.valid() &&
           _fn_bias_rows.valid();
    // The bias in the epilogue, one rounding (see the kernel); without it
    // the route adds the bias in a second pass.
    _mma_bias = _mma && _fn_gemm_mma_b.valid() &&
                _fn_gemm_mma_wide_b.valid() &&
                std::getenv("VPIPE_RELAYVSR_BIAS_PASS") == nullptr;
  }
  // The window attention on the matrix cores, and its int8 prologue.
  _nax_ok = false;
  if (mc->supports_matrix_cores()) {
    _lib_lwan = mc->load_library("local_window_attn_nax");
    const char* et = bf16 ? "bf16" : "f16";
    bool ok = true;
    for (int st = 0; st < 2; ++st) {
      for (int b = 0; b < 2; ++b) {
        for (int m = 0; m < 3; ++m) {
          metal_compute::FunctionConstants fc;
          fc.set_bool(0, m >= 1).set_bool(1, m == 2);
          _fn_lwan[st][b][m] = _lib_lwan.function(
              fmt("local_window_attn_nax{}_bd{}_{}", st == 1 ? "_tg" : "",
                  b == 0 ? 64 : 128, et)(),
              fc);
          ok = ok && _fn_lwan[st][b][m].valid();
        }
      }
    }
    _fn_lwa_qtok = _lib_lwan.function(fmt("lwa_quant_token_{}", et)());
    _fn_lwa_stats = _lib_lwan.function(fmt("lwa_channel_range_{}", et)());
    _fn_lwa_reduce = _lib_lwan.function("lwa_channel_reduce");
    _fn_lwa_qv = _lib_lwan.function(fmt("lwa_quant_v_{}", et)());
    _nax_ok = ok && _fn_lwa_qtok.valid() && _fn_lwa_stats.valid() &&
              _fn_lwa_reduce.valid() && _fn_lwa_qv.valid();
  }
  _nax_attn = _nax_ok &&
              std::getenv("VPIPE_RELAYVSR_NO_NAX_ATTN") == nullptr;
  _lwa_staged = std::getenv("VPIPE_RELAYVSR_LWA_DIRECT") == nullptr;
  // The cross-attention's matrix-core flash kernel, and SageAttention's
  // int8 twins of it where the host carries their kernels.
  if (_nax_ok) {
    _lib_attn_nax = mc->load_library("attn_steel_nax");
    std::string serr;
    _sage_x = MetalSageAttention::load(mc, bf16, &serr);
    const bool pv8 = _sage_x && _sage_x->pv8_available();
    bool ok = _lib_attn_nax.valid();
    for (int m = 0; m < 3; ++m) {
      if ((m >= 1 && !_sage_x) || (m == 2 && !pv8)) { break; }
      for (int a = 0; a < 4; ++a) {
        metal_compute::FunctionConstants fc;
        fc.set_bool(200, (a & 1) != 0).set_bool(201, (a & 2) != 0)
            .set_bool(300, false).set_bool(301, false)
            .set_bool(302, false).set_bool(sage::kQkInt8Constant, m >= 1);
        if (pv8) { MetalSageAttention::pv8_constants(fc, m == 2); }
        _fn_xnax[m][a] = _lib_attn_nax.function(
            bf16 ? "attn_steel_nax_h_bd128_bf16" : "attn_steel_nax_h_bd128",
            fc);
        ok = ok && (m > 0 || _fn_xnax[m][a].valid());
      }
    }
    _xnax = ok && _nax_attn;
  }
  return true;
}

void
Ops::gemm(ComputeEncoder& enc, const SharedBuffer& x, std::size_t x_off,
          const SharedBuffer& w, const SharedBuffer* b, const SharedBuffer& y,
          std::size_t y_off, int M, int N, int K) const
{
  if (M <= 0) { return; }
  const bool has_b = b != nullptr && !b->empty();
  if (_mma && M >= 64 && N >= 64 && K % 32 == 0) {
    const bool wide = mma_use_wide_tile(N, K);
    const bool fused = has_b && _mma_bias;
    const ComputeFunction& fn =
        fused ? (wide ? _fn_gemm_mma_wide_b : _fn_gemm_mma_b)
              : (wide ? _fn_gemm_mma_wide : _fn_gemm_mma);
    const int RN = wide ? 256 : 128;
    const int band = mma_row_band(N, K);
    for (int r0 = 0; r0 < M; r0 += band) {
      const int rows = std::min(band, M - r0);
      enc.set_function(fn);
      enc.set_buffer(0, x, x_off + (std::size_t)r0 * K * 2);
      enc.set_buffer(1, w);
      enc.set_buffer(2, fused ? *b : w);
      enc.set_buffer(3, y, y_off + (std::size_t)r0 * N * 2);
      enc.set_constant(4, K); enc.set_constant(5, N);
      enc.set_constant(6, rows); enc.set_constant(7, fused ? 1 : 0);
      enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                    (unsigned)((rows + 127) / 128), 1}, {256, 1, 1});
    }
    if (has_b && !fused) {
      enc.set_function(_fn_bias_rows);
      enc.set_buffer(0, y, y_off); enc.set_buffer(1, *b);
      enc.set_constant(2, N);
      enc.set_constant(3, (unsigned)((std::size_t)M * N));
      enc.dispatch({(unsigned)N, (unsigned)M, 1}, {32, 8, 1});
    }
    return;
  }
  // The ALU tiles: BM 64 x BN 64 where N is wide, BN 32 otherwise.
  const bool bn64 = N >= 512;
  enc.set_function(bn64 ? _fn_gemm_bn64 : _fn_gemm);
  enc.set_buffer(0, x, x_off); enc.set_buffer(1, w);
  enc.set_buffer(2, has_b ? *b : w);
  enc.set_buffer(3, y, y_off);
  enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
  enc.set_constant(7, has_b ? 1 : 0);
  if (bn64) {
    enc.dispatch({(unsigned)(((N + 63) / 64) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
  } else {
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                  (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
  }
}

void
Ops::conv3x3(ComputeEncoder& enc, const SharedBuffer& in,
             const SharedBuffer& w, const SharedBuffer& b,
             const SharedBuffer& out, int H, int W, int Cin, int Cout,
             int stride, bool silu) const
{
  const int OH = stride == 2 ? (H + 1) / 2 : H;
  const int OW = stride == 2 ? (W + 1) / 2 : W;
  const ComputeFunction& fn =
      stride == 2 ? (silu ? _fn_conv_s2_silu : _fn_conv_s2)
                  : _fn_conv_s1_silu;
  const int BN = (stride == 2 && !silu) ? 128 : 64;
  enc.set_function(fn);
  enc.set_buffer(0, in); enc.set_buffer(1, w); enc.set_buffer(2, b);
  enc.set_buffer(3, out);
  enc.set_constant(4, H); enc.set_constant(5, W);
  enc.set_constant(6, Cin); enc.set_constant(7, Cout);
  enc.set_constant(8, OH); enc.set_constant(9, OW);
  enc.set_constant(10, b.empty() ? 0 : 1);
  enc.set_buffer(11, out);                     // unread at these epilogues
  enc.dispatch({(unsigned)(((Cout + BN - 1) / BN) * 32),
                (unsigned)(((OH * OW + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
Ops::window_attention(ComputeEncoder& enc, const SharedBuffer& q,
                      std::size_t q_off, int q_tok,
                      const std::vector<WindowPlane>& planes,
                      const SharedBuffer& out, std::size_t o_off, int o_tok,
                      int H, int W, int Hq, int Hkv, int head_dim,
                      int i8) const
{
  LocalWindowAttnParams p{};
  p.H = H; p.W = W; p.Hq = Hq; p.Hkv = Hkv;
  p.planes = (int)std::min<std::size_t>(planes.size(), 4);
  p.tiles_x = (W + 7) / 8;
  p.q_tok = q_tok; p.q_head = head_dim;
  p.o_tok = o_tok; p.o_head = head_dim;
  p.k_head = head_dim; p.v_head = head_dim;
  p.scale = 1.0f / std::sqrt((float)head_dim);
  for (int i = 0; i < p.planes; ++i) {
    const WindowPlane& pl = planes[(std::size_t)i];
    p.k_tok[i] = pl.k_tok; p.v_tok[i] = pl.v_tok;
    p.win_h[i] = pl.win_h; p.win_w[i] = pl.win_w;
    p.shifted[i] = pl.shifted ? 1 : 0;
  }
  // Unused slots are bound to plane 0 (never read).
  auto bind_planes = [&](int base, bool v) {
    for (int i = 0; i < 4; ++i) {
      const WindowPlane& pl = planes[(std::size_t)std::min(i, p.planes - 1)];
      if (v) {
        enc.set_buffer(base + i, *pl.v, pl.v_off);
      } else {
        enc.set_buffer(base + i, *pl.k, pl.k_off);
      }
    }
  };
  if (!_nax_attn) {
    enc.set_function(head_dim == 64 ? _fn_lwa64 : _fn_lwa128);
    enc.set_buffer(0, q, q_off);
    bind_planes(1, false);
    bind_planes(5, true);
    enc.set_buffer(9, out, o_off);
    enc.set_constant_bytes(10, &p, sizeof(p));
    const int tiles = p.tiles_x * ((H + 3) / 4);
    enc.dispatch({32u * (unsigned)tiles, 4u * (unsigned)Hq, 1}, {32, 4, 1});
    return;
  }
  const int mode = std::clamp(i8, 0, 2);
  const int n = H * W, D = head_dim, P = p.planes;
  if (mode > 0 && _w8_prologue) {
    // Every plane's K and V at the most this model attends, once: grown
    // plane by plane, each step held the old copy beside the new one.
    auto grow = [&](SharedBuffer& b, std::uint64_t bytes) {
      if (b.byte_size() < bytes) {
        b = SharedBuffer{};
        b = _mc->make_shared_buffer(std::max<std::uint64_t>(bytes, 16));
        ++_w8_allocs;
      }
    };
    const std::uint64_t N = (std::uint64_t)n, C = (std::uint64_t)Hkv * D;
    const std::uint64_t PC = (std::uint64_t)std::max(P, _w8_planes);
    grow(_w8_q, (std::uint64_t)Hq * N * D);
    grow(_w8_qs, (std::uint64_t)Hq * N * 4);
    grow(_w8_k, PC * Hkv * N * D);
    grow(_w8_ks, PC * Hkv * N * 4);
    grow(_w8_km, C * 4);
    grow(_w8_vs, C * 4);
    grow(_w8_part, (std::uint64_t)kStatChunks * C * 8);
    if (mode == 2) {
      grow(_w8_v, PC * Hkv * N * D);
      grow(_w8_vm, C * 4);
    }
    // Q per token.
    LwaPlanes g{};
    g.planes = 1; g.n = n; g.heads = Hq; g.D = D; g.head_stride = D;
    g.tok[0] = q_tok;
    const int zero = 0, one = 1;
    enc.set_function(_fn_lwa_qtok);
    for (int i = 0; i < 4; ++i) { enc.set_buffer(i, q, q_off); }
    enc.set_buffer(4, _w8_km);
    enc.set_buffer(5, _w8_q); enc.set_buffer(6, _w8_qs);
    enc.set_constant_bytes(7, &g, sizeof(g));
    enc.set_constant(8, zero);
    enc.dispatch({32u * (unsigned)((n + 3) / 4), 4u * (unsigned)Hq, 1},
                 {32, 4, 1});
    // Each channel's centre (its midrange over every plane) and int8
    // step, for K or V.
    LwaPlanes kg{};
    kg.planes = P; kg.n = n; kg.heads = Hkv; kg.D = D; kg.head_stride = D;
    for (int i = 0; i < P; ++i) {
      kg.tok[i] = planes[(std::size_t)i].k_tok;
    }
    LwaPlanes vg = kg;
    for (int i = 0; i < P; ++i) {
      vg.tok[i] = planes[(std::size_t)i].v_tok;
    }
    const int Ci = Hkv * D, chunks = kStatChunks;
    auto range = [&](bool v, const LwaPlanes& pg, const SharedBuffer& centre,
                     const SharedBuffer& step) {
      enc.set_function(_fn_lwa_stats);
      bind_planes(0, v);
      enc.set_buffer(4, _w8_part);
      enc.set_constant_bytes(5, &pg, sizeof(pg));
      enc.dispatch({32u * (unsigned)((Ci + 31) / 32), 8u * (unsigned)chunks,
                    1}, {32, 8, 1});
      enc.set_function(_fn_lwa_reduce);
      enc.set_buffer(0, _w8_part); enc.set_buffer(1, centre);
      enc.set_buffer(2, step);
      enc.set_constant(3, Ci); enc.set_constant(4, chunks);
      enc.dispatch({(unsigned)Ci, 1, 1}, {64, 1, 1});
    };
    // K: its centre over every plane, then each plane per token over it.
    range(false, kg, _w8_km, _w8_vs);
    enc.set_function(_fn_lwa_qtok);
    bind_planes(0, false);
    enc.set_buffer(4, _w8_km);
    enc.set_buffer(5, _w8_k); enc.set_buffer(6, _w8_ks);
    enc.set_constant_bytes(7, &kg, sizeof(kg));
    enc.set_constant(8, one);
    enc.dispatch({32u * (unsigned)((n + 3) / 4), 4u * (unsigned)Hkv,
                  (unsigned)P}, {32, 4, 1});
    if (mode == 2) {
      // V: its centre and step over every plane, every plane at those.
      range(true, vg, _w8_vm, _w8_vs);
      enc.set_function(_fn_lwa_qv);
      bind_planes(0, true);
      enc.set_buffer(4, _w8_vm); enc.set_buffer(5, _w8_vs);
      enc.set_buffer(6, _w8_v);
      enc.set_constant_bytes(7, &vg, sizeof(vg));
      enc.dispatch({(unsigned)D * (unsigned)((n + 7) / 8), (unsigned)Hkv,
                    (unsigned)P}, {(unsigned)D, 1, 1});
    }
  }
  enc.set_function(
      _fn_lwan[_lwa_staged ? 1 : 0][head_dim == 64 ? 0 : 1][mode]);
  enc.set_buffer(0, q, q_off);
  bind_planes(1, false);
  bind_planes(5, true);
  enc.set_buffer(9, out, o_off);
  enc.set_constant_bytes(10, &p, sizeof(p));
  if (mode > 0) {
    enc.set_buffer(11, _w8_q); enc.set_buffer(12, _w8_qs);
    enc.set_buffer(13, _w8_k); enc.set_buffer(14, _w8_ks);
  }
  if (mode == 2) {
    enc.set_buffer(15, _w8_v); enc.set_buffer(16, _w8_vs);
    enc.set_buffer(17, _w8_vm);
  }
  const int tiles = p.tiles_x * ((H + 7) / 8);
  enc.dispatch({32u * (unsigned)tiles, 4u * (unsigned)Hq, 1}, {32, 4, 1});
}

std::uint64_t
Ops::cross_int8_bytes(int heads, int n, int kn, int i8)
{
  if (i8 <= 0 || n < cross_sage_min_rows()) { return 0; }
  return MetalSageAttention::scratch_bytes(
      heads, heads, n, kn, 128, MetalSageAttention::nax_query_block(),
      MetalSageAttention::nax_key_block(), i8 >= 2);
}

void
Ops::cross_attention(ComputeEncoder& enc, const SharedBuffer& q, int n,
                     const SharedBuffer& k, const SharedBuffer& v, int kn,
                     const SharedBuffer& out, int heads, int i8) const
{
  constexpr int HD = 128;
  const bool nax = _xnax;
  const int BQ = nax ? MetalSageAttention::nax_query_block() : 32;
  const int BK = nax ? MetalSageAttention::nax_key_block() : 16;
  SteelAttnParams prm{};
  SteelAttnParams* pp = &prm;
  const std::int64_t ld = (std::int64_t)heads * HD;
  pp->B = 1; pp->H = heads; pp->D = HD;
  pp->qL = n; pp->kL = kn;
  pp->gqa_factor = 1;
  pp->scale = 1.0f / std::sqrt((float)HD);
  pp->NQ = (n + BQ - 1) / BQ;
  pp->NK = (kn + BK - 1) / BK;
  pp->NQ_aligned = n / BQ;
  pp->NK_aligned = kn / BK;
  pp->qL_rem = n - pp->NQ_aligned * BQ;
  pp->kL_rem = kn - pp->NK_aligned * BK;
  pp->qL_off = 0;
  // TOKEN-MAJOR everywhere: a head is HD columns of a row, a row is the
  // whole width -- the projections' output as it lies.
  for (std::int64_t* s : {pp->Q_strides, pp->K_strides, pp->V_strides,
                          pp->O_strides}) {
    s[0] = 0; s[1] = HD; s[2] = ld;
  }
  pp->Q_strides[0] = pp->O_strides[0] = (std::int64_t)n * ld;
  pp->K_strides[0] = pp->V_strides[0] = (std::int64_t)kn * ld;
  // The params go by value (setBytes): every block encodes its own, and
  // a shared buffer rewritten between encodes would hand the GPU the last.
  const int al = (n % BQ == 0 ? 1 : 0) | (kn % BK == 0 ? 2 : 0);
  // SageAttention's prologue: Q and K (and V, for the P*V) quantized
  // from where they lie, token-major.
  int mode = 0;
  if (nax && i8 > 0 && _sage_x && n >= cross_sage_min_rows()) {
    mode = std::min(i8, _fn_xnax[2][al].valid() ? 2 : 1);
    if (!_fn_xnax[mode][al].valid()) { mode = 0; }
  }
  if (mode > 0) {
    sage::Config cfg;
    cfg.enabled = true;
    cfg.pv8 = mode == 2;
    const MetalSageAttention::Operand qo{&q, 0, (int)ld, HD};
    const MetalSageAttention::Operand ko{&k, 0, (int)ld, HD};
    const MetalSageAttention::Operand vo{&v, 0, (int)ld, HD};
    std::string serr;
    if (!_sage_x->prepare(enc, qo, ko, heads, heads, n, kn, HD, BQ, BK,
                          cfg, &serr, &vo) ||
        _sage_x->pv8_for(cfg) != _sage_x->pv8_ready()) {
      // A prologue that made none (or not the P*V's): dense.
      mode = 0;
    }
  }
  enc.set_function(nax ? _fn_xnax[mode][al] : _fn_xattn[al]);
  enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
  enc.set_buffer(3, out);
  enc.set_constant_bytes(4, pp, sizeof(SteelAttnParams));
  if (mode > 0) { _sage_x->bind(enc); }
  enc.dispatch({32u * (unsigned)pp->NQ, 4u * (unsigned)heads, 1},
               {32, 4, 1});
}

}  // namespace relayvsr
}  // namespace genai
}  // namespace vpipe
