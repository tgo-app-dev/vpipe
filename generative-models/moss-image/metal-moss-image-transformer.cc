#include "generative-models/moss-image/metal-moss-image-transformer.h"
#include "common/memory-owner.h"

#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/signed-affine.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/metal-sol-attention.h"
#include "generative-models/shared/lora-names.h"
#include "generative-models/shared/mma-tile.h"
#include "generative-models/shared/tp-cut.h"
#include "generative-models/shared/tp-overlap.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <utility>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::CommandStream;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

namespace fs = std::filesystem;

// The steel flash-attention parameter block, as attn_steel.metal reads
// it. Mirrored per family because each one builds its own strides.
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

inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t r = (u >> 16) & 1u;
  u += 0x7fffu + r;
  return (std::uint16_t)(u >> 16);
}

inline float
bf16_round_(float f)
{
  const std::uint32_t u = (std::uint32_t)f32_to_bf16_(f) << 16;
  float r;
  std::memcpy(&r, &u, 4);
  return r;
}

bool
narrow_to_bf16_(const void* src, const std::string& dtype, std::size_t n,
                std::uint16_t* dst)
{
  if (dtype == "BF16") {
    std::memcpy(dst, src, n * 2);
    return true;
  }
  if (dtype == "F32") {
    const auto* s = static_cast<const float*>(src);
    for (std::size_t i = 0; i < n; ++i) { dst[i] = f32_to_bf16_(s[i]); }
    return true;
  }
  if (dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(src);
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = f32_to_bf16_((float)s[i]);
    }
    return true;
  }
  return false;
}

SharedBuffer
to_bf16_(const MetalLlamaWeights& wts, MetalCompute* mc,
         const std::string& nm)
{
  const auto* info = wts.info(nm);
  if (info == nullptr || info->shape.empty()) { return {}; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { return {}; }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  if (!narrow_to_bf16_(raw.contents(), info->dtype, n,
                       static_cast<std::uint16_t*>(out.contents()))) {
    return {};
  }
  return out;
}

// Derived-cache namespace. A WeightSet is shared by everything reading
// one checkpoint, so a key names the TRANSFORM, not just the tensor.
constexpr const char* kKey = "moss-image-dit/";

// The modulation's six chunks, in the reference's `unbind(1)` order.
enum Mod : int {
  kShiftA = 0, kScaleA, kGateA, kShiftF, kScaleF, kGateF, kModChunks
};

}  // namespace

// ---- the reference's sampler arithmetic ------------------------------

namespace moss_image {

namespace {

// torch.linspace(start, end, steps) in f32, its two-sided formula: the
// first half counts up from start, the second down from end -- each
// node ONE fused multiply-add, which is what ATen's scalar kernel
// compiles to on arm64 (MEASURED: every node of seven grids against
// torch 2.14 on this box, where the twice-rounded form missed 311 of
// them). An x86 torch rounds twice and lands an ulp away; the snap to
// the training grid absorbs most of that.
std::vector<float>
linspace_f32_(float start, float end, int n)
{
  std::vector<float> out((std::size_t)std::max(n, 0));
  if (n <= 0) { return out; }
  if (n == 1) { out[0] = start; return out; }
  const float step = (end - start) / (float)(n - 1);
  const int halfway = n / 2;
  for (int i = 0; i < n; ++i) {
    out[(std::size_t)i] =
        i < halfway ? std::fma(step, (float)i, start)
                    : std::fma(-step, (float)(n - i - 1), end);
  }
  return out;
}

// The reference's `grid(count, shift)`: shift * v / (1 + (shift-1) * v)
// over linspace(1, shift / (total + shift - 1), count), in f32 with the
// python scalars cast the way torch's tensor-scalar arithmetic casts
// them.
std::vector<float>
grid_(int count, double shift, int total)
{
#pragma clang fp contract(off)
  const double lower = shift / ((double)total + shift - 1.0);
  std::vector<float> v = linspace_f32_(1.0f, (float)lower, count);
  const float s = (float)shift;
  const float sm1 = (float)(shift - 1.0);
  for (float& x : v) { x = (s * x) / (1.0f + sm1 * x); }
  return v;
}

}  // namespace

std::vector<float>
sigmas(int steps, double shift, double training_shift, int train_steps)
{
#pragma clang fp contract(off)
  if (steps < 1 || shift <= 0.0 || training_shift <= 0.0 ||
      train_steps < 2) {
    return {};
  }
  const std::vector<float> want = grid_(steps, shift, train_steps);
  std::vector<float> train = grid_(train_steps, training_shift, train_steps);
  std::reverse(train.begin(), train.end());           // ascending
  const int T = (int)train.size();
  std::vector<float> out;
  out.reserve((std::size_t)steps + 1);
  for (float s : want) {
    // searchsorted (left): the first node >= s.
    int up = (int)(std::lower_bound(train.begin(), train.end(), s) -
                   train.begin());
    up = std::min(up, T - 1);
    const int lo = std::max(up - 1, 0);
    const float dl = std::fabs(s - train[(std::size_t)lo]);
    const float du = std::fabs(train[(std::size_t)up] - s);
    const float pick = train[(std::size_t)(dl <= du ? lo : up)];
    // unique_consecutive
    if (out.empty() || out.back() != pick) { out.push_back(pick); }
  }
  out.push_back(0.0f);   // FlowMatchEulerDiscreteScheduler's terminal
  return out;
}

void
rope_tables(int grid_h, int grid_w, int text_rows, int head_dim, float base,
            bool round_bf16, std::vector<float>* cos_out,
            std::vector<float>* sin_out)
{
#pragma clang fp contract(off)
  const int bands = head_dim / 8;
  const int half = head_dim / 2;              // 4 axes x bands
  const int img = grid_h * grid_w;
  const int rows = img + text_rows;
  cos_out->assign((std::size_t)rows * half, 0.0f);
  sin_out->assign((std::size_t)rows * half, 0.0f);
  // 1 / base ** (arange(bands) / bands), in f32 as the reference forms
  // it on the CPU.
  std::vector<float> band((std::size_t)bands);
  for (int j = 0; j < bands; ++j) {
    const float e = (float)j / (float)bands;
    band[(std::size_t)j] = 1.0f / std::pow(base, e);
  }
  const float ch = (float)((double)(grid_h - 1) / 2.0);
  const float cw = (float)((double)(grid_w - 1) / 2.0);
  auto put = [&](int row, const float coord[4]) {
    for (int a = 0; a < 4; ++a) {
      for (int j = 0; j < bands; ++j) {
        const float ph = coord[a] * band[(std::size_t)j];
        float c = (float)std::cos((double)ph);
        float s = (float)std::sin((double)ph);
        if (round_bf16) { c = bf16_round_(c); s = bf16_round_(s); }
        const std::size_t k = (std::size_t)row * half + a * bands + j;
        (*cos_out)[k] = c;
        (*sin_out)[k] = s;
      }
    }
  };
  for (int y = 0; y < grid_h; ++y) {
    for (int x = 0; x < grid_w; ++x) {
      const float coord[4] = {0.0f, (float)text_rows, (float)y - ch,
                              (float)x - cw};
      put(y * grid_w + x, coord);
    }
  }
  for (int i = 0; i < text_rows; ++i) {
    const float coord[4] = {0.0f, (float)i, 0.0f, 0.0f};
    put(img + i, coord);
  }
}

}  // namespace moss_image

MetalMossImageTransformer::~MetalMossImageTransformer()
{
  // A rank's link goes first: its worker may still hold an exchange.
  end_tp_generation();
  // GIVE THE POOL BACK: only unwire_from_pool decrements its counter.
  if (_wire.on()) {
    wire_fixed_(false);
    for (Block& b : _blocks) { wire_block_(b, false); }
  }
}

std::string
MetalMossImageTransformer::prefix_(int L) const
{
  return "layers." + std::to_string(L) + ".";
}

// ---- residency + the wired pool -------------------------------------

std::size_t
MetalMossImageTransformer::wire_block_(Block& b, bool on)
{
  std::size_t changed = 0;
  bool stop = false;
  auto one = [&](SharedBuffer& p) {
    if (stop) { return; }
    const std::size_t n = _wire.wire_one(_mc, p, on);
    // STOP AT THE FIRST REFUSAL: the pool is full.
    if (on && n == 0 && p.byte_size() > 0 && !p.is_wired() &&
        _wire.refused(_mc, p)) {
      stop = true;
      return;
    }
    changed += n;
  };
  auto qw = [&](QWeight& w) {
    one(w.w); one(w.codes); one(w.scales); one(w.qbias); one(w.cscales);
  };
  QWeight* q[] = {&b.qw, &b.kw, &b.vw, &b.ow, &b.ug, &b.dn};
  for (QWeight* w : q) { qw(*w); }
  one(b.n1); one(b.n2); one(b.nq); one(b.nk);
  return changed;
}

std::size_t
MetalMossImageTransformer::wire_fixed_(bool on)
{
  if (!_ws) { return 0; }
  std::size_t changed = 0;
  _ws->for_each_weight([&](SharedBuffer& b) {
    changed += _wire.wire_one(_mc, b, on);
  });
  return changed;
}

std::size_t
MetalMossImageTransformer::wire_retry_slack_() const
{
  if (_wire_block_hint > 0) { return _wire_block_hint; }
  if (_resid.count() > 0) {
    return _resid.bytes() / (std::size_t)_resid.count();
  }
  return 0;
}

std::size_t
MetalMossImageTransformer::evict_tail_block_()
{
  // From the END: the next forward starts at block 0, so the deepest
  // resident block is the one whose re-read is furthest away.
  for (int i = (int)_blocks.size() - 1; i >= 0; --i) {
    Block& b = _blocks[(std::size_t)i];
    const std::size_t n = block_bytes_(b);
    if (n == 0) { continue; }
    _wire.note_unwired(wire_block_(b, false));
    b = Block{};
    return n;
  }
  return 0;
}

void
MetalMossImageTransformer::resident_pages_(std::size_t* examined,
                                           std::size_t* incore,
                                           std::size_t* paged_out) const
{
  *examined = 0;
  *incore = 0;
  if (paged_out != nullptr) { *paged_out = 0; }
  auto add = [&](const SharedBuffer& p) {
    if (p.byte_size() == 0) { return; }
    if (p.is_wired()) { return; }
    if (!GenerativeModelManager::pool_wirable(p)) { return; }
    const auto r = p.page_residency(64);
    if (!r.valid) { return; }
    *examined += r.examined;
    *incore += r.incore;
    if (paged_out != nullptr) { *paged_out += r.paged_out; }
  };
  auto qw = [&](const QWeight& w) {
    add(w.w); add(w.codes); add(w.scales); add(w.qbias);
  };
  for (const Block& b : _blocks) {
    if (block_bytes_(b) == 0) { continue; }
    qw(b.qw); qw(b.kw); qw(b.vw); qw(b.ow); qw(b.ug); qw(b.dn);
    add(b.n1); add(b.n2); add(b.nq); add(b.nk);
  }
}

// ---- config ---------------------------------------------------------

bool
MetalMossImageTransformer::Config::read_dims(const std::string& dir,
                                             Config* out)
{
  if (out == nullptr) { return false; }
  std::ifstream in(fs::path(dir) / "config.json");
  if (!in) { return false; }
  FlexData fd;
  try {
    fd = FlexData::from_json(in);
  } catch (...) {
    return false;
  }
  if (!fd.is_object()) { return false; }
  auto o = fd.as_object();
  if (!o.contains("_class_name")) { return false; }
  if (std::string(o.at("_class_name").as_string("")) !=
      "MOSSImageTransformer") {
    return false;
  }
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  auto getf = [&](const char* k, float d) {
    return o.contains(k) ? (float)o.at(k).as_real(d) : d;
  };
  out->hidden = geti("dim", out->hidden);
  out->ffn_inner = geti("ffn_dim", out->ffn_inner);
  out->n_heads = geti("num_heads", out->n_heads);
  out->n_layers = geti("num_layers", out->n_layers);
  out->text_dim = geti("text_dim", out->text_dim);
  out->freq_dim = geti("freq_dim", out->freq_dim);
  out->latent_channels = geti("latent_channels", out->latent_channels);
  out->rope_base = getf("rope_base", out->rope_base);
  out->head_dim = out->n_heads > 0 ? out->hidden / out->n_heads : 0;
  // The flash entries here are bd128, and the reference's own asserts:
  // dim divisible by heads, head_dim by 8, an even sinusoid.
  return out->hidden > 0 && out->n_heads > 0 &&
         out->n_heads * out->head_dim == out->hidden &&
         out->head_dim == 128 && out->n_layers > 0 &&
         (out->freq_dim % 2) == 0 && out->ffn_inner > 0 &&
         out->latent_channels > 0 && out->text_dim > 0;
}

// ---- weights --------------------------------------------------------

SharedBuffer
MetalMossImageTransformer::to_elt_(WeightSet& ws, const std::string& name,
                                   Retain r)
{
  auto build = [&]() { return to_bf16_(ws.src(), _mc, name); };
  if (r == Retain::Streamed) { return ws.stream_derived(build); }
  return ws.derived(std::string(kKey) + "elt|" + name, build);
}

MetalMossImageTransformer::QWeight
MetalMossImageTransformer::load_qw_(WeightSet& ws, const std::string& name,
                                    Retain r, bool sign, bool fast)
{
  QWeight qw;
  const MetalLlamaWeights& wts = ws.src();
  const auto* si = wts.info(name + ".scales");
  const auto* ci = wts.info(name + ".weight");
  if (_quant_bits > 0 && si != nullptr && ci != nullptr &&
      si->shape.size() == 2 && ci->shape.size() == 2) {
    const long gcols = ci->shape[1];              // K*bits/32
    const long scols = si->shape[1];              // K/group
    const long K = scols * (long)_quant_group;
    const int bits = K > 0 ? (int)(gcols * 32 / K) : 0;
    qw.bits = (bits == 8) ? 8 : 4;
    // THE FAST TIER: the codes dequantized, group-rotated and requantized
    // at the coarse group (or per row on the token tier). A w8 pack's
    // once -- in the builder of the entry a retained block keeps (one
    // buffer, codes then scales), in place in a streamed read with the
    // scales in a buffer the model owns. A w4 pack's per GEMM, into the
    // context's scratch (its layout would be twice its size), so it is
    // held as stored.
    const int fgc0 =
        (fast && _fast && _quant_group == 64) ? _i8->fast_group((int)K) : 0;
    const int fgc =
        fgc0 > 0 && _i8->fast_converts(qw.bits, fgc0) ? fgc0 : 0;
    if (fgc > 0) {
      const std::string cn = name + ".weight", sn = name + ".scales",
                        bn = name + ".biases";
      const int N = (int)ci->shape[0], KK = (int)K, qb = qw.bits;
      const std::size_t cb = (std::size_t)N * KK;
      const std::size_t sb = I8GemmContext::fast_scale_bytes(N, KK, fgc);
      QWeight f;
      f.bits = qb; f.gc = fgc; f.n = N; f.k = KK;
      if (qb == 4 && (r == Retain::Streamed || !_cfg.i8_hold_w4)) {
        // A W4 PACK IS HELD AS STORED: its layout is twice its size,
        // so each GEMM converts it into the context's scratch right
        // before reading it (fast_operand_).
        f.codes = r == Retain::Streamed
                      ? ws.stream_tensor(cn, _mc, WeightSet::Residency::Copied)
                      : ws.tensor(cn, _mc, WeightSet::Residency::Copied);
        f.scales = to_elt_(ws, sn, r);
        f.qbias = to_elt_(ws, bn, r);
        f.quantized = !f.codes.empty() && !f.scales.empty() &&
                      !f.qbias.empty();
        f.pc = true;
      } else if (r == Retain::Streamed) {
        f.codes = ws.stream_tensor(cn, _mc, WeightSet::Residency::Copied);
        f.scales = to_elt_(ws, sn, r);
        f.qbias = to_elt_(ws, bn, r);
        f.cscales = _mc->make_shared_buffer(sb);
        f.quantized = !f.codes.empty() && !f.scales.empty() &&
                      !f.qbias.empty() && !f.cscales.empty() && fast_(f);
      } else {
        SharedBuffer both = ws.derived(
            std::string(kKey) + _i8->layout_tag(fgc) + "|" + cn,
            [&]() -> SharedBuffer {
              SharedBuffer c = ws.read(cn, _mc,
                                       WeightSet::Residency::Copied);
              SharedBuffer s = to_bf16_(ws.src(), _mc, sn);
              SharedBuffer b = to_bf16_(ws.src(), _mc, bn);
              SharedBuffer out = _mc->make_shared_buffer(cb + sb);
              if (c.empty() || s.empty() || b.empty() || out.empty()) {
                return {};
              }
              CommandStream st = _mc->make_command_stream();
              bool ok = false;
              {
                ComputeEncoder e = st.begin_compute();
                ok = _i8->encode_to_fast(e, c, s, b, out.subview(0, cb),
                                         out.subview(cb, sb), fgc, N, KK,
                                         qb);
              }
              return ok && st.commit().wait_ok(nullptr) ? std::move(out)
                                                        : SharedBuffer{};
            });
        if (!both.empty()) {
          f.codes = both.subview(0, cb);
          f.cscales = both.subview(cb, sb);
          f.quantized = !f.codes.empty() && !f.cscales.empty();
        }
      }
      if (f.quantized) { return f; }
    }
    // THE SIGNED LAYOUT, converted where the bytes are first the model's
    // own: in the builder of a retained block's entries, in place in a
    // streamed read.
    if (sign && _sgn && _i8 && _i8->signed_available(qw.bits)) {
      const std::string cn = name + ".weight", bn = name + ".biases";
      const auto cp = WeightSet::Residency::Copied;
      qw.scales = to_elt_(ws, name + ".scales", r);
      if (r == Retain::Streamed) {
        qw.codes = ws.stream_tensor(cn, _mc, cp);
        qw.qbias = ws.stream_derived(
            [&]() { return to_bf16_(ws.src(), _mc, bn); });
        qw.sgn = !qw.codes.empty() && !qw.scales.empty() &&
                 genai::to_signed_affine(qw.codes, qw.scales, qw.qbias,
                                         qw.bits, /*bf16=*/true);
      } else {
        const int qb = qw.bits;
        qw.codes = ws.derived(std::string(kKey) + "sgn|" + cn, [&]() {
          SharedBuffer c = ws.read(cn, _mc, cp);
          return !c.empty() && genai::to_signed_codes(c, qb)
                     ? std::move(c) : SharedBuffer{};
        });
        if (!qw.scales.empty()) {
          qw.qbias = ws.derived(std::string(kKey) + "sgn|" + bn, [&]() {
            SharedBuffer b = to_bf16_(ws.src(), _mc, bn);
            return !b.empty() && genai::to_signed_zero_points(
                                     qw.scales, b, qb, /*bf16=*/true)
                       ? std::move(b) : SharedBuffer{};
          });
        }
        qw.sgn = !qw.codes.empty() && !qw.qbias.empty();
      }
      if (qw.sgn) {
        qw.quantized = true;
        return qw;
      }
      qw.codes = {}; qw.scales = {}; qw.qbias = {};
      qw.sgn = false;
    }
    qw.codes = r == Retain::Streamed
                   ? ws.stream_tensor(name + ".weight", _mc,
                                      WeightSet::Residency::Copied)
                   : ws.tensor(name + ".weight", _mc,
                               WeightSet::Residency::Copied);
    qw.scales = to_elt_(ws, name + ".scales", r);
    qw.qbias = to_elt_(ws, name + ".biases", r);
    if (!qw.codes.empty() && !qw.scales.empty() && !qw.qbias.empty()) {
      qw.quantized = true;
      return qw;
    }
    qw.codes = {}; qw.scales = {}; qw.qbias = {};
  }
  // THE FAST TIER, a dense checkpoint: quantized once into the entry a
  // retained block keeps. A STREAMED one keeps the checkpoint's bf16 as
  // its refill target and quantizes it after every refill
  // (QWeight::ds), so the slot runs the tier a preloaded block runs and
  // a promotion keeps what the entry keeps. A pack's dense tensor stays
  // dense: the pack left it out on purpose.
  if (fast && _fast && _quant_bits == 0 && ci != nullptr &&
      ci->shape.size() == 2) {
    const int N = (int)ci->shape[0], KK = (int)ci->shape[1];
    const int fgc = _i8->fast_group(KK);
    if (fgc > 0 && r == Retain::Streamed) {
      QWeight f;
      f.bits = 8; f.gc = fgc; f.n = N; f.k = KK; f.ds = true;
      f.w = to_elt_(ws, name + ".weight", r);
      f.codes = _mc->make_shared_buffer((std::size_t)N * KK);
      f.cscales = _mc->make_shared_buffer(
          I8GemmContext::fast_scale_bytes(N, KK, fgc));
      f.quantized = !f.w.empty() && !f.codes.empty() && !f.cscales.empty();
      if (f.quantized && fast_(f)) { return f; }
    } else if (fgc > 0) {
      const std::size_t cb = (std::size_t)N * KK;
      const std::size_t sb = I8GemmContext::fast_scale_bytes(N, KK, fgc);
      const std::string wn = name + ".weight";
      SharedBuffer both = ws.derived(
          std::string(kKey) + _i8->layout_tag(fgc) + "bf|" + wn,
          [&]() -> SharedBuffer {
            SharedBuffer w = to_bf16_(ws.src(), _mc, wn);
            SharedBuffer out = _mc->make_shared_buffer(cb + sb);
            if (w.empty() || out.empty()) { return {}; }
            CommandStream st = _mc->make_command_stream();
            bool ok = false;
            {
              ComputeEncoder e = st.begin_compute();
              ok = _i8->encode_quant_fast(e, w, out.subview(0, cb),
                                          out.subview(cb, sb), fgc, N, KK);
            }
            return ok && st.commit().wait_ok(nullptr) ? std::move(out)
                                                      : SharedBuffer{};
          });
      if (!both.empty()) {
        QWeight f;
        f.bits = 8; f.gc = fgc; f.n = N; f.k = KK;
        f.codes = both.subview(0, cb);
        f.cscales = both.subview(cb, sb);
        f.quantized = !f.codes.empty() && !f.cscales.empty();
        if (f.quantized) { return f; }
      }
    }
  }
  qw.w = to_elt_(ws, name + ".weight", r);
  return qw;
}

bool
MetalMossImageTransformer::load_block_(WeightSet& ws, int L, Block& b,
                                       Retain r)
{
  // Its blocks, kept or streamed (common/memory-owner.h).
  MemoryUseScope blocks_use(MemoryUse::kBlocks);
  const std::string p = prefix_(L);
  // Every projection of a main block runs at the joint sequence's rows,
  // so all six take the signed and fast layouts when they are on.
  b.qw = load_qw_(ws, p + "proj.q", r, true, true);
  b.kw = load_qw_(ws, p + "proj.k", r, true, true);
  b.vw = load_qw_(ws, p + "proj.v", r, true, true);
  b.ow = load_qw_(ws, p + "proj.o", r, true, true);
  b.ug = load_qw_(ws, p + "proj.ffn_up_gate", r, true, true);
  b.dn = load_qw_(ws, p + "proj.ffn_down", r, true, true);
  b.n1 = to_elt_(ws, p + "norm1.weight", r);
  b.n2 = to_elt_(ws, p + "norm2.weight", r);
  b.nq = to_elt_(ws, p + "q_norm.weight", r);
  b.nk = to_elt_(ws, p + "k_norm.weight", r);
  return !b.qw.empty() && !b.kw.empty() && !b.vw.empty() && !b.ow.empty() &&
         !b.ug.empty() && !b.dn.empty() && !b.n1.empty() && !b.n2.empty() &&
         !b.nq.empty() && !b.nk.empty();
}

bool
MetalMossImageTransformer::load_text_block_(WeightSet& ws, int i,
                                            TextBlock& b)
{
  const Retain r = Retain::Cached;
  const std::string p = "text_embedder.blocks." + std::to_string(i) + ".";
  b.qkv = load_qw_(ws, p + "qkv", r);
  b.qkv_b = to_elt_(ws, p + "qkv.bias", r);
  b.proj = load_qw_(ws, p + "proj", r);
  b.proj_b = to_elt_(ws, p + "proj.bias", r);
  // ffn_up is the HIDDEN half and ffn_gate the gate: _situ(ffn_up(y),
  // ffn_gate(y)).
  b.up = load_qw_(ws, p + "ffn_up", r);
  b.up_b = to_elt_(ws, p + "ffn_up.bias", r);
  b.gate = load_qw_(ws, p + "ffn_gate", r);
  b.gate_b = to_elt_(ws, p + "ffn_gate.bias", r);
  b.down = load_qw_(ws, p + "ffn_down", r);
  b.down_b = to_elt_(ws, p + "ffn_down.bias", r);
  b.n1 = to_elt_(ws, p + "norm1.weight", r);
  b.n2 = to_elt_(ws, p + "norm2.weight", r);
  b.nq = to_elt_(ws, p + "q_norm.weight", r);
  b.nk = to_elt_(ws, p + "k_norm.weight", r);
  return !b.qkv.empty() && !b.qkv_b.empty() && !b.proj.empty() &&
         !b.proj_b.empty() && !b.up.empty() && !b.up_b.empty() &&
         !b.gate.empty() && !b.gate_b.empty() && !b.down.empty() &&
         !b.down_b.empty() && !b.n1.empty() &&
         !b.n2.empty() && !b.nq.empty() && !b.nk.empty();
}

// A rank's share of one projection. CUT BY ITS ROWS -- proj.q / k / v
// and ffn_up_gate, most of a block -- a dense bf16 weight or an MLX
// affine pack has only the share's rows READ, straight into the share
// (tp_read_rows); the rest of the tensor never leaves the disk. CUT BY
// ITS COLUMNS -- proj.o, ffn_down -- every row is needed: the whole
// weight is read UNCACHED (Retain::Streamed, plain: no signed or tier
// layout) and the share copied out of it, as is any weight the row read
// does not serve (an F32 checkpoint; VPIPE_TP_WHOLE_READ, the A/B). An
// affine pack's columns are cut in whole groups. Then the share is laid
// out as a whole model lays out its own block projections: on the int8
// fast / token tier, a w8 share converted, a w4 share held as stored
// and converted per GEMM (unless Config::i8_hold_w4), and a DENSE share
// quantized -- from a dense checkpoint only, as load_qw_ quantizes (a
// pack's dense tensors stay dense) -- each at the group the share's own
// columns take, a width none divides staying on the accurate tier; off
// those tiers an affine share takes the SIGNED layout where the model's
// projections do (_sgn: per byte and per group, so a share converts as
// its whole weight would).
MetalMossImageTransformer::QWeight
MetalMossImageTransformer::tp_qw_(
    WeightSet& ws, const std::string& name, int N,
    const std::vector<std::pair<int, int>>& rows, int k0, int k1,
    std::string* err)
{
  auto fail = [&](const std::string& why) {
    if (err != nullptr) { *err = name + ": " + why; }
    return QWeight{};
  };
  int n = 0;
  for (const auto& [a, b] : rows) { n += b - a; }
  QWeight l;
  l.n = n;
  bool read = false;
  if (k1 == 0 && std::getenv("VPIPE_TP_WHOLE_READ") == nullptr) {
    const MetalLlamaWeights& src = ws.src();
    // A pack's scales and biases as the model holds them, bf16: read in
    // the dtype stored, an F16 pack's widened in place (one width).
    auto rows_bf16 = [&](const std::string& nm) -> SharedBuffer {
      const auto* ti = src.info(nm);
      if (ti == nullptr || (ti->dtype != "BF16" && ti->dtype != "F16")) {
        return {};
      }
      SharedBuffer b =
          tp_read_rows(_mc, src, nm, ti->dtype.c_str(), N, rows);
      if (b.empty() || ti->dtype == "BF16") { return b; }
      auto* p = static_cast<std::uint16_t*>(b.contents());
      for (std::size_t i = 0; i < b.byte_size() / 2; ++i) {
        _Float16 h;
        std::memcpy(&h, &p[i], 2);
        p[i] = f32_to_bf16_((float)h);
      }
      return b;
    };
    const auto* ci = src.info(name + ".weight");
    const auto* si = src.info(name + ".scales");
    if (_quant_bits > 0 && ci != nullptr && si != nullptr &&
        ci->shape.size() == 2 && si->shape.size() == 2) {
      const long K = si->shape[1] * (long)_quant_group;
      const long bits = K > 0 ? ci->shape[1] * 32 / K : 0;
      if (bits == 4 || bits == 8) {
        l.codes = tp_read_rows(_mc, src, name + ".weight", "U32", N, rows);
        l.scales = rows_bf16(name + ".scales");
        l.qbias = rows_bf16(name + ".biases");
        if (!l.codes.empty() && !l.scales.empty() && !l.qbias.empty()) {
          l.quantized = true;
          l.bits = (int)bits;
          l.k = (int)K;
          read = true;
        } else {
          l.codes = {}; l.scales = {}; l.qbias = {};
        }
      }
    } else if (si == nullptr && ci != nullptr && ci->shape.size() == 2) {
      l.w = tp_read_rows(_mc, src, name + ".weight", "BF16", N, rows);
      if (!l.w.empty()) {
        l.k = (int)ci->shape[1];
        read = true;
      }
    }
  }
  if (!read) {
    QWeight w = load_qw_(ws, name, Retain::Streamed, /*sign=*/false,
                         /*fast=*/false);
    if (w.empty()) { return fail("not in the checkpoint"); }
    if (w.gc > 0 || w.sgn) {
      return fail("a split DiT reads dense or affine weights only");
    }
    if (w.quantized) {
      const int g = _quant_group;
      if (k1 > 0 && (k0 % g != 0 || k1 % g != 0)) {
        return fail(fmt("its share's columns [{}, {}) are not whole "
                        "groups of {}", k0, k1, g)());
      }
      l.quantized = true;
      l.bits = w.bits;
      // Codes: `32 / bits` columns a word; scales and biases: one a group.
      l.codes = tp_cut(_mc, w.codes, N, rows, (std::size_t)k0 * w.bits / 8,
                       (std::size_t)k1 * w.bits / 8);
      l.scales = tp_cut(_mc, w.scales, N, rows, (std::size_t)k0 / g * 2,
                        (std::size_t)k1 / g * 2);
      l.qbias = tp_cut(_mc, w.qbias, N, rows, (std::size_t)k0 / g * 2,
                       (std::size_t)k1 / g * 2);
      if (l.codes.empty() || l.scales.empty() || l.qbias.empty()) {
        return fail("its share could not be cut");
      }
      l.k = k1 > 0 ? k1 - k0 : (int)(w.codes.byte_size() * 8 /
                                     (std::size_t)N / (std::size_t)w.bits);
    } else {
      l.w = tp_cut(_mc, w.w, N, rows, (std::size_t)k0 * 2,
                   (std::size_t)k1 * 2);
      if (l.w.empty()) { return fail("its share could not be cut"); }
      l.k = k1 > 0 ? k1 - k0
                   : (int)(w.w.byte_size() / 2 / (std::size_t)N);
    }
  }
  // Off the int8 fast / token tier: the signed layout, where the model's
  // projections take it.
  auto plain = [&]() -> QWeight {
    if (!sign_(l)) {
      return fail("its share did not take the signed layout");
    }
    return std::move(l);
  };
  if (!_fast || !_i8) { return plain(); }
  // A pack's dense tensor stays dense, as the whole model keeps it.
  if (!l.quantized && _quant_bits != 0) { return plain(); }
  // The tier's layout, within the share's own columns.
  const int fgc = _i8->fast_group(l.k);
  if (fgc <= 0) { return plain(); }
  const std::size_t cb = (std::size_t)l.n * l.k;
  const std::size_t sb = I8GemmContext::fast_scale_bytes(l.n, l.k, fgc);
  if (l.quantized) {
    if (_quant_group != 64 || !_i8->fast_converts(l.bits, fgc)) {
      return plain();
    }
    if (l.bits == 4 && !_cfg.i8_hold_w4) {
      // A W4 SHARE IS HELD AS STORED, converted before each GEMM
      // (fast_operand_), as a whole model's w4 block is.
      l.gc = fgc;
      l.pc = true;
      return l;
    }
    QWeight f;
    f.bits = l.bits; f.gc = fgc; f.n = l.n; f.k = l.k;
    f.codes = _mc->make_shared_buffer(cb);
    f.cscales = _mc->make_shared_buffer(sb);
    if (f.codes.empty() || f.cscales.empty()) { return plain(); }
    CommandStream st = _mc->make_command_stream();
    bool ok = false;
    {
      ComputeEncoder e = st.begin_compute();
      ok = _i8->encode_to_fast(e, l.codes, l.scales, l.qbias, f.codes,
                               f.cscales, fgc, l.n, l.k, l.bits);
    }
    if (!ok || !st.commit().wait_ok(nullptr)) { return plain(); }
    f.quantized = true;
    return f;
  }
  QWeight f;
  f.bits = 8; f.gc = fgc; f.n = l.n; f.k = l.k;
  f.codes = _mc->make_shared_buffer(cb);
  f.cscales = _mc->make_shared_buffer(sb);
  if (f.codes.empty() || f.cscales.empty()) { return l; }
  CommandStream st = _mc->make_command_stream();
  bool ok = false;
  {
    ComputeEncoder e = st.begin_compute();
    ok = _i8->encode_quant_fast(e, l.w, f.codes, f.cscales, fgc, l.n, l.k);
  }
  if (!ok || !st.commit().wait_ok(nullptr)) { return l; }
  f.quantized = true;
  return f;
}

// A RANK's block. THE FUSED ffn_up_gate is [hidden | gate] by its rows --
// hidden FIRST -- and SITU pairs hidden row j with gate row F + j, so a
// share of feed-forward rows [f0, f1) takes [f0, f1) of the hidden half
// AND [F + f0, F + f1) of the gate half: its [2f, H] share is again
// [hidden | gate], which situ_split reads at the share's width.
bool
MetalMossImageTransformer::load_tp_block_(WeightSet& ws, int L, Block& b,
                                          std::string* err)
{
  MemoryUseScope blocks_use(MemoryUse::kBlocks);
  const TpShare& t = _cfg.tp;
  const int H = _cfg.hidden, Hd = _cfg.head_dim, F = _cfg.ffn_inner;
  const int i0 = t.head0 * Hd, i1 = t.head1 * Hd;
  const std::vector<std::pair<int, int>> heads = {{i0, i1}};
  const std::vector<std::pair<int, int>> all = {{0, H}};
  const std::vector<std::pair<int, int>> ug = {{t.ffn0, t.ffn1},
                                               {F + t.ffn0, F + t.ffn1}};
  const std::string p = prefix_(L);
  struct One {
    QWeight* w;
    const char* leaf;
    int N;
    const std::vector<std::pair<int, int>>* rows;
    int k0, k1;
  };
  const One all_of[] = {
      {&b.qw, "proj.q", H, &heads, 0, 0},
      {&b.kw, "proj.k", H, &heads, 0, 0},
      {&b.vw, "proj.v", H, &heads, 0, 0},
      {&b.ow, "proj.o", H, &all, i0, i1},
      {&b.ug, "proj.ffn_up_gate", 2 * F, &ug, 0, 0},
      {&b.dn, "proj.ffn_down", H, &all, t.ffn0, t.ffn1}};
  for (const One& o : all_of) {
    *o.w = tp_qw_(ws, p + o.leaf, o.N, *o.rows, o.k0, o.k1, err);
    if (o.w->empty()) { return false; }
  }
  // The norms -- the block's two over the hidden width and the q / k
  // RMSNorms, a head's own [head_dim] -- are whole on every rank.
  b.n1 = to_elt_(ws, p + "norm1.weight", Retain::Cached);
  b.n2 = to_elt_(ws, p + "norm2.weight", Retain::Cached);
  b.nq = to_elt_(ws, p + "q_norm.weight", Retain::Cached);
  b.nk = to_elt_(ws, p + "k_norm.weight", Retain::Cached);
  if (b.n1.empty() || b.n2.empty() || b.nq.empty() || b.nk.empty()) {
    if (err != nullptr) { *err = p + ": a norm is missing"; }
    return false;
  }
  return true;
}

bool
MetalMossImageTransformer::tp_buffer_(SharedBuffer& b, std::size_t bytes)
{
  if (b.byte_size() >= bytes) { return true; }
  if (!b.empty() && _tp_pipe != nullptr) { _tp_pipe->forget(b); }
  b = SharedBuffer{};
  b = _mc->make_shared_buffer(bytes);
  return !b.empty();
}

bool
MetalMossImageTransformer::begin_tp_generation(std::shared_ptr<TpChannel> ch,
                                               std::string* err)
{
  auto fail = [&](const std::string& why) {
    if (err != nullptr) { *err = "tensor parallel: " + why; }
    return false;
  };
  end_tp_generation();
  if (!_cfg.tp.active()) { return fail("this model is not a rank"); }
  if (!ch) { return fail("no channel to the other ranks"); }
  // Four lanes: a block's bands alternate between two for the attention's
  // partials and two for the feed-forward's (forward()'s rank tail).
  auto pipe = std::make_unique<TpOverlap>(_mc, ch, 4, TpDType::kBF16);
  if (!pipe->valid()) {
    return fail("no events or thread for the exchanges");
  }
  _tp_ch = std::move(ch);
  _tp_pipe = std::move(pipe);
  _tp_seq = 0;
  // The stop the stage set before the picture, which the channel polls
  // while it waits on the others.
  try {
    _tp_ch->set_stop(_stream_stop);
  } catch (...) {
  }
  return true;
}

void
MetalMossImageTransformer::end_tp_generation()
{
  if (_tp_pipe == nullptr && !_tp_ch) { return; }
  // An exchange still waiting on a peer would hold the worker the reset
  // below joins: the channel told to stop waiting first.
  if (_tp_pipe != nullptr && _tp_ch &&
      (_tp_pipe->pending() || _tp_pipe->failed())) {
    try {
      _tp_ch->set_stop([]() { return true; });
    } catch (...) {
    }
  }
  _tp_pipe.reset();
  // Every buffer the channel was handed forgotten before it goes (a
  // transport may keep a device's registration of each).
  if (_tp_ch) {
    for (const SharedBuffer* b :
         {&_tp_part[0], &_tp_part[1], &_tp_part[2], &_tp_part[3],
          &_tp_recv[0], &_tp_recv[1], &_tp_recv[2], &_tp_recv[3]}) {
      if (b->empty()) { continue; }
      try {
        _tp_ch->forget(b->contents(), b->byte_size());
      } catch (...) {
      }
    }
    try {
      _tp_ch->set_stop({});
    } catch (...) {
    }
  }
  _tp_ch.reset();
  _tp_seq = 0;
  // THE BUFFERS GO WITH THE PICTURE. The plan books them with the
  // forward's activations alone (forward_scratch_bytes), and the
  // transport's registration of them went with the forget above; kept,
  // they sat unbooked beside the decode and in a DiT kept warm.
  for (SharedBuffer& b : _tp_part) { b = SharedBuffer{}; }
  for (SharedBuffer& b : _tp_recv) { b = SharedBuffer{}; }
}

bool
MetalMossImageTransformer::load_fixed_(WeightSet& ws)
{
  const Retain r = Retain::Cached;
  _x_in = load_qw_(ws, "video_embedder", r);
  _x_in_b = to_elt_(ws, "video_embedder.bias", r);
  _t_in = load_qw_(ws, "text_embedder.proj", r);
  _t_in_b = to_elt_(ws, "text_embedder.proj.bias", r);
  _time_1 = load_qw_(ws, "time_embedding.0", r);
  _time_1b = to_elt_(ws, "time_embedding.0.bias", r);
  _time_2 = load_qw_(ws, "time_embedding.2", r);
  _time_2b = to_elt_(ws, "time_embedding.2.bias", r);
  _mod = load_qw_(ws, "shared_block_modulation", r);
  _mod_b = to_elt_(ws, "shared_block_modulation.bias", r);
  _head_mod = load_qw_(ws, "video_head_modulation", r);
  _head_mod_b = to_elt_(ws, "video_head_modulation.bias", r);
  _final_norm = to_elt_(ws, "final_norm_video.weight", r);
  _final_lin = load_qw_(ws, "final_linear_video", r);
  return !_x_in.empty() && !_x_in_b.empty() && !_t_in.empty() &&
         !_t_in_b.empty() && !_time_1.empty() && !_time_1b.empty() &&
         !_time_2.empty() && !_time_2b.empty() && !_mod.empty() &&
         !_mod_b.empty() && !_head_mod.empty() && !_head_mod_b.empty() &&
         !_final_norm.empty() && !_final_lin.empty();
}

std::size_t
MetalMossImageTransformer::qw_bytes_(const QWeight& w)
{
  if (w.quantized) {
    return w.codes.byte_size() + w.scales.byte_size() + w.qbias.byte_size() +
           w.cscales.byte_size() + (w.ds ? w.w.byte_size() : 0);
  }
  return w.w.byte_size();
}

std::size_t
MetalMossImageTransformer::kept_block_bytes_(const Block& b) const
{
  auto kq = [&](const QWeight& w) -> std::size_t {
    if (w.pc && _cfg.i8_hold_w4 && _i8 != nullptr) {
      return (std::size_t)w.n * w.k +
             I8GemmContext::fast_scale_bytes(w.n, w.k, w.gc);
    }
    if (w.quantized && w.gc > 0 && !w.pc) {
      return w.codes.byte_size() + w.cscales.byte_size();
    }
    return qw_bytes_(w);
  };
  return kq(b.qw) + kq(b.kw) + kq(b.vw) + kq(b.ow) + kq(b.ug) + kq(b.dn) +
         b.n1.byte_size() + b.n2.byte_size() + b.nq.byte_size() +
         b.nk.byte_size();
}

std::size_t
MetalMossImageTransformer::block_bytes_(const Block& b)
{
  return qw_bytes_(b.qw) + qw_bytes_(b.kw) + qw_bytes_(b.vw) +
         qw_bytes_(b.ow) + qw_bytes_(b.ug) + qw_bytes_(b.dn) +
         b.n1.byte_size() + b.n2.byte_size() + b.nq.byte_size() +
         b.nk.byte_size();
}

std::size_t
MetalMossImageTransformer::text_block_bytes_(const TextBlock& b)
{
  return qw_bytes_(b.qkv) + b.qkv_b.byte_size() + qw_bytes_(b.proj) +
         b.proj_b.byte_size() + qw_bytes_(b.up) + b.up_b.byte_size() +
         qw_bytes_(b.gate) + b.gate_b.byte_size() + qw_bytes_(b.down) +
         b.down_b.byte_size() + b.n1.byte_size() +
         b.n2.byte_size() + b.nq.byte_size() + b.nk.byte_size();
}

void
MetalMossImageTransformer::each_block_tensor_(
    int L, Block& b, const BlockSlots<Block>::TensorFn& fn) const
{
  const std::string p = prefix_(L);
  auto qw = [&](QWeight& w, const char* nm) {
    if (w.ds) {
      // The dense source only: the layout is its product (post_refill).
      fn(p + nm + ".weight", w.w, Placement::kBf16);
    } else if (w.quantized) {
      fn(p + nm + ".weight", w.codes, Placement::kRaw);
      fn(p + nm + ".scales", w.scales, Placement::kBf16);
      fn(p + nm + ".biases", w.qbias, Placement::kBf16);
    } else {
      fn(p + nm + ".weight", w.w, Placement::kBf16);
    }
  };
  qw(b.qw, "proj.q");
  qw(b.kw, "proj.k");
  qw(b.vw, "proj.v");
  qw(b.ow, "proj.o");
  qw(b.ug, "proj.ffn_up_gate");
  qw(b.dn, "proj.ffn_down");
  fn(p + "norm1.weight", b.n1, Placement::kBf16);
  fn(p + "norm2.weight", b.n2, Placement::kBf16);
  fn(p + "q_norm.weight", b.nq, Placement::kBf16);
  fn(p + "k_norm.weight", b.nk, Placement::kBf16);
}

namespace {

bool
clone_buf_(MetalCompute* mc, const SharedBuffer& s, SharedBuffer& d, bool copy)
{
  if (s.empty()) { d = {}; return true; }
  d = mc->make_shared_buffer(s.byte_size());
  if (d.empty()) { return false; }
  if (copy) { std::memcpy(d.contents(), s.contents(), s.byte_size()); }
  return true;
}

}  // namespace

bool
MetalMossImageTransformer::clone_block_(const Block& src, Block& dst,
                                        bool copy) const
{
  // Its blocks, kept or streamed (common/memory-owner.h).
  MemoryUseScope blocks_use(MemoryUse::kBlocks);
  bool ok = true;
  // A PROMOTION keeps what a preloaded block keeps (see
  // kept_block_bytes_): a w4 pack whose layout is held (Config::
  // i8_hold_w4) is converted into it here, all of a block's in one
  // command buffer.
  CommandStream cs;
  ComputeEncoder ce;
  auto cq = [&](const QWeight& s, QWeight& d) {
    const bool hold = copy && s.pc && _cfg.i8_hold_w4 && _i8 != nullptr;
    d.quantized = s.quantized;
    d.bits = s.bits;
    d.sgn = s.sgn;
    d.gc = s.gc;
    d.pc = s.pc && !hold;
    d.n = s.n;
    d.k = s.k;
    // A dense source is the slot's refill target, and only a second
    // slot (copy false) keeps one.
    d.ds = s.ds && !copy;
    static const SharedBuffer kNone;
    if (hold) {
      d.codes = _mc->make_shared_buffer((std::size_t)s.n * s.k);
      d.cscales = _mc->make_shared_buffer(
          I8GemmContext::fast_scale_bytes(s.n, s.k, s.gc));
      if (!cs.valid()) {
        cs = _mc->make_command_stream();
        ce = cs.begin_compute();
      }
      ok = !d.codes.empty() && !d.cscales.empty() && ce.valid() &&
           _i8->encode_to_fast(ce, s.codes, s.scales, s.qbias, d.codes,
                               d.cscales, s.gc, s.n, s.k, s.bits) &&
           ok;
      return;
    }
    ok = clone_buf_(_mc, s.cscales, d.cscales, copy) && ok;
    ok = clone_buf_(_mc, s.ds && copy ? kNone : s.w, d.w, copy) && ok;
    ok = clone_buf_(_mc, s.codes, d.codes, copy) && ok;
    // A weight already in its int8 layout (gc > 0, not converted per
    // call) never reads the pack's scales and biases again, so a
    // PROMOTED copy leaves them behind -- ~6% of a w8 block, wired. A
    // second slot (copy false) keeps them: its next refill converts.
    const bool dead = copy && s.gc > 0 && !s.pc;
    ok = clone_buf_(_mc, dead ? kNone : s.scales, d.scales, copy) && ok;
    ok = clone_buf_(_mc, dead ? kNone : s.qbias, d.qbias, copy) && ok;
  };
  cq(src.qw, dst.qw); cq(src.kw, dst.kw); cq(src.vw, dst.vw);
  cq(src.ow, dst.ow); cq(src.ug, dst.ug); cq(src.dn, dst.dn);
  ok = clone_buf_(_mc, src.n1, dst.n1, copy) && ok;
  ok = clone_buf_(_mc, src.n2, dst.n2, copy) && ok;
  ok = clone_buf_(_mc, src.nq, dst.nq, copy) && ok;
  ok = clone_buf_(_mc, src.nk, dst.nk, copy) && ok;
  if (cs.valid()) {
    ce.end();
    ok = cs.commit().wait_ok(nullptr) && ok;
  }
  return ok;
}

bool
MetalMossImageTransformer::keep_block_(Block& b) const
{
  // Its blocks, kept or streamed (common/memory-owner.h).
  MemoryUseScope blocks_use(MemoryUse::kBlocks);
  const bool hold = _cfg.i8_hold_w4 && _i8 != nullptr;
  CommandStream cs;
  bool ok = true;
  std::vector<std::pair<QWeight*, QWeight>> held;
  {
    ComputeEncoder e;
    for (QWeight* w : {&b.qw, &b.kw, &b.vw, &b.ow, &b.ug, &b.dn}) {
      if (w->ds) {
        w->w = SharedBuffer{};
        w->ds = false;
      }
      if (w->quantized && w->gc > 0 && !w->pc) {
        w->scales = SharedBuffer{};
        w->qbias = SharedBuffer{};
      }
      if (!(w->quantized && w->pc && hold)) { continue; }
      // The w4 pack into the layout the preload holds: the call and the
      // operands its derived entry is built from, so the same bytes.
      QWeight f;
      f.quantized = true; f.bits = w->bits; f.gc = w->gc;
      f.n = w->n; f.k = w->k;
      f.codes = _mc->make_shared_buffer((std::size_t)w->n * w->k);
      f.cscales = _mc->make_shared_buffer(
          I8GemmContext::fast_scale_bytes(w->n, w->k, w->gc));
      if (f.codes.empty() || f.cscales.empty()) { ok = false; continue; }
      if (!cs.valid()) {
        cs = _mc->make_command_stream();
        e = cs.begin_compute();
      }
      if (!e.valid() ||
          !_i8->encode_to_fast(e, w->codes, w->scales, w->qbias, f.codes,
                               f.cscales, w->gc, w->n, w->k, w->bits)) {
        ok = false;
        continue;
      }
      held.emplace_back(w, std::move(f));
    }
  }
  // Swapped in only once the GPU has written them: until then the pack
  // is what the block runs.
  if (cs.valid() && !cs.commit().wait_ok(nullptr)) { return false; }
  for (auto& [w, f] : held) { *w = std::move(f); }
  return ok;
}

SharedBuffer
MetalMossImageTransformer::rebuild_one_(const std::string& nm, Placement how)
{
  // Its blocks, kept or streamed (common/memory-owner.h).
  MemoryUseScope blocks_use(MemoryUse::kBlocks);
  if (_ws == nullptr) { return {}; }
  if (how == Placement::kRaw) {
    return _ws->stream_tensor(nm, _mc, WeightSet::Residency::Copied);
  }
  return _ws->stream_derived([&]() { return to_bf16_(_ws->src(), _mc, nm); });
}

bool
MetalMossImageTransformer::fill_narrow_(const std::string& nm,
                                        const SharedBuffer& dst)
{
  // An F32 checkpoint's bytes are twice the width of the buffers this
  // model reads, so a raw read has nowhere to put them: one scratch for
  // the run, narrowed into the slot.
  if (_ws == nullptr || dst.empty()) { return false; }
  const auto* info = _ws->src().info(nm);
  if (info == nullptr || info->shape.empty()) { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  if (dst.byte_size() < n * 2) { return false; }
  const std::size_t need = (std::size_t)info->nbytes;
  if (_narrow_scratch.size() < need) { _narrow_scratch.resize(need); }
  if (!_ws->stream_into(nm, _narrow_scratch.data(), need)) { return false; }
  return narrow_to_bf16_(_narrow_scratch.data(), info->dtype, n,
                         static_cast<std::uint16_t*>(dst.contents()));
}

bool
MetalMossImageTransformer::sign_(QWeight& w) const
{
  if (!w.quantized || w.gc > 0 || !_sgn || !_i8 ||
      !_i8->signed_available(w.bits)) {
    return true;
  }
  if (!genai::to_signed_affine(w.codes, w.scales, w.qbias, w.bits,
                               /*bf16=*/true)) {
    return false;
  }
  w.sgn = true;
  return true;
}

// A slot built for a block of the OTHER width -- a mixed pack holds whole
// blocks at w8 and the rest at w4 -- is rebuilt rather than refilled
// (BlockSlots::Ops::same_layout): the refill would place the codes and
// keep the slot's width, fast-tier scales and signed flag. Every main
// block projection is compared; a dense checkpoint has nothing to.
bool
MetalMossImageTransformer::same_layout_(int L, const Block& b) const
{
  if (_quant_bits == 0 || _ws == nullptr) { return true; }
  const std::string p = prefix_(L);
  const std::pair<const QWeight*, const char*> all[] = {
      {&b.qw, "proj.q"}, {&b.kw, "proj.k"}, {&b.vw, "proj.v"},
      {&b.ow, "proj.o"}, {&b.ug, "proj.ffn_up_gate"},
      {&b.dn, "proj.ffn_down"}};
  for (const auto& [w, nm] : all) {
    if (stored_affine_bits(*_ws, p + nm, _quant_group) !=
        (w->quantized ? w->bits : 0)) {
      return false;
    }
  }
  return true;
}

int
MetalMossImageTransformer::fast_weights(int bits) const
{
  int n = 0;
  for (const Block& b : _blocks) {
    for (const QWeight* w : {&b.qw, &b.kw, &b.vw, &b.ow, &b.ug, &b.dn}) {
      n += w->quantized && w->gc > 0 && w->bits == bits;
    }
  }
  return n;
}

bool
MetalMossImageTransformer::encode_fast_(ComputeEncoder& e, QWeight& w) const
{
  // A w4 pack kept as stored converts per GEMM (fast_operand_), not here.
  if (w.gc <= 0 || w.pc) { return true; }
  if (!_i8 || w.codes.empty() || w.cscales.empty()) { return false; }
  // A dense source: quantized out of the bf16 the refill landed, by the
  // call and at the group the preloaded entry is built with.
  if (w.ds) {
    return !w.w.empty() &&
           _i8->encode_quant_fast(e, w.w, w.codes, w.cscales, w.gc, w.n,
                                  w.k);
  }
  if (w.scales.empty() || w.qbias.empty()) { return false; }
  // The slot's codes are the model's own: converted in place, before
  // anything reads the block.
  return _i8->encode_to_fast(e, w.codes, w.scales, w.qbias, w.codes,
                             w.cscales, w.gc, w.n, w.k);
}

bool
MetalMossImageTransformer::fast_(QWeight& w) const
{
  if (w.gc <= 0 || w.pc) { return true; }
  CommandStream st = _mc->make_command_stream();
  bool ok = false;
  {
    ComputeEncoder e = st.begin_compute();
    ok = encode_fast_(e, w);
  }
  return ok && st.commit().wait_ok(nullptr);
}

bool
MetalMossImageTransformer::fast_operand_(ComputeEncoder& enc, const QWeight& w,
                                         const SharedBuffer** codes,
                                         const SharedBuffer** scales) const
{
  if (!w.pc) {
    *codes = &w.codes;
    *scales = &w.cscales;
    return true;
  }
  if (!_i8 || !_i8->encode_to_fast_scratch(enc, w.codes, w.scales, w.qbias,
                                           w.gc, w.n, w.k)) {
    return false;
  }
  *codes = &_i8->fast_scratch_codes();
  *scales = &_i8->fast_scratch_scales();
  return true;
}

bool
MetalMossImageTransformer::fast_dense_(ComputeEncoder& enc, const QWeight& w,
                                       const SharedBuffer& out, int N,
                                       int K) const
{
  const SharedBuffer* c = nullptr;
  const SharedBuffer* cs = nullptr;
  return fast_operand_(enc, w, &c, &cs) &&
         _i8->encode_dequant_fast(enc, *c, *cs, w.gc, out, N, K);
}

const MetalMossImageTransformer::QWeight*
MetalMossImageTransformer::signed_dense_(ComputeEncoder& enc,
                                         const QWeight& w, QWeight& view,
                                         int N, int K) const
{
  if (!w.quantized || (!w.sgn && w.gc <= 0)) { return &w; }
  const std::size_t need = (std::size_t)N * (std::size_t)K * 2;
  if (_w_deq.empty() || _w_deq.byte_size() < need) {
    _w_deq = _mc->make_shared_buffer(need);
    if (_w_deq.empty()) { return nullptr; }
  }
  const bool ok =
      _i8 != nullptr &&
      (w.gc > 0 ? fast_dense_(enc, w, _w_deq, N, K)
                : _i8->encode_dequant_signed(enc, w.codes, w.scales,
                                             w.qbias, w.bits, _w_deq, N, K));
  if (!ok) { return nullptr; }
  view = QWeight{};
  view.w = _w_deq.subview(0, need);
  if (view.w.empty()) { return nullptr; }
  return &view;
}

void
MetalMossImageTransformer::configure_slots_()
{
  BlockSlots<Block>::Ops ops;
  ops.each = [this](int L, Block& b, const BlockSlots<Block>::TensorFn& fn) {
    each_block_tensor_(L, b, fn);
  };
  ops.rebuild_one = [this](const std::string& nm, Placement how) {
    return rebuild_one_(nm, how);
  };
  ops.fill_unservable = [this](const std::string& nm, const SharedBuffer& d) {
    return fill_narrow_(nm, d);
  };
  ops.build = [this](int L, Block& b) {
    if (_ws == nullptr) { return false; }
    if (!_cfg.tp.active()) {
      return load_block_(*_ws, L, b, Retain::Streamed);
    }
    // A RANK's block is its share, cut at every read (load() turns the
    // slots off for it).
    std::string terr;
    if (load_tp_block_(*_ws, L, b, &terr)) { return true; }
    if (_mc->session() != nullptr) {
      _mc->session()->warn(fmt("MetalMossImageTransformer: "
                               "tensor_parallel: block {}: {}", L, terr));
    }
    return false;
  };
  ops.clone = [this](const Block& s, Block& d, bool copy) {
    return clone_block_(s, d, copy);
  };
  ops.bytes = [](const Block& b) { return block_bytes_(b); };
  ops.empty = [](const Block& b) { return b.qw.empty(); };
  // Every quantized projection just landed RAW and converts here, on
  // the reader thread, under the GPU's work.
  ops.post_refill = [this](int, Block& b) {
    QWeight* all[] = {&b.qw, &b.kw, &b.vw, &b.ow, &b.ug, &b.dn};
    // The fast layouts in ONE command buffer for the block, on the
    // reader's own stream; the signed layout is the CPU's.
    bool any = false;
    for (QWeight* w : all) {
      if (w->gc > 0 && !w->pc) { any = true; }
    }
    if (any) {
      if (!_cvt_stream.valid()) { _cvt_stream = _mc->make_command_stream(); }
      bool ok = true;
      {
        ComputeEncoder e = _cvt_stream.begin_compute();
        for (QWeight* w : all) { ok = encode_fast_(e, *w) && ok; }
      }
      if (!_cvt_stream.commit().wait_ok(nullptr) || !ok) { return false; }
    }
    for (QWeight* w : all) {
      if (!sign_(*w)) { return false; }
    }
    return true;
  };
  ops.same_layout = [this](int L, const Block& b) {
    return same_layout_(L, b);
  };
  _slots.set_weight_set(_ws.get());
  _slots.configure(_mc, std::move(ops), "MetalMossImageTransformer",
                   "VPIPE_MOSS_IMAGE_NO_SLOTS");
}

// ---- load -----------------------------------------------------------

std::unique_ptr<MetalMossImageTransformer>
MetalMossImageTransformer::load(const std::string& model_dir,
                                MetalCompute* mc, const Config& cfg,
                                bool stream_blocks,
                                const std::vector<LoraSpec>& loras)
{
  return load(WeightSet::open(model_dir, nullptr), mc, cfg, stream_blocks,
              loras);
}

std::unique_ptr<MetalMossImageTransformer>
MetalMossImageTransformer::load(std::shared_ptr<WeightSet> ws,
                                MetalCompute* mc, const Config& cfg,
                                bool stream_blocks,
                                const std::vector<LoraSpec>& loras)
{
  if (ws == nullptr || mc == nullptr) { return nullptr; }
  std::unique_ptr<MetalMossImageTransformer> m(
      new MetalMossImageTransformer());
  m->_mc = mc;
  m->_cfg = cfg;
  m->_ws = std::move(ws);
  m->_stream_blocks = stream_blocks;
  auto err = [&](const std::string& what) {
    if (mc->session() != nullptr) {
      mc->session()->error(fmt("MetalMossImageTransformer: {}", what));
    }
    return std::unique_ptr<MetalMossImageTransformer>();
  };
  if (cfg.head_dim != 128 || cfg.n_heads * cfg.head_dim != cfg.hidden) {
    return err("only head_dim 128 is supported");
  }

  // Quantization is read off the checkpoint: a pack carries `.scales`.
  {
    const MetalLlamaWeights& w = m->_ws->src();
    const std::string p0 = m->prefix_(0);
    const auto* si = w.info(p0 + "proj.q.scales");
    const auto* ci = w.info(p0 + "proj.q.weight");
    if (si != nullptr && ci != nullptr && si->shape.size() == 2 &&
        ci->shape.size() == 2 && si->shape[1] > 0) {
      m->_quant_group = (int)(m->_cfg.hidden / si->shape[1]);
      if (m->_quant_group <= 0) { m->_quant_group = 64; }
      const long K = si->shape[1] * (long)m->_quant_group;
      m->_quant_bits = K > 0 ? (int)(ci->shape[1] * 32 / K) : 0;
      if (m->_quant_bits != 4 && m->_quant_bits != 8) { m->_quant_bits = 0; }
    }
    if (m->_quant_bits == 0 && ci != nullptr) {
      m->_src_f32 = ci->dtype == "F32";
    }
  }
  if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_ROPE_F32")) {
    m->_rope_bf16 = std::atoi(e) == 0;
  }

  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms = mc->load_library("rms_norm_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_lib_attn = mc->load_library("attn_steel");
  m->_lib_mma = mc->load_library("dense_gemm_mma_bf16");
  m->_lib_dequant = mc->load_library("affine_dequant_bf16");
  m->_lib_qmm = mc->load_library("affine_qmm_steel_bf16");
  if (mc->supports_matrix_cores()) {
    m->_lib_attn_nax = mc->load_library("attn_steel_nax");
  }

  m->_fn_gemm_t = m->_lib_gemm.function("dense_gemm_t_f16");
  // THE WIDER TILES, which a box WITHOUT matrix cores runs for every
  // GEMM in the model.
  m->_fn_gemm_bm64 = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_gemm_bm64bn64 = m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  m->_fn_adaln = m->_lib_elt.function("adaln_modulate_f16");
  m->_fn_adaln4 = m->_lib_elt.function("adaln_modulate_v4_f16");
  m->_fn_gated = m->_lib_elt.function("gated_residual_f16");
  m->_fn_gated4 = m->_lib_elt.function("gated_residual_v4_f16");
  m->_fn_situ = m->_lib_elt.function("situ_split_f16");
  m->_fn_situ2 = m->_lib_elt.function("situ_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_residual4 = m->_lib_elt.function("residual_add_v4_f16");
  m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_silu = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_rms = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_rms_rope = m->_lib_rope.function("rms_rope_half_ftab_tm_f16");
  if (m->_quant_bits > 0) {
    const std::string g = "g" + std::to_string(m->_quant_group);
    m->_fn_dequant4 = m->_lib_dequant.function("affine_dequant_w4" + g);
    m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant_w8" + g);
    m->_fn_qmm4 = m->_lib_qmm.function("affine_qmm_steel_w4" + g);
    m->_fn_qmm8 = m->_lib_qmm.function("affine_qmm_steel_w8" + g);
  }
  m->_fn_gemm_mma = m->_lib_mma.function("dense_gemm_mma_t_n128_f16");
  m->_fn_gemm_mma_deep =
      m->_lib_mma.function("dense_gemm_mma_t_n128x256_f16");
  m->_fn_gemm_mma_tn2 =
      m->_lib_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");

  // VALIDATE. An invalid ComputeFunction poisons its dispatch.
  if (!m->_fn_gemm_t.valid() || !m->_fn_adaln.valid() ||
      !m->_fn_gated.valid() || !m->_fn_situ.valid() ||
      !m->_fn_situ2.valid() ||
      !m->_fn_rms.valid() || !m->_fn_rms_rope.valid() ||
      !m->_fn_residual.valid() || !m->_fn_silu.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_bias_add.valid()) {
    return err("a required kernel did not build");
  }
  if (m->_quant_bits > 0 &&
      (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid())) {
    return err(fmt("no quantized GEMM for group {} -- this checkpoint "
                   "cannot be run (re-quantize at group 64)",
                   m->_quant_group)());
  }
  m->_steel_attn_ok = m->_lib_attn.valid();
  m->_use_attn_nax = m->_steel_attn_ok && m->_lib_attn_nax.valid() &&
                     mc->supports_matrix_cores() &&
                     std::getenv("VPIPE_MOSS_IMAGE_NO_NAX_ATTN") == nullptr;
  m->_use_mma2 = mc->supports_matrix_cores() && m->_fn_gemm_mma.valid() &&
                 std::getenv("VPIPE_MOSS_IMAGE_NO_MMA2") == nullptr;
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm,
                                              /*bf16=*/true);
    i8->set_tier(cfg.i8_tier >= 2   ? I8GemmContext::Tier::kToken
                 : cfg.i8_tier == 1 ? I8GemmContext::Tier::kFast
                                    : I8GemmContext::Tier::kAccurate);
    i8->set_token_rotation(cfg.i8_tier == 2);
    i8->set_autotune(cfg.autotune);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }
  if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_ANE")) {
    m->_cfg.ane_ffn = std::atoi(e) != 0;
  }
  if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_ANE_ROWS")) {
    m->_cfg.ane_rows = (float)std::atof(e);
  }
  if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_ANE_QKV")) {
    m->_cfg.ane_qkv = std::atoi(e) != 0;
  }
  // q / k / v on the ANE rides the feed-forward's worker and balancer.
  if (m->_cfg.ane_qkv && !m->_cfg.ane_ffn) { m->_cfg.ane_qkv = false; }
  if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_SOL")) {
    m->_cfg.sol.enabled = std::atoi(e) != 0;
  }
  // HIGHER tau keeps FEWER blocks: it is a threshold, not a budget.
  if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_SOL_TAU")) {
    m->_cfg.sol.tau = (float)std::atof(e);
  }
  if (m->_cfg.sol.enabled) {
    std::string serr;
    m->_sol = MetalSolAttention::load(mc, /*bf16=*/true, &serr);
    if (m->_sol == nullptr && mc->session() != nullptr) {
      mc->session()->log_normal(fmt(
          "MetalMossImageTransformer: sol_attn unavailable -- {}",
          serr.empty() ? "kernels did not load" : serr));
    }
  }
  {
    if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_SAGE")) {
      m->_cfg.sage.enabled = std::atoi(e) != 0;
    }
    bool sage_fatal = false;
    m->_sage = MetalSageAttention::load_for_model(
        mc, /*bf16=*/true, m->_cfg.sage, "MetalMossImageTransformer",
        &sage_fatal);
    if (sage_fatal) { return nullptr; }
    if (m->_sage && !m->_use_attn_nax) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt(
            "MetalMossImageTransformer: sage_attn is off -- the matrix-core "
            "flash entry is unavailable here"));
      }
      m->_sage.reset();
    }
  }
  // A CAP on the matmul2d row band, so a test can run the banded route
  // at a geometry far below the 2 GB line it exists for.
  if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_NO_TM")) {
    m->_attn_tm = std::atoi(e) == 0;
  }
  if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_BAND_ROWS")) {
    const int b = std::atoi(e);
    m->_band_cap = b > 0 ? std::max(128, (b / 128) * 128) : 0;
  }
  if (const char* t = std::getenv("VPIPE_MOSS_IMAGE_GEMM_TILE")) {
    m->_gemm_tile = std::atoi(t);
  }
  if (m->_gemm_tile == 2 && !m->_fn_gemm_bm64bn64.valid()) {
    m->_gemm_tile = 1;
  }
  if (m->_gemm_tile == 1 && !m->_fn_gemm_bm64.valid()) { m->_gemm_tile = 0; }

  // A RANK (Config::tp): its share, streamed then kept as any MOSS-Image
  // block is, nothing that reads the checkpoint's widths -- the ANE's
  // modules, a runtime LoRA -- and weights it can cut: dense or an
  // affine pack.
  const bool rank = m->_cfg.tp.active();
  if (rank) {
    const TpShare& t = m->_cfg.tp;
    std::string why;
    const auto* q0 = m->_ws->src().info(m->prefix_(0) + "proj.q.weight");
    const bool dense = q0 != nullptr &&
                       (q0->dtype == "BF16" || q0->dtype == "F16" ||
                        q0->dtype == "F32");
    if (t.n_heads != m->_cfg.n_heads || t.n_ffn != m->_cfg.ffn_inner ||
        t.heads() <= 0 || t.ffn() <= 0) {
      why = fmt("a share of {} heads and {} feed-forward rows does not "
                "fit this model's {} and {}", t.n_heads, t.n_ffn,
                m->_cfg.n_heads, m->_cfg.ffn_inner)();
    } else if (!loras.empty()) {
      why = "not with a runtime LoRA yet";
    } else if (m->_cfg.ane_ffn || m->_cfg.ane_qkv) {
      why = "not with the Neural Engine's share";
    } else if (m->_quant_bits == 0 && !dense) {
      why = "a split DiT reads dense or affine weights only";
    }
    if (!why.empty()) {
      return err(fmt("tensor_parallel: {}", why)());
    }
  }
  if (!m->load_fixed_(*m->_ws)) {
    return err("the model-level weights are incomplete");
  }

  // THE SIGNED LAYOUT for the block projections (see _sgn): decided
  // before any block loads.
  m->_sgn = m->_i8 && m->_use_mma2 && m->_quant_bits > 0 &&
            m->_quant_group == 64 &&
            !m->_cfg.ane_ffn && !m->_cfg.ane_qkv &&
            (m->_i8->signed_available(8) || m->_i8->signed_available(4));
  // THE FAST OR TOKEN TIER for the main blocks: g64 packs (w8, w4, or
  // both in a mixed one) and dense checkpoints, on the matrix-core route.
  m->_fast = m->_i8 && m->_use_mma2 &&
             m->_i8->tier() != I8GemmContext::Tier::kAccurate &&
             (m->_quant_bits == 0 || m->_quant_group == 64) &&
             !m->_cfg.ane_ffn && !m->_cfg.ane_qkv &&
             m->_i8->fast_group(m->_cfg.hidden) > 0;
  if (m->_fast && mc->session() != nullptr) {
    const int fg = m->_i8->fast_group(m->_cfg.hidden);
    mc->session()->info(fmt(
        "MetalMossImageTransformer: the main blocks' projections take the "
        "{}{}",
        fg == I8GemmContext::kTokenGroup
            ? std::string(m->_i8->token_rotated()
                              ? "TOKEN int8 tier (per channel x per token, "
                                "Hadamard-256 rotated"
                              : "TOKEN int8 tier (per channel x per token") +
                  "; " + m->_i8->token_kernel() + " GEMM)"
            : fmt("FAST int8 tier (rotated, group {})", fg)(),
        m->_stream_blocks && m->_quant_bits == 0
            ? "; a streamed dense block is quantized after each refill"
            : ""));
  }
  if (m->_sgn && mc->session() != nullptr) {
    mc->session()->info(fmt(
        "MetalMossImageTransformer: the quantized block projections take "
        "the signed int8 layout"));
  }

  // THE TEXT BLOCKS ARE ALWAYS HELD: two blocks is a slot pair's worth,
  // so streaming them would cost the same bytes AND re-read them every
  // forward.
  m->_text.resize((std::size_t)m->_cfg.n_text_blocks);
  for (int i = 0; i < m->_cfg.n_text_blocks; ++i) {
    if (!m->load_text_block_(*m->_ws, i, m->_text[(std::size_t)i])) {
      return err(fmt("text block {} is incomplete", i)());
    }
  }

  m->_wire.open(mc);
  const int NL = m->_cfg.n_layers;
  m->_blocks.resize((std::size_t)NL);
  if (rank && mc->session() != nullptr) {
    const TpShare& t = m->_cfg.tp;
    mc->session()->info(fmt(
        "MetalMossImageTransformer: rank {} of {}: heads [{}, {}) of {}, "
        "feed-forward rows [{}, {}) of {}{}", t.rank, t.ranks, t.head0,
        t.head1, t.n_heads, t.ffn0, t.ffn1, t.n_ffn,
        stream_blocks ? ", streamed then kept" : ""));
  }
  if (stream_blocks) {
    // Sized to the stack so a PROMOTED block has somewhere to land; an
    // unfilled entry reads as empty.
    m->configure_slots_();
    // A RANK STREAMS ON THE PER-BLOCK PATH, never a slot: a slot refills
    // a block by its tensors whole, and a rank's block is its share, cut
    // afresh at each read (load_tp_block_) in the layout its tier holds.
    // Serial reads, then, until residency keeps the stack -- which a box
    // with room does on the first forward.
    if (rank) { m->_slots.disable(); }
  } else if (rank) {
    for (int L = 0; L < NL; ++L) {
      std::string terr;
      if (!m->load_tp_block_(*m->_ws, L, m->_blocks[(std::size_t)L],
                             &terr)) {
        return err(fmt("tensor_parallel: block {}: {}", L, terr)());
      }
    }
  } else {
    for (int L = 0; L < NL; ++L) {
      if (!m->load_block_(*m->_ws, L, m->_blocks[(std::size_t)L],
                          Retain::Cached)) {
        return err(fmt("block {} is incomplete", L)());
      }
    }
  }
  if (!m->bind_loras_(loras) && mc->session() != nullptr) {
    mc->session()->warn(fmt(
        "MetalMossImageTransformer: the LoRA applier did not load; the "
        "adapters are ignored and the base model runs"));
  }
  if (mc->session() != nullptr) {
    mc->session()->log_normal(fmt(
        "MetalMossImageTransformer: {} blocks + {} text blocks, hidden {}, "
        "{} heads x {}, ffn {}, {}{}{}{}",
        NL, m->_cfg.n_text_blocks, m->_cfg.hidden, m->_cfg.n_heads,
        m->_cfg.head_dim, m->_cfg.ffn_inner,
        m->_quant_bits > 0
            ? "w" + std::to_string(m->_quant_bits) + " g" +
                  std::to_string(m->_quant_group)
            : std::string(m->_src_f32 ? "f32 -> bf16" : "bf16"),
        stream_blocks ? ", streaming" : "",
        m->_use_mma2 ? ", matmul2d" : "",
        m->_i8            ? ", i8"
        : m->_cfg.i8_gemm ? ", i8 REQUESTED BUT UNAVAILABLE"
                          : ""));
  }
  return m;
}

// ---- forward --------------------------------------------------------

SharedBuffer
MetalMossImageTransformer::forward(const Request& req, std::string* err)
{
  _fast_failed = false;
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return SharedBuffer{};
  };
  if (req.latents == nullptr || req.text == nullptr) {
    return fail("moss-image forward: missing input");
  }
  // A RANK (Config::tp) runs its share's heads and feed-forward rows in
  // the main blocks: NH and FFI are its own there, HI its heads' width;
  // the text blocks run whole (NHT heads, FFT rows) and H stays the
  // model's.
  const bool rank = _cfg.tp.active();
  const int H = _cfg.hidden, NHT = _cfg.n_heads, Hd = _cfg.head_dim;
  const int NH = rank ? _cfg.tp.heads() : NHT;
  const int HI = NH * Hd;
  const int FFT = _cfg.ffn_inner;
  const int FFI = rank ? _cfg.tp.ffn() : FFT;
  const int LC = _cfg.latent_channels;
  const int IMG = req.grid_h * req.grid_w, TXT = req.text_rows;
  const int JT = IMG + TXT;
  const int NL = _cfg.n_layers;
  if (IMG <= 0 || TXT <= 0 || NHT * Hd != H) {
    return fail("moss-image forward: degenerate geometry");
  }
  if (rank && _tp_pipe == nullptr) {
    return fail("moss-image forward: a rank with no channel "
                "(begin_tp_generation)");
  }
  std::string tp_err;
  if (req.latents->byte_size() < (std::size_t)IMG * LC * 2 ||
      req.text->byte_size() < (std::size_t)TXT * _cfg.text_dim * 2) {
    return fail("moss-image forward: an input is smaller than its shape");
  }
  if (!_steel_attn_ok) {
    return fail("moss-image forward: the steel attention kernel is "
                "unavailable");
  }
  std::vector<char> skip((std::size_t)NL, 0);
  for (int s : req.skip_layers) {
    if (s < 0 || s >= NL) {
      return fail("moss-image forward: a skip layer is outside the stack");
    }
    skip[(std::size_t)s] = 1;
  }

  PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDit,
                     kPerfLlmDitBegin, (std::uint64_t)JT);

  MetalCompute* mc = _mc;
  const float eps = _cfg.norm_eps;
  // SOL reads head-major q/k/v; everything else reads the projections
  // where they lie (token-major).
  const bool sol_on = _sol && _cfg.sol.enabled;
  // TOKEN-MAJOR attention unless the A/B asks for the transposing,
  // head-major route (VPIPE_MOSS_IMAGE_NO_TM=1), which Sol always takes.
  const bool head_major = sol_on || !_attn_tm;

  auto buf = [&](std::size_t elems) {
    return mc->make_shared_buffer(elems * 2);
  };
  // A rank's main-block planes are its share's widths; the text blocks
  // (whole) use `ob`, `af` and the feed-forward scratch at the model's,
  // over the text rows alone.
  const std::size_t jt = (std::size_t)JT, tx = (std::size_t)TXT;
  SharedBuffer joint = buf(jt * H);
  SharedBuffer nrm = buf(jt * H);
  SharedBuffer ob = buf((rank ? tx : jt) * H);
  SharedBuffer qb = buf(jt * HI);
  SharedBuffer kb = buf(jt * HI);
  SharedBuffer vb = buf(jt * HI);
  SharedBuffer af = buf(std::max(jt * HI, tx * H));
  SharedBuffer ffu = buf(std::max(jt * 2 * FFI, tx * FFT));
  SharedBuffer ffg = buf(std::max(jt * FFI, tx * FFT));
  SharedBuffer tqkv = buf(tx * 3 * H);
  SharedBuffer qh, kh, vh, ao;
  if (head_major) {
    qh = buf(jt * HI);
    kh = buf(jt * HI);
    vh = buf(jt * HI);
    ao = buf(jt * HI);
  }
  SharedBuffer tin = buf((std::size_t)_cfg.freq_dim);
  SharedBuffer te1 = buf((std::size_t)H);
  SharedBuffer temb = buf((std::size_t)H);
  SharedBuffer mods = buf((std::size_t)kModChunks * H);
  SharedBuffer hmod = buf((std::size_t)2 * H);
  SharedBuffer vel = buf((std::size_t)IMG * LC);
  SharedBuffer rcos, rsin;
  {
    std::vector<float> c, s;
    moss_image::rope_tables(req.grid_h, req.grid_w, TXT, Hd,
                            _cfg.rope_base, _rope_bf16, &c, &s);
    rcos = mc->make_shared_buffer(c.size() * sizeof(float));
    rsin = mc->make_shared_buffer(s.size() * sizeof(float));
    if (!rcos.empty() && !rsin.empty()) {
      std::memcpy(rcos.contents(), c.data(), c.size() * sizeof(float));
      std::memcpy(rsin.contents(), s.data(), s.size() * sizeof(float));
    }
  }
  if (joint.empty() || nrm.empty() || ob.empty() || qb.empty() ||
      kb.empty() || vb.empty() || af.empty() || ffu.empty() ||
      ffg.empty() || tqkv.empty() || vel.empty() || rcos.empty() ||
      rsin.empty() || mods.empty() || hmod.empty() ||
      (head_major &&
       (qh.empty() || kh.empty() || vh.empty() || ao.empty()))) {
    return fail("moss-image forward: scratch allocation failed");
  }
  // The timestep sinusoid on the host: [cos | sin], in f64 off the f32
  // timestep as the reference forms it, then f32, then bf16 for the
  // Linear.
  {
    const int half = _cfg.freq_dim / 2;
    auto* d = static_cast<std::uint16_t*>(tin.contents());
    const double t = (double)req.timestep;
    for (int i = 0; i < half; ++i) {
      const double f = std::pow(10000.0, -(double)i / (double)half);
      const double ph = t * f;
      d[i] = f32_to_bf16_((float)std::cos(ph));
      d[half + i] = f32_to_bf16_((float)std::sin(ph));
    }
  }

  if (_i8) { _i8->tune_pending(mc); }

  // A RANK'S EXCHANGE BUFFERS, grown HERE to this forward's widest band
  // (every main block runs JT rows): a buffer replaced is forgotten
  // first, and forget() waits out the exchanges posted on it --
  // mid-forward those wait on GPU work not yet committed. Nothing is
  // pending between forwards.
  std::vector<std::pair<int, int>> tp_bands;
  int tp_lag = 1;
  if (rank) {
    int forced = 0;
    if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_TP_BANDS")) {
      forced = std::max(0, std::atoi(e));
    }
    if (const char* e = std::getenv("VPIPE_MOSS_IMAGE_TP_LAG")) {
      tp_lag = std::atoi(e) == 0 ? 0 : 1;
    }
    tp_bands = tp_row_bands(JT, tp_band_floor(_cfg.tp), forced);
    int widest = 0;
    for (const auto& bd : tp_bands) { widest = std::max(widest, bd.second); }
    const std::size_t pb = (std::size_t)widest * (std::size_t)H * 2;
    const int lanes = tp_bands.size() > 1 ? 4 : 1;
    for (int k = 0; k < lanes; ++k) {
      if (!tp_buffer_(_tp_part[k], pb) || !tp_buffer_(_tp_recv[k], pb)) {
        return fail("moss-image forward: tensor parallel: no room for the "
                    "partial sums");
      }
    }
  }

  CommandStream stream = mc->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  if (!enc.valid()) { return fail("moss-image forward: no compute encoder"); }
  // A RANK'S EXCHANGES LEFT POSTED when the forward ends early -- a stop
  // between blocks, a failure -- their partials in work never committed,
  // would hold the GPU's waits and the worker: abandoned here, before the
  // encoder and stream go (declared after them). A forward that ran to
  // its end has none pending.
  struct TpEnd {
    TpOverlap* pipe;
    ~TpEnd()
    {
      if (pipe != nullptr && pipe->pending()) {
        pipe->abandon("the forward ended early");
      }
    }
  } tp_end{rank ? _tp_pipe.get() : nullptr};
  _flops.begin();
  if (_sol) {
    _sol->reset_counts();
    _sol_flops.begin();
  }

  // Per-stage TRACE, for locating a numeric gap against the reference.
  // It COMMITS, so a traced run is not a timing run.
  const char* trace_dir = std::getenv("VPIPE_MOSS_IMAGE_TRACE");
  bool trace_ok = true;
  auto trace = [&](const char* tag, const SharedBuffer& b, std::size_t off,
                   std::size_t n) {
    if (trace_dir == nullptr || *trace_dir == '\0' || !trace_ok) { return; }
    enc.end();
    _flops.flush(stream);
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) { trace_ok = false; return; }
    std::error_code ec;
    fs::create_directories(trace_dir, ec);
    if (b.byte_size() >= (off + n) * 2) {
      const auto* sp = static_cast<const std::uint16_t*>(b.contents()) + off;
      std::vector<float> v(n);
      for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t u = (std::uint32_t)sp[i] << 16;
        std::memcpy(&v[i], &u, 4);
      }
      std::ofstream o(fs::path(trace_dir) / (std::string(tag) + ".f32"),
                      std::ios::binary);
      if (o) {
        o.write(reinterpret_cast<const char*>(v.data()),
                (std::streamsize)(v.size() * sizeof(float)));
      }
    }
    // The same stream: a fresh one is a new queue, and its first
    // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
    // stream_per_commit_costs).
    enc = stream.begin_compute();
  };

  // Per-CATEGORY profile. READ IT WITH ITS MARK COUNT: every mark is a
  // commit AND a wait, and it defeats the prefetch (see Z-Image's twin
  // of this profiler for the measurements behind both warnings).
  const bool prof = [] {
    const char* e = std::getenv("VPIPE_MOSS_IMAGE_DIT_PROFILE");
    return e != nullptr && *e != '\0' && std::atoi(e) != 0;
  }();
  struct Bucket { const char* name; double ms; long marks; };
  std::vector<Bucket> pbuckets;
  long pmarks = 0;
  auto pclock = std::chrono::steady_clock::now();
  auto mark = [&](const char* bucket) {
    if (!prof) { return; }
    enc.end();
    _flops.flush(stream);
    std::string ge;
    (void)stream.commit().wait_ok(&ge);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pclock).count();
    ++pmarks;
    bool found = false;
    for (auto& b : pbuckets) {
      if (std::strcmp(b.name, bucket) == 0) {
        b.ms += ms; ++b.marks; found = true;
      }
    }
    if (!found) { pbuckets.push_back(Bucket{bucket, ms, 1}); }
    // The same stream: a fresh one is a new queue, and its first
    // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
    // stream_per_commit_costs).
    enc = stream.begin_compute();
    pclock = std::chrono::steady_clock::now();
  };

  // ---- dispatch helpers ---------------------------------------------
  auto lin = [&](const SharedBuffer& x, std::size_t xe, const QWeight& w_in,
                 const SharedBuffer& y, std::size_t ye, int M, int N, int K,
                 const lora::Stack& lf = lora::Stack{}) {
    auto lora_after = [&]() {
      if (!lf.empty()) {
        _lora.apply(enc, x, xe, lf, y, ye, M, N, K, _mma_min_m);
      }
    };
    _flops.gemm(M, N, K);
    if (gemm_mma_(enc, x, xe, w_in, y, ye, M, N, K)) {
      lora_after();
      return;
    }
    QWeight sv;
    const QWeight* sp = signed_dense_(enc, w_in, sv, N, K);
    if (sp == nullptr) { return; }
    const QWeight& w = *sp;
    if (!w.quantized) {
      int bm = 32, bn = 32;
      const metal_compute::ComputeFunction* fn = &_fn_gemm_t;
      if (M >= 128 && _gemm_tile == 1) { fn = &_fn_gemm_bm64; bm = 64; }
      else if (M >= 128 && _gemm_tile == 2) {
        fn = &_fn_gemm_bm64bn64; bm = 64; bn = 64;
      }
      enc.set_function(*fn);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w.w);
      enc.set_buffer(2, w.w); enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                    (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
      lora_after();
      return;
    }
    enc.set_function(w.bits == 8 ? _fn_qmm8 : _fn_qmm4);
    enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
    enc.set_buffer(2, w.qbias); enc.set_buffer(3, x, xe * 2);
    enc.set_buffer(4, y, ye * 2);
    enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                  (unsigned)(((M + 31) / 32) * 2), 2u}, {32, 2, 2});
    lora_after();
  };
  auto bias = [&](const SharedBuffer& b, const SharedBuffer& y,
                  std::size_t ye, int M, int N) {
    if (b.empty()) { return; }
    enc.set_function(_fn_bias_add);
    enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, b);
    enc.set_constant(2, N); enc.set_constant(3, M * N);
    enc.dispatch({(unsigned)((std::size_t)M * N), 1, 1}, {256, 1, 1});
  };
  auto rms = [&](const SharedBuffer& x, std::size_t xe, const SharedBuffer& w,
                 const SharedBuffer& y, std::size_t ye, int R, int D) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w);
    enc.set_buffer(2, y, ye * 2);
    enc.set_constant(3, D); enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  auto silu = [&](const SharedBuffer& x, const SharedBuffer& y, int n) {
    enc.set_function(_fn_silu);
    enc.set_buffer(0, x); enc.set_buffer(1, x); enc.set_buffer(2, y);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  // out = (1 + scale) * x + shift, one H-wide (scale, shift) over R rows.
  auto modulate = [&](const SharedBuffer& x, std::size_t xe,
                      const SharedBuffer& sc, std::size_t sce,
                      const SharedBuffer& sh, std::size_t she,
                      const SharedBuffer& y, std::size_t ye, int R) {
    if ((H % 4) == 0 && _fn_adaln4.valid()) {
      enc.set_function(_fn_adaln4);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, sc, sce * 2);
      enc.set_buffer(2, sh, she * 2); enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, H / 4); enc.set_constant(5, R);
      enc.dispatch({(unsigned)(H / 4), (unsigned)R, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_adaln);
    enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, sc, sce * 2);
    enc.set_buffer(2, sh, she * 2); enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, H);
    enc.set_constant(5, (int)((std::size_t)R * H));
    enc.dispatch({(unsigned)((std::size_t)R * H), 1, 1}, {256, 1, 1});
  };
  // h += gate * sub, PLAIN: no tanh and no (1 + gate).
  auto gated = [&](const SharedBuffer& h, std::size_t he,
                   const SharedBuffer& gate, std::size_t ge,
                   const SharedBuffer& sub, std::size_t se, int R) {
    if ((H % 4) == 0 && _fn_gated4.valid()) {
      enc.set_function(_fn_gated4);
      enc.set_buffer(0, h, he * 2); enc.set_buffer(1, gate, ge * 2);
      enc.set_buffer(2, sub, se * 2);
      enc.set_constant(3, H / 4); enc.set_constant(4, R);
      enc.dispatch({(unsigned)(H / 4), (unsigned)R, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_gated);
    enc.set_buffer(0, h, he * 2); enc.set_buffer(1, gate, ge * 2);
    enc.set_buffer(2, sub, se * 2);
    enc.set_constant(3, H);
    enc.set_constant(4, (int)((std::size_t)R * H));
    enc.dispatch({(unsigned)((std::size_t)R * H), 1, 1}, {256, 1, 1});
  };
  auto resid = [&](const SharedBuffer& a, std::size_t ae,
                   const SharedBuffer& b, std::size_t be, int n) {
    if ((n % 4) == 0 && _fn_residual4.valid()) {
      enc.set_function(_fn_residual4);
      enc.set_buffer(0, a, ae * 2); enc.set_buffer(1, b, be * 2);
      enc.set_buffer(2, a, ae * 2); enc.set_constant(3, n / 4);
      enc.dispatch({(unsigned)(n / 4), 1, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_residual);
    enc.set_buffer(0, a, ae * 2); enc.set_buffer(1, b, be * 2);
    enc.set_buffer(2, a, ae * 2); enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  // SITU over a fused [rows, 2F] = [hidden | gate] into [rows, F].
  auto situ = [&](const SharedBuffer& in, const SharedBuffer& out, int rows) {
    enc.set_function(_fn_situ);
    enc.set_buffer(0, in); enc.set_buffer(1, out);
    enc.set_constant(2, rows); enc.set_constant(3, FFI);
    enc.dispatch({(unsigned)((std::size_t)rows * FFI), 1, 1}, {256, 1, 1});
  };
  // Per-head RMSNorm + the half-split rotation, IN PLACE on a
  // token-major [rows, stride] buffer whose heads start at element `xe`
  // of row 0; tables from row `row0` of the joint table.
  auto rms_rope = [&](const SharedBuffer& x, std::size_t xe,
                      const SharedBuffer& gamma, int rows, int row0,
                      int stride, int nh) {
    const std::size_t tb = (std::size_t)row0 * (Hd / 2) * sizeof(float);
    enc.set_function(_fn_rms_rope);
    enc.set_buffer(0, x, xe * 2);
    enc.set_buffer(1, rcos, tb); enc.set_buffer(2, rsin, tb);
    enc.set_buffer(3, gamma);
    enc.set_constant(4, nh); enc.set_constant(5, rows);
    enc.set_constant(6, Hd); enc.set_constant(7, stride);
    enc.set_constant(8, eps);
    enc.dispatch({32, (unsigned)rows, (unsigned)nh}, {32, 1, 1});
  };
  auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                       int A, int B) {
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, in); enc.set_buffer(1, out);
    enc.set_constant(2, A); enc.set_constant(3, B); enc.set_constant(4, Hd);
    enc.dispatch({(unsigned)Hd, (unsigned)B, (unsigned)A},
                 {(unsigned)Hd, 1, 1});
  };

  // ---- attention ----------------------------------------------------
  //
  // DENSE AND BIDIRECTIONAL. Two geometries per forward (the text alone,
  // the joint sequence), so a params block and a specialized function
  // are memoized per (rows, row stride, layout).
  const bool nax = _use_attn_nax;
  const int A_BQ = nax ? 64 : 32;
  const int A_BK = nax ? 32 : 16;
  const float ascale = 1.0f / std::sqrt((float)Hd);
  struct PKey { int rows, stride, nh; bool hm; SharedBuffer b; };
  std::vector<PKey> aparams;
  // `stride` is the row stride of a TOKEN-major operand; `hm` takes the
  // head-major layout instead ([nh, rows, Hd], Sol's). `nh` heads: the
  // model's in a text block, a rank's share in a main one.
  auto params_for = [&](int R, int stride, bool hm,
                        int nh) -> const SharedBuffer* {
    for (auto& p : aparams) {
      if (p.rows == R && p.stride == stride && p.hm == hm && p.nh == nh) {
        return &p.b;
      }
    }
    SharedBuffer b = mc->make_shared_buffer(sizeof(SteelAttnParams));
    if (b.empty()) { return nullptr; }
    auto* p = static_cast<SteelAttnParams*>(b.contents());
    p->B = 1; p->H = nh; p->D = Hd; p->qL = R; p->kL = R;
    p->gqa_factor = 1; p->scale = ascale;
    p->NQ = (R + A_BQ - 1) / A_BQ; p->NK = (R + A_BK - 1) / A_BK;
    p->NQ_aligned = R / A_BQ; p->NK_aligned = R / A_BK;
    p->qL_rem = R - p->NQ_aligned * A_BQ;
    p->kL_rem = R - p->NK_aligned * A_BK;
    p->qL_off = 0;
    const std::int64_t rs = (std::int64_t)R * Hd;
    for (std::int64_t* s : {p->Q_strides, p->K_strides, p->V_strides,
                            p->O_strides}) {
      s[0] = (std::int64_t)nh * rs;
      if (hm) {
        s[1] = rs; s[2] = Hd;
      } else {
        // [rows, heads, head_dim]: a head is Hd along, a row `stride`.
        s[1] = Hd; s[2] = stride;
      }
    }
    // The output is always a plain [rows, nh * Hd] (token-major) or [nh,
    // rows, Hd] (head-major) buffer, whatever the inputs' stride.
    if (!hm) { p->O_strides[2] = (std::int64_t)nh * Hd; }
    aparams.push_back(PKey{R, stride, nh, hm, std::move(b)});
    return &aparams.back().b;
  };
  auto attn_fn = [&](int R, bool i8) -> const metal_compute::ComputeFunction* {
    const unsigned key = (nax ? 8u : 0u) | (i8 ? 4u : 0u) |
                         (((R % A_BQ) == 0) ? 2u : 0u) |
                         (((R % A_BK) == 0) ? 1u : 0u);
    auto it = _attn_fn.find(key);
    if (it != _attn_fn.end()) {
      return it->second.valid() ? &it->second : nullptr;
    }
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (R % A_BQ) == 0).set_bool(201, (R % A_BK) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false)
        .set_bool(sage::kQkInt8Constant, i8);
    MetalSageAttention::pv8_constants(
        fc, i8 && _sage && _sage->pv8_for(_cfg.sage));
    metal_compute::ComputeFunction fn =
        nax ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
            : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
    auto ins = _attn_fn.emplace(key, std::move(fn));
    return ins.first->second.valid() ? &ins.first->second : nullptr;
  };
  bool attn_ok = true;
  // Token-major attention over `rows` rows: q / k / v at their element
  // offsets in their buffers, row stride `stride`; the output into `o`
  // ([rows, H]).
  auto run_attn_tm = [&](int L, const SharedBuffer& q, std::size_t qe,
                         const SharedBuffer& k, std::size_t ke,
                         const SharedBuffer& v, std::size_t ve,
                         const SharedBuffer& o, int rows, int stride,
                         int nh) {
    if (!attn_ok) { return; }
    bool i8_ok = false;
    if (_sage && nax && _cfg.sage.enabled && L >= _cfg.sage.dense_layers &&
        rows >= kSageMinSegment) {
      const MetalSageAttention::Operand qo{&q, qe, stride, Hd};
      const MetalSageAttention::Operand ko{&k, ke, stride, Hd};
      const MetalSageAttention::Operand vo{&v, ve, stride, Hd};
      std::string gerr;
      i8_ok = _sage->prepare(enc, qo, ko, nh, nh, rows, rows, Hd, A_BQ,
                             A_BK, _cfg.sage, &gerr, &vo);
      // The int8 function takes the int8 P*V's operands iff pv8_for()
      // says so: a prologue that made none runs dense.
      if (i8_ok && _sage->pv8_for(_cfg.sage) != _sage->pv8_ready()) {
        i8_ok = false;
      }
    }
    const SharedBuffer* pp = params_for(rows, stride, false, nh);
    const metal_compute::ComputeFunction* fn = attn_fn(rows, i8_ok);
    if (pp == nullptr || fn == nullptr) { attn_ok = false; return; }
    enc.set_function(*fn);
    enc.set_buffer(0, q, qe * 2); enc.set_buffer(1, k, ke * 2);
    enc.set_buffer(2, v, ve * 2);
    enc.set_buffer(3, o); enc.set_buffer(4, *pp);
    if (i8_ok) { _sage->bind(enc); }
    enc.dispatch({(unsigned)(32 * ((rows + A_BQ - 1) / A_BQ)),
                  (unsigned)(4 * nh), 1}, {32, 4, 1});
    _flops.attention(nh, (long long)rows * rows, Hd);
  };
  // The main blocks' attention: Sol where it is on and the sequence
  // long enough (head-major, through transposes), token-major steel
  // otherwise. Reads qb / kb / vb, writes af.
  auto run_attn_main = [&](int L, int rows) {
    if (!attn_ok) { return; }
    if (sol_on && L >= _cfg.sol.dense_layers && rows >= kSolMinSegment) {
      const int blk = _cfg.sol.key_block > 0 ? _cfg.sol.key_block
                                             : sol::kBlock;
      if ((rows + blk - 1) / blk >= 16) {
        transpose(qb, qh, rows, NH);
        transpose(kb, kh, rows, NH);
        transpose(vb, vh, rows, NH);
        MetalSolAttention::Band band{rows, rows, rows, rows, 0, 0};
        std::string serr;
        if (_sol->encode(enc, qh, kh, vh, ao, NH, NH, band, Hd, ascale,
                         _cfg.sol, &serr)) {
          _flops.add(_sol_flops.routed(
              attention_flops(NH, (long long)rows * rows, Hd),
              _sol->total_blocks()));
          transpose(ao, af, NH, rows);
          return;
        }
        if (_mc->session() != nullptr) {
          _mc->session()->log_debug(fmt(
              "moss-image: sol declined block {} ({}); dense", L,
              serr.empty() ? "no reason" : serr));
        }
      }
    }
    if (!_attn_tm) {
      // THE HEAD-MAJOR A/B: [NH, rows, Hd] planes, contiguous per head.
      transpose(qb, qh, rows, NH);
      transpose(kb, kh, rows, NH);
      transpose(vb, vh, rows, NH);
      bool i8_ok = false;
      if (_sage && nax && _cfg.sage.enabled &&
          L >= _cfg.sage.dense_layers && rows >= kSageMinSegment) {
        const MetalSageAttention::Operand qo{&qh, 0, Hd, rows * Hd};
        const MetalSageAttention::Operand ko{&kh, 0, Hd, rows * Hd};
        const MetalSageAttention::Operand vo{&vh, 0, Hd, rows * Hd};
        std::string gerr;
        i8_ok = _sage->prepare(enc, qo, ko, NH, NH, rows, rows, Hd, A_BQ,
                               A_BK, _cfg.sage, &gerr, &vo);
        // The int8 function takes the int8 P*V's operands iff pv8_for()
        // says so: a prologue that made none runs dense.
        if (i8_ok && _sage->pv8_for(_cfg.sage) != _sage->pv8_ready()) {
          i8_ok = false;
        }
      }
      const SharedBuffer* pp = params_for(rows, Hd, true, NH);
      const metal_compute::ComputeFunction* fn = attn_fn(rows, i8_ok);
      if (pp == nullptr || fn == nullptr) { attn_ok = false; return; }
      enc.set_function(*fn);
      enc.set_buffer(0, qh); enc.set_buffer(1, kh); enc.set_buffer(2, vh);
      enc.set_buffer(3, ao); enc.set_buffer(4, *pp);
      if (i8_ok) { _sage->bind(enc); }
      enc.dispatch({(unsigned)(32 * ((rows + A_BQ - 1) / A_BQ)),
                    (unsigned)(4 * NH), 1}, {32, 4, 1});
      _flops.attention(NH, (long long)rows * rows, Hd);
      transpose(ao, af, NH, rows);
      return;
    }
    run_attn_tm(L, qb, 0, kb, 0, vb, 0, af, rows, HI, NH);
  };

  // ---- the shared modulation and the head's ---------------------------
  //
  // time = MLP(sinusoid); mods = Linear(time) -- NO SiLU in front of it,
  // nor in front of the head's Linear.
  lin(tin, 0, _time_1, te1, 0, 1, H, _cfg.freq_dim);
  bias(_time_1b, te1, 0, 1, H);
  silu(te1, te1, H);
  lin(te1, 0, _time_2, temb, 0, 1, H, H);
  bias(_time_2b, temb, 0, 1, H);
  lin(temb, 0, _mod, mods, 0, 1, kModChunks * H, H);
  bias(_mod_b, mods, 0, 1, kModChunks * H);
  lin(temb, 0, _head_mod, hmod, 0, 1, 2 * H, H);
  bias(_head_mod_b, hmod, 0, 1, 2 * H);
  auto mod_at = [&](int chunk) { return (std::size_t)chunk * H; };

  // ---- the joint stream: image rows, then text rows -------------------
  lin(*req.latents, 0, _x_in, joint, 0, IMG, H, LC);
  bias(_x_in_b, joint, 0, IMG, H);
  const std::size_t txe = (std::size_t)IMG * H;   // the text half's start
  lin(*req.text, 0, _t_in, joint, txe, TXT, H, _cfg.text_dim);
  bias(_t_in_b, joint, txe, TXT, H);
  trace("embed", joint, 0, (std::size_t)JT * H);
  mark("embed");

  if (_lora_rank_total > 0) {
    _lora_scratch_rows = (std::size_t)JT;
    _lora.ensure_scratch(_lora_scratch_rows *
                         (std::size_t)_lora_rank_total);
  }

  const int TOTAL_BLOCKS = _cfg.n_text_blocks + NL;
  int done = 0;

  // ---- the text blocks, over the text rows alone ----------------------
  //
  // UNMODULATED, every Linear biased, q|k|v fused in the checkpoint; the
  // rotation reads the TEXT rows of the joint table.
  for (int i = 0; i < _cfg.n_text_blocks; ++i) {
    if (_stream_stop && _stream_stop()) { return SharedBuffer{}; }
    _flops.flush(stream);
    if (_block_progress) {
      report_block(stream, _block_progress, done, TOTAL_BLOCKS);
    }
    const TextBlock& b = _text[(std::size_t)i];
    rms(joint, txe, b.n1, nrm, 0, TXT, H);
    lin(nrm, 0, b.qkv, tqkv, 0, TXT, 3 * H, H);
    bias(b.qkv_b, tqkv, 0, TXT, 3 * H);
    rms_rope(tqkv, 0, b.nq, TXT, IMG, 3 * H, NHT);
    rms_rope(tqkv, (std::size_t)H, b.nk, TXT, IMG, 3 * H, NHT);
    run_attn_tm(-1, tqkv, 0, tqkv, (std::size_t)H, tqkv, (std::size_t)2 * H,
                af, TXT, 3 * H, NHT);
    lin(af, 0, b.proj, ob, 0, TXT, H, H);
    bias(b.proj_b, ob, 0, TXT, H);
    resid(joint, txe, ob, 0, TXT * H);
    rms(joint, txe, b.n2, nrm, 0, TXT, H);
    // The hidden half (ffn_up) and the gate (ffn_gate) as two GEMMs,
    // then the two-input SITU in place on the hidden one.
    lin(nrm, 0, b.up, ffu, 0, TXT, FFT, H);
    bias(b.up_b, ffu, 0, TXT, FFT);
    lin(nrm, 0, b.gate, ffg, 0, TXT, FFT, H);
    bias(b.gate_b, ffg, 0, TXT, FFT);
    {
      const int n = TXT * FFT;
      enc.set_function(_fn_situ2);
      enc.set_buffer(0, ffu); enc.set_buffer(1, ffg); enc.set_buffer(2, ffu);
      enc.set_constant(3, n);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    }
    lin(ffu, 0, b.down, ob, 0, TXT, H, FFT);
    bias(b.down_b, ob, 0, TXT, H);
    resid(joint, txe, ob, 0, TXT * H);
    ++done;
    trace(("text" + std::to_string(i)).c_str(), joint, txe,
          (std::size_t)TXT * H);
  }
  mark("text");

  // ---- the main stack ---------------------------------------------------
  struct SlotJoin {
    BlockSlots<Block>* s;
    bool on;
    ~SlotJoin() { if (on) { s->join(); } }
  } slot_join{&_slots, _stream_blocks};
  if (_stream_blocks) {
    (void)_slots.release_if_saturated(_blocks);
    _slots.begin_forward();
  }

  // ---- how much of the stack this box can KEEP ------------------------
  if (_stream_blocks) {
    if (_wire.on()) {
      if (_wire.retry(_mc, wire_retry_slack_())) {
        _resid.note_landscape_changed();
      }
      // The trunk takes its place BEFORE this forward's admissions start
      // asking: the blocks are the shed-able half.
      wire_fixed_(true);
    }
    const auto mbudget = mc->memory_budget();
    _resid.begin_forward(mbudget, [this]() -> std::size_t {
      return evict_tail_block_();
    });
    bool shortfall = false;
    if (_resid.count() > 0 &&
        _resid.self_compression_grew(mbudget.self_compressed)) {
      std::size_t examined = 0, incore = 0, paged_out = 0;
      resident_pages_(&examined, &incore, &paged_out);
      if (examined > 0 && incore < examined) {
        shortfall = true;
        const std::size_t freed = _resid.note_weight_residency(
            examined, incore,
            [this]() -> std::size_t { return evict_tail_block_(); });
        if (mc->session() != nullptr) {
          mc->session()->log_normal(fmt(
              "MetalMossImageTransformer: resident weights are only {}% in "
              "RAM ({} of {} sampled pages paged out, {} MB wired) -- "
              "released {} MB, now {} blocks resident",
              (int)(100.0 * (double)incore / (double)examined), paged_out,
              examined, _wire.wired_bytes() >> 20, freed >> 20,
              _resid.count()));
        }
      }
    }
    if (!shortfall) { _resid.note_healthy_forward(); }
  }

  // ---- env-gated streaming split (VPIPE_MOSS_IMAGE_STREAM_PROFILE) ----
  const bool sprof =
      std::getenv("VPIPE_MOSS_IMAGE_STREAM_PROFILE") != nullptr;
  double sp_read_ms = 0.0, sp_gpu_ms = 0.0;
  std::size_t sp_read_bytes = 0;
  int sp_blocks = 0;
  auto sp_now = [&]() {
    return sprof ? std::chrono::steady_clock::now()
                 : std::chrono::steady_clock::time_point{};
  };
  auto sp_add = [&](double& sink, std::chrono::steady_clock::time_point t0) {
    if (sprof) {
      sink += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();
    }
  };

  // The next block this forward will READ after L: not skipped, and not
  // already held.
  auto next_read = [&](int L) {
    for (int n = L + 1; n < NL; ++n) {
      if (skip[(std::size_t)n]) { continue; }
      const bool h = !_blocks[(std::size_t)n].qw.empty();
      if (!h) { return n; }
    }
    return -1;
  };
  // Whether any block has planned an ANE split or probe yet: the first
  // one drains everything queued before it.
  bool ane_planned = false;
  for (int L = 0; L < NL; ++L) {
    if (skip[(std::size_t)L]) {
      ++done;
      continue;
    }
    // COOPERATIVE STOP, every block: empty WITHOUT an error.
    if (_stream_stop && _stream_stop()) { return SharedBuffer{}; }
    const bool held = !_blocks[(std::size_t)L].qw.empty();
    const bool streaming = _stream_blocks && !held;
    const Block* b = nullptr;
    if (!streaming) {
      b = &_blocks[(std::size_t)L];
    } else {
      const auto rd0 = sp_now();
      b = _slots.acquire(L);
      if (b == nullptr) {
        return fail("moss-image forward: a streamed block did not load");
      }
      sp_add(sp_read_ms, rd0);
      if (sprof) {
        sp_read_bytes += _slots.last_bytes();
        ++sp_blocks;
      }
      // THE READ IS NOT ELEMENTWISE WORK: its own bucket.
      mark("stream");
    }
    const int rows = JT;
    // THE ANE'S PLANS FOR THIS BLOCK, decided before anything is encoded
    // because the staging has to ride under the attention. The modules
    // are sized once, for the joint length.
    AneFeedForward::Plan ane_plan = AneFeedForward::Plan::kGpu;
    if (_cfg.ane_ffn && ane_setup_(JT) && ane_eligible_(L, *b)) {
      ane_plan = _ane->plan_block();
    }
    AneFeedForward::Plan qkv_plan = AneFeedForward::Plan::kGpu;
    if (_cfg.ane_qkv && ane_qkv_setup_(JT) && ane_qkv_eligible_(L, *b)) {
      qkv_plan = _ane_qkv->plan_block();
    }
    const bool any_plan = ane_plan != AneFeedForward::Plan::kGpu ||
                          qkv_plan != AneFeedForward::Plan::kGpu;
    if (any_plan &&
        (!ane_planned ||
         (ane_plan != AneFeedForward::Plan::kGpu &&
          _ane->needs_barrier()) ||
         (qkv_plan != AneFeedForward::Plan::kGpu &&
          _ane_qkv->needs_barrier()))) {
      // An unsplit block commits nothing mid-block, so the split point's
      // drain would wait for every block queued before it -- the text
      // blocks and the prelude at the first one. Drain what is queued
      // FIRST.
      enc.end();
      _flops.flush(stream);
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        return fail("moss-image forward: " + ge2);
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
    }
    if (any_plan) { ane_planned = true; }
    // ONE ANE worker serves both tiers, one job at a time: q / k / v are
    // needed first, so they stage first and the feed-forward's weights
    // follow their join.
    if (qkv_plan == AneFeedForward::Plan::kSplit) {
      ane_qkv_stage_(L, *b);
    } else if (ane_plan == AneFeedForward::Plan::kSplit) {
      ane_stage_(L, *b);
    }
    _flops.flush(stream);
    if (_block_progress) {
      // `done` of `total`: the blocks finished BEFORE this one.
      report_block(stream, _block_progress, done, TOTAL_BLOCKS);
    }
    auto LA = [&](lora::Factors BlockLora::* w) { return lora_for_(L, w); };

    // y = norm1(x) * (1 + scale_a) + shift_a
    rms(joint, 0, b->n1, nrm, 0, rows, H);
    modulate(nrm, 0, mods, mod_at(kScaleA), mods, mod_at(kShiftA), nrm, 0,
             rows);
    mark("elt");
    // ---- the q / k / v split point ----------------------------------
    //
    // `nrm` holds the attention input for every row: drain, START THE ANE
    // on the tail rows (its output columns land in qb / kb / vb at the
    // rows they came from), and only then encode the GPU's head rows.
    const bool qkv_split = qkv_plan == AneFeedForward::Plan::kSplit;
    const bool qkv_probe = qkv_plan == AneFeedForward::Plan::kProbe;
    int q_rows = 0;
    double q_drain_ms = 0.0;
    if (qkv_split || qkv_probe) {
      enc.end();
      _flops.flush(stream);
      const auto t_d0 = std::chrono::steady_clock::now();
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        return fail("moss-image forward: " + ge2);
      }
      q_drain_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_d0).count();
      if (qkv_split && _ane_qkv->join_stage(L)) {
        q_rows = _ane_qkv->begin(
            nrm,
            std::vector<AneFeedForward::OutSeg>{
                {&qb, H, 0}, {&kb, H, 0}, {&vb, H, 0}},
            rows);
      } else if (qkv_split && _mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "moss-image: staging block {}'s q/k/v for the ANE failed; it "
            "keeps the GPU projections", L));
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
    }
    // A rank's q / k / v are its heads' width (HI); the input is the
    // model's.
    auto gpu_qkv = [&](int r0, int n) {
      if (n <= 0) { return; }
      const std::size_t e0 = (std::size_t)r0 * H;
      const std::size_t y0 = (std::size_t)r0 * HI;
      if (!fast_shared_(enc, nrm, e0, n, H,
                        {{&b->qw, &qb, y0, HI, !LA(&BlockLora::q).empty()},
                         {&b->kw, &kb, y0, HI, !LA(&BlockLora::k).empty()},
                         {&b->vw, &vb, y0, HI,
                          !LA(&BlockLora::v).empty()}})) {
        lin(nrm, e0, b->qw, qb, y0, n, HI, H, LA(&BlockLora::q));
        lin(nrm, e0, b->kw, kb, y0, n, HI, H, LA(&BlockLora::k));
        lin(nrm, e0, b->vw, vb, y0, n, HI, H, LA(&BlockLora::v));
      }
    };
    const auto t_q0 = std::chrono::steady_clock::now();
    gpu_qkv(0, rows - q_rows);
    mark("gemm.qkv");
    if (qkv_split || qkv_probe) {
      enc.end();
      _flops.flush(stream);
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        if (q_rows > 0) { (void)_ane_qkv->finish(L, rows, 0.0, q_drain_ms); }
        return fail("moss-image forward: " + ge2);
      }
      const double t_gpu = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_q0).count();
      if (qkv_probe) { _ane_qkv->note_probe(q_drain_ms, t_gpu); }
      // THE GPU FALLBACK: rows the ANE lost are projected here; `nrm` is
      // only the tier's input, so every row of it is still there.
      int q_lost = -1;
      if (!_ane_qkv->finish(L, rows, t_gpu, q_drain_ms) && q_rows > 0) {
        q_lost = _ane_qkv->lost_row0();
        if (q_lost < 0 || q_lost >= rows) {
          return fail("moss-image forward: the ANE q/k/v lost its rows");
        }
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
      if (q_lost >= 0) { gpu_qkv(q_lost, rows - q_lost); }
    }
    // The worker is free again: the feed-forward's weights now, so they
    // stage under the attention exactly as they would alone.
    if (qkv_split && ane_plan == AneFeedForward::Plan::kSplit) {
      ane_stage_(L, *b);
    }
    rms_rope(qb, 0, b->nq, rows, 0, HI, NH);
    rms_rope(kb, 0, b->nk, rows, 0, HI, NH);
    mark("rope");
    run_attn_main(L, rows);
    mark("attn");
    // The GPU's feed-forward over rows [r0, r0 + n), `nrm` into `out`
    // from element `oe` -- `ob`'s own rows for the whole model (and the
    // rows the ANE lost), a band's exchange buffer for a RANK. The
    // up|gate and SITU scratch is used from its first row.
    auto gpu_ff = [&](int r0, int n, const SharedBuffer& out,
                      std::size_t oe) {
      if (n <= 0) { return; }
      const std::size_t e0 = (std::size_t)r0 * H;
      lin(nrm, e0, b->ug, ffu, 0, n, 2 * FFI, H, LA(&BlockLora::ug));
      mark("gemm.ff");
      situ(ffu, ffg, n);
      mark("elt");
      lin(ffg, 0, b->dn, out, oe, n, H, FFI, LA(&BlockLora::dn));
      mark("gemm.ff");
    };
    // A RANK'S BLOCK TAIL (Config::tp), from the attention's token-major
    // output on. Its out projection and feed-forward are PARTIAL sums --
    // its heads' and rows' share of the block's -- each completed by the
    // other ranks' sum before its gate. The rows are cut into bands
    // (tp_row_bands: one, the block whole, unless the share was given a
    // floor), each band's partial written straight into an EXCHANGE
    // BUFFER (no copy) and POSTED (TpOverlap): the GPU signals it
    // written, the worker hands it to the channel, and the GPU waits for
    // the others' sum only where it adds it -- the encoder never waits.
    // In GPU order step i runs A(i) -- band i's out projection, posted --
    // then B(i - 1) -- that band's sum added, its attention gate, the
    // feed-forward's norm and modulation and the feed-forward, posted --
    // then C(i - 2) -- that sum added, its gate. So band i's first partial
    // crosses under A(i + 1) and its second under the next step's work;
    // what stays exposed is the last band's feed-forward, once a block.
    // VPIPE_MOSS_IMAGE_TP_LAG=0 runs the steps band after band: no
    // overlap, the same arithmetic. Lanes 0 / 1 carry the attention's
    // partials, 2 / 3 the feed-forward's (a whole block: lane 0 both); a
    // lane's next write follows the GPU's read of its sum before it. The
    // modulation is ONE for every row, so a band reads it at no offset.
    auto rank_tail = [&]() -> bool {
      const int nb = (int)tp_bands.size();
      const int la = nb > 1 ? 2 : 1, lf = nb > 1 ? 2 : 0;
      std::vector<std::uint64_t> va((std::size_t)nb), vc((std::size_t)nb);
      auto post = [&](int lane, int n) {
        const TpOverlap::Ticket t =
            _tp_pipe->post(lane, _tp_part[lane], _tp_recv[lane],
                           (std::size_t)n * (std::size_t)H * 2, _tp_seq++);
        // Between encoders, into the open command buffer: the signal
        // after the dispatch that wrote the partial.
        enc.end();
        stream.encode_signal(*t.ready, t.ready_at);
        enc = stream.begin_compute();
        return t.done_at;
      };
      auto await = [&](std::uint64_t v) {
        enc.end();
        stream.encode_wait(_tp_pipe->done(), v);
        enc = stream.begin_compute();
      };
      auto step_a = [&](int k) {
        const int r0 = tp_bands[(std::size_t)k].first;
        const int n = tp_bands[(std::size_t)k].second;
        lin(af, (std::size_t)r0 * HI, b->ow, _tp_part[k % la], 0, n, H, HI);
        va[(std::size_t)k] = post(k % la, n);
      };
      auto step_b = [&](int k) {
        const int r0 = tp_bands[(std::size_t)k].first;
        const int n = tp_bands[(std::size_t)k].second;
        const std::size_t e0 = (std::size_t)r0 * H;
        const SharedBuffer& p = _tp_part[k % la];
        await(va[(std::size_t)k]);
        resid(p, 0, _tp_recv[k % la], 0, n * H);
        gated(joint, e0, mods, mod_at(kGateA), p, 0, n);
        rms(joint, e0, b->n2, nrm, e0, n, H);
        modulate(nrm, e0, mods, mod_at(kScaleF), mods, mod_at(kShiftF), nrm,
                 e0, n);
        gpu_ff(r0, n, _tp_part[lf + k % la], 0);
        vc[(std::size_t)k] = post(lf + k % la, n);
      };
      auto step_c = [&](int k) {
        const int r0 = tp_bands[(std::size_t)k].first;
        const int n = tp_bands[(std::size_t)k].second;
        const SharedBuffer& p = _tp_part[lf + k % la];
        await(vc[(std::size_t)k]);
        resid(p, 0, _tp_recv[lf + k % la], 0, n * H);
        gated(joint, (std::size_t)r0 * H, mods, mod_at(kGateF), p, 0, n);
      };
      for (int i = 0; i < nb + 2 * tp_lag; ++i) {
        if (i < nb) { step_a(i); }
        if (i - tp_lag >= 0 && i - tp_lag < nb) { step_b(i - tp_lag); }
        if (i - 2 * tp_lag >= 0 && i - 2 * tp_lag < nb) {
          step_c(i - 2 * tp_lag);
        }
      }
      mark("tp");
      if (_tp_pipe->failed()) {
        tp_err = _tp_pipe->error();
        return false;
      }
      return true;
    };
    if (rank) {
      if (!rank_tail()) {
        if (_stream_stop && _stream_stop()) { return SharedBuffer{}; }
        return fail("moss-image forward: tensor parallel: " + tp_err);
      }
    } else {
      lin(af, 0, b->ow, ob, 0, rows, H, H, LA(&BlockLora::o));
      mark("gemm.out");
      gated(joint, 0, mods, mod_at(kGateA), ob, 0, rows);
      // y = norm2(x) * (1 + scale_f) + shift_f
      rms(joint, 0, b->n2, nrm, 0, rows, H);
      modulate(nrm, 0, mods, mod_at(kScaleF), mods, mod_at(kShiftF), nrm, 0,
               rows);
      mark("elt");
      // ---- the feed-forward split point -------------------------------
      //
      // THE ORDER IS LOAD-BEARING: commit the attention without waiting,
      // drain, START THE ANE on the tail rows, and only then encode the
      // GPU's -- after them, only the down projection would be left to
      // hide behind.
      const bool ane_split = ane_plan == AneFeedForward::Plan::kSplit;
      const bool ane_probe = ane_plan == AneFeedForward::Plan::kProbe;
      int a_rows = 0;
      double ane_drain_ms = 0.0;
      if (ane_split || ane_probe) {
        enc.end();
        _flops.flush(stream);
        const auto t_d0 = std::chrono::steady_clock::now();
        std::string ge2;
        if (!stream.commit().wait_ok(&ge2)) {
          return fail("moss-image forward: " + ge2);
        }
        ane_drain_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_d0).count();
        if (ane_split && _ane->join_stage(L)) {
          a_rows = _ane->begin(nrm, ob, rows);
        } else if (ane_split && _mc->session() != nullptr) {
          _mc->session()->warn(fmt(
              "moss-image: staging block {}'s weights for the ANE failed; it "
              "keeps the GPU feed-forward", L));
        }
        // The same stream: a fresh one is a new queue, and its first
        // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
        // stream_per_commit_costs).
        enc = stream.begin_compute();
      }
      const auto t_g0 = std::chrono::steady_clock::now();
      gpu_ff(0, rows - a_rows, ob, 0);
      if (ane_split || ane_probe) {
        enc.end();
        _flops.flush(stream);
        std::string ge2;
        if (!stream.commit().wait_ok(&ge2)) {
          if (a_rows > 0) { (void)_ane->finish(L, rows, 0.0, ane_drain_ms); }
          return fail("moss-image forward: " + ge2);
        }
        const double t_gpu = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_g0).count();
        if (ane_probe) { _ane->note_probe(ane_drain_ms, t_gpu); }
        int ff_lost = -1;
        if (!_ane->finish(L, rows, t_gpu, ane_drain_ms) && a_rows > 0) {
          ff_lost = _ane->lost_row0();
          if (ff_lost < 0 || ff_lost >= rows) {
            return fail("moss-image forward: the ANE feed-forward lost its "
                        "rows");
          }
        }
        // The same stream: a fresh one is a new queue, and its first
        // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
        // stream_per_commit_costs).
        enc = stream.begin_compute();
        if (ff_lost >= 0) {
          gpu_ff(ff_lost, rows - ff_lost, ob, (std::size_t)ff_lost * H);
        }
      }
      gated(joint, 0, mods, mod_at(kGateF), ob, 0, rows);
      mark("elt");
    }
    ++done;

    // ---- the streaming boundary --------------------------------------
    //
    // A streamed block MUST commit: the two slots are reused, so anything
    // encoded-but-not-run still points at bytes the next read
    // overwrites, and the commit -> wait window is where the next read
    // goes.
    if (streaming) {
      const auto gp0 = sp_now();
      enc.end();
      _flops.flush(stream);
      std::string ge;
      {
        CommandStream::Fence bf = stream.commit();
        _slots.prefetch(next_read(L));
        if (!bf.wait_ok(&ge)) { return fail("moss-image forward: " + ge); }
      }
      // ---- keep this block, if the box can hold it -------------------
      //
      // KEPT bytes, not the slot's: a promotion drops a dense source and
      // dead pack scales, and converts a held w4 layout (clone_block_).
      const std::size_t nb =
          _slots.last_bytes() > 0 ? kept_block_bytes_(*b) : 0;
      if (nb > 0 && _wire.wirable(nb) && _resid.admit(_mc, nb) &&
          _slots.promote_into(_blocks[(std::size_t)L])) {
        // What a preloaded block holds, whichever way promote_into went
        // -- see keep_block_. Wired after it: mlock pins what exists now.
        (void)keep_block_(_blocks[(std::size_t)L]);
        _wire.note_wired(_mc, wire_block_(_blocks[(std::size_t)L], true),
                         nb);
        _resid.note_admitted(nb);
      }
      sp_add(sp_gpu_ms, gp0);
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
      if (!enc.valid()) { return fail("moss-image forward: no encoder"); }
      mark("stream");
    } else if (_stream_stop && kStopDrainBlocks > 0 &&
               ((L + 1) % kStopDrainBlocks) == 0 && L + 1 < NL) {
      // A HELD BLOCK COMMITS NOTHING, so without this the whole stack
      // runs inside one wait and a stop mid-forward is never seen.
      enc.end();
      _flops.flush(stream);
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) {
        return fail("moss-image forward: " + ge);
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
      if (!enc.valid()) { return fail("moss-image forward: no encoder"); }
    }
    trace(("blk" + std::to_string(L)).c_str(), joint, 0,
          (std::size_t)JT * H);
  }
  if (!attn_ok) {
    return fail("moss-image forward: the steel attention kernel did not "
                "specialize for this geometry");
  }

  // ---- the head: image rows only ---------------------------------------
  rms(joint, 0, _final_norm, nrm, 0, IMG, H);
  // x * (1 + head[1]) + head[0]: scale is the SECOND chunk.
  modulate(nrm, 0, hmod, (std::size_t)H, hmod, 0, nrm, 0, IMG);
  lin(nrm, 0, _final_lin, vel, 0, IMG, LC, H);
  enc.end();
  _flops.flush(stream);
  {
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) {
      return fail("moss-image forward: " + (ge.empty() ? "GPU failed" : ge));
    }
  }
  // A RANK's exchange that failed after its block was encoded let the GPU
  // run on with wrong sums (TpOverlap releases its waits): said here.
  if (rank && _tp_pipe->failed()) {
    if (_stream_stop && _stream_stop()) { return SharedBuffer{}; }
    return fail("moss-image forward: tensor parallel: " +
                _tp_pipe->error());
  }
  if (_sol) { _sol_flops.finish(_sol->exact_blocks()); }
  if (_fast_failed) {
    _fast_failed = false;
    return fail("moss-image forward: a fast-tier GEMM was declined after "
                "its input was quantized");
  }
  if (prof && mc->session() != nullptr) {
    double tot = 0.0;
    for (const auto& b : pbuckets) { tot += b.ms; }
    std::string line;
    for (const auto& b : pbuckets) {
      line += " | " + std::string(b.name) + " " +
              std::to_string((long)b.ms) + " ms (" +
              std::to_string(tot > 0.0 ? (int)(100.0 * b.ms / tot) : 0) +
              "%, " + std::to_string(b.marks) + " marks)";
    }
    mc->session()->log_normal(fmt(
        "MOSS-Image DiT profile (joint={} img={} text={}, {} blocks, {}, "
        "{} marks -- each is a commit AND a wait, so a bucket with many "
        "marks and little work is mostly measuring this): total {} ms{}",
        JT, IMG, TXT, TOTAL_BLOCKS,
        _use_mma2 ? "matmul2d" : (_gemm_tile == 2 ? "bm64bn64"
                                                  : (_gemm_tile == 1
                                                         ? "bm64"
                                                         : "bm32")),
        pmarks, (long)tot, line));
  }
  if (sprof && sp_blocks > 0 && mc->session() != nullptr) {
    const double tot = sp_read_ms + sp_gpu_ms;
    const double gbs = sp_read_ms > 0.0
        ? (double)sp_read_bytes / (sp_read_ms / 1000.0) / 1e9 : 0.0;
    mc->session()->log_normal(fmt(
        "MetalMossImageTransformer: streamed {} of {} blocks -- read {} ms "
        "({} MB at {:.2f} GB/s), gpu {} ms; a perfect prefetch hides at "
        "most {:.1f}% of this pass",
        sp_blocks, NL, (long)sp_read_ms, sp_read_bytes >> 20, gbs,
        (long)sp_gpu_ms, tot > 0.0 ? 100.0 * sp_read_ms / tot : 0.0));
  }
  return vel;
}

bool
MetalMossImageTransformer::fast_shared_(ComputeEncoder& enc,
                                        const SharedBuffer& x,
                                        std::size_t xe, int M, int K,
                                        std::initializer_list<FastOut> outs)
{
  if (!_i8 || outs.size() < 2 ||
      std::getenv("VPIPE_MOSS_IMAGE_NO_FAST_SHARE") != nullptr) {
    return false;
  }
  const int gc = outs.begin()->w->gc;
  for (const FastOut& o : outs) {
    const QWeight& w = *o.w;
    if (!w.quantized || w.gc <= 0 || w.gc != gc || o.adapted ||
        M > mma_row_band(o.n, K) || !_i8->fast_accepts(M, o.n, K)) {
      return false;
    }
  }
  if (!_i8->quantize_fast(enc, x, xe, M, K, gc)) { return false; }
  for (const FastOut& o : outs) {
    const SharedBuffer* c = nullptr;
    const SharedBuffer* cs = nullptr;
    if (!fast_operand_(enc, *o.w, &c, &cs) ||
        !_i8->gemm_fast_quantized(enc, *c, *cs, gc, SharedBuffer{}, *o.y, o.ye,
                                  M, o.n, K)) {
      _fast_failed = true;
    }
    _flops.gemm(M, o.n, K);
  }
  return true;
}

// ---- the matrix-core GEMM -------------------------------------------
//
// Returns false with NOTHING encoded when this box or this shape is not
// for it, and the caller keeps its tiled path.
bool
MetalMossImageTransformer::gemm_mma_(ComputeEncoder& enc,
                                     const SharedBuffer& xin, std::size_t xe,
                                     const QWeight& w, const SharedBuffer& y,
                                     std::size_t ye, int M, int N,
                                     int K) const
{
  if (!_use_mma2 || N < 16) { return false; }
  // IN ROW BANDS, every route: past 2^31 BYTES on any operand the tiles
  // stop storing, SILENTLY. Declining there would put the widest GEMM on
  // steel at exactly the size where it costs most; below the line the
  // band IS M -- one pass. A route is taken only when its SHORTEST band
  // (the last) clears its floor, so no route declines after encoding
  // some of its bands.
  int band = mma_row_band(N, K);
  if (_band_cap > 0) { band = std::min(band, _band_cap); }
  const int last = M <= band ? M : (M % band == 0 ? band : M % band);
  auto in_bands = [&](auto&& one) -> bool {
    for (int m0 = 0; m0 < M; m0 += band) {
      if (!one((std::size_t)m0, std::min(band, M - m0))) { return false; }
    }
    return true;
  };
  // THE FAST / TOKEN TIER's GEMM (a weight held in its layout).
  if (_i8 && w.quantized && w.gc > 0 && _i8->fast_accepts(last, N, K)) {
    // A w4 pack converts once for all of its bands.
    const SharedBuffer* c = nullptr;
    const SharedBuffer* cs = nullptr;
    if (!fast_operand_(enc, w, &c, &cs)) { return false; }
    return in_bands([&](std::size_t m0, int rows) {
      return _i8->gemm_fast(enc, xin, xe + m0 * K, *c, *cs, w.gc,
                            SharedBuffer{}, y, ye + m0 * N, rows, N, K);
    });
  }
  // A signed weight's int8 GEMM, at the int8 route's own floor.
  if (_i8 && w.quantized && w.sgn &&
      _i8->signed_accepts(last, N, K, w.bits)) {
    return in_bands([&](std::size_t m0, int rows) {
      return _i8->gemm_signed(enc, xin, xe + m0 * K, w.codes, w.scales,
                              w.qbias, w.bits, y, ye + m0 * N, rows, N, K);
    });
  }
  if (M < _mma_min_m) { return false; }

  // Tile by REDUCTION DEPTH: 3072 for q/k/v/o and ffn_up_gate, 8192 for
  // ffn_down, 128 for the patch embedder. Chosen before anything is
  // encoded, so a decline leaves the encoder as it found it.
  int RN = 256;
  const metal_compute::ComputeFunction* fn = &_fn_gemm_mma_deep;
  if (K < 6144) {
    fn = &_fn_gemm_mma; RN = 128;
  } else if (K < 12288 && _fn_gemm_mma_tn2.valid()) {
    fn = &_fn_gemm_mma_tn2; RN = 512;
  }
  if (!fn->valid()) { return false; }

  const SharedBuffer* wdense = nullptr;
  if (w.quantized && (w.sgn || w.gc > 0)) {
    QWeight view;
    if (signed_dense_(enc, w, view, N, K) == nullptr) { return false; }
    wdense = &_w_deq;
  } else if (w.quantized) {
    const metal_compute::ComputeFunction& dq =
        (w.bits == 8) ? _fn_dequant8 : _fn_dequant4;
    if (!dq.valid()) { return false; }
    const std::size_t need = (std::size_t)N * K * 2;   // bf16
    if (_w_deq.empty() || _w_deq.byte_size() < need) {
      _w_deq = _mc->make_shared_buffer(need);
      if (_w_deq.empty()) { return false; }
    }
    enc.set_function(dq);
    enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
    enc.set_buffer(2, w.qbias); enc.set_buffer(3, _w_deq);
    enc.set_constant(4, K); enc.set_constant(5, N);
    const unsigned words = (unsigned)(w.bits == 8 ? (K / 4) : (K / 8));
    enc.dispatch({words, (unsigned)N, 1}, {64, 1, 1});
    wdense = &_w_deq;
  } else {
    wdense = &w.w;
  }
  // Accelerated mode, from the SAME dense operand the tiles read.
  if (_i8 && _i8->accepts(last, N, K)) {
    return in_bands([&](std::size_t m0, int rows) {
      return _i8->gemm(enc, xin, xe + m0 * K, *wdense, y, ye + m0 * N, rows,
                       N, K);
    });
  }
  return in_bands([&](std::size_t m0, int rows) {
    enc.set_function(*fn);
    enc.set_buffer(0, xin, (xe + m0 * K) * 2);
    enc.set_buffer(1, *wdense); enc.set_buffer(2, *wdense);
    enc.set_buffer(3, y, (ye + m0 * N) * 2);
    enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, rows);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                  (unsigned)((rows + 127) / 128), 1}, {256, 1, 1});
    return true;
  });
}

// ---- the ANE tiers ---------------------------------------------------
//
// The feed-forward and the q / k / v projections are row-independent, so
// the ANE takes the LAST rows of a block's and the GPU the head rows,
// concurrently -- exact, the two computing one function over disjoint
// rows. WHERE IT IS FOR: the M4 series (no matrix cores, so the GPU's
// GEMMs leave proportionally more to take). Do not judge it on a fanless
// box, whose engines share one power budget.
int
MetalMossImageTransformer::ane_chunk_rows() noexcept
{
  return AneFeedForward::kChunkDefault;
}

std::size_t
MetalMossImageTransformer::ane_runtime_bytes(const Config& c, int seq) noexcept
{
  std::size_t n = AneFeedForward::runtime_bytes(c.hidden, c.ffn_inner, seq,
                                                ane_chunk_rows());
  // The q|k|v module is a second fixed unit, and has to be booked too.
  if (c.ane_qkv) {
    n += AneFeedForward::matmul_runtime_bytes(c.hidden, 3 * c.hidden, seq,
                                              ane_chunk_rows());
  }
  return n;
}

int
MetalMossImageTransformer::ane_plan_seq(int width, int height) noexcept
{
  // One row per 16x16 pixels plus a prompt this cannot know; 128 rows is
  // noise against the image at any size the tier is for.
  const int w = width > 0 ? width : 1024;
  const int h = height > 0 ? height : 1024;
  return (w / moss_image::kPixelsPerToken) * (h / moss_image::kPixelsPerToken)
         + 128;
}

bool
MetalMossImageTransformer::ane_setup_(int seq)
{
  if (_ane_tried) { return _ane != nullptr; }
  _ane_tried = true;
  if (_mc == nullptr || _mc->session() == nullptr || seq <= 0) {
    return false;
  }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "moss-image";
  o.hidden  = _cfg.hidden;
  o.ffn     = _cfg.ffn_inner;
  o.seq     = seq;
  o.rows    = (float)_cfg.ane_rows;
  o.chunk   = ane_chunk_rows();
  o.situ    = true;
  o.int8    = _cfg.ane_int8;
  o.int8_plain = _cfg.ane_int8_plain;
  o.profile = std::getenv("VPIPE_MOSS_IMAGE_ANE_PROFILE") != nullptr;
  _ane = AneFeedForward::create(o);
  return _ane != nullptr;
}

namespace {

// A projection's weights as the ANE stages them -- dense bf16 or affine
// 4/8-bit, read from row `row0` -- WITH the runtime adapters on it, their
// B rows from the same `row0`: the module is fed fp16 weights per block,
// so an adapter has to be merged into what it stages, or the ANE's rows
// run the base model while the GPU's run the adapted one.
template <typename Q>
AneFfnSource
ane_src_(const Q& q, const lora::Stack& st, std::size_t row0 = 0)
{
  AneFfnSource s;
  s.w         = &q.w;
  s.codes     = q.quantized ? &q.codes : nullptr;
  s.scales    = q.quantized ? &q.scales : nullptr;
  s.qbias     = q.quantized ? &q.qbias : nullptr;
  s.quantized = q.quantized;
  s.bits      = q.quantized ? q.bits : 0;
  s.stride    = 1;
  s.offset    = row0;
  for (int i = 0; i < st.n && s.deltas < AneFfnSource::kMaxDeltas; ++i) {
    AneFfnSource::Delta& d = s.delta[s.deltas++];
    d.a        = &st.s[i].f->a;
    d.b        = &st.s[i].f->b;
    d.rank     = st.s[i].f->b_rank();
    d.group    = st.s[i].f->group;
    d.parts    = st.s[i].f->parts;
    d.scale    = st.s[i].scale;
    d.b_stride = 1;
    d.b_offset = row0;
  }
  return s;
}

bool
ane_stageable_(bool quantized, int gc, int bits, bool has_scales)
{
  if (!quantized) { return true; }
  // The fast layout is not affine: the stager cannot read it.
  return gc == 0 && (bits == 4 || bits == 8) && has_scales;
}

}  // namespace

bool
MetalMossImageTransformer::ane_eligible_(int L, const Block& b) const
{
  if (_ane == nullptr) { return false; }
  const int cap = _cfg.ane_layers > 0
                      ? std::min(_cfg.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  if (L >= cap) { return false; }
  auto ok = [](const QWeight& q) {
    return !q.empty() &&
           ane_stageable_(q.quantized, q.gc, q.bits,
                          !q.scales.empty() && !q.qbias.empty());
  };
  return ok(b.ug) && ok(b.dn);
}

void
MetalMossImageTransformer::ane_stage_(int L, const Block& b)
{
  if (_ane == nullptr) { return; }
  // ffn_up_gate is [hidden | gate] in one matrix, so the GATE is read from
  // row F on and the hidden ("up") half from row 0 -- and its adapter's B
  // rows likewise, a wrong offset being silent (the wrong rows of B on
  // the ANE's rows only).
  const std::size_t F = (std::size_t)_cfg.ffn_inner;
  const lora::Stack ug = lora_for_(L, &BlockLora::ug);
  _ane->stage(L, ane_src_(b.ug, ug, F), ane_src_(b.ug, ug, 0),
              ane_src_(b.dn, lora_for_(L, &BlockLora::dn)), _quant_group);
}

// THE q / k / v TIER: one matmul module, hidden -> 3*hidden, its single
// weight slot holding q, k and v stacked by rows; the output COLUMNS
// scatter straight into qb / kb / vb.
bool
MetalMossImageTransformer::ane_qkv_setup_(int seq)
{
  if (_ane_qkv_tried) { return _ane_qkv != nullptr; }
  _ane_qkv_tried = true;
  if (_mc == nullptr || _mc->session() == nullptr || seq <= 0) {
    return false;
  }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "moss-image-qkv";
  o.hidden  = _cfg.hidden;
  o.ffn     = 3 * _cfg.hidden;       // the matmul's output width
  o.seq     = seq;
  o.rows    = (float)_cfg.ane_rows;
  o.chunk   = ane_chunk_rows();
  o.matmul  = true;
  o.int8    = _cfg.ane_int8;
  o.int8_plain = _cfg.ane_int8_plain;
  o.profile = std::getenv("VPIPE_MOSS_IMAGE_ANE_PROFILE") != nullptr;
  _ane_qkv = AneFeedForward::create(o);
  return _ane_qkv != nullptr;
}

bool
MetalMossImageTransformer::ane_qkv_eligible_(int L, const Block& b) const
{
  if (_ane_qkv == nullptr) { return false; }
  const int cap = _cfg.ane_layers > 0
                      ? std::min(_cfg.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  if (L >= cap) { return false; }
  auto ok = [](const QWeight& q) {
    return !q.empty() &&
           ane_stageable_(q.quantized, q.gc, q.bits,
                          !q.scales.empty() && !q.qbias.empty());
  };
  return ok(b.qw) && ok(b.kw) && ok(b.vw);
}

void
MetalMossImageTransformer::ane_qkv_stage_(int L, const Block& b)
{
  if (_ane_qkv == nullptr) { return; }
  const std::size_t H = (std::size_t)_cfg.hidden;
  auto part = [&](const QWeight& q, std::size_t row0,
                  lora::Factors BlockLora::* which) {
    AneFfnSource s = ane_src_(q, lora_for_(L, which));
    s.slot     = 0;
    s.slot_row = row0;
    s.rows     = H;
    return s;
  };
  _ane_qkv->stage(L,
                  std::vector<AneFfnSource>{
                      part(b.qw, 0, &BlockLora::q),
                      part(b.kw, H, &BlockLora::k),
                      part(b.vw, 2 * H, &BlockLora::v)},
                  _quant_group);
}

// ---- runtime LoRA ---------------------------------------------------
//
// THE MODULE NAMES ARE THE MODEL'S OWN (the main blocks' proj.*), under
// whatever wrapper the shared reader's rename strips. What is verified
// here is the arithmetic and the plumbing (the synthesised-adapter
// test); a published adapter that binds 0 modules wants a rename row.
bool
MetalMossImageTransformer::bind_loras_(const std::vector<LoraSpec>& loras)
{
  if (loras.empty()) { return true; }
  if (!_lora.init(_mc, _use_mma2)) { return false; }
  if (!_lora.valid()) { return false; }
  const int NL = _cfg.n_layers;
  int slot = 0;
  for (const LoraSpec& spec : loras) {
    if (slot >= kMaxLoraSlots || spec.path.empty()) { break; }
    std::string err;
    auto ad = lora::Adapter::open(spec.path, _mc, &err,
                                  lora::remap_ai_toolkit_module);
    if (ad == nullptr) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "MetalMossImageTransformer: lora '{}' did not open: {}",
            spec.path, err.empty() ? "unreadable" : err));
      }
      // The slot is still taken, EMPTY: a live strength is addressed by
      // the index the caller gave it.
      _lora_blk[slot].clear();
      _lora_bound[slot] = 0;
      _lora_scale[slot] = 0.0f;
      _lora_rank_prefix[slot] = _lora_rank_total;
      ++slot;
      continue;
    }
    _lora_blk[slot].clear();
    _lora_blk[slot].resize((std::size_t)NL);
    const int H = _cfg.hidden, FFI = _cfg.ffn_inner;
    for (int L = 0; L < NL; ++L) {
      const std::string pre = prefix_(L);
      BlockLora& d = _lora_blk[slot][(std::size_t)L];
      ad->bind(pre + "proj.q", H, H, &d.q);
      ad->bind(pre + "proj.k", H, H, &d.k);
      ad->bind(pre + "proj.v", H, H, &d.v);
      ad->bind(pre + "proj.o", H, H, &d.o);
      ad->bind(pre + "proj.ffn_up_gate", 2 * FFI, H, &d.ug);
      ad->bind(pre + "proj.ffn_down", H, FFI, &d.dn);
    }
    _lora_bound[slot] = ad->modules();
    _lora_scale[slot] = spec.scale;
    _lora_rank_prefix[slot] = _lora_rank_total;
    _lora_rank_total += ad->max_rank();
    if (_mc->session() != nullptr) {
      _mc->session()->log_normal(
          fmt("MetalMossImageTransformer: {}", ad->summary(spec.path,
                                                           spec.scale)));
      if (ad->modules() == 0) {
        _mc->session()->warn(fmt(
            "MetalMossImageTransformer: lora '{}' bound NO modules -- it is "
            "for a different model, or its names are in a convention this "
            "family does not map yet", spec.path));
      }
    }
    ++slot;
  }
  _lora_slots = slot;
  return true;
}

lora::Stack
MetalMossImageTransformer::lora_for_(int L,
                                     lora::Factors BlockLora::* which) const
{
  lora::Stack st;
  for (int i = 0; i < _lora_slots; ++i) {
    const std::vector<BlockLora>& v = _lora_blk[i];
    if (L < 0 || L >= (int)v.size()) { continue; }
    const lora::Factors& f = v[(std::size_t)L].*which;
    // Each slot its OWN region of the [rows, rank] scratch.
    st.add(f, _lora_scale[i],
           _lora_scratch_rows * (std::size_t)_lora_rank_prefix[i]);
  }
  return st;
}

void
MetalMossImageTransformer::set_lora_scale(int slot, float s)
{
  if (slot < 0 || slot >= kMaxLoraSlots) { return; }
  _lora_scale[slot] = s;
}

int
MetalMossImageTransformer::lora_modules(int slot) const
{
  if (slot < 0 || slot >= kMaxLoraSlots) { return 0; }
  return _lora_bound[slot];
}

// ---- memory -----------------------------------------------------------

std::size_t
MetalMossImageTransformer::streaming_floor_bytes(const std::string& dir)
{
  if (dir.empty()) { return 0; }
  auto wts = MetalLlamaWeights::open_model(dir);
  if (!wts.has_value()) { return 0; }
  static const char kStem[] = "layers.";
  std::size_t trunk = 0;
  std::vector<std::size_t> blocks;
  bool any_f32 = false;
  for (const std::string& nm : wts->tensor_names()) {
    const auto* ti = wts->info(nm);
    const std::size_t nb = ti != nullptr ? (std::size_t)ti->nbytes : 0;
    if (ti != nullptr && ti->dtype == "F32" &&
        nm.find("rope.bands") == std::string::npos) {
      any_f32 = true;
    }
    if (nm.rfind(kStem, 0) != 0) { trunk += nb; continue; }
    const std::size_t i0 = sizeof(kStem) - 1;
    const std::size_t dot = nm.find('.', i0);
    if (dot == std::string::npos) { trunk += nb; continue; }
    const std::size_t idx =
        (std::size_t)std::atol(nm.substr(i0, dot - i0).c_str());
    if (blocks.size() <= idx) { blocks.resize(idx + 1, 0); }
    blocks[idx] += nb;
  }
  std::size_t widest = 0;
  for (std::size_t b : blocks) { widest = std::max(widest, b); }
  if (widest == 0) { return 0; }
  std::size_t floor = trunk + 2 * widest;
  // An F32 checkpoint is narrowed to bf16 on the way in.
  if (any_f32) { floor /= 2; }
  return floor;
}

std::size_t
MetalMossImageTransformer::forward_scratch_bytes(const Config& c, int grid_h,
                                                 int grid_w, int text_rows,
                                                 bool sol, bool packed,
                                                 bool matrix_cores)
{
  const std::size_t img = (std::size_t)grid_h * grid_w;
  const std::size_t tx = (std::size_t)std::max(text_rows, 0);
  const std::size_t jt = img + tx;
  const std::size_t H = (std::size_t)c.hidden, F = (std::size_t)c.ffn_inner;
  // A RANK's main-block planes are its share's: q / k / v, af and the
  // head-major planes its heads' width, the feed-forward its rows'; the
  // text blocks still run whole over the text rows.
  const bool rank = c.tp.active();
  const int nh = rank ? c.tp.heads() : c.n_heads;
  const std::size_t HI = (std::size_t)nh * (std::size_t)c.head_dim;
  const std::size_t FF = rank ? (std::size_t)c.tp.ffn() : F;
  // joint, nrm, ob (a rank's the text rows' alone), q, k, v, af (+ four
  // head-major planes under Sol), the [jt, 2F] up|gate and the [jt, F]
  // product, the text block's fused q|k|v, the velocity, and the f32
  // rotary tables.
  std::size_t n = 2 * jt * H * 2 + (rank ? tx : jt) * H * 2 +
                  (3 + (sol ? 4 : 0)) * jt * HI * 2 +
                  std::max(jt * HI, tx * H) * 2 +
                  std::max(jt * 2 * FF, tx * F) * 2 +
                  std::max(jt * FF, tx * F) * 2 + tx * 3 * H * 2 +
                  img * (std::size_t)c.latent_channels * 2 +
                  2 * jt * (std::size_t)(c.head_dim / 2) * 4;
  // A rank's EXCHANGE BUFFERS: a partial and the others' sum a lane, at
  // the widest band of the joint rows (four lanes banded, one whole).
  if (rank && jt > 0) {
    const auto bands = tp_row_bands((int)jt, tp_band_floor(c.tp));
    std::size_t widest = 0;
    for (const auto& bd : bands) {
      widest = std::max(widest, (std::size_t)bd.second);
    }
    n += (bands.size() > 1 ? 4u : 1u) * 2 * widest * H * 2;
  }
  // HELD between forwards, which the planes above leave out: a pack's
  // dense stand-in (the fused [hidden | gate], the widest projection),
  // and where the box runs them the int8 context, Sage's int8 Q / K and
  // Sol's routing scratch -- ~0.6 GB at 2048^2 that the claim had not
  // booked.
  if (packed) { n += 2 * F * H * 2; }
  if (matrix_cores) {
    if (c.i8_gemm) {
      n += I8GemmContext::forward_bytes_bound((int)jt, (int)F, (int)(2 * F),
                                              (int)H, c.i8_tier == 0);
    }
    if (c.sage.enabled) {
      n += MetalSageAttention::scratch_bytes(nh, nh, (int)jt, (int)jt,
                                             c.head_dim, 64, 32,
                                             c.sage.pv8);
    }
    if (sol) {
      n += MetalSolAttention::scratch_bytes(nh, (int)jt, c.head_dim,
                                            c.sol.key_block, /*nax=*/true);
    }
  }
  return n;
}

// One block's SHARE as tp_qw_ leaves it, off the headers: its six
// projections at the share's rows and columns, in the layout each takes
// -- on the int8 fast / token tier (`fast`, the context load() makes
// when the tier engages) an affine share converted where its columns
// take a group (a w4 one only where Config::i8_hold_w4 holds it, else
// its pack) and a dense one quantized (a dense checkpoint's alone);
// else bf16, or an affine pack's codes, scales and biases -- and its
// four norms whole.
std::size_t
MetalMossImageTransformer::tp_block_bytes_(const MetalLlamaWeights& src,
                                           const Config& c,
                                           const I8GemmContext* fast,
                                           const std::string& pfx)
{
  const TpShare& t = c.tp;
  const std::size_t H = (std::size_t)c.hidden;
  const std::size_t F = (std::size_t)c.ffn_inner;
  const std::size_t hi = (std::size_t)t.heads() * (std::size_t)c.head_dim;
  const std::size_t f = (std::size_t)t.ffn();
  const bool pack = src.info(pfx + "proj.q.scales") != nullptr;
  // [rows] x [cols] of an [N, K] weight.
  auto proj = [&](const std::string& nm, std::size_t N, std::size_t K,
                  std::size_t rows, std::size_t cols) -> std::size_t {
    const auto* ci = src.info(nm + ".weight");
    const auto* si = src.info(nm + ".scales");
    const std::size_t share = rows * cols;
    const int fgc = fast != nullptr ? fast->fast_group((int)cols) : 0;
    const std::size_t layout =
        fgc > 0 ? share + I8GemmContext::fast_scale_bytes(
                              (int)rows, (int)cols, fgc)
                : 0;
    if (ci != nullptr && si != nullptr && si->shape.size() == 2 &&
        si->shape[1] > 0 && N * K > 0) {
      const std::size_t bits = (std::size_t)ci->nbytes * 8 / (N * K);
      const std::size_t g = std::max<std::size_t>(
          K / (std::size_t)si->shape[1], 1);
      const std::size_t as_pack = share * bits / 8 + 2 * (share / g) * 2;
      if (fgc > 0 && g == 64 && fast->fast_converts((int)bits, fgc) &&
          (bits == 8 || c.i8_hold_w4)) {
        return layout;
      }
      return as_pack;
    }
    return fgc > 0 && !pack ? layout : share * 2;
  };
  return proj(pfx + "proj.q", H, H, hi, H) +
         proj(pfx + "proj.k", H, H, hi, H) +
         proj(pfx + "proj.v", H, H, hi, H) +
         proj(pfx + "proj.o", H, H, H, hi) +
         proj(pfx + "proj.ffn_up_gate", 2 * F, H, 2 * f, H) +
         proj(pfx + "proj.ffn_down", H, F, H, f) +
         2 * H * 2 + 2 * (std::size_t)c.head_dim * 2;
}

std::size_t
MetalMossImageTransformer::tp_held_bytes(const std::string& dir,
                                         const Config& c, MetalCompute* mc,
                                         std::size_t* floor)
{
  if (floor != nullptr) { *floor = 0; }
  if (!c.tp.active() || dir.empty()) { return 0; }
  auto wts = MetalLlamaWeights::open_model(dir);
  if (!wts.has_value()) { return 0; }
  // The int8 tier as load() decides it, on a context made as it makes
  // one: no matrix cores, the accurate tier, or a pack of another group,
  // and the shares stay as cut.
  std::unique_ptr<I8GemmContext> i8;
  if (mc != nullptr && c.i8_gemm && c.i8_tier > 0 &&
      mc->supports_matrix_cores() &&
      std::getenv("VPIPE_MOSS_IMAGE_NO_MMA2") == nullptr) {
    i8 = std::make_unique<I8GemmContext>(mc, true, /*bf16=*/true);
    i8->set_tier(c.i8_tier >= 2   ? I8GemmContext::Tier::kToken
                 : c.i8_tier == 1 ? I8GemmContext::Tier::kFast
                                  : I8GemmContext::Tier::kAccurate);
    i8->set_token_rotation(c.i8_tier == 2);
    if (!i8->enabled() || i8->fast_group(c.hidden) <= 0) { i8.reset(); }
  }
  if (i8 != nullptr) {
    const auto* si = wts->info("layers.0.proj.q.scales");
    if (si != nullptr && si->shape.size() == 2 && si->shape[1] > 0 &&
        c.hidden / (int)si->shape[1] != 64) {
      i8.reset();
    }
  }
  // The trunk: every tensor outside the main blocks -- the model-level
  // weights and both text blocks, held whole -- bf16 (narrowed from F32
  // or F16) or a pack's codes as stored. The checkpoint's `rope.bands`
  // is the reference's own table, which this model computes and never
  // loads.
  std::size_t trunk = 0;
  for (const std::string& nm : wts->tensor_names()) {
    if (nm.rfind("layers.", 0) == 0 || nm.rfind("rope.", 0) == 0) {
      continue;
    }
    const auto* ti = wts->info(nm);
    if (ti == nullptr) { continue; }
    const bool fl = ti->dtype == "F32" || ti->dtype == "F16" ||
                    ti->dtype == "BF16";
    std::size_t n = 1;
    for (auto d : ti->shape) { n *= (std::size_t)d; }
    trunk += fl ? n * 2 : (std::size_t)ti->nbytes;
  }
  std::size_t held = trunk, widest = 0;
  for (int L = 0; L < c.n_layers; ++L) {
    const std::size_t b = tp_block_bytes_(*wts, c, i8.get(),
                                          "layers." + std::to_string(L) +
                                              ".");
    held += b;
    widest = std::max(widest, b);
  }
  if (floor != nullptr) { *floor = trunk + 2 * widest; }
  return held;
}

std::size_t
MetalMossImageTransformer::checkpoint_block_bytes_() const
{
  if (_ws == nullptr) { return 0; }
  // A RANK keeps its share of a block, in its tier's layout.
  if (_cfg.tp.active()) {
    return tp_block_bytes_(_ws->src(), _cfg, _fast ? _i8.get() : nullptr,
                           prefix_(0));
  }
  const MetalLlamaWeights& w = _ws->src();
  const std::string pre = prefix_(0);
  std::size_t n = 0;
  for (const std::string& nm : w.tensor_names()) {
    if (nm.rfind(pre, 0) != 0) { continue; }
    const auto* ti = w.info(nm);
    if (ti != nullptr) { n += (std::size_t)ti->nbytes; }
  }
  // WHAT A BLOCK COSTS RESIDENT, not on disk.
  if (_src_f32) { n /= 2; }
  return n;
}

void
MetalMossImageTransformer::set_residency_schedule(int steps)
{
  const std::size_t blk = checkpoint_block_bytes_();
  _wire_block_hint = blk;
  _wire.new_run();
  _resid.set_schedule(steps, _cfg.n_layers, blk, _wire.on(),
                      _mc != nullptr ? _mc->memory_budget()
                                     : metal_compute::MetalCompute::
                                           MemoryBudget{});
}

std::size_t
MetalMossImageTransformer::release_resident_blocks(std::size_t bytes)
{
  const std::size_t freed = _resid.release(bytes, [this]() -> std::size_t {
    return evict_tail_block_();
  });
  if (freed > 0 && _mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalMossImageTransformer: released {} MB of resident blocks "
        "({} left)", freed >> 20, _resid.count()));
  }
  return freed;
}

std::uint64_t
MetalMossImageTransformer::resident_bytes() const
{
  std::uint64_t n = qw_bytes_(_x_in) + _x_in_b.byte_size() +
                    qw_bytes_(_t_in) + _t_in_b.byte_size() +
                    qw_bytes_(_time_1) + _time_1b.byte_size() +
                    qw_bytes_(_time_2) + _time_2b.byte_size() +
                    qw_bytes_(_mod) + _mod_b.byte_size() +
                    qw_bytes_(_head_mod) + _head_mod_b.byte_size() +
                    _final_norm.byte_size() + qw_bytes_(_final_lin);
  for (const TextBlock& b : _text) { n += text_block_bytes_(b); }
  // `_blocks` IS the answer in both modes: preloaded it holds the whole
  // stack, streaming exactly what residency promoted.
  for (const Block& b : _blocks) { n += block_bytes_(b); }
  // And the slot pair while it exists: it stays allocated beside every
  // promoted block, and a holding taken without it dropped two blocks
  // from both ledgers once the promoted set passed the floor.
  n += _slots.held_bytes();
  return n;
}

void
MetalMossImageTransformer::release_forward_scratch()
{
  _w_deq = {};
  if (_i8) { _i8->release_scratch(); }
}

}  // namespace genai
}  // namespace vpipe
