#include "generative-models/fibo2/metal-fibo2-transformer.h"

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

// Narrow a block of checkpoint words to bf16 in place of the caller's
// buffer. Shared by the cached loader and by the streaming refill's
// fill_unservable, which is the whole reason it takes raw pointers.
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

// Load a checkpoint tensor as bf16 (BF16 copied; F16/F32 converted).
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
constexpr const char* kKey = "fibo2-dit/";

}  // namespace

MetalFibo2Transformer::~MetalFibo2Transformer()
{
  // GIVE THE POOL BACK. Freeing a wired buffer unwires it in the kernel,
  // so the machine recovers either way -- but the pool's own counter
  // would not, and a DiT destroyed and reloaded per prompt would leak
  // its whole share of the budget per prompt until nothing could wire.
  if (_wire.on()) {
    wire_fixed_(false);
    for (Block& b : _blocks) { wire_block_(b, false); }
  }
}

// ---- block names ----------------------------------------------------
//
// The streaming stack is noise_refiner THEN layers, the order the
// reference runs them in.

std::string
MetalFibo2Transformer::mod_prefix_(int i) const
{
  if (i < _cfg.n_refiner) {
    return "noise_refiner." + std::to_string(i) + ".";
  }
  return "layers." + std::to_string(i - _cfg.n_refiner) + ".";
}

// ---- residency + the wired pool -------------------------------------

std::size_t
MetalFibo2Transformer::wire_block_(Block& b, bool on)
{
  std::size_t changed = 0;
  bool stop = false;
  auto one = [&](SharedBuffer& p) {
    if (stop) { return; }
    const std::size_t n = _wire.wire_one(_mc, p, on);
    // STOP AT THE FIRST REFUSAL rather than asking for the rest: the
    // pool is full, and the remaining asks would each cost a syscall to
    // be told so. The half-wired remainder is what resident_pages_ then
    // measures, which is the right thing for it to measure.
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
  QWeight* q[] = {&b.qw, &b.kw, &b.vw, &b.ow, &b.w1, &b.w3, &b.w2, &b.ada};
  for (QWeight* w : q) { qw(*w); }
  one(b.nq); one(b.nk);
  one(b.an1); one(b.an2); one(b.fn1); one(b.fn2);
  one(b.ada_b);
  return changed;
}

std::size_t
MetalFibo2Transformer::wire_fixed_(bool on)
{
  if (!_ws) { return 0; }
  std::size_t changed = 0;
  _ws->for_each_weight([&](SharedBuffer& b) {
    changed += _wire.wire_one(_mc, b, on);
  });
  return changed;
}

std::size_t
MetalFibo2Transformer::wire_retry_slack_() const
{
  if (_wire_block_hint > 0) { return _wire_block_hint; }
  if (_resid.count() > 0) {
    return _resid.bytes() / (std::size_t)_resid.count();
  }
  return 0;
}

std::size_t
MetalFibo2Transformer::evict_tail_block_()
{
  // From the END. The next forward starts at block 0, so the deepest
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
MetalFibo2Transformer::resident_pages_(std::size_t* examined,
                                        std::size_t* incore,
                                        std::size_t* paged_out) const
{
  *examined = 0;
  *incore = 0;
  if (paged_out != nullptr) { *paged_out = 0; }
  auto add = [&](const SharedBuffer& p) {
    if (p.byte_size() == 0) { return; }
    // A WIRED BUFFER CANNOT HAVE LEFT RAM, so asking spends the walk to
    // be told what mlock already guarantees.
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
    qw(b.qw); qw(b.kw); qw(b.vw); qw(b.ow);
    qw(b.w1); qw(b.w3); qw(b.w2); qw(b.ada);
    add(b.nq); add(b.nk); add(b.an1); add(b.an2); add(b.fn1); add(b.fn2);
  }
}

// ---- config ---------------------------------------------------------

bool
MetalFibo2Transformer::Config::read_dims(const std::string& dir,
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
  if (!o.contains("_class_name") ||
      o.at("_class_name").as_string("") != "BriaFibo2Transformer2DModel") {
    return false;
  }
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  auto getf = [&](const char* k, float d) {
    return o.contains(k) ? (float)o.at(k).as_real(d) : d;
  };
  out->hidden = geti("dim", out->hidden);
  out->n_heads = geti("n_heads", out->n_heads);
  out->n_layers = geti("n_layers", out->n_layers);
  out->n_refiner = geti("n_refiner_layers", out->n_refiner);
  out->in_channels = geti("in_channels", out->in_channels);
  out->patch = geti("patch_size", out->patch);
  out->cap_dim = geti("cap_feat_dim", out->cap_dim);
  out->norm_eps = getf("norm_eps", out->norm_eps);
  out->rope_theta = getf("rope_theta", out->rope_theta);
  out->t_scale = getf("t_scale", out->t_scale);
  out->perceiver_layers =
      geti("perceiver_num_layers", out->perceiver_layers);
  out->min_gist = geti("min_num_gist_tokens", out->min_gist);
  out->max_gist = geti("max_num_gist_tokens", out->max_gist);
  out->gist_step = geti("gist_step", out->gist_step);
  out->gist_min_text = geti("gist_min_text_len", out->gist_min_text);
  out->gist_max_text = geti("gist_max_text_len", out->gist_max_text);
  auto axes = [&](const char* k, int* dst) {
    if (!o.contains(k)) { return; }
    const FlexData a = o.at(k);
    if (!a.is_array()) { return; }
    auto arr = a.as_array();
    for (std::size_t i = 0; i < 3 && i < arr.size(); ++i) {
      dst[i] = (int)arr.at(i).as_int(dst[i]);
    }
  };
  axes("axes_dims", out->axes);
  axes("axes_lens", out->axes_lens);
  if (o.contains("injection_layer_ids")) {
    const FlexData a = o.at("injection_layer_ids");
    if (!a.is_array()) { return false; }
    out->injection_ids.clear();
    auto arr = a.as_array();
    for (std::size_t i = 0; i < arr.size(); ++i) {
      out->injection_ids.push_back((int)arr.at(i).as_int(-1));
    }
  }
  for (int id : out->injection_ids) {
    if (id < 0 || id >= out->n_layers) { return false; }
  }
  out->head_dim = out->n_heads > 0 ? out->hidden / out->n_heads
                                   : out->head_dim;
  // The SwiGLU width is not in config.json: the reference computes
  // (dim * 8 // 3) // 64 * 64 -- NOT Z-Image's int(dim / 3 * 8).
  out->ffn_inner = (out->hidden * 8 / 3) / 64 * 64;
  out->adaln_dim = std::min(out->hidden, 256);
  if (out->axes[0] + out->axes[1] + out->axes[2] != out->head_dim) {
    return false;
  }
  // The rotary planes: the image sits on max_gist + 1, so the t axis
  // must reach past it.
  if (out->max_gist + 2 > out->axes_lens[0] || out->gist_step <= 0 ||
      out->min_gist <= 0 || out->max_gist < out->min_gist ||
      out->gist_max_text <= out->gist_min_text) {
    return false;
  }
  return out->hidden > 0 && out->n_layers > 0 && out->n_heads > 0 &&
         out->n_refiner > 0 && out->perceiver_layers > 0 &&
         out->n_heads * out->head_dim == out->hidden;
}

// ---- weights --------------------------------------------------------

SharedBuffer
MetalFibo2Transformer::to_elt_(WeightSet& ws, const std::string& name,
                                Retain r)
{
  auto build = [&]() { return to_bf16_(ws.src(), _mc, name); };
  if (r == Retain::Streamed) { return ws.stream_derived(build); }
  return ws.derived(std::string(kKey) + "elt|" + name, build);
}

MetalFibo2Transformer::QWeight
MetalFibo2Transformer::load_qw_(WeightSet& ws, const std::string& name,
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
      // Not convertible: the layouts below, as if never asked.
    }
    // THE SIGNED LAYOUT, converted where the bytes are first the model's
    // own: in the builder of the entries the set keeps for a retained
    // block (so nothing unsigned stays cached beside them), in place in
    // a streamed read. As FLUX.2's.
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
      // Not convertible: the unsigned layout below, as if never asked.
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
  // THE FAST TIER, a dense checkpoint (bf16, or F32 narrowed on the way
  // in): quantized once into the entry a retained block keeps, which
  // then holds no bf16 copy at all. A STREAMED one keeps the bf16 as its
  // refill target and quantizes it after every refill (QWeight::ds), so
  // the slot runs the tier a preloaded block runs and a promotion keeps
  // what the entry keeps. A quantized pack's dense tensor stays dense:
  // the pack left it out on purpose.
  if (fast && _fast && _quant_bits == 0 && ci != nullptr &&
      ci->shape.size() == 2) {
    const int N = (int)ci->shape[0], KK = (int)ci->shape[1];
    const int fgc = _i8->fast_group(KK);
    if (fgc > 0 && r == Retain::Streamed) {
      const std::size_t cb = (std::size_t)N * KK;
      const std::size_t sb = I8GemmContext::fast_scale_bytes(N, KK, fgc);
      QWeight f;
      f.bits = 8; f.gc = fgc; f.n = N; f.k = KK; f.ds = true;
      f.w = to_elt_(ws, name + ".weight", r);
      f.codes = _mc->make_shared_buffer(cb);
      f.cscales = _mc->make_shared_buffer(sb);
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
MetalFibo2Transformer::load_block_(WeightSet& ws, const std::string& p,
                                    bool with_ada, Block& b, Retain r)
{
  // Every block of this family is modulated (with_ada); the flag stays
  // for the shape the streaming machinery shares with Z-Image.
  b.qw = load_qw_(ws, p + "attention.to_q", r, true, with_ada);
  b.kw = load_qw_(ws, p + "attention.to_k", r, true, with_ada);
  b.vw = load_qw_(ws, p + "attention.to_v", r, true, with_ada);
  // to_out is a ModuleList, so the projection is ".0" -- not a typo.
  b.ow = load_qw_(ws, p + "attention.to_out.0", r, true, with_ada);
  b.nq = to_elt_(ws, p + "attention.norm_q.weight", r);
  b.nk = to_elt_(ws, p + "attention.norm_k.weight", r);
  // SwiGLU as Llama names it: w1 gates, w3 is the up projection and w2
  // the down one. Reading w1 and w3 the other way round is a model that
  // still denoises.
  b.w1 = load_qw_(ws, p + "feed_forward.w1", r, true, with_ada);
  b.w3 = load_qw_(ws, p + "feed_forward.w3", r, true, with_ada);
  b.w2 = load_qw_(ws, p + "feed_forward.w2", r, true, with_ada);
  b.an1 = to_elt_(ws, p + "attention_norm1.weight", r);
  b.an2 = to_elt_(ws, p + "attention_norm2.weight", r);
  b.fn1 = to_elt_(ws, p + "ffn_norm1.weight", r);
  b.fn2 = to_elt_(ws, p + "ffn_norm2.weight", r);
  bool ok = !b.qw.empty() && !b.kw.empty() && !b.vw.empty() &&
            !b.ow.empty() && !b.nq.empty() && !b.nk.empty() &&
            !b.w1.empty() && !b.w3.empty() && !b.w2.empty() &&
            !b.an1.empty() && !b.an2.empty() && !b.fn1.empty() &&
            !b.fn2.empty();
  if (with_ada) {
    // `.0` -- a bare Linear with NO SiLU in front of it. The final
    // layer's adaLN is `.1` because that one does have one.
    b.ada = load_qw_(ws, p + "adaLN_modulation.0", r);
    b.ada_b = to_elt_(ws, p + "adaLN_modulation.0.bias", r);
    ok = ok && !b.ada.empty() && !b.ada_b.empty();
  }
  return ok;
}

bool
MetalFibo2Transformer::load_cross_(WeightSet& ws, const std::string& p,
                                    CrossAttn& ca)
{
  const Retain r = Retain::Cached;
  ca.qw = load_qw_(ws, p + "to_q", r);
  ca.kw = load_qw_(ws, p + "to_k", r);
  ca.vw = load_qw_(ws, p + "to_v", r);
  ca.ow = load_qw_(ws, p + "to_out.0", r);
  ca.nq = to_elt_(ws, p + "norm_q.weight", r);
  ca.nk = to_elt_(ws, p + "norm_k.weight", r);
  return !ca.qw.empty() && !ca.kw.empty() && !ca.vw.empty() &&
         !ca.ow.empty() && !ca.nq.empty() && !ca.nk.empty();
}

bool
MetalFibo2Transformer::load_fixed_(WeightSet& ws)
{
  const Retain r = Retain::Cached;
  _x_in = load_qw_(ws, "x_embedder", r);
  _x_in_b = to_elt_(ws, "x_embedder.bias", r);
  _time_1 = load_qw_(ws, "t_embedder.mlp.0", r);
  _time_1b = to_elt_(ws, "t_embedder.mlp.0.bias", r);
  _time_2 = load_qw_(ws, "t_embedder.mlp.2", r);
  _time_2b = to_elt_(ws, "t_embedder.mlp.2.bias", r);
  _final_ada = load_qw_(ws, "final_layer.adaLN_modulation.1", r);
  _final_ada_b = to_elt_(ws, "final_layer.adaLN_modulation.1.bias", r);
  _final_lin = load_qw_(ws, "final_layer.linear", r);
  _final_lin_b = to_elt_(ws, "final_layer.linear.bias", r);
  bool ok = !_x_in.empty() && !_x_in_b.empty() && !_time_1.empty() &&
            !_time_1b.empty() && !_time_2.empty() && !_time_2b.empty() &&
            !_final_ada.empty() && !_final_ada_b.empty() &&
            !_final_lin.empty() && !_final_lin_b.empty();
  // THE PERCEIVER, which turns text bundle 0 into the gist tokens.
  _queries = to_elt_(ws, "perceiver.latent_queries", r);
  _p_norm_out = to_elt_(ws, "perceiver.norm_out.weight", r);
  ok = ok && !_queries.empty() && !_p_norm_out.empty() &&
       _queries.byte_size() >= (std::size_t)_cfg.max_gist * _cfg.hidden * 2;
  _perceiver.clear();
  _perceiver.resize((std::size_t)_cfg.perceiver_layers);
  for (int i = 0; ok && i < _cfg.perceiver_layers; ++i) {
    PerceiverLayer& l = _perceiver[(std::size_t)i];
    const std::string p = "perceiver.layers." + std::to_string(i) + ".";
    l.n_ca = to_elt_(ws, p + "cross_attn_norm.weight", r);
    l.n_sa = to_elt_(ws, p + "self_attn_norm.weight", r);
    l.n_ff = to_elt_(ws, p + "ffn_norm.weight", r);
    ok = !l.n_ca.empty() && !l.n_sa.empty() && !l.n_ff.empty() &&
         load_cross_(ws, p + "cross_attention.", l.ca);
    const std::string sp = p + "self_attention.";
    l.sq = load_qw_(ws, sp + "to_q", r);
    l.sk = load_qw_(ws, sp + "to_k", r);
    l.sv = load_qw_(ws, sp + "to_v", r);
    l.so = load_qw_(ws, sp + "to_out.0", r);
    l.snq = to_elt_(ws, sp + "norm_q.weight", r);
    l.snk = to_elt_(ws, sp + "norm_k.weight", r);
    l.w1 = load_qw_(ws, p + "ffn.w1", r);
    l.w3 = load_qw_(ws, p + "ffn.w3", r);
    l.w2 = load_qw_(ws, p + "ffn.w2", r);
    ok = ok && !l.sq.empty() && !l.sk.empty() && !l.sv.empty() &&
         !l.so.empty() && !l.snq.empty() && !l.snk.empty() &&
         !l.w1.empty() && !l.w3.empty() && !l.w2.empty();
  }
  // THE INJECTION SUBLAYERS, held in both modes: five blocks of a
  // uniform streamed stack carry them, and a slot refills only what every
  // block has.
  _inject.clear();
  for (std::size_t i = 0; ok && i < _cfg.injection_ids.size(); ++i) {
    Inject in;
    in.layer = _cfg.injection_ids[i];
    const std::string p = "layers." + std::to_string(in.layer) + ".";
    in.n1 = to_elt_(ws, p + "cross_attn_norm1.weight", r);
    in.n2 = to_elt_(ws, p + "cross_attn_norm2.weight", r);
    ok = !in.n1.empty() && !in.n2.empty() &&
         load_cross_(ws, p + "cross_attention.", in.ca);
    // The gate is ONE scalar; the gated residual kernel reads a row of
    // H, tanh'd in the kernel, so the scalar is broadcast once here.
    SharedBuffer g = ws.read(p + "cross_attn_gate", _mc,
                           WeightSet::Residency::Copied);
    const auto* gi = ws.src().info(p + "cross_attn_gate");
    in.gate = _mc->make_shared_buffer((std::size_t)_cfg.hidden * 2);
    if (g.empty() || gi == nullptr || in.gate.empty()) {
      ok = false;
      break;
    }
    std::uint16_t v = 0;
    if (!narrow_to_bf16_(g.contents(), gi->dtype, 1, &v)) {
      ok = false;
      break;
    }
    auto* d = static_cast<std::uint16_t*>(in.gate.contents());
    for (int k = 0; k < _cfg.hidden; ++k) { d[k] = v; }
    _inject.push_back(std::move(in));
  }
  return ok;
}

std::size_t
MetalFibo2Transformer::qw_bytes_(const QWeight& w)
{
  if (w.quantized) {
    return w.codes.byte_size() + w.scales.byte_size() + w.qbias.byte_size() +
           w.cscales.byte_size() + (w.ds ? w.w.byte_size() : 0);
  }
  return w.w.byte_size();
}

std::size_t
MetalFibo2Transformer::block_bytes_(const Block& b)
{
  return qw_bytes_(b.qw) + qw_bytes_(b.kw) + qw_bytes_(b.vw) +
         qw_bytes_(b.ow) + b.nq.byte_size() + b.nk.byte_size() +
         qw_bytes_(b.w1) + qw_bytes_(b.w3) + qw_bytes_(b.w2) +
         b.an1.byte_size() + b.an2.byte_size() + b.fn1.byte_size() +
         b.fn2.byte_size() + qw_bytes_(b.ada) + b.ada_b.byte_size();
}

void
MetalFibo2Transformer::each_block_tensor_(
    int L, Block& b, const BlockSlots<Block>::TensorFn& fn) const
{
  const std::string p = mod_prefix_(L);
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
  qw(b.qw, "attention.to_q");
  qw(b.kw, "attention.to_k");
  qw(b.vw, "attention.to_v");
  qw(b.ow, "attention.to_out.0");
  fn(p + "attention.norm_q.weight", b.nq, Placement::kBf16);
  fn(p + "attention.norm_k.weight", b.nk, Placement::kBf16);
  qw(b.w1, "feed_forward.w1");
  qw(b.w3, "feed_forward.w3");
  qw(b.w2, "feed_forward.w2");
  fn(p + "attention_norm1.weight", b.an1, Placement::kBf16);
  fn(p + "attention_norm2.weight", b.an2, Placement::kBf16);
  fn(p + "ffn_norm1.weight", b.fn1, Placement::kBf16);
  fn(p + "ffn_norm2.weight", b.fn2, Placement::kBf16);
  qw(b.ada, "adaLN_modulation.0");
  fn(p + "adaLN_modulation.0.bias", b.ada_b, Placement::kBf16);
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
MetalFibo2Transformer::clone_block_(const Block& src, Block& dst,
                                     bool copy) const
{
  bool ok = true;
  // A PROMOTION keeps what a preloaded block keeps (see
  // kept_block_bytes_): a w4 pack whose layout is held (Config::
  // i8_hold_w4) is converted into it here, all of a block's in one
  // command buffer, by the call the preloaded entry's builder makes.
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
  cq(src.ow, dst.ow);
  ok = clone_buf_(_mc, src.nq, dst.nq, copy) && ok;
  ok = clone_buf_(_mc, src.nk, dst.nk, copy) && ok;
  cq(src.w1, dst.w1); cq(src.w3, dst.w3); cq(src.w2, dst.w2);
  ok = clone_buf_(_mc, src.an1, dst.an1, copy) && ok;
  ok = clone_buf_(_mc, src.an2, dst.an2, copy) && ok;
  ok = clone_buf_(_mc, src.fn1, dst.fn1, copy) && ok;
  ok = clone_buf_(_mc, src.fn2, dst.fn2, copy) && ok;
  cq(src.ada, dst.ada);
  ok = clone_buf_(_mc, src.ada_b, dst.ada_b, copy) && ok;
  if (cs.valid()) {
    ce.end();
    ok = cs.commit().wait_ok(nullptr) && ok;
  }
  return ok;
}

bool
MetalFibo2Transformer::keep_block_(Block& b) const
{
  const bool hold = _cfg.i8_hold_w4 && _i8 != nullptr;
  CommandStream cs;
  bool ok = true;
  std::vector<std::pair<QWeight*, QWeight>> held;
  {
    ComputeEncoder e;
    for (QWeight* w : {&b.qw, &b.kw, &b.vw, &b.ow, &b.w1, &b.w3, &b.w2}) {
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

std::size_t
MetalFibo2Transformer::kept_block_bytes_(const Block& b) const
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
  return kq(b.qw) + kq(b.kw) + kq(b.vw) + kq(b.ow) + b.nq.byte_size() +
         b.nk.byte_size() + kq(b.w1) + kq(b.w3) + kq(b.w2) +
         b.an1.byte_size() + b.an2.byte_size() + b.fn1.byte_size() +
         b.fn2.byte_size() + kq(b.ada) + b.ada_b.byte_size();
}

SharedBuffer
MetalFibo2Transformer::rebuild_one_(const std::string& nm, Placement how)
{
  if (_ws == nullptr) { return {}; }
  if (how == Placement::kRaw) {
    return _ws->stream_tensor(nm, _mc, WeightSet::Residency::Copied);
  }
  return _ws->stream_derived([&]() { return to_bf16_(_ws->src(), _mc, nm); });
}

bool
MetalFibo2Transformer::fill_narrow_(const std::string& nm,
                                     const SharedBuffer& dst)
{
  // THE TURBO CHECKPOINT IS F32. Its bytes are twice the width of every
  // buffer this model reads, so the raw read has nowhere to put them --
  // which is exactly what BlockSlots' fill_unservable exists for. The
  // alternative it falls back to (rebuild_one_) ALLOCATES a block's
  // worth of buffers per block per forward, which is the cost the whole
  // slot mechanism was built to remove.
  //
  // One scratch for the run, grown to the largest tensor asked for: the
  // blocks are uniform, so it settles on the first one.
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
MetalFibo2Transformer::sign_(QWeight& w) const
{
  // A weight this model keeps unsigned (dense, the layout off) has
  // nothing to do. A signed one in a slot was just REFILLED raw, whatever
  // its flag says from the build, so it converts unconditionally.
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
// blocks at w8 and the rest at w4, adaLN included -- is rebuilt rather
// than refilled (BlockSlots::Ops::same_layout): the refill would place
// the codes and keep the slot's width, fast-tier scales and signed flag.
bool
MetalFibo2Transformer::same_layout_(int L, const Block& b) const
{
  if (_quant_bits == 0 || _ws == nullptr) { return true; }
  const std::string p = mod_prefix_(L);
  const std::pair<const QWeight*, const char*> all[] = {
      {&b.qw, "attention.to_q"}, {&b.kw, "attention.to_k"},
      {&b.vw, "attention.to_v"}, {&b.ow, "attention.to_out.0"},
      {&b.w1, "feed_forward.w1"}, {&b.w3, "feed_forward.w3"},
      {&b.w2, "feed_forward.w2"}, {&b.ada, "adaLN_modulation.0"}};
  for (const auto& [w, nm] : all) {
    if (stored_affine_bits(*_ws, p + nm, _quant_group) !=
        (w->quantized ? w->bits : 0)) {
      return false;
    }
  }
  return true;
}

int
MetalFibo2Transformer::fast_weights(int bits) const
{
  int n = 0;
  for (const Block& b : _blocks) {
    for (const QWeight* w : {&b.qw, &b.kw, &b.vw, &b.ow, &b.w1, &b.w3,
                             &b.w2, &b.ada}) {
      n += w->quantized && w->gc > 0 && w->bits == bits;
    }
  }
  return n;
}

bool
MetalFibo2Transformer::encode_fast_(ComputeEncoder& e, QWeight& w) const
{
  // A w4 pack kept as stored converts per GEMM (fast_operand_), not here.
  if (w.gc <= 0 || w.pc) { return true; }
  if (!_i8 || w.codes.empty() || w.cscales.empty()) { return false; }
  // A dense source: quantized out of the bf16 the refill landed, by the
  // call the preloaded entry's builder makes.
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
MetalFibo2Transformer::fast_(QWeight& w) const
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
MetalFibo2Transformer::fast_operand_(ComputeEncoder& enc, const QWeight& w,
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
MetalFibo2Transformer::fast_dense_(ComputeEncoder& enc, const QWeight& w,
                                    const SharedBuffer& out, int N, int K) const
{
  const SharedBuffer* c = nullptr;
  const SharedBuffer* cs = nullptr;
  return fast_operand_(enc, w, &c, &cs) &&
         _i8->encode_dequant_fast(enc, *c, *cs, w.gc, out, N, K);
}

const MetalFibo2Transformer::QWeight*
MetalFibo2Transformer::signed_dense_(ComputeEncoder& enc, const QWeight& w,
                                      QWeight& view, int N, int K) const
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
MetalFibo2Transformer::configure_slots_()
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
    return _ws != nullptr &&
           load_block_(*_ws, mod_prefix_(L), true, b, Retain::Streamed);
  };
  ops.clone = [this](const Block& s, Block& d, bool copy) {
    return clone_block_(s, d, copy);
  };
  ops.bytes = [](const Block& b) { return block_bytes_(b); };
  ops.empty = [](const Block& b) { return b.qw.empty(); };
  // Every quantized projection just landed RAW, and converts to its
  // layout here, on the reader thread, under the GPU's work: the fast
  // layouts (a pack's codes, a dense source) in ONE command buffer for
  // the block on the reader's own stream, the signed layout on the CPU.
  // The adaLN, which load_block_ keeps as stored, is left as read.
  ops.post_refill = [this](int, Block& b) {
    QWeight* all[] = {&b.qw, &b.kw, &b.vw, &b.ow, &b.w1, &b.w3, &b.w2};
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
  _slots.configure(_mc, std::move(ops), "MetalFibo2Transformer",
                   "VPIPE_FIBO2_NO_SLOTS");
}

// ---- load -----------------------------------------------------------

std::unique_ptr<MetalFibo2Transformer>
MetalFibo2Transformer::load(const std::string& model_dir, MetalCompute* mc,
                             const Config& cfg, bool stream_blocks,
                             const std::vector<LoraSpec>& loras)
{
  return load(WeightSet::open(model_dir, nullptr), mc, cfg, stream_blocks,
              loras);
}

std::unique_ptr<MetalFibo2Transformer>
MetalFibo2Transformer::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                             const Config& cfg, bool stream_blocks,
                             const std::vector<LoraSpec>& loras)
{
  if (ws == nullptr || mc == nullptr) { return nullptr; }
  std::unique_ptr<MetalFibo2Transformer> m(new MetalFibo2Transformer());
  m->_mc = mc;
  m->_cfg = cfg;
  m->_ws = std::move(ws);
  m->_stream_blocks = stream_blocks;

  // Quantization is read off the checkpoint, not configured: a pack
  // carries `.scales` beside its codes and a dense one does not.
  {
    const MetalLlamaWeights& w = m->_ws->src();
    const std::string p0 = m->mod_prefix_(0);
    const auto* si = w.info(p0 + "attention.to_q.scales");
    const auto* ci = w.info(p0 + "attention.to_q.weight");
    if (si != nullptr && ci != nullptr && si->shape.size() == 2 &&
        ci->shape.size() == 2 && si->shape[1] > 0) {
      m->_quant_group = (int)(m->_cfg.hidden / si->shape[1]);
      if (m->_quant_group <= 0) { m->_quant_group = 64; }
      const long K = si->shape[1] * (long)m->_quant_group;
      m->_quant_bits = K > 0 ? (int)(ci->shape[1] * 32 / K) : 0;
      if (m->_quant_bits != 4 && m->_quant_bits != 8) { m->_quant_bits = 0; }
    }
    // Whether the checkpoint is F32, which decides how a streamed block
    // lands. Turbo is; the undistilled base is bf16.
    if (m->_quant_bits == 0 && ci != nullptr) {
      m->_src_f32 = ci->dtype == "F32";
    }
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
  // THE WIDER TILES, which are what a box WITHOUT matrix cores runs for
  // every GEMM in the model -- the M4 series is where this family is
  // most likely to be run, so this is not a corner.
  m->_fn_gemm_bm64 = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_gemm_bm64bn64 = m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  m->_fn_ln_plain = m->_lib_elt.function("layer_norm_plain_f16");
  m->_fn_adaln = m->_lib_elt.function("adaln_modulate_f16");
  m->_fn_adaln4 = m->_lib_elt.function("adaln_modulate_v4_f16");
  m->_fn_gated_tanh = m->_lib_elt.function("gated_residual_tanh_f16");
  m->_fn_gated_tanh4 = m->_lib_elt.function("gated_residual_tanh_v4_f16");
  m->_fn_swiglu = m->_lib_elt.function("swiglu_f16");
  m->_fn_swiglu4 = m->_lib_elt.function("swiglu_v4_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_residual4 = m->_lib_elt.function("residual_add_v4_f16");
  m->_fn_copy_rows = m->_lib_elt.function("copy_rect_f16");
  m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_silu = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_rms = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_tr_rope = m->_lib_rope.function("transpose_rope_pair_ftab_f16");
  // THE QUANTIZED ENTRY POINTS ARE NAMED BY THEIR GROUP, and the group
  // is read off the checkpoint above -- `affine_qmm_steel_w4g64`, not a
  // `_f16` suffix (that suffix belongs to the kernels whose MSL name
  // carries it, e.g. dense_gemm_t_f16; these do not). Both bit widths,
  // because a mixed pack holds both. Only loaded for a quantized
  // checkpoint, so a dense one cannot be refused for their absence.
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

  // VALIDATE. An unvalidated ComputeFunction encodes NOTHING and poisons
  // its dispatch, which in a 34-block stack is a picture, not a failure --
  // and before ComputeEncoder was taught to refuse it, it was worse than
  // that: the previous op's kernel ran again in its place.
  if (!m->_fn_gemm_t.valid() || !m->_fn_ln_plain.valid() ||
      !m->_fn_adaln.valid() || !m->_fn_gated_tanh.valid() ||
      !m->_fn_swiglu.valid() || !m->_fn_rms.valid() ||
      !m->_fn_tr_rope.valid() || !m->_fn_residual.valid() ||
      !m->_fn_silu.valid() || !m->_fn_transpose.valid() ||
      !m->_fn_copy_rows.valid() || !m->_fn_bias_add.valid()) {
    if (mc->session() != nullptr) {
      mc->session()->error(fmt("MetalFibo2Transformer: a required kernel "
                               "did not build"));
    }
    return nullptr;
  }
  // A QUANTIZED CHECKPOINT WHOSE GEMM DID NOT LOAD IS A REFUSAL, not a
  // degraded run. `set_function` on an invalid handle leaves the
  // PREVIOUS kernel bound and the dispatch that follows runs THAT one
  // over 4-bit codes -- which is how a mis-spelt entry point surfaced
  // as a GPU hang rather than as a wrong picture. Nothing downstream
  // can recover from it, so say which group failed and stop.
  if (m->_quant_bits > 0 &&
      (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid())) {
    if (mc->session() != nullptr) {
      mc->session()->error(fmt(
          "MetalFibo2Transformer: no quantized GEMM for group {} -- this "
          "checkpoint cannot be run (re-quantize at group 64)",
          m->_quant_group));
    }
    return nullptr;
  }
  m->_steel_attn_ok = m->_lib_attn.valid();
  m->_use_attn_nax = m->_steel_attn_ok && m->_lib_attn_nax.valid() &&
                     mc->supports_matrix_cores() &&
                     std::getenv("VPIPE_FIBO2_NO_NAX_ATTN") == nullptr;
  m->_use_mma2 = mc->supports_matrix_cores() && m->_fn_gemm_mma.valid() &&
                 std::getenv("VPIPE_FIBO2_NO_MMA2") == nullptr;
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm,
                                              /*bf16=*/true);
    i8->set_tier(cfg.i8_tier >= 2   ? I8GemmContext::Tier::kToken
                 : cfg.i8_tier == 1 ? I8GemmContext::Tier::kFast
                                    : I8GemmContext::Tier::kAccurate);
    i8->set_token_rotation(cfg.i8_tier == 2);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }
  if (const char* e = std::getenv("VPIPE_FIBO2_SOL")) {
    m->_cfg.sol.enabled = std::atoi(e) != 0;
  }
  // HIGHER tau keeps FEWER blocks: it is a threshold, not a budget.
  if (const char* e = std::getenv("VPIPE_FIBO2_SOL_TAU")) {
    m->_cfg.sol.tau = (float)std::atof(e);
  }
  if (const char* e = std::getenv("VPIPE_FIBO2_ANE")) {
    m->_cfg.ane_ffn = std::atoi(e) != 0;
  }
  if (const char* e = std::getenv("VPIPE_FIBO2_ANE_ROWS")) {
    m->_cfg.ane_rows = std::atoi(e);
  }
  if (const char* e = std::getenv("VPIPE_FIBO2_ANE_QKV")) {
    m->_cfg.ane_qkv = std::atoi(e) != 0;
  }
  if (m->_cfg.sol.enabled) {
    std::string serr;
    m->_sol = MetalSolAttention::load(mc, /*bf16=*/true, &serr);
    if (m->_sol == nullptr && mc->session() != nullptr) {
      mc->session()->log_normal(fmt(
          "MetalFibo2Transformer: sol_attn unavailable -- {}",
          serr.empty() ? "kernels did not load" : serr));
    }
  }
  {
    if (const char* e = std::getenv("VPIPE_FIBO2_SAGE")) {
      m->_cfg.sage.enabled = std::atoi(e) != 0;
    }
    bool sage_fatal = false;
    m->_sage = MetalSageAttention::load_for_model(
        mc, /*bf16=*/true, m->_cfg.sage, "MetalFibo2Transformer",
        &sage_fatal);
    if (sage_fatal) { return nullptr; }
    if (m->_sage && !m->_use_attn_nax) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt(
            "MetalFibo2Transformer: sage_attn is off -- the matrix-core "
            "flash entry is unavailable here"));
      }
      m->_sage.reset();
    }
    m->_sage_reserve =
        std::getenv("VPIPE_FIBO2_SAGE_NO_RESERVE") == nullptr;
  }
  if (const char* e = std::getenv("VPIPE_FIBO2_NO_FOLD")) {
    m->_fold_mod = std::atoi(e) == 0;
  }
  if (const char* t = std::getenv("VPIPE_FIBO2_GEMM_TILE")) {
    m->_gemm_tile = std::atoi(t);
  }
  if (m->_gemm_tile == 2 && !m->_fn_gemm_bm64bn64.valid()) {
    m->_gemm_tile = 1;
  }
  if (m->_gemm_tile == 1 && !m->_fn_gemm_bm64.valid()) { m->_gemm_tile = 0; }

  if (!m->load_fixed_(*m->_ws)) {
    if (mc->session() != nullptr) {
      mc->session()->error(fmt("MetalFibo2Transformer: the model-level "
                               "weights are incomplete"));
    }
    return nullptr;
  }

  // THE SIGNED LAYOUT for the block projections (see _sgn): decided
  // before any block loads. Its GEMM runs
  // only on the matrix-core route, and the ANE tiers' stagers read the
  // codes unsigned.
  m->_sgn = m->_i8 && m->_use_mma2 && m->_quant_bits > 0 &&
            m->_quant_group == 64 &&
            !m->_cfg.ane_ffn && !m->_cfg.ane_qkv &&
            (m->_i8->signed_available(8) || m->_i8->signed_available(4));
  // THE FAST OR TOKEN TIER for the modulated blocks (see _fast): g64
  // packs (w8, w4, or both in a mixed one) and dense checkpoints, on the
  // matrix-core route, away from the ANE tiers. Its weights never meet
  // the signed layout (sign_ leaves them), so the two decide
  // independently.
  m->_fast = m->_i8 && m->_use_mma2 &&
             m->_i8->tier() != I8GemmContext::Tier::kAccurate &&
             (m->_quant_bits == 0 || m->_quant_group == 64) &&
             !m->_cfg.ane_ffn && !m->_cfg.ane_qkv &&
             m->_i8->fast_group(m->_cfg.hidden) > 0;
  if (m->_fast && mc->session() != nullptr) {
    const int fg = m->_i8->fast_group(m->_cfg.hidden);
    mc->session()->info(fmt(
        "MetalFibo2Transformer: the modulated blocks' projections take "
        "the {}{}",
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
        "MetalFibo2Transformer: the quantized block projections take the "
        "signed int8 layout"));
  }

  m->_wire.open(mc);
  const int NM = m->_cfg.n_mod_blocks();
  if (stream_blocks) {
    m->configure_slots_();
    // Sized to the stack so a PROMOTED block has somewhere to land; an
    // unfilled entry reads as empty, which is how the forward tells a
    // resident block from one that is still streamed.
    m->_blocks.resize((std::size_t)NM);
  } else {
    m->_blocks.resize((std::size_t)NM);
    for (int L = 0; L < NM; ++L) {
      if (!m->load_block_(*m->_ws, m->mod_prefix_(L), /*with_ada=*/true,
                          m->_blocks[(std::size_t)L], Retain::Cached)) {
        if (mc->session() != nullptr) {
          mc->session()->error(fmt("MetalFibo2Transformer: block {} is "
                                   "incomplete", L));
        }
        return nullptr;
      }
    }
  }
  // AFTER the blocks, because binding needs the projections' real
  // dimensions and a streamed stack has none until its slots exist --
  // the factors are per NAME, so they bind the same either way.
  if (!m->bind_loras_(loras) && mc->session() != nullptr) {
    mc->session()->warn(fmt(
        "MetalFibo2Transformer: the LoRA applier did not load; the "
        "adapters are ignored and the base model runs"));
  }
  if (mc->session() != nullptr) {
    mc->session()->log_normal(fmt(
        "MetalFibo2Transformer: {} blocks ({} noise + {} main, {} "
        "injecting), {}-layer Perceiver, hidden {}, {} heads x {}, ffn {}, "
        "{}{}{}{}",
        m->_cfg.n_blocks(), m->_cfg.n_refiner, m->_cfg.n_layers,
        m->_cfg.injection_ids.size(), m->_cfg.perceiver_layers,
        m->_cfg.hidden, m->_cfg.n_heads, m->_cfg.head_dim, m->_cfg.ffn_inner,
        m->_quant_bits > 0
            ? "w" + std::to_string(m->_quant_bits) + " g" +
                  std::to_string(m->_quant_group)
            : std::string(m->_src_f32 ? "f32 -> bf16" : "bf16"),
        stream_blocks ? ", streaming" : "",
        m->_use_mma2 ? ", matmul2d" : "",
        // WHETHER THE TIER TOOK, not whether it was asked for -- and the
        // two spelled apart, because they are different answers and
        // silence could mean either. i8_gemm is opt-in and declines on
        // its own for a machine without matrix cores or a metallib
        // without its entries, so a run that asked for it and did not
        // get it looked exactly like a run that never asked: the load
        // line, the block count and the progress beat all read the same,
        // and the only evidence was a clock that did not move.
        m->_i8            ? ", i8"
        : m->_cfg.i8_gemm ? ", i8 REQUESTED BUT UNAVAILABLE"
                          : ""));
  }
  return m;
}

// The timestep sinusoid, on the host, for a model input `t` -- sigma *
// t_scale for the image rows, 0 for the clean ones. Cos occupies the
// first half and sin the second. Formed as the reference forms it, in
// f32 throughout (the frequencies, their product with t), so only the
// cos / sin themselves are computed wider.
std::vector<float>
MetalFibo2Transformer::time_proj_(float t) const
{
  const int D = _cfg.time_freq;
  const int half = D / 2;
  std::vector<float> out((std::size_t)D, 0.0f);
  const float nl = (float)(-std::log(10000.0));
  for (int i = 0; i < half; ++i) {
    const float f = std::exp(nl * (float)i / (float)half);
    const float a = t * f;
    out[(std::size_t)i] = (float)std::cos((double)a);
    out[(std::size_t)(half + i)] = (float)std::sin((double)a);
  }
  return out;
}

// ONE joint table for the whole sequence; the noise refiner reads its
// prefix (the image and context rows lead the sequence).
//
// f32, as the reference applies it. The ANGLE is rounded to f32 before
// the cos / sin, as the reference's float64 outer product cast to f32
// is: at the image's plane (t = max_gist + 1 = 513) the rounding moves
// an angle by ~3e-5, which a wider angle would not reproduce.
void
MetalFibo2Transformer::build_rope_(const fibo2::Layout& lay,
                                    SharedBuffer* cos_out,
                                    SharedBuffer* sin_out) const
{
  const int HD = _cfg.head_dim;
  const int pairs = HD / 2;
  const int seq = lay.joint_len();
  *cos_out = _mc->make_shared_buffer((std::size_t)seq * HD * sizeof(float));
  *sin_out = _mc->make_shared_buffer((std::size_t)seq * HD * sizeof(float));
  if (cos_out->empty() || sin_out->empty()) { return; }
  auto* c = static_cast<float*>(cos_out->contents());
  auto* s = static_cast<float*>(sin_out->contents());
  const double theta = (double)_cfg.rope_theta;
  std::vector<double> pair_freq((std::size_t)pairs);
  std::vector<int> pair_axis((std::size_t)pairs);
  {
    int pair = 0;
    for (int a = 0; a < 3; ++a) {
      const int adim = _cfg.axes[a];
      for (int j = 0; j < adim / 2 && pair < pairs; ++j, ++pair) {
        pair_freq[(std::size_t)pair] =
            1.0 / std::pow(theta, (double)(2 * j) / (double)adim);
        pair_axis[(std::size_t)pair] = a;
      }
    }
  }
  for (int t = 0; t < seq; ++t) {
    const double pos[3] = {(double)lay.pos_t[(std::size_t)t],
                           (double)lay.pos_h[(std::size_t)t],
                           (double)lay.pos_w[(std::size_t)t]};
    for (int pair = 0; pair < pairs; ++pair) {
      const float ang = (float)(pos[pair_axis[(std::size_t)pair]] *
                                pair_freq[(std::size_t)pair]);
      const float cb = (float)std::cos((double)ang);
      const float sb = (float)std::sin((double)ang);
      // Written TWICE per pair: the rotation is over INTERLEAVED
      // complex pairs (x[2k], x[2k+1]), and the kernel reads one table
      // entry per element.
      c[(std::size_t)t * HD + 2 * pair] = cb;
      c[(std::size_t)t * HD + 2 * pair + 1] = cb;
      s[(std::size_t)t * HD + 2 * pair] = sb;
      s[(std::size_t)t * HD + 2 * pair + 1] = sb;
    }
  }
}

// ---- the text ---------------------------------------------------------

bool
MetalFibo2Transformer::set_text(const SharedBuffer& bundles, int text_len,
                                std::string* err, int slot)
{
  if (slot < 0 || slot >= kTextSlots) {
    if (err != nullptr) {
      *err = "fibo2 set_text: no text slot " + std::to_string(slot);
    }
    return false;
  }
  const std::size_t need = (std::size_t)_cfg.n_bundles() *
                           (std::size_t)(text_len > 0 ? text_len : 0) *
                           (std::size_t)_cfg.cap_dim * 2;
  if (text_len <= 0 || bundles.byte_size() < need) {
    if (err != nullptr) {
      *err = "fibo2 set_text: expected " + std::to_string(_cfg.n_bundles()) +
             " bundles of " + std::to_string(text_len) + " x " +
             std::to_string(_cfg.cap_dim) + " bf16";
    }
    return false;
  }
  TextState& ts = _txt[slot];
  // A handle on the caller's buffer, not a copy: a subview shares its
  // allocation.
  ts.text = bundles.subview(0, need);
  if (ts.text.empty()) {
    if (err != nullptr) { *err = "fibo2 set_text: no handle on the text"; }
    return false;
  }
  ts.text_len = text_len;
  ts.dirty = true;
  ts.gist_n = fibo2::gist_count(text_len, _cfg.min_gist, _cfg.max_gist,
                                _cfg.gist_step, _cfg.gist_min_text,
                                _cfg.gist_max_text);
  ts.gist = {};
  ts.inj_k.clear();
  ts.inj_v.clear();
  return true;
}

void
MetalFibo2Transformer::clear_text(int slot)
{
  if (slot >= 0 && slot < kTextSlots) { _txt[slot] = TextState{}; }
}

std::size_t
MetalFibo2Transformer::TextState::bytes() const
{
  std::size_t n = gist.byte_size();
  for (const SharedBuffer& b : inj_k) { n += b.byte_size(); }
  for (const SharedBuffer& b : inj_v) { n += b.byte_size(); }
  return n;
}

// ---- forward --------------------------------------------------------

SharedBuffer
MetalFibo2Transformer::forward(const Request& req, std::string* err)
{
  _fast_failed = false;   // per forward: a latch, not a verdict on the model
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return SharedBuffer{};
  };
  if (req.layout == nullptr || req.latents == nullptr) {
    return fail("fibo2 forward: missing input");
  }
  if (req.text_slot < 0 || req.text_slot >= kTextSlots) {
    return fail("fibo2 forward: no text slot " +
                std::to_string(req.text_slot));
  }
  TextState& ts = _txt[req.text_slot];
  if (ts.text_len <= 0 || (ts.dirty && ts.text.empty())) {
    return fail("fibo2 forward: no text was set (set_text)");
  }
  const fibo2::Layout& lay = *req.layout;
  const int H = _cfg.hidden, NH = _cfg.n_heads, Hd = _cfg.head_dim;
  const int FFI = _cfg.ffn_inner, PD = _cfg.patch_dim();
  const int JT = lay.joint_len(), RT = lay.refiner_len();
  const int IMG = lay.img_tokens, G = lay.gist, T = ts.text_len;
  const int CAP = _cfg.cap_dim;
  const int NM = _cfg.n_mod_blocks();
  if (JT <= 0 || IMG <= 0 || NH * Hd != H) {
    return fail("fibo2 forward: degenerate geometry");
  }
  // THE GIST COUNT IS THE TEXT'S, and the layout has to agree with it:
  // the gist rows are a function of the text alone.
  if (G != ts.gist_n || (int)lay.pos_t.size() != JT) {
    return fail("fibo2 forward: the layout's gist count (" +
                std::to_string(G) + ") is not the text's (" +
                std::to_string(ts.gist_n) + ")");
  }
  if (!_steel_attn_ok) {
    return fail("fibo2 forward: the steel attention kernel is "
                "unavailable");
  }
  // THE ROTARY TABLES ARE BOUNDED. The reference indexes a precomputed
  // table of `axes_lens` rows per axis, so a position past the end is
  // an index error there and would be a silently plausible angle here.
  for (int t = 0; t < JT; ++t) {
    if (lay.pos_t[(std::size_t)t] >= _cfg.axes_lens[0] ||
        lay.pos_h[(std::size_t)t] >= _cfg.axes_lens[1] ||
        lay.pos_w[(std::size_t)t] >= _cfg.axes_lens[2]) {
      return fail("fibo2 forward: a rotary position is past the "
                  "checkpoint's axes_lens -- the picture or the prompt is "
                  "larger than this model was built for");
    }
  }

  // LLM-lane perf event: one DiT forward per sampler step, two per step
  // under CFG. Value = the joint sequence length, which is what the
  // cost actually scales with.
  PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDit,
                     kPerfLlmDitBegin, (std::uint64_t)JT);

  MetalCompute* mc = _mc;
  const float eps = _cfg.norm_eps;

  auto buf = [&](std::size_t elems) {
    return mc->make_shared_buffer(elems * 2);
  };
  // Peak scratch is ~10 [JT,H] planes plus 2 [JT,FFI] ones: 0.5 GB at
  // 1024^2 and 2 GB at 2048^2, which is the number a memory plan has to
  // carry for this family beside the weights.
  SharedBuffer joint = buf((std::size_t)JT * H);
  SharedBuffer nrm = buf((std::size_t)JT * H);
  SharedBuffer ob = buf((std::size_t)JT * H);
  SharedBuffer qb = buf((std::size_t)JT * H);
  SharedBuffer kb = buf((std::size_t)JT * H);
  SharedBuffer vb = buf((std::size_t)JT * H);
  SharedBuffer qh = buf((std::size_t)JT * H);
  SharedBuffer kh = buf((std::size_t)JT * H);
  SharedBuffer vh = buf((std::size_t)JT * H);
  SharedBuffer ao = buf((std::size_t)JT * H);
  SharedBuffer ffg = buf((std::size_t)JT * FFI);
  SharedBuffer ffu = buf((std::size_t)JT * FFI);
  // TWO timestep rows throughout: row 0 the image's (sigma * t_scale),
  // row 1 every clean row's (t = 0) -- the per-token modulation.
  SharedBuffer te1 = buf((std::size_t)2 * _cfg.time_mid);
  SharedBuffer temb = buf((std::size_t)2 * _cfg.adaln_dim);
  SharedBuffer tsilu = buf((std::size_t)_cfg.adaln_dim);
  SharedBuffer mods = buf((std::size_t)2 * 4 * H);
  // THE FOLDED NORM GAINS. `scale_msa` and `scale_mlp` are ONE H-wide
  // vector each per row range (noisy, clean), so `rms(x)*w*(1+scale)` is
  // `rms(x)*(w*(1+scale))` -- exact algebra that turns two full-width
  // passes over [rows, H] into one, plus an H-element multiply.
  SharedBuffer nw1 = buf((std::size_t)2 * H);
  SharedBuffer nw3 = buf((std::size_t)2 * H);
  SharedBuffer outmod = buf((std::size_t)H);
  SharedBuffer tin = buf((std::size_t)2 * _cfg.time_freq);
  SharedBuffer zero = buf((std::size_t)H);
  SharedBuffer vel = buf((std::size_t)IMG * PD);
  SharedBuffer rcos, rsin;
  build_rope_(lay, &rcos, &rsin);
  if (joint.empty() || nrm.empty() || ob.empty() || ffg.empty() ||
      ffu.empty() || zero.empty() || vel.empty() || rcos.empty() ||
      rsin.empty() || ao.empty() || vh.empty() || nw1.empty() ||
      nw3.empty() || mods.empty() || temb.empty()) {
    return fail("fibo2 forward: scratch allocation failed");
  }
  std::memset(zero.contents(), 0, zero.byte_size());
  // The text's pass (see below): projections of a bundle, [T, H] each,
  // and what it leaves for every later forward.
  SharedBuffer tk, tv, tkh, tvh;
  if (ts.dirty) {
    tk = buf((std::size_t)T * H);
    tv = buf((std::size_t)T * H);
    tkh = buf((std::size_t)T * H);
    tvh = buf((std::size_t)T * H);
    ts.gist = buf((std::size_t)G * H);
    ts.inj_k.clear();
    ts.inj_v.clear();
    ts.inj_k.resize(_inject.size());
    ts.inj_v.resize(_inject.size());
    bool ok = !tk.empty() && !tv.empty() && !tkh.empty() && !tvh.empty() &&
              !ts.gist.empty();
    for (std::size_t i = 0; ok && i < _inject.size(); ++i) {
      ts.inj_k[i] = buf((std::size_t)T * H);
      ts.inj_v[i] = buf((std::size_t)T * H);
      ok = !ts.inj_k[i].empty() && !ts.inj_v[i].empty();
    }
    if (!ok) { return fail("fibo2 forward: text scratch allocation failed"); }
  }
  {
    // The model's input is sigma * t_scale, formed in f32 as the
    // reference's `timestep * t_scale` is.
    const std::vector<float> tn = time_proj_(req.timestep * _cfg.t_scale);
    const std::vector<float> tc = time_proj_(0.0f);
    auto* d = static_cast<std::uint16_t*>(tin.contents());
    for (int i = 0; i < _cfg.time_freq; ++i) {
      d[i] = f32_to_bf16_(tn[(std::size_t)i]);
      d[_cfg.time_freq + i] = f32_to_bf16_(tc[(std::size_t)i]);
    }
  }

  // The int8 split's width, for the shapes an earlier forward recorded.
  // HERE because it runs its own command streams: forward 1 records, 2
  // measures, every one after reads the cache. No-op when nothing is
  // pending, and on a run with i8 off.
  if (_i8) { _i8->tune_pending(mc); }

  CommandStream stream = mc->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  if (!enc.valid()) { return fail("fibo2 forward: no compute encoder"); }
  // perf-flops. Whatever an aborted forward left uncredited is dropped,
  // and Sol's GPU counter starts this forward at zero so finish() reads
  // this forward's keep alone.
  _flops.begin();
  if (_sol) {
    _sol->reset_counts();
    _sol_flops.begin();
  }

  // Per-stage TRACE, for locating a numeric gap against the reference.
  // It COMMITS, so a traced run is not a timing run.
  const char* trace_dir = std::getenv("VPIPE_FIBO2_TRACE");
  bool trace_ok = true;
  auto trace = [&](const char* tag, const SharedBuffer& b, std::size_t n) {
    if (trace_dir == nullptr || *trace_dir == '\0' || !trace_ok) { return; }
    enc.end();
    _flops.flush(stream);
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) { trace_ok = false; return; }
    std::error_code ec;
    fs::create_directories(trace_dir, ec);
    if (b.byte_size() >= n * 2) {
      const auto* sp = static_cast<const std::uint16_t*>(b.contents());
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

  // Per-CATEGORY profile.
  //
  // READ IT WITH ITS MARK COUNT, which is why that is printed. Each
  // mark is a commit AND a wait, so it charges its bucket a full
  // CPU-GPU round trip on top of the work -- and the buckets do not
  // carry the same number of marks. This model's elementwise bucket
  // takes three marks a block against one for each GEMM bucket, and
  // its real traffic at 1024^2 is ~13 GB a forward, which is ~35 ms at
  // this box's bandwidth. Any elementwise figure far above that is the
  // profiler measuring itself.
  //
  // So: TRUST THE BUCKETS WITH FEW MARKS AND MUCH WORK (the GEMMs,
  // attention), and treat a small-work bucket's share as an upper
  // bound. The total is not a step time at all.
  //
  // AND IT DEFEATS THE PREFETCH, which makes the `stream` bucket read
  // as the largest term in the model when it is nearly free. Every
  // mark commits AND waits, so there is no commit->wait window left
  // for the next block's read to run under, and the disk goes fully
  // exposed. MEASURED at 1024^2 on the M5, same binary, same
  // geometry: PROFILED the stream bucket is 3252 ms of an 8612 ms
  // total (37%); UNPROFILED the stream profile reports 3.9 s of
  // pread of which only 0.15-0.27 s is ever waited on (2.7-4.3% of
  // the pass). Use VPIPE_FIBO2_STREAM_PROFILE for the read, and
  // read this one for the GPU split alone.
  const bool prof = [] {
    const char* e = std::getenv("VPIPE_FIBO2_DIT_PROFILE");
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
    // The adapter accumulates ONTO the base result, so it runs after
    // whichever base route was taken and is identical for all of them.
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
    // The steel tiles do not read the signed layout: a signed weight
    // here (a shape matmul2d declines) is dequantized first.
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
      // Bound twice: the kernel takes the weight as both operands of
      // its biasless form.
      enc.set_buffer(2, w.w); enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
      enc.set_constant(7, 0);
      // THREADGROUP-indexed: a thread-indexed grid here computes a
      // fraction of the output and leaves the rest as it found it.
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
                 const SharedBuffer& y, std::size_t ye, int R, int D,
                 std::size_t we = 0) {
    if (R <= 0) { return; }
    enc.set_function(_fn_rms);
    enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w, we * 2);
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
  // out = (1 + scale) * x, the model's modulation. The kernel's shift
  // slot takes the zero row: this model emits no shift at all.
  auto modulate = [&](const SharedBuffer& x, std::size_t xe,
                      const SharedBuffer& sc, std::size_t sce,
                      const SharedBuffer& y, std::size_t ye, int R) {
    if (R <= 0) { return; }
    if ((H % 4) == 0 && _fn_adaln4.valid()) {
      enc.set_function(_fn_adaln4);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, sc, sce * 2);
      enc.set_buffer(2, zero); enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, H / 4); enc.set_constant(5, R);
      enc.dispatch({(unsigned)(H / 4), (unsigned)R, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_adaln);
    enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, sc, sce * 2);
    enc.set_buffer(2, zero); enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, H);
    enc.set_constant(5, (int)((std::size_t)R * H));
    enc.dispatch({(unsigned)((std::size_t)R * H), 1, 1}, {256, 1, 1});
  };
  auto gated = [&](const SharedBuffer& h, std::size_t he,
                   const SharedBuffer& gate, std::size_t ge,
                   const SharedBuffer& sub, std::size_t se, int R) {
    if (R <= 0) { return; }
    if ((H % 4) == 0 && _fn_gated_tanh4.valid()) {
      enc.set_function(_fn_gated_tanh4);
      enc.set_buffer(0, h, he * 2); enc.set_buffer(1, gate, ge * 2);
      enc.set_buffer(2, sub, se * 2);
      enc.set_constant(3, H / 4); enc.set_constant(4, R);
      enc.dispatch({(unsigned)(H / 4), (unsigned)R, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_gated_tanh);
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
  auto swiglu = [&](const SharedBuffer& g, const SharedBuffer& u,
                    const SharedBuffer& o, int n) {
    if ((n % 4) == 0 && _fn_swiglu4.valid()) {
      enc.set_function(_fn_swiglu4);
      enc.set_buffer(0, g); enc.set_buffer(1, u); enc.set_buffer(2, o);
      enc.set_constant(3, n / 4);
      enc.dispatch({(unsigned)(n / 4), 1, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_swiglu);
    enc.set_buffer(0, g); enc.set_buffer(1, u); enc.set_buffer(2, o);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  // src_stride 0 BROADCASTS one row over `rows` destination rows, which
  // is how the two learned pad tokens fill their ranges.
  auto copy_rows = [&](const SharedBuffer& src, std::size_t so,
                       const SharedBuffer& dst, std::size_t do_, int rows,
                       int row_elems, int sstride, int dstride) {
    if (rows <= 0) { return; }
    enc.set_function(_fn_copy_rows);
    enc.set_buffer(0, src); enc.set_buffer(1, dst);
    enc.set_constant(2, (int)so); enc.set_constant(3, (int)do_);
    enc.set_constant(4, rows); enc.set_constant(5, row_elems);
    enc.set_constant(6, sstride); enc.set_constant(7, dstride);
    // ONE-DIMENSIONAL: the kernel takes a flat grid over
    // rows*row_elems.
    enc.dispatch({(unsigned)((std::size_t)rows * row_elems), 1, 1},
                 {256, 1, 1});
  };
  auto ln_plain = [&](const SharedBuffer& x, std::size_t xe,
                      const SharedBuffer& y, std::size_t ye, int R) {
    enc.set_function(_fn_ln_plain);
    enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, y, ye * 2);
    enc.set_constant(2, H);
    // eps 1e-6 -- the FINAL layer's norm, not the model's norm_eps.
    enc.set_constant(3, 1e-6f);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  auto tr_rope = [&](const SharedBuffer& in, const SharedBuffer& out,
                     int rows, int row0) {
    const std::size_t tb = (std::size_t)row0 * Hd * sizeof(float);
    enc.set_function(_fn_tr_rope);
    enc.set_buffer(0, in); enc.set_buffer(1, out);
    enc.set_buffer(2, rcos, tb); enc.set_buffer(3, rsin, tb);
    enc.set_constant(4, NH); enc.set_constant(5, rows);
    enc.set_constant(6, Hd);
    enc.dispatch({(unsigned)(Hd / 2), (unsigned)rows, (unsigned)NH},
                 {(unsigned)(Hd / 2), 1, 1});
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
  // DENSE AND BIDIRECTIONAL, one dispatch per block. Self-attention at
  // the refiner's and the main stack's lengths, cross-attention from
  // those rows (and the gist's) to the text's -- so a params block and a
  // specialized function are memoized per (queries, keys) rather than
  // rebuilt per block.
  const bool nax = _use_attn_nax;
  const int A_BQ = nax ? 64 : 32;
  const int A_BK = nax ? 32 : 16;
  const float ascale = 1.0f / std::sqrt((float)Hd);
  std::vector<std::pair<long long, SharedBuffer>> aparams;
  auto params_for = [&](int qL, int kL) -> const SharedBuffer* {
    const long long key = ((long long)qL << 32) | (unsigned)kL;
    for (auto& p : aparams) {
      if (p.first == key) { return &p.second; }
    }
    SharedBuffer b = mc->make_shared_buffer(sizeof(SteelAttnParams));
    if (b.empty()) { return nullptr; }
    auto* p = static_cast<SteelAttnParams*>(b.contents());
    p->B = 1; p->H = NH; p->D = Hd; p->qL = qL; p->kL = kL;
    p->gqa_factor = 1; p->scale = ascale;
    p->NQ = (qL + A_BQ - 1) / A_BQ; p->NK = (kL + A_BK - 1) / A_BK;
    p->NQ_aligned = qL / A_BQ; p->NK_aligned = kL / A_BK;
    p->qL_rem = qL - p->NQ_aligned * A_BQ;
    p->kL_rem = kL - p->NK_aligned * A_BK;
    p->qL_off = 0;
    const std::int64_t qs = (std::int64_t)qL * Hd;
    const std::int64_t ks = (std::int64_t)kL * Hd;
    p->Q_strides[0] = (std::int64_t)NH * qs;
    p->Q_strides[1] = qs; p->Q_strides[2] = Hd;
    p->K_strides[0] = (std::int64_t)NH * ks;
    p->K_strides[1] = ks; p->K_strides[2] = Hd;
    p->V_strides[0] = p->K_strides[0];
    p->V_strides[1] = ks; p->V_strides[2] = Hd;
    p->O_strides[0] = p->Q_strides[0];
    p->O_strides[1] = qs; p->O_strides[2] = Hd;
    aparams.emplace_back(key, std::move(b));
    return &aparams.back().second;
  };
  auto attn_fn = [&](int qL, int kL,
                     bool i8) -> const metal_compute::ComputeFunction* {
    const unsigned key = (nax ? 8u : 0u) | (i8 ? 4u : 0u) |
                         (((qL % A_BQ) == 0) ? 2u : 0u) |
                         (((kL % A_BK) == 0) ? 1u : 0u);
    auto it = _attn_fn.find(key);
    if (it != _attn_fn.end()) {
      return it->second.valid() ? &it->second : nullptr;
    }
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (qL % A_BQ) == 0).set_bool(201, (kL % A_BK) == 0)
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
  // Dense attention of head-major q [NH, qL, Hd] over k / v [NH, kL, Hd]
  // into o [NH, qL, Hd] -- the cross-attentions and the Perceiver's.
  auto attend = [&](const SharedBuffer& q, const SharedBuffer& k,
                    const SharedBuffer& v, const SharedBuffer& o, int qL,
                    int kL) {
    if (!attn_ok) { return; }
    const SharedBuffer* pp = params_for(qL, kL);
    const metal_compute::ComputeFunction* fn = attn_fn(qL, kL, false);
    if (pp == nullptr || fn == nullptr) { attn_ok = false; return; }
    enc.set_function(*fn);
    enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
    enc.set_buffer(3, o); enc.set_buffer(4, *pp);
    enc.dispatch({(unsigned)(32 * ((qL + A_BQ - 1) / A_BQ)),
                  (unsigned)(4 * NH), 1}, {32, 4, 1});
    _flops.attention(NH, (long long)qL * kL, Hd);
  };
  auto run_attn = [&](int L, int rows) {
    if (!attn_ok) { return; }
    // SOL, where the sequence is long enough to have something to skip.
    // Every attention in this model is square and non-causal, which is
    // exactly the shape Sol's exact half is built for -- there is no
    // causal segment to decline, unlike the block-causal families.
    // NOT gated on the matrix cores: Sol picks its own arm, and its ALU
    // arm is what an M4 -- where the attention is the larger share --
    // runs (bit-identical to dense with every block kept).
    if (_sol && _cfg.sol.enabled && L >= _cfg.sol.dense_layers &&
        rows >= kSolMinSegment) {
      const int blk = _cfg.sol.key_block > 0 ? _cfg.sol.key_block
                                             : sol::kBlock;
      if ((rows + blk - 1) / blk >= 16) {
        MetalSolAttention::Band band{rows, rows, rows, rows, 0, 0};
        std::string serr;
        if (_sol->encode(enc, qh, kh, vh, ao, NH, NH, band, Hd, ascale,
                         _cfg.sol, &serr)) {
          _flops.add(_sol_flops.routed(
              attention_flops(NH, (long long)rows * rows, Hd),
              _sol->total_blocks()));
          return;
        }
        if (_mc->session() != nullptr) {
          _mc->session()->log_debug(fmt(
              "fibo2: sol declined this stage ({}); dense",
              serr.empty() ? "no reason" : serr));
        }
      }
    }
    bool i8_ok = false;
    if (_sage && nax && _cfg.sage.enabled && L >= _cfg.sage.dense_layers &&
        rows >= kSageMinSegment) {
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
    const SharedBuffer* pp = params_for(rows, rows);
    const metal_compute::ComputeFunction* fn = attn_fn(rows, rows, i8_ok);
    if (pp == nullptr || fn == nullptr) { attn_ok = false; return; }
    enc.set_function(*fn);
    enc.set_buffer(0, qh); enc.set_buffer(1, kh); enc.set_buffer(2, vh);
    enc.set_buffer(3, ao); enc.set_buffer(4, *pp);
    if (i8_ok) { _sage->bind(enc); }
    enc.dispatch({(unsigned)(32 * ((rows + A_BQ - 1) / A_BQ)),
                  (unsigned)(4 * NH), 1}, {32, 4, 1});
    _flops.attention(NH, (long long)rows * rows, Hd);
  };

  // ---- one block ----------------------------------------------------
  //
  // The sandwich: norm1 BEFORE the sublayer and norm2 on its OUTPUT,
  // inside the residual and under the gate. `mods` holds the four
  // chunks (scale_msa | gate_msa | scale_mlp | gate_mlp) for the block
  // about to run, twice: row 0 the image's timestep, row 1 t = 0's.
  //
  // PER-TOKEN MODULATION: rows [0, nN) are the image's (noisy), rows
  // [nN, rows) the context images' and the gist's (clean). The image
  // leads the sequence, so that is two row ranges, not a mask.
  auto run_block = [&](const Block& b, int rows, int L,
                       AneFeedForward::Plan ane_plan,
                       AneFeedForward::Plan qkv_plan) -> bool {
    auto LA = [&](lora::Factors BlockLora::* w) {
      return lora_for_(L, w);
    };
    const int nN = std::min(IMG, rows), nC = rows - nN;
    const std::size_t cH = (std::size_t)nN * H;   // the clean rows
    const std::size_t M4 = (std::size_t)4 * H;    // mods' row stride
    // The input norm and its (1 + scale): pre-folded into the gain (see
    // `nw1`), one pass a range; or normed, then modulated per range.
    auto norm_in = [&](const SharedBuffer& w, const SharedBuffer& folded,
                       std::size_t sc) {
      if (_fold_mod) {
        rms(joint, 0, folded, nrm, 0, nN, H, 0);
        rms(joint, cH, folded, nrm, cH, nC, H, (std::size_t)H);
        return;
      }
      rms(joint, 0, w, nrm, 0, rows, H);
      modulate(nrm, 0, mods, sc, nrm, 0, nN);
      modulate(nrm, cH, mods, M4 + sc, nrm, cH, nC);
    };
    auto gate_in = [&](std::size_t g) {
      gated(joint, 0, mods, g, ob, 0, nN);
      gated(joint, cH, mods, M4 + g, ob, cH, nC);
    };
    norm_in(b.an1, nw1, 0);
    mark("elt");
    // ---- the q/k/v split point --------------------------------------
    //
    // The feed-forward's choreography, one sublayer earlier: `nrm` holds
    // the attention input for every row, so drain, START THE ANE on the
    // tail rows -- its output columns land in qb / kb / vb at the rows
    // they came from -- and only then encode the GPU's head rows. The
    // attention reads all of q/k/v, so the join comes before it.
    const bool qkv_split = qkv_plan == AneFeedForward::Plan::kSplit;
    const bool qkv_probe = qkv_plan == AneFeedForward::Plan::kProbe;
    int q_rows = 0;
    double q_drain_ms = 0.0;
    if (qkv_split || qkv_probe) {
      enc.end();
      _flops.flush(stream);
      const auto t_d0 = std::chrono::steady_clock::now();
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) { return false; }
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
            "fibo2: staging block {}'s q/k/v for the ANE failed; it "
            "keeps the GPU projections", L));
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
    }
    const int qg_rows = rows - q_rows;
    const auto t_q0 = std::chrono::steady_clock::now();
    if (qg_rows > 0 &&
        !fast_shared_(enc, nrm, 0, qg_rows, H,
                      {{&b.qw, &qb, 0, H, !LA(&BlockLora::q).empty()},
                       {&b.kw, &kb, 0, H, !LA(&BlockLora::k).empty()},
                       {&b.vw, &vb, 0, H, !LA(&BlockLora::v).empty()}})) {
      lin(nrm, 0, b.qw, qb, 0, qg_rows, H, H, LA(&BlockLora::q));
      lin(nrm, 0, b.kw, kb, 0, qg_rows, H, H, LA(&BlockLora::k));
      lin(nrm, 0, b.vw, vb, 0, qg_rows, H, H, LA(&BlockLora::v));
    }
    mark("gemm.qkv");
    if (qkv_split || qkv_probe) {
      enc.end();
      _flops.flush(stream);
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        if (q_rows > 0) { (void)_ane_qkv->finish(L, rows, 0.0, q_drain_ms); }
        return false;
      }
      const double t_gpu = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_q0).count();
      if (qkv_probe) { _ane_qkv->note_probe(q_drain_ms, t_gpu); }
      // THE GPU FALLBACK: rows the ANE lost are projected here, on the
      // stream the attention is about to use. `nrm` is only the tier's
      // input, so every row of it is still there.
      int q_lost = -1;
      if (!_ane_qkv->finish(L, rows, t_gpu, q_drain_ms) && q_rows > 0) {
        q_lost = _ane_qkv->lost_row0();
        if (q_lost < 0 || q_lost >= rows) { return false; }
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
      if (q_lost >= 0) {
        const std::size_t e0 = (std::size_t)q_lost * H;
        const int n = rows - q_lost;
        lin(nrm, e0, b.qw, qb, e0, n, H, H, LA(&BlockLora::q));
        lin(nrm, e0, b.kw, kb, e0, n, H, H, LA(&BlockLora::k));
        lin(nrm, e0, b.vw, vb, e0, n, H, H, LA(&BlockLora::v));
      }
    }
    // The worker is free again: the feed-forward's weights now, so they
    // stage under the attention exactly as they would alone.
    if (qkv_split && ane_plan == AneFeedForward::Plan::kSplit) {
      ane_stage_(L, b);
    }
    // Per-head RMSNorm over the last Hd of a token-major [rows, NH, Hd].
    rms(qb, 0, b.nq, qb, 0, rows * NH, Hd);
    rms(kb, 0, b.nk, kb, 0, rows * NH, Hd);
    tr_rope(qb, qh, rows, 0);
    tr_rope(kb, kh, rows, 0);
    // V is transposed but NOT rotated.
    transpose(vb, vh, rows, NH);
    mark("rope");
    run_attn(L, rows);
    mark("attn");
    transpose(ao, nrm, NH, rows);
    mark("transpose");
    lin(nrm, 0, b.ow, ob, 0, rows, H, H, LA(&BlockLora::o));
    mark("gemm.out");
    rms(ob, 0, b.an2, ob, 0, rows, H);
    gate_in((std::size_t)H);

    // ---- an injection block's gated cross-attention -------------------
    //
    // After the attention's residual and before the feed-forward, every
    // row (image, context, gist) reads this block's text bundle:
    // unmodulated, no rotary, keys and values from set_text's pass, into
    // the residual through tanh(cross_attn_gate).
    const int Lm = L - _cfg.n_refiner;
    const int bi = Lm >= 0 ? _cfg.bundle_of(Lm) : -1;
    if (bi > 0) {
      const std::size_t k = (std::size_t)(bi - 1);
      const Inject& in = _inject[k];
      rms(joint, 0, in.n1, nrm, 0, rows, H);
      lin(nrm, 0, in.ca.qw, qb, 0, rows, H, H);
      rms(qb, 0, in.ca.nq, qb, 0, rows * NH, Hd);
      transpose(qb, qh, rows, NH);
      attend(qh, ts.inj_k[k], ts.inj_v[k], ao, rows, T);
      transpose(ao, nrm, NH, rows);
      lin(nrm, 0, in.ca.ow, ob, 0, rows, H, H);
      rms(ob, 0, in.n2, ob, 0, rows, H);
      gated(joint, 0, in.gate, 0, ob, 0, rows);
      mark("cross");
    }

    norm_in(b.fn1, nw3, (std::size_t)2 * H);
    mark("elt");
    // ---- the ANE split point ----------------------------------------
    //
    // `nrm` now holds the feed-forward input for every row. THE ORDER
    // HERE IS LOAD-BEARING: commit the attention without waiting,
    // drain, START THE ANE on the tail rows, and only then encode the
    // GPU's. Starting it after the GPU's w1/w3 are encoded would leave
    // only the down projection to hide behind.
    const bool ane_split = ane_plan == AneFeedForward::Plan::kSplit;
    const bool ane_probe = ane_plan == AneFeedForward::Plan::kProbe;
    int a_rows = 0;
    double ane_drain_ms = 0.0;
    if (ane_split || ane_probe) {
      enc.end();
      _flops.flush(stream);
      const auto t_d0 = std::chrono::steady_clock::now();
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) { return false; }
      ane_drain_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_d0).count();
      if (ane_split && _ane->join_stage(L)) {
        a_rows = _ane->begin(nrm, ob, rows);
      } else if (ane_split && _mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "fibo2: staging block {}'s weights for the ANE failed; it "
            "keeps the GPU feed-forward", L));
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
    }
    // The GPU's feed-forward over rows [r0, r0 + n), `nrm` into `ob`: its
    // share of a split, every row of an unsplit block, and the rows the
    // ANE lost. The scratch is used from its first row whatever r0 is.
    auto gpu_ff = [&](int r0, int n) {
      if (n <= 0) { return; }
      const std::size_t e0 = (std::size_t)r0 * H;
      if (!fast_shared_(enc, nrm, e0, n, H,
                        {{&b.w1, &ffg, 0, FFI, !LA(&BlockLora::w1).empty()},
                         {&b.w3, &ffu, 0, FFI,
                          !LA(&BlockLora::w3).empty()}})) {
        lin(nrm, e0, b.w1, ffg, 0, n, FFI, H, LA(&BlockLora::w1));
        lin(nrm, e0, b.w3, ffu, 0, n, FFI, H, LA(&BlockLora::w3));
      }
      // FUSED: the down projection's int8 input quantized straight from
      // gate and up, so the product is never stored -- the same codes
      // swiglu + gemm_fast make. Not with an adapter on the down
      // projection (it reads the bf16 product).
      // VPIPE_FIBO2_NO_SWIGLU_FUSE=1 for the A/B.
      {
        const QWeight& dw = b.w2;
        const SharedBuffer* dc = nullptr;
        const SharedBuffer* dcs = nullptr;
        if (_i8 && dw.quantized && dw.gc > 0 && LA(&BlockLora::w2).empty() &&
            n <= mma_row_band(H, FFI) &&
            std::getenv("VPIPE_FIBO2_NO_SWIGLU_FUSE") == nullptr &&
            _i8->fast_accepts(n, H, FFI) &&
            fast_operand_(enc, dw, &dc, &dcs) &&
            _i8->gemm_fast_swiglu(enc, ffg, 0, ffu, 0, *dc, *dcs, dw.gc, ob,
                                  e0, n, H, FFI)) {
          _flops.gemm(n, H, FFI);
          return;
        }
      }
      // In place on the gate: same index in, same index out. The
      // largest elementwise dispatch in the model, so it is the one
      // that pays most for being four-wide.
      swiglu(ffg, ffu, ffg, n * FFI);
      lin(ffg, 0, b.w2, ob, e0, n, H, FFI, LA(&BlockLora::w2));
    };
    // The GPU's share. A probe block deliberately takes EVERY row:
    // that is what makes its time comparable with the split's.
    const int g_rows = rows - a_rows;
    const auto t_g0 = std::chrono::steady_clock::now();
    gpu_ff(0, g_rows);
    mark("gemm.ff");
    // The GPU's rows have to be DONE before the ANE's are folded in and
    // before the norm below reads the whole tensor, so this commit is
    // the join and its time is the GPU's share for the balancer.
    if (ane_split || ane_probe) {
      enc.end();
      _flops.flush(stream);
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        if (a_rows > 0) { (void)_ane->finish(L, rows, 0.0, ane_drain_ms); }
        return false;
      }
      const double t_gpu = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_g0).count();
      if (ane_probe) { _ane->note_probe(ane_drain_ms, t_gpu); }
      // THE GPU FALLBACK: `nrm` is only the tier's input, so the rows it
      // lost can be run again from it.
      int ff_lost = -1;
      if (!_ane->finish(L, rows, t_gpu, ane_drain_ms) && a_rows > 0) {
        ff_lost = _ane->lost_row0();
        if (ff_lost < 0 || ff_lost >= rows) { return false; }
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
      if (ff_lost >= 0) { gpu_ff(ff_lost, rows - ff_lost); }
    }
    rms(ob, 0, b.fn2, ob, 0, rows, H);
    gate_in((std::size_t)3 * H);
    mark("elt");
    return true;
  };
  // mods = adaLN_modulation(temb) -- a BARE Linear, no activation in
  // front of it. The final layer's is SiLU + Linear; the reference
  // spells that difference only as the index inside an nn.Sequential.
  // Two rows: the image's timestep and t = 0.
  auto fill_mod = [&](const Block& b) {
    const std::size_t M4 = (std::size_t)4 * H;
    lin(temb, 0, b.ada, mods, 0, 2, 4 * H, _cfg.adaln_dim);
    bias(b.ada_b, mods, 0, 2, 4 * H);
    // ... and fold each (1 + scale) into its norm's gain, a row each.
    if (_fold_mod) {
      modulate(b.an1, 0, mods, 0, nw1, 0, 1);
      modulate(b.an1, 0, mods, M4, nw1, (std::size_t)H, 1);
      modulate(b.fn1, 0, mods, (std::size_t)2 * H, nw3, 0, 1);
      modulate(b.fn1, 0, mods, M4 + (std::size_t)2 * H, nw3,
               (std::size_t)H, 1);
    }
  };

  // ---- timestep: the image's and t = 0's, one GEMM row each ----------
  lin(tin, 0, _time_1, te1, 0, 2, _cfg.time_mid, _cfg.time_freq);
  bias(_time_1b, te1, 0, 2, _cfg.time_mid);
  silu(te1, te1, 2 * _cfg.time_mid);
  lin(te1, 0, _time_2, temb, 0, 2, _cfg.adaln_dim, _cfg.time_mid);
  bias(_time_2b, temb, 0, 2, _cfg.adaln_dim);
  // The final layer's SiLU, the image's row only.
  silu(temb, tsilu, _cfg.adaln_dim);

  // ---- the text: the Perceiver and the injection keys and values ------
  //
  // ONCE PER TEXT, on the first forward after set_text and on its
  // stream, ahead of its blocks: every one of these is a function of the
  // text alone, which the reference recomputes at every step.
  if (ts.dirty) {
    const std::size_t bstride = (std::size_t)T * CAP;
    // A cross-attention's keys (normed) and values over text bundle
    // `te`, head-major [NH, T, Hd].
    auto text_kv = [&](const CrossAttn& ca, std::size_t te,
                       const SharedBuffer& ko, const SharedBuffer& vo) {
      lin(ts.text, te, ca.kw, tk, 0, T, H, CAP);
      rms(tk, 0, ca.nk, tk, 0, T * NH, Hd);
      transpose(tk, ko, T, NH);
      lin(ts.text, te, ca.vw, tv, 0, T, H, CAP);
      transpose(tv, vo, T, NH);
    };
    copy_rows(_queries, 0, ts.gist, 0, G, H, H, H);
    for (const PerceiverLayer& l : _perceiver) {
      // Cross-attention to bundle 0.
      rms(ts.gist, 0, l.n_ca, nrm, 0, G, H);
      lin(nrm, 0, l.ca.qw, qb, 0, G, H, H);
      rms(qb, 0, l.ca.nq, qb, 0, G * NH, Hd);
      transpose(qb, qh, G, NH);
      text_kv(l.ca, 0, tkh, tvh);
      attend(qh, tkh, tvh, ao, G, T);
      transpose(ao, nrm, NH, G);
      lin(nrm, 0, l.ca.ow, ob, 0, G, H, H);
      resid(ts.gist, 0, ob, 0, G * H);
      // Self-attention, no rotary.
      rms(ts.gist, 0, l.n_sa, nrm, 0, G, H);
      lin(nrm, 0, l.sq, qb, 0, G, H, H);
      lin(nrm, 0, l.sk, kb, 0, G, H, H);
      lin(nrm, 0, l.sv, vb, 0, G, H, H);
      rms(qb, 0, l.snq, qb, 0, G * NH, Hd);
      rms(kb, 0, l.snk, kb, 0, G * NH, Hd);
      transpose(qb, qh, G, NH);
      transpose(kb, kh, G, NH);
      transpose(vb, vh, G, NH);
      attend(qh, kh, vh, ao, G, G);
      transpose(ao, nrm, NH, G);
      lin(nrm, 0, l.so, ob, 0, G, H, H);
      resid(ts.gist, 0, ob, 0, G * H);
      // SwiGLU.
      rms(ts.gist, 0, l.n_ff, nrm, 0, G, H);
      lin(nrm, 0, l.w1, ffg, 0, G, FFI, H);
      lin(nrm, 0, l.w3, ffu, 0, G, FFI, H);
      swiglu(ffg, ffu, ffg, G * FFI);
      lin(ffg, 0, l.w2, ob, 0, G, H, FFI);
      resid(ts.gist, 0, ob, 0, G * H);
    }
    rms(ts.gist, 0, _p_norm_out, ts.gist, 0, G, H);
    for (std::size_t i = 0; i < _inject.size(); ++i) {
      text_kv(_inject[i].ca, (i + 1) * bstride, ts.inj_k[i], ts.inj_v[i]);
    }
    trace("gist", ts.gist, (std::size_t)G * H);
    mark("text");
  }

  // ---- the joint stream ---------------------------------------------
  //
  // [image | context images | gist]: the image and context latents
  // share the patch embedding; the gist rows follow as the Perceiver
  // left them. The noise refiner reads only the first RT rows, so the
  // gist can take its place now.
  lin(*req.latents, 0, _x_in, joint, 0, RT, H, PD);
  bias(_x_in_b, joint, 0, RT, H);
  copy_rows(ts.gist, 0, joint, (std::size_t)RT * H, G, H, H, H);
  trace("embed", joint, (std::size_t)JT * H);
  mark("embed");

  // ---- blocks --------------------------------------------------------
  //
  // Freeing or reading a slot while its reader thread is still writing
  // into it is a use-after-free, and a scope guard is the only version
  // of this that survives a return added later.
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
      // The trunk takes its place BEFORE this forward's admissions
      // start asking: the blocks are the shed-able half, so a pool that
      // runs out should run out on them. The Perceiver and the
      // injection sublayers are part of that trunk -- held in both modes.
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
              "MetalFibo2Transformer: resident weights are only {}% in "
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

  // ---- env-gated streaming split (VPIPE_FIBO2_STREAM_PROFILE) ------
  //
  // WHAT A PREFETCH COULD HIDE. The streamed path takes strict turns
  // with the disk -- a block is acquired, run, committed and waited
  // before its slot is refilled -- so both ends are already on the
  // critical path and timing them adds no barrier of its own. The
  // ceiling on any prefetch is read/(read+gpu), measured rather than
  // assumed, because every term moves with the machine: block bytes
  // with the checkpoint's DTYPE (this family's Turbo pack is F32, so
  // its reads are twice the base model's for the same parameters),
  // read rate with the storage medium, gpu with the geometry.
  const bool sprof = std::getenv("VPIPE_FIBO2_STREAM_PROFILE") != nullptr;
  double sp_read_ms = 0.0, sp_gpu_ms = 0.0;
  std::size_t sp_read_bytes = 0;
  int sp_blocks = 0;
  const double sp_alloc0 = sprof && _ws ? _ws->stats().streamed_alloc_ms : 0.0;
  const double sp_fetch0 = sprof && _ws ? _ws->stats().streamed_fetch_ms : 0.0;
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

  // THE ADAPTER'S [rows, rank] INTERMEDIATE, sized ONCE for the widest
  // projection this pass takes -- the joint sequence. Without it
  // apply() returns early and the adapter silently does nothing, which
  // looks exactly like an adapter that had no effect.
  if (_lora_rank_total > 0) {
    _lora_scratch_rows = (std::size_t)JT;
    _lora.ensure_scratch(_lora_scratch_rows *
                         (std::size_t)_lora_rank_total);
  }

  const int TOTAL_BLOCKS = _cfg.n_blocks();
  int done = 0;

  // SAGE'S SCRATCH, sized ONCE for the forward's largest attention. The
  // stacks attend at two row counts -- the noise refiner over the image
  // and context rows, the main stack over those and the gist -- and a
  // prepare() at a new count rebuilt the scratch. Reserved at the
  // largest Sage will see, the smaller calls reuse it; a forward at the
  // same size builds nothing. (The cross-attentions run dense.) A GUIDED
  // generation alternates two joint lengths, the prompt's gist and the
  // negative's over the same rows, so the main stack is reserved at the
  // longer of the texts held: reserved at its own, each pass rebuilt the
  // scratch the other had just built.
  if (_sage && nax && _cfg.sage.enabled && _sage_reserve) {
    int big = 0;
    auto take = [&](int rows) {
      if (rows >= kSageMinSegment) { big = std::max(big, rows); }
    };
    int gmax = G;
    for (const TextState& t : _txt) { gmax = std::max(gmax, t.gist_n); }
    // The noise refiner is blocks 0 .. n_refiner - 1 to dense_layers.
    if (_cfg.sage.dense_layers < _cfg.n_refiner) { take(RT); }
    if (_cfg.sage.dense_layers < NM) { take(RT + gmax); }
    std::string rerr;
    if (big > 0 &&
        !_sage->reserve(NH, NH, big, big, Hd, A_BQ, A_BK,
                        _sage->pv8_for(_cfg.sage), &rerr) &&
        _mc->session() != nullptr) {
      // Not fatal: each prepare() sizes its own call, as without one.
      _mc->session()->log_debug(fmt(
          "fibo2: sage scratch not reserved at {} rows ({})", big, rerr));
    }
  }

  // ---- noise_refiner then layers --------------------------------------
  for (int L = 0; L < NM; ++L) {
    // COOPERATIVE STOP, checked every block. It returns EMPTY WITHOUT
    // AN ERROR, because a stop is not a failure: the caller tells the
    // two apart by whether a reason came back.
    if (_stream_stop && _stream_stop()) { return SharedBuffer{}; }
    // RESIDENT once residency has promoted it, and read into a slot
    // until then. `_blocks` is sized to the stack in streaming mode and
    // an unfilled entry reads as empty, so this one test covers both.
    const bool held = L < (int)_blocks.size() &&
                      !_blocks[(std::size_t)L].qw.empty();
    const bool streaming = _stream_blocks && !held;
    const Block* b = nullptr;
    if (!streaming) {
      b = &_blocks[(std::size_t)L];
    } else {
      const auto rd0 = sp_now();
      b = _slots.acquire(L);
      if (b == nullptr) {
        return fail("fibo2 forward: a streamed block did not load");
      }
      sp_add(sp_read_ms, rd0);
      if (sprof) {
        sp_read_bytes += _slots.last_bytes();
        ++sp_blocks;
      }
      // THE READ IS NOT ELEMENTWISE WORK. acquire() blocks on the
      // slot's pread, and it sits before run_block's first mark, so
      // without this the disk lands in whatever bucket follows.
      mark("stream");
    }
    // The refiner's rows are the image and context ones; the main
    // stack's the whole joint sequence.
    const bool is_refiner = L < _cfg.n_refiner;
    const int rows = is_refiner ? RT : JT;

    // THE ANE'S PLAN FOR THIS BLOCK, decided before anything is encoded
    // because the staging has to ride under the attention. Only for the
    // MAIN stack: the module is sized once, for the joint length, and a
    // refiner block runs a shorter sequence than it was built for.
    AneFeedForward::Plan ane_plan = AneFeedForward::Plan::kGpu;
    if (_cfg.ane_ffn && !is_refiner && ane_setup_(JT) &&
        ane_eligible_(L, *b)) {
      ane_plan = _ane->plan_block();
    }
    // q/k/v's plan, the same way and independently: where one tier pays
    // and the other does not, the one that does is kept.
    AneFeedForward::Plan qkv_plan = AneFeedForward::Plan::kGpu;
    if (_cfg.ane_qkv && !is_refiner && ane_qkv_setup_(JT) &&
        ane_qkv_eligible_(L, *b)) {
      qkv_plan = _ane_qkv->plan_block();
    }
    if (((ane_plan != AneFeedForward::Plan::kGpu) &&
         (L == _cfg.n_refiner || _ane->needs_barrier())) ||
        ((qkv_plan != AneFeedForward::Plan::kGpu) &&
         (L == _cfg.n_refiner || _ane_qkv->needs_barrier()))) {
      // An unsplit block commits nothing mid-block, so on a stack that
      // defers its commits the split point's drain would wait for every
      // such block queued before it. Drain what is queued FIRST.
      enc.end();
      _flops.flush(stream);
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        return fail("fibo2 forward: " + ge2);
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
    }
    // ONE ANE worker serves both tiers and takes one job at a time -- a
    // second stage() would block this thread until the first finished.
    // q/k/v are needed first, so they stage first; the feed-forward's
    // weights follow their join inside run_block.
    if (qkv_plan == AneFeedForward::Plan::kSplit) {
      ane_qkv_stage_(L, *b);
    } else if (ane_plan == AneFeedForward::Plan::kSplit) {
      ane_stage_(L, *b);
    }

    _flops.flush(stream);
    if (_block_progress) {
      // `done` of `total`: the blocks finished BEFORE this one. The
      // helper's contract wants a report that lags the truth and never
      // leads it.
      report_block(stream, _block_progress, done, TOTAL_BLOCKS);
    }

    fill_mod(*b);
    if (!run_block(*b, rows, L, ane_plan, qkv_plan)) {
      return fail("fibo2 forward: block " + std::to_string(L) +
                  " failed");
    }
    ++done;

    // ---- the streaming boundary --------------------------------------
    //
    // A preloaded run encodes every block into ONE stream and commits
    // once. A STREAMED one cannot: the two slots are reused block after
    // block, so anything encoded but not yet run still points at bytes
    // the next read would overwrite. The commit is a correctness
    // boundary here, not a flush, and it buys the window the prefetch
    // needs.
    if (streaming) {
      const auto gp0 = sp_now();
      enc.end();
      // This block's FLOPs ride the buffer committed and waited on here;
      // flushed at the next block's start they would be credited a block
      // late, on that block's buffer.
      _flops.flush(stream);
      std::string ge;
      {
        CommandStream::Fence bf = stream.commit();
        // BETWEEN THE COMMIT AND THE WAIT the GPU is busy with this
        // block and this thread has nothing to do. That window is where
        // the next block's read goes -- into the OTHER slot, the only
        // destination this block is not using. Skip what residency
        // already kept: prefetching one would leave a read outstanding
        // that no acquire ever collects.
        int nxt = -1;
        for (int n = L + 1; n < NM; ++n) {
          const bool h = n < (int)_blocks.size() &&
                         !_blocks[(std::size_t)n].qw.empty();
          if (!h) { nxt = n; break; }
        }
        _slots.prefetch(nxt);
        if (!bf.wait_ok(&ge)) { return fail("fibo2 forward: " + ge); }
      }
      // ---- keep this block, if the box can hold it -------------------
      //
      // AFTER the wait, so nothing encoded still points at the slot,
      // and promotion CLONES out of it so the slot stays usable for the
      // read already in flight.
      // KEPT bytes, not the slot's: a promotion drops a dense source
      // and dead pack scales, and converts a held w4 layout
      // (clone_block_).
      const std::size_t nb =
          _slots.last_bytes() > 0 ? kept_block_bytes_(*b) : 0;
      if (nb > 0 && _wire.wirable(nb) && _resid.admit(_mc, nb) &&
          _slots.promote_into(_blocks[(std::size_t)L])) {
        // What a preloaded block holds, whichever way it was promoted --
        // see keep_block_.
        (void)keep_block_(_blocks[(std::size_t)L]);
        // Wired LAST, after every write this block will ever get: mlock
        // pins the pages that exist NOW.
        _wire.note_wired(_mc, wire_block_(_blocks[(std::size_t)L], true),
                         nb);
        _resid.note_admitted(nb);
      }
      sp_add(sp_gpu_ms, gp0);
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
      if (!enc.valid()) { return fail("fibo2 forward: no encoder"); }
      // ITS OWN BUCKET. The commit, the wait and the promotion happen
      // between two of run_block's marks, so without this they are
      // charged to whichever bucket comes next -- which was the
      // elementwise one, and at 1024^2 that put ~2 s of DISK READ
      // under a heading whose real traffic is ~13 GB (about 35 ms at
      // this box's bandwidth). The GEMM buckets matched theory to
      // within 2% while the elementwise one read 40%; that gap WAS
      // the streaming.
      mark("stream");
    } else if (_stream_stop && kStopDrainBlocks > 0 &&
               ((L + 1) % kStopDrainBlocks) == 0 && L + 1 < NM) {
      // A HELD BLOCK COMMITS NOTHING, so without this the rest of the
      // stack encodes into one buffer and the whole forward runs
      // inside a single wait -- and the per-block check above, which
      // races through in microseconds, can only ever see a stop that
      // was ALREADY set. Draining on a cadence is what lets a request
      // arriving mid-forward be seen at all. Only when a stop is
      // wired: a run nobody can interrupt pays nothing.
      enc.end();
      _flops.flush(stream);
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) {
        return fail("fibo2 forward: " + ge);
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
      if (!enc.valid()) { return fail("fibo2 forward: no encoder"); }
    }
    trace(("blk" + std::to_string(L)).c_str(), joint, (std::size_t)JT * H);
  }
  if (!attn_ok) {
    return fail("fibo2 forward: the steel attention kernel did not "
                "specialize for this geometry");
  }

  // ---- the final layer -------------------------------------------------
  //
  // LN(x) * (1 + linear(SiLU(temb))) then a projection back to patch
  // space -- and THIS adaLN does have the SiLU in front of it, which
  // the per-block one does not. Only the IMAGE rows are projected: they
  // are the front of the joint sequence and the whole of what a denoise
  // step consumes.
  lin(tsilu, 0, _final_ada, outmod, 0, 1, H, _cfg.adaln_dim);
  bias(_final_ada_b, outmod, 0, 1, H);
  ln_plain(joint, 0, nrm, 0, IMG);
  modulate(nrm, 0, outmod, 0, nrm, 0, IMG);
  lin(nrm, 0, _final_lin, vel, 0, IMG, PD, H);
  bias(_final_lin_b, vel, 0, IMG, PD);
  enc.end();
  _flops.flush(stream);
  {
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) {
      return fail("fibo2 forward: " + (ge.empty() ? "GPU failed" : ge));
    }
  }
  // Every routed layer has run, so Sol's count is this forward's whole.
  if (_sol) { _sol_flops.finish(_sol->exact_blocks()); }
  // The text's pass ran with this forward; the bundles are not read
  // again until the next set_text.
  if (ts.dirty) {
    ts.dirty = false;
    ts.text = {};
  }
  if (_fast_failed) {
    _fast_failed = false;
    return fail("fibo2 forward: a fast-tier GEMM was declined after its "
                "input was quantized");
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
        "Fibo2 DiT profile (joint={} refiner={} gist={}, {} blocks, {}, "
        "{} marks -- each is a commit AND a wait, so a bucket with many "
        "marks and little work is mostly measuring this): total {} ms{}",
        JT, RT, G, TOTAL_BLOCKS,
        _use_mma2 ? "matmul2d" : (_gemm_tile == 2 ? "bm64bn64"
                                                  : (_gemm_tile == 1
                                                         ? "bm64"
                                                         : "bm32")),
        pmarks, (long)tot, line));
  }
  if (sprof && sp_blocks > 0 && mc->session() != nullptr) {
    const double tot = sp_read_ms + sp_gpu_ms;
    const double alloc_ms =
        (_ws ? _ws->stats().streamed_alloc_ms : 0.0) - sp_alloc0;
    const double fetch_ms =
        (_ws ? _ws->stats().streamed_fetch_ms : 0.0) - sp_fetch0;
    const double gbs = sp_read_ms > 0.0
        ? (double)sp_read_bytes / (sp_read_ms / 1000.0) / 1e9 : 0.0;
    mc->session()->log_normal(fmt(
        "MetalFibo2Transformer: streamed {} of {} blocks -- read {} ms "
        "({} MB at {:.2f} GB/s, {:.1f} ms/block), gpu {} ms ({:.0f} "
        "ms/block); a perfect prefetch hides at most {:.1f}% of this "
        "pass [read = {:.0f} ms alloc + {:.0f} ms fetch]{}",
        sp_blocks, _cfg.n_mod_blocks(), (long)sp_read_ms,
        sp_read_bytes >> 20, gbs, sp_read_ms / (double)sp_blocks,
        (long)sp_gpu_ms, sp_gpu_ms / (double)sp_blocks,
        tot > 0.0 ? 100.0 * sp_read_ms / tot : 0.0, alloc_ms, fetch_ms,
        _src_f32 ? " -- NOTE the checkpoint is F32, so these are twice "
                   "the bytes the same parameters cost in bf16"
                 : ""));
  }
  return vel;
}

bool
MetalFibo2Transformer::fast_shared_(ComputeEncoder& enc, const SharedBuffer& x,
                                     std::size_t xe, int M, int K,
                                     std::initializer_list<FastOut> outs)
{
  if (!_i8 || outs.size() < 2 ||
      std::getenv("VPIPE_FIBO2_NO_FAST_SHARE") != nullptr) {
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
      // Unreachable after the checks above; a hole would be silent, so
      // the forward reports it.
      _fast_failed = true;
    }
    _flops.gemm(M, o.n, K);
  }
  return true;
}

// ---- the matrix-core GEMM -------------------------------------------
//
// Returns false with NOTHING encoded when this box or this shape is not
// for it, and the caller keeps its tiled path -- so a decline is a
// fallback, never a hole.
bool
MetalFibo2Transformer::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& xin,
                                  std::size_t xe, const QWeight& w,
                                  const SharedBuffer& y, std::size_t ye,
                                  int M, int N, int K) const
{
  // Only where the hardware has matrix cores, where M amortizes the
  // 128-row tile, and where N is not degenerate. The timestep, the
  // modulation and the final adaLN are M=1 -- exactly the shape the
  // tile is wrong for -- and stay on steel at no cost either way.
  if (!_use_mma2 || N < 16) { return false; }
  // And decline a shape whose operands would not survive the tiles'
  // 32-bit addressing: past 2^31 BYTES on ANY operand the mma tiles
  // stop storing, SILENTLY. See shared/mma-tile.h.
  if (M > mma_row_band(N, K)) { return false; }
  // A signed weight's int8 GEMM, before the matmul2d tile's row gate:
  // its floor is the int8 route's own (I8GemmContext::signed_accepts --
  // 1024 rows, or lower by VPIPE_I8_SIGNED_MIN_M, which it then beats
  // the dequant + matmul2d below at).
  // THE FAST TIER's GEMM (a weight held in its layout), at the same
  // floor; below it the weight dequantizes like a signed one.
  if (_i8 && w.quantized && w.gc > 0 &&
      _i8->fast_accepts(M, N, K)) {
    const SharedBuffer* c = nullptr;
    const SharedBuffer* cs = nullptr;
    if (fast_operand_(enc, w, &c, &cs) &&
        _i8->gemm_fast(enc, xin, xe, *c, *cs, w.gc, SharedBuffer{}, y, ye,
                       M, N, K)) {
      return true;
    }
  }
  if (_i8 && w.quantized && w.sgn &&
      _i8->gemm_signed(enc, xin, xe, w.codes, w.scales, w.qbias, w.bits, y,
                       ye, M, N, K)) {
    return true;
  }
  if (M < _mma_min_m) { return false; }

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

  // Accelerated mode, from the SAME dense operand the tiles below would
  // read -- so it sits after the dequant and before the tile choice,
  // and a shape it declines falls through with nothing encoded.
  if (_i8 && _i8->gemm(enc, xin, xe, *wdense, y, ye, M, N, K)) {
    return true;
  }

  // Tile by REDUCTION DEPTH, which is what decides whether the deeper
  // N-region pays for itself. This model has three depths: 3840 for
  // qkv / out / ff-up, 10240 for ff-down, and 64 for the patch
  // embedder (which M rarely clears anyway).
  int RN = 256;                                  // TN*BN, the N per group
  const metal_compute::ComputeFunction* fn = &_fn_gemm_mma_deep;
  if (K < 6144) {
    fn = &_fn_gemm_mma; RN = 128;
  } else if (K < 12288 && _fn_gemm_mma_tn2.valid()) {
    fn = &_fn_gemm_mma_tn2; RN = 512;
  }
  if (!fn->valid()) { return false; }
  enc.set_function(*fn);
  enc.set_buffer(0, xin, xe * 2);
  // Bound twice: the kernel takes the weight as both operands of its
  // biasless form. Every big GEMM in this model IS biasless; the
  // handful that carry a bias add it in a separate pass.
  enc.set_buffer(1, *wdense); enc.set_buffer(2, *wdense);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  return true;
}

// ---- the ANE tier ---------------------------------------------------
//
// The feed-forward is row-independent, so the ANE takes the LAST rows
// of a block's feed-forward and the GPU the head rows, concurrently.
// Exact: the two compute the same function over disjoint rows of the
// same tensors, so there is no seam to blend.
//
// WHERE IT IS FOR: the M4 series, which has no GPU matrix cores, so its
// GEMMs run the tiled path and there is proportionally more for the ANE
// to take. A fanless box shares one SoC power budget between the two
// engines and does not truly overlap them -- do not judge the tier
// there.
int
MetalFibo2Transformer::ane_chunk_rows() noexcept
{
  return AneFeedForward::kChunkDefault;
}

std::size_t
MetalFibo2Transformer::ane_runtime_bytes(const Config& c, int seq) noexcept
{
  std::size_t n = AneFeedForward::runtime_bytes(c.hidden, c.ffn_inner, seq,
                                                ane_chunk_rows());
  // The q/k/v module is a second fixed unit, and it has to be booked:
  // one claimed at zero is memory CoreML holds that no plan can see.
  if (c.ane_qkv) {
    n += AneFeedForward::matmul_runtime_bytes(c.hidden, 3 * c.hidden, seq,
                                              ane_chunk_rows());
  }
  return n;
}

int
MetalFibo2Transformer::ane_plan_seq(int width, int height) noexcept
{
  // The joint sequence a picture implies: one row per 16x16 pixels
  // (the VAE's 8x times the DiT's patch 2), plus the gist (at most 512)
  // and whatever context images an edit brings, which this cannot know.
  // An edit at the picture's own size doubles the rows; the tier sizes
  // its module for the larger of the two.
  const int w = width > 0 ? width : 1024;
  const int h = height > 0 ? height : 1024;
  return 2 * (w / 16) * (h / 16) + 512;
}

bool
MetalFibo2Transformer::ane_setup_(int seq)
{
  if (_ane_tried) { return _ane != nullptr; }
  _ane_tried = true;
  if (_mc == nullptr || _mc->session() == nullptr || seq <= 0) {
    return false;
  }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "fibo2";
  o.hidden  = _cfg.hidden;
  o.ffn     = _cfg.ffn_inner;
  o.seq     = seq;
  o.rows    = _cfg.ane_rows;
  o.chunk   = ane_chunk_rows();
  o.profile = std::getenv("VPIPE_FIBO2_ANE_PROFILE") != nullptr;
  _ane = AneFeedForward::create(o);
  return _ane != nullptr;
}

bool
MetalFibo2Transformer::ane_eligible_(int L, const Block& b) const
{
  if (_ane == nullptr) { return false; }
  // Counted over the MAIN stack, which is what the tier runs on: L is
  // an index into the modulated stack and its first n_refiner entries
  // are the noise refiner.
  const int main_idx = L - _cfg.n_refiner;
  if (main_idx < 0) { return false; }
  const int cap = (_cfg.ane_layers > 0)
                      ? std::min(_cfg.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  if (main_idx >= cap) { return false; }
  auto ok = [](const QWeight& q) {
    if (q.empty()) { return false; }
    if (!q.quantized) { return true; }
    // The fast layout is not affine: the stager cannot read it.
    return q.gc == 0 && (q.bits == 4 || q.bits == 8) && !q.scales.empty() &&
           !q.qbias.empty();
  };
  return ok(b.w1) && ok(b.w3) && ok(b.w2);
}

namespace {

// One projection as the ANE stages it: dense bf16 or affine 4/8-bit,
// read whole, WITH the runtime adapters on it. The module is fed fp16
// weights per block, so an adapter has to be merged into what it
// stages -- left out, the ANE's rows run the base model while the
// GPU's run the adapted one, and the split becomes a seam in the
// picture that looks like an adapter applied at half strength.
// A template for the reason FLUX.2's twin is one: the weight type is
// private to the model, and deduction does not need to name it.
template <typename Q>
AneFfnSource
ane_src_(const Q& q, const lora::Stack& st)
{
  AneFfnSource s;
  s.w         = &q.w;
  s.codes     = q.quantized ? &q.codes : nullptr;
  s.scales    = q.quantized ? &q.scales : nullptr;
  s.qbias     = q.quantized ? &q.qbias : nullptr;
  s.quantized = q.quantized;
  s.bits      = q.quantized ? q.bits : 0;
  s.stride    = 1;
  s.offset    = 0;
  for (int i = 0; i < st.n && s.deltas < AneFfnSource::kMaxDeltas; ++i) {
    AneFfnSource::Delta& d = s.delta[s.deltas++];
    d.a     = &st.s[i].f->a;
    d.b     = &st.s[i].f->b;
    d.rank  = st.s[i].f->b_rank();
    d.group = st.s[i].f->group;
    d.parts = st.s[i].f->parts;
    d.scale = st.s[i].scale;
  }
  return s;
}

}  // namespace

void
MetalFibo2Transformer::ane_stage_(int L, const Block& b)
{
  if (_ane == nullptr) { return; }
  // This model keeps the gate and the up projection as SEPARATE
  // matrices and never fuses them, so each is read whole -- stride 1,
  // offset 0. A family that fuses reads one matrix twice with stride 2.
  _ane->stage(L, ane_src_(b.w1, lora_for_(L, &BlockLora::w1)),
              ane_src_(b.w3, lora_for_(L, &BlockLora::w3)),
              ane_src_(b.w2, lora_for_(L, &BlockLora::w2)),
              _quant_group);
}

// THE q/k/v TIER: one matmul module, hidden -> 3*hidden, whose single
// weight slot holds q, k and v stacked by rows, and whose output
// COLUMNS scatter straight into qb / kb / vb -- so the three stay the
// separate matrices this model keeps. Same row split, balancer and
// probe as the feed-forward's; the ANE's rows are the TAIL.
bool
MetalFibo2Transformer::ane_qkv_setup_(int seq)
{
  if (_ane_qkv_tried) { return _ane_qkv != nullptr; }
  _ane_qkv_tried = true;
  if (_mc == nullptr || _mc->session() == nullptr || seq <= 0) {
    return false;
  }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "fibo2-qkv";
  o.hidden  = _cfg.hidden;
  o.ffn     = 3 * _cfg.hidden;       // the matmul's output width
  o.seq     = seq;
  o.rows    = _cfg.ane_rows;
  o.chunk   = ane_chunk_rows();
  o.matmul  = true;
  o.profile = std::getenv("VPIPE_FIBO2_ANE_PROFILE") != nullptr;
  _ane_qkv = AneFeedForward::create(o);
  return _ane_qkv != nullptr;
}

bool
MetalFibo2Transformer::ane_qkv_eligible_(int L, const Block& b) const
{
  if (_ane_qkv == nullptr) { return false; }
  const int main_idx = L - _cfg.n_refiner;
  if (main_idx < 0) { return false; }
  const int cap = (_cfg.ane_layers > 0)
                      ? std::min(_cfg.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  if (main_idx >= cap) { return false; }
  auto ok = [](const QWeight& q) {
    if (q.empty()) { return false; }
    if (!q.quantized) { return true; }
    // The fast layout is not affine: the stager cannot read it.
    return q.gc == 0 && (q.bits == 4 || q.bits == 8) && !q.scales.empty() &&
           !q.qbias.empty();
  };
  return ok(b.qw) && ok(b.kw) && ok(b.vw);
}

void
MetalFibo2Transformer::ane_qkv_stage_(int L, const Block& b)
{
  if (_ane_qkv == nullptr) { return; }
  const std::size_t H = (std::size_t)_cfg.hidden;
  // Stacked along the slot's rows in the order the split scatters the
  // output columns: q, then k, then v -- each with its own adapters.
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
// THE MODULE NAMES ARE THE MODEL'S OWN, which is the convention a
// diffusers-style export of this architecture carries: the adapter's
// keys are the base tensor's path without `.weight`, optionally under
// a `transformer.`/`diffusion_model.` wrapper that the shared reader's
// rename strips. No Fibo 2.0 adapter has been published yet, so that is
// a READING of the convention rather than a verified match -- what is
// verified here is the arithmetic and the plumbing (see the
// synthesised-adapter test). When a real adapter appears and binds 0
// modules, the fix is a rename row, not this list.
//
// adaLN_modulation is deliberately absent: nothing in this lineage
// trains it, and binding it would read a file's silence as a zero. So
// are the Perceiver and the injection cross-attention: they run once a
// prompt, off the per-block GEMM helper's adapter path.
bool
MetalFibo2Transformer::bind_loras_(const std::vector<LoraSpec>& loras)
{
  if (loras.empty()) { return true; }
  if (!_lora.init(_mc, _use_mma2)) { return false; }
  if (!_lora.valid()) { return false; }
  const int nmod = _cfg.n_mod_blocks();
  int slot = 0;
  for (const LoraSpec& spec : loras) {
    if (slot >= kMaxLoraSlots || spec.path.empty()) { break; }
    std::string err;
    auto ad = lora::Adapter::open(spec.path, _mc, &err,
                                  lora::remap_ai_toolkit_module);
    if (ad == nullptr) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "MetalFibo2Transformer: lora '{}' did not open: {}",
            spec.path, err.empty() ? "unreadable" : err));
      }
      // The slot is still taken, EMPTY, so every later adapter keeps the
      // index the caller gave it: a live strength is addressed by that
      // index, and closing the gap would send it to the next adapter.
      _lora_mod[slot].clear();
      _lora_bound[slot] = 0;
      _lora_scale[slot] = 0.0f;
      _lora_rank_prefix[slot] = _lora_rank_total;
      ++slot;
      continue;
    }
    // resize, not assign: Factors holds SharedBuffers and is move-only.
    _lora_mod[slot].clear();
    _lora_mod[slot].resize((std::size_t)nmod);
    const int H = _cfg.hidden, FFI = _cfg.ffn_inner;
    auto one = [&](const std::string& pre, BlockLora& d) {
      ad->bind(pre + "attention.to_q", H, H, &d.q);
      ad->bind(pre + "attention.to_k", H, H, &d.k);
      ad->bind(pre + "attention.to_v", H, H, &d.v);
      ad->bind(pre + "attention.to_out.0", H, H, &d.o);
      ad->bind(pre + "feed_forward.w1", FFI, H, &d.w1);
      ad->bind(pre + "feed_forward.w3", FFI, H, &d.w3);
      ad->bind(pre + "feed_forward.w2", H, FFI, &d.w2);
    };
    for (int L = 0; L < nmod; ++L) {
      one(mod_prefix_(L), _lora_mod[slot][(std::size_t)L]);
    }
    _lora_bound[slot] = ad->modules();
    _lora_scale[slot] = spec.scale;
    _lora_rank_prefix[slot] = _lora_rank_total;
    _lora_rank_total += ad->max_rank();
    if (_mc->session() != nullptr) {
      _mc->session()->log_normal(
          fmt("MetalFibo2Transformer: {}", ad->summary(spec.path,
                                                        spec.scale)));
      if (ad->modules() == 0) {
        // LOUD. An adapter for another model binds nothing and would
        // otherwise report success by saying nothing at all.
        _mc->session()->warn(fmt(
            "MetalFibo2Transformer: lora '{}' bound NO modules -- it is "
            "for a different model, or its names are in a convention "
            "this family does not map yet", spec.path));
      }
    }
    ++slot;
  }
  _lora_slots = slot;
  return true;
}

lora::Stack
MetalFibo2Transformer::lora_for_(int L,
                                  lora::Factors BlockLora::* which) const
{
  lora::Stack st;
  for (int i = 0; i < _lora_slots; ++i) {
    const std::vector<BlockLora>& v = _lora_mod[i];
    if (L < 0 || L >= (int)v.size()) { continue; }
    const lora::Factors& f = v[(std::size_t)L].*which;
    // Each slot gets its OWN region of the [rows, rank] scratch --
    // sharing one would make the last writer's intermediate the thing
    // every B factor multiplies.
    st.add(f, _lora_scale[i],
           _lora_scratch_rows * (std::size_t)_lora_rank_prefix[i]);
  }
  return st;
}

void
MetalFibo2Transformer::set_lora_scale(int slot, float s)
{
  if (slot < 0 || slot >= kMaxLoraSlots) { return; }
  _lora_scale[slot] = s;
}

int
MetalFibo2Transformer::lora_modules(int slot) const
{
  if (slot < 0 || slot >= kMaxLoraSlots) { return 0; }
  return _lora_bound[slot];
}

std::size_t
MetalFibo2Transformer::streaming_floor_bytes(const std::string& dir)
{
  if (dir.empty()) { return 0; }
  auto wts = MetalLlamaWeights::open_model(dir);
  if (!wts.has_value()) { return 0; }
  static const char* const kStreamed[] = {"noise_refiner.", "layers."};
  std::size_t trunk = 0;
  // An injection block's cross-attention sublayer is HELD (the trunk):
  // the slots refill only what every block has.
  auto held = [](const std::string& nm) {
    return nm.find(".cross_attention.") != std::string::npos ||
           nm.find(".cross_attn_") != std::string::npos;
  };
  // Keyed by the STREAM index, so noise_refiner.0 and layers.0 are
  // different blocks of one stack rather than the same one twice.
  std::vector<std::size_t> blocks;
  bool any_f32 = false;
  // An injection gate is ONE scalar in the file and a row of H bf16 in
  // the model (load_fixed_ broadcasts it for the gated residual), so it
  // is counted at what is held, after any narrowing.
  Config dims;
  const bool have_dims = Config::read_dims(dir, &dims);
  std::size_t gates = 0;
  for (const std::string& nm : wts->tensor_names()) {
    const auto* ti = wts->info(nm);
    const std::size_t nb = ti != nullptr ? (std::size_t)ti->nbytes : 0;
    if (ti != nullptr && ti->dtype == "F32") { any_f32 = true; }
    if (have_dims && nm.size() > 16 &&
        nm.compare(nm.size() - 16, 16, ".cross_attn_gate") == 0) {
      gates += (std::size_t)dims.hidden * 2;
      continue;
    }
    int which = -1;
    for (int i = 0; i < 2; ++i) {
      if (nm.rfind(kStreamed[i], 0) == 0) { which = i; break; }
    }
    if (which < 0 || held(nm)) { trunk += nb; continue; }
    const std::size_t i0 = std::strlen(kStreamed[which]);
    const std::size_t dot = nm.find('.', i0);
    if (dot == std::string::npos) { trunk += nb; continue; }
    const std::size_t idx =
        (std::size_t)(which * 1000 +
                      std::atol(nm.substr(i0, dot - i0).c_str()));
    if (blocks.size() <= idx) { blocks.resize(idx + 1, 0); }
    blocks[idx] += nb;
  }
  std::size_t widest = 0;
  for (std::size_t b : blocks) { widest = std::max(widest, b); }
  // Nothing streams: the floor is the whole thing, and saying so is
  // what stops a declaration promising a reduction that cannot happen.
  if (widest == 0) { return 0; }
  std::size_t floor = trunk + 2 * widest;
  // An F32 checkpoint is narrowed to bf16 on the way in -- the trunk,
  // the slots and every promoted block alike.
  if (any_f32) { floor /= 2; }
  return floor + gates;
}

std::size_t
MetalFibo2Transformer::checkpoint_block_bytes_() const
{
  if (_ws == nullptr) { return 0; }
  const MetalLlamaWeights& w = _ws->src();
  // Block 0 of the streaming stack, which is noise_refiner.0 -- the
  // same shape as every block in the stack.
  const std::string pre = mod_prefix_(0);
  std::size_t n = 0;
  for (const std::string& nm : w.tensor_names()) {
    if (nm.rfind(pre, 0) != 0) { continue; }
    const auto* ti = w.info(nm);
    if (ti != nullptr) { n += (std::size_t)ti->nbytes; }
  }
  // WHAT A BLOCK COSTS RESIDENT, not what it costs on disk. The Turbo
  // checkpoint is F32 and every block is narrowed to bf16 on the way
  // in, so a residency ledger fed the on-disk figure would book twice
  // what it keeps -- and admit half as many blocks as the box can hold.
  if (_src_f32) { n /= 2; }
  return n;
}

void
MetalFibo2Transformer::set_residency_schedule(int steps)
{
  const std::size_t blk = checkpoint_block_bytes_();
  _wire_block_hint = blk;
  _wire.new_run();
  _resid.set_schedule(steps, _cfg.n_mod_blocks(), blk, _wire.on(),
                      _mc != nullptr ? _mc->memory_budget()
                                     : metal_compute::MetalCompute::
                                           MemoryBudget{});
}

std::size_t
MetalFibo2Transformer::release_resident_blocks(std::size_t bytes)
{
  const std::size_t freed = _resid.release(bytes, [this]() -> std::size_t {
    return evict_tail_block_();
  });
  if (freed > 0 && _mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalFibo2Transformer: released {} MB of resident blocks "
        "({} left)", freed >> 20, _resid.count()));
  }
  return freed;
}

std::uint64_t
MetalFibo2Transformer::resident_bytes() const
{
  std::uint64_t n = qw_bytes_(_x_in) + _x_in_b.byte_size() +
                    qw_bytes_(_time_1) + _time_1b.byte_size() +
                    qw_bytes_(_time_2) + _time_2b.byte_size() +
                    qw_bytes_(_final_ada) + _final_ada_b.byte_size() +
                    qw_bytes_(_final_lin) + _final_lin_b.byte_size() +
                    _queries.byte_size() + _p_norm_out.byte_size();
  // The Perceiver and the injection sublayers are held in BOTH modes, so
  // they belong to the trunk rather than to the residency ledger.
  auto cross = [](const CrossAttn& c) {
    return qw_bytes_(c.qw) + qw_bytes_(c.kw) + qw_bytes_(c.vw) +
           qw_bytes_(c.ow) + c.nq.byte_size() + c.nk.byte_size();
  };
  for (const PerceiverLayer& l : _perceiver) {
    n += l.n_ca.byte_size() + l.n_sa.byte_size() + l.n_ff.byte_size() +
         cross(l.ca) + qw_bytes_(l.sq) + qw_bytes_(l.sk) + qw_bytes_(l.sv) +
         qw_bytes_(l.so) + l.snq.byte_size() + l.snk.byte_size() +
         qw_bytes_(l.w1) + qw_bytes_(l.w3) + qw_bytes_(l.w2);
  }
  for (const Inject& in : _inject) {
    n += cross(in.ca) + in.n1.byte_size() + in.n2.byte_size() +
         in.gate.byte_size();
  }
  // And what the texts left for the forwards: the gist rows and every
  // injection block's keys and values, a prompt's worth a slot (two
  // under guidance).
  for (const TextState& t : _txt) { n += t.bytes(); }
  // `_blocks` IS the answer in both modes, and adding the residency
  // ledger to it would double every promoted block. Preloaded, it holds
  // the whole stack and `_resid` is empty; streaming, it holds exactly
  // what residency promoted.
  for (const Block& b : _blocks) { n += block_bytes_(b); }
  // And the slot pair while it exists: it stays allocated beside every
  // promoted block, and a holding taken without it dropped two blocks
  // from both ledgers once the promoted set passed the floor.
  n += _slots.held_bytes();
  return n;
}

void
MetalFibo2Transformer::release_forward_scratch()
{
  _w_deq = {};
  if (_i8) { _i8->release_scratch(); }
}

std::size_t
MetalFibo2Transformer::forward_scratch_bytes(const Config& c, int img_rows,
                                              int txt_rows, bool packed,
                                              bool matrix_cores,
                                              int neg_txt_rows)
{
  // Mirrors forward(): `img_rows` the image's and every context image's
  // rows (the refiner's), `txt_rows` the text's tokens (<= 0: the
  // longest the gist rule distinguishes). Ten [JT, H] planes (joint,
  // nrm, ob, q / k / v, their head-major copies, ao), the [JT, F] gate
  // and up, the f32 rotary pair and the velocity -- all live for the
  // whole forward. The text's pass adds four [T, H] planes on its
  // forward, and what it leaves stays between forwards: the gist and
  // every injection block's keys and values. Held between forwards too:
  // a pack's dense stand-in, the int8 context, Sol's and Sage's scratch
  // (MetalZImageTransformer's notes; this is that class's arithmetic).
  //
  // A guided generation runs a second text, the negative, between the
  // prompt's forwards: its pass leaves its own gist and keys and values
  // beside the prompt's, and its forward's planes are the larger set
  // when its gist is (the two forwards never overlap).
  if (img_rows <= 0) { return 0; }
  auto rows_of = [&](int r) {
    return r > 0 ? (std::size_t)r : (std::size_t)c.gist_max_text;
  };
  auto gist_of = [&](std::size_t t) {
    return (std::size_t)fibo2::gist_count(
        (int)t, c.min_gist, c.max_gist, c.gist_step, c.gist_min_text,
        c.gist_max_text);
  };
  const bool guided = neg_txt_rows >= 0;
  const std::size_t T = rows_of(txt_rows);
  const std::size_t Tn = guided ? rows_of(neg_txt_rows) : 0;
  const std::size_t G = gist_of(T);
  const std::size_t Gn = guided ? gist_of(Tn) : 0;
  const std::size_t RT = (std::size_t)img_rows;
  const std::size_t JT = RT + std::max(G, Gn);
  const std::size_t H = (std::size_t)c.hidden;
  const std::size_t F = (std::size_t)c.ffn_inner;
  const std::size_t Hd = (std::size_t)c.head_dim;
  const std::size_t NI = c.injection_ids.size();
  std::size_t n = 10 * JT * H * 2 + 2 * JT * F * 2 + 2 * JT * Hd * 4 +
                  RT * (std::size_t)c.patch_dim() * 2;
  n += 4 * std::max(T, Tn) * H * 2 + G * H * 2 + 2 * NI * T * H * 2;
  if (guided) { n += Gn * H * 2 + 2 * NI * Tn * H * 2; }
  if (packed) { n += F * H * 2; }
  const int NH = c.n_heads;
  // Sol on either arm: the matrix-core one's bf16 partial, or the ALU
  // one's fp32 padded one.
  if (c.sol.enabled) {
    for (std::size_t r : {JT, RT}) {
      if (r >= (std::size_t)kSolMinSegment) {
        n += MetalSolAttention::scratch_bytes(NH, (int)r, (int)Hd,
                                              c.sol.key_block,
                                              matrix_cores);
      }
    }
  }
  if (!matrix_cores) { return n; }
  if (c.i8_gemm) {
    n += I8GemmContext::forward_bytes_bound((int)JT, (int)F, (int)F, (int)H,
                                            c.i8_tier == 0);
  }
  if (c.sage.enabled && JT >= (std::size_t)kSageMinSegment) {
    n += MetalSageAttention::scratch_bytes(NH, NH, (int)JT, (int)JT,
                                           (int)Hd, 64, 32, c.sage.pv8);
  }
  return n;
}

}  // namespace genai
}  // namespace vpipe
