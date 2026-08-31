#include "generative-models/shared/mma-tile.h"
#include "generative-models/flux2/metal-flux2-transformer.h"

#include "generative-models/shared/riffle-rows.h"
#include "generative-models/shared/i8-gemm.h"

#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
using metal_compute::ComputeEncoder;
using metal_compute::CommandStream;
using metal_compute::ComputeFunction;

namespace {

// C++ mirror of mlx::steel::AttnParams (identical layout to the LM / Krea-2).
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

// bf16 <-> f32 host helpers (the flux2 DiT runs in bf16: f16's 65504 range
// overflows on real conditioning outliers -- e.g. the <|im_start|> attention-
// sink activation -- through the deep residual/attention stream, exactly the
// QIE "residual 1e7" class. bf16's f32 exponent range holds them).
// Namespace for this class's derived-tensor cache keys. A WeightSet is
// shared by everything reading one checkpoint, so a key has to say which
// class's transform produced the bytes, not just which tensor.
constexpr const char* kKey = "flux2-dit/";

inline std::uint16_t f32_to_bf16_(float f)
{
  std::uint32_t u; std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

// Load a checkpoint tensor -> bf16 SharedBuffer (F32/F16/BF16 sources). The DiT
// compute is bf16; the name keeps the historical "_f16" for call-site churn.
SharedBuffer
to_f16_(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm)
{
  const auto* info = wts.info(nm);
  if (info == nullptr || info->shape.empty()) { return {}; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { return {}; }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  auto* d = static_cast<std::uint16_t*>(out.contents());
  if (info->dtype == "F32") {
    const auto* s = static_cast<const float*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_(s[i]); }
  } else if (info->dtype == "BF16") {
    std::memcpy(d, raw.contents(), n * 2);
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_((float)s[i]); }
  } else {
    return {};
  }
  return out;
}

}  // namespace

MetalFlux2Transformer::GenerationParams
MetalFlux2Transformer::GenerationParams::from_flex(const FlexData& fd,
                                                   std::string* err)
{
  GenerationParams p;
  if (!fd.is_object()) {
    if (err != nullptr) { *err = "not a JSON object; using the defaults"; }
    return p;
  }
  auto o = fd.as_object();
  if (o.contains("klein_kv")) {
    p.klein_kv = o.at("klein_kv").as_bool(p.klein_kv);
  }
  return p;
}

SharedBuffer
MetalFlux2Transformer::elt_(WeightSet& ws, const std::string& nm, Retain r)
{
  if (r == Retain::Streamed) {
    return ws.stream_derived([&]() { return to_f16_(ws.src(), _mc, nm); });
  }
  return ws.derived(std::string(kKey) + "elt|" + nm,
                    [&]() { return to_f16_(ws.src(), _mc, nm); });
}

MetalFlux2Transformer::QWeight
MetalFlux2Transformer::load_qw_(WeightSet& ws, const std::string& name,
                                Retain r)
{
  QWeight qw;
  const MetalLlamaWeights& wts = ws.src();
  const auto* si = wts.info(name + ".scales");
  const auto* ci = wts.info(name + ".weight");
  if (_quant_bits > 0 && si != nullptr && ci != nullptr &&
      si->shape.size() == 2 && ci->shape.size() == 2) {
    const long gcols = ci->shape[1];                 // K*bits/32
    const long scols = si->shape[1];                 // K/group
    const long K = scols * (long)_quant_group;
    const int bits = K > 0 ? (int)(gcols * 32 / K) : 0;
    qw.bits = (bits == 8) ? 8 : 4;
    qw.n = (int)ci->shape[0];
    qw.k = (int)K;
    // Zero-copy mmap views (evictable) when enabled; else owned copies. These
    // codes/scales/qbias feed the GEMM read-only and are never CPU-modified, so
    // aliasing the file is safe. interleave_gu_/slice_rows_ still copy where a
    // CPU relayout is needed, correctly dropping the mapping for those.
    // Codes are U32 (raw; mmap-aliased when enabled). Scales/biases are F16 on
    // disk but the affine kernels are now bf16 (the DiT runs bf16), so convert
    // them F16->bf16 via to_f16_ (which emits bf16). Mirrors the QIE quant path.
    const auto res = _mmap_weights ? WeightSet::Residency::Mapped
                                   : WeightSet::Residency::Copied;
    qw.codes = r == Retain::Streamed
                   ? ws.stream_tensor(name + ".weight", _mc, res)
                   : ws.tensor(name + ".weight", _mc, res);
    qw.scales = elt_(ws, name + ".scales", r);
    qw.qbias  = elt_(ws, name + ".biases", r);
    if (!qw.codes.empty() && !qw.scales.empty() && !qw.qbias.empty()) {
      qw.quantized = true;
      return qw;
    }
    qw.codes = {}; qw.scales = {}; qw.qbias = {};
  }
  const auto* wi = wts.info(name + ".weight");
  if (wi != nullptr && wi->shape.size() == 2) {
    qw.n = (int)wi->shape[0];
    qw.k = (int)wi->shape[1];
  }
  qw.w = to_f16_(wts, _mc, name + ".weight");
  return qw;
}

void
MetalFlux2Transformer::interleave_gu_(WeightSet& ws, const std::string& key,
                                     QWeight& qw, Retain r)
{
  const int n = qw.n;
  if (n <= 1) { return; }
  auto perm = [&](const SharedBuffer& src) -> SharedBuffer {
    if (src.empty()) { return {}; }
    const std::size_t rb = src.byte_size() / (std::size_t)n;   // row bytes
    if (rb == 0) { return {}; }
    const int inner = n / 2;
    SharedBuffer dst = _mc->make_shared_buffer((std::size_t)n * rb);
    if (dst.empty()) { return {}; }
    const auto* s = static_cast<const std::uint8_t*>(src.contents());
    auto* d = static_cast<std::uint8_t*>(dst.contents());
    for (int g = 0; g < inner; ++g) {
      std::memcpy(d + (std::size_t)(2 * g) * rb,
                  s + (std::size_t)g * rb, rb);
      std::memcpy(d + (std::size_t)(2 * g + 1) * rb,
                  s + (std::size_t)(inner + g) * rb, rb);
    }
    return dst;
  };
  // The interleaved gate|up weight is the largest tensor in a block --
  // building it outside the set would leave the biggest single sharing
  // win on the table. Streamed blocks rebuild it per forward and keep
  // nothing, as they must.
  auto cached = [&](const char* tag, const SharedBuffer& src) -> SharedBuffer {
    if (r == Retain::Streamed) {
      return ws.stream_derived([&]() { return perm(src); });
    }
    return ws.derived(std::string(kKey) + "gu|" + key + "|" + tag,
                      [&]() { return perm(src); });
  };
  if (qw.quantized) {
    qw.codes  = cached("codes", qw.codes);
    qw.scales = cached("scales", qw.scales);
    qw.qbias  = cached("qbias", qw.qbias);
  } else {
    qw.w = cached("w", qw.w);
  }
}

bool
MetalFlux2Transformer::load_double_(WeightSet& ws,
                                    const std::string& pre, DoubleBlock& b,
                                    Retain r)
{
  b.q  = load_qw_(ws, pre + "attn.to_q", r);
  b.k  = load_qw_(ws, pre + "attn.to_k", r);
  b.v  = load_qw_(ws, pre + "attn.to_v", r);
  b.o  = load_qw_(ws, pre + "attn.to_out.0", r);
  b.aq = load_qw_(ws, pre + "attn.add_q_proj", r);
  b.ak = load_qw_(ws, pre + "attn.add_k_proj", r);
  b.av = load_qw_(ws, pre + "attn.add_v_proj", r);
  b.ao = load_qw_(ws, pre + "attn.to_add_out", r);
  b.qn  = elt_(ws, pre + "attn.norm_q.weight", r);
  b.kn  = elt_(ws, pre + "attn.norm_k.weight", r);
  b.aqn = elt_(ws, pre + "attn.norm_added_q.weight", r);
  b.akn = elt_(ws, pre + "attn.norm_added_k.weight", r);
  b.ff_in   = load_qw_(ws, pre + "ff.linear_in", r);
  b.ff_out  = load_qw_(ws, pre + "ff.linear_out", r);
  b.cff_in  = load_qw_(ws, pre + "ff_context.linear_in", r);
  b.cff_out = load_qw_(ws, pre + "ff_context.linear_out", r);
  if (_cfg.double_ff_hidden == 0 && b.ff_in.n > 0) {
    _cfg.double_ff_hidden = b.ff_in.n;
  }
  // Fused-SwiGLU: interleave the gate|up rows of linear_in so the fused kernel
  // reads even col = gate, odd col = up. (ff_out is unchanged.)
  if (_fuse_ff) {
    interleave_gu_(ws, pre + "ff.linear_in", b.ff_in, r);
    interleave_gu_(ws, pre + "ff_context.linear_in",
                   b.cff_in, r);
  }
  return !b.q.empty() && !b.k.empty() && !b.v.empty() && !b.o.empty() &&
         !b.aq.empty() && !b.ak.empty() && !b.av.empty() && !b.ao.empty() &&
         !b.qn.empty() && !b.kn.empty() && !b.aqn.empty() && !b.akn.empty() &&
         !b.ff_in.empty() && !b.ff_out.empty() && !b.cff_in.empty() &&
         !b.cff_out.empty();
}

MetalFlux2Transformer::QWeight
MetalFlux2Transformer::slice_rows_(WeightSet& ws, const std::string& key,
                                   const QWeight& src, int start, int count,
                                   Retain r)
{
  QWeight d;
  d.quantized = src.quantized;
  d.bits = src.bits;
  d.n = count;
  d.k = src.k;
  auto ext = [&](const SharedBuffer& s) -> SharedBuffer {
    if (s.empty() || src.n <= 0) { return {}; }
    const std::size_t rb = s.byte_size() / (std::size_t)src.n;   // row bytes
    SharedBuffer o = _mc->make_shared_buffer((std::size_t)count * rb);
    if (o.empty()) { return {}; }
    std::memcpy(o.contents(),
                static_cast<const std::uint8_t*>(s.contents())
                    + (std::size_t)start * rb,
                (std::size_t)count * rb);
    return o;
  };
  auto cached = [&](const char* tag, const SharedBuffer& s) -> SharedBuffer {
    if (r == Retain::Streamed) {
      return ws.stream_derived([&]() { return ext(s); });
    }
    return ws.derived(std::string(kKey) + "rows|" + key + "|" + tag,
                      [&]() { return ext(s); });
  };
  if (src.quantized) {
    d.codes = cached("codes", src.codes);
    d.scales = cached("scales", src.scales);
    d.qbias = cached("qbias", src.qbias);
  } else {
    d.w = cached("w", src.w);
  }
  return d;
}

bool
MetalFlux2Transformer::load_single_(WeightSet& ws,
                                    const std::string& pre, SingleBlock& b,
                                    Retain r, bool keep_split)
{
  b.qkv_mlp = load_qw_(ws, pre + "attn.to_qkv_mlp_proj", r);
  b.o  = load_qw_(ws, pre + "attn.to_out", r);
  b.qn = elt_(ws, pre + "attn.norm_q.weight", r);
  b.kn = elt_(ws, pre + "attn.norm_k.weight", r);
  if (_cfg.single_mlp_in == 0 && b.qkv_mlp.n > 0) {
    const int rest = b.qkv_mlp.n - 3 * _cfg.hidden;   // 2 * single_mlp_in
    if (rest > 0) { _cfg.single_mlp_in = rest / 2; }
  }
  if (!b.qkv_mlp.empty() && !b.o.empty() && !b.qn.empty() && !b.kn.empty()) {
    // Fused single block: split to_qkv_mlp_proj into the attention qkv rows and
    // the INTERLEAVED gate|up mlp rows, so the mlp runs as a fused-SwiGLU GEMM.
    if (_fuse_ff) {
      const int qkv_rows = 3 * _cfg.hidden;
      b.qkv    = slice_rows_(ws, pre + "qkv", b.qkv_mlp, 0,
                             qkv_rows, r);
      b.mlp_gu = slice_rows_(ws, pre + "mlp", b.qkv_mlp, qkv_rows,
                             b.qkv_mlp.n - qkv_rows, r);
      interleave_gu_(ws, pre + "mlp", b.mlp_gu, r);
      // KEEP THE RAW PROJECTION IN A REFILLABLE BLOCK. qkv and mlp_gu
      // are a slice and a row permutation of it, so neither has a
      // checkpoint name a raw read could address; the raw tensor does,
      // and split_single_ rebuilds both out of it after every refill.
      // A promoted block drops it again, so only the slots pay.
      if (!keep_split) {
        b.qkv_mlp = QWeight{};   // fused path uses qkv + mlp_gu instead
      }
      return !b.qkv.empty() && !b.mlp_gu.empty();
    }
    return true;
  }
  return false;
}

// ===== two reusable destinations per stack (shared/block-slots.h) =====
//
// WHAT IS AWKWARD ABOUT THIS MODEL, and it is the whole of the work
// here: the two tensors a block spends most of its bytes on are not on
// disk in the layout the forward reads. `ff.linear_in` arrives as
// gate-then-up and the fused-SwiGLU epilogue wants them INTERLEAVED, and
// `to_qkv_mlp_proj` arrives as one matrix that the fused path splits
// into an attention qkv and an interleaved gate|up. Neither product has
// a checkpoint name, so neither can be the destination of a raw read.
//
// The two are handled differently because their shapes differ. The
// double blocks' interleave is a PERMUTATION of one tensor into a buffer
// of the same size, so the read lands raw in the destination and
// riffle_rows_ permutes it where it lies -- no staging at all. The
// single blocks' split has two products, so the slot keeps the raw
// tensor (see load_single_'s keep_split) and split_single_ builds both
// out of it. That costs a slot one extra copy of its largest tensor and
// is why a promoted single block drops the raw again: the price is paid
// by the two slots, not by the resident set.

// Shorthand for the per-tensor placement the refill needs stated.
using P = vpipe::genai::Placement;

void
MetalFlux2Transformer::each_double_tensor_(
    int L, DoubleBlock& b, const BlockSlots<DoubleBlock>::TensorFn& fn) const
{
  const std::string p = "transformer_blocks." + std::to_string(L) + ".";
  // Codes are the checkpoint's own u32 words; the group scales and
  // minima are read as bf16 (F16 in the pack, converted in place), and
  // a dense weight is bf16 -- which is what to_f16_ produces for it.
  const auto qw = [&fn](const std::string& base, QWeight& w) {
    if (w.quantized) {
      fn(base + ".weight", w.codes, P::kRaw);
      fn(base + ".scales", w.scales, P::kBf16);
      fn(base + ".biases", w.qbias, P::kBf16);
    } else {
      fn(base + ".weight", w.w, P::kBf16);
    }
  };
  qw(p + "attn.to_q", b.q);
  qw(p + "attn.to_k", b.k);
  qw(p + "attn.to_v", b.v);
  qw(p + "attn.to_out.0", b.o);
  qw(p + "attn.add_q_proj", b.aq);
  qw(p + "attn.add_k_proj", b.ak);
  qw(p + "attn.add_v_proj", b.av);
  qw(p + "attn.to_add_out", b.ao);
  fn(p + "attn.norm_q.weight", b.qn, P::kBf16);
  fn(p + "attn.norm_k.weight", b.kn, P::kBf16);
  fn(p + "attn.norm_added_q.weight", b.aqn, P::kBf16);
  fn(p + "attn.norm_added_k.weight", b.akn, P::kBf16);
  // The two linear_in weights land RAW here -- gate rows then up rows --
  // and post_refill riffles them into the interleaved order. Enumerating
  // them is therefore correct even though the bytes the forward reads
  // are not the bytes on disk: the destination is the right size and the
  // permutation happens after the read, in place.
  qw(p + "ff.linear_in", b.ff_in);
  qw(p + "ff.linear_out", b.ff_out);
  qw(p + "ff_context.linear_in", b.cff_in);
  qw(p + "ff_context.linear_out", b.cff_out);
}

void
MetalFlux2Transformer::each_single_tensor_(
    int L, SingleBlock& b, const BlockSlots<SingleBlock>::TensorFn& fn) const
{
  const std::string p =
      "single_transformer_blocks." + std::to_string(L) + ".";
  const auto qw = [&fn](const std::string& base, QWeight& w) {
    if (w.quantized) {
      fn(base + ".weight", w.codes, P::kRaw);
      fn(base + ".scales", w.scales, P::kBf16);
      fn(base + ".biases", w.qbias, P::kBf16);
    } else {
      fn(base + ".weight", w.w, P::kBf16);
    }
  };
  // qkv and mlp_gu are DELIBERATELY absent: they are a slice and a
  // permutation of this one tensor, split_single_ rebuilds them after
  // the read, and a slot that skipped that would keep the FIRST block's
  // projection for the whole run.
  qw(p + "attn.to_qkv_mlp_proj", b.qkv_mlp);
  qw(p + "attn.to_out", b.o);
  fn(p + "attn.norm_q.weight", b.qn, P::kBf16);
  fn(p + "attn.norm_k.weight", b.kn, P::kBf16);
}

// [g0..g_{m-1} | u0..u_{m-1}] -> [g0,u0,g1,u1,...], where it lies.
// The walk itself is shared/riffle-rows.h -- MiniMax-H3's fc1 wants the
// identical permutation over the identical layout.
//
// EVERY buffer is checked before ANY is permuted. A quantized weight is
// three buffers that only mean something together, and the walk cannot
// report a failure once started, so a refusal discovered between two
// walks would leave the weight half-permuted.
bool
MetalFlux2Transformer::riffle_rows_(QWeight& qw) const
{
  const std::size_t n = (std::size_t)qw.n;
  if (qw.n <= 1) { return true; }
  if (qw.quantized) {
    if (!genai::riffle_rows_ok(qw.codes, n)
        || !genai::riffle_rows_ok(qw.scales, n)
        || !genai::riffle_rows_ok(qw.qbias, n)) {
      return false;
    }
    genai::riffle_rows(qw.codes, n);
    genai::riffle_rows(qw.scales, n);
    genai::riffle_rows(qw.qbias, n);
    return true;
  }
  if (!genai::riffle_rows_ok(qw.w, n)) { return false; }
  genai::riffle_rows(qw.w, n);
  return true;
}

// Rebuild a single block's fused halves out of the raw projection it
// kept. Both destinations already exist and keep their allocations --
// that is what makes this a refill rather than a reload.
bool
MetalFlux2Transformer::split_single_(SingleBlock& b) const
{
  if (b.qkv.empty() || b.mlp_gu.empty()) { return true; }   // not fused
  if (b.qkv_mlp.empty()) { return false; }
  const std::size_t src_n = (std::size_t)b.qkv_mlp.n;
  const std::size_t head  = (std::size_t)(3 * _cfg.hidden);
  if (src_n <= head) { return false; }
  const std::size_t mlp = src_n - head;         // 2 * single_mlp_in
  if ((mlp & 1) != 0) { return false; }
  const std::size_t inner = mlp / 2;
  bool ok = true;
  const auto one = [&](const SharedBuffer& src, SharedBuffer& qkv,
                       SharedBuffer& gu) {
    if (!ok || src.empty()) { return; }
    const std::size_t rb = src.byte_size() / src_n;
    if (rb == 0 || rb * src_n != src.byte_size() ||
        qkv.byte_size() != head * rb || gu.byte_size() != mlp * rb) {
      ok = false;
      return;
    }
    const auto* sp = static_cast<const std::uint8_t*>(src.contents());
    std::memcpy(qkv.contents(), sp, head * rb);
    auto* gp = static_cast<std::uint8_t*>(gu.contents());
    for (std::size_t g = 0; g < inner; ++g) {
      std::memcpy(gp + (2 * g) * rb, sp + (head + g) * rb, rb);
      std::memcpy(gp + (2 * g + 1) * rb, sp + (head + inner + g) * rb, rb);
    }
  };
  if (b.qkv_mlp.quantized) {
    one(b.qkv_mlp.codes, b.qkv.codes, b.mlp_gu.codes);
    one(b.qkv_mlp.scales, b.qkv.scales, b.mlp_gu.scales);
    one(b.qkv_mlp.qbias, b.qkv.qbias, b.mlp_gu.qbias);
  } else {
    one(b.qkv_mlp.w, b.qkv.w, b.mlp_gu.w);
  }
  return ok;
}

// Allocate `dst` with `src`'s shapes and flags, optionally copying the
// bytes. One function for two uses: a promotion and the second slot
// differ only in whether the contents come along.
bool
MetalFlux2Transformer::clone_double_(const DoubleBlock& src, DoubleBlock& dst,
                                     bool copy) const
{
  bool ok = true;
  const auto one = [&](const SharedBuffer& s, SharedBuffer& d) {
    if (!ok || s.empty()) { d = SharedBuffer{}; return; }
    d = _mc->make_shared_buffer(s.byte_size());
    if (d.empty()) { ok = false; return; }
    if (copy) { std::memcpy(d.contents(), s.contents(), s.byte_size()); }
  };
  const auto qw = [&](const QWeight& s, QWeight& d) {
    d.quantized = s.quantized; d.bits = s.bits; d.n = s.n; d.k = s.k;
    one(s.w, d.w); one(s.codes, d.codes);
    one(s.scales, d.scales); one(s.qbias, d.qbias);
  };
  qw(src.q, dst.q); qw(src.k, dst.k); qw(src.v, dst.v); qw(src.o, dst.o);
  qw(src.aq, dst.aq); qw(src.ak, dst.ak); qw(src.av, dst.av);
  qw(src.ao, dst.ao);
  one(src.qn, dst.qn); one(src.kn, dst.kn);
  one(src.aqn, dst.aqn); one(src.akn, dst.akn);
  qw(src.ff_in, dst.ff_in); qw(src.ff_out, dst.ff_out);
  qw(src.cff_in, dst.cff_in); qw(src.cff_out, dst.cff_out);
  if (!ok) { dst = DoubleBlock{}; }
  return ok;
}

bool
MetalFlux2Transformer::clone_single_(const SingleBlock& src, SingleBlock& dst,
                                     bool copy) const
{
  bool ok = true;
  const auto one = [&](const SharedBuffer& s, SharedBuffer& d) {
    if (!ok || s.empty()) { d = SharedBuffer{}; return; }
    d = _mc->make_shared_buffer(s.byte_size());
    if (d.empty()) { ok = false; return; }
    if (copy) { std::memcpy(d.contents(), s.contents(), s.byte_size()); }
  };
  const auto qw = [&](const QWeight& s, QWeight& d) {
    d.quantized = s.quantized; d.bits = s.bits; d.n = s.n; d.k = s.k;
    one(s.w, d.w); one(s.codes, d.codes);
    one(s.scales, d.scales); one(s.qbias, d.qbias);
  };
  qw(src.qkv_mlp, dst.qkv_mlp);
  qw(src.qkv, dst.qkv); qw(src.mlp_gu, dst.mlp_gu);
  qw(src.o, dst.o);
  one(src.qn, dst.qn); one(src.kn, dst.kn);
  if (!ok) { dst = SingleBlock{}; }
  return ok;
}

SharedBuffer
MetalFlux2Transformer::rebuild_one_(const std::string& nm, P how)
{
  if (!_ws) { return {}; }
  if (how == P::kRaw) {
    // Quantized CODES, which load_qw_ reads as the checkpoint's own
    // words. Copied rather than Mapped: a streamed block is never
    // mapped (weights_may_be_mapped says so), and a mapped destination
    // could not be refilled next time round.
    return _ws->stream_tensor(nm, _mc, WeightSet::Residency::Copied);
  }
  return elt_(*_ws, nm, Retain::Streamed);
}

void
MetalFlux2Transformer::configure_slots_()
{
  {
    BlockSlots<DoubleBlock>::Ops o;
    o.each = [this](int L, DoubleBlock& b,
                    const BlockSlots<DoubleBlock>::TensorFn& fn) {
      each_double_tensor_(L, b, fn);
    };
    o.rebuild_one = [this](const std::string& nm, P how) {
      return rebuild_one_(nm, how);
    };
    o.build = [this](int L, DoubleBlock& b) {
      return _ws && load_double_(
          *_ws, "transformer_blocks." + std::to_string(L) + ".", b,
          Retain::Streamed);
    };
    o.clone = [this](const DoubleBlock& s, DoubleBlock& d, bool copy) {
      return clone_double_(s, d, copy);
    };
    o.bytes = [](const DoubleBlock& b) { return double_bytes_(b); };
    o.empty = [](const DoubleBlock& b) { return b.q.empty(); };
    o.post_refill = [this](int, DoubleBlock& b) {
      if (!_fuse_ff) { return true; }
      return riffle_rows_(b.ff_in) && riffle_rows_(b.cff_in);
    };
    _double_slots.set_weight_set(_ws.get());
    _double_slots.configure(_mc, std::move(o),
                            "MetalFlux2Transformer(double)",
                            "VPIPE_FLUX2_NO_SLOTS");
  }
  {
    BlockSlots<SingleBlock>::Ops o;
    o.each = [this](int L, SingleBlock& b,
                    const BlockSlots<SingleBlock>::TensorFn& fn) {
      each_single_tensor_(L, b, fn);
    };
    o.rebuild_one = [this](const std::string& nm, P how) {
      return rebuild_one_(nm, how);
    };
    o.build = [this](int L, SingleBlock& b) {
      return _ws && load_single_(
          *_ws, "single_transformer_blocks." + std::to_string(L) + ".", b,
          Retain::Streamed, /*keep_split=*/true);
    };
    o.clone = [this](const SingleBlock& s, SingleBlock& d, bool copy) {
      return clone_single_(s, d, copy);
    };
    o.bytes = [](const SingleBlock& b) { return single_bytes_(b); };
    o.empty = [](const SingleBlock& b) { return b.o.empty(); };
    o.post_refill = [this](int, SingleBlock& b) { return split_single_(b); };
    _single_slots.set_weight_set(_ws.get());
    _single_slots.configure(_mc, std::move(o),
                            "MetalFlux2Transformer(single)",
                            "VPIPE_FLUX2_NO_SLOTS");
  }
}

MetalFlux2Transformer::~MetalFlux2Transformer()
{
  // GIVE THE POOL BACK. Freeing a wired buffer unwires it in the kernel,
  // so the machine recovers either way -- but the pool's own counter
  // would not, and a DiT freed for the vae-decode and reloaded on the
  // next prompt (free_flux2_dit_for_decode_) would leak its whole share
  // of the budget per prompt until nothing could wire at all.
  if (_wire.on()) {
    wire_fixed_(false);
    for (DoubleBlock& b : _double) { wire_block_(b, false); }
    for (SingleBlock& b : _single) { wire_block_(b, false); }
  }
}

// Does the adapter name a projection the fused paths swallow?
//
// TWO of them here, against Krea-2's one. `ff.linear_in` (and its
// ff_context twin) is a gate|up matrix the fused-SwiGLU epilogue reads
// INTERLEAVED, and the single blocks' `to_qkv_mlp_proj` is sliced into
// an attention qkv plus an interleaved gate|up. In both the activation
// is applied inside the GEMM's register-local epilogue, so a delta
// computed from the same input has nowhere to land.
//
// The single-block case is the one that bites: `to_qkv_mlp_proj` is
// ONE module covering the attention rows AND the mlp rows, so its B
// could be split by rows and the attention half applied on its own --
// and that is exactly the trap. Half an adapter is not a cheaper
// adapter, it is a wrong one that looks like it worked. So the whole
// module is served, and the fusion is what gives way.
//
// NO NAME MAP is passed, because this family has none: FLUX.2 adapters
// are diffusers-spelled and a BFL/ComfyUI repack is refused outright at
// bind. If one is ever added it MUST be passed here too -- the needles
// are the model's names, so a renamed file would bind through the map
// and then meet a fused kernel that drops its delta. Krea-2 keeps its
// map in one accessor for exactly that reason.
bool
MetalFlux2Transformer::lora_forbids_fusion(const std::string& path)
{
  return lora::Adapter::file_touches(path, ".ff.linear_in.lora_") ||
         lora::Adapter::file_touches(path, ".ff_context.linear_in.lora_") ||
         lora::Adapter::file_touches(path, ".to_qkv_mlp_proj.lora_");
}

// ---- the module tables ---------------------------------------------
//
// The names ARE the diffusers spelling, which is what the checkpoint
// itself carries -- the double blocks keep to_q / to_k / to_v /
// to_out.0 and their add_* twins split, and the single blocks really do
// have ONE Linear named `to_qkv_mlp_proj` in the model rather than
// three that this loader happened to fuse. So an adapter trained
// against diffusers names the same modules these bind, and nothing has
// to be de-fused to find them.
//
// Dimensions come from Config, not from the loaded QWeights: in
// streaming mode no block exists when the binder runs, and the config's
// dims are derived before it either way.

std::vector<MetalFlux2Transformer::DoubleLoraModule>
MetalFlux2Transformer::lora_double_modules_(const Config& c)
{
  const int H = c.hidden;
  const int DFF = c.double_ff_hidden;      // ff.linear_in out = 2*INNER
  const int INNER = DFF / 2;
  std::vector<DoubleLoraModule> out;
  out.reserve((std::size_t)c.n_double * 12);
  for (int i = 0; i < c.n_double; ++i) {
    const std::string p = "transformer_blocks." + std::to_string(i) + ".";
    auto add = [&](const char* nm, int n, int k,
                   lora::Factors DoubleLora::*dst) {
      out.push_back({p + nm, n, k, i, dst});
    };
    add("attn.to_q",       H, H, &DoubleLora::q);
    add("attn.to_k",       H, H, &DoubleLora::k);
    add("attn.to_v",       H, H, &DoubleLora::v);
    add("attn.to_out.0",   H, H, &DoubleLora::o);
    add("attn.add_q_proj", H, H, &DoubleLora::aq);
    add("attn.add_k_proj", H, H, &DoubleLora::ak);
    add("attn.add_v_proj", H, H, &DoubleLora::av);
    add("attn.to_add_out", H, H, &DoubleLora::ao);
    add("ff.linear_in",          DFF, H,     &DoubleLora::ff_in);
    add("ff.linear_out",         H,   INNER, &DoubleLora::ff_out);
    add("ff_context.linear_in",  DFF, H,     &DoubleLora::cff_in);
    add("ff_context.linear_out", H,   INNER, &DoubleLora::cff_out);
  }
  return out;
}

std::vector<MetalFlux2Transformer::SingleLoraModule>
MetalFlux2Transformer::lora_single_modules_(const Config& c)
{
  const int H = c.hidden, SMLP = c.single_mlp_in;
  const int PW = 3 * H + 2 * SMLP;         // to_qkv_mlp_proj out width
  std::vector<SingleLoraModule> out;
  out.reserve((std::size_t)c.n_single * 2);
  for (int i = 0; i < c.n_single; ++i) {
    const std::string p =
        "single_transformer_blocks." + std::to_string(i) + ".";
    out.push_back({p + "attn.to_qkv_mlp_proj", PW, H, i,
                   &SingleLora::qkv_mlp});
    out.push_back({p + "attn.to_out", H, H + SMLP, i, &SingleLora::o});
  }
  return out;
}

std::vector<MetalFlux2Transformer::LoraModule>
MetalFlux2Transformer::lora_module_list(const Config& cfg)
{
  std::vector<LoraModule> out;
  for (const auto& m : lora_double_modules_(cfg)) {
    out.push_back({m.name, m.n, m.k});
  }
  for (const auto& m : lora_single_modules_(cfg)) {
    out.push_back({m.name, m.n, m.k});
  }
  return out;
}

// Read the adapter and bind its factors to the modules they name.
//
// Scope is the transformer BLOCKS. The embedders, the shared modulation
// and proj_out are deliberately not bound: no style adapter touches
// them, and the ones that do (control adapters widening x_embedder's
// input) change a SHAPE, which is a different conversation than a
// rank-r side GEMM.
bool
MetalFlux2Transformer::bind_lora_(const LoraSpec& spec, std::string* err)
{
  std::string oerr;
  auto ad = lora::Adapter::open(spec.path, _mc, &oerr);
  if (!ad) {
    if (err != nullptr) { *err = "flux2 lora: " + oerr; }
    return false;
  }
  _lora_scale = spec.scale;

  _lora_double.clear();
  _lora_double.resize((std::size_t)_cfg.n_double);
  for (const auto& m : lora_double_modules_(_cfg)) {
    ad->bind(m.name, m.n, m.k,
             &(_lora_double[(std::size_t)m.block].*m.dst));
  }
  _lora_single.clear();
  _lora_single.resize((std::size_t)_cfg.n_single);
  for (const auto& m : lora_single_modules_(_cfg)) {
    ad->bind(m.name, m.n, m.k,
             &(_lora_single[(std::size_t)m.block].*m.dst));
  }
  _lora_modules  = ad->modules();
  _lora_max_rank = ad->max_rank();
  if (_lora_modules == 0) {
    _lora_double.clear();
    _lora_single.clear();
    if (err != nullptr) {
      *err = "flux2 lora: '" + spec.path +
             "' adapts none of this model's modules. The names must be the "
             "diffusers spelling the checkpoint uses, optionally under a "
             "'diffusion_model.' prefix";
    }
    return false;
  }
  if (_mc->session() != nullptr) {
    _mc->session()->info(fmt("MetalFlux2Transformer: {}",
                                   ad->summary(spec.path, _lora_scale)));
  }
  return true;
}

std::unique_ptr<MetalFlux2Transformer>
MetalFlux2Transformer::load(const std::string& model_dir, MetalCompute* mc,
                            const Config& cfg, bool stream_blocks,
                            const LoraSpec* lora)
{
  return load(WeightSet::open(model_dir, nullptr), mc, cfg, stream_blocks,
              lora);
}

std::unique_ptr<MetalFlux2Transformer>
MetalFlux2Transformer::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                            const Config& cfg, bool stream_blocks,
                            const LoraSpec* lora)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  const std::string model_dir = ws_in->dir();

  auto m = std::unique_ptr<MetalFlux2Transformer>(new MetalFlux2Transformer());
  m->_ws = std::move(ws_in);
  m->_mc = mc;
  m->_cfg = cfg;
  m->_stream_blocks = stream_blocks;
  // BEFORE the first weight is read, because it decides how they are read.
  m->_wire.open(mc);
  // Zero-copy mmap of the quantized weights (see _mmap_weights) so the DiT's
  // resident footprint stays reclaimable under memory pressure. Off when
  // streaming (blocks re-read JIT), when the wired pool is on (a mapped view
  // can be neither mlock'd nor parked -- see shared/wired-pool.h), or via
  // VPIPE_FLUX2_NO_MMAP_WEIGHTS. Retain the source mmap for the model's
  // lifetime so the mapped views stay valid.
  m->_mmap_weights = weights_may_be_mapped(stream_blocks, m->_wire.on()) &&
                     std::getenv("VPIPE_FLUX2_NO_MMAP_WEIGHTS") == nullptr;
  WeightSet& ws = *m->_ws;
  // Everything loaded from here to the end of load() is RETAINED for the
  // model's life. The streamed blocks are read in forward(), and there
  // only.
  const Retain r = Retain::Cached;

  {
    namespace fs = std::filesystem;
    std::ifstream in(fs::path(model_dir) / "config.json");
    if (in) {
      FlexData cfgj = FlexData::from_json(in);
      if (cfgj.is_object()) {
        auto obj = cfgj.as_object();
        // Structural dims from config.json so the same code drives every
        // Flux2Transformer2DModel size (klein-4B: 5+20 blocks, 24 heads, no
        // guidance; klein-9B: a larger DiT with guidance_embeds). Absent keys
        // keep the Config default. hidden = num_attention_heads * head_dim.
        auto geti = [&](const char* k, int cur) -> int {
          return obj.contains(k) ? (int)obj.at(k).as_int(cur) : cur;
        };
        auto getf = [&](const char* k, float cur) -> float {
          return obj.contains(k) ? (float)obj.at(k).as_real(cur) : cur;
        };
        m->_cfg.n_heads      = geti("num_attention_heads", m->_cfg.n_heads);
        m->_cfg.head_dim     = geti("attention_head_dim", m->_cfg.head_dim);
        m->_cfg.hidden       = m->_cfg.n_heads * m->_cfg.head_dim;
        m->_cfg.n_double     = geti("num_layers", m->_cfg.n_double);
        m->_cfg.n_single     = geti("num_single_layers", m->_cfg.n_single);
        m->_cfg.in_channels  = geti("in_channels", m->_cfg.in_channels);
        m->_cfg.joint_dim    = geti("joint_attention_dim", m->_cfg.joint_dim);
        m->_cfg.timestep_dim =
            geti("timestep_guidance_channels", m->_cfg.timestep_dim);
        m->_cfg.mlp_ratio    = getf("mlp_ratio", m->_cfg.mlp_ratio);
        m->_cfg.norm_eps     = getf("eps", m->_cfg.norm_eps);
        m->_cfg.rope_theta   = getf("rope_theta", m->_cfg.rope_theta);
        if (obj.contains("guidance_embeds")) {
          m->_cfg.guidance_embeds =
              obj.at("guidance_embeds").as_bool(m->_cfg.guidance_embeds);
        }
        if (obj.contains("axes_dims_rope")) {
          FlexData ax = obj.at("axes_dims_rope");
          if (ax.is_array()) {
            auto av = ax.as_array();
            for (int i = 0; i < 4 && i < (int)av.size(); ++i) {
              m->_cfg.axes_dim[i] = (int)av[i].as_int(m->_cfg.axes_dim[i]);
            }
          }
        }
        if (obj.contains("quantization")) {
          FlexData q = obj.at("quantization");
          if (q.is_object()) {
            auto qo = q.as_object();
            const int b = (int)qo.at("bits").as_int(0);
            const int g = (int)qo.at("group_size").as_int(64);
            if (b == 4 || b == 8) {
              m->_quant_bits = b;
              m->_quant_group = (g == 32 || g == 64) ? g : 64;
            }
          }
        }
      }
    }
  }

  // bf16 metallibs (VPIPE_ELT=bfloat); entry-point names keep the "_f16" label.
  // gelu/layernorm come from llm_elementwise (the qwen3_5_vision gelu_tanh_f16 /
  // layer_norm_bias_f16 are half-only): gelu_tanh_ff_f16 (same x,out,n sig) and
  // layer_norm_plain_f16 (no-affine; flux2's op.ln uses identity weight/bias).
  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_lib_sdpa = mc->load_library("sdpa_bf16");
  m->_lib_vis  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_fn_gemm        = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_gemm_bm64   = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_gemm_bm64bn64 = m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  m->_fn_gemm_bm64_a16 = m->_lib_gemm.function("dense_gemm_t_bm64_acc16_f16");
  m->_fn_ff_swiglu    = m->_lib_gemm.function("dense_gemm_swiglu_bm64_f16");
  m->_fn_ff_swiglu_a16 =
      m->_lib_gemm.function("dense_gemm_swiglu_bm64_acc16_f16");
  m->_fn_gemm_bias   = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_rms         = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_swiglu      = m->_lib_elt.function("swiglu_f16");
  m->_fn_residual    = m->_lib_elt.function("residual_add_f16");
  m->_fn_transpose   = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_sdpa        = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_gelu_tanh   = m->_lib_elt.function("gelu_tanh_ff_f16");
  m->_fn_layernorm   = m->_lib_elt.function("layer_norm_plain_f16");
  m->_fn_rope_table  = m->_lib_rope.function("rope_pair_table_ftab_f16");
  m->_fn_transpose_rope =
      m->_lib_rope.function("transpose_rope_pair_ftab_f16");
  m->_fn_ln_mod      = m->_lib_elt.function("layernorm_modulate_f16");
  m->_fn_adaln       = m->_lib_elt.function("adaln_modulate_f16");
  m->_fn_gated       = m->_lib_elt.function("gated_residual_f16");
  // vec4 twins of the adaLN / gate passes. One element per thread leaves those
  // kernels at ~37-54 GB/s where the same bytes through a vec4 2-D grid run at
  // 143-181 (measured on an M4 Pro, boogu_perf.elementwise_shapes): the limit is
  // threads retired, not bandwidth. Same arithmetic per element, so the result
  // is bit-identical. VPIPE_NO_ELT_V4 reverts to the scalar kernels for A/B.
  if (std::getenv("VPIPE_NO_ELT_V4") == nullptr) {
    m->_fn_adaln4 = m->_lib_elt.function("adaln_modulate_v4_f16");
    m->_fn_gated4 = m->_lib_elt.function("gated_residual_v4_f16");
  }
  m->_fn_bias_add    = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_headslice   = m->_lib_elt.function("head_slice_f16");
  m->_fn_mulsig      = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_concat      = m->_lib_elt.function("concat_cols_f16");
  m->_fn_transpose_rs = m->_lib_elt.function("transpose_abd_rs_f16");
  m->_fn_swiglu_rs   = m->_lib_elt.function("swiglu_rs_f16");
  m->_fn_colabsmax   = m->_lib_elt.function("col_absmax_f16");   // AWQ tap
  if (!m->_fn_gemm.valid() || !m->_fn_gemm_bias.valid() ||
      !m->_fn_rms.valid() ||
      !m->_fn_swiglu.valid() || !m->_fn_residual.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_gelu_tanh.valid() || !m->_fn_layernorm.valid() ||
      !m->_fn_rope_table.valid() || !m->_fn_adaln.valid() ||
      !m->_fn_gated.valid() || !m->_fn_bias_add.valid() ||
      !m->_fn_headslice.valid() || !m->_fn_mulsig.valid() ||
      !m->_fn_concat.valid() ||
      !m->_fn_colabsmax.valid()) {
    return nullptr;
  }
  if (m->_quant_bits > 0) {
    m->_lib_qmm = mc->load_library("affine_qmm_steel_bf16");
    const std::string g = "g" + std::to_string(m->_quant_group);
    m->_fn_qmm4 = m->_lib_qmm.function("affine_qmm_steel_w4" + g);
    m->_fn_qmm8 = m->_lib_qmm.function("affine_qmm_steel_w8" + g);
    if (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid()) { return nullptr; }
    // BM128 twins (g64 only) for the big M = seq quant GEMMs.
    m->_fn_qmm4_bm128 = m->_lib_qmm.function("affine_qmm_steel_w4g64_bm128");
    m->_fn_qmm8_bm128 = m->_lib_qmm.function("affine_qmm_steel_w8g64_bm128");
    m->_qmm_tile = (m->_quant_group == 64 && m->_fn_qmm4_bm128.valid()
                    && m->_fn_qmm8_bm128.valid()) ? 1 : 0;
    if (const char* t = std::getenv("VPIPE_FLUX2_QMM_TILE")) {
      m->_qmm_tile = std::atoi(t);
    }
    m->_fn_qmm_swiglu4_bm64 =
        m->_lib_qmm.function("affine_qmm_swiglu_w4g64_bm64");
    m->_fn_qmm_swiglu8_bm64 =
        m->_lib_qmm.function("affine_qmm_swiglu_w8g64_bm64");
    m->_fn_qmm_swiglu4_bm64_a16 =
        m->_lib_qmm.function("affine_qmm_swiglu_w4g64_bm64_acc16");
    m->_fn_qmm_swiglu8_bm64_a16 =
        m->_lib_qmm.function("affine_qmm_swiglu_w8g64_bm64_acc16");
    m->_fn_qmm_swiglu4_bm64_rs =
        m->_lib_qmm.function("affine_qmm_swiglu_w4g64_bm64_rs");
    m->_fn_qmm_swiglu8_bm64_rs =
        m->_lib_qmm.function("affine_qmm_swiglu_w8g64_bm64_rs");
    m->_fn_qmm_swiglu4_bm64_rs_a16 =
        m->_lib_qmm.function("affine_qmm_swiglu_w4g64_bm64_rs_acc16");
    m->_fn_qmm_swiglu8_bm64_rs_a16 =
        m->_lib_qmm.function("affine_qmm_swiglu_w8g64_bm64_rs_acc16");
  }
  // M5 matrix-core matmul2d for the block/projection GEMMs (mirrors Krea-2):
  // dense weights feed dense_gemm_mma directly; quantized weights dequant-
  // expand into _w_deq (affine_dequant) then run the SAME dense matmul2d --
  // dequant-once -> dense beats the fused steel qmm on matrix-core GPUs (the
  // qmm's tgmem staging cannot feed the MMA rate). Gated on matrix cores
  // (M4/older keep steel); VPIPE_FLUX2_NO_MMA2 A/B off. Decided BEFORE the
  // fused-FF choice below (the mma path defaults the fusion off).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_FLUX2_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    // TN=2 tile for the mid-K band (x-reuse doubles); NO_TN2 forces the plain
    // 128x256 deep tile for A/B (leaves the fn invalid -> routing skips it).
    if (std::getenv("VPIPE_FLUX2_NO_TN2") == nullptr) {
      m->_fn_dense_mma_tn2 =
          m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");
    }
    m->_fn_dense_mma_splitk =
        m->_lib_dense_mma.function("dense_gemm_mma_splitk_n128x256_k8192_f16");
    m->_use_mma2 =
        m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid();
    // Quantized checkpoint: group-matched dequant kernels feed the dense
    // matmul2d. Both bit widths loaded (mixed-precision per-weight w4/w8).
    if (m->_use_mma2 && m->_quant_bits > 0) {
      m->_lib_dequant = mc->load_library("affine_dequant_bf16");
      const std::string dg = "g" + std::to_string(m->_quant_group);
      m->_fn_dequant4 = m->_lib_dequant.function("affine_dequant_w4" + dg);
      m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant_w8" + dg);
      if (!m->_fn_dequant4.valid() || !m->_fn_dequant8.valid()) {
        m->_use_mma2 = false;   // fall back to steel qmm
      }
    }
    if (const char* e = std::getenv("VPIPE_FLUX2_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    m->_use_splitk = m->_use_mma2 && m->_fn_dense_mma_splitk.valid()
                     && m->_fn_residual.valid()
                     && std::getenv("VPIPE_FLUX2_NO_SPLITK") == nullptr;
  }
  // Dynamic-int8 accelerated GEMMs (opt-in, LOSSY): tried first in
  // gemm_mma_ for the big block matmuls. Config::i8_gemm is the stage
  // switch; VPIPE_I8_GEMM overrides (the context self-gates on matrix
  // cores + kernel availability).
  {
    // The flux2 DiT runs bf16, so the i8 path must read/write bf16 (its x, the
    // dequant scratch weight, and y are all bf16); load the _bf16 i8 kernels.
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm, /*bf16=*/true);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }
  // Fuse the SwiGLU FF (default on, EXCEPT on the matmul2d path): needs the
  // dense swiglu twin, and -- for a quantized DiT -- the g64 qmm swiglu twins.
  // The weight dequant is F16 (the f16 metallib's native-half path).
  // Accumulate defaults to FLOAT: it is MORE accurate than the unfused path
  // (no [seq,2*INNER] f16 round-trip -> 4B golden 0.0036 vs unfused 0.0044) at
  // ~the same speed, and keeps clear of the golden bound. VPIPE_FLUX2_FF_ACC16
  // opts into the FP16-pipe accumulate (~1% faster, ~4x more drift);
  // VPIPE_FLUX2_NO_FUSE_FF disables the fusion entirely. On the mma path the
  // fused GEMM's register-local epilogue would keep the FF OFF the matrix
  // cores (steel rate), while the unfused two-GEMM+swiglu runs at the matmul2d
  // rate -- so _use_mma2 defaults the fusion off; VPIPE_FLUX2_FUSE_FF=1
  // forces it back on there (A/B).
  m->_ff_acc16 = std::getenv("VPIPE_FLUX2_FF_ACC16") != nullptr;
  m->_fuse_ff = std::getenv("VPIPE_FLUX2_NO_FUSE_FF") == nullptr
                && (!m->_use_mma2
                    || std::getenv("VPIPE_FLUX2_FUSE_FF") != nullptr)
                && m->_fn_ff_swiglu.valid() && m->_fn_ff_swiglu_a16.valid()
                && (m->_quant_bits == 0
                    || (m->_quant_group == 64
                        && m->_fn_qmm_swiglu4_bm64.valid()
                        && m->_fn_qmm_swiglu8_bm64.valid()));
  // AN ADAPTED FF CANNOT BE FUSED, and on this model that reaches the
  // single blocks too -- their whole projection is one fused matrix.
  // Asked from the FILE HEADER because the answer decides how the
  // blocks are BUILT (interleave_gu_ / slice_rows_ below both read
  // _fuse_ff), which happens long before there is a model to bind an
  // adapter to.
  if (m->_fuse_ff && lora != nullptr && !lora->path.empty() &&
      lora_forbids_fusion(lora->path)) {
    m->_fuse_ff = false;
    if (mc->session() != nullptr) {
      mc->session()->log_normal(fmt(
          "MetalFlux2Transformer: the adapter touches a FUSED projection "
          "(ff.linear_in / to_qkv_mlp_proj), so the fused-SwiGLU weave is "
          "off for this load -- the split path is the only one a "
          "pre-activation delta can reach"));
    }
  }
  m->_lib_attn = mc->load_library("attn_steel");
  m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));
  m->_steel_attn_ok = m->_lib_attn.valid() && !m->_attn_params.empty()
                      && std::getenv("VPIPE_FLUX2_NO_STEEL_ATTN") == nullptr;
  m->_lib_attn_nax = mc->load_library("attn_steel_nax");
  // The DiT runs bf16; use the bf16 NAX entry (attn_steel_nax_h_bd128_bf16) on
  // M5 matrix-core GPUs, else the non-nax bf16 steel attention.
  m->_use_attn_nax = m->_steel_attn_ok && mc->supports_matrix_cores()
                     && m->_lib_attn_nax.valid()
                     && std::getenv("VPIPE_FLUX2_NO_ATTN_NAX") == nullptr;
  // Dense f16 STEEL GEMM tile (the fallback when the matmul2d route above is
  // off/absent). MEASURED on M4 (9B @1024, M=seq=4136): the dense f16 GEMM is
  // ALU/compute-bound (M amortizes weight reuse), so larger CTA tiles do NOT
  // help -- BM64 ran ~7% SLOWER and BM64xBN64 ~even vs the 32x32 base. So
  // default to the base tile. VPIPE_FLUX2_GEMM_TILE (0|1|2) +
  // VPIPE_FLUX2_GEMM_ACC16 keep the A/B knob.
  m->_gemm_tile = 0;
  if (const char* t = std::getenv("VPIPE_FLUX2_GEMM_TILE")) {
    m->_gemm_tile = std::atoi(t);
  }
  m->_acc16 = std::getenv("VPIPE_FLUX2_GEMM_ACC16") != nullptr;
  if (m->_gemm_tile == 1 && !m->_fn_gemm_bm64.valid()) { m->_gemm_tile = 0; }
  if (m->_gemm_tile == 2 && !m->_fn_gemm_bm64bn64.valid()) {
    m->_gemm_tile = m->_fn_gemm_bm64.valid() ? 1 : 0;
  }

  m->_x_embed   = m->load_qw_(ws, "x_embedder", r);
  m->_ctx_embed = m->load_qw_(ws, "context_embedder", r);
  m->_t_emb1 =
      m->load_qw_(ws, "time_guidance_embed.timestep_embedder.linear_1", r);
  m->_t_emb1_b =
      m->elt_(ws, "time_guidance_embed.timestep_embedder.linear_1.bias", r);
  m->_t_emb2 =
      m->load_qw_(ws, "time_guidance_embed.timestep_embedder.linear_2", r);
  m->_t_emb2_b =
      m->elt_(ws, "time_guidance_embed.timestep_embedder.linear_2.bias", r);
  // Guidance-distilled variants (klein-9B): the guidance_embedder is a second
  // TimestepEmbedding whose output is added to the timestep embedding. Absent
  // in the distilled 4B (guidance_embeds=false), so load only when configured.
  if (m->_cfg.guidance_embeds) {
    m->_g_emb1 =
        m->load_qw_(ws, "time_guidance_embed.guidance_embedder.linear_1", r);
    m->_g_emb1_b =
        m->elt_(ws, "time_guidance_embed.guidance_embedder.linear_1.bias", r);
    m->_g_emb2 =
        m->load_qw_(ws, "time_guidance_embed.guidance_embedder.linear_2", r);
    m->_g_emb2_b =
        m->elt_(ws, "time_guidance_embed.guidance_embedder.linear_2.bias", r);
    if (m->_g_emb1.empty() || m->_g_emb2.empty()) { return nullptr; }
  }
  m->_mod_img    = m->load_qw_(ws, "double_stream_modulation_img.linear", r);
  m->_mod_txt    = m->load_qw_(ws, "double_stream_modulation_txt.linear", r);
  m->_mod_single = m->load_qw_(ws, "single_stream_modulation.linear", r);
  m->_proj_out   = m->load_qw_(ws, "proj_out", r);
  m->_norm_out_lin = m->load_qw_(ws, "norm_out.linear", r);
  if (m->_x_embed.empty() || m->_ctx_embed.empty() || m->_t_emb1.empty() ||
      m->_t_emb2.empty() || m->_mod_img.empty() || m->_mod_txt.empty() ||
      m->_mod_single.empty() || m->_proj_out.empty() ||
      m->_norm_out_lin.empty()) {
    return nullptr;
  }
  if (m->_cfg.out_channels == 0) {
    m->_cfg.out_channels =
        m->_proj_out.n > 0 ? m->_proj_out.n : m->_cfg.in_channels;
  }

  const int H = m->_cfg.hidden;
  m->_ln_w1 = mc->make_shared_buffer((std::size_t)H * 2);
  m->_ln_b0 = mc->make_shared_buffer((std::size_t)H * 2);
  if (m->_ln_w1.empty() || m->_ln_b0.empty()) { return nullptr; }
  {
    auto* w1 = static_cast<_Float16*>(m->_ln_w1.contents());
    auto* b0 = static_cast<_Float16*>(m->_ln_b0.contents());
    for (int i = 0; i < H; ++i) {
      w1[i] = (_Float16)1.0f; b0[i] = (_Float16)0.0f;
    }
  }

  if (!stream_blocks) {
    m->_double.resize((std::size_t)m->_cfg.n_double);
    for (int i = 0; i < m->_cfg.n_double; ++i) {
      if (!m->load_double_(ws, "transformer_blocks." + std::to_string(i) + ".",
                           m->_double[(std::size_t)i], r)) {
        if (mc->session() != nullptr) {
          mc->session()->warn(fmt(
              "MetalFlux2Transformer: failed to load double block {}", i));
        }
        return nullptr;
      }
    }
    m->_single.resize((std::size_t)m->_cfg.n_single);
    for (int i = 0; i < m->_cfg.n_single; ++i) {
      if (!m->load_single_(
              ws, "single_transformer_blocks." + std::to_string(i) + ".",
              m->_single[(std::size_t)i], r)) {
        if (mc->session() != nullptr) {
          mc->session()->warn(fmt(
              "MetalFlux2Transformer: failed to load single block {}", i));
        }
        return nullptr;
      }
    }
  } else {
    // Streaming: blocks load JIT in forward_dit. Still derive the FF dims that
    // load_double_/load_single_ would set -- from the tensor SHAPES (info only,
    // no weight load) so forward_dit has DFF / single_mlp_in.
    if (m->_cfg.double_ff_hidden == 0) {
      const auto* fi =
          ws.src().info("transformer_blocks.0.ff.linear_in.weight");
      if (fi != nullptr && !fi->shape.empty()) {
        m->_cfg.double_ff_hidden = (int)fi->shape[0];
      }
    }
    if (m->_cfg.single_mlp_in == 0) {
      const auto* qi =
          ws.src().info(
              "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight");
      if (qi != nullptr && !qi->shape.empty()) {
        const int rest = (int)qi->shape[0] - 3 * m->_cfg.hidden;
        if (rest > 0) { m->_cfg.single_mlp_in = rest / 2; }
      }
    }
    if (m->_cfg.double_ff_hidden == 0 || m->_cfg.single_mlp_in == 0) {
      if (mc->session() != nullptr) {
        mc->session()->warn(fmt(
            "MetalFlux2Transformer: streaming -- could not derive FF dims"));
      }
      return nullptr;
    }
    // Streaming preloads NOTHING. Both stacks are sized to FULL depth
    // all the same: the empty slots are where forward() promotes
    // streamed blocks as free memory allows (see set_residency_reserve).
    // An unfilled slot reads as empty, which is what `held` tests.
    m->_double.resize((std::size_t)m->_cfg.n_double);
    m->_single.resize((std::size_t)m->_cfg.n_single);
    // Retain the source mmap so forward_dit can re-read each block on demand.
    if (mc->session() != nullptr) {
      mc->session()->info(fmt(
          "MetalFlux2Transformer: streaming {}+{} blocks (memory-bounded)",
          m->_cfg.n_double, m->_cfg.n_single));
    }
    // Two reusable destinations per stack, refilled with pread and with
    // the next block's read issued under the current one's GPU work.
    // Only the streaming path ever asks for one.
    m->configure_slots_();
  }
  // The adapter LAST: the dims it binds against (double_ff_hidden,
  // single_mlp_in) are derived by the branches above, on both routes.
  if (lora != nullptr && !lora->path.empty()) {
    std::string lerr;
    if (!m->_lora.init(mc, m->_use_mma2)) {
      if (mc->session() != nullptr) {
        mc->session()->warn(fmt(
            "MetalFlux2Transformer: the LoRA GEMM kernels did not load; "
            "the adapter is NOT applied"));
      }
    } else if (!m->bind_lora_(*lora, &lerr)) {
      if (mc->session() != nullptr) { mc->session()->warn(fmt("{}", lerr)); }
      m->_lora_modules = 0;
    }
  }
  return m;
}

// Build the joint [text; image(+refs)] 4-axis RoPE cos/sin tables [seq,
// head_dim]. Text tokens sit at the origin on axis-3 (0,0,0,l); the generated
// image carries (0, row, col, 0); each reference image carries (T, row, col, 0)
// with T its per-reference index band. Adjacent-pair layout matches
// rope_pair_table_ftab_f16 (f32 cos/sin tables).
// NOTE: the axis->coordinate mapping is a best reading of Flux2PosEmbed; VERIFY
// against a diffusers golden.
void
MetalFlux2Transformer::build_rope_tables_(int text_seq,
                                          const std::vector<ImgSeg>& segs,
                                          SharedBuffer& cos_out,
                                          SharedBuffer& sin_out)
{
  int img_seq = 0;
  for (const auto& sg : segs) { img_seq += sg.seq; }
  const int seq = text_seq + img_seq;
  const int HD = _cfg.head_dim;
  const int pairs = HD / 2;
  // f32 cos/sin tables (only x is bf16): RoPE rotation error is STRUCTURED, so
  // the ~4e-3 bf16-table rounding compounds over blocks and, worse, over denoise
  // steps -- exactly what rope_pair_table_ftab_f16 was written to avoid (see
  // QwenImage). Keep full precision here.
  cos_out = _mc->make_shared_buffer((std::size_t)seq * HD * sizeof(float));
  sin_out = _mc->make_shared_buffer((std::size_t)seq * HD * sizeof(float));
  auto* c = static_cast<float*>(cos_out.contents());
  auto* s = static_cast<float*>(sin_out.contents());
  const float theta = _cfg.rope_theta;
  // Per-pair inverse frequency + which of the 4 position axes it reads. These
  // depend only on (axis, j) -- NOT the token -- so precompute once (pairs pow()
  // calls) instead of recomputing inside the per-token emit (seq*pairs). Angles
  // stay f64 to match diffusers' FLUX rope() (arange/theta**scale in float64);
  // the nonlinear cos/sin makes the angle precision matter more than the table
  // storage. Flux2 position ids (T,H,W,L): axes_dim [32,32,32,32] -> axis0=T,
  // axis1=H, axis2=W, axis3=L.
  std::vector<double> pair_freq((std::size_t)pairs);
  std::vector<int> pair_axis((std::size_t)pairs);
  {
    int pair = 0;
    for (int a = 0; a < 4; ++a) {
      const int adim = _cfg.axes_dim[a];
      const int apairs = adim / 2;
      for (int j = 0; j < apairs && pair < pairs; ++j, ++pair) {
        pair_freq[(std::size_t)pair] =
            1.0 / std::pow((double)theta, (double)(2 * j) / (double)adim);
        pair_axis[(std::size_t)pair] = a;
      }
    }
  }
  // Emit one token row from its 4 coordinates.
  auto emit = [&](int t, float p0, float p1, float p2, float p3) {
    const double pos[4] = {p0, p1, p2, p3};
    for (int pair = 0; pair < pairs; ++pair) {
      const double ang = pos[pair_axis[(std::size_t)pair]]
                         * pair_freq[(std::size_t)pair];
      const float cb = (float)std::cos(ang);
      const float sb = (float)std::sin(ang);
      c[(std::size_t)t * HD + 2 * pair]     = cb;
      c[(std::size_t)t * HD + 2 * pair + 1] = cb;
      s[(std::size_t)t * HD + 2 * pair]     = sb;
      s[(std::size_t)t * HD + 2 * pair + 1] = sb;
    }
  };
  int t = 0;
  for (; t < text_seq; ++t) { emit(t, 0, 0, 0, (float)t); }   // text: axis-3 = l
  for (const auto& sg : segs) {                               // image + refs
    for (int p = 0; p < sg.seq; ++p, ++t) {
      emit(t, (float)sg.t_off, (float)(p / sg.grid_w),
           (float)(p % sg.grid_w), 0.0f);
    }
  }
}

void
MetalFlux2Transformer::calib_begin()
{
  _calib_acc.clear();
  const int H = _cfg.hidden;
  const int INNER = _cfg.double_ff_hidden / 2;
  const int SMLP = _cfg.single_mlp_in;
  const int nD = _cfg.n_double, nS = _cfg.n_single;
  auto add = [&](const char* g, int rows, int dim) {
    SharedBuffer b = _mc->make_shared_buffer((std::size_t)rows * dim * 2);
    if (!b.empty()) { std::memset(b.contents(), 0, b.byte_size()); }
    _calib_acc[g] = std::move(b);
  };
  add("dbl_norm1_img", nD, H);   add("dbl_norm1_txt", nD, H);
  add("dbl_attn_img", nD, H);    add("dbl_attn_txt", nD, H);
  add("dbl_norm2_img", nD, H);   add("dbl_ffact_img", nD, INNER);
  add("dbl_norm2_txt", nD, H);   add("dbl_ffact_txt", nD, INNER);
  add("sgl_norm", nS, H);        add("sgl_cat", nS, H + SMLP);
  add("emb_x", 1, _cfg.in_channels);
  add("emb_ctx", 1, _cfg.joint_dim);
  add("emb_proj", 1, H);
  _calib_on = true;
}

std::map<std::string, std::vector<float>>
MetalFlux2Transformer::calib_stats() const
{
  std::map<std::string, std::vector<float>> out;
  for (const auto& kv : _calib_acc) {
    const std::size_t n = kv.second.empty() ? 0 : kv.second.byte_size() / 2;
    std::vector<float> v(n);
    // The calib accumulator (col_absmax) is bf16 now that the DiT runs bf16.
    const auto* s = static_cast<const std::uint16_t*>(kv.second.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = bf16_to_f32_(s[i]); }
    out[kv.first] = std::move(v);
  }
  return out;
}

bool
MetalFlux2Transformer::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                                 std::size_t xe, const QWeight& w,
                                 const SharedBuffer& y, std::size_t ye,
                                 int M, int N, int K)
{
  // Matrix-core matmul2d only when present, M amortizes the 128-row tile, and
  // N is non-degenerate (the M=1 conditioning GEMMs stay on steel).
  if (!_use_mma2 || M < _mma_min_m || N < 16) { return false; }
  // And decline a shape whose operands would not survive the tiles'
  // 32-bit addressing: past 2^31 BYTES on ANY operand the mma tiles stop
  // storing, silently. DECLINING rather than banding is deliberate here.
  // The steel path the caller falls back to is int64 and therefore
  // correct, and this shape is not reachable: K peaks at the ff-down's
  // 16384, so the line sits at 65536 tokens = a 4096px square, where one
  // token is 16x16 px. MiniMax-H3's fc2 got a band instead because it was
  // live at 1376x768x243 and a band is free; paying a restructure of this
  // GEMM for a resolution nothing can allocate is not the same trade.
  // See shared/mma-tile.h.
  if (M > mma_row_band(N, K)) { return false; }
  const SharedBuffer* wdense;
  if (w.quantized) {
    const metal_compute::ComputeFunction& dq =
        (w.bits == 8) ? _fn_dequant8 : _fn_dequant4;
    if (!dq.valid()) { return false; }
    const std::size_t need = (std::size_t)N * K * 2;
    if (_w_deq.empty() || _w_deq.byte_size() < need) {
      _w_deq = _mc->make_shared_buffer(need);
      if (_w_deq.empty()) { return false; }
    }
    // codes/scales/qbias -> _w_deq[N,K] (one thread per packed u32 word: w4
    // has 8 nibbles/word so K/8 words, w8 has 4 bytes/word so K/4). The
    // forward's streams commit serially (one live encoder), so Metal's WAR
    // hazard tracking makes the shared _w_deq safe to reuse across GEMMs
    // (each dequant->matmul pair runs before the next dequant overwrites).
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
  // Dynamic-int8 accelerated mode: quantize activations + the (dequanted)
  // f16 weight on the fly and run the int8 matmul (LOSSY; opt-in via
  // Config::i8_gemm / VPIPE_I8_GEMM). Falls through to the f16 tiles for
  // non-qualifying shapes (small M, K not a 512-multiple).
  if (_i8 && _i8->gemm(enc, x, xe, *wdense, y, ye, M, N, K)) {
    return true;
  }
  // Split-K deep reduction for the very deep K (the single-stream to_out,
  // K = H + SMLP = 16384 on the 9B): the single-op full reduction sits on the
  // deep-K cliff (~0.7x the K<=9728 rate; see Krea-2). Fire when K is >= 2
  // exact chunks of kSplitKC; each of the S = K/kSplitKC splits gets its own
  // threadgroup plane (grid.z), then residual_add folds the planes into y.
  // The extra f16 rounding per fold is fine for the rel-L2-verified DiT.
  // Falls through to the single-op path when the scratch alloc fails.
  const int splits = (_use_splitk && K >= 2 * kSplitKC && K % kSplitKC == 0)
                     ? K / kSplitKC : 0;
  if (splits >= 2) {
    const std::size_t plane = (std::size_t)M * N;
    const std::size_t need = plane * (std::size_t)splits * 2;
    if (_splitk.empty() || _splitk.byte_size() < need) {
      _splitk = _mc->make_shared_buffer(need);
    }
    if (!_splitk.empty()) {
      enc.set_function(_fn_dense_mma_splitk);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, *wdense);
      enc.set_buffer(2, _splitk);
      enc.set_constant(3, K); enc.set_constant(4, N); enc.set_constant(5, M);
      enc.dispatch({(unsigned)(((N + 255) / 256) * 256),   // BM=128, BN=256
                    (unsigned)((M + 127) / 128), (unsigned)splits},
                   {256, 1, 1});
      // Fold the S partial planes: y = plane0 + plane1 (+ plane_s ...).
      enc.set_function(_fn_residual);
      enc.set_buffer(0, _splitk, 0);
      enc.set_buffer(1, _splitk, plane * 2);
      enc.set_buffer(2, y, ye * 2);
      enc.set_constant(3, (int)plane);
      enc.dispatch({(unsigned)plane, 1, 1}, {256, 1, 1});
      for (int s = 2; s < splits; ++s) {
        enc.set_function(_fn_residual);
        enc.set_buffer(0, y, ye * 2);
        enc.set_buffer(1, _splitk, plane * (std::size_t)s * 2);
        enc.set_buffer(2, y, ye * 2);
        enc.set_constant(3, (int)plane);
        enc.dispatch({(unsigned)plane, 1, 1}, {256, 1, 1});
      }
      return true;
    }
  }
  // Tile-adaptive dense matmul2d (no bias slot; klein's block Linears are
  // bias-free and the conditioning biases are folded by the caller's bias()).
  // Krea-2-tuned routing: 128x128 for K < 6144 (the H=4096/5120 projections),
  // the TN=2 (128x512-region) tile for the mid-K band, plain 128x256 for
  // deeper unsplit K (the split-K path above normally owns that regime).
  int RN = 256;   // effective N-region per tg (TN*BN); grid divides N by it
  const metal_compute::ComputeFunction* fn = &_fn_dense_mma_deep;
  if (K < 6144) {
    fn = &_fn_dense_mma; RN = 128;
  } else if (K < 12288 && _fn_dense_mma_tn2.valid()) {
    fn = &_fn_dense_mma_tn2; RN = 512;   // TN=2: two 256-wide N-tiles per tg
  }
  enc.set_function(*fn);
  enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, *wdense);
  enc.set_buffer(2, *wdense);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  return true;
}

// ---- the wired pool ---------------------------------------------------
//
// One buffer at a time, STOPPING at the first refusal rather than
// unwinding: a partly wired block is partly protected, which is strictly
// better than none -- and giving protection back on the way out means
// competing for it again on the next block, against a pool that has just
// said no.
//
// The lists mirror double_bytes_/single_bytes_ exactly. They have to:
// wirable() gates admission on the byte count those return, so a buffer
// counted there and not wired here is a block the model believes is
// protected and is not.
std::size_t
MetalFlux2Transformer::wire_block_(DoubleBlock& b, bool on)
{
  std::size_t changed = 0;
  bool stop = false;
  auto one = [&](metal_compute::SharedBuffer& p) {
    if (stop) { return; }
    const std::size_t n = _wire.wire_one(_mc, p, on);
    if (on && n == 0 && p.byte_size() > 0 && !p.is_wired()) {
      stop = true;
      return;
    }
    changed += n;
  };
  auto qw = [&](QWeight& w) {
    one(w.w); one(w.codes); one(w.scales); one(w.qbias);
  };
  QWeight* q[] = {&b.q, &b.k, &b.v, &b.o, &b.aq, &b.ak, &b.av,
                  &b.ao, &b.ff_in, &b.ff_out, &b.cff_in, &b.cff_out};
  for (QWeight* w : q) { qw(*w); }
  one(b.qn); one(b.kn); one(b.aqn); one(b.akn);
  return changed;
}

std::size_t
MetalFlux2Transformer::wire_block_(SingleBlock& b, bool on)
{
  std::size_t changed = 0;
  bool stop = false;
  auto one = [&](metal_compute::SharedBuffer& p) {
    if (stop) { return; }
    const std::size_t n = _wire.wire_one(_mc, p, on);
    if (on && n == 0 && p.byte_size() > 0 && !p.is_wired()) {
      stop = true;
      return;
    }
    changed += n;
  };
  auto qw = [&](QWeight& w) {
    one(w.w); one(w.codes); one(w.scales); one(w.qbias);
  };
  QWeight* q[] = {&b.qkv_mlp, &b.qkv, &b.mlp_gu, &b.o};
  for (QWeight* w : q) { qw(*w); }
  one(b.qn); one(b.kn);
  return changed;
}

// The TRUNK: everything the weight set cached for this model, which for a
// streaming DiT is the non-block tensors it holds for the whole run. Read
// on every block of every forward and never shed, so it has a better
// claim on the pool than any single resident block does -- which is why
// the forward wires this BEFORE it starts admitting blocks.
//
// The activation scratch is deliberately not here: this model allocates
// it as locals inside forward_dit, so there is nothing that persists to
// wire. See the note in shared/wired-pool.h.
std::size_t
MetalFlux2Transformer::wire_fixed_(bool on)
{
  if (!_ws) { return 0; }
  std::size_t changed = 0;
  _ws->for_each_weight([&](metal_compute::SharedBuffer& b) {
    changed += _wire.wire_one(_mc, b, on);
  });
  return changed;
}

std::size_t
MetalFlux2Transformer::double_bytes_(const DoubleBlock& b)
{
  auto qb = [](const QWeight& w) {
    return w.w.byte_size() + w.codes.byte_size() + w.scales.byte_size()
         + w.qbias.byte_size();
  };
  const QWeight* q[] = {&b.q, &b.k, &b.v, &b.o, &b.aq, &b.ak, &b.av,
                        &b.ao, &b.ff_in, &b.ff_out, &b.cff_in, &b.cff_out};
  std::size_t n = b.qn.byte_size() + b.kn.byte_size() + b.aqn.byte_size()
                + b.akn.byte_size();
  for (const QWeight* w : q) { n += qb(*w); }
  return n;
}

std::size_t
MetalFlux2Transformer::single_bytes_(const SingleBlock& b)
{
  auto qb = [](const QWeight& w) {
    return w.w.byte_size() + w.codes.byte_size() + w.scales.byte_size()
         + w.qbias.byte_size();
  };
  const QWeight* q[] = {&b.qkv_mlp, &b.qkv, &b.mlp_gu, &b.o};
  std::size_t n = b.qn.byte_size() + b.kn.byte_size();
  for (const QWeight* w : q) { n += qb(*w); }
  return n;
}

std::size_t
MetalFlux2Transformer::single_read_bytes_(const SingleBlock& b)
{
  auto qb = [](const QWeight& w) {
    return w.w.byte_size() + w.codes.byte_size() + w.scales.byte_size()
         + w.qbias.byte_size();
  };
  std::size_t n = single_bytes_(b);
  if (!b.qkv_mlp.empty()) { n -= qb(b.qkv) + qb(b.mlp_gu); }
  return n;
}

std::size_t
MetalFlux2Transformer::single_resident_bytes_(const SingleBlock& b)
{
  auto qb = [](const QWeight& w) {
    return w.w.byte_size() + w.codes.byte_size() + w.scales.byte_size()
         + w.qbias.byte_size();
  };
  std::size_t n = single_bytes_(b);
  if (!b.qkv.empty()) { n -= qb(b.qkv_mlp); }
  return n;
}

// Singles before doubles.
//
// The two stacks are not interchangeable: a double block carries the
// joint text/image attention and is the more expensive half to re-read,
// so what survives an eviction should be a prefix of THOSE. Within a
// stack it is the tail, as everywhere else -- in a cyclic scan every
// block is worth the same, and a prefix keeps the bookkeeping trivial.
std::size_t
MetalFlux2Transformer::evict_tail_block_()
{
  for (int i = (int)_single.size() - 1; i >= 0; --i) {
    SingleBlock& b = _single[(std::size_t)i];
    const std::size_t n = single_bytes_(b);
    if (n == 0) { continue; }
    // Before the buffers go: give the wiring back. Dropping a wired
    // buffer unwires it in the kernel anyway, but only unwire_from_pool()
    // decrements the pool's counter -- so doing it here is what keeps the
    // budget honest instead of leaking a block's worth per eviction.
    _wire.note_unwired(wire_block_(b, false));
    b = SingleBlock{};
    return n;
  }
  for (int i = (int)_double.size() - 1; i >= 0; --i) {
    DoubleBlock& b = _double[(std::size_t)i];
    const std::size_t n = double_bytes_(b);
    if (n == 0) { continue; }
    _wire.note_unwired(wire_block_(b, false));
    b = DoubleBlock{};
    return n;
  }
  return 0;
}

// How much of the RESIDENT set is still in RAM. Sampled inside each
// buffer: a pin either holds or it does not, so every 64th page finds
// it. See BlockResidency property 5 -- free-memory arithmetic cannot
// answer this, and it is the question that actually matters.
void
MetalFlux2Transformer::resident_pages_(std::size_t* examined,
                                       std::size_t* incore) const
{
  *examined = 0;
  *incore = 0;
  auto add = [&](const metal_compute::SharedBuffer& p) {
    if (p.byte_size() == 0) { return; }
    // A WIRED BUFFER CANNOT HAVE LEFT RAM, so asking is spending the walk
    // to be told what mlock already guarantees. Skipped PER BUFFER rather
    // than per block, because wire_block_ stops at the first refusal and
    // leaves the rest of that block unwired -- the remainder is exactly
    // what still needs measuring. With everything wired `examined` stays
    // 0, which the caller reads as "no evidence" rather than as a
    // shortfall, and that is the correct answer: there is nothing this
    // walk could have found.
    if (p.is_wired()) { return; }
    const auto r = p.page_residency(64);
    if (!r.valid) { return; }
    *examined += r.examined;
    *incore += r.incore;
  };
  auto addq = [&](const QWeight& w) { add(w.w); add(w.codes); };
  for (const DoubleBlock& b : _double) {
    addq(b.q); addq(b.k); addq(b.v); addq(b.o);
    addq(b.aq); addq(b.ak); addq(b.av); addq(b.ao);
    addq(b.ff_in); addq(b.ff_out); addq(b.cff_in); addq(b.cff_out);
  }
  for (const SingleBlock& b : _single) {
    addq(b.qkv_mlp); addq(b.qkv); addq(b.mlp_gu); addq(b.o);
  }
}

// How much the box must have freed since a wiring refusal before it is
// worth asking again -- one block's worth, so a genuinely full box is
// never asked. Reopening the pool's ceiling makes its own check pass, and
// the mlock behind it would then fail and leave that block resident but
// UNWIRED, one per forward, which is exactly the state the wirable gate
// exists to avoid.
//
// The checkpoint figure when the schedule has been set, the resident
// average otherwise. Never zero if anything is known: a zero slack
// re-opens on any increase at all.
std::size_t
MetalFlux2Transformer::wire_retry_slack_() const
{
  if (_wire_block_hint > 0) { return _wire_block_hint; }
  if (_resid.count() > 0) {
    return _resid.bytes() / (std::size_t)_resid.count();
  }
  return 0;
}

void
MetalFlux2Transformer::set_residency_schedule(int steps)
{
  if (!_ws) { return; }
  const MetalLlamaWeights& src = _ws->src();
  const std::size_t blk = widest_block_bytes(
      src.tensor_names(),
      [&](const std::string& n) {
        const auto* ti = src.info(n);
        return ti != nullptr ? (std::size_t)ti->nbytes : (std::size_t)0;
      },
      {"transformer_blocks.", "single_transformer_blocks."});
  const int nl = _cfg.n_double + _cfg.n_single;
  _wire_block_hint = blk;
  _resid.set_schedule(steps, nl, blk, _wire.on(),
                      _mc != nullptr ? _mc->memory_budget()
                                     : metal_compute::MetalCompute::
                                           MemoryBudget{});
  if (_mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalFlux2Transformer: residency probe {} blocks of {} "
        "({} MB each, {} MB reclaimable, wire budget {} MB){}",
        _resid.per_forward_cap(), nl, blk >> 20,
        _mc->memory_budget().available_physical >> 20,
        _wire.budget() >> 20,
        _wire.on() ? " -- uncapped, the wire budget is the gate"
                   : ", doubling per healthy forward"));
  }
}

std::size_t
MetalFlux2Transformer::release_resident_blocks(std::size_t bytes)
{
  const std::size_t freed =
      _resid.release(bytes, [this]() -> std::size_t {
        return evict_tail_block_();
      });
  if (freed > 0 && _mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalFlux2Transformer: released {} MB of resident blocks "
        "({} left)", freed >> 20, _resid.count()));
  }
  return freed;
}

SharedBuffer
MetalFlux2Transformer::forward_dit(const SharedBuffer& context, int text_seq,
                                   const SharedBuffer& latents, int img_seq,
                                   int grid_h, int grid_w, float timestep,
                                   float guidance,
                                   const std::vector<RefImage>& refs,
                                   KvCache* kv)
{
  const Config& c = _cfg;
  const int H = c.hidden, HED = c.n_heads, HD = c.head_dim;
  const int IC = c.in_channels, TD = c.timestep_dim, OC = c.out_channels;
  const int DFF = c.double_ff_hidden, SMLP = c.single_mlp_in;
  const int INNER = DFF / 2;   // Flux2FeedForward linear_in -> 2*inner SwiGLU
  const float eps = c.norm_eps;
  // The image token stream is [generated; ref0; ref1; ...]: IS_GEN generated
  // tokens (the only ones we predict velocity for) followed by the reference
  // tokens. IS is the full image-token count that drives every block; IS_GEN
  // is what proj_out emits.
  const int TS = text_seq, IS_GEN = img_seq;
  int IS_REF_ALL = 0;
  for (const auto& r : refs) { IS_REF_ALL += r.seq; }
  // The klein-9b-kv recipe (Config::klein_kv): reference tokens are isolated
  // from the rest of the sequence and modulated off a fixed timestep 0. Only
  // then is their K/V step-independent, so the cache below is a consequence
  // of the recipe rather than an option on top of it.
  const bool kv_recipe = c.klein_kv && IS_REF_ALL > 0;
  const int  n_blocks  = c.n_double + c.n_single;
  // Reuse requires a populated cache built for THIS geometry -- a resolution
  // or reference change rebuilds it rather than silently mismatching.
  const bool kv_reuse = kv_recipe && kv != nullptr && kv->populated()
                        && kv->ref_seq == IS_REF_ALL && kv->text_seq == TS
                        && kv->img_seq == IS_GEN
                        && (int)kv->k.size() == n_blocks
                        && (int)kv->v.size() == n_blocks;
  const bool kv_fill  = kv_recipe && kv != nullptr && !kv_reuse;
  // Reference tokens ride the token stream unless a populated cache lets us
  // drop them: their contribution then enters attention purely as extra KEYS,
  // so every projection, norm and FF over them leaves the step.
  const int IS_REF = kv_reuse ? 0 : IS_REF_ALL;
  const int IS = IS_GEN + IS_REF, seq = TS + IS;
  // Queries that attend over the whole key set (text + generated). Under the
  // recipe the reference queries are the tail [QA, seq) and attend only to
  // themselves; without it QA == seq and there is one undivided attention.
  const int QA = (kv_recipe && !kv_reuse) ? (TS + IS_GEN) : seq;
  // Attention key length: the live sequence plus any cached reference keys
  // spliced onto its tail.
  const int KL = seq + (kv_reuse ? IS_REF_ALL : 0);
  // Flux2 reference-image T (time/index) coordinate step (_prepare_image_ids
  // scale): ref i -> T = kRefPosScale*(i+1) so each reference sits in its own
  // position band, distinct from the generated image (T=0).
  constexpr int kRefPosScale = 10;
  if (TS <= 0 || IS_GEN <= 0 || DFF <= 0 || SMLP <= 0
      || context.byte_size() < (std::size_t)TS * c.joint_dim * 2
      || latents.byte_size() < (std::size_t)IS_GEN * IC * 2) {
    return {};
  }
  for (const auto& r : refs) {
    if (r.seq <= 0 || r.latents.byte_size() < (std::size_t)r.seq * IC * 2) {
      return {};
    }
  }
  // The DiT computes in bf16 (f16's 65504 range overflows on real conditioning
  // outliers through the deep residual/attention stream). The stage/conditioner
  // boundary is f16, so upcast the inputs to bf16 once here (lossless: bf16's
  // exponent range covers all f16 values); the returned velocity is downcast
  // back to f16 at the end so callers/tests are unchanged.
  auto up_bf16 = [&](const SharedBuffer& src, std::size_t n) -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer(n * 2);
    if (o.empty()) { return o; }
    const auto* s = static_cast<const _Float16*>(src.contents());
    auto* d = static_cast<std::uint16_t*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_((float)s[i]); }
    return o;
  };
  const SharedBuffer context_b =
      up_bf16(context, (std::size_t)TS * c.joint_dim);
  const SharedBuffer latents_b = up_bf16(latents, (std::size_t)IS_GEN * IC);
  std::vector<SharedBuffer> refs_b;
  refs_b.reserve(refs.size());
  for (const auto& r : refs) {
    refs_b.push_back(up_bf16(r.latents, (std::size_t)r.seq * IC));
  }
  if (context_b.empty() || latents_b.empty()) { return {}; }
  const int PW = 3 * H + 2 * SMLP;                 // to_qkv_mlp_proj out width
  const float scale = 1.0f / std::sqrt((float)HD);

  // LLM-lane perf event (perf-visualizer): one DiT forward per sampler step,
  // value = the joint sequence length. Mirrors the Krea-2 DiT event.
  PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDit,
                     kPerfLlmDitBegin, (std::uint64_t)seq);

  // Per-section GPU timing (VPIPE_FLUX2_DIT_PROFILE). Marks + commit-boundary
  // waits split the deferred streams into timed slices; single-block attention
  // is isolated from its GEMMs by an extra barrier. The barriers serialize
  // (removing any overlap) but the per-section GPU wall time is what we want.
  // No effect unless the env var is set.
  const bool prof = std::getenv("VPIPE_FLUX2_DIT_PROFILE") != nullptr;
  // ---- env-gated streaming split (VPIPE_FLUX2_STREAM_PROFILE) ----------
  // What a weight PREFETCH could hide. The streamed path loads a block
  // serially and then commits-and-waits before the weights free, so the
  // disk and the GPU take strict turns; both ends are already on the
  // critical path, so timing them adds no barriers of its own. The
  // ceiling on any prefetch is read/(read+gpu) -- measured rather than
  // assumed, because every term moves with the machine: block bytes with
  // the quantization, read rate with the storage medium, gpu with the
  // geometry. Covers the double AND single stacks, which stream
  // separately but compete for the same disk.
  const bool sprof = std::getenv("VPIPE_FLUX2_STREAM_PROFILE") != nullptr;
  double sp_read_ms = 0, sp_gpu_ms = 0;
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
  double t_dbl = 0, t_sgl_gemm = 0, t_sgl_attn = 0, t_sgl_cat = 0,
         t_sgl_out = 0, t_final = 0;
  auto tnow = [] { return std::chrono::steady_clock::now(); };
  auto ms_since = [](std::chrono::steady_clock::time_point m) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - m).count();
  };
  std::chrono::steady_clock::time_point mk;

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer rcos, rsin;
  {
    // Segment 0 = generated (T=0); one segment per reference at its own T band.
    std::vector<ImgSeg> segs;
    segs.push_back({0, grid_h, grid_w, IS_GEN});
    for (int i = 0; i < (int)refs.size(); ++i) {
      segs.push_back({kRefPosScale * (i + 1), refs[i].grid_h, refs[i].grid_w,
                      refs[i].seq});
    }
    build_rope_tables_(TS, segs, rcos, rsin);
  }

  SharedBuffer te_in = buf((std::size_t)TD);
  {
    auto* ti = static_cast<std::uint16_t*>(te_in.contents());
    const int half = TD / 2;
    for (int i = 0; i < half; ++i) {
      const double fr = std::exp(-std::log(1e4) * (double)i / (double)half);
      const double ang = (double)timestep * 1000.0 * fr;
      ti[i] = f32_to_bf16_((float)std::cos(ang));
      ti[half + i] = f32_to_bf16_((float)std::sin(ang));
    }
  }
  // Reference tokens are modulated off a FIXED timestep 0 (BFL
  // ref_fixed_timestep), not the step's sigma -- a reference latent is clean,
  // never noised. At t = 0 every angle is 0, so the sinusoid degenerates to
  // cos = 1 / sin = 0 and needs no trig.
  SharedBuffer te_ref, tr1, temb_r, tsilu_r;
  if (kv_recipe) {
    te_ref = buf((std::size_t)TD);
    tr1 = buf((std::size_t)H);
    temb_r = buf((std::size_t)H);
    tsilu_r = buf((std::size_t)H);
    if (te_ref.empty() || tr1.empty() || temb_r.empty() || tsilu_r.empty()) {
      return {};
    }
    auto* ri = static_cast<std::uint16_t*>(te_ref.contents());
    const int half = TD / 2;
    for (int i = 0; i < half; ++i) {
      ri[i] = f32_to_bf16_(1.0f);
      ri[half + i] = f32_to_bf16_(0.0f);
    }
  }
  // Embedded guidance (guidance-distilled klein-9B): the same cos-first
  // sinusoid of guidance*1000, embedded by guidance_embedder and added to the
  // timestep embedding. Skipped when the model has no guidance_embeds or the
  // caller passes guidance < 0.
  const bool use_g = c.guidance_embeds && guidance >= 0.0f && !_g_emb1.empty();
  SharedBuffer ge_in;
  if (use_g) {
    ge_in = buf((std::size_t)TD);
    auto* gi = static_cast<std::uint16_t*>(ge_in.contents());
    const int half = TD / 2;
    for (int i = 0; i < half; ++i) {
      const double fr = std::exp(-std::log(1e4) * (double)i / (double)half);
      const double ang = (double)guidance * 1000.0 * fr;
      gi[i] = f32_to_bf16_((float)std::cos(ang));
      gi[half + i] = f32_to_bf16_((float)std::sin(ang));
    }
  }

  // Scratch.
  SharedBuffer temb = buf((std::size_t)H), tsilu = buf((std::size_t)H),
               te1 = buf((std::size_t)H);
  SharedBuffer ge1 = buf((std::size_t)H), gemb = buf((std::size_t)H);
  SharedBuffer mimg = buf((std::size_t)6 * H), mtxt = buf((std::size_t)6 * H),
               msin = buf((std::size_t)3 * H);
  // Reference-token modulation twins (klein_kv): the same two Linears driven
  // by a timestep-0 embedding instead of the step's sigma.
  SharedBuffer mimg_r, msin_r;
  if (kv_recipe) {
    mimg_r = buf((std::size_t)6 * H);
    msin_r = buf((std::size_t)3 * H);
  }
  SharedBuffer img = buf((std::size_t)IS * H), txt = buf((std::size_t)TS * H);
  SharedBuffer nrm = buf((std::size_t)seq * H);
  // k/v carry KL rows -- the sequence plus any cached reference keys spliced
  // onto the tail; q/out only ever cover the live sequence.
  SharedBuffer jq = buf((std::size_t)seq * H), jk = buf((std::size_t)KL * H),
               jv = buf((std::size_t)KL * H);
  SharedBuffer qt = buf((std::size_t)HED * seq * HD),
               kt = buf((std::size_t)HED * KL * HD),
               vt = buf((std::size_t)HED * KL * HD),
               atb = buf((std::size_t)HED * seq * HD);
  SharedBuffer att = buf((std::size_t)seq * H), ob = buf((std::size_t)seq * H);
  SharedBuffer ff1 = buf((std::size_t)seq * DFF);
  SharedBuffer joint = buf((std::size_t)seq * H);
  SharedBuffer sproj = buf((std::size_t)seq * PW);
  SharedBuffer sg = buf((std::size_t)seq * SMLP),
               su = buf((std::size_t)seq * SMLP),
               smlp = buf((std::size_t)seq * SMLP);
  SharedBuffer scat = buf((std::size_t)seq * ((std::size_t)H + SMLP));
  SharedBuffer velocity = buf((std::size_t)IS_GEN * OC);

  // KV cache storage: per block the reference band of k and v, token-major
  // [IS_REF_ALL, H] and PRE-RoPE. Pre-RoPE is what lets a cached step re-run
  // the ordinary fused transpose+rope over the whole KL rows: the rope table
  // is still built for [text, generated, refs], so the spliced band picks up
  // exactly the rotation it had when it was extracted.
  if (kv_fill) {
    kv->clear();
    kv->k.resize((std::size_t)n_blocks);
    kv->v.resize((std::size_t)n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
      kv->k[(std::size_t)i] = buf((std::size_t)IS_REF_ALL * H);
      kv->v[(std::size_t)i] = buf((std::size_t)IS_REF_ALL * H);
      if (kv->k[(std::size_t)i].empty() || kv->v[(std::size_t)i].empty()) {
        kv->clear();   // stays unpopulated -> next step just refills
        return {};
      }
    }
    if (_mc->session() != nullptr) {
      _mc->session()->info(fmt(
          "MetalFlux2Transformer: KV-caching {} reference tokens over {} "
          "blocks ({} MB)", IS_REF_ALL, n_blocks,
          (long)((std::size_t)n_blocks * 2 * IS_REF_ALL * H * 2 / (1 << 20))));
    }
  }

  // ---- helper wrappers over a live ComputeEncoder (redefined per stream) ----
  auto make_ops = [&](ComputeEncoder& enc) {
    struct Ops {
      MetalFlux2Transformer* self;
      ComputeEncoder* e;
      SharedBuffer *rcos, *rsin;
      float eps;
      // y[M,N] (elem offset ye) = x[M,K] (elem offset xe) @ W[N,K]^T.
      //
      // `lf`, when given, is the runtime adapter for THIS projection:
      // the base GEMM runs unchanged and the pair t = x A^T,
      // y += s t B^T is encoded after it. Threaded through the lambda
      // rather than called at each site so the adapter cannot be
      // forgotten on one projection and applied on the other eleven.
      void gemm(const SharedBuffer& x, const QWeight& w, const SharedBuffer& y,
                std::size_t ye, int M, int N, int K, std::size_t xe = 0,
                const lora::Factors* lf = nullptr) {
        auto lora_after = [&]() {
          if (lf != nullptr) {
            self->_lora.apply(*e, x, xe, *lf, y, ye, M, N, K,
                              self->_lora_scale, self->_mma_min_m);
          }
        };
        // Matrix-core matmul2d first (M5); false -> steel below.
        if (self->gemm_mma_(*e, x, xe, w, y, ye, M, N, K)) {
          lora_after();
          return;
        }
        int bm = 32, bn = 32;                    // 32x32 base tile
        if (w.quantized) {
          // BM128 tile once M amortizes the 128-row re-use (the DiT block GEMMs
          // at high res, M = seq >= 1024); small-M keeps the base tile.
          const bool huge = self->_qmm_tile >= 1 && M >= 1024
                            && self->_fn_qmm4_bm128.valid();
          if (huge) { bm = 128; }
          e->set_function(w.bits == 8
              ? (huge ? self->_fn_qmm8_bm128 : self->_fn_qmm8)
              : (huge ? self->_fn_qmm4_bm128 : self->_fn_qmm4));
          e->set_buffer(0, w.codes); e->set_buffer(1, w.scales);
          e->set_buffer(2, w.qbias); e->set_buffer(3, x, xe * 2);
          e->set_buffer(4, y, ye * 2);
          e->set_constant(5, K); e->set_constant(6, N); e->set_constant(7, M);
        } else {
          // Larger tiles for the big M = seq GEMMs (fewer weight re-reads).
          const metal_compute::ComputeFunction* f = &self->_fn_gemm;
          if (M >= 128 && self->_gemm_tile == 1) {
            f = (self->_acc16 && self->_fn_gemm_bm64_a16.valid())
                    ? &self->_fn_gemm_bm64_a16 : &self->_fn_gemm_bm64;
            bm = 64;
          } else if (M >= 128 && self->_gemm_tile == 2) {
            f = &self->_fn_gemm_bm64bn64; bm = 64; bn = 64;
          }
          e->set_function(*f);
          e->set_buffer(0, x, xe * 2); e->set_buffer(1, w.w);
          e->set_buffer(2, w.w);
          e->set_buffer(3, y, ye * 2);
          e->set_constant(4, K); e->set_constant(5, N); e->set_constant(6, M);
          e->set_constant(7, 0);
        }
        // BM=128 uses WM=4 (256 threads): threadgroup {32,2,4}, grid z=4.
        const unsigned tgz = (bm == 128) ? 4u : 2u;
        e->dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                     (unsigned)(((M + bm - 1) / bm) * 2), tgz}, {32, 2, tgz});
        lora_after();
      }
      // Fused SwiGLU FF: out[M, Nf/2] = silu(gate)*up from the INTERLEAVED
      // [Nf, K] linear_in weight (Nf = 2*INNER). One BM64 GEMM whose register-
      // local epilogue writes silu(gate)*up -- no [M, Nf] intermediate + no
      // slice/swiglu passes. Dense or quant (w4/w8), float or acc16.
      // out_stride/out_off (dense only): write silu(gate)*up into out[:, off:]
      // of a wider buffer (row stride out_stride) so the single block can drop
      // the [att|mlp] concat. The quant kernels are shared (no strided output),
      // so the quant path must pass out_stride 0 (contiguous) + concat.
      void swiglu_ff(const SharedBuffer& x, const QWeight& w,
                     const SharedBuffer& out, int M, int K, int Nf,
                     int out_stride = 0, int out_off = 0) {
        const bool a16 = self->_ff_acc16;
        const bool strided = out_stride > 0;
        if (w.quantized) {
          e->set_function(strided
              ? (w.bits == 8
                     ? (a16 ? self->_fn_qmm_swiglu8_bm64_rs_a16
                            : self->_fn_qmm_swiglu8_bm64_rs)
                     : (a16 ? self->_fn_qmm_swiglu4_bm64_rs_a16
                            : self->_fn_qmm_swiglu4_bm64_rs))
              : (w.bits == 8
                     ? (a16 ? self->_fn_qmm_swiglu8_bm64_a16
                            : self->_fn_qmm_swiglu8_bm64)
                     : (a16 ? self->_fn_qmm_swiglu4_bm64_a16
                            : self->_fn_qmm_swiglu4_bm64)));
          e->set_buffer(0, w.codes); e->set_buffer(1, w.scales);
          e->set_buffer(2, w.qbias); e->set_buffer(3, x); e->set_buffer(4, out);
          e->set_constant(5, K); e->set_constant(6, Nf); e->set_constant(7, M);
          if (strided) {
            e->set_constant(8, out_stride); e->set_constant(9, out_off);
          }
        } else {
          e->set_function(a16 ? self->_fn_ff_swiglu_a16 : self->_fn_ff_swiglu);
          e->set_buffer(0, x); e->set_buffer(1, w.w); e->set_buffer(2, out);
          e->set_constant(3, K); e->set_constant(4, Nf); e->set_constant(5, M);
          e->set_constant(6, out_stride); e->set_constant(7, out_off);
        }
        e->dispatch({(unsigned)(((Nf + 31) / 32) * 32),
                     (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
      }
      // Strided unfused SwiGLU: silu(gate)*up (gate/up contiguous [rows,width])
      // -> out[row*out_stride + out_off + col].
      void swiglu_rs(const SharedBuffer& gate, const SharedBuffer& up,
                     const SharedBuffer& out, int rows, int width,
                     int out_stride, int out_off) {
        e->set_function(self->_fn_swiglu_rs);
        e->set_buffer(0, gate); e->set_buffer(1, up); e->set_buffer(2, out);
        e->set_constant(3, rows); e->set_constant(4, width);
        e->set_constant(5, out_stride); e->set_constant(6, out_off);
        e->dispatch({(unsigned)(rows * width), 1, 1}, {256, 1, 1});
      }
      // AWQ calib tap: acc[group] row L (dim) max-accumulates the per-column
      // |activation| over M rows of `in` (elem offset xe). No-op when calib off.
      void tap(const char* group, int L, const SharedBuffer& in, std::size_t xe,
               int M, int dim) {
        if (!self->_calib_on) { return; }
        auto it = self->_calib_acc.find(group);
        if (it == self->_calib_acc.end() || it->second.empty()) { return; }
        e->set_function(self->_fn_colabsmax);
        e->set_buffer(0, in, xe * 2);
        e->set_buffer(1, it->second, (std::size_t)L * dim * 2);
        e->set_constant(2, M); e->set_constant(3, dim);
        e->dispatch({(unsigned)dim, 1, 1}, {256, 1, 1});
      }
      void bias(const SharedBuffer& bs, const SharedBuffer& y, int M, int N) {
        if (bs.empty()) { return; }   // klein is bias-free (bias=False)
        e->set_function(self->_fn_bias_add);
        e->set_buffer(0, y); e->set_buffer(1, bs);
        e->set_constant(2, N); e->set_constant(3, M * N);
        e->dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
      }
      void rms(const SharedBuffer& x, std::size_t xe, const SharedBuffer& w,
               const SharedBuffer& y, std::size_t ye, int R, int Hd) {
        e->set_function(self->_fn_rms);
        e->set_buffer(0, x, xe * 2); e->set_buffer(1, w);
        e->set_buffer(2, y, ye * 2);
        e->set_constant(3, Hd); e->set_constant(4, eps);
        e->dispatch({256, (unsigned)R, 1}, {256, 1, 1});
      }
      void ln(const SharedBuffer& x, std::size_t xe, const SharedBuffer& y,
              std::size_t ye, int R, int Hd) {
        // layer_norm_plain_f16 (no affine): 0:x 1:out 2:H 3:eps.
        e->set_function(self->_fn_layernorm);
        e->set_buffer(0, x, xe * 2); e->set_buffer(1, y, ye * 2);
        e->set_constant(2, Hd); e->set_constant(3, eps);
        e->dispatch({256, (unsigned)R, 1}, {256, 1, 1});
      }
      // Fused LayerNorm(no-affine) + adaLN modulate: out = (1+scale)*LN(x)+shift.
      // scale = mod[sc_e:], shift = mod[sh_e:] ([Hd] each, broadcast over R
      // rows). Falls back to ln + adaln if the fused kernel is unavailable.
      void ln_mod(const SharedBuffer& x, std::size_t xe, const SharedBuffer& mod,
                  std::size_t sc_e, std::size_t sh_e, const SharedBuffer& out,
                  std::size_t oe, int Hd, int R) {
        if (!self->_fn_ln_mod.valid()) {
          ln(x, xe, out, oe, R, Hd);
          adaln(out, oe, mod, sc_e, sh_e, out, oe, Hd, R * Hd);
          return;
        }
        e->set_function(self->_fn_ln_mod);
        e->set_buffer(0, x, xe * 2); e->set_buffer(1, mod, sc_e * 2);
        e->set_buffer(2, mod, sh_e * 2); e->set_buffer(3, out, oe * 2);
        e->set_constant(4, Hd); e->set_constant(5, eps);
        e->dispatch({256, (unsigned)R, 1}, {256, 1, 1});
      }
      void tr(const SharedBuffer& in, std::size_t ie, const SharedBuffer& out,
              std::size_t oe, int A, int Bd, int D) {
        e->set_function(self->_fn_transpose);
        e->set_buffer(0, in, ie * 2); e->set_buffer(1, out, oe * 2);
        e->set_constant(2, A); e->set_constant(3, Bd); e->set_constant(4, D);
        e->dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A},
                    {(unsigned)D, 1, 1});
      }
      // transpose with an output ROW-STRIDE: out[b*out_rs + a*D + d] -- writes
      // [Bd, A*D] as columns [0:A*D] of a wider buffer (e.g. att -> scat[:, :H]).
      void tr_rs(const SharedBuffer& in, const SharedBuffer& out, int A, int Bd,
                 int D, int out_rs) {
        e->set_function(self->_fn_transpose_rs);
        e->set_buffer(0, in); e->set_buffer(1, out);
        e->set_constant(2, A); e->set_constant(3, Bd); e->set_constant(4, D);
        e->set_constant(5, out_rs);
        e->dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A},
                    {(unsigned)D, 1, 1});
      }
      void rope(const SharedBuffer& x, int nh, int T, int D) {
        e->set_function(self->_fn_rope_table);
        e->set_buffer(0, x); e->set_buffer(1, *rcos); e->set_buffer(2, *rsin);
        e->set_constant(3, nh); e->set_constant(4, T); e->set_constant(5, D);
        e->dispatch({(unsigned)(D / 2), (unsigned)T, (unsigned)nh},
                    {(unsigned)(D / 2), 1, 1});
      }
      // Fused transpose [T,nh,D] (token-major) -> [nh,T,D] (head-major) + pair
      // RoPE (f32 tables), for the q/k path -- one pass replaces tr + rope (saves
      // the roped buffer's extra read/write). Falls back to tr + rope.
      void tr_rope(const SharedBuffer& in, const SharedBuffer& out, int T,
                   int nh, int D) {
        if (!self->_fn_transpose_rope.valid()) {
          tr(in, 0, out, 0, T, nh, D);
          rope(out, nh, T, D);
          return;
        }
        e->set_function(self->_fn_transpose_rope);
        e->set_buffer(0, in); e->set_buffer(1, out);
        e->set_buffer(2, *rcos); e->set_buffer(3, *rsin);
        e->set_constant(4, nh); e->set_constant(5, T); e->set_constant(6, D);
        e->dispatch({(unsigned)(D / 2), (unsigned)T, (unsigned)nh},
                    {(unsigned)(D / 2), 1, 1});
      }
      void sdpa(const SharedBuffer& q, const SharedBuffer& k,
                const SharedBuffer& v, const SharedBuffer& out, float sc, int T,
                int D, int nh) {
        e->set_function(self->_fn_sdpa);
        e->set_buffer(0, q); e->set_buffer(1, k); e->set_buffer(2, v);
        e->set_buffer(3, out);
        e->set_constant(4, sc); e->set_constant(5, T); e->set_constant(6, D);
        e->set_constant(7, nh); e->set_constant(8, nh); e->set_constant(9, T);
        e->set_constant(10, T);
        e->dispatch({32, (unsigned)nh, (unsigned)T}, {32, 1, 1});
      }
      // out[r, 0:W] = in[r*S + off : +W]  (strided column slice; block=0).
      void slice(const SharedBuffer& in, const SharedBuffer& out, int rows,
                 int S, int W, int off) {
        e->set_function(self->_fn_headslice);
        e->set_buffer(0, in); e->set_buffer(1, out);
        e->set_constant(2, rows); e->set_constant(3, S); e->set_constant(4, W);
        e->set_constant(5, off); e->set_constant(6, 0); e->set_constant(7, 0);
        e->dispatch({(unsigned)(rows * W), 1, 1}, {256, 1, 1});
      }
      // Contiguous [rows, W] copy between two element offsets -- a slice whose
      // input stride equals its width, bound at both ends. Used to lift the
      // reference band of k/v into the KV cache and to splice it back.
      void copy_rows(const SharedBuffer& in, std::size_t ie,
                     const SharedBuffer& out, std::size_t oe, int rows,
                     int W) {
        e->set_function(self->_fn_headslice);
        e->set_buffer(0, in, ie * 2); e->set_buffer(1, out, oe * 2);
        e->set_constant(2, rows); e->set_constant(3, W); e->set_constant(4, W);
        e->set_constant(5, 0); e->set_constant(6, 0); e->set_constant(7, 0);
        e->dispatch({(unsigned)rows * (unsigned)W, 1, 1}, {256, 1, 1});
      }
      // GPU [a | b] row-wise column concat -> dst[rows, wa+wb] (in-stream).
      void concat_cols(const SharedBuffer& dst, const SharedBuffer& a,
                       const SharedBuffer& b, int rows, int wa, int wb) {
        e->set_function(self->_fn_concat);
        e->set_buffer(0, a); e->set_buffer(1, b); e->set_buffer(2, dst);
        e->set_constant(3, rows); e->set_constant(4, wa);
        e->set_constant(5, wb);
        e->dispatch({(unsigned)(rows * (wa + wb)), 1, 1}, {256, 1, 1});
      }
      void elt(const ComputeFunction& fn, const SharedBuffer& a, std::size_t ae,
               const SharedBuffer& b, std::size_t be, const SharedBuffer& out,
               std::size_t oe, int nn) {
        e->set_function(fn);
        e->set_buffer(0, a, ae * 2); e->set_buffer(1, b, be * 2);
        e->set_buffer(2, out, oe * 2); e->set_constant(3, nn);
        e->dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
      }
      // vec4 eligibility: the twins load/store 8 bytes at a time off each bound
      // base, so EVERY element offset must be 4-aligned, not just the row width
      // -- a vec4 read off an odd offset is undefined, not merely slow. `total`
      // must also be a whole number of rows for the 2-D grid to cover it.
      bool elt4_ok(int N, int total, std::initializer_list<std::size_t> offs) {
        if (!self->_fn_adaln4.valid() || (N % 4) != 0 || N <= 0 ||
            (total % N) != 0) {
          return false;
        }
        for (std::size_t o : offs) { if ((o % 4) != 0) { return false; } }
        return true;
      }
      void adaln(const SharedBuffer& x, std::size_t xe, const SharedBuffer& mod,
                 std::size_t sc_e, std::size_t sh_e, const SharedBuffer& out,
                 std::size_t oe, int N, int total) {
        if (elt4_ok(N, total, {xe, sc_e, sh_e, oe})) {
          e->set_function(self->_fn_adaln4);
          e->set_buffer(0, x, xe * 2); e->set_buffer(1, mod, sc_e * 2);
          e->set_buffer(2, mod, sh_e * 2); e->set_buffer(3, out, oe * 2);
          e->set_constant(4, N / 4); e->set_constant(5, total / N);
          e->dispatch({(unsigned)(N / 4), (unsigned)(total / N), 1},
                      {256, 1, 1});
          return;
        }
        e->set_function(self->_fn_adaln);
        e->set_buffer(0, x, xe * 2); e->set_buffer(1, mod, sc_e * 2);
        e->set_buffer(2, mod, sh_e * 2); e->set_buffer(3, out, oe * 2);
        e->set_constant(4, N); e->set_constant(5, total);
        e->dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
      }
      void gated(const SharedBuffer& h, std::size_t he, const SharedBuffer& mod,
                 std::size_t g_e, const SharedBuffer& sub, std::size_t se,
                 int N, int total) {
        if (self->_fn_gated4.valid() && elt4_ok(N, total, {he, g_e, se})) {
          e->set_function(self->_fn_gated4);
          e->set_buffer(0, h, he * 2); e->set_buffer(1, mod, g_e * 2);
          e->set_buffer(2, sub, se * 2);
          e->set_constant(3, N / 4); e->set_constant(4, total / N);
          e->dispatch({(unsigned)(N / 4), (unsigned)(total / N), 1},
                      {256, 1, 1});
          return;
        }
        e->set_function(self->_fn_gated);
        e->set_buffer(0, h, he * 2); e->set_buffer(1, mod, g_e * 2);
        e->set_buffer(2, sub, se * 2);
        e->set_constant(3, N); e->set_constant(4, total);
        e->dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
      }
    };
    return Ops{this, &enc, &rcos, &rsin, eps};
  };

  // Steel flash-attention setup (built ONCE; seq is constant across all blocks
  // of a forward). FLUX.2 attention is MHA -- 32 q/k/v heads, head_dim 128, no
  // GQA (gqa_factor 1) -- so Q/K/V/O are all [HED, seq, HD] (the head-major
  // transposes below produce exactly that). Register-resident flash attention
  // is O(seq) memory + tiled, vs the scalar sdpa_full_f16's O(seq^2) inner loop
  // (the ~83% bottleneck at 1024px). Falls back to scalar sdpa when the steel
  // library / function is unavailable. On matrix-core GPUs (M5) prefer the NAX
  // variant. Params written once on CPU, read by every block's GPU dispatch.
  const int KVH = HED;                          // MHA: kv heads == q heads
  const bool nax = _use_attn_nax && _lib_attn_nax.valid();
  const int A_BQ = nax ? 64 : 32;
  const int A_BK = nax ? 32 : 16;
  // Q/O span the live sequence; K/V span KL (sequence + spliced cached refs).
  auto fill_params = [&](const SharedBuffer& pb, int qL, int kL) {
    auto* p = static_cast<SteelAttnParams*>(pb.contents());
    p->B = 1; p->H = HED; p->D = HD; p->qL = qL; p->kL = kL;
    p->gqa_factor = HED / KVH; p->scale = scale;
    p->NQ = (qL + A_BQ - 1) / A_BQ; p->NK = (kL + A_BK - 1) / A_BK;
    p->NQ_aligned = qL / A_BQ; p->NK_aligned = kL / A_BK;
    p->qL_rem = qL - p->NQ_aligned * A_BQ;
    p->kL_rem = kL - p->NK_aligned * A_BK;
    p->qL_off = 0;
    p->Q_strides[0] = (std::int64_t)HED * seq * HD;
    p->Q_strides[1] = (std::int64_t)seq * HD; p->Q_strides[2] = HD;
    p->K_strides[0] = (std::int64_t)KVH * KL * HD;
    p->K_strides[1] = (std::int64_t)KL * HD; p->K_strides[2] = HD;
    p->V_strides[0] = p->K_strides[0];
    p->V_strides[1] = p->K_strides[1]; p->V_strides[2] = HD;
    p->O_strides[0] = p->Q_strides[0];
    p->O_strides[1] = p->Q_strides[1]; p->O_strides[2] = HD;
  };
  auto attn_fn = [&](int qL, int kL) {
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (qL % A_BQ) == 0).set_bool(201, (kL % A_BK) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false);
    return nax ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
               : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
  };
  // Reference queries present => two groups. Each needs its OWN params buffer:
  // both dispatches are encoded into one stream, so a single mutable buffer
  // would hand the second group's shape to the first at execution time.
  const bool split_attn = kv_recipe && QA < seq;
  metal_compute::ComputeFunction fn_attn, fn_attn_ref;
  SharedBuffer attn_params_ref;
  bool use_steel = _steel_attn_ok && !_attn_params.empty();
  if (use_steel) {
    fill_params(_attn_params, QA, KL);
    fn_attn = attn_fn(QA, KL);
    use_steel = fn_attn.valid();
  }
  if (use_steel && split_attn) {
    attn_params_ref = _mc->make_shared_buffer(sizeof(SteelAttnParams));
    if (attn_params_ref.empty()) { return {}; }
    fill_params(attn_params_ref, IS_REF, IS_REF);
    fn_attn_ref = attn_fn(IS_REF, IS_REF);
    use_steel = fn_attn_ref.valid();
  }
  // The recipe's masking IS the pair of grouped dispatches; the scalar
  // fallback has no way to express it, and quietly running one undivided
  // attention would yield plausible-looking WRONG images. Fail loudly.
  if (kv_recipe && !use_steel) {
    if (_mc->session() != nullptr) {
      _mc->session()->warn(fmt(
          "MetalFlux2Transformer: klein_kv needs the steel flash-attention "
          "library; refusing to run reference-isolated attention on the "
          "scalar fallback"));
    }
    return {};
  }
  const unsigned a_nqb = (unsigned)((QA + A_BQ - 1) / A_BQ);
  const unsigned a_nqb_ref =
      split_attn ? (unsigned)((IS_REF + A_BQ - 1) / A_BQ) : 0u;

  // Per-token modulation. Under the recipe the reference tail carries its own
  // timestep-0 modulation, so each op becomes two dispatches over contiguous
  // row spans. Keeping the references at the END of both the image stream and
  // the joint sequence is what holds this to a plain offset instead of a
  // masked kernel -- BFL's [text, refs, generated] order would split the
  // generated tokens in two.
  const bool mod_split = kv_recipe && IS_REF > 0;
  auto ln_mod_img = [&](auto& op, const SharedBuffer& x, std::size_t sc,
                        std::size_t sh, const SharedBuffer& out) {
    if (!mod_split) { op.ln_mod(x, 0, mimg, sc, sh, out, 0, H, IS); return; }
    op.ln_mod(x, 0, mimg, sc, sh, out, 0, H, IS_GEN);
    op.ln_mod(x, (std::size_t)IS_GEN * H, mimg_r, sc, sh, out,
              (std::size_t)IS_GEN * H, H, IS_REF);
  };
  auto gated_img = [&](auto& op, const SharedBuffer& x, std::size_t g,
                       const SharedBuffer& sub) {
    if (!mod_split) { op.gated(x, 0, mimg, g, sub, 0, H, IS * H); return; }
    op.gated(x, 0, mimg, g, sub, 0, H, IS_GEN * H);
    op.gated(x, (std::size_t)IS_GEN * H, mimg_r, g, sub,
             (std::size_t)IS_GEN * H, H, IS_REF * H);
  };
  auto ln_mod_joint = [&](auto& op, const SharedBuffer& x, std::size_t sc,
                          std::size_t sh, const SharedBuffer& out) {
    if (!mod_split) { op.ln_mod(x, 0, msin, sc, sh, out, 0, H, seq); return; }
    op.ln_mod(x, 0, msin, sc, sh, out, 0, H, QA);
    op.ln_mod(x, (std::size_t)QA * H, msin_r, sc, sh, out,
              (std::size_t)QA * H, H, IS_REF);
  };
  auto gated_joint = [&](auto& op, const SharedBuffer& x, std::size_t g,
                         const SharedBuffer& sub) {
    if (!mod_split) { op.gated(x, 0, msin, g, sub, 0, H, seq * H); return; }
    op.gated(x, 0, msin, g, sub, 0, H, QA * H);
    op.gated(x, (std::size_t)QA * H, msin_r, g, sub, (std::size_t)QA * H, H,
             IS_REF * H);
  };

  // Joint self-attention over jq [seq, H] / jk,jv [KL, H] (already q/k
  // RMSNorm'd). head-major transpose -> rope -> flash attn -> transpose the
  // result into `out` (row stride out_rs; out_rs > 0 writes a sub-view, e.g.
  // att -> scat[:, :H]).
  auto attention = [&](auto& op, const SharedBuffer& out, int out_rs) {
    op.tr_rope(jq, qt, seq, HED, HD);   // fused transpose + rope (q)
    op.tr_rope(jk, kt, KL, HED, HD);    // fused transpose + rope (k)
    op.tr(jv, 0, vt, 0, KL, HED, HD);   // v: transpose only (no rope)
    if (use_steel) {
      // Group A: text + generated queries over every key.
      op.e->set_function(fn_attn);
      op.e->set_buffer(0, qt); op.e->set_buffer(1, kt); op.e->set_buffer(2, vt);
      op.e->set_buffer(3, atb); op.e->set_buffer(4, _attn_params);
      op.e->dispatch({32 * a_nqb, 4 * (unsigned)HED, 1}, {32, 4, 1});
      if (split_attn) {
        // Group B: reference queries over reference keys ONLY. Same buffers
        // re-based on the reference band; the per-head strides stay the full
        // row counts, so each head still lands on its own slice.
        const std::size_t ro = (std::size_t)QA * HD * 2;
        op.e->set_function(fn_attn_ref);
        op.e->set_buffer(0, qt, ro); op.e->set_buffer(1, kt, ro);
        op.e->set_buffer(2, vt, ro); op.e->set_buffer(3, atb, ro);
        op.e->set_buffer(4, attn_params_ref);
        op.e->dispatch({32 * a_nqb_ref, 4 * (unsigned)HED, 1}, {32, 4, 1});
      }
    } else {
      op.sdpa(qt, kt, vt, atb, scale, seq, HD, HED);
    }
    if (out_rs > 0) {
      op.tr_rs(atb, out, HED, seq, HD, out_rs);
    } else {
      op.tr(atb, 0, out, 0, HED, seq, HD);
    }
  };

  // JOIN ANY OUTSTANDING READ ON EVERY EXIT FROM HERE ON. A prefetch may
  // be filling a slot in either stack, and this forward returns from a
  // dozen places below. Freeing or reading a slot while a reader thread
  // writes into it is a use-after-free, and a scope guard is the only
  // version of this that cannot be forgotten at the thirteenth return.
  struct SlotJoin {
    BlockSlots<DoubleBlock>* d;
    BlockSlots<SingleBlock>* s;
    ~SlotJoin() { d->join(); s->join(); }
  } slot_join{&_double_slots, &_single_slots};
  _double_slots.begin_forward();
  _single_slots.begin_forward();
  // The adapter's [M, rank] intermediate, sized ONCE for the widest
  // projection this pass will take -- the single blocks' full joint
  // sequence. Without it apply() returns early and the adapter silently
  // does nothing, which looks exactly like an adapter that had no
  // effect, so it is allocated here rather than lazily in a block loop.
  if (_lora_modules > 0 && _lora_scale != 0.0f) {
    _lora.ensure_scratch((std::size_t)seq * (std::size_t)_lora_max_rank);
  }

  // ===== stream 1: conditioning + embed + double blocks =====
  if (prof) { mk = tnow(); }
  {
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    auto op = make_ops(enc);
    // Streaming: commit+reopen the stream at block boundaries so a just-in-time
    // block's weights are GPU-done before it is freed. No-op when preloaded.
    //
    // `between` runs AFTER the commit and BEFORE the wait -- the window
    // the next block's read is issued into, and the only reason this
    // takes a callback at all.
    auto flush_with = [&](auto&& between) {
      enc.end();
      CommandStream::Fence f = stream.commit();
      between();
      f.wait();
      stream = _mc->make_command_stream(); enc = stream.begin_compute();
      op = make_ops(enc);
    };
    auto flush = [&]() { flush_with([] {}); };
    // TimestepEmbedding: linear_1 -> SiLU -> linear_2.
    op.gemm(te_in, _t_emb1, te1, 0, 1, H, TD); op.bias(_t_emb1_b, te1, 1, H);
    op.elt(_fn_mulsig, te1, 0, te1, 0, te1, 0, H);       // SiLU
    op.gemm(te1, _t_emb2, temb, 0, 1, H, H); op.bias(_t_emb2_b, temb, 1, H);
    // Embedded guidance: guidance_embedder(SiLU) then temb += guidance_emb.
    if (use_g) {
      op.gemm(ge_in, _g_emb1, ge1, 0, 1, H, TD); op.bias(_g_emb1_b, ge1, 1, H);
      op.elt(_fn_mulsig, ge1, 0, ge1, 0, ge1, 0, H);       // SiLU
      op.gemm(ge1, _g_emb2, gemb, 0, 1, H, H); op.bias(_g_emb2_b, gemb, 1, H);
      op.elt(_fn_residual, temb, 0, gemb, 0, temb, 0, H);  // += guidance_emb
    }
    // Shared modulation: linear(SiLU(temb)).
    op.elt(_fn_mulsig, temb, 0, temb, 0, tsilu, 0, H);
    op.gemm(tsilu, _mod_img, mimg, 0, 1, 6 * H, H);
    op.gemm(tsilu, _mod_txt, mtxt, 0, 1, 6 * H, H);
    op.gemm(tsilu, _mod_single, msin, 0, 1, 3 * H, H);
    // The same two Linears again over the timestep-0 embedding: the reference
    // tokens' own modulation. They take the IMAGE double-stream set (they ride
    // the image stream) and the single-stream set, never the text one.
    if (kv_recipe) {
      op.gemm(te_ref, _t_emb1, tr1, 0, 1, H, TD);
      op.bias(_t_emb1_b, tr1, 1, H);
      op.elt(_fn_mulsig, tr1, 0, tr1, 0, tr1, 0, H);          // SiLU
      op.gemm(tr1, _t_emb2, temb_r, 0, 1, H, H);
      op.bias(_t_emb2_b, temb_r, 1, H);
      // Guidance rides the reference embedding too (BFL adds the SAME
      // guidance_in output to both vectors).
      if (use_g) {
        op.elt(_fn_residual, temb_r, 0, gemb, 0, temb_r, 0, H);
      }
      op.elt(_fn_mulsig, temb_r, 0, temb_r, 0, tsilu_r, 0, H);
      op.gemm(tsilu_r, _mod_img, mimg_r, 0, 1, 6 * H, H);
      op.gemm(tsilu_r, _mod_single, msin_r, 0, 1, 3 * H, H);
    }
    // Embed image + text. The generated tokens embed into img[0:IS_GEN]; each
    // reference embeds (same x_embedder) into the tail img[IS_GEN..IS]. (Calib
    // taps the generated tokens only -- calibration runs text-only, refs empty.)
    op.tap("emb_x", 0, latents_b, 0, IS_GEN, IC);
    op.gemm(latents_b, _x_embed, img, 0, IS_GEN, H, IC);
    // A reused cache carries the references' whole contribution as attention
    // keys, so they are not embedded and take no place in the token stream.
    if (!kv_reuse) {
      std::size_t ro = (std::size_t)IS_GEN;   // ref token offset into img
      for (std::size_t i = 0; i < refs.size(); ++i) {
        op.gemm(refs_b[i], _x_embed, img, ro * H, refs[i].seq, H, IC);
        ro += (std::size_t)refs[i].seq;
      }
    }
    op.tap("emb_ctx", 0, context_b, 0, TS, c.joint_dim);
    op.gemm(context_b, _ctx_embed, txt, 0, TS, H, c.joint_dim);
    if (_stream_blocks) { flush(); }   // commit conditioning before streaming
    // Re-arm residency growth for this forward; the ratchet from any
    // earlier eviction deliberately survives. Then the measurement that
    // actually finds the limit -- are the blocks we kept still in RAM?
    // Gated on our OWN compressed footprint moving, because the page
    // walk costs ~57 ms per 4.3 GB and a healthy run would pay it every
    // step to be told nothing.
    if (_wire.on()) {
      // Retry a pool that refused earlier -- the refusal may have been
      // another process spiking, and a run that never asks again holds a
      // small resident set for the whole schedule on the strength of one
      // syscall. Growth stopped when the budget ran out and cannot see
      // that the budget moved, so a successful retry has to say so.
      if (_wire.retry(_mc, wire_retry_slack_())) {
        _resid.note_landscape_changed();
      }
      // The trunk takes its place in the pool BEFORE this forward's block
      // admissions start asking for room: the blocks are the shed-able
      // half, so a pool that runs out should run out on them.
      wire_fixed_(true);
    }
    const auto mbudget = _mc->memory_budget();
    _resid.begin_forward(mbudget, [this]() -> std::size_t {
      return evict_tail_block_();
    });
    bool shortfall = false;
    if (_resid.count() > 0 &&
        _resid.self_compression_grew(mbudget.self_compressed)) {
      std::size_t examined = 0, incore = 0;
      resident_pages_(&examined, &incore);
      if (examined > 0 && incore < examined) {
        shortfall = true;
        std::size_t freed = _resid.note_weight_residency(
            examined, incore, [this]() -> std::size_t {
              return evict_tail_block_();
            });
        if (_mc->session() != nullptr) {
          _mc->session()->log_normal(fmt(
              "MetalFlux2Transformer: resident weights are only {}% in "
              "RAM ({} MB wired) -- released {} MB, now {} blocks resident",
              (int)(100.0 * (double)incore / (double)examined),
              _wire.wired_bytes() >> 20, freed >> 20,
              _resid.count()));
        }
      }
    }

    // Nothing of ours had left RAM this step -- either the walk said
    // so or there was no compression to make it worth walking.
    // Enough in a row lifts the ratchet by a block, so a shed taken
    // during a momentary squeeze is not the last word on the run.
    if (!shortfall) { _resid.note_healthy_forward(); }
    for (int L = 0; L < c.n_double; ++L) {
      // Pipeline stop -> abandon: checked EVERY block (not just the streamed
      // tail) so a slow high-res step responds within ~one block on the
      // preloaded path too.
      if (_stream_stop && _stream_stop()) { return {}; }
      if (_block_progress) {
        _block_progress(L, c.n_double + c.n_single);
      }
      // Resident once residency has promoted it, and read into a slot
      // until then. The vector is sized to n_double in streaming mode
      // and an unfilled entry reads as empty -- an INDEX test would be
      // wrong, since the resident set is not a prefix.
      const bool held = L < (int)_double.size() &&
                        !_double[(std::size_t)L].q.empty();
      const bool streaming = _stream_blocks && !held;
      const DoubleBlock* streamed = nullptr;
      if (streaming) {
        const auto rd0 = sp_now();
        // Two reusable destinations, refilled in place with pread and
        // with this block's read already issued under the previous
        // block's GPU work. See shared/block-slots.h.
        streamed = _double_slots.acquire(L);
        if (streamed == nullptr) { return {}; }
        sp_add(sp_read_ms, rd0);
        if (sprof) {
          sp_read_bytes += _double_slots.last_bytes();
          ++sp_blocks;
        }
      }
      const DoubleBlock& b =
          streaming ? *streamed : _double[(std::size_t)L];
      // Null unless an adapter is attached; every site below then passes
      // nullptr and takes its base path unchanged.
      const DoubleLora* dl = L < (int)_lora_double.size()
                                 ? &_lora_double[(std::size_t)L] : nullptr;
      // MSA: img (mod set 0) + txt.  mod layout [shift,scale,gate]*2 (each H).
      ln_mod_img(op, img, H, 0, nrm);
      op.tap("dbl_norm1_img", L, nrm, 0, IS, H);
      op.gemm(nrm, b.q, jq, (std::size_t)TS * H, IS, H, H, 0,
              dl ? &dl->q : nullptr);
      op.gemm(nrm, b.k, jk, (std::size_t)TS * H, IS, H, H, 0,
              dl ? &dl->k : nullptr);
      op.gemm(nrm, b.v, jv, (std::size_t)TS * H, IS, H, H, 0,
              dl ? &dl->v : nullptr);
      op.ln_mod(txt, 0, mtxt, H, 0, nrm, 0, H, TS);
      op.tap("dbl_norm1_txt", L, nrm, 0, TS, H);
      op.gemm(nrm, b.aq, jq, 0, TS, H, H, 0, dl ? &dl->aq : nullptr);
      op.gemm(nrm, b.ak, jk, 0, TS, H, H, 0, dl ? &dl->ak : nullptr);
      op.gemm(nrm, b.av, jv, 0, TS, H, H, 0, dl ? &dl->av : nullptr);
      op.rms(jq, 0, b.aqn, jq, 0, TS * HED, HD);
      op.rms(jk, 0, b.akn, jk, 0, TS * HED, HD);
      const std::size_t io = (std::size_t)TS * H;   // image region offset
      op.rms(jq, io, b.qn, jq, io, IS * HED, HD);
      op.rms(jk, io, b.kn, jk, io, IS * HED, HD);
      // KV cache, post-RMSNorm: lift the reference band out on the extracting
      // step, splice it back onto the key tail on every later one. Neither
      // touches the live rows, so the norms above are unaffected either way.
      if (kv_fill) {
        op.copy_rows(jk, (std::size_t)QA * H, kv->k[(std::size_t)L], 0,
                     IS_REF, H);
        op.copy_rows(jv, (std::size_t)QA * H, kv->v[(std::size_t)L], 0,
                     IS_REF, H);
      } else if (kv_reuse) {
        op.copy_rows(kv->k[(std::size_t)L], 0, jk, (std::size_t)seq * H,
                     IS_REF_ALL, H);
        op.copy_rows(kv->v[(std::size_t)L], 0, jv, (std::size_t)seq * H,
                     IS_REF_ALL, H);
      }
      attention(op, att, 0);                               // -> att (contiguous)
      op.tap("dbl_attn_txt", L, att, 0, TS, H);            // to_add_out input
      op.gemm(att, b.ao, ob, 0, TS, H, H, 0,               // text att[0:TS]
              dl ? &dl->ao : nullptr);
      op.gated(txt, 0, mtxt, 2 * H, ob, 0, H, TS * H);
      // img att is att[TS:seq] -> read at input offset TS*H (xe).
      op.tap("dbl_attn_img", L, att, (std::size_t)TS * H, IS, H);
      op.gemm(att, b.o, ob, 0, IS, H, H, (std::size_t)TS * H,
              dl ? &dl->o : nullptr);
      gated_img(op, img, 2 * H, ob);
      // FF (mod set 1: shift_mlp=3H, scale_mlp=4H, gate_mlp=5H). Flux2FeedForward
      // is SwiGLU: linear_in -> [gate|up] (2*INNER) -> silu(gate)*up -> linear_out.
      ln_mod_img(op, img, 4 * H, 3 * H, nrm);
      op.tap("dbl_norm2_img", L, nrm, 0, IS, H);
      if (_fuse_ff) {
        op.swiglu_ff(nrm, b.ff_in, smlp, IS, H, DFF);    // silu(gate)*up [IS,INNER]
      } else {
        op.gemm(nrm, b.ff_in, ff1, 0, IS, DFF, H, 0,     // [IS, 2*INNER]
                dl ? &dl->ff_in : nullptr);
        op.slice(ff1, sg, IS, DFF, INNER, 0);            // gate = first half
        op.slice(ff1, su, IS, DFF, INNER, INNER);        // up = second half
        op.elt(_fn_swiglu, sg, 0, su, 0, smlp, 0, IS * INNER);
      }
      op.tap("dbl_ffact_img", L, smlp, 0, IS, INNER);
      op.gemm(smlp, b.ff_out, ob, 0, IS, H, INNER, 0,
              dl ? &dl->ff_out : nullptr);
      gated_img(op, img, 5 * H, ob);
      op.ln_mod(txt, 0, mtxt, 4 * H, 3 * H, nrm, 0, H, TS);
      op.tap("dbl_norm2_txt", L, nrm, 0, TS, H);
      if (_fuse_ff) {
        op.swiglu_ff(nrm, b.cff_in, smlp, TS, H, DFF);
      } else {
        op.gemm(nrm, b.cff_in, ff1, 0, TS, DFF, H, 0,
                dl ? &dl->cff_in : nullptr);
        op.slice(ff1, sg, TS, DFF, INNER, 0);
        op.slice(ff1, su, TS, DFF, INNER, INNER);
        op.elt(_fn_swiglu, sg, 0, su, 0, smlp, 0, TS * INNER);
      }
      op.tap("dbl_ffact_txt", L, smlp, 0, TS, INNER);
      op.gemm(smlp, b.cff_out, ob, 0, TS, H, INNER, 0,
              dl ? &dl->cff_out : nullptr);
      op.gated(txt, 0, mtxt, 5 * H, ob, 0, H, TS * H);
      if (streaming) {
        const auto gp0 = sp_now();
        // Commit block L, issue the NEXT block's read into the free
        // slot while the GPU works through it, then wait.
        flush_with([&]() {
          int nxt = -1;
          for (int n = L + 1; n < c.n_double; ++n) {
            const bool h = n < (int)_double.size() &&
                           !_double[(std::size_t)n].q.empty();
            if (!h) { nxt = n; break; }
          }
          _double_slots.prefetch(nxt);
        });
        sp_add(sp_gpu_ms, gp0);
        // The flush above has been WAITED for, so nothing encoded still
        // points at this block's buffers -- which is why the promotion
        // happens here and not at the top of the iteration. Keeping it
        // costs nothing extra: the bytes are already resident, and the
        // alternative is re-reading the same block on every later step.
        const std::size_t nb = _double_slots.last_bytes();
        // Past the wire budget there is nothing to gain: the block would
        // be kept unprotected, the compressor would take it (it is the
        // coldest memory in the process), and the next residency walk
        // would shed a block and ratchet the ceiling over the whole
        // resident set. Better not to hold it at all.
        if (_wire.wirable(nb) && _resid.admit(_mc, nb) &&
            _double_slots.promote_into(_double[(std::size_t)L])) {
          // Wired LAST, after every write this block will ever get:
          // mlock pins the pages that exist NOW.
          _wire.note_wired(_mc,
                           wire_block_(_double[(std::size_t)L], true), nb);
          _resid.note_admitted(nb);
        }
      }
    }
    enc.end();
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt("MetalFlux2Transformer::forward_dit: {}",
                                 gpu_err.empty() ? "GPU failed" : gpu_err));
      }
      return {};
    }
  }
  if (prof) { t_dbl += ms_since(mk); }
  // Join streams: joint = [text; image].
  std::memcpy(joint.contents(), txt.contents(), (std::size_t)TS * H * 2);
  std::memcpy(static_cast<_Float16*>(joint.contents()) + (std::size_t)TS * H,
              img.contents(), (std::size_t)IS * H * 2);

  // ===== single-stream blocks. Each is a pure feed-forward graph, so it runs
  // as ONE command stream: proj + mlp -> attention -> GPU concat[att|mlp] ->
  // to_out -> gated. The [att|mlp] assembly is a GPU kernel (concat_cols), so
  // -- unlike the old host memcpy concat -- nothing forces a mid-block
  // commit().wait(); the whole block commits once. When ff_direct (dense fused
  // + strided-transpose kernel available), the two producers write straight
  // into scat (att -> [:, :H], mlp -> [:, H:]) so there is no concat at all;
  // the quant/unfused paths still concat (their swiglu is a shared kernel with
  // no strided output). =====
  // Needs the transpose_rs kernel (att -> scat[:, :H]) + a strided mlp kernel
  // for the active path: dense uses the flux2-only dense swiglu (out_stride
  // built in); quant + unfused use the new _rs twins.
  const bool have_mlp_rs =
      _fuse_ff ? (_quant_bits == 0
                      ? true
                      : (_fn_qmm_swiglu4_bm64_rs.valid()
                         && _fn_qmm_swiglu8_bm64_rs.valid()))
               : _fn_swiglu_rs.valid();
  const bool ff_direct = _fn_transpose_rs.valid() && have_mlp_rs;
  for (int L = 0; L < c.n_single; ++L) {
    if (_stream_stop && _stream_stop()) { return {}; }   // pipeline stop, any mode
    // Continues the double stack's numbering: one progress sequence over
    // the whole forward, not two that each restart at zero.
    if (_block_progress) {
      _block_progress(c.n_double + L, c.n_double + c.n_single);
    }
    // As above: resident once promoted, read into a slot until then.
    const bool held = L < (int)_single.size() &&
                      !_single[(std::size_t)L].o.empty();
    const bool streaming = _stream_blocks && !held;
    const SingleBlock* streamed = nullptr;
    if (streaming) {
      const auto rd0 = sp_now();
      streamed = _single_slots.acquire(L);
      if (streamed == nullptr) { return {}; }
      sp_add(sp_read_ms, rd0);
      if (sprof) {
        sp_read_bytes += single_read_bytes_(*streamed);
        ++sp_blocks;
      }
    }
    const SingleBlock& b = streaming ? *streamed : _single[(std::size_t)L];
    // As in the double stack: null unless an adapter is attached. The
    // fused branch below carries none by construction -- an adapter that
    // names to_qkv_mlp_proj turned the fusion off at load, so whenever
    // that factor exists this block is on the unfused path.
    const SingleLora* sl = L < (int)_lora_single.size()
                               ? &_lora_single[(std::size_t)L] : nullptr;
    if (prof) { mk = tnow(); }
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    auto op = make_ops(enc);
    ln_mod_joint(op, joint, H, 0, nrm);                    // (1+scale)*LN+shift
    op.tap("sgl_norm", L, nrm, 0, seq, H);
    if (_fuse_ff) {
      // qkv-only proj [seq, 3H] + a fused-SwiGLU mlp GEMM writing smlp directly
      // (no [seq, 2*SMLP] gate|up intermediate + slice + swiglu).
      op.gemm(nrm, b.qkv, sproj, 0, seq, 3 * H, H);
      op.slice(sproj, jq, seq, 3 * H, H, 0);               // q
      op.slice(sproj, jk, seq, 3 * H, H, H);               // k
      op.slice(sproj, jv, seq, 3 * H, H, 2 * H);           // v
      if (ff_direct) {   // mlp -> scat[:, H:] (att will land in scat[:, :H])
        op.swiglu_ff(nrm, b.mlp_gu, scat, seq, H, 2 * SMLP, H + SMLP, H);
      } else {
        op.swiglu_ff(nrm, b.mlp_gu, smlp, seq, H, 2 * SMLP);
      }
    } else {
      op.gemm(nrm, b.qkv_mlp, sproj, 0, seq, PW, H, 0,
              sl ? &sl->qkv_mlp : nullptr);
      op.slice(sproj, jq, seq, PW, H, 0);                  // q
      op.slice(sproj, jk, seq, PW, H, H);                  // k
      op.slice(sproj, jv, seq, PW, H, 2 * H);              // v
      op.slice(sproj, sg, seq, PW, SMLP, 3 * H);           // mlp gate
      op.slice(sproj, su, seq, PW, SMLP, 3 * H + SMLP);    // mlp up
      if (ff_direct) {   // silu(gate)*up -> scat[:, H:] (no concat)
        op.swiglu_rs(sg, su, scat, seq, SMLP, H + SMLP, H);
      } else {
        op.elt(_fn_swiglu, sg, 0, su, 0, smlp, 0, seq * SMLP);   // -> smlp
      }
    }
    op.rms(jq, 0, b.qn, jq, 0, seq * HED, HD);
    op.rms(jk, 0, b.kn, jk, 0, seq * HED, HD);
    {
      const std::size_t ci = (std::size_t)(c.n_double + L);
      if (kv_fill) {
        op.copy_rows(jk, (std::size_t)QA * H, kv->k[ci], 0, IS_REF, H);
        op.copy_rows(jv, (std::size_t)QA * H, kv->v[ci], 0, IS_REF, H);
      } else if (kv_reuse) {
        op.copy_rows(kv->k[ci], 0, jk, (std::size_t)seq * H, IS_REF_ALL, H);
        op.copy_rows(kv->v[ci], 0, jv, (std::size_t)seq * H, IS_REF_ALL, H);
      }
    }
    // Profiling barriers split the one stream into timed slices (prof only).
    if (prof) {
      enc.end(); stream.commit().wait(); t_sgl_gemm += ms_since(mk);
      stream = _mc->make_command_stream(); enc = stream.begin_compute();
      op = make_ops(enc); mk = tnow();
    }
    if (ff_direct) {
      attention(op, scat, H + SMLP);   // att -> scat[:, :H] (mlp already in [H:])
    } else {
      attention(op, att, 0);           // att -> att, concat below
    }
    if (prof) {
      enc.end(); stream.commit().wait(); t_sgl_attn += ms_since(mk);
      stream = _mc->make_command_stream(); enc = stream.begin_compute();
      op = make_ops(enc); mk = tnow();
    }
    // scat = [att | mlp]. Direct-write already placed both; else concat on GPU.
    if (!ff_direct) { op.concat_cols(scat, att, smlp, seq, H, SMLP); }
    op.tap("sgl_cat", L, scat, 0, seq, H + SMLP);
    op.gemm(scat, b.o, ob, 0, seq, H, H + SMLP, 0,
            sl ? &sl->o : nullptr);
    gated_joint(op, joint, 2 * H, ob);                     // += gate * to_out
    enc.end();
    std::string gpu_err;
    const auto sgl_gp0 = sp_now();
    CommandStream::Fence sgl_fence = stream.commit();
    if (streaming) {
      // The next block's read, issued under THIS block's GPU work --
      // the whole reason the commit and the wait are separated here.
      int nxt = -1;
      for (int n = L + 1; n < c.n_single; ++n) {
        const bool h = n < (int)_single.size() &&
                       !_single[(std::size_t)n].o.empty();
        if (!h) { nxt = n; break; }
      }
      _single_slots.prefetch(nxt);
    }
    if (!sgl_fence.wait_ok(&gpu_err)) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt("MetalFlux2Transformer::forward_dit: {}",
                                 gpu_err.empty() ? "GPU failed" : gpu_err));
      }
      return {};
    }
    if (streaming) { sp_add(sp_gpu_ms, sgl_gp0); }
    if (streaming) {
      // Same safe point as the double stack: this block's stream has
      // been committed AND waited for just above.
      // What promotion will KEEP, not what the slot holds: the raw
      // projection is dropped just below and is neither wired nor
      // admitted. Asking with the slot's own bytes would book a
      // partial grant the box never refused, which clamps the wired
      // budget at the first promoted block.
      const std::size_t nb = single_resident_bytes_(*streamed);
      if (_wire.wirable(nb) && _resid.admit(_mc, nb) &&
          _single_slots.promote_into(_single[(std::size_t)L])) {
        // A RESIDENT single block does not need the raw projection --
        // qkv and mlp_gu are what the forward reads, and the slot keeps
        // its own copy for the next refill. Dropping it here is what
        // stops promotion costing the block twice over.
        if (!_single[(std::size_t)L].qkv.empty()) {
          _single[(std::size_t)L].qkv_mlp = QWeight{};
        }
        _wire.note_wired(_mc, wire_block_(_single[(std::size_t)L], true), nb);
        _resid.note_admitted(nb);
      }
    }
    if (prof) { t_sgl_out += ms_since(mk); }
  }

  if (sprof && sp_blocks > 0 && _mc->session() != nullptr) {
    const double tot = sp_read_ms + sp_gpu_ms;
    const double sp_alloc_ms =
        (_ws ? _ws->stats().streamed_alloc_ms : 0.0) - sp_alloc0;
    const double sp_fetch_ms =
        (_ws ? _ws->stats().streamed_fetch_ms : 0.0) - sp_fetch0;
    const double gbs = sp_read_ms > 0.0
        ? (double)sp_read_bytes / (sp_read_ms / 1000.0) / 1e9 : 0.0;
    _mc->session()->log_normal(fmt(
        "MetalFlux2Transformer: streamed {} blocks -- read {:.0f} ms ({} MB "
        "at {:.2f} GB/s, {:.1f} ms/block), gpu {:.0f} ms ({:.0f} ms/block); "
        "a perfect prefetch hides at most {:.1f}% of this pass [read = {:.0f} ms alloc + {:.0f} ms fetch at {:.2f} GB/s]",
        sp_blocks, sp_read_ms, sp_read_bytes >> 20, gbs,
        sp_read_ms / (double)sp_blocks, sp_gpu_ms,
        sp_gpu_ms / (double)sp_blocks,
        tot > 0.0 ? 100.0 * sp_read_ms / tot : 0.0,
        sp_alloc_ms, sp_fetch_ms,
        sp_fetch_ms > 0.0
            ? (double)sp_read_bytes / (sp_fetch_ms / 1000.0) / 1e9
            : 0.0));
  }

  // ===== final: AdaLayerNormContinuous(image tail, temb) + proj_out =====
  // norm_out.linear(SiLU(temb)) -> [2H] = (shift, scale); x = (1+scale)*LN(x) +
  // shift; velocity = proj_out(x).
  if (prof) { mk = tnow(); }
  {
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    auto op = make_ops(enc);
    op.elt(_fn_mulsig, temb, 0, temb, 0, tsilu, 0, H);
    op.gemm(tsilu, _norm_out_lin, mimg, 0, 1, 2 * H, H);   // reuse mimg [2H]
    // Only the generated image tokens (joint[TS .. TS+IS_GEN], the head of the
    // image region) get proj_out; the trailing reference tokens are discarded.
    op.ln(joint, (std::size_t)TS * H, nrm, 0, IS_GEN, H);
    // AdaLayerNormContinuous: chunk(emb,2) -> (scale@0, shift@H).
    op.adaln(nrm, 0, mimg, 0, H, nrm, 0, H, IS_GEN * H);
    op.tap("emb_proj", 0, nrm, 0, IS_GEN, H);
    op.gemm(nrm, _proj_out, velocity, 0, IS_GEN, OC, H);
    enc.end();
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt("MetalFlux2Transformer::forward_dit: {}",
                                 gpu_err.empty() ? "GPU failed" : gpu_err));
      }
      return {};
    }
  }
  if (prof) {
    t_final += ms_since(mk);
    const double tot = t_dbl + t_sgl_gemm + t_sgl_attn + t_sgl_cat +
                       t_sgl_out + t_final;
    if (_mc->session() != nullptr) {
      _mc->session()->log_normal(fmt(
          "FLUX.2 DiT profile (seq={} img={} txt={}, {}+{} blocks): total "
          "{} ms | double(embed+Nblk) {} | single-gemm(qkv_mlp+norm+swiglu) {} "
          "| single-attn {} | single-concat(host) {} | single-out(to_out) {} | "
          "final {}",
          seq, IS, TS, (int)_double.size(), (int)_single.size(),
          (long)tot, (long)t_dbl, (long)t_sgl_gemm, (long)t_sgl_attn,
          (long)t_sgl_cat, (long)t_sgl_out, (long)t_final));
    }
  }
  // Mark the cache usable only now: every earlier exit is a failure path, and
  // a half-filled cache flagged valid would be reused as if complete.
  if (kv_fill) {
    kv->ref_seq  = IS_REF_ALL;
    kv->text_seq = TS;
    kv->img_seq  = IS_GEN;
  }
  // Downcast the bf16 velocity back to f16 for the stage/test boundary (the
  // velocity is O(1), so f16 is lossless here).
  SharedBuffer vel_f16 = _mc->make_shared_buffer((std::size_t)IS_GEN * OC * 2);
  if (vel_f16.empty()) { return {}; }
  {
    const auto* s = static_cast<const std::uint16_t*>(velocity.contents());
    auto* d = static_cast<_Float16*>(vel_f16.contents());
    for (std::size_t i = 0; i < (std::size_t)IS_GEN * OC; ++i) {
      d[i] = (_Float16)bf16_to_f32_(s[i]);
    }
  }
  return vel_f16;
}

}  // namespace genai
}  // namespace vpipe
