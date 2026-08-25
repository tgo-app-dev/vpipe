#include "generative-models/boogu/metal-boogu-transformer.h"

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
#include <map>
#include <string>

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
using metal_compute::ComputeEncoder;
using metal_compute::CommandStream;
using metal_compute::ComputeFunction;

namespace {

// C++ mirror of mlx::steel::AttnParams (identical layout to the LM / FLUX.2).
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

// bf16 <-> f32 host helpers. The Boogu DiT runs in bf16: it is a 10B model with
// a 46-block residual stream and its conditioning comes straight off a Qwen3-VL
// last hidden state, whose attention-sink outliers overflow f16's 65504 range
// (the same class of failure that forced FLUX.2 / QIE to bf16).
// Namespace for this class's derived-tensor cache keys. A WeightSet is
// shared by everything reading one checkpoint, so a key has to say which
// class's transform produced the bytes, not just which tensor.
constexpr const char* kKey = "boogu-dit/";

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

// Load a checkpoint tensor -> bf16 SharedBuffer (F32/F16/BF16 sources).
SharedBuffer
to_bf16_(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm)
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

SharedBuffer
MetalBooguTransformer::bf16_(WeightSet& ws, const std::string& nm, Retain r)
{
  if (r == Retain::Streamed) {
    // Rebuilt per forward and retained by nobody -- going through the
    // set anyway is what keeps the streaming traffic visible to the
    // manager instead of happening behind its back.
    return ws.stream_derived(
        [&]() { return to_bf16_(ws.src(), _mc, nm); });
  }
  return ws.derived(std::string(kKey) + "bf16|" + nm,
                    [&]() { return to_bf16_(ws.src(), _mc, nm); });
}

MetalBooguTransformer::QWeight
MetalBooguTransformer::load_qw_(WeightSet& ws, const std::string& name,
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
    // Zero-copy mmap views (evictable) when enabled; else owned copies. The
    // codes/scales/qbias feed the GEMM read-only and are never CPU-modified,
    // so aliasing the file is safe. fuse_gu_ copies (a CPU relayout), which
    // correctly drops the mapping for the fused weight.
    const auto res = _mmap_weights ? WeightSet::Residency::Mapped
                                   : WeightSet::Residency::Copied;
    qw.codes = r == Retain::Streamed
                   ? ws.stream_tensor(name + ".weight", _mc, res)
                   : ws.tensor(name + ".weight", _mc, res);
    qw.scales = bf16_(ws, name + ".scales", r);
    qw.qbias  = bf16_(ws, name + ".biases", r);
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
  qw.w = bf16_(ws, name + ".weight", r);
  return qw;
}

MetalBooguTransformer::QWeight
MetalBooguTransformer::fuse_gu_(WeightSet& ws, const std::string& key,
                                const QWeight& gate, const QWeight& up,
                                Retain r)
{
  QWeight d;
  if (gate.empty() || up.empty() || gate.n != up.n || gate.k != up.k ||
      gate.quantized != up.quantized || gate.bits != up.bits || gate.n <= 0) {
    return d;
  }
  d.quantized = gate.quantized;
  d.bits = gate.bits;
  d.n = 2 * gate.n;
  d.k = gate.k;
  const int inner = gate.n;
  // Interleave row-wise: row 2g = gate_g, row 2g+1 = up_g, for each of the
  // parallel tensors (dense w, or codes/scales/qbias).
  auto weave = [&](const SharedBuffer& g, const SharedBuffer& u) -> SharedBuffer {
    if (g.empty() || u.empty()) { return {}; }
    const std::size_t rb = g.byte_size() / (std::size_t)inner;   // row bytes
    if (rb == 0 || u.byte_size() / (std::size_t)inner != rb) { return {}; }
    SharedBuffer dst = _mc->make_shared_buffer((std::size_t)2 * inner * rb);
    if (dst.empty()) { return {}; }
    const auto* gs = static_cast<const std::uint8_t*>(g.contents());
    const auto* us = static_cast<const std::uint8_t*>(u.contents());
    auto* dd = static_cast<std::uint8_t*>(dst.contents());
    for (int i = 0; i < inner; ++i) {
      std::memcpy(dd + (std::size_t)(2 * i) * rb, gs + (std::size_t)i * rb, rb);
      std::memcpy(dd + (std::size_t)(2 * i + 1) * rb, us + (std::size_t)i * rb,
                  rb);
    }
    return dst;
  };
  // The fused weight is the largest tensor in a block -- twice the gate
  // -- so it goes through the set like any other derived tensor: two
  // models over this checkpoint pay for the relayout once. Streamed
  // blocks rebuild it per forward and keep nothing, as they must.
  auto cached = [&](const char* tag, const SharedBuffer& g,
                    const SharedBuffer& u) -> SharedBuffer {
    if (r == Retain::Streamed) {
      return ws.stream_derived([&]() { return weave(g, u); });
    }
    return ws.derived(std::string(kKey) + "gu|" + key + "|" + tag,
                      [&]() { return weave(g, u); });
  };
  if (d.quantized) {
    d.codes  = cached("codes", gate.codes, up.codes);
    d.scales = cached("scales", gate.scales, up.scales);
    d.qbias  = cached("qbias", gate.qbias, up.qbias);
    if (d.codes.empty() || d.scales.empty() || d.qbias.empty()) {
      return QWeight{};
    }
  } else {
    d.w = cached("w", gate.w, up.w);
    if (d.w.empty()) { return QWeight{}; }
  }
  return d;
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
MetalBooguTransformer::wire_block_(DoubleBlock& b, bool on)
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
  QWeight* q[] = {&b.jq_i, &b.jk_i, &b.jv_i, &b.jq_t, &b.jk_t, &b.jv_t,
                  &b.jout_i, &b.jout_t, &b.jo, &b.sq, &b.sk, &b.sv, &b.so,
                  &b.iff_gate, &b.iff_up, &b.iff_down, &b.iff_gu,
                  &b.tff_gate, &b.tff_up, &b.tff_down, &b.tff_gu,
                  &b.mi1, &b.mi2, &b.mi3, &b.mt1, &b.mt2};
  for (QWeight* w : q) { qw(*w); }
  one(b.jqn); one(b.jkn); one(b.sqn); one(b.skn);
  return changed;
}

std::size_t
MetalBooguTransformer::wire_block_(Block& b, bool on)
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
  QWeight* q[] = {&b.q, &b.k, &b.v, &b.o, &b.ff_gate, &b.ff_up,
                  &b.ff_down, &b.ff_gu, &b.mod};
  for (QWeight* w : q) { qw(*w); }
  one(b.qn); one(b.kn); one(b.mod_b);
  one(b.n1); one(b.n2); one(b.fn1); one(b.fn2);
  return changed;
}

// The TRUNK: everything the weight set cached for this model -- the five
// refiner stacks, the embedders, the caption tower. Read on every forward
// and never shed, so it has a better claim on the pool than any single
// resident block does, which is why the forward wires this BEFORE it
// starts admitting blocks.
//
// The activation scratch is deliberately not here: this model allocates
// it as locals inside forward_dit, so there is nothing that persists to
// wire. See the note in shared/wired-pool.h.
std::size_t
MetalBooguTransformer::wire_fixed_(bool on)
{
  if (!_ws) { return 0; }
  std::size_t changed = 0;
  _ws->for_each_weight([&](metal_compute::SharedBuffer& b) {
    changed += _wire.wire_one(_mc, b, on);
  });
  return changed;
}

std::size_t
MetalBooguTransformer::qw_bytes_(const QWeight& w)
{
  return w.w.byte_size() + w.codes.byte_size() + w.scales.byte_size() +
         w.qbias.byte_size();
}

std::size_t
MetalBooguTransformer::double_bytes_(const DoubleBlock& b)
{
  return qw_bytes_(b.jq_i) + qw_bytes_(b.jk_i) + qw_bytes_(b.jv_i) +
         qw_bytes_(b.jq_t) + qw_bytes_(b.jk_t) + qw_bytes_(b.jv_t) +
         qw_bytes_(b.jout_i) + qw_bytes_(b.jout_t) + qw_bytes_(b.jo) +
         b.jqn.byte_size() + b.jkn.byte_size() +
         qw_bytes_(b.sq) + qw_bytes_(b.sk) + qw_bytes_(b.sv) +
         qw_bytes_(b.so) + b.sqn.byte_size() + b.skn.byte_size() +
         qw_bytes_(b.iff_gate) + qw_bytes_(b.iff_up) + qw_bytes_(b.iff_down) +
         qw_bytes_(b.iff_gu) + qw_bytes_(b.tff_gate) + qw_bytes_(b.tff_up) +
         qw_bytes_(b.tff_down) + qw_bytes_(b.tff_gu) +
         qw_bytes_(b.mi1) + qw_bytes_(b.mi2) + qw_bytes_(b.mi3) +
         qw_bytes_(b.mt1) + qw_bytes_(b.mt2);
}

std::size_t
MetalBooguTransformer::single_bytes_(const Block& b)
{
  return qw_bytes_(b.q) + qw_bytes_(b.k) + qw_bytes_(b.v) + qw_bytes_(b.o) +
         qw_bytes_(b.ff_gate) + qw_bytes_(b.ff_up) + qw_bytes_(b.ff_down) +
         qw_bytes_(b.ff_gu) + qw_bytes_(b.mod) +
         b.qn.byte_size() + b.kn.byte_size() + b.mod_b.byte_size() +
         b.n1.byte_size() + b.n2.byte_size() + b.fn1.byte_size() +
         b.fn2.byte_size();
}

void
MetalBooguTransformer::resident_pages_(std::size_t* examined,
                                       std::size_t* incore,
                                       std::size_t* paged_out) const
{
  *examined = 0;
  *incore = 0;
  if (paged_out != nullptr) { *paged_out = 0; }
  auto add = [&](const metal_compute::SharedBuffer& p) {
    if (p.byte_size() == 0) { return; }
    // A WIRED BUFFER CANNOT HAVE LEFT RAM, so asking is spending the walk
    // to be told what mlock already guarantees. Skipped PER BUFFER rather
    // than per block, because wire_block_ stops at the first refusal and
    // leaves the rest of that block unwired -- the remainder is exactly
    // what still needs measuring. With everything wired `examined` stays
    // 0, which the caller reads as "no evidence" rather than as a
    // shortfall, and that is the correct answer.
    if (p.is_wired()) { return; }
    const auto r = p.page_residency(64);
    if (!r.valid) { return; }
    *examined += r.examined;
    *incore += r.incore;
    if (paged_out != nullptr) { *paged_out += r.paged_out; }
  };
  // The DOMINANT buffers only: a block's bytes are almost all in the
  // projections, and walking the norms as well would double the syscalls
  // to sharpen a fraction the answer does not turn on.
  auto addq = [&](const QWeight& w) { add(w.w); add(w.codes); };
  for (const DoubleBlock& b : _double) {
    addq(b.jq_i); addq(b.jk_i); addq(b.jv_i);
    addq(b.jq_t); addq(b.jk_t); addq(b.jv_t);
    addq(b.jout_i); addq(b.jout_t); addq(b.jo);
    addq(b.sq); addq(b.sk); addq(b.sv); addq(b.so);
    addq(b.iff_gate); addq(b.iff_up); addq(b.iff_down); addq(b.iff_gu);
    addq(b.tff_gate); addq(b.tff_up); addq(b.tff_down); addq(b.tff_gu);
  }
  for (const Block& b : _single) {
    addq(b.q); addq(b.k); addq(b.v); addq(b.o);
    addq(b.ff_gate); addq(b.ff_up); addq(b.ff_down); addq(b.ff_gu);
  }
}

std::size_t
MetalBooguTransformer::evict_tail_block_()
{
  for (int i = (int)_single.size() - 1; i >= 0; --i) {
    Block& b = _single[(std::size_t)i];
    const std::size_t n = single_bytes_(b);
    if (n == 0) { continue; }
    // Before the buffers go: give the wiring back. Dropping a wired
    // buffer unwires it in the kernel anyway, but only unwire_from_pool()
    // decrements the pool's counter -- so doing it here is what keeps the
    // budget honest instead of leaking a block's worth per eviction.
    _wire.note_unwired(wire_block_(b, false));
    b = Block{};
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
MetalBooguTransformer::wire_retry_slack_() const
{
  if (_wire_block_hint > 0) { return _wire_block_hint; }
  if (_resid.count() > 0) {
    return _resid.bytes() / (std::size_t)_resid.count();
  }
  return 0;
}

void
MetalBooguTransformer::set_residency_schedule(int steps)
{
  if (!_ws) { return; }
  const MetalLlamaWeights& src = _ws->src();
  const std::size_t blk = widest_block_bytes(
      src.tensor_names(),
      [&](const std::string& n) {
        const auto* ti = src.info(n);
        return ti != nullptr ? (std::size_t)ti->nbytes : (std::size_t)0;
      },
      {"double_stream_layers.", "single_stream_layers."});
  const int nl = _cfg.n_double + _cfg.n_single;
  _wire_block_hint = blk;
  _resid.set_schedule(steps, nl, blk, _wire.on(),
                      _mc != nullptr ? _mc->memory_budget()
                                     : metal_compute::MetalCompute::
                                           MemoryBudget{});
  if (_mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalBooguTransformer: residency probe {} blocks of {} "
        "({} MB each, {} MB reclaimable, wire budget {} MB){}",
        _resid.per_forward_cap(), nl, blk >> 20,
        _mc->memory_budget().available_physical >> 20,
        _wire.budget() >> 20,
        _wire.on() ? " -- uncapped, the wire budget is the gate"
                   : ", doubling per healthy forward"));
  }
}

std::size_t
MetalBooguTransformer::release_resident_blocks(std::size_t bytes)
{
  const std::size_t freed =
      _resid.release(bytes, [this]() -> std::size_t {
        return evict_tail_block_();
      });
  if (freed > 0 && _mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalBooguTransformer: released {} MB of resident blocks "
        "({} left)", freed >> 20, _resid.count()));
  }
  return freed;
}

bool
MetalBooguTransformer::load_block_(WeightSet& ws,
                                   const std::string& pre, Block& b,
                                   bool modulated, Retain r)
{
  b.modulated = modulated;
  b.q = load_qw_(ws, pre + "attn.to_q", r);
  b.k = load_qw_(ws, pre + "attn.to_k", r);
  b.v = load_qw_(ws, pre + "attn.to_v", r);
  b.o = load_qw_(ws, pre + "attn.to_out.0", r);
  b.qn = bf16_(ws, pre + "attn.norm_q.weight", r);
  b.kn = bf16_(ws, pre + "attn.norm_k.weight", r);
  b.ff_gate = load_qw_(ws, pre + "feed_forward.linear_1", r);
  b.ff_up   = load_qw_(ws, pre + "feed_forward.linear_3", r);
  b.ff_down = load_qw_(ws, pre + "feed_forward.linear_2", r);
  if (_cfg.ff_inner == 0 && b.ff_gate.n > 0) { _cfg.ff_inner = b.ff_gate.n; }
  if (modulated) {
    b.mod   = load_qw_(ws, pre + "norm1.linear", r);
    b.mod_b = bf16_(ws, pre + "norm1.linear.bias", r);
    b.n1    = bf16_(ws, pre + "norm1.norm.weight", r);
    if (b.mod.empty() || b.mod_b.empty()) { return false; }
  } else {
    b.n1 = bf16_(ws, pre + "norm1.weight", r);
  }
  b.n2  = bf16_(ws, pre + "norm2.weight", r);
  b.fn1 = bf16_(ws, pre + "ffn_norm1.weight", r);
  b.fn2 = bf16_(ws, pre + "ffn_norm2.weight", r);
  if (_fuse_ff) {
    b.ff_gu = fuse_gu_(ws, pre + "feed_forward", b.ff_gate, b.ff_up, r);
  }
  return !b.q.empty() && !b.k.empty() && !b.v.empty() && !b.o.empty() &&
         !b.qn.empty() && !b.kn.empty() && !b.ff_gate.empty() &&
         !b.ff_up.empty() && !b.ff_down.empty() && !b.n1.empty() &&
         !b.n2.empty() && !b.fn1.empty() && !b.fn2.empty() &&
         (!_fuse_ff || !b.ff_gu.empty());
}

bool
MetalBooguTransformer::load_double_(WeightSet& ws,
                                    const std::string& pre, DoubleBlock& b,
                                    Retain r)
{
  const std::string jp = pre + "img_instruct_attn.";
  const std::string pp = jp + "processor.";
  b.jq_i = load_qw_(ws, pp + "img_to_q", r);
  b.jk_i = load_qw_(ws, pp + "img_to_k", r);
  b.jv_i = load_qw_(ws, pp + "img_to_v", r);
  b.jq_t = load_qw_(ws, pp + "instruct_to_q", r);
  b.jk_t = load_qw_(ws, pp + "instruct_to_k", r);
  b.jv_t = load_qw_(ws, pp + "instruct_to_v", r);
  b.jout_i = load_qw_(ws, pp + "img_out", r);
  b.jout_t = load_qw_(ws, pp + "instruct_out", r);
  b.jo  = load_qw_(ws, jp + "to_out.0", r);
  b.jqn = bf16_(ws, jp + "norm_q.weight", r);
  b.jkn = bf16_(ws, jp + "norm_k.weight", r);

  const std::string sp = pre + "img_self_attn.";
  b.sq = load_qw_(ws, sp + "to_q", r);
  b.sk = load_qw_(ws, sp + "to_k", r);
  b.sv = load_qw_(ws, sp + "to_v", r);
  b.so = load_qw_(ws, sp + "to_out.0", r);
  b.sqn = bf16_(ws, sp + "norm_q.weight", r);
  b.skn = bf16_(ws, sp + "norm_k.weight", r);

  b.iff_gate = load_qw_(ws, pre + "img_feed_forward.linear_1", r);
  b.iff_up   = load_qw_(ws, pre + "img_feed_forward.linear_3", r);
  b.iff_down = load_qw_(ws, pre + "img_feed_forward.linear_2", r);
  b.tff_gate = load_qw_(ws, pre + "instruct_feed_forward.linear_1", r);
  b.tff_up   = load_qw_(ws, pre + "instruct_feed_forward.linear_3", r);
  b.tff_down = load_qw_(ws, pre + "instruct_feed_forward.linear_2", r);
  if (_cfg.ff_inner == 0 && b.iff_gate.n > 0) { _cfg.ff_inner = b.iff_gate.n; }
  if (_fuse_ff) {
    b.iff_gu = fuse_gu_(ws, pre + "img_feed_forward",
                        b.iff_gate, b.iff_up, r);
    b.tff_gu = fuse_gu_(ws, pre + "instruct_feed_forward",
                        b.tff_gate, b.tff_up, r);
  }

  auto mod = [&](const char* nm, QWeight& w, SharedBuffer& bias,
                 SharedBuffer& nw) {
    w    = load_qw_(ws, pre + nm + ".linear", r);
    bias = bf16_(ws, pre + std::string(nm) + ".linear.bias", r);
    nw   = bf16_(ws, pre + std::string(nm) + ".norm.weight", r);
  };
  mod("img_norm1", b.mi1, b.mi1_b, b.ni1);
  mod("img_norm2", b.mi2, b.mi2_b, b.ni2);
  mod("img_norm3", b.mi3, b.mi3_b, b.ni3);
  mod("instruct_norm1", b.mt1, b.mt1_b, b.nt1);
  mod("instruct_norm2", b.mt2, b.mt2_b, b.nt2);

  b.i_attn_n = bf16_(ws, pre + "img_attn_norm.weight", r);
  b.i_self_n = bf16_(ws, pre + "img_self_attn_norm.weight", r);
  b.i_ffn1   = bf16_(ws, pre + "img_ffn_norm1.weight", r);
  b.i_ffn2   = bf16_(ws, pre + "img_ffn_norm2.weight", r);
  b.t_attn_n = bf16_(ws, pre + "instruct_attn_norm.weight", r);
  b.t_ffn1   = bf16_(ws, pre + "instruct_ffn_norm1.weight", r);
  b.t_ffn2   = bf16_(ws, pre + "instruct_ffn_norm2.weight", r);

  return !b.jq_i.empty() && !b.jk_i.empty() && !b.jv_i.empty() &&
         !b.jq_t.empty() && !b.jk_t.empty() && !b.jv_t.empty() &&
         !b.jout_i.empty() && !b.jout_t.empty() && !b.jo.empty() &&
         !b.jqn.empty() && !b.jkn.empty() &&
         !b.sq.empty() && !b.sk.empty() && !b.sv.empty() && !b.so.empty() &&
         !b.sqn.empty() && !b.skn.empty() &&
         !b.iff_gate.empty() && !b.iff_up.empty() && !b.iff_down.empty() &&
         !b.tff_gate.empty() && !b.tff_up.empty() && !b.tff_down.empty() &&
         !b.mi1.empty() && !b.mi2.empty() && !b.mi3.empty() &&
         !b.mt1.empty() && !b.mt2.empty() &&
         !b.mi1_b.empty() && !b.mi2_b.empty() && !b.mi3_b.empty() &&
         !b.mt1_b.empty() && !b.mt2_b.empty() &&
         !b.ni1.empty() && !b.ni2.empty() && !b.ni3.empty() &&
         !b.nt1.empty() && !b.nt2.empty() &&
         !b.i_attn_n.empty() && !b.i_self_n.empty() && !b.i_ffn1.empty() &&
         !b.i_ffn2.empty() && !b.t_attn_n.empty() && !b.t_ffn1.empty() &&
         !b.t_ffn2.empty() &&
         (!_fuse_ff || (!b.iff_gu.empty() && !b.tff_gu.empty()));
}


// Shorthand for the per-tensor placement the refill needs stated.
using P = vpipe::genai::Placement;

// ---- the streamed blocks' reusable destinations ------------------------

// A MATRIX is three tensors when quantized and one when dense, and which
// it is was decided when the slot was built -- so the layout comes from
// the QWeight, not from the checkpoint. A pack that disagrees fails the
// refill's size check and forces a rebuild, which is the right answer to
// "these are not the same weights".
namespace {
template <class Fn, class QW>
void
boogu_qw_(const std::string& base, QW& w, const Fn& fn)
{
  if (w.quantized) {
    fn(base + ".weight", w.codes, P::kRaw);
    fn(base + ".scales", w.scales, P::kBf16);
    fn(base + ".biases", w.qbias, P::kBf16);
  } else {
    fn(base + ".weight", w.w, P::kBf16);
  }
}
}  // namespace

void
MetalBooguTransformer::each_single_tensor_(
    int L, Block& b, const BlockSlots<Block>::TensorFn& fn) const
{
  const std::string p = "single_stream_layers." + std::to_string(L) + ".";
  boogu_qw_(p + "attn.to_q", b.q, fn);
  boogu_qw_(p + "attn.to_k", b.k, fn);
  boogu_qw_(p + "attn.to_v", b.v, fn);
  boogu_qw_(p + "attn.to_out.0", b.o, fn);
  fn(p + "attn.norm_q.weight", b.qn, P::kBf16);
  fn(p + "attn.norm_k.weight", b.kn, P::kBf16);
  boogu_qw_(p + "feed_forward.linear_1", b.ff_gate, fn);
  boogu_qw_(p + "feed_forward.linear_3", b.ff_up, fn);
  boogu_qw_(p + "feed_forward.linear_2", b.ff_down, fn);
  if (b.modulated) {
    boogu_qw_(p + "norm1.linear", b.mod, fn);
    fn(p + "norm1.linear.bias", b.mod_b, P::kBf16);
    fn(p + "norm1.norm.weight", b.n1, P::kBf16);
  } else {
    fn(p + "norm1.weight", b.n1, P::kBf16);
  }
  fn(p + "norm2.weight", b.n2, P::kBf16);
  fn(p + "ffn_norm1.weight", b.fn1, P::kBf16);
  fn(p + "ffn_norm2.weight", b.fn2, P::kBf16);
  // ff_gu is DERIVED -- an interleave of gate|up with no checkpoint name
  // -- and is rebuilt by weave_into_ after the refill, not here.
}

void
MetalBooguTransformer::each_double_tensor_(
    int L, DoubleBlock& b, const BlockSlots<DoubleBlock>::TensorFn& fn) const
{
  const std::string p = "double_stream_layers." + std::to_string(L) + ".";
  const std::string jp = p + "img_instruct_attn.";
  const std::string pp = jp + "processor.";
  boogu_qw_(pp + "img_to_q", b.jq_i, fn);
  boogu_qw_(pp + "img_to_k", b.jk_i, fn);
  boogu_qw_(pp + "img_to_v", b.jv_i, fn);
  boogu_qw_(pp + "instruct_to_q", b.jq_t, fn);
  boogu_qw_(pp + "instruct_to_k", b.jk_t, fn);
  boogu_qw_(pp + "instruct_to_v", b.jv_t, fn);
  boogu_qw_(pp + "img_out", b.jout_i, fn);
  boogu_qw_(pp + "instruct_out", b.jout_t, fn);
  boogu_qw_(jp + "to_out.0", b.jo, fn);
  fn(jp + "norm_q.weight", b.jqn, P::kBf16);
  fn(jp + "norm_k.weight", b.jkn, P::kBf16);

  const std::string sp = p + "img_self_attn.";
  boogu_qw_(sp + "to_q", b.sq, fn);
  boogu_qw_(sp + "to_k", b.sk, fn);
  boogu_qw_(sp + "to_v", b.sv, fn);
  boogu_qw_(sp + "to_out.0", b.so, fn);
  fn(sp + "norm_q.weight", b.sqn, P::kBf16);
  fn(sp + "norm_k.weight", b.skn, P::kBf16);

  boogu_qw_(p + "img_feed_forward.linear_1", b.iff_gate, fn);
  boogu_qw_(p + "img_feed_forward.linear_3", b.iff_up, fn);
  boogu_qw_(p + "img_feed_forward.linear_2", b.iff_down, fn);
  boogu_qw_(p + "instruct_feed_forward.linear_1", b.tff_gate, fn);
  boogu_qw_(p + "instruct_feed_forward.linear_3", b.tff_up, fn);
  boogu_qw_(p + "instruct_feed_forward.linear_2", b.tff_down, fn);

  const auto mod = [&](const char* nm, QWeight& w,
                       metal_compute::SharedBuffer& bias,
                       metal_compute::SharedBuffer& nw) {
    boogu_qw_(p + nm + ".linear", w, fn);
    fn(p + std::string(nm) + ".linear.bias", bias, P::kBf16);
    fn(p + std::string(nm) + ".norm.weight", nw, P::kBf16);
  };
  mod("img_norm1", b.mi1, b.mi1_b, b.ni1);
  mod("img_norm2", b.mi2, b.mi2_b, b.ni2);
  mod("img_norm3", b.mi3, b.mi3_b, b.ni3);
  mod("instruct_norm1", b.mt1, b.mt1_b, b.nt1);
  mod("instruct_norm2", b.mt2, b.mt2_b, b.nt2);

  fn(p + "img_attn_norm.weight", b.i_attn_n, P::kBf16);
  fn(p + "img_self_attn_norm.weight", b.i_self_n, P::kBf16);
  fn(p + "img_ffn_norm1.weight", b.i_ffn1, P::kBf16);
  fn(p + "img_ffn_norm2.weight", b.i_ffn2, P::kBf16);
  fn(p + "instruct_attn_norm.weight", b.t_attn_n, P::kBf16);
  fn(p + "instruct_ffn_norm1.weight", b.t_ffn1, P::kBf16);
  fn(p + "instruct_ffn_norm2.weight", b.t_ffn2, P::kBf16);
  // iff_gu / tff_gu are DERIVED -- see weave_into_.
}

bool
MetalBooguTransformer::weave_into_(const QWeight& gate, const QWeight& up,
                                   QWeight& dst) const
{
  if (dst.empty()) { return true; }        // not fused: nothing to rebuild
  if (gate.empty() || up.empty() || gate.n <= 0 || gate.n != up.n) {
    return false;
  }
  bool ok = true;
  const int inner = gate.n;
  const auto one = [&](const metal_compute::SharedBuffer& g,
                       const metal_compute::SharedBuffer& u,
                       metal_compute::SharedBuffer& d) {
    if (!ok || d.empty()) { return; }
    if (g.empty() || u.empty()) { ok = false; return; }
    const std::size_t rb = g.byte_size() / (std::size_t)inner;
    if (rb == 0 || u.byte_size() / (std::size_t)inner != rb ||
        d.byte_size() != (std::size_t)2 * inner * rb) {
      ok = false;
      return;
    }
    const auto* gs = static_cast<const std::uint8_t*>(g.contents());
    const auto* us = static_cast<const std::uint8_t*>(u.contents());
    auto* dd = static_cast<std::uint8_t*>(d.contents());
    for (int i = 0; i < inner; ++i) {
      std::memcpy(dd + (std::size_t)(2 * i) * rb,
                  gs + (std::size_t)i * rb, rb);
      std::memcpy(dd + (std::size_t)(2 * i + 1) * rb,
                  us + (std::size_t)i * rb, rb);
    }
  };
  one(gate.w, up.w, dst.w);
  one(gate.codes, up.codes, dst.codes);
  one(gate.scales, up.scales, dst.scales);
  one(gate.qbias, up.qbias, dst.qbias);
  return ok;
}


// Allocate `dst` with `src`'s shapes and flags, optionally copying the
// bytes. One function for two uses: a promotion and the second slot
// differ only in whether the contents come along.
//
// Unlike the each_*_tensor_ walks, these cover the DERIVED fused
// weights too -- a slot without them would have nowhere for
// weave_into_ to write.
bool
MetalBooguTransformer::clone_single_(const Block& src, Block& dst,
                                     bool copy) const
{
  bool ok = true;
  const auto one = [&](const metal_compute::SharedBuffer& s,
                       metal_compute::SharedBuffer& d) {
    if (!ok || s.empty()) { d = metal_compute::SharedBuffer{}; return; }
    d = _mc->make_shared_buffer(s.byte_size());
    if (d.empty()) { ok = false; return; }
    if (copy) { std::memcpy(d.contents(), s.contents(), s.byte_size()); }
  };
  const auto qw = [&](const QWeight& s, QWeight& d) {
    d.quantized = s.quantized; d.bits = s.bits; d.n = s.n; d.k = s.k;
    one(s.w, d.w); one(s.codes, d.codes);
    one(s.scales, d.scales); one(s.qbias, d.qbias);
  };
  dst.modulated = src.modulated;
  qw(src.q, dst.q); qw(src.k, dst.k); qw(src.v, dst.v); qw(src.o, dst.o);
  one(src.qn, dst.qn); one(src.kn, dst.kn);
  qw(src.ff_gate, dst.ff_gate); qw(src.ff_up, dst.ff_up);
  qw(src.ff_down, dst.ff_down); qw(src.ff_gu, dst.ff_gu);
  qw(src.mod, dst.mod); one(src.mod_b, dst.mod_b);
  one(src.n1, dst.n1); one(src.n2, dst.n2);
  one(src.fn1, dst.fn1); one(src.fn2, dst.fn2);
  if (!ok) { dst = Block{}; }
  return ok;
}

bool
MetalBooguTransformer::clone_double_(const DoubleBlock& src, DoubleBlock& dst,
                                     bool copy) const
{
  bool ok = true;
  const auto one = [&](const metal_compute::SharedBuffer& s,
                       metal_compute::SharedBuffer& d) {
    if (!ok || s.empty()) { d = metal_compute::SharedBuffer{}; return; }
    d = _mc->make_shared_buffer(s.byte_size());
    if (d.empty()) { ok = false; return; }
    if (copy) { std::memcpy(d.contents(), s.contents(), s.byte_size()); }
  };
  const auto qw = [&](const QWeight& s, QWeight& d) {
    d.quantized = s.quantized; d.bits = s.bits; d.n = s.n; d.k = s.k;
    one(s.w, d.w); one(s.codes, d.codes);
    one(s.scales, d.scales); one(s.qbias, d.qbias);
  };
  qw(src.jq_i, dst.jq_i); qw(src.jk_i, dst.jk_i); qw(src.jv_i, dst.jv_i);
  qw(src.jq_t, dst.jq_t); qw(src.jk_t, dst.jk_t); qw(src.jv_t, dst.jv_t);
  qw(src.jout_i, dst.jout_i); qw(src.jout_t, dst.jout_t); qw(src.jo, dst.jo);
  one(src.jqn, dst.jqn); one(src.jkn, dst.jkn);
  qw(src.sq, dst.sq); qw(src.sk, dst.sk); qw(src.sv, dst.sv);
  qw(src.so, dst.so);
  one(src.sqn, dst.sqn); one(src.skn, dst.skn);
  qw(src.iff_gate, dst.iff_gate); qw(src.iff_up, dst.iff_up);
  qw(src.iff_down, dst.iff_down); qw(src.iff_gu, dst.iff_gu);
  qw(src.tff_gate, dst.tff_gate); qw(src.tff_up, dst.tff_up);
  qw(src.tff_down, dst.tff_down); qw(src.tff_gu, dst.tff_gu);
  qw(src.mi1, dst.mi1); qw(src.mi2, dst.mi2); qw(src.mi3, dst.mi3);
  qw(src.mt1, dst.mt1); qw(src.mt2, dst.mt2);
  one(src.mi1_b, dst.mi1_b); one(src.mi2_b, dst.mi2_b);
  one(src.mi3_b, dst.mi3_b); one(src.mt1_b, dst.mt1_b);
  one(src.mt2_b, dst.mt2_b);
  one(src.ni1, dst.ni1); one(src.ni2, dst.ni2); one(src.ni3, dst.ni3);
  one(src.nt1, dst.nt1); one(src.nt2, dst.nt2);
  one(src.i_attn_n, dst.i_attn_n); one(src.i_self_n, dst.i_self_n);
  one(src.i_ffn1, dst.i_ffn1); one(src.i_ffn2, dst.i_ffn2);
  one(src.t_attn_n, dst.t_attn_n); one(src.t_ffn1, dst.t_ffn1);
  one(src.t_ffn2, dst.t_ffn2);
  if (!ok) { dst = DoubleBlock{}; }
  return ok;
}

metal_compute::SharedBuffer
MetalBooguTransformer::rebuild_one_(const std::string& nm,
                                        vpipe::genai::Placement how)
{
  if (!_ws) { return {}; }
  if (how == P::kRaw) {
    // The same residency load_qw_ uses for a STREAMED read, which is
    // Copied whenever the model streams (weights_may_be_mapped is false
    // then) -- so this never hands back a read-only shard view.
    const auto res = _mmap_weights ? WeightSet::Residency::Mapped
                                   : WeightSet::Residency::Copied;
    return _ws->stream_tensor(nm, _mc, res);
  }
  return bf16_(*_ws, nm, Retain::Streamed);
}

void
MetalBooguTransformer::configure_slots_()
{
  {
    BlockSlots<Block>::Ops o;
    o.each = [this](int L, Block& b,
                    const BlockSlots<Block>::TensorFn& fn) {
      each_single_tensor_(L, b, fn);
    };
    o.rebuild_one = [this](const std::string& nm,
                           vpipe::genai::Placement how) {
      return rebuild_one_(nm, how);
    };
    o.build = [this](int L, Block& b) {
      return _ws && load_block_(*_ws,
                                "single_stream_layers." + std::to_string(L) +
                                    ".",
                                b, /*modulated=*/true, Retain::Streamed);
    };
    o.clone = [this](const Block& s, Block& d, bool copy) {
      return clone_single_(s, d, copy);
    };
    o.bytes = [](const Block& b) { return single_bytes_(b); };
    o.empty = [](const Block& b) { return b.q.empty(); };
    o.post_refill = [this](int, Block& b) {
      return weave_into_(b.ff_gate, b.ff_up, b.ff_gu);
    };
    _single_slots.set_weight_set(_ws.get());
    _single_slots.configure(_mc, std::move(o),
                            "MetalBooguTransformer(single)",
                            "VPIPE_BOOGU_NO_SLOTS");
  }
  {
    BlockSlots<DoubleBlock>::Ops o;
    o.each = [this](int L, DoubleBlock& b,
                    const BlockSlots<DoubleBlock>::TensorFn& fn) {
      each_double_tensor_(L, b, fn);
    };
    o.rebuild_one = [this](const std::string& nm,
                           vpipe::genai::Placement how) {
      return rebuild_one_(nm, how);
    };
    o.build = [this](int L, DoubleBlock& b) {
      return _ws && load_double_(*_ws,
                                 "double_stream_layers." + std::to_string(L) +
                                     ".",
                                 b, Retain::Streamed);
    };
    o.clone = [this](const DoubleBlock& s, DoubleBlock& d, bool copy) {
      return clone_double_(s, d, copy);
    };
    o.bytes = [](const DoubleBlock& b) { return double_bytes_(b); };
    o.empty = [](const DoubleBlock& b) { return b.jq_i.empty(); };
    o.post_refill = [this](int, DoubleBlock& b) {
      return weave_into_(b.iff_gate, b.iff_up, b.iff_gu) &&
             weave_into_(b.tff_gate, b.tff_up, b.tff_gu);
    };
    _double_slots.set_weight_set(_ws.get());
    _double_slots.configure(_mc, std::move(o),
                            "MetalBooguTransformer(double)",
                            "VPIPE_BOOGU_NO_SLOTS");
  }
}

MetalBooguTransformer::~MetalBooguTransformer()
{
  // GIVE THE POOL BACK. Freeing a wired buffer unwires it in the kernel,
  // so the machine recovers either way -- but the pool's own counter
  // would not, and a DiT freed for the vae-decode and reloaded on the
  // next prompt (free_boogu_dit_for_decode_) would leak its whole share
  // of the budget per prompt until nothing could wire at all.
  if (_wire.on()) {
    wire_fixed_(false);
    for (DoubleBlock& b : _double) { wire_block_(b, false); }
    for (Block& b : _single) { wire_block_(b, false); }
  }
}

std::unique_ptr<MetalBooguTransformer>
MetalBooguTransformer::load(const std::string& model_dir, MetalCompute* mc,
                            const Config& cfg, bool stream_blocks)
{
  return load(WeightSet::open(model_dir, nullptr), mc, cfg, stream_blocks);
}

std::unique_ptr<MetalBooguTransformer>
MetalBooguTransformer::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                            const Config& cfg, bool stream_blocks)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  const std::string model_dir = ws_in->dir();

  auto m = std::unique_ptr<MetalBooguTransformer>(new MetalBooguTransformer());
  m->_ws = std::move(ws_in);
  m->_mc = mc;
  m->_cfg = cfg;
  m->_stream_blocks = stream_blocks;
  // BEFORE the first weight is read, because it decides how they are read:
  // a mapped view can be neither mlock'd nor parked. See
  // shared/wired-pool.h.
  m->_wire.open(mc);
  // The reusable read destinations. Configured whether or not this run
  // streams: they cost nothing until the first streamed block asks for
  // one, and a model that preloads never gets there.
  m->configure_slots_();
  m->_mmap_weights = weights_may_be_mapped(stream_blocks, m->_wire.on()) &&
                     std::getenv("VPIPE_BOOGU_NO_MMAP_WEIGHTS") == nullptr;
  WeightSet& ws = *m->_ws;
  // Everything loaded from here to the end of load() is RETAINED for the
  // model's life. The streamed blocks are read in forward(), and only
  // there.
  const Retain r = Retain::Cached;

  {
    namespace fs = std::filesystem;
    std::ifstream in(fs::path(model_dir) / "config.json");
    if (in) {
      FlexData cfgj = FlexData::from_json(in);
      if (cfgj.is_object()) {
        auto obj = cfgj.as_object();
        // Structural dims from config.json so one code path drives every
        // BooguImageTransformer2DModel size. Absent keys keep the default.
        auto geti = [&](const char* k, int cur) -> int {
          return obj.contains(k) ? (int)obj.at(k).as_int(cur) : cur;
        };
        auto getf = [&](const char* k, float cur) -> float {
          return obj.contains(k) ? (float)obj.at(k).as_real(cur) : cur;
        };
        m->_cfg.hidden      = geti("hidden_size", m->_cfg.hidden);
        m->_cfg.n_heads     = geti("num_attention_heads", m->_cfg.n_heads);
        m->_cfg.n_kv_heads  = geti("num_kv_heads", m->_cfg.n_kv_heads);
        m->_cfg.in_channels = geti("in_channels", m->_cfg.in_channels);
        m->_cfg.patch       = geti("patch_size", m->_cfg.patch);
        m->_cfg.n_refiner   = geti("num_refiner_layers", m->_cfg.n_refiner);
        m->_cfg.multiple_of = geti("multiple_of", m->_cfg.multiple_of);
        m->_cfg.norm_eps    = getf("norm_eps", m->_cfg.norm_eps);
        m->_cfg.timestep_scale =
            getf("timestep_scale", m->_cfg.timestep_scale);
        const int n_layers = geti("num_layers", m->_cfg.n_double +
                                                    m->_cfg.n_single);
        m->_cfg.n_double =
            geti("num_double_stream_layers", m->_cfg.n_double);
        m->_cfg.n_single = n_layers - m->_cfg.n_double;
        if (m->_cfg.n_heads > 0) {
          m->_cfg.head_dim = m->_cfg.hidden / m->_cfg.n_heads;
        }
        m->_cfg.temb_dim =
            m->_cfg.hidden < 1024 ? m->_cfg.hidden : 1024;   // min(hidden,1024)
        if (obj.contains("axes_dim_rope")) {
          FlexData ax = obj.at("axes_dim_rope");
          if (ax.is_array()) {
            auto av = ax.as_array();
            for (int i = 0; i < 3 && i < (int)av.size(); ++i) {
              m->_cfg.axes_dim[i] = (int)av[i].as_int(m->_cfg.axes_dim[i]);
            }
          }
        }
        if (obj.contains("instruction_feature_configs")) {
          FlexData ifc = obj.at("instruction_feature_configs");
          if (ifc.is_object()) {
            auto io = ifc.as_object();
            // reduce_type "mean" keeps the feature dim; "concat" multiplies it
            // by the number of tapped layers (the stage does the reduction, so
            // this is only the width the caption_embedder expects).
            int dim = io.contains("instruction_feat_dim")
                          ? (int)io.at("instruction_feat_dim").as_int(
                                m->_cfg.instruct_dim)
                          : m->_cfg.instruct_dim;
            const std::string red =
                io.contains("reduce_type")
                    ? std::string(io.at("reduce_type").as_string("mean"))
                    : std::string("mean");
            const int taps =
                io.contains("num_instruction_feature_layers")
                    ? (int)io.at("num_instruction_feature_layers").as_int(1)
                    : 1;
            if (red.find("cat") != std::string::npos && taps > 1) {
              dim *= taps;
            }
            m->_cfg.instruct_dim = dim;
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
  if (m->_cfg.hidden <= 0 || m->_cfg.n_heads <= 0 || m->_cfg.n_kv_heads <= 0 ||
      m->_cfg.head_dim <= 0 || m->_cfg.n_heads % m->_cfg.n_kv_heads != 0 ||
      m->_cfg.n_single < 0) {
    return nullptr;
  }
  {
    int axsum = 0;
    for (int i = 0; i < 3; ++i) { axsum += m->_cfg.axes_dim[i]; }
    if (axsum != m->_cfg.head_dim) {
      if (mc->session() != nullptr) {
        mc->session()->warn(fmt(
            "MetalBooguTransformer: sum(axes_dim_rope)={} != head_dim={}",
            axsum, m->_cfg.head_dim));
      }
      return nullptr;
    }
  }

  // bf16 metallibs (VPIPE_ELT=bfloat); entry-point names keep the "_f16" label.
  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_lib_sdpa = mc->load_library("sdpa_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_fn_gemm          = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_gemm_bm64     = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_gemm_bm64bn64 = m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  m->_fn_gemm_bm64_a16 = m->_lib_gemm.function("dense_gemm_t_bm64_acc16_f16");
  m->_fn_ff_swiglu     = m->_lib_gemm.function("dense_gemm_swiglu_bm64_f16");
  m->_fn_ff_swiglu_a16 =
      m->_lib_gemm.function("dense_gemm_swiglu_bm64_acc16_f16");
  m->_fn_gemm_bias  = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_rms        = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_swiglu     = m->_lib_elt.function("swiglu_f16");
  m->_fn_residual   = m->_lib_elt.function("residual_add_f16");
  m->_fn_transpose  = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_sdpa       = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_layernorm  = m->_lib_elt.function("layer_norm_plain_f16");
  m->_fn_rope_table = m->_lib_rope.function("rope_pair_table_ftab_f16");
  m->_fn_transpose_rope =
      m->_lib_rope.function("transpose_rope_pair_ftab_f16");
  m->_fn_adaln      = m->_lib_elt.function("adaln_modulate_f16");
  m->_fn_gated      = m->_lib_elt.function("gated_residual_f16");
  m->_fn_gated_tanh = m->_lib_elt.function("gated_residual_tanh_f16");
  if (std::getenv("VPIPE_BOOGU_NO_ELT_V4") == nullptr) {
    m->_fn_adaln4      = m->_lib_elt.function("adaln_modulate_v4_f16");
    m->_fn_gated_tanh4 = m->_lib_elt.function("gated_residual_tanh_v4_f16");
    m->_fn_residual4   = m->_lib_elt.function("residual_add_v4_f16");
  }
  m->_fn_bias_add   = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_headslice  = m->_lib_elt.function("head_slice_f16");
  m->_fn_mulsig     = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_colabsmax  = m->_lib_elt.function("col_absmax_f16");   // AWQ tap
  // Padded-head-dim twins for the steel bd128 attention route.
  m->_fn_tr_pad      = m->_lib_elt.function("transpose_abd_pad_f16");
  m->_fn_tr_unpad    = m->_lib_elt.function("transpose_abd_unpad_f16");
  m->_fn_tr_rope_pad =
      m->_lib_rope.function("transpose_rope_pair_ftab_pad_f16");
  if (!m->_fn_gemm.valid() || !m->_fn_rms.valid() || !m->_fn_swiglu.valid() ||
      !m->_fn_residual.valid() || !m->_fn_transpose.valid() ||
      !m->_fn_sdpa.valid() || !m->_fn_layernorm.valid() ||
      !m->_fn_rope_table.valid() || !m->_fn_adaln.valid() ||
      !m->_fn_gated.valid() || !m->_fn_gated_tanh.valid() ||
      !m->_fn_bias_add.valid() || !m->_fn_headslice.valid() ||
      !m->_fn_mulsig.valid() || !m->_fn_colabsmax.valid()) {
    return nullptr;
  }
  if (m->_quant_bits > 0) {
    m->_lib_qmm = mc->load_library("affine_qmm_steel_bf16");
    const std::string g = "g" + std::to_string(m->_quant_group);
    m->_fn_qmm4 = m->_lib_qmm.function("affine_qmm_steel_w4" + g);
    m->_fn_qmm8 = m->_lib_qmm.function("affine_qmm_steel_w8" + g);
    if (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid()) { return nullptr; }
    m->_fn_qmm4_bm128 = m->_lib_qmm.function("affine_qmm_steel_w4g64_bm128");
    m->_fn_qmm8_bm128 = m->_lib_qmm.function("affine_qmm_steel_w8g64_bm128");
    m->_qmm_tile = (m->_quant_group == 64 && m->_fn_qmm4_bm128.valid()
                    && m->_fn_qmm8_bm128.valid()) ? 1 : 0;
    if (const char* t = std::getenv("VPIPE_BOOGU_QMM_TILE")) {
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
  }
  // M5 matrix-core matmul2d for the block/projection GEMMs.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_BOOGU_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    if (std::getenv("VPIPE_BOOGU_NO_TN2") == nullptr) {
      m->_fn_dense_mma_tn2 =
          m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");
    }
    if (std::getenv("VPIPE_BOOGU_NO_SPLITK") == nullptr) {
      m->_fn_dense_mma_splitk =
          m->_lib_dense_mma.function(
              "dense_gemm_mma_splitk_n128x256_k6784_f16");
      m->_use_splitk = m->_fn_dense_mma_splitk.valid()
                       && m->_fn_residual.valid();
    }
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid();
    m->set_gemm_route(true, true);   // live route = whatever the env allowed
    if (m->_use_mma2 && m->_quant_bits > 0) {
      m->_lib_dequant = mc->load_library("affine_dequant_bf16");
      const std::string dg = "g" + std::to_string(m->_quant_group);
      m->_fn_dequant4 = m->_lib_dequant.function("affine_dequant_w4" + dg);
      m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant_w8" + dg);
      if (!m->_fn_dequant4.valid() || !m->_fn_dequant8.valid()) {
        m->_use_mma2 = false;   // fall back to steel qmm
      }
    }
    if (const char* e = std::getenv("VPIPE_BOOGU_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
  }
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm, /*bf16=*/true);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }
  // Fuse the SwiGLU FF (default on EXCEPT on the matmul2d path, where the
  // fused GEMM's register-local epilogue would keep the FF off the matrix
  // cores). VPIPE_BOOGU_FUSE_FF=1 forces it back on there;
  // VPIPE_BOOGU_NO_FUSE_FF disables it entirely.
  m->_ff_acc16 = std::getenv("VPIPE_BOOGU_FF_ACC16") != nullptr;
  m->_fuse_ff = std::getenv("VPIPE_BOOGU_NO_FUSE_FF") == nullptr
                && (!m->_use_mma2
                    || std::getenv("VPIPE_BOOGU_FUSE_FF") != nullptr)
                && m->_fn_ff_swiglu.valid() && m->_fn_ff_swiglu_a16.valid()
                && (m->_quant_bits == 0
                    || (m->_quant_group == 64
                        && m->_fn_qmm_swiglu4_bm64.valid()
                        && m->_fn_qmm_swiglu8_bm64.valid()));
  // Steel flash attention over ZERO-PADDED 128-dim heads (the model's head_dim
  // is 120, which no flash kernel supports natively). Needs the pad/unpad
  // twins; otherwise the scalar sdpa_full_f16 carries the attention.
  m->_lib_attn = mc->load_library("attn_steel");
  m->_steel_attn_ok = m->_lib_attn.valid()
                      && m->_fn_tr_pad.valid() && m->_fn_tr_unpad.valid()
                      && m->_fn_tr_rope_pad.valid()
                      && m->_cfg.head_dim <= kPadAttn
                      && std::getenv("VPIPE_BOOGU_NO_STEEL_ATTN") == nullptr;
  // NATIVE head-dim steel attention (attn_steel.metal instantiates the steel
  // template at bd120, and it is numerically identical -- 0.01993 vs the scalar
  // sdpa, the same as the padded route). OFF BY DEFAULT: it is MEASURABLY
  // SLOWER despite doing 6.7% less arithmetic and skipping the pad/unpad
  // transposes -- 0.985x at seq 2271 and 0.965x at seq 8415 (alternating A/B in
  // one process, boogu_perf.attention_native_vs_padded; an end-to-end A/B
  // cannot see it, the run-to-run spread at 1024px is ~4%). The pad buys
  // 16-byte-aligned block loads (vec_size 32 at 64-byte strides vs 30 at
  // 60-byte ones) and MLX's BD==128 simdgroup scheduling hints, and that is
  // worth more than the arithmetic saved. Kept opt-in (VPIPE_BOOGU_ATTN_NATIVE)
  // so a future kernel or GPU can be re-measured cheaply.
  if (m->_steel_attn_ok && m->_cfg.head_dim != kPadAttn &&
      std::getenv("VPIPE_BOOGU_ATTN_NATIVE") != nullptr) {
    // The steel attention is specialized by function constants, so probing it
    // needs the SAME specialized form the forward will use -- asking for the
    // bare function builds an invalid pipeline state (a hard Metal assertion,
    // not a null return). Any constant set will do for the probe; the PSO cache
    // makes it free when the forward asks again.
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, true).set_bool(201, true)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false);
    m->_attn_native =
        m->_lib_attn
            .function("attn_steel_h_bd" + std::to_string(m->_cfg.head_dim)
                          + "_bf16", fc)
            .valid();
  }
  m->_lib_attn_nax = mc->load_library("attn_steel_nax");
  m->_use_attn_nax = m->_steel_attn_ok && mc->supports_matrix_cores()
                     && m->_lib_attn_nax.valid()
                     && std::getenv("VPIPE_BOOGU_NO_ATTN_NAX") == nullptr;
  m->_gemm_tile = 0;
  if (const char* t = std::getenv("VPIPE_BOOGU_GEMM_TILE")) {
    m->_gemm_tile = std::atoi(t);
  }
  m->_acc16 = std::getenv("VPIPE_BOOGU_GEMM_ACC16") != nullptr;
  if (m->_gemm_tile == 1 && !m->_fn_gemm_bm64.valid()) { m->_gemm_tile = 0; }
  if (m->_gemm_tile == 2 && !m->_fn_gemm_bm64bn64.valid()) {
    m->_gemm_tile = m->_fn_gemm_bm64.valid() ? 1 : 0;
  }

  // ---- embedders / conditioning / final layer ----
  m->_x_embed    = m->load_qw_(ws, "x_embedder", r);
  m->_x_embed_b  = m->bf16_(ws, "x_embedder.bias", r);
  m->_ref_embed  = m->load_qw_(ws, "ref_image_patch_embedder", r);
  m->_ref_embed_b = m->bf16_(ws, "ref_image_patch_embedder.bias", r);
  m->_img_index  = m->bf16_(ws, "image_index_embedding", r);
  m->_t_emb1 =
      m->load_qw_(ws, "time_caption_embed.timestep_embedder.linear_1", r);
  m->_t_emb1_b =
      m->bf16_(ws, "time_caption_embed.timestep_embedder.linear_1.bias", r);
  m->_t_emb2 =
      m->load_qw_(ws, "time_caption_embed.timestep_embedder.linear_2", r);
  m->_t_emb2_b =
      m->bf16_(ws, "time_caption_embed.timestep_embedder.linear_2.bias", r);
  m->_cap_norm =
      m->bf16_(ws, "time_caption_embed.caption_embedder.0.weight", r);
  m->_cap_lin  = m->load_qw_(ws, "time_caption_embed.caption_embedder.1", r);
  m->_cap_lin_b =
      m->bf16_(ws, "time_caption_embed.caption_embedder.1.bias", r);
  m->_out_lin1   = m->load_qw_(ws, "norm_out.linear_1", r);
  m->_out_lin1_b = m->bf16_(ws, "norm_out.linear_1.bias", r);
  m->_out_lin2   = m->load_qw_(ws, "norm_out.linear_2", r);
  m->_out_lin2_b = m->bf16_(ws, "norm_out.linear_2.bias", r);
  if (m->_x_embed.empty() || m->_ref_embed.empty() || m->_t_emb1.empty() ||
      m->_t_emb2.empty() || m->_cap_norm.empty() || m->_cap_lin.empty() ||
      m->_out_lin1.empty() || m->_out_lin2.empty() || m->_img_index.empty()) {
    return nullptr;
  }
  if (m->_cfg.out_channels == 0) {
    m->_cfg.out_channels =
        m->_out_lin2.n > 0 ? m->_out_lin2.n : m->_cfg.x_in();
  }

  // ---- refiner stacks (always resident: 6 small blocks) ----
  auto load_stack = [&](const char* name, std::vector<Block>& v,
                        bool modulated) -> bool {
    v.resize((std::size_t)m->_cfg.n_refiner);
    for (int i = 0; i < m->_cfg.n_refiner; ++i) {
      if (!m->load_block_(ws, std::string(name) + "." + std::to_string(i) + ".",
                          v[(std::size_t)i], modulated, r)) {
        if (mc->session() != nullptr) {
          mc->session()->warn(fmt(
              "MetalBooguTransformer: failed to load {} block {}", name, i));
        }
        return false;
      }
    }
    return true;
  };
  if (!load_stack("context_refiner", m->_ctx_refiner, /*modulated=*/false) ||
      !load_stack("noise_refiner", m->_noise_refiner, /*modulated=*/true) ||
      !load_stack("ref_image_refiner", m->_ref_refiner, /*modulated=*/true)) {
    return nullptr;
  }

  if (!stream_blocks) {
    m->_double.resize((std::size_t)m->_cfg.n_double);
    for (int i = 0; i < m->_cfg.n_double; ++i) {
      if (!m->load_double_(ws,
                           "double_stream_layers." + std::to_string(i) + ".",
                           m->_double[(std::size_t)i], r)) {
        if (mc->session() != nullptr) {
          mc->session()->warn(fmt(
              "MetalBooguTransformer: failed to load double block {}", i));
        }
        return nullptr;
      }
    }
    m->_single.resize((std::size_t)m->_cfg.n_single);
    for (int i = 0; i < m->_cfg.n_single; ++i) {
      if (!m->load_block_(ws,
                          "single_stream_layers." + std::to_string(i) + ".",
                          m->_single[(std::size_t)i], /*modulated=*/true,
                          r)) {
        if (mc->session() != nullptr) {
          mc->session()->warn(fmt(
              "MetalBooguTransformer: failed to load single block {}", i));
        }
        return nullptr;
      }
    }
  } else {
    // Streaming: the double + single blocks load JIT in forward_dit. The FF
    // inner width is already known (the refiners share it).
    if (m->_cfg.ff_inner == 0) {
      const auto* fi =
          ws.src().info("single_stream_layers.0.feed_forward.linear_1.weight");
      if (fi != nullptr && !fi->shape.empty()) {
        m->_cfg.ff_inner = (int)fi->shape[0];
      }
    }
    if (m->_cfg.ff_inner == 0) {
      if (mc->session() != nullptr) {
        mc->session()->warn(fmt(
            "MetalBooguTransformer: streaming -- could not derive the FF dim"));
      }
      return nullptr;
    }
    // Streaming preloads NOTHING. Both stacks are sized to FULL depth
    // all the same: the empty slots are where forward() promotes
    // streamed blocks as free memory allows. An unfilled slot reads as
    // empty, which is what `held` tests.
    m->_double.resize((std::size_t)m->_cfg.n_double);
    m->_single.resize((std::size_t)m->_cfg.n_single);
    if (mc->session() != nullptr) {
      mc->session()->info(fmt(
          "MetalBooguTransformer: streaming {}+{} blocks (memory-bounded)",
          m->_cfg.n_double, m->_cfg.n_single));
    }
  }

  // A [hidden] run of zeros: Lumina's RMSNormZero modulates by (1+scale) with
  // NO shift, so adaln_modulate_f16 gets this as its shift operand.
  m->_zero_h = mc->make_shared_buffer((std::size_t)m->_cfg.hidden * 2);
  if (m->_zero_h.empty()) { return nullptr; }
  std::memset(m->_zero_h.contents(), 0, m->_zero_h.byte_size());
  return m;
}

// Build the joint [text; refs...; target] 3-axis RoPE cos/sin tables
// [seq, head_dim]. Text token l sits at (l,l,l); each image segment shares the
// axis-0 band `t_off` and tiles its rows x cols patch grid row-major on axes
// 1/2. Adjacent-pair layout matches rope_pair_table_ftab_f16, which is exactly
// Lumina's `apply_rotary_emb(use_real=False)` complex-pair rotation.
void
MetalBooguTransformer::build_rope_tables_(int text_seq,
                                          const std::vector<ImgSeg>& segs,
                                          SharedBuffer& cos_out,
                                          SharedBuffer& sin_out)
{
  int img_seq = 0;
  for (const auto& sg : segs) { img_seq += sg.seq; }
  const int seq = text_seq + img_seq;
  const int HD = _cfg.head_dim;
  const int pairs = HD / 2;
  // f32 tables (only x is bf16): the rotation error is STRUCTURED, so bf16
  // table rounding compounds over 46 blocks and, worse, over denoise steps.
  cos_out = _mc->make_shared_buffer((std::size_t)seq * HD * sizeof(float));
  sin_out = _mc->make_shared_buffer((std::size_t)seq * HD * sizeof(float));
  if (cos_out.empty() || sin_out.empty()) { return; }
  auto* c = static_cast<float*>(cos_out.contents());
  auto* s = static_cast<float*>(sin_out.contents());
  const double theta = (double)_cfg.rope_theta;
  // Per-pair inverse frequency + which of the 3 position axes it reads --
  // functions of (axis, j) only, so precompute once. Angles stay f64 to match
  // diffusers' get_1d_rotary_pos_embed (float64 freqs).
  std::vector<double> pair_freq((std::size_t)pairs);
  std::vector<int> pair_axis((std::size_t)pairs);
  {
    int pair = 0;
    for (int a = 0; a < 3; ++a) {
      const int adim = _cfg.axes_dim[a];
      const int apairs = adim / 2;
      for (int j = 0; j < apairs && pair < pairs; ++j, ++pair) {
        pair_freq[(std::size_t)pair] =
            1.0 / std::pow(theta, (double)(2 * j) / (double)adim);
        pair_axis[(std::size_t)pair] = a;
      }
    }
  }
  auto emit = [&](int t, double p0, double p1, double p2) {
    const double pos[3] = {p0, p1, p2};
    for (int pair = 0; pair < pairs; ++pair) {
      const double ang =
          pos[pair_axis[(std::size_t)pair]] * pair_freq[(std::size_t)pair];
      const float cb = (float)std::cos(ang);
      const float sb = (float)std::sin(ang);
      c[(std::size_t)t * HD + 2 * pair]     = cb;
      c[(std::size_t)t * HD + 2 * pair + 1] = cb;
      s[(std::size_t)t * HD + 2 * pair]     = sb;
      s[(std::size_t)t * HD + 2 * pair + 1] = sb;
    }
  };
  int t = 0;
  for (; t < text_seq; ++t) { emit(t, t, t, t); }   // text: (l, l, l)
  for (const auto& sg : segs) {
    for (int p = 0; p < sg.seq; ++p, ++t) {
      emit(t, (double)sg.t_off, (double)(p / sg.cols), (double)(p % sg.cols));
    }
  }
}

void
MetalBooguTransformer::calib_begin()
{
  _calib_acc.clear();
  const int H = _cfg.hidden;
  const int FFI = _cfg.ff_inner;
  const int nR = _cfg.n_refiner, nD = _cfg.n_double, nS = _cfg.n_single;
  auto add = [&](const char* g, int rows, int dim) {
    if (rows <= 0 || dim <= 0) { return; }
    SharedBuffer b = _mc->make_shared_buffer((std::size_t)rows * dim * 2);
    if (!b.empty()) { std::memset(b.contents(), 0, b.byte_size()); }
    _calib_acc[g] = std::move(b);
  };
  add("ctx_attn", nR, H);      add("ctx_ffn", nR, H);
  add("ctx_ffact", nR, FFI);
  add("noise_attn", nR, H);    add("noise_ffn", nR, H);
  add("noise_ffact", nR, FFI);
  add("ref_attn", nR, H);      add("ref_ffn", nR, H);
  add("ref_ffact", nR, FFI);
  add("dbl_jattn_img", nD, H); add("dbl_jattn_txt", nD, H);
  add("dbl_sattn_img", nD, H); add("dbl_jout", nD, H);
  add("dbl_ffn_img", nD, H);   add("dbl_ffn_txt", nD, H);
  add("dbl_ffact_img", nD, FFI); add("dbl_ffact_txt", nD, FFI);
  add("sgl_attn", nS, H);      add("sgl_ffn", nS, H);
  add("sgl_ffact", nS, FFI);
  add("emb_x", 1, _cfg.x_in());
  add("emb_ref", 1, _cfg.x_in());
  add("emb_ctx", 1, _cfg.instruct_dim);
  add("emb_proj", 1, H);
  _calib_on = true;
}

std::map<std::string, std::vector<float>>
MetalBooguTransformer::calib_stats() const
{
  std::map<std::string, std::vector<float>> out;
  for (const auto& kv : _calib_acc) {
    const std::size_t n = kv.second.empty() ? 0 : kv.second.byte_size() / 2;
    std::vector<float> v(n);
    const auto* s = static_cast<const std::uint16_t*>(kv.second.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = bf16_to_f32_(s[i]); }
    out[kv.first] = std::move(v);
  }
  return out;
}

bool
MetalBooguTransformer::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                                 std::size_t xe, const QWeight& w,
                                 const SharedBuffer& y, std::size_t ye,
                                 int M, int N, int K)
{
  if (!_use_mma2 || M < _mma_min_m || N < 16) { return false; }
  const SharedBuffer* wdense;
  if (w.quantized) {
    const ComputeFunction& dq = (w.bits == 8) ? _fn_dequant8 : _fn_dequant4;
    if (!dq.valid()) { return false; }
    const std::size_t need = (std::size_t)N * K * 2;
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
  if (_i8 && _i8->gemm(enc, x, xe, *wdense, y, ye, M, N, K)) { return true; }
  // Split-K for the very deep ff-down contraction (K = ff_inner = 13568): each
  // of the 2 KC=6784 chunks gets its own threadgroup plane (grid.z), which
  // multiplies the threadgroup count and shortens each serial reduction back
  // into the fast regime, then a residual_add folds the planes. 1.14x at
  // M=1032, 1.16x at M=2271, 1.17x at M=4104 over the unsplit 128x256, and it
  // beats the TN=2 tile (which is bit-exact but 1.08-1.13x) at every M.
  if (_splitk_on && K == 2 * kSplitKC) {
    const std::size_t plane = (std::size_t)M * N;
    const std::size_t need = plane * 2 * 2;
    if (_splitk.empty() || _splitk.byte_size() < need) {
      _splitk = _mc->make_shared_buffer(need);
    }
    if (!_splitk.empty()) {
      enc.set_function(_fn_dense_mma_splitk);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, *wdense);
      enc.set_buffer(2, _splitk);
      enc.set_constant(3, K); enc.set_constant(4, N); enc.set_constant(5, M);
      enc.dispatch({(unsigned)(((N + 255) / 256) * 256),
                    (unsigned)((M + 127) / 128), 2u}, {256, 1, 1});
      enc.set_function(_fn_residual);
      enc.set_buffer(0, _splitk, 0);
      enc.set_buffer(1, _splitk, plane * 2);
      enc.set_buffer(2, y, ye * 2);
      enc.set_constant(3, (int)plane);
      enc.dispatch({(unsigned)plane, 1, 1}, {256, 1, 1});
      return true;
    }
  }
  int RN = 256;   // effective N-region per threadgroup
  const ComputeFunction* fn = &_fn_dense_mma_deep;
  if (_tn2_on && M >= kTn2MinM && N >= kTn2MinN) {
    // Enough threadgroups to keep the GPU busy at half the tg count, and a wide
    // enough N to have two tiles to reuse x across: the TN=2 region wins.
    fn = &_fn_dense_mma_tn2; RN = 512;
  } else if (K < 6144) {
    fn = &_fn_dense_mma; RN = 128;
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

SharedBuffer
MetalBooguTransformer::forward_dit(const SharedBuffer& instruct, int instr_seq,
                                   const SharedBuffer& latents, int img_seq,
                                   int grid_h, int grid_w, float timestep,
                                   const std::vector<RefImage>& refs)
{
  const Config& c = _cfg;
  // JOIN ANY OUTSTANDING READ ON EVERY EXIT. A prefetch may be filling a
  // slot, and this function returns from a dozen places -- a stop
  // request, a GPU error, an allocation that failed. Freeing or reading
  // a slot while a reader thread writes into it is a use-after-free, and
  // a scope guard is the only version of this that cannot be forgotten
  // at the thirteenth return.
  struct SlotJoin {
    MetalBooguTransformer* t;
    ~SlotJoin()
    {
      t->_single_slots.join();
      t->_double_slots.join();
    }
  } slot_join{this};
  _single_slots.begin_forward();
  _double_slots.begin_forward();
  const int H = c.hidden, HED = c.n_heads, KVH = c.n_kv_heads;
  const int HD = c.head_dim, KD = KVH * HD;
  const int XIN = c.x_in(), OC = c.out_channels, FFI = c.ff_inner;
  const int TE = c.temb_dim, FD = c.freq_dim, P = c.patch;
  const float eps = c.norm_eps;
  const int TS = instr_seq, IS_GEN = img_seq;
  int IS_REF = 0;
  for (const auto& r : refs) { IS_REF += r.seq; }
  const int IS = IS_REF + IS_GEN, seq = TS + IS;
  if (TS <= 0 || IS_GEN <= 0 || FFI <= 0 || grid_w < P || grid_h < P ||
      instruct.byte_size() < (std::size_t)TS * c.instruct_dim * 2 ||
      latents.byte_size() < (std::size_t)IS_GEN * XIN * 2) {
    return {};
  }
  if ((grid_h / P) * (grid_w / P) != IS_GEN) { return {}; }
  if ((int)refs.size() > c.max_ref_images) { return {}; }
  for (const auto& r : refs) {
    if (r.seq <= 0 || r.grid_h < P || r.grid_w < P ||
        (r.grid_h / P) * (r.grid_w / P) != r.seq ||
        r.latents.byte_size() < (std::size_t)r.seq * XIN * 2) {
      return {};
    }
  }
  const float scale = 1.0f / std::sqrt((float)HD);

  // LLM-lane perf event (perf-visualizer): one DiT forward per sampler step,
  // value = the joint sequence length. Mirrors the FLUX.2 / Krea-2 DiT event.
  PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDit,
                     kPerfLlmDitBegin, (std::uint64_t)seq);

  // Per-section GPU timing (VPIPE_BOOGU_DIT_PROFILE), mirroring the FLUX.2
  // klein DiT profile: marks + commit-boundary waits split the deferred
  // streams into timed slices. The barriers serialize (removing any overlap)
  // but the per-section GPU wall time is what we want. No effect unless set.
  const bool prof = std::getenv("VPIPE_BOOGU_DIT_PROFILE") != nullptr;
  double t_setup = 0, t_cond = 0, t_refine = 0, t_dbl_proj = 0, t_dbl_attn = 0,
         t_dbl_ff = 0, t_sgl_proj = 0, t_sgl_attn = 0, t_sgl_ff = 0,
         t_join = 0, t_final = 0;
  // Phase selectors for the run_block/psplit profiling barriers: the
  // projections (qkv + to_out) and the feed-forward are separated from the
  // attention so a slow GEMM tile cannot hide behind a slow attention.
  enum : int { kProfProj = 0, kProfAttn = 1, kProfFf = 2 };
  auto tnow = [] { return std::chrono::steady_clock::now(); };
  auto ms_since = [](std::chrono::steady_clock::time_point m) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - m).count();
  };
  // Timed from HERE so the buckets sum to the whole call: the CPU-side setup
  // (scratch allocation, the RoPE tables, the steel specializations) used to
  // fall outside the report.
  std::chrono::steady_clock::time_point mk = tnow();

  // Every buffer that crosses this interface is bf16 (see the header): the
  // conditioning arrives bf16 straight off the encoder's final norm -- its
  // attention-sink outliers do not survive an f16 round-trip -- and the latents
  // and returned velocity match so nothing on the path narrows. The
  // conditioning is RMSNormed in place by the caption_embedder below, so copy
  // it rather than mutating the caller's buffer.
  SharedBuffer instruct_b =
      _mc->make_shared_buffer((std::size_t)TS * c.instruct_dim * 2);
  if (instruct_b.empty()) { return {}; }
  std::memcpy(instruct_b.contents(), instruct.contents(),
              (std::size_t)TS * c.instruct_dim * 2);
  const SharedBuffer& latents_b = latents;
  std::vector<const SharedBuffer*> refs_b;
  refs_b.reserve(refs.size());
  for (const auto& r : refs) { refs_b.push_back(&r.latents); }

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };

  // ---- RoPE: ONE joint table, sliced by row offset ------------------------
  // Row layout: [0,TS) text | [TS, TS+IS_REF) references | tail = target.
  // The reference advances the axis-0 band by max(rows, cols) per reference
  // (BooguImageRotaryPosEmbed pe_shift), starting from the text length.
  SharedBuffer rcos, rsin;
  {
    std::vector<ImgSeg> segs;
    int band = TS;
    for (const auto& r : refs) {
      const int rr = r.grid_h / P, rc = r.grid_w / P;
      segs.push_back({band, rr, rc, r.seq});
      band += rr > rc ? rr : rc;
    }
    segs.push_back({band, grid_h / P, grid_w / P, IS_GEN});
    build_rope_tables_(TS, segs, rcos, rsin);
  }
  if (rcos.empty() || rsin.empty()) { return {}; }
  // Row offsets into the joint table (in ROWS; the tables are f32 [seq, HD]).
  const int rowText = 0, rowImg = TS, rowGen = TS + IS_REF;

  // ---- attention plumbing -------------------------------------------------
  // head_dim 120 is not a flash-kernel width, so the steel route pads each
  // head to 128 with zeros (exact: zero q/k dims add nothing to a dot product
  // and zero v dims contribute nothing to the output).
  // The nax (matrix-core) twin exists only at bd128, so it implies the padded
  // route; _attn_native is only set when nax is unavailable or opted out.
  const bool nax = _use_attn_nax && _lib_attn_nax.valid() && !_attn_native;
  const int A_BQ = nax ? 64 : 32;
  const int A_BK = nax ? 32 : 16;
  // Head width the attention runs at: the model's own when a native-width steel
  // kernel exists, otherwise zero-padded to the bd128 kernel. DP == HD makes
  // tr_rope / tr_pad / tr_unpad take their unpadded twins.
  const int DP = (_steel_attn_ok && !_attn_native) ? kPadAttn : HD;
  // One steel param block + specialized function per DISTINCT sequence length
  // used in this forward (the refiners, the image self-attention and the joint
  // attention all run at different lengths), memoized in _attn_cfgs so a
  // sampler's repeated steps pay for the specialization once. A missing entry
  // means this length has not been seen yet; a failed build leaves the length
  // absent, and run_attn then takes the scalar sdpa for it.
  bool use_steel = false;
  if (_steel_attn_ok) {
    std::vector<int> lens{TS, IS_GEN, IS, seq};
    for (const auto& r : refs) { lens.push_back(r.seq); }
    bool all_ok = true;
    for (int L : lens) {
      if (L <= 0 || _attn_cfgs.count(L) != 0) { continue; }
      AttnCfg ac;
      ac.params = _mc->make_shared_buffer(sizeof(SteelAttnParams));
      if (ac.params.empty()) { all_ok = false; break; }
      auto* p = static_cast<SteelAttnParams*>(ac.params.contents());
      p->B = 1; p->H = HED; p->D = DP; p->qL = L; p->kL = L;
      p->gqa_factor = HED / KVH; p->scale = scale;
      p->NQ = (L + A_BQ - 1) / A_BQ; p->NK = (L + A_BK - 1) / A_BK;
      p->NQ_aligned = L / A_BQ; p->NK_aligned = L / A_BK;
      p->qL_rem = L - p->NQ_aligned * A_BQ;
      p->kL_rem = L - p->NK_aligned * A_BK;
      p->qL_off = 0;
      p->Q_strides[0] = (std::int64_t)HED * L * DP;
      p->Q_strides[1] = (std::int64_t)L * DP; p->Q_strides[2] = DP;
      p->K_strides[0] = (std::int64_t)KVH * L * DP;
      p->K_strides[1] = (std::int64_t)L * DP; p->K_strides[2] = DP;
      p->V_strides[0] = p->K_strides[0];
      p->V_strides[1] = p->K_strides[1]; p->V_strides[2] = DP;
      p->O_strides[0] = p->Q_strides[0];
      p->O_strides[1] = p->Q_strides[1]; p->O_strides[2] = DP;
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (L % A_BQ) == 0).set_bool(201, (L % A_BK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      const std::string an =
          "attn_steel_h_bd" + std::to_string(DP) + "_bf16";
      ac.fn = nax ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
                  : _lib_attn.function(an, fc);
      ac.nqb = (unsigned)((L + A_BQ - 1) / A_BQ);
      if (!ac.fn.valid()) { all_ok = false; break; }
      _attn_cfgs.emplace(L, std::move(ac));
    }
    use_steel = all_ok;
    for (int L : lens) {
      if (L > 0 && _attn_cfgs.count(L) == 0) { use_steel = false; }
    }
  }

  // ---- scratch ------------------------------------------------------------
  SharedBuffer te_in = buf((std::size_t)FD);
  {
    auto* ti = static_cast<std::uint16_t*>(te_in.contents());
    const int half = FD / 2;
    for (int i = 0; i < half; ++i) {
      const double fr = std::exp(-std::log(1e4) * (double)i / (double)half);
      const double ang = (double)timestep * (double)c.timestep_scale * fr;
      ti[i] = f32_to_bf16_((float)std::cos(ang));         // flip_sin_to_cos
      ti[half + i] = f32_to_bf16_((float)std::sin(ang));
    }
  }
  SharedBuffer temb = buf((std::size_t)TE), tsilu = buf((std::size_t)TE),
               te1 = buf((std::size_t)TE);
  // Five modulation slots (the double block's img_norm1/2/3 + instruct_norm1/2),
  // each 4*H: [scale_msa | gate_msa | scale_mlp | gate_mlp].
  SharedBuffer mods = buf((std::size_t)5 * 4 * H);
  SharedBuffer outmod = buf((std::size_t)H);          // norm_out linear_1
  SharedBuffer txt = buf((std::size_t)TS * H);
  SharedBuffer img = buf((std::size_t)(IS > 0 ? IS : 1) * H);
  SharedBuffer joint = buf((std::size_t)seq * H);
  SharedBuffer nrm = buf((std::size_t)seq * H);
  SharedBuffer nrm2 = buf((std::size_t)seq * H);
  SharedBuffer qb = buf((std::size_t)seq * H), kb = buf((std::size_t)seq * KD),
               vb = buf((std::size_t)seq * KD);
  SharedBuffer qt = buf((std::size_t)HED * seq * DP),
               kt = buf((std::size_t)KVH * seq * DP),
               vt = buf((std::size_t)KVH * seq * DP),
               atb = buf((std::size_t)HED * seq * DP);
  SharedBuffer att = buf((std::size_t)seq * H), ob = buf((std::size_t)seq * H);
  // The separate gate/up landing buffers are only touched on the UNFUSED FF
  // path. Each is seq*ff_inner (228 MB at 1024px), so allocating them when
  // every block has an interleaved gate|up weight is half a gigabyte of dead
  // scratch per step. Streaming blocks are not resident yet, so keep them.
  bool ff_split = !_fuse_ff || _stream_blocks;
  for (const auto& b : _single) { ff_split = ff_split || b.ff_gu.empty(); }
  for (const auto& b : _double) {
    ff_split = ff_split || b.iff_gu.empty() || b.tff_gu.empty();
  }
  for (const auto* v : {&_ctx_refiner, &_noise_refiner, &_ref_refiner}) {
    for (const auto& b : *v) { ff_split = ff_split || b.ff_gu.empty(); }
  }
  SharedBuffer ffg, ffu;
  if (ff_split) {
    ffg = buf((std::size_t)seq * FFI);
    ffu = buf((std::size_t)seq * FFI);
    if (ffg.empty() || ffu.empty()) { return {}; }
  }
  SharedBuffer ffm = buf((std::size_t)seq * FFI);
  SharedBuffer velocity = buf((std::size_t)IS_GEN * OC);
  if (temb.empty() || joint.empty() || qt.empty() || ffm.empty() ||
      velocity.empty()) {
    return {};
  }

  // ---- helper wrappers over a live ComputeEncoder --------------------------
  auto make_ops = [&](ComputeEncoder& enc) {
    struct Ops {
      MetalBooguTransformer* self;
      ComputeEncoder* e;
      SharedBuffer *rcos, *rsin;
      float eps;
      // y[M,N] (elem offset ye) = x[M,K] (elem offset xe) @ W[N,K]^T.
      void gemm(const SharedBuffer& x, const QWeight& w, const SharedBuffer& y,
                std::size_t ye, int M, int N, int K, std::size_t xe = 0) {
        if (self->gemm_mma_(*e, x, xe, w, y, ye, M, N, K)) { return; }
        int bm = 32, bn = 32;
        if (w.quantized) {
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
          const ComputeFunction* f = &self->_fn_gemm;
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
        const unsigned tgz = (bm == 128) ? 4u : 2u;
        e->dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                     (unsigned)(((M + bm - 1) / bm) * 2), tgz}, {32, 2, tgz});
      }
      // Broadcast a per-column vector over M rows of y (no-op when absent).
      // `be` selects a row of a multi-row source -- image_index_embedding is
      // [max_refs, H] and reference i adds its own row.
      void bias(const SharedBuffer& bs, std::size_t be, const SharedBuffer& y,
                std::size_t ye, int M, int N) {
        if (bs.empty()) { return; }
        e->set_function(self->_fn_bias_add);
        e->set_buffer(0, y, ye * 2); e->set_buffer(1, bs, be * 2);
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
              std::size_t ye, int R, int Hd, float e_ps) {
        e->set_function(self->_fn_layernorm);
        e->set_buffer(0, x, xe * 2); e->set_buffer(1, y, ye * 2);
        e->set_constant(2, Hd); e->set_constant(3, e_ps);
        e->dispatch({256, (unsigned)R, 1}, {256, 1, 1});
      }
      // out = (1 + scale) * x + shift, with scale/shift taken from possibly
      // DIFFERENT buffers (the double block's mlp input mixes a scale from
      // img_norm1 with a shift from img_norm2).
      void modulate(const SharedBuffer& x, std::size_t xe,
                    const SharedBuffer& sc, std::size_t sce,
                    const SharedBuffer& sh, std::size_t she,
                    const SharedBuffer& out, std::size_t oe, int N, int total) {
        if ((N % 4) == 0 && self->_fn_adaln4.valid()) {
          e->set_function(self->_fn_adaln4);
          e->set_buffer(0, x, xe * 2); e->set_buffer(1, sc, sce * 2);
          e->set_buffer(2, sh, she * 2); e->set_buffer(3, out, oe * 2);
          e->set_constant(4, N / 4); e->set_constant(5, total / N);
          e->dispatch({(unsigned)(N / 4), (unsigned)(total / N), 1},
                      {256, 1, 1});
          return;
        }
        e->set_function(self->_fn_adaln);
        e->set_buffer(0, x, xe * 2); e->set_buffer(1, sc, sce * 2);
        e->set_buffer(2, sh, she * 2); e->set_buffer(3, out, oe * 2);
        e->set_constant(4, N); e->set_constant(5, total);
        e->dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
      }
      // h += tanh(gate) * sub  (Lumina RMSNormZero gating).
      void gated_tanh(const SharedBuffer& h, std::size_t he,
                      const SharedBuffer& g, std::size_t ge,
                      const SharedBuffer& sub, std::size_t se, int N,
                      int total) {
        if ((N % 4) == 0 && self->_fn_gated_tanh4.valid()) {
          e->set_function(self->_fn_gated_tanh4);
          e->set_buffer(0, h, he * 2); e->set_buffer(1, g, ge * 2);
          e->set_buffer(2, sub, se * 2);
          e->set_constant(3, N / 4); e->set_constant(4, total / N);
          e->dispatch({(unsigned)(N / 4), (unsigned)(total / N), 1},
                      {256, 1, 1});
          return;
        }
        e->set_function(self->_fn_gated_tanh);
        e->set_buffer(0, h, he * 2); e->set_buffer(1, g, ge * 2);
        e->set_buffer(2, sub, se * 2);
        e->set_constant(3, N); e->set_constant(4, total);
        e->dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
      }
      void elt(const ComputeFunction& fn, const SharedBuffer& a, std::size_t ae,
               const SharedBuffer& b, std::size_t be, const SharedBuffer& out,
               std::size_t oe, int nn) {
        if (&fn == &self->_fn_residual && (nn % 4) == 0 &&
            self->_fn_residual4.valid()) {
          e->set_function(self->_fn_residual4);
          e->set_buffer(0, a, ae * 2); e->set_buffer(1, b, be * 2);
          e->set_buffer(2, out, oe * 2); e->set_constant(3, nn / 4);
          e->dispatch({(unsigned)(nn / 4), 1, 1}, {256, 1, 1});
          return;
        }
        e->set_function(fn);
        e->set_buffer(0, a, ae * 2); e->set_buffer(1, b, be * 2);
        e->set_buffer(2, out, oe * 2); e->set_constant(3, nn);
        e->dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
      }
      // Fused SwiGLU FF from an INTERLEAVED [2*inner, K] gate|up weight.
      void swiglu_ff(const SharedBuffer& x, std::size_t xe, const QWeight& w,
                     const SharedBuffer& out, int M, int K, int Nf) {
        const bool a16 = self->_ff_acc16;
        if (w.quantized) {
          e->set_function(w.bits == 8
              ? (a16 ? self->_fn_qmm_swiglu8_bm64_a16
                     : self->_fn_qmm_swiglu8_bm64)
              : (a16 ? self->_fn_qmm_swiglu4_bm64_a16
                     : self->_fn_qmm_swiglu4_bm64));
          e->set_buffer(0, w.codes); e->set_buffer(1, w.scales);
          e->set_buffer(2, w.qbias); e->set_buffer(3, x, xe * 2);
          e->set_buffer(4, out);
          e->set_constant(5, K); e->set_constant(6, Nf); e->set_constant(7, M);
        } else {
          e->set_function(a16 ? self->_fn_ff_swiglu_a16 : self->_fn_ff_swiglu);
          e->set_buffer(0, x, xe * 2); e->set_buffer(1, w.w);
          e->set_buffer(2, out);
          e->set_constant(3, K); e->set_constant(4, Nf); e->set_constant(5, M);
          e->set_constant(6, 0); e->set_constant(7, 0);
        }
        e->dispatch({(unsigned)(((Nf + 31) / 32) * 32),
                     (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
      }
      // AWQ calib tap: acc[group] row L (dim) max-accumulates the per-column
      // |activation| over M rows of `in`. No-op when calibration is off.
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
      // Fused transpose [T,nh,D] (token-major) -> [nh,T,Dp] (head-major) + pair
      // RoPE from the joint table starting at row `row0`, zero-padding the head
      // to Dp. Dp == D uses the unpadded twin.
      void tr_rope(const SharedBuffer& in, std::size_t ie,
                   const SharedBuffer& out, int T, int nh, int D, int Dp,
                   int row0) {
        const std::size_t tb = (std::size_t)row0 * D * sizeof(float);
        if (Dp == D) {
          e->set_function(self->_fn_transpose_rope);
          e->set_buffer(0, in, ie * 2); e->set_buffer(1, out);
          e->set_buffer(2, *rcos, tb); e->set_buffer(3, *rsin, tb);
          e->set_constant(4, nh); e->set_constant(5, T); e->set_constant(6, D);
          e->dispatch({(unsigned)(D / 2), (unsigned)T, (unsigned)nh},
                      {(unsigned)(D / 2), 1, 1});
          return;
        }
        e->set_function(self->_fn_tr_rope_pad);
        e->set_buffer(0, in, ie * 2); e->set_buffer(1, out);
        e->set_buffer(2, *rcos, tb); e->set_buffer(3, *rsin, tb);
        e->set_constant(4, nh); e->set_constant(5, T); e->set_constant(6, D);
        e->set_constant(7, Dp);
        e->dispatch({(unsigned)(Dp / 2), (unsigned)T, (unsigned)nh},
                    {(unsigned)(Dp / 2), 1, 1});
      }
      // Transpose [A,B,D] -> [B,A,Dp] (zero-padded when Dp > D).
      void tr_pad(const SharedBuffer& in, std::size_t ie,
                  const SharedBuffer& out, int A, int B, int D, int Dp) {
        if (Dp == D) {
          e->set_function(self->_fn_transpose);
          e->set_buffer(0, in, ie * 2); e->set_buffer(1, out);
          e->set_constant(2, A); e->set_constant(3, B); e->set_constant(4, D);
          e->dispatch({(unsigned)D, (unsigned)B, (unsigned)A},
                      {(unsigned)D, 1, 1});
          return;
        }
        e->set_function(self->_fn_tr_pad);
        e->set_buffer(0, in, ie * 2); e->set_buffer(1, out);
        e->set_constant(2, A); e->set_constant(3, B); e->set_constant(4, D);
        e->set_constant(5, Dp);
        e->dispatch({(unsigned)Dp, (unsigned)B, (unsigned)A},
                    {(unsigned)Dp, 1, 1});
      }
      // Transpose [A,B,Dp] -> [B,A,D], dropping the padding tail.
      void tr_unpad(const SharedBuffer& in, const SharedBuffer& out,
                    std::size_t oe, int A, int B, int D, int Dp) {
        if (Dp == D) {
          e->set_function(self->_fn_transpose);
          e->set_buffer(0, in); e->set_buffer(1, out, oe * 2);
          e->set_constant(2, A); e->set_constant(3, B); e->set_constant(4, D);
          e->dispatch({(unsigned)D, (unsigned)B, (unsigned)A},
                      {(unsigned)D, 1, 1});
          return;
        }
        e->set_function(self->_fn_tr_unpad);
        e->set_buffer(0, in); e->set_buffer(1, out, oe * 2);
        e->set_constant(2, A); e->set_constant(3, B); e->set_constant(4, D);
        e->set_constant(5, Dp);
        e->dispatch({(unsigned)D, (unsigned)B, (unsigned)A},
                    {(unsigned)D, 1, 1});
      }
      void sdpa(const SharedBuffer& q, const SharedBuffer& k,
                const SharedBuffer& v, const SharedBuffer& out, float sc, int T,
                int D, int nh, int nkv) {
        e->set_function(self->_fn_sdpa);
        e->set_buffer(0, q); e->set_buffer(1, k); e->set_buffer(2, v);
        e->set_buffer(3, out);
        e->set_constant(4, sc); e->set_constant(5, T); e->set_constant(6, D);
        e->set_constant(7, nh); e->set_constant(8, nkv); e->set_constant(9, T);
        e->set_constant(10, T);
        e->dispatch({32, (unsigned)nh, (unsigned)T}, {32, 1, 1});
      }
    };
    return Ops{this, &enc, &rcos, &rsin, eps};
  };

  // The whole attention tail shared by every block kind: q/k already hold the
  // per-head-RMSNormed projections for `rows` tokens; rope them from joint-table
  // row `row0`, run flash (or scalar) attention, and land [rows, H] in `att`.
  auto run_attn = [&](auto& op, int rows, int row0) {
    op.tr_rope(qb, 0, qt, rows, HED, HD, DP, row0);
    op.tr_rope(kb, 0, kt, rows, KVH, HD, DP, row0);
    op.tr_pad(vb, 0, vt, rows, KVH, HD, DP);
    auto it = _attn_cfgs.find(rows);
    if (use_steel && it != _attn_cfgs.end()) {
      op.e->set_function(it->second.fn);
      op.e->set_buffer(0, qt); op.e->set_buffer(1, kt); op.e->set_buffer(2, vt);
      op.e->set_buffer(3, atb);
      op.e->set_buffer(4, it->second.params);
      op.e->dispatch({32 * it->second.nqb, 4 * (unsigned)HED, 1}, {32, 4, 1});
    } else {
      op.sdpa(qt, kt, vt, atb, scale, rows, DP, HED, KVH);
    }
    op.tr_unpad(atb, att, 0, HED, rows, HD, DP);
  };

  // One BooguImageTransformerBlock (refiner / single-stream) over `rows` tokens
  // of `x` starting at element offset `xe`, with rope from joint row `row0`.
  // Modulation (when the block has it) is read from mods slot 0.
  // `split(kProf*)` is the profiling barrier: a no-op unless the caller wants
  // the block's phases timed separately (see kProfProj/Attn/Ff).
  auto run_block = [&](auto& op, const Block& b, const SharedBuffer& x,
                       std::size_t xe, int rows, int row0, const char* g_attn,
                       const char* g_ffn, const char* g_ffact, int gidx,
                       auto&& split) {
    const std::size_t m0 = 0;   // mods slot 0
    // 1. norm1 (+ (1+scale_msa) when modulated) -> q/k/v
    op.rms(x, xe, b.n1, nrm, 0, rows, H);
    if (b.modulated) {
      op.modulate(nrm, 0, mods, m0 + 0, _zero_h, 0, nrm, 0, H, rows * H);
    }
    op.tap(g_attn, gidx, nrm, 0, rows, H);
    op.gemm(nrm, b.q, qb, 0, rows, H, H);
    op.gemm(nrm, b.k, kb, 0, rows, KD, H);
    op.gemm(nrm, b.v, vb, 0, rows, KD, H);
    op.rms(qb, 0, b.qn, qb, 0, rows * HED, HD);
    op.rms(kb, 0, b.kn, kb, 0, rows * KVH, HD);
    split(kProfProj);
    run_attn(op, rows, row0);
    split(kProfAttn);
    op.gemm(att, b.o, ob, 0, rows, H, H);
    // 2. x += [tanh(gate_msa) *] norm2(attn_out)
    op.rms(ob, 0, b.n2, ob, 0, rows, H);
    if (b.modulated) {
      op.gated_tanh(x, xe, mods, m0 + (std::size_t)H, ob, 0, H, rows * H);
    } else {
      op.elt(_fn_residual, x, xe, ob, 0, x, xe, rows * H);
    }
    split(kProfProj);
    // 3. mlp = feed_forward(ffn_norm1(x) [* (1+scale_mlp)])
    op.rms(x, xe, b.fn1, nrm, 0, rows, H);
    if (b.modulated) {
      op.modulate(nrm, 0, mods, m0 + (std::size_t)2 * H, _zero_h, 0, nrm, 0, H,
                  rows * H);
    }
    op.tap(g_ffn, gidx, nrm, 0, rows, H);
    if (_fuse_ff && !b.ff_gu.empty()) {
      op.swiglu_ff(nrm, 0, b.ff_gu, ffm, rows, H, 2 * FFI);
    } else {
      op.gemm(nrm, b.ff_gate, ffg, 0, rows, FFI, H);
      op.gemm(nrm, b.ff_up, ffu, 0, rows, FFI, H);
      op.elt(_fn_swiglu, ffg, 0, ffu, 0, ffm, 0, rows * FFI);
    }
    op.tap(g_ffact, gidx, ffm, 0, rows, FFI);
    op.gemm(ffm, b.ff_down, ob, 0, rows, H, FFI);
    // 4. x += [tanh(gate_mlp) *] ffn_norm2(mlp)
    op.rms(ob, 0, b.fn2, ob, 0, rows, H);
    if (b.modulated) {
      op.gated_tanh(x, xe, mods, m0 + (std::size_t)3 * H, ob, 0, H, rows * H);
    } else {
      op.elt(_fn_residual, x, xe, ob, 0, x, xe, rows * H);
    }
    split(kProfFf);
  };

  // Fill mods slot `slot` from a block's norm1.linear: mod = linear(SiLU(temb)).
  auto fill_mod = [&](auto& op, int slot, const QWeight& w,
                      const SharedBuffer& b) {
    op.gemm(tsilu, w, mods, (std::size_t)slot * 4 * H, 1, 4 * H, TE);
    op.bias(b, 0, mods, (std::size_t)slot * 4 * H, 1, 4 * H);
  };

  // ===== stream 1: conditioning + patch embed + the three refiner stacks ====
  if (prof) { t_setup = ms_since(mk); mk = tnow(); }
  {
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    auto op = make_ops(enc);
    // TimestepEmbedding: linear_1 -> SiLU -> linear_2  (-> temb [TE]).
    op.gemm(te_in, _t_emb1, te1, 0, 1, TE, FD);
    op.bias(_t_emb1_b, 0, te1, 0, 1, TE);
    op.elt(_fn_mulsig, te1, 0, te1, 0, te1, 0, TE);          // SiLU
    op.gemm(te1, _t_emb2, temb, 0, 1, TE, TE);
    op.bias(_t_emb2_b, 0, temb, 0, 1, TE);
    op.elt(_fn_mulsig, temb, 0, temb, 0, tsilu, 0, TE);      // SiLU(temb)
    // caption_embedder: RMSNorm(instruct) -> Linear -> txt [TS, H]. The AWQ tap
    // sits AFTER the norm -- that is the Linear's actual input distribution.
    op.rms(instruct_b, 0, _cap_norm, instruct_b, 0, TS, c.instruct_dim);
    op.tap("emb_ctx", 0, instruct_b, 0, TS, c.instruct_dim);
    op.gemm(instruct_b, _cap_lin, txt, 0, TS, H, c.instruct_dim);
    op.bias(_cap_lin_b, 0, txt, 0, TS, H);
    // Patch embed. The image stream is laid out [refs...; target], so the
    // references embed at img[0..IS_REF) through ref_image_patch_embedder (+
    // their image_index_embedding row) and the target at img[IS_REF..IS)
    // through x_embedder.
    {
      std::size_t ro = 0;
      for (std::size_t i = 0; i < refs.size(); ++i) {
        op.tap("emb_ref", 0, *refs_b[i], 0, refs[i].seq, XIN);
        op.gemm(*refs_b[i], _ref_embed, img, ro * H, refs[i].seq, H, XIN);
        op.bias(_ref_embed_b, 0, img, ro * H, refs[i].seq, H);
        // + image_index_embedding[i]: reference i's own [H] row, broadcast
        // over its tokens -- what distinguishes multiple references.
        op.bias(_img_index, i * (std::size_t)H, img, ro * H, refs[i].seq, H);
        ro += (std::size_t)refs[i].seq;
      }
    }
    op.tap("emb_x", 0, latents_b, 0, IS_GEN, XIN);
    op.gemm(latents_b, _x_embed, img, (std::size_t)IS_REF * H, IS_GEN, H, XIN);
    op.bias(_x_embed_b, 0, img, (std::size_t)IS_REF * H, IS_GEN, H);
    enc.end();
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt("MetalBooguTransformer::forward_dit: {}",
                                 gpu_err.empty() ? "GPU failed" : gpu_err));
      }
      return {};
    }
  }
  if (prof) { t_cond += ms_since(mk); mk = tnow(); }
  {
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    auto op = make_ops(enc);
    // The refiners are a small slice of the step, so they are timed as one
    // bucket -- no per-phase barriers.
    auto nosplit = [](int) {};
    // context_refiner: UNMODULATED, over the instruction tokens.
    for (int L = 0; L < (int)_ctx_refiner.size(); ++L) {
      run_block(op, _ctx_refiner[(std::size_t)L], txt, 0, TS, rowText,
                "ctx_attn", "ctx_ffn", "ctx_ffact", L, nosplit);
    }
    // noise_refiner: modulated, over the TARGET tokens only.
    for (int L = 0; L < (int)_noise_refiner.size(); ++L) {
      const Block& b = _noise_refiner[(std::size_t)L];
      fill_mod(op, 0, b.mod, b.mod_b);
      run_block(op, b, img, (std::size_t)IS_REF * H, IS_GEN, rowGen,
                "noise_attn", "noise_ffn", "noise_ffact", L, nosplit);
    }
    // ref_image_refiner: modulated, EACH REFERENCE ON ITS OWN (the reference
    // flattens the references into a temporary batch, so a reference attends
    // only its own tokens here -- not the other references, not the target).
    {
      int ro = 0;
      for (std::size_t i = 0; i < refs.size(); ++i) {
        for (int L = 0; L < (int)_ref_refiner.size(); ++L) {
          const Block& b = _ref_refiner[(std::size_t)L];
          fill_mod(op, 0, b.mod, b.mod_b);
          run_block(op, b, img, (std::size_t)ro * H, refs[i].seq,
                    rowImg + ro, "ref_attn", "ref_ffn", "ref_ffact", L,
                    nosplit);
        }
        ro += refs[i].seq;
      }
    }
    enc.end();
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt("MetalBooguTransformer::forward_dit: {}",
                                 gpu_err.empty() ? "GPU failed" : gpu_err));
      }
      return {};
    }
  }
  if (prof) { t_refine += ms_since(mk); }

  // ---- the resident set, once per forward --------------------------------
  //
  // THIS WAS MISSING. Everything below it -- resident_pages_,
  // evict_tail_block_, release_resident_blocks -- was written when the
  // residency machinery landed, but nothing ever called begin_forward(),
  // note_weight_residency() or note_healthy_forward(). The effect was
  // silent and complete: `_admitted_this_forward` never reset, so growth
  // stopped for good after the per-forward cap was reached ONCE; the
  // first hysteresis refusal cleared `_growing` for the rest of the
  // process; the paging backstop never ran; and resident_pages_ had no
  // callers at all, so a block that was compressed was never noticed and
  // never given back. Boogu kept a handful of blocks on its first pass
  // and streamed everything for the rest of the run.
  //
  // Re-arm growth for this forward (the ratchet from any earlier eviction
  // deliberately survives), then take the measurement that actually finds
  // the limit: are the blocks we kept still in RAM? Gated on our OWN
  // compressed footprint moving, because the page walk costs ~57 ms per
  // 4.3 GB and a healthy run would pay it every step to be told nothing.
  {
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
      std::size_t examined = 0, incore = 0, paged_out = 0;
      resident_pages_(&examined, &incore, &paged_out);
      if (examined > 0 && incore < examined) {
        shortfall = true;
        std::size_t freed = _resid.note_weight_residency(
            examined, incore, [this]() -> std::size_t {
              return evict_tail_block_();
            });
        if (_mc->session() != nullptr) {
          _mc->session()->log_normal(fmt(
              "MetalBooguTransformer: resident weights are only {}% in RAM "
              "({} of {} sampled pages paged out, {} MB wired) -- released "
              "{} MB, now {} blocks resident",
              (int)(100.0 * (double)incore / (double)examined),
              paged_out, examined, _wire.wired_bytes() >> 20, freed >> 20,
              _resid.count()));
        }
      }
    }
    // Nothing of ours had left RAM this step -- either the walk said so or
    // there was no compression to make it worth walking. Enough in a row
    // lifts the ratchet by a block, so a shed taken during a momentary
    // squeeze is not the last word on the run.
    if (!shortfall) { _resid.note_healthy_forward(); }
  }

  // ===== double-stream blocks ==============================================
  for (int L = 0; L < c.n_double; ++L) {
    if (_stream_stop && _stream_stop()) { return {}; }
    if (_block_progress) { _block_progress(L, c.n_double + c.n_single); }
    const bool held = L < (int)_double.size() &&
                      !_double[(std::size_t)L].jq_i.empty();
    const bool streaming = _stream_blocks && !held;
    const DoubleBlock* streamed = nullptr;
    if (streaming) {
      streamed = _double_slots.acquire(L);
      if (streamed == nullptr) { return {}; }
    }
    const DoubleBlock& b = streaming ? *streamed : _double[(std::size_t)L];
    if (prof) { mk = tnow(); }
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    auto op = make_ops(enc);
    double* const acc[3] = {&t_dbl_proj, &t_dbl_attn, &t_dbl_ff};
    auto psplit = [&](int which) {
      if (!prof) { return; }
      enc.end();
      stream.commit().wait();
      *acc[which] += ms_since(mk);
      stream = _mc->make_command_stream();
      enc = stream.begin_compute();
      op = make_ops(enc);
      mk = tnow();
    };
    // Five modulation vectors: slots 0/1/2 = img_norm1/2/3, 3/4 = instruct
    // norm1/2. Chunk order per slot is [scale_msa | gate_msa | scale_mlp |
    // gate_mlp]; the reference reads norm2's chunk[1] as the mlp SHIFT and
    // norm3's chunk[1] as the image self-attention gate.
    fill_mod(op, 0, b.mi1, b.mi1_b);
    fill_mod(op, 1, b.mi2, b.mi2_b);
    fill_mod(op, 2, b.mi3, b.mi3_b);
    fill_mod(op, 3, b.mt1, b.mt1_b);
    fill_mod(op, 4, b.mt2, b.mt2_b);
    const std::size_t MI1 = 0, MI2 = (std::size_t)4 * H,
                      MI3 = (std::size_t)8 * H, MT1 = (std::size_t)12 * H,
                      MT2 = (std::size_t)16 * H;
    // Pre-modulation of BOTH streams, all read from the PRE-block residuals.
    // nrm  = img_norm1  (joint-attention image input)
    // nrm2 = img_norm2  (mlp input base, kept until the mlp)
    // The instruct equivalents live in the tail rows of the same buffers.
    op.rms(img, 0, b.ni1, nrm, 0, IS, H);
    op.modulate(nrm, 0, mods, MI1 + 0, _zero_h, 0, nrm, 0, H, IS * H);
    op.rms(img, 0, b.ni2, nrm2, 0, IS, H);
    op.modulate(nrm2, 0, mods, MI2 + 0, _zero_h, 0, nrm2, 0, H, IS * H);
    // The instruction stream's norm1/norm2 land after the image rows.
    const std::size_t TOFF = (std::size_t)IS * H;
    op.rms(txt, 0, b.nt1, nrm, TOFF, TS, H);
    op.modulate(nrm, TOFF, mods, MT1 + 0, _zero_h, 0, nrm, TOFF, H, TS * H);
    op.rms(txt, 0, b.nt2, nrm2, TOFF, TS, H);
    op.modulate(nrm2, TOFF, mods, MT2 + 0, _zero_h, 0, nrm2, TOFF, H, TS * H);

    // ---- joint attention over [instruct; image] ----
    // Per-stream q/k/v projections written into the SHARED q/k/v buffers at the
    // joint row offsets: text at rows [0,TS), image at [TS, seq).
    op.tap("dbl_jattn_txt", L, nrm, TOFF, TS, H);
    op.gemm(nrm, b.jq_t, qb, 0, TS, H, H, TOFF);
    op.gemm(nrm, b.jk_t, kb, 0, TS, KD, H, TOFF);
    op.gemm(nrm, b.jv_t, vb, 0, TS, KD, H, TOFF);
    op.tap("dbl_jattn_img", L, nrm, 0, IS, H);
    op.gemm(nrm, b.jq_i, qb, (std::size_t)TS * H, IS, H, H);
    op.gemm(nrm, b.jk_i, kb, (std::size_t)TS * KD, IS, KD, H);
    op.gemm(nrm, b.jv_i, vb, (std::size_t)TS * KD, IS, KD, H);
    op.rms(qb, 0, b.jqn, qb, 0, seq * HED, HD);
    op.rms(kb, 0, b.jkn, kb, 0, seq * KVH, HD);
    psplit(kProfProj);
    run_attn(op, seq, rowText);
    psplit(kProfAttn);
    // Per-stream output projections, then the SHARED to_out.0 over the rejoined
    // sequence. Writing instruct_out at ob[0] and img_out at ob[TS] IS the
    // rejoin the reference does with a split + concat.
    op.gemm(att, b.jout_t, ob, 0, TS, H, H);
    op.gemm(att, b.jout_i, ob, (std::size_t)TS * H, IS, H, H,
            (std::size_t)TS * H);
    op.tap("dbl_jout", L, ob, 0, seq, H);
    op.gemm(ob, b.jo, att, 0, seq, H, H);   // att reused as the to_out result

    // ---- image self-attention (image tokens only) ----
    // nrm is free again for the image rows: img_norm3.
    op.rms(img, 0, b.ni3, nrm, 0, IS, H);
    op.modulate(nrm, 0, mods, MI3 + 0, _zero_h, 0, nrm, 0, H, IS * H);
    op.tap("dbl_sattn_img", L, nrm, 0, IS, H);
    op.gemm(nrm, b.sq, qb, 0, IS, H, H);
    op.gemm(nrm, b.sk, kb, 0, IS, KD, H);
    op.gemm(nrm, b.sv, vb, 0, IS, KD, H);
    op.rms(qb, 0, b.sqn, qb, 0, IS * HED, HD);
    op.rms(kb, 0, b.skn, kb, 0, IS * KVH, HD);
    // `att` holds the joint to_out result; stash the image self-attention in
    // `ob` via run_attn's `att` then project -- so first consume `att`.
    // Residual 1: img += tanh(img_gate_msa) * img_attn_norm(joint_att_img)
    op.rms(att, (std::size_t)TS * H, b.i_attn_n, ob, 0, IS, H);
    op.gated_tanh(img, 0, mods, MI1 + (std::size_t)H, ob, 0, H, IS * H);
    // instruct += tanh(instruct_gate_msa) * instruct_attn_norm(joint_att_txt)
    op.rms(att, 0, b.t_attn_n, ob, 0, TS, H);
    op.gated_tanh(txt, 0, mods, MT1 + (std::size_t)H, ob, 0, H, TS * H);
    // Now run the image self-attention (overwrites att).
    psplit(kProfProj);
    run_attn(op, IS, rowImg);
    psplit(kProfAttn);
    op.gemm(att, b.so, ob, 0, IS, H, H);
    op.rms(ob, 0, b.i_self_n, ob, 0, IS, H);
    // Residual 2: img += tanh(img_gate_self) * img_self_attn_norm(self_att)
    op.gated_tanh(img, 0, mods, MI3 + (std::size_t)H, ob, 0, H, IS * H);
    psplit(kProfProj);

    // ---- feed-forwards ----
    // img mlp input = (1 + img_scale_mlp) * img_norm2_out + img_shift_mlp,
    // where scale_mlp is img_norm1's chunk[2] and shift_mlp img_norm2's
    // chunk[1] -- two DIFFERENT modulation slots over the PRE-block norm2.
    op.modulate(nrm2, 0, mods, MI1 + (std::size_t)2 * H, mods,
                MI2 + (std::size_t)H, nrm, 0, H, IS * H);
    op.rms(nrm, 0, b.i_ffn1, nrm, 0, IS, H);
    op.tap("dbl_ffn_img", L, nrm, 0, IS, H);
    if (_fuse_ff && !b.iff_gu.empty()) {
      op.swiglu_ff(nrm, 0, b.iff_gu, ffm, IS, H, 2 * FFI);
    } else {
      op.gemm(nrm, b.iff_gate, ffg, 0, IS, FFI, H);
      op.gemm(nrm, b.iff_up, ffu, 0, IS, FFI, H);
      op.elt(_fn_swiglu, ffg, 0, ffu, 0, ffm, 0, IS * FFI);
    }
    op.tap("dbl_ffact_img", L, ffm, 0, IS, FFI);
    op.gemm(ffm, b.iff_down, ob, 0, IS, H, FFI);
    op.rms(ob, 0, b.i_ffn2, ob, 0, IS, H);
    op.gated_tanh(img, 0, mods, MI1 + (std::size_t)3 * H, ob, 0, H, IS * H);
    // instruct mlp: same shape with instruct_norm1 chunk[2] / norm2 chunk[1].
    op.modulate(nrm2, TOFF, mods, MT1 + (std::size_t)2 * H, mods,
                MT2 + (std::size_t)H, nrm, 0, H, TS * H);
    op.rms(nrm, 0, b.t_ffn1, nrm, 0, TS, H);
    op.tap("dbl_ffn_txt", L, nrm, 0, TS, H);
    if (_fuse_ff && !b.tff_gu.empty()) {
      op.swiglu_ff(nrm, 0, b.tff_gu, ffm, TS, H, 2 * FFI);
    } else {
      op.gemm(nrm, b.tff_gate, ffg, 0, TS, FFI, H);
      op.gemm(nrm, b.tff_up, ffu, 0, TS, FFI, H);
      op.elt(_fn_swiglu, ffg, 0, ffu, 0, ffm, 0, TS * FFI);
    }
    op.tap("dbl_ffact_txt", L, ffm, 0, TS, FFI);
    op.gemm(ffm, b.tff_down, ob, 0, TS, H, FFI);
    op.rms(ob, 0, b.t_ffn2, ob, 0, TS, H);
    op.gated_tanh(txt, 0, mods, MT1 + (std::size_t)3 * H, ob, 0, H, TS * H);
    enc.end();
    std::string gpu_err;
    {
      metal_compute::CommandStream::Fence bf = stream.commit();
      if (streaming) {
        // BETWEEN THE COMMIT AND THE WAIT the GPU is busy with this
        // block and this thread has nothing to do. That window is
        // where the next block's read goes.
        {
          int nxt = -1;
          for (int n = L + 1; n < c.n_double; ++n) {
            const bool h = n < (int)_double.size() &&
                           !_double[(std::size_t)n].jq_i.empty();
            if (!h) { nxt = n; break; }
          }
          _double_slots.prefetch(nxt);
        }
      }
      if (!bf.wait_ok(&gpu_err)) {
        if (_mc->session() != nullptr) {
          _mc->session()->warn(fmt("MetalBooguTransformer::forward_dit: {}",
                                   gpu_err.empty() ? "GPU failed" : gpu_err));
        }
        return {};
      }
    }
    // The commit above has been WAITED for, so nothing encoded still
    // points at this block's buffers -- which is why the promotion is
    // here and not at the top of the iteration.
    if (streaming && L < (int)_double.size()) {
      const std::size_t nb = _double_slots.last_bytes();
      // Past the wire budget there is nothing to gain: the block would be
      // kept unprotected, the compressor would take it (it is the coldest
      // memory in the process), and the next residency walk would shed a
      // block and ratchet the ceiling over the whole resident set. Better
      // not to hold it at all.
      if (_wire.wirable(nb) && _resid.admit(_mc, nb) &&
          _double_slots.promote_into(_double[(std::size_t)L])) {
        // Wired LAST, after every write this block will ever get: mlock
        // pins the pages that exist NOW.
        _wire.note_wired(_mc, wire_block_(_double[(std::size_t)L], true), nb);
        _resid.note_admitted(nb);
      }
    }
    if (prof) { t_dbl_ff += ms_since(mk); }
  }

  // Fuse the streams: joint = [instruct; image].
  if (prof) { mk = tnow(); }
  std::memcpy(joint.contents(), txt.contents(), (std::size_t)TS * H * 2);
  std::memcpy(static_cast<std::uint16_t*>(joint.contents())
                  + (std::size_t)TS * H,
              img.contents(), (std::size_t)IS * H * 2);
  if (prof) { t_join = ms_since(mk); }

  // ===== single-stream blocks ==============================================
  for (int L = 0; L < c.n_single; ++L) {
    if (_stream_stop && _stream_stop()) { return {}; }
    // Continues the double stack's numbering: one sequence per forward.
    if (_block_progress) {
      _block_progress(c.n_double + L, c.n_double + c.n_single);
    }
    const bool held = L < (int)_single.size() &&
                      !_single[(std::size_t)L].q.empty();
    const bool streaming = _stream_blocks && !held;
    const Block* streamed = nullptr;
    if (streaming) {
      // Two reusable destinations, refilled in place with pread and with
      // the next block's read already issued under the previous block's
      // GPU work. See shared/block-slots.h; the fallback to a per-block
      // allocation lives inside it and is sticky.
      streamed = _single_slots.acquire(L);
      if (streamed == nullptr) { return {}; }
    }
    const Block& b = streaming ? *streamed : _single[(std::size_t)L];
    if (prof) { mk = tnow(); }
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    auto op = make_ops(enc);
    double* const acc[3] = {&t_sgl_proj, &t_sgl_attn, &t_sgl_ff};
    auto split = [&](int which) {
      if (!prof) { return; }
      enc.end();
      stream.commit().wait();
      *acc[which] += ms_since(mk);
      stream = _mc->make_command_stream();
      enc = stream.begin_compute();
      op = make_ops(enc);
      mk = tnow();
    };
    fill_mod(op, 0, b.mod, b.mod_b);
    run_block(op, b, joint, 0, seq, rowText, "sgl_attn", "sgl_ffn",
              "sgl_ffact", L, split);
    enc.end();
    std::string gpu_err;
    {
      metal_compute::CommandStream::Fence bf = stream.commit();
      if (streaming) {
        // BETWEEN THE COMMIT AND THE WAIT the GPU is busy with this
        // block and this thread has nothing to do. That window is
        // where the next block's read goes.
        {
          int nxt = -1;
          for (int n = L + 1; n < c.n_single; ++n) {
            const bool h = n < (int)_single.size() &&
                           !_single[(std::size_t)n].q.empty();
            if (!h) { nxt = n; break; }
          }
          _single_slots.prefetch(nxt);
        }
      }
      if (!bf.wait_ok(&gpu_err)) {
        if (_mc->session() != nullptr) {
          _mc->session()->warn(fmt("MetalBooguTransformer::forward_dit: {}",
                                   gpu_err.empty() ? "GPU failed" : gpu_err));
        }
        return {};
      }
    }
    // Waited above, so nothing encoded still points at this block.
    if (streaming && L < (int)_single.size()) {
      const std::size_t nb = _single_slots.last_bytes();
      if (_wire.wirable(nb) && _resid.admit(_mc, nb) &&
          _single_slots.promote_into(_single[(std::size_t)L])) {
        _wire.note_wired(_mc, wire_block_(_single[(std::size_t)L], true), nb);
        _resid.note_admitted(nb);
      }
    }
  }

  // ===== final: LuminaLayerNormContinuous(target tail, temb) + linear_2 =====
  // scale = linear_1(SiLU(temb)) [H]; x = LN(x) * (1 + scale); v = linear_2(x).
  // Only the TARGET tokens (the tail of the joint sequence) are projected.
  if (prof) { mk = tnow(); }
  {
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    auto op = make_ops(enc);
    op.gemm(tsilu, _out_lin1, outmod, 0, 1, H, TE);
    op.bias(_out_lin1_b, 0, outmod, 0, 1, H);
    const std::size_t gen_off = (std::size_t)(seq - IS_GEN) * H;
    op.ln(joint, gen_off, nrm, 0, IS_GEN, H, 1e-6f);   // affine-free, eps 1e-6
    op.modulate(nrm, 0, outmod, 0, _zero_h, 0, nrm, 0, H, IS_GEN * H);
    op.tap("emb_proj", 0, nrm, 0, IS_GEN, H);
    op.gemm(nrm, _out_lin2, velocity, 0, IS_GEN, OC, H);
    op.bias(_out_lin2_b, 0, velocity, 0, IS_GEN, OC);
    enc.end();
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt("MetalBooguTransformer::forward_dit: {}",
                                 gpu_err.empty() ? "GPU failed" : gpu_err));
      }
      return {};
    }
  }
  if (prof) {
    t_final += ms_since(mk);
    const double tot = t_setup + t_cond + t_refine + t_dbl_proj + t_dbl_attn +
                       t_dbl_ff + t_sgl_proj + t_sgl_attn + t_sgl_ff + t_join +
                       t_final;
    if (_mc->session() != nullptr) {
      _mc->session()->log_normal(fmt(
          "Boogu DiT profile (seq={} img={} ref={} txt={}, {}+{} blocks, "
          "attn={} gemm={}{}): total {} ms | setup {} | cond(embed) {} "
          "| refiners {} | double(proj {} attn {} ff {}) "
          "| single(proj {} attn {} ff {}) | join {} | final {}",
          seq, IS_GEN, IS_REF, TS, c.n_double, c.n_single,
          use_steel ? (nax ? "steel-nax(pad128)"
                           : (_attn_native ? "steel(bd120 native)"
                                           : "steel(pad128)"))
                    : "scalar",
          _use_mma2 ? "mma2"
                    : (_quant_bits > 0
                           ? (_qmm_tile >= 1 ? "qmm-bm128" : "qmm-bm32")
                           : (_gemm_tile == 2 ? "bm64bn64"
                                              : (_gemm_tile == 1 ? "bm64"
                                                                 : "bm32"))),
          _fuse_ff ? " fused-ff" : "",
          (long)tot, (long)t_setup, (long)t_cond, (long)t_refine,
          (long)t_dbl_proj, (long)t_dbl_attn, (long)t_dbl_ff, (long)t_sgl_proj,
          (long)t_sgl_attn, (long)t_sgl_ff, (long)t_join, (long)t_final));
    }
  }
  return velocity;
}

}  // namespace genai
}  // namespace vpipe
