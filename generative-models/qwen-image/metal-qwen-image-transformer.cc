#include "generative-models/shared/mma-tile.h"
#include "generative-models/qwen-image/metal-qwen-image-transformer.h"

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

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// C++ mirror of mlx::steel::AttnParams (steel/attn/params.h) -- the param block
// the vendored steel flash-attention kernel reads. Same layout as the Krea-2 /
// FLUX.2 / LM copies.
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
  std::uint32_t u; std::memcpy(&u, &f, 4);
  // round-to-nearest-even.
  const std::uint32_t r = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
  return (std::uint16_t)r;
}

inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

// Load a checkpoint tensor -> bf16 SharedBuffer (BF16 memcpy'd; F16/F32 -> bf16).
SharedBuffer
to_elt_(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm)
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
  if (info->dtype == "BF16") {
    std::memcpy(d, raw.contents(), n * 2);
  } else if (info->dtype == "F32") {
    const auto* s = static_cast<const float*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_(s[i]); }
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_((float)s[i]); }
  } else {
    return {};
  }
  return out;
}

// Namespace for this class's derived-tensor cache keys. A WeightSet is
// shared by everything reading one checkpoint, so a key has to say which
// class's transform produced the bytes, not just which tensor.
constexpr const char* kKey = "qwen-image-dit/";

}  // namespace

MetalQwenImageTransformer::~MetalQwenImageTransformer()
{
  // GIVE THE POOL BACK. Freeing a wired buffer unwires it in the kernel,
  // so the machine recovers either way -- but the pool's own counter
  // would not, and a DiT destroyed and reloaded per prompt would leak its
  // whole share of the budget per prompt until nothing could wire at all.
  if (_wire.on()) {
    wire_fixed_(false);
    for (Block& b : _blocks) { wire_block_(b, false); }
  }
}

// bf16 view of a checkpoint tensor. Cached ones go through the weight
// set so a second model over this checkpoint shares them; streamed ones
// are rebuilt per forward and retained by nobody.
SharedBuffer
MetalQwenImageTransformer::to_elt_(WeightSet& ws, const std::string& name,
                                   Retain r)
{
  auto build = [&]() { return vpipe::genai::to_elt_(ws.src(), _mc, name); };
  if (r == Retain::Streamed) { return ws.stream_derived(build); }
  return ws.derived(std::string(kKey) + "elt|" + name, build);
}

bool
MetalQwenImageTransformer::load_linear_(WeightSet& ws,
                                        const std::string& pre, SharedBuffer& w,
                                        SharedBuffer& b, Retain r)
{
  w = to_elt_(ws, pre + ".weight", r);
  b = to_elt_(ws, pre + ".bias", r);
  return !w.empty() && !b.empty();
}

// Load `<name>` as a (possibly-quantized) linear weight. Quantized iff the
// model is quantized (_quant_bits>0) AND `<name>.scales` is present: then
// `<name>.weight` is U32 packed codes ([N, K*bits/32], loaded RAW) and
// `.scales`/`.biases` are F16 -> bf16. Bits are derived from the tensor shapes
// so a mixed-precision checkpoint (w4/w8 per layer) loads correctly. Otherwise
// `<name>.weight` is a dense bf16 matrix.
MetalQwenImageTransformer::QWeight
MetalQwenImageTransformer::load_qw_(WeightSet& ws, const std::string& name,
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
    qw.bits   = (bits == 8) ? 8 : 4;
    // Codes stay OWNED copies (as before this class went through the
    // set) -- switching them to mapped views is a residency change, not
    // a refactor, and belongs in its own measured step.
    qw.codes = r == Retain::Streamed
                   ? ws.stream_tensor(name + ".weight", _mc,
                                      WeightSet::Residency::Copied)
                   : ws.tensor(name + ".weight", _mc,
                               WeightSet::Residency::Copied);
    qw.scales = to_elt_(ws, name + ".scales", r);    // F16 -> bf16
    qw.qbias  = to_elt_(ws, name + ".biases", r);
    if (!qw.codes.empty() && !qw.scales.empty() && !qw.qbias.empty()) {
      qw.quantized = true;
      return qw;
    }
    qw.codes = {}; qw.scales = {}; qw.qbias = {};
  }
  qw.w = to_elt_(ws, name + ".weight", r);           // dense bf16
  return qw;
}

bool
MetalQwenImageTransformer::load_linear_q_(WeightSet& ws,
                                          const std::string& pre, QWeight& qw,
                                          SharedBuffer& b, Retain r)
{
  qw = load_qw_(ws, pre, r);
  b = to_elt_(ws, pre + ".bias", r);
  return !qw.empty() && !b.empty();
}

std::size_t
MetalQwenImageTransformer::qw_bytes_(const QWeight& w)
{
  return w.w.byte_size() + w.codes.byte_size() + w.scales.byte_size() +
         w.qbias.byte_size();
}


// Shorthand for the per-tensor placement the refill needs stated.
using P = vpipe::genai::Placement;

// ---- the streamed block's reusable destinations ------------------------

// Every (checkpoint name, destination, raw) of block L, in a fixed
// order. `raw` marks the checkpoint's OWN words -- a quantized pack's
// u32 codes -- which the REBUILD route must copy rather than read as
// bf16; the refill itself asks for bf16 either way and places u32
// untouched.
void
MetalQwenImageTransformer::each_block_tensor_(
    int L, Block& b, const BlockSlots<Block>::TensorFn& fn) const
{
  const std::string p = "transformer_blocks." + std::to_string(L) + ".";
  // A MATRIX is three tensors when quantized and one when dense, and
  // which it is was decided when the slot was built -- so the layout
  // comes from the QWeight, not from the checkpoint. A pack that
  // disagrees fails the refill's size check and forces a rebuild, which
  // is the right answer to "these are not the same weights".
  auto qw = [&fn](const std::string& base, QWeight& w) {
    if (w.quantized) {
      fn(base + ".weight", w.codes, P::kRaw);
      fn(base + ".scales", w.scales, P::kBf16);
      fn(base + ".biases", w.qbias, P::kBf16);
    } else {
      fn(base + ".weight", w.w, P::kBf16);
    }
  };
  auto lin = [&](const std::string& base, QWeight& w,
                 metal_compute::SharedBuffer& bias) {
    qw(base, w);
    fn(base + ".bias", bias, P::kBf16);
  };
  lin(p + "img_mod.1", b.img_mod_w, b.img_mod_b);
  lin(p + "txt_mod.1", b.txt_mod_w, b.txt_mod_b);
  lin(p + "attn.to_q", b.qw, b.qb);
  lin(p + "attn.to_k", b.kw, b.kb);
  lin(p + "attn.to_v", b.vw, b.vb);
  lin(p + "attn.to_out.0", b.ow, b.ob);
  lin(p + "attn.add_q_proj", b.aqw, b.aqb);
  lin(p + "attn.add_k_proj", b.akw, b.akb);
  lin(p + "attn.add_v_proj", b.avw, b.avb);
  lin(p + "attn.to_add_out", b.aow, b.aob);
  fn(p + "attn.norm_q.weight", b.nq, P::kBf16);
  fn(p + "attn.norm_k.weight", b.nk, P::kBf16);
  fn(p + "attn.norm_added_q.weight", b.naq, P::kBf16);
  fn(p + "attn.norm_added_k.weight", b.nak, P::kBf16);
  lin(p + "img_mlp.net.0.proj", b.img_fc1_w, b.img_fc1_b);
  lin(p + "img_mlp.net.2", b.img_fc2_w, b.img_fc2_b);
  lin(p + "txt_mlp.net.0.proj", b.txt_fc1_w, b.txt_fc1_b);
  lin(p + "txt_mlp.net.2", b.txt_fc2_w, b.txt_fc2_b);
}

// Allocate `dst` with `src`'s shapes and flags, optionally copying the
// bytes. One function for two uses: a promotion and a second slot
// differ only in whether the contents come along.
bool
MetalQwenImageTransformer::clone_block_(const Block& src, Block& dst,
                                        bool copy) const
{
  bool ok = true;
  auto one = [&](const metal_compute::SharedBuffer& s,
                 metal_compute::SharedBuffer& d) {
    if (!ok || s.empty()) { d = metal_compute::SharedBuffer{}; return; }
    d = _mc->make_shared_buffer(s.byte_size());
    if (d.empty()) { ok = false; return; }
    if (copy) { std::memcpy(d.contents(), s.contents(), s.byte_size()); }
  };
  auto qw = [&](const QWeight& s, QWeight& d) {
    d.quantized = s.quantized;
    d.bits      = s.bits;
    one(s.w, d.w);
    one(s.codes, d.codes);
    one(s.scales, d.scales);
    one(s.qbias, d.qbias);
  };
  qw(src.img_mod_w, dst.img_mod_w); qw(src.txt_mod_w, dst.txt_mod_w);
  one(src.img_mod_b, dst.img_mod_b); one(src.txt_mod_b, dst.txt_mod_b);
  qw(src.qw, dst.qw); qw(src.kw, dst.kw);
  qw(src.vw, dst.vw); qw(src.ow, dst.ow);
  one(src.qb, dst.qb); one(src.kb, dst.kb);
  one(src.vb, dst.vb); one(src.ob, dst.ob);
  qw(src.aqw, dst.aqw); qw(src.akw, dst.akw);
  qw(src.avw, dst.avw); qw(src.aow, dst.aow);
  one(src.aqb, dst.aqb); one(src.akb, dst.akb);
  one(src.avb, dst.avb); one(src.aob, dst.aob);
  one(src.nq, dst.nq); one(src.nk, dst.nk);
  one(src.naq, dst.naq); one(src.nak, dst.nak);
  qw(src.img_fc1_w, dst.img_fc1_w); qw(src.img_fc2_w, dst.img_fc2_w);
  qw(src.txt_fc1_w, dst.txt_fc1_w); qw(src.txt_fc2_w, dst.txt_fc2_w);
  one(src.img_fc1_b, dst.img_fc1_b); one(src.img_fc2_b, dst.img_fc2_b);
  one(src.txt_fc1_b, dst.txt_fc1_b); one(src.txt_fc2_b, dst.txt_fc2_b);
  if (!ok) { dst = Block{}; }
  return ok;
}

// Rebuild ONE tensor a raw read could not place -- an f32 dtype, or a
// shape that disagrees with the slot. The same two routes load_block_
// uses, so a repaired tensor is byte-identical to a freshly built one.
metal_compute::SharedBuffer
MetalQwenImageTransformer::rebuild_one_(const std::string& nm,
                                        vpipe::genai::Placement how)
{
  if (_ws == nullptr) { return {}; }
  if (how == P::kRaw) {
    return _ws->stream_tensor(nm, _mc, WeightSet::Residency::Copied);
  }
  return to_elt_(*_ws, nm, Retain::Streamed);
}

void
MetalQwenImageTransformer::configure_slots_()
{
  BlockSlots<Block>::Ops ops;
  ops.each = [this](int L, Block& b,
                    const BlockSlots<Block>::TensorFn& fn) {
    each_block_tensor_(L, b, fn);
  };
  ops.rebuild_one = [this](const std::string& nm,
                           vpipe::genai::Placement how) {
    return rebuild_one_(nm, how);
  };
  ops.build = [this](int L, Block& b) {
    return _ws != nullptr && load_block_(*_ws, L, b, Retain::Streamed);
  };
  ops.clone = [this](const Block& s, Block& d, bool copy) {
    return clone_block_(s, d, copy);
  };
  ops.bytes = [](const Block& b) { return block_bytes_(b); };
  ops.empty = [](const Block& b) { return b.qw.empty(); };
  _slots.set_weight_set(_ws.get());
  _slots.configure(_mc, std::move(ops), "MetalQwenImageTransformer",
                   "VPIPE_QIE_NO_SLOTS");
}

// ---- the wired pool ---------------------------------------------------
//
// One buffer at a time, STOPPING at the first refusal rather than
// unwinding: a partly wired block is partly protected, which is strictly
// better than none -- and giving protection back on the way out means
// competing for it again on the next block, against a pool that has just
// said no.
//
// The list mirrors block_bytes_ exactly. It has to: wirable() gates
// admission on the byte count that returns, so a buffer counted there and
// not wired here is a block the model believes is protected and is not.
std::size_t
MetalQwenImageTransformer::wire_block_(Block& b, bool on)
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
  QWeight* q[] = {&b.img_mod_w, &b.txt_mod_w, &b.qw, &b.kw, &b.vw, &b.ow,
                  &b.aqw, &b.akw, &b.avw, &b.aow, &b.img_fc1_w,
                  &b.img_fc2_w, &b.txt_fc1_w, &b.txt_fc2_w};
  for (QWeight* w : q) { qw(*w); }
  metal_compute::SharedBuffer* p[] = {
      &b.img_mod_b, &b.txt_mod_b, &b.qb, &b.kb, &b.vb, &b.ob,
      &b.aqb, &b.akb, &b.avb, &b.aob, &b.nq, &b.nk, &b.naq, &b.nak,
      &b.img_fc1_b, &b.img_fc2_b, &b.txt_fc1_b, &b.txt_fc2_b};
  for (metal_compute::SharedBuffer* q2 : p) { one(*q2); }
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
MetalQwenImageTransformer::wire_fixed_(bool on)
{
  if (!_ws) { return 0; }
  std::size_t changed = 0;
  _ws->for_each_weight([&](metal_compute::SharedBuffer& b) {
    changed += _wire.wire_one(_mc, b, on);
  });
  return changed;
}

std::size_t
MetalQwenImageTransformer::block_bytes_(const Block& b)
{
  return qw_bytes_(b.img_mod_w) + qw_bytes_(b.txt_mod_w) +
         b.img_mod_b.byte_size() + b.txt_mod_b.byte_size() +
         qw_bytes_(b.qw) + qw_bytes_(b.kw) + qw_bytes_(b.vw) +
         qw_bytes_(b.ow) + b.qb.byte_size() + b.kb.byte_size() +
         b.vb.byte_size() + b.ob.byte_size() +
         qw_bytes_(b.aqw) + qw_bytes_(b.akw) + qw_bytes_(b.avw) +
         qw_bytes_(b.aow) + b.aqb.byte_size() + b.akb.byte_size() +
         b.avb.byte_size() + b.aob.byte_size() +
         b.nq.byte_size() + b.nk.byte_size() + b.naq.byte_size() +
         b.nak.byte_size() +
         qw_bytes_(b.img_fc1_w) + qw_bytes_(b.img_fc2_w) +
         qw_bytes_(b.txt_fc1_w) + qw_bytes_(b.txt_fc2_w) +
         b.img_fc1_b.byte_size() + b.img_fc2_b.byte_size() +
         b.txt_fc1_b.byte_size() + b.txt_fc2_b.byte_size();
}

void
MetalQwenImageTransformer::resident_pages_(std::size_t* examined,
                                           std::size_t* incore,
                                           std::size_t* paged_out) const
{
  *examined = 0;
  *incore = 0;
  if (paged_out != nullptr) { *paged_out = 0; }
  // The DOMINANT buffers only. A block's bytes are almost all in the
  // projections; walking the norms too would double the syscalls to
  // sharpen a fraction the answer does not turn on.
  for (const Block& b : _blocks) {
    const metal_compute::SharedBuffer* all[] = {
        &b.qw.w,  &b.qw.codes,  &b.kw.w,  &b.kw.codes,
        &b.vw.w,  &b.vw.codes,  &b.ow.w,  &b.ow.codes,
        &b.img_fc1_w.w, &b.img_fc1_w.codes,
        &b.img_fc2_w.w, &b.img_fc2_w.codes,
        &b.txt_fc1_w.w, &b.txt_fc1_w.codes,
        &b.txt_fc2_w.w, &b.txt_fc2_w.codes};
    for (const metal_compute::SharedBuffer* p : all) {
      if (p->byte_size() == 0) { continue; }
      // A WIRED BUFFER CANNOT HAVE LEFT RAM, so asking is spending the
      // walk to be told what mlock already guarantees. Skipped PER BUFFER
      // rather than per block, because wire_block_ stops at the first
      // refusal and leaves the rest of that block unwired -- the
      // remainder is exactly what still needs measuring. With everything
      // wired `examined` stays 0, which the caller reads as "no evidence"
      // rather than as a shortfall, and that is the correct answer.
      if (p->is_wired()) { continue; }
      const auto r = p->page_residency(64);
      if (!r.valid) { continue; }
      *examined += r.examined;
      *incore += r.incore;
      if (paged_out != nullptr) { *paged_out += r.paged_out; }
    }
  }
}

std::size_t
MetalQwenImageTransformer::evict_tail_block_()
{
  for (int i = (int)_blocks.size() - 1; i >= 0; --i) {
    Block& b = _blocks[(std::size_t)i];
    const std::size_t n = block_bytes_(b);
    if (n == 0) { continue; }
    // Before the buffers go: give the wiring back. Dropping a wired
    // buffer unwires it in the kernel anyway, but only unwire_from_pool()
    // decrements the pool's counter -- so doing it here is what keeps the
    // budget honest instead of leaking a block's worth per eviction.
    _wire.note_unwired(wire_block_(b, false));
    b = Block{};
    // Taking one out of the pinned prefix UN-pins it: that prefix was
    // sized against what the box was believed to hold, and a measurement
    // saying its pages have left RAM is that belief being wrong.
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
MetalQwenImageTransformer::wire_retry_slack_() const
{
  if (_wire_block_hint > 0) { return _wire_block_hint; }
  if (_resid.count() > 0) {
    return _resid.bytes() / (std::size_t)_resid.count();
  }
  return 0;
}

void
MetalQwenImageTransformer::set_residency_schedule(int steps)
{
  if (!_ws) { return; }
  const MetalLlamaWeights& src = _ws->src();
  const std::size_t blk = widest_block_bytes(
      src.tensor_names(),
      [&](const std::string& n) {
        const auto* ti = src.info(n);
        return ti != nullptr ? (std::size_t)ti->nbytes : (std::size_t)0;
      },
      {"transformer_blocks."});
  const int nl = _cfg.n_layers;
  _wire_block_hint = blk;
  _resid.set_schedule(steps, nl, blk, _wire.on(),
                      _mc != nullptr ? _mc->memory_budget()
                                     : metal_compute::MetalCompute::
                                           MemoryBudget{});
  if (_mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalQwenImageTransformer: residency probe {} blocks of {} "
        "({} MB each, {} MB reclaimable, wire budget {} MB){}",
        _resid.per_forward_cap(), nl, blk >> 20,
        _mc->memory_budget().available_physical >> 20,
        _wire.budget() >> 20,
        _wire.on() ? " -- uncapped, the wire budget is the gate"
                   : ", doubling per healthy forward"));
  }
}

std::size_t
MetalQwenImageTransformer::release_resident_blocks(std::size_t bytes)
{
  const std::size_t freed =
      _resid.release(bytes, [this]() -> std::size_t {
        return evict_tail_block_();
      });
  if (freed > 0 && _mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalQwenImageTransformer: released {} MB of resident blocks "
        "({} left)", freed >> 20, _resid.count()));
  }
  return freed;
}

bool
MetalQwenImageTransformer::load_block_(WeightSet& ws, int L,
                                       Block& b, Retain r)
{
  const std::string p = "transformer_blocks." + std::to_string(L) + ".";
  bool ok = true;
  // AdaLN modulation: dense bf16 by default, or affine-quantized when the
  // checkpoint was built with model-quantize quant_modulation (load_linear_q_
  // auto-detects via the presence of *_mod.1.scales).
  ok = ok && load_linear_q_(ws, p + "img_mod.1", b.img_mod_w, b.img_mod_b, r);
  ok = ok && load_linear_q_(ws, p + "txt_mod.1", b.txt_mod_w, b.txt_mod_b, r);
  ok = ok && load_linear_q_(ws, p + "attn.to_q", b.qw, b.qb, r);
  ok = ok && load_linear_q_(ws, p + "attn.to_k", b.kw, b.kb, r);
  ok = ok && load_linear_q_(ws, p + "attn.to_v", b.vw, b.vb, r);
  ok = ok && load_linear_q_(ws, p + "attn.to_out.0", b.ow, b.ob, r);
  ok = ok && load_linear_q_(ws, p + "attn.add_q_proj", b.aqw, b.aqb, r);
  ok = ok && load_linear_q_(ws, p + "attn.add_k_proj", b.akw, b.akb, r);
  ok = ok && load_linear_q_(ws, p + "attn.add_v_proj", b.avw, b.avb, r);
  ok = ok && load_linear_q_(ws, p + "attn.to_add_out", b.aow, b.aob, r);
  b.nq  = to_elt_(ws, p + "attn.norm_q.weight", r);
  b.nk  = to_elt_(ws, p + "attn.norm_k.weight", r);
  b.naq = to_elt_(ws, p + "attn.norm_added_q.weight", r);
  b.nak = to_elt_(ws, p + "attn.norm_added_k.weight", r);
  ok = ok && !b.nq.empty() && !b.nk.empty() && !b.naq.empty() && !b.nak.empty();
  ok = ok && load_linear_q_(ws, p + "img_mlp.net.0.proj", b.img_fc1_w,
                            b.img_fc1_b, r);
  ok = ok && load_linear_q_(ws, p + "img_mlp.net.2", b.img_fc2_w,
                            b.img_fc2_b, r);
  ok = ok && load_linear_q_(ws, p + "txt_mlp.net.0.proj", b.txt_fc1_w,
                            b.txt_fc1_b, r);
  ok = ok && load_linear_q_(ws, p + "txt_mlp.net.2", b.txt_fc2_w,
                            b.txt_fc2_b, r);
  return ok;
}

std::unique_ptr<MetalQwenImageTransformer>
MetalQwenImageTransformer::load(const std::string& model_dir, MetalCompute* mc,
                                const Config& cfg, bool stream_blocks)
{
  return load(WeightSet::open(model_dir, nullptr), mc, cfg, stream_blocks);
}

std::unique_ptr<MetalQwenImageTransformer>
MetalQwenImageTransformer::load(std::shared_ptr<WeightSet> ws_in,
                                MetalCompute* mc, const Config& cfg,
                                bool stream_blocks)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  const std::string model_dir = ws_in->dir();

  auto m = std::unique_ptr<MetalQwenImageTransformer>(
      new MetalQwenImageTransformer());
  m->_ws = std::move(ws_in);
  WeightSet& ws = *m->_ws;
  // Everything loaded from here to the end of load() is RETAINED for the
  // model's life. The streamed blocks are read in forward(), and there
  // only.
  const Retain r = Retain::Cached;
  m->_mc = mc;
  m->_cfg = cfg;
  m->_stream_blocks = stream_blocks;
  // Join the manager's wired pool. Nothing to decide about residency here
  // -- this class already reads every weight Copied (see load_qw_) -- so
  // what this buys is that a kept block can be mlock'd instead of being
  // the first thing the compressor takes. See shared/wired-pool.h.
  m->_wire.open(mc);
  // The reusable read destinations. Configured whether or not this run
  // streams: they cost nothing until the first streamed block asks for
  // one, and a model that preloads never gets there.
  m->configure_slots_();

  // Affine group-quant detection: config.json `quantization {bits, group_size}`
  // (written by model-quantize target=dit). Absent => dense bf16.
  {
    namespace fs = std::filesystem;
    std::ifstream in(fs::path(model_dir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto o = fd.as_object();
        // zero_cond_t: modulate the reference tokens at timestep 0 (see the
        // Config comment). Qwen-Image-Edit-2511 sets it.
        if (o.contains("zero_cond_t")) {
          m->_cfg.zero_cond_t = o.at("zero_cond_t").as_bool(false);
        }
        if (o.contains("quantization")) {
          FlexData q = o.at("quantization");
          if (q.is_object()) {
            auto qo = q.as_object();
            const int b = qo.contains("bits") ? (int)qo.at("bits").as_int(0) : 0;
            const int g = qo.contains("group_size")
                ? (int)qo.at("group_size").as_int(64) : 64;
            if (b == 4 || b == 8) {
              m->_quant_bits  = b;
              m->_quant_group = (g == 32 || g == 64) ? g : 64;
            }
          }
        }
      }
    }
  }

  // bf16 metallib variants (the residual stream exceeds f16 range).
  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_lib_sdpa = mc->load_library("sdpa_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_fn_gemm      = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_gemm_bias = m->_lib_gemm.function("dense_gemm_bias_f16");
  // Fast-path twins (best-effort): BM=64 tile for tall-M block GEMMs + a GEMV
  // for the M=1 conditioning/modulation rows (bias-less -> bias_add_rows_f16).
  m->_fn_gemm_bm64 = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_gemv      = m->_lib_gemm.function("dense_gemv_t_f16");
  m->_fn_bias_add  = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_rms       = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_layernorm = m->_lib_elt.function("layer_norm_plain_f16");
  m->_fn_silu      = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_gelu      = m->_lib_elt.function("gelu_tanh_ff_f16");
  m->_fn_residual  = m->_lib_elt.function("residual_add_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_sdpa      = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_rope_table = m->_lib_rope.function("rope_pair_table_ftab_f16");
  if (std::getenv("VPIPE_QIE_NO_FUSE_ROPE") == nullptr) {
    m->_fn_transpose_rope =
        m->_lib_rope.function("transpose_rope_pair_ftab_f16");
  }
  m->_fn_adaln     = m->_lib_elt.function("adaln_modulate_f16");
  m->_fn_gated     = m->_lib_elt.function("gated_residual_f16");
  // vec4 twins: same arithmetic per element (bit-identical), 3-4x throughput --
  // one element per thread leaves these at ~37-54 GB/s where the same bytes
  // through a vec4 2-D grid run at 143-181. Serves Qwen-Image-Edit AND
  // Mage-Flow (same class, different Config). VPIPE_NO_ELT_V4 reverts.
  if (std::getenv("VPIPE_NO_ELT_V4") == nullptr) {
    m->_fn_adaln4 = m->_lib_elt.function("adaln_modulate_v4_f16");
    m->_fn_gated4 = m->_lib_elt.function("gated_residual_v4_f16");
  }
  m->_fn_colabsmax = m->_lib_elt.function("col_absmax_f16");
  if (!m->_fn_gemm.valid() || !m->_fn_gemm_bias.valid() || !m->_fn_rms.valid() ||
      !m->_fn_layernorm.valid() || !m->_fn_silu.valid() || !m->_fn_gelu.valid() ||
      !m->_fn_residual.valid() || !m->_fn_transpose.valid() ||
      !m->_fn_sdpa.valid() || !m->_fn_rope_table.valid() ||
      !m->_fn_adaln.valid() || !m->_fn_gated.valid()) {
    return nullptr;
  }

  // Steel register-resident flash attention (bf16, head_dim 128) for the joint
  // attention. Best-effort: if the library / entry point is missing, forward()
  // keeps the scalar sdpa fallback. VPIPE_QIE_NO_STEEL_ATTN forces scalar.
  m->_lib_attn = mc->load_library("attn_steel");
  m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));
  m->_steel_attn_ok = m->_lib_attn.valid() && !m->_attn_params.empty() &&
                      cfg.head_dim == 128 &&
                      std::getenv("VPIPE_QIE_NO_STEEL_ATTN") == nullptr;

  // Affine qmm kernels (bf16 variant) -- only when the checkpoint is quantized.
  if (m->_quant_bits > 0) {
    m->_lib_qmm = mc->load_library("affine_qmm_steel_bf16");
    const std::string g = "g" + std::to_string(m->_quant_group);
    m->_fn_qmm4     = m->_lib_qmm.function("affine_qmm_steel_w4" + g);
    m->_fn_qmm8     = m->_lib_qmm.function("affine_qmm_steel_w8" + g);
    m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    const bool need8 = (m->_quant_bits == 8);
    if (!m->_fn_bias_add.valid() || !m->_fn_qmm4.valid() ||
        (need8 && !m->_fn_qmm8.valid())) {
      return nullptr;
    }
    // Wider steel-qmm tiles for the tall-M DiT block GEMMs (M ~= seq): BM=64,
    // then BM=128 at high resolution. Same buffer layout + f32 accumulate as the
    // base tile, so token-exact -- only the output-row tiling differs. On M4 the
    // QIE w4 GEMMs are compute/dequant-bound (not weight-BW-bound like the
    // Krea-2 / FLUX.2 cases), so BM=64 is a small win but BM=128 tends to
    // over-tile (lower occupancy) at QIE's dims -- default BM=64. 0 disables
    // (base BM=32); g32 falls back to base. VPIPE_QIE_QMM_TILE overrides (0/1/2).
    m->_qmm_tile = 1;
    if (const char* t = std::getenv("VPIPE_QIE_QMM_TILE")) {
      m->_qmm_tile = std::atoi(t);
    }
    if (m->_qmm_tile >= 1) {
      m->_fn_qmm4_bm64 = m->_lib_qmm.function("affine_qmm_steel_w4" + g + "_bm64");
      m->_fn_qmm8_bm64 = m->_lib_qmm.function("affine_qmm_steel_w8" + g + "_bm64");
      if (!m->_fn_qmm4_bm64.valid() || (need8 && !m->_fn_qmm8_bm64.valid())) {
        m->_qmm_tile = 0;
      }
    }
    if (m->_qmm_tile == 2) {
      m->_fn_qmm4_bm128 =
          m->_lib_qmm.function("affine_qmm_steel_w4" + g + "_bm128");
      m->_fn_qmm8_bm128 =
          m->_lib_qmm.function("affine_qmm_steel_w8" + g + "_bm128");
      if (!m->_fn_qmm4_bm128.valid() || (need8 && !m->_fn_qmm8_bm128.valid())) {
        m->_qmm_tile = 1;   // BM128 unavailable (e.g. g32) -> stay on BM64
      }
    }
    // Peak M=1 quantized GEMV for the per-block modulation projection (used when
    // the mod weights are quantized -- e.g. w8 mod from model-quantize). Streams
    // the codes at ~peak bandwidth instead of the steel qmm tile's 31/32-wasted
    // M=1 rows. Best-effort: null -> gemm_bias_q fallback. VPIPE_QIE_NO_MOD_QMV
    // disables.
    if (std::getenv("VPIPE_QIE_NO_MOD_QMV") == nullptr) {
      m->_lib_qmv = mc->load_library("affine_qmv_bf16");
      m->_fn_qmv4 = m->_lib_qmv.function("affine_qmv_w4" + g);
      m->_fn_qmv8 = m->_lib_qmv.function("affine_qmv_w8" + g);
    }
  }

  // M5 matrix-core matmul2d for the block/projection GEMMs (the *_bf16 metallib
  // variants, since the DiT runs bf16). Dense weights feed dense_gemm_mma
  // directly; quantized weights dequant-expand into _w_deq (affine_dequant) then
  // run through the SAME dense matmul2d -- the dequant-once -> dense-GEMM shape
  // that matches affine_qmm_steel to f32 rounding. Gated on matrix cores
  // (M4/older keep steel); VPIPE_QIE_NO_MMA2 A/B off.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_QIE_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    // TN=2 tile (the K=6144..12288 band; QIE has no K there, but load for
    // parity). NO_TN2 forces the plain deep tile (A/B).
    if (std::getenv("VPIPE_QIE_NO_TN2") == nullptr) {
      m->_fn_dense_mma_tn2 =
          m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");
    }
    m->_fn_dense_mma_splitk =
        m->_lib_dense_mma.function("dense_gemm_mma_splitk_n128x256_k8192_f16");
    // matmul2d has no bias slot -> a row-broadcast bias_add folds it. Load it
    // even for a dense checkpoint (the quant block above loads it only when
    // quantized, for the steel qmm path).
    m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid() &&
                   m->_fn_bias_add.valid();
    // Quantized checkpoint: group-matched dequant kernels feed the dense
    // matmul2d. Both bit widths loaded (mixed-precision per-weight w4/w8).
    if (m->_use_mma2 && m->_quant_bits > 0) {
      m->_lib_dequant = mc->load_library("affine_dequant_bf16");
      const std::string g = "g" + std::to_string(m->_quant_group);
      m->_fn_dequant4 = m->_lib_dequant.function("affine_dequant_w4" + g);
      m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant_w8" + g);
      if (!m->_fn_dequant4.valid() || !m->_fn_dequant8.valid()) {
        m->_use_mma2 = false;   // fall back to steel qmm
      }
    }
    if (const char* e = std::getenv("VPIPE_QIE_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    m->_use_splitk = m->_use_mma2 && m->_fn_dense_mma_splitk.valid()
                     && m->_fn_residual.valid()
                     && std::getenv("VPIPE_QIE_NO_SPLITK") == nullptr;
  }

  // Top-level weights.
  bool ok = m->load_linear_(ws, "img_in", m->_img_in_w, m->_img_in_b, r);
  m->_txt_norm_w = m->to_elt_(ws, "txt_norm.weight", r);
  ok = ok && !m->_txt_norm_w.empty();
  ok = ok && m->load_linear_(ws, "txt_in", m->_txt_in_w, m->_txt_in_b, r);
  ok = ok && m->load_linear_(
      ws, "time_text_embed.timestep_embedder.linear_1", m->_t1_w,
      m->_t1_b, r);
  ok = ok && m->load_linear_(
      ws, "time_text_embed.timestep_embedder.linear_2", m->_t2_w,
      m->_t2_b, r);
  ok = ok && m->load_linear_(ws, "norm_out.linear", m->_normout_w,
                             m->_normout_b, r);
  ok = ok && m->load_linear_(ws, "proj_out", m->_projout_w, m->_projout_b, r);
  if (!ok) { return nullptr; }

  // Blocks: preload all 60 (default), or -- in streaming mode -- skip the
  // preload so forward() reads/frees each block from the weight set on
  // demand (memory-bounded, for 16GB boxes).
  if (!stream_blocks) {
    m->_blocks.resize((std::size_t)cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; ++L) {
      if (!m->load_block_(ws, L, m->_blocks[(std::size_t)L], r)) {
        return nullptr;
      }
    }
  } else {
    // Streaming preloads NOTHING. Sized to the FULL depth all the same:
    // the empty slots are where forward() promotes streamed blocks as
    // free memory allows. An unfilled slot reads as empty, which is what
    // `held` tests.
    m->_blocks.resize((std::size_t)cfg.n_layers);
  }
  return m;
}

// ---- AWQ calibration --------------------------------------------------------
void
MetalQwenImageTransformer::calib_begin()
{
  const int nL = _cfg.n_layers, H = _cfg.hidden, FF = _cfg.ffn;
  _calib_acc.clear();
  // dim = hidden for qkv/o/fc1 inputs, ffn for fc2 inputs (GELU output).
  const std::pair<const char*, int> groups[] = {
      {"img_qkv", H}, {"txt_qkv", H}, {"img_o", H}, {"txt_o", H},
      {"img_fc1", H}, {"txt_fc1", H}, {"img_fc2", FF}, {"txt_fc2", FF}};
  for (const auto& g : groups) {
    SharedBuffer b =
        _mc->make_shared_buffer((std::size_t)nL * g.second * 2);
    std::memset(b.contents(), 0, b.byte_size());   // zero before the first tap
    _calib_acc[g.first] = std::move(b);
  }
  _calib_on = true;
}

std::map<std::string, std::vector<float>>
MetalQwenImageTransformer::calib_stats() const
{
  const int nL = _cfg.n_layers, H = _cfg.hidden, FF = _cfg.ffn;
  std::map<std::string, std::vector<float>> out;
  for (const auto& kv : _calib_acc) {
    const int dim = (kv.first == "img_fc2" || kv.first == "txt_fc2") ? FF : H;
    std::vector<float> v((std::size_t)nL * dim);
    const auto* s = static_cast<const std::uint16_t*>(kv.second.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { v[i] = bf16_to_f32_(s[i]); }
    out[kv.first] = std::move(v);
  }
  return out;
}

std::vector<float>
MetalQwenImageTransformer::time_proj_(float sigma) const
{
  // diffusers Timesteps(num_channels=256, flip_sin_to_cos=True,
  // downscale_freq_shift=0, scale=1000): arg = sigma*1000 * exp(-ln(1e4)*i/128),
  // emb = [cos(arg), sin(arg)] (flip_sin_to_cos).
  const int C = _cfg.time_proj, half = C / 2;
  std::vector<float> out((std::size_t)C);
  const float sc = 1000.0f;
  // Round-to-nearest-even down to bf16 (what torch's .to(bfloat16) does).
  auto to_bf16 = [](float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float o;
    std::memcpy(&o, &u, 4);
    return o;
  };
  // Mage-Flow's forward does `timesteps = timesteps.to(img.dtype)` with the
  // model in bf16, so the TIMESTEP is rounded before the angle is formed --
  // the same trap as the frequency table below, and just as consequential:
  // the angle reaches sigma*1000 ~ 950 rad, so bf16's ~2e-3 ulp near 1.0 is
  // ~2 RADIANS of phase and moves temb by 30-40%. Only sigmas that are
  // exactly representable in bf16 (1.0, 0.75, 0.5) are unaffected -- which is
  // why a golden pinned at sigma 0.75 cannot see this, while the real
  // FlowMatchEuler schedule (0.947, 0.857, 0.667) is wrong at every step but
  // the first.
  if (_cfg.bf16_timestep) { sigma = to_bf16(sigma); }
  for (int i = 0; i < half; ++i) {
    float freq = std::exp(-std::log(10000.0f) * (float)i / (float)half);
    if (_cfg.bf16_time_freqs) { freq = to_bf16(freq); }
    const float arg = sigma * sc * freq;
    out[(std::size_t)i] = std::cos(arg);
    out[(std::size_t)(half + i)] = std::sin(arg);
  }
  return out;
}

void
MetalQwenImageTransformer::build_rope_(int txt_seq,
                                       const std::vector<ImgSeg>& segs,
                                       SharedBuffer& cos_out,
                                       SharedBuffer& sin_out)
{
  const int D = _cfg.head_dim;          // 128
  const int P = D / 2;                  // 64 pairs
  const int a0 = _cfg.axes[0], a1 = _cfg.axes[1], a2 = _cfg.axes[2];  // 16,56,56
  const int p0 = a0 / 2, p1 = a1 / 2;   // pairs per axis: 8, 28, (28)
  int total_img = 0, max_vid = 0;
  for (const ImgSeg& s : segs) {
    total_img += s.seq;
    max_vid = std::max(max_vid, std::max(s.grid_h / 2, s.grid_w / 2));
  }
  const int T = txt_seq + total_img;
  cos_out = _mc->make_shared_buffer((std::size_t)T * D * 4);   // f32 tables
  sin_out = _mc->make_shared_buffer((std::size_t)T * D * 4);
  auto* cb = static_cast<float*>(cos_out.contents());
  auto* sb = static_cast<float*>(sin_out.contents());

  // invfreq for pair j (per axis): theta^(-2*local/axis_dim).
  auto invfreq = [&](int j) -> double {
    int axis_dim, local;
    if (j < p0) { axis_dim = a0; local = j; }
    else if (j < p0 + p1) { axis_dim = a1; local = j - p0; }
    else { axis_dim = a2; local = j - p0 - p1; }
    return std::pow((double)_cfg.rope_theta,
                    -2.0 * (double)local / (double)axis_dim);
  };
  // Fill row `t` with the 3-axis angles from (pf, ph, pw) positions.
  auto fill = [&](int t, double pf, double ph, double pw) {
    for (int j = 0; j < P; ++j) {
      double pos = (j < p0) ? pf : (j < p0 + p1 ? ph : pw);
      const double ang = pos * invfreq(j);
      const float c = (float)std::cos(ang), s = (float)std::sin(ang);
      const std::size_t o = (std::size_t)t * D + 2 * j;
      cb[o] = c; cb[o + 1] = c; sb[o] = s; sb[o + 1] = s;
    }
  };
  // Text rows first: position = max_vid + tt on all axes. When the model
  // leaves text unrotated (Mage-Flow), write the identity rotation instead
  // -- cos 1 / sin 0 -- so the shared roped-attention path needs no branch.
  for (int tt = 0; tt < txt_seq; ++tt) {
    if (!_cfg.rotate_txt) {
      for (int j = 0; j < P; ++j) {
        const std::size_t o = (std::size_t)tt * D + 2 * j;
        cb[o] = 1.0f; cb[o + 1] = 1.0f; sb[o] = 0.0f; sb[o + 1] = 0.0f;
      }
      continue;
    }
    const double p = (double)(max_vid + tt);
    fill(tt, p, p, p);
  }
  // Then each image segment: frame band + centered height/width.
  int row = txt_seq;
  for (const ImgSeg& s : segs) {
    const int gh = s.grid_h, gw = s.grid_w;
    const int hoff = gh - gh / 2, woff = gw - gw / 2;   // centering offsets
    for (int r = 0; r < gh; ++r) {
      for (int c = 0; c < gw; ++c) {
        fill(row++, (double)s.frame, (double)(r - hoff), (double)(c - woff));
      }
    }
  }
}

bool
MetalQwenImageTransformer::gemm_mma_(ComputeEncoder& enc,
                                     const SharedBuffer& xin, std::size_t xe,
                                     const QWeight& w, const SharedBuffer& y,
                                     std::size_t ye, int M, int N, int K)
{
  // Matrix-core matmul2d only when present, M amortizes the 128-row tile, and N
  // is non-degenerate. Otherwise the caller keeps its steel path.
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
    const std::size_t need = (std::size_t)N * K * 2;   // bf16
    if (_w_deq.empty() || _w_deq.byte_size() < need) {
      _w_deq = _mc->make_shared_buffer(need);
      if (_w_deq.empty()) { return false; }
    }
    // codes/scales/qbias -> _w_deq[N,K] (one thread per packed u32 word: w4 has
    // 8 nibbles/word so K/8 words, w8 has 4 bytes/word so K/4). Serial ordering
    // + Metal's WAR hazard tracking make the shared _w_deq safe to reuse across
    // a block's GEMMs (each dequant->matmul pair runs before the next writes).
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
  // Split-K deep-reduction path (K a multiple of kSplitKC, >= 2 chunks): each
  // split gets its own threadgroup plane (grid.z); a residual_add fold sums the
  // planes into y. QIE's dims never trip this (kept for parity); falls through
  // to the single-op path when the scratch alloc fails or the shape doesn't
  // split.
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
      enc.set_buffer(0, xin, xe * 2); enc.set_buffer(1, *wdense);
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
  // Tile-adaptive dense matmul2d (no bias); matmul2d tensor extents clamp the
  // M/N tails. 128x128 for K < 6144 (QIE qkv/out/fc1, K=3072/3584); the plain
  // 128x256 deep tile for K >= 12288 (ff-down); the TN=2 tile fills the 6144..
  // 12288 band (unused by QIE's dims).
  int RN = 256;   // effective N-region per tg (TN*BN); grid divides N by it
  const metal_compute::ComputeFunction* fn = &_fn_dense_mma_deep;
  if (K < 6144) {
    fn = &_fn_dense_mma; RN = 128;
  } else if (K < 12288 && _fn_dense_mma_tn2.valid()) {
    fn = &_fn_dense_mma_tn2; RN = 512;
  }
  enc.set_function(*fn);
  enc.set_buffer(0, xin, xe * 2);
  enc.set_buffer(1, *wdense); enc.set_buffer(2, *wdense);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  return true;
}

SharedBuffer
MetalQwenImageTransformer::forward(const SharedBuffer& hidden, int gen_seq,
                                   const SharedBuffer& txt, int txt_seq,
                                   int grid_h, int grid_w, float sigma,
                                   const std::vector<RefImage>& refs,
                                   int stop_after_block)
{
  const int H = _cfg.hidden, Hd = _cfg.head_dim, NH = _cfg.n_heads;
  const int IC = _cfg.in_channels, FF = _cfg.ffn;
  const float eps = _cfg.norm_eps;

  // Image segments: generated grid (frame 0) + each reference (frame idx).
  std::vector<ImgSeg> segs;
  segs.push_back({0, grid_h, grid_w, gen_seq});
  int total_img = gen_seq, frame = 1;
  for (const RefImage& r : refs) {
    segs.push_back({frame++, r.grid_h, r.grid_w, r.seq});
    total_img += r.seq;
  }
  const int JT = txt_seq + total_img;   // joint sequence length

  // LLM-lane perf event (perf-visualizer): one dual-stream DiT forward per
  // sampler step (2 per step under CFG). value = the joint sequence length.
  // Mirrors the Krea-2 / FLUX.2 DiT events so QIE shows on the LLM lane too.
  PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDit,
                     kPerfLlmDitBegin, (std::uint64_t)JT);

  auto buf = [&](std::size_t n) { return _mc->make_shared_buffer(n * 2); };

  // Persistent host-built inputs.
  const std::vector<float> tproj = time_proj_(sigma);
  SharedBuffer tproj_b = buf((std::size_t)_cfg.time_proj);
  { auto* d = static_cast<std::uint16_t*>(tproj_b.contents());
    for (int i = 0; i < _cfg.time_proj; ++i) { d[i] = f32_to_bf16_(tproj[i]); } }
  SharedBuffer rcos, rsin;
  build_rope_(txt_seq, segs, rcos, rsin);

  // Stream-persistent scratch.
  SharedBuffer x_img = buf((std::size_t)total_img * H);   // image stream
  SharedBuffer x_txt = buf((std::size_t)txt_seq * H);     // text stream
  SharedBuffer temb = buf((std::size_t)H), tsilu = buf((std::size_t)H);
  SharedBuffer th1 = buf((std::size_t)H);
  // Modulation scratch [6H] (per-block when not precomputing; also the final
  // norm_out's [2H] scratch). The block loop shadows these with per-block views
  // into imod_all/tmod_all when modulation is precomputed (preloaded path).
  SharedBuffer imod_s = buf((std::size_t)6 * H), tmod_s = buf((std::size_t)6 * H);
  SharedBuffer msilu = buf((std::size_t)H);
  // zero_cond_t twins: the timestep-0 embedding and its image modulation, used
  // for the reference rows only. Allocated only when references are present.
  const int ref_seq = total_img - gen_seq;
  const bool want_zct = _cfg.zero_cond_t && ref_seq > 0;
  SharedBuffer tproj0_b, temb0, msilu0, zmod_s;
  if (want_zct) {
    tproj0_b = buf((std::size_t)_cfg.time_proj);
    temb0 = buf((std::size_t)H);
    msilu0 = buf((std::size_t)H);
    zmod_s = buf((std::size_t)6 * H);
  }
  SharedBuffer nrm = buf((std::size_t)JT * H), mdl = buf((std::size_t)JT * H);
  SharedBuffer jq = buf((std::size_t)JT * H), jk = buf((std::size_t)JT * H),
               jv = buf((std::size_t)JT * H);
  SharedBuffer qt = buf((std::size_t)JT * H), kt = buf((std::size_t)JT * H),
               vt = buf((std::size_t)JT * H), at = buf((std::size_t)JT * H);
  SharedBuffer att = buf((std::size_t)JT * H);
  SharedBuffer g1 = buf((std::size_t)total_img * FF),
               g2 = buf((std::size_t)txt_seq * FF);
  SharedBuffer velo = buf((std::size_t)gen_seq * IC);
  // Staged-verification sentinels (alongside the existing -2 post-embedder
  // and -3 post-attention-residual): -4 returns block 0's joint attention
  // output for the image rows BEFORE the output projection and gate, and -5
  // returns the timestep conditioning vector temb. Together they split a
  // mismatch into conditioning / attention / out-proj+gate -- which is how
  // the Mage-Flow bf16-frequency bug was localized.
  bool dbg_att = false;          // stop_after_block == -4 fired
  std::size_t dbg_ioff = 0;

  // Env-gated per-section GPU timing (VPIPE_QIE_DIT_PROFILE). Each section
  // boundary inserts a commit+wait barrier and accumulates wall time, splitting
  // the deferred stream into timed slices. Preloaded path only (the streaming /
  // pinned path already flushes per block); the barriers serialize the GPU so
  // absolute step time inflates a little, but the RELATIVE breakdown is
  // faithful. Mirrors the Krea-2 / FLUX.2 DiT profilers.
  const bool prof = !_stream_blocks &&
                    (std::getenv("VPIPE_QIE_DIT_PROFILE") != nullptr
                     // Mage-Flow drives this same class, so accept a
                     // Mage-named alias -- section timing for a Mage run
                     // should not hide behind a QIE-named variable.
                     || std::getenv("VPIPE_MAGE_DIT_PROFILE") != nullptr);
  // Measurement-only: skip the modulation GEMVs to isolate their bandwidth cost
  // (imod/tmod left stale -> output is garbage; timing only). VPIPE_QIE_SKIP_MOD.
  const bool skip_mod = std::getenv("VPIPE_QIE_SKIP_MOD") != nullptr;
  double t_embed = 0, t_mod = 0, t_norm = 0, t_qkv = 0, t_attn = 0,
         t_oproj = 0, t_ff = 0, t_final = 0;
  std::chrono::steady_clock::time_point mark;

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // Commit the current encoder + start a fresh one (streaming mode): forces
    // the block's GPU work to complete before its weights are freed, so only
    // ~one block is ever resident. `enc`/`stream` are captured by reference by
    // every lambda below, so reassigning them here is transparent.
    auto flush = [&]() {
      enc.end();
      stream.commit().wait();
      stream = _mc->make_command_stream();
      enc = stream.begin_compute();
    };
    // Profiling barrier: commit+wait the accumulated ops, add the elapsed slice
    // to `acc`, reopen the stream, reset the mark. No-op unless profiling.
    auto psplit = [&](double& acc) {
      if (!prof) { return; }
      enc.end();
      stream.commit().wait();
      acc += std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - mark).count();
      stream = _mc->make_command_stream();
      enc = stream.begin_compute();
      mark = std::chrono::steady_clock::now();
    };
    // ---- kernel helper lambdas (mirror the Krea toolbox contracts) ----
    auto gemm_bias = [&](const SharedBuffer& x, std::size_t xe,
                         const SharedBuffer& w, const SharedBuffer& b,
                         const SharedBuffer& y, std::size_t ye, int M, int N,
                         int K) {
      // M == 1 (conditioning / modulation / head rows): a GEMV avoids the steel
      // tile's 31/32 wasted rows. dense_gemv_t is bias-less -> a row bias-add
      // follows (all QIE gemm_bias sites carry a bias).
      if (M == 1 && _fn_gemv.valid() && _fn_bias_add.valid()) {
        enc.set_function(_fn_gemv);
        enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w);
        enc.set_buffer(2, y, ye * 2);
        enc.set_constant(3, K); enc.set_constant(4, N);
        enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
        enc.set_function(_fn_bias_add);
        enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, b);
        enc.set_constant(2, N); enc.set_constant(3, (unsigned)N);
        enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
        return;
      }
      // M > 1: steel MMA dense GEMM (simdgroup-matrix, f32 accumulate) -- the
      // fast path, same machinery as affine_qmm_steel minus the dequant. Bias is
      // folded in-kernel (buffer 2 + has_bias). Prefer the BM=64 tile: it
      // computes 64 output rows per threadgroup with the same 128 threads, so
      // each [BN,BK] weight tile is re-read M/64 times instead of M/32 --
      // halving the f16 weight bandwidth for tall M (and never worse for small
      // M, where these large-N/K GEMMs are weight-bandwidth-bound anyway). NOTE
      // the constant order is 4:K 5:N 6:M (the scalar dense_gemm_bias_f16
      // fallback uses 4:M 5:N 6:K). Every QIE linear has K % 32 == 0; M/N tails
      // are handled in-kernel. The scalar 16x16 kernel is a naive
      // one-thread-per-output tiled GEMM (kept only as a fallback).
      const bool bm64 = _fn_gemm_bm64.valid();
      if (bm64 || _fn_gemm.valid()) {
        const int BM = bm64 ? 64 : 32;
        enc.set_function(bm64 ? _fn_gemm_bm64 : _fn_gemm);
        enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w); enc.set_buffer(2, b);
        enc.set_buffer(3, y, ye * 2);
        enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
        enc.set_constant(7, 1);
        enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                      (unsigned)(((M + BM - 1) / BM) * 2), 2}, {32, 2, 2});
        return;
      }
      enc.set_function(_fn_gemm_bias);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w); enc.set_buffer(2, b);
      enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
      enc.set_constant(7, 1);
      enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                    (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
    };
    // Quant-aware Linear+bias for the leaf set: dense -> gemm_bias; quantized ->
    // affine_qmm_steel (bias-less) then a row-broadcast bias add.
    auto gemm_bias_q = [&](const SharedBuffer& x, std::size_t xe,
                           const QWeight& w, const SharedBuffer& b,
                           const SharedBuffer& y, std::size_t ye, int M, int N,
                           int K) {
      // M5 matmul2d (NAX): a biasless matmul (dense direct, or dequant-once for
      // a quantized weight) + a row-broadcast bias add. Handles both dense and
      // quantized weights; returns false on M4/older or sub-threshold shapes,
      // falling through to the steel gemm_bias / affine_qmm paths below.
      if (gemm_mma_(enc, x, xe, w, y, ye, M, N, K)) {
        enc.set_function(_fn_bias_add);
        enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, b);
        enc.set_constant(2, N);
        enc.set_constant(3, (unsigned)((std::size_t)M * N));
        enc.dispatch({(unsigned)((std::size_t)M * N), 1, 1}, {256, 1, 1});
        return;
      }
      if (!w.quantized) {
        gemm_bias(x, xe, w.w, b, y, ye, M, N, K);
        return;
      }
      // Tall-M block GEMMs (M ~= seq) amortize a wider tile: BM=64 (>=128 rows),
      // BM=128 at high res (>=1024). Small-M (txt, M=1 head) keep the base tile.
      int bm = 32;
      const bool huge = _qmm_tile == 2 && M >= 1024;
      const bool big  = _qmm_tile >= 1 && M >= 128;
      if (huge)     { bm = 128; }
      else if (big) { bm = 64; }
      enc.set_function(
          w.bits == 8 ? (huge ? _fn_qmm8_bm128 : big ? _fn_qmm8_bm64 : _fn_qmm8)
                      : (huge ? _fn_qmm4_bm128 : big ? _fn_qmm4_bm64 : _fn_qmm4));
      enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
      enc.set_buffer(2, w.qbias); enc.set_buffer(3, x, xe * 2);
      enc.set_buffer(4, y, ye * 2);
      enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
      // BM=128 uses WM=4 (256 threads): threadgroup {32,2,4}, grid z=4.
      const unsigned tgz = (bm == 128) ? 4u : 2u;
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + bm - 1) / bm) * 2), tgz}, {32, 2, tgz});
      // bias add: y[g] += b[g % N] over M*N rows.
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, b);
      enc.set_constant(2, N); enc.set_constant(3, (unsigned)((std::size_t)M * N));
      enc.dispatch({(unsigned)((std::size_t)M * N), 1, 1}, {256, 1, 1});
    };
    // Modulation projection y[ye..] = w @ x + bias (M=1, N=6H, K=H). When the mod
    // weights are quantized, dispatch the peak M=1 affine_qmv (streams the codes
    // at ~peak bandwidth) instead of the steel qmm tile (31/32 M=1 rows wasted);
    // dense (or missing qmv) falls back to gemm_bias_q's M=1 GEMV path.
    auto mod_gemv = [&](const SharedBuffer& x, const QWeight& w,
                        const SharedBuffer& bias, const SharedBuffer& y,
                        std::size_t ye) {
      const int N = 6 * H, K = H;
      const bool qmv = w.quantized &&
          ((w.bits == 8 && _fn_qmv8.valid()) ||
           (w.bits == 4 && _fn_qmv4.valid())) && (N % 4 == 0);
      if (!qmv) { gemm_bias_q(x, 0, w, bias, y, ye, 1, N, K); return; }
      enc.set_function(w.bits == 8 ? _fn_qmv8 : _fn_qmv4);
      enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
      enc.set_buffer(2, w.qbias); enc.set_buffer(3, x);
      enc.set_buffer(4, y, ye * 2);
      enc.set_constant(5, K); enc.set_constant(6, N);
      enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
      enc.set_function(_fn_bias_add);   // y[n] += bias[n]
      enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, bias);
      enc.set_constant(2, N); enc.set_constant(3, (unsigned)N);
      enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
    };
    // AWQ tap: acc[group][L] = max(., col-absmax of in[xe.., M, dim]) -- the
    // per-input-channel activation magnitude at a quantized linear's input.
    auto tap = [&](const char* group, int L, const SharedBuffer& in,
                   std::size_t xe, int M, int dim) {
      if (!_calib_on) { return; }
      auto it = _calib_acc.find(group);
      if (it == _calib_acc.end()) { return; }
      enc.set_function(_fn_colabsmax);
      enc.set_buffer(0, in, xe * 2);
      enc.set_buffer(1, it->second, (std::size_t)L * dim * 2);
      enc.set_constant(2, M); enc.set_constant(3, dim);
      enc.dispatch({(unsigned)dim, 1, 1}, {256, 1, 1});
    };
    auto layernorm = [&](const SharedBuffer& x, std::size_t xe,
                         const SharedBuffer& y, std::size_t ye, int R) {
      enc.set_function(_fn_layernorm);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, y, ye * 2);
      enc.set_constant(2, H); enc.set_constant(3, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto rms = [&](const SharedBuffer& x, std::size_t xe, const SharedBuffer& w,
                   const SharedBuffer& y, std::size_t ye, int R, int Dd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w);
      enc.set_buffer(2, y, ye * 2);
      enc.set_constant(3, Dd); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto silu = [&](const SharedBuffer& x, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_silu);
      enc.set_buffer(0, x); enc.set_buffer(1, x); enc.set_buffer(2, y);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto gelu = [&](const SharedBuffer& x, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_gelu);
      enc.set_buffer(0, x); enc.set_buffer(1, y);
      enc.set_constant(2, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out, int A,
                         int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, A); enc.set_constant(3, Bd); enc.set_constant(4, Hd);
      enc.dispatch({(unsigned)Hd, (unsigned)Bd, (unsigned)A}, {(unsigned)Hd, 1, 1});
    };
    auto rope = [&](const SharedBuffer& x) {
      enc.set_function(_fn_rope_table);
      enc.set_buffer(0, x); enc.set_buffer(1, rcos); enc.set_buffer(2, rsin);
      enc.set_constant(3, NH); enc.set_constant(4, JT); enc.set_constant(5, Hd);
      enc.dispatch({(unsigned)(Hd / 2), (unsigned)JT, (unsigned)NH},
                   {(unsigned)(Hd / 2), 1, 1});
    };
    // Fused transpose [JT,NH,Hd] -> [NH,JT,Hd] + rope in one pass (q/k path).
    auto transpose_rope = [&](const SharedBuffer& in, const SharedBuffer& out) {
      enc.set_function(_fn_transpose_rope);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_buffer(2, rcos); enc.set_buffer(3, rsin);
      enc.set_constant(4, NH); enc.set_constant(5, JT); enc.set_constant(6, Hd);
      enc.dispatch({(unsigned)(Hd / 2), (unsigned)JT, (unsigned)NH},
                   {(unsigned)(Hd / 2), 1, 1});
    };
    auto sdpa = [&](const SharedBuffer& qb, const SharedBuffer& kb,
                    const SharedBuffer& vb, const SharedBuffer& out) {
      const float scale = 1.0f / std::sqrt((float)Hd);
      enc.set_function(_fn_sdpa);
      enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
      enc.set_buffer(3, out);
      enc.set_constant(4, scale); enc.set_constant(5, JT); enc.set_constant(6, Hd);
      enc.set_constant(7, NH); enc.set_constant(8, NH); enc.set_constant(9, JT);
      enc.set_constant(10, JT);
      enc.dispatch({32, (unsigned)NH, (unsigned)JT}, {32, 1, 1});
    };
    // out[ye..] = (1 + mod[scale_e..]) * x[xe..] + mod[shift_e..], broadcasting
    // the single-row mod over `total`/H token rows.
    // vec4 eligibility for the twins below: they load/store 8 bytes at a time
    // off each bound base, so every element offset must be 4-aligned (an
    // unaligned vec4 access is undefined, not just slow) and `total` must be a
    // whole number of H-wide rows for the 2-D grid to cover it.
    auto elt4_ok = [&](int total, std::initializer_list<std::size_t> offs) {
      if ((H % 4) != 0 || H <= 0 || (total % H) != 0) { return false; }
      for (std::size_t o : offs) { if ((o % 4) != 0) { return false; } }
      return true;
    };
    auto adaln = [&](const SharedBuffer& x, std::size_t xe,
                     const SharedBuffer& mod, std::size_t scale_e,
                     std::size_t shift_e, const SharedBuffer& out,
                     std::size_t ye, int total) {
      if (_fn_adaln4.valid() && elt4_ok(total, {xe, scale_e, shift_e, ye})) {
        enc.set_function(_fn_adaln4);
        enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, mod, scale_e * 2);
        enc.set_buffer(2, mod, shift_e * 2); enc.set_buffer(3, out, ye * 2);
        enc.set_constant(4, H / 4); enc.set_constant(5, total / H);
        enc.dispatch({(unsigned)(H / 4), (unsigned)(total / H), 1},
                     {256, 1, 1});
        return;
      }
      enc.set_function(_fn_adaln);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, mod, scale_e * 2);
      enc.set_buffer(2, mod, shift_e * 2); enc.set_buffer(3, out, ye * 2);
      enc.set_constant(4, H); enc.set_constant(5, total);
      enc.dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
    };
    // h[he..] += mod[gate_e..] * sub[se..]. `he`/`se` let the image stream be
    // gated in two row-ranges (generated vs reference) with different mod sets.
    auto gated = [&](const SharedBuffer& h, std::size_t he,
                     const SharedBuffer& mod, std::size_t gate_e,
                     const SharedBuffer& sub, std::size_t se, int total) {
      if (_fn_gated4.valid() && elt4_ok(total, {he, gate_e, se})) {
        enc.set_function(_fn_gated4);
        enc.set_buffer(0, h, he * 2); enc.set_buffer(1, mod, gate_e * 2);
        enc.set_buffer(2, sub, se * 2);
        enc.set_constant(3, H / 4); enc.set_constant(4, total / H);
        enc.dispatch({(unsigned)(H / 4), (unsigned)(total / H), 1},
                     {256, 1, 1});
        return;
      }
      enc.set_function(_fn_gated);
      enc.set_buffer(0, h, he * 2); enc.set_buffer(1, mod, gate_e * 2);
      enc.set_buffer(2, sub, se * 2);
      enc.set_constant(3, H); enc.set_constant(4, total);
      enc.dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
    };
    if (prof) { mark = std::chrono::steady_clock::now(); }
    // ---- embedders ----
    // img_in: hidden[total_img, IC] -> x_img[total_img, H]. Generated tokens
    // first; then each reference latent (same img_in). refs supply their own
    // packed latents.
    gemm_bias(hidden, 0, _img_in_w, _img_in_b, x_img, 0, gen_seq, H, IC);
    { int roff = gen_seq;
      for (const RefImage& r : refs) {
        gemm_bias(r.latents, 0, _img_in_w, _img_in_b, x_img,
                  (std::size_t)roff * H, r.seq, H, IC);
        roff += r.seq;
      }
    }
    // txt: RMSNorm(txt_dim) -> txt_in -> x_txt[txt_seq, H].
    { SharedBuffer tn = buf((std::size_t)txt_seq * _cfg.txt_dim);
      rms(txt, 0, _txt_norm_w, tn, 0, txt_seq, _cfg.txt_dim);
      gemm_bias(tn, 0, _txt_in_w, _txt_in_b, x_txt, 0, txt_seq, H, _cfg.txt_dim);
    }
    // time embedding: temb = t2(silu(t1(time_proj(sigma)))).
    gemm_bias(tproj_b, 0, _t1_w, _t1_b, th1, 0, 1, H, _cfg.time_proj);
    silu(th1, tsilu, H);
    gemm_bias(tsilu, 0, _t2_w, _t2_b, temb, 0, 1, H, H);
    // zero_cond_t: a SECOND embedding at timestep 0 for the (clean) reference
    // tokens -- diffusers' `torch.cat([timestep, timestep * 0])` + per-token
    // `modulate_index`. Only needed when references are actually present.
    const bool zct = want_zct;
    if (zct) {
      const std::vector<float> tp0 = time_proj_(0.0f);
      auto* d0 = static_cast<std::uint16_t*>(tproj0_b.contents());
      for (int i = 0; i < _cfg.time_proj; ++i) { d0[i] = f32_to_bf16_(tp0[i]); }
      gemm_bias(tproj0_b, 0, _t1_w, _t1_b, th1, 0, 1, H, _cfg.time_proj);
      silu(th1, tsilu, H);
      gemm_bias(tsilu, 0, _t2_w, _t2_b, temb0, 0, 1, H, H);
    }
    psplit(t_embed);

    // Build the bf16 steel flash-attention function for this joint length + fill
    // its param block ONCE (same shape across all blocks) -- replaces the scalar
    // O(JT^2) sdpa at high resolution (mirrors Krea-2 / FLUX.2). Full MHA: NH
    // query heads = NH kv heads (gqa_factor 1); Q/K/V/O are [NH, JT, Hd], exactly
    // what the head-major transposes below produce. Falls back to scalar sdpa.
    const int A_BQ = 32, A_BK = 16;   // ALU steel bq/bk (bd128)
    metal_compute::ComputeFunction fn_attn;
    bool use_steel = _steel_attn_ok;
    if (use_steel) {
      auto* p = static_cast<SteelAttnParams*>(_attn_params.contents());
      p->B = 1; p->H = NH; p->D = Hd; p->qL = JT; p->kL = JT;
      p->gqa_factor = 1; p->scale = 1.0f / std::sqrt((float)Hd);
      p->NQ = (JT + A_BQ - 1) / A_BQ; p->NK = (JT + A_BK - 1) / A_BK;
      p->NQ_aligned = JT / A_BQ; p->NK_aligned = JT / A_BK;
      p->qL_rem = JT - p->NQ_aligned * A_BQ;
      p->kL_rem = JT - p->NK_aligned * A_BK;
      p->qL_off = 0;
      p->Q_strides[0] = (std::int64_t)NH * JT * Hd;
      p->Q_strides[1] = (std::int64_t)JT * Hd; p->Q_strides[2] = Hd;
      p->K_strides[0] = p->Q_strides[0];
      p->K_strides[1] = p->Q_strides[1]; p->K_strides[2] = Hd;
      p->V_strides[0] = p->K_strides[0];
      p->V_strides[1] = p->K_strides[1]; p->V_strides[2] = Hd;
      p->O_strides[0] = p->Q_strides[0];
      p->O_strides[1] = p->Q_strides[1]; p->O_strides[2] = Hd;
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (JT % A_BQ) == 0).set_bool(201, (JT % A_BK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      fn_attn = _lib_attn.function("attn_steel_h_bd128_bf16", fc);
      use_steel = fn_attn.valid();
    }
    const unsigned a_nqb = (unsigned)((JT + A_BQ - 1) / A_BQ);

    // ---- transformer blocks ----
    //
    // Re-arm growth for this forward, and let the box answer back: the
    // arithmetic that admits cannot see the limit -- it is a sum over a
    // cache whose future it cannot read -- so what finds it is checking
    // whether the blocks already kept are still in RAM. See
    // BlockResidency::note_weight_residency.
    if (_wire.on()) {
      // Retry a pool that refused earlier -- the refusal may have been
      // another process spiking, and a run that never asks again holds a
      // small resident set for the whole schedule on the strength of one
      // syscall. Growth stopped when the budget ran out and cannot see
      // that the budget moved, so a successful retry has to say so.
      if (_wire.retry(_mc, wire_retry_slack_())) {
        _resid.note_landscape_changed();
      }
      // OUTSIDE the streaming gate below on purpose: a preloaded stack has
      // no blocks to shed, but its weights are the same cold read-only
      // pages the compressor takes first, and the trunk is read on every
      // block of every forward either way. Before any block admission, so
      // a pool that runs out runs out on the shed-able half.
      wire_fixed_(true);
    }
    if (_stream_blocks) {
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
                "MetalQwenImageTransformer: resident weights are only {}% in "
                "RAM ({} of {} sampled pages paged out, {} MB wired) -- "
                "released {} MB, now {} blocks resident",
                (int)(100.0 * (double)incore / (double)examined),
                paged_out, examined, _wire.wired_bytes() >> 20, freed >> 20,
                _resid.count()));
          }
        }
      }
      if (!shortfall) { _resid.note_healthy_forward(); }
    }
    const int n_layers =
        (stop_after_block == -2 || stop_after_block == -5) ? 0 : _cfg.n_layers;
    // Streaming mode: commit the embedders before the first streamed block so
    // the per-block command buffers stay bounded.
    if (_stream_blocks && n_layers > 0) { flush(); }
    // Precompute all blocks' modulation before the loop (preloaded path). silu
    // (temb) is a per-step constant, so the 2*n_layers M=1 mod GEMVs are
    // independent -- running them back-to-back lets the GPU pipeline them
    // (vs interleaving each with the attn/FF that depend on it), and with
    // quantized mod each uses the peak affine_qmv. Streaming keeps the per-block
    // path (blocks aren't all resident). VPIPE_QIE_NO_MOD_PRECOMP disables.
    const bool precomp_mod = !_stream_blocks && !skip_mod && n_layers > 0 &&
        std::getenv("VPIPE_QIE_NO_MOD_PRECOMP") == nullptr;
    SharedBuffer imod_all, tmod_all, zmod_all;
    if (precomp_mod) {
      imod_all = buf((std::size_t)n_layers * 6 * H);
      tmod_all = buf((std::size_t)n_layers * 6 * H);
      if (zct) { zmod_all = buf((std::size_t)n_layers * 6 * H); }
      silu(temb, msilu, H);
      if (zct) { silu(temb0, msilu0, H); }
      for (int L = 0; L < n_layers; ++L) {
        const Block& bb = _blocks[(std::size_t)L];
        const std::size_t off = (std::size_t)L * 6 * H;
        mod_gemv(msilu, bb.img_mod_w, bb.img_mod_b, imod_all, off);
        mod_gemv(msilu, bb.txt_mod_w, bb.txt_mod_b, tmod_all, off);
        // same img_mod weights, driven by the timestep-0 embedding
        if (zct) { mod_gemv(msilu0, bb.img_mod_w, bb.img_mod_b, zmod_all, off); }
      }
      psplit(t_mod);
    }
    // JOIN ANY OUTSTANDING READ ON EVERY EXIT FROM THE STACK. A prefetch
    // may be filling a slot, and this forward returns from a dozen
    // places -- a stop request, a GPU error, an allocation that failed.
    // Freeing or reading a slot while a reader thread is writing into it
    // is a use-after-free, and a scope guard is the only version of this
    // that cannot be forgotten at the thirteenth return.
    struct SlotJoin {
      BlockSlots<Block>* s;
      ~SlotJoin() { s->join(); }
    } slot_join{&_slots};
    _slots.begin_forward();

    for (int L = 0; L < n_layers; ++L) {
      // Pipeline stop -> abandon the forward. Checked EVERY block (not just the
      // streamed tail) so a slow preloaded step at high resolution responds to a
      // stop request within one block (~ms) instead of running all 60.
      if (_stream_stop && _stream_stop()) { return {}; }
      if (_block_progress) { _block_progress(L, n_layers); }
      // Resident in _blocks once residency has promoted it, and read
      // into a slot until then. `_blocks` is sized to n_layers in
      // streaming mode and an unfilled entry reads as empty.
      const bool held = L < (int)_blocks.size() &&
                        !_blocks[(std::size_t)L].qw.empty();
      const bool streaming = _stream_blocks && !held;
      const Block* streamed = nullptr;
      if (streaming) {
        // Two reusable destinations, refilled in place with pread and
        // with the next block's read already issued under the previous
        // block's GPU work. See shared/block-slots.h; the fallback to a
        // per-block allocation is inside it and is sticky.
        streamed = _slots.acquire(L);
        if (streamed == nullptr) { return {}; }
      }
      const Block& b = streaming ? *streamed : _blocks[(std::size_t)L];
      // Modulation params: mod = mod_linear(silu(temb)) [6H]. Precomputed slice
      // (preloaded) or computed per-block (streaming).
      const std::size_t moff = (std::size_t)L * 6 * H * 2;    // bytes
      const std::size_t msz = (std::size_t)6 * H * 2;
      SharedBuffer imod = precomp_mod ? imod_all.subview(moff, msz)
                                      : imod_s.subview(0, msz);
      SharedBuffer tmod = precomp_mod ? tmod_all.subview(moff, msz)
                                      : tmod_s.subview(0, msz);
      // zero_cond_t: the reference rows' modulation (timestep 0).
      SharedBuffer zmod;
      if (zct) {
        zmod = precomp_mod ? zmod_all.subview(moff, msz) : zmod_s.subview(0, msz);
      }
      if (!precomp_mod) {
        silu(temb, msilu, H);
        if (zct) { silu(temb0, msilu0, H); }
        if (!skip_mod) {
          mod_gemv(msilu, b.img_mod_w, b.img_mod_b, imod, 0);
          mod_gemv(msilu, b.txt_mod_w, b.txt_mod_b, tmod, 0);
          if (zct) { mod_gemv(msilu0, b.img_mod_w, b.img_mod_b, zmod, 0); }
        }
        psplit(t_mod);
      }
      // Image-stream modulation is applied in two row-ranges when zero_cond_t
      // is on: generated rows [0, gen_seq) at the current sigma, reference rows
      // [gen_seq, total_img) at timestep 0. `img_mod` picks the mod set and
      // `img_rows` the row count for each range.
      const int nrange = zct ? 2 : 1;
      const SharedBuffer* img_mod[2] = {&imod, &zmod};   // move-only: by ref
      const int img_rows[2] = {zct ? gen_seq : total_img, ref_seq};
      const std::size_t img_off[2] = {0, (std::size_t)gen_seq * H};

      const std::size_t ioff = (std::size_t)txt_seq * H;   // image row offset
      SharedBuffer io = buf((std::size_t)total_img * H);
      SharedBuffer to = buf((std::size_t)txt_seq * H);

      // --- attention ---
      // norm1 (LayerNorm, no affine) + adaLN modulate. Joint buffers `mdl` hold
      // txt-modulated rows [0:txt] then img-modulated rows [txt:].
      layernorm(x_txt, 0, nrm, 0, txt_seq);
      layernorm(x_img, 0, nrm, ioff, total_img);
      // mod chunk layout [shift1|scale1|gate1|shift2|scale2|gate2] (H each).
      adaln(nrm, 0, tmod, H, 0, mdl, 0, txt_seq * H);              // txt norm1
      for (int r = 0; r < nrange; ++r) {                            // img norm1
        adaln(nrm, ioff + img_off[r], *img_mod[r], H, 0, mdl,
              ioff + img_off[r], img_rows[r] * H);
      }
      psplit(t_norm);
      // q/k/v projections (txt: add_*; img: to_*) into the joint buffers.
      tap("txt_qkv", L, mdl, 0, txt_seq, H);
      tap("img_qkv", L, mdl, ioff, total_img, H);
      gemm_bias_q(mdl, 0, b.aqw, b.aqb, jq, 0, txt_seq, H, H);
      gemm_bias_q(mdl, 0, b.akw, b.akb, jk, 0, txt_seq, H, H);
      gemm_bias_q(mdl, 0, b.avw, b.avb, jv, 0, txt_seq, H, H);
      gemm_bias_q(mdl, ioff, b.qw, b.qb, jq, ioff, total_img, H, H);
      gemm_bias_q(mdl, ioff, b.kw, b.kb, jk, ioff, total_img, H, H);
      gemm_bias_q(mdl, ioff, b.vw, b.vb, jv, ioff, total_img, H, H);
      psplit(t_qkv);
      // q/k per-head RMSNorm (txt: norm_added_*, img: norm_*).
      rms(jq, 0, b.naq, jq, 0, txt_seq * NH, Hd);
      rms(jk, 0, b.nak, jk, 0, txt_seq * NH, Hd);
      rms(jq, ioff, b.nq, jq, ioff, total_img * NH, Hd);
      rms(jk, ioff, b.nk, jk, ioff, total_img * NH, Hd);
      // transpose [JT,NH,Hd] -> [NH,JT,Hd], apply RoPE, joint attention. q/k
      // fuse the transpose + rope into one pass; v is a plain transpose (no rope).
      if (_fn_transpose_rope.valid()) {
        transpose_rope(jq, qt);
        transpose_rope(jk, kt);
      } else {
        transpose(jq, qt, JT, NH);
        transpose(jk, kt, JT, NH);
        rope(qt); rope(kt);
      }
      transpose(jv, vt, JT, NH);
      if (use_steel) {
        // Register-resident flash attention over the joint sequence: Q/K/V/O
        // [NH, JT, Hd]. Grid (32*NQ, 4*NH, 1), tg (32, 4, 1) per MLX steel.
        enc.set_function(fn_attn);
        enc.set_buffer(0, qt); enc.set_buffer(1, kt); enc.set_buffer(2, vt);
        enc.set_buffer(3, at); enc.set_buffer(4, _attn_params);
        enc.dispatch({32 * a_nqb, 4 * (unsigned)NH, 1}, {32, 4, 1});
      } else {
        sdpa(qt, kt, vt, at);
      }
      transpose(at, att, NH, JT);   // [NH,JT,Hd] -> [JT,NH,Hd]
      psplit(t_attn);
      // output projections + gated residual (gate1 @ 2H).
      tap("img_o", L, att, ioff, total_img, H);
      tap("txt_o", L, att, 0, txt_seq, H);
      gemm_bias_q(att, ioff, b.ow, b.ob, io, 0, total_img, H, H);
      gemm_bias_q(att, 0, b.aow, b.aob, to, 0, txt_seq, H, H);
      for (int r = 0; r < nrange; ++r) {                            // img gate1
        gated(x_img, img_off[r], *img_mod[r], 2 * H, io, img_off[r],
              img_rows[r] * H);
      }
      gated(x_txt, 0, tmod, 2 * H, to, 0, txt_seq * H);
      psplit(t_oproj);
      if (stop_after_block == -4 && L == 0) {
        // Flag only -- the GPU work is still deferred here, so `att` cannot
        // be read until after stream.commit().wait() below.
        dbg_att = true;
        dbg_ioff = ioff;
        break;
      }
      if (stop_after_block == -3 && L == 0) { break; }   // post-attention debug

      // --- MLP (norm2 + adaLN + GELU FeedForward + gated residual, gate2) ---
      layernorm(x_img, 0, nrm, ioff, total_img);
      for (int r = 0; r < nrange; ++r) {                            // img norm2
        adaln(nrm, ioff + img_off[r], *img_mod[r], 4 * H, 3 * H, mdl,
              ioff + img_off[r], img_rows[r] * H);
      }
      tap("img_fc1", L, mdl, ioff, total_img, H);
      gemm_bias_q(mdl, ioff, b.img_fc1_w, b.img_fc1_b, g1, 0, total_img, FF, H);
      gelu(g1, g1, total_img * FF);
      tap("img_fc2", L, g1, 0, total_img, FF);
      gemm_bias_q(g1, 0, b.img_fc2_w, b.img_fc2_b, io, 0, total_img, H, FF);
      for (int r = 0; r < nrange; ++r) {                            // img gate2
        gated(x_img, img_off[r], *img_mod[r], 5 * H, io, img_off[r],
              img_rows[r] * H);
      }

      layernorm(x_txt, 0, nrm, 0, txt_seq);
      adaln(nrm, 0, tmod, 4 * H, 3 * H, mdl, 0, txt_seq * H);
      tap("txt_fc1", L, mdl, 0, txt_seq, H);
      gemm_bias_q(mdl, 0, b.txt_fc1_w, b.txt_fc1_b, g2, 0, txt_seq, FF, H);
      gelu(g2, g2, txt_seq * FF);
      tap("txt_fc2", L, g2, 0, txt_seq, FF);
      gemm_bias_q(g2, 0, b.txt_fc2_w, b.txt_fc2_b, to, 0, txt_seq, H, FF);
      gated(x_txt, 0, tmod, 5 * H, to, 0, txt_seq * H);
      psplit(t_ff);

      // Commit block L before `streamed` (its weights) frees at iteration end.
      // Pinned blocks stay resident, so no per-block flush is needed for them.
      if (streaming) {
        // COMMIT, then issue the next block's read, then WAIT. Between
        // the commit and the wait the GPU is busy with block L and this
        // thread has nothing to do, which is the whole opportunity.
        enc.end();
        metal_compute::CommandStream::Fence bf = stream.commit();
        {
          // The next block that will actually be STREAMED. Resident ones
          // are skipped -- prefetching one would read bytes the forward
          // already has. Safe to look ahead: promotion only ever adds
          // the block just finished, never one further down the stack.
          int nxt = -1;
          for (int n = L + 1; n < n_layers; ++n) {
            const bool h = n < (int)_blocks.size() &&
                           !_blocks[(std::size_t)n].qw.empty();
            if (!h) { nxt = n; break; }
          }
          _slots.prefetch(nxt);
        }
        bf.wait();
        stream = _mc->make_command_stream();
        enc = stream.begin_compute();
        // The commit above has been WAITED for, which is why the
        // promotion happens here and not at the top of the iteration:
        // an encoded GEMM holds these buffers by pointer. Keeping them
        // costs nothing extra -- the bytes are already resident --
        // against the per-step re-read it retires.
        if (L < (int)_blocks.size()) {
          const std::size_t nb = _slots.last_bytes();
          // Past the wire budget there is nothing to gain: the block would
          // be kept unprotected, the compressor would take it (it is the
          // coldest memory in the process), and the next residency walk
          // would shed a block and ratchet the ceiling over the whole
          // resident set. Better not to hold it at all.
          if (_wire.wirable(nb) && _resid.admit(_mc, nb) &&
              _slots.promote_into(_blocks[(std::size_t)L])) {
            // Wired LAST, after every write this block will ever get:
            // mlock pins the pages that exist NOW.
            _wire.note_wired(_mc,
                             wire_block_(_blocks[(std::size_t)L], true), nb);
            _resid.note_admitted(nb);
            if (_mc->session() != nullptr) {
              _mc->session()->log_debug(fmt(
                  "MetalQwenImageTransformer: block {} resident ({} of {}, "
                  "{} MB, {} MB wired)", L, _resid.count(),
                  _cfg.n_layers, _resid.bytes() >> 20,
                  _wire.wired_bytes() >> 20));
            }
          }
        }
      }
      if (stop_after_block == L) { break; }
    }

    if (stop_after_block < 0) {
      // norm_out (AdaLayerNormContinuous: scale@0, shift@H) on the generated
      // image tokens, then proj_out -> velocity.
      silu(temb, msilu, H);
      gemm_bias(msilu, 0, _normout_w, _normout_b, imod_s, 0, 1, 2 * H, H);
      layernorm(x_img, 0, nrm, 0, gen_seq);
      adaln(nrm, 0, imod_s, 0, H, mdl, 0, gen_seq * H);
      gemm_bias(mdl, 0, _projout_w, _projout_b, velo, 0, gen_seq, IC, H);
      psplit(t_final);
    }
  }
  stream.commit().wait();

  if (prof && stop_after_block < 0 && _mc->session() != nullptr) {
    const double tot = t_embed + t_mod + t_norm + t_qkv + t_attn + t_oproj +
                       t_ff + t_final;
    _mc->session()->log_normal(fmt(
        "QIE DiT profile (gen={} txt={} img={} blocks={}, 1 step): total {} ms "
        "| embed {} | mod {} | norm+adaln {} | qkv-gemm {} | attn {} | "
        "o-gemm {} | ff(gelu) {} | final {} (ms; barriers inflate absolute "
        "time)", gen_seq, txt_seq, total_img, _cfg.n_layers, (long)tot,
        (long)t_embed, (long)t_mod, (long)t_norm, (long)t_qkv, (long)t_attn,
        (long)t_oproj, (long)t_ff, (long)t_final));
  }

  if (stop_after_block == -5) {          // staged debug: temb [1, H]
    SharedBuffer out = buf((std::size_t)H);
    std::memcpy(out.contents(), temb.contents(), (std::size_t)H * 2);
    return out;
  }
  if (stop_after_block == -4) {
    if (!dbg_att) { return {}; }
    SharedBuffer out = buf((std::size_t)gen_seq * H);
    std::memcpy(out.contents(),
                static_cast<const char*>(att.contents()) + dbg_ioff * 2,
                (std::size_t)gen_seq * H * 2);
    return out;
  }
  if (stop_after_block == -2 || stop_after_block == -3) {
    SharedBuffer out = buf((std::size_t)gen_seq * H);
    std::memcpy(out.contents(), x_img.contents(), (std::size_t)gen_seq * H * 2);
    return out;
  }
  if (stop_after_block >= 0) {
    // Return the image-stream hidden [gen_seq, H] after that block.
    SharedBuffer out = buf((std::size_t)gen_seq * H);
    std::memcpy(out.contents(), x_img.contents(), (std::size_t)gen_seq * H * 2);
    return out;
  }
  return velo;
}

}  // namespace genai
}  // namespace vpipe
