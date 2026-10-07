// MetalQwenModel's raw-source layers: in-memory w8 quantization and
// decode-capable layer streaming. See metal-qwen-raw-layers.h for what
// they are and why they share one builder.

#include "generative-models/qwen3/metal-qwen-raw-layers.h"

#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <cstring>
#include <utility>

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {

constexpr const char* kRawNames[kRawCount] = {
  "input_layernorm.weight",  "post_attention_layernorm.weight",
  "self_attn.q_norm.weight", "self_attn.k_norm.weight",
  "self_attn.q_proj.weight", "self_attn.k_proj.weight",
  "self_attn.v_proj.weight", "self_attn.o_proj.weight",
  "mlp.gate_proj.weight",    "mlp.up_proj.weight",
  "mlp.down_proj.weight",
};

// An affine w8 group-64 [N, K]: a byte per weight plus a 2-byte scale and
// a 2-byte bias per 64.
std::size_t
w8_bytes_(std::int64_t N, std::int64_t K)
{
  return (std::size_t)(N * K) + (std::size_t)(N * (K / 64)) * 4;
}

// Every distinct allocation among `bufs`, summed once. A product that
// aliases a source (a norm, or a bf16 projection kept as it sits) is the
// same allocation and must not be counted twice.
std::size_t
unique_bytes_(std::initializer_list<const SharedBuffer*> bufs,
              const std::vector<SharedBuffer>* more = nullptr)
{
  std::vector<const void*> seen;
  std::size_t n = 0;
  auto add = [&](const SharedBuffer& b) {
    if (b.empty()) { return; }
    const void* p = b.contents();
    for (const void* q : seen) { if (q == p) { return; } }
    seen.push_back(p);
    n += b.byte_size();
  };
  for (const SharedBuffer* b : bufs) { add(*b); }
  if (more != nullptr) { for (const SharedBuffer& b : *more) { add(b); } }
  return n;
}

// A refcount-sharing window over all of `b` (empty stays empty). Not
// owned and never wired: an alias, for as long as the owner lives.
SharedBuffer
alias_(const SharedBuffer& b)
{
  return b.empty() ? SharedBuffer{} : b.subview(0, b.byte_size());
}

}  // namespace

// ---- eligibility + planning sizes -------------------------------------

bool
MetalQwenModel::raw_eligible_(const Config& c)
{
  // Dense full attention with Qwen3 per-head q/k norms and no bias,
  // standard RMSNorm (the raw norm is the weight the kernel multiplies by,
  // so a slot can hand it over as it sits), bf16 compute, and no output
  // head to stream. Every MOSS backbone; no Qwen3.5 (GDN, zero-centred
  // norms) and no MoE.
  return c.dense && !c.is_moe() && c.backbone_only && c.use_bf16 &&
         !c.zero_centered_norm && c.qk_norm && !c.attention_bias &&
         !c.calib_stream && c.hidden % 64 == 0 && c.qd() % 64 == 0 &&
         c.ffn_inner % 64 == 0;
}

MetalQwenModel::RawSizes
MetalQwenModel::raw_sizes(const MetalLlamaWeights& wts,
                          const std::string& layer_prefix, int n_layers,
                          int quant_bits)
{
  RawSizes out;
  for (int L = 0; L < n_layers; ++L) {
    const std::string p = layer_prefix + std::to_string(L) + ".";
    const auto* probe = wts.info(p + kRawNames[kRawQ]);
    if (probe == nullptr) { return RawSizes{}; }
    // A pack's projections are (u32 codes, scales, biases) triples. Every
    // layer must agree: a stack half raw and half packed is neither.
    const bool pk = probe->dtype == "U32";
    if (L > 0 && pk != out.pack) { return RawSizes{}; }
    out.pack = pk;
    if (pk) {
      std::size_t raw = 0, built = 0, copies = 0;
      for (int i = 0; i < kRawCount; ++i) {
        const std::string nm = p + kRawNames[i];
        const auto* ti = wts.info(nm);
        if (ti == nullptr) { return RawSizes{}; }
        std::int64_t numel = 1;
        for (std::int64_t d : ti->shape) { numel *= d; }
        if (i < kRawQ) {                        // a norm, bf16 as held
          raw += (std::size_t)ti->nbytes;
          built += (std::size_t)numel * 2;
          continue;
        }
        const std::string stem = nm.substr(0, nm.size() - 7);   // .weight
        const auto* si = wts.info(stem + ".scales");
        const auto* bi = wts.info(stem + ".biases");
        if (si == nullptr || bi == nullptr || ti->shape.size() != 2 ||
            si->shape.size() != 2) {
          return RawSizes{};
        }
        const std::int64_t N = ti->shape[0], K = si->shape[1] * 64;
        // Eight bits: a byte per weight. A 4-bit pack is not this. And
        // rows the build copies in 32-bit words: K of q|k|v and gate|up is
        // the hidden size, whose scale rows are K/32 bytes.
        if ((std::int64_t)ti->nbytes != N * K) { return RawSizes{}; }
        if (i != kRawO && i != kRawDown && K % 128 != 0) {
          return RawSizes{};
        }
        raw += (std::size_t)(ti->nbytes + si->nbytes + bi->nbytes);
        const std::size_t w = w8_bytes_(N, K);
        built += w;
        // q, k, v and gate, up are copied into the fused products.
        if (i != kRawO && i != kRawDown) { copies += w; }
      }
      out.raw += raw;
      out.built += built;
      if (built > out.layer) { out.layer = built; }
      // A slot holds the pack's tensors as they sit (o, down and the
      // norms ARE products) plus the two fused copies.
      if (built + copies > out.slot) { out.slot = built + copies; }
      ++out.n_layers;
      continue;
    }
    std::size_t raw = 0, built = 0;
    for (int i = 0; i < kRawCount; ++i) {
      const auto* ti = wts.info(p + kRawNames[i]);
      if (ti == nullptr) { return RawSizes{}; }
      raw += (std::size_t)ti->nbytes;
      std::int64_t numel = 1;
      for (std::int64_t d : ti->shape) { numel *= d; }
      const bool proj = i >= kRawQ && ti->shape.size() == 2;
      if (proj && quant_bits == 8) {
        built += w8_bytes_(ti->shape[0], ti->shape[1]);
      } else {
        built += (std::size_t)numel * 2;   // compute dtype is bf16
      }
    }
    out.raw += raw;
    out.built += built;
    if (built > out.layer) { out.layer = built; }
    // A slot holds the sources AND the products built from them; in bf16
    // the products beyond the norms and o/gate/up/down are the fused
    // q|k|v alone, which the line below over-counts by at most that.
    const std::size_t slot = raw + built;
    if (slot > out.slot) { out.slot = slot; }
    ++out.n_layers;
  }
  return out;
}

bool
MetalQwenModel::pack_streamable_(const MetalLlamaWeights& wts,
                                 const Config& c)
{
  // The fused copies move rows in 32-bit words: a row of scales is H/64
  // bf16 values, so H must be a multiple of 128.
  if (c.hidden % 128 != 0) { return false; }
  const std::string pfx = c.weight_prefix + c.model_seg + "layers.";
  const RawSizes rs = raw_sizes(wts, pfx, c.n_layers, 8);
  if (!rs.pack || rs.n_layers != c.n_layers) { return false; }
  // The shapes the forward was configured for, on every layer.
  for (int L = 0; L < c.n_layers; ++L) {
    const std::string p = pfx + std::to_string(L) + ".";
    const int qdo = c.attn_output_gate ? 2 * c.qd() : c.qd();
    const struct { int i; int N, K; } want[] = {
      {kRawQ, qdo, c.hidden},          {kRawK, c.kd(), c.hidden},
      {kRawV, c.kd(), c.hidden},       {kRawO, c.hidden, c.qd()},
      {kRawGate, c.ffn_inner, c.hidden}, {kRawUp, c.ffn_inner, c.hidden},
      {kRawDown, c.hidden, c.ffn_inner},
    };
    for (const auto& w : want) {
      const std::string nm = p + kRawNames[w.i];
      const auto* ti = wts.info(nm);
      const auto* si = wts.info(nm.substr(0, nm.size() - 7) + ".scales");
      if (ti == nullptr || si == nullptr || ti->shape[0] != w.N ||
          si->shape[1] * 64 != w.K) {
        return false;
      }
    }
  }
  return true;
}

// ---- RawStream --------------------------------------------------------

MetalQwenModel::RawStream::RawStream(MetalQwenModel* model,
                                     metal_compute::MetalCompute* c,
                                     bool quantize, bool stream, bool pack_src)
    : m(model), mc(c), quant(quantize || pack_src), streaming(stream),
      pack(pack_src)
{
}

MetalQwenModel::RawStream::~RawStream() { shutdown(); }

std::string
MetalQwenModel::RawStream::prefix_(int L) const
{
  const Config& c = m->_cfg;
  return c.weight_prefix + c.model_seg + "layers." + std::to_string(L) + ".";
}

std::vector<std::string>
MetalQwenModel::RawStream::source_names_(int L) const
{
  const std::string p = prefix_(L);
  std::vector<std::string> out;
  if (!pack) {
    for (int i = 0; i < kRawCount; ++i) { out.push_back(p + kRawNames[i]); }
    return out;
  }
  for (int i = 0; i < kRawQ; ++i) { out.push_back(p + kRawNames[i]); }
  for (int i = kRawQ; i < kRawCount; ++i) {
    const std::string nm = p + kRawNames[i];
    const std::string stem = nm.substr(0, nm.size() - 7);   // .weight
    out.push_back(nm);
    out.push_back(stem + ".scales");
    out.push_back(stem + ".biases");
  }
  return out;
}

SharedBuffer
MetalQwenModel::RawStream::read_source_(const std::string& name) const
{
  // The compute-dtype converter widens; u32 codes are not a dtype to
  // convert, so a pack's codes are read as they sit.
  if (pack) {
    const auto* ti = m->_ws->src().info(name);
    if (ti != nullptr && ti->dtype == "U32") {
      return m->_ws->read(name, mc, WeightSet::Residency::Copied);
    }
  }
  return m->raw_read_(name);
}

bool
MetalQwenModel::RawStream::init()
{
  const Config& c = m->_cfg;
  H   = c.hidden;
  qd  = c.qd();
  qdo = c.attn_output_gate ? 2 * qd : qd;
  kd  = c.kd();
  F   = c.ffn_inner;
  D   = c.head_dim;
  if (pack) {
    // Type-blind row copies; the library is the one the build already
    // takes for a raw checkpoint.
    _lib_quant = mc->load_library("affine_dequant_bf16");
    _fn_rows   = _lib_quant.function("copy_rows_u32");
    if (!_fn_rows.valid()) { return false; }
    kept_bytes = w8_bytes_(qdo + 2 * kd, H) + w8_bytes_(H, qd) +
                 w8_bytes_(2 * F, H) + w8_bytes_(H, F);
  } else if (quant) {
    // The bf16 library: the build reads bf16 sources and writes bf16
    // scales, which is what the bf16 qmv / steel kernels read.
    _lib_quant = mc->load_library("affine_dequant_bf16");
    _fn_quant  = _lib_quant.function("affine_quant_rows_w8g64");
    if (!_fn_quant.valid()) { return false; }
    kept_bytes = w8_bytes_(qdo + 2 * kd, H) + w8_bytes_(H, qd) +
                 w8_bytes_(2 * F, H) + w8_bytes_(H, F);
  } else {
    _fn_copy = m->_lib_elt.function("copy_f16");
    if (!_fn_copy.valid()) { return false; }
    kept_bytes = ((std::size_t)(qdo + 2 * kd) * H + (std::size_t)H * qd +
                  (std::size_t)2 * F * H + (std::size_t)H * F) * 2;
  }
  kept_bytes += (std::size_t)(2 * H + 2 * D) * 2;   // the four norms
  const std::size_t raw =
      ((std::size_t)(qdo + 2 * kd) * H + (std::size_t)H * qd +
       (std::size_t)3 * F * H + (std::size_t)(2 * H + 2 * D)) * 2;
  slot_bytes = raw + (quant ? kept_bytes - (std::size_t)(2 * H + 2 * D) * 2
                            : (std::size_t)(qdo + 2 * kd) * H * 2);
  if (pack) {
    // The pack's tensors as they sit (o, down and the norms are products
    // already) plus the fused q|k|v and interleaved gate|up copies.
    slot_bytes = kept_bytes + w8_bytes_(qdo + 2 * kd, H) + w8_bytes_(2 * F, H);
  }
  held.assign((std::size_t)c.n_layers, 0);
  _wire.open(mc);
  if (!streaming) { return true; }

  BlockSlots<RawLayer>::Ops ops;
  ops.each = [this](int L, RawLayer& b,
                    const BlockSlots<RawLayer>::TensorFn& fn) {
    const std::vector<std::string> names = source_names_(L);
    for (std::size_t i = 0; i < names.size(); ++i) {
      // bf16 as it sits (a pack's u32 codes too); an f16 tensor is
      // converted in place, which is legitimate only because the two are
      // the same width.
      fn(names[i], b.src[i], Placement::kBf16);
    }
  };
  ops.rebuild_one = [this](const std::string& name, Placement) {
    return read_source_(name);
  };
  ops.build = [this](int L, RawLayer& b) {
    return read_sources_(L, b) && alloc_products_(b);
  };
  ops.clone = [this](const RawLayer& s, RawLayer& d, bool copy) {
    return copy ? copy_products_(s, d) : alloc_like_(s, d);
  };
  ops.bytes = [](const RawLayer& b) {
    const Layer& y = b.prod;
    return unique_bytes_({&y.qw, &y.qs, &y.qb, &y.ow, &y.os, &y.ob,
                          &y.guw, &y.gus, &y.gub, &y.uw, &y.dw, &y.ds,
                          &y.db, &y.in_ln, &y.post_ln, &y.q_norm,
                          &y.k_norm}, &b.src);
  };
  ops.empty = [](const RawLayer& b) {
    return b.src.empty() || b.src[0].empty();
  };
  // A source a refill had to REPLACE (an f32 tensor, a short read) is a
  // new allocation, and a product aliasing the old one would read stale
  // bytes. Re-point the aliases after every refill; the real products
  // are rebuilt in the command buffer that runs the layer.
  ops.post_refill = [this](int, RawLayer& b) {
    alias_products_(b);
    return true;
  };
  _slots.set_weight_set(m->_ws.get());
  _slots.configure(mc, std::move(ops), "MetalQwenModel",
                   "VPIPE_QWEN_NO_SLOTS");
  return true;
}

bool
MetalQwenModel::RawStream::read_sources_(int L, RawLayer& r)
{
  const std::vector<std::string> names = source_names_(L);
  r.src.clear();
  r.src.resize(names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    r.src[i] = read_source_(names[i]);
    if (r.src[i].empty()) { return false; }
  }
  return true;
}

void
MetalQwenModel::RawStream::alias_products_(RawLayer& r) const
{
  Layer& y = r.prod;
  y.is_full = true;
  y.in_ln   = alias_(r.src[(std::size_t)kRawInLn]);
  y.post_ln = alias_(r.src[(std::size_t)kRawPostLn]);
  y.q_norm  = alias_(r.src[(std::size_t)kRawQNorm]);
  y.k_norm  = alias_(r.src[(std::size_t)kRawKNorm]);
  if (pack) {
    // A pack's o and down are already the products.
    y.ow = alias_(r.src[(std::size_t)kPackO]);
    y.os = alias_(r.src[(std::size_t)kPackO + 1]);
    y.ob = alias_(r.src[(std::size_t)kPackO + 2]);
    y.dw = alias_(r.src[(std::size_t)kPackDown]);
    y.ds = alias_(r.src[(std::size_t)kPackDown + 1]);
    y.db = alias_(r.src[(std::size_t)kPackDown + 2]);
  } else if (!quant) {
    // The dense forward reads these as they sit: o_proj, and gate / up /
    // down de-fused (two GEMVs + SwiGLU), exactly as LayerLoad's dense
    // branch lays them out.
    y.ow  = alias_(r.src[(std::size_t)kRawO]);
    y.guw = alias_(r.src[(std::size_t)kRawGate]);
    y.uw  = alias_(r.src[(std::size_t)kRawUp]);
    y.dw  = alias_(r.src[(std::size_t)kRawDown]);
  }
}

// What a RESIDENT layer keeps: the products, with every alias of a source
// replaced by the source itself, so each kept buffer is OWNED -- on the
// books, wirable and parkable -- rather than a window onto one.
MetalQwenModel::Layer
MetalQwenModel::RawStream::take_products_(RawLayer& r) const
{
  Layer y = std::move(r.prod);
  // A clone out of a slot carries products only, already owned; a block
  // MOVED out of the per-block fallback still has its sources to hand
  // over in place of the windows onto them.
  const int n_src = pack ? (int)kPackCount : (int)kRawCount;
  if (r.src.size() != (std::size_t)n_src) {
    y.is_full = true;
    return y;
  }
  auto own = [&](int si, SharedBuffer& dst) {
    dst = std::move(r.src[(std::size_t)si]);
  };
  own(kRawInLn, y.in_ln);
  own(kRawPostLn, y.post_ln);
  own(kRawQNorm, y.q_norm);
  own(kRawKNorm, y.k_norm);
  if (pack) {
    own(kPackO, y.ow);
    own(kPackO + 1, y.os);
    own(kPackO + 2, y.ob);
    own(kPackDown, y.dw);
    own(kPackDown + 1, y.ds);
    own(kPackDown + 2, y.db);
  } else if (!quant) {
    own(kRawO, y.ow);
    own(kRawGate, y.guw);
    own(kRawUp, y.uw);
    own(kRawDown, y.dw);
  }
  y.is_full = true;
  return y;
}

// The forward's view of a slot's layer: windows onto the slot's buffers,
// so binding it as _layers[L] neither copies nor takes ownership.
MetalQwenModel::Layer
MetalQwenModel::RawStream::alias_layer_(const Layer& a)
{
  Layer y;
  y.is_full = true;
  for (auto [from, to] : std::initializer_list<
           std::pair<const SharedBuffer*, SharedBuffer*>>{
           {&a.qw, &y.qw}, {&a.qs, &y.qs}, {&a.qb, &y.qb},
           {&a.ow, &y.ow}, {&a.os, &y.os}, {&a.ob, &y.ob},
           {&a.guw, &y.guw}, {&a.gus, &y.gus}, {&a.gub, &y.gub},
           {&a.uw, &y.uw}, {&a.dw, &y.dw}, {&a.ds, &y.ds}, {&a.db, &y.db},
           {&a.in_ln, &y.in_ln}, {&a.post_ln, &y.post_ln},
           {&a.q_norm, &y.q_norm}, {&a.k_norm, &y.k_norm}}) {
    *to = alias_(*from);
  }
  return y;
}

bool
MetalQwenModel::RawStream::alloc_products_(RawLayer& r)
{
  Layer& y = r.prod;
  y = Layer{};
  auto mk = [&](std::size_t n, SharedBuffer& b) {
    b = mc->make_shared_buffer(n);
    return !b.empty();
  };
  const std::size_t nqkv = (std::size_t)(qdo + 2 * kd);
  bool ok = true;
  if (pack) {
    // Only the fusions are new buffers; o and down alias the sources.
    const std::size_t gH = (std::size_t)H / 64;
    ok = ok && mk(nqkv * H, y.qw) && mk(nqkv * gH * 2, y.qs) &&
         mk(nqkv * gH * 2, y.qb);
    ok = ok && mk((std::size_t)2 * F * H, y.guw) &&
         mk((std::size_t)2 * F * gH * 2, y.gus) &&
         mk((std::size_t)2 * F * gH * 2, y.gub);
  } else if (quant) {
    const std::size_t gH = (std::size_t)H / 64, gq = (std::size_t)qd / 64,
                      gF = (std::size_t)F / 64;
    ok = ok && mk(nqkv * H, y.qw) && mk(nqkv * gH * 2, y.qs) &&
         mk(nqkv * gH * 2, y.qb);
    ok = ok && mk((std::size_t)H * qd, y.ow) &&
         mk((std::size_t)H * gq * 2, y.os) && mk((std::size_t)H * gq * 2, y.ob);
    ok = ok && mk((std::size_t)2 * F * H, y.guw) &&
         mk((std::size_t)2 * F * gH * 2, y.gus) &&
         mk((std::size_t)2 * F * gH * 2, y.gub);
    ok = ok && mk((std::size_t)H * F, y.dw) &&
         mk((std::size_t)H * gF * 2, y.ds) && mk((std::size_t)H * gF * 2, y.db);
  } else {
    ok = mk(nqkv * H * 2, y.qw);
  }
  if (!ok) { return false; }
  alias_products_(r);
  return true;
}

bool
MetalQwenModel::RawStream::copy_products_(const RawLayer& s, RawLayer& d)
{
  // PROMOTION: the products only, deep-copied out of the slot (which has
  // to stay put for the next read). The sources stay behind -- what a
  // promoted layer keeps is what the residency ledger was asked for.
  d.src.clear();
  d.prod = Layer{};
  d.prod.is_full = true;
  auto cp = [&](const SharedBuffer& from, SharedBuffer& to) {
    if (from.empty()) { return true; }
    to = mc->make_shared_buffer(from.byte_size());
    if (to.empty()) { return false; }
    std::memcpy(to.contents(), from.contents(), from.byte_size());
    return true;
  };
  const Layer& a = s.prod;
  Layer& b = d.prod;
  return cp(a.qw, b.qw) && cp(a.qs, b.qs) && cp(a.qb, b.qb) &&
         cp(a.ow, b.ow) && cp(a.os, b.os) && cp(a.ob, b.ob) &&
         cp(a.guw, b.guw) && cp(a.gus, b.gus) && cp(a.gub, b.gub) &&
         cp(a.uw, b.uw) && cp(a.dw, b.dw) && cp(a.ds, b.ds) &&
         cp(a.db, b.db) && cp(a.in_ln, b.in_ln) &&
         cp(a.post_ln, b.post_ln) && cp(a.q_norm, b.q_norm) &&
         cp(a.k_norm, b.k_norm);
}

bool
MetalQwenModel::RawStream::alloc_like_(const RawLayer& s, RawLayer& d)
{
  // The SECOND slot: same shapes, contents to come from the next refill.
  d.src.clear();
  d.src.resize(s.src.size());
  for (std::size_t i = 0; i < s.src.size(); ++i) {
    d.src[i] = mc->make_shared_buffer(s.src[i].byte_size());
    if (d.src[i].empty()) { return false; }
  }
  return alloc_products_(d);
}

void
MetalQwenModel::RawStream::encode_build_(ComputeEncoder& enc,
                                         const RawLayer& r) const
{
  const Layer& y = r.prod;
  if (pack) {
    // The pack's q, k, v rows concatenated, and its gate, up rows
    // interleaved -- codes, scales and biases alike, by the row map the
    // raw build quantizes through, so the layout is the one either
    // resident route produces.
    auto rows = [&](const SharedBuffer& src, const SharedBuffer& dst, int N,
                    int words, int mul, int add) {
      enc.set_function(_fn_rows);
      enc.set_buffer(0, src);
      enc.set_buffer(1, dst);
      enc.set_constant(2, words);
      enc.set_constant(3, mul);
      enc.set_constant(4, add);
      enc.dispatch({(unsigned)words, (unsigned)N, 1}, {32, 8, 1});
    };
    auto trip = [&](int si, const SharedBuffer& w, const SharedBuffer& s,
                    const SharedBuffer& b, int N, int mul, int add) {
      rows(r.src[(std::size_t)si], w, N, H / 4, mul, add);        // H bytes
      rows(r.src[(std::size_t)si + 1], s, N, H / 128, mul, add);  // H/64 x2
      rows(r.src[(std::size_t)si + 2], b, N, H / 128, mul, add);
    };
    trip(kPackQ, y.qw, y.qs, y.qb, qdo, 1, 0);
    trip(kPackK, y.qw, y.qs, y.qb, kd, 1, qdo);
    trip(kPackV, y.qw, y.qs, y.qb, kd, 1, qdo + kd);
    trip(kPackGate, y.guw, y.gus, y.gub, F, 2, 0);
    trip(kPackUp, y.guw, y.gus, y.gub, F, 2, 1);
    return;
  }
  if (quant) {
    // One dispatch per projection, each writing its rows straight into the
    // fused / interleaved destination the affine forward reads.
    auto q = [&](int si, const SharedBuffer& w, const SharedBuffer& s,
                 const SharedBuffer& b, int N, int K, int mul, int add) {
      enc.set_function(_fn_quant);
      enc.set_buffer(0, r.src[(std::size_t)si]);
      enc.set_buffer(1, w);
      enc.set_buffer(2, s);
      enc.set_buffer(3, b);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, mul);
      enc.set_constant(7, add);
      enc.dispatch({(unsigned)(K / 64), (unsigned)N, 1}, {32, 8, 1});
    };
    q(kRawQ, y.qw, y.qs, y.qb, qdo, H, 1, 0);
    q(kRawK, y.qw, y.qs, y.qb, kd, H, 1, qdo);
    q(kRawV, y.qw, y.qs, y.qb, kd, H, 1, qdo + kd);
    q(kRawO, y.ow, y.os, y.ob, H, qd, 1, 0);
    q(kRawGate, y.guw, y.gus, y.gub, F, H, 2, 0);
    q(kRawUp, y.guw, y.gus, y.gub, F, H, 2, 1);
    q(kRawDown, y.dw, y.ds, y.db, H, F, 1, 0);
    return;
  }
  // bf16: the only product is the row-concatenated q|k|v.
  std::size_t off = 0;
  const int zero = 0;
  for (int si : {kRawQ, kRawK, kRawV}) {
    const SharedBuffer& src = r.src[(std::size_t)si];
    const int n = (int)(src.byte_size() / 2);
    enc.set_function(_fn_copy);
    enc.set_buffer(0, src);
    enc.set_buffer(1, y.qw, off);
    enc.set_constant(2, zero);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    off += src.byte_size();
  }
}

bool
MetalQwenModel::RawStream::build_resident(int L)
{
  RawLayer r;
  if (!read_sources_(L, r) || !alloc_products_(r)) { return false; }
  CommandStream st = mc->make_command_stream();
  {
    ComputeEncoder enc = st.begin_compute();
    encode_build_(enc, r);
  }
  std::string why;
  if (!st.commit().wait_ok(&why)) {
    if (const SessionContextIntf* s = mc->session()) {
      s->warn(fmt("[qwen] building layer {} from the raw checkpoint failed "
                  "on the GPU: {}", L, why));
    }
    return false;
  }
  // Sources the products do not keep go with `r`.
  m->_layers[(std::size_t)L] = take_products_(r);
  return true;
}

// ---- the wired pool ---------------------------------------------------

std::size_t
MetalQwenModel::RawStream::wire_layer_(Layer& y, bool on)
{
  std::size_t changed = 0;
  bool stop = false;
  for (SharedBuffer* p : {&y.qw, &y.qs, &y.qb, &y.ow, &y.os, &y.ob,
                          &y.guw, &y.gus, &y.gub, &y.uw, &y.dw, &y.ds,
                          &y.db, &y.in_ln, &y.post_ln, &y.q_norm,
                          &y.k_norm}) {
    if (stop) { break; }
    const std::size_t n = _wire.wire_one(mc, *p, on);
    // Stop at the first refusal: the pool is full, and asking for the
    // rest costs a syscall each to be told so.
    if (on && n == 0 && p->byte_size() > 0 && !p->is_wired() &&
        _wire.refused(mc, *p)) {
      stop = true;
      break;
    }
    changed += n;
  }
  return changed;
}

std::size_t
MetalQwenModel::RawStream::wire_trunk_(bool on)
{
  // The checkpoint's CACHED weights -- for a MOSS model its embedding
  // tables and output heads, which every token reads -- and the final
  // norm. The trunk goes before any layer: a layer is the shed-able half.
  std::size_t changed = 0;
  if (m->_ws) {
    m->_ws->for_each_weight([&](SharedBuffer& b) {
      changed += _wire.wire_one(mc, b, on);
    });
  }
  // What the owner built from the checkpoint and handed over (a MOSS
  // model's w8 embedding table and text head): trunk too, and not the
  // set's, so the walk above never sees it.
  for (SharedBuffer& b : m->_trunk_extra) {
    changed += _wire.wire_one(mc, b, on);
  }
  changed += _wire.wire_one(mc, m->_final_ln, on);
  return changed;
}

void
MetalQwenModel::RawStream::wire_preloaded()
{
  if (!_wire.on()) { return; }
  wire_trunk_(true);
  for (Layer& y : m->_layers) {
    const std::size_t want = unique_bytes_(
        {&y.qw, &y.qs, &y.qb, &y.ow, &y.os, &y.ob, &y.guw, &y.gus, &y.gub,
         &y.uw, &y.dw, &y.ds, &y.db, &y.in_ln, &y.post_ln, &y.q_norm,
         &y.k_norm});
    if (want == 0) { continue; }
    if (!_wire.wirable(want)) { break; }
    _wire.note_wired(mc, wire_layer_(y, true), want);
  }
}

// ---- residency --------------------------------------------------------

void
MetalQwenModel::RawStream::resident_pages_(std::size_t* examined,
                                           std::size_t* incore,
                                           std::size_t* paged_out) const
{
  *examined = 0;
  *incore = 0;
  *paged_out = 0;
  auto add = [&](const SharedBuffer& p) {
    if (p.byte_size() == 0 || p.is_wired()) { return; }
    if (!GenerativeModelManager::pool_wirable(p)) { return; }
    const auto r = p.page_residency(64);
    if (!r.valid) { return; }
    *examined += r.examined;
    *incore += r.incore;
    *paged_out += r.paged_out;
  };
  for (std::size_t L = 0; L < held.size(); ++L) {
    if (!held[L]) { continue; }
    const Layer& y = m->_layers[L];
    for (const SharedBuffer* p : {&y.qw, &y.qs, &y.qb, &y.ow, &y.os, &y.ob,
                                  &y.guw, &y.gus, &y.gub, &y.uw, &y.dw,
                                  &y.ds, &y.db}) {
      add(*p);
    }
  }
}

std::size_t
MetalQwenModel::RawStream::evict_tail()
{
  // From the END: the next forward starts at layer 0, so the deepest
  // resident layer is the one whose re-read is furthest away.
  for (int L = (int)held.size() - 1; L >= 0; --L) {
    if (!held[(std::size_t)L] || L == _pending) { continue; }
    Layer& y = m->_layers[(std::size_t)L];
    const std::size_t n = unique_bytes_(
        {&y.qw, &y.qs, &y.qb, &y.ow, &y.os, &y.ob, &y.guw, &y.gus, &y.gub,
         &y.uw, &y.dw, &y.ds, &y.db, &y.in_ln, &y.post_ln, &y.q_norm,
         &y.k_norm});
    _wire.note_unwired(wire_layer_(y, false));
    y = Layer{};
    y.is_full = true;
    held[(std::size_t)L] = 0;
    return n;
  }
  return 0;
}

void
MetalQwenModel::RawStream::begin_forward()
{
  if (!streaming) { return; }
  failed = false;
  _slots.begin_forward();
  if (_wire.on()) {
    if (_wire.retry(mc, kept_bytes)) { _resid.note_landscape_changed(); }
    wire_trunk_(true);
  }
  const auto mb = mc->memory_budget();
  _resid.begin_forward(mb, [this]() -> std::size_t { return evict_tail(); });
  bool shortfall = false;
  // The page walk is not free, so only when our own compressed footprint
  // moved -- the one sign the box may have taken resident pages back.
  if (_resid.count() > 0 && _resid.self_compression_grew(mb.self_compressed)) {
    std::size_t examined = 0, incore = 0, paged_out = 0;
    resident_pages_(&examined, &incore, &paged_out);
    if (examined > 0 && incore < examined) {
      shortfall = true;
      const std::size_t freed = _resid.note_weight_residency(
          examined, incore, [this]() -> std::size_t { return evict_tail(); });
      if (const SessionContextIntf* s = mc->session()) {
        s->log_normal(fmt(
            "[qwen] resident layers are only {}% in RAM ({} of {} sampled "
            "pages paged out, {} MB wired) -- released {} MB, now {} layers "
            "resident",
            (int)(100.0 * (double)incore / (double)examined), paged_out,
            examined, _wire.wired_bytes() >> 20, freed >> 20,
            _resid.count()));
      }
    }
  }
  if (!shortfall) { _resid.note_healthy_forward(); }
}

int
MetalQwenModel::RawStream::next_streamed_(int L) const
{
  const int n = (int)held.size();
  for (int i = L + 1; i < n; ++i) { if (!held[(std::size_t)i]) { return i; } }
  // Past the last layer, the next forward's first streamed layer: an
  // autoregressive decode runs its next forward straight after this one.
  for (int i = 0; i <= L && i < n; ++i) {
    if (!held[(std::size_t)i]) { return i; }
  }
  return -1;
}

bool
MetalQwenModel::RawStream::enter(int L, CommandStream& st,
                                 ComputeEncoder& enc)
{
  // Whatever is encoded so far reads only resident layers: hand it to the
  // GPU now, so it runs while this layer's read lands.
  enc.end();
  (void)st.commit();
  const RawLayer* b = _slots.acquire(L);
  enc = st.begin_compute();
  if (b == nullptr) {
    failed = true;
    if (const SessionContextIntf* s = mc->session()) {
      s->error(fmt("[qwen] streamed layer {} could not be read", L));
    }
    return false;
  }
  encode_build_(enc, *b);
  m->_layers[(std::size_t)L] = alias_layer_(b->prod);
  _pending = L;
  return true;
}

bool
MetalQwenModel::RawStream::finish(CommandStream& st, ComputeEncoder& enc)
{
  if (_pending < 0) { return true; }
  const int L = _pending;
  _pending = -1;
  enc.end();
  auto fence = st.commit();
  // Under this layer's GPU work: that window is the whole point.
  _slots.prefetch(next_streamed_(L));
  std::string why;
  const bool ok = fence.wait_ok(&why);
  bool kept = false;
  // AFTER the wait, so nothing encoded still reads the slot; promotion
  // clones out of it, so the slot stays usable for the read in flight.
  // Asked with what promotion KEEPS -- the products, not the sources.
  if (ok && _wire.wirable(kept_bytes) && _resid.admit(mc, kept_bytes)) {
    RawLayer keep;
    if (_slots.promote_into(keep)) {
      Layer& y = m->_layers[(std::size_t)L];
      y = take_products_(keep);
      held[(std::size_t)L] = 1;
      // Wired LAST, after every write this layer will ever get.
      _wire.note_wired(mc, wire_layer_(y, true), kept_bytes);
      _resid.note_admitted(kept_bytes);
      kept = true;
    }
  }
  if (!kept) {
    Layer& y = m->_layers[(std::size_t)L];
    y = Layer{};
    y.is_full = true;
  }
  enc = st.begin_compute();
  if (!ok) {
    failed = true;
    if (const SessionContextIntf* s = mc->session()) {
      s->error(fmt("[qwen] streamed layer {} failed on the GPU: {}", L, why));
    }
  }
  return ok;
}

std::size_t
MetalQwenModel::RawStream::resident_bytes() const
{
  std::size_t n = m->_final_ln.byte_size();
  for (std::size_t L = 0; L < m->_layers.size(); ++L) {
    if ((int)L == _pending) { continue; }   // a slot's, counted below
    const Layer& y = m->_layers[L];
    n += unique_bytes_({&y.qw, &y.qs, &y.qb, &y.ow, &y.os, &y.ob, &y.guw,
                        &y.gus, &y.gub, &y.uw, &y.dw, &y.ds, &y.db,
                        &y.in_ln, &y.post_ln, &y.q_norm, &y.k_norm});
  }
  if (streaming && _slots.on()) {
    n += slot_bytes * (_slots.paired() ? 2u : 1u);
  }
  return n;
}

int
MetalQwenModel::RawStream::resident_count() const
{
  if (!streaming) { return (int)held.size(); }
  int n = 0;
  for (char h : held) { n += h ? 1 : 0; }
  return n;
}

void
MetalQwenModel::RawStream::shutdown()
{
  if (_shut) { return; }
  _shut = true;
  // A read may be in flight into a slot; nothing below may free one first.
  _slots.join();
  // GIVE THE POOL BACK. Freeing a wired buffer unwires it in the kernel,
  // but only unwire_from_pool decrements the pool's counter.
  if (_wire.on()) {
    for (Layer& y : m->_layers) { _wire.note_unwired(wire_layer_(y, false)); }
    wire_trunk_(false);
  }
  _slots.reset();
}

// ---- MetalQwenModel's side --------------------------------------------

bool
MetalQwenModel::streams_decode() const
{
  return _rs != nullptr && _rs->streaming;
}

void
MetalQwenModel::set_residency_reserve(std::size_t bytes)
{
  if (_rs) { _rs->_resid.set_reserve(bytes); }
}

void
MetalQwenModel::set_residency_schedule(int forwards)
{
  if (!_rs || !_rs->streaming) { return; }
  _rs->_wire.new_run();
  _rs->_resid.set_schedule(forwards, _cfg.n_layers, _rs->kept_bytes,
                           _rs->_wire.on(), _mc->memory_budget());
}

std::size_t
MetalQwenModel::release_resident_layers(std::size_t bytes)
{
  if (!_rs || !_rs->streaming) { return 0; }
  return _rs->_resid.release(
      bytes, [this]() -> std::size_t { return _rs->evict_tail(); });
}

SharedBuffer&
MetalQwenModel::adopt_trunk(SharedBuffer b)
{
  _trunk_extra.push_back(std::move(b));
  return _trunk_extra.back();
}

void
MetalQwenModel::wire_resident()
{
  if (!_rs) { return; }
  if (_rs->streaming) {
    if (_rs->_wire.on()) { _rs->wire_trunk_(true); }
    return;
  }
  _rs->wire_preloaded();
}

int
MetalQwenModel::resident_layer_count() const
{
  if (_rs) { return _rs->resident_count(); }
  return _stream_layers ? _pinned_layers : _cfg.n_layers;
}

std::uint64_t
MetalQwenModel::resident_bytes() const
{
  if (_rs) { return _rs->resident_bytes(); }
  std::uint64_t n = _final_ln.byte_size();
  for (const Layer& y : _layers) {
    n += unique_bytes_({&y.qw, &y.qs, &y.qb, &y.ow, &y.os, &y.ob, &y.guw,
                        &y.gus, &y.gub, &y.uw, &y.dw, &y.ds, &y.db,
                        &y.in_ln, &y.post_ln, &y.q_norm, &y.k_norm,
                        &y.kw, &y.ks, &y.kb, &y.vw, &y.vs, &y.vb});
  }
  return n;
}

}  // namespace vpipe::genai
