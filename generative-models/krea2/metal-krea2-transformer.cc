#include "generative-models/shared/mma-tile.h"
#include "generative-models/krea2/metal-krea2-transformer.h"

#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/lora-names.h"

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

namespace vpipe {
namespace genai {

using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
using metal_compute::ComputeEncoder;
using metal_compute::CommandStream;

namespace {

// C++ mirror of mlx::steel::AttnParams (steel/attn/params.h) -- the param block
// the vendored steel flash-attention kernel reads. Identical layout to the LM's
// copy (metal-llama-model.cc) and the vision tower's (metal-qwen-vision.cc).
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

// bf16 <-> f32 host helpers. The Krea-2 DiT runs in bf16 (like the QIE + FLUX.2
// siblings on the same Qwen-Image MMDiT stack): f16's 65504 range overflows on
// real conditioning outliers through the deep 28-block residual/attention
// stream, and the reference itself runs bf16. bf16's f32 exponent range holds
// them.
// Namespace for this class's derived-tensor cache keys. A WeightSet is
// shared by everything reading one checkpoint, so a key has to say which
// class's transform produced the bytes, not just which tensor.
constexpr const char* kKey = "krea2-dit/";

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

// Load a ZERO-CENTERED RMSNorm weight, folding the +1 (the checkpoint stores
// the weight as an offset from 1.0; Krea2RMSNorm applies (1+weight)).
SharedBuffer
to_f16_norm_(const MetalLlamaWeights& wts, MetalCompute* mc,
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
  auto* d = static_cast<std::uint16_t*>(out.contents());
  auto put = [&](std::size_t i, float f) { d[i] = f32_to_bf16_(f + 1.0f); };
  if (info->dtype == "F32") {
    const auto* s = static_cast<const float*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { put(i, s[i]); }
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { put(i, (float)s[i]); }
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { put(i, bf16_to_f32_(s[i])); }
  } else {
    return {};
  }
  return out;
}

}  // namespace

// Load a Linear weight: an affine-quantized triple (codes/scales/qbias) when
// `<name>.scales` is present in a quantized checkpoint, else dense f16.
SharedBuffer
MetalKrea2Transformer::elt_(WeightSet& ws, const std::string& nm, Retain r,
                            bool norm)
{
  auto build = [&]() {
    return norm ? to_f16_norm_(ws.src(), _mc, nm)
                : to_f16_(ws.src(), _mc, nm);
  };
  if (r == Retain::Streamed) { return ws.stream_derived(build); }
  // The (1 + w) norm fold produces DIFFERENT bytes from the plain
  // narrowing, so it gets its own key -- sharing one would hand a
  // caller the wrong tensor.
  return ws.derived(
      std::string(kKey) + (norm ? "eltn|" : "elt|") + nm, build);
}

MetalKrea2Transformer::QWeight
MetalKrea2Transformer::load_qw_(WeightSet& ws, const std::string& name,
                                Retain r)
{
  QWeight qw;
  const MetalLlamaWeights& wts = ws.src();
  const auto* si = wts.info(name + ".scales");
  const auto* ci = wts.info(name + ".weight");
  if (_quant_bits > 0 && si != nullptr && ci != nullptr &&
      si->shape.size() == 2 && ci->shape.size() == 2) {
    // Per-weight bit-width (mixed precision): codes cols = K*bits/32, scales
    // cols = K/group -> bits = codes_cols*32 / (scales_cols*group).
    const long gcols = ci->shape[1];                 // K*bits/32
    const long scols = si->shape[1];                 // K/group
    const long K = scols * (long)_quant_group;
    const int bits = K > 0 ? (int)(gcols * 32 / K) : 0;
    qw.bits = (bits == 8) ? 8 : 4;
    // Zero-copy (read-only mmap view) when enabled: the codes/scales/biases go
    // straight to the GEMM and are never modified on the CPU, so they are safe
    // to alias the file. Otherwise (default) an owned copy.
    // Codes are U32 (raw; mmap-aliased when enabled). Scales/biases are F16 on
    // disk but the affine kernels are now bf16 (the DiT runs bf16), so convert
    // them F16->bf16 via to_f16_ (which emits bf16). Mirrors the QIE/FLUX.2 path.
    const auto res = _mmap_weights ? WeightSet::Residency::Mapped
                                   : WeightSet::Residency::Copied;
    qw.codes = r == Retain::Streamed
                   ? ws.stream_tensor(name + ".weight", _mc, res)
                   : ws.tensor(name + ".weight", _mc, res);
    qw.scales = elt_(ws, name + ".scales", r);   // -> bf16 (group scale)
    qw.qbias  = elt_(ws, name + ".biases", r);   // -> bf16 (group min)
    if (!qw.codes.empty() && !qw.scales.empty() && !qw.qbias.empty()) {
      qw.quantized = true;
      return qw;
    }
    qw.codes = {}; qw.scales = {}; qw.qbias = {};
  }
  qw.w = elt_(ws, name + ".weight", r);              // dense fallback
  return qw;
}

bool
MetalKrea2Transformer::load_block_(WeightSet& ws,
                                   const std::string& pre, Block& b,
                                   bool main_block, Retain r,
                                   bool keep_split)
{
  if (main_block) {
    b.sst = elt_(ws, pre + "scale_shift_table", r);
    if (b.sst.empty()) { return false; }
  }
  b.n1 = elt_(ws, pre + "norm1.weight", r, true);
  b.n2 = elt_(ws, pre + "norm2.weight", r, true);
  b.q  = load_qw_(ws, pre + "attn.to_q", r);
  b.k  = load_qw_(ws, pre + "attn.to_k", r);
  b.v  = load_qw_(ws, pre + "attn.to_v", r);
  b.gate = load_qw_(ws, pre + "attn.to_gate", r);
  b.qn = elt_(ws, pre + "attn.norm_q.weight", r, true);
  b.kn = elt_(ws, pre + "attn.norm_k.weight", r, true);
  b.o  = load_qw_(ws, pre + "attn.to_out.0", r);
  b.ff_gate = load_qw_(ws, pre + "ff.gate", r);
  b.ff_up   = load_qw_(ws, pre + "ff.up", r);
  b.ff_down = load_qw_(ws, pre + "ff.down", r);
  // Fused-FF interleave (main blocks): gate|up -> one [2*ffn, K] quantized
  // weight (row 2g = gate feature g, 2g+1 = up feature g) -- the layout both
  // affine_qmm_swiglu's register-local epilogue (steel) and the matmul2d +
  // swiglu_interleaved fold (M5 mma) expect. Requires both sides quantized at
  // the SAME bit width (a mixed gate/up pair keeps the split path). The split
  // weights are released after a successful build.
  if (_fuse_ff && main_block && b.ff_gate.quantized && b.ff_up.quantized &&
      b.ff_gate.bits == b.ff_up.bits) {
    const std::size_t FF = (std::size_t)_cfg.ffn;
    const std::size_t wrow = (std::size_t)_cfg.hidden * b.ff_gate.bits / 32;
    const std::size_t grow = (std::size_t)_cfg.hidden / _quant_group;
    // Interleave gate|up rows into one buffer: row 2g = gate g, 2g+1 = up g.
    auto row2 = [&](const SharedBuffer& g0, const SharedBuffer& u0,
                    std::size_t rb) -> SharedBuffer {
      if (g0.empty() || u0.empty()) { return {}; }
      SharedBuffer dst = _mc->make_shared_buffer(2 * FF * rb);
      if (dst.empty()) { return {}; }
      auto* d = static_cast<std::uint8_t*>(dst.contents());
      const auto* gp = static_cast<const std::uint8_t*>(g0.contents());
      const auto* up = static_cast<const std::uint8_t*>(u0.contents());
      for (std::size_t i = 0; i < FF; ++i) {
        std::memcpy(d + (2 * i) * rb, gp + i * rb, rb);
        std::memcpy(d + (2 * i + 1) * rb, up + i * rb, rb);
      }
      return dst;
    };
    // The interleaved weight is the largest tensor in a block, so it
    // goes through the set like any other derived tensor: two models
    // over this checkpoint pay for the relayout once. Streamed blocks
    // rebuild it per forward and keep nothing, as they must. Building
    // INSIDE the builder matters -- allocating first and filling after
    // would rewrite a cache hit's bytes under a peer model's feet.
    auto fused = [&](const char* tag, const SharedBuffer& g0,
                     const SharedBuffer& u0,
                     std::size_t rb) -> SharedBuffer {
      auto mk = [&]() { return row2(g0, u0, rb); };
      if (r == Retain::Streamed) { return ws.stream_derived(mk); }
      return ws.derived(std::string(kKey) + "gu|" + pre + "ff|" + tag, mk);
    };
    QWeight gu;
    gu.codes  = fused("codes", b.ff_gate.codes, b.ff_up.codes, wrow * 4);
    gu.scales = fused("scales", b.ff_gate.scales, b.ff_up.scales, grow * 2);
    gu.qbias  = fused("qbias", b.ff_gate.qbias, b.ff_up.qbias, grow * 2);
    if (!gu.codes.empty() && !gu.scales.empty() && !gu.qbias.empty()) {
      gu.quantized = true;
      gu.bits = b.ff_gate.bits;
      b.ff_gu = std::move(gu);
      if (!keep_split) {
        b.ff_gate = QWeight{};
        b.ff_up = QWeight{};
      }
    }
  }
  const bool ff_in_ok = !b.ff_gu.empty()
                        || (!b.ff_gate.empty() && !b.ff_up.empty());
  return !b.n1.empty() && !b.n2.empty() && !b.q.empty() && !b.k.empty() &&
         !b.v.empty() && !b.gate.empty() && !b.qn.empty() && !b.kn.empty() &&
         !b.o.empty() && ff_in_ok && !b.ff_down.empty();
}


// ---- the streamed blocks' reusable destinations ------------------------

// Shorthand for the per-tensor placement the refill needs stated.
using P = vpipe::genai::Placement;

// Every (checkpoint name, destination, placement) of block L, in a fixed
// order.
//
// THE NORMS ARE `kDerived` AND THAT IS THE WHOLE TRAP HERE. This model
// stores its RMSNorm weights zero-centred and folds the +1 at load
// (to_f16_norm_), so the bytes in the destination are NOT the bytes in
// the checkpoint. A raw refill would place the checkpoint's and be
// silently wrong -- the buffer is the right size and full of plausible
// numbers, and the only visible symptom would be a slightly different
// picture. `ff_gu` is derived for the same reason and is rebuilt by
// weave_into_ after the refill, so it is not enumerated at all.
void
MetalKrea2Transformer::each_block_tensor_(
    int L, Block& b, const BlockSlots<Block>::TensorFn& fn) const
{
  const std::string p = "transformer_blocks." + std::to_string(L) + ".";
  const auto qw = [&fn](const std::string& base, QWeight& w) {
    if (w.quantized) {
      // Codes are the checkpoint's own u32 words; the group scales and
      // minima are read as bf16 (F16 in the pack, converted in place).
      fn(base + ".weight", w.codes, P::kRaw);
      fn(base + ".scales", w.scales, P::kBf16);
      fn(base + ".biases", w.qbias, P::kBf16);
    } else {
      fn(base + ".weight", w.w, P::kBf16);
    }
  };
  // Empty in a non-main block, and BlockSlots skips an empty
  // destination -- so naming it unconditionally is safe.
  fn(p + "scale_shift_table", b.sst, P::kBf16);
  fn(p + "norm1.weight", b.n1, P::kDerived);
  fn(p + "norm2.weight", b.n2, P::kDerived);
  qw(p + "attn.to_q", b.q);
  qw(p + "attn.to_k", b.k);
  qw(p + "attn.to_v", b.v);
  qw(p + "attn.to_gate", b.gate);
  fn(p + "attn.norm_q.weight", b.qn, P::kDerived);
  fn(p + "attn.norm_k.weight", b.kn, P::kDerived);
  qw(p + "attn.to_out.0", b.o);
  qw(p + "ff.gate", b.ff_gate);
  qw(p + "ff.up", b.ff_up);
  qw(p + "ff.down", b.ff_down);
}

// Re-interleave gate|up into a destination that ALREADY exists. The
// fused weight has no checkpoint name -- it is a transform of two
// tensors -- so a refill cannot address it and this runs after one.
bool
MetalKrea2Transformer::weave_into_(const QWeight& gate, const QWeight& up,
                                   QWeight& dst) const
{
  if (dst.empty()) { return true; }        // not fused: nothing to rebuild
  if (gate.empty() || up.empty()) { return false; }
  const std::size_t FF = (std::size_t)_cfg.ffn;
  if (FF == 0) { return false; }
  bool ok = true;
  const auto one = [&](const SharedBuffer& g, const SharedBuffer& u,
                       SharedBuffer& d) {
    if (!ok || d.empty()) { return; }
    if (g.empty() || u.empty()) { ok = false; return; }
    const std::size_t rb = g.byte_size() / FF;
    if (rb == 0 || u.byte_size() / FF != rb ||
        d.byte_size() != 2 * FF * rb) {
      ok = false;
      return;
    }
    const auto* gs = static_cast<const std::uint8_t*>(g.contents());
    const auto* us = static_cast<const std::uint8_t*>(u.contents());
    auto* dd = static_cast<std::uint8_t*>(d.contents());
    for (std::size_t i = 0; i < FF; ++i) {
      std::memcpy(dd + (2 * i) * rb, gs + i * rb, rb);
      std::memcpy(dd + (2 * i + 1) * rb, us + i * rb, rb);
    }
  };
  one(gate.codes, up.codes, dst.codes);
  one(gate.scales, up.scales, dst.scales);
  one(gate.qbias, up.qbias, dst.qbias);
  return ok;
}

// Allocate `dst` with `src`'s shapes and flags, optionally copying the
// bytes. One function for two uses: a promotion and the second slot
// differ only in whether the contents come along.
bool
MetalKrea2Transformer::clone_block_(const Block& src, Block& dst,
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
    d.quantized = s.quantized;
    d.bits      = s.bits;
    one(s.w, d.w); one(s.codes, d.codes);
    one(s.scales, d.scales); one(s.qbias, d.qbias);
  };
  one(src.n1, dst.n1); one(src.n2, dst.n2);
  qw(src.q, dst.q); qw(src.k, dst.k); qw(src.v, dst.v);
  qw(src.gate, dst.gate);
  one(src.qn, dst.qn); one(src.kn, dst.kn);
  qw(src.o, dst.o);
  qw(src.ff_gate, dst.ff_gate); qw(src.ff_up, dst.ff_up);
  qw(src.ff_down, dst.ff_down); qw(src.ff_gu, dst.ff_gu);
  one(src.sst, dst.sst);
  if (!ok) { dst = Block{}; }
  return ok;
}

SharedBuffer
MetalKrea2Transformer::rebuild_one_(const std::string& nm, P how)
{
  if (!_ws) { return {}; }
  if (how == P::kRaw) {
    const auto res = _mmap_weights ? WeightSet::Residency::Mapped
                                   : WeightSet::Residency::Copied;
    return _ws->stream_tensor(nm, _mc, res);
  }
  // A DERIVED tensor is a norm: the (1 + w) fold. Everything else that
  // could not be placed raw is the plain narrowing. The two produce
  // DIFFERENT bytes, which is why elt_ keys them apart, and why the
  // placement has to travel with the name rather than be guessed here.
  return elt_(*_ws, nm, Retain::Streamed, how == P::kDerived);
}

void
MetalKrea2Transformer::configure_slots_()
{
  BlockSlots<Block>::Ops o;
  o.each = [this](int L, Block& b, const BlockSlots<Block>::TensorFn& fn) {
    each_block_tensor_(L, b, fn);
  };
  o.rebuild_one = [this](const std::string& nm, P how) {
    return rebuild_one_(nm, how);
  };
  o.build = [this](int L, Block& b) {
    // KEEP THE SPLIT gate|up in a slot. The ordinary load releases them
    // once the fused weight exists, which is right for a resident block
    // and impossible for a refillable one: the fuse has to be rebuilt
    // from them after every read.
    return _ws && load_block_(*_ws,
                              "transformer_blocks." + std::to_string(L) + ".",
                              b, /*main_block=*/true, Retain::Streamed,
                              /*keep_split=*/true);
  };
  o.clone = [this](const Block& s, Block& d, bool copy) {
    return clone_block_(s, d, copy);
  };
  o.bytes = [](const Block& b) { return block_bytes_(b); };
  o.empty = [](const Block& b) { return b.q.empty(); };
  o.post_refill = [this](int, Block& b) {
    return weave_into_(b.ff_gate, b.ff_up, b.ff_gu);
  };
  _slots.set_weight_set(_ws.get());
  _slots.configure(_mc, std::move(o), "MetalKrea2Transformer",
                   "VPIPE_KREA2_NO_SLOTS");
}

MetalKrea2Transformer::~MetalKrea2Transformer()
{
  // GIVE THE POOL BACK. Freeing a wired buffer unwires it in the kernel,
  // so the machine recovers either way -- but the pool's own counter
  // would not, and a DiT freed for the vae-decode and reloaded on the
  // next prompt (free_krea2_dit_for_decode_) would leak its whole share
  // of the budget per prompt until nothing could wire at all.
  if (_wire.on()) {
    wire_fixed_(false);
    for (Block& b : _blocks) { wire_block_(b, false); }
  }
}

// ---- the module table ----------------------------------------------
//
// Krea-2's own projection names are the DIFFUSERS spelling -- attn.to_q
// / to_k / to_v / to_gate / to_out.0, ff.gate / ff.up / ff.down -- and
// that is what these rows carry. The ai-toolkit / ComfyUI spelling
// (`diffusion_model.blocks.N.attn.wq`, `mlp.gate`) is NOT a second
// table: the adapter is opened with shared/lora-names.h's map
// installed, which indexes such a file under these names, so one list
// serves both conventions and cannot describe them differently.
//
// THREE containers, not one. The main stack plus the text-fusion
// tower's layerwise and refiner blocks -- every one of them the same
// `Block`, so one row type serves. The tower is not optional: an
// adapter that touches it and is bound only on the main blocks applies
// in part, silently.
std::vector<MetalKrea2Transformer::LoraModuleRef>
MetalKrea2Transformer::lora_modules_(const Config& c)
{
  const int H = c.hidden, F = c.ffn;
  const int qd = c.n_heads * c.head_dim;
  const int kd = c.n_kv_heads * c.head_dim;
  const int TH = c.text_hidden, TF = c.text_ffn;
  std::vector<LoraModuleRef> out;
  out.reserve((std::size_t)(c.n_layers + c.n_layerwise + c.n_refiner) * 8);
  // One block's eight projections. The fusion tower has no GQA
  // (text_kv_heads == text_heads), so its k/v are as wide as its q.
  auto stack = [&](LoraStack st, const char* pre, int n, int nq, int nkv,
                   int hid, int ffn) {
    for (int i = 0; i < n; ++i) {
      const std::string p = pre + std::to_string(i) + ".";
      auto add = [&](const char* nm, int on, int k,
                     lora::Factors BlockLora::*dst) {
        out.push_back({p + nm, on, k, st, i, dst});
      };
      add("attn.to_q",     nq,  hid, &BlockLora::q);
      add("attn.to_k",     nkv, hid, &BlockLora::k);
      add("attn.to_v",     nkv, hid, &BlockLora::v);
      add("attn.to_gate",  hid, hid, &BlockLora::gate);
      add("attn.to_out.0", hid, hid, &BlockLora::o);
      add("ff.gate",       ffn, hid, &BlockLora::ff_gate);
      add("ff.up",         ffn, hid, &BlockLora::ff_up);
      add("ff.down",       hid, ffn, &BlockLora::ff_down);
    }
  };
  stack(LoraStack::kMain, "transformer_blocks.", c.n_layers, qd, kd, H, F);
  stack(LoraStack::kLayerwise, "text_fusion.layerwise_blocks.",
        c.n_layerwise, TH, TH, TH, TF);
  stack(LoraStack::kRefiner, "text_fusion.refiner_blocks.",
        c.n_refiner, TH, TH, TH, TF);
  return out;
}

genai::lora::Adapter::Rename
MetalKrea2Transformer::lora_rename()
{
  return &lora::remap_ai_toolkit_module;
}

std::vector<MetalKrea2Transformer::LoraModule>
MetalKrea2Transformer::lora_module_list(const Config& cfg)
{
  std::vector<LoraModule> out;
  for (const auto& m : lora_modules_(cfg)) {
    out.push_back({m.name, m.n, m.k});
  }
  return out;
}

// Read the adapter and bind its factors to the modules they name.
//
// Everything about reading the file (the publisher prefixes, the
// ai-toolkit name map, alpha/rank, bf16, the shape check) is shared;
// what is here is the module list, which is the model.
bool
MetalKrea2Transformer::bind_lora_(const LoraSpec& spec, LoraSlot& slot,
                                  std::string* err)
{
  std::string oerr;
  auto ad = lora::Adapter::open(spec.path, _mc, &oerr, lora_rename());
  if (!ad) {
    if (err != nullptr) { *err = "krea2 lora: " + oerr; }
    return false;
  }
  slot.scale = spec.scale;

  slot.blocks.clear();
  slot.blocks.resize((std::size_t)_cfg.n_layers);
  slot.layerwise.clear();
  slot.layerwise.resize((std::size_t)_cfg.n_layerwise);
  slot.refiner.clear();
  slot.refiner.resize((std::size_t)_cfg.n_refiner);
  for (const auto& m : lora_modules_(_cfg)) {
    std::vector<BlockLora>& v = m.stack == LoraStack::kMain ? slot.blocks
                              : m.stack == LoraStack::kLayerwise
                                    ? slot.layerwise
                                    : slot.refiner;
    ad->bind(m.name, m.n, m.k, &(v[(std::size_t)m.block].*m.dst));
  }
  slot.modules  = ad->modules();
  slot.max_rank = ad->max_rank();
  if (slot.modules == 0) {
    slot.blocks.clear();
    slot.layerwise.clear();
    slot.refiner.clear();
    if (err != nullptr) {
      *err = "krea2 lora: '" + spec.path +
             "' adapts none of this model's modules, under either the "
             "diffusers or the ai-toolkit/ComfyUI names. A LoKr adapter "
             "(lokr_w1/lokr_w2) is not a runtime LoRA -- fuse it with the "
             "lora-fuse stage instead";
    }
    return false;
  }
  if (_mc->session() != nullptr) {
    _mc->session()->info(fmt("MetalKrea2Transformer: {}",
                                   ad->summary(spec.path, slot.scale)));
  }
  return true;
}

// The live adapters for one projection of one block, in slot order.
genai::lora::Stack
MetalKrea2Transformer::lora_at_(LoraStack which, int block,
                                lora::Factors BlockLora::* dst) const
{
  lora::Stack st;
  for (const LoraSlot& sl : _lora_slots) {
    const std::vector<BlockLora>& v = which == LoraStack::kMain ? sl.blocks
                                    : which == LoraStack::kLayerwise
                                          ? sl.layerwise
                                          : sl.refiner;
    if (block < 0 || (std::size_t)block >= v.size()) { continue; }
    st.add(v[(std::size_t)block].*dst, sl.scale,
           _lora_scratch_rows * (std::size_t)sl.rank_prefix);
  }
  return st;
}

int
MetalKrea2Transformer::lora_modules() const
{
  int n = 0;
  for (const LoraSlot& sl : _lora_slots) { n += sl.modules; }
  return n;
}

int
MetalKrea2Transformer::lora_modules(int slot) const
{
  return (slot >= 0 && (std::size_t)slot < _lora_slots.size())
      ? _lora_slots[(std::size_t)slot].modules : 0;
}

void
MetalKrea2Transformer::set_lora_scale(int slot, float s)
{
  if (slot >= 0 && (std::size_t)slot < _lora_slots.size()) {
    _lora_slots[(std::size_t)slot].scale = s;
  }
}

float
MetalKrea2Transformer::lora_scale(int slot) const
{
  return (slot >= 0 && (std::size_t)slot < _lora_slots.size())
      ? _lora_slots[(std::size_t)slot].scale : 0.0f;
}

std::unique_ptr<MetalKrea2Transformer>
MetalKrea2Transformer::load(const std::string& model_dir, MetalCompute* mc,
                            const Config& cfg, bool stream_blocks,
                            const std::vector<LoraSpec>& loras)
{
  return load(WeightSet::open(model_dir, nullptr), mc, cfg, stream_blocks,
              loras);
}

std::unique_ptr<MetalKrea2Transformer>
MetalKrea2Transformer::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                            const Config& cfg, bool stream_blocks,
                            const std::vector<LoraSpec>& loras)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  const std::string model_dir = ws_in->dir();

  auto m = std::unique_ptr<MetalKrea2Transformer>(new MetalKrea2Transformer());
  m->_ws = ws_in;
  WeightSet& ws = *m->_ws;
  // Everything loaded from here to the end of load() is RETAINED for the
  // model's life. The streamed blocks are read in forward(), and there
  // only.
  const Retain r = Retain::Cached;
  m->_mc = mc;
  m->_cfg = cfg;
  m->_stream_blocks = stream_blocks;
  // BEFORE the first weight is read, because it decides how they are read.
  m->_wire.open(mc);
  // The reusable read destinations. Configured whether or not this run
  // streams: they cost nothing until the first streamed block asks for
  // one, and a model that preloads never gets there.
  m->configure_slots_();
  // Zero-copy (mmap) quantized weights -- see _mmap_weights. On by default;
  // VPIPE_KREA2_NO_MMAP_WEIGHTS reverts to copying loads (e.g. for a model on a
  // slow/network mount, where GPU page-faults into the file could stall).
  // Retains the source mmap for the model's lifetime (below).
  //
  // OFF WHEN STREAMING, which it was not before. A streamed block that is a
  // mapped view is not this model's memory at all: keeping it "resident"
  // keeps the page CACHE warm, which is the self-generated signal
  // block-residency.h exists to avoid trusting -- and page_residency() then
  // measures the cache rather than anything the model owns. It also cannot
  // be wired. Off with the pool on for the same reason (shared/wired-pool.h).
  m->_mmap_weights = weights_may_be_mapped(stream_blocks, m->_wire.on()) &&
                     std::getenv("VPIPE_KREA2_NO_MMAP_WEIGHTS") == nullptr;

  // Detect an affine-quantized checkpoint via config.json's `quantization`
  // block ({bits, group_size}); the quantizable Linears then load as
  // codes/scales/biases and gemm_ dispatches affine_qmm_steel.
  {
    namespace fs = std::filesystem;
    std::ifstream in(fs::path(model_dir) / "config.json");
    if (in) {
      FlexData cfgj = FlexData::from_json(in);
      if (cfgj.is_object()) {
        auto obj = cfgj.as_object();
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

  // The Krea-2 DiT runs in bf16 (f16's 65504 range overflows on real
  // conditioning through the 28-block residual/attention stream, and the
  // reference is bf16). Load the bf16 metallibs (VPIPE_ELT=bfloat); entry-point
  // names keep the "_f16" label. gelu_tanh comes from llm_elementwise's
  // VPIPE_ELT gelu_tanh_ff_f16 (the vision gelu_tanh_f16 is half-only).
  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_lib_sdpa = mc->load_library("sdpa_bf16");
  m->_lib_vis  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_fn_gemm        = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_gemm_bm64   = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_gemm_bm64bn64 = m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  m->_fn_gemm_bias   = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_rms         = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_swiglu      = m->_lib_elt.function("swiglu_f16");
  m->_fn_mul_sigmoid = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_residual    = m->_lib_elt.function("residual_add_f16");
  m->_fn_transpose   = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_sdpa        = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_gelu_tanh   = m->_lib_elt.function("gelu_tanh_ff_f16");
  // Steel flash-attention (head_dim 128) for the joint-sequence GQA attention.
  // Best-effort: if the library / bd128 entry point is missing, forward_dit
  // keeps the scalar sdpa fallback. VPIPE_KREA2_NO_STEEL_ATTN forces scalar.
  m->_lib_attn = mc->load_library("attn_steel");
  m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));
  m->_steel_attn_ok = m->_lib_attn.valid() && !m->_attn_params.empty()
                      && std::getenv("VPIPE_KREA2_NO_STEEL_ATTN") == nullptr;
  // M5 matrix-core NAX steel flash: prefer it when the GPU has matrix cores
  // (Apple10+), like the vision tower. On M4 supports_matrix_cores() is false
  // -> stays on the ALU steel above. VPIPE_KREA2_NO_ATTN_NAX forces ALU steel.
  m->_lib_attn_nax = mc->load_library("attn_steel_nax");
  m->_use_attn_nax = m->_steel_attn_ok && mc->supports_matrix_cores()
                     && m->_lib_attn_nax.valid()
                     && std::getenv("VPIPE_KREA2_NO_ATTN_NAX") == nullptr;
  m->_fn_rope_table  = m->_lib_rope.function("rope_pair_table_ftab_f16");
  m->_fn_adaln       = m->_lib_elt.function("adaln_modulate_f16");
  m->_fn_gated       = m->_lib_elt.function("gated_residual_f16");
  // vec4 twins: same arithmetic per element (so bit-identical), 3-4x the
  // throughput -- one element per thread leaves these at ~37-54 GB/s where the
  // same bytes through a vec4 2-D grid run at 143-181. VPIPE_NO_ELT_V4 reverts.
  if (std::getenv("VPIPE_NO_ELT_V4") == nullptr) {
    m->_fn_adaln4 = m->_lib_elt.function("adaln_modulate_v4_f16");
    m->_fn_gated4 = m->_lib_elt.function("gated_residual_v4_f16");
  }
  m->_fn_colabsmax   = m->_lib_elt.function("col_absmax_f16");
  if (!m->_fn_gemm.valid() || !m->_fn_gemm_bias.valid() ||
      !m->_fn_rms.valid() || !m->_fn_swiglu.valid() ||
      !m->_fn_mul_sigmoid.valid() || !m->_fn_residual.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_gelu_tanh.valid() || !m->_fn_rope_table.valid() ||
      !m->_fn_adaln.valid() || !m->_fn_gated.valid() ||
      !m->_fn_colabsmax.valid()) {
    return nullptr;
  }
  // bf16 block-GEMM tile: default BM64 (measured ~5% faster than BM32 at the DiT
  // block shapes -- it halves f16 weight re-reads at M~=seq; BN64 spills and
  // regresses). VPIPE_KREA2_GEMM_TILE overrides (0 BM32, 1 BM64, 2 BM64/BN64)
  // for A/B; falls back to BM32 if the larger-tile function is absent.
  m->_gemm_tile = 1;
  if (const char* t = std::getenv("VPIPE_KREA2_GEMM_TILE")) {
    m->_gemm_tile = std::atoi(t);
  }
  if ((m->_gemm_tile == 1 && !m->_fn_gemm_bm64.valid()) ||
      (m->_gemm_tile == 2 && !m->_fn_gemm_bm64bn64.valid())) {
    m->_gemm_tile = 0;
  }
  // Affine-quant matmul (steel GEMM) + row-broadcast bias-add for the
  // quantized-Linear path. Both bit widths are loaded so a MIXED-precision
  // checkpoint (per-weight w4/w8 at group g) dispatches by each weight's bits.
  if (m->_quant_bits > 0) {
    m->_lib_qmm = mc->load_library("affine_qmm_steel_bf16");
    const std::string g = "g" + std::to_string(m->_quant_group);
    m->_fn_qmm4 = m->_lib_qmm.function("affine_qmm_steel_w4" + g);
    m->_fn_qmm8 = m->_lib_qmm.function("affine_qmm_steel_w8" + g);
    m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    if (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid() ||
        !m->_fn_bias_add.valid()) {
      return nullptr;
    }
    // BM=64 steel qmm tile for the tall-M (M ~= seq) DiT block GEMMs (g64-only
    // entry points; the lookup fails harmlessly for g32 -> routing skips).
    // VPIPE_KREA2_QMM_TILE: 0 forces the 32x32 tile, 1 (default) BM64,
    // 2 BM128 (only kicks in at seq >= 1024; falls back to BM64 below that).
    m->_qmm_tile = 1;
    if (const char* t = std::getenv("VPIPE_KREA2_QMM_TILE")) {
      m->_qmm_tile = std::atoi(t);
    }
    if (m->_qmm_tile >= 1) {
      m->_fn_qmm4_bm64 =
          m->_lib_qmm.function("affine_qmm_steel_w4" + g + "_bm64");
      m->_fn_qmm8_bm64 =
          m->_lib_qmm.function("affine_qmm_steel_w8" + g + "_bm64");
      if (!m->_fn_qmm4_bm64.valid() || !m->_fn_qmm8_bm64.valid()) {
        m->_qmm_tile = 0;
      }
    }
    if (m->_qmm_tile == 2) {
      m->_fn_qmm4_bm128 =
          m->_lib_qmm.function("affine_qmm_steel_w4" + g + "_bm128");
      m->_fn_qmm8_bm128 =
          m->_lib_qmm.function("affine_qmm_steel_w8" + g + "_bm128");
      if (!m->_fn_qmm4_bm128.valid() || !m->_fn_qmm8_bm128.valid()) {
        m->_qmm_tile = 1;   // BM128 unavailable (e.g. g32) -> stay on BM64
      }
    }
  }

  // M5 matrix-core matmul2d for the block/projection GEMMs. Dense weights feed
  // dense_gemm_mma directly; quantized weights are dequant-expanded into _w_deq
  // (affine_dequant) then run through the SAME dense matmul2d -- the dequant-once
  // -> dense-GEMM shape the LM prefill uses, which matches affine_qmm_steel to
  // f32 rounding. Gated on matrix cores (M4/older keep steel); NO_MMA2 A/B off.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_KREA2_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    // TN=2 tile: the K=6144 block-projection sweet spot (~1.14-1.19x over the
    // plain 128x256 deep tile at M~=seq; x-reuse doubles). NO_TN2 forces the
    // plain 128x256 deep tile for A/B (leaves the fn invalid -> routing skips it).
    if (std::getenv("VPIPE_KREA2_NO_TN2") == nullptr) {
      m->_fn_dense_mma_tn2 =
          m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");
    }
    // Split-K deep-reduction kernel (K=16384 ff-down); opt-out via NO_SPLITK.
    m->_fn_dense_mma_splitk =
        m->_lib_dense_mma.function("dense_gemm_mma_splitk_n128x256_k8192_f16");
    // matmul2d has no bias slot -> a row-broadcast bias_add pass folds it. Load
    // it even for a dense checkpoint (the quant block above only loads it when
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
    if (const char* e = std::getenv("VPIPE_KREA2_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    // Split-K deep-reduction path for the very deep K (ff-down): opt-out A/B.
    m->_use_splitk = m->_use_mma2 && m->_fn_dense_mma_splitk.valid()
                     && m->_fn_residual.valid()
                     && std::getenv("VPIPE_KREA2_NO_SPLITK") == nullptr;
  }
  // Dynamic-int8 accelerated GEMMs (opt-in, LOSSY): tried first in
  // gemm_mma_ for the big block matmuls. Config::i8_gemm is the stage
  // switch; VPIPE_I8_GEMM overrides (the context self-gates on matrix
  // cores + kernel availability). The DiT runs bf16, so the i8 path reads/
  // writes bf16 (its activation, dequant-scratch weight, and output); load
  // the _bf16 i8 kernels (the f16 ones would misread bf16 buffers -> NaN).
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm, /*bf16=*/true);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }

  // Fused SwiGLU FF: interleave each MAIN block's quantized ff gate/up at load.
  // Steel path (M4): a real win -- one affine_qmm_swiglu GEMM whose register-
  // local epilogue writes silu(gate)*up directly, killing the two GEMMs' `u`
  // intermediate AND the separate swiglu pass.
  // M5 matmul2d path: measured NEUTRAL -- matmul2d can't fold the epilogue, so
  // it dequants the interleaved weight (2*ffn*K), runs one matmul2d, then a
  // swiglu_interleaved fold: same FLOPs/traffic as the two-GEMM+swiglu path
  // (both already NAX/tn2-accelerated) but +~470MB at 1024px (the interleaved
  // _w_deq doubles + the [seq, 2*ffn] gu intermediate). So on the mma path it
  // defaults OFF (unfused = marginally faster on ff-up + less memory);
  // VPIPE_KREA2_FUSED_FF=1 forces it on there (verified rel-L2 4.9e-4 vs steel).
  // Steel keeps it on. Skipped for dense/g32 (kernels are g64-only). Decided
  // BEFORE the blocks load (load_block_ interleaves when set).
  // VPIPE_KREA2_NO_FUSED_FF opts out entirely (A/B).
  if (m->_quant_bits > 0 && m->_quant_group == 64 &&
      std::getenv("VPIPE_KREA2_NO_FUSED_FF") == nullptr &&
      (!m->_use_mma2 || std::getenv("VPIPE_KREA2_FUSED_FF") != nullptr)) {
    m->_fn_qmm_swiglu4 = m->_lib_qmm.function("affine_qmm_swiglu_w4g64");
    m->_fn_qmm_swiglu8 = m->_lib_qmm.function("affine_qmm_swiglu_w8g64");
    m->_fuse_ff = m->_fn_qmm_swiglu4.valid() && m->_fn_qmm_swiglu8.valid();
    // AN ADAPTED FF CANNOT BE FUSED. The fused kernel's epilogue writes
    // silu(gate)*up straight out of the accumulator, so there is nowhere
    // for a pre-activation delta to land -- the adapter would simply not
    // apply, silently, to the two biggest projections in the block.
    // Asked from the FILE HEADER because the answer decides how the
    // blocks are BUILT, which happens before there is a model to bind an
    // adapter to.
    // The needles are the MODEL's names; the rename is passed so an
    // ai-toolkit file -- which spells them `mlp.gate` / `mlp.up` -- is
    // caught too. Without it such an adapter would bind through the
    // rename and then meet a fused kernel that silently drops its
    // delta on the two biggest projections in the block.
    // Every slot is asked: ONE adapter needing the split path settles it
    // for all of them, because the weight layout is the block's and not
    // the adapter's.
    for (const LoraSpec& sp : loras) {
      if (!m->_fuse_ff || sp.path.empty()) { continue; }
      if (lora::Adapter::file_touches(sp.path, ".ff.gate.lora_",
                                      lora_rename()) ||
          lora::Adapter::file_touches(sp.path, ".ff.up.lora_",
                                      lora_rename())) {
        m->_fuse_ff = false;
        if (mc->session() != nullptr) {
          mc->session()->log_normal(fmt(
              "MetalKrea2Transformer: an adapter touches ff.gate/ff.up, so "
              "the fused-SwiGLU weave is off for this load -- the split "
              "path is the only one a pre-activation delta can reach"));
        }
      }
    }
    if (m->_fuse_ff && m->_use_mma2) {
      m->_fn_swiglu_inter = m->_lib_elt.function("swiglu_interleaved_f16");
      if (!m->_fn_swiglu_inter.valid()) { m->_fuse_ff = false; }
    }
    // BM=128 fused-SwiGLU twins for the tall-M path (steel only; the FF gate+up
    // is the biggest DiT GEMM and had stayed at BM=32). Fall back to BM=64
    // (steel) qmm elsewhere; if either is missing, drop _qmm_tile off BM128.
    if (m->_fuse_ff && m->_qmm_tile == 2) {
      m->_fn_qmm_swiglu4_bm128 =
          m->_lib_qmm.function("affine_qmm_swiglu_w4g64_bm128");
      m->_fn_qmm_swiglu8_bm128 =
          m->_lib_qmm.function("affine_qmm_swiglu_w8g64_bm128");
      if (!m->_fn_qmm_swiglu4_bm128.valid() ||
          !m->_fn_qmm_swiglu8_bm128.valid()) {
        m->_qmm_tile = 1;
      }
    }
  }

  // FP16-pipe half-accumulate GEMM twins (_acc16): keep the simdgroup MMA on
  // half8x8 instead of widening to f32 -- targets the doubled f16 ALU rate.
  // Applied to the tall-M block GEMMs (bm64 + fused swiglu + dense bm64);
  // small-M/conditioning GEMMs stay f32-accumulate. VPIPE_KREA2_ACC16=1
  // opts in (accuracy bar: the DiT goldens; f16 accumulation drifts over
  // K=6144..16384).
  if (const char* e = std::getenv("VPIPE_KREA2_ACC16")) {
    m->_acc16 = std::atoi(e) != 0;
  }
  if (m->_acc16) {
    m->_fn_gemm_bm64_a16 =
        m->_lib_gemm.function("dense_gemm_t_bm64_acc16_f16");
    bool ok = m->_fn_gemm_bm64_a16.valid();
    if (m->_quant_bits > 0 && m->_quant_group == 64) {
      m->_fn_qmm4_bm64_a16 =
          m->_lib_qmm.function("affine_qmm_steel_w4g64_bm64_acc16");
      m->_fn_qmm8_bm64_a16 =
          m->_lib_qmm.function("affine_qmm_steel_w8g64_bm64_acc16");
      ok = ok && m->_fn_qmm4_bm64_a16.valid() &&
           m->_fn_qmm8_bm64_a16.valid();
      if (m->_fuse_ff) {
        m->_fn_qmm_swiglu4_a16 =
            m->_lib_qmm.function("affine_qmm_swiglu_w4g64_acc16");
        m->_fn_qmm_swiglu8_a16 =
            m->_lib_qmm.function("affine_qmm_swiglu_w8g64_acc16");
        ok = ok && m->_fn_qmm_swiglu4_a16.valid() &&
             m->_fn_qmm_swiglu8_a16.valid();
      }
    }
    m->_acc16 = ok;
  }

  m->_layerwise.resize((std::size_t)cfg.n_layerwise);
  for (int i = 0; i < cfg.n_layerwise; ++i) {
    if (!m->load_block_(ws,
            "text_fusion.layerwise_blocks." + std::to_string(i) + ".",
            m->_layerwise[(std::size_t)i], false, r)) {
      return nullptr;
    }
  }
  m->_refiner.resize((std::size_t)cfg.n_refiner);
  for (int i = 0; i < cfg.n_refiner; ++i) {
    if (!m->load_block_(ws,
            "text_fusion.refiner_blocks." + std::to_string(i) + ".",
            m->_refiner[(std::size_t)i], false, r)) {
      return nullptr;
    }
  }
  m->_projector = m->load_qw_(ws, "text_fusion.projector", r);
  m->_txt_norm  = m->elt_(ws, "txt_in.norm.weight", r, true);
  m->_txt_l1w   = m->load_qw_(ws, "txt_in.linear_1", r);
  m->_txt_l1b   = m->elt_(ws, "txt_in.linear_1.bias", r);
  m->_txt_l2w   = m->load_qw_(ws, "txt_in.linear_2", r);
  m->_txt_l2b   = m->elt_(ws, "txt_in.linear_2.bias", r);

  // DiT image path.
  m->_img_in_w = m->load_qw_(ws, "img_in", r);
  m->_img_in_b = m->elt_(ws, "img_in.bias", r);
  m->_te_l1w = m->load_qw_(ws, "time_embed.linear_1", r);
  m->_te_l1b = m->elt_(ws, "time_embed.linear_1.bias", r);
  m->_te_l2w = m->load_qw_(ws, "time_embed.linear_2", r);
  m->_te_l2b = m->elt_(ws, "time_embed.linear_2.bias", r);
  m->_tmp_w  = m->load_qw_(ws, "time_mod_proj", r);
  m->_tmp_b  = m->elt_(ws, "time_mod_proj.bias", r);
  // Main blocks: preload all 28 (default), or stream them on demand from the
  // retained mmap (memory-bounded mode; loaded per-block in forward_dit). In
  // Streaming mode preloads NOTHING: the resident set is grown by
  // BlockResidency as free memory allows, which is what replaced the
  // pinned prefix. Loaded from `wts` here (read from the weight set).
  if (!stream_blocks) {
    m->_blocks.resize((std::size_t)cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
      if (!m->load_block_(ws,
              "transformer_blocks." + std::to_string(i) + ".",
              m->_blocks[(std::size_t)i], true, r)) {
        return nullptr;
      }
    }
  } else {
    // Sized to the FULL depth though nothing is preloaded: the empty
    // slots are where forward() promotes streamed blocks as free memory
    // allows (set_residency_reserve). An unfilled slot reads as empty,
    // which is what the `held` test below keys on.
    m->_blocks.resize((std::size_t)cfg.n_layers);
  }
  m->_final_sst  = m->elt_(ws, "final_layer.scale_shift_table", r);
  m->_final_norm = m->elt_(ws, "final_layer.norm.weight", r, true);
  m->_final_lw   = m->load_qw_(ws, "final_layer.linear", r);
  m->_final_lb   = m->elt_(ws, "final_layer.linear.bias", r);

  if (m->_projector.empty() || m->_txt_norm.empty() || m->_txt_l1w.empty() ||
      m->_txt_l1b.empty() || m->_txt_l2w.empty() || m->_txt_l2b.empty() ||
      m->_img_in_w.empty() || m->_img_in_b.empty() || m->_te_l1w.empty() ||
      m->_te_l1b.empty() || m->_te_l2w.empty() || m->_te_l2b.empty() ||
      m->_tmp_w.empty() || m->_tmp_b.empty() || m->_final_sst.empty() ||
      m->_final_norm.empty() || m->_final_lw.empty() || m->_final_lb.empty()) {
    return nullptr;
  }
  // Retain the source mmap: streaming mode loads main blocks from it on demand,
  // and zero-copy mode keeps it alive because the quantized weights are views
  // into it (moving the weights leaves the mmap base + the wrapped whole-shard
  // buffers intact; the subviews handed out keep their own refcount).
  if (stream_blocks || m->_mmap_weights) {
    if (m->_mmap_weights && mc->session() != nullptr) {
      mc->session()->log_debug(fmt(
          "MetalKrea2Transformer: zero-copy (mmap) quantized weights enabled"));
    }
  }
  // The adapter LAST: every projection it names has to exist, and the
  // fused-FF decision above has already been taken with it in view.
  // Slots bind in the order given and keep it: `lora_scale(0)` is the
  // first adapter the caller named, whatever it turned out to be.
  for (const LoraSpec& sp : loras) {
    if (sp.path.empty()) { continue; }
    if ((int)m->_lora_slots.size() >= kMaxLoraSlots) {
      if (mc->session() != nullptr) {
        mc->session()->warn(fmt(
            "MetalKrea2Transformer: {} adapters given, {} slots -- '{}' is "
            "IGNORED", loras.size(), kMaxLoraSlots, sp.path));
      }
      break;
    }
    if (!m->_lora.init(mc, m->_use_mma2)) {
      if (mc->session() != nullptr) {
        mc->session()->warn(fmt(
            "MetalKrea2Transformer: the LoRA GEMM kernels did not load; "
            "the adapter is NOT applied"));
      }
      break;
    }
    LoraSlot slot;
    std::string lerr;
    if (!m->bind_lora_(sp, slot, &lerr)) {
      if (mc->session() != nullptr) { mc->session()->warn(fmt("{}", lerr)); }
      continue;
    }
    // Each slot owns its own rows of the [rows, rank] scratch, so two
    // adapters' first GEMMs are not ordered against each other's second.
    slot.rank_prefix = m->_lora_rank_total;
    m->_lora_rank_total += slot.max_rank;
    m->_lora_slots.push_back(std::move(slot));
  }
  return m;
}

void
MetalKrea2Transformer::build_rope_tables_(int text_seq,
                                          const std::vector<ImgSeg>& segs,
                                          SharedBuffer& cos_out,
                                          SharedBuffer& sin_out)
{
  int img_seq = 0;
  for (const auto& sg : segs) { img_seq += sg.seq; }
  const int seq = text_seq + img_seq;
  const int D = _cfg.head_dim;                 // 128
  const int axes[3] = {_cfg.rope_t, _cfg.rope_h, _cfg.rope_w};   // 32,48,48
  const double theta = (double)_cfg.rope_theta;
  // f32 cos/sin tables (only x is bf16), via rope_pair_table_ftab_f16: RoPE
  // rotation error is STRUCTURED, so the ~4e-3 bf16-table rounding compounds
  // over the DiT's blocks (and denoise steps) rather than averaging out -- the
  // same reason the shared Qwen-Image stack (QIE) keeps these tables f32.
  cos_out = _mc->make_shared_buffer((std::size_t)seq * D * sizeof(float));
  sin_out = _mc->make_shared_buffer((std::size_t)seq * D * sizeof(float));
  auto* cb = static_cast<float*>(cos_out.contents());
  auto* sb = static_cast<float*>(sin_out.contents());
  // Per-pair inverse frequency + coordinate axis + element offset within the D
  // row -- depend only on (axis, i), NOT the token, so precompute once (pairs
  // pow() calls) instead of recomputing inside the per-token emit (seq*pairs).
  const int pairs = D / 2;
  std::vector<double> pair_freq((std::size_t)pairs);
  std::vector<int> pair_axis((std::size_t)pairs), pair_off((std::size_t)pairs);
  {
    int base = 0, pair = 0;
    for (int a = 0; a < 3; ++a) {
      const int Dax = axes[a];
      for (int i = 0; i < Dax / 2; ++i, ++pair) {
        pair_freq[(std::size_t)pair] =
            1.0 / std::pow(theta, (2.0 * i) / (double)Dax);
        pair_axis[(std::size_t)pair] = a;
        pair_off[(std::size_t)pair] = base + 2 * i;
      }
      base += Dax;
    }
  }
  // Emit one token row (s) from its 3 coordinates (frame/t, h, w).
  auto emit = [&](int s, double c0, double c1, double c2) {
    const double coord[3] = {c0, c1, c2};
    for (int pair = 0; pair < pairs; ++pair) {
      const double ang =
          coord[pair_axis[(std::size_t)pair]] * pair_freq[(std::size_t)pair];
      const float c = (float)std::cos(ang);
      const float sn = (float)std::sin(ang);
      const std::size_t o = (std::size_t)s * D + pair_off[(std::size_t)pair];
      cb[o] = c; cb[o + 1] = c;
      sb[o] = sn; sb[o + 1] = sn;
    }
  };
  // Text tokens sit at the origin (0,0,0).
  int s = 0;
  for (; s < text_seq; ++s) { emit(s, 0.0, 0.0, 0.0); }
  // Image segments: generated (frame 0) then each reference (frame = index),
  // each token (frame, row, col) over its grid.
  for (const auto& sg : segs) {
    for (int j = 0; j < sg.seq; ++j, ++s) {
      emit(s, (double)sg.frame, (double)(j / sg.grid_w),
           (double)(j % sg.grid_w));
    }
  }
}

void
MetalKrea2Transformer::ensure_dit_scratch_(int text_seq, int grid_h, int grid_w,
                                           const std::vector<ImgSeg>& segs)
{
  const int TS = text_seq;
  int IS = 0;
  for (const auto& sg : segs) { IS += sg.seq; }   // total image tokens
  const int seq = TS + IS;
  // Layout signature: captures the reference segmentation so a changed set of
  // references (same total seq but a different split) rebuilds the RoPE.
  std::size_t sig = (std::size_t)text_seq * 1000003u + segs.size();
  for (const auto& sg : segs) {
    sig = sig * 1000003u + (std::size_t)(sg.frame + 1) * 131 +
          (std::size_t)sg.grid_h * 17 + (std::size_t)sg.grid_w * 7 + sg.seq;
  }
  if (_dit.seq == seq && _dit.gh == grid_h && _dit.gw == grid_w &&
      _dit.sig == sig) {
    return;
  }
  // UNWIRE THE OLD SCRATCH FIRST, before the assignments below drop it.
  // Destroying a wired buffer unwires it in the kernel but does NOT
  // decrement the pool's counter -- only unwire_from_pool does -- so a
  // geometry change would leave the pool believing those bytes were still
  // held. Every change would leak a scratch's worth, and a pool that has
  // lost budget to bytes nothing holds wires less of what comes next: the
  // resident set shrinks for a reason nothing in the log names.
  if (_wire.on()) {
    for (metal_compute::SharedBuffer* b : scratch_buffers_()) {
      _wire.wire_one(_mc, *b, false);
    }
  }
  const Config& c = _cfg;
  const int HID = c.hidden, FF = c.ffn, TD = c.timestep_dim;
  const int qd = c.n_heads * c.head_dim, kd = c.n_kv_heads * c.head_dim;
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  _dit.te_in = buf((std::size_t)TD);
  _dit.joint = buf((std::size_t)seq * HID);
  _dit.te1   = buf((std::size_t)HID);
  _dit.temb  = buf((std::size_t)HID);
  _dit.tmp   = buf((std::size_t)HID);
  _dit.tmod  = buf((std::size_t)6 * HID);
  _dit.mod   = buf((std::size_t)6 * HID);
  _dit.n1    = buf((std::size_t)seq * HID);
  _dit.n2    = buf((std::size_t)seq * HID);
  _dit.nm    = buf((std::size_t)seq * HID);
  _dit.gate  = buf((std::size_t)seq * HID);
  _dit.att   = buf((std::size_t)seq * HID);
  _dit.o     = buf((std::size_t)seq * HID);
  _dit.q     = buf((std::size_t)seq * qd);
  _dit.k     = buf((std::size_t)seq * kd);
  _dit.v     = buf((std::size_t)seq * kd);
  _dit.qt    = buf((std::size_t)seq * qd);
  _dit.kt    = buf((std::size_t)seq * kd);
  _dit.vt    = buf((std::size_t)seq * kd);
  _dit.atb   = buf((std::size_t)seq * qd);
  _dit.g     = buf((std::size_t)seq * FF);
  // `u` (the split-path up-projection intermediate) is dead when every main
  // block runs the fused gate|up GEMM; streaming mode loads blocks later, so
  // it stays conservative and keeps the buffer.
  bool need_u = _stream_blocks || _blocks.empty();
  for (const Block& b : _blocks) {
    if (b.ff_gu.empty()) { need_u = true; break; }
  }
  _dit.u = need_u ? buf((std::size_t)seq * FF) : SharedBuffer{};
  // `gu` (M5 mma fused path): the interleaved gate|up matmul2d output
  // [seq, 2*FF] before the swiglu_interleaved fold. Only when the mma path
  // may run over a fused weight.
  bool need_gu = false;
  if (_use_mma2 && _fuse_ff) {
    need_gu = _stream_blocks || _blocks.empty();
    for (const Block& b : _blocks) {
      if (!b.ff_gu.empty()) { need_gu = true; break; }
    }
  }
  _dit.gu = need_gu ? buf((std::size_t)seq * 2 * FF) : SharedBuffer{};
  _dit.modf  = buf((std::size_t)2 * HID);
  build_rope_tables_(TS, segs, _dit.rcos, _dit.rsin);
  _dit.seq = seq; _dit.gh = grid_h; _dit.gw = grid_w; _dit.sig = sig;
}

void
MetalKrea2Transformer::release_forward_scratch()
{
  // Reset the shape key to -1 so ensure_dit_scratch_ rebuilds every buffer on
  // the next forward; dropping the struct frees the held activation scratch.
  // Unwired first, for the reason in ensure_dit_scratch_: the kernel
  // recovers on free, the pool's counter does not.
  if (_wire.on()) {
    for (metal_compute::SharedBuffer* b : scratch_buffers_()) {
      _wire.wire_one(_mc, *b, false);
    }
  }
  _dit = DitScratch{};
  _w_deq = {};                         // dequant scratch (regrows in gemm_mma_)
  _splitk = {};                        // split-K partial planes
  if (_i8) { _i8->release_scratch(); }  // i8 act/weight requant scratch
}

void
MetalKrea2Transformer::calib_begin()
{
  const int nL = _cfg.n_layers, HID = _cfg.hidden, FF = _cfg.ffn;
  auto zeros = [&](std::vector<SharedBuffer>& v, int dim) {
    v.clear();
    v.resize((std::size_t)nL);
    for (int L = 0; L < nL; ++L) {
      v[(std::size_t)L] = _mc->make_shared_buffer((std::size_t)dim * 2);
      std::memset(v[(std::size_t)L].contents(), 0, (std::size_t)dim * 2);
    }
  };
  zeros(_cb_qkv, HID);
  zeros(_cb_o, HID);
  zeros(_cb_gu, HID);
  zeros(_cb_dn, FF);
  _calib_on = true;
}

std::vector<std::vector<float>>
MetalKrea2Transformer::read_calib_(const std::vector<SharedBuffer>& acc,
                                   int dim) const
{
  std::vector<std::vector<float>> out(acc.size());
  for (std::size_t L = 0; L < acc.size(); ++L) {
    out[L].resize((std::size_t)dim);
    // The col_absmax (AWQ) accumulator is bf16 now that the DiT runs bf16.
    const auto* s = static_cast<const std::uint16_t*>(acc[L].contents());
    for (int d = 0; d < dim; ++d) {
      out[L][(std::size_t)d] = bf16_to_f32_(s[d]);
    }
  }
  return out;
}

std::vector<std::vector<float>>
MetalKrea2Transformer::calib_qkv() const { return read_calib_(_cb_qkv, _cfg.hidden); }
std::vector<std::vector<float>>
MetalKrea2Transformer::calib_o() const { return read_calib_(_cb_o, _cfg.hidden); }
std::vector<std::vector<float>>
MetalKrea2Transformer::calib_gateup() const { return read_calib_(_cb_gu, _cfg.hidden); }
std::vector<std::vector<float>>
MetalKrea2Transformer::calib_down() const { return read_calib_(_cb_dn, _cfg.ffn); }

std::size_t
MetalKrea2Transformer::scratch_resident_bytes() const
{
  const metal_compute::SharedBuffer* all[] = {
      &_dit.te_in, &_dit.rcos, &_dit.rsin, &_dit.joint, &_dit.te1,
      &_dit.temb, &_dit.tmp, &_dit.tmod, &_dit.mod, &_dit.n1, &_dit.n2,
      &_dit.nm, &_dit.gate, &_dit.att, &_dit.o, &_dit.q, &_dit.k, &_dit.v,
      &_dit.qt, &_dit.kt, &_dit.vt, &_dit.atb, &_dit.g, &_dit.u, &_dit.gu,
      &_dit.modf, &_w_deq, &_splitk};
  std::size_t n = 0;
  for (const metal_compute::SharedBuffer* p : all) { n += p->byte_size(); }
  return n;
}

// Bytes one resident block holds. Summed from the buffers rather than
// derived from the config, for the same reason scratch is: a derived
// number drifts the moment a tensor is added.
std::size_t
MetalKrea2Transformer::qw_bytes_(const QWeight& w)
{
  return w.w.byte_size() + w.codes.byte_size() + w.scales.byte_size() +
         w.qbias.byte_size();
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
MetalKrea2Transformer::wire_block_(Block& b, bool on)
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
  QWeight* q[] = {&b.q, &b.k, &b.v, &b.gate, &b.o,
                  &b.ff_gate, &b.ff_up, &b.ff_down, &b.ff_gu};
  for (QWeight* w : q) { qw(*w); }
  one(b.n1); one(b.n2); one(b.qn); one(b.kn);
  return changed;
}

// Everything this model holds that is NOT a streamed block, in the order
// that matters: the persistent activation scratch first, then the TRUNK
// (everything the weight set cached for this model). Both are read on
// every block of every forward and neither is ever shed, so they have a
// better claim on the pool than any single resident block -- which is why
// the forward wires this BEFORE it starts admitting blocks. Protecting
// the optional half first is how a run ends up with wired blocks beside
// an activation buffer the compressor is free to take.
//
// Unlike the other image DiTs, this one's activations live in a
// persistent DitScratch, so they can be wired at all. Every buffer of it
// is replaced wholesale by ensure_dit_scratch_ and dropped by
// release_forward_scratch, and BOTH unwire first -- a buffer replaced
// while wired leaks its bytes from the pool's counter, since only
// unwire_from_pool() decrements it.
std::vector<metal_compute::SharedBuffer*>
MetalKrea2Transformer::scratch_buffers_()
{
  return {&_dit.te_in, &_dit.rcos, &_dit.rsin, &_dit.joint, &_dit.te1,
          &_dit.temb, &_dit.tmp, &_dit.tmod, &_dit.mod, &_dit.n1, &_dit.n2,
          &_dit.nm, &_dit.gate, &_dit.att, &_dit.o, &_dit.q, &_dit.k,
          &_dit.v, &_dit.qt, &_dit.kt, &_dit.vt, &_dit.atb, &_dit.g,
          &_dit.u, &_dit.gu, &_dit.modf};
}

std::size_t
MetalKrea2Transformer::wire_fixed_(bool on)
{
  std::size_t changed = 0;
  for (metal_compute::SharedBuffer* b : scratch_buffers_()) {
    changed += _wire.wire_one(_mc, *b, on);
  }
  if (_ws) {
    _ws->for_each_weight([&](metal_compute::SharedBuffer& b) {
      changed += _wire.wire_one(_mc, b, on);
    });
  }
  return changed;
}

std::size_t
MetalKrea2Transformer::block_bytes_(const Block& b)
{
  return b.n1.byte_size() + b.n2.byte_size() + b.qn.byte_size() +
         b.kn.byte_size() + qw_bytes_(b.q) + qw_bytes_(b.k) +
         qw_bytes_(b.v) + qw_bytes_(b.gate) + qw_bytes_(b.o) +
         qw_bytes_(b.ff_gate) + qw_bytes_(b.ff_up) + qw_bytes_(b.ff_down) +
         qw_bytes_(b.ff_gu);
}

std::size_t
MetalKrea2Transformer::resident_bytes_(const Block& b)
{
  std::size_t n = block_bytes_(b);
  if (!b.ff_gu.empty()) { n -= qw_bytes_(b.ff_gate) + qw_bytes_(b.ff_up); }
  return n;
}

std::size_t
MetalKrea2Transformer::read_bytes_(const Block& b)
{
  std::size_t n = block_bytes_(b);
  if (!b.ff_gate.empty()) { n -= qw_bytes_(b.ff_gu); }
  return n;
}

// How much of the RESIDENT set is still in RAM. Sampled inside each
// buffer: a pin either holds or it does not, so every 64th page finds
// it. See BlockResidency property 5 -- free-memory arithmetic cannot
// answer this, and this is the question that actually matters.
void
MetalKrea2Transformer::resident_pages_(std::size_t* examined,
                                       std::size_t* incore) const
{
  *examined = 0;
  *incore = 0;
  for (const Block& b : _blocks) {
    const metal_compute::SharedBuffer* all[] = {
        &b.q.w, &b.q.codes, &b.k.w, &b.k.codes, &b.v.w, &b.v.codes,
        &b.o.w, &b.o.codes, &b.ff_gate.w, &b.ff_gate.codes,
        &b.ff_up.w, &b.ff_up.codes, &b.ff_down.w, &b.ff_down.codes,
        &b.ff_gu.w, &b.ff_gu.codes};
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
    }
  }
}

// From the TAIL down so what remains stays a contiguous prefix; in a
// cyclic scan every block is worth the same. The pinned prefix is never
// touched.
std::size_t
MetalKrea2Transformer::evict_tail_block_()
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
    // Taking one out of the PINNED prefix un-pins it: that prefix was
    // sized at load against what the box was believed to hold, and a
    // measurement saying its pages are no longer in RAM is that belief
    // being wrong. The forward decides resident-or-streamed by whether
    // the slot is EMPTY, not by this count, so it simply streams now.
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
MetalKrea2Transformer::wire_retry_slack_() const
{
  if (_wire_block_hint > 0) { return _wire_block_hint; }
  if (_resid.count() > 0) {
    return _resid.bytes() / (std::size_t)_resid.count();
  }
  return 0;
}

void
MetalKrea2Transformer::set_residency_schedule(int steps)
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
        "MetalKrea2Transformer: residency probe {} blocks of {} "
        "({} MB each, {} MB reclaimable, wire budget {} MB){}",
        _resid.per_forward_cap(), nl, blk >> 20,
        _mc->memory_budget().available_physical >> 20,
        _wire.budget() >> 20,
        _wire.on() ? " -- uncapped, the wire budget is the gate"
                   : ", doubling per healthy forward"));
  }
}

std::size_t
MetalKrea2Transformer::release_resident_blocks(std::size_t bytes)
{
  const std::size_t freed =
      _resid.release(bytes, [this]() -> std::size_t {
        return evict_tail_block_();
      });
  if (freed > 0 && _mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalKrea2Transformer: released {} MB of resident blocks ({} left)",
        freed >> 20, _resid.count()));
  }
  return freed;
}

bool
MetalKrea2Transformer::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& xin,
                                 const QWeight& w, const SharedBuffer& y,
                                 std::size_t ye, int M, int N, int K)
{
  // Matrix-core matmul2d only when present, M amortizes the 128-row tile, and N
  // is non-degenerate (a 1-wide projector stays on steel). Otherwise the caller
  // keeps its steel path.
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
    // codes/scales/qbias -> _w_deq[N,K] (one thread per packed u32 word: w4 has
    // 8 nibbles/word so K/8 words, w8 has 4 bytes/word so K/4). Serial ordering
    // + Metal's WAR hazard tracking make the shared _w_deq safe to reuse across
    // the block's GEMMs (each dequant->matmul pair runs before the next writes).
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
  // non-qualifying shapes.
  if (_i8 && _i8->gemm(enc, xin, 0, *wdense, y, ye, M, N, K)) {
    return true;
  }
  // Split-K deep-reduction path (ff-down, K=16384): the single-op full reduction
  // runs ~0.7x the K<=9728 rate. Fire when K is >= 2 chunks of kSplitKC and an
  // exact multiple (so the static-KC kernel bounds each split cleanly). Each of
  // the S = K/kSplitKC splits gets its own threadgroup plane (grid.z); a
  // residual_add fold sums the planes into y. Extra f16 rounding per fold is
  // fine for the rel-L2-verified DiT. Falls through to the single-op path when
  // the scratch alloc fails or the shape does not split.
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
      enc.set_buffer(0, xin); enc.set_buffer(1, *wdense);
      enc.set_buffer(2, _splitk);
      enc.set_constant(3, K); enc.set_constant(4, N); enc.set_constant(5, M);
      enc.dispatch({(unsigned)(((N + 255) / 256) * 256),   // BM=128, BN=256
                    (unsigned)((M + 127) / 128), (unsigned)splits},
                   {256, 1, 1});
      // Fold the S partial planes: y = plane0 + plane1 (+ plane_s ...). Metal
      // hazard tracking serializes each fold after the split matmul (and the
      // in-place y += plane_s accumulation) via the shared-buffer deps.
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
  // M/N tails. Tuned on the Krea2 shapes (M~=seq): 128x128 for K < 6144; the
  // TN=2 (128x512-region) tile for the K=6144 sweet spot (all the block
  // projections) where it beats plain 128x256 ~1.14-1.19x; plain 128x256 for
  // deeper unsplit K (>=12288), where the TN=2 tile collapses (that regime is
  // normally split-K; this is the NO_SPLITK fallback).
  int RN = 256;   // effective N-region per tg (TN*BN); grid divides N by it
  const metal_compute::ComputeFunction* fn = &_fn_dense_mma_deep;
  if (K < 6144) {
    fn = &_fn_dense_mma; RN = 128;
  } else if (K < 12288 && _fn_dense_mma_tn2.valid()) {
    fn = &_fn_dense_mma_tn2; RN = 512;   // TN=2: two 256-wide N-tiles per tg
  }
  enc.set_function(*fn);
  enc.set_buffer(0, xin); enc.set_buffer(1, *wdense); enc.set_buffer(2, *wdense);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  return true;
}

SharedBuffer
MetalKrea2Transformer::forward_text(const SharedBuffer& ehs, int text_seq)
{
  const Config& c = _cfg;
  const int TS = text_seq, NL = c.n_text_layers, TH = c.text_hidden;
  const int THD = c.text_head_dim();          // 128
  const int HED = c.text_heads;               // 20
  const int TFF = c.text_ffn;                 // 6912
  const int HID = c.hidden;                   // 6144
  const float eps = c.norm_eps;
  if (TS <= 0 || ehs.byte_size() < (std::size_t)TS * NL * TH * 2) { return {}; }

  // LLM-lane event: the text-fusion tower runs once per image. value = seq.
  PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDitText,
                     kPerfLlmDitTextBegin, (std::uint64_t)TS);

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  const int BTmax = TS * NL;                  // layerwise batch*T (>= refiner)
  const std::size_t HMmax = (std::size_t)TS * HED * NL * THD;   // head-major max

  // Residual + scratch (all live until stream.commit().wait()).
  SharedBuffer xlw = buf((std::size_t)TS * NL * TH);   // layerwise residual
  // Upcast the f16 encoder taps -> bf16 residual (the DiT computes in bf16; the
  // conditioner boundary stays f16).
  {
    const auto* s = static_cast<const _Float16*>(ehs.contents());
    auto* d = static_cast<std::uint16_t*>(xlw.contents());
    const std::size_t n = (std::size_t)TS * NL * TH;
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_((float)s[i]); }
  }
  SharedBuffer xt    = buf((std::size_t)TS * TH * NL); // projector transpose
  SharedBuffer fused = buf((std::size_t)TS * TH);      // projector out / refiner
  SharedBuffer sn = buf((std::size_t)BTmax * TH), sq = buf((std::size_t)BTmax * TH),
               sk = buf((std::size_t)BTmax * TH), sv = buf((std::size_t)BTmax * TH),
               sgate = buf((std::size_t)BTmax * TH);
  SharedBuffer qt = buf(HMmax), kt = buf(HMmax), vt = buf(HMmax),
               atb = buf(HMmax);
  SharedBuffer att = buf((std::size_t)BTmax * TH), o = buf((std::size_t)BTmax * TH);
  SharedBuffer h1 = buf((std::size_t)BTmax * TFF), h2 = buf((std::size_t)BTmax * TFF);
  SharedBuffer y2 = buf((std::size_t)TS * HID), y3 = buf((std::size_t)TS * HID);

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // y[M,N] = x[M,K] @ W[N,K]^T. dense_gemm_t (f16 W) or affine_qmm_steel
    // (quantized W). y written at element offset `ye`.
    auto gemm_off = [&](const SharedBuffer& xin, const QWeight& w,
                        const SharedBuffer& y, std::size_t ye, int M, int N,
                        int K) {
      if (gemm_mma_(enc, xin, w, y, ye, M, N, K)) { return; }
      if (w.quantized) {
        enc.set_function(w.bits == 8 ? _fn_qmm8 : _fn_qmm4);
        enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
        enc.set_buffer(2, w.qbias); enc.set_buffer(3, xin);
        enc.set_buffer(4, y, ye * 2);
        enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
      } else {
        enc.set_function(_fn_gemm);
        enc.set_buffer(0, xin); enc.set_buffer(1, w.w); enc.set_buffer(2, w.w);
        enc.set_buffer(3, y, ye * 2);
        enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
        enc.set_constant(7, 0);
      }
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
    };
    // `lf`, when given, is the runtime adapter for THIS projection.
    // Threaded through the lambda for the same reason the DiT's is: so
    // it cannot be forgotten on one projection and applied on the rest.
    auto gemm = [&](const SharedBuffer& xin, const QWeight& w,
                    const SharedBuffer& y, int M, int N, int K,
                    const lora::Stack& lf = lora::Stack{}) {
      gemm_off(xin, w, y, 0, M, N, K);
      _lora.apply(enc, xin, 0, lf, y, 0, M, N, K, _mma_min_m);
    };
    auto bias_add = [&](const SharedBuffer& bs, const SharedBuffer& y, int M,
                        int N) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y); enc.set_buffer(1, bs);
      enc.set_constant(2, N); enc.set_constant(3, M * N);
      enc.dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
    };
    auto gemm_bias = [&](const SharedBuffer& xin, const QWeight& w,
                         const SharedBuffer& bs, const SharedBuffer& y, int M,
                         int N, int K) {
      if (gemm_mma_(enc, xin, w, y, 0, M, N, K)) {   // dense OR quant on M5
        bias_add(bs, y, M, N);
        return;
      }
      if (w.quantized) {
        gemm_off(xin, w, y, 0, M, N, K);
        bias_add(bs, y, M, N);
        return;
      }
      enc.set_function(_fn_gemm_bias);
      enc.set_buffer(0, xin); enc.set_buffer(1, w.w); enc.set_buffer(2, bs);
      enc.set_buffer(3, y);
      enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
      enc.set_constant(7, 1);
      enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                    (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
    };
    auto rms = [&](const SharedBuffer& x, std::size_t xe, const SharedBuffer& w,
                   const SharedBuffer& y, std::size_t ye, int R, int Hd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w);
      enc.set_buffer(2, y, ye * 2);
      enc.set_constant(3, Hd); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, std::size_t ie,
                         const SharedBuffer& out, std::size_t oe, int A, int Bd,
                         int D) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in, ie * 2); enc.set_buffer(1, out, oe * 2);
      enc.set_constant(2, A); enc.set_constant(3, Bd); enc.set_constant(4, D);
      enc.dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A},
                   {(unsigned)D, 1, 1});
    };
    auto sdpa = [&](const SharedBuffer& q, const SharedBuffer& k,
                    const SharedBuffer& v, const SharedBuffer& out, float scale,
                    int T, int D, int Hq, int Hkv, int nq, int kvstride) {
      enc.set_function(_fn_sdpa);
      enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
      enc.set_buffer(3, out);
      enc.set_constant(4, scale); enc.set_constant(5, T); enc.set_constant(6, D);
      enc.set_constant(7, Hq); enc.set_constant(8, Hkv); enc.set_constant(9, nq);
      enc.set_constant(10, kvstride);
      enc.dispatch({32, (unsigned)Hq, (unsigned)nq}, {32, 1, 1});
    };
    auto elt3 = [&](const metal_compute::ComputeFunction& fn,
                    const SharedBuffer& a, const SharedBuffer& b2,
                    const SharedBuffer& out, int nn) {
      enc.set_function(fn);
      enc.set_buffer(0, a); enc.set_buffer(1, b2); enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto gelu = [&](const SharedBuffer& x, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_gelu_tanh);
      enc.set_buffer(0, x); enc.set_buffer(1, y);
      enc.set_constant(2, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };

    // Krea2 attention+SwiGLU block over `batch` independent length-T sequences
    // (dim TH, heads HED, ff TFF, no rope). x [batch*T, TH], updated in place.
    auto block = [&](const Block& b, const SharedBuffer& x, int batch, int T,
                     LoraStack which, int li) {
      const int BT = batch * T, BH = batch * HED;
      auto la = [&](lora::Factors BlockLora::* d) {
        return lora_at_(which, li, d);
      };
      const float scale = 1.0f / std::sqrt((float)THD);
      rms(x, 0, b.n1, sn, 0, BT, TH);
      gemm(sn, b.q, sq, BT, TH, TH, la(&BlockLora::q));
      gemm(sn, b.k, sk, BT, TH, TH, la(&BlockLora::k));
      gemm(sn, b.v, sv, BT, TH, TH, la(&BlockLora::v));
      gemm(sn, b.gate, sgate, BT, TH, TH, la(&BlockLora::gate));
      rms(sq, 0, b.qn, sq, 0, BT * HED, THD);   // per-head q-norm
      rms(sk, 0, b.kn, sk, 0, BT * HED, THD);   // per-head k-norm
      for (int bi = 0; bi < batch; ++bi) {      // -> head-major [BH, T, THD]
        const std::size_t si = (std::size_t)bi * T * TH;
        const std::size_t di = (std::size_t)bi * HED * T * THD;
        transpose(sq, si, qt, di, T, HED, THD);
        transpose(sk, si, kt, di, T, HED, THD);
        transpose(sv, si, vt, di, T, HED, THD);
      }
      sdpa(qt, kt, vt, atb, scale, T, THD, BH, BH, T, T);
      for (int bi = 0; bi < batch; ++bi) {      // -> token-major [batch*T, TH]
        transpose(atb, (std::size_t)bi * HED * T * THD, att,
                  (std::size_t)bi * T * TH, HED, T, THD);
      }
      elt3(_fn_mul_sigmoid, att, sgate, att, BT * TH);   // att *= sigmoid(gate)
      gemm(att, b.o, o, BT, TH, TH, la(&BlockLora::o));
      elt3(_fn_residual, x, o, x, BT * TH);
      rms(x, 0, b.n2, sn, 0, BT, TH);
      gemm(sn, b.ff_gate, h1, BT, TFF, TH, la(&BlockLora::ff_gate));
      gemm(sn, b.ff_up, h2, BT, TFF, TH, la(&BlockLora::ff_up));
      elt3(_fn_swiglu, h1, h2, h1, BT * TFF);            // silu(gate)*up
      gemm(h1, b.ff_down, o, BT, TH, TFF, la(&BlockLora::ff_down));
      elt3(_fn_residual, x, o, x, BT * TH);
    };

    // Text-fusion: layerwise (over the 12-layer axis, per token) -> projector
    // (Linear 12->1) -> refiner (over the token axis).
    // The tower's widest projection is [TS*NL, rank] in the layerwise
    // stack. Sized before the first block for the same reason the DiT's
    // is: apply() returns early on an empty scratch, and an adapter
    // that does nothing looks exactly like one that had no effect.
    if (_lora_rank_total > 0) {
      // One region per SLOT, so two adapters do not serialise on the
      // same rows.
      _lora_scratch_rows = (std::size_t)TS * (std::size_t)NL;
      _lora.ensure_scratch(_lora_scratch_rows *
                           (std::size_t)_lora_rank_total);
    }
    for (int i = 0; i < (int)_layerwise.size(); ++i) {
      block(_layerwise[(std::size_t)i], xlw, TS, NL, LoraStack::kLayerwise,
            i);
    }
    for (int t = 0; t < TS; ++t) {              // (NL,TH) -> (TH,NL) per token
      transpose(xlw, (std::size_t)t * NL * TH, xt, (std::size_t)t * TH * NL,
                NL, TH, 1);
    }
    gemm(xt, _projector, fused, TS * TH, 1, NL);
    for (int i = 0; i < (int)_refiner.size(); ++i) {
      block(_refiner[(std::size_t)i], fused, 1, TS, LoraStack::kRefiner, i);
    }

    // txt_in: RMSNorm(+1) -> Linear+bias -> gelu-tanh -> Linear+bias.
    rms(fused, 0, _txt_norm, sn, 0, TS, TH);
    gemm_bias(sn, _txt_l1w, _txt_l1b, y2, TS, HID, TH);
    gelu(y2, y2, TS * HID);
    gemm_bias(y2, _txt_l2w, _txt_l2b, y3, TS, HID, HID);
  }
  stream.commit().wait();
  // Downcast the bf16 fused-text to the public f16 boundary (forward_dit upcasts
  // it back -- lossless: the txt_in output is f16-safe, the overflow that needs
  // bf16 is deeper in the forward_dit residual stream).
  SharedBuffer out = _mc->make_shared_buffer((std::size_t)TS * HID * 2);
  {
    const auto* s = static_cast<const std::uint16_t*>(y3.contents());
    auto* d = static_cast<_Float16*>(out.contents());
    const std::size_t n = (std::size_t)TS * HID;
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)bf16_to_f32_(s[i]); }
  }
  return out;
}

SharedBuffer
MetalKrea2Transformer::forward_dit(const SharedBuffer& fused_text, int text_seq,
                                   const SharedBuffer& latents, int img_seq,
                                   int grid_h, int grid_w, float timestep,
                                   int stop_after_block,
                                   const std::vector<RefImage>& refs)
{
  const Config& c = _cfg;
  // The image token stream is [generated; ref0; ref1; ...]: IS_GEN generated
  // tokens (the only ones we predict velocity for) followed by the reference
  // tokens. IS is the full image-token count that drives every block; IS_GEN is
  // what the final layer emits.
  const int TS = text_seq, IS_GEN = img_seq;
  int IS_REF = 0;
  for (const auto& r : refs) { IS_REF += r.seq; }
  const int IS = IS_GEN + IS_REF, seq = TS + IS;
  const int HID = c.hidden, IC = c.in_channels, HED = c.n_heads;
  const int KVH = c.n_kv_heads, HD = c.head_dim, FF = c.ffn, TD = c.timestep_dim;
  const int qd = HED * HD, kd = KVH * HD;         // 6144, 1536
  const float eps = c.norm_eps;
  if (TS <= 0 || IS_GEN <= 0
      || fused_text.byte_size() < (std::size_t)TS * HID * 2
      || latents.byte_size() < (std::size_t)IS_GEN * IC * 2) {
    return {};
  }
  for (const auto& r : refs) {
    if (r.seq <= 0 || r.latents.byte_size() < (std::size_t)r.seq * IC * 2) {
      return {};
    }
  }
  // Image layout for scratch + RoPE: generated (frame 0) then each reference
  // (frame = index). grid_h/grid_w anchor the generated grid + shape key.
  std::vector<ImgSeg> segs;
  segs.push_back({0, grid_h, grid_w, IS_GEN});
  for (int i = 0; i < (int)refs.size(); ++i) {
    segs.push_back({i + 1, refs[i].grid_h, refs[i].grid_w, refs[i].seq});
  }
  // LLM-lane event: one MMDiT forward per sampler step. value = joint seq.
  PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDit,
                     kPerfLlmDitBegin, (std::uint64_t)seq);
  // Persistent per-shape scratch + RoPE tables: built once and reused across
  // the denoising steps of one image (rope is timestep-independent; the rest is
  // overwritten each call). Callers copy the returned velocity before the next
  // call, so sharing `noise`/`joint` across calls is safe.
  ensure_dit_scratch_(TS, grid_h, grid_w, segs);
  SharedBuffer& te_in = _dit.te_in;
  SharedBuffer& rcos = _dit.rcos; SharedBuffer& rsin = _dit.rsin;
  SharedBuffer& joint = _dit.joint;
  SharedBuffer& te1 = _dit.te1; SharedBuffer& temb = _dit.temb;
  SharedBuffer& tmp = _dit.tmp; SharedBuffer& tmod = _dit.tmod;
  SharedBuffer& mod = _dit.mod;
  SharedBuffer& n1 = _dit.n1; SharedBuffer& n2 = _dit.n2;
  SharedBuffer& nm = _dit.nm; SharedBuffer& gate = _dit.gate;
  SharedBuffer& att = _dit.att; SharedBuffer& o = _dit.o;
  SharedBuffer& q = _dit.q; SharedBuffer& k = _dit.k; SharedBuffer& v = _dit.v;
  SharedBuffer& qt = _dit.qt; SharedBuffer& kt = _dit.kt;
  SharedBuffer& vt = _dit.vt; SharedBuffer& atb = _dit.atb;
  SharedBuffer& g = _dit.g; SharedBuffer& u = _dit.u;
  SharedBuffer& modf = _dit.modf;
  // The returned velocity is a fresh per-call buffer (moved out at return).
  // Only the generated tokens get a velocity; reference tokens are discarded.
  SharedBuffer noise = _mc->make_shared_buffer((std::size_t)IS_GEN * IC * 2);

  // The DiT computes in bf16 (f16's 65504 range overflows on the deep residual/
  // attention stream). The sampler boundary is f16, so upcast the latents inputs
  // to bf16 once here (lossless -- bf16's exponent range covers f16); the
  // returned velocity is downcast back to f16 at the end so callers are
  // unchanged. (fused_text is upcast into `joint` above.)
  auto up_bf16 = [&](const SharedBuffer& src, std::size_t n) -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer(n * 2);
    if (o.empty()) { return o; }
    const auto* s = static_cast<const _Float16*>(src.contents());
    auto* d = static_cast<std::uint16_t*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_((float)s[i]); }
    return o;
  };
  const SharedBuffer latents_b = up_bf16(latents, (std::size_t)IS_GEN * IC);
  std::vector<SharedBuffer> refs_b;
  refs_b.reserve(refs.size());
  for (const auto& r : refs) {
    refs_b.push_back(up_bf16(r.latents, (std::size_t)r.seq * IC));
  }
  if (latents_b.empty()) { return {}; }

  // Host-built sinusoidal timestep row [1, TD] (cos-first, input scaled x1000).
  // Timestep-dependent -> rewritten every call.
  {
    auto* ti = static_cast<std::uint16_t*>(te_in.contents());   // bf16 (DiT bf16)
    const int half = TD / 2;
    for (int i = 0; i < half; ++i) {
      const double fr = std::exp(-std::log(1e4) * (double)i / (double)half);
      const double ang = (double)timestep * 1000.0 * fr;
      ti[i] = f32_to_bf16_((float)std::cos(ang));
      ti[half + i] = f32_to_bf16_((float)std::sin(ang));
    }
  }

  // Joint residual [seq, HID]: text rows = fused_text (host copy), image rows =
  // img_in(latents) (written in-stream at the image offset). The DiT computes in
  // bf16; upcast the f16 fused-text boundary into the bf16 joint residual.
  {
    const auto* s = static_cast<const _Float16*>(fused_text.contents());
    auto* d = static_cast<std::uint16_t*>(joint.contents());
    const std::size_t n = (std::size_t)TS * HID;
    for (std::size_t i = 0; i < n; ++i) { d[i] = f32_to_bf16_((float)s[i]); }
  }
  const bool tap = (stop_after_block >= 0 && stop_after_block < c.n_layers);

  // Env-gated per-section GPU timing (VPIPE_KREA2_DIT_PROFILE). Inserts a
  // commit+wait barrier at each section boundary and accumulates wall time per
  // section, so the single deferred stream is split into timed slices. Only the
  // preloaded path (no _stream_blocks) is profiled; the barriers serialize the
  // GPU so absolute step time inflates a little, but the RELATIVE breakdown is
  // faithful. Persistent scratch is all _dit members, so flushing mid-block is
  // safe (buffers outlive each commit; commit().wait() preserves ordering).
  const bool prof = !_stream_blocks
                    && std::getenv("VPIPE_KREA2_DIT_PROFILE") != nullptr;
  // ---- env-gated streaming split (VPIPE_KREA2_STREAM_PROFILE) ----------
  // What a weight PREFETCH could hide. The streamed path loads a block
  // serially and then commits-and-waits before the weights free, so the
  // disk and the GPU take strict turns; both ends are already on the
  // critical path, so timing them adds no barriers of its own. Unlike the
  // section profile above this is NOT disabled while streaming -- the
  // streamed run is the only one it has anything to say about.
  const bool sprof = std::getenv("VPIPE_KREA2_STREAM_PROFILE") != nullptr;
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
  double t_cond = 0, t_qkv = 0, t_attn = 0, t_oproj = 0, t_ff = 0, t_norm = 0,
         t_final = 0, t_ffup = 0, t_ffact = 0;
  std::chrono::steady_clock::time_point mark;

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // Streaming-blocks mode commits the stream at block boundaries so each
    // just-in-time block's weights (freed after its iteration) are done on the
    // GPU before release. No-op-equivalent to one big stream (same ops, same
    // data deps via the persistent `joint`); only used when _stream_blocks.
    auto flush = [&]() {
      enc.end();
      stream.commit().wait();
      stream = _mc->make_command_stream();
      enc = stream.begin_compute();
    };
    // Profiling barrier: commit+wait the accumulated ops, add the elapsed slice
    // to `acc`, reopen the stream, and reset the mark. No-op unless profiling.
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
    // `lf`, when given, is the runtime adapter for THIS projection: the
    // base GEMM runs unchanged and the pair t = x A^T, y += s t B^T is
    // encoded after it. Threaded through the lambda rather than called
    // at each site so the adapter cannot be forgotten on one projection
    // and applied on the other seven.
    auto gemm = [&](const SharedBuffer& xin, const QWeight& w,
                    const SharedBuffer& y, std::size_t ye, int M, int N,
                    int K, const lora::Stack& lf = lora::Stack{}) {
      auto lora_after = [&]() {
        _lora.apply(enc, xin, 0, lf, y, ye, M, N, K, _mma_min_m);
      };
      if (gemm_mma_(enc, xin, w, y, ye, M, N, K)) { lora_after(); return; }
      int bm = 32, bn = 32;               // steel tile
      if (w.quantized) {
        // BM=64 tile once M amortizes it (the DiT block GEMMs, M ~= seq);
        // small-M shapes keep the 32x32 tile. At high res (M >= 1024) the
        // BM=128 tile (_qmm_tile == 2) halves code re-reads again. _acc16
        // swaps in the FP16-pipe half-accumulate twin for the BM64 GEMMs.
        const bool huge = _qmm_tile == 2 && M >= 1024;
        const bool big  = _qmm_tile >= 1 && M >= 128;   // BM64 (or BM128 fallback)
        if (huge)     { bm = 128; }
        else if (big) { bm = 64; }
        const bool a16 = _acc16 && big && !huge;
        enc.set_function(
            w.bits == 8
                ? (huge ? _fn_qmm8_bm128
                        : big ? (a16 ? _fn_qmm8_bm64_a16 : _fn_qmm8_bm64)
                              : _fn_qmm8)
                : (huge ? _fn_qmm4_bm128
                        : big ? (a16 ? _fn_qmm4_bm64_a16 : _fn_qmm4_bm64)
                              : _fn_qmm4));
        enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
        enc.set_buffer(2, w.qbias); enc.set_buffer(3, xin);
        enc.set_buffer(4, y, ye * 2);
        enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
      } else {
        const metal_compute::ComputeFunction* f = &_fn_gemm;
        if (_gemm_tile == 1) {
          f = (_acc16 && M >= 128) ? &_fn_gemm_bm64_a16 : &_fn_gemm_bm64;
          bm = 64;
        }
        else if (_gemm_tile == 2) { f = &_fn_gemm_bm64bn64; bm = 64; bn = 64; }
        enc.set_function(*f);
        enc.set_buffer(0, xin); enc.set_buffer(1, w.w); enc.set_buffer(2, w.w);
        enc.set_buffer(3, y, ye * 2);
        enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
        enc.set_constant(7, 0);
      }
      // BM=128 uses WM=4 (256 threads): threadgroup {32,2,4}, grid z=4.
      const unsigned tgz = (bm == 128) ? 4u : 2u;
      enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                    (unsigned)(((M + bm - 1) / bm) * 2), tgz},
                   {32, 2, tgz});
      lora_after();
    };
    auto gemm_bias = [&](const SharedBuffer& xin, const QWeight& w,
                         const SharedBuffer& bs, const SharedBuffer& y,
                         std::size_t ye, int M, int N, int K) {
      if (gemm_mma_(enc, xin, w, y, ye, M, N, K)) {   // dense OR quant on M5
        enc.set_function(_fn_bias_add);
        enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, bs);
        enc.set_constant(2, N); enc.set_constant(3, M * N);
        enc.dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
        return;
      }
      if (w.quantized) {
        gemm(xin, w, y, ye, M, N, K);
        enc.set_function(_fn_bias_add);
        enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, bs);
        enc.set_constant(2, N); enc.set_constant(3, M * N);
        enc.dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
        return;
      }
      enc.set_function(_fn_gemm_bias);
      enc.set_buffer(0, xin); enc.set_buffer(1, w.w); enc.set_buffer(2, bs);
      enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
      enc.set_constant(7, 1);
      enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                    (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
    };
    auto rms = [&](const SharedBuffer& x, std::size_t xe, const SharedBuffer& w,
                   const SharedBuffer& y, std::size_t ye, int R, int Hd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w);
      enc.set_buffer(2, y, ye * 2);
      enc.set_constant(3, Hd); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out, int A,
                         int Bd, int D) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, A); enc.set_constant(3, Bd); enc.set_constant(4, D);
      enc.dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A}, {(unsigned)D, 1, 1});
    };
    auto rope = [&](const SharedBuffer& x, int H, int T, int D) {
      enc.set_function(_fn_rope_table);
      enc.set_buffer(0, x); enc.set_buffer(1, rcos); enc.set_buffer(2, rsin);
      enc.set_constant(3, H); enc.set_constant(4, T); enc.set_constant(5, D);
      enc.dispatch({(unsigned)(D / 2), (unsigned)T, (unsigned)H},
                   {(unsigned)(D / 2), 1, 1});
    };
    auto sdpa = [&](const SharedBuffer& qb, const SharedBuffer& kb,
                    const SharedBuffer& vb, const SharedBuffer& out, float scale,
                    int T, int D, int Hq, int Hkv, int nq, int kvstride) {
      enc.set_function(_fn_sdpa);
      enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
      enc.set_buffer(3, out);
      enc.set_constant(4, scale); enc.set_constant(5, T); enc.set_constant(6, D);
      enc.set_constant(7, Hq); enc.set_constant(8, Hkv); enc.set_constant(9, nq);
      enc.set_constant(10, kvstride);
      enc.dispatch({32, (unsigned)Hq, (unsigned)nq}, {32, 1, 1});
    };
    auto elt3 = [&](const metal_compute::ComputeFunction& fn,
                    const SharedBuffer& a, const SharedBuffer& b2,
                    const SharedBuffer& out, int nn) {
      enc.set_function(fn);
      enc.set_buffer(0, a); enc.set_buffer(1, b2); enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto gelu = [&](const SharedBuffer& x, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_gelu_tanh);
      enc.set_buffer(0, x); enc.set_buffer(1, y);
      enc.set_constant(2, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    // AWQ calib tap: acc[n] = max(acc[n], max_row |in[row,n]|) over M rows.
    auto colmax = [&](const SharedBuffer& in, const SharedBuffer& acc, int M,
                      int N) {
      enc.set_function(_fn_colabsmax);
      enc.set_buffer(0, in); enc.set_buffer(1, acc);
      enc.set_constant(2, M); enc.set_constant(3, N);
      enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
    };
    // adaLN modulate out = (1+scale)*x + shift (scale/shift = offset slices).
    // vec4 eligibility for the twins below: they load/store 8 bytes at a time
    // off each bound base, so every element offset must be 4-aligned (an
    // unaligned vec4 access is undefined, not just slow) and `total` must be a
    // whole number of rows for the 2-D grid to cover it.
    auto elt4_ok = [&](int N, int total,
                       std::initializer_list<std::size_t> offs) {
      if ((N % 4) != 0 || N <= 0 || (total % N) != 0) { return false; }
      for (std::size_t o : offs) { if ((o % 4) != 0) { return false; } }
      return true;
    };
    auto adaln = [&](const SharedBuffer& x, const SharedBuffer& sh,
                     std::size_t scale_e, std::size_t shift_e,
                     const SharedBuffer& out, int N, int total) {
      if (_fn_adaln4.valid() && elt4_ok(N, total, {scale_e, shift_e})) {
        enc.set_function(_fn_adaln4);
        enc.set_buffer(0, x); enc.set_buffer(1, sh, scale_e * 2);
        enc.set_buffer(2, sh, shift_e * 2); enc.set_buffer(3, out);
        enc.set_constant(4, N / 4); enc.set_constant(5, total / N);
        enc.dispatch({(unsigned)(N / 4), (unsigned)(total / N), 1},
                     {256, 1, 1});
        return;
      }
      enc.set_function(_fn_adaln);
      enc.set_buffer(0, x); enc.set_buffer(1, sh, scale_e * 2);
      enc.set_buffer(2, sh, shift_e * 2); enc.set_buffer(3, out);
      enc.set_constant(4, N); enc.set_constant(5, total);
      enc.dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
    };
    // gated residual h += gate * sub (gate = offset slice).
    auto gated = [&](const SharedBuffer& h, const SharedBuffer& gt,
                     std::size_t gate_e, const SharedBuffer& sub, int N,
                     int total) {
      if (_fn_gated4.valid() && elt4_ok(N, total, {gate_e})) {
        enc.set_function(_fn_gated4);
        enc.set_buffer(0, h); enc.set_buffer(1, gt, gate_e * 2);
        enc.set_buffer(2, sub);
        enc.set_constant(3, N / 4); enc.set_constant(4, total / N);
        enc.dispatch({(unsigned)(N / 4), (unsigned)(total / N), 1},
                     {256, 1, 1});
        return;
      }
      enc.set_function(_fn_gated);
      enc.set_buffer(0, h); enc.set_buffer(1, gt, gate_e * 2);
      enc.set_buffer(2, sub);
      enc.set_constant(3, N); enc.set_constant(4, total);
      enc.dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
    };

    // ---- conditioning ----
    if (prof) { mark = std::chrono::steady_clock::now(); }
    gemm_bias(te_in, _te_l1w, _te_l1b, te1, 0, 1, HID, TD);   // 256 -> HID
    gelu(te1, te1, HID);
    gemm_bias(te1, _te_l2w, _te_l2b, temb, 0, 1, HID, HID);   // temb [1, HID]
    gelu(temb, tmp, HID);
    gemm_bias(tmp, _tmp_w, _tmp_b, tmod, 0, 1, 6 * HID, HID); // temb_mod
    // ---- img_in -> joint image region ----
    // Generated tokens embed into joint[TS .. TS+IS_GEN]; each reference embeds
    // (same img_in) into the tail joint[TS+IS_GEN .. TS+IS].
    gemm_bias(latents_b, _img_in_w, _img_in_b, joint, (std::size_t)TS * HID,
              IS_GEN, HID, IC);
    {
      std::size_t ro = (std::size_t)TS + IS_GEN;   // ref token row offset
      for (std::size_t ri = 0; ri < refs.size(); ++ri) {
        gemm_bias(refs_b[ri], _img_in_w, _img_in_b, joint, ro * HID,
                  refs[ri].seq, HID, IC);
        ro += (std::size_t)refs[ri].seq;
      }
    }

    // ---- transformer blocks over the joint [text; image] sequence ----
    const float scale = 1.0f / std::sqrt((float)HD);
    // Steel flash-attention: build the head_dim-128 function for this joint
    // length (align_Q/align_K are shape-dependent function constants) and fill
    // its param block ONCE -- reused across all 28 blocks (same shape). GQA:
    // Hq=HED query heads, Hkv=KVH kv heads, gqa_factor = HED/KVH. Q/O are
    // [HED, seq, HD]; K/V are [KVH, seq, HD] (the head-major transposes already
    // produce this). Falls back to the scalar sdpa when unavailable.
    // On M5 (matrix cores) prefer the NAX variant attn_steel_nax_h_bd128
    // (matmul2d, bq=64/bk=32) over the ALU steel attn_steel_h_bd128
    // (simdgroup, bq=32/bk=16). Same AttnParams + func-const contract + tg
    // {32,wm=4,1}; only the tile sizes + the kernel differ (mirrors the vision
    // tower). VPIPE_KREA2_NO_ATTN_NAX forces the ALU steel (A/B).
    const bool nax = _use_attn_nax && _lib_attn_nax.valid();
    const int A_BQ = nax ? 64 : 32;
    const int A_BK = nax ? 32 : 16;
    metal_compute::ComputeFunction fn_attn;
    bool use_steel = _steel_attn_ok && KVH > 0 && HED % KVH == 0
                     && !_attn_params.empty();
    if (use_steel) {
      auto* p = static_cast<SteelAttnParams*>(_attn_params.contents());
      p->B = 1; p->H = HED; p->D = HD; p->qL = seq; p->kL = seq;
      p->gqa_factor = HED / KVH; p->scale = scale;
      p->NQ = (seq + A_BQ - 1) / A_BQ; p->NK = (seq + A_BK - 1) / A_BK;
      p->NQ_aligned = seq / A_BQ; p->NK_aligned = seq / A_BK;
      p->qL_rem = seq - p->NQ_aligned * A_BQ;
      p->kL_rem = seq - p->NK_aligned * A_BK;
      p->qL_off = 0;
      p->Q_strides[0] = (std::int64_t)HED * seq * HD;
      p->Q_strides[1] = (std::int64_t)seq * HD; p->Q_strides[2] = HD;
      p->K_strides[0] = (std::int64_t)KVH * seq * HD;
      p->K_strides[1] = (std::int64_t)seq * HD; p->K_strides[2] = HD;
      p->V_strides[0] = p->K_strides[0];
      p->V_strides[1] = p->K_strides[1]; p->V_strides[2] = HD;
      p->O_strides[0] = p->Q_strides[0];
      p->O_strides[1] = p->Q_strides[1]; p->O_strides[2] = HD;
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (seq % A_BQ) == 0).set_bool(201, (seq % A_BK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      // The DiT runs bf16: use the bf16 attention entries (Q/K/V are bf16).
      fn_attn = nax
          ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
          : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
      use_steel = fn_attn.valid();
    }
    const unsigned a_nqb = (unsigned)((seq + A_BQ - 1) / A_BQ);
    psplit(t_cond);
    // Re-arm residency growth for this forward; the ratchet from any
    // earlier eviction deliberately survives. Also the one place the OS
    // gets to answer back that it has started swapping.
    if (_wire.on()) {
      // Retry a pool that refused earlier -- the refusal may have been
      // another process spiking, and a run that never asks again holds a
      // small resident set for the whole schedule on the strength of one
      // syscall. Growth stopped when the budget ran out and cannot see
      // that the budget moved, so a successful retry has to say so.
      if (_wire.retry(_mc, wire_retry_slack_())) {
        _resid.note_landscape_changed();
      }
      // Scratch and trunk take their place in the pool BEFORE this
      // forward's block admissions start asking for room: the blocks are
      // the shed-able half, so a pool that runs out should run out there.
      wire_fixed_(true);
    }
    const auto mbudget = _mc->memory_budget();
    _resid.begin_forward(mbudget, [this]() -> std::size_t {
      return evict_tail_block_();
    });
    // And the measurement behind the cheap gate: only when our own
    // compressed footprint has moved is the page walk worth its ~57 ms
    // per 4.3 GB.
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
              "MetalKrea2Transformer: resident weights are only {}% in "
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
    // JOIN ANY OUTSTANDING READ ON EVERY EXIT FROM THE STACK. A prefetch
    // may be filling a slot, and this forward returns from a dozen
    // places. Freeing or reading a slot while a reader thread writes
    // into it is a use-after-free, and a scope guard is the only version
    // of this that cannot be forgotten at the thirteenth return.
    struct SlotJoin {
      BlockSlots<Block>* s;
      ~SlotJoin() { s->join(); }
    } slot_join{&_slots};
    _slots.begin_forward();
    // The adapter's [seq, rank] intermediate, sized ONCE for the widest
    // projection this pass will take. Without it apply() returns early
    // and the adapter silently does nothing -- which looks exactly like
    // an adapter that had no effect, so it is allocated here rather than
    // lazily inside the block loop.
    if (_lora_rank_total > 0) {
      _lora_scratch_rows = (std::size_t)seq;
      _lora.ensure_scratch(_lora_scratch_rows *
                           (std::size_t)_lora_rank_total);
    }
    if (_stream_blocks) { flush(); }   // commit conditioning before streaming
    for (int L = 0; L < c.n_layers; ++L) {
      // Cooperative stop: bail EVERY block (not just the streamed tail) so a
      // pipeline stop is honored within ~one block even on the preloaded path
      // (a slow high-res step otherwise runs all blocks before responding).
      if (_stream_stop && _stream_stop()) { return {}; }
      if (_block_progress) { _block_progress(L, c.n_layers); }
      // Resident in _blocks once residency has promoted it, and
      // streamed until then. An unfilled slot reads as empty.
      const bool held = L < (int)_blocks.size() &&
                        !_blocks[(std::size_t)L].q.empty();
      const bool streaming = _stream_blocks && !held;
      const Block* streamed = nullptr;
      if (streaming) {
        const auto rd0 = sp_now();
        // Two reusable destinations, refilled in place with pread and
        // with the next block's read already issued under the previous
        // block's GPU work. See shared/block-slots.h.
        streamed = _slots.acquire(L);
        if (streamed == nullptr) { return {}; }
        sp_add(sp_read_ms, rd0);
        if (sprof) { sp_read_bytes += read_bytes_(*streamed); ++sp_blocks; }
        if (_mc->session() != nullptr) {
          _mc->session()->log_debug(fmt(
              "DiT forward: streamed block {}/{} (seq {})", L + 1, c.n_layers,
              seq));
        }
      }
      const Block& b = streaming ? *streamed : _blocks[(std::size_t)L];
      elt3(_fn_residual, tmod, b.sst, mod, 6 * HID);          // mod = temb_mod+sst
      rms(joint, 0, b.n1, n1, 0, seq, HID);
      adaln(n1, mod, 0, HID, nm, HID, seq * HID);             // (1+pre_s)*n1+pre_sh
      psplit(t_norm);
      if (_calib_on) { colmax(nm, _cb_qkv[(std::size_t)L], seq, HID); }
      auto lb = [&](lora::Factors BlockLora::* d) {
        return lora_at_(LoraStack::kMain, L, d);
      };
      gemm(nm, b.q, q, 0, seq, qd, HID, lb(&BlockLora::q));
      gemm(nm, b.k, k, 0, seq, kd, HID, lb(&BlockLora::k));
      gemm(nm, b.v, v, 0, seq, kd, HID, lb(&BlockLora::v));
      gemm(nm, b.gate, gate, 0, seq, HID, HID, lb(&BlockLora::gate));
      psplit(t_qkv);
      rms(q, 0, b.qn, q, 0, seq * HED, HD);
      rms(k, 0, b.kn, k, 0, seq * KVH, HD);
      transpose(q, qt, seq, HED, HD);            // [seq,H,hd] -> [H,seq,hd]
      transpose(k, kt, seq, KVH, HD);
      transpose(v, vt, seq, KVH, HD);
      rope(qt, HED, seq, HD);
      rope(kt, KVH, seq, HD);
      if (use_steel) {
        // Register-resident flash attention: Q/O [HED,seq,HD], K/V [KVH,seq,HD],
        // GQA via gqa_factor. Grid (32*NQ, 4*Hq, 1), tg (32,4,1) per MLX steel.
        enc.set_function(fn_attn);
        enc.set_buffer(0, qt); enc.set_buffer(1, kt); enc.set_buffer(2, vt);
        enc.set_buffer(3, atb); enc.set_buffer(4, _attn_params);
        enc.dispatch({32 * a_nqb, 4 * (unsigned)HED, 1}, {32, 4, 1});
      } else {
        sdpa(qt, kt, vt, atb, scale, seq, HD, HED, KVH, seq, seq);
      }
      transpose(atb, att, HED, seq, HD);         // [H,seq,hd] -> [seq,H,hd]
      elt3(_fn_mul_sigmoid, att, gate, att, seq * HID);
      psplit(t_attn);
      if (_calib_on) { colmax(att, _cb_o[(std::size_t)L], seq, HID); }
      gemm(att, b.o, o, 0, seq, HID, HID, lb(&BlockLora::o));
      gated(joint, mod, 2 * HID, o, HID, seq * HID);          // += pregate*attn
      psplit(t_oproj);
      rms(joint, 0, b.n2, n2, 0, seq, HID);
      adaln(n2, mod, 3 * HID, 4 * HID, nm, HID, seq * HID);
      psplit(t_norm);
      if (_calib_on) { colmax(nm, _cb_gu[(std::size_t)L], seq, HID); }
      if (!b.ff_gu.empty()) {
        // Fused interleaved gate|up. M5 mma: dequant + one matmul2d over the
        // fused weight -> _dit.gu [seq, 2*FF], folded to g[seq, FF] by
        // swiglu_interleaved. Steel: one GEMM whose register-local epilogue
        // writes g = silu(gate)*up directly (no intermediates, no separate
        // swiglu pass). N passed to either kernel is the fused 2*FF.
        if (!_dit.gu.empty() &&
            gemm_mma_(enc, nm, b.ff_gu, _dit.gu, 0, seq, 2 * FF, HID)) {
          psplit(t_ffup);
          enc.set_function(_fn_swiglu_inter);
          enc.set_buffer(0, _dit.gu); enc.set_buffer(1, g);
          enc.set_constant(2, seq); enc.set_constant(3, FF);
          enc.dispatch({(unsigned)(seq * FF), 1, 1}, {256, 1, 1});
          psplit(t_ffact);
        } else {
          // BM=128 fused-SwiGLU at high res (seq >= 1024): 4x fewer weight
          // re-reads than the default BM=32 tile for the biggest DiT GEMM.
          const bool huge = _qmm_tile == 2 && seq >= 1024 &&
                            _fn_qmm_swiglu4_bm128.valid();
          const bool a16  = _acc16 && !huge;
          const int  bm   = huge ? 128 : 32;
          const unsigned tgz = huge ? 4u : 2u;
          enc.set_function(
              b.ff_gu.bits == 8
                  ? (huge ? _fn_qmm_swiglu8_bm128
                          : (a16 ? _fn_qmm_swiglu8_a16 : _fn_qmm_swiglu8))
                  : (huge ? _fn_qmm_swiglu4_bm128
                          : (a16 ? _fn_qmm_swiglu4_a16 : _fn_qmm_swiglu4)));
          enc.set_buffer(0, b.ff_gu.codes); enc.set_buffer(1, b.ff_gu.scales);
          enc.set_buffer(2, b.ff_gu.qbias); enc.set_buffer(3, nm);
          enc.set_buffer(4, g);
          enc.set_constant(5, HID); enc.set_constant(6, 2 * FF);
          enc.set_constant(7, seq);
          enc.dispatch({(unsigned)(((2 * FF + 31) / 32) * 32),
                        (unsigned)(((seq + bm - 1) / bm) * 2), tgz},
                       {32, 2, tgz});
          psplit(t_ffup);
          psplit(t_ffact);   // activation fused into the GEMM epilogue
        }
      } else {
        gemm(nm, b.ff_gate, g, 0, seq, FF, HID,
             lb(&BlockLora::ff_gate));
        gemm(nm, b.ff_up, u, 0, seq, FF, HID, lb(&BlockLora::ff_up));
        psplit(t_ffup);
        elt3(_fn_swiglu, g, u, g, seq * FF);
        psplit(t_ffact);
      }
      if (_calib_on) { colmax(g, _cb_dn[(std::size_t)L], seq, FF); }
      gemm(g, b.ff_down, o, 0, seq, HID, FF,
           lb(&BlockLora::ff_down));
      gated(joint, mod, 5 * HID, o, HID, seq * HID);          // += postgate*ff
      psplit(t_ff);
      if (streaming) {
        const auto gp0 = sp_now();
        // COMMIT, then issue the next block's read, then WAIT. Between
        // the two the GPU is busy with block L and this thread has
        // nothing to do, which is the whole opportunity.
        enc.end();
        {
          metal_compute::CommandStream::Fence bf = stream.commit();
          int nxt = -1;
          for (int n = L + 1; n < c.n_layers; ++n) {
            const bool h = n < (int)_blocks.size() &&
                           !_blocks[(std::size_t)n].q.empty();
            if (!h) { nxt = n; break; }
          }
          _slots.prefetch(nxt);
          bf.wait();
        }
        stream = _mc->make_command_stream();
        enc = stream.begin_compute();
        sp_add(sp_gpu_ms, gp0);
        // The flush has been WAITED for, so nothing encoded still points
        // at this block's buffers -- which is why promotion happens here
        // and not at the top of the iteration. Keeping it costs no extra
        // memory (the bytes are already resident); the alternative is
        // re-reading the same block on every remaining step.
        if (L < (int)_blocks.size()) {
          // What promotion will KEEP, not what the slot holds: the
          // split gate/up are dropped just below and are neither wired
          // nor admitted. See resident_bytes_.
          const std::size_t nb = resident_bytes_(*streamed);
          // Past the wire budget there is nothing to gain: the block would
          // be kept unprotected, the compressor would take it (it is the
          // coldest memory in the process), and the next residency walk
          // would shed a block and ratchet the ceiling over the whole
          // resident set. Better not to hold it at all.
          if (_wire.wirable(nb) && _resid.admit(_mc, nb) &&
              _slots.promote_into(_blocks[(std::size_t)L])) {
            // A RESIDENT block does not need the split gate|up -- the
            // fused weight is what the forward reads, and the slot keeps
            // its own copies for the next refill. Dropping them here is
            // what stops promotion costing 25% more than it used to.
            if (!_blocks[(std::size_t)L].ff_gu.empty()) {
              _blocks[(std::size_t)L].ff_gate = QWeight{};
              _blocks[(std::size_t)L].ff_up = QWeight{};
            }
            // Wired LAST, after every write this block will ever get:
            // mlock pins the pages that exist NOW.
            _wire.note_wired(_mc,
                             wire_block_(_blocks[(std::size_t)L], true), nb);
            _resid.note_admitted(nb);
            if (_mc->session() != nullptr) {
              _mc->session()->log_debug(fmt(
                  "MetalKrea2Transformer: block {} resident ({} of {}, "
                  "{} MB, {} MB wired)", L, _resid.count(),
                  c.n_layers, _resid.bytes() >> 20,
                  _wire.wired_bytes() >> 20));
            }
          }
        }
      }
      if (stop_after_block == L) { break; }   // joint holds the block-L output
    }
    if (!tap) {
      // ---- final layer on the image tail ----
      elt3(_fn_residual, temb, _final_sst, modf, HID);        // modf0 = temb+sst0
      // modf1 = temb + sst[HID:] : reuse gemm-less add with offsets
      enc.set_function(_fn_residual);
      enc.set_buffer(0, temb); enc.set_buffer(1, _final_sst, (std::size_t)HID * 2);
      enc.set_buffer(2, modf, (std::size_t)HID * 2);
      enc.set_constant(3, HID);
      enc.dispatch({(unsigned)HID, 1, 1}, {256, 1, 1});
      // Only the generated image tokens (joint[TS .. TS+IS_GEN], the head of
      // the image region) get the final layer; reference tokens are discarded.
      rms(joint, (std::size_t)TS * HID, _final_norm, n1, 0, IS_GEN, HID);
      adaln(n1, modf, 0, HID, nm, HID, IS_GEN * HID);
      gemm_bias(nm, _final_lw, _final_lb, noise, 0, IS_GEN, IC, HID);
      psplit(t_final);
    }
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
        "MetalKrea2Transformer: streamed {} blocks -- read {:.0f} ms ({} MB "
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
  {
    // Backstop for GPU over-commit (OOM / page-fault on non-resident memory):
    // fail the step instead of returning a half-written velocity, which the
    // caller (generate_) would otherwise feed back into the sampler as noise.
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "MetalKrea2Transformer::forward_dit: {}",
            gpu_err.empty() ? "GPU command failed" : gpu_err));
      }
      return {};
    }
  }
  if (prof && _mc->session() != nullptr) {
    const double tot = t_cond + t_norm + t_qkv + t_attn + t_oproj + t_ffup
                       + t_ffact + t_ff + t_final;
    _mc->session()->log_normal(fmt(
        "DiT profile (seq={} img={} steps=1): total {} ms | cond {} | "
        "norm {} | qkv-gemm {} | attn {} | o-gemm {} | ff-up/gate-gemm {} | "
        "ff-act(swiglu) {} | ff-down-gemm {} | final {} "
        "(ms; barriers inflate absolute time)",
        seq, IS, (long)tot, (long)t_cond, (long)t_norm, (long)t_qkv,
        (long)t_attn, (long)t_oproj, (long)t_ffup, (long)t_ffact, (long)t_ff,
        (long)t_final));
  }
  // Return owned buffers (the persistent scratch is move-only + reused next
  // call). `noise` is a fresh per-call local -> move it out; the tap path hands
  // back an owned copy of the persistent joint.
  // Downcast the bf16 compute buffers to the public f16 boundary (callers/tests
  // consume f16; the sampler feeds the velocity back as f16 latents).
  auto down_f16 = [&](const SharedBuffer& src, std::size_t n) -> SharedBuffer {
    SharedBuffer o = _mc->make_shared_buffer(n * 2);
    if (o.empty()) { return o; }
    const auto* s = static_cast<const std::uint16_t*>(src.contents());
    auto* d = static_cast<_Float16*>(o.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)bf16_to_f32_(s[i]); }
    return o;
  };
  if (tap) { return down_f16(joint, (std::size_t)seq * HID); }
  return down_f16(noise, (std::size_t)IS_GEN * IC);
}

}  // namespace genai
}  // namespace vpipe
