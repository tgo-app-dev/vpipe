#include "generative-models/qwen-image/metal-qwen-image21-transformer.h"

#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/i8-gemm.h"
#include "generative-models/shared/metal-sage-attention.h"
#include "generative-models/shared/metal-sol-attention.h"
#include "generative-models/shared/mma-tile.h"

#include <chrono>
#include "generative-models/generative-model-manager.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
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
  // Round-to-nearest-even, which is what every other loader here does;
  // truncating instead costs half an ulp on every weight in the model.
  const std::uint32_t r = (u >> 16) & 1u;
  u += 0x7fffu + r;
  return (std::uint16_t)(u >> 16);
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

// The same, with 1 ADDED to every element.
//
// txt_in.text_norm is a ZERO-CENTERED RMS norm: the checkpoint stores
// `scale - 1` and the effective scale is `weight + 1`. Reading it as an
// ordinary norm gain leaves the text stream scaled by roughly nothing,
// which the model survives -- it just conditions on noise. Same trap as
// Krea-2's (1 + w) norms and the MTP norms.
SharedBuffer
to_bf16_plus1_(const MetalLlamaWeights& wts, MetalCompute* mc,
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
  for (std::size_t i = 0; i < n; ++i) {
    float v = 0.0f;
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      std::uint32_t u = (std::uint32_t)s[i] << 16;
      std::memcpy(&v, &u, 4);
    } else if (info->dtype == "F32") {
      v = static_cast<const float*>(raw.contents())[i];
    } else if (info->dtype == "F16") {
      v = (float)static_cast<const _Float16*>(raw.contents())[i];
    } else {
      return {};
    }
    d[i] = f32_to_bf16_(v + 1.0f);
  }
  return out;
}

// Derived-cache namespace. A WeightSet is shared by everything reading
// one checkpoint, so a key has to name the TRANSFORM, not just the
// tensor -- the +1 norm above is exactly why.
constexpr const char* kKey = "qwen-image21-dit/";

std::string
block_pre_(int L)
{
  return "transformer_blocks." + std::to_string(L) + ".";
}

}  // namespace

MetalQwenImage21Transformer::~MetalQwenImage21Transformer()
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

// ---- residency + the wired pool -------------------------------------

std::size_t
MetalQwenImage21Transformer::wire_block_(Block& b, bool on)
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
    one(w.w); one(w.codes); one(w.scales); one(w.qbias);
  };
  QWeight* q[] = {&b.qw, &b.kw, &b.vw, &b.ow,
                  &b.gate_w, &b.proj_w, &b.out_w};
  for (QWeight* w : q) { qw(*w); }
  one(b.nq); one(b.nk);
  return changed;
}

std::size_t
MetalQwenImage21Transformer::wire_fixed_(bool on)
{
  if (!_ws) { return 0; }
  std::size_t changed = 0;
  _ws->for_each_weight([&](SharedBuffer& b) {
    changed += _wire.wire_one(_mc, b, on);
  });
  return changed;
}

std::size_t
MetalQwenImage21Transformer::wire_retry_slack_() const
{
  // How much the box must have freed since a refusal before it is worth
  // asking again -- one block's worth, so a genuinely full box is never
  // asked. Re-opening the pool's ceiling makes its own check pass while
  // the mlock behind it still fails, which would leave a block resident
  // but UNWIRED, one per forward.
  if (_wire_block_hint > 0) { return _wire_block_hint; }
  if (_resid.count() > 0) {
    return _resid.bytes() / (std::size_t)_resid.count();
  }
  return 0;
}

std::size_t
MetalQwenImage21Transformer::evict_tail_block_()
{
  // From the END. The next forward starts at block 0, so the deepest
  // resident block is the one whose re-read is furthest away.
  for (int i = (int)_blocks.size() - 1; i >= 0; --i) {
    Block& b = _blocks[(std::size_t)i];
    const std::size_t n = block_bytes_(b);
    if (n == 0) { continue; }
    // Before the buffers go: give the wiring back. Dropping a wired
    // buffer unwires it in the kernel anyway, but only the pool's own
    // accounting keeps the budget honest -- without this a block's worth
    // leaks per eviction.
    _wire.note_unwired(wire_block_(b, false));
    b = Block{};
    return n;
  }
  return 0;
}

void
MetalQwenImage21Transformer::resident_pages_(std::size_t* examined,
                                             std::size_t* incore,
                                             std::size_t* paged_out) const
{
  *examined = 0;
  *incore = 0;
  if (paged_out != nullptr) { *paged_out = 0; }
  auto add = [&](const SharedBuffer& p) {
    if (p.byte_size() == 0) { return; }
    // A WIRED BUFFER CANNOT HAVE LEFT RAM, so asking spends the walk to
    // be told what mlock already guarantees. Skipped per BUFFER, not per
    // block: wire_block_ stops at the first refusal, so the unwired
    // remainder of a half-wired block is exactly what still needs
    // measuring. With everything wired `examined` stays 0, which the
    // caller reads as "no evidence" rather than as a shortfall.
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
    qw(b.gate_w); qw(b.proj_w); qw(b.out_w);
    add(b.nq); add(b.nk);
  }
}

// ---- config ---------------------------------------------------------

bool
MetalQwenImage21Transformer::Config::read_dims(const std::string& dir,
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
  if (o.contains("_class_name")) {
    const std::string cls(o.at("_class_name").as_string(""));
    // Named explicitly. "QwenImageTransformer2DModel" is the 20B
    // dual-stream model whose config would load here without complaint
    // and describe a different network entirely.
    if (cls != "QwenImage21Transformer2DModel") { return false; }
  }
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  out->n_heads = geti("num_attention_heads", out->n_heads);
  out->head_dim = geti("attention_head_dim", out->head_dim);
  out->hidden = out->n_heads * out->head_dim;
  out->n_layers = geti("num_layers", out->n_layers);
  out->in_channels = geti("in_channels", out->in_channels);
  out->out_channels = geti("out_channels", out->in_channels);
  out->txt_dim = geti("context_in_dim", out->txt_dim);
  out->mlp_ratio = geti("mlp_ratio", out->mlp_ratio);
  if (o.contains("eps")) {
    out->norm_eps = (float)o.at("eps").as_real(out->norm_eps);
  }
  if (o.contains("causal_condition")) {
    out->causal_condition =
        o.at("causal_condition").as_bool(out->causal_condition);
  }
  if (o.contains("axes_dims_rope")) {
    const FlexData a = o.at("axes_dims_rope");
    if (a.is_array()) {
      auto arr = a.as_array();
      for (std::size_t i = 0; i < 3 && i < arr.size(); ++i) {
        out->axes[i] = (int)arr.at(i).as_int(out->axes[i]);
      }
    }
  }
  // patch_size is 1 for 2.1 -- latents are consumed UNPATCHED -- and
  // in_channels already counts whatever a patch would have folded in, so
  // there is nothing to multiply here. Refuse a checkpoint that says
  // otherwise rather than quietly running the wrong stride.
  if (geti("patch_size", 1) != 1) { return false; }
  return out->hidden > 0 && out->n_layers > 0;
}

// ---- weights --------------------------------------------------------

SharedBuffer
MetalQwenImage21Transformer::to_elt_(WeightSet& ws, const std::string& name,
                                     Retain r)
{
  auto build = [&]() { return to_bf16_(ws.src(), _mc, name); };
  if (r == Retain::Streamed) { return ws.stream_derived(build); }
  return ws.derived(std::string(kKey) + "elt|" + name, build);
}

MetalQwenImage21Transformer::QWeight
MetalQwenImage21Transformer::load_qw_(WeightSet& ws, const std::string& name,
                                      Retain r)
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
  qw.w = to_elt_(ws, name + ".weight", r);
  return qw;
}

bool
MetalQwenImage21Transformer::load_block_(WeightSet& ws, int L, Block& b,
                                         Retain r)
{
  const std::string p = block_pre_(L);
  b.qw = load_qw_(ws, p + "attn.to_q", r);
  b.kw = load_qw_(ws, p + "attn.to_k", r);
  b.vw = load_qw_(ws, p + "attn.to_v", r);
  // to_out is a ModuleList, so the projection is ".0" -- not a typo.
  b.ow = load_qw_(ws, p + "attn.to_out.0", r);
  b.nq = to_elt_(ws, p + "attn.norm_q.weight", r);
  b.nk = to_elt_(ws, p + "attn.norm_k.weight", r);
  // SwiGLU: out(silu(gate_layer(x)) * proj(x)). `proj` is the UP
  // projection and `out` the down one, which is not the naming any other
  // family here uses.
  b.gate_w = load_qw_(ws, p + "img_mlp.gate_layer", r);
  b.proj_w = load_qw_(ws, p + "img_mlp.proj", r);
  b.out_w = load_qw_(ws, p + "img_mlp.out", r);
  return !b.qw.empty() && !b.kw.empty() && !b.vw.empty() && !b.ow.empty() &&
         !b.nq.empty() && !b.nk.empty() && !b.gate_w.empty() &&
         !b.proj_w.empty() && !b.out_w.empty();
}

bool
MetalQwenImage21Transformer::load_fixed_(WeightSet& ws)
{
  const Retain r = Retain::Cached;
  _img_in = load_qw_(ws, "img_in", r);
  _txt_in_a = load_qw_(ws, "txt_in.in_layer", r);
  _txt_in_b = load_qw_(ws, "txt_in.out_layer", r);
  // THE +1. See to_bf16_plus1_.
  _txt_norm = ws.derived(std::string(kKey) + "plus1|txt_in.text_norm.weight",
                         [&]() {
                           return to_bf16_plus1_(ws.src(), _mc,
                                                 "txt_in.text_norm.weight");
                         });
  _time_1 = load_qw_(ws, "time_text_embed.timestep_embedder.linear_1", r);
  _time_2 = load_qw_(ws, "time_text_embed.timestep_embedder.linear_2", r);
  // modulation is nn.Sequential(SiLU, Linear), so the Linear is ".1".
  _mod = load_qw_(ws, "modulation.1", r);
  _norm_out = load_qw_(ws, "norm_out.linear", r);
  _proj_out = load_qw_(ws, "proj_out", r);
  return !_img_in.empty() && !_txt_in_a.empty() && !_txt_in_b.empty() &&
         !_txt_norm.empty() && !_time_1.empty() && !_time_2.empty() &&
         !_mod.empty() && !_norm_out.empty() && !_proj_out.empty();
}

std::size_t
MetalQwenImage21Transformer::qw_bytes_(const QWeight& w)
{
  if (w.quantized) {
    return w.codes.byte_size() + w.scales.byte_size() + w.qbias.byte_size();
  }
  return w.w.byte_size();
}

std::size_t
MetalQwenImage21Transformer::block_bytes_(const Block& b)
{
  return qw_bytes_(b.qw) + qw_bytes_(b.kw) + qw_bytes_(b.vw) +
         qw_bytes_(b.ow) + b.nq.byte_size() + b.nk.byte_size() +
         qw_bytes_(b.gate_w) + qw_bytes_(b.proj_w) + qw_bytes_(b.out_w);
}

void
MetalQwenImage21Transformer::each_block_tensor_(
    int L, Block& b, const BlockSlots<Block>::TensorFn& fn) const
{
  const std::string p = block_pre_(L);
  auto qw = [&](QWeight& w, const char* nm) {
    if (w.quantized) {
      // Codes are the checkpoint's own u32 words; scales and biases are
      // f16 in the pack and bf16 here, which kBf16 converts in place.
      fn(p + nm + ".weight", w.codes, Placement::kRaw);
      fn(p + nm + ".scales", w.scales, Placement::kBf16);
      fn(p + nm + ".biases", w.qbias, Placement::kBf16);
    } else {
      fn(p + nm + ".weight", w.w, Placement::kBf16);
    }
  };
  qw(b.qw, "attn.to_q");
  qw(b.kw, "attn.to_k");
  qw(b.vw, "attn.to_v");
  qw(b.ow, "attn.to_out.0");
  fn(p + "attn.norm_q.weight", b.nq, Placement::kBf16);
  fn(p + "attn.norm_k.weight", b.nk, Placement::kBf16);
  qw(b.gate_w, "img_mlp.gate_layer");
  qw(b.proj_w, "img_mlp.proj");
  qw(b.out_w, "img_mlp.out");
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
MetalQwenImage21Transformer::clone_block_(const Block& src, Block& dst,
                                          bool copy) const
{
  bool ok = true;
  auto cq = [&](const QWeight& s, QWeight& d) {
    d.quantized = s.quantized;
    d.bits = s.bits;
    ok = clone_buf_(_mc, s.w, d.w, copy) && ok;
    ok = clone_buf_(_mc, s.codes, d.codes, copy) && ok;
    ok = clone_buf_(_mc, s.scales, d.scales, copy) && ok;
    ok = clone_buf_(_mc, s.qbias, d.qbias, copy) && ok;
  };
  cq(src.qw, dst.qw); cq(src.kw, dst.kw); cq(src.vw, dst.vw);
  cq(src.ow, dst.ow);
  ok = clone_buf_(_mc, src.nq, dst.nq, copy) && ok;
  ok = clone_buf_(_mc, src.nk, dst.nk, copy) && ok;
  cq(src.gate_w, dst.gate_w); cq(src.proj_w, dst.proj_w);
  cq(src.out_w, dst.out_w);
  return ok;
}

SharedBuffer
MetalQwenImage21Transformer::rebuild_one_(const std::string& nm, Placement how)
{
  if (_ws == nullptr) { return {}; }
  if (how == Placement::kRaw) {
    return _ws->stream_tensor(nm, _mc, WeightSet::Residency::Copied);
  }
  return _ws->stream_derived([&]() { return to_bf16_(_ws->src(), _mc, nm); });
}

void
MetalQwenImage21Transformer::configure_slots_()
{
  BlockSlots<Block>::Ops ops;
  ops.each = [this](int L, Block& b, const BlockSlots<Block>::TensorFn& fn) {
    each_block_tensor_(L, b, fn);
  };
  ops.rebuild_one = [this](const std::string& nm, Placement how) {
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
  _slots.configure(_mc, std::move(ops), "MetalQwenImage21Transformer",
                   "VPIPE_QWEN_IMAGE21_NO_SLOTS");
}

// ---- load -----------------------------------------------------------

std::unique_ptr<MetalQwenImage21Transformer>
MetalQwenImage21Transformer::load(const std::string& model_dir,
                                  MetalCompute* mc, const Config& cfg,
                                  bool stream_blocks)
{
  return load(WeightSet::open(model_dir, nullptr), mc, cfg, stream_blocks);
}

std::unique_ptr<MetalQwenImage21Transformer>
MetalQwenImage21Transformer::load(std::shared_ptr<WeightSet> ws,
                                  MetalCompute* mc, const Config& cfg,
                                  bool stream_blocks)
{
  if (ws == nullptr || mc == nullptr) { return nullptr; }
  std::unique_ptr<MetalQwenImage21Transformer> m(
      new MetalQwenImage21Transformer());
  m->_mc = mc;
  m->_cfg = cfg;
  m->_ws = std::move(ws);
  m->_stream_blocks = stream_blocks;

  // Quantization is read off the checkpoint, not configured: a pack
  // carries `.scales` beside its codes and a dense one does not.
  {
    const MetalLlamaWeights& w = m->_ws->src();
    const auto* si = w.info(block_pre_(0) + "attn.to_q.scales");
    const auto* ci = w.info(block_pre_(0) + "attn.to_q.weight");
    if (si != nullptr && ci != nullptr && si->shape.size() == 2 &&
        ci->shape.size() == 2 && si->shape[1] > 0) {
      m->_quant_group = (int)(ci->shape[1] * 32 / si->shape[1]) > 0
                            ? (int)(m->_cfg.hidden / si->shape[1])
                            : 64;
      if (m->_quant_group <= 0) { m->_quant_group = 64; }
      const long K = si->shape[1] * (long)m->_quant_group;
      m->_quant_bits = K > 0 ? (int)(ci->shape[1] * 32 / K) : 0;
      if (m->_quant_bits != 4 && m->_quant_bits != 8) { m->_quant_bits = 0; }
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
  // every GEMM in the model. Same arithmetic; a bigger BM computes more
  // output rows per threadgroup from the same 128 threads, so each
  // [BN,BK] weight tile is re-read M/BM times instead of M/32 --
  // and these GEMMs are weight-bandwidth-bound, which is why the wider
  // tile is not worse at small M either. Both siblings on this shape
  // (Qwen-Image-Edit, Krea-2) prefer it; this model did not load it.
  m->_fn_gemm_bm64 = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_gemm_bm64bn64 =
      m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  m->_fn_ln_mod = m->_lib_elt.function("layernorm_modulate_f16");
  m->_fn_ln_plain = m->_lib_elt.function("layer_norm_plain_f16");
  m->_fn_gated_tanh = m->_lib_elt.function("gated_residual_tanh_f16");
  // The four-wide twins. Both are the SAME arithmetic over a quarter
  // of the threads; the scalar forms stay as the fallback for a
  // width that is not a multiple of 4 (nothing in this model), and
  // because an older metallib may not carry them.
  m->_fn_gated_tanh4 = m->_lib_elt.function("gated_residual_tanh_v4_f16");
  m->_fn_swiglu4 = m->_lib_elt.function("swiglu_v4_f16");
  m->_fn_swiglu = m->_lib_elt.function("swiglu_f16");
  m->_fn_gelu_tanh = m->_lib_elt.function("gelu_tanh_ff_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_copy_rows = m->_lib_elt.function("copy_rect_f16");
  m->_fn_scale_rows = m->_lib_elt.function("adaln_modulate_f16");
  m->_fn_rms = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_silu = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_tr_rope = m->_lib_rope.function("transpose_rope_pair_ftab_f16");
  // NOT fetched by name here. The steel entries are declared with
  // function constants, so an unspecialized handle cannot build a
  // pipeline state at all -- it fails inside Metal's validator rather
  // than coming back invalid. They are built per forward, with the
  // alignment constants the geometry decides.

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
  // its dispatch, which in a 32-block stack is a picture, not a failure --
  // and before ComputeEncoder was taught to refuse it, it was worse than
  // that: the previous op's kernel ran again in its place.
  if (!m->_fn_gemm_t.valid() || !m->_fn_ln_mod.valid() ||
      !m->_fn_gated_tanh.valid() || !m->_fn_swiglu.valid() ||
      !m->_fn_rms.valid() || !m->_fn_tr_rope.valid() ||
      !m->_fn_residual.valid() || !m->_fn_silu.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_gelu_tanh.valid() ||
      !m->_fn_copy_rows.valid()) {
    if (mc->session() != nullptr) {
      mc->session()->error(fmt("MetalQwenImage21Transformer: a required "
                               "kernel did not build"));
    }
    return nullptr;
  }
  // A QUANTIZED CHECKPOINT WHOSE GEMM DID NOT LOAD IS A REFUSAL, not a
  // degraded run. `set_function` on an invalid handle leaves the
  // PREVIOUS kernel bound and the dispatch that follows runs THAT one
  // over 4-bit codes, so a mis-spelt entry point does not read as a
  // missing kernel at all -- on this family it read as a picture with
  // no alpha, three times faster than bf16.
  if (m->_quant_bits > 0 &&
      (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid())) {
    if (mc->session() != nullptr) {
      mc->session()->error(fmt(
          "MetalQwenImage21Transformer: no quantized GEMM for group {} -- "
          "this checkpoint cannot be run (re-quantize at group 64)",
          m->_quant_group));
    }
    return nullptr;
  }
  m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));
  m->_steel_attn_ok = m->_lib_attn.valid() && !m->_attn_params.empty();
  m->_use_attn_nax = m->_steel_attn_ok && m->_lib_attn_nax.valid() &&
                     mc->supports_matrix_cores() &&
                     std::getenv("VPIPE_QWEN_IMAGE21_NO_NAX_ATTN") == nullptr;
  m->_use_mma2 = mc->supports_matrix_cores() && m->_fn_gemm_mma.valid() &&
                 std::getenv("VPIPE_QWEN_IMAGE21_NO_MMA2") == nullptr;
  // INT8 ACCELERATED GEMM. Not a weight quantization: the bf16
  // activation is quantized per (row, group of 64) on the fly and the
  // dense weight per (out-channel, group) into a reusable scratch, and
  // the product runs on the matrix units' int8 pipe at roughly 1.5x the
  // matmul2d rate. The checkpoint on disk is untouched, which is what
  // separates this from w4 -- there is no stored precision to lose, only
  // the per-group rounding of one matmul.
  //
  // Kept ONLY if it turned itself on: the context answers `enabled()`
  // false where the kernels or the hardware are not there, and holding
  // a disabled one would report an accelerated run that never happened.
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm,
                                              /*bf16=*/true);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }
  // SAGE: int8 Q/K for the QK^T half of attention, which is the half
  // that grows with the square of the sequence. Only on the matrix-core
  // flash entry -- the int8 twin is a function constant on that kernel,
  // and building it where there is no driver to fill buffers 15..18 is
  // a dispatch that reads unbound memory.
  // SOL-ATTN: skip the key blocks that contribute almost nothing. The
  // only tier whose value RISES with the picture -- attention is 10.4%
  // of a 1024^2 step here and 36.3% of a 2048^2 one -- and, unlike
  // sage, it does not try to make the same arithmetic cheaper. It does
  // less of it.
  if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_SOL")) {
    m->_cfg.sol.enabled = std::atoi(e) != 0;
  }
  // The routing threshold, for the speed/accuracy A/B. HIGHER tau keeps
  // FEWER blocks: it is a threshold, not a budget.
  if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_SOL_TAU")) {
    m->_cfg.sol.tau = (float)std::atof(e);
  }
  // The ANE tier's A/B knob. It is otherwise reachable only through a
  // graph's accel bag, and it has to be switchable from a test.
  if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_ANE")) {
    m->_cfg.ane_ffn = std::atoi(e) != 0;
  }
  if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_ANE_ROWS")) {
    m->_cfg.ane_rows = std::atoi(e);
  }
  if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_ANE_QKV")) {
    m->_cfg.ane_qkv = std::atoi(e) != 0;
  }
  if (m->_cfg.sol.enabled) {
    std::string serr;
    m->_sol = MetalSolAttention::load(mc, /*bf16=*/true, &serr);
    if (m->_sol == nullptr && mc->session() != nullptr) {
      mc->session()->log_normal(fmt(
          "MetalQwenImage21Transformer: sol_attn unavailable -- {}",
          serr.empty() ? "kernels did not load" : serr));
    }
  }
  {
    // A/B knob, matching VPIPE_NO_SAGE_ATTN in the other direction: the
    // tier is otherwise only reachable through a graph's accel bag, and
    // a per-block trace of it needs to be switchable from a test.
    if (const char* e = std::getenv("VPIPE_QWEN_IMAGE21_SAGE")) {
      m->_cfg.sage.enabled = std::atoi(e) != 0;
    }
    bool sage_fatal = false;
    m->_sage = MetalSageAttention::load_for_model(
        mc, /*bf16=*/true, m->_cfg.sage, "MetalQwenImage21Transformer",
        &sage_fatal);
    if (sage_fatal) { return nullptr; }
    if (m->_sage && !m->_use_attn_nax) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt(
            "MetalQwenImage21Transformer: sage_attn is off -- the "
            "matrix-core flash entry is unavailable here"));
      }
      m->_sage.reset();
    }
  }
  // BM=64 by default wherever it loaded. VPIPE_QWEN_IMAGE21_GEMM_TILE
  // pins it for an A/B; a tile whose kernel is missing falls back to
  // the next one down rather than to an invalid function.
  if (const char* t = std::getenv("VPIPE_QWEN_IMAGE21_GEMM_TILE")) {
    m->_gemm_tile = std::atoi(t);
  }
  if (m->_gemm_tile == 2 && !m->_fn_gemm_bm64bn64.valid()) {
    m->_gemm_tile = 1;
  }
  if (m->_gemm_tile == 1 && !m->_fn_gemm_bm64.valid()) {
    m->_gemm_tile = 0;
  }

  if (!m->load_fixed_(*m->_ws)) {
    if (mc->session() != nullptr) {
      mc->session()->error(fmt("MetalQwenImage21Transformer: the "
                               "model-level weights are incomplete"));
    }
    return nullptr;
  }

  m->_wire.open(mc);
  if (stream_blocks) {
    m->configure_slots_();
    // Sized to n_layers so a PROMOTED block has somewhere to land; an
    // unfilled entry reads as empty, which is how the forward tells a
    // resident block from one that is still streamed.
    m->_blocks.resize((std::size_t)m->_cfg.n_layers);
  } else {
    m->_blocks.resize((std::size_t)m->_cfg.n_layers);
    for (int L = 0; L < m->_cfg.n_layers; ++L) {
      if (!m->load_block_(*m->_ws, L, m->_blocks[(std::size_t)L],
                          Retain::Cached)) {
        if (mc->session() != nullptr) {
          mc->session()->error(fmt("MetalQwenImage21Transformer: block {} "
                                   "is incomplete", L));
        }
        return nullptr;
      }
    }
  }
  if (mc->session() != nullptr) {
    mc->session()->log_normal(fmt(
        "MetalQwenImage21Transformer: {} blocks, hidden {}, {} heads x {}, "
        "ffn {}, {}{}", m->_cfg.n_layers, m->_cfg.hidden, m->_cfg.n_heads,
        m->_cfg.head_dim, m->_cfg.hidden * m->_cfg.mlp_ratio,
        m->_quant_bits > 0
            ? "w" + std::to_string(m->_quant_bits) + " g" +
                  std::to_string(m->_quant_group)
            : std::string("bf16"),
        stream_blocks ? ", streaming" : ""));
  }
  return m;
}

// The timestep sinusoid, on the host.
//
// diffusers' QwenImage21TemporalTimesteps, and TWO details of it are
// load-bearing. COS occupies the first half and sin the second, which is
// the opposite of the sin-first convention most of this tree's families
// use. And `time_factor` is 1000, so a sigma in [0,1] reaches an angle
// of ~1000 rad, where a small relative error in the frequency is whole
// radians of phase -- which is why the table is built in double and only
// the result is narrowed.
std::vector<float>
MetalQwenImage21Transformer::time_proj_(float sigma) const
{
  const int D = _cfg.time_proj;
  const int half = D / 2;
  std::vector<float> out((std::size_t)D, 0.0f);
  const double t = 1000.0 * (double)sigma;
  for (int i = 0; i < half; ++i) {
    const double f = std::exp(-std::log(10000.0) * (double)i / (double)half);
    const double a = t * f;
    out[(std::size_t)i] = (float)std::cos(a);
    out[(std::size_t)(half + i)] = (float)std::sin(a);
  }
  return out;
}

// cos/sin tables over the joint sequence, f32 [JT, head_dim], in the
// duplicated-pair layout transpose_rope_pair_ftab_f16 reads (each pair's
// value written twice). The positions come from the layout, including
// the NEGATIVE ones a centred image grid produces -- the angle is formed
// from the signed position directly, so no table of negative indices is
// needed.
void
MetalQwenImage21Transformer::build_rope_(const qi21::Layout& lay,
                                         SharedBuffer* cos_out,
                                         SharedBuffer* sin_out) const
{
  const int D = _cfg.head_dim;
  const int P = D / 2;
  const int a0 = _cfg.axes[0], a1 = _cfg.axes[1], a2 = _cfg.axes[2];
  const int p0 = a0 / 2, p1 = a1 / 2;
  const int T = lay.joint_len;
  *cos_out = _mc->make_shared_buffer((std::size_t)T * D * 4);
  *sin_out = _mc->make_shared_buffer((std::size_t)T * D * 4);
  if (cos_out->empty() || sin_out->empty()) { return; }
  auto* cb = static_cast<float*>(cos_out->contents());
  auto* sb = static_cast<float*>(sin_out->contents());
  auto invfreq = [&](int j) -> double {
    int axis_dim, local;
    if (j < p0) { axis_dim = a0; local = j; }
    else if (j < p0 + p1) { axis_dim = a1; local = j - p0; }
    else { axis_dim = a2; local = j - p0 - p1; }
    return std::pow((double)_cfg.rope_theta,
                    -2.0 * (double)local / (double)axis_dim);
  };
  for (int t = 0; t < T; ++t) {
    const double pf = (double)lay.pos_t[(std::size_t)t];
    const double ph = (double)lay.pos_h[(std::size_t)t];
    const double pw = (double)lay.pos_w[(std::size_t)t];
    for (int j = 0; j < P; ++j) {
      const double pos = (j < p0) ? pf : (j < p0 + p1 ? ph : pw);
      const double ang = pos * invfreq(j);
      const float c = (float)std::cos(ang), sn = (float)std::sin(ang);
      const std::size_t o = (std::size_t)t * D + 2 * j;
      cb[o] = c; cb[o + 1] = c; sb[o] = sn; sb[o + 1] = sn;
    }
  }
}

SharedBuffer
MetalQwenImage21Transformer::forward(const Request& req, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return SharedBuffer{};
  };
  if (req.layout == nullptr || req.latents == nullptr || req.txt == nullptr) {
    return fail("qwen-image-2.1 forward: missing input");
  }
  const qi21::Layout& lay = *req.layout;
  const int H = _cfg.hidden, NH = _cfg.n_heads, Hd = _cfg.head_dim;
  const int FF = H * _cfg.mlp_ratio;
  const int JT = lay.joint_len;
  const int IMG = lay.image_len;
  if (JT <= 0 || IMG <= 0 || NH * Hd != H) {
    return fail("qwen-image-2.1 forward: degenerate geometry");
  }
  const bool kv_fill = req.kv_mode == KvMode::kFill;
  const bool kv_use = req.kv_mode == KvMode::kCached;
  if ((kv_fill || kv_use) && req.kv == nullptr) {
    return fail("qwen-image-2.1 forward: a kv_mode was asked for with no "
                "cache to put it in");
  }
  if ((kv_fill || kv_use) && !_cfg.causal_condition) {
    // The cache is only sound because causal_condition modulates the
    // prefix from t=0, which is what makes its K and V independent of
    // the step. Without it the cached rows would be stale, and stale
    // here means a plausible picture.
    return fail("qwen-image-2.1 forward: the prefix KV cache requires "
                "causal_condition");
  }
  if (kv_use) {
    const KvCache& c = *req.kv;
    if (!c.populated() || (int)c.k.size() != _cfg.n_layers ||
        c.prefix != lay.prefix_len || c.joint != lay.joint_len ||
        c.target != lay.target_len) {
      return fail("qwen-image-2.1 forward: the cache was filled at a "
                  "different geometry -- refill it rather than reusing it");
    }
  }
  if (lay.prefix_len <= 0 && (kv_fill || kv_use)) {
    return fail("qwen-image-2.1 forward: nothing to cache (empty prefix)");
  }

  // ROWS is what this step actually runs as QUERIES. A cached step runs
  // the target rows only -- the prefix's K and V come from the cache and
  // its outputs are discarded -- so the whole stream below is that
  // slice, exactly as the reference slices hidden_states, rotary_emb and
  // the modulation mask together.
  const int ROWS = kv_use ? lay.target_len : lay.joint_len;
  const int ROW0 = kv_use ? lay.prefix_len : 0;
  // Left padding would have to be excluded as an attention KEY, and the
  // steel path here carries no mask. Batch-1 never pads, which is every
  // caller today; refuse rather than attend to padding.
  for (std::uint8_t v : lay.key_valid) {
    if (v == 0) {
      return fail("qwen-image-2.1 forward: a padded prompt needs a key "
                  "mask this path does not carry");
    }
  }
  if (!_steel_attn_ok) {
    return fail("qwen-image-2.1 forward: the steel attention kernel is "
                "unavailable, and the block-causal decomposition IS the "
                "masking -- there is no scalar fallback that is right");
  }

  // LLM-lane perf event (perf-visualizer): one DiT forward per sampler
  // step, two per step under true-CFG. Value = the joint sequence
  // length, which on this model is what the cost actually scales with.
  // Mirrors the Krea-2 / FLUX.2 / Qwen-Image-Edit DiT events so this
  // family shows on the same lane as the rest.
  //
  // The joint length is read from the LAYOUT rather than the request: a
  // cached step runs only the target rows, and reporting the full joint
  // for it would draw every step the same width when they are not.
  PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDit,
                     kPerfLlmDitBegin,
                     (std::uint64_t)(kv_use ? lay.target_len
                                            : lay.joint_len));

  MetalCompute* mc = _mc;
  const float eps = _cfg.norm_eps;
  const int MOD_ROWS = _cfg.causal_condition ? 2 : 1;

  // ---- scratch ------------------------------------------------------
  auto buf = [&](std::size_t elems) {
    return mc->make_shared_buffer(elems * 2);
  };
  SharedBuffer jh = buf((std::size_t)ROWS * H);      // the running stream
  SharedBuffer tmp = buf((std::size_t)ROWS * H);
  SharedBuffer qb = buf((std::size_t)ROWS * H);
  SharedBuffer kb = buf((std::size_t)ROWS * H);
  SharedBuffer vb = buf((std::size_t)ROWS * H);
  SharedBuffer qh = buf((std::size_t)ROWS * H);
  SharedBuffer kh = buf((std::size_t)ROWS * H);
  SharedBuffer vh = buf((std::size_t)ROWS * H);
  SharedBuffer ao = buf((std::size_t)ROWS * H);
  SharedBuffer af = buf((std::size_t)ROWS * H);
  SharedBuffer g1 = buf((std::size_t)ROWS * FF);
  SharedBuffer u1 = buf((std::size_t)ROWS * FF);
  SharedBuffer s1 = buf((std::size_t)ROWS * FF);
  // A cached step attends over the WHOLE sequence, so its keys and
  // values are the cache spliced in front of what it just computed.
  SharedBuffer kf, vf;
  if (kv_use) {
    kf = buf((std::size_t)JT * H);
    vf = buf((std::size_t)JT * H);
    if (kf.empty() || vf.empty()) {
      return fail("qwen-image-2.1 forward: splice alloc failed");
    }
  }
  SharedBuffer temb = buf((std::size_t)MOD_ROWS * H);
  SharedBuffer tsil = buf((std::size_t)MOD_ROWS * H);
  SharedBuffer modb = buf((std::size_t)MOD_ROWS * 4 * H);
  SharedBuffer nsc = buf((std::size_t)MOD_ROWS * H);
  SharedBuffer tin = buf((std::size_t)MOD_ROWS * _cfg.time_proj);
  // Biasless throughout, so every GEMM below binds this: one zero row is
  // cheaper than a second kernel that differs only in not adding it.
  SharedBuffer zero = buf((std::size_t)std::max(4 * H, FF));
  SharedBuffer rcos, rsin;
  build_rope_(lay, &rcos, &rsin);
  if (jh.empty() || tmp.empty() || g1.empty() || zero.empty() ||
      rcos.empty() || rsin.empty()) {
    return fail("qwen-image-2.1 forward: scratch allocation failed");
  }
  std::memset(zero.contents(), 0, zero.byte_size());
  {
    auto* d = static_cast<std::uint16_t*>(tin.contents());
    std::memset(d, 0, tin.byte_size());
    for (int r = 0; r < MOD_ROWS; ++r) {
      // Row 0 is the sampled timestep; row 1, when causal_condition is
      // on, is t = 0 -- the row text and condition-image tokens read.
      const std::vector<float> tp = time_proj_(r == 0 ? req.timestep : 0.0f);
      for (int i = 0; i < _cfg.time_proj; ++i) {
        d[(std::size_t)r * _cfg.time_proj + i] =
            f32_to_bf16_(tp[(std::size_t)i]);
      }
    }
  }
  // One attention params block PER SEGMENT. They are encoded into one
  // stream, so a shared mutable buffer would hand the last segment's
  // shape to every earlier dispatch.
  std::vector<SharedBuffer> aparams;
  aparams.reserve(lay.segments.size());
  for (std::size_t i = 0; i < lay.segments.size(); ++i) {
    aparams.push_back(mc->make_shared_buffer(sizeof(SteelAttnParams)));
    if (aparams.back().empty()) {
      return fail("qwen-image-2.1 forward: attention params alloc failed");
    }
  }

  CommandStream stream = mc->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  if (!enc.valid()) {
    return fail("qwen-image-2.1 forward: no compute encoder");
  }

  // Per-stage TRACE, for locating a numeric gap against the reference.
  // VPIPE_QWEN_IMAGE21_TRACE names a directory; each call commits what
  // is encoded so far and writes the buffer as raw f32. Off by default
  // and free when unset -- but it COMMITS, so a traced run is not a
  // timing run.
  const char* trace_dir = std::getenv("VPIPE_QWEN_IMAGE21_TRACE");
  bool trace_ok = true;
  auto trace = [&](const char* tag, const SharedBuffer& b, std::size_t n) {
    if (trace_dir == nullptr || *trace_dir == '\0' || !trace_ok) { return; }
    enc.end();
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
    stream = mc->make_command_stream();
    enc = stream.begin_compute();
  };

  // Per-CATEGORY profile. VPIPE_QWEN_IMAGE21_DIT_PROFILE=1 commits at each
  // boundary and charges the elapsed time to a bucket, so the split
  // between GEMM, attention and the elementwise tail is measured rather
  // than reasoned about. It COMMITS ~10 times per block, so a profiled
  // run is not a timing run -- read the shares, not the total.
  const bool prof = [] {
    const char* e = std::getenv("VPIPE_QWEN_IMAGE21_DIT_PROFILE");
    return e != nullptr && *e != '\0' && std::atoi(e) != 0;
  }();
  std::vector<std::pair<const char*, double>> pbuckets;
  auto pclock = std::chrono::steady_clock::now();
  auto mark = [&](const char* bucket) {
    if (!prof) { return; }
    enc.end();
    std::string ge;
    (void)stream.commit().wait_ok(&ge);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pclock).count();
    bool found = false;
    for (auto& b : pbuckets) {
      if (std::strcmp(b.first, bucket) == 0) { b.second += ms; found = true; }
    }
    if (!found) { pbuckets.emplace_back(bucket, ms); }
    stream = mc->make_command_stream();
    enc = stream.begin_compute();
    pclock = std::chrono::steady_clock::now();
  };

  // ---- dispatch helpers ---------------------------------------------
  auto lin = [&](const SharedBuffer& x, std::size_t xe, const QWeight& w,
                 const SharedBuffer& y, std::size_t ye, int M, int N, int K) {
    // Matrix cores first; a decline encodes nothing and falls through.
    if (gemm_mma_(enc, x, xe, w, y, ye, M, N, K)) { return; }
    if (!w.quantized) {
      // The tile, which is the whole of this path's performance on a box
      // with no matrix cores.
      int bm = 32, bn = 32;
      const metal_compute::ComputeFunction* fn = &_fn_gemm_t;
      if (_gemm_tile == 1) { fn = &_fn_gemm_bm64; bm = 64; }
      else if (_gemm_tile == 2) {
        fn = &_fn_gemm_bm64bn64; bm = 64; bn = 64;
      }
      enc.set_function(*fn);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w.w);
      enc.set_buffer(2, zero); enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
      enc.set_constant(7, 0);
      // THREADGROUP-indexed: x = ceil(N/BN)*32, y = ceil(M/BM)*2, z = 2
      // over a {32,2,2} group. A thread-indexed grid here computes a
      // fraction of the output and leaves the rest as it found it.
      enc.dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                    (unsigned)(((M + bm - 1) / bm) * 2), 2}, {32, 2, 2});
      return;
    }
    enc.set_function(w.bits == 8 ? _fn_qmm8 : _fn_qmm4);
    enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
    enc.set_buffer(2, w.qbias); enc.set_buffer(3, x, xe * 2);
    enc.set_buffer(4, y, ye * 2);
    enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                  (unsigned)(((M + 31) / 32) * 2), 2u}, {32, 2, 2});
  };
  auto rms = [&](const SharedBuffer& x, std::size_t xe, const SharedBuffer& w,
                 const SharedBuffer& y, std::size_t ye, int R, int D) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w);
    enc.set_buffer(2, y, ye * 2);
    enc.set_constant(3, D); enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  auto mul_sig = [&](const SharedBuffer& x, const SharedBuffer& y, int n) {
    enc.set_function(_fn_silu);
    enc.set_buffer(0, x); enc.set_buffer(1, x); enc.set_buffer(2, y);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };
  // LN(x) * (1 + scale), with NO shift: this model's modulation emits
  // none, so the fused kernel's shift slot takes the zero row.
  auto ln_mod = [&](const SharedBuffer& x, std::size_t xe,
                    const SharedBuffer& scale, std::size_t se,
                    const SharedBuffer& y, std::size_t ye, int R) {
    enc.set_function(_fn_ln_mod);
    enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, scale, se * 2);
    enc.set_buffer(2, zero); enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, H); enc.set_constant(5, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  auto gated = [&](const SharedBuffer& h, std::size_t he,
                   const SharedBuffer& gate, std::size_t ge,
                   const SharedBuffer& sub, std::size_t se, int R) {
    // Four elements a thread where the width allows, which for this
    // model is always: the gate is ONE H-vector broadcast down the
    // rows, so the wide kernel's 2-D grid indexes it by x alone.
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
  auto copy_rows = [&](const SharedBuffer& src, std::size_t so,
                       const SharedBuffer& dst, std::size_t do_, int rows,
                       int row_elems, int sstride, int dstride) {
    enc.set_function(_fn_copy_rows);
    enc.set_buffer(0, src); enc.set_buffer(1, dst);
    enc.set_constant(2, (int)so); enc.set_constant(3, (int)do_);
    enc.set_constant(4, rows); enc.set_constant(5, row_elems);
    enc.set_constant(6, sstride); enc.set_constant(7, dstride);
    // ONE-DIMENSIONAL. The kernel takes a flat
    // thread_position_in_grid over rows*row_elems; a 2-D grid leaves it
    // reading only the x extent, which copies the first row and leaves
    // every other destination row as it found it.
    enc.dispatch({(unsigned)((std::size_t)rows * row_elems), 1, 1},
                 {256, 1, 1});
  };

  // ---- timestep, and the shared modulation --------------------------
  lin(tin, 0, _time_1, temb, 0, MOD_ROWS, H, _cfg.time_proj);
  mul_sig(temb, tsil, MOD_ROWS * H);
  lin(tsil, 0, _time_2, temb, 0, MOD_ROWS, H, H);
  // modulation is Sequential(SiLU, Linear): the SiLU is on temb, and the
  // SAME silu'd temb feeds norm_out's scale at the end.
  mul_sig(temb, tsil, MOD_ROWS * H);
  lin(tsil, 0, _mod, modb, 0, MOD_ROWS, 4 * H, H);
  lin(tsil, 0, _norm_out, nsc, 0, MOD_ROWS, H, H);

  // Row ranges. Under causal_condition the prefix reads the t=0 row and
  // the target its own; otherwise everything reads row 0. Contiguous by
  // construction, so this is two dispatches rather than a per-token
  // index -- see the layout header.
  struct Band { int start, rows, mod_row; };
  std::vector<Band> bands;
  if (kv_use) {
    // Every row this step runs is a TARGET row, so they all read the
    // sampled timestep -- the t=0 rows are the ones it is not running.
    bands.push_back(Band{0, ROWS, 0});
  } else if (_cfg.causal_condition && lay.prefix_len > 0 &&
             lay.prefix_len < JT) {
    bands.push_back(Band{0, lay.prefix_len, 1});
    bands.push_back(Band{lay.prefix_len, JT - lay.prefix_len, 0});
  } else {
    bands.push_back(Band{0, JT, _cfg.causal_condition ? 1 : 0});
    if (_cfg.causal_condition && lay.prefix_len == 0) {
      bands[0].mod_row = 0;
    }
  }

  // ---- the joint stream ---------------------------------------------
  //
  // Image rows take a latent through img_in; text rows take the
  // encoder's hidden state through txt_in. Both land by CONTIGUOUS RUNS
  // -- the layout's segments are exactly those runs -- so this is a
  // handful of row copies rather than a gather.
  if (kv_use) {
    // The target block's latents are the TAIL of the packed input, and
    // nothing else in the sequence is run.
    lin(*req.latents,
        (std::size_t)(IMG - lay.target_len) * _cfg.in_channels, _img_in,
        jh, 0, ROWS, H, _cfg.in_channels);
  } else {
    SharedBuffer ih = buf((std::size_t)IMG * H);
    if (ih.empty()) { return fail("qwen-image-2.1 forward: alloc failed"); }
    lin(*req.latents, 0, _img_in, ih, 0, IMG, H, _cfg.in_channels);
    // txt_in: zero-centered RMS norm (the +1 is folded into the weight
    // at load), in_layer, GELU(tanh), out_layer.
    const int ENC = (int)lay.text_src.size() + 0;
    (void)ENC;
    const int enc_rows = (int)(req.txt->byte_size() /
                               ((std::size_t)_cfg.txt_dim * 2));
    SharedBuffer tn = buf((std::size_t)enc_rows * _cfg.txt_dim);
    SharedBuffer th = buf((std::size_t)enc_rows * H);
    SharedBuffer tg = buf((std::size_t)enc_rows * H);
    if (tn.empty() || th.empty() || tg.empty()) {
      return fail("qwen-image-2.1 forward: alloc failed");
    }
    rms(*req.txt, 0, _txt_norm, tn, 0, enc_rows, _cfg.txt_dim);
    lin(tn, 0, _txt_in_a, th, 0, enc_rows, H, _cfg.txt_dim);
    enc.set_function(_fn_gelu_tanh);
    enc.set_buffer(0, th); enc.set_buffer(1, tg);
    enc.set_constant(2, (int)((std::size_t)enc_rows * H));
    enc.dispatch({(unsigned)((std::size_t)enc_rows * H), 1, 1}, {256, 1, 1});
    lin(tg, 0, _txt_in_b, th, 0, enc_rows, H, H);

    int img_at = 0, txt_at = 0;
    for (const qi21::Segment& sg : lay.segments) {
      const int rows = sg.end - sg.start;
      if (sg.is_text) {
        copy_rows(th, (std::size_t)lay.text_src[(std::size_t)txt_at] * H,
                  jh, (std::size_t)sg.start * H, rows, H, H, H);
        txt_at += rows;
      } else {
        copy_rows(ih, (std::size_t)img_at * H, jh, (std::size_t)sg.start * H,
                  rows, H, H, H);
        img_at += rows;
      }
    }
  }

  trace("jh_in", jh, (std::size_t)ROWS * H);
  trace("mod", modb, (std::size_t)MOD_ROWS * 4 * H);
  trace("temb", temb, (std::size_t)MOD_ROWS * H);

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
  if (_stream_blocks) { _slots.begin_forward(); }

  // ---- how much of the stack this box can KEEP ------------------------
  //
  // Re-arm growth for this forward (a ratchet from an earlier eviction
  // deliberately survives), then take the measurement that actually
  // finds the limit: are the blocks already kept still in RAM? Free-
  // memory arithmetic cannot answer that on a streaming model -- the
  // file cache it would read is mostly this checkpoint, so the signal is
  // self-generated. Gated on our OWN compressed footprint moving,
  // because the page walk is not free and a healthy run would pay it
  // every step to be told nothing.
  if (_stream_blocks) {
    if (_wire.on()) {
      // Retry a pool that refused earlier: the refusal may have been
      // another process spiking, and a run that never asks again holds a
      // small resident set for the whole schedule on one syscall.
      if (_wire.retry(_mc, wire_retry_slack_())) {
        _resid.note_landscape_changed();
      }
      // The trunk takes its place BEFORE this forward's admissions start
      // asking: the blocks are the shed-able half, so a pool that runs
      // out should run out on them.
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
              "MetalQwenImage21Transformer: resident weights are only {}% "
              "in RAM ({} of {} sampled pages paged out, {} MB wired) -- "
              "released {} MB, now {} blocks resident",
              (int)(100.0 * (double)incore / (double)examined), paged_out,
              examined, _wire.wired_bytes() >> 20, freed >> 20,
              _resid.count()));
        }
      }
    }
    // Nothing of ours had left RAM this step. Enough in a row lifts the
    // ratchet by a block, so a shed taken during a momentary squeeze is
    // not the last word on the run.
    if (!shortfall) { _resid.note_healthy_forward(); }
  }

  const bool nax = _use_attn_nax;
  const int A_BQ = nax ? 64 : 32;
  const int A_BK = nax ? 32 : 16;
  const float ascale = 1.0f / std::sqrt((float)Hd);

  for (int L = 0; L < _cfg.n_layers; ++L) {
    // COOPERATIVE STOP, checked every block rather than only on the
    // streamed tail, so a slow high-resolution step answers within
    // roughly one block on both paths. It returns EMPTY WITHOUT AN
    // ERROR, because a stop is not a failure: the caller distinguishes
    // the two by whether a reason came back, and reporting one here
    // made an ordinary user stop log as "DiT forward failed".
    //
    // Everything that needs unwinding does so on its own: SlotJoin
    // waits out any read still landing in a slot, and the KV cache is
    // only STAMPED valid after a forward that completed, so a cache
    // half-filled by an abandoned step reads as unpopulated rather than
    // as a prefix the next step would trust.
    if (_stream_stop && _stream_stop()) { return SharedBuffer{}; }
    // RESIDENT once residency has promoted it, and read into a slot
    // until then. `_blocks` is sized to n_layers in streaming mode and
    // an unfilled entry reads as empty, so this one test covers both.
    const bool held = L < (int)_blocks.size() &&
                      !_blocks[(std::size_t)L].qw.empty();
    const bool streaming = _stream_blocks && !held;
    const Block* b = nullptr;
    if (!streaming) {
      b = &_blocks[(std::size_t)L];
    } else {
      b = _slots.acquire(L);
      if (b == nullptr) {
        return fail("qwen-image-2.1 forward: a streamed block did not load");
      }
      // The next block's read is NOT issued here. It goes between this
      // block's commit and its wait, at the bottom of the loop -- see
      // there for why that is a correctness rule and not a tuning
      // choice.
    }
    // THE ANE'S PLAN FOR THIS BLOCK, decided before anything is encoded
    // because the staging has to ride under the attention. kProbe runs
    // the GPU over every row and times both sides, which is how the
    // tier learns whether the split pays on this box and geometry.
    AneFeedForward::Plan ane_plan = AneFeedForward::Plan::kGpu;
    if (_cfg.ane_ffn && ane_setup_(ROWS) && ane_eligible_(L, *b)) {
      ane_plan = _ane->plan_block();
    }
    const bool ane_split = ane_plan == AneFeedForward::Plan::kSplit;
    const bool ane_probe = ane_plan == AneFeedForward::Plan::kProbe;
    // The q/k/v tier's plan, decided the same way and independently: a
    // box and geometry where one pays and the other does not keeps the
    // one that does.
    AneFeedForward::Plan qkv_plan = AneFeedForward::Plan::kGpu;
    if (_cfg.ane_qkv && ane_qkv_setup_(ROWS) && ane_qkv_eligible_(L, *b)) {
      qkv_plan = _ane_qkv->plan_block();
    }
    const bool qkv_split = qkv_plan == AneFeedForward::Plan::kSplit;
    const bool qkv_probe = qkv_plan == AneFeedForward::Plan::kProbe;
    if (((ane_split || ane_probe) && (L == 0 || _ane->needs_barrier())) ||
        ((qkv_split || qkv_probe) &&
         (L == 0 || _ane_qkv->needs_barrier()))) {
      // An unsplit block commits nothing mid-block, so on a stack that
      // defers its commits the split point's drain would wait for every
      // such block queued before it. Drain what is queued FIRST.
      enc.end();
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        return fail("qwen-image-2.1 forward: " + ge2);
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
    }
    // ONE ANE worker serves both tiers, and it takes one job at a time: a
    // second stage() would block this thread until the first finished.
    // So q/k/v stage first -- they are needed first -- and the feed-
    // forward's weights follow their join, still ahead of the attention
    // they hide under.
    if (qkv_split) {
      ane_qkv_stage_(L, *b);
    } else if (ane_split) {
      ane_stage_(L, *b);
    }

    if (_block_progress) {
      // `done` of `total`, so L -- the blocks finished BEFORE this one.
      // Every other DiT here reports L, and the helper's contract wants
      // a report that lags the truth and never leads it. L+1 read as
      // exact only because this stack commits per block while
      // streaming; on the preloaded path, where one buffer spans
      // several blocks, it was a claim made before the work was
      // encoded.
      report_block(stream, _block_progress, L, _cfg.n_layers);
    }

    // --- attention half ---
    for (const Band& bd : bands) {
      ln_mod(jh, (std::size_t)bd.start * H,
             modb, (std::size_t)bd.mod_row * 4 * H + 0,
             tmp, (std::size_t)bd.start * H, bd.rows);
    }
    mark("elt");
    // ---- the q/k/v split point --------------------------------------
    //
    // The feed-forward's choreography, one sublayer earlier: `tmp` holds
    // the modulated attention input for every row, so drain, START THE
    // ANE on the tail rows -- its output columns land in qb / kb / vb at
    // the rows they came from -- and only then encode the GPU's head
    // rows. The attention reads all of q/k/v, so the join is before it.
    int q_rows = 0;
    double q_drain_ms = 0.0;
    if (qkv_split || qkv_probe) {
      enc.end();
      const auto t_d0 = std::chrono::steady_clock::now();
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        return fail("qwen-image-2.1 forward: " + ge2);
      }
      q_drain_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_d0).count();
      if (qkv_split && _ane_qkv->join_stage(L)) {
        q_rows = _ane_qkv->begin(
            tmp,
            std::vector<AneFeedForward::OutSeg>{
                {&qb, H, 0}, {&kb, H, 0}, {&vb, H, 0}},
            ROWS);
      } else if (qkv_split && _mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "qwen-image-2.1: staging block {}'s q/k/v for the ANE failed; "
            "it keeps the GPU projections", L));
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
    }
    const int qg_rows = ROWS - q_rows;
    const auto t_q0 = std::chrono::steady_clock::now();
    if (qg_rows > 0) {
      lin(tmp, 0, b->qw, qb, 0, qg_rows, H, H);
      lin(tmp, 0, b->kw, kb, 0, qg_rows, H, H);
      lin(tmp, 0, b->vw, vb, 0, qg_rows, H, H);
    }
    mark("gemm.qkv");
    if (qkv_split || qkv_probe) {
      enc.end();
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        if (q_rows > 0) {
          (void)_ane_qkv->finish(L, ROWS, 0.0, q_drain_ms);
        }
        return fail("qwen-image-2.1 forward: " + ge2);
      }
      const double t_gpu = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_q0).count();
      if (qkv_probe) { _ane_qkv->note_probe(q_drain_ms, t_gpu); }
      if (!_ane_qkv->finish(L, ROWS, t_gpu, q_drain_ms) && q_rows > 0) {
        return fail("qwen-image-2.1 forward: the ANE left its rows of a "
                    "q/k/v projection uncomputed");
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
    }
    // The worker is free again: the feed-forward's weights now, so they
    // stage under the attention exactly as they would alone.
    if (qkv_split && ane_split) { ane_stage_(L, *b); }
    // Per-head RMSNorm, then the fused transpose+rope. Token-major
    // [JT, NH, Hd] normalizes over the last Hd with JT*NH rows.
    rms(qb, 0, b->nq, qb, 0, ROWS * NH, Hd);
    rms(kb, 0, b->nk, kb, 0, ROWS * NH, Hd);
    // The rotary table is bound at this step's FIRST GLOBAL ROW, so a
    // cached step rotates its target rows with the angles they would
    // have had in the whole sequence. f32 table, hence *4.
    const std::size_t rope_off = (std::size_t)ROW0 * Hd * 4;
    auto tr_rope = [&](const SharedBuffer& in, const SharedBuffer& out) {
      enc.set_function(_fn_tr_rope);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_buffer(2, rcos, rope_off); enc.set_buffer(3, rsin, rope_off);
      enc.set_constant(4, NH); enc.set_constant(5, ROWS);
      enc.set_constant(6, Hd);
      enc.dispatch({(unsigned)(Hd / 2), (unsigned)ROWS, (unsigned)NH},
                   {(unsigned)(Hd / 2), 1, 1});
    };
    tr_rope(qb, qh);
    tr_rope(kb, kh);
    // V is transposed but NOT rotated.
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, vb); enc.set_buffer(1, vh);
    enc.set_constant(2, ROWS); enc.set_constant(3, NH);
    enc.set_constant(4, Hd);
    enc.dispatch({(unsigned)Hd, (unsigned)NH, (unsigned)ROWS},
                 {(unsigned)Hd, 1, 1});

    // ---- the cache -------------------------------------------------
    //
    // STORED POST-ROPE, which is what the reference does: its store
    // happens after apply_rotary_emb_qwen, and a cached step then
    // concatenates the already-rotated band in front of rows it rotated
    // with the SLICED table. (flux2's klein_kv caches PRE-rope instead
    // and re-rotates the whole concatenation; both are self-consistent,
    // and mixing them is not.)
    const std::size_t pf_elems = (std::size_t)lay.prefix_len * Hd;
    if (kv_fill) {
      KvCache& c = *req.kv;
      if ((int)c.k.size() != _cfg.n_layers) {
        // resize, not assign: SharedBuffer is move-only.
        c.k.clear(); c.v.clear();
        c.k.resize((std::size_t)_cfg.n_layers);
        c.v.resize((std::size_t)_cfg.n_layers);
      }
      auto& ck = c.k[(std::size_t)L];
      auto& cv = c.v[(std::size_t)L];
      if (ck.byte_size() < (std::size_t)NH * pf_elems * 2) {
        ck = buf((std::size_t)NH * pf_elems);
        cv = buf((std::size_t)NH * pf_elems);
      }
      if (ck.empty() || cv.empty()) {
        c.clear();
        return fail("qwen-image-2.1 forward: the KV cache did not fit");
      }
      // Head-major, so the prefix band is one contiguous run PER HEAD.
      copy_rows(kh, 0, ck, 0, NH, (int)pf_elems, ROWS * Hd, (int)pf_elems);
      copy_rows(vh, 0, cv, 0, NH, (int)pf_elems, ROWS * Hd, (int)pf_elems);
    }
    if (kv_use) {
      const KvCache& c = *req.kv;
      copy_rows(c.k[(std::size_t)L], 0, kf, 0, NH, (int)pf_elems,
                (int)pf_elems, JT * Hd);
      copy_rows(c.v[(std::size_t)L], 0, vf, 0, NH, (int)pf_elems,
                (int)pf_elems, JT * Hd);
      copy_rows(kh, 0, kf, pf_elems, NH, (int)((std::size_t)ROWS * Hd),
                ROWS * Hd, JT * Hd);
      copy_rows(vh, 0, vf, pf_elems, NH, (int)((std::size_t)ROWS * Hd),
                ROWS * Hd, JT * Hd);
    }

    mark("rope+kv");
    // BLOCK-CAUSAL, as one dispatch per segment. An image block's rows
    // see [0, end) with no mask at all (everything before it by
    // causality, all of itself by bidirectionality); a text run is
    // causal against [0, end) with qL_off carrying its global row base.
    // A cached step has ONE group: every target row sees the entire
    // prefix plus its own block, so the block-causal structure
    // degenerates to dense attention and the segments are not needed.
    struct Group { int qrow, qL, kL, koff; bool causal; };
    std::vector<Group> groups;
    if (kv_use) {
      groups.push_back(Group{0, ROWS, JT, 0, false});
    } else {
      for (const qi21::Segment& sg : lay.segments) {
        groups.push_back(Group{sg.start, sg.end - sg.start, sg.end,
                               sg.is_text ? sg.start : 0, sg.is_text});
      }
    }
    const SharedBuffer& kuse = kv_use ? kf : kh;
    const SharedBuffer& vuse = kv_use ? vf : vh;
    const int kstride = kv_use ? JT : ROWS;
    if (aparams.size() < groups.size()) {
      return fail("qwen-image-2.1 forward: too few attention params");
    }
    for (std::size_t si = 0; si < groups.size(); ++si) {
      const Group& sg = groups[si];
      const int qL = sg.qL;
      const int kL = sg.kL;
      auto* p = static_cast<SteelAttnParams*>(aparams[si].contents());
      p->B = 1; p->H = NH; p->D = Hd; p->qL = qL; p->kL = kL;
      p->gqa_factor = 1; p->scale = ascale;
      p->NQ = (qL + A_BQ - 1) / A_BQ; p->NK = (kL + A_BK - 1) / A_BK;
      p->NQ_aligned = qL / A_BQ; p->NK_aligned = kL / A_BK;
      p->qL_rem = qL - p->NQ_aligned * A_BQ;
      p->kL_rem = kL - p->NK_aligned * A_BK;
      // The causal kernel measures a row against its GLOBAL position.
      p->qL_off = sg.koff;
      // Q and O stride over the rows THIS step runs; K and V over the
      // rows they span, which differ on a cached step.
      const std::int64_t qs = (std::int64_t)ROWS * Hd;
      const std::int64_t ks = (std::int64_t)kstride * Hd;
      p->Q_strides[0] = (std::int64_t)NH * qs;
      p->Q_strides[1] = qs; p->Q_strides[2] = Hd;
      p->K_strides[0] = (std::int64_t)NH * ks;
      p->K_strides[1] = ks; p->K_strides[2] = Hd;
      p->V_strides[0] = p->K_strides[0];
      p->V_strides[1] = ks; p->V_strides[2] = Hd;
      p->O_strides[0] = p->Q_strides[0];
      p->O_strides[1] = qs; p->O_strides[2] = Hd;

      // SOL, for a segment big enough to have something to skip.
      //
      // NON-CAUSAL ONLY: Sol's exact half is the steel kernel under a
      // CSR of kept blocks, built without the causal predicate, so it
      // answers an image block's "everything up to and including me"
      // but not a text run's per-row triangle. The text runs are dozens
      // of rows against an image block's thousands, so what it declines
      // is the cheap part.
      //
      // THE TWO OFFSETS DIFFER HERE and that is not hypothetical: on a
      // cached step `qh` holds only the target rows while those rows
      // sit at ROW0 in the sequence, so the buffer row and the sequence
      // position are different numbers and the local band wants the
      // second.
      if (_sol && nax && !sg.causal && _cfg.sol.enabled &&
          L >= _cfg.sol.dense_layers && qL >= kSolMinSegment) {
        const int blk = _cfg.sol.key_block > 0 ? _cfg.sol.key_block
                                               : sol::kBlock;
        if ((kL + blk - 1) / blk >= 16) {
          MetalSolAttention::Band band{qL, kL, ROWS, kstride,
                                       (int)ROW0 + sg.qrow, sg.qrow};
          std::string serr;
          if (_sol->encode(enc, qh, kuse, vuse, ao, NH, NH, band, Hd,
                           ascale, _cfg.sol, &serr)) {
            continue;
          }
          if (_mc->session() != nullptr) {
            _mc->session()->log_debug(fmt(
                "qwen-image-2.1: sol declined this segment ({}); dense",
                serr.empty() ? "no reason" : serr));
          }
        }
      }

      // SAGE, PER SEGMENT rather than per block, which is what this
      // model's layout forces. The kernel indexes Q8/K8 by params->qL
      // and params->kL, so the quantized operands belong to ONE
      // segment's geometry; the single-group families prepare once per
      // block and cannot be copied here.
      //
      // Skipped for a segment too small to pay for its own quantization
      // pass -- the text runs are dozens of rows against an image
      // block's thousands, and quantizing them costs more than the
      // dense dispatch they replace. In a text-to-image layout exactly
      // one segment qualifies, which is also what keeps the scratch
      // from being rebuilt inside a block.
      bool i8_ok = false;
      if (_sage && nax && _cfg.sage.enabled &&
          L >= _cfg.sage.dense_layers && qL >= kSageMinSegment) {
        const MetalSageAttention::Operand qo{
            &qh, (std::size_t)sg.qrow * Hd, Hd, ROWS * Hd};
        const MetalSageAttention::Operand ko{&kuse, 0, Hd, kstride * Hd};
        std::string gerr;
        i8_ok = _sage->prepare(enc, qo, ko, NH, NH, qL, kL, Hd, A_BQ, A_BK,
                               _cfg.sage, &gerr);
      }
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (qL % A_BQ) == 0).set_bool(201, (kL % A_BK) == 0)
          .set_bool(300, false).set_bool(301, sg.causal).set_bool(302, false)
          .set_bool(sage::kQkInt8Constant, i8_ok);
      metal_compute::ComputeFunction fn =
          nax ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
              : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
      if (!fn.valid()) {
        return fail("qwen-image-2.1 forward: the steel attention kernel "
                    "did not specialize");
      }
      enc.set_function(fn);
      // Q and O are bound at the segment's row offset; K and V at zero,
      // because a segment's keys always start at the beginning.
      enc.set_buffer(0, qh, (std::size_t)sg.qrow * Hd * 2);
      enc.set_buffer(1, kuse);
      enc.set_buffer(2, vuse);
      enc.set_buffer(3, ao, (std::size_t)sg.qrow * Hd * 2);
      enc.set_buffer(4, aparams[si]);
      if (i8_ok) { _sage->bind(enc); }
      enc.dispatch({(unsigned)(32 * ((qL + A_BQ - 1) / A_BQ)),
                    (unsigned)(4 * NH), 1}, {32, 4, 1});
    }
    mark("attn");
    // Back to token-major, then the output projection.
    enc.set_function(_fn_transpose);
    enc.set_buffer(0, ao); enc.set_buffer(1, af);
    enc.set_constant(2, NH); enc.set_constant(3, ROWS);
    enc.set_constant(4, Hd);
    enc.dispatch({(unsigned)Hd, (unsigned)ROWS, (unsigned)NH},
                 {(unsigned)Hd, 1, 1});
    mark("transpose");
    lin(af, 0, b->ow, tmp, 0, ROWS, H, H);
    mark("gemm.out");
    for (const Band& bd : bands) {
      gated(jh, (std::size_t)bd.start * H,
            modb, (std::size_t)bd.mod_row * 4 * H + H,
            tmp, (std::size_t)bd.start * H, bd.rows);
    }

    // --- feed-forward half ---
    for (const Band& bd : bands) {
      ln_mod(jh, (std::size_t)bd.start * H,
             modb, (std::size_t)bd.mod_row * 4 * H + 2 * H,
             tmp, (std::size_t)bd.start * H, bd.rows);
    }
    mark("elt");
    // ---- the ANE split point ----------------------------------------
    //
    // `tmp` now holds the modulated feed-forward input for every row.
    // Commit the attention WITHOUT waiting, drain, and start the ANE on
    // the tail rows BEFORE any of the GPU's feed-forward is encoded --
    // encoding gate/up first would put them inside the drained section,
    // where they overlap nothing.
    int a_rows = 0;
    double ane_drain_ms = 0.0;
    if (ane_split || ane_probe) {
      enc.end();
      const auto t_d0 = std::chrono::steady_clock::now();
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        return fail("qwen-image-2.1 forward: " + ge2);
      }
      ane_drain_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_d0).count();
      if (ane_split && _ane->join_stage(L)) {
        a_rows = _ane->begin(tmp, tmp, ROWS);
      } else if (ane_split && _mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "qwen-image-2.1: staging block {}'s weights for the ANE "
            "failed; it keeps the GPU feed-forward", L));
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
    }
    // The GPU's share. A probe block deliberately takes every row: that
    // is what makes its time comparable with the split's.
    const int g_rows = ROWS - a_rows;
    const auto t_g0 = std::chrono::steady_clock::now();
    if (g_rows > 0) {
    lin(tmp, 0, b->gate_w, g1, 0, g_rows, FF, H);
    lin(tmp, 0, b->proj_w, u1, 0, g_rows, FF, H);
    }
    mark("gemm.ff_up");
    // out(silu(gate_layer(x)) * proj(x)). GATE FIRST: `gate_layer` is the
    // silu'd half and `proj` the multiplicand, which the names do not
    // make obvious and which no shape check can tell apart.
    // The largest elementwise dispatch in the model -- ROWS*FF, which
    // at 1024^2 is 51M elements -- so it is the one that pays most for
    // being four-wide.
    const std::size_t sw_n = (std::size_t)g_rows * FF;
    if (g_rows > 0 && (sw_n % 4) == 0 && _fn_swiglu4.valid()) {
      enc.set_function(_fn_swiglu4);
      enc.set_buffer(0, g1); enc.set_buffer(1, u1); enc.set_buffer(2, s1);
      enc.set_constant(3, (int)(sw_n / 4));
      enc.dispatch({(unsigned)(sw_n / 4), 1, 1}, {256, 1, 1});
    } else if (g_rows > 0) {
      enc.set_function(_fn_swiglu);
      enc.set_buffer(0, g1); enc.set_buffer(1, u1); enc.set_buffer(2, s1);
      enc.set_constant(3, (int)sw_n);
      enc.dispatch({(unsigned)sw_n, 1, 1}, {256, 1, 1});
    }
    mark("elt");
    if (g_rows > 0) {
      lin(s1, 0, b->out_w, tmp, 0, g_rows, H, FF);
    }
    mark("gemm.ff_down");
    // ---- rejoin ------------------------------------------------------
    //
    // The GPU's rows have to be DONE before the ANE's are folded in and
    // before the gated residual reads the whole tensor, so this commit
    // is the join and its time is the GPU's share for the balancer.
    if (ane_split || ane_probe) {
      enc.end();
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        if (a_rows > 0) { (void)_ane->finish(L, ROWS, 0.0, ane_drain_ms); }
        return fail("qwen-image-2.1 forward: " + ge2);
      }
      const double t_gpu = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_g0).count();
      if (ane_probe) { _ane->note_probe(ane_drain_ms, t_gpu); }
      if (!_ane->finish(L, ROWS, t_gpu, ane_drain_ms) && a_rows > 0) {
        return fail("qwen-image-2.1 forward: the ANE left its rows of a "
                    "feed-forward uncomputed");
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
    }
    for (const Band& bd : bands) {
      gated(jh, (std::size_t)bd.start * H,
            modb, (std::size_t)bd.mod_row * 4 * H + 3 * H,
            tmp, (std::size_t)bd.start * H, bd.rows);
    }
    // ---- the streaming boundary --------------------------------------
    //
    // A preloaded run encodes all 32 blocks into ONE stream and commits
    // once, which is both faster and simpler. A STREAMED one cannot: the
    // two slots are reused block after block, so every block encoded but
    // not yet run still points at bytes the next read would overwrite.
    // The commit is therefore a correctness boundary here, not a flush,
    // and it buys the window the prefetch needs.
    if (streaming) {
      enc.end();
      std::string ge;
      {
        CommandStream::Fence bf = stream.commit();
        // BETWEEN THE COMMIT AND THE WAIT the GPU is busy with this
        // block and this thread has nothing to do. That window is where
        // the next block's read goes -- into the OTHER slot, which is
        // the only destination this block is not using.
        //
        // Skip what residency already kept: those blocks are served
        // from `_blocks` and never acquired, and prefetching one would
        // leave a read outstanding that no acquire ever collects.
        int nxt = -1;
        for (int n = L + 1; n < _cfg.n_layers; ++n) {
          const bool h = n < (int)_blocks.size() &&
                         !_blocks[(std::size_t)n].qw.empty();
          if (!h) { nxt = n; break; }
        }
        _slots.prefetch(nxt);
        if (!bf.wait_ok(&ge)) {
          if (kv_fill && req.kv != nullptr) { req.kv->clear(); }
          return fail("qwen-image-2.1 forward: " + ge);
        }
      }
      // ---- keep this block, if the box can hold it -------------------
      //
      // AFTER the wait, so nothing encoded still points at the slot, and
      // promotion CLONES out of it so the slot stays usable for the read
      // already in flight. Past the wire budget there is nothing to
      // gain: the block would be kept UNPROTECTED, the compressor would
      // take it (it is the coldest memory in the process), and the next
      // residency walk would shed a block and ratchet the ceiling over
      // the whole resident set. Better not to hold it at all.
      const std::size_t nb = _slots.last_bytes();
      if (nb > 0 && _wire.wirable(nb) && _resid.admit(_mc, nb) &&
          _slots.promote_into(_blocks[(std::size_t)L])) {
        // Wired LAST, after every write this block will ever get: mlock
        // pins the pages that exist NOW.
        _wire.note_wired(_mc, wire_block_(_blocks[(std::size_t)L], true),
                         nb);
        _resid.note_admitted(nb);
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
      if (!enc.valid()) {
        return fail("qwen-image-2.1 forward: no compute encoder");
      }
    } else if (_stream_stop && kStopDrainBlocks > 0 &&
               ((L + 1) % kStopDrainBlocks) == 0 && L + 1 < _cfg.n_layers) {
      // A HELD BLOCK COMMITS NOTHING, so without this the stop check
      // at the top of the loop can only ever see a request that was
      // ALREADY set -- the loop encodes the whole stack in
      // microseconds and then waits out the entire forward inside one
      // commit. Draining on a cadence is what lets a request arriving
      // mid-forward be seen at all.
      enc.end();
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) {
        if (kv_fill && req.kv != nullptr) { req.kv->clear(); }
        return fail("qwen-image-2.1 forward: " + ge);
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
      if (!enc.valid()) {
        return fail("qwen-image-2.1 forward: no compute encoder");
      }
    }
    trace(("blk" + std::to_string(L)).c_str(), jh, (std::size_t)ROWS * H);
  }

  // ---- head ----------------------------------------------------------
  //
  // norm_out is scale-ONLY: LN(x) * (1 + scale), no shift, which is why
  // its Linear emits H rather than 2H.
  for (const Band& bd : bands) {
    ln_mod(jh, (std::size_t)bd.start * H,
           nsc, (std::size_t)bd.mod_row * H,
           tmp, (std::size_t)bd.start * H, bd.rows);
  }
  SharedBuffer out_all = buf((std::size_t)ROWS * _cfg.out_channels);
  SharedBuffer out = buf((std::size_t)lay.target_len * _cfg.out_channels);
  if (out_all.empty() || out.empty()) {
    return fail("qwen-image-2.1 forward: output alloc failed");
  }
  lin(tmp, 0, _proj_out, out_all, 0, ROWS, _cfg.out_channels, H);
  // The TARGET rows only -- the tail by construction, so one copy
  // rather than a gather.
  //
  // The offset is a LOCAL index into what this step ran, not the global
  // row: an uncached step ran the whole sequence and the target starts
  // at prefix_len, a cached step ran nothing else and it starts at 0.
  // ROW0 is the global base and is exactly the wrong number here.
  const std::size_t tail = kv_use ? 0 : (std::size_t)lay.prefix_len;
  copy_rows(out_all, tail * _cfg.out_channels,
            out, 0, lay.target_len, _cfg.out_channels, _cfg.out_channels,
            _cfg.out_channels);

  mark("head");
  enc.end();
  std::string ge;
  if (!stream.commit().wait_ok(&ge)) {
    if (kv_fill && req.kv != nullptr) { req.kv->clear(); }
    return fail("qwen-image-2.1 forward: " + ge);
  }
  if (prof && _mc->session() != nullptr) {
    double tot = 0.0;
    for (const auto& b : pbuckets) { tot += b.second; }
    std::string line = fmt(
        "qwen-image-2.1 profile: joint {}, {} blocks, {:.0f} ms accounted",
        lay.joint_len, _cfg.n_layers, tot)();
    for (const auto& b : pbuckets) {
      line += fmt("; {} {:.0f} ms ({:.1f}%)", b.first, b.second,
                  tot > 0.0 ? 100.0 * b.second / tot : 0.0)();
    }
    _mc->session()->info(fmt("{}", line));
  }
  // Stamped only HERE, after a forward that completed: a half-filled
  // cache flagged valid is worse than none, because the next step reads
  // it instead of refilling.
  if (kv_fill) {
    req.kv->prefix = lay.prefix_len;
    req.kv->joint = lay.joint_len;
    req.kv->target = lay.target_len;
  }
  return out;
}

// Bytes one block costs, taken from the CHECKPOINT rather than from a
// loaded block: while streaming there is no loaded block to measure, and
// the schedule has to be set before the first forward.
// The GEMM on the matrix cores (M5 and later), which is where this
// model's time is.
//
// MEASURED at 1024^2 (joint 4160, 32 blocks) on the steel path: GEMMs
// are 86.4% of the step -- ff_up 39.7%, ff_down 20.5%, qkv 19.5%,
// out 6.7% -- against 4.4% for attention, which already runs the NAX
// flash kernel. So this one function is the whole optimization, and
// nothing else in the forward is worth touching until it is done.
//
// DEQUANT-ONCE, then dense. A quantized weight is expanded to bf16 in
// one pass and the tiles read that, rather than every tile re-decoding
// the codes; the scratch is shared across a block's GEMMs because they
// are serial and Metal tracks the write-after-read.
bool
MetalQwenImage21Transformer::gemm_mma_(ComputeEncoder& enc,
                                       const SharedBuffer& xin,
                                       std::size_t xe, const QWeight& w,
                                       const SharedBuffer& y, std::size_t ye,
                                       int M, int N, int K) const
{
  // Only where the hardware has matrix cores, where M amortizes the
  // 128-row tile, and where N is not degenerate. The timestep and
  // modulation GEMMs are M=2, which is exactly the shape the tile is
  // wrong for -- they stay on steel and cost nothing either way.
  if (!_use_mma2 || M < _mma_min_m || N < 16) { return false; }
  // And decline a shape whose operands would not survive the tiles'
  // 32-bit addressing: past 2^31 BYTES on ANY operand the mma tiles
  // stop storing, SILENTLY. Declining rather than banding is right
  // here: the steel path the caller falls back to is int64 and
  // therefore correct, and this line is not reachable at any
  // resolution this model can allocate -- K peaks at the ff-down's
  // 12288, which puts the limit at 87381 joint rows against the 4160 a
  // 1024^2 image has. See shared/mma-tile.h.
  if (M > mma_row_band(N, K)) { return false; }

  const SharedBuffer* wdense = nullptr;
  if (w.quantized) {
    const metal_compute::ComputeFunction& dq =
        (w.bits == 8) ? _fn_dequant8 : _fn_dequant4;
    if (!dq.valid()) { return false; }
    const std::size_t need = (std::size_t)N * K * 2;   // bf16
    if (_w_deq.empty() || _w_deq.byte_size() < need) {
      _w_deq = _mc->make_shared_buffer(need);
      if (_w_deq.empty()) { return false; }
    }
    // One thread per packed u32 word: w4 has 8 nibbles to a word and w8
    // four bytes, so K/8 and K/4 respectively.
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
  // N-region pays for itself. This model has exactly two depths: 4096
  // for qkv / out / ff-up, and 12288 for ff-down.
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
  // biasless form.
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
// The feed-forward is row-independent, so the ANE takes the LAST rows of
// a block's feed-forward and the GPU the head rows, concurrently. Exact:
// the two compute the same function over disjoint rows of the same
// tensors, so there is no seam to blend.
//
// The choreography is the family's, and its order is load-bearing --
// stage under the attention, commit, drain, START THE ANE, then encode
// the GPU's rows. Starting it after the GPU's gate/up were encoded
// leaves only the down projection to hide behind.
int
MetalQwenImage21Transformer::ane_chunk_rows() noexcept
{
  return AneFeedForward::kChunkDefault;
}

std::size_t
MetalQwenImage21Transformer::ane_runtime_bytes(const Config& c,
                                               int seq) noexcept
{
  std::size_t n = AneFeedForward::runtime_bytes(
      c.hidden, c.hidden * c.mlp_ratio, seq, ane_chunk_rows());
  // The q/k/v module is a second fixed unit, and it has to be booked:
  // one claimed at zero is memory CoreML holds that no plan can see.
  if (c.ane_qkv) {
    n += AneFeedForward::matmul_runtime_bytes(c.hidden, 3 * c.hidden, seq,
                                              ane_chunk_rows());
  }
  return n;
}

int
MetalQwenImage21Transformer::ane_plan_seq(int width, int height) noexcept
{
  // The joint sequence a picture implies: the /16 latent, unpatched, plus
  // a conditioning run this cannot know. 64 rows is the prompt's order
  // and is noise against the image at any size the tier is for.
  const int w = width > 0 ? width : 1024;
  const int h = height > 0 ? height : 1024;
  return (w / 16) * (h / 16) + 64;
}

bool
MetalQwenImage21Transformer::ane_setup_(int seq)
{
  if (_ane_tried) { return _ane != nullptr; }
  _ane_tried = true;
  if (_mc == nullptr || _mc->session() == nullptr || seq <= 0) {
    return false;
  }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "qwen-image-2.1";
  o.hidden  = _cfg.hidden;
  o.ffn     = _cfg.hidden * _cfg.mlp_ratio;
  o.seq     = seq;
  o.rows    = _cfg.ane_rows;
  o.chunk   = ane_chunk_rows();
  o.profile = std::getenv("VPIPE_QWEN_IMAGE21_ANE_PROFILE") != nullptr;
  _ane = AneFeedForward::create(o);
  return _ane != nullptr;
}

bool
MetalQwenImage21Transformer::ane_eligible_(int L, const Block& b) const
{
  if (_ane == nullptr) { return false; }
  const int cap = (_cfg.ane_layers > 0)
                      ? std::min(_cfg.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  if (L >= cap) { return false; }
  // Stageable: a dense bf16 matrix, or an affine one whose codes, scales
  // and biases are all present. The module is fp16 either way.
  auto ok = [](const QWeight& q) {
    if (q.empty()) { return false; }
    if (!q.quantized) { return true; }
    return (q.bits == 4 || q.bits == 8) && !q.scales.empty() &&
           !q.qbias.empty();
  };
  return ok(b.gate_w) && ok(b.proj_w) && ok(b.out_w);
}

void
MetalQwenImage21Transformer::ane_stage_(int L, const Block& b)
{
  if (_ane == nullptr) { return; }
  // This model keeps gate and up as SEPARATE matrices and never fuses
  // them, so each is read whole -- stride 1, offset 0. A family that
  // fuses reads one matrix twice with stride 2.
  auto src = [&](const QWeight& q) {
    AneFfnSource s;
    s.w         = &q.w;
    s.codes     = q.quantized ? &q.codes : nullptr;
    s.scales    = q.quantized ? &q.scales : nullptr;
    s.qbias     = q.quantized ? &q.qbias : nullptr;
    s.quantized = q.quantized;
    s.bits      = q.quantized ? q.bits : 0;
    s.stride    = 1;
    s.offset    = 0;
    return s;
  };
  _ane->stage(L, src(b.gate_w), src(b.proj_w), src(b.out_w), _quant_group);
}

// THE q/k/v TIER: one matmul module, hidden -> 3*hidden, whose single
// weight slot holds q, k and v stacked by rows, and whose output
// COLUMNS scatter straight into qb / kb / vb -- so the three stay the
// separate matrices this model keeps, and nothing is fused on the GPU
// side to feed it. Same row split, balancer and probe as the
// feed-forward's; the ANE's rows are the TAIL of the block's.
bool
MetalQwenImage21Transformer::ane_qkv_setup_(int seq)
{
  if (_ane_qkv_tried) { return _ane_qkv != nullptr; }
  _ane_qkv_tried = true;
  if (_mc == nullptr || _mc->session() == nullptr || seq <= 0) {
    return false;
  }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "qwen-image-2.1-qkv";
  o.hidden  = _cfg.hidden;
  o.ffn     = 3 * _cfg.hidden;       // the matmul's output width
  o.seq     = seq;
  o.rows    = _cfg.ane_rows;
  o.chunk   = ane_chunk_rows();
  o.matmul  = true;
  o.profile = std::getenv("VPIPE_QWEN_IMAGE21_ANE_PROFILE") != nullptr;
  _ane_qkv = AneFeedForward::create(o);
  return _ane_qkv != nullptr;
}

bool
MetalQwenImage21Transformer::ane_qkv_eligible_(int L, const Block& b) const
{
  if (_ane_qkv == nullptr) { return false; }
  const int cap = (_cfg.ane_layers > 0)
                      ? std::min(_cfg.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  if (L >= cap) { return false; }
  auto ok = [](const QWeight& q) {
    if (q.empty()) { return false; }
    if (!q.quantized) { return true; }
    return (q.bits == 4 || q.bits == 8) && !q.scales.empty() &&
           !q.qbias.empty();
  };
  return ok(b.qw) && ok(b.kw) && ok(b.vw);
}

void
MetalQwenImage21Transformer::ane_qkv_stage_(int L, const Block& b)
{
  if (_ane_qkv == nullptr) { return; }
  const std::size_t H = (std::size_t)_cfg.hidden;
  // Stacked along the slot's rows in the order the split scatters the
  // output columns: q, then k, then v.
  auto part = [&](const QWeight& q, std::size_t row0) {
    AneFfnSource s;
    s.w         = &q.w;
    s.codes     = q.quantized ? &q.codes : nullptr;
    s.scales    = q.quantized ? &q.scales : nullptr;
    s.qbias     = q.quantized ? &q.qbias : nullptr;
    s.quantized = q.quantized;
    s.bits      = q.quantized ? q.bits : 0;
    s.stride    = 1;
    s.offset    = 0;
    s.slot      = 0;
    s.slot_row  = row0;
    s.rows      = H;
    return s;
  };
  _ane_qkv->stage(L,
                  std::vector<AneFfnSource>{part(b.qw, 0), part(b.kw, H),
                                            part(b.vw, 2 * H)},
                  _quant_group);
}

std::size_t
MetalQwenImage21Transformer::checkpoint_block_bytes_() const
{
  if (_ws == nullptr) { return 0; }
  const MetalLlamaWeights& w = _ws->src();
  const std::string pre = block_pre_(0);
  std::size_t n = 0;
  for (const std::string& nm : w.tensor_names()) {
    if (nm.rfind(pre, 0) != 0) { continue; }
    const auto* ti = w.info(nm);
    if (ti != nullptr) { n += (std::size_t)ti->nbytes; }
  }
  return n;
}

void
MetalQwenImage21Transformer::set_residency_schedule(int steps)
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
MetalQwenImage21Transformer::release_resident_blocks(std::size_t bytes)
{
  const std::size_t freed = _resid.release(bytes, [this]() -> std::size_t {
    return evict_tail_block_();
  });
  if (freed > 0 && _mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalQwenImage21Transformer: released {} MB of resident blocks "
        "({} left)", freed >> 20, _resid.count()));
  }
  return freed;
}

std::uint64_t
MetalQwenImage21Transformer::resident_bytes() const
{
  std::uint64_t n = qw_bytes_(_img_in) + qw_bytes_(_txt_in_a) +
                    qw_bytes_(_txt_in_b) + _txt_norm.byte_size() +
                    qw_bytes_(_time_1) + qw_bytes_(_time_2) +
                    qw_bytes_(_mod) + qw_bytes_(_norm_out) +
                    qw_bytes_(_proj_out);
  // `_blocks` IS the answer in both modes, and adding the residency
  // ledger to it would double every promoted block. Preloaded, it holds
  // the whole stack and `_resid` is empty; streaming, it holds exactly
  // what residency promoted -- which is the same set `_resid.bytes()`
  // tallies, counted from the buffers instead of from the ledger.
  for (const Block& b : _blocks) { n += block_bytes_(b); }
  return n;
}

}  // namespace genai
}  // namespace vpipe
