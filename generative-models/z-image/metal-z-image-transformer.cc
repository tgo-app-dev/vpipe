#include "generative-models/z-image/metal-z-image-transformer.h"

#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/i8-gemm.h"
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
constexpr const char* kKey = "z-image-dit/";

}  // namespace

MetalZImageTransformer::~MetalZImageTransformer()
{
  // GIVE THE POOL BACK. Freeing a wired buffer unwires it in the kernel,
  // so the machine recovers either way -- but the pool's own counter
  // would not, and a DiT destroyed and reloaded per prompt would leak
  // its whole share of the budget per prompt until nothing could wire.
  if (_wire.on()) {
    wire_fixed_(false);
    for (Block& b : _blocks) { wire_block_(b, false); }
    for (Block& b : _ctx) { wire_block_(b, false); }
  }
}

// ---- block names ----------------------------------------------------
//
// The streaming stack is noise_refiner THEN layers, which is not the
// order the reference runs them in -- see the class comment. What makes
// that safe is that the two refiners touch disjoint rows.

std::string
MetalZImageTransformer::mod_prefix_(int i) const
{
  if (i < _cfg.n_refiner) {
    return "noise_refiner." + std::to_string(i) + ".";
  }
  return "layers." + std::to_string(i - _cfg.n_refiner) + ".";
}

std::string
MetalZImageTransformer::ctx_prefix_(int i) const
{
  return "context_refiner." + std::to_string(i) + ".";
}

// The patch key the embedder and the final layer are stored under:
// f"{patch_size}-{f_patch_size}". A ModuleDict, so it is part of the
// tensor name rather than a config lookup.
static std::string
patch_key_(const MetalZImageTransformer::Config& c)
{
  return std::to_string(c.patch) + "-" + std::to_string(c.f_patch);
}

// ---- residency + the wired pool -------------------------------------

std::size_t
MetalZImageTransformer::wire_block_(Block& b, bool on)
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
  QWeight* q[] = {&b.qw, &b.kw, &b.vw, &b.ow, &b.w1, &b.w3, &b.w2, &b.ada};
  for (QWeight* w : q) { qw(*w); }
  one(b.nq); one(b.nk);
  one(b.an1); one(b.an2); one(b.fn1); one(b.fn2);
  one(b.ada_b);
  return changed;
}

std::size_t
MetalZImageTransformer::wire_fixed_(bool on)
{
  if (!_ws) { return 0; }
  std::size_t changed = 0;
  _ws->for_each_weight([&](SharedBuffer& b) {
    changed += _wire.wire_one(_mc, b, on);
  });
  return changed;
}

std::size_t
MetalZImageTransformer::wire_retry_slack_() const
{
  if (_wire_block_hint > 0) { return _wire_block_hint; }
  if (_resid.count() > 0) {
    return _resid.bytes() / (std::size_t)_resid.count();
  }
  return 0;
}

std::size_t
MetalZImageTransformer::evict_tail_block_()
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
MetalZImageTransformer::resident_pages_(std::size_t* examined,
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
MetalZImageTransformer::Config::read_dims(const std::string& dir,
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
    if (cls != "ZImageTransformer2DModel") { return false; }
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
  out->cap_dim = geti("cap_feat_dim", out->cap_dim);
  out->norm_eps = getf("norm_eps", out->norm_eps);
  out->rope_theta = getf("rope_theta", out->rope_theta);
  out->t_scale = getf("t_scale", out->t_scale);
  if (o.contains("qk_norm")) {
    out->qk_norm = o.at("qk_norm").as_bool(out->qk_norm);
  }
  // GQA IS NOT SUPPORTED HERE and no published checkpoint uses it:
  // n_kv_heads == n_heads in both. Refuse rather than run MHA over a
  // GQA checkpoint, which would read past the k/v projections.
  if (geti("n_kv_heads", out->n_heads) != out->n_heads) { return false; }
  // A SigLIP tower belongs to the unreleased Omni variant. Its weights
  // would simply be absent here, so the refusal is about the sequence:
  // the omni path also changes the token ORDER and adds a per-token
  // modulation, neither of which this implements.
  if (o.contains("siglip_feat_dim") && !o.at("siglip_feat_dim").is_null()) {
    return false;
  }
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
  // all_patch_size / all_f_patch_size are LISTS -- the model can carry
  // several embedders keyed by "{p}-{f}". Every published checkpoint
  // ships exactly one, and the tensor names below name it, so take the
  // first and refuse a multi-patch checkpoint rather than guessing.
  auto one_patch = [&](const char* k, int* dst) {
    if (!o.contains(k)) { return true; }
    const FlexData a = o.at(k);
    if (!a.is_array()) { return true; }
    auto arr = a.as_array();
    if (arr.size() != 1) { return false; }
    *dst = (int)arr.at(0).as_int(*dst);
    return true;
  };
  if (!one_patch("all_patch_size", &out->patch)) { return false; }
  if (!one_patch("all_f_patch_size", &out->f_patch)) { return false; }

  out->head_dim = out->n_heads > 0 ? out->hidden / out->n_heads
                                   : out->head_dim;
  // The SwiGLU width is not in config.json at all -- the reference
  // computes int(dim / 3 * 8). Stated here so a checkpoint whose
  // feed_forward disagrees fails at load rather than at the first GEMM.
  out->ffn_inner = (int)((double)out->hidden / 3.0 * 8.0);
  // sum(axes_dims) == head_dim is the reference's own assert.
  if (out->axes[0] + out->axes[1] + out->axes[2] != out->head_dim) {
    return false;
  }
  return out->hidden > 0 && out->n_layers > 0 && out->n_heads > 0 &&
         out->n_refiner > 0;
}

// ---- weights --------------------------------------------------------

SharedBuffer
MetalZImageTransformer::to_elt_(WeightSet& ws, const std::string& name,
                                Retain r)
{
  auto build = [&]() { return to_bf16_(ws.src(), _mc, name); };
  if (r == Retain::Streamed) { return ws.stream_derived(build); }
  return ws.derived(std::string(kKey) + "elt|" + name, build);
}

MetalZImageTransformer::QWeight
MetalZImageTransformer::load_qw_(WeightSet& ws, const std::string& name,
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
MetalZImageTransformer::load_block_(WeightSet& ws, const std::string& p,
                                    bool with_ada, Block& b, Retain r)
{
  b.qw = load_qw_(ws, p + "attention.to_q", r);
  b.kw = load_qw_(ws, p + "attention.to_k", r);
  b.vw = load_qw_(ws, p + "attention.to_v", r);
  // to_out is a ModuleList, so the projection is ".0" -- not a typo.
  b.ow = load_qw_(ws, p + "attention.to_out.0", r);
  b.nq = to_elt_(ws, p + "attention.norm_q.weight", r);
  b.nk = to_elt_(ws, p + "attention.norm_k.weight", r);
  // SwiGLU as Llama names it: w1 gates, w3 is the up projection and w2
  // the down one. Reading w1 and w3 the other way round is a model that
  // still denoises.
  b.w1 = load_qw_(ws, p + "feed_forward.w1", r);
  b.w3 = load_qw_(ws, p + "feed_forward.w3", r);
  b.w2 = load_qw_(ws, p + "feed_forward.w2", r);
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
MetalZImageTransformer::load_fixed_(WeightSet& ws)
{
  const Retain r = Retain::Cached;
  const std::string pk = patch_key_(_cfg);
  _x_in = load_qw_(ws, "all_x_embedder." + pk, r);
  _x_in_b = to_elt_(ws, "all_x_embedder." + pk + ".bias", r);
  _cap_norm = to_elt_(ws, "cap_embedder.0.weight", r);
  _cap_in = load_qw_(ws, "cap_embedder.1", r);
  _cap_in_b = to_elt_(ws, "cap_embedder.1.bias", r);
  _time_1 = load_qw_(ws, "t_embedder.mlp.0", r);
  _time_1b = to_elt_(ws, "t_embedder.mlp.0.bias", r);
  _time_2 = load_qw_(ws, "t_embedder.mlp.2", r);
  _time_2b = to_elt_(ws, "t_embedder.mlp.2.bias", r);
  _x_pad = to_elt_(ws, "x_pad_token", r);
  _cap_pad = to_elt_(ws, "cap_pad_token", r);
  _final_ada = load_qw_(ws, "all_final_layer." + pk + ".adaLN_modulation.1",
                        r);
  _final_ada_b = to_elt_(
      ws, "all_final_layer." + pk + ".adaLN_modulation.1.bias", r);
  _final_lin = load_qw_(ws, "all_final_layer." + pk + ".linear", r);
  _final_lin_b = to_elt_(ws, "all_final_layer." + pk + ".linear.bias", r);
  return !_x_in.empty() && !_x_in_b.empty() && !_cap_norm.empty() &&
         !_cap_in.empty() && !_cap_in_b.empty() && !_time_1.empty() &&
         !_time_1b.empty() && !_time_2.empty() && !_time_2b.empty() &&
         !_x_pad.empty() && !_cap_pad.empty() && !_final_ada.empty() &&
         !_final_ada_b.empty() && !_final_lin.empty() &&
         !_final_lin_b.empty();
}

std::size_t
MetalZImageTransformer::qw_bytes_(const QWeight& w)
{
  if (w.quantized) {
    return w.codes.byte_size() + w.scales.byte_size() + w.qbias.byte_size();
  }
  return w.w.byte_size();
}

std::size_t
MetalZImageTransformer::block_bytes_(const Block& b)
{
  return qw_bytes_(b.qw) + qw_bytes_(b.kw) + qw_bytes_(b.vw) +
         qw_bytes_(b.ow) + b.nq.byte_size() + b.nk.byte_size() +
         qw_bytes_(b.w1) + qw_bytes_(b.w3) + qw_bytes_(b.w2) +
         b.an1.byte_size() + b.an2.byte_size() + b.fn1.byte_size() +
         b.fn2.byte_size() + qw_bytes_(b.ada) + b.ada_b.byte_size();
}

void
MetalZImageTransformer::each_block_tensor_(
    int L, Block& b, const BlockSlots<Block>::TensorFn& fn) const
{
  const std::string p = mod_prefix_(L);
  auto qw = [&](QWeight& w, const char* nm) {
    if (w.quantized) {
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
MetalZImageTransformer::clone_block_(const Block& src, Block& dst,
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
  cq(src.w1, dst.w1); cq(src.w3, dst.w3); cq(src.w2, dst.w2);
  ok = clone_buf_(_mc, src.an1, dst.an1, copy) && ok;
  ok = clone_buf_(_mc, src.an2, dst.an2, copy) && ok;
  ok = clone_buf_(_mc, src.fn1, dst.fn1, copy) && ok;
  ok = clone_buf_(_mc, src.fn2, dst.fn2, copy) && ok;
  cq(src.ada, dst.ada);
  ok = clone_buf_(_mc, src.ada_b, dst.ada_b, copy) && ok;
  return ok;
}

SharedBuffer
MetalZImageTransformer::rebuild_one_(const std::string& nm, Placement how)
{
  if (_ws == nullptr) { return {}; }
  if (how == Placement::kRaw) {
    return _ws->stream_tensor(nm, _mc, WeightSet::Residency::Copied);
  }
  return _ws->stream_derived([&]() { return to_bf16_(_ws->src(), _mc, nm); });
}

bool
MetalZImageTransformer::fill_narrow_(const std::string& nm,
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

void
MetalZImageTransformer::configure_slots_()
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
  _slots.set_weight_set(_ws.get());
  _slots.configure(_mc, std::move(ops), "MetalZImageTransformer",
                   "VPIPE_Z_IMAGE_NO_SLOTS");
}

// ---- load -----------------------------------------------------------

std::unique_ptr<MetalZImageTransformer>
MetalZImageTransformer::load(const std::string& model_dir, MetalCompute* mc,
                             const Config& cfg, bool stream_blocks,
                             const std::vector<LoraSpec>& loras)
{
  return load(WeightSet::open(model_dir, nullptr), mc, cfg, stream_blocks,
              loras);
}

std::unique_ptr<MetalZImageTransformer>
MetalZImageTransformer::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                             const Config& cfg, bool stream_blocks,
                             const std::vector<LoraSpec>& loras)
{
  if (ws == nullptr || mc == nullptr) { return nullptr; }
  std::unique_ptr<MetalZImageTransformer> m(new MetalZImageTransformer());
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
  m->_fn_dequant4 = m->_lib_dequant.function("affine_dequant4_bf16");
  m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant8_bf16");
  m->_fn_gemm_mma = m->_lib_mma.function("dense_gemm_mma_t_n128_f16");
  m->_fn_gemm_mma_deep =
      m->_lib_mma.function("dense_gemm_mma_t_n128x256_f16");
  m->_fn_gemm_mma_tn2 =
      m->_lib_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");
  m->_fn_qmm = m->_lib_qmm.function(
      m->_quant_bits == 8 ? "affine_qmm_steel_w8g64_f16"
                          : "affine_qmm_steel_w4g64_f16");

  // VALIDATE. An unvalidated ComputeFunction dispatches as a silent
  // no-op, which in a 34-block stack is a picture, not a failure.
  if (!m->_fn_gemm_t.valid() || !m->_fn_ln_plain.valid() ||
      !m->_fn_adaln.valid() || !m->_fn_gated_tanh.valid() ||
      !m->_fn_swiglu.valid() || !m->_fn_rms.valid() ||
      !m->_fn_tr_rope.valid() || !m->_fn_residual.valid() ||
      !m->_fn_silu.valid() || !m->_fn_transpose.valid() ||
      !m->_fn_copy_rows.valid() || !m->_fn_bias_add.valid()) {
    if (mc->session() != nullptr) {
      mc->session()->error(fmt("MetalZImageTransformer: a required kernel "
                               "did not build"));
    }
    return nullptr;
  }
  m->_steel_attn_ok = m->_lib_attn.valid();
  m->_use_attn_nax = m->_steel_attn_ok && m->_lib_attn_nax.valid() &&
                     mc->supports_matrix_cores() &&
                     std::getenv("VPIPE_Z_IMAGE_NO_NAX_ATTN") == nullptr;
  m->_use_mma2 = mc->supports_matrix_cores() && m->_fn_gemm_mma.valid() &&
                 std::getenv("VPIPE_Z_IMAGE_NO_MMA2") == nullptr;
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm,
                                              /*bf16=*/true);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }
  if (const char* e = std::getenv("VPIPE_Z_IMAGE_SOL")) {
    m->_cfg.sol.enabled = std::atoi(e) != 0;
  }
  // HIGHER tau keeps FEWER blocks: it is a threshold, not a budget.
  if (const char* e = std::getenv("VPIPE_Z_IMAGE_SOL_TAU")) {
    m->_cfg.sol.tau = (float)std::atof(e);
  }
  if (const char* e = std::getenv("VPIPE_Z_IMAGE_ANE")) {
    m->_cfg.ane_ffn = std::atoi(e) != 0;
  }
  if (const char* e = std::getenv("VPIPE_Z_IMAGE_ANE_ROWS")) {
    m->_cfg.ane_rows = std::atoi(e);
  }
  if (m->_cfg.sol.enabled) {
    std::string serr;
    m->_sol = MetalSolAttention::load(mc, /*bf16=*/true, &serr);
    if (m->_sol == nullptr && mc->session() != nullptr) {
      mc->session()->log_normal(fmt(
          "MetalZImageTransformer: sol_attn unavailable -- {}",
          serr.empty() ? "kernels did not load" : serr));
    }
  }
  {
    if (const char* e = std::getenv("VPIPE_Z_IMAGE_SAGE")) {
      m->_cfg.sage.enabled = std::atoi(e) != 0;
    }
    bool sage_fatal = false;
    m->_sage = MetalSageAttention::load_for_model(
        mc, /*bf16=*/true, m->_cfg.sage, "MetalZImageTransformer",
        &sage_fatal);
    if (sage_fatal) { return nullptr; }
    if (m->_sage && !m->_use_attn_nax) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt(
            "MetalZImageTransformer: sage_attn is off -- the matrix-core "
            "flash entry is unavailable here"));
      }
      m->_sage.reset();
    }
  }
  if (const char* e = std::getenv("VPIPE_Z_IMAGE_NO_FOLD")) {
    m->_fold_mod = std::atoi(e) == 0;
  }
  if (const char* t = std::getenv("VPIPE_Z_IMAGE_GEMM_TILE")) {
    m->_gemm_tile = std::atoi(t);
  }
  if (m->_gemm_tile == 2 && !m->_fn_gemm_bm64bn64.valid()) {
    m->_gemm_tile = 1;
  }
  if (m->_gemm_tile == 1 && !m->_fn_gemm_bm64.valid()) { m->_gemm_tile = 0; }

  if (!m->load_fixed_(*m->_ws)) {
    if (mc->session() != nullptr) {
      mc->session()->error(fmt("MetalZImageTransformer: the model-level "
                               "weights are incomplete"));
    }
    return nullptr;
  }

  // THE CONTEXT REFINER IS ALWAYS HELD, in both modes. It is two blocks
  // and a slot pair is also two blocks, so streaming it would cost the
  // same resident bytes AND re-read them every forward -- there is
  // nothing to win. It is also why the streaming stack can stay one
  // uniform shape: these two carry no adaLN, and a slot has to be
  // refillable by every index that uses it.
  m->_ctx.resize((std::size_t)m->_cfg.n_refiner);
  for (int i = 0; i < m->_cfg.n_refiner; ++i) {
    if (!m->load_block_(*m->_ws, m->ctx_prefix_(i), /*with_ada=*/false,
                        m->_ctx[(std::size_t)i], Retain::Cached)) {
      if (mc->session() != nullptr) {
        mc->session()->error(fmt("MetalZImageTransformer: context_refiner "
                                 "block {} is incomplete", i));
      }
      return nullptr;
    }
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
          mc->session()->error(fmt("MetalZImageTransformer: block {} is "
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
        "MetalZImageTransformer: the LoRA applier did not load; the "
        "adapters are ignored and the base model runs"));
  }
  if (mc->session() != nullptr) {
    mc->session()->log_normal(fmt(
        "MetalZImageTransformer: {} blocks ({} ctx + {} noise + {} main), "
        "hidden {}, {} heads x {}, ffn {}, {}{}{}",
        m->_cfg.n_blocks(), m->_cfg.n_refiner, m->_cfg.n_refiner,
        m->_cfg.n_layers, m->_cfg.hidden, m->_cfg.n_heads, m->_cfg.head_dim,
        m->_cfg.ffn_inner,
        m->_quant_bits > 0
            ? "w" + std::to_string(m->_quant_bits) + " g" +
                  std::to_string(m->_quant_group)
            : std::string(m->_src_f32 ? "f32 -> bf16" : "bf16"),
        stream_blocks ? ", streaming" : "",
        m->_use_mma2 ? ", matmul2d" : ""));
  }
  return m;
}

// The timestep sinusoid, on the host.
//
// TWO details are load-bearing. COS occupies the first half and sin the
// second, which is the opposite of the sin-first convention several
// families here use. And the argument is (1 - sigma) * t_scale -- the
// pipeline hands the model `(1000 - t)/1000` where t is the scheduler's
// own sigma*1000, so a model at the noisy end sees an input near 0
// where every other DiT in this tree sees one near 1. Getting this
// backwards produces a clean, plausible, completely wrong picture.
std::vector<float>
MetalZImageTransformer::time_proj_(float sigma) const
{
  const int D = _cfg.time_freq;
  const int half = D / 2;
  std::vector<float> out((std::size_t)D, 0.0f);
  const double t = (double)_cfg.t_scale * (1.0 - (double)sigma);
  for (int i = 0; i < half; ++i) {
    const double f =
        std::exp(-std::log(10000.0) * (double)i / (double)half);
    const double a = t * f;
    out[(std::size_t)i] = (float)std::cos(a);
    out[(std::size_t)(half + i)] = (float)std::sin(a);
  }
  return out;
}

// ONE joint table for the whole sequence, which the refiners then read
// as a prefix and a suffix -- the reference builds the two halves'
// tables separately and concatenates them in exactly this order.
//
// f32, because the rotation error is STRUCTURED: bf16 table rounding
// compounds over 34 blocks and, worse, over denoise steps. Angles stay
// f64 to match the reference's float64 freqs.
void
MetalZImageTransformer::build_rope_(const zimage::Layout& lay,
                                    SharedBuffer* cos_out,
                                    SharedBuffer* sin_out) const
{
  const int HD = _cfg.head_dim;
  const int pairs = HD / 2;
  const int seq = lay.joint_len;
  *cos_out = _mc->make_shared_buffer((std::size_t)seq * HD * sizeof(float));
  *sin_out = _mc->make_shared_buffer((std::size_t)seq * HD * sizeof(float));
  if (cos_out->empty() || sin_out->empty()) { return; }
  auto* c = static_cast<float*>(cos_out->contents());
  auto* s = static_cast<float*>(sin_out->contents());
  const double theta = (double)_cfg.rope_theta;
  // Per-pair inverse frequency and which of the three axes it reads --
  // functions of (axis, j) only, so precomputed once.
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
      const double ang =
          pos[pair_axis[(std::size_t)pair]] * pair_freq[(std::size_t)pair];
      const float cb = (float)std::cos(ang);
      const float sb = (float)std::sin(ang);
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

// ---- forward --------------------------------------------------------

SharedBuffer
MetalZImageTransformer::forward(const Request& req, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return SharedBuffer{};
  };
  if (req.layout == nullptr || req.latents == nullptr || req.cap == nullptr) {
    return fail("z-image forward: missing input");
  }
  const zimage::Layout& lay = *req.layout;
  const int H = _cfg.hidden, NH = _cfg.n_heads, Hd = _cfg.head_dim;
  const int FFI = _cfg.ffn_inner, PD = _cfg.patch_dim();
  const int JT = lay.joint_len, IMGT = lay.img_total;
  const int CAPT = lay.cap_total, IMG = lay.img_tokens;
  const int NM = _cfg.n_mod_blocks();
  if (JT <= 0 || IMG <= 0 || NH * Hd != H) {
    return fail("z-image forward: degenerate geometry");
  }
  if (!_steel_attn_ok) {
    return fail("z-image forward: the steel attention kernel is "
                "unavailable");
  }
  // THE ROTARY TABLES ARE BOUNDED. The reference indexes a precomputed
  // table of `axes_lens` rows per axis, so a position past the end is
  // an index error there and would be a silently plausible angle here.
  for (int t = 0; t < JT; ++t) {
    if (lay.pos_t[(std::size_t)t] >= _cfg.axes_lens[0] ||
        lay.pos_h[(std::size_t)t] >= _cfg.axes_lens[1] ||
        lay.pos_w[(std::size_t)t] >= _cfg.axes_lens[2]) {
      return fail("z-image forward: a rotary position is past the "
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
  SharedBuffer capn = buf((std::size_t)lay.cap_len * _cfg.cap_dim);
  SharedBuffer te1 = buf((std::size_t)_cfg.time_mid);
  SharedBuffer temb = buf((std::size_t)_cfg.adaln_dim);
  SharedBuffer tsilu = buf((std::size_t)_cfg.adaln_dim);
  SharedBuffer mods = buf((std::size_t)4 * H);
  // THE FOLDED NORM GAINS. `scale_msa` and `scale_mlp` are ONE H-wide
  // vector each for the whole forward, so `rms(x)*w*(1+scale)` is
  // `rms(x)*(w*(1+scale))` -- exact algebra that turns two full-width
  // passes over [rows, H] into one, plus an H-element multiply. At
  // 1024^2 each saved pass is 16M elements a block across 32 blocks.
  SharedBuffer nw1 = buf((std::size_t)H);
  SharedBuffer nw3 = buf((std::size_t)H);
  SharedBuffer outmod = buf((std::size_t)H);
  SharedBuffer tin = buf((std::size_t)_cfg.time_freq);
  SharedBuffer zero = buf((std::size_t)H);
  SharedBuffer vel = buf((std::size_t)IMG * PD);
  SharedBuffer rcos, rsin;
  build_rope_(lay, &rcos, &rsin);
  if (joint.empty() || nrm.empty() || ob.empty() || ffg.empty() ||
      ffu.empty() || zero.empty() || vel.empty() || rcos.empty() ||
      rsin.empty() || ao.empty() || vh.empty() || capn.empty() ||
      nw1.empty() || nw3.empty()) {
    return fail("z-image forward: scratch allocation failed");
  }
  std::memset(zero.contents(), 0, zero.byte_size());
  {
    const std::vector<float> tp = time_proj_(req.timestep);
    auto* d = static_cast<std::uint16_t*>(tin.contents());
    for (int i = 0; i < _cfg.time_freq; ++i) {
      d[i] = f32_to_bf16_(tp[(std::size_t)i]);
    }
  }

  CommandStream stream = mc->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  if (!enc.valid()) { return fail("z-image forward: no compute encoder"); }

  // Per-stage TRACE, for locating a numeric gap against the reference.
  // It COMMITS, so a traced run is not a timing run.
  const char* trace_dir = std::getenv("VPIPE_Z_IMAGE_TRACE");
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
  // the pass). Use VPIPE_Z_IMAGE_STREAM_PROFILE for the read, and
  // read this one for the GPU split alone.
  const bool prof = [] {
    const char* e = std::getenv("VPIPE_Z_IMAGE_DIT_PROFILE");
    return e != nullptr && *e != '\0' && std::atoi(e) != 0;
  }();
  struct Bucket { const char* name; double ms; long marks; };
  std::vector<Bucket> pbuckets;
  long pmarks = 0;
  auto pclock = std::chrono::steady_clock::now();
  auto mark = [&](const char* bucket) {
    if (!prof) { return; }
    enc.end();
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
    stream = mc->make_command_stream();
    enc = stream.begin_compute();
    pclock = std::chrono::steady_clock::now();
  };

  // ---- dispatch helpers ---------------------------------------------
  auto lin = [&](const SharedBuffer& x, std::size_t xe, const QWeight& w,
                 const SharedBuffer& y, std::size_t ye, int M, int N, int K,
                 const lora::Stack& lf = lora::Stack{}) {
    // The adapter accumulates ONTO the base result, so it runs after
    // whichever base route was taken and is identical for all of them.
    auto lora_after = [&]() {
      if (!lf.empty()) {
        _lora.apply(enc, x, xe, lf, y, ye, M, N, K, _mma_min_m);
      }
    };
    if (gemm_mma_(enc, x, xe, w, y, ye, M, N, K)) { lora_after(); return; }
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
    enc.set_function(_fn_qmm);
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
  // out = (1 + scale) * x, the model's modulation. The kernel's shift
  // slot takes the zero row: this model emits no shift at all.
  auto modulate = [&](const SharedBuffer& x, std::size_t xe,
                      const SharedBuffer& sc, std::size_t sce,
                      const SharedBuffer& y, std::size_t ye, int R) {
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
  // DENSE AND BIDIRECTIONAL, one dispatch per block. The three stages
  // run at three sequence lengths, so a params block and a specialized
  // function are memoized per length rather than rebuilt per block.
  const bool nax = _use_attn_nax;
  const int A_BQ = nax ? 64 : 32;
  const int A_BK = nax ? 32 : 16;
  const float ascale = 1.0f / std::sqrt((float)Hd);
  std::vector<std::pair<int, SharedBuffer>> aparams;
  auto params_for = [&](int R) -> const SharedBuffer* {
    for (auto& p : aparams) {
      if (p.first == R) { return &p.second; }
    }
    SharedBuffer b = mc->make_shared_buffer(sizeof(SteelAttnParams));
    if (b.empty()) { return nullptr; }
    auto* p = static_cast<SteelAttnParams*>(b.contents());
    p->B = 1; p->H = NH; p->D = Hd; p->qL = R; p->kL = R;
    p->gqa_factor = 1; p->scale = ascale;
    p->NQ = (R + A_BQ - 1) / A_BQ; p->NK = (R + A_BK - 1) / A_BK;
    p->NQ_aligned = R / A_BQ; p->NK_aligned = R / A_BK;
    p->qL_rem = R - p->NQ_aligned * A_BQ;
    p->kL_rem = R - p->NK_aligned * A_BK;
    p->qL_off = 0;
    const std::int64_t rs = (std::int64_t)R * Hd;
    p->Q_strides[0] = (std::int64_t)NH * rs;
    p->Q_strides[1] = rs; p->Q_strides[2] = Hd;
    p->K_strides[0] = p->Q_strides[0];
    p->K_strides[1] = rs; p->K_strides[2] = Hd;
    p->V_strides[0] = p->Q_strides[0];
    p->V_strides[1] = rs; p->V_strides[2] = Hd;
    p->O_strides[0] = p->Q_strides[0];
    p->O_strides[1] = rs; p->O_strides[2] = Hd;
    aparams.emplace_back(R, std::move(b));
    return &aparams.back().second;
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
    metal_compute::ComputeFunction fn =
        nax ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
            : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
    auto ins = _attn_fn.emplace(key, std::move(fn));
    return ins.first->second.valid() ? &ins.first->second : nullptr;
  };
  bool attn_ok = true;
  auto run_attn = [&](int L, int rows) {
    if (!attn_ok) { return; }
    // SOL, where the sequence is long enough to have something to skip.
    // Every attention in this model is square and non-causal, which is
    // exactly the shape Sol's exact half is built for -- there is no
    // causal segment to decline, unlike the block-causal families.
    if (_sol && nax && _cfg.sol.enabled && L >= _cfg.sol.dense_layers &&
        rows >= kSolMinSegment) {
      const int blk = _cfg.sol.key_block > 0 ? _cfg.sol.key_block
                                             : sol::kBlock;
      if ((rows + blk - 1) / blk >= 16) {
        MetalSolAttention::Band band{rows, rows, rows, rows, 0, 0};
        std::string serr;
        if (_sol->encode(enc, qh, kh, vh, ao, NH, NH, band, Hd, ascale,
                         _cfg.sol, &serr)) {
          return;
        }
        if (_mc->session() != nullptr) {
          _mc->session()->log_debug(fmt(
              "z-image: sol declined this stage ({}); dense",
              serr.empty() ? "no reason" : serr));
        }
      }
    }
    bool i8_ok = false;
    if (_sage && nax && _cfg.sage.enabled && L >= _cfg.sage.dense_layers &&
        rows >= kSageMinSegment) {
      const MetalSageAttention::Operand qo{&qh, 0, Hd, rows * Hd};
      const MetalSageAttention::Operand ko{&kh, 0, Hd, rows * Hd};
      std::string gerr;
      i8_ok = _sage->prepare(enc, qo, ko, NH, NH, rows, rows, Hd, A_BQ,
                             A_BK, _cfg.sage, &gerr);
    }
    const SharedBuffer* pp = params_for(rows);
    const metal_compute::ComputeFunction* fn = attn_fn(rows, i8_ok);
    if (pp == nullptr || fn == nullptr) { attn_ok = false; return; }
    enc.set_function(*fn);
    enc.set_buffer(0, qh); enc.set_buffer(1, kh); enc.set_buffer(2, vh);
    enc.set_buffer(3, ao); enc.set_buffer(4, *pp);
    if (i8_ok) { _sage->bind(enc); }
    enc.dispatch({(unsigned)(32 * ((rows + A_BQ - 1) / A_BQ)),
                  (unsigned)(4 * NH), 1}, {32, 4, 1});
  };

  // ---- one block ----------------------------------------------------
  //
  // The sandwich: norm1 BEFORE the sublayer and norm2 on its OUTPUT,
  // inside the residual and under the gate. `mods` holds the four
  // chunks (scale_msa | gate_msa | scale_mlp | gate_mlp) for the block
  // about to run.
  auto run_block = [&](const Block& b, std::size_t xe, int rows, int row0,
                       bool modulated, int L,
                       AneFeedForward::Plan ane_plan,
                       bool ctx = false) -> bool {
    auto LA = [&](lora::Factors BlockLora::* w) {
      return lora_for_(ctx, L, w);
    };
    // The gain is pre-folded for a modulated block (see `nw1`), so
    // this is ONE pass rather than two.
    const bool fold = modulated && _fold_mod;
    rms(joint, xe, fold ? nw1 : b.an1, nrm, 0, rows, H);
    if (modulated && !fold) { modulate(nrm, 0, mods, 0, nrm, 0, rows); }
    mark("elt");
    lin(nrm, 0, b.qw, qb, 0, rows, H, H, LA(&BlockLora::q));
    lin(nrm, 0, b.kw, kb, 0, rows, H, H, LA(&BlockLora::k));
    lin(nrm, 0, b.vw, vb, 0, rows, H, H, LA(&BlockLora::v));
    mark("gemm.qkv");
    // Per-head RMSNorm over the last Hd of a token-major [rows, NH, Hd].
    rms(qb, 0, b.nq, qb, 0, rows * NH, Hd);
    rms(kb, 0, b.nk, kb, 0, rows * NH, Hd);
    tr_rope(qb, qh, rows, row0);
    tr_rope(kb, kh, rows, row0);
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
    if (modulated) {
      gated(joint, xe, mods, (std::size_t)H, ob, 0, rows);
    } else {
      resid(joint, xe, ob, 0, rows * H);
    }

    rms(joint, xe, fold ? nw3 : b.fn1, nrm, 0, rows, H);
    if (modulated && !fold) {
      modulate(nrm, 0, mods, (std::size_t)2 * H, nrm, 0, rows);
    }
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
      const auto t_d0 = std::chrono::steady_clock::now();
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) { return false; }
      ane_drain_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_d0).count();
      if (ane_split && _ane->join_stage(L)) {
        a_rows = _ane->begin(nrm, ob, rows);
      } else if (ane_split && _mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "z-image: staging block {}'s weights for the ANE failed; it "
            "keeps the GPU feed-forward", L));
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
    }
    // The GPU's share. A probe block deliberately takes EVERY row:
    // that is what makes its time comparable with the split's.
    const int g_rows = rows - a_rows;
    const auto t_g0 = std::chrono::steady_clock::now();
    if (g_rows > 0) {
      lin(nrm, 0, b.w1, ffg, 0, g_rows, FFI, H, LA(&BlockLora::w1));
      lin(nrm, 0, b.w3, ffu, 0, g_rows, FFI, H, LA(&BlockLora::w3));
      // In place on the gate: same index in, same index out. The
      // largest elementwise dispatch in the model, so it is the one
      // that pays most for being four-wide.
      swiglu(ffg, ffu, ffg, g_rows * FFI);
      lin(ffg, 0, b.w2, ob, 0, g_rows, H, FFI, LA(&BlockLora::w2));
    }
    mark("gemm.ff");
    // The GPU's rows have to be DONE before the ANE's are folded in and
    // before the norm below reads the whole tensor, so this commit is
    // the join and its time is the GPU's share for the balancer.
    if (ane_split || ane_probe) {
      enc.end();
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        if (a_rows > 0) { (void)_ane->finish(L, rows, 0.0, ane_drain_ms); }
        return false;
      }
      const double t_gpu = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_g0).count();
      if (ane_probe) { _ane->note_probe(ane_drain_ms, t_gpu); }
      if (!_ane->finish(L, rows, t_gpu, ane_drain_ms) && a_rows > 0) {
        return false;
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
    }
    rms(ob, 0, b.fn2, ob, 0, rows, H);
    if (modulated) {
      gated(joint, xe, mods, (std::size_t)3 * H, ob, 0, rows);
    } else {
      resid(joint, xe, ob, 0, rows * H);
    }
    mark("elt");
    return true;
  };
  // mods = adaLN_modulation(temb) -- a BARE Linear, no activation in
  // front of it. The final layer's is SiLU + Linear; the reference
  // spells that difference only as the index inside an nn.Sequential.
  auto fill_mod = [&](const Block& b) {
    lin(temb, 0, b.ada, mods, 0, 1, 4 * H, _cfg.adaln_dim);
    bias(b.ada_b, mods, 0, 1, 4 * H);
    // ... and fold each (1 + scale) into its norm's gain, one row wide.
    if (_fold_mod) {
      modulate(b.an1, 0, mods, 0, nw1, 0, 1);
      modulate(b.fn1, 0, mods, (std::size_t)2 * H, nw3, 0, 1);
    }
  };

  // ---- timestep -----------------------------------------------------
  lin(tin, 0, _time_1, te1, 0, 1, _cfg.time_mid, _cfg.time_freq);
  bias(_time_1b, te1, 0, 1, _cfg.time_mid);
  silu(te1, te1, _cfg.time_mid);
  lin(te1, 0, _time_2, temb, 0, 1, _cfg.adaln_dim, _cfg.time_mid);
  bias(_time_2b, temb, 0, 1, _cfg.adaln_dim);
  silu(temb, tsilu, _cfg.adaln_dim);

  // ---- the joint stream ---------------------------------------------
  //
  // Image first, then the caption; each half padded to a multiple of 32
  // with its own LEARNED pad token, which attends like any other row.
  lin(*req.latents, 0, _x_in, joint, 0, IMG, H, PD);
  bias(_x_in_b, joint, 0, IMG, H);
  copy_rows(_x_pad, 0, joint, (std::size_t)IMG * H, lay.img_pad(), H, 0, H);
  rms(*req.cap, 0, _cap_norm, capn, 0, lay.cap_len, _cfg.cap_dim);
  lin(capn, 0, _cap_in, joint, (std::size_t)IMGT * H, lay.cap_len, H,
      _cfg.cap_dim);
  bias(_cap_in_b, joint, (std::size_t)IMGT * H, lay.cap_len, H);
  copy_rows(_cap_pad, 0, joint,
            (std::size_t)(IMGT + lay.cap_len) * H, lay.cap_pad(), H, 0, H);
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
  if (_stream_blocks) { _slots.begin_forward(); }

  // ---- how much of the stack this box can KEEP ------------------------
  if (_stream_blocks) {
    if (_wire.on()) {
      if (_wire.retry(_mc, wire_retry_slack_())) {
        _resid.note_landscape_changed();
      }
      // The trunk takes its place BEFORE this forward's admissions
      // start asking: the blocks are the shed-able half, so a pool that
      // runs out should run out on them. The context refiner is part of
      // that trunk here -- it is held in both modes.
      wire_fixed_(true);
      for (Block& b : _ctx) { _wire.note_wired(_mc, wire_block_(b, true), 0); }
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
              "MetalZImageTransformer: resident weights are only {}% in "
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

  // ---- env-gated streaming split (VPIPE_Z_IMAGE_STREAM_PROFILE) ------
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
  const bool sprof = std::getenv("VPIPE_Z_IMAGE_STREAM_PROFILE") != nullptr;
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

  // ---- context_refiner, over the caption rows alone -------------------
  //
  // The reference runs the NOISE refiner first. These two touch
  // disjoint row ranges and never see each other's output, so the order
  // is free -- and taking the caption first leaves the 32 modulated
  // blocks contiguous below, which is what the streaming stack and the
  // prefetch want. UNMODULATED: no timestep enters the caption stream.
  for (int i = 0; i < _cfg.n_refiner; ++i) {
    if (_stream_stop && _stream_stop()) { return SharedBuffer{}; }
    if (_block_progress) {
      report_block(stream, _block_progress, done, TOTAL_BLOCKS);
    }
    if (!run_block(_ctx[(std::size_t)i], (std::size_t)IMGT * H, CAPT, IMGT,
                   /*modulated=*/false, i, AneFeedForward::Plan::kGpu,
                   /*ctx=*/true)) {
      return fail("z-image forward: the context refiner failed");
    }
    ++done;
  }
  trace("ctx", joint, (std::size_t)JT * H);

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
        return fail("z-image forward: a streamed block did not load");
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
    // The refiner's rows are the image half; the main stack's are the
    // whole joint sequence.
    const bool is_refiner = L < _cfg.n_refiner;
    const int rows = is_refiner ? IMGT : JT;

    // THE ANE'S PLAN FOR THIS BLOCK, decided before anything is encoded
    // because the staging has to ride under the attention. Only for the
    // MAIN stack: the module is sized once, for the joint length, and a
    // refiner block runs a shorter sequence than it was built for.
    AneFeedForward::Plan ane_plan = AneFeedForward::Plan::kGpu;
    if (_cfg.ane_ffn && !is_refiner && ane_setup_(JT) &&
        ane_eligible_(L, *b)) {
      ane_plan = _ane->plan_block();
    }
    if ((ane_plan != AneFeedForward::Plan::kGpu) &&
        (L == _cfg.n_refiner || _ane->needs_barrier())) {
      // An unsplit block commits nothing mid-block, so on a stack that
      // defers its commits the split point's drain would wait for every
      // such block queued before it. Drain what is queued FIRST.
      enc.end();
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        return fail("z-image forward: " + ge2);
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
    }
    if (ane_plan == AneFeedForward::Plan::kSplit) { ane_stage_(L, *b); }

    if (_block_progress) {
      // `done` of `total`: the blocks finished BEFORE this one. The
      // helper's contract wants a report that lags the truth and never
      // leads it.
      report_block(stream, _block_progress, done, TOTAL_BLOCKS);
    }

    fill_mod(*b);
    if (!run_block(*b, 0, rows, 0, /*modulated=*/true, L, ane_plan)) {
      return fail("z-image forward: block " + std::to_string(L) +
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
        if (!bf.wait_ok(&ge)) { return fail("z-image forward: " + ge); }
      }
      // ---- keep this block, if the box can hold it -------------------
      //
      // AFTER the wait, so nothing encoded still points at the slot,
      // and promotion CLONES out of it so the slot stays usable for the
      // read already in flight.
      const std::size_t nb = _slots.last_bytes();
      if (nb > 0 && _wire.wirable(nb) && _resid.admit(_mc, nb) &&
          _slots.promote_into(_blocks[(std::size_t)L])) {
        // Wired LAST, after every write this block will ever get: mlock
        // pins the pages that exist NOW.
        _wire.note_wired(_mc, wire_block_(_blocks[(std::size_t)L], true),
                         nb);
        _resid.note_admitted(nb);
      }
      sp_add(sp_gpu_ms, gp0);
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
      if (!enc.valid()) { return fail("z-image forward: no encoder"); }
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
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) {
        return fail("z-image forward: " + ge);
      }
      stream = mc->make_command_stream();
      enc = stream.begin_compute();
      if (!enc.valid()) { return fail("z-image forward: no encoder"); }
    }
    trace(("blk" + std::to_string(L)).c_str(), joint, (std::size_t)JT * H);
  }
  if (!attn_ok) {
    return fail("z-image forward: the steel attention kernel did not "
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
  {
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) {
      return fail("z-image forward: " + (ge.empty() ? "GPU failed" : ge));
    }
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
        "Z-Image DiT profile (joint={} img={} cap={}, {} blocks, {}, "
        "{} marks -- each is a commit AND a wait, so a bucket with many "
        "marks and little work is mostly measuring this): total {} ms{}",
        JT, IMGT, CAPT, TOTAL_BLOCKS,
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
        "MetalZImageTransformer: streamed {} of {} blocks -- read {} ms "
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

// ---- the matrix-core GEMM -------------------------------------------
//
// Returns false with NOTHING encoded when this box or this shape is not
// for it, and the caller keeps its tiled path -- so a decline is a
// fallback, never a hole.
bool
MetalZImageTransformer::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& xin,
                                  std::size_t xe, const QWeight& w,
                                  const SharedBuffer& y, std::size_t ye,
                                  int M, int N, int K) const
{
  // Only where the hardware has matrix cores, where M amortizes the
  // 128-row tile, and where N is not degenerate. The timestep, the
  // modulation and the final adaLN are M=1 -- exactly the shape the
  // tile is wrong for -- and stay on steel at no cost either way.
  if (!_use_mma2 || M < _mma_min_m || N < 16) { return false; }
  // And decline a shape whose operands would not survive the tiles'
  // 32-bit addressing: past 2^31 BYTES on ANY operand the mma tiles
  // stop storing, SILENTLY. See shared/mma-tile.h.
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
MetalZImageTransformer::ane_chunk_rows() noexcept
{
  return AneFeedForward::kChunkDefault;
}

std::size_t
MetalZImageTransformer::ane_runtime_bytes(const Config& c, int seq) noexcept
{
  return AneFeedForward::runtime_bytes(c.hidden, c.ffn_inner, seq,
                                       ane_chunk_rows());
}

int
MetalZImageTransformer::ane_plan_seq(int width, int height) noexcept
{
  // The joint sequence a picture implies: one row per 16x16 pixels
  // (the VAE's 8x times the DiT's patch 2), plus a caption this cannot
  // know. 128 rows is the prompt's order after its pad to 32 and is
  // noise against the image at any size the tier is for.
  const int w = width > 0 ? width : 1024;
  const int h = height > 0 ? height : 1024;
  return (w / zimage::kPixelsPerToken) * (h / zimage::kPixelsPerToken) + 128;
}

bool
MetalZImageTransformer::ane_setup_(int seq)
{
  if (_ane_tried) { return _ane != nullptr; }
  _ane_tried = true;
  if (_mc == nullptr || _mc->session() == nullptr || seq <= 0) {
    return false;
  }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "z-image";
  o.hidden  = _cfg.hidden;
  o.ffn     = _cfg.ffn_inner;
  o.seq     = seq;
  o.rows    = _cfg.ane_rows;
  o.chunk   = ane_chunk_rows();
  o.profile = std::getenv("VPIPE_Z_IMAGE_ANE_PROFILE") != nullptr;
  _ane = AneFeedForward::create(o);
  return _ane != nullptr;
}

bool
MetalZImageTransformer::ane_eligible_(int L, const Block& b) const
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
    return (q.bits == 4 || q.bits == 8) && !q.scales.empty() &&
           !q.qbias.empty();
  };
  return ok(b.w1) && ok(b.w3) && ok(b.w2);
}

void
MetalZImageTransformer::ane_stage_(int L, const Block& b)
{
  if (_ane == nullptr) { return; }
  // This model keeps the gate and the up projection as SEPARATE
  // matrices and never fuses them, so each is read whole -- stride 1,
  // offset 0. A family that fuses reads one matrix twice with stride 2.
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
  _ane->stage(L, src(b.w1), src(b.w3), src(b.w2), _quant_group);
}

// ---- runtime LoRA ---------------------------------------------------
//
// THE MODULE NAMES ARE THE MODEL'S OWN, which is the convention a
// diffusers-style export of this architecture carries: the adapter's
// keys are the base tensor's path without `.weight`, optionally under
// a `transformer.`/`diffusion_model.` wrapper that the shared reader's
// rename strips. No Z-Image adapter has been published yet, so that is
// a READING of the convention rather than a verified match -- what is
// verified here is the arithmetic and the plumbing (see the
// synthesised-adapter test). When a real adapter appears and binds 0
// modules, the fix is a rename row, not this list.
//
// adaLN_modulation is deliberately absent: nothing in this lineage
// trains it, and binding it would read a file's silence as a zero.
bool
MetalZImageTransformer::bind_loras_(const std::vector<LoraSpec>& loras)
{
  if (loras.empty()) { return true; }
  if (!_lora.init(_mc, _use_mma2)) { return false; }
  if (!_lora.valid()) { return false; }
  const int nmod = _cfg.n_mod_blocks(), nctx = _cfg.n_refiner;
  int slot = 0;
  for (const LoraSpec& spec : loras) {
    if (slot >= kMaxLoraSlots || spec.path.empty()) { break; }
    std::string err;
    auto ad = lora::Adapter::open(spec.path, _mc, &err,
                                  lora::remap_ai_toolkit_module);
    if (ad == nullptr) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "MetalZImageTransformer: lora '{}' did not open: {}",
            spec.path, err.empty() ? "unreadable" : err));
      }
      continue;
    }
    // resize, not assign: Factors holds SharedBuffers and is move-only.
    _lora_mod[slot].clear();
    _lora_mod[slot].resize((std::size_t)nmod);
    _lora_ctx[slot].clear();
    _lora_ctx[slot].resize((std::size_t)nctx);
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
    for (int i = 0; i < nctx; ++i) {
      one(ctx_prefix_(i), _lora_ctx[slot][(std::size_t)i]);
    }
    _lora_bound[slot] = ad->modules();
    _lora_scale[slot] = spec.scale;
    _lora_rank_prefix[slot] = _lora_rank_total;
    _lora_rank_total += ad->max_rank();
    if (_mc->session() != nullptr) {
      _mc->session()->log_normal(
          fmt("MetalZImageTransformer: {}", ad->summary(spec.path,
                                                        spec.scale)));
      if (ad->modules() == 0) {
        // LOUD. An adapter for another model binds nothing and would
        // otherwise report success by saying nothing at all.
        _mc->session()->warn(fmt(
            "MetalZImageTransformer: lora '{}' bound NO modules -- it is "
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
MetalZImageTransformer::lora_for_(bool ctx, int L,
                                  lora::Factors BlockLora::* which) const
{
  lora::Stack st;
  for (int i = 0; i < _lora_slots; ++i) {
    const std::vector<BlockLora>& v = ctx ? _lora_ctx[i] : _lora_mod[i];
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
MetalZImageTransformer::set_lora_scale(int slot, float s)
{
  if (slot < 0 || slot >= kMaxLoraSlots) { return; }
  _lora_scale[slot] = s;
}

int
MetalZImageTransformer::lora_modules(int slot) const
{
  if (slot < 0 || slot >= kMaxLoraSlots) { return 0; }
  return _lora_bound[slot];
}

std::size_t
MetalZImageTransformer::streaming_floor_bytes(const std::string& dir)
{
  if (dir.empty()) { return 0; }
  auto wts = MetalLlamaWeights::open_model(dir);
  if (!wts.has_value()) { return 0; }
  static const char* const kStreamed[] = {"noise_refiner.", "layers."};
  std::size_t trunk = 0;
  // Keyed by the STREAM index, so noise_refiner.0 and layers.0 are
  // different blocks of one stack rather than the same one twice.
  std::vector<std::size_t> blocks;
  bool any_f32 = false;
  for (const std::string& nm : wts->tensor_names()) {
    const auto* ti = wts->info(nm);
    const std::size_t nb = ti != nullptr ? (std::size_t)ti->nbytes : 0;
    if (ti != nullptr && ti->dtype == "F32") { any_f32 = true; }
    int which = -1;
    for (int i = 0; i < 2; ++i) {
      if (nm.rfind(kStreamed[i], 0) == 0) { which = i; break; }
    }
    if (which < 0) { trunk += nb; continue; }
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
  return floor;
}

std::size_t
MetalZImageTransformer::checkpoint_block_bytes_() const
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
MetalZImageTransformer::set_residency_schedule(int steps)
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
MetalZImageTransformer::release_resident_blocks(std::size_t bytes)
{
  const std::size_t freed = _resid.release(bytes, [this]() -> std::size_t {
    return evict_tail_block_();
  });
  if (freed > 0 && _mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalZImageTransformer: released {} MB of resident blocks "
        "({} left)", freed >> 20, _resid.count()));
  }
  return freed;
}

std::uint64_t
MetalZImageTransformer::resident_bytes() const
{
  std::uint64_t n = qw_bytes_(_x_in) + _x_in_b.byte_size() +
                    _cap_norm.byte_size() + qw_bytes_(_cap_in) +
                    _cap_in_b.byte_size() + qw_bytes_(_time_1) +
                    _time_1b.byte_size() + qw_bytes_(_time_2) +
                    _time_2b.byte_size() + _x_pad.byte_size() +
                    _cap_pad.byte_size() + qw_bytes_(_final_ada) +
                    _final_ada_b.byte_size() + qw_bytes_(_final_lin) +
                    _final_lin_b.byte_size();
  // The context refiner is held in BOTH modes, so it belongs to the
  // trunk rather than to the residency ledger. Leaving it out is what
  // once made a streaming DiT report a floor it did not have.
  for (const Block& b : _ctx) { n += block_bytes_(b); }
  // `_blocks` IS the answer in both modes, and adding the residency
  // ledger to it would double every promoted block. Preloaded, it holds
  // the whole stack and `_resid` is empty; streaming, it holds exactly
  // what residency promoted.
  for (const Block& b : _blocks) { n += block_bytes_(b); }
  return n;
}

}  // namespace genai
}  // namespace vpipe
