#include "generative-models/flux3/metal-flux3-transformer.h"
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

inline float
bf16_to_f32_(std::uint16_t h)
{
  const std::uint32_t u = (std::uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline float
round_bf16_(float f)
{
  return bf16_to_f32_(f32_to_bf16_(f));
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
constexpr const char* kKey = "flux3-dit/";

// The three token modes, as the modulation tables index them.
constexpr int kModeImage = 0;
constexpr int kModeCond  = 1;
constexpr int kModeText  = 2;
constexpr const char* kModeName[3] = {"image", "image_cond", "txt"};

}  // namespace

MetalFlux3Transformer::~MetalFlux3Transformer()
{
  // GIVE THE POOL BACK: only unwire_from_pool decrements its counter,
  // so a DiT destroyed while wired would leak its share per reload.
  if (_wire.on()) {
    wire_fixed_(false);
    for (Block& b : _blocks) { wire_block_(b, false); }
  }
}

// ---- block names ----------------------------------------------------

std::string
MetalFlux3Transformer::prefix_(int L) const
{
  const int M = _cfg.n_mode;
  if (L < M) { return "mode_blocks.image." + std::to_string(L) + "."; }
  if (L < 2 * M) {
    return "txt_mode_blocks." + std::to_string(L - M) + ".";
  }
  if (L < 3 * M) {
    return "mode_blocks.image_cond." + std::to_string(L - 2 * M) + ".";
  }
  return "single_transformer_blocks." + std::to_string(L - 3 * M) + ".";
}

const std::vector<std::string>&
MetalFlux3Transformer::block_stems()
{
  static const std::vector<std::string> k = {
      "mode_blocks.image.", "txt_mode_blocks.", "mode_blocks.image_cond.",
      "single_transformer_blocks."};
  return k;
}

// ---- residency + the wired pool -------------------------------------

std::size_t
MetalFlux3Transformer::wire_block_(Block& b, bool on)
{
  std::size_t changed = 0;
  bool stop = false;
  auto one = [&](SharedBuffer& p) {
    if (stop) { return; }
    const std::size_t n = _wire.wire_one(_mc, p, on);
    // STOP AT THE FIRST REFUSAL: the pool is full, and the remaining
    // asks would each cost a syscall to be told so.
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
  QWeight* q[] = {&b.q, &b.k, &b.v, &b.o, &b.ff_in, &b.ff_out};
  for (QWeight* w : q) { qw(*w); }
  one(b.nq); one(b.nk);
  return changed;
}

std::size_t
MetalFlux3Transformer::wire_fixed_(bool on)
{
  if (!_ws) { return 0; }
  std::size_t changed = 0;
  _ws->for_each_weight([&](SharedBuffer& b) {
    changed += _wire.wire_one(_mc, b, on);
  });
  return changed;
}

std::size_t
MetalFlux3Transformer::wire_retry_slack_() const
{
  if (_wire_block_hint > 0) { return _wire_block_hint; }
  if (_resid.count() > 0) {
    return _resid.bytes() / (std::size_t)_resid.count();
  }
  return 0;
}

std::size_t
MetalFlux3Transformer::evict_tail_block_()
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
MetalFlux3Transformer::resident_pages_(std::size_t* examined,
                                       std::size_t* incore,
                                       std::size_t* paged_out) const
{
  *examined = 0;
  *incore = 0;
  if (paged_out != nullptr) { *paged_out = 0; }
  auto add = [&](const SharedBuffer& p) {
    if (p.byte_size() == 0) { return; }
    // A WIRED BUFFER CANNOT HAVE LEFT RAM.
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
    qw(b.q); qw(b.k); qw(b.v); qw(b.o); qw(b.ff_in); qw(b.ff_out);
    add(b.nq); add(b.nk);
  }
}

// ---- config ---------------------------------------------------------

bool
MetalFlux3Transformer::Config::read_dims(const std::string& dir,
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
    if (cls != "Flux3Transformer2DModel") { return false; }
  }
  auto geti = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  auto getf = [&](const char* k, float d) {
    return o.contains(k) ? (float)o.at(k).as_real(d) : d;
  };
  out->n_heads = geti("num_attention_heads", out->n_heads);
  out->head_dim = geti("attention_head_dim", out->head_dim);
  out->hidden = out->n_heads * out->head_dim;
  out->n_mode = geti("num_layers", out->n_mode);
  out->n_single = geti("num_single_layers", out->n_single);
  out->in_channels = geti("in_channels", out->in_channels);
  out->cond_in = geti("cond_in_channels", out->cond_in);
  out->joint_dim = geti("joint_attention_dim", out->joint_dim);
  out->time_freq = geti("timestep_channels", out->time_freq);
  out->norm_eps = getf("eps", out->norm_eps);
  out->rope_theta = getf("rope_theta", out->rope_theta);
  // int(dim * mlp_ratio), the reference's own arithmetic. Stated here
  // so a checkpoint whose linear_in disagrees fails at load rather than
  // at the first GEMM.
  const float ratio = getf("mlp_ratio", 3.0f);
  out->mlp_hidden = (int)((double)out->hidden * (double)ratio);
  if (o.contains("flow_map")) {
    out->flow_map = o.at("flow_map").as_bool(out->flow_map);
  }
  if (o.contains("axes_dims_rope")) {
    const FlexData a = o.at("axes_dims_rope");
    if (a.is_array()) {
      auto arr = a.as_array();
      // FOUR axes (t, h, w, l) is the model: the reference loops over
      // len(axes_dim), and a checkpoint with another count is a
      // different position scheme, not a resize of this one.
      if (arr.size() != 4) { return false; }
      for (std::size_t i = 0; i < 4; ++i) {
        out->axes[i] = (int)arr.at(i).as_int(out->axes[i]);
      }
    }
  }
  if (out->axes[0] + out->axes[1] + out->axes[2] + out->axes[3] !=
      out->head_dim) {
    return false;
  }
  // The steel attention is bd128; nothing else is built here.
  if (out->head_dim != 128) { return false; }
  return out->hidden > 0 && out->n_mode > 0 && out->n_single > 0 &&
         out->in_channels > 0 && out->joint_dim > 0;
}

// ---- weights --------------------------------------------------------

SharedBuffer
MetalFlux3Transformer::to_elt_(WeightSet& ws, const std::string& name,
                               Retain r)
{
  auto build = [&]() { return to_bf16_(ws.src(), _mc, name); };
  if (r == Retain::Streamed) { return ws.stream_derived(build); }
  return ws.derived(std::string(kKey) + "elt|" + name, build);
}

MetalFlux3Transformer::QWeight
MetalFlux3Transformer::load_qw_(WeightSet& ws, const std::string& name,
                                Retain r, bool sign)
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
  qw.w = to_elt_(ws, name + ".weight", r);
  return qw;
}

bool
MetalFlux3Transformer::load_block_(WeightSet& ws, int L, Block& b, Retain r)
{
  // Its blocks, kept or streamed (common/memory-owner.h).
  MemoryUseScope blocks_use(MemoryUse::kBlocks);
  const std::string p = prefix_(L);
  b.q = load_qw_(ws, p + "attn.to_q", r, true);
  b.k = load_qw_(ws, p + "attn.to_k", r, true);
  b.v = load_qw_(ws, p + "attn.to_v", r, true);
  // to_out is a ModuleList (Linear, Dropout), so the projection is ".0".
  b.o = load_qw_(ws, p + "attn.to_out.0", r, true);
  b.nq = to_elt_(ws, p + "attn.norm_q.weight", r);
  b.nk = to_elt_(ws, p + "attn.norm_k.weight", r);
  // [gate rows | up rows], GATE FIRST: Flux3SwiGLU (and BFL's chunk)
  // take silu of the first half. diffusers' plain `SwiGLU` is the other
  // way round, and a swapped half order is a picture, not a failure.
  b.ff_in = load_qw_(ws, p + "ff.linear_in", r, true);
  b.ff_out = load_qw_(ws, p + "ff.linear_out", r, true);
  return !b.q.empty() && !b.k.empty() && !b.v.empty() && !b.o.empty() &&
         !b.nq.empty() && !b.nk.empty() && !b.ff_in.empty() &&
         !b.ff_out.empty();
}

bool
MetalFlux3Transformer::load_fixed_(WeightSet& ws)
{
  const Retain r = Retain::Cached;
  _x_img = load_qw_(ws, "x_embedders.image", r);
  _x_cond = load_qw_(ws, "x_embedders.image_cond", r);
  _ctx_in = load_qw_(ws, "context_embedder", r);
  _time_1 = load_qw_(ws, "timestep_embedder.linear_1", r);
  _time_2 = load_qw_(ws, "timestep_embedder.linear_2", r);
  bool ok = !_x_img.empty() && !_x_cond.empty() && !_ctx_in.empty() &&
            !_time_1.empty() && !_time_2.empty();
  if (_cfg.flow_map) {
    _tgt_1 = load_qw_(ws, "target_timestep_embedder.linear_1", r);
    _tgt_2 = load_qw_(ws, "target_timestep_embedder.linear_2", r);
    ok = ok && !_tgt_1.empty() && !_tgt_2.empty();
  }
  for (int m = 0; m < 3; ++m) {
    _early_mod[m] = load_qw_(
        ws, std::string("early_stream_modulations.") + kModeName[m] +
                ".linear", r);
    _single_mod[m] = load_qw_(
        ws, std::string("single_stream_modulations.") + kModeName[m] +
                ".linear", r);
    ok = ok && !_early_mod[m].empty() && !_single_mod[m].empty();
  }
  // (scale | shift), in THAT order: diffusers' Flux3AdaLayerNorm chunks
  // scale first, and its checkpoint stores the halves for it -- BFL's own
  // file holds the same weight with the halves swapped.
  _out_mod = load_qw_(ws, "norm_out.linear", r);
  _proj_out = load_qw_(ws, "proj_out", r);
  return ok && !_out_mod.empty() && !_proj_out.empty();
}

std::size_t
MetalFlux3Transformer::qw_bytes_(const QWeight& w)
{
  if (w.quantized) {
    return w.codes.byte_size() + w.scales.byte_size() + w.qbias.byte_size();
  }
  return w.w.byte_size();
}

std::size_t
MetalFlux3Transformer::block_bytes_(const Block& b)
{
  return qw_bytes_(b.q) + qw_bytes_(b.k) + qw_bytes_(b.v) +
         qw_bytes_(b.o) + b.nq.byte_size() + b.nk.byte_size() +
         qw_bytes_(b.ff_in) + qw_bytes_(b.ff_out);
}

void
MetalFlux3Transformer::each_block_tensor_(
    int L, Block& b, const BlockSlots<Block>::TensorFn& fn) const
{
  const std::string p = prefix_(L);
  auto qw = [&](QWeight& w, const char* nm) {
    if (w.quantized) {
      fn(p + nm + ".weight", w.codes, Placement::kRaw);
      fn(p + nm + ".scales", w.scales, Placement::kBf16);
      fn(p + nm + ".biases", w.qbias, Placement::kBf16);
    } else {
      fn(p + nm + ".weight", w.w, Placement::kBf16);
    }
  };
  qw(b.q, "attn.to_q");
  qw(b.k, "attn.to_k");
  qw(b.v, "attn.to_v");
  qw(b.o, "attn.to_out.0");
  fn(p + "attn.norm_q.weight", b.nq, Placement::kBf16);
  fn(p + "attn.norm_k.weight", b.nk, Placement::kBf16);
  // linear_in lands RAW, gate rows then up rows, and is READ that way:
  // the split SwiGLU takes the halves where they lie, so no product
  // without a checkpoint name exists and a raw read refills it.
  qw(b.ff_in, "ff.linear_in");
  qw(b.ff_out, "ff.linear_out");
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
MetalFlux3Transformer::clone_block_(const Block& src, Block& dst,
                                    bool copy) const
{
  // Its blocks, kept or streamed (common/memory-owner.h).
  MemoryUseScope blocks_use(MemoryUse::kBlocks);
  bool ok = true;
  auto cq = [&](const QWeight& s, QWeight& d) {
    d.quantized = s.quantized;
    d.bits = s.bits;
    d.sgn = s.sgn;
    ok = clone_buf_(_mc, s.w, d.w, copy) && ok;
    ok = clone_buf_(_mc, s.codes, d.codes, copy) && ok;
    ok = clone_buf_(_mc, s.scales, d.scales, copy) && ok;
    ok = clone_buf_(_mc, s.qbias, d.qbias, copy) && ok;
  };
  cq(src.q, dst.q); cq(src.k, dst.k); cq(src.v, dst.v); cq(src.o, dst.o);
  ok = clone_buf_(_mc, src.nq, dst.nq, copy) && ok;
  ok = clone_buf_(_mc, src.nk, dst.nk, copy) && ok;
  cq(src.ff_in, dst.ff_in);
  cq(src.ff_out, dst.ff_out);
  return ok;
}

SharedBuffer
MetalFlux3Transformer::rebuild_one_(const std::string& nm, Placement how)
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
MetalFlux3Transformer::fill_narrow_(const std::string& nm,
                                    const SharedBuffer& dst)
{
  // An F32 checkpoint's bytes are twice the width of the bf16 buffer
  // the forward reads, so a raw read has nowhere to land: read it into
  // one scratch for the run and narrow into the slot.
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
MetalFlux3Transformer::configure_slots_()
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
    return _ws != nullptr && load_block_(*_ws, L, b, Retain::Streamed);
  };
  ops.clone = [this](const Block& s, Block& d, bool copy) {
    return clone_block_(s, d, copy);
  };
  ops.bytes = [](const Block& b) { return block_bytes_(b); };
  ops.empty = [](const Block& b) { return b.q.empty(); };
  // Every quantized projection just landed RAW, and converts to the
  // signed layout here, on the reader thread, under the GPU's work.
  ops.post_refill = [this](int, Block& b) {
    for (QWeight* w : {&b.q, &b.k, &b.v, &b.o, &b.ff_in, &b.ff_out}) {
      if (!sign_(*w)) { return false; }
    }
    return true;
  };
  ops.same_layout = [this](int L, const Block& b) {
    return same_layout_(L, b);
  };
  _slots.set_weight_set(_ws.get());
  _slots.configure(_mc, std::move(ops), "MetalFlux3Transformer",
                   "VPIPE_FLUX3_NO_SLOTS");
}

// A slot built for a block of the OTHER width -- a mixed pack holds
// whole joint blocks at w8 and the rest at w4, and one slot pair streams
// the per-mode and joint blocks alike -- is rebuilt rather than refilled
// (BlockSlots::Ops::same_layout): the refill would place the codes and
// keep the slot's width and signed flag.
bool
MetalFlux3Transformer::same_layout_(int L, const Block& b) const
{
  if (_quant_bits == 0 || _ws == nullptr) { return true; }
  const std::string p = prefix_(L);
  const std::pair<const QWeight*, const char*> all[] = {
      {&b.q, "attn.to_q"},          {&b.k, "attn.to_k"},
      {&b.v, "attn.to_v"},          {&b.o, "attn.to_out.0"},
      {&b.ff_in, "ff.linear_in"},   {&b.ff_out, "ff.linear_out"}};
  for (const auto& [w, nm] : all) {
    if (stored_affine_bits(*_ws, p + nm, _quant_group) !=
        (w->quantized ? w->bits : 0)) {
      return false;
    }
  }
  return true;
}

bool
MetalFlux3Transformer::sign_(QWeight& w) const
{
  // A weight this model keeps unsigned (dense, the layout off) has
  // nothing to do. A signed one in a slot was just REFILLED raw, whatever
  // its flag says from the build, so it converts unconditionally.
  if (!w.quantized || !_sgn || !_i8 || !_i8->signed_available(w.bits)) {
    return true;
  }
  if (!genai::to_signed_affine(w.codes, w.scales, w.qbias, w.bits,
                               /*bf16=*/true)) {
    return false;
  }
  w.sgn = true;
  return true;
}

const MetalFlux3Transformer::QWeight*
MetalFlux3Transformer::signed_dense_(ComputeEncoder& enc, const QWeight& w,
                                     QWeight& view, int N, int K) const
{
  if (!w.quantized || !w.sgn) { return &w; }
  const std::size_t need = (std::size_t)N * (std::size_t)K * 2;
  if (_w_deq.empty() || _w_deq.byte_size() < need) {
    _w_deq = _mc->make_shared_buffer(need);
    if (_w_deq.empty()) { return nullptr; }
  }
  if (!_i8 || !_i8->encode_dequant_signed(enc, w.codes, w.scales, w.qbias,
                                          w.bits, _w_deq, N, K)) {
    return nullptr;
  }
  view = QWeight{};
  view.w = _w_deq.subview(0, need);
  if (view.w.empty()) { return nullptr; }
  return &view;
}

// ---- load -----------------------------------------------------------

std::unique_ptr<MetalFlux3Transformer>
MetalFlux3Transformer::load(const std::string& model_dir, MetalCompute* mc,
                            const Config& cfg, bool stream_blocks,
                            const std::vector<LoraSpec>& loras)
{
  return load(WeightSet::open(model_dir, nullptr), mc, cfg, stream_blocks,
              loras);
}

std::unique_ptr<MetalFlux3Transformer>
MetalFlux3Transformer::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                            const Config& cfg, bool stream_blocks,
                            const std::vector<LoraSpec>& loras)
{
  if (ws == nullptr || mc == nullptr) { return nullptr; }
  std::unique_ptr<MetalFlux3Transformer> m(new MetalFlux3Transformer());
  m->_mc = mc;
  m->_cfg = cfg;
  m->_ws = std::move(ws);
  m->_stream_blocks = stream_blocks;
  auto refuse = [&](const std::string& why) {
    if (mc->session() != nullptr) {
      mc->session()->error(fmt("MetalFlux3Transformer: {}", why));
    }
    return nullptr;
  };

  {
    const MetalLlamaWeights& w = m->_ws->src();
    const std::string p0 = m->prefix_(0);
    const auto* si = w.info(p0 + "attn.to_q.scales");
    const auto* ci = w.info(p0 + "attn.to_q.weight");
    if (ci == nullptr) {
      return refuse("no " + p0 + "attn.to_q.weight -- not a FLUX.3 DiT "
                    "in the diffusers layout");
    }
    // Quantization is read off the checkpoint: a pack carries `.scales`
    // beside its codes and a dense one does not.
    if (si != nullptr && si->shape.size() == 2 && ci->shape.size() == 2 &&
        si->shape[1] > 0) {
      m->_quant_group = (int)(m->_cfg.hidden / si->shape[1]);
      if (m->_quant_group <= 0) { m->_quant_group = 64; }
      const long K = si->shape[1] * (long)m->_quant_group;
      m->_quant_bits = K > 0 ? (int)(ci->shape[1] * 32 / K) : 0;
      if (m->_quant_bits != 4 && m->_quant_bits != 8) { m->_quant_bits = 0; }
    }
    if (m->_quant_bits == 0) {
      // BFL's fp8r variant stores its projections as float8 codes with a
      // per-row scale. That is a different loader, not a dtype this one
      // can narrow -- refuse it by name rather than reading the codes as
      // something else.
      if (ci->dtype.rfind("F8", 0) == 0) {
        return refuse("this checkpoint stores FP8 weights (the fp8r "
                      "variant), which this build does not read -- use the "
                      "bf16 transformer/");
      }
      m->_src_f32 = ci->dtype == "F32";
    }
    // A FLOW-MAP checkpoint has the target embedder and a plain one does
    // not -- the tensors decide, as the reference's own loader does
    // (checkpoint.is_flow_map), whatever config.json says.
    const bool has_tgt =
        w.info("target_timestep_embedder.linear_1.weight") != nullptr;
    if (has_tgt != m->_cfg.flow_map) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt(
            "MetalFlux3Transformer: config.json says flow_map {} but the "
            "tensors say {}; the tensors win", m->_cfg.flow_map, has_tgt));
      }
      m->_cfg.flow_map = has_tgt;
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
  // every GEMM in the model.
  m->_fn_gemm_bm64 = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_gemm_bm64bn64 = m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  m->_fn_ln_mod = m->_lib_elt.function("layernorm_modulate_f16");
  m->_fn_gated = m->_lib_elt.function("gated_residual_f16");
  m->_fn_gated4 = m->_lib_elt.function("gated_residual_v4_f16");
  m->_fn_swiglu_halves = m->_lib_elt.function("swiglu_split_gate_first_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_residual4 = m->_lib_elt.function("residual_add_v4_f16");
  m->_fn_silu = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_rms = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_tr_rope = m->_lib_rope.function("transpose_rope_pair_ftab_f16");
  m->_fn_tr_rms_rope =
      m->_lib_rope.function("transpose_rms_rope_pair_ftab_f16");
  // THE QUANTIZED ENTRY POINTS ARE NAMED BY THEIR GROUP -- no `_f16`
  // suffix. Only loaded for a quantized checkpoint.
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
  // its dispatch, which in a 43-block stack is a picture, not a failure.
  if (!m->_fn_gemm_t.valid() || !m->_fn_ln_mod.valid() ||
      !m->_fn_gated.valid() || !m->_fn_swiglu_halves.valid() ||
      !m->_fn_rms.valid() || !m->_fn_tr_rope.valid() ||
      !m->_fn_residual.valid() || !m->_fn_silu.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_tr_rms_rope.valid()) {
    return refuse("a required kernel did not build");
  }
  m->_fused_qk = std::getenv("VPIPE_FLUX3_UNFUSED_QK_NORM") == nullptr;
  if (m->_quant_bits > 0 &&
      (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid())) {
    return refuse(fmt("no quantized GEMM for group {} -- this checkpoint "
                      "cannot be run (re-quantize at group 64)",
                      m->_quant_group)());
  }
  m->_steel_attn_ok = m->_lib_attn.valid();
  m->_use_attn_nax = m->_steel_attn_ok && m->_lib_attn_nax.valid() &&
                     mc->supports_matrix_cores() &&
                     std::getenv("VPIPE_FLUX3_NO_NAX_ATTN") == nullptr;
  m->_use_mma2 = mc->supports_matrix_cores() && m->_fn_gemm_mma.valid() &&
                 std::getenv("VPIPE_FLUX3_NO_MMA2") == nullptr;
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm,
                                              /*bf16=*/true);
    i8->set_autotune(cfg.autotune);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }
  if (const char* e = std::getenv("VPIPE_FLUX3_SOL")) {
    m->_cfg.sol.enabled = std::atoi(e) != 0;
  }
  // HIGHER tau keeps FEWER blocks: it is a threshold, not a budget.
  if (const char* e = std::getenv("VPIPE_FLUX3_SOL_TAU")) {
    m->_cfg.sol.tau = (float)std::atof(e);
  }
  if (const char* e = std::getenv("VPIPE_FLUX3_ANE")) {
    m->_cfg.ane_ffn = std::atoi(e) != 0;
  }
  if (const char* e = std::getenv("VPIPE_FLUX3_ANE_ROWS")) {
    m->_cfg.ane_rows = (float)std::atof(e);
  }
  if (const char* e = std::getenv("VPIPE_FLUX3_ANE_QKV")) {
    m->_cfg.ane_qkv = std::atoi(e) != 0;
  }
  if (m->_cfg.ane_qkv && mc->session() != nullptr) {
    // Said, not silently dropped: see ane_qkv_setup_ for why this family
    // gives the one ANE worker to the feed-forward alone.
    mc->session()->log_normal(fmt(
        "MetalFlux3Transformer: ane_qkv is not built for FLUX.3 -- the ANE "
        "takes the joint blocks' feed-forward only"));
  }
  if (m->_cfg.sol.enabled) {
    std::string serr;
    m->_sol = MetalSolAttention::load(mc, /*bf16=*/true, &serr);
    if (m->_sol == nullptr && mc->session() != nullptr) {
      mc->session()->log_normal(fmt(
          "MetalFlux3Transformer: sol_attn unavailable -- {}",
          serr.empty() ? "kernels did not load" : serr));
    }
  }
  {
    if (const char* e = std::getenv("VPIPE_FLUX3_SAGE")) {
      m->_cfg.sage.enabled = std::atoi(e) != 0;
    }
    bool sage_fatal = false;
    m->_sage = MetalSageAttention::load_for_model(
        mc, /*bf16=*/true, m->_cfg.sage, "MetalFlux3Transformer",
        &sage_fatal);
    if (sage_fatal) { return nullptr; }
    if (m->_sage && !m->_use_attn_nax) {
      if (mc->session() != nullptr) {
        mc->session()->log_normal(fmt(
            "MetalFlux3Transformer: sage_attn is off -- the matrix-core "
            "flash entry is unavailable here"));
      }
      m->_sage.reset();
    }
    m->_sage_reserve =
        std::getenv("VPIPE_FLUX3_SAGE_NO_RESERVE") == nullptr;
  }
  if (const char* t = std::getenv("VPIPE_FLUX3_GEMM_TILE")) {
    m->_gemm_tile = std::atoi(t);
  }
  if (m->_gemm_tile == 2 && !m->_fn_gemm_bm64bn64.valid()) {
    m->_gemm_tile = 1;
  }
  if (m->_gemm_tile == 1 && !m->_fn_gemm_bm64.valid()) { m->_gemm_tile = 0; }

  if (!m->load_fixed_(*m->_ws)) {
    return refuse("the model-level weights are incomplete");
  }

  // THE SIGNED LAYOUT for the block projections (see _sgn): decided
  // before any block loads. Its GEMM runs only on the matrix-core route,
  // and the ANE tier's stager reads the codes unsigned.
  m->_sgn = m->_i8 && m->_use_mma2 && m->_quant_bits > 0 &&
            m->_quant_group == 64 &&
            !m->_cfg.ane_ffn && !m->_cfg.ane_qkv &&
            (m->_i8->signed_available(8) || m->_i8->signed_available(4));
  if (m->_sgn && mc->session() != nullptr) {
    mc->session()->info(fmt(
        "MetalFlux3Transformer: the quantized block projections take the "
        "signed int8 layout"));
  }
  m->_wire.open(mc);
  const int NB = m->n_blocks();
  // Sized to the stack in BOTH modes: preloaded it holds every block,
  // streaming it holds what residency promoted, and an unfilled entry
  // reads as empty.
  m->_blocks.resize((std::size_t)NB);
  if (stream_blocks) {
    m->configure_slots_();
  } else {
    for (int L = 0; L < NB; ++L) {
      if (!m->load_block_(*m->_ws, L, m->_blocks[(std::size_t)L],
                          Retain::Cached)) {
        return refuse(
            fmt("block {} ({}) is incomplete", L, m->prefix_(L))());
      }
    }
  }
  if (!m->bind_loras_(loras) && mc->session() != nullptr) {
    mc->session()->warn(fmt(
        "MetalFlux3Transformer: the LoRA applier did not load; the "
        "adapters are ignored and the base model runs"));
  }
  if (mc->session() != nullptr) {
    mc->session()->log_normal(fmt(
        "MetalFlux3Transformer: {} blocks (3 x {} per-mode + {} joint), "
        "hidden {}, {} heads x {}, mlp {}, {}{}{}{}{}",
        NB, m->_cfg.n_mode, m->_cfg.n_single, m->_cfg.hidden,
        m->_cfg.n_heads, m->_cfg.head_dim, m->_cfg.mlp_hidden,
        m->_quant_bits > 0
            ? "w" + std::to_string(m->_quant_bits) + " g" +
                  std::to_string(m->_quant_group)
            : std::string(m->_src_f32 ? "f32 -> bf16" : "bf16"),
        m->_cfg.flow_map ? ", flow map" : "",
        stream_blocks ? ", streaming" : "",
        m->_use_mma2 ? ", matmul2d" : "",
        // WHETHER THE TIER TOOK, not whether it was asked for.
        m->_i8            ? ", i8"
        : m->_cfg.i8_gemm ? ", i8 REQUESTED BUT UNAVAILABLE"
                          : ""));
  }
  return m;
}

// ---- the reference's sampler arithmetic ------------------------------

namespace flux3 {

std::vector<float>
schedule(int steps, float alpha)
{
  // NO FMA. Clang contracts `1 + b * s` into one fused multiply-add on
  // arm64, which rounds once where torch rounds twice -- and that moved
  // 2 of the 16-step schedule's 17 nodes by an f32 ulp, which is all a
  // bf16 rounding needs to land a node on the other side of a tie.
#pragma clang fp contract(off)
  if (steps < 1) { return {}; }
  // torch.linspace(1.0, 0.0, steps + 1) in f32, its two-sided formula:
  // the first half counts up from start, the second down from end.
  const int n = steps + 1;
  const float start = 1.0f, end = 0.0f;
  const float step = (end - start) / (float)(n - 1);
  const int halfway = n / 2;
  // alpha * t / (1.0 + (alpha - 1.0) * t), the python scalars cast to
  // f32 the way torch's tensor-scalar arithmetic casts them.
  const float a = alpha;
  const float am1 = (float)((double)alpha - 1.0);
  std::vector<float> out((std::size_t)n);
  for (int i = 0; i < n; ++i) {
    const float s = i < halfway ? start + step * (float)i
                                : end - step * (float)(n - i - 1);
    out[(std::size_t)i] = (a * s) / (1.0f + am1 * s);
  }
  return out;
}

float
quantized_time(float t)
{
  // `packed.timesteps` is bf16, and timestep_embedding multiplies it by
  // 1000 in that dtype: two roundings, the second at 4-unit resolution
  // between 512 and 1024.
  return round_bf16_(round_bf16_(t) * 1000.0f);
}

std::vector<float>
time_embedding(float t, int dim)
{
  const int half = dim / 2;
  const float tq = quantized_time(t);
  std::vector<float> out((std::size_t)dim);
  // -log(10000) * arange(half) / half, in f32 as the reference forms it.
  const float nl = (float)(-std::log(10000.0));
  for (int i = 0; i < half; ++i) {
    const float f = std::exp((nl * (float)i) / (float)half);
    const float a = tq * f;
    out[(std::size_t)i] = round_bf16_(std::cos(a));
    out[(std::size_t)(half + i)] = round_bf16_(std::sin(a));
  }
  return out;
}

bool
is_bfl_root_layout(const std::string& root)
{
  if (root.empty()) { return false; }
  std::error_code ec;
  const fs::path r(root);
  // The diffusers layout wins wherever both are present (the published
  // repo ships both): only a root WITHOUT transformer/ is the BFL one.
  if (fs::exists(r / "transformer" / "config.json", ec)) { return false; }
  auto read = [&](const char* name, FlexData* out) {
    std::ifstream in(r / name);
    if (!in) { return false; }
    try {
      *out = FlexData::from_json(in);
    } catch (...) {
      return false;
    }
    return out->is_object();
  };
  FlexData fd;
  if (read("manifest.json", &fd)) {
    auto o = fd.as_object();
    if (o.contains("kind") &&
        std::string(o.at("kind").as_string("")) == "f3_dit_export") {
      return true;
    }
  }
  if (read("config.json", &fd)) {
    auto o = fd.as_object();
    if (o.contains("model")) {
      const std::string m(o.at("model").as_string(""));
      if (m.rfind("flux-3", 0) == 0) { return true; }
    }
  }
  return false;
}

std::string
bfl_root_layout_message(const std::string& root)
{
  return "'" + root + "' is BFL's own single-file layout of FLUX.3 Image "
         "Klein (model.safetensors + ae.safetensors at the root, as BFL's "
         "README downloads it). vpipe reads the diffusers layout of the "
         "same weights -- transformer/ and vae/ beside text_encoder/ -- "
         "which model-fetch fetches for black-forest-labs/"
         "flux-3-image-klein";
}

}  // namespace flux3

// ---- forward --------------------------------------------------------

SharedBuffer
MetalFlux3Transformer::forward(const Request& req, std::string* err)
{
  auto fail = [&](const std::string& m) {
    if (err != nullptr) { *err = m; }
    return SharedBuffer{};
  };
  if (req.latents == nullptr || req.text == nullptr) {
    return fail("flux3 forward: missing input");
  }
  const int H = _cfg.hidden, NH = _cfg.n_heads, Hd = _cfg.head_dim;
  const int MLP = _cfg.mlp_hidden, IC = _cfg.in_channels;
  const int CI = _cfg.cond_in, JD = _cfg.joint_dim;
  const int TXT = req.text_rows;
  const int IMG = req.grid_h * req.grid_w;
  int COND = 0;
  for (const Ref& r : req.refs) {
    if (r.latents == nullptr || r.grid_h <= 0 || r.grid_w <= 0) {
      return fail("flux3 forward: an empty reference");
    }
    const std::size_t need = (std::size_t)r.grid_h * r.grid_w * CI * 2;
    if (r.latents->byte_size() < need) {
      return fail("flux3 forward: a reference's latents are shorter than "
                  "its grid");
    }
    COND += r.grid_h * r.grid_w;
  }
  const int JT = TXT + IMG + COND;
  const int R_IMG = TXT, R_COND = TXT + IMG;   // first row of each mode
  if (TXT <= 0 || IMG <= 0 || NH * Hd != H) {
    return fail("flux3 forward: degenerate geometry");
  }
  if (req.latents->byte_size() < (std::size_t)IMG * IC * 2 ||
      req.text->byte_size() < (std::size_t)TXT * JD * 2) {
    return fail("flux3 forward: an input is shorter than its shape");
  }
  if (!_steel_attn_ok) {
    return fail("flux3 forward: the steel attention kernel is "
                "unavailable");
  }

  // LLM-lane perf event: one DiT forward per sampler step. Value = the
  // joint sequence length, which is what the cost scales with.
  PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDit,
                     kPerfLlmDitBegin, (std::uint64_t)JT);

  MetalCompute* mc = _mc;
  const float eps = _cfg.norm_eps;
  auto buf = [&](std::size_t elems) {
    return mc->make_shared_buffer(elems * 2);
  };
  // Scratch: ten [JT, H] planes and the [JT, 3 MLP] feed-forward pair.
  // `qb` is dead once rope has written its head-major copy, so it carries
  // the attention's transposed output.
  //
  // THE MLP'S OUTPUT GETS ITS OWN PLANE, and that is a correctness rule,
  // not a budget choice: under the ANE split the ANE writes its rows of
  // it CONCURRENTLY with the GPU's q/k/v GEMMs. Aliased to `kb` -- which
  // was free in a GPU-only block -- the K projection and the ANE's rows
  // raced for the same bytes, and the picture came out with ghosts.
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
  SharedBuffer ffi = buf((std::size_t)JT * 2 * MLP);
  SharedBuffer ffh = buf((std::size_t)JT * MLP);
  SharedBuffer ob2 = buf((std::size_t)JT * H);   // the MLP's contribution
  SharedBuffer& ab = qb;     // attention out, token-major
  const int TF = _cfg.time_freq;
  SharedBuffer t_img = buf((std::size_t)TF);
  SharedBuffer t_zero = buf((std::size_t)TF);
  SharedBuffer t_tgt = buf((std::size_t)TF);
  SharedBuffer te1 = buf((std::size_t)H);
  SharedBuffer v_img = buf((std::size_t)H);
  SharedBuffer v_zero = buf((std::size_t)H);
  SharedBuffer v_tgt = buf((std::size_t)H);
  SharedBuffer s_img = buf((std::size_t)H);
  SharedBuffer s_zero = buf((std::size_t)H);
  // [mode][shift | scale | gate], early and single.
  SharedBuffer emods = buf((std::size_t)3 * 3 * H);
  SharedBuffer smods = buf((std::size_t)3 * 3 * H);
  SharedBuffer outmod = buf((std::size_t)2 * H);      // scale | shift
  SharedBuffer vel = buf((std::size_t)IMG * IC);
  if (joint.empty() || nrm.empty() || ob.empty() || qb.empty() ||
      kb.empty() || vb.empty() || qh.empty() || kh.empty() || vh.empty() ||
      ao.empty() || ffi.empty() || ffh.empty() || ob2.empty() ||
      emods.empty() ||
      smods.empty() || vel.empty() || t_img.empty() || te1.empty()) {
    return fail("flux3 forward: scratch allocation failed");
  }

  // ---- the rotary tables -----------------------------------------------
  //
  // f32, angles in f64 to match the reference's float64 freqs; written
  // TWICE per pair because the kernel rotates INTERLEAVED pairs
  // (x[2k], x[2k+1]) and reads one entry per element.
  SharedBuffer rcos =
      mc->make_shared_buffer((std::size_t)JT * Hd * sizeof(float));
  SharedBuffer rsin =
      mc->make_shared_buffer((std::size_t)JT * Hd * sizeof(float));
  if (rcos.empty() || rsin.empty()) {
    return fail("flux3 forward: rope allocation failed");
  }
  {
    auto* c = static_cast<float*>(rcos.contents());
    auto* s = static_cast<float*>(rsin.contents());
    const int pairs = Hd / 2;
    std::vector<double> freq((std::size_t)pairs);
    std::vector<int> axis((std::size_t)pairs);
    for (int a = 0, p = 0; a < 4; ++a) {
      for (int j = 0; j < _cfg.axes[a] / 2; ++j, ++p) {
        freq[(std::size_t)p] = 1.0 / std::pow((double)_cfg.rope_theta,
                                              (double)(2 * j) /
                                                  (double)_cfg.axes[a]);
        axis[(std::size_t)p] = a;
      }
    }
    auto row = [&](int t, double p0, double p1, double p2, double p3) {
      const double pos[4] = {p0, p1, p2, p3};
      for (int p = 0; p < pairs; ++p) {
        const double ang = pos[axis[(std::size_t)p]] * freq[(std::size_t)p];
        const float cb = (float)std::cos(ang);
        const float sb = (float)std::sin(ang);
        const std::size_t o = (std::size_t)t * Hd + 2 * p;
        c[o] = cb; c[o + 1] = cb;
        s[o] = sb; s[o + 1] = sb;
      }
    };
    // Text: (0, 0, 0, row) -- the PAD rows too, at their own rows.
    for (int i = 0; i < TXT; ++i) { row(i, 0, 0, 0, (double)i); }
    // Image: (0, r/2, c/2, 0). HALF the grid index: the checkpoint's
    // image_rope_scale, applied by the pipeline before the ids reach it.
    for (int r = 0; r < req.grid_h; ++r) {
      for (int q = 0; q < req.grid_w; ++q) {
        row(R_IMG + r * req.grid_w + q, 0, 0.5 * r, 0.5 * q, 0);
      }
    }
    // Reference n: the same half grid, told apart on l = -(1 + n).
    int at = R_COND;
    for (std::size_t n = 0; n < req.refs.size(); ++n) {
      const Ref& rf = req.refs[n];
      for (int r = 0; r < rf.grid_h; ++r) {
        for (int q = 0; q < rf.grid_w; ++q) {
          row(at++, 0, 0.5 * r, 0.5 * q, -(1.0 + (double)n));
        }
      }
    }
  }

  // ---- the three timestep embeddings, on the host --------------------
  {
    auto put = [&](const SharedBuffer& b, const std::vector<float>& v) {
      auto* d = static_cast<std::uint16_t*>(b.contents());
      for (std::size_t i = 0; i < v.size(); ++i) { d[i] = f32_to_bf16_(v[i]); }
    };
    put(t_img, flux3::time_embedding(req.t, TF));
    put(t_zero, flux3::time_embedding(0.0f, TF));
    if (_cfg.flow_map) { put(t_tgt, flux3::time_embedding(req.t_next, TF)); }
  }

  if (_i8) { _i8->tune_pending(mc); }

  CommandStream stream = mc->make_command_stream();
  ComputeEncoder enc = stream.begin_compute();
  if (!enc.valid()) { return fail("flux3 forward: no compute encoder"); }
  _flops.begin();
  if (_sol) {
    _sol->reset_counts();
    _sol_flops.begin();
  }

  // Per-stage TRACE, for locating a numeric gap against the reference.
  // It COMMITS, so a traced run is not a timing run.
  const char* trace_dir = std::getenv("VPIPE_FLUX3_TRACE");
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
      for (std::size_t i = 0; i < n; ++i) { v[i] = bf16_to_f32_(sp[i]); }
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

  // Per-CATEGORY profile. READ IT WITH ITS MARK COUNT: each mark is a
  // commit AND a wait, so a bucket with many marks and little work is
  // mostly measuring the profiler -- and it defeats the prefetch, so the
  // stream bucket is only honest from VPIPE_FLUX3_STREAM_PROFILE.
  const bool prof = [] {
    const char* e = std::getenv("VPIPE_FLUX3_DIT_PROFILE");
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
    // The adapter accumulates ONTO the base result, after whichever
    // base route ran.
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
      // THREADGROUP-indexed: a thread-indexed grid computes a fraction
      // of the output and leaves the rest as it found it.
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
  auto rms = [&](const SharedBuffer& x, const SharedBuffer& w,
                 const SharedBuffer& y, int R, int D) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, x); enc.set_buffer(1, w); enc.set_buffer(2, y);
    enc.set_constant(3, D); enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  auto silu = [&](const SharedBuffer& x, const SharedBuffer& y, int n) {
    enc.set_function(_fn_silu);
    enc.set_buffer(0, x); enc.set_buffer(1, x); enc.set_buffer(2, y);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
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
  // out[r] = (1 + scale) * LN(x[r]) + shift, in one pass. `mod` holds a
  // mode's (shift | scale | gate) from element `me` on.
  auto ln_mod = [&](const SharedBuffer& x, std::size_t xe,
                    const SharedBuffer& mod, std::size_t me,
                    const SharedBuffer& y, std::size_t ye, int R) {
    if (R <= 0) { return; }
    enc.set_function(_fn_ln_mod);
    enc.set_buffer(0, x, xe * 2);
    enc.set_buffer(1, mod, (me + (std::size_t)H) * 2);   // scale
    enc.set_buffer(2, mod, me * 2);                      // shift
    enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, H); enc.set_constant(5, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  // h[r] += gate * sub[r], gate = the mode's third chunk.
  auto gated = [&](const SharedBuffer& h, std::size_t he,
                   const SharedBuffer& mod, std::size_t me,
                   const SharedBuffer& sub, std::size_t se, int R) {
    if (R <= 0) { return; }
    const std::size_t ge = me + 2 * (std::size_t)H;
    if ((H % 4) == 0 && _fn_gated4.valid()) {
      enc.set_function(_fn_gated4);
      enc.set_buffer(0, h, he * 2); enc.set_buffer(1, mod, ge * 2);
      enc.set_buffer(2, sub, se * 2);
      enc.set_constant(3, H / 4); enc.set_constant(4, R);
      enc.dispatch({(unsigned)(H / 4), (unsigned)R, 1}, {256, 1, 1});
      return;
    }
    enc.set_function(_fn_gated);
    enc.set_buffer(0, h, he * 2); enc.set_buffer(1, mod, ge * 2);
    enc.set_buffer(2, sub, se * 2);
    enc.set_constant(3, H);
    enc.set_constant(4, (int)((std::size_t)R * H));
    enc.dispatch({(unsigned)((std::size_t)R * H), 1, 1}, {256, 1, 1});
  };
  // silu(in[r, :MLP]) * in[r, MLP:], GATE FIRST.
  auto swiglu = [&](const SharedBuffer& in, const SharedBuffer& out, int R) {
    enc.set_function(_fn_swiglu_halves);
    enc.set_buffer(0, in); enc.set_buffer(1, out);
    enc.set_constant(2, R); enc.set_constant(3, MLP);
    enc.dispatch({(unsigned)((std::size_t)R * MLP), 1, 1}, {256, 1, 1});
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
  // rms(x) * gamma, then the pair rotation, in f32 with ONE rounding:
  // token-major [rows, NH, Hd] in, head-major [NH, rows, Hd] out.
  auto tr_rms_rope = [&](const SharedBuffer& in, const SharedBuffer& gamma,
                         const SharedBuffer& out, int rows, int row0) {
    const std::size_t tb = (std::size_t)row0 * Hd * sizeof(float);
    enc.set_function(_fn_tr_rms_rope);
    enc.set_buffer(0, in); enc.set_buffer(1, out);
    enc.set_buffer(2, rcos, tb); enc.set_buffer(3, rsin, tb);
    enc.set_buffer(4, gamma);
    enc.set_constant(5, NH); enc.set_constant(6, rows);
    enc.set_constant(7, Hd); enc.set_constant(8, eps);
    enc.dispatch({32, (unsigned)rows, (unsigned)NH}, {32, 1, 1});
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
  // Dense, bidirectional, one dispatch per block over that block's rows.
  // Four sequence lengths run (text, image, references, joint), so the
  // params and the specialized function are memoized per length.
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
    MetalSageAttention::pv8_constants(
        fc, i8 && _sage && _sage->pv8_for(_cfg.sage));
    metal_compute::ComputeFunction fn =
        nax ? _lib_attn_nax.function("attn_steel_nax_h_bd128_bf16", fc)
            : _lib_attn.function("attn_steel_h_bd128_bf16", fc);
    auto ins = _attn_fn.emplace(key, std::move(fn));
    return ins.first->second.valid() ? &ins.first->second : nullptr;
  };
  bool attn_ok = true;
  // `joint_idx` is the block's index within the joint stack, -1 for a
  // per-mode block: Sol routes the joint attention, whose prompt rows
  // are its exact sink.
  auto run_attn = [&](int joint_idx, int rows) {
    if (!attn_ok) { return; }
    // NOT gated on the matrix-core path: Sol has an ALU arm, and on an
    // M4 it is where the attention's cost is (bit-identical to dense with
    // every block kept).
    if (_sol && _cfg.sol.enabled && joint_idx >= 0 &&
        joint_idx >= _cfg.sol.dense_layers && rows >= kSolMinSegment) {
      const int blk = _cfg.sol.key_block > 0 ? _cfg.sol.key_block
                                             : sol::kBlock;
      if ((rows + blk - 1) / blk >= 16) {
        sol::Config sc = _cfg.sol;
        // THE PROMPT IS THE SINK: a centroid over text rows is a prompt
        // half-read. It sits at row 0 of the joint sequence.
        sc.sink_start = 0;
        sc.sink_tokens = TXT;
        MetalSolAttention::Band band{rows, rows, rows, rows, 0, 0};
        std::string serr;
        if (_sol->encode(enc, qh, kh, vh, ao, NH, NH, band, Hd, ascale, sc,
                         &serr)) {
          _flops.add(_sol_flops.routed(
              attention_flops(NH, (long long)rows * rows, Hd),
              _sol->total_blocks()));
          return;
        }
        if (_mc->session() != nullptr) {
          _mc->session()->log_debug(fmt(
              "flux3: sol declined this block ({}); dense",
              serr.empty() ? "no reason" : serr));
        }
      }
    }
    bool i8_ok = false;
    if (_sage && nax && _cfg.sage.enabled && rows >= kSageMinSegment &&
        (joint_idx < 0 || joint_idx >= _cfg.sage.dense_layers)) {
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
    const SharedBuffer* pp = params_for(rows);
    const metal_compute::ComputeFunction* fn = attn_fn(rows, i8_ok);
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
  // Over joint rows [r0, r0 + R). Each Seg is a row range (absolute) and
  // where its mode's (shift | scale | gate) start in `mods`: one Seg for
  // a per-mode block, three for a joint one. The block's own scratch is
  // indexed from 0.
  struct Seg { int row0; int rows; std::size_t me; };
  auto run_block = [&](const Block& b, int L, int joint_idx, int r0, int R,
                       const SharedBuffer& mods,
                       const std::vector<Seg>& segs,
                       AneFeedForward::Plan ane_plan) -> bool {
    auto LA = [&](lora::Factors BlockLora::* w) { return lora_for_(L, w); };
    for (const Seg& s : segs) {
      ln_mod(joint, (std::size_t)s.row0 * H, mods, s.me, nrm,
             (std::size_t)(s.row0 - r0) * H, s.rows);
    }
    mark("elt");
    // ---- the ANE split point ------------------------------------------
    //
    // PARALLEL BLOCK: `nrm` is the feed-forward's input as well as the
    // attention's, so the ANE can start on its rows NOW and run under
    // the GPU's whole attention -- the most room any family here gives
    // it. Drain, START THE ANE on the tail rows, then encode.
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
        a_rows = _ane->begin(nrm, ob2, R);
      } else if (ane_split && _mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "flux3: staging block {}'s weights for the ANE failed; it keeps "
            "the GPU feed-forward", L));
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
    }
    const auto t_g0 = std::chrono::steady_clock::now();
    lin(nrm, 0, b.q, qb, 0, R, H, H, LA(&BlockLora::q));
    lin(nrm, 0, b.k, kb, 0, R, H, H, LA(&BlockLora::k));
    lin(nrm, 0, b.v, vb, 0, R, H, H, LA(&BlockLora::v));
    mark("gemm.qkv");
    // Per-head RMSNorm over the last Hd of a token-major [R, NH, Hd], then
    // RoPE -- fused, so q and k are rounded once, as the reference rounds
    // them. Two passes cost the late joint blocks (which attend sharply)
    // ~1.5x the reference's own bf16 error at one timestep of an edit.
    if (_fused_qk) {
      tr_rms_rope(qb, b.nq, qh, R, r0);
      tr_rms_rope(kb, b.nk, kh, R, r0);
    } else {
      rms(qb, b.nq, qb, R * NH, Hd);
      rms(kb, b.nk, kb, R * NH, Hd);
      tr_rope(qb, qh, R, r0);
      tr_rope(kb, kh, R, r0);
    }
    // V is transposed but NOT rotated.
    transpose(vb, vh, R, NH);
    mark("rope");
    run_attn(joint_idx, R);
    mark("attn");
    transpose(ao, ab, NH, R);
    lin(ab, 0, b.o, ob, 0, R, H, H, LA(&BlockLora::o));
    mark("gemm.out");
    // The GPU's feed-forward over rows [f0, f0 + n): its share of a
    // split, every row of an unsplit block, and the rows the ANE lost.
    auto gpu_ff = [&](int f0, int n) {
      if (n <= 0) { return; }
      const std::size_t e0 = (std::size_t)f0 * H;
      lin(nrm, e0, b.ff_in, ffi, 0, n, 2 * MLP, H, LA(&BlockLora::ff_in));
      swiglu(ffi, ffh, n);
      lin(ffh, 0, b.ff_out, ob2, e0, n, H, MLP, LA(&BlockLora::ff_out));
    };
    gpu_ff(0, R - a_rows);
    mark("gemm.ff");
    if (ane_split || ane_probe) {
      enc.end();
      _flops.flush(stream);
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        if (a_rows > 0) { (void)_ane->finish(L, R, 0.0, ane_drain_ms); }
        return false;
      }
      const double t_gpu = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_g0).count();
      if (ane_probe) { _ane->note_probe(ane_drain_ms, t_gpu); }
      // THE GPU FALLBACK: `nrm` is only the tier's input, so the rows it
      // lost can be run again from it.
      int ff_lost = -1;
      if (!_ane->finish(L, R, t_gpu, ane_drain_ms) && a_rows > 0) {
        ff_lost = _ane->lost_row0();
        if (ff_lost < 0 || ff_lost >= R) { return false; }
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
      if (ff_lost >= 0) { gpu_ff(ff_lost, R - ff_lost); }
    }
    // x += gate * (attn + mlp): ONE gate over the SUM.
    resid(ob, 0, ob2, 0, R * H);
    for (const Seg& s : segs) {
      gated(joint, (std::size_t)s.row0 * H, mods, s.me, ob,
            (std::size_t)(s.row0 - r0) * H, s.rows);
    }
    mark("elt");
    return true;
  };

  // ---- the timestep embeddings and the modulation tables -------------
  //
  // vec(mode) = time_in(emb(t_mode)) + target_in(emb(t_next)): the image
  // at t, the other two modes at 0, and the flow map's target term in
  // all three. Six (shift | scale | gate) tables, computed ONCE: each
  // serves every block of its stack.
  lin(t_img, 0, _time_1, te1, 0, 1, H, TF);
  silu(te1, te1, H);
  lin(te1, 0, _time_2, v_img, 0, 1, H, H);
  lin(t_zero, 0, _time_1, te1, 0, 1, H, TF);
  silu(te1, te1, H);
  lin(te1, 0, _time_2, v_zero, 0, 1, H, H);
  if (_cfg.flow_map) {
    lin(t_tgt, 0, _tgt_1, te1, 0, 1, H, TF);
    silu(te1, te1, H);
    lin(te1, 0, _tgt_2, v_tgt, 0, 1, H, H);
    resid(v_img, 0, v_tgt, 0, H);
    resid(v_zero, 0, v_tgt, 0, H);
  }
  silu(v_img, s_img, H);
  silu(v_zero, s_zero, H);
  const SharedBuffer* vec_of[3] = {&s_img, &s_zero, &s_zero};
  for (int m = 0; m < 3; ++m) {
    lin(*vec_of[m], 0, _early_mod[m], emods, (std::size_t)m * 3 * H, 1,
        3 * H, H);
    lin(*vec_of[m], 0, _single_mod[m], smods, (std::size_t)m * 3 * H, 1,
        3 * H, H);
  }
  lin(s_img, 0, _out_mod, outmod, 0, 1, 2 * H, H);
  trace("v_img", v_img, (std::size_t)H);
  trace("v_zero", v_zero, (std::size_t)H);

  // ---- the three embeddings, into [text | image | references] ---------
  lin(*req.text, 0, _ctx_in, joint, 0, TXT, H, JD);
  lin(*req.latents, 0, _x_img, joint, (std::size_t)R_IMG * H, IMG, H, IC);
  {
    int at = R_COND;
    for (const Ref& rf : req.refs) {
      const int n = rf.grid_h * rf.grid_w;
      lin(*rf.latents, 0, _x_cond, joint, (std::size_t)at * H, n, H, CI);
      at += n;
    }
  }
  trace("embed", joint, (std::size_t)JT * H);
  mark("embed");

  // ---- blocks --------------------------------------------------------
  struct SlotJoin {
    BlockSlots<Block>* s;
    bool on;
    ~SlotJoin() { if (on) { s->join(); } }
  } slot_join{&_slots, _stream_blocks};
  if (_stream_blocks) {
    (void)_slots.release_if_saturated(_blocks);
    _slots.begin_forward();
  }

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
              "MetalFlux3Transformer: resident weights are only {}% in "
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

  const bool sprof = std::getenv("VPIPE_FLUX3_STREAM_PROFILE") != nullptr;
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

  if (_lora_rank_total > 0) {
    _lora_scratch_rows = (std::size_t)JT;
    _lora.ensure_scratch(_lora_scratch_rows *
                         (std::size_t)_lora_rank_total);
  }

  // SAGE'S SCRATCH, sized ONCE for the forward's largest attention. The
  // stacks attend at up to four row counts -- each mode's own rows, then
  // the joint's -- and a prepare() at a new count rebuilt the scratch: a
  // private free and allocation two or three times a forward. Reserved
  // at the largest Sage will see (the joint covers every mode's), the
  // smaller calls reuse it; a forward at the same size builds nothing.
  if (_sage && nax && _cfg.sage.enabled && _sage_reserve) {
    int big = 0;
    auto take = [&](int rows) {
      if (rows >= kSageMinSegment) { big = std::max(big, rows); }
    };
    take(IMG);
    take(TXT);
    if (COND > 0) { take(COND); }
    if (_cfg.sage.dense_layers < _cfg.n_single) { take(JT); }
    std::string rerr;
    if (big > 0 &&
        !_sage->reserve(NH, NH, big, big, Hd, A_BQ, A_BK,
                        _sage->pv8_for(_cfg.sage), &rerr) &&
        _mc->session() != nullptr) {
      // Not fatal: each prepare() sizes its own call, as without one.
      _mc->session()->log_debug(fmt(
          "flux3: sage scratch not reserved at {} rows ({})", big, rerr));
    }
  }

  // THE ORDER THE BLOCKS RUN IN, which is also the order streaming reads
  // them: the image mode's stack, the text's, the references' (skipped
  // whole when there are none -- never read), then the joint stack. The
  // per-mode stacks touch disjoint rows, so their order is free.
  const int M5 = _cfg.n_mode;
  std::vector<int> order;
  order.reserve((std::size_t)n_blocks());
  for (int L = 0; L < 2 * M5; ++L) { order.push_back(L); }
  if (COND > 0) {
    for (int L = 2 * M5; L < 3 * M5; ++L) { order.push_back(L); }
  }
  for (int L = 3 * M5; L < n_blocks(); ++L) { order.push_back(L); }
  const int TOTAL = (int)order.size();
  auto held_at = [&](int L) {
    return L < (int)_blocks.size() && !_blocks[(std::size_t)L].q.empty();
  };

  for (int oi = 0; oi < TOTAL; ++oi) {
    const int L = order[(std::size_t)oi];
    // COOPERATIVE STOP, checked every block: EMPTY WITHOUT AN ERROR.
    if (_stream_stop && _stream_stop()) { return SharedBuffer{}; }
    const bool streaming = _stream_blocks && !held_at(L);
    const Block* b = nullptr;
    if (!streaming) {
      b = &_blocks[(std::size_t)L];
    } else {
      const auto rd0 = sp_now();
      b = _slots.acquire(L);
      if (b == nullptr) {
        return fail("flux3 forward: a streamed block did not load");
      }
      sp_add(sp_read_ms, rd0);
      if (sprof) {
        sp_read_bytes += _slots.last_bytes();
        ++sp_blocks;
      }
      mark("stream");
    }

    // Which rows, and whose modulation.
    int r0 = 0, R = JT, joint_idx = -1;
    std::vector<Seg> segs;
    if (L < M5) {
      r0 = R_IMG; R = IMG;
      segs.push_back(Seg{R_IMG, IMG, (std::size_t)kModeImage * 3 * H});
    } else if (L < 2 * M5) {
      r0 = 0; R = TXT;
      segs.push_back(Seg{0, TXT, (std::size_t)kModeText * 3 * H});
    } else if (L < 3 * M5) {
      r0 = R_COND; R = COND;
      segs.push_back(Seg{R_COND, COND, (std::size_t)kModeCond * 3 * H});
    } else {
      joint_idx = L - 3 * M5;
      segs.push_back(Seg{0, TXT, (std::size_t)kModeText * 3 * H});
      segs.push_back(Seg{R_IMG, IMG, (std::size_t)kModeImage * 3 * H});
      if (COND > 0) {
        segs.push_back(Seg{R_COND, COND, (std::size_t)kModeCond * 3 * H});
      }
    }
    const SharedBuffer& mods = joint_idx >= 0 ? smods : emods;

    // THE ANE'S PLAN, joint blocks only: the module is sized once, for
    // the joint length.
    AneFeedForward::Plan ane_plan = AneFeedForward::Plan::kGpu;
    if (_cfg.ane_ffn && joint_idx >= 0 && ane_setup_(JT) &&
        ane_eligible_(L, *b)) {
      ane_plan = _ane->plan_block();
    }
    if (ane_plan != AneFeedForward::Plan::kGpu &&
        (joint_idx == 0 || _ane->needs_barrier())) {
      // A held stack defers its commits, so the split point's drain
      // would wait for every block queued before it. Drain FIRST.
      enc.end();
      _flops.flush(stream);
      std::string ge2;
      if (!stream.commit().wait_ok(&ge2)) {
        return fail("flux3 forward: " + ge2);
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
    }
    if (ane_plan == AneFeedForward::Plan::kSplit) { ane_stage_(L, *b); }

    _flops.flush(stream);
    if (_block_progress) {
      // `oi` of `TOTAL`: the blocks finished BEFORE this one -- a report
      // that lags the truth and never leads it.
      report_block(stream, _block_progress, oi, TOTAL);
    }
    if (!run_block(*b, L, joint_idx, r0, R, mods, segs, ane_plan)) {
      return fail("flux3 forward: block " + std::to_string(L) + " failed");
    }

    // ---- the streaming boundary --------------------------------------
    //
    // A STREAMED block must commit: the two slots are reused, so anything
    // encoded but not run still points at bytes the next read would
    // overwrite. The commit buys the prefetch its window too.
    if (streaming) {
      const auto gp0 = sp_now();
      enc.end();
      _flops.flush(stream);
      std::string ge;
      {
        CommandStream::Fence bf = stream.commit();
        // The next block's read goes into the OTHER slot, between the
        // commit and the wait. Skip what residency already kept.
        int nxt = -1;
        for (int n = oi + 1; n < TOTAL; ++n) {
          const int Ln = order[(std::size_t)n];
          if (!held_at(Ln)) { nxt = Ln; break; }
        }
        _slots.prefetch(nxt);
        if (!bf.wait_ok(&ge)) { return fail("flux3 forward: " + ge); }
      }
      // Keep this block if the box can hold it -- AFTER the wait, cloned
      // out of the slot so the read in flight keeps its destination.
      const std::size_t nb = _slots.last_bytes();
      if (nb > 0 && _wire.wirable(nb) && _resid.admit(_mc, nb) &&
          _slots.promote_into(_blocks[(std::size_t)L])) {
        _wire.note_wired(_mc, wire_block_(_blocks[(std::size_t)L], true),
                         nb);
        _resid.note_admitted(nb);
      }
      sp_add(sp_gpu_ms, gp0);
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
      if (!enc.valid()) { return fail("flux3 forward: no encoder"); }
      mark("stream");
    } else if (_stream_stop && kStopDrainBlocks > 0 &&
               ((oi + 1) % kStopDrainBlocks) == 0 && oi + 1 < TOTAL) {
      // A HELD BLOCK COMMITS NOTHING, so without this the whole forward
      // runs inside one wait and a stop arriving mid-forward is never
      // seen. Only when a stop is wired.
      enc.end();
      _flops.flush(stream);
      std::string ge;
      if (!stream.commit().wait_ok(&ge)) {
        return fail("flux3 forward: " + ge);
      }
      // The same stream: a fresh one is a new queue, and its first
      // commit costs ~1-1.9 ms against ~0.25 reused (gemm_i8.a_fresh_
      // stream_per_commit_costs).
      enc = stream.begin_compute();
      if (!enc.valid()) { return fail("flux3 forward: no encoder"); }
    }
    trace(("blk" + std::to_string(L)).c_str(), joint, (std::size_t)JT * H);
  }
  if (!attn_ok) {
    return fail("flux3 forward: the steel attention kernel did not "
                "specialize for this geometry");
  }

  // ---- the output layer: the IMAGE rows only ---------------------------
  //
  // (1 + scale) * LN(x) + shift, then the projection back to latent
  // tokens. The table is (scale | shift) -- the reverse of a block's
  // (shift | scale | gate) -- so the halves are bound here explicitly
  // rather than through ln_mod's block layout.
  enc.set_function(_fn_ln_mod);
  enc.set_buffer(0, joint, (std::size_t)R_IMG * H * 2);
  enc.set_buffer(1, outmod, 0);                          // scale
  enc.set_buffer(2, outmod, (std::size_t)H * 2);         // shift
  enc.set_buffer(3, nrm, 0);
  enc.set_constant(4, H); enc.set_constant(5, eps);
  enc.dispatch({256, (unsigned)IMG, 1}, {256, 1, 1});
  trace("outmod", outmod, (std::size_t)2 * H);
  trace("final_in", nrm, (std::size_t)IMG * H);
  lin(nrm, 0, _proj_out, vel, 0, IMG, IC, H);
  enc.end();
  _flops.flush(stream);
  {
    std::string ge;
    if (!stream.commit().wait_ok(&ge)) {
      return fail("flux3 forward: " + (ge.empty() ? "GPU failed" : ge));
    }
  }
  if (_sol) { _sol_flops.finish(_sol->exact_blocks()); }
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
        "FLUX.3 DiT profile (joint={} txt={} img={} ref={}, {} blocks, {}, "
        "{} marks -- each is a commit AND a wait, so a bucket with many "
        "marks and little work is mostly measuring this): total {} ms{}",
        JT, TXT, IMG, COND, TOTAL,
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
        "MetalFlux3Transformer: streamed {} of {} blocks -- read {} ms "
        "({} MB at {:.2f} GB/s, {:.1f} ms/block), gpu {} ms ({:.0f} "
        "ms/block); a perfect prefetch hides at most {:.1f}% of this pass",
        sp_blocks, TOTAL, (long)sp_read_ms, sp_read_bytes >> 20, gbs,
        sp_read_ms / (double)sp_blocks, (long)sp_gpu_ms,
        sp_gpu_ms / (double)sp_blocks,
        tot > 0.0 ? 100.0 * sp_read_ms / tot : 0.0));
  }
  return vel;
}

// ---- the matrix-core GEMM -------------------------------------------
//
// Returns false with NOTHING encoded when this box or this shape is not
// for it, and the caller keeps its tiled path.
bool
MetalFlux3Transformer::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& xin,
                                 std::size_t xe, const QWeight& w,
                                 const SharedBuffer& y, std::size_t ye,
                                 int M, int N, int K) const
{
  if (!_use_mma2 || N < 16) { return false; }
  // A signed weight's int8 GEMM, before the matmul2d tile's row gate:
  // its floor is the int8 route's own (I8GemmContext::signed_accepts --
  // 1024 rows, or lower by VPIPE_I8_SIGNED_MIN_M). In the same row bands
  // as the tiles (see below), and only when every band takes it, so a
  // decline encodes nothing.
  if (_i8 && w.quantized && w.sgn) {
    const int band = mma_row_band(N, K);
    // The shortest band is the last one; the floor is all that varies.
    const int last = M <= band ? M : (M % band == 0 ? band : M % band);
    if (_i8->signed_accepts(last, N, K, w.bits)) {
      for (int m0 = 0; m0 < M; m0 += band) {
        const int rows = std::min(band, M - m0);
        if (!_i8->gemm_signed(enc, xin, xe + (std::size_t)m0 * K, w.codes,
                              w.scales, w.qbias, w.bits, y,
                              ye + (std::size_t)m0 * N, rows, N, K)) {
          return false;   // unreachable: every band was accepted above
        }
      }
      return true;
    }
  }
  // M=1 (the timestep, the modulation tables) is exactly the shape the
  // 128-row tile is wrong for, and stays on steel.
  if (M < _mma_min_m) { return false; }

  // Tile by REDUCTION DEPTH: 3072 for qkv / out / ff-in, 9216 for
  // ff-out, 20480 for the text embedder. Chosen before anything is
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
  if (w.quantized && w.sgn) {
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

  // IN ROW BANDS: past 2^31 BYTES on any operand the tiles stop storing,
  // SILENTLY, and at 4096^2 the joint feed-forward's [65616, 18432]
  // output is 2.4 GB. Declining there, as this did, put the model's
  // widest GEMM on steel at exactly the size where it costs the most.
  // Below the line the band IS M -- one pass, as before.
  const int band = mma_row_band(N, K);
  for (int m0 = 0; m0 < M; m0 += band) {
    const int rows = std::min(band, M - m0);
    const std::size_t xo = xe + (std::size_t)m0 * K;
    const std::size_t yo = ye + (std::size_t)m0 * N;
    // Accelerated mode, from the SAME dense operand the tiles would
    // read.
    if (_i8 && _i8->gemm(enc, xin, xo, *wdense, y, yo, rows, N, K)) {
      continue;
    }
    enc.set_function(*fn);
    enc.set_buffer(0, xin, xo * 2);
    enc.set_buffer(1, *wdense); enc.set_buffer(2, *wdense);
    enc.set_buffer(3, y, yo * 2);
    enc.set_constant(4, K); enc.set_constant(5, N);
    enc.set_constant(6, rows);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                  (unsigned)((rows + 127) / 128), 1}, {256, 1, 1});
  }
  return true;
}

// ---- the ANE tier ---------------------------------------------------
//
// The joint blocks' feed-forward is row-independent, so the ANE takes
// the LAST rows and the GPU the head rows, concurrently. Exact: the two
// compute one function over disjoint rows.
//
// WHERE IT IS FOR: the M4 series, which has no GPU matrix cores. Do not
// judge it on a fanless box.
int
MetalFlux3Transformer::ane_chunk_rows() noexcept
{
  return AneFeedForward::kChunkDefault;
}

std::size_t
MetalFlux3Transformer::ane_runtime_bytes(const Config& c, int seq) noexcept
{
  return AneFeedForward::runtime_bytes(c.hidden, c.mlp_hidden, seq,
                                       ane_chunk_rows());
}

int
MetalFlux3Transformer::ane_plan_seq(int width, int height) noexcept
{
  // One row per 16x16 pixels, plus a prompt this cannot know: 160 rows
  // is two of the reference's 80-row pads and is noise against the
  // image at any size the tier is for. References are not counted.
  const int w = width > 0 ? width : 1024;
  const int h = height > 0 ? height : 1024;
  return (w / 16) * (h / 16) + 160;
}

std::size_t
MetalFlux3Transformer::forward_scratch_bytes(const Config& c,
                                             int rows) noexcept
{
  // Mirrors forward()'s allocations: eleven [rows, H] planes (joint, nrm,
  // ob, q/k/v, their head-major copies, ao, ob2), the [rows, 2 MLP] and
  // [rows, MLP] feed-forward pair, and the two f32 rotary tables. The
  // per-forward tables of hidden width and the velocity are noise.
  const std::size_t R = rows > 0 ? (std::size_t)rows : 0;
  const std::size_t H = (std::size_t)c.hidden;
  const std::size_t F = (std::size_t)c.mlp_hidden;
  const std::size_t Hd = (std::size_t)c.head_dim;
  std::size_t n = R * H * 2 * 11 + R * F * 3 * 2 + R * Hd * 4 * 2;
  // AND THE ATTENTION TIERS' OWN, which this forward lends them nothing
  // for: Sage's int8 q / k (and with sage_pv8 V's uint8 copy and Q's
  // per-token scales) and Sol's routing, each allocated privately at the
  // widest attention -- the joint stack's, every row -- and kept.
  if (c.sage.enabled && R > 0) {
    n += MetalSageAttention::scratch_bytes(
        c.n_heads, c.n_heads, (int)R, (int)R, c.head_dim,
        MetalSageAttention::nax_query_block(),
        MetalSageAttention::nax_key_block(), c.sage.pv8);
  }
  if (c.sol.enabled && R > 0) {
    n += MetalSolAttention::scratch_bytes(c.n_heads, (int)R, c.head_dim,
                                          c.sol.key_block, /*nax=*/true);
  }
  return n;
}

bool
MetalFlux3Transformer::ane_setup_(int seq)
{
  if (_ane_tried) { return _ane != nullptr; }
  _ane_tried = true;
  if (_mc == nullptr || _mc->session() == nullptr || seq <= 0) {
    return false;
  }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "flux3";
  o.hidden  = _cfg.hidden;
  o.ffn     = _cfg.mlp_hidden;
  o.seq     = seq;
  o.rows    = _cfg.ane_rows;
  o.chunk   = ane_chunk_rows();
  o.int8    = _cfg.ane_int8;
  o.int8_plain = _cfg.ane_int8_plain;
  o.profile = std::getenv("VPIPE_FLUX3_ANE_PROFILE") != nullptr;
  _ane = AneFeedForward::create(o);
  return _ane != nullptr;
}

bool
MetalFlux3Transformer::ane_eligible_(int L, const Block& b) const
{
  if (_ane == nullptr) { return false; }
  const int j = L - 3 * _cfg.n_mode;
  if (j < 0) { return false; }
  const int cap = (_cfg.ane_layers > 0)
                      ? std::min(_cfg.ane_layers, _cfg.n_single)
                      : _cfg.n_single;
  if (j >= cap) { return false; }
  auto ok = [](const QWeight& q) {
    if (q.empty()) { return false; }
    if (!q.quantized) { return true; }
    return (q.bits == 4 || q.bits == 8) && !q.scales.empty() &&
           !q.qbias.empty();
  };
  return ok(b.ff_in) && ok(b.ff_out);
}

namespace {

// One projection as the ANE stages it, WITH the runtime adapters folded
// in: left out, the ANE's rows would run the base model while the GPU's
// run the adapted one. `row0` selects a half of a fused weight -- the
// adapter's B rows follow the weight's, so they take the same offset.
template <typename Q>
AneFfnSource
ane_src_(const Q& q, const lora::Stack& st, std::size_t row0)
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

}  // namespace

void
MetalFlux3Transformer::ane_stage_(int L, const Block& b)
{
  if (_ane == nullptr) { return; }
  // linear_in is [gate | up] in one matrix: the gate is its first
  // mlp_hidden rows and the up projection the rest.
  const std::size_t M = (std::size_t)_cfg.mlp_hidden;
  const lora::Stack in = lora_for_(L, &BlockLora::ff_in);
  _ane->stage(L, ane_src_(b.ff_in, in, 0), ane_src_(b.ff_in, in, M),
              ane_src_(b.ff_out, lora_for_(L, &BlockLora::ff_out), 0),
              _quant_group);
}

// The q/k/v tier is not built for this family: q/k/v and the feed-
// forward read the same `nrm` and the one ANE worker takes one job at
// a time, so the feed-forward -- the wider matmul, and the one that can
// run under the attention -- is the one it gets.
bool
MetalFlux3Transformer::ane_qkv_setup_(int)
{
  _ane_qkv_tried = true;
  return false;
}

bool
MetalFlux3Transformer::ane_qkv_eligible_(int, const Block&) const
{
  return false;
}

void
MetalFlux3Transformer::ane_qkv_stage_(int, const Block&)
{
}

// ---- runtime LoRA ---------------------------------------------------
//
// THE MODULE NAMES ARE THE MODEL'S OWN (diffusers) -- the adapter's keys
// are the base tensor's path without `.weight`, optionally under a
// `transformer.`/`diffusion_model.` wrapper the shared reader strips.
// The modulation tables and the embedders are not adapted.
bool
MetalFlux3Transformer::bind_loras_(const std::vector<LoraSpec>& loras)
{
  if (loras.empty()) { return true; }
  if (!_lora.init(_mc, _use_mma2)) { return false; }
  if (!_lora.valid()) { return false; }
  const int NB = n_blocks();
  int slot = 0;
  for (const LoraSpec& spec : loras) {
    if (slot >= kMaxLoraSlots || spec.path.empty()) { break; }
    std::string err;
    auto ad = lora::Adapter::open(spec.path, _mc, &err,
                                  lora::remap_ai_toolkit_module);
    if (ad == nullptr) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "MetalFlux3Transformer: lora '{}' did not open: {}", spec.path,
            err.empty() ? "unreadable" : err));
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
    _lora_blk[slot].resize((std::size_t)NB);
    const int H = _cfg.hidden, MLP = _cfg.mlp_hidden;
    for (int L = 0; L < NB; ++L) {
      const std::string p = prefix_(L);
      BlockLora& d = _lora_blk[slot][(std::size_t)L];
      ad->bind(p + "attn.to_q", H, H, &d.q);
      ad->bind(p + "attn.to_k", H, H, &d.k);
      ad->bind(p + "attn.to_v", H, H, &d.v);
      ad->bind(p + "attn.to_out.0", H, H, &d.o);
      ad->bind(p + "ff.linear_in", 2 * MLP, H, &d.ff_in);
      ad->bind(p + "ff.linear_out", H, MLP, &d.ff_out);
    }
    _lora_bound[slot] = ad->modules();
    _lora_scale[slot] = spec.scale;
    _lora_rank_prefix[slot] = _lora_rank_total;
    _lora_rank_total += ad->max_rank();
    if (_mc->session() != nullptr) {
      _mc->session()->log_normal(
          fmt("MetalFlux3Transformer: {}", ad->summary(spec.path,
                                                       spec.scale)));
      if (ad->modules() == 0) {
        _mc->session()->warn(fmt(
            "MetalFlux3Transformer: lora '{}' bound NO modules -- it is for "
            "a different model, or its names are in a convention this "
            "family does not map yet", spec.path));
      }
    }
    ++slot;
  }
  _lora_slots = slot;
  return true;
}

lora::Stack
MetalFlux3Transformer::lora_for_(int L,
                                 lora::Factors BlockLora::* which) const
{
  lora::Stack st;
  for (int i = 0; i < _lora_slots; ++i) {
    const std::vector<BlockLora>& v = _lora_blk[i];
    if (L < 0 || L >= (int)v.size()) { continue; }
    const lora::Factors& f = v[(std::size_t)L].*which;
    // Each slot gets its OWN region of the [rows, rank] scratch.
    st.add(f, _lora_scale[i],
           _lora_scratch_rows * (std::size_t)_lora_rank_prefix[i]);
  }
  return st;
}

void
MetalFlux3Transformer::set_lora_scale(int slot, float s)
{
  if (slot < 0 || slot >= kMaxLoraSlots) { return; }
  _lora_scale[slot] = s;
}

int
MetalFlux3Transformer::lora_modules(int slot) const
{
  if (slot < 0 || slot >= kMaxLoraSlots) { return 0; }
  return _lora_bound[slot];
}

// ---- memory ---------------------------------------------------------

std::size_t
MetalFlux3Transformer::streaming_floor_bytes(const std::string& dir)
{
  if (dir.empty()) { return 0; }
  auto wts = MetalLlamaWeights::open_model(dir);
  if (!wts.has_value()) { return 0; }
  const std::vector<std::string>& stems = block_stems();
  std::size_t trunk = 0;
  // Keyed by stem AND index: mode_blocks.image.0 and txt_mode_blocks.0
  // are different blocks of one stack.
  std::map<std::string, std::size_t> blocks;
  bool any_f32 = false;
  for (const std::string& nm : wts->tensor_names()) {
    const auto* ti = wts->info(nm);
    const std::size_t nb = ti != nullptr ? (std::size_t)ti->nbytes : 0;
    if (ti != nullptr && ti->dtype == "F32") { any_f32 = true; }
    const std::string* stem = nullptr;
    for (const std::string& s : stems) {
      if (nm.rfind(s, 0) == 0) { stem = &s; break; }
    }
    if (stem == nullptr) { trunk += nb; continue; }
    const std::size_t dot = nm.find('.', stem->size());
    if (dot == std::string::npos) { trunk += nb; continue; }
    blocks[nm.substr(0, dot)] += nb;
  }
  std::size_t widest = 0;
  for (const auto& kv : blocks) { widest = std::max(widest, kv.second); }
  if (widest == 0) { return 0; }
  // ONE slot pair: every block of every stack has the same shape.
  std::size_t floor = trunk + 2 * widest;
  if (any_f32) { floor /= 2; }
  return floor;
}

std::size_t
MetalFlux3Transformer::checkpoint_block_bytes_() const
{
  if (_ws == nullptr) { return 0; }
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
MetalFlux3Transformer::set_residency_schedule(int steps)
{
  const std::size_t blk = checkpoint_block_bytes_();
  _wire_block_hint = blk;
  _wire.new_run();
  _resid.set_schedule(steps, n_blocks(), blk, _wire.on(),
                      _mc != nullptr ? _mc->memory_budget()
                                     : metal_compute::MetalCompute::
                                           MemoryBudget{});
}

std::size_t
MetalFlux3Transformer::release_resident_blocks(std::size_t bytes)
{
  const std::size_t freed = _resid.release(bytes, [this]() -> std::size_t {
    return evict_tail_block_();
  });
  if (freed > 0 && _mc != nullptr && _mc->session() != nullptr) {
    _mc->session()->log_debug(fmt(
        "MetalFlux3Transformer: released {} MB of resident blocks "
        "({} left)", freed >> 20, _resid.count()));
  }
  return freed;
}

std::uint64_t
MetalFlux3Transformer::resident_bytes() const
{
  std::uint64_t n = qw_bytes_(_x_img) + qw_bytes_(_x_cond) +
                    qw_bytes_(_ctx_in) + qw_bytes_(_time_1) +
                    qw_bytes_(_time_2) + qw_bytes_(_tgt_1) +
                    qw_bytes_(_tgt_2) + qw_bytes_(_out_mod) +
                    qw_bytes_(_proj_out);
  for (int m = 0; m < 3; ++m) {
    n += qw_bytes_(_early_mod[m]) + qw_bytes_(_single_mod[m]);
  }
  // `_blocks` IS the answer in both modes: preloaded it holds the stack,
  // streaming exactly what residency promoted.
  for (const Block& b : _blocks) { n += block_bytes_(b); }
  // And the slot pair while it exists: it stays allocated beside every
  // promoted block, and a holding taken without it dropped two blocks
  // from both ledgers once the promoted set passed the floor.
  n += _slots.held_bytes();
  return n;
}

void
MetalFlux3Transformer::release_forward_scratch()
{
  _w_deq = {};
  if (_i8) { _i8->release_scratch(); }
}

}  // namespace genai
}  // namespace vpipe
