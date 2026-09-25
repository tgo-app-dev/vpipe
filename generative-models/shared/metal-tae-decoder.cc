#include "generative-models/shared/metal-tae-decoder.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// The largest f16 -- the ReLU's upper bound.
constexpr float kF16Max = 65504.0f;

// The head goes to the direct conv at or under this many output channels
// (3 * patch^2: 3, 12, 48). The gather conv tiles output channels 32 or
// more at a time, so a 12-channel head would compute mostly padding.
constexpr int kSmallCoutMax = 48;

// A tensor's host copy as f32, whatever the file stores. Read UNCACHED:
// every weight here is transformed into the decoder's own layout and the
// source is dropped (the WeightSet contract's "consumed" case).
bool
read_host_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
           std::vector<float>* out, std::vector<std::int64_t>* shape)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr) { return false; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty() || raw.byte_size() < info->nbytes) { return false; }
  out->resize(n);
  if (info->dtype == "F32") {
    std::memcpy(out->data(), raw.contents(), n * 4);
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { (*out)[i] = (float)s[i]; }
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t u = (std::uint32_t)s[i] << 16;
      std::memcpy(&(*out)[i], &u, 4);
    }
  } else {
    return false;
  }
  if (shape != nullptr) { *shape = info->shape; }
  return true;
}

SharedBuffer
f16_buf_(MetalCompute* mc, const std::vector<float>& src)
{
  SharedBuffer b = mc->make_shared_buffer(src.size() * 2);
  if (b.empty()) { return {}; }
  auto* d = static_cast<_Float16*>(b.contents());
  for (std::size_t i = 0; i < src.size(); ++i) { d[i] = (_Float16)src[i]; }
  return b;
}

SharedBuffer
zeros_(MetalCompute* mc, std::size_t elems)
{
  SharedBuffer b = mc->make_shared_buffer(std::max<std::size_t>(elems, 1) * 2);
  if (!b.empty()) { std::memset(b.contents(), 0, b.byte_size()); }
  return b;
}

// Is raw frame `raw` one the family keeps?
bool
kept_(long raw, int tu, MetalTaeDecoder::Trim trim)
{
  if (tu <= 1) { return true; }
  if (trim == MetalTaeDecoder::Trim::kH3Chunks) {
    return raw % (5L * tu) >= tu - 1;
  }
  return raw >= tu - 1;
}

}  // namespace

MetalTaeDecoder::~MetalTaeDecoder() = default;

std::unique_ptr<MetalTaeDecoder>
MetalTaeDecoder::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                      std::string* err)
{
  auto fail = [&](std::string m) -> std::unique_ptr<MetalTaeDecoder> {
    if (err != nullptr) { *err = std::move(m); }
    return nullptr;
  };
  if (ws == nullptr || mc == nullptr) { return fail("no checkpoint"); }
  std::unique_ptr<MetalTaeDecoder> d(new MetalTaeDecoder());
  d->_ws = std::move(ws);
  d->_mc = mc;
  std::string e;
  if (!d->parse_(d->_ws->src(), /*read=*/true, &e)) { return fail(e); }
  if (!d->build_kernels_(&e)) { return fail(e); }
  return d;
}

bool
MetalTaeDecoder::parse_(const MetalLlamaWeights& src, bool read,
                        std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  // A TAEHV file carries its encoder beside the decoder under
  // `encoder.` / `decoder.`; Kijai's 2D decoder is the decoder alone, with
  // no prefix at all.
  if (src.has("decoder.layers.0.weight")) {
    // Diffusers' AutoencoderTiny (taef1, taef2, taesd...): the same
    // Sequential with its Clamp lifted out of the list into forward(), so
    // every index is one lower and slot 0 is already the first conv.
    _prefix = "decoder.layers.";
    _info.layout = "diffusers";
    _clamp_input = true;
  } else if (src.has("decoder.1.weight")) {
    _prefix = "decoder.";
    _info.layout = "taehv";
  } else if (src.has("1.weight")) {
    _prefix.clear();
    _info.layout = "2d";
  } else {
    return fail("no `decoder.layers.0.weight`, `decoder.1.weight` or "
                "`1.weight`: not a tiny autoencoder decoder this reader "
                "knows");
  }
  // The highest index present, so an empty slot (Clamp, ReLU, Upsample)
  // is walked like any other.
  // Every tensor under each index is collected too: a slot is only read
  // as a kind when EVERY tensor it carries is one that kind has, so a
  // layout variant this reader does not know (a norm, a gate, a pooled
  // branch it has never seen) is refused by name rather than half-run
  // with the unknown part silently dropped.
  int max_idx = -1;
  std::map<int, std::vector<std::string>> suffixes;
  for (const std::string& nm : src.tensor_names()) {
    if (nm.compare(0, _prefix.size(), _prefix) != 0) { continue; }
    std::size_t p = _prefix.size(), q = p;
    while (q < nm.size() && nm[q] >= '0' && nm[q] <= '9') { ++q; }
    if (q == p || q >= nm.size() || nm[q] != '.') { continue; }
    const int idx = std::stoi(nm.substr(p, q - p));
    max_idx = std::max(max_idx, idx);
    suffixes[idx].push_back(nm.substr(q + 1));
  }
  auto only = [&](int idx, std::initializer_list<const char*> allowed,
                  std::string* bad) {
    for (const std::string& sfx : suffixes[idx]) {
      bool ok = false;
      for (const char* a : allowed) { ok = ok || sfx == a; }
      if (!ok) { *bad = sfx; return false; }
    }
    return true;
  };
  if (max_idx < 1) { return fail("the decoder has no layers"); }

  MetalCompute* mc = _mc;
  std::vector<float> host;
  std::vector<std::int64_t> shp;
  int C = 0;          // channels flowing between ops
  int scale = 1;      // spatial scale of the current op's input
  // Build a 3x3 conv from a [cout, cin, 3, 3] tensor, K laid (ky, kx, ci).
  // `pad_cin` rounds the input channels up with zero columns, so the
  // gather conv's vectorized path (cin % 32 == 0) takes the first conv
  // too; its input is padded the same way at upload.
  // Shapes only, when planning: the geometry and the byte counts the
  // real build would produce, from the header, with nothing read.
  auto shape_of = [&](const std::string& nm) -> std::vector<std::int64_t> {
    const auto* inf = src.info(nm);
    return inf != nullptr ? inf->shape : std::vector<std::int64_t>{};
  };
  auto conv3 = [&](const std::string& nm, int pad_cin, Conv* c,
                   bool want_hwio) -> bool {
    if (!read) {
      shp = shape_of(nm + ".weight");
      if (shp.size() != 4 || shp[2] != 3 || shp[3] != 3) { return false; }
      const int co = (int)shp[0], ci = (int)shp[1];
      const int cp = std::max(ci, pad_cin);
      const bool bias = src.has(nm + ".bias");
      _info.params += (std::size_t)co * ci * 9 + (bias ? co : 0);
      _weight_bytes += (std::size_t)2 * co * 9 * cp *
                           (want_hwio ? 2 : 1) +
                       (bias ? (std::size_t)2 * co : 0);
      c->cin = cp;
      c->cout = co;
      c->k3 = true;
      return true;
    }
    if (!read_host_(*_ws, mc, nm + ".weight", &host, &shp)) {
      return false;
    }
    if (shp.size() != 4 || shp[2] != 3 || shp[3] != 3) { return false; }
    const int co = (int)shp[0], ci = (int)shp[1];
    const int cp = std::max(ci, pad_cin);
    std::vector<float> w((std::size_t)co * 9 * cp, 0.0f);
    for (int o = 0; o < co; ++o) {
      for (int i = 0; i < ci; ++i) {
        for (int t = 0; t < 9; ++t) {
          w[(std::size_t)o * 9 * cp + (std::size_t)t * cp + i] =
              host[((std::size_t)o * ci + i) * 9 + t];
        }
      }
    }
    c->w = f16_buf_(mc, w);
    if (want_hwio) {
      std::vector<float> hw((std::size_t)9 * cp * co, 0.0f);
      for (int o = 0; o < co; ++o) {
        for (int i = 0; i < ci; ++i) {
          for (int t = 0; t < 9; ++t) {
            hw[((std::size_t)t * cp + i) * co + o] =
                host[((std::size_t)o * ci + i) * 9 + t];
          }
        }
      }
      c->hwio = f16_buf_(mc, hw);
      if (c->hwio.empty()) { return false; }
    }
    _info.params += (std::size_t)co * ci * 9;
    _weight_bytes += c->w.byte_size() + c->hwio.byte_size();
    if (src.has(nm + ".bias")) {
      if (!read_host_(*_ws, mc, nm + ".bias", &host, &shp)) {
        return false;
      }
      if ((int)host.size() != co) { return false; }
      c->b = f16_buf_(mc, host);
      _info.params += (std::size_t)co;
      _weight_bytes += c->b.byte_size();
    }
    c->cin = cp;
    c->cout = co;
    c->k3 = true;
    return !c->w.empty();
  };
  auto conv1 = [&](const std::string& nm, Conv* c) -> bool {
    if (!read) {
      shp = shape_of(nm);
      if (shp.size() != 4 || shp[2] != 1 || shp[3] != 1) { return false; }
      c->cout = (int)shp[0];
      c->cin = (int)shp[1];
      c->k3 = false;
      _info.params += (std::size_t)c->cout * c->cin;
      _weight_bytes += (std::size_t)2 * c->cout * c->cin;
      return true;
    }
    if (!read_host_(*_ws, mc, nm, &host, &shp)) { return false; }
    if (shp.size() != 4 || shp[2] != 1 || shp[3] != 1) { return false; }
    c->cout = (int)shp[0];
    c->cin = (int)shp[1];
    c->k3 = false;
    c->w = f16_buf_(mc, host);   // [cout, cin, 1, 1] IS [cout, cin]
    _info.params += host.size();
    _weight_bytes += c->w.byte_size();
    return !c->w.empty();
  };

  for (int i = 0; i <= max_idx; ++i) {
    const std::string nm = _prefix + std::to_string(i);
    Op op{};
    op.scale = scale;
    op.in_ch = C;
    if (src.has(nm + ".conv.0.weight")) {
      const auto* inf = src.info(nm + ".conv.0.weight");
      if (inf == nullptr || inf->shape.size() != 4) {
        return fail(fmt("{}: a block whose first conv is not 4-D", nm)());
      }
      if (inf->shape[2] != 3) {
        // The Super decoder's blocks open with a depthwise 7x7.
        return fail(fmt("{}: a {}x{} block conv -- a Super decoder "
                        "(`*_super`), which this reader does not run",
                        nm, inf->shape[2], inf->shape[3])());
      }
      const int cin0 = (int)inf->shape[1];
      if (C > 0 && cin0 == 2 * C) {
        op.kind = Op::kMemBlock;
        op.mem = _n_mem++;
      } else if (C > 0 && cin0 == C) {
        op.kind = Op::kBlock;
      } else {
        return fail(fmt("{}: block reads {} channels where {} flow in",
                        nm, cin0, C)());
      }
      if (!conv3(nm + ".conv.0", 0, &op.c0, false) ||
          !conv3(nm + ".conv.2", 0, &op.c1, false) ||
          !conv3(nm + ".conv.4", 0, &op.c2, false)) {
        return fail(fmt("{}: unreadable block convs", nm)());
      }
      op.out_ch = op.c2.cout;
      std::string bad;
      if (!only(i, {"conv.0.weight", "conv.0.bias", "conv.2.weight",
                    "conv.2.bias", "conv.4.weight", "conv.4.bias",
                    "skip.weight", "pool.0.weight", "pool.1.weight",
                    "pool.1.bias", "pool.3.weight"}, &bad)) {
        return fail(fmt("{}: a block tensor '{}' this reader does not "
                        "run", nm, bad)());
      }
      // TAESD's mid-block POOL (taef2): x += conv1x1(relu(GroupNorm(4)(
      // conv1x1(x)))), before the block reads x. The GroupNorm is over the
      // whole picture, which is the global context the FLUX.2 distillation
      // needed; its group count is TAESD's constant, not in the file.
      if (src.has(nm + ".pool.0.weight")) {
        if (op.kind != Op::kBlock) {
          return fail(fmt("{}: a pooled MemBlock, which no published "
                          "decoder has", nm)());
        }
        if (!conv1(nm + ".pool.0.weight", &op.pool_in) ||
            !conv1(nm + ".pool.3.weight", &op.pool_out) ||
            op.pool_in.cin != C || op.pool_out.cout != C ||
            op.pool_out.cin != op.pool_in.cout ||
            op.pool_in.cout % kPoolGroups != 0) {
          return fail(fmt("{}: pool convs are not [n, {}] / [{}, n]", nm, C,
                          C)());
        }
        const int n = op.pool_in.cout;
        auto vec = [&](const std::string& t, metal_compute::SharedBuffer* b) {
          const auto sh = shape_of(t);
          if (sh.size() != 1 || sh[0] != n) { return false; }
          _info.params += (std::size_t)n;
          _weight_bytes += (std::size_t)2 * n;
          if (!read) { return true; }
          if (!read_host_(*_ws, mc, t, &host, &shp)) { return false; }
          *b = f16_buf_(mc, host);
          return !b->empty();
        };
        if (!vec(nm + ".pool.1.weight", &op.gn_gamma) ||
            !vec(nm + ".pool.1.bias", &op.gn_beta)) {
          return fail(fmt("{}: pool GroupNorm affine is not [{}]", nm, n)());
        }
        op.has_pool = true;
      }
      if (src.has(nm + ".skip.weight")) {
        if (!conv1(nm + ".skip.weight", &op.skip) || op.skip.cin != C ||
            op.skip.cout != op.out_ch) {
          return fail(fmt("{}: skip conv does not map {} -> {}", nm, C,
                          op.out_ch)());
        }
        op.has_skip = true;
      } else if (op.out_ch != C) {
        return fail(fmt("{}: {} -> {} with no skip conv", nm, C, op.out_ch)());
      }
      C = op.out_ch;
    } else if (src.has(nm + ".conv.weight")) {
      op.kind = Op::kTGrow;
      std::string bad;
      if (!only(i, {"conv.weight"}, &bad)) {
        return fail(fmt("{}: a TGrow tensor '{}' this reader does not run",
                        nm, bad)());
      }
      if (!conv1(nm + ".conv.weight", &op.c) || op.c.cin != C || C <= 0 ||
          op.c.cout % C != 0) {
        return fail(fmt("{}: TGrow is not [stride * {}, {}]", nm, C, C)());
      }
      op.stride = op.c.cout / C;
      op.out_ch = C;
      _info.time_upscale *= op.stride;
    } else if (src.has(nm + ".weight")) {
      op.kind = Op::kConv;
      std::string bad;
      if (!only(i, {"weight", "bias"}, &bad)) {
        return fail(fmt("{}: a conv tensor '{}' this reader does not run",
                        nm, bad)());
      }
      const auto* inf = src.info(nm + ".weight");
      if (inf == nullptr || inf->shape.size() != 4 || inf->shape[2] != 3) {
        return fail(fmt("{}: not a 3x3 conv", nm)());
      }
      const bool first = (C == 0);
      if (first) {
        C = (int)inf->shape[1];
        _info.latent_channels = C;
        op.in_ch = C;
      }
      if ((int)inf->shape[1] != C) {
        return fail(fmt("{}: conv reads {} channels where {} flow in", nm,
                        inf->shape[1], C)());
      }
      const bool head = (i == max_idx);
      const int pad = first ? ((C + 31) / 32) * 32 : 0;
      if (!conv3(nm, pad, &op.c,
                 head && (int)inf->shape[0] <= kSmallCoutMax)) {
        return fail(fmt("{}: unreadable conv", nm)());
      }
      op.out_ch = op.c.cout;
      C = op.out_ch;
    } else if (!suffixes[i].empty()) {
      return fail(fmt("{}: tensors ('{}', ...) of no kind this reader "
                      "knows", nm, suffixes[i].front())());
    } else if (i == 0) {
      // The input Clamp, tanh(x / 3) * 3, applied on the host at upload:
      // the latent is the one tensor here that starts out there.
      _clamp_input = true;
      continue;
    } else if (!_ops.empty() && _ops.back().kind == Op::kConv) {
      op.kind = Op::kRelu;
      op.out_ch = C;
    } else if (!_ops.empty()) {
      op.kind = Op::kUpsample;
      op.out_ch = C;
      scale *= 2;
    } else {
      return fail(fmt("{}: an empty slot before any layer", nm)());
    }
    _ops.push_back(std::move(op));
  }
  if (_ops.empty() || _ops.back().kind != Op::kConv) {
    return fail("the decoder does not end in a conv");
  }
  // The head writes image_channels * patch^2: RGB for every TAE but
  // taeqi2_1, whose Qwen-Image-2.1 VAE is RGBA (straight alpha). RGB is
  // tried first, and the two never collide on a real head: 12 is RGB at
  // patch 2, 16 is RGBA at patch 2.
  const int head = _ops.back().c.cout;
  int p = 0, ic = 0;
  for (const int c : {3, 4}) {
    int q = 1;
    while (c * q * q < head) { ++q; }
    if (c * q * q == head) {
      p = q;
      ic = c;
      break;
    }
  }
  if (p == 0) {
    return fail(fmt("the head writes {} channels, not 3 or 4 * patch^2",
                    head)());
  }
  _info.patch = p;
  _info.image_channels = ic;
  _info.spatial = scale * p;
  _info.temporal = _n_mem > 0;

  // A 1x1 TGrow commutes with a NEAREST upsample, so run it first, at a
  // quarter of the pixels. Exact, and in the stock TAEHV layout it is
  // every stride-2 TGrow.
  for (std::size_t i = 0; i + 1 < _ops.size(); ++i) {
    if (_ops[i].kind == Op::kUpsample && _ops[i + 1].kind == Op::kTGrow) {
      Op up = std::move(_ops[i]);
      Op tg = std::move(_ops[i + 1]);
      tg.scale = up.scale;   // `scale` is the INPUT's; the upsample's holds
      _ops[i] = std::move(tg);
      _ops[i + 1] = std::move(up);
      ++i;
    }
  }
  _last_state_op = 0;
  for (std::size_t i = 0; i < _ops.size(); ++i) {
    if (_ops[i].kind == Op::kMemBlock) { _last_state_op = i; }
  }
  return true;
}

bool
MetalTaeDecoder::build_kernels_(std::string* err)
{
  _lib_gemm = _mc->load_library("dense_gemm");
  _lib_elt  = _mc->load_library("llm_elementwise");
  _fn_conv_bn32  = _lib_gemm.function("conv3x3_gemm_s1_bn32_f16");
  _fn_conv_bn64  = _lib_gemm.function("conv3x3_gemm_s1_bn64_f16");
  _fn_gemm_bm64     = _lib_gemm.function("dense_gemm_t_bm64_f16");
  _fn_gemm_bm64bn64 = _lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  _fn_small_cout = _lib_elt.function("conv3x3_hwc_small_cout_f16");
  _fn_relu     = _lib_elt.function("clamp_f16");
  _fn_add      = _lib_elt.function("residual_add_f16");
  _fn_upsample = _lib_elt.function("upsample_nearest2x_hwc_f16");
  _fn_concat   = _lib_elt.function("concat_cols_f16");
  _fn_copy     = _lib_elt.function("copy_f16");
  _fn_gn_stats  = _lib_elt.function("group_norm_chan_stats_f16");
  _fn_gn_reduce = _lib_elt.function("group_norm_group_stats_f32");
  _fn_gn_apply  = _lib_elt.function("group_norm_apply_f16");
  // Everything the forward can reach. An unresolved name binds nothing
  // and the op it serves would run its predecessor's kernel again -- the
  // loader refuses rather than find out at decode.
  const std::pair<const char*, const metal_compute::ComputeFunction*>
      need[] = {
          {"conv3x3_gemm_s1_bn32_f16", &_fn_conv_bn32},
          {"conv3x3_gemm_s1_bn64_f16", &_fn_conv_bn64},
          {"dense_gemm_t_bm64_f16", &_fn_gemm_bm64},
          {"conv3x3_hwc_small_cout_f16", &_fn_small_cout},
          {"clamp_f16", &_fn_relu},
          {"residual_add_f16", &_fn_add},
          {"upsample_nearest2x_hwc_f16", &_fn_upsample},
          {"concat_cols_f16", &_fn_concat},
          {"copy_f16", &_fn_copy},
          {"group_norm_chan_stats_f16", &_fn_gn_stats},
          {"group_norm_group_stats_f32", &_fn_gn_reduce},
          {"group_norm_apply_f16", &_fn_gn_apply},
      };
  for (const auto& [name, fn] : need) {
    if (!fn->valid()) {
      if (err != nullptr) { *err = fmt("kernel {} did not load", name)(); }
      return false;
    }
  }
  return true;
}

int
MetalTaeDecoder::frames_for(int latent_frames, Trim trim) const
{
  return frames_for(latent_frames, _info.time_upscale, trim);
}

int
MetalTaeDecoder::frames_for(int latent_frames, int tu, Trim trim)
{
  if (latent_frames <= 0) { return 0; }
  int n = 0;
  for (long j = 0; j < (long)latent_frames * tu; ++j) {
    if (kept_(j, tu, trim)) { ++n; }
  }
  return n;
}

std::size_t
MetalTaeDecoder::resident_bytes() const noexcept
{
  return _weight_bytes + _scratch_bytes;
}

MetalTaeDecoder::Scratch
MetalTaeDecoder::scratch_for_(int h, int w) const
{
  Scratch z;
  const std::size_t hw = (std::size_t)h * w;
  z.in = hw * (std::size_t)_ops.front().c.cin;
  z.op_out.assign(_ops.size(), 0);
  z.mem.assign((std::size_t)_n_mem, 0);
  for (std::size_t i = 0; i < _ops.size(); ++i) {
    const Op& op = _ops[i];
    const std::size_t m_in = hw * op.scale * op.scale;
    switch (op.kind) {
      case Op::kRelu:
        break;   // in place
      case Op::kConv:
        // The head writes its per-frame slot instead.
        if (i + 1 < _ops.size()) { z.op_out[i] = m_in * op.out_ch; }
        break;
      case Op::kUpsample:
        z.op_out[i] = m_in * 4 * op.out_ch;
        break;
      case Op::kTGrow:
        z.op_out[i] = m_in * op.out_ch;
        break;
      case Op::kMemBlock:
        z.mem[(std::size_t)op.mem] = m_in * op.in_ch;
        z.cat = std::max(z.cat, m_in * 2 * op.in_ch);
        [[fallthrough]];
      case Op::kBlock:
        z.op_out[i] = m_in * op.out_ch;
        z.tmp = std::max(z.tmp, m_in * op.out_ch);
        if (op.has_skip) { z.skip = std::max(z.skip, m_in * op.out_ch); }
        if (op.has_pool) {
          const std::size_t n = (std::size_t)op.pool_in.cout;
          z.pool = std::max(z.pool, m_in * n);
          z.pooled = std::max(z.pooled, m_in * op.in_ch);
          z.gn_part_bytes = std::max(z.gn_part_bytes,
                                     (std::size_t)kGnBlocks * 2 * n * 4);
          z.gn_stats_bytes = (std::size_t)2 * kPoolGroups * 4;
        }
        break;
    }
  }
  const Op& head = _ops.back();
  z.head_frame = hw * head.scale * head.scale * head.c.cout;
  return z;
}

std::size_t
MetalTaeDecoder::Scratch::fixed_bytes() const
{
  std::size_t n = in + cat + 3 * tmp + skip;
  if (gn_part_bytes > 0) { n += 2 * pool + pooled; }
  for (std::size_t e : op_out) { n += e; }
  for (std::size_t e : mem) { n += e; }
  return n * 2 + gn_part_bytes + gn_stats_bytes;
}

int
MetalTaeDecoder::slots_per_commit_(std::size_t head_frame, int want) const
{
  const std::size_t slot_bytes = std::max<std::size_t>(head_frame * 2, 1);
  return std::max(_info.time_upscale,
                  (int)std::min<std::size_t>((std::size_t)std::max(want, 1),
                                             kHeadBudget / slot_bytes));
}

bool
MetalTaeDecoder::plan(const std::string& file, int T, int h, int w,
                      Trim trim, Plan* out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (out == nullptr || T <= 0 || h <= 0 || w <= 0) {
    return fail("empty geometry");
  }
  auto src = MetalLlamaWeights::open_model(file);
  if (!src) { return fail(fmt("no readable checkpoint at '{}'", file)()); }
  MetalTaeDecoder d;
  std::string e;
  if (!d.parse_(*src, /*read=*/false, &e)) { return fail(e); }
  const Scratch z = d.scratch_for_(h, w);
  const int want = frames_for(T, d._info.time_upscale, trim);
  out->info = d._info;
  out->weight_bytes = d._weight_bytes;
  out->scratch_bytes =
      z.fixed_bytes() +
      (std::size_t)T * z.in * 2 +   // the whole latent, uploaded once
      (std::size_t)d.slots_per_commit_(z.head_frame, want) * z.head_frame *
          2;
  return true;
}

bool
MetalTaeDecoder::ensure_scratch_(int h, int w)
{
  if (h == _sh && w == _sw && !_op_out.empty()) { return true; }
  _op_out.clear();
  _mem.clear();
  _scratch_bytes = 0;
  MetalCompute* mc = _mc;
  const Scratch z = scratch_for_(h, w);
  auto take = [&](std::size_t elems) {
    SharedBuffer b = zeros_(mc, elems);
    _scratch_bytes += b.byte_size();
    return b;
  };
  _in = take(z.in);
  _op_out.resize(_ops.size());
  _mem.resize((std::size_t)_n_mem);
  for (std::size_t i = 0; i < _ops.size(); ++i) {
    if (z.op_out[i] > 0) { _op_out[i] = take(z.op_out[i]); }
  }
  for (std::size_t k = 0; k < z.mem.size(); ++k) { _mem[k] = take(z.mem[k]); }
  _cat  = take(z.cat);
  _tmp0 = take(z.tmp);
  _tmp1 = take(z.tmp);
  _tmp2 = take(z.tmp);
  _skip = take(z.skip);
  if (z.gn_part_bytes > 0) {
    _pool_a = take(z.pool);
    _pool_b = take(z.pool);
    _pooled = take(z.pooled);
    // f32, so sized in bytes: half as many f16 elements.
    _gn_part  = take(z.gn_part_bytes / 2);
    _gn_stats = take(z.gn_stats_bytes / 2);
  }
  // One head slot per output frame, at the head's input resolution (the
  // pixel shuffle happens on the host). decode() sizes the slot count:
  // it is a property of the clip, not of the geometry.
  _head_frame = z.head_frame;
  _head = {};
  _in_all = {};
  bool ok = !_in.empty() && !_cat.empty();
  for (std::size_t i = 0; i < _ops.size(); ++i) {
    if (z.op_out[i] > 0 && _op_out[i].empty()) { ok = false; }
  }
  for (const SharedBuffer& b : _mem) {
    if (b.empty()) { ok = false; }
  }
  if (!ok) {
    _op_out.clear();
    _sh = _sw = 0;
    return false;
  }
  _sh = h;
  _sw = w;
  return true;
}

void
MetalTaeDecoder::conv3_(ComputeEncoder& enc, const Conv& c,
                        const SharedBuffer& in, const SharedBuffer& out,
                        int H, int W)
{
  const int M = H * W;
  // BN 64 wherever the output is that wide: the FLUX.2 VAE measured it
  // ahead of BN 128 once the gather vectorized (the narrower tile's
  // smaller threadgroup footprint wins on occupancy).
  const int bn = c.cout >= 64 ? 64 : 32;
  enc.set_function(bn == 64 ? _fn_conv_bn64 : _fn_conv_bn32);
  enc.set_buffer(0, in);
  enc.set_buffer(1, c.w);
  enc.set_buffer(2, c.b.empty() ? c.w : c.b);
  enc.set_buffer(3, out);
  enc.set_constant(4, H);      enc.set_constant(5, W);
  enc.set_constant(6, c.cin);  enc.set_constant(7, c.cout);
  enc.set_constant(8, H);      enc.set_constant(9, W);
  enc.set_constant(10, c.b.empty() ? 0 : 1);
  enc.dispatch({(unsigned)(((c.cout + bn - 1) / bn) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
MetalTaeDecoder::conv1_(ComputeEncoder& enc, const Conv& c,
                        std::size_t w_row0, int rows, const SharedBuffer& in,
                        const SharedBuffer& out, int M)
{
  // y[M, rows] = x[M, cin] @ w[w_row0 : w_row0 + rows, cin]^T. The row
  // offset is how a TGrow's sub-frame k takes its channel block without a
  // reshape: the block IS a contiguous band of the weight.
  const bool wide = rows >= 64 && _fn_gemm_bm64bn64.valid();
  const int bn = wide ? 64 : 32;
  enc.set_function(wide ? _fn_gemm_bm64bn64 : _fn_gemm_bm64);
  enc.set_buffer(0, in);
  enc.set_buffer(1, c.w, w_row0 * (std::size_t)c.cin * 2);
  enc.set_buffer(2, c.w);
  enc.set_buffer(3, out);
  enc.set_constant(4, c.cin);
  enc.set_constant(5, rows);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((rows + bn - 1) / bn) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
MetalTaeDecoder::relu_(ComputeEncoder& enc, const SharedBuffer& x,
                       std::size_t n)
{
  enc.set_function(_fn_relu);
  enc.set_buffer(0, x);
  enc.set_buffer(1, x);
  enc.set_constant(2, (int)n);
  enc.set_constant(3, 0.0f);
  enc.set_constant(4, kF16Max);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

void
MetalTaeDecoder::run_(ComputeEncoder& enc, std::size_t i,
                      const SharedBuffer& x, int H, int W, long raw,
                      const std::function<int(long)>& slot_of)
{
  const SharedBuffer* cur = &x;
  for (; i < _ops.size(); ++i) {
    const Op& op = _ops[i];
    const int M = H * W;
    // Past the last stateful op a frame nobody keeps has nothing left to
    // feed. How many raw frames this path still fans out to is the
    // product of the TGrows ahead of it.
    if (i > _last_state_op) {
      long fan = 1;
      for (std::size_t k = i; k < _ops.size(); ++k) {
        if (_ops[k].kind == Op::kTGrow) { fan *= _ops[k].stride; }
      }
      bool any = false;
      for (long k = 0; k < fan && !any; ++k) {
        any = slot_of(raw * fan + k) >= 0;
      }
      if (!any) { return; }
    }
    switch (op.kind) {
      case Op::kConv: {
        if (i + 1 == _ops.size()) {
          const int slot = slot_of(raw);
          if (slot < 0) { return; }
          const std::size_t off = (std::size_t)slot * _head_frame * 2;
          if (op.c.cout <= kSmallCoutMax && !op.c.hwio.empty()) {
            enc.set_function(_fn_small_cout);
            enc.set_buffer(0, *cur);
            enc.set_buffer(1, op.c.hwio);
            enc.set_buffer(2, op.c.b.empty() ? op.c.hwio : op.c.b);
            enc.set_buffer(3, _head, off);
            enc.set_constant(4, W);        enc.set_constant(5, H);
            enc.set_constant(6, op.c.cin); enc.set_constant(7, op.c.cout);
            enc.set_constant(8, op.c.b.empty() ? 0 : 1);
            enc.dispatch({(unsigned)W, (unsigned)H, 1}, {256, 1, 1});
          } else {
            // A head too wide for the direct conv writes its slot through
            // the gather conv; bind the slot as its own buffer offset.
            const SharedBuffer& o = _head;
            int bn = op.c.cout >= 64 ? 64 : 32;
            enc.set_function(bn == 64 ? _fn_conv_bn64 : _fn_conv_bn32);
            enc.set_buffer(0, *cur);
            enc.set_buffer(1, op.c.w);
            enc.set_buffer(2, op.c.b.empty() ? op.c.w : op.c.b);
            enc.set_buffer(3, o, off);
            enc.set_constant(4, H);         enc.set_constant(5, W);
            enc.set_constant(6, op.c.cin);  enc.set_constant(7, op.c.cout);
            enc.set_constant(8, H);         enc.set_constant(9, W);
            enc.set_constant(10, op.c.b.empty() ? 0 : 1);
            enc.dispatch({(unsigned)(((op.c.cout + bn - 1) / bn) * 32),
                          (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
          }
          return;
        }
        conv3_(enc, op.c, *cur, _op_out[i], H, W);
        cur = &_op_out[i];
        break;
      }
      case Op::kRelu:
        relu_(enc, *cur, (std::size_t)M * op.out_ch);
        break;
      case Op::kUpsample: {
        enc.set_function(_fn_upsample);
        enc.set_buffer(0, *cur);
        enc.set_buffer(1, _op_out[i]);
        enc.set_constant(2, H);
        enc.set_constant(3, W);
        enc.set_constant(4, op.out_ch);
        enc.dispatch({(unsigned)op.out_ch, (unsigned)(4 * M), 1},
                     {(unsigned)std::min(op.out_ch, 256), 1, 1});
        cur = &_op_out[i];
        H *= 2;
        W *= 2;
        break;
      }
      case Op::kTGrow: {
        // Depth first, sub-frame 0 first: that is the order the frames
        // come out in, and the order every MemBlock below must see them.
        // Each sub-frame's GEMM overwrites this op's output only after
        // everything the previous one fed has been encoded, and the
        // encoder is serial.
        for (int k = 0; k < op.stride; ++k) {
          conv1_(enc, op.c, (std::size_t)k * op.in_ch, op.in_ch, *cur,
                 _op_out[i], M);
          run_(enc, i + 1, _op_out[i], H, W, raw * op.stride + k, slot_of);
        }
        return;
      }
      case Op::kBlock:
      case Op::kMemBlock: {
        const int C = op.in_ch, N = op.out_ch;
        if (op.has_pool) {
          // x' = x + pool_out(relu(GroupNorm(pool_in(x)))), then the block
          // reads x' everywhere it read x -- its convs AND its skip.
          const int n = op.pool_in.cout;
          conv1_(enc, op.pool_in, 0, n, *cur, _pool_a, M);
          const int NB = std::min(kGnBlocks, std::max(1, M));
          enc.set_function(_fn_gn_stats);
          enc.set_buffer(0, _pool_a);
          enc.set_buffer(1, _gn_part);
          enc.set_constant(2, M);
          enc.set_constant(3, n);
          enc.set_constant(4, NB);
          enc.dispatch({(unsigned)(256 * NB), 1, 1}, {256, 1, 1});
          enc.set_function(_fn_gn_reduce);
          enc.set_buffer(0, _gn_part);
          enc.set_buffer(1, _gn_stats);
          enc.set_constant(2, M);
          enc.set_constant(3, n);
          enc.set_constant(4, kPoolGroups);
          enc.set_constant(5, NB);
          enc.set_constant(6, kGnEps);
          enc.dispatch({(unsigned)(256 * kPoolGroups), 1, 1}, {256, 1, 1});
          enc.set_function(_fn_gn_apply);
          enc.set_buffer(0, _pool_a);
          enc.set_buffer(1, op.gn_gamma);
          enc.set_buffer(2, op.gn_beta);
          enc.set_buffer(3, _pool_b);
          enc.set_buffer(4, _gn_stats);
          enc.set_constant(5, n);
          enc.set_constant(6, kPoolGroups);
          enc.dispatch({(unsigned)n, (unsigned)M, 1}, {256, 1, 1});
          relu_(enc, _pool_b, (std::size_t)M * n);
          conv1_(enc, op.pool_out, 0, C, _pool_b, _pooled, M);
          enc.set_function(_fn_add);
          enc.set_buffer(0, *cur);
          enc.set_buffer(1, _pooled);
          enc.set_buffer(2, _pooled);
          enc.set_constant(3, M * C);
          enc.dispatch({(unsigned)((std::size_t)M * C), 1, 1}, {256, 1, 1});
          cur = &_pooled;
        }
        const SharedBuffer* in0 = cur;
        if (op.kind == Op::kMemBlock) {
          // [x, past]: the current frame's channels first.
          const SharedBuffer& mem = _mem[(std::size_t)op.mem];
          enc.set_function(_fn_concat);
          enc.set_buffer(0, *cur);
          enc.set_buffer(1, mem);
          enc.set_buffer(2, _cat);
          enc.set_constant(3, M);
          enc.set_constant(4, C);
          enc.set_constant(5, C);
          enc.dispatch({(unsigned)((std::size_t)M * 2 * C), 1, 1},
                       {256, 1, 1});
          in0 = &_cat;
        }
        conv3_(enc, op.c0, *in0, _tmp0, H, W);
        relu_(enc, _tmp0, (std::size_t)M * N);
        conv3_(enc, op.c1, _tmp0, _tmp1, H, W);
        relu_(enc, _tmp1, (std::size_t)M * N);
        conv3_(enc, op.c2, _tmp1, _tmp2, H, W);
        if (op.kind == Op::kMemBlock) {
          // The NEXT frame's past is THIS frame's input. After the concat
          // has read the old one; the encoder orders the two.
          enc.set_function(_fn_copy);
          enc.set_buffer(0, *cur);
          enc.set_buffer(1, _mem[(std::size_t)op.mem]);
          enc.set_constant(2, 0);
          enc.set_constant(3, M * C);
          enc.dispatch({(unsigned)((std::size_t)M * C), 1, 1}, {256, 1, 1});
        }
        const SharedBuffer* skip = cur;
        if (op.has_skip) {
          conv1_(enc, op.skip, 0, N, *cur, _skip, M);
          skip = &_skip;
        }
        enc.set_function(_fn_add);
        enc.set_buffer(0, _tmp2);
        enc.set_buffer(1, *skip);
        enc.set_buffer(2, _op_out[i]);
        enc.set_constant(3, M * N);
        enc.dispatch({(unsigned)((std::size_t)M * N), 1, 1}, {256, 1, 1});
        relu_(enc, _op_out[i], (std::size_t)M * N);
        cur = &_op_out[i];
        break;
      }
    }
  }
}

bool
MetalTaeDecoder::decode(const float* latent, int C, int T, int h, int w,
                        Trim trim, int max_frames, int shrink, Frames* out,
                        std::string* err,
                        const std::function<bool()>& cancel)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  if (err != nullptr) { err->clear(); }
  if (latent == nullptr || out == nullptr || T <= 0 || h <= 0 || w <= 0) {
    return fail("empty latent");
  }
  if (C != _info.latent_channels) {
    return fail(fmt("the latent has {} channels and this decoder reads {}",
                    C, _info.latent_channels)());
  }
  if (!ensure_scratch_(h, w)) {
    return fail(fmt("scratch for a {}x{} latent did not allocate", w, h)());
  }
  const int tu = _info.time_upscale;
  const int p = _info.patch;
  const Op& head = _ops.back();
  const int Hh = h * head.scale, Wh = w * head.scale;   // head input
  const int NH = Hh * p, NW = Wh * p;   // native picture
  const int k = std::max(1, shrink);
  int OH = NH / k, OW = NW / k;
  if (k > 1) {
    OH &= ~1;
    OW &= ~1;
  }
  if (OH <= 0 || OW <= 0) {
    return fail(fmt("shrinking {}x{} by {} leaves no picture", NW, NH, k)());
  }
  int want = frames_for(T, trim);
  if (max_frames > 0) { want = std::min(want, max_frames); }
  out->height = OH;
  out->width = OW;
  out->frames = 0;
  const int ic = _info.image_channels;
  out->channels = ic;
  out->rgb.assign((std::size_t)want * ic * OH * OW, 0);
  // One native frame as f32 [3, NH, NW], the pixel shuffle's target and
  // the box filter's source.
  std::vector<float> native(k > 1 ? (std::size_t)ic * NH * NW : 0);

  // Every memory starts at zeros -- the first frame's past.
  for (const SharedBuffer& m : _mem) {
    std::memset(m.contents(), 0, m.byte_size());
  }
  metal_compute::CommandStream cs = _mc->make_command_stream();
  if (!cs.valid()) { return fail("no command stream"); }

  // The whole latent, channel-last, zero-padded to the first conv's width
  // and through the input Clamp, uploaded ONCE: each latent frame is
  // copied into the first op's input on the GPU, in encode order.
  const int cin = _ops.front().c.cin;
  const std::size_t hw = (std::size_t)h * w;
  const std::size_t in_frame = hw * (std::size_t)cin;
  if (_in_all.byte_size() < (std::size_t)T * in_frame * 2) {
    _scratch_bytes -= _in_all.byte_size();
    _in_all = _mc->make_shared_buffer((std::size_t)T * in_frame * 2);
    if (_in_all.empty()) { return fail("latent upload did not allocate"); }
    _scratch_bytes += _in_all.byte_size();
  }
  {
    auto* in = static_cast<_Float16*>(_in_all.contents());
    for (int t = 0; t < T; ++t) {
      for (std::size_t px = 0; px < hw; ++px) {
        for (int c = 0; c < cin; ++c) {
          float v = 0.0f;
          if (c < C) {
            v = latent[((std::size_t)c * T + t) * hw + px];
            if (_clamp_input) { v = std::tanh(v / 3.0f) * 3.0f; }
          }
          in[(std::size_t)t * in_frame + px * cin + c] = (_Float16)v;
        }
      }
    }
  }

  // AS FEW COMMITS AS THE OUTPUT ALLOWS. A preview decodes while a DiT
  // owns the GPU, and a held DiT forward is ONE command buffer that runs
  // for seconds: every commit here queues behind one of them. MEASURED
  // with a commit per latent frame on a resident H3 at 512x288 -- 12
  // commits, and a preview took several denoise steps to come out. So a
  // decode commits once, or in as many pieces as its head outputs need
  // to fit kHeadBudget.
  const std::size_t slot_bytes = _head_frame * 2;
  const int per_commit = slots_per_commit_(_head_frame, want);
  if (_head.byte_size() < (std::size_t)per_commit * slot_bytes) {
    _scratch_bytes -= _head.byte_size();
    _head = _mc->make_shared_buffer((std::size_t)per_commit * slot_bytes);
    if (_head.empty()) { return fail("preview output did not allocate"); }
    _scratch_bytes += _head.byte_size();
  }

  const std::size_t plane = (std::size_t)OH * OW;
  const std::size_t nplane = (std::size_t)NH * NW;
  const int hc = head.c.cout;
  int t = 0;
  while (t < T && out->frames < want) {
    // Encode latent frames until the head slots are full.
    std::map<long, int> slots;
    int used = 0;
    {
      ComputeEncoder enc = cs.begin_compute();
      while (t < T && out->frames + used < want &&
             used + tu <= per_commit) {
        if (cancel && cancel()) { return false; }
        for (int s = 0; s < tu; ++s) {
          const long raw = (long)t * tu + s;
          if (kept_(raw, tu, trim) && out->frames + used < want) {
            slots[raw] = used++;
          }
        }
        enc.set_function(_fn_copy);
        enc.set_buffer(0, _in_all, (std::size_t)t * in_frame * 2);
        enc.set_buffer(1, _in);
        enc.set_constant(2, 0);
        enc.set_constant(3, (int)in_frame);
        enc.dispatch({(unsigned)in_frame, 1, 1}, {256, 1, 1});
        const std::function<int(long)> slot_of = [&slots](long raw) {
          auto it = slots.find(raw);
          return it == slots.end() ? -1 : it->second;
        };
        run_(enc, 0, _in, h, w, (long)t, slot_of);
        ++t;
      }
      enc.end();
    }
    std::string ferr;
    if (!cs.commit().wait_ok(&ferr)) {
      return fail(fmt("latent frame {}: {}", t - 1,
                      ferr.empty() ? "GPU failure" : ferr)());
    }
    // Pixel shuffle + [0, 1] clamp on the host: pic[c, y * p + i, x * p +
    // j] = head[(y, x), c * p^2 + i * p + j]; then the box filter.
    const auto* hd = static_cast<const _Float16*>(_head.contents());
    for (int s = 0; s < used; ++s) {
      const _Float16* src = hd + (std::size_t)s * _head_frame;
      std::uint8_t* dst = out->rgb.data() +
                          (std::size_t)out->frames * ic * plane;
      for (int y = 0; y < Hh; ++y) {
        for (int x = 0; x < Wh; ++x) {
          const _Float16* px = src + ((std::size_t)y * Wh + x) * hc;
          for (int c = 0; c < ic; ++c) {
            for (int i = 0; i < p; ++i) {
              for (int j = 0; j < p; ++j) {
                float v = (float)px[c * p * p + i * p + j];
                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                const std::size_t py = (std::size_t)(y * p + i);
                const std::size_t qx = (std::size_t)(x * p + j);
                if (k == 1) {
                  dst[(std::size_t)c * plane + py * OW + qx] =
                      (std::uint8_t)std::lround(v * 255.0f);
                } else {
                  native[(std::size_t)c * nplane + py * NW + qx] = v;
                }
              }
            }
          }
        }
      }
      if (k > 1) {
        // A box filter per output pixel. With alpha it is PREMULTIPLIED:
        // colour is weighted by alpha before averaging and divided back
        // out, or the arbitrary colour under transparent pixels -- black,
        // on a generated RGBA picture -- bleeds into every edge.
        const float inv = 1.0f / (float)(k * k);
        const float* ap =
            ic == 4 ? native.data() + (std::size_t)3 * nplane : nullptr;
        for (int y = 0; y < OH; ++y) {
          for (int x = 0; x < OW; ++x) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int dy = 0; dy < k; ++dy) {
              const std::size_t row =
                  (std::size_t)(y * k + dy) * NW + (std::size_t)x * k;
              for (int dx = 0; dx < k; ++dx) {
                const float a = ap != nullptr ? ap[row + dx] : 1.0f;
                for (int c = 0; c < 3; ++c) {
                  acc[c] += native[(std::size_t)c * nplane + row + dx] * a;
                }
                acc[3] += a;
              }
            }
            const std::size_t o = (std::size_t)y * OW + x;
            const float wa = acc[3] > 0.0f ? 1.0f / acc[3] : 0.0f;
            for (int c = 0; c < 3; ++c) {
              dst[(std::size_t)c * plane + o] =
                  (std::uint8_t)std::lround(acc[c] * wa * 255.0f);
            }
            if (ic == 4) {
              dst[(std::size_t)3 * plane + o] =
                  (std::uint8_t)std::lround(acc[3] * inv * 255.0f);
            }
          }
        }
      }
      ++out->frames;
    }
  }
  out->rgb.resize((std::size_t)out->frames * ic * plane);
  return true;
}

}  // namespace genai
}  // namespace vpipe
