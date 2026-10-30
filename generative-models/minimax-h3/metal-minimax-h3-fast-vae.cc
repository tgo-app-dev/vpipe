#include "generative-models/minimax-h3/metal-minimax-h3-fast-vae.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/minimax-h3/metal-minimax-h3-video-vae.h"
#include "generative-models/shared/i8-gemm.h"
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
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;
namespace fs = std::filesystem;

namespace {

// The derived-tensor keys this class files its transforms under.
constexpr const char* kKey = "h3-fast-vae/";

// Rows a GEMM, a norm or a residual add runs at once. Bounds every
// row-sized scratch but the stream and its bf16 twin.
constexpr int kBand = 8192;

// The flash kernels' query and key blocks: steel's, and the matrix-core
// one's.
constexpr int kBqAlu = 32, kBkAlu = 16;
constexpr int kBqNax = 64, kBkNax = 32;

std::uint16_t
to_bf16_(float f)
{
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

float
elem_f32_(const void* p, const std::string& dtype, std::size_t i)
{
  if (dtype == "F32") { return static_cast<const float*>(p)[i]; }
  if (dtype == "F16") {
    return (float)static_cast<const _Float16*>(p)[i];
  }
  const std::uint32_t u =
      (std::uint32_t)static_cast<const std::uint16_t*>(p)[i] << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

bool
float_dtype_(const std::string& d)
{
  return d == "F32" || d == "F16" || d == "BF16";
}

std::size_t
numel_(const std::vector<std::int64_t>& sh)
{
  std::size_t n = 1;
  for (auto d : sh) { n *= (std::size_t)d; }
  return n;
}

// A tensor read UNCACHED (Copied: the file is 6.5 GB of f32, and a mapped
// view would make all of it resident) and handed to `fn` with its dtype.
// The read buffer dies here, so a converted product built inside `fn` is
// all that stays.
template <class Fn>
bool
with_tensor_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
             std::size_t want, Fn&& fn)
{
  const auto* info = ws.src().info(nm);
  if (info == nullptr || !float_dtype_(info->dtype) ||
      numel_(info->shape) != want) {
    return false;
  }
  SharedBuffer raw = ws.read(nm, mc, WeightSet::Residency::Copied);
  if (raw.empty() || raw.contents() == nullptr) { return false; }
  return fn(raw.contents(), info->dtype);
}

// A tensor narrowed to bf16, straight from the read buffer (no host f32
// copy: macOS keeps a freed large heap block dirty in the footprint).
SharedBuffer
bf16_tensor_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
             std::size_t want)
{
  SharedBuffer out;
  with_tensor_(ws, mc, nm, want, [&](const void* p, const std::string& dt) {
    out = mc->make_shared_buffer(want * 2);
    if (out.empty()) { return false; }
    auto* d = static_cast<std::uint16_t*>(out.contents());
    for (std::size_t i = 0; i < want; ++i) {
      d[i] = to_bf16_(elem_f32_(p, dt, i));
    }
    return true;
  });
  return out;
}

// ...and widened to f32 (the norms, gates and the head's bias).
SharedBuffer
f32_tensor_(WeightSet& ws, MetalCompute* mc, const std::string& nm,
            std::size_t want)
{
  SharedBuffer out;
  with_tensor_(ws, mc, nm, want, [&](const void* p, const std::string& dt) {
    out = mc->make_shared_buffer(want * 4);
    if (out.empty()) { return false; }
    auto* d = static_cast<float*>(out.contents());
    for (std::size_t i = 0; i < want; ++i) { d[i] = elem_f32_(p, dt, i); }
    return true;
  });
  return out;
}

std::string
blk_(int i, const char* rest)
{
  return "blocks." + std::to_string(i) + "." + rest;
}

}  // namespace

int
MetalMiniMaxH3FastVae::effective_i8_tier(int i8_tier, bool matrix_cores)
{
  bool on = i8_tier >= 0;
  if (const char* e = std::getenv("VPIPE_I8_GEMM")) { on = std::atoi(e) != 0; }
  if (!on || !matrix_cores) { return -1; }
  int t = std::clamp(i8_tier, 0, 3);
  bool rot = t == 2;
  if (const char* e = std::getenv("VPIPE_I8_TIER")) {
    if (std::strcmp(e, "token") == 0 || std::strcmp(e, "2") == 0) {
      t = 2;
    } else if (std::strcmp(e, "fast") == 0 || std::strcmp(e, "1") == 0) {
      t = 1;
    } else if (std::strcmp(e, "accurate") == 0 || std::strcmp(e, "0") == 0) {
      t = 0;
    }
  }
  if (const char* e = std::getenv("VPIPE_I8_TOKEN_ROT")) {
    rot = std::atoi(e) != 0;
  }
  if (t >= 2) { return rot ? 2 : 3; }
  return t;
}

namespace {

// The fast tier's group for K, as I8GemmContext::fast_group picks it.
int
fast_group_for_(int K)
{
  int target = 512;
  if (const char* e = std::getenv("VPIPE_I8_FAST_GROUP")) {
    target = std::atoi(e);
  }
  for (int g : {1024, 512, 256}) {
    if (g <= target && K % g == 0) { return g; }
  }
  return 0;
}

struct AttnP {
  int B, H, D, qL, kL, gqa_factor;
  float scale;
  int NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
  std::int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};

// The qkv post kernel's geometry (h3fv_qkv_post_f16's H3fvPost).
struct PostP {
  int tok0, T, H, W, s, st, tb, ws, ring_rows, rope, ring, heads;
};

}  // namespace

// ---- files and config ----------------------------------------------------

std::string
MetalMiniMaxH3FastVae::resolve_weights(const std::string& path,
                                       std::string* err)
{
  std::error_code ec;
  const fs::path p(path);
  auto st_in = [&](const fs::path& d) {
    std::vector<fs::path> v;
    for (const auto& e : fs::directory_iterator(d, ec)) {
      if (e.is_regular_file(ec) && e.path().extension() == ".safetensors") {
        v.push_back(e.path());
      }
    }
    std::sort(v.begin(), v.end());
    return v;
  };
  if (fs::is_regular_file(p, ec)) { return path; }
  if (!fs::is_directory(p, ec)) {
    if (err != nullptr) { *err = "'" + path + "' does not exist"; }
    return {};
  }
  for (const fs::path& d : {p / "vae_fast", p}) {
    if (!fs::is_directory(d, ec)) { continue; }
    const auto v = st_in(d);
    if (v.size() == 1) { return v[0].string(); }
    if (v.size() > 1) {
      if (err != nullptr) {
        *err = "'" + d.string() + "' holds " + std::to_string(v.size()) +
               " .safetensors files; name the fast decoder's file";
      }
      return {};
    }
  }
  if (err != nullptr) {
    *err = "no fast H3 decoder in '" + path + "' (expected "
           "vae_fast/<file>.safetensors or one .safetensors)";
  }
  return {};
}

bool
MetalMiniMaxH3FastVae::config_from(const WeightSet& ws,
                                   const std::string& vae_dir, Config* out,
                                   std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  MetalMiniMaxH3VideoVae::Config vc;
  std::string verr;
  if (!MetalMiniMaxH3VideoVae::config_from_json(vae_dir, vc, &verr)) {
    return fail("the released VAE's config: " + verr);
  }
  Config c;
  c.z_channels     = vc.z_channels;
  c.out_channels   = vc.out_channels;
  c.n_heads        = vc.n_heads;
  c.head_dim       = vc.head_dim;
  c.patch          = vc.patch;
  c.patch_t        = vc.patch_t;
  c.n_register     = vc.n_register;
  c.rope_theta     = vc.rope_theta;
  c.rope_dim_ratio = vc.rope_dim_ratio;
  c.norm_eps       = vc.norm_eps;
  c.latents_mean   = vc.latents_mean;
  c.latents_std    = vc.latents_std;
  const MetalLlamaWeights& src = ws.src();
  const auto* n1 = src.info("blocks.0.norm1.weight");
  const auto* w2 = src.info("blocks.0.ff.w2.weight");
  const auto* xe = src.info("x_embedder.weight");
  const auto* rt = src.info("register_tokens");
  const auto* po = src.info("proj_out.weight");
  if (n1 == nullptr || w2 == nullptr || xe == nullptr || rt == nullptr ||
      po == nullptr || src.info("post_quant_conv.weight") == nullptr ||
      src.info("norm_out.bias") == nullptr) {
    return fail("not the fast H3 decoder (blocks.*, x_embedder, "
                "register_tokens, proj_out, post_quant_conv, norm_out)");
  }
  c.dim = (int)n1->shape[0];
  if (w2->shape.size() != 2 || (int)w2->shape[0] != c.dim) {
    return fail("blocks.0.ff.w2 is not [dim, ffn]");
  }
  c.ffn_inner = (int)w2->shape[1];
  int L = 0;
  while (src.info(blk_(L, "norm1.weight")) != nullptr) { ++L; }
  c.n_layers = L;
  if (c.dim != c.n_heads * c.head_dim) {
    return fail(fmt("dim {} is not the config's {} heads x {}", c.dim,
                    c.n_heads, c.head_dim)());
  }
  if (xe->shape.size() != 2 || (int)xe->shape[0] != c.dim ||
      (int)xe->shape[1] != c.z_channels) {
    return fail("x_embedder is not [dim, z_channels]");
  }
  if (rt->shape.size() != 3 || (int)rt->shape[1] != c.n_register) {
    return fail("register_tokens is not [1, n_register, dim]");
  }
  if (po->shape.size() != 2 || (int)po->shape[0] != c.patch_elems()) {
    return fail("proj_out is not [3 * patch_t * patch^2, dim]");
  }
  if (c.rope_dim() != 48 || c.head_dim != 64 || c.dim % 256 != 0 ||
      c.dim > 4096 || L == 0) {
    return fail("a decoder shape this port does not run (head 64, rotary "
                "48, dim a multiple of 256 up to 4096)");
  }
  if ((int)c.latents_mean.size() != c.z_channels ||
      (int)c.latents_std.size() != c.z_channels) {
    return fail("the released VAE's config states no latent statistics");
  }
  *out = c;
  return true;
}

int
MetalMiniMaxH3FastVae::decoded_frames(const Config& c, int T)
{
  if (T <= 0) { return 0; }
  const int nb = (T + c.block - 1) / c.block;
  return T * c.patch_t - c.frames_to_trim * nb;
}

std::size_t
MetalMiniMaxH3FastVae::scratch_bytes(const Config& c, int T, int h, int w)
{
  if (T <= 0 || h <= 0 || w <= 0) { return 0; }
  const std::size_t D = (std::size_t)c.dim, F = (std::size_t)c.ffn_inner;
  const std::size_t hw = (std::size_t)h * w;
  const std::size_t nb = (std::size_t)(T + c.block - 1) / c.block;
  const std::size_t S = (std::size_t)c.global_tokens();
  const std::size_t N = (std::size_t)T * hw, G = nb * S;
  const std::size_t M = N + G;
  const std::size_t band = (std::size_t)kBand;
  std::size_t b = M * D * 4 + M * D * 2;                 // stream + rows
  b += band * std::max(3 * D, F) * 2 + band * 2 * F * 2 + band * D * 2;
  // One slab's pools: at most a block of frames, plus every window's two
  // blocks of global keys (the shifted grid has the most windows).
  const std::size_t ws = (std::size_t)c.window;
  const std::size_t nwin = ((h + ws / 2 + ws - 1) / ws) *
                           ((w + ws / 2 + ws - 1) / ws);
  const std::size_t qr = (std::size_t)c.block * hw;
  const std::size_t kr = qr + nwin * 2 * S;
  b += (2 * qr + 2 * kr) * D * 2;
  b += 2 * 2 * ((std::size_t)c.block * hw + S) * D * 2;  // the ring
  b += 4 * G * D * 2;                                     // global rows
  // The tables: every token's q and k rows, both shifts, and the global
  // keys' moves.
  b += (4 * N + 4 * nwin * 2 * S * (nb + 1)) * 4;
  // The latent as handed over (f32), and the output, ImageNet-normalized
  // bf16.
  b += (std::size_t)c.z_channels * N * 4;
  b += (std::size_t)c.out_channels * decoded_frames(c, T) * hw *
       (std::size_t)(c.patch * c.patch) * 2;
  // The small ones -- the rotary table, the identity map, the global
  // keys' moves -- and each buffer's page rounding.
  b += (std::size_t)8 << 20;
  // The int8 context's scratch, at the widest operand a band reads.
  if (c.i8_tier >= 0) {
    b += I8GemmContext::forward_bytes_bound(kBand, (int)F, (int)(2 * F),
                                            (int)D, c.i8_tier == 0);
    b += 2 * F * D * 2;            // a held layout widened under its floor
  }
  return b;
}

std::size_t
MetalMiniMaxH3FastVae::weight_bytes(const std::string& file, int i8_tier)
{
  auto wts = MetalLlamaWeights::open_model(file);
  if (!wts.has_value()) { return 0; }
  const bool held = i8_tier >= 1;
  std::size_t b = 0;
  for (const std::string& nm : wts->tensor_names()) {
    const auto* ti = wts->info(nm);
    if (ti == nullptr) { continue; }
    const std::size_t n = numel_(ti->shape);
    // post_quant_conv folds into the embedding's affine map, which is
    // x_embedder's size: nothing of it is held.
    if (nm.rfind("post_quant_conv", 0) == 0) { continue; }
    if (ti->shape.size() == 1 || nm == "register_tokens" ||
        nm.rfind("x_embedder", 0) == 0) {
      // f32: norms, gates, the head's bias; the embedding (a [dim, z]
      // affine map and its offset); the registers. Biases of the block
      // projections are held bf16.
      const bool proj_bias = nm.rfind("blocks.", 0) == 0 &&
                             nm.size() > 5 &&
                             nm.compare(nm.size() - 5, 5, ".bias") == 0;
      b += proj_bias ? n * 2 : n * 4;
      continue;
    }
    if (held && nm.rfind("blocks.", 0) == 0 && ti->shape.size() == 2) {
      const int N = (int)ti->shape[0], K = (int)ti->shape[1];
      const int gc = i8_tier >= 2 ? I8GemmContext::kTokenGroup
                                  : fast_group_for_(K);
      if (gc > 0) {
        b += n + I8GemmContext::fast_scale_bytes(N, K, gc);
        continue;
      }
    }
    b += n * 2;
  }
  return b;
}

std::size_t
MetalMiniMaxH3FastVae::ane_runtime_bytes(const Config& c) noexcept
{
  const std::size_t biases =
      (std::size_t)(2 * c.ffn_inner + c.dim) * 2 * 2;
  return AneFeedForward::runtime_bytes(c.dim, c.ffn_inner, 0,
                                       AneFeedForward::kChunkDefault) +
         biases;
}

// ---- load ------------------------------------------------------------------

MetalMiniMaxH3FastVae::~MetalMiniMaxH3FastVae()
{
  if (_ane) { _ane->join(); }
}

std::unique_ptr<MetalMiniMaxH3FastVae>
MetalMiniMaxH3FastVae::load(std::shared_ptr<WeightSet> ws, MetalCompute* mc,
                            const Config& cfg, std::string* err)
{
  if (mc == nullptr || !ws) {
    if (err != nullptr) { *err = "no weights or no GPU"; }
    return nullptr;
  }
  auto m = std::unique_ptr<MetalMiniMaxH3FastVae>(new MetalMiniMaxH3FastVae());
  m->_ws = std::move(ws);
  m->_mc = mc;
  m->_cfg = cfg;
  if (!m->bind_kernels_(err) || !m->load_(err)) { return nullptr; }
  if (mc->session() != nullptr) {
    mc->session()->log_debug(fmt(
        "MetalMiniMaxH3FastVae: {} layers, dim {}, {} heads x {}, window "
        "{} x {}^2, {} MB held; GEMMs {}{}, attention {}",
        cfg.n_layers, cfg.dim, cfg.n_heads, cfg.head_dim, cfg.block,
        cfg.window, m->_weight_bytes >> 20,
        m->_use_mma2 ? "matmul2d" : "steel",
        m->_i8_tier >= 0
            ? fmt(" + int8 ({})", m->_i8_tier == 0   ? "accurate"
                                  : m->_i8_tier == 1 ? "fast"
                                  : m->_i8_tier == 2 ? "token"
                                                     : "token_plain")()
            : std::string(),
        m->_attn_nax ? "matrix-core flash" : "steel flash"));
  }
  return m;
}

bool
MetalMiniMaxH3FastVae::bind_kernels_(std::string* err)
{
  MetalCompute* mc = _mc;
  _lib_gemm  = mc->load_library("dense_gemm_bf16");
  _lib_elt   = mc->load_library("llm_elementwise_bf16");
  _lib_regen = mc->load_library("minimax_h3_regenerate_bf16");
  _lib_attn  = mc->load_library("attn_steel");
  _fn_gemm     = _lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  _fn_bias_add = _lib_elt.function("bias_add_rows_f16");
  _fn_swiglu   = _lib_elt.function("swiglu_split_gate_first_f16");
  _fn_embed    = _lib_regen.function("h3fv_embed_f32");
  _fn_norm     = _lib_regen.function("h3fv_norm_f32");
  _fn_gated    = _lib_regen.function("h3fv_gated_f32");
  _fn_post     = _lib_regen.function("h3fv_qkv_post_f16");
  _fn_copy     = _lib_regen.function("h3fv_copy_rows_f16");
  _fn_unpatch  = _lib_regen.function("h3fv_unpatch_f16");
  if (!_fn_gemm.valid() || !_fn_bias_add.valid() || !_fn_swiglu.valid() ||
      !_fn_embed.valid() || !_fn_norm.valid() || !_fn_gated.valid() ||
      !_fn_post.valid() || !_fn_copy.valid() || !_fn_unpatch.valid()) {
    if (err != nullptr) { *err = "a fast H3 decoder kernel is missing"; }
    return false;
  }
  // M5: matmul2d GEMMs (VPIPE_H3_FVAE_NO_MMA2 the A/B) and the matrix-core
  // flash attention (VPIPE_H3_FVAE_NO_NAX_ATTN).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_H3_FVAE_NO_MMA2") == nullptr) {
    _lib_mma = mc->load_library("dense_gemm_mma_bf16");
    _fn_mma = _lib_mma.function("dense_gemm_mma_t_n128_f16");
    _fn_mma_wide = _lib_mma.function("dense_gemm_mma_t_n128x256_f16");
    _use_mma2 = _fn_mma.valid() && _fn_mma_wide.valid();
    if (const char* e = std::getenv("VPIPE_H3_MMA_MIN_M")) {
      _mma_min_m = std::max(1, std::atoi(e));
    }
  }
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_H3_FVAE_NO_NAX_ATTN") == nullptr) {
    _lib_attn_nax = mc->load_library("attn_steel_nax");
    metal_compute::FunctionConstants probe;
    probe.set_bool(200, false).set_bool(201, false).set_bool(300, false)
        .set_bool(301, false).set_bool(302, false);
    _attn_nax = _lib_attn_nax.valid() &&
                _lib_attn_nax.function("attn_steel_nax_h_bd64_bf16", probe)
                    .valid();
  }
  for (int aq = 0; aq < 2; ++aq) {
    for (int ak = 0; ak < 2; ++ak) {
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, aq != 0).set_bool(201, ak != 0).set_bool(300, false)
          .set_bool(301, false).set_bool(302, false);
      _fn_attn[aq][ak] =
          _attn_nax ? _lib_attn_nax.function("attn_steel_nax_h_bd64_bf16", fc)
                    : _lib_attn.function("attn_steel_h_bd64_bf16", fc);
      if (!_fn_attn[aq][ak].valid()) {
        if (err != nullptr) { *err = "the bd64 flash attention is missing"; }
        return false;
      }
    }
  }
  return true;
}

MetalMiniMaxH3FastVae::Proj
MetalMiniMaxH3FastVae::proj_(const std::string& name, int N, int K,
                             std::string* err)
{
  WeightSet& w = *_ws;
  MetalCompute* mc = _mc;
  Proj p;
  p.n = N;
  p.k = K;
  const std::size_t nk = (std::size_t)N * K;
  p.b = w.derived(std::string(kKey) + "bf16|" + name + ".bias",
                  [&]() -> SharedBuffer {
                    return bf16_tensor_(w, mc, name + ".bias",
                                        (std::size_t)N);
                  });
  // A tier that holds the int8 layout converts each projection once, the
  // bf16 built only to be converted and then dropped.
  const int gc = (_i8_tier >= 1) ? _i8->fast_group(K) : 0;
  if (gc == 0) {
    p.w = w.derived(std::string(kKey) + "bf16|" + name + ".weight",
                    [&]() -> SharedBuffer {
                      return bf16_tensor_(w, mc, name + ".weight", nk);
                    });
    _weight_bytes += p.w.byte_size();
  } else {
    const std::size_t sb = I8GemmContext::fast_scale_bytes(N, K, gc);
    SharedBuffer both = w.derived(
        std::string(kKey) + _i8->layout_tag(gc) + "|" + name + ".weight",
        [&]() -> SharedBuffer {
          SharedBuffer f = bf16_tensor_(w, mc, name + ".weight", nk);
          SharedBuffer out = mc->make_shared_buffer(nk + sb);
          if (f.empty() || out.empty()) { return {}; }
          CommandStream st = mc->make_command_stream();
          bool ok = false;
          {
            ComputeEncoder e = st.begin_compute();
            ok = _i8->encode_quant_fast(e, f, out.subview(0, nk),
                                        out.subview(nk, sb), gc, N, K);
          }
          return ok && st.commit().wait_ok(nullptr) ? std::move(out)
                                                    : SharedBuffer{};
        });
    if (!both.empty()) {
      p.gc = gc;
      p.codes = both.subview(0, nk);
      p.scales = both.subview(nk, sb);
      _weight_bytes += both.byte_size();
    }
  }
  if ((p.gc == 0 && p.w.empty()) || (p.gc > 0 && p.codes.empty()) ||
      p.b.empty()) {
    if (err != nullptr) { *err = "cannot read " + name; }
    return {};
  }
  _weight_bytes += p.b.byte_size();
  return p;
}

bool
MetalMiniMaxH3FastVae::load_(std::string* err)
{
  WeightSet& w = *_ws;
  MetalCompute* mc = _mc;
  const Config& c = _cfg;
  const int D = c.dim, F = c.ffn_inner, Z = c.z_channels;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };

  // The int8 context first: a tier that holds the layout converts each
  // projection as it loads. It declines without matrix cores, and
  // VPIPE_I8_GEMM / VPIPE_I8_TIER override the config.
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, c.i8_tier >= 0,
                                              /*bf16=*/true);
    i8->set_tier(c.i8_tier >= 2   ? I8GemmContext::Tier::kToken
                 : c.i8_tier == 1 ? I8GemmContext::Tier::kFast
                                  : I8GemmContext::Tier::kAccurate);
    i8->set_token_rotation(c.i8_tier == 2);
    if (i8->enabled()) {
      _i8 = std::move(i8);
      _i8_tier = _i8->tier() == I8GemmContext::Tier::kAccurate ? 0
                 : _i8->tier() == I8GemmContext::Tier::kFast   ? 1
                 : _i8->token_rotated()                        ? 2
                                                               : 3;
      // A held layout takes the int8 route at any row count -- a slab's
      // last band, the global rows -- rather than widening the layout.
      if (_i8_tier >= 1) { _i8->set_min_rows(1); }
    }
  }

  // The embedding: (z * std + mean) through post_quant_conv (a 1x1x1
  // conv) and x_embedder is one affine map, folded once in double.
  {
    std::vector<double> wc((std::size_t)Z * Z), bc(Z), we((std::size_t)D * Z),
        be(D);
    auto rd = [&](const std::string& nm, std::size_t n,
                  std::vector<double>& out) {
      return with_tensor_(w, mc, nm, n,
                          [&](const void* p, const std::string& dt) {
                            for (std::size_t i = 0; i < n; ++i) {
                              out[i] = elem_f32_(p, dt, i);
                            }
                            return true;
                          });
    };
    if (!rd("post_quant_conv.weight", (std::size_t)Z * Z, wc) ||
        !rd("post_quant_conv.bias", Z, bc) ||
        !rd("x_embedder.weight", (std::size_t)D * Z, we) ||
        !rd("x_embedder.bias", D, be)) {
      return fail("cannot read the embedding (post_quant_conv, "
                  "x_embedder)");
    }
    _embed_a = mc->make_shared_buffer((std::size_t)D * Z * 4);
    _embed_c = mc->make_shared_buffer((std::size_t)D * 4);
    if (_embed_a.empty() || _embed_c.empty()) {
      return fail("allocation failed");
    }
    auto* A = static_cast<float*>(_embed_a.contents());
    auto* cc = static_cast<float*>(_embed_c.contents());
    for (int d = 0; d < D; ++d) {
      double acc = be[d];
      for (int j = 0; j < Z; ++j) {
        double wcm = bc[j];
        for (int k = 0; k < Z; ++k) {
          wcm += wc[(std::size_t)j * Z + k] * c.latents_mean[(std::size_t)k];
        }
        acc += we[(std::size_t)d * Z + j] * wcm;
      }
      cc[d] = (float)acc;
      for (int k = 0; k < Z; ++k) {
        double a = 0.0;
        for (int j = 0; j < Z; ++j) {
          a += we[(std::size_t)d * Z + j] * wc[(std::size_t)j * Z + k];
        }
        A[(std::size_t)d * Z + k] =
            (float)(a * c.latents_std[(std::size_t)k]);
      }
    }
    _weight_bytes += _embed_a.byte_size() + _embed_c.byte_size();
  }
  _registers = f32_tensor_(w, mc, "register_tokens",
                           (std::size_t)c.n_register * D);
  _ln_w = f32_tensor_(w, mc, "norm_out.weight", (std::size_t)D);
  _ln_b = f32_tensor_(w, mc, "norm_out.bias", (std::size_t)D);
  _head_bias = f32_tensor_(w, mc, "proj_out.bias",
                           (std::size_t)c.patch_elems());
  if (_registers.empty() || _ln_w.empty() || _ln_b.empty() ||
      _head_bias.empty()) {
    return fail("cannot read register_tokens / norm_out / proj_out.bias");
  }
  _weight_bytes += _registers.byte_size() + _ln_w.byte_size() +
                   _ln_b.byte_size() + _head_bias.byte_size();
  // The head stays bf16 at every tier: it writes the pixels.
  _head.n = c.patch_elems();
  _head.k = D;
  _head.w = w.derived(std::string(kKey) + "bf16|proj_out.weight",
                      [&]() -> SharedBuffer {
                        return bf16_tensor_(
                            w, mc, "proj_out.weight",
                            (std::size_t)c.patch_elems() * D);
                      });
  if (_head.w.empty()) { return fail("cannot read proj_out.weight"); }
  _weight_bytes += _head.w.byte_size();

  _blocks.resize((std::size_t)c.n_layers);
  for (int L = 0; L < c.n_layers; ++L) {
    Block& b = _blocks[(std::size_t)L];
    b.n1 = f32_tensor_(w, mc, blk_(L, "norm1.weight"), (std::size_t)D);
    b.n2 = f32_tensor_(w, mc, blk_(L, "norm2.weight"), (std::size_t)D);
    b.s1 = f32_tensor_(w, mc, blk_(L, "scale1"), (std::size_t)D);
    b.s2 = f32_tensor_(w, mc, blk_(L, "scale2"), (std::size_t)D);
    if (b.n1.empty() || b.n2.empty() || b.s1.empty() || b.s2.empty()) {
      return fail(fmt("cannot read block {}'s norms and gates", L)());
    }
    _weight_bytes += 4 * (std::size_t)D * 4;
    b.qkv = proj_(blk_(L, "attn.to_qkv"), 3 * D, D, err);
    if (b.qkv.n == 0) { return false; }
    b.out = proj_(blk_(L, "attn.to_out"), D, D, err);
    if (b.out.n == 0) { return false; }
    b.w1 = proj_(blk_(L, "ff.w1"), 2 * F, D, err);
    if (b.w1.n == 0) { return false; }
    b.w2 = proj_(blk_(L, "ff.w2"), D, F, err);
    if (b.w2.n == 0) { return false; }
  }

  // The rotary table, [cos | sin][axis][window-local index][frequency]:
  // ids are cell centres in [-1, 1) of the window (time over `time_norm`
  // frames), the angle 2 pi id / theta^(6 f / rope_dim).
  _rope = mc->make_shared_buffer(2 * 3 * 16 * 8 * 4);
  if (_rope.empty() || c.window > 16 || c.block > 16) {
    return fail("the rotary table (window and block at most 16)");
  }
  {
    auto* t = static_cast<float*>(_rope.contents());
    std::memset(t, 0, _rope.byte_size());
    const int nf = c.rope_dim() / 6;
    for (int a = 0; a < 3; ++a) {
      const int n = a == 0 ? c.block : c.window;
      const double norm = a == 0 ? c.time_norm : (double)c.window;
      for (int l = 0; l < n; ++l) {
        const double id = ((double)l + 0.5) / norm * 2.0 - 1.0;
        for (int f = 0; f < nf && f < 8; ++f) {
          const double inv =
              1.0 / std::pow(c.rope_theta, (double)f * 6.0 / c.rope_dim());
          const double ang = 2.0 * M_PI * id * inv;
          t[(a * 16 + l) * 8 + f] = (float)std::cos(ang);
          t[384 + (a * 16 + l) * 8 + f] = (float)std::sin(ang);
        }
      }
    }
  }
  return true;
}

// ---- the geometry -----------------------------------------------------------

bool
MetalMiniMaxH3FastVae::build_geometry_(int T, int h, int w, std::string* err)
{
  if (_geo.T == T && _geo.H == h && _geo.W == w) { return true; }
  const Config& c = _cfg;
  const int tb = c.block, ws = c.window, S = c.global_tokens();
  const int hw = h * w;
  Geometry g;
  g.T = T;
  g.H = h;
  g.W = w;
  g.nb = (T + tb - 1) / tb;
  std::vector<int> all;
  for (int sh = 0; sh < 2; ++sh) {
    const int s = sh ? ws / 2 : 0, st = sh ? c.t_shift : 0;
    const int nwt = (T + st + tb - 1) / tb;
    const int nwy = (h + s + ws - 1) / ws, nwx = (w + s + ws - 1) / ws;
    for (int j = 0; j < nwt; ++j) {
      Slab sl;
      sl.t_lo = std::max(j * tb - st, 0);
      sl.t_hi = std::min((j + 1) * tb - st, T);
      if (sl.t_hi <= sl.t_lo) { continue; }
      const int nt = sl.t_hi - sl.t_lo;
      int blocks[2] = {sl.t_lo / tb, (sl.t_hi - 1) / tb};
      const int ng = blocks[1] != blocks[0] ? 2 : 1;
      for (int b = blocks[0]; b <= blocks[1]; ++b) {
        const int last = std::min((b + 1) * tb, T) - 1;
        if (last >= sl.t_lo && last < sl.t_hi) { sl.done.push_back(b); }
      }
      // The windows, batched by shape.
      struct Win { int ylo, ny, xlo, nx; };
      std::map<std::pair<int, int>, std::vector<Win>> by_shape;
      for (int i = 0; i < nwy; ++i) {
        const int ylo = std::max(i * ws - s, 0);
        const int yhi = std::min((i + 1) * ws - s, h);
        if (yhi <= ylo) { continue; }
        for (int k = 0; k < nwx; ++k) {
          const int xlo = std::max(k * ws - s, 0);
          const int xhi = std::min((k + 1) * ws - s, w);
          if (xhi <= xlo) { continue; }
          by_shape[{yhi - ylo, xhi - xlo}].push_back(
              {ylo, yhi - ylo, xlo, xhi - xlo});
        }
      }
      const int ntok = nt * hw;
      sl.qrow.assign((std::size_t)ntok, -1);
      sl.krow.assign((std::size_t)ntok, -1);
      int q = 0, kr = 0;
      for (const auto& [shape, wins] : by_shape) {
        const int qL = nt * shape.first * shape.second;
        const int kL = qL + ng * S;
        sl.classes.push_back({q, kr, (int)wins.size(), qL, kL});
        for (const Win& wn : wins) {
          for (int tt = 0; tt < nt; ++tt) {
            for (int y = 0; y < wn.ny; ++y) {
              for (int x = 0; x < wn.nx; ++x) {
                const int tok = tt * hw + (wn.ylo + y) * w + (wn.xlo + x);
                const int pos = (tt * wn.ny + y) * wn.nx + x;
                sl.qrow[(std::size_t)tok] = q + pos;
                sl.krow[(std::size_t)tok] = kr + pos;
              }
            }
          }
          for (int bi = 0; bi < ng; ++bi) {
            for (int si = 0; si < S; ++si) {
              sl.gdst.push_back(kr + qL + bi * S + si);
              sl.gsrc.push_back(blocks[bi] * S + si);
            }
          }
          q += qL;
          kr += kL;
        }
      }
      for (int v : sl.qrow) {
        if (v < 0) {
          if (err != nullptr) { *err = "a token no window holds"; }
          return false;
        }
      }
      sl.q_rows = q;
      sl.k_rows = kr;
      g.slabs[sh].push_back(std::move(sl));
    }
  }
  // Every slab's four tables in one buffer.
  std::size_t total = 0;
  for (int sh = 0; sh < 2; ++sh) {
    for (const Slab& sl : g.slabs[sh]) {
      total += sl.qrow.size() + sl.krow.size() + sl.gdst.size() +
               sl.gsrc.size();
    }
  }
  g.tables = _mc->make_shared_buffer(std::max<std::size_t>(total, 1) * 4);
  if (g.tables.empty()) {
    if (err != nullptr) { *err = "allocation failed (window tables)"; }
    return false;
  }
  auto* d = static_cast<int*>(g.tables.contents());
  std::size_t at = 0;
  for (int sh = 0; sh < 2; ++sh) {
    for (Slab& sl : g.slabs[sh]) {
      g.off[sh].push_back(at);
      for (const auto* v : {&sl.qrow, &sl.krow, &sl.gdst, &sl.gsrc}) {
        std::memcpy(d + at, v->data(), v->size() * 4);
        at += v->size();
      }
    }
  }
  _geo = std::move(g);
  return true;
}

bool
MetalMiniMaxH3FastVae::ensure_scratch_(int T, int h, int w)
{
  if (_s.T == T && _s.h == h && _s.w == w) { return true; }
  const Config& c = _cfg;
  const std::size_t D = (std::size_t)c.dim, F = (std::size_t)c.ffn_inner;
  const int S = c.global_tokens();
  const std::size_t G = (std::size_t)_geo.nb * S;
  const std::size_t N = (std::size_t)T * h * w;
  const std::size_t M = N + G;
  std::size_t qmax = 0, kmax = 0;
  for (int sh = 0; sh < 2; ++sh) {
    for (const Slab& sl : _geo.slabs[sh]) {
      qmax = std::max(qmax, (std::size_t)sl.q_rows);
      kmax = std::max(kmax, (std::size_t)sl.k_rows);
    }
  }
  Scratch s;
  s.T = T;
  s.h = h;
  s.w = w;
  s.band = kBand;
  s.ring_rows = c.block * h * w + S;
  auto mk = [&](std::size_t bytes) { return _mc->make_shared_buffer(bytes); };
  s.x   = mk(M * D * 4);
  s.hb  = mk(M * D * 2);
  s.qkv = mk((std::size_t)kBand * std::max(3 * D, F) * 2);
  s.ff  = mk((std::size_t)kBand * 2 * F * 2);
  s.y   = mk((std::size_t)kBand * D * 2);
  s.qp  = mk(qmax * D * 2);
  s.op  = mk(qmax * D * 2);
  s.kp  = mk(kmax * D * 2);
  s.vp  = mk(kmax * D * 2);
  s.ringk = mk(2 * (std::size_t)s.ring_rows * D * 2);
  s.ringv = mk(2 * (std::size_t)s.ring_rows * D * 2);
  s.gq = mk(G * D * 2);
  s.gk = mk(G * D * 2);
  s.gv = mk(G * D * 2);
  s.go = mk(G * D * 2);
  const std::size_t nid = std::max<std::size_t>((std::size_t)kBand, G);
  s.ident = mk(nid * 4);
  if (s.x.empty() || s.hb.empty() || s.qkv.empty() || s.ff.empty() ||
      s.y.empty() || s.qp.empty() || s.op.empty() || s.kp.empty() ||
      s.vp.empty() || s.ringk.empty() || s.ringv.empty() || s.gq.empty() ||
      s.gk.empty() || s.gv.empty() || s.go.empty() || s.ident.empty()) {
    return false;
  }
  auto* id = static_cast<int*>(s.ident.contents());
  for (std::size_t i = 0; i < nid; ++i) { id[i] = (int)i; }
  _s = std::move(s);
  return true;
}

// ---- the pieces -------------------------------------------------------------

void
MetalMiniMaxH3FastVae::dense_(ComputeEncoder& enc, const SharedBuffer& x,
                              std::size_t xe, const SharedBuffer& w,
                              const SharedBuffer* bias, const SharedBuffer& y,
                              std::size_t ye, int M, int N, int K)
{
  ++_dense_gemms;
  if (_use_mma2 && M >= _mma_min_m && N >= 16 && M <= mma_row_band(N, K)) {
    const bool wide = mma_use_wide_tile(N, K);
    const int BN = wide ? 256 : 128;
    enc.set_function(wide ? _fn_mma_wide : _fn_mma);
    enc.set_buffer(0, x, xe * 2);
    enc.set_buffer(1, w);
    enc.set_buffer(2, w);          // the bias slot, unread
    enc.set_buffer(3, y, ye * 2);
    enc.set_constant(4, K);
    enc.set_constant(5, N);
    enc.set_constant(6, M);
    enc.set_constant(7, 0);
    enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    if (bias != nullptr) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, ye * 2);
      enc.set_buffer(1, *bias);
      enc.set_constant(2, N);
      enc.set_constant(3, (unsigned)((std::size_t)M * N));
      enc.dispatch({(unsigned)N, (unsigned)M, 1}, {256, 1, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x, xe * 2);
  enc.set_buffer(1, w);
  enc.set_buffer(2, bias != nullptr ? *bias : w);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias != nullptr ? 1 : 0);
  enc.dispatch({(unsigned)(((N + 63) / 64) * 32),
                (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
}

void
MetalMiniMaxH3FastVae::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                             std::size_t xe, const Proj& p,
                             const SharedBuffer& y, std::size_t ye, int M)
{
  const int N = p.n, K = p.k;
  // Counted once, where it is asked for, whatever route runs it.
  _flops.gemm(M, N, K);
  if (p.gc > 0) {
    // A held layout: its tier's GEMM, else the layout widened to bf16.
    if (_i8->gemm_fast(enc, x, xe, p.codes, p.scales, p.gc, p.b, y, ye, M,
                       N, K)) {
      ++_i8_gemms;
      return;
    }
    const std::size_t need = (std::size_t)N * K * 2;
    if (_wdeq.byte_size() < need) {
      _wdeq = _mc->make_shared_buffer(need);
      if (_wdeq.empty()) { return; }
    }
    if (!_i8->encode_dequant_fast(enc, p.codes, p.scales, p.gc, _wdeq, N,
                                  K)) {
      return;
    }
    dense_(enc, x, xe, _wdeq, &p.b, y, ye, M, N, K);
    return;
  }
  // The accurate tier: the bf16 weight quantized per call; a shape it
  // declines falls through with nothing encoded.
  if (_i8 && _i8->gemm(enc, x, xe, p.w, p.b, y, ye, M, N, K)) {
    ++_i8_gemms;
    return;
  }
  dense_(enc, x, xe, p.w, p.b.empty() ? nullptr : &p.b, y, ye, M, N, K);
}

void
MetalMiniMaxH3FastVae::norm_(ComputeEncoder& enc, const SharedBuffer& x,
                             std::size_t row0, const SharedBuffer& w,
                             const SharedBuffer* b, const SharedBuffer& out,
                             std::size_t orow0, int rows, bool ln)
{
  if (rows <= 0) { return; }
  const std::size_t D = (std::size_t)_cfg.dim;
  enc.set_function(_fn_norm);
  enc.set_buffer(0, x, row0 * D * 4);
  enc.set_buffer(1, w);
  enc.set_buffer(2, b != nullptr ? *b : w);
  enc.set_buffer(3, out, orow0 * D * 2);
  enc.set_constant(4, _cfg.dim);
  enc.set_constant(5, _cfg.norm_eps);
  enc.set_constant(6, ln ? 1 : 0);
  enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
}

void
MetalMiniMaxH3FastVae::gated_(ComputeEncoder& enc, std::size_t row0,
                              const SharedBuffer& s, const SharedBuffer& y,
                              std::size_t yrow0, int rows)
{
  if (rows <= 0) { return; }
  const std::size_t D = (std::size_t)_cfg.dim;
  const unsigned n = (unsigned)((std::size_t)rows * D);
  enc.set_function(_fn_gated);
  enc.set_buffer(0, _s.x, row0 * D * 4);
  enc.set_buffer(1, s);
  enc.set_buffer(2, y, yrow0 * D * 2);
  enc.set_constant(3, _cfg.dim);
  enc.set_constant(4, n);
  enc.dispatch({n, 1, 1}, {256, 1, 1});
}

void
MetalMiniMaxH3FastVae::copy_rows_(ComputeEncoder& enc, const SharedBuffer& src,
                                  const SharedBuffer& dst,
                                  const SharedBuffer* src2,
                                  const SharedBuffer* dst2,
                                  const SharedBuffer& smap, std::size_t s_off,
                                  const SharedBuffer* dmap, std::size_t d_off,
                                  int d0, int n)
{
  if (n <= 0) { return; }
  enc.set_function(_fn_copy);
  enc.set_buffer(0, src);
  enc.set_buffer(1, dst);
  enc.set_buffer(2, src2 != nullptr ? *src2 : src);
  enc.set_buffer(3, dst2 != nullptr ? *dst2 : dst);
  enc.set_buffer(4, smap, s_off * 4);
  enc.set_buffer(5, dmap != nullptr ? *dmap : smap,
                 dmap != nullptr ? d_off * 4 : 0);
  enc.set_constant(6, _cfg.dim);
  enc.set_constant(7, n);
  enc.set_constant(8, d0);
  enc.set_constant(9, dmap != nullptr ? 1 : 0);
  enc.set_constant(10, src2 != nullptr ? 1 : 0);
  enc.dispatch({(unsigned)(_cfg.dim / 4), (unsigned)n, 1}, {64, 4, 1});
}

void
MetalMiniMaxH3FastVae::attend_(ComputeEncoder& enc, const SharedBuffer& q,
                               std::size_t q0, const SharedBuffer& k,
                               const SharedBuffer& v, std::size_t k0,
                               const SharedBuffer& o, int count, int qL,
                               int kL)
{
  if (count <= 0 || qL <= 0 || kL <= 0) { return; }
  const int NH = _cfg.n_heads, HD = _cfg.head_dim, D = _cfg.dim;
  const int BQ = _attn_nax ? kBqNax : kBqAlu;
  const int BK = _attn_nax ? kBkNax : kBkAlu;
  AttnP p{};
  p.B = count;
  p.H = NH;
  p.D = HD;
  p.qL = qL;
  p.kL = kL;
  p.gqa_factor = 1;
  p.scale = 1.0f / std::sqrt((float)HD);
  p.NQ = (qL + BQ - 1) / BQ;
  p.NK = (kL + BK - 1) / BK;
  p.NQ_aligned = qL / BQ;
  p.NK_aligned = kL / BK;
  p.qL_rem = qL - p.NQ_aligned * BQ;
  p.kL_rem = kL - p.NK_aligned * BK;
  p.qL_off = 0;
  // Token-major pools: a row is every head's 64 channels.
  p.Q_strides[0] = (std::int64_t)qL * D;
  p.Q_strides[1] = HD;
  p.Q_strides[2] = D;
  p.K_strides[0] = (std::int64_t)kL * D;
  p.K_strides[1] = HD;
  p.K_strides[2] = D;
  for (int i = 0; i < 3; ++i) {
    p.V_strides[i] = p.K_strides[i];
    p.O_strides[i] = p.Q_strides[i];
  }
  _flops.attention(NH, (long long)count * qL * kL, HD);
  enc.set_function(_fn_attn[(qL % BQ) == 0][(kL % BK) == 0]);
  enc.set_buffer(0, q, q0 * (std::size_t)D * 2);
  enc.set_buffer(1, k, k0 * (std::size_t)D * 2);
  enc.set_buffer(2, v, k0 * (std::size_t)D * 2);
  enc.set_buffer(3, o, q0 * (std::size_t)D * 2);
  enc.set_constant(4, p);
  enc.dispatch({32u * (unsigned)p.NQ, 4u * (unsigned)NH, (unsigned)count},
               {32, 4, 1});
}

// ---- the ANE tier -----------------------------------------------------------

bool
MetalMiniMaxH3FastVae::ane_setup_(int rows)
{
  if (_ane_tried) { return _ane != nullptr; }
  _ane_tried = true;
  if (_mc->session() == nullptr || rows <= 0) { return false; }
  AneFeedForward::Options o;
  o.session = _mc->session();
  o.mc      = _mc;
  o.tag     = "minimax-h3-fast-vae";
  o.hidden  = _cfg.dim;
  o.ffn     = _cfg.ffn_inner;
  o.seq     = rows;
  o.rows    = _cfg.ane_rows;
  o.biases  = true;
  o.profile = std::getenv("VPIPE_H3_FVAE_ANE_PROFILE") != nullptr;
  _ane = AneFeedForward::create(o);
  return _ane != nullptr;
}

bool
MetalMiniMaxH3FastVae::ane_eligible_(int L) const
{
  if (_ane == nullptr) { return false; }
  const int cap = _cfg.ane_layers > 0
                      ? std::min(_cfg.ane_layers, _cfg.n_layers)
                      : _cfg.n_layers;
  const Block& b = _blocks[(std::size_t)L];
  // A projection held in the int8 layout is not a source the staging
  // reads; the GPU keeps that block's feed-forward.
  return L < cap && b.w1.gc == 0 && b.w2.gc == 0;
}

void
MetalMiniMaxH3FastVae::ane_stage_(int L)
{
  const Block& b = _blocks[(std::size_t)L];
  auto view = [](const Proj& p, std::size_t offset) {
    AneFfnSource s;
    s.w      = &p.w;
    s.offset = offset;
    s.bias   = &p.b;
    return s;
  };
  // w1 is [gate | up] rows, gate first.
  _ane->stage(L, view(b.w1, 0), view(b.w1, (std::size_t)_cfg.ffn_inner),
              view(b.w2, 0), 64);
}

// ---- decode -----------------------------------------------------------------

SharedBuffer
MetalMiniMaxH3FastVae::decode(const float* z, int T, int h, int w,
                              std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  if (z == nullptr || T <= 0 || h <= 0 || w <= 0) {
    return fail("empty latent");
  }
  const int D = c.dim, F = c.ffn_inner, Z = c.z_channels;
  const int S = c.global_tokens(), tb = c.block;
  const int hw = h * w;
  const int N = T * hw;
  if (!build_geometry_(T, h, w, err)) { return {}; }
  const int nb = _geo.nb;
  const int G = nb * S;
  const int M = N + G;
  if (!ensure_scratch_(T, h, w)) {
    return fail("activation allocation failed (out of GPU memory)");
  }
  const int Fo = decoded_frames(T);
  const std::size_t H16 = (std::size_t)h * c.patch;
  const std::size_t W16 = (std::size_t)w * c.patch;
  SharedBuffer out = _mc->make_shared_buffer(
      (std::size_t)c.out_channels * Fo * H16 * W16 * 2);
  SharedBuffer zb = _mc->make_shared_buffer((std::size_t)Z * N * 4);
  if (out.empty() || zb.empty()) { return fail("allocation failed"); }
  std::memcpy(zb.contents(), z, (std::size_t)Z * N * 4);
  Scratch& s = _s;
  _i8_gemms = _dense_gemms = 0;
  _ane_rows = 0;
  if (_i8) { _i8->tune_pending(_mc); }
  const bool ane_on = c.ane_ffn && ane_setup_(M);
  using Clk = std::chrono::steady_clock;

  _flops.begin();
  CommandStream stream = _mc->make_command_stream();
  auto commit_wait = [&](ComputeEncoder& enc) -> bool {
    enc.end();
    _flops.flush(stream);
    std::string e;
    if (!stream.commit().wait_ok(&e)) {
      if (err != nullptr) {
        *err = e.empty() ? std::string("fast H3 decode failed") : e;
      }
      return false;
    }
    return true;
  };

  // The embedding, and the global rows: four registers and a zero token
  // for every native block.
  {
    ComputeEncoder enc = stream.begin_compute();
    enc.set_function(_fn_embed);
    enc.set_buffer(0, zb);
    enc.set_buffer(1, _embed_a);
    enc.set_buffer(2, _embed_c);
    enc.set_buffer(3, s.x);
    enc.set_constant(4, N);
    enc.set_constant(5, D);
    enc.set_constant(6, Z);
    enc.dispatch({(unsigned)D, (unsigned)N, 1}, {64, 4, 1});
    // Host-written: the embedding writes rows [0, N), these [N, M).
    auto* xg = static_cast<float*>(s.x.contents()) + (std::size_t)N * D;
    const auto* reg = static_cast<const float*>(_registers.contents());
    for (int b = 0; b < nb; ++b) {
      float* gb = xg + (std::size_t)b * S * D;
      std::memcpy(gb, reg, (std::size_t)c.n_register * D * 4);
      std::memset(gb + (std::size_t)c.n_register * D, 0,
                  (std::size_t)D * 4);
    }
    _flops.add(2ull * (std::uint64_t)N * D * Z);
    if (!commit_wait(enc)) { return {}; }
  }

  const SharedBuffer& tab = _geo.tables;
  for (int L = 0; L < c.n_layers; ++L) {
    if (_stop && _stop()) { return {}; }
    const Block& b = _blocks[(std::size_t)L];
    const bool last = L == c.n_layers - 1;
    const int sh = L & 1;
    const int sft = sh ? c.window / 2 : 0, st = sh ? c.t_shift : 0;
    const AneFeedForward::Plan plan =
        ane_on && ane_eligible_(L) ? _ane->plan_block()
                                   : AneFeedForward::Plan::kGpu;
    const bool ane_block = plan == AneFeedForward::Plan::kSplit;
    const bool ane_probe = plan == AneFeedForward::Plan::kProbe;
    // Stage the feed-forward for the ANE now, so the conversion runs under
    // this layer's attention.
    if (ane_block) { ane_stage_(L); }

    ComputeEncoder enc = stream.begin_compute();
    PostP pp{};
    pp.T = T;
    pp.H = h;
    pp.W = w;
    pp.s = sft;
    pp.st = st;
    pp.tb = tb;
    pp.ws = c.window;
    pp.ring_rows = s.ring_rows;
    pp.heads = c.n_heads;
    auto post = [&](const SharedBuffer& qd, const SharedBuffer& kd,
                    const SharedBuffer& vd, const SharedBuffer& qmap,
                    std::size_t qmo, const SharedBuffer& kmap,
                    std::size_t kmo, int rows) {
      enc.set_function(_fn_post);
      enc.set_buffer(0, s.qkv);
      enc.set_buffer(1, qd);
      enc.set_buffer(2, kd);
      enc.set_buffer(3, vd);
      enc.set_buffer(4, qmap, qmo * 4);
      enc.set_buffer(5, kmap, kmo * 4);
      enc.set_buffer(6, _rope);
      enc.set_buffer(7, s.ringk);
      enc.set_buffer(8, s.ringv);
      enc.set_constant(9, pp);
      enc.set_constant(10, c.norm_eps);
      enc.dispatch({32, (unsigned)c.n_heads, (unsigned)rows}, {32, 4, 1});
    };

    // The global rows' q, k and v: normed, not rotated. Every layer's
    // windows read the keys; only the last layer's global tokens are never
    // read themselves.
    norm_(enc, s.x, (std::size_t)N, b.n1, nullptr, s.hb, (std::size_t)N, G,
          false);
    gemm_(enc, s.hb, (std::size_t)N * D, b.qkv, s.qkv, 0, G);
    pp.tok0 = -1;
    pp.rope = 0;
    pp.ring = 0;
    post(s.gq, s.gk, s.gv, s.ident, 0, s.ident, 0, G);

    // The windows, a temporal window (slab) at a time.
    const std::vector<Slab>& slabs = _geo.slabs[sh];
    for (std::size_t j = 0; j < slabs.size(); ++j) {
      const Slab& sl = slabs[j];
      const std::size_t o = _geo.off[sh][j];
      const std::size_t o_q = o, o_k = o + sl.qrow.size();
      const std::size_t o_gd = o_k + sl.krow.size();
      const std::size_t o_gs = o_gd + sl.gdst.size();
      const int r0 = sl.t_lo * hw;
      const int nr = (sl.t_hi - sl.t_lo) * hw;
      pp.rope = 1;
      pp.ring = last ? 0 : 1;
      for (int a = 0; a < nr; a += kBand) {
        const int n = std::min(kBand, nr - a);
        norm_(enc, s.x, (std::size_t)(r0 + a), b.n1, nullptr, s.hb,
              (std::size_t)(r0 + a), n, false);
        gemm_(enc, s.hb, (std::size_t)(r0 + a) * D, b.qkv, s.qkv, 0, n);
        pp.tok0 = r0 + a;
        post(s.qp, s.kp, s.vp, tab, o_q + a, tab, o_k + a, n);
      }
      // Each window's blocks' global keys, after its own.
      copy_rows_(enc, s.gk, s.kp, &s.gv, &s.vp, tab, o_gs, &tab, o_gd, 0,
                 (int)sl.gdst.size());
      for (const Slab::Class& cl : sl.classes) {
        attend_(enc, s.qp, (std::size_t)cl.q0, s.kp, s.vp,
                (std::size_t)cl.k0, s.op, cl.count, cl.qL, cl.kL);
      }
      // Back to token order, over the rows the norm wrote, then the
      // output projection and the first residual.
      copy_rows_(enc, s.op, s.hb, nullptr, nullptr, tab, o_q, nullptr, 0,
                 r0, nr);
      for (int a = 0; a < nr; a += kBand) {
        const int n = std::min(kBand, nr - a);
        gemm_(enc, s.hb, (std::size_t)(r0 + a) * D, b.out, s.y, 0, n);
        gated_(enc, (std::size_t)(r0 + a), b.s1, s.y, 0, n);
      }
      // The global tokens of every block whose last frame was in this
      // slab: their keys after the block's own, in the block's ring slot.
      if (!last) {
        for (int bi : sl.done) {
          const int ntok = (std::min((bi + 1) * tb, T) - bi * tb) * hw;
          const std::size_t ring0 = (std::size_t)(bi & 1) * s.ring_rows;
          copy_rows_(enc, s.gk, s.ringk, &s.gv, &s.ringv, s.ident,
                     (std::size_t)bi * S, nullptr, 0,
                     (int)ring0 + ntok, S);
          attend_(enc, s.gq, (std::size_t)bi * S, s.ringk, s.ringv, ring0,
                  s.go, 1, S, ntok + S);
        }
      }
    }
    if (!last) {
      gemm_(enc, s.go, 0, b.out, s.y, 0, G);
      gated_(enc, (std::size_t)N, b.s1, s.y, 0, G);
    }

    // ---- the feed-forward, every row (the last layer: the tokens) ----
    const int rows = last ? N : M;
    for (int a = 0; a < rows; a += kBand) {
      const int n = std::min(kBand, rows - a);
      norm_(enc, s.x, (std::size_t)a, b.n2, nullptr, s.hb, (std::size_t)a,
            n, false);
    }
    int a_rows = 0;
    double drain_ms = 0.0;
    if (ane_block || ane_probe) {
      const auto t0 = Clk::now();
      enc.end();
      _flops.flush(stream);
      CommandStream::Fence f = stream.commit();
      const bool staged = ane_block && _ane->join_stage(L);
      if (!f.wait_ok(nullptr)) { return fail("fast H3 decode failed"); }
      drain_ms = std::chrono::duration<double, std::milli>(Clk::now() - t0)
                     .count();
      if (staged) {
        a_rows = _ane->begin(s.hb, s.hb, rows);
      } else if (ane_block && _mc->session() != nullptr) {
        _mc->session()->warn(fmt(
            "minimax-h3-fast-vae: staging layer {}'s feed-forward for the "
            "ANE failed; it keeps the GPU", L));
      }
      enc = stream.begin_compute();
    }
    // The GPU's rows [r0, r0 + n) of hb, in place, in bands.
    auto gpu_ff = [&](int r0, int n) {
      for (int a = 0; a < n; a += kBand) {
        const int m = std::min(kBand, n - a);
        const std::size_t e0 = (std::size_t)(r0 + a) * D;
        gemm_(enc, s.hb, e0, b.w1, s.ff, 0, m);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, s.ff);
        enc.set_buffer(1, s.qkv);
        enc.set_constant(2, m);
        enc.set_constant(3, F);
        enc.dispatch({(unsigned)((std::size_t)m * F), 1, 1}, {256, 1, 1});
        gemm_(enc, s.qkv, 0, b.w2, s.hb, e0, m);
      }
    };
    const auto t_g0 = Clk::now();
    gpu_ff(0, rows - a_rows);
    if (a_rows > 0) {
      if (!commit_wait(enc)) { return {}; }
      const double gpu_ms =
          _ane->timing() ? std::chrono::duration<double, std::milli>(
                               Clk::now() - t_g0).count()
                         : 0.0;
      const bool ok = _ane->finish(L, rows, gpu_ms, drain_ms);
      _ane_rows += a_rows;
      enc = stream.begin_compute();
      const int lost = ok ? -1 : _ane->lost_row0();
      if (lost >= 0 && lost < rows) { gpu_ff(lost, rows - lost); }
    } else if (ane_probe) {
      if (!commit_wait(enc)) { return {}; }
      _ane->note_probe(drain_ms, std::chrono::duration<double, std::milli>(
                                     Clk::now() - t_g0).count());
      enc = stream.begin_compute();
    }
    for (int a = 0; a < rows; a += kBand) {
      const int n = std::min(kBand, rows - a);
      gated_(enc, (std::size_t)a, b.s2, s.hb, (std::size_t)a, n);
    }

    // ---- the head, after the last layer ------------------------------
    if (last) {
      _flops.gemm(N, c.patch_elems(), D);
      for (int a = 0; a < N; a += kBand) {
        const int n = std::min(kBand, N - a);
        norm_(enc, s.x, (std::size_t)a, _ln_w, &_ln_b, s.hb, 0, n, true);
        dense_(enc, s.hb, 0, _head.w, nullptr, s.ff, 0, n, c.patch_elems(),
               D);
        enc.set_function(_fn_unpatch);
        enc.set_buffer(0, s.ff);
        enc.set_buffer(1, _head_bias);
        enc.set_buffer(2, out);
        enc.set_constant(3, a);
        enc.set_constant(4, n);
        enc.set_constant(5, h);
        enc.set_constant(6, w);
        enc.set_constant(7, Fo);
        enc.set_constant(8, tb);
        enc.set_constant(9, c.frames_to_trim);
        enc.set_constant(10, c.patch_t);
        enc.dispatch({(unsigned)c.patch_elems(), (unsigned)n, 1},
                     {256, 1, 1});
      }
    }
    if (!commit_wait(enc)) { return {}; }
    if (_progress) { _progress(L + 1, c.n_layers); }
  }
  if (_i8) { _i8->release_scratch(); }
  return out;
}

}  // namespace genai
}  // namespace vpipe
