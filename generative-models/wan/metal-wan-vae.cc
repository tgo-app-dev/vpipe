#include "generative-models/wan/metal-wan-vae.h"

#include "generative-models/shared/mma-tile.h"

#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/weight-set.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
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

// Namespace for this class's derived-tensor cache keys. A WeightSet is
// shared by everything reading one checkpoint, so a key has to say which
// class's transform produced the bytes, not just which tensor they came
// from. Deliberately DISTINCT from the Qwen-Image VAE's "krea2-vae/": the
// two read the same tensor names off compatible checkpoints but flatten
// them differently (27 taps here, the kt=2 slice there), so sharing a key
// would hand one class the other's bytes.
constexpr const char* kKey = "wan-vae/";

// Read a raw checkpoint tensor as float (F32/F16/BF16 source).
std::vector<float>
read_f32_(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm,
          std::size_t& n_out)
{
  const auto* info = wts.info(nm);
  std::vector<float> v;
  if (info == nullptr || info->shape.empty()) { n_out = 0; return v; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { n_out = 0; return v; }
  v.resize(n);
  if (info->dtype == "F32") {
    std::memcpy(v.data(), raw.contents(), n * 4);
  } else if (info->dtype == "F16") {
    const auto* s = static_cast<const _Float16*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)s[i]; }
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)s[i] << 16;
      float f; std::memcpy(&f, &u, 4); v[i] = f;
    }
  } else {
    n_out = 0; return {};
  }
  n_out = n;
  return v;
}

SharedBuffer
f16_buf_(MetalCompute* mc, const float* src, std::size_t n)
{
  SharedBuffer b = mc->make_shared_buffer(n * 2);
  if (b.empty()) { return {}; }
  auto* d = static_cast<_Float16*>(b.contents());
  for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)src[i]; }
  return b;
}

}  // namespace

// Everything one chunk's dispatches share. The buffer pool is the same
// trick the Qwen-Image VAE uses: the net is a serial feed-forward chain, so
// a released buffer is safe to reuse for a later op (serial dispatch orders
// the reuse after the last read), which bounds the live set to the
// concurrent working set rather than the whole chunk.
struct MetalWanVae::Ctx {
  ComputeEncoder* enc = nullptr;
  struct Slot { SharedBuffer buf; std::size_t cap; bool used; };
  std::deque<Slot> pool;
  bool        alloc_ok = true;
  bool        use_pool = true;
  SharedBuffer col;                  // shared im2col band scratch
  std::size_t  col_cap = 0;          // ELEMENTS

  SharedBuffer&
  alloc(MetalCompute* mc, std::size_t elems)
  {
    const std::size_t bytes = elems * 2;
    if (use_pool) {
      for (auto& s : pool) {
        if (!s.used && !s.buf.empty() && s.cap >= bytes) {
          s.used = true;
          return s.buf;
        }
      }
    }
    pool.push_back(Slot{mc->make_shared_buffer(bytes), bytes, true});
    if (pool.back().buf.empty()) { alloc_ok = false; }
    return pool.back().buf;
  }

  void
  release(const SharedBuffer& b)
  {
    if (!use_pool) { return; }
    for (auto& s : pool) {
      if (&s.buf == &b) { s.used = false; return; }
    }
  }
};

// ---- config ------------------------------------------------------------

bool
MetalWanVae::config_from_json(const std::string& vae_dir, Config& out,
                              std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  fs::path p(vae_dir);
  if (fs::is_directory(p) && !fs::exists(p / "config.json")) {
    p = p / "vae";
  }
  if (fs::is_directory(p)) { p = p / "config.json"; }
  std::ifstream f(p);
  if (!f) { return fail("cannot open " + p.string()); }
  FlexData cfg;
  try {
    cfg = FlexData::from_json(f);
  } catch (...) {
    return fail("cannot parse " + p.string());
  }
  if (!cfg.is_object()) { return fail(p.string() + " is not a JSON object"); }
  auto o = cfg.as_object();
  const std::string cls =
      o.contains("_class_name") ? std::string(o.at("_class_name").as_string(""))
                                : std::string();
  if (cls != "AutoencoderKLWan") {
    return fail("not an AutoencoderKLWan config (_class_name=" + cls + ")");
  }
  auto get_int = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  out.base_dim       = get_int("base_dim", 96);
  out.z_dim          = get_int("z_dim", 16);
  out.num_res_blocks = get_int("num_res_blocks", 2);
  // as_array() returns a VIEW into the owning FlexData, so each one is
  // bound to a named local first -- a temporary would dangle.
  if (o.contains("dim_mult")) {
    const FlexData dm = o.at("dim_mult");
    if (dm.is_array()) {
      auto a = dm.as_array();
      for (std::size_t i = 0; i < 4 && i < a.size(); ++i) {
        out.dim_mult[i] = (int)a.at(i).as_int(out.dim_mult[i]);
      }
    }
  }
  if (o.contains("temperal_downsample")) {
    const FlexData td = o.at("temperal_downsample");
    if (td.is_array()) {
      auto a = td.as_array();
      for (std::size_t i = 0; i < 3 && i < a.size(); ++i) {
        out.temperal_downsample[i] =
            a.at(i).as_bool(out.temperal_downsample[i]);
      }
    }
  }
  auto read_vec = [&](const char* k, std::vector<float>& dst) {
    if (!o.contains(k)) { return; }
    const FlexData v = o.at(k);
    if (!v.is_array()) { return; }
    auto a = v.as_array();
    dst.clear();
    for (std::size_t i = 0; i < a.size(); ++i) {
      dst.push_back((float)a.at(i).as_real(0.0));
    }
  };
  read_vec("latents_mean", out.latents_mean);
  read_vec("latents_std", out.latents_std);
  // A `patch_size` / `is_residual` config is the Wan 2.2 (16x) VAE, whose
  // residual down/up blocks are a different net -- refuse rather than load
  // its weights into this topology and produce noise.
  if (o.contains("is_residual") && o.at("is_residual").as_bool(false)) {
    return fail("is_residual (the Wan2.2 16x VAE) is not supported here");
  }
  return true;
}

// ---- weight loading ----------------------------------------------------

// A causal conv3d [Cout,Cin,3,3,3] as a dense-GEMM weight [Cout, 27*Cin],
// flattened (kt,ky,kx,cin) to pair with im2col_hwc_3x3x3_tiled.
MetalWanVae::Conv
MetalWanVae::load_conv3d_(WeightSet& ws, const std::string& nm)
{
  Conv c;
  const auto* info = ws.src().info(nm + ".weight");
  if (info == nullptr || info->shape.size() < 5) { return c; }
  const auto& sh = info->shape;
  const int Cout = (int)sh[0], Cin = (int)sh[1];
  const int kt = (int)sh[2], kh = (int)sh[3], kw = (int)sh[4];
  if (kt != 3 || kh != 3 || kw != 3) {
    // The only 5-D weights that are not 3x3x3 are the 1x1x1 shortcuts and
    // quant convs, which load through load_conv1x1_.
    return c;
  }
  c.cin = Cin; c.cout = Cout; c.kt = 3; c.ks = 9; c.k = 27 * Cin;
  c.w = ws.derived(std::string(kKey) + "c3d|" + nm, [&]() -> SharedBuffer {
    std::size_t n = 0;
    std::vector<float> w = read_f32_(ws.src(), _mc, nm + ".weight", n);
    if (w.empty()) { return {}; }
    std::vector<float> flat((std::size_t)Cout * 27 * Cin, 0.0f);
    for (int o = 0; o < Cout; ++o) {
      for (int t = 0; t < 3; ++t) {
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            for (int i = 0; i < Cin; ++i) {
              const std::size_t si =
                  ((((std::size_t)o * Cin + i) * 3 + t) * 3 + ky) * 3 + kx;
              const std::size_t di =
                  ((std::size_t)o * 27 + ((t * 3 + ky) * 3 + kx)) * Cin + i;
              flat[di] = w[si];
            }
          }
        }
      }
    }
    return f16_buf_(_mc, flat.data(), flat.size());
  }, _part);
  if (c.w.empty()) { return Conv{}; }
  c.b = load_vec_(ws, nm + ".bias");
  return c;
}

// A plain 2D conv [Cout,Cin,3,3] (the spatial half of a resample) as
// [Cout, 9*Cin], flattened (ky,kx,cin) to pair with im2col_hwc_3x3_tiled.
MetalWanVae::Conv
MetalWanVae::load_conv2d_(WeightSet& ws, const std::string& nm)
{
  Conv c;
  const auto* info = ws.src().info(nm + ".weight");
  if (info == nullptr || info->shape.size() < 4) { return c; }
  const auto& sh = info->shape;
  const int Cout = (int)sh[0], Cin = (int)sh[1];
  c.cin = Cin; c.cout = Cout; c.kt = 1; c.ks = 9; c.k = 9 * Cin;
  c.w = ws.derived(std::string(kKey) + "c2d|" + nm, [&]() -> SharedBuffer {
    std::size_t n = 0;
    std::vector<float> w = read_f32_(ws.src(), _mc, nm + ".weight", n);
    if (w.empty()) { return {}; }
    std::vector<float> flat((std::size_t)Cout * 9 * Cin, 0.0f);
    for (int o = 0; o < Cout; ++o) {
      for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
          for (int i = 0; i < Cin; ++i) {
            const std::size_t si =
                (((std::size_t)o * Cin + i) * 3 + ky) * 3 + kx;
            const std::size_t di =
                ((std::size_t)o * 9 + (ky * 3 + kx)) * Cin + i;
            flat[di] = w[si];
          }
        }
      }
    }
    return f16_buf_(_mc, flat.data(), flat.size());
  }, _part);
  if (c.w.empty()) { return Conv{}; }
  c.b = load_vec_(ws, nm + ".bias");
  return c;
}

// A temporal conv [Cout,Cin,3,1,1] as [Cout, 3*Cin], flattened (kt,cin) to
// pair with concat3_frames.
MetalWanVae::Conv
MetalWanVae::load_time_conv_(WeightSet& ws, const std::string& nm)
{
  Conv c;
  const auto* info = ws.src().info(nm + ".weight");
  if (info == nullptr || info->shape.size() < 5) { return c; }
  const auto& sh = info->shape;
  const int Cout = (int)sh[0], Cin = (int)sh[1];
  if ((int)sh[2] != 3) { return c; }
  c.cin = Cin; c.cout = Cout; c.kt = 3; c.ks = 1; c.k = 3 * Cin;
  c.w = ws.derived(std::string(kKey) + "ct|" + nm, [&]() -> SharedBuffer {
    std::size_t n = 0;
    std::vector<float> w = read_f32_(ws.src(), _mc, nm + ".weight", n);
    if (w.empty()) { return {}; }
    std::vector<float> flat((std::size_t)Cout * 3 * Cin, 0.0f);
    for (int o = 0; o < Cout; ++o) {
      for (int t = 0; t < 3; ++t) {
        for (int i = 0; i < Cin; ++i) {
          const std::size_t si = ((std::size_t)o * Cin + i) * 3 + t;
          flat[((std::size_t)o * 3 + t) * Cin + i] = w[si];
        }
      }
    }
    return f16_buf_(_mc, flat.data(), flat.size());
  }, _part);
  if (c.w.empty()) { return Conv{}; }
  c.b = load_vec_(ws, nm + ".bias");
  return c;
}

// A 1x1(x1) conv as a dense-GEMM weight [Cout, Cin] (the trailing singleton
// dims flatten away).
MetalWanVae::Conv
MetalWanVae::load_conv1x1_(WeightSet& ws, const std::string& nm)
{
  Conv c;
  const auto* info = ws.src().info(nm + ".weight");
  if (info == nullptr || info->shape.size() < 2) { return c; }
  const auto& sh = info->shape;
  c.cout = (int)sh[0]; c.cin = (int)sh[1];
  c.kt = 1; c.ks = 1; c.k = c.cin;
  c.w = load_vec_(ws, nm + ".weight");
  if (c.w.empty()) { return Conv{}; }
  c.b = load_vec_(ws, nm + ".bias");
  return c;
}

// Every scalar/vector/matrix tensor this VAE keeps is stored as f16
// regardless of its on-disk dtype, so "read it as f32 and narrow" IS the
// transform and the result is cached like any other derived tensor.
SharedBuffer
MetalWanVae::load_vec_(WeightSet& ws, const std::string& nm)
{
  return ws.derived(std::string(kKey) + "f16|" + nm, [&]() -> SharedBuffer {
    std::size_t n = 0;
    std::vector<float> v = read_f32_(ws.src(), _mc, nm, n);
    if (v.empty()) { return {}; }
    return f16_buf_(_mc, v.data(), n);
  }, _part);
}

bool
MetalWanVae::load_resblock_(WeightSet& ws, const std::string& pre,
                            ResBlock& rb, int cin, int cout)
{
  rb.cin = cin; rb.cout = cout;
  rb.n1g = load_vec_(ws, pre + "norm1.gamma");
  rb.n2g = load_vec_(ws, pre + "norm2.gamma");
  rb.c1 = load_conv3d_(ws, pre + "conv1");
  rb.c2 = load_conv3d_(ws, pre + "conv2");
  rb.has_short = (cin != cout);
  if (rb.has_short) { rb.shortcut = load_conv1x1_(ws, pre + "conv_shortcut"); }
  return !rb.n1g.empty() && !rb.n2g.empty() && !rb.c1.empty() &&
         !rb.c2.empty() && (!rb.has_short || !rb.shortcut.empty());
}

bool
MetalWanVae::load_attn_(WeightSet& ws, const std::string& pre, Attn& a,
                        int dim)
{
  a.dim = dim;
  a.ng = load_vec_(ws, pre + "norm.gamma");
  // to_qkv is one 1x1 conv [3*dim, dim]; split output channels into q/k/v.
  // Each third is its own derived tensor, keyed by which third it is.
  const std::string qbase = pre + "to_qkv";
  std::size_t n = 0, nb = 0;
  std::vector<float> qkv, qkvb;
  auto read_qkv = [&]() {
    if (!qkv.empty()) { return; }
    qkv  = read_f32_(ws.src(), _mc, qbase + ".weight", n);
    qkvb = read_f32_(ws.src(), _mc, qbase + ".bias", nb);
  };
  const int C = dim;
  auto slice = [&](int off) {
    Conv c; c.cin = C; c.cout = C; c.k = C; c.kt = 1; c.ks = 1;
    const std::string key = std::string(kKey) + "qkv|" + qbase + "|" +
                            std::to_string(off);
    c.w = ws.derived(key + "|w", [&]() -> SharedBuffer {
      read_qkv();
      if (qkv.size() != (std::size_t)3 * C * C) { return {}; }
      return f16_buf_(_mc, qkv.data() + (std::size_t)off * C * C,
                      (std::size_t)C * C);
    }, _part);
    c.b = ws.derived(key + "|b", [&]() -> SharedBuffer {
      read_qkv();
      if (qkvb.size() != (std::size_t)3 * C) { return {}; }
      return f16_buf_(_mc, qkvb.data() + (std::size_t)off * C, (std::size_t)C);
    }, _part);
    return c;
  };
  a.q = slice(0);
  a.k = slice(1);
  a.v = slice(2);
  a.proj = load_conv1x1_(ws, pre + "proj");
  return !a.ng.empty() && !a.q.empty() && !a.k.empty() && !a.v.empty() &&
         !a.proj.empty();
}

std::unique_ptr<MetalWanVae>
MetalWanVae::load(const std::string& model_dir, MetalCompute* mc,
                  const Config& cfg, bool with_encoder)
{
  namespace fs = std::filesystem;
  fs::path p(model_dir);
  // Accept either the pipeline root or the vae/ subdir, as the image VAEs do.
  if (fs::is_directory(p) && fs::exists(p / "vae") &&
      !fs::exists(p / "diffusion_pytorch_model.safetensors")) {
    p = p / "vae";
  }
  return load(WeightSet::open(p.string(), nullptr), mc, cfg, with_encoder);
}

std::unique_ptr<MetalWanVae>
MetalWanVae::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                  const Config& cfg, bool with_encoder)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  WeightSet& wts = *ws_in;

  auto m = std::unique_ptr<MetalWanVae>(new MetalWanVae());
  m->_ws = std::move(ws_in);
  m->_mc = mc;
  m->_cfg = cfg;

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_elt  = mc->load_library("llm_elementwise");
  m->_lib_rms  = mc->load_library("rms_norm");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_fn_gemm_bias   = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_rms         = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_mul_sigmoid = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_residual    = m->_lib_elt.function("residual_add_f16");
  m->_fn_clamp       = m->_lib_elt.function("clamp_f16");
  m->_fn_copy        = m->_lib_elt.function("copy_f16");
  m->_fn_sdpa        = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_sdpa_full_smm = m->_lib_sdpa.function("sdpa_full_mma_f16");
  m->_fn_im2col_tiled = m->_lib_elt.function("im2col_hwc_3x3_tiled_f16");
  m->_fn_im2col_s2_tiled =
      m->_lib_elt.function("im2col_hwc_3x3_s2_tiled_f16");
  m->_fn_im2col3d_tiled =
      m->_lib_elt.function("im2col_hwc_3x3x3_tiled_f16");
  m->_fn_concat3     = m->_lib_elt.function("concat3_frames_f16");
  m->_fn_time_unshuffle = m->_lib_elt.function("wan_time_unshuffle_f16");
  m->_fn_upsample    = m->_lib_elt.function("upsample_nearest2x_hwc_f16");
  if (!m->_fn_gemm_bias.valid() || !m->_fn_rms.valid() ||
      !m->_fn_mul_sigmoid.valid() || !m->_fn_residual.valid() ||
      !m->_fn_clamp.valid() || !m->_fn_copy.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_im2col_tiled.valid() || !m->_fn_im2col_s2_tiled.valid() ||
      !m->_fn_im2col3d_tiled.valid() || !m->_fn_concat3.valid() ||
      !m->_fn_time_unshuffle.valid() || !m->_fn_upsample.valid()) {
    return nullptr;
  }
  // Matrix-core dense GEMM (matmul2d) for the conv/1x1 GEMMs. Same guards
  // as the Qwen-Image VAE: bias folds separately, and a tall GEMM splits at
  // _mma_max_m. VPIPE_WAN_NO_MMA2 forces steel (A/B).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_WAN_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid() &&
                   m->_fn_bias_add.valid();
    if (const char* e = std::getenv("VPIPE_WAN_VAE_MMA_MAX_M")) {
      m->_mma_max_m = std::atoi(e);
    }
  }
  // Mid-block attention: the same single-head spatial attention as the
  // Qwen-Image VAE, run per frame, so the same kernel set and the same
  // measured pick. VPIPE_WAN_NO_MMA_ATTN forces scalar.
  if (std::getenv("VPIPE_WAN_NO_MMA_ATTN") == nullptr) {
    const int mid_d = cfg.base_dim * cfg.dim_mult[3];
    const char* fn = (mid_d == 384) ? "sdpa_full_mma2_d384_f16"
                   : (mid_d == 512) ? "sdpa_full_mma2_d512_f16"
                                    : nullptr;
    if (fn != nullptr) {
      m->_lib_sdpa_mma = mc->load_library("sdpa_mma");
      m->_fn_sdpa_full_mma = m->_lib_sdpa_mma.function(fn);
      m->load_wide_attn_(mid_d);
    }
    m->autotune_mid_attn_(mc, mid_d);
  }

  const int base = cfg.base_dim;                         // 96
  const int dims0 = base * cfg.dim_mult[3];              // 384

  m->_post_quant = m->load_conv1x1_(wts, "post_quant_conv");
  m->_conv_in    = m->load_conv3d_(wts, "decoder.conv_in");
  bool ok = !m->_post_quant.empty() && !m->_conv_in.empty();
  ok = ok && m->load_resblock_(wts, "decoder.mid_block.resnets.0.",
                               m->_mid_res0, dims0, dims0);
  ok = ok && m->load_attn_(wts, "decoder.mid_block.attentions.0.",
                           m->_mid_attn, dims0);
  ok = ok && m->load_resblock_(wts, "decoder.mid_block.resnets.1.",
                               m->_mid_res1, dims0, dims0);

  // Decoder dims = [dim*mult[-1]] + dim*mult[::-1] = [384,384,384,192,96];
  // up i maps dims[i] -> dims[i+1] (for i>0 the resnet in_dim is halved by
  // the previous upsample conv), with an upsampler for i != 3. The
  // temporal flag is temperal_downsample REVERSED.
  const int dims[5] = {dims0, base * cfg.dim_mult[3], base * cfg.dim_mult[2],
                       base * cfg.dim_mult[1], base * cfg.dim_mult[0]};
  const bool tup[3] = {cfg.temperal_downsample[2], cfg.temperal_downsample[1],
                       cfg.temperal_downsample[0]};
  m->_up_blocks.resize(4);
  for (int i = 0; i < 4; ++i) {
    UpBlock& ub = m->_up_blocks[(std::size_t)i];
    int in_dim = (i > 0) ? dims[i] / 2 : dims[i];
    const int out_dim = dims[i + 1];
    ub.resnets.resize((std::size_t)cfg.num_res_blocks + 1);
    int cin = in_dim;
    for (int r = 0; r <= cfg.num_res_blocks; ++r) {
      ok = ok && m->load_resblock_(
          wts, "decoder.up_blocks." + std::to_string(i) + ".resnets." +
                   std::to_string(r) + ".",
          ub.resnets[(std::size_t)r], cin, out_dim);
      cin = out_dim;
    }
    ub.up.present = (i != 3);
    if (ub.up.present) {
      ub.up_dim = out_dim;
      const std::string pre =
          "decoder.up_blocks." + std::to_string(i) + ".upsamplers.0.";
      ub.up.space = m->load_conv2d_(wts, pre + "resample.1");
      ub.up.temporal = tup[i];
      if (ub.up.temporal) {
        ub.up.time = m->load_time_conv_(wts, pre + "time_conv");
        ok = ok && !ub.up.time.empty();
      }
      ok = ok && !ub.up.space.empty();
    }
  }

  m->_norm_out_g = m->load_vec_(wts, "decoder.norm_out.gamma");
  m->_conv_out = m->load_conv3d_(wts, "decoder.conv_out");
  ok = ok && !m->_norm_out_g.empty() && !m->_conv_out.empty();

  if (!ok) { return nullptr; }
  if (with_encoder && !m->ensure_encoder()) { return nullptr; }
  return m;
}

bool
MetalWanVae::load_encoder_(WeightSet& ws)
{
  const int base = _cfg.base_dim;
  // Encoder dims = [base*u for u in [1] + dim_mult] = [96,96,192,384,384].
  const int dims[5] = {base, base * _cfg.dim_mult[0], base * _cfg.dim_mult[1],
                       base * _cfg.dim_mult[2], base * _cfg.dim_mult[3]};
  const int dtop = dims[4];

  _enc_conv_in = load_conv3d_(ws, "encoder.conv_in");
  bool ok = !_enc_conv_in.empty();

  // The encoder's down_blocks are a FLAT ModuleList: per stage,
  // num_res_blocks residual blocks then (for all but the last stage) one
  // resample. So the index advances by num_res_blocks + 1 per stage and the
  // names are not grouped the way the decoder's up_blocks are.
  _enc_down.resize(4);
  int idx = 0;
  for (int i = 0; i < 4; ++i) {
    DownStage& ds = _enc_down[(std::size_t)i];
    int cin = dims[i];
    const int cout = dims[i + 1];
    ds.resnets.resize((std::size_t)_cfg.num_res_blocks);
    for (int r = 0; r < _cfg.num_res_blocks; ++r) {
      ok = ok && load_resblock_(ws,
                                "encoder.down_blocks." + std::to_string(idx) +
                                    ".",
                                ds.resnets[(std::size_t)r], cin, cout);
      cin = cout;
      ++idx;
    }
    ds.down.present = (i != 3);
    if (ds.down.present) {
      const std::string pre = "encoder.down_blocks." + std::to_string(idx) + ".";
      ds.down.space = load_conv2d_(ws, pre + "resample.1");
      ds.down.temporal = _cfg.temperal_downsample[i];
      if (ds.down.temporal) {
        ds.down.time = load_time_conv_(ws, pre + "time_conv");
        ok = ok && !ds.down.time.empty();
      }
      ok = ok && !ds.down.space.empty();
      ++idx;
    }
  }

  ok = ok && load_resblock_(ws, "encoder.mid_block.resnets.0.", _enc_mid_res0,
                            dtop, dtop);
  ok = ok && load_attn_(ws, "encoder.mid_block.attentions.0.", _enc_mid_attn,
                        dtop);
  ok = ok && load_resblock_(ws, "encoder.mid_block.resnets.1.", _enc_mid_res1,
                            dtop, dtop);
  _enc_norm_out_g = load_vec_(ws, "encoder.norm_out.gamma");
  _enc_conv_out = load_conv3d_(ws, "encoder.conv_out");
  _quant_conv = load_conv1x1_(ws, "quant_conv");
  ok = ok && !_enc_norm_out_g.empty() && !_enc_conv_out.empty() &&
       !_quant_conv.empty();
  return ok;
}

bool
MetalWanVae::ensure_encoder()
{
  if (_has_encoder) { return true; }
  if (!_ws) { return false; }
  // Through the WeightSet so the encoder half loads ONCE per checkpoint
  // however many VAEs over it need it -- and, when nobody does, never.
  const bool ok = _ws->ensure_part("encoder", [this]() {
    _part = "encoder";
    const bool r = load_encoder_(*_ws);
    _part.clear();
    return r;
  });
  if (!ok) { return false; }
  // A second VAE over a checkpoint whose encoder another one already loaded
  // still has to populate ITS OWN members; ensure_part reports the cached
  // success without re-running the loader, so bind the (now cache-hit)
  // tensors here.
  if (_enc_conv_in.empty()) {
    _part = "encoder";
    const bool r = load_encoder_(*_ws);
    _part.clear();
    if (!r) { return false; }
  }
  _has_encoder = true;
  return true;
}

// ---- mid-block attention (shared kernel set) ---------------------------

void
MetalWanVae::load_wide_attn_(int mid_d)
{
  const std::string base = "sdpa_full_mma2_d" + std::to_string(mid_d) + "_q";
  _fn_sdpa_full_wide16 = _lib_sdpa_mma.function(base + "16_f16");
  _fn_sdpa_full_wide32 = _lib_sdpa_mma.function(base + "32_f16");
  _fn_sdpa_full_wide64 = _lib_sdpa_mma.function(base + "64_f16");
}

bool
MetalWanVae::mid_attn_available_(MidAttn k) const
{
  switch (k) {
    case MidAttn::kScalar: return _fn_sdpa.valid();
    case MidAttn::kSmm:    return _fn_sdpa_full_smm.valid();
    case MidAttn::kMma8:   return _fn_sdpa_full_mma.valid();
    case MidAttn::kWide16: return _fn_sdpa_full_wide16.valid();
    case MidAttn::kWide32: return _fn_sdpa_full_wide32.valid();
    case MidAttn::kWide64: return _fn_sdpa_full_wide64.valid();
    case MidAttn::kMat:    return false;   // no materialized path here
  }
  return false;
}

void
MetalWanVae::encode_mid_attn_(ComputeEncoder& enc, MidAttn kind,
                              const SharedBuffer& q, const SharedBuffer& k,
                              const SharedBuffer& v, const SharedBuffer& att,
                              std::size_t hw, int C, float scale)
{
  enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
  enc.set_buffer(3, att);
  enc.set_constant(4, scale); enc.set_constant(5, (int)hw);
  enc.set_constant(6, C); enc.set_constant(7, 1); enc.set_constant(8, 1);
  enc.set_constant(9, (int)hw); enc.set_constant(10, (int)hw);
  switch (kind) {
    case MidAttn::kWide16:
    case MidAttn::kWide32:
    case MidAttn::kWide64: {
      const int bq = (kind == MidAttn::kWide16) ? 16
                   : (kind == MidAttn::kWide32) ? 32 : 64;
      enc.set_function(kind == MidAttn::kWide16 ? _fn_sdpa_full_wide16
                     : kind == MidAttn::kWide32 ? _fn_sdpa_full_wide32
                                                : _fn_sdpa_full_wide64);
      const unsigned nt = attn_threads_(bq);
      enc.dispatch({nt, 1, (unsigned)(((int)hw + bq - 1) / bq)}, {nt, 1, 1});
      break;
    }
    case MidAttn::kMma8:
      enc.set_function(_fn_sdpa_full_mma);
      enc.dispatch({4 * 32, 1, (unsigned)((hw + 7) / 8)}, {4 * 32, 1, 1});
      break;
    case MidAttn::kSmm: {
      enc.set_function(_fn_sdpa_full_smm);
      const unsigned nt = 4u * (unsigned)(C / 64) * 32u;   // WM*WD*32
      enc.dispatch({nt, 1, (unsigned)((hw + 31) / 32)}, {nt, 1, 1});
      break;
    }
    default:
      enc.set_function(_fn_sdpa);                    // scalar O(N^2)
      enc.dispatch({32, 1, (unsigned)hw}, {32, 1, 1});
      break;
  }
}

void
MetalWanVae::autotune_mid_attn_(MetalCompute* mc, int C)
{
  _attn_pick = MidAttn::kScalar;
  const bool smm_ok = mid_attn_available_(MidAttn::kSmm) &&
                      (C % 64 == 0) && (C <= 512);
  if (smm_ok) { _attn_pick = MidAttn::kSmm; }
  if (mc->supports_matrix_cores()) {
    if (mid_attn_available_(MidAttn::kMma8))   { _attn_pick = MidAttn::kMma8; }
    if (mid_attn_available_(MidAttn::kWide32)) { _attn_pick = MidAttn::kWide32; }
  }
  if (const char* e = std::getenv("VPIPE_WAN_VAE_ATTN_BQ")) {
    const int b = std::atoi(e);
    if (b == 8  && mid_attn_available_(MidAttn::kMma8))   { _attn_pick = MidAttn::kMma8; }
    if (b == 16 && mid_attn_available_(MidAttn::kWide16)) { _attn_pick = MidAttn::kWide16; }
    if (b == 32 && mid_attn_available_(MidAttn::kWide32)) { _attn_pick = MidAttn::kWide32; }
    if (b == 64 && mid_attn_available_(MidAttn::kWide64)) { _attn_pick = MidAttn::kWide64; }
    return;
  }
  std::vector<MidAttn> cands;
  for (MidAttn k : {MidAttn::kSmm, MidAttn::kMma8, MidAttn::kWide16,
                    MidAttn::kWide32, MidAttn::kWide64}) {
    if (k == MidAttn::kSmm && !smm_ok) { continue; }
    if (mid_attn_available_(k)) { cands.push_back(k); }
  }
  std::string detail;
  const auto t0 = std::chrono::steady_clock::now();
  _attn_pick = vae_mid_attn::autotune<MetalCompute, ComputeEncoder>(
      mc, C, cands, _attn_pick,
      [this](ComputeEncoder& enc, MidAttn kind, const SharedBuffer& qq,
             const SharedBuffer& kk, const SharedBuffer& vv,
             const SharedBuffer& oo, std::size_t hw, int c, float sc,
             const vae_mid_attn::Alloc&, const vae_mid_attn::Release&) {
        encode_mid_attn_(enc, kind, qq, kk, vv, oo, hw, c, sc);
      },
      &detail);
  const double tune_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  if (mc->session() != nullptr && !detail.empty()) {
    const auto line = fmt(
        "MetalWanVae: mid-attn autotune (D={}) -> {} [{}] in {} ms",
        C, vae_mid_attn::name(_attn_pick), detail, (long long)tune_ms);
    if (std::getenv("VPIPE_VAE_ATTN_TUNE_LOG") != nullptr) {
      mc->session()->log_normal(line);
    } else {
      mc->session()->log_debug(line);
    }
  }
}

// ---- forward primitives ------------------------------------------------

int
MetalWanVae::mma_row_chunk(int M, int N, int K, int max_m)
{
  int chunk = M > 0 ? M : 1;
  if (max_m > 0 && chunk > max_m) { chunk = max_m; }
  const int band = mma_row_band(N, K);
  if (chunk > band) { chunk = band; }
  return chunk;
}

void
MetalWanVae::gemm_bias_(Ctx& cx, const SharedBuffer& x, const SharedBuffer& w,
                        const SharedBuffer& b, const SharedBuffer& y, int M,
                        int N, int K, int y_row0, std::size_t x_off_rows)
{
  ComputeEncoder& enc = *cx.enc;
  const std::size_t ybase = (std::size_t)y_row0 * N;
  const std::size_t xbase = x_off_rows * (std::size_t)K;
  if (_use_mma2 && M >= _mma_min_m && N >= _mma_min_n) {
    const bool deep = (K >= 6144);
    const int BN = deep ? 256 : 128;
    // Split a tall GEMM: past ~2^19 rows matmul2d corrupts its output,
    // and past 2^31 bytes on ANY operand it stops storing. See
    // mma_row_chunk -- the second limit is the one the row cap alone
    // misses, because it binds on the 27-tap im2col SOURCE.
    const int chunk = mma_row_chunk(M, N, K, _mma_max_m);
    for (int r0 = 0; r0 < M; r0 += chunk) {
      const int mc = (M - r0 < chunk) ? (M - r0) : chunk;
      enc.set_function(deep ? _fn_dense_mma_deep : _fn_dense_mma);
      enc.set_buffer(0, x, (xbase + (std::size_t)r0 * K) * 2);
      enc.set_buffer(1, w);
      enc.set_buffer(2, w);        // bias slot unused (has_bias=0)
      enc.set_buffer(3, y, (ybase + (std::size_t)r0 * N) * 2);
      enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, mc);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                    (unsigned)((mc + 127) / 128), 1}, {256, 1, 1});
    }
    if (!b.empty()) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, ybase * 2); enc.set_buffer(1, b);
      enc.set_constant(2, N);
      enc.set_constant(3, (unsigned)((std::size_t)M * N));
      enc.dispatch({(unsigned)N, (unsigned)M, 1}, {256, 1, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm_bias);
  enc.set_buffer(0, x, xbase * 2); enc.set_buffer(1, w);
  enc.set_buffer(2, b.empty() ? w : b); enc.set_buffer(3, y, ybase * 2);
  enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
  enc.set_constant(7, b.empty() ? 0 : 1);
  enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
}

// One output FRAME. `taps[kt]` is the buffer holding temporal tap kt and
// `tap_off[kt]` its ELEMENT offset; a null tap is masked to zero (a frame
// before the start of the sequence). Rows are streamed in bands so the col
// scratch stays bounded -- at 27 taps a full-res [H*W, 27*Cin] would be
// several GB.
void
MetalWanVae::conv_frame_(Ctx& cx, const Conv& c,
                         const SharedBuffer* const taps[3],
                         const std::size_t tap_off[3], const SharedBuffer& out,
                         std::size_t out_row0, int H, int W, int stride)
{
  ComputeEncoder& enc = *cx.enc;
  const int OH = (stride == 2) ? H / 2 : H;
  const int OW = (stride == 2) ? W / 2 : W;
  const std::size_t ohw = (std::size_t)OH * OW;

  // 1x1: no gather at all, the activation IS the GEMM input.
  if (c.ks == 1 && c.kt == 1) {
    gemm_bias_(cx, *taps[0], c.w, c.b, out, (int)ohw, c.cout, c.cin,
               (int)out_row0, tap_off[0] / (std::size_t)std::max(c.cin, 1));
    return;
  }

  // Temporal-only (3,1,1): concat the three frames channel-wise.
  if (c.ks == 1) {
    const std::size_t per_row = (std::size_t)3 * c.cin;
    std::size_t rows = (per_row > 0) ? (cx.col_cap / per_row) : ohw;
    if (_mma_max_m > 0 && rows > (std::size_t)_mma_max_m / 2) {
      rows = (std::size_t)_mma_max_m / 2;
    }
    rows = std::min(std::max<std::size_t>(rows, 1), ohw);
    int mask = 0;
    for (int t = 0; t < 3; ++t) { if (taps[t] != nullptr) { mask |= 1 << t; } }
    const SharedBuffer* any = taps[0] != nullptr ? taps[0]
                            : taps[1] != nullptr ? taps[1] : taps[2];
    for (std::size_t r0 = 0; r0 < ohw; r0 += rows) {
      const int mc = (int)std::min(rows, ohw - r0);
      enc.set_function(_fn_concat3);
      for (int t = 0; t < 3; ++t) {
        const SharedBuffer* b = taps[t] != nullptr ? taps[t] : any;
        const std::size_t off =
            (taps[t] != nullptr) ? (tap_off[t] + r0 * (std::size_t)c.cin) : 0;
        enc.set_buffer((unsigned)t, *b, off * 2);
      }
      enc.set_buffer(3, cx.col);
      enc.set_constant(4, c.cin); enc.set_constant(5, mc);
      enc.set_constant(6, mask);
      enc.dispatch({(unsigned)(3 * c.cin), (unsigned)mc, 1}, {64, 1, 1});
      gemm_bias_(cx, cx.col, c.w, c.b, out, mc, c.cout, c.k,
                 (int)(out_row0 + r0));
    }
    return;
  }

  // Spatial 3x3, either the 27-tap causal conv3d or the plain 2D resample.
  const std::size_t per_row = (std::size_t)c.k;
  std::size_t rows = (per_row > 0) ? (cx.col_cap / per_row) : ohw;
  if (_mma_max_m > 0 && rows > (std::size_t)_mma_max_m / 2) {
    rows = (std::size_t)_mma_max_m / 2;
  }
  rows = std::min(std::max<std::size_t>(rows, 1), ohw);
  int mask = 0;
  for (int t = 0; t < 3; ++t) { if (taps[t] != nullptr) { mask |= 1 << t; } }
  const SharedBuffer* any = taps[0] != nullptr ? taps[0]
                          : taps[1] != nullptr ? taps[1] : taps[2];
  for (std::size_t r0 = 0; r0 < ohw; r0 += rows) {
    const int mc = (int)std::min(rows, ohw - r0);
    if (c.kt == 3) {
      enc.set_function(_fn_im2col3d_tiled);
      for (int t = 0; t < 3; ++t) {
        const SharedBuffer* b = taps[t] != nullptr ? taps[t] : any;
        enc.set_buffer((unsigned)t, *b,
                       (taps[t] != nullptr ? tap_off[t] : 0) * 2);
      }
      enc.set_buffer(3, cx.col);
      enc.set_constant(4, H); enc.set_constant(5, W);
      enc.set_constant(6, c.cin);
      enc.set_constant(7, (int)r0); enc.set_constant(8, mc);
      enc.set_constant(9, mask);
      enc.dispatch({(unsigned)c.k, (unsigned)mc, 1}, {64, 1, 1});
    } else {
      // The whole [H*W, cin] frame stays bound: the 3x3 halo spans band
      // edges, so the band applies to OUTPUT rows only.
      enc.set_function(stride == 2 ? _fn_im2col_s2_tiled : _fn_im2col_tiled);
      enc.set_buffer(0, *taps[0], tap_off[0] * 2);
      enc.set_buffer(1, cx.col);
      enc.set_constant(2, H); enc.set_constant(3, W);
      enc.set_constant(4, c.cin);
      enc.set_constant(5, (int)r0); enc.set_constant(6, mc);
      enc.dispatch({(unsigned)(9 * c.cin), (unsigned)mc, 1}, {64, 1, 1});
    }
    gemm_bias_(cx, cx.col, c.w, c.b, out, mc, c.cout, c.k,
               (int)(out_row0 + r0));
  }
}

// Whole-chunk convolution. Output frame o of a stride-1 causal conv reads
// input frames o-2, o-1, o; anything below 0 comes from `carry` (the
// previous chunk's trailing frames) and anything below -carry->frames is
// before the start of the sequence and masks to zero. `stride` here is the
// SPATIAL stride; the encoder's temporal downsample has its own routine.
SharedBuffer&
MetalWanVae::conv_chunk_(Ctx& cx, const Conv& c, const SharedBuffer& in,
                         int t_in, int H, int W, int stride, Carry* carry)
{
  const int OH = (stride == 2) ? H / 2 : H;
  const int OW = (stride == 2) ? W / 2 : W;
  const std::size_t ihw = (std::size_t)H * W;
  const std::size_t ohw = (std::size_t)OH * OW;
  SharedBuffer& out = cx.alloc(_mc, (std::size_t)t_in * ohw * c.cout);
  if (!cx.alloc_ok) { return out; }

  // A 1x1 with no temporal extent is the same GEMM for every frame, so run
  // the whole chunk as one tall GEMM instead of t of them.
  if (c.kt == 1 && c.ks == 1) {
    gemm_bias_(cx, in, c.w, c.b, out, (int)((std::size_t)t_in * ohw), c.cout,
               c.cin);
    return out;
  }

  for (int o = 0; o < t_in; ++o) {
    const SharedBuffer* taps[3] = {nullptr, nullptr, nullptr};
    std::size_t offs[3] = {0, 0, 0};
    if (c.kt == 1) {
      taps[0] = &in;
      offs[0] = (std::size_t)o * ihw * c.cin;
    } else {
      for (int kt = 0; kt < 3; ++kt) {
        const int fi = o - 2 + kt;         // input frame index in the chunk
        if (fi >= 0) {
          taps[kt] = &in;
          offs[kt] = (std::size_t)fi * ihw * c.cin;
        } else if (carry != nullptr && carry->total + fi >= 0) {
          // Slot j % 2 for sequence frame j; fi is -1 or -2 here, so the
          // absolute index is carry->total + fi.
          taps[kt] = &carry->buf;
          offs[kt] =
              (std::size_t)((carry->total + fi) % 2) * ihw * c.cin;
        }
        // else: before the sequence -- left null, masked to zero.
      }
    }
    conv_frame_(cx, c, taps, offs, out, (std::size_t)o * ohw, H, W, stride);
  }
  if (carry != nullptr && c.kt == 3) {
    save_carry_(cx, *carry, in, t_in, ihw, c.cin);
  }
  return out;
}

void
MetalWanVae::save_carry_(Ctx& cx, Carry& carry, const SharedBuffer& x, int t,
                         std::size_t hw, int cin)
{
  const std::size_t frame_el = hw * (std::size_t)cin;
  if (carry.buf.empty() || carry.hw != hw || carry.cin != cin) {
    carry.buf = _mc->make_shared_buffer(2 * frame_el * 2);
    if (carry.buf.empty()) { cx.alloc_ok = false; return; }
    carry.hw = hw;
    carry.cin = cin;
    carry.total = 0;
  }
  // Only the last two frames of the chunk can still be read, and each goes
  // to the slot its SEQUENCE index selects -- so nothing has to move when
  // a one-frame chunk lands next to a two-frame history. The copies are
  // GPU-side: x exists only on the GPU timeline at this point.
  ComputeEncoder& enc = *cx.enc;
  const int first = std::max(0, t - 2);
  for (int j = first; j < t; ++j) {
    const std::size_t slot = (std::size_t)((carry.total + j) % 2);
    enc.set_function(_fn_copy);
    enc.set_buffer(0, x, (std::size_t)j * frame_el * 2);
    enc.set_buffer(1, carry.buf);
    enc.set_constant(2, (int)(slot * frame_el));
    enc.set_constant(3, (int)frame_el);
    enc.dispatch({(unsigned)frame_el, 1, 1}, {256, 1, 1});
  }
  carry.total += t;
}

SharedBuffer&
MetalWanVae::normc_(Ctx& cx, const SharedBuffer& in, std::size_t rows, int C,
                    const SharedBuffer& g)
{
  // WanRMS_norm over the channel axis: x/||x|| * sqrt(C) * gamma. The rms
  // kernel's x*rsqrt(mean+eps)*w matches with w=gamma and eps ~ 0.
  SharedBuffer& out = cx.alloc(_mc, rows * (std::size_t)C);
  const float eps = 1e-12f / (float)C;
  ComputeEncoder& enc = *cx.enc;
  enc.set_function(_fn_rms);
  enc.set_buffer(0, in); enc.set_buffer(1, g); enc.set_buffer(2, out);
  enc.set_constant(3, C); enc.set_constant(4, eps);
  enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
  return out;
}

void
MetalWanVae::silu_(Ctx& cx, const SharedBuffer& x, std::size_t n)
{
  ComputeEncoder& enc = *cx.enc;
  enc.set_function(_fn_mul_sigmoid);
  enc.set_buffer(0, x); enc.set_buffer(1, x); enc.set_buffer(2, x);
  enc.set_constant(3, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
}

SharedBuffer&
MetalWanVae::resadd_(Ctx& cx, const SharedBuffer& a, const SharedBuffer& b,
                     std::size_t n)
{
  SharedBuffer& out = cx.alloc(_mc, n);
  ComputeEncoder& enc = *cx.enc;
  enc.set_function(_fn_residual);
  enc.set_buffer(0, a); enc.set_buffer(1, b); enc.set_buffer(2, out);
  enc.set_constant(3, (int)n);
  enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  return out;
}

SharedBuffer&
MetalWanVae::upsample2x_(Ctx& cx, const SharedBuffer& in, int t, int H, int W,
                         int C)
{
  const std::size_t ihw = (std::size_t)H * W;
  SharedBuffer& out = cx.alloc(_mc, (std::size_t)t * 4 * ihw * C);
  ComputeEncoder& enc = *cx.enc;
  for (int f = 0; f < t; ++f) {
    enc.set_function(_fn_upsample);
    enc.set_buffer(0, in, (std::size_t)f * ihw * C * 2);
    enc.set_buffer(1, out, (std::size_t)f * 4 * ihw * C * 2);
    enc.set_constant(2, H); enc.set_constant(3, W); enc.set_constant(4, C);
    enc.dispatch({(unsigned)C, (unsigned)(4 * ihw), 1}, {256, 1, 1});
  }
  return out;
}

SharedBuffer&
MetalWanVae::resblock_(Ctx& cx, const ResBlock& rb, const SharedBuffer& x,
                       int t, int H, int W, Carry* c1, Carry* c2)
{
  const std::size_t rows = (std::size_t)t * H * W;
  SharedBuffer& n1 = normc_(cx, x, rows, rb.cin, rb.n1g);
  silu_(cx, n1, rows * (std::size_t)rb.cin);
  SharedBuffer& a = conv_chunk_(cx, rb.c1, n1, t, H, W, 1, c1);
  cx.release(n1);
  SharedBuffer& n2 = normc_(cx, a, rows, rb.cout, rb.n2g);
  cx.release(a);
  silu_(cx, n2, rows * (std::size_t)rb.cout);
  SharedBuffer& b = conv_chunk_(cx, rb.c2, n2, t, H, W, 1, c2);
  cx.release(n2);
  if (rb.has_short) {
    SharedBuffer& h = conv_chunk_(cx, rb.shortcut, x, t, H, W, 1, nullptr);
    SharedBuffer& out = resadd_(cx, b, h, rows * (std::size_t)rb.cout);
    cx.release(b); cx.release(h);
    return out;
  }
  SharedBuffer& out = resadd_(cx, b, x, rows * (std::size_t)rb.cout);
  cx.release(b);
  return out;
}

SharedBuffer&
MetalWanVae::attention_(Ctx& cx, const Attn& a, const SharedBuffer& x, int t,
                        int H, int W)
{
  const std::size_t hw = (std::size_t)H * W;
  const int C = a.dim;
  const std::size_t rows = (std::size_t)t * hw;
  SharedBuffer& n = normc_(cx, x, rows, C, a.ng);
  SharedBuffer& q = cx.alloc(_mc, rows * C);
  SharedBuffer& k = cx.alloc(_mc, rows * C);
  SharedBuffer& v = cx.alloc(_mc, rows * C);
  gemm_bias_(cx, n, a.q.w, a.q.b, q, (int)rows, C, C);
  gemm_bias_(cx, n, a.k.w, a.k.b, k, (int)rows, C, C);
  gemm_bias_(cx, n, a.v.w, a.v.b, v, (int)rows, C, C);
  cx.release(n);
  SharedBuffer& att = cx.alloc(_mc, rows * C);
  const float scale = 1.0f / std::sqrt((float)C);
  // The attention is per FRAME (the reference folds time into the batch),
  // so each frame is its own hw-long sequence.
  for (int f = 0; f < t; ++f) {
    // A frame view of each of q/k/v/att. The attention kernels take whole
    // buffers, so bind the slices through the encoder's byte offset.
    ComputeEncoder& enc = *cx.enc;
    const std::size_t off = (std::size_t)f * hw * C * 2;
    enc.set_buffer(0, q, off); enc.set_buffer(1, k, off);
    enc.set_buffer(2, v, off); enc.set_buffer(3, att, off);
    enc.set_constant(4, scale); enc.set_constant(5, (int)hw);
    enc.set_constant(6, C); enc.set_constant(7, 1); enc.set_constant(8, 1);
    enc.set_constant(9, (int)hw); enc.set_constant(10, (int)hw);
    switch (_attn_pick) {
      case MidAttn::kWide16:
      case MidAttn::kWide32:
      case MidAttn::kWide64: {
        const int bq = (_attn_pick == MidAttn::kWide16) ? 16
                     : (_attn_pick == MidAttn::kWide32) ? 32 : 64;
        enc.set_function(_attn_pick == MidAttn::kWide16 ? _fn_sdpa_full_wide16
                       : _attn_pick == MidAttn::kWide32 ? _fn_sdpa_full_wide32
                                                        : _fn_sdpa_full_wide64);
        const unsigned nt = attn_threads_(bq);
        enc.dispatch({nt, 1, (unsigned)(((int)hw + bq - 1) / bq)}, {nt, 1, 1});
        break;
      }
      case MidAttn::kMma8:
        enc.set_function(_fn_sdpa_full_mma);
        enc.dispatch({4 * 32, 1, (unsigned)((hw + 7) / 8)}, {4 * 32, 1, 1});
        break;
      case MidAttn::kSmm: {
        enc.set_function(_fn_sdpa_full_smm);
        const unsigned nt = 4u * (unsigned)(C / 64) * 32u;
        enc.dispatch({nt, 1, (unsigned)((hw + 31) / 32)}, {nt, 1, 1});
        break;
      }
      default:
        enc.set_function(_fn_sdpa);
        enc.dispatch({32, 1, (unsigned)hw}, {32, 1, 1});
        break;
    }
  }
  cx.release(q); cx.release(k); cx.release(v);
  SharedBuffer& p = cx.alloc(_mc, rows * C);
  gemm_bias_(cx, att, a.proj.w, a.proj.b, p, (int)rows, C, C);
  cx.release(att);
  SharedBuffer& out = resadd_(cx, p, x, rows * (std::size_t)C);
  cx.release(p);
  return out;
}

// The decoder's temporal upsample. The FIRST chunk is passed through
// untouched (the reference's "Rep" sentinel) -- so the sequence the
// time_conv sees starts at chunk 1, and its own causal zero-padding then
// falls at that chunk rather than at the clip's first frame. Every later
// chunk doubles its frame count.
SharedBuffer&
MetalWanVae::time_up_(Ctx& cx, const Resample& rs, const SharedBuffer& x,
                      int& t, std::size_t hw, int C, Carry* carry)
{
  if (carry != nullptr && !carry->seen) {
    carry->seen = true;
    return const_cast<SharedBuffer&>(x);
  }
  // Causal (3,1,1) over the frames, producing 2*C channels.
  SharedBuffer& wide = conv_chunk_(cx, rs.time, x, t, (int)hw, 1, 1, carry);
  SharedBuffer& out = cx.alloc(_mc, (std::size_t)t * 2 * hw * C);
  if (!cx.alloc_ok) { return out; }
  ComputeEncoder& enc = *cx.enc;
  enc.set_function(_fn_time_unshuffle);
  enc.set_buffer(0, wide); enc.set_buffer(1, out);
  enc.set_constant(2, C); enc.set_constant(3, (int)hw); enc.set_constant(4, t);
  enc.dispatch({(unsigned)C, (unsigned)hw, (unsigned)(2 * t)}, {64, 1, 1});
  cx.release(wide);
  t *= 2;
  return out;
}

// The encoder's temporal downsample. The first chunk keeps its frames and
// only seeds the carry; every later chunk prepends the single carried
// frame and strides by two, so a four-frame chunk becomes two frames and a
// two-frame chunk becomes one.
SharedBuffer&
MetalWanVae::time_down_(Ctx& cx, const Resample& rs, const SharedBuffer& x,
                        int& t, std::size_t hw, int C, Carry* carry)
{
  if (carry != nullptr && !carry->seen) {
    carry->seen = true;
    save_carry_(cx, *carry, x, t, hw, C);
    return const_cast<SharedBuffer&>(x);
  }
  // The concatenated sequence is [carry_last, chunk...]; output frame o
  // reads concatenated frames 2o, 2o+1, 2o+2 (kernel 3, stride 2, NO
  // padding), i.e. chunk frames 2o-1, 2o, 2o+1.
  const int t_cat = t + 1;
  const int t_out = (t_cat >= 3) ? ((t_cat - 3) / 2 + 1) : 0;
  SharedBuffer& out = cx.alloc(_mc, (std::size_t)std::max(t_out, 1) * hw * C);
  if (!cx.alloc_ok) { return out; }
  const std::size_t frame_el = hw * (std::size_t)C;
  for (int o = 0; o < t_out; ++o) {
    const SharedBuffer* taps[3] = {nullptr, nullptr, nullptr};
    std::size_t offs[3] = {0, 0, 0};
    for (int kt = 0; kt < 3; ++kt) {
      const int ci = 2 * o + kt - 1;       // chunk-local frame index
      if (ci >= 0) {
        taps[kt] = &x;
        offs[kt] = (std::size_t)ci * frame_el;
      } else if (carry != nullptr && carry->total >= 1) {
        // The single carried frame is the previous chunk's last, at the
        // slot its sequence index selects.
        taps[kt] = &carry->buf;
        offs[kt] = (std::size_t)((carry->total - 1) % 2) * frame_el;
      }
    }
    conv_frame_(cx, rs.time, taps, offs, out, (std::size_t)o * hw, (int)hw, 1,
                1);
  }
  if (carry != nullptr) { save_carry_(cx, *carry, x, t, hw, C); }
  t = t_out;
  return out;
}

// ---- decode ------------------------------------------------------------

std::size_t
MetalWanVae::decode_peak_bytes(int h8, int w8) const noexcept
{
  // One chunk is at most four frames at full resolution. The dominant
  // terms are the top level's activations (4 frames x H x W x base) held
  // alongside a resblock's two intermediates, plus the level below at
  // twice the channels and a quarter the pixels.
  const std::size_t hw = (std::size_t)h8 * w8;
  const std::size_t base = (std::size_t)_cfg.base_dim;
  const std::size_t top = 4 * 64 * hw * base * 2;         // 4 frames, 8x8 px
  return top * 3 + top / 2;
}

bool
MetalWanVae::decode(const SharedBuffer& z, int T, int h8, int w8,
                    const FrameSink& on_frame, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  const int Cz = _cfg.z_dim;
  const std::size_t hw0 = (std::size_t)h8 * w8;
  if (T <= 0 || h8 <= 0 || w8 <= 0) { return fail("empty latent"); }
  if (z.byte_size() < (std::size_t)Cz * T * hw0 * 2) {
    return fail("input latent smaller than [z_dim, T, h8, w8]");
  }
  if (!on_frame) { return fail("no frame sink"); }
  MetalCompute* mc = _mc;
  const int Hout = h8 * 8, Wout = w8 * 8;
  const int base = _cfg.base_dim;

  // Preflight: refuse a decode that clearly won't fit rather than
  // allocating into an out-of-memory mid-clip (which corrupts the output).
  std::size_t headroom = 0;
  {
    const MetalCompute::MemoryBudget mb = mc->memory_budget();
    headroom = (mb.recommended != 0) ? mb.headroom : 0;
    const std::size_t need = decode_peak_bytes(h8, w8);
    if (mb.recommended != 0 && !mb.fits(need)) {
      return fail(fmt(
          "insufficient GPU memory for a {}x{} video decode: need ~{} MB per "
          "chunk, {} MB free of {} MB working set (lower the resolution or "
          "free other resident models)", Wout, Hout, need >> 20,
          mb.headroom >> 20, mb.recommended >> 20)());
    }
    if (!mb.fits_physical(need)) {
      return fail(fmt(
          "insufficient free RAM for a {}x{} video decode: need ~{} MB per "
          "chunk, ~{} MB reclaimable", Wout, Hout, need >> 20,
          mb.available_physical >> 20)());
    }
  }

  // The im2col band. At 27 taps the full [H*W, 27*cin] of a top-level conv
  // is multi-GB, so it is always streamed; the cap gets whatever headroom
  // the chunk activations leave, floored at eight output rows.
  const std::size_t widest = (std::size_t)27 * base * _cfg.dim_mult[1];
  const std::size_t floor_band = (std::size_t)Wout * widest * 8;
  const std::size_t full_band = (std::size_t)Hout * Wout * widest;
  std::size_t col_cap = full_band;
  if (headroom > 0) {
    const std::size_t reserve = decode_peak_bytes(h8, w8);
    const std::size_t avail = headroom > reserve ? (headroom - reserve) / 2 : 0;
    col_cap = std::min(full_band, std::max(floor_band, avail));
  }
  if (_mma_max_m > 0) {
    col_cap = std::min(col_cap, (std::size_t)_mma_max_m / 2 * widest);
  }
  if (const char* e = std::getenv("VPIPE_WAN_VAE_BAND_ROWS")) {
    const long r = std::atol(e);
    if (r > 0) { col_cap = (std::size_t)r * widest; }
  }

  // Per-conv carries, in traversal order. The count is fixed by the
  // topology, so index them by a counter reset at each chunk exactly as
  // the reference resets feat_idx.
  std::vector<Carry> carry;
  carry.resize(256);
  std::size_t ci = 0;
  auto next_carry = [&]() -> Carry* {
    if (ci >= carry.size()) { carry.resize(ci + 64); }
    return &carry[ci++];
  };

  for (int f = 0; f < T; ++f) {
    ci = 0;
    CommandStream stream = mc->make_command_stream();
    Ctx cx;
    cx.use_pool = std::getenv("VPIPE_WAN_NO_VAE_POOL") == nullptr;
    cx.col = mc->make_shared_buffer(col_cap * 2);
    if (cx.col.empty()) {
      return fail("im2col band scratch allocation failed (out of GPU memory)");
    }
    cx.col_cap = col_cap;
    const SharedBuffer* rgb = nullptr;
    int t = 1;                       // frames in this chunk
    int H = h8, W = w8;
    {
      ComputeEncoder enc = stream.begin_compute();
      cx.enc = &enc;

      // The latent frame, channel-first [Cz, T, h8, w8] -> channel-last
      // [hw0, Cz] (host-side; the latent arrives from the sampler).
      SharedBuffer& x0 = cx.alloc(mc, hw0 * Cz);
      if (!cx.alloc_ok) { return fail("chunk allocation failed"); }
      {
        const auto* s = static_cast<const _Float16*>(z.contents());
        auto* d = static_cast<_Float16*>(x0.contents());
        for (int c = 0; c < Cz; ++c) {
          const std::size_t src = ((std::size_t)c * T + f) * hw0;
          for (std::size_t p = 0; p < hw0; ++p) {
            d[p * Cz + c] = s[src + p];
          }
        }
      }
      // post_quant_conv (1x1), then the decoder proper.
      SharedBuffer& pq = cx.alloc(mc, hw0 * Cz);
      gemm_bias_(cx, x0, _post_quant.w, _post_quant.b, pq, (int)hw0, Cz, Cz);
      cx.release(x0);

      const SharedBuffer* x = &conv_chunk_(cx, _conv_in, pq, t, H, W, 1,
                                           next_carry());
      cx.release(pq);
      auto step = [&](SharedBuffer& nx) { cx.release(*x); x = &nx; };

      {
        Carry* a = next_carry(); Carry* b = next_carry();
        step(resblock_(cx, _mid_res0, *x, t, H, W, a, b));
      }
      step(attention_(cx, _mid_attn, *x, t, H, W));
      {
        Carry* a = next_carry(); Carry* b = next_carry();
        step(resblock_(cx, _mid_res1, *x, t, H, W, a, b));
      }

      for (const UpBlock& ub : _up_blocks) {
        for (const ResBlock& rb : ub.resnets) {
          Carry* a = next_carry(); Carry* b = next_carry();
          step(resblock_(cx, rb, *x, t, H, W, a, b));
        }
        if (ub.up.present) {
          if (ub.up.temporal) {
            SharedBuffer& up = time_up_(cx, ub.up, *x, t, (std::size_t)H * W,
                                        ub.up_dim, next_carry());
            if (&up != x) { step(up); }
          }
          step(upsample2x_(cx, *x, t, H, W, ub.up_dim));
          H *= 2; W *= 2;
          step(conv_chunk_(cx, ub.up.space, *x, t, H, W, 1, nullptr));
        }
        if (!cx.alloc_ok) { return fail("chunk allocation failed"); }
      }

      const std::size_t rows = (std::size_t)t * H * W;
      SharedBuffer& xn = normc_(cx, *x, rows, base, _norm_out_g);
      cx.release(*x);
      silu_(cx, xn, rows * (std::size_t)base);
      SharedBuffer& out = conv_chunk_(cx, _conv_out, xn, t, H, W, 1,
                                      next_carry());
      cx.release(xn);
      const std::size_t n = rows * 3;
      enc.set_function(_fn_clamp);
      enc.set_buffer(0, out); enc.set_buffer(1, out);
      enc.set_constant(2, (int)n);
      enc.set_constant(3, -1.0f); enc.set_constant(4, 1.0f);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
      rgb = &out;
    }
    if (!cx.alloc_ok) {
      return fail("a decode intermediate allocation failed (out of GPU "
                  "memory)");
    }
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      return fail(gpu_err.empty() ? std::string("GPU video decode failed")
                                  : gpu_err);
    }

    // Channel-last [t*H*W, 3] -> channel-first [3, t, H, W] for the sink.
    const std::size_t hw = (std::size_t)H * W;
    SharedBuffer frames = mc->make_shared_buffer((std::size_t)3 * t * hw * 2);
    if (frames.empty()) { return fail("frame buffer allocation failed"); }
    {
      const auto* s = static_cast<const _Float16*>(rgb->contents());
      auto* d = static_cast<_Float16*>(frames.contents());
      for (int ff = 0; ff < t; ++ff) {
        for (std::size_t p = 0; p < hw; ++p) {
          for (int c = 0; c < 3; ++c) {
            d[((std::size_t)c * t + ff) * hw + p] =
                s[((std::size_t)ff * hw + p) * 3 + c];
          }
        }
      }
    }
    const int frame0 = (f == 0) ? 0 : (1 + 4 * (f - 1));
    if (!on_frame(frames, frame0, t)) { return true; }   // sink stopped
  }
  return true;
}

SharedBuffer
MetalWanVae::decode_frame(const SharedBuffer& z, int h8, int w8,
                          std::string* err)
{
  SharedBuffer out;
  const bool ok = decode(z, 1, h8, w8,
                         [&](const SharedBuffer& rgb, int, int) {
                           out = _mc->make_shared_buffer(rgb.byte_size());
                           if (out.empty()) { return false; }
                           std::memcpy(out.contents(), rgb.contents(),
                                       rgb.byte_size());
                           return true;
                         },
                         err);
  if (!ok) { return {}; }
  return out;
}

// ---- encode ------------------------------------------------------------

SharedBuffer
MetalWanVae::encode(const SharedBuffer& video, int F, int H, int W,
                    std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  if (!_has_encoder && !ensure_encoder()) {
    return fail("the VAE encoder half is not loaded");
  }
  if (F <= 0 || (F - 1) % 4 != 0) {
    return fail("frame count must satisfy F % 4 == 1 (1, 5, ... 81, ...)");
  }
  if ((H % 8) != 0 || (W % 8) != 0) {
    return fail("height and width must be multiples of 8");
  }
  const std::size_t hw = (std::size_t)H * W;
  if (video.byte_size() < (std::size_t)3 * F * hw * 2) {
    return fail("input video smaller than [3, F, H, W]");
  }
  MetalCompute* mc = _mc;
  const int base = _cfg.base_dim;
  const int Cz = _cfg.z_dim;
  const int T = latent_frames(F);
  const int h8 = H / 8, w8 = W / 8;
  const std::size_t lhw = (std::size_t)h8 * w8;

  const std::size_t widest = (std::size_t)27 * base * _cfg.dim_mult[1];
  std::size_t col_cap = (std::size_t)W * widest * 64;
  {
    const MetalCompute::MemoryBudget mb = mc->memory_budget();
    if (mb.recommended != 0 && mb.headroom > 0) {
      col_cap = std::min(col_cap, mb.headroom / 4);
    }
  }
  if (_mma_max_m > 0) {
    col_cap = std::min(col_cap, (std::size_t)_mma_max_m / 2 * widest);
  }
  col_cap = std::max(col_cap, (std::size_t)W * widest);

  // The mean half of the posterior, accumulated across chunks.
  SharedBuffer moments =
      mc->make_shared_buffer((std::size_t)2 * Cz * T * lhw * 2);
  if (moments.empty()) { return fail("latent allocation failed"); }

  std::vector<Carry> carry;
  carry.resize(256);
  std::size_t ci = 0;
  auto next_carry = [&]() -> Carry* {
    if (ci >= carry.size()) { carry.resize(ci + 64); }
    return &carry[ci++];
  };

  const int n_chunks = 1 + (F - 1) / 4;
  int lat_out = 0;
  for (int k = 0; k < n_chunks; ++k) {
    ci = 0;
    const int f0 = (k == 0) ? 0 : (1 + 4 * (k - 1));
    int t = (k == 0) ? 1 : 4;
    CommandStream stream = mc->make_command_stream();
    Ctx cx;
    cx.use_pool = std::getenv("VPIPE_WAN_NO_VAE_POOL") == nullptr;
    cx.col = mc->make_shared_buffer(col_cap * 2);
    if (cx.col.empty()) { return fail("im2col band scratch allocation failed"); }
    cx.col_cap = col_cap;
    const SharedBuffer* zout = nullptr;
    int Hc = H, Wc = W;
    int t_final = 0;
    {
      ComputeEncoder enc = stream.begin_compute();
      cx.enc = &enc;
      // Channel-first [3, F, H, W] -> channel-last [t*hw, 3] for this chunk.
      SharedBuffer& x0 = cx.alloc(mc, (std::size_t)t * hw * 3);
      if (!cx.alloc_ok) { return fail("chunk allocation failed"); }
      {
        const auto* s = static_cast<const _Float16*>(video.contents());
        auto* d = static_cast<_Float16*>(x0.contents());
        for (int ff = 0; ff < t; ++ff) {
          for (int c = 0; c < 3; ++c) {
            const std::size_t src = ((std::size_t)c * F + (f0 + ff)) * hw;
            for (std::size_t p = 0; p < hw; ++p) {
              d[((std::size_t)ff * hw + p) * 3 + c] = s[src + p];
            }
          }
        }
      }
      const SharedBuffer* x =
          &conv_chunk_(cx, _enc_conv_in, x0, t, Hc, Wc, 1, next_carry());
      cx.release(x0);
      auto step = [&](SharedBuffer& nx) { cx.release(*x); x = &nx; };

      for (const DownStage& ds : _enc_down) {
        for (const ResBlock& rb : ds.resnets) {
          Carry* a = next_carry(); Carry* b = next_carry();
          step(resblock_(cx, rb, *x, t, Hc, Wc, a, b));
        }
        if (ds.down.present) {
          step(conv_chunk_(cx, ds.down.space, *x, t, Hc, Wc, 2, nullptr));
          Hc /= 2; Wc /= 2;
          if (ds.down.temporal) {
            SharedBuffer& d = time_down_(cx, ds.down, *x, t,
                                         (std::size_t)Hc * Wc,
                                         ds.down.space.cout, next_carry());
            if (&d != x) { step(d); }
          }
        }
        if (!cx.alloc_ok) { return fail("chunk allocation failed"); }
      }
      {
        Carry* a = next_carry(); Carry* b = next_carry();
        step(resblock_(cx, _enc_mid_res0, *x, t, Hc, Wc, a, b));
      }
      step(attention_(cx, _enc_mid_attn, *x, t, Hc, Wc));
      {
        Carry* a = next_carry(); Carry* b = next_carry();
        step(resblock_(cx, _enc_mid_res1, *x, t, Hc, Wc, a, b));
      }
      const std::size_t rows = (std::size_t)t * Hc * Wc;
      SharedBuffer& xn = normc_(cx, *x, rows, _enc_conv_out.cin,
                                _enc_norm_out_g);
      cx.release(*x);
      silu_(cx, xn, rows * (std::size_t)_enc_conv_out.cin);
      SharedBuffer& h = conv_chunk_(cx, _enc_conv_out, xn, t, Hc, Wc, 1,
                                    next_carry());
      cx.release(xn);
      // quant_conv (1x1) over the chunk's latent frames.
      SharedBuffer& q = cx.alloc(mc, rows * (std::size_t)(2 * Cz));
      gemm_bias_(cx, h, _quant_conv.w, _quant_conv.b, q, (int)rows, 2 * Cz,
                 2 * Cz);
      cx.release(h);
      zout = &q;
      t_final = t;
    }
    if (!cx.alloc_ok) { return fail("an encode allocation failed"); }
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      return fail(gpu_err.empty() ? std::string("GPU video encode failed")
                                  : gpu_err);
    }
    // Channel-last [t*lhw, 2*Cz] -> channel-first [2*Cz, T, h8, w8] at the
    // chunk's latent frame offset.
    {
      const auto* s = static_cast<const _Float16*>(zout->contents());
      auto* d = static_cast<_Float16*>(moments.contents());
      for (int ff = 0; ff < t_final && lat_out + ff < T; ++ff) {
        for (int c = 0; c < 2 * Cz; ++c) {
          const std::size_t dst =
              ((std::size_t)c * T + (lat_out + ff)) * lhw;
          for (std::size_t p = 0; p < lhw; ++p) {
            d[dst + p] = s[((std::size_t)ff * lhw + p) * (2 * Cz) + c];
          }
        }
      }
    }
    lat_out += t_final;
  }

  // The posterior MODE (mean) is the first z_dim channels; whiten it.
  SharedBuffer out = mc->make_shared_buffer((std::size_t)Cz * T * lhw * 2);
  if (out.empty()) { return fail("latent allocation failed"); }
  {
    const auto* s = static_cast<const _Float16*>(moments.contents());
    auto* d = static_cast<_Float16*>(out.contents());
    const bool whiten = (int)_cfg.latents_mean.size() == Cz &&
                        (int)_cfg.latents_std.size() == Cz;
    for (int c = 0; c < Cz; ++c) {
      const float mu = whiten ? _cfg.latents_mean[(std::size_t)c] : 0.0f;
      const float sd = whiten ? _cfg.latents_std[(std::size_t)c] : 1.0f;
      for (std::size_t p = 0; p < (std::size_t)T * lhw; ++p) {
        const std::size_t i = (std::size_t)c * T * lhw + p;
        d[i] = (_Float16)(((float)s[i] - mu) / (sd != 0.0f ? sd : 1.0f));
      }
    }
  }
  return out;
}

SharedBuffer
MetalWanVae::unwhiten(const SharedBuffer& z, int T, int h8, int w8)
{
  const int C = _cfg.z_dim;
  const std::size_t n = (std::size_t)T * h8 * w8;
  if ((int)_cfg.latents_mean.size() != C ||
      (int)_cfg.latents_std.size() != C ||
      z.byte_size() < (std::size_t)C * n * 2) {
    return {};
  }
  SharedBuffer out = _mc->make_shared_buffer((std::size_t)C * n * 2);
  if (out.empty()) { return {}; }
  const auto* s = static_cast<const _Float16*>(z.contents());
  auto* d = static_cast<_Float16*>(out.contents());
  for (int c = 0; c < C; ++c) {
    const float mu = _cfg.latents_mean[(std::size_t)c];
    const float sd = _cfg.latents_std[(std::size_t)c];
    for (std::size_t p = 0; p < n; ++p) {
      const std::size_t i = (std::size_t)c * n + p;
      d[i] = (_Float16)((float)s[i] * sd + mu);
    }
  }
  return out;
}

}  // namespace genai
}  // namespace vpipe
