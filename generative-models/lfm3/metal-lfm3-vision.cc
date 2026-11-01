#include "generative-models/lfm3/metal-lfm3-vision.h"

#include "generative-models/lfm3/lfm3-weights.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/image-ops.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-bag.h"
#include "common/flex-data.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {

// Mirrors the vendored MLX steel `attention` AttnParams (attn_steel.metal).
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

bool
read_json_(const std::filesystem::path& p, FlexData* out)
{
  std::ifstream in(p);
  if (!in) { return false; }
  std::stringstream ss;
  ss << in.rdbuf();
  try {
    *out = FlexData::from_json(ss.str());
  } catch (const std::exception&) {
    return false;
  }
  return out->is_object();
}

// Pillow's 8-bit BICUBIC resize of planar u8 [3,sh,sw] -> [3,dh,dw]: a
// horizontal pass to an 8-bit intermediate, then a vertical one, each
// rounded and clipped (Pillow's 8bpc path; torchvision's antialiased
// uint8 resize follows it).
void
resize_bicubic_u8_(const std::uint8_t* src, int sh, int sw,
                   std::uint8_t* dst, int dh, int dw)
{
  std::vector<int> bx, by;
  std::vector<float> wx, wy;
  const int kx = metal_compute::build_cubic_coeffs(
      sw, 0.0, dw, (double)sw / (double)dw, bx, wx);
  const int ky = metal_compute::build_cubic_coeffs(
      sh, 0.0, dh, (double)sh / (double)dh, by, wy);
  auto clip8 = [](float v) {
    return (std::uint8_t)std::min(255.0f, std::max(0.0f, std::round(v)));
  };
  std::vector<std::uint8_t> tmp((std::size_t)sh * dw);
  for (int c = 0; c < 3; ++c) {
    const std::uint8_t* s = src + (std::size_t)c * sh * sw;
    for (int y = 0; y < sh; ++y) {
      for (int x = 0; x < dw; ++x) {
        const float* w = &wx[(std::size_t)x * kx];
        const int x0 = bx[(std::size_t)x];
        float acc = 0.0f;
        for (int t = 0; t < kx && x0 + t < sw; ++t) {
          acc += w[t] * (float)s[(std::size_t)y * sw + x0 + t];
        }
        tmp[(std::size_t)y * dw + x] = clip8(acc);
      }
    }
    std::uint8_t* d = dst + (std::size_t)c * dh * dw;
    for (int y = 0; y < dh; ++y) {
      const float* w = &wy[(std::size_t)y * ky];
      const int y0 = by[(std::size_t)y];
      for (int x = 0; x < dw; ++x) {
        float acc = 0.0f;
        for (int t = 0; t < ky && y0 + t < sh; ++t) {
          acc += w[t] * (float)tmp[(std::size_t)(y0 + t) * dw + x];
        }
        d[(std::size_t)y * dw + x] = clip8(acc);
      }
    }
  }
}

}  // namespace

bool
MetalLfm3VisionEncoder::Config::from_dir(const std::string& dir,
                                         Config* out, std::string* err)
{
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = "LFM3-VL vision config: " + std::move(m); }
    return false;
  };
  FlexData root;
  if (!read_json_(std::filesystem::path(dir) / "config.json", &root)) {
    return fail("no readable config.json under " + dir);
  }
  const FlexData vc = bag::value(root, "vision_config");
  if (!vc.is_object()) { return fail("no vision_config"); }
  Config c;
  c.layers = (int)bag::integer(vc, "num_hidden_layers", 0);
  c.hidden = (int)bag::integer(vc, "hidden_size", 0);
  c.n_heads = (int)bag::integer(vc, "num_attention_heads", 0);
  c.n_kv_heads = (int)bag::integer(vc, "num_key_value_heads", c.n_heads);
  c.head_dim = c.n_heads > 0 ? c.hidden / c.n_heads : 0;
  c.intermediate = (int)bag::integer(vc, "intermediate_size", 0);
  c.patch = (int)bag::integer(vc, "patch_size", 16);
  c.merge = (int)bag::integer(vc, "spatial_merge_size", 2);
  c.temporal = (int)bag::integer(vc, "temporal_patch_size", 2);
  c.conv_k = (int)bag::integer(vc, "conv_kernel_size", 3);
  c.eps = (float)bag::real(vc, "norm_eps", 1e-5);
  const FlexData rp = bag::value(vc, "rope_parameters");
  if (bag::text(rp, "rope_type", "axial") != "axial") {
    return fail("rope type '" + std::string(bag::text(rp, "rope_type")) +
                "' is not supported");
  }
  c.rope_theta = (float)bag::real(rp, "rope_theta", 1e6);
  const FlexData lt = bag::value(vc, "layer_types");
  if (!lt.is_array() || (int)lt.as_array().size() != c.layers) {
    return fail("layer_types does not list every layer");
  }
  for (FlexData e : lt.as_array()) {
    const std::string t(e.as_string(""));
    if (t != "conv" && t != "full_attention") {
      return fail("layer type '" + t + "' is not supported");
    }
    c.attn.push_back(t == "full_attention");
  }
  c.proj_hidden = (int)bag::integer(root, "projector_hidden_size", 0);
  c.proj_eps = (float)bag::real(root, "projector_layer_norm_eps", 1e-6);
  c.out_hidden = (int)bag::integer(bag::value(root, "text_config"),
                                   "hidden_size", 0);
  if ((int)bag::integer(root, "downsample_factor", c.merge) != c.merge) {
    return fail("downsample_factor differs from spatial_merge_size");
  }
  if (bag::text(root, "projector_hidden_act", "gelu") != "gelu"
      || !bag::flag(root, "projector_bias", true)) {
    return fail("only the projector's biased GELU form is supported");
  }
  if (c.layers <= 0 || c.hidden <= 0 || c.head_dim != 64
      || c.n_heads % std::max(c.n_kv_heads, 1) != 0 || c.intermediate <= 0
      || c.proj_hidden <= 0 || c.out_hidden <= 0 || c.conv_k != 3
      || c.merge != 2) {
    return fail("missing or unsupported sizes (head_dim 64, a 3x3 conv "
                "and a 2x2 merge are what this port runs)");
  }
  FlexData pc;
  if (read_json_(std::filesystem::path(dir) / "processor_config.json",
                 &pc)) {
    const FlexData ip = bag::value(pc, "image_processor");
    const FlexData size = bag::value(ip, "size");
    if (bag::has(size, "shortest_edge")) {
      c.min_pixels = bag::integer(size, "shortest_edge");
    }
    if (bag::has(size, "longest_edge")) {
      c.max_pixels = bag::integer(size, "longest_edge");
    }
    auto triple = [&](std::string_view k, float (&dst)[3]) {
      const FlexData a = bag::value(ip, k);
      if (!a.is_array() || a.as_array().size() != 3) { return; }
      for (int i = 0; i < 3; ++i) {
        dst[i] = (float)a.as_array()[(std::size_t)i].as_real(dst[i]);
      }
    };
    triple("image_mean", c.mean);
    triple("image_std", c.stdv);
  }
  *out = std::move(c);
  return true;
}

std::size_t
MetalLfm3VisionEncoder::weight_bytes(const std::string& model_dir)
{
  Config c;
  if (!Config::from_dir(model_dir, &c, nullptr)) { return 0; }
  const std::size_t C = (std::size_t)c.hidden, I = (std::size_t)c.intermediate;
  const std::size_t kvd = (std::size_t)c.n_kv_heads * c.head_dim;
  std::size_t e = C * 3 * c.temporal * c.patch * c.patch + C;
  for (bool a : c.attn) {
    e += 2 * C + 3 * C * I;
    e += a ? (C + 2 * kvd) * C + C * C + 2 * (std::size_t)c.head_dim
           : 3 * C * C + C * 9 + C * C;
  }
  const std::size_t S2 = (std::size_t)c.merge * c.merge;
  e += 2 * C + S2 * C * c.proj_hidden + c.proj_hidden +
       (std::size_t)c.proj_hidden * c.out_hidden + c.out_hidden;
  return e * 2;
}

std::unique_ptr<MetalLfm3VisionEncoder>
MetalLfm3VisionEncoder::load(std::shared_ptr<WeightSet> ws,
                             metal_compute::MetalCompute* mc,
                             const Config& cfg, std::string* err)
{
  auto fail = [&](std::string m) -> std::unique_ptr<MetalLfm3VisionEncoder> {
    if (err != nullptr) { *err = "MetalLfm3VisionEncoder: " + std::move(m); }
    return nullptr;
  };
  if (mc == nullptr || !mc->valid() || ws == nullptr) {
    return fail("no GPU or no weights");
  }
  auto m = std::unique_ptr<MetalLfm3VisionEncoder>(
      new MetalLfm3VisionEncoder());
  m->_cfg = cfg;
  m->_mc = mc;
  m->_ws = std::move(ws);
  m->_lib_dense = mc->load_library("dense_gemm");
  m->_lib_rms = mc->load_library("rms_norm");
  m->_lib_rope = mc->load_library("rope");
  m->_lib_elt = mc->load_library("llm_elementwise");
  m->_lib_vis = mc->load_library("qwen3_5_vision");
  m->_lib_attn = mc->load_library("attn_steel");
  m->_lib_lfm3 = mc->load_library("lfm3");
  m->_fn_gemm = m->_lib_dense.function("dense_gemm_t_bm64bn64_f16");
  m->_fn_rms = m->_lib_rms.function("rms_norm_f16");
  m->_fn_norm_rope = m->_lib_rope.function("rms_rope_half_ftab_tm_f16");
  m->_fn_heads = m->_lib_rope.function("transpose_rope_half_part_ftab_f16");
  m->_fn_tokens = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_swiglu = m->_lib_elt.function("swiglu_split_gate_first_f16");
  m->_fn_bias = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_conv = m->_lib_lfm3.function("lfm3_vis_conv3x3_f16");
  m->_fn_ln = m->_lib_vis.function("layer_norm_bias_f16");
  m->_fn_gelu = m->_lib_vis.function("gelu_erf_f16");
  for (const auto* f :
       {&m->_fn_gemm, &m->_fn_rms, &m->_fn_norm_rope, &m->_fn_heads,
        &m->_fn_tokens, &m->_fn_residual, &m->_fn_swiglu, &m->_fn_bias,
        &m->_fn_conv, &m->_fn_ln, &m->_fn_gelu}) {
    if (!f->valid()) { return fail("a kernel did not load"); }
  }
  if (!m->_lib_attn.valid()) { return fail("no steel attention"); }
  m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));
  if (mc->supports_matrix_cores()) {
    if (std::getenv("VPIPE_LFM3_NO_MMA2") == nullptr) {
      m->_lib_mma = mc->load_library("dense_gemm_mma");
      m->_fn_mma = m->_lib_mma.function("dense_gemm_mma_t_n128_f16");
      m->_fn_mma_deep =
          m->_lib_mma.function("dense_gemm_mma_t_n128x256_f16");
      m->_use_mma2 = m->_fn_mma.valid() && m->_fn_mma_deep.valid();
    }
    if (std::getenv("VPIPE_LFM3_NO_NAX_ATTN") == nullptr) {
      m->_lib_attn_nax = mc->load_library("attn_steel_nax");
      m->_use_attn_nax = m->_lib_attn_nax.valid();
    }
  }

  Lfm3Weights w(m->_ws.get(), mc, "lfm3-vision/f16/");
  const std::string vt = "model.vision_tower.";
  const std::string pj = "model.multi_modal_projector.";
  m->_patch_w = w.f16(vt + "patchifier.weight");
  m->_embed_norm = w.f16(vt + "embedding_norm.weight");
  const std::size_t feat =
      (std::size_t)3 * cfg.temporal * cfg.patch * cfg.patch;
  if (m->_patch_w.byte_size() != (std::size_t)cfg.hidden * feat * 2
      || m->_embed_norm.empty()) {
    return fail("no patchifier / embedding_norm of the expected shape");
  }
  m->_layers.resize((std::size_t)cfg.layers);
  for (int L = 0; L < cfg.layers; ++L) {
    const std::string p = vt + "layers." + std::to_string(L) + ".";
    Layer& ly = m->_layers[(std::size_t)L];
    ly.attn = cfg.attn[(std::size_t)L];
    ly.op_norm = w.f16(p + "operator_norm.weight");
    ly.ffn_norm = w.f16(p + "ffn_norm.weight");
    ly.w13 = w.fused({p + "feed_forward.w1.weight",
                      p + "feed_forward.w3.weight"});
    ly.w2 = w.f16(p + "feed_forward.w2.weight");
    bool ok = !ly.op_norm.empty() && !ly.ffn_norm.empty()
        && !ly.w13.empty() && !ly.w2.empty();
    if (ly.attn) {
      const std::string a = p + "self_attn.";
      ly.qkv_w = w.fused({a + "q_proj.weight", a + "k_proj.weight",
                          a + "v_proj.weight"});
      ly.o_w = w.f16(a + "out_proj.weight");
      ly.q_norm = w.f16(a + "q_layernorm.weight");
      ly.k_norm = w.f16(a + "k_layernorm.weight");
      ok = ok && !ly.qkv_w.empty() && !ly.o_w.empty()
          && !ly.q_norm.empty() && !ly.k_norm.empty();
    } else {
      ly.in_w = w.f16(p + "conv.in_proj.weight");
      ly.conv_w = w.f16(p + "conv.spatial_conv.weight");   // [C,1,3,3]
      ly.out_w = w.f16(p + "conv.out_proj.weight");
      ok = ok && !ly.in_w.empty() && !ly.out_w.empty()
          && ly.conv_w.byte_size() == (std::size_t)cfg.hidden * 9 * 2;
    }
    if (!ok) {
      return fail("layer " + std::to_string(L) + ": a weight is missing");
    }
  }
  m->_ln_w = w.f16(pj + "layer_norm.weight");
  m->_ln_b = w.f16(pj + "layer_norm.bias");
  m->_l1_w = w.f16(pj + "linear_1.weight");
  m->_l1_b = w.f16(pj + "linear_1.bias");
  m->_l2_w = w.f16(pj + "linear_2.weight");
  m->_l2_b = w.f16(pj + "linear_2.bias");
  for (const auto* b : {&m->_ln_w, &m->_ln_b, &m->_l1_w, &m->_l1_b,
                        &m->_l2_w, &m->_l2_b}) {
    if (b->empty()) { return fail("a projector weight is missing"); }
  }
  return m;
}

void
MetalLfm3VisionEncoder::smart_resize(int H, int W, int factor,
                                     std::int64_t lo, std::int64_t hi,
                                     int* th, int* tw)
{
  // The reference's smart_resize, Python's round (half to even) included.
  auto pyround = [](double v) { return std::nearbyint(v); };
  int hb = (int)pyround((double)H / factor) * factor;
  int wb = (int)pyround((double)W / factor) * factor;
  if ((std::int64_t)hb * wb > hi) {
    const double beta = std::sqrt(((double)H * W) / (double)hi);
    hb = std::max(factor, (int)std::floor(H / beta / factor) * factor);
    wb = std::max(factor, (int)std::floor(W / beta / factor) * factor);
  } else if ((std::int64_t)hb * wb < lo) {
    const double beta = std::sqrt((double)lo / ((double)H * W));
    hb = (int)std::ceil(H * beta / factor) * factor;
    wb = (int)std::ceil(W * beta / factor) * factor;
  }
  *th = hb;
  *tw = wb;
}

void
MetalLfm3VisionEncoder::target_size(int H, int W, int* th, int* tw,
                                    std::int64_t max_pixels) const
{
  const std::int64_t hi =
      max_pixels > 0 ? std::min(max_pixels, _cfg.max_pixels)
                     : _cfg.max_pixels;
  smart_resize(H, W, _cfg.patch * _cfg.merge,
               std::min(_cfg.min_pixels, hi), hi, th, tw);
}

void
MetalLfm3VisionEncoder::preprocess(const std::uint8_t* rgb, int H, int W,
                                   std::int64_t max_pixels,
                                   std::vector<float>* px, int* th,
                                   int* tw) const
{
  target_size(H, W, th, tw, max_pixels);
  std::vector<std::uint8_t> rs;
  const std::uint8_t* img = rgb;
  if (*th != H || *tw != W) {
    rs.resize((std::size_t)3 * *th * *tw);
    resize_bicubic_u8_(rgb, H, W, rs.data(), *th, *tw);
    img = rs.data();
  }
  const std::size_t plane = (std::size_t)*th * *tw;
  px->resize(3 * plane);
  const float rescale = 1.0f / 255.0f;
  for (int c = 0; c < 3; ++c) {
    const float mean = _cfg.mean[c], sd = _cfg.stdv[c];
    for (std::size_t i = 0; i < plane; ++i) {
      (*px)[(std::size_t)c * plane + i] =
          ((float)img[(std::size_t)c * plane + i] * rescale - mean) / sd;
    }
  }
}

MetalLfm3VisionEncoder::Result
MetalLfm3VisionEncoder::encode(const std::uint8_t* rgb, int H, int W,
                               std::int64_t max_pixels)
{
  if (rgb == nullptr || H <= 0 || W <= 0) { return {}; }
  std::vector<float> px;
  int th = 0, tw = 0;
  preprocess(rgb, H, W, max_pixels, &px, &th, &tw);
  return encode_normalized(px.data(), th, tw);
}

std::size_t
MetalLfm3VisionEncoder::encode_scratch_bytes(const Config& c,
                                             std::int64_t pixels)
{
  // smart_resize keeps the area under the bound, so a picture of at
  // most `px` pixels is at most px / P^2 patches.
  const std::int64_t px =
      pixels > 0 ? std::min(pixels, c.max_pixels) : c.max_pixels;
  const std::size_t n = (std::size_t)std::max<std::int64_t>(px, 0) /
                        ((std::size_t)c.patch * c.patch);
  const std::size_t C = (std::size_t)c.hidden;
  const std::size_t feat = (std::size_t)3 * c.temporal * c.patch * c.patch;
  const std::size_t D = (std::size_t)c.head_dim;
  // feat, x, hn, o, the fused projections (max 3C, 2I), the
  // feed-forward's act, four head-major tensors, the rope tables, the
  // projector's two planes
  std::size_t e = n * (feat + 3 * C +
                       std::max(3 * C, 2 * (std::size_t)c.intermediate) +
                       (std::size_t)c.intermediate + 4 * C) * 2;
  e += n * (D / 2) * 2 * 4;
  const std::size_t tok = n / ((std::size_t)c.merge * c.merge);
  e += tok * ((std::size_t)c.proj_hidden + (std::size_t)c.out_hidden) * 2 *
       2;
  return e;
}

MetalLfm3VisionEncoder::Result
MetalLfm3VisionEncoder::encode_normalized(const float* px, int h, int w,
                                          std::vector<float>* vit_out,
                                          std::vector<float>* proj_out)
{
  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmVision,
                     kPerfLlmVisionBegin, 1);
  const Config& c = _cfg;
  const int P = c.patch, M = c.merge, S2 = M * M, T = c.temporal;
  const int C = c.hidden, Hq = c.n_heads, Hkv = c.n_kv_heads,
            D = c.head_dim, I = c.intermediate;
  const int unit = P * M;
  Result res;
  if (px == nullptr || h <= 0 || w <= 0 || h % unit != 0 || w % unit != 0) {
    return res;
  }
  const int gh = h / P, gw = w / P;     // patch grid
  const int n = gh * gw;
  const int ntok = n / S2;
  const int feat_dim = 3 * T * P * P;
  const int qkv = (Hq + 2 * Hkv) * D;

  // Patchify in MERGE order: row r = ((I*bw + J)*M + ki)*M + kj is patch
  // (I*M + ki, J*M + kj); features (c, t, y, x), every temporal copy the
  // same frame. RoPE (axial, f32 tables [n, D/2]): of the head's D/2
  // pairs, pair k < D/4 turns at row * f_k, the next D/4 at col * f_k,
  // f_k = theta^(-2k / (D/2)).
  const std::size_t plane = (std::size_t)h * w;
  std::vector<_Float16> feat((std::size_t)n * feat_dim);
  std::vector<float> cosb((std::size_t)n * (D / 2)),
      sinb((std::size_t)n * (D / 2));
  const int sd = D / 2;                 // per-axis rotary dims
  std::vector<float> freq((std::size_t)sd / 2);
  for (int k = 0; k < sd / 2; ++k) {
    freq[(std::size_t)k] =
        1.0f / std::pow(c.rope_theta, (float)(2 * k) / (float)sd);
  }
  const int bw = gw / M;
  for (int r = 0; r < n; ++r) {
    const int blk = r / S2, k = r % S2;
    const int pi = (blk / bw) * M + k / M, pj = (blk % bw) * M + k % M;
    _Float16* fp = feat.data() + (std::size_t)r * feat_dim;
    int o = 0;
    for (int ch = 0; ch < 3; ++ch) {
      for (int t = 0; t < T; ++t) {
        for (int y = 0; y < P; ++y) {
          const std::size_t row = (std::size_t)(pi * P + y) * w;
          for (int x = 0; x < P; ++x) {
            fp[o++] = (_Float16)px[(std::size_t)ch * plane + row +
                                   (std::size_t)(pj * P + x)];
          }
        }
      }
    }
    float* cr = cosb.data() + (std::size_t)r * (D / 2);
    float* sr = sinb.data() + (std::size_t)r * (D / 2);
    for (int q = 0; q < D / 2; ++q) {
      const float a = q < sd / 2 ? (float)pi * freq[(std::size_t)q]
                                 : (float)pj * freq[(std::size_t)(q - sd / 2)];
      cr[q] = std::cos(a);
      sr[q] = std::sin(a);
    }
  }

  auto buf = [&](std::size_t bytes) {
    return _mc->make_shared_buffer(std::max<std::size_t>(bytes, 16));
  };
  SharedBuffer featb = buf(feat.size() * 2);
  SharedBuffer cosbuf = buf(cosb.size() * 4), sinbuf = buf(sinb.size() * 4);
  std::memcpy(featb.contents(), feat.data(), feat.size() * 2);
  std::memcpy(cosbuf.contents(), cosb.data(), cosb.size() * 4);
  std::memcpy(sinbuf.contents(), sinb.data(), sinb.size() * 4);
  const std::size_t nC = (std::size_t)n * C;
  SharedBuffer x = buf(nC * 2), hn = buf(nC * 2), o = buf(nC * 2);
  SharedBuffer big = buf((std::size_t)n * std::max(3 * C, 2 * I) * 2);
  SharedBuffer act = buf((std::size_t)n * I * 2);
  SharedBuffer qt = buf((std::size_t)Hq * n * D * 2);
  SharedBuffer kt = buf((std::size_t)Hkv * n * D * 2);
  SharedBuffer vtb = buf((std::size_t)Hkv * n * D * 2);
  SharedBuffer at = buf((std::size_t)Hq * n * D * 2);
  SharedBuffer h1 = buf((std::size_t)ntok * c.proj_hidden * 2);
  SharedBuffer h2 = buf((std::size_t)ntok * c.out_hidden * 2);

  // Bidirectional attention over the frame's patches: the vendored MLX
  // steel flash kernel (NAX on M5), GQA through its factor.
  const bool nax = _use_attn_nax;
  const int a_bq = nax ? 64 : 32, a_bk = nax ? 32 : 16;
  {
    auto* ap = static_cast<SteelAttnParams*>(_attn_params.contents());
    ap->B = 1; ap->H = Hq; ap->D = D; ap->qL = n; ap->kL = n;
    ap->gqa_factor = Hq / Hkv;
    ap->scale = 1.0f / std::sqrt((float)D);
    ap->NQ = (n + a_bq - 1) / a_bq; ap->NK = (n + a_bk - 1) / a_bk;
    ap->NQ_aligned = n / a_bq; ap->NK_aligned = n / a_bk;
    ap->qL_rem = n - ap->NQ_aligned * a_bq;
    ap->kL_rem = n - ap->NK_aligned * a_bk;
    ap->qL_off = 0;
    ap->Q_strides[0] = (std::int64_t)Hq * n * D;
    ap->Q_strides[1] = (std::int64_t)n * D; ap->Q_strides[2] = D;
    ap->K_strides[0] = (std::int64_t)Hkv * n * D;
    ap->K_strides[1] = (std::int64_t)n * D; ap->K_strides[2] = D;
    for (int i = 0; i < 3; ++i) {
      ap->V_strides[i] = ap->K_strides[i];
      ap->O_strides[i] = ap->Q_strides[i];
    }
  }
  metal_compute::FunctionConstants fc;
  fc.set_bool(200, (n % a_bq) == 0).set_bool(201, (n % a_bk) == 0)
      .set_bool(300, false).set_bool(301, false).set_bool(302, false);
  const metal_compute::ComputeFunction fn_attn = nax
      ? _lib_attn_nax.function("attn_steel_nax_h_bd64", fc)
      : _lib_attn.function("attn_steel_h_bd64", fc);
  if (!fn_attn.valid()) { return res; }

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm = [&](const SharedBuffer& xin, const SharedBuffer& wt,
                    const SharedBuffer& y, int Mr, int N, int K) {
      if (_use_mma2) {
        const bool deep = K >= 6144;
        const int BN = deep ? 256 : 128;
        enc.set_function(deep ? _fn_mma_deep : _fn_mma);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, wt);
        enc.set_buffer(2, wt);          // bias slot unused (has_bias 0)
        enc.set_buffer(3, y);
        enc.set_constant(4, K);
        enc.set_constant(5, N);
        enc.set_constant(6, Mr);
        enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                      (unsigned)((Mr + 127) / 128), 1}, {256, 1, 1});
        return;
      }
      enc.set_function(_fn_gemm);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, wt);
      enc.set_buffer(2, wt);
      enc.set_buffer(3, y);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, Mr);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + 63) / 64) * 32),
                    (unsigned)(((Mr + 63) / 64) * 2), 2}, {32, 2, 2});
    };
    auto bias = [&](const SharedBuffer& y, const SharedBuffer& b, int Mr,
                    int N) {
      const std::uint32_t total = (std::uint32_t)Mr * (std::uint32_t)N;
      enc.set_function(_fn_bias);
      enc.set_buffer(0, y);
      enc.set_buffer(1, b);
      enc.set_constant(2, N);
      enc.set_constant(3, total);
      enc.dispatch({(unsigned)N, (unsigned)Mr, 1}, {256, 1, 1});
    };
    auto rms = [&](const SharedBuffer& xin, const SharedBuffer& wt,
                   const SharedBuffer& y) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, wt);
      enc.set_buffer(2, y);
      enc.set_constant(3, C);
      enc.set_constant(4, c.eps);
      enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    };
    auto residual = [&]() {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, x);
      enc.set_buffer(1, o);
      enc.set_buffer(2, x);
      enc.set_constant(3, (int)nC);
      enc.dispatch({(unsigned)nC, 1, 1}, {256, 1, 1});
    };
    auto norm_rope = [&](int col, int heads, const SharedBuffer& g) {
      enc.set_function(_fn_norm_rope);
      enc.set_buffer(0, big, (std::size_t)col * 2);
      enc.set_buffer(1, cosbuf);
      enc.set_buffer(2, sinbuf);
      enc.set_buffer(3, g);
      enc.set_constant(4, heads);
      enc.set_constant(5, n);
      enc.set_constant(6, D);
      enc.set_constant(7, qkv);
      enc.set_constant(8, c.eps);
      enc.dispatch({32, (unsigned)n, (unsigned)heads}, {32, 1, 1});
    };
    auto heads = [&](int col, int nh, const SharedBuffer& out) {
      const int rot = 0;
      enc.set_function(_fn_heads);
      enc.set_buffer(0, big);
      enc.set_buffer(1, out);
      enc.set_buffer(2, cosbuf);
      enc.set_buffer(3, sinbuf);
      enc.set_constant(4, nh);
      enc.set_constant(5, n);
      enc.set_constant(6, D);
      enc.set_constant(7, rot);
      enc.set_constant(8, qkv);
      enc.set_constant(9, col);
      enc.set_constant(10, D);
      enc.dispatch({(unsigned)D, (unsigned)n, (unsigned)nh},
                   {(unsigned)D, 1, 1});
    };

    gemm(featb, _patch_w, o, n, C, feat_dim);
    rms(o, _embed_norm, x);
    for (int L = 0; L < c.layers; ++L) {
      const Layer& ly = _layers[(std::size_t)L];
      rms(x, ly.op_norm, hn);
      if (ly.attn) {
        gemm(hn, ly.qkv_w, big, n, qkv, C);
        norm_rope(0, Hq, ly.q_norm);
        norm_rope(Hq * D, Hkv, ly.k_norm);
        heads(0, Hq, qt);
        heads(Hq * D, Hkv, kt);
        heads((Hq + Hkv) * D, Hkv, vtb);
        enc.set_function(fn_attn);
        enc.set_buffer(0, qt);
        enc.set_buffer(1, kt);
        enc.set_buffer(2, vtb);
        enc.set_buffer(3, at);
        enc.set_buffer(4, _attn_params);
        enc.dispatch({32 * (unsigned)((n + a_bq - 1) / a_bq),
                      4 * (unsigned)Hq, 1}, {32, 4, 1});
        // [Hq, n, D] -> [n, Hq, D]
        enc.set_function(_fn_tokens);
        enc.set_buffer(0, at);
        enc.set_buffer(1, hn);
        enc.set_constant(2, Hq);
        enc.set_constant(3, n);
        enc.set_constant(4, D);
        enc.dispatch({(unsigned)D, (unsigned)n, (unsigned)Hq},
                     {(unsigned)D, 1, 1});
        gemm(hn, ly.o_w, o, n, C, C);
      } else {
        gemm(hn, ly.in_w, big, n, 3 * C, C);
        enc.set_function(_fn_conv);
        enc.set_buffer(0, big);
        enc.set_buffer(1, ly.conv_w);
        enc.set_buffer(2, hn);
        enc.set_constant(3, C);
        enc.set_constant(4, gh);
        enc.set_constant(5, gw);
        enc.set_constant(6, M);
        enc.set_constant(7, 3 * C);
        enc.dispatch({(unsigned)C, (unsigned)n, 1}, {256, 1, 1});
        gemm(hn, ly.out_w, o, n, C, C);
      }
      residual();
      rms(x, ly.ffn_norm, hn);
      gemm(hn, ly.w13, big, n, 2 * I, C);
      enc.set_function(_fn_swiglu);
      enc.set_buffer(0, big);
      enc.set_buffer(1, act);
      enc.set_constant(2, n);
      enc.set_constant(3, I);
      enc.dispatch({(unsigned)(n * I), 1, 1}, {256, 1, 1});
      gemm(act, ly.w2, o, n, C, I);
      residual();
    }
    // Projector: LayerNorm per patch; in merge order a block's S2
    // patches are adjacent rows, so [n, C] IS [ntok, S2*C].
    enc.set_function(_fn_ln);
    enc.set_buffer(0, x);
    enc.set_buffer(1, _ln_w);
    enc.set_buffer(2, _ln_b);
    enc.set_buffer(3, hn);
    enc.set_constant(4, C);
    enc.set_constant(5, c.proj_eps);
    enc.dispatch({256, (unsigned)n, 1}, {256, 1, 1});
    gemm(hn, _l1_w, h1, ntok, c.proj_hidden, S2 * C);
    bias(h1, _l1_b, ntok, c.proj_hidden);
    enc.set_function(_fn_gelu);
    enc.set_buffer(0, h1);
    enc.set_buffer(1, h1);
    enc.set_constant(2, ntok * c.proj_hidden);
    enc.dispatch({(unsigned)(ntok * c.proj_hidden), 1, 1}, {256, 1, 1});
    gemm(h1, _l2_w, h2, ntok, c.out_hidden, c.proj_hidden);
    bias(h2, _l2_b, ntok, c.out_hidden);
  }
  std::string why;
  if (!stream.commit().wait_ok(&why)) {
    if (_session) {
      _session->warn(fmt("MetalLfm3VisionEncoder: the encode failed on "
                         "the GPU ({})", why));
    }
    return res;
  }
  res.embeddings = std::move(h2);
  res.n_tokens = ntok;
  res.out_hidden = c.out_hidden;
  res.grid_h = gh / M;
  res.grid_w = gw / M;
  auto to_f32 = [](const SharedBuffer& b, std::size_t count,
                   std::vector<float>* out) {
    const auto* s = static_cast<const _Float16*>(b.contents());
    out->resize(count);
    for (std::size_t i = 0; i < count; ++i) { (*out)[i] = (float)s[i]; }
  };
  if (vit_out != nullptr) { to_f32(x, nC, vit_out); }
  if (proj_out != nullptr) {
    to_f32(res.embeddings, (std::size_t)ntok * c.out_hidden, proj_out);
  }
  return res;
}

}  // namespace vpipe::genai
