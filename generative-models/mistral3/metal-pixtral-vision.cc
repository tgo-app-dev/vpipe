#include "generative-models/mistral3/metal-pixtral-vision.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"
#include "generative-models/weight-set.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/image-ops.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-bag.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

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

std::size_t
numel_(const std::vector<std::int64_t>& s)
{
  std::size_t n = 1;
  for (auto d : s) { n *= (std::size_t)d; }
  return n;
}

inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t bits = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// One element of a BF16 / F16 / F32 tensor as f32.
inline float
elem_f32_(const void* p, const std::string& dtype, std::size_t i)
{
  if (dtype == "BF16") {
    return bf16_to_f32_(static_cast<const std::uint16_t*>(p)[i]);
  }
  if (dtype == "F16") {
    return (float)static_cast<const _Float16*>(p)[i];
  }
  return static_cast<const float*>(p)[i];
}

// Pillow's 8-bit BICUBIC resize of planar u8 [3,sh,sw] -> [3,dh,dw]:
// a horizontal pass to an 8-bit intermediate, then a vertical one, each
// rounded and clipped to 0..255 -- the u8 temp image Pillow's 8bpc path
// writes between passes (skipping it was measured up to 6 levels off
// PIL in the Qwen2.5-VL tower). Taps from the shared Pillow-exact table.
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

std::unique_ptr<MetalPixtralVisionEncoder>
MetalPixtralVisionEncoder::load(std::shared_ptr<WeightSet> ws_in,
                                metal_compute::MetalCompute* mc,
                                const Config& cfg)
{
  if (mc == nullptr || !mc->valid() || ws_in == nullptr) { return nullptr; }
  if (cfg.hidden != cfg.n_heads * cfg.head_dim || cfg.head_dim != 64
      || cfg.merge < 1 || cfg.patch_size < 1) {
    return nullptr;   // the attention entry point is head_dim 64
  }
  WeightSet* wts = ws_in.get();
  const SessionContextIntf* sess = mc->session();
  auto fail = [&](const std::string& why)
      -> std::unique_ptr<MetalPixtralVisionEncoder> {
    if (sess) {
      sess->warn(fmt("MetalPixtralVisionEncoder: {}; no image input",
                     why));
    }
    return nullptr;
  };

  auto m = std::unique_ptr<MetalPixtralVisionEncoder>(
      new MetalPixtralVisionEncoder());
  m->_ws = std::move(ws_in);
  m->_cfg = cfg;
  m->_mc = mc;

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_vis  = mc->load_library("qwen3_5_vision");
  m->_lib_elt  = mc->load_library("llm_elementwise");
  m->_lib_rms  = mc->load_library("rms_norm");
  m->_lib_qmm  = mc->load_library("affine_qmm_steel");
  m->_lib_attn = mc->load_library("attn_steel");
  m->_fn_rms       = m->_lib_rms.function("rms_norm_f16");
  m->_fn_rope      = m->_lib_vis.function("vision_rope_f16");
  m->_fn_gelu      = m->_lib_vis.function("gelu_erf_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_residual  = m->_lib_elt.function("residual_add_f16");
  m->_fn_swiglu    = m->_lib_elt.function("swiglu_f16");
  m->_fn_qmm = m->_lib_qmm.function(cfg.quant_bits == 8
      ? "affine_qmm_steel_w8g64" : "affine_qmm_steel_w4g64");
  // M4 (no matrix cores): the widest steel tile that loaded -- the
  // narrow BM32 default is the weakest of the three.
  m->_fn_gemm = m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  m->_gemm_bm = 64; m->_gemm_bn = 64;
  if (!m->_fn_gemm.valid()) {
    m->_fn_gemm = m->_lib_gemm.function("dense_gemm_t_f16");
    m->_gemm_bm = 32; m->_gemm_bn = 32;
  }
  if (!m->_fn_gemm.valid() || !m->_fn_rms.valid() || !m->_fn_rope.valid()
      || !m->_fn_gelu.valid() || !m->_fn_transpose.valid()
      || !m->_fn_residual.valid() || !m->_fn_swiglu.valid()
      || !m->_fn_qmm.valid() || !m->_lib_attn.valid()) {
    return fail("a kernel did not load");
  }
  m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));
  if (mc->supports_matrix_cores()) {
    if (std::getenv("VPIPE_PIXTRAL_NO_MMA2") == nullptr) {
      m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
      m->_fn_dense_mma =
          m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
      m->_fn_dense_mma_deep =
          m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
      m->_use_mma2 = m->_fn_dense_mma.valid()
          && m->_fn_dense_mma_deep.valid();
    }
    if (std::getenv("VPIPE_PIXTRAL_NO_NAX_ATTN") == nullptr) {
      m->_lib_attn_nax = mc->load_library("attn_steel_nax");
      m->_use_attn_nax = m->_lib_attn_nax.valid();
    }
  }

  // ---- weights -----------------------------------------------------
  // mlx-vlm conversions nest the ViT one level deeper than HF does.
  const auto& src = wts->src();
  std::string vit;
  for (const char* p : {"vision_tower.vision_model.", "vision_tower.",
                        "model.vision_tower."}) {
    if (src.info(std::string(p) + "ln_pre.weight") != nullptr) {
      vit = p;
      break;
    }
  }
  std::string proj;
  for (const char* p : {"multi_modal_projector.",
                        "model.multi_modal_projector."}) {
    if (src.info(std::string(p) + "norm.weight") != nullptr) {
      proj = p;
      break;
    }
  }
  if (vit.empty() || proj.empty()) {
    return fail("no Pixtral tower / Mistral3 projector in the checkpoint");
  }

  const std::string dk = "pixtral-vit/f16/";
  auto to_f16 = [&](const std::string& name) -> SharedBuffer {
    const auto* info = src.info(name);
    if (info == nullptr) { return {}; }
    if (info->dtype == "F16") {
      return wts->tensor(name, mc, WeightSet::Residency::Copied);
    }
    return wts->derived(dk + name, [&]() -> SharedBuffer {
      SharedBuffer raw = wts->read(name, mc, WeightSet::Residency::Copied);
      if (raw.empty()
          || (info->dtype != "BF16" && info->dtype != "F32")) {
        return {};
      }
      const std::size_t n = numel_(info->shape);
      SharedBuffer out = mc->make_shared_buffer(n * 2);
      auto* o = static_cast<_Float16*>(out.contents());
      for (std::size_t i = 0; i < n; ++i) {
        o[i] = (_Float16)elem_f32_(raw.contents(), info->dtype, i);
      }
      return out;
    });
  };

  // patch_conv: [O, kh, kw, C] (mlx) or [O, C, kh, kw] (HF), to the
  // [O, (kh, kw, c)] rows the patchify below writes.
  {
    const std::string name = vit + "patch_conv.weight";
    const auto* info = src.info(name);
    const int P = cfg.patch_size, O = cfg.hidden;
    if (info == nullptr || info->shape.size() != 4
        || info->shape[0] != O) {
      return fail("patch_conv.weight has an unexpected shape");
    }
    const bool hwc = info->shape[3] == 3 && info->shape[1] == P;
    const bool chw = info->shape[1] == 3 && info->shape[3] == P;
    if (!hwc && !chw) {
      return fail("patch_conv.weight has an unexpected layout");
    }
    m->_patch_w = wts->derived(
        dk + name + (hwc ? "/hwc" : "/chw->hwc"), [&]() -> SharedBuffer {
      SharedBuffer raw = wts->read(name, mc, WeightSet::Residency::Copied);
      if (raw.empty()) { return {}; }
      const std::size_t K = (std::size_t)P * P * 3;
      SharedBuffer out = mc->make_shared_buffer((std::size_t)O * K * 2);
      auto* o = static_cast<_Float16*>(out.contents());
      for (int oc = 0; oc < O; ++oc) {
        for (int y = 0; y < P; ++y) {
          for (int x = 0; x < P; ++x) {
            for (int c = 0; c < 3; ++c) {
              const std::size_t si = hwc
                  ? (((std::size_t)oc * P + y) * P + x) * 3 + c
                  : (((std::size_t)oc * 3 + c) * P + y) * P + x;
              o[(std::size_t)oc * K + ((std::size_t)y * P + x) * 3 + c] =
                  (_Float16)elem_f32_(raw.contents(), info->dtype, si);
            }
          }
        }
      }
      return out;
    });
  }
  m->_ln_pre = to_f16(vit + "ln_pre.weight");
  bool ok = !m->_patch_w.empty() && !m->_ln_pre.empty();

  m->_blocks.resize((std::size_t)cfg.depth);
  for (int b = 0; b < cfg.depth && ok; ++b) {
    const std::string p =
        vit + "transformer.layers." + std::to_string(b) + ".";
    Block& blk = m->_blocks[(std::size_t)b];
    blk.attn_norm = to_f16(p + "attention_norm.weight");
    blk.ffn_norm  = to_f16(p + "ffn_norm.weight");
    blk.qw = to_f16(p + "attention.q_proj.weight");
    blk.kw = to_f16(p + "attention.k_proj.weight");
    blk.vw = to_f16(p + "attention.v_proj.weight");
    blk.ow = to_f16(p + "attention.o_proj.weight");
    blk.gw = to_f16(p + "feed_forward.gate_proj.weight");
    blk.uw = to_f16(p + "feed_forward.up_proj.weight");
    blk.dw = to_f16(p + "feed_forward.down_proj.weight");
    ok = !blk.attn_norm.empty() && !blk.ffn_norm.empty()
        && !blk.qw.empty() && !blk.kw.empty() && !blk.vw.empty()
        && !blk.ow.empty() && !blk.gw.empty() && !blk.uw.empty()
        && !blk.dw.empty();
  }
  if (!ok) { return fail("a ViT weight is missing"); }

  m->_proj_norm = to_f16(proj + "norm.weight");

  // The merging layer, dense f16 with its columns permuted from the
  // reference's unfold order (c * S^2 + k) to merge order (k * hidden +
  // c), where k = ki * S + kj is the patch inside the S x S block.
  // Dequantised when stored affine: a column permutation splits 4-bit
  // groups.
  {
    const int S2 = cfg.merge * cfg.merge, Hd = cfg.hidden;
    const std::size_t Kin = (std::size_t)S2 * Hd;
    const std::string wn = proj + "patch_merger.merging_layer.weight";
    const auto* wi = src.info(wn);
    const auto* si = src.info(proj + "patch_merger.merging_layer.scales");
    const auto* bi = src.info(proj + "patch_merger.merging_layer.biases");
    if (wi == nullptr || wi->shape.size() != 2 || wi->shape[0] != Hd) {
      return fail("patch_merger.merging_layer is missing");
    }
    const bool quant = wi->dtype == "U32" && si != nullptr && bi != nullptr;
    int bits = 0, group = 0;
    if (quant) {
      bits = (int)((std::int64_t)wi->shape[1] * 32 / (std::int64_t)Kin);
      group = (int)(Kin / (std::size_t)si->shape[1]);
      if ((bits != 4 && bits != 8) || group <= 0) {
        return fail("patch_merger.merging_layer: unsupported quantization");
      }
    } else if ((std::size_t)wi->shape[1] != Kin) {
      return fail("patch_merger.merging_layer has an unexpected shape");
    }
    m->_merge_w = wts->derived(
        dk + wn + "/merge-order/s" + std::to_string(cfg.merge),
        [&]() -> SharedBuffer {
      SharedBuffer w = wts->read(wn, mc, WeightSet::Residency::Copied);
      if (w.empty()) { return {}; }
      SharedBuffer sc, bs;
      if (quant) {
        sc = wts->read(proj + "patch_merger.merging_layer.scales", mc,
                       WeightSet::Residency::Copied);
        bs = wts->read(proj + "patch_merger.merging_layer.biases", mc,
                       WeightSet::Residency::Copied);
        if (sc.empty() || bs.empty()) { return {}; }
      }
      SharedBuffer out = mc->make_shared_buffer((std::size_t)Hd * Kin * 2);
      auto* o = static_cast<_Float16*>(out.contents());
      const int per = quant ? 32 / bits : 1;
      const std::uint32_t mask = quant ? ((1u << bits) - 1u) : 0u;
      const std::size_t words = quant ? Kin / (std::size_t)per : 0;
      const std::size_t groups = quant ? Kin / (std::size_t)group : 0;
      for (int r = 0; r < Hd; ++r) {
        for (std::size_t i = 0; i < Kin; ++i) {
          float v;
          if (quant) {
            const auto* cw = static_cast<const std::uint32_t*>(w.contents());
            const std::uint32_t word =
                cw[(std::size_t)r * words + i / (std::size_t)per];
            const float q = (float)((word >> (bits * (int)(i % per))) & mask);
            const std::size_t g = (std::size_t)r * groups + i / group;
            v = q * elem_f32_(sc.contents(), si->dtype, g)
                + elem_f32_(bs.contents(), bi->dtype, g);
          } else {
            v = elem_f32_(w.contents(), wi->dtype, (std::size_t)r * Kin + i);
          }
          // Reference column i = c * S2 + k  ->  merge order k * Hd + c.
          const std::size_t c = i / (std::size_t)S2;
          const std::size_t k = i % (std::size_t)S2;
          o[(std::size_t)r * Kin + k * (std::size_t)Hd + c] = (_Float16)v;
        }
      }
      return out;
    });
  }

  // linear_1 / linear_2: affine as stored (scales/biases -> f16).
  auto qlinear = [&](const std::string& base, QLinear* q) {
    q->w = wts->tensor(base + ".weight", mc, WeightSet::Residency::Copied);
    q->s = to_f16(base + ".scales");
    q->b = to_f16(base + ".biases");
    return !q->w.empty() && !q->s.empty() && !q->b.empty();
  };
  if (m->_proj_norm.empty() || m->_merge_w.empty()
      || !qlinear(proj + "linear_1", &m->_lin1)
      || !qlinear(proj + "linear_2", &m->_lin2)) {
    return fail("a projector weight is missing (linear_1 / linear_2 must "
                "be affine-quantized)");
  }

  const int zb = std::max({cfg.hidden, cfg.intermediate, cfg.out_hidden});
  m->_zero_bias = mc->make_shared_buffer((std::size_t)zb * 2);
  std::memset(m->_zero_bias.contents(), 0, (std::size_t)zb * 2);
  return m;
}

MetalPixtralVisionEncoder::Config
MetalPixtralVisionEncoder::config_from(const ModelConfig& c)
{
  Config m;
  const auto& v = c.vision;
  if (v.depth > 0)             { m.depth = v.depth; }
  if (v.hidden_size > 0)       { m.hidden = v.hidden_size; }
  if (v.num_heads > 0)         { m.n_heads = v.num_heads; }
  m.head_dim = v.vit_head_dim > 0
      ? v.vit_head_dim : (m.n_heads > 0 ? m.hidden / m.n_heads : 0);
  if (v.intermediate_size > 0) { m.intermediate = v.intermediate_size; }
  if (v.patch_size > 0)        { m.patch_size = v.patch_size; }
  if (v.spatial_merge_size > 0) { m.merge = v.spatial_merge_size; }
  m.out_hidden = v.out_hidden_size > 0 ? v.out_hidden_size : c.hidden;
  if (v.vit_rope_theta > 0.0f) { m.rope_theta = v.vit_rope_theta; }
  m.proj_eps = c.rms_eps > 0.0f ? c.rms_eps : m.proj_eps;
  const std::int64_t edge =
      bag::integer(c.extra, model_config::kVisionImageSize, 0);
  if (edge > 0) { m.longest_edge = (int)edge; }
  if (c.quantization.bits == 8) { m.quant_bits = 8; }
  for (int i = 0; i < 3; ++i) {
    m.image_mean[i] = v.image_mean[i];
    m.image_std[i]  = v.image_std[i];
  }
  return m;
}

void
MetalPixtralVisionEncoder::target_size(int H, int W, int* th, int* tw,
                                       int max_tokens) const
{
  // The processor: scale down (never up) so the longest edge fits, by
  // flooring each side; then round each UP to the token unit.
  const int unit = _cfg.patch_size * _cfg.merge;
  auto fit = [&](int edge, int* oh, int* ow) {
    const double ratio = std::max((double)H / edge, (double)W / edge);
    int h = H, w = W;
    if (ratio > 1.0) {
      h = (int)std::floor((double)H / ratio);
      w = (int)std::floor((double)W / ratio);
    }
    *oh = ((std::max(h, 1) - 1) / unit + 1) * unit;
    *ow = ((std::max(w, 1) - 1) / unit + 1) * unit;
  };
  int edge = _cfg.longest_edge;
  fit(edge, th, tw);
  if (max_tokens <= 0) { return; }
  // A token budget: the same rule at a shorter longest edge, stepped
  // down a unit at a time from the area estimate until the grid fits.
  const double tok = (double)(*th / unit) * (*tw / unit);
  if (tok > max_tokens) {
    edge = std::min(edge, (int)((double)std::max(H, W)
                                * std::sqrt((double)max_tokens / tok)));
    for (; edge > unit; edge -= unit / 2) {
      fit(edge, th, tw);
      if ((*th / unit) * (*tw / unit) <= max_tokens) { return; }
    }
    fit(unit, th, tw);
  }
}

void
MetalPixtralVisionEncoder::set_layout_rows(
    const std::vector<float>& image_break,
    const std::vector<float>& image_end)
{
  auto to16 = [&](const std::vector<float>& v) {
    std::vector<std::uint16_t> o(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      const _Float16 h = (_Float16)v[i];
      std::memcpy(&o[i], &h, 2);
    }
    return o;
  };
  _break_row = to16(image_break);
  _end_row = to16(image_end);
}

void
MetalPixtralVisionEncoder::preprocess(const std::uint8_t* rgb, int H, int W,
                                      int max_tokens, std::vector<float>* px,
                                      int* th, int* tw) const
{
  target_size(H, W, th, tw, max_tokens);
  std::vector<std::uint8_t> rs;
  const std::uint8_t* img = rgb;
  if (*th != H || *tw != W) {
    rs.resize((std::size_t)3 * *th * *tw);
    resize_bicubic_u8_(rgb, H, W, rs.data(), *th, *tw);
    img = rs.data();
  }
  // Rescale (1/255) and normalise in f32, as the processor does.
  const std::size_t plane = (std::size_t)*th * *tw;
  px->resize(3 * plane);
  const float rescale = 1.0f / 255.0f;
  for (int c = 0; c < 3; ++c) {
    const float mean = _cfg.image_mean[c], sd = _cfg.image_std[c];
    for (std::size_t i = 0; i < plane; ++i) {
      (*px)[(std::size_t)c * plane + i] =
          ((float)img[(std::size_t)c * plane + i] * rescale - mean) / sd;
    }
  }
}

MetalPixtralVisionEncoder::Result
MetalPixtralVisionEncoder::encode(const std::uint8_t* rgb, int H, int W,
                                  int max_tokens)
{
  if (rgb == nullptr || H <= 0 || W <= 0) { return {}; }
  std::vector<float> px;
  int th = 0, tw = 0;
  preprocess(rgb, H, W, max_tokens, &px, &th, &tw);
  return encode_normalized(px.data(), th, tw);
}

MetalPixtralVisionEncoder::Result
MetalPixtralVisionEncoder::encode_normalized(const float* px, int h, int w,
                                             std::vector<float>* vit_out,
                                             std::vector<float>* proj_out)
{
  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmVision,
                     kPerfLlmVisionBegin, 1);
  using Clock = std::chrono::steady_clock;
  const auto t_enter = Clock::now();
  const Config& c = _cfg;
  const int P = c.patch_size, S = c.merge, S2 = S * S;
  const int hidden = c.hidden, heads = c.n_heads, hd = c.head_dim;
  const int inter = c.intermediate, outh = c.out_hidden;
  const int unit = P * S;
  Result res;
  if (px == nullptr || h <= 0 || w <= 0 || h % unit != 0 || w % unit != 0) {
    return res;
  }
  if (_break_row.size() != (std::size_t)outh
      || _end_row.size() != (std::size_t)outh) {
    if (_session) {
      _session->warn(fmt("MetalPixtralVisionEncoder: no [IMG_BREAK] / "
                         "[IMG_END] rows set; cannot lay the image out"));
    }
    return res;
  }
  const int gh = h / unit, gw = w / unit;     // merged grid
  const int pW = gw * S;                      // patch grid width
  const int n = gh * gw * S2;                 // patches
  const int n_im = gh * gw;                   // merged tokens

  // Patchify in MERGE order: patch r = (I*gw + J)*S2 + ki*S + kj is
  // patch (S*I + ki, S*J + kj); features (kh, kw, c), the order the
  // patch_conv rows were laid out in. The 2-D RoPE tables follow the
  // same order. Pixtral's axial RoPE: of the head's D/2 rotation pairs,
  // the first D/4 turn with the patch ROW at the even-indexed base
  // frequencies, the next D/4 with the COLUMN at the odd-indexed ones
  // (theta^(-2i/D), i < D/2), the second half of the head repeating the
  // first (rotate-half layout).
  const int feat_dim = 3 * P * P;
  const std::size_t plane = (std::size_t)h * w;
  std::vector<_Float16> feat((std::size_t)n * feat_dim);
  std::vector<_Float16> cosb((std::size_t)n * hd), sinb((std::size_t)n * hd);
  std::vector<float> freq((std::size_t)hd / 2);
  for (int i = 0; i < hd / 2; ++i) {
    freq[(std::size_t)i] = 1.0f / std::pow(c.rope_theta,
                                           (2.0f * (float)i) / (float)hd);
  }
  const int q4 = hd / 4;
  for (int I = 0; I < gh; ++I) {
    for (int J = 0; J < gw; ++J) {
      for (int k = 0; k < S2; ++k) {
        const int r = (I * gw + J) * S2 + k;
        const int pi = I * S + k / S, pj = J * S + k % S;
        _Float16* fp = feat.data() + (std::size_t)r * feat_dim;
        int o = 0;
        for (int y = 0; y < P; ++y) {
          const std::size_t row = (std::size_t)(pi * P + y) * w;
          for (int x = 0; x < P; ++x) {
            const std::size_t at = row + (std::size_t)(pj * P + x);
            for (int ch = 0; ch < 3; ++ch) {
              fp[o++] = (_Float16)px[(std::size_t)ch * plane + at];
            }
          }
        }
        _Float16* cp = cosb.data() + (std::size_t)r * hd;
        _Float16* sp = sinb.data() + (std::size_t)r * hd;
        for (int d = 0; d < hd / 2; ++d) {
          const float ang = (d < q4)
              ? (float)pi * freq[(std::size_t)(2 * d)]
              : (float)pj * freq[(std::size_t)(2 * (d - q4) + 1)];
          cp[d] = cp[d + hd / 2] = (_Float16)std::cos(ang);
          sp[d] = sp[d + hd / 2] = (_Float16)std::sin(ang);
        }
      }
    }
  }

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer featb = buf((std::size_t)n * feat_dim);
  SharedBuffer cosbuf = buf((std::size_t)n * hd);
  SharedBuffer sinbuf = buf((std::size_t)n * hd);
  std::memcpy(featb.contents(), feat.data(), feat.size() * 2);
  std::memcpy(cosbuf.contents(), cosb.data(), cosb.size() * 2);
  std::memcpy(sinbuf.contents(), sinb.data(), sinb.size() * 2);

  const std::size_t nh = (std::size_t)n * hidden;
  SharedBuffer x = buf(nh), x0 = buf(nh), n1 = buf(nh);
  SharedBuffer q = buf(nh), kk = buf(nh), v = buf(nh);
  SharedBuffer qr = buf(nh), kr = buf(nh);
  SharedBuffer qt = buf(nh), kt = buf(nh), vt = buf(nh);
  SharedBuffer atb = buf(nh), att = buf(nh), proj = buf(nh);
  SharedBuffer gbuf = buf((std::size_t)n * inter);
  SharedBuffer ubuf = buf((std::size_t)n * inter);
  SharedBuffer xn = buf(nh);
  SharedBuffer merged = buf((std::size_t)n_im * hidden);
  SharedBuffer h1 = buf((std::size_t)n_im * outh);
  SharedBuffer h2 = buf((std::size_t)n_im * outh);

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm = [&](const SharedBuffer& xin, const SharedBuffer& wt,
                    const SharedBuffer& y, int M, int N, int Kk) {
      if (_use_mma2) {
        // matmul2d y = x @ w^T; the tensor extents clamp M/N tails.
        const bool deep = (Kk >= 6144);
        const int BN = deep ? 256 : 128;
        enc.set_function(deep ? _fn_dense_mma_deep : _fn_dense_mma);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, wt);
        enc.set_buffer(2, wt);     // bias slot unused (has_bias=0)
        enc.set_buffer(3, y);
        enc.set_constant(4, Kk);
        enc.set_constant(5, N);
        enc.set_constant(6, M);
        enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                      (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        return;
      }
      enc.set_function(_fn_gemm);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, wt);
      enc.set_buffer(2, _zero_bias);
      enc.set_buffer(3, y);
      enc.set_constant(4, Kk);
      enc.set_constant(5, N);
      enc.set_constant(6, M);
      enc.set_constant(7, 0);
      // Threadgroup-indexed: x = ceil(N/BN)*32, y = ceil(M/BM)*2, z = 2.
      enc.dispatch({(unsigned)(((N + _gemm_bn - 1) / _gemm_bn) * 32),
                    (unsigned)(((M + _gemm_bm - 1) / _gemm_bm) * 2), 2},
                   {32, 2, 2});
    };
    auto qmm = [&](const QLinear& ql, const SharedBuffer& xin,
                   const SharedBuffer& y, int M, int N, int Kk) {
      enc.set_function(_fn_qmm);
      enc.set_buffer(0, ql.w);
      enc.set_buffer(1, ql.s);
      enc.set_buffer(2, ql.b);
      enc.set_buffer(3, xin);
      enc.set_buffer(4, y);
      enc.set_constant(5, Kk);
      enc.set_constant(6, N);
      enc.set_constant(7, M);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
    };
    auto rms = [&](const SharedBuffer& xin, const SharedBuffer& wt,
                   const SharedBuffer& y, int R, int Hd, float eps) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, wt);
      enc.set_buffer(2, y);
      enc.set_constant(3, Hd);
      enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto rope = [&](const SharedBuffer& in, const SharedBuffer& out) {
      enc.set_function(_fn_rope);
      enc.set_buffer(0, in);
      enc.set_buffer(1, cosbuf);
      enc.set_buffer(2, sinbuf);
      enc.set_buffer(3, out);
      enc.set_constant(4, heads);
      enc.set_constant(5, hd);
      enc.dispatch({(unsigned)(n * heads * hd), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in);
      enc.set_buffer(1, out);
      enc.set_constant(2, A);
      enc.set_constant(3, Bd);
      enc.set_constant(4, hd);
      enc.dispatch({(unsigned)hd, (unsigned)Bd, (unsigned)A},
                   {(unsigned)hd, 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& b,
                        const SharedBuffer& out, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a);
      enc.set_buffer(1, b);
      enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto unary = [&](const metal_compute::ComputeFunction& fn,
                     const SharedBuffer& in, const SharedBuffer& out,
                     int nn) {
      enc.set_function(fn);
      enc.set_buffer(0, in);
      enc.set_buffer(1, out);
      enc.set_constant(2, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };

    // Patch embed (bias-free conv as a GEMM) + ln_pre.
    gemm(featb, _patch_w, x0, n, hidden, feat_dim);
    rms(x0, _ln_pre, x, n, hidden, c.rms_eps);

    // Bidirectional attention over the image's patches: the vendored
    // MLX steel flash kernel (NAX on M5), scale 1/sqrt(head_dim). One
    // image per encode, so the reference's block-diagonal mask is the
    // whole square.
    const bool nax = _use_attn_nax;
    const int a_bq = nax ? 64 : 32, a_bk = nax ? 32 : 16;
    auto* ap = static_cast<SteelAttnParams*>(_attn_params.contents());
    ap->B = 1; ap->H = heads; ap->D = hd; ap->qL = n; ap->kL = n;
    ap->gqa_factor = 1;
    ap->scale = 1.0f / std::sqrt((float)hd);
    ap->NQ = (n + a_bq - 1) / a_bq; ap->NK = (n + a_bk - 1) / a_bk;
    ap->NQ_aligned = n / a_bq; ap->NK_aligned = n / a_bk;
    ap->qL_rem = n - ap->NQ_aligned * a_bq;
    ap->kL_rem = n - ap->NK_aligned * a_bk;
    ap->qL_off = 0;
    ap->Q_strides[0] = (std::int64_t)heads * n * hd;
    ap->Q_strides[1] = (std::int64_t)n * hd; ap->Q_strides[2] = hd;
    for (int i = 0; i < 3; ++i) {
      ap->K_strides[i] = ap->Q_strides[i];
      ap->V_strides[i] = ap->Q_strides[i];
      ap->O_strides[i] = ap->Q_strides[i];
    }
    metal_compute::FunctionConstants fc;
    fc.set_bool(200, (n % a_bq) == 0).set_bool(201, (n % a_bk) == 0)
        .set_bool(300, false).set_bool(301, false).set_bool(302, false);
    const metal_compute::ComputeFunction fn_attn = nax
        ? _lib_attn_nax.function("attn_steel_nax_h_bd64", fc)
        : _lib_attn.function("attn_steel_h_bd64", fc);
    if (!fn_attn.valid()) { return res; }
    const unsigned nqb = (unsigned)((n + a_bq - 1) / a_bq);

    for (int b = 0; b < c.depth; ++b) {
      const Block& blk = _blocks[(std::size_t)b];
      rms(x, blk.attn_norm, n1, n, hidden, c.rms_eps);
      gemm(n1, blk.qw, q, n, hidden, hidden);
      gemm(n1, blk.kw, kk, n, hidden, hidden);
      gemm(n1, blk.vw, v, n, hidden, hidden);
      rope(q, qr);
      rope(kk, kr);
      transpose(qr, qt, n, heads);               // [n,H,hd] -> [H,n,hd]
      transpose(kr, kt, n, heads);
      transpose(v, vt, n, heads);
      enc.set_function(fn_attn);
      enc.set_buffer(0, qt);
      enc.set_buffer(1, kt);
      enc.set_buffer(2, vt);
      enc.set_buffer(3, atb);
      enc.set_buffer(4, _attn_params);
      enc.dispatch({32 * nqb, 4 * (unsigned)heads, 1}, {32, 4, 1});
      transpose(atb, att, heads, n);             // [H,n,hd] -> [n,H,hd]
      gemm(att, blk.ow, proj, n, hidden, hidden);
      residual(x, proj, x, n * hidden);

      rms(x, blk.ffn_norm, n1, n, hidden, c.rms_eps);
      gemm(n1, blk.gw, gbuf, n, inter, hidden);
      gemm(n1, blk.uw, ubuf, n, inter, hidden);
      enc.set_function(_fn_swiglu);              // silu(gate) * up
      enc.set_buffer(0, gbuf);
      enc.set_buffer(1, ubuf);
      enc.set_buffer(2, gbuf);
      enc.set_constant(3, n * inter);
      enc.dispatch({(unsigned)(n * inter), 1, 1}, {256, 1, 1});
      gemm(gbuf, blk.dw, proj, n, hidden, inter);
      residual(x, proj, x, n * hidden);
    }

    // Projector: RMSNorm per patch; in merge order a block's S2 patches
    // are adjacent rows, so [n, hidden] IS [n_im, S2*hidden] for the
    // merging GEMM; then linear_1, exact GELU, linear_2.
    rms(x, _proj_norm, xn, n, hidden, c.proj_eps);
    gemm(xn, _merge_w, merged, n_im, hidden, S2 * hidden);
    qmm(_lin1, merged, h1, n_im, outh, hidden);
    unary(_fn_gelu, h1, h1, n_im * outh);
    qmm(_lin2, h1, h2, n_im, outh, outh);
  }
  const auto t_host = Clock::now();
  if (!stream.commit().wait_ok()) {
    if (_session) {
      _session->warn(fmt("MetalPixtralVisionEncoder: the encode failed "
                         "on the GPU"));
    }
    return res;
  }

  // Lay the image out: each grid row of gw tokens, then [IMG_BREAK] --
  // [IMG_END] after the last.
  res.grid_h = gh;
  res.grid_w = gw;
  res.out_hidden = outh;
  res.n_tokens = gh * (gw + 1);
  res.embeddings = buf((std::size_t)res.n_tokens * outh);
  {
    const std::size_t rowb = (std::size_t)outh * 2;
    auto* dst = static_cast<std::uint8_t*>(res.embeddings.contents());
    const auto* srcp = static_cast<const std::uint8_t*>(h2.contents());
    for (int I = 0; I < gh; ++I) {
      std::memcpy(dst, srcp + (std::size_t)I * gw * rowb,
                  (std::size_t)gw * rowb);
      dst += (std::size_t)gw * rowb;
      std::memcpy(dst, (I + 1 < gh ? _break_row : _end_row).data(), rowb);
      dst += rowb;
    }
  }
  if (vit_out != nullptr) {
    // Back to RASTER patch order: patch (pi, pj) -> merge-order row.
    vit_out->assign(nh, 0.0f);
    const auto* xs = static_cast<const _Float16*>(x.contents());
    for (int r = 0; r < n; ++r) {
      const int blk = r / S2, k = r % S2;
      const int pi = (blk / gw) * S + k / S;
      const int pj = (blk % gw) * S + k % S;
      const _Float16* s = xs + (std::size_t)r * hidden;
      float* d = vit_out->data() + ((std::size_t)pi * pW + pj) * hidden;
      for (int i = 0; i < hidden; ++i) { d[i] = (float)s[i]; }
    }
  }
  if (proj_out != nullptr) {
    const auto* hs = static_cast<const _Float16*>(h2.contents());
    proj_out->assign((std::size_t)n_im * outh, 0.0f);
    for (std::size_t i = 0; i < proj_out->size(); ++i) {
      (*proj_out)[i] = (float)hs[i];
    }
  }
  if (_session && std::getenv("VPIPE_VISION_PROFILE") != nullptr) {
    auto ms = [](Clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };
    _session->info(fmt(
        "MetalPixtralVisionEncoder: {}x{} -> {} patches, {}x{} tokens | "
        "host {:.2f} ms | gpu {:.2f} ms", h, w, n, gh, gw,
        ms(t_host - t_enter), ms(Clock::now() - t_host)));
  }
  return res;
}

}  // namespace vpipe::genai
