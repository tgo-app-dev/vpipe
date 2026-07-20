#include "generative-models/flux2/metal-flux2-transformer.h"

#include "generative-models/shared/i8-gemm.h"

#include "common/flex-data.h"
#include "common/perf-scope.h"
#include "common/vpipe-format.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/stream-pin.h"
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
using metal_compute::ComputeFunction;

namespace {

// C++ mirror of mlx::steel::AttnParams (identical layout to the LM / Krea-2).
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

// Load a checkpoint tensor -> f16 SharedBuffer (F32/F16/BF16 sources).
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
  auto* d = static_cast<_Float16*>(out.contents());
  if (info->dtype == "F32") {
    const auto* s = static_cast<const float*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)s[i]; }
  } else if (info->dtype == "F16") {
    std::memcpy(d, raw.contents(), n * 2);
  } else if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t u = (std::uint32_t)s[i] << 16;
      float f; std::memcpy(&f, &u, 4); d[i] = (_Float16)f;
    }
  } else {
    return {};
  }
  return out;
}

}  // namespace

MetalFlux2Transformer::QWeight
MetalFlux2Transformer::load_qw_(const MetalLlamaWeights& wts,
                                const std::string& name)
{
  QWeight qw;
  const auto* si = wts.info(name + ".scales");
  const auto* ci = wts.info(name + ".weight");
  if (_quant_bits > 0 && si != nullptr && ci != nullptr &&
      si->shape.size() == 2 && ci->shape.size() == 2) {
    const long gcols = ci->shape[1];                 // K*bits/32
    const long scols = si->shape[1];                 // K/group
    const long K = scols * (long)_quant_group;
    const int bits = K > 0 ? (int)(gcols * 32 / K) : 0;
    qw.bits = (bits == 8) ? 8 : 4;
    qw.n = (int)ci->shape[0];
    qw.k = (int)K;
    // Zero-copy mmap views (evictable) when enabled; else owned copies. These
    // codes/scales/qbias feed the GEMM read-only and are never CPU-modified, so
    // aliasing the file is safe. interleave_gu_/slice_rows_ still copy where a
    // CPU relayout is needed, correctly dropping the mapping for those.
    if (_mmap_weights) {
      qw.codes  = wts.load_mapped(name + ".weight", _mc);
      qw.scales = wts.load_mapped(name + ".scales", _mc);
      qw.qbias  = wts.load_mapped(name + ".biases", _mc);
    } else {
      qw.codes  = wts.load(name + ".weight", _mc);
      qw.scales = wts.load(name + ".scales", _mc);
      qw.qbias  = wts.load(name + ".biases", _mc);
    }
    if (!qw.codes.empty() && !qw.scales.empty() && !qw.qbias.empty()) {
      qw.quantized = true;
      return qw;
    }
    qw.codes = {}; qw.scales = {}; qw.qbias = {};
  }
  const auto* wi = wts.info(name + ".weight");
  if (wi != nullptr && wi->shape.size() == 2) {
    qw.n = (int)wi->shape[0];
    qw.k = (int)wi->shape[1];
  }
  qw.w = to_f16_(wts, _mc, name + ".weight");
  return qw;
}

void
MetalFlux2Transformer::interleave_gu_(QWeight& qw)
{
  const int n = qw.n;
  if (n <= 1) { return; }
  auto perm = [&](const SharedBuffer& src) -> SharedBuffer {
    if (src.empty()) { return {}; }
    const std::size_t rb = src.byte_size() / (std::size_t)n;   // row bytes
    if (rb == 0) { return {}; }
    const int inner = n / 2;
    SharedBuffer dst = _mc->make_shared_buffer((std::size_t)n * rb);
    if (dst.empty()) { return {}; }
    const auto* s = static_cast<const std::uint8_t*>(src.contents());
    auto* d = static_cast<std::uint8_t*>(dst.contents());
    for (int g = 0; g < inner; ++g) {
      std::memcpy(d + (std::size_t)(2 * g) * rb,
                  s + (std::size_t)g * rb, rb);
      std::memcpy(d + (std::size_t)(2 * g + 1) * rb,
                  s + (std::size_t)(inner + g) * rb, rb);
    }
    return dst;
  };
  if (qw.quantized) {
    qw.codes  = perm(qw.codes);
    qw.scales = perm(qw.scales);
    qw.qbias  = perm(qw.qbias);
  } else {
    qw.w = perm(qw.w);
  }
}

bool
MetalFlux2Transformer::load_double_(const MetalLlamaWeights& wts,
                                    const std::string& pre, DoubleBlock& b)
{
  b.q  = load_qw_(wts, pre + "attn.to_q");
  b.k  = load_qw_(wts, pre + "attn.to_k");
  b.v  = load_qw_(wts, pre + "attn.to_v");
  b.o  = load_qw_(wts, pre + "attn.to_out.0");
  b.aq = load_qw_(wts, pre + "attn.add_q_proj");
  b.ak = load_qw_(wts, pre + "attn.add_k_proj");
  b.av = load_qw_(wts, pre + "attn.add_v_proj");
  b.ao = load_qw_(wts, pre + "attn.to_add_out");
  b.qn  = to_f16_(wts, _mc, pre + "attn.norm_q.weight");
  b.kn  = to_f16_(wts, _mc, pre + "attn.norm_k.weight");
  b.aqn = to_f16_(wts, _mc, pre + "attn.norm_added_q.weight");
  b.akn = to_f16_(wts, _mc, pre + "attn.norm_added_k.weight");
  b.ff_in   = load_qw_(wts, pre + "ff.linear_in");
  b.ff_out  = load_qw_(wts, pre + "ff.linear_out");
  b.cff_in  = load_qw_(wts, pre + "ff_context.linear_in");
  b.cff_out = load_qw_(wts, pre + "ff_context.linear_out");
  if (_cfg.double_ff_hidden == 0 && b.ff_in.n > 0) {
    _cfg.double_ff_hidden = b.ff_in.n;
  }
  // Fused-SwiGLU: interleave the gate|up rows of linear_in so the fused kernel
  // reads even col = gate, odd col = up. (ff_out is unchanged.)
  if (_fuse_ff) {
    interleave_gu_(b.ff_in);
    interleave_gu_(b.cff_in);
  }
  return !b.q.empty() && !b.k.empty() && !b.v.empty() && !b.o.empty() &&
         !b.aq.empty() && !b.ak.empty() && !b.av.empty() && !b.ao.empty() &&
         !b.qn.empty() && !b.kn.empty() && !b.aqn.empty() && !b.akn.empty() &&
         !b.ff_in.empty() && !b.ff_out.empty() && !b.cff_in.empty() &&
         !b.cff_out.empty();
}

MetalFlux2Transformer::QWeight
MetalFlux2Transformer::slice_rows_(const QWeight& src, int start, int count)
{
  QWeight d;
  d.quantized = src.quantized;
  d.bits = src.bits;
  d.n = count;
  d.k = src.k;
  auto ext = [&](const SharedBuffer& s) -> SharedBuffer {
    if (s.empty() || src.n <= 0) { return {}; }
    const std::size_t rb = s.byte_size() / (std::size_t)src.n;   // row bytes
    SharedBuffer o = _mc->make_shared_buffer((std::size_t)count * rb);
    if (o.empty()) { return {}; }
    std::memcpy(o.contents(),
                static_cast<const std::uint8_t*>(s.contents())
                    + (std::size_t)start * rb,
                (std::size_t)count * rb);
    return o;
  };
  if (src.quantized) {
    d.codes = ext(src.codes); d.scales = ext(src.scales);
    d.qbias = ext(src.qbias);
  } else {
    d.w = ext(src.w);
  }
  return d;
}

bool
MetalFlux2Transformer::load_single_(const MetalLlamaWeights& wts,
                                    const std::string& pre, SingleBlock& b)
{
  b.qkv_mlp = load_qw_(wts, pre + "attn.to_qkv_mlp_proj");
  b.o  = load_qw_(wts, pre + "attn.to_out");
  b.qn = to_f16_(wts, _mc, pre + "attn.norm_q.weight");
  b.kn = to_f16_(wts, _mc, pre + "attn.norm_k.weight");
  if (_cfg.single_mlp_in == 0 && b.qkv_mlp.n > 0) {
    const int rest = b.qkv_mlp.n - 3 * _cfg.hidden;   // 2 * single_mlp_in
    if (rest > 0) { _cfg.single_mlp_in = rest / 2; }
  }
  if (!b.qkv_mlp.empty() && !b.o.empty() && !b.qn.empty() && !b.kn.empty()) {
    // Fused single block: split to_qkv_mlp_proj into the attention qkv rows and
    // the INTERLEAVED gate|up mlp rows, so the mlp runs as a fused-SwiGLU GEMM.
    if (_fuse_ff) {
      const int qkv_rows = 3 * _cfg.hidden;
      b.qkv    = slice_rows_(b.qkv_mlp, 0, qkv_rows);
      b.mlp_gu = slice_rows_(b.qkv_mlp, qkv_rows, b.qkv_mlp.n - qkv_rows);
      interleave_gu_(b.mlp_gu);
      b.qkv_mlp = QWeight{};   // fused path uses qkv + mlp_gu instead
      return !b.qkv.empty() && !b.mlp_gu.empty();
    }
    return true;
  }
  return false;
}

MetalFlux2Transformer::~MetalFlux2Transformer() = default;

std::unique_ptr<MetalFlux2Transformer>
MetalFlux2Transformer::load(const std::string& model_dir, MetalCompute* mc,
                            const Config& cfg, bool stream_blocks,
                            double pin_frac)
{
  if (mc == nullptr) { return nullptr; }
  auto wtsopt = MetalLlamaWeights::open_model(model_dir);
  if (!wtsopt.has_value()) { return nullptr; }

  auto m = std::unique_ptr<MetalFlux2Transformer>(new MetalFlux2Transformer());
  m->_mc = mc;
  m->_cfg = cfg;
  m->_stream_blocks = stream_blocks;
  // Zero-copy mmap of the quantized weights (see _mmap_weights) so the DiT's
  // resident footprint stays reclaimable under memory pressure. Off when
  // streaming (blocks re-read JIT) or via VPIPE_FLUX2_NO_MMAP_WEIGHTS. Retain
  // the source mmap for the model's lifetime so the mapped views stay valid.
  m->_mmap_weights =
      !stream_blocks && std::getenv("VPIPE_FLUX2_NO_MMAP_WEIGHTS") == nullptr;
  if (m->_mmap_weights) {
    m->_stream_wts = std::make_unique<MetalLlamaWeights>(std::move(*wtsopt));
  }
  const MetalLlamaWeights& wts =
      m->_mmap_weights ? *m->_stream_wts : *wtsopt;

  {
    namespace fs = std::filesystem;
    std::ifstream in(fs::path(model_dir) / "config.json");
    if (in) {
      FlexData cfgj = FlexData::from_json(in);
      if (cfgj.is_object()) {
        auto obj = cfgj.as_object();
        // Structural dims from config.json so the same code drives every
        // Flux2Transformer2DModel size (klein-4B: 5+20 blocks, 24 heads, no
        // guidance; klein-9B: a larger DiT with guidance_embeds). Absent keys
        // keep the Config default. hidden = num_attention_heads * head_dim.
        auto geti = [&](const char* k, int cur) -> int {
          return obj.contains(k) ? (int)obj.at(k).as_int(cur) : cur;
        };
        auto getf = [&](const char* k, float cur) -> float {
          return obj.contains(k) ? (float)obj.at(k).as_real(cur) : cur;
        };
        m->_cfg.n_heads      = geti("num_attention_heads", m->_cfg.n_heads);
        m->_cfg.head_dim     = geti("attention_head_dim", m->_cfg.head_dim);
        m->_cfg.hidden       = m->_cfg.n_heads * m->_cfg.head_dim;
        m->_cfg.n_double     = geti("num_layers", m->_cfg.n_double);
        m->_cfg.n_single     = geti("num_single_layers", m->_cfg.n_single);
        m->_cfg.in_channels  = geti("in_channels", m->_cfg.in_channels);
        m->_cfg.joint_dim    = geti("joint_attention_dim", m->_cfg.joint_dim);
        m->_cfg.timestep_dim =
            geti("timestep_guidance_channels", m->_cfg.timestep_dim);
        m->_cfg.mlp_ratio    = getf("mlp_ratio", m->_cfg.mlp_ratio);
        m->_cfg.norm_eps     = getf("eps", m->_cfg.norm_eps);
        m->_cfg.rope_theta   = getf("rope_theta", m->_cfg.rope_theta);
        if (obj.contains("guidance_embeds")) {
          m->_cfg.guidance_embeds =
              obj.at("guidance_embeds").as_bool(m->_cfg.guidance_embeds);
        }
        if (obj.contains("axes_dims_rope")) {
          FlexData ax = obj.at("axes_dims_rope");
          if (ax.is_array()) {
            auto av = ax.as_array();
            for (int i = 0; i < 4 && i < (int)av.size(); ++i) {
              m->_cfg.axes_dim[i] = (int)av[i].as_int(m->_cfg.axes_dim[i]);
            }
          }
        }
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

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_elt  = mc->load_library("llm_elementwise");
  m->_lib_rms  = mc->load_library("rms_norm");
  m->_lib_sdpa = mc->load_library("sdpa");
  m->_lib_vis  = mc->load_library("qwen3_5_vision");
  m->_lib_rope = mc->load_library("rope");
  m->_fn_gemm        = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_gemm_bm64   = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_gemm_bm64bn64 = m->_lib_gemm.function("dense_gemm_t_bm64bn64_f16");
  m->_fn_gemm_bm64_a16 = m->_lib_gemm.function("dense_gemm_t_bm64_acc16_f16");
  m->_fn_ff_swiglu    = m->_lib_gemm.function("dense_gemm_swiglu_bm64_f16");
  m->_fn_ff_swiglu_a16 =
      m->_lib_gemm.function("dense_gemm_swiglu_bm64_acc16_f16");
  m->_fn_gemm_bias   = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_rms         = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_swiglu      = m->_lib_elt.function("swiglu_f16");
  m->_fn_residual    = m->_lib_elt.function("residual_add_f16");
  m->_fn_transpose   = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_sdpa        = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_gelu_tanh   = m->_lib_vis.function("gelu_tanh_f16");
  m->_fn_layernorm   = m->_lib_vis.function("layer_norm_bias_f16");
  m->_fn_rope_table  = m->_lib_rope.function("rope_pair_table_f16");
  m->_fn_adaln       = m->_lib_elt.function("adaln_modulate_f16");
  m->_fn_gated       = m->_lib_elt.function("gated_residual_f16");
  m->_fn_bias_add    = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_headslice   = m->_lib_elt.function("head_slice_f16");
  m->_fn_mulsig      = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_concat      = m->_lib_elt.function("concat_cols_f16");
  m->_fn_transpose_rs = m->_lib_elt.function("transpose_abd_rs_f16");
  m->_fn_swiglu_rs   = m->_lib_elt.function("swiglu_rs_f16");
  m->_fn_colabsmax   = m->_lib_elt.function("col_absmax_f16");   // AWQ tap
  if (!m->_fn_gemm.valid() || !m->_fn_gemm_bias.valid() ||
      !m->_fn_rms.valid() ||
      !m->_fn_swiglu.valid() || !m->_fn_residual.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_gelu_tanh.valid() || !m->_fn_layernorm.valid() ||
      !m->_fn_rope_table.valid() || !m->_fn_adaln.valid() ||
      !m->_fn_gated.valid() || !m->_fn_bias_add.valid() ||
      !m->_fn_headslice.valid() || !m->_fn_mulsig.valid() ||
      !m->_fn_concat.valid() ||
      !m->_fn_colabsmax.valid()) {
    return nullptr;
  }
  if (m->_quant_bits > 0) {
    m->_lib_qmm = mc->load_library("affine_qmm_steel");
    const std::string g = "g" + std::to_string(m->_quant_group);
    m->_fn_qmm4 = m->_lib_qmm.function("affine_qmm_steel_w4" + g);
    m->_fn_qmm8 = m->_lib_qmm.function("affine_qmm_steel_w8" + g);
    if (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid()) { return nullptr; }
    // BM128 twins (g64 only) for the big M = seq quant GEMMs.
    m->_fn_qmm4_bm128 = m->_lib_qmm.function("affine_qmm_steel_w4g64_bm128");
    m->_fn_qmm8_bm128 = m->_lib_qmm.function("affine_qmm_steel_w8g64_bm128");
    m->_qmm_tile = (m->_quant_group == 64 && m->_fn_qmm4_bm128.valid()
                    && m->_fn_qmm8_bm128.valid()) ? 1 : 0;
    if (const char* t = std::getenv("VPIPE_FLUX2_QMM_TILE")) {
      m->_qmm_tile = std::atoi(t);
    }
    m->_fn_qmm_swiglu4_bm64 =
        m->_lib_qmm.function("affine_qmm_swiglu_w4g64_bm64");
    m->_fn_qmm_swiglu8_bm64 =
        m->_lib_qmm.function("affine_qmm_swiglu_w8g64_bm64");
    m->_fn_qmm_swiglu4_bm64_a16 =
        m->_lib_qmm.function("affine_qmm_swiglu_w4g64_bm64_acc16");
    m->_fn_qmm_swiglu8_bm64_a16 =
        m->_lib_qmm.function("affine_qmm_swiglu_w8g64_bm64_acc16");
    m->_fn_qmm_swiglu4_bm64_rs =
        m->_lib_qmm.function("affine_qmm_swiglu_w4g64_bm64_rs");
    m->_fn_qmm_swiglu8_bm64_rs =
        m->_lib_qmm.function("affine_qmm_swiglu_w8g64_bm64_rs");
    m->_fn_qmm_swiglu4_bm64_rs_a16 =
        m->_lib_qmm.function("affine_qmm_swiglu_w4g64_bm64_rs_acc16");
    m->_fn_qmm_swiglu8_bm64_rs_a16 =
        m->_lib_qmm.function("affine_qmm_swiglu_w8g64_bm64_rs_acc16");
  }
  // M5 matrix-core matmul2d for the block/projection GEMMs (mirrors Krea-2):
  // dense weights feed dense_gemm_mma directly; quantized weights dequant-
  // expand into _w_deq (affine_dequant) then run the SAME dense matmul2d --
  // dequant-once -> dense beats the fused steel qmm on matrix-core GPUs (the
  // qmm's tgmem staging cannot feed the MMA rate). Gated on matrix cores
  // (M4/older keep steel); VPIPE_FLUX2_NO_MMA2 A/B off. Decided BEFORE the
  // fused-FF choice below (the mma path defaults the fusion off).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_FLUX2_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    // TN=2 tile for the mid-K band (x-reuse doubles); NO_TN2 forces the plain
    // 128x256 deep tile for A/B (leaves the fn invalid -> routing skips it).
    if (std::getenv("VPIPE_FLUX2_NO_TN2") == nullptr) {
      m->_fn_dense_mma_tn2 =
          m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");
    }
    m->_fn_dense_mma_splitk =
        m->_lib_dense_mma.function("dense_gemm_mma_splitk_n128x256_k8192_f16");
    m->_use_mma2 =
        m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid();
    // Quantized checkpoint: group-matched dequant kernels feed the dense
    // matmul2d. Both bit widths loaded (mixed-precision per-weight w4/w8).
    if (m->_use_mma2 && m->_quant_bits > 0) {
      m->_lib_dequant = mc->load_library("affine_dequant");
      const std::string dg = "g" + std::to_string(m->_quant_group);
      m->_fn_dequant4 = m->_lib_dequant.function("affine_dequant_w4" + dg);
      m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant_w8" + dg);
      if (!m->_fn_dequant4.valid() || !m->_fn_dequant8.valid()) {
        m->_use_mma2 = false;   // fall back to steel qmm
      }
    }
    if (const char* e = std::getenv("VPIPE_FLUX2_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    m->_use_splitk = m->_use_mma2 && m->_fn_dense_mma_splitk.valid()
                     && m->_fn_residual.valid()
                     && std::getenv("VPIPE_FLUX2_NO_SPLITK") == nullptr;
  }
  // Dynamic-int8 accelerated GEMMs (opt-in, LOSSY): tried first in
  // gemm_mma_ for the big block matmuls. Config::i8_gemm is the stage
  // switch; VPIPE_I8_GEMM overrides (the context self-gates on matrix
  // cores + kernel availability).
  {
    auto i8 = std::make_unique<I8GemmContext>(mc, cfg.i8_gemm);
    if (i8->enabled()) { m->_i8 = std::move(i8); }
  }
  // Fuse the SwiGLU FF (default on, EXCEPT on the matmul2d path): needs the
  // dense swiglu twin, and -- for a quantized DiT -- the g64 qmm swiglu twins.
  // The weight dequant is F16 (the f16 metallib's native-half path).
  // Accumulate defaults to FLOAT: it is MORE accurate than the unfused path
  // (no [seq,2*INNER] f16 round-trip -> 4B golden 0.0036 vs unfused 0.0044) at
  // ~the same speed, and keeps clear of the golden bound. VPIPE_FLUX2_FF_ACC16
  // opts into the FP16-pipe accumulate (~1% faster, ~4x more drift);
  // VPIPE_FLUX2_NO_FUSE_FF disables the fusion entirely. On the mma path the
  // fused GEMM's register-local epilogue would keep the FF OFF the matrix
  // cores (steel rate), while the unfused two-GEMM+swiglu runs at the matmul2d
  // rate -- so _use_mma2 defaults the fusion off; VPIPE_FLUX2_FUSE_FF=1
  // forces it back on there (A/B).
  m->_ff_acc16 = std::getenv("VPIPE_FLUX2_FF_ACC16") != nullptr;
  m->_fuse_ff = std::getenv("VPIPE_FLUX2_NO_FUSE_FF") == nullptr
                && (!m->_use_mma2
                    || std::getenv("VPIPE_FLUX2_FUSE_FF") != nullptr)
                && m->_fn_ff_swiglu.valid() && m->_fn_ff_swiglu_a16.valid()
                && (m->_quant_bits == 0
                    || (m->_quant_group == 64
                        && m->_fn_qmm_swiglu4_bm64.valid()
                        && m->_fn_qmm_swiglu8_bm64.valid()));
  m->_lib_attn = mc->load_library("attn_steel");
  m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));
  m->_steel_attn_ok = m->_lib_attn.valid() && !m->_attn_params.empty()
                      && std::getenv("VPIPE_FLUX2_NO_STEEL_ATTN") == nullptr;
  m->_lib_attn_nax = mc->load_library("attn_steel_nax");
  m->_use_attn_nax = m->_steel_attn_ok && mc->supports_matrix_cores()
                     && m->_lib_attn_nax.valid()
                     && std::getenv("VPIPE_FLUX2_NO_ATTN_NAX") == nullptr;
  // Dense f16 STEEL GEMM tile (the fallback when the matmul2d route above is
  // off/absent). MEASURED on M4 (9B @1024, M=seq=4136): the dense f16 GEMM is
  // ALU/compute-bound (M amortizes weight reuse), so larger CTA tiles do NOT
  // help -- BM64 ran ~7% SLOWER and BM64xBN64 ~even vs the 32x32 base. So
  // default to the base tile. VPIPE_FLUX2_GEMM_TILE (0|1|2) +
  // VPIPE_FLUX2_GEMM_ACC16 keep the A/B knob.
  m->_gemm_tile = 0;
  if (const char* t = std::getenv("VPIPE_FLUX2_GEMM_TILE")) {
    m->_gemm_tile = std::atoi(t);
  }
  m->_acc16 = std::getenv("VPIPE_FLUX2_GEMM_ACC16") != nullptr;
  if (m->_gemm_tile == 1 && !m->_fn_gemm_bm64.valid()) { m->_gemm_tile = 0; }
  if (m->_gemm_tile == 2 && !m->_fn_gemm_bm64bn64.valid()) {
    m->_gemm_tile = m->_fn_gemm_bm64.valid() ? 1 : 0;
  }

  m->_x_embed   = m->load_qw_(wts, "x_embedder");
  m->_ctx_embed = m->load_qw_(wts, "context_embedder");
  m->_t_emb1 =
      m->load_qw_(wts, "time_guidance_embed.timestep_embedder.linear_1");
  m->_t_emb1_b =
      to_f16_(wts, mc, "time_guidance_embed.timestep_embedder.linear_1.bias");
  m->_t_emb2 =
      m->load_qw_(wts, "time_guidance_embed.timestep_embedder.linear_2");
  m->_t_emb2_b =
      to_f16_(wts, mc, "time_guidance_embed.timestep_embedder.linear_2.bias");
  // Guidance-distilled variants (klein-9B): the guidance_embedder is a second
  // TimestepEmbedding whose output is added to the timestep embedding. Absent
  // in the distilled 4B (guidance_embeds=false), so load only when configured.
  if (m->_cfg.guidance_embeds) {
    m->_g_emb1 =
        m->load_qw_(wts, "time_guidance_embed.guidance_embedder.linear_1");
    m->_g_emb1_b =
        to_f16_(wts, mc, "time_guidance_embed.guidance_embedder.linear_1.bias");
    m->_g_emb2 =
        m->load_qw_(wts, "time_guidance_embed.guidance_embedder.linear_2");
    m->_g_emb2_b =
        to_f16_(wts, mc, "time_guidance_embed.guidance_embedder.linear_2.bias");
    if (m->_g_emb1.empty() || m->_g_emb2.empty()) { return nullptr; }
  }
  m->_mod_img    = m->load_qw_(wts, "double_stream_modulation_img.linear");
  m->_mod_txt    = m->load_qw_(wts, "double_stream_modulation_txt.linear");
  m->_mod_single = m->load_qw_(wts, "single_stream_modulation.linear");
  m->_proj_out   = m->load_qw_(wts, "proj_out");
  m->_norm_out_lin = m->load_qw_(wts, "norm_out.linear");
  if (m->_x_embed.empty() || m->_ctx_embed.empty() || m->_t_emb1.empty() ||
      m->_t_emb2.empty() || m->_mod_img.empty() || m->_mod_txt.empty() ||
      m->_mod_single.empty() || m->_proj_out.empty() ||
      m->_norm_out_lin.empty()) {
    return nullptr;
  }
  if (m->_cfg.out_channels == 0) {
    m->_cfg.out_channels =
        m->_proj_out.n > 0 ? m->_proj_out.n : m->_cfg.in_channels;
  }

  const int H = m->_cfg.hidden;
  m->_ln_w1 = mc->make_shared_buffer((std::size_t)H * 2);
  m->_ln_b0 = mc->make_shared_buffer((std::size_t)H * 2);
  if (m->_ln_w1.empty() || m->_ln_b0.empty()) { return nullptr; }
  {
    auto* w1 = static_cast<_Float16*>(m->_ln_w1.contents());
    auto* b0 = static_cast<_Float16*>(m->_ln_b0.contents());
    for (int i = 0; i < H; ++i) {
      w1[i] = (_Float16)1.0f; b0[i] = (_Float16)0.0f;
    }
  }

  if (!stream_blocks) {
    m->_double.resize((std::size_t)m->_cfg.n_double);
    for (int i = 0; i < m->_cfg.n_double; ++i) {
      if (!m->load_double_(wts, "transformer_blocks." + std::to_string(i) + ".",
                           m->_double[(std::size_t)i])) {
        if (mc->session() != nullptr) {
          mc->session()->warn(fmt(
              "MetalFlux2Transformer: failed to load double block {}", i));
        }
        return nullptr;
      }
    }
    m->_single.resize((std::size_t)m->_cfg.n_single);
    for (int i = 0; i < m->_cfg.n_single; ++i) {
      if (!m->load_single_(
              wts, "single_transformer_blocks." + std::to_string(i) + ".",
              m->_single[(std::size_t)i])) {
        if (mc->session() != nullptr) {
          mc->session()->warn(fmt(
              "MetalFlux2Transformer: failed to load single block {}", i));
        }
        return nullptr;
      }
    }
  } else {
    // Streaming: blocks load JIT in forward_dit. Still derive the FF dims that
    // load_double_/load_single_ would set -- from the tensor SHAPES (info only,
    // no weight load) so forward_dit has DFF / single_mlp_in.
    if (m->_cfg.double_ff_hidden == 0) {
      const auto* fi = wts.info("transformer_blocks.0.ff.linear_in.weight");
      if (fi != nullptr && !fi->shape.empty()) {
        m->_cfg.double_ff_hidden = (int)fi->shape[0];
      }
    }
    if (m->_cfg.single_mlp_in == 0) {
      const auto* qi =
          wts.info("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight");
      if (qi != nullptr && !qi->shape.empty()) {
        const int rest = (int)qi->shape[0] - 3 * m->_cfg.hidden;
        if (rest > 0) { m->_cfg.single_mlp_in = rest / 2; }
      }
    }
    if (m->_cfg.double_ff_hidden == 0 || m->_cfg.single_mlp_in == 0) {
      if (mc->session() != nullptr) {
        mc->session()->warn(fmt(
            "MetalFlux2Transformer: streaming -- could not derive FF dims"));
      }
      return nullptr;
    }
    // Pinned-prefix: pin as many LEADING blocks as fit in pin_frac of RAM, in
    // stream order (double blocks first, then single). Greedy over the actual
    // per-block bytes (double blocks are larger). Loaded from `wts` before it is
    // moved into _stream_wts; the pinned buffers survive the move.
    if (pin_frac > 0.0) {
      std::vector<std::string> prefixes;
      prefixes.reserve((std::size_t)(m->_cfg.n_double + m->_cfg.n_single));
      for (int i = 0; i < m->_cfg.n_double; ++i) {
        prefixes.push_back("transformer_blocks." + std::to_string(i) + ".");
      }
      for (int i = 0; i < m->_cfg.n_single; ++i) {
        prefixes.push_back("single_transformer_blocks." + std::to_string(i) +
                           ".");
      }
      int pin = stream_pin_count(wts, prefixes, pin_frac);
      if (pin > (int)prefixes.size()) { pin = (int)prefixes.size(); }
      m->_pinned_d = pin < m->_cfg.n_double ? pin : m->_cfg.n_double;
      m->_pinned_s = pin - m->_pinned_d;
      m->_double.resize((std::size_t)m->_pinned_d);
      for (int i = 0; i < m->_pinned_d; ++i) {
        if (!m->load_double_(wts,
                "transformer_blocks." + std::to_string(i) + ".",
                m->_double[(std::size_t)i])) {
          return nullptr;
        }
      }
      m->_single.resize((std::size_t)m->_pinned_s);
      for (int i = 0; i < m->_pinned_s; ++i) {
        if (!m->load_single_(wts,
                "single_transformer_blocks." + std::to_string(i) + ".",
                m->_single[(std::size_t)i])) {
          return nullptr;
        }
      }
    }
    // Retain the source mmap so forward_dit can re-read each block on demand.
    m->_stream_wts = std::make_unique<MetalLlamaWeights>(std::move(*wtsopt));
    if (mc->session() != nullptr) {
      mc->session()->info(fmt(
          "MetalFlux2Transformer: streaming {}+{} blocks (memory-bounded)",
          m->_cfg.n_double, m->_cfg.n_single));
    }
  }
  return m;
}

// Build the joint [text; image(+refs)] 4-axis RoPE cos/sin tables [seq,
// head_dim]. Text tokens sit at the origin on axis-3 (0,0,0,l); the generated
// image carries (0, row, col, 0); each reference image carries (T, row, col, 0)
// with T its per-reference index band. Adjacent-pair layout matches
// rope_pair_table_f16.
// NOTE: the axis->coordinate mapping is a best reading of Flux2PosEmbed; VERIFY
// against a diffusers golden.
void
MetalFlux2Transformer::build_rope_tables_(int text_seq,
                                          const std::vector<ImgSeg>& segs,
                                          SharedBuffer& cos_out,
                                          SharedBuffer& sin_out)
{
  int img_seq = 0;
  for (const auto& sg : segs) { img_seq += sg.seq; }
  const int seq = text_seq + img_seq;
  const int HD = _cfg.head_dim;
  const int pairs = HD / 2;
  cos_out = _mc->make_shared_buffer((std::size_t)seq * HD * 2);
  sin_out = _mc->make_shared_buffer((std::size_t)seq * HD * 2);
  auto* c = static_cast<_Float16*>(cos_out.contents());
  auto* s = static_cast<_Float16*>(sin_out.contents());
  const float theta = _cfg.rope_theta;
  // Flux2 position ids (T, H, W, L). axes_dim [32,32,32,32] -> axis0=T, axis1=H,
  // axis2=W, axis3=L. Emit one token row from its 4 coordinates.
  auto emit = [&](int t, float p0, float p1, float p2, float p3) {
    const float pos[4] = {p0, p1, p2, p3};
    int pair = 0;
    for (int a = 0; a < 4; ++a) {
      const int adim = _cfg.axes_dim[a];
      const int apairs = adim / 2;
      for (int j = 0; j < apairs && pair < pairs; ++j, ++pair) {
        const float freq = 1.0f / std::pow(theta, (float)(2 * j) / (float)adim);
        const float ang = pos[a] * freq;
        c[(std::size_t)t * HD + 2 * pair]     = (_Float16)std::cos(ang);
        c[(std::size_t)t * HD + 2 * pair + 1] = (_Float16)std::cos(ang);
        s[(std::size_t)t * HD + 2 * pair]     = (_Float16)std::sin(ang);
        s[(std::size_t)t * HD + 2 * pair + 1] = (_Float16)std::sin(ang);
      }
    }
  };
  int t = 0;
  for (; t < text_seq; ++t) { emit(t, 0, 0, 0, (float)t); }   // text: axis-3 = l
  for (const auto& sg : segs) {                               // image + refs
    for (int p = 0; p < sg.seq; ++p, ++t) {
      emit(t, (float)sg.t_off, (float)(p / sg.grid_w),
           (float)(p % sg.grid_w), 0.0f);
    }
  }
}

void
MetalFlux2Transformer::calib_begin()
{
  _calib_acc.clear();
  const int H = _cfg.hidden;
  const int INNER = _cfg.double_ff_hidden / 2;
  const int SMLP = _cfg.single_mlp_in;
  const int nD = _cfg.n_double, nS = _cfg.n_single;
  auto add = [&](const char* g, int rows, int dim) {
    SharedBuffer b = _mc->make_shared_buffer((std::size_t)rows * dim * 2);
    if (!b.empty()) { std::memset(b.contents(), 0, b.byte_size()); }
    _calib_acc[g] = std::move(b);
  };
  add("dbl_norm1_img", nD, H);   add("dbl_norm1_txt", nD, H);
  add("dbl_attn_img", nD, H);    add("dbl_attn_txt", nD, H);
  add("dbl_norm2_img", nD, H);   add("dbl_ffact_img", nD, INNER);
  add("dbl_norm2_txt", nD, H);   add("dbl_ffact_txt", nD, INNER);
  add("sgl_norm", nS, H);        add("sgl_cat", nS, H + SMLP);
  add("emb_x", 1, _cfg.in_channels);
  add("emb_ctx", 1, _cfg.joint_dim);
  add("emb_proj", 1, H);
  _calib_on = true;
}

std::map<std::string, std::vector<float>>
MetalFlux2Transformer::calib_stats() const
{
  std::map<std::string, std::vector<float>> out;
  for (const auto& kv : _calib_acc) {
    const std::size_t n = kv.second.empty() ? 0 : kv.second.byte_size() / 2;
    std::vector<float> v(n);
    const auto* s = static_cast<const _Float16*>(kv.second.contents());
    for (std::size_t i = 0; i < n; ++i) { v[i] = (float)s[i]; }
    out[kv.first] = std::move(v);
  }
  return out;
}

bool
MetalFlux2Transformer::gemm_mma_(ComputeEncoder& enc, const SharedBuffer& x,
                                 std::size_t xe, const QWeight& w,
                                 const SharedBuffer& y, std::size_t ye,
                                 int M, int N, int K)
{
  // Matrix-core matmul2d only when present, M amortizes the 128-row tile, and
  // N is non-degenerate (the M=1 conditioning GEMMs stay on steel).
  if (!_use_mma2 || M < _mma_min_m || N < 16) { return false; }
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
    // codes/scales/qbias -> _w_deq[N,K] (one thread per packed u32 word: w4
    // has 8 nibbles/word so K/8 words, w8 has 4 bytes/word so K/4). The
    // forward's streams commit serially (one live encoder), so Metal's WAR
    // hazard tracking makes the shared _w_deq safe to reuse across GEMMs
    // (each dequant->matmul pair runs before the next dequant overwrites).
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
  // non-qualifying shapes (small M, K not a 512-multiple).
  if (_i8 && _i8->gemm(enc, x, xe, *wdense, y, ye, M, N, K)) {
    return true;
  }
  // Split-K deep reduction for the very deep K (the single-stream to_out,
  // K = H + SMLP = 16384 on the 9B): the single-op full reduction sits on the
  // deep-K cliff (~0.7x the K<=9728 rate; see Krea-2). Fire when K is >= 2
  // exact chunks of kSplitKC; each of the S = K/kSplitKC splits gets its own
  // threadgroup plane (grid.z), then residual_add folds the planes into y.
  // The extra f16 rounding per fold is fine for the rel-L2-verified DiT.
  // Falls through to the single-op path when the scratch alloc fails.
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
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, *wdense);
      enc.set_buffer(2, _splitk);
      enc.set_constant(3, K); enc.set_constant(4, N); enc.set_constant(5, M);
      enc.dispatch({(unsigned)(((N + 255) / 256) * 256),   // BM=128, BN=256
                    (unsigned)((M + 127) / 128), (unsigned)splits},
                   {256, 1, 1});
      // Fold the S partial planes: y = plane0 + plane1 (+ plane_s ...).
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
  // Tile-adaptive dense matmul2d (no bias slot; klein's block Linears are
  // bias-free and the conditioning biases are folded by the caller's bias()).
  // Krea-2-tuned routing: 128x128 for K < 6144 (the H=4096/5120 projections),
  // the TN=2 (128x512-region) tile for the mid-K band, plain 128x256 for
  // deeper unsplit K (the split-K path above normally owns that regime).
  int RN = 256;   // effective N-region per tg (TN*BN); grid divides N by it
  const metal_compute::ComputeFunction* fn = &_fn_dense_mma_deep;
  if (K < 6144) {
    fn = &_fn_dense_mma; RN = 128;
  } else if (K < 12288 && _fn_dense_mma_tn2.valid()) {
    fn = &_fn_dense_mma_tn2; RN = 512;   // TN=2: two 256-wide N-tiles per tg
  }
  enc.set_function(*fn);
  enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, *wdense);
  enc.set_buffer(2, *wdense);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  return true;
}

SharedBuffer
MetalFlux2Transformer::forward_dit(const SharedBuffer& context, int text_seq,
                                   const SharedBuffer& latents, int img_seq,
                                   int grid_h, int grid_w, float timestep,
                                   float guidance,
                                   const std::vector<RefImage>& refs)
{
  const Config& c = _cfg;
  const int H = c.hidden, HED = c.n_heads, HD = c.head_dim;
  const int IC = c.in_channels, TD = c.timestep_dim, OC = c.out_channels;
  const int DFF = c.double_ff_hidden, SMLP = c.single_mlp_in;
  const int INNER = DFF / 2;   // Flux2FeedForward linear_in -> 2*inner SwiGLU
  const float eps = c.norm_eps;
  // The image token stream is [generated; ref0; ref1; ...]: IS_GEN generated
  // tokens (the only ones we predict velocity for) followed by the reference
  // tokens. IS is the full image-token count that drives every block; IS_GEN
  // is what proj_out emits.
  const int TS = text_seq, IS_GEN = img_seq;
  int IS_REF = 0;
  for (const auto& r : refs) { IS_REF += r.seq; }
  const int IS = IS_GEN + IS_REF, seq = TS + IS;
  // Flux2 reference-image T (time/index) coordinate step (_prepare_image_ids
  // scale): ref i -> T = kRefPosScale*(i+1) so each reference sits in its own
  // position band, distinct from the generated image (T=0).
  constexpr int kRefPosScale = 10;
  if (TS <= 0 || IS_GEN <= 0 || DFF <= 0 || SMLP <= 0
      || context.byte_size() < (std::size_t)TS * c.joint_dim * 2
      || latents.byte_size() < (std::size_t)IS_GEN * IC * 2) {
    return {};
  }
  for (const auto& r : refs) {
    if (r.seq <= 0 || r.latents.byte_size() < (std::size_t)r.seq * IC * 2) {
      return {};
    }
  }
  const int PW = 3 * H + 2 * SMLP;                 // to_qkv_mlp_proj out width
  const float scale = 1.0f / std::sqrt((float)HD);

  // LLM-lane perf event (perf-visualizer): one DiT forward per sampler step,
  // value = the joint sequence length. Mirrors the Krea-2 DiT event.
  PerfAuxScope _perf(_mc->session(), kPerfLaneLLM, kGvidLlmDit,
                     kPerfLlmDitBegin, (std::uint64_t)seq);

  // Per-section GPU timing (VPIPE_FLUX2_DIT_PROFILE). Marks + commit-boundary
  // waits split the deferred streams into timed slices; single-block attention
  // is isolated from its GEMMs by an extra barrier. The barriers serialize
  // (removing any overlap) but the per-section GPU wall time is what we want.
  // No effect unless the env var is set.
  const bool prof = std::getenv("VPIPE_FLUX2_DIT_PROFILE") != nullptr;
  double t_dbl = 0, t_sgl_gemm = 0, t_sgl_attn = 0, t_sgl_cat = 0,
         t_sgl_out = 0, t_final = 0;
  auto tnow = [] { return std::chrono::steady_clock::now(); };
  auto ms_since = [](std::chrono::steady_clock::time_point m) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - m).count();
  };
  std::chrono::steady_clock::time_point mk;

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer rcos, rsin;
  {
    // Segment 0 = generated (T=0); one segment per reference at its own T band.
    std::vector<ImgSeg> segs;
    segs.push_back({0, grid_h, grid_w, IS_GEN});
    for (int i = 0; i < (int)refs.size(); ++i) {
      segs.push_back({kRefPosScale * (i + 1), refs[i].grid_h, refs[i].grid_w,
                      refs[i].seq});
    }
    build_rope_tables_(TS, segs, rcos, rsin);
  }

  SharedBuffer te_in = buf((std::size_t)TD);
  {
    auto* ti = static_cast<_Float16*>(te_in.contents());
    const int half = TD / 2;
    for (int i = 0; i < half; ++i) {
      const double fr = std::exp(-std::log(1e4) * (double)i / (double)half);
      const double ang = (double)timestep * 1000.0 * fr;
      ti[i] = (_Float16)std::cos(ang);
      ti[half + i] = (_Float16)std::sin(ang);
    }
  }
  // Embedded guidance (guidance-distilled klein-9B): the same cos-first
  // sinusoid of guidance*1000, embedded by guidance_embedder and added to the
  // timestep embedding. Skipped when the model has no guidance_embeds or the
  // caller passes guidance < 0.
  const bool use_g = c.guidance_embeds && guidance >= 0.0f && !_g_emb1.empty();
  SharedBuffer ge_in;
  if (use_g) {
    ge_in = buf((std::size_t)TD);
    auto* gi = static_cast<_Float16*>(ge_in.contents());
    const int half = TD / 2;
    for (int i = 0; i < half; ++i) {
      const double fr = std::exp(-std::log(1e4) * (double)i / (double)half);
      const double ang = (double)guidance * 1000.0 * fr;
      gi[i] = (_Float16)std::cos(ang);
      gi[half + i] = (_Float16)std::sin(ang);
    }
  }

  // Scratch.
  SharedBuffer temb = buf((std::size_t)H), tsilu = buf((std::size_t)H),
               te1 = buf((std::size_t)H);
  SharedBuffer ge1 = buf((std::size_t)H), gemb = buf((std::size_t)H);
  SharedBuffer mimg = buf((std::size_t)6 * H), mtxt = buf((std::size_t)6 * H),
               msin = buf((std::size_t)3 * H);
  SharedBuffer img = buf((std::size_t)IS * H), txt = buf((std::size_t)TS * H);
  SharedBuffer nrm = buf((std::size_t)seq * H);
  SharedBuffer jq = buf((std::size_t)seq * H), jk = buf((std::size_t)seq * H),
               jv = buf((std::size_t)seq * H);
  SharedBuffer qt = buf((std::size_t)HED * seq * HD),
               kt = buf((std::size_t)HED * seq * HD),
               vt = buf((std::size_t)HED * seq * HD),
               atb = buf((std::size_t)HED * seq * HD);
  SharedBuffer att = buf((std::size_t)seq * H), ob = buf((std::size_t)seq * H);
  SharedBuffer ff1 = buf((std::size_t)seq * DFF);
  SharedBuffer joint = buf((std::size_t)seq * H);
  SharedBuffer sproj = buf((std::size_t)seq * PW);
  SharedBuffer sg = buf((std::size_t)seq * SMLP),
               su = buf((std::size_t)seq * SMLP),
               smlp = buf((std::size_t)seq * SMLP);
  SharedBuffer scat = buf((std::size_t)seq * ((std::size_t)H + SMLP));
  SharedBuffer velocity = buf((std::size_t)IS_GEN * OC);

  // ---- helper wrappers over a live ComputeEncoder (redefined per stream) ----
  auto make_ops = [&](ComputeEncoder& enc) {
    struct Ops {
      MetalFlux2Transformer* self;
      ComputeEncoder* e;
      SharedBuffer *rcos, *rsin;
      float eps;
      // y[M,N] (elem offset ye) = x[M,K] (elem offset xe) @ W[N,K]^T.
      void gemm(const SharedBuffer& x, const QWeight& w, const SharedBuffer& y,
                std::size_t ye, int M, int N, int K, std::size_t xe = 0) {
        // Matrix-core matmul2d first (M5); false -> steel below.
        if (self->gemm_mma_(*e, x, xe, w, y, ye, M, N, K)) { return; }
        int bm = 32, bn = 32;                    // 32x32 base tile
        if (w.quantized) {
          // BM128 tile once M amortizes the 128-row re-use (the DiT block GEMMs
          // at high res, M = seq >= 1024); small-M keeps the base tile.
          const bool huge = self->_qmm_tile >= 1 && M >= 1024
                            && self->_fn_qmm4_bm128.valid();
          if (huge) { bm = 128; }
          e->set_function(w.bits == 8
              ? (huge ? self->_fn_qmm8_bm128 : self->_fn_qmm8)
              : (huge ? self->_fn_qmm4_bm128 : self->_fn_qmm4));
          e->set_buffer(0, w.codes); e->set_buffer(1, w.scales);
          e->set_buffer(2, w.qbias); e->set_buffer(3, x, xe * 2);
          e->set_buffer(4, y, ye * 2);
          e->set_constant(5, K); e->set_constant(6, N); e->set_constant(7, M);
        } else {
          // Larger tiles for the big M = seq GEMMs (fewer weight re-reads).
          const metal_compute::ComputeFunction* f = &self->_fn_gemm;
          if (M >= 128 && self->_gemm_tile == 1) {
            f = (self->_acc16 && self->_fn_gemm_bm64_a16.valid())
                    ? &self->_fn_gemm_bm64_a16 : &self->_fn_gemm_bm64;
            bm = 64;
          } else if (M >= 128 && self->_gemm_tile == 2) {
            f = &self->_fn_gemm_bm64bn64; bm = 64; bn = 64;
          }
          e->set_function(*f);
          e->set_buffer(0, x, xe * 2); e->set_buffer(1, w.w);
          e->set_buffer(2, w.w);
          e->set_buffer(3, y, ye * 2);
          e->set_constant(4, K); e->set_constant(5, N); e->set_constant(6, M);
          e->set_constant(7, 0);
        }
        // BM=128 uses WM=4 (256 threads): threadgroup {32,2,4}, grid z=4.
        const unsigned tgz = (bm == 128) ? 4u : 2u;
        e->dispatch({(unsigned)(((N + bn - 1) / bn) * 32),
                     (unsigned)(((M + bm - 1) / bm) * 2), tgz}, {32, 2, tgz});
      }
      // Fused SwiGLU FF: out[M, Nf/2] = silu(gate)*up from the INTERLEAVED
      // [Nf, K] linear_in weight (Nf = 2*INNER). One BM64 GEMM whose register-
      // local epilogue writes silu(gate)*up -- no [M, Nf] intermediate + no
      // slice/swiglu passes. Dense or quant (w4/w8), float or acc16.
      // out_stride/out_off (dense only): write silu(gate)*up into out[:, off:]
      // of a wider buffer (row stride out_stride) so the single block can drop
      // the [att|mlp] concat. The quant kernels are shared (no strided output),
      // so the quant path must pass out_stride 0 (contiguous) + concat.
      void swiglu_ff(const SharedBuffer& x, const QWeight& w,
                     const SharedBuffer& out, int M, int K, int Nf,
                     int out_stride = 0, int out_off = 0) {
        const bool a16 = self->_ff_acc16;
        const bool strided = out_stride > 0;
        if (w.quantized) {
          e->set_function(strided
              ? (w.bits == 8
                     ? (a16 ? self->_fn_qmm_swiglu8_bm64_rs_a16
                            : self->_fn_qmm_swiglu8_bm64_rs)
                     : (a16 ? self->_fn_qmm_swiglu4_bm64_rs_a16
                            : self->_fn_qmm_swiglu4_bm64_rs))
              : (w.bits == 8
                     ? (a16 ? self->_fn_qmm_swiglu8_bm64_a16
                            : self->_fn_qmm_swiglu8_bm64)
                     : (a16 ? self->_fn_qmm_swiglu4_bm64_a16
                            : self->_fn_qmm_swiglu4_bm64)));
          e->set_buffer(0, w.codes); e->set_buffer(1, w.scales);
          e->set_buffer(2, w.qbias); e->set_buffer(3, x); e->set_buffer(4, out);
          e->set_constant(5, K); e->set_constant(6, Nf); e->set_constant(7, M);
          if (strided) {
            e->set_constant(8, out_stride); e->set_constant(9, out_off);
          }
        } else {
          e->set_function(a16 ? self->_fn_ff_swiglu_a16 : self->_fn_ff_swiglu);
          e->set_buffer(0, x); e->set_buffer(1, w.w); e->set_buffer(2, out);
          e->set_constant(3, K); e->set_constant(4, Nf); e->set_constant(5, M);
          e->set_constant(6, out_stride); e->set_constant(7, out_off);
        }
        e->dispatch({(unsigned)(((Nf + 31) / 32) * 32),
                     (unsigned)(((M + 63) / 64) * 2), 2}, {32, 2, 2});
      }
      // Strided unfused SwiGLU: silu(gate)*up (gate/up contiguous [rows,width])
      // -> out[row*out_stride + out_off + col].
      void swiglu_rs(const SharedBuffer& gate, const SharedBuffer& up,
                     const SharedBuffer& out, int rows, int width,
                     int out_stride, int out_off) {
        e->set_function(self->_fn_swiglu_rs);
        e->set_buffer(0, gate); e->set_buffer(1, up); e->set_buffer(2, out);
        e->set_constant(3, rows); e->set_constant(4, width);
        e->set_constant(5, out_stride); e->set_constant(6, out_off);
        e->dispatch({(unsigned)(rows * width), 1, 1}, {256, 1, 1});
      }
      // AWQ calib tap: acc[group] row L (dim) max-accumulates the per-column
      // |activation| over M rows of `in` (elem offset xe). No-op when calib off.
      void tap(const char* group, int L, const SharedBuffer& in, std::size_t xe,
               int M, int dim) {
        if (!self->_calib_on) { return; }
        auto it = self->_calib_acc.find(group);
        if (it == self->_calib_acc.end() || it->second.empty()) { return; }
        e->set_function(self->_fn_colabsmax);
        e->set_buffer(0, in, xe * 2);
        e->set_buffer(1, it->second, (std::size_t)L * dim * 2);
        e->set_constant(2, M); e->set_constant(3, dim);
        e->dispatch({(unsigned)dim, 1, 1}, {256, 1, 1});
      }
      void bias(const SharedBuffer& bs, const SharedBuffer& y, int M, int N) {
        if (bs.empty()) { return; }   // klein is bias-free (bias=False)
        e->set_function(self->_fn_bias_add);
        e->set_buffer(0, y); e->set_buffer(1, bs);
        e->set_constant(2, N); e->set_constant(3, M * N);
        e->dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
      }
      void rms(const SharedBuffer& x, std::size_t xe, const SharedBuffer& w,
               const SharedBuffer& y, std::size_t ye, int R, int Hd) {
        e->set_function(self->_fn_rms);
        e->set_buffer(0, x, xe * 2); e->set_buffer(1, w);
        e->set_buffer(2, y, ye * 2);
        e->set_constant(3, Hd); e->set_constant(4, eps);
        e->dispatch({256, (unsigned)R, 1}, {256, 1, 1});
      }
      void ln(const SharedBuffer& x, std::size_t xe, const SharedBuffer& y,
              std::size_t ye, int R, int Hd) {
        e->set_function(self->_fn_layernorm);
        e->set_buffer(0, x, xe * 2); e->set_buffer(1, self->_ln_w1);
        e->set_buffer(2, self->_ln_b0); e->set_buffer(3, y, ye * 2);
        e->set_constant(4, Hd); e->set_constant(5, eps);
        e->dispatch({256, (unsigned)R, 1}, {256, 1, 1});
      }
      void tr(const SharedBuffer& in, std::size_t ie, const SharedBuffer& out,
              std::size_t oe, int A, int Bd, int D) {
        e->set_function(self->_fn_transpose);
        e->set_buffer(0, in, ie * 2); e->set_buffer(1, out, oe * 2);
        e->set_constant(2, A); e->set_constant(3, Bd); e->set_constant(4, D);
        e->dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A},
                    {(unsigned)D, 1, 1});
      }
      // transpose with an output ROW-STRIDE: out[b*out_rs + a*D + d] -- writes
      // [Bd, A*D] as columns [0:A*D] of a wider buffer (e.g. att -> scat[:, :H]).
      void tr_rs(const SharedBuffer& in, const SharedBuffer& out, int A, int Bd,
                 int D, int out_rs) {
        e->set_function(self->_fn_transpose_rs);
        e->set_buffer(0, in); e->set_buffer(1, out);
        e->set_constant(2, A); e->set_constant(3, Bd); e->set_constant(4, D);
        e->set_constant(5, out_rs);
        e->dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A},
                    {(unsigned)D, 1, 1});
      }
      void rope(const SharedBuffer& x, int nh, int T, int D) {
        e->set_function(self->_fn_rope_table);
        e->set_buffer(0, x); e->set_buffer(1, *rcos); e->set_buffer(2, *rsin);
        e->set_constant(3, nh); e->set_constant(4, T); e->set_constant(5, D);
        e->dispatch({(unsigned)(D / 2), (unsigned)T, (unsigned)nh},
                    {(unsigned)(D / 2), 1, 1});
      }
      void sdpa(const SharedBuffer& q, const SharedBuffer& k,
                const SharedBuffer& v, const SharedBuffer& out, float sc, int T,
                int D, int nh) {
        e->set_function(self->_fn_sdpa);
        e->set_buffer(0, q); e->set_buffer(1, k); e->set_buffer(2, v);
        e->set_buffer(3, out);
        e->set_constant(4, sc); e->set_constant(5, T); e->set_constant(6, D);
        e->set_constant(7, nh); e->set_constant(8, nh); e->set_constant(9, T);
        e->set_constant(10, T);
        e->dispatch({32, (unsigned)nh, (unsigned)T}, {32, 1, 1});
      }
      // out[r, 0:W] = in[r*S + off : +W]  (strided column slice; block=0).
      void slice(const SharedBuffer& in, const SharedBuffer& out, int rows,
                 int S, int W, int off) {
        e->set_function(self->_fn_headslice);
        e->set_buffer(0, in); e->set_buffer(1, out);
        e->set_constant(2, rows); e->set_constant(3, S); e->set_constant(4, W);
        e->set_constant(5, off); e->set_constant(6, 0); e->set_constant(7, 0);
        e->dispatch({(unsigned)(rows * W), 1, 1}, {256, 1, 1});
      }
      // GPU [a | b] row-wise column concat -> dst[rows, wa+wb] (in-stream).
      void concat_cols(const SharedBuffer& dst, const SharedBuffer& a,
                       const SharedBuffer& b, int rows, int wa, int wb) {
        e->set_function(self->_fn_concat);
        e->set_buffer(0, a); e->set_buffer(1, b); e->set_buffer(2, dst);
        e->set_constant(3, rows); e->set_constant(4, wa);
        e->set_constant(5, wb);
        e->dispatch({(unsigned)(rows * (wa + wb)), 1, 1}, {256, 1, 1});
      }
      void elt(const ComputeFunction& fn, const SharedBuffer& a, std::size_t ae,
               const SharedBuffer& b, std::size_t be, const SharedBuffer& out,
               std::size_t oe, int nn) {
        e->set_function(fn);
        e->set_buffer(0, a, ae * 2); e->set_buffer(1, b, be * 2);
        e->set_buffer(2, out, oe * 2); e->set_constant(3, nn);
        e->dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
      }
      void adaln(const SharedBuffer& x, std::size_t xe, const SharedBuffer& mod,
                 std::size_t sc_e, std::size_t sh_e, const SharedBuffer& out,
                 std::size_t oe, int N, int total) {
        e->set_function(self->_fn_adaln);
        e->set_buffer(0, x, xe * 2); e->set_buffer(1, mod, sc_e * 2);
        e->set_buffer(2, mod, sh_e * 2); e->set_buffer(3, out, oe * 2);
        e->set_constant(4, N); e->set_constant(5, total);
        e->dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
      }
      void gated(const SharedBuffer& h, std::size_t he, const SharedBuffer& mod,
                 std::size_t g_e, const SharedBuffer& sub, std::size_t se,
                 int N, int total) {
        e->set_function(self->_fn_gated);
        e->set_buffer(0, h, he * 2); e->set_buffer(1, mod, g_e * 2);
        e->set_buffer(2, sub, se * 2);
        e->set_constant(3, N); e->set_constant(4, total);
        e->dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
      }
    };
    return Ops{this, &enc, &rcos, &rsin, eps};
  };

  // Steel flash-attention setup (built ONCE; seq is constant across all blocks
  // of a forward). FLUX.2 attention is MHA -- 32 q/k/v heads, head_dim 128, no
  // GQA (gqa_factor 1) -- so Q/K/V/O are all [HED, seq, HD] (the head-major
  // transposes below produce exactly that). Register-resident flash attention
  // is O(seq) memory + tiled, vs the scalar sdpa_full_f16's O(seq^2) inner loop
  // (the ~83% bottleneck at 1024px). Falls back to scalar sdpa when the steel
  // library / function is unavailable. On matrix-core GPUs (M5) prefer the NAX
  // variant. Params written once on CPU, read by every block's GPU dispatch.
  const int KVH = HED;                          // MHA: kv heads == q heads
  const bool nax = _use_attn_nax && _lib_attn_nax.valid();
  const int A_BQ = nax ? 64 : 32;
  const int A_BK = nax ? 32 : 16;
  metal_compute::ComputeFunction fn_attn;
  bool use_steel = _steel_attn_ok && !_attn_params.empty();
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
    fn_attn = nax ? _lib_attn_nax.function("attn_steel_nax_h_bd128", fc)
                  : _lib_attn.function("attn_steel_h_bd128", fc);
    use_steel = fn_attn.valid();
  }
  const unsigned a_nqb = (unsigned)((seq + A_BQ - 1) / A_BQ);

  // Joint self-attention over jq/jk/jv [seq, H] (already q/k RMSNorm'd). head-
  // major transpose -> rope -> flash attn -> transpose the result into `out`
  // (row stride out_rs; out_rs > 0 writes a sub-view, e.g. att -> scat[:, :H]).
  auto attention = [&](auto& op, const SharedBuffer& out, int out_rs) {
    op.tr(jq, 0, qt, 0, seq, HED, HD);
    op.tr(jk, 0, kt, 0, seq, HED, HD);
    op.tr(jv, 0, vt, 0, seq, HED, HD);
    op.rope(qt, HED, seq, HD);
    op.rope(kt, HED, seq, HD);
    if (use_steel) {
      op.e->set_function(fn_attn);
      op.e->set_buffer(0, qt); op.e->set_buffer(1, kt); op.e->set_buffer(2, vt);
      op.e->set_buffer(3, atb); op.e->set_buffer(4, _attn_params);
      op.e->dispatch({32 * a_nqb, 4 * (unsigned)HED, 1}, {32, 4, 1});
    } else {
      op.sdpa(qt, kt, vt, atb, scale, seq, HD, HED);
    }
    if (out_rs > 0) {
      op.tr_rs(atb, out, HED, seq, HD, out_rs);
    } else {
      op.tr(atb, 0, out, 0, HED, seq, HD);
    }
  };

  // ===== stream 1: conditioning + embed + double blocks =====
  if (prof) { mk = tnow(); }
  {
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    auto op = make_ops(enc);
    // Streaming: commit+reopen the stream at block boundaries so a just-in-time
    // block's weights are GPU-done before it is freed. No-op when preloaded.
    auto flush = [&]() {
      enc.end(); stream.commit().wait();
      stream = _mc->make_command_stream(); enc = stream.begin_compute();
      op = make_ops(enc);
    };
    // TimestepEmbedding: linear_1 -> SiLU -> linear_2.
    op.gemm(te_in, _t_emb1, te1, 0, 1, H, TD); op.bias(_t_emb1_b, te1, 1, H);
    op.elt(_fn_mulsig, te1, 0, te1, 0, te1, 0, H);       // SiLU
    op.gemm(te1, _t_emb2, temb, 0, 1, H, H); op.bias(_t_emb2_b, temb, 1, H);
    // Embedded guidance: guidance_embedder(SiLU) then temb += guidance_emb.
    if (use_g) {
      op.gemm(ge_in, _g_emb1, ge1, 0, 1, H, TD); op.bias(_g_emb1_b, ge1, 1, H);
      op.elt(_fn_mulsig, ge1, 0, ge1, 0, ge1, 0, H);       // SiLU
      op.gemm(ge1, _g_emb2, gemb, 0, 1, H, H); op.bias(_g_emb2_b, gemb, 1, H);
      op.elt(_fn_residual, temb, 0, gemb, 0, temb, 0, H);  // += guidance_emb
    }
    // Shared modulation: linear(SiLU(temb)).
    op.elt(_fn_mulsig, temb, 0, temb, 0, tsilu, 0, H);
    op.gemm(tsilu, _mod_img, mimg, 0, 1, 6 * H, H);
    op.gemm(tsilu, _mod_txt, mtxt, 0, 1, 6 * H, H);
    op.gemm(tsilu, _mod_single, msin, 0, 1, 3 * H, H);
    // Embed image + text. The generated tokens embed into img[0:IS_GEN]; each
    // reference embeds (same x_embedder) into the tail img[IS_GEN..IS]. (Calib
    // taps the generated tokens only -- calibration runs text-only, refs empty.)
    op.tap("emb_x", 0, latents, 0, IS_GEN, IC);
    op.gemm(latents, _x_embed, img, 0, IS_GEN, H, IC);
    {
      std::size_t ro = (std::size_t)IS_GEN;   // ref token offset into img
      for (const auto& r : refs) {
        op.gemm(r.latents, _x_embed, img, ro * H, r.seq, H, IC);
        ro += (std::size_t)r.seq;
      }
    }
    op.tap("emb_ctx", 0, context, 0, TS, c.joint_dim);
    op.gemm(context, _ctx_embed, txt, 0, TS, H, c.joint_dim);
    if (_stream_blocks) { flush(); }   // commit conditioning before streaming

    for (int L = 0; L < c.n_double; ++L) {
      // Pipeline stop -> abandon: checked EVERY block (not just the streamed
      // tail) so a slow high-res step responds within ~one block on the
      // preloaded path too.
      if (_stream_stop && _stream_stop()) { return {}; }
      // Pinned prefix (L < _pinned_d) is resident in _double; the tail streams.
      const bool streaming = _stream_blocks && L >= _pinned_d;
      DoubleBlock streamed;
      if (streaming) {
        if (!load_double_(*_stream_wts,
                          "transformer_blocks." + std::to_string(L) + ".",
                          streamed)) {
          return {};
        }
      }
      const DoubleBlock& b =
          streaming ? streamed : _double[(std::size_t)L];
      // MSA: img (mod set 0) + txt.  mod layout [shift,scale,gate]*2 (each H).
      op.ln(img, 0, nrm, 0, IS, H);
      op.adaln(nrm, 0, mimg, H, 0, nrm, 0, H, IS * H);
      op.tap("dbl_norm1_img", L, nrm, 0, IS, H);
      op.gemm(nrm, b.q, jq, (std::size_t)TS * H, IS, H, H);
      op.gemm(nrm, b.k, jk, (std::size_t)TS * H, IS, H, H);
      op.gemm(nrm, b.v, jv, (std::size_t)TS * H, IS, H, H);
      op.ln(txt, 0, nrm, 0, TS, H);
      op.adaln(nrm, 0, mtxt, H, 0, nrm, 0, H, TS * H);
      op.tap("dbl_norm1_txt", L, nrm, 0, TS, H);
      op.gemm(nrm, b.aq, jq, 0, TS, H, H);
      op.gemm(nrm, b.ak, jk, 0, TS, H, H);
      op.gemm(nrm, b.av, jv, 0, TS, H, H);
      op.rms(jq, 0, b.aqn, jq, 0, TS * HED, HD);
      op.rms(jk, 0, b.akn, jk, 0, TS * HED, HD);
      const std::size_t io = (std::size_t)TS * H;   // image region offset
      op.rms(jq, io, b.qn, jq, io, IS * HED, HD);
      op.rms(jk, io, b.kn, jk, io, IS * HED, HD);
      attention(op, att, 0);                               // -> att (contiguous)
      op.tap("dbl_attn_txt", L, att, 0, TS, H);            // to_add_out input
      op.gemm(att, b.ao, ob, 0, TS, H, H);                 // text att[0:TS]
      op.gated(txt, 0, mtxt, 2 * H, ob, 0, H, TS * H);
      // img att is att[TS:seq] -> read at input offset TS*H (xe).
      op.tap("dbl_attn_img", L, att, (std::size_t)TS * H, IS, H);
      op.gemm(att, b.o, ob, 0, IS, H, H, (std::size_t)TS * H);
      op.gated(img, 0, mimg, 2 * H, ob, 0, H, IS * H);
      // FF (mod set 1: shift_mlp=3H, scale_mlp=4H, gate_mlp=5H). Flux2FeedForward
      // is SwiGLU: linear_in -> [gate|up] (2*INNER) -> silu(gate)*up -> linear_out.
      op.ln(img, 0, nrm, 0, IS, H);
      op.adaln(nrm, 0, mimg, 4 * H, 3 * H, nrm, 0, H, IS * H);
      op.tap("dbl_norm2_img", L, nrm, 0, IS, H);
      if (_fuse_ff) {
        op.swiglu_ff(nrm, b.ff_in, smlp, IS, H, DFF);    // silu(gate)*up [IS,INNER]
      } else {
        op.gemm(nrm, b.ff_in, ff1, 0, IS, DFF, H);       // [IS, 2*INNER]
        op.slice(ff1, sg, IS, DFF, INNER, 0);            // gate = first half
        op.slice(ff1, su, IS, DFF, INNER, INNER);        // up = second half
        op.elt(_fn_swiglu, sg, 0, su, 0, smlp, 0, IS * INNER);
      }
      op.tap("dbl_ffact_img", L, smlp, 0, IS, INNER);
      op.gemm(smlp, b.ff_out, ob, 0, IS, H, INNER);
      op.gated(img, 0, mimg, 5 * H, ob, 0, H, IS * H);
      op.ln(txt, 0, nrm, 0, TS, H);
      op.adaln(nrm, 0, mtxt, 4 * H, 3 * H, nrm, 0, H, TS * H);
      op.tap("dbl_norm2_txt", L, nrm, 0, TS, H);
      if (_fuse_ff) {
        op.swiglu_ff(nrm, b.cff_in, smlp, TS, H, DFF);
      } else {
        op.gemm(nrm, b.cff_in, ff1, 0, TS, DFF, H);
        op.slice(ff1, sg, TS, DFF, INNER, 0);
        op.slice(ff1, su, TS, DFF, INNER, INNER);
        op.elt(_fn_swiglu, sg, 0, su, 0, smlp, 0, TS * INNER);
      }
      op.tap("dbl_ffact_txt", L, smlp, 0, TS, INNER);
      op.gemm(smlp, b.cff_out, ob, 0, TS, H, INNER);
      op.gated(txt, 0, mtxt, 5 * H, ob, 0, H, TS * H);
      if (streaming) { flush(); }   // commit block L before it frees
    }
    enc.end();
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt("MetalFlux2Transformer::forward_dit: {}",
                                 gpu_err.empty() ? "GPU failed" : gpu_err));
      }
      return {};
    }
  }
  if (prof) { t_dbl += ms_since(mk); }
  // Join streams: joint = [text; image].
  std::memcpy(joint.contents(), txt.contents(), (std::size_t)TS * H * 2);
  std::memcpy(static_cast<_Float16*>(joint.contents()) + (std::size_t)TS * H,
              img.contents(), (std::size_t)IS * H * 2);

  // ===== single-stream blocks. Each is a pure feed-forward graph, so it runs
  // as ONE command stream: proj + mlp -> attention -> GPU concat[att|mlp] ->
  // to_out -> gated. The [att|mlp] assembly is a GPU kernel (concat_cols), so
  // -- unlike the old host memcpy concat -- nothing forces a mid-block
  // commit().wait(); the whole block commits once. When ff_direct (dense fused
  // + strided-transpose kernel available), the two producers write straight
  // into scat (att -> [:, :H], mlp -> [:, H:]) so there is no concat at all;
  // the quant/unfused paths still concat (their swiglu is a shared kernel with
  // no strided output). =====
  // Needs the transpose_rs kernel (att -> scat[:, :H]) + a strided mlp kernel
  // for the active path: dense uses the flux2-only dense swiglu (out_stride
  // built in); quant + unfused use the new _rs twins.
  const bool have_mlp_rs =
      _fuse_ff ? (_quant_bits == 0
                      ? true
                      : (_fn_qmm_swiglu4_bm64_rs.valid()
                         && _fn_qmm_swiglu8_bm64_rs.valid()))
               : _fn_swiglu_rs.valid();
  const bool ff_direct = _fn_transpose_rs.valid() && have_mlp_rs;
  for (int L = 0; L < c.n_single; ++L) {
    if (_stream_stop && _stream_stop()) { return {}; }   // pipeline stop, any mode
    // Pinned prefix (L < _pinned_s) is resident in _single; the tail streams.
    const bool streaming = _stream_blocks && L >= _pinned_s;
    SingleBlock streamed;
    if (streaming) {
      if (!load_single_(*_stream_wts,
                        "single_transformer_blocks." + std::to_string(L) + ".",
                        streamed)) {
        return {};
      }
    }
    const SingleBlock& b = streaming ? streamed : _single[(std::size_t)L];
    if (prof) { mk = tnow(); }
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    auto op = make_ops(enc);
    op.ln(joint, 0, nrm, 0, seq, H);
    op.adaln(nrm, 0, msin, H, 0, nrm, 0, H, seq * H);      // (1+scale)+shift
    op.tap("sgl_norm", L, nrm, 0, seq, H);
    if (_fuse_ff) {
      // qkv-only proj [seq, 3H] + a fused-SwiGLU mlp GEMM writing smlp directly
      // (no [seq, 2*SMLP] gate|up intermediate + slice + swiglu).
      op.gemm(nrm, b.qkv, sproj, 0, seq, 3 * H, H);
      op.slice(sproj, jq, seq, 3 * H, H, 0);               // q
      op.slice(sproj, jk, seq, 3 * H, H, H);               // k
      op.slice(sproj, jv, seq, 3 * H, H, 2 * H);           // v
      if (ff_direct) {   // mlp -> scat[:, H:] (att will land in scat[:, :H])
        op.swiglu_ff(nrm, b.mlp_gu, scat, seq, H, 2 * SMLP, H + SMLP, H);
      } else {
        op.swiglu_ff(nrm, b.mlp_gu, smlp, seq, H, 2 * SMLP);
      }
    } else {
      op.gemm(nrm, b.qkv_mlp, sproj, 0, seq, PW, H);
      op.slice(sproj, jq, seq, PW, H, 0);                  // q
      op.slice(sproj, jk, seq, PW, H, H);                  // k
      op.slice(sproj, jv, seq, PW, H, 2 * H);              // v
      op.slice(sproj, sg, seq, PW, SMLP, 3 * H);           // mlp gate
      op.slice(sproj, su, seq, PW, SMLP, 3 * H + SMLP);    // mlp up
      if (ff_direct) {   // silu(gate)*up -> scat[:, H:] (no concat)
        op.swiglu_rs(sg, su, scat, seq, SMLP, H + SMLP, H);
      } else {
        op.elt(_fn_swiglu, sg, 0, su, 0, smlp, 0, seq * SMLP);   // -> smlp
      }
    }
    op.rms(jq, 0, b.qn, jq, 0, seq * HED, HD);
    op.rms(jk, 0, b.kn, jk, 0, seq * HED, HD);
    // Profiling barriers split the one stream into timed slices (prof only).
    if (prof) {
      enc.end(); stream.commit().wait(); t_sgl_gemm += ms_since(mk);
      stream = _mc->make_command_stream(); enc = stream.begin_compute();
      op = make_ops(enc); mk = tnow();
    }
    if (ff_direct) {
      attention(op, scat, H + SMLP);   // att -> scat[:, :H] (mlp already in [H:])
    } else {
      attention(op, att, 0);           // att -> att, concat below
    }
    if (prof) {
      enc.end(); stream.commit().wait(); t_sgl_attn += ms_since(mk);
      stream = _mc->make_command_stream(); enc = stream.begin_compute();
      op = make_ops(enc); mk = tnow();
    }
    // scat = [att | mlp]. Direct-write already placed both; else concat on GPU.
    if (!ff_direct) { op.concat_cols(scat, att, smlp, seq, H, SMLP); }
    op.tap("sgl_cat", L, scat, 0, seq, H + SMLP);
    op.gemm(scat, b.o, ob, 0, seq, H, H + SMLP);
    op.gated(joint, 0, msin, 2 * H, ob, 0, H, seq * H);    // += gate * to_out
    enc.end();
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt("MetalFlux2Transformer::forward_dit: {}",
                                 gpu_err.empty() ? "GPU failed" : gpu_err));
      }
      return {};
    }
    if (prof) { t_sgl_out += ms_since(mk); }
  }

  // ===== final: AdaLayerNormContinuous(image tail, temb) + proj_out =====
  // norm_out.linear(SiLU(temb)) -> [2H] = (shift, scale); x = (1+scale)*LN(x) +
  // shift; velocity = proj_out(x).
  if (prof) { mk = tnow(); }
  {
    CommandStream stream = _mc->make_command_stream();
    ComputeEncoder enc = stream.begin_compute();
    auto op = make_ops(enc);
    op.elt(_fn_mulsig, temb, 0, temb, 0, tsilu, 0, H);
    op.gemm(tsilu, _norm_out_lin, mimg, 0, 1, 2 * H, H);   // reuse mimg [2H]
    // Only the generated image tokens (joint[TS .. TS+IS_GEN], the head of the
    // image region) get proj_out; the trailing reference tokens are discarded.
    op.ln(joint, (std::size_t)TS * H, nrm, 0, IS_GEN, H);
    // AdaLayerNormContinuous: chunk(emb,2) -> (scale@0, shift@H).
    op.adaln(nrm, 0, mimg, 0, H, nrm, 0, H, IS_GEN * H);
    op.tap("emb_proj", 0, nrm, 0, IS_GEN, H);
    op.gemm(nrm, _proj_out, velocity, 0, IS_GEN, OC, H);
    enc.end();
    std::string gpu_err;
    if (!stream.commit().wait_ok(&gpu_err)) {
      if (_mc->session() != nullptr) {
        _mc->session()->warn(fmt("MetalFlux2Transformer::forward_dit: {}",
                                 gpu_err.empty() ? "GPU failed" : gpu_err));
      }
      return {};
    }
  }
  if (prof) {
    t_final += ms_since(mk);
    const double tot = t_dbl + t_sgl_gemm + t_sgl_attn + t_sgl_cat +
                       t_sgl_out + t_final;
    if (_mc->session() != nullptr) {
      _mc->session()->log_normal(fmt(
          "FLUX.2 DiT profile (seq={} img={} txt={}, {}+{} blocks): total "
          "{} ms | double(embed+Nblk) {} | single-gemm(qkv_mlp+norm+swiglu) {} "
          "| single-attn {} | single-concat(host) {} | single-out(to_out) {} | "
          "final {}",
          seq, IS, TS, (int)_double.size(), (int)_single.size(),
          (long)tot, (long)t_dbl, (long)t_sgl_gemm, (long)t_sgl_attn,
          (long)t_sgl_cat, (long)t_sgl_out, (long)t_final));
    }
  }
  return velocity;
}

}  // namespace genai
}  // namespace vpipe
