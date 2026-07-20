#include "generative-models/qwen-image/metal-qwen-image-transformer.h"

#include "common/flex-data.h"
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
#include <vector>

namespace vpipe {
namespace genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

// C++ mirror of mlx::steel::AttnParams (steel/attn/params.h) -- the param block
// the vendored steel flash-attention kernel reads. Same layout as the Krea-2 /
// FLUX.2 / LM copies.
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
  std::uint32_t u; std::memcpy(&u, &f, 4);
  // round-to-nearest-even.
  const std::uint32_t r = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
  return (std::uint16_t)r;
}

inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

// Load a checkpoint tensor -> bf16 SharedBuffer (BF16 memcpy'd; F16/F32 -> bf16).
SharedBuffer
to_elt_(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm)
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

}  // namespace

MetalQwenImageTransformer::~MetalQwenImageTransformer() = default;

SharedBuffer
MetalQwenImageTransformer::to_elt_(const MetalLlamaWeights& wts,
                                   const std::string& name)
{
  return vpipe::genai::to_elt_(wts, _mc, name);
}

bool
MetalQwenImageTransformer::load_linear_(const MetalLlamaWeights& wts,
                                        const std::string& pre, SharedBuffer& w,
                                        SharedBuffer& b)
{
  w = to_elt_(wts, pre + ".weight");
  b = to_elt_(wts, pre + ".bias");
  return !w.empty() && !b.empty();
}

// Load `<name>` as a (possibly-quantized) linear weight. Quantized iff the
// model is quantized (_quant_bits>0) AND `<name>.scales` is present: then
// `<name>.weight` is U32 packed codes ([N, K*bits/32], loaded RAW) and
// `.scales`/`.biases` are F16 -> bf16. Bits are derived from the tensor shapes
// so a mixed-precision checkpoint (w4/w8 per layer) loads correctly. Otherwise
// `<name>.weight` is a dense bf16 matrix.
MetalQwenImageTransformer::QWeight
MetalQwenImageTransformer::load_qw_(const MetalLlamaWeights& wts,
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
    qw.bits   = (bits == 8) ? 8 : 4;
    qw.codes  = wts.load(name + ".weight", _mc);     // raw U32 codes
    qw.scales = to_elt_(wts, name + ".scales");      // F16 -> bf16
    qw.qbias  = to_elt_(wts, name + ".biases");
    if (!qw.codes.empty() && !qw.scales.empty() && !qw.qbias.empty()) {
      qw.quantized = true;
      return qw;
    }
    qw.codes = {}; qw.scales = {}; qw.qbias = {};
  }
  qw.w = to_elt_(wts, name + ".weight");             // dense bf16
  return qw;
}

bool
MetalQwenImageTransformer::load_linear_q_(const MetalLlamaWeights& wts,
                                          const std::string& pre, QWeight& qw,
                                          SharedBuffer& b)
{
  qw = load_qw_(wts, pre);
  b = to_elt_(wts, pre + ".bias");
  return !qw.empty() && !b.empty();
}

bool
MetalQwenImageTransformer::load_block_(const MetalLlamaWeights& wts, int L,
                                       Block& b)
{
  const std::string p = "transformer_blocks." + std::to_string(L) + ".";
  bool ok = true;
  // AdaLN modulation: dense bf16 by default, or affine-quantized when the
  // checkpoint was built with model-quantize quant_modulation (load_linear_q_
  // auto-detects via the presence of *_mod.1.scales).
  ok = ok && load_linear_q_(wts, p + "img_mod.1", b.img_mod_w, b.img_mod_b);
  ok = ok && load_linear_q_(wts, p + "txt_mod.1", b.txt_mod_w, b.txt_mod_b);
  ok = ok && load_linear_q_(wts, p + "attn.to_q", b.qw, b.qb);
  ok = ok && load_linear_q_(wts, p + "attn.to_k", b.kw, b.kb);
  ok = ok && load_linear_q_(wts, p + "attn.to_v", b.vw, b.vb);
  ok = ok && load_linear_q_(wts, p + "attn.to_out.0", b.ow, b.ob);
  ok = ok && load_linear_q_(wts, p + "attn.add_q_proj", b.aqw, b.aqb);
  ok = ok && load_linear_q_(wts, p + "attn.add_k_proj", b.akw, b.akb);
  ok = ok && load_linear_q_(wts, p + "attn.add_v_proj", b.avw, b.avb);
  ok = ok && load_linear_q_(wts, p + "attn.to_add_out", b.aow, b.aob);
  b.nq  = to_elt_(wts, p + "attn.norm_q.weight");
  b.nk  = to_elt_(wts, p + "attn.norm_k.weight");
  b.naq = to_elt_(wts, p + "attn.norm_added_q.weight");
  b.nak = to_elt_(wts, p + "attn.norm_added_k.weight");
  ok = ok && !b.nq.empty() && !b.nk.empty() && !b.naq.empty() && !b.nak.empty();
  ok = ok && load_linear_q_(wts, p + "img_mlp.net.0.proj", b.img_fc1_w,
                            b.img_fc1_b);
  ok = ok && load_linear_q_(wts, p + "img_mlp.net.2", b.img_fc2_w, b.img_fc2_b);
  ok = ok && load_linear_q_(wts, p + "txt_mlp.net.0.proj", b.txt_fc1_w,
                            b.txt_fc1_b);
  ok = ok && load_linear_q_(wts, p + "txt_mlp.net.2", b.txt_fc2_w, b.txt_fc2_b);
  return ok;
}

std::unique_ptr<MetalQwenImageTransformer>
MetalQwenImageTransformer::load(const std::string& model_dir, MetalCompute* mc,
                                const Config& cfg, bool stream_blocks,
                                double pin_frac)
{
  if (mc == nullptr) { return nullptr; }
  auto wtsopt = MetalLlamaWeights::open_model(model_dir);
  if (!wtsopt.has_value()) { return nullptr; }
  const MetalLlamaWeights& wts = *wtsopt;

  auto m = std::unique_ptr<MetalQwenImageTransformer>(
      new MetalQwenImageTransformer());
  m->_mc = mc;
  m->_cfg = cfg;
  m->_stream_blocks = stream_blocks;

  // Affine group-quant detection: config.json `quantization {bits, group_size}`
  // (written by model-quantize target=dit). Absent => dense bf16.
  {
    namespace fs = std::filesystem;
    std::ifstream in(fs::path(model_dir) / "config.json");
    if (in) {
      FlexData fd = FlexData::from_json(in);
      if (fd.is_object()) {
        auto o = fd.as_object();
        if (o.contains("quantization")) {
          FlexData q = o.at("quantization");
          if (q.is_object()) {
            auto qo = q.as_object();
            const int b = qo.contains("bits") ? (int)qo.at("bits").as_int(0) : 0;
            const int g = qo.contains("group_size")
                ? (int)qo.at("group_size").as_int(64) : 64;
            if (b == 4 || b == 8) {
              m->_quant_bits  = b;
              m->_quant_group = (g == 32 || g == 64) ? g : 64;
            }
          }
        }
      }
    }
  }

  // bf16 metallib variants (the residual stream exceeds f16 range).
  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_lib_sdpa = mc->load_library("sdpa_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_fn_gemm      = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_gemm_bias = m->_lib_gemm.function("dense_gemm_bias_f16");
  // Fast-path twins (best-effort): BM=64 tile for tall-M block GEMMs + a GEMV
  // for the M=1 conditioning/modulation rows (bias-less -> bias_add_rows_f16).
  m->_fn_gemm_bm64 = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_gemv      = m->_lib_gemm.function("dense_gemv_t_f16");
  m->_fn_bias_add  = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_rms       = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_layernorm = m->_lib_elt.function("layer_norm_plain_f16");
  m->_fn_silu      = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_gelu      = m->_lib_elt.function("gelu_tanh_ff_f16");
  m->_fn_residual  = m->_lib_elt.function("residual_add_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_sdpa      = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_rope_table = m->_lib_rope.function("rope_pair_table_ftab_f16");
  if (std::getenv("VPIPE_QIE_NO_FUSE_ROPE") == nullptr) {
    m->_fn_transpose_rope =
        m->_lib_rope.function("transpose_rope_pair_ftab_f16");
  }
  m->_fn_adaln     = m->_lib_elt.function("adaln_modulate_f16");
  m->_fn_gated     = m->_lib_elt.function("gated_residual_f16");
  m->_fn_colabsmax = m->_lib_elt.function("col_absmax_f16");
  if (!m->_fn_gemm.valid() || !m->_fn_gemm_bias.valid() || !m->_fn_rms.valid() ||
      !m->_fn_layernorm.valid() || !m->_fn_silu.valid() || !m->_fn_gelu.valid() ||
      !m->_fn_residual.valid() || !m->_fn_transpose.valid() ||
      !m->_fn_sdpa.valid() || !m->_fn_rope_table.valid() ||
      !m->_fn_adaln.valid() || !m->_fn_gated.valid()) {
    return nullptr;
  }

  // Steel register-resident flash attention (bf16, head_dim 128) for the joint
  // attention. Best-effort: if the library / entry point is missing, forward()
  // keeps the scalar sdpa fallback. VPIPE_QIE_NO_STEEL_ATTN forces scalar.
  m->_lib_attn = mc->load_library("attn_steel");
  m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));
  m->_steel_attn_ok = m->_lib_attn.valid() && !m->_attn_params.empty() &&
                      cfg.head_dim == 128 &&
                      std::getenv("VPIPE_QIE_NO_STEEL_ATTN") == nullptr;

  // Affine qmm kernels (bf16 variant) -- only when the checkpoint is quantized.
  if (m->_quant_bits > 0) {
    m->_lib_qmm = mc->load_library("affine_qmm_steel_bf16");
    const std::string g = "g" + std::to_string(m->_quant_group);
    m->_fn_qmm4     = m->_lib_qmm.function("affine_qmm_steel_w4" + g);
    m->_fn_qmm8     = m->_lib_qmm.function("affine_qmm_steel_w8" + g);
    m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    const bool need8 = (m->_quant_bits == 8);
    if (!m->_fn_bias_add.valid() || !m->_fn_qmm4.valid() ||
        (need8 && !m->_fn_qmm8.valid())) {
      return nullptr;
    }
    // Wider steel-qmm tiles for the tall-M DiT block GEMMs (M ~= seq): BM=64,
    // then BM=128 at high resolution. Same buffer layout + f32 accumulate as the
    // base tile, so token-exact -- only the output-row tiling differs. On M4 the
    // QIE w4 GEMMs are compute/dequant-bound (not weight-BW-bound like the
    // Krea-2 / FLUX.2 cases), so BM=64 is a small win but BM=128 tends to
    // over-tile (lower occupancy) at QIE's dims -- default BM=64. 0 disables
    // (base BM=32); g32 falls back to base. VPIPE_QIE_QMM_TILE overrides (0/1/2).
    m->_qmm_tile = 1;
    if (const char* t = std::getenv("VPIPE_QIE_QMM_TILE")) {
      m->_qmm_tile = std::atoi(t);
    }
    if (m->_qmm_tile >= 1) {
      m->_fn_qmm4_bm64 = m->_lib_qmm.function("affine_qmm_steel_w4" + g + "_bm64");
      m->_fn_qmm8_bm64 = m->_lib_qmm.function("affine_qmm_steel_w8" + g + "_bm64");
      if (!m->_fn_qmm4_bm64.valid() || (need8 && !m->_fn_qmm8_bm64.valid())) {
        m->_qmm_tile = 0;
      }
    }
    if (m->_qmm_tile == 2) {
      m->_fn_qmm4_bm128 =
          m->_lib_qmm.function("affine_qmm_steel_w4" + g + "_bm128");
      m->_fn_qmm8_bm128 =
          m->_lib_qmm.function("affine_qmm_steel_w8" + g + "_bm128");
      if (!m->_fn_qmm4_bm128.valid() || (need8 && !m->_fn_qmm8_bm128.valid())) {
        m->_qmm_tile = 1;   // BM128 unavailable (e.g. g32) -> stay on BM64
      }
    }
    // Peak M=1 quantized GEMV for the per-block modulation projection (used when
    // the mod weights are quantized -- e.g. w8 mod from model-quantize). Streams
    // the codes at ~peak bandwidth instead of the steel qmm tile's 31/32-wasted
    // M=1 rows. Best-effort: null -> gemm_bias_q fallback. VPIPE_QIE_NO_MOD_QMV
    // disables.
    if (std::getenv("VPIPE_QIE_NO_MOD_QMV") == nullptr) {
      m->_lib_qmv = mc->load_library("affine_qmv_bf16");
      m->_fn_qmv4 = m->_lib_qmv.function("affine_qmv_w4" + g);
      m->_fn_qmv8 = m->_lib_qmv.function("affine_qmv_w8" + g);
    }
  }

  // M5 matrix-core matmul2d for the block/projection GEMMs (the *_bf16 metallib
  // variants, since the DiT runs bf16). Dense weights feed dense_gemm_mma
  // directly; quantized weights dequant-expand into _w_deq (affine_dequant) then
  // run through the SAME dense matmul2d -- the dequant-once -> dense-GEMM shape
  // that matches affine_qmm_steel to f32 rounding. Gated on matrix cores
  // (M4/older keep steel); VPIPE_QIE_NO_MMA2 A/B off.
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_QIE_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma_bf16");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    // TN=2 tile (the K=6144..12288 band; QIE has no K there, but load for
    // parity). NO_TN2 forces the plain deep tile (A/B).
    if (std::getenv("VPIPE_QIE_NO_TN2") == nullptr) {
      m->_fn_dense_mma_tn2 =
          m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_tn2_f16");
    }
    m->_fn_dense_mma_splitk =
        m->_lib_dense_mma.function("dense_gemm_mma_splitk_n128x256_k8192_f16");
    // matmul2d has no bias slot -> a row-broadcast bias_add folds it. Load it
    // even for a dense checkpoint (the quant block above loads it only when
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
    if (const char* e = std::getenv("VPIPE_QIE_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    m->_use_splitk = m->_use_mma2 && m->_fn_dense_mma_splitk.valid()
                     && m->_fn_residual.valid()
                     && std::getenv("VPIPE_QIE_NO_SPLITK") == nullptr;
  }

  // Top-level weights.
  bool ok = m->load_linear_(wts, "img_in", m->_img_in_w, m->_img_in_b);
  m->_txt_norm_w = m->to_elt_(wts, "txt_norm.weight");
  ok = ok && !m->_txt_norm_w.empty();
  ok = ok && m->load_linear_(wts, "txt_in", m->_txt_in_w, m->_txt_in_b);
  ok = ok && m->load_linear_(
      wts, "time_text_embed.timestep_embedder.linear_1", m->_t1_w, m->_t1_b);
  ok = ok && m->load_linear_(
      wts, "time_text_embed.timestep_embedder.linear_2", m->_t2_w, m->_t2_b);
  ok = ok && m->load_linear_(wts, "norm_out.linear", m->_normout_w,
                             m->_normout_b);
  ok = ok && m->load_linear_(wts, "proj_out", m->_projout_w, m->_projout_b);
  if (!ok) { return nullptr; }

  // Blocks: preload all 60 (default), or -- in streaming mode -- skip the
  // preload and retain the source mmap so forward() loads/frees each block on
  // demand (memory-bounded, for 16GB boxes).
  if (!stream_blocks) {
    m->_blocks.resize((std::size_t)cfg.n_layers);
    for (int L = 0; L < cfg.n_layers; ++L) {
      if (!m->load_block_(wts, L, m->_blocks[(std::size_t)L])) {
        return nullptr;
      }
    }
  } else {
    m->_stream_wts =
        std::make_unique<MetalLlamaWeights>(std::move(*wtsopt));
    // Pinned-prefix: pin as many LEADING blocks as fit in pin_frac of RAM;
    // forward() reuses them and streams only the tail (blocks >= _pinned).
    if (pin_frac > 0.0) {
      std::vector<std::string> prefixes((std::size_t)cfg.n_layers);
      for (int L = 0; L < cfg.n_layers; ++L) {
        prefixes[(std::size_t)L] =
            "transformer_blocks." + std::to_string(L) + ".";
      }
      m->_pinned = stream_pin_count(*m->_stream_wts, prefixes, pin_frac);
      if (m->_pinned > cfg.n_layers) { m->_pinned = cfg.n_layers; }
      m->_blocks.resize((std::size_t)m->_pinned);
      for (int L = 0; L < m->_pinned; ++L) {
        if (!m->load_block_(*m->_stream_wts, L, m->_blocks[(std::size_t)L])) {
          return nullptr;
        }
      }
    }
  }
  return m;
}

// ---- AWQ calibration --------------------------------------------------------
void
MetalQwenImageTransformer::calib_begin()
{
  const int nL = _cfg.n_layers, H = _cfg.hidden, FF = _cfg.ffn;
  _calib_acc.clear();
  // dim = hidden for qkv/o/fc1 inputs, ffn for fc2 inputs (GELU output).
  const std::pair<const char*, int> groups[] = {
      {"img_qkv", H}, {"txt_qkv", H}, {"img_o", H}, {"txt_o", H},
      {"img_fc1", H}, {"txt_fc1", H}, {"img_fc2", FF}, {"txt_fc2", FF}};
  for (const auto& g : groups) {
    SharedBuffer b =
        _mc->make_shared_buffer((std::size_t)nL * g.second * 2);
    std::memset(b.contents(), 0, b.byte_size());   // zero before the first tap
    _calib_acc[g.first] = std::move(b);
  }
  _calib_on = true;
}

std::map<std::string, std::vector<float>>
MetalQwenImageTransformer::calib_stats() const
{
  const int nL = _cfg.n_layers, H = _cfg.hidden, FF = _cfg.ffn;
  std::map<std::string, std::vector<float>> out;
  for (const auto& kv : _calib_acc) {
    const int dim = (kv.first == "img_fc2" || kv.first == "txt_fc2") ? FF : H;
    std::vector<float> v((std::size_t)nL * dim);
    const auto* s = static_cast<const std::uint16_t*>(kv.second.contents());
    for (std::size_t i = 0; i < v.size(); ++i) { v[i] = bf16_to_f32_(s[i]); }
    out[kv.first] = std::move(v);
  }
  return out;
}

std::vector<float>
MetalQwenImageTransformer::time_proj_(float sigma) const
{
  // diffusers Timesteps(num_channels=256, flip_sin_to_cos=True,
  // downscale_freq_shift=0, scale=1000): arg = sigma*1000 * exp(-ln(1e4)*i/128),
  // emb = [cos(arg), sin(arg)] (flip_sin_to_cos).
  const int C = _cfg.time_proj, half = C / 2;
  std::vector<float> out((std::size_t)C);
  const float sc = 1000.0f;
  for (int i = 0; i < half; ++i) {
    const float freq = std::exp(-std::log(10000.0f) * (float)i / (float)half);
    const float arg = sigma * sc * freq;
    out[(std::size_t)i] = std::cos(arg);
    out[(std::size_t)(half + i)] = std::sin(arg);
  }
  return out;
}

void
MetalQwenImageTransformer::build_rope_(int txt_seq,
                                       const std::vector<ImgSeg>& segs,
                                       SharedBuffer& cos_out,
                                       SharedBuffer& sin_out)
{
  const int D = _cfg.head_dim;          // 128
  const int P = D / 2;                  // 64 pairs
  const int a0 = _cfg.axes[0], a1 = _cfg.axes[1], a2 = _cfg.axes[2];  // 16,56,56
  const int p0 = a0 / 2, p1 = a1 / 2;   // pairs per axis: 8, 28, (28)
  int total_img = 0, max_vid = 0;
  for (const ImgSeg& s : segs) {
    total_img += s.seq;
    max_vid = std::max(max_vid, std::max(s.grid_h / 2, s.grid_w / 2));
  }
  const int T = txt_seq + total_img;
  cos_out = _mc->make_shared_buffer((std::size_t)T * D * 4);   // f32 tables
  sin_out = _mc->make_shared_buffer((std::size_t)T * D * 4);
  auto* cb = static_cast<float*>(cos_out.contents());
  auto* sb = static_cast<float*>(sin_out.contents());

  // invfreq for pair j (per axis): theta^(-2*local/axis_dim).
  auto invfreq = [&](int j) -> double {
    int axis_dim, local;
    if (j < p0) { axis_dim = a0; local = j; }
    else if (j < p0 + p1) { axis_dim = a1; local = j - p0; }
    else { axis_dim = a2; local = j - p0 - p1; }
    return std::pow((double)_cfg.rope_theta,
                    -2.0 * (double)local / (double)axis_dim);
  };
  // Fill row `t` with the 3-axis angles from (pf, ph, pw) positions.
  auto fill = [&](int t, double pf, double ph, double pw) {
    for (int j = 0; j < P; ++j) {
      double pos = (j < p0) ? pf : (j < p0 + p1 ? ph : pw);
      const double ang = pos * invfreq(j);
      const float c = (float)std::cos(ang), s = (float)std::sin(ang);
      const std::size_t o = (std::size_t)t * D + 2 * j;
      cb[o] = c; cb[o + 1] = c; sb[o] = s; sb[o + 1] = s;
    }
  };
  // Text rows first: position = max_vid + tt on all axes.
  for (int tt = 0; tt < txt_seq; ++tt) {
    const double p = (double)(max_vid + tt);
    fill(tt, p, p, p);
  }
  // Then each image segment: frame band + centered height/width.
  int row = txt_seq;
  for (const ImgSeg& s : segs) {
    const int gh = s.grid_h, gw = s.grid_w;
    const int hoff = gh - gh / 2, woff = gw - gw / 2;   // centering offsets
    for (int r = 0; r < gh; ++r) {
      for (int c = 0; c < gw; ++c) {
        fill(row++, (double)s.frame, (double)(r - hoff), (double)(c - woff));
      }
    }
  }
}

bool
MetalQwenImageTransformer::gemm_mma_(ComputeEncoder& enc,
                                     const SharedBuffer& xin, std::size_t xe,
                                     const QWeight& w, const SharedBuffer& y,
                                     std::size_t ye, int M, int N, int K)
{
  // Matrix-core matmul2d only when present, M amortizes the 128-row tile, and N
  // is non-degenerate. Otherwise the caller keeps its steel path.
  if (!_use_mma2 || M < _mma_min_m || N < 16) { return false; }
  const SharedBuffer* wdense;
  if (w.quantized) {
    const metal_compute::ComputeFunction& dq =
        (w.bits == 8) ? _fn_dequant8 : _fn_dequant4;
    if (!dq.valid()) { return false; }
    const std::size_t need = (std::size_t)N * K * 2;   // bf16
    if (_w_deq.empty() || _w_deq.byte_size() < need) {
      _w_deq = _mc->make_shared_buffer(need);
      if (_w_deq.empty()) { return false; }
    }
    // codes/scales/qbias -> _w_deq[N,K] (one thread per packed u32 word: w4 has
    // 8 nibbles/word so K/8 words, w8 has 4 bytes/word so K/4). Serial ordering
    // + Metal's WAR hazard tracking make the shared _w_deq safe to reuse across
    // a block's GEMMs (each dequant->matmul pair runs before the next writes).
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
  // Split-K deep-reduction path (K a multiple of kSplitKC, >= 2 chunks): each
  // split gets its own threadgroup plane (grid.z); a residual_add fold sums the
  // planes into y. QIE's dims never trip this (kept for parity); falls through
  // to the single-op path when the scratch alloc fails or the shape doesn't
  // split.
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
      enc.set_buffer(0, xin, xe * 2); enc.set_buffer(1, *wdense);
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
  // Tile-adaptive dense matmul2d (no bias); matmul2d tensor extents clamp the
  // M/N tails. 128x128 for K < 6144 (QIE qkv/out/fc1, K=3072/3584); the plain
  // 128x256 deep tile for K >= 12288 (ff-down); the TN=2 tile fills the 6144..
  // 12288 band (unused by QIE's dims).
  int RN = 256;   // effective N-region per tg (TN*BN); grid divides N by it
  const metal_compute::ComputeFunction* fn = &_fn_dense_mma_deep;
  if (K < 6144) {
    fn = &_fn_dense_mma; RN = 128;
  } else if (K < 12288 && _fn_dense_mma_tn2.valid()) {
    fn = &_fn_dense_mma_tn2; RN = 512;
  }
  enc.set_function(*fn);
  enc.set_buffer(0, xin, xe * 2);
  enc.set_buffer(1, *wdense); enc.set_buffer(2, *wdense);
  enc.set_buffer(3, y, ye * 2);
  enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
  enc.set_constant(7, 0);
  enc.dispatch({(unsigned)(((N + RN - 1) / RN) * 256),
                (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
  return true;
}

SharedBuffer
MetalQwenImageTransformer::forward(const SharedBuffer& hidden, int gen_seq,
                                   const SharedBuffer& txt, int txt_seq,
                                   int grid_h, int grid_w, float sigma,
                                   const std::vector<RefImage>& refs,
                                   int stop_after_block)
{
  const int H = _cfg.hidden, Hd = _cfg.head_dim, NH = _cfg.n_heads;
  const int IC = _cfg.in_channels, FF = _cfg.ffn;
  const float eps = _cfg.norm_eps;

  // Image segments: generated grid (frame 0) + each reference (frame idx).
  std::vector<ImgSeg> segs;
  segs.push_back({0, grid_h, grid_w, gen_seq});
  int total_img = gen_seq, frame = 1;
  for (const RefImage& r : refs) {
    segs.push_back({frame++, r.grid_h, r.grid_w, r.seq});
    total_img += r.seq;
  }
  const int JT = txt_seq + total_img;   // joint sequence length

  auto buf = [&](std::size_t n) { return _mc->make_shared_buffer(n * 2); };

  // Persistent host-built inputs.
  const std::vector<float> tproj = time_proj_(sigma);
  SharedBuffer tproj_b = buf((std::size_t)_cfg.time_proj);
  { auto* d = static_cast<std::uint16_t*>(tproj_b.contents());
    for (int i = 0; i < _cfg.time_proj; ++i) { d[i] = f32_to_bf16_(tproj[i]); } }
  SharedBuffer rcos, rsin;
  build_rope_(txt_seq, segs, rcos, rsin);

  // Stream-persistent scratch.
  SharedBuffer x_img = buf((std::size_t)total_img * H);   // image stream
  SharedBuffer x_txt = buf((std::size_t)txt_seq * H);     // text stream
  SharedBuffer temb = buf((std::size_t)H), tsilu = buf((std::size_t)H);
  SharedBuffer th1 = buf((std::size_t)H);
  // Modulation scratch [6H] (per-block when not precomputing; also the final
  // norm_out's [2H] scratch). The block loop shadows these with per-block views
  // into imod_all/tmod_all when modulation is precomputed (preloaded path).
  SharedBuffer imod_s = buf((std::size_t)6 * H), tmod_s = buf((std::size_t)6 * H);
  SharedBuffer msilu = buf((std::size_t)H);
  SharedBuffer nrm = buf((std::size_t)JT * H), mdl = buf((std::size_t)JT * H);
  SharedBuffer jq = buf((std::size_t)JT * H), jk = buf((std::size_t)JT * H),
               jv = buf((std::size_t)JT * H);
  SharedBuffer qt = buf((std::size_t)JT * H), kt = buf((std::size_t)JT * H),
               vt = buf((std::size_t)JT * H), at = buf((std::size_t)JT * H);
  SharedBuffer att = buf((std::size_t)JT * H);
  SharedBuffer g1 = buf((std::size_t)total_img * FF),
               g2 = buf((std::size_t)txt_seq * FF);
  SharedBuffer velo = buf((std::size_t)gen_seq * IC);

  // Env-gated per-section GPU timing (VPIPE_QIE_DIT_PROFILE). Each section
  // boundary inserts a commit+wait barrier and accumulates wall time, splitting
  // the deferred stream into timed slices. Preloaded path only (the streaming /
  // pinned path already flushes per block); the barriers serialize the GPU so
  // absolute step time inflates a little, but the RELATIVE breakdown is
  // faithful. Mirrors the Krea-2 / FLUX.2 DiT profilers.
  const bool prof = !_stream_blocks &&
                    std::getenv("VPIPE_QIE_DIT_PROFILE") != nullptr;
  // Measurement-only: skip the modulation GEMVs to isolate their bandwidth cost
  // (imod/tmod left stale -> output is garbage; timing only). VPIPE_QIE_SKIP_MOD.
  const bool skip_mod = std::getenv("VPIPE_QIE_SKIP_MOD") != nullptr;
  double t_embed = 0, t_mod = 0, t_norm = 0, t_qkv = 0, t_attn = 0,
         t_oproj = 0, t_ff = 0, t_final = 0;
  std::chrono::steady_clock::time_point mark;

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // Commit the current encoder + start a fresh one (streaming mode): forces
    // the block's GPU work to complete before its weights are freed, so only
    // ~one block is ever resident. `enc`/`stream` are captured by reference by
    // every lambda below, so reassigning them here is transparent.
    auto flush = [&]() {
      enc.end();
      stream.commit().wait();
      stream = _mc->make_command_stream();
      enc = stream.begin_compute();
    };
    // Profiling barrier: commit+wait the accumulated ops, add the elapsed slice
    // to `acc`, reopen the stream, reset the mark. No-op unless profiling.
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
    // ---- kernel helper lambdas (mirror the Krea toolbox contracts) ----
    auto gemm_bias = [&](const SharedBuffer& x, std::size_t xe,
                         const SharedBuffer& w, const SharedBuffer& b,
                         const SharedBuffer& y, std::size_t ye, int M, int N,
                         int K) {
      // M == 1 (conditioning / modulation / head rows): a GEMV avoids the steel
      // tile's 31/32 wasted rows. dense_gemv_t is bias-less -> a row bias-add
      // follows (all QIE gemm_bias sites carry a bias).
      if (M == 1 && _fn_gemv.valid() && _fn_bias_add.valid()) {
        enc.set_function(_fn_gemv);
        enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w);
        enc.set_buffer(2, y, ye * 2);
        enc.set_constant(3, K); enc.set_constant(4, N);
        enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
        enc.set_function(_fn_bias_add);
        enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, b);
        enc.set_constant(2, N); enc.set_constant(3, (unsigned)N);
        enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
        return;
      }
      // M > 1: steel MMA dense GEMM (simdgroup-matrix, f32 accumulate) -- the
      // fast path, same machinery as affine_qmm_steel minus the dequant. Bias is
      // folded in-kernel (buffer 2 + has_bias). Prefer the BM=64 tile: it
      // computes 64 output rows per threadgroup with the same 128 threads, so
      // each [BN,BK] weight tile is re-read M/64 times instead of M/32 --
      // halving the f16 weight bandwidth for tall M (and never worse for small
      // M, where these large-N/K GEMMs are weight-bandwidth-bound anyway). NOTE
      // the constant order is 4:K 5:N 6:M (the scalar dense_gemm_bias_f16
      // fallback uses 4:M 5:N 6:K). Every QIE linear has K % 32 == 0; M/N tails
      // are handled in-kernel. The scalar 16x16 kernel is a naive
      // one-thread-per-output tiled GEMM (kept only as a fallback).
      const bool bm64 = _fn_gemm_bm64.valid();
      if (bm64 || _fn_gemm.valid()) {
        const int BM = bm64 ? 64 : 32;
        enc.set_function(bm64 ? _fn_gemm_bm64 : _fn_gemm);
        enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w); enc.set_buffer(2, b);
        enc.set_buffer(3, y, ye * 2);
        enc.set_constant(4, K); enc.set_constant(5, N); enc.set_constant(6, M);
        enc.set_constant(7, 1);
        enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                      (unsigned)(((M + BM - 1) / BM) * 2), 2}, {32, 2, 2});
        return;
      }
      enc.set_function(_fn_gemm_bias);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w); enc.set_buffer(2, b);
      enc.set_buffer(3, y, ye * 2);
      enc.set_constant(4, M); enc.set_constant(5, N); enc.set_constant(6, K);
      enc.set_constant(7, 1);
      enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                    (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
    };
    // Quant-aware Linear+bias for the leaf set: dense -> gemm_bias; quantized ->
    // affine_qmm_steel (bias-less) then a row-broadcast bias add.
    auto gemm_bias_q = [&](const SharedBuffer& x, std::size_t xe,
                           const QWeight& w, const SharedBuffer& b,
                           const SharedBuffer& y, std::size_t ye, int M, int N,
                           int K) {
      // M5 matmul2d (NAX): a biasless matmul (dense direct, or dequant-once for
      // a quantized weight) + a row-broadcast bias add. Handles both dense and
      // quantized weights; returns false on M4/older or sub-threshold shapes,
      // falling through to the steel gemm_bias / affine_qmm paths below.
      if (gemm_mma_(enc, x, xe, w, y, ye, M, N, K)) {
        enc.set_function(_fn_bias_add);
        enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, b);
        enc.set_constant(2, N);
        enc.set_constant(3, (unsigned)((std::size_t)M * N));
        enc.dispatch({(unsigned)((std::size_t)M * N), 1, 1}, {256, 1, 1});
        return;
      }
      if (!w.quantized) {
        gemm_bias(x, xe, w.w, b, y, ye, M, N, K);
        return;
      }
      // Tall-M block GEMMs (M ~= seq) amortize a wider tile: BM=64 (>=128 rows),
      // BM=128 at high res (>=1024). Small-M (txt, M=1 head) keep the base tile.
      int bm = 32;
      const bool huge = _qmm_tile == 2 && M >= 1024;
      const bool big  = _qmm_tile >= 1 && M >= 128;
      if (huge)     { bm = 128; }
      else if (big) { bm = 64; }
      enc.set_function(
          w.bits == 8 ? (huge ? _fn_qmm8_bm128 : big ? _fn_qmm8_bm64 : _fn_qmm8)
                      : (huge ? _fn_qmm4_bm128 : big ? _fn_qmm4_bm64 : _fn_qmm4));
      enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
      enc.set_buffer(2, w.qbias); enc.set_buffer(3, x, xe * 2);
      enc.set_buffer(4, y, ye * 2);
      enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, M);
      // BM=128 uses WM=4 (256 threads): threadgroup {32,2,4}, grid z=4.
      const unsigned tgz = (bm == 128) ? 4u : 2u;
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + bm - 1) / bm) * 2), tgz}, {32, 2, tgz});
      // bias add: y[g] += b[g % N] over M*N rows.
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, b);
      enc.set_constant(2, N); enc.set_constant(3, (unsigned)((std::size_t)M * N));
      enc.dispatch({(unsigned)((std::size_t)M * N), 1, 1}, {256, 1, 1});
    };
    // Modulation projection y[ye..] = w @ x + bias (M=1, N=6H, K=H). When the mod
    // weights are quantized, dispatch the peak M=1 affine_qmv (streams the codes
    // at ~peak bandwidth) instead of the steel qmm tile (31/32 M=1 rows wasted);
    // dense (or missing qmv) falls back to gemm_bias_q's M=1 GEMV path.
    auto mod_gemv = [&](const SharedBuffer& x, const QWeight& w,
                        const SharedBuffer& bias, const SharedBuffer& y,
                        std::size_t ye) {
      const int N = 6 * H, K = H;
      const bool qmv = w.quantized &&
          ((w.bits == 8 && _fn_qmv8.valid()) ||
           (w.bits == 4 && _fn_qmv4.valid())) && (N % 4 == 0);
      if (!qmv) { gemm_bias_q(x, 0, w, bias, y, ye, 1, N, K); return; }
      enc.set_function(w.bits == 8 ? _fn_qmv8 : _fn_qmv4);
      enc.set_buffer(0, w.codes); enc.set_buffer(1, w.scales);
      enc.set_buffer(2, w.qbias); enc.set_buffer(3, x);
      enc.set_buffer(4, y, ye * 2);
      enc.set_constant(5, K); enc.set_constant(6, N);
      enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
      enc.set_function(_fn_bias_add);   // y[n] += bias[n]
      enc.set_buffer(0, y, ye * 2); enc.set_buffer(1, bias);
      enc.set_constant(2, N); enc.set_constant(3, (unsigned)N);
      enc.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
    };
    // AWQ tap: acc[group][L] = max(., col-absmax of in[xe.., M, dim]) -- the
    // per-input-channel activation magnitude at a quantized linear's input.
    auto tap = [&](const char* group, int L, const SharedBuffer& in,
                   std::size_t xe, int M, int dim) {
      if (!_calib_on) { return; }
      auto it = _calib_acc.find(group);
      if (it == _calib_acc.end()) { return; }
      enc.set_function(_fn_colabsmax);
      enc.set_buffer(0, in, xe * 2);
      enc.set_buffer(1, it->second, (std::size_t)L * dim * 2);
      enc.set_constant(2, M); enc.set_constant(3, dim);
      enc.dispatch({(unsigned)dim, 1, 1}, {256, 1, 1});
    };
    auto layernorm = [&](const SharedBuffer& x, std::size_t xe,
                         const SharedBuffer& y, std::size_t ye, int R) {
      enc.set_function(_fn_layernorm);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, y, ye * 2);
      enc.set_constant(2, H); enc.set_constant(3, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto rms = [&](const SharedBuffer& x, std::size_t xe, const SharedBuffer& w,
                   const SharedBuffer& y, std::size_t ye, int R, int Dd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, w);
      enc.set_buffer(2, y, ye * 2);
      enc.set_constant(3, Dd); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto silu = [&](const SharedBuffer& x, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_silu);
      enc.set_buffer(0, x); enc.set_buffer(1, x); enc.set_buffer(2, y);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto gelu = [&](const SharedBuffer& x, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_gelu);
      enc.set_buffer(0, x); enc.set_buffer(1, y);
      enc.set_constant(2, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out, int A,
                         int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, A); enc.set_constant(3, Bd); enc.set_constant(4, Hd);
      enc.dispatch({(unsigned)Hd, (unsigned)Bd, (unsigned)A}, {(unsigned)Hd, 1, 1});
    };
    auto rope = [&](const SharedBuffer& x) {
      enc.set_function(_fn_rope_table);
      enc.set_buffer(0, x); enc.set_buffer(1, rcos); enc.set_buffer(2, rsin);
      enc.set_constant(3, NH); enc.set_constant(4, JT); enc.set_constant(5, Hd);
      enc.dispatch({(unsigned)(Hd / 2), (unsigned)JT, (unsigned)NH},
                   {(unsigned)(Hd / 2), 1, 1});
    };
    // Fused transpose [JT,NH,Hd] -> [NH,JT,Hd] + rope in one pass (q/k path).
    auto transpose_rope = [&](const SharedBuffer& in, const SharedBuffer& out) {
      enc.set_function(_fn_transpose_rope);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_buffer(2, rcos); enc.set_buffer(3, rsin);
      enc.set_constant(4, NH); enc.set_constant(5, JT); enc.set_constant(6, Hd);
      enc.dispatch({(unsigned)(Hd / 2), (unsigned)JT, (unsigned)NH},
                   {(unsigned)(Hd / 2), 1, 1});
    };
    auto sdpa = [&](const SharedBuffer& qb, const SharedBuffer& kb,
                    const SharedBuffer& vb, const SharedBuffer& out) {
      const float scale = 1.0f / std::sqrt((float)Hd);
      enc.set_function(_fn_sdpa);
      enc.set_buffer(0, qb); enc.set_buffer(1, kb); enc.set_buffer(2, vb);
      enc.set_buffer(3, out);
      enc.set_constant(4, scale); enc.set_constant(5, JT); enc.set_constant(6, Hd);
      enc.set_constant(7, NH); enc.set_constant(8, NH); enc.set_constant(9, JT);
      enc.set_constant(10, JT);
      enc.dispatch({32, (unsigned)NH, (unsigned)JT}, {32, 1, 1});
    };
    // out[ye..] = (1 + mod[scale_e..]) * x[xe..] + mod[shift_e..], broadcasting
    // the single-row mod over `total`/H token rows.
    auto adaln = [&](const SharedBuffer& x, std::size_t xe,
                     const SharedBuffer& mod, std::size_t scale_e,
                     std::size_t shift_e, const SharedBuffer& out,
                     std::size_t ye, int total) {
      enc.set_function(_fn_adaln);
      enc.set_buffer(0, x, xe * 2); enc.set_buffer(1, mod, scale_e * 2);
      enc.set_buffer(2, mod, shift_e * 2); enc.set_buffer(3, out, ye * 2);
      enc.set_constant(4, H); enc.set_constant(5, total);
      enc.dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
    };
    auto gated = [&](const SharedBuffer& h, const SharedBuffer& mod,
                     std::size_t gate_e, const SharedBuffer& sub, int total) {
      enc.set_function(_fn_gated);
      enc.set_buffer(0, h); enc.set_buffer(1, mod, gate_e * 2);
      enc.set_buffer(2, sub);
      enc.set_constant(3, H); enc.set_constant(4, total);
      enc.dispatch({(unsigned)total, 1, 1}, {256, 1, 1});
    };
    if (prof) { mark = std::chrono::steady_clock::now(); }
    // ---- embedders ----
    // img_in: hidden[total_img, IC] -> x_img[total_img, H]. Generated tokens
    // first; then each reference latent (same img_in). refs supply their own
    // packed latents.
    gemm_bias(hidden, 0, _img_in_w, _img_in_b, x_img, 0, gen_seq, H, IC);
    { int roff = gen_seq;
      for (const RefImage& r : refs) {
        gemm_bias(r.latents, 0, _img_in_w, _img_in_b, x_img,
                  (std::size_t)roff * H, r.seq, H, IC);
        roff += r.seq;
      }
    }
    // txt: RMSNorm(txt_dim) -> txt_in -> x_txt[txt_seq, H].
    { SharedBuffer tn = buf((std::size_t)txt_seq * _cfg.txt_dim);
      rms(txt, 0, _txt_norm_w, tn, 0, txt_seq, _cfg.txt_dim);
      gemm_bias(tn, 0, _txt_in_w, _txt_in_b, x_txt, 0, txt_seq, H, _cfg.txt_dim);
    }
    // time embedding: temb = t2(silu(t1(time_proj(sigma)))).
    gemm_bias(tproj_b, 0, _t1_w, _t1_b, th1, 0, 1, H, _cfg.time_proj);
    silu(th1, tsilu, H);
    gemm_bias(tsilu, 0, _t2_w, _t2_b, temb, 0, 1, H, H);
    psplit(t_embed);

    // Build the bf16 steel flash-attention function for this joint length + fill
    // its param block ONCE (same shape across all blocks) -- replaces the scalar
    // O(JT^2) sdpa at high resolution (mirrors Krea-2 / FLUX.2). Full MHA: NH
    // query heads = NH kv heads (gqa_factor 1); Q/K/V/O are [NH, JT, Hd], exactly
    // what the head-major transposes below produce. Falls back to scalar sdpa.
    const int A_BQ = 32, A_BK = 16;   // ALU steel bq/bk (bd128)
    metal_compute::ComputeFunction fn_attn;
    bool use_steel = _steel_attn_ok;
    if (use_steel) {
      auto* p = static_cast<SteelAttnParams*>(_attn_params.contents());
      p->B = 1; p->H = NH; p->D = Hd; p->qL = JT; p->kL = JT;
      p->gqa_factor = 1; p->scale = 1.0f / std::sqrt((float)Hd);
      p->NQ = (JT + A_BQ - 1) / A_BQ; p->NK = (JT + A_BK - 1) / A_BK;
      p->NQ_aligned = JT / A_BQ; p->NK_aligned = JT / A_BK;
      p->qL_rem = JT - p->NQ_aligned * A_BQ;
      p->kL_rem = JT - p->NK_aligned * A_BK;
      p->qL_off = 0;
      p->Q_strides[0] = (std::int64_t)NH * JT * Hd;
      p->Q_strides[1] = (std::int64_t)JT * Hd; p->Q_strides[2] = Hd;
      p->K_strides[0] = p->Q_strides[0];
      p->K_strides[1] = p->Q_strides[1]; p->K_strides[2] = Hd;
      p->V_strides[0] = p->K_strides[0];
      p->V_strides[1] = p->K_strides[1]; p->V_strides[2] = Hd;
      p->O_strides[0] = p->Q_strides[0];
      p->O_strides[1] = p->Q_strides[1]; p->O_strides[2] = Hd;
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (JT % A_BQ) == 0).set_bool(201, (JT % A_BK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      fn_attn = _lib_attn.function("attn_steel_h_bd128_bf16", fc);
      use_steel = fn_attn.valid();
    }
    const unsigned a_nqb = (unsigned)((JT + A_BQ - 1) / A_BQ);

    // ---- transformer blocks ----
    const int n_layers = (stop_after_block == -2) ? 0 : _cfg.n_layers;
    // Streaming mode: commit the embedders before the first streamed block so
    // the per-block command buffers stay bounded.
    if (_stream_blocks && n_layers > 0) { flush(); }
    // Precompute all blocks' modulation before the loop (preloaded path). silu
    // (temb) is a per-step constant, so the 2*n_layers M=1 mod GEMVs are
    // independent -- running them back-to-back lets the GPU pipeline them
    // (vs interleaving each with the attn/FF that depend on it), and with
    // quantized mod each uses the peak affine_qmv. Streaming keeps the per-block
    // path (blocks aren't all resident). VPIPE_QIE_NO_MOD_PRECOMP disables.
    const bool precomp_mod = !_stream_blocks && !skip_mod && n_layers > 0 &&
        std::getenv("VPIPE_QIE_NO_MOD_PRECOMP") == nullptr;
    SharedBuffer imod_all, tmod_all;
    if (precomp_mod) {
      imod_all = buf((std::size_t)n_layers * 6 * H);
      tmod_all = buf((std::size_t)n_layers * 6 * H);
      silu(temb, msilu, H);
      for (int L = 0; L < n_layers; ++L) {
        const Block& bb = _blocks[(std::size_t)L];
        const std::size_t off = (std::size_t)L * 6 * H;
        mod_gemv(msilu, bb.img_mod_w, bb.img_mod_b, imod_all, off);
        mod_gemv(msilu, bb.txt_mod_w, bb.txt_mod_b, tmod_all, off);
      }
      psplit(t_mod);
    }
    for (int L = 0; L < n_layers; ++L) {
      // Pipeline stop -> abandon the forward. Checked EVERY block (not just the
      // streamed tail) so a slow preloaded step at high resolution responds to a
      // stop request within one block (~ms) instead of running all 60.
      if (_stream_stop && _stream_stop()) { return {}; }
      // Pinned prefix (L < _pinned) is resident in _blocks; the tail streams
      // from the retained mmap into a loop-local Block, freed at the end of the
      // iteration (after the flush commits its work).
      const bool streaming = _stream_blocks && L >= _pinned;
      Block streamed;
      if (streaming) {
        if (!load_block_(*_stream_wts, L, streamed)) { return {}; }
      }
      const Block& b = streaming ? streamed : _blocks[(std::size_t)L];
      // Modulation params: mod = mod_linear(silu(temb)) [6H]. Precomputed slice
      // (preloaded) or computed per-block (streaming).
      const std::size_t moff = (std::size_t)L * 6 * H * 2;    // bytes
      const std::size_t msz = (std::size_t)6 * H * 2;
      SharedBuffer imod = precomp_mod ? imod_all.subview(moff, msz)
                                      : imod_s.subview(0, msz);
      SharedBuffer tmod = precomp_mod ? tmod_all.subview(moff, msz)
                                      : tmod_s.subview(0, msz);
      if (!precomp_mod) {
        silu(temb, msilu, H);
        if (!skip_mod) {
          mod_gemv(msilu, b.img_mod_w, b.img_mod_b, imod, 0);
          mod_gemv(msilu, b.txt_mod_w, b.txt_mod_b, tmod, 0);
        }
        psplit(t_mod);
      }

      const std::size_t ioff = (std::size_t)txt_seq * H;   // image row offset
      SharedBuffer io = buf((std::size_t)total_img * H);
      SharedBuffer to = buf((std::size_t)txt_seq * H);

      // --- attention ---
      // norm1 (LayerNorm, no affine) + adaLN modulate. Joint buffers `mdl` hold
      // txt-modulated rows [0:txt] then img-modulated rows [txt:].
      layernorm(x_txt, 0, nrm, 0, txt_seq);
      layernorm(x_img, 0, nrm, ioff, total_img);
      // mod chunk layout [shift1|scale1|gate1|shift2|scale2|gate2] (H each).
      adaln(nrm, 0, tmod, H, 0, mdl, 0, txt_seq * H);              // txt norm1
      adaln(nrm, ioff, imod, H, 0, mdl, ioff, total_img * H);       // img norm1
      psplit(t_norm);
      // q/k/v projections (txt: add_*; img: to_*) into the joint buffers.
      tap("txt_qkv", L, mdl, 0, txt_seq, H);
      tap("img_qkv", L, mdl, ioff, total_img, H);
      gemm_bias_q(mdl, 0, b.aqw, b.aqb, jq, 0, txt_seq, H, H);
      gemm_bias_q(mdl, 0, b.akw, b.akb, jk, 0, txt_seq, H, H);
      gemm_bias_q(mdl, 0, b.avw, b.avb, jv, 0, txt_seq, H, H);
      gemm_bias_q(mdl, ioff, b.qw, b.qb, jq, ioff, total_img, H, H);
      gemm_bias_q(mdl, ioff, b.kw, b.kb, jk, ioff, total_img, H, H);
      gemm_bias_q(mdl, ioff, b.vw, b.vb, jv, ioff, total_img, H, H);
      psplit(t_qkv);
      // q/k per-head RMSNorm (txt: norm_added_*, img: norm_*).
      rms(jq, 0, b.naq, jq, 0, txt_seq * NH, Hd);
      rms(jk, 0, b.nak, jk, 0, txt_seq * NH, Hd);
      rms(jq, ioff, b.nq, jq, ioff, total_img * NH, Hd);
      rms(jk, ioff, b.nk, jk, ioff, total_img * NH, Hd);
      // transpose [JT,NH,Hd] -> [NH,JT,Hd], apply RoPE, joint attention. q/k
      // fuse the transpose + rope into one pass; v is a plain transpose (no rope).
      if (_fn_transpose_rope.valid()) {
        transpose_rope(jq, qt);
        transpose_rope(jk, kt);
      } else {
        transpose(jq, qt, JT, NH);
        transpose(jk, kt, JT, NH);
        rope(qt); rope(kt);
      }
      transpose(jv, vt, JT, NH);
      if (use_steel) {
        // Register-resident flash attention over the joint sequence: Q/K/V/O
        // [NH, JT, Hd]. Grid (32*NQ, 4*NH, 1), tg (32, 4, 1) per MLX steel.
        enc.set_function(fn_attn);
        enc.set_buffer(0, qt); enc.set_buffer(1, kt); enc.set_buffer(2, vt);
        enc.set_buffer(3, at); enc.set_buffer(4, _attn_params);
        enc.dispatch({32 * a_nqb, 4 * (unsigned)NH, 1}, {32, 4, 1});
      } else {
        sdpa(qt, kt, vt, at);
      }
      transpose(at, att, NH, JT);   // [NH,JT,Hd] -> [JT,NH,Hd]
      psplit(t_attn);
      // output projections + gated residual (gate1 @ 2H).
      tap("img_o", L, att, ioff, total_img, H);
      tap("txt_o", L, att, 0, txt_seq, H);
      gemm_bias_q(att, ioff, b.ow, b.ob, io, 0, total_img, H, H);
      gemm_bias_q(att, 0, b.aow, b.aob, to, 0, txt_seq, H, H);
      gated(x_img, imod, 2 * H, io, total_img * H);
      gated(x_txt, tmod, 2 * H, to, txt_seq * H);
      psplit(t_oproj);
      if (stop_after_block == -3 && L == 0) { break; }   // post-attention debug

      // --- MLP (norm2 + adaLN + GELU FeedForward + gated residual, gate2) ---
      layernorm(x_img, 0, nrm, ioff, total_img);
      adaln(nrm, ioff, imod, 4 * H, 3 * H, mdl, ioff, total_img * H);
      tap("img_fc1", L, mdl, ioff, total_img, H);
      gemm_bias_q(mdl, ioff, b.img_fc1_w, b.img_fc1_b, g1, 0, total_img, FF, H);
      gelu(g1, g1, total_img * FF);
      tap("img_fc2", L, g1, 0, total_img, FF);
      gemm_bias_q(g1, 0, b.img_fc2_w, b.img_fc2_b, io, 0, total_img, H, FF);
      gated(x_img, imod, 5 * H, io, total_img * H);

      layernorm(x_txt, 0, nrm, 0, txt_seq);
      adaln(nrm, 0, tmod, 4 * H, 3 * H, mdl, 0, txt_seq * H);
      tap("txt_fc1", L, mdl, 0, txt_seq, H);
      gemm_bias_q(mdl, 0, b.txt_fc1_w, b.txt_fc1_b, g2, 0, txt_seq, FF, H);
      gelu(g2, g2, txt_seq * FF);
      tap("txt_fc2", L, g2, 0, txt_seq, FF);
      gemm_bias_q(g2, 0, b.txt_fc2_w, b.txt_fc2_b, to, 0, txt_seq, H, FF);
      gated(x_txt, tmod, 5 * H, to, txt_seq * H);
      psplit(t_ff);

      // Commit block L before `streamed` (its weights) frees at iteration end.
      // Pinned blocks stay resident, so no per-block flush is needed for them.
      if (streaming) { flush(); }
      if (stop_after_block == L) { break; }
    }

    if (stop_after_block < 0) {
      // norm_out (AdaLayerNormContinuous: scale@0, shift@H) on the generated
      // image tokens, then proj_out -> velocity.
      silu(temb, msilu, H);
      gemm_bias(msilu, 0, _normout_w, _normout_b, imod_s, 0, 1, 2 * H, H);
      layernorm(x_img, 0, nrm, 0, gen_seq);
      adaln(nrm, 0, imod_s, 0, H, mdl, 0, gen_seq * H);
      gemm_bias(mdl, 0, _projout_w, _projout_b, velo, 0, gen_seq, IC, H);
      psplit(t_final);
    }
  }
  stream.commit().wait();

  if (prof && stop_after_block < 0 && _mc->session() != nullptr) {
    const double tot = t_embed + t_mod + t_norm + t_qkv + t_attn + t_oproj +
                       t_ff + t_final;
    _mc->session()->log_normal(fmt(
        "QIE DiT profile (gen={} txt={} img={} blocks={}, 1 step): total {} ms "
        "| embed {} | mod {} | norm+adaln {} | qkv-gemm {} | attn {} | "
        "o-gemm {} | ff(gelu) {} | final {} (ms; barriers inflate absolute "
        "time)", gen_seq, txt_seq, total_img, _cfg.n_layers, (long)tot,
        (long)t_embed, (long)t_mod, (long)t_norm, (long)t_qkv, (long)t_attn,
        (long)t_oproj, (long)t_ff, (long)t_final));
  }

  if (stop_after_block == -2 || stop_after_block == -3) {
    SharedBuffer out = buf((std::size_t)gen_seq * H);
    std::memcpy(out.contents(), x_img.contents(), (std::size_t)gen_seq * H * 2);
    return out;
  }
  if (stop_after_block >= 0) {
    // Return the image-stream hidden [gen_seq, H] after that block.
    SharedBuffer out = buf((std::size_t)gen_seq * H);
    std::memcpy(out.contents(), x_img.contents(), (std::size_t)gen_seq * H * 2);
    return out;
  }
  return velo;
}

}  // namespace genai
}  // namespace vpipe
