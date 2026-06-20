#include "generative-models/qwen3/metal-qwen-model.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/event.h"
#include "apple-silicon/metal-compute/metal-compute.h"

#include <chrono>
#include <limits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {
std::size_t
numel_(const std::vector<std::int64_t>& s)
{
  std::size_t n = 1;
  for (auto d : s) { n *= (std::size_t)d; }
  return n;
}
// Length of the page-id prefix the N branches share (identical leading page
// ids -- they were all branched off one base, so the prefix pages are
// refcount-shared and byte-identical; the first private page differs per
// branch). 0 for N<2. `pgt` is the [N, max_pages, 3] int32 page-table block
// (stride pt_stride = max_pages*3); n_pages_v[i] is branch i's page count.
int
shared_prefix_pages_(const std::int32_t* pgt,
                     const std::vector<int>& n_pages_v, int N,
                     std::size_t pt_stride)
{
  if (N < 2 || pgt == nullptr) { return 0; }
  int minp = n_pages_v[0];
  for (int i = 1; i < N; ++i) {
    if (n_pages_v[(std::size_t)i] < minp) { minp = n_pages_v[(std::size_t)i]; }
  }
  int sp = 0;
  while (sp < minp) {
    const std::int32_t pid0 = pgt[(std::size_t)sp * 3 + 0];
    bool same = true;
    for (int i = 1; i < N && same; ++i) {
      if (pgt[(std::size_t)i * pt_stride + (std::size_t)sp * 3 + 0] != pid0) {
        same = false;
      }
    }
    if (!same) { break; }
    ++sp;
  }
  return sp;
}
inline float
bf16_to_f32_(std::uint16_t b)
{
  std::uint32_t bits = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}
inline std::uint16_t
f32_to_bf16_(float f)
{
  std::uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  // round-to-nearest-even
  b += 0x7fffu + ((b >> 16) & 1u);
  return (std::uint16_t)(b >> 16);
}
// Read a 16-bit compute-element buffer (f16 or bf16) into f32.
inline void
read_elt_(const void* src, float* dst, std::size_t n, bool bf16)
{
  if (bf16) {
    const auto* s = static_cast<const std::uint16_t*>(src);
    for (std::size_t i = 0; i < n; ++i) { dst[i] = bf16_to_f32_(s[i]); }
  } else {
    const auto* s = static_cast<const _Float16*>(src);
    for (std::size_t i = 0; i < n; ++i) { dst[i] = (float)s[i]; }
  }
}
}  // namespace

std::unique_ptr<MetalQwenModel>
MetalQwenModel::load(const std::string& model_dir,
                     metal_compute::MetalCompute* mc, const Config& cfg)
{
  if (mc == nullptr || !mc->valid()) {
    return nullptr;
  }
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts) {
    return nullptr;
  }

  auto m = std::unique_ptr<MetalQwenModel>(new MetalQwenModel());
  m->_cfg = cfg;
  m->_mc = mc;

  // ---- Mixed-precision affine (OptiQ) detection ----------------------
  // Infer an affine linear's bit width from its packed-weight column count
  // vs its input dim K: weight is [N, K*bits/32], so bits = wcols*32/K. A
  // k-quant weight (raw blocks, no .scales) returns 0. If a checkpoint mixes
  // 4-bit and 8-bit linears (mlx-optiq sensitivity quant), take the de-fused
  // per-tensor path; uniform 4-bit (standard Qwen) and uniform 8-bit (ASR,
  // cfg.quant_bits==8) both stay on the fused single-width path unchanged.
  auto affine_bits = [&](const std::string& pfx, int K) -> int {
    const auto* wi = wts->info(pfx + ".weight");
    if (wi == nullptr || wi->shape.size() < 2 || K <= 0) { return 0; }
    return (int)((wi->shape[1] * 32) / K);
  };
  {
    bool saw4 = false, saw8 = false;
    auto note = [&](const std::string& nm, int K) {
      const int b = affine_bits(nm, K);
      if (b == 4) { saw4 = true; } else if (b == 8) { saw8 = true; }
    };
    for (int L = 0; L < cfg.n_layers; ++L) {
      const std::string p =
          cfg.weight_prefix + "model.layers." + std::to_string(L) + ".";
      if (cfg.layer_is_full(L)) {
        note(p + "self_attn.q_proj", cfg.hidden);
        note(p + "self_attn.k_proj", cfg.hidden);
        note(p + "self_attn.v_proj", cfg.hidden);
        note(p + "self_attn.o_proj", cfg.qd());
      } else {
        note(p + "linear_attn.in_proj_qkv", cfg.hidden);
        note(p + "linear_attn.in_proj_z", cfg.hidden);
        note(p + "linear_attn.out_proj", cfg.value_dim());
      }
      note(p + "mlp.gate_proj", cfg.hidden);
      note(p + "mlp.up_proj", cfg.hidden);
      note(p + "mlp.down_proj", cfg.ffn_inner);
    }
    m->_mixed = saw4 && saw8;
    // Embed/lm_head bit width (q6k k-quant -> 0 -> fall back to the scalar).
    const int eb = affine_bits(cfg.weight_prefix + "model.embed_tokens",
                               cfg.hidden);
    m->_embed_bits = (eb == 4 || eb == 8) ? eb : cfg.quant_bits;
    m->_lm_bits = m->_embed_bits;
  }

  const std::string sfx = cfg.use_bf16 ? "_bf16" : "";
  m->_lib_qmv = mc->load_library("affine_qmv" + sfx);
  m->_lib_qmm = mc->load_library("affine_qmm_steel" + sfx);
  m->_lib_rms = mc->load_library("rms_norm" + sfx);
  m->_lib_elt = mc->load_library("llm_elementwise" + sfx);
  m->_lib_rope = mc->load_library("rope" + sfx);
  m->_lib_sdpa = mc->load_library("sdpa" + sfx);
  m->_lib_gdn = mc->load_library("qwen3_5_gated_delta" + sfx);
  // 4-bit (Qwen3.5/Llama) vs 8-bit (Qwen3-ASR) quantized-linear kernels.
  const std::string g = cfg.quant_bits == 8 ? "w8g64" : "w4g64";
  m->_fn_qmv = m->_lib_qmv.function("affine_qmv_" + g);
  m->_fn_qmv_add = m->_lib_qmv.function("affine_qmv_" + g + "_add");
  m->_fn_qmv_swiglu = m->_lib_qmv.function("affine_qmv_swiglu_" + g);
  m->_fn_qmm = m->_lib_qmm.function("affine_qmm_steel_" + g);
  m->_fn_qmm_swiglu = m->_lib_qmm.function("affine_qmm_swiglu_" + g);
  // Batched-decode GEMV (M-row qmv): weights read once across N branches,
  // at qmv bandwidth (steel GEMM is ~3x slower per weight-read at small M).
  // MAXM=2 (N>2 tiles along grid.z; N==1 stays on plain qmv). 4-bit only
  // -- 8-bit batches fall back to the steel GEMM.
  if (cfg.quant_bits != 8) {
    m->_fn_qmv_batch = m->_lib_qmv.function("affine_qmv_batch_w4g64");
    m->_fn_qmv_batch_swiglu =
        m->_lib_qmv.function("affine_qmv_batch_swiglu_w4g64");
  }
  m->_fn_rms = m->_lib_rms.function("rms_norm_f16");
  m->_fn_swiglu = m->_lib_elt.function("swiglu_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  m->_fn_mul_sigmoid = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_head_slice = m->_lib_elt.function("head_slice_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_kv_write_paged = m->_lib_elt.function("kv_write_paged_f16");
  m->_fn_rope_partial = m->_lib_rope.function("rope_partial_f16");
  // Fused per-head RMSNorm + partial RoPE (decode): folds q_norm+rope_q
  // and k_norm+rope_k into one dispatch each. Optional -> falls back to
  // the separate rms + rope kernels.
  m->_fn_rms_rope = m->_lib_rope.function("rms_rope_partial_f16");
  m->_fn_mrope = m->_lib_rope.function("mrope_partial_f16");
  m->_fn_sdpa_paged = m->_lib_sdpa.function("sdpa_paged_causal_f16");
  m->_fn_sdpa_paged_mb256 = m->_lib_sdpa.function("sdpa_paged_mb256_f16");
  // D<=128 multi-simdgroup paged decode (shared with the Llama path);
  // optional -- decode falls back to the scalar kernel if absent.
  m->_fn_sdpa_paged_mb = m->_lib_sdpa.function("sdpa_paged_mb_f16");
  m->_fn_sdpa_paged_qtile = m->_lib_sdpa.function("sdpa_paged_qtile_f16");
  // simdgroup_matrix key-split flash prefill (head_dim 256) -- M4-capable
  // (runs on every Apple GPU; the matmul2d sdpa_mma is M5-only).
  m->_fn_sdpa_paged_flash = m->_lib_sdpa.function("sdpa_paged_flash_f16");
  // Shared-prefix batched decode attention (head_dim 256). Optional: the
  // batched step falls back to the per-branch SDPA when either is absent.
  m->_fn_sdpa_shared_mb256 = m->_lib_sdpa.function("sdpa_shared_mb256_f16");
  m->_fn_sdpa_merge_mb256 = m->_lib_sdpa.function("sdpa_merge_mb256_f16");
  // Flash-decode-GQA serial attention (head_dim 256). Optional: serial decode
  // falls back to mb256 when either is absent or the shape is unsupported.
  m->_fn_sdpa_gqa = m->_lib_sdpa.function("sdpa_paged_gqa_mb256_f16");
  // Latency-optimal decode form (per-head simdgroup, UK key-unroll + vec4):
  // ~2x the all-G mb256 form at decode (head_dim % 128 == 0; D=256/128).
  m->_fn_sdpa_gqa_vec = m->_lib_sdpa.function("sdpa_paged_gqa_vec_f16");
  m->_fn_sdpa_gqa_merge = m->_lib_sdpa.function("sdpa_gqa_merge_f16");
  m->_fn_gdn_step = m->_lib_gdn.function("qwen3_5_gated_delta_step_f16");
  // Prefill GDN step variant: 4 dv per simdgroup, k/q loaded once/step ->
  // 1.33x the per-dv kernel at prefill T (token-exact, plain MSL so it runs
  // on every GPU). Used only for prefill (T large); decode (T=1) keeps v1.
  m->_fn_gdn_step_ndv4 =
      m->_lib_gdn.function("qwen3_5_gated_delta_step_ndv4_f16");
  if (const char* e = std::getenv("VPIPE_QWEN_GDN_V1")) {
    m->_gdn_force_v1 = (std::atoi(e) != 0);   // diagnostic A/B
  }
  m->_fn_gdn_conv1d = m->_lib_gdn.function("qwen3_5_gdn_conv1d_silu_f16");
  m->_fn_gdn_g_beta = m->_lib_gdn.function("qwen3_5_gdn_g_beta_f32");
  m->_fn_gdn_qk_norm = m->_lib_gdn.function("qwen3_5_gdn_qk_norm_f16");
  m->_fn_gdn_gated_rms = m->_lib_gdn.function("qwen3_5_gdn_gated_rms_f16");
  // Matrix-core (M5+) prefill GEMM: only when the GPU has hardware matrix
  // units and the model is 4-bit. dequant a projection weight -> dense
  // compute-elt scratch, then run dense matmul2d (matrix units). Gated so
  // older GPUs keep the steel quantized GEMM (byte-identical behaviour).
  // VPIPE_QWEN_NO_MMA=1 forces the steel path even on M5 (A/B + safety).
  if (cfg.quant_bits != 8 && mc->supports_matrix_cores()) {
    m->_lib_dequant = mc->load_library("affine_dequant" + sfx);
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma" + sfx);
    m->_fn_dequant = m->_lib_dequant.function("affine_dequant_w4g64");
    // Tile-adaptive dense matmul2d: 128x128 (8 simdgroups) for K <= 4096,
    // 128x256 for deeper K (down_proj) -- the 128x128 tile degrades past
    // K~4096 while 128x256 stays ~10 TFLOP/s at all depths (M5 tile sweep,
    // gemm_mma.tune). ~1.3-1.7x the prior 64x64 tile at prefill shapes.
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_fn_swiglu_inter = m->_lib_elt.function("swiglu_interleaved_f16");
    m->_use_mma = m->_fn_dequant.valid() && m->_fn_dense_mma.valid() &&
                  m->_fn_dense_mma_deep.valid() &&
                  m->_fn_swiglu_inter.valid();
    // Matrix-core flash attention is head_dim 256 only (Qwen3.5). Optional:
    // its absence just leaves prefill attention on the scalar qtile path.
    if (cfg.head_dim == 256) {
      m->_lib_sdpa_mma = mc->load_library("sdpa_mma" + sfx);
      m->_fn_sdpa_mma = m->_lib_sdpa_mma.function("sdpa_mma_f16");
    }
    if (const char* e = std::getenv("VPIPE_QWEN_MMA_MIN_M")) {
      m->_mma_min_m = std::atoi(e);
    }
    if (const char* e = std::getenv("VPIPE_QWEN_MMA_ATTN_MIN_N")) {
      m->_mma_attn_min_n = std::atoi(e);
    }
    if (const char* e = std::getenv("VPIPE_QWEN_NO_MMA")) {
      if (std::atoi(e) != 0) { m->_use_mma = false; }
    }
    if (const char* e = std::getenv("VPIPE_QWEN_MMA_NODQ")) {
      m->_skip_dequant = (std::atoi(e) != 0);   // diagnostic only
    }
  }
  // Pipelined-decode kernels (validated lazily in decode_pipelined, not
  // here, so non-pipelined model loads never depend on them).
  m->_fn_embed = m->_lib_elt.function(
      m->_embed_bits == 8 ? "dequant_embed_gather_w8_f16"
                          : "dequant_embed_gather_f16");
  m->_fn_argmax = m->_lib_elt.function("argmax_f16");
  m->_fn_sample = m->_lib_elt.function("sample_topp_f16");
  if (!m->_fn_qmv.valid() || !m->_fn_qmv_add.valid() ||
      !m->_fn_qmv_swiglu.valid() || !m->_fn_qmm.valid() ||
      !m->_fn_qmm_swiglu.valid() || !m->_fn_transpose.valid() ||
      !m->_fn_rms.valid() ||
      !m->_fn_swiglu.valid() || !m->_fn_residual.valid() ||
      !m->_fn_mul_sigmoid.valid() || !m->_fn_head_slice.valid() ||
      !m->_fn_kv_write_paged.valid() || !m->_fn_rope_partial.valid() ||
      !m->_fn_mrope.valid() || !m->_fn_sdpa_paged_mb256.valid() ||
      !m->_fn_sdpa_paged_qtile.valid() ||
      !m->_fn_sdpa_paged.valid() || !m->_fn_gdn_step.valid() ||
      !m->_fn_gdn_conv1d.valid() || !m->_fn_gdn_g_beta.valid() ||
      !m->_fn_gdn_qk_norm.valid() ||
      !m->_fn_gdn_gated_rms.valid()) {
    return nullptr;
  }
  if (const char* e = std::getenv("VPIPE_SDPA_MB_MIN")) {
    const int v = std::atoi(e);
    if (v >= 0) { m->_sdpa_mb_min = v; }
  }
  // Key-split flash prefill (M4 scalar path): on when the kernel loaded;
  // VPIPE_QWEN_NO_FLASH=1 forces the scalar qtile (A/B + safety).
  m->_flash_attn = m->_fn_sdpa_paged_flash.valid();
  if (const char* e = std::getenv("VPIPE_QWEN_NO_FLASH")) {
    if (std::atoi(e) != 0) { m->_flash_attn = false; }
  }
  // Shared-prefix batched decode attention: default ON, requires both
  // kernels; VPIPE_QWEN_SHARED_ATTN=0 forces the per-branch path (A/B).
  m->_shared_attn = m->_fn_sdpa_shared_mb256.valid()
                 && m->_fn_sdpa_merge_mb256.valid();
  if (const char* e = std::getenv("VPIPE_QWEN_SHARED_ATTN")) {
    if (std::atoi(e) == 0) { m->_shared_attn = false; }
  }
  // Flash-decode-GQA serial attention (head_dim <= 256, GQA G=Hq/Hkv<=4):
  // read each KV head once for all G query heads. Covers Qwen3.5 (D=256) and
  // the dense Qwen3-ASR text decoder (D=128, G=2), which both route here.
  // Default ON when capable; VPIPE_GQA_ATTN=0/1 overrides (legacy alias
  // VPIPE_QWEN_GQA_ATTN), VPIPE_GQA_SPLIT sets the split.
  const int gqa_g =
      (cfg.n_kv_heads > 0) ? cfg.n_heads / cfg.n_kv_heads : 0;
  const bool gqa_capable =
      m->_fn_sdpa_gqa.valid() && m->_fn_sdpa_gqa_merge.valid()
      && cfg.head_dim >= 32 && cfg.head_dim <= 256
      && (cfg.head_dim % 32 == 0) && cfg.n_kv_heads > 0
      && (cfg.n_heads % cfg.n_kv_heads == 0) && gqa_g >= 1 && gqa_g <= 4;
  m->_gqa_attn = gqa_capable;
  const char* gqa_e = std::getenv("VPIPE_GQA_ATTN");
  if (!gqa_e) { gqa_e = std::getenv("VPIPE_QWEN_GQA_ATTN"); }
  if (gqa_e) { m->_gqa_attn = gqa_capable && (std::atoi(gqa_e) != 0); }
  const char* gqa_nv = std::getenv("VPIPE_GQA_NO_VEC");
  if (gqa_nv && std::atoi(gqa_nv) != 0) { m->_gqa_vec = false; }
  const char* gqa_s = std::getenv("VPIPE_GQA_SPLIT");
  if (!gqa_s) { gqa_s = std::getenv("VPIPE_QWEN_GQA_SPLIT"); }
  if (gqa_s) {
    const int v = std::atoi(gqa_s);
    if (v >= 1 && v <= 64) { m->_gqa_split = v; }
  }
  // Decode category profiler: only when VPIPE_QWEN_CATPROF is set does the
  // decode step read VPIPE_QWEN_DUP_CAT (so production never pays a
  // per-step getenv). See encode_decode_step_ / metal_asr_decode_catprof.
  m->_catprof = std::getenv("VPIPE_QWEN_CATPROF") != nullptr;

  // Native k-quant (GGUF) detection: GgufQwen35Converter tags linear
  // weights with a k-quant family dtype ("Q4K"/"Q5K"/"Q6K") instead of the
  // affine U32+scales+biases triple. Probe a tensor present on every layer.
  {
    const auto* probe =
        wts->info(cfg.weight_prefix + "model.layers.0.mlp.gate_proj.weight");
    m->_kquant = probe != nullptr &&
        (probe->dtype == "Q4K" || probe->dtype == "Q5K" ||
         probe->dtype == "Q6K");
  }
  if (m->_kquant) {
    m->_fn_qmv_q4k = m->_lib_elt.function("qmv_q4k_f16");
    m->_fn_qmv_q5k = m->_lib_elt.function("qmv_q5k_f16");
    m->_fn_qmv_q6k = m->_lib_elt.function("qmv_q6k_v2_f16");
    if (!m->_fn_qmv_q6k.valid()) {
      m->_fn_qmv_q6k = m->_lib_elt.function("qmv_q6k_f16");
    }
    // Batched k-quant GEMV (MTP verify weight-bound matmul, weight read once
    // across the 2-row tile). Optional -- kqmv_batch_ loops single-row qmv when
    // unavailable or when VPIPE_GGUF_MTP_LOOPED_GEMV is set (A/B baseline).
    m->_fn_qmv_q4k_batch = m->_lib_elt.function("qmv_q4k_batch_f16");
    m->_fn_qmv_q5k_batch = m->_lib_elt.function("qmv_q5k_batch_f16");
    m->_fn_qmv_q6k_batch = m->_lib_elt.function("qmv_q6k_batch_f16");
    m->_fn_embed_q6k = m->_lib_elt.function("embed_gather_q6k_f16");
    m->_fn_dequant_q4k = m->_lib_elt.function("dequant_q4k_f16");
    m->_fn_dequant_q5k = m->_lib_elt.function("dequant_q5k_f16");
    m->_fn_dequant_q6k = m->_lib_elt.function("dequant_q6k_f16");
    m->_fn_copy = m->_lib_elt.function("copy_f16");
    m->_lib_dense = mc->load_library("dense_gemm" + sfx);
    m->_fn_dense_gemm = m->_lib_dense.function("dense_gemm_t_f16");
    m->_fn_dense_gemv = m->_lib_dense.function("dense_gemv_t_f16");
    if (!m->_fn_qmv_q4k.valid() || !m->_fn_qmv_q5k.valid() ||
        !m->_fn_qmv_q6k.valid() || !m->_fn_embed_q6k.valid() ||
        !m->_fn_dequant_q4k.valid() || !m->_fn_dequant_q5k.valid() ||
        !m->_fn_dequant_q6k.valid() || !m->_fn_copy.valid() ||
        !m->_fn_dense_gemm.valid() || !m->_fn_dense_gemv.valid()) {
      return nullptr;
    }
  }

  // Mixed-precision affine (OptiQ): the de-fused per-tensor path needs BOTH
  // bit widths' kernels (the 4-bit set is already loaded above as _fn_qmv/
  // _add/_qmm). Add the 8-bit qmv/qmv_add/steel-qmm, the affine dequant
  // (w4+w8) and the dense f16 GEMM/GEMV for the heterogeneous-bit prefill
  // (dequant each part into one f16 scratch -> one dense GEMM, the k-quant
  // strategy). The dequant + dense path is needed on EVERY GPU here, so load
  // them independent of matrix cores (M5 already loaded _fn_dequant +
  // dense_mma in the block above; dense_gemm_ picks the matrix units there).
  if (m->_mixed) {
    m->_fn_qmv8 = m->_lib_qmv.function("affine_qmv_w8g64");
    m->_fn_qmv8_add = m->_lib_qmv.function("affine_qmv_w8g64_add");
    m->_fn_qmv8_batch = m->_lib_qmv.function("affine_qmv_batch_w8g64");
    m->_fn_qmm8 = m->_lib_qmm.function("affine_qmm_steel_w8g64");
    if (!m->_fn_dequant.valid()) {       // not taken the M5 mma path above
      m->_lib_dequant = mc->load_library("affine_dequant" + sfx);
      m->_fn_dequant = m->_lib_dequant.function("affine_dequant_w4g64");
    }
    m->_fn_dequant8 = m->_lib_dequant.function("affine_dequant_w8g64");
    if (!m->_fn_dense_gemm.valid()) {    // not taken the k-quant path
      m->_lib_dense = mc->load_library("dense_gemm" + sfx);
      m->_fn_dense_gemm = m->_lib_dense.function("dense_gemm_t_f16");
      m->_fn_dense_gemv = m->_lib_dense.function("dense_gemv_t_f16");
    }
    if (!m->_fn_qmv8.valid() || !m->_fn_qmv8_add.valid() ||
        !m->_fn_qmm8.valid() || !m->_fn_dequant.valid() ||
        !m->_fn_dequant8.valid() || !m->_fn_dense_gemm.valid() ||
        !m->_fn_dense_gemv.valid()) {
      return nullptr;
    }
  }

  // Qwen 4-bit checkpoints store scales/biases and all the non-quantized
  // tensors (norms, conv1d, dt_bias) as BF16; our kernels want F16 (and
  // F32 for the recurrence's A_log/dt_bias). Convert at load.
  // Convert a tensor to the compute element type (f16 or bf16). In bf16
  // mode the Qwen checkpoint's BF16 scales/norms/conv pass through raw
  // (no conversion); only f32 sources are narrowed.
  const bool bf16 = cfg.use_bf16;
  auto to_elt = [&](const std::string& name) -> SharedBuffer {
    const auto* info = wts->info(name);
    if (info == nullptr) { return {}; }
    const std::string want = bf16 ? "BF16" : "F16";
    if (info->dtype == want) { return wts->load(name, mc); }
    SharedBuffer raw = wts->load(name, mc);
    if (raw.empty()) { return {}; }
    const std::size_t n = numel_(info->shape);
    SharedBuffer out = mc->make_shared_buffer(n * 2);
    auto* o = static_cast<std::uint16_t*>(out.contents());
    auto to16 = [&](float f) -> std::uint16_t {
      if (bf16) { return f32_to_bf16_(f); }
      _Float16 h = (_Float16)f; std::uint16_t b; std::memcpy(&b, &h, 2); return b;
    };
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = to16(bf16_to_f32_(s[i])); }
    } else if (info->dtype == "F16") {
      const auto* s = static_cast<const _Float16*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = to16((float)s[i]); }
    } else if (info->dtype == "F32") {
      const auto* s = static_cast<const float*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = to16(s[i]); }
    } else {
      return {};
    }
    return out;
  };
  auto to_f16 = to_elt;   // alias: all callers want the compute element type
  auto to_f32 = [&](const std::string& name) -> SharedBuffer {
    const auto* info = wts->info(name);
    if (info == nullptr) { return {}; }
    if (info->dtype == "F32") { return wts->load(name, mc); }
    SharedBuffer raw = wts->load(name, mc);
    if (raw.empty()) { return {}; }
    const std::size_t n = numel_(info->shape);
    SharedBuffer out = mc->make_shared_buffer(n * 4);
    auto* o = static_cast<float*>(out.contents());
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = bf16_to_f32_(s[i]); }
    } else {
      return {};
    }
    return out;
  };
  // Quantized linear: U32 weight (raw), F16 scales+biases (converted).
  auto qtri = [&](const std::string& pfx, SharedBuffer& w, SharedBuffer& s,
                  SharedBuffer& b) -> bool {
    w = wts->load(pfx + ".weight", mc);
    s = to_f16(pfx + ".scales");
    b = to_f16(pfx + ".biases");
    return !w.empty() && !s.empty() && !b.empty();
  };
  // Fuse several quantized projections that share the input (K = hidden)
  // into ONE [sum(N_i), K] matrix by row-concatenation -- one wide GEMM
  // replaces N separate ones (MLX issues them individually). Rows are
  // contiguous in the packed/scale/bias layout so concat is a memcpy.
  // Kc/8 u32 per weight row, Kc/64 (16-bit) per scale & bias row.
  auto fuse_q = [&](const std::vector<std::string>& names,
                    SharedBuffer& fw, SharedBuffer& fs, SharedBuffer& fb)
      -> bool {
    // u32 words per weight row = K * bits / 32 (4-bit: K/8, 8-bit: K/4).
    const std::size_t wrow =
        (std::size_t)cfg.hidden * cfg.quant_bits / 32;
    const std::size_t grow = (std::size_t)cfg.hidden / 64;  // 16-bit / scale row
    std::vector<SharedBuffer> ws, ss, bs;
    std::size_t Ntot = 0;
    for (const auto& nm : names) {
      SharedBuffer w, s, b;
      if (!qtri(nm, w, s, b)) { return false; }
      const auto* info = wts->info(nm + ".scales");   // [N, Kc/64]
      if (info == nullptr) { return false; }
      Ntot += (std::size_t)info->shape[0];
      ws.push_back(std::move(w)); ss.push_back(std::move(s));
      bs.push_back(std::move(b));
    }
    fw = mc->make_shared_buffer(Ntot * wrow * 4);
    fs = mc->make_shared_buffer(Ntot * grow * 2);
    fb = mc->make_shared_buffer(Ntot * grow * 2);
    if (fw.empty() || fs.empty() || fb.empty()) { return false; }
    std::size_t wo = 0, go = 0;
    for (std::size_t i = 0; i < ws.size(); ++i) {
      const std::size_t wn = ws[i].byte_size();
      const std::size_t gn = ss[i].byte_size();
      std::memcpy((char*)fw.contents() + wo, ws[i].contents(), wn);
      std::memcpy((char*)fs.contents() + go, ss[i].contents(), gn);
      std::memcpy((char*)fb.contents() + go, bs[i].contents(), gn);
      wo += wn; go += gn;
    }
    return true;
  };

  // Fused interleaved gate/up (row 2g=gate g, 2g+1=up g) -- same as the
  // Llama path; scales/biases are already F16 here.
  const int Kc = cfg.hidden, Fc = cfg.ffn_inner;
  const std::size_t wrow = (std::size_t)Kc * cfg.quant_bits / 32, grow = Kc / 64;
  auto interleave = [&](const SharedBuffer& gw, const SharedBuffer& gs,
                        const SharedBuffer& gb, const SharedBuffer& uw,
                        const SharedBuffer& us, const SharedBuffer& ub,
                        SharedBuffer& ow, SharedBuffer& os,
                        SharedBuffer& ob) -> bool {
    ow = mc->make_shared_buffer((std::size_t)2 * Fc * wrow * 4);
    os = mc->make_shared_buffer((std::size_t)2 * Fc * grow * 2);
    ob = mc->make_shared_buffer((std::size_t)2 * Fc * grow * 2);
    if (ow.empty() || os.empty() || ob.empty()) { return false; }
    const auto* gwp = static_cast<const std::uint32_t*>(gw.contents());
    const auto* uwp = static_cast<const std::uint32_t*>(uw.contents());
    auto* owp = static_cast<std::uint32_t*>(ow.contents());
    const auto* gsp = static_cast<const std::uint16_t*>(gs.contents());
    const auto* usp = static_cast<const std::uint16_t*>(us.contents());
    auto* osp = static_cast<std::uint16_t*>(os.contents());
    const auto* gbp = static_cast<const std::uint16_t*>(gb.contents());
    const auto* ubp = static_cast<const std::uint16_t*>(ub.contents());
    auto* obp = static_cast<std::uint16_t*>(ob.contents());
    for (std::size_t g = 0; g < (std::size_t)Fc; ++g) {
      std::memcpy(owp + (2 * g) * wrow, gwp + g * wrow, wrow * 4);
      std::memcpy(owp + (2 * g + 1) * wrow, uwp + g * wrow, wrow * 4);
      std::memcpy(osp + (2 * g) * grow, gsp + g * grow, grow * 2);
      std::memcpy(osp + (2 * g + 1) * grow, usp + g * grow, grow * 2);
      std::memcpy(obp + (2 * g) * grow, gbp + g * grow, grow * 2);
      std::memcpy(obp + (2 * g + 1) * grow, ubp + g * grow, grow * 2);
    }
    return true;
  };

  // ---- Native k-quant (GGUF) weight loaders ------------------------
  // The .weight tensors are raw k-quant blocks (no scales/biases); the
  // dtype tag picks the family. Q4_K_M is heterogeneous, so only same-type
  // projections fuse (q+k); v / o / in_proj parts / mlp stay separate.
  auto kq_of = [](const std::string& dt) -> KQ {
    if (dt == "Q4K") { return KQ::kQ4K; }
    if (dt == "Q5K") { return KQ::kQ5K; }
    if (dt == "Q6K") { return KQ::kQ6K; }
    return KQ::kNone;
  };
  auto load_kq = [&](const std::string& name, SharedBuffer& buf,
                     KQ& ty) -> bool {
    const auto* info = wts->info(name);
    if (info == nullptr) { return false; }
    ty = kq_of(info->dtype);
    if (ty == KQ::kNone) { return false; }
    buf = wts->load(name, mc);
    return !buf.empty();
  };
  // Fuse two same-family raw k-quant weights by row (byte) concatenation:
  // [N1+N2, K] -- the qmv reads it as one taller matrix. nrows = N1+N2.
  auto fuse_kq = [&](const std::string& n1, const std::string& n2,
                     SharedBuffer& buf, KQ& ty, int& nrows) -> bool {
    const auto* i1 = wts->info(n1);
    const auto* i2 = wts->info(n2);
    if (i1 == nullptr || i2 == nullptr || i1->dtype != i2->dtype) {
      return false;
    }
    ty = kq_of(i1->dtype);
    if (ty == KQ::kNone) { return false; }
    SharedBuffer b1 = wts->load(n1, mc), b2 = wts->load(n2, mc);
    if (b1.empty() || b2.empty()) { return false; }
    buf = mc->make_shared_buffer(b1.byte_size() + b2.byte_size());
    if (buf.empty()) { return false; }
    std::memcpy(buf.contents(), b1.contents(), b1.byte_size());
    std::memcpy(static_cast<char*>(buf.contents()) + b1.byte_size(),
                b2.contents(), b2.byte_size());
    nrows = static_cast<int>(i1->shape[0] + i2->shape[0]);
    return true;
  };
  // The two tiny Q8_0 alpha/beta projections come from the converter as
  // f32; narrow to f16 and concat into one [2*Hv, H] dense GEMV weight.
  auto load_ab = [&](const std::string& na, const std::string& nb,
                     SharedBuffer& buf) -> bool {
    SharedBuffer a = to_f16(na), b = to_f16(nb);
    if (a.empty() || b.empty()) { return false; }
    buf = mc->make_shared_buffer(a.byte_size() + b.byte_size());
    if (buf.empty()) { return false; }
    std::memcpy(buf.contents(), a.contents(), a.byte_size());
    std::memcpy(static_cast<char*>(buf.contents()) + a.byte_size(),
                b.contents(), b.byte_size());
    return true;
  };

  m->_layers.resize(cfg.n_layers);
  for (int L = 0; L < cfg.n_layers; ++L) {
    const std::string p =
        cfg.weight_prefix + "model.layers." + std::to_string(L) + ".";
    Layer& ly = m->_layers[L];
    ly.is_full = cfg.layer_is_full(L);
    ly.in_ln = to_f16(p + "input_layernorm.weight");
    ly.post_ln = to_f16(p + "post_attention_layernorm.weight");
    bool ok = !ly.in_ln.empty() && !ly.post_ln.empty();
    if (ly.is_full) {
      if (m->_kquant) {
        // q+k fuse (both q4_K); v is often q6_K in Q4_K_M -> separate.
        ok = ok && fuse_kq(p + "self_attn.q_proj.weight",
                           p + "self_attn.k_proj.weight",
                           ly.kqk, ly.kqk_t, ly.kqk_n);
        ok = ok && load_kq(p + "self_attn.v_proj.weight", ly.kqv, ly.kqv_t);
        ok = ok && load_kq(p + "self_attn.o_proj.weight", ly.kqo, ly.kqo_t);
      } else if (m->_mixed) {
        // De-fused per-tensor: q|k|v|o each keep their own triple + bits, the
        // decode writes them into the same _d_qfull[q|k|v] offsets the fused
        // path produces, the prefill dequants each into _w_deq for one GEMM.
        ok = ok && qtri(p + "self_attn.q_proj", ly.qw, ly.qs, ly.qb);
        ok = ok && qtri(p + "self_attn.k_proj", ly.kw, ly.ks, ly.kb);
        ok = ok && qtri(p + "self_attn.v_proj", ly.vw, ly.vs, ly.vb);
        ok = ok && qtri(p + "self_attn.o_proj", ly.ow, ly.os, ly.ob);
        ly.q_bits = affine_bits(p + "self_attn.q_proj", cfg.hidden);
        ly.k_bits = affine_bits(p + "self_attn.k_proj", cfg.hidden);
        ly.v_bits = affine_bits(p + "self_attn.v_proj", cfg.hidden);
        ly.o_bits = affine_bits(p + "self_attn.o_proj", cfg.qd());
      } else {
        // Fuse q|k|v into ONE [2*qd+2*kd, H] GEMM (q_proj is gated, 2*qd).
        ok = ok && fuse_q({p + "self_attn.q_proj", p + "self_attn.k_proj",
                           p + "self_attn.v_proj"}, ly.qw, ly.qs, ly.qb);
        ok = ok && qtri(p + "self_attn.o_proj", ly.ow, ly.os, ly.ob);
      }
      ly.q_norm = to_f16(p + "self_attn.q_norm.weight");
      ly.k_norm = to_f16(p + "self_attn.k_norm.weight");
      ok = ok && !ly.q_norm.empty() && !ly.k_norm.empty();
    } else {
      if (m->_kquant) {
        // in_proj is heterogeneous (qkv q5_K | z q4_K | a,b q8_0) -> the
        // decode path runs one qmv per part into the mixqkv offsets.
        ok = ok && load_kq(p + "linear_attn.in_proj_qkv.weight",
                           ly.kqkv, ly.kqkv_t);
        ok = ok && load_kq(p + "linear_attn.in_proj_z.weight",
                           ly.kqz, ly.kqz_t);
        ok = ok && load_ab(p + "linear_attn.in_proj_a.weight",
                           p + "linear_attn.in_proj_b.weight", ly.kqab);
        ok = ok && load_kq(p + "linear_attn.out_proj.weight",
                           ly.kqout, ly.kqout_t);
      } else if (m->_mixed) {
        // De-fused per-tensor in_proj: qkv->iqw, z->izw, a->iaw, b->ibw, each
        // its own triple + bits, written to the same _d_mixqkv[qkv|z|a|b]
        // offsets the fused path produces; out_proj->gow.
        ok = ok && qtri(p + "linear_attn.in_proj_qkv", ly.iqw, ly.iqs, ly.iqb);
        ok = ok && qtri(p + "linear_attn.in_proj_z", ly.izw, ly.izs, ly.izb);
        ok = ok && qtri(p + "linear_attn.in_proj_a", ly.iaw, ly.ias, ly.iab);
        ok = ok && qtri(p + "linear_attn.in_proj_b", ly.ibw, ly.ibs, ly.ibb);
        ok = ok && qtri(p + "linear_attn.out_proj", ly.gow, ly.gos, ly.gob);
        ly.qkv_bits = affine_bits(p + "linear_attn.in_proj_qkv", cfg.hidden);
        ly.z_bits = affine_bits(p + "linear_attn.in_proj_z", cfg.hidden);
        ly.a_bits = affine_bits(p + "linear_attn.in_proj_a", cfg.hidden);
        ly.b_bits = affine_bits(p + "linear_attn.in_proj_b", cfg.hidden);
        ly.gout_bits = affine_bits(p + "linear_attn.out_proj", cfg.value_dim());
      } else {
        // Fuse the four in_proj projections into ONE [Cd+vald+2Hv, H] GEMM.
        ok = ok && fuse_q({p + "linear_attn.in_proj_qkv",
                           p + "linear_attn.in_proj_z",
                           p + "linear_attn.in_proj_a",
                           p + "linear_attn.in_proj_b"},
                          ly.iqw, ly.iqs, ly.iqb);
        ok = ok && qtri(p + "linear_attn.out_proj", ly.gow, ly.gos, ly.gob);
      }
      ly.conv_w = to_f16(p + "linear_attn.conv1d.weight");
      ly.A_log = to_f32(p + "linear_attn.A_log");
      ly.dt_bias = to_f32(p + "linear_attn.dt_bias");
      ly.gdn_norm = to_f16(p + "linear_attn.norm.weight");
      ok = ok && !ly.conv_w.empty() && !ly.A_log.empty() &&
           !ly.dt_bias.empty() && !ly.gdn_norm.empty();
    }
    // MLP (every layer).
    if (m->_kquant) {
      // gate/up stay separate raw q4_K (two qmv + swiglu at decode); the
      // affine path's interleaved-swiglu kernel has no k-quant variant.
      ok = ok && load_kq(p + "mlp.gate_proj.weight", ly.kqgate, ly.kqgate_t);
      ok = ok && load_kq(p + "mlp.up_proj.weight", ly.kqup, ly.kqup_t);
      ok = ok && load_kq(p + "mlp.down_proj.weight", ly.kqdown, ly.kqdown_t);
    } else if (m->_mixed) {
      // De-fused per-tensor MLP: gate->guw, up->uw, down->dw (no interleave;
      // decode runs two qmv + swiglu, prefill two dense GEMMs + swiglu).
      ok = ok && qtri(p + "mlp.gate_proj", ly.guw, ly.gus, ly.gub);
      ok = ok && qtri(p + "mlp.up_proj", ly.uw, ly.us, ly.ub);
      ok = ok && qtri(p + "mlp.down_proj", ly.dw, ly.ds, ly.db);
      ly.gate_bits = affine_bits(p + "mlp.gate_proj", cfg.hidden);
      ly.up_bits = affine_bits(p + "mlp.up_proj", cfg.hidden);
      ly.down_bits = affine_bits(p + "mlp.down_proj", cfg.ffn_inner);
    } else {
      SharedBuffer gw, gs, gb, uw, us, ub;
      ok = ok && qtri(p + "mlp.gate_proj", gw, gs, gb);
      ok = ok && qtri(p + "mlp.up_proj", uw, us, ub);
      ok = ok && interleave(gw, gs, gb, uw, us, ub, ly.guw, ly.gus, ly.gub);
      ok = ok && qtri(p + "mlp.down_proj", ly.dw, ly.ds, ly.db);
    }
    if (!ok) {
      return nullptr;
    }
  }

  bool ok = true;
  if (!cfg.backbone_only) {
    if (m->_kquant) {
      // token_embd is a raw Q6_K table (tied lm_head): gathered + matvec'd
      // natively (embed_gather_q6k / qmv_q6k), no affine requant.
      m->_embed_q6k =
          wts->load(cfg.weight_prefix + "model.embed_tokens.q6k", mc);
      m->_embed_is_q6k = !m->_embed_q6k.empty();
      ok = m->_embed_is_q6k;
    } else {
      ok = qtri(cfg.weight_prefix + "model.embed_tokens", m->_embed_w,
                m->_embed_s, m->_embed_b);
    }
  }
  m->_final_ln = to_f16(cfg.weight_prefix + "model.norm.weight");
  if (!ok || m->_final_ln.empty()) {
    return nullptr;
  }
  if (!cfg.backbone_only) {
    m->_tied = cfg.tie_embeddings &&
               !wts->has(cfg.weight_prefix + "lm_head.weight");
    if (!m->_tied && !m->_kquant) {
      if (m->_mixed) {
        m->_lm_bits = affine_bits(cfg.weight_prefix + "lm_head", cfg.hidden);
        if (m->_lm_bits != 4 && m->_lm_bits != 8) { m->_lm_bits = 4; }
      }
      ok = qtri(cfg.weight_prefix + "lm_head", m->_lm_w, m->_lm_s, m->_lm_b);
      if (!ok) { return nullptr; }
    }
  }

  // Standard RoPE inv_freq over rotary_dim (base = rope_theta). Qwen3.5
  // has no rope scaling -- the partial rotary fraction is the only twist.
  const int half = cfg.rotary_dim / 2;
  m->_inv_freq = mc->make_shared_buffer((std::size_t)half * sizeof(float));
  auto* invf = static_cast<float*>(m->_inv_freq.contents());
  for (int i = 0; i < half; ++i) {
    invf[i] = 1.0f / std::pow(cfg.rope_theta,
                              (2.0f * (float)i) / (float)cfg.rotary_dim);
  }

  // Interleaved mROPE axis lookup over the rotary_dim/2 pairs (matches
  // mlx-vlm apply_interleaved_mrope): T everywhere, then H overrides
  // slots 1+3k (mrope_section[1] of them), W overrides 2+3k
  // (mrope_section[2]). Used only by the multimodal prefill.
  m->_mrope_axis.assign((std::size_t)half, 0);
  if (cfg.mrope_section.size() >= 3) {
    for (int k = 0; k < cfg.mrope_section[1]; ++k) {
      const int idx = 1 + 3 * k;
      if (idx < half) { m->_mrope_axis[(std::size_t)idx] = 1; }
    }
    for (int k = 0; k < cfg.mrope_section[2]; ++k) {
      const int idx = 2 + 3 * k;
      if (idx < half) { m->_mrope_axis[(std::size_t)idx] = 2; }
    }
  }

  // GDN q/k RMS-no-weight is rms_norm_f16 with a constant weight vector:
  // q uses inv_scale^2, k uses inv_scale, inv_scale = 1/sqrt(k_head_dim).
  const float inv_scale = 1.0f / std::sqrt((float)cfg.gdn_k_dim);
  m->_gdn_qscale = mc->make_shared_buffer((std::size_t)cfg.gdn_k_dim * 2);
  m->_gdn_kscale = mc->make_shared_buffer((std::size_t)cfg.gdn_k_dim * 2);
  auto* qsc = static_cast<std::uint16_t*>(m->_gdn_qscale.contents());
  auto* ksc = static_cast<std::uint16_t*>(m->_gdn_kscale.contents());
  auto store_elt = [&](float f) -> std::uint16_t {
    if (bf16) { return f32_to_bf16_(f); }
    _Float16 h = (_Float16)f; std::uint16_t b; std::memcpy(&b, &h, 2); return b;
  };
  for (int i = 0; i < cfg.gdn_k_dim; ++i) {
    qsc[i] = store_elt(inv_scale * inv_scale);
    ksc[i] = store_elt(inv_scale);
  }

  if (!cfg.backbone_only) {
    if (m->_embed_is_q6k) {
      m->_muxer = std::make_unique<MetalTokenMuxer>(
          mc, &m->_embed_q6k, cfg.hidden, bf16);
    } else {
      m->_muxer = std::make_unique<MetalTokenMuxer>(
          mc, &m->_embed_w, &m->_embed_s, &m->_embed_b, cfg.hidden, bf16,
          m->_embed_bits);
    }
    if (!m->_muxer->valid()) {
      return nullptr;
    }
  }

  // Shared ContextManager with its metal-compute backend (Spec::metal).
  // is_linear_layer marks the GDN layers (no paged K/V; conv/ssm state
  // instead); the SSM dims map the Qwen GDN config onto the generic
  // ssm_* spec fields.
  ContextManager::Spec cspec;
  cspec.metal = mc;
  cspec.n_layers = cfg.n_layers;
  cspec.n_kv_heads = cfg.n_kv_heads;
  cspec.head_dim = cfg.head_dim;
  cspec.max_seq = cfg.max_seq;
  cspec.page_tokens = cfg.page_tokens;
  cspec.max_pages = cfg.max_pages;
  cspec.is_linear_layer.assign((std::size_t)cfg.n_layers, false);
  for (int L = 0; L < cfg.n_layers; ++L) {
    if (!cfg.layer_is_full(L)) { cspec.is_linear_layer[(std::size_t)L] = true; }
  }
  cspec.ssm_conv_dim    = cfg.gdn_conv_dim;
  cspec.ssm_conv_kernel = cfg.gdn_conv_kernel;
  cspec.ssm_num_v_heads = cfg.gdn_v_heads;
  cspec.ssm_v_head_dim  = cfg.gdn_v_dim;
  cspec.ssm_k_head_dim  = cfg.gdn_k_dim;
  m->_ctx = std::make_unique<ContextManager>(cspec, nullptr);
  m->_cid = m->_ctx->acquire_root();
  if (!m->_cid.valid()) {
    return nullptr;
  }
  m->_pgtab = mc->make_shared_buffer(
      (std::size_t)m->_ctx->max_pages() * 3 * sizeof(std::int32_t));

  // ---- MTP head (mtp.safetensors), optional --------------------------
  // A bundled Multi-Token-Prediction drafter: the eh-fusion fc + pre-norms,
  // ONE full-attn decoder layer (4-bit affine, same shape as a main full
  // layer), and a final norm before the shared lm_head. Loaded if present;
  // its absence just leaves has_mtp() false. Read from the sibling file with
  // its own weight handle + 4-bit helpers (the main lambdas are bound to the
  // main `wts`). Embed/lm_head are shared with the main model (already loaded).
  if (!m->_kquant) {     // MTP path uses the affine embed/lm_head, not q6_K
    const std::string mtp_path = model_dir + "/mtp.safetensors";
    auto mwts = MetalLlamaWeights::open(mtp_path);
    if (mwts && mwts->has("mtp.fc.weight")) {
      auto& W = *mwts;
      // The MTP eh-fusion fc is a dense f16 GEMV; ensure the kernel is loaded
      // (it already is when _mixed/_kquant; a uniform-4-bit MTP model needs it
      // here). If unavailable the MTP head stays off (has_mtp() false).
      if (!m->_fn_dense_gemv.valid()) {
        m->_lib_dense = mc->load_library("dense_gemm" + sfx);
        m->_fn_dense_gemv = m->_lib_dense.function("dense_gemv_t_f16");
      }
      auto mto_elt = [&](const std::string& name) -> SharedBuffer {
        const auto* info = W.info(name);
        if (info == nullptr) { return {}; }
        const std::string want = bf16 ? "BF16" : "F16";
        if (info->dtype == want) { return W.load(name, mc); }
        SharedBuffer raw = W.load(name, mc);
        if (raw.empty()) { return {}; }
        const std::size_t n = numel_(info->shape);
        SharedBuffer out = mc->make_shared_buffer(n * 2);
        auto* o = static_cast<std::uint16_t*>(out.contents());
        auto to16 = [&](float f) -> std::uint16_t {
          if (bf16) { return f32_to_bf16_(f); }
          _Float16 h = (_Float16)f; std::uint16_t b;
          std::memcpy(&b, &h, 2); return b;
        };
        if (info->dtype == "BF16") {
          const auto* s = static_cast<const std::uint16_t*>(raw.contents());
          for (std::size_t i = 0; i < n; ++i) { o[i] = to16(bf16_to_f32_(s[i])); }
        } else if (info->dtype == "F32") {
          const auto* s = static_cast<const float*>(raw.contents());
          for (std::size_t i = 0; i < n; ++i) { o[i] = to16(s[i]); }
        } else { return {}; }
        return out;
      };
      // RMSNorm weight loaded with the +1 shift. The base Qwen3.5 stores norm
      // weights as (w-1) and adds 1 at runtime; mlx-optiq baked the +1 into the
      // MAIN model's norms during conversion (they read back as true weights),
      // but the MTP head's norms are stripped by mlx-lm so they were never
      // shifted -- they're still raw (w-1) here and need the +1 applied.
      auto mto_norm = [&](const std::string& name) -> SharedBuffer {
        SharedBuffer buf = mto_elt(name);
        if (buf.empty()) { return buf; }
        const std::size_t n = buf.byte_size() / 2;
        auto* o = static_cast<std::uint16_t*>(buf.contents());
        for (std::size_t i = 0; i < n; ++i) {
          if (bf16) {
            o[i] = f32_to_bf16_(bf16_to_f32_(o[i]) + 1.0f);
          } else {
            _Float16 h; std::memcpy(&h, &o[i], 2);
            h = (_Float16)((float)h + 1.0f); std::memcpy(&o[i], &h, 2);
          }
        }
        return buf;
      };
      // 4-bit affine triple (raw u32 weight + f16 scales/biases).
      auto mqtri = [&](const std::string& pfx, SharedBuffer& w,
                       SharedBuffer& s, SharedBuffer& b) -> bool {
        w = W.load(pfx + ".weight", mc);
        s = mto_elt(pfx + ".scales");
        b = mto_elt(pfx + ".biases");
        return !w.empty() && !s.empty() && !b.empty();
      };
      const int Hh = cfg.hidden, Fc = cfg.ffn_inner;
      const std::size_t wrow = (std::size_t)Hh * 4 / 32, grow = Hh / 64;
      // Row-concat q|k|v into one [2*qd+2*kd, H] (4-bit), like the main fuse_q.
      auto mfuse = [&](const std::vector<std::string>& names, SharedBuffer& fw,
                       SharedBuffer& fs, SharedBuffer& fb) -> bool {
        std::vector<SharedBuffer> ws, ss, bs; std::size_t Nt = 0;
        for (const auto& nm : names) {
          SharedBuffer w, s, b;
          if (!mqtri(nm, w, s, b)) { return false; }
          const auto* in = W.info(nm + ".scales");
          if (in == nullptr) { return false; }
          Nt += (std::size_t)in->shape[0];
          ws.push_back(std::move(w)); ss.push_back(std::move(s));
          bs.push_back(std::move(b));
        }
        fw = mc->make_shared_buffer(Nt * wrow * 4);
        fs = mc->make_shared_buffer(Nt * grow * 2);
        fb = mc->make_shared_buffer(Nt * grow * 2);
        if (fw.empty() || fs.empty() || fb.empty()) { return false; }
        std::size_t wo = 0, go = 0;
        for (std::size_t i = 0; i < ws.size(); ++i) {
          std::memcpy((char*)fw.contents() + wo, ws[i].contents(),
                      ws[i].byte_size());
          std::memcpy((char*)fs.contents() + go, ss[i].contents(),
                      ss[i].byte_size());
          std::memcpy((char*)fb.contents() + go, bs[i].contents(),
                      bs[i].byte_size());
          wo += ws[i].byte_size(); go += ss[i].byte_size();
        }
        return true;
      };
      // Interleave gate|up (row 2g=gate, 2g+1=up) -> [2*ffn, H] (4-bit).
      auto minter = [&](const SharedBuffer& gw, const SharedBuffer& gs,
                        const SharedBuffer& gb, const SharedBuffer& uw,
                        const SharedBuffer& us, const SharedBuffer& ub,
                        SharedBuffer& ow, SharedBuffer& os,
                        SharedBuffer& ob) -> bool {
        ow = mc->make_shared_buffer((std::size_t)2 * Fc * wrow * 4);
        os = mc->make_shared_buffer((std::size_t)2 * Fc * grow * 2);
        ob = mc->make_shared_buffer((std::size_t)2 * Fc * grow * 2);
        if (ow.empty() || os.empty() || ob.empty()) { return false; }
        const auto* gwp = static_cast<const std::uint32_t*>(gw.contents());
        const auto* uwp = static_cast<const std::uint32_t*>(uw.contents());
        auto* owp = static_cast<std::uint32_t*>(ow.contents());
        const auto* gsp = static_cast<const std::uint16_t*>(gs.contents());
        const auto* usp = static_cast<const std::uint16_t*>(us.contents());
        auto* osp = static_cast<std::uint16_t*>(os.contents());
        const auto* gbp = static_cast<const std::uint16_t*>(gb.contents());
        const auto* ubp = static_cast<const std::uint16_t*>(ub.contents());
        auto* obp = static_cast<std::uint16_t*>(ob.contents());
        for (std::size_t g = 0; g < (std::size_t)Fc; ++g) {
          std::memcpy(owp + (2 * g) * wrow, gwp + g * wrow, wrow * 4);
          std::memcpy(owp + (2 * g + 1) * wrow, uwp + g * wrow, wrow * 4);
          std::memcpy(osp + (2 * g) * grow, gsp + g * grow, grow * 2);
          std::memcpy(osp + (2 * g + 1) * grow, usp + g * grow, grow * 2);
          std::memcpy(obp + (2 * g) * grow, gbp + g * grow, grow * 2);
          std::memcpy(obp + (2 * g + 1) * grow, ubp + g * grow, grow * 2);
        }
        return true;
      };
      MtpHead& M = m->_mtp;
      Layer& ly = M.lyr;
      ly.is_full = true;
      bool ok = true;
      // fc is [H, 2H] dense; split each output row into its embedding half
      // (cols 0:H) and hidden half (cols H:2H), both contiguous [H, H].
      {
        SharedBuffer fcw = mto_elt("mtp.fc.weight");
        if (!fcw.empty()) {
          M.fc_e = mc->make_shared_buffer((std::size_t)Hh * Hh * 2);
          M.fc_h = mc->make_shared_buffer((std::size_t)Hh * Hh * 2);
          const auto* src = static_cast<const std::uint16_t*>(fcw.contents());
          auto* de = static_cast<std::uint16_t*>(M.fc_e.contents());
          auto* dh = static_cast<std::uint16_t*>(M.fc_h.contents());
          for (int i = 0; i < Hh; ++i) {
            std::memcpy(de + (std::size_t)i * Hh,
                        src + (std::size_t)i * 2 * Hh, (std::size_t)Hh * 2);
            std::memcpy(dh + (std::size_t)i * Hh,
                        src + (std::size_t)i * 2 * Hh + Hh, (std::size_t)Hh * 2);
          }
        }
      }
      M.prenorm_e = mto_norm("mtp.pre_fc_norm_embedding.weight");
      M.prenorm_h = mto_norm("mtp.pre_fc_norm_hidden.weight");
      M.final_norm = mto_norm("mtp.norm.weight");
      ly.in_ln = mto_norm("mtp.layers.0.input_layernorm.weight");
      ly.post_ln = mto_norm("mtp.layers.0.post_attention_layernorm.weight");
      ly.q_norm = mto_norm("mtp.layers.0.self_attn.q_norm.weight");
      ly.k_norm = mto_norm("mtp.layers.0.self_attn.k_norm.weight");
      ok = ok && !M.fc_e.empty() && !M.fc_h.empty() && !M.prenorm_e.empty() &&
           !M.prenorm_h.empty() && !M.final_norm.empty() &&
           !ly.in_ln.empty() && !ly.post_ln.empty() &&
           !ly.q_norm.empty() && !ly.k_norm.empty();
      ok = ok && mfuse({"mtp.layers.0.self_attn.q_proj",
                        "mtp.layers.0.self_attn.k_proj",
                        "mtp.layers.0.self_attn.v_proj"},
                       ly.qw, ly.qs, ly.qb);
      ok = ok && mqtri("mtp.layers.0.self_attn.o_proj", ly.ow, ly.os, ly.ob);
      {
        SharedBuffer gw, gs, gb, uw, us, ub;
        ok = ok && mqtri("mtp.layers.0.mlp.gate_proj", gw, gs, gb);
        ok = ok && mqtri("mtp.layers.0.mlp.up_proj", uw, us, ub);
        ok = ok && minter(gw, gs, gb, uw, us, ub, ly.guw, ly.gus, ly.gub);
        ok = ok && mqtri("mtp.layers.0.mlp.down_proj", ly.dw, ly.ds, ly.db);
      }
      ok = ok && m->_fn_dense_gemv.valid();
      if (ok) {
        // The MTP layer's own 1-layer full-attn paged context (reset each
        // drafting round; the draft window is tiny, so a small cap suffices).
        ContextManager::Spec ms;
        ms.metal = mc;
        ms.n_layers = 1;
        ms.n_kv_heads = cfg.n_kv_heads;
        ms.head_dim = cfg.head_dim;
        ms.max_seq = 256;
        ms.page_tokens = 256;
        ms.max_pages = 0;
        ms.is_linear_layer.assign(1, false);
        M.ctx = std::make_unique<ContextManager>(ms, nullptr);
        M.cid = M.ctx->acquire_root();
        M.ok = M.cid.valid();
        if (M.ok) {
          m->_mtp_h = mc->make_shared_buffer((std::size_t)cfg.hidden * 2);
          M.pgt = mc->make_shared_buffer(
              (std::size_t)M.ctx->max_pages() * 3 * sizeof(std::int32_t));
        }
      }
    }
  }

  // ---- MTP head (GGUF NextN block), optional -------------------------
  // The k-quant twin of the affine block above. The bundled NextN draft layer
  // lives in the SAME GGUF (exposed by the converter under the mtp.* names) as
  // native k-quant / q8_0 payloads, so it loads with the main k-quant helpers
  // off `wts` (not a sibling file). GGUF norms are TRUE weights (the +1 is
  // baked at conversion), so they load via plain to_f16 -- no (w-1)+1 shift
  // like the stripped mtp.safetensors norms. The eh-fusion fc arrives as f32
  // (q8_0 dequant), narrowed to f16 and split into its embedding/hidden halves.
  if (m->_kquant && wts->has("mtp.fc.weight")) {
    MtpHead& M = m->_mtp;
    Layer& ly = M.lyr;
    ly.is_full = true;
    const int Hh = cfg.hidden;
    bool ok = true;
    {
      SharedBuffer fcw = to_f16("mtp.fc.weight");   // [H, 2H] f16
      if (!fcw.empty()) {
        M.fc_e = mc->make_shared_buffer((std::size_t)Hh * Hh * 2);
        M.fc_h = mc->make_shared_buffer((std::size_t)Hh * Hh * 2);
        const auto* src = static_cast<const std::uint16_t*>(fcw.contents());
        auto* de = static_cast<std::uint16_t*>(M.fc_e.contents());
        auto* dh = static_cast<std::uint16_t*>(M.fc_h.contents());
        for (int i = 0; i < Hh; ++i) {
          std::memcpy(de + (std::size_t)i * Hh,
                      src + (std::size_t)i * 2 * Hh, (std::size_t)Hh * 2);
          std::memcpy(dh + (std::size_t)i * Hh,
                      src + (std::size_t)i * 2 * Hh + Hh, (std::size_t)Hh * 2);
        }
      }
    }
    M.prenorm_e = to_f16("mtp.pre_fc_norm_embedding.weight");
    M.prenorm_h = to_f16("mtp.pre_fc_norm_hidden.weight");
    M.final_norm = to_f16("mtp.norm.weight");
    ly.in_ln = to_f16("mtp.layers.0.input_layernorm.weight");
    ly.post_ln = to_f16("mtp.layers.0.post_attention_layernorm.weight");
    ly.q_norm = to_f16("mtp.layers.0.self_attn.q_norm.weight");
    ly.k_norm = to_f16("mtp.layers.0.self_attn.k_norm.weight");
    ok = ok && !M.fc_e.empty() && !M.fc_h.empty() && !M.prenorm_e.empty() &&
         !M.prenorm_h.empty() && !M.final_norm.empty() &&
         !ly.in_ln.empty() && !ly.post_ln.empty() &&
         !ly.q_norm.empty() && !ly.k_norm.empty();
    // q+k fuse (both q4_K); v / o / mlp parts stay separate raw k-quant.
    ok = ok && fuse_kq("mtp.layers.0.self_attn.q_proj.weight",
                       "mtp.layers.0.self_attn.k_proj.weight",
                       ly.kqk, ly.kqk_t, ly.kqk_n);
    ok = ok && load_kq("mtp.layers.0.self_attn.v_proj.weight",
                       ly.kqv, ly.kqv_t);
    ok = ok && load_kq("mtp.layers.0.self_attn.o_proj.weight",
                       ly.kqo, ly.kqo_t);
    ok = ok && load_kq("mtp.layers.0.mlp.gate_proj.weight",
                       ly.kqgate, ly.kqgate_t);
    ok = ok && load_kq("mtp.layers.0.mlp.up_proj.weight", ly.kqup, ly.kqup_t);
    ok = ok && load_kq("mtp.layers.0.mlp.down_proj.weight",
                       ly.kqdown, ly.kqdown_t);
    if (ok) {
      // The MTP layer's own 1-layer full-attn paged context (reset per round).
      ContextManager::Spec ms;
      ms.metal = mc;
      ms.n_layers = 1;
      ms.n_kv_heads = cfg.n_kv_heads;
      ms.head_dim = cfg.head_dim;
      ms.max_seq = 256;
      ms.page_tokens = 256;
      ms.max_pages = 0;
      ms.is_linear_layer.assign(1, false);
      M.ctx = std::make_unique<ContextManager>(ms, nullptr);
      M.cid = M.ctx->acquire_root();
      M.ok = M.cid.valid();
      if (M.ok) {
        m->_mtp_h = mc->make_shared_buffer((std::size_t)cfg.hidden * 2);
        M.pgt = mc->make_shared_buffer(
            (std::size_t)M.ctx->max_pages() * 3 * sizeof(std::int32_t));
      }
    }
  }
  return m;
}

MetalQwenModel::Config
MetalQwenModel::config_from(const ModelConfig& c)
{
  Config m;
  m.n_layers   = c.n_layers;
  m.hidden     = c.hidden;
  m.n_heads    = c.n_heads;
  m.n_kv_heads = c.n_kv_heads;
  m.head_dim   = c.head_dim;
  m.ffn_inner  = c.ffn_inner;
  m.vocab      = c.vocab_size;
  m.rope_theta = c.rope_theta;
  m.rms_eps    = c.rms_eps;
  m.rotary_dim =
      static_cast<int>(c.head_dim * c.partial_rotary_factor);
  m.tie_embeddings = c.tie_word_embeddings;
  if (c.mrope_section.size() >= 3) {
    m.mrope_section = c.mrope_section;
  }
  // Dense = no linear-attention layers (Qwen3-ASR text decoder).
  bool any_linear = false;
  for (bool b : c.is_linear_layer) { any_linear = any_linear || b; }
  m.dense = !any_linear;
  m.attn_output_gate = c.attn_output_gate;
  m.quant_bits = c.quantization.bits > 0 ? c.quantization.bits : 4;
  m.weight_prefix =
      (c.architecture == "Qwen3ASRForConditionalGeneration")
          ? "" : "language_model.";
  m.full_attn_interval =
      c.full_attention_interval > 0 ? c.full_attention_interval : 4;
  if (!m.dense) {
    m.gdn_conv_kernel = c.linear_conv_kernel;
    m.gdn_k_heads = c.linear_num_k_heads;
    m.gdn_v_heads = c.linear_num_v_heads;
    m.gdn_k_dim   = c.linear_k_head_dim;
    m.gdn_v_dim   = c.linear_v_head_dim;
    m.gdn_conv_dim =
        2 * c.linear_num_k_heads * c.linear_k_head_dim +
        c.linear_num_v_heads * c.linear_v_head_dim;
  }
  // Per-context logical length cap. The KV page pool grows lazily up to
  // a cap derived from this (max_pages = ceil(max_seq/page_tokens) * 16),
  // so a generous value costs only page-table bytes -- no KV memory until
  // the tokens are actually appended. 2048 was too small for streaming
  // video QA: realtime-vqa accumulates one ~160-256-token grid per
  // (fused) frame up to its max_frames_per_scene safety cap (default 64),
  // so the multimodal base prefill could need ~10k+ tokens. Overflowing
  // the cap made append() return an invalid slot -> empty logits ->
  // prefill returned -1 (every scene failed). 16384 holds 64 grids at up
  // to 256 tokens each plus the describe decode, so the stage's own frame
  // cap -- not the LM context -- is the binding limit; text / audio
  // decoders never approach it. This mirrors the MLX path, whose context
  // manager grows on demand with no fixed per-context cap.
  m.max_seq    = 16384;
  return m;
}

bool
MetalQwenModel::ensure_decode_scratch_()
{
  if (_dec_ready) { return true; }
  const Config& c = _cfg;
  const int H = c.hidden, qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, vald = c.value_dim(), Hv = c.gdn_v_heads;
  auto b = [&](std::size_t elems) { return _mc->make_shared_buffer(elems * 2); };
  _d_x = b((std::size_t)H);
  _d_hn = b((std::size_t)H);
  _d_logits = b((std::size_t)c.vocab);
  _d_qfull = b((std::size_t)(2 * qd + 2 * kd));
  _d_kbuf = b((std::size_t)kd);
  _d_vbuf = b((std::size_t)kd);
  _d_q3 = b((std::size_t)qd);
  _d_gate3 = b((std::size_t)qd);
  _d_attn = b((std::size_t)qd);
  _d_mixqkv = b((std::size_t)(Cd + vald + 2 * Hv));
  _d_zbuf = b((std::size_t)vald);
  _d_abuf = b((std::size_t)Hv);
  _d_bbuf = b((std::size_t)Hv);
  _d_convout = b((std::size_t)Cd);
  _d_gbuf = _mc->make_shared_buffer((std::size_t)Hv * 4);
  _d_betabuf = _mc->make_shared_buffer((std::size_t)Hv * 4);
  _d_ygdn = b((std::size_t)vald);
  _d_normout = b((std::size_t)vald);
  _d_sg = b((std::size_t)c.ffn_inner);
  if (_kquant || _mixed) {
    // k-quant / mixed-affine decode temps: [H] for qmv-then-residual-add
    // (k-quant only), [ffn] up-proj temp (both -- gate/up run as two
    // separate qmv + SwiGLU, no fused-swiglu kernel).
    _d_radd = b((std::size_t)H);
    _d_up = b((std::size_t)c.ffn_inner);
  }
  _d_tok_in = _mc->make_shared_buffer(sizeof(std::int32_t));
  _d_argmax_id = _mc->make_shared_buffer(sizeof(std::int32_t));
  // Flash-decode-GQA partials (f32): O [Hq,split,D], m/l [Hq,split].
  if (_gqa_attn) {
    const std::size_t sp = (std::size_t)_gqa_split;
    const std::size_t Hq = (std::size_t)c.n_heads, Dd = (std::size_t)c.head_dim;
    _d_gqa_oacc = _mc->make_shared_buffer(Hq * sp * Dd * sizeof(float));
    _d_gqa_m = _mc->make_shared_buffer(Hq * sp * sizeof(float));
    _d_gqa_l = _mc->make_shared_buffer(Hq * sp * sizeof(float));
  }
  _dec_ready = !_d_logits.empty();
  return _dec_ready;
}

std::int32_t
MetalQwenModel::decode_step_fast(ContextId cid, std::int32_t token_id,
                                 int rope_pos)
{
  const Config& c = _cfg;
  const int H = c.hidden;
  // Requires the in-stream embed gather + on-GPU argmax kernels. Checked
  // BEFORE any state mutation so the caller's fallback re-runs cleanly.
  // k-quant gathers from the raw Q6_K table (embed_q6k_); affine uses the
  // muxer's dequant gather. Both fold into the decode command buffer so the
  // token never round-trips to host (vs forward()'s separate embed submit).
  const bool embed_ok = _kquant ? _embed_is_q6k : _fn_embed.valid();
  if (!embed_ok || !_fn_argmax.valid()) {
    return std::numeric_limits<std::int32_t>::min();
  }
  if (!ensure_decode_scratch_() || _d_tok_in.empty() ||
      _d_argmax_id.empty()) {
    return std::numeric_limits<std::int32_t>::min();
  }

  ContextManager::AppendSlot slot = _ctx->append(cid, 1);
  if (!slot.valid()) { return -1; }
  const int pos = slot.position;
  const int rpos = (rope_pos < 0) ? pos : rope_pos;
  const std::size_t page_off =
      (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));

  *static_cast<std::int32_t*>(_d_tok_in.contents()) = token_id;

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // In-stream embed gather (token id -> _d_x), folded into the decode
    // command buffer (vs forward()'s separate muxer command buffer +
    // host memcpy). Same table as the muxer -> identical embedding.
    if (_kquant) {
      embed_q6k_(enc, _d_tok_in, 0, _d_x, 1);
    } else {
      enc.set_function(_fn_embed);
      enc.set_buffer(0, _d_tok_in);
      enc.set_buffer(1, _embed_w);
      enc.set_buffer(2, _embed_s);
      enc.set_buffer(3, _embed_b);
      enc.set_buffer(4, _d_x);
      enc.set_constant(5, H);
      enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
    }

    encode_decode_step_(enc, cid, pos, rpos, page_off, n_pages, slot,
                        _pgtab, 0);

    // On-GPU greedy argmax -> _d_argmax_id (1 int): no full [vocab] host
    // logit pull + host scan. argmax_f16 casts to f32 and breaks ties by
    // lowest index, matching the synchronous host argmax exactly.
    enc.set_function(_fn_argmax);
    enc.set_buffer(0, _d_logits);
    enc.set_buffer(1, _d_argmax_id);
    enc.set_constant(2, c.vocab);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
  }
  stream.commit().wait();
  return *static_cast<const std::int32_t*>(_d_argmax_id.contents());
}

// ---------------------------------------------------------------------
// Native k-quant (GGUF) matmul dispatch.
// ---------------------------------------------------------------------
void
MetalQwenModel::kqmv_(ComputeEncoder& enc, KQ type, const SharedBuffer& w,
                      const SharedBuffer& x, std::size_t xoff,
                      const SharedBuffer& y, std::size_t yoff, int K, int N)
{
  // y[0:N] = dequant(w[N,K]) @ x[0:K]. q4_K/q5_K: one simdgroup per row
  // (grid (32,N,1)); q6_K: the optimized v2 form (grid (32,ceil(N/8)*2,1),
  // tg (32,2,1)). Offsets are in elements (the helper converts to bytes).
  enc.set_buffer(0, w);
  enc.set_buffer(1, x, xoff * 2);
  enc.set_buffer(2, y, yoff * 2);
  enc.set_constant(3, K);
  enc.set_constant(4, N);
  if (type == KQ::kQ6K) {
    // q6k_v2: RPS=4 x NSG=2 = 8 rows/threadgroup.
    enc.set_function(_fn_qmv_q6k);
    enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
  } else {
    // q4k/q5k full-byte: 1 row/simdgroup x NSG=2 = 2 rows/threadgroup.
    enc.set_function(type == KQ::kQ5K ? _fn_qmv_q5k : _fn_qmv_q4k);
    enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2), 1}, {32, 2, 1});
  }
}

void
MetalQwenModel::kqmv_batch_(ComputeEncoder& enc, KQ type, const SharedBuffer& w,
                            const SharedBuffer& x, const SharedBuffer& y, int K,
                            int N, int M, int ystride, int yoff)
{
  // y[base+m, yoff + 0:N] = dequant(w[N,K]) @ x[base+m, 0:K] with the raw
  // k-quant weight read ONCE across each 2-row tile (compile-time MAXM=2; grid.z
  // tiles ceil(M/2)). At the depth-1 draft window M=2 this is one tile -> weight
  // read once -> a 2-token verify costs ~1 decode step (vs M looped qmv reading
  // the weight M times, or a full f16 dequant). Falls back to looped single-row
  // qmv when the batched kernel for `type` is unavailable, or when
  // VPIPE_GGUF_MTP_LOOPED_GEMV is set (the A/B baseline). M==1 -> plain qmv.
  static const bool force_looped =
      std::getenv("VPIPE_GGUF_MTP_LOOPED_GEMV") != nullptr;
  if (M == 1) {
    kqmv_(enc, type, w, x, 0, y, (std::size_t)yoff, K, N);
    return;
  }
  const metal_compute::ComputeFunction& fn =
      type == KQ::kQ6K ? _fn_qmv_q6k_batch
                       : (type == KQ::kQ5K ? _fn_qmv_q5k_batch
                                           : _fn_qmv_q4k_batch);
  if (force_looped || !fn.valid()) {
    for (int m = 0; m < M; ++m) {
      kqmv_(enc, type, w, x, (std::size_t)m * K, y,
            (std::size_t)m * ystride + yoff, K, N);
    }
    return;
  }
  enc.set_function(fn);
  enc.set_buffer(0, w);
  enc.set_buffer(1, x);
  enc.set_buffer(2, y);
  enc.set_constant(3, K);
  enc.set_constant(4, N);
  enc.set_constant(5, M);
  enc.set_constant(6, ystride);
  enc.set_constant(7, yoff);
  enc.dispatch({32, (unsigned)(((N + 1) / 2) * 2), (unsigned)((M + 1) / 2)},
               {32, 2, 1});
}

void
MetalQwenModel::dense_gemv_(ComputeEncoder& enc, const SharedBuffer& w,
                            const SharedBuffer& x, std::size_t xoff,
                            const SharedBuffer& y, std::size_t yoff,
                            int K, int N)
{
  // y[0:N] = w[N,K] @ x[0:K], all f16 (dense_gemv_t_f16: RPS=4 rows/simd).
  enc.set_function(_fn_dense_gemv);
  enc.set_buffer(0, x, xoff * 2);
  enc.set_buffer(1, w);
  enc.set_buffer(2, y, yoff * 2);
  enc.set_constant(3, K);
  enc.set_constant(4, N);
  enc.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
}

void
MetalQwenModel::dense_gemm_(ComputeEncoder& enc, const SharedBuffer& w,
                            const SharedBuffer& x, const SharedBuffer& y,
                            int K, int N, int M)
{
  // y[M,N] = x[M,K] @ w[N,K]^T, all f16, no bias. The caller (kqmm_ / the
  // fused-dequant QKV+GDN paths) has already dequantized the k-quant weight
  // into the f16 scratch, so this is the same plain f16 GEMM the affine
  // prefill runs post-dequant -- and on M5 it should ride the same matrix
  // units. Matrix-core (M5+) matmul2d when the units are present AND the
  // prompt is tall enough for the tiled kernel to amortize (M >= _mma_min_m):
  // tile-adaptive 128x128 for K < 6144, 128x256 for deeper K (down_proj),
  // matching the affine dense_mma path. The steel dense_gemm_t_f16 (BM/BN/
  // BK=32 tiles, tails clamped) stays the M4 / small-M fallback. _use_mma
  // already honours VPIPE_QWEN_NO_MMA, so that A/B toggle covers k-quant too.
  // Both kernels take the identical buffer/constant binding; only the
  // selected function + dispatch grid differ.
  const bool mma = _use_mma && M >= _mma_min_m;
  const bool deep = mma && K >= 6144;
  enc.set_function(mma ? (deep ? _fn_dense_mma_deep : _fn_dense_mma)
                       : _fn_dense_gemm);
  enc.set_buffer(0, x);
  enc.set_buffer(1, w);
  enc.set_buffer(2, w);          // bias slot unused (has_bias=0)
  enc.set_buffer(3, y);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, 0);
  if (mma) {
    const int BN = deep ? 256 : 128;
    enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                  (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
    return;
  }
  // BM=BN=32 tiles, WM=WN=2 -> 128 threads/tg ({32,2,2}); tid.x/tid.y index
  // the N/M tiles (grid.y carries M-tiles via tg.y=2, z=2 is the 4th dim).
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
}

void
MetalQwenModel::kqmm_(ComputeEncoder& enc, KQ type, const SharedBuffer& w,
                      const SharedBuffer& x, const SharedBuffer& y,
                      int K, int N, int M)
{
  // Prefill (M rows): dequant the k-quant weight into the reusable f16
  // scratch (sized to the largest N*K seen), then a dense f16 GEMM. The
  // M-independent dequant amortizes across the M prompt rows.
  const std::size_t need = (std::size_t)N * K * 2;
  if (_w_deq.empty() || _w_deq.byte_size() < need) {
    _w_deq = _mc->make_shared_buffer(need);
  }
  enc.set_function(type == KQ::kQ6K ? _fn_dequant_q6k
                  : (type == KQ::kQ5K ? _fn_dequant_q5k : _fn_dequant_q4k));
  enc.set_buffer(0, w);
  enc.set_buffer(1, _w_deq);
  enc.set_constant(2, N * K);
  enc.dispatch({(unsigned)(N * K), 1, 1}, {256, 1, 1});
  dense_gemm_(enc, _w_deq, x, y, K, N, M);
}

void
MetalQwenModel::kdequant_(ComputeEncoder& enc, KQ type, const SharedBuffer& w,
                          std::size_t dst_off, int count)
{
  // Dequant `count` k-quant elements of `w` into _w_deq[dst_off:], for a
  // fused prefill GEMM (heterogeneous parts share one f16 scratch).
  enc.set_function(type == KQ::kQ6K ? _fn_dequant_q6k
                  : (type == KQ::kQ5K ? _fn_dequant_q5k : _fn_dequant_q4k));
  enc.set_buffer(0, w);
  enc.set_buffer(1, _w_deq, dst_off * 2);
  enc.set_constant(2, count);
  enc.dispatch({(unsigned)count, 1, 1}, {256, 1, 1});
}

void
MetalQwenModel::kcopy_(ComputeEncoder& enc, const SharedBuffer& src,
                       std::size_t dst_off, int count)
{
  // copy_f16(src, _w_deq, off, count): fold an f16 sub-tensor (a/b) into the
  // fused dequant scratch alongside the k-quant parts.
  const int off = (int)dst_off;
  enc.set_function(_fn_copy);
  enc.set_buffer(0, src);
  enc.set_buffer(1, _w_deq);
  enc.set_constant(2, off);
  enc.set_constant(3, count);
  enc.dispatch({(unsigned)count, 1, 1}, {256, 1, 1});
}

void
MetalQwenModel::adequant_(ComputeEncoder& enc, const SharedBuffer& w,
                          const SharedBuffer& s, const SharedBuffer& b,
                          int bits, std::size_t dst_off, int N, int K)
{
  // Dequant an affine weight [N,K] (bits 4 or 8, group 64) into _w_deq at
  // element offset dst_off. 4-bit: K/8 u32 words/row; 8-bit: K/4. Grid is
  // one thread per word: {K*bits/32, N}. Byte/nibble order matches the
  // steel/qmv loaders, so the post-dequant dense GEMM equals the qmm.
  enc.set_function(bits == 8 ? _fn_dequant8 : _fn_dequant);
  enc.set_buffer(0, w);
  enc.set_buffer(1, s);
  enc.set_buffer(2, b);
  enc.set_buffer(3, _w_deq, dst_off * 2);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.dispatch({(unsigned)((std::size_t)K * bits / 32), (unsigned)N, 1},
               {64, 1, 1});
}

void
MetalQwenModel::aqmm_(ComputeEncoder& enc, const SharedBuffer& w,
                      const SharedBuffer& s, const SharedBuffer& b, int bits,
                      const SharedBuffer& x, const SharedBuffer& y, int K,
                      int N, int M)
{
  // Single affine projection prefill: dequant the whole weight into the
  // reusable f16 scratch, then a dense f16 GEMM (matrix-core on M5). The
  // affine twin of kqmm_; the M-independent dequant amortizes over M rows.
  const std::size_t need = (std::size_t)N * K * 2;
  if (_w_deq.empty() || _w_deq.byte_size() < need) {
    _w_deq = _mc->make_shared_buffer(need);
  }
  adequant_(enc, w, s, b, bits, 0, N, K);
  dense_gemm_(enc, _w_deq, x, y, K, N, M);
}

void
MetalQwenModel::embed_q6k_(ComputeEncoder& enc, const SharedBuffer& ids,
                           std::size_t ioff, const SharedBuffer& out, int n)
{
  enc.set_function(_fn_embed_q6k);
  enc.set_buffer(0, ids, ioff * 4);
  enc.set_buffer(1, _embed_q6k);
  enc.set_buffer(2, out);
  enc.set_constant(3, _cfg.hidden);
  enc.dispatch({(unsigned)_cfg.hidden, (unsigned)n, 1}, {256, 1, 1});
}

void
MetalQwenModel::encode_decode_step_(
    ComputeEncoder& enc, ContextId cid, int pos, int rpos,
    std::size_t page_off, int n_pages,
    const ContextManager::AppendSlot& slot,
    const SharedBuffer& pgtab, std::size_t pgtab_off, bool return_hidden)
{
  const Config& c = _cfg;
  const int H = c.hidden, D = c.head_dim;
  const int Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, K = c.gdn_conv_kernel;
  const float eps = c.rms_eps;
  const float scale = 1.0f / std::sqrt((float)D);
  const int page_tokens = _ctx->page_tokens();

  auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                 const SharedBuffer& w, const SharedBuffer& y,
                 std::size_t yoff, int R, int Hd) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, xin, xoff);
    enc.set_buffer(1, w);
    enc.set_buffer(2, y, yoff);
    enc.set_constant(3, Hd);
    enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  auto qmv = [&](const SharedBuffer& w, const SharedBuffer& s,
                 const SharedBuffer& b, const SharedBuffer& xin,
                 std::size_t xoff, const SharedBuffer& y, int Kk, int N) {
    enc.set_function(_fn_qmv);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, xin, xoff);
    enc.set_buffer(4, y);
    enc.set_constant(5, Kk);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  auto qmv_add = [&](const SharedBuffer& w, const SharedBuffer& s,
                     const SharedBuffer& b, const SharedBuffer& xin,
                     const SharedBuffer& y, const SharedBuffer& res, int Kk,
                     int N) {
    enc.set_function(_fn_qmv_add);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, xin);
    enc.set_buffer(4, y);
    enc.set_constant(5, Kk);
    enc.set_constant(6, N);
    enc.set_buffer(7, res);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  // Mixed-precision affine (OptiQ) per-tensor matvec: dequant(w at `bits`) @
  // x -> y[yoff_e : yoff_e+N]. Writes a sub-range of a wider de-fused buffer
  // (the q|k|v or qkv|z|a|b layout the fused qmv produces in one shot), so the
  // downstream hslices are unchanged. Same dispatch as qmv (w4 and w8 share
  // the qmv_fast grid). `bits` 4 -> _fn_qmv, 8 -> _fn_qmv8.
  auto amv = [&](const SharedBuffer& w, const SharedBuffer& s,
                 const SharedBuffer& b, int bits, const SharedBuffer& xin,
                 const SharedBuffer& y, std::size_t yoff_e, int Kk, int N) {
    enc.set_function(bits == 8 ? _fn_qmv8 : _fn_qmv);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, xin, 0);
    enc.set_buffer(4, y, yoff_e * 2);
    enc.set_constant(5, Kk);
    enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  auto amv_add = [&](const SharedBuffer& w, const SharedBuffer& s,
                     const SharedBuffer& b, int bits, const SharedBuffer& xin,
                     const SharedBuffer& y, const SharedBuffer& res, int Kk,
                     int N) {
    enc.set_function(bits == 8 ? _fn_qmv8_add : _fn_qmv_add);
    enc.set_buffer(0, w);
    enc.set_buffer(1, s);
    enc.set_buffer(2, b);
    enc.set_buffer(3, xin);
    enc.set_buffer(4, y);
    enc.set_constant(5, Kk);
    enc.set_constant(6, N);
    enc.set_buffer(7, res);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
  };
  auto hslice = [&](const SharedBuffer& in, std::size_t ioff,
                    const SharedBuffer& out, std::size_t ooff, int Hh,
                    int S, int W, int off, int block = 0, int gstride = 0) {
    enc.set_function(_fn_head_slice);
    enc.set_buffer(0, in, ioff);
    enc.set_buffer(1, out, ooff);
    enc.set_constant(2, Hh);
    enc.set_constant(3, S);
    enc.set_constant(4, W);
    enc.set_constant(5, off);
    enc.set_constant(6, block);
    enc.set_constant(7, gstride);
    enc.dispatch({(unsigned)(Hh * W), 1, 1}, {256, 1, 1});
  };
  auto rope = [&](const SharedBuffer& xb, int heads) {
    enc.set_function(_fn_rope_partial);
    enc.set_buffer(0, xb);
    enc.set_buffer(1, _inv_freq);
    enc.set_constant(2, heads);
    const int one = 1;
    enc.set_constant(3, one);
    enc.set_constant(4, D);
    enc.set_constant(5, c.rotary_dim);
    enc.set_constant(6, rpos);
    enc.dispatch({(unsigned)(c.rotary_dim / 2), 1, (unsigned)heads},
                 {(unsigned)(c.rotary_dim / 2), 1, 1});
  };
  // Fused per-head RMSNorm + partial RoPE (one dispatch vs rms + rope).
  auto rms_rope = [&](const SharedBuffer& xb, const SharedBuffer& w,
                      int heads) {
    enc.set_function(_fn_rms_rope);
    enc.set_buffer(0, xb);
    enc.set_buffer(1, w);
    enc.set_buffer(2, _inv_freq);
    enc.set_constant(3, D);
    enc.set_constant(4, c.rotary_dim);
    enc.set_constant(5, rpos);
    enc.set_constant(6, eps);
    enc.dispatch({256, (unsigned)heads, 1}, {256, 1, 1});
  };
  auto kv_write = [&](const SharedBuffer& src, const SharedBuffer& pool) {
    enc.set_function(_fn_kv_write_paged);
    enc.set_buffer(0, src);
    enc.set_buffer(1, pool, page_off);
    enc.set_constant(2, page_tokens);
    enc.set_constant(3, D);
    const int one = 1;
    enc.set_constant(4, one);
    const int zero = 0;
    enc.set_constant(5, zero);
    enc.set_constant(6, slot.slot_offset);
    enc.dispatch({(unsigned)D, 1, (unsigned)Hkv}, {(unsigned)D, 1, 1});
  };
  // Category profiler: D(cat, fn) runs fn() once, and AGAIN if profiling
  // that category (re-issues the dispatch(es) in the same command buffer).
  // The bench diffs decode time vs the `none` baseline -> delta == that
  // category's GPU cost (compute + its hazard-barrier drains). Off in
  // production (dup<0): fn() runs exactly once. Idempotent on the dense
  // ASR path (no accumulating state). Profiling forces the unfused
  // norm+rope so norm vs rope separate.
  enum { DC_PROJ = 1, DC_FFN = 2, DC_LMHEAD = 3, DC_ATTN = 4, DC_NORM = 5,
         DC_ROPE = 6, DC_MISC = 7 };
  // Production: dup<0 -> DUP(cat,fn) runs fn() exactly once (zero cost, no
  // getenv). Only a profiling session (VPIPE_QWEN_CATPROF set at load)
  // reads VPIPE_QWEN_DUP_CAT per step, so a within-process profiler can
  // toggle the duplicated category between passes -- the only reliable way
  // to compare categories at a STEADY GPU clock (cross-process runs start
  // at a low clock and ramp, corrupting cross-run deltas).
  int dup = -1;
  if (_catprof) {
    if (const char* e = std::getenv("VPIPE_QWEN_DUP_CAT")) {
      const std::string s = e;
      dup = (s == "proj") ? 1 : (s == "ffn") ? 2 : (s == "lmhead") ? 3
          : (s == "attn") ? 4 : (s == "norm") ? 5 : (s == "rope") ? 6
          : (s == "misc") ? 7 : 0;
    }
  }
  auto DUP = [&](int cat, auto&& fn) { fn(); if (dup == cat) { fn(); } };
  // k-quant residual add: _d_x[0:n] += _d_radd[0:n] (the qmv-then-add the
  // affine qmv_*_add fuses; native k-quant has no fused-add qmv variant).
  auto radd = [&](int n) {
    enc.set_function(_fn_residual);
    enc.set_buffer(0, _d_x);
    enc.set_buffer(1, _d_radd);
    enc.set_buffer(2, _d_x);
    enc.set_constant(3, n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
  };

  for (int L = 0; L < c.n_layers; ++L) {
    Layer& ly = _layers[L];
    DUP(DC_NORM, [&] { rms(_d_x, 0, ly.in_ln, _d_hn, 0, 1, H); });
    if (ly.is_full) {
      const SharedBuffer& kp = *_ctx->kpool(L);
      const SharedBuffer& vp = *_ctx->vpool(L);
      const bool gate = c.attn_output_gate;
      const int qdo = gate ? 2 * qd : qd;
      const int Nfqkv = qdo + 2 * kd;
      DUP(DC_PROJ, [&] {
        if (_kquant) {
          // q+k fused (q4_K) -> _d_qfull[0:kqk_n]; v (q6_K) -> the kd tail
          // [kqk_n : kqk_n+kd], reproducing the [q|k|v] fused layout.
          kqmv_(enc, ly.kqk_t, ly.kqk, _d_hn, 0, _d_qfull, 0, H, ly.kqk_n);
          kqmv_(enc, ly.kqv_t, ly.kqv, _d_hn, 0, _d_qfull,
                (std::size_t)ly.kqk_n, H, kd);
        } else if (_mixed) {
          // q|k|v each its own bit width into the [q(qdo)|k(kd)|v(kd)] layout.
          amv(ly.qw, ly.qs, ly.qb, ly.q_bits, _d_hn, _d_qfull, 0, H, qdo);
          amv(ly.kw, ly.ks, ly.kb, ly.k_bits, _d_hn, _d_qfull,
              (std::size_t)qdo, H, kd);
          amv(ly.vw, ly.vs, ly.vb, ly.v_bits, _d_hn, _d_qfull,
              (std::size_t)(qdo + kd), H, kd);
        } else {
          qmv(ly.qw, ly.qs, ly.qb, _d_hn, 0, _d_qfull, H, Nfqkv);
        }
      });
      DUP(DC_MISC, [&] {
        if (gate) {
          hslice(_d_qfull, 0, _d_q3, 0, Hq, 2 * D, D, 0);
          hslice(_d_qfull, 0, _d_gate3, 0, Hq, 2 * D, D, D);
        } else {
          hslice(_d_qfull, 0, _d_q3, 0, 1, Nfqkv, qd, 0);
        }
        hslice(_d_qfull, 0, _d_kbuf, 0, 1, Nfqkv, kd, qdo);
        hslice(_d_qfull, 0, _d_vbuf, 0, 1, Nfqkv, kd, qdo + kd);
      });
      if (_fn_rms_rope.valid() && dup < 0) {
        // Fused: q_norm+rope_q and k_norm+rope_k -> one dispatch each.
        rms_rope(_d_q3, ly.q_norm, Hq);
        rms_rope(_d_kbuf, ly.k_norm, Hkv);
      } else {
        DUP(DC_NORM, [&] {
          rms(_d_q3, 0, ly.q_norm, _d_q3, 0, Hq, D);
          rms(_d_kbuf, 0, ly.k_norm, _d_kbuf, 0, Hkv, D);
        });
        DUP(DC_ROPE, [&] { rope(_d_q3, Hq); rope(_d_kbuf, Hkv); });
      }
      DUP(DC_MISC, [&] { kv_write(_d_kbuf, kp); kv_write(_d_vbuf, vp); });
      // Multi-simdgroup paged attention past the long-context threshold
      // (the scalar kernel scans KV with only 32 lanes -> dominates the
      // per-layer critical path at high pos). D==256 (Qwen3.5) -> mb256
      // (BN=16); D<=128 (Qwen3-ASR) -> mb (BN=32, shared with Llama);
      // else scalar.
      const bool long_ctx = (pos + 1) >= _sdpa_mb_min;
      // Flash-decode-GQA (head_dim 256): read each KV head ONCE for all G
      // query heads (scan) + position-split merge -> G x less KV bandwidth
      // than mb256's per-q-head re-scan. Writes the same _d_attn [Hq*D].
      const bool use_gqa = _gqa_attn && long_ctx && (D <= 256)
          && (D % 32 == 0) && !_d_gqa_oacc.empty();
      if (use_gqa) {
        const int sp = _gqa_split;
        // Per-head UK+vec4 form (~2x the all-G mb256 form at decode -- Qwen
        // full-attn decode is latency-bound; see gqa_decode_micro). vec needs
        // head_dim % 128 == 0 (D=256/128); else the all-G kernel.
        const bool use_vec =
            _gqa_vec && (D % 128 == 0) && _fn_sdpa_gqa_vec.valid();
        const int G = Hq / Hkv;
        DUP(DC_ATTN, [&] {
          enc.set_function(use_vec ? _fn_sdpa_gqa_vec : _fn_sdpa_gqa);
          enc.set_buffer(0, _d_q3);
          enc.set_buffer(1, kp);
          enc.set_buffer(2, vp);
          enc.set_buffer(3, _d_gqa_oacc);
          enc.set_buffer(4, _d_gqa_m);
          enc.set_buffer(5, _d_gqa_l);
          enc.set_constant(6, scale);
          enc.set_constant(7, D);
          enc.set_constant(8, Hq);
          enc.set_constant(9, Hkv);
          enc.set_constant(10, pos);
          enc.set_constant(11, page_tokens);
          enc.set_constant(12, n_pages);
          enc.set_buffer(13, pgtab, pgtab_off);
          enc.set_constant(14, sp);
          if (use_vec) {
            enc.dispatch({32, (unsigned)Hq, (unsigned)sp},
                         {32, (unsigned)G, 1});
          } else {
            enc.dispatch({32, (unsigned)Hkv, (unsigned)sp}, {32, 1, 1});
          }
          enc.set_function(_fn_sdpa_gqa_merge);
          enc.set_buffer(0, _d_gqa_oacc);
          enc.set_buffer(1, _d_gqa_m);
          enc.set_buffer(2, _d_gqa_l);
          enc.set_buffer(3, _d_attn);
          enc.set_constant(4, D);
          enc.set_constant(5, sp);
          enc.set_constant(6, Hq);
          enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
        });
      } else {
      const bool use_mb256 = long_ctx && (D == 256);
      const bool use_mb128 =
          long_ctx && (D <= 128) && _fn_sdpa_paged_mb.valid();
      enc.set_function(use_mb256 ? _fn_sdpa_paged_mb256
                                 : (use_mb128 ? _fn_sdpa_paged_mb
                                              : _fn_sdpa_paged));
      enc.set_buffer(0, _d_q3);
      enc.set_buffer(1, kp);
      enc.set_buffer(2, vp);
      enc.set_buffer(3, _d_attn);
      enc.set_constant(4, scale);
      enc.set_constant(5, D);
      enc.set_constant(6, Hq);
      enc.set_constant(7, Hkv);
      const int one = 1;
      enc.set_constant(8, one);
      enc.set_constant(9, pos);
      enc.set_constant(10, page_tokens);
      enc.set_constant(11, n_pages);
      enc.set_buffer(12, pgtab, pgtab_off);
      const unsigned w =
          use_mb256 ? 32u * 16u : (use_mb128 ? 32u * 32u : 32u);
      DUP(DC_ATTN, [&] { enc.dispatch({w, (unsigned)Hq, 1}, {w, 1, 1}); });
      }
      if (gate) {
        enc.set_function(_fn_mul_sigmoid);
        enc.set_buffer(0, _d_attn);
        enc.set_buffer(1, _d_gate3);
        enc.set_buffer(2, _d_attn);
        enc.set_constant(3, qd);
        enc.dispatch({(unsigned)qd, 1, 1}, {256, 1, 1});
      }
      DUP(DC_PROJ, [&] {
        if (_kquant) {
          kqmv_(enc, ly.kqo_t, ly.kqo, _d_attn, 0, _d_radd, 0, qd, H);
          radd(H);
        } else if (_mixed) {
          amv_add(ly.ow, ly.os, ly.ob, ly.o_bits, _d_attn, _d_x, _d_x, qd, H);
        } else {
          qmv_add(ly.ow, ly.os, ly.ob, _d_attn, _d_x, _d_x, qd, H);
        }
      });
    } else {
      // GDN recurrent state via the run-ahead ring: read the current slot,
      // write the next. The ring is OFF in the sync/decode path (read ==
      // write == canonical -> in-place, identical to before); pdecode depth>1
      // ping-pongs the slots so a speculative advance can be rolled back.
      const SharedBuffer* cs_r = _ctx->conv_read(cid, L);
      const SharedBuffer* cs_w = _ctx->conv_write(cid, L);
      const SharedBuffer* ss_r = _ctx->ssm_read(cid, L);
      const SharedBuffer* ss_w = _ctx->ssm_write(cid, L);
      const int one = 1;
      const int Nf = Cd + vald + 2 * Hv;
      if (_kquant) {
        // in_proj is heterogeneous: qkv (q5_K) + z (q4_K) + a|b (f16) each
        // run their own matmul into the [qkv|z|a|b] mixqkv layout; the
        // hslices below then carve out z/a/b exactly as the fused path.
        kqmv_(enc, ly.kqkv_t, ly.kqkv, _d_hn, 0, _d_mixqkv, 0, H, Cd);
        kqmv_(enc, ly.kqz_t, ly.kqz, _d_hn, 0, _d_mixqkv, (std::size_t)Cd,
              H, vald);
        dense_gemv_(enc, ly.kqab, _d_hn, 0, _d_mixqkv,
                    (std::size_t)(Cd + vald), H, 2 * Hv);
      } else if (_mixed) {
        // qkv|z|a|b each its own bit width into the [qkv(Cd)|z(vald)|a(Hv)|
        // b(Hv)] fused layout the hslices below carve up.
        amv(ly.iqw, ly.iqs, ly.iqb, ly.qkv_bits, _d_hn, _d_mixqkv, 0, H, Cd);
        amv(ly.izw, ly.izs, ly.izb, ly.z_bits, _d_hn, _d_mixqkv,
            (std::size_t)Cd, H, vald);
        amv(ly.iaw, ly.ias, ly.iab, ly.a_bits, _d_hn, _d_mixqkv,
            (std::size_t)(Cd + vald), H, Hv);
        amv(ly.ibw, ly.ibs, ly.ibb, ly.b_bits, _d_hn, _d_mixqkv,
            (std::size_t)(Cd + vald + Hv), H, Hv);
      } else {
        qmv(ly.iqw, ly.iqs, ly.iqb, _d_hn, 0, _d_mixqkv, H, Nf);
      }
      hslice(_d_mixqkv, 0, _d_zbuf, 0, 1, Nf, vald, Cd);
      hslice(_d_mixqkv, 0, _d_abuf, 0, 1, Nf, Hv, Cd + vald);
      hslice(_d_mixqkv, 0, _d_bbuf, 0, 1, Nf, Hv, Cd + vald + Hv);
      enc.set_function(_fn_gdn_conv1d);
      enc.set_buffer(0, *cs_r);
      enc.set_buffer(1, _d_mixqkv);
      enc.set_buffer(2, ly.conv_w);
      enc.set_buffer(3, _d_convout);
      enc.set_constant(4, one);
      enc.set_constant(5, Cd);
      enc.set_constant(6, K);
      enc.set_constant(7, Nf);
      enc.set_constant(8, keyd);
      enc.set_buffer(9, *cs_w);
      enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
      const float inv_scale = 1.0f / std::sqrt((float)Dk);
      const float s_q = inv_scale * inv_scale, s_k = inv_scale;
      enc.set_function(_fn_gdn_qk_norm);
      enc.set_buffer(0, _d_convout);
      enc.set_constant(1, Hk);
      enc.set_constant(2, Dk);
      enc.set_constant(3, s_q);
      enc.set_constant(4, s_k);
      enc.set_constant(5, eps);
      enc.dispatch({128, (unsigned)(2 * Hk), 1}, {128, 1, 1});
      enc.set_function(_fn_gdn_g_beta);
      enc.set_buffer(0, _d_abuf);
      enc.set_buffer(1, _d_bbuf);
      enc.set_buffer(2, ly.A_log);
      enc.set_buffer(3, ly.dt_bias);
      enc.set_buffer(4, _d_gbuf);
      enc.set_buffer(5, _d_betabuf);
      enc.set_constant(6, Hv);
      enc.set_constant(7, one);
      enc.dispatch({(unsigned)Hv, 1, 1}, {256, 1, 1});
      enc.set_function(_fn_gdn_step);
      enc.set_buffer(0, _d_convout, 0);
      enc.set_buffer(1, _d_convout, (std::size_t)keyd * 2);
      enc.set_buffer(2, _d_convout, (std::size_t)2 * keyd * 2);
      enc.set_buffer(3, _d_gbuf);
      enc.set_buffer(4, _d_betabuf);
      enc.set_buffer(5, *ss_r);
      enc.set_buffer(6, _d_ygdn);
      enc.set_buffer(7, *ss_w);
      enc.set_constant(8, one);
      // Negative Hk selects strided GQA (hk = hv % Hk): the GGUF (llama.cpp)
      // stores GDN v-heads de-interleaved [0,2,..,30,1,3,..,31], vs the HF/
      // safetensors interleaved order that the contiguous mapping
      // (hk = hv / (Hv/Hk)) assumes. No-op when Hv == Hk (2B).
      enc.set_constant(9, _kquant ? -Hk : Hk);
      enc.set_constant(10, Hv);
      enc.dispatch({32, (unsigned)Dv, (unsigned)Hv}, {32, 4, 1});
      enc.set_function(_fn_gdn_gated_rms);
      enc.set_buffer(0, _d_ygdn);
      enc.set_buffer(1, ly.gdn_norm);
      enc.set_buffer(2, _d_zbuf);
      enc.set_buffer(3, _d_normout);
      enc.set_constant(4, Dv);
      enc.set_constant(5, eps);
      enc.dispatch({128, (unsigned)Hv, 1}, {128, 1, 1});
      if (_kquant) {
        kqmv_(enc, ly.kqout_t, ly.kqout, _d_normout, 0, _d_radd, 0, vald, H);
        radd(H);
      } else if (_mixed) {
        amv_add(ly.gow, ly.gos, ly.gob, ly.gout_bits, _d_normout, _d_x, _d_x,
                vald, H);
      } else {
        qmv_add(ly.gow, ly.gos, ly.gob, _d_normout, _d_x, _d_x, vald, H);
      }
    }
    DUP(DC_NORM, [&] { rms(_d_x, 0, ly.post_ln, _d_hn, 0, 1, H); });
    if (_kquant) {
      // gate + up as two q4_K qmv into _d_sg / _d_up, then SwiGLU + down.
      DUP(DC_FFN, [&] {
        kqmv_(enc, ly.kqgate_t, ly.kqgate, _d_hn, 0, _d_sg, 0, H, c.ffn_inner);
        kqmv_(enc, ly.kqup_t, ly.kqup, _d_hn, 0, _d_up, 0, H, c.ffn_inner);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, _d_sg);
        enc.set_buffer(1, _d_up);
        enc.set_buffer(2, _d_sg);
        enc.set_constant(3, c.ffn_inner);
        enc.dispatch({(unsigned)c.ffn_inner, 1, 1}, {256, 1, 1});
      });
      DUP(DC_FFN, [&] {
        kqmv_(enc, ly.kqdown_t, ly.kqdown, _d_sg, 0, _d_radd, 0,
              c.ffn_inner, H);
        radd(H);
      });
    } else if (_mixed) {
      // gate + up as two affine qmv (own bits) into _d_sg / _d_up, SwiGLU,
      // then down (own bits) with fused residual add.
      DUP(DC_FFN, [&] {
        amv(ly.guw, ly.gus, ly.gub, ly.gate_bits, _d_hn, _d_sg, 0, H,
            c.ffn_inner);
        amv(ly.uw, ly.us, ly.ub, ly.up_bits, _d_hn, _d_up, 0, H, c.ffn_inner);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, _d_sg);
        enc.set_buffer(1, _d_up);
        enc.set_buffer(2, _d_sg);
        enc.set_constant(3, c.ffn_inner);
        enc.dispatch({(unsigned)c.ffn_inner, 1, 1}, {256, 1, 1});
      });
      DUP(DC_FFN, [&] {
        amv_add(ly.dw, ly.ds, ly.db, ly.down_bits, _d_sg, _d_x, _d_x,
                c.ffn_inner, H);
      });
    } else {
      enc.set_function(_fn_qmv_swiglu);
      enc.set_buffer(0, ly.guw);
      enc.set_buffer(1, ly.gus);
      enc.set_buffer(2, ly.gub);
      enc.set_buffer(3, _d_hn);
      enc.set_buffer(4, _d_sg);
      enc.set_constant(5, H);
      enc.set_constant(6, 2 * c.ffn_inner);
      DUP(DC_FFN, [&] {
        enc.dispatch({32, (unsigned)(c.ffn_inner / 2), 1}, {32, 2, 1});
      });
      DUP(DC_FFN, [&] {
        qmv_add(ly.dw, ly.ds, ly.db, _d_sg, _d_x, _d_x, c.ffn_inner, H);
      });
    }
  }

  DUP(DC_NORM, [&] { rms(_d_x, 0, _final_ln, _d_hn, 0, 1, H); });
  // MOSS-TTS: the caller (decode_embedding_hidden) consumes _d_hn directly;
  // skip the lm_head (backbone_only models never loaded one).
  if (return_hidden) { return; }
  if (_kquant) {
    DUP(DC_LMHEAD, [&] {
      kqmv_(enc, KQ::kQ6K, _embed_q6k, _d_hn, 0, _d_logits, 0, H, c.vocab);
    });
  } else {
    // Mixed: lm_head is the (tied) 8-bit embed in OptiQ -> pick the w8 qmv.
    const int lb = _tied ? _embed_bits : _lm_bits;
    enc.set_function((_mixed && lb == 8) ? _fn_qmv8 : _fn_qmv);
    enc.set_buffer(0, _tied ? _embed_w : _lm_w);
    enc.set_buffer(1, _tied ? _embed_s : _lm_s);
    enc.set_buffer(2, _tied ? _embed_b : _lm_b);
    enc.set_buffer(3, _d_hn);
    enc.set_buffer(4, _d_logits);
    enc.set_constant(5, H);
    enc.set_constant(6, c.vocab);
    DUP(DC_LMHEAD, [&] {
      enc.dispatch({32, (unsigned)(c.vocab / 4), 1}, {32, 2, 1});
    });
  }
}

// ---------------------------------------------------------------------
// Batched (N-branch parallel) decode.
// ---------------------------------------------------------------------
bool
MetalQwenModel::ensure_bscratch_(BScratch& bs, int n)
{
  if (bs.n == n && !bs.logits.empty()) { return true; }
  const Config& c = _cfg;
  const int H = c.hidden, qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, vald = c.value_dim(), Hv = c.gdn_v_heads;
  const int ffn = c.ffn_inner;
  auto f16 = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  auto i32 = [&](std::size_t e) {
    return _mc->make_shared_buffer(e * sizeof(std::int32_t));
  };
  bs.x      = f16((std::size_t)n * H);
  bs.hn     = f16((std::size_t)n * H);
  bs.qfull  = f16((std::size_t)n * (2 * qd + 2 * kd));
  bs.q3     = f16((std::size_t)n * qd);
  bs.gate3  = f16((std::size_t)n * qd);
  bs.kbuf   = f16((std::size_t)n * kd);
  bs.vbuf   = f16((std::size_t)n * kd);
  bs.at     = f16((std::size_t)n * qd);
  bs.ao     = f16((std::size_t)n * H);
  bs.mixqkv = f16((std::size_t)n * (Cd + vald + 2 * Hv));
  bs.zbuf   = f16((std::size_t)n * vald);
  bs.abuf   = f16((std::size_t)n * Hv);
  bs.bbuf   = f16((std::size_t)n * Hv);
  bs.convout = f16((std::size_t)n * Cd);
  bs.gbuf   = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  bs.betabuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  bs.ygdn   = f16((std::size_t)n * vald);
  bs.normout = f16((std::size_t)n * vald);
  bs.sg     = f16((std::size_t)n * ffn);
  bs.logits = f16((std::size_t)n * c.vocab);
  bs.pgt    = i32((std::size_t)n * _ctx->max_pages() * 3);
  bs.tok_in = i32((std::size_t)n);
  bs.argmax_id = i32((std::size_t)n);
  // Shared-prefix attention partials (f32): O_acc [n, Hq, D]=[n*qd], m/l
  // [n, Hq]=[n*n_heads]. Only used by the head_dim-256 shared-attn path.
  auto f32 = [&](std::size_t e) {
    return _mc->make_shared_buffer(e * sizeof(float));
  };
  bs.shacc = f32((std::size_t)n * qd);
  bs.shm   = f32((std::size_t)n * c.n_heads);
  bs.shl   = f32((std::size_t)n * c.n_heads);
  bs.n = n;
  return !bs.logits.empty() && !bs.pgt.empty();
}

void
MetalQwenModel::qmm_auto_(
    ComputeEncoder& enc, int m, const SharedBuffer& w, const SharedBuffer& s,
    const SharedBuffer& b, const SharedBuffer& xin, const SharedBuffer& y,
    int Kk, int Nout)
{
  // 2..kQmvBatchMaxRows: batched GEMV (weights read once across the rows,
  // MAXM=2 -> grid.z tiles ceil(m/2)). Decode is DRAM-bound on the weight
  // read, so this beats the steel GEMM (compute-tiled for prefill) until m
  // is large enough that the steel matrix cores win.
  if (m > 1 && m <= kQmvBatchMaxRows && _fn_qmv_batch.valid()) {
    enc.set_function(_fn_qmv_batch);
    enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
    enc.set_buffer(3, xin); enc.set_buffer(4, y);
    enc.set_constant(5, Kk); enc.set_constant(6, Nout); enc.set_constant(7, m);
    enc.dispatch({32, (unsigned)(Nout / 4), (unsigned)((m + 1) / 2)},
                 {32, 2, 1});
    return;
  }
  if (m == 1) {
    enc.set_function(_fn_qmv);
    enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
    enc.set_buffer(3, xin); enc.set_buffer(4, y);
    enc.set_constant(5, Kk); enc.set_constant(6, Nout);
    enc.dispatch({32, (unsigned)(Nout / 4), 1}, {32, 2, 1});
    return;
  }
  // m == 0 shouldn't happen; large m (or 8-bit, no batched-GEMV) -> steel.
  enc.set_function(_fn_qmm);
  enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
  enc.set_buffer(3, xin); enc.set_buffer(4, y);
  enc.set_constant(5, Kk); enc.set_constant(6, Nout); enc.set_constant(7, m);
  enc.dispatch({(unsigned)(((Nout + 31) / 32) * 32),
               (unsigned)(((m + 31) / 32) * 2), 2}, {32, 2, 2});
}

void
MetalQwenModel::qmm_auto_swiglu_(
    ComputeEncoder& enc, int m, const SharedBuffer& w, const SharedBuffer& s,
    const SharedBuffer& b, const SharedBuffer& xin, const SharedBuffer& y,
    int Kk, int Nout)
{
  const unsigned gy = (unsigned)(Nout / 4);   // == ffn/2 (fused width 2*ffn)
  if (m > 1 && m <= kQmvBatchMaxRows && _fn_qmv_batch_swiglu.valid()) {
    enc.set_function(_fn_qmv_batch_swiglu);
    enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
    enc.set_buffer(3, xin); enc.set_buffer(4, y);
    enc.set_constant(5, Kk); enc.set_constant(6, Nout); enc.set_constant(7, m);
    enc.dispatch({32, gy, (unsigned)((m + 1) / 2)}, {32, 2, 1});
    return;
  }
  if (m == 1) {
    enc.set_function(_fn_qmv_swiglu);
    enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
    enc.set_buffer(3, xin); enc.set_buffer(4, y);
    enc.set_constant(5, Kk); enc.set_constant(6, Nout);
    enc.dispatch({32, gy, 1}, {32, 2, 1});
    return;
  }
  enc.set_function(_fn_qmm_swiglu);
  enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
  enc.set_buffer(3, xin); enc.set_buffer(4, y);
  enc.set_constant(5, Kk); enc.set_constant(6, Nout); enc.set_constant(7, m);
  enc.dispatch({(unsigned)(((Nout + 31) / 32) * 32),
               (unsigned)(((m + 31) / 32) * 2), 2}, {32, 2, 2});
}

void
MetalQwenModel::encode_batched_step_(
    ComputeEncoder& enc, BScratch& bs,
    std::span<const ContextId> cids,
    const std::vector<ContextManager::AppendSlot>& slots,
    const std::vector<std::size_t>& page_offs,
    const std::vector<int>& n_pages_v,
    const std::vector<int>& rope_pos_v,
    int shared_pages)
{
  const Config& c = _cfg;
  const int H = c.hidden, D = c.head_dim;
  const int Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, K = c.gdn_conv_kernel;
  const int ffn = c.ffn_inner;
  const float eps = c.rms_eps;
  const float scale = 1.0f / std::sqrt((float)D);
  const int N = bs.n;
  const int page_tokens = _ctx->page_tokens();
  const std::size_t pt_stride = (std::size_t)_ctx->max_pages() * 3;

  // --- batched helpers (M=N steel GEMM / N-row elementwise) -----------
  auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                 const SharedBuffer& w, const SharedBuffer& y,
                 std::size_t yoff, int R, int Hd) {
    enc.set_function(_fn_rms);
    enc.set_buffer(0, xin, xoff); enc.set_buffer(1, w);
    enc.set_buffer(2, y, yoff);
    enc.set_constant(3, Hd); enc.set_constant(4, eps);
    enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
  };
  // Weight-bound projection (q/k/v/o, gdn in/out, lm_head). Prefer the
  // size-based kernel via the shared qmm_auto_ entrance (qmv / batched
  // GEMV / steel by the active row count N) -- same selector the single
  // decode + prefill conceptually use, so the batched path adapts to the
  // CURRENT (possibly shrunk) branch count exactly like MLX's
  // quantized_matmul.
  auto qmm = [&](const SharedBuffer& w, const SharedBuffer& s,
                 const SharedBuffer& b, const SharedBuffer& xin,
                 const SharedBuffer& y, int Kk, int Nout) {
    qmm_auto_(enc, N, w, s, b, xin, y, Kk, Nout);
  };
  auto hslice = [&](const SharedBuffer& in, std::size_t ioff,
                    const SharedBuffer& out, std::size_t ooff, int Hh,
                    int S, int W, int off, int block = 0, int gstride = 0) {
    enc.set_function(_fn_head_slice);
    enc.set_buffer(0, in, ioff); enc.set_buffer(1, out, ooff);
    enc.set_constant(2, Hh); enc.set_constant(3, S); enc.set_constant(4, W);
    enc.set_constant(5, off); enc.set_constant(6, block);
    enc.set_constant(7, gstride);
    enc.dispatch({(unsigned)(Hh * W), 1, 1}, {256, 1, 1});
  };
  // Partial RoPE on one branch's q or k block (offset `xoff` bytes, `heads`
  // head-rows of D), at that branch's own position `rp` -- so branches need
  // NOT share a seq_len. The norm is done batched beforehand (rms).
  auto rope = [&](const SharedBuffer& xb, std::size_t xoff, int heads,
                  int rp) {
    enc.set_function(_fn_rope_partial);
    enc.set_buffer(0, xb, xoff); enc.set_buffer(1, _inv_freq);
    enc.set_constant(2, heads); const int one = 1; enc.set_constant(3, one);
    enc.set_constant(4, D); enc.set_constant(5, c.rotary_dim);
    enc.set_constant(6, rp);
    enc.dispatch({(unsigned)(c.rotary_dim / 2), 1, (unsigned)heads},
                 {(unsigned)(c.rotary_dim / 2), 1, 1});
  };
  auto residual = [&](const SharedBuffer& a, const SharedBuffer& bb,
                      const SharedBuffer& out, int nn) {
    enc.set_function(_fn_residual);
    enc.set_buffer(0, a); enc.set_buffer(1, bb); enc.set_buffer(2, out);
    enc.set_constant(3, nn);
    enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
  };

  for (int L = 0; L < c.n_layers; ++L) {
    Layer& ly = _layers[L];
    rms(bs.x, 0, ly.in_ln, bs.hn, 0, N, H);
    if (ly.is_full) {
      const SharedBuffer& kp = *_ctx->kpool(L);
      const SharedBuffer& vp = *_ctx->vpool(L);
      const bool gate = c.attn_output_gate;
      const int qdo = gate ? 2 * qd : qd;
      const int Nfqkv = qdo + 2 * kd;
      qmm(ly.qw, ly.qs, ly.qb, bs.hn, bs.qfull, H, Nfqkv);
      if (gate) {
        hslice(bs.qfull, 0, bs.q3, 0, N * Hq, 2 * D, D, 0, Hq, Nfqkv);
        hslice(bs.qfull, 0, bs.gate3, 0, N * Hq, 2 * D, D, D, Hq, Nfqkv);
      } else {
        hslice(bs.qfull, 0, bs.q3, 0, N, Nfqkv, qd, 0);
      }
      hslice(bs.qfull, 0, bs.kbuf, 0, N, Nfqkv, kd, qdo);
      hslice(bs.qfull, 0, bs.vbuf, 0, N, Nfqkv, kd, qdo + kd);
      // Batched per-head q/k RMSNorm (position-independent). RoPE is applied
      // per branch below at its own position (no transpose: the decode SDPA
      // reads q/k head-major [Hq|Hkv, D]).
      rms(bs.q3, 0, ly.q_norm, bs.q3, 0, N * Hq, D);
      rms(bs.kbuf, 0, ly.k_norm, bs.kbuf, 0, N * Hkv, D);
      // Per-branch RoPE at this branch's position + write K/V to its slot
      // (cheap, not weight-bound). Branches need NOT share a seq_len.
      //
      // Attention: when the N branches share a prefix (shared_pages>0) and
      // head_dim is 256, read that shared K/V ONCE for all branches (phase A,
      // sdpa_shared_mb256) then merge each branch's private pages (phase B,
      // sdpa_merge_mb256) -- avoids the per-branch re-read of the (large)
      // shared prefix. Otherwise each branch runs its own paged SDPA over its
      // whole page table. SDPA_SHARED_MAXN (=4) caps the batched query rows.
      const bool use_shared =
          _shared_attn && N > 1 && D == 256 && shared_pages > 0 && N <= 4;
      for (int i = 0; i < N; ++i) {
        const std::size_t qoff = (std::size_t)i * qd * 2;
        const std::size_t koff = (std::size_t)i * kd * 2;
        const int pos_i = slots[(std::size_t)i].position;
        rope(bs.q3, qoff, Hq, rope_pos_v[(std::size_t)i]);
        rope(bs.kbuf, koff, Hkv, rope_pos_v[(std::size_t)i]);
        // kv_write k_i, v_i.
        auto kvw = [&](const SharedBuffer& src, std::size_t soff,
                       const SharedBuffer& pool) {
          enc.set_function(_fn_kv_write_paged);
          enc.set_buffer(0, src, soff);
          enc.set_buffer(1, pool, page_offs[(std::size_t)i]);
          enc.set_constant(2, page_tokens); enc.set_constant(3, D);
          const int one = 1; enc.set_constant(4, one);
          const int zero = 0; enc.set_constant(5, zero);
          enc.set_constant(6, slots[(std::size_t)i].slot_offset);
          enc.dispatch({(unsigned)D, 1, (unsigned)Hkv}, {(unsigned)D, 1, 1});
        };
        kvw(bs.kbuf, koff, kp);
        kvw(bs.vbuf, koff, vp);
        if (use_shared) { continue; }   // attention runs in phase A/B below
        const bool long_ctx = (pos_i + 1) >= _sdpa_mb_min;
        const bool use_mb256 = long_ctx && (D == 256);
        const bool use_mb128 =
            long_ctx && (D <= 128) && _fn_sdpa_paged_mb.valid();
        enc.set_function(use_mb256 ? _fn_sdpa_paged_mb256
                                   : (use_mb128 ? _fn_sdpa_paged_mb
                                                : _fn_sdpa_paged));
        enc.set_buffer(0, bs.q3, qoff);
        enc.set_buffer(1, kp); enc.set_buffer(2, vp);
        enc.set_buffer(3, bs.at, qoff);
        enc.set_constant(4, scale); enc.set_constant(5, D);
        enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
        const int one = 1; enc.set_constant(8, one);
        enc.set_constant(9, pos_i); enc.set_constant(10, page_tokens);
        enc.set_constant(11, n_pages_v[(std::size_t)i]);
        enc.set_buffer(12, bs.pgt, (std::size_t)i * pt_stride * 4);
        const unsigned w =
            use_mb256 ? 32u * 16u : (use_mb128 ? 32u * 32u : 32u);
        enc.dispatch({w, (unsigned)Hq, 1}, {w, 1, 1});
      }
      if (use_shared) {
        // PHASE A: shared prefix over all N branches, read once. Branch 0's
        // page table (offset 0) is the shared prefix (identical across
        // branches). No causal mask -- the whole prefix precedes every
        // branch's decode position. Writes un-normalized O/m/l to shacc/shm/shl.
        enc.set_function(_fn_sdpa_shared_mb256);
        enc.set_buffer(0, bs.q3);
        enc.set_buffer(1, kp); enc.set_buffer(2, vp);
        enc.set_buffer(3, bs.shacc); enc.set_buffer(4, bs.shm);
        enc.set_buffer(5, bs.shl);
        enc.set_constant(6, scale); enc.set_constant(7, D);
        enc.set_constant(8, Hq); enc.set_constant(9, Hkv);
        enc.set_constant(10, N); enc.set_constant(11, page_tokens);
        enc.set_constant(12, shared_pages);
        enc.set_buffer(13, bs.pgt, 0);
        enc.dispatch({32u * 16u, (unsigned)Hq, 1}, {32u * 16u, 1, 1});
        // PHASE B: per branch -- scan its PRIVATE pages (offset past the
        // shared prefix) and merge with its phase-A shared partial.
        for (int i = 0; i < N; ++i) {
          const std::size_t qoff = (std::size_t)i * qd * 2;
          const int pos_i = slots[(std::size_t)i].position;
          const int n_priv = n_pages_v[(std::size_t)i] - shared_pages;
          enc.set_function(_fn_sdpa_merge_mb256);
          enc.set_buffer(0, bs.q3, qoff);
          enc.set_buffer(1, kp); enc.set_buffer(2, vp);
          enc.set_buffer(3, bs.at, qoff);
          enc.set_constant(4, scale); enc.set_constant(5, D);
          enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
          const int one = 1; enc.set_constant(8, one);
          enc.set_constant(9, pos_i); enc.set_constant(10, page_tokens);
          enc.set_constant(11, n_priv);
          enc.set_buffer(12, bs.pgt,
              ((std::size_t)i * pt_stride + (std::size_t)shared_pages * 3) * 4);
          enc.set_buffer(13, bs.shacc, (std::size_t)i * qd * sizeof(float));
          enc.set_buffer(14, bs.shm, (std::size_t)i * Hq * sizeof(float));
          enc.set_buffer(15, bs.shl, (std::size_t)i * Hq * sizeof(float));
          enc.dispatch({32u * 16u, (unsigned)Hq, 1}, {32u * 16u, 1, 1});
        }
      }
      if (gate) {
        enc.set_function(_fn_mul_sigmoid);
        enc.set_buffer(0, bs.at); enc.set_buffer(1, bs.gate3);
        enc.set_buffer(2, bs.at); enc.set_constant(3, N * qd);
        enc.dispatch({(unsigned)(N * qd), 1, 1}, {256, 1, 1});
      }
      qmm(ly.ow, ly.os, ly.ob, bs.at, bs.ao, qd, H);
      residual(bs.x, bs.ao, bs.x, N * H);
    } else {
      const int Nf = Cd + vald + 2 * Hv;
      qmm(ly.iqw, ly.iqs, ly.iqb, bs.hn, bs.mixqkv, H, Nf);
      hslice(bs.mixqkv, 0, bs.zbuf, 0, N, Nf, vald, Cd);
      hslice(bs.mixqkv, 0, bs.abuf, 0, N, Nf, Hv, Cd + vald);
      hslice(bs.mixqkv, 0, bs.bbuf, 0, N, Nf, Hv, Cd + vald + Hv);
      // Per-branch GDN: conv1d + delta-rule recurrence over this branch's
      // own conv_state / ssm_state (cheap, not weight-bound). n=1 per branch.
      for (int i = 0; i < N; ++i) {
        const SharedBuffer* cs = _ctx->conv_state(cids[(std::size_t)i], L);
        const SharedBuffer* ss = _ctx->ssm_state(cids[(std::size_t)i], L);
        const int one = 1;
        const std::size_t coff = (std::size_t)i * Cd * 2;       // convout row
        const std::size_t moff = (std::size_t)i * Nf * 2;       // mixqkv row
        enc.set_function(_fn_gdn_conv1d);
        enc.set_buffer(0, *cs); enc.set_buffer(1, bs.mixqkv, moff);
        enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, bs.convout, coff);
        enc.set_constant(4, one); enc.set_constant(5, Cd);
        enc.set_constant(6, K); enc.set_constant(7, Nf);
        enc.set_constant(8, keyd);
        enc.set_buffer(9, *cs);          // in-place (no run-ahead ring here)
        enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
        const float inv_scale = 1.0f / std::sqrt((float)Dk);
        const float s_q = inv_scale * inv_scale, s_k = inv_scale;
        enc.set_function(_fn_gdn_qk_norm);
        enc.set_buffer(0, bs.convout, coff);
        enc.set_constant(1, Hk); enc.set_constant(2, Dk);
        enc.set_constant(3, s_q); enc.set_constant(4, s_k);
        enc.set_constant(5, eps);
        enc.dispatch({128, (unsigned)(2 * Hk), 1}, {128, 1, 1});
        enc.set_function(_fn_gdn_g_beta);
        enc.set_buffer(0, bs.abuf, (std::size_t)i * Hv * 2);
        enc.set_buffer(1, bs.bbuf, (std::size_t)i * Hv * 2);
        enc.set_buffer(2, ly.A_log); enc.set_buffer(3, ly.dt_bias);
        enc.set_buffer(4, bs.gbuf, (std::size_t)i * Hv * 4);
        enc.set_buffer(5, bs.betabuf, (std::size_t)i * Hv * 4);
        enc.set_constant(6, Hv); enc.set_constant(7, one);
        enc.dispatch({(unsigned)Hv, 1, 1}, {256, 1, 1});
        enc.set_function(_fn_gdn_step);
        enc.set_buffer(0, bs.convout, coff);
        enc.set_buffer(1, bs.convout, coff + (std::size_t)keyd * 2);
        enc.set_buffer(2, bs.convout, coff + (std::size_t)2 * keyd * 2);
        enc.set_buffer(3, bs.gbuf, (std::size_t)i * Hv * 4);
        enc.set_buffer(4, bs.betabuf, (std::size_t)i * Hv * 4);
        enc.set_buffer(5, *ss);
        enc.set_buffer(6, bs.ygdn, (std::size_t)i * vald * 2);
        enc.set_buffer(7, *ss);
        enc.set_constant(8, one);
        enc.set_constant(9, _kquant ? -Hk : Hk);   // strided GQA for GGUF
        enc.set_constant(10, Hv);
        enc.dispatch({32, (unsigned)Dv, (unsigned)Hv}, {32, 4, 1});
        enc.set_function(_fn_gdn_gated_rms);
        enc.set_buffer(0, bs.ygdn, (std::size_t)i * vald * 2);
        enc.set_buffer(1, ly.gdn_norm);
        enc.set_buffer(2, bs.zbuf, (std::size_t)i * vald * 2);
        enc.set_buffer(3, bs.normout, (std::size_t)i * vald * 2);
        enc.set_constant(4, Dv); enc.set_constant(5, eps);
        enc.dispatch({128, (unsigned)Hv, 1}, {128, 1, 1});
      }
      qmm(ly.gow, ly.gos, ly.gob, bs.normout, bs.ao, vald, H);
      residual(bs.x, bs.ao, bs.x, N * H);
    }
    // MLP (batched over ALL N rows -- every branch needs its logits). The
    // fused gate/up GEMV picks its kernel by N via the shared entrance.
    rms(bs.x, 0, ly.post_ln, bs.hn, 0, N, H);
    qmm_auto_swiglu_(enc, N, ly.guw, ly.gus, ly.gub, bs.hn, bs.sg, H,
                     2 * ffn);
    qmm(ly.dw, ly.ds, ly.db, bs.sg, bs.ao, ffn, H);
    residual(bs.x, bs.ao, bs.x, N * H);
  }
  // Final norm + lm_head over ALL N rows.
  rms(bs.x, 0, _final_ln, bs.hn, 0, N, H);
  qmm(_tied ? _embed_w : _lm_w, _tied ? _embed_s : _lm_s,
      _tied ? _embed_b : _lm_b, bs.hn, bs.logits, H, c.vocab);
}

bool
MetalQwenModel::decode_batched_step(
    std::span<const ContextId>    cids,
    std::span<const std::int32_t> in_tokens,
    std::span<const std::int32_t> rope_pos,
    std::vector<float>&           out_logits)
{
  const int N = (int)cids.size();
  if (N <= 0 || (int)in_tokens.size() != N) { return false; }
  if (!rope_pos.empty() && (int)rope_pos.size() != N) { return false; }
  if (!_fn_embed.valid()) { return false; }
  // Mixed-precision affine de-fuses the projections, so the fused buffers the
  // batched encoder reads (ly.qw|iqw|guw) are unbuilt -> not yet supported;
  // the caller (realtime-vqa) falls back to per-branch serial decode.
  if (_mixed) { return false; }
  if (!ensure_bscratch_(_bdec, N)) { return false; }
  BScratch& bs = _bdec;
  const Config& c = _cfg;
  const int H = c.hidden;
  const std::size_t pt_stride = (std::size_t)_ctx->max_pages() * 3;

  std::vector<ContextManager::AppendSlot> slots((std::size_t)N);
  std::vector<std::size_t> page_offs((std::size_t)N);
  std::vector<int> n_pages_v((std::size_t)N);
  std::vector<int> rope_pos_v((std::size_t)N);
  auto* tok = static_cast<std::int32_t*>(bs.tok_in.contents());
  auto* pgt = static_cast<std::int32_t*>(bs.pgt.contents());
  for (int i = 0; i < N; ++i) {
    slots[(std::size_t)i] = _ctx->append(cids[(std::size_t)i], 1);
    if (!slots[(std::size_t)i].valid()) { return false; }
    page_offs[(std::size_t)i] =
        (std::size_t)slots[(std::size_t)i].page_id.v * _ctx->page_stride_bytes();
    n_pages_v[(std::size_t)i] = _ctx->fill_page_table(
        cids[(std::size_t)i], pgt + (std::size_t)i * pt_stride);
    // rope_pos[i] < 0 (or an empty span) => sequential text decode at the KV
    // slot position; >= 0 => the mROPE-advanced position (post-image).
    rope_pos_v[(std::size_t)i] =
        (rope_pos.empty() || rope_pos[(std::size_t)i] < 0)
            ? slots[(std::size_t)i].position
            : rope_pos[(std::size_t)i];
    tok[i] = in_tokens[(std::size_t)i];
  }
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // In-stream embed gather: N tokens -> bs.x[N, H].
    enc.set_function(_fn_embed);
    enc.set_buffer(0, bs.tok_in); enc.set_buffer(1, _embed_w);
    enc.set_buffer(2, _embed_s); enc.set_buffer(3, _embed_b);
    enc.set_buffer(4, bs.x); enc.set_constant(5, H);
    enc.dispatch({(unsigned)H, (unsigned)N, 1}, {256, 1, 1});
    encode_batched_step_(enc, bs, cids, slots, page_offs, n_pages_v,
                         rope_pos_v,
                         shared_prefix_pages_(pgt, n_pages_v, N, pt_stride));
  }
  stream.commit().wait();
  // [N, vocab] f16 -> host f32 (the caller samples each row).
  out_logits.resize((std::size_t)N * c.vocab);
  read_elt_(bs.logits.contents(), out_logits.data(),
            (std::size_t)N * c.vocab, c.use_bf16);
  return true;
}

std::vector<std::vector<std::int32_t>>
MetalQwenModel::decode_batched_argmax(
    std::span<const ContextId> cids,
    std::span<const std::int32_t> first_tokens, int n_steps)
{
  const int N = (int)cids.size();
  std::vector<std::vector<std::int32_t>> out((std::size_t)N);
  if (N == 0 || (int)first_tokens.size() != N || n_steps <= 0) { return out; }
  const Config& c = _cfg;
  std::vector<std::int32_t> cur(first_tokens.begin(), first_tokens.end());
  std::vector<float> logits;
  for (int s = 0; s < n_steps; ++s) {
    if (!decode_batched_step(
            cids, std::span<const std::int32_t>(cur.data(), cur.size()),
            std::span<const std::int32_t>(), logits)) {
      break;
    }
    for (int i = 0; i < N; ++i) {
      const float* row = logits.data() + (std::size_t)i * c.vocab;
      std::int32_t best = 0; float bv = row[0];
      for (int v = 1; v < c.vocab; ++v) {
        if (row[v] > bv) { bv = row[v]; best = v; }   // ties -> lowest index
      }
      out[(std::size_t)i].push_back(best);
      cur[(std::size_t)i] = best;
    }
  }
  return out;
}

// ---- Batched pipelined decode (bdecode_*) ----------------------------

void
MetalQwenModel::encode_sample_batched_(
    metal_compute::ComputeEncoder& enc,
    const metal_compute::SharedBuffer& gen_ids, int out_idx, int n,
    const GpuSamplerParams& sp, std::uint32_t step_seed)
{
  const int V = _cfg.vocab;
  for (int b = 0; b < n; ++b) {
    // gen_ids is [cap][N] step-major: row out_idx, column b.
    const std::size_t out_off =
        (std::size_t)(out_idx * n + b) * sizeof(std::int32_t);
    enc.set_buffer(1, gen_ids, out_off);
    if (sp.greedy) {
      enc.set_function(_fn_argmax);
      enc.set_buffer(0, _bdec.logits, (std::size_t)b * V * 2);   // 2B / elt
      enc.set_constant(2, V);
      enc.dispatch({256, 1, 1}, {256, 1, 1});
    } else {
      // Per-branch seed so branches don't sample in lock-step.
      const std::uint32_t seed_b =
          step_seed ^ (std::uint32_t)(0x9e3779b9u * (std::uint32_t)(b + 1));
      enc.set_function(_fn_sample);
      enc.set_buffer(0, _bdec.logits, (std::size_t)b * V * 2);
      enc.set_constant(2, V);
      enc.set_constant(3, sp.temperature);
      enc.set_constant(4, sp.top_p);
      enc.set_constant(5, seed_b);
      enc.set_buffer(6, _bdec_sess.sample_ws, (std::size_t)b * V * 2);
      enc.set_constant(7, sp.n_iter);
      enc.set_constant(8, sp.repetition_penalty);
      enc.set_constant(9, sp.presence_penalty);
      enc.set_constant(10, sp.top_k);
      enc.set_constant(11, sp.min_p);
      enc.set_buffer(12, _bdec_sess.seen, (std::size_t)b * V);
      enc.dispatch({256, 1, 1}, {256, 1, 1});
    }
  }
}

bool
MetalQwenModel::bdecode_begin(std::span<const ContextId> cids,
                              std::span<const std::int32_t> first_tokens,
                              const GpuSamplerParams& sp, int max_tokens,
                              std::span<const std::int32_t> rope_pos)
{
  const int N = (int)cids.size();
  if (N <= 0 || (int)first_tokens.size() != N) { return false; }
  if (!rope_pos.empty() && (int)rope_pos.size() != N) { return false; }
  if (!_fn_embed.valid() || !_fn_argmax.valid()
      || (!sp.greedy && !_fn_sample.valid())) {
    return false;
  }
  // Mixed-precision affine batched decode not yet supported (fused buffers
  // unbuilt) -> caller falls back to per-branch serial pdecode.
  if (_mixed) { return false; }
  if (!ensure_bscratch_(_bdec, N)) { return false; }
  if (max_tokens < 1) { max_tokens = 1; }

  BDecode& bd = _bdec_sess;
  bd = BDecode{};
  bd.cids.assign(cids.begin(), cids.end());
  bd.sp = sp;
  bd.n = N;
  bd.cap = max_tokens + 1;
  bd.produced = 1;                 // row 0 = first_tokens
  bd.rope_base.resize((std::size_t)N);
  bd.kv_base.resize((std::size_t)N);
  for (int i = 0; i < N; ++i) {
    bd.rope_base[(std::size_t)i] =
        (rope_pos.empty() || rope_pos[(std::size_t)i] < 0)
            ? -1 : rope_pos[(std::size_t)i];
    bd.kv_base[(std::size_t)i] = _ctx->seq_len_of(cids[(std::size_t)i]);
  }
  bd.gen_ids = _mc->make_shared_buffer(
      (std::size_t)bd.cap * N * sizeof(std::int32_t));
  if (bd.gen_ids.empty()) { bd = BDecode{}; return false; }
  auto* g = static_cast<std::int32_t*>(bd.gen_ids.contents());
  for (int i = 0; i < N; ++i) { g[i] = first_tokens[(std::size_t)i]; }

  if (!sp.greedy) {
    bd.sample_ws = _mc->make_shared_buffer((std::size_t)N * _cfg.vocab * 2);
    bd.seen = _mc->make_shared_buffer((std::size_t)N * _cfg.vocab);   // uint8
    if (bd.sample_ws.empty() || bd.seen.empty()) { bd = BDecode{}; return false; }
    std::memset(bd.seen.contents(), 0, (std::size_t)N * _cfg.vocab);
    auto* sb = static_cast<std::uint8_t*>(bd.seen.contents());
    for (int i = 0; i < N; ++i) {
      const std::int32_t t = first_tokens[(std::size_t)i];
      if (t >= 0 && t < _cfg.vocab) { sb[(std::size_t)i * _cfg.vocab + t] = 1; }
    }
  }
  bd.stream = _mc->make_command_stream();
  bd.ev = _mc->make_event();
  bd.active = true;
  return true;
}

bool
MetalQwenModel::bdecode_commit()
{
  BDecode& bd = _bdec_sess;
  if (!bd.active || bd.have_inflight) { return false; }
  const int N = bd.n;
  const int in_idx = bd.produced - 1;
  const int out_idx = bd.produced;
  if (out_idx >= bd.cap) { return false; }
  BScratch& bs = _bdec;
  const int H = _cfg.hidden;
  const std::size_t pt_stride = (std::size_t)_ctx->max_pages() * 3;

  std::vector<ContextManager::AppendSlot> slots((std::size_t)N);
  std::vector<std::size_t> page_offs((std::size_t)N);
  std::vector<int> n_pages_v((std::size_t)N), rope_pos_v((std::size_t)N);
  auto* pgt = static_cast<std::int32_t*>(bs.pgt.contents());
  for (int i = 0; i < N; ++i) {
    slots[(std::size_t)i] = _ctx->append(bd.cids[(std::size_t)i], 1);
    if (!slots[(std::size_t)i].valid()) { return false; }
    page_offs[(std::size_t)i] =
        (std::size_t)slots[(std::size_t)i].page_id.v * _ctx->page_stride_bytes();
    n_pages_v[(std::size_t)i] = _ctx->fill_page_table(
        bd.cids[(std::size_t)i], pgt + (std::size_t)i * pt_stride);
    const int pos = slots[(std::size_t)i].position;
    rope_pos_v[(std::size_t)i] =
        (bd.rope_base[(std::size_t)i] < 0)
            ? pos
            : bd.rope_base[(std::size_t)i] + (pos - bd.kv_base[(std::size_t)i]);
  }

  const std::uint64_t s = bd.gpu_step;
  bd.stream.encode_wait(bd.ev, s);
  {
    ComputeEncoder enc = bd.stream.begin_compute();
    // In-stream batched embed gather: gen_ids row in_idx [N] -> bs.x [N,H].
    enc.set_function(_fn_embed);
    enc.set_buffer(0, bd.gen_ids,
                   (std::size_t)in_idx * N * sizeof(std::int32_t));
    enc.set_buffer(1, _embed_w); enc.set_buffer(2, _embed_s);
    enc.set_buffer(3, _embed_b); enc.set_buffer(4, bs.x);
    enc.set_constant(5, H);
    enc.dispatch({(unsigned)H, (unsigned)N, 1}, {256, 1, 1});

    encode_batched_step_(enc, bs,
                         std::span<const ContextId>(bd.cids.data(),
                                                    bd.cids.size()),
                         slots, page_offs, n_pages_v, rope_pos_v,
                         shared_prefix_pages_(pgt, n_pages_v, N, pt_stride));

    const std::uint32_t step_seed = (std::uint32_t)(
        bd.sp.seed + 0x9e3779b9ull * (std::uint64_t)(s + 1));
    encode_sample_batched_(enc, bd.gen_ids, out_idx, N, bd.sp, step_seed);
  }
  bd.stream.encode_signal(bd.ev, s + 1);
  bd.inflight = bd.stream.commit();
  bd.gpu_step = s + 1;
  bd.pending = out_idx;
  bd.have_inflight = true;
  return true;
}

bool
MetalQwenModel::bdecode_next(std::vector<std::int32_t>& out_tokens)
{
  BDecode& bd = _bdec_sess;
  if (!bd.have_inflight) { return false; }
  bd.inflight.wait();
  const int N = bd.n;
  out_tokens.resize((std::size_t)N);
  const auto* g = static_cast<const std::int32_t*>(bd.gen_ids.contents());
  for (int i = 0; i < N; ++i) {
    out_tokens[(std::size_t)i] = g[(std::size_t)bd.pending * N + i];
  }
  bd.produced = bd.pending + 1;
  bd.have_inflight = false;
  bd.pending = -1;
  return true;
}

void
MetalQwenModel::bdecode_end()
{
  BDecode& bd = _bdec_sess;
  if (bd.have_inflight) { bd.inflight.wait(); }
  bd = BDecode{};
}

std::vector<float>
MetalQwenModel::forward(ContextId cid, std::int32_t token_id, int rope_pos)
{
  const Config& c = _cfg;
  const int H = c.hidden;
  if (!ensure_decode_scratch_()) {
    return {};
  }

  ContextManager::AppendSlot slot = _ctx->append(cid, 1);
  if (!slot.valid()) {
    return {};
  }
  const int pos = slot.position;
  // RoPE position for the full-attn layers: the KV slot position for text,
  // or an explicit override for post-image decode (where mROPE positions
  // diverge from KV slots).
  const int rpos = (rope_pos < 0) ? pos : rope_pos;
  const std::size_t page_off =
      (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));

  // Embed via the muxer (its own command buffer), then copy into the
  // shared residual stream _d_x that the encode helper consumes.
  std::int32_t id1 = token_id;
  SharedBuffer emb = _muxer->fetch_text(std::span<const std::int32_t>(&id1, 1));
  if (emb.empty()) {
    return {};
  }
  std::memcpy(_d_x.contents(), emb.contents(), (std::size_t)H * 2);

  // Decode profiler (VPIPE_QWEN_DECODE_PROFILE): split CPU encode time
  // from commit+GPU time (decode issues ~500 serial dispatches/token).
  static const bool kProf =
      std::getenv("VPIPE_QWEN_DECODE_PROFILE") != nullptr;
  const auto t_enc0 = std::chrono::steady_clock::now();
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    encode_decode_step_(enc, cid, pos, rpos, page_off, n_pages, slot,
                        _pgtab, 0);
  }
  const auto t_enc1 = std::chrono::steady_clock::now();
  stream.commit().wait();
  if (kProf) {
    const auto t_done = std::chrono::steady_clock::now();
    static double enc_ms = 0.0, gpu_ms = 0.0;
    static int cnt = 0;
    enc_ms += std::chrono::duration<double, std::milli>(t_enc1 - t_enc0).count();
    gpu_ms += std::chrono::duration<double, std::milli>(t_done - t_enc1).count();
    if (++cnt % 16 == 0) {
      std::printf("[qwen-decode-prof] encode %.3f ms | commit+gpu %.3f ms "
                  "(%d steps avg)\n", enc_ms / cnt, gpu_ms / cnt, cnt);
    }
  }

  std::vector<float> out((std::size_t)c.vocab);
  read_elt_(_d_logits.contents(), out.data(), (std::size_t)c.vocab,
            _cfg.use_bf16);
  return out;
}

std::vector<float>
MetalQwenModel::prefill(ContextId cid, const std::vector<std::int32_t>& ids)
{
  const int n = (int)ids.size();
  if (n <= 0) { return {}; }
  if (n == 1) { return forward(cid, ids[0]); }   // decode path handles n=1
  SharedBuffer emb =
      _muxer->fetch_text(std::span<const std::int32_t>(ids.data(), ids.size()));
  if (emb.empty()) { return {}; }
  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * _cfg.hidden * 2);
  std::memcpy(x.contents(), emb.contents(), (std::size_t)n * _cfg.hidden * 2);
  return forward_chunk_(cid, x, n, nullptr, nullptr);
}

std::vector<float>
MetalQwenModel::prefill_multimodal(ContextId cid,
                                   const std::vector<float>& embeddings,
                                   const std::vector<std::int32_t>& position_ids)
{
  const int H = _cfg.hidden;
  const int n = (int)(embeddings.size() / (std::size_t)H);
  if (n <= 0) { return {}; }
  const bool bf16 = _cfg.use_bf16;
  auto store_elt = [&](float f) -> std::uint16_t {
    if (bf16) { return f32_to_bf16_(f); }
    _Float16 h = (_Float16)f; std::uint16_t b; std::memcpy(&b, &h, 2); return b;
  };
  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * H * 2);
  auto* xp = static_cast<std::uint16_t*>(x.contents());
  for (std::size_t i = 0; i < (std::size_t)n * H; ++i) {
    xp[i] = store_elt(embeddings[i]);
  }
  return prefill_multimodal_buf(cid, std::move(x), position_ids, n);
}

std::vector<float>
MetalQwenModel::prefill_multimodal_buf(
    ContextId cid, SharedBuffer&& x,
    const std::vector<std::int32_t>& position_ids, int n)
{
  const int H = _cfg.hidden;
  if (n <= 0 || x.byte_size() < (std::size_t)n * H * 2) { return {}; }
  const bool bf16 = _cfg.use_bf16;
  auto store_elt = [&](float f) -> std::uint16_t {
    if (bf16) { return f32_to_bf16_(f); }
    _Float16 h = (_Float16)f; std::uint16_t b; std::memcpy(&b, &h, 2); return b;
  };
  // mROPE cos/sin tables [n, rotary_dim] from the 3-axis position_ids +
  // per-pair axis lookup + inv_freq (cat([f,f]) layout: [d]==[d+half]).
  const int rd = _cfg.rotary_dim, half = rd / 2;
  const auto* invf = static_cast<const float*>(_inv_freq.contents());
  SharedBuffer cosb = _mc->make_shared_buffer((std::size_t)n * rd * 2);
  SharedBuffer sinb = _mc->make_shared_buffer((std::size_t)n * rd * 2);
  auto* cp = static_cast<std::uint16_t*>(cosb.contents());
  auto* sp = static_cast<std::uint16_t*>(sinb.contents());
  for (int t = 0; t < n; ++t) {
    for (int d = 0; d < half; ++d) {
      const int axis = (d < (int)_mrope_axis.size()) ? _mrope_axis[d] : 0;
      const float pos = (float)position_ids[(std::size_t)axis * n + t];
      const float ang = pos * invf[d];
      const std::uint16_t cc = store_elt(std::cos(ang));
      const std::uint16_t ss = store_elt(std::sin(ang));
      cp[(std::size_t)t * rd + d] = cc;
      cp[(std::size_t)t * rd + half + d] = cc;
      sp[(std::size_t)t * rd + d] = ss;
      sp[(std::size_t)t * rd + half + d] = ss;
    }
  }
  return forward_chunk_(cid, x, n, &cosb, &sinb);
}

std::vector<float>
MetalQwenModel::prefill_embeddings(ContextId cid,
                                   const std::vector<float>& embeddings, int n)
{
  const int H = _cfg.hidden;
  if (n <= 0 || embeddings.size() < (std::size_t)n * H) { return {}; }
  const bool bf16 = _cfg.use_bf16;
  auto store_elt = [&](float f) -> std::uint16_t {
    if (bf16) { return f32_to_bf16_(f); }
    _Float16 h = (_Float16)f; std::uint16_t b; std::memcpy(&b, &h, 2); return b;
  };
  SharedBuffer x = _mc->make_shared_buffer((std::size_t)n * H * 2);
  auto* xp = static_cast<std::uint16_t*>(x.contents());
  for (std::size_t i = 0; i < (std::size_t)n * H; ++i) {
    xp[i] = store_elt(embeddings[i]);
  }
  return prefill_embeddings_buf(cid, std::move(x), n);
}

std::vector<float>
MetalQwenModel::prefill_embeddings_buf(ContextId cid, SharedBuffer&& x, int n)
{
  const int H = _cfg.hidden;
  if (n <= 0 || x.byte_size() < (std::size_t)n * H * 2) { return {}; }
  // Plain 1-D RoPE over sequential positions (nullptr mrope tables).
  return forward_chunk_(cid, x, n, nullptr, nullptr);
}

metal_compute::SharedBuffer
MetalQwenModel::forward_embeddings_hidden(ContextId cid, const SharedBuffer& x,
                                          int n)
{
  const int H = _cfg.hidden;
  if (n <= 0 || x.byte_size() < (std::size_t)n * H * 2) { return {}; }
  SharedBuffer hidden;
  // return_hidden mode: appends to cid's paged KV, runs the layer stack +
  // final norm, and moves the last position's normed hidden into `hidden`.
  forward_chunk_(cid, x, n, nullptr, nullptr,
                 /*verify_all=*/false, /*preds_out=*/nullptr,
                 /*return_hidden=*/true, &hidden);
  return hidden;
}

const metal_compute::SharedBuffer*
MetalQwenModel::decode_embedding_hidden(
    ContextId cid, const SharedBuffer& emb, int rope_pos,
    const std::function<void(ComputeEncoder&, const SharedBuffer&)>* post_hidden)
{
  const int H = _cfg.hidden;
  if (emb.byte_size() < (std::size_t)H * 2) { return nullptr; }
  if (!ensure_decode_scratch_()) { return nullptr; }
  ContextManager::AppendSlot slot = _ctx->append(cid, 1);
  if (!slot.valid()) { return nullptr; }
  const int pos = slot.position;
  const int rpos = (rope_pos < 0) ? pos : rope_pos;
  const std::size_t page_off =
      (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));
  // Splice the caller's [H] embedding into the decode residual stream.
  std::memcpy(_d_x.contents(), emb.contents(), (std::size_t)H * 2);

  static const bool kProf =
      std::getenv("VPIPE_QWEN_DECODE_PROFILE") != nullptr;
  const auto t_enc0 = std::chrono::steady_clock::now();
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    encode_decode_step_(enc, cid, pos, rpos, page_off, n_pages, slot, _pgtab,
                        0, /*return_hidden=*/true);
    // Fuse the caller's dispatches (MOSS heads) into the same command buffer,
    // reading _d_hn which the final norm above wrote (GPU-ordered).
    if (post_hidden != nullptr) { (*post_hidden)(enc, _d_hn); }
  }
  const auto t_enc1 = std::chrono::steady_clock::now();
  stream.commit().wait();
  if (kProf) {
    static double enc_ms = 0.0, gpu_ms = 0.0;
    static int cnt = 0;
    enc_ms += std::chrono::duration<double, std::milli>(t_enc1 - t_enc0).count();
    gpu_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_enc1).count();
    if (++cnt % 16 == 0) {
      std::printf("[moss-decode-prof] encode %.2f ms | commit+gpu %.2f ms "
                  "(%d steps avg)\n", enc_ms / cnt, gpu_ms / cnt, cnt);
    }
  }
  return &_d_hn;
}

metal_compute::SharedBuffer
MetalQwenModel::embed_text_buf(const std::vector<std::int32_t>& ids)
{
  if (ids.empty() || _muxer == nullptr) { return {}; }
  return _muxer->fetch_text(
      std::span<const std::int32_t>(ids.data(), ids.size()));
}

std::vector<float>
MetalQwenModel::forward_chunk_(ContextId cid, const SharedBuffer& x, int n,
                               const SharedBuffer* mrope_cos,
                               const SharedBuffer* mrope_sin, bool verify_all,
                               std::vector<std::int32_t>* preds_out,
                               bool return_hidden, SharedBuffer* hidden_out)
{
  const Config& c = _cfg;
  const int H = c.hidden, D = c.head_dim, Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, K = c.gdn_conv_kernel, ffn = c.ffn_inner;
  const float eps = c.rms_eps;
  const float scale = 1.0f / std::sqrt((float)D);

  // Reserve KV slots (chunked into pages), build the page table.
  struct Chunk { std::size_t page_off; int slot; int src_off; int cnt; };
  std::vector<Chunk> chunks;
  int q_offset = -1;
  for (int written = 0; written < n; ) {
    const int cap = _ctx->next_append_capacity(cid);
    const int cnt = std::min(n - written, cap);
    ContextManager::AppendSlot s = _ctx->append(cid, cnt);
    if (!s.valid()) { return {}; }
    if (q_offset < 0) { q_offset = s.position; }
    chunks.push_back({(std::size_t)s.page_id.v * _ctx->page_stride_bytes(),
                      s.slot_offset, written, cnt});
    written += cnt;
  }
  if (q_offset < 0) { q_offset = 0; }
  const int page_tokens = _ctx->page_tokens();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));

  static const bool kPProf =
      std::getenv("VPIPE_QWEN_PREFILL_PROFILE") != nullptr;
  const auto t_p0 = std::chrono::steady_clock::now();
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer hn = buf((std::size_t)n * H);
  // qfull holds the FUSED q|k|v projection [n, 2*qd+2*kd] (q gated = 2*qd).
  SharedBuffer qfull = buf((std::size_t)n * (2 * qd + 2 * kd));
  SharedBuffer kbuf = buf((std::size_t)n * kd), vbuf = buf((std::size_t)n * kd);
  SharedBuffer q3 = buf((std::size_t)n * qd), gate3 = buf((std::size_t)n * qd);
  SharedBuffer qt = buf((std::size_t)n * qd), kt = buf((std::size_t)n * kd),
               vt = buf((std::size_t)n * kd);
  SharedBuffer at = buf((std::size_t)n * qd), att = buf((std::size_t)n * qd),
               ao = buf((std::size_t)n * H);
  // mixqkv holds the FUSED in_proj output [n, Cd+vald+2*Hv] (qkv|z|a|b).
  SharedBuffer mixqkv = buf((std::size_t)n * (Cd + vald + 2 * Hv));
  SharedBuffer zbuf = buf((std::size_t)n * vald);
  SharedBuffer abuf = buf((std::size_t)n * Hv), bbuf = buf((std::size_t)n * Hv);
  SharedBuffer convout = buf((std::size_t)n * Cd);
  SharedBuffer gq = buf((std::size_t)n * keyd), gk = buf((std::size_t)n * keyd),
               gv = buf((std::size_t)n * vald);
  SharedBuffer gbuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  SharedBuffer betabuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  SharedBuffer ygdn = buf((std::size_t)n * vald), normout = buf((std::size_t)n * vald);
  SharedBuffer sg = buf((std::size_t)n * ffn);
  SharedBuffer logits = buf((std::size_t)c.vocab);
  // MTP batched-verify: per-position logits [n, vocab] + per-position argmax.
  SharedBuffer vlogits = verify_all ? buf((std::size_t)n * c.vocab)
                                    : SharedBuffer{};
  SharedBuffer vamax = verify_all
      ? _mc->make_shared_buffer((std::size_t)n * sizeof(std::int32_t))
      : SharedBuffer{};
  // Native k-quant prefill: a second [n, ffn] MLP temp (gate -> sg, up ->
  // this, then SwiGLU); the dense GEMM weight scratch _w_deq is shared.
  SharedBuffer kqubuf =
      (_kquant || _mixed) ? buf((std::size_t)n * ffn) : SharedBuffer{};
  auto ensure_wdeq = [&](std::size_t elems) {
    if (_w_deq.empty() || _w_deq.byte_size() < elems * 2) {
      _w_deq = _mc->make_shared_buffer(elems * 2);
    }
  };
  // Interleaved gate|up output [n, 2*ffn] for the matrix-core MLP path
  // (dense matmul2d then swiglu_interleaved -> sg). Only when used.
  const bool mma_mlp = _use_mma && n >= _mma_min_m;
  SharedBuffer gu_full =
      mma_mlp ? buf((std::size_t)n * 2 * ffn) : SharedBuffer{};

  const auto t_alloc = std::chrono::steady_clock::now();
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                   const SharedBuffer& w, const SharedBuffer& y,
                   std::size_t yoff, int R, int Hd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin, xoff);
      enc.set_buffer(1, w);
      enc.set_buffer(2, y, yoff);
      enc.set_constant(3, Hd);
      enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    // Tile-adaptive matrix-core dense GEMM: y[n,N] = xin[n,Kk]@wdeq[N,Kk]^T
    // (no bias). 128x128 tile (8 simdgroups) for K <= 4096; 128x256 for
    // deeper K (down_proj) where the square tile is bandwidth-starved on
    // weight streaming. Both clamp M/N tails via the matmul2d tensor
    // extents, so n and N need not be tile multiples.
    auto dense_mma = [&](const SharedBuffer& xin, const SharedBuffer& wdeq,
                         const SharedBuffer& y, int Kk, int Nn) {
      const bool deep = (Kk >= 6144);
      const int BN = deep ? 256 : 128;
      enc.set_function(deep ? _fn_dense_mma_deep : _fn_dense_mma);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, wdeq);
      enc.set_buffer(2, wdeq);      // bias slot unused (has_bias=0)
      enc.set_buffer(3, y);
      enc.set_constant(4, Kk);
      enc.set_constant(5, Nn);
      enc.set_constant(6, n);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((Nn + BN - 1) / BN) * 256),
                    (unsigned)((n + 127) / 128), 1}, {256, 1, 1});
    };
    auto qmm = [&](const SharedBuffer& w, const SharedBuffer& s,
                   const SharedBuffer& b, const SharedBuffer& xin,
                   const SharedBuffer& y, int Kk, int N) {
      // Matrix-core path (M5+): dequant the 4-bit weight into a reusable
      // dense scratch, then run the dense matmul2d GEMM on the hardware
      // matrix units. ~2-2.5x the steel quantized GEMM at prefill row
      // counts; M-independent dequant cost amortizes over the n rows.
      // Restricted to enough rows that the dequant pass pays off.
      if (_use_mma && n >= _mma_min_m) {
        const std::size_t need = (std::size_t)N * Kk * 2;  // compute-elt bytes
        if (_w_deq.empty() || _w_deq.byte_size() < need) {
          _w_deq = _mc->make_shared_buffer(need);
        }
        // dequant: (w, s, b) -> _w_deq[N,Kk]  (one thread per weight byte)
        if (!_skip_dequant) {
          enc.set_function(_fn_dequant);
          enc.set_buffer(0, w);
          enc.set_buffer(1, s);
          enc.set_buffer(2, b);
          enc.set_buffer(3, _w_deq);
          enc.set_constant(4, Kk);
          enc.set_constant(5, N);
          enc.dispatch({(unsigned)(Kk / 8), (unsigned)N, 1}, {64, 1, 1});
        }
        // dense matmul2d: y[n,N] = xin[n,Kk] @ _w_deq[N,Kk]^T (no bias).
        dense_mma(xin, _w_deq, y, Kk, N);
        return;
      }
      enc.set_function(_fn_qmm);
      enc.set_buffer(0, w);
      enc.set_buffer(1, s);
      enc.set_buffer(2, b);
      enc.set_buffer(3, xin);
      enc.set_buffer(4, y);
      enc.set_constant(5, Kk);
      enc.set_constant(6, N);
      enc.set_constant(7, n);
      const unsigned gx = (unsigned)(((N + 31) / 32) * 32);
      const unsigned gy = (unsigned)(((n + 31) / 32) * 2);
      enc.dispatch({gx, gy, 2}, {32, 2, 2});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in);
      enc.set_buffer(1, out);
      enc.set_constant(2, A);
      enc.set_constant(3, Bd);
      enc.set_constant(4, D);
      enc.dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A}, {(unsigned)D, 1, 1});
    };
    auto rope = [&](const SharedBuffer& xb, int heads) {
      enc.set_function(_fn_rope_partial);
      enc.set_buffer(0, xb);
      enc.set_buffer(1, _inv_freq);
      enc.set_constant(2, heads);
      enc.set_constant(3, n);
      enc.set_constant(4, D);
      enc.set_constant(5, c.rotary_dim);
      enc.set_constant(6, q_offset);
      enc.dispatch({(unsigned)(c.rotary_dim / 2), (unsigned)n, (unsigned)heads},
                   {(unsigned)(c.rotary_dim / 2), 1, 1});
    };
    // Multimodal: table-driven mROPE (cos/sin built from 3-axis positions).
    auto mrope = [&](const SharedBuffer& xb, int heads) {
      enc.set_function(_fn_mrope);
      enc.set_buffer(0, xb);
      enc.set_buffer(1, *mrope_cos);
      enc.set_buffer(2, *mrope_sin);
      enc.set_constant(3, heads);
      enc.set_constant(4, n);
      enc.set_constant(5, D);
      enc.set_constant(6, c.rotary_dim);
      enc.dispatch({(unsigned)(c.rotary_dim / 2), (unsigned)n, (unsigned)heads},
                   {(unsigned)(c.rotary_dim / 2), 1, 1});
    };
    auto hslice = [&](const SharedBuffer& in, std::size_t ioff,
                      const SharedBuffer& out, std::size_t ooff, int Hh,
                      int S, int W, int off, int block = 0, int gstride = 0) {
      enc.set_function(_fn_head_slice);
      enc.set_buffer(0, in, ioff);
      enc.set_buffer(1, out, ooff);
      enc.set_constant(2, Hh);
      enc.set_constant(3, S);
      enc.set_constant(4, W);
      enc.set_constant(5, off);
      enc.set_constant(6, block);
      enc.set_constant(7, gstride);
      enc.dispatch({(unsigned)(Hh * W), 1, 1}, {256, 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& bb,
                        const SharedBuffer& out, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a);
      enc.set_buffer(1, bb);
      enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto kv_write = [&](const SharedBuffer& src, const SharedBuffer& pool) {
      for (const Chunk& ch : chunks) {
        enc.set_function(_fn_kv_write_paged);
        enc.set_buffer(0, src);
        enc.set_buffer(1, pool, ch.page_off);
        enc.set_constant(2, page_tokens);
        enc.set_constant(3, D);
        enc.set_constant(4, n);             // n_src (row stride of kt/vt)
        enc.set_constant(5, ch.src_off);
        enc.set_constant(6, ch.slot);
        enc.dispatch({(unsigned)D, (unsigned)ch.cnt, (unsigned)Hkv},
                     {(unsigned)D, 1, 1});
      }
    };

    // A/B probe (VPIPE_QWEN_SKIP_ATTN): skip a single GPU dispatch to measure
    // its share of prefill (logits become garbage -- timing only; every other
    // op still runs, so the with/without delta is pure kernel time). Modes:
    //   1 (or "all") -- skip the full-attention SDPA (the O(n^2) softmax attn
    //                   on the full layers; the key-split flash-port target)
    //   2 (or "gdn") -- skip the gated-delta step (the linear/GDN-layer
    //                   recurrence) to attribute GDN's prefill cost separately
    static const int kSkipMode = []() {
      const char* e = std::getenv("VPIPE_QWEN_SKIP_ATTN");
      if (e == nullptr || *e == '\0') { return 0; }
      if (std::strcmp(e, "gdn") == 0 || std::atoi(e) == 2) { return 2; }
      return 1;
    }();

    for (int L = 0; L < c.n_layers; ++L) {
      Layer& ly = _layers[L];
      rms(x, 0, ly.in_ln, hn, 0, n, H);
      if (ly.is_full) {
        const SharedBuffer& kp = *_ctx->kpool(L);
        const SharedBuffer& vp = *_ctx->vpool(L);
        // Single fused q|k|v GEMM. q_proj width is 2*qd (gated, Qwen3.5)
        // or qd (Qwen3-ASR); k/v tails follow at qdo, qdo+kd.
        const bool gate = c.attn_output_gate;
        const int qdo = gate ? 2 * qd : qd;
        const int Nfqkv = qdo + 2 * kd;
        if (_kquant) {
          // q+k (q4_K) + v (q6_K) dequant'd into one f16 scratch, one dense
          // -> qfull[n, Nfqkv] (the exact [q|k|v] fused layout downstream).
          ensure_wdeq((std::size_t)Nfqkv * H);
          kdequant_(enc, ly.kqk_t, ly.kqk, 0, ly.kqk_n * H);
          kdequant_(enc, ly.kqv_t, ly.kqv, (std::size_t)ly.kqk_n * H, kd * H);
          dense_gemm_(enc, _w_deq, hn, qfull, H, Nfqkv, n);
        } else if (_mixed) {
          // q|k|v each its own bits -> one f16 scratch [Nfqkv,H], one dense
          // -> qfull[n, Nfqkv] (the same [q|k|v] layout downstream slices).
          ensure_wdeq((std::size_t)Nfqkv * H);
          adequant_(enc, ly.qw, ly.qs, ly.qb, ly.q_bits, 0, qdo, H);
          adequant_(enc, ly.kw, ly.ks, ly.kb, ly.k_bits,
                    (std::size_t)qdo * H, kd, H);
          adequant_(enc, ly.vw, ly.vs, ly.vb, ly.v_bits,
                    (std::size_t)(qdo + kd) * H, kd, H);
          dense_gemm_(enc, _w_deq, hn, qfull, H, Nfqkv, n);
        } else {
          qmm(ly.qw, ly.qs, ly.qb, hn, qfull, H, Nfqkv);
        }
        if (gate) {
          // q is the gated [n_heads, 2*head_dim] front block; block-strided
          // read extracts q / gate straight from the fused buffer.
          hslice(qfull, 0, q3, 0, n * Hq, 2 * D, D, 0, Hq, Nfqkv);
          hslice(qfull, 0, gate3, 0, n * Hq, 2 * D, D, D, Hq, Nfqkv);
        } else {
          // q is contiguous [n, Hq*D]; plain row-strided slice (stride Nfqkv).
          hslice(qfull, 0, q3, 0, n, Nfqkv, qd, 0);
        }
        // k/v are contiguous tails -> plain row-strided slice into kbuf/vbuf.
        hslice(qfull, 0, kbuf, 0, n, Nfqkv, kd, qdo);
        hslice(qfull, 0, vbuf, 0, n, Nfqkv, kd, qdo + kd);
        rms(q3, 0, ly.q_norm, q3, 0, n * Hq, D);
        rms(kbuf, 0, ly.k_norm, kbuf, 0, n * Hkv, D);
        transpose(q3, qt, n, Hq);       // [n,Hq,D] -> [Hq,n,D]
        transpose(kbuf, kt, n, Hkv);
        transpose(vbuf, vt, n, Hkv);
        if (mrope_cos != nullptr) { mrope(qt, Hq); mrope(kt, Hkv); }
        else { rope(qt, Hq); rope(kt, Hkv); }
        kv_write(kt, kp);
        kv_write(vt, vp);
        // Query-tiled flash (head_dim 256): stages each K/V block once and
        // reuses it across a 16-query tile -> ~16x less K/V traffic than the
        // per-query scalar kernel, which is what dominates O(n^2) prefill
        // attention at long context. The tg-staging overhead isn't repaid for
        // short prompts (attention is tiny there, prefill is GEMM-bound), so
        // use it only past a crossover. Same buffer contract.
        // Matrix-core flash attention (M5+, head_dim 256): same buffer
        // contract as the scalar query-tiled kernel, QK^T/P*V on the
        // hardware matrix units -- ~1.4-2.3x the scalar qtile past the
        // crossover. Falls back to qtile (scalar, no matrix cores) / the
        // per-query scalar kernel for short prompts where attention is tiny.
        const bool use_mma_attn =
            _use_mma && _fn_sdpa_mma.valid() && (D == 256) &&
            (n >= _mma_attn_min_n);
        // simdgroup_matrix key-split flash (M4-capable): preferred over the
        // scalar qtile when the matrix-core mma path isn't taken. Same
        // crossover -- attention only matters past ~384 tokens; below it
        // prefill is GEMM-bound and the per-query scalar kernel avoids the
        // flash tg setup.
        const bool use_flash = !use_mma_attn && _flash_attn &&
            _fn_sdpa_paged_flash.valid() && (D == 256) && (n >= 384);
        const bool use_qt =
            !use_mma_attn && !use_flash && (D == 256) && (n >= 384);
        enc.set_function(
            use_mma_attn ? _fn_sdpa_mma
            : (use_flash ? _fn_sdpa_paged_flash
               : (use_qt ? _fn_sdpa_paged_qtile : _fn_sdpa_paged)));
        enc.set_buffer(0, qt);
        enc.set_buffer(1, kp);
        enc.set_buffer(2, vp);
        enc.set_buffer(3, at);
        enc.set_constant(4, scale);
        enc.set_constant(5, D);
        enc.set_constant(6, Hq);
        enc.set_constant(7, Hkv);
        enc.set_constant(8, n);
        enc.set_constant(9, q_offset);
        enc.set_constant(10, page_tokens);
        enc.set_constant(11, n_pages);
        enc.set_buffer(12, _pgtab);
        if (kSkipMode != 1) {     // VPIPE_QWEN_SKIP_ATTN full-attn SDPA probe
          if (use_flash) {
            // FL_NSG*32 = 256 threads, FL_Q = 8 query rows per tile.
            const unsigned ntg = (unsigned)((n + 7) / 8);
            enc.dispatch({256, (unsigned)Hq, ntg}, {256, 1, 1});
          } else if (use_mma_attn || use_qt) {
            const unsigned n_tiles = (unsigned)((n + 15) / 16);   // BQ=16
            const unsigned tgsz = use_mma_attn ? 128u : 512u;
            enc.dispatch({tgsz, (unsigned)Hq, n_tiles}, {tgsz, 1, 1});
          } else {
            enc.dispatch({32, (unsigned)Hq, (unsigned)n}, {32, 1, 1});
          }
        }
        transpose(at, att, Hq, n);      // [Hq,n,D] -> [n,Hq,D]
        if (gate) {
          enc.set_function(_fn_mul_sigmoid);
          enc.set_buffer(0, att);
          enc.set_buffer(1, gate3);
          enc.set_buffer(2, att);
          enc.set_constant(3, n * qd);
          enc.dispatch({(unsigned)(n * qd), 1, 1}, {256, 1, 1});
        }
        if (_kquant) { kqmm_(enc, ly.kqo_t, ly.kqo, att, ao, qd, H, n); }
        else if (_mixed) {
          aqmm_(enc, ly.ow, ly.os, ly.ob, ly.o_bits, att, ao, qd, H, n);
        } else { qmm(ly.ow, ly.os, ly.ob, att, ao, qd, H); }
        residual(x, ao, x, n * H);
      } else {
        const SharedBuffer* csb = _ctx->conv_state(cid, L);
        const SharedBuffer* ssb = _ctx->ssm_state(cid, L);
        // Single fused in_proj GEMM -> mixqkv[n, Nf] laid out qkv|z|a|b.
        const int Nf = Cd + vald + 2 * Hv;
        if (_kquant) {
          // qkv (q5_K) + z (q4_K) dequant'd, a|b (f16) copied, into one f16
          // scratch -> one dense GEMM -> mixqkv[n, Nf] (the fused layout).
          ensure_wdeq((std::size_t)Nf * H);
          kdequant_(enc, ly.kqkv_t, ly.kqkv, 0, Cd * H);
          kdequant_(enc, ly.kqz_t, ly.kqz, (std::size_t)Cd * H, vald * H);
          kcopy_(enc, ly.kqab, (std::size_t)(Cd + vald) * H, 2 * Hv * H);
          dense_gemm_(enc, _w_deq, hn, mixqkv, H, Nf, n);
        } else if (_mixed) {
          // qkv|z|a|b each its own bits -> one f16 scratch [Nf,H], one dense
          // GEMM -> mixqkv[n, Nf] (the fused qkv|z|a|b layout).
          ensure_wdeq((std::size_t)Nf * H);
          adequant_(enc, ly.iqw, ly.iqs, ly.iqb, ly.qkv_bits, 0, Cd, H);
          adequant_(enc, ly.izw, ly.izs, ly.izb, ly.z_bits,
                    (std::size_t)Cd * H, vald, H);
          adequant_(enc, ly.iaw, ly.ias, ly.iab, ly.a_bits,
                    (std::size_t)(Cd + vald) * H, Hv, H);
          adequant_(enc, ly.ibw, ly.ibs, ly.ibb, ly.b_bits,
                    (std::size_t)(Cd + vald + Hv) * H, Hv, H);
          dense_gemm_(enc, _w_deq, hn, mixqkv, H, Nf, n);
        } else {
          qmm(ly.iqw, ly.iqs, ly.iqb, hn, mixqkv, H, Nf);
        }
        hslice(mixqkv, 0, zbuf, 0, n, Nf, vald, Cd);
        hslice(mixqkv, 0, abuf, 0, n, Nf, Hv, Cd + vald);
        hslice(mixqkv, 0, bbuf, 0, n, Nf, Hv, Cd + vald + Hv);
        // Conv1d reads the qkv segment (offset 0, row stride Nf) of the fused
        // buffer and writes its output DE-INTERLEAVED into three contiguous
        // blocks q|k|v inside convout (keyd arg), so q/k-norm and the
        // delta-rule step read each block directly -- no hslice copy.
        enc.set_function(_fn_gdn_conv1d);
        enc.set_buffer(0, *csb);
        enc.set_buffer(1, mixqkv);
        enc.set_buffer(2, ly.conv_w);
        enc.set_buffer(3, convout);
        enc.set_constant(4, n);
        enc.set_constant(5, Cd);
        enc.set_constant(6, K);
        enc.set_constant(7, Nf);       // x_stride: qkv is first Cd cols of fused
        enc.set_constant(8, keyd);     // de-interleave q|k|v
        enc.set_buffer(9, *csb);       // in-place (prefill bumps state directly)
        enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
        const std::size_t kb_off = (std::size_t)n * keyd * 2;   // bytes
        const std::size_t vb_off = 2 * kb_off;
        rms(convout, 0, _gdn_qscale, convout, 0, n * Hk, Dk);
        rms(convout, kb_off, _gdn_kscale, convout, kb_off, n * Hk, Dk);
        enc.set_function(_fn_gdn_g_beta);
        enc.set_buffer(0, abuf);
        enc.set_buffer(1, bbuf);
        enc.set_buffer(2, ly.A_log);
        enc.set_buffer(3, ly.dt_bias);
        enc.set_buffer(4, gbuf);
        enc.set_buffer(5, betabuf);
        enc.set_constant(6, Hv);
        enc.set_constant(7, n);
        enc.dispatch({(unsigned)(n * Hv), 1, 1}, {256, 1, 1});
        // Prefill: prefer the ndv4 step (1.33x; 4 dv/simdgroup, k/q read
        // once/step). Token-exact with v1; falls back to v1 if Dv isn't a
        // multiple of 4 or the variant is unavailable.
        const bool gdn4 = _fn_gdn_step_ndv4.valid() && (Dv % 4 == 0) &&
                          !_gdn_force_v1;
        enc.set_function(gdn4 ? _fn_gdn_step_ndv4 : _fn_gdn_step);
        enc.set_buffer(0, convout, 0);        // q block
        enc.set_buffer(1, convout, kb_off);   // k block
        enc.set_buffer(2, convout, vb_off);   // v block
        enc.set_buffer(3, gbuf);
        enc.set_buffer(4, betabuf);
        enc.set_buffer(5, *ssb);
        enc.set_buffer(6, ygdn);
        enc.set_buffer(7, *ssb);
        enc.set_constant(8, n);                       // T = n
        enc.set_constant(9, _kquant ? -Hk : Hk);      // strided GQA for GGUF
        enc.set_constant(10, Hv);
        const unsigned gdn_dvy = gdn4 ? (unsigned)(Dv / 4) : (unsigned)Dv;
        if (kSkipMode != 2) {     // VPIPE_QWEN_SKIP_ATTN=gdn step probe
          enc.dispatch({32, gdn_dvy, (unsigned)Hv}, {32, 4, 1});
        }
        rms(ygdn, 0, ly.gdn_norm, normout, 0, n * Hv, Dv);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, zbuf);
        enc.set_buffer(1, normout);
        enc.set_buffer(2, normout);
        enc.set_constant(3, n * vald);
        enc.dispatch({(unsigned)(n * vald), 1, 1}, {256, 1, 1});
        if (_kquant) {
          kqmm_(enc, ly.kqout_t, ly.kqout, normout, ao, vald, H, n);
        } else if (_mixed) {
          aqmm_(enc, ly.gow, ly.gos, ly.gob, ly.gout_bits, normout, ao, vald,
                H, n);
        } else {
          qmm(ly.gow, ly.gos, ly.gob, normout, ao, vald, H);
        }
        residual(x, ao, x, n * H);
      }
      // MLP (every layer): fused interleaved gate/up + SwiGLU, then down.
      // The MLP and final norm are position-wise and prefill only consumes
      // the LAST token's logits, so on the final layer we run the MLP for
      // that one row (GEMV path) -- rows 0..n-2 never get the pruned
      // last-layer FFN and no one reads them. Bit-identical for row n-1.
      if (_kquant) {
        // Native k-quant MLP over all n rows (no last-layer prune; lm_head
        // still consumes only the last position). gate/up are two q4_K
        // dequant+dense into sg / kqubuf, SwiGLU, then down (q6_K).
        rms(x, 0, ly.post_ln, hn, 0, n, H);
        kqmm_(enc, ly.kqgate_t, ly.kqgate, hn, sg, H, ffn, n);
        kqmm_(enc, ly.kqup_t, ly.kqup, hn, kqubuf, H, ffn, n);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, sg);
        enc.set_buffer(1, kqubuf);
        enc.set_buffer(2, sg);
        enc.set_constant(3, n * ffn);
        enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
        kqmm_(enc, ly.kqdown_t, ly.kqdown, sg, ao, ffn, H, n);
        residual(x, ao, x, n * H);
      } else if (_mixed) {
        // Mixed MLP over all n rows (no last-layer prune): gate/up each its
        // own bits (dequant+dense into sg / kqubuf), SwiGLU, then down.
        rms(x, 0, ly.post_ln, hn, 0, n, H);
        aqmm_(enc, ly.guw, ly.gus, ly.gub, ly.gate_bits, hn, sg, H, ffn, n);
        aqmm_(enc, ly.uw, ly.us, ly.ub, ly.up_bits, hn, kqubuf, H, ffn, n);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, sg);
        enc.set_buffer(1, kqubuf);
        enc.set_buffer(2, sg);
        enc.set_constant(3, n * ffn);
        enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
        aqmm_(enc, ly.dw, ly.ds, ly.db, ly.down_bits, sg, ao, ffn, H, n);
        residual(x, ao, x, n * H);
      } else if (L + 1 < c.n_layers || verify_all) {
        rms(x, 0, ly.post_ln, hn, 0, n, H);
        if (mma_mlp) {
          // Matrix-core MLP: dequant interleaved gate|up weight -> dense
          // matmul2d -> gu_full[n, 2*ffn], then SwiGLU-combine -> sg[n,ffn].
          const int N2 = 2 * ffn;
          const std::size_t need = (std::size_t)N2 * H * 2;
          if (_w_deq.empty() || _w_deq.byte_size() < need) {
            _w_deq = _mc->make_shared_buffer(need);
          }
          if (!_skip_dequant) {
            enc.set_function(_fn_dequant);
            enc.set_buffer(0, ly.guw); enc.set_buffer(1, ly.gus);
            enc.set_buffer(2, ly.gub); enc.set_buffer(3, _w_deq);
            enc.set_constant(4, H); enc.set_constant(5, N2);
            enc.dispatch({(unsigned)(H / 8), (unsigned)N2, 1}, {64, 1, 1});
          }
          dense_mma(hn, _w_deq, gu_full, H, N2);   // K=H (<=4096) -> 128x128
          enc.set_function(_fn_swiglu_inter);
          enc.set_buffer(0, gu_full); enc.set_buffer(1, sg);
          enc.set_constant(2, n); enc.set_constant(3, ffn);
          enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
        } else {
          enc.set_function(_fn_qmm_swiglu);
          enc.set_buffer(0, ly.guw);
          enc.set_buffer(1, ly.gus);
          enc.set_buffer(2, ly.gub);
          enc.set_buffer(3, hn);
          enc.set_buffer(4, sg);
          enc.set_constant(5, H);
          enc.set_constant(6, 2 * ffn);
          enc.set_constant(7, n);
          const unsigned gx = (unsigned)(((2 * ffn + 31) / 32) * 32);
          const unsigned gy = (unsigned)(((n + 31) / 32) * 2);
          enc.dispatch({gx, gy, 2}, {32, 2, 2});
        }
        qmm(ly.dw, ly.ds, ly.db, sg, ao, ffn, H);
        residual(x, ao, x, n * H);
      } else {
        const std::size_t roff = (std::size_t)(n - 1) * H * 2;  // bytes
        rms(x, roff, ly.post_ln, hn, 0, 1, H);   // hn[0] = norm(x[n-1])
        enc.set_function(_fn_qmv_swiglu);
        enc.set_buffer(0, ly.guw);
        enc.set_buffer(1, ly.gus);
        enc.set_buffer(2, ly.gub);
        enc.set_buffer(3, hn);
        enc.set_buffer(4, sg);
        enc.set_constant(5, H);
        enc.set_constant(6, 2 * ffn);
        enc.dispatch({32, (unsigned)(ffn / 2), 1}, {32, 2, 1});
        enc.set_function(_fn_qmv_add);          // x[n-1] += down(sg)
        enc.set_buffer(0, ly.dw);
        enc.set_buffer(1, ly.ds);
        enc.set_buffer(2, ly.db);
        enc.set_buffer(3, sg);
        enc.set_buffer(4, x, roff);
        enc.set_constant(5, ffn);
        enc.set_constant(6, H);
        enc.set_buffer(7, x, roff);
        enc.dispatch({32, (unsigned)(H / 4), 1}, {32, 2, 1});
      }
    }

    if (verify_all && preds_out) {
      // MTP batched verify: final-norm ALL n rows, lm_head over the whole
      // [n, H] stack (steel GEMM M=n -> weights read ONCE for all drafts),
      // then a per-row argmax -> the per-position greedy predictions.
      rms(x, 0, _final_ln, hn, 0, n, H);
      const int lb = _tied ? _embed_bits : _lm_bits;
      enc.set_function((_mixed && lb == 8) ? _fn_qmm8 : _fn_qmm);
      enc.set_buffer(0, _tied ? _embed_w : _lm_w);
      enc.set_buffer(1, _tied ? _embed_s : _lm_s);
      enc.set_buffer(2, _tied ? _embed_b : _lm_b);
      enc.set_buffer(3, hn);
      enc.set_buffer(4, vlogits);
      enc.set_constant(5, H);
      enc.set_constant(6, c.vocab);
      enc.set_constant(7, n);
      enc.dispatch({(unsigned)(((c.vocab + 31) / 32) * 32),
                   (unsigned)(((n + 31) / 32) * 2), 2}, {32, 2, 2});
      for (int k = 0; k < n; ++k) {
        enc.set_function(_fn_argmax);
        enc.set_buffer(0, vlogits, (std::size_t)k * c.vocab * 2);
        enc.set_buffer(1, vamax, (std::size_t)k * sizeof(std::int32_t));
        enc.set_constant(2, c.vocab);
        enc.dispatch({256, 1, 1}, {256, 1, 1});
      }
    } else if (return_hidden) {
      // MOSS-TTS: final-norm the last position into hn[0:H]; the caller
      // applies its own output heads. No lm_head, no logit pull.
      rms(x, (std::size_t)(n - 1) * H * 2, _final_ln, hn, 0, 1, H);
    } else {
      // Final norm + lm_head on the last position only.
      rms(x, (std::size_t)(n - 1) * H * 2, _final_ln, hn, 0, 1, H);
      if (_kquant) {
        kqmv_(enc, KQ::kQ6K, _embed_q6k, hn, 0, logits, 0, H, c.vocab);
      } else {
        const int lb = _tied ? _embed_bits : _lm_bits;
        enc.set_function((_mixed && lb == 8) ? _fn_qmv8 : _fn_qmv);
        enc.set_buffer(0, _tied ? _embed_w : _lm_w);
        enc.set_buffer(1, _tied ? _embed_s : _lm_s);
        enc.set_buffer(2, _tied ? _embed_b : _lm_b);
        enc.set_buffer(3, hn);
        enc.set_buffer(4, logits);
        enc.set_constant(5, H);
        enc.set_constant(6, c.vocab);
        enc.dispatch({32, (unsigned)(c.vocab / 4), 1}, {32, 2, 1});
      }
    }
  }
  const auto t_enc = std::chrono::steady_clock::now();
  stream.commit().wait();

  // MTP batched verify: return the per-position greedy predictions; the
  // per-position pre-final-norm hidden stays in `x` for the caller. No
  // [vocab] logit pull (the empty return signals verify mode).
  if (verify_all && preds_out) {
    preds_out->resize((std::size_t)n);
    const auto* a = static_cast<const std::int32_t*>(vamax.contents());
    for (int k = 0; k < n; ++k) { (*preds_out)[(std::size_t)k] = a[k]; }
    return {};
  }

  // MOSS-TTS: hand the last position's final-normed hidden (hn[0:H]) to the
  // caller's own output heads. No [vocab] logit pull (the empty return).
  if (return_hidden && hidden_out) {
    *hidden_out = std::move(hn);
    return {};
  }

  // MTP drafter input: snapshot the last position's pre-final-norm hidden
  // (x holds the post-last-layer residual stream; the final norm read it but
  // did not modify it). Cheap host copy; only when an MTP head is present.
  if (_mtp.ok && !_mtp_h.empty()) {
    std::memcpy(_mtp_h.contents(),
                static_cast<const char*>(x.contents()) +
                    (std::size_t)(n - 1) * H * 2,
                (std::size_t)H * 2);
  }

  if (kPProf) {
    const auto t_gpu = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::printf("[qwen-prefill-prof n=%4d] alloc %.3f ms | encode %.3f ms | "
                "commit+gpu %.3f ms\n", n, ms(t_p0, t_alloc),
                ms(t_alloc, t_enc), ms(t_enc, t_gpu));
  }

  std::vector<float> out((std::size_t)c.vocab);
  read_elt_(logits.contents(), out.data(), (std::size_t)c.vocab, _cfg.use_bf16);
  return out;
}

void
MetalQwenModel::vqmm_(ComputeEncoder& enc, int m, const SharedBuffer& w,
                      const SharedBuffer& s, const SharedBuffer& b, int bits,
                      const SharedBuffer& xin, const SharedBuffer& y, int K,
                      int N)
{
  if (bits != 8) {
    qmm_auto_(enc, m, w, s, b, xin, y, K, N);   // 4-bit: qmv_batch/qmv/steel
    return;
  }
  // 8-bit, 2..kQmvBatchMaxRows: batched-GEMV (weight read once across the rows,
  // MAXM=2 -> ceil(m/2) tiles along grid.z) -- qmv bandwidth, so a small-M
  // verify costs ~1 decode step instead of steel's ~3x.
  if (m > 1 && m <= kQmvBatchMaxRows && _fn_qmv8_batch.valid()) {
    enc.set_function(_fn_qmv8_batch);
    enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
    enc.set_buffer(3, xin); enc.set_buffer(4, y);
    enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, m);
    enc.dispatch({32, (unsigned)(N / 4), (unsigned)((m + 1) / 2)}, {32, 2, 1});
    return;
  }
  if (m == 1) {
    enc.set_function(_fn_qmv8);
    enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
    enc.set_buffer(3, xin); enc.set_buffer(4, y);
    enc.set_constant(5, K); enc.set_constant(6, N);
    enc.dispatch({32, (unsigned)(N / 4), 1}, {32, 2, 1});
    return;
  }
  // 8-bit, large m: no batched-GEMV -> steel (reads the weight once across m).
  enc.set_function(_fn_qmm8);
  enc.set_buffer(0, w); enc.set_buffer(1, s); enc.set_buffer(2, b);
  enc.set_buffer(3, xin); enc.set_buffer(4, y);
  enc.set_constant(5, K); enc.set_constant(6, N); enc.set_constant(7, m);
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
               (unsigned)(((m + 31) / 32) * 2), 2}, {32, 2, 2});
}

bool
MetalQwenModel::mtp_verify_chunk_(ContextId cid, const SharedBuffer& x, int n,
                                  std::vector<std::int32_t>* preds,
                                  std::vector<std::int32_t>* mtp_preds,
                                  std::vector<std::int32_t>* mtp_preds2,
                                  int rope_delta, const GpuSamplerParams& sp,
                                  int seed_slot0, GdnVerifyCache* gcache)
{
  // Direct-quantized batched verify. One forward over the n drafts, a per-
  // position decision (argmax or speculative sample), and the fused MTP head.
  // Two weight layouts:
  //  * mixed-affine (OptiQ): the matmuls are direct quantized batched-GEMV
  //    (vqmm_: weight read ONCE per group across the n rows, NO f16 dequant),
  //    DE-FUSED (a batched-GEMV can't write a column slice of a fused buffer),
  //    so q|k|v and the in_proj parts each land in their own buffer.
  //  * native k-quant (GGUF NextN): the matmuls dequant the raw k-quant block
  //    into the shared f16 scratch then a dense GEMM (kqmm_), exactly as the
  //    main k-quant prefill -- the FUSED q|k|v layout (qfull) is reused here,
  //    sliced into q|gate|k|v just like encode_decode_step_.
  const Config& c = _cfg;
  const int H = c.hidden, D = c.head_dim, Hq = c.n_heads, Hkv = c.n_kv_heads;
  const int qd = c.qd(), kd = c.kd();
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, Kc = c.gdn_conv_kernel, ffn = c.ffn_inner;
  const float eps = c.rms_eps, scale = 1.0f / std::sqrt((float)D);
  const int gate = c.attn_output_gate ? 1 : 0;
  const int qdo = gate ? 2 * qd : qd;

  struct Chunk { std::size_t page_off; int slot; int src_off; int cnt; };
  std::vector<Chunk> chunks;
  int q_offset = -1;
  for (int written = 0; written < n;) {
    const int cap = _ctx->next_append_capacity(cid);
    const int cnt = std::min(n - written, cap);
    ContextManager::AppendSlot s = _ctx->append(cid, cnt);
    if (!s.valid()) { return false; }
    if (q_offset < 0) { q_offset = s.position; }
    chunks.push_back({(std::size_t)s.page_id.v * _ctx->page_stride_bytes(),
                      s.slot_offset, written, cnt});
    written += cnt;
  }
  if (q_offset < 0) { q_offset = 0; }
  const int page_tokens = _ctx->page_tokens();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(_pgtab.contents()));

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer hn = buf((std::size_t)n * H);
  SharedBuffer qraw = buf((std::size_t)n * qdo);
  // k-quant fused scratch: q|k|v dequant into one [n, qdo+2kd] GEMM output
  // (qfull), and the a|b in_proj split [n, 2*Hv] (the affine path GEMVs each
  // part straight into qkv/zbuf/abuf/bbuf instead).
  SharedBuffer qfull =
      _kquant ? buf((std::size_t)n * (qdo + 2 * kd)) : SharedBuffer{};
  SharedBuffer abtmp =
      _kquant ? buf((std::size_t)n * 2 * Hv) : SharedBuffer{};
  SharedBuffer q3 = buf((std::size_t)n * qd), gate3 = buf((std::size_t)n * qd);
  SharedBuffer kbuf = buf((std::size_t)n * kd), vbuf = buf((std::size_t)n * kd);
  SharedBuffer qt = buf((std::size_t)n * qd), kt = buf((std::size_t)n * kd),
               vt = buf((std::size_t)n * kd);
  SharedBuffer at = buf((std::size_t)n * qd), att = buf((std::size_t)n * qd),
               ao = buf((std::size_t)n * H);
  SharedBuffer qkv = buf((std::size_t)n * Cd);
  SharedBuffer zbuf = buf((std::size_t)n * vald);
  SharedBuffer abuf = buf((std::size_t)n * Hv), bbuf = buf((std::size_t)n * Hv);
  SharedBuffer convout = buf((std::size_t)n * Cd);
  SharedBuffer gbuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  SharedBuffer betabuf = _mc->make_shared_buffer((std::size_t)n * Hv * 4);
  SharedBuffer ygdn = buf((std::size_t)n * vald);
  SharedBuffer normout = buf((std::size_t)n * vald);
  SharedBuffer sg = buf((std::size_t)n * ffn), upb = buf((std::size_t)n * ffn);
  SharedBuffer vlogits = buf((std::size_t)n * c.vocab);
  SharedBuffer vamax =
      _mc->make_shared_buffer((std::size_t)n * sizeof(std::int32_t));
  // Speculative-sampling scratch (only when the verify samples): per-position
  // softmax-weight workspace [n,vocab] (the sample kernel can't alias one ws
  // across the n in-flight dispatches) + a zeroed seen-set the kernel still
  // binds (penalties are off on this path).
  SharedBuffer vsample_ws, vseen;
  if (!sp.greedy && _fn_sample.valid()) {
    vsample_ws = buf((std::size_t)n * c.vocab);
    vseen = _mc->make_shared_buffer((std::size_t)c.vocab);
    if (!vsample_ws.empty() && !vseen.empty()) {
      std::memset(vseen.contents(), 0, (std::size_t)c.vocab);
    }
  }

  // Fused MTP head: append the verify positions to the MTP layer's own
  // (fresh-per-verify) KV so the MTP drafts in the SAME command buffer. For
  // depth-2 a SECOND window of n positions is appended (the chained second MTP
  // application), so the ctx holds 2n positions and depth-2 attends causally
  // over both windows (a small leak to rejected drafts -- verification keeps
  // the output exact regardless).
  const bool run_mtp = (mtp_preds != nullptr) && _mtp.ok;
  const bool run_mtp2 = run_mtp && (mtp_preds2 != nullptr);
  SharedBuffer emb_P, mnorm_e, mnorm_h, mcomb, mcomb2, mctmp, mqfull, mvamax,
      mvamax2;
  ContextManager::AppendSlot mslot;
  std::size_t m_page_off = 0;
  int m_npages = 0;
  if (run_mtp) {
    mtp_ctx_reset_();
    mslot = _mtp.ctx->append(_mtp.cid, run_mtp2 ? 2 * n : n);
    if (!mslot.valid()) { return false; }
    m_page_off = (std::size_t)mslot.page_id.v * _mtp.ctx->page_stride_bytes();
    m_npages = _mtp.ctx->fill_page_table(
        _mtp.cid, static_cast<std::int32_t*>(_mtp.pgt.contents()));
    emb_P = buf((std::size_t)n * H);
    mnorm_e = buf((std::size_t)n * H);
    mnorm_h = buf((std::size_t)n * H);
    mcomb = buf((std::size_t)n * H);
    mctmp = buf((std::size_t)n * H);
    mqfull = buf((std::size_t)n * (qdo + 2 * kd));
    mvamax = _mc->make_shared_buffer((std::size_t)n * sizeof(std::int32_t));
    if (run_mtp2) {
      mcomb2 = buf((std::size_t)n * H);
      mvamax2 = _mc->make_shared_buffer((std::size_t)n * sizeof(std::int32_t));
    }
  }

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                   const SharedBuffer& w, const SharedBuffer& y,
                   std::size_t yoff, int R, int Hd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin, xoff); enc.set_buffer(1, w);
      enc.set_buffer(2, y, yoff);
      enc.set_constant(3, Hd); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& in, const SharedBuffer& out, int Hh,
                      int S, int W, int off, int block, int gstride) {
      enc.set_function(_fn_head_slice);
      enc.set_buffer(0, in, 0); enc.set_buffer(1, out, 0);
      enc.set_constant(2, Hh); enc.set_constant(3, S); enc.set_constant(4, W);
      enc.set_constant(5, off); enc.set_constant(6, block);
      enc.set_constant(7, gstride);
      enc.dispatch({(unsigned)(Hh * W), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, A); enc.set_constant(3, Bd); enc.set_constant(4, D);
      enc.dispatch({(unsigned)D, (unsigned)Bd, (unsigned)A},
                   {(unsigned)D, 1, 1});
    };
    // Main-model RoPE base = KV slot + rope_delta: sequential text passes
    // rope_delta 0 (rope == slot); a post-multimodal decode passes the mROPE
    // offset so the verify uses the same rotary position the serial decode
    // would. Attention masking + KV writes still use the true slot (q_offset).
    const int rope_base = q_offset + rope_delta;
    auto rope = [&](const SharedBuffer& xb, int heads) {
      enc.set_function(_fn_rope_partial);
      enc.set_buffer(0, xb); enc.set_buffer(1, _inv_freq);
      enc.set_constant(2, heads); enc.set_constant(3, n);
      enc.set_constant(4, D); enc.set_constant(5, c.rotary_dim);
      enc.set_constant(6, rope_base);
      enc.dispatch({(unsigned)(c.rotary_dim / 2), (unsigned)n, (unsigned)heads},
                   {(unsigned)(c.rotary_dim / 2), 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& bb,
                        const SharedBuffer& out, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a); enc.set_buffer(1, bb); enc.set_buffer(2, out);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto kv_write = [&](const SharedBuffer& src, const SharedBuffer& pool) {
      for (const Chunk& ch : chunks) {
        enc.set_function(_fn_kv_write_paged);
        enc.set_buffer(0, src); enc.set_buffer(1, pool, ch.page_off);
        enc.set_constant(2, page_tokens); enc.set_constant(3, D);
        enc.set_constant(4, n); enc.set_constant(5, ch.src_off);
        enc.set_constant(6, ch.slot);
        enc.dispatch({(unsigned)D, (unsigned)ch.cnt, (unsigned)Hkv},
                     {(unsigned)D, 1, 1});
      }
    };

    int gci = 0;   // linear-layer index into gcache (when capturing GDN inputs)
    for (int L = 0; L < c.n_layers; ++L) {
      Layer& ly = _layers[L];
      rms(x, 0, ly.in_ln, hn, 0, n, H);
      if (ly.is_full) {
        const SharedBuffer& kp = *_ctx->kpool(L);
        const SharedBuffer& vp = *_ctx->vpool(L);
        if (_kquant) {
          // Batched k-quant GEMV (weight read once across the n rows): q|k
          // (kqk) into qfull cols [0, kqk_n), v into [kqk_n, Nfqkv), giving the
          // exact encode_decode_step_ fused layout, then slice q|gate|k|v.
          const int Nfqkv = qdo + 2 * kd;
          kqmv_batch_(enc, ly.kqk_t, ly.kqk, hn, qfull, H, ly.kqk_n, n,
                      Nfqkv, 0);
          kqmv_batch_(enc, ly.kqv_t, ly.kqv, hn, qfull, H, kd, n, Nfqkv,
                      ly.kqk_n);
          if (gate) {
            hslice(qfull, q3, n * Hq, 2 * D, D, 0, Hq, Nfqkv);
            hslice(qfull, gate3, n * Hq, 2 * D, D, D, Hq, Nfqkv);
          } else {
            hslice(qfull, q3, n, Nfqkv, qd, 0, 0, 0);
          }
          hslice(qfull, kbuf, n, Nfqkv, kd, qdo, 0, 0);
          hslice(qfull, vbuf, n, Nfqkv, kd, qdo + kd, 0, 0);
        } else {
          // De-fused q|k|v, each a direct quantized batched-GEMV.
          vqmm_(enc, n, ly.qw, ly.qs, ly.qb, ly.q_bits, hn, qraw, H, qdo);
          vqmm_(enc, n, ly.kw, ly.ks, ly.kb, ly.k_bits, hn, kbuf, H, kd);
          vqmm_(enc, n, ly.vw, ly.vs, ly.vb, ly.v_bits, hn, vbuf, H, kd);
          if (gate) {
            hslice(qraw, q3, n * Hq, 2 * D, D, 0, Hq, qdo);
            hslice(qraw, gate3, n * Hq, 2 * D, D, D, Hq, qdo);
          } else {
            hslice(qraw, q3, n, qdo, qd, 0, 0, 0);
          }
        }
        rms(q3, 0, ly.q_norm, q3, 0, n * Hq, D);
        rms(kbuf, 0, ly.k_norm, kbuf, 0, n * Hkv, D);
        transpose(q3, qt, n, Hq);
        transpose(kbuf, kt, n, Hkv);
        transpose(vbuf, vt, n, Hkv);
        rope(qt, Hq); rope(kt, Hkv);
        kv_write(kt, kp); kv_write(vt, vp);
        // Attention. The scalar paged kernel runs one simdgroup per (head,
        // query) scanning the WHOLE KV with 32 lanes -- fine at short KV, but
        // it dominates the verify (and erodes MTP) at long context. Past the
        // mb threshold switch to the same flash-decode-GQA the single-token
        // decode uses (each KV head read ONCE for all G query heads, KV split
        // across simdgroups + merge), run once PER query token (n is tiny).
        // Query i attends causally to KV 0..q_offset+i (pos = q_offset+i), so
        // the result equals the scalar multi-query kernel -- and the serial
        // decode it's verified against ALSO uses this kernel, so token-exact.
        const bool long_ctx = (q_offset + n) >= _sdpa_mb_min;
        const bool use_gqa = _gqa_attn && long_ctx && (D <= 256)
            && (D % 32 == 0) && !_d_gqa_oacc.empty();
        if (use_gqa) {
          const int sp = _gqa_split, G = Hq / Hkv;
          const bool use_vec =
              _gqa_vec && (D % 128 == 0) && _fn_sdpa_gqa_vec.valid();
          // Per-query roped q must be contiguous [Hq, D]: transpose qt
          // [Hq,n,D] -> at [n,Hq,D] so query i = at + i*qd. The merges write
          // att [n,Hq,D] directly (no post-attention transpose needed).
          transpose(qt, at, Hq, n);
          for (int i = 0; i < n; ++i) {
            enc.set_function(use_vec ? _fn_sdpa_gqa_vec : _fn_sdpa_gqa);
            enc.set_buffer(0, at, (std::size_t)i * qd * 2);
            enc.set_buffer(1, kp); enc.set_buffer(2, vp);
            enc.set_buffer(3, _d_gqa_oacc); enc.set_buffer(4, _d_gqa_m);
            enc.set_buffer(5, _d_gqa_l);
            enc.set_constant(6, scale); enc.set_constant(7, D);
            enc.set_constant(8, Hq); enc.set_constant(9, Hkv);
            enc.set_constant(10, q_offset + i);
            enc.set_constant(11, page_tokens); enc.set_constant(12, n_pages);
            enc.set_buffer(13, _pgtab); enc.set_constant(14, sp);
            if (use_vec) {
              enc.dispatch({32, (unsigned)Hq, (unsigned)sp},
                           {32, (unsigned)G, 1});
            } else {
              enc.dispatch({32, (unsigned)Hkv, (unsigned)sp}, {32, 1, 1});
            }
            enc.set_function(_fn_sdpa_gqa_merge);
            enc.set_buffer(0, _d_gqa_oacc); enc.set_buffer(1, _d_gqa_m);
            enc.set_buffer(2, _d_gqa_l);
            enc.set_buffer(3, att, (std::size_t)i * qd * 2);
            enc.set_constant(4, D); enc.set_constant(5, sp);
            enc.set_constant(6, Hq);
            enc.dispatch({(unsigned)(Hq * D), 1, 1}, {256, 1, 1});
          }
        } else {
          // n is tiny (<= draft window): the scalar paged kernel is cheapest
          // at short KV.
          enc.set_function(_fn_sdpa_paged);
          enc.set_buffer(0, qt); enc.set_buffer(1, kp); enc.set_buffer(2, vp);
          enc.set_buffer(3, at);
          enc.set_constant(4, scale); enc.set_constant(5, D);
          enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
          enc.set_constant(8, n); enc.set_constant(9, q_offset);
          enc.set_constant(10, page_tokens); enc.set_constant(11, n_pages);
          enc.set_buffer(12, _pgtab);
          enc.dispatch({32, (unsigned)Hq, (unsigned)n}, {32, 1, 1});
          transpose(at, att, Hq, n);
        }
        if (gate) {
          enc.set_function(_fn_mul_sigmoid);
          enc.set_buffer(0, att); enc.set_buffer(1, gate3);
          enc.set_buffer(2, att); enc.set_constant(3, n * qd);
          enc.dispatch({(unsigned)(n * qd), 1, 1}, {256, 1, 1});
        }
        if (_kquant) {
          kqmv_batch_(enc, ly.kqo_t, ly.kqo, att, ao, qd, H, n, H, 0);
        } else {
          vqmm_(enc, n, ly.ow, ly.os, ly.ob, ly.o_bits, att, ao, qd, H);
        }
        residual(x, ao, x, n * H);
      } else {
        const SharedBuffer* csb = _ctx->conv_state(cid, L);
        const SharedBuffer* ssb = _ctx->ssm_state(cid, L);
        // GDN recurrent-step inputs (conv input qkv, gdn_step g/beta) land in
        // this layer's verify cache when capturing, so a partial accept can
        // replay them (gdn_replay_); else shared scratch. The non-recurrent
        // parts (zbuf/abuf/bbuf/convout/ygdn) stay shared -- not needed to
        // re-advance the recurrent state.
        const SharedBuffer& qkv_c =
            (gcache && gci < (int)gcache->qkv.size()) ? gcache->qkv[gci] : qkv;
        const SharedBuffer& gbuf_c =
            (gcache && gci < (int)gcache->gbuf.size()) ? gcache->gbuf[gci]
                                                       : gbuf;
        const SharedBuffer& betabuf_c =
            (gcache && gci < (int)gcache->betabuf.size())
                ? gcache->betabuf[gci] : betabuf;
        ++gci;
        // De-fused in_proj parts. qkv lands in its own [n, Cd] buffer, so
        // conv1d reads it with x_stride = Cd (vs Nf for the fused layout).
        if (_kquant) {
          // qkv (q5_K) + z (q4_K) batched-GEMV into their own buffers; a|b are
          // f16 (kqab [2Hv,H]) -> one dense GEMM -> abtmp[n,2Hv] split a|b.
          kqmv_batch_(enc, ly.kqkv_t, ly.kqkv, hn, qkv_c, H, Cd, n, Cd, 0);
          kqmv_batch_(enc, ly.kqz_t, ly.kqz, hn, zbuf, H, vald, n, vald, 0);
          dense_gemm_(enc, ly.kqab, hn, abtmp, H, 2 * Hv, n);
          hslice(abtmp, abuf, n, 2 * Hv, Hv, 0, 0, 0);
          hslice(abtmp, bbuf, n, 2 * Hv, Hv, Hv, 0, 0);
        } else {
          vqmm_(enc, n, ly.iqw, ly.iqs, ly.iqb, ly.qkv_bits, hn, qkv_c, H, Cd);
          vqmm_(enc, n, ly.izw, ly.izs, ly.izb, ly.z_bits, hn, zbuf, H, vald);
          vqmm_(enc, n, ly.iaw, ly.ias, ly.iab, ly.a_bits, hn, abuf, H, Hv);
          vqmm_(enc, n, ly.ibw, ly.ibs, ly.ibb, ly.b_bits, hn, bbuf, H, Hv);
        }
        enc.set_function(_fn_gdn_conv1d);
        enc.set_buffer(0, *csb); enc.set_buffer(1, qkv_c);
        enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, convout);
        enc.set_constant(4, n); enc.set_constant(5, Cd);
        enc.set_constant(6, Kc); enc.set_constant(7, Cd);   // x_stride = Cd
        enc.set_constant(8, keyd); enc.set_buffer(9, *csb);
        enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
        const std::size_t kb_off = (std::size_t)n * keyd * 2;
        const std::size_t vb_off = 2 * kb_off;
        rms(convout, 0, _gdn_qscale, convout, 0, n * Hk, Dk);
        rms(convout, kb_off, _gdn_kscale, convout, kb_off, n * Hk, Dk);
        enc.set_function(_fn_gdn_g_beta);
        enc.set_buffer(0, abuf); enc.set_buffer(1, bbuf);
        enc.set_buffer(2, ly.A_log); enc.set_buffer(3, ly.dt_bias);
        enc.set_buffer(4, gbuf_c); enc.set_buffer(5, betabuf_c);
        enc.set_constant(6, Hv); enc.set_constant(7, n);
        enc.dispatch({(unsigned)(n * Hv), 1, 1}, {256, 1, 1});
        const bool gdn4 = _fn_gdn_step_ndv4.valid() && (Dv % 4 == 0) &&
                          !_gdn_force_v1;
        enc.set_function(gdn4 ? _fn_gdn_step_ndv4 : _fn_gdn_step);
        enc.set_buffer(0, convout, 0); enc.set_buffer(1, convout, kb_off);
        enc.set_buffer(2, convout, vb_off); enc.set_buffer(3, gbuf_c);
        enc.set_buffer(4, betabuf_c); enc.set_buffer(5, *ssb);
        enc.set_buffer(6, ygdn); enc.set_buffer(7, *ssb);
        enc.set_constant(8, n);
        enc.set_constant(9, _kquant ? -Hk : Hk);   // strided GQA for GGUF
        enc.set_constant(10, Hv);
        const unsigned gdn_dvy = gdn4 ? (unsigned)(Dv / 4) : (unsigned)Dv;
        enc.dispatch({32, gdn_dvy, (unsigned)Hv}, {32, 4, 1});
        rms(ygdn, 0, ly.gdn_norm, normout, 0, n * Hv, Dv);
        enc.set_function(_fn_swiglu);
        enc.set_buffer(0, zbuf); enc.set_buffer(1, normout);
        enc.set_buffer(2, normout); enc.set_constant(3, n * vald);
        enc.dispatch({(unsigned)(n * vald), 1, 1}, {256, 1, 1});
        if (_kquant) {
          kqmv_batch_(enc, ly.kqout_t, ly.kqout, normout, ao, vald, H, n, H, 0);
        } else {
          vqmm_(enc, n, ly.gow, ly.gos, ly.gob, ly.gout_bits, normout, ao,
                vald, H);
        }
        residual(x, ao, x, n * H);
      }
      // MLP: gate/up GEMM + SwiGLU + down (all rows). k-quant batched-GEMV
      // (weight read once) into sg/upb; affine direct batched-GEMV (vqmm_).
      rms(x, 0, ly.post_ln, hn, 0, n, H);
      if (_kquant) {
        kqmv_batch_(enc, ly.kqgate_t, ly.kqgate, hn, sg, H, ffn, n, ffn, 0);
        kqmv_batch_(enc, ly.kqup_t, ly.kqup, hn, upb, H, ffn, n, ffn, 0);
      } else {
        vqmm_(enc, n, ly.guw, ly.gus, ly.gub, ly.gate_bits, hn, sg, H, ffn);
        vqmm_(enc, n, ly.uw, ly.us, ly.ub, ly.up_bits, hn, upb, H, ffn);
      }
      enc.set_function(_fn_swiglu);
      enc.set_buffer(0, sg); enc.set_buffer(1, upb); enc.set_buffer(2, sg);
      enc.set_constant(3, n * ffn);
      enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
      if (_kquant) {
        kqmv_batch_(enc, ly.kqdown_t, ly.kqdown, sg, ao, ffn, H, n, H, 0);
      } else {
        vqmm_(enc, n, ly.dw, ly.ds, ly.db, ly.down_bits, sg, ao, ffn, H);
      }
      residual(x, ao, x, n * H);
    }
    // Final norm + per-position lm_head (direct quantized, M=n) + the verifier's
    // per-position DECISION -> vamax[k] (the token after drafts[0..k]). Greedy:
    // argmax. Speculative sampling: a token sampled from P at slot q_offset+k+1,
    // seeded by that ABSOLUTE slot (q_offset+k+1 - seed_slot0) so it is
    // byte-identical to decode_pipelined's per-step seed -- every committed
    // token is then the verifier's sample from the right conditional P, i.e.
    // exact autoregressive sampling; the drafts only gate how many land/round.
    rms(x, 0, _final_ln, hn, 0, n, H);
    if (_kquant) {
      // Tied q6_K lm_head (the embed table doubles as the head): ONE batched
      // q6_K GEMV reads the [vocab, H] table once for all n positions (vs n
      // re-reads of the huge table, or a ~1 GB fused dequant of it).
      kqmv_batch_(enc, KQ::kQ6K, _embed_q6k, hn, vlogits, H, c.vocab, n,
                  c.vocab, 0);
    } else {
      const int lb = _tied ? _embed_bits : _lm_bits;
      vqmm_(enc, n, _tied ? _embed_w : _lm_w, _tied ? _embed_s : _lm_s,
            _tied ? _embed_b : _lm_b, lb, hn, vlogits, H, c.vocab);
    }
    const bool vsample = !sp.greedy && !vsample_ws.empty();
    for (int k = 0; k < n; ++k) {
      enc.set_buffer(1, vamax, (std::size_t)k * sizeof(std::int32_t));
      if (!vsample) {
        enc.set_function(_fn_argmax);
        enc.set_buffer(0, vlogits, (std::size_t)k * c.vocab * 2);
        enc.set_constant(2, c.vocab);
        enc.dispatch({256, 1, 1}, {256, 1, 1});
      } else {
        const std::uint32_t step_seed = (std::uint32_t)(
            sp.seed + 0x9e3779b9ull *
                          (std::uint64_t)(q_offset + k + 1 - seed_slot0));
        enc.set_function(_fn_sample);
        enc.set_buffer(0, vlogits, (std::size_t)k * c.vocab * 2);
        enc.set_constant(2, c.vocab);
        enc.set_constant(3, sp.temperature);
        enc.set_constant(4, sp.top_p);
        enc.set_constant(5, step_seed);
        enc.set_buffer(6, vsample_ws, (std::size_t)k * c.vocab * 2);
        enc.set_constant(7, sp.n_iter);
        enc.set_constant(8, sp.repetition_penalty);
        enc.set_constant(9, sp.presence_penalty);
        enc.set_constant(10, sp.top_k);
        enc.set_constant(11, sp.min_p);
        enc.set_buffer(12, vseen);
        enc.dispatch({256, 1, 1}, {256, 1, 1});
      }
    }

    // ---- Fused MTP head (the "verify is a decode step that drafts") ----
    // At each position i: combined = fc_e@norm_e(embed(P_i)) + fc_h@norm_h(H_i),
    // one full-attn MTP layer (its own KV), then argmax(lm_head(mtp.norm(.))) =
    // the next-next-token draft. Runs in THIS command buffer on the on-GPU main
    // preds (vamax) + hiddens (x) -- no separate draft forward.
    if (run_mtp) {
      Layer& mly = _mtp.lyr;
      const int Nfqkv = qdo + 2 * kd;
      const SharedBuffer& mkp = *_mtp.ctx->kpool(0);
      const SharedBuffer& mvp = *_mtp.ctx->vpool(0);
      const int mpt = _mtp.ctx->page_tokens();
      // One MTP application over the n positions: combined = fc_e@norm_e(emb(
      // ids)) + fc_h@norm_h(hsrc); the full-attn MTP layer (MTP-ctx window at
      // slot_off / RoPE qpos); mtp.norm + shared lm_head + argmax -> the draft
      // (argmax_out). hidden_out gets the pre-final-norm MTP hidden, which the
      // depth-2 application chains from.
      auto mtp_step = [&](const SharedBuffer& ids, const SharedBuffer& hsrc,
                          int slot_off, int qpos, const SharedBuffer& hidden_out,
                          const SharedBuffer& argmax_out) {
        if (_kquant) {
          embed_q6k_(enc, ids, 0, emb_P, n);    // tied q6_K table
        } else {
          enc.set_function(_fn_embed);
          enc.set_buffer(0, ids);
          enc.set_buffer(1, _embed_w); enc.set_buffer(2, _embed_s);
          enc.set_buffer(3, _embed_b); enc.set_buffer(4, emb_P);
          enc.set_constant(5, H);
          enc.dispatch({(unsigned)H, (unsigned)n, 1}, {256, 1, 1});
        }
        rms(emb_P, 0, _mtp.prenorm_e, mnorm_e, 0, n, H);
        rms(hsrc, 0, _mtp.prenorm_h, mnorm_h, 0, n, H);
        dense_gemm_(enc, _mtp.fc_e, mnorm_e, hidden_out, H, H, n);
        dense_gemm_(enc, _mtp.fc_h, mnorm_h, mctmp, H, H, n);
        residual(hidden_out, mctmp, hidden_out, n * H);
        rms(hidden_out, 0, mly.in_ln, hn, 0, n, H);
        if (_kquant) {
          // q|k (kqk) + v (kqv) batched-GEMV into mqfull[n, Nfqkv].
          kqmv_batch_(enc, mly.kqk_t, mly.kqk, hn, mqfull, H, mly.kqk_n, n,
                      Nfqkv, 0);
          kqmv_batch_(enc, mly.kqv_t, mly.kqv, hn, mqfull, H, kd, n, Nfqkv,
                      mly.kqk_n);
        } else {
          vqmm_(enc, n, mly.qw, mly.qs, mly.qb, 4, hn, mqfull, H, Nfqkv);
        }
        hslice(mqfull, q3, n * Hq, 2 * D, D, 0, Hq, Nfqkv);
        hslice(mqfull, gate3, n * Hq, 2 * D, D, D, Hq, Nfqkv);
        hslice(mqfull, kbuf, n, Nfqkv, kd, qdo, 0, 0);
        hslice(mqfull, vbuf, n, Nfqkv, kd, qdo + kd, 0, 0);
        rms(q3, 0, mly.q_norm, q3, 0, n * Hq, D);
        rms(kbuf, 0, mly.k_norm, kbuf, 0, n * Hkv, D);
        transpose(q3, qt, n, Hq);
        transpose(kbuf, kt, n, Hkv);
        transpose(vbuf, vt, n, Hkv);
        for (int two = 0; two < 2; ++two) {     // RoPE q, then k, at qpos
          enc.set_function(_fn_rope_partial);
          enc.set_buffer(0, two == 0 ? qt : kt);
          enc.set_buffer(1, _inv_freq);
          enc.set_constant(2, two == 0 ? Hq : Hkv); enc.set_constant(3, n);
          enc.set_constant(4, D); enc.set_constant(5, c.rotary_dim);
          enc.set_constant(6, qpos);
          enc.dispatch({(unsigned)(c.rotary_dim / 2), (unsigned)n,
                        (unsigned)(two == 0 ? Hq : Hkv)},
                       {(unsigned)(c.rotary_dim / 2), 1, 1});
        }
        for (int two = 0; two < 2; ++two) {     // KV write k, then v, at slot
          enc.set_function(_fn_kv_write_paged);
          enc.set_buffer(0, two == 0 ? kt : vt);
          enc.set_buffer(1, two == 0 ? mkp : mvp, m_page_off);
          enc.set_constant(2, mpt); enc.set_constant(3, D);
          enc.set_constant(4, n); enc.set_constant(5, 0);
          enc.set_constant(6, mslot.slot_offset + slot_off);
          enc.dispatch({(unsigned)D, (unsigned)n, (unsigned)Hkv},
                       {(unsigned)D, 1, 1});
        }
        enc.set_function(_fn_sdpa_paged);
        enc.set_buffer(0, qt); enc.set_buffer(1, mkp); enc.set_buffer(2, mvp);
        enc.set_buffer(3, at);
        enc.set_constant(4, scale); enc.set_constant(5, D);
        enc.set_constant(6, Hq); enc.set_constant(7, Hkv);
        enc.set_constant(8, n); enc.set_constant(9, qpos);
        enc.set_constant(10, mpt); enc.set_constant(11, m_npages);
        enc.set_buffer(12, _mtp.pgt);
        enc.dispatch({32, (unsigned)Hq, (unsigned)n}, {32, 1, 1});
        transpose(at, att, Hq, n);
        enc.set_function(_fn_mul_sigmoid);
        enc.set_buffer(0, att); enc.set_buffer(1, gate3);
        enc.set_buffer(2, att); enc.set_constant(3, n * qd);
        enc.dispatch({(unsigned)(n * qd), 1, 1}, {256, 1, 1});
        if (_kquant) {
          kqmv_batch_(enc, mly.kqo_t, mly.kqo, att, ao, qd, H, n, H, 0);
        } else {
          vqmm_(enc, n, mly.ow, mly.os, mly.ob, 4, att, ao, qd, H);
        }
        residual(hidden_out, ao, hidden_out, n * H);
        rms(hidden_out, 0, mly.post_ln, hn, 0, n, H);
        if (_kquant) {
          // gate/up batched-GEMV into sg/upb, SwiGLU, then down.
          kqmv_batch_(enc, mly.kqgate_t, mly.kqgate, hn, sg, H, ffn, n, ffn, 0);
          kqmv_batch_(enc, mly.kqup_t, mly.kqup, hn, upb, H, ffn, n, ffn, 0);
          enc.set_function(_fn_swiglu);
          enc.set_buffer(0, sg); enc.set_buffer(1, upb); enc.set_buffer(2, sg);
          enc.set_constant(3, n * ffn);
          enc.dispatch({(unsigned)(n * ffn), 1, 1}, {256, 1, 1});
          kqmv_batch_(enc, mly.kqdown_t, mly.kqdown, sg, ao, ffn, H, n, H, 0);
        } else {
          // Interleaved gate/up SwiGLU via the size-adaptive path: at the small
          // M of a verify it picks the batched-GEMV swiglu (weight read once
          // across rows) instead of steel (~3x slower; run 1-2x/round).
          qmm_auto_swiglu_(enc, n, mly.guw, mly.gus, mly.gub, hn, sg, H,
                           2 * ffn);
          vqmm_(enc, n, mly.dw, mly.ds, mly.db, 4, sg, ao, ffn, H);
        }
        residual(hidden_out, ao, hidden_out, n * H);
        rms(hidden_out, 0, _mtp.final_norm, hn, 0, n, H);
        if (_kquant) {
          kqmv_batch_(enc, KQ::kQ6K, _embed_q6k, hn, vlogits, H, c.vocab, n,
                      c.vocab, 0);
        } else {
          const int mlb = _tied ? _embed_bits : _lm_bits;
          vqmm_(enc, n, _tied ? _embed_w : _lm_w, _tied ? _embed_s : _lm_s,
                _tied ? _embed_b : _lm_b, mlb, hn, vlogits, H, c.vocab);
        }
        for (int k = 0; k < n; ++k) {
          enc.set_function(_fn_argmax);
          enc.set_buffer(0, vlogits, (std::size_t)k * c.vocab * 2);
          enc.set_buffer(1, argmax_out, (std::size_t)k * sizeof(std::int32_t));
          enc.set_constant(2, c.vocab);
          enc.dispatch({256, 1, 1}, {256, 1, 1});
        }
      };
      // depth-1: from the main hidden + emb(main pred). depth-2: chain from the
      // depth-1 MTP hidden + emb(depth-1 draft), into the SECOND ctx window.
      // Conditioning on the chained ARGMAX (not the candidate token) is what
      // gives good drafts here: the boundary's next token is the main model's
      // prediction, which lies past the candidate window -- conditioning on a
      // candidate instead collapsed acceptance to ~0 (measured). This argmax
      // dependency is also why the MTP output heads can't be batched.
      mtp_step(vamax, x, 0, mslot.position, mcomb, mvamax);
      if (run_mtp2) {
        mtp_step(mvamax, mcomb, n, mslot.position + n, mcomb2, mvamax2);
      }
    }
  }
  stream.commit().wait();
  preds->resize((std::size_t)n);
  const auto* a = static_cast<const std::int32_t*>(vamax.contents());
  for (int k = 0; k < n; ++k) { (*preds)[(std::size_t)k] = a[k]; }
  if (run_mtp) {
    mtp_preds->resize((std::size_t)n);
    const auto* mp = static_cast<const std::int32_t*>(mvamax.contents());
    for (int k = 0; k < n; ++k) { (*mtp_preds)[(std::size_t)k] = mp[k]; }
  }
  if (run_mtp2) {
    mtp_preds2->resize((std::size_t)n);
    const auto* mp = static_cast<const std::int32_t*>(mvamax2.contents());
    for (int k = 0; k < n; ++k) { (*mtp_preds2)[(std::size_t)k] = mp[k]; }
  }
  return true;
}

void
MetalQwenModel::gdn_replay_(ContextId cid, int keep, const GdnVerifyCache& gc)
{
  // Re-advance every linear layer's GDN recurrent state to S_keep using the
  // verify's cached conv/step inputs (the caller has already restored each
  // state to the round's S0 snapshot). Same kernels + same inputs as the
  // verify's first `keep` tokens -> bit-identical state. No attention /
  // projections / lm_head.
  if (keep <= 0 || gc.layers.empty()) { return; }
  const Config& c = _cfg;
  const int Cd = c.gdn_conv_dim, keyd = c.key_dim(), vald = c.value_dim();
  const int Hv = c.gdn_v_heads, Dv = c.gdn_v_dim, Dk = c.gdn_k_dim;
  const int Hk = c.gdn_k_heads, Kc = c.gdn_conv_kernel;
  const float eps = c.rms_eps;
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer convout = buf((std::size_t)keep * Cd);
  SharedBuffer ygdn = buf((std::size_t)keep * vald);
  const std::size_t kb_off = (std::size_t)keep * keyd * 2;
  const std::size_t vb_off = 2 * kb_off;
  const bool gdn4 =
      _fn_gdn_step_ndv4.valid() && (Dv % 4 == 0) && !_gdn_force_v1;
  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto rms = [&](const SharedBuffer& xin, std::size_t xoff,
                   const SharedBuffer& w, const SharedBuffer& y,
                   std::size_t yoff, int R, int Hd) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, xin, xoff); enc.set_buffer(1, w);
      enc.set_buffer(2, y, yoff);
      enc.set_constant(3, Hd); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    for (std::size_t gi = 0; gi < gc.layers.size(); ++gi) {
      const int L = gc.layers[gi];
      const Layer& ly = _layers[(std::size_t)L];
      const SharedBuffer* csb = _ctx->conv_state(cid, L);
      const SharedBuffer* ssb = _ctx->ssm_state(cid, L);
      if (csb == nullptr || ssb == nullptr) { continue; }
      enc.set_function(_fn_gdn_conv1d);
      enc.set_buffer(0, *csb); enc.set_buffer(1, gc.qkv[gi]);
      enc.set_buffer(2, ly.conv_w); enc.set_buffer(3, convout);
      enc.set_constant(4, keep); enc.set_constant(5, Cd);
      enc.set_constant(6, Kc); enc.set_constant(7, Cd);   // x_stride = Cd
      enc.set_constant(8, keyd); enc.set_buffer(9, *csb);
      enc.dispatch({(unsigned)Cd, 1, 1}, {256, 1, 1});
      rms(convout, 0, _gdn_qscale, convout, 0, keep * Hk, Dk);
      rms(convout, kb_off, _gdn_kscale, convout, kb_off, keep * Hk, Dk);
      enc.set_function(gdn4 ? _fn_gdn_step_ndv4 : _fn_gdn_step);
      enc.set_buffer(0, convout, 0); enc.set_buffer(1, convout, kb_off);
      enc.set_buffer(2, convout, vb_off); enc.set_buffer(3, gc.gbuf[gi]);
      enc.set_buffer(4, gc.betabuf[gi]); enc.set_buffer(5, *ssb);
      enc.set_buffer(6, ygdn); enc.set_buffer(7, *ssb);
      enc.set_constant(8, keep);
      enc.set_constant(9, _kquant ? -Hk : Hk);   // strided GQA for GGUF
      enc.set_constant(10, Hv);
      const unsigned gdn_dvy = gdn4 ? (unsigned)(Dv / 4) : (unsigned)Dv;
      enc.dispatch({32, gdn_dvy, (unsigned)Hv}, {32, 4, 1});
    }
  }
  stream.commit().wait();
}

std::vector<float>
MetalQwenModel::embed_text(const std::vector<std::int32_t>& ids)
{
  std::vector<float> out;
  if (ids.empty()) { return out; }
  SharedBuffer e =
      _muxer->fetch_text(std::span<const std::int32_t>(ids.data(), ids.size()));
  if (e.empty()) { return out; }
  out.resize(ids.size() * (std::size_t)_cfg.hidden);
  read_elt_(e.contents(), out.data(), out.size(), _cfg.use_bf16);
  return out;
}

std::int32_t
MetalQwenModel::forward_argmax(std::int32_t token_id)
{
  // Prefer the in-stream fast path (embed gather + on-GPU argmax folded into
  // the decode command buffer -- no separate embed submit, no [vocab] host
  // pull); fall back to forward()+host argmax when unavailable.
  const std::int32_t r = decode_step_fast(_cid, token_id);
  if (r != std::numeric_limits<std::int32_t>::min()) { return r; }
  std::vector<float> l = forward(token_id);
  std::int32_t best = 0;
  float bv = l.empty() ? 0.0f : l[0];
  for (std::size_t i = 1; i < l.size(); ++i) {
    if (l[i] > bv) { bv = l[i]; best = (std::int32_t)i; }
  }
  return best;
}

bool
MetalQwenModel::decode_pipelined(
    ContextId cid, std::int32_t first_token, int n_steps,
    std::vector<std::int32_t>& out_ids, float temperature, float top_p,
    std::uint64_t seed)
{
  out_ids.clear();
  if (n_steps <= 0) { return true; }
  const bool embed_ok = _kquant ? _embed_is_q6k : _fn_embed.valid();
  if (!embed_ok || !_fn_argmax.valid() || !_fn_sample.valid()) {
    return false;
  }
  if (!ensure_decode_scratch_()) { return false; }
  const Config& c = _cfg;
  const int H = c.hidden;
  const bool greedy = (temperature <= 0.0f);

  // GPU-resident token chain (gen_ids[0]=seed; argmax/sample writes [s+1],
  // embed reads [s] -- the token never round-trips to the host).
  metal_compute::SharedBuffer gen_ids = _mc->make_shared_buffer(
      (std::size_t)(n_steps + 1) * sizeof(std::int32_t));
  if (gen_ids.empty()) { return false; }
  static_cast<std::int32_t*>(gen_ids.contents())[0] = first_token;

  // Per-step page-table slices for the full-attn layers (host fills all
  // ahead of GPU execution, so they must not alias). GDN conv/ssm state
  // is updated IN PLACE each step -- the event chain orders those writes.
  const int pt_stride = _ctx->max_pages() * 3;
  metal_compute::SharedBuffer pgt = _mc->make_shared_buffer(
      (std::size_t)n_steps * pt_stride * sizeof(std::int32_t));
  if (pgt.empty()) { return false; }

  // Top-p sampling scratch: [vocab] cached softmax weights + a (here
  // unused, rep=1/pres=0) seen-set the shared sampler kernel still binds.
  metal_compute::SharedBuffer sample_ws, seen_batch;
  int sample_iters = 16;
  if (!greedy) {
    sample_ws = _mc->make_shared_buffer((std::size_t)c.vocab * 2);
    seen_batch = _mc->make_shared_buffer((std::size_t)c.vocab);
    if (sample_ws.empty() || seen_batch.empty()) { return false; }
    std::memset(seen_batch.contents(), 0, (std::size_t)c.vocab);
    if (const char* e = std::getenv("VPIPE_SAMPLE_ITERS")) {
      sample_iters = std::atoi(e);
      if (sample_iters < 1) { sample_iters = 1; }
    }
  }
  GpuSamplerParams bsp;          // batch path: plain temperature + top-p
  bsp.greedy = greedy;
  bsp.temperature = temperature;
  bsp.top_p = top_p;
  bsp.seed = seed;
  bsp.n_iter = sample_iters;

  metal_compute::CommandStream stream = _mc->make_command_stream();
  metal_compute::Event ev = _mc->make_event();
  metal_compute::CommandStream::Fence last;

  for (int s = 0; s < n_steps; ++s) {
    ContextManager::AppendSlot slot = _ctx->append(cid, 1);
    if (!slot.valid()) { return false; }
    const int pos = slot.position;
    const std::size_t page_off =
        (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
    std::int32_t* pt_ptr = static_cast<std::int32_t*>(pgt.contents()) +
                           (std::size_t)s * pt_stride;
    const int n_pages = _ctx->fill_page_table(cid, pt_ptr);
    const std::size_t pt_off =
        (std::size_t)s * pt_stride * sizeof(std::int32_t);

    stream.encode_wait(ev, (std::uint64_t)s);
    {
      ComputeEncoder enc = stream.begin_compute();
      // Zero-copy embed: gather row gen_ids[s] -> _d_x in this command
      // buffer (no muxer command buffer + host memcpy).
      if (_kquant) {
        embed_q6k_(enc, gen_ids, (std::size_t)s, _d_x, 1);
      } else {
        enc.set_function(_fn_embed);
        enc.set_buffer(0, gen_ids, (std::size_t)s * sizeof(std::int32_t));
        enc.set_buffer(1, _embed_w);
        enc.set_buffer(2, _embed_s);
        enc.set_buffer(3, _embed_b);
        enc.set_buffer(4, _d_x);
        enc.set_constant(5, H);
        enc.dispatch({(unsigned)H, 1, 1}, {256, 1, 1});
      }

      // Text decode: rope position == KV slot position.
      encode_decode_step_(enc, cid, pos, pos, page_off, n_pages, slot, pgt,
                          pt_off);

      // Next token -> gen_ids[s+1], on-GPU (greedy argmax or temp+top-p).
      const std::uint32_t step_seed =
          (std::uint32_t)(seed + 0x9e3779b9ull * (std::uint64_t)(s + 1));
      encode_sample_(enc, _d_logits, gen_ids,
                     (std::size_t)(s + 1) * sizeof(std::int32_t), bsp,
                     step_seed, sample_ws, seen_batch);
    }
    stream.encode_signal(ev, (std::uint64_t)(s + 1));
    last = stream.commit();
  }
  last.wait();
  out_ids.resize(n_steps);
  const auto* g = static_cast<const std::int32_t*>(gen_ids.contents());
  for (int s = 0; s < n_steps; ++s) { out_ids[s] = g[s + 1]; }
  return true;
}

void
MetalQwenModel::encode_sample_(
    metal_compute::ComputeEncoder& enc,
    const metal_compute::SharedBuffer& logits,
    const metal_compute::SharedBuffer& out_id, std::size_t out_off,
    const GpuSamplerParams& sp, std::uint32_t step_seed,
    const metal_compute::SharedBuffer& sample_ws,
    const metal_compute::SharedBuffer& seen)
{
  enc.set_buffer(1, out_id, out_off);
  if (sp.greedy) {
    enc.set_function(_fn_argmax);
    enc.set_buffer(0, logits);
    enc.set_constant(2, _cfg.vocab);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
  } else {
    enc.set_function(_fn_sample);
    enc.set_buffer(0, logits);
    enc.set_constant(2, _cfg.vocab);
    enc.set_constant(3, sp.temperature);
    enc.set_constant(4, sp.top_p);
    enc.set_constant(5, step_seed);
    enc.set_buffer(6, sample_ws);
    enc.set_constant(7, sp.n_iter);
    enc.set_constant(8, sp.repetition_penalty);
    enc.set_constant(9, sp.presence_penalty);
    enc.set_constant(10, sp.top_k);
    enc.set_constant(11, sp.min_p);
    enc.set_buffer(12, seen);
    enc.dispatch({256, 1, 1}, {256, 1, 1});
  }
}

bool
MetalQwenModel::pdecode_begin(ContextId cid, std::int32_t first_token,
                              std::span<const std::int32_t> prompt,
                              const GpuSamplerParams& sp, int max_tokens,
                              int rope_first)
{
  const bool embed_ok = _kquant ? _embed_is_q6k : _fn_embed.valid();
  if (!embed_ok || !_fn_argmax.valid() ||
      (!sp.greedy && !_fn_sample.valid())) {
    return false;
  }
  if (!ensure_decode_scratch_()) { return false; }
  if (max_tokens < 1) { max_tokens = 1; }

  PDecode& pd = _pdec[cid.v];
  pd = PDecode{};                 // reset any stale session for this cid
  pd.cid = cid;
  pd.sp = sp;
  pd.produced = 1;                // gen_ids[0] = first_token
  pd.committed = 1;               // next forward writes gen_ids[1]
  // Default depth-1 (no run-ahead). The GDN ssm/conv ring makes depth>=2
  // CORRECT (token-exact + rollback-safe), but measured depth-2 ~= depth-1 for
  // Qwen3.5-4B: its CPU-encode bubble (~0.26ms) is tiny vs the ~21ms/token GPU
  // decode, so run-ahead has almost nothing to hide (unlike gemma-e4b's
  // +3.6%). Keeping depth-1 avoids the ~96MB ssm shadow alloc for no gain.
  // VPIPE_QWEN_PDECODE_DEPTH=2 opts in (lower variance; MTP foundation).
  pd.depth = 1;
  if (const char* e = std::getenv("VPIPE_QWEN_PDECODE_DEPTH")) {
    pd.depth = std::max(1, std::min(4, std::atoi(e)));
  }
  pd.cap = max_tokens + 1 + pd.depth;
  // mROPE anchor: the first decode token lands at KV slot seq_len with
  // rotary position `rope_first` (-1 => sequential, rope == KV slot).
  pd.rope_base = rope_first;
  pd.kv_base = _ctx->seq_len_of(cid);

  pd.gen_ids = _mc->make_shared_buffer(
      (std::size_t)pd.cap * sizeof(std::int32_t));
  const int pt_stride = _ctx->max_pages() * 3;
  pd.pgt = _mc->make_shared_buffer(
      (std::size_t)pt_stride * sizeof(std::int32_t));
  if (pd.gen_ids.empty() || pd.pgt.empty()) { _pdec.erase(cid.v); return false; }
  static_cast<std::int32_t*>(pd.gen_ids.contents())[0] = first_token;

  if (!sp.greedy) {
    pd.sample_ws = _mc->make_shared_buffer((std::size_t)_cfg.vocab * 2);
    pd.seen = _mc->make_shared_buffer((std::size_t)_cfg.vocab);   // uint8
    if (pd.sample_ws.empty() || pd.seen.empty()) {
      _pdec.erase(cid.v); return false;
    }
    // Prime the penalty seen-set with the prompt tokens + first id.
    std::memset(pd.seen.contents(), 0, (std::size_t)_cfg.vocab);
    auto* sb = static_cast<std::uint8_t*>(pd.seen.contents());
    for (std::int32_t t : prompt) {
      if (t >= 0 && t < _cfg.vocab) { sb[t] = 1; }
    }
    if (first_token >= 0 && first_token < _cfg.vocab) { sb[first_token] = 1; }
  }

  pd.stream = _mc->make_command_stream();
  pd.ev = _mc->make_event();

  // Arm the GDN recurrent-state run-ahead ring for depth>1 (no-op on a dense
  // model with no GDN layers, or when depth==1). If the shadow alloc fails,
  // fall back to depth-1 (no run-ahead) rather than abort decode. Done LAST so
  // there is no failure path after it that could leave the ring half-armed.
  if (!_ctx->gdn_ring_begin(cid, pd.depth)) { pd.depth = 1; }
  return true;
}

bool
MetalQwenModel::pdecode_commit(ContextId cid)
{
  auto it = _pdec.find(cid.v);
  if (it == _pdec.end()) { return false; }
  PDecode& pd = it->second;
  if ((int)pd.ring.size() >= pd.depth) { return false; }  // pipeline full
  const int in_idx = pd.committed - 1;
  const int out_idx = pd.committed;
  if (out_idx >= pd.cap) { return false; }

  ContextManager::AppendSlot slot = _ctx->append(cid, 1);
  if (!slot.valid()) { return false; }
  const int pos = slot.position;
  // Rotary position: sequential text decode (rope_base<0) uses the KV
  // slot; after a multimodal prefill rope advances from the mROPE anchor.
  const int rpos = (pd.rope_base < 0) ? pos
                                      : pd.rope_base + (pos - pd.kv_base);
  const std::size_t page_off =
      (std::size_t)slot.page_id.v * _ctx->page_stride_bytes();
  const int n_pages =
      _ctx->fill_page_table(cid, static_cast<std::int32_t*>(pd.pgt.contents()));

  const std::uint64_t s = pd.gpu_step;
  pd.stream.encode_wait(pd.ev, s);
  {
    ComputeEncoder enc = pd.stream.begin_compute();
    // In-stream embed gather of gen_ids[in_idx] -> _d_x.
    if (_kquant) {
      embed_q6k_(enc, pd.gen_ids, (std::size_t)in_idx, _d_x, 1);
    } else {
      enc.set_function(_fn_embed);
      enc.set_buffer(0, pd.gen_ids,
                     (std::size_t)in_idx * sizeof(std::int32_t));
      enc.set_buffer(1, _embed_w);
      enc.set_buffer(2, _embed_s);
      enc.set_buffer(3, _embed_b);
      enc.set_buffer(4, _d_x);
      enc.set_constant(5, _cfg.hidden);
      enc.dispatch({(unsigned)_cfg.hidden, 1, 1}, {256, 1, 1});
    }

    encode_decode_step_(enc, cid, pos, rpos, page_off, n_pages, slot, pd.pgt, 0);

    const std::uint32_t step_seed = (std::uint32_t)(
        pd.sp.seed + 0x9e3779b9ull * (std::uint64_t)(s + 1));
    encode_sample_(enc, _d_logits, pd.gen_ids,
                   (std::size_t)out_idx * sizeof(std::int32_t), pd.sp,
                   step_seed, pd.sample_ws, pd.seen);
  }
  pd.stream.encode_signal(pd.ev, s + 1);
  PDecode::InFlight f;
  f.fence = pd.stream.commit();
  f.idx = out_idx;
  pd.ring.push_back(std::move(f));
  pd.gpu_step = s + 1;
  pd.committed = out_idx + 1;
  // One GDN advance per committed forward: the slot this step WROTE becomes
  // the next commit's read slot. No-op when the ring is off (depth-1).
  _ctx->gdn_ring_advance(cid);
  return true;
}

std::int32_t
MetalQwenModel::pdecode_next(ContextId cid)
{
  auto it = _pdec.find(cid.v);
  if (it == _pdec.end()) { return -1; }
  PDecode& pd = it->second;
  if (pd.ring.empty()) { return -1; }
  PDecode::InFlight f = std::move(pd.ring.front());
  pd.ring.pop_front();
  f.fence.wait();
  const std::int32_t tok =
      static_cast<const std::int32_t*>(pd.gen_ids.contents())[f.idx];
  pd.produced = f.idx + 1;
  return tok;
}

void
MetalQwenModel::pdecode_end(ContextId cid)
{
  auto it = _pdec.find(cid.v);
  if (it == _pdec.end()) { return; }
  PDecode& pd = it->second;
  for (auto& f : pd.ring) { f.fence.wait(); }
  pd.ring.clear();
  // Discard a run-ahead speculative tail: roll the paged KV AND the GDN
  // recurrent (ssm/conv) ring back to the last drained token, matching the
  // synchronous loop (a stop token's forward is never committed). depth-1
  // has committed == produced -> spec 0 (no behavior change); gdn_ring_end
  // then restores the canonical GDN buffers and disarms the ring. All ring
  // fences are already waited above, so the end-time handle swap is safe.
  const int spec = pd.committed - pd.produced;
  if (spec > 0) { _ctx->kv_rollback(pd.cid, spec); }
  _ctx->gdn_ring_rollback(pd.cid, spec);
  _ctx->gdn_ring_end(pd.cid);
  _pdec.erase(it);
}

// ---------------------------------------------------------------------
// MTP speculative decode (MTP head FUSED into the verify + rollback).
// ---------------------------------------------------------------------
void
MetalQwenModel::mtp_ctx_reset_()
{
  if (!_mtp.ok) { return; }
  const int seq = _mtp.ctx->seq_len_of(_mtp.cid);
  if (seq > 0) { _mtp.ctx->kv_rollback(_mtp.cid, seq); }
}

bool
MetalQwenModel::mtp_decode(ContextId cid, std::int32_t first_token,
                           int n_steps, std::vector<std::int32_t>& out_ids,
                           int draft_len, long* accepted_out, long* rounds_out,
                           const MtpDecodeCtl& ctl)
{
  out_ids.clear();
  if (ctl.hit_stop) { *ctl.hit_stop = false; }
  // The MTP head is fused into mtp_verify_chunk_, supported on the de-fused
  // affine (mixed) path and the native k-quant (GGUF NextN) path.
  if (!_mtp.ok || n_steps <= 0 || !(_mixed || _kquant)) { return false; }
  if (!ensure_decode_scratch_()) { return false; }
  const Config& c = _cfg;
  // Perf-investigation knobs (default off): VPIPE_MTP_DRAFT_LEN overrides the
  // speculative depth (1 = two drafts/round, 2 = three); VPIPE_MTP_PROFILE
  // prints a per-round wall-clock breakdown so the speedup ceiling can be
  // attributed to the verify forward vs the partial-accept re-run vs the GDN
  // snapshot. mtp_verify_chunk_ commits+waits internally, so these wraps time
  // the GPU+sync of each piece.
  if (const char* e = std::getenv("VPIPE_MTP_DRAFT_LEN")) {
    const int d = std::atoi(e);
    if (d >= 1) { draft_len = d; }
  }
  static const bool mtp_prof = (std::getenv("VPIPE_MTP_PROFILE") != nullptr);
  using mtp_clock = std::chrono::steady_clock;
  double prof_verify_ms = 0.0, prof_rerun_ms = 0.0, prof_snap_ms = 0.0;
  long prof_reruns = 0;
  auto prof_ms = [](mtp_clock::time_point a, mtp_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  // k-quant gathers the MTP drafter's embed via embed_q6k (tied q6_K table),
  // not the affine _fn_embed; both paths still need the per-position argmax.
  const bool kembed_ok = _kquant ? (_embed_is_q6k && _fn_embed_q6k.valid())
                                  : _fn_embed.valid();
  if (!kembed_ok || !_fn_argmax.valid()) { return false; }
  // Speculative SAMPLING needs the sample kernel.
  const bool sampling = !ctl.sampler.greedy;
  if (sampling && !_fn_sample.valid()) { return false; }

  // KV slot of the first decoded token (= seq_len now). Anchors both the RoPE
  // offset and (for sampling) the per-slot seed so the verify reproduces the
  // serial decode path's positions AND its per-slot randomness.
  const int seed_slot0 = _ctx->seq_len_of(cid);

  // RoPE offset: rope_first (the first decoded token's rotary position) minus
  // its KV slot (the current seq length). 0 for sequential text; the mROPE
  // gap after a multimodal prefill otherwise. Constant across the whole decode
  // (every token's rope advances in lockstep with its KV slot), so the verify
  // just adds it to each token's slot -- matching the serial decode path.
  const int rope_delta =
      (ctl.rope_first < 0) ? 0 : (ctl.rope_first - seed_slot0);

  // GDN recurrent-state snapshot (one conv + ssm buffer per linear layer),
  // reused across rounds. The batched verify advances the canonical GDN state
  // in place by K; unlike the paged KV (positionally truncatable), the
  // recurrent state has no position to rewind, so a partial accept restores
  // this snapshot and re-runs the accepted tokens. (The per-token gdn ring
  // can't be used here -- its cursor is per-context, so it would force the
  // verify back to one-token-through-all-layers, defeating the batching.)
  std::vector<int> glayers;
  std::vector<SharedBuffer> gconv, gssm;
  for (int L = 0; L < c.n_layers; ++L) {
    if (c.layer_is_full(L)) { continue; }
    const SharedBuffer* cs = _ctx->conv_state(cid, L);
    const SharedBuffer* ss = _ctx->ssm_state(cid, L);
    if (cs == nullptr || ss == nullptr) { continue; }
    glayers.push_back(L);
    gconv.push_back(_mc->make_shared_buffer(cs->byte_size()));
    gssm.push_back(_mc->make_shared_buffer(ss->byte_size()));
  }
  auto snapshot_gdn = [&](bool save) {
    for (std::size_t i = 0; i < glayers.size(); ++i) {
      const SharedBuffer* cs = _ctx->conv_state(cid, glayers[i]);
      const SharedBuffer* ss = _ctx->ssm_state(cid, glayers[i]);
      if (save) {
        std::memcpy(gconv[i].contents(), cs->contents(), cs->byte_size());
        std::memcpy(gssm[i].contents(), ss->contents(), ss->byte_size());
      } else {
        std::memcpy(cs->contents(), gconv[i].contents(), cs->byte_size());
        std::memcpy(ss->contents(), gssm[i].contents(), ss->byte_size());
      }
    }
  };

  // Depth: draft_len>=2 chains a 2nd MTP application (3 drafts/round); else
  // depth-1 (2 drafts/round). Both are FULLY fused -- the verify emits all the
  // next round's drafts, no separate draft forward.
  const bool d2 = (draft_len >= 2);

  // Per-(linear layer) GDN-input cache for the partial-accept GDN replay: sized
  // for the max drafts/round (depth-1: 2, depth-2: 3). Each verify writes its
  // recurrent-step inputs here; gdn_replay_ reads them on a partial accept.
  // Allocated once, reused across rounds.
  GdnVerifyCache gcache;
  gcache.layers = glayers;
  {
    const int maxK = d2 ? 3 : 2;
    const int Cd = c.gdn_conv_dim, Hv = c.gdn_v_heads;
    for (std::size_t i = 0; i < glayers.size(); ++i) {
      gcache.qkv.push_back(
          _mc->make_shared_buffer((std::size_t)maxK * Cd * 2));
      gcache.gbuf.push_back(
          _mc->make_shared_buffer((std::size_t)maxK * Hv * 4));
      gcache.betabuf.push_back(
          _mc->make_shared_buffer((std::size_t)maxK * Hv * 4));
    }
  }

  // Round state: the next round's drafts, produced by THIS round's fused
  // verify at the accepted boundary. drafts[0] is always the verified next
  // token; d1/d2v are the depth-1/depth-2 MTP drafts (absent for round 1).
  std::int32_t d0 = first_token, d1 = -1, d2v = -1;
  bool have_d1 = false, have_d2 = false;
  long accepted = 0, rounds = 0;

  bool terminate = false;   // stop token or caller abort
  while (!terminate && (int)out_ids.size() < n_steps) {
    ++rounds;
    std::vector<std::int32_t> drafts;
    drafts.push_back(d0);
    if (have_d1) { drafts.push_back(d1); }
    if (d2 && have_d2) { drafts.push_back(d2v); }
    const int K = (int)drafts.size();

    // ONE fused forward: the main model verifies the K drafts AND the MTP head
    // (run 1x for depth-1, 2x chained for depth-2, in the SAME command buffer
    // on the main preds + hiddens) emits the next round's drafts. The paged KV
    // grows by K, GDN advances by K.
    SharedBuffer xK = _muxer->fetch_text(
        std::span<const std::int32_t>(drafts.data(), drafts.size()));
    if (xK.empty()) { break; }
    auto t_snap0 = mtp_clock::now();
    snapshot_gdn(/*save=*/true);
    auto t_ver0 = mtp_clock::now();
    prof_snap_ms += prof_ms(t_snap0, t_ver0);
    std::vector<std::int32_t> preds, mpreds, mpreds2;
    const bool vok = mtp_verify_chunk_(cid, xK, K, &preds, &mpreds,
                                       d2 ? &mpreds2 : nullptr, rope_delta,
                                       ctl.sampler, seed_slot0, &gcache);
    prof_verify_ms += prof_ms(t_ver0, mtp_clock::now());
    if (!vok ||
        (int)preds.size() != K || (int)mpreds.size() != K ||
        (d2 && (int)mpreds2.size() != K)) {
      snapshot_gdn(/*save=*/false);
      break;
    }
    // Accept the longest prefix matching the verifier's greedy preds (drafts[0]
    // is always correct -- it IS the prior round's verified next token).
    int j = 1;
    for (int i = 1; i < K; ++i) {
      if (drafts[(std::size_t)i] == preds[(std::size_t)(i - 1)]) { ++j; }
      else { break; }
    }
    // Stop-token cap: the serial loop never appends a stop token's KV nor emits
    // it, so accept only up to (excluding) the first stop within [0, j) and end
    // the decode. j becomes the stop's index, so the rollback below truncates
    // the stop (and any tail) out of the KV/GDN, leaving the context at the
    // last kept token -- ready for the caller's assistant_close commit.
    if (ctl.is_stop) {
      for (int i = 0; i < j; ++i) {
        if (ctl.is_stop(drafts[(std::size_t)i])) {
          j = i;
          terminate = true;
          if (ctl.hit_stop) { *ctl.hit_stop = true; }
          break;
        }
      }
    }
    // Keep at most the remaining budget. `keep` (<= j) drives both out_ids and
    // the KV/GDN rollback boundary, unifying partial-accept, stop, and budget.
    int keep = 0;
    for (int i = 0; i < j && (int)out_ids.size() < n_steps; ++i) {
      out_ids.push_back(drafts[(std::size_t)i]);
      ++keep;
    }
    if (keep > 0) { accepted += (keep - 1); }
    const bool budget_done = ((int)out_ids.size() >= n_steps);
    // Next round's drafts come straight out of THIS verify at the boundary --
    // only when we will actually continue (no stop, abort, or budget end).
    if (!terminate && !budget_done && keep > 0) {
      d0 = preds[(std::size_t)(keep - 1)];
      d1 = mpreds[(std::size_t)(keep - 1)];
      have_d1 = true;
      if (d2) { d2v = mpreds2[(std::size_t)(keep - 1)]; have_d2 = true; }
    }
    // Over-append rollback: the main KV + GDN advanced by K but we keep only
    // `keep`. Drop ONLY the rejected tokens' KV (kv_rollback K-keep) -- the kept
    // tokens' KV is valid (each was written with the same causal prefix a
    // keep-token forward sees). The GDN recurrent state has no positional
    // rewind, so restore the round's S0 snapshot and replay conv1d->qk_norm->
    // gdn_step for the kept tokens from the verify's cached inputs: bit-
    // identical to S_{p+keep}, GDN-only (no attention / projections / lm_head),
    // so far cheaper than the full main re-forward this used to do.
    if (keep < K) {
      auto t_rb0 = mtp_clock::now();
      _ctx->kv_rollback(cid, K - keep);
      snapshot_gdn(/*save=*/false);
      auto t_rr0 = mtp_clock::now();
      prof_snap_ms += prof_ms(t_rb0, t_rr0);
      if (keep > 0) {
        gdn_replay_(cid, keep, gcache);
        prof_rerun_ms += prof_ms(t_rr0, mtp_clock::now());
        ++prof_reruns;
      }
    }
    // Stream this round's kept tokens; an on_round false return aborts.
    if (ctl.on_round && keep > 0) {
      std::span<const std::int32_t> fresh(
          out_ids.data() + (out_ids.size() - (std::size_t)keep),
          (std::size_t)keep);
      if (!ctl.on_round(fresh)) { terminate = true; }
    }
  }
  if (accepted_out) { *accepted_out = accepted; }
  if (rounds_out) { *rounds_out = rounds; }
  if (mtp_prof) {
    const double tot = prof_verify_ms + prof_rerun_ms + prof_snap_ms;
    std::printf(
        "[mtp_prof] depth=%d tok=%d rounds=%ld reruns=%ld (%.0f%% of rounds) "
        "| verify=%.1fms (%.2f/round) rerun=%.1fms (%.2f/rerun) snap=%.1fms "
        "| sum=%.1fms\n",
        (draft_len >= 2 ? 2 : 1), (int)out_ids.size(), rounds, prof_reruns,
        rounds > 0 ? 100.0 * (double)prof_reruns / (double)rounds : 0.0,
        prof_verify_ms, rounds > 0 ? prof_verify_ms / (double)rounds : 0.0,
        prof_rerun_ms,
        prof_reruns > 0 ? prof_rerun_ms / (double)prof_reruns : 0.0,
        prof_snap_ms, tot);
  }
  return true;
}

}  // namespace vpipe::genai
