#include "generative-models/qwen3/metal-qwen-vision.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/model-loader.h"
#include "generative-models/shared/gguf-file.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/compute-library.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/perf-event.h"
#include "common/perf-scope.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {
// C++ mirror of mlx::steel::AttnParams (steel/attn/params.h) -- the param
// block the vendored steel flash-attention kernel reads. Identical layout
// to the LM's copy in metal-llama-model.cc.
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
// Qwen2/3-VL smart_resize: edges -> multiples of `factor`, pixels in
// [min_px, max_px], aspect preserved.
void
smart_resize_(int h, int w, int factor, int min_px, int max_px,
              int* oh, int* ow)
{
  int hb = std::max(factor, (int)std::round((double)h / factor) * factor);
  int wb = std::max(factor, (int)std::round((double)w / factor) * factor);
  if (hb * wb > max_px) {
    double beta = std::sqrt(((double)h * w) / max_px);
    hb = std::max(factor, (int)std::floor(h / beta / factor) * factor);
    wb = std::max(factor, (int)std::floor(w / beta / factor) * factor);
  } else if (hb * wb < min_px) {
    double beta = std::sqrt((double)min_px / ((double)h * w));
    hb = (int)std::ceil(h * beta / factor) * factor;
    wb = (int)std::ceil(w * beta / factor) * factor;
  }
  *oh = hb;
  *ow = wb;
}
// Bilinear resize + per-channel normalize, U8 [3,H,W] -> f32 [3,th,tw].
void
preprocess_(const std::uint8_t* rgb, int H, int W, int th, int tw,
            const float mean[3], const float stdv[3], float* out)
{
  if (H <= 0 || W <= 0) { return; }
  const double sy = (double)H / th, sx = (double)W / tw;
  const int pin = H * W, pout = th * tw;
  for (int c = 0; c < 3; ++c) {
    const std::uint8_t* src = rgb + (std::size_t)c * pin;
    float* dst = out + (std::size_t)c * pout;
    const float inv = 1.0f / (255.0f * stdv[c]);
    const float bias = -mean[c] / stdv[c];
    for (int yy = 0; yy < th; ++yy) {
      double y = (yy + 0.5) * sy - 0.5;
      y = std::min(std::max(y, 0.0), (double)(H - 1));
      int y0 = (int)std::floor(y), y1 = std::min(y0 + 1, H - 1);
      float dy = (float)(y - y0);
      for (int xx = 0; xx < tw; ++xx) {
        double x = (xx + 0.5) * sx - 0.5;
        x = std::min(std::max(x, 0.0), (double)(W - 1));
        int x0 = (int)std::floor(x), x1 = std::min(x0 + 1, W - 1);
        float dx = (float)(x - x0);
        float v00 = src[y0 * W + x0], v01 = src[y0 * W + x1];
        float v10 = src[y1 * W + x0], v11 = src[y1 * W + x1];
        float v0 = v00 * (1 - dx) + v01 * dx;
        float v1 = v10 * (1 - dx) + v11 * dx;
        dst[yy * tw + xx] = (v0 * (1 - dy) + v1 * dy) * inv + bias;
      }
    }
  }
}
}  // namespace

std::unique_ptr<MetalQwenVisionEncoder>
MetalQwenVisionEncoder::load(const std::string& model_dir,
                             metal_compute::MetalCompute* mc, const Config& cfg)
{
  if (mc == nullptr || !mc->valid()) { return nullptr; }
  // Weights come either from the safetensors model dir (HF tensor names) or,
  // when cfg.gguf_mmproj is set, from an mmproj-*.gguf (llama.cpp CLIP names,
  // BF16/F32 -- same values, just a rename + a split patch-embed conv).
  const bool gguf = !cfg.gguf_mmproj.empty();
  std::optional<MetalLlamaWeights> wts;
  std::optional<GgufFile> gg;
  if (gguf) {
    gg = GgufFile::open(cfg.gguf_mmproj);
    if (!gg) { return nullptr; }
  } else {
    wts = MetalLlamaWeights::open_model(model_dir);
    if (!wts) { return nullptr; }
  }

  auto m = std::unique_ptr<MetalQwenVisionEncoder>(new MetalQwenVisionEncoder());
  m->_cfg = cfg;
  m->_mc = mc;
  m->_feat_dim = cfg.temporal_patch * cfg.patch_size * cfg.patch_size * 3;

  m->_lib_gemm = mc->load_library("dense_gemm");
  m->_lib_vis = mc->load_library("qwen3_5_vision");
  m->_lib_elt = mc->load_library("llm_elementwise");
  m->_lib_sdpa = mc->load_library("sdpa");
  // Steel flash-attention (MMA) library, shared with the LM prefill path.
  // Optional: if it (or the bd64 entry point) is unavailable we fall back
  // to the scalar sdpa_full_f16 in encode().
  m->_lib_attn = mc->load_library("attn_steel");
  m->_attn_params = mc->make_shared_buffer(sizeof(SteelAttnParams));
  m->_fn_gemm = m->_lib_gemm.function("dense_gemm_bias_f16");
  m->_fn_gemm_t = m->_lib_gemm.function("dense_gemm_t_f16");
  m->_fn_ln = m->_lib_vis.function("layer_norm_bias_f16");
  m->_fn_gelu_tanh = m->_lib_vis.function("gelu_tanh_f16");
  m->_fn_gelu_erf = m->_lib_vis.function("gelu_erf_f16");
  m->_fn_vrope = m->_lib_vis.function("vision_rope_f16");
  m->_fn_sdpa_full = m->_lib_sdpa.function("sdpa_full_f16");
  m->_fn_head_slice = m->_lib_elt.function("head_slice_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_residual = m->_lib_elt.function("residual_add_f16");
  // M5-only matrix-core dense GEMM (matmul2d/NAX) -- the same NAX path the
  // LM prefill uses. Gated on supports_matrix_cores() (GPUFamilyApple10+) so
  // M4 / older GPUs never load it and keep the scalar dense_gemm (byte-
  // identical, performance unchanged). VPIPE_QWEN_VISION_NO_MMA2=1 forces the
  // scalar path even on M5 (A/B + safety).
  if (mc->supports_matrix_cores() &&
      std::getenv("VPIPE_QWEN_VISION_NO_MMA2") == nullptr) {
    m->_lib_dense_mma = mc->load_library("dense_gemm_mma");
    m->_fn_dense_mma = m->_lib_dense_mma.function("dense_gemm_mma_t_n128_f16");
    m->_fn_dense_mma_deep =
        m->_lib_dense_mma.function("dense_gemm_mma_t_n128x256_f16");
    m->_fn_bias_add = m->_lib_elt.function("bias_add_rows_f16");
    m->_use_mma2 = m->_fn_dense_mma.valid() && m->_fn_dense_mma_deep.valid() &&
                   m->_fn_bias_add.valid();
    // OPT-IN matrix-core flash attention (head_dim 64, transpose-free). The
    // steel simdgroup_matrix flash is the faster DEFAULT on M5 at full clocks;
    // this matmul2d path is ~16-26% slower at >=256 tokens (overhead-bound at
    // hd=64: tg staging + barriers + tg-resident softmax). Kept opt-in
    // (VPIPE_QWEN_VISION_MMA2_ATTN=1) for future retuning. NOTE: the throttled
    // bench that originally shipped this as default was a thermal artifact.
    if (cfg.head_dim() == 64 &&
        std::getenv("VPIPE_QWEN_VISION_MMA2_ATTN") != nullptr) {
      m->_lib_sdpa_mma = mc->load_library("sdpa_mma");
      m->_fn_sdpa_mma_d64 = m->_lib_sdpa_mma.function("sdpa_full_mma2_d64_f16");
      m->_use_mma2_attn = m->_fn_sdpa_mma_d64.valid();
    }
    // DEFAULT on M5: MLX's matrix-core (NAX) steel attention -- the kernel MLX
    // itself dispatches on M5 (bq=64/bk=32, register-resident softmax). ~1.2-
    // 1.7x faster than the ALU steel (takes the tower PAST the external mlx-vlm
    // bar at every resolution, 0.72-0.81x) and token-correct vs the ALU steel
    // (0.6% rms, verified by metal_vision_nax_matches_steel) -- PROVIDED the
    // NAX kernels are built with deployment target 26.2 (see metal-compute
    // CMakeLists / VPIPE_NAX_MIN_OS; without it the matmul2d miscompiles ~400%
    // off). VPIPE_QWEN_VISION_NO_NAX_ATTN=1 falls back to the ALU steel.
    if (cfg.head_dim() == 64 &&
        std::getenv("VPIPE_QWEN_VISION_NO_NAX_ATTN") == nullptr) {
      m->_lib_attn_nax = mc->load_library("attn_steel_nax");
      m->_use_attn_nax = m->_lib_attn_nax.valid();
    }
  }
  if (!m->_fn_gemm.valid() || !m->_fn_ln.valid() || !m->_fn_gelu_tanh.valid() ||
      !m->_fn_gelu_erf.valid() || !m->_fn_vrope.valid() ||
      !m->_fn_sdpa_full.valid() || !m->_fn_head_slice.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_residual.valid()) {
    return nullptr;
  }

  // HF (safetensors) tensor name -> llama.cpp CLIP name. Block / merger /
  // patch-embed-bias / pos-embed are plain renames; patch_embed.proj.weight
  // is handled separately (split + transposed) below.
  auto clip_name = [](const std::string& hf) -> std::string {
    const std::string s = hf.substr(std::string("vision_tower.").size());
    if (s.rfind("blocks.", 0) == 0) {
      const std::size_t d = s.find('.', 7);
      const std::string n = s.substr(7, d - 7), rest = s.substr(d + 1);
      std::string c;
      if (rest.rfind("norm1.", 0) == 0)            c = "ln1." + rest.substr(6);
      else if (rest.rfind("norm2.", 0) == 0)       c = "ln2." + rest.substr(6);
      else if (rest.rfind("attn.qkv.", 0) == 0)    c = "attn_qkv." + rest.substr(9);
      else if (rest.rfind("attn.proj.", 0) == 0)   c = "attn_out." + rest.substr(10);
      else if (rest.rfind("mlp.linear_fc1.", 0) == 0) c = "ffn_up." + rest.substr(15);
      else if (rest.rfind("mlp.linear_fc2.", 0) == 0) c = "ffn_down." + rest.substr(15);
      return "v.blk." + n + "." + c;
    }
    if (s == "patch_embed.proj.bias") { return "v.patch_embd.bias"; }
    if (s == "pos_embed.weight")      { return "v.position_embd.weight"; }
    if (s.rfind("merger.norm.", 0) == 0)       { return "v.post_ln." + s.substr(12); }
    if (s.rfind("merger.linear_fc1.", 0) == 0) { return "mm.0." + s.substr(18); }
    if (s.rfind("merger.linear_fc2.", 0) == 0) { return "mm.2." + s.substr(18); }
    return s;
  };
  // Narrow a GGUF (BF16/F32/F16) tensor into a fresh f16 SharedBuffer.
  auto gg_f16 = [&](const GgufFile::Tensor* t) -> SharedBuffer {
    if (t == nullptr) { return {}; }
    std::size_t n = 1;
    for (auto d : t->dims) { n *= (std::size_t)d; }
    SharedBuffer out = mc->make_shared_buffer(n * 2);
    auto* o = static_cast<_Float16*>(out.contents());
    if (t->type == GgufFile::kBF16) {
      const auto* s = reinterpret_cast<const std::uint16_t*>(t->data);
      for (std::size_t i = 0; i < n; ++i) { o[i] = (_Float16)bf16_to_f32_(s[i]); }
    } else if (t->type == GgufFile::kF32) {
      const auto* s = reinterpret_cast<const float*>(t->data);
      for (std::size_t i = 0; i < n; ++i) { o[i] = (_Float16)s[i]; }
    } else if (t->type == GgufFile::kF16) {
      std::memcpy(o, t->data, n * 2);
    } else {
      return {};
    }
    return out;
  };
  auto to_f16 = [&](const std::string& name) -> SharedBuffer {
    if (gguf) { return gg_f16(gg->tensor(clip_name(name))); }
    const auto* info = wts->info(name);
    if (info == nullptr) { return {}; }
    if (info->dtype == "F16") { return wts->load(name, mc); }
    SharedBuffer raw = wts->load(name, mc);
    if (raw.empty()) { return {}; }
    const std::size_t n = numel_(info->shape);
    SharedBuffer out = mc->make_shared_buffer(n * 2);
    auto* o = static_cast<_Float16*>(out.contents());
    if (info->dtype == "BF16") {
      const auto* s = static_cast<const std::uint16_t*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = (_Float16)bf16_to_f32_(s[i]); }
    } else if (info->dtype == "F32") {
      const auto* s = static_cast<const float*>(raw.contents());
      for (std::size_t i = 0; i < n; ++i) { o[i] = (_Float16)s[i]; }
    } else {
      return {};
    }
    return out;
  };
  // Patch-embed conv weight. Safetensors: one [hidden, kT,kH,kW,in_ch] tensor
  // (channels-last per patch). GGUF CLIP splits it per temporal frame into
  // v.patch_embd.weight[.t] each [hidden, in_ch,kH,kW] (channels-first), so
  // rebuild [hidden, (kT,kH,kW,in_ch)] by transposing each frame's
  // (in_ch, kH*kW) -> (kH*kW, in_ch) and concatenating the frames.
  auto patch_embed_gguf = [&]() -> SharedBuffer {
    const int outc = cfg.hidden, kHW = cfg.patch_size * cfg.patch_size;
    const int inch = 3, per_frame = kHW * inch;
    const int feat = cfg.temporal_patch * per_frame;
    SharedBuffer w = mc->make_shared_buffer((std::size_t)outc * feat * 2);
    auto* o = static_cast<_Float16*>(w.contents());
    for (int t = 0; t < cfg.temporal_patch; ++t) {
      const std::string nm = (t == 0)
          ? std::string("v.patch_embd.weight")
          : "v.patch_embd.weight." + std::to_string(t);
      const GgufFile::Tensor* tn = gg->tensor(nm);
      if (tn == nullptr) { return {}; }
      const float* s = reinterpret_cast<const float*>(tn->data);  // F32
      if (tn->type != GgufFile::kF32) { return {}; }
      for (int oi = 0; oi < outc; ++oi) {
        for (int c = 0; c < inch; ++c) {
          for (int hw = 0; hw < kHW; ++hw) {
            o[(std::size_t)oi * feat + t * per_frame + hw * inch + c] =
                (_Float16)s[((std::size_t)oi * inch + c) * kHW + hw];
          }
        }
      }
    }
    return w;
  };

  const std::string r = "vision_tower.";
  m->_pe_w = gguf ? patch_embed_gguf() : to_f16(r + "patch_embed.proj.weight");
  m->_pe_b = to_f16(r + "patch_embed.proj.bias");
  m->_pos_w = to_f16(r + "pos_embed.weight");
  bool ok = !m->_pe_w.empty() && !m->_pe_b.empty() && !m->_pos_w.empty();
  m->_blocks.resize(cfg.depth);
  for (int b = 0; b < cfg.depth; ++b) {
    const std::string p = r + "blocks." + std::to_string(b) + ".";
    Block& blk = m->_blocks[b];
    blk.n1w = to_f16(p + "norm1.weight");
    blk.n1b = to_f16(p + "norm1.bias");
    blk.qkvw = to_f16(p + "attn.qkv.weight");
    blk.qkvb = to_f16(p + "attn.qkv.bias");
    blk.ow = to_f16(p + "attn.proj.weight");
    blk.ob = to_f16(p + "attn.proj.bias");
    blk.n2w = to_f16(p + "norm2.weight");
    blk.n2b = to_f16(p + "norm2.bias");
    blk.fc1w = to_f16(p + "mlp.linear_fc1.weight");
    blk.fc1b = to_f16(p + "mlp.linear_fc1.bias");
    blk.fc2w = to_f16(p + "mlp.linear_fc2.weight");
    blk.fc2b = to_f16(p + "mlp.linear_fc2.bias");
    ok = ok && !blk.n1w.empty() && !blk.qkvw.empty() && !blk.qkvb.empty() &&
         !blk.ow.empty() && !blk.fc1w.empty() && !blk.fc2w.empty();
  }
  m->_mnw = to_f16(r + "merger.norm.weight");
  m->_mnb = to_f16(r + "merger.norm.bias");
  m->_mfc1w = to_f16(r + "merger.linear_fc1.weight");
  m->_mfc1b = to_f16(r + "merger.linear_fc1.bias");
  m->_mfc2w = to_f16(r + "merger.linear_fc2.weight");
  m->_mfc2b = to_f16(r + "merger.linear_fc2.bias");
  ok = ok && !m->_mnw.empty() && !m->_mfc1w.empty() && !m->_mfc2w.empty();
  if (!ok) { return nullptr; }

  const int rope_dim = cfg.head_dim() / 2;
  for (int i = 0; i < rope_dim; i += 2) {
    m->_rope_inv_freq.push_back(
        1.0f / std::pow(10000.0f, (float)i / (float)rope_dim));
  }
  return m;
}

MetalQwenVisionEncoder::Config
MetalQwenVisionEncoder::config_from(const ModelConfig& c)
{
  Config m;
  const auto& v = c.vision;
  m.depth          = v.depth;
  m.hidden         = v.hidden_size;
  m.n_heads        = v.num_heads;
  m.patch_size     = v.patch_size;
  m.spatial_merge  = v.spatial_merge_size;
  m.temporal_patch = v.temporal_patch_size;
  m.out_hidden     = v.out_hidden_size;
  m.num_pos_embed  = v.num_position_embeddings;
  m.intermediate   = v.intermediate_size;
  m.gguf_mmproj    = v.mmproj_path;
  for (int i = 0; i < 3; ++i) {
    m.image_mean[i] = v.image_mean[i];
    m.image_std[i]  = v.image_std[i];
  }
  return m;
}

MetalQwenVisionEncoder::Result
MetalQwenVisionEncoder::encode(const std::uint8_t* rgb, int H, int W)
{
  // Record this GPU ViT pass on the profiler's LLM lane (vision-tower
  // category) so it's categorized like the MLX and CoreML towers. When
  // no CoreML model is configured the vision tower runs here on the GPU;
  // without this it was invisible to the profiler. One image per call.
  PerfAuxScope _perf(_session, kPerfLaneLLM, kGvidLlmVision,
                     kPerfLlmVisionBegin, 1);
  using Clock = std::chrono::steady_clock;
  const bool profile = std::getenv("VPIPE_VISION_PROFILE") != nullptr;
  const auto t_enter = Clock::now();
  const Config& c = _cfg;
  const int P = c.patch_size, S = c.spatial_merge, T = c.temporal_patch;
  const int hidden = c.hidden, heads = c.n_heads, hd = c.head_dim();
  const int inter = c.intermediate, outh = c.out_hidden;
  // The patch merger's hidden width is S*S*hidden (the flattened
  // spatial-merge window), NOT the ViT MLP intermediate. They coincide
  // only when S*S*hidden == intermediate (true for the 4B: 4*1024 ==
  // 4096, false for the 9B: 4*1152 == 4608 != 4304). fc1 is [mdim,mdim],
  // fc2 is [out_hidden,mdim].
  const int mdim = S * S * hidden;
  const float eps = c.ln_eps;
  const float scale = 1.0f / std::sqrt((float)hd);
  Result res;

  const int factor = P * S;
  int th, tw;
  smart_resize_(H, W, factor, 56 * 56, 28 * 28 * 1280, &th, &tw);
  const int grid_h = th / P, grid_w = tw / P;
  const int mh = grid_h / S, mw = grid_w / S;
  const int n_patches = grid_h * grid_w;
  const int n_im = mh * mw;
  res.n_tokens = n_im;
  res.grid_h = grid_h;
  res.grid_w = grid_w;

  std::vector<float> px((std::size_t)3 * th * tw);
  preprocess_(rgb, H, W, th, tw, c.image_mean, c.image_std, px.data());

  // Patchify in merger order ([mh,mw,S,S]) with channels-last [T,P,P,c]
  // inner layout, still image tiled across the temporal axis.
  std::vector<_Float16> feat((std::size_t)n_patches * _feat_dim);
  const int plane = th * tw;
  for (int m = 0; m < n_patches; ++m) {
    int bi = m / (mw * S * S), rem = m % (mw * S * S);
    int bj = rem / (S * S), rem2 = rem % (S * S);
    int ii = rem2 / S, jj = rem2 % S;
    const int gi = bi * S + ii, gj = bj * S + jj;
    _Float16* fp = feat.data() + (std::size_t)m * _feat_dim;
    int o = 0;
    for (int t = 0; t < T; ++t) {
      for (int ph = 0; ph < P; ++ph) {
        for (int pw = 0; pw < P; ++pw) {
          const int yy = gi * P + ph, xx = gj * P + pw;
          for (int ch = 0; ch < 3; ++ch) {
            fp[o++] = (_Float16)px[(std::size_t)ch * plane + yy * tw + xx];
          }
        }
      }
    }
  }

  // Bilinear pos-embed in merger order (host) from the [G*G,hidden] table.
  const int G = (int)std::round(std::sqrt((double)c.num_pos_embed));
  const auto* pos_tab = static_cast<const _Float16*>(_pos_w.contents());
  std::vector<_Float16> pos((std::size_t)n_patches * hidden);
  // 2D-RoPE cos/sin tables in merger order (host).
  std::vector<_Float16> cosb((std::size_t)n_patches * hd),
      sinb((std::size_t)n_patches * hd);
  const int hd4 = hd / 4;
  for (int m = 0; m < n_patches; ++m) {
    int bi = m / (mw * S * S), rem = m % (mw * S * S);
    int bj = rem / (S * S), rem2 = rem % (S * S);
    int ii = rem2 / S, jj = rem2 % S;
    const int row = bi * S + ii, col = bj * S + jj;
    // pos embed
    const float hi = (grid_h > 1) ? (float)row * (G - 1) / (grid_h - 1) : 0.0f;
    const float wi = (grid_w > 1) ? (float)col * (G - 1) / (grid_w - 1) : 0.0f;
    int hf = (int)hi, hc = std::min(hf + 1, G - 1);
    int wf = (int)wi, wc = std::min(wf + 1, G - 1);
    float dh = hi - hf, dw = wi - wf;
    const int cidx[4] = {hf * G + wf, hf * G + wc, hc * G + wf, hc * G + wc};
    const float cw[4] = {(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw),
                         dh * dw};
    _Float16* pp = pos.data() + (std::size_t)m * hidden;
    for (int d = 0; d < hidden; ++d) {
      float acc = 0.0f;
      for (int q = 0; q < 4; ++q) {
        acc += cw[q] * (float)pos_tab[(std::size_t)cidx[q] * hidden + d];
      }
      pp[d] = (_Float16)acc;
    }
    // rope
    _Float16* cp = cosb.data() + (std::size_t)m * hd;
    _Float16* sp = sinb.data() + (std::size_t)m * hd;
    for (int k = 0; k < hd4; ++k) {
      const float fr = row * _rope_inv_freq[k];
      const float fc = col * _rope_inv_freq[k];
      const float vals[4] = {fr, fc, fr, fc};
      for (int q = 0; q < 4; ++q) {
        cp[q * hd4 + k] = (_Float16)std::cos(vals[q]);
        sp[q * hd4 + k] = (_Float16)std::sin(vals[q]);
      }
    }
  }

  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  SharedBuffer featb = buf((std::size_t)n_patches * _feat_dim);
  SharedBuffer posb = buf((std::size_t)n_patches * hidden);
  SharedBuffer cosbuf = buf((std::size_t)n_patches * hd);
  SharedBuffer sinbuf = buf((std::size_t)n_patches * hd);
  std::memcpy(featb.contents(), feat.data(), feat.size() * 2);
  std::memcpy(posb.contents(), pos.data(), pos.size() * 2);
  std::memcpy(cosbuf.contents(), cosb.data(), cosb.size() * 2);
  std::memcpy(sinbuf.contents(), sinb.data(), sinb.size() * 2);

  SharedBuffer x = buf((std::size_t)n_patches * hidden);
  SharedBuffer n1 = buf((std::size_t)n_patches * hidden);
  SharedBuffer qkv = buf((std::size_t)n_patches * 3 * hidden);
  SharedBuffer q3 = buf((std::size_t)n_patches * hidden),
               k3 = buf((std::size_t)n_patches * hidden),
               v3 = buf((std::size_t)n_patches * hidden);
  // qr/kr (post-rope) and v3 feed the matrix-core attn directly from the
  // [n, hidden] layout (it stages each head's strided tile into threadgroup
  // memory, bounds-checked -- no padding). Only the steel/scalar fallback
  // needs the head-major transposes into qt/kt/vt.
  SharedBuffer qr = buf((std::size_t)n_patches * hidden),
               kr = buf((std::size_t)n_patches * hidden);
  SharedBuffer qt = buf((std::size_t)n_patches * hidden),
               kt = buf((std::size_t)n_patches * hidden),
               vt = buf((std::size_t)n_patches * hidden);
  SharedBuffer atb = buf((std::size_t)n_patches * hidden),
               att = buf((std::size_t)n_patches * hidden);
  SharedBuffer proj = buf((std::size_t)n_patches * hidden);
  SharedBuffer hbuf = buf((std::size_t)n_patches * inter),
               h2 = buf((std::size_t)n_patches * inter);
  SharedBuffer out2 = buf((std::size_t)n_patches * hidden);
  SharedBuffer mg = buf((std::size_t)n_im * mdim);
  SharedBuffer emb = buf((std::size_t)n_im * outh);

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    const bool use_mma_gemm = _fn_gemm_t.valid();
    auto gemm = [&](const SharedBuffer& xin, const SharedBuffer& w,
                    const SharedBuffer& bias, const SharedBuffer& y, int M,
                    int N, int Kk) {
      if (_use_mma2) {
        // M5 matrix-core matmul2d GEMM y = x @ w^T (no bias), then fold the
        // linear bias over the rows. Same NAX path as the LM prefill; the
        // matmul2d tensor extents clamp M/N tails so M=n_patches and N need
        // not be tile multiples. 128x256 tile for deep K (K>=6144).
        const bool deep = (Kk >= 6144);
        const int BN = deep ? 256 : 128;
        enc.set_function(deep ? _fn_dense_mma_deep : _fn_dense_mma);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, w);
        enc.set_buffer(2, w);     // bias slot unused (has_bias=0)
        enc.set_buffer(3, y);
        enc.set_constant(4, Kk);
        enc.set_constant(5, N);
        enc.set_constant(6, M);
        enc.set_constant(7, 0);
        enc.dispatch({(unsigned)(((N + BN - 1) / BN) * 256),
                      (unsigned)((M + 127) / 128), 1}, {256, 1, 1});
        enc.set_function(_fn_bias_add);
        enc.set_buffer(0, y);
        enc.set_buffer(1, bias);
        enc.set_constant(2, N);
        enc.set_constant(3, M * N);
        enc.dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
      } else if (use_mma_gemm) {
        // Steel MMA GEMM y = x @ w^T + bias (bias folded in the kernel).
        enc.set_function(_fn_gemm_t);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, w);
        enc.set_buffer(2, bias);
        enc.set_buffer(3, y);
        enc.set_constant(4, Kk);
        enc.set_constant(5, N);
        enc.set_constant(6, M);
        enc.set_constant(7, 1);   // has_bias (vision linears all have bias)
        enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                      (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
      } else {
        enc.set_function(_fn_gemm);
        enc.set_buffer(0, xin);
        enc.set_buffer(1, w);
        enc.set_buffer(2, bias);
        enc.set_buffer(3, y);
        enc.set_constant(4, M);
        enc.set_constant(5, N);
        enc.set_constant(6, Kk);
        enc.set_constant(7, 1);
        const unsigned gx = (unsigned)(((N + 15) / 16) * 16);
        const unsigned gy = (unsigned)(((M + 15) / 16) * 16);
        enc.dispatch({gx, gy, 1}, {16, 16, 1});
      }
    };
    auto ln = [&](const SharedBuffer& xin, const SharedBuffer& w,
                  const SharedBuffer& b, const SharedBuffer& y, int R, int Hd) {
      enc.set_function(_fn_ln);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, b);
      enc.set_buffer(3, y);
      enc.set_constant(4, Hd);
      enc.set_constant(5, eps);
      enc.dispatch({256, (unsigned)R, 1}, {256, 1, 1});
    };
    auto gelu = [&](metal_compute::ComputeFunction& fn, const SharedBuffer& xin,
                    const SharedBuffer& y, int nn) {
      enc.set_function(fn);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, y);
      enc.set_constant(2, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& in, const SharedBuffer& out, int Hh,
                      int Sd, int Wd, int off) {
      enc.set_function(_fn_head_slice);
      enc.set_buffer(0, in);
      enc.set_buffer(1, out);
      enc.set_constant(2, Hh);
      enc.set_constant(3, Sd);
      enc.set_constant(4, Wd);
      enc.set_constant(5, off);
      enc.dispatch({(unsigned)(Hh * Wd), 1, 1}, {256, 1, 1});
    };
    auto vrope = [&](const SharedBuffer& in, const SharedBuffer& out) {
      enc.set_function(_fn_vrope);
      enc.set_buffer(0, in);
      enc.set_buffer(1, cosbuf);
      enc.set_buffer(2, sinbuf);
      enc.set_buffer(3, out);
      enc.set_constant(4, heads);
      enc.set_constant(5, hd);
      enc.dispatch({(unsigned)(n_patches * heads * hd), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in);
      enc.set_buffer(1, out);
      enc.set_constant(2, A);
      enc.set_constant(3, Bd);
      enc.set_constant(4, hd);
      enc.dispatch({(unsigned)hd, (unsigned)Bd, (unsigned)A}, {(unsigned)hd, 1, 1});
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

    // Patch embed + position embed.
    gemm(featb, _pe_w, _pe_b, x, n_patches, hidden, _feat_dim);
    residual(x, posb, x, n_patches * hidden);

    // Steel flash-attention (MMA) setup -- shared across all blocks (same
    // n_patches/heads/hd). Non-causal (do_causal=false) bidirectional ViT
    // attention. ~10x over the scalar sdpa_full_f16 at large grids; falls
    // back to scalar when the bd64 entry point / 64-wide head isn't there.
    // Prefer MLX's matrix-core (NAX) steel attention on M5 (attn_steel_nax,
    // bq=64/bk=32, register-resident softmax) over the ALU steel (attn_steel,
    // bq=32/bk=16). Same AttnParams + func-const contract; only the tiles and
    // the kernel differ.
    const bool attn_nax = _use_attn_nax && _lib_attn_nax.valid();
    const int a_bq = attn_nax ? 64 : 32;
    const int a_bk = attn_nax ? 32 : 16;
    metal_compute::ComputeFunction fn_steel;
    if ((attn_nax || _lib_attn.valid()) && hd == 64 && !_attn_params.empty()) {
      auto* p = static_cast<SteelAttnParams*>(_attn_params.contents());
      p->B = 1; p->H = heads; p->D = hd; p->qL = n_patches; p->kL = n_patches;
      p->gqa_factor = 1; p->scale = scale;
      p->NQ = (n_patches + a_bq - 1) / a_bq;
      p->NK = (n_patches + a_bk - 1) / a_bk;
      p->NQ_aligned = n_patches / a_bq; p->NK_aligned = n_patches / a_bk;
      p->qL_rem = n_patches - p->NQ_aligned * a_bq;
      p->kL_rem = n_patches - p->NK_aligned * a_bk;
      p->qL_off = 0;
      p->Q_strides[0] = (std::int64_t)heads * n_patches * hd;
      p->Q_strides[1] = (std::int64_t)n_patches * hd; p->Q_strides[2] = hd;
      p->K_strides[0] = p->Q_strides[0];
      p->K_strides[1] = p->Q_strides[1]; p->K_strides[2] = hd;
      p->V_strides[0] = p->Q_strides[0];
      p->V_strides[1] = p->Q_strides[1]; p->V_strides[2] = hd;
      p->O_strides[0] = p->Q_strides[0];
      p->O_strides[1] = p->Q_strides[1]; p->O_strides[2] = hd;
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (n_patches % a_bq) == 0)
          .set_bool(201, (n_patches % a_bk) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      fn_steel = attn_nax
          ? _lib_attn_nax.function("attn_steel_nax_h_bd64", fc)
          : _lib_attn.function("attn_steel_h_bd64", fc);
    }
    const bool use_steel = fn_steel.valid();
    // Perf probe only: skip the attention dispatch to isolate its cost (the
    // downstream proj/MLP GPU work is unchanged, so the wall-time delta is the
    // attention kernel). NOT a correct forward -- profiling use only.
    const bool skip_attn = std::getenv("VPIPE_VISION_SKIP_ATTN") != nullptr;
    const unsigned nqb = (unsigned)((n_patches + a_bq - 1) / a_bq);

    for (int b = 0; b < c.depth; ++b) {
      Block& blk = _blocks[b];
      ln(x, blk.n1w, blk.n1b, n1, n_patches, hidden);
      gemm(n1, blk.qkvw, blk.qkvb, qkv, n_patches, 3 * hidden, hidden);
      hslice(qkv, q3, n_patches, 3 * hidden, hidden, 0);
      hslice(qkv, k3, n_patches, 3 * hidden, hidden, hidden);
      hslice(qkv, v3, n_patches, 3 * hidden, hidden, 2 * hidden);
      vrope(q3, qr);
      vrope(k3, kr);
      // The matrix-core attn reads qr/kr/v3 straight from [n,heads,hd] and
      // writes att in the same layout -- only the steel/scalar fallback needs
      // the head-major transposes.
      if (!_use_mma2_attn) {
        transpose(qr, qt, n_patches, heads);   // [n,heads,hd]->[heads,n,hd]
        transpose(kr, kt, n_patches, heads);
        transpose(v3, vt, n_patches, heads);
      }
      if (skip_attn) {
        // probe: leave atb/att untouched (no attention compute)
      } else if (_use_mma2_attn) {
        // Matrix-core (matmul2d) non-causal flash attention, head_dim 64.
        // Reads q/k/v in [n,heads,hd] and writes att in [n,heads,hd] (no
        // transposes); full MHA (Hkv==Hq).
        enc.set_function(_fn_sdpa_mma_d64);
        enc.set_buffer(0, qr);
        enc.set_buffer(1, kr);
        enc.set_buffer(2, v3);
        enc.set_buffer(3, att);
        enc.set_constant(4, scale);
        enc.set_constant(5, n_patches);   // T_kv
        enc.set_constant(6, hd);          // D (== 64)
        enc.set_constant(7, heads);       // Hq
        enc.set_constant(8, heads);       // Hkv (full MHA)
        enc.set_constant(9, n_patches);   // n_q
        enc.set_constant(10, 0);          // q_offset
        enc.set_constant(11, n_patches);  // (unused)
        enc.dispatch({128, (unsigned)heads, (unsigned)((n_patches + 31) / 32)},
                     {128, 1, 1});
      } else if (use_steel) {
        // Steel MMA flash attention: Q/K/V/O all [heads, n_patches, hd].
        enc.set_function(fn_steel);
        enc.set_buffer(0, qt);
        enc.set_buffer(1, kt);
        enc.set_buffer(2, vt);
        enc.set_buffer(3, atb);
        enc.set_buffer(4, _attn_params);
        enc.dispatch({32 * nqb, 4 * (unsigned)heads, 1}, {32, 4, 1});
      } else {
        enc.set_function(_fn_sdpa_full);
        enc.set_buffer(0, qt);
        enc.set_buffer(1, kt);
        enc.set_buffer(2, vt);
        enc.set_buffer(3, atb);
        enc.set_constant(4, scale);
        enc.set_constant(5, n_patches);   // T_kv
        enc.set_constant(6, hd);
        enc.set_constant(7, heads);
        enc.set_constant(8, heads);
        enc.set_constant(9, n_patches);   // n_q
        enc.set_constant(10, n_patches);  // kv_stride
        enc.dispatch({32, (unsigned)heads, (unsigned)n_patches}, {32, 1, 1});
      }
      // Fallback attn output is head-major [heads,n,hd]; transpose back to
      // [n,heads,hd] for proj. The matrix-core attn already wrote att directly.
      if (!_use_mma2_attn && !skip_attn) {
        transpose(atb, att, heads, n_patches);
      }
      gemm(att, blk.ow, blk.ob, proj, n_patches, hidden, hidden);
      residual(x, proj, x, n_patches * hidden);
      ln(x, blk.n2w, blk.n2b, n1, n_patches, hidden);
      gemm(n1, blk.fc1w, blk.fc1b, hbuf, n_patches, inter, hidden);
      gelu(_fn_gelu_tanh, hbuf, h2, n_patches * inter);
      gemm(h2, blk.fc2w, blk.fc2b, out2, n_patches, hidden, inter);
      residual(x, out2, x, n_patches * hidden);
    }

    // Patch merger: per-patch LN, then reshape [n_im, S^2*hidden] -> fc1
    // -> GELU(erf) -> fc2.
    ln(x, _mnw, _mnb, n1, n_patches, hidden);
    // Merger fc1 [mdim,mdim] -> GELU -> fc2 [out_hidden,mdim], where the
    // input rows are the [n_im, S*S*hidden] flattened windows (mdim wide).
    gemm(n1, _mfc1w, _mfc1b, mg, n_im, mdim, mdim);
    gelu(_fn_gelu_erf, mg, mg, n_im * mdim);
    gemm(mg, _mfc2w, _mfc2b, emb, n_im, outh, mdim);
  }
  const auto t_host = Clock::now();
  stream.commit().wait();
  const auto t_gpu = Clock::now();
  if (profile) {
    auto ms = [](Clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };
    std::printf("MetalQwenVisionEncoder profile: %dx%d -> %d tok (grid "
                "%dx%d) | host-preproc %.2f | gpu %.2f | total %.2f ms\n",
                H, W, res.n_tokens, grid_h, grid_w, ms(t_host - t_enter),
                ms(t_gpu - t_host), ms(t_gpu - t_enter));
  }

  // Native-f16 zero-copy: hand back the GPU f16 buffer directly (no
  // host-f32 readback). The LM splice copies f16 rows straight in.
  res.embeddings = std::move(emb);
  res.out_hidden = outh;
  return res;
}

}  // namespace vpipe::genai
