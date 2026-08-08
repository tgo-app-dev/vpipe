#include "generative-models/wan/metal-wan-transformer.h"

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
// shared by everything reading one checkpoint, so the key has to name the
// transform, not just the tensor it came from.
constexpr const char* kKey = "wan-dit/bf16|";

// C++ mirror of mlx::steel::AttnParams (steel/attn/params.h) -- the param
// block the vendored steel flash-attention kernel reads. Identical layout
// to the LM's copy (metal-llama-model.cc) and the Krea-2 DiT's.
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
  return (std::uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// A checkpoint tensor as bf16, converted straight out of the source bytes.
// Deliberately NOT via an f32 staging vector: the Wan checkpoints ship
// fp32 and the largest tensor here is 630 MB, so the intermediate would be
// the peak.
SharedBuffer
to_bf16_(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm)
{
  const auto* info = wts.info(nm);
  if (info == nullptr || info->shape.empty()) { return {}; }
  std::size_t n = 1;
  for (auto d : info->shape) { n *= (std::size_t)d; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { return {}; }
  if (info->dtype == "BF16") { return raw; }
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  auto* d = static_cast<std::uint16_t*>(out.contents());
  if (info->dtype == "F32") {
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

std::string
blk_(int i, const char* rest)
{
  return "blocks." + std::to_string(i) + "." + rest;
}

}  // namespace

bool
MetalWanTransformer::config_from_json(const std::string& dit_dir, Config& out,
                                      std::string* err)
{
  namespace fs = std::filesystem;
  auto fail = [&](std::string m) {
    if (err != nullptr) { *err = std::move(m); }
    return false;
  };
  fs::path p(dit_dir);
  if (fs::is_directory(p) && !fs::exists(p / "config.json")) {
    p = p / "transformer";
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
  if (cls != "WanTransformer3DModel") {
    return fail("not a WanTransformer3DModel config (_class_name=" + cls + ")");
  }
  auto gi = [&](const char* k, int d) {
    return o.contains(k) ? (int)o.at(k).as_int(d) : d;
  };
  out.n_heads      = gi("num_attention_heads", 40);
  out.head_dim     = gi("attention_head_dim", 128);
  out.hidden       = out.n_heads * out.head_dim;
  out.ffn          = gi("ffn_dim", 13824);
  out.n_layers     = gi("num_layers", 40);
  out.in_channels  = gi("in_channels", 36);
  out.out_channels = gi("out_channels", 16);
  out.text_dim     = gi("text_dim", 4096);
  out.freq_dim     = gi("freq_dim", 256);
  if (o.contains("eps")) { out.norm_eps = (float)o.at("eps").as_real(1e-6); }
  if (o.contains("patch_size")) {
    auto ps = o.at("patch_size");
    auto arr = ps.as_array();
    if (arr.size() == 3) {
      out.patch_t = (int)arr[0].as_int(1);
      out.patch_h = (int)arr[1].as_int(2);
      out.patch_w = (int)arr[2].as_int(2);
    }
  }
  // Wan 2.1-I2V's CLIP tower is a different conditioning path entirely (an
  // extra set of image keys and values in every self-attention). Refuse
  // rather than run it as if the tower were not there: the checkpoint
  // would load and produce plausible, wrong video.
  if (o.contains("image_dim") && !o.at("image_dim").is_null()) {
    return fail("image_dim is set: this is a Wan 2.1-style I2V checkpoint "
                "with a CLIP image tower, which is not supported here");
  }
  if (out.n_heads * out.head_dim != out.hidden) {
    return fail("num_attention_heads * attention_head_dim != inner dim");
  }
  if (out.patch_t != 1) {
    return fail("only patch_size[0] == 1 (no temporal patching) is supported");
  }
  return true;
}

SharedBuffer
MetalWanTransformer::weight_(WeightSet& ws, const std::string& nm)
{
  // The checkpoint is FP32 and the forward is bf16, so every tensor is a
  // TRANSFORM the model then keeps -- derived(), not tensor(). The source
  // f32 bytes are dropped as soon as the conversion is done, which is why
  // caching the converted copy costs nothing extra.
  return ws.derived(std::string(kKey) + nm,
                    [&]() -> SharedBuffer { return to_bf16_(ws.src(), _mc, nm); });
}

MetalWanTransformer::Linear
MetalWanTransformer::linear_(WeightSet& ws, const std::string& nm)
{
  Linear l;
  l.b = weight_(ws, nm + ".bias");
  const MetalLlamaWeights& src = ws.src();
  const auto* si = src.info(nm + ".scales");
  const auto* ci = src.info(nm + ".weight");
  if (_quant_bits > 0 && si != nullptr && ci != nullptr &&
      si->shape.size() == 2 && ci->shape.size() == 2) {
    // Per-WEIGHT bit width rather than the config's, so a mixed 4/8
    // checkpoint loads as-is: codes cols = K*bits/32 and scales cols =
    // K/group, so bits = codes_cols*32 / (scales_cols*group).
    const long gcols = ci->shape[1];
    const long scols = si->shape[1];
    const long K = scols * (long)_quant_group;
    const int bits = K > 0 ? (int)(gcols * 32 / K) : 0;
    l.bits = (bits == 8) ? 8 : 4;
    // Codes are U32 and go to the GEMM untouched, so they are read as-is;
    // scales/biases are f16 on disk and the kernels here are bf16, so
    // those two convert (which is what weight_ does).
    l.codes  = ws.tensor(nm + ".weight", _mc, WeightSet::Residency::Copied);
    l.scales = weight_(ws, nm + ".scales");
    l.qbias  = weight_(ws, nm + ".biases");
    if (!l.codes.empty() && !l.scales.empty() && !l.qbias.empty()) {
      l.quantized = true;
      return l;
    }
    l.codes = {}; l.scales = {}; l.qbias = {};
  }
  l.w = weight_(ws, nm + ".weight");
  return l;
}

bool
MetalWanTransformer::load_block_(WeightSet& ws, int i, Block& b)
{
  b.q1 = linear_(ws, blk_(i, "attn1.to_q"));
  b.k1 = linear_(ws, blk_(i, "attn1.to_k"));
  b.v1 = linear_(ws, blk_(i, "attn1.to_v"));
  b.o1 = linear_(ws, blk_(i, "attn1.to_out.0"));
  b.qn1 = weight_(ws, blk_(i, "attn1.norm_q.weight"));
  b.kn1 = weight_(ws, blk_(i, "attn1.norm_k.weight"));
  b.q2 = linear_(ws, blk_(i, "attn2.to_q"));
  b.k2 = linear_(ws, blk_(i, "attn2.to_k"));
  b.v2 = linear_(ws, blk_(i, "attn2.to_v"));
  b.o2 = linear_(ws, blk_(i, "attn2.to_out.0"));
  b.qn2 = weight_(ws, blk_(i, "attn2.norm_q.weight"));
  b.kn2 = weight_(ws, blk_(i, "attn2.norm_k.weight"));
  b.n2w = weight_(ws, blk_(i, "norm2.weight"));
  b.n2b = weight_(ws, blk_(i, "norm2.bias"));
  b.ff_in  = linear_(ws, blk_(i, "ffn.net.0.proj"));
  b.ff_out = linear_(ws, blk_(i, "ffn.net.2"));
  b.sst = weight_(ws, blk_(i, "scale_shift_table"));
  // norm1 and norm3 are elementwise_affine=False, so they contribute no
  // tensors at all -- that is why they are absent from this list rather
  // than missing from the checkpoint.
  return !b.q1.empty() && !b.k1.empty() && !b.v1.empty() && !b.o1.empty() &&
         !b.qn1.empty() && !b.kn1.empty() && !b.q2.empty() && !b.k2.empty() &&
         !b.v2.empty() && !b.o2.empty() && !b.qn2.empty() && !b.kn2.empty() &&
         !b.n2w.empty() && !b.n2b.empty() && !b.ff_in.empty() &&
         !b.ff_out.empty() && !b.sst.empty();
}

MetalWanTransformer::~MetalWanTransformer() = default;

std::unique_ptr<MetalWanTransformer>
MetalWanTransformer::load(const std::string& dit_dir, MetalCompute* mc,
                          const Config& cfg)
{
  namespace fs = std::filesystem;
  fs::path p(dit_dir);
  if (fs::is_directory(p) && fs::exists(p / "transformer") &&
      !fs::exists(p / "config.json")) {
    p = p / "transformer";
  }
  return load(WeightSet::open(p.string(), nullptr), mc, cfg);
}

std::unique_ptr<MetalWanTransformer>
MetalWanTransformer::load(std::shared_ptr<WeightSet> ws_in, MetalCompute* mc,
                          const Config& cfg)
{
  if (mc == nullptr || !ws_in) { return nullptr; }
  WeightSet& ws = *ws_in;
  auto m = std::unique_ptr<MetalWanTransformer>(new MetalWanTransformer());
  m->_ws = std::move(ws_in);
  m->_mc = mc;
  m->_cfg = cfg;

  // BF16 metallibs, reached by name with the SAME *_f16 entry points. The
  // DiT runs bf16 for the reason the Qwen-Image/FLUX.2 siblings do: f16's
  // 65504 ceiling is inside the range this residual stream reaches, and
  // the reference is bf16 anyway.
  m->_lib_gemm = mc->load_library("dense_gemm_bf16");
  m->_lib_elt  = mc->load_library("llm_elementwise_bf16");
  m->_lib_rms  = mc->load_library("rms_norm_bf16");
  m->_lib_rope = mc->load_library("rope_bf16");
  m->_fn_gemm      = m->_lib_gemm.function("dense_gemm_t_bm64_f16");
  m->_fn_rms       = m->_lib_rms.function("rms_norm_fast_f16");
  m->_fn_ln_mod    = m->_lib_elt.function("layernorm_modulate_f16");
  m->_fn_ln_affine = m->_lib_elt.function("layer_norm_affine_f16");
  m->_fn_gelu      = m->_lib_elt.function("gelu_tanh_ff_f16");
  m->_fn_residual  = m->_lib_elt.function("residual_add_f16");
  m->_fn_gated     = m->_lib_elt.function("gated_residual_f16");
  m->_fn_transpose = m->_lib_elt.function("transpose_abd_f16");
  m->_fn_silu      = m->_lib_elt.function("mul_sigmoid_f16");
  m->_fn_bias_add  = m->_lib_elt.function("bias_add_rows_f16");
  m->_fn_trope     = m->_lib_rope.function("transpose_rope_pair_ftab_f16");
  {
    metal_compute::ComputeLibrary sdpa = mc->load_library("sdpa_bf16");
    m->_fn_sdpa = sdpa.function("sdpa_full_f16");
  }
  if (!m->_fn_gemm.valid() || !m->_fn_rms.valid() || !m->_fn_ln_mod.valid() ||
      !m->_fn_ln_affine.valid() || !m->_fn_gelu.valid() ||
      !m->_fn_residual.valid() || !m->_fn_gated.valid() ||
      !m->_fn_transpose.valid() || !m->_fn_silu.valid() ||
      !m->_fn_trope.valid() || !m->_fn_sdpa.valid() ||
      !m->_fn_bias_add.valid()) {
    return nullptr;
  }
  // Steel flash-attention. The scalar sdpa_full_f16 is O(seq^2) at a few
  // percent of peak, and this model's sequences are video-sized, so the
  // fallback is a correctness A/B rather than a path anyone should run.
  m->_lib_attn = mc->load_library("attn_steel");
  m->_attn_p_self  = mc->make_shared_buffer(sizeof(SteelAttnParams));
  m->_attn_p_cross = mc->make_shared_buffer(sizeof(SteelAttnParams));
  m->_steel_ok = m->_lib_attn.valid() && !m->_attn_p_self.empty() &&
                 !m->_attn_p_cross.empty() && cfg.head_dim == 128 &&
                 std::getenv("VPIPE_WAN_NO_STEEL_ATTN") == nullptr;

  // Quantization block from the checkpoint's own config.json. The loader
  // still auto-detects each weight's width; this only says "expect
  // quantized weights" and which group size the scales were built at.
  {
    std::ifstream qin(std::filesystem::path(ws.dir()) / "config.json");
    if (qin) {
      FlexData fd;
      try {
        fd = FlexData::from_json(qin);
      } catch (...) {
        fd = FlexData::make_null();
      }
      if (fd.is_object()) {
        auto o = fd.as_object();
        if (o.contains("quantization")) {
          FlexData q = o.at("quantization");
          if (q.is_object()) {
            auto qo = q.as_object();
            const int b = qo.contains("bits") ? (int)qo.at("bits").as_int(0) : 0;
            if (b == 4 || b == 8) { m->_quant_bits = b; }
            if (qo.contains("group_size")) {
              m->_quant_group = (int)qo.at("group_size").as_int(64);
            }
          }
        }
      }
    }
  }
  if (m->_quant_bits > 0) {
    const std::string g = "g" + std::to_string(m->_quant_group);
    m->_lib_qmm = mc->load_library("affine_qmm_steel_bf16");
    m->_fn_qmm4 = m->_lib_qmm.function("affine_qmm_steel_w4" + g);
    m->_fn_qmm8 = m->_lib_qmm.function("affine_qmm_steel_w8" + g);
    if (!m->_fn_qmm4.valid() || !m->_fn_qmm8.valid()) { return nullptr; }
    // The wide tiles, when the group size has them (only g64 does). Built
    // unconditionally rather than on first tall GEMM: a function build is
    // milliseconds and the denoise loop should not pay it mid-step.
    m->_qmm_tile = 0;
    m->_fn_qmm4_bm64 =
        m->_lib_qmm.function("affine_qmm_steel_w4" + g + "_bm64");
    m->_fn_qmm8_bm64 =
        m->_lib_qmm.function("affine_qmm_steel_w8" + g + "_bm64");
    if (m->_fn_qmm4_bm64.valid() && m->_fn_qmm8_bm64.valid()) {
      m->_qmm_tile = 1;
      m->_fn_qmm4_bm128 =
          m->_lib_qmm.function("affine_qmm_steel_w4" + g + "_bm128");
      m->_fn_qmm8_bm128 =
          m->_lib_qmm.function("affine_qmm_steel_w8" + g + "_bm128");
      if (m->_fn_qmm4_bm128.valid() && m->_fn_qmm8_bm128.valid()) {
        m->_qmm_tile = 2;
      }
    }
    if (const char* t = std::getenv("VPIPE_WAN_QMM_TILE")) {
      m->_qmm_tile = std::min(m->_qmm_tile, std::atoi(t));
    }
    if (mc->session() != nullptr) {
      mc->session()->log_debug(fmt(
          "MetalWanTransformer: quantized checkpoint (w{} {})",
          m->_quant_bits, g));
    }
  }

  m->_patch     = m->linear_(ws, "patch_embedding");
  m->_time1     = m->linear_(ws, "condition_embedder.time_embedder.linear_1");
  m->_time2     = m->linear_(ws, "condition_embedder.time_embedder.linear_2");
  m->_time_proj = m->linear_(ws, "condition_embedder.time_proj");
  m->_text1     = m->linear_(ws, "condition_embedder.text_embedder.linear_1");
  m->_text2     = m->linear_(ws, "condition_embedder.text_embedder.linear_2");
  m->_proj_out  = m->linear_(ws, "proj_out");
  m->_final_sst = m->weight_(ws, "scale_shift_table");
  if (m->_patch.empty() || m->_time1.empty() || m->_time2.empty() ||
      m->_time_proj.empty() || m->_text1.empty() || m->_text2.empty() ||
      m->_proj_out.empty() || m->_final_sst.empty()) {
    return nullptr;
  }

  m->_blocks.resize((std::size_t)cfg.n_layers);
  for (int i = 0; i < cfg.n_layers; ++i) {
    if (!m->load_block_(ws, i, m->_blocks[(std::size_t)i])) { return nullptr; }
  }
  return m;
}

void
MetalWanTransformer::set_qmm_tile(int cap)
{
  // Only ever lowers below what load() built: asking for BM128 on a
  // checkpoint whose group size has no BM128 twin has to stay a no-op,
  // not an invalid function.
  const int built = (_fn_qmm4_bm128.valid() && _fn_qmm8_bm128.valid()) ? 2
                    : (_fn_qmm4_bm64.valid() && _fn_qmm8_bm64.valid()) ? 1
                                                                       : 0;
  if (cap < 0) { cap = 0; }
  _qmm_tile = std::min(cap, built);
}

void
MetalWanTransformer::gemm_(ComputeEncoder& enc, const SharedBuffer& x,
                           const Linear& l, const SharedBuffer& y, int M, int N,
                           int K)
{
  const bool bias = !l.b.empty();
  if (l.quantized) {
    // The steel affine GEMM dequantizes each [BN, BK] weight tile in
    // registers, so the 4-bit codes are never expanded in memory -- which
    // is the whole point at 14B. It has no bias epilogue, so the bias is
    // a separate row-broadcast pass.
    //
    // Tile height from M. A taller tile re-reads each weight tile fewer
    // times; it also costs registers, and therefore occupancy, and at 4
    // bits these GEMMs are ALU-bound rather than weight-bandwidth-bound.
    // So the taller tile only pays once M is large enough that the
    // re-reads dominate, and on M4 Pro that crossover is FAR later than
    // the image DiTs' M>=128. MEASURED w4, alternating the tiles inside
    // one process (best-of; cross-process comparison cannot resolve this
    // -- the box has ~4% of thermal spread between runs):
    //
    //   seq    BM32      BM64       BM128
    //   2304   10.68 s   +2.0%      +2.0%
    //   5760   28.73 s   -4.8%      -5.0%
    //
    // Hence 4096 rather than 128: copying the Krea-2 threshold would have
    // made the 2304-token case -- every 256px clip, and every smoke test
    // -- 2% slower. The win grows with sequence, which is the direction
    // video resolution moves in.
    //
    // The BM64 -> BM128 boundary is NOT resolved: the two are within
    // 0.2% of each other at 5760, and above that only cross-process
    // numbers exist (BM128 -4.1% vs BM32 at 14040), which is inside the
    // spread. 8192 is a placeholder between two arms that measured the
    // same. Both boundaries are per-machine anyway -- a different GPU has
    // a different occupancy cliff -- so the real answer is an autotune
    // over the arms at load, which is what set_qmm_tile() and the
    // step_bench's VPIPE_WAN_BENCH_TILE_AB mode exist to drive.
    int bm = 32;
    if (_qmm_tile == 2 && M >= 8192)      { bm = 128; }
    else if (_qmm_tile >= 1 && M >= 4096) { bm = 64; }
    enc.set_function(
        l.bits == 8
            ? (bm == 128 ? _fn_qmm8_bm128 : bm == 64 ? _fn_qmm8_bm64 : _fn_qmm8)
            : (bm == 128 ? _fn_qmm4_bm128 : bm == 64 ? _fn_qmm4_bm64
                                                     : _fn_qmm4));
    enc.set_buffer(0, l.codes);
    enc.set_buffer(1, l.scales);
    enc.set_buffer(2, l.qbias);
    enc.set_buffer(3, x);
    enc.set_buffer(4, y);
    enc.set_constant(5, K);
    enc.set_constant(6, N);
    enc.set_constant(7, M);
    // BM=128 is the WM=4 variant: 256 threads, threadgroup z 4.
    const unsigned tgz = (bm == 128) ? 4u : 2u;
    enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                  (unsigned)(((M + bm - 1) / bm) * 2), tgz}, {32, 2, tgz});
    if (bias) {
      enc.set_function(_fn_bias_add);
      enc.set_buffer(0, y);
      enc.set_buffer(1, l.b);
      enc.set_constant(2, N);
      enc.set_constant(3, M * N);
      enc.dispatch({(unsigned)(M * N), 1, 1}, {256, 1, 1});
    }
    return;
  }
  enc.set_function(_fn_gemm);
  enc.set_buffer(0, x);
  enc.set_buffer(1, l.w);
  enc.set_buffer(2, bias ? l.b : l.w);
  enc.set_buffer(3, y);
  enc.set_constant(4, K);
  enc.set_constant(5, N);
  enc.set_constant(6, M);
  enc.set_constant(7, bias ? 1 : 0);
  // dense_gemm_t_bm64: BM 64 / BN 32, threadgroup (32, 2, 2).
  enc.dispatch({(unsigned)(((N + 31) / 32) * 32), (unsigned)(((M + 63) / 64) * 2),
                2},
               {32, 2, 2});
}

void
MetalWanTransformer::build_rope_(int T, int ph, int pw, SharedBuffer& cos_out,
                                 SharedBuffer& sin_out) const
{
  const int D = _cfg.head_dim;
  const int seq = T * ph * pw;
  const int axes[3] = {_cfg.rope_t(), _cfg.rope_h(), _cfg.rope_w()};
  const double theta = _cfg.rope_theta;
  auto* cb = static_cast<float*>(cos_out.contents());
  auto* sb = static_cast<float*>(sin_out.contents());

  // Per-pair inverse frequency, coordinate axis and offset within the D
  // row. These depend on (axis, i) only, so they are computed once rather
  // than per token.
  const int pairs = D / 2;
  std::vector<double> pair_freq((std::size_t)pairs);
  std::vector<int> pair_axis((std::size_t)pairs), pair_off((std::size_t)pairs);
  {
    int base = 0, pair = 0;
    for (int a = 0; a < 3; ++a) {
      const int Dax = axes[a];
      for (int i = 0; i < Dax / 2; ++i, ++pair) {
        pair_freq[(std::size_t)pair] =
            1.0 / std::pow(theta, (2.0 * i) / (double)Dax);
        pair_axis[(std::size_t)pair] = a;
        pair_off[(std::size_t)pair] = base + 2 * i;
      }
      base += Dax;
    }
  }
  for (int s = 0; s < seq; ++s) {
    const double coord[3] = {(double)(s / (ph * pw)), (double)((s / pw) % ph),
                             (double)(s % pw)};
    for (int pair = 0; pair < pairs; ++pair) {
      const double ang =
          coord[pair_axis[(std::size_t)pair]] * pair_freq[(std::size_t)pair];
      const float c = (float)std::cos(ang);
      const float sn = (float)std::sin(ang);
      const std::size_t o = (std::size_t)s * D + pair_off[(std::size_t)pair];
      cb[o] = c; cb[o + 1] = c;
      sb[o] = sn; sb[o + 1] = sn;
    }
  }
}

bool
MetalWanTransformer::ensure_scratch_(int T, int ph, int pw, int text_seq)
{
  const int seq = T * ph * pw;
  if (_s.seq == seq && _s.text_seq == text_seq) { return true; }
  const Config& c = _cfg;
  const std::size_t S = (std::size_t)seq, H = (std::size_t)c.hidden;
  const std::size_t TS = (std::size_t)text_seq;
  auto mk = [&](std::size_t elems) { return _mc->make_shared_buffer(elems * 2); };
  Scratch s;
  s.seq = seq;
  s.text_seq = text_seq;
  s.rcos = _mc->make_shared_buffer(S * (std::size_t)c.head_dim * sizeof(float));
  s.rsin = _mc->make_shared_buffer(S * (std::size_t)c.head_dim * sizeof(float));
  s.x     = mk(S * (std::size_t)c.patch_elems());
  s.joint = mk(S * H);
  s.nm    = mk(S * H);
  s.qb    = mk(S * H);
  s.kb    = mk(S * H);
  s.vb    = mk(S * H);
  s.qh    = mk(S * H);
  s.kh    = mk(S * H);
  s.vh    = mk(S * H);
  s.oh    = mk(S * H);
  s.ob    = mk(S * H);
  s.ffb   = mk(S * (std::size_t)c.ffn);
  s.outp  = mk(S * (std::size_t)c.out_patch_elems());
  s.tk    = mk(TS * H);
  s.tv    = mk(TS * H);
  s.tkh   = mk(TS * H);
  s.tvh   = mk(TS * H);
  s.te_in = mk((std::size_t)c.freq_dim);
  s.te1   = mk(H);
  s.temb  = mk(H);
  s.tproj = mk(6 * H);
  s.mod   = mk(6 * H);
  s.mod2  = mk(2 * H);
  if (s.rcos.empty() || s.rsin.empty() || s.x.empty() || s.joint.empty() ||
      s.nm.empty() || s.qb.empty() || s.kb.empty() || s.vb.empty() ||
      s.qh.empty() || s.kh.empty() || s.vh.empty() || s.oh.empty() ||
      s.ob.empty() || s.ffb.empty() || s.outp.empty() || s.tk.empty() ||
      s.tv.empty() || s.tkh.empty() || s.tvh.empty() || s.te_in.empty() ||
      s.te1.empty() || s.temb.empty() || s.tproj.empty() || s.mod.empty() ||
      s.mod2.empty()) {
    return false;
  }
  build_rope_(T, ph, pw, s.rcos, s.rsin);
  _s = std::move(s);
  return true;
}

SharedBuffer
MetalWanTransformer::encode_text(const SharedBuffer& text, int text_seq,
                                 std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  if (text_seq <= 0 ||
      text.byte_size() < (std::size_t)text_seq * c.text_dim * 2) {
    return fail("text conditioning is smaller than text_seq * text_dim");
  }
  const std::size_t n = (std::size_t)text_seq * (std::size_t)c.hidden;
  SharedBuffer mid = _mc->make_shared_buffer(n * 2);
  SharedBuffer out = _mc->make_shared_buffer(n * 2);
  if (mid.empty() || out.empty()) { return fail("allocation failed"); }
  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    gemm_(enc, text, _text1, mid, text_seq, c.hidden, c.text_dim);
    enc.set_function(_fn_gelu);
    enc.set_buffer(0, mid);
    enc.set_buffer(1, mid);
    enc.set_constant(2, (int)n);
    enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    gemm_(enc, mid, _text2, out, text_seq, c.hidden, c.hidden);
  }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? std::string("text projection failed")
                                : gpu_err);
  }
  return out;
}

SharedBuffer
MetalWanTransformer::forward(const SharedBuffer& latents, int T, int h, int w,
                             const SharedBuffer& text_proj, int text_seq,
                             float timestep, std::string* err)
{
  auto fail = [&](std::string m) -> SharedBuffer {
    if (err != nullptr) { *err = std::move(m); }
    return {};
  };
  const Config& c = _cfg;
  const int HID = c.hidden, HD = c.head_dim, NH = c.n_heads, FF = c.ffn;
  const int CIN = c.in_channels, COUT = c.out_channels;
  const int PH = c.patch_h, PW = c.patch_w;
  if (T <= 0 || h <= 0 || w <= 0) { return fail("empty latent video"); }
  if (h % PH != 0 || w % PW != 0) {
    return fail(fmt("latent {}x{} is not a multiple of the {}x{} patch", h, w,
                    PH, PW)());
  }
  const int ph = h / PH, pw = w / PW;
  const int seq = T * ph * pw;
  if (latents.byte_size() < (std::size_t)CIN * T * h * w * 2) {
    return fail("latents are smaller than in_channels * T * h * w");
  }
  if (text_seq <= 0 ||
      text_proj.byte_size() < (std::size_t)text_seq * HID * 2) {
    return fail("text_proj is smaller than text_seq * hidden");
  }
  if (!ensure_scratch_(T, ph, pw, text_seq)) {
    return fail("activation allocation failed (out of GPU memory)");
  }
  Scratch& s = _s;
  const int PE = c.patch_elems(), OPE = c.out_patch_elems();
  const float eps = c.norm_eps;

  // ---- patchify -------------------------------------------------------
  // The reference is a Conv3d with kernel == stride, so it is a Linear over
  // one patch. The flattening order is the conv weight's [out, in, kt, kh,
  // kw]: the CHANNEL is slowest and the patch column fastest. (The output
  // side reverses this -- see the unpatchify below -- and getting the two
  // the same way round is the classic way to produce a plausible, wrong
  // frame.)
  {
    const auto* src = static_cast<const std::uint16_t*>(latents.contents());
    auto* dst = static_cast<std::uint16_t*>(s.x.contents());
    const std::size_t plane = (std::size_t)h * w;
    for (int t = 0; t < T; ++t) {
      for (int hh = 0; hh < ph; ++hh) {
        for (int ww = 0; ww < pw; ++ww) {
          std::uint16_t* row =
              dst + ((std::size_t)((t * ph + hh) * pw + ww)) * PE;
          for (int ch = 0; ch < CIN; ++ch) {
            const std::uint16_t* cp =
                src + ((std::size_t)ch * T + t) * plane;
            for (int ky = 0; ky < PH; ++ky) {
              for (int kx = 0; kx < PW; ++kx) {
                row[(ch * PH + ky) * PW + kx] =
                    cp[(std::size_t)(hh * PH + ky) * w + (ww * PW + kx)];
              }
            }
          }
        }
      }
    }
  }

  // ---- the sinusoidal timestep row ------------------------------------
  // diffusers Timesteps(freq_dim, flip_sin_to_cos=True,
  // downscale_freq_shift=0): cos first, then sin, over exp(-ln(1e4)*i/half).
  {
    auto* ti = static_cast<std::uint16_t*>(s.te_in.contents());
    const int half = c.freq_dim / 2;
    for (int i = 0; i < half; ++i) {
      const double fr = std::exp(-std::log(1e4) * (double)i / (double)half);
      const double ang = (double)timestep * fr;
      ti[i] = f32_to_bf16_((float)std::cos(ang));
      ti[half + i] = f32_to_bf16_((float)std::sin(ang));
    }
  }
  // The final layer's modulation is scale_shift_table + temb, and temb is
  // only known on the GPU -- so seed mod2 with the table on the host and
  // let a bias_add fold temb in below.
  std::memcpy(s.mod2.contents(), _final_sst.contents(),
              (std::size_t)2 * HID * 2);

  // ---- steel flash-attention setup ------------------------------------
  // Two shapes: self-attention (qL = kL = seq) and cross-attention into the
  // text (qL = seq, kL = text_seq). The alignment function constants differ
  // with the lengths, so each gets its own built function and param block.
  const float scale = 1.0f / std::sqrt((float)HD);
  const int A_BQ = 32, A_BK = 16;
  bool use_steel = _steel_ok;
  // Built per SHAPE, not per call: the alignment function constants and the
  // param blocks depend only on (seq, text_seq), which the denoise loop
  // holds fixed across all of its steps.
  if (use_steel && (_attn_seq != seq || _attn_kv != text_seq)) {
    auto fill = [&](SharedBuffer& pb, int qL, int kL) {
      auto* p = static_cast<SteelAttnParams*>(pb.contents());
      p->B = 1; p->H = NH; p->D = HD;
      p->qL = qL; p->kL = kL;
      p->gqa_factor = 1; p->scale = scale;
      p->NQ = (qL + A_BQ - 1) / A_BQ; p->NK = (kL + A_BK - 1) / A_BK;
      p->NQ_aligned = qL / A_BQ; p->NK_aligned = kL / A_BK;
      p->qL_rem = qL - p->NQ_aligned * A_BQ;
      p->kL_rem = kL - p->NK_aligned * A_BK;
      p->qL_off = 0;
      p->Q_strides[0] = (std::int64_t)NH * qL * HD;
      p->Q_strides[1] = (std::int64_t)qL * HD;
      p->Q_strides[2] = HD;
      p->K_strides[0] = (std::int64_t)NH * kL * HD;
      p->K_strides[1] = (std::int64_t)kL * HD;
      p->K_strides[2] = HD;
      p->V_strides[0] = p->K_strides[0];
      p->V_strides[1] = p->K_strides[1];
      p->V_strides[2] = HD;
      p->O_strides[0] = p->Q_strides[0];
      p->O_strides[1] = p->Q_strides[1];
      p->O_strides[2] = HD;
    };
    fill(_attn_p_self, seq, seq);
    fill(_attn_p_cross, seq, text_seq);
    auto build = [&](int qL, int kL) {
      metal_compute::FunctionConstants fc;
      fc.set_bool(200, (qL % A_BQ) == 0).set_bool(201, (kL % A_BK) == 0)
          .set_bool(300, false).set_bool(301, false).set_bool(302, false);
      return _lib_attn.function("attn_steel_h_bd128_bf16", fc);
    };
    _fn_attn_self = build(seq, seq);
    _fn_attn_cross = build(seq, text_seq);
    _attn_seq = seq;
    _attn_kv = text_seq;
  }
  if (use_steel) {
    use_steel = _fn_attn_self.valid() && _fn_attn_cross.valid();
  }
  const unsigned a_nqb = (unsigned)((seq + A_BQ - 1) / A_BQ);

  // ---- env-gated per-section GPU timing (VPIPE_WAN_DIT_PROFILE) --------
  // The whole forward is ONE deferred stream, so there is nothing to time
  // inside it without splitting: each psplit() ends the encoder, commits,
  // waits, and charges the elapsed slice to a bucket. That serializes the
  // GPU and inflates absolute step time by the per-commit overhead times
  // ~11 splits per block, so read the SHARE, not the total. All the
  // scratch is a _s member and outlives every commit, so splitting
  // mid-block changes nothing but the timing.
  const bool prof = std::getenv("VPIPE_WAN_DIT_PROFILE") != nullptr;
  double t_cond = 0, t_patch = 0, t_qkv = 0, t_prep = 0, t_attn_s = 0,
         t_oproj = 0, t_xqkv = 0, t_attn_x = 0, t_xoproj = 0, t_ffup = 0,
         t_gelu = 0, t_ffdown = 0, t_elt = 0, t_final = 0;
  const auto t_begin = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point mark = t_begin;

  CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
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
    auto rms = [&](const SharedBuffer& x, const SharedBuffer& gamma,
                   const SharedBuffer& y, int rows, int width) {
      enc.set_function(_fn_rms);
      enc.set_buffer(0, x); enc.set_buffer(1, gamma); enc.set_buffer(2, y);
      enc.set_constant(3, width); enc.set_constant(4, eps);
      enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
    };
    auto ln_mod = [&](const SharedBuffer& x, const SharedBuffer& m,
                      std::size_t scale_off, std::size_t shift_off,
                      const SharedBuffer& y, int rows) {
      enc.set_function(_fn_ln_mod);
      enc.set_buffer(0, x);
      enc.set_buffer(1, m, scale_off * 2);
      enc.set_buffer(2, m, shift_off * 2);
      enc.set_buffer(3, y);
      enc.set_constant(4, HID); enc.set_constant(5, eps);
      enc.dispatch({256, (unsigned)rows, 1}, {256, 1, 1});
    };
    auto elt3 = [&](const metal_compute::ComputeFunction& fn,
                    const SharedBuffer& a, const SharedBuffer& b,
                    const SharedBuffer& y, std::size_t n) {
      enc.set_function(fn);
      enc.set_buffer(0, a); enc.set_buffer(1, b); enc.set_buffer(2, y);
      enc.set_constant(3, (int)n);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    };
    auto gated = [&](const SharedBuffer& x, std::size_t gate_off,
                     const SharedBuffer& sub, std::size_t n) {
      enc.set_function(_fn_gated);
      enc.set_buffer(0, x);
      enc.set_buffer(1, s.mod, gate_off * 2);
      enc.set_buffer(2, sub);
      enc.set_constant(3, HID); enc.set_constant(4, (int)n);
      enc.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& in, const SharedBuffer& out, int A,
                         int B, int D) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_constant(2, A); enc.set_constant(3, B); enc.set_constant(4, D);
      enc.dispatch({(unsigned)D, (unsigned)B, (unsigned)A}, {(unsigned)D, 1, 1});
    };
    auto trope = [&](const SharedBuffer& in, const SharedBuffer& out) {
      enc.set_function(_fn_trope);
      enc.set_buffer(0, in); enc.set_buffer(1, out);
      enc.set_buffer(2, s.rcos); enc.set_buffer(3, s.rsin);
      enc.set_constant(4, NH); enc.set_constant(5, seq); enc.set_constant(6, HD);
      enc.dispatch({(unsigned)(HD / 2), (unsigned)seq, (unsigned)NH},
                   {(unsigned)(HD / 2), 1, 1});
    };
    auto attn = [&](const SharedBuffer& q, const SharedBuffer& k,
                    const SharedBuffer& v, const SharedBuffer& out, bool self) {
      if (use_steel) {
        enc.set_function(self ? _fn_attn_self : _fn_attn_cross);
        enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
        enc.set_buffer(3, out);
        enc.set_buffer(4, self ? _attn_p_self : _attn_p_cross);
        enc.dispatch({32 * a_nqb, 4 * (unsigned)NH, 1}, {32, 4, 1});
        return;
      }
      const int kvl = self ? seq : text_seq;
      enc.set_function(_fn_sdpa);
      enc.set_buffer(0, q); enc.set_buffer(1, k); enc.set_buffer(2, v);
      enc.set_buffer(3, out);
      enc.set_constant(4, scale); enc.set_constant(5, kvl);
      enc.set_constant(6, HD); enc.set_constant(7, NH); enc.set_constant(8, NH);
      enc.set_constant(9, seq); enc.set_constant(10, kvl);
      enc.dispatch({32, (unsigned)NH, (unsigned)seq}, {32, 1, 1});
    };

    // ---- conditioning ------------------------------------------------
    gemm_(enc, s.te_in, _time1, s.te1, 1, HID, c.freq_dim);
    elt3(_fn_silu, s.te1, s.te1, s.te1, (std::size_t)HID);
    gemm_(enc, s.te1, _time2, s.temb, 1, HID, HID);
    elt3(_fn_silu, s.temb, s.temb, s.te1, (std::size_t)HID);
    gemm_(enc, s.te1, _time_proj, s.tproj, 1, 6 * HID, HID);
    // mod2 (seeded with scale_shift_table on the host) += temb.
    enc.set_function(_fn_bias_add);
    enc.set_buffer(0, s.mod2); enc.set_buffer(1, s.temb);
    enc.set_constant(2, HID);
    enc.set_constant(3, 2 * HID);
    enc.dispatch({(unsigned)(2 * HID), 1, 1}, {256, 1, 1});
    psplit(t_cond);

    // ---- patch embedding ---------------------------------------------
    gemm_(enc, s.x, _patch, s.joint, seq, HID, PE);
    psplit(t_patch);

    // ---- blocks -------------------------------------------------------
    for (int L = 0; L < c.n_layers; ++L) {
      if (_block_progress) { _block_progress(L, c.n_layers); }
      const Block& b = _blocks[(std::size_t)L];
      // shift, scale, gate, c_shift, c_scale, c_gate -- the reference's
      // chunk order over the 6-way table.
      elt3(_fn_residual, s.tproj, b.sst, s.mod, (std::size_t)6 * HID);

      // --- self-attention ---
      ln_mod(s.joint, s.mod, (std::size_t)HID, 0, s.nm, seq);
      psplit(t_elt);
      gemm_(enc, s.nm, b.q1, s.qb, seq, HID, HID);
      gemm_(enc, s.nm, b.k1, s.kb, seq, HID, HID);
      gemm_(enc, s.nm, b.v1, s.vb, seq, HID, HID);
      psplit(t_qkv);
      // rms_norm_across_heads: ONE RMS over the whole 5120-wide row, before
      // the head split. Per-head would be a different normalization.
      rms(s.qb, b.qn1, s.qb, seq, HID);
      rms(s.kb, b.kn1, s.kb, seq, HID);
      trope(s.qb, s.qh);
      trope(s.kb, s.kh);
      transpose(s.vb, s.vh, seq, NH, HD);
      psplit(t_prep);
      attn(s.qh, s.kh, s.vh, s.oh, true);
      psplit(t_attn_s);
      transpose(s.oh, s.ob, NH, seq, HD);
      psplit(t_prep);
      gemm_(enc, s.ob, b.o1, s.qb, seq, HID, HID);
      psplit(t_oproj);
      gated(s.joint, (std::size_t)2 * HID, s.qb, (std::size_t)seq * HID);

      // --- cross-attention into the text ---
      enc.set_function(_fn_ln_affine);
      enc.set_buffer(0, s.joint); enc.set_buffer(1, b.n2w);
      enc.set_buffer(2, b.n2b); enc.set_buffer(3, s.nm);
      enc.set_constant(4, HID); enc.set_constant(5, eps);
      enc.dispatch({256, (unsigned)seq, 1}, {256, 1, 1});
      psplit(t_elt);
      gemm_(enc, s.nm, b.q2, s.qb, seq, HID, HID);
      gemm_(enc, text_proj, b.k2, s.tk, text_seq, HID, HID);
      gemm_(enc, text_proj, b.v2, s.tv, text_seq, HID, HID);
      rms(s.qb, b.qn2, s.qb, seq, HID);
      rms(s.tk, b.kn2, s.tk, text_seq, HID);
      transpose(s.qb, s.qh, seq, NH, HD);          // no RoPE across streams
      transpose(s.tk, s.tkh, text_seq, NH, HD);
      transpose(s.tv, s.tvh, text_seq, NH, HD);
      psplit(t_xqkv);
      attn(s.qh, s.tkh, s.tvh, s.oh, false);
      psplit(t_attn_x);
      transpose(s.oh, s.ob, NH, seq, HD);
      gemm_(enc, s.ob, b.o2, s.qb, seq, HID, HID);
      elt3(_fn_residual, s.joint, s.qb, s.joint, (std::size_t)seq * HID);
      psplit(t_xoproj);

      // --- feed-forward (UNGATED: one 13824-wide hidden through gelu) ---
      ln_mod(s.joint, s.mod, (std::size_t)4 * HID, (std::size_t)3 * HID, s.nm,
             seq);
      psplit(t_elt);
      gemm_(enc, s.nm, b.ff_in, s.ffb, seq, FF, HID);
      psplit(t_ffup);
      enc.set_function(_fn_gelu);
      enc.set_buffer(0, s.ffb); enc.set_buffer(1, s.ffb);
      enc.set_constant(2, (int)((std::size_t)seq * FF));
      enc.dispatch({(unsigned)((std::size_t)seq * FF), 1, 1}, {256, 1, 1});
      psplit(t_gelu);
      gemm_(enc, s.ffb, b.ff_out, s.qb, seq, HID, FF);
      psplit(t_ffdown);
      gated(s.joint, (std::size_t)5 * HID, s.qb, (std::size_t)seq * HID);
      psplit(t_elt);
    }

    // ---- output norm + projection -------------------------------------
    ln_mod(s.joint, s.mod2, (std::size_t)HID, 0, s.nm, seq);
    gemm_(enc, s.nm, _proj_out, s.outp, seq, OPE, HID);
    psplit(t_final);
  }
  std::string gpu_err;
  if (!stream.commit().wait_ok(&gpu_err)) {
    return fail(gpu_err.empty() ? std::string("Wan DiT forward failed")
                                : gpu_err);
  }

  // ---- unpatchify ------------------------------------------------------
  // The output patch flattens the OTHER way round from the input: the
  // reference reshapes proj_out's 64 columns as (p_t, p_h, p_w, c), so the
  // CHANNEL is fastest here where it was slowest going in.
  SharedBuffer out =
      _mc->make_shared_buffer((std::size_t)COUT * T * h * w * 2);
  if (out.empty()) { return fail("output allocation failed"); }
  {
    const auto* src = static_cast<const std::uint16_t*>(s.outp.contents());
    auto* dst = static_cast<std::uint16_t*>(out.contents());
    const std::size_t plane = (std::size_t)h * w;
    for (int t = 0; t < T; ++t) {
      for (int hh = 0; hh < ph; ++hh) {
        for (int ww = 0; ww < pw; ++ww) {
          const std::uint16_t* row =
              src + ((std::size_t)((t * ph + hh) * pw + ww)) * OPE;
          for (int ky = 0; ky < PH; ++ky) {
            for (int kx = 0; kx < PW; ++kx) {
              const std::uint16_t* pat = row + (std::size_t)(ky * PW + kx) * COUT;
              for (int ch = 0; ch < COUT; ++ch) {
                dst[((std::size_t)ch * T + t) * plane +
                    (std::size_t)(hh * PH + ky) * w + (ww * PW + kx)] = pat[ch];
              }
            }
          }
        }
      }
    }
  }
  if (prof && _mc->session() != nullptr) {
    const double host = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - mark).count();
    const double tot = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_begin).count();
    // The GEMM buckets in achieved GFLOP/s, so a slow kernel is visible as
    // a rate against the box's roofline rather than only as a share. 2*M*N*K
    // per GEMM, summed over the 40 blocks.
    const double L = (double)c.n_layers;
    auto rate = [](double flops, double ms) {
      return ms > 0.0 ? flops / (ms * 1e6) : 0.0;
    };
    const double f_qkv  = L * 3.0 * 2.0 * seq * HID * HID;
    const double f_op   = L * 2.0 * (double)seq * HID * HID;
    const double f_xqkv = L * 2.0 * ((double)seq + 2.0 * text_seq) * HID * HID;
    const double f_ffup = L * 2.0 * (double)seq * FF * HID;
    _mc->session()->info(fmt(
        "[wan-dit] {} tokens ({}x{}x{}), {} blocks, {:.0f} ms total\n"
        "  cond      {:8.1f} ms  patch  {:8.1f} ms  final {:8.1f} ms\n"
        "  qkv       {:8.1f} ms ({:6.0f} GF/s)   o-proj  {:8.1f} ms "
        "({:6.0f} GF/s)\n"
        "  x-qkv     {:8.1f} ms ({:6.0f} GF/s)   x-oproj {:8.1f} ms\n"
        "  ff-up     {:8.1f} ms ({:6.0f} GF/s)   ff-down {:8.1f} ms "
        "({:6.0f} GF/s)\n"
        "  attn-self {:8.1f} ms  attn-cross {:8.1f} ms  ({})\n"
        "  prep      {:8.1f} ms  gelu   {:8.1f} ms  elt   {:8.1f} ms\n"
        "  unpatchify (HOST) {:.1f} ms",
        seq, T, ph, pw, c.n_layers, tot,
        t_cond, t_patch, t_final,
        t_qkv, rate(f_qkv, t_qkv), t_oproj, rate(f_op, t_oproj),
        t_xqkv, rate(f_xqkv, t_xqkv), t_xoproj,
        t_ffup, rate(f_ffup, t_ffup), t_ffdown, rate(f_ffup, t_ffdown),
        t_attn_s, t_attn_x, use_steel ? "steel flash" : "SCALAR sdpa",
        t_prep, t_gelu, t_elt, host));
  }
  return out;
}

}  // namespace genai
}  // namespace vpipe
