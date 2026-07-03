#include "generative-models/moss/metal-moss-local-transformer.h"

#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "generative-models/llama3/metal-llama-weights.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vpipe::genai {

using metal_compute::CommandStream;
using metal_compute::ComputeEncoder;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

float bf16_to_f32(std::uint16_t b)
{
  std::uint32_t u = (std::uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::size_t numel(const std::vector<std::int64_t>& s)
{
  std::size_t n = 1;
  for (auto d : s) { n *= (std::size_t)d; }
  return s.empty() ? 0 : n;
}

// Load a BF16/F16/F32 tensor as an f16 SharedBuffer.
SharedBuffer
load_f16(const MetalLlamaWeights& wts, MetalCompute* mc, const std::string& nm)
{
  const auto* info = wts.info(nm);
  if (info == nullptr) { return {}; }
  SharedBuffer raw = wts.load(nm, mc);
  if (raw.empty()) { return {}; }
  const std::size_t n = numel(info->shape);
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  if (out.empty()) { return {}; }
  auto* dst = static_cast<_Float16*>(out.contents());
  if (info->dtype == "BF16") {
    const auto* s = static_cast<const std::uint16_t*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { dst[i] = (_Float16)bf16_to_f32(s[i]); }
  } else if (info->dtype == "F16") {
    std::memcpy(out.contents(), raw.contents(), n * 2);
  } else if (info->dtype == "F32") {
    const auto* s = static_cast<const float*>(raw.contents());
    for (std::size_t i = 0; i < n; ++i) { dst[i] = (_Float16)s[i]; }
  } else {
    return {};
  }
  return out;
}

}  // namespace

std::unique_ptr<MetalMossLocalTransformer>
MetalMossLocalTransformer::load(const std::string& model_dir, MetalCompute* mc,
                                const Config& cfg)
{
  auto wts = MetalLlamaWeights::open_model(model_dir);
  if (!wts.has_value()) { return nullptr; }
  return load(*wts, mc, cfg);
}

std::unique_ptr<MetalMossLocalTransformer>
MetalMossLocalTransformer::load(const MetalLlamaWeights& wts, MetalCompute* mc,
                                const Config& cfg)
{
  auto self = std::make_unique<MetalMossLocalTransformer>();
  if (!self->init_(wts, mc, cfg)) { return nullptr; }
  return self;
}

bool
MetalMossLocalTransformer::init_(const MetalLlamaWeights& wts, MetalCompute* mc,
                                 const Config& cfg)
{
  _mc = mc;
  _cfg = cfg;
  const std::string p = "local_transformer.h.0.";
  _ca_w = load_f16(wts, mc, p + "attn.c_attn.weight");
  _ca_b = load_f16(wts, mc, p + "attn.c_attn.bias");
  _cp_w = load_f16(wts, mc, p + "attn.c_proj.weight");
  _cp_b = load_f16(wts, mc, p + "attn.c_proj.bias");
  _fi_w = load_f16(wts, mc, p + "mlp.fc_in.weight");
  _fi_b = load_f16(wts, mc, p + "mlp.fc_in.bias");
  _fo_w = load_f16(wts, mc, p + "mlp.fc_out.weight");
  _fo_b = load_f16(wts, mc, p + "mlp.fc_out.bias");
  _ln1_w = load_f16(wts, mc, p + "ln_1.weight");
  _ln1_b = load_f16(wts, mc, p + "ln_1.bias");
  _ln2_w = load_f16(wts, mc, p + "ln_2.weight");
  _ln2_b = load_f16(wts, mc, p + "ln_2.bias");
  _lnf_w = load_f16(wts, mc, "local_transformer.ln_f.weight");
  _lnf_b = load_f16(wts, mc, "local_transformer.ln_f.bias");
  if (_ca_w.empty() || _ca_b.empty() || _cp_w.empty() || _cp_b.empty() ||
      _fi_w.empty() || _fi_b.empty() || _fo_w.empty() || _fo_b.empty() ||
      _ln1_w.empty() || _ln1_b.empty() || _ln2_w.empty() || _ln2_b.empty() ||
      _lnf_w.empty() || _lnf_b.empty()) {
    return false;
  }

  // RoPE inv_freq[i] = 1 / base^(2i/head_dim), f32, head_dim/2 entries.
  const int half = _cfg.head_dim / 2;
  _inv_freq = mc->make_shared_buffer((std::size_t)half * 4);
  if (_inv_freq.empty()) { return false; }
  auto* iv = static_cast<float*>(_inv_freq.contents());
  for (int i = 0; i < half; ++i) {
    iv[i] = 1.0f / std::pow(_cfg.rope_base,
                            (float)(2 * i) / (float)_cfg.head_dim);
  }

  const int H = _cfg.hidden, I = _cfg.inner;
  const int pmax = _cfg.n_vq + 1;
  const std::size_t kvn = (std::size_t)_cfg.n_head * pmax * _cfg.head_dim;
  _kc = mc->make_shared_buffer(kvn * 2);
  _vc = mc->make_shared_buffer(kvn * 2);
  auto a = [&](int n) { return mc->make_shared_buffer((std::size_t)n * 2); };
  _ln1 = a(H); _qkv = a(3 * H); _attn = a(H); _attn2 = a(H); _resid1 = a(H);
  _ln2 = a(H); _fc = a(I); _mlp = a(H); _resid2 = a(H); _hidden = a(H);
  if (_kc.empty() || _vc.empty() || _hidden.empty() || _qkv.empty() ||
      _fc.empty()) {
    return false;
  }

  _lib_vis   = mc->load_library("qwen3_5_vision");
  _lib_dense = mc->load_library("dense_gemm");
  _lib_elt   = mc->load_library("llm_elementwise");
  _lib_rope  = mc->load_library("rope");
  if (!_lib_vis.valid() || !_lib_dense.valid() || !_lib_elt.valid() ||
      !_lib_rope.valid()) {
    return false;
  }
  _fn_ln       = _lib_vis.function("layer_norm_bias_f16");
  _fn_gemv     = _lib_dense.function("dense_gemv_t_f16");
  _fn_bias     = _lib_elt.function("bias_add_rows_f16");
  _fn_residual = _lib_elt.function("residual_add_f16");
  _fn_silu     = _lib_elt.function("mul_sigmoid_f16");
  _fn_attn     = _lib_elt.function("local_attn_step_f16");
  _fn_rope     = _lib_rope.function("rope_interleaved_f16");
  return _fn_ln.valid() && _fn_gemv.valid() && _fn_bias.valid() &&
         _fn_residual.valid() && _fn_silu.valid() && _fn_attn.valid() &&
         _fn_rope.valid();
}

const SharedBuffer*
MetalMossLocalTransformer::step(const SharedBuffer& x_in)
{
  const int pmax = _cfg.n_vq + 1;
  if (_pos >= pmax) { return nullptr; }
  CommandStream s = _mc->make_command_stream();
  if (!s.valid()) { return nullptr; }
  {
    ComputeEncoder e = s.begin_compute();
    encode_step(e, x_in, _pos);
  }
  s.commit().wait();
  _pos++;
  return &_hidden;
}

void
MetalMossLocalTransformer::encode_step(ComputeEncoder& e,
                                       const SharedBuffer& x_in, int pos)
{
  const int H = _cfg.hidden, I = _cfg.inner, nh = _cfg.n_head;
  const int hd = _cfg.head_dim, P = pos, pmax = _cfg.n_vq + 1;
  {
    auto ln = [&](const SharedBuffer& x, const SharedBuffer& w,
                  const SharedBuffer& b, const SharedBuffer& y, int Hd) {
      e.set_function(_fn_ln);
      e.set_buffer(0, x); e.set_buffer(1, w); e.set_buffer(2, b);
      e.set_buffer(3, y); e.set_constant(4, Hd); e.set_constant(5, _cfg.ln_eps);
      e.dispatch({256, 1, 1}, {256, 1, 1});
    };
    auto gemv = [&](const SharedBuffer& x, const SharedBuffer& w,
                    const SharedBuffer& y, int K, int N) {
      e.set_function(_fn_gemv);
      e.set_buffer(0, x); e.set_buffer(1, w); e.set_buffer(2, y);
      e.set_constant(3, K); e.set_constant(4, N);
      e.dispatch({32, (unsigned)(((N + 7) / 8) * 2), 1}, {32, 2, 1});
    };
    auto bias = [&](const SharedBuffer& y, const SharedBuffer& b, int N) {
      e.set_function(_fn_bias);
      e.set_buffer(0, y); e.set_buffer(1, b);
      e.set_constant(2, N); e.set_constant(3, N);
      e.dispatch({(unsigned)N, 1, 1}, {256, 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& b,
                        const SharedBuffer& o, int n) {
      e.set_function(_fn_residual);
      e.set_buffer(0, a); e.set_buffer(1, b); e.set_buffer(2, o);
      e.set_constant(3, n);
      e.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    };
    auto silu = [&](const SharedBuffer& z, int n) {
      e.set_function(_fn_silu);
      e.set_buffer(0, z); e.set_buffer(1, z); e.set_buffer(2, z);
      e.set_constant(3, n);
      e.dispatch({(unsigned)n, 1, 1}, {256, 1, 1});
    };
    auto rope = [&](std::size_t off_elts) {
      e.set_function(_fn_rope);
      e.set_buffer(0, _qkv, off_elts * 2);
      e.set_buffer(1, _inv_freq);
      e.set_constant(2, nh); e.set_constant(3, 1);
      e.set_constant(4, hd); e.set_constant(5, P);
      e.dispatch({(unsigned)(hd / 2), 1, (unsigned)nh}, {(unsigned)(hd / 2),
                 1, 1});
    };

    ln(x_in, _ln1_w, _ln1_b, _ln1, H);
    gemv(_ln1, _ca_w, _qkv, H, 3 * H);
    bias(_qkv, _ca_b, 3 * H);
    rope(0);                       // q
    rope((std::size_t)H);          // k

    e.set_function(_fn_attn);
    e.set_buffer(0, _qkv, 0);
    e.set_buffer(1, _qkv, (std::size_t)H * 2);
    e.set_buffer(2, _qkv, (std::size_t)2 * H * 2);
    e.set_buffer(3, _kc); e.set_buffer(4, _vc); e.set_buffer(5, _attn);
    e.set_constant(6, nh); e.set_constant(7, hd);
    e.set_constant(8, pmax); e.set_constant(9, P);
    e.dispatch({(unsigned)hd, (unsigned)nh, 1}, {(unsigned)hd, 1, 1});

    gemv(_attn, _cp_w, _attn2, H, H);
    bias(_attn2, _cp_b, H);
    residual(x_in, _attn2, _resid1, H);
    ln(_resid1, _ln2_w, _ln2_b, _ln2, H);
    gemv(_ln2, _fi_w, _fc, H, I);
    bias(_fc, _fi_b, I);
    silu(_fc, I);
    gemv(_fc, _fo_w, _mlp, I, H);
    bias(_mlp, _fo_b, H);
    residual(_resid1, _mlp, _resid2, H);
    ln(_resid2, _lnf_w, _lnf_b, _hidden, H);
  }
}

}  // namespace vpipe::genai
