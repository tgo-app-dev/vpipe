#include "generative-models/moss/metal-moss-codec.h"

#include "generative-models/llama3/metal-llama-weights.h"
#include "apple-silicon/metal-compute/command-stream.h"
#include "apple-silicon/metal-compute/compute-encoder.h"
#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace vpipe::genai {

using metal_compute::ComputeEncoder;
using metal_compute::SharedBuffer;

namespace {
std::size_t numel_(const std::vector<std::int64_t>& s) {
  std::size_t n = 1;
  for (auto d : s) { n *= (std::size_t)d; }
  return n;
}
// Convert an F32 checkpoint tensor to an f16 SharedBuffer (same row-major
// layout). Empty on a missing tensor.
SharedBuffer to_f16_(const MetalLlamaWeights& wts, metal_compute::MetalCompute* mc,
                     const std::string& name) {
  const auto* info = wts.info(name);
  if (info == nullptr) { return {}; }
  SharedBuffer src = wts.load(name, mc);
  if (src.empty()) { return {}; }
  const std::size_t n = numel_(info->shape);
  SharedBuffer out = mc->make_shared_buffer(n * 2);
  const auto* s = static_cast<const float*>(src.contents());
  auto* d = static_cast<_Float16*>(out.contents());
  for (std::size_t i = 0; i < n; ++i) { d[i] = (_Float16)s[i]; }
  return out;
}
// to_f16 with each output ROW scaled by scale[row] (folds a per-channel
// LayerScale into the projection weight: weight is [out, in], scale is [out]).
SharedBuffer to_f16_rowscale_(const MetalLlamaWeights& wts,
                              metal_compute::MetalCompute* mc,
                              const std::string& wname,
                              const std::string& sname) {
  const auto* wi = wts.info(wname);
  const auto* si = wts.info(sname);
  if (wi == nullptr || si == nullptr || wi->shape.size() != 2) { return {}; }
  const int out = (int)wi->shape[0], in = (int)wi->shape[1];
  SharedBuffer ws = wts.load(wname, mc), ss = wts.load(sname, mc);
  if (ws.empty() || ss.empty()) { return {}; }
  SharedBuffer o = mc->make_shared_buffer((std::size_t)out * in * 2);
  const auto* w = static_cast<const float*>(ws.contents());
  const auto* sc = static_cast<const float*>(ss.contents());
  auto* d = static_cast<_Float16*>(o.contents());
  for (int r = 0; r < out; ++r) {
    for (int c = 0; c < in; ++c) {
      d[(std::size_t)r * in + c] = (_Float16)(sc[r] * w[(std::size_t)r * in + c]);
    }
  }
  return o;
}
// Fold a weight-normalized 1x1 conv (kernel_size 1) into a plain [out,in]
// f16 matrix: w_eff[o,i] = g[o] * v[o,i] / ||v[o,:]||_2  (parametrizations
// original0=g [out,1,1], original1=v [out,in,1]).
SharedBuffer fold_wnconv_(const MetalLlamaWeights& wts,
                          metal_compute::MetalCompute* mc,
                          const std::string& prefix) {
  const std::string gn = prefix + ".parametrizations.weight.original0";
  const std::string vn = prefix + ".parametrizations.weight.original1";
  const auto* vi = wts.info(vn);
  const auto* gi = wts.info(gn);
  if (vi == nullptr || gi == nullptr || vi->shape.size() < 2) { return {}; }
  const int out = (int)vi->shape[0], in = (int)vi->shape[1];
  SharedBuffer vs = wts.load(vn, mc), gs = wts.load(gn, mc);
  if (vs.empty() || gs.empty()) { return {}; }
  const auto* v = static_cast<const float*>(vs.contents());
  const auto* g = static_cast<const float*>(gs.contents());
  SharedBuffer o = mc->make_shared_buffer((std::size_t)out * in * 2);
  auto* d = static_cast<_Float16*>(o.contents());
  for (int r = 0; r < out; ++r) {
    double ss = 0.0;
    for (int c = 0; c < in; ++c) {
      const double x = v[(std::size_t)r * in + c];
      ss += x * x;
    }
    const float inv = (float)(1.0 / std::sqrt(ss));
    for (int c = 0; c < in; ++c) {
      d[(std::size_t)r * in + c] =
          (_Float16)(g[r] * v[(std::size_t)r * in + c] * inv);
    }
  }
  return o;
}
}  // namespace

std::unique_ptr<MetalMossCodec>
MetalMossCodec::load(const std::string& model_dir,
                     metal_compute::MetalCompute* mc) {
  if (mc == nullptr || !mc->valid()) { return nullptr; }
  auto self = std::unique_ptr<MetalMossCodec>(new MetalMossCodec());
  self->_mc = mc;

  {
    std::ifstream in(model_dir + "/config.json");
    if (in) {
      try {
        FlexData root = FlexData::from_json(in);
        if (root.is_object()) {
          const auto ro = root.as_object();
          if (ro.contains("sample_rate")) {
            self->_sample_rate = (int)ro.at("sample_rate").as_int(24000);
          }
        }
      } catch (...) {}
    }
  }

  // The 8B MOSS-Audio-Tokenizer decoder structure (stable; see config.json
  // decoder_kwargs). 4 ProjectedTransformers, patch upsamplers x2,x2,x2,x240.
  // context = round(frame_rate * 10s); frame_rate = 12.5,25,50,100 Hz.
  self->_stages.resize(4);
  const StageCfg cfgs[4] = {
      {768, 1280, 32, 20, 5120, 1280,   2,  125},  // decoder.0
      {640,  768, 12, 12, 3072,  768,   2,  250},  // decoder.2
      {384,  768, 12, 12, 3072,  768,   2,  500},  // decoder.4
      {384,  768, 12, 12, 3072,  240, 240, 1000},  // decoder.6
  };
  const int module_idx[4] = {0, 2, 4, 6};

  auto wts_opt = MetalLlamaWeights::open_model(model_dir);
  if (!wts_opt) { return nullptr; }
  const MetalLlamaWeights& wts = *wts_opt;

  bool ok = true;
  // ---- RVQ (quantizer) decode weights -------------------------------
  self->_codebook.resize((std::size_t)self->_n_vq);
  self->_q_outw.resize((std::size_t)self->_n_vq);
  self->_q_outb.resize((std::size_t)self->_n_vq);
  for (int i = 0; i < self->_n_vq; ++i) {
    const std::string q = "quantizer.quantizers." + std::to_string(i);
    self->_codebook[(std::size_t)i] = to_f16_(wts, mc, q + ".codebook.weight");
    self->_q_outw[(std::size_t)i] = fold_wnconv_(wts, mc, q + ".out_proj");
    self->_q_outb[(std::size_t)i] = to_f16_(wts, mc, q + ".out_proj.bias");
    ok = ok && !self->_codebook[(std::size_t)i].empty() &&
         !self->_q_outw[(std::size_t)i].empty() &&
         !self->_q_outb[(std::size_t)i].empty();
  }
  self->_rvq_outw = fold_wnconv_(wts, mc, "quantizer.output_proj");
  self->_rvq_outb = to_f16_(wts, mc, "quantizer.output_proj.bias");
  ok = ok && !self->_rvq_outw.empty() && !self->_rvq_outb.empty();

  // ---- decoder transformer stages -----------------------------------
  for (int s = 0; s < 4; ++s) {
    Stage& st = self->_stages[(std::size_t)s];
    st.cfg = cfgs[s];
    const std::string base = "decoder." + std::to_string(module_idx[s]) + ".";
    st.in_proj = to_f16_(wts, mc, base + "input_proj.weight");
    ok = ok && !st.in_proj.empty();
    if (st.cfg.out_dim != st.cfg.d_model) {
      st.out_proj = to_f16_(wts, mc, base + "output_proj.weight");
      ok = ok && !st.out_proj.empty();
    }
    st.layers.resize((std::size_t)st.cfg.n_layers);
    for (int l = 0; l < st.cfg.n_layers; ++l) {
      Layer& L = st.layers[(std::size_t)l];
      const std::string p =
          base + "transformer.layers." + std::to_string(l) + ".";
      L.n1w = to_f16_(wts, mc, p + "norm1.weight");
      L.n1b = to_f16_(wts, mc, p + "norm1.bias");
      L.n2w = to_f16_(wts, mc, p + "norm2.weight");
      L.n2b = to_f16_(wts, mc, p + "norm2.bias");
      L.qkvw = to_f16_(wts, mc, p + "self_attn.in_projs.0.weight");
      // Fold the two per-channel LayerScales into out_proj / linear2.
      L.ow = to_f16_rowscale_(wts, mc, p + "self_attn.out_projs.0.weight",
                              p + "layer_scale_1.scale");
      L.fc1 = to_f16_(wts, mc, p + "linear1.weight");
      L.fc2 = to_f16_rowscale_(wts, mc, p + "linear2.weight",
                               p + "layer_scale_2.scale");
      ok = ok && !L.n1w.empty() && !L.n1b.empty() && !L.n2w.empty() &&
           !L.n2b.empty() && !L.qkvw.empty() && !L.ow.empty() &&
           !L.fc1.empty() && !L.fc2.empty();
    }
  }
  if (!ok) { return nullptr; }

  // inv_freq for the interleaved RoPE (head_dim 64, max_period 10000).
  const int hd = 64, half = hd / 2;
  self->_inv_freq = mc->make_shared_buffer((std::size_t)half * sizeof(float));
  auto* invf = static_cast<float*>(self->_inv_freq.contents());
  for (int i = 0; i < half; ++i) {
    invf[i] = 1.0f / std::pow(10000.0f, (2.0f * (float)i) / (float)hd);
  }

  // ---- kernels (f16) ------------------------------------------------
  self->_lib_gemm = mc->load_library("dense_gemm");
  self->_lib_vis = mc->load_library("qwen3_5_vision");
  self->_lib_elt = mc->load_library("llm_elementwise");
  self->_lib_sdpa = mc->load_library("sdpa");
  self->_lib_rope = mc->load_library("rope");
  self->_fn_gemm = self->_lib_gemm.function("dense_gemm_t_f16");
  self->_fn_ln = self->_lib_vis.function("layer_norm_bias_f16");
  self->_fn_gelu = self->_lib_vis.function("gelu_erf_f16");
  self->_fn_hslice = self->_lib_elt.function("head_slice_f16");
  self->_fn_transpose = self->_lib_elt.function("transpose_abd_f16");
  self->_fn_residual = self->_lib_elt.function("residual_add_f16");
  self->_fn_sdpa = self->_lib_sdpa.function("sdpa_causal_window_f16");
  self->_fn_rope = self->_lib_rope.function("rope_interleaved_f16");
  if (!self->_fn_gemm.valid() || !self->_fn_ln.valid() ||
      !self->_fn_gelu.valid() || !self->_fn_hslice.valid() ||
      !self->_fn_transpose.valid() || !self->_fn_residual.valid() ||
      !self->_fn_sdpa.valid() || !self->_fn_rope.valid()) {
    return nullptr;
  }
  self->_ok = true;
  return self;
}

SharedBuffer
MetalMossCodec::rvq_decode_(const std::vector<std::vector<std::int32_t>>& codes,
                            int T) {
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };
  const int CD = _codebook_dim, RV = _rvq_dim, OD = _code_dim;

  // Host-gather each codebook's rows for the T frames.
  std::vector<SharedBuffer> gathered((std::size_t)_n_vq);
  for (int cb = 0; cb < _n_vq; ++cb) {
    gathered[(std::size_t)cb] = buf((std::size_t)T * CD);
    auto* g = static_cast<_Float16*>(gathered[(std::size_t)cb].contents());
    const auto* tbl =
        static_cast<const _Float16*>(_codebook[(std::size_t)cb].contents());
    for (int t = 0; t < T; ++t) {
      int code = codes[(std::size_t)t][(std::size_t)cb];
      if (code < 0) { code = 0; }
      if (code >= _codebook_size) { code = _codebook_size - 1; }
      for (int e = 0; e < CD; ++e) {
        g[(std::size_t)t * CD + e] = tbl[(std::size_t)code * CD + e];
      }
    }
  }
  SharedBuffer emb = buf((std::size_t)T * RV);
  std::memset(emb.contents(), 0, (std::size_t)T * RV * 2);
  SharedBuffer tmp = buf((std::size_t)T * RV);
  SharedBuffer hidden = buf((std::size_t)T * OD);

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    // scalar bias-GEMM (handles tiny K=8 + always applies the conv bias).
    auto lib_gemm_bias = _lib_gemm.function("dense_gemm_bias_f16");
    auto gemm_bias = [&](const SharedBuffer& xin, const SharedBuffer& w,
                         const SharedBuffer& b, const SharedBuffer& y, int M,
                         int N, int K) {
      enc.set_function(lib_gemm_bias);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, b);
      enc.set_buffer(3, y);
      enc.set_constant(4, M);
      enc.set_constant(5, N);
      enc.set_constant(6, K);
      enc.set_constant(7, 1);
      enc.dispatch({(unsigned)(((N + 15) / 16) * 16),
                    (unsigned)(((M + 15) / 16) * 16), 1}, {16, 16, 1});
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
    for (int cb = 0; cb < _n_vq; ++cb) {
      gemm_bias(gathered[(std::size_t)cb], _q_outw[(std::size_t)cb],
                _q_outb[(std::size_t)cb], tmp, T, RV, CD);
      residual(emb, tmp, emb, T * RV);
    }
    gemm_bias(emb, _rvq_outw, _rvq_outb, hidden, T, OD, RV);
  }
  stream.commit().wait();
  return hidden;
}

SharedBuffer
MetalMossCodec::run_stage_(const Stage& st, int T, const SharedBuffer& in) {
  const StageCfg& c = st.cfg;
  const int d = c.d_model, heads = c.n_heads, hd = d / heads, ff = c.ff;
  const int in_dim = c.in_dim, out_dim = c.out_dim;
  const float scale = 1.0f / std::sqrt((float)hd);
  const float eps = 1e-5f;
  auto buf = [&](std::size_t e) { return _mc->make_shared_buffer(e * 2); };

  SharedBuffer x = buf((std::size_t)T * d);
  SharedBuffer n1 = buf((std::size_t)T * d), qkv = buf((std::size_t)T * 3 * d);
  SharedBuffer q3 = buf((std::size_t)T * d), k3 = buf((std::size_t)T * d),
               v3 = buf((std::size_t)T * d);
  SharedBuffer qt = buf((std::size_t)T * d), kt = buf((std::size_t)T * d),
               vt = buf((std::size_t)T * d);
  SharedBuffer atb = buf((std::size_t)T * d), att = buf((std::size_t)T * d);
  SharedBuffer o = buf((std::size_t)T * d), o2 = buf((std::size_t)T * d);
  SharedBuffer h = buf((std::size_t)T * ff);
  SharedBuffer out =
      (out_dim != d) ? buf((std::size_t)T * out_dim) : SharedBuffer{};

  metal_compute::CommandStream stream = _mc->make_command_stream();
  {
    ComputeEncoder enc = stream.begin_compute();
    auto gemm = [&](const SharedBuffer& xin, const SharedBuffer& w,
                    const SharedBuffer& y, int M, int N, int K) {
      enc.set_function(_fn_gemm);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, w);
      enc.set_buffer(2, w);    // bias slot unused (has_bias=0)
      enc.set_buffer(3, y);
      enc.set_constant(4, K);
      enc.set_constant(5, N);
      enc.set_constant(6, M);
      enc.set_constant(7, 0);
      enc.dispatch({(unsigned)(((N + 31) / 32) * 32),
                    (unsigned)(((M + 31) / 32) * 2), 2}, {32, 2, 2});
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
    auto gelu = [&](const SharedBuffer& xin, const SharedBuffer& y, int nn) {
      enc.set_function(_fn_gelu);
      enc.set_buffer(0, xin);
      enc.set_buffer(1, y);
      enc.set_constant(2, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto hslice = [&](const SharedBuffer& inb, const SharedBuffer& outb, int Hh,
                      int Sd, int Wd, int off) {
      enc.set_function(_fn_hslice);
      enc.set_buffer(0, inb);
      enc.set_buffer(1, outb);
      enc.set_constant(2, Hh);
      enc.set_constant(3, Sd);
      enc.set_constant(4, Wd);
      enc.set_constant(5, off);
      const int zero = 0;
      enc.set_constant(6, zero);
      enc.set_constant(7, zero);
      enc.dispatch({(unsigned)(Hh * Wd), 1, 1}, {256, 1, 1});
    };
    auto transpose = [&](const SharedBuffer& inb, const SharedBuffer& outb,
                         int A, int Bd) {
      enc.set_function(_fn_transpose);
      enc.set_buffer(0, inb);
      enc.set_buffer(1, outb);
      enc.set_constant(2, A);
      enc.set_constant(3, Bd);
      enc.set_constant(4, hd);
      enc.dispatch({(unsigned)hd, (unsigned)Bd, (unsigned)A}, {(unsigned)hd, 1, 1});
    };
    auto residual = [&](const SharedBuffer& a, const SharedBuffer& b,
                        const SharedBuffer& outb, int nn) {
      enc.set_function(_fn_residual);
      enc.set_buffer(0, a);
      enc.set_buffer(1, b);
      enc.set_buffer(2, outb);
      enc.set_constant(3, nn);
      enc.dispatch({(unsigned)nn, 1, 1}, {256, 1, 1});
    };
    auto rope = [&](const SharedBuffer& xb) {
      enc.set_function(_fn_rope);
      enc.set_buffer(0, xb);
      enc.set_buffer(1, _inv_freq);
      enc.set_constant(2, heads);
      enc.set_constant(3, T);
      enc.set_constant(4, hd);
      const int off = 0;
      enc.set_constant(5, off);
      enc.dispatch({(unsigned)(hd / 2), (unsigned)T, (unsigned)heads}, {1, 1, 1});
    };
    auto attn = [&](const SharedBuffer& q, const SharedBuffer& k,
                    const SharedBuffer& v, const SharedBuffer& outb) {
      enc.set_function(_fn_sdpa);
      enc.set_buffer(0, q);
      enc.set_buffer(1, k);
      enc.set_buffer(2, v);
      enc.set_buffer(3, outb);
      enc.set_constant(4, scale);
      enc.set_constant(5, T);          // T_kv
      enc.set_constant(6, hd);         // D
      enc.set_constant(7, heads);      // Hq
      enc.set_constant(8, heads);      // Hkv
      enc.set_constant(9, T);          // n_q
      const int zero = 0;
      enc.set_constant(10, zero);      // q_offset
      enc.set_constant(11, T);         // kv_stride
      enc.set_constant(12, c.context); // window (>= T => plain causal)
      enc.set_constant(13, T);         // ring_cap (contiguous KV)
      enc.dispatch({32, (unsigned)heads, (unsigned)T}, {32, 1, 1});
    };

    gemm(in, st.in_proj, x, T, d, in_dim);   // input_proj
    for (int l = 0; l < c.n_layers; ++l) {
      const Layer& L = st.layers[(std::size_t)l];
      ln(x, L.n1w, L.n1b, n1, T, d);
      gemm(n1, L.qkvw, qkv, T, 3 * d, d);
      hslice(qkv, q3, T, 3 * d, d, 0);
      hslice(qkv, k3, T, 3 * d, d, d);
      hslice(qkv, v3, T, 3 * d, d, 2 * d);
      transpose(q3, qt, T, heads);     // [T,heads,hd] -> [heads,T,hd]
      transpose(k3, kt, T, heads);
      transpose(v3, vt, T, heads);
      rope(qt);
      rope(kt);
      attn(qt, kt, vt, atb);
      transpose(atb, att, heads, T);   // [heads,T,hd] -> [T,heads,hd]
      gemm(att, L.ow, o, T, d, d);
      residual(x, o, x, T * d);
      ln(x, L.n2w, L.n2b, n1, T, d);
      gemm(n1, L.fc1, h, T, ff, d);
      gelu(h, h, T * ff);
      gemm(h, L.fc2, o2, T, d, ff);
      residual(x, o2, x, T * d);
    }
    if (out_dim != d) { gemm(x, st.out_proj, out, T, out_dim, d); }
  }
  stream.commit().wait();
  if (out_dim != d) { return out; }
  return x;
}

std::vector<float>
MetalMossCodec::decode(const std::vector<std::vector<std::int32_t>>& codes,
                       std::vector<std::vector<float>>* stages) {
  std::vector<float> wave;
  const int T = (int)codes.size();
  if (T <= 0 || !_ok) { return wave; }

  // Channel-major [C][T] copy of a time-major [T,C] f16 buffer (for golden
  // comparison: the reference dumps [C, T]).
  auto chan_major = [](const SharedBuffer& b, int t, int cc) {
    std::vector<float> out((std::size_t)cc * t);
    const auto* s = static_cast<const _Float16*>(b.contents());
    for (int i = 0; i < t; ++i) {
      for (int c = 0; c < cc; ++c) {
        out[(std::size_t)c * t + i] = (float)s[(std::size_t)i * cc + c];
      }
    }
    return out;
  };

  SharedBuffer cur = rvq_decode_(codes, T);   // [T, 768]
  if (stages != nullptr) { stages->push_back(chan_major(cur, T, _code_dim)); }

  int curT = T;
  for (int si = 0; si < 4; ++si) {
    const Stage& st = _stages[(std::size_t)si];
    SharedBuffer out = run_stage_(st, curT, cur);   // [curT, out_dim]
    if (stages != nullptr) {
      stages->push_back(chan_major(out, curT, st.cfg.out_dim));
    }
    // Patch upsample (host): out[t][c*P+p] -> up[t*P+p][c].
    const int P = st.cfg.patch, Cout = st.cfg.out_dim / P, Tout = curT * P;
    SharedBuffer up = _mc->make_shared_buffer((std::size_t)Tout * Cout * 2);
    const auto* s = static_cast<const _Float16*>(out.contents());
    auto* d = static_cast<_Float16*>(up.contents());
    for (int t = 0; t < curT; ++t) {
      for (int c = 0; c < Cout; ++c) {
        for (int p = 0; p < P; ++p) {
          d[(std::size_t)(t * P + p) * Cout + c] =
              s[(std::size_t)t * st.cfg.out_dim + c * P + p];
        }
      }
    }
    if (stages != nullptr) { stages->push_back(chan_major(up, Tout, Cout)); }
    cur = std::move(up);
    curT = Tout;
  }

  // cur is [curT, 1] = [samples, 1].
  wave.resize((std::size_t)curT);
  const auto* w = static_cast<const _Float16*>(cur.contents());
  for (int i = 0; i < curT; ++i) { wave[(std::size_t)i] = (float)w[i]; }
  return wave;
}

}  // namespace vpipe::genai
